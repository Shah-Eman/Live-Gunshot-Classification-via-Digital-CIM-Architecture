#!/usr/bin/env python3
"""Step 1.7 — ADC sample-rate, bit-depth, and window-length sweep (Decision D6).

Pre-registered selection rule (written down BEFORE running the sweep, per the
Step 1.7 review gate in docs/runbook.md):
  1. Evaluate all 15 (sample_rate, adc_bits) points in
     {8000, 16000, 22050} Hz x {16, 12, 10, 8, 6} bits at window_s = 0.50 s,
     training the fixed 1D-CNN (`sw/models/cnn1d.py:build_cnn1d`) across all 5
     session-disjoint CV folds with 3 seeds (1701, 1702, 1703), scoring each
     seed on pooled out-of-fold predictions (Decision D23).
  2. Eligible points are all (sr, bits) with
       mean_macro_f1 >= max_grid_mean_macro_f1 - 0.01   (within 1.0 pp of best).
     Choose the cheapest eligible point ordered lexicographically by
       (sample_rate ASC, adc_bits ASC),
     because sample_rate scales inference MACs and buffer RAM linearly while
     adc_bits sets ADC and ring-buffer word width.
  3. At the chosen (sample_rate, adc_bits), compare window lengths
     {0.25, 0.40, 0.50} s (3 seeds x 5 CV folds; 0.50 s reused from step 1).
     Choose the shortest window_s within 0.01 (1.0 pp) of the best window's
     3-seed mean pooled CV macro-F1 (ordered by MACs / window_s ASC).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cim")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SW_DIR = HERE.parent
REPO_DIR = SW_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SW_DIR / "models"))

from cnn1d import build_cnn1d  # noqa: E402
from data import DatasetBundle, make_window_config  # noqa: E402
from harness import CompiledKerasTrainer  # noqa: E402
from metrics import aggregate_seed_metrics, classification_metrics  # noqa: E402
from models import compute_keras_macs_and_params  # noqa: E402

SELECTION_RULE = {
    "metric": "3-seed mean of pooled 5-fold session-disjoint CV macro_f1",
    "margin_below_best": 0.01,
    "grid_cost_order": ["sample_rate ASC", "adc_bits ASC"],
    "window_cost_order": ["total_macs ASC", "window_s ASC"],
    "rates_hz": [8000, 16000, 22050],
    "bits": [16, 12, 10, 8, 6],
    "windows_s": [0.25, 0.40, 0.50],
    "seeds": [1701, 1702, 1703],
}


def check_disk_space(paths: list[Path], min_gb: float = 1.0) -> None:
    for p in paths:
        target = p if p.exists() else p.parent
        free_gb = shutil.disk_usage(target).free / (1024**3)
        if free_gb < min_gb:
            raise RuntimeError(f"Disk space below {min_gb} GB at {target}: {free_gb:.2f} GB free. Stopping.")


def run_cv_point(
    bundle: DatasetBundle,
    trainer: CompiledKerasTrainer,
    fold_contexts: list[dict],
    adc_bits: int,
    seeds: list[int],
    n_epochs: int,
) -> dict:
    """Run 3 seeds x 5 CV folds for one (sample_rate, adc_bits, window_s) point."""
    val_sets = [bundle.build_val_set(ctx, adc_bits=adc_bits) for ctx in fold_contexts]
    seed_results = []

    for seed in seeds:
        oof_true, oof_pred, oof_src, oof_fold = [], [], [], []
        for k, ctx in enumerate(fold_contexts):
            X_va, y_va, src_va = val_sets[k]

            def epoch_fn(ep: int, _ctx=ctx, _seed=seed, _k=k):
                X_ep, y_ep = bundle.sample_train_epoch(
                    _ctx, ep, adc_bits=adc_bits, seed=_seed * 10 + _k, augment=False
                )
                return X_ep[..., None], y_ep

            model, _ = trainer.fit_custom_epochs(
                epoch_fn, n_epochs=n_epochs, n_classes=len(bundle.classes), seed=seed * 100 + k
            )
            preds = trainer.predict_classes(model, X_va[..., None])
            oof_true.append(y_va)
            oof_pred.append(preds)
            oof_src.extend(src_va)
            oof_fold.append(np.full(len(y_va), k, dtype=np.int64))

        met = classification_metrics(
            np.concatenate(oof_true),
            np.concatenate(oof_pred),
            bundle.classes,
            sources=oof_src,
            fold_ids=np.concatenate(oof_fold),
        )
        met["seed"] = int(seed)
        seed_results.append(met)

    summary = aggregate_seed_metrics(seed_results, bundle.classes)
    summary["seeds"] = seed_results
    return summary


def plot_step17(
    grid_results: dict[tuple[int, int], dict],
    win_results: dict[float, dict],
    chosen_sr: int,
    chosen_bits: int,
    chosen_win: float,
    out_png: Path,
) -> None:
    rates = SELECTION_RULE["rates_hz"]
    bits_list = SELECTION_RULE["bits"]
    mat = np.zeros((len(rates), len(bits_list)))
    std_mat = np.zeros_like(mat)
    for i, sr in enumerate(rates):
        for j, b in enumerate(bits_list):
            mat[i, j] = grid_results[(sr, b)]["macro_f1_mean"] * 100.0
            std_mat[i, j] = grid_results[(sr, b)]["macro_f1_std"] * 100.0

    best_f1 = mat.max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    im = ax1.imshow(mat, cmap="viridis", aspect="auto", vmin=max(0.0, best_f1 - 8.0), vmax=best_f1 + 0.5)
    ax1.set_xticks(range(len(bits_list)), [f"{b}-bit" for b in bits_list])
    ax1.set_yticks(range(len(rates)), [f"{sr/1000:g} kHz" for sr in rates])
    ax1.set_xlabel("ADC bit depth (MSB truncation before hardware gain)")
    ax1.set_ylabel("Sample rate")
    ax1.set_title("Step 1.7a: Pooled 5-fold CV Macro-F1 (%) — 3 seeds")

    for i, sr in enumerate(rates):
        for j, b in enumerate(bits_list):
            val, sd = mat[i, j], std_mat[i, j]
            eligible = val >= (best_f1 - 1.0 - 1e-9)
            star = " ★" if (sr == chosen_sr and b == chosen_bits) else (" ✓" if eligible else "")
            color = "black" if val > (mat.min() + mat.max()) / 2 else "white"
            ax1.text(
                j, i, f"{val:.2f}±{sd:.2f}{star}", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold" if star == " ★" else "normal"
            )
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Macro-F1 (%)")

    ws = sorted(win_results.keys())
    macs_m = [win_results[w]["cost"]["total_macs"] / 1e6 for w in ws]
    wf1 = [win_results[w]["macro_f1_mean"] * 100.0 for w in ws]
    wsd = [win_results[w]["macro_f1_std"] * 100.0 for w in ws]
    ax2.errorbar(macs_m, wf1, yerr=wsd, fmt="-o", capsize=5, lw=1.8, color="#1f77b4")
    w_best = max(wf1)
    ax2.axhline(w_best - 1.0, ls="--", color="gray", lw=1.0, label="Best − 1.0 pp threshold")
    for w, x_m, y_f in zip(ws, macs_m, wf1):
        tag = f"{w:.2f} s ★" if abs(w - chosen_win) < 1e-6 else f"{w:.2f} s"
        ax2.annotate(tag, (x_m, y_f), textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold" if "★" in tag else "normal")
    ax2.set_xlabel("1D-CNN MACs per inference (millions)")
    ax2.set_ylabel("Pooled 5-fold CV Macro-F1 (%)")
    ax2.set_title(f"Step 1.7b: Window sweep @ {chosen_sr/1000:g} kHz, {chosen_bits}-bit")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=REPO_DIR.parent / "Dataset-Final" / "manifest.csv")
    ap.add_argument("--splits", type=Path, default=SW_DIR / "data" / "splits.json")
    ap.add_argument("--window-index", type=Path, default=REPO_DIR / "results" / "step16" / "window_index.csv")
    ap.add_argument("--background-index", type=Path, default=REPO_DIR / "results" / "step16" / "background_index.csv")
    ap.add_argument("--gun-npz", type=Path, default=REPO_DIR.parent / "step16_outputs" / "gun_windows.npz")
    ap.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "step17")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space([args.out_dir, Path("/tmp")])

    # Record selection rule BEFORE looking at any results
    rule_path = args.out_dir / "selection_rule.json"
    rule_path.write_text(json.dumps(SELECTION_RULE, indent=2) + "\n")

    ckpt_path = args.out_dir / "checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text()) if (args.resume and ckpt_path.is_file()) else {"grid": {}, "windows": {}}

    rates = SELECTION_RULE["rates_hz"]
    bits_list = SELECTION_RULE["bits"]
    seeds = SELECTION_RULE["seeds"]

    # ------------------------------------------------------------------
    # Part A: (sample_rate, adc_bits) grid at window_s = 0.50 s
    # ------------------------------------------------------------------
    for sr in rates:
        missing_bits = [b for b in bits_list if f"{sr}_{b}" not in ckpt["grid"]]
        if not missing_bits:
            print(f"[Step 1.7] sr={sr} already in checkpoint, skipping decode.")
            continue

        check_disk_space([args.out_dir, Path("/tmp")])
        t_sr = time.time()
        print(f"\n[Step 1.7] Decoding CV pool at sample_rate={sr} Hz ...", flush=True)
        bundle = DatasetBundle(
            manifest_path=args.manifest,
            splits_path=args.splits,
            window_index_path=args.window_index,
            background_index_path=args.background_index,
            gun_npz_path=args.gun_npz if sr == 16000 else None,
            sample_rate=sr,
        )
        win_cfg = make_window_config(sr, 0.50)
        fold_contexts = [bundle.prepare_split(*bundle.splits.fold(k), window_s=0.50) for k in range(bundle.splits.n_folds)]
        avg_train_len = int(np.mean([len(c["train_gun_rows"]) * 3 // 2 for c in fold_contexts]))
        total_steps = args.epochs * ((avg_train_len + args.batch_size - 1) // args.batch_size)
        trainer = CompiledKerasTrainer(
            lambda _L=win_cfg.window_samples, _C=len(bundle.classes): build_cnn1d(_C, _L),
            total_steps=total_steps,
            batch_size=args.batch_size,
            lr_max=5e-3,
            lr_min=2e-4,
        )
        cost = compute_keras_macs_and_params(trainer.template, frontend_macs=0)
        print(f"  ready in {time.time()-t_sr:.1f}s | input_len={win_cfg.window_samples}, MACs={cost['total_macs']:,}", flush=True)

        for b in missing_bits:
            t0 = time.time()
            res = run_cv_point(bundle, trainer, fold_contexts, adc_bits=b, seeds=seeds, n_epochs=args.epochs)
            res["sample_rate"] = sr
            res["adc_bits"] = b
            res["window_s"] = 0.50
            res["cost"] = cost
            ckpt["grid"][f"{sr}_{b}"] = res
            ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
            print(
                f"  sr={sr:5d} Hz, bits={b:2d} | macro_f1={res['macro_f1_mean']*100:.2f}±{res['macro_f1_std']*100:.2f}% "
                f"| bal_acc={res['balanced_accuracy_mean']*100:.2f}% | .38_rec={res['revolver_38spl_recall_mean']*100:.1f}% "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )

        del bundle, trainer, fold_contexts
        gc.collect()

    grid_results = {(sr, b): ckpt["grid"][f"{sr}_{b}"] for sr in rates for b in bits_list}
    best_grid_f1 = max(v["macro_f1_mean"] for v in grid_results.values())
    eligible_grid = [
        (sr, b)
        for sr in sorted(rates)
        for b in sorted(bits_list)
        if grid_results[(sr, b)]["macro_f1_mean"] >= best_grid_f1 - SELECTION_RULE["margin_below_best"] - 1e-12
    ]
    chosen_sr, chosen_bits = eligible_grid[0]
    print(
        f"\n[Step 1.7a Decision] Best grid macro-F1 = {best_grid_f1*100:.2f}%. "
        f"Cheapest within 1.0 pp: sr={chosen_sr} Hz, bits={chosen_bits} "
        f"(macro-F1 = {grid_results[(chosen_sr, chosen_bits)]['macro_f1_mean']*100:.2f}%).",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Part B: Window sweep {0.25, 0.40, 0.50} s at (chosen_sr, chosen_bits)
    # ------------------------------------------------------------------
    ckpt["windows"][f"{chosen_sr}_{chosen_bits}_0.5"] = grid_results[(chosen_sr, chosen_bits)]
    extra_ws = [w for w in SELECTION_RULE["windows_s"] if f"{chosen_sr}_{chosen_bits}_{w:g}" not in ckpt["windows"]]
    if extra_ws:
        check_disk_space([args.out_dir, Path("/tmp")])
        print(f"\n[Step 1.7b] Window sweep {extra_ws} at ({chosen_sr} Hz, {chosen_bits}-bit) ...", flush=True)
        bundle = DatasetBundle(
            manifest_path=args.manifest,
            splits_path=args.splits,
            window_index_path=args.window_index,
            background_index_path=args.background_index,
            gun_npz_path=None,  # preload full clips so arbitrary window_s cuts cleanly
            sample_rate=chosen_sr,
            preload_gun_clips=True,
        )
        for w_s in extra_ws:
            t0 = time.time()
            win_cfg = make_window_config(chosen_sr, w_s)
            fold_contexts = [bundle.prepare_split(*bundle.splits.fold(k), window_s=w_s) for k in range(bundle.splits.n_folds)]
            avg_train_len = int(np.mean([len(c["train_gun_rows"]) * 3 // 2 for c in fold_contexts]))
            total_steps = args.epochs * ((avg_train_len + args.batch_size - 1) // args.batch_size)
            trainer = CompiledKerasTrainer(
                lambda _L=win_cfg.window_samples, _C=len(bundle.classes): build_cnn1d(_C, _L),
                total_steps=total_steps,
                batch_size=args.batch_size,
                lr_max=5e-3,
                lr_min=2e-4,
            )
            cost = compute_keras_macs_and_params(trainer.template, frontend_macs=0)
            res = run_cv_point(bundle, trainer, fold_contexts, adc_bits=chosen_bits, seeds=seeds, n_epochs=args.epochs)
            res["sample_rate"] = chosen_sr
            res["adc_bits"] = chosen_bits
            res["window_s"] = w_s
            res["cost"] = cost
            ckpt["windows"][f"{chosen_sr}_{chosen_bits}_{w_s:g}"] = res
            ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
            print(
                f"  window={w_s:.2f} s (MACs={cost['total_macs']:,}) | macro_f1={res['macro_f1_mean']*100:.2f}±{res['macro_f1_std']*100:.2f}% "
                f"| bal_acc={res['balanced_accuracy_mean']*100:.2f}% ({time.time()-t0:.1f}s)",
                flush=True,
            )
        del bundle
        gc.collect()

    win_results = {w: ckpt["windows"][f"{chosen_sr}_{chosen_bits}_{w:g}"] for w in SELECTION_RULE["windows_s"]}
    best_win_f1 = max(v["macro_f1_mean"] for v in win_results.values())
    eligible_wins = sorted(
        [
            w
            for w, v in win_results.items()
            if v["macro_f1_mean"] >= best_win_f1 - SELECTION_RULE["margin_below_best"] - 1e-12
        ],
        key=lambda w: (win_results[w]["cost"]["total_macs"], w),
    )
    chosen_win = eligible_wins[0]
    print(
        f"[Step 1.7b Decision] Best window macro-F1 = {best_win_f1*100:.2f}%. "
        f"Cheapest within 1.0 pp: window_s={chosen_win:.2f} s "
        f"(MACs={win_results[chosen_win]['cost']['total_macs']:,}, macro-F1={win_results[chosen_win]['macro_f1_mean']*100:.2f}%).",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Write CSVs, JSON summary, plot, and Markdown report
    # ------------------------------------------------------------------
    with open(args.out_dir / "adc_sweep_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_rate", "adc_bits", "window_s", "macs", "params", "macro_f1_mean", "macro_f1_std", "bal_acc_mean", "bal_acc_std", "rev38_recall_mean", "within_1pp_of_best", "chosen"])
        for sr in rates:
            for b in bits_list:
                r = grid_results[(sr, b)]
                elig = (sr, b) in eligible_grid
                w.writerow([sr, b, 0.50, r["cost"]["total_macs"], r["cost"]["params"], round(r["macro_f1_mean"], 5), round(r["macro_f1_std"], 5), round(r["balanced_accuracy_mean"], 5), round(r["balanced_accuracy_std"], 5), round(r["revolver_38spl_recall_mean"], 5), elig, (sr == chosen_sr and b == chosen_bits)])

    with open(args.out_dir / "window_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_rate", "adc_bits", "window_s", "input_samples", "macs", "params", "macro_f1_mean", "macro_f1_std", "bal_acc_mean", "bal_acc_std", "within_1pp_of_best", "chosen"])
        for ws in SELECTION_RULE["windows_s"]:
            r = win_results[ws]
            w.writerow([chosen_sr, chosen_bits, ws, int(round(ws * chosen_sr)), r["cost"]["total_macs"], r["cost"]["params"], round(r["macro_f1_mean"], 5), round(r["macro_f1_std"], 5), round(r["balanced_accuracy_mean"], 5), round(r["balanced_accuracy_std"], 5), ws in eligible_wins, abs(ws - chosen_win) < 1e-6])

    plot_step17(grid_results, win_results, chosen_sr, chosen_bits, chosen_win, args.out_dir / "adc_heatmap.png")

    chosen_summary = {
        "rule": SELECTION_RULE,
        "best_grid_macro_f1": best_grid_f1,
        "chosen_sample_rate": chosen_sr,
        "chosen_adc_bits": chosen_bits,
        "best_window_macro_f1": best_win_f1,
        "chosen_window_s": chosen_win,
        "chosen_input_samples": int(round(chosen_win * chosen_sr)),
        "chosen_macs": win_results[chosen_win]["cost"]["total_macs"],
        "chosen_params": win_results[chosen_win]["cost"]["params"],
        "chosen_macro_f1_mean": win_results[chosen_win]["macro_f1_mean"],
        "chosen_macro_f1_std": win_results[chosen_win]["macro_f1_std"],
        "chosen_balanced_accuracy_mean": win_results[chosen_win]["balanced_accuracy_mean"],
        "default_16k_16b_05s_macro_f1_mean": grid_results[(16000, 16)]["macro_f1_mean"],
    }
    (args.out_dir / "step17_summary.json").write_text(json.dumps(chosen_summary, indent=2) + "\n")

    md = [
        "# Step 1.7 — ADC Sample-Rate, Bit-Depth & Window Sweep (Decision D6)",
        "",
        "**Pre-registered selection rule:** Cheapest `(sample_rate ASC, adc_bits ASC)` within `1.0` percentage point (`0.01`) of the highest 3-seed mean pooled 5-fold session-disjoint CV macro-F1; then shortest `window_s` in `{0.25, 0.40, 0.50}` s within `1.0` pp of the best window at that `(rate, bits)`.",
        "",
        "## Part A: Sample rate × ADC bit depth (at 0.50 s window, 3 seeds × 5 folds)",
        "",
        "| Rate (Hz) | Bits | MACs | Macro-F1 (mean ± std) | Balanced Acc | .38 Recall | Within 1 pp | Chosen |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for sr in rates:
        for b in bits_list:
            r = grid_results[(sr, b)]
            md.append(
                f"| {sr} | {b} | {r['cost']['total_macs']:,} | {r['macro_f1_mean']*100:.2f} ± {r['macro_f1_std']*100:.2f} % "
                f"| {r['balanced_accuracy_mean']*100:.2f} % | {r['revolver_38spl_recall_mean']*100:.1f} % "
                f"| {'yes' if (sr, b) in eligible_grid else 'no'} | {'**D6**' if (sr == chosen_sr and b == chosen_bits) else ''} |"
            )
    md += [
        "",
        f"## Part B: Window length sweep at ({chosen_sr} Hz, {chosen_bits}-bit)",
        "",
        "| Window (s) | Samples | MACs | Macro-F1 (mean ± std) | Balanced Acc | Within 1 pp | Chosen |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for ws in SELECTION_RULE["windows_s"]:
        r = win_results[ws]
        md.append(
            f"| {ws:.2f} | {int(round(ws * chosen_sr))} | {r['cost']['total_macs']:,} "
            f"| {r['macro_f1_mean']*100:.2f} ± {r['macro_f1_std']*100:.2f} % "
            f"| {r['balanced_accuracy_mean']*100:.2f} % | {'yes' if ws in eligible_wins else 'no'} "
            f"| {'**D6**' if abs(ws - chosen_win) < 1e-6 else ''} |"
        )
    md += [
        "",
        f"## Chosen operating point (Decision D6): `{chosen_sr} Hz, {chosen_bits}-bit, {chosen_win:.2f} s`",
        f"- Pooled 5-fold CV Macro-F1: **{win_results[chosen_win]['macro_f1_mean']*100:.2f} ± {win_results[chosen_win]['macro_f1_std']*100:.2f} %**",
        f"- 1D-CNN MACs: **{win_results[chosen_win]['cost']['total_macs']:,}** ({win_results[chosen_win]['cost']['params']:,} parameters)",
        "- Final test set (`splits.test`) touched: **No (0 clips)**.",
    ]
    (args.out_dir / "step17_report.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote Step 1.7 outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
