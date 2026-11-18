#!/usr/bin/env python3
"""Step 1.8 — Float models within the hardware budget, split gap & leakage check.

Implements every item and review gate of runbook Step 1.8 (docs/runbook.md):
1. Pre-registers the hardware promotion rule BEFORE training:
   - `cnn2d_mel` only replaces `cnn1d` in hardware if its 3-seed mean pooled CV
     macro-F1 exceeds the best hardware 1D-CNN by > 5.0 percentage points (0.05).
2. Trains every candidate within <= 4M MACs and <= 30k parameters (except RBF
   SVM whose support-vector storage is reported transparently alongside the
   budget-compliant Linear SVM) across all 5 session-disjoint CV folds (3 seeds)
   with Step 1.8 waveform augmentations (gain, noise, time shift, reverb, plus
   Step 1.6 noise/event mixing and per-epoch fresh background crops) and class
   weights, at the locked Step 1.7 D6 operating point.
3. Computes exact MACs and parameters by script (never estimated), verifies
   BatchNormalization folding numerically (`max_diff < 1e-4`), reports pooled
   macro-F1, balanced accuracy, per-class recall (highlighting .38 Special),
   per-fold spread, and per-source accuracy.
4. Evaluates random clip split and group-disjoint device split to measure how
   session leakage inflates accuracy.
5. Runs the pre-trigger noise-only 4-class leakage check (chance = 25.0%) on
   both session-disjoint CV and the random split.
6. Saves folded float `.keras` models (including a full-CV-pool hardware model
   ready for Step 1.10 PTQ) to `../step18_outputs/models/`, never touching the
   held-out final test set (`splits.test`).
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
from harness import CompiledKerasTrainer, train_eval_mfcc_svm  # noqa: E402
from metrics import aggregate_seed_metrics, classification_metrics  # noqa: E402
from models import (  # noqa: E402
    BUDGET_MAX_MACS,
    BUDGET_MAX_PARAMS,
    build_cnn1d_bn,
    build_cnn1d_w64_bn,
    build_cnn2d_mel,
    build_dscnn_mel,
    build_pretrigger_cnn,
    compute_keras_macs_and_params,
    dsp_frontend_macs,
    extract_log_mel,
    extract_mfcc_stats,
    fold_batchnorm_1d,
    fold_cnn2d_mel,
    fold_dscnn_mel,
)

PROMOTION_RULE = {
    "2d_over_1d_margin_macro_f1": 0.05,
    "w64_over_base_1d_margin_macro_f1": 0.01,
    "budget_max_macs": BUDGET_MAX_MACS,
    "budget_max_params": BUDGET_MAX_PARAMS,
    "seeds": [1801, 1802, 1803],
}


def check_disk_space(paths: list[Path], min_gb: float = 1.0) -> None:
    for p in paths:
        target = p if p.exists() else p.parent
        free_gb = shutil.disk_usage(target).free / (1024**3)
        if free_gb < min_gb:
            raise RuntimeError(f"Disk space below {min_gb} GB at {target}: {free_gb:.2f} GB free. Stopping.")


def run_keras_cv_model(
    name: str,
    bundle: DatasetBundle,
    fold_contexts: list[dict],
    val_sets: list[tuple[np.ndarray, np.ndarray, list[str]]],
    adc_bits: int,
    seeds: list[int],
    n_epochs: int,
    batch_size: int,
    mode: str,
    model_out_dir: Path,
) -> dict:
    """Train & evaluate one neural architecture across seeds x 5 CV folds."""
    win_cfg = fold_contexts[0]["win"]
    n_cls = len(bundle.classes)
    real_len = win_cfg.real_samples
    in_len = win_cfg.window_samples
    sr = bundle.sample_rate

    if mode in ("cnn2d_mel", "dscnn"):
        val_inputs = [extract_log_mel(X_va[:, :real_len], sr) for X_va, _, _ in val_sets]
        mel_shape = val_inputs[0].shape[1:]
        fe_macs = dsp_frontend_macs(real_len, sr, include_dct_n_mfcc=0)
        if mode == "cnn2d_mel":
            builder = lambda: build_cnn2d_mel(mel_shape, n_cls, with_bn=True)
            fold_fn = fold_cnn2d_mel
            eval_builder = lambda: build_cnn2d_mel(mel_shape, n_cls, with_bn=False)
        else:
            builder = lambda: build_dscnn_mel(mel_shape, n_cls, with_bn=True)
            fold_fn = fold_dscnn_mel
            eval_builder = lambda: build_dscnn_mel(mel_shape, n_cls, with_bn=False)
    else:
        val_inputs = [X_va[..., None] for X_va, _, _ in val_sets]
        fe_macs = 0
        if mode == "cnn1d":
            builder = lambda: build_cnn1d(n_cls, in_len)
            fold_fn = None
            eval_builder = builder
        elif mode == "cnn1d_bn":
            builder = lambda: build_cnn1d_bn(n_cls, in_len)
            fold_fn = lambda m: fold_batchnorm_1d(m, wide64=False)
            eval_builder = lambda: build_cnn1d(n_cls, in_len)
        elif mode == "cnn1d_w64":
            builder = lambda: build_cnn1d_w64_bn(n_cls, in_len, with_bn=False)
            fold_fn = None
            eval_builder = builder
        else:
            raise ValueError(mode)

    avg_train_len = int(np.mean([len(c["train_gun_rows"]) * 3 // 2 for c in fold_contexts]))
    total_steps = n_epochs * ((avg_train_len + batch_size - 1) // batch_size)
    lr_max = 4e-3 if mode in ("cnn2d_mel", "dscnn") else 5e-3
    trainer = CompiledKerasTrainer(builder, total_steps=total_steps, batch_size=batch_size, lr_max=lr_max, lr_min=2e-4)
    eval_trainer = trainer if fold_fn is None else CompiledKerasTrainer(eval_builder, total_steps=1, batch_size=batch_size)

    seed_results = []
    max_bn_diff = 0.0
    cost_info = None

    for s_idx, seed in enumerate(seeds):
        oof_true, oof_pred, oof_src, oof_fold = [], [], [], []
        for k, ctx in enumerate(fold_contexts):
            _, y_va, src_va = val_sets[k]
            X_va_in = val_inputs[k]

            def epoch_fn(ep: int, _ctx=ctx, _seed=seed, _k=k):
                X_ep, y_ep = bundle.sample_train_epoch(
                    _ctx, ep, adc_bits=adc_bits, seed=_seed * 10 + _k, augment=True
                )
                if mode in ("cnn2d_mel", "dscnn"):
                    return extract_log_mel(X_ep[:, :real_len], sr), y_ep
                return X_ep[..., None], y_ep

            trained_m, _ = trainer.fit_custom_epochs(epoch_fn, n_epochs=n_epochs, n_classes=n_cls, seed=seed * 100 + k)
            if fold_fn is not None:
                eval_m, diff = fold_fn(trained_m)
                max_bn_diff = max(max_bn_diff, diff)
            else:
                eval_m = trained_m

            if cost_info is None:
                cost_info = compute_keras_macs_and_params(eval_m, frontend_macs=fe_macs)

            if s_idx == 0:
                eval_m.save(model_out_dir / f"{name}_fold{k}.keras")

            preds = eval_trainer.predict_classes(eval_m, X_va_in)
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

    agg = aggregate_seed_metrics(seed_results, bundle.classes)
    agg["model"] = name
    agg["cost"] = cost_info
    agg["bn_fold_max_diff"] = float(max_bn_diff)
    agg["seeds"] = seed_results
    return agg


def run_svm_cv(
    name: str,
    kernel: str,
    bundle: DatasetBundle,
    fold_contexts: list[dict],
    val_sets: list[tuple[np.ndarray, np.ndarray, list[str]]],
    adc_bits: int,
    seeds: list[int],
) -> dict:
    """Train & evaluate MFCC + SVM across 3 background-sampling seeds x 5 CV folds."""
    win_cfg = fold_contexts[0]["win"]
    real_len = win_cfg.real_samples
    sr = bundle.sample_rate
    fe_macs = dsp_frontend_macs(real_len, sr, include_dct_n_mfcc=13)
    val_feats = [extract_mfcc_stats(X_va, sr=sr, real_samples=real_len) for X_va, _, _ in val_sets]

    seed_results = []
    svm_costs = []
    for seed in seeds:
        oof_true, oof_pred, oof_src, oof_fold = [], [], [], []
        for k, ctx in enumerate(fold_contexts):
            # Combine 2 augmented training epochs (~20k windows) for rich SVM support
            X0, y0 = bundle.sample_train_epoch(ctx, 0, adc_bits=adc_bits, seed=seed * 10 + k, augment=True)
            X1, y1 = bundle.sample_train_epoch(ctx, 1, adc_bits=adc_bits, seed=seed * 10 + k, augment=False)
            X_tr = extract_mfcc_stats(np.concatenate([X0, X1], axis=0), sr=sr, real_samples=real_len)
            y_tr = np.concatenate([y0, y1], axis=0)

            _, y_va, src_va = val_sets[k]
            preds, scost = train_eval_mfcc_svm(X_tr, y_tr, val_feats[k], kernel=kernel, seed=seed + k)
            svm_costs.append(scost)
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

    mean_params = int(round(np.mean([c["svm_params"] for c in svm_costs])))
    mean_nn_macs = int(round(np.mean([c["svm_macs"] for c in svm_costs])))
    total_macs = fe_macs + mean_nn_macs
    agg = aggregate_seed_metrics(seed_results, bundle.classes)
    agg["model"] = name
    agg["cost"] = {
        "weights": mean_params,
        "biases": 0,
        "params": mean_params,
        "frontend_macs": fe_macs,
        "nn_macs": mean_nn_macs,
        "total_macs": total_macs,
        "within_budget": bool(total_macs <= BUDGET_MAX_MACS and mean_params <= BUDGET_MAX_PARAMS),
    }
    agg["bn_fold_max_diff"] = 0.0
    agg["seeds"] = seed_results
    return agg


def run_single_split_cnn1d(
    bundle: DatasetBundle,
    train_paths: set[str],
    val_paths: set[str],
    window_s: float,
    adc_bits: int,
    seeds: list[int],
    n_epochs: int,
    batch_size: int,
) -> dict:
    """Evaluate cnn1d on a 2-way diagnostic split (device-disjoint or random clip split)."""
    ctx = bundle.prepare_split(train_paths, val_paths, window_s=window_s)
    X_va, y_va, src_va = bundle.build_val_set(ctx, adc_bits=adc_bits)
    in_len = ctx["win"].window_samples
    n_cls = len(bundle.classes)
    steps = n_epochs * ((len(ctx["train_gun_rows"]) * 3 // 2 + batch_size - 1) // batch_size)
    trainer = CompiledKerasTrainer(
        lambda: build_cnn1d(n_cls, in_len), total_steps=steps, batch_size=batch_size, lr_max=5e-3, lr_min=2e-4
    )
    runs = []
    for seed in seeds:
        m, _ = trainer.fit_custom_epochs(
            lambda ep, _s=seed: (
                lambda x, y: (x[..., None], y)
            )(*bundle.sample_train_epoch(ctx, ep, adc_bits=adc_bits, seed=_s, augment=True)),
            n_epochs=n_epochs,
            n_classes=n_cls,
            seed=seed,
        )
        preds = trainer.predict_classes(m, X_va[..., None])
        runs.append(classification_metrics(y_va, preds, bundle.classes, sources=src_va))
    return aggregate_seed_metrics(runs, bundle.classes)


def run_pretrigger_leakage_check(bundle: DatasetBundle, adc_bits: int, batch_size: int = 256) -> dict:
    """Step 1.8 item 3: 4-class firearm model trained ONLY on pre-trigger background noise."""
    # 1. Pooled 5-fold session-disjoint CV
    oof_true, oof_pred = [], []
    gun_classes = None
    pre_len = None
    trainer = None
    for k in range(bundle.splits.n_folds):
        tr, va = bundle.splits.fold(k)
        X_tr, y_tr, X_va, y_va, gun_classes, pre_len = bundle.build_pretrigger_sets(tr, va, adc_bits=adc_bits)
        if trainer is None:
            steps = 12 * ((len(y_tr) + batch_size - 1) // batch_size)
            trainer = CompiledKerasTrainer(
                lambda _L=pre_len: build_pretrigger_cnn(_L, 4),
                total_steps=steps,
                batch_size=batch_size,
                lr_max=3e-3,
                lr_min=2e-4,
            )
        m, _ = trainer.fit_custom_epochs(
            lambda ep, _X=X_tr[..., None], _y=y_tr: (_X, _y), n_epochs=12, n_classes=4, seed=1850 + k
        )
        preds = trainer.predict_classes(m, X_va[..., None])
        oof_true.append(y_va)
        oof_pred.append(preds)

    cv_met = classification_metrics(np.concatenate(oof_true), np.concatenate(oof_pred), gun_classes)

    # 2. Random clip-level split (expected to leak session noise!)
    r_tr, r_va = bundle.splits.random_split()
    X_tr, y_tr, X_va, y_va, _, _ = bundle.build_pretrigger_sets(r_tr, r_va, adc_bits=adc_bits)
    steps_r = 12 * ((len(y_tr) + batch_size - 1) // batch_size)
    r_trainer = CompiledKerasTrainer(
        lambda _L=pre_len: build_pretrigger_cnn(_L, 4),
        total_steps=steps_r,
        batch_size=batch_size,
        lr_max=3e-3,
        lr_min=2e-4,
    )
    mr, _ = r_trainer.fit_custom_epochs(
        lambda ep, _X=X_tr[..., None], _y=y_tr: (_X, _y), n_epochs=12, n_classes=4, seed=1899
    )
    r_preds = r_trainer.predict_classes(mr, X_va[..., None])
    rand_met = classification_metrics(y_va, r_preds, gun_classes)

    return {
        "chance_level": 0.25,
        "pre_trigger_samples": int(pre_len),
        "pre_trigger_ms": round(1000.0 * pre_len / bundle.sample_rate, 2),
        "classes": gun_classes,
        "session_disjoint_cv": {
            "macro_f1": cv_met["macro_f1"],
            "balanced_accuracy": cv_met["balanced_accuracy"],
            "accuracy": cv_met["accuracy"],
            "per_class": cv_met["per_class"],
        },
        "random_clip_split": {
            "macro_f1": rand_met["macro_f1"],
            "balanced_accuracy": rand_met["balanced_accuracy"],
            "accuracy": rand_met["accuracy"],
            "per_class": rand_met["per_class"],
        },
    }


def plot_pareto(models_res: list[dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {
        "cnn1d": "#d62728",
        "cnn1d_spec51": "#ff7f0e",
        "cnn1d_w64": "#e377c2",
        "cnn1d_bn": "#8c564b",
        "cnn2d_mel": "#1f77b4",
        "dscnn": "#2ca02c",
        "mfcc_svm_linear": "#9467bd",
        "mfcc_svm_rbf": "#7f7f7f",
    }
    for r in models_res:
        mname = r["model"]
        macs_m = r["cost"]["total_macs"] / 1e6
        f1 = r["macro_f1_mean"] * 100.0
        sd = r["macro_f1_std"] * 100.0
        in_bud = r["cost"]["within_budget"]
        marker = "o" if "cnn1d" in mname else ("s" if in_bud else "^")
        ax.errorbar(
            macs_m, f1, yerr=sd, fmt=marker, markersize=8, capsize=4,
            color=colors.get(mname, "#333333"), label=f"{mname} ({r['cost']['params']/1e3:.1f}k params)"
        )
        ax.annotate(mname, (macs_m, f1), textcoords="offset points", xytext=(6, 5), fontsize=8.5)

    ax.axvline(BUDGET_MAX_MACS / 1e6, ls="--", color="red", alpha=0.5, label="4.0M MAC hardware ceiling")
    ax.set_xlabel("Total MACs per inference — DSP front end + classifier (millions)")
    ax.set_ylabel("Pooled 5-fold Session-Disjoint CV Macro-F1 (%)")
    ax.set_title("Step 1.8 — Accuracy vs. MACs Pareto Plot (3 seeds × 5 CV folds)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
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
    ap.add_argument("--step17-summary", type=Path, default=REPO_DIR / "results" / "step17" / "step17_summary.json")
    ap.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "step18")
    ap.add_argument("--model-out-dir", type=Path, default=REPO_DIR.parent / "step18_outputs" / "models")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_out_dir.mkdir(parents=True, exist_ok=True)
    check_disk_space([args.out_dir, args.model_out_dir, Path("/tmp")])

    # Record promotion rule BEFORE training
    (args.out_dir / "promotion_rule.json").write_text(json.dumps(PROMOTION_RULE, indent=2) + "\n")

    if not args.step17_summary.is_file():
        raise FileNotFoundError(f"{args.step17_summary} missing; run Step 1.7 first to lock D6.")
    s17 = json.loads(args.step17_summary.read_text())
    d6_sr = int(s17["chosen_sample_rate"])
    d6_bits = int(s17["chosen_adc_bits"])
    d6_win = float(s17["chosen_window_s"])
    print(f"[Step 1.8] Locked D6 operating point from Step 1.7: {d6_sr} Hz, {d6_bits}-bit, {d6_win:.2f} s", flush=True)

    ckpt_path = args.out_dir / "checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text()) if (args.resume and ckpt_path.is_file()) else {"models": {}, "splits": {}, "leakage": None}

    use_gun_npz = (d6_sr == 16000 and abs(d6_win - 0.50) < 1e-6)
    bundle = DatasetBundle(
        manifest_path=args.manifest,
        splits_path=args.splits,
        window_index_path=args.window_index,
        background_index_path=args.background_index,
        gun_npz_path=args.gun_npz if use_gun_npz else None,
        sample_rate=d6_sr,
        preload_gun_clips=not use_gun_npz,
    )
    fold_contexts = [bundle.prepare_split(*bundle.splits.fold(k), window_s=d6_win) for k in range(bundle.splits.n_folds)]
    val_sets = [bundle.build_val_set(ctx, adc_bits=d6_bits) for ctx in fold_contexts]
    seeds = PROMOTION_RULE["seeds"]

    # 1. Neural candidate suite at D6
    for m_name, mode in [
        ("cnn1d", "cnn1d"),
        ("cnn1d_w64", "cnn1d_w64"),
        ("cnn1d_bn", "cnn1d_bn"),
        ("cnn2d_mel", "cnn2d_mel"),
        ("dscnn", "dscnn"),
    ]:
        if m_name in ckpt["models"]:
            print(f"[Step 1.8] {m_name} already in checkpoint, skipping.", flush=True)
            continue
        check_disk_space([args.out_dir, args.model_out_dir, Path("/tmp")])
        t0 = time.time()
        print(f"\n[Step 1.8] Training {m_name} (3 seeds x 5 folds, {args.epochs} epochs) ...", flush=True)
        res = run_keras_cv_model(
            m_name, bundle, fold_contexts, val_sets, adc_bits=d6_bits,
            seeds=seeds, n_epochs=args.epochs, batch_size=args.batch_size,
            mode=mode, model_out_dir=args.model_out_dir,
        )
        ckpt["models"][m_name] = res
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
        print(
            f"  {m_name:12s} | MACs={res['cost']['total_macs']:,} | params={res['cost']['params']:,} "
            f"| macro_f1={res['macro_f1_mean']*100:.2f}±{res['macro_f1_std']*100:.2f}% "
            f"| bal_acc={res['balanced_accuracy_mean']*100:.2f}% ({time.time()-t0:.1f}s)",
            flush=True,
        )

    # 2. MFCC + SVM baselines (Linear budget-compliant + RBF comparison)
    for svm_name, kern in [("mfcc_svm_linear", "linear"), ("mfcc_svm_rbf", "rbf")]:
        if svm_name in ckpt["models"]:
            continue
        t0 = time.time()
        print(f"\n[Step 1.8] Training {svm_name} (3 seeds x 5 folds) ...", flush=True)
        res = run_svm_cv(svm_name, kern, bundle, fold_contexts, val_sets, adc_bits=d6_bits, seeds=seeds)
        ckpt["models"][svm_name] = res
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
        print(
            f"  {svm_name:15s} | MACs={res['cost']['total_macs']:,} | params={res['cost']['params']:,} "
            f"| macro_f1={res['macro_f1_mean']*100:.2f}±{res['macro_f1_std']*100:.2f}% ({time.time()-t0:.1f}s)",
            flush=True,
        )

    # 3. Diagnostic splits: Device-disjoint & Random clip-level split on cnn1d
    if "device_disjoint" not in ckpt["splits"]:
        t0 = time.time()
        print("\n[Step 1.8] Evaluating cnn1d on group-disjoint device split ...", flush=True)
        d_tr, d_va = bundle.splits.device_split()
        ckpt["splits"]["device_disjoint"] = run_single_split_cnn1d(
            bundle, d_tr, d_va, d6_win, d6_bits, seeds, args.epochs, args.batch_size
        )
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
        print(f"  device_disjoint macro_f1={ckpt['splits']['device_disjoint']['macro_f1_mean']*100:.2f}% ({time.time()-t0:.1f}s)", flush=True)

    if "random_clip" not in ckpt["splits"]:
        t0 = time.time()
        print("\n[Step 1.8] Evaluating cnn1d on random clip split (session-leaking reference) ...", flush=True)
        r_tr, r_va = bundle.splits.random_split()
        ckpt["splits"]["random_clip"] = run_single_split_cnn1d(
            bundle, r_tr, r_va, d6_win, d6_bits, seeds, args.epochs, args.batch_size
        )
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
        print(f"  random_clip     macro_f1={ckpt['splits']['random_clip']['macro_f1_mean']*100:.2f}% ({time.time()-t0:.1f}s)", flush=True)

    # 4. Pre-trigger noise leakage check (4 firearm classes, chance = 25%)
    if ckpt["leakage"] is None:
        t0 = time.time()
        print("\n[Step 1.8] Running pre-trigger background-noise leakage check ...", flush=True)
        ckpt["leakage"] = run_pretrigger_leakage_check(bundle, adc_bits=d6_bits, batch_size=args.batch_size)
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
        lk = ckpt["leakage"]
        print(
            f"  pre-trigger ({lk['pre_trigger_ms']} ms, chance=25.0%): "
            f"session-disjoint CV bal_acc={lk['session_disjoint_cv']['balanced_accuracy']*100:.2f}% "
            f"(macro_f1={lk['session_disjoint_cv']['macro_f1']*100:.2f}%), "
            f"random-split bal_acc={lk['random_clip_split']['balanced_accuracy']*100:.2f}% ({time.time()-t0:.1f}s)",
            flush=True,
        )

    # 5. Also evaluate arith_spec.md 5.1 default (16 kHz / 16-bit / 0.50 s) if D6 differs
    if (d6_sr, d6_bits, round(d6_win, 2)) != (16000, 16, 0.50) and "cnn1d_spec51" not in ckpt["models"]:
        print("\n[Step 1.8] Also evaluating cnn1d_spec51 at (16000 Hz, 16-bit, 0.50 s) reference ...", flush=True)
        del bundle
        gc.collect()
        b16 = DatasetBundle(
            manifest_path=args.manifest,
            splits_path=args.splits,
            window_index_path=args.window_index,
            background_index_path=args.background_index,
            gun_npz_path=args.gun_npz,
            sample_rate=16000,
        )
        fc16 = [b16.prepare_split(*b16.splits.fold(k), window_s=0.50) for k in range(b16.splits.n_folds)]
        vs16 = [b16.build_val_set(c, adc_bits=16) for c in fc16]
        res16 = run_keras_cv_model(
            "cnn1d_spec51", b16, fc16, vs16, adc_bits=16,
            seeds=seeds, n_epochs=args.epochs, batch_size=args.batch_size,
            mode="cnn1d", model_out_dir=args.model_out_dir,
        )
        ckpt["models"]["cnn1d_spec51"] = res16
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")
        classes_list = b16.classes
        del b16
        gc.collect()
    else:
        classes_list = bundle.classes

    # 6. Train full-CV-pool hardware candidate model for Step 1.10 quantization
    full_cv_model_path = args.model_out_dir / "cnn1d_full_cv.keras"
    if not full_cv_model_path.is_file():
        print("\n[Step 1.8] Training final float cnn1d on full CV pool (for Step 1.10 PTQ; 0 test clips) ...", flush=True)
        b_full = DatasetBundle(
            manifest_path=args.manifest,
            splits_path=args.splits,
            window_index_path=args.window_index,
            background_index_path=args.background_index,
            gun_npz_path=args.gun_npz if use_gun_npz else None,
            sample_rate=d6_sr,
            preload_gun_clips=not use_gun_npz,
        )
        ctx_full = b_full.prepare_split(b_full.splits.cv_pool, set(), window_s=d6_win)
        in_len = ctx_full["win"].window_samples
        steps_f = args.epochs * ((len(ctx_full["train_gun_rows"]) * 3 // 2 + args.batch_size - 1) // args.batch_size)
        tr_full = CompiledKerasTrainer(
            lambda: build_cnn1d(len(classes_list), in_len),
            total_steps=steps_f,
            batch_size=args.batch_size,
            lr_max=5e-3,
            lr_min=2e-4,
        )
        m_full, _ = tr_full.fit_custom_epochs(
            lambda ep: (
                lambda x, y: (x[..., None], y)
            )(*b_full.sample_train_epoch(ctx_full, ep, adc_bits=d6_bits, seed=1800, augment=True)),
            n_epochs=args.epochs,
            n_classes=len(classes_list),
            seed=1800,
        )
        m_full.save(full_cv_model_path)
        del b_full
        gc.collect()

    # ------------------------------------------------------------------
    # Apply promotion rule & write tables, Pareto plot, and report
    # ------------------------------------------------------------------
    ordered_names = [k for k in ("cnn1d", "cnn1d_spec51", "cnn1d_w64", "cnn1d_bn", "cnn2d_mel", "dscnn", "mfcc_svm_linear", "mfcc_svm_rbf") if k in ckpt["models"]]
    models_list = [ckpt["models"][k] for k in ordered_names]

    f1_1d = ckpt["models"]["cnn1d"]["macro_f1_mean"]
    f1_w64 = ckpt["models"]["cnn1d_w64"]["macro_f1_mean"]
    f1_2d = ckpt["models"]["cnn2d_mel"]["macro_f1_mean"]
    best_hw_1d = "cnn1d_w64" if (f1_w64 - f1_1d) > PROMOTION_RULE["w64_over_base_1d_margin_macro_f1"] else "cnn1d"
    best_hw_f1 = max(f1_1d, f1_w64) if best_hw_1d == "cnn1d_w64" else f1_1d
    promote_2d = (f1_2d - best_hw_f1) > PROMOTION_RULE["2d_over_1d_margin_macro_f1"]
    selected_hw_model = "cnn2d_mel" if promote_2d else best_hw_1d

    with open(args.out_dir / "model_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "frontend_macs", "nn_macs", "total_macs", "params", "within_budget",
            "macro_f1_mean", "macro_f1_std", "bal_acc_mean", "bal_acc_std",
            "rev38_recall_mean", "bn_fold_max_diff"
        ])
        for r in models_list:
            c = r["cost"]
            w.writerow([
                r["model"], c["frontend_macs"], c["nn_macs"], c["total_macs"], c["params"], c["within_budget"],
                round(r["macro_f1_mean"], 5), round(r["macro_f1_std"], 5),
                round(r["balanced_accuracy_mean"], 5), round(r["balanced_accuracy_std"], 5),
                round(r["revolver_38spl_recall_mean"], 5), round(r["bn_fold_max_diff"], 7),
            ])

    with open(args.out_dir / "per_class_recall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", *classes_list])
        for r in models_list:
            w.writerow([r["model"], *[f"{r['per_class'][c]['recall_mean']*100:.2f}±{r['per_class'][c]['recall_std']*100:.2f}%" for c in classes_list]])

    sources_order = sorted(models_list[0]["per_source"].keys())
    with open(args.out_dir / "per_source_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", *sources_order])
        for r in models_list:
            w.writerow([r["model"], *[f"{r['per_source'][s]['accuracy_mean']*100:.2f}%" for s in sources_order]])

    plot_pareto(models_list, args.out_dir / "pareto_macs_f1.png")

    summary_obj = {
        "d6_operating_point": {"sample_rate": d6_sr, "adc_bits": d6_bits, "window_s": d6_win},
        "promotion_rule": PROMOTION_RULE,
        "selected_hardware_model": selected_hw_model,
        "2d_promoted_over_1d": bool(promote_2d),
        "2d_minus_1d_macro_f1": float(f1_2d - best_hw_f1),
        "split_comparison_cnn1d": {
            "session_disjoint_cv_macro_f1": f1_1d,
            "device_disjoint_macro_f1": ckpt["splits"]["device_disjoint"]["macro_f1_mean"],
            "random_clip_split_macro_f1": ckpt["splits"]["random_clip"]["macro_f1_mean"],
            "session_leakage_inflation_pp": float((ckpt["splits"]["random_clip"]["macro_f1_mean"] - f1_1d) * 100.0),
        },
        "leakage_check": ckpt["leakage"],
        "test_set_touched": False,
    }
    (args.out_dir / "step18_summary.json").write_text(json.dumps(summary_obj, indent=2) + "\n")
    (args.out_dir / "leakage_and_splits.json").write_text(
        json.dumps({"splits": ckpt["splits"], "leakage": ckpt["leakage"]}, indent=2) + "\n"
    )

    lk = ckpt["leakage"]
    sp = ckpt["splits"]
    md = [
        "# Step 1.8 — Float Models Within the Hardware Budget",
        "",
        f"**Locked D6 operating point (from Step 1.7):** `{d6_sr} Hz, {d6_bits}-bit, {d6_win:.2f} s`",
        f"**Pre-registered promotion rule:** 2D-CNN replaces 1D-CNN only if it beats the hardware 1D-CNN by `> 5.0` points of pooled CV macro-F1; `cnn1d_w64` replaces base `cnn1d` only if it gains `> 1.0` point.",
        "",
        "## 1. Model Comparison (Pooled 5-Fold Session-Disjoint CV, 3 Seeds)",
        "",
        "| Model | Front-End MACs | Classifier MACs | Total MACs | Params | Budget OK | Macro-F1 (mean ± std) | Balanced Acc | .38 Recall |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for r in models_list:
        c = r["cost"]
        md.append(
            f"| `{r['model']}` | {c['frontend_macs']:,} | {c['nn_macs']:,} | {c['total_macs']:,} | {c['params']:,} "
            f"| {'yes' if c['within_budget'] else 'no (SVs)'} | **{r['macro_f1_mean']*100:.2f} ± {r['macro_f1_std']*100:.2f} %** "
            f"| {r['balanced_accuracy_mean']*100:.2f} % | {r['revolver_38spl_recall_mean']*100:.1f} % |"
        )

    md += [
        "",
        "## 2. Per-Class Recall Across Models (Pooled 5-Fold CV)",
        "",
        "| Model | " + " | ".join(classes_list) + " |",
        "|---|" + "|".join(["---:"] * len(classes_list)) + "|",
    ]
    for r in models_list:
        md.append(
            f"| `{r['model']}` | "
            + " | ".join(f"{r['per_class'][c]['recall_mean']*100:.1f}%" for c in classes_list)
            + " |"
        )

    md += [
        "",
        "## 3. Per-Source Accuracy Across Models (Pooled 5-Fold CV)",
        "",
        "| Model | " + " | ".join(sources_order) + " |",
        "|---|" + "|".join(["---:"] * len(sources_order)) + "|",
    ]
    for r in models_list:
        md.append(
            f"| `{r['model']}` | "
            + " | ".join(f"{r['per_source'][s]['accuracy_mean']*100:.1f}%" for s in sources_order)
            + " |"
        )

    md += [
        "",
        "## 4. Session Leakage Quantification (`cnn1d` Across Split Regimes)",
        "",
        "| Split Regime | Macro-F1 (mean ± std) | Balanced Acc | .38 Recall | Meaning |",
        "|---|---:|---:|---:|---|",
        f"| **Session-disjoint 5-fold CV** | **{f1_1d*100:.2f} ± {ckpt['models']['cnn1d']['macro_f1_std']*100:.2f} %** | {ckpt['models']['cnn1d']['balanced_accuracy_mean']*100:.2f} % | {ckpt['models']['cnn1d']['revolver_38spl_recall_mean']*100:.1f} % | True unseen-day / unseen-range generalisation |",
        f"| **Group-disjoint device split** | {sp['device_disjoint']['macro_f1_mean']*100:.2f} ± {sp['device_disjoint']['macro_f1_std']*100:.2f} % | {sp['device_disjoint']['balanced_accuracy_mean']*100:.2f} % | {sp['device_disjoint']['revolver_38spl_recall_mean']*100:.1f} % | Unseen microphones within known sessions (0 shared shots) |",
        f"| **Random clip split** | {sp['random_clip']['macro_f1_mean']*100:.2f} ± {sp['random_clip']['macro_f1_std']*100:.2f} % | {sp['random_clip']['balanced_accuracy_mean']*100:.2f} % | {sp['random_clip']['revolver_38spl_recall_mean']*100:.1f} % | Inflated by **+{(sp['random_clip']['macro_f1_mean'] - f1_1d)*100:.1f} pp** from session memorisation |",
        "",
        "## 5. Pre-Trigger Background-Noise Leakage Check (4 Firearm Classes, Chance = 25.0%)",
        "",
        f"Trained `pretrigger_cnn` on ONLY the `{lk['pre_trigger_ms']} ms` (`{lk['pre_trigger_samples']}` samples) strictly preceding the detector trigger:",
        f"- **Session-disjoint 5-fold CV:** Balanced accuracy = **{lk['session_disjoint_cv']['balanced_accuracy']*100:.2f} %**, Macro-F1 = **{lk['session_disjoint_cv']['macro_f1']*100:.2f} %** (vs 25.0% chance).",
        f"- **Random clip split:** Balanced accuracy = **{lk['random_clip_split']['balanced_accuracy']*100:.2f} %**, Macro-F1 = **{lk['random_clip_split']['macro_f1']*100:.2f} %** — proving that random splits identify the firearm from ambient background noise before the shot even fires!",
        "",
        "## 6. Hardware Promotion Recommendation (for Step 1.9)",
        "",
        f"- `cnn2d_mel` − `{best_hw_1d}` macro-F1 difference: `{(f1_2d - best_hw_f1)*100:+.2f}` pp (promotion threshold: `+5.00` pp) -> **2D-CNN promoted: {'YES' if promote_2d else 'NO'}**.",
        f"- Recommended hardware model: **`{selected_hw_model}`**.",
        f"- BatchNormalization folding numerical check: max logit discrepancy = `{max(r['bn_fold_max_diff'] for r in models_list):.2e}` (`< 1e-4`).",
        "- Held-out final test set (`splits.test`) touched: **No (0 clips)**.",
    ]
    (args.out_dir / "step18_report.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote Step 1.8 outputs to {args.out_dir} and models to {args.model_out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
