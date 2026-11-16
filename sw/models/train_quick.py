#!/usr/bin/env python3
"""Quick model on quick windows (runbook Step 1.1, items 3-5).

Realistic weights early, so the integer lane (Steps 1.3-1.5) can work on real
value ranges while the full data pipeline (Steps 1.6-1.9) is built. This is NOT
the final model and its accuracy must never be quoted: the split is a plain
random clip split, which leaks session information.

Windows (item 3), 0.5 s = 8,000 samples, starting 50 ms before the shot:
  * edge:  the dataset's labelled shot times (`gunshot-audio-labels-only.csv`),
           up to `--shots-per-clip` shots per clip;
  * C3GD:  the fixed 0.1 s pre-roll, so the window starts at 0.05 s;
  * Cadre: the first sample within 30 dB of the clip's peak;
  * background: one random 0.5 s crop per clip, from a random subset of
           `--max-background` clips.
Audio goes through `manifest.load_audio` (16 kHz mono) and is clamped to
[-1, 1]. A window that runs past either end of its clip is zero-filled there:
acceptable for a sanity-check model, and exactly the shortcut Step 1.6 removes.

Training (item 4): Adam, class weights, a seeded random 80/20 split, a few
epochs on a CPU. Output: `model_quick.keras` and `quick_train_log.json`
(window counts, loss per epoch, the sanity checks; no accuracy figures).

    python sw/models/train_quick.py \\
        --manifest ../Dataset-Final/manifest.csv \\
        --edge-labels ../Dataset/edge-collected-gunshot-audio/gunshot-audio-labels-only.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))

from cnn1d import INPUT_LEN, build_cnn1d  # noqa: E402
from manifest import TARGET_SR, classes, load_audio, load_manifest  # noqa: E402

PRE_SHOT_S = 0.05        # window starts this long before the shot
C3GD_PREROLL_S = 0.1     # C3GD clips start 0.1 s before the shot (dataset_audit.md)
CADRE_ONSET_DB = 30.0    # Cadre onset: first sample within 30 dB of the peak
SEED = 20261116


# --------------------------------------------------------------------------
# Window cutting
# --------------------------------------------------------------------------
def read_edge_labels(path: str | Path) -> dict[str, list[float]]:
    """Clip stem -> labelled shot times in seconds, from the Kabealo labels file."""
    labels = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                                     # header
        for row in reader:
            if row and row[0].strip():
                times = [float(v) for v in row[2:] if v.strip()]
                labels[row[0].strip()] = times
    return labels


def edge_stem(row: dict) -> str:
    """'edge:glock_17_9mm_caliber/<stem>.wav' -> '<stem>'."""
    return Path(row["orig_path"].split(":", 1)[1]).stem


def cut(x: np.ndarray, start: int) -> np.ndarray:
    """INPUT_LEN samples from `start`, zero-filled where the clip has none."""
    out = np.zeros(INPUT_LEN, np.float32)
    lo, hi = max(start, 0), min(start + INPUT_LEN, len(x))
    if hi > lo:
        out[lo - start:hi - start] = x[lo:hi]
    return out


def shot_starts(row: dict, x: np.ndarray, edge_times: list[float] | None,
                shots_per_clip: int) -> list[int]:
    """Window start samples for a gun clip, by the simple per-source rule."""
    pre = int(round(PRE_SHOT_S * TARGET_SR))
    src = row["source"]
    if src == "edge":
        return [int(round(t * TARGET_SR)) - pre for t in (edge_times or [])[:shots_per_clip]]
    if src == "c3gd":
        return [int(round(C3GD_PREROLL_S * TARGET_SR)) - pre]
    if src == "cadre":
        peak = float(np.abs(x).max())
        if peak == 0.0:
            return []
        onset = int(np.argmax(np.abs(x) >= peak * 10 ** (-CADRE_ONSET_DB / 20)))
        return [onset - pre]
    raise ValueError(f"no window rule for gun source {src!r}")


def windows_for(job: tuple) -> tuple[list[np.ndarray], str | None]:
    """Worker: load one clip and cut its windows. Returns (windows, skip reason)."""
    row, edge_times, shots_per_clip, crop_seed = job
    x = np.clip(load_audio(row), -1.0, 1.0)
    if row["class"] == "background":
        g = np.random.default_rng(crop_seed)
        start = int(g.integers(0, max(len(x) - INPUT_LEN, 0) + 1))
        return [cut(x, start)], None
    if row["source"] == "edge" and not edge_times:
        return [], "edge clip without labelled shots"
    starts = shot_starts(row, x, edge_times, shots_per_clip)
    if not starts:
        return [], "silent clip"
    return [cut(x, s) for s in starts], None


def build_windows(rows: list[dict], edge_labels: dict, *, max_background: int,
                  shots_per_clip: int, workers: int, seed: int):
    names = classes(rows)
    g = np.random.default_rng(seed)
    guns = [r for r in rows if r["class"] != "background"]
    bg = [r for r in rows if r["class"] == "background"]
    if len(bg) > max_background:
        bg = [bg[i] for i in sorted(g.choice(len(bg), max_background, replace=False))]
    chosen = guns + bg
    jobs = [(r, edge_labels.get(edge_stem(r)) if r["source"] == "edge" else None,
             shots_per_clip, seed + i) for i, r in enumerate(chosen)]

    X, y, skipped = [], [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r, (wins, why) in zip(chosen, pool.map(windows_for, jobs, chunksize=32)):
            if why:
                skipped[why] = skipped.get(why, 0) + 1
            X.extend(wins)
            y.extend([names.index(r["class"])] * len(wins))
    return np.stack(X)[..., None], np.asarray(y, np.int64), names, skipped


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def random_split(y: np.ndarray, frac_val: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Plain random 80/20 split, stratified by class only (it leaks sessions; Step 1.1 only)."""
    g = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = g.permutation(np.flatnonzero(y == c))
        n_val = int(round(len(idx) * frac_val))
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    return g.permutation(tr), g.permutation(va)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = [np.mean(y_pred[y_true == c] == c) for c in range(n_classes) if np.any(y_true == c)]
    return float(np.mean(recalls))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--edge-labels", required=True,
                    help="Kabealo gunshot-audio-labels-only.csv (shot times of the edge clips)")
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-background", type=int, default=6000)
    ap.add_argument("--shots-per-clip", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import keras                                         # after argparse: --help stays fast
    keras.utils.set_random_seed(args.seed)

    rows = load_manifest(args.manifest)
    X, y, names, skipped = build_windows(
        rows, read_edge_labels(args.edge_labels), max_background=args.max_background,
        shots_per_clip=args.shots_per_clip, workers=args.workers, seed=args.seed)
    counts = {n: int(np.sum(y == i)) for i, n in enumerate(names)}
    print(f"windows: {len(y):,} {counts}; skipped clips: {skipped}")

    tr, va = random_split(y, 0.2, args.seed)
    weights = {i: len(tr) / (len(names) * max(int(np.sum(y[tr] == i)), 1))
               for i in range(len(names))}

    model = build_cnn1d(num_classes=len(names))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=args.epochs,
                     batch_size=args.batch, class_weight=weights, verbose=2)

    # Sanity checks only (item 5). The accuracy itself is not printed or saved.
    pred = np.argmax(model.predict(X[va], batch_size=256, verbose=0), axis=1)
    chance = 1.0 / len(names)
    loss, val_loss = hist.history["loss"], hist.history["val_loss"]
    checks = {
        "training_loss_fell": bool(loss[-1] < loss[0]),
        "validation_loss_fell": bool(val_loss[-1] < val_loss[0]),
        "beats_chance_balanced_accuracy": bool(balanced_accuracy(y[va], pred, len(names)) > chance),
    }

    out = Path(args.out_dir)
    model.save(out / "model_quick.keras")
    log = {
        "note": "Step 1.1 sanity-check model. Random clip split leaks sessions: never quote "
                "its accuracy. The final model comes from Steps 1.6-1.9.",
        "classes": names, "windows_per_class": counts, "skipped_clips": skipped,
        "train_windows": int(len(tr)), "val_windows": int(len(va)),
        "class_weights": {names[i]: round(w, 4) for i, w in weights.items()},
        "epochs": args.epochs, "batch": args.batch, "seed": args.seed,
        "max_background": args.max_background, "shots_per_clip": args.shots_per_clip,
        "loss": [round(v, 4) for v in loss], "val_loss": [round(v, 4) for v in val_loss],
        "checks": checks,
        "keras": keras.__version__, "backend": keras.backend.backend(),
        "python": platform.python_version(),
    }
    (out / "quick_train_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    print(f"saved {out / 'model_quick.keras'} and {out / 'quick_train_log.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
