#!/usr/bin/env python3
"""Review-gate checks for the windows (Step 1.6). Exit code 0 only if every check passes.

  1. Every gun window was cut by detector.py: gun_windows.npz is rebuilt from scratch for a
     random sample of rows and must match bit for bit (window, trigger and start).
  2. No window was padded beyond the agreed zeroed tail: every window starts inside its clip
     (start >= 0, start + REAL <= clip length) and its last TAIL samples are zero. Real parts
     that end in more than 4 ms of zeros (silence inside the recording) are listed as a note.
  3. No feature of the window differs by source: one table, one row per source, with
     length, real/zero layout, sample rate, gain shift and 95% spectral roll-off. The layout
     columns must be identical; read the roll-off column for bandwidth (a source far below
     the others would need the low-pass of runbook item 5).
  4. At least 15% of background windows per epoch come from the gun microphones: one epoch
     is drawn with BackgroundSampler on the development rows (or fold --fold's train rows).
  5. Writes 10 windows per class (and one per source) as WAV to <out>/listen/ for the reviewer,
     who confirms the shot is inside each one.

Usage:
    python sw/frontend/check_windows.py --dataset-final PATH [--windows DIR] [--fold K] [--epoch N]
"""
from __future__ import annotations

import argparse
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
import windows as W                         # noqa: E402
from manifest import load_manifest          # noqa: E402

MAX_ZERO_RUN = int(0.004 * W.SR)


def write_wav(path: Path, x: np.ndarray):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(W.SR)
        f.writeframes(np.asarray(x, dtype="<i2").tobytes())


def trailing_zero_run(x: np.ndarray) -> int:
    nz = np.flatnonzero(x)
    return len(x) if nz.size == 0 else len(x) - 1 - int(nz[-1])


def rolloff(x: np.ndarray, frac: float = 0.95) -> float:
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * np.hanning(len(x)))) ** 2
    c = np.cumsum(spec)
    return float(np.searchsorted(c, frac * c[-1]) * W.SR / (2 * (len(spec) - 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-final", type=Path, required=True)
    ap.add_argument("--windows", type=Path, help="default: <dataset-final>/windows")
    ap.add_argument("--edge-meta", type=Path, help="same file given to cut_windows.py")
    ap.add_argument("--fold", type=int, help="draw the background epoch from this fold's train rows")
    ap.add_argument("--epoch", type=int, default=2000, help="background windows to draw")
    ap.add_argument("--rebuild", type=int, default=50, help="gun rows to re-cut for check 1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = a.dataset_final.resolve()
    wdir = a.windows or root / "windows"
    rows = load_manifest(root / "manifest.csv")
    z = np.load(wdir / "gun_windows.npz")
    X, row, trig, start, clen, gain = (z[k] for k in ("windows", "row", "trigger", "start", "clip_len", "gain_shift"))
    rng = np.random.default_rng(a.seed)
    fails = []

    def check(ok, msg):
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    layout = tuple(int(v) for v in z["layout"])
    check(layout == (W.INPUT, W.PRE, W.REAL, W.TAIL), f"0. npz layout {layout} == windows.py "
          f"{(W.INPUT, W.PRE, W.REAL, W.TAIL)} (INPUT, PRE, REAL, TAIL)")

    # 1. rebuild a sample with detector.py
    sys.path.insert(0, str(HERE))
    from cut_windows import load_edge_shots
    edge = load_edge_shots(a.edge_meta)
    gun_rows = sorted(set(row.tolist()))
    pick = rng.choice(gun_rows, size=min(a.rebuild, len(gun_rows)), replace=False) if gun_rows else []
    bad = 0
    for i in pick:
        c = W.cut_gun(rows[int(i)], edge)
        mine = np.flatnonzero(row == i)
        got = [(c.triggers[k], c.starts[k]) for k in c.kept]
        exp = [(int(trig[j]), int(start[j])) for j in mine]
        if got != exp or not all(np.array_equal(c.windows[n], X[j]) for n, j in enumerate(mine)):
            bad += 1
    check(bad == 0 and len(X) > 0, f"1. {len(pick)} re-cut clips match gun_windows.npz bit for bit ({bad} differ); "
          f"{len(X)} gun windows in total")

    # 2. padding
    inside = (start >= 0) & (start + W.REAL <= clen)
    tail0 = ~np.any(X[:, W.REAL:], axis=1) if len(X) else np.array([], bool)
    runs = np.array([trailing_zero_run(x[:W.REAL]) for x in X]) if len(X) else np.array([])
    check(bool(np.all(inside)), f"2. every window inside its clip ({int(np.sum(~inside))} outside)")
    check(bool(np.all(tail0)), f"   last {W.TAIL} samples zero in every window ({int(np.sum(~tail0))} not)")
    # Not a failure: check 2 already proves no window reaches past its clip. A real part that
    # ends in zeros is digital silence inside the recording itself; listed so the reviewer can look.
    long_runs = np.flatnonzero(runs > MAX_ZERO_RUN)
    print(f"NOTE     {len(long_runs)} window(s) whose real part ends in > {MAX_ZERO_RUN} zeros "
          f"(silence inside the source recording, not padding)")
    for j in long_runs[:5]:
        print(f"         {z['paths'][int(row[j])]}  start {int(start[j])}  clip {int(clen[j])} samples")

    # 3. per-source feature table (gun windows + one background epoch)
    feats = defaultdict(list)
    for x, i, g in zip(X, row, gain):
        r = rows[int(i)]
        feats[r["source"]].append((len(x), trailing_zero_run(x), g, rolloff(x[:W.REAL])))

    dev = rows
    split_note = "all rows"
    sp = root / "splits.json"
    if sp.exists():
        from splits import load_splits
        s = load_splits(sp, rows)
        dev = s["folds"][a.fold]["train"] if a.fold is not None else s["dev"]
        split_note = f"fold {a.fold} train" if a.fold is not None else "development rows (test excluded)"
    sampler = W.BackgroundSampler(dev, seed=a.seed)
    bgX, info = sampler.epoch(a.epoch)
    for x, rec in zip(bgX, info):
        key = rec["source"] + ("+event" if rec["mixed_with"] else "")
        feats[key].append((len(x), trailing_zero_run(x), -1, rolloff(x[:W.REAL])))
    print("\n   source              windows  length  zero-tail  gain-shift  roll-off-95% Hz (p10/p50/p90)")
    lay = set()
    for src, v in sorted(feats.items()):
        v = np.array(v, dtype=float)
        lay.add((int(v[0, 0]), int(np.min(v[:, 1])) >= W.TAIL))
        g = v[:, 2][v[:, 2] >= 0]
        print(f"   {src:<20}{len(v):>7}  {int(v[0, 0]):>6}  {int(np.min(v[:, 1])):>9}  "
              f"{(f'{np.median(g):.0f}' if g.size else '-'):>10}  "
              f"{np.percentile(v[:, 3], 10):>6.0f}/{np.percentile(v[:, 3], 50):.0f}/{np.percentile(v[:, 3], 90):.0f}")
    check(lay == {(W.INPUT, True)}, f"3. identical layout for every source: {W.INPUT} samples, >= {W.TAIL} zero tail")

    # 4. gun-mic share
    gm = np.mean([rec["gun_mic"] for rec in info])
    check(gm >= 0.15, f"4. gun-microphone share of one background epoch ({split_note}, n={a.epoch}): {gm:.1%} >= 15%")

    # 5. listening set
    out = wdir / "listen"
    out.mkdir(exist_ok=True)
    by_cls, by_src = defaultdict(list), {}
    for j, i in enumerate(row):
        r = rows[int(i)]
        by_cls[r["class"]].append(j)
        by_src.setdefault(r["source"], j)
    n = 0
    for c, js in by_cls.items():
        for j in rng.choice(js, size=min(10, len(js)), replace=False):
            r = rows[int(row[j])]
            write_wav(out / f"{c}__{r['source']}__{Path(r['path']).stem[:60]}__t{int(trig[j])}.wav", X[j])
            n += 1
    for k in rng.choice(len(bgX), size=min(10, len(bgX)), replace=False):
        write_wav(out / f"background__{info[k]['source']}{'+event' if info[k]['mixed_with'] else ''}__{k}.wav", bgX[k])
        n += 1
    print(f"\n5. wrote {n} windows to {out}: listen and confirm the shot is inside each one "
          f"(it starts {W.PRE / W.SR * 1000:.0f} ms in)")

    print("\nALL CHECKS PASS" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
