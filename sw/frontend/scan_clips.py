#!/usr/bin/env python3
"""Scan every clip once: bandwidth per source, and the background index (Step 1.6, item 5).

For each clip of the manifest (streamed, one clip per worker task, nothing
decoded is kept): native sample rate, length at 16 kHz, RMS, longest run of
exact zeros, the 95 % spectral roll-off and the share of energy above 5.5 kHz
(both measured on the int16 16 kHz stream the model sees). For background clips
also the detector triggers whose window fits the D22 placement, so the epoch
sampler can cut background windows exactly the way gun windows are cut.

The roll-off is a measurement for the bandwidth decision, not part of the
hardware path, so it uses floating point.

Outputs
  <report-dir>/background_index.csv   one row per background clip (the epoch sampler reads it)
  <report-dir>/bandwidth_by_source.csv / .md   roll-off and high-band share per source and native rate
  <out>/clip_scan.csv                 every clip, every measurement

Usage
  python sw/frontend/scan_clips.py --manifest DATASET/manifest.csv --out OUTDIR --report-dir results/step16
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from audio import load_clip  # noqa: E402
from config import FRONTEND  # noqa: E402
from detector import detect  # noqa: E402
from manifest import load_manifest  # noqa: E402
from windows import placement  # noqa: E402

HF_EDGE_HZ = 5500.0      # a clip recorded at 11.025 kHz has nothing above this


def longest_zero_run(x: np.ndarray) -> int:
    z = np.concatenate(([0], (x == 0).astype(np.int8), [0]))
    d = np.diff(z)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max()) if len(starts) else 0


def spectrum_stats(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(95 % roll-off in Hz, energy above HF_EDGE_HZ relative to the total in dB)."""
    if len(x) < 512 or not np.any(x):
        return float("nan"), float("nan")
    f, p = welch(x.astype(np.float64) / 32768.0, fs=sr, nperseg=512)
    c = np.cumsum(p)
    roll = float(f[np.searchsorted(c, 0.95 * c[-1])])
    hf = float(10 * np.log10(max(p[f >= HF_EDGE_HZ].sum(), 1e-30) / max(c[-1], 1e-30)))
    return roll, hf


def scan(row: dict) -> dict:
    cfg = FRONTEND
    x = load_clip(row, cfg.window)
    roll, hf = spectrum_stats(x, cfg.window.sample_rate)
    rms = float(np.sqrt(np.mean((x.astype(np.float64) / 32768.0) ** 2))) if len(x) else 0.0
    out = {k: row[k] for k in ("path", "class", "session_id", "group", "source", "device")}
    out.update(native_sr=row["sample_rate"], n_samples=len(x), rms_dbfs=round(20 * np.log10(max(rms, 1e-10)), 2),
               zero_run=longest_zero_run(x), rolloff_hz=round(roll, 1), hf_db=round(hf, 2), triggers="")
    if row["class"] == "background":
        t = detect(x, cfg.detector).triggers
        out["triggers"] = " ".join(str(int(v)) for v in t if placement(int(v), len(x), cfg.window)[1] == "ok")
    return out


def pct(v, q):
    v = np.asarray([a for a in v if np.isfinite(a)])
    return [round(float(np.percentile(v, p)), 1) for p in q] if len(v) else [float("nan")] * len(q)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.report_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(a.manifest)
    root = a.manifest.resolve().parent
    res = []
    with Pool(a.jobs) as pool:
        for k, r in enumerate(pool.imap(scan, rows, chunksize=8), 1):
            r["path"] = os.path.relpath(r["path"], root)
            res.append(r)
            if k % 2000 == 0:
                print(f"  {k}/{len(rows)}", flush=True)

    cols = list(res[0].keys())
    with open(a.out / "clip_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(res)
    with open(a.report_dir / "background_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(r for r in res if r["class"] == "background")

    min_sr = FRONTEND.background.min_native_sr
    groups = defaultdict(list)
    for r in res:
        groups[(r["source"], f"native >= {min_sr}" if r["native_sr"] >= min_sr else f"native < {min_sr}")].append(r)
    q = (5, 50, 95)
    lines = ["# Bandwidth per source (Step 1.6, item 5)", "",
             "Measured by `sw/frontend/scan_clips.py` on every clip after resampling to 16 kHz and int16. "
             f"Roll-off = frequency below which 95 % of the energy lies; high band = energy above "
             f"{HF_EDGE_HZ / 1000:.1f} kHz relative to the total.", "",
             "| source | native rate | clips | roll-off p5 / p50 / p95 (Hz) | high band p5 / p50 / p95 (dB) |",
             "|---|---|---:|---|---|"]
    with open(a.report_dir / "bandwidth_by_source.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "native_rate", "clips", "rolloff_p5", "rolloff_p50", "rolloff_p95",
                    "hf_db_p5", "hf_db_p50", "hf_db_p95"])
        for (src, grp), rs in sorted(groups.items()):
            ro, hf = pct([r["rolloff_hz"] for r in rs], q), pct([r["hf_db"] for r in rs], q)
            w.writerow([src, grp, len(rs)] + ro + hf)
            lines.append(f"| {src} | {grp} | {len(rs)} | {' / '.join(map(str, ro))} | {' / '.join(map(str, hf))} |")
    (a.report_dir / "bandwidth_by_source.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
