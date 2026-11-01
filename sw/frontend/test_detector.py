#!/usr/bin/env python3
"""detect() == detect_reference_loop() on synthetic edge cases and, optionally, real clips.

The per-sample loop is the form the RTL takes (Step 4.7); this test is what lets the RTL
testbench use detect() as its golden model. Also exports a test vector for the RTL.

Usage:
    python sw/frontend/test_detector.py [--dataset-final PATH --n 40] [--vector OUT.npz]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import detector as D  # noqa: E402

VARIANTS = [D.DEFAULT, replace(D.DEFAULT, ratio_shift=3, floor_shift=4), replace(D.DEFAULT, hold=64, retrig_shift=2),
            D.DEFAULT.scaled(12)]


def synthetic(rng) -> list[np.ndarray]:
    s = [np.zeros(20000, np.int32), np.full(9000, -32768, np.int32),
         np.tile([32767, -32768], 6000).astype(np.int32), np.arange(50, dtype=np.int32), np.zeros(64, np.int32)]
    x = rng.normal(0, 30, 40000).astype(np.int32)
    x[[63, 64, 127, 8191, 16000]] = 30000                      # impulses on frame edges
    s.append(x)
    y = rng.normal(0, 40, 30000).astype(np.int32)                # weak event, then a loud one inside refractory
    y[4000:4100] += 3000
    y[6000:6300] += rng.normal(0, 20000, 300).astype(np.int32)
    s.append(np.clip(y, -32768, 32767))
    z = rng.normal(0, 40, 30000).astype(np.int32)                # decaying burst (tail must not retrigger)
    z[5000:13000] += (rng.normal(0, 1, 8000) * 25000 * np.exp(-np.arange(8000) / 1500)).astype(np.int32)
    s.append(np.clip(z, -32768, 32767))
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-final", type=Path)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--vector", type=Path, help="write one clip's PCM and triggers for the RTL testbench")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cases = [("synthetic", x) for x in synthetic(rng)]
    if a.dataset_final:
        sys.path.insert(0, str(HERE.parent / "data"))
        import windows as W
        from manifest import load_manifest
        rows = load_manifest(a.dataset_final / "manifest.csv")
        for i in rng.choice(len(rows), size=min(a.n, len(rows)), replace=False):
            cases.append((rows[int(i)]["path"], W.load_pcm(rows[int(i)])))
    bad = 0
    for name, x in cases:
        for p in VARIANTS:
            q = x if p.bits == 16 else (x >> (16 - p.bits))
            if D.detect(q, p) != D.detect_reference_loop(q, p):
                bad += 1
                print("MISMATCH", name, p)
    print(f"{len(cases) * len(VARIANTS) - bad}/{len(cases) * len(VARIANTS)} cases identical")
    if a.vector:
        x = cases[6][1]
        np.savez(a.vector, pcm=x.astype(np.int16), triggers=np.asarray(D.detect(x), np.int64),
                 energy=D.energy(x), params=np.asarray(list(D.DEFAULT.__dict__.values()), np.int64),
                 param_names=np.asarray(list(D.DEFAULT.__dict__.keys())))
        print(f"wrote {a.vector}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
