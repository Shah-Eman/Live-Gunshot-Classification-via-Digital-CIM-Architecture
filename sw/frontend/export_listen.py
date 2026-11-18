#!/usr/bin/env python3
"""Listening set for the review gate: 10 windows per class as 16 kHz WAVs (Step 1.6).

Windows are exported exactly as the model sees them (after the gain, with the
zeroed tail), from the cross-validation pool only (never the test set). Gun
windows are spread over their sources; the trigger is pre_trigger_samples (0.04 s) into
every gun window, so the shot should be heard right at the start. Background windows are
drawn by the epoch sampler, mixing its kinds (detector-cut, gun-microphone,
gun-microphone with an event, random).

This gate is passed by a person listening, not by this script.

Usage
  python sw/frontend/export_listen.py --manifest DATASET/manifest.csv --splits sw/data/splits.json \
      --window-index results/step16/window_index.csv --background-index results/step16/background_index.csv \
      --gun-npz OUT/gun_windows.npz --out OUT/listen
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from audio import apply_gain, normalise  # noqa: E402
from background import EpochBuilder  # noqa: E402
from config import FRONTEND  # noqa: E402
from splits import load_splits  # noqa: E402

BG_KINDS = ["triggered"] * 3 + ["gunmic"] * 2 + ["gunmic_event"] * 2 + ["random"] * 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("manifest", "splits", "window-index", "background-index", "gun-npz", "out"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=10)
    a = ap.parse_args()

    cfg = FRONTEND
    sr = cfg.window.sample_rate
    s = load_splits(a.splits, manifest=a.manifest)
    pool = s.cv_pool
    with open(a.window_index, newline="") as f:
        index = list(csv.DictReader(f))
    d = np.load(a.gun_npz)
    X, K = d["windows"], d["gain_shift"]
    rng = np.random.default_rng(cfg.seeds.listen)
    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("*.wav"):
        old.unlink()

    listing = []
    for cls in [c for c in s.raw["classes"] if c != "background"]:
        cand = [i for i, r in enumerate(index) if r["class"] == cls and r["path"] in pool]
        by_src: dict[str, list[int]] = {}
        for i in cand:
            by_src.setdefault(index[i]["source"], []).append(i)
        picks, srcs = [], sorted(by_src)
        while len(picks) < a.per_class:
            src = srcs[len(picks) % len(srcs)]
            picks.append(int(rng.choice(by_src[src])))
        for n, i in enumerate(picks):
            r = index[i]
            name = f"{cls}_{n:02d}_{r['source']}.wav"
            sf.write(a.out / name, apply_gain(X[i], int(K[i])), sr, subtype="PCM_16")
            listing.append({"file": name, "class": cls, "kind": "gun", "source": r["source"], "clip": r["path"],
                            "window_start": r["window_start"], "trigger_sample": r["trigger_sample"],
                            "note": f"shot expected at {cfg.window.pre_trigger_samples / sr:.2f} s"})

    b = EpochBuilder.from_files(a.manifest, a.splits, a.window_index, a.gun_npz, a.background_index,
                                train_paths=pool, seed=cfg.seeds.listen)
    for n, kind in enumerate(BG_KINDS[: a.per_class]):
        w, m = b.background(rng, kind)
        out, _ = normalise(w, cfg.window, cfg.gain)
        name = f"background_{n:02d}_{kind}_{m['source']}.wav"
        sf.write(a.out / name, out, sr, subtype="PCM_16")
        listing.append({"file": name, "class": "background", "kind": kind, "source": m["source"], "clip": m["path"],
                        "window_start": m["start"], "trigger_sample": m.get("trigger", ""),
                        "note": f"mixed with {m['mixed_with']} at {m['snr_db']} dB" if "mixed_with" in m else ""})

    with open(a.out / "listen_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(listing[0].keys()))
        w.writeheader()
        w.writerows(listing)
    print(f"Wrote {len(listing)} WAVs to {a.out} (seed {cfg.seeds.listen}). Listen to them: this gate is not "
          f"passed until a person confirms the shot is inside every gun window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
