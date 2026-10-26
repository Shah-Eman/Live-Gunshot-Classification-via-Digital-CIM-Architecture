"""Dataset access through manifest.csv only (runbook Step 0.4, item 2).

Everything downstream (splits, training, test-vector generation) reads clips
through this module and never looks at raw folder names. Switching to another
dataset means producing a manifest.csv with the same columns; nothing here
changes.

Required columns: path, class, session_id, device, sample_rate, duration, source.
group (leakage id) is used when present and falls back to session_id.

    from manifest import load_manifest, load_audio
    rows = load_manifest("Dataset-Final/manifest.csv")
    x = load_audio(rows[0])             # float32 mono at 16 kHz, range [-1, 1]
"""
from __future__ import annotations

import csv
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REQUIRED = ("path", "class", "session_id", "device", "sample_rate", "duration", "source")
TARGET_SR = 16000


def load_manifest(manifest: str | Path) -> list[dict]:
    """Rows of manifest.csv, with path made absolute and group always set."""
    manifest = Path(manifest).resolve()
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{manifest} lacks columns {missing}; run sw/data/audit.py")
        rows = list(reader)
    for r in rows:
        r["path"] = str(manifest.parent / r["path"])
        r["sample_rate"] = int(r["sample_rate"])
        r["duration"] = float(r["duration"])
        r["group"] = r.get("group") or r["session_id"]
    return rows


def load_audio(row: dict, sr: int = TARGET_SR) -> np.ndarray:
    """One clip as float32 mono at sr: channels averaged, polyphase resampling."""
    x, native = sf.read(row["path"], dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if native != sr:
        g = gcd(native, sr)
        x = resample_poly(x, sr // g, native // g).astype(np.float32)
    return x


def classes(rows: list[dict]) -> list[str]:
    """Class names in a fixed order: background last, the rest alphabetical."""
    names = sorted({r["class"] for r in rows})
    if "background" in names:
        names.remove("background")
        names.append("background")
    return names
