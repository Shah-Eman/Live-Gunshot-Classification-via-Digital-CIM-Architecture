#!/usr/bin/env python3
"""Audit the final dataset (runbook Step 0.4, items 5 and 6).

Reads Dataset-Final/manifest.csv (written by sw/dataset/build_final_dataset.py),
measures every clip's sample rate, duration and channel count from its header,
and rewrites the manifest with the runbook's columns first:

    path, class, session_id, device, sample_rate, duration, source,
    then channels, gun, caliber, group, sub_label, orig_path

session_id is the recording session (a day or event at one place), so every
device that captured the same shot shares it. group is the finer leakage id
(one shot, or one recording of several shots) that must never be split.

Then prints, per class: clips, distinct sessions, distinct devices, sample
rates and sources, and writes the same table to Dataset-Final/audit_counts.csv.
Running it again is safe: it only fills in measured columns.

Usage:
    python sw/data/audit.py [--dataset-final PATH]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

COLUMNS = ["path", "class", "session_id", "device", "sample_rate", "duration", "source",
           "channels", "gun", "caliber", "group", "sub_label", "orig_path"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-final", type=Path,
                    default=Path(__file__).resolve().parents[3] / "Dataset-Final")
    root = ap.parse_args().dataset_final.resolve()
    manifest = root / "manifest.csv"

    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if "path" not in r:                     # build_final_dataset.py names it "file"
            r["path"] = r.pop("file")
        info = sf.info(root / r["path"])
        r["sample_rate"] = info.samplerate
        r["duration"] = f"{info.frames / info.samplerate:.4f}"
        r["channels"] = info.channels

    tmp = manifest.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in COLUMNS} for r in rows)
    tmp.replace(manifest)

    per = defaultdict(lambda: {"clips": 0, "sessions": set(), "devices": set(), "sr": Counter(),
                               "sources": Counter(), "seconds": 0.0})
    for r in rows:
        s = per[r["class"]]
        s["clips"] += 1
        s["sessions"].add(r["session_id"])
        if r["device"]:
            s["devices"].add(r["device"])
        s["sr"][int(r["sample_rate"])] += 1
        s["sources"][r["source"]] += 1
        s["seconds"] += float(r["duration"])

    table = []
    for c, s in sorted(per.items()):
        table.append({
            "class": c, "clips": s["clips"], "sessions": len(s["sessions"]), "devices": len(s["devices"]),
            "hours": f"{s['seconds'] / 3600:.2f}",
            "sample_rates": "; ".join(f"{k} Hz x{v}" for k, v in sorted(s["sr"].items())),
            "sources": "; ".join(f"{k} {v}" for k, v in s["sources"].most_common()),
        })
    with open(root / "audit_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    for t in table:
        print(f"\n{t['class']}: {t['clips']} clips, {t['sessions']} sessions, {t['devices']} devices, "
              f"{t['hours']} h")
        print(f"  sample rates: {t['sample_rates']}")
        print(f"  sources:      {t['sources']}")
    print(f"\nWrote {manifest} and {root / 'audit_counts.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
