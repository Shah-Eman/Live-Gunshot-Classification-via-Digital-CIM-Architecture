#!/usr/bin/env python3
"""Cut every gun window with detector.py and report how well it found the shots (Step 1.6, item 3).

Reads Dataset-Final/manifest.csv through sw/data/manifest.py, runs the integer detector on
every gun clip, places each trigger's window (windows.py, Decision D22), and writes

    <out>/gun_windows.npz      windows int16[N, 8000], row (manifest index), trigger, start,
                               gain_shift, clip_len, and the manifest paths
    <out>/detector_report.csv  one line per gun clip: triggers, which matched a known shot,
                               which fitted, windows kept
    <out>/detector_summary.csv recall per source, class and session (the review-gate number)

Reference shots, used only to check the detector: C3GD clips hold one shot 46-141 ms in,
Cadre clips one shot at the clip's peak, edge clips the labelled times in Kabealo's
gunshot-audio-all-metadata.csv (pass --edge-meta; without it edge recall is not measured
and edge triggers are kept unverified).

Usage:
    python sw/frontend/cut_windows.py --dataset-final PATH [--edge-meta CSV] [--out DIR] [--jobs N]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
import windows as W                         # noqa: E402
from manifest import load_manifest          # noqa: E402

RECALL_GATE = 0.95


def load_edge_shots(path: Path | None) -> dict | None:
    if path is None:
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        r = {k.strip(): v for k, v in r.items() if k}
        t = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", r.get("gunshot_location_in_seconds", ""))]
        out[Path(r["filename"]).stem] = t
    return out


_EDGE = None


def _init(edge):
    global _EDGE
    _EDGE = edge


def _job(item):
    i, row = item
    try:
        c = W.cut_gun(row, _EDGE)
        return i, c, c.n_samples, ""
    except Exception as e:                              # a broken file must not stop the run
        return i, None, 0, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-final", type=Path, required=True)
    ap.add_argument("--edge-meta", type=Path, help="Kabealo gunshot-audio-all-metadata.csv")
    ap.add_argument("--out", type=Path, help="default: <dataset-final>/windows")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    out = a.out or a.dataset_final / "windows"
    out.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(a.dataset_final / "manifest.csv")
    guns = [(i, r) for i, r in enumerate(rows) if r["class"] != "background"]
    edge = load_edge_shots(a.edge_meta)
    print(f"{len(guns)} gun clips; edge reference {'loaded' if edge else 'NOT given (edge recall not measured)'}")

    with Pool(a.jobs, initializer=_init, initargs=(edge,)) as pool:
        results = pool.map(_job, guns, chunksize=16)

    win, meta, report = [], defaultdict(list), []
    summ = defaultdict(lambda: {"clips": 0, "ref_shots": 0, "found": 0, "triggers": 0, "unmatched": 0,
                                "no_fit": 0, "windows": 0, "errors": 0, "no_ref": 0})
    for (i, row), (_, c, n, err) in zip(guns, results):
        keys = [("source", row["source"]), ("class", row["class"]), ("session", row["session_id"]), ("all", "all")]
        for k in keys:
            summ[k]["clips"] += 1
        if c is None:
            for k in keys:
                summ[k]["errors"] += 1
            report.append({"path": row["path"], "class": row["class"], "source": row["source"],
                           "session_id": row["session_id"], "error": err})
            continue
        for k in keys:
            s = summ[k]
            s["triggers"] += len(c.triggers)
            s["unmatched"] += sum(1 for m in c.matched if m is False)
            s["no_fit"] += sum(1 for st, m in zip(c.starts, c.matched) if st is None and m is not False)
            s["windows"] += len(c.windows)
            if c.n_ref is None:
                s["no_ref"] += 1
            else:
                s["ref_shots"] += c.n_ref
                s["found"] += c.n_found
        for w, ti, g in zip(c.windows, c.kept, c.gains):
            win.append(w)
            meta["row"].append(i)
            meta["trigger"].append(c.triggers[ti])
            meta["start"].append(c.starts[ti])
            meta["gain_shift"].append(g)
            meta["clip_len"].append(n)
        report.append({"path": row["path"], "class": row["class"], "source": row["source"],
                       "session_id": row["session_id"], "n_ref": "" if c.n_ref is None else c.n_ref,
                       "found": c.n_found, "triggers": " ".join(map(str, c.triggers)),
                       "matched": " ".join({True: "y", False: "n", None: "?"}[m] for m in c.matched),
                       "fits": " ".join("n" if s is None else "y" for s in c.starts),
                       "windows": len(c.windows), "error": ""})

    np.savez_compressed(out / "gun_windows.npz",
                        windows=np.stack(win) if win else np.zeros((0, W.INPUT), np.int16),
                        **{k: np.asarray(v, dtype=np.int64) for k, v in meta.items()},
                        paths=np.asarray([r["path"] for r in rows]),
                        layout=np.asarray([W.INPUT, W.PRE, W.REAL, W.TAIL]))
    cols = ["path", "class", "source", "session_id", "n_ref", "found", "triggers", "matched", "fits", "windows", "error"]
    with open(out / "detector_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(report)

    table = []
    for (kind, name), s in sorted(summ.items()):
        rec = s["found"] / s["ref_shots"] if s["ref_shots"] else None
        table.append({"group_by": kind, "name": name, **s,
                      "recall": "" if rec is None else f"{rec:.4f}",
                      "gate": "" if rec is None else ("pass" if rec >= RECALL_GATE else "FAIL")})
    with open(out / "detector_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for t in table:
        if t["group_by"] in ("source", "all"):
            print(f"{t['group_by']:>7} {t['name']:<14} clips {t['clips']:>6}  shots {t['ref_shots']:>6}  "
                  f"found {t['found']:>6}  recall {t['recall'] or 'n/a':>6} {t['gate']:<4}  "
                  f"windows {t['windows']:>6}  unmatched triggers {t['unmatched']}  did not fit {t['no_fit']}")
    print(f"\nWrote {out / 'gun_windows.npz'} ({len(win)} windows), detector_report.csv, detector_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
