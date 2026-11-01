#!/usr/bin/env python3
"""Prove splits.json has no leakage (Step 1.6 review gate). Exit code 0 only if every check passes.

Checks
  1. splits.json was made from this manifest.csv (sha256).
  2. Every row is in exactly one of: test, one CV validation fold.
  3. No session_id and no group appears in more than one CV fold.
  4. No session_id and no group appears in both the test set and the development set.
  5. Every row of every D23 test session is in the test set.
  6. Every firearm class is tested on sessions from at least two different datasets (sources).
  7. Every firearm class has at least one validation fold, and every fold has background.
  8. Device split: no held-out device appears in its training side, and its rows are all
     development rows (never the test set). Random split: development rows only.

How could information from the test set reach training? Each path below is closed by a check:
  - the same shot on several microphones         -> same group        -> check 3/4
  - quiet stretch cut from a test gun clip        -> same session_id   -> check 4/5
  - another clip from the same Freesound source   -> same session_id (us8k-<fsID>, esc50-<src>) -> 4
  - another clip of the same MAD video            -> same session_id (mad-<video>) -> 4
  - normalisation or thresholds fitted on all data -> none: detector and window gain are fixed rules
  - choosing the model on the test set            -> the test set is used once, in Step 1.11

Usage:
    python sw/data/check_splits.py --dataset-final PATH [--splits FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import load_manifest  # noqa: E402
from splits import rel, sha256  # noqa: E402

BG = "background"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-final", type=Path, required=True)
    ap.add_argument("--splits", type=Path)
    a = ap.parse_args()
    root = a.dataset_final.resolve()
    sp = a.splits or root / "splits.json"
    doc = json.load(open(sp))
    rows = load_manifest(root / "manifest.csv")
    by = dict(zip(rel(rows, root), rows))
    fails: list[str] = []

    def check(ok: bool, msg: str):
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    check(sha256(root / "manifest.csv") == doc["meta"]["manifest_sha256"], "1. splits.json matches manifest.csv")

    test = doc["test"]
    folds = doc["cv"]["folds"]
    unknown = [p for p in test + [p for f in folds for p in f] if p not in by]
    check(not unknown, f"   all split paths exist in the manifest ({len(unknown)} unknown)")
    seen = defaultdict(int)
    for p in test:
        seen[p] += 1
    for f in folds:
        for p in f:
            seen[p] += 1
    dup = [p for p, n in seen.items() if n > 1]
    lost = [p for p in by if p not in seen]
    check(not dup and not lost, f"2. every row in exactly one of test / folds (duplicated {len(dup)}, missing {len(lost)})")

    for key in ("session_id", "group"):
        where = defaultdict(set)
        for k, f in enumerate(folds):
            for p in f:
                where[by[p][key]].add(k)
        multi = {v: s for v, s in where.items() if len(s) > 1}
        check(not multi, f"3. no {key} in more than one fold ({len(multi)} found"
              + (f", e.g. {next(iter(multi))}" if multi else "") + ")")
        t = {by[p][key] for p in test}
        d = {by[p][key] for f in folds for p in f}
        both = t & d
        check(not both, f"4. no {key} in both test and development ({len(both)} found"
              + (f", e.g. {next(iter(both))}" if both else "") + ")")

    ts = set(doc["meta"]["test_sessions"])
    tset = set(test)
    out_of_test = [p for p, r in by.items() if r["session_id"] in ts and p not in tset]
    check(not out_of_test, f"5. every row of test sessions {sorted(ts)} is in the test set ({len(out_of_test)} outside)")

    src = defaultdict(set)
    for p in test:
        r = by[p]
        if r["class"] != BG:
            src[r["class"]].add(r["source"])
    gun_classes = sorted({r["class"] for r in rows if r["class"] != BG})
    weak = [c for c in gun_classes if len(src[c]) < 2]
    check(not weak, "6. every firearm class tested on two datasets: "
          + ", ".join(f"{c}={'+'.join(sorted(src[c])) or 'none'}" for c in gun_classes))

    per_fold = [{by[p]["class"] for p in f} for f in folds]
    no_val = [c for c in gun_classes if not any(c in s for s in per_fold)]
    no_bg = [k for k, s in enumerate(per_fold) if BG not in s]
    check(not no_val and not no_bg, f"7. every firearm class validated somewhere (missing {no_val}); "
          f"background in every fold (missing in {no_bg})")
    for c in gun_classes:
        ks = [k for k, s in enumerate(per_fold) if c in s]
        sess = sorted({by[p]["session_id"] for f in folds for p in f if by[p]["class"] == c})
        print(f"      {c}: validated in folds {ks}; development sessions {sess}")

    dv = doc["device"]
    held = set(dv["held_out_devices"])
    leak_dev = [p for p in dv["train"] if by[p]["device"] in held]
    in_test = [p for p in dv["train"] + dv["test"] + doc["random"]["train"] + doc["random"]["test"] if p in tset]
    check(not leak_dev, f"8. device split: no held-out device in its training side ({len(leak_dev)} found)")
    check(not in_test, f"   device and random splits use development rows only ({len(in_test)} test rows found)")
    print(f"      device split: {len(held)} held-out devices; the same shot on a train and a test device: "
          f"{dv.get('groups_on_both_sides', '?')} groups (by design, it tests the microphone, not the shot)")

    print("\nALL CHECKS PASS" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
