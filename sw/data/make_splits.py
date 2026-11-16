#!/usr/bin/env python3
"""Final test set, cross-validation folds, device-disjoint and random splits (Step 1.6, items 6-9).

Decision D23 (test set): every clip of every class, background included, whose
session_id is c3gd_nj_farm_2 or edge-day2, plus a seeded 15 % of the remaining
background-only sessions (chosen by session_id; sessions that also hold gun
clips, such as the cadre and edge-day1 quiet stretches, are never candidates,
so a session is never split between test and folds).

Excluded from every split: background clips recorded below 16 kHz (D22,
bandwidth note).

Cross-validation: StratifiedGroupKFold on everything else, grouped by
session_id, stratified by class. Groups (one shot, or one recording) never span
sessions in this manifest, so session-disjoint folds are also group-disjoint;
check_splits.py proves both.

Device-disjoint split (a harder second test, within the CV sessions): per gun
source, a seeded random set of devices is held out until it holds at least
--device-share of that source's gun clips; the quiet stretches recorded on a
held-out device go with it, and a seeded share of the background-only sessions
is added so the test has background. The same shot recorded on several
devices shares a group, so this split is device-disjoint but not
group-disjoint; the number of shared groups is recorded.

Random split: a stratified clip-level split of the CV clips. It leaks sessions
on purpose, to show how much that inflates accuracy.

Usage
  python sw/data/make_splits.py --manifest DATASET/manifest.csv --out sw/data/splits.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
from config import FRONTEND  # noqa: E402
from manifest import classes, load_manifest  # noqa: E402
from splits import file_sha256  # noqa: E402

TEST_SESSIONS = ("c3gd_nj_farm_2", "edge-day2")
GUN_MIC_QUIET = {"edge-quiet": "edge", "cadre-quiet": "cadre"}


def device_key(r: dict) -> str:
    """Physical recorder: source family plus device name."""
    return f"{GUN_MIC_QUIET.get(r['source'], r['source'])}:{r['device']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--bg-test-share", type=float, default=0.15)
    ap.add_argument("--device-share", type=float, default=0.20)
    ap.add_argument("--random-test-share", type=float, default=0.20)
    a = ap.parse_args()

    seeds = FRONTEND.seeds
    min_sr = FRONTEND.background.min_native_sr
    root = a.manifest.resolve().parent
    rows = load_manifest(a.manifest)
    for r in rows:
        r["rel"] = str(Path(r["path"]).relative_to(root))
    cls_order = classes(rows)

    excluded = [{"path": r["rel"], "reason": f"native_sr_{r['sample_rate']}_below_{min_sr}"}
                for r in rows if r["sample_rate"] < min_sr]
    assert all(r["class"] == "background" for r in rows if r["sample_rate"] < min_sr)
    ex = {e["path"] for e in excluded}
    kept = [r for r in rows if r["rel"] not in ex]

    gun_sessions = {r["session_id"] for r in kept if r["class"] != "background"}
    bg_only = sorted({r["session_id"] for r in kept} - gun_sessions - set(TEST_SESSIONS))
    rng = np.random.default_rng(seeds.background_test_sessions)
    n_bg = int(round(a.bg_test_share * len(bg_only)))
    bg_test = sorted(rng.choice(bg_only, n_bg, replace=False).tolist())
    test_sessions = set(TEST_SESSIONS) | set(bg_test)
    test = [r for r in kept if r["session_id"] in test_sessions]
    rest = [r for r in kept if r["session_id"] not in test_sessions]

    y = np.array([cls_order.index(r["class"]) for r in rest])
    g = np.array([r["session_id"] for r in rest])
    sgkf = StratifiedGroupKFold(n_splits=a.n_folds, shuffle=True, random_state=seeds.splits)
    folds = []
    for _, val in sgkf.split(np.zeros(len(rest)), y, g):
        v = [rest[i] for i in val]
        folds.append({"val_sessions": sorted({r["session_id"] for r in v}), "val_paths": sorted(r["rel"] for r in v)})

    # Device-disjoint split inside the CV pool.
    rng = np.random.default_rng(seeds.splits + 1)
    held = set()
    for src in ("c3gd", "cadre", "edge"):
        guns = [r for r in rest if r["source"] == src]
        devs = sorted({device_key(r) for r in guns})
        rng.shuffle(devs)
        per_dev = Counter(device_key(r) for r in guns)
        n = 0
        for d in devs:
            if n >= a.device_share * len(guns):
                break
            held.add(d)
            n += per_dev[d]
    rest_bg_only = sorted({r["session_id"] for r in rest} - gun_sessions)
    dev_bg = set(rng.choice(rest_bg_only, int(round(a.device_share * len(rest_bg_only))), replace=False).tolist())
    dev_test = [r for r in rest if (r["device"] and device_key(r) in held) or r["session_id"] in dev_bg]
    dev_test_paths = {r["rel"] for r in dev_test}
    dev_train = [r for r in rest if r["rel"] not in dev_test_paths]
    shared_groups = {r["group"] for r in dev_test if r["class"] != "background"} & \
                    {r["group"] for r in dev_train if r["class"] != "background"}

    _, rnd_test = train_test_split(np.arange(len(rest)), test_size=a.random_test_share, stratify=y,
                                   random_state=seeds.splits + 2)

    out = {
        "version": 1,
        "created_by": "sw/data/make_splits.py",
        "manifest_sha256": file_sha256(a.manifest),
        "paths_relative_to": "the folder holding manifest.csv",
        "classes": cls_order,
        "seeds": {"cv": seeds.splits, "background_test_sessions": seeds.background_test_sessions,
                  "device_split": seeds.splits + 1, "random_split": seeds.splits + 2},
        "excluded": excluded,
        "test": {"decision": "D23", "sessions_fixed": list(TEST_SESSIONS), "background_share": a.bg_test_share,
                 "background_candidate_sessions": len(bg_only), "background_sessions_random": bg_test,
                 "paths": sorted(r["rel"] for r in test)},
        "cv": {"method": "sklearn StratifiedGroupKFold(shuffle=True)", "n_splits": a.n_folds, "groups": "session_id",
               "stratify": "class", "folds": folds},
        "device_split": {"held_out_devices": sorted(held), "device_share": a.device_share,
                         "background_sessions": sorted(dev_bg), "shared_groups": len(shared_groups),
                         "train": "cv pool minus test_paths", "test_paths": sorted(dev_test_paths)},
        "random_split": {"test_share": a.random_test_share, "stratify": "class", "train": "cv pool minus test_paths",
                         "test_paths": sorted(rest[i]["rel"] for i in rnd_test)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=0) + "\n")
    print(f"Wrote {a.out}: {len(test)} test clips ({len(bg_test)} random background sessions), "
          f"{len(rest)} CV clips in {a.n_folds} folds, {len(excluded)} excluded")
    print(f"sha256 {file_sha256(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
