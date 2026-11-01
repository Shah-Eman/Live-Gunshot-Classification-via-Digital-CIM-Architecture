#!/usr/bin/env python3
"""Session-disjoint test set, cross-validation folds and the two extra splits (Step 1.6, items 6-9).

Writes splits.json next to manifest.csv (or --out). Run once, before any training, and
commit the file's sha256 with Decision D23; training and evaluation read it through
`load_splits()` and never re-split.

Leakage unit. Rows that share a session_id or a group must stay together. Groups are
nested in sessions in the current manifest, but this script does not assume it: it joins
every session_id and group that co-occur into one "unit" (union-find) and splits units.

1. Test set (D23). Every row of every unit that contains a session in --test-sessions
   (default c3gd_nj_farm_2 and edge-day2, background quiet stretches included because
   they carry their gun clip's session_id), plus --bg-test-frac (15%) of the remaining
   background-only units, drawn at random with --seed. Used once, in Step 1.11.
2. Cross-validation on the rest: StratifiedGroupKFold(--folds, shuffle, --seed),
   stratified by class, grouped by unit. Each fold lists its validation rows; the
   training rows of fold k are all development rows not in fold k.
3. Device-disjoint split (a second, harder test): devices of the development gun clips
   are held out (shuffled with --seed, added until every firearm class has at least
   --device-frac of its development clips held out). Held-out-device rows are the test;
   development background without a device is split 80/20 by unit. Sessions are shared
   on purpose; the same shot can be on a train device and a test device (reported).
4. Random clip-level split (80/20, stratified by class, groups ignored), only to show
   how much session leakage inflates accuracy. Never use it for model selection.

Usage:
    python sw/data/splits.py --dataset-final PATH [--seed 0] [--folds 5] [--out FILE]

    from splits import load_splits
    s = load_splits("Dataset-Final/splits.json", rows)   # rows from load_manifest
    s["test"], s["folds"][k]["train"], s["folds"][k]["val"], s["device"]["train"], ...
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import classes, load_manifest  # noqa: E402

TEST_SESSIONS = ("c3gd_nj_farm_2", "edge-day2")
BG = "background"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leak_units(rows: list[dict]) -> list[int]:
    """Unit id per row: connected components of rows linked by session_id or group."""
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in rows:
        a, b = find("s:" + r["session_id"]), find("g:" + r["group"])
        if a != b:
            parent[b] = a
    roots = [find("s:" + r["session_id"]) for r in rows]
    ids = {k: i for i, k in enumerate(sorted(set(roots)))}
    return [ids[k] for k in roots]


def rel(rows: list[dict], root: Path) -> list[str]:
    """Manifest-relative paths (as written in manifest.csv). Symlinks are not followed."""
    return [Path(os.path.relpath(r["path"], root)).as_posix() for r in rows]


def make_splits(rows: list[dict], test_sessions=TEST_SESSIONS, bg_test_frac=0.15, n_folds=5,
                device_frac=0.2, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    unit = leak_units(rows)
    cls = np.array([r["class"] for r in rows])

    sessions_of_unit = defaultdict(set)
    classes_of_unit = defaultdict(set)
    for r, u in zip(rows, unit):
        sessions_of_unit[u].add(r["session_id"])
        classes_of_unit[u].add(r["class"])
    missing = set(test_sessions) - {r["session_id"] for r in rows}
    if missing:
        raise ValueError(f"test sessions not in manifest: {sorted(missing)}")

    test_units = {u for u, ss in sessions_of_unit.items() if ss & set(test_sessions)}
    bg_only = sorted(u for u, cs in classes_of_unit.items() if cs == {BG} and u not in test_units)
    n_bg = int(round(bg_test_frac * len(bg_only)))
    test_units |= set(rng.choice(bg_only, size=n_bg, replace=False).tolist()) if n_bg else set()

    test_idx = [i for i, u in enumerate(unit) if u in test_units]
    dev_idx = np.array([i for i, u in enumerate(unit) if u not in test_units])

    # 2. cross-validation
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for _, val in sgkf.split(dev_idx, cls[dev_idx], groups=np.array(unit)[dev_idx]):
        folds.append(sorted(dev_idx[val].tolist()))

    # 3. device-disjoint
    gun_dev = [i for i in dev_idx if rows[i]["class"] != BG and rows[i]["device"]]
    per_dev = defaultdict(Counter)
    for i in gun_dev:
        per_dev[rows[i]["device"]][rows[i]["class"]] += 1
    total = Counter(rows[i]["class"] for i in gun_dev)
    devices = sorted(per_dev)
    rng.shuffle(devices)
    held, got = [], Counter()
    for d in devices:
        short = [c for c in total if got[c] < device_frac * total[c]]
        if not short:
            break
        if any(per_dev[d][c] for c in short):
            held.append(d)
            got.update(per_dev[d])
    held_set = set(held)
    dev_test, dev_train = [], []
    bg_nodev_units = sorted({unit[i] for i in dev_idx if rows[i]["class"] == BG and not rows[i]["device"]})
    bg_test_u = set(rng.choice(bg_nodev_units, size=int(round(0.2 * len(bg_nodev_units))), replace=False).tolist()) \
        if bg_nodev_units else set()
    for i in dev_idx:
        r = rows[i]
        if r["device"]:
            (dev_test if r["device"] in held_set else dev_train).append(int(i))
        else:
            (dev_test if (r["class"] == BG and unit[i] in bg_test_u) else dev_train).append(int(i))
    shared_groups = {rows[i]["group"] for i in dev_test} & {rows[i]["group"] for i in dev_train}

    # 4. random clip-level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(sss.split(dev_idx, cls[dev_idx]))

    return {"test": test_idx, "folds": folds, "unit": unit,
            "device": {"held_out_devices": sorted(held), "train": sorted(dev_train), "test": sorted(dev_test),
                       "groups_on_both_sides": len(shared_groups)},
            "random": {"train": sorted(dev_idx[tr].tolist()), "test": sorted(dev_idx[te].tolist())}}


def counts(rows, idx, key="class"):
    return dict(sorted(Counter(rows[i][key] for i in idx).items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-final", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: <dataset-final>/splits.json")
    ap.add_argument("--test-sessions", nargs="+", default=list(TEST_SESSIONS))
    ap.add_argument("--bg-test-frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.json")
    a = ap.parse_args()

    root = a.dataset_final.resolve()
    manifest = root / "manifest.csv"
    out = a.out or root / "splits.json"
    if out.exists() and not a.force:
        print(f"{out} exists. The test set is chosen once (D23); pass --force only if the team agreed.")
        return 1
    rows = load_manifest(manifest)
    s = make_splits(rows, a.test_sessions, a.bg_test_frac, a.folds, a.device_frac, a.seed)

    P = lambda idx: rel([rows[i] for i in idx], root)  # noqa: E731
    doc = {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha256": sha256(manifest), "n_rows": len(rows), "classes": classes(rows),
            "seed": a.seed, "decision": "D23", "test_sessions": a.test_sessions,
            "bg_test_frac": a.bg_test_frac, "n_folds": a.folds, "device_frac": a.device_frac,
            "test_session_ids": sorted({rows[i]["session_id"] for i in s["test"]}),
            "note": "paths are relative to the manifest's folder; the test set is used once, in Step 1.11",
        },
        "test": P(s["test"]),
        "cv": {"folds": [P(f) for f in s["folds"]]},
        "device": {"held_out_devices": s["device"]["held_out_devices"], "train": P(s["device"]["train"]),
                   "test": P(s["device"]["test"]), "groups_on_both_sides": s["device"]["groups_on_both_sides"]},
        "random": {"train": P(s["random"]["train"]), "test": P(s["random"]["test"])},
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)

    print(f"test: {len(s['test'])} rows, {len(doc['meta']['test_session_ids'])} sessions  {counts(rows, s['test'])}")
    print(f"  test gun clips per source: {counts(rows, [i for i in s['test'] if rows[i]['class'] != BG], 'source')}")
    for k, f in enumerate(s["folds"]):
        gs = sorted({rows[i]["session_id"] for i in f if rows[i]["class"] != BG})
        print(f"fold {k}: {len(f)} rows {counts(rows, f)}  gun sessions {gs}")
    d = s["device"]
    print(f"device split: {len(d['held_out_devices'])} held-out devices, train {len(d['train'])}, test {len(d['test'])} "
          f"{counts(rows, d['test'])}; groups on both sides: {d['groups_on_both_sides']}")
    print(f"random split: train {len(s['random']['train'])}, test {len(s['random']['test'])}")
    print(f"\nWrote {out}  (sha256 {sha256(out)[:16]}...). Record it in D23, then run check_splits.py.")
    return 0


def load_splits(path: str | Path, rows: list[dict]) -> dict:
    """splits.json -> lists of manifest rows. Refuses a splits.json made from another manifest."""
    path = Path(path).resolve()
    with open(path) as f:
        doc = json.load(f)
    root = path.parent if (path.parent / "manifest.csv").exists() else Path(rows[0]["path"]).parent
    while not (root / "manifest.csv").exists() and root != root.parent:
        root = root.parent
    if sha256(root / "manifest.csv") != doc["meta"]["manifest_sha256"]:
        raise ValueError("manifest.csv changed since splits.json was made; re-run splits.py (team decision)")
    by = dict(zip(rel(rows, root), rows))
    get = lambda ps: [by[p] for p in ps]  # noqa: E731
    test = get(doc["test"])
    folds_val = [get(f) for f in doc["cv"]["folds"]]
    test_set = {id(r) for r in test}
    dev = [r for r in rows if id(r) not in test_set]
    folds = []
    for v in folds_val:
        vs = {id(r) for r in v}
        folds.append({"train": [r for r in dev if id(r) not in vs], "val": v})
    return {"test": test, "dev": dev, "folds": folds,
            "device": {"train": get(doc["device"]["train"]), "test": get(doc["device"]["test"]),
                       "held_out_devices": doc["device"]["held_out_devices"]},
            "random": {"train": get(doc["random"]["train"]), "test": get(doc["random"]["test"])},
            "meta": doc["meta"]}


if __name__ == "__main__":
    sys.exit(main())
