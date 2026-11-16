#!/usr/bin/env python3
"""Review gate for splits.json (Step 1.6): exits 1 on any leak or count mismatch.

Checks
  1. splits.json was made from this manifest (sha256).
  2. Every clip is in exactly one place: excluded, the test set, or one CV fold.
  3. No session_id and no group appears in two folds, or in a fold and the test set.
  4. The test set holds every clip of c3gd_nj_farm_2 and edge-day2 (all classes), its random background
     sessions are background-only and 15 % of the candidates, and the gun counts are those of D23.
  5. Every firearm class is tested on sessions from at least two datasets (sources).
  6. Device split: no device on both sides; its test set lies inside the CV pool.
  7. Random split: inside the CV pool, stratified (every class on both sides).
  8. With --window-index: every gun window belongs to a clip of the splits; windows per fold are reported
     (a fold with no gun window of a class it holds clips of is flagged in the report).
.38 Special is reported separately: only two of its sessions (cadre, edge-day1) are left for CV.

Usage
  python sw/data/check_splits.py --manifest DATASET/manifest.csv --splits sw/data/splits.json \
      [--window-index results/step16/window_index.csv] [--report results/step16/splits_report.md]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import load_manifest  # noqa: E402
from splits import file_sha256, load_splits  # noqa: E402

D23_GUN_COUNTS = {"pistol_9mm": 1143, "rifle_556_223": 937, "revolver_38spl": 251, "shotgun_12ga": 216}
GUN_MIC = {"edge-quiet": "edge", "cadre-quiet": "cadre"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--window-index", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()

    fails: list[str] = []
    lines: list[str] = []

    def check(ok: bool, msg: str) -> None:
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {msg}")
        if not ok:
            fails.append(msg)

    s = load_splits(a.splits, manifest=a.manifest)          # raises on a missing file or another manifest
    raw = s.raw
    check(True, f"splits.json matches the manifest (sha256 {raw['manifest_sha256'][:16]}...)")
    root = a.manifest.resolve().parent
    rows = {str(Path(r["path"]).relative_to(root)): r for r in load_manifest(a.manifest)}
    guns = [c for c in raw["classes"] if c != "background"]

    places: dict[str, list[str]] = defaultdict(list)
    for p in s.excluded:
        places[p].append("excluded")
    for p in s.test:
        places[p].append("test")
    for k, f in enumerate(raw["cv"]["folds"]):
        for p in f["val_paths"]:
            places[p].append(f"fold{k}")
    unknown = set(places) - set(rows)
    check(not unknown, f"every path in splits.json is a manifest clip ({len(unknown)} unknown)")
    missing = set(rows) - set(places)
    multi = {p: v for p, v in places.items() if len(v) > 1}
    check(not missing and not multi,
          f"every clip is in exactly one of excluded / test / one fold ({len(missing)} missing, {len(multi)} in several)")
    check(all(rows[p]["sample_rate"] < 16000 for p in s.excluded if p in rows),
          f"only clips recorded below 16 kHz are excluded ({len(s.excluded)})")

    part = {p: v[0] for p, v in places.items() if len(v) == 1 and v[0] != "excluded" and p in rows}
    for key in ("session_id", "group"):
        where = defaultdict(set)
        for p, w in part.items():
            where[rows[p][key]].add(w)
        bad = {k: v for k, v in where.items() if len(v) > 1}
        check(not bad, f"no {key} in two folds or in a fold and the test set ({len(bad)} found"
                       f"{', e.g. ' + repr(next(iter(bad.items()))) if bad else ''})")

    fixed = set(raw["test"]["sessions_fixed"])
    want = {p for p, r in rows.items() if r["session_id"] in fixed and p not in s.excluded}
    check(want <= s.test, f"every clip of {sorted(fixed)} is in the test set ({len(want - s.test)} missing)")
    gun_sessions = {r["session_id"] for r in rows.values() if r["class"] != "background"}
    rnd = set(raw["test"]["background_sessions_random"])
    check(not rnd & gun_sessions, "the random test sessions hold no gun clip (so no session is split)")
    n_cand = raw["test"]["background_candidate_sessions"]
    check(abs(len(rnd) - raw["test"]["background_share"] * n_cand) <= 1,
          f"random background test sessions: {len(rnd)} of {n_cand} candidates "
          f"({100 * len(rnd) / n_cand:.1f} %, seed {raw['seeds']['background_test_sessions']})")
    other = {rows[p]["session_id"] for p in s.test} - fixed - rnd
    check(not other, f"the test set holds no other session ({sorted(other)[:5]})")

    test_counts = Counter(rows[p]["class"] for p in s.test)
    got = {c: test_counts[c] for c in D23_GUN_COUNTS}
    check(got == D23_GUN_COUNTS, f"D23 gun test counts {got} == expected {D23_GUN_COUNTS}")
    for c in guns:
        srcs = {rows[p]["source"] for p in s.test if rows[p]["class"] == c}
        sess = sorted({rows[p]["session_id"] for p in s.test if rows[p]["class"] == c})
        check(len(srcs) >= 2, f"{c} is tested on sessions {sess} from {len(srcs)} datasets {sorted(srcs)}")

    pool = s.cv_pool
    dev_train, dev_test = s.device_split()
    dkey = lambda r: f"{GUN_MIC.get(r['source'], r['source'])}:{r['device']}"  # noqa: E731
    d_tr = {dkey(rows[p]) for p in dev_train if rows[p]["device"]}
    d_te = {dkey(rows[p]) for p in dev_test if rows[p]["device"]}
    check(dev_test <= pool and not d_tr & d_te,
          f"device split: test inside the CV pool, {len(d_te)} held-out devices, none in training "
          f"({len(d_tr & d_te)} shared); {raw['device_split']['shared_groups']} shot groups recorded on devices "
          f"of both sides (same shot, different microphone: expected for a within-session device split)")
    for c in guns:
        n_te = sum(rows[p]["class"] == c for p in dev_test)
        check(n_te > 0, f"device split tests {c} ({n_te} clips)")
    r_train, r_test = s.random_split()
    check(r_test <= pool and all(any(rows[p]["class"] == c for p in r_test) and
                                 any(rows[p]["class"] == c for p in r_train) for c in raw["classes"]),
          f"random split: {len(r_train)} train / {len(r_test)} test clips, every class on both sides")

    # Counts tables.
    tables = ["", "## Clips per class", "", "| set | " + " | ".join(raw["classes"]) + " | sessions |",
              "|---|" + "---:|" * (len(raw["classes"]) + 1)]
    sets = [("test", s.test)] + [(f"fold {k}", s.fold(k)[1]) for k in range(s.n_folds)] + \
           [("device test", dev_test), ("random test", r_test), ("excluded", set(s.excluded))]
    for name, ps in sets:
        cnt = Counter(rows[p]["class"] for p in ps)
        tables.append(f"| {name} | " + " | ".join(str(cnt[c]) for c in raw["classes"]) +
                      f" | {len({rows[p]['session_id'] for p in ps})} |")
    tables += ["", "Gun sessions per fold (validation side): " +
               "; ".join(f"fold {k}: {', '.join(x for x in f['val_sessions'] if x in gun_sessions)}"
                         for k, f in enumerate(raw["cv"]["folds"])), ""]
    tables += ["## .38 Special (reported separately)", "",
               "| fold | .38 train clips | .38 validation clips | validation sessions |", "|---|---:|---:|---|"]
    for k in range(s.n_folds):
        tr, va = s.fold(k)
        n_tr = sum(rows[p]["class"] == "revolver_38spl" for p in tr)
        va38 = [p for p in va if rows[p]["class"] == "revolver_38spl"]
        tables.append(f"| {k} | {n_tr} | {len(va38)} | {', '.join(sorted({rows[p]['session_id'] for p in va38})) or '-'} |")

    if a.window_index:
        with open(a.window_index, newline="") as f:
            win = list(csv.DictReader(f))
        place_of = {p: w for p, w in part.items()}
        orphan = [w["path"] for w in win if w["path"] not in place_of]
        check(not orphan, f"every gun window ({len(win)}) belongs to a clip of the test set or a fold "
                          f"({len(orphan)} do not)")
        tables += ["", "## Gun windows per set (window_index.csv)", "",
                   "| set | " + " | ".join(guns) + " |", "|---|" + "---:|" * len(guns)]
        per = defaultdict(Counter)
        for w in win:
            per[place_of.get(w["path"], "?")][w["class"]] += 1
        empty = []
        for name in ["test"] + [f"fold{k}" for k in range(s.n_folds)]:
            tables.append(f"| {name} | " + " | ".join(str(per[name][c]) for c in guns) + " |")
            clips_here = Counter(rows[p]["class"] for p, w in part.items() if w == name)
            empty += [f"{name}/{c}" for c in guns if clips_here[c] and not per[name][c]]
        if empty:
            tables += ["", f"Note: these sets hold clips of a class but no gun window of it (every clip dropped by the "
                           f"D22 placement): {', '.join(empty)}."]

    seeds = raw["seeds"]
    head = ["# Split check (Step 1.6, Decision D23)", "",
            f"`sw/data/check_splits.py` on `{a.splits.name}` (sha256 `{file_sha256(a.splits)}`), seeds {seeds}.", ""]
    report = "\n".join(head + lines + tables) + "\n"
    print(report)
    if a.report:
        a.report.write_text(report)
    if fails:
        print(f"SPLIT CHECK FAILED ({len(fails)})", file=sys.stderr)
        return 1
    print("SPLIT CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
