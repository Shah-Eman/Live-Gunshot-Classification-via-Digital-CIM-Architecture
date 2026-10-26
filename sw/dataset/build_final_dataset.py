#!/usr/bin/env python3
"""Build the final 5-class dataset in its own folder (Dataset-Final/).

Classes
  pistol_9mm       9x19 mm pistols                 edge Glock 17, Cadre Glock 19 / Sig P225, C3GD 9 mm pistols
  rifle_556_223    5.56x45 / .223 AR-type rifles   edge Ruger AR-556, Cadre M16A1, C3GD ARs
  shotgun_12ga     12 gauge shotguns               edge + C3GD Remington 870, C3GD Benelli M4, Kel-Tec KS7
  revolver_38spl   .38 Special revolvers           edge S&W .38, Cadre S&W 10-8, C3GD Taurus 66 / 856
  background       no gunshot                      MAD (filtered), UrbanSound8K (all but gun_shot), ESC-50

Gun clips are taken from Dataset/ after build_dataset.py and filter_dataset.py
have run, and only from the three field datasets with known sessions (edge,
C3GD, Cadre). original_dataset copies and Gun_Shot_DataSet clips are left out.
Excluded on purpose: the S&W M&P FPC (a 9 mm carbine) and the bolt-action
Savage .223.

UrbanSound8K and ESC-50 clips are measured with audio_metrics.analyze and
must pass the background checks of filter_dataset.py, except the "impulsive"
check: their labels are trustworthy and the impulsive sounds (dog barks, door
knocks, fireworks, car horns) are exactly the hard negatives the model needs.
UrbanSound8K's gun_shot class is dropped.

Quiet stretches: the audio before the first shot of every edge and Cadre gun
clip is cut out as extra background (sources ``edge-quiet`` / ``cadre-quiet``),
so some background is recorded on the same microphones as the guns. It keeps
the gun clip's session_id and group, so it always lands on the same side of a
split as its shot. A stretch ends 50 ms before the first shot (edge: the
dataset's labelled shot time, used only when the clip's peak matches a label;
Cadre: the first sample within 30 dB of the peak, used only when that onset is
within 100 ms of the peak), must be at least 0.5 s long, and must pass every
background check including "impulsive" (other shooters on the range). C3GD
clips start 0.1 s before the shot, so they give no stretches.

Output (hard links, no extra disk space; nothing outside Dataset-Final/ is
touched)
  Dataset-Final/audio/<class>/<file>
  Dataset-Final/manifest.csv       one row per clip: class, source, gun, caliber,
                                   session_id, device, group (leakage id), ...
  Dataset-Final/class_summary.csv
  Dataset-Final/rejected.csv       UrbanSound8K / ESC-50 clips that failed a check

Splitting: keep each ``group`` on one side; for session-disjoint evaluation
hold out whole ``session_id`` values.

The build happens in Dataset-Final.partial/ and replaces Dataset-Final/ only
when it succeeds. A rebuild that would drop a source the existing folder still
holds is refused, because that folder may hold the only copy of those clips.

Usage:
    python build_final_dataset.py [--us8k PATH] [--esc50 PATH] [--force]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_metrics import analyze

C3GD_CLASS = {"9x19mm": "pistol_9mm", "5.56x45mm": "rifle_556_223", ".223 Remington": "rifle_556_223",
              "12 Ga": "shotgun_12ga", ".38 Special": "revolver_38spl"}
C3GD_DROP = {"sw-mp-fpc-compliant", "savage-223"}
GUN_CLASS = {  # edge and Cadre gun folders
    "glock-17": "pistol_9mm", "glock-19": "pistol_9mm", "sig-p225": "pistol_9mm",
    "ruger-ar-556": "rifle_556_223", "m16": "rifle_556_223",
    "remington-870": "shotgun_12ga",
    "sw-38-caliber": "revolver_38spl", "sw-10-8-38spl-revolver": "revolver_38spl",
}
CALIBER = {"pistol_9mm": "9x19mm", "rifle_556_223": "5.56x45mm/.223", "shotgun_12ga": "12 Ga",
           "revolver_38spl": ".38 Special"}
BACKGROUND = "background"

# Background checks: filter_dataset.py's background rules minus "impulsive".
BG_MIN_DURATION = 0.5
BG_MIN_PEAK_DBFS = -50.0
MAX_CLIP_FRAC = 0.05
MIN_SR = 8000
BG_MAX_KURTOSIS = 6.0         # quiet stretches only: same impulsive rule as MAD
BG_MAX_EVENTS = 0

QUIET_GUARD_S = 0.05          # stop this long before the first shot
QUIET_MIN_S = 0.5             # the model's input window
LABEL_MATCH_S = 0.05          # edge: clip peak must be this close to a labelled shot
ONSET_DB = -30.0              # Cadre: onset = first sample within 30 dB of the peak
ONSET_MAX_LEAD_S = 0.10       # Cadre: and that onset must be this close to the peak


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def edge_meta(dataset: Path) -> dict[str, dict]:
    """edge filename stem -> metadata row (device_name, gunshot_location_in_seconds, ...)."""
    f = dataset / "edge-collected-gunshot-audio" / "gunshot-audio-all-metadata.csv"
    return {r["filename"]: {k.strip(): v for k, v in r.items() if k} for r in read_csv(f)}


def shot_times(s: str) -> list[float]:
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s)]


def gun_and_mad_rows(dataset: Path) -> list[dict]:
    kept = {r["file"] for r in read_csv(dataset / "filter_report.csv") if r["verdict"] == "kept"}
    c3gd = {r["path"]: r for r in read_csv(dataset / "Sorted-C3GD" / "c3gd_sorted.csv")}
    edge = edge_meta(dataset)
    rows = []
    for r in read_csv(dataset / "manifest.csv"):
        if r["status"] != "kept" or r["combined_name"] not in kept:
            continue
        src, gun, rel = r["source"], r["gun"], r["orig_relpath"]
        row = {"src_path": dataset / "Combined" / r["combined_name"], "name": r["combined_name"],
               "source": src, "gun": gun, "group": r["group"], "device": r["device"],
               "sub_label": "", "orig_path": f"{src}:{rel}"}
        if gun == "background-nogun":
            row.update(cls=BACKGROUND, session_id=f"mad-{r['group'].removeprefix('mad-')}",
                       sub_label=f"MAD label {r['source_label']}")
        elif src == "c3gd":
            m = c3gd[rel]
            cls = C3GD_CLASS.get(m["caliber"])
            if cls is None or gun in C3GD_DROP:
                continue
            row.update(cls=cls, session_id=m["session_id"], group=f"c3gd-{m['file_id']}")
        elif src in ("edge", "cadre") and gun in GUN_CLASS:
            cls = GUN_CLASS[gun]
            if src == "edge":
                day = re.search(r"-s(\d)$", r["session"]).group(1)
                meta = edge.get(Path(rel).stem, {})
                row.update(session_id=f"edge-day{day}", device=meta.get("device_name", r["device"]),
                           shots=shot_times(meta.get("gunshot_location_in_seconds", "")))
            else:
                row.update(session_id="cadre")
            row["cls"] = cls
        else:
            continue
        rows.append(row)
    return rows


def us8k_rows(root: Path) -> list[dict]:
    rows = []
    for r in read_csv(root / "UrbanSound8K.csv"):
        if r["class"] == "gun_shot":
            continue
        p = root / f"fold{r['fold']}" / r["slice_file_name"]
        rows.append({"src_path": p, "name": f"us8k__fold{r['fold']}__{r['slice_file_name']}", "source": "us8k",
                     "cls": BACKGROUND, "gun": "", "session_id": f"us8k-{r['fsID']}", "group": f"us8k-{r['fsID']}",
                     "device": "", "sub_label": r["class"], "orig_path": f"us8k:fold{r['fold']}/{r['slice_file_name']}"})
    return rows


def esc50_rows(root: Path) -> list[dict]:
    meta = next(root.rglob("esc50.csv"), None)
    if meta is None:
        sys.exit(f"esc50.csv not found under {root}")
    entries = read_csv(meta)
    # Official layout: meta/esc50.csv + audio/. Kaggle copies nest the originals in
    # audio/audio/ next to resampled 16000/ and 44100/ folders; take the originals.
    hits = sorted(root.rglob(entries[0]["filename"]), key=lambda p: (p.parent.name.isdigit(), len(p.parts)))
    if not hits:
        sys.exit(f"ESC-50 audio not found under {root}")
    audio = hits[0].parent
    rows = []
    for r in entries:
        rows.append({"src_path": audio / r["filename"], "name": f"esc50__{r['filename']}", "source": "esc50",
                     "cls": BACKGROUND, "gun": "", "session_id": f"esc50-{r['src_file']}",
                     "group": f"esc50-{r['src_file']}", "device": "", "sub_label": r["category"],
                     "orig_path": f"esc50:{r['filename']}"})
    return rows


def background_fails(m: dict) -> list[str]:
    if not m.get("decode_ok") or m.get("error") or m.get("finite") is False or m.get("wav_truncated") is True:
        return ["corrupted"]
    fails = []
    if m["duration_s"] < BG_MIN_DURATION:
        fails.append("too_short")
    if m["peak_dbfs"] < BG_MIN_PEAK_DBFS:
        fails.append("low_level")
    if m["clip_frac_active"] > MAX_CLIP_FRAC:
        fails.append("heavy_clipping")
    if m["sr"] < MIN_SR:
        fails.append("low_sample_rate")
    if m.get("dead_channel"):
        fails.append("dead_channel")
    return fails


def cut_quiet(job: tuple) -> tuple[str, dict]:
    """Write the pre-shot stretch of one gun clip to dst. Returns (outcome, metrics).

    outcome: "kept", "no_stretch" (nothing usable before the shot) or the
    "|"-joined background checks the written stretch failed (it is then removed).
    """
    src, dst, shots = job
    x, sr = sf.read(src, always_2d=True)
    a = np.abs(x.mean(axis=1))
    peak = int(np.argmax(a))
    if shots:
        if min(abs(peak / sr - t) for t in shots) > LABEL_MATCH_S:
            return "no_stretch", {}
        end = min(shots) - QUIET_GUARD_S
    else:
        onset = int(np.argmax(a >= a[peak] * 10 ** (ONSET_DB / 20)))
        if (peak - onset) / sr > ONSET_MAX_LEAD_S:
            return "no_stretch", {}
        end = onset / sr - QUIET_GUARD_S
    if end < QUIET_MIN_S:
        return "no_stretch", {}
    subtype = sf.info(src).subtype
    sf.write(dst, x[:int(end * sr)], sr, subtype=subtype if subtype.startswith("PCM") else "PCM_16")
    m = analyze(dst)
    fails = background_fails(m)
    if m.get("kurtosis", 0) > BG_MAX_KURTOSIS or m.get("n_events", 0) > BG_MAX_EVENTS:
        fails.append("impulsive")
    if fails:
        os.remove(dst)
        return "|".join(fails), m
    return "kept", m


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=here.parents[3] / "Dataset")
    ap.add_argument("--out", type=Path, default=here.parents[3] / "Dataset-Final")
    ap.add_argument("--us8k", type=Path, help="folder holding UrbanSound8K.csv and fold1..fold10")
    ap.add_argument("--esc50", type=Path, help="ESC-50 folder (holding esc50.csv and the audio)")
    ap.add_argument("--force", action="store_true", help="replace an existing output folder")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()

    final = a.out.resolve()
    if final.exists() and not a.force:
        print(f"{final} exists; use --force to rebuild it", file=sys.stderr)
        return 1

    # Read and check every input before touching any output: an existing
    # Dataset-Final may hold the only remaining copy of a source's clips.
    rows = gun_and_mad_rows(a.dataset.resolve())
    extra = []
    for name, path, reader in (("UrbanSound8K", a.us8k, us8k_rows), ("ESC-50", a.esc50, esc50_rows)):
        if path is None:
            print(f"WARNING: no {name} path given; background has no {name} clips", file=sys.stderr)
            continue
        if not path.expanduser().is_dir():
            print(f"{name} folder not found: {path}", file=sys.stderr)
            return 1
        extra += reader(path.expanduser())

    missing = [r for r in rows + extra if not Path(r["src_path"]).is_file()]
    if missing:
        print(f"{len(missing)} input files missing, e.g. {missing[0]['src_path']}", file=sys.stderr)
        return 1

    if (final / "manifest.csv").exists():
        had = {r["source"] for r in read_csv(final / "manifest.csv")} - {"edge-quiet", "cadre-quiet"}
        lost = had - {r["source"] for r in rows + extra}
        if lost:
            print(f"Refusing: the rebuild would drop {sorted(lost)}, which {final.name} still holds "
                  f"(possibly the only copy). Pass their input folders.", file=sys.stderr)
            return 1

    # Build next to the final folder; it replaces the old one only on success.
    out = final.with_name(final.name + ".partial")
    if out.exists():
        shutil.rmtree(out)

    with Pool(a.jobs) as pool:
        metrics = pool.map(analyze, [r["src_path"] for r in extra], chunksize=16)
    rejected = []
    for r, m in zip(extra, metrics):
        fails = background_fails(m)
        r["duration_s"], r["sr"] = m.get("duration_s", ""), m.get("sr", "")
        if fails:
            rejected.append({"orig_path": r["orig_path"], "sub_label": r["sub_label"], "reasons": "|".join(fails)})
        else:
            rows.append(r)

    out.mkdir(parents=True)
    (out / "audio" / BACKGROUND).mkdir(parents=True)

    # Quiet stretches before the shot, from edge and Cadre gun clips.
    guns = [r for r in rows if r["source"] in ("edge", "cadre")]
    jobs = [(str(r["src_path"]), str(out / "audio" / BACKGROUND / f"{r['source']}-quiet__{r['name']}"),
             r.get("shots", [])) for r in guns]
    with Pool(a.jobs) as pool:
        results = pool.map(cut_quiet, jobs, chunksize=8)
    quiet = Counter()
    for r, (src, dst, _), (outcome, m) in zip(guns, jobs, results):
        quiet[(r["source"], outcome.split("|")[0])] += 1
        if outcome == "no_stretch":
            continue
        q = {"orig_path": f"{r['orig_path']} (before shot)", "sub_label": f"quiet before {r['gun']} shot"}
        if outcome != "kept":
            rejected.append({**q, "reasons": outcome})
            continue
        rows.append({**q, "src_path": dst, "written": True, "name": Path(dst).name, "source": f"{r['source']}-quiet",
                     "cls": BACKGROUND, "gun": "", "session_id": r["session_id"], "group": r["group"],
                     "device": r["device"]})
    print("Quiet stretches:", dict(sorted(quiet.items())))

    cols = ["file", "class", "source", "gun", "caliber", "session_id", "device", "group", "sub_label", "orig_path"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["cls"], r["source"], r["name"])):
            rel = f"audio/{r['cls']}/{r['name']}"
            if not r.get("written"):
                link(Path(r["src_path"]), out / rel)
            w.writerow({"file": rel, "class": r["cls"], "source": r["source"], "gun": r["gun"],
                        "caliber": CALIBER.get(r["cls"], ""), "session_id": r["session_id"], "device": r["device"],
                        "group": r["group"], "sub_label": r["sub_label"], "orig_path": r["orig_path"]})
    with open(out / "rejected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["orig_path", "sub_label", "reasons"])
        w.writeheader()
        w.writerows(rejected)

    per = defaultdict(lambda: {"clips": 0, "sessions": set(), "groups": set(), "sources": Counter(), "guns": set()})
    for r in rows:
        s = per[r["cls"]]
        s["clips"] += 1
        s["sessions"].add(r["session_id"])
        s["groups"].add(r["group"])
        s["sources"][r["source"]] += 1
        if r["gun"]:
            s["guns"].add(r["gun"])
    with open(out / "class_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "clips", "sessions", "groups", "guns", "sources"])
        for c, s in sorted(per.items()):
            src = "; ".join(f"{k} {v}" for k, v in s["sources"].most_common())
            w.writerow([c, s["clips"], len(s["sessions"]), len(s["groups"]), len(s["guns"]), src])
            print(f"{c:15s} {s['clips']:6d} clips  {len(s['sessions']):4d} sessions  {len(s['groups']):5d} groups  {src}")
    print(f"\nBackground clips rejected: {len(rejected)} "
          f"{dict(Counter(x['reasons'].split('|')[0] for x in rejected))}")

    if final.exists():
        old = final.with_name(final.name + ".old")
        final.rename(old)
        out.rename(final)
        shutil.rmtree(old)
    else:
        out.rename(final)
    print(f"Wrote {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
