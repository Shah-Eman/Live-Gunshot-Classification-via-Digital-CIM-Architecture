#!/usr/bin/env python3
"""Quality-filter the per-gun dataset (Dataset/Sorted -> Dataset/Filtered).

Every file in Sorted/<gun>/ is measured (see audio_metrics.py) and then passed
through the checks below. Files that fail ANY check go to
Rejected/<reason>/<gun>/ (so they can be inspected); the rest go to
Filtered/<gun>/. Nothing is modified: outputs are hard links (or copies with
--copy). The per-file verdict and every measurement are written to
Dataset/filter_report.csv.

Checks (first failing one is the "reason"; all failing ones are listed in the report)

  corrupted        cannot be decoded, zero frames, NaN/Inf samples, WAV header
                   promises more data than the file holds, or the decoder
                   delivered <99% of the frames the header declared
  too_short        shorter than --min-duration (default 0.10 s)
  low_level        loudest peak below --min-peak-dbfs (default -40 dBFS), after
                   removing DC offset. Includes silent files.
  no_gunshot       no discernible event: the loudest 4 ms frame is less than
                   --min-contrast-db (default 6 dB) above the 10th-percentile
                   frame, i.e. steady noise / tone / saturated garbage
  low_snr          quiet AND poorly separated from background:
                   contrast < --weak-contrast-db (10 dB) and peak below
                   --weak-peak-dbfs (-20 dBFS). Loud bursts are exempt.
  heavy_clipping   more than --max-clip-frac (default 5 %) of the audible samples
                   (those above about -26 dBFS) sit at full scale
  low_sample_rate  sample rate below --min-sr (default 8 kHz)
  dead_channel     one channel of a multi-channel file is silent while the
                   other is not (broken stereo capture)

Background / no-gun class (background-nogun, from the MAD dataset)
-----------------------------------------------------------------
This class must NOT contain gunfire, so the gun rules are replaced by:

  corrupted, too_short (--bg-min-duration 0.5 s), heavy_clipping,
  low_sample_rate, dead_channel   same as above
  low_level          peak below --bg-min-peak-dbfs (default -50 dBFS): silent
  excluded_label     MAD class listed in --bg-exclude-labels (default "1": its
                     source videos are live-fire drills, so it is mostly gunfire)
  impulsive          any impulsive event: kurtosis above --bg-max-kurtosis (6)
                     or more than --bg-max-events (0) transient events. MAD
                     classes 0-3 mix speech, gunfire and explosions; only their
                     transient-free clips survive. Classes 4-6 (vehicles,
                     helicopters, jets) are steady noise and mostly pass.

Measurements are cached in Dataset/filter_metrics.csv; re-run with
--reuse-metrics to retune thresholds in a second without re-reading audio.

Usage:
    python filter_dataset.py [--dataset-root PATH] [--reuse-metrics] [--copy] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from audio_metrics import analyze

BACKGROUND = "background-nogun"
REASON_ORDER = ["corrupted", "too_short", "low_level", "no_gunshot", "low_snr",
                "heavy_clipping", "low_sample_rate", "dead_channel",
                "excluded_label", "impulsive"]


def evaluate(m: dict, a: argparse.Namespace, gun: str = "", source_label: str = "") -> list[str]:
    """Return the list of failed checks for one file's metrics (empty = accept)."""
    if not m.get("decode_ok") or m.get("error"):
        return ["corrupted"]
    if m.get("finite") is False or m.get("wav_truncated") is True or m.get("missing_frac", 0) > 0.01:
        return ["corrupted"]

    if gun == BACKGROUND:
        return evaluate_background(m, a, source_label)

    fails = []
    if m["duration_s"] < a.min_duration:
        fails.append("too_short")
    if m["peak_dbfs"] < a.min_peak_dbfs:
        fails.append("low_level")
    contrast = m["peak_to_p10_db"]
    if contrast < a.min_contrast_db:
        fails.append("no_gunshot")
    elif contrast < a.weak_contrast_db and m["peak_dbfs"] < a.weak_peak_dbfs:
        fails.append("low_snr")
    if m["clip_frac_active"] > a.max_clip_frac:
        fails.append("heavy_clipping")
    if m["sr"] < a.min_sr:
        fails.append("low_sample_rate")
    if m.get("dead_channel"):
        fails.append("dead_channel")
    return sorted(fails, key=REASON_ORDER.index)


def evaluate_background(m: dict, a: argparse.Namespace, source_label: str) -> list[str]:
    fails = []
    if m["duration_s"] < a.bg_min_duration:
        fails.append("too_short")
    if m["peak_dbfs"] < a.bg_min_peak_dbfs:
        fails.append("low_level")
    if m["clip_frac_active"] > a.max_clip_frac:
        fails.append("heavy_clipping")
    if m["sr"] < a.min_sr:
        fails.append("low_sample_rate")
    if m.get("dead_channel"):
        fails.append("dead_channel")
    if source_label and source_label in a.bg_exclude_labels.split(","):
        fails.append("excluded_label")
    if m["kurtosis"] > a.bg_max_kurtosis or m["n_events"] > a.bg_max_events:
        fails.append("impulsive")
    return sorted(fails, key=REASON_ORDER.index)


def place(src: Path, dst: Path, force_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force_copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def safe_reset(d: Path, root: Path) -> None:
    if d.parent.resolve() != root.resolve() or d.name not in {"Filtered", "Rejected"}:
        raise RuntimeError(f"refusing to delete {d}")
    if d.exists():
        shutil.rmtree(d)


def fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4g}"
    return v


def main() -> int:
    default_root = Path(__file__).resolve().parents[3] / "Dataset"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, default=default_root)
    ap.add_argument("--reuse-metrics", action="store_true", help="skip audio analysis, reuse filter_metrics.csv")
    ap.add_argument("--copy", action="store_true", help="copy files instead of hard-linking")
    ap.add_argument("--dry-run", action="store_true", help="evaluate and report, write no folders")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--min-duration", type=float, default=0.10, help="seconds")
    ap.add_argument("--min-peak-dbfs", type=float, default=-40.0)
    ap.add_argument("--min-contrast-db", type=float, default=6.0)
    ap.add_argument("--weak-contrast-db", type=float, default=10.0)
    ap.add_argument("--weak-peak-dbfs", type=float, default=-20.0)
    ap.add_argument("--max-clip-frac", type=float, default=0.05)
    ap.add_argument("--min-sr", type=int, default=8000)
    ap.add_argument("--bg-min-duration", type=float, default=0.5)
    ap.add_argument("--bg-min-peak-dbfs", type=float, default=-50.0)
    ap.add_argument("--bg-exclude-labels", default="1", help="comma-separated MAD labels to drop")
    ap.add_argument("--bg-max-kurtosis", type=float, default=6.0)
    ap.add_argument("--bg-max-events", type=int, default=0)
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    sorted_dir = root / "Sorted"
    if not sorted_dir.is_dir():
        print(f"{sorted_dir} not found - run build_dataset.py first.", file=sys.stderr)
        return 1

    files = sorted(p for p in sorted_dir.glob("*/*") if p.is_file())
    metrics_csv = root / "filter_metrics.csv"

    if args.reuse_metrics and metrics_csv.exists():
        with open(metrics_csv, newline="") as f:
            cached = {r["file"]: r for r in csv.DictReader(f)}
        rows = []
        for p in files:
            r = cached.get(p.name)
            if r is None:
                print(f"{p.name} missing from cache; re-run without --reuse-metrics", file=sys.stderr)
                return 1
            rows.append(parse_cached(r))
        print(f"Reused cached metrics for {len(rows)} files.")
    else:
        t0 = time.time()
        with Pool(args.jobs) as pool:
            rows = pool.map(analyze, files, chunksize=16)
        print(f"Analysed {len(rows)} files in {time.time() - t0:.0f}s.")
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "file", k))
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: fmt(r.get(k, "")) for k in keys})

    labels: dict[str, str] = {}
    sess_dev: dict[str, tuple[str, str, str]] = {}   # file -> (source, session, device)
    manifest = root / "manifest.csv"
    if manifest.exists():
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] == "kept":
                    labels[r["combined_name"]] = r.get("source_label", "")
                    sess_dev[r["combined_name"]] = (r["source"], r.get("session", ""), r.get("device", ""))

    verdicts = []
    for p, m in zip(files, rows):
        gun = p.parent.name
        m["source_label"] = labels.get(p.name, "")
        fails = evaluate(m, args, gun, m["source_label"])
        verdicts.append((p, gun, m, fails))

    per_gun_total, per_gun_kept = Counter(), Counter()
    reasons = Counter()
    multi = Counter()
    for _, gun, _, fails in verdicts:
        per_gun_total[gun] += 1
        if not fails:
            per_gun_kept[gun] += 1
        else:
            reasons[fails[0]] += 1
            for fl in fails:
                multi[fl] += 1

    total, kept = len(verdicts), sum(per_gun_kept.values())
    print(f"\nKept {kept} / {total} ({100 * kept / total:.1f}%)   rejected {total - kept}")
    print("\nRejected by primary reason (files failing several checks counted once):")
    for r in REASON_ORDER:
        if reasons[r]:
            print(f"  {reasons[r]:5d}  {r:16s} (fails this check in total: {multi[r]})")
    ranking = sorted(per_gun_total, key=lambda g: (-per_gun_kept[g], g))
    print("\nClasses ranked by amount of filtered (kept) data:")
    print(f"  {'#':>3} {'kept':>5} / {'total':<5}  class")
    kept_sessions: dict[str, set] = defaultdict(set)
    kept_devices: dict[str, set] = defaultdict(set)
    for p, gun, _, fails in verdicts:
        if not fails:
            _, se, de = sess_dev.get(p.name, ("", "", ""))
            if se:
                kept_sessions[gun].add(se)
            if de:
                kept_devices[gun].add(de)
    with open(root / "class_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "class", "kind", "kept", "rejected", "total", "pct_kept",
                    "sessions_known", "devices_known"])
        for i, gun in enumerate(ranking, 1):
            kind = "background" if gun == BACKGROUND else (
                "unidentified" if gun.startswith("_") or gun.endswith("-unspecified") else "gun")
            k, t = per_gun_kept[gun], per_gun_total[gun]
            print(f"  {i:3d} {k:5d} / {t:<5d}  {gun}")
            w.writerow([i, gun, kind, k, t - k, t, f"{100 * k / t:.1f}",
                        len(kept_sessions[gun]), len(kept_devices[gun])])

    with open(root / "filter_report.csv", "w", newline="") as f:
        cols = ["file", "gun", "verdict", "reasons", "source_label", "duration_s", "sr", "channels", "peak_dbfs",
                "rms_dbfs", "peak_to_p10_db", "peak_to_median_db", "clip_frac", "clip_frac_active", "dc_offset",
                "attack_ms", "n_events", "rolloff95_hz", "error"]
        w = csv.writer(f)
        w.writerow(cols)
        for p, gun, m, fails in verdicts:
            w.writerow([p.name, gun, "rejected" if fails else "kept", "|".join(fails)]
                       + [fmt(m.get(c, "")) for c in cols[4:]])
    print(f"\nWrote {root / 'filter_report.csv'}, {root / 'class_summary.csv'} and {metrics_csv}")

    if args.dry_run:
        print("--dry-run: Filtered/ and Rejected/ not written.")
        return 0

    filtered, rejected = root / "Filtered", root / "Rejected"
    safe_reset(filtered, root)
    safe_reset(rejected, root)
    for p, gun, _, fails in verdicts:
        dst = (filtered / gun / p.name) if not fails else (rejected / fails[0] / gun / p.name)
        place(p, dst, args.copy)
    print(f"Wrote {filtered} and {rejected}")
    return 0


NUMERIC_BOOL = {"decode_ok", "finite", "dead_channel"}


def parse_cached(r: dict) -> dict:
    """Convert a CSV row back to the types analyze() produces."""
    out: dict = {}
    for k, v in r.items():
        if v == "":
            continue
        if k in ("file", "error", "subtype", "fmt"):
            out[k] = v
        elif k in NUMERIC_BOOL or k == "wav_truncated":
            out[k] = v in ("True", "true", "1")
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    out.setdefault("decode_ok", False)
    return out


if __name__ == "__main__":
    sys.exit(main())
