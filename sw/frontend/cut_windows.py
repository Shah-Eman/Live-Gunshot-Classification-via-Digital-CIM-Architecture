#!/usr/bin/env python3
"""Cut every gun window with the detector and measure the detector's recall (Step 1.6, items 1-3).

Streams every gun clip of the manifest (edge, C3GD, Cadre) through
audio.load_clip -> detector.detect -> windows.process_gun_clip, one clip per
worker task. Nothing decoded is written to disk except the final window array.

Outputs
  <report-dir>/window_index.csv        one row per gun window (by reference: clip path, trigger, start, ...)
  <report-dir>/detector_summary.md     recall per source, per class, per session (+ D22 drop counts)
  <report-dir>/detector_by_*.csv       the same tables as CSV
  <report-dir>/detector_run.json       config, seeds, input hashes, totals
  <out>/detector_clips.csv             per-clip detector statistics
  <out>/detector_refs.csv              per-reference outcome (matched / inside_other_window / missed / ...)
  <out>/gun_windows.npz                int16 windows [N, window_samples], raw (before the gain), plus
                                       gain_shift, trigger_sample, window_start; row i = window_index row i

Exit status 1 if any recall gate (per source, per class, per session) is below
the --gate (0.95). Everything is written first, so the failure can be studied.

Usage
  python sw/frontend/cut_windows.py --manifest DATASET/manifest.csv \
      --edge-labels .../gunshot-audio-all-metadata.csv --out OUTDIR --report-dir results/step16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from config import FRONTEND  # noqa: E402
from detector import register_widths  # noqa: E402
from manifest import load_manifest  # noqa: E402
from windows import GUN_SOURCES, process_gun_clip, read_edge_labels  # noqa: E402

INDEX_COLS = ["path", "class", "session_id", "group", "source", "device", "trigger_sample", "window_start",
              "ref_sample", "trigger_minus_ref", "gain_shift", "verified", "flags"]
CLIP_COLS = ["path", "class", "session_id", "group", "source", "device", "n_samples", "audio_lo", "audio_hi", "verify", "triggers",
             "preempted", "refs", "matched", "inside_other_window", "missed", "at_clip_start", "beyond_clip",
             "unmatched_triggers", "unverified_triggers", "windows", "dropped_starts_before_audio",
             "dropped_short_after_start"]
SUM_KEYS = ["clips", "unverifiable_clips", "refs", "matched", "inside_other_window", "missed", "at_clip_start",
            "beyond_clip", "triggers", "unmatched_triggers", "unverified_triggers", "preempted", "windows",
            "dropped_starts_before_audio", "dropped_short_after_start"]

_LABELS: dict = {}


def _init(labels_path: str) -> None:
    global _LABELS
    _LABELS = read_edge_labels(labels_path)


def _work(row: dict) -> dict:
    return process_gun_clip(row, _LABELS, FRONTEND)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def summarise(clips: list[dict], key) -> dict:
    out = defaultdict(lambda: dict.fromkeys(SUM_KEYS, 0))
    for c in clips:
        s = out[key(c)]
        s["clips"] += 1
        s["unverifiable_clips"] += c["verify"] != "ok"
        for k in SUM_KEYS[2:]:
            s[k] += c[k]
    for s in out.values():
        s["recall"] = s["matched"] / s["refs"] if s["refs"] else float("nan")
    return dict(sorted(out.items()))


def write_table(path: Path, table: dict, name: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([name, "recall"] + SUM_KEYS)
        for k, s in table.items():
            w.writerow([k if isinstance(k, str) else "/".join(k), f"{s['recall']:.4f}"] + [s[c] for c in SUM_KEYS])


def md_table(table: dict, name: str) -> list[str]:
    lines = [f"| {name} | clips | known shots | matched | recall | inside other window | truly missed | "
             f"unmatched triggers | preempted | windows | dropped: starts before audio | dropped: short after start "
             f"| unverifiable clips |",
             "|---|" + "---:|" * 12]
    for k, s in table.items():
        k = k if isinstance(k, str) else " / ".join(k)
        lines.append(f"| {k} | {s['clips']} | {s['refs']} | {s['matched']} | {100 * s['recall']:.1f} % | "
                     f"{s['inside_other_window']} | {s['missed']} | {s['unmatched_triggers']} | {s['preempted']} | "
                     f"{s['windows']} | {s['dropped_starts_before_audio']} | {s['dropped_short_after_start']} | "
                     f"{s['unverifiable_clips']} |")
    return lines


def whatif(refs: list[dict], clips: list[dict], sr: int, real: int) -> list[str]:
    """Windows kept under other pre-trigger lengths (same 0.40 s of real audio), for D22."""
    bounds = {c["path"]: (c["audio_lo"], c["audio_hi"]) for c in clips}
    matched = [r for r in refs if r["outcome"] == "matched"]
    lines = ["| pre-trigger | windows kept | dropped: starts before audio | dropped: short after start |",
             "|---|---:|---:|---:|"]
    for pre_s in (0.03, 0.04, 0.05):
        pre = int(round(pre_s * sr))
        before = sum(1 for r in matched if r["trigger_sample"] - pre < bounds[r["path"]][0])
        after = sum(1 for r in matched if r["trigger_sample"] - pre >= bounds[r["path"]][0] and
                    r["trigger_sample"] - pre + real > bounds[r["path"]][1])
        lines.append(f"| {pre_s:.2f} s | {len(matched) - before - after} | {before} | {after} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--edge-labels", type=Path, required=True,
                    help="Kabealo gunshot-audio-all-metadata.csv (gunshot_location_in_seconds)")
    ap.add_argument("--out", type=Path, required=True, help="output folder outside the repo (npz, per-clip CSVs)")
    ap.add_argument("--report-dir", type=Path, required=True, help="repo folder for the index and summaries")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--gate", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0, help="first N gun clips only (debugging; never for D22)")
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    a = ap.parse_args()

    cfg = FRONTEND
    a.out.mkdir(parents=True, exist_ok=True)
    a.report_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_manifest(a.manifest) if r["source"] in GUN_SOURCES]
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} gun clips, {a.jobs} workers", flush=True)

    clips, refs, index, arrays = [], [], [], []
    with Pool(a.jobs, initializer=_init, initargs=(str(a.edge_labels),)) as pool:
        for k, res in enumerate(pool.imap(_work, rows, chunksize=4), 1):
            clips.append(res["clip"])
            refs.extend(res["refs"])
            for meta, w in res["windows"]:
                index.append(meta)
                arrays.append(w)
            if k % 1000 == 0:
                print(f"  {k}/{len(rows)} clips, {len(index)} windows", flush=True)

    root = a.manifest.resolve().parent
    rel = lambda p: os.path.relpath(p, root)  # noqa: E731
    order = sorted(range(len(index)), key=lambda i: (index[i]["class"], index[i]["path"], index[i]["trigger_sample"]))
    index = [{**index[i], "path": rel(index[i]["path"])} for i in order]
    arrays = [arrays[i] for i in order]
    for c in clips:
        c["path"] = rel(c["path"])
    for r in refs:
        r["path"] = rel(r["path"])

    with open(a.report_dir / "window_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        w.writeheader()
        w.writerows(index)
    with open(a.out / "detector_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLIP_COLS)
        w.writeheader()
        w.writerows(clips)
    with open(a.out / "detector_refs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "session_id", "group", "source", "device",
                                          "ref_sample", "outcome", "trigger_sample"])
        w.writeheader()
        w.writerows(refs)

    tables = {"source": summarise(clips, lambda c: c["source"]),
              "class": summarise(clips, lambda c: c["class"]),
              "session": summarise(clips, lambda c: c["session_id"]),
              "source_class_session": summarise(clips, lambda c: (c["source"], c["class"], c["session_id"]))}
    for name, t in tables.items():
        write_table(a.report_dir / f"detector_by_{name}.csv", t, name)
    total = summarise(clips, lambda c: "all")["all"]

    failures = [(name, k if isinstance(k, str) else "/".join(k), s["recall"])
                for name in ("source", "class", "session") for k, s in tables[name].items()
                if not s["recall"] >= a.gate]
    sr, real = cfg.window.sample_rate, cfg.window.real_samples
    md = ["# Detector recall and gun windows (Step 1.6, Decision D22)", "",
          f"Generated by `sw/frontend/cut_windows.py` on "
          f"{'the FIRST ' + str(a.limit) + ' gun clips only (debug run, not valid for D22)' if a.limit else 'every gun clip of the manifest'}"
          f" ({len(rows)} clips). "
          f"Recall = known shots with their own matched trigger / known shots. Gate: {a.gate:.0%} per source, "
          f"per class and per session.", "",
          f"Totals: {total['refs']} known shots, {total['matched']} matched ({100 * total['recall']:.2f} %), "
          f"{total['inside_other_window']} inside another trigger's window, {total['missed']} truly missed, "
          f"{total['at_clip_start']} in the first {cfg.detector.warmup_samples} samples of a clip (detector warm-up, "
          f"not counted), {total['beyond_clip']} labelled past the clip end (not counted). "
          f"{total['triggers']} triggers: {total['unmatched_triggers']} unmatched (echoes, other shooters; never "
          f"windows), {total['unverified_triggers']} in unverifiable clips, {total['preempted']} preempted. "
          f"{total['windows']} gun windows kept; dropped by placement: {total['dropped_starts_before_audio']} start "
          f"before the clip audio, {total['dropped_short_after_start']} have less than {real / sr:.2f} s after the start.", ""]
    for name, title in (("source", "Per source"), ("class", "Per class"), ("session", "Per session"),
                        ("source_class_session", "Per source, class and session")):
        md += [f"## {title}", ""] + md_table(tables[name], name) + [""]
    md += ["## Windows kept under other pre-trigger lengths (0.40 s of real audio each)", ""]
    md += whatif(refs, clips, sr, real) + [""]
    md += ["## Gate", ""]
    md += ([f"- FAIL {n} `{k}`: {100 * r:.1f} %" for n, k, r in failures] or [f"- PASS: every source, class and "
                                                                               f"session at or above {a.gate:.0%}."])
    (a.report_dir / "detector_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    run = {"config": cfg.to_dict(), "register_widths": register_widths(cfg.detector), "gate": a.gate,
           "manifest_sha256": sha256(a.manifest), "edge_labels_sha256": sha256(a.edge_labels),
           "window_index_sha256": sha256(a.report_dir / "window_index.csv"), "git_commit": git_commit(HERE),
           "limit": a.limit, "totals": total, "gate_failures": failures}

    est = len(arrays) * cfg.window.window_samples * 2
    free = shutil.disk_usage(a.out).free
    if free - est < a.min_free_gb * 2**30:
        print(f"Refusing to write gun_windows.npz: {free / 2**30:.2f} GB free, up to {est / 2**30:.2f} GB needed",
              file=sys.stderr)
        return 2
    X = np.stack(arrays) if arrays else np.zeros((0, cfg.window.window_samples), np.int16)
    np.savez_compressed(a.out / "gun_windows.npz", windows=X,
                        gain_shift=np.array([r["gain_shift"] for r in index], np.int8),
                        trigger_sample=np.array([r["trigger_sample"] for r in index], np.int64),
                        window_start=np.array([r["window_start"] for r in index], np.int64),
                        sample_rate=np.int32(sr), window_index_sha256=run["window_index_sha256"],
                        config_json=json.dumps(cfg.to_dict()))
    run["gun_windows_npz_bytes"] = (a.out / "gun_windows.npz").stat().st_size
    (a.report_dir / "detector_run.json").write_text(json.dumps(run, indent=1, default=str) + "\n")
    print(f"Wrote {len(index)} windows to {a.out / 'gun_windows.npz'} ({run['gun_windows_npz_bytes'] / 2**20:.1f} MB)")
    if failures:
        print(f"GATE FAILED: {len(failures)} recall gate(s) below {a.gate:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
