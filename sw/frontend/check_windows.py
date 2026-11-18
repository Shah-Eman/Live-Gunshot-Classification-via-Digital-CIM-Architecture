#!/usr/bin/env python3
"""Review gate for the windows (Step 1.6): exits 1 if any check fails.

1. Provenance: every gun window is what detector.py + windows.py produce today: the clip is re-read, the
   detector re-run and the triggers re-verified; the stored trigger, start, gain shift and all 8000 samples
   must match bit for bit, and every stored window must come from a matched (verified) trigger.
2. Format and padding: every window (gun windows, and the background windows of the sampled epochs) has
   window_samples samples at 16 kHz, an all-zero tail of exactly tail_samples, and no run of exact zeros at
   the start or end of its real part longer than max_zero_run (padding). Zero runs inside the real audio
   (recording dropouts) are reported, not failed.
3. No per-source differences, per gun source and per background source:
   - sample rate: every window comes from a clip recorded at >= 16 kHz;
   - bandwidth: the source's mean spectrum (windows normalised to unit energy) stays within 50 dB of its
     peak up to at least 7.5 kHz. The 55 excluded clips are run through the same test and must fail it,
     which shows the test can see a band limit;
   - gain: after the power-of-two gain every window's peak lies in the top octave [16384, 32767] (unless the
     shift hit max_shift, allowed for at most 1 % of a source), and the median peaks of any two sources
     differ by at most 3 dB;
   - the 95 % roll-off per source is reported (it depends on content and microphone distance, not on the
     window cutting, so it is not gated).
4. Epochs: several epochs are actually built with background.EpochBuilder; in each one at least 15 % of
   the background windows come from the gun microphones, some background windows are detector-cut, and some
   gun windows are mixed onto US8K / ESC-50 noise.

Usage
  python sw/frontend/check_windows.py --manifest DATASET/manifest.csv --edge-labels LABELS.csv \
      --splits sw/data/splits.json --window-index results/step16/window_index.csv \
      --background-index results/step16/background_index.csv --gun-npz OUT/gun_windows.npz \
      [--report results/step16/check_windows_report.md]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from audio import apply_gain, load_clip  # noqa: E402
from background import EpochBuilder, zero_run  # noqa: E402
from config import FRONTEND  # noqa: E402
from manifest import load_manifest  # noqa: E402
from windows import process_gun_clip, read_edge_labels  # noqa: E402

BAND_EDGE_DB, BAND_EDGE_MIN_HZ = -50.0, 7500.0
MAX_MEDIAN_PEAK_SPREAD_DB = 3.0
MAX_SATURATED_GAIN_SHARE = 0.01
_LABELS: dict = {}


def _init(path: str) -> None:
    global _LABELS
    _LABELS = read_edge_labels(path)


def _redo(row: dict) -> tuple[str, list[tuple[dict, np.ndarray]]]:
    return row["path"], process_gun_clip(row, _LABELS, FRONTEND)["windows"]


def edge_run(x: np.ndarray) -> int:
    """Longest exact-zero run touching the start or the end of x."""
    nz = np.flatnonzero(x)
    return len(x) if len(nz) == 0 else int(max(nz[0], len(x) - 1 - nz[-1]))


def band_edge(psd_mean: np.ndarray, f: np.ndarray) -> float:
    db = 10 * np.log10(psd_mean / psd_mean.max() + 1e-30)
    return float(f[np.flatnonzero(db >= BAND_EDGE_DB).max()])


def unit_psd(w: np.ndarray, real: int) -> tuple[np.ndarray, np.ndarray] | None:
    x = w[:real].astype(np.float64)
    if not np.any(x):
        return None
    f, p = welch(x, fs=FRONTEND.window.sample_rate, nperseg=512)
    return f, p / p.sum()


def rolloff(f: np.ndarray, p: np.ndarray) -> float:
    c = np.cumsum(p)
    return float(f[np.searchsorted(c, 0.95 * c[-1])])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--edge-labels", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--window-index", type=Path, required=True)
    ap.add_argument("--background-index", type=Path, required=True)
    ap.add_argument("--gun-npz", type=Path, required=True)
    ap.add_argument("--epochs", default="0:0,0:1,0:2,1:0,2:0,3:0,4:0",
                    help="fold:epoch pairs to build and measure")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()

    cfg = FRONTEND
    win, gain, bc = cfg.window, cfg.gain, cfg.background
    real, tail = win.real_samples, win.tail_samples
    fails: list[str] = []
    lines: list[str] = []

    def check(ok: bool, msg: str) -> None:
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {msg}")
        print(lines[-1], flush=True)
        if not ok:
            fails.append(msg)

    root = a.manifest.resolve().parent
    rows = {str(Path(r["path"]).relative_to(root)): r for r in load_manifest(a.manifest)}
    with open(a.window_index, newline="") as f:
        index = list(csv.DictReader(f))
    d = np.load(a.gun_npz)
    X, K = d["windows"], d["gain_shift"]
    idx_sha = hashlib.sha256(a.window_index.read_bytes()).hexdigest()

    # 1. Provenance -----------------------------------------------------------
    check(str(d["window_index_sha256"]) == idx_sha and len(X) == len(index),
          f"gun_windows.npz ({len(X)} windows) belongs to this window_index.csv ({len(index)} rows, sha256 match)")
    check(int(d["sample_rate"]) == win.sample_rate and X.dtype == np.int16 and X.shape[1:] == (win.window_samples,),
          f"gun windows are int16 [{win.window_samples}] at {int(d['sample_rate'])} Hz")
    by_clip = defaultdict(list)
    for i, r in enumerate(index):
        by_clip[r["path"]].append(i)
    clip_rows = [rows[p] for p in by_clip]
    mismatch, n_checked = [], 0
    with Pool(a.jobs, initializer=_init, initargs=(str(a.edge_labels),)) as pool:
        for path, redo in pool.imap(_redo, clip_rows, chunksize=4):
            rel = str(Path(path).relative_to(root))
            stored = by_clip[rel]
            if len(redo) != len(stored):
                mismatch.append(f"{rel}: {len(stored)} stored vs {len(redo)} recomputed windows")
                continue
            for i, (meta, w) in zip(stored, redo):
                r = index[i]
                n_checked += 1
                if (int(r["trigger_sample"]), int(r["window_start"]), int(r["gain_shift"])) != \
                        (meta["trigger_sample"], meta["window_start"], meta["gain_shift"]) or \
                        not np.array_equal(X[i], w) or r["verified"] != "1" or int(K[i]) != meta["gain_shift"]:
                    mismatch.append(f"{rel} trigger {r['trigger_sample']}")
    check(not mismatch and n_checked == len(index),
          f"every gun window re-derived from its clip by detector.py and verified against a reference shot: "
          f"{n_checked} identical, {len(mismatch)} different{': ' + mismatch[0] if mismatch else ''}")
    check(all(int(r["window_start"]) == int(r["trigger_sample"]) - win.pre_trigger_samples for r in index),
          f"every gun window starts {win.pre_trigger_samples} samples before its trigger")

    # 4. Epochs (built first: their background windows feed the per-source checks) ------
    per_source = defaultdict(list)          # source -> list of (normalised window, gain shift)
    for i, r in enumerate(index):
        per_source[("gun", r["source"])].append((apply_gain(X[i], int(K[i])), int(K[i])))
    ep_lines = ["", "## Sampled epochs", "",
                "| fold | epoch | windows | gun | gun mixed onto noise | background | from gun microphones | "
                "share | detector-cut background | random background |", "|---|---:|" + "---:|" * 8]
    builders = {}
    for pair in a.epochs.split(","):
        fold, ep = (int(v) for v in pair.split(":"))
        if fold not in builders:
            builders[fold] = EpochBuilder.from_files(a.manifest, a.splits, a.window_index, a.gun_npz,
                                                     a.background_index, fold=fold)
        Xe, ye, me = builders[fold].epoch(ep)
        kinds = Counter(m["kind"] for m in me)
        bg = [m for m in me if m["kind"] not in ("gun", "gun_mixed")]
        gunmic = sum(m["source"] in ("edge-quiet", "cadre-quiet") for m in bg)
        share = gunmic / len(bg)
        ep_lines.append(f"| {fold} | {ep} | {len(me)} | {kinds['gun']} | {kinds['gun_mixed']} | {len(bg)} | {gunmic} | "
                        f"{100 * share:.1f} % | {kinds['triggered']} | {kinds['random']} |")
        check(share >= bc.min_gunmic_share and kinds["triggered"] > 0 and kinds["gun_mixed"] > 0,
              f"fold {fold} epoch {ep}: {100 * share:.1f} % of {len(bg)} background windows from the gun microphones "
              f"(>= {100 * bc.min_gunmic_share:.0f} %), {kinds['triggered']} detector-cut, "
              f"{kinds['gun_mixed']} gun windows mixed onto noise")
        bad = [i for i in range(len(me)) if Xe[i].shape != (win.window_samples,) or np.any(Xe[i][real:])
               or edge_run(Xe[i][:real]) > bc.max_zero_run]
        check(not bad, f"fold {fold} epoch {ep}: every window has {win.window_samples} samples, a zero tail of {tail} "
                       f"and no zero padding at either end of its real part ({len(bad)} bad)")
        if (fold, ep) == (0, 0):
            for i, m in enumerate(me):
                if m["kind"] not in ("gun", "gun_mixed"):
                    per_source[("background", m["source"])].append((Xe[i], int(m["gain_shift"])))
            bg_paths = {m["path"] for m in bg}
            low = [p for p in bg_paths if rows[p]["sample_rate"] < bc.min_native_sr]
            check(not low, f"no background window comes from a clip recorded below {bc.min_native_sr} Hz ({len(low)})")

    # 2. Gun window format and padding -----------------------------------------
    tails = int(np.count_nonzero(np.any(X[:, real:] != 0, axis=1)))
    edges = [edge_run(x[:real]) for x in X]
    mids = [zero_run(x[:real]) for x in X]
    check(tails == 0, f"every gun window's last {tail} samples are zero ({tails} are not)")
    check(max(edges) <= bc.max_zero_run, f"no gun window is zero-padded at either end of its real part "
                                          f"(longest zero run at an end: {max(edges)} samples, limit {bc.max_zero_run})")
    lines.append(f"- INFO: zero runs inside the real audio of gun windows (recording dropouts): longest {max(mids)} "
                 f"samples, {sum(m > bc.max_zero_run for m in mids)} windows above {bc.max_zero_run}")
    low_gun = [r["path"] for r in index if rows[r["path"]]["sample_rate"] < bc.min_native_sr]
    check(not low_gun, f"every gun window comes from a clip recorded at >= {bc.min_native_sr} Hz ({len(low_gun)} not)")

    # 3. Per-source features ----------------------------------------------------
    tab = ["", "## Per-source window features", "",
           "| kind | source | windows | length | zero tail | band edge (-50 dB) | roll-off p5 / p50 / p95 (Hz) | "
           "median peak after gain | at max shift | gain shift histogram |", "|---|---|---:|---:|---:|---:|---|---:|---:|---|"]
    medians = {}
    for (kind, src), ws in sorted(per_source.items()):
        psds = [unit_psd(w, real) for w, _ in ws]
        psds = [p for p in psds if p is not None]
        f = psds[0][0]
        P = np.mean([p for _, p in psds], axis=0)
        edge = band_edge(P, f)
        ro = np.percentile([rolloff(f, p) for _, p in psds], [5, 50, 95]).round(0)
        ks = np.array([k for _, k in ws])
        pk = np.array([int(np.abs(w[:real].astype(np.int32)).max()) for w, _ in ws])
        top = (pk >= 16384) & (pk <= 32768)
        sat = float(np.mean(ks >= gain.max_shift))
        med = float(np.median(20 * np.log10(np.maximum(pk, 1) / 32768)))
        medians[(kind, src)] = med
        hist = ", ".join(f"{k}:{n}" for k, n in sorted(Counter(ks.tolist()).items()))
        tab.append(f"| {kind} | {src} | {len(ws)} | {win.window_samples} | {tail} | {edge:.0f} Hz | "
                   f"{' / '.join(str(int(v)) for v in ro)} | {med:.2f} dBFS | {100 * sat:.2f} % | {hist} |")
        check(edge >= BAND_EDGE_MIN_HZ, f"{kind}/{src}: mean spectrum within {-BAND_EDGE_DB:.0f} dB of its peak up to "
                                        f"{edge:.0f} Hz (>= {BAND_EDGE_MIN_HZ:.0f} Hz)")
        check(bool(np.all(top | (ks >= gain.max_shift))) and sat <= MAX_SATURATED_GAIN_SHARE,
              f"{kind}/{src}: peak after gain in [16384, 32767] for every window below max shift; "
              f"{100 * sat:.2f} % at max shift (<= {100 * MAX_SATURATED_GAIN_SHARE:.0f} %)")
    spread = max(medians.values()) - min(medians.values())
    check(spread <= MAX_MEDIAN_PEAK_SPREAD_DB,
          f"median peak after gain differs by {spread:.2f} dB between sources (<= {MAX_MEDIAN_PEAK_SPREAD_DB} dB)")

    # The bandwidth test must be able to see a band limit: run it on the excluded clips.
    excl = [r for r in rows.values() if r["sample_rate"] < bc.min_native_sr]
    psds = []
    for r in excl:
        x = load_clip(r, win)
        if len(x) >= real:
            st = (len(x) - real) // 2
            p = unit_psd(x[st:st + real], real)
            if p is not None:
                psds.append(p)
    if psds:
        e_excl = band_edge(np.mean([p for _, p in psds], axis=0), psds[0][0])
        check(e_excl < BAND_EDGE_MIN_HZ, f"self-test: the {len(excl)} excluded clips (recorded below "
                                         f"{bc.min_native_sr} Hz) fail the bandwidth test (band edge {e_excl:.0f} Hz)")
    else:
        check(False, "self-test: no excluded clip to run the bandwidth test on")

    head = ["# Window check (Step 1.6)", "",
            f"`sw/frontend/check_windows.py`; window_index.csv sha256 `{idx_sha}`; epochs {a.epochs} "
            f"(seed {cfg.seeds.epochs}).", ""]
    report = "\n".join(head + lines + tab + ep_lines) + "\n"
    if a.report:
        a.report.write_text(report)
    print("\n".join(tab + ep_lines))
    if fails:
        print(f"WINDOW CHECK FAILED ({len(fails)})", file=sys.stderr)
        return 1
    print("WINDOW CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
