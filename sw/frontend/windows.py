"""Gun windows: detector triggers checked against the dataset's shot times, then cut (D22).

Every gun window comes from a detector trigger. The dataset's timestamps are
used only to decide whether a trigger is a known shot:

- edge:  the labelled shot times (gunshot_location_in_seconds in the Kabealo
         gunshot-audio-all-metadata.csv), every labelled shot in the clip;
- c3gd:  the dataset's fixed pre-roll: 0.1 s, except c3gd_nj_farm, whose clips
         are all aligned at 0.046 s (measured; see config.ReferenceConfig);
- cadre: the onset of the clip's loudest event: the first sample within 6 dB
         of the peak, searched in the 0.1 s before the peak (one shot per file).

A trigger matches a reference when it lies between match_before_s before it
and match_after_s after it. The lead of 0.10 s is there because a supersonic
bullet's crack reaches the microphone before the muzzle blast that the edge
labels mark (5.56 shows it on about a quarter of its shots, the subsonic .38
almost never). Each reference matches at most one trigger and the other way
round. Triggers that match nothing (echoes, other shooters, handling noise) are
counted and never become windows. A clip with no reference cannot be verified
and gives no gun window.

Placement (config.WindowConfig): start = trigger - pre_trigger_samples; the
window holds real_samples of real audio from start, then zeros up to
window_samples. A window that would start before the clip or run past its end
is dropped, never padded. The clip's real audio runs from its first to its last
non-zero sample: zero padding inside a source file is treated like the file's
end, so it can never enter a window either.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from audio import gain_shift, load_clip
from config import FrontendConfig, ReferenceConfig, WindowConfig
from detector import detect

GUN_SOURCES = ("edge", "c3gd", "cadre")


def read_edge_labels(path: str | Path) -> dict[str, list[float]]:
    """Kabealo filename stem -> sorted labelled shot times in seconds."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "filename" not in rows[0] or "gunshot_location_in_seconds" not in rows[0]:
        raise ValueError(f"{path} is not the Kabealo gunshot-audio-all-metadata.csv")
    num = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")
    return {r["filename"]: sorted(float(v) for v in num.findall(r["gunshot_location_in_seconds"]))
            for r in rows}


def edge_stem(row: dict) -> str:
    """Kabealo filename stem of an edge row (orig_path is 'edge:<folder>/<stem>.wav')."""
    return Path(row["orig_path"].split(":", 1)[1]).stem


def reference_samples(row: dict, x: np.ndarray, edge_labels: dict, ref: ReferenceConfig,
                      sr: int) -> tuple[list[int], str]:
    """Known shot positions (samples at sr) of one gun clip, and a note.

    The note is 'ok', or why the clip cannot be verified.
    """
    src = row["source"]
    if src == "edge":
        stem = edge_stem(row)
        if stem not in edge_labels:
            return [], "no_edge_label"
        refs = [int(round(t * sr)) for t in edge_labels[stem]]
        return refs, "ok" if refs else "empty_edge_label"
    if src == "c3gd":
        pre = ref.c3gd_preroll_by_session.get(row["session_id"], ref.c3gd_preroll_s)
        return [int(round(pre * sr))], "ok"
    if src == "cadre":
        a = np.abs(x.astype(np.int32))
        if len(a) == 0 or a.max() == 0:
            return [], "silent"
        pk = int(a.argmax())
        lo = max(0, pk - int(round(ref.cadre_search_s * sr)))
        onset = lo + int(np.argmax(2 * a[lo:pk + 1] >= a[pk]))
        return [onset], "ok"
    raise ValueError(f"not a gun source: {src}")


def match(triggers: np.ndarray, refs: list[int], ref: ReferenceConfig, sr: int) -> dict[int, int]:
    """One-to-one trigger index -> reference index, nearest pairs first."""
    before, after = int(round(ref.match_before_s * sr)), int(round(ref.match_after_s * sr))
    pairs = sorted((abs(int(t) - r), i, j) for i, t in enumerate(triggers) for j, r in enumerate(refs)
                   if -before <= int(t) - r <= after)
    used_t, used_r, out = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_t and j not in used_r:
            out[i] = j
            used_t.add(i)
            used_r.add(j)
    return out


def audio_bounds(x: np.ndarray) -> tuple[int, int]:
    """[first, last + 1) of the non-zero samples: exact zeros at either end of a file are padding, not audio."""
    nz = np.flatnonzero(x)
    return (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)


def placement(trigger: int, hi: int, window: WindowConfig, lo: int = 0) -> tuple[int, str]:
    """(window start, 'ok' or the drop reason) for a trigger in real audio [lo, hi)."""
    start = int(trigger) - window.pre_trigger_samples
    if start < lo:
        return start, "starts_before_audio"
    if start + window.real_samples > hi:
        return start, "short_after_start"
    return start, "ok"


def cut(x: np.ndarray, start: int, window: WindowConfig) -> np.ndarray:
    """The model input: real_samples of x from start, then zeros. Never pads real audio."""
    assert 0 <= start and start + window.real_samples <= len(x), "window outside the clip"
    w = np.zeros(window.window_samples, dtype=np.int16)
    w[: window.real_samples] = x[start:start + window.real_samples]
    return w


def process_gun_clip(row: dict, edge_labels: dict, cfg: FrontendConfig) -> dict:
    """Detect, verify and cut one gun clip.

    Returns {'clip': per-clip stats, 'refs': per-reference outcomes,
    'windows': [(index row, int16 window)]}.
    """
    win, sr = cfg.window, cfg.window.sample_rate
    x = load_clip(row, win)
    det = detect(x, cfg.detector)
    trig = det.triggers
    refs, note = reference_samples(row, x, edge_labels, cfg.reference, sr)
    base = {k: row[k] for k in ("path", "class", "session_id", "group", "source", "device")}
    clip = {**base, "n_samples": len(x), "verify": note, "triggers": len(trig),
            "preempted": len(det.preempted), "refs": 0, "matched": 0, "inside_other_window": 0,
            "missed": 0, "at_clip_start": 0, "beyond_clip": 0, "unmatched_triggers": 0,
            "unverified_triggers": 0, "windows": 0, "dropped_starts_before_audio": 0,
            "dropped_short_after_start": 0}
    lo, hi = audio_bounds(x)
    clip["audio_lo"], clip["audio_hi"] = lo, hi
    if note != "ok":
        clip["unverified_triggers"] = len(trig)
        return {"clip": clip, "refs": [], "windows": []}

    pairs = match(trig, refs, cfg.reference, sr)
    by_ref = {j: i for i, j in pairs.items()}
    ref_rows = []
    for j, r in enumerate(refs):
        if r >= len(x):
            outcome = "beyond_clip"
        elif j in by_ref:
            outcome = "matched"
        elif r < cfg.detector.warmup_samples:
            outcome = "at_clip_start"
        elif any(t - win.pre_trigger_samples <= r < t - win.pre_trigger_samples + win.real_samples for t in trig):
            outcome = "inside_other_window"
        else:
            outcome = "missed"
        clip[outcome] += 1
        if outcome != "beyond_clip":
            clip["refs"] += 1 if outcome != "at_clip_start" else 0
        ref_rows.append({**base, "ref_sample": r, "outcome": outcome,
                         "trigger_sample": int(trig[by_ref[j]]) if j in by_ref else ""})
    clip["unmatched_triggers"] = len(trig) - len(pairs)

    windows = []
    for i, j in sorted(pairs.items()):
        t = int(trig[i])
        start, why = placement(t, hi, win, lo)
        if why != "ok":
            clip[f"dropped_{why}"] += 1
            continue
        w = cut(x, start, win)
        windows.append(({**base, "trigger_sample": t, "window_start": start, "ref_sample": refs[j],
                         "trigger_minus_ref": t - refs[j], "gain_shift": gain_shift(w, win, cfg.gain),
                         "verified": 1, "flags": f"ref={row['source']}"}, w))
    clip["windows"] = len(windows)
    return {"clip": clip, "refs": ref_rows, "windows": windows}
