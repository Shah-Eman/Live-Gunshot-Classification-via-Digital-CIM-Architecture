"""Per-file audio measurements used by filter_dataset.py.

Everything here is a pure measurement; the accept/reject decision lives in
filter_dataset.py so thresholds can be tuned without re-reading any audio.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

EPS = 1e-12


def db(x: float) -> float:
    return float(10.0 * np.log10(max(x, EPS)))


def wav_truncated(path: Path) -> bool | None:
    """True if a RIFF/WAVE 'data' chunk declares more bytes than the file holds.

    libsndfile silently clamps such files, so this has to be checked by hand.
    Returns None for non-WAV files or if the header can't be parsed.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            hdr = f.read(12)
            if hdr[:4] not in (b"RIFF", b"RF64") or hdr[8:12] != b"WAVE":
                return None
            while True:
                ch = f.read(8)
                if len(ch) < 8:
                    return None
                cid, csz = struct.unpack("<4sI", ch)
                if cid == b"data":
                    if csz == 0xFFFFFFFF or csz == 0:  # streaming / unknown length
                        return False
                    return f.tell() + csz > size + 1
                f.seek(csz + (csz & 1), 1)
    except Exception:
        return None


def frame_energy_db(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Mean-square energy per frame in dB (dBFS^2 style), via cumulative sums."""
    if len(x) < win:
        return np.array([db(float(np.mean(x * x)))])
    c = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    starts = np.arange(0, len(x) - win + 1, hop)
    e = (c[starts + win] - c[starts]) / win
    return 10.0 * np.log10(np.maximum(e, EPS))


def analyze(path: str | Path) -> dict:
    path = Path(path)
    m: dict = {"file": path.name, "decode_ok": False, "error": ""}
    m["bytes"] = path.stat().st_size
    if m["bytes"] == 0:
        m["error"] = "empty file"
        return m

    m["wav_truncated"] = wav_truncated(path)
    try:
        info = sf.info(str(path))
        m.update(sr=info.samplerate, channels=info.channels, subtype=info.subtype,
                 fmt=info.format, header_frames=info.frames)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # decoder warnings => treat as corruption
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - any decode failure means "corrupted/unreadable"
        m["error"] = f"{type(e).__name__}: {e}"[:200]
        return m

    n = data.shape[0]
    m["frames"] = n
    m["duration_s"] = n / sr if sr else 0.0
    if n == 0:
        m["error"] = "zero frames"
        return m
    m["finite"] = bool(np.isfinite(data).all())
    if not m["finite"]:
        m["decode_ok"] = True
        return m
    m["decode_ok"] = True

    # Frames the header promised but the decoder could not deliver (not for mp3: estimated).
    hf = info.frames
    m["missing_frac"] = max(0.0, (hf - n) / hf) if hf and info.format != "MP3" else 0.0

    raw = data.mean(axis=1)
    # Remove DC first: a constant offset (common in these recordings) otherwise
    # inflates the "background" level and hides real shots.
    x = raw - np.median(raw)
    peak = float(np.abs(x).max())
    m["peak_dbfs"] = 20 * np.log10(max(peak, EPS))
    m["rms_dbfs"] = db(float(np.mean(x.astype(np.float64) ** 2)))  # 10*log10(mean square) == 20*log10(RMS)
    m["dc_offset"] = float(abs(np.median(raw)))
    a_all = np.abs(data)
    clipped = int(np.count_nonzero(a_all >= 0.999))
    m["clip_frac"] = clipped / a_all.size                    # share of the whole file
    active = int(np.count_nonzero(a_all >= 0.05))            # samples above about -26 dBFS
    m["clip_frac_active"] = clipped / active if active else 0.0  # share of the audible part
    m["zero_frac"] = float(np.mean(raw == 0.0))
    # Channel disagreement (one dead channel is common with broken stereo mics).
    if data.shape[1] > 1:
        ch_peaks = np.abs(data).max(axis=0)
        m["dead_channel"] = bool(ch_peaks.min() < 1e-4 and ch_peaks.max() > 1e-2)
    else:
        m["dead_channel"] = False

    # --- envelope / transient analysis -----------------------------------
    win = max(int(0.004 * sr), 8)
    hop = max(win // 4, 1)
    hop_s = hop / sr
    env = frame_energy_db(x, win, hop)
    pk_i = int(np.argmax(env))
    pk_db = float(env[pk_i])
    m["env_peak_db"] = pk_db
    m["peak_time_s"] = (pk_i * hop + win / 2) / sr
    med_db = float(np.median(env))
    m["peak_to_median_db"] = pk_db - med_db           # impulsiveness over the whole clip
    m["peak_to_p10_db"] = pk_db - float(np.percentile(env, 10))
    xd = x.astype(np.float64) - x.mean()
    v = float(np.mean(xd ** 2))
    m["kurtosis"] = float(np.mean(xd ** 4) / (v * v)) if v > 0 else 0.0

    guard = int(0.25 / hop_s)  # +-250 ms around the loudest event is "the shot"
    mask = np.ones(len(env), bool)
    mask[max(0, pk_i - guard): pk_i + guard + 1] = False
    mask &= env > -90.0  # ignore digital-silence padding
    floor_db = float(np.median(env[mask])) if mask.any() else -90.0
    m["noise_floor_db"] = floor_db
    m["snr_db"] = pk_db - floor_db

    # Attack: how long the envelope takes to climb the last 20 dB to its peak.
    j = pk_i
    while j > 0 and env[j] > pk_db - 20.0:
        j -= 1
    m["attack_ms"] = (pk_i - j) * hop_s * 1000.0
    # Decay: time after the peak to fall 20 dB.
    k = pk_i
    while k < len(env) - 1 and env[k] > pk_db - 20.0:
        k += 1
    m["decay_ms"] = (k - pk_i) * hop_s * 1000.0
    m["dense_frac"] = float(np.mean(env > pk_db - 10.0))  # ~1 for tones / steady noise

    # Rough count of distinct impulsive events (peaks >=15 dB over floor, >=60 ms apart).
    thr = max(floor_db + 15.0, pk_db - 25.0)
    min_gap = max(int(0.06 / hop_s), 1)
    above = np.where(env >= thr)[0]
    events, last = 0, -10**9
    if len(above):
        run_start = above[0]
        prev = above[0]
        for idx in list(above[1:]) + [None]:
            if idx is None or idx - prev > 1:
                seg = env[run_start: prev + 1]
                loc = run_start + int(np.argmax(seg))
                if loc - last >= min_gap:
                    events += 1
                    last = loc
                if idx is None:
                    break
                run_start = idx
            prev = idx
    m["n_events"] = events

    # --- spectrum around the loudest event -------------------------------
    c = int(pk_i * hop + win // 2)
    seg_len = min(max(int(0.1 * sr), 256), n)
    s0 = int(np.clip(c - seg_len // 2, 0, n - seg_len))
    seg = x[s0: s0 + seg_len] * np.hanning(seg_len)
    ps = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(seg_len, 1 / sr)
    tot = ps.sum()
    if tot > 0:
        cum = np.cumsum(ps) / tot
        m["rolloff95_hz"] = float(freqs[min(np.searchsorted(cum, 0.95), len(freqs) - 1)])
        m["hf_ratio"] = float(ps[freqs >= 1000].sum() / tot)
    else:
        m["rolloff95_hz"] = 0.0
        m["hf_ratio"] = 0.0
    return m
