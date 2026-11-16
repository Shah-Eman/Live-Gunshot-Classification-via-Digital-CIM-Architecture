"""From a manifest row to the int16 stream the detector sees, and the window gain.

Every source goes through exactly the same steps:

1. manifest.load_audio(): channels averaged, polyphase resampling to 16 kHz.
2. clamp to [-1, 1] (resampling a clipped clip can overshoot full scale).
3. quantise to int16 like a 16-bit ADC: q = clip(rint(x * 32768), -32768, 32767)
   (rint rounds half to even).

No band-limiting filter is applied: clips recorded below 16 kHz are dropped
instead (Decision D22, bandwidth note), and every remaining clip has passed
through the same resampler and its anti-aliasing filter.

Normalisation is a per-window power-of-two gain, a hardware operation:

    peak = max |w[i]| over the real part of the window (i < real_samples)
    k    = clamp(15 - bit_length(peak), 0, max_shift)   (peak = 0 gives k = max_shift)
    out  = saturate_int16(w << k)

In RTL: a 16-bit max register while the window is captured, a priority encoder
for k (4 bits), and a barrel shift when the window is read out to the engine.
With k chosen this way the shift never saturates (peak << k <= 32767), except
for a window whose peak is exactly -32768 (k = 0, value unchanged).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from manifest import load_audio  # noqa: E402

from config import GainConfig, WindowConfig  # noqa: E402


def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)


def load_clip(row: dict, window: WindowConfig) -> np.ndarray:
    """One clip as the int16 16 kHz stream the detector and the windows use."""
    return to_int16(load_audio(row, sr=window.sample_rate))


def gain_shift(w: np.ndarray, window: WindowConfig, gain: GainConfig) -> int:
    real = w[: window.real_samples].astype(np.int32)
    peak = int(np.abs(real).max()) if len(real) else 0
    if peak == 0:
        return gain.max_shift
    return int(min(max(15 - peak.bit_length(), 0), gain.max_shift))


def apply_gain(w: np.ndarray, k: int) -> np.ndarray:
    return np.clip(w.astype(np.int32) << k, -32768, 32767).astype(np.int16)


def normalise(w: np.ndarray, window: WindowConfig, gain: GainConfig) -> tuple[np.ndarray, int]:
    k = gain_shift(w, window, gain)
    return apply_gain(w, k), k
