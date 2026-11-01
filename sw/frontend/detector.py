"""Event detector: the exact algorithm the hardware front end will run (runbook Step 1.6, item 1).

Integer arithmetic only. The RTL detector of Step 4.7 must match this file sample for
sample: `detect()` is the reference, `detect_reference_loop()` is the same algorithm written
one sample at a time the way the RTL is, and the two are checked against each other
(test_detector.py). No floats appear between the PCM samples in and the trigger indices out.

Per input sample x[n] (signed PCM, BITS bits, 16 kHz):

    a[n] = |x[n]|, saturated to 2^(B-1)-1
    E[n] = a[n] + a[n-1] + ... + a[n-N+1]                 running sum, N = FRAME samples
                                                           (E[n] = E[n-1] + a[n] - a[n-N])
    Noise floor F (Q.FLOOR_FRAC fixed point), starts at E[N-1] << FLOOR_FRAC and is
    updated at the last sample of every frame (n % N == N-1), always:
        F += ((E[n] << FLOOR_FRAC) - F) >> FLOOR_SHIFT        (arithmetic shift, floor)
        F  = max(F, FLOOR_MIN << FLOOR_FRAC)
    Trigger at n (armed, i.e. not in refractory) when
        (E[n] << FLOOR_FRAC) >= (F << RATIO_SHIFT)  and  E[n] >= E_MIN
    After a trigger at t: refractory until t + REFRACTORY. During the first HOLD samples
    the peak P = max E[t .. t+HOLD-1] of the triggering event is recorded. After the hold,
    a much louder event re-triggers even in refractory:
        E[n] >= (P << RETRIG_SHIFT)  and  E[n] >= E_MIN   ->  trigger at n (restart at n)
    The check at sample n uses F as it was before that sample's own frame-end update.

Why the re-trigger: a weak sound just before the shot (handling noise, another shooter,
the first 20 ms of a C3GD clip) would otherwise trigger and hide the real shot behind
the refractory period. A shot's own tail and its echoes are never 2x louder than its
peak, so they cannot re-trigger. Why the floor keeps tracking in refractory: a frozen
floor lets the reverberant tail retrigger as soon as refractory ends.

Every multiply is a power-of-two shift. The RTL needs an N-deep |x| delay line, one
accumulator, a floor register, a peak register, shifters and comparators.

Other bit depths: `Params.scaled(bits)` shifts E_MIN and FLOOR_MIN by (bits - 16), so the
ADC sweep (Step 1.7) reuses the same detector.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

SR = 16000


@dataclass(frozen=True)
class Params:
    """Detector constants. Changing any of them changes the RTL; record it in arith_spec.md."""
    bits: int = 16              # PCM word length of the input
    frame: int = 64             # N: running-sum length and floor update period (4 ms)
    floor_frac: int = 4         # fractional bits of the floor register
    floor_shift: int = 7        # floor time constant = 2^7 frames = 512 ms; 128 ms let
                                # pre-shot handling noise raise the floor above the shot
    ratio_shift: int = 2        # trigger when E >= 4 x floor (12 dB for |x| sums); x8 missed
                                # 9 of 136 sample gun clips on the noisy edge range, x4 missed 0
    e_min: int = 64 * 82        # absolute minimum: mean |x| over a frame >= 82/32768 (-52 dBFS)
    floor_min: int = 64 * 8     # floor never drops below mean |x| = 8 LSB (no dither triggers)
    refractory: int = 8000      # 0.5 s: no new trigger unless a much louder event (below)
    hold: int = 320             # 20 ms after a trigger: measure the triggering event's peak
    retrig_shift: int = 1       # re-trigger in refractory when E >= 2 x that peak

    def scaled(self, bits: int) -> "Params":
        """The same detector for a different PCM word length (levels shifted, not rescaled)."""
        d = bits - 16
        sh = (lambda v: v << d) if d >= 0 else (lambda v: v >> -d)
        return replace(self, bits=bits, e_min=max(1, sh(self.e_min)), floor_min=max(1, sh(self.floor_min)))


DEFAULT = Params()


def to_pcm(x: np.ndarray, bits: int = 16) -> np.ndarray:
    """Float waveform in [-1, 1] to signed integer PCM, as the ADC delivers it.

    The only float step; it models the ADC, not the detector. Round half to even,
    then saturate. Returns int32 so numpy sums never overflow.
    """
    full = 1 << (bits - 1)
    q = np.rint(np.asarray(x, dtype=np.float64) * full)
    return np.clip(q, -full, full - 1).astype(np.int32)


def energy(pcm: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """E[n], the running |x| sum over the last N samples (partial sums for n < N-1)."""
    a = np.minimum(np.abs(np.asarray(pcm, dtype=np.int64)), (1 << (p.bits - 1)) - 1)
    c = np.cumsum(a)
    e = c.copy()
    e[p.frame:] -= c[:-p.frame]
    return e


def floor_per_frame(e: np.ndarray, p: Params = DEFAULT) -> np.ndarray:
    """f[k] = the floor used by samples n with n // N == k (n >= N-1)."""
    N = p.frame
    fmin = p.floor_min << p.floor_frac
    k_total = len(e) // N + 1
    f = np.empty(k_total, dtype=np.int64)
    fl = max(int(e[N - 1]) << p.floor_frac, fmin)
    f[0] = fl
    for k in range(k_total - 1):               # update at the end of frame k -> f[k+1]
        n = k * N + N - 1
        if n >= len(e):
            f[k + 1:] = fl
            break
        fl = max(fl + (((int(e[n]) << p.floor_frac) - fl) >> p.floor_shift), fmin)
        f[k + 1] = fl
    return f


def detect(pcm: np.ndarray, p: Params = DEFAULT, return_trace: bool = False):
    """Trigger sample indices for one integer PCM stream (1-D, int). Bit-exact RTL reference.

    With return_trace, returns (triggers, E, floor_per_sample) for plots and test vectors.
    """
    pcm = np.asarray(pcm)
    if pcm.ndim != 1 or not np.issubdtype(pcm.dtype, np.integer):
        raise TypeError("detect() takes a 1-D integer PCM array; use to_pcm() first")
    n_total, N = len(pcm), p.frame
    if n_total < N:
        return ([], None, None) if return_trace else []
    e = energy(pcm, p)
    f = floor_per_frame(e, p)
    fl_n = f[np.arange(n_total) // N]
    cond = ((e << p.floor_frac) >= (fl_n << p.ratio_shift)) & (e >= p.e_min)
    cond[:N - 1] = False
    loud = e >= p.e_min

    trig: list[int] = []
    pos = N - 1
    while pos < n_total:
        hit = np.flatnonzero(cond[pos:])
        if not hit.size:
            break
        t = pos + int(hit[0])
        while True:                                   # t is a trigger; look for a re-trigger
            trig.append(t)
            h_end, r_end = min(t + p.hold, n_total), min(t + p.refractory, n_total)
            peak = int(e[t:h_end].max())
            seg = np.flatnonzero((e[h_end:r_end] >= (peak << p.retrig_shift)) & loud[h_end:r_end])
            if not seg.size:
                pos = t + p.refractory
                break
            t = h_end + int(seg[0])
    if return_trace:
        return trig, e, fl_n
    return trig


def detect_reference_loop(pcm: np.ndarray, p: Params = DEFAULT) -> list[int]:
    """The same algorithm one sample at a time, the way the RTL is written. Slow; for tests."""
    N, full = p.frame, (1 << (p.bits - 1)) - 1
    fmin = p.floor_min << p.floor_frac
    line = [0] * N
    E, fl = 0, None
    t_last, peak = None, 0          # last trigger and its event peak
    trig = []
    for n, x in enumerate(int(v) for v in pcm):
        a = min(abs(x), full)
        E += a - line[n % N]
        line[n % N] = a
        if n < N - 1:
            continue
        if fl is None:
            fl = max(E << p.floor_frac, fmin)
        fire = False
        if t_last is None or n >= t_last + p.refractory:                       # armed
            fire = (E << p.floor_frac) >= (fl << p.ratio_shift) and E >= p.e_min
        elif n < t_last + p.hold:                                              # hold: track peak
            peak = max(peak, E)
        else:                                                                  # refractory
            fire = E >= (peak << p.retrig_shift) and E >= p.e_min
        if fire:
            trig.append(n)
            t_last, peak = n, E
        if n % N == N - 1:
            fl = max(fl + (((E << p.floor_frac) - fl) >> p.floor_shift), fmin)
    return trig
