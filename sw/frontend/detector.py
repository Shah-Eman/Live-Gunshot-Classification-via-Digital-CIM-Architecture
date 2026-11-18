"""Event detector: energy threshold plus refractory period, integers only.

The RTL detector of Step 4.7 must trigger on exactly the same samples as this
file. Two implementations are given and tested against each other on real
clips (test_detector.py):

- detect():           numpy, vectorised over the clip;
- detect_reference(): one sample at a time with plain Python integers, written
                      the way the RTL is clocked. Copy this one into RTL.

Input: int16 samples at 16 kHz (the ADC output; see audio.py for how a clip
becomes int16). Nothing in the decision path is floating point.

Per sample n (all registers start at zero after reset, x[-1] = 0):

    d[n]  = x[n] - x[n-1]                          first difference (removes DC and rumble)
    p[n]  = (d[n] * d[n]) >> PS                    instantaneous power
    Ef[n] = p[n-NF+1] + ... + p[n]                 short-term energy, NF samples
    Eb[n] = p[n-NF-NB+1] + ... + p[n-NF]           background energy, the NB samples before Ef's window
    cand  = (Ef << log2(NB/NF)) >= (Eb << RATIO)   mean power at least 2**RATIO times the background mean
            and Ef >= EMIN                         and above an absolute floor
            and n >= NF + NB - 1                   both windows full (warm-up after reset)

Trigger logic (one refractory counter R_cnt and one peak register Epk):

    if R_cnt == 0:
        if cand: trigger at n; R_cnt = R - 1; Epk = Ef
    elif cand and Ef > (Epk << PRE):               a much louder event preempts the last trigger
        replace the last trigger by n; R_cnt = R - 1; Epk = Ef
    else:
        R_cnt -= 1; Epk = max(Epk, Ef)

So after a trigger at t the next free trigger can come at t + R. Preemption
exists because ranges have other shooters: without it, a faint unrelated shot
starts a refractory period that hides the labelled shot right after it. In
hardware, a preempted trigger abandons the window capture it started and the
capture restarts at the new trigger; only final triggers produce windows.

The running sums are updated as Ef += p[n] - p[n-NF] and
Eb += p[n-NF] - p[n-NF-NB], so the RTL needs the last NF+NB values of p (a
delay line), or recomputes p[n-NF] and p[n-NF-NB] from the audio ring buffer
that captures the window anyway.

Register widths (register_widths() derives them from the config, and both
implementations assert them on every sample), for the default config
(NF=32, NB=512, PS=4, RATIO=3, PRE=2, R=1600):

    x               16 bit signed
    d               17 bit signed   (-65535 .. 65535)
    d*d             32 bit unsigned (<= 65535**2 < 2**32)
    p               28 bit unsigned
    p delay line    544 entries x 28 bit
    Ef              33 bit unsigned (NF * (2**28 - 1))
    Eb              37 bit unsigned (NB * (2**28 - 1))
    Ef << 4         37 bit          ratio comparison, left side
    Eb << 3         40 bit          ratio comparison, right side: one 40-bit unsigned comparator
    Epk             33 bit unsigned
    Epk << 2        35 bit          preemption comparison: one 35-bit unsigned comparator
    R_cnt           11 bit unsigned (0 .. 1599)

No multiplier is needed except d*d (17 x 17 bit); every scaling is a shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DetectorConfig


def _bits(v: int) -> int:
    return max(1, int(v).bit_length())


def register_widths(cfg: DetectorConfig) -> dict[str, int]:
    """Bit width of every detector register, derived from the config."""
    d_max = 2 * 32767 + 1                       # 65535 = 32767 - (-32768)
    p_max = (d_max * d_max) >> cfg.power_shift
    ef_max = cfg.fast_samples * p_max
    eb_max = cfg.background_samples * p_max
    sh = (cfg.background_samples // cfg.fast_samples).bit_length() - 1
    return {
        "x": 16,
        "d": _bits(d_max) + 1,                  # signed
        "d2": _bits(d_max * d_max),
        "p": _bits(p_max),
        "Ef": _bits(ef_max),
        "Eb": _bits(eb_max),
        "ratio_lhs": _bits(ef_max << sh),
        "ratio_rhs": _bits(eb_max << cfg.ratio_log2),
        "Epk": _bits(ef_max),
        "preempt_rhs": _bits(ef_max << cfg.preempt_log2),
        "R_cnt": _bits(cfg.refractory_samples - 1),
        "delay_line_entries": cfg.fast_samples + cfg.background_samples,
    }


@dataclass
class Detection:
    triggers: np.ndarray      # int64 sample indices of the final triggers
    preempted: np.ndarray     # int64 sample indices of triggers that a louder event replaced


def _check_input(x: np.ndarray) -> None:
    if not isinstance(x, np.ndarray) or x.dtype != np.int16 or x.ndim != 1:
        raise TypeError("detector input must be a 1-D int16 array (16 kHz ADC samples)")


def energies(x: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p, Ef, Eb) for every sample, int64, exactly as the sample loop computes them."""
    _check_input(x)
    xi = x.astype(np.int64)
    d = np.diff(xi, prepend=0)
    p = (d * d) >> cfg.power_shift
    c = np.concatenate(([0], np.cumsum(p)))
    n1 = np.arange(1, len(x) + 1)
    nf = cfg.fast_samples
    w = cfg.fast_samples + cfg.background_samples
    ef = c[n1] - c[np.maximum(n1 - nf, 0)]
    eb = c[np.maximum(n1 - nf, 0)] - c[np.maximum(n1 - w, 0)]
    widths = register_widths(cfg)
    assert len(x) == 0 or (ef.max() < 1 << widths["Ef"] and eb.max() < 1 << widths["Eb"])
    return p, ef, eb


def candidates(ef: np.ndarray, eb: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    sh = (cfg.background_samples // cfg.fast_samples).bit_length() - 1
    n = np.arange(len(ef))
    return ((ef << sh) >= (eb << cfg.ratio_log2)) & (ef >= cfg.min_energy) & (n >= cfg.warmup_samples)


def detect(x: np.ndarray, cfg: DetectorConfig) -> Detection:
    """Vectorised detector. Same triggers as detect_reference(), sample for sample."""
    _, ef, eb = energies(x, cfg)
    cand = candidates(ef, eb, cfg)
    r = cfg.refractory_samples
    triggers: list[int] = []
    preempted: list[int] = []
    idx = np.flatnonzero(cand)
    t = int(idx[0]) if len(idx) else None
    while t is not None:
        triggers.append(t)
        seg = ef[t:t + r]
        running_max = np.maximum.accumulate(seg)          # Epk before each sample of the refractory period
        pre = np.flatnonzero(cand[t + 1:t + r] & (seg[1:] > (running_max[:-1] << cfg.preempt_log2)))
        if len(pre):
            preempted.append(triggers.pop())
            t = t + 1 + int(pre[0])
            continue
        nxt = np.flatnonzero(cand[t + r:])
        t = t + r + int(nxt[0]) if len(nxt) else None
    return Detection(np.asarray(triggers, dtype=np.int64), np.asarray(preempted, dtype=np.int64))


def detect_reference(x: np.ndarray, cfg: DetectorConfig) -> Detection:
    """Sample-by-sample detector with Python integers: the model for the RTL."""
    _check_input(x)
    w = register_widths(cfg)
    nf, nb = cfg.fast_samples, cfg.background_samples
    sh = (nb // nf).bit_length() - 1
    line = [0] * (nf + nb)        # p delay line: line[(n - k) % L] holds p[n - k]
    L = nf + nb
    x_prev = ef = eb = 0
    r_cnt = 0
    e_pk = 0
    triggers: list[int] = []
    preempted: list[int] = []
    for n, xv in enumerate(x.tolist()):
        d = xv - x_prev
        x_prev = xv
        p = (d * d) >> cfg.power_shift
        p_nf = line[(n - nf) % L]          # p[n-NF]   (0 before it exists)
        p_old = line[n % L]                # p[n-NF-NB], about to be overwritten
        line[n % L] = p
        ef += p - p_nf
        eb += p_nf - p_old
        assert 0 <= ef < 1 << w["Ef"] and 0 <= eb < 1 << w["Eb"] and p < 1 << w["p"]
        lhs, rhs = ef << sh, eb << cfg.ratio_log2
        assert lhs < 1 << w["ratio_lhs"] and rhs < 1 << w["ratio_rhs"]
        cand = lhs >= rhs and ef >= cfg.min_energy and n >= cfg.warmup_samples
        if r_cnt == 0:
            if cand:
                triggers.append(n)
                r_cnt = cfg.refractory_samples - 1
                e_pk = ef
        elif cand and ef > (e_pk << cfg.preempt_log2):
            preempted.append(triggers.pop())
            triggers.append(n)
            r_cnt = cfg.refractory_samples - 1
            e_pk = ef
        else:
            r_cnt -= 1
            e_pk = max(e_pk, ef)
        assert 0 <= r_cnt < 1 << w["R_cnt"] and e_pk < 1 << w["Epk"]
    return Detection(np.asarray(triggers, dtype=np.int64), np.asarray(preempted, dtype=np.int64))
