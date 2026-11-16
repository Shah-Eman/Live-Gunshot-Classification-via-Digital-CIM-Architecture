"""Every number of the Step 1.6 front end, in one place.

Step 1.7 (ADC sweep, window length) changes the front end by editing or
replacing these objects, never by editing the code that uses them. Each object
serialises to a dict, and every report and output file records the dict it was
made with.

    from config import FRONTEND
    FRONTEND.window.window_samples      # 8000
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class WindowConfig:
    """Decision D22, option (a): where the window sits around a trigger.

    The model input is window_samples long. It holds real audio from
    pre_trigger_samples before the trigger to real_samples after the window
    start, and zeros after that. A clip with less real audio than that after
    the window start (or a window that would start before the clip) is dropped,
    never padded.
    """

    sample_rate: int = 16000
    window_samples: int = 8000        # 0.50 s model input (arith_spec.md 5.1)
    pre_trigger_samples: int = 800    # 0.05 s of audio before the trigger
    real_samples: int = 6400          # 0.40 s of real audio, then zeros

    @property
    def tail_samples(self) -> int:
        return self.window_samples - self.real_samples

    @property
    def post_trigger_samples(self) -> int:
        return self.real_samples - self.pre_trigger_samples

    def validate(self) -> None:
        assert 0 < self.real_samples <= self.window_samples
        assert 0 <= self.pre_trigger_samples < self.real_samples


@dataclass(frozen=True)
class DetectorConfig:
    """Energy detector of sw/frontend/detector.py (register widths there).

    A sample is a candidate when the short-term energy of the first difference
    is at least 2**ratio_log2 times the mean energy of the preceding
    background_samples, and at least min_energy. A candidate triggers when the
    refractory counter has run out, or (preemption) when its short-term energy
    is more than 2**preempt_log2 times the largest short-term energy seen since
    the last trigger.
    """

    fast_samples: int = 32            # NF: 2 ms short-term window
    background_samples: int = 512     # NB: 32 ms background window, right before the short-term one
    power_shift: int = 4              # p = (d*d) >> power_shift
    ratio_log2: int = 3               # threshold 8x (9 dB) over the background mean
    min_energy: int = 256             # absolute floor on the short-term sum (rejects digital silence)
    refractory_samples: int = 1600    # 0.10 s
    preempt_log2: int = 2             # a 4x (6 dB) louder event retriggers during refractory

    @property
    def warmup_samples(self) -> int:
        """No trigger before both windows hold real samples."""
        return self.fast_samples + self.background_samples - 1

    def validate(self, window: WindowConfig) -> None:
        nb_over_nf = self.background_samples // self.fast_samples
        assert self.background_samples == nb_over_nf * self.fast_samples
        assert nb_over_nf & (nb_over_nf - 1) == 0, "NB/NF must be a power of two (shift, no multiplier)"
        assert 1 <= self.refractory_samples <= window.post_trigger_samples, \
            "refractory period longer than the real audio after the trigger"
        assert self.warmup_samples <= window.pre_trigger_samples


@dataclass(frozen=True)
class GainConfig:
    """Per-window power-of-two gain (identical for every source and in hardware).

    k = clamp(15 - bit_length(max |x|), 0, max_shift) over the real part of the
    window; the window is then x << k (saturated to int16). After the gain the
    peak lies in [16384, 32767] unless k hit max_shift.
    """

    max_shift: int = 10


@dataclass(frozen=True)
class ReferenceConfig:
    """How detector triggers are checked against the dataset's own shot times."""

    match_before_s: float = 0.10      # a trigger may lead the reference by this much (supersonic crack)
    match_after_s: float = 0.05       # or lag it by this much
    c3gd_preroll_s: float = 0.100     # documented C3GD alignment
    # Measured: every c3gd_nj_farm clip has its shot at 0.046 s, not 0.1 s
    # (half-peak onset, 1st-99th percentile 0.038-0.047 s over all 335 clips).
    c3gd_preroll_by_session: dict = field(default_factory=lambda: {"c3gd_nj_farm": 0.046})
    cadre_search_s: float = 0.10      # Cadre onset: first sample >= peak/2 within this long before the peak


@dataclass(frozen=True)
class BackgroundConfig:
    """Per-epoch background sampling (sw/frontend/background.py)."""

    min_native_sr: int = 16000        # drop clips recorded below this (the 55 clips of dataset_audit.md 4.3)
    silence_rms_dbfs: float = -50.0   # reject a window whose real part is quieter than this
    max_zero_run: int = 160           # reject a window holding an exact-zero run longer than 10 ms (padding)
    bg_per_gun: float = 0.5           # background windows per gun window in an epoch
    share_gunmic: float = 0.20        # edge-quiet / cadre-quiet crops, half of them with an event mixed on
    share_gunmic_mixed: float = 0.5   # fraction of the gun-mic share that gets a US8K/ESC-50 event mixed on
    share_triggered: float = 0.40     # windows cut by the detector, same placement as gun windows
    gun_mix_prob: float = 0.30        # gun windows mixed onto US8K/ESC-50 noise
    snr_db: tuple = (0.0, 20.0)       # gun-over-noise SNR range for mixing
    event_gain_db: tuple = (-20.0, 0.0)  # event level relative to its own normalised peak when mixed onto quiet
    min_gunmic_share: float = 0.15    # review gate (runbook Step 1.6 item 5)
    max_tries: int = 20


@dataclass(frozen=True)
class Seeds:
    splits: int = 16
    background_test_sessions: int = 1623
    epochs: int = 2026
    listen: int = 7


@dataclass(frozen=True)
class FrontendConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    seeds: Seeds = field(default_factory=Seeds)
    adc_bits: int = 16

    def validate(self) -> "FrontendConfig":
        self.window.validate()
        self.detector.validate(self.window)
        assert self.adc_bits == 16, "detector register widths are derived for 16-bit samples"
        return self

    def to_dict(self) -> dict:
        return asdict(self)


FRONTEND = FrontendConfig().validate()
