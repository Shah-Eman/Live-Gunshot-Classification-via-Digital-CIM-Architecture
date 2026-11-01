"""Windows cut exactly as the hardware cuts them (runbook Step 1.6, items 2-5; Decision D22).

Every window, from every source and in the live front end, has the same layout:

    sample  0 .............. REAL-1 | REAL ........ INPUT-1
            real audio (0.35 s)     | zeros (0.15 s)
            starts PRE samples (40 ms) before the trigger

    INPUT = 8000 samples (0.5 s at 16 kHz), unchanged, so arith_spec.md §5.1 is unchanged.

Gun windows are placed by detector.py (never by the dataset's timestamps; those only
check that the detector found the right shot). Background windows are random crops
with the same real/zero layout. A clip that has less than REAL samples after the
window start, or fewer than PRE samples before the trigger, is dropped, never padded.

Why 40 ms / 0.35 s and not the runbook's 50 ms / 0.40 s: measured on the 0.459 s C3GD
clips, the shot is at 45.8-47.3 ms in c3gd_nj_farm and at 40-141 ms in
c3gd_ny_gravel_pit (not 0.1 s). 50 ms before the trigger keeps 0 of 66 nj_farm clips;
40 ms / 0.35 s keeps 62/66 and 64/68 (see docs/decisions.md, D22).

Level: every window gets the same integer power-of-two gain rule (`normalize`): shift
left by the largest k <= MAX_GAIN_SHIFT that keeps the peak of the real part in range.
It removes device gain differences without a divider, and the RTL can do it with a
peak register and a barrel shifter after the capture.

Pipeline for one gun clip (float only until the ADC model `to_pcm`):
    load_audio (16 kHz mono) -> clamp [-1, 1] -> to_pcm (int16) -> detect -> place
    -> slice REAL samples -> normalize -> append INPUT-REAL zeros  ==> int16[8000]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import detector as det                      # noqa: E402
from manifest import load_audio             # noqa: E402

SR = det.SR
INPUT = 8000                # model input, samples (0.5 s); arith_spec.md §5.1
PRE = 640                   # samples of real audio before the trigger (40 ms)
REAL = 5600                 # samples of real audio in a window (0.35 s)
TAIL = INPUT - REAL         # zeroed tail (0.15 s)
BITS = 16
MAX_GAIN_SHIFT = 6          # normalize by at most 2^6 (36 dB)

# Background crops: "almost silent" means mostly digital zeros (ESC-50 padding) or no signal.
BG_MAX_ZERO_FRAC = 0.25
BG_MIN_RMS_DBFS = -75.0
BG_TRIES = 10
BG_MIN_SR = 16000           # drop background clips recorded below 16 kHz (audit §4, finding 3)

GUN_MIC_SOURCES = ("edge-quiet", "cadre-quiet")     # background recorded on the gun microphones
NOISE_SOURCES = ("us8k", "esc50")                    # noise / events mixed onto other clips

# Matching triggers to the dataset's own shot times (only to check the detector).
EDGE_MATCH = int(0.05 * SR)          # edge: trigger within 50 ms of a labelled shot time
C3GD_SHOT = (int(0.03 * SR), int(0.15 * SR))   # C3GD: the shot sits 46-141 ms into each clip
PEAK_LEAD = int(0.10 * SR)           # Cadre: trigger at most 100 ms before the clip's peak
                                     #        (loudest sample or loudest 4 ms of energy)
PEAK_LAG = int(0.01 * SR)            #        and at most 10 ms after it


def clamp(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def load_pcm(row: dict) -> np.ndarray:
    """One manifest row as int16-range PCM at 16 kHz (resampled, clamped, quantized)."""
    return det.to_pcm(clamp(load_audio(row)), BITS)


def place(trigger: int, n: int) -> int | None:
    """Window start for a trigger in a clip of n samples, or None if it does not fit."""
    start = trigger - PRE
    if start < 0 or start + REAL > n:
        return None
    return start


def normalize(real: np.ndarray, bits: int = BITS) -> tuple[np.ndarray, int]:
    """Integer power-of-two gain: the same rule for every window. Returns (scaled, k)."""
    full = (1 << (bits - 1)) - 1
    peak = int(np.max(np.abs(real.astype(np.int64)))) if real.size else 0
    k = 0
    while k < MAX_GAIN_SHIFT and peak > 0 and (peak << (k + 1)) <= full:
        k += 1
    out = np.clip(real.astype(np.int64) << k, -full - 1, full)
    return out.astype(np.int32), k


def finish(real: np.ndarray) -> tuple[np.ndarray, int]:
    """REAL samples of int PCM -> the int16 input window (normalized, tail zeroed)."""
    assert real.shape == (REAL,), real.shape
    scaled, k = normalize(real)
    w = np.zeros(INPUT, dtype=np.int16)
    w[:REAL] = scaled
    return w, k


def to_model(windows: np.ndarray) -> np.ndarray:
    """int16 windows [..., 8000] -> float32 [..., 8000, 1] in [-1, 1) for Keras."""
    return (np.asarray(windows, dtype=np.float32) / (1 << (BITS - 1)))[..., None]


# --------------------------------------------------------------------------- reference shots

def edge_stem(row: dict) -> str:
    """Kabealo filename stem of an edge row (key of gunshot-audio-all-metadata.csv)."""
    op = row.get("orig_path", "")
    if op.startswith("edge:"):
        return Path(op.split(":", 1)[1]).stem
    return Path(row["path"]).name.split("__", 2)[-1].rsplit(".", 1)[0]


def match_shot(row: dict, pcm: np.ndarray, trigger: int, edge_shots: dict | None) -> int | None:
    """Which known shot does this trigger belong to? Shot index, -1 for none, None if no reference.

    C3GD and Cadre clips hold one shot (index 0); edge clips hold the labelled times.
    """
    src = row["source"]
    if src == "c3gd":
        return 0 if C3GD_SHOT[0] <= trigger <= C3GD_SHOT[1] else -1
    if src == "cadre":
        # the shot is the loudest sample or the loudest 4 ms (a click or echo can win either one)
        refs = (int(np.argmax(np.abs(pcm))), int(np.argmax(det.energy(pcm))))
        return 0 if any(r - PEAK_LEAD <= trigger <= r + PEAK_LAG for r in refs) else -1
    if src == "edge" and edge_shots is not None:
        shots = edge_shots.get(edge_stem(row))
        if shots:
            d = [abs(trigger - int(round(t * SR))) for t in shots]
            k = int(np.argmin(d))
            return k if d[k] <= EDGE_MATCH else -1
    return None


def reference_count(row: dict, edge_shots: dict | None) -> int | None:
    """Number of known shots in a clip (1 for C3GD and Cadre; labelled count for edge)."""
    if row["source"] in ("c3gd", "cadre"):
        return 1
    if row["source"] == "edge" and edge_shots is not None:
        s = edge_shots.get(edge_stem(row))
        return len(s) if s else None
    return None


# --------------------------------------------------------------------------- gun windows

@dataclass
class GunCut:
    path: str
    triggers: list[int]
    matched: list[bool | None]          # per trigger: matches a known shot?
    starts: list[int | None]            # per trigger: window start, None = did not fit
    windows: list[np.ndarray] = field(default_factory=list)   # int16[8000] for kept triggers
    kept: list[int] = field(default_factory=list)             # trigger index of each window
    gains: list[int] = field(default_factory=list)
    n_ref: int | None = None            # known shots in the clip
    n_found: int = 0                    # known shots with a matching trigger
    n_samples: int = 0                  # clip length at 16 kHz


def cut_gun(row: dict, edge_shots: dict | None = None, p: det.Params = det.DEFAULT,
            keep_unverified: bool = True) -> GunCut:
    """Every detector trigger of one gun clip, and the windows it gives.

    One window per known shot: the last trigger matched to it (a re-trigger is always
    louder than the trigger it replaces, so the last one is the shot itself, not a weak
    precursor). Triggers that match no known shot (noise before the shot, a second event)
    are not labelled as gunshots. Sources without a reference keep every trigger that fits
    (keep_unverified). A trigger whose window does not fit in the clip is dropped.
    """
    pcm = load_pcm(row)
    trig = det.detect(pcm, p)
    cut = GunCut(path=row["path"], triggers=trig, matched=[], starts=[],
                 n_ref=reference_count(row, edge_shots), n_samples=len(pcm))
    shot_of = [match_shot(row, pcm, t, edge_shots) for t in trig]
    last = {}
    for i, k in enumerate(shot_of):
        if k is not None and k >= 0:
            last[k] = i
    for i, t in enumerate(trig):
        k = shot_of[i]
        cut.matched.append(None if k is None else k >= 0)
        s = place(t, len(pcm))
        cut.starts.append(s)
        use = (k is None and keep_unverified) or (k is not None and k >= 0 and last[k] == i)
        if s is None or not use:
            continue
        w, g = finish(pcm[s:s + REAL])
        cut.windows.append(w)
        cut.kept.append(i)
        cut.gains.append(g)
    if cut.n_ref is not None:
        cut.n_found = len(last)
    return cut


# --------------------------------------------------------------------------- background

class ClipCache:
    """Float 16 kHz clips kept in memory, so per-epoch background crops do not reread disk."""

    def __init__(self, max_items: int = 50000):
        self.d: dict[str, np.ndarray] = {}
        self.bad: set[str] = set()
        self.max_items = max_items

    def get(self, row: dict) -> np.ndarray | None:
        """The clip, or None if it cannot be read (missing or corrupt file; reported once)."""
        p = row["path"]
        if p in self.bad:
            return None
        x = self.d.get(p)
        if x is None:
            try:
                x = clamp(load_audio(row))
            except Exception as e:                       # noqa: BLE001 - any unreadable file
                self.bad.add(p)
                print(f"warning: skipping unreadable clip {p}: {type(e).__name__}", file=sys.stderr)
                return None
            if len(self.d) < self.max_items:
                self.d[p] = x
        return x


def silent(real_pcm: np.ndarray) -> bool:
    if np.mean(real_pcm == 0) > BG_MAX_ZERO_FRAC:
        return True
    rms = np.sqrt(np.mean(real_pcm.astype(np.float64) ** 2)) / (1 << (BITS - 1))
    return 20 * np.log10(rms + 1e-12) < BG_MIN_RMS_DBFS


def random_crop(x: np.ndarray | None, rng: np.random.Generator) -> np.ndarray | None:
    """REAL float samples from a random place in x that are not almost silent, or None."""
    if x is None or len(x) < REAL:
        return None
    for _ in range(BG_TRIES):
        s = int(rng.integers(0, len(x) - REAL + 1))
        seg = x[s:s + REAL]
        if not silent(det.to_pcm(seg, BITS)):
            return seg
    return None


def mix(a: np.ndarray, b: np.ndarray, snr_db: float) -> np.ndarray:
    """a + b scaled so that rms(a)/rms(b) = snr_db; clamped. Float, before the ADC model."""
    ra = np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12
    rb = np.sqrt(np.mean(b.astype(np.float64) ** 2)) + 1e-12
    return clamp(a + b * (ra / rb) / (10 ** (snr_db / 20)))


def usable_background(rows: list[dict]) -> list[dict]:
    """Background rows the sampler may use: full band only (drop clips below 16 kHz)."""
    return [r for r in rows if r["class"] == "background" and int(r["sample_rate"]) >= BG_MIN_SR]


class BackgroundSampler:
    """A fresh set of background windows every epoch (runbook Step 1.6, item 5).

    Per epoch, at least `gunmic_frac` (default 0.20, the runbook minimum is 0.15) of the
    windows come from the gun microphones: half plain quiet stretches, half with a
    UrbanSound8K / ESC-50 event mixed onto the stretch. The rest are crops of any other
    background clip. Only rows of the split it is given are used, so a test session
    never reaches training.
    """

    def __init__(self, rows: list[dict], seed: int = 0, gunmic_frac: float = 0.20,
                 event_mix_frac: float = 0.5, snr_db: tuple[float, float] = (-5.0, 10.0),
                 cache: ClipCache | None = None):
        bg = usable_background(rows)
        self.gunmic = [r for r in bg if r["source"] in GUN_MIC_SOURCES]
        self.noise = [r for r in bg if r["source"] in NOISE_SOURCES]
        self.other = [r for r in bg if r["source"] not in GUN_MIC_SOURCES]
        if not self.gunmic:
            raise ValueError("no gun-microphone background (edge-quiet / cadre-quiet) in these rows")
        self.rng = np.random.default_rng(seed)
        self.gunmic_frac, self.event_mix_frac, self.snr_db = gunmic_frac, event_mix_frac, snr_db
        self.cache = cache or ClipCache()

    def _crop(self, pool: list[dict]) -> tuple[np.ndarray, dict]:
        for _ in range(50):
            r = pool[int(self.rng.integers(len(pool)))]
            seg = random_crop(self.cache.get(r), self.rng)
            if seg is not None:
                return seg, r
        raise RuntimeError("could not find a non-silent background crop in 50 clips")

    def epoch(self, n: int) -> tuple[np.ndarray, list[dict]]:
        """n background windows (int16 [n, 8000]) and their provenance."""
        n_gm = int(np.ceil(self.gunmic_frac * n))
        out = np.zeros((n, INPUT), dtype=np.int16)
        info = []
        for i in range(n):
            if i < n_gm:
                seg, r = self._crop(self.gunmic)
                rec = {"path": r["path"], "source": r["source"], "gun_mic": True, "mixed_with": ""}
                if self.noise and self.rng.random() < self.event_mix_frac:
                    ev, er = self._crop(self.noise)
                    seg = mix(seg, ev, float(self.rng.uniform(*self.snr_db)))
                    rec["mixed_with"] = er["path"]
            else:
                seg, r = self._crop(self.other)
                rec = {"path": r["path"], "source": r["source"], "gun_mic": False, "mixed_with": ""}
            out[i], _ = finish(det.to_pcm(seg, BITS))
            info.append(rec)
        order = self.rng.permutation(n)
        return out[order], [info[j] for j in order]


def mix_gun_onto_noise(gun_window: np.ndarray, noise_rows: list[dict], rng: np.random.Generator,
                       snr_db: tuple[float, float] = (0.0, 20.0), cache: ClipCache | None = None
                       ) -> np.ndarray:
    """Augmentation: a gun window's real part mixed onto UrbanSound8K / ESC-50 noise.

    Keeps the window layout (same real part, same zero tail, same normalize), so the
    noise type never identifies the class. noise_rows must come from the same split.
    """
    cache = cache or ClipCache()
    real = gun_window[:REAL].astype(np.float32) / (1 << (BITS - 1))
    for _ in range(50):
        r = noise_rows[int(rng.integers(len(noise_rows)))]
        seg = random_crop(cache.get(r), rng)
        if seg is not None:
            w, _ = finish(det.to_pcm(mix(real, seg, float(rng.uniform(*snr_db))), BITS))
            return w
    return gun_window
