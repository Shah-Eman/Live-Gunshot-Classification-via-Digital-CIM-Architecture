"""Leak-free dataset, ADC sweep front end, and per-epoch sampler (Steps 1.7 & 1.8).

Guarantees:
1. Splits come strictly from splits.json via load_splits (no fallback). Any
   overlap between training/validation paths and splits.test raises ValueError.
2. No decoded or resampled audio is ever written to disk. Clips in splits.cv_pool
   are decoded once in memory per sample rate via manifest.load_audio -> to_int16.
3. At the default 16 kHz / 16-bit / 0.50 s point with augment=False,
   DatasetBundle.sample_train_epoch() is bit-for-bit identical to
   sw/frontend/background.py:EpochBuilder.epoch() (verified in test_train.py).
4. ADC bit-depth truncation (16 / 12 / 10 / 8 / 6 bits) happens BEFORE the
   per-window hardware power-of-two gain (audio.normalise), matching Step 4.7.
5. Waveform augmentations (Step 1.8) operate only on the real audio portion and
   always re-zero the D22 tail so zero-tail length never leaks class or session.
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SW_DIR = HERE.parent
sys.path.insert(0, str(SW_DIR / "frontend"))
sys.path.insert(0, str(SW_DIR / "data"))

from audio import normalise, to_int16  # noqa: E402
from background import EVENT_SOURCES, GUN_MIC_SOURCES, mix, rms_dbfs, zero_run  # noqa: E402
from config import FRONTEND, FrontendConfig, WindowConfig  # noqa: E402
from manifest import load_audio, load_manifest  # noqa: E402
from splits import Splits, load_splits  # noqa: E402
from windows import audio_bounds, cut  # noqa: E402


def make_window_config(sample_rate: int = 16000, window_s: float = 0.50) -> WindowConfig:
    """WindowConfig scaled to (sample_rate, window_s) preserving D22 timing.

    - pre_trigger_s = 0.04 s (D22)
    - real_s = min(0.40 s, window_s): 0.50 s has 0.40 s real + 0.10 s zero tail
      (D22 option a); 0.40 s has 0.40 s real + 0 tail (D22 option c); 0.25 s has
      0.25 s real + 0 tail.
    """
    win_samples = int(round(window_s * sample_rate))
    pre_samples = int(round(0.04 * sample_rate))
    real_s = min(0.40, window_s)
    real_samples = min(int(round(real_s * sample_rate)), win_samples)
    wc = WindowConfig(
        sample_rate=sample_rate,
        window_samples=win_samples,
        pre_trigger_samples=pre_samples,
        real_samples=real_samples,
    )
    wc.validate()
    return wc


def truncate_adc_bits(x: np.ndarray, adc_bits: int) -> np.ndarray:
    """Hardware MSB truncation of signed 16-bit ADC samples to `adc_bits` bits.

    When adc_bits == 16, returns x unchanged. For B in {12, 10, 8, 6}, zeroes
    the (16 - B) LSBs (signed arithmetic right-shift then left-shift), keeping
    samples on the int16 full-scale grid before the barrel-shift window gain.
    """
    if adc_bits == 16:
        return x
    if not (2 <= adc_bits < 16):
        raise ValueError(f"invalid adc_bits={adc_bits}")
    shift = 16 - adc_bits
    return ((x.astype(np.int32) >> shift) << shift).astype(np.int16)


def _decode_one(job: tuple[str, dict, int]) -> tuple[str, np.ndarray]:
    rel, row, sr = job
    return rel, to_int16(load_audio(row, sr=sr))


def _path_seed(rel_path: str, base_seed: int) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{rel_path}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def augment_waveform_float(
    X: np.ndarray,
    real_samples: int,
    rng: np.random.Generator,
    max_shift_samples: int,
) -> np.ndarray:
    """Step 1.8 in-place-safe float32 waveform augmentations on [-1, 1] windows.

    Applies:
    1. Small random time shift (+-max_shift_samples, ~+-5 ms) within real_samples.
    2. Short synthetic outdoor early-reflection FIR reverb (p=0.35) within real_samples.
    3. Light additive white/pink-like sensor noise (p=0.40, -45 to -28 dBFS) within real_samples.
    4. Random gain jitter (+-3 dB), clamped to [-1, 1].
    Always forces X[:, real_samples:] = 0.0 so the D22 zeroed tail stays exact.
    """
    N, L = X.shape
    out = X.copy()
    R = real_samples

    # 1. Time shift within real_samples (replicate edge sample, never zero-pad)
    if max_shift_samples > 0:
        shifts = rng.integers(-max_shift_samples, max_shift_samples + 1, size=N)
        for i in np.flatnonzero(shifts != 0):
            sh = int(shifts[i])
            seg = out[i, :R]
            if sh > 0:
                out[i, sh:R] = seg[: R - sh]
                out[i, :sh] = seg[0]
            else:
                sh_abs = -sh
                out[i, : R - sh_abs] = seg[sh_abs:R]
                out[i, R - sh_abs : R] = seg[-1]

    # 2. Early-reflection outdoor FIR reverb (2 sparse reflections within 4-18 ms)
    rev_mask = rng.random(N) < 0.35
    for i in np.flatnonzero(rev_mask):
        d1 = int(rng.integers(max(4, max_shift_samples // 2), max(8, max_shift_samples * 2)))
        d2 = int(rng.integers(max(8, max_shift_samples * 2), max(16, max_shift_samples * 4)))
        g1 = float(rng.uniform(0.08, 0.25)) * (-1.0 if rng.random() < 0.5 else 1.0)
        g2 = float(rng.uniform(0.04, 0.14)) * (-1.0 if rng.random() < 0.5 else 1.0)
        seg = out[i, :R].copy()
        if d1 < R:
            out[i, d1:R] += g1 * seg[: R - d1]
        if d2 < R:
            out[i, d2:R] += g2 * seg[: R - d2]

    # 3. Light sensor / wind noise within real_samples
    ns_mask = rng.random(N) < 0.40
    n_ns = int(ns_mask.sum())
    if n_ns > 0:
        stds = (10.0 ** (rng.uniform(-45.0, -28.0, size=(n_ns, 1)) / 20.0)).astype(np.float32)
        noise = rng.standard_normal((n_ns, R), dtype=np.float32) * stds
        out[ns_mask, :R] += noise

    # 4. Gain jitter +-3 dB and clamp to [-1, 1]
    gains = (10.0 ** (rng.uniform(-3.0, 3.0, size=(N, 1)) / 20.0)).astype(np.float32)
    out[:, :R] = np.clip(out[:, :R] * gains, -1.0, 1.0)
    if R < L:
        out[:, R:] = 0.0
    return out


class DatasetBundle:
    """In-memory CV pool at one sample_rate, serving any fold, bit depth, or window length."""

    def __init__(
        self,
        manifest_path: str | Path,
        splits_path: str | Path,
        window_index_path: str | Path,
        background_index_path: str | Path,
        gun_npz_path: str | Path | None = None,
        sample_rate: int = 16000,
        workers: int | None = None,
        preload_gun_clips: bool | None = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.splits: Splits = load_splits(splits_path, manifest=self.manifest_path)
        self.classes: list[str] = list(self.splits.raw["classes"])
        self.sample_rate = int(sample_rate)

        root = self.manifest_path.parent
        all_rows = load_manifest(self.manifest_path)
        self.rows_by_path: dict[str, dict] = {
            str(Path(r["path"]).relative_to(root)): r for r in all_rows
        }

        with open(window_index_path, newline="") as f:
            self.gun_index: list[dict] = list(csv.DictReader(f))
        with open(background_index_path, newline="") as f:
            self.bg_index: list[dict] = list(csv.DictReader(f))

        # Safety check: CV pool and test set must be strictly disjoint
        if self.splits.cv_pool & self.splits.test:
            raise ValueError("splits.cv_pool overlaps splits.test")

        self.gun_npz_windows: np.ndarray | None = None
        if gun_npz_path is not None and Path(gun_npz_path).is_file() and sample_rate == 16000:
            d = np.load(gun_npz_path)
            self.gun_npz_windows = d["windows"]
            if len(self.gun_npz_windows) != len(self.gun_index):
                raise ValueError("gun_windows.npz row count mismatch with window_index.csv")

        need_gun_decode = (self.gun_npz_windows is None) if preload_gun_clips is None else preload_gun_clips

        # Decode all usable CV clips once into RAM (never onto disk)
        cv_pool = self.splits.cv_pool
        bc = FRONTEND.background
        bg_paths = {
            r["path"]
            for r in self.bg_index
            if r["path"] in cv_pool and int(r["native_sr"]) >= bc.min_native_sr
        }
        gun_paths = {r["path"] for r in self.gun_index if r["path"] in cv_pool} if need_gun_decode else set()
        to_decode = sorted(bg_paths | gun_paths)

        n_workers = workers or max(1, min(10, (os.cpu_count() or 4) - 2))
        jobs = [(p, self.rows_by_path[p], self.sample_rate) for p in to_decode]
        self.clips: dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for rel, arr in pool.map(_decode_one, jobs):
                self.clips[rel] = arr
        self.bounds: dict[str, tuple[int, int]] = {
            rel: audio_bounds(arr) for rel, arr in self.clips.items()
        }

    def _assert_not_test(self, paths: set[str], label: str) -> None:
        if paths & self.splits.test:
            raise ValueError(f"{label} overlaps the held-out final test set!")
        if paths & set(self.splits.excluded):
            raise ValueError(f"{label} contains excluded (<16 kHz) clips!")

    def _raw_gun_window(self, idx: int, win: WindowConfig) -> np.ndarray:
        """Raw int16 gun window BEFORE normalise and BEFORE ADC bit truncation."""
        if (
            self.gun_npz_windows is not None
            and win.sample_rate == 16000
            and win.window_samples == FRONTEND.window.window_samples
            and win.real_samples == FRONTEND.window.real_samples
            and win.pre_trigger_samples == FRONTEND.window.pre_trigger_samples
        ):
            return self.gun_npz_windows[idx]
        r = self.gun_index[idx]
        x = self.clips[r["path"]]
        # Scale trigger sample from 16 kHz reference index and place with `win`
        trig_sr = int(round(int(r["trigger_sample"]) * win.sample_rate / 16000.0))
        start_sr = trig_sr - win.pre_trigger_samples
        lo, hi = self.bounds[r["path"]]
        start_sr = min(max(start_sr, lo), max(lo, hi - win.real_samples))
        return cut(x, start_sr, win)

    def _usable_bg_pools(self, allowed_paths: set[str], win: WindowConfig):
        bc = FRONTEND.background
        scale = win.sample_rate / 16000.0
        usable = []
        for r in self.bg_index:
            if r["path"] not in allowed_paths or int(r["native_sr"]) < bc.min_native_sr:
                continue
            n_sr = len(self.clips[r["path"]])
            if int(r["n_samples"]) >= FRONTEND.window.real_samples and n_sr >= win.real_samples:
                usable.append(r)
        gunmic = [r for r in usable if r["source"] in GUN_MIC_SOURCES]
        events = [r for r in usable if r["source"] in EVENT_SOURCES]
        general = [r for r in usable if r["source"] not in GUN_MIC_SOURCES]
        triggered = []
        for r in usable:
            x_len = len(self.clips[r["path"]])
            lo, hi = self.bounds[r["path"]]
            for t_str in r["triggers"].split():
                t_sr = int(round(int(t_str) * scale))
                s_sr = t_sr - win.pre_trigger_samples
                if s_sr >= lo and s_sr + win.real_samples <= min(hi, x_len):
                    triggered.append((r, t_sr))
        event_triggered = [(r, t) for r, t in triggered if r["source"] in EVENT_SOURCES]
        return usable, gunmic, events, general, triggered, event_triggered

    @staticmethod
    def _ok(w: np.ndarray, win: WindowConfig) -> bool:
        bc = FRONTEND.background
        max_zr = max(1, int(round(bc.max_zero_run * win.sample_rate / 16000.0)))
        return rms_dbfs(w[: win.real_samples]) >= bc.silence_rms_dbfs and zero_run(w[: win.real_samples]) <= max_zr

    def _crop(self, rng: np.random.Generator, pool: list[dict], win: WindowConfig) -> tuple[np.ndarray, dict]:
        real = win.real_samples
        for _ in range(FRONTEND.background.max_tries):
            r = pool[rng.integers(len(pool))]
            x = self.clips[r["path"]]
            lo, hi = self.bounds[r["path"]]
            if hi - lo < real:
                continue
            start = int(rng.integers(lo, hi - real + 1))
            w = cut(x, start, win)
            if self._ok(w, win):
                return w, {"path": r["path"], "source": r["source"], "start": start}
        raise RuntimeError("no non-silent crop found; check background pool")

    def _triggered(
        self, rng: np.random.Generator, pool: list[tuple[dict, int]], win: WindowConfig
    ) -> tuple[np.ndarray, dict]:
        pre = win.pre_trigger_samples
        for _ in range(FRONTEND.background.max_tries):
            r, t = pool[rng.integers(len(pool))]
            w = cut(self.clips[r["path"]], t - pre, win)
            if self._ok(w, win):
                return w, {"path": r["path"], "source": r["source"], "start": t - pre, "trigger": t}
        raise RuntimeError("no non-silent triggered window found")

    def prepare_split(
        self,
        train_paths: set[str],
        val_paths: set[str],
        window_s: float = 0.50,
    ) -> dict:
        """Pre-index gun and background pools for one (train_paths, val_paths, window_s)."""
        self._assert_not_test(train_paths, "train_paths")
        self._assert_not_test(val_paths, "val_paths")
        if train_paths & val_paths:
            raise ValueError("train_paths and val_paths overlap!")

        win = make_window_config(self.sample_rate, window_s)
        train_gun_rows = [i for i, r in enumerate(self.gun_index) if r["path"] in train_paths]
        val_gun_rows = [i for i, r in enumerate(self.gun_index) if r["path"] in val_paths]
        _, gunmic, events, general, triggered, event_triggered = self._usable_bg_pools(train_paths, win)
        val_usable, _, _, _, val_triggered, _ = self._usable_bg_pools(val_paths, win)

        val_trig_by_path: dict[str, list[int]] = {}
        for r, t in val_triggered:
            val_trig_by_path.setdefault(r["path"], []).append(t)

        return {
            "win": win,
            "train_gun_rows": train_gun_rows,
            "val_gun_rows": val_gun_rows,
            "gunmic": gunmic,
            "events": events,
            "general": general,
            "triggered": triggered,
            "event_triggered": event_triggered,
            "val_usable": val_usable,
            "val_trig_by_path": val_trig_by_path,
        }

    def sample_train_epoch(
        self,
        split_ctx: dict,
        epoch: int,
        adc_bits: int = 16,
        seed: int | None = None,
        augment: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample one epoch of training windows as float32 in [-1, 1] (or int16-exact at 16-bit).

        When sample_rate=16000, window_s=0.50, adc_bits=16, augment=False, the
        underlying int16 array X_i16 is 100% bit-identical to EpochBuilder.epoch(epoch).
        Returns (X_f32 [N, L], y [N]).
        """
        X_i16, y = self.sample_train_epoch_int16(split_ctx, epoch, adc_bits=adc_bits, seed=seed)
        X_f32 = (X_i16.astype(np.float32)) / 32768.0
        if augment:
            win: WindowConfig = split_ctx["win"]
            aug_rng = np.random.default_rng([seed if seed is not None else FRONTEND.seeds.epochs, epoch, 991])
            max_shift = int(round(0.005 * win.sample_rate))  # +-5 ms
            X_f32 = augment_waveform_float(X_f32, win.real_samples, aug_rng, max_shift)
        return X_f32, y

    def sample_train_epoch_int16(
        self,
        split_ctx: dict,
        epoch: int,
        adc_bits: int = 16,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        bc, gain = FRONTEND.background, FRONTEND.gain
        win: WindowConfig = split_ctx["win"]
        ep_seed = FRONTEND.seeds.epochs if seed is None else seed
        rng = np.random.default_rng([ep_seed, epoch])

        gun_rows = split_ctx["train_gun_rows"]
        n_gun = len(gun_rows)
        n_bg = int(round(bc.bg_per_gun * n_gun))
        n_gm = int(np.ceil(bc.share_gunmic * n_bg))
        n_gm_ev = int(round(bc.share_gunmic_mixed * n_gm))
        n_trig = int(round(bc.share_triggered * n_bg))
        kinds = ["gunmic_event"] * n_gm_ev + ["gunmic"] * (n_gm - n_gm_ev) + ["triggered"] * n_trig
        kinds += ["random"] * (n_bg - len(kinds))

        X = np.zeros((n_gun + n_bg, win.window_samples), np.int16)
        y = np.zeros(n_gun + n_bg, np.int64)

        events = split_ctx["events"]
        gunmic = split_ctx["gunmic"]
        general = split_ctx["general"]
        triggered = split_ctx["triggered"]
        event_triggered = split_ctx["event_triggered"]

        for j, i in enumerate(gun_rows):
            g = self._raw_gun_window(i, win)
            r = self.gun_index[i]
            if rng.random() < bc.gun_mix_prob:
                noise, _ = self._crop(rng, events, win)
                snr = float(rng.uniform(*bc.snr_db))
                g = mix(g, noise, snr, win.real_samples)
            g = truncate_adc_bits(g, adc_bits)
            X[j], _ = normalise(g, win, gain)
            y[j] = self.classes.index(r["class"])

        bg_cls = self.classes.index("background")
        for j, kind in enumerate(kinds, start=n_gun):
            if kind == "gunmic":
                w, _ = self._crop(rng, gunmic, win)
            elif kind == "gunmic_event":
                q, _ = self._crop(rng, gunmic, win)
                ev, _ = self._triggered(rng, event_triggered, win)
                snr = float(rng.uniform(*bc.snr_db))
                w = mix(ev, q, snr, win.real_samples)
            elif kind == "triggered":
                w, _ = self._triggered(rng, triggered, win)
            elif kind == "random":
                w, _ = self._crop(rng, general, win)
            else:
                raise ValueError(kind)
            w = truncate_adc_bits(w, adc_bits)
            X[j], _ = normalise(w, win, gain)
            y[j] = bg_cls

        order = rng.permutation(len(y))
        return X[order], y[order]

    def build_val_set(
        self,
        split_ctx: dict,
        adc_bits: int = 16,
        val_seed: int = 2026,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Deterministic unmixed validation set: all val gun windows + 1 window per usable val bg clip."""
        win: WindowConfig = split_ctx["win"]
        gain = FRONTEND.gain
        bc = FRONTEND.background

        val_gun_rows = split_ctx["val_gun_rows"]
        val_usable = split_ctx["val_usable"]
        val_trig_by_path = split_ctx["val_trig_by_path"]

        xs: list[np.ndarray] = []
        ys: list[int] = []
        sources: list[str] = []

        for i in val_gun_rows:
            r = self.gun_index[i]
            g = truncate_adc_bits(self._raw_gun_window(i, win), adc_bits)
            gn, _ = normalise(g, win, gain)
            xs.append(gn)
            ys.append(self.classes.index(r["class"]))
            sources.append(r["source"])

        bg_cls = self.classes.index("background")
        for r in val_usable:
            p = r["path"]
            rng = np.random.default_rng(_path_seed(p, val_seed))
            x = self.clips[p]
            trigs = val_trig_by_path.get(p)
            picked = None
            if trigs and rng.random() < bc.share_triggered:
                t = trigs[int(rng.integers(len(trigs)))]
                cand = cut(x, t - win.pre_trigger_samples, win)
                if self._ok(cand, win):
                    picked = cand
            if picked is None:
                lo, hi = self.bounds[p]
                if hi - lo >= win.real_samples:
                    for _ in range(bc.max_tries):
                        s = int(rng.integers(lo, hi - win.real_samples + 1))
                        cand = cut(x, s, win)
                        if self._ok(cand, win):
                            picked = cand
                            break
                    if picked is None:
                        # Deterministic fallback at highest-energy frame if clip is quiet
                        s = lo
                        picked = cut(x, s, win)
            if picked is not None:
                wn, _ = normalise(truncate_adc_bits(picked, adc_bits), win, gain)
                xs.append(wn)
                ys.append(bg_cls)
                sources.append(r["source"])

        X_f32 = np.stack(xs).astype(np.float32) / 32768.0
        return X_f32, np.asarray(ys, dtype=np.int64), sources

    def build_pretrigger_sets(
        self,
        train_paths: set[str],
        val_paths: set[str],
        adc_bits: int = 16,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
        """Step 1.8 item 3: 4-class firearm leakage check on ONLY pre-trigger noise.

        Extracts [0 : pre_trigger_samples - guard] strictly before the shot
        (guard = 2 ms = NF samples so zero fast-window shot onset can enter).
        Normalises each snippet independently with the hardware power-of-two gain.
        """
        self._assert_not_test(train_paths, "pretrigger train_paths")
        self._assert_not_test(val_paths, "pretrigger val_paths")
        win = make_window_config(self.sample_rate, 0.50)
        guard = max(1, int(round(0.002 * self.sample_rate)))  # 2 ms guard before trigger
        pre_len = win.pre_trigger_samples - guard
        if pre_len < 64:
            raise ValueError(f"pre_len={pre_len} too short")
        pre_win = WindowConfig(
            sample_rate=self.sample_rate,
            window_samples=pre_len,
            pre_trigger_samples=0,
            real_samples=pre_len,
        )
        gun_classes = [c for c in self.classes if c != "background"]

        def _collect(paths: set[str]) -> tuple[np.ndarray, np.ndarray]:
            xs, ys = [], []
            for i, r in enumerate(self.gun_index):
                if r["path"] not in paths:
                    continue
                raw = self._raw_gun_window(i, win)[:pre_len].copy()
                raw = truncate_adc_bits(raw, adc_bits)
                normed, _ = normalise(raw, pre_win, FRONTEND.gain)
                xs.append(normed)
                ys.append(gun_classes.index(r["class"]))
            return np.stack(xs).astype(np.float32) / 32768.0, np.asarray(ys, dtype=np.int64)

        X_tr, y_tr = _collect(train_paths)
        X_va, y_va = _collect(val_paths)
        return X_tr, y_tr, X_va, y_va, gun_classes, pre_len
