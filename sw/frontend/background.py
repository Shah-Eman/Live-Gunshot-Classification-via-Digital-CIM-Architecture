"""Training data for one epoch: gun windows plus freshly sampled background (Step 1.6, item 5).

This is the training-data path Steps 1.7 and 1.8 call. Every epoch draws new
background windows, and every window, gun or background, leaves through the
same gain (audio.normalise) with the same D22 placement and zeroed tail.

Background kinds (shares from config.BackgroundConfig):

- gunmic:       a crop of an edge-quiet / cadre-quiet stretch (the gun microphones);
- gunmic_event: the same, with a US8K / ESC-50 event mixed on top
                (gunmic + gunmic_event >= 15 % of each epoch's background);
- triggered:    a window cut at a detector trigger of a background clip, with the
                same placement as the gun windows, so "energy jump at sample 800"
                does not identify the gun classes;
- random:       a random crop of any other background clip.

Every background window is rejected and redrawn if its real part is nearly
silent or holds a run of exact zeros longer than max_zero_run (ESC-50 pads its
clips with zeros). Gun windows are mixed onto US8K / ESC-50 noise with
probability gun_mix_prob, so the noise type never identifies the class.

Only clips of the given training paths are used, for the gun windows and for
every background, event and noise clip. Randomness comes from
numpy.random.default_rng([seed, epoch]); the seed is recorded in the metadata.

    b = EpochBuilder.from_files(manifest, splits_json, window_index, gun_npz, background_index, fold=0)
    X, y, meta = b.epoch(0)        # X int16 [N, 8000] after the gain, y class index (manifest.classes order)
"""
from __future__ import annotations

import csv
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from audio import load_clip, normalise  # noqa: E402
from config import FRONTEND, FrontendConfig  # noqa: E402
from manifest import load_manifest  # noqa: E402
from splits import load_splits  # noqa: E402
from windows import audio_bounds, cut  # noqa: E402

GUN_MIC_SOURCES = ("edge-quiet", "cadre-quiet")
EVENT_SOURCES = ("us8k", "esc50")


def zero_run(x: np.ndarray) -> int:
    z = np.concatenate(([0], (x == 0).astype(np.int8), [0]))
    d = np.diff(z)
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((e - s).max()) if len(s) else 0


def rms_dbfs(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean((x.astype(np.float64) / 32768.0) ** 2))) if len(x) else 0.0
    return 20 * np.log10(max(r, 1e-10))


def mix(base: np.ndarray, add: np.ndarray, snr_db: float, real: int) -> np.ndarray:
    """base + add scaled so that power(base) / power(add) = snr_db over the real part. Augmentation only."""
    b, a = base.astype(np.float64), add.astype(np.float64)
    pb, pa = np.mean(b[:real] ** 2), np.mean(a[:real] ** 2)
    g = np.sqrt(pb / (pa * 10 ** (snr_db / 10))) if pa > 0 else 0.0
    return np.clip(np.rint(b + g * a), -32768, 32767).astype(np.int16)


class ClipCache:
    """Bounded in-memory cache of decoded clips (never on disk)."""

    def __init__(self, rows_by_path: dict, cfg: FrontendConfig, size: int = 256):
        self.rows, self.cfg, self.size = rows_by_path, cfg, size
        self.data: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, rel: str) -> np.ndarray:
        if rel in self.data:
            self.data.move_to_end(rel)
            return self.data[rel]
        x = load_clip(self.rows[rel], self.cfg.window)
        self.data[rel] = x
        if len(self.data) > self.size:
            self.data.popitem(last=False)
        return x


class EpochBuilder:
    def __init__(self, rows_by_path: dict, classes: list[str], train_paths: set[str], gun_index: list[dict],
                 gun_windows: np.ndarray, bg_index: list[dict], cfg: FrontendConfig = FRONTEND, seed: int | None = None):
        self.cfg, self.classes = cfg, classes
        self.seed = cfg.seeds.epochs if seed is None else seed
        bc, win = cfg.background, cfg.window
        self.gun_rows = [i for i, r in enumerate(gun_index) if r["path"] in train_paths]
        self.gun_index, self.gun_windows = gun_index, gun_windows
        usable = [r for r in bg_index if r["path"] in train_paths and int(r["native_sr"]) >= bc.min_native_sr
                  and int(r["n_samples"]) >= win.real_samples]
        self.gunmic = [r for r in usable if r["source"] in GUN_MIC_SOURCES]
        self.events = [r for r in usable if r["source"] in EVENT_SOURCES]
        self.general = [r for r in usable if r["source"] not in GUN_MIC_SOURCES]
        self.triggered = [(r, int(t)) for r in usable for t in r["triggers"].split()]
        self.event_triggered = [(r, t) for r, t in self.triggered if r["source"] in EVENT_SOURCES]
        for name in ("gun_rows", "gunmic", "events", "general", "triggered", "event_triggered"):
            if not getattr(self, name):
                raise ValueError(f"no {name} in the training paths")
        self.cache = ClipCache(rows_by_path, cfg)

    @classmethod
    def from_files(cls, manifest: str | Path, splits_json: str | Path, window_index: str | Path,
                   gun_npz: str | Path, background_index: str | Path, fold: int | None = None,
                   train_paths: set[str] | None = None, cfg: FrontendConfig = FRONTEND, seed: int | None = None):
        """Builder for CV fold `fold` (its training side), or for explicit train_paths."""
        s = load_splits(splits_json, manifest=manifest)               # no splits.json: no training data
        if train_paths is None:
            if fold is None:
                raise ValueError("give a fold or explicit train_paths")
            train_paths = s.fold(fold)[0]
        if train_paths & s.test:
            raise ValueError("training paths overlap the final test set")
        root = Path(manifest).resolve().parent
        rows = {str(Path(r["path"]).relative_to(root)): r for r in load_manifest(manifest)}
        with open(window_index, newline="") as f:
            gun_index = list(csv.DictReader(f))
        with open(background_index, newline="") as f:
            bg_index = list(csv.DictReader(f))
        d = np.load(gun_npz)
        X = d["windows"]
        if len(X) != len(gun_index):
            raise ValueError(f"{gun_npz} has {len(X)} windows, {window_index} has {len(gun_index)} rows")
        return cls(rows, s.raw["classes"], train_paths, gun_index, X, bg_index, cfg, seed)

    # -- background windows ---------------------------------------------------
    def _ok(self, w: np.ndarray) -> bool:
        bc, real = self.cfg.background, self.cfg.window.real_samples
        return rms_dbfs(w[:real]) >= bc.silence_rms_dbfs and zero_run(w[:real]) <= bc.max_zero_run

    def _crop(self, rng, pool: list[dict]) -> tuple[np.ndarray, dict]:
        real = self.cfg.window.real_samples
        for _ in range(self.cfg.background.max_tries):
            r = pool[rng.integers(len(pool))]
            x = self.cache.get(r["path"])
            lo, hi = audio_bounds(x)
            if hi - lo < real:
                continue
            start = int(rng.integers(lo, hi - real + 1))
            w = cut(x, start, self.cfg.window)
            if self._ok(w):
                return w, {"path": r["path"], "source": r["source"], "start": start}
        raise RuntimeError("no non-silent crop found; check the background pool")

    def _triggered(self, rng, pool: list[tuple[dict, int]]) -> tuple[np.ndarray, dict]:
        pre = self.cfg.window.pre_trigger_samples
        for _ in range(self.cfg.background.max_tries):
            r, t = pool[rng.integers(len(pool))]
            w = cut(self.cache.get(r["path"]), t - pre, self.cfg.window)
            if self._ok(w):
                return w, {"path": r["path"], "source": r["source"], "start": t - pre, "trigger": t}
        raise RuntimeError("no non-silent triggered window found")

    def background(self, rng, kind: str) -> tuple[np.ndarray, dict]:
        bc, real = self.cfg.background, self.cfg.window.real_samples
        if kind == "gunmic":
            w, m = self._crop(rng, self.gunmic)
        elif kind == "gunmic_event":
            q, m = self._crop(rng, self.gunmic)
            ev, me = self._triggered(rng, self.event_triggered)
            snr = float(rng.uniform(*bc.snr_db))
            w = mix(ev, q, snr, real)
            m = {**m, "mixed_with": me["path"], "snr_db": round(snr, 2)}
        elif kind == "triggered":
            w, m = self._triggered(rng, self.triggered)
        elif kind == "random":
            w, m = self._crop(rng, self.general)
        else:
            raise ValueError(kind)
        return w, {**m, "kind": kind}

    def epoch(self, e: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        bc, win, gain = self.cfg.background, self.cfg.window, self.cfg.gain
        rng = np.random.default_rng([self.seed, e])
        n_gun = len(self.gun_rows)
        n_bg = int(round(bc.bg_per_gun * n_gun))
        n_gm = int(np.ceil(bc.share_gunmic * n_bg))
        n_gm_ev = int(round(bc.share_gunmic_mixed * n_gm))
        n_trig = int(round(bc.share_triggered * n_bg))
        kinds = ["gunmic_event"] * n_gm_ev + ["gunmic"] * (n_gm - n_gm_ev) + ["triggered"] * n_trig
        kinds += ["random"] * (n_bg - len(kinds))

        X = np.zeros((n_gun + n_bg, win.window_samples), np.int16)
        y = np.zeros(n_gun + n_bg, np.int64)
        meta: list[dict] = []
        for j, i in enumerate(self.gun_rows):
            g = self.gun_windows[i]
            r = self.gun_index[i]
            m = {"kind": "gun", "path": r["path"], "source": r["source"], "start": int(r["window_start"]),
                 "trigger": int(r["trigger_sample"])}
            if rng.random() < bc.gun_mix_prob:
                noise, mn = self._crop(rng, self.events)
                snr = float(rng.uniform(*bc.snr_db))
                g = mix(g, noise, snr, win.real_samples)
                m.update(kind="gun_mixed", mixed_with=mn["path"], snr_db=round(snr, 2))
            X[j], m["gain_shift"] = normalise(g, win, gain)
            y[j] = self.classes.index(r["class"])
            meta.append(m)
        bg = self.classes.index("background")
        for j, kind in enumerate(kinds, start=n_gun):
            w, m = self.background(rng, kind)
            X[j], m["gain_shift"] = normalise(w, win, gain)
            y[j] = bg
            meta.append(m)
        order = rng.permutation(len(y))
        for m in meta:
            m["seed"], m["epoch"] = self.seed, e
        return X[order], y[order], [meta[i] for i in order]
