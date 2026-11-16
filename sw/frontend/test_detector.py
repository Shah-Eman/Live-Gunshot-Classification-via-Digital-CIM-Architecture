"""detector.detect() and detector.detect_reference() must agree sample for sample.

Real audio comes from CIM_MANIFEST (a manifest.csv: whole clips, every source)
or, when the dataset is no longer on this machine, from CIM_GUN_WINDOWS (the
gun_windows.npz of cut_windows.py). With neither set the real-clip test fails:
it is never skipped.

    CIM_MANIFEST=../Dataset-Final/manifest.csv python -m unittest discover -s sw/frontend -p 'test_*.py'
"""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from config import FRONTEND, DetectorConfig, WindowConfig  # noqa: E402
from detector import detect, detect_reference, register_widths  # noqa: E402

CFG = FRONTEND.detector
CLIPS_PER_SOURCE = 12


def real_signals() -> list[tuple[str, np.ndarray]]:
    man, npz = os.environ.get("CIM_MANIFEST"), os.environ.get("CIM_GUN_WINDOWS")
    if man and Path(man).is_file():
        from audio import load_clip
        from manifest import load_manifest
        rows = load_manifest(man)
        rng = np.random.default_rng(FRONTEND.seeds.listen)
        out = []
        for src in sorted({r["source"] for r in rows}):
            rs = [r for r in rows if r["source"] == src]
            for i in rng.choice(len(rs), min(CLIPS_PER_SOURCE, len(rs)), replace=False):
                out.append((f"{src}:{Path(rs[i]['path']).name}", load_clip(rs[i], FRONTEND.window)))
        return out
    if npz and Path(npz).is_file():
        d = np.load(npz)
        rng = np.random.default_rng(FRONTEND.seeds.listen)
        X = d["windows"]
        return [(f"window {i}", X[i]) for i in rng.choice(len(X), min(24, len(X)), replace=False)]
    raise RuntimeError("set CIM_MANIFEST or CIM_GUN_WINDOWS to real audio")


def same(test: unittest.TestCase, x: np.ndarray, cfg: DetectorConfig = CFG, name: str = "") -> None:
    a, b = detect(x, cfg), detect_reference(x, cfg)
    test.assertTrue(np.array_equal(a.triggers, b.triggers), f"{name}: triggers differ {a.triggers} {b.triggers}")
    test.assertTrue(np.array_equal(a.preempted, b.preempted), f"{name}: preempted differ")


class TestDetector(unittest.TestCase):
    def test_real_clips_agree(self):
        sigs = real_signals()
        self.assertGreater(len(sigs), 0)
        n_trig = 0
        for name, x in sigs:
            same(self, x, name=name)
            n_trig += len(detect(x, CFG).triggers)
        self.assertGreater(n_trig, 0, "real clips gave no trigger at all")

    def test_synthetic_agree_across_configs(self):
        rng = np.random.default_rng(0)
        x = (rng.normal(0, 300, 40000)).astype(np.int16)
        for pos, amp in ((3000, 8000), (3500, 30000), (9000, 2000), (9050, 20000), (20000, 32767)):
            x[pos:pos + 200] = np.clip(x[pos:pos + 200] + rng.normal(0, amp, 200), -32768, 32767).astype(np.int16)
        for cfg in (CFG, replace(CFG, refractory_samples=1), replace(CFG, preempt_log2=0),
                    replace(CFG, ratio_log2=1, min_energy=0), replace(CFG, fast_samples=16, background_samples=256)):
            same(self, x, cfg, str(cfg))

    def test_silence_and_warmup(self):
        self.assertEqual(len(detect(np.zeros(20000, np.int16), CFG).triggers), 0)
        x = np.zeros(20000, np.int16)
        x[100] = 20000                       # inside the warm-up: never a trigger
        x[5000] = 20000
        t = detect(x, CFG).triggers
        self.assertTrue(np.array_equal(t, [5000]), t)

    def test_refractory_and_preemption(self):
        x = np.zeros(30000, np.int16)
        x[5000:5010] = 3000
        x[5000 + CFG.refractory_samples // 2:5000 + CFG.refractory_samples // 2 + 10] = 3000   # equal: blocked
        x[12000:12010] = 1000
        x[12400:12410] = 30000                                                                  # louder: preempts
        det = detect(x, CFG)
        self.assertEqual(det.triggers.tolist()[0], 5000)
        self.assertIn(12400, det.triggers.tolist())
        self.assertIn(12000, det.preempted.tolist())
        same(self, x)

    def test_register_widths_hold_at_full_scale(self):
        x = np.tile(np.array([32767, -32768], np.int16), 3000)    # largest possible |d|
        detect_reference(x, CFG)                                   # asserts every width on every sample
        w = register_widths(CFG)
        self.assertEqual((w["d"], w["d2"], w["p"], w["Ef"], w["Eb"]), (17, 32, 28, 33, 37))
        self.assertEqual((w["ratio_rhs"], w["preempt_rhs"], w["R_cnt"]), (40, 35, 11))

    def test_refractory_never_longer_than_post_trigger_audio(self):
        self.assertLessEqual(CFG.refractory_samples, FRONTEND.window.post_trigger_samples)
        with self.assertRaises(AssertionError):
            replace(CFG, refractory_samples=5601).validate(WindowConfig())

    def test_rejects_non_int16(self):
        with self.assertRaises(TypeError):
            detect(np.zeros(10, np.float32), CFG)
        with self.assertRaises(TypeError):
            detect_reference(np.zeros(10, np.int32), CFG)


if __name__ == "__main__":
    unittest.main()
