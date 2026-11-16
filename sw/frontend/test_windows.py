"""Window placement (D22), trigger matching and the power-of-two gain."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio import apply_gain, gain_shift, normalise, to_int16  # noqa: E402
from config import FRONTEND  # noqa: E402
from windows import audio_bounds, cut, match, placement, reference_samples  # noqa: E402

W, G, R = FRONTEND.window, FRONTEND.gain, FRONTEND.reference
SR = W.sample_rate


class TestPlacement(unittest.TestCase):
    def test_option_a_numbers(self):
        self.assertEqual((W.window_samples, W.pre_trigger_samples, W.real_samples, W.tail_samples),
                         (8000, 800, 6400, 1600))
        self.assertEqual(W.post_trigger_samples, 5600)

    def test_cut_is_real_audio_then_zeros(self):
        x = (np.arange(10000) % 1000 + 1).astype(np.int16)
        start, why = placement(2000, len(x), W)
        self.assertEqual((start, why), (1200, "ok"))
        w = cut(x, start, W)
        self.assertEqual(len(w), 8000)
        self.assertTrue(np.array_equal(w[:6400], x[1200:7600]))
        self.assertFalse(np.any(w[6400:]))

    def test_never_pads(self):
        self.assertEqual(placement(799, 10000, W)[1], "starts_before_audio")
        self.assertEqual(placement(1000, 10000, W, lo=201)[1], "starts_before_audio")
        self.assertEqual(placement(1000, 10000, W, lo=200)[1], "ok")
        self.assertEqual(placement(800, 6399, W)[1], "short_after_start")
        self.assertEqual(placement(800, 6400, W)[1], "ok")
        with self.assertRaises(AssertionError):
            cut(np.ones(7000, np.int16), 800, W)

    def test_file_padding_is_not_audio(self):
        x = np.zeros(10000, np.int16)
        x[487:9000] = 5
        self.assertEqual(audio_bounds(x), (487, 9000))
        self.assertEqual(audio_bounds(np.zeros(5, np.int16)), (0, 0))

    def test_c3gd_short_clip_fits_only_if_trigger_early_enough(self):
        n = int(0.4594 * SR)                     # 7350 samples
        self.assertEqual(placement(int(0.100 * SR), n, W)[1], "ok")
        self.assertEqual(placement(int(0.112 * SR), n, W)[1], "short_after_start")


class TestMatching(unittest.TestCase):
    def test_lead_and_lag_tolerance(self):
        refs = [10000]
        lead, lag = int(R.match_before_s * SR), int(R.match_after_s * SR)
        self.assertEqual(match(np.array([10000 - lead]), refs, R, SR), {0: 0})
        self.assertEqual(match(np.array([10000 + lag]), refs, R, SR), {0: 0})
        self.assertEqual(match(np.array([10000 - lead - 1]), refs, R, SR), {})
        self.assertEqual(match(np.array([10000 + lag + 1]), refs, R, SR), {})

    def test_one_to_one_nearest_first(self):
        # 10010 takes 10000 (nearest pair); 9900 then takes 10500 (600 samples lead, within 0.10 s).
        m = match(np.array([9900, 10010, 30000]), [10000, 10500], R, SR)
        self.assertEqual(m, {1: 0, 0: 1})
        m = match(np.array([9990, 10450]), [10000, 10500], R, SR)
        self.assertEqual(m, {0: 0, 1: 1})

    def test_references(self):
        x = np.zeros(16000, np.int16)
        x[8000:8010] = [1000, 3000, 9000, 20000, 30000, 25000, 9000, 3000, 1000, 0]
        refs, note = reference_samples({"source": "cadre"}, x, {}, R, SR)
        self.assertEqual((refs, note), ([8003], "ok"))       # first sample >= peak/2
        refs, _ = reference_samples({"source": "c3gd", "session_id": "c3gd_oh_farm"}, x, {}, R, SR)
        self.assertEqual(refs, [1600])
        refs, _ = reference_samples({"source": "c3gd", "session_id": "c3gd_nj_farm"}, x, {}, R, SR)
        self.assertEqual(refs, [736])
        row = {"source": "edge", "orig_path": "edge:glock/abc_v0.wav"}
        self.assertEqual(reference_samples(row, x, {"abc_v0": [0.5, 1.25]}, R, SR), ([8000, 20000], "ok"))
        self.assertEqual(reference_samples(row, x, {}, R, SR), ([], "no_edge_label"))


class TestGain(unittest.TestCase):
    def test_shift_puts_peak_in_top_octave(self):
        rng = np.random.default_rng(0)
        for peak in (1, 2, 3, 100, 127, 128, 16383, 16384, 32767):
            w = np.zeros(8000, np.int16)
            w[: W.real_samples] = rng.integers(-peak, peak + 1, W.real_samples)
            w[5] = peak
            out, k = normalise(w, W, G)
            p = int(np.abs(out[: W.real_samples].astype(np.int32)).max())
            if k < G.max_shift:
                self.assertTrue(16384 <= p <= 32767, (peak, k, p))
            self.assertEqual(int(np.abs(out.astype(np.int32)).max()), p)       # tail stays zero
            self.assertTrue(np.array_equal(out.astype(np.int32), w.astype(np.int32) << k))   # no saturation

    def test_edge_cases(self):
        w = np.zeros(8000, np.int16)
        self.assertEqual(gain_shift(w, W, G), G.max_shift)
        w[0] = -32768
        self.assertEqual(gain_shift(w, W, G), 0)
        w[0] = -16384
        self.assertEqual(gain_shift(w, W, G), 0)
        w[0] = -16383
        self.assertEqual(gain_shift(w, W, G), 1)
        w = np.zeros(8000, np.int16)
        w[W.real_samples] = 30000                       # in the zeroed tail: ignored by the gain
        self.assertEqual(gain_shift(w, W, G), G.max_shift)
        self.assertTrue(np.array_equal(apply_gain(np.array([20000], np.int16), 1), [32767]))

    def test_to_int16(self):
        self.assertTrue(np.array_equal(to_int16(np.array([-1.5, -1.0, 0.0, 0.5, 1.0, 1.2])),
                                       [-32768, -32768, 0, 16384, 32767, 32767]))


if __name__ == "__main__":
    unittest.main()
