"""Unit tests for the shared training & evaluation harness (Steps 1.7 and 1.8)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

HERE = Path(__file__).resolve().parent
SW_DIR = HERE.parent
REPO_DIR = SW_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SW_DIR / "frontend"))
sys.path.insert(0, str(SW_DIR / "data"))
sys.path.insert(0, str(SW_DIR / "models"))

from background import EpochBuilder  # noqa: E402
from cnn1d import build_cnn1d, check_against_spec, layer_table  # noqa: E402
from data import DatasetBundle, augment_waveform_float, truncate_adc_bits  # noqa: E402
from models import (  # noqa: E402
    BUDGET_MAX_MACS,
    BUDGET_MAX_PARAMS,
    build_cnn1d_bn,
    build_cnn1d_w64_bn,
    build_cnn2d_mel,
    build_dscnn_mel,
    compute_keras_macs_and_params,
    dsp_frontend_macs,
    extract_log_mel,
    fold_batchnorm_1d,
    fold_cnn2d_mel,
    fold_dscnn_mel,
)


class TestModelsAndBudgets(unittest.TestCase):
    def test_all_models_within_budget_and_bn_folds_cleanly(self):
        # 1. cnn1d_bn folds into exact spec 5.1 cnn1d
        m_bn = build_cnn1d_bn(5, 8000)
        # Perturb BN stats so folding is non-trivial
        for bname in ("bn1", "bn2", "bn3"):
            bn = m_bn.get_layer(bname)
            c = bn.gamma.shape[0]
            bn.set_weights(
                [
                    np.linspace(0.7, 1.4, c, dtype=np.float32),
                    np.linspace(-0.2, 0.3, c, dtype=np.float32),
                    np.linspace(-0.5, 0.5, c, dtype=np.float32),
                    np.linspace(0.2, 2.0, c, dtype=np.float32),
                ]
            )
        folded, diff = fold_batchnorm_1d(m_bn, wide64=False)
        self.assertLess(diff, 1e-5)
        self.assertEqual(check_against_spec(layer_table(folded)), [])
        cost_1d = compute_keras_macs_and_params(folded, frontend_macs=0)
        self.assertEqual(cost_1d["total_macs"], 2_731_168)
        self.assertEqual(cost_1d["params"], 15_093)
        self.assertTrue(cost_1d["within_budget"])

        # 2. cnn1d_w64
        m_w64_bn = build_cnn1d_w64_bn(5, 8000, with_bn=True)
        folded_w64, diff_w64 = fold_batchnorm_1d(m_w64_bn, wide64=True)
        self.assertLess(diff_w64, 1e-5)
        cost_w64 = compute_keras_macs_and_params(folded_w64, frontend_macs=0)
        self.assertTrue(cost_w64["within_budget"])
        self.assertLessEqual(cost_w64["total_macs"], BUDGET_MAX_MACS)
        self.assertLessEqual(cost_w64["params"], BUDGET_MAX_PARAMS)

        # 3. cnn2d_mel and dscnn_mel
        dummy_wave = np.zeros((2, 8000), dtype=np.float32)
        mel = extract_log_mel(dummy_wave, sr=16000)
        mel_shape = mel.shape[1:]
        fe_macs = dsp_frontend_macs(8000, 16000)

        m_2d, d2 = fold_cnn2d_mel(build_cnn2d_mel(mel_shape, 5, with_bn=True))
        self.assertLess(d2, 1e-5)
        c_2d = compute_keras_macs_and_params(m_2d, frontend_macs=fe_macs)
        self.assertTrue(c_2d["within_budget"], f"cnn2d over budget: {c_2d}")

        m_ds, d_ds = fold_dscnn_mel(build_dscnn_mel(mel_shape, 5, with_bn=True))
        self.assertLess(d_ds, 1e-5)
        c_ds = compute_keras_macs_and_params(m_ds, frontend_macs=fe_macs)
        self.assertTrue(c_ds["within_budget"], f"dscnn over budget: {c_ds}")

    def test_adc_truncation_and_augmentation_preserve_zero_tail(self):
        x = np.array([-32768, -12345, -1, 0, 1, 12345, 32767], dtype=np.int16)
        self.assertTrue(np.array_equal(truncate_adc_bits(x, 16), x))
        x8 = truncate_adc_bits(x, 8)
        self.assertTrue(np.all((x8.astype(np.int32) & 0xFF) == 0))

        rng = np.random.default_rng(0)
        w = rng.uniform(-0.5, 0.5, size=(8, 8000)).astype(np.float32)
        w[:, 6400:] = 0.0
        aug = augment_waveform_float(w, 6400, rng, max_shift_samples=80)
        self.assertTrue(np.all(aug[:, 6400:] == 0.0))
        self.assertLessEqual(float(np.max(np.abs(aug))), 1.0)


class TestDatasetBundleIntegration(unittest.TestCase):
    def test_matches_step16_epoch_builder_and_blocks_test_leakage(self):
        manifest = REPO_DIR.parent / "Dataset-Final" / "manifest.csv"
        gun_npz = REPO_DIR.parent / "step16_outputs" / "gun_windows.npz"
        if not manifest.is_file() or not gun_npz.is_file():
            self.skipTest("Dataset-Final or gun_windows.npz not present")

        splits_json = SW_DIR / "data" / "splits.json"
        win_idx = REPO_DIR / "results" / "step16" / "window_index.csv"
        bg_idx = REPO_DIR / "results" / "step16" / "background_index.csv"

        bundle = DatasetBundle(
            manifest_path=manifest,
            splits_path=splits_json,
            window_index_path=win_idx,
            background_index_path=bg_idx,
            gun_npz_path=gun_npz,
            sample_rate=16000,
        )

        # Must block any overlap with final test set
        tr0, va0 = bundle.splits.fold(0)
        some_test = {next(iter(bundle.splits.test))}
        with self.assertRaises(ValueError):
            bundle.prepare_split(tr0 | some_test, va0)
        with self.assertRaises(ValueError):
            bundle.prepare_split(tr0, va0 | some_test)

        # Must match EpochBuilder.epoch(0) bit-for-bit at 16kHz / 16-bit / 0.50s
        ctx = bundle.prepare_split(tr0, va0, window_s=0.50)
        X_bundle, y_bundle = bundle.sample_train_epoch_int16(ctx, epoch=0, adc_bits=16)

        eb = EpochBuilder.from_files(manifest, splits_json, win_idx, gun_npz, bg_idx, fold=0)
        X_eb, y_eb, _ = eb.epoch(0)
        self.assertTrue(np.array_equal(y_bundle, y_eb))
        self.assertTrue(np.array_equal(X_bundle, X_eb))


if __name__ == "__main__":
    unittest.main()
