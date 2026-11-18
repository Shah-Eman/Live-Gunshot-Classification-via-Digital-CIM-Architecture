"""Float model architectures, spectral front ends, BN folding, and exact MAC counter.

Covers Steps 1.7 and 1.8 (docs/runbook.md):
- Hardware candidate: `cnn1d` (imported directly from `sw/models/cnn1d.py`),
  `cnn1d_bn` (same topology with BatchNormalization folded into `cnn1d` after
  training and verified numerically), and `cnn1d_w64` (64-ch conv3 fallback
  within the <=4M MAC / <=30k param budget).
- Software comparison models: `cnn2d_mel` (2D-CNN on log-mel spectrogram),
  `dscnn` (depthwise-separable 2D-CNN on log-mel spectrogram), and `mfcc_svm`
  (MFCC + delta + delta-delta summary statistics + SVM).
- Pre-trigger leakage probe: `pretrigger_cnn` (small 4-class 1D-CNN on the
  noise snippet strictly preceding the shot).
- Script-computed MACs and parameter counts (`compute_model_cost`), counting
  both neural layers and DSP front-end (window + FFT + mel + DCT) operations.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
jax.config.update("jax_default_matmul_precision", "float32")
import keras
from keras import layers
import numpy as np
from scipy.fft import dct, rfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "models"))
from cnn1d import NUM_CLASSES, build_cnn1d, check_against_spec, layer_table  # noqa: E402

BUDGET_MAX_MACS = 4_000_000
BUDGET_MAX_PARAMS = 30_000


# ---------------------------------------------------------------------------
# Mel / MFCC front end (pure numpy/scipy, identical across train & eval)
# ---------------------------------------------------------------------------
def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def make_mel_filterbank(sr: int, n_fft: int = 256, n_mels: int = 40) -> np.ndarray:
    """Triangular mel filterbank [n_mels, n_fft // 2 + 1], float32."""
    fmin, fmax = 50.0, sr / 2.0
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


def mel_config_for_sr(sr: int) -> tuple[int, int, int]:
    """(n_fft, hop_length, n_mels) with ~10 ms hop and tile-friendly 32 mel bins."""
    if sr <= 8000:
        return 128, 80, 32
    if sr <= 16000:
        return 256, 160, 32
    return 512, 220, 32


def extract_log_mel(X_wave: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Multithreaded float32 log-mel spectrogram: X_wave [N, L] -> [N, T, n_mels, 1]."""
    n_fft, hop, n_mels = mel_config_for_sr(sr)
    fb = make_mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels)  # [M, F]
    win = np.hanning(n_fft).astype(np.float32)
    X_wave = np.ascontiguousarray(X_wave, dtype=np.float32)
    N, L = X_wave.shape
    n_frames = (L - n_fft) // hop + 1
    s0, s1 = X_wave.strides
    frames = np.lib.stride_tricks.as_strided(
        X_wave, shape=(N, n_frames, n_fft), strides=(s0, s1 * hop, s1)
    )
    spec = rfft(frames * win, n=n_fft, axis=-1, workers=-1)
    power = (spec.real ** 2 + spec.imag ** 2) * np.float32(1.0 / n_fft)
    mel = np.matmul(power, fb.T)
    # Per-window 60 dB dynamic range floor (prevents quiet-floor noise-augmentation mismatch)
    peak = np.maximum(mel.max(axis=(1, 2), keepdims=True), 1e-7)
    db = 10.0 * np.log10(np.maximum(mel, peak * 1e-6) / peak)  # in [-60, 0] dB
    norm_mel = (db + 30.0) * np.float32(1.0 / 30.0)             # in [-1, +1]
    return norm_mel[..., None].astype(np.float32)


def extract_mfcc_stats(X_wave: np.ndarray, sr: int = 16000, real_samples: int | None = None, n_mfcc: int = 13) -> np.ndarray:
    """MFCC + delta + delta-delta summary stats [N, 6 * n_mfcc] over real audio."""
    if real_samples is not None and real_samples < X_wave.shape[1]:
        X_wave = X_wave[:, :real_samples]
    db_mel = (extract_log_mel(X_wave, sr=sr)[..., 0] * 30.0) - 30.0  # [N, T, M] in dB
    mfcc = dct(db_mel, type=2, axis=-1, norm="ortho", workers=-1)[..., :n_mfcc].astype(np.float32)
    # First and second temporal differences
    d1 = np.zeros_like(mfcc)
    d1[:, 1:-1] = 0.5 * (mfcc[:, 2:] - mfcc[:, :-2])
    d1[:, 0] = mfcc[:, 1] - mfcc[:, 0]
    d1[:, -1] = mfcc[:, -1] - mfcc[:, -2]
    d2 = np.zeros_like(d1)
    d2[:, 1:-1] = 0.5 * (d1[:, 2:] - d1[:, :-2])
    feats = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            mfcc.max(axis=1),
            d1.mean(axis=1),
            d1.std(axis=1),
            d2.std(axis=1),
        ],
        axis=-1,
    )
    return feats.astype(np.float32)


def dsp_frontend_macs(input_len: int, sr: int, include_dct_n_mfcc: int = 0) -> int:
    """Exact real MAC count of STFT + power + sparse mel filterbank (+ optional DCT)."""
    n_fft, hop, n_mels = mel_config_for_sr(sr)
    n_frames = (input_len - n_fft) // hop + 1
    fb = make_mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels)
    nnz_mel = int(np.count_nonzero(fb))
    # Per frame: Hann window (n_fft) + real-FFT radix-2 ((n_fft/2)*log2(n_fft/2)*4 + 2*n_fft)
    # + |X|^2 (2*(n_fft//2 + 1)) + sparse mel multiply-adds (nnz_mel)
    half = n_fft // 2
    fft_macs = int(half * int(round(math.log2(half))) * 4 + 2 * n_fft)
    per_frame = n_fft + fft_macs + 2 * (half + 1) + nnz_mel
    if include_dct_n_mfcc > 0:
        per_frame += n_mels * include_dct_n_mfcc
    return int(n_frames * per_frame)


# ---------------------------------------------------------------------------
# Neural architectures (Keras 3)
# ---------------------------------------------------------------------------
def build_cnn1d_bn(num_classes: int = NUM_CLASSES, input_len: int = 8000) -> keras.Model:
    """Tile-friendly 1D-CNN with BatchNormalization (folded into cnn1d at export)."""
    x_in = keras.Input(shape=(input_len, 1), name="input")
    x = layers.Conv1D(16, 64, strides=8, padding="valid", use_bias=True, name="conv1")(x_in)
    x = layers.BatchNormalization(momentum=0.9, name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool1")(x)
    x = layers.Conv1D(32, 9, strides=1, padding="same", use_bias=True, name="conv2")(x)
    x = layers.BatchNormalization(momentum=0.9, name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool2")(x)
    x = layers.Conv1D(32, 9, strides=1, padding="same", use_bias=True, name="conv3")(x)
    x = layers.BatchNormalization(momentum=0.9, name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    logits = layers.Dense(num_classes, activation=None, name="dense")(x)
    return keras.Model(x_in, logits, name="cnn1d_bn")


def build_cnn1d_w64_bn(num_classes: int = NUM_CLASSES, input_len: int = 8000, with_bn: bool = True) -> keras.Model:
    """Wider conv3=64 hardware candidate (runbook Step 1.8 fallback, <=3.31M MACs, <=25k params)."""
    x_in = keras.Input(shape=(input_len, 1), name="input")
    if with_bn:
        x = layers.Conv1D(16, 64, strides=8, padding="valid", use_bias=True, name="conv1")(x_in)
        x = layers.ReLU(name="relu1")(layers.BatchNormalization(momentum=0.9, name="bn1")(x))
        x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool1")(x)
        x = layers.Conv1D(32, 9, strides=1, padding="same", use_bias=True, name="conv2")(x)
        x = layers.ReLU(name="relu2")(layers.BatchNormalization(momentum=0.9, name="bn2")(x))
        x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool2")(x)
        x = layers.Conv1D(64, 9, strides=1, padding="same", use_bias=True, name="conv3")(x)
        x = layers.ReLU(name="relu3")(layers.BatchNormalization(momentum=0.9, name="bn3")(x))
    else:
        x = layers.Conv1D(16, 64, strides=8, padding="valid", activation="relu", name="conv1")(x_in)
        x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool1")(x)
        x = layers.Conv1D(32, 9, strides=1, padding="same", activation="relu", name="conv2")(x)
        x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool2")(x)
        x = layers.Conv1D(64, 9, strides=1, padding="same", activation="relu", name="conv3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    logits = layers.Dense(num_classes, activation=None, name="dense")(x)
    return keras.Model(x_in, logits, name="cnn1d_w64_bn" if with_bn else "cnn1d_w64")


def fold_conv_bn_weights(
    conv_layer: keras.layers.Layer, bn_layer: keras.layers.BatchNormalization
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNormalization parameters into preceding Conv1D/Conv2D/DepthwiseConv2D."""
    w_list = conv_layer.get_weights()
    W = w_list[0].astype(np.float64)
    gamma, beta, mean, var = [arr.astype(np.float64) for arr in bn_layer.get_weights()]
    b = w_list[1].astype(np.float64) if len(w_list) > 1 else np.zeros(gamma.shape[0], dtype=np.float64)
    scale = gamma / np.sqrt(var + float(bn_layer.epsilon))
    if isinstance(conv_layer, layers.DepthwiseConv2D):
        # Depthwise kernel shape: (kh, kw, cin, depth_multiplier)
        W_fold = W * scale.reshape(1, 1, W.shape[2], W.shape[3])
        b_fold = (b - mean) * scale + beta
    else:
        # Conv1D (k, cin, cout) or Conv2D (kh, kw, cin, cout): channel out is axis -1
        shape = [1] * (W.ndim - 1) + [-1]
        W_fold = W * scale.reshape(shape)
        b_fold = (b - mean) * scale + beta
    return W_fold.astype(np.float32), b_fold.astype(np.float32)


def fold_batchnorm_1d(bn_model: keras.Model, wide64: bool = False) -> tuple[keras.Model, float]:
    """Fold BN layers of `cnn1d_bn` (or `cnn1d_w64_bn`) into pure hardware Conv1D+ReLU model.

    Returns (folded_model, max_abs_logit_diff_on_random_probe).
    """
    in_len = int(bn_model.input_shape[1])
    n_cls = int(bn_model.output_shape[-1])
    folded = build_cnn1d_w64_bn(n_cls, in_len, with_bn=False) if wide64 else build_cnn1d(n_cls, in_len)
    for cname, bname in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
        wf, bf = fold_conv_bn_weights(bn_model.get_layer(cname), bn_model.get_layer(bname))
        folded.get_layer(cname).set_weights([wf, bf])
    folded.get_layer("dense").set_weights(bn_model.get_layer("dense").get_weights())

    rng = np.random.default_rng(42)
    probe = rng.standard_normal((16, in_len, 1), dtype=np.float32)
    y_bn = np.asarray(bn_model(probe, training=False))
    y_fd = np.asarray(folded(probe, training=False))
    max_diff = float(np.max(np.abs(y_bn - y_fd)))
    if max_diff > 1e-4:
        raise AssertionError(f"BN folding mismatch: max_diff={max_diff}")
    return folded, max_diff


def build_cnn2d_mel(
    input_shape: tuple[int, int, int], num_classes: int = NUM_CLASSES, with_bn: bool = True
) -> keras.Model:
    """2D-CNN on log-mel spectrogram [T, M, 1], within <=4M MACs & <=30k params."""
    x_in = keras.Input(shape=input_shape, name="input")

    def blk(x, ch: int, idx: int):
        x = layers.Conv2D(ch, 3, padding="same", use_bias=not with_bn, name=f"conv{idx}")(x)
        if with_bn:
            x = layers.BatchNormalization(momentum=0.9, name=f"bn{idx}")(x)
        return layers.ReLU(name=f"relu{idx}")(x)

    x = blk(x_in, 16, 1)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = blk(x, 32, 2)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = blk(x, 32, 3)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    logits = layers.Dense(num_classes, activation=None, name="dense")(x)
    return keras.Model(x_in, logits, name="cnn2d_mel_bn" if with_bn else "cnn2d_mel")


def fold_cnn2d_mel(bn_model: keras.Model) -> tuple[keras.Model, float]:
    shape = tuple(int(d) for d in bn_model.input_shape[1:])
    n_cls = int(bn_model.output_shape[-1])
    folded = build_cnn2d_mel(shape, n_cls, with_bn=False)
    for idx in (1, 2, 3):
        wf, bf = fold_conv_bn_weights(bn_model.get_layer(f"conv{idx}"), bn_model.get_layer(f"bn{idx}"))
        folded.get_layer(f"conv{idx}").set_weights([wf, bf])
    folded.get_layer("dense").set_weights(bn_model.get_layer("dense").get_weights())
    probe = np.random.default_rng(42).standard_normal((8, *shape), dtype=np.float32)
    diff = float(np.max(np.abs(np.asarray(bn_model(probe, training=False)) - np.asarray(folded(probe, training=False)))))
    if diff > 1e-4:
        raise AssertionError(f"2D BN fold mismatch: {diff}")
    return folded, diff


def build_dscnn_mel(
    input_shape: tuple[int, int, int], num_classes: int = NUM_CLASSES, with_bn: bool = True
) -> keras.Model:
    """Depthwise-Separable 2D-CNN on log-mel spectrogram [T, M, 1]."""
    x_in = keras.Input(shape=input_shape, name="input")
    x = layers.Conv2D(16, 3, padding="same", use_bias=not with_bn, name="stem_conv")(x_in)
    if with_bn:
        x = layers.BatchNormalization(momentum=0.9, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    specs = [(32, 2), (32, 1), (64, 2), (64, 1)]
    for i, (ch, st) in enumerate(specs, start=1):
        x = layers.DepthwiseConv2D(3, strides=st, padding="same", use_bias=not with_bn, name=f"dw{i}")(x)
        if with_bn:
            x = layers.BatchNormalization(momentum=0.9, name=f"dw_bn{i}")(x)
        x = layers.ReLU(name=f"dw_relu{i}")(x)
        x = layers.Conv2D(ch, 1, strides=1, padding="same", use_bias=not with_bn, name=f"pw{i}")(x)
        if with_bn:
            x = layers.BatchNormalization(momentum=0.9, name=f"pw_bn{i}")(x)
        x = layers.ReLU(name=f"pw_relu{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    logits = layers.Dense(num_classes, activation=None, name="dense")(x)
    return keras.Model(x_in, logits, name="dscnn_mel_bn" if with_bn else "dscnn_mel")


def fold_dscnn_mel(bn_model: keras.Model) -> tuple[keras.Model, float]:
    shape = tuple(int(d) for d in bn_model.input_shape[1:])
    n_cls = int(bn_model.output_shape[-1])
    folded = build_dscnn_mel(shape, n_cls, with_bn=False)
    wf, bf = fold_conv_bn_weights(bn_model.get_layer("stem_conv"), bn_model.get_layer("stem_bn"))
    folded.get_layer("stem_conv").set_weights([wf, bf])
    for i in (1, 2, 3, 4):
        w_dw, b_dw = fold_conv_bn_weights(bn_model.get_layer(f"dw{i}"), bn_model.get_layer(f"dw_bn{i}"))
        folded.get_layer(f"dw{i}").set_weights([w_dw, b_dw])
        w_pw, b_pw = fold_conv_bn_weights(bn_model.get_layer(f"pw{i}"), bn_model.get_layer(f"pw_bn{i}"))
        folded.get_layer(f"pw{i}").set_weights([w_pw, b_pw])
    folded.get_layer("dense").set_weights(bn_model.get_layer("dense").get_weights())
    probe = np.random.default_rng(42).standard_normal((8, *shape), dtype=np.float32)
    diff = float(np.max(np.abs(np.asarray(bn_model(probe, training=False)) - np.asarray(folded(probe, training=False)))))
    if diff > 1e-4:
        raise AssertionError(f"DS-CNN BN fold mismatch: {diff}")
    return folded, diff


def build_pretrigger_cnn(input_len: int, num_classes: int = 4) -> keras.Model:
    """Small 1D-CNN for the 4-class pre-trigger noise leakage check (Step 1.8 item 3)."""
    x_in = keras.Input(shape=(input_len, 1), name="input")
    x = layers.Conv1D(16, 32, strides=4, padding="valid", use_bias=False, name="conv1")(x_in)
    x = layers.ReLU(name="relu1")(layers.BatchNormalization(momentum=0.9, name="bn1")(x))
    x = layers.MaxPooling1D(2, name="pool1")(x)
    x = layers.Conv1D(32, 5, padding="same", use_bias=False, name="conv2")(x)
    x = layers.ReLU(name="relu2")(layers.BatchNormalization(momentum=0.9, name="bn2")(x))
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    logits = layers.Dense(num_classes, activation=None, name="dense")(x)
    return keras.Model(x_in, logits, name="pretrigger_cnn")


# ---------------------------------------------------------------------------
# Script-computed MAC and parameter counter (never estimated!)
# ---------------------------------------------------------------------------
def compute_keras_macs_and_params(model: keras.Model, frontend_macs: int = 0) -> dict:
    """Compute exact inference MACs and parameters of a folded Keras model."""
    nn_macs = 0
    weights = 0
    biases = 0
    per_layer = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        if isinstance(layer, layers.BatchNormalization):
            raise ValueError(
                f"Fold BatchNormalization ({layer.name}) before computing inference MACs/params"
            )
        out_shape = tuple(int(d) for d in layer.output.shape[1:])
        lw = lb = lm = 0
        if isinstance(layer, layers.Conv1D):
            k, cin, cout = (int(d) for d in layer.kernel.shape)
            lw = k * cin * cout
            lb = cout if layer.use_bias else 0
            lm = out_shape[0] * cout * k * cin
        elif isinstance(layer, layers.DepthwiseConv2D):
            kh, kw, cin, dm = (int(d) for d in layer.kernel.shape)
            cout = cin * dm
            lw = kh * kw * cin * dm
            lb = cout if layer.use_bias else 0
            lm = out_shape[0] * out_shape[1] * cin * dm * kh * kw
        elif isinstance(layer, layers.Conv2D):
            kh, kw, cin, cout = (int(d) for d in layer.kernel.shape)
            lw = kh * kw * cin * cout
            lb = cout if layer.use_bias else 0
            lm = out_shape[0] * out_shape[1] * cout * kh * kw * cin
        elif isinstance(layer, layers.Dense):
            nin, nout = (int(d) for d in layer.kernel.shape)
            lw = nin * nout
            lb = nout if layer.use_bias else 0
            lm = nin * nout
        assert lw + lb == layer.count_params(), f"param count mismatch on {layer.name}"
        nn_macs += lm
        weights += lw
        biases += lb
        per_layer.append(
            {"name": layer.name, "type": type(layer).__name__, "out": out_shape, "params": lw + lb, "macs": lm}
        )

    total_params = weights + biases
    total_macs = int(frontend_macs) + int(nn_macs)
    return {
        "weights": int(weights),
        "biases": int(biases),
        "params": int(total_params),
        "frontend_macs": int(frontend_macs),
        "nn_macs": int(nn_macs),
        "total_macs": int(total_macs),
        "within_budget": bool(total_macs <= BUDGET_MAX_MACS and total_params <= BUDGET_MAX_PARAMS),
        "layers": per_layer,
    }
