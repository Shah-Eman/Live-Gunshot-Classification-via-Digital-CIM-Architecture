#!/usr/bin/env python3
"""The network, defined once (runbook Step 1.1). Every training script imports this file.

It must match `docs/arith_spec.md` section 5.1 layer for layer:

    Input 8000x1 -> Conv1D k64 s8 16ch -> MaxPool4 -> Conv1D k9 32ch -> MaxPool4
    -> Conv1D k9 32ch -> GlobalAvgPool -> Dense 32->5

Layer names are the spec's tensor names (conv1, pool1, conv2, pool2, conv3, gap,
dense), so quantization, the integer reference and the test vectors can find
every layer by name. The dense layer outputs logits (no softmax): argmax picks
the class, in the order of `manifest.classes()`.

Plain Keras 3 only, no backend-specific code, so it runs on the TensorFlow,
JAX or PyTorch backend and a saved `.keras` file loads on any of them.

Run it to print the per-layer table (output shape, parameters, MACs), computed
from the built model, and compare it with spec 5.1:

    python sw/models/cnn1d.py           # exits non-zero if the table disagrees
"""
from __future__ import annotations

import sys

import keras
from keras import layers

INPUT_LEN = 8000         # 0.5 s at 16 kHz (spec 5.1, Decision D6)
NUM_CLASSES = 5          # pistol_9mm, revolver_38spl, rifle_556_223, shotgun_12ga, background

# spec 5.1 totals; the table printed below must reproduce them exactly
SPEC_WEIGHTS = 15_008
SPEC_BIASES = 85
SPEC_MACS = 2_731_168
SPEC_LAYERS = {          # name: (output shape (L, C) or (C,), weights, biases, MACs)
    "conv1": ((993, 16), 1_024, 16, 1_016_832),
    "pool1": ((248, 16), 0, 0, 0),
    "conv2": ((248, 32), 4_608, 32, 1_142_784),
    "pool2": ((62, 32), 0, 0, 0),
    "conv3": ((62, 32), 9_216, 32, 571_392),
    "gap":   ((32,), 0, 0, 0),
    "dense": ((5,), 160, 5, 160),
}


def build_cnn1d(num_classes: int = NUM_CLASSES, input_len: int = INPUT_LEN) -> keras.Model:
    """The 1D-CNN of spec 5.1, on a raw waveform of `input_len` samples, logits out."""
    x_in = keras.Input(shape=(input_len, 1), name="input")
    x = layers.Conv1D(16, 64, strides=8, padding="valid", activation="relu", name="conv1")(x_in)
    x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool1")(x)
    x = layers.Conv1D(32, 9, strides=1, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(4, strides=4, padding="valid", name="pool2")(x)
    x = layers.Conv1D(32, 9, strides=1, padding="same", activation="relu", name="conv3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    logits = layers.Dense(num_classes, activation=None, name="dense")(x)
    return keras.Model(x_in, logits, name="cnn1d")


def layer_table(model: keras.Model) -> list[dict]:
    """One row per layer, every number read from the built model, none typed in.

    MACs per inference: Conv1D = L_out x Cout x k x Cin, Dense = in x out.
    Pooling and global average pooling count no MACs (spec 5.1).
    """
    rows = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        out_shape = tuple(layer.output.shape[1:])
        weights = biases = macs = 0
        if isinstance(layer, layers.Conv1D):
            k, cin, cout = layer.kernel.shape
            weights, biases = k * cin * cout, cout
            macs = out_shape[0] * cout * k * cin
        elif isinstance(layer, layers.Dense):
            n_in, n_out = layer.kernel.shape
            weights, biases = n_in * n_out, n_out
            macs = n_in * n_out
        assert weights + biases == layer.count_params(), layer.name
        rows.append({"name": layer.name, "type": type(layer).__name__, "out": out_shape,
                     "weights": int(weights), "biases": int(biases), "macs": int(macs)})
    return rows


def check_against_spec(rows: list[dict]) -> list[str]:
    """Differences between the table and spec 5.1; an empty list means they agree."""
    errors = []
    names = [r["name"] for r in rows]
    if names != list(SPEC_LAYERS):
        errors.append(f"layers {names} != spec {list(SPEC_LAYERS)}")
    for r in rows:
        spec = SPEC_LAYERS.get(r["name"])
        if spec is None:
            continue
        got = (r["out"], r["weights"], r["biases"], r["macs"])
        if got != spec:
            errors.append(f"{r['name']}: (shape, W, B, MACs) {got} != spec {spec}")
    totals = (sum(r["weights"] for r in rows), sum(r["biases"] for r in rows),
              sum(r["macs"] for r in rows))
    if totals != (SPEC_WEIGHTS, SPEC_BIASES, SPEC_MACS):
        errors.append(f"totals (W, B, MACs) {totals} != spec "
                      f"{(SPEC_WEIGHTS, SPEC_BIASES, SPEC_MACS)}")
    return errors


def print_table(rows: list[dict]) -> None:
    print(f"{'layer':<6} {'type':<22} {'output':>10} {'weights':>8} {'biases':>7} {'MACs':>10}")
    for r in rows:
        shape = " x ".join(str(d) for d in r["out"])
        print(f"{r['name']:<6} {r['type']:<22} {shape:>10} {r['weights']:>8,} "
              f"{r['biases']:>7,} {r['macs']:>10,}")
    print(f"{'total':<6} {'':<22} {'':>10} {sum(r['weights'] for r in rows):>8,} "
          f"{sum(r['biases'] for r in rows):>7,} {sum(r['macs'] for r in rows):>10,}")


def main() -> int:
    rows = layer_table(build_cnn1d())
    print_table(rows)
    errors = check_against_spec(rows)
    if errors:
        print("\nDISAGREES with arith_spec.md 5.1:")
        for e in errors:
            print("  " + e)
        return 1
    print("\nMatches arith_spec.md 5.1 line by line "
          f"({SPEC_WEIGHTS:,} weights, {SPEC_BIASES} biases, {SPEC_MACS:,} MACs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
