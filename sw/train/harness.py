"""Compiled JAX/Keras training and SVM harness for Steps 1.7 and 1.8.

Key speed & reproducibility design:
- Compiles one stateless `@jax.jit` train step and eval step per (model_type,
  input_shape, num_classes). Re-initializing weights & optimizer state per
  (seed, fold, bit_depth) reuses the compiled XLA executable with ZERO
  recompilation overhead across hundreds of CV runs.
- Uses balanced inverse-frequency class weights on every fold's training set.
- Uses a leak-free cosine learning-rate decay schedule (`lr_max -> lr_min`)
  with no validation-loss peeking for early stopping.
"""
from __future__ import annotations

import math
import os
from typing import Callable

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import keras
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced class weight vector [n_classes]: N / (C * max(N_c, 1))."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = float(len(y))
    w = total / (n_classes * np.maximum(counts, 1.0))
    return w.astype(np.float32)


class CompiledKerasTrainer:
    """Reusable XLA-compiled trainer for a fixed Keras model architecture & shape."""

    def __init__(
        self,
        builder_fn: Callable[[], keras.Model],
        total_steps: int,
        batch_size: int = 256,
        lr_max: float = 3e-3,
        lr_min: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.builder_fn = builder_fn
        self.batch_size = int(batch_size)
        self.total_steps = max(1, int(total_steps))
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.weight_decay = float(weight_decay)
        self.template = builder_fn()
        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.lr_max,
            decay_steps=self.total_steps,
            alpha=self.lr_min / self.lr_max,
        )
        self.opt = keras.optimizers.AdamW(learning_rate=sched, weight_decay=self.weight_decay)
        self.opt.build(self.template.trainable_variables)

        model = self.template
        opt = self.opt

        @jax.jit
        def _train_step(t_vars, nt_vars, o_vars, xb, yb, wb):
            def loss_fn(tv):
                logits, new_nt = model.stateless_call(tv, nt_vars, xb, training=True)
                lse = jax.nn.logsumexp(logits, axis=-1)
                ce = lse - jnp.take_along_axis(logits, yb[:, None], axis=-1)[:, 0]
                return jnp.mean(ce * wb), new_nt

            (loss, new_nt), grads = jax.value_and_grad(loss_fn, has_aux=True)(t_vars)
            new_tv, new_ov = opt.stateless_apply(o_vars, grads, t_vars)
            return new_tv, new_nt, new_ov, loss

        @jax.jit
        def _eval_step(t_vars, nt_vars, xb):
            logits, _ = model.stateless_call(t_vars, nt_vars, xb, training=False)
            return logits

        self._train_step = _train_step
        self._eval_step = _eval_step

    def fresh_state(self, seed: int):
        keras.utils.set_random_seed(int(seed))
        m = self.builder_fn()
        sched = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.lr_max,
            decay_steps=self.total_steps,
            alpha=self.lr_min / self.lr_max,
        )
        opt = keras.optimizers.AdamW(learning_rate=sched, weight_decay=self.weight_decay)
        opt.build(m.trainable_variables)
        tv = [jnp.asarray(v.numpy()) for v in m.trainable_variables]
        nt = [jnp.asarray(v.numpy()) for v in m.non_trainable_variables]
        ov = [jnp.asarray(v.numpy()) for v in opt.variables]
        return m, tv, nt, ov

    def fit_custom_epochs(
        self,
        epoch_data_fn: Callable[[int], tuple[np.ndarray, np.ndarray]],
        n_epochs: int,
        n_classes: int,
        seed: int,
    ) -> tuple[keras.Model, list[float]]:
        """Train for `n_epochs`, calling `epoch_data_fn(e) -> (X_tr, y_tr)` each epoch."""
        m, tv, nt, ov = self.fresh_state(seed)
        bs = self.batch_size
        rng = np.random.default_rng([seed, 777])
        epoch_losses: list[float] = []

        for ep in range(n_epochs):
            X_ep, y_ep = epoch_data_fn(ep)
            cw = compute_class_weights(y_ep, n_classes)
            perm = rng.permutation(len(y_ep))
            X_ep = X_ep[perm]
            y_ep = y_ep[perm].astype(np.int32)
            w_ep = cw[y_ep]

            # Pad tail to full batch so XLA input shape never changes!
            rem = len(y_ep) % bs
            if rem != 0:
                pad = bs - rem
                pad_idx = rng.integers(0, len(y_ep), size=pad)
                X_ep = np.concatenate([X_ep, X_ep[pad_idx]], axis=0)
                y_ep = np.concatenate([y_ep, y_ep[pad_idx]], axis=0)
                w_ep = np.concatenate([w_ep, np.zeros(pad, dtype=np.float32)], axis=0)

            losses = []
            for i in range(0, len(y_ep), bs):
                tv, nt, ov, l = self._train_step(
                    tv, nt, ov, X_ep[i : i + bs], y_ep[i : i + bs], w_ep[i : i + bs]
                )
                losses.append(l)
            epoch_losses.append(float(jnp.mean(jnp.stack(losses))))

        # Copy trained variables back into the Keras model instance
        for var, val in zip(m.trainable_variables, tv):
            var.assign(np.asarray(val))
        for var, val in zip(m.non_trainable_variables, nt):
            var.assign(np.asarray(val))
        return m, epoch_losses

    def predict_classes(self, model: keras.Model, X: np.ndarray) -> np.ndarray:
        tv = [jnp.asarray(v.numpy()) for v in model.trainable_variables]
        nt = [jnp.asarray(v.numpy()) for v in model.non_trainable_variables]
        bs = self.batch_size
        n = len(X)
        rem = n % bs
        if rem != 0:
            pad = bs - rem
            X_pad = np.concatenate([X, np.zeros((pad, *X.shape[1:]), dtype=X.dtype)], axis=0)
        else:
            X_pad = X
        preds = []
        for i in range(0, len(X_pad), bs):
            logits = self._eval_step(tv, nt, X_pad[i : i + bs])
            preds.append(np.argmax(np.asarray(logits), axis=-1))
        return np.concatenate(preds, axis=0)[:n].astype(np.int64)


def train_eval_mfcc_svm(
    X_tr_feat: np.ndarray,
    y_tr: np.ndarray,
    X_va_feat: np.ndarray,
    kernel: str = "linear",
    seed: int = 1801,
) -> tuple[np.ndarray, dict]:
    """Train MFCC + SVM (linear or RBF) and return (y_pred, cost_dict)."""
    n_feat = int(X_tr_feat.shape[1])
    n_cls = int(len(np.unique(y_tr)))
    if kernel == "linear":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=5000)),
            ]
        )
        clf.fit(X_tr_feat, y_tr)
        y_pred = clf.predict(X_va_feat).astype(np.int64)
        # StandardScaler (2 * n_feat) + LinearSVC (n_cls * n_feat + n_cls)
        svm_params = 2 * n_feat + n_cls * n_feat + n_cls
        svm_macs = n_feat + n_cls * n_feat
    else:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(C=3.0, kernel="rbf", class_weight="balanced", random_state=seed)),
            ]
        )
        clf.fit(X_tr_feat, y_tr)
        y_pred = clf.predict(X_va_feat).astype(np.int64)
        svc: SVC = clf.named_steps["svm"]
        n_sv = int(svc.support_vectors_.shape[0])
        n_pairs = n_cls * (n_cls - 1) // 2
        svm_params = 2 * n_feat + n_sv * n_feat + int(svc.dual_coef_.size) + n_pairs
        svm_macs = n_feat + n_sv * n_feat + int(svc.dual_coef_.size)
    return y_pred, {"svm_params": int(svm_params), "svm_macs": int(svm_macs)}
