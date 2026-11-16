#!/usr/bin/env python3
"""Directed edge cases for the 16 x 16 CIM tile (runbook Step 1.4).

One list, shared by everything that tests a tile: the Python tile model's tests
(`test_tile_model.py`), the test vector generator (Step 1.5, `emit.py`) and the
RTL tile testbench (Step 3.1, "directed tests for every edge case from Step
1.4"). Add new cases here, never in a copy.

Each case is `(name, W, a)` with `W` of shape (ROWS, COLS) int8, where `W[r, c]`
is the weight at tile row r, column c (see `tile_model.py`), and `a` of shape
(ROWS,) int8. The expected result is `W.T @ a`. The list is deterministic: the
two random members are drawn from a fixed seed.
"""
from __future__ import annotations

import numpy as np

from tile_model import COLS, ROWS

SEED = 20261029


def _randint8(g: np.random.Generator, shape, lo: int = -128, hi: int = 127) -> np.ndarray:
    return g.integers(lo, hi + 1, size=shape, dtype=np.int64).astype(np.int8)


def edge_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    g = np.random.default_rng(SEED + 1)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    def full(W_val=None):
        return np.full((ROWS, COLS), W_val, dtype=np.int8)

    alt_rows = np.where(np.arange(ROWS) % 2 == 0, 1, -1).astype(np.int8)
    alt_w = (alt_rows[:, None] * np.ones(COLS, dtype=np.int8)[None, :]).astype(np.int8)
    ramp_w = (np.arange(ROWS * COLS).reshape(ROWS, COLS) % 255 - 127).astype(np.int8)
    weight_sets = {
        "randW": _randint8(g, (ROWS, COLS)),
        "W=127": full(127), "W=-128": full(-128), "W=-127": full(-127),
        "W=1": full(1), "W=-1": full(-1), "W=0": full(0), "W=5": full(5),
        "Walt": alt_w, "Wramp": ramp_w,
    }
    act_sets = {
        "a=-128": np.full(ROWS, -128, np.int8), "a=127": np.full(ROWS, 127, np.int8),
        "a=0": np.zeros(ROWS, np.int8), "a=-1": np.full(ROWS, -1, np.int8),
        "a=1": np.full(ROWS, 1, np.int8), "a=-127": np.full(ROWS, -127, np.int8),
        "aalt": (alt_rows * 127).astype(np.int8),
        "aalt128": np.where(np.arange(ROWS) % 2 == 0, -128, 127).astype(np.int8),
        "aramp": (np.arange(ROWS) * 17 - 128).astype(np.int8),
        "arand": _randint8(g, (ROWS,)),
    }

    # The four named by the runbook and spec 8.2, each against every weight set
    # and every activation set (all combinations, 100 cases).
    for wn, W in weight_sets.items():
        for an, a in act_sets.items():
            cases.append((f"{wn}|{an}", W.copy(), a.copy()))

    # Largest possible magnitudes (spec 6.4): +262,144, -260,096, ...
    for wv in (-128, 127):
        for av in (-128, 127):
            cases.append((f"max W={wv} a={av}", full(wv), np.full(ROWS, av, np.int8)))

    # A single -128 activation on row r with a positive weight must give a
    # NEGATIVE result: the bit-7 subtract (spec 4.6 item 3). Every row.
    for r in range(ROWS):
        a = np.zeros(ROWS, np.int8)
        a[r] = -128
        cases.append((f"single -128 on row {r}, W=+5", full(5), a))
        W = np.zeros((ROWS, COLS), np.int8)
        W[r, :] = np.arange(COLS) - 8          # positive, zero and negative weights
        cases.append((f"single -128 on row {r}, W row mixed", W, a.copy()))

    # Every single-bit pattern and every single-bit-cleared pattern, on all rows
    # (positive and negative, so each of the 8 bit positions is the only 1, and
    # the only 0, in turn).
    patterns = sorted({1 << k for k in range(7)} | {-(1 << k) for k in range(8)}
                      | {127 ^ (1 << k) for k in range(7)} | {-1 - (1 << k) for k in range(7)})
    for v in patterns:
        cases.append((f"pattern {v}", weight_sets["randW"].copy(), np.full(ROWS, v, np.int8)))

    # Orientation: distinct weight in every cell and a distinct activation on
    # every row, so W.T @ a, W @ a and row/column swaps all disagree.
    cases.append(("orientation", ramp_w.copy(), (np.arange(ROWS) * 3 - 20).astype(np.int8)))
    # One-hot columns: column c copies activation c, so psum == a exactly.
    eye = np.eye(ROWS, COLS, dtype=np.int8)
    cases.append(("identity weights", eye, (np.arange(ROWS) * 17 - 128).astype(np.int8)))
    cases.append(("reversed identity", eye[::-1].copy(), (np.arange(ROWS) * 17 - 128).astype(np.int8)))

    # Exported-weight range [-127, 127] (spec 2.2) at the extremes.
    cases.append(("exported W=-127 a=-128", full(-127), np.full(ROWS, -128, np.int8)))
    cases.append(("exported W=127 a=-128", full(127), np.full(ROWS, -128, np.int8)))
    return cases


def edge_batch() -> tuple[np.ndarray, np.ndarray]:
    """All edge cases stacked: W (N, ROWS, COLS) and a (N, ROWS), in list order."""
    cs = edge_cases()
    return np.stack([c[1] for c in cs]), np.stack([c[2] for c in cs])
