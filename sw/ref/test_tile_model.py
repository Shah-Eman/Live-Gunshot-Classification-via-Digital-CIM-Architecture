#!/usr/bin/env python3
"""Tests for sw/ref/tile_model.py (runbook Step 1.4, verification ladder rung 3).

Run from the repo root, with no extra packages (stdlib unittest + numpy):

    python sw/ref/test_tile_model.py            # everything
    python -m unittest discover -s sw/ref -v    # same, verbose

Review-gate items covered here
------------------------------
* 100% match with numpy `W.T @ a` on 100,000 random vectors (RandomVsNumpy).
* 100% match on the edge cases of the runbook and of spec 8.2: all activations
  -128, all +127, all weights -128, all zeros, a single -128 activation with a
  positive weight, alternating signs, maximum-size inputs (EdgeCases).
* Every claim of spec 4.6 and 6.4 (BitSplit, PerCycle, Widths).
* The spec 7 worked example, cycle by cycle (WorkedExample).
* The tests themselves can catch a broken tile: four deliberately wrong tiles
  are each caught by the directed edge cases alone (BrokenTilesAreCaught).

Every random draw uses a fixed seed, so a failure reproduces exactly.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tile_model as tm  # noqa: E402
from tile_model import (  # noqa: E402
    COLS, ROWS, CimTile, TileOverflowError, TileScheduleError,
    bit_plane, bit_planes, bitserial_dot, numpy_reference, shift_term)

SEED = 20261029
N_RANDOM = 100_000          # runbook Step 1.4, item 3
CHUNK = 10_000              # keeps the int64 working arrays small


def rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def randint8(g, shape, lo=-128, hi=127) -> np.ndarray:
    return g.integers(lo, hi + 1, size=shape, dtype=np.int64).astype(np.int8)


def extreme_int8(g, shape) -> np.ndarray:
    """Values biased to the edges of int8: -128, -127, -1, 0, 1, 126, 127, or uniform."""
    picks = np.array([-128, -127, -1, 0, 1, 126, 127], dtype=np.int64)
    u = g.integers(0, 3, size=shape)
    out = np.where(u == 0, picks[g.integers(0, len(picks), size=shape)],
                   g.integers(-128, 128, size=shape))
    return out.astype(np.int8)


def mismatches(W, a, **kw) -> int:
    """How many vectors the bit-serial result gets wrong vs numpy `W.T @ a`."""
    got = bitserial_dot(W, a, **kw)
    want = numpy_reference(W, a)
    return int(np.count_nonzero(np.any(got != want, axis=-1)))


# --------------------------------------------------------------------------
# Directed edge cases, as (name, W, a) with W (16,16) and a (16,)
# --------------------------------------------------------------------------
def edge_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    g = rng(1)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    def full(W_val=None):
        return np.full((ROWS, COLS), W_val, dtype=np.int8)

    alt_rows = np.where(np.arange(ROWS) % 2 == 0, 1, -1).astype(np.int8)
    alt_w = (alt_rows[:, None] * np.ones(COLS, dtype=np.int8)[None, :]).astype(np.int8)
    ramp_w = (np.arange(ROWS * COLS).reshape(ROWS, COLS) % 255 - 127).astype(np.int8)
    weight_sets = {
        "randW": randint8(g, (ROWS, COLS)),
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
        "arand": randint8(g, (ROWS,)),
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
    cs = edge_cases()
    return np.stack([c[1] for c in cs]), np.stack([c[2] for c in cs])



# --------------------------------------------------------------------------
class BitSplit(unittest.TestCase):
    """Spec 4.2 and 4.6 item 1."""

    def test_rebuilds_every_activation(self):
        a = np.arange(-128, 128, dtype=np.int64)
        planes = bit_planes(a)                           # (8, 256)
        weights = np.array([1, 2, 4, 8, 16, 32, 64, -128])
        rebuilt = (planes * weights[:, None]).sum(axis=0)
        np.testing.assert_array_equal(rebuilt, a)

    def test_bits_are_zero_or_one(self):
        planes = bit_planes(np.arange(-128, 128, dtype=np.int64))
        self.assertTrue(np.isin(planes, (0, 1)).all())

    def test_spec_table(self):
        # spec 4.2 examples: value -> pattern x7..x0
        patterns = {5: "00000101", 127: "01111111", -1: "11111111",
                    -3: "11111101", -128: "10000000"}
        for v, bits in patterns.items():
            got = "".join(str(int(bit_plane(np.int64(v), b))) for b in range(7, -1, -1))
            self.assertEqual(got, bits, f"x={v}")

    def test_matches_python_two_complement(self):
        for v in range(-128, 128):
            for b in range(8):
                self.assertEqual(int(bit_plane(np.int64(v), b)), ((v & 0xFF) >> b) & 1)

    def test_bad_bit_index(self):
        for b in (-1, 8, 9):
            with self.assertRaises(ValueError):
                bit_plane(np.int64(0), b)


class WorkedExample(unittest.TestCase):
    """Spec section 7, every cycle."""

    P = [2, -1, 2, 4, 4, 4, 4, 8]
    TERM = [2, -2, 8, 32, 64, 128, 256, -1024]
    T = [2, 0, 8, 40, 104, 232, 488, -536]
    BITS = [(0, 1, 1, 1), (0, 0, 0, 1), (0, 1, 1, 1), (0, 0, 1, 1),
            (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (1, 0, 1, 0)]

    def setUp(self):
        self.W, self.a = tm.worked_example()
        self.tile = CimTile()
        self.tile.load_weights(self.W)
        self.psum = self.tile.run(self.a)

    def test_direct_answer(self):
        self.assertEqual(int(np.dot([3, -2, 5, -1], [-128, 5, -3, 127])), -536)
        self.assertEqual(int(self.psum[0]), -536)

    def test_every_cycle(self):
        tr = self.tile.trace
        self.assertEqual(len(tr), 8)
        for b, rec in enumerate(tr):
            self.assertEqual(rec.bit_index, b)
            self.assertEqual(tuple(int(v) for v in rec.plane[:4]), self.BITS[b], f"bits b={b}")
            self.assertEqual(int(rec.P[0]), self.P[b], f"P b={b}")
            self.assertEqual(int(rec.term[0]), self.TERM[b], f"term b={b}")
            self.assertEqual(int(rec.T[0]), self.T[b], f"T b={b}")

    def test_unused_rows_and_columns_are_zero(self):
        self.assertTrue((self.psum[1:] == 0).all())
        for rec in self.tile.trace:
            self.assertTrue((rec.P[1:] == 0).all())
            self.assertTrue((rec.plane[4:] == 0).all())

    def test_adding_the_msb_would_give_1512(self):
        # spec 7.3: "if cycle 7 added instead of subtracting ... 488 + 1024 = 1512"
        W, a = self.W, self.a
        bad = bitserial_dot(W, a, term_fn=lambda P, b: P * (1 << b))
        self.assertEqual(int(bad[0]), 1512)


class RandomVsNumpy(unittest.TestCase):
    """Review gate: 100% match on 100,000 random vectors."""

    def _run(self, make_W, make_a, n=N_RANDOM, offset=0):
        g = rng(offset)
        bad = 0
        for start in range(0, n, CHUNK):
            m = min(CHUNK, n - start)
            W = make_W(g, (m, ROWS, COLS))
            a = make_a(g, (m, ROWS))
            bad += mismatches(W, a)
        self.assertEqual(bad, 0, f"{bad} of {n} vectors differ from numpy W.T @ a")

    def test_uniform_full_int8_range(self):
        self._run(lambda g, s: randint8(g, s), lambda g, s: randint8(g, s), offset=10)

    def test_extreme_biased_values(self):
        self._run(extreme_int8, extreme_int8, offset=11)

    def test_exported_weight_range(self):
        # Weights as exported never hit -128 (spec 2.2).
        self._run(lambda g, s: randint8(g, s, -127, 127), lambda g, s: randint8(g, s),
                  n=50_000, offset=12)

    def test_post_relu_activations(self):
        # Every activation after conv1 is 0..127 (spec 4.1): bit 7 is always 0.
        self._run(lambda g, s: randint8(g, s, -127, 127),
                  lambda g, s: randint8(g, s, 0, 127), n=50_000, offset=13)

    def test_sparse_activations(self):
        def sparse(g, s):
            x = randint8(g, s)
            return np.where(g.random(s) < 0.8, 0, x).astype(np.int8)
        self._run(lambda g, s: randint8(g, s), sparse, n=50_000, offset=14)

    def test_exhaustive_single_row_all_weight_activation_pairs(self):
        # All 256 x 256 (w, x) pairs, each placed at a random row and column,
        # everything else zero. The psum must be w*x in that column and 0 elsewhere.
        g = rng(15)
        w, x = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128), indexing="ij")
        w, x = w.ravel(), x.ravel()
        n = w.size
        row = g.integers(0, ROWS, n)
        col = g.integers(0, COLS, n)
        for s in range(0, n, 8192):
            sl = slice(s, s + 8192)
            m = len(w[sl])
            W = np.zeros((m, ROWS, COLS), np.int8)
            a = np.zeros((m, ROWS), np.int8)
            idx = np.arange(m)
            W[idx, row[sl], col[sl]] = w[sl]
            a[idx, row[sl]] = x[sl]
            got = bitserial_dot(W, a)
            want = np.zeros((m, COLS), np.int64)
            want[idx, col[sl]] = w[sl] * x[sl]
            np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(got, numpy_reference(W, a))


class EdgeCases(unittest.TestCase):
    """Runbook Step 1.4 and spec 8.2, 'Tile' bullet."""

    def test_all_edge_cases_match_numpy(self):
        W, a = edge_batch()
        got = bitserial_dot(W, a)
        want = numpy_reference(W, a)
        names = [c[0] for c in edge_cases()]
        for i in np.flatnonzero(np.any(got != want, axis=-1)):
            self.fail(f"edge case '{names[i]}': got {got[i]}, want {want[i]}")
        self.assertGreaterEqual(len(names), 150)   # guard against the list shrinking

    def test_named_runbook_cases_explicitly(self):
        g = rng(20)
        Wr = randint8(g, (ROWS, COLS))
        tile = CimTile()
        # all activations -128
        tile.load_weights(Wr)
        np.testing.assert_array_equal(tile.run(np.full(ROWS, -128, np.int8)),
                                      Wr.astype(np.int64).T @ np.full(ROWS, -128))
        # all activations +127
        np.testing.assert_array_equal(tile.run(np.full(ROWS, 127, np.int8)),
                                      Wr.astype(np.int64).T @ np.full(ROWS, 127))
        # all weights -128
        tile.load_weights(np.full((ROWS, COLS), -128, np.int8))
        ar = randint8(g, (ROWS,))
        np.testing.assert_array_equal(tile.run(ar), np.full(COLS, -128) * int(ar.astype(np.int64).sum()))
        # all zeros
        tile.load_weights(np.zeros((ROWS, COLS), np.int8))
        self.assertTrue((tile.run(np.zeros(ROWS, np.int8)) == 0).all())

    def test_extreme_values(self):
        tile = CimTile()
        tile.load_weights(np.full((ROWS, COLS), -128, np.int8))
        self.assertTrue((tile.run(np.full(ROWS, -128, np.int8)) == 262_144).all())
        tile.load_weights(np.full((ROWS, COLS), 127, np.int8))
        self.assertTrue((tile.run(np.full(ROWS, -128, np.int8)) == -260_096).all())
        self.assertTrue((tile.run(np.full(ROWS, 127, np.int8)) == 258_064).all())

    def test_single_minus_128_with_positive_weight_is_negative(self):
        for r in range(ROWS):
            W = np.zeros((ROWS, COLS), np.int8)
            W[r, :] = 7
            a = np.zeros(ROWS, np.int8)
            a[r] = -128
            tile = CimTile()
            tile.load_weights(W)
            self.assertTrue((tile.run(a) == -896).all(), f"row {r}")

    def test_orientation_is_W_transpose_a(self):
        W = (np.arange(ROWS * COLS).reshape(ROWS, COLS) % 255 - 127).astype(np.int8)
        a = (np.arange(ROWS) * 3 - 20).astype(np.int8)
        tile = CimTile()
        tile.load_weights(W)
        got = tile.run(a)
        np.testing.assert_array_equal(got, W.astype(np.int64).T @ a.astype(np.int64))
        # and it is NOT the transposed product, which has the same shape here
        self.assertFalse(np.array_equal(got, W.astype(np.int64) @ a.astype(np.int64)))


class PerCycle(unittest.TestCase):
    """What the RTL waveform is compared against (spec 4.4 and 8.3)."""

    def test_each_cycle_matches_independent_loop_oracle(self):
        # Pure-Python ints, nested loops, straight from spec 4.3 and 4.4.
        g = rng(30)
        W = extreme_int8(g, (ROWS, COLS))
        a = extreme_int8(g, (ROWS,))
        tile = CimTile()
        tile.load_weights(W)
        tile.run(a)
        T = [0] * COLS
        for b in range(8):
            rec = tile.trace[b]
            for c in range(COLS):
                P = sum(int(W[r, c]) * (((int(a[r]) & 0xFF) >> b) & 1) for r in range(ROWS))
                self.assertEqual(int(rec.P[c]), P, f"P b={b} c={c}")
                T[c] = T[c] + P * (1 << b) if b < 7 else T[c] - P * (1 << 7)
                self.assertEqual(int(rec.T[c]), T[c], f"T b={b} c={c}")

    def test_partial_sum_invariant(self):
        # After cycle b < 7, T = W.T @ (the low b+1 bits of each activation, unsigned).
        # After cycle 7, T = W.T @ a. Checked on 5,000 random vectors.
        g = rng(31)
        n = 5000
        W = extreme_int8(g, (n, ROWS, COLS))
        a = extreme_int8(g, (n, ROWS))
        _, tr = bitserial_dot(W, a, trace=True)
        for b in range(8):
            if b < 7:
                low = (a.astype(np.int64) & 0xFF) & ((1 << (b + 1)) - 1)
            else:
                low = a.astype(np.int64)
            np.testing.assert_array_equal(tr["T"][b], numpy_reference(W, low), f"b={b}")

    def test_P_is_sum_of_included_weights(self):
        g = rng(32)
        W = randint8(g, (2000, ROWS, COLS))
        a = randint8(g, (2000, ROWS))
        _, tr = bitserial_dot(W, a, trace=True)
        for b in range(8):
            plane = (((a.astype(np.int64) & 0xFF) >> b) & 1)
            np.testing.assert_array_equal(tr["plane"][b], plane)
            np.testing.assert_array_equal(tr["P"][b], numpy_reference(W, plane), f"b={b}")

    def test_adder_tree_equals_plain_sum(self):
        g = rng(33)
        x = randint8(g, (500, ROWS, COLS)).astype(np.int64)
        np.testing.assert_array_equal(tm.adder_tree(x), x.sum(axis=-2))

    def test_and_mask_is_pass_or_zero(self):
        g = rng(34)
        W = randint8(g, (ROWS, COLS)).astype(np.int64)
        ones = np.ones(ROWS, np.int64)
        zeros = np.zeros(ROWS, np.int64)
        np.testing.assert_array_equal(tm.and_mask(W, ones), W)
        np.testing.assert_array_equal(tm.and_mask(W, zeros), np.zeros_like(W))

    def test_msb_cycle_subtracts_and_others_add(self):
        P = np.array([5, -3], np.int64)
        for b in range(7):
            np.testing.assert_array_equal(shift_term(P, b), P * (1 << b))
        np.testing.assert_array_equal(shift_term(P, 7), -P * 128)

    def test_batch_path_equals_cycle_model_cycle_by_cycle(self):
        g = rng(35)
        n = 300
        W = extreme_int8(g, (n, ROWS, COLS))
        a = extreme_int8(g, (n, ROWS))
        psum, tr = bitserial_dot(W, a, trace=True)
        tile = CimTile()
        for i in range(n):
            tile.load_weights(W[i])
            np.testing.assert_array_equal(tile.run(a[i]), psum[i])
            for b, rec in enumerate(tile.trace):
                np.testing.assert_array_equal(rec.plane, tr["plane"][b][i])
                np.testing.assert_array_equal(rec.P, tr["P"][b][i])
                np.testing.assert_array_equal(rec.term, tr["term"][b][i])
                np.testing.assert_array_equal(rec.T, tr["T"][b][i])


class Widths(unittest.TestCase):
    """Spec 6.4: P needs 12 bits, T needs 20 bits, and both are enough."""

    def test_observed_extremes(self):
        W = np.full((ROWS, COLS), -128, np.int8)
        a = np.full(ROWS, -128, np.int8)
        _, tr = bitserial_dot(W, a, trace=True)
        self.assertEqual(int(tr["P"].min()), -2048)         # 16 * -128, fits 12 bits
        self.assertEqual(int(tr["T"].max()), 262_144)       # one more than 19 bits hold

    def test_random_stays_inside_widths(self):
        g = rng(40)
        W = extreme_int8(g, (20_000, ROWS, COLS))
        a = extreme_int8(g, (20_000, ROWS))
        _, tr = bitserial_dot(W, a, trace=True)             # raises if a width is exceeded
        self.assertGreaterEqual(int(tr["P"].min()), -2048)
        self.assertLessEqual(int(tr["P"].max()), 2047)
        self.assertGreaterEqual(int(tr["T"].min()), -(1 << 19))
        self.assertLessEqual(int(tr["T"].max()), (1 << 19) - 1)

    def test_t_needs_20_bits_not_19(self):
        W = np.full((ROWS, COLS), -128, np.int8)
        a = np.full(ROWS, -128, np.int8)
        with self.assertRaises(TileOverflowError):
            bitserial_dot(W, a, t_bits=19)
        self.assertTrue((bitserial_dot(W, a, t_bits=20) == 262_144).all())

    def test_p_needs_12_bits_not_11(self):
        W = np.full((ROWS, COLS), -128, np.int8)
        a = np.full(ROWS, -1, np.int8)                      # every bit-plane is all ones
        with self.assertRaises(TileOverflowError):
            bitserial_dot(W, a, p_bits=11)
        bitserial_dot(W, a, p_bits=12)

    def test_cycle_model_checks_widths_too(self):
        tile = CimTile(t_bits=19)
        tile.load_weights(np.full((ROWS, COLS), -128, np.int8))
        with self.assertRaises(TileOverflowError):
            tile.run(np.full(ROWS, -128, np.int8))


class CycleApi(unittest.TestCase):
    """Schedule, state and weight-stationary behaviour (spec 4.4)."""

    def setUp(self):
        g = rng(50)
        self.W = randint8(g, (ROWS, COLS))
        self.tile = CimTile()
        self.tile.load_weights(self.W)

    def test_always_8_cycles_whatever_the_data(self):
        g = rng(51)
        for a in (np.zeros(ROWS, np.int8), np.full(ROWS, 127, np.int8),
                  np.full(ROWS, -128, np.int8), randint8(g, (ROWS,))):
            before = self.tile.total_cycles
            self.tile.run(a)
            self.assertEqual(self.tile.total_cycles - before, 8)
            self.assertEqual(len(self.tile.trace), 8)

    def test_manual_stepping_equals_run(self):
        a = randint8(rng(52), (ROWS,))
        want = self.tile.run(a)
        self.tile.clear()
        for b in range(8):
            self.assertFalse(self.tile.valid)
            self.tile.step(bit_plane(a.astype(np.int64), b), b)
        self.assertTrue(self.tile.valid)
        np.testing.assert_array_equal(self.tile.result, want)

    def test_result_not_readable_early(self):
        self.tile.clear()
        self.tile.step(np.zeros(ROWS, np.int64), 0)
        with self.assertRaises(TileScheduleError):
            _ = self.tile.result

    def test_schedule_is_lsb_first_and_in_order(self):
        self.tile.clear()
        with self.assertRaises(TileScheduleError):
            self.tile.step(np.zeros(ROWS, np.int64), 1)      # must start at bit 0
        self.tile.step(np.zeros(ROWS, np.int64), 0)
        with self.assertRaises(TileScheduleError):
            self.tile.step(np.zeros(ROWS, np.int64), 0)      # no repeats
        with self.assertRaises(TileScheduleError):
            self.tile.step(np.zeros(ROWS, np.int64), 7)      # no skipping

    def test_no_ninth_cycle_without_clear(self):
        self.tile.run(np.zeros(ROWS, np.int8))
        with self.assertRaises(TileScheduleError):
            self.tile.step(np.zeros(ROWS, np.int64), 0)

    def test_accumulator_is_cleared_between_chunks(self):
        g = rng(53)
        a1, a2 = randint8(g, (ROWS,)), randint8(g, (ROWS,))
        self.tile.run(a1)
        got = self.tile.run(a2)                              # must not carry a1's T
        np.testing.assert_array_equal(got, self.W.astype(np.int64).T @ a2.astype(np.int64))

    def test_weights_are_stationary_and_reloadable(self):
        g = rng(54)
        a = randint8(g, (ROWS,))
        self.tile.run(a)
        self.tile.run(a)
        self.assertEqual(self.tile.weight_loads, 1)          # activations moved, weights did not
        W2 = randint8(g, (ROWS, COLS))
        self.tile.load_weights(W2)
        self.assertEqual(self.tile.weight_loads, 2)
        np.testing.assert_array_equal(self.tile.run(a), W2.astype(np.int64).T @ a.astype(np.int64))
        self.tile.load_weights(self.W)                       # back to the first set
        np.testing.assert_array_equal(self.tile.run(a), self.W.astype(np.int64).T @ a.astype(np.int64))

    def test_inputs_are_not_modified(self):
        g = rng(55)
        W, a = randint8(g, (ROWS, COLS)), randint8(g, (ROWS,))
        W0, a0 = W.copy(), a.copy()
        tile = CimTile()
        tile.load_weights(W)
        tile.run(a)
        np.testing.assert_array_equal(W, W0)
        np.testing.assert_array_equal(a, a0)
        W[:] = 0                                             # caller edits must not reach the tile
        np.testing.assert_array_equal(tile.weights, W0)

    def test_dots_done_counter(self):
        before = self.tile.dots_done
        self.tile.run(np.zeros(ROWS, np.int8))
        self.tile.run(np.zeros(ROWS, np.int8))
        self.assertEqual(self.tile.dots_done - before, 2)

    def test_no_weights_loaded(self):
        t = CimTile()
        with self.assertRaises(TileScheduleError):
            t.run(np.zeros(ROWS, np.int8))

    def test_integer_dtypes_everywhere(self):
        self.tile.run(randint8(rng(56), (ROWS,)))
        for rec in self.tile.trace:
            for arr in (rec.plane, rec.P, rec.term, rec.T):
                self.assertEqual(arr.dtype, np.int64)
        self.assertEqual(self.tile.result.dtype, np.int64)


class BpcAndValidation(unittest.TestCase):
    def test_only_bpc_1_exists(self):
        CimTile(bpc=1)
        for bpc in (2, 4, 8):
            with self.assertRaises(NotImplementedError):
                CimTile(bpc=bpc)
            with self.assertRaises(NotImplementedError):
                bitserial_dot(np.zeros((ROWS, COLS), np.int8), np.zeros(ROWS, np.int8), bpc=bpc)
        for bpc in (0, 3, 16, -1):
            with self.assertRaises(ValueError):
                CimTile(bpc=bpc)

    def test_rejects_floats_and_bools(self):
        t = CimTile()
        with self.assertRaises(TypeError):
            t.load_weights(np.zeros((ROWS, COLS), np.float32))
        t.load_weights(np.zeros((ROWS, COLS), np.int8))
        with self.assertRaises(TypeError):
            t.run(np.zeros(ROWS, np.float64))
        with self.assertRaises(TypeError):
            t.run(np.zeros(ROWS, bool))

    def test_rejects_out_of_range_values(self):
        t = CimTile()
        with self.assertRaises(ValueError):
            t.load_weights(np.full((ROWS, COLS), 128, np.int16))
        t.load_weights(np.zeros((ROWS, COLS), np.int8))
        with self.assertRaises(ValueError):
            t.run(np.full(ROWS, 200, np.int16))
        with self.assertRaises(ValueError):
            t.run(np.full(ROWS, -129, np.int16))
        with self.assertRaises(ValueError):                  # uint8 255 is not an int8 value
            t.run(np.full(ROWS, 255, np.uint8))

    def test_rejects_wrong_shapes(self):
        t = CimTile()
        with self.assertRaises(ValueError):
            t.load_weights(np.zeros((COLS, ROWS + 1), np.int8))
        with self.assertRaises(ValueError):
            t.load_weights(np.zeros((2, ROWS, COLS), np.int8))
        t.load_weights(np.zeros((ROWS, COLS), np.int8))
        with self.assertRaises(ValueError):
            t.run(np.zeros(ROWS - 1, np.int8))
        with self.assertRaises(ValueError):
            t.run(np.zeros((2, ROWS), np.int8))
        with self.assertRaises(ValueError):
            t.step(np.full(ROWS, 2, np.int64), 0)            # a plane holds only 0 and 1
        with self.assertRaises(ValueError):
            bitserial_dot(np.zeros((3, ROWS, COLS), np.int8), np.zeros((4, ROWS), np.int8))


class BrokenTilesAreCaught(unittest.TestCase):
    """The tests must be able to fail. Each wrong tile below is caught by the
    directed edge cases alone, so a pass is not luck from random data. The
    reviewer's check 'change the MSB subtract to an add and see it fail'
    (runbook Step 3.1) is mutant 1."""

    MUTANTS = {
        "MSB added instead of subtracted":
            lambda P, b: P * (1 << b),
        "subtract on bit 6 instead of bit 7":
            lambda P, b: -P * (1 << b) if b == 6 else P * (1 << b),
        "MSB subtracted without the << 7":
            lambda P, b: -P if b == 7 else P * (1 << b),
        "bit 7 ignored (7 cycles only)":
            lambda P, b: np.zeros_like(P) if b == 7 else P * (1 << b),
        "bits shifted by b+1":
            lambda P, b: -P * (1 << (b + 1)) if b == 7 else P * (1 << (b + 1)),
    }

    def test_each_mutant_fails_the_edge_cases(self):
        W, a = edge_batch()
        for name, fn in self.MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertGreater(mismatches(W, a, term_fn=fn, t_bits=40, p_bits=40), 0)

    def test_each_mutant_fails_random_too(self):
        g = rng(60)
        W, a = randint8(g, (2000, ROWS, COLS)), randint8(g, (2000, ROWS))
        for name, fn in self.MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertGreater(mismatches(W, a, term_fn=fn, t_bits=40, p_bits=40), 0)

    def test_cycle_model_mutant_is_caught(self):
        tile = CimTile(term_fn=self.MUTANTS["MSB added instead of subtracted"], t_bits=40)
        W, a = tm.worked_example()
        tile.load_weights(W)
        self.assertEqual(int(tile.run(a)[0]), 1512)          # the spec's wrong answer, not -536

    def test_transposed_weights_are_caught(self):
        W, a = edge_batch()
        got = bitserial_dot(np.swapaxes(W, -1, -2), a)
        self.assertGreater(int(np.count_nonzero(np.any(got != numpy_reference(W, a), axis=-1))), 0)

    def test_correct_tile_is_not_flagged(self):
        W, a = edge_batch()
        self.assertEqual(mismatches(W, a), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
