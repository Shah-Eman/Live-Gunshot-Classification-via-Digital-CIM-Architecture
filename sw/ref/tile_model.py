#!/usr/bin/env python3
"""Cycle-level Python model of the 16 x 16 bit-serial CIM tile (runbook Step 1.4).

This is the bridge between the "math" reference (`ref_int.py`) and the RTL
(`rtl/cim/cim_tile.sv`). It implements `docs/arith_spec.md` section 4 and the
register widths of section 6.4, and nothing else: no bias, no requantization,
no layer logic. Those live outside the tile (spec sections 3 and 5).

What the tile computes, for one chunk of 16 inputs (spec 4.1)::

    psum[c] = sum_{r=0}^{15} W[r][c] * x[r]          c = 0..15

Weights are int8 (the tile must also be right for -128), activations are int8.

Orientation of the weight array
-------------------------------
`W` has shape (ROWS, COLS) = (16, 16): `W[r, c]` is the weight stored at tile
row `r`, column `c`. Row `r` receives activation `x[r]`; column `c` is one
output channel. So the golden answer is the runbook's numpy `W.T @ a`.
The spec writes the same weight as `W[c][r]` (column first); that is this
array's `W[r, c]`. See `docs/arith_spec.md` 4.1.

Schedule (spec 4.4), for every column at the same time::

    T = 0                                  # cleared at the start of every chunk
    for b in 0..7:                         # LSB first, one bit-plane per cycle
        P = sum_r W[r][c] * bit_b(x[r])    # AND mask + column adder tree
        T = T + (P << b)   if b < 7        # bits 0..6 have weight +2^b
        T = T - (P << 7)   if b == 7       # bit 7 has weight -128
    psum[c] = T

Spec cross-reference (for the line-by-line review gate)
-------------------------------------------------------
=====================================  ==============================================
spec                                   code
=====================================  ==============================================
4.2  bit split, bit_b(x)               `bit_plane`, `bit_planes`
4.3  AND of weight with activation bit `and_mask`
4.3  column sum P_b[c]                 `adder_tree`
4.4  T += P << b, T -= P << 7          `shift_term`, `CimTile.step`
4.4  T cleared per chunk, 8 cycles     `CimTile.clear`, `CimTile.run`
4.5  signed shift is multiply by 2^b   `shift_term` (uses `*`, never `<<` on P)
4.6  P and T never leave their widths  `_check_width` (P: 12 bit, T: 20 bit)
6.4  P 12 bits signed, T 20 bits       `P_BITS`, `T_BITS`
=====================================  ==============================================

Rules this file follows (spec 1.1 and runbook Step 1.3):

* Integers only. Every array is int64 inside the datapath (spec 3.2 and 4.5:
  never a narrow dtype, so nothing can wrap). Floats are rejected at the door.
* All 8 cycles always run, whatever the data (the cycle count is data
  independent, spec 4.4).
* `BPC` (bit-planes per cycle) is a parameter, but only `BPC = 1` exists. The
  values 2, 4, 8 are added in Step 2.5 with the spec addendum 4.7 (Decision
  D24); asking for them now raises `NotImplementedError`, never a silent
  wrong answer.

Two ways to use it
------------------
* `CimTile`: the stateful, cycle-by-cycle model with the same ports the RTL has
  (`load_weights`, `clear`, `step(plane, bit_index)`). Use this when you need
  to see or compare individual cycles, as in spec 8.3 and in Step 3.1 debug.
* `bitserial_dot(W, a)`: the same datapath, vectorised over any number of
  leading batch dimensions, for bulk testing (100,000 vectors). It calls the
  very same `bit_plane` / `and_mask` / `adder_tree` / `shift_term` functions,
  so it is the same arithmetic, not a second implementation. The tests check
  the two paths against each other, cycle by cycle.

Run `python sw/ref/tile_model.py` to print the spec section 7 worked example
cycle by cycle (the table the Step 1.2 review gate asked two people to redo).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# --------------------------------------------------------------------------
# Constants fixed by the spec
# --------------------------------------------------------------------------
ROWS = 16               # R: tile rows (spec 5.3, fixed)
COLS = 16               # C: tile columns
ABITS = 8               # activation bits = bit-cycles per dot product at BPC=1
WBITS = 8               # weight bits
CYCLES_PER_DOT = ABITS  # 8 / BPC with BPC = 1
MSB = ABITS - 1         # bit 7, the one with weight -128

P_BITS = 12             # bit-plane column sum, signed (spec 6.4: |P| <= 2,048)
T_BITS = 20             # tile accumulator, signed (spec 6.4: |T| <= 262,144)

INT8_MIN, INT8_MAX = -128, 127

SUPPORTED_BPC = (1,)
_FUTURE_BPC = (2, 4, 8)


class TileOverflowError(AssertionError):
    """A value left its register width. The spec (6.4) says this never happens."""


class TileScheduleError(RuntimeError):
    """The bit-cycles were not issued as the fixed schedule requires (4.4)."""


# --------------------------------------------------------------------------
# Input checking
# --------------------------------------------------------------------------
def _as_int64(name: str, x, lo: int, hi: int, tail: tuple[int, ...]) -> np.ndarray:
    """Validate `x` as an integer array in [lo, hi] ending in shape `tail`.

    Returns an int64 copy. Floats and bools are rejected rather than converted:
    the datapath is integer only, and a float that happens to hold an integer
    value would hide a bug upstream.
    """
    arr = np.asarray(x)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array, got dtype {arr.dtype}")
    n = len(tail)
    if arr.ndim < n or arr.shape[arr.ndim - n:] != tail:
        raise ValueError(f"{name} must end in shape {tail}, got {arr.shape}")
    if arr.size and (int(arr.min()) < lo or int(arr.max()) > hi):
        raise ValueError(
            f"{name} must lie in [{lo}, {hi}], got [{int(arr.min())}, {int(arr.max())}]")
    return arr.astype(np.int64)


def _check_width(name: str, x: np.ndarray, bits: int) -> None:
    """Assert every element of `x` fits a signed register of `bits` bits."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if x.size and (int(x.min()) < lo or int(x.max()) > hi):
        raise TileOverflowError(
            f"{name} left its {bits}-bit signed range [{lo}, {hi}]: "
            f"min {int(x.min())}, max {int(x.max())}")


def _check_bpc(bpc: int) -> None:
    if bpc in SUPPORTED_BPC:
        return
    if bpc in _FUTURE_BPC:
        raise NotImplementedError(
            f"BPC={bpc} is added in runbook Step 2.5 together with the spec "
            f"addendum 4.7 (Decision D24). Only BPC=1 (spec section 4) exists now.")
    raise ValueError(f"BPC must be one of 1, 2, 4, 8; got {bpc!r}")


# --------------------------------------------------------------------------
# The datapath: four small pure functions, one per hardware block
# --------------------------------------------------------------------------
def bit_plane(a: np.ndarray, b: int) -> np.ndarray:
    """Bit `b` of the 8-bit two's-complement pattern of every activation (4.2).

    `((a & 0xFF) >> b) & 1`. Taking the bit of the pattern, not of |a|, is what
    makes negative activations work. `a` is int64 with values in [-128, 127].
    """
    if not 0 <= b < ABITS:
        raise ValueError(f"bit index must be 0..{ABITS - 1}, got {b}")
    return ((a & 0xFF) >> b) & 1


def bit_planes(a: np.ndarray) -> np.ndarray:
    """All 8 bit-planes, shape (8, ..., ROWS): plane `b` is `bit_plane(a, b)`."""
    return np.stack([bit_plane(a, b) for b in range(ABITS)])


def and_mask(W: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Gate every stored weight with its row's activation bit (4.3).

    Hardware ANDs each weight bit with the row's bit-plane bit. On a
    two's-complement word that is `W & (-bit)`: bit = 1 gives all-ones (the
    weight passes unchanged, sign included), bit = 0 gives all-zeros. No
    multiplier is involved.

    W: (..., ROWS, COLS), plane: (..., ROWS) of 0/1 -> (..., ROWS, COLS).
    """
    return W & (-plane)[..., :, None]


def adder_tree(masked: np.ndarray) -> np.ndarray:
    """Column adder tree: sum the 16 rows of every column (4.3).

    Four levels of pairwise adds (16 -> 8 -> 4 -> 2 -> 1), as the RTL tree.
    (..., ROWS, COLS) -> (..., COLS). Equals `masked.sum(axis=-2)`; the tests
    check that.
    """
    n = masked.shape[-2]
    if n & (n - 1):
        raise ValueError("the adder tree needs a power-of-two number of rows")
    x = masked
    while n > 1:
        x = x[..., 0::2, :] + x[..., 1::2, :]
        n //= 2
    return x[..., 0, :]


def shift_term(P: np.ndarray, b: int) -> np.ndarray:
    """The signed value added to T on bit-cycle `b` (4.4).

    +P * 2^b for b = 0..6 and -P * 2^7 for b = 7: the top bit of a
    two's-complement number is worth -128. Written as a multiplication by
    2^b, exactly as spec 4.5 prescribes, because P can be negative.
    """
    scaled = P * (1 << b)
    return -scaled if b == MSB else scaled


TermFn = Callable[[np.ndarray, int], np.ndarray]


# --------------------------------------------------------------------------
# Per-cycle record (what the RTL waveform is compared against, spec 8.3)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CycleRecord:
    """Everything visible on one bit-cycle."""
    cycle: int            # 0..7, the position in the schedule
    bit_index: int        # b: which bit-plane was broadcast
    plane: np.ndarray     # (ROWS,)  bit b of every activation, 0/1
    P: np.ndarray         # (COLS,)  column sums of the masked weights
    term: np.ndarray      # (COLS,)  signed value added to T this cycle
    T: np.ndarray         # (COLS,)  accumulator after this cycle


# --------------------------------------------------------------------------
# The stateful, cycle-level tile
# --------------------------------------------------------------------------
class CimTile:
    """16 x 16 weight-stationary bit-serial tile, one bit-plane per cycle.

    Ports mirror the RTL (runbook Step 3.1): a weight-load port, a clear, and a
    per-cycle step that takes a bit-plane of the 16 activations plus the bit
    index. Weights stay put between dot products; only activations move.

    Typical use::

        tile = CimTile()
        tile.load_weights(W)             # (16, 16) int8
        psum = tile.run(a)               # (16,) int64, == W.T @ a
        for rec in tile.trace: ...       # the 8 CycleRecords of that run

    or, cycle by cycle, as a controller would::

        tile.clear()
        for b in range(8):
            rec = tile.step(bit_plane(a64, b), b)
        psum = tile.result

    `p_bits`, `t_bits` and `term_fn` exist so the tests can narrow a register
    or break the MSB subtract on purpose. Leave them at their defaults.
    """

    def __init__(self, bpc: int = 1, *, p_bits: int = P_BITS, t_bits: int = T_BITS,
                 term_fn: TermFn = shift_term):
        _check_bpc(bpc)
        self.bpc = bpc
        self._p_bits = p_bits
        self._t_bits = t_bits
        self._term_fn = term_fn
        self._W: Optional[np.ndarray] = None
        self._T = np.zeros(COLS, dtype=np.int64)
        self._next_bit = 0
        self._trace: list[CycleRecord] = []
        # Counters, in the spirit of the engine's perf counters (Step 2.2).
        self.weight_loads = 0
        self.total_cycles = 0
        self.dots_done = 0

    # -- weight port ------------------------------------------------------
    def load_weights(self, W) -> None:
        """Write all 16 x 16 weights. `W[r, c]` is row r, column c, int8."""
        self._W = _as_int64("weights", W, INT8_MIN, INT8_MAX, (ROWS, COLS))
        if self._W.ndim != 2:
            raise ValueError(f"weights must have shape ({ROWS}, {COLS}), got {self._W.shape}")
        self.weight_loads += 1
        self.clear()

    @property
    def weights(self) -> np.ndarray:
        """A copy of the stored weights, shape (ROWS, COLS)."""
        if self._W is None:
            raise TileScheduleError("no weights loaded")
        return self._W.copy()

    # -- control ----------------------------------------------------------
    def clear(self) -> None:
        """Start of a chunk: T = 0 and the schedule restarts at bit 0 (4.4)."""
        self._T = np.zeros(COLS, dtype=np.int64)
        self._next_bit = 0
        self._trace = []

    @property
    def valid(self) -> bool:
        """True once all 8 bit-cycles of the current chunk have run."""
        return self._next_bit == CYCLES_PER_DOT

    @property
    def cycle(self) -> int:
        """How many bit-cycles of the current chunk have run (0..8)."""
        return self._next_bit

    @property
    def result(self) -> np.ndarray:
        """psum[c] for the 16 columns. Only readable after the 8th cycle."""
        if not self.valid:
            raise TileScheduleError(
                f"result read after {self._next_bit} of {CYCLES_PER_DOT} bit-cycles")
        return self._T.copy()

    @property
    def trace(self) -> list[CycleRecord]:
        """The CycleRecords of the current (or last) chunk, in order."""
        return list(self._trace)

    # -- one clock cycle --------------------------------------------------
    def step(self, plane, bit_index: int) -> CycleRecord:
        """Run one bit-cycle: AND mask, column adder tree, shift-accumulate.

        `plane` is bit `bit_index` of the 16 activations (16 values, 0 or 1),
        which is what the RTL receives. The schedule is fixed: bit_index must be
        0, 1, ..., 7 in that order, after `clear()`.
        """
        if self._W is None:
            raise TileScheduleError("no weights loaded")
        if self.valid:
            raise TileScheduleError("chunk already finished; call clear() first")
        if bit_index != self._next_bit:
            raise TileScheduleError(
                f"bit-cycles run LSB first: expected bit {self._next_bit}, got {bit_index}")
        plane64 = _as_int64("bit-plane", plane, 0, 1, (ROWS,))
        if plane64.ndim != 1:
            raise ValueError(f"bit-plane must have shape ({ROWS},), got {plane64.shape}")

        P = adder_tree(and_mask(self._W, plane64))
        _check_width("P", P, self._p_bits)
        term = self._term_fn(P, bit_index)
        T = self._T + term
        _check_width("T", T, self._t_bits)

        rec = CycleRecord(cycle=self._next_bit, bit_index=bit_index, plane=plane64.copy(),
                          P=P.copy(), term=term.copy(), T=T.copy())
        self._T = T
        self._next_bit += 1
        self._trace.append(rec)
        self.total_cycles += 1
        if self.valid:
            self.dots_done += 1
        return rec

    # -- a whole dot product ---------------------------------------------
    def run(self, a) -> np.ndarray:
        """Clear, broadcast the 8 bit-planes of `a` LSB first, return psum.

        `a` is 16 int8 activations, one per row. Returns an int64 array of 16
        column results equal to `W.T @ a`. The 8 records are in `self.trace`.
        """
        a64 = _as_int64("activations", a, INT8_MIN, INT8_MAX, (ROWS,))
        if a64.ndim != 1:
            raise ValueError(f"activations must have shape ({ROWS},), got {a64.shape}")
        self.clear()
        for b in range(CYCLES_PER_DOT):
            self.step(bit_plane(a64, b), b)
        return self.result


# --------------------------------------------------------------------------
# Vectorised path: the same datapath over any leading batch dimensions
# --------------------------------------------------------------------------
def bitserial_dot(W, a, *, bpc: int = 1, p_bits: int = P_BITS, t_bits: int = T_BITS,
                  term_fn: TermFn = shift_term, trace: bool = False):
    """Bit-serial dot products for many (weights, activations) pairs at once.

    W: (..., ROWS, COLS) int8, a: (..., ROWS) int8, same leading dimensions.
    Returns psum, shape (..., COLS), int64.

    With `trace=True` returns `(psum, tr)` where `tr` is a dict of stacked
    per-cycle arrays, cycle axis first: `plane` (8, ..., ROWS) and `P`, `term`,
    `T` (8, ..., COLS). Entry `b` is exactly what `CimTile.step` records on
    cycle `b`.
    """
    _check_bpc(bpc)
    W64 = _as_int64("weights", W, INT8_MIN, INT8_MAX, (ROWS, COLS))
    a64 = _as_int64("activations", a, INT8_MIN, INT8_MAX, (ROWS,))
    if W64.shape[:-2] != a64.shape[:-1]:
        raise ValueError(
            f"batch shapes differ: weights {W64.shape[:-2]}, activations {a64.shape[:-1]}")

    T = np.zeros(a64.shape[:-1] + (COLS,), dtype=np.int64)
    planes, Ps, terms, Ts = [], [], [], []
    for b in range(CYCLES_PER_DOT):
        plane = bit_plane(a64, b)
        P = adder_tree(and_mask(W64, plane))
        _check_width("P", P, p_bits)
        term = term_fn(P, b)
        T = T + term
        _check_width("T", T, t_bits)
        if trace:
            planes.append(plane)
            Ps.append(P)
            terms.append(term)
            Ts.append(T)
    if not trace:
        return T
    return T, {"plane": np.stack(planes), "P": np.stack(Ps),
               "term": np.stack(terms), "T": np.stack(Ts)}


def numpy_reference(W, a) -> np.ndarray:
    """The golden answer: ordinary integer `W.T @ a` in int64 (runbook Step 1.4).

    Accepts the same shapes as `bitserial_dot`. It does not go through any of
    the tile code, so it is an independent check.
    """
    W64 = np.asarray(W).astype(np.int64)
    a64 = np.asarray(a).astype(np.int64)
    return np.matmul(np.swapaxes(W64, -1, -2), a64[..., None])[..., 0]


# --------------------------------------------------------------------------
# Worked example (spec section 7), printed when run as a script
# --------------------------------------------------------------------------
def worked_example() -> tuple[np.ndarray, np.ndarray]:
    """The spec 7.1 inputs on a real tile: 4 used rows, column 0, rest zero."""
    W = np.zeros((ROWS, COLS), dtype=np.int8)
    a = np.zeros(ROWS, dtype=np.int8)
    W[:4, 0] = [3, -2, 5, -1]
    a[:4] = [-128, 5, -3, 127]
    return W, a


def _print_worked_example() -> None:
    W, a = worked_example()
    tile = CimTile()
    tile.load_weights(W)
    psum = tile.run(a)
    print("Spec section 7 worked example, column 0 "
          "(weights 3, -2, 5, -1; activations -128, 5, -3, 127)\n")
    print("cycle b | bits of rows 0..3 | P | signed term added | T after")
    print("--------+-------------------+---+-------------------+--------")
    for rec in tile.trace:
        bits = ", ".join(str(int(v)) for v in rec.plane[:4])
        print(f"   {rec.bit_index}    | {bits:<17} | {int(rec.P[0]):>2}"
              f"| {int(rec.term[0]):>17} | {int(rec.T[0]):>6}")
    direct = int(numpy_reference(W, a)[0])
    print(f"\nbit-serial result {int(psum[0])}   direct sum(w*a) {direct}   "
          f"{'MATCH' if int(psum[0]) == direct else 'MISMATCH'}")


if __name__ == "__main__":
    _print_worked_example()
