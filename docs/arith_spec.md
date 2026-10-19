# Arithmetic Spec

Status: sections 1-3 written (T1.2.a), sections 4 and 6 written (T1.2.b), section 5 written (T1.2.c), section 7 written (T1.2.d), section 8 text written (T1.2.e, diagram still to draw), section 9 added.

Python (`sw/`), C (`hls/`) and RTL (`rtl/`) implement every operation in this file
bit-exactly. Where this file and an implementation disagree, the implementation is
wrong.

## 1. Number formats

### 1.1 Notation

- All integers are two's complement.
- `intN` is a signed N-bit integer with range `[-2^(N-1), 2^(N-1) - 1]`.
- `/` on real numbers is exact real division. Integer code in this spec never uses
  `/`, `//` or `%`.
- That rule is about the data path: activations, weights, biases, accumulators and
  requantization. Index and shape arithmetic in §5 (positions, chunk numbers, column
  groups, output lengths) works on non-negative integers and uses `div`, `mod`, `floor`
  and `ceil`. For non-negative integers these have their usual meaning and there is no
  rounding question: `a div b = floor(a / b)` and `a mod b = a − b × (a div b)`.
- `>>` is an arithmetic right shift. It is defined in 3.4.
- `clamp(v, lo, hi) = min(max(v, lo), hi)`, with `lo <= hi`. It is applied to the exact
  integer value of `v`, at full width, before any narrowing.
- `floor(v)` is the largest integer `<= v`.

### 1.2 Formats

| Quantity | Type | Range | Notes |
|---|---|---|---|
| activation `a_q` | `int8` | `[-128, 127]` | zero point is 0 |
| weight `w_q` | `int8` | `[-128, 127]` | in practice `[-127, 127]`, see 2.2 |
| bias `b_q` | `int32` | `[-2^31, 2^31 - 1]` | one per output channel |
| accumulator `acc` | `int32` | `[-2^31, 2^31 - 1]` | |
| requant multiplier `m0` | `int16` | `[16384, 32767]` | one per output channel, see 3.3 |
| requant shift `n` | unsigned 6-bit | `[1, 46]` | one per output channel, see 3.3 |
| requant product `p` | `int48` | `[-2^47, 2^47 - 1]` | 64-bit in C and Python, see 3.2 |

- `acc` never wraps. Section 6 proves this for the largest layer. Implementations do not
  saturate `acc`. The Python reference asserts `acc` stays in `int32` range.
- Scales `s` are positive float64 values. They exist only offline. The integer pipeline
  sees only `w_q`, `b_q`, `m0` and `n`.

## 2. Quantization

### 2.1 Scheme

- Quantization is symmetric. The zero point is 0 for every tensor.
- Weights use one scale per output channel: `s_w[c]`.
- Activations use one scale per tensor: `s_a`. Each layer output has one `s_a`. The
  network input has its own `s_in`.
- All float arithmetic in this section is IEEE 754 binary64 (float64). float32 inputs
  are converted to float64 first.

### 2.2 Scales

- Weight scale: `s_w[c] = max_abs_w[c] / 127.0`, where `max_abs_w[c]` is the largest
  `|w|` over all weights of output channel `c`. If `max_abs_w[c] == 0`, then
  `s_w[c] = 1.0`.
- With this scale, `|w / s_w[c]| <= 127`, so `w_q` never takes the value `-128`.
- Activation scale: `s_a = max_abs_a / 127.0`, where `max_abs_a` comes from calibration.
  The calibration method is outside this spec. `s_a` must be `> 0`.
- Scales are computed once and exported. Python, C and RTL use the exported integers
  derived from them. Nothing recomputes them.

### 2.3 Rounding mode

The rounding mode is round half toward positive infinity, called `round_half_up`:

```
round_half_up(v):
    f = floor(v)
    if v - f >= 0.5: return f + 1
    else:            return f
```

- `v - f` is exact in float64, so the comparison has no rounding error.
- Do not write it as `floor(v + 0.5)`. That form fails for
  `v = 0.49999999999999994`, where `v + 0.5` rounds to `1.0`.
- Do not use Python `round()`, `numpy.round`, `numpy.rint`, or C `round`, `lround`,
  `rint`, `nearbyint`. Their modes differ from this one.
- This is the same mode as requantization in 3.4. One rounding mode is used everywhere.

| `v` | `round_half_up(v)` |
|---|---|
| `2.5` | `3` |
| `1.4` | `1` |
| `0.5` | `1` |
| `-0.5` | `0` |
| `-1.5` | `-1` |
| `-2.5` | `-2` |
| `-2.6` | `-3` |

### 2.4 Quantize

For a finite float value `x` and scale `s`:

```
v   = x / s                         # float64 division, not x * (1 / s)
r   = round_half_up(v)              # integer-valued float64
x_q = clamp(r, -128, 127)           # clamp in float64, then convert to int8
```

- The conversion to integer happens after the clamp, so it is always exact and in range.
- `x` must be finite. The Python reference asserts this.
- Weights: `w_q = quantize(w, s_w[c])`.
- Network input: `a_q = quantize(x, s_in)`.
- Activations inside the network are not produced by this formula. They come only from
  requantization (section 3).

### 2.5 Bias

- Bias scale: `s_b[c] = s_in_layer * s_w[c]`, computed in float64 in that order, where
  `s_in_layer` is the activation scale of the layer input.
- `b_q[c] = clamp(round_half_up(b[c] / s_b[c]), -2^31, 2^31 - 1)`, converted to `int32`
  after the clamp.
- `b_q` is added to `acc` before requantization: `acc = b_q[c] + sum(w_q * a_q)`.

## 3. Requantization

Requantization maps an `int32` accumulator to an `int8` activation for output channel `c`.

### 3.1 Formula

```
p = acc * m0                     # int48 signed product
r = p + (1 << (n - 1))           # int48 signed, rounding constant
q = r >> n                       # arithmetic shift, see 3.4
y = clamp(q, lo, hi)             # int8 result
```

`m0` and `n` are the values for channel `c`. `lo` and `hi` are set in 3.5.

### 3.2 Widths

- `acc` is `int32`. `m0` is `int16` with value in `[16384, 32767]`.
- `p = acc * m0` is a 48-bit signed intermediate (`int48`).
- `|p| <= 2^31 * 32767 < 2^46`.
- `1 << (n - 1) <= 2^45` because `n <= 46`.
- So `|r| < 2^46 + 2^45 < 2^47`. `p` and `r` fit in `int48` and never wrap.
- C and Python compute `p`, `r` and `q` in 64-bit signed integers (`int64_t` in C,
  `int` or `numpy.int64` in Python). By the bound above, the 64-bit value equals the
  48-bit value. All three implementations give the same bits.
- The Python reference asserts `-2^47 <= r <= 2^47 - 1`.
- `q` is not narrowed before the clamp. The clamp in 3.5 reads the full-width `q`.

Width rules per language:

- Python: operate on Python `int` or `numpy.int64`. Never on `numpy.int32` or smaller.
- C: `p = (int64_t)acc * (int64_t)m0;` and `r = p + ((int64_t)1 << (n - 1));`. A plain
  `1 << (n - 1)` is a 32-bit `int` and overflows for `n > 31`.
- RTL: `acc` and `m0` are declared `signed`. `p` and `r` are `logic signed [47:0]`.
  The multiply sign-extends both operands to 48 bits. No operand in the expression is
  unsigned, because one unsigned operand makes the whole expression unsigned in
  SystemVerilog.

### 3.3 Choosing `m0` and `n`

The real multiplier is `M = (s_in_layer * s_w[c]) / s_out`, in float64, evaluated in
that order. `s_out` is the activation scale of the layer output. Section 5 may fold
extra factors into `M` (for example `1/L` for global average pooling) before this step.

```
(mant, k) = frexp(M)             # M = mant * 2^k, 0.5 <= mant < 1, exact
e  = k - 1                       # e = floor(log2(M)), exact
n  = 14 - e
m0 = round_half_up(M * 2^n)      # M * 2^n is exact in float64, lies in [2^14, 2^15)
if m0 == 32768:
    m0 = 16384
    n  = n - 1
require 1 <= n <= 46             # else export fails with an error
```

- `m0` always lands in `[16384, 32767]`.
- `n = 0` is not allowed, because `1 << (n - 1)` is undefined for it.
- Do not use `log2` to find `e`. It is not exact. Use `frexp`.
- `m0` and `n` are computed once at export, per output channel, and stored as integers.
  Python, C and RTL all read the same exported `m0` and `n`.

### 3.4 Shift and rounding

- `>>` is an arithmetic right shift. It fills vacated high bits with the sign bit.
- For any integer `r` and `n >= 1`, `r >> n == floor(r / 2^n)`. For negative `r` this
  rounds toward negative infinity, not toward zero.
- With the `+ (1 << (n - 1))` term, `q == floor(p / 2^n + 0.5)`. This is round half
  toward positive infinity, the same `round_half_up` mode as 2.3.
- Ties on negative values round up: `-1.5 -> -1`, `-2.5 -> -2`.
- Integer division that truncates toward zero gives wrong results for negative `r` and
  is not allowed.

Per language:

- Python: `r >> n` on `int` or `numpy.int64` is arithmetic. Use it as written. Do not
  use `int(r / 2**n)` or `r // 2**n` in place of the shift.
- C: right shift of a negative signed integer is implementation-defined. The C model
  uses this helper, which is arithmetic on every compiler:

  ```c
  static inline int64_t asr64(int64_t v, int n) {
      return v >= 0 ? (v >> n) : ~(~v >> n);
  }
  ```

- RTL: use `>>>` on a `signed` operand. `>>` in SystemVerilog is a logical shift and
  is wrong here. `>>>` on an unsigned operand is also logical and is wrong.

Examples with `n = 2` (`2^n = 4`):

| `p` | `p / 4` | `r = p + 2` | `q = r >> 2` |
|---|---|---|---|
| `6` | `1.5` | `8` | `2` |
| `5` | `1.25` | `7` | `1` |
| `2` | `0.5` | `4` | `1` |
| `-2` | `-0.5` | `0` | `0` |
| `-5` | `-1.25` | `-3` | `-1` |
| `-6` | `-1.5` | `-4` | `-1` |
| `-7` | `-1.75` | `-5` | `-2` |

### 3.5 Clamp and ReLU

- Without ReLU: `lo = -128`, `hi = 127`.
- With ReLU: `lo = 0`, `hi = 127`. ReLU is folded into the clamp. There is no separate
  ReLU operation.
- The clamp reads the full-width `q`. Truncating `q` to 8 bits and then clamping is
  wrong, because the low 8 bits of a large `q` can be any value.
- After the clamp, `y` fits in `int8` and is stored as `int8`.

### 3.6 End-to-end examples

`m0 = 20000`, `n = 20`, so `1 << (n - 1) = 524288` and `2^n = 1048576`.

| `acc` | `p` | `r` | `q` | `y` (no ReLU) | `y` (ReLU) |
|---|---|---|---|---|---|
| `1000` | `20000000` | `20524288` | `19` | `19` | `19` |
| `-1000` | `-20000000` | `-19475712` | `-19` | `-19` | `0` |

For `acc = -1000`, truncating division gives `-18`. The correct `q` is `-19`.

Saturation: `m0 = 30000`, `n = 16`, `acc = 100000`.

- `p = 3000000000`, `r = 3000032768`, `q = 45776`.
- `y = clamp(45776, -128, 127) = 127`.
- Truncating `q` to 8 bits first gives `-48`. That is wrong.

## 4. Bit-serial dot product

This section defines how the tile computes a dot product. It is the heart of the CIM engine: everything in §5 is
built out of this one operation. The tile does not contain multipliers. Instead it feeds the activations in one bit
at a time, adds up the weights whose activation bit is 1, and shifts each of those sums into place. The result is
exactly the ordinary integer dot product `Σ w × a`, just computed in a different order.

### 4.1 What the tile computes

The tile has 16 rows and 16 columns. Each row holds one activation, and each column holds the 16 weights of one
output channel. For one chunk (§5.3), column `c` computes

```
psum[c] = Σ_{r=0}^{15}  W[c][r] × x_r
```

where `x_r` is the int8 activation on row `r` and `W[c][r]` is the int8 weight stored at row `r`, column `c`. The
weights stay in the tile (weight-stationary). Only the activations move.

The ranges come from §1 and §2:

- Activations `x_r` are int8, `−128..127`. Only the network input (the input of `conv1`) can be negative. Every
  later activation is post-ReLU and lies in `0..127`, but it is still treated as signed int8.
- Weights `W[c][r]` are int8 in `−127..127`. The value `−128` never occurs in an exported model, because of the
  scale rule in §2.2. The tile itself must still give the right answer for a weight of `−128`, because the runbook's
  tile tests (Step 1.4) include "all weights −128". §6.4 sizes the tile registers for that case.

The bias is **not** part of this section. It is added once per output, outside the tile, as §5.3 describes.

### 4.2 Splitting an activation into bits

Every int8 activation `x` is written as its 8-bit two's-complement pattern, with bits `x_0` (least significant)
to `x_7` (most significant), each 0 or 1. The value of the pattern is

```
x = x_0 × 1 + x_1 × 2 + x_2 × 4 + x_3 × 8 + x_4 × 16 + x_5 × 32 + x_6 × 64 − x_7 × 128
```

The important detail is the **minus sign on the top bit**. In two's complement, bit 7 is worth −128, not +128.
Bits 0 to 6 add, bit 7 subtracts. Some examples:

| `x` | Bit pattern (`x_7 … x_0`) | Check |
|---|---|---|
| `5` | `0000 0101` | 4 + 1 = 5 |
| `127` | `0111 1111` | 64 + 32 + 16 + 8 + 4 + 2 + 1 = 127 |
| `−1` | `1111 1111` | 127 − 128 = −1 |
| `−3` | `1111 1101` | 125 − 128 = −3 |
| `−128` | `1000 0000` | 0 − 128 = −128 |

`bit_b(x)` means bit `b` of this 8-bit pattern. In Python that is `(x & 0xFF) >> b & 1`. Taking the bit of the
pattern, not of the absolute value, is what makes negative activations work.

### 4.3 The bit-plane sum

Bit `b` of all 16 activations together is called **bit-plane `b`**. For each bit-plane, column `c` adds up the
weights on the rows where that bit is 1:

```
P_b[c] = Σ_{r=0}^{15}  W[c][r] × bit_b(x_r)          for b = 0..7
```

Because `bit_b(x_r)` is either 0 or 1, the "multiplication" is really a choice: the weight is either included or
not. In hardware this is an AND of each weight with the activation bit, followed by the column's adder tree. No
multiplier is needed anywhere in the tile.

Putting §4.2 and §4.3 together gives the exact identity the tile relies on:

```
psum[c] = P_0[c] × 1 + P_1[c] × 2 + … + P_6[c] × 64 − P_7[c] × 128
```

This is just the sum of §4.1 regrouped by bit position, so it is exact. No rounding happens anywhere in §4.

### 4.4 The 8-cycle schedule

The schedule is fixed. Python (`tile_model.py`), C and RTL all follow it, so their per-cycle values can be compared
directly, not just the final results.

For every chunk, each column runs exactly **8 bit-cycles**, from the least significant bit to the most significant:

```
T = 0                                   # tile accumulator, cleared at the start of every chunk
for b = 0 .. 7:                         # one bit-plane per cycle, b = 0 first
    P = Σ_{r=0}^{15}  W[c][r] × bit_b(x_r)
    if b < 7:  T = T + (P << b)         # bits 0..6 have positive weight
    else:      T = T − (P << 7)         # bit 7 has weight −128, so it is subtracted
psum_q[c] = T                           # the chunk result used in §5.3
```

- `P << b` means `P × 2^b`, computed exactly. `P` can be negative, so this is a signed shift.
- **The last cycle subtracts.** Treating bit 7 as +128 gives the wrong answer for every negative activation. This is
  the single most likely bug in the tile, and the tests in §8 check it directly.
- All 8 cycles always run, even when bit 7 is known to be 0 (every layer after `conv1`). Then `P_7 = 0` and the
  last cycle subtracts 0. The cycle count does not depend on the data.
- `T` starts at 0 for every chunk. The bias is never added inside the tile and never added once per bit-cycle. It
  enters only once, as the starting value of `acc` in §5.3.
- After the 8th cycle, `T` equals `Σ W[c][r] × x_r` exactly. §5.3 then adds the chunk results together in order
  q = 0, 1, …, Q−1.

All 16 columns run the same 8 cycles at the same time, on the same activation bits. Each column just uses its own
weights.

One full output therefore takes `Q × 8` bit-cycles. For `conv1` (Q = 4) that is 32 cycles per output position.

### 4.5 Signed shifts in each language

`P << b` has to be a true multiplication by `2^b`, including for negative `P`. Each language has its own trap:

- **Python:** `P << b` on `int` or `numpy.int64` is exact. Do not use `numpy.int16` or smaller, because it can wrap.
- **C:** left-shifting a negative signed integer is undefined behaviour. Write `T += P * ((int32_t)1 << b);` and
  `T -= P * ((int32_t)1 << 7);` instead of shifting `P` itself.
- **RTL:** declare `P` and `T` as `signed`, and sign-extend `P` to the full width of `T` before shifting. Shifting
  a narrow `P` and then extending it loses the top bits. If any operand in the expression is unsigned, the whole
  expression becomes unsigned in SystemVerilog, so keep every operand signed.

The minimum widths for `P` and `T` are proved in §6.4.

### 4.6 What the tests must check

These checks are required for the tile model and the RTL. The full test list lives in §8.

1. The bit split of §4.2 rebuilds every activation from −128 to 127 exactly.
2. `T` after 8 cycles equals `Σ W × x`, computed directly with wide integers, for every test vector.
3. Bit 7 is subtracted: a test with `x = −128` on one row and a positive weight must give a negative result.
4. The bias is added once per output, not once per chunk and not once per cycle.
5. No `P`, `T` or `acc` value ever goes outside the widths of §6.
6. `acc` is compared before requantization, not only the final int8 output. A wrong accumulator can still give
   the right int8 value when the output saturates at 127, so checking only `y` can hide a bug.

The hand-computed, cycle-by-cycle example is in §7.

## 5. Layer types

This section defines every layer of the starting 1D-CNN (runbook Step 2.4) as plain integer operations. It uses the
number formats of §1, the requantization function of §3 and the bit-serial dot product of §4. Nothing here
introduces a new kind of arithmetic. Every layer is either a dot product (§4) followed by a requantization (§3), or a
comparison or sum on int8 values.

**Layer names.** Layers are referred to by name, never by position number. The names are `conv1`, `pool1`, `conv2`,
`pool2`, `conv3`, `gap`, `dense` and `argmax`. The reason is that the runbook counts only the conv layers ("layer 1",
"layer 2", "layer 3"), while counting every layer in the pipeline would make the second conv "layer 3". Using names
removes that confusion. Wherever the runbook or a file name says "layer 1" (for example `expected_l1.hex`), it means
`conv1`. "Layers 2 and 3" in the runbook's im2col description mean `conv2` and `conv3`.

Throughout this section:

- `requant(acc, c, lo, hi)` means the §3.1 operation
  `clamp((acc * m0[c] + (1 << (n[c] − 1))) >> n[c], lo, hi)`, using the layer's own per-channel `m0[c]` and `n[c]`.
- All indices start at 0. Ranges `a..b` include both ends.
- "int8" always means signed two's-complement int8, range −128..127, even where the values happen to be ≥ 0.

### 5.1 The network

| Name | Layer | Input shape (L × C) | Output shape (L × C) | Parameters | K = k × Cin | Chunks of 16 | MACs |
|------|-------|--------------------|----------------------|-----------|-------------|--------------|------|
| input | Input (int8 waveform) | n/a | 8000 × 1 | n/a | n/a | n/a | n/a |
| conv1 | Conv1D k=64, s=8, valid, 16 ch, ReLU | 8000 × 1 | 993 × 16 | 1,024 W + 16 B | 64 | 4 | 1,016,832 |
| pool1 | MaxPool1D 4, s=4, valid | 993 × 16 | 248 × 16 | n/a | n/a | n/a | n/a |
| conv2 | Conv1D k=9, s=1, same, 32 ch, ReLU | 248 × 16 | 248 × 32 | 4,608 W + 32 B | 144 | 9 | 1,142,784 |
| pool2 | MaxPool1D 4, s=4, valid | 248 × 32 | 62 × 32 | n/a | n/a | n/a | n/a |
| conv3 | Conv1D k=9, s=1, same, 32 ch, ReLU | 62 × 32 | 62 × 32 | 9,216 W + 32 B | 288 | 18 | 571,392 |
| gap | GlobalAvgPool (sum + requant, §5.6) | 62 × 32 | 32 | n/a | n/a | n/a | n/a |
| dense | Dense 32 → 5, no ReLU | 32 | 5 | 160 W + 5 B | 32 | 2 | 160 |
| argmax | Argmax | 5 | 1 (class index 0..4) | n/a | n/a | n/a | n/a |

Totals: 15,008 weights, 85 biases, 2,731,168 MACs. The per-layer table printed by `sw/models/cnn1d.py`
(Step 1.1) must reproduce these numbers exactly. If it doesn't, either the model or this table is wrong, and that is
fixed before anything else is built.

The largest dot product is `conv3`, with K = 288. §6 uses this for the accumulator bound.

### 5.2 Tensor layout

Every activation tensor is stored **position-major, channel-minor**: `A[p][c]`, where `p` is the position (time)
index and `c` is the channel index. This is Keras's `channels_last` order, so no transpose is needed between Keras and
the reference.

Per-channel parameters (`W`, `bias`, `m0`, `n`) are indexed by the output channel `c`, in the order of the last
axis of the Keras kernel (`kernel[:, :, c]`).

### 5.3 Convolution as a dot product (`conv1`, `conv2`, `conv3`)

A conv layer slides a small window along the signal. At each position it multiplies the window by the filter
weights, adds up the products, adds the bias and requantizes the result. That is a dot product, so the tile of §4
can do the work. This subsection pins down exactly which inputs each output reads and in what order they reach the
tile.

**Parameters of a conv layer:** kernel size `k`, stride `s`, left padding `P` (0 for valid), input length `L_in`,
input channels `Cin`, output channels `Cout`, output length `L_out`.

**Flattened input index.** One output reads `K = k × Cin` inputs. The flat index `j` of tap `t` (0..k−1) and input
channel `ci` (0..Cin−1) is

```
j = t × Cin + ci            (tap-major, channel-minor)
t = j div Cin,  ci = j mod Cin
```

**Weights.** `W[c][j] = kernel[t][ci][c]`, where `kernel` is the Keras Conv1D kernel of shape `(k, Cin, Cout)`.
This is the same as `kernel.reshape(K, Cout)[j][c]`.

**Input sample read by tap t of output p:**

```
i = s × p + t − P
x_j = A_in[i][ci]   if 0 ≤ i ≤ L_in − 1
x_j = 0             otherwise (padding)
```

The padding value is the int8 integer **0**. Activations are symmetric with no zero point (§2), so integer 0 is
exactly real 0, and padding with 0 matches Keras `padding='same'` exactly. A padded input is still fed to the
tile as an ordinary activation of value 0. It contributes nothing to the sum. Whether the hardware spends cycles
on it is a scheduling question for `mapping_spec.md`, not an arithmetic one.

**Chunks.** The tile has 16 rows, so the K inputs are split into `Q = ceil(K / 16)` chunks. Chunk `q` (0..Q−1)
holds flat indices `j = 16q .. 16q + 15`, in that order, on tile rows 0..15. Every K in this network is a multiple of
16, so here `Q` is simply `K / 16` and no chunk is partly empty. If a later model has a K that isn't a multiple of 16,
the last chunk is padded with rows whose weight is 0 and whose activation is 0, so it still holds 16 rows.

**Accumulation.**

```
acc[p][c] = bias[c] + Σ_{q=0}^{Q−1}  psum_q[p][c]
psum_q[p][c] = Σ_{r=0}^{15}  W[c][16q + r] × x_{16q + r}       (computed bit-serially, §4)
```

- `acc` is int32 and starts at `bias[c]` before chunk 0.
- Chunks are added in order q = 0, 1, …, Q−1. Integer addition is associative, and §6 proves no partial sum
  can overflow, so the order doesn't change the result. It is fixed anyway, so the reference `trace()` and the RTL
  waveforms show the same intermediate values.
- Requantization happens once per output, after the last chunk, never per chunk.

**Output.**

```
A_out[p][c] = requant(acc[p][c], c, lo = 0, hi = 127)       (ReLU folded in, §5.8)
```

**Output channels on the tile.** Each tile column computes one output channel. When `Cout > 16`, output channel
`c` belongs to column group `g = c div 16`, column `c mod 16`. Which group runs on which tile, and when, is
defined in `mapping_spec.md`. The arithmetic is per channel and does not depend on that choice.

### 5.4 Window table for each conv layer

**`conv1`: Conv1D k=64, s=8, valid.** `P = 0`, `Cin = 1`, so `j = t`.

```
L_out = floor((8000 − 64) / 8) + 1 = 993
Output p (0..992) reads samples  x[8p .. 8p + 63]
Chunk q (0..3) = samples 8p + 16q .. 8p + 16q + 15
First window: p = 0   → samples 0..63
Last window:  p = 992 → samples 7936..7999   (the last sample is used; nothing is dropped)
```

No padding ever occurs in `conv1`.

**`conv2`: Conv1D k=9, s=1, same.** `L_in = 248`, `Cin = 16`, `Cout = 32`.

```
Total padding = (L_out − 1) × s + k − L_in = 8  →  P = 4 on the left, 4 on the right
L_out = 248
Output p (0..247) reads positions p − 4 .. p + 4, all 16 channels
Chunk q (0..8) = tap t = q = position p − 4 + q, channels 0..15
Padded reads: p = 0..3 (left) and p = 244..247 (right)
```

Because `Cin = 16`, each chunk is exactly one tap across all 16 input channels.

**`conv3`: Conv1D k=9, s=1, same.** `L_in = 62`, `Cin = 32`, `Cout = 32`.

```
P = 4 left, 4 right;  L_out = 62
Output p (0..61) reads positions p − 4 .. p + 4, all 32 channels
Chunk q (0..17) = tap t = q div 2  (position p − 4 + q div 2),
                  channels 16 × (q mod 2) .. 16 × (q mod 2) + 15
Padded reads: p = 0..3 and p = 58..61
```

Here each tap has 32 channels, which is two chunks, so two consecutive chunks share one tap: the first holds
channels 0..15 and the second holds channels 16..31.

Example for review: in `conv3`, chunk q = 5 is tap 2, channels 16..31, so for output p it reads `A[p − 2][16..31]`.

**General rule for `same` padding** (if the model changes): total padding
`T = max((ceil(L_in / s) − 1) × s + k − L_in, 0)`, `P = floor(T / 2)` on the left, `T − P` on the right. This is
TensorFlow's rule: when T is odd, the extra zero goes on the **right**.

### 5.5 Max pooling (`pool1`, `pool2`)

Max pooling looks at small groups of neighbouring positions and keeps the largest value in each group. Pool size
is 4, stride is 4, and there is no padding:

```
L_out = floor(L_in / 4)
B[p][c] = max( A[4p][c], A[4p + 1][c], A[4p + 2][c], A[4p + 3][c] )      for p = 0..L_out − 1
```

- It runs on int8 values and produces int8 values. There is no requantization, and the scale doesn't change.
- `pool1`: `L_out = floor(993 / 4) = 248`. Inputs 0..991 are used. **Position 992 of `conv1` is computed but
  never read** (Keras `padding='valid'` drops it). The hardware still computes it, because `expected_l1.hex`
  contains all 993 positions.
- `pool2`: `L_out = 248 / 4 = 62`, nothing dropped.
- Equal values need no tie rule, because max returns the value, not an index.
- This is exact with respect to the float model: requantization and ReLU never reverse the order of two values
  (they are monotonic non-decreasing), so pooling after requantization gives the same result as quantizing after
  pooling.

### 5.6 Global average pooling (`gap`)

An average over 62 positions is a sum divided by 62, and division is expensive in hardware. So GAP is done in two
integer steps, with no division anywhere.

**Step 1, sum** (block `gap_sum`):

```
S[c] = Σ_{p=0}^{61} a3[p][c]          int32, for c = 0..31
```

Here `a3` is the output of `conv3` (§5.10). Its values are post-ReLU (0..127), so `0 ≤ S[c] ≤ 62 × 127 = 7,874`.
That fits easily in int32 (and in 14 bits). The order of the sum doesn't matter.

**Step 2, requantize to int8** using the §3 unit:

```
G[c] = requant(S[c], c, lo = 0, hi = 127)
```

`m0_gap` and `n_gap` come from the real multiplier

```
M_gap = s_3 / (62.0 * s_G)       # float64; first 62.0 * s_G, then s_3 divided by that
```

where `s_3` is the output scale of `conv3` and `s_G` is the GAP output scale (calibrated per tensor like every
activation). `quantize.py` evaluates `M_gap` exactly as written, in that order, and then turns it into
`(m0_gap, n_gap)` with the §3.3 procedure, unchanged. The order matters for the same reason as in §3.3: a different
order of float operations can move `m0` by one. The result satisfies `m0_gap / 2^n_gap ≈ M_gap`, with the usual
§3.3 rounding. The 1/L is folded into this multiplier, so no integer division happens in hardware. Because input and output are both
per tensor, all 32 channels use the same `(m0, n)`. The table is still stored per channel so that the same
requantization unit is reused unchanged.

> **Decision (needs team sign-off, recorded as D19 in `docs/decisions.md`, see also §5.11):** the runbook says to
> fold 1/L into the *next* layer's requantization. That would make the dense layer's inputs the 14-bit sums `S[c]`,
> which the 8-bit, 8-cycle tile of §4 can't take. This spec requantizes the sum to int8 first instead, so every
> tile input in the network is int8 and §4 holds unchanged. The extra rounding is part of the integer model, so its
> accuracy cost is measured in Step 2.6 like every other quantization effect.

### 5.7 Dense (`dense`)

A dense layer is a conv layer with `L_in = L_out = 1`, `k = 1`, `s = 1`, `P = 0`, so `j = ci`:

```
W[c][j] = kernel[j][c]         (Keras Dense kernel, shape (32, 5))
acc[c]  = bias[c] + Σ_{q=0}^{1} Σ_{r=0}^{15} W[c][16q + r] × G[16q + r]      (bit-serial, §4)
Z[c]    = requant(acc[c], c, lo = −128, hi = 127)                          for c = 0..4
```

- **No ReLU:** `lo = −128`. The outputs `Z` are int8 logits.
- **Why a requantization is needed before argmax.** Weights are per channel, so each class's `acc[c]` has its
  own scale `s_G × s_w[c]`. Comparing raw accumulators across classes would compare numbers in different units.
  Each class is mapped onto one common logit scale `s_Z`, so the `Z[c]` are comparable. This is the ordinary §3.3
  multiplier with `s_in_layer = s_G` and `s_out = s_Z`, evaluated in float64 in the §3.3 order:
  `M[c] = (s_G * s_w[c]) / s_Z`. `quantize.py` then turns each `M[c]` into `(m0[c], n[c])` with §3.3, unchanged.
- **Tile columns:** only columns 0..4 are used. Columns 5..15 hold weight 0 and their outputs are ignored.

### 5.8 ReLU

ReLU is not a separate operation anywhere. It is the `lo = 0` bound of the requantization that follows each
conv layer (and the `gap` requantization, whose input is already ≥ 0). `clamp(v, 0, 127)` is exactly
`min(max(v, 0), 127)`, and since requantization is monotonic, this matches `quantize(ReLU(x))` in the float
model.

| Requantization after | lo | hi |
|---|---|---|
| `conv1`, `conv2`, `conv3` | 0 | 127 |
| `gap` | 0 | 127 |
| `dense` | −128 | 127 |

### 5.9 Argmax (`argmax`)

```
class = the smallest c in 0..4 such that Z[c] = max(Z[0..4])
```

- **Tie rule: the lowest index wins.** Ties are possible because `Z` is int8 (two logits can round or saturate
  to the same value). This is the same rule as `numpy.argmax`, so the reference needs no special handling.
- The output is an unsigned integer 0..4 (3 bits). The mapping from index to class name is the class order
  fixed in `manifest.csv` / `docs/dataset_audit.md`, and Keras training uses the same order.
- Hardware compares sequentially c = 0..4 and replaces the running maximum only on a **strictly greater** value.
  That implements the lowest-index rule.

### 5.10 Named intermediate tensors

`ref_int.py`'s `trace(x)` returns these tensors, with these names, and the RTL testbenches compare against them.
The `acc*` tensors are the int32 values before requantization. They are for debugging and are not stored by the
engine.

| Name | Shape | dtype | Range | Definition |
|---|---|---|---|---|
| `x` | 8000 × 1 | int8 | −128..127 | Quantized input (§2) |
| `acc1` | 993 × 16 | int32 | §6 | §5.3, `conv1` |
| `a1` | 993 × 16 | int8 | 0..127 | requant(acc1) |
| `p1` | 248 × 16 | int8 | 0..127 | maxpool(a1), `pool1` |
| `acc2` | 248 × 32 | int32 | §6 | §5.3, `conv2` |
| `a2` | 248 × 32 | int8 | 0..127 | requant(acc2) |
| `p2` | 62 × 32 | int8 | 0..127 | maxpool(a2), `pool2` |
| `acc3` | 62 × 32 | int32 | §6 | §5.3, `conv3` |
| `a3` | 62 × 32 | int8 | 0..127 | requant(acc3) |
| `gsum` | 32 | int32 | 0..7,874 | §5.6 step 1 |
| `g` | 32 | int8 | 0..127 | §5.6 step 2 |
| `acc_dense` | 5 | int32 | §6 | §5.7 |
| `z` | 5 | int8 | −128..127 | requant(acc_dense) |
| `cls` | 1 | int | 0..4 | §5.9 |

`model_int.npz` holds, for each of `conv1`, `conv2`, `conv3`, `gap` and `dense`: `m0` and `n` (one per output
channel). For `conv1`, `conv2`, `conv3` and `dense` it also holds `W` (shape `Cout × K`, flattened as in §5.3) and
`bias` (int32, `Cout`). `gap` has no weights or bias.

### 5.11 Points to confirm before the S1.2 review

Bias handling is already fixed by §2.5 (`acc = b_q[c] + sum(w_q * a_q)`), and §5.3 follows it.

1. **Padding of `conv2` and `conv3` is `same`.** The runbook says im2col needs padding for the second and third
   conv layers, and the MAC total (2.73 M) matches. `cnn1d.py` must use `padding='same'` for those two convs and
   `'valid'` for `conv1` and both pools. (Owner of T1.1.)
2. **Every conv is followed by ReLU, and the dense layer is not.** `cnn1d.py` must match.
3. **`gap` requantizes to int8 (§5.6)**, not "fold 1/L into the dense layer". The alternative is to let the dense
   layer take 14-bit inputs, which means changing §4 to a variable number of bit-cycles. Agree this with the team.
   It is recorded as D19 in `docs/decisions.md`.
4. **The dense output is requantized to int8, with lowest-index argmax.** The alternative is per-tensor weights
   for the dense layer only, so the raw `acc_dense` can go straight to argmax. That avoids saturation ties but adds
   a special case to `quantize.py`. This is recorded as D20 in `docs/decisions.md`.

## 6. Accumulator bound

§1.2 promises that `acc` never wraps and is never saturated. This section proves it for this network and writes the
numbers down. It also gives the minimum widths for the two registers inside the tile, `P` and `T` from §4.

The idea is simple: find the largest value a single product can have, multiply by the number of products, add the
bias, and check that the total fits in the register.

### 6.1 The largest single product

- A weight is at most 127 in size. `|w| ≤ 127`, because §2.2 never produces −128.
- An activation is at most 128 in size. `|x| ≤ 128`, because int8 reaches −128.

So one product is at most

```
|w × x| ≤ 127 × 128 = 16,256
```

The runbook writes the bound as `inputs × 128 × 128 + |bias|`, which assumes a weight could be −128. That form is
also valid, just slightly looser. Both are given in the table below.

### 6.2 The bound for each layer

A dot product with `K` inputs is at most `16,256 × K` in size:

| Layer | K | `16,256 × K` (this spec) | `16,384 × K` (runbook form) | Room left for the bias in int32 |
|---|---|---|---|---|
| `conv1` | 64 | 1,040,384 | 1,048,576 | 2,146,443,263 |
| `conv2` | 144 | 2,340,864 | 2,359,296 | 2,145,142,783 |
| `conv3` | 288 | **4,681,728** | **4,718,592** | **2,142,801,919** |
| `dense` | 32 | 520,192 | 524,288 | 2,146,963,455 |

"Room left for the bias" is `2^31 − 1 − 16,256 × K`, where `2^31 − 1 = 2,147,483,647` is the largest int32
value.

The largest layer is `conv3`, with K = 288. Its dot product is at most **4,681,728** in size. That is about 0.2% of
the int32 range, so the dot product alone can never come close to overflowing.

(`gap` has no dot product. Its sum is at most 7,874, see §5.6.)

### 6.3 Including the bias, and every partial sum

`acc` starts at the bias and then adds the chunk results one at a time (§5.3). It is not enough for the final value
to fit. Every value `acc` takes along the way has to fit too.

Every one of those partial values is at most

```
|bias| + (sum of |w × x| over the products added so far)  ≤  |bias| + 16,256 × K
```

This uses absolute values, so it holds whatever order the products are added in and whatever their signs are.
Signs can only cancel, never push the total past this bound.

So `acc` can never overflow as long as

```
|b_q[c]| ≤ 2^31 − 1 − 16,256 × K
```

for every output channel `c` of every layer, which is the last column of the table in §6.2. For `conv3` that means
`|b_q[c]| ≤ 2,142,801,919`.

**This is a rule for the exporter.** `quantize.py` checks it for every channel of every layer and stops the export
with an error if any bias breaks it. The export never clamps the bias to make it fit. In practice the quantized bias
`b / (s_in × s_w)` of a trained layer is many orders of magnitude smaller than this limit. If a bias ever gets
close, something is wrong with the scales, and the fix is in the model or the calibration, not in the accumulator
width.

With that check in place, every `acc` in the network stays inside int32, and the claim in §1.2 holds.

### 6.4 Widths inside the tile

The tile works on one chunk of 16 inputs at a time (§4), so its registers can be much narrower than 32 bits.

**Bit-plane sum `P`.** At most 16 weights are added. With exported weights (at most 127 in size):

```
|P| ≤ 16 × 127 = 2,032
```

A signed 12-bit value covers `−2,048..2,047`, so **`P` needs at least 12 bits, signed**.

**Tile accumulator `T`.** Two kinds of value appear in `T`:

- After cycles 0 to 6, `T` holds the dot product of the weights with the low 7 bits of each activation. Those 7
  bits are worth at most 127, so `|T| ≤ 16 × 127 × 127 = 258,064`.
- After cycle 7, `T` holds the full chunk dot product, so `|T| ≤ 16 × 127 × 128 = 260,096`.

With exported weights, every value `T` takes is at most 260,096 in size, which would fit in a signed 19-bit
register (`−262,144..262,143`).

The tile, however, is also tested with weights of −128 (§4.1). With all 16 weights and all 16 activations at −128,
the chunk result is `16 × (−128) × (−128) = 262,144`. That is one more than a signed 19-bit register can hold, so
it would wrap. To keep the tile correct for every int8 weight, **`T` uses at least 20 bits, signed**
(`−524,288..524,287`). The cost is one extra bit per column, and the tile then needs no exceptions.

For the same reason, `P` is checked with weights of −128 too. The most negative `P` is `16 × (−128) = −2,048`,
which still fits in 12 signed bits, so the 12-bit width for `P` holds for every int8 weight.

**Shifted value.** The largest shifted term is `P << 7`, at most `2,048 × 128 = 262,144` in size. It needs the same
20 bits as `T`, which is why §4.5 says to sign-extend `P` to the width of `T` before shifting.

The RTL may use wider registers than these. Extra width never changes the result, because nothing overflows either
way. It may not use narrower ones.

| Register | Where | Largest size | Minimum width |
|---|---|---|---|
| `P` | tile column, per bit-cycle | 2,048 | 12 bits signed |
| `T` | tile column, per chunk | 262,144 | 20 bits signed |
| `acc` | outside the tile, per output | `|bias| + 4,681,728` | 32 bits signed (int32) |
| `p`, `r` | requantization, §3 | below `2^47` | 48 bits signed |

### 6.5 Checks in the Python reference

The proof above is the guarantee. The checks below catch the case where the code doesn't follow the spec.

- `quantize.py` asserts the bias rule of §6.3 for every channel, and records the limit it checked against.
- `quantize.py` also computes the exact worst case for each output channel from the real weights, as runbook Step
  1.3 asks: `128 × Σ|w| + |bias|`, summed over that channel's weights. It asserts this is at most `2^31 − 1`. This
  bound is tighter than §6.3, because it uses the actual weights instead of assuming every weight is 127.
- `ref_int.py` asserts that every `acc` value it computes stays inside int32.
- `tile_model.py` asserts that every `P` fits in 12 signed bits and every `T` fits in 20 signed bits.

If the model changes (a new K, or a new layer), the table in §6.2 is recomputed and the new largest layer is written
down here before anything else is built.

## 7. Worked example

This example runs one small dot product through the tile by hand, one bit-cycle at a time, exactly as §4.4
describes. It is small enough to check on paper, and it is chosen to hit the cases that most often go wrong:

- an activation of **−128**, which has only the sign bit set;
- other **negative activations** (−3), whose bit patterns are full of 1s;
- **negative weights** (−2 and −1);
- a **negative final result**, so the requantization exercises the arithmetic shift and the ReLU clamp.

The review gate in runbook Step 1.2 asks two team members to redo this example independently from the spec, without
looking at each other's work. Both must get the same answer as the ordinary `sum(w × a)`.

### 7.1 Inputs

Four inputs (`K = 4`). On the real tile the other 12 rows would hold activation 0, which adds nothing.

| Row `r` | Weight `w_r` | Activation `a_r` | Bit pattern of `a_r` (`a_7 … a_0`) |
|---|---|---|---|
| 0 | 3 | −128 | `1000 0000` |
| 1 | −2 | 5 | `0000 0101` |
| 2 | 5 | −3 | `1111 1101` |
| 3 | −1 | 127 | `0111 1111` |

The bias is `b_q = 10`. The requantization values are `m0 = 20000`, `n = 20`. These are made up for the example and
do not come from a real layer.

### 7.2 The direct answer, for comparison

```
3 × (−128) + (−2) × 5 + 5 × (−3) + (−1) × 127
= −384 − 10 − 15 − 127
= −536
```

The bit-serial result must be exactly −536.

### 7.3 The 8 bit-cycles

On each cycle `b`, take bit `b` of each activation, add up the weights on the rows where that bit is 1 (that is
`P`), shift `P` into place and update `T`. Cycles 0 to 6 add. Cycle 7 subtracts.

| Cycle `b` | Bits of rows 0, 1, 2, 3 | Weights included | `P` | Update | `T` after the cycle |
|---|---|---|---|---|---|
| 0 | 0, 1, 1, 1 | −2 + 5 − 1 | 2 | `+ 2 × 1 = +2` | 2 |
| 1 | 0, 0, 0, 1 | −1 | −1 | `+ (−1) × 2 = −2` | 0 |
| 2 | 0, 1, 1, 1 | −2 + 5 − 1 | 2 | `+ 2 × 4 = +8` | 8 |
| 3 | 0, 0, 1, 1 | 5 − 1 | 4 | `+ 4 × 8 = +32` | 40 |
| 4 | 0, 0, 1, 1 | 5 − 1 | 4 | `+ 4 × 16 = +64` | 104 |
| 5 | 0, 0, 1, 1 | 5 − 1 | 4 | `+ 4 × 32 = +128` | 232 |
| 6 | 0, 0, 1, 1 | 5 − 1 | 4 | `+ 4 × 64 = +256` | 488 |
| 7 | 1, 0, 1, 0 | 3 + 5 | 8 | `− 8 × 128 = −1024` | **−536** |

After 8 cycles `T = −536`, which matches the direct answer in §7.2.

Things worth noticing:

- The −128 on row 0 does nothing for seven cycles, because only its sign bit is set. Its whole effect arrives on
  cycle 7, as part of the subtraction.
- If cycle 7 added instead of subtracting, the result would be `488 + 1024 = 1512`. That is completely wrong, which
  is why the tests in §4.6 and §8 check the subtraction directly.
- `T` goes positive and then negative. That is normal, and §6.4 already covers every value it can take.

### 7.4 Adding the bias

The bias is added once, after the dot product (§5.3):

```
acc = b_q + T = 10 + (−536) = −526
```

### 7.5 Requantization

Following §3.1 step by step, with `m0 = 20000` and `n = 20`:

```
p = acc × m0          = −526 × 20000            = −10,520,000
r = p + (1 << 19)     = −10,520,000 + 524,288   = −9,995,712
q = r >> 20           = floor(−9,995,712 / 1,048,576) = floor(−9.53…) = −10
```

The shift is an arithmetic shift, so it rounds toward minus infinity: −9.53 becomes −10, not −9. (A division that
truncates toward zero would give −9, which is wrong, see §3.4.)

Finally the clamp (§3.5):

- Without ReLU (`lo = −128`): `y = clamp(−10, −128, 127) = −10`.
- With ReLU (`lo = 0`): `y = clamp(−10, 0, 127) = 0`.

As a check in real numbers: `−526 × 20000 / 2^20 = −10.03…`, which rounds to −10. The integer pipeline agrees.

## 8. Verification ladder

The verification ladder is the order in which the design is checked, from pure software down to the real board.
Each rung is compared against the rung above it, and nothing moves down a rung until it passes. When something
breaks, this tells you exactly where: the first rung that fails is where the bug was introduced.

The diagram of this ladder is `docs/diagrams/verification_ladder.png`. (The runbook asks for the editable
`.drawio` source and an `.svg` export as well, under the same name.) The table below lists the same rungs and pass
criteria as the diagram. If the two ever disagree, fix them so they match.

You climb the ladder from the bottom. A rung counts as passed only when a reviewer who is not the author has ticked
every one of its boxes.

### 8.1 The rungs and their pass criteria

| # | Rung | What is compared | Pass criteria | Runbook step |
|---|---|---|---|---|
| 1 | Float model (TensorFlow / Keras) | The quantized W8A8 integer model against the float Keras model | Integer accuracy is within 1 percentage point of float (trained model). Batchnorm folding is verified numerically: folded equals unfolded. | 2.6 |
| 2 | Integer reference (`ref_int.py`) | `ref_int.py` against the quantized model, and against an independent `conv1` implementation | No floats in the forward pass (the dtype asserts pass), and the worst-case accumulator is below the int32 bound (§6). The independent `conv1` version matches on 100 random inputs, and one hand-computed output matches `trace()`. `ref_int.py` accuracy equals the quantized-model accuracy. | 1.3, 2.7 |
| 3 | Python tile model (`tile_model.py`) | The cycle-level 16 × 16 bit-serial tile model against numpy `W.T @ a` | 100% match on 100,000 random vectors. 100% match on the edge cases: all activations −128, all activations +127, all weights −128, all zeros. | 1.4 |
| 4 | RTL simulation (cocotb + Icarus, `make sim`) | `cim_tile.sv`, the `conv1` engine and the full engine against `tile_model.py`, `expected_l1.hex` and `ref_int.py` | `cim_tile`: 100% on 10,000 random vectors plus directed edge cases plus weight preload and reload, and a deliberately broken bit-7 subtract is caught. `conv1` engine: bit-exact on every position and channel of every input, and the weight-write counter stays 0. Full engine: bit-exact on all 200 clips at every layer, and the cycle count equals the Step 3.1 formula. | 1.6, 1.8, 3.5 |
| 5 | Post-synthesis (Vivado hierarchical reports) | The synthesized netlist (`make synth BOARD=nexys`, 100 MHz) against the reference | The post-synthesis functional simulation is bit-exact on 5 inputs. DSP = 0 inside the tile array, with the requantization DSPs reported separately along with the whole-design total. WNS ≥ 0 at 100 MHz, or the achieved frequency and critical path are recorded. | 1.9 |
| 6 | Post-implementation (post-route, Vivado) | The placed-and-routed design, with timing, against the reference | The post-route timing simulation is bit-exact on 20 clips, for at least the largest CIM and baseline configurations. WNS ≥ 0 at the target clock, or the achieved frequency is recorded. | 4.1 |
| 7 | Board (Nexys A7-100T, then ZCU104) | The bitstream running on the board, through the host link and host script, against `ref_int.py` | `conv1`: every position and channel matches on every input, and the board cycle count equals the simulation cycle count. Full test set: 100% logit match with `ref_int.py`, for both the CIM and the baseline engine. The ZCU104 cycle count equals the Nexys cycle count exactly, because cycles come from the design, not the board. | 1.10, 1.11, 4.2 |

Rung 1 is the only place where an accuracy tolerance is allowed, because that is where quantization error enters
on purpose. Everything from rung 2 down must be **bit-exact**: the same integers, not "close". Hardware is correct
when it produces exactly the same integers as `ref_int.py`. A result that is 99% right nearly always means a
rounding, sign or off-by-one error, not noise.

The optional ASIC synthesis in Stage 5 (Step 5.3) hangs off rung 5 as a side rung. Its criterion is that a
gate-level simulation of the Genus netlist passes the tile testbench. It uses the same arithmetic as everything
else, so it is checked against the same references.

### 8.2 Edge cases every rung must include

Random vectors find most bugs, but not the ones at the edges. Every rung from 3 down runs these directed cases
as well:

- **Quantization and rounding:** inputs that land on `−128`, `−127`, `−1`, `0`, `1`, `126`, `127`, and just outside
  the range. Rounding ties at `0.5`, `−0.5`, `1.5`, `−1.5`, `2.5`, `−2.5` (§2.3).
- **Tile:** all activations −128; all activations +127; all weights −128; all zeros (runbook Step 1.4). A single
  −128 activation with a positive weight, to check the bit-7 subtract. Alternating signs. Maximum-size inputs.
- **Bias:** bias added exactly once, including biases near the §6.3 limit.
- **Requantization:** positive and negative accumulators, exact ties (`−1.5 → −1`, `−2.5 → −2`), negative values
  that need the arithmetic shift, saturation above 127 and below −128, and ReLU on negative, zero and positive
  values (§3).
- **Multiplier export:** an `m0` that rounds up to 32768 and must be renormalized, and values of `n` at both ends of
  the allowed range (§3.3).
- **Layers:** the first and last window of every conv layer, every padded position of `conv2` and `conv3`, the
  dropped position 992 of `conv1`, and argmax ties (§5).
- **Accumulator:** the largest real K (`conv3`, 288 inputs) with worst-case values (§6).

### 8.3 How to compare, and what to do with a mismatch

Compare the intermediate values in this order, and report the **first** one that differs:

1. the quantized inputs, weights and biases;
2. the bit-plane sums `P` on each cycle, then the chunk result `T`;
3. `acc` before requantization;
4. the requantization steps `p`, `r` and `q`;
5. the clamped int8 output;
6. the full layer output tensor, then the logits and the class.

Never accept a mismatch because the final output happens to agree. Saturation at 127 can make a wrong `acc` produce
the right int8 value, and a wrong logit can still pick the right class. The bug is still there and will show up on
another input.

When a rung fails, the spec is the referee. Check both sides against this document. If the spec itself turns out to
be ambiguous, fix the spec first, then the implementation (§9).

## 9. Changing this spec

This file is the contract that Python, C and RTL are all written against, so changes to it are handled carefully:

- **Any change that can alter a result is a real change.** That covers number formats, rounding, shifts, clamps,
  layer definitions, the bit-serial schedule and the bounds. It needs a decision entry in `docs/decisions.md`, an
  update to every implementation, and new test vectors.
- **Wording-only changes** (clearer text, typos, examples) don't need a decision, but must not change any number or
  formula.
- **If the model changes** (a new layer, a different K or a different shape), update §5.1 and the table in §6.2
  first, then everything else.
- **If two people read the spec differently,** the spec is wrong, not either reader. Fix the text so that only one
  reading is possible, as the runbook's review gate for Step 1.2 asks.
