# Arithmetic Spec

Status: sections 1-3 written (T1.2.a), section 5 written (T1.2.c). Sections 4, 6, 7 and 8 are placeholders.

Python (`sw/`), C (`hls/`) and RTL (`rtl/`) implement every operation in this file
bit-exactly. Where this file and an implementation disagree, the implementation is
wrong.

## 1. Number formats

### 1.1 Notation

- All integers are two's complement.
- `intN` is a signed N-bit integer with range `[-2^(N-1), 2^(N-1) - 1]`.
- `/` on real numbers is exact real division. Integer code in this spec never uses
  `/`, `//` or `%`.
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

Placeholder (T1.2.b).

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

**Chunks.** The tile has 16 rows, so the K inputs are split into `Q = K / 16` chunks. Chunk `q` (0..Q−1) holds flat
indices `j = 16q .. 16q + 15`, in that order, on tile rows 0..15. Every K in this network is a multiple of 16. If a
later model has a K that isn't, the last chunk is padded with rows whose weight is 0 and whose activation is 0.

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

`m0_gap` and `n_gap` are chosen by `quantize.py` so that `m0_gap / 2^n_gap ≈ s_3 / (62 × s_G)`, where `s_3` is the
output scale of `conv3` and `s_G` is the GAP output scale (calibrated per tensor like every activation). The 1/L
is folded into this multiplier, so no integer division happens in hardware. Because input and output are both
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
  own scale `s_G × s_W[c]`. Comparing raw accumulators across classes would compare numbers in different units.
  `quantize.py` chooses each `(m0[c], n[c])` to map class `c` onto one common logit scale `s_Z`
  (`m0[c] / 2^n[c] ≈ s_G × s_W[c] / s_Z`), so the `Z[c]` are comparable.
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

Placeholder (T1.2.b). The largest dot product is `conv3`, with K = 288 (§5.1).

## 7. Worked example

Placeholder (T1.2.d).

## 8. Verification ladder

Placeholder (T1.2.e), diagram in `docs/diagrams/verification_ladder.{drawio,svg}`.
