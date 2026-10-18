# Arithmetic Spec

Status: sections 1-3 written (T1.2.a). Sections 4-8 are placeholders.

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

## Later sections

- 4. Bit-serial dot product. Placeholder (T1.2.b).
- 5. Layer types. Placeholder (T1.2.c).
- 6. Accumulator bound. Placeholder (T1.2.b).
- 7. Worked example. Placeholder (T1.2.d).
- 8. Verification ladder. Placeholder (T1.2.e), diagram in `docs/diagrams/verification_ladder.{drawio,svg}`.
