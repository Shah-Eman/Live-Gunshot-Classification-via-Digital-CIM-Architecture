# Decisions

Every decision with its reason, so nobody has to argue it again in Week 18.

The master list is the Decisions sheet of the project tracker (D1 to D18). This file holds the full write-up of a
decision once it touches the design, with the options that were considered and why one was chosen. Each entry has a
status:

- **Proposed**: written down, waiting for the team to agree.
- **Accepted**: the team agreed. Say who decided and on what date.
- **Superseded**: replaced by a later decision. Link to it.

Nothing here is deleted. If a decision changes, add a new entry and mark the old one as superseded.

---

## D19. How global average pooling feeds the dense layer

| Field | Value |
|---|---|
| Step | S1.2 (task T1.2.c), `docs/arith_spec.md` §5.6 |
| Needed by | Before the S1.2 review gate, and before `quantize.py` (T1.3.a) is written |
| Status | **Proposed**, waiting for team sign-off |
| Decided by | (to be filled in) |
| Date | (to be filled in) |

**The question.** Global average pooling (`gap`) averages 62 positions, which is a sum divided by 62. Hardware should
not divide. So where does the 1/62 go?

**Option A: fold 1/L into the dense layer's requantization (what the runbook says).**
The `gap` step is only the sum `S[c]`, which can be as large as 62 × 127 = 7,874. That needs 14 bits. The dense layer
would then receive 14-bit inputs. The tile of §4 takes 8-bit inputs and spends exactly 8 bit-cycles on them, so it
cannot take a 14-bit input as it stands. Supporting it means changing §4 to a variable number of bit-cycles, which
touches the tile, its model, its testbenches and the cycle count.

**Option B: requantize the sum to int8 right after summing (chosen in the spec).**
`G[c] = requant(S[c], c, 0, 127)`, with the 1/62 folded into this multiplier (`M_gap = s_3 / (62.0 * s_G)`, evaluated in float64 as written in `arith_spec.md` §5.6).
Every input to every tile in the network is then int8, and §4 stays exactly as written. The cost is one extra
rounding step.

**Proposed choice: Option B.**

**Reason.**
- It keeps the tile, the bit-serial schedule and the cycle model unchanged.
- It reuses the §3 requantization unit as it is. No new arithmetic is introduced.
- The extra rounding error is not hidden. It is part of the integer model, so Step 2.6 measures its accuracy cost
  together with every other quantization effect.

**What changes if the team picks Option A instead.** §4 needs a variable bit-cycle count, §5.6 and §5.7 are
rewritten, `gap_sum` feeds the dense layer directly, and the tile model and cycle counts have to be revisited.

---

## D20. How the dense output is compared in argmax

| Field | Value |
|---|---|
| Step | S1.2 (task T1.2.c), `docs/arith_spec.md` §5.7 and §5.9 |
| Needed by | Before `quantize.py` (T1.3.a) is written |
| Status | **Proposed**, waiting for team sign-off |
| Decided by | (to be filled in) |
| Date | (to be filled in) |

**The question.** The dense layer uses one weight scale per output channel, so each class's raw accumulator
`acc_dense[c]` is in its own units. Comparing them directly would be like comparing 300 cm with 5 m. How do we get
numbers that argmax can compare?

**Option A: requantize each class onto one common logit scale (chosen in the spec).**
`Z[c] = requant(acc_dense[c], c, −128, 127)`, where each class's `(m0[c], n[c])` maps it onto a shared scale `s_Z`.
Argmax then runs on the int8 values `Z`. The downside is that `Z` is int8, so two classes can round or saturate to the
same value. The tie rule is then needed: the lowest index wins, the same as `numpy.argmax`.

**Option B: per-tensor weights for the dense layer only.**
All classes share one weight scale, so the raw `acc_dense` values are already comparable and can go straight into
argmax. That avoids saturation ties. The cost is a special case in `quantize.py`, because the dense layer would
quantize differently from every other layer.

**Proposed choice: Option A.**

**Reason.**
- Every layer is quantized the same way (per-channel weights), so `quantize.py` has no special case.
- The requantization unit is reused for the last layer too.
- The tie risk is real but small. `s_Z` should be chosen so that clipping at 127 is rare. Check this in Step 2.6, and
  switch to Option B if ties turn out to matter.

---

## Confirmations needed from the model owner (T1.1)

These are not choices. They are facts about `sw/models/cnn1d.py` that `docs/arith_spec.md` §5 assumes. They have to be
checked against the file once it exists:

1. `conv2` and `conv3` use `padding='same'`. `conv1` and both max-pool layers use `'valid'`.
2. Every conv layer is followed by ReLU, and the dense layer is not.
3. The per-layer table printed by `cnn1d.py` reproduces §5.1 exactly: 15,008 weights, 85 biases and 2,731,168 MACs.

If any of these fails, either the model or the spec is wrong, and it is fixed before anything else is built.
