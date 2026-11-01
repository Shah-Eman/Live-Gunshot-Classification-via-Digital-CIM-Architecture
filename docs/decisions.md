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

---

## D22. Window placement

| Field | Value |
|---|---|
| Step | S1.6 (runbook v4, item 2); code: `sw/frontend/windows.py` |
| Needed by | Before any window is cut for training |
| Status | **Proposed**, waiting for team sign-off |
| Decided by | (to be filled in) |
| Date | (to be filled in) |

**The question.** Where does the 0.5 s input window sit around the trigger, so that every source and the live front
end produce exactly the same kind of window, and no clip needs padding that only its session would have?

**Choice: runbook option (a), with the split re-measured.** The input stays 8,000 samples (0.5 s at 16 kHz), so
`arith_spec.md` §5.1 does not change. Every window, from every source and in the live front end:

| Samples | Content |
|---|---|
| 0 – 639 | real audio, 40 ms before the trigger (`PRE = 640`) |
| 640 – 5,599 | real audio after the trigger; real part `REAL = 5,600` samples = 0.35 s |
| 5,600 – 7,999 | zeros (`TAIL = 2,400` samples = 0.15 s) |

A clip with fewer than 640 samples before its trigger, or fewer than 5,600 from the window start to its end, is
**dropped, never padded**. Level: one rule for every window: an integer power-of-two gain (shift left by the largest
k ≤ 6 that keeps the peak in range), done in hardware with a peak register and a shifter after the capture.

**Why not the runbook's 50 ms / 0.40 s.** The runbook assumed every C3GD clip starts 0.1 s before the shot. Measured
on the 0.459 s clips available locally (134 clips, 2026-10-31), that is not true:

| Session | Shot position (detector trigger) | 50 ms / 0.40 s keeps | 40 ms / 0.35 s keeps |
|---|---|---|---|
| `c3gd_nj_farm` (0.459 s) | 45.8 – 47.3 ms | 0 of 66 | 62 of 66 |
| `c3gd_ny_gravel_pit` (0.459 s) | 40 – 141 ms | 35 of 68 | 64 of 68 |
| `c3gd_oh_farm`, `c3gd_nj_farm_2` (1.0 s) | about 100 ms | all | all |

With 50 ms, `c3gd_nj_farm` (all 335 of its 9 mm clips, one whole session) would be lost. 40 ms keeps it. 0.35 s of
real audio keeps nearly all of `ny_gravel_pit` (12 ga has only 5 sessions); 0.40 s would lose about half of it.
A gunshot and its first reflections fit easily in 0.31 s after the trigger.

**Re-measure on the full dataset** with `sw/frontend/cut_windows.py` and replace the counts above with the numbers
in `detector_summary.csv` (these are from a local sample, not results). If the team prefers 0.40 s of real audio, change
`REAL` in `windows.py` and accept the loss; the ADC sweep (Step 1.7) compares 0.25 / 0.4 / 0.5 s anyway.

**What changes if the team picks another option.** (b) dropping the two short sessions: set nothing in the code, filter
`c3gd_nj_farm` and `c3gd_ny_gravel_pit` out of the manifest before splitting. (c) a 0.4 s input: `INPUT = 6400`, a
spec amendment of §5.1 (`conv1` gives 793 positions) and a new Step 1.2 review.

### The detector (runbook item 1), recorded here because its constants are now RTL constants

`sw/frontend/detector.py`, integer only, every multiply a shift. Running |x| sum over 64 samples (4 ms); noise floor
tracked every frame with a 512 ms time constant; trigger when the sum is ≥ 4× the floor and above −52 dBFS; 0.5 s
refractory; inside refractory a re-trigger only for an event ≥ 2× the peak of the one that triggered (measured over
20 ms). `detect()` and the per-sample `detect_reference_loop()` (the RTL's form) agree bit for bit
(`test_detector.py`). Design choices found on the local sample:

- Threshold ×4 instead of ×8: ×8 missed edge shots recorded over loud range noise (other shooters, wind).
- Floor keeps tracking in refractory: a frozen floor let reverberant tails retrigger every 0.5 s.
- Floor time constant 512 ms, not 128 ms: on the full dataset (first run, 2026-11-01) Cadre phone clips with
  handling noise before the shot raised a 128 ms floor above the shot; 512 ms recovered 12 of 30 sampled
  misses and lost none of the clips already found.
- C3GD clips whose loud event is not near the documented 0.1 s pre-roll (some at 0.5–0.9 s, some in the
  first 30 ms) count as misses by the reference rule even though the detector triggers on the event; they
  give no window and are part of the recorded miss rate.
- Re-trigger on a 2× louder event: a weak sound just before the shot (handling noise, the first 20 ms of a C3GD
  clip) otherwise used up the refractory period and the shot was missed (Cadre recall rose from 0.81 to 1.00 on
  the sample).

Gun windows are cut at detector triggers only. The dataset's times decide only which triggers count as the shot: C3GD
one shot 30–150 ms in, Cadre one shot at the clip's loudest sample or loudest 4 ms (trigger ≤ 100 ms
before it), edge the labelled
`gunshot_location_in_seconds` (±50 ms). One window per known shot (the last trigger matched to it); triggers that
match no shot are not labelled as gunshots. Without the Kabealo metadata file, edge triggers are kept unverified.

---

## D23. Final test set

| Field | Value |
|---|---|
| Step | S1.6 (runbook v4, items 6–9); code: `sw/data/splits.py`, `sw/data/check_splits.py` |
| Needed by | Before any training |
| Status | **Proposed**, waiting for team sign-off; `splits.json` sha256 to be filled in when it is generated |
| Decided by | (to be filled in) |
| Date | (to be filled in) |
| splits.json sha256 | (to be filled in) |

**Choice: the runbook's recommendation.** Every row (background included) whose `session_id` is `c3gd_nj_farm_2` or
`edge-day2`, plus 15% of the remaining background-only sessions, drawn with seed 0. Quiet stretches carry their gun
clip's `session_id` and `group`, so they go with their shots. Every firearm class is then tested on two datasets
(C3GD and edge). The test set is used once, in Step 1.11.

**Leakage unit.** `splits.py` does not assume groups are nested in sessions: it joins every `session_id` and `group`
that occur together into one unit and splits units, so a shot recorded on several microphones, a quiet stretch and its
shot, several clips of one Freesound source or one MAD video can never be split.

**The other splits, in the same file.**
- Cross-validation: `StratifiedGroupKFold(5)` on the rest, grouped by unit. With the D23 test set there are five gun
  sessions left (`edge-day1`, `cadre`, `c3gd_oh_farm`, `c3gd_nj_farm`, `c3gd_ny_gravel_pit`), so each fold
  validates on about one gun session. .38 has only `cadre` and `edge-day1`: report its fold results separately.
- Device-disjoint: devices of the development gun clips held out until every class has ≥ 20% of its clips on
  held-out devices. The same shot can be on a training and a held-out device (by design, it tests the microphone);
  the count is stored in `splits.json`.
- Random clip-level 80/20, only to show how much session leakage inflates accuracy. Never for model selection.

**Proof.** `check_splits.py` must print ALL CHECKS PASS: no `session_id` or `group` in two folds or in both the
folds and the test set, every row of the test sessions in the test set, every firearm class tested on two datasets.

**What changes if the team picks other test sessions.** `splits.py --test-sessions ...`, then re-run
`check_splits.py`; check 6 says whether every class is still tested on two datasets.
