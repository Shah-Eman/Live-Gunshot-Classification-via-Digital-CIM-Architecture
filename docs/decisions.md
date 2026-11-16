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

## D22. Window placement around the detector trigger

| Field | Value |
|---|---|
| Step | S1.6, runbook Step 1.6 item 2; front-end spec in `docs/frontend.md` |
| Needed by | Before any training window is cut (Step 1.7 onwards), and before the Step 4.7 window capture |
| Status | **Proposed**, waiting for team sign-off |
| Decided by | (to be filled in) |
| Date | (to be filled in) |

**The question.** C3GD clips start about 0.1 s before the shot, and the `c3gd_nj_farm` and `c3gd_ny_gravel_pit` clips
are only 0.459 s long. A 0.5 s window starting 50 ms before the trigger does not fit them. Padding only some clips
would make the padding a session label. The runbook lists (a) a 0.5 s input with 0.40 s of real audio and the last
0.10 s zeroed for every clip, (b) dropping the two short sessions, (c) a 0.4 s input, which changes `arith_spec.md`.

**Proposed choice: option (a).** The model input stays 8000 samples (`arith_spec.md` §5.1 unchanged). It holds
6400 samples of real audio, from 800 samples (0.05 s) before the trigger to 5600 samples (0.35 s) after it, and 1600
zeros. A clip that cannot supply this is dropped, never padded. All numbers live in one object,
`sw/frontend/config.py: WindowConfig`.

**Every gun window comes from the detector** (`sw/frontend/detector.py`, integer only, spec and register widths in
`docs/frontend.md`). The dataset's shot times are used only to verify a trigger: edge labels from
`gunshot_location_in_seconds` (copied to `sw/data/meta/`), the C3GD pre-roll (0.1 s; measured 0.046 s for
`c3gd_nj_farm`), and the Cadre onset (first sample at or above half the peak within 0.1 s before it). A trigger
matches a reference shot when it lies from 0.10 s before to 0.05 s after it, one to one, nearest pairs first. The
early side is wider because supersonic rounds put their crack ahead of the muzzle blast. A window with no matched
reference shot is never kept, and the refractory period is 0.10 s (≤ 0.35 s).

**Detector recall on every gun clip of the dataset** (`results/step16/detector_summary.md`, from
`sw/frontend/cut_windows.py`, which exits non-zero if any source, class or session falls below 95 %):

| Group | Known shots | Matched | Recall | Inside another window | Truly missed |
|---|---:|---:|---:|---:|---:|
| c3gd | 4,233 | 4,163 | 98.3 % | 0 | 70 |
| cadre | 1,712 | 1,705 | 99.6 % | 7 | 0 |
| edge | 5,965 | 5,891 | 98.8 % | 51 | 23 |
| 9 mm | 4,886 | 4,826 | 98.8 % | 18 | 42 |
| .38 Special | 1,435 | 1,425 | 99.3 % | 4 | 6 |
| 5.56 / .223 | 4,372 | 4,304 | 98.4 % | 32 | 36 |
| 12 ga | 1,217 | 1,204 | 98.9 % | 4 | 9 |
| c3gd_nj_farm | 335 | 333 | 99.4 % | 0 | 2 |
| c3gd_nj_farm_2 | 1,950 | 1,919 | 98.4 % | 0 | 31 |
| c3gd_ny_gravel_pit | 345 | 345 | 100.0 % | 0 | 0 |
| c3gd_oh_farm | 1,603 | 1,566 | 97.7 % | 0 | 37 |
| cadre | 1,712 | 1,705 | 99.6 % | 7 | 0 |
| edge-day1 | 4,575 | 4,510 | 98.6 % | 45 | 20 |
| edge-day2 | 1,390 | 1,381 | 99.4 % | 6 | 3 |
| **All** | **11,910** | **11,759** | **98.73 %** | **58** | **93** |

"Inside another window" means the shot fell inside the refractory period of an earlier trigger, so it is in that
window's audio but has no window of its own. 6 shots in the first 543 samples of a clip (detector warm-up) are not
counted. Of 26,245 triggers, 14,486 matched no reference shot (echoes, other shooters, background) and none of them
became a window; 13,744 more were preempted by a louder event within 0.10 s.

**Exceptions.** Every source, class and session passes 95 %. Two finer cells (source × class × session) do not, and
in both the reference time is wrong rather than the detector:

- c3gd / .38 / `c3gd_nj_farm_2`: 51 of 57 (89.5 %). All 6 misses are microphone `acorn_DOM2`, whose clips are not
  aligned to the 0.1 s pre-roll: their loudest event sits between 0.00 and 0.82 s, and in 4 of them the detector
  triggers on it.
- c3gd / 12 ga / `c3gd_oh_farm`: 91 of 98 (92.9 %). In all 7 misses the peak is within the first 12 ms of the clip:
  the recording starts after the shot began, so there is no onset to detect.

**Drops by placement** (11,759 matched shots; 11,186 windows kept):

| Reason | Windows | Where |
|---|---:|---|
| window would start before the clip's audio | 383 | 333 are all of `c3gd_nj_farm` (shot at 0.046 s) |
| less than 0.40 s of audio after the start | 190 | 150 are `c3gd_ny_gravel_pit` 5.56 (150 of 171) |

"Audio" is the span between the first and the last non-zero sample, so file padding is never used as signal.
Consequence: CV fold 2 (validation session `c3gd_nj_farm`) has no gun window, and `ny_gravel_pit` keeps only 21
5.56 windows. Measured alternatives for the pre-trigger length, each with 0.40 s of real audio:

| Pre-trigger | Windows kept | Start before audio | Short after start |
|---|---:|---:|---:|
| 0.03 s | 11,476 | 1 | 282 |
| 0.04 s | 11,497 | 40 | 222 |
| 0.05 s (proposed) | 11,186 | 383 | 190 |

0.04 s would keep 320 of the 333 `c3gd_nj_farm` windows (and fill fold 2) for 10 ms less context before the trigger. It is a one-line change
in `WindowConfig`; the team should choose before Step 1.7.

**Bandwidth.** Every clip is resampled to 16 kHz, so content up to 8 kHz is kept. The 55 background clips recorded
below 16 kHz (`dataset_audit.md` §4.3) have nothing above their own Nyquist: their median energy share above 5.5 kHz
is −53 dB against −27 dB for the other UrbanSound8K clips (`results/step16/bandwidth_by_source.md`), and C3GD gunshots reach 7.4 kHz (median roll-off). Low-passing
every clip to the poorest band would remove that, so the 55 clips are **dropped** (`splits.json` lists them as
excluded). `check_windows.py` fails if any source's mean spectrum is more than 50 dB below its peak below 7.5 kHz;
the 55 dropped clips fail it (band edge 6,031 Hz), every kept source passes (8,000 Hz).

**Normalisation.** Identical for every source and done the same way in hardware: a per-window power-of-two gain,
`k = clamp(15 − bit_length(peak), 0, 10)` over the real part, then `w << k` with saturation (`docs/frontend.md` §4).
Measured: no window at the maximum shift, median peak after gain within 2.56 dB across sources. The int16-to-int8 step
is a proposal for `arith_spec.md` §2.4 (`docs/frontend.md` §4).

**Reason.**
- (a) keeps the input size and `arith_spec.md` as they are, and every window of every source has the same layout, so
  neither padding nor length identifies a session.
- (b) would lose 9 mm's `nj_farm` session anyway and also all of 12 ga's `ny_gravel_pit`; (a) keeps the latter.
- (c) needs a spec amendment and a fresh Step 1.2 review; the Step 1.7 sweep can still show whether 0.4 s is enough.

**What changes if the team picks something else.** Edit `WindowConfig`, rerun `make windows scan check_splits
check_windows`. For (c), also amend `arith_spec.md` §5.1.

---

## D23. Final test set

| Field | Value |
|---|---|
| Step | S1.6, runbook Step 1.6 item 6 |
| Needed by | Before any training; the test set is used once, in Step 1.11 |
| Status | **Proposed**, waiting for team sign-off |
| Decided by | (to be filled in) |
| Date | (to be filled in) |

**The question.** Which recordings are held out, so that every firearm class is tested on sessions it was never
trained on, from two different datasets?

**Proposed choice (the runbook's recommendation).** Every clip of every class, background included, whose
`session_id` is `c3gd_nj_farm_2` or `edge-day2`, plus 15 % of the remaining background sessions (452 of 3,010,
drawn by `session_id` with seed 1623). No session with gun clips is drawn at random, so no session is split. The
clips are stored as explicit paths in `sw/data/splits.json`
(sha256 `e5037db37ec301cac37bf801cf7b281a6020ae287ffd8b0947643e9dd657c399`, written by `sw/data/make_splits.py`,
checked by `sw/data/check_splits.py`). Loading it fails if the file is missing or the manifest has changed.

| Set | 9 mm | .38 Special | 5.56 / .223 | 12 ga | Background | Sessions |
|---|---:|---:|---:|---:|---:|---:|
| Test clips | 1,143 | 251 | 937 | 216 | 2,564 | 454 |
| Test gun windows | 1,127 | 290 | 1,385 | 467 | (sampled) | |

The gun clip counts equal the runbook's expected 1,143 / 937 / 251 / 216. Every class is tested on `c3gd` and
`edge`.

**The rest (17,335 clips; 55 dropped by D22).** `StratifiedGroupKFold(5, shuffle, seed 16)` grouped by `session_id`;
`check_splits.py` also proves that no `group` (one shot recorded by several microphones) crosses a fold or the test
set. Each fold validates on exactly one gun session: fold 0 `c3gd_oh_farm`, fold 1 `edge-day1`, fold 2 `c3gd_nj_farm`
(no gun windows under D22 with 0.05 s pre-trigger), fold 3 `cadre`, fold 4 `c3gd_ny_gravel_pit`. .38 is validated only
in folds 1 and 3 and is reported separately. Also in `splits.json`: a device-disjoint split inside the CV pool (seed
17, 7 held-out devices, every class tested) and a random clip-level split (seed 18, 80/20).

**Reason.** Both held-out sessions record all four classes, from two datasets and microphone types, so test accuracy
measures generalisation to a new day and site rather than session recognition. Holding out background by session keeps
each quiet stretch on the same side as the shots it was cut from.

---

## Confirmations needed from the model owner (T1.1)

These are not choices. They are facts about `sw/models/cnn1d.py` that `docs/arith_spec.md` §5 assumes. They have to be
checked against the file once it exists:

1. `conv2` and `conv3` use `padding='same'`. `conv1` and both max-pool layers use `'valid'`.
2. Every conv layer is followed by ReLU, and the dense layer is not.
3. The per-layer table printed by `cnn1d.py` reproduces §5.1 exactly: 15,008 weights, 85 biases and 2,731,168 MACs.

If any of these fails, either the model or the spec is wrong, and it is fixed before anything else is built.
