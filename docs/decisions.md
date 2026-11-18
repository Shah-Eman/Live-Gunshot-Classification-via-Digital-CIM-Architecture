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
are only 0.459 s long. A 0.5 s window starting 40–50 ms before the trigger does not fit them. Padding only some clips
would make the padding a session label. The runbook lists (a) a 0.5 s input with 0.40 s of real audio and the last
0.10 s zeroed for every clip, (b) dropping the two short sessions, (c) a 0.4 s input, which changes `arith_spec.md`.

**Proposed choice: option (a).** The model input stays 8000 samples (`arith_spec.md` §5.1 unchanged). It holds
6400 samples of real audio, from 640 samples (0.04 s) before the trigger to 5760 samples (0.36 s) after it, and 1600
zeros. The runbook suggested 0.05 s before the trigger; 0.04 s is used because every `c3gd_nj_farm` clip has its shot
at 0.046 s, so 0.05 s dropped that whole session and left CV fold 2 without gun windows (measured below).
A clip that cannot supply this is dropped, never padded. All numbers live in one object,
`sw/frontend/config.py: WindowConfig`.

**Every gun window comes from the detector** (`sw/frontend/detector.py`, integer only, spec and register widths in
`docs/frontend.md`). The dataset's shot times are used only to verify a trigger: edge labels from
`gunshot_location_in_seconds` (copied to `sw/data/meta/`), the C3GD pre-roll (0.1 s; measured 0.046 s for
`c3gd_nj_farm`), and the Cadre onset (first sample at or above half the peak within 0.1 s before it). A trigger
matches a reference shot when it lies from 0.10 s before to 0.05 s after it, one to one, nearest pairs first. The
early side is wider because supersonic rounds put their crack ahead of the muzzle blast. A window with no matched
reference shot is never kept, and the refractory period is 0.10 s (≤ 0.36 s).

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

**Drops by placement** (11,759 matched shots; 11,497 windows kept):

| Reason | Windows | Where |
|---|---:|---|
| window would start before the clip's audio | 40 | 13 `c3gd_nj_farm` 9 mm, 16 `c3gd_nj_farm_2`, 5 `c3gd_oh_farm`, 6 others |
| less than 0.40 s of audio after the start | 222 | 162 `c3gd_ny_gravel_pit` 5.56, 16 `ny_gravel_pit` 12 ga, 40 `edge-day1` |

"Audio" is the span between the first and the last non-zero sample, so file padding is never used as signal.
`c3gd_ny_gravel_pit` keeps only 9 of its 171 5.56 clips: those clips are 0.459 s long with the shot at 0.1 s, so
0.40 s of audio from 0.04 s before the trigger fits only when the trigger comes early. Only option (c) would keep
them. Measured alternatives for the pre-trigger length, each with 0.40 s of real audio:

| Pre-trigger | Windows kept | Start before audio | Short after start |
|---|---:|---:|---:|
| 0.03 s | 11,476 | 1 | 282 |
| 0.04 s (proposed) | 11,497 | 40 | 222 |
| 0.05 s (runbook) | 11,186 | 383 | 190 |

0.04 s keeps 320 of the 333 `c3gd_nj_farm` windows (so CV fold 2 has gun windows) for 10 ms less context before
the trigger, and loses 32 more `ny_gravel_pit` windows than 0.05 s. 0.03 s would also keep the 40 but lose 60 more
short clips and leave only 32 ms before the trigger, under the detector's 34 ms warm-up.

**Bandwidth.** Every clip is resampled to 16 kHz, so content up to 8 kHz is kept. The 55 background clips recorded
below 16 kHz (`dataset_audit.md` §4.3) have nothing above their own Nyquist: their median energy share above 5.5 kHz
is −53 dB against −27 dB for the other UrbanSound8K clips (`results/step16/bandwidth_by_source.md`), and C3GD gunshots reach 7.4 kHz (median roll-off). Low-passing
every clip to the poorest band would remove that, so the 55 clips are **dropped** (`splits.json` lists them as
excluded). `check_windows.py` fails if any source's mean spectrum is more than 50 dB below its peak below 7.5 kHz;
the 55 dropped clips fail it (band edge 6,031 Hz), every kept source passes (8,000 Hz).

**Normalisation.** Identical for every source and done the same way in hardware: a per-window power-of-two gain,
`k = clamp(15 − bit_length(peak), 0, 10)` over the real part, then `w << k` with saturation (`docs/frontend.md` §4).
Measured: no window at the maximum shift, median peak after gain within 2.65 dB across sources. The int16-to-int8 step
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
(sha256 `27a9836334dccecb62629f7df729cb5e8cc735c9f4c93e8605899e1b1181ebd2`, written by `sw/data/make_splits.py`,
checked by `sw/data/check_splits.py`). Loading it fails if the file is missing or the manifest has changed.

| Set | 9 mm | .38 Special | 5.56 / .223 | 12 ga | Background | Sessions |
|---|---:|---:|---:|---:|---:|---:|
| Test clips | 1,143 | 251 | 937 | 216 | 2,564 | 454 |
| Test gun windows | 1,134 | 293 | 1,389 | 467 | (sampled) | |

The gun clip counts equal the runbook's expected 1,143 / 937 / 251 / 216. Every class is tested on `c3gd` and
`edge`.

**The rest (17,335 clips; 55 dropped by D22).** `StratifiedGroupKFold(5, shuffle, seed 16)` grouped by `session_id`;
`check_splits.py` also proves that no `group` (one shot recorded by several microphones) crosses a fold or the test
set. Each fold validates on exactly one gun session: fold 0 `c3gd_oh_farm`, fold 1 `edge-day1`, fold 2 `c3gd_nj_farm`
(320 gun windows, 9 mm only), fold 3 `cadre`, fold 4 `c3gd_ny_gravel_pit` (9 5.56 and 158 12 ga windows). .38 is
validated only in folds 1 and 3 and is reported separately. Also in `splits.json`: a device-disjoint split inside the
CV pool (seed 17, 7 held-out devices, every class tested) and a random clip-level split (seed 18, 80/20). The device
split is also group-disjoint: the 1,090 training clips (all Cadre) whose recording run was also captured on a
held-out device are left out of its training side, so no shot is heard on both sides.

**How Steps 1.7 and 1.8 must score cross-validation.** With one gun session per fold, most folds have no validation
clips for some classes (.38 is in folds 1 and 3 only, 12 ga is missing from folds 2 and 3). Per-fold accuracy, or
an average of per-fold scores, is therefore meaningless for those classes. Collect the out-of-fold predictions of
all five folds and compute every per-class metric (recall, balanced accuracy, confusion matrix) once, on that pooled
set.

**Reason.** Both held-out sessions record all four classes, from two datasets and microphone types, so test accuracy
measures generalisation to a new day and site rather than session recognition. Holding out background by session keeps
each quiet stretch on the same side as the shots it was cut from.

---

## D6. ADC sample rate, bit depth and window length

| Field | Value |
|---|---|
| Step | S1.7 (`sw/train/run_step17.py`, `results/step17/step17_report.md`) |
| Needed by | Step 1.8 (float model suite), Step 1.9 (architecture freeze), Step 4.7 (live ADC & window capture) |
| Status | **Proposed**, waiting for team sign-off at Step 1.9 |
| Decided by | (to be filled in) |
| Date | (to be filled in) |

**The question.** What sample rate (`8 / 16 / 22.05 kHz`), ADC bit depth (`16 / 12 / 10 / 8 / 6` bits, MSB-truncated
before the per-window power-of-two hardware gain), and window length (`0.25 / 0.40 / 0.50 s`) give the cheapest
hardware input within `1.0` percentage point of the best cross-validated macro-F1?

**Pre-registered selection rule** (`results/step17/selection_rule.json`, committed in `9fe223b` before running the
sweep): score every point by its 3-seed mean (`seeds = 1701, 1702, 1703`) of pooled 5-fold session-disjoint CV
`macro-F1` using the fixed `cnn1d` topology (`sw/models/cnn1d.py`). Pick the cheapest `(sample_rate ASC, adc_bits ASC)`
within `1.0` percentage point (`0.01`) of the grid maximum, then at that `(rate, bits)` pick the shortest window in
`{0.25, 0.40, 0.50} s` within `1.0` pp of the best window.

**Measured `(rate × bits)` grid at `0.50 s`** (`225` fold trainings; `results/step17/adc_sweep_grid.csv`):

| Sample rate | 16-bit | 12-bit | 10-bit | 8-bit | 6-bit | 1D-CNN MACs |
|---:|---:|---:|---:|---:|---:|---:|
| **8,000 Hz** | 42.07 ± 1.53 % | **42.15 ± 0.97 %** | **41.42 ± 1.96 % (D6)** | 37.78 ± 0.27 % | 33.53 ± 0.19 % | 1,348,256 |
| **16,000 Hz** | 41.58 ± 1.11 % | 41.30 ± 0.35 % | 40.98 ± 0.46 % | 37.17 ± 0.87 % | 34.78 ± 1.56 % | 2,731,168 |
| **22,050 Hz** | 41.75 ± 1.19 % | 41.49 ± 0.99 % | 40.64 ± 0.61 % | 36.11 ± 1.09 % | 33.53 ± 1.53 % | 3,763,360 |

**Measured window sweep at `(8,000 Hz, 10-bit)`** (`results/step17/window_sweep.csv`):

| Window | Samples | Real audio + zero tail | 1D-CNN MACs | Macro-F1 (mean ± std) | Balanced Acc |
|---:|---:|---|---:|---:|---:|
| 0.25 s | 2,000 | 0.25 s + 0.00 s | 663,712 | 40.97 ± 1.60 % | 42.10 % |
| **0.40 s (D6)** | **3,200** | **0.40 s + 0.00 s (D22 option c)** | **1,075,360** | **42.38 ± 1.46 %** | **43.24 %** |
| 0.50 s | 4,000 | 0.40 s + 0.10 s (D22 option a) | 1,348,256 | 41.42 ± 1.96 % | 42.62 % |

**Proposed choice (by the pre-registered 1-pp rule): `8,000 Hz, 10-bit, 0.40 s` (`3,200` samples, `1,075,360` MACs).**

**Key physical findings & trade-off for the Step 1.9 freeze meeting:**
1. **ADC bit-depth knee is sharp and rate-invariant between 10-bit and 8-bit:** Across all three sample rates, `16`,
   `12`, and `10` bits sit on a flat plateau, while `8-bit` drops by `~4.4` pp and `6-bit` drops by `~7.5` pp because
   distant gunshots lose dynamic range before the barrel-shift window gain. `10-bit` (or `12-bit` on standard FPGA
   XADC, which is free on Artix-7!) captures the full accuracy.
2. **`22.05 kHz` adds nothing over `16 kHz`:** `22.05 kHz` matches `16 kHz` within `0.17` pp (`41.75 %` vs `41.58 %`)
   at `+38 %` MAC cost, so `22.05 kHz` is eliminated.
3. **`8 kHz` vs `16 kHz` nuance on `.38 Special`:** While `8 kHz` matches `16 kHz` on 5-class `macro-F1` at half the
   MACs, `.38 Special` recall is higher at `16 kHz` (`57.3 %` at 16-bit / `55.2 %` at 12-bit vs `45.2 % / 37.9 %` at
   `8 kHz`). Step 1.8 therefore evaluates both the D6 winner (`8 kHz, 10-bit, 0.40 s`) and `cnn1d_spec51` (`16 kHz,
   16-bit, 0.50 s`, keeping `arith_spec.md` §5.1 unchanged) side-by-side so the team can decide at Step 1.9 whether
   the `2.5×` MAC & buffer reduction outweighs amending `arith_spec.md` §5.1 (`L_in = 3,200`, `conv1 -> 393`,
   `pool1 -> 98`, `pool2 -> 24`) and `.38` recall.
4. **Revisiting D22 (`0.40 s` vs `0.50 s`):** Removing the `0.10 s` artificial zero tail (`0.40 s` = D22 option c)
   slightly *improves* macro-F1 (`42.38 %` vs `41.42 %`) while cutting MACs by `20 %`. If the team adopts `0.40 s` in
   Step 1.9, D22 moves from option (a) to option (c) and `arith_spec.md` §5.1 updates `INPUT_LEN` accordingly.

---

## D24. Hardware model selection (Step 1.8 float model suite, for Step 1.9 freeze)

| Field | Value |
|---|---|
| Step | S1.8 (`sw/train/run_step18.py`, `results/step18/step18_report.md`) |
| Needed by | Step 1.9 (architecture freeze gate), Step 1.10 (PTQ quantization) |
| Status | **Proposed**, waiting for Step 1.9 team sign-off |
| Decided by | (to be filled in) |
| Date | (to be filled in) |

**The question.** Within the hardware budget (`≤ 4.0 M` MACs, `≤ 30 k` parameters, tile-friendly multiples of 16),
does a spectrogram 2D-CNN beat the raw-waveform 1D-CNN by more than the pre-registered `+5.0` percentage-point
macro-F1 promotion margin, or does the raw-waveform 1D-CNN stay the frozen hardware architecture?

**Pre-registered promotion rule** (`results/step18/promotion_rule.json`, committed before training): `cnn2d_mel`
replaces `cnn1d` in hardware only if `macro_f1(cnn2d_mel) − macro_f1(cnn1d) > 0.05` (`+5.0` pp) on pooled 5-fold
session-disjoint CV (`3` seeds); `cnn1d_w64` (`conv3 = 64`) replaces base `cnn1d` (`conv3 = 32`) only if it gains
`> 0.01` (`+1.0` pp).

**Measured candidate suite** (`results/step18/model_comparison.csv`, script-computed DSP + neural MACs):

| Model | DSP MACs | Classifier MACs | Total MACs | Params | Budget OK | Pooled Macro-F1 | Balanced Acc | `.38` Recall |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| `cnn1d` (`8 kHz, 10b, 0.40 s`) | 0 | 1,075,360 | 1,075,360 | 15,093 | yes | 41.02 ± 0.87 % | 41.57 % | 35.3 % |
| `cnn1d_bn` (folded into `cnn1d`) | 0 | 1,075,360 | 1,075,360 | 15,093 | yes | 42.24 ± 0.15 % | 43.00 % | 32.7 % |
| `cnn1d_spec51` (`16 kHz, 16b, 0.50 s`) | 0 | 2,731,168 | 2,731,168 | 15,093 | yes | 42.36 ± 0.70 % | 44.76 % | 50.8 % |
| `cnn1d_w64` (`conv3 = 64`) | 0 | 1,296,704 | 1,296,704 | 24,501 | yes | 40.20 ± 1.52 % | 40.23 % | 31.1 % |
| `cnn2d_mel` (log-mel 2D-CNN) | 83,577 | 2,244,256 | 2,327,833 | 14,213 | yes | 42.85 ± 0.79 % | 42.60 % | 10.2 % |
| `dscnn` (log-mel DS-CNN) | 83,577 | 1,370,432 | 1,454,009 | 9,797 | yes | 41.66 ± 0.61 % | 41.78 % | 26.2 % |
| `mfcc_svm_linear` | 99,801 | 468 | 100,269 | 551 | yes | 45.22 ± 0.16 % | 46.41 % | 43.9 % |
| `mfcc_svm_rbf` | 99,801 | 858,749 | 958,550 | 858,837 | no (SVs) | 41.86 ± 0.53 % | 41.88 % | 20.1 % |

**Leakage & split-regime falsification (`results/step18/leakage_and_splits.json`):**
- **Session-disjoint 5-fold CV vs. Device-disjoint vs. Random clip split (`cnn1d`):**
  `41.02 %` (session-disjoint) → `54.89 %` (group-disjoint unseen microphones, `.38` recall `72.5 %`) → **`68.44 %`**
  (random clip split, `.38` recall `74.4 %`) — proving naive clip splits inflate macro-F1 by **`+27.4` percentage
  points** via session memorisation.
- **`38 ms` pre-trigger background-noise leakage check (4 firearm classes, `25.0 %` chance):**
  - Session-disjoint 5-fold CV: **`21.00 %` balanced accuracy (`20.36 %` macro-F1)** — at/below random chance (`0`
    ambient-noise shortcut).
  - Random clip split: **`43.49 %` balanced accuracy (`39.14 %` macro-F1)** — nearly double chance before the shot
    even occurs.

**Proposed choice: keep the raw-waveform `cnn1d` hardware architecture (`cnn1d_bn` folded training recipe; choose D6
`8 kHz / 0.40 s` vs `arith_spec.md` §5.1 `16 kHz / 0.50 s` at Step 1.9).**
- `cnn2d_mel` exceeds `cnn1d` by `+1.83` pp (`+0.61` pp vs folded `cnn1d_bn`, `+0.49` pp vs `cnn1d_spec51`), well
  under the `+5.0` pp threshold, requires an STFT/mel front end, and drops `.38` recall to `10.2 %`.
- BatchNormalization folding into `cnn1d` (`fold_batchnorm_1d`) is verified to `1.53 × 10⁻⁵` max logit difference in
  IEEE `float32`, improving stability (`±0.15 %` std) at zero hardware cost.
- Note for Step 1.9 (`docs/runbook.md` *"If it fails"*): unweighted 5-class session-disjoint CV macro-F1 (`~42–45 %`
  across all models, while `background` recall is `92–97 %` and overall accuracy is `~71 %`) is bounded by `12 ga`
  cross-dataset sparsity (`0` Cadre clips; `65 %` of CV `12 ga` windows sit in `edge-day1`). Decide with Sir at Step
  1.9 whether to freeze 5-class `cnn1d` as-is or also report a 4-class merged head (`handgun / rifle / shotgun /
  background`), which costs zero hardware changes (`dense` uses 4 of 16 tile columns instead of 5).

---

## Confirmations needed from the model owner (T1.1)

These are not choices. They are facts about `sw/models/cnn1d.py` that `docs/arith_spec.md` §5 assumes. They have to be
checked against the file once it exists:

1. `conv2` and `conv3` use `padding='same'`. `conv1` and both max-pool layers use `'valid'`.
2. Every conv layer is followed by ReLU, and the dense layer is not.
3. The per-layer table printed by `cnn1d.py` reproduces §5.1 exactly: 15,008 weights, 85 biases and 2,731,168 MACs.

If any of these fails, either the model or the spec is wrong, and it is fixed before anything else is built.
