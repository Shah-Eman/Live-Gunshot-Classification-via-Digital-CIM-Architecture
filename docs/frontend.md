# Front end: detector, window and gain (Step 1.6)

What happens between the ADC and the network input, written so that Step 4.7 (RTL detector and window capture) and
`arith_spec.md` can take it over unchanged. The Python code is the reference; every number below comes from
`sw/frontend/config.py`, and Step 1.7 changes the front end only by editing that file.

```
ADC (int16, 16 kHz) ──> detector ──trigger──> window capture (ring buffer) ──> gain ──> int8 input ──> network
```

| Stage | Python reference | Decision |
|---|---|---|
| Clip to int16 stream (training only) | `audio.py: load_clip`, `to_int16` | D22 (bandwidth) |
| Detector | `detector.py: detect_reference` (sample loop), `detect` (vectorised, same triggers) | D22 |
| Window placement | `windows.py: placement`, `cut` | D22 option (a) |
| Gain | `audio.py: gain_shift`, `apply_gain` | D22 (normalisation) |
| int16 to int8 | not written yet: proposal in section 4 | to be added to `arith_spec.md` |

## 1. Input stream

16 kHz, mono, int16, as a 16-bit ADC would deliver it. Training clips are turned into the same stream by averaging
channels, polyphase resampling to 16 kHz (`manifest.load_audio`), clamping to [−1, 1] and quantising with
`clip(rint(x · 32768), −32768, 32767)`. Every source goes through exactly these steps. Clips recorded below 16 kHz
are dropped rather than low-passed (D22).

## 2. Detector

Integer only. Per sample `n`, with every register zero after reset and `x[−1] = 0`:

```
d[n]  = x[n] − x[n−1]
p[n]  = (d[n] · d[n]) >> PS
Ef[n] = p[n−NF+1] + … + p[n]                    running sum: Ef += p[n] − p[n−NF]
Eb[n] = p[n−NF−NB+1] + … + p[n−NF]              running sum: Eb += p[n−NF] − p[n−NF−NB]
cand  = (Ef << log2(NB/NF)) ≥ (Eb << RATIO)  and  Ef ≥ EMIN  and  n ≥ NF+NB−1

if R_cnt == 0:
    if cand:                     trigger at n;            R_cnt = R−1;  Epk = Ef
elif cand and Ef > (Epk << PRE): replace last trigger by n; R_cnt = R−1;  Epk = Ef
else:                            R_cnt −= 1;              Epk = max(Epk, Ef)
```

| Parameter | Value | Meaning |
|---|---:|---|
| `NF` | 32 (2 ms) | short-term window |
| `NB` | 512 (32 ms) | background window, the samples just before the short-term window |
| `PS` | 4 | power shift |
| `RATIO` | 3 | candidate when short-term mean power ≥ 8× (9 dB) the background mean |
| `EMIN` | 256 | absolute floor on `Ef` (rejects digital silence and dither) |
| `R` | 1600 (0.10 s) | refractory period; must be ≤ 5600 (the real audio after the trigger), checked by `config.validate` |
| `PRE` | 2 | a 4× (6 dB) louder candidate during the refractory period replaces the trigger |

Warm-up: no trigger in the first `NF + NB − 1 = 543` samples after reset.

**Register widths** (derived by `detector.register_widths` from the config and asserted on every sample by both
implementations):

| Register | Width | Range / note |
|---|---:|---|
| `x` | 16 signed | ADC sample |
| `d` | 17 signed | −65535 … 65535 |
| `d · d` | 32 unsigned | ≤ 65535², the only multiplier (17 × 17 bit) |
| `p` | 28 unsigned | after `>> 4` |
| `p` delay line | 544 × 28 | `NF + NB` entries; or recompute `p[n−NF]`, `p[n−NF−NB]` from the audio ring buffer |
| `Ef` | 33 unsigned | `NF · (2²⁸ − 1)` |
| `Eb` | 37 unsigned | `NB · (2²⁸ − 1)` |
| `Ef << 4` | 37 | ratio comparison, left side |
| `Eb << 3` | 40 | ratio comparison, right side: one 40-bit unsigned comparator |
| `Epk` | 33 unsigned | running maximum of `Ef` since the trigger |
| `Epk << 2` | 35 | preemption comparison: one 35-bit unsigned comparator |
| `R_cnt` | 11 unsigned | 0 … 1599 |

Every scaling is a shift. The RTL must give the same trigger sample indices as `detect_reference` on the same int16
stream; `test_detector.py` already proves `detect` and `detect_reference` agree on 12 real clips of every source.

**Why preemption.** Ranges have other shooters. Without it, a faint unrelated shot starts a refractory period that
hides the labelled shot right after it (recall was about 85–90 % without it). A preempted trigger abandons the window
capture it started; only final triggers produce windows. A trigger becomes final at most `R` samples after it fired.

## 3. Window placement (D22, option a)

| | Samples | Time |
|---|---:|---:|
| window start | trigger − 800 | −0.05 s |
| real audio | 6400 | −0.05 s … +0.35 s around the trigger |
| zeroed tail | 1600 | the last 0.10 s of the 0.50 s input |
| network input | 8000 | `arith_spec.md` §5.1, unchanged |

A window whose start falls before the clip's audio, or that has less than 6400 samples of audio after its start, is
dropped, never padded. "Audio" means between the first and the last non-zero sample (`windows.audio_bounds`), so
file padding is never taken for signal.

**Notes for Step 4.7.**

- The ring buffer must hold at least 800 samples before the trigger plus 5600 after it. The tail is not captured; the
  read-out writes 1600 zeros.
- `R` (1600) is shorter than the 5600 samples captured after a trigger, so a new trigger can arrive while the previous
  window is still being captured. Training uses every such window (2,245 of the 11,186 gun windows start 1600 to 6399
  samples after the previous window of the same clip), so
  the capture logic needs either two windows in flight or a ring buffer of at least 6400 + 1600 samples with
  start-address bookkeeping. Choose this in Step 4.7; do not drop the second window silently.
- Preemption can move the trigger up to `R` samples later, which is well before the capture ends.

## 4. Gain (normalisation) and the network input

**Gain** (`audio.py`), identical for every source and in hardware, applied to the 6400 real samples:

```
peak = max |w[i]|, i < 6400            (16-bit max register, updated while the window is captured)
k    = clamp(15 − bit_length(peak), 0, 10)     (peak = 0 gives k = 10; 4-bit priority encoder)
out  = saturate_int16(w << k)          (barrel shift on read-out; the zero tail stays zero)
```

With this `k`, `peak << k ≤ 32767`, so the shift never saturates except for a window whose peak is exactly −32768
(`k = 0`, value unchanged). After the gain the peak lies in [16384, 32767] unless `k` hit 10. Measured on every
gun window and on sampled background epochs: no window at `k = 10`, and the median peak after gain differs by 2.56 dB
between sources (`results/step16/check_windows_report.md`).

**Network input: proposal for `arith_spec.md` §2.4.** The spec quantises the input with `a_q = quantize(x, s_in)`,
`s_in` from calibration. The proposal is to fix `s_in = 1/128` (in units of full scale, `x = out / 32768`). Then

```
a_q = clamp((out + 128) >> 8, −128, 127)        (arithmetic shift)
```

is bit-exact to `quantize(out / 32768, 1/128)` with `round_half_up`: checked for all 65,536 int16 values, 0
mismatches. The front end then needs no multiplier and no calibrated scale, and the gained peak uses 64 … 127 of the
int8 range. If the team keeps a calibrated `s_in` instead, the gain still applies and the int8 step becomes a
requantization with its own `(m0, n)`. Either way this has to be written into `arith_spec.md` before `quantize.py`
(Step 1.3) fixes the input scale, and Step 4.7 implements the same lines.

## 5. Training-only parts

These exist only in software and never reach the RTL:

- reference shot times used to verify triggers (`windows.reference_samples`, `match`);
- per-epoch background sampling and mixing (`background.py`): windows are mixed before the gain, then gained by the
  same rule, with the same placement and zeroed tail, so the network never sees a window the hardware could not
  produce.

## 6. Running it

```
make step16_checks PYTHON=../.venv-audio/bin/python            # tests, split check, window check
make windows scan splits listen PYTHON=../.venv-audio/bin/python
```

Paths default to `../Dataset-Final`, `../step16_outputs` and the edge labels in `sw/data/meta/`; override with
`DATASET=…`, `OUT=…`. The checks re-read the audio, so they need a copy of `Dataset-Final`.
