# Window check (Step 1.6)

`sw/frontend/check_windows.py`; window_index.csv sha256 `e7b9dc854ca20d31ac6318af30770770bda46d5475911255d5173c8d1ea72f06`; epochs 0:0,0:1,0:2,1:0,2:0,3:0,4:0 (seed 2026).

- PASS: gun_windows.npz (11186 windows) belongs to this window_index.csv (11186 rows, sha256 match)
- PASS: gun windows are int16 [8000] at 16000 Hz
- PASS: every gun window re-derived from its clip by detector.py and verified against a reference shot: 11186 identical, 0 different
- PASS: every gun window starts 800 samples before its trigger
- PASS: fold 0 epoch 0: 20.4 % of 3182 background windows from the gun microphones (>= 15 %), 1273 detector-cut, 1922 gun windows mixed onto noise
- PASS: fold 0 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: no background window comes from a clip recorded below 16000 Hz (0)
- PASS: fold 0 epoch 1: 20.6 % of 3182 background windows from the gun microphones (>= 15 %), 1273 detector-cut, 1805 gun windows mixed onto noise
- PASS: fold 0 epoch 1: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 0 epoch 2: 20.3 % of 3182 background windows from the gun microphones (>= 15 %), 1273 detector-cut, 1870 gun windows mixed onto noise
- PASS: fold 0 epoch 2: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 1 epoch 0: 20.3 % of 1726 background windows from the gun microphones (>= 15 %), 690 detector-cut, 1004 gun windows mixed onto noise
- PASS: fold 1 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 2 epoch 0: 20.3 % of 3958 background windows from the gun microphones (>= 15 %), 1583 detector-cut, 2395 gun windows mixed onto noise
- PASS: fold 2 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 3 epoch 0: 20.1 % of 3108 background windows from the gun microphones (>= 15 %), 1243 detector-cut, 1874 gun windows mixed onto noise
- PASS: fold 3 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 4 epoch 0: 20.4 % of 3861 background windows from the gun microphones (>= 15 %), 1544 detector-cut, 2327 gun windows mixed onto noise
- PASS: fold 4 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: every gun window's last 1600 samples are zero (0 are not)
- PASS: no gun window is zero-padded at either end of its real part (longest zero run at an end: 80 samples, limit 160)
- INFO: zero runs inside the real audio of gun windows (recording dropouts): longest 219 samples, 4 windows above 160
- PASS: every gun window comes from a clip recorded at >= 16000 Hz (0 not)
- PASS: background/cadre-quiet: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: background/cadre-quiet: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: background/edge-quiet: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: background/edge-quiet: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: background/esc50: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: background/esc50: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: background/unfiltered: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: background/unfiltered: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: background/us8k: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: background/us8k: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: gun/c3gd: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: gun/c3gd: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: gun/cadre: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: gun/cadre: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: gun/edge: mean spectrum within 50 dB of its peak up to 8000 Hz (>= 7500 Hz)
- PASS: gun/edge: peak after gain in [16384, 32767] for every window below max shift; 0.00 % at max shift (<= 1 %)
- PASS: median peak after gain differs by 2.56 dB between sources (<= 3.0 dB)
- PASS: self-test: the 55 excluded clips (recorded below 16000 Hz) fail the bandwidth test (band edge 6031 Hz)

## Per-source window features

| kind | source | windows | length | zero tail | band edge (-50 dB) | roll-off p5 / p50 / p95 (Hz) | median peak after gain | at max shift | gain shift histogram |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| background | cadre-quiet | 228 | 8000 | 1600 | 8000 Hz | 125 / 1750 / 6781 | -2.89 dBFS | 0.00 % | 0:37, 1:27, 2:29, 3:5, 4:5, 5:37, 6:51, 7:28, 8:8, 9:1 |
| background | edge-quiet | 420 | 8000 | 1600 | 8000 Hz | 188 / 1219 / 6720 | -2.77 dBFS | 0.00 % | 0:65, 1:61, 2:56, 3:51, 4:32, 5:13, 6:10, 7:49, 8:79, 9:4 |
| background | esc50 | 551 | 8000 | 1600 | 8000 Hz | 281 / 3125 / 7422 | -2.71 dBFS | 0.00 % | 0:177, 1:125, 2:104, 3:66, 4:29, 5:26, 6:11, 7:8, 8:4, 9:1 |
| background | unfiltered | 593 | 8000 | 1600 | 8000 Hz | 344 / 1719 / 4800 | -2.76 dBFS | 0.00 % | 0:165, 1:167, 2:149, 3:65, 4:29, 5:13, 6:5 |
| background | us8k | 1390 | 8000 | 1600 | 8000 Hz | 281 / 2594 / 6844 | -2.72 dBFS | 0.00 % | 0:365, 1:376, 2:300, 3:183, 4:92, 5:47, 6:15, 7:6, 8:4, 9:2 |
| gun | c3gd | 3638 | 8000 | 1600 | 8000 Hz | 2620 / 7453 / 7812 | -1.02 dBFS | 0.00 % | 0:1694, 1:1397, 2:406, 3:83, 4:12, 5:10, 6:4, 7:3, 8:23, 9:6 |
| gun | cadre | 1702 | 8000 | 1600 | 8000 Hz | 1094 / 2625 / 4500 | -0.33 dBFS | 0.00 % | 0:1543, 1:104, 2:27, 3:21, 4:4, 5:3 |
| gun | edge | 5846 | 8000 | 1600 | 8000 Hz | 1281 / 3812 / 6469 | -1.97 dBFS | 0.00 % | 0:1680, 1:659, 2:1029, 3:1124, 4:851, 5:308, 6:113, 7:64, 8:18 |

## Sampled epochs

| fold | epoch | windows | gun | gun mixed onto noise | background | from gun microphones | share | detector-cut background | random background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 9545 | 4441 | 1922 | 3182 | 648 | 20.4 % | 1273 | 1272 |
| 0 | 1 | 9545 | 4558 | 1805 | 3182 | 656 | 20.6 % | 1273 | 1272 |
| 0 | 2 | 9545 | 4493 | 1870 | 3182 | 647 | 20.3 % | 1273 | 1272 |
| 1 | 0 | 5177 | 2447 | 1004 | 1726 | 350 | 20.3 % | 690 | 690 |
| 2 | 0 | 11875 | 5522 | 2395 | 3958 | 804 | 20.3 % | 1583 | 1583 |
| 3 | 0 | 9323 | 4341 | 1874 | 3108 | 625 | 20.1 % | 1243 | 1243 |
| 4 | 0 | 11583 | 5395 | 2327 | 3861 | 787 | 20.4 % | 1544 | 1544 |
