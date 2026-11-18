# Window check (Step 1.6)

`sw/frontend/check_windows.py`; window_index.csv sha256 `be82c4a1a4d45ee11dd1aecc689b4a43ccfcc544049947114a495040d7016667`; epochs 0:0,0:1,0:2,1:0,2:0,3:0,4:0 (seed 2026).

- PASS: gun_windows.npz (11497 windows) belongs to this window_index.csv (11497 rows, sha256 match)
- PASS: gun windows are int16 [8000] at 16000 Hz
- PASS: every gun window re-derived from its clip by detector.py and verified against a reference shot: 11497 identical, 0 different
- PASS: every gun window starts 640 samples before its trigger
- PASS: fold 0 epoch 0: 20.6 % of 3328 background windows from the gun microphones (>= 15 %), 1331 detector-cut, 2019 gun windows mixed onto noise
- PASS: fold 0 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: no background window comes from a clip recorded below 16000 Hz (0)
- PASS: fold 0 epoch 1: 20.5 % of 3328 background windows from the gun microphones (>= 15 %), 1331 detector-cut, 1881 gun windows mixed onto noise
- PASS: fold 0 epoch 1: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 0 epoch 2: 20.4 % of 3328 background windows from the gun microphones (>= 15 %), 1331 detector-cut, 1942 gun windows mixed onto noise
- PASS: fold 0 epoch 2: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 1 epoch 0: 20.0 % of 1874 background windows from the gun microphones (>= 15 %), 750 detector-cut, 1107 gun windows mixed onto noise
- PASS: fold 1 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 2 epoch 0: 20.3 % of 3947 background windows from the gun microphones (>= 15 %), 1579 detector-cut, 2388 gun windows mixed onto noise
- PASS: fold 2 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 3 epoch 0: 20.5 % of 3256 background windows from the gun microphones (>= 15 %), 1302 detector-cut, 1967 gun windows mixed onto noise
- PASS: fold 3 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: fold 4 epoch 0: 20.4 % of 4024 background windows from the gun microphones (>= 15 %), 1610 detector-cut, 2433 gun windows mixed onto noise
- PASS: fold 4 epoch 0: every window has 8000 samples, a zero tail of 1600 and no zero padding at either end of its real part (0 bad)
- PASS: every gun window's last 1600 samples are zero (0 are not)
- PASS: no gun window is zero-padded at either end of its real part (longest zero run at an end: 101 samples, limit 160)
- INFO: zero runs inside the real audio of gun windows (recording dropouts): longest 242 samples, 6 windows above 160
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
- PASS: median peak after gain differs by 2.65 dB between sources (<= 3.0 dB)
- PASS: self-test: the 55 excluded clips (recorded below 16000 Hz) fail the bandwidth test (band edge 6031 Hz)

## Per-source window features

| kind | source | windows | length | zero tail | band edge (-50 dB) | roll-off p5 / p50 / p95 (Hz) | median peak after gain | at max shift | gain shift histogram |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| background | cadre-quiet | 247 | 8000 | 1600 | 8000 Hz | 125 / 1656 / 6750 | -2.99 dBFS | 0.00 % | 0:45, 1:28, 2:22, 3:12, 4:8, 5:37, 6:56, 7:34, 8:4, 9:1 |
| background | edge-quiet | 439 | 8000 | 1600 | 8000 Hz | 156 / 1219 / 6441 | -2.87 dBFS | 0.00 % | 0:92, 1:62, 2:51, 3:37, 4:21, 5:18, 6:20, 7:44, 8:88, 9:6 |
| background | esc50 | 571 | 8000 | 1600 | 8000 Hz | 250 / 2844 / 7203 | -2.59 dBFS | 0.00 % | 0:182, 1:129, 2:100, 3:76, 4:43, 5:16, 6:12, 7:7, 8:4, 9:2 |
| background | unfiltered | 620 | 8000 | 1600 | 8000 Hz | 312 / 1656 / 4753 | -2.79 dBFS | 0.00 % | 0:205, 1:181, 2:143, 3:55, 4:16, 5:15, 6:4, 7:1 |
| background | us8k | 1451 | 8000 | 1600 | 8000 Hz | 250 / 2438 / 7016 | -2.82 dBFS | 0.00 % | 0:379, 1:425, 2:296, 3:200, 4:84, 5:40, 6:11, 7:9, 8:4, 9:3 |
| gun | c3gd | 3947 | 8000 | 1600 | 8000 Hz | 2750 / 6688 / 7781 | -0.81 dBFS | 0.00 % | 0:1985, 1:1406, 2:411, 3:86, 4:13, 5:10, 6:4, 7:3, 8:23, 9:6 |
| gun | cadre | 1703 | 8000 | 1600 | 8000 Hz | 1031 / 2594 / 4434 | -0.33 dBFS | 0.00 % | 0:1544, 1:104, 2:27, 3:21, 4:4, 5:3 |
| gun | edge | 5847 | 8000 | 1600 | 8000 Hz | 1344 / 3812 / 6469 | -1.96 dBFS | 0.00 % | 0:1684, 1:660, 2:1031, 3:1123, 4:851, 5:303, 6:113, 7:64, 8:18 |

## Sampled epochs

| fold | epoch | windows | gun | gun mixed onto noise | background | from gun microphones | share | detector-cut background | random background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 9984 | 4637 | 2019 | 3328 | 686 | 20.6 % | 1331 | 1331 |
| 0 | 1 | 9984 | 4775 | 1881 | 3328 | 683 | 20.5 % | 1331 | 1331 |
| 0 | 2 | 9984 | 4714 | 1942 | 3328 | 678 | 20.4 % | 1331 | 1331 |
| 1 | 0 | 5622 | 2641 | 1107 | 1874 | 375 | 20.0 % | 750 | 749 |
| 2 | 0 | 11841 | 5506 | 2388 | 3947 | 802 | 20.3 % | 1579 | 1578 |
| 3 | 0 | 9767 | 4544 | 1967 | 3256 | 666 | 20.5 % | 1302 | 1302 |
| 4 | 0 | 12071 | 5614 | 2433 | 4024 | 820 | 20.4 % | 1610 | 1609 |
