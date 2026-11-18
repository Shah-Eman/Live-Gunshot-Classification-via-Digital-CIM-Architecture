# Step 1.7 — ADC Sample-Rate, Bit-Depth & Window Sweep (Decision D6)

**Pre-registered selection rule:** Cheapest `(sample_rate ASC, adc_bits ASC)` within `1.0` percentage point (`0.01`) of the highest 3-seed mean pooled 5-fold session-disjoint CV macro-F1; then shortest `window_s` in `{0.25, 0.40, 0.50}` s within `1.0` pp of the best window at that `(rate, bits)`.

## Part A: Sample rate × ADC bit depth (at 0.50 s window, 3 seeds × 5 folds)

| Rate (Hz) | Bits | MACs | Macro-F1 (mean ± std) | Balanced Acc | .38 Recall | Within 1 pp | Chosen |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 8000 | 16 | 1,348,256 | 42.07 ± 1.53 % | 44.07 % | 45.2 % | yes |  |
| 8000 | 12 | 1,348,256 | 42.15 ± 0.97 % | 43.76 % | 42.3 % | yes |  |
| 8000 | 10 | 1,348,256 | 41.42 ± 1.96 % | 42.62 % | 37.9 % | yes | **D6** |
| 8000 | 8 | 1,348,256 | 37.78 ± 0.27 % | 38.15 % | 26.5 % | no |  |
| 8000 | 6 | 1,348,256 | 33.53 ± 0.19 % | 33.29 % | 16.6 % | no |  |
| 16000 | 16 | 2,731,168 | 41.58 ± 1.11 % | 44.66 % | 57.3 % | yes |  |
| 16000 | 12 | 2,731,168 | 41.30 ± 0.35 % | 44.11 % | 55.2 % | yes |  |
| 16000 | 10 | 2,731,168 | 40.98 ± 0.46 % | 43.60 % | 52.0 % | no |  |
| 16000 | 8 | 2,731,168 | 37.17 ± 0.87 % | 37.81 % | 29.0 % | no |  |
| 16000 | 6 | 2,731,168 | 34.78 ± 1.56 % | 34.32 % | 18.4 % | no |  |
| 22050 | 16 | 3,763,360 | 41.75 ± 1.19 % | 44.70 % | 56.2 % | yes |  |
| 22050 | 12 | 3,763,360 | 41.49 ± 0.99 % | 44.33 % | 53.8 % | yes |  |
| 22050 | 10 | 3,763,360 | 40.64 ± 0.61 % | 42.77 % | 46.2 % | no |  |
| 22050 | 8 | 3,763,360 | 36.11 ± 1.09 % | 36.34 % | 23.4 % | no |  |
| 22050 | 6 | 3,763,360 | 33.53 ± 1.53 % | 32.95 % | 14.6 % | no |  |

## Part B: Window length sweep at (8000 Hz, 10-bit)

| Window (s) | Samples | MACs | Macro-F1 (mean ± std) | Balanced Acc | Within 1 pp | Chosen |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0.25 | 2000 | 663,712 | 40.97 ± 1.60 % | 42.10 % | no |  |
| 0.40 | 3200 | 1,075,360 | 42.38 ± 1.46 % | 43.24 % | yes | **D6** |
| 0.50 | 4000 | 1,348,256 | 41.42 ± 1.96 % | 42.62 % | yes |  |

## Chosen operating point (Decision D6): `8000 Hz, 10-bit, 0.40 s`
- Pooled 5-fold CV Macro-F1: **42.38 ± 1.46 %**
- 1D-CNN MACs: **1,075,360** (15,093 parameters)
- Final test set (`splits.test`) touched: **No (0 clips)**.
