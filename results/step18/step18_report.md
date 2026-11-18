# Step 1.8 — Float Models Within the Hardware Budget

**Locked D6 operating point (from Step 1.7):** `8000 Hz, 10-bit, 0.40 s`
**Pre-registered promotion rule:** 2D-CNN replaces 1D-CNN only if it beats the hardware 1D-CNN by `> 5.0` points of pooled CV macro-F1; `cnn1d_w64` replaces base `cnn1d` only if it gains `> 1.0` point.

## 1. Model Comparison (Pooled 5-Fold Session-Disjoint CV, 3 Seeds)

| Model | Front-End MACs | Classifier MACs | Total MACs | Params | Budget OK | Macro-F1 (mean ± std) | Balanced Acc | .38 Recall |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| `cnn1d` | 0 | 1,075,360 | 1,075,360 | 15,093 | yes | **41.02 ± 0.87 %** | 41.57 % | 35.3 % |
| `cnn1d_spec51` | 0 | 2,731,168 | 2,731,168 | 15,093 | yes | **42.36 ± 0.70 %** | 44.76 % | 50.8 % |
| `cnn1d_w64` | 0 | 1,296,704 | 1,296,704 | 24,501 | yes | **40.20 ± 1.52 %** | 40.23 % | 31.1 % |
| `cnn1d_bn` | 0 | 1,075,360 | 1,075,360 | 15,093 | yes | **42.24 ± 0.15 %** | 43.00 % | 32.7 % |
| `cnn2d_mel` | 83,577 | 2,244,256 | 2,327,833 | 14,213 | yes | **42.85 ± 0.79 %** | 42.60 % | 10.2 % |
| `dscnn` | 83,577 | 1,370,432 | 1,454,009 | 9,797 | yes | **41.66 ± 0.61 %** | 41.78 % | 26.2 % |
| `mfcc_svm_linear` | 99,801 | 468 | 100,269 | 551 | yes | **45.22 ± 0.16 %** | 46.41 % | 43.9 % |
| `mfcc_svm_rbf` | 99,801 | 858,749 | 958,550 | 858,837 | no (SVs) | **41.86 ± 0.53 %** | 41.88 % | 20.1 % |

## 2. Per-Class Recall Across Models (Pooled 5-Fold CV)

| Model | pistol_9mm | revolver_38spl | rifle_556_223 | shotgun_12ga | background |
|---|---:|---:|---:|---:|---:|
| `cnn1d` | 53.4% | 35.3% | 22.7% | 2.1% | 94.3% |
| `cnn1d_spec51` | 56.1% | 50.8% | 17.8% | 4.5% | 94.7% |
| `cnn1d_w64` | 49.9% | 31.1% | 22.1% | 3.5% | 94.6% |
| `cnn1d_bn` | 61.8% | 32.7% | 20.3% | 5.3% | 95.0% |
| `cnn2d_mel` | 51.9% | 10.2% | 49.7% | 5.4% | 95.8% |
| `dscnn` | 57.1% | 26.2% | 25.6% | 3.3% | 96.6% |
| `mfcc_svm_linear` | 71.6% | 43.9% | 15.9% | 8.6% | 92.1% |
| `mfcc_svm_rbf` | 69.3% | 20.1% | 22.5% | 0.8% | 96.7% |

## 3. Per-Source Accuracy Across Models (Pooled 5-Fold CV)

| Model | c3gd | cadre | cadre-quiet | edge | edge-quiet | esc50 | unfiltered | us8k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnn1d` | 62.5% | 49.7% | 90.3% | 19.1% | 99.4% | 82.7% | 98.2% | 95.2% |
| `cnn1d_spec51` | 49.5% | 52.6% | 99.5% | 27.4% | 99.5% | 82.0% | 97.9% | 95.9% |
| `cnn1d_w64` | 54.2% | 45.6% | 92.1% | 20.3% | 99.5% | 83.4% | 98.2% | 95.7% |
| `cnn1d_bn` | 55.7% | 57.8% | 93.5% | 24.2% | 99.7% | 85.4% | 97.7% | 95.9% |
| `cnn2d_mel` | 54.5% | 41.4% | 96.4% | 35.3% | 98.3% | 88.4% | 97.6% | 96.8% |
| `dscnn` | 48.9% | 48.1% | 95.5% | 28.6% | 99.5% | 90.8% | 97.5% | 97.5% |
| `mfcc_svm_linear` | 58.8% | 61.4% | 99.1% | 30.1% | 99.7% | 78.8% | 96.3% | 92.9% |
| `mfcc_svm_rbf` | 59.4% | 54.2% | 99.1% | 27.4% | 100.0% | 90.4% | 97.9% | 97.4% |

## 4. Session Leakage Quantification (`cnn1d` Across Split Regimes)

| Split Regime | Macro-F1 (mean ± std) | Balanced Acc | .38 Recall | Meaning |
|---|---:|---:|---:|---|
| **Session-disjoint 5-fold CV** | **41.02 ± 0.87 %** | 41.57 % | 35.3 % | True unseen-day / unseen-range generalisation |
| **Group-disjoint device split** | 54.89 ± 0.54 % | 59.63 % | 72.5 % | Unseen microphones within known sessions (0 shared shots) |
| **Random clip split** | 68.44 ± 0.99 % | 73.61 % | 74.4 % | Inflated by **+27.4 pp** from session memorisation |

## 5. Pre-Trigger Background-Noise Leakage Check (4 Firearm Classes, Chance = 25.0%)

Trained `pretrigger_cnn` on ONLY the `38.0 ms` (`304` samples) strictly preceding the detector trigger:
- **Session-disjoint 5-fold CV:** Balanced accuracy = **21.00 %**, Macro-F1 = **20.36 %** (vs 25.0% chance).
- **Random clip split:** Balanced accuracy = **43.49 %**, Macro-F1 = **39.14 %** — proving that random splits identify the firearm from ambient background noise before the shot even fires!

## 6. Hardware Promotion Recommendation (for Step 1.9)

- `cnn2d_mel` − `cnn1d` macro-F1 difference: `+1.83` pp (promotion threshold: `+5.00` pp) -> **2D-CNN promoted: NO**.
- Recommended hardware model: **`cnn1d`**.
- BatchNormalization folding numerical check: max logit discrepancy = `1.53e-05` (`< 1e-4`).
- Held-out final test set (`splits.test`) touched: **No (0 clips)**.
