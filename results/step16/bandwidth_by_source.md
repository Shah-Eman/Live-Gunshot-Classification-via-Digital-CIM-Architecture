# Bandwidth per source (Step 1.6, item 5)

Measured by `sw/frontend/scan_clips.py` on every clip after resampling to 16 kHz and int16. Roll-off = frequency below which 95 % of the energy lies; high band = energy above 5.5 kHz relative to the total.

| source | native rate | clips | roll-off p5 / p50 / p95 (Hz) | high band p5 / p50 / p95 (dB) |
|---|---|---:|---|---|
| c3gd | native >= 16000 | 4233 | 2750.0 / 6500.0 / 7781.2 | -20.9 / -9.8 / -1.4 |
| cadre | native >= 16000 | 1712 | 906.2 / 2375.0 / 4281.2 | -29.9 / -21.3 / -15.8 |
| cadre-quiet | native >= 16000 | 258 | 93.8 / 1281.2 / 2871.9 | -43.6 / -28.8 / -19.1 |
| edge | native >= 16000 | 2049 | 1137.5 / 3187.5 / 6406.2 | -29.3 / -19.2 / -9.8 |
| edge-quiet | native >= 16000 | 830 | 218.8 / 1000.0 / 2312.5 | -48.8 / -34.7 / -22.7 |
| esc50 | native >= 16000 | 1964 | 406.2 / 2828.2 / 7151.5 | -46.8 / -22.2 / -6.4 |
| unfiltered | native >= 16000 | 3271 | 453.1 / 2031.2 / 4718.8 | -48.1 / -27.5 / -14.8 |
| us8k | native < 16000 | 55 | 400.0 / 1625.0 / 4009.4 | -69.7 / -53.1 / -33.9 |
| us8k | native >= 16000 | 8129 | 281.2 / 1968.8 / 6781.2 | -45.4 / -27.2 / -7.9 |
