# Split check (Step 1.6, Decision D23)

`sw/data/check_splits.py` on `splits.json` (sha256 `e5037db37ec301cac37bf801cf7b281a6020ae287ffd8b0947643e9dd657c399`), seeds {'cv': 16, 'background_test_sessions': 1623, 'device_split': 17, 'random_split': 18}.

- PASS: splits.json matches the manifest (sha256 5a9d14c09d594140...)
- PASS: every path in splits.json is a manifest clip (0 unknown)
- PASS: every clip is in exactly one of excluded / test / one fold (0 missing, 0 in several)
- PASS: only clips recorded below 16 kHz are excluded (55)
- PASS: no session_id in two folds or in a fold and the test set (0 found)
- PASS: no group in two folds or in a fold and the test set (0 found)
- PASS: every clip of ['c3gd_nj_farm_2', 'edge-day2'] is in the test set (0 missing)
- PASS: the random test sessions hold no gun clip (so no session is split)
- PASS: random background test sessions: 452 of 3010 candidates (15.0 %, seed 1623)
- PASS: the test set holds no other session ([])
- PASS: D23 gun test counts {'pistol_9mm': 1143, 'rifle_556_223': 937, 'revolver_38spl': 251, 'shotgun_12ga': 216} == expected {'pistol_9mm': 1143, 'rifle_556_223': 937, 'revolver_38spl': 251, 'shotgun_12ga': 216}
- PASS: pistol_9mm is tested on sessions ['c3gd_nj_farm_2', 'edge-day2'] from 2 datasets ['c3gd', 'edge']
- PASS: revolver_38spl is tested on sessions ['c3gd_nj_farm_2', 'edge-day2'] from 2 datasets ['c3gd', 'edge']
- PASS: rifle_556_223 is tested on sessions ['c3gd_nj_farm_2', 'edge-day2'] from 2 datasets ['c3gd', 'edge']
- PASS: shotgun_12ga is tested on sessions ['c3gd_nj_farm_2', 'edge-day2'] from 2 datasets ['c3gd', 'edge']
- PASS: device split: test inside the CV pool, 7 held-out devices, none in training (0 shared); 45 shot groups recorded on devices of both sides (same shot, different microphone: expected for a within-session device split)
- PASS: device split tests pistol_9mm (896 clips)
- PASS: device split tests revolver_38spl (192 clips)
- PASS: device split tests rifle_556_223 (409 clips)
- PASS: device split tests shotgun_12ga (158 clips)
- PASS: random split: 13868 train / 3467 test clips, every class on both sides
- PASS: every gun window (11186) belongs to a clip of the test set or a fold (0 do not)

## Clips per class

| set | pistol_9mm | revolver_38spl | rifle_556_223 | shotgun_12ga | background | sessions |
|---|---:|---:|---:|---:|---:|---:|
| test | 1143 | 251 | 937 | 216 | 2564 | 454 |
| fold 0 | 1020 | 0 | 485 | 98 | 2377 | 516 |
| fold 1 | 616 | 279 | 330 | 227 | 2378 | 503 |
| fold 2 | 335 | 0 | 0 | 0 | 2378 | 516 |
| fold 3 | 1031 | 336 | 345 | 0 | 2377 | 512 |
| fold 4 | 0 | 0 | 171 | 174 | 2378 | 516 |
| device test | 896 | 192 | 409 | 158 | 2491 | 517 |
| random test | 600 | 123 | 266 | 100 | 2378 | 1112 |
| excluded | 0 | 0 | 0 | 0 | 55 | 16 |

Gun sessions per fold (validation side): fold 0: c3gd_oh_farm; fold 1: edge-day1; fold 2: c3gd_nj_farm; fold 3: cadre; fold 4: c3gd_ny_gravel_pit

## .38 Special (reported separately)

| fold | .38 train clips | .38 validation clips | validation sessions |
|---|---:|---:|---|
| 0 | 615 | 0 | - |
| 1 | 336 | 279 | edge-day1 |
| 2 | 615 | 0 | - |
| 3 | 279 | 336 | cadre |
| 4 | 615 | 0 | - |

## Gun windows per set (window_index.csv)

| set | pistol_9mm | revolver_38spl | rifle_556_223 | shotgun_12ga |
|---|---:|---:|---:|---:|
| test | 1127 | 290 | 1385 | 467 |
| fold0 | 1001 | 0 | 462 | 91 |
| fold1 | 1277 | 796 | 1924 | 469 |
| fold2 | 0 | 0 | 0 | 0 |
| fold3 | 1026 | 334 | 342 | 0 |
| fold4 | 0 | 0 | 21 | 174 |

Note: these sets hold clips of a class but no gun window of it (every clip dropped by the D22 placement): fold2/pistol_9mm.
