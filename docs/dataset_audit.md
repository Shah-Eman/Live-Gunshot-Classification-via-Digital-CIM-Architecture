# Dataset audit (Step 0.4)

What the dataset is, where every part came from, and whether it can support five honest classes. The counts
below come from `sw/data/audit.py` (output `Dataset-Final/audit_counts.csv`); re-run it after any rebuild
instead of editing numbers here by hand.

Status: items 1–6 done (Google Speech Commands skipped by decision, §6). Provenance and permission questions
answered from the datasets' own records (§5). Review gate not yet run.

---

## 1. Result: the five classes

Classes are defined by **cartridge and gun type**, not by gun model. In every source each gun model was
recorded in only one or two sessions, so a per-model class cannot be split by session; a caliber class
pools several guns recorded at different sessions.

| Class | Guns | Clips | Sessions | Devices | Hours |
|---|---|---|---|---|---|
| `pistol_9mm` | Glock 17, Glock 19 (2 guns), Sig P225, Sig P365, P365 Micro Compact, P226, P226 Legion, P320, Steyr C9A2, HK P2000, HK P30L | 4,145 | 6 | 47 | 1.96 |
| `rifle_556_223` | Ruger AR-556, Colt M16A1, Spike's M4, Radical AR-15 (.223), Aero AR-15, S&W M&P15, HK MR556 | 2,268 | 6 | 50 | 1.32 |
| `revolver_38spl` | S&W .38 (edge), S&W 10-8, Taurus 66, Taurus 856 | 866 | 4 | 40 | 0.76 |
| `shotgun_12ga` | Remington 870, Benelli M4, Kel-Tec KS7 | 715 | 5 | 37 | 0.54 |
| `background` | no gunshot (see §3) | 14,507 | 3,029 | 35 | 17.02 |

Clips per session (firearm classes):

| Session | 9 mm | 5.56/.223 | .38 Spl | 12 ga |
|---|---|---|---|---|
| `edge-day1` (2021-11-15) | 616 | 330 | 279 | 227 |
| `edge-day2` (2021-11-22) | 23 | 236 | 194 | 144 |
| `cadre` (2017–18 campaign) | 1,031 | 345 | 336 | — |
| `c3gd_oh_farm` (2025-07-30) | 1,020 | 485 | — | 98 |
| `c3gd_nj_farm` (2025-11-13) | 335 | — | — | — |
| `c3gd_ny_gravel_pit` (2025-12-21) | — | 171 | — | 174 |
| `c3gd_nj_farm_2` (2026-04-07) | 1,120 | 701 | 57 | 72 |

**Every firearm class has at least 4 sessions** (runbook minimum: 3). Thin sessions: 9 mm `edge-day2` (23
clips) and .38 `c3gd_nj_farm_2` (57 clips) count as sessions but are too small to be a test fold alone.

Left out on purpose: S&W M&P FPC (a 9 mm carbine, 16" barrel), Savage .223 (bolt action).

Side-experiment classes (.22 LR, .45 ACP, 7.62x39) stay in `Dataset/` and are not part of `Dataset-Final/`.

---

## 2. The item-1 questions, answered per source

Only three field datasets supply gun clips. Background comes from four more.

### 2.1 Kabealo et al., edge-collected gunshot audio (`source = edge`)

| Question | Answer |
|---|---|
| Classes | Glock 17 (9 mm), Ruger AR-556 (.223), S&W .38 revolver, Remington 870 (12 ga). No non-gunshot class. |
| Collection | 2 sessions: 2021-11-15 and 2021-11-22, outdoor range at Eastern Florida State College. The Glock is almost entirely day 1 (616 vs 23 clips). Noisy real-world conditions (highway, voices, wildlife). |
| Devices | 26 devices: Raspberry Pi Zero W with USB mics, Samsung Galaxy S8/S9/S21 phones. **Yes**, each shot was recorded by several devices (`uuid`, `chanN` files). |
| Format | WAV, 44.1 kHz, mono per channel file. No lossy compression. |
| Labels | Dataset authors. **Shot times are marked** (`gunshot_location_in_seconds`); the clip peak sits within 50 ms of a label in 87.5% of clips. |
| Sources | Field recordings only. |
| Use | CC BY 4.0 (Zenodo 10.5281/zenodo.7004819; paper: Data in Brief 48, 109091, 2023). Results, figures and short examples may be published with attribution. |
| When | Available. |

### 2.2 Certus Caliber Classification Gunshot Dataset, C3GD v2 (`source = c3gd`)

| Question | Answer |
|---|---|
| Classes | 18 calibers, 45 firearms (48 platform ids; some guns appear under two ids). No non-gunshot class. |
| Collection | 4 sessions: Ohio farm 2025-07-30, New Jersey farm 2025-11-13, New York gravel pit 2025-12-21, New Jersey farm (same site) 2026-04-07. Each gun model appears in one session (Sig P365 in two). |
| Devices | 12 microphones (Tascam DR-05XP, DJI, AKG, Samson, Galaxy Tab, Pixel 7, ...). **Yes**, the same shot shares `FileId` + `ClipId` across mics. |
| Format | WAV, 48 kHz, 16-bit mono (v2 fixed a sample-rate issue). Clip length depends on session: 0.459 s (NJ farm, NY gravel pit) or 1.0 s (others). The shot sits about 0.1 s into each clip. |
| Labels | Collectors, cross-checked against an audio call-out of the gun; caliber, platform, cartridge and mic per clip in `metadata.csv`. No shot times beyond the 0.1 s alignment. |
| Sources | Field recordings only (open fields, little reverberation). |
| Use | CC BY 4.0 (Zenodo 10.5281/zenodo.22286299; md5 of our zip matched). Funded by AFRL (FA8750-24-C-B082). |
| When | Available. |

Known issues: metadata typos (Winchester 1400 listed as 12 ga, really 20 ga; KS7 ammo listed as 20 ga); 1,769 clips
fail our 5% clipping rule (Tascam mics close to the muzzle); the dataset's own `split` column is a random per-clip
split and leaks sessions, so it is not used.

### 2.3 Cadre Research Labs gunshot dataset (`source = cadre`)

| Question | Answer |
|---|---|
| Classes | 21 firearms in the full set; 15 downloaded. Used here: Glock 19 (2 guns), Sig P225, M16A1, S&W 10-8. No non-gunshot class. |
| Collection | One test-firing campaign (NIJ grant 2016-DN-BX-0183, 2017–18), counted as **1 session**. Recording dates per run are not given. |
| Devices | iPhone (44.1 kHz mono), Samsung (48 kHz stereo), Zoom recorder (96 kHz stereo), each at positions A and B. **Yes**, each shot is on up to 6 devices; the run number (`NNN` in `<rec>_<NNN><pos>_S<shot>.wav`) groups them. |
| Format | WAV, about 2 s per shot, uncompressed. |
| Labels | Gun identified by the run number (`NNN mod 20`); no shot times. |
| Sources | Field recordings. |
| Use | **Not confirmed**: downloaded after registration; terms not found publicly. Used for training and aggregated results; no Cadre audio in published examples. See §5. |
| When | Available (MP40 has no iPhone recordings). |

### 2.4 Background sources

| Source | What | Sessions (`session_id`) | Format | Licence |
|---|---|---|---|---|
| MAD (`source = unfiltered`, from `GUN_DATASET/archive`) | Military Audio Dataset (Kim, Yoon & Jung, *Scientific Data* 2024): clips from YouTube military videos. Class 1 (live fire) dropped; classes 0–3 kept only if free of impulsive events | one per video (`mad-*`) | 16 kHz mono WAV | CC BY-SA 4.0 (secondary source, see §5) |
| UrbanSound8K (`us8k`) | 10 urban classes from Freesound; `gun_shot` dropped | one per Freesound source recording (`us8k-<fsID>`) | 8–48 kHz, mono/stereo, ≤4 s | CC BY-NC 3.0 |
| ESC-50 (`esc50`) | 50 environmental classes incl. fireworks, door knock, clapping, glass breaking | one per Freesound source (`esc50-<src_file>`) | 44.1 kHz mono, 5 s (often zero-padded) | CC BY-NC 3.0 |
| Quiet stretches (`edge-quiet`, `cadre-quiet`) | Audio before the first shot of edge and Cadre clips, so some background is on the gun microphones | the gun clip's session | as the gun clip | as the gun source |

MAD comes from the internet, so it is used only as background, never as gunfire.

### 2.5 Sources removed

The Drive folder `Unfiltered/GUN_DATASET` (believed to be the data handed over by Sir; to confirm in §5) held
several sub-collections. Only its MAD part is used.

| Sub-collection | Why it was removed |
|---|---|
| `original_dataset/` | 697 clips of **video-game audio** (PUBG / "Enemy Spotted" naming `<gun>_<dist>m_<direction>_<n>`), 269 re-encoded copies of Cadre clips (the 3 in `glock/` were really Glock 21 .45 ACP), 67 re-encoded copies of Kabealo clips, copies of `Gun_Shot_DataSet` |
| `Gun_Shot_DataSet/` | Unknown origin; the same audio was filed under both M4 and M16 (200 clips) |
| `rifel audios/`, `Gunshot-data/` | Unknown guns and sessions; `rifel audios/winchester-rifle` was game audio |
| `Sniper Data (06-11-2026)/` | 25 of its kar98k / M24 clips are copies of the game audio |
| `GUN/` | A flat dump of files that exist elsewhere |

Every removed file is listed with its reason in `Dataset/trashed_files.csv` and `Dataset/manifest.csv`
(statuses `excluded_game`, `duplicate`, `label_conflict`).

---

## 3. Filtering

Gun clips (`sw/dataset/filter_dataset.py`) are rejected if any check fails: corrupted, under 0.1 s, peak below
−40 dBFS, no discernible event (loudest 4 ms less than 6 dB above the 10th-percentile frame), weak and quiet,
more than 5% of audible samples at full scale, sample rate under 8 kHz, or a dead channel.

Background clips (`sw/dataset/build_final_dataset.py`): corrupted, under 0.5 s, peak below −50 dBFS, more than 5%
clipping, sample rate under 8 kHz, dead channel. MAD and the quiet stretches must also be free of impulsive events
(kurtosis ≤ 6, no transient), because they may hide gunfire; UrbanSound8K and ESC-50 skip that check because their
impulsive sounds are the hard negatives the model needs.

Quiet stretches end 50 ms before the first shot (edge: labelled time; Cadre: first sample within 30 dB of the peak,
only if that is within 100 ms of the peak) and must be at least 0.5 s. C3GD clips start 0.1 s before the shot and
give none.

---

## 4. Findings to act on before training

1. **Clip length differs by source** (C3GD 0.46 / 1.0 s, Cadre ~2 s, edge 1.4–13 s, ESC-50 5 s, UrbanSound8K up to
   4 s). Every clip must be cut to the same window (guns aligned to the shot onset), or length gives away the session.
2. **Background is about 16x the smallest gun class** (14,507 vs 715). Sample background per epoch and use class
   weights; report macro-F1 and per-class recall.
3. **55 background clips are below 16 kHz** (8–11 kHz). After resampling they have no content above 4–5.5 kHz, while
   every gun clip reaches 8 kHz. Drop them or low-pass every clip to the same band.
4. **Resampling can overshoot full scale** on clipped C3GD clips (a peak of 1.16 seen). Clamp to [−1, 1] after
   resampling.
5. **Many ESC-50 clips are zero-padded to 5 s.** Random background windows must contain signal.
6. **12 ga has no Cadre clips** and Cadre dominates no class, but sources are not evenly spread across classes;
   report results per source as well as per class.

---

## 5. Provenance, permissions and scope

Answered by the team on 2026-10-25 from the final dataset itself (`Dataset-Final/`), instead of asking Sir: every
clip in it comes from a published dataset with known origin, so the answers follow from those datasets' records
and licences.

| # | Question | Answer |
|---|---|---|
| 1 | Where did the data come from? | Only public, documented datasets: Kabealo (edge), C3GD and Cadre for gunfire; MAD, UrbanSound8K, ESC-50 and quiet stretches of edge/Cadre for background. Of the Drive folder `Unfiltered/GUN_DATASET`, only its MAD sub-folder is used (as background). Everything else in it is excluded (§2.5), so its unknown origin does not affect the final dataset. |
| 2 | Can results, figures and short audio examples be published? | **Results and figures: yes**, with attribution to every dataset (all allow academic use). **Audio examples:** use only Kabealo or C3GD clips (CC BY 4.0) for gunshots and ESC-50 / UrbanSound8K clips (CC BY-NC 3.0, non-commercial) for background. Do not publish Cadre audio (terms not confirmed) or MAD audio (taken from YouTube videos). |
| 3 | Can the data go onto Colab or Google Drive? | **Yes, in a private Drive or Colab session** for the team's own processing: none of the licences restricts where licensed users keep their copy. Do not share the folder publicly or re-publish it; Cadre was obtained through registration and its redistribution terms are not confirmed. |
| 4 | Which classes? | Caliber-based: `pistol_9mm`, `rifle_556_223`, `revolver_38spl`, `shotgun_12ga`, `background`. Per-model classes are impossible to split honestly, because each gun model appears in only one or two sessions; each caliber class pools guns from 4–6 sessions (§1). |
| 5 | Are there more recordings? | No own gunfire recordings. Recommended addition, not a requirement: about 10 minutes of ordinary room and street sound recorded on the demo microphone, as extra background for the live demo. |

Licences used above:

| Dataset | Licence | Status |
|---|---|---|
| Kabealo (edge) | CC BY 4.0 | confirmed (Zenodo) |
| C3GD v2 | CC BY 4.0 | confirmed (Zenodo record 22286299) |
| UrbanSound8K | CC BY-NC 3.0 | dataset's published licence |
| ESC-50 | CC BY-NC 3.0 | dataset's published licence |
| MAD (Kim, Yoon & Jung, *Scientific Data* 2024) | CC BY-SA 4.0 | reported by a secondary source; confirm on the Kaggle page before quoting |
| Cadre Research Labs | unknown | accepted at registration; check the terms on the download page before publishing anything that uses Cadre audio directly |

---

## 6. Item 4 status

| Download | Status |
|---|---|
| UrbanSound8K | Done: `Dataset/urbansound8k/`, all 8,732 files checked against the zip's CRC32 |
| ESC-50 | Done: `Dataset/esc50/`, 2,000 original clips |
| Kabealo | Done: `Dataset/edge-collected-gunshot-audio/` |
| Google Speech Commands (fallback workload) | **Skipped (team decision, 2026-10-25).** The fallback is only needed if the gunshot data cannot support honest classes; every firearm class has 4 or more sessions (§1). Download later if that changes. |

---

## 7. Where the data lives

No audio and no manifest is kept in git. The final dataset (`Dataset-Final/`: `audio/`, `manifest.csv`,
`audit_counts.csv`, `class_summary.csv`, `rejected.csv`) is shared through the team's private Google Drive
folder (link: ____). It must not be shared publicly (§5, question 3). The raw sources (`Dataset/`) stay on the
machine that built it; the scripts below rebuild everything from them.

## 8. Rebuilding

```bash
.venv-audio/bin/python "Live Gunshot Classification via Digital CIM Architecture/sw/dataset/build_dataset.py"
.venv-audio/bin/python "Live Gunshot Classification via Digital CIM Architecture/sw/dataset/filter_dataset.py"
.venv-audio/bin/python "Live Gunshot Classification via Digital CIM Architecture/sw/dataset/build_final_dataset.py" --us8k Dataset/urbansound8k --esc50 Dataset/esc50 --force
.venv-audio/bin/python "Live Gunshot Classification via Digital CIM Architecture/sw/data/audit.py"
```

Downstream code reads clips only through `sw/data/manifest.py` (`load_manifest`, `load_audio`), never raw folder
names. Use another dataset by producing a `manifest.csv` with the same columns.

For splits: keep each `group` (one shot or one recording) on one side; hold out whole `session_id` values for the
session-disjoint test; `device` allows a second, harder device-disjoint test.
