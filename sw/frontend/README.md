# Front end: detector, windows, splits (runbook Step 1.6)

| File | What it is |
|---|---|
| `detector.py` | The integer event detector the RTL (Step 4.7) must match sample for sample |
| `test_detector.py` | `detect()` vs the per-sample loop (the RTL's form); exports an RTL test vector |
| `windows.py` | Window layout (D22), gun windows from detector triggers, per-epoch background sampler, mixing |
| `cut_windows.py` | Cuts every gun window with the detector; writes `gun_windows.npz` and the detector recall report |
| `check_windows.py` | Review-gate checks for the windows; writes 10 windows per class to `listen/` |
| `../data/splits.py` | Test set (D23), 5 CV folds, device-disjoint and random splits → `splits.json` |
| `../data/check_splits.py` | Proves there is no session or group leakage in `splits.json` |

Run once, in this order, on the full `Dataset-Final/` (with `manifest.csv` from `sw/data/audit.py`):

```bash
DS=../Dataset-Final
EDGE=../Dataset/edge-collected-gunshot-audio/gunshot-audio-all-metadata.csv   # Kabealo labels, checks edge triggers

python sw/frontend/test_detector.py --dataset-final $DS
python sw/data/splits.py       --dataset-final $DS           # once; record its sha256 in D23
python sw/data/check_splits.py --dataset-final $DS           # must print ALL CHECKS PASS
python sw/frontend/cut_windows.py   --dataset-final $DS --edge-meta $EDGE --jobs 8
python sw/frontend/check_windows.py --dataset-final $DS --edge-meta $EDGE   # then listen to windows/listen/
```

`detector_summary.csv` gives the recall per source, class and session (gate: ≥ 0.95, or record the miss rate).
Without `--edge-meta`, edge recall is not measured and edge triggers are kept unverified.

In training code:

```python
from manifest import load_manifest
from splits import load_splits
import windows as W

rows = load_manifest(DS / "manifest.csv")
s = load_splits(DS / "splits.json", rows)            # s["folds"][k]["train"/"val"], s["test"] (Step 1.11 only)
z = np.load(DS / "windows/gun_windows.npz")           # gun windows; z["row"] indexes rows
bg = W.BackgroundSampler(s["folds"][k]["train"], seed=epoch)   # new background crops every epoch
Xb, info = bg.epoch(n_background)                     # >= 20% from the gun microphones
x = W.to_model(windows)                               # int16 [n, 8000] -> float32 [n, 8000, 1]
```

Select gun windows of a split by row: `np.isin(z["row"], [rows.index(r) for r in split_rows])`.
