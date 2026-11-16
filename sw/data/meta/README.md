# Edge shot labels

`gunshot-audio-all-metadata.csv` and `gunshot-audio-labels-only.csv` are the
label files of the edge-collected gunshot audio dataset, copied unchanged:

Kabealo et al., Data in Brief 48, 109091, 2023. Zenodo 10.5281/zenodo.7004819.
Licence: CC BY 4.0 (see `docs/dataset_audit.md` section 2.1).

`sw/frontend/cut_windows.py` and `check_windows.py` read the shot times
(`gunshot_location_in_seconds`) from `gunshot-audio-all-metadata.csv`
(sha256 `219e242ec89b20395fc94ab519387f143b7f748939c7e72aed8a0a32fc804d15`,
recorded in `results/step16/detector_run.json`). The audio itself is not in git.
