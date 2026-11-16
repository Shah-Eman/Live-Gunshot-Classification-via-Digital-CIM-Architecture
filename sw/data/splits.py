"""Read splits.json (written by make_splits.py). There is no fallback.

If splits.json is missing or was made from another manifest, every function
here raises: nothing ever trains or evaluates on "all clips" by accident.

Paths in splits.json are relative to the dataset folder (the folder holding
manifest.csv), exactly as in the manifest's path column.

    from splits import load_splits
    s = load_splits("sw/data/splits.json", manifest="Dataset-Final/manifest.csv")
    train, val = s.fold(0)            # sets of relative clip paths
    test = s.test                     # used once, in Step 1.11
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Splits:
    raw: dict

    @property
    def test(self) -> set[str]:
        return set(self.raw["test"]["paths"])

    @property
    def excluded(self) -> dict[str, str]:
        return {e["path"]: e["reason"] for e in self.raw["excluded"]}

    @property
    def n_folds(self) -> int:
        return len(self.raw["cv"]["folds"])

    @property
    def cv_pool(self) -> set[str]:
        """Every clip that cross-validation may use (all folds together)."""
        return set().union(*(set(f["val_paths"]) for f in self.raw["cv"]["folds"]))

    def fold(self, k: int) -> tuple[set[str], set[str]]:
        """(train, validation) clip paths of cross-validation fold k."""
        if not 0 <= k < self.n_folds:
            raise IndexError(f"fold {k} does not exist ({self.n_folds} folds)")
        val = set(self.raw["cv"]["folds"][k]["val_paths"])
        return self.cv_pool - val, val

    def device_split(self) -> tuple[set[str], set[str]]:
        test = set(self.raw["device_split"]["test_paths"])
        return self.cv_pool - test, test

    def random_split(self) -> tuple[set[str], set[str]]:
        test = set(self.raw["random_split"]["test_paths"])
        return self.cv_pool - test, test


def load_splits(path: str | Path, manifest: str | Path | None = None) -> Splits:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run sw/data/make_splits.py; there is no fallback to all clips.")
    raw = json.loads(path.read_text())
    if manifest is not None:
        got = file_sha256(manifest)
        if got != raw["manifest_sha256"]:
            raise ValueError(f"{path} was made from a different manifest (sha256 {raw['manifest_sha256'][:12]}..., "
                             f"this one is {got[:12]}...). Re-run make_splits.py.")
    return Splits(raw)
