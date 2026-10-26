#!/usr/bin/env python3
"""Combine the two raw gunshot datasets into one folder, then sort by gun.

Sources: edge-collected-gunshot-audio, Sorted-C3GD (already sorted per gun, with
session/device metadata), cadre-gunshot-audio (Cadre Research Labs test firings,
one folder per gun and recorder) and the Unfiltered GUN_DATASET.

original_dataset/ contains re-encoded copies of some Cadre Samsung clips
(``..._SA_214B_S03.wav``). They are not byte-identical, so they are matched to
the Cadre original by recorder/run/position/shot id and dropped as duplicates.

original_dataset also contains video-game audio (PUBG, the "Enemy Spotted"/BGG
dataset), recognisable by its ``<gun>_<distance>m_<direction>_<n>`` names. It is
not real gunfire, so every file with the same audio as one of these clips, under
any name or folder, is dropped as ``excluded_game``.

Step 1  Combined/            every audio file from both datasets, flat, with a
                             provenance-preserving unique name. Byte-identical
                             duplicates are dropped (first copy wins).
Step 2  Sorted/<gun>/        the same files, grouped by gun.

Audio that appears byte-identical under two *different* guns (e.g. the "M4" and
"M16" folders of Gun_Shot_DataSet are the same recordings) has an untrustworthy
label, so every copy is dropped and recorded as ``label_conflict``.

Raw data is never modified or moved. Outputs are hard links to the originals
(zero extra disk space) and fall back to real copies if hard links are not
possible (different filesystem). Use --copy to force real copies.

The MAD dataset (``archive/MAD_dataset``: battlefield ambience, soldiers, tanks,
helicopters, jets; labels 0-6, no gun identity) is added as the no-gun
class ``background-nogun``. Its original MAD label and video folder are kept in
the manifest (``source_label`` and ``group``). filter_dataset.py applies
different rules to this class (it must NOT contain gunfire).

A manifest (Dataset/manifest.csv) records, for every input file, where it came
from, which gun it was assigned, its leakage ``group`` and whether it was kept
or dropped as a duplicate.

Usage:
    python build_dataset.py [--dataset-root PATH] [--copy] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Empty placeholder folder; the real MAD data is under archive/.
EXCLUDED_TOP_DIRS = {"MAD_dataset"}

BACKGROUND = "background-nogun"

# When two files are byte-identical, the one from the higher-priority source
# (lower number) is kept. GUN/ is a flat dump of files that also live elsewhere.
SOURCE_PRIORITY = {
    "edge": 0,
    "cadre": 0,
    "original_dataset": 1,
    "c3gd": 1,
    "Gun_Shot_DataSet": 2,
    "rifel audios": 3,
    "Sniper Data": 4,
    "Gunshot-data": 5,
    "MAD": 6,
    "GUN": 9,
}

# Raw folder / prefix name (after kebab-casing) -> canonical gun folder.
GUN_ALIASES = {
    # edge-collected dataset
    "38s&ws-dot38-caliber": "sw-38-caliber",
    "38sws-dot38-caliber": "sw-38-caliber",
    "ruger-ar-556-dot223-caliber": "ruger-ar-556",
    "remington-870-12-gauge": "remington-870",
    "glock-17-9mm-caliber": "glock-17",
    # same gun, different spelling across datasets
    "glock": "glock-17",  # original_dataset/glock/* are edge-style recordings
    "mg-42": "mg42",
    "mg42-lmg": "mg42",
    "mp5-smg": "mp5",
    "m249-lmg": "m249",
    "norinco-sks": "sks-rifle",  # C3GD name for the same rifle type
    "imi-desert-eagle": "desert-eagle",
    "zastava-m92": "zastava-m92",
    "winchester-rifle": "winchester-rifle",
    "m24-sws": "m24-sws",
    "kar98k": "kar98k",
    "g3": "g3",
}


# Cadre gun index -> canonical gun folder. A clip's run number NNN (in
# <rec>_<NNN><pos>_S<shot>.wav) cycles through the guns, so NNN % 20 is the gun.
# Names match the folders the original_dataset copies already used.
CADRE_GUNS = {
    1: "high-standard-22lr",          # High Standard Sport King, .22 LR pistol
    2: "sw-34-1-22lr-revolver",       # S&W 34-1, .22 LR revolver
    3: "ruger-22lr",                  # Ruger 10/22, .22 LR carbine
    4: "remington-22lr",              # Remington 33, .22 LR bolt-action rifle
    5: "lorcin-380acp",               # Lorcin L380, .380 ACP pistol
    6: "sw-10-8-38spl-revolver",      # S&W 10-8, .38 Special revolver
    8: "glock-19",                    # Glock 19, 9 mm (gun 1 of 2)
    9: "glock-19",                    # Glock 19, 9 mm (gun 2 of 2)
    10: "sig-p225",                   # Sig P225, 9 mm
    11: "mp-40-40-sw-pistol",         # S&W M&P 40, .40 S&W (gun 1 of 2)
    12: "mp-40-40-sw-pistol",         # S&W M&P 40, .40 S&W (gun 2 of 2)
    13: "hk-usp-compact-40-sw-pistol",  # HK USP Compact, .40 S&W
    14: "glock-21",                   # Glock 21, .45 ACP
    15: "colt-1911",                  # Colt 1911, .45 ACP
    16: "kimber-45acp",               # Kimber Tactical Custom, .45 ACP
    17: "m16",                        # M16A1, 5.56
    18: "wasr-10-63-ak47",            # WASR-10/63 (AK-47 pattern), 7.62x39
}
CADRE_RECORDERS = {"IP": "iPhone", "SA": "Samsung", "ZM": "Zoom"}
CADRE_RE = re.compile(r"(?:^|_)(IP|SA|ZM)_(\d{3})([A-Z])_S(\d+)$")

# PUBG game-audio clip names: deagle_0m_center_1776, sks_100m_back_12
GAME_RE = re.compile(r"_\d+m_(?:center|left|right|front|back)_\d+$")


def cadre_fields(stem: str) -> dict | None:
    """Parse a Cadre clip name: shot key, gun, leakage group and device."""
    m = CADRE_RE.search(stem)
    if not m:
        return None
    rec, run, pos, shot = m.groups()
    return {"key": f"{rec}_{run}{pos}_S{shot}", "gun": CADRE_GUNS.get(int(run) % 20, ""),
            "group": f"cadre-{run}", "device": f"{CADRE_RECORDERS[rec]}-{pos}"}


def kebab(name: str) -> str:
    return re.sub(r"[\s_]+", "-", name.strip().lower())


def canon_gun(raw: str) -> str:
    k = kebab(raw)
    return GUN_ALIASES.get(k, k)


@dataclass
class Item:
    path: Path
    source: str          # edge | unfiltered
    rel: str             # path relative to the dataset root it came from
    top: str             # priority key
    gun: str
    group: str
    source_label: str = ""
    session: str = ""    # recording session, where the dataset says (edge, C3GD)
    device: str = ""     # recorder / microphone, where the dataset says
    xkey: str = ""       # id of the same recording in another source (Cadre shot key)


def leakage_group(stem: str) -> str:
    """Id shared by recordings that are the same physical event / same take.

    Splitting train/val/test by this id avoids near-duplicate leakage.
    """
    # edge + original glock/remington style: <uuid>_chanN_vM  /  <uuid>_vM
    m = re.match(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", stem)
    if m:
        return m.group(1)
    # Cadre: one run (NNN) is one string of shots recorded by every device
    c = cadre_fields(stem)
    if c:
        return c["group"]
    # trimmed rifle parts: A105_part3 -> A105
    m = re.match(r"(A\d+)(?:_part\d+)?$", stem)
    if m:
        return m.group(1)
    return stem


def classify_unfiltered(rel: Path) -> tuple[str, str] | None:
    """Return (priority_key, gun) for a path relative to GUN_DATASET, or None to skip."""
    parts = rel.parts
    top = parts[0]
    stem = rel.stem
    if top in EXCLUDED_TOP_DIRS:
        return None
    if top == "archive":
        return "MAD", BACKGROUND
    if top == "original_dataset" or top == "Gun_Shot_DataSet":
        return top, canon_gun(parts[1])
    if top == "rifel audios":
        sub = parts[1]
        if sub in ("winchester-rifle", "winchester-m14"):
            return top, canon_gun(sub)
        if re.match(r"A\d+", stem):
            return top, "_unknown-rifle-a-series"
        return top, "_unknown-local-recordings"
    if top == "Sniper Data (06-11-2026)":
        if len(parts) > 3 and parts[2] in ("kar98k", "m24-sws"):
            return "Sniper Data", canon_gun(parts[2])
        return "Sniper Data", "sniper-unspecified"
    if top == "Gunshot-data":
        sub = parts[1] if len(parts) > 2 else ""
        if sub == "Gunshot_hunting":
            return top, "_unknown-hunting"
        label = sub or re.match(r"[A-Za-z]+\d*", stem).group(0).split("-")[0]
        label = re.sub(r"\d+$", "", label) if label != "G3" else "G3"
        if label == "S" or label.lower() == "sniper":
            return top, "sniper-unspecified"
        if label.upper() == "SMG":
            return top, "smg-unspecified"
        if label.upper() == "G3":
            return top, "g3"
        return top, f"_unknown-{kebab(label)}"
    if top == "GUN":
        m = re.match(r"([a-z0-9]+(?:-[a-z0-9]+)*)_\d{3}_", stem)
        if m:
            return top, canon_gun(m.group(1))
        if stem.lower() == "9mm":
            return top, "9mm-unspecified"
        return top, "_unknown-local-recordings"
    return top, "_unknown-" + kebab(top)


def load_mad_labels(mad_root: Path) -> dict[str, str]:
    """'training/398/0.wav' -> MAD class label ('0'..'6'), from training.csv / test.csv."""
    out: dict[str, str] = {}
    for name in ("training.csv", "test.csv"):
        f = mad_root / name
        if f.exists():
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    out[r["path"]] = r["label"]
    return out


def load_edge_meta(dataset_root: Path) -> dict[str, tuple[str, str]]:
    """edge filename stem -> (session, device) from gunshot-audio-all-metadata.csv."""
    f = dataset_root / "edge-collected-gunshot-audio" / "gunshot-audio-all-metadata.csv"
    out: dict[str, tuple[str, str]] = {}
    if f.exists():
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                r = {k.strip(): v for k, v in r.items() if k}
                out[r["filename"]] = (f"edge-{r['label']}-s{r['recording_session'].strip()}",
                                      r.get("device_model", ""))
    return out


def discover_c3gd(dataset_root: Path) -> list[Item]:
    """Dataset/Sorted-C3GD/<gun>/*.wav, with c3gd_sorted.csv giving session and device."""
    root = dataset_root / "Sorted-C3GD"
    items: list[Item] = []
    if not root.is_dir():
        return items
    meta: dict[str, dict] = {}
    f = root / "c3gd_sorted.csv"
    if f.exists():
        with open(f, newline="") as fh:
            meta = {r["path"]: r for r in csv.DictReader(fh)}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
            continue
        rel = p.relative_to(root)
        m = meta.get(rel.as_posix(), {})
        session = m.get("session_id", "")
        it = Item(p, "c3gd", rel.as_posix(), "c3gd", canon_gun(rel.parts[0]),
                  session or leakage_group(p.stem), session=session, device=m.get("device", ""))
        items.append(it)
    return items


def discover_cadre(dataset_root: Path) -> list[Item]:
    """Dataset/cadre-gunshot-audio/<Gun>[_n]_<Recorder>/<rec>_<NNN><pos>_S<shot>.wav.

    The whole collection is one test-firing campaign, so it is one session.
    """
    root = dataset_root / "cadre-gunshot-audio"
    items: list[Item] = []
    if not root.is_dir():
        return items
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS or not p.is_file() or "__MACOSX" in p.parts:
            continue
        c = cadre_fields(p.stem)
        if c is None or not c["gun"]:
            print(f"WARNING: unrecognised Cadre file {p}", file=sys.stderr)
            continue
        items.append(Item(p, "cadre", p.relative_to(root).as_posix(), "cadre", c["gun"], c["group"],
                          session="cadre", device=c["device"], xkey=c["key"]))
    return items


def discover(dataset_root: Path) -> list[Item]:
    items: list[Item] = []
    edge_meta = load_edge_meta(dataset_root)

    edge_root = dataset_root / "edge-collected-gunshot-audio" / "edge-collected-gunshot-audio"
    if edge_root.is_dir():
        for p in sorted(edge_root.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS and p.is_file():
                rel = p.relative_to(edge_root)
                sess, dev = edge_meta.get(p.stem, ("", ""))
                items.append(Item(p, "edge", rel.as_posix(), "edge", canon_gun(rel.parts[0]),
                                  leakage_group(p.stem), session=sess, device=dev))
    else:
        print(f"WARNING: edge dataset not found at {edge_root}", file=sys.stderr)

    items.extend(discover_c3gd(dataset_root))
    items.extend(discover_cadre(dataset_root))

    unf_roots = sorted(dataset_root.glob("Unfiltered-*/Unfiltered/GUN_DATASET"))
    if not unf_roots:
        print("WARNING: Unfiltered GUN_DATASET not found", file=sys.stderr)
    for unf in unf_roots:
        mad_labels = load_mad_labels(unf / "archive" / "MAD_dataset")
        for p in sorted(unf.rglob("*")):
            if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
                continue
            rel = p.relative_to(unf)
            c = classify_unfiltered(rel)
            if c is None:
                continue
            top, gun = c
            it = Item(p, "unfiltered", rel.as_posix(), top, gun, leakage_group(p.stem))
            c = cadre_fields(p.stem) if top == "original_dataset" else None
            if c:
                # Re-encoded Cadre clip: the run number gives the true gun (the
                # "glock" folder holds Glock 21 .45 ACP clips, not Glock 17).
                it.gun = c["gun"] or gun
                it.session, it.device, it.xkey = "cadre", c["device"], c["key"]
            if top == "MAD":
                # archive/MAD_dataset/<split>/<video-folder>/<n>.wav ; all clips of one
                # video folder share a group so splits don't leak.
                it.group = f"mad-{rel.parts[-2]}"
                it.source_label = mad_labels.get(Path(*rel.parts[2:]).as_posix(), "")
            items.append(it)
    return items


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_name(item: Item) -> str:
    flat = re.sub(r"[^\w.\-()&# ]", "_", item.rel.replace("/", "__"))
    name = f"{item.source}__{flat}"
    if len(name.encode()) > 200:
        stem, ext = os.path.splitext(name)
        digest = hashlib.sha1(item.rel.encode()).hexdigest()[:10]
        name = f"{stem[:150]}__{digest}{ext}"
    return name


def place(src: Path, dst: Path, force_copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force_copy:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def safe_reset(d: Path, dataset_root: Path) -> None:
    """Delete a derived output folder (only ever Combined/ or Sorted/ in the dataset root)."""
    if d.parent.resolve() != dataset_root.resolve() or d.name not in {"Combined", "Sorted"}:
        raise RuntimeError(f"refusing to delete {d}")
    if d.exists():
        shutil.rmtree(d)


def main() -> int:
    default_root = Path(__file__).resolve().parents[3] / "Dataset"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, default=default_root)
    ap.add_argument("--copy", action="store_true", help="copy files instead of hard-linking")
    ap.add_argument("--dry-run", action="store_true", help="scan and report, write nothing")
    args = ap.parse_args()
    root: Path = args.dataset_root.resolve()

    items = discover(root)
    items.sort(key=lambda i: (SOURCE_PRIORITY.get(i.top, 8), i.source, i.rel))
    print(f"Discovered {len(items)} audio files.")

    # Hash everything, then resolve duplicates per content hash.
    hashed = [(it, sha1_of(it.path)) for it in items]
    by_hash: dict[str, list[Item]] = defaultdict(list)
    for it, digest in hashed:
        by_hash[digest].append(it)

    def is_specific(gun: str) -> bool:
        return not gun.startswith("_") and not gun.endswith("-unspecified")

    # Cadre clips re-encoded elsewhere differ in bytes; match them by shot key.
    cadre_keys = {it.xkey: it.rel for it in items if it.source == "cadre"}
    # Game audio, and every byte-identical copy of it under another name.
    game_hashes = {digest for it, digest in hashed if GAME_RE.search(it.path.stem)}

    rows = []
    kept: list[tuple[Item, str]] = []
    used_names: set[str] = set()
    conflicts = 0
    reencoded = 0
    games = 0
    for it, digest in hashed:
        group = by_hash[digest]          # already priority-ordered
        size = it.path.stat().st_size
        if digest in game_hashes:
            rows.append([None, it.source, it.rel, it.gun, it.group, digest, size,
                         "excluded_game", "", it.source_label, it.session, it.device])
            games += 1
            continue
        if it.source != "cadre" and it.xkey in cadre_keys:
            rows.append([None, it.source, it.rel, it.gun, it.group, digest, size,
                         "duplicate", "cadre:" + cadre_keys[it.xkey], it.source_label, it.session, it.device])
            reencoded += 1
            continue
        specific = {g.gun for g in group if is_specific(g.gun)}
        if len(specific) > 1:
            # Same audio filed under different guns: label cannot be trusted.
            rows.append([None, it.source, it.rel, it.gun, it.group, digest, size,
                         "label_conflict", "|".join(sorted(specific)), it.source_label, it.session, it.device])
            conflicts += 1
            continue
        if group[0] is not it:
            rows.append([None, it.source, it.rel, it.gun, it.group, digest, size,
                         "duplicate", group[0].rel, it.source_label, it.session, it.device])
            continue
        if len(specific) == 1:           # prefer the specific label over a vague one
            it.gun = next(iter(specific))
        name = combined_name(it)
        if name in used_names:           # extremely unlikely, but never overwrite
            name = f"{Path(name).stem}__{digest[:8]}{Path(name).suffix}"
        used_names.add(name)
        kept.append((it, name))
        rows.append([name, it.source, it.rel, it.gun, it.group, digest, size, "kept", "", it.source_label, it.session, it.device])

    n_dup = len(items) - len(kept) - conflicts - reencoded - games
    print(f"Unique files kept: {len(kept)}")
    print(f"Byte-identical duplicates dropped: {n_dup}")
    print(f"Re-encoded copies of Cadre clips dropped: {reencoded}")
    print(f"Video-game (PUBG) clips and their copies dropped: {games}")
    print(f"Dropped for label conflict (same audio filed under different guns): {conflicts}")

    by_gun = Counter(it.gun for it, _ in kept)
    print("\nFiles per gun folder:")
    for gun, n in sorted(by_gun.items()):
        print(f"  {n:5d}  {gun}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    combined, sorted_dir = root / "Combined", root / "Sorted"
    safe_reset(combined, root)
    safe_reset(sorted_dir, root)

    modes = Counter()
    for it, name in kept:
        modes[place(it.path, combined / name, args.copy)] += 1
        modes[place(combined / name, sorted_dir / it.gun / name, args.copy)] += 1
    print(f"\nWrote Combined/ and Sorted/ ({dict(modes)})")

    with open(root / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combined_name", "source", "orig_relpath", "gun", "group", "sha1", "bytes",
                    "status", "duplicate_of", "source_label", "session", "device"])
        w.writerows(rows)
    print(f"Wrote {root / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
