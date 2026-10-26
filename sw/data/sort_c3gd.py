"""Sort the C3GD clips into one folder per gun, using the dataset's own metadata.

Usage:
    python sw/data/sort_c3gd.py <C3GD root> <output dir>

Each clip is hard-linked (no extra disk space) to <output dir>/<gun>/c3gd__<filename>.
The source files are never modified. <output dir>/c3gd_sorted.csv lists every clip
with its gun, caliber, form factor, session, microphone and original-recording id.
"""

import os
import sys

import pandas as pd

# platform_id -> folder name. Matches the slug style of Dataset/Sorted, so the
# guns that also appear in other sources (glock-17, remington-870) share a name.
RENAME = {
    "glock_m17": "glock-17",
    "remington_870_12ga": "remington-870",
    "tauros_66": "taurus-66-38spl-revolver",
    "tauros_856": "taurus-856-38spl-revolver",
    "springfieldd_xd": "springfield-xd-40sw",
    # Platform ids that are missing from metadata/platforms.csv: name them by caliber.
    "m1911": "m1911-22lr",
    "bergara": "bergara-22lr",
    "ar_15": "ar-15-300blk",
    "sig_p226_legion": "sig-p226-legion",
    "savage_magpul_hunter_110": "savage-110-6.5cm",
}


def slug(platform_id):
    return RENAME.get(platform_id, platform_id.lower().replace("_", "-"))


def main(root, out):
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    plat = pd.read_csv(os.path.join(root, "metadata", "platforms.csv"))
    meta = meta.merge(plat[["platform_id", "manufacturer", "model", "form_factor"]],
                      on="platform_id", how="left")
    meta["gun"] = meta["platform_id"].map(slug)

    rows, missing = [], 0
    for r in meta.itertuples():
        src = os.path.join(root, "data", r.filename)
        if not os.path.exists(src):
            missing += 1
            continue
        dst_dir = os.path.join(out, r.gun)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "c3gd__" + r.filename)
        if not os.path.exists(dst):
            os.link(src, dst)
        rows.append({
            "path": os.path.relpath(dst, out),
            "gun": r.gun,
            "manufacturer": r.manufacturer,
            "model": r.model,
            "form_factor": r.form_factor,
            "caliber": r.class_name,
            "session_id": "c3gd_" + r.event_id,
            "device": r.mic_id,
            "file_id": r.file_id,
            "clip_id": r.clip_id,
            "duration_s": r.duration_secs,
            "source": "c3gd",
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "c3gd_sorted.csv"), index=False)
    summary = (df.groupby(["caliber", "gun"])
                 .agg(clips=("path", "size"),
                      sessions=("session_id", "nunique"),
                      recordings=("file_id", "nunique"),
                      devices=("device", "nunique")))
    print(summary.to_string())
    print(f"\n{len(df)} clips sorted into {df.gun.nunique()} gun folders; {missing} missing files")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
