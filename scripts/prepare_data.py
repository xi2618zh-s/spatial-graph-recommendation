"""Join SNAP raw Gowalla check-ins onto the official LightGCN ID space.

Inputs
------
data/gowalla/user_list.txt, item_list.txt   (org_id -> remap_id maps, committed)
data/gowalla/train.txt                       (official training split, committed)
data/raw/loc-gowalla_totalCheckins.txt.gz    (SNAP raw: user, time, lat, lon, loc_id)

Outputs (data/processed/)
-------------------------
poi_coords.csv        remap_item_id, lat, lon        -> for Spatial-LightGCN edges
train_sequences.pkl   {remap_user_id: [item ids sorted by check-in time]}
                                                      -> for SASRec
prepare_report.json   coverage statistics for the join (sanity audit)

Usage
-----
python scripts/prepare_data.py
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOWALLA = ROOT / "data" / "gowalla"
RAW = ROOT / "data" / "raw" / "loc-gowalla_totalCheckins.txt.gz"
OUT = ROOT / "data" / "processed"


def load_id_map(path: Path) -> dict:
    df = pd.read_csv(path, sep=" ")
    return dict(zip(df["org_id"].astype(int), df["remap_id"].astype(int)))


def load_official_train(path: Path) -> dict:
    """train.txt: each line = user_id item_id item_id ... (adjacency list)."""
    train = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            train[int(parts[0])] = set(int(x) for x in parts[1:] if x)
    return train


def main() -> None:
    if not RAW.exists():
        sys.exit(
            f"Missing raw file: {RAW}\n"
            "Download loc-gowalla_totalCheckins.txt.gz from "
            "https://snap.stanford.edu/data/loc-gowalla.html and place it there."
        )
    OUT.mkdir(parents=True, exist_ok=True)

    user_map = load_id_map(GOWALLA / "user_list.txt")
    item_map = load_id_map(GOWALLA / "item_list.txt")
    train = load_official_train(GOWALLA / "train.txt")
    print(f"Official split: {len(user_map)} users, {len(item_map)} items")

    # SNAP raw: tab-separated [user] [check-in time ISO] [lat] [lon] [location id]
    print("Reading SNAP raw check-ins (~6.4M rows)...")
    with gzip.open(RAW, "rt") as f:
        raw = pd.read_csv(
            f,
            sep="\t",
            header=None,
            names=["org_user", "time", "lat", "lon", "org_item"],
        )
    raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["time"])

    # Map to official IDs; keep only rows present in the benchmark ID space
    raw["u"] = raw["org_user"].map(user_map)
    raw["i"] = raw["org_item"].map(item_map)
    matched = raw.dropna(subset=["u", "i"]).astype({"u": int, "i": int})

    # --- Output 1: POI coordinates (median over check-ins, robust to GPS noise)
    coords = (
        matched.groupby("i")[["lat", "lon"]].median().reindex(range(len(item_map)))
    )
    n_missing = int(coords["lat"].isna().sum())
    coords.rename_axis("item_id").reset_index().to_csv(
        OUT / "poi_coords.csv", index=False
    )

    # --- Output 2: time-ordered TRAINING sequences per user
    # Only interactions in the official train split (never test) enter sequences,
    # so the sequential model sees no leaked test items.
    matched_sorted = matched.sort_values("time")
    sequences = {}
    for u, grp in matched_sorted.groupby("u"):
        allowed = train.get(u, set())
        seq, seen = [], set()
        for it in grp["i"]:
            if it in allowed and it not in seen:  # first check-in defines order
                seq.append(it)
                seen.add(it)
        sequences[u] = seq
    with open(OUT / "train_sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)

    # --- Output 3: coverage audit
    train_pairs = sum(len(v) for v in train.values())
    seq_pairs = sum(len(v) for v in sequences.values())
    report = {
        "raw_checkins": int(len(raw)),
        "matched_checkins": int(len(matched)),
        "items_total": len(item_map),
        "items_missing_coords": n_missing,
        "train_pairs_official": train_pairs,
        "train_pairs_with_timestamp": seq_pairs,
        "timestamp_coverage": round(seq_pairs / train_pairs, 4),
        "seq_len_median": float(np.median([len(v) for v in sequences.values()])),
    }
    with open(OUT / "prepare_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nDone. Artifacts in {OUT}")


if __name__ == "__main__":
    main()
