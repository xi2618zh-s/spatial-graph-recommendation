"""M6: build the point-in-time ranking dataset.

Pipeline: official-train sequences -> leave-last-out prefix/target per user
-> frozen recall model generates candidates (prefix-masked) -> candidate
Recall@K table -> layered negative sampling -> user/item/cross/context
features -> samples table + stats + audit sample.

Usage:
    python scripts/build_ranking_data.py --config configs/ranking_data.yaml

Outputs:
    data/processed/ranking_samples.csv           full samples table (gitignored, regenerable)
    experiments/results/candidate_recall.csv     Recall@{50,100,200} (committed)
    experiments/results/ranking_data_stats.json  quantiles/missing/source counts/hash (committed)
    experiments/results/ranking_data_audit_sample.csv  20 hand-auditable rows (committed)
    experiments/results/feature_schema.json      column name -> dtype/group (committed)
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.ranking_dataset import (
    build_prefix_targets, candidate_recall, generate_candidates,
    load_timestamped_sequences, train_val_user_split,
)
from src.data.sampling import build_geo_tree, build_popularity_pool, build_samples_for_batch
from src.features.item_features import ItemFeatureStore, compute_item_popularity
from src.models.registry import load_checkpoint
from src.utils.common import ROOT, load_config

RESULTS = ROOT / "experiments" / "results"
PROCESSED = ROOT / "data" / "processed"

FEATURE_GROUPS = {
    "user_": "user", "item_": "item", "cross_": "cross", "ctx_": "context",
}


def feature_group(col: str) -> str:
    for prefix, group in FEATURE_GROUPS.items():
        if col.startswith(prefix):
            return group
    return "identifier"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ranking_data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    seqs = load_timestamped_sequences(PROCESSED / "train_sequences_ts.pkl")
    prefix_targets = build_prefix_targets(seqs, min_history=cfg["min_history"])
    print(f"eligible users: {len(prefix_targets)} / {len(seqs)} "
          f"(min_history={cfg['min_history']})")

    _, data, _, score_fn = load_checkpoint(cfg["recall_run"], device="cpu")

    # --- Candidate Recall@K report (separate pass, cheap, keeps this table
    # independent of the sampling internals below) ---
    max_k = cfg["candidate"]["max_k"]
    candidates = generate_candidates(score_fn, prefix_targets, max_k, batch_size=cfg["batch_size"])
    rec = candidate_recall(prefix_targets, candidates, tuple(cfg["candidate"]["report_ks"]))
    print("candidate recall:", rec)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "candidate_recall.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recall_run", "k", "candidate_recall", "n_users"])
        for k, v in rec.items():
            w.writerow([cfg["recall_run"], k, round(v, 5), len(prefix_targets)])

    # --- Shared feature infrastructure ---
    item_pop = compute_item_popularity(prefix_targets, data.n_items)
    item_store = ItemFeatureStore(
        prefix_targets, data.n_items,
        coords_csv=PROCESSED / "poi_coords.csv",
        density_radius_km=cfg["item_density_radius_km"],
    )
    pop_pool = build_popularity_pool(item_pop, top_frac=cfg["negatives"]["pop_top_frac"])
    geo_tree, geo_idx_valid = build_geo_tree(item_store.coords)
    split = train_val_user_split(list(prefix_targets), val_frac=cfg["val_frac"], seed=cfg["seed"])

    # --- Sampling pass: re-scores in batches (same masking as candidate
    # generation) so recall_score is available for every sampled item, not
    # just the top-K ---
    users = np.array(sorted(prefix_targets))
    rows = []
    for start in range(0, len(users), cfg["batch_size"]):
        batch = users[start:start + cfg["batch_size"]]
        scores = np.asarray(score_fn(batch), dtype=np.float32)
        for r, u in enumerate(batch):
            scores[r, prefix_targets[int(u)]["prefix_items"]] = -np.inf
        rows.extend(build_samples_for_batch(
            batch, scores, prefix_targets, max_k, item_pop, item_store, pop_pool,
            geo_tree, geo_idx_valid, split, cfg["seed"],
            n_easy=cfg["negatives"]["n_easy"], n_pop=cfg["negatives"]["n_pop"],
            n_hard=cfg["negatives"]["n_hard"], n_geo=cfg["negatives"]["n_geo"],
        ))
        print(f"  sampled {start + len(batch)}/{len(users)} users -> {len(rows)} rows so far",
              flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(["user_id", "negative_source", "item_id"]).reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "ranking_samples.csv"
    df.to_csv(out_path, index=False)
    samples_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()

    # --- Stats: quantiles / missing rate / label & source distribution ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("user_id", "item_id", "query_ts")]
    quantiles = df[numeric_cols].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_dict()
    missing_rate = df[numeric_cols].isna().mean().to_dict()
    source_counts = df["negative_source"].value_counts().to_dict()
    label_counts = df["label"].value_counts().to_dict()

    stats = {
        "n_rows": len(df),
        "n_users": int(df["user_id"].nunique()),
        "samples_sha256": samples_hash,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "negative_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "candidate_recall": {str(k): round(v, 5) for k, v in rec.items()},
        "quantiles": {c: {str(q): (None if pd.isna(v) else round(float(v), 5))
                          for q, v in qs.items()} for c, qs in quantiles.items()},
        "missing_rate": {c: round(float(v), 5) for c, v in missing_rate.items()},
    }
    (RESULTS / "ranking_data_stats.json").write_text(json.dumps(stats, indent=2))

    # --- Feature schema ---
    schema = [
        {"column": c, "dtype": str(df[c].dtype), "group": feature_group(c)}
        for c in df.columns
    ]
    (RESULTS / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    # --- Audit sample: deterministic random draw of N rows for manual review ---
    rng = np.random.default_rng(cfg["seed"])
    audit_idx = rng.choice(len(df), size=min(cfg["audit_sample_size"], len(df)), replace=False)
    df.iloc[sorted(audit_idx)].to_csv(RESULTS / "ranking_data_audit_sample.csv", index=False)

    print(f"\nrows={len(df)}  users={stats['n_users']}  sha256={samples_hash[:16]}...")
    print("label_counts:", label_counts)
    print("negative_source_counts:", source_counts)
    print(f"\nWrote {out_path}")
    print(f"Wrote {RESULTS / 'candidate_recall.csv'}, ranking_data_stats.json, "
          f"feature_schema.json, ranking_data_audit_sample.csv")


if __name__ == "__main__":
    main()
