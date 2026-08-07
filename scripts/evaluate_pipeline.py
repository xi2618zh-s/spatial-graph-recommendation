"""M7: train ranker(s) and run the end-to-end candidate -> rank -> top-K
evaluation in one command.

Usage:
    python scripts/evaluate_pipeline.py --config configs/ranking_data.yaml

For every val-split user, regenerates the FULL top-K candidate list from the
frozen recall model (not the sampled training rows), scores it under four
methods (retrieval-score baseline, LR, GBDT, GBDT feature-group ablation),
and reports Recall@20/NDCG@20/MRR@20 for each — all four share the exact
same candidate pool and user set, per the M7 evaluation protocol.

Outputs:
    experiments/results/ranking_eval.csv          method x metrics table
    experiments/results/ranking_eval_timing.json  feature-gen/train/serve timing
    experiments/logs/ranker_{lr,gbdt}/{model.pkl,config.json}
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.eval.eval_setup import build_val_eval_context
from src.eval.pipeline_evaluator import ranking_metrics_from_frame
from src.train.ranker_trainer import feature_columns, score as ranker_score, train_gbdt, train_lr
from src.utils.common import ROOT, load_config

RESULTS = ROOT / "experiments" / "results"
LOGS = ROOT / "experiments" / "logs"
ABLATION_ORDER = ["recall_only", "stats", "spatial", "full"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ranking_data.yaml")
    ap.add_argument("--lr-config", default="configs/ranker_lr.yaml")
    ap.add_argument("--gbdt-config", default="configs/ranker_gbdt.yaml")
    ap.add_argument("--samples", default=str(ROOT / "data" / "processed" / "ranking_samples.csv"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--latency-sample-users", type=int, default=500)
    args = ap.parse_args()
    cfg = load_config(args.config)
    lr_cfg = load_config(args.lr_config)
    gbdt_cfg = load_config(args.gbdt_config)

    t0 = time.time()
    df = pd.read_csv(args.samples)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"loaded {len(df)} sample rows ({len(train_df)} train) in {time.time() - t0:.1f}s")

    # --- rebuild the full candidate pool + features for val users (not the sampled rows) ---
    t0 = time.time()
    ctx = build_val_eval_context(cfg)
    feature_gen_time = time.time() - t0
    eval_df, val_users = ctx.eval_df, ctx.val_users
    print(f"regenerated candidates + eval frame for {len(val_users)} val users "
          f"({len(eval_df)} rows) in {feature_gen_time:.1f}s")

    results = []

    # --- mandatory baseline: sort candidates by the frozen recall model's own score ---
    m = ranking_metrics_from_frame(eval_df, "cross_recall_score", k=args.k)
    results.append({"method": "retrieval_score_sort", "feature_group": "n/a", "n_features": 1, **m})
    print("retrieval_score_sort:", m)

    # --- LR, hyperparams from configs/ranker_lr.yaml ---
    lr_group = lr_cfg["feature_group"]
    lr_cols = feature_columns(train_df, lr_group)
    lr_kwargs = {k: v for k, v in lr_cfg["model"].items() if k != "type"}
    t0 = time.time()
    lr_model = train_lr(train_df, lr_cols, seed=lr_cfg["seed"], **lr_kwargs)
    lr_train_s = time.time() - t0
    eval_df["lr_score"] = ranker_score(lr_model, eval_df, lr_cols)
    m = ranking_metrics_from_frame(eval_df, "lr_score", k=args.k)
    results.append({"method": "ranker_lr", "feature_group": lr_group, "n_features": len(lr_cols),
                    "train_seconds": round(lr_train_s, 2), **m})
    print("ranker_lr:", m)
    _persist_model(lr_model, "ranker_lr", "lr", lr_group, lr_cols, len(train_df), lr_cfg["seed"])

    # --- GBDT feature-group ablation (recall_only -> stats -> spatial -> full),
    # hyperparams from configs/ranker_gbdt.yaml; the "full" entry doubles as
    # the headline ranker_gbdt result ---
    gbdt_kwargs = {k: v for k, v in gbdt_cfg["model"].items() if k != "type"}
    gbdt_full_model = None
    for group in ABLATION_ORDER:
        cols = feature_columns(train_df, group)
        t0 = time.time()
        model = train_gbdt(train_df, cols, seed=gbdt_cfg["seed"], **gbdt_kwargs)
        train_s = time.time() - t0
        eval_df["_gbdt_score"] = ranker_score(model, eval_df, cols)
        m = ranking_metrics_from_frame(eval_df, "_gbdt_score", k=args.k)
        method = "ranker_gbdt" if group == "full" else "ranker_gbdt_ablation"
        results.append({"method": method, "feature_group": group, "n_features": len(cols),
                        "train_seconds": round(train_s, 2), **m})
        print(f"gbdt[{group}] ({len(cols)} features):", m)
        if group == "full":
            gbdt_full_model = model
            gbdt_full_cols = cols
    eval_df.drop(columns=["_gbdt_score"], inplace=True)
    _persist_model(gbdt_full_model, "ranker_gbdt", "gbdt", "full", gbdt_full_cols,
                   len(train_df), gbdt_cfg["seed"])

    # --- per-user scoring latency (P50/P95), full GBDT model, on a subsample of val users ---
    sample_users = val_users[:args.latency_sample_users]
    latencies = []
    for u in sample_users:
        g = eval_df[eval_df["user_id"] == u]
        t0 = time.perf_counter()
        ranker_score(gbdt_full_model, g, gbdt_full_cols)
        latencies.append(time.perf_counter() - t0)
    latencies = np.asarray(latencies)
    timing = {
        "feature_gen_seconds_total": round(feature_gen_time, 2),
        "feature_gen_seconds_per_user": round(feature_gen_time / len(val_users), 5),
        "per_user_score_p50_ms": round(float(np.percentile(latencies, 50)) * 1000, 3),
        "per_user_score_p95_ms": round(float(np.percentile(latencies, 95)) * 1000, 3),
        "n_users_timed": len(latencies),
        "candidates_per_user": cfg["candidate"]["max_k"],
    }
    print("timing:", timing)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS / "ranking_eval.csv", index=False)
    (RESULTS / "ranking_eval_timing.json").write_text(json.dumps(timing, indent=2))
    print(f"\nWrote {RESULTS / 'ranking_eval.csv'} and ranking_eval_timing.json")


def _persist_model(model, run_name: str, model_type: str, feature_group: str,
                   feature_cols: list[str], n_train_rows: int, seed: int) -> None:
    run_dir = LOGS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.pkl")
    (run_dir / "config.json").write_text(json.dumps({
        "model_type": model_type, "feature_group": feature_group,
        "feature_columns": feature_cols, "n_train_rows": n_train_rows, "seed": seed,
    }, indent=2))


if __name__ == "__main__":
    main()
