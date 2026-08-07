"""M8: business-proxy metrics, popularity bias, and cold-start/long-tail
slice diagnostics, run over the same val-user candidate pool M7 uses.

Usage:
    python scripts/evaluate_slices.py --config configs/ranking_data.yaml

Compares two methods head-to-head on IDENTICAL users/candidates:
  retrieval_score_sort   the frozen recall model's own ordering (M7 baseline)
  ranker_gbdt             the persisted M7 GBDT ranker (experiments/logs/ranker_gbdt/)

Outputs (experiments/results/):
    slice_metrics.csv        Recall/NDCG@20 x {overall, user-activity, target
                             popularity, target distance, near-cold-start}
                             x {method}, with 95% bootstrap CI and n
    bias_metrics.json        coverage / tail exposure / ARP / popularity lift /
                             exposure Gini / distance / list diversity, per method
    cold_start_report.json   strict vs near cold-start counts (kept separate)
    bucket_boundaries.json   the train-fixed bucket edges used above
    figures/slice_recall_by_user_activity.png
    figures/popularity_coverage_tradeoff.png
    figures/recommendation_distance_distribution.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import GowallaData
from src.eval.bias_metrics import (
    average_recommendation_popularity, catalog_coverage, effective_user_coverage,
    exposure_gini, list_internal_diversity_km, mean_neg_log_popularity,
    popularity_lift, recommendation_distance_stats, tail_exposure_share,
)
from src.eval.bootstrap import bootstrap_ci
from src.eval.eval_setup import build_val_eval_context
from src.eval.pipeline_evaluator import ranking_metrics_per_user
from src.eval.slices import (
    distance_buckets, item_popularity_buckets, strict_cold_start_counts,
    user_activity_buckets,
)
from src.features.user_features import haversine_km
from src.train.ranker_trainer import score as ranker_score
from src.utils.common import ROOT, load_config

RESULTS = ROOT / "experiments" / "results"
FIGURES = RESULTS / "figures"
LOGS = ROOT / "experiments" / "logs"
K = 20


def target_distance_km(prefix_targets: dict, item_store, users) -> dict[int, float]:
    out = {}
    for u in users:
        pt = prefix_targets[u]
        latlon = item_store.coords[pt["prefix_items"]]
        valid = ~np.isnan(latlon[:, 0])
        t_lat, t_lon = item_store.coords[pt["target_item"]]
        if not valid.any() or np.isnan(t_lat):
            out[u] = np.nan
            continue
        c_lat, c_lon = latlon[valid].mean(axis=0)
        out[u] = haversine_km(c_lat, c_lon, t_lat, t_lon)
    return out


def slice_report(user_ids, values, bucket_of, method, slice_name, n_boot=2000, seed=2020):
    rows = []
    values_by_user = dict(zip(user_ids, values))
    buckets = sorted(set(bucket_of.get(u, "n/a") for u in user_ids), key=str)
    for b in buckets:
        sub = np.array([values_by_user[u] for u in user_ids if bucket_of.get(u, "n/a") == b])
        if len(sub) == 0:
            continue
        ci = bootstrap_ci(sub, n_boot=n_boot, seed=seed)
        rows.append({"method": method, "slice": slice_name, "bucket": b, **ci})
    # overall (bucket="all")
    ci = bootstrap_ci(np.array(values), n_boot=n_boot, seed=seed)
    rows.append({"method": method, "slice": slice_name, "bucket": "all", **ci})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ranking_data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ctx = build_val_eval_context(cfg)
    eval_df, prefix_targets, item_pop, item_store = ctx.eval_df, ctx.prefix_targets, ctx.item_pop, ctx.item_store
    val_users = ctx.val_users
    print(f"eval frame: {len(eval_df)} rows, {len(val_users)} val users")

    gbdt_cfg = json.loads((LOGS / "ranker_gbdt" / "config.json").read_text())
    gbdt_model = joblib.load(LOGS / "ranker_gbdt" / "model.pkl")
    eval_df["gbdt_score"] = ranker_score(gbdt_model, eval_df, gbdt_cfg["feature_columns"])

    methods = {"retrieval_score_sort": "cross_recall_score", "ranker_gbdt": "gbdt_score"}
    per_method = {m: ranking_metrics_per_user(eval_df, col, k=K) for m, col in methods.items()}

    # --- bucket boundaries, fixed over ALL eligible train users ---
    activity_bucket, activity_edges = user_activity_buckets(prefix_targets)
    item_bucket, item_share = item_popularity_buckets(item_pop)
    dist_km = target_distance_km(prefix_targets, item_store, val_users)
    target_pop_bucket = {u: item_bucket[prefix_targets[u]["target_item"]] for u in val_users}
    dist_bucket_arr = distance_buckets(np.array([dist_km[u] for u in val_users]))
    target_dist_bucket = dict(zip(val_users, dist_bucket_arr))

    data = GowallaData(ROOT / "data" / "gowalla")
    cold = strict_cold_start_counts(data)
    print("strict cold-start:", cold)

    low_history_users = {u: (activity_bucket[u] == "low") for u in prefix_targets}

    # --- slice metrics + bootstrap CI ---
    all_rows = []
    for method, per in per_method.items():
        uids = per["user_ids"]
        recall = per[f"recall@{K}"]
        ndcg = per[f"ndcg@{K}"]
        all_rows += slice_report(uids, recall, activity_bucket, method, "user_activity", seed=cfg["seed"])
        all_rows += slice_report(uids, ndcg, activity_bucket, method, "user_activity_ndcg", seed=cfg["seed"])
        all_rows += slice_report(uids, recall, target_pop_bucket, method, "target_popularity", seed=cfg["seed"])
        all_rows += slice_report(uids, recall, target_dist_bucket, method, "target_distance", seed=cfg["seed"])
        near_cold = {u: ("low_history" if low_history_users.get(u) else "rest") for u in val_users}
        all_rows += slice_report(uids, recall, near_cold, method, "near_cold_start", seed=cfg["seed"])
    slice_df = pd.DataFrame(all_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    slice_df.to_csv(RESULTS / "slice_metrics.csv", index=False)

    # --- business-proxy / bias metrics ---
    n_items = item_store.n_items
    catalog_mean_pop = float(item_pop.mean())
    eval_indexed = eval_df.set_index(["user_id", "item_id"])["cross_dist_to_center_km"]
    bias = {}
    for method, per in per_method.items():
        top_lists = per["top_k_items"]
        rec_dist = []
        for u, items in top_lists.items():
            for it in items:
                try:
                    rec_dist.append(eval_indexed.loc[(u, it)])
                except KeyError:
                    rec_dist.append(np.nan)
        rec_dist = np.array(rec_dist, dtype=float)
        arp = average_recommendation_popularity(top_lists, item_pop)
        bias[method] = {
            "catalog_coverage_at_20": catalog_coverage(top_lists, n_items),
            "effective_user_coverage_at_20": effective_user_coverage(top_lists, K),
            "tail_exposure_share": tail_exposure_share(top_lists, item_bucket),
            "mean_neg_log_popularity": mean_neg_log_popularity(top_lists, item_pop),
            "average_recommendation_popularity": arp,
            "popularity_lift_vs_catalog_mean": popularity_lift(arp, catalog_mean_pop),
            "exposure_gini": exposure_gini(top_lists, n_items),
            "recommendation_distance": recommendation_distance_stats(rec_dist),
            "list_internal_diversity_km": list_internal_diversity_km(top_lists, item_store.coords),
        }
        print(f"{method} bias metrics:", json.dumps(bias[method], indent=2))

    (RESULTS / "bias_metrics.json").write_text(json.dumps(bias, indent=2))
    (RESULTS / "cold_start_report.json").write_text(json.dumps({
        "strict_cold_start": cold,
        "near_cold_start_definition": "bottom user-activity tertile (prefix history length) "
                                      "and bottom 50% items by prefix-universe interaction count "
                                      "-- NOT the same as strict cold-start above",
        "near_cold_start_low_history_users": int(sum(low_history_users.values())),
        "near_cold_start_low_freq_items": int((item_bucket == "tail").sum()),
        "item_popularity_bucket_share": item_share,
    }, indent=2))
    (RESULTS / "bucket_boundaries.json").write_text(json.dumps({
        "user_activity_history_count": activity_edges,
        "item_popularity_head_frac": 0.2, "item_popularity_tail_frac": 0.5,
        "target_distance_edges_km": [1, 5, 20, 100],
    }, indent=2))

    _make_figures(slice_df, bias, per_method)
    print(f"\nWrote slice_metrics.csv, bias_metrics.json, cold_start_report.json, "
          f"bucket_boundaries.json, and figures/ under {RESULTS}")


def _deg_coords(item_store) -> np.ndarray:
    return np.degrees(item_store.coords)


def _make_figures(slice_df: pd.DataFrame, bias: dict, per_method: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. slice effects: Recall@20 by user-activity bucket, both methods, with CI
    sub = slice_df[(slice_df["slice"] == "user_activity") & (slice_df["bucket"] != "all")]
    fig, ax = plt.subplots(figsize=(6, 4))
    buckets = ["low", "mid", "high"]
    width = 0.35
    x = np.arange(len(buckets))
    for i, method in enumerate(per_method):
        m = sub[sub["method"] == method].set_index("bucket").reindex(buckets)
        err = [m["mean"] - m["ci_low"], m["ci_high"] - m["mean"]]
        ax.bar(x + (i - 0.5) * width, m["mean"], width, yerr=err, capsize=3, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel(f"Recall@{K}")
    ax.set_title("Recall@20 by user-activity bucket (95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "slice_recall_by_user_activity.png", dpi=150)
    plt.close(fig)

    # 2. popularity/coverage trade-off
    methods = list(bias.keys())
    metrics = ["catalog_coverage_at_20", "tail_exposure_share", "popularity_lift_vs_catalog_mean"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    for ax, metric in zip(axes, metrics):
        vals = [bias[m][metric] for m in methods]
        ax.bar(methods, vals, color=["#4C72B0", "#DD8452"])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Coverage / long-tail / popularity-bias trade-off")
    fig.tight_layout()
    fig.savefig(FIGURES / "popularity_coverage_tradeoff.png", dpi=150)
    plt.close(fig)

    # 3. recommendation distance distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        d = bias[method]["recommendation_distance"]
        ax.bar(method, d.get("p50_km", 0), alpha=0.6, label=f"{method} median")
    ax.set_ylabel("Recommended-item distance to user activity center (km, median)")
    ax.set_title("Recommendation spatial cost by method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "recommendation_distance_distribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
