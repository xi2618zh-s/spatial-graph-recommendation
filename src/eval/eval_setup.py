"""Shared val-split evaluation context: regenerate the full candidate pool
and feature rows once, reused by M7 (ranker training/eval) and M8 (business
slices/bias diagnostics) so both read the exact same setup instead of two
scripts silently drifting apart."""

from dataclasses import dataclass

from src.data.ranking_dataset import (
    build_prefix_targets, generate_candidates_with_scores,
    load_timestamped_sequences, train_val_user_split,
)
from src.eval.pipeline_evaluator import build_eval_frame
from src.features.item_features import ItemFeatureStore, compute_item_popularity
from src.models.registry import load_checkpoint
from src.utils.common import ROOT


@dataclass
class EvalContext:
    cfg: dict
    prefix_targets: dict
    item_pop: object
    item_store: object
    split: dict
    val_users: list
    eval_df: object


def build_val_eval_context(cfg: dict) -> EvalContext:
    seqs = load_timestamped_sequences(ROOT / "data" / "processed" / "train_sequences_ts.pkl")
    prefix_targets = build_prefix_targets(seqs, min_history=cfg["min_history"])
    _, data, _, score_fn = load_checkpoint(cfg["recall_run"], device="cpu")
    item_pop = compute_item_popularity(prefix_targets, data.n_items)
    item_store = ItemFeatureStore(
        prefix_targets, data.n_items, coords_csv=ROOT / "data" / "processed" / "poi_coords.csv",
        density_radius_km=cfg["item_density_radius_km"],
    )
    split = train_val_user_split(list(prefix_targets), val_frac=cfg["val_frac"], seed=cfg["seed"])
    val_users = [u for u, s in split.items() if s == "val"]

    cand_items, cand_scores = generate_candidates_with_scores(
        score_fn, {u: prefix_targets[u] for u in val_users}, cfg["candidate"]["max_k"],
        batch_size=cfg["batch_size"],
    )
    eval_df = build_eval_frame(val_users, prefix_targets, cand_items, cand_scores,
                               item_pop, item_store, split)
    return EvalContext(cfg, prefix_targets, item_pop, item_store, split, val_users, eval_df)
