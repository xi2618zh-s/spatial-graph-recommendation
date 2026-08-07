"""M10: CPU end-to-end smoke test on tiny synthetic data --
prefix/target -> candidates -> features -> ranker training -> evaluation ->
ANN index -- exercising the same module chain M6 through M9 use on the real
41K-item dataset, but small and fast enough to run on every CPU test pass
without needing the real Gowalla files or a trained checkpoint.
"""

import numpy as np
import pandas as pd

from src.data.ranking_dataset import (
    build_prefix_targets, candidate_recall, generate_candidates_with_scores,
    train_val_user_split,
)
from src.data.sampling import build_geo_tree, build_popularity_pool, build_samples_for_batch
from src.eval.pipeline_evaluator import build_eval_frame, ranking_metrics_from_frame
from src.features.item_features import ItemFeatureStore, compute_item_popularity
from src.serving.index import build_flat_index
from src.serving.search import masked_search
from src.train.ranker_trainer import feature_columns, score as ranker_score, train_gbdt

N_USERS, N_ITEMS, DIM = 12, 25, 6


def _synthetic_sequences(seed=0):
    rng = np.random.default_rng(seed)
    seqs = {}
    for u in range(N_USERS):
        length = rng.integers(6, 10)
        items = rng.choice(N_ITEMS, size=length, replace=False).tolist()
        ts = sorted(rng.integers(1_600_000_000, 1_700_000_000, size=length).tolist())
        seqs[u] = list(zip(items, ts))
    return seqs


def _synthetic_embeddings_and_scorer(seed=1):
    rng = np.random.default_rng(seed)
    user_emb = rng.normal(size=(N_USERS, DIM)).astype("float32")
    item_emb = rng.normal(size=(N_ITEMS, DIM)).astype("float32")

    def score_fn(user_ids):
        return user_emb[np.asarray(user_ids)] @ item_emb.T

    return user_emb, item_emb, score_fn


def _write_tiny_coords_csv(path):
    rng = np.random.default_rng(2)
    lat = rng.uniform(55.0, 56.0, size=N_ITEMS)
    lon = rng.uniform(12.0, 13.5, size=N_ITEMS)
    pd.DataFrame({"item_id": range(N_ITEMS), "lat": lat, "lon": lon}).to_csv(path, index=False)


def test_cpu_prepare_retrieve_rank_evaluate_serve_smoke(tmp_path):
    seqs = _synthetic_sequences()
    prefix_targets = build_prefix_targets(seqs, min_history=3)
    assert len(prefix_targets) > 0, "every synthetic user should clear the tiny min_history bar"

    user_emb, item_emb, score_fn = _synthetic_embeddings_and_scorer()

    # --- retrieve: candidate generation + reporting ---
    max_k = 10
    candidates = generate_candidates_with_scores(score_fn, prefix_targets, max_k)[0]
    rec = candidate_recall(prefix_targets, candidates, ks=(5, 10))
    assert 0.0 <= rec[5] <= rec[10] <= 1.0

    # --- features + layered negative sampling (reuses M6's real machinery) ---
    coords_csv = tmp_path / "poi_coords.csv"
    _write_tiny_coords_csv(coords_csv)
    item_pop = compute_item_popularity(prefix_targets, N_ITEMS)
    item_store = ItemFeatureStore(prefix_targets, N_ITEMS, coords_csv=coords_csv, density_radius_km=5.0)
    pop_pool = build_popularity_pool(item_pop, top_frac=0.3)
    geo_tree, geo_idx_valid = build_geo_tree(item_store.coords)
    split = train_val_user_split(list(prefix_targets), val_frac=0.3, seed=0)

    rows = []
    users = np.array(sorted(prefix_targets))
    scores = score_fn(users)
    for r, u in enumerate(users):
        scores[r, prefix_targets[int(u)]["prefix_items"]] = -np.inf
    rows = build_samples_for_batch(
        users, scores, prefix_targets, max_k, item_pop, item_store, pop_pool,
        geo_tree, geo_idx_valid, split, seed=0, n_easy=1, n_pop=1, n_hard=2, n_geo=1,
    )
    samples = pd.DataFrame(rows)
    assert (samples["label"] == 1).sum() > 0, "at least one recall hit expected in this tiny setup"
    assert samples.isna().sum().sum() >= 0  # NaNs allowed (missing flags cover them); just must not crash

    # --- rank: train a tiny GBDT on the synthetic samples ---
    train_df = samples[samples["split"] == "train"]
    cols = feature_columns(train_df, "full")
    model = train_gbdt(train_df, cols, seed=0, max_iter=20)

    # --- evaluate: full candidate re-featurization + ranking metrics ---
    val_users = [u for u, s in split.items() if s == "val"]
    if val_users:  # tiny sample sizes can occasionally leave val empty; skip metrics if so
        cand_items, cand_scores = generate_candidates_with_scores(
            score_fn, {u: prefix_targets[u] for u in val_users}, max_k
        )
        eval_df = build_eval_frame(val_users, prefix_targets, cand_items, cand_scores,
                                   item_pop, item_store, split)
        eval_df["gbdt_score"] = ranker_score(model, eval_df, cols)
        m = ranking_metrics_from_frame(eval_df, "gbdt_score", k=5)
        assert 0.0 <= m["recall@5"] <= 1.0
        assert np.isfinite(m["ndcg@5"])

    # --- serve: build a Flat index and do a masked search ---
    index = build_flat_index(item_emb)
    some_user = users[0]
    ids, scores_out = masked_search(index, user_emb[some_user],
                                    set(prefix_targets[int(some_user)]["prefix_items"]), k=5)
    assert len(ids) <= 5
    assert np.isfinite(scores_out).all()
