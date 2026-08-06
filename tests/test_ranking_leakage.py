"""Leakage tests for the M6 ranking dataset (PROJECT_HANDOFF_V2.md §0.2 rule 5):
"any ranking sample, feature statistic, negative pool, or candidate
generation may only use data before its own prediction time."

Two levels:
  - unit tests on small synthetic prefix/target data, isolating exact
    failure modes (fast, no dependency on generated files)
  - integration checks against the actual data/processed/ranking_samples.csv
    produced by scripts/build_ranking_data.py, so a silent regression in the
    real pipeline is caught, not just in the isolated helper functions.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import GowallaData
from src.data.ranking_dataset import build_prefix_targets
from src.features.item_features import compute_item_popularity
from src.features.user_features import user_feature_row
from src.utils.common import ROOT

SAMPLES_PATH = ROOT / "data" / "processed" / "ranking_samples.csv"
SEQ_TS_PATH = ROOT / "data" / "processed" / "train_sequences_ts.pkl"


def _load_ts_fixture():
    with open(SEQ_TS_PATH, "rb") as f:
        return pickle.load(f)


# ---------- unit: moving the cutoff earlier must never expose later items ----------

def test_user_features_depend_only_on_the_passed_prefix():
    item_pop = np.array([5, 3, 0, 10, 1], dtype=np.int64)
    coords = np.full((5, 2), np.nan)
    full_items, full_ts = [0, 1, 2, 3], [100, 200, 300, 400]
    truncated_items, truncated_ts = full_items[:-1], full_ts[:-1]  # earlier cutoff

    full_row = user_feature_row(full_items, full_ts, item_pop, coords)
    trunc_row = user_feature_row(truncated_items, truncated_ts, item_pop, coords)

    assert trunc_row["user_history_count"] == 3
    assert full_row["user_history_count"] == 4
    # the truncated view must not "see" item 3's popularity contribution
    assert (trunc_row["user_avg_visited_log1p_popularity"]
            != full_row["user_avg_visited_log1p_popularity"])
    # span can only shrink or stay equal when the cutoff moves earlier
    assert trunc_row["user_active_span_days"] <= full_row["user_active_span_days"]


# ---------- unit: item popularity must never count a user's own held-out target ----------

def test_item_popularity_excludes_all_held_out_targets():
    prefix_targets = {
        1: {"prefix_items": [10, 11], "prefix_ts": [1, 2], "target_item": 12, "target_ts": 3},
        2: {"prefix_items": [12, 13], "prefix_ts": [1, 2], "target_item": 10, "target_ts": 3},
    }
    pop = compute_item_popularity(prefix_targets, n_items=20)
    assert pop[12] == 1  # only user 2's prefix contains item 12 -- user 1's target=12 must not count
    assert pop[10] == 1  # only user 1's prefix contains item 10 -- user 2's target=10 must not count
    total_prefix_pairs = sum(len(v["prefix_items"]) for v in prefix_targets.values())
    assert pop.sum() == total_prefix_pairs  # would be +2 if either target had leaked in


# ---------- integration: official test.txt is never the source of an internal target ----------

def test_targets_are_official_train_items_never_official_test():
    data = GowallaData(ROOT / "data" / "gowalla")
    prefix_targets = build_prefix_targets(_load_ts_fixture(), min_history=5)
    for u, pt in prefix_targets.items():
        assert pt["target_item"] not in set(data.test.get(u, [])), (
            f"user {u}'s internal validation target {pt['target_item']} "
            "leaked from the sealed official test set"
        )
        assert pt["target_item"] in data._train_sets.get(u, set())


# ---------- integration: the generated samples table is internally consistent ----------

@pytest.mark.skipif(not SAMPLES_PATH.exists(), reason="run scripts/build_ranking_data.py first")
def test_generated_positive_rows_are_in_candidates():
    df = pd.read_csv(SAMPLES_PATH)
    positives = df[df["negative_source"] == "positive"]
    assert len(positives) > 0
    assert (positives["label"] == 1).all()
    assert (positives["cross_candidate_rank"] >= 0).all()


@pytest.mark.skipif(not SAMPLES_PATH.exists(), reason="run scripts/build_ranking_data.py first")
def test_generated_negatives_never_equal_the_users_own_target():
    df = pd.read_csv(SAMPLES_PATH)
    target_by_user = df.loc[df["negative_source"] == "positive"].set_index("user_id")["item_id"]
    negatives = df[df["label"] == 0].copy()
    negatives["target"] = negatives["user_id"].map(target_by_user)
    assert not (negatives["target"] == negatives["item_id"]).any()


@pytest.mark.skipif(not SAMPLES_PATH.exists(), reason="run scripts/build_ranking_data.py first")
def test_generated_negatives_never_in_the_users_own_prefix():
    """A negative the user actually visited before the cutoff would test
    'repeat this visit', a different question from the one this dataset is
    built to answer -- and would understate the model's real negative signal."""
    prefix_targets = build_prefix_targets(_load_ts_fixture(), min_history=5)
    df = pd.read_csv(SAMPLES_PATH)
    negatives = df[df["label"] == 0]
    bad = sum(
        1 for row in negatives.itertuples()
        if row.item_id in prefix_targets[row.user_id]["prefix_items"]
    )
    assert bad == 0


@pytest.mark.skipif(not SAMPLES_PATH.exists(), reason="run scripts/build_ranking_data.py first")
def test_generated_dataset_is_reproducible_hash():
    """scripts/build_ranking_data.py must be deterministic given the same
    config and seed -- rerun it and compare against the stored hash if this
    ever fails, rather than assuming machine nondeterminism."""
    import hashlib
    import json

    stats = json.loads((ROOT / "experiments" / "results" / "ranking_data_stats.json").read_text())
    actual = hashlib.sha256(SAMPLES_PATH.read_bytes()).hexdigest()
    assert actual == stats["samples_sha256"], (
        "ranking_samples.csv does not match the hash recorded in "
        "ranking_data_stats.json -- rerun scripts/build_ranking_data.py "
        "and re-verify before trusting this file"
    )
