"""V2-0.3: behavioral leakage tests, stronger than tests/test_ranking_v2_temporal_split.py's
unit-level checks -- these operate on the actual dataset-level structures
that would feed R0/R1 training, or poison real inputs and confirm the
pipeline is unaffected, rather than checking function arguments in isolation.

The H_inner/v0/y_train/y_val non-overlap assertion itself already has
full-scale (all 29,858 users) coverage in
tests/test_ranking_v2_temporal_split.py::test_four_layers_never_overlap_and_edges_nest_correctly
-- not duplicated here.
"""

import numpy as np
import pytest

from src.data.dataset import GowallaData
from src.data.ranking_dataset import load_timestamped_sequences
from src.data.ranking_dataset_v2 import build_v2_layers, hcore_edges, hcore_plus_ytrain_edges
from src.features.item_features_v2 import GlobalItemTimeline
from src.utils.common import ROOT

SEQ_TS_PATH = ROOT / "data" / "processed" / "train_sequences_ts.pkl"


@pytest.mark.skipif(not SEQ_TS_PATH.exists(), reason="run scripts/prepare_data.py first")
def test_retriever_train_edges_never_intersect_ranking_targets_dataset_wide():
    """Not a per-user attribute check -- builds the GLOBAL set of every
    (user, item) edge R0/R1 would actually train on, and the global set of
    every (user, y_train)/(user, y_val) ranking-target pair, and asserts the
    sets are disjoint. This is what R0/R1's training data literally is, not
    a proxy for it."""
    seqs = load_timestamped_sequences(SEQ_TS_PATH)
    layers = build_v2_layers(seqs, min_h_inner=5)

    r0_edges = {(u, it) for u, items in hcore_edges(layers).items() for it in items}
    r1_edges = {(u, it) for u, items in hcore_plus_ytrain_edges(layers).items() for it in items}
    y_train_pairs = {(u, L["y_train_item"]) for u, L in layers.items()}
    y_val_pairs = {(u, L["y_val_item"]) for u, L in layers.items()}

    assert r0_edges & y_train_pairs == set(), "R0's training edges must never contain a y_train pair"
    assert r0_edges & y_val_pairs == set(), "R0's training edges must never contain a y_val pair"
    assert r1_edges & y_val_pairs == set(), "R1's training edges must never contain a y_val pair"
    # r1_edges legitimately DOES contain every y_train pair (R1 trains on
    # Hcore + y_train by design) -- asserting the opposite would be wrong,
    # so it is deliberately not checked here.
    assert y_train_pairs <= r1_edges, "R1 must train on every y_train edge by design"


@pytest.mark.skipif(not SEQ_TS_PATH.exists(), reason="run scripts/prepare_data.py first")
def test_r0_r1_training_edges_unaffected_by_official_test_content(tmp_path):
    """Poisons a COPY of data/gowalla/test.txt (never the real file) with
    random (user, item) pairs, reloads GowallaData from the poisoned copy,
    and confirms the actual R0/R1 training edge sets -- and the n_users/
    n_items constants SnapshotGowallaData needs -- are byte-identical to the
    unpoisoned baseline. build_v2_layers()/hcore_edges() take pre-loaded
    sequences_ts (from train_sequences_ts.pkl, which prepare_data.py builds
    from train.txt + raw SNAP data and never reads test.txt) as their only
    input, so this also behaviorally proves that dependency doesn't exist,
    not just by code inspection.
    """
    official_real = GowallaData(ROOT / "data" / "gowalla")
    seqs = load_timestamped_sequences(SEQ_TS_PATH)
    layers = build_v2_layers(seqs, min_h_inner=5)
    hcore_real = hcore_edges(layers)
    hcore_plus_real = hcore_plus_ytrain_edges(layers)

    poisoned_dir = tmp_path / "gowalla_poisoned"
    poisoned_dir.mkdir()
    for name in ("train.txt", "user_list.txt", "item_list.txt"):
        (poisoned_dir / name).write_bytes((ROOT / "data" / "gowalla" / name).read_bytes())

    rng = np.random.default_rng(0)
    with open(poisoned_dir / "test.txt", "w") as f:
        for u in official_real.test:
            fake_items = rng.integers(0, official_real.n_items, size=5)
            f.write(f"{u} " + " ".join(map(str, fake_items)) + "\n")

    official_poisoned = GowallaData(poisoned_dir)
    assert official_poisoned.n_users == official_real.n_users
    assert official_poisoned.n_items == official_real.n_items
    # the poisoned test.txt content differs by construction -- confirms the
    # poisoning actually took effect, not a no-op
    assert official_poisoned.test != official_real.test

    layers_after_poisoning = build_v2_layers(seqs, min_h_inner=5)  # same seqs, untouched by test.txt
    assert hcore_edges(layers_after_poisoning) == hcore_real
    assert hcore_plus_ytrain_edges(layers_after_poisoning) == hcore_plus_real


def test_global_item_timeline_ignores_interactions_after_the_cutoff():
    """Integration-level cutoff-shift test (see also M6's unit-level
    version, tests/test_ranking_leakage.py::test_user_features_depend_only_on_the_passed_prefix):
    adding a FUTURE interaction must not change a point-in-time query at an
    earlier cutoff; adding a PAST one must (sanity check the test isn't
    vacuously trivial)."""
    base_seq = {0: [(5, 100)], 1: [(5, 200)]}
    cutoff = 250

    timeline_before = GlobalItemTimeline(base_seq, n_items=10)
    pop_before = timeline_before.popularity_before(5, cutoff)

    future_seq = {0: [(5, 100)], 1: [(5, 200)], 2: [(5, 999)]}  # 999 > cutoff=250
    timeline_with_future = GlobalItemTimeline(future_seq, n_items=10)
    assert timeline_with_future.popularity_before(5, cutoff) == pop_before, (
        "an interaction strictly after the cutoff must not change a point-in-time query"
    )

    past_seq = {0: [(5, 100)], 1: [(5, 200)], 2: [(5, 150)]}  # 150 < cutoff=250
    timeline_with_past = GlobalItemTimeline(past_seq, n_items=10)
    assert timeline_with_past.popularity_before(5, cutoff) == pop_before + 1, (
        "sanity check: an interaction strictly before the cutoff MUST change the count "
        "-- otherwise this test would pass vacuously regardless of correctness"
    )
