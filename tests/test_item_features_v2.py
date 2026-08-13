"""V2-0.2: correctness of GlobalItemTimeline's exact point-in-time item
statistics (src/features/item_features_v2.py), including a brute-force
cross-check against real Gowalla data -- not just a synthetic toy example."""

import time

import numpy as np
import pytest

from src.data.ranking_dataset import load_timestamped_sequences
from src.features.item_features_v2 import GlobalItemTimeline, build_global_item_timeline
from src.utils.common import ROOT

SEQ_TS_PATH = ROOT / "data" / "processed" / "train_sequences_ts.pkl"


def test_popularity_before_matches_hand_computation():
    # item 5 interacted with at ts=[10, 20, 20, 30]; item 6 never interacted with
    seq = {
        0: [(5, 10), (6, 999999)],  # 6 only appears far in the future, irrelevant to item 5's timeline
        1: [(5, 20)],
        2: [(5, 20)],
        3: [(5, 30)],
    }
    tl = GlobalItemTimeline(seq, n_items=10)
    assert tl.popularity_before(5, cutoff_ts=5) == 0
    assert tl.popularity_before(5, cutoff_ts=10) == 0   # strict "<": the ts=10 event itself excluded
    assert tl.popularity_before(5, cutoff_ts=11) == 1
    assert tl.popularity_before(5, cutoff_ts=20) == 1   # both ts=20 events excluded (strict)
    assert tl.popularity_before(5, cutoff_ts=21) == 3
    assert tl.popularity_before(5, cutoff_ts=31) == 4
    assert tl.popularity_before(7, cutoff_ts=100) == 0  # item never interacted with at all


def test_last_active_before_matches_hand_computation():
    seq = {0: [(5, 10)], 1: [(5, 20)], 2: [(5, 30)]}
    tl = GlobalItemTimeline(seq, n_items=10)
    assert np.isnan(tl.last_active_before(5, cutoff_ts=10))  # strict: ts=10 itself excluded
    assert tl.last_active_before(5, cutoff_ts=11) == 10
    assert tl.last_active_before(5, cutoff_ts=25) == 20
    assert tl.last_active_before(5, cutoff_ts=999) == 30
    assert np.isnan(tl.last_active_before(6, cutoff_ts=999))  # never interacted with


def test_a_users_own_query_event_never_counts_toward_its_own_item_popularity():
    """The exact leakage failure mode this module exists to close: an
    item's popularity at the moment of a specific interaction must not
    include that interaction itself."""
    seq = {0: [(5, 100)], 1: [(5, 100)]}  # two users interact with item 5 at the SAME instant
    tl = GlobalItemTimeline(seq, n_items=10)
    # querying "how popular was item 5 right at ts=100" must exclude BOTH
    # same-instant events (their relative order is unknown/ambiguous)
    assert tl.popularity_before(5, cutoff_ts=100) == 0


@pytest.mark.skipif(not SEQ_TS_PATH.exists(), reason="run scripts/prepare_data.py first")
def test_matches_brute_force_on_real_data_for_sampled_queries():
    seqs = load_timestamped_sequences(SEQ_TS_PATH)
    n_items = 1 + max(it for seq in seqs.values() for it, _ in seq)
    timeline, build_s = build_global_item_timeline(seqs, n_items)
    print(f"\nbuild time: {build_s:.2f}s, {timeline.total_interactions} interactions indexed")

    # sample real (item, cutoff) pairs straight from the data
    rng = np.random.default_rng(0)
    all_pairs = [(it, ts) for seq in seqs.values() for it, ts in seq]
    sample = [all_pairs[i] for i in rng.choice(len(all_pairs), size=200, replace=False)]

    for item, cutoff_ts in sample:
        expected = sum(
            1 for seq in seqs.values() for it, ts in seq if it == item and ts < cutoff_ts
        )
        assert timeline.popularity_before(item, cutoff_ts) == expected, (item, cutoff_ts)


@pytest.mark.skipif(not SEQ_TS_PATH.exists(), reason="run scripts/prepare_data.py first")
def test_total_popularity_sums_to_total_official_train_interactions():
    """Sanity check against the known dataset scale (810,128 official train
    pairs, prepare_report.json) -- every interaction must be indexed exactly
    once, on exactly one item."""
    seqs = load_timestamped_sequences(SEQ_TS_PATH)
    n_items = 1 + max(it for seq in seqs.values() for it, _ in seq)
    timeline, _ = build_global_item_timeline(seqs, n_items)
    assert timeline.total_interactions == 810_128


@pytest.mark.skipif(not SEQ_TS_PATH.exists(), reason="run scripts/prepare_data.py first")
def test_query_throughput_is_practical_for_pipeline_integration():
    """Not a hard pass/fail gate -- reports timing so later phases can
    decide whether per-row Python-level calls are fast enough or need
    batching, per the project's own 30-minutes-then-report-and-decide rule."""
    seqs = load_timestamped_sequences(SEQ_TS_PATH)
    n_items = 1 + max(it for seq in seqs.values() for it, _ in seq)
    timeline, build_s = build_global_item_timeline(seqs, n_items)

    rng = np.random.default_rng(0)
    items = rng.integers(0, n_items, size=100_000)
    cutoffs = rng.integers(1_200_000_000, 1_300_000_000, size=100_000)
    t0 = time.perf_counter()
    for it, ts in zip(items, cutoffs):
        timeline.popularity_before(int(it), int(ts))
    elapsed = time.perf_counter() - t0
    print(f"\nbuild={build_s:.2f}s, 100k popularity_before() calls={elapsed:.2f}s "
         f"({elapsed / 100_000 * 1e6:.1f}us/call)")
    assert elapsed < 60, (
        f"100k point-in-time queries took {elapsed:.1f}s -- M7's eval frame alone needs "
        "~900k such lookups; report this timing before wiring GlobalItemTimeline into "
        "the full V2 sampling pipeline"
    )
