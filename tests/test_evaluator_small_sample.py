"""M10: hand-computed small-sample correctness check for
src/eval/evaluator.py -- the full-ranking Recall@K/NDCG@K protocol this
entire project's headline numbers are built on has never had a test that
verifies its arithmetic against an independently derived expected value."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.evaluator import evaluate


@pytest.fixture
def toy_data():
    # user 0 trained on item 0 (masked); test target = item 2
    # user 1 trained on item 1 (masked); test targets = {3, 4}
    return SimpleNamespace(train={0: [0], 1: [1]}, test={0: [2], 1: [3, 4]})


@pytest.fixture
def toy_scores():
    return np.array([
        [10.0, 1.0, 9.0, 2.0, 3.0],   # user 0 (item 0 will be masked regardless of its score here)
        [5.0, -100.0, 4.0, 20.0, 1.0],  # user 1 (item 1 will be masked)
    ], dtype=np.float32)


def _score_fn(scores):
    def fn(user_ids):
        return scores[np.asarray(user_ids)]
    return fn


def test_recall_and_ndcg_at_1_match_hand_computation(toy_data, toy_scores):
    m = evaluate(_score_fn(toy_scores), toy_data, topks=(1,))
    # user 0: unmasked ranking = [2, 4, 3, 1] (scores 9,3,2,1) -> top1=item2, a hit (n_rel=1)
    #   recall@1=1.0, ndcg@1 = 1/log2(2) / (1/log2(2)) = 1.0
    # user 1: unmasked ranking = [3, 0, 2, 4] (scores 20,5,4,1) -> top1=item3, a hit (n_rel=2)
    #   recall@1 = 1/2 = 0.5, ndcg@1 = (1/log2(2)) / (1/log2(2)) = 1.0
    assert m["recall@1"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["ndcg@1"] == pytest.approx((1.0 + 1.0) / 2)


def test_recall_and_ndcg_at_3_match_hand_computation(toy_data, toy_scores):
    m = evaluate(_score_fn(toy_scores), toy_data, topks=(3,))
    # user 0: top3 = [2, 4, 3], target {2} hit at rank 0 -> recall@3=1.0, ndcg@3=1.0 (n_rel=1)
    # user 1: top3 = [3, 0, 2], targets {3, 4}: only item3 hits, at rank 0
    #   recall@3 = 1/2 = 0.5
    #   idcg (n_rel=2, first 2 weights) = 1/log2(2) + 1/log2(3)
    #   dcg = 1/log2(2) (only rank-0 hit contributes)
    idcg_u1 = 1 / math.log2(2) + 1 / math.log2(3)
    dcg_u1 = 1 / math.log2(2)
    ndcg_u1 = dcg_u1 / idcg_u1
    assert m["recall@3"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["ndcg@3"] == pytest.approx((1.0 + ndcg_u1) / 2)


def test_masked_training_item_never_wins_despite_highest_raw_score(toy_data, toy_scores):
    # user 0's item 0 has the highest raw score (10.0) but is a training item
    # -- it must never appear as the top-ranked recommendation.
    m = evaluate(_score_fn(toy_scores), toy_data, topks=(1,))
    # If item 0 leaked through the mask, user 0 would score a miss at k=1
    # (item 0 isn't in test={2}), dragging recall@1 below what we assert above.
    assert m["recall@1"] == pytest.approx((1.0 + 0.5) / 2)
