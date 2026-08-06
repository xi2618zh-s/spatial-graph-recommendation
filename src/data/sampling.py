"""Layered negative sampling + full row assembly for the M6 ranking dataset.

Four negative sources (see docs/02_samples_features.md for the full
rationale and false-negative risk of each):

  random_negative       uniform draw from items the user never interacted
                         with in official train -- easy negative.
  popularity_negative   a globally popular item (top `pop_top_frac` by
                         prefix-universe interaction count) the user did not
                         visit -- exposes the popularity confound.
  recall_hard_negative   a candidate from the frozen recall model's own
                         top-`max_k` that is NOT the target -- the closest
                         analogue to a real industrial ranking hard negative.
  geo_hard_negative      an item near the user's prefix activity center that
                         was not the target -- diagnostic only. Being nearby
                         and unvisited is NOT evidence of dislike (implicit
                         feedback has no exposure signal); treat with the
                         same caution as recall_hard_negative, or more.

Every sampled item — regardless of source — gets the SAME feature row built
from the SAME user/item/cross/context feature functions, so the ranker in M7
cannot trivially tell sources apart from feature availability alone.
"""

import numpy as np
from sklearn.neighbors import BallTree

from src.features.row_builder import build_full_row, build_user_context


def build_popularity_pool(item_pop: np.ndarray, top_frac: float = 0.05) -> np.ndarray:
    nonzero = np.where(item_pop > 0)[0]
    n_top = max(1, int(round(len(nonzero) * top_frac)))
    order = nonzero[np.argsort(-item_pop[nonzero])]
    return order[:n_top]


def build_geo_tree(coords: np.ndarray):
    valid = ~np.isnan(coords[:, 0])
    idx_valid = np.where(valid)[0]
    tree = BallTree(coords[valid], metric="haversine") if len(idx_valid) else None
    return tree, idx_valid


def _sample_excluding(rng, pool: np.ndarray, exclude: set, n: int) -> list[int]:
    if len(pool) == 0:
        return []
    out, tries = [], 0
    max_tries = n * 50 + 100
    while len(out) < n and tries < max_tries:
        cand = int(pool[rng.integers(0, len(pool))])
        if cand not in exclude and cand not in out:
            out.append(cand)
        tries += 1
    return out


def sample_geo_negatives(rng, tree: BallTree, idx_valid: np.ndarray,
                         center_lat: float, center_lon: float, exclude: set,
                         n: int, pool_k: int = 50) -> list[int]:
    if tree is None or np.isnan(center_lat) or np.isnan(center_lon):
        return []
    k = min(pool_k, len(idx_valid))
    _, nbr = tree.query(
        np.radians([[center_lat, center_lon]]), k=k
    )
    pool = idx_valid[nbr[0]]
    candidates = [int(it) for it in pool if it not in exclude]
    rng.shuffle(candidates)
    return candidates[:n]


def build_samples_for_batch(
    users: np.ndarray, scores: np.ndarray, prefix_targets: dict[int, dict],
    max_k: int, item_pop: np.ndarray, item_feature_store, pop_pool: np.ndarray,
    geo_tree, geo_idx_valid, split: dict[int, str], seed: int,
    n_easy: int, n_pop: int, n_hard: int, n_geo: int,
) -> list[dict]:
    """`scores` is the already-prefix-masked (user, item) score matrix for
    this batch, straight from ranking_dataset.generate_candidates' inner
    loop -- reused here so recall_score is available for every sampled item,
    not just the ones that made the top-K candidate cut."""
    rows = []
    row_idx = np.arange(len(users))[:, None]
    k = min(max_k, scores.shape[1] - 1)
    part = np.argpartition(-scores, k, axis=1)[:, :max_k]
    order = np.argsort(-scores[row_idx, part], axis=1)
    topk = part[row_idx, order]

    for r, u in enumerate(users):
        u = int(u)
        pt = prefix_targets[u]
        rng = np.random.default_rng((seed, u))  # per-user determinism, order-independent
        prefix_set = set(pt["prefix_items"])
        target = pt["target_item"]
        query_ts = pt["target_ts"]
        candidates = topk[r]
        rank_of = {int(it): i for i, it in enumerate(candidates)}

        ufeat, ctxfeat = build_user_context(
            pt["prefix_items"], pt["prefix_ts"], item_pop, item_feature_store.coords, query_ts
        )
        exclude_all = prefix_set | {target}

        def make_row(item_id: int, label: int, source: str) -> dict:
            return build_full_row(
                u, item_id, label, source, split[u], query_ts,
                score=scores[r, item_id], rank=rank_of.get(item_id),
                ufeat=ufeat, ctxfeat=ctxfeat, item_store=item_feature_store,
            )

        target_rank = rank_of.get(target)
        if target_rank is not None:
            rows.append(make_row(target, 1, "positive"))

        hard_pool = [int(it) for it in candidates if int(it) != target]
        for it in _sample_excluding(rng, np.array(hard_pool), set(), n_hard) if hard_pool else []:
            rows.append(make_row(it, 0, "recall_hard_negative"))

        for it in _sample_excluding(rng, pop_pool, exclude_all, n_pop):
            rows.append(make_row(it, 0, "popularity_negative"))

        for it in _sample_excluding(rng, np.arange(item_feature_store.n_items), exclude_all, n_easy):
            rows.append(make_row(it, 0, "random_negative"))

        for it in sample_geo_negatives(
            rng, geo_tree, geo_idx_valid, ufeat["user_center_lat"], ufeat["user_center_lon"],
            exclude_all, n_geo,
        ):
            rows.append(make_row(it, 0, "geo_hard_negative"))

    return rows
