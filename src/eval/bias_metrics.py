"""M8 business-proxy / popularity-bias metrics (PROJECT_HANDOFF_V2.md §M8).

Every metric here is an OFFLINE PROXY over the top-K recommendation lists
this project actually produces — none of them are CTR/CVR/GMV, and none of
them should be reported without that qualifier (see docs/04_business_slices.md).
"""

import numpy as np

from src.data.spatial_graph import EARTH_RADIUS_KM


def catalog_coverage(top_k_lists: dict[int, list[int]], n_items: int) -> float:
    recommended = set()
    for items in top_k_lists.values():
        recommended.update(items)
    return len(recommended) / n_items


def effective_user_coverage(top_k_lists: dict[int, list[int]], k: int) -> float:
    """Fraction of users who actually received a FULL k-item list — i.e. had
    at least k valid candidates to rank. A user with fewer than k candidates
    is a serving-completeness gap, not a ranking-quality one, so it is kept
    separate from Recall/NDCG@k."""
    if not top_k_lists:
        return 0.0
    full = sum(1 for items in top_k_lists.values() if len(items) >= k)
    return full / len(top_k_lists)


def tail_exposure_share(top_k_lists: dict[int, list[int]], item_bucket: np.ndarray) -> float:
    total, tail = 0, 0
    for items in top_k_lists.values():
        for it in items:
            total += 1
            if item_bucket[it] == "tail":
                tail += 1
    return tail / total if total else 0.0


def mean_neg_log_popularity(top_k_lists: dict[int, list[int]], item_pop: np.ndarray) -> float:
    vals = [-np.log1p(item_pop[it]) for items in top_k_lists.values() for it in items]
    return float(np.mean(vals)) if vals else float("nan")


def average_recommendation_popularity(top_k_lists: dict[int, list[int]],
                                      item_pop: np.ndarray) -> float:
    vals = [item_pop[it] for items in top_k_lists.values() for it in items]
    return float(np.mean(vals)) if vals else float("nan")


def popularity_lift(arp: float, catalog_mean_pop: float) -> float:
    """ARP relative to the catalog's own mean popularity; 1.0 = recommends
    at the same average popularity as picking items uniformly at random,
    >1.0 = systematically favors popular items."""
    return arp / catalog_mean_pop if catalog_mean_pop else float("nan")


def exposure_gini(top_k_lists: dict[int, list[int]], n_items: int) -> float:
    """Gini coefficient of how many times each item is exposed across all
    lists; 0 = every item exposed equally often, 1 = all exposure on one item."""
    counts = np.zeros(n_items, dtype=np.int64)
    for items in top_k_lists.values():
        for it in items:
            counts[it] += 1
    if counts.sum() == 0:
        return float("nan")
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cum = np.cumsum(sorted_counts)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def recommendation_distance_stats(distances_km: np.ndarray) -> dict:
    valid = distances_km[~np.isnan(distances_km)]
    if len(valid) == 0:
        return {"n": 0}
    return {
        "n": int(len(valid)),
        "mean_km": float(valid.mean()),
        "p50_km": float(np.percentile(valid, 50)),
        "p90_km": float(np.percentile(valid, 90)),
        "missing_coord_share": float(1 - len(valid) / len(distances_km)),
    }


def list_internal_diversity_km(top_k_lists: dict[int, list[int]], coords: np.ndarray) -> float:
    """Mean pairwise haversine distance among the items in each user's own
    list, averaged over users -- a diversity/redundancy proxy, reported
    alongside accuracy, never as an optimization target on its own."""
    per_user_means = []
    for items in top_k_lists.values():
        latlon = coords[items]
        valid = ~np.isnan(latlon[:, 0])
        pts = latlon[valid]
        if len(pts) < 2:
            continue
        lat, lon = pts[:, 0], pts[:, 1]
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
        d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        iu = np.triu_indices(len(pts), k=1)
        if len(iu[0]) == 0:
            continue
        per_user_means.append(float(d[iu].mean()))
    return float(np.mean(per_user_means)) if per_user_means else float("nan")
