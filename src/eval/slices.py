"""M8 slice boundaries — all fixed from TRAIN only (PROJECT_HANDOFF_V2.md
§M8: "分桶边界只由 train 决定，并固化到 config"), then applied unchanged to
val-user reporting. Strict cold-start, near cold-start, and long-tail are
kept as three distinct concepts throughout — never merged into one bucket.
"""

import numpy as np


def user_activity_buckets(prefix_targets: dict[int, dict],
                          quantiles: tuple[float, float] = (1 / 3, 2 / 3)) -> tuple[dict, dict]:
    """Low/mid/high by prefix history length, tertile boundaries fixed over
    ALL eligible train users (not just val), per the "boundaries from train"
    rule. Returns (user -> bucket, boundaries)."""
    users = sorted(prefix_targets)
    lens = np.array([len(prefix_targets[u]["prefix_items"]) for u in users])
    q_lo, q_hi = np.quantile(lens, quantiles)
    buckets = {}
    for u, n in zip(users, lens):
        buckets[u] = "low" if n <= q_lo else ("high" if n > q_hi else "mid")
    return buckets, {"q_lo": float(q_lo), "q_hi": float(q_hi)}


def item_popularity_buckets(item_pop: np.ndarray, head_frac: float = 0.2,
                            tail_frac: float = 0.5) -> tuple[np.ndarray, dict]:
    """head = top `head_frac` of items BY INTERACTION COUNT RANK (not by
    equal item-count tertiles, which would hide how few items the "head"
    actually is relative to interaction mass); tail = bottom `tail_frac`.
    Items with zero prefix-universe interactions are folded into "tail" —
    they are exactly what "long tail" means here, not a separate bucket."""
    n = len(item_pop)
    order = np.argsort(-item_pop)  # most popular first
    bucket = np.full(n, "mid", dtype=object)
    n_head = int(round(n * head_frac))
    n_tail = int(round(n * tail_frac))
    bucket[order[:n_head]] = "head"
    bucket[order[-n_tail:]] = "tail"
    interactions_total = item_pop.sum()
    share = {}
    for b in ("head", "mid", "tail"):
        idx = np.where(bucket == b)[0]
        share[b] = {
            "n_items": int(len(idx)),
            "item_share": float(len(idx) / n),
            "interaction_share": float(item_pop[idx].sum() / interactions_total) if interactions_total else 0.0,
        }
    return bucket, share


def distance_buckets(distances_km: np.ndarray,
                     edges_km: tuple[float, ...] = (1, 5, 20, 100)) -> np.ndarray:
    """Fixed-edge distance buckets (km), labeled by the upper bound of each
    bin; NaN distances (missing coordinates) get their own bucket."""
    labels = np.full(len(distances_km), "nan", dtype=object)
    valid = ~np.isnan(distances_km)
    edges = list(edges_km) + [np.inf]
    bucket_names = [f"<{e}km" for e in edges_km] + [f">{edges_km[-1]}km"]
    idx = np.digitize(distances_km[valid], edges, right=True)
    labels[valid] = np.array(bucket_names)[np.clip(idx, 0, len(bucket_names) - 1)]
    return labels


def strict_cold_start_counts(data) -> dict:
    """Users/items in official test with ZERO official-train interactions.
    Distinct from "near cold-start" (low history/low popularity) below."""
    test_users, train_users = set(data.test.keys()), set(data.train.keys())
    test_items = {it for items in data.test.values() for it in items}
    train_items = {it for items in data.train.values() for it in items}
    return {
        "test_users": len(test_users),
        "zero_train_test_users": len(test_users - train_users),
        "distinct_test_items": len(test_items),
        "zero_train_test_items": len(test_items - train_items),
    }


def near_cold_start_flags(prefix_targets: dict[int, dict], item_pop: np.ndarray,
                          user_low_quantile: float = 1 / 3,
                          item_low_frac: float = 0.5) -> tuple[dict, np.ndarray]:
    """Low-history users / low-frequency items — reported as "near
    cold-start", never called "cold-start" outright (RISK_REGISTER.md)."""
    buckets, _ = user_activity_buckets(prefix_targets, (user_low_quantile, 2 / 3))
    low_history_users = {u: (b == "low") for u, b in buckets.items()}
    item_bucket, _ = item_popularity_buckets(item_pop, tail_frac=item_low_frac)
    low_freq_items = item_bucket == "tail"
    return low_history_users, low_freq_items
