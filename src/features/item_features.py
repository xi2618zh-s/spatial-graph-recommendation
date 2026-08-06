"""POI-level features, computed once over the prefix universe (§2.4).

All statistics here are aggregated over every user's PREFIX only — never a
held-out target — so no single user's own future interaction can inflate the
popularity/recency of the item they are about to visit. This is a
population-level simplification, not a fully per-user point-in-time cutoff:
other users' prefixes may chronologically extend past the querying user's own
cutoff. See docs/02_samples_features.md for why this trade-off was made and
what a stricter version would require.
"""

import numpy as np
from sklearn.neighbors import BallTree

from src.data.spatial_graph import EARTH_RADIUS_KM, load_coords


def compute_item_popularity(prefix_targets: dict[int, dict], n_items: int) -> np.ndarray:
    """Count of distinct users whose prefix contains each item. Gowalla
    sequences are de-duplicated per user at construction time (prepare_data.py),
    so this is simultaneously "interaction count" and "distinct-user count"
    and "bipartite graph degree" for the prefix-only graph -- one column, not
    three redundant ones."""
    counts = np.zeros(n_items, dtype=np.int64)
    for pt in prefix_targets.values():
        for it in pt["prefix_items"]:
            counts[it] += 1
    return counts


def compute_item_last_active(prefix_targets: dict[int, dict], n_items: int) -> np.ndarray:
    """Most recent prefix timestamp seen for each item, across all users'
    prefixes (unix seconds; NaN if the item never appears in any prefix)."""
    last = np.full(n_items, np.nan, dtype=np.float64)
    for pt in prefix_targets.values():
        for it, ts in zip(pt["prefix_items"], pt["prefix_ts"]):
            if np.isnan(last[it]) or ts > last[it]:
                last[it] = ts
    return last


def compute_item_local_density(coords: np.ndarray, radius_km: float = 1.0) -> np.ndarray:
    """Number of other POIs within `radius_km` (excludes self); NaN where the
    item has no known coordinates."""
    n = coords.shape[0]
    density = np.full(n, np.nan, dtype=np.float64)
    valid = ~np.isnan(coords[:, 0])
    idx_valid = np.where(valid)[0]
    if len(idx_valid) == 0:
        return density
    tree = BallTree(coords[valid], metric="haversine")
    counts = tree.query_radius(coords[valid], r=radius_km / EARTH_RADIUS_KM, count_only=True)
    density[idx_valid] = counts - 1  # exclude self
    return density


class ItemFeatureStore:
    """Bundles the three precomputed arrays + a per-item row lookup."""

    def __init__(self, prefix_targets: dict[int, dict], n_items: int,
                coords_csv: str, density_radius_km: float = 1.0):
        self.n_items = n_items
        self.popularity = compute_item_popularity(prefix_targets, n_items)
        self.last_active = compute_item_last_active(prefix_targets, n_items)
        self.coords = load_coords(coords_csv, n_items)
        self.density = compute_item_local_density(self.coords, density_radius_km)

    def row(self, item_id: int, query_ts: int) -> dict:
        pop = int(self.popularity[item_id])
        last = self.last_active[item_id]
        density = self.density[item_id]
        coord_missing = bool(np.isnan(self.coords[item_id, 0]))
        return {
            "item_log1p_popularity": float(np.log1p(pop)),
            "item_prefix_interactions": pop,
            "item_cold_in_prefix": pop == 0,  # never seen in ANY user's prefix
            "item_days_since_last_active": (
                float((query_ts - last) / 86400.0) if not np.isnan(last) else np.nan
            ),
            "item_last_active_missing": bool(np.isnan(last)),
            "item_local_density_1km": float(density) if not np.isnan(density) else np.nan,
            "item_coord_missing": coord_missing,
        }
