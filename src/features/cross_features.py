"""User-POI cross features (§2.4).

`recall_score` comes directly from the frozen Spatial-LightGCN checkpoint
used for candidate generation — for this bilinear-scoring model family the
"embedding dot product" feature in the design doc IS the recall score, so
only one column is kept (see docs/02_samples_features.md).

Known limitation: the frozen recall model was trained on the FULL official
train split, including every user's held-out validation target. Its score
for a user's own target is therefore not leakage-free the way the user/item
features above are — flagged prominently in docs/02_samples_features.md
rather than silently treated as clean.
"""

import numpy as np

from src.features.user_features import haversine_km


def cross_feature_row(recall_score: float, user_center_lat: float, user_center_lon: float,
                      item_lat: float, item_lon: float, candidate_rank: int | None) -> dict:
    has_center = not (np.isnan(user_center_lat) or np.isnan(user_center_lon))
    has_item_coord = not (np.isnan(item_lat) or np.isnan(item_lon))
    if has_center and has_item_coord:
        dist_km = haversine_km(
            np.radians(user_center_lat), np.radians(user_center_lon),
            np.radians(item_lat), np.radians(item_lon),
        )
    else:
        dist_km = np.nan
    return {
        "cross_recall_score": float(recall_score),
        "cross_dist_to_center_km": float(dist_km) if not np.isnan(dist_km) else np.nan,
        "cross_dist_missing": not (has_center and has_item_coord),
        "cross_candidate_rank": candidate_rank if candidate_rank is not None else -1,
        "cross_in_candidates": candidate_rank is not None,
    }
