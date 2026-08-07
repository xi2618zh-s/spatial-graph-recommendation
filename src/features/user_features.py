"""User-level features (§2.4), computed strictly from each user's own prefix
— no cross-user information, so these carry zero leakage risk by
construction (unlike the item-popularity group, which pools across users).

Category/spatial-preference entropy is listed as "if available" in the
design doc; Gowalla check-ins carry no POI category labels in this project,
so that feature is not computed (documented gap, not a silent omission).
"""

import numpy as np

from src.data.spatial_graph import EARTH_RADIUS_KM

RECENT_WINDOW_DAYS = 30


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """All args in radians."""
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def user_feature_row(prefix_items: list[int], prefix_ts: list[int],
                     item_pop: np.ndarray, coords: np.ndarray) -> dict:
    n = len(prefix_items)
    span_days = (prefix_ts[-1] - prefix_ts[0]) / 86400.0
    recent_cut = prefix_ts[-1] - RECENT_WINDOW_DAYS * 86400
    recent_count = sum(1 for t in prefix_ts if t >= recent_cut)
    avg_pop = float(np.mean([np.log1p(item_pop[it]) for it in prefix_items]))

    latlon = coords[prefix_items]
    valid = ~np.isnan(latlon[:, 0])
    coord_coverage = float(valid.mean())
    if valid.any():
        center_lat, center_lon = latlon[valid].mean(axis=0)
        radius_km = float(np.mean([
            haversine_km(center_lat, center_lon, lat, lon)
            for lat, lon in latlon[valid]
        ]))
    else:
        center_lat = center_lon = radius_km = np.nan

    return {
        "user_history_count": n,
        "user_unique_poi_count": len(set(prefix_items)),  # == n by construction; see module docstring
        "user_active_span_days": float(span_days),
        "user_recent_count_30d": recent_count,
        "user_avg_visited_log1p_popularity": avg_pop,
        "user_center_lat": float(np.degrees(center_lat)) if valid.any() else np.nan,
        "user_center_lon": float(np.degrees(center_lon)) if valid.any() else np.nan,
        "user_activity_radius_km": radius_km,
        "user_coord_coverage": coord_coverage,
    }
