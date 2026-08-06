"""Shared row-assembly used by BOTH training-sample construction (M6,
src/data/sampling.py) and end-to-end candidate scoring (M7,
src/eval/pipeline_evaluator.py) — one code path, so a ranker trained on M6
rows sees features computed exactly the same way at evaluation time."""

import numpy as np

from src.features.context_features import context_feature_row
from src.features.cross_features import cross_feature_row
from src.features.user_features import user_feature_row


def build_user_context(prefix_items: list[int], prefix_ts: list[int],
                       item_pop: np.ndarray, coords: np.ndarray, query_ts: int) -> tuple[dict, dict]:
    """Per-user features that do not depend on which item is being scored."""
    ufeat = user_feature_row(prefix_items, prefix_ts, item_pop, coords)
    ctxfeat = context_feature_row(query_ts, prefix_ts[-1])
    return ufeat, ctxfeat


def build_item_cross_row(item_id: int, item_store, query_ts: int, score: float,
                         rank: int | None, ufeat: dict) -> dict:
    """Per-(user, item) features: item stats + cross features referencing
    this specific user's context (already computed in `ufeat`)."""
    ifeat = item_store.row(item_id, query_ts)
    item_lat, item_lon = item_store.coords[item_id]
    item_lat = float(np.degrees(item_lat)) if not np.isnan(item_lat) else np.nan
    item_lon = float(np.degrees(item_lon)) if not np.isnan(item_lon) else np.nan
    cfeat = cross_feature_row(
        recall_score=score,
        user_center_lat=ufeat["user_center_lat"], user_center_lon=ufeat["user_center_lon"],
        item_lat=item_lat, item_lon=item_lon, candidate_rank=rank,
    )
    row = dict(ifeat)
    row.update(cfeat)
    return row


def build_full_row(user_id: int, item_id: int, label: int, source: str, split: str,
                   query_ts: int, score: float, rank: int | None,
                   ufeat: dict, ctxfeat: dict, item_store) -> dict:
    row = {"user_id": user_id, "item_id": item_id, "label": label,
          "negative_source": source, "split": split, "query_ts": query_ts}
    row.update(ufeat)
    row.update(build_item_cross_row(item_id, item_store, query_ts, score, rank, ufeat))
    row.update(ctxfeat)
    return row
