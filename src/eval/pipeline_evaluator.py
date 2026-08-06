"""M7: candidate -> ranking end-to-end evaluation.

Regenerates the FULL top-K candidate list per user from the frozen recall
model (same prefix-masking as M6) and builds a feature row for every single
candidate — not just the handful of sampled training negatives. A ranker
trained on M6's sampled rows must still be scored against the full pool it
would actually face at serving time, or Recall@20 is meaningless (with ~10
items per user, it would trivially be ~1.0).

Users whose target is not among their top-K candidates at all (a recall
miss) are kept in the evaluation with every candidate row labeled 0 — their
contribution to Recall/NDCG/MRR@20 is correctly 0 under every scoring
method, which is what makes these numbers bounded by Candidate Recall@K
rather than overstating what re-ranking alone can fix.
"""

import numpy as np
import pandas as pd

from src.features.row_builder import build_full_row, build_user_context


def build_eval_frame(user_ids, prefix_targets: dict[int, dict], cand_items: dict[int, np.ndarray],
                     cand_scores: dict[int, np.ndarray], item_pop: np.ndarray, item_store,
                     split: dict[int, str]) -> pd.DataFrame:
    """One row per (user, candidate) — every candidate kept, none sampled out."""
    rows = []
    for u in user_ids:
        pt = prefix_targets[u]
        ufeat, ctxfeat = build_user_context(
            pt["prefix_items"], pt["prefix_ts"], item_pop, item_store.coords, pt["target_ts"]
        )
        items, scores = cand_items[u], cand_scores[u]
        target = pt["target_item"]
        for rank, (it, sc) in enumerate(zip(items, scores)):
            it = int(it)
            label = 1 if it == target else 0
            rows.append(build_full_row(
                u, it, label, "eval_candidate", split[u], pt["target_ts"],
                score=float(sc), rank=rank, ufeat=ufeat, ctxfeat=ctxfeat, item_store=item_store,
            ))
    return pd.DataFrame(rows)


def ranking_metrics_from_frame(df: pd.DataFrame, score_col: str, k: int = 20) -> dict:
    """NDCG@k / Recall@k / MRR@k, grouped by user, ranking each user's
    candidate rows by `score_col` (descending). `df` need not be pre-sorted."""
    ndcg_w = 1.0 / np.log2(np.arange(2, k + 2))
    recall_hits, mrr_vals, ndcg_vals = [], [], []
    for _, g in df.groupby("user_id", sort=False):
        order = np.argsort(-g[score_col].to_numpy())
        labels = g["label"].to_numpy()[order]
        pos = np.flatnonzero(labels == 1)
        if len(pos) == 0:
            recall_hits.append(0.0)
            mrr_vals.append(0.0)
            ndcg_vals.append(0.0)
            continue
        rank = int(pos[0])  # 0-indexed
        hit = rank < k
        recall_hits.append(1.0 if hit else 0.0)
        mrr_vals.append(1.0 / (rank + 1) if hit else 0.0)
        ndcg_vals.append(float(ndcg_w[rank]) if hit else 0.0)
    return {
        f"recall@{k}": float(np.mean(recall_hits)),
        f"mrr@{k}": float(np.mean(mrr_vals)),
        f"ndcg@{k}": float(np.mean(ndcg_vals)),
        "n_users": len(recall_hits),
    }
