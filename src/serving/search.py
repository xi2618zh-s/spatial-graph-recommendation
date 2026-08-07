"""Masked ANN search shared by the M9 benchmark and the FastAPI service.

FAISS indices have no notion of "exclude these ids" built in, so seen-item
masking (the same masking src/eval/evaluator.py applies for full-ranking
evaluation: a user's own official-train items) is done by over-fetching a
larger candidate pool from the index than requested, filtering client-side,
and retrying with a bigger pool on the rare occasion filtering leaves fewer
than `k` items -- the standard pattern for ANN indices that can't mask.
"""

import numpy as np


def masked_search(index, query_emb: np.ndarray, exclude_ids: set, k: int,
                  over_fetch_factor: int = 4, max_attempts: int = 4):
    """query_emb: (dim,) or (1, dim). Returns (item_ids, scores), both length
    <= k, best first, with every id in `exclude_ids` removed."""
    q = query_emb.reshape(1, -1).astype("float32")
    fetch_k = min(index.ntotal, k * over_fetch_factor)
    for _ in range(max_attempts):
        scores, ids = index.search(q, fetch_k)
        ids, scores = ids[0], scores[0]
        keep = [(i, s) for i, s in zip(ids, scores) if i >= 0 and int(i) not in exclude_ids]
        if len(keep) >= k or fetch_k >= index.ntotal:
            keep = keep[:k]
            return np.array([i for i, _ in keep]), np.array([s for _, s in keep])
        fetch_k = min(index.ntotal, fetch_k * over_fetch_factor)
    keep = keep[:k]
    return np.array([i for i, _ in keep]), np.array([s for _, s in keep])
