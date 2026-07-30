"""Non-learned baselines: Popularity and ItemCF (cosine, top-N neighbors).

Both run on CPU. ItemCF computes the item-item cosine similarity in column
chunks (dense chunk of ~2k items x 41k items ≈ 330 MB) and keeps only the
top-200 neighbors per item, so peak RAM stays under ~2 GB.

Usage:
    python scripts/run_baselines.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

from src.data.dataset import GowallaData
from src.eval.evaluator import evaluate
from src.utils.common import Timer, append_result

TOP_NEIGHBORS = 200
CHUNK = 2000


def popularity(data: GowallaData) -> dict:
    pop = np.asarray(data.R.sum(axis=0)).flatten()

    def score_fn(user_ids):
        return np.tile(pop, (len(user_ids), 1))

    with Timer("evaluate Popularity"):
        return evaluate(score_fn, data)


def item_cf(data: GowallaData) -> dict:
    R = data.R  # (n_users, n_items) binary
    deg = np.asarray(R.sum(axis=0)).flatten()  # item degrees
    inv_sqrt = np.power(deg, -0.5, out=np.zeros_like(deg), where=deg > 0)

    rows, cols, vals = [], [], []
    with Timer("ItemCF: chunked item-item cosine + top-N truncation"):
        Rt = R.T.tocsr()
        for start in range(0, data.n_items, CHUNK):
            end = min(start + CHUNK, data.n_items)
            # co-occurrence counts for this chunk of items vs all items
            co = (Rt[start:end] @ R).toarray()  # (chunk, n_items)
            sim = co * inv_sqrt[start:end, None] * inv_sqrt[None, :]
            sim[np.arange(end - start), np.arange(start, end)] = 0.0  # no self
            k = min(TOP_NEIGHBORS, sim.shape[1] - 1)
            top = np.argpartition(-sim, k, axis=1)[:, :k]
            r = np.repeat(np.arange(start, end), k)
            c = top.flatten()
            v = sim[np.arange(end - start)[:, None], top].flatten()
            keep = v > 0
            rows.append(r[keep]); cols.append(c[keep]); vals.append(v[keep])
    S = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(data.n_items, data.n_items),
    )

    def score_fn(user_ids):
        return (R[user_ids] @ S).toarray()

    with Timer("evaluate ItemCF"):
        return evaluate(score_fn, data)


def main() -> None:
    data = GowallaData()
    m = popularity(data)
    print("Popularity:", {k: round(v, 4) for k, v in m.items()})
    append_result("popularity_gowalla", "popularity", m)

    m = item_cf(data)
    print("ItemCF:", {k: round(v, 4) for k, v in m.items()})
    append_result("itemcf_gowalla", "itemcf",
                  m, notes=f"cosine top{TOP_NEIGHBORS}")


if __name__ == "__main__":
    main()
