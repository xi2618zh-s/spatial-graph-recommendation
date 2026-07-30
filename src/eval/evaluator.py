"""Full-ranking evaluation protocol (the NGCF/LightGCN standard).

For every test user: score ALL items, mask items seen in training, take top-K,
compute Recall@K and NDCG@K against the held-out test items. No sampled
negatives — sampled evaluation is known to be biased (Krichene & Rendle, 2020).

`score_fn(user_ids: np.ndarray) -> np.ndarray[batch, n_items]` lets the same
evaluator serve matrix-factorization models, GNNs, ItemCF, and popularity.
"""

import numpy as np


def _dcg_weights(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def evaluate(score_fn, data, topks=(10, 20, 50), batch_size=2048) -> dict:
    max_k = max(topks)
    w = _dcg_weights(max_k)
    test_users = np.array([u for u, items in data.test.items() if items])
    agg = {f"recall@{k}": 0.0 for k in topks} | {f"ndcg@{k}": 0.0 for k in topks}

    for start in range(0, len(test_users), batch_size):
        batch = test_users[start:start + batch_size]
        scores = np.asarray(score_fn(batch), dtype=np.float32)
        # mask training items so the model is only judged on unseen items
        for r, u in enumerate(batch):
            scores[r, data.train.get(u, [])] = -np.inf
        # top-K via argpartition then exact sort of the head
        part = np.argpartition(-scores, max_k, axis=1)[:, :max_k]
        row_idx = np.arange(len(batch))[:, None]
        order = np.argsort(-scores[row_idx, part], axis=1)
        topk_items = part[row_idx, order]  # (batch, max_k), best first

        for r, u in enumerate(batch):
            truth = set(data.test[u])
            hits = np.fromiter(
                (1.0 if it in truth else 0.0 for it in topk_items[r]),
                dtype=np.float32, count=max_k,
            )
            for k in topks:
                n_rel = len(truth)
                agg[f"recall@{k}"] += hits[:k].sum() / n_rel
                idcg = w[: min(k, n_rel)].sum()
                agg[f"ndcg@{k}"] += (hits[:k] * w[:k]).sum() / idcg

    return {m: float(v) / len(test_users) for m, v in agg.items()}