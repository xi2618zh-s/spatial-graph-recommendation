"""Point-in-time ranking dataset construction (M6).

For every user, the LAST check-in in their official-train, time-ordered
sequence becomes a held-out validation target; everything before it is the
"prefix" — the only history a recall/ranking model is allowed to see for
that user. Official `data/gowalla/test.txt` is never read here; it stays
sealed for the final full-ranking evaluation (see PROJECT_HANDOFF_V2.md §0.2
rule 5). This mirrors a standard leave-one-out next-item protocol layered
*inside* the official train split, not a replacement for it.

Candidate generation masks only each user's own prefix items — NOT the full
official train set — because the held-out target must remain a possible
candidate; that is the entire point of measuring candidate Recall@K.
"""

from pathlib import Path

import numpy as np


def load_timestamped_sequences(path: str | Path) -> dict[int, list[tuple[int, int]]]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def build_prefix_targets(sequences_ts: dict[int, list[tuple[int, int]]],
                         min_history: int = 5) -> dict[int, dict]:
    """Leave-last-out per user.

    Users whose prefix (sequence length minus the held-out target) would be
    shorter than `min_history` are skipped entirely — both as a target AND
    as a source of "other users'" feature statistics elsewhere, since we
    only ever pass this function's output downstream.
    """
    out = {}
    for u, seq in sequences_ts.items():
        if len(seq) < min_history + 1:
            continue
        items = [it for it, _ in seq]
        ts = [t for _, t in seq]
        out[u] = {
            "prefix_items": items[:-1],
            "prefix_ts": ts[:-1],
            "target_item": items[-1],
            "target_ts": ts[-1],
        }
    return out


def train_val_user_split(users, val_frac: float = 0.15, seed: int = 2020) -> dict[int, str]:
    """User-level split for the future ranker's train/validation sets (M7).
    Deterministic given `seed`, independent of iteration order."""
    users = np.array(sorted(users))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    n_val = int(round(len(users) * val_frac))
    val_users = set(users[perm[:n_val]].tolist())
    return {int(u): ("val" if u in val_users else "train") for u in users}


def generate_candidates(score_fn, prefix_targets: dict[int, dict], max_k: int,
                        batch_size: int = 2048) -> dict[int, np.ndarray]:
    """Score all items per user, mask the user's own prefix items only, keep
    the top `max_k` (best first). Returns dict[user] -> np.ndarray[max_k]."""
    users = np.array(sorted(prefix_targets))
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(score_fn(batch), dtype=np.float32)
        for r, u in enumerate(batch):
            scores[r, prefix_targets[int(u)]["prefix_items"]] = -np.inf
        k = min(max_k, scores.shape[1] - 1)
        part = np.argpartition(-scores, k, axis=1)[:, :max_k]
        row_idx = np.arange(len(batch))[:, None]
        order = np.argsort(-scores[row_idx, part], axis=1)
        topk = part[row_idx, order]
        for r, u in enumerate(batch):
            out[int(u)] = topk[r]
    return out


def generate_candidates_with_scores(score_fn, prefix_targets: dict[int, dict], max_k: int,
                                    batch_size: int = 2048):
    """Same as `generate_candidates`, but also returns each candidate's raw
    score — needed by M7's end-to-end evaluator, which re-featurizes the
    full candidate list (not just the sampled training rows) and must be
    able to populate `cross_recall_score` for every one of them."""
    users = np.array(sorted(prefix_targets))
    items_out: dict[int, np.ndarray] = {}
    scores_out: dict[int, np.ndarray] = {}
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.asarray(score_fn(batch), dtype=np.float32)
        for r, u in enumerate(batch):
            scores[r, prefix_targets[int(u)]["prefix_items"]] = -np.inf
        k = min(max_k, scores.shape[1] - 1)
        part = np.argpartition(-scores, k, axis=1)[:, :max_k]
        row_idx = np.arange(len(batch))[:, None]
        order = np.argsort(-scores[row_idx, part], axis=1)
        topk = part[row_idx, order]
        topk_scores = scores[row_idx, part][row_idx, order]
        for r, u in enumerate(batch):
            items_out[int(u)] = topk[r]
            scores_out[int(u)] = topk_scores[r]
    return items_out, scores_out


def candidate_recall(prefix_targets: dict[int, dict], candidates: dict[int, np.ndarray],
                     ks: tuple[int, ...]) -> dict[int, float]:
    """Fraction of users whose held-out target appears in the top-K candidates."""
    users = list(prefix_targets)
    out = {}
    for k in ks:
        hits = sum(
            1 for u in users if prefix_targets[u]["target_item"] in candidates[u][:k]
        )
        out[k] = hits / len(users)
    return out
