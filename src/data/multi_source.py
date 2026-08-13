"""V2-0.4: multi-source candidate-recall PREVIEW analysis.

Answers one question: is fusing multiple recall sources worth the Phase 2
engineering investment, or does the strongest single source already cover
almost everything a union would? Uses Track A checkpoints (LightGCN,
Spatial-LightGCN, SASRec -- all trained on the FULL official train split,
same target-leakage caveat as everywhere else in the v1 pipeline) plus the
two non-learned, leakage-free-by-construction baselines (Popularity,
ItemCF). MF-BPR is excluded: no local checkpoint survives (see
experiments/results/summary.csv's notes column -- recovered from a Colab
console log, not a saved checkpoint).

PREVIEW, not a final result: uses the same M6-style single-target-per-user
leave-last-out split (src/data/ranking_dataset.py) and inherits its
target-leakage caveat. Numbers here must never be quoted as final candidate
Recall figures -- see docs/02_samples_features.md's known-simplification #1
for the underlying caveat this preview also inherits.
"""

import numpy as np
import scipy.sparse as sp


def popularity_score_fn(data):
    pop = np.asarray(data.R.sum(axis=0)).flatten()

    def score_fn(user_ids):
        return np.tile(pop, (len(user_ids), 1))

    return score_fn


def itemcf_score_fn(data, top_neighbors: int = 200, chunk: int = 2000):
    """Mirrors scripts/run_baselines.py's ItemCF exactly (duplicated, not
    imported, to keep this preview's model-loading code self-contained and
    independent of that script's own CLI/side effects)."""
    R = data.R
    deg = np.asarray(R.sum(axis=0)).flatten()
    inv_sqrt = np.power(deg, -0.5, out=np.zeros_like(deg), where=deg > 0)

    rows, cols, vals = [], [], []
    Rt = R.T.tocsr()
    for start in range(0, data.n_items, chunk):
        end = min(start + chunk, data.n_items)
        co = (Rt[start:end] @ R).toarray()
        sim = co * inv_sqrt[start:end, None] * inv_sqrt[None, :]
        sim[np.arange(end - start), np.arange(start, end)] = 0.0
        k = min(top_neighbors, sim.shape[1] - 1)
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

    return score_fn


def load_sasrec_with_prefix_scoring(run_name: str, data, prefix_targets: dict, device: str = "cpu"):
    """Loads a SASRec checkpoint the same way src/models/registry.py does
    (same config.json, same best.pt), but returns a score_fn built by
    sasrec_prefix_score_fn() below instead of registry's generic one --
    duplicated rather than importing registry.py's internals, to keep this
    preview module self-contained (mirrors item_features_v2.py's rationale
    for not modifying shared v1 files)."""
    import json

    import torch

    from src.data.sequences import SequenceData
    from src.models.sasrec import SASRec
    from src.utils.common import ROOT

    cfg = json.loads((ROOT / "experiments" / "logs" / run_name / "config.json").read_text())
    mc = cfg["model"]
    seq_data = SequenceData(
        ROOT / "data" / "processed" / "train_sequences.pkl",
        n_items=data.n_items, train_sets=data._train_sets, max_len=mc["max_len"],
    )
    model = SASRec(data.n_items, dim=mc["embedding_dim"], max_len=mc["max_len"],
                   n_blocks=mc["n_blocks"], n_heads=mc["n_heads"], dropout=mc["dropout"])
    state = torch.load(ROOT / "experiments" / "logs" / run_name / "best.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return sasrec_prefix_score_fn(model, seq_data, prefix_targets, device)


def sasrec_prefix_score_fn(model, seq_data, prefix_targets: dict, device: str = "cpu"):
    """SASRec-specific score_fn that conditions on `prefix_targets[u]["prefix_items"]`
    (M6's prefix, target EXCLUDED) -- NOT src/models/registry.py's generic
    sasrec score_fn, which conditions on SequenceData.eval_inputs()'s own
    `self.seqs` lookup (the FULL official-train sequence, correct for
    official-test evaluation but wrong here).

    Why this matters: M6's target_item is literally the last item of that
    same full official-train sequence, so the generic score_fn's input
    window already contains target_item as its most recent token. Under
    causal self-attention a query position can attend to itself, so the
    hidden state used to score "what comes next" is heavily correlated with
    target_item's OWN embedding -- inflating its predicted score toward
    certainty regardless of genuine next-item preference. This was caught
    empirically: candidate Recall@200 for SASRec came out at 99.997% using
    the generic score_fn, wildly inconsistent with SASRec's own
    properly-evaluated full-ranking Recall@20=0.126 (far weaker than
    LightGCN's 0.177) -- a huge gap that had no explanation other than a
    methodology bug in how the input window was built, not a genuine result.
    """
    import torch

    def score_fn(user_ids):
        model.eval()
        with torch.no_grad():
            B = len(user_ids)
            inp = np.full((B, seq_data.max_len), seq_data.pad, dtype=np.int64)
            length = np.zeros(B, dtype=np.int64)
            for r, u in enumerate(user_ids):
                items = prefix_targets[int(u)]["prefix_items"][-seq_data.max_len:]
                inp[r, :len(items)] = items
                length[r] = len(items)
            inp_t = torch.as_tensor(inp, device=device)
            length_t = torch.as_tensor(length, device=device)
            return model.full_scores(inp_t, length_t).cpu().numpy()

    return score_fn


def recall_at_k(prefix_targets: dict, candidates: dict, k: int) -> float:
    """`candidates`: dict[user] -> ranked np.ndarray of item ids (best first)."""
    hits = sum(1 for u, pt in prefix_targets.items() if pt["target_item"] in candidates[u][:k])
    return hits / len(prefix_targets)


def union_recall_at_k(prefix_targets: dict, candidates_by_source: dict[str, dict], k: int) -> float:
    hits = 0
    for u, pt in prefix_targets.items():
        target = pt["target_item"]
        if any(target in cands[u][:k] for cands in candidates_by_source.values()):
            hits += 1
    return hits / len(prefix_targets)


def unique_hit_counts(prefix_targets: dict, candidates_by_source: dict[str, dict], k: int) -> dict[str, int]:
    """How many targets were found by exactly one source (not by any other) at this K."""
    counts = {src: 0 for src in candidates_by_source}
    for u, pt in prefix_targets.items():
        target = pt["target_item"]
        hit_sources = [src for src, cands in candidates_by_source.items() if target in cands[u][:k]]
        if len(hit_sources) == 1:
            counts[hit_sources[0]] += 1
    return counts


def pairwise_jaccard_overlap(candidates_by_source: dict[str, dict], k: int, users) -> dict[tuple[str, str], float]:
    """Mean Jaccard overlap of top-K candidate sets between every pair of
    sources, averaged over `users` -- low overlap means the sources are
    finding genuinely different candidates, not just re-discovering the
    same ones under a different score."""
    sources = list(candidates_by_source)
    out = {}
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            vals = []
            for u in users:
                sa = set(candidates_by_source[a][u][:k].tolist())
                sb = set(candidates_by_source[b][u][:k].tolist())
                union = sa | sb
                vals.append(len(sa & sb) / len(union) if union else 0.0)
            out[(a, b)] = float(np.mean(vals))
    return out
