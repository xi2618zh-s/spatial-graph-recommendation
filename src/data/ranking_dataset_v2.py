"""Ranking V2: leakage-safe temporal protocol (see ENGINEERING_LOG.md / the
approved `ranking-v2` plan for the full design discussion).

Every user's official-train, time-ordered sequence is split into four
non-overlapping layers:

    H_inner | v0 | y_train | y_val

  H_inner   everything except the last 3 interactions -- the ONLY edges any
            retriever's inner-validation run trains on.
  v0        3rd-from-last interaction. Never a training edge for ANY final
            retriever checkpoint (R0 or R1) -- used exclusively to pick the
            fixed epoch budget E* via a throwaway inner-validation run, then
            reused unchanged as H_inner's own history once E* is frozen.
  y_train   2nd-from-last interaction -- the RANKING model's training target
            (M7-style: LR/GBDT/DeepFM/DIN all learn from this).
  y_val     last interaction -- the RANKING model's validation target
            (hyperparameter/checkpoint selection for the ranker).

Protocol (train -> validate -> freeze hyperparameters -> refit on
train+validation, applied to the RETRIEVER, not the ranker):

  1. Inner-validation run: train Spatial-LightGCN on H_inner only, evaluate
     periodically against v0 using the existing, UNMODIFIED
     src/train/bpr_trainer.py + src/eval/evaluator.py (via SnapshotGowallaData
     below, which duck-types GowallaData so those two shared files never need
     to change). Early stopping against v0 picks a fixed epoch budget E*.
     v0's value is never inspected beyond that -- it does not select between
     any other hyperparameter, and the run that produces this E* is
     discarded, not used as a candidate generator.
  2. R0: FRESH initialization, train Spatial-LightGCN on H_inner + v0 (=
     "Hcore") for EXACTLY E* epochs, no periodic evaluation/early-stopping
     against anything -- E* was already decided in step 1. Used to generate
     candidates for y_train.
  3. R1: FRESH initialization, train Spatial-LightGCN on H_inner + v0 +
     y_train for EXACTLY the same E* epochs. Used to generate candidates for
     y_val.

Neither R0 nor R1 ever trains on or early-stops against y_train, y_val, or
official test.txt -- see tests/test_ranking_v2_leakage.py for the behavioral
proof, not just this docstring's claim.
"""

import numpy as np
import scipy.sparse as sp


def build_v2_layers(sequences_ts: dict[int, list[tuple[int, int]]],
                    min_h_inner: int = 5) -> dict[int, dict]:
    """Split each user's (item, ts) sequence into H_inner/v0/y_train/y_val.
    Users whose H_inner would be shorter than `min_h_inner` are dropped
    entirely (matches src/data/ranking_dataset.py's min_history convention)."""
    out = {}
    for u, seq in sequences_ts.items():
        if len(seq) < min_h_inner + 3:  # H_inner + v0 + y_train + y_val
            continue
        items = [it for it, _ in seq]
        ts = [t for _, t in seq]
        out[u] = {
            "h_inner_items": items[:-3], "h_inner_ts": ts[:-3],
            "v0_item": items[-3], "v0_ts": ts[-3],
            "y_train_item": items[-2], "y_train_ts": ts[-2],
            "y_val_item": items[-1], "y_val_ts": ts[-1],
        }
    return out


def hcore_edges(layers: dict[int, dict]) -> dict[int, list[int]]:
    """H_inner + v0 -- R0's training edges, and the correct point-in-time
    exclusion mask when generating candidates for y_train (v0 has genuinely
    already happened by y_train's timestamp, even though step 1's
    inner-validation run never trained on it)."""
    return {u: L["h_inner_items"] + [L["v0_item"]] for u, L in layers.items()}


def hcore_plus_ytrain_edges(layers: dict[int, dict]) -> dict[int, list[int]]:
    """H_inner + v0 + y_train -- R1's training edges, and the exclusion mask
    for generating candidates for y_val."""
    return {u: L["h_inner_items"] + [L["v0_item"], L["y_train_item"]] for u, L in layers.items()}


def v0_targets(layers: dict[int, dict]) -> dict[int, list[int]]:
    """The shared, never-trained-on validation target for both the inner
    E*-selection run (trained on H_inner) and, as a harmless fixed-epoch
    trigger only, the final R0/R1 refits (see module docstring step 2-3)."""
    return {u: [L["v0_item"]] for u, L in layers.items()}


class SnapshotGowallaData:
    """Duck-typed drop-in for src/data/dataset.py::GowallaData, built from
    explicit edge dicts instead of reading train.txt/test.txt from disk --
    lets Track B retriever runs reuse train_bpr()/evaluate() completely
    unmodified. n_users/n_items are always the GLOBAL official split sizes,
    never inferred from which ids happen to appear in a particular
    snapshot's edges, so every V2 snapshot's embedding tables stay the same
    shape as the official split's.

    norm_adj()/sample_bpr_batch() logic is intentionally duplicated from
    GowallaData rather than refactored to share code: src/data/dataset.py is
    imported by the currently-running Colab ablation queue, and this file's
    entire reason for existing is to avoid ANY edit to that shared file
    while it's in use.
    """

    def __init__(self, train_edges: dict[int, list[int]], test_edges: dict[int, list[int]],
                n_users: int, n_items: int):
        self.train = train_edges
        self.test = test_edges
        self.n_users = n_users
        self.n_items = n_items
        nonempty = [(u, v) for u, v in train_edges.items() if v]
        if nonempty:
            rows = np.concatenate([np.full(len(v), u) for u, v in nonempty])
            cols = np.concatenate([np.asarray(v) for _, v in nonempty])
        else:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
        self.R = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
        )
        self.train_pairs = np.stack([rows, cols], axis=1)
        self._train_sets = {u: set(v) for u, v in train_edges.items()}

    def norm_adj(self) -> sp.coo_matrix:
        A = sp.bmat([[None, self.R], [self.R.T, None]], format="csr", dtype=np.float32)
        deg = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5, out=np.zeros_like(deg), where=deg > 0), 0.0)
        D = sp.diags(d_inv_sqrt)
        return (D @ A @ D).tocoo()

    def sample_bpr_batch(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.train_pairs), size=batch_size)
        users = self.train_pairs[idx, 0]
        pos = self.train_pairs[idx, 1]
        neg = rng.integers(0, self.n_items, size=batch_size)
        for k in range(batch_size):
            u_set = self._train_sets[users[k]]
            while neg[k] in u_set:
                neg[k] = rng.integers(0, self.n_items)
        return users, pos, neg
