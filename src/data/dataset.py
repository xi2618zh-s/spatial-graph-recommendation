"""Gowalla benchmark split loading, sparse matrices, and BPR sampling."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]


def _read_adj(path: Path) -> dict[int, list[int]]:
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            out[int(parts[0])] = [int(x) for x in parts[1:] if x]
    return out


class GowallaData:
    """Official LightGCN split. Users/items are already contiguous ids."""

    def __init__(self, split_dir: str | Path = ROOT / "data" / "gowalla"):
        split_dir = Path(split_dir)
        self.train = _read_adj(split_dir / "train.txt")
        self.test = _read_adj(split_dir / "test.txt")
        self.n_users = 1 + max(max(self.train), max(self.test))
        self.n_items = 1 + max(
            max((max(v) for v in self.train.values() if v), default=0),
            max((max(v) for v in self.test.values() if v), default=0),
        )
        rows = np.concatenate([np.full(len(v), u) for u, v in self.train.items()])
        cols = np.concatenate([np.asarray(v) for v in self.train.values()])
        self.R = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        self.train_pairs = np.stack([rows, cols], axis=1)  # (N, 2) for sampling
        self._train_sets = {u: set(v) for u, v in self.train.items()}
        print(f"GowallaData: {self.n_users} users, {self.n_items} items, "
              f"{self.R.nnz} train pairs, {sum(len(v) for v in self.test.values())} test pairs")

    def norm_adj(self) -> sp.coo_matrix:
        """Symmetric-normalized bipartite adjacency D^-1/2 A D^-1/2 (LightGCN)."""
        A = sp.bmat(
            [[None, self.R], [self.R.T, None]], format="csr", dtype=np.float32
        )
        deg = np.asarray(A.sum(axis=1)).flatten()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5, out=np.zeros_like(deg), where=deg > 0), 0.0)
        D = sp.diags(d_inv_sqrt)
        return (D @ A @ D).tocoo()

    def sample_bpr_batch(self, batch_size: int, rng: np.random.Generator):
        """Uniform interaction sampling + uniform negative sampling with rejection."""
        idx = rng.integers(0, len(self.train_pairs), size=batch_size)
        users = self.train_pairs[idx, 0]
        pos = self.train_pairs[idx, 1]
        neg = rng.integers(0, self.n_items, size=batch_size)
        for k in range(batch_size):  # rejection resample collisions
            u_set = self._train_sets[users[k]]
            while neg[k] in u_set:
                neg[k] = rng.integers(0, self.n_items)
        return users, pos, neg
