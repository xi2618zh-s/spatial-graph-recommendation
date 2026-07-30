"""LightGCN (He et al., SIGIR 2020): linear propagation on the normalized
user-item bipartite graph, final representation = mean of all layer outputs.

Inherits embeddings, BPR loss, and scoring from MFBPR — the ONLY difference
is `propagate()`, which makes the MF-vs-LightGCN comparison a clean ablation
of graph propagation itself.
"""

import numpy as np
import scipy.sparse as sp
import torch

from .mf import MFBPR


def _to_torch_sparse(m: sp.coo_matrix) -> torch.Tensor:
    idx = torch.from_numpy(np.vstack([m.row, m.col])).long()
    val = torch.from_numpy(m.data).float()
    return torch.sparse_coo_tensor(idx, val, m.shape).coalesce()


class LightGCN(MFBPR):
    def __init__(self, n_users, n_items, dim=64, n_layers=3, norm_adj=None):
        super().__init__(n_users, n_items, dim)
        self.n_users, self.n_layers = n_users, n_layers
        self.register_buffer("A", _to_torch_sparse(norm_adj))

    def propagate(self):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        out = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.A, x)
            out.append(x)
        final = torch.stack(out, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users:]
