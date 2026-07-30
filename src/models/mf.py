"""Matrix Factorization trained with BPR loss (Rendle et al., 2009)."""

import torch
import torch.nn as nn


class MFBPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self):
        """No graph propagation in plain MF; interface shared with LightGCN."""
        return self.user_emb.weight, self.item_emb.weight

    def bpr_loss(self, users, pos, neg):
        ue, ie = self.propagate()
        u, p, n = ue[users], ie[pos], ie[neg]
        pos_s = (u * p).sum(-1)
        neg_s = (u * n).sum(-1)
        loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-10).mean()
        # L2 on the *ego* embeddings of the sampled triple (LightGCN convention)
        reg = 0.5 * (
            self.user_emb(users).norm(2).pow(2)
            + self.item_emb(pos).norm(2).pow(2)
            + self.item_emb(neg).norm(2).pow(2)
        ) / users.shape[0]
        return loss, reg

    @torch.no_grad()
    def full_scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        ue, ie = self.propagate()
        return ue[user_ids] @ ie.T
