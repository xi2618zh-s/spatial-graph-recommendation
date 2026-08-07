"""SASRec (Kang & McAuley, ICDM 2018): causal self-attention over the user's
check-in sequence; the hidden state at the last real position scores all items
by dot product with the shared item embedding table.

Implementation notes:
- Shared item embedding for input and output (standard SASRec).
- Row `n_items` is the padding embedding (padding_idx, frozen at zero).
- Causal mask + key_padding_mask with RIGHT-padded inputs (see sequences.py
  for why right-padding avoids NaN rows).
"""

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, max_len: int = 50,
                 n_blocks: int = 2, n_heads: int = 1, dropout: float = 0.2):
        super().__init__()
        self.n_items, self.pad, self.max_len = n_items, n_items, max_len
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.pos_emb = nn.Embedding(max_len, dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[n_items].zero_()
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.ln = nn.LayerNorm(dim)
        causal = torch.triu(
            torch.full((max_len, max_len), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) right-padded with pad_id. Returns (B, L, dim).

        Right-padding guarantees every REAL (non-pad) query position has at
        least itself as a valid causal key, so rows with >=1 real item never
        produce NaN. A row that is entirely padding (length 0) is the one
        case this does not cover -- every key is masked out for every query
        position, so softmax(-inf) -> NaN internally here. `full_scores()`
        below handles that case by overriding those rows to zero AFTER
        calling this method, not by preventing the NaN in this method's own
        output (see tests/test_sasrec_padding.py). Confirm length > 0 before
        relying on this method's output directly.
        """
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        h = self.encoder(
            x, mask=self.causal_mask, src_key_padding_mask=(seq == self.pad)
        )
        return self.ln(h)

    def training_loss(self, inp, tgt, neg) -> torch.Tensor:
        """Masked BCE over (positive next item, sampled negative) per position."""
        h = self.encode(inp)                              # (B, L, d)
        pos_logit = (h * self.item_emb(tgt)).sum(-1)      # (B, L)
        neg_logit = (h * self.item_emb(neg)).sum(-1)
        mask = (tgt != self.pad).float()
        bce = nn.functional.binary_cross_entropy_with_logits
        loss = (
            bce(pos_logit, torch.ones_like(pos_logit), reduction="none") * mask
            + bce(neg_logit, torch.zeros_like(neg_logit), reduction="none") * mask
        )
        return loss.sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def full_scores(self, seq: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Score all items from the last real position. Empty sequences -> zeros."""
        h = self.encode(seq)
        idx = (length - 1).clamp(min=0)
        last = h[torch.arange(seq.size(0), device=seq.device), idx]  # (B, d)
        scores = last @ self.item_emb.weight[: self.n_items].T
        scores[length == 0] = 0.0
        return scores
