"""M10: SASRec's right-padding + causal-mask design (src/models/sasrec.py
docstring: "avoids the all-masked-row NaN pathology of left-padding") has
never had a test that actually checks for NaNs, including the fully-empty
sequence edge case the code explicitly special-cases."""

import torch

from src.models.sasrec import SASRec


def _model():
    torch.manual_seed(0)
    return SASRec(n_items=12, dim=8, max_len=6, n_blocks=2, n_heads=2, dropout=0.1)


def test_encode_produces_no_nan_for_any_row_with_at_least_one_real_item():
    model = _model()
    model.eval()
    pad = model.pad
    # lengths: 4, 1 -- both have >=1 real item, so every query position has
    # at least itself as a valid causal key (see encode()'s docstring).
    seq = torch.tensor([
        [1, 2, 3, 4, pad, pad],
        [5, pad, pad, pad, pad, pad],
    ], dtype=torch.long)
    with torch.no_grad():
        h = model.encode(seq)
    assert not torch.isnan(h).any(), "encode() produced NaN -- likely an all-masked-row softmax"


def test_encode_of_a_fully_empty_row_is_the_one_documented_nan_case():
    """A row that is ENTIRELY padding has no valid key for any query
    position, so encode() itself does produce NaN there -- this is
    documented in encode()'s docstring, not silently true. full_scores()
    (tested below) is what actually guarantees a NaN-free contract by
    overriding length==0 rows; encode() alone does not."""
    model = _model()
    model.eval()
    pad = model.pad
    seq = torch.tensor([[pad, pad, pad, pad, pad, pad]], dtype=torch.long)
    with torch.no_grad():
        h = model.encode(seq)
    assert torch.isnan(h).all()


def test_full_scores_zero_for_empty_sequence_and_finite_otherwise():
    model = _model()
    model.eval()
    pad = model.pad
    seq = torch.tensor([
        [1, 2, 3, 4, pad, pad],
        [pad, pad, pad, pad, pad, pad],
    ], dtype=torch.long)
    length = torch.tensor([4, 0], dtype=torch.long)
    with torch.no_grad():
        scores = model.full_scores(seq, length)
    assert not torch.isnan(scores).any()
    assert torch.equal(scores[1], torch.zeros(model.n_items)), \
        "empty sequence must score as all-zero, per full_scores' documented contract"
    assert scores[0].abs().sum() > 0, "non-empty sequence should not trivially score all-zero"


def test_padding_embedding_row_stays_zero_after_a_training_step():
    model = _model()
    model.train()
    pad = model.pad
    inp = torch.tensor([[1, 2, 3, pad, pad, pad]], dtype=torch.long)
    tgt = torch.tensor([[2, 3, pad, pad, pad, pad]], dtype=torch.long)
    neg = torch.tensor([[7, 8, pad, pad, pad, pad]], dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = model.training_loss(inp, tgt, neg)
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert torch.equal(model.item_emb.weight[pad], torch.zeros(model.item_emb.weight.shape[1])), \
        "padding_idx row must never receive gradient updates"
