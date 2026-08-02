"""Sequence data handling for SASRec (torch-free, so it is unit-testable).

Sequences come from data/processed/train_sequences.pkl (built by
prepare_data.py from SNAP timestamps, restricted to official TRAIN items only
— no test leakage by construction).

Conventions:
- RIGHT-padding with pad_id = n_items (an extra embedding row). Right-padding
  guarantees every query position has at least one real key under a causal
  mask, avoiding the all-masked-row NaN pathology of left-padding.
- Autoregressive training pairs: input = seq[:-1], target = seq[1:], both
  truncated to the last `max_len` steps.
- One uniform negative per position, rejection-sampled against the user's
  full training item set.
"""

import pickle
from pathlib import Path

import numpy as np


class SequenceData:
    def __init__(self, pkl_path: str | Path, n_items: int, train_sets: dict,
                 max_len: int = 50):
        with open(pkl_path, "rb") as f:
            self.seqs: dict[int, list[int]] = pickle.load(f)
        self.n_items = n_items
        self.pad = n_items
        self.max_len = max_len
        self.train_sets = train_sets
        self.train_users = np.array(
            [u for u, s in self.seqs.items() if len(s) >= 2]
        )
        lens = [len(self.seqs[u]) for u in self.train_users]
        print(f"SequenceData: {len(self.train_users)} trainable users, "
              f"median seq len {np.median(lens):.0f}, max_len {max_len}")

    def _pad_right(self, seq: list[int]) -> tuple[np.ndarray, int]:
        seq = seq[-self.max_len:]
        out = np.full(self.max_len, self.pad, dtype=np.int64)
        out[: len(seq)] = seq
        return out, len(seq)

    def train_batches(self, batch_size: int, rng: np.random.Generator):
        """Yield (inp, tgt, neg, length) right-padded numpy batches."""
        order = rng.permutation(self.train_users)
        for start in range(0, len(order), batch_size):
            users = order[start:start + batch_size]
            B = len(users)
            inp = np.full((B, self.max_len), self.pad, dtype=np.int64)
            tgt = np.full((B, self.max_len), self.pad, dtype=np.int64)
            neg = np.full((B, self.max_len), self.pad, dtype=np.int64)
            length = np.zeros(B, dtype=np.int64)
            for r, u in enumerate(users):
                s = self.seqs[u]
                inp[r], length[r] = self._pad_right(s[:-1])
                tgt[r], _ = self._pad_right(s[1:])
                u_set = self.train_sets[u]
                for t in range(length[r]):
                    n = rng.integers(0, self.n_items)
                    while n in u_set:
                        n = rng.integers(0, self.n_items)
                    neg[r, t] = n
            yield inp, tgt, neg, length

    def eval_inputs(self, user_ids: np.ndarray):
        """Right-padded full train sequences + lengths for next-item scoring.
        Users with empty sequences get length 0 (caller scores them as zeros)."""
        B = len(user_ids)
        inp = np.full((B, self.max_len), self.pad, dtype=np.int64)
        length = np.zeros(B, dtype=np.int64)
        for r, u in enumerate(user_ids):
            s = self.seqs.get(u, [])
            if s:
                inp[r], length[r] = self._pad_right(s)
        return inp, length
