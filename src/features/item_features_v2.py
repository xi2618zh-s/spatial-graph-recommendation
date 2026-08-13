"""V2: exact point-in-time item statistics, replacing M6's "prefix-universe"
approximation (src/features/item_features.py -- see that module's docstring
and docs/02_samples_features.md's known-simplification #2 for what this
fixes). src/features/item_features.py is left completely untouched: this is
a new, separate module, so the M6-M9 "v1" pipeline's already-documented,
already-verified results are unaffected by anything here.

M6's approximation counted an item's popularity over the union of every
user's prefix (excluding only that specific user's own held-out target),
which lets population-level future information leak in: other users'
interactions that happen to be chronologically AFTER the querying user's own
cutoff still counted. This module instead indexes every item's FULL
interaction timeline (sorted timestamps, across ALL users, from the complete
official-train history -- there is no "prefix" concept here, only "before
this exact cutoff or not") and answers point-in-time queries with
`np.searchsorted`, so a candidate's popularity/recency feature reflects
exactly what was knowable in the world at the query's own timestamp, not at
some other user's.
"""

import time

import numpy as np


class GlobalItemTimeline:
    """Sorted interaction timestamps per item, across every user's full
    official-train sequence. `cutoff_ts` is always treated as the querying
    event's own timestamp; comparisons are STRICT ("<"), so an item's own
    triggering interaction (item == candidate, ts == cutoff_ts) is always
    excluded from its own popularity/recency count -- correct by
    construction, not by having to separately track "whose target is this."
    """

    def __init__(self, sequences_ts: dict[int, list[tuple[int, int]]], n_items: int):
        per_item_ts: list[list[int]] = [[] for _ in range(n_items)]
        for seq in sequences_ts.values():
            for item, ts in seq:
                per_item_ts[item].append(ts)
        self.n_items = n_items
        self._sorted = [np.asarray(sorted(ts), dtype=np.int64) for ts in per_item_ts]
        self.total_interactions = sum(len(ts) for ts in per_item_ts)

    def popularity_before(self, item: int, cutoff_ts: int) -> int:
        """Count of interactions with `item` strictly before `cutoff_ts`."""
        return int(np.searchsorted(self._sorted[item], cutoff_ts, side="left"))

    def popularity_before_many_cutoffs(self, item: int, cutoffs: np.ndarray) -> np.ndarray:
        """Vectorized: popularity of a single `item` at many different
        cutoff timestamps (e.g. once per candidate row referencing the same
        item across different users/queries)."""
        return np.searchsorted(self._sorted[item], cutoffs, side="left")

    def last_active_before(self, item: int, cutoff_ts: int) -> float:
        """Most recent interaction timestamp with `item` strictly before
        `cutoff_ts`, or NaN if `item` has no interaction before that point."""
        arr = self._sorted[item]
        idx = np.searchsorted(arr, cutoff_ts, side="left")
        return float(arr[idx - 1]) if idx > 0 else float("nan")

    def total_popularity(self, item: int) -> int:
        """Popularity with no cutoff (all-time count) -- for sanity checks
        against M6's prefix-universe numbers, not for use as a feature."""
        return len(self._sorted[item])


def build_global_item_timeline(sequences_ts: dict[int, list[tuple[int, int]]],
                               n_items: int) -> tuple[GlobalItemTimeline, float]:
    """Convenience constructor that also reports build time, since this is
    the one-time cost users of this module should budget for."""
    t0 = time.perf_counter()
    timeline = GlobalItemTimeline(sequences_ts, n_items)
    return timeline, time.perf_counter() - t0
