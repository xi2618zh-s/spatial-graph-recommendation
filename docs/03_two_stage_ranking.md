# M7 — Two-stage retrieval → ranking

Pipeline: `python scripts/evaluate_pipeline.py --config configs/ranking_data.yaml`

```
data/processed/ranking_samples.csv (M6, train split)
    -> train LR (configs/ranker_lr.yaml) and GBDT (configs/ranker_gbdt.yaml)
val-split users -> regenerate FULL top-200 candidates (frozen recall model)
    -> build a feature row for every candidate (not just the sampled training rows)
    -> score with: retrieval_score_sort (no training) / LR / GBDT / GBDT feature-group ablation
    -> Recall@20 / MRR@20 / NDCG@20, all four under the identical candidate pool and user set
```

One command produces `experiments/results/ranking_eval.csv` (top-K reranking under every method) and
`experiments/results/ranking_eval_timing.json` — no separate script is needed to go from the
candidate file to a ranked top-20.

## Why rank a ~200-item shortlist instead of the full 41K-item catalog

Two independent reasons, not one:

1. **Cost.** Scoring a GBDT or MLP over all 40,981 items for every request scales with catalog
   size; scoring the same model over ~200 pre-filtered candidates does not. The frozen
   Spatial-LightGCN recall stage already does the cheap, catalog-scale filtering (a single
   embedding dot product per item); the ranker only has to be fast enough for a few hundred items.
2. **Label quality.** A ranker trained against uniform negatives from the full catalog would
   spend almost all its capacity learning "is this item even geographically/behaviorally
   plausible" — a question the recall stage already answers. Restricting training (and
   evaluation) to the recall stage's own candidate set forces the ranker to solve the harder,
   more useful problem: *given a set of already-plausible items, which one is right*. This is
   exactly why `recall_hard_negative` is the largest negative-sampling group in M6.

## Results (2026-08-07, val split, 4,479 users, K=20)

| Method | Feature group | # features | Recall@20 | MRR@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|
| `retrieval_score_sort` (no training) | n/a | 1 | 0.41103 | 0.10025 | 0.16714 |
| `ranker_lr` | full | 24 | 0.50368 | 0.20470 | 0.27073 |
| `ranker_gbdt` (ablation: recall_only) | recall_only | 1 | 0.40455 | 0.09999 | 0.16556 |
| `ranker_gbdt` (ablation: stats) | + user/item stats | 17 | 0.54164 | 0.22309 | 0.29310 |
| `ranker_gbdt` (ablation: spatial) | + distance-to-center | 19 | 0.56039 | 0.24309 | 0.31305 |
| **`ranker_gbdt`** (headline) | full | 24 | **0.57714** | **0.24867** | **0.32102** |

Reran the full pipeline twice: all three metrics for every method matched to the last printed
digit (only wall-clock timing columns varied), confirming the training + evaluation path is
deterministic under `seed=2020`.

## Reading these numbers correctly

- **Candidate Recall@200 for this same recall model is 0.93489** (docs/02_samples_features.md).
  No ranker here can exceed that ceiling — 6.5% of val users' targets are simply absent from the
  candidate pool. The gap between 0.93489 (target is *somewhere* in 200 candidates) and 0.57714
  (best ranker gets it into the *top 20*) is the real work re-ranking does: it is far from
  saturated, meaning further ranker improvement is plausible rather than pointless.
- **`gbdt[recall_only]` (0.40455) is not just "≈ baseline" but very slightly *worse* than
  `retrieval_score_sort` (0.41103)** despite using the identical single feature. This is a real,
  small effect of histogram binning: `HistGradientBoostingClassifier` bins continuous inputs
  before splitting, which can introduce ties that a raw sort does not have. It is reported as-is
  rather than smoothed over — a reminder that "more machinery" is not automatically "better,"
  even holding the feature set fixed.
- **User/item statistical features (`stats`) contribute the largest single jump** (+0.137
  absolute Recall@20 over recall-only) — history length, visited-popularity, and item popularity
  carry most of the exploitable signal in this dataset. Spatial distance (+0.019) and the
  remaining context/candidate-metadata features (+0.017) are smaller but still positive and
  monotonic — no feature group here made things worse, so no group is dropped from the "full"
  ranker.
- **GBDT (0.57714) clearly beats LR (0.50368) on the same full feature set** — consistent with
  the RISK_REGISTER's expected pattern that tree models capture non-linear feature interactions
  (e.g. popularity × distance) better than a linear model on this kind of tabular mix. LR was
  still worth training: it is the interpretability/calibration check, not a candidate for the
  headline number.
- **This whole evaluation inherits M6's frozen-recall-model caveat.** `spatial_lightgcn_k10_lam0.3`
  was trained on the full official train split, including every val user's held-out target, so
  both the 0.93489 candidate ceiling and the `cross_recall_score` feature it feeds every ranker
  are not leakage-free with respect to that target. Read every number on this page as an
  optimistic upper bound relative to what a strictly prefix-only recall stage would produce, not
  as a clean estimate — same caveat as documented in `docs/02_samples_features.md`, not repeated
  fine print here for effect.

## Efficiency

| | |
|---|---:|
| Feature generation, full val set (4,479 users × 200 candidates = 895,800 rows) | 32.0s total, 7.1ms/user |
| GBDT training (full feature set, 277,528 training rows) | 5.6s |
| LR training (full feature set) | 1.7s |
| Per-user scoring latency (GBDT, 200 candidates), 500-user sample | P50 5.1ms, P95 6.4ms |

All numbers are from a single local CPU run (no GPU), not a load-tested service — this is a
benchmark harness, not the serving path M9 will build. At this scale (41K items, ~200 candidates
per request) even the "expensive" stage stays comfortably in single-digit milliseconds, which is
itself a useful data point for M9's later Flat-vs-ANN decision.

## What "not improving" would have required (didn't happen here, documented for completeness)

Every feature group tested here was non-negative, and GBDT-full beat every simpler baseline
cleanly, so the diagnostic checklist in RISK_REGISTER.md §2.3 (candidate ceiling, train/val
leakage, false-negative noise, feature redundancy, overfitting) was not needed to explain a
negative result. It stays relevant for M8's slice-level analysis, where an aggregate win can
still hide a subgroup loss.

## Not done in this milestone (by design)

- **MLP**: PROJECT_HANDOFF_V2.md gates a shallow MLP on LR/GBDT results being stable first — they
  are, so MLP is a candidate for a future iteration, not required for M7's acceptance bar.
- **DeepFM/DIN**: explicitly a conditional/M12 extension, not attempted here.
- **Business-proxy / slice metrics** (coverage, popularity bias, cold-start, distance trade-offs):
  M8, using this same trained `ranker_gbdt` checkpoint (`experiments/logs/ranker_gbdt/model.pkl`).
