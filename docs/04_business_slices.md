# M8 — Business-proxy metrics, popularity bias, cold-start & long-tail diagnostics

Pipeline: `python scripts/evaluate_slices.py --config configs/ranking_data.yaml`

Compares `retrieval_score_sort` (M7 baseline) against the persisted `ranker_gbdt` model on the
**identical** val-user candidate pool. Every metric here is an **offline proxy** — Gowalla has no
impressions, clicks, or purchases, so nothing below should be read as CTR/CVR/GMV (see §0.2 of
PROJECT_HANDOFF_V2.md). Same frozen-recall-model caveat as M6/M7 applies throughout: the recall
stage saw every target during its own training, so absolute levels skew optimistic; the
*comparisons between methods* below are the trustworthy part, since both methods share the same
candidate pool and the same leakage.

## Bucket boundaries (fixed from train, `experiments/results/bucket_boundaries.json`)

- **User activity** (prefix history length, tertiles over all 29,858 eligible users): low ≤ 11,
  mid 12–22, high > 22.
- **Item popularity** (by prefix-universe interaction rank, not equal item-count thirds): head =
  top 20% of items by interaction count, tail = bottom 50%. Deliberately unequal split because
  equal-count tertiles would hide how concentrated interactions actually are:

  | Bucket | Item share | Interaction share |
  |---|---:|---:|
  | head | 20.0% | 53.1% |
  | mid | 30.0% | 24.0% |
  | tail | 50.0% | 22.9% |

  One-fifth of the catalog absorbs over half of all interaction volume — this is the number the
  "long tail" framing below is built on.
- **Target distance**: fixed edges at 1 / 5 / 20 / 100 km, measured from the user's prefix
  activity center to the actual held-out target's coordinates.

## Strict vs. near cold-start (kept separate, never conflated)

| | Count |
|---|---:|
| Strict cold-start test users (zero official-train interactions) | **0 / 29,858** |
| Strict cold-start test items (zero official-train interactions) | **0 / 38,546** |
| Near-cold-start: low-history users (bottom activity tertile) | 11,337 / 29,858 |
| Near-cold-start: low-frequency items (bottom 50% by popularity) | 20,490 / 40,981 |

The official LightGCN Gowalla split guarantees every test user/item already appears in train —
this matches the accepted explanation in RISK_REGISTER.md §1: **strict cold-start cannot be
evaluated on this benchmark split**, full stop. This project does not claim to solve cold-start;
what follows is near-cold-start (low history / low frequency) behavior only, reported under that
name and never rebranded as "cold-start."

## Slice results (Recall@20, 95% bootstrap CI, `experiments/results/slice_metrics.csv`)

| Slice | Bucket | `retrieval_score_sort` | `ranker_gbdt` |
|---|---|---:|---:|
| User activity | low | 0.6086 [0.585, 0.633] | 0.7510 [0.730, 0.772] |
| | mid | 0.4020 [0.375, 0.430] | 0.5925 [0.565, 0.621] |
| | high | 0.2024 [0.183, 0.223] | 0.3739 [0.348, 0.399] |
| Target popularity | head | 0.4891 [0.469, 0.510] | 0.4982 [0.478, 0.519] |
| | mid | 0.3758 [0.344, 0.406] | 0.5798 [0.549, 0.610] |
| | tail | 0.3046 [0.280, 0.330] | **0.7102** [0.685, 0.735] |
| Target distance | <1km | 0.6169 [0.539, 0.695] | 0.5714 [0.494, 0.649] |
| | <5km | 0.4890 [0.447, 0.529] | 0.5588 [0.515, 0.599] |
| | <20km | 0.4505 [0.411, 0.488] | 0.5981 [0.560, 0.634] |
| | <100km | 0.4159 [0.382, 0.448] | 0.5280 [0.496, 0.563] |
| | >100km | 0.3669 [0.347, 0.386] | 0.5929 [0.572, 0.612] |
| Overall | all | 0.4110 [0.396, 0.426] | 0.5771 [0.563, 0.592] |

(`near_cold_start` slice numbers are omitted here — its "low_history" bucket is the exact same
user population as User activity/"low" above, just reported against a combined "rest" group; see
`slice_metrics.csv` for the raw rows.)

## Diagnostic chain 1 — GBDT wins everywhere, but the *size* of the win is not uniform

**Observation:** GBDT beats the baseline in every user-activity bucket, but the relative gain is
far larger for high-activity users (0.202 → 0.374, +85%) than low-activity users (0.609 → 0.751,
+23%). Absolute Recall@20 also *drops* as activity increases for both methods.

**Hypothesis:** high-activity users have visited more places already, so their "next" check-in
draws from a larger, more diffuse set of plausible POIs (higher effective entropy) — harder for
any method, but the raw recall-score baseline (which has no explicit notion of a user's own
history breadth) is hit hardest, while GBDT's `user_history_count` / `user_avg_visited_popularity`
features give it something concrete to condition on.

**Validation:** consistent with the User activity table above and with `user_activity_radius_km`
in M6's feature stats (median 114km, p99 3,931km — Gowalla's most active users are travelers, not
neighborhood-loyal locals), which independently supports "more history → more diffuse future
behavior" rather than "more history → easier prediction."

**Conclusion / next action:** the aggregate 0.577 headline number is not representative of
high-activity users specifically. If this system were to ship a guardrail, it should be evaluated
per-activity-bucket, not on the blended average.

## Diagnostic chain 2 — GBDT does *best*, not worst, on tail targets (reverses the baseline's pattern)

**Observation:** `retrieval_score_sort` degrades monotonically as the target gets less popular
(head 0.489 → tail 0.305, the classic embedding-popularity-bias pattern). `ranker_gbdt` does the
**opposite**: tail 0.710 > mid 0.580 > head 0.498.

**Hypothesis A (real signal):** `item_log1p_popularity` is a directly informative feature, and a
low-popularity item is a strong, almost-deterministic tell when the *true* label happens to be
one of the few low-popularity items in a mostly-high-popularity candidate list — GBDT can exploit
that far more directly than a model relying only on embedding geometry.

**Hypothesis B (selection artifact):** users whose true target is a tail item may simply have
*fewer* competing tail-popularity candidates in their top-200 list to confuse the ranker with, so
"tail target" and "easy discrimination problem" could be correlated for reasons unrelated to the
ranker being generally better at tail recommendation.

**Validation status:** not yet distinguished — doing so would require conditioning on candidate-list
popularity composition per user, which this milestone did not build. Flagged as a specific
follow-up, not resolved here.

**Conclusion / next action:** do **not** claim "GBDT solves the popularity-bias problem for
individual recommendations" from this table alone — the bias-metrics section below (aggregate,
not per-target-bucket) is the more defensible basis for that claim, and even there the framing is
"less biased than the baseline," not "unbiased."

## Diagnostic chain 3 — accuracy, coverage, and long-tail exposure improved *together*, not traded off

This is the result RISK_REGISTER.md §2.4 explicitly warns not to assume either direction on.

| Metric | `retrieval_score_sort` | `ranker_gbdt` |
|---|---:|---:|
| Catalog Coverage@20 | 0.3268 | **0.5526** |
| Tail Exposure Share | 0.0493 | **0.2034** |
| Mean −log(popularity) | −4.351 | **−3.721** (higher = more novel) |
| Average Recommendation Popularity | 162.2 | **111.7** |
| Popularity lift vs. catalog mean | 8.52× | **5.87×** |
| Exposure Gini | 0.897 | **0.760** (lower = less concentrated) |

**Observation:** the ranker with substantially better Recall/NDCG (§docs/03) *also* recommends
from a broader slice of the catalog, exposes more tail items, and concentrates exposure on fewer
items less. Every coverage/bias metric moved in the "healthier" direction alongside accuracy.

**Hypothesis:** BPR-trained embedding models are known to systematically inflate scores for
popular items (more frequent positive gradient updates during training) — `retrieval_score_sort`
inherits that bias directly. GBDT, given `item_log1p_popularity` as an explicit feature alongside
the recall score, can *down-weight* that embedding-driven popularity skew when the behavioral
features point elsewhere, rather than being stuck amplifying it.

**Conclusion:** for this specific pair of methods, there is no accuracy-vs-coverage trade-off to
report — the ranking stage is a strict improvement on both axes. This is reported as an honest
positive result, not assumed in advance; a different ranker or feature set could easily show the
opposite pattern the RISK_REGISTER anticipates, and M9+ work should keep checking this rather than
treat it as settled.

## Diagnostic chain 4 — GBDT recommends items *farther* from the user's activity center, not closer

| Metric | `retrieval_score_sort` | `ranker_gbdt` |
|---|---:|---:|
| Median recommended-item distance to activity center | 79.1 km | **109.7 km** |
| Mean | 398.2 km | **546.0 km** |
| In-list spatial diversity (mean pairwise distance) | 340.6 km | **510.0 km** |

**Observation:** contrary to a naive "adding spatial features should make recommendations more
local" expectation, GBDT's recommendations are on average *farther* from the user's own activity
center, and its lists are spatially *more* diverse.

**Hypothesis:** consistent with Diagnostic chain 1 — Gowalla's true next check-ins are often far
from a user's historical center (wide `user_activity_radius_km` distribution), and the raw
recall-score baseline is biased toward nearby, popular, urban-dense POIs (which is also why it has
lower coverage and higher popularity lift above). GBDT correctly learns that "far" targets are
common rather than defaulting to a nearby/popular prior.

**Validation:** matches the Target distance slice table — GBDT's largest relative improvement
over the baseline is in the `>100km` bucket (0.367 → 0.593), the single biggest slice gap in that
table.

**Conclusion / next action:** "the spatial-aware model recommends closer" is **not** true here —
if that framing is ever used in the resume/interview evidence chain, it needs to be corrected to
"the ranker learned that distance alone is not the deciding factor, distinguishing it from a
naive proximity heuristic."

## Effective coverage

`effective_user_coverage_at_20` is 1.0 for both methods — every val user had ≥200 candidates
available (this internal validation split's candidate pool is fixed at max_k=200), so no user was
served a short list. This metric would only differ from 1.0 in a serving configuration with a
smaller or more variable candidate budget than what M6 generated.

## Figures

- `experiments/results/figures/slice_recall_by_user_activity.png` — Diagnostic chain 1
- `experiments/results/figures/popularity_coverage_tradeoff.png` — Diagnostic chain 3
- `experiments/results/figures/recommendation_distance_distribution.png` — Diagnostic chain 4

## What this milestone does not claim

- Does not claim to have solved cold-start (none exists to solve in this split).
- Does not claim GBDT is unbiased — only less popularity-biased than the raw recall-score
  baseline, on this specific comparison.
- Does not claim the tail-target finding (chain 2) generalizes without the follow-up analysis
  named there.
- Does not use any business-proxy number as a stand-in for CTR/CVR/satisfaction.
