# M6 — Point-in-time samples & features

Pipeline: `python scripts/build_ranking_data.py --config configs/ranking_data.yaml`

```
official train, per-user time-ordered sequence (data/processed/train_sequences_ts.pkl)
    -> leave-last-out: prefix = all but last check-in, target = last check-in
    -> frozen recall model (Spatial-LightGCN, k10/lam0.3 best.pt), scores masked to the
       user's own prefix only, top-200 kept as candidates
    -> candidate Recall@{50,100,200} report
    -> layered negative sampling (4 sources) over the candidate/global item space
    -> user / item / cross / context features, all built from data available at the
       target's own timestamp or earlier
    -> data/processed/ranking_samples.csv + stats + audit sample
```

`data/gowalla/test.txt` is never opened by any module in this pipeline — verified by
`tests/test_ranking_leakage.py::test_targets_are_official_train_items_never_official_test`,
which checks every internal target against the official test set directly, not just by code
inspection.

## Why "not checked in" isn't "disliked"

Gowalla check-ins are a positive-only implicit signal: no impression log, no click, no skip.
An item absent from a user's history could mean genuine disinterest, or it could mean the user
never had the opportunity to see it. Treating every non-interacted item as an equally confident
negative overstates how much the label actually tells the ranker. This dataset handles that by
never treating "negative" as one thing — every negative row is tagged with `negative_source`,
and the four sources carry different amounts of that risk (see next section).

## Negative sources and their risk

| `negative_source` | How it's chosen | What it tests | False-negative risk |
|---|---|---|---|
| `random_negative` | uniform draw from items outside the user's official-train set | can the ranker separate visited-plausible from arbitrary | low — most POIs are geographically/behaviorally irrelevant to the user |
| `popularity_negative` | top 5% by prefix-universe interaction count, excluding the user's own train items | does the ranker just learn "predict popular", or something more | low-moderate — a popular item near the user could plausibly have been visited next |
| `recall_hard_negative` | another item from the same frozen recall model's own top-200 for this user, excluding the target | the actual ranking problem: within a strong candidate set, does the ranker improve on the recall order | **highest** — these are the candidates the recall model itself considered plausible; some are very likely real preferences the model never got to observe |
| `geo_hard_negative` | near the user's prefix activity center (mean lat/lon of visited POIs), excluding the user's own train items | is proximity alone driving the ranker, independent of the recall model's opinion | **high** — nearness is evidence of *reachability*, not evidence of *disinterest* |

`recall_hard_negative` and `geo_hard_negative` should not be read as "the user was offered
these and declined" — they were never offered anything; the risk column above should shape how
much weight M7 gives each source and is what the M7 feature-group ablation is meant to probe,
not something resolved here.

## Point-in-time discipline: what's protected and how it's tested

Every feature aggregator takes only the fields inside a user's own `prefix_items`/`prefix_ts`,
never the held-out `target_item`/`target_ts` (except `ctx_*`, which is explicitly allowed to use
`target_ts` as "now" — see `src/features/context_features.py` docstring — this is the moment the
request happens, not a look-ahead). `tests/test_ranking_leakage.py` checks this at two levels:

- **unit**: `test_user_features_depend_only_on_the_passed_prefix` truncates a synthetic prefix
  and asserts the computed features change — proving the function has no hidden access to
  anything beyond what was passed in.
- **unit**: `test_item_popularity_excludes_all_held_out_targets` asserts the sum of the global
  popularity array equals the total prefix-pair count exactly, which would be off by the number
  of users if any held-out target had leaked into its own popularity count.
- **integration**: `test_generated_negatives_never_equal_the_users_own_target` and
  `test_generated_negatives_never_in_the_users_own_prefix` run against the actual generated
  `ranking_samples.csv`, so a regression introduced later in `scripts/build_ranking_data.py`
  is caught even if the unit-level helpers stay correct in isolation.
- **integration**: `test_generated_dataset_is_reproducible_hash` locks the exact sha256 of the
  current `ranking_samples.csv` into `ranking_data_stats.json`; two runs of
  `scripts/build_ranking_data.py` with the same config produced byte-identical output
  (verified 2026-08-07, hash `3a1168d2...d294e5`).

## Known simplifications (read before quoting a number from this milestone)

1. **The frozen recall model saw every target during its own training.** `spatial_lightgcn_k10_lam0.3`
   was trained on the full official train split (per its own `config.json`), which includes every
   item this milestone later holds out as an internal validation target. Its `cross_recall_score`
   and the candidates it generates are therefore **not leakage-free with respect to a user's own
   target** — the model has already fit that edge. This was a deliberate scope decision (retraining
   a prefix-only recall model was out of scope for M6 without another Colab GPU cycle) and the
   consequence is that **Candidate Recall@K reported below is likely an optimistic upper bound**,
   not a clean estimate of what a truly prefix-only recall stage would achieve. A leak-free version
   would retrain the recall model on prefix-only data; flagged as a follow-up, not hidden.
2. **Item popularity/recency use a "prefix-universe" approximation, not a strict global
   timestamp cutoff.** `item_log1p_popularity`, `item_days_since_last_active`, and the popularity
   negative pool are computed over the union of *every* user's prefix (excluding only that
   specific user's own held-out target), not truncated to interactions strictly before that
   user's individual cutoff timestamp relative to every other user. Since Gowalla users' cutoffs
   are scattered over ~2 years, a small amount of population-level future information can enter
   another user's item statistics. This mirrors the same convention the official LightGCN
   train/test split itself uses (§2.2 of PROJECT_HANDOFF_V2.md: not a strict global point-in-time
   exposure cut) and is documented rather than silently assumed correct.
3. **`user_unique_poi_count == user_history_count` for every user.** `train_sequences_ts.pkl`
   is built by `scripts/prepare_data.py`, which de-duplicates repeat check-ins to the same POI
   (first visit defines sequence order). A "repeat visit" cross feature would therefore be
   degenerate (always 0) with the current data artifact and was not implemented — this is a data
   construction property, not a bug.
4. **No POI category data.** §2.4 lists "category or spatial-preference entropy" as a user
   feature "if available." Gowalla check-ins in this project carry no category labels, so this
   feature was not implemented rather than faked.
5. **`cross_dist_to_center_km`** uses the user's *prefix activity centroid* (mean lat/lon of
   visited POIs), not a predicted next location — the simplest anchor point available; a more
   sophisticated version (e.g. most-recent-location, or a learned anchor) is future work.

## Results (2026-08-07, `spatial_lightgcn_k10_lam0.3` as the frozen recall model)

- 29,858 / 29,858 official-train users had `min_history=5` satisfied (every Gowalla user in this
  split has at least 8 timestamped train check-ins, so no one was dropped at this threshold).
- Candidate Recall (see caveat #1 above):

  | K | Candidate Recall |
  |---:|---:|
  | 50 | 0.65527 |
  | 100 | 0.82450 |
  | 200 | 0.93489 |

  Read together with M7: even a perfect ranker over top-200 candidates cannot exceed ~93.5%
  Recall@K on this internal validation split — the remaining ~6.5% of users' targets are simply
  not retrievable by this recall model at this K, and no downstream ranker can recover them.
- 326,494 total sample rows over 29,858 users: 27,914 positive (one per user whose target
  landed in the top-200 — matches the K=200 candidate recall count exactly), 298,580 negative
  (119,432 `recall_hard_negative`, 59,716 each of `random_negative` / `popularity_negative` /
  `geo_hard_negative`).
- Missing data: only `item_days_since_last_active` has any missing values (0.002% of rows,
  items never seen in any prefix). Every other numeric feature is fully populated — Gowalla's
  official item list has 0 missing coordinates (`prepare_report.json`), so the coordinate-derived
  features have no gaps either.
- User-level split for M7: 15% of users (4,479 / 29,858) reserved as `split=val`, deterministic
  under `seed=2020`, recorded per-row so M7 does not need to re-derive it.

Full numeric distributions (quantiles, missing rate per column) are in
`experiments/results/ranking_data_stats.json`; column-by-column schema (name, dtype, feature
group) is in `experiments/results/feature_schema.json`; 20 hand-auditable rows are in
`experiments/results/ranking_data_audit_sample.csv`.

## Feature groups (see `feature_schema.json` for the exact column list)

- **user_\*** — from the user's own prefix only: history count, active span, recent-30-day
  count, average visited popularity, activity center (lat/lon), activity radius, coordinate
  coverage.
- **item_\*** — from the prefix universe (caveat #2): log1p popularity, prefix interaction
  count, cold-in-prefix flag, days since last active + missing flag, local POI density within
  1km, coordinate-missing flag.
- **cross_\*** — `cross_recall_score` (the frozen model's raw score — doubles as the "embedding
  dot product" feature from §2.4, since for this bilinear model family they are the same number),
  distance from the user's activity center to the candidate item, candidate rank (-1 if the item
  was never in the user's top-200), and an in-candidates flag.
- **ctx_\*** — request hour/day-of-week (from the target's own check-in timestamp) and days
  since the user's last prefix check-in.

## Not done in this milestone (by design)

Training a ranker on this dataset is M7's job, not M6's — this milestone only had to prove the
samples/features are constructible, leakage-free by the tests above, and reproducible. No DNN or
any model was trained to pass this milestone, matching the M6 acceptance bar in
PROJECT_HANDOFF_V2.md.
