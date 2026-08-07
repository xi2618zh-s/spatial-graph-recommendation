# Spatial Graph Recommendation

Graph-enhanced, location-aware personalized recommendation system on the Gowalla
check-in dataset. **Retrieval, two-stage ranking, business-proxy/bias diagnostics, and an
ANN serving benchmark are all implemented and evaluated** end-to-end (M6–M9); remaining
work is engineering-reliability hardening (M10) and final README/interview packaging
(M11) — see `docs/00_project_plan.md` for exact status per milestone.

```
Retrieval (implemented, full-ranking eval)
  Popularity → ItemCF → MF-BPR → LightGCN → Spatial-LightGCN
                                                 +
                                          SASRec (sequential)

Ranking (implemented, end-to-end evaluated — M6 + M7)
  leave-last-out prefix/target → frozen recall candidates → layered negatives
  → user/item/cross/context features → retrieval-score / LR / GBDT ranker
                                                 ↓
Diagnostics (implemented — M8)
  slice metrics (activity/popularity/distance) + coverage/bias/cold-start proxies
                                                 ↓
Serving (implemented — M9, benchmark harness scale: 41K items)
  FAISS Flat/HNSW/IVF recall-latency benchmark → FastAPI /health + /recommend demo
```

Benchmark: the standard NGCF/LightGCN split of Gowalla
(29,858 users · 40,981 POIs · ~1.03M interactions).
Reference numbers to reproduce: LightGCN Recall@20 ≈ 0.183, NDCG@20 ≈ 0.155.

## Current results

| Model | Recall@20 | NDCG@20 | Status |
|---|---:|---:|---|
| Popularity | 0.04163 | 0.03169 | committed to `experiments/results/summary.csv` |
| ItemCF (cosine, top200) | 0.11787 | 0.08610 | committed to `experiments/results/summary.csv` |
| MF-BPR | 0.12950 | 0.10760 | committed to `experiments/results/summary.csv` (epoch 40; recovered from Colab console log — see notes column) |
| LightGCN (epoch 440) | 0.17724 | 0.15123 | synced from Drive, config+history.json 3-way verified (see `experiments/results/summary.csv` notes); -3.1% vs paper's 0.183 |
| Spatial-LightGCN (k10, λ0.3, epoch 360) | 0.18335 | 0.15638 | synced from Drive, 4-way verified; +3.45% Recall@20 / +3.40% NDCG@20 over LightGCN, best epoch 18.2% earlier |
| SASRec (max_len 50, epoch 200) | 0.12577 | 0.09961 | synced from Drive, 4-way verified; weaker than graph models — best metric still at final epoch, motivating a queued 400-epoch extended run |

Spatial λ/k ablations and the SASRec 400-epoch run are **[进行中/待回收]** — not in this table until their runs finish and are synced.
The Spatial-LightGCN run above uses an adaptive Gaussian bandwidth (`sigma_km: null` in config, k=10 nearest
geographic neighbors, max_dist_km=100): rerunning `src/data/spatial_graph.py` locally against this exact config
reproduces **σ=0.21km (auto-median) and 447,797 spatial edges** exactly, confirming the previously-quoted figures.

A result only exists once it has a row in `experiments/results/summary.csv` backed by a config and a raw log —
do not treat numbers from prior notes or console output as final until they land there.

Per-user bootstrap 95% CIs (`scripts/compute_bootstrap_ci.py`, 2000 resamples over the test-user set) are in
`experiments/results/bootstrap_ci.csv`. These quantify test-user sampling variance for a single trained checkpoint,
not training-seed variance — a separate multi-seed re-run is still needed before claiming the Spatial-LightGCN gain
is stable across random initialization (planned in P1, alongside the still-running λ/k ablations).

## Ranking samples & features (M6)

`python scripts/build_ranking_data.py --config configs/ranking_data.yaml` builds a leave-last-out,
point-in-time ranking dataset inside the official train split (official `test.txt` stays sealed).
Full writeup, known simplifications, and false-negative risk per source: `docs/02_samples_features.md`.

| K | Candidate Recall (frozen Spatial-LightGCN recall model) |
|---:|---:|
| 50 | 0.65527 |
| 100 | 0.82450 |
| 200 | 0.93489 |

326,494 sample rows over 29,858 users (27,914 positive, 298,580 negative across 4 layered
negative sources). 7 leakage/reproducibility tests pass (`pytest tests/`), including a byte-exact
hash check that two runs of the build script produce an identical output file.

## Two-stage ranking (M7)

`python scripts/evaluate_pipeline.py --config configs/ranking_data.yaml` trains LR + GBDT
rankers on the M6 samples and evaluates all methods on the identical, freshly-regenerated
top-200 candidate pool for every val user. Full writeup + how to read these numbers correctly
(they inherit M6's frozen-recall-model caveat): `docs/03_two_stage_ranking.md`.

| Method | Recall@20 | MRR@20 | NDCG@20 |
|---|---:|---:|---:|
| `retrieval_score_sort` (no training) | 0.41103 | 0.10025 | 0.16714 |
| `ranker_lr` | 0.50368 | 0.20470 | 0.27073 |
| **`ranker_gbdt`** | **0.57714** | **0.24867** | **0.32102** |

Feature-group ablation (GBDT): recall-score-only ≈ baseline (0.40455) → + user/item stats
(0.54164, the largest single jump) → + spatial distance (0.56039) → full feature set (0.57714).
Per-user scoring latency (GBDT, 200 candidates): P50 5.1ms / P95 6.4ms on a single CPU core.

## Business-proxy metrics & bias diagnostics (M8)

`python scripts/evaluate_slices.py --config configs/ranking_data.yaml` compares
`retrieval_score_sort` against `ranker_gbdt` on identical users/candidates across coverage,
popularity bias, and slice-level accuracy. Full "observation → hypothesis → validation →
conclusion" diagnostic chains: `docs/04_business_slices.md`.

- **Strict cold-start**: 0 users / 0 items — the official Gowalla/LightGCN split guarantees every
  test user and item already appears in train, so this cannot be evaluated on this benchmark
  (documented, not glossed over).
- **Accuracy, coverage, and long-tail exposure improved together** going from the raw recall
  score to the GBDT ranker — not a trade-off in this comparison: Catalog Coverage@20 0.327→0.553,
  Tail Exposure Share 0.049→0.203, popularity lift 8.5×→5.9× (lower = less popularity bias).
- Recall@20 by user-activity tertile: low 0.609→0.751, mid 0.402→0.593, high 0.202→0.374 — GBDT
  wins everywhere, but the *relative* gain is largest for high-activity (harder) users.
- Reran against a reversed-from-expectation finding: GBDT's recommendations land *farther* from
  users' activity centers on average (109.7km vs 79.1km median) — Gowalla's true next check-ins
  are often far from a user's history, and the ranker learns that instead of defaulting to a
  nearby/popular prior. See `docs/04_business_slices.md` for why this is not a bug.

## ANN retrieval & serving benchmark (M9)

```bash
python scripts/build_ann_index.py --config configs/ann_index.yaml
python scripts/benchmark_serving.py --config configs/ann_index.yaml
uvicorn src.serving.app:app --reload   # GET /health, GET /recommend?user_id=0&k=10
```

Full trade-off discussion (and why the honest conclusion at this scale is "use exact search"):
`docs/05_ann_serving.md`.

| Index | Recall@200 vs exact | P50 latency | P95 latency | Build time | Disk |
|---|---:|---:|---:|---:|---:|
| Flat (exact) | 1.0 (gold standard) | 1.60ms | 2.64ms | 0.01s | 10.5MB |
| HNSW (efSearch=1024) | 0.875 (plateauing) | 0.77ms† | 1.15ms† | 2.25s | 21.6MB |
| IVF (nprobe=64) | **0.999** | 0.59ms† | 0.96ms† | 0.13s | 10.8MB |

†latency measured at a mid-sweep operating point (efSearch=256 / nprobe=8), not the recall-maximizing one above — see docs/05 for the full sweep.

At 40,981 items, exact search is already sub-2ms; `IndexFlatIP` is verified to reproduce the
frozen model's own raw score exactly (100/100 sampled users, `tests/test_ann_consistency.py`).
IVF dominates HNSW on recall, build time, and disk at this scale — a real, measured finding, not
an assumption that ANN would obviously help.

## Repository layout

```
spatial-graph-recommendation/
├── configs/                  # One YAML per experiment. Never hardcode hyperparameters in code.
├── data/
│   ├── raw/                  # SNAP raw check-ins (loc-gowalla_totalCheckins.txt.gz). Git-ignored.
│   ├── gowalla/              # Official LightGCN split: train/test + org_id→remap_id maps. Committed.
│   └── processed/            # Generated artifacts (poi_coords.csv, sequences.pkl, ...). Git-ignored.
├── src/
│   ├── data/                 # Dataset classes, graph construction, ranking_dataset.py, sampling.py
│   ├── features/             # user/item/cross/context point-in-time feature builders
│   ├── models/               # One file per model (mf.py, lightgcn.py, sasrec.py) + registry.py (checkpoint reload)
│   ├── train/                # Training loops (BPR trainer, sequential trainer, ranker trainer)
│   ├── eval/                 # Metrics, full-ranking evaluator, per-user bootstrap CI
│   └── utils/                # Seeding, logging, config loading
├── scripts/                  # Entry points only — thin wrappers around src/ (prepare_data.py, train.py, build_ranking_data.py, ...)
├── notebooks/                # Colab launchers. No logic lives here; notebooks only call scripts/.
├── experiments/
│   ├── logs/                 # Raw training logs, one subdir per run (git-ignored)
│   └── results/              # Final metric tables (CSV/JSON), committed — the source of truth for the paper trail
├── tests/                    # Leakage, reproducibility, RNG-resume, SASRec padding, ANN consistency, smoke tests
└── docs/                     # Design notes and experiment reports, one MD per milestone
```

Rules that keep this maintainable:

1. **`scripts/` are thin, `src/` is the library.** Every script is runnable as
   `python scripts/xxx.py --config configs/yyy.yaml` and does nothing but parse
   args and call into `src/`.
2. **`data/raw` and `data/processed` are never committed** (large / regenerable).
   `data/gowalla` (the small official split) is committed so results are
   reproducible from a fresh clone.
3. **Every run gets a config file and a results row.** A result that isn't in
   `experiments/results/` doesn't exist.
4. **Notebooks contain no logic** — they exist only to run scripts on Colab GPUs.

## Quick start

Everything below runs on CPU except step 4 (GPU strongly recommended for LightGCN/Spatial-LightGCN/
SASRec; MF-BPR and the CPU-only steps after step 5 are fine on a laptop). Steps 1–3 and 6–10 are
fully reproducible from a fresh clone; step 4's exact numbers additionally require the SNAP raw
file (licensing reasons keep it out of the repo, see Data sources below) and — for anything past
MF-BPR in reasonable time — a GPU.

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Place the SNAP raw file (see below) at data/raw/loc-gowalla_totalCheckins.txt.gz

# 3. Build processed artifacts (coordinates + timestamped sequences joined onto official IDs)
python scripts/prepare_data.py

# 4. Train a retrieval model (example: MF-BPR, CPU-feasible; LightGCN/Spatial-LightGCN/SASRec want a GPU)
python scripts/train.py --config configs/mf_gowalla.yaml

# 5. Per-user metrics + bootstrap 95% CI for any completed run
python scripts/compute_bootstrap_ci.py --run mf_bpr_gowalla

# 6. Build the point-in-time ranking dataset (needs a completed recall run's best.pt --
#    the repo's own results were built on spatial_lightgcn_k10_lam0.3, see configs/ranking_data.yaml)
python scripts/build_ranking_data.py --config configs/ranking_data.yaml

# 7. Train LR/GBDT rankers and run the end-to-end candidate->rank->top-K evaluation
python scripts/evaluate_pipeline.py --config configs/ranking_data.yaml

# 8. Business-proxy metrics, popularity-bias and slice diagnostics
python scripts/evaluate_slices.py --config configs/ranking_data.yaml

# 9. FAISS ANN index + recall-latency benchmark, then the minimal serving demo
python scripts/build_ann_index.py --config configs/ann_index.yaml
python scripts/benchmark_serving.py --config configs/ann_index.yaml
uvicorn src.serving.app:app --reload   # GET /health, GET /recommend?user_id=0&k=10

# 10. Run the test suite (leakage, reproducibility, RNG-resume, SASRec padding, ANN consistency, smoke)
pytest tests/
```

## Data sources

| File | Source | Role |
|---|---|---|
| `data/gowalla/{train,test,user_list,item_list}.txt` | [LightGCN official repo](https://github.com/kuandeng/LightGCN) `Data/gowalla/` | Benchmark split (interaction IDs only); **committed**, no download needed |
| `data/raw/loc-gowalla_totalCheckins.txt.gz` | [SNAP: Gowalla](https://snap.stanford.edu/data/loc-gowalla.html) | Coordinates + timestamps, joined back via `org_id`; **not committed** (105MB, SNAP's own terms govern redistribution) — download it yourself and verify: `sha256sum data/raw/loc-gowalla_totalCheckins.txt.gz` should print `c1c3e19effba649b6c89aeab3c1f9459fad88cfdc2b460fc70fd54e295d83ea0` |

`scripts/prepare_data.py`'s own output (`data/processed/prepare_report.json`) is a second integrity
check: a fresh run against the file above should print `"matched_checkins": 1585044`,
`"train_pairs_with_timestamp": 810128`, `"timestamp_coverage": 1.0` — if any of those differ, the
raw file or the official split has drifted from what this repo's results were built on.
