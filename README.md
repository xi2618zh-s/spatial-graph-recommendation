# Spatial Graph Recommendation

Graph-enhanced, location-aware personalized recommendation system on the Gowalla
check-in dataset. **Retrieval stage implemented and evaluated** with progressively
stronger candidate generators; two-stage ranking, business-proxy diagnostics, and
ANN serving are planned, not yet built (see `docs/00_project_plan.md` for status).

```
Retrieval (implemented, full-ranking eval)
  Popularity → ItemCF → MF-BPR → LightGCN → Spatial-LightGCN
                                                 +
                                          SASRec (sequential)

Ranking / serving (planned)
  point-in-time samples & features → LR/GBDT ranker → FAISS ANN → FastAPI serving
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

## Repository layout

```
spatial-graph-recommendation/
├── configs/                  # One YAML per experiment. Never hardcode hyperparameters in code.
├── data/
│   ├── raw/                  # SNAP raw check-ins (loc-gowalla_totalCheckins.txt.gz). Git-ignored.
│   ├── gowalla/              # Official LightGCN split: train/test + org_id→remap_id maps. Committed.
│   └── processed/            # Generated artifacts (poi_coords.csv, sequences.pkl, ...). Git-ignored.
├── src/
│   ├── data/                 # Dataset classes, graph construction, negative samplers
│   ├── models/               # One file per model: mf.py, lightgcn.py, spatial_lightgcn.py, sasrec.py, ranker.py
│   ├── train/                # Training loops (BPR trainer, sequential trainer, ranker trainer)
│   ├── eval/                 # Metrics (Recall@K, NDCG@K, ...) and full-ranking evaluation protocol
│   └── utils/                # Seeding, logging, config loading
├── scripts/                  # Entry points only — thin wrappers around src/ (prepare_data.py, train.py, evaluate.py)
├── notebooks/                # Colab launchers. No logic lives here; notebooks only call scripts/.
├── experiments/
│   ├── logs/                 # Raw training logs, one subdir per run (git-ignored)
│   └── results/              # Final metric tables (CSV/MD), committed — the source of truth for the paper trail
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

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Place the SNAP raw file (see below) at data/raw/loc-gowalla_totalCheckins.txt.gz

# 3. Build processed artifacts (coordinates + timestamped sequences joined onto official IDs)
python scripts/prepare_data.py

# 4. Train (example)
python scripts/train.py --config configs/lightgcn_gowalla.yaml
```

## Data sources

| File | Source | Role |
|---|---|---|
| `data/gowalla/{train,test,user_list,item_list}.txt` | [LightGCN official repo](https://github.com/kuandeng/LightGCN) `Data/gowalla/` | Benchmark split (interaction IDs only) |
| `data/raw/loc-gowalla_totalCheckins.txt.gz` | [SNAP: Gowalla](https://snap.stanford.edu/data/loc-gowalla.html) | Coordinates + timestamps, joined back via `org_id` |
