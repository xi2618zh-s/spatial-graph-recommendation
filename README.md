# Spatial Graph Recommendation

Graph-enhanced, location-aware personalized recommendation system on the Gowalla
check-in dataset. Implements a full two-stage pipeline (candidate retrieval →
ranking) with progressively stronger retrieval models:

```
Popularity → ItemCF → MF-BPR → LightGCN → Spatial-LightGCN
                                              +
                                   SASRec (sequential) · DNN Ranker · FAISS ANN retrieval
```

Benchmark: the standard NGCF/LightGCN split of Gowalla
(29,858 users · 40,981 POIs · ~1.03M interactions).
Reference numbers to reproduce: LightGCN Recall@20 ≈ 0.183, NDCG@20 ≈ 0.155.

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
