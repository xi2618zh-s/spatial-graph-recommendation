# Project plan & milestone log

> Status reflects `main` as of the last reconciliation against `experiments/results/summary.csv`
> and `git log` (see `PROJECT_HANDOFF_V2.md` for the full audit). "runs pending" below means no
> row in `summary.csv` yet on this machine/branch — some of these runs may already be complete on
> Colab/Drive but are not yet synced and verified locally, so they are not claimed as done.

| # | Milestone | Deliverable | Status |
|---|---|---|---|
| 0 | Repo skeleton + data verified | this repo, prepare_data.py | done |
| 1 | Data pipeline + eval protocol | metrics module, full-ranking evaluator | done |
| 2 | Baselines: Pop / ItemCF / MF-BPR | results table v1 | **done** — 3 rows in `summary.csv` |
| 3 | LightGCN reproduction | Recall@20 within ~5% of 0.183 | **done** — R@20=0.17724 (-3.1% vs paper), synced 2026-08-07, 3-way verified (no DONE marker; run predates that feature, completion confirmed via history.json patience pattern) |
| 4 | Spatial-LightGCN (geo-enhanced graph) | ablation: spatial edges on/off | main run (k10, λ0.3) **done** — R@20=0.18335 (+3.45% vs LightGCN), synced 2026-08-07, 4-way verified; λ0.1/λ0.5/k20 ablations still running on Colab, pending sync |
| 5 | SASRec sequential retrieval | results table v2 | main run (200ep) **done** — R@20=0.12577, synced 2026-08-07, 4-way verified, weaker than graph models and still improving at final epoch; 400ep extended run still running on Colab, pending sync |
| 6 | Sample construction & feature engineering | point-in-time ranking dataset | planned |
| 7 | Two-stage ranking (retrieval-score / LR / GBDT) | end-to-end Recall/NDCG | planned |
| 8 | Business-proxy metrics & bias/cold-start diagnostics | slice reports | planned |
| 9 | FAISS ANN retrieval + minimal serving | recall-latency curves, FastAPI smoke test | planned |
| 10 | Engineering reliability (RNG, idempotent results, smoke tests) | test suite | persistence guard (Colab ephemeral-storage check) implemented; remaining items planned |
| 11 | README, resume & interview evidence packaging | final docs | planned |
| 12 | (Conditional extension) multi-source retrieval, σ=1km, cold-item fallback, RQ-VAE | condition-triggered only | not started |

Experiment discipline: every run = 1 config in `configs/` + 1 row in
`experiments/results/` + raw log in `experiments/logs/<run_name>/`.
