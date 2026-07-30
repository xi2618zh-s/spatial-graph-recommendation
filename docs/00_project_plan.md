# Project plan & milestone log

| # | Milestone | Deliverable | Status |
|---|---|---|---|
| 0 | Repo skeleton + data verified | this repo, prepare_data.py | done |
| 1 | Data pipeline + eval protocol | metrics module, full-ranking evaluator | done |
| 2 | Baselines: Pop / ItemCF / MF-BPR | results table v1 | code done, runs pending |
| 3 | LightGCN reproduction | Recall@20 within ~5% of 0.183 | code done, runs pending |
| 4 | Spatial-LightGCN (geo-enhanced graph) | ablation: spatial edges on/off | |
| 5 | SASRec sequential retrieval | results table v2 | |
| 6 | DNN ranking stage (two-stage pipeline) | end-to-end Recall/NDCG | |
| 7 | FAISS ANN retrieval + latency notes | serving writeup | |
| 8 | (Extension) RQ-VAE semantic IDs | in progress marker only | |

Experiment discipline: every run = 1 config in `configs/` + 1 row in
`experiments/results/` + raw log in `experiments/logs/<run_name>/`.
