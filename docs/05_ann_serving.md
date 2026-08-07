# M9 — FAISS ANN retrieval & minimal serving

```
python scripts/build_ann_index.py --config configs/ann_index.yaml
python scripts/benchmark_serving.py --config configs/ann_index.yaml
uvicorn src.serving.app:app --reload   # then GET /health, GET /recommend?user_id=0&k=10
```

**Scale disclosure, up front:** 40,981 POIs is far below industrial catalog sizes (tens of
millions to billions). Every number on this page is a **serving benchmark harness**, not evidence
of solving large-scale retrieval — the point of this milestone is showing the recall/latency/
resource trade-offs are understood well enough to make a correct decision, not to prove FAISS
"works." As the results below show, the correct decision at this scale may well be "don't use
ANN" — that is a legitimate engineering conclusion, not a failure to demonstrate FAISS.

## Exactness: FlatIP is the model's own score, not an approximation of it

The frozen recall model scores with a raw, un-normalized inner product (`src/models/mf.py::full_scores`).
`IndexFlatIP` on the same un-normalized item embeddings computes the identical mathematical
quantity — no cosine normalization is applied anywhere in this pipeline. `tests/test_ann_consistency.py`
verifies this is not just true in theory: for 20 users compared here (and 100 in the benchmark
script's own regression check), **`IndexFlatIP`'s masked top-K exactly matched the raw model
score's masked top-K in 100% of cases** — same masking convention (`data.train[user]`) as
`src/eval/evaluator.py` uses for the official full-ranking evaluation.

## Recall@200 vs Flat (masked, 500 sampled users, `experiments/results/ann_recall_sweep.csv`)

| HNSW `efSearch` | Recall@200 | | IVF `nprobe` | Recall@200 |
|---:|---:|---|---:|---:|
| 16 | 0.4857 | | 1 | 0.5375 |
| 32 | 0.5712 | | 4 | 0.8527 |
| 64 | 0.6554 | | 8 | 0.9387 |
| 128 | 0.7278 | | 16 | 0.9767 |
| 256 | 0.7864 | | 32 | 0.9925 |
| 512 | 0.8352 | | 64 | **0.9991** |
| 1024 | 0.8754 | | | |

**IVF reaches near-exact recall (99.9%) at `nprobe=64`. HNSW, even pushed to `efSearch=1024`
(4x the largest value in the original sweep, tested specifically because the curve had not
plateaued), only reaches 87.5%** and is still climbing slowly, not saturating. This is reported
exactly as measured — the instinct to stop at a "reasonable-looking" `efSearch=256` and call it
done would have hidden that HNSW simply behaves worse than IVF for this embedding distribution
and index build (`M=32`, `efConstruction=200`), not merely "not yet tuned enough."

**Why:** IVF's cluster-based partitioning apparently aligns well with how `spatial_lightgcn_k10_lam0.3`'s
item embeddings are distributed (plausibly because the spatial graph term pulls geographically
close items' embeddings together into cluster-like structure); HNSW's greedy graph search is
comparatively less sample-efficient here. This is inferred from the recall curves, not verified
by inspecting the embedding geometry directly — a next step if this distinction mattered for a
production decision, not resolved in this milestone.

## Latency, build time, disk (1,200 timed single-query searches after 50 discarded warm-up queries)

| Index | P50 | P95 | P99 | Throughput | Build time | Disk size |
|---|---:|---:|---:|---:|---:|---:|
| Flat (exact) | 1.60ms | 2.64ms | 3.25ms | 569 qps | 0.006s | 10.5MB |
| HNSW (`efSearch`=256) | 0.77ms | 1.15ms | 1.41ms | 1,235 qps | 2.25s | 21.6MB |
| IVF (`nprobe`=8) | 0.59ms | 0.96ms | 1.40ms | 1,568 qps | 0.13s | 10.8MB |

(Single CPU core, no batching, no warm cache tricks beyond the discarded warm-up queries; see
`configs/ann_index.yaml` for exact parameters and `scripts/benchmark_serving.py` for the timing
methodology.)

## Reading this correctly: is ANN worth it here?

- **In absolute terms, no.** Exact search over the full 40,981-item catalog already answers in
  ~1.6ms median, ~3.3ms P99 — nowhere near a latency budget any reasonable serving SLA would
  struggle with. The ANN speedup (roughly 2–3x lower median latency) is real but operates on an
  already-negligible number.
- **HNSW is not a free win here**: it costs the most to build (2.25s vs IVF's 0.13s), uses the
  most disk (2x Flat), and — per the recall table above — is also the *least* accurate of the
  three at any latency-comparable operating point. At this catalog size, HNSW is dominated by IVF
  on every axis measured.
- **IVF at `nprobe=8`–`16` is the closest thing to a "free" ANN choice** (93.9%–97.7% recall,
  lower latency than Flat, similar disk footprint to Flat, cheap to build) — but "free" here means
  "the cost is well below the noise floor of this benchmark," not "meaningfully better than doing
  nothing."
- **The correct engineering call at 41K items is Flat/exact**, and that conclusion itself is the
  deliverable: recognizing that a technique is unnecessary at the current scale is the same skill
  as recognizing when it becomes necessary. RISK_REGISTER.md §2.5 names this exact outcome as an
  acceptable, expected result — not a failure to make ANN work.
- **Where the decision would flip**: at catalog sizes where a single exact search meaningfully
  competes for CPU/memory bandwidth with concurrent request load — roughly when per-query exact
  latency crosses into the same order of magnitude as the serving SLA, or when the embedding
  matrix stops fitting comfortably in cache/RAM. Neither condition holds at 41K × 64-dim floats
  (~10MB).

## Minimal FastAPI service

`src/serving/app.py` loads the frozen recall model, all three FAISS indices, and the persisted M7
GBDT ranker once at startup (`lifespan`), then:

- `GET /health` — returns recall/ranker run names, item/user counts, available index types, and
  when the index was built — enough to trace which model/index version answered a request.
- `GET /recommend?user_id=&k=&candidate_k=&index=flat|hnsw|ivf` — retrieves `candidate_k`
  candidates from the chosen index (masked against the user's known history — every interaction
  in `data/processed/train_sequences_ts.pkl`, not the M6/M7 leave-last-out internal-validation
  prefix, since this endpoint emulates "recommend right now" rather than reproducing an offline
  eval point), reranks with `ranker_gbdt` through the exact same feature pipeline M6–M8 already
  validated, and returns both `candidate_score` (the raw recall-model score) and `ranked_score`
  (the GBDT ranker's output) per item, plus per-request retrieval/ranking timing.
- `tests/test_ann_consistency.py::test_recommend_endpoint_smoke` exercises `/health` and
  `/recommend` (including a 404 path for an unknown user) via FastAPI's `TestClient` — no live
  server process needed to verify the wiring is correct.

**What this is not**: no auth, no request batching/queueing, no autoscaling, no cache warm-up
strategy beyond the benchmark's own discarded warm-up queries, single process. Calling this a
"production service" would be exactly the kind of overclaim PROJECT_HANDOFF_V2.md §0.2 prohibits —
it is accurately described as a reproducible serving *benchmark harness* with a working demo
endpoint on top.

## Known limitations

- IVF's `train()` step used the full 40,981-item embedding set as its own training data (the only
  reasonable choice at this catalog size); this is not the "train on a held-out sample" pattern
  used for the ranking dataset's leakage discipline, but is standard practice for ANN index
  training and does not affect any reported recall/latency number's validity for its own purpose.
- HNSW's plateau was investigated one extra order of magnitude past the original config
  (`efSearch` up to 1024) but not exhaustively past that point, since the practical conclusion
  (IVF dominates HNSW at this scale) was already clear and further HNSW tuning would not change
  the Flat-vs-ANN decision above.
- All latency numbers are single-process, single-core CPU timings on this machine — not
  representative of any specific production hardware/network profile.
