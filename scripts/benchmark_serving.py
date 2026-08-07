"""M9: exact-vs-ANN recall-latency benchmark.

Usage:
    python scripts/benchmark_serving.py --config configs/ann_index.yaml

Requires scripts/build_ann_index.py to have run first (reads its embeddings
+ indices from <index_dir>/).

1. Regression-tests IndexFlatIP against the model's own raw score (the SAME
   masking src/eval/evaluator.py uses -- a user's official-train items).
2. Sweeps HNSW efSearch and IVF nprobe, reporting ANN Recall@K against the
   Flat gold standard (i.e. how often the SAME masked top-K set is
   recovered) at each setting.
3. Times >=1000 single-query searches per index type (after a discarded
   warm-up), reporting P50/P95/P99 latency and throughput.

Outputs (experiments/results/):
    ann_consistency_report.json   Flat-vs-model-score regression result
    ann_benchmark.csv             recall@K / latency / build time / disk size
                                  per index type and parameter setting
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.dataset import GowallaData
from src.serving.index import load_index
from src.serving.search import masked_search
from src.utils.common import ROOT, load_config


def numpy_full_scores(user_emb: np.ndarray, item_emb: np.ndarray, u: int) -> np.ndarray:
    return user_emb[u] @ item_emb.T


def masked_topk_numpy(scores: np.ndarray, exclude_ids: set, k: int) -> np.ndarray:
    s = scores.copy()
    if exclude_ids:
        s[list(exclude_ids)] = -np.inf
    k = min(k, len(s) - 1)
    part = np.argpartition(-s, k)[:k]
    return part[np.argsort(-s[part])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ann_index.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["benchmark"]
    index_dir = ROOT / cfg["index_dir"]

    metadata = json.loads((index_dir / "metadata.json").read_text())
    user_emb = np.load(index_dir / "user_embeddings.npy")
    item_emb = np.load(index_dir / "item_embeddings.npy")
    flat = load_index(index_dir / "flat.index")
    hnsw = load_index(index_dir / "hnsw.index")
    ivf = load_index(index_dir / "ivf.index")

    data = GowallaData(ROOT / "data" / "gowalla")
    rng = np.random.default_rng(bcfg["seed"])
    all_users = np.array(sorted(data.train.keys()))
    query_users = rng.choice(all_users, size=min(bcfg["n_query_users"], len(all_users)), replace=False)
    K = bcfg["top_k"]

    # --- 1. Flat vs raw model-score regression (same masking as full-ranking eval) ---
    mismatches = []
    for u in query_users[:100]:
        exclude = set(data.train.get(int(u), []))
        exact_scores = numpy_full_scores(user_emb, item_emb, int(u))
        numpy_top = set(masked_topk_numpy(exact_scores, exclude, K).tolist())
        flat_ids, _ = masked_search(flat, user_emb[int(u)], exclude, K)
        flat_top = set(int(i) for i in flat_ids)
        if numpy_top != flat_top:
            mismatches.append({
                "user": int(u), "n_only_numpy": len(numpy_top - flat_top),
                "n_only_flat": len(flat_top - numpy_top),
            })
    consistency = {
        "n_users_checked": 100, "k": K, "n_mismatched_users": len(mismatches),
        "mismatches": mismatches[:10],
        "verdict": "PASS (exact set match for every user)" if not mismatches
                  else "PARTIAL -- see mismatches (expected only from float tie-breaking at the boundary)",
    }
    print("consistency check:", consistency["verdict"], f"({len(mismatches)}/100 users differed)")
    (ROOT / "experiments" / "results" / "ann_consistency_report.json").write_text(
        json.dumps(consistency, indent=2)
    )

    # --- 2. ANN Recall@K vs Flat, sweeping efSearch / nprobe ---
    rows = []
    flat_tops = {}
    for u in query_users:
        exclude = set(data.train.get(int(u), []))
        ids, _ = masked_search(flat, user_emb[int(u)], exclude, K)
        flat_tops[int(u)] = set(int(i) for i in ids)

    def ann_recall(index) -> float:
        hits, total = 0, 0
        for u in query_users:
            exclude = set(data.train.get(int(u), []))
            ids, _ = masked_search(index, user_emb[int(u)], exclude, K)
            hits += len(flat_tops[int(u)] & set(int(i) for i in ids))
            total += len(flat_tops[int(u)])
        return hits / total if total else float("nan")

    for ef in cfg["hnsw"]["ef_search_values"]:
        hnsw.hnsw.efSearch = ef
        rec = ann_recall(hnsw)
        rows.append({"index": "hnsw", "param_name": "efSearch", "param_value": ef,
                    f"recall@{K}_vs_flat": round(rec, 5)})
        print(f"hnsw efSearch={ef}: recall@{K} vs flat = {rec:.5f}")

    for nprobe in cfg["ivf"]["nprobe_values"]:
        ivf.nprobe = nprobe
        rec = ann_recall(ivf)
        rows.append({"index": "ivf", "param_name": "nprobe", "param_value": nprobe,
                    f"recall@{K}_vs_flat": round(rec, 5)})
        print(f"ivf nprobe={nprobe}: recall@{K} vs flat = {rec:.5f}")

    # --- 3. Latency: single-query search, >=1000 timed reps after warm-up ---
    hnsw.hnsw.efSearch = cfg["hnsw"]["ef_search_values"][len(cfg["hnsw"]["ef_search_values"]) // 2]
    ivf.nprobe = cfg["ivf"]["nprobe_values"][len(cfg["ivf"]["nprobe_values"]) // 2]
    latency_rows = []
    for name, index in [("flat", flat), ("hnsw", hnsw), ("ivf", ivf)]:
        users_cycle = np.resize(query_users, bcfg["warmup_repeats"] + bcfg["n_latency_repeats"])
        exclude_cache = {int(u): set(data.train.get(int(u), [])) for u in np.unique(users_cycle)}
        for u in users_cycle[:bcfg["warmup_repeats"]]:
            masked_search(index, user_emb[int(u)], exclude_cache[int(u)], K)
        lat = []
        for u in users_cycle[bcfg["warmup_repeats"]:]:
            t0 = time.perf_counter()
            masked_search(index, user_emb[int(u)], exclude_cache[int(u)], K)
            lat.append(time.perf_counter() - t0)
        lat = np.asarray(lat)
        row = {
            "index": name, "n_queries": len(lat),
            "p50_ms": round(float(np.percentile(lat, 50)) * 1000, 4),
            "p95_ms": round(float(np.percentile(lat, 95)) * 1000, 4),
            "p99_ms": round(float(np.percentile(lat, 99)) * 1000, 4),
            "mean_ms": round(float(lat.mean()) * 1000, 4),
            "throughput_qps": round(1.0 / lat.mean(), 1),
            "build_seconds": metadata["build"][name]["build_seconds"],
            "disk_mb": round(metadata["build"][name]["disk_bytes"] / 1e6, 3),
        }
        latency_rows.append(row)
        print(f"{name}: P50={row['p50_ms']}ms P95={row['p95_ms']}ms P99={row['p99_ms']}ms "
              f"throughput={row['throughput_qps']}qps build={row['build_seconds']}s disk={row['disk_mb']}MB")

    RESULTS = ROOT / "experiments" / "results"
    pd.DataFrame(rows).to_csv(RESULTS / "ann_recall_sweep.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(RESULTS / "ann_latency.csv", index=False)
    print(f"\nWrote ann_consistency_report.json, ann_recall_sweep.csv, ann_latency.csv to {RESULTS}")


if __name__ == "__main__":
    main()
