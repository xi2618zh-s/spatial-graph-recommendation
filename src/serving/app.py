"""M9 minimal FastAPI serving demo.

Loads the frozen recall model, its FAISS indices, and the persisted M7 GBDT
ranker once at startup; `/recommend` does retrieval (candidate_score) then
re-ranking (ranked_score) through the SAME feature pipeline M6/M7/M8 already
validated, using each user's FULL known history (not the M6 leave-last-out
internal-validation prefix, which only exists for offline evaluation).

This is a benchmark/demo harness, not a production deployment: single
process, no auth, no batching queue, no autoscaling. See docs/05_ann_serving.md
for the capacity reasoning behind that scope.

Run:
    uvicorn src.serving.app:app --reload
"""

import json
import pickle
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.data.dataset import GowallaData
from src.features.item_features import ItemFeatureStore, compute_item_popularity
from src.features.row_builder import build_full_row, build_user_context
from src.models.registry import load_checkpoint
from src.serving.index import load_index
from src.serving.search import masked_search
from src.train.ranker_trainer import score as ranker_score
from src.utils.common import ROOT, load_config

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(ROOT / "configs" / "ann_index.yaml")
    index_dir = ROOT / cfg["index_dir"]
    metadata = json.loads((index_dir / "metadata.json").read_text())

    run_cfg, data, model, _ = load_checkpoint(cfg["recall_run"], device="cpu")
    user_emb = np.load(index_dir / "user_embeddings.npy")
    item_emb = np.load(index_dir / "item_embeddings.npy")
    indices = {
        "flat": load_index(index_dir / "flat.index"),
        "hnsw": load_index(index_dir / "hnsw.index"),
        "ivf": load_index(index_dir / "ivf.index"),
    }
    indices["ivf"].nprobe = cfg["ivf"]["nprobe_values"][-1]
    indices["hnsw"].hnsw.efSearch = cfg["hnsw"]["ef_search_values"][-1]

    with open(ROOT / "data" / "processed" / "train_sequences_ts.pkl", "rb") as f:
        seqs_ts = pickle.load(f)
    serving_context = {
        u: {"prefix_items": [it for it, _ in seq], "prefix_ts": [ts for _, ts in seq]}
        for u, seq in seqs_ts.items() if seq
    }
    item_pop = compute_item_popularity(serving_context, data.n_items)
    item_store = ItemFeatureStore(
        serving_context, data.n_items, coords_csv=ROOT / "data" / "processed" / "poi_coords.csv",
    )

    gbdt_dir = ROOT / "experiments" / "logs" / "ranker_gbdt"
    gbdt_cfg = json.loads((gbdt_dir / "config.json").read_text())
    gbdt_model = joblib.load(gbdt_dir / "model.pkl")

    STATE.update({
        "cfg": cfg, "metadata": metadata, "data": data, "user_emb": user_emb, "item_emb": item_emb,
        "indices": indices, "serving_context": serving_context, "item_pop": item_pop,
        "item_store": item_store, "gbdt_model": gbdt_model, "gbdt_cols": gbdt_cfg["feature_columns"],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    yield
    STATE.clear()


app = FastAPI(title="spatial-graph-recommendation (demo)", lifespan=lifespan)


@app.get("/health")
def health():
    if not STATE:
        raise HTTPException(503, "not ready")
    return {
        "status": "ok",
        "recall_run": STATE["cfg"]["recall_run"],
        "ranker": "ranker_gbdt",
        "n_users": STATE["data"].n_users,
        "n_items": STATE["data"].n_items,
        "index_types": list(STATE["indices"].keys()),
        "index_built_at": STATE["metadata"]["built_at"],
        "started_at": STATE["started_at"],
    }


@app.get("/recommend")
def recommend(user_id: int, k: int = 20, candidate_k: int = 200, index: str = "flat"):
    if index not in STATE.get("indices", {}):
        raise HTTPException(400, f"unknown index '{index}', choose from {list(STATE['indices'])}")
    ctx = STATE["serving_context"].get(user_id)
    if ctx is None:
        raise HTTPException(404, f"unknown user_id {user_id}")

    t0 = time.perf_counter()
    query_ts = int(time.time())
    exclude = set(ctx["prefix_items"])
    faiss_index = STATE["indices"][index]
    cand_ids, cand_scores = masked_search(
        faiss_index, STATE["user_emb"][user_id], exclude, candidate_k
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ufeat, ctxfeat = build_user_context(
        ctx["prefix_items"], ctx["prefix_ts"], STATE["item_pop"], STATE["item_store"].coords, query_ts
    )
    rows = [
        build_full_row(user_id, int(it), 0, "serving_candidate", "n/a", query_ts,
                       score=float(sc), rank=rank, ufeat=ufeat, ctxfeat=ctxfeat,
                       item_store=STATE["item_store"])
        for rank, (it, sc) in enumerate(zip(cand_ids, cand_scores))
    ]
    cand_df = pd.DataFrame(rows)
    ranked = ranker_score(STATE["gbdt_model"], cand_df, STATE["gbdt_cols"])
    cand_df["ranked_score"] = ranked
    cand_df = cand_df.sort_values("ranked_score", ascending=False).head(k)
    ranking_ms = (time.perf_counter() - t0) * 1000

    return {
        "user_id": user_id,
        "index_used": index,
        "recall_run": STATE["cfg"]["recall_run"],
        "ranker": "ranker_gbdt",
        "timing_ms": {"retrieval": round(retrieval_ms, 3), "feature_and_rank": round(ranking_ms, 3)},
        "recommendations": [
            {"item_id": int(r.item_id), "candidate_score": float(r.cross_recall_score),
             "ranked_score": float(r.ranked_score), "candidate_rank": int(r.cross_candidate_rank)}
            for r in cand_df.itertuples()
        ],
    }
