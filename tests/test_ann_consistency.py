"""M9: IndexFlatIP must reproduce the frozen model's own raw score exactly
(same masking convention as src/eval/evaluator.py), and the FastAPI service
must return a valid response for a smoke-test request."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.data.dataset import GowallaData
from src.serving.index import load_index
from src.serving.search import masked_search
from src.utils.common import ROOT

INDEX_DIR = ROOT / "experiments" / "ann_index"
GBDT_MODEL = ROOT / "experiments" / "logs" / "ranker_gbdt" / "model.pkl"


@pytest.mark.skipif(not INDEX_DIR.exists(), reason="run scripts/build_ann_index.py first")
def test_flat_index_matches_raw_model_score():
    data = GowallaData(ROOT / "data" / "gowalla")
    user_emb = np.load(INDEX_DIR / "user_embeddings.npy")
    item_emb = np.load(INDEX_DIR / "item_embeddings.npy")
    flat = load_index(INDEX_DIR / "flat.index")

    users = sorted(data.train.keys())[:20]
    k = 50
    for u in users:
        exclude = set(data.train.get(u, []))
        exact_scores = user_emb[u] @ item_emb.T
        exact_scores[list(exclude)] = -np.inf
        exact_top = set(np.argsort(-exact_scores)[:k].tolist())

        flat_ids, _ = masked_search(flat, user_emb[u], exclude, k)
        flat_top = set(int(i) for i in flat_ids)

        assert exact_top == flat_top, f"user {u}: Flat index top-{k} diverged from raw model score"


@pytest.mark.skipif(not INDEX_DIR.exists() or not GBDT_MODEL.exists(),
                    reason="run scripts/build_ann_index.py and scripts/evaluate_pipeline.py first")
def test_recommend_endpoint_smoke():
    from src.serving.app import app

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        # user 0 always exists in the official split
        resp = client.get("/recommend", params={"user_id": 0, "k": 10})
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == 0
        assert len(body["recommendations"]) == 10
        item_ids = [r["item_id"] for r in body["recommendations"]]
        assert len(set(item_ids)) == len(item_ids), "duplicate items in one recommendation list"

        missing = client.get("/recommend", params={"user_id": 10**9, "k": 10})
        assert missing.status_code == 404
