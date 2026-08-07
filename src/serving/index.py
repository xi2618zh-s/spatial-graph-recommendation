"""Item/user embedding export + FAISS index builders for M9.

The frozen recall model scores with a RAW inner product
(`ue[user_ids] @ ie.T`, see src/models/mf.py::full_scores) — no L2
normalization anywhere in training or evaluation. So `IndexFlatIP` on the
UN-normalized propagated embeddings is the exact mathematical equivalent of
`model.full_scores()`, not an approximation of it; that equivalence is what
tests/test_ann_consistency.py checks. Do not normalize these embeddings
before indexing — that would silently change the metric to cosine and break
the exact/gold-standard property Flat is supposed to have here.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch


def export_embeddings(model, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Returns (user_embeddings, item_embeddings), both (n, dim) float32,
    straight from the model's own propagate() -- the same call full_scores()
    uses internally."""
    model.eval()
    with torch.no_grad():
        ue, ie = model.propagate()
    return ue.cpu().numpy().astype("float32"), ie.cpu().numpy().astype("float32")


def build_flat_index(item_emb: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(item_emb)
    return index


def build_hnsw_index(item_emb: np.ndarray, M: int = 32, ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(item_emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(item_emb)
    return index


def build_ivf_index(item_emb: np.ndarray, nlist: int = 100) -> faiss.IndexIVFFlat:
    dim = item_emb.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(item_emb)
    index.add(item_emb)
    return index


def build_all_indices(item_emb: np.ndarray, cfg: dict) -> dict:
    """Returns {name: (index, build_seconds)}."""
    out = {}
    for name, builder, kwargs in [
        ("flat", build_flat_index, {}),
        ("hnsw", build_hnsw_index, {"M": cfg["hnsw"]["M"], "ef_construction": cfg["hnsw"]["ef_construction"]}),
        ("ivf", build_ivf_index, {"nlist": cfg["ivf"]["nlist"]}),
    ]:
        t0 = time.perf_counter()
        index = builder(item_emb, **kwargs)
        out[name] = (index, time.perf_counter() - t0)
    return out


def save_index(index: faiss.Index, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return path.stat().st_size


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))
