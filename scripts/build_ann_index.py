"""M9: export item/user embeddings from the frozen recall model and build
Flat (exact gold standard) / HNSW / IVF FAISS indices.

Usage:
    python scripts/build_ann_index.py --config configs/ann_index.yaml

Outputs under `<index_dir>/` (local, gitignored -- like experiments/logs):
    item_embeddings.npy, user_embeddings.npy
    flat.index, hnsw.index, ivf.index
    metadata.json   dim, n_items, recall_run, per-index build time + disk size
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.models.registry import load_checkpoint
from src.serving.index import build_all_indices, export_embeddings, save_index, write_metadata
from src.utils.common import ROOT, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ann_index.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    index_dir = ROOT / cfg["index_dir"]
    index_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run_cfg, data, model, _ = load_checkpoint(cfg["recall_run"], device="cpu")
    user_emb, item_emb = export_embeddings(model)
    print(f"exported embeddings: users {user_emb.shape}, items {item_emb.shape} "
          f"in {time.time() - t0:.1f}s")

    np.save(index_dir / "item_embeddings.npy", item_emb)
    np.save(index_dir / "user_embeddings.npy", user_emb)

    indices = build_all_indices(item_emb, cfg)
    sizes = {}
    for name, (index, build_s) in indices.items():
        size = save_index(index, index_dir / f"{name}.index")
        sizes[name] = size
        print(f"{name}: build={build_s:.3f}s, disk={size / 1e6:.2f}MB, ntotal={index.ntotal}")

    write_metadata(index_dir / "metadata.json", {
        "recall_run": cfg["recall_run"], "run_config": run_cfg,
        "n_items": data.n_items, "n_users": data.n_users, "dim": item_emb.shape[1],
        "build": {name: {"build_seconds": round(indices[name][1], 4), "disk_bytes": sizes[name]}
                 for name in indices},
        "hnsw_params": cfg["hnsw"], "ivf_params": cfg["ivf"],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"\nWrote embeddings + indices + metadata.json to {index_dir}")


if __name__ == "__main__":
    main()
