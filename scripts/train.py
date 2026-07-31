"""Unified training entry point.

Usage:
    python scripts/train.py --config configs/mf_gowalla.yaml
    python scripts/train.py --config configs/lightgcn_gowalla.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import GowallaData
from src.models.lightgcn import LightGCN
from src.models.mf import MFBPR
from src.train.bpr_trainer import train_bpr
from src.utils.common import ROOT, Timer, load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.ckpt if present (Colab disconnects)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected — LightGCN training on CPU is very slow. "
              "Run this on Colab.")

    data = GowallaData(ROOT / cfg["data"]["split_dir"])
    mc = cfg["model"]
    if mc["name"] == "mf":
        model = MFBPR(data.n_users, data.n_items, dim=mc["embedding_dim"])
    elif mc["name"] == "lightgcn":
        with Timer("build normalized adjacency"):
            adj = data.norm_adj()
        model = LightGCN(data.n_users, data.n_items, dim=mc["embedding_dim"],
                         n_layers=mc["n_layers"], norm_adj=adj)
    elif mc["name"] == "sasrec":
        from src.data.sequences import SequenceData
        from src.models.sasrec import SASRec
        from src.train.sasrec_trainer import train_sasrec
        seq_data = SequenceData(
            ROOT / "data" / "processed" / "train_sequences.pkl",
            n_items=data.n_items, train_sets=data._train_sets,
            max_len=mc["max_len"],
        )
        model = SASRec(data.n_items, dim=mc["embedding_dim"],
                       max_len=mc["max_len"], n_blocks=mc["n_blocks"],
                       n_heads=mc["n_heads"], dropout=mc["dropout"])
        train_sasrec(model, seq_data, data, cfg, device, resume=args.resume)
        return
    elif mc["name"] == "spatial_lightgcn":
        from src.data.spatial_graph import build_combined_adj
        with Timer("build combined interaction+spatial adjacency"):
            adj = build_combined_adj(
                data,
                coords_csv=ROOT / "data" / "processed" / "poi_coords.csv",
                k=mc["spatial"]["knn"],
                lam=mc["spatial"]["lambda"],
                max_dist_km=mc["spatial"]["max_dist_km"],
                sigma_km=mc["spatial"].get("sigma_km"),
            )
        model = LightGCN(data.n_users, data.n_items, dim=mc["embedding_dim"],
                         n_layers=mc["n_layers"], norm_adj=adj)
    else:
        sys.exit(f"unknown model: {mc['name']}")

    train_bpr(model, data, cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
