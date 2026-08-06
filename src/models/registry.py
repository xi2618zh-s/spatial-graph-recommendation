"""Reconstruct a model + full-item score_fn from a run's own logged config —
mirrors scripts/train.py's dispatch, but for reloading a trained checkpoint
(bootstrap CI, candidate generation) rather than training one from scratch."""

import sys

import numpy as np
import torch

from src.data.dataset import GowallaData
from src.models.lightgcn import LightGCN
from src.models.mf import MFBPR
from src.utils.common import ROOT


def build_model_and_scorer(cfg: dict, data: GowallaData, device: str = "cpu"):
    mc = cfg["model"]
    if mc["name"] == "mf":
        model = MFBPR(data.n_users, data.n_items, dim=mc["embedding_dim"])
    elif mc["name"] == "lightgcn":
        adj = data.norm_adj()
        model = LightGCN(data.n_users, data.n_items, dim=mc["embedding_dim"],
                         n_layers=mc["n_layers"], norm_adj=adj)
    elif mc["name"] == "spatial_lightgcn":
        from src.data.spatial_graph import build_combined_adj
        adj = build_combined_adj(
            data, coords_csv=ROOT / "data" / "processed" / "poi_coords.csv",
            k=mc["spatial"]["knn"], lam=mc["spatial"]["lambda"],
            max_dist_km=mc["spatial"]["max_dist_km"],
            sigma_km=mc["spatial"].get("sigma_km"),
        )
        model = LightGCN(data.n_users, data.n_items, dim=mc["embedding_dim"],
                         n_layers=mc["n_layers"], norm_adj=adj)
    elif mc["name"] == "sasrec":
        from src.data.sequences import SequenceData
        from src.models.sasrec import SASRec
        seq_data = SequenceData(
            ROOT / "data" / "processed" / "train_sequences.pkl",
            n_items=data.n_items, train_sets=data._train_sets,
            max_len=mc["max_len"],
        )
        model = SASRec(data.n_items, dim=mc["embedding_dim"], max_len=mc["max_len"],
                       n_blocks=mc["n_blocks"], n_heads=mc["n_heads"], dropout=mc["dropout"])
        model = model.to(device)

        def score_fn(user_ids):
            model.eval()
            with torch.no_grad():
                inp, length = seq_data.eval_inputs(np.asarray(user_ids))
                inp = torch.as_tensor(inp, device=device)
                length = torch.as_tensor(length, device=device)
                return model.full_scores(inp, length).cpu().numpy()

        return model, score_fn
    else:
        sys.exit(f"unknown model: {mc['name']}")

    model = model.to(device)

    def score_fn(user_ids):
        model.eval()
        with torch.no_grad():
            u = torch.as_tensor(np.asarray(user_ids), device=device)
            return model.full_scores(u).cpu().numpy()

    return model, score_fn


def load_checkpoint(run_name: str, device: str = "cpu"):
    """Load (cfg, data, model, score_fn) for a completed run's best.pt."""
    import json

    log_dir = ROOT / "experiments" / "logs" / run_name
    cfg = json.loads((log_dir / "config.json").read_text())
    ckpt_path = log_dir / "best.pt"
    if not ckpt_path.exists():
        sys.exit(f"no best.pt found for run {run_name} at {ckpt_path}")

    data = GowallaData(ROOT / cfg["data"]["split_dir"])
    model, score_fn = build_model_and_scorer(cfg, data, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return cfg, data, model, score_fn
