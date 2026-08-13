"""V2 retriever training entry point (Track B, leakage-safe protocol).

Trains one of three Spatial-LightGCN snapshots against the H_inner|v0|
y_train|y_val temporal split (src/data/ranking_dataset_v2.py). Does NOT
import or modify scripts/train.py -- entirely separate entry point so the
currently-running Colab ablation queue (which uses scripts/train.py) is
never at risk from changes here.

Usage (protocol order matters):
    # 1. inner-validation: train on H_inner, early-stop against v0 -> picks E*
    python scripts/train_retriever_v2.py --config configs/ranking_v2/retriever_inner_validation.yaml --snapshot inner_validation

    # read E* from the DONE file it just wrote, e.g. best_epoch=240, then:

    # 2. R0: fresh init, fixed E* epochs on Hcore(=H_inner+v0), no early stopping
    python scripts/train_retriever_v2.py --config configs/ranking_v2/retriever_r0.yaml --snapshot r0 --epochs 240

    # 3. R1: fresh init, same fixed E* epochs on Hcore+y_train
    python scripts/train_retriever_v2.py --config configs/ranking_v2/retriever_r1.yaml --snapshot r1 --epochs 240

Not run in this session -- see the approved ranking-v2 plan: Phase 1 (GPU)
only starts once the user confirms the k20 ablation queue is finished,
recovered, and verified.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.utils.common as common_module
from src.data.dataset import GowallaData
from src.data.ranking_dataset import load_timestamped_sequences
from src.data.ranking_dataset_v2 import (
    SnapshotGowallaData, build_v2_layers, hcore_edges, hcore_plus_ytrain_edges, v0_targets,
)
from src.data.spatial_graph import build_combined_adj
from src.models.lightgcn import LightGCN
from src.train.bpr_trainer import train_bpr
from src.utils.common import ROOT, check_persistent_storage, load_config, set_seed

TRAIN_EDGE_BUILDERS = {
    "inner_validation": lambda layers: {u: L["h_inner_items"] for u, L in layers.items()},
    "r0": hcore_edges,
    "r1": hcore_plus_ytrain_edges,
}

# train_bpr() calls append_result() at the end, which writes to
# src.utils.common.RESULTS_CSV -- the SAME shared experiments/results/summary.csv
# the Track A ablation queue's results live in. None of the three V2
# snapshots (inner_validation/r0/r1) are meant to land there: inner_validation
# is a throwaway run whose checkpoint is discarded, and r0/r1 are retrieval
# infrastructure for the ranker, not Track A results to compare against
# LightGCN/Spatial-LightGCN in the shared table. Redirect before ANY training
# starts so a run interrupted mid-training never has a chance to write there.
common_module.RESULTS_CSV = ROOT / "experiments" / "results" / "ranking_v2_retriever_results.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--snapshot", required=True, choices=list(TRAIN_EDGE_BUILDERS))
    ap.add_argument("--epochs", type=int, default=None,
                    help="fixed E* epoch budget for r0/r1 (required for those two; "
                         "inner_validation uses the config's own early-stopping instead)")
    ap.add_argument("--allow-ephemeral", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.snapshot in ("r0", "r1") and args.epochs is None:
        sys.exit(f"--snapshot {args.snapshot} requires --epochs E* "
                 "(read it from the inner_validation run's DONE file first)")

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        cfg["train"]["eval_every"] = args.epochs  # exactly one save, at epoch E*
        cfg["train"]["early_stop_patience"] = 10 ** 6  # no early-stopping decision for r0/r1

    check_persistent_storage(
        ROOT / "experiments" / "logs" / cfg["experiment_name"], allow_ephemeral=args.allow_ephemeral
    )
    set_seed(cfg["seed"])

    official = GowallaData(ROOT / "data" / "gowalla")  # read-only, only used for n_users/n_items
    seqs = load_timestamped_sequences(ROOT / "data" / "processed" / "train_sequences_ts.pkl")
    layers = build_v2_layers(seqs, min_h_inner=cfg.get("min_h_inner", 5))
    print(f"eligible users: {len(layers)} / {len(seqs)}")

    train_edges = TRAIN_EDGE_BUILDERS[args.snapshot](layers)
    test_edges = v0_targets(layers)  # shared validation target; see module docstring for why r0/r1 reuse is harmless
    data = SnapshotGowallaData(train_edges, test_edges, official.n_users, official.n_items)

    mc = cfg["model"]
    adj = build_combined_adj(
        data, coords_csv=ROOT / "data" / "processed" / "poi_coords.csv",
        k=mc["spatial"]["knn"], lam=mc["spatial"]["lambda"],
        max_dist_km=mc["spatial"]["max_dist_km"], sigma_km=mc["spatial"].get("sigma_km"),
    )
    model = LightGCN(data.n_users, data.n_items, dim=mc["embedding_dim"],
                     n_layers=mc["n_layers"], norm_adj=adj)

    train_bpr(model, data, cfg, device="cpu", resume=args.resume)


if __name__ == "__main__":
    main()
