"""Per-user metrics + bootstrap 95% CI for a completed, checkpointed run.

Reloads best.pt against the exact config the run was trained with
(experiments/logs/<run>/config.json — the source of truth, not configs/*.yaml,
since configs/ could have changed since the run happened), re-scores every
test user, and reports a confidence interval instead of a single point
estimate.

Usage:
    python scripts/compute_bootstrap_ci.py --run lightgcn_gowalla_repro
    python scripts/compute_bootstrap_ci.py --run spatial_lightgcn_k10_lam0.3
    python scripts/compute_bootstrap_ci.py --run sasrec_gowalla

Writes, under experiments/logs/<run>/:
    per_user_metrics.npz   raw per-user recall/ndcg arrays (for audit/reuse)
    bootstrap_ci.json      {metric: {mean, ci_low, ci_high, n_users, n_boot}}
and appends/updates one row per (run, metric) in experiments/results/bootstrap_ci.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.dataset import GowallaData
from src.eval.bootstrap import bootstrap_ci
from src.eval.evaluator import evaluate_per_user
from src.models.lightgcn import LightGCN
from src.models.mf import MFBPR
from src.utils.common import ROOT

CI_CSV = ROOT / "experiments" / "results" / "bootstrap_ci.csv"
CI_FIELDS = ["run_name", "model", "metric", "mean", "ci_low", "ci_high", "n_users", "n_boot"]


def build_model_and_scorer(cfg: dict, data: GowallaData, device: str):
    """Mirrors scripts/train.py's model dispatch, but for reloading a trained
    checkpoint rather than training one."""
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


def write_ci_csv(rows: list[dict]) -> None:
    existing = []
    if CI_CSV.exists():
        with open(CI_CSV, newline="") as f:
            existing = list(csv.DictReader(f))
    keep = [r for r in existing if r["run_name"] not in {row["run_name"] for row in rows}]
    keep.extend(rows)
    keep.sort(key=lambda r: (r["run_name"], r["metric"]))
    CI_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CI_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CI_FIELDS)
        w.writeheader()
        w.writerows(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_name under experiments/logs/")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--topk", type=int, nargs="+", default=[10, 20, 50])
    args = ap.parse_args()

    log_dir = ROOT / "experiments" / "logs" / args.run
    cfg = json.loads((log_dir / "config.json").read_text())
    ckpt_path = log_dir / "best.pt"
    if not ckpt_path.exists():
        sys.exit(f"no best.pt found for run {args.run} at {ckpt_path}")

    device = "cpu"
    data = GowallaData(ROOT / cfg["data"]["split_dir"])
    model, score_fn = build_model_and_scorer(cfg, data, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    per_user = evaluate_per_user(score_fn, data, topks=tuple(args.topk))

    ci = {m: bootstrap_ci(v, n_boot=args.n_boot, seed=cfg["seed"]) for m, v in per_user.items()}
    for m, c in ci.items():
        print(f"{m}: mean={c['mean']:.5f}  95% CI=[{c['ci_low']:.5f}, {c['ci_high']:.5f}]  "
              f"n_users={c['n_users']}")

    np.savez(log_dir / "per_user_metrics.npz", **per_user)
    (log_dir / "bootstrap_ci.json").write_text(json.dumps(ci, indent=2))

    rows = [
        {"run_name": args.run, "model": cfg["model"]["name"], "metric": m,
         "mean": round(c["mean"], 5), "ci_low": round(c["ci_low"], 5),
         "ci_high": round(c["ci_high"], 5), "n_users": c["n_users"], "n_boot": c["n_boot"]}
        for m, c in ci.items()
    ]
    write_ci_csv(rows)
    print(f"\nWrote {log_dir / 'per_user_metrics.npz'}, {log_dir / 'bootstrap_ci.json'}, "
          f"and updated {CI_CSV}")


if __name__ == "__main__":
    main()
