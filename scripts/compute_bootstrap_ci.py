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

from src.eval.bootstrap import bootstrap_ci
from src.eval.evaluator import evaluate_per_user
from src.models.registry import load_checkpoint
from src.utils.common import ROOT

CI_CSV = ROOT / "experiments" / "results" / "bootstrap_ci.csv"
CI_FIELDS = ["run_name", "model", "metric", "mean", "ci_low", "ci_high", "n_users", "n_boot"]


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
    cfg, data, model, score_fn = load_checkpoint(args.run, device="cpu")

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
