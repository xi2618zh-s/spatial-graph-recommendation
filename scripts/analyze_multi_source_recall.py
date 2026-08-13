"""V2-0.4: multi-source candidate-recall PREVIEW.

Answers "is multi-source recall worth pursuing in Phase 2" using Track A
checkpoints (LightGCN, Spatial-LightGCN, SASRec) + the two non-learned
baselines (Popularity, ItemCF). See src/data/multi_source.py's module
docstring for the full caveat: this is a preview, not a final result, and
inherits the target-leakage caveat the rest of the v1 pipeline already has.

Usage:
    python scripts/analyze_multi_source_recall.py

Output:
    experiments/results/ranking_v2_multi_source_preview.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.dataset import GowallaData
from src.data.multi_source import (
    itemcf_score_fn, load_sasrec_with_prefix_scoring, pairwise_jaccard_overlap,
    popularity_score_fn, recall_at_k, union_recall_at_k, unique_hit_counts,
)
from src.data.ranking_dataset import build_prefix_targets, generate_candidates, load_timestamped_sequences
from src.models.registry import load_checkpoint
from src.utils.common import ROOT

MAX_K = 200
REPORT_KS = (50, 100, 200)
OVERLAP_SAMPLE_USERS = 1000
OVERLAP_SEED = 2020


def main() -> None:
    seqs = load_timestamped_sequences(ROOT / "data" / "processed" / "train_sequences_ts.pkl")
    prefix_targets = build_prefix_targets(seqs, min_history=5)
    print(f"eligible users: {len(prefix_targets)}")

    official = GowallaData(ROOT / "data" / "gowalla")

    print("scoring: popularity, itemcf (deterministic, no checkpoint)")
    score_fns = {
        "popularity": (popularity_score_fn(official), official),
        "itemcf": (itemcf_score_fn(official), official),
    }
    for run in ("lightgcn_gowalla_repro", "spatial_lightgcn_k10_lam0.3"):
        print(f"loading checkpoint: {run}")
        _, data_r, _, score_fn = load_checkpoint(run, device="cpu")
        score_fns[run] = (score_fn, data_r)

    # SASRec needs its own loader: registry.py's generic score_fn conditions
    # on the model's own full official-train sequence (correct for official
    # test evaluation), NOT on prefix_targets' prefix (target excluded) --
    # see src/data/multi_source.py::sasrec_prefix_score_fn's docstring for
    # the self-attention identity-effect bug this avoids.
    print("loading checkpoint: sasrec_gowalla (prefix-conditioned scoring)")
    sasrec_score_fn = load_sasrec_with_prefix_scoring("sasrec_gowalla", official, prefix_targets)
    score_fns["sasrec_gowalla"] = (sasrec_score_fn, official)

    candidates_by_source = {}
    timing = {}
    for name, (score_fn, _) in score_fns.items():
        t0 = time.time()
        candidates_by_source[name] = generate_candidates(score_fn, prefix_targets, MAX_K)
        timing[name] = round(time.time() - t0, 2)
        print(f"  {name}: candidate generation took {timing[name]}s")

    report = {"sources": list(candidates_by_source), "max_k": MAX_K, "n_users": len(prefix_targets),
              "candidate_gen_seconds": timing, "single_source_recall": {}, "union_recall": {},
              "unique_hit_share": {}}

    for k in REPORT_KS:
        report["single_source_recall"][k] = {
            name: round(recall_at_k(prefix_targets, cands, k), 5)
            for name, cands in candidates_by_source.items()
        }
        report["union_recall"][k] = round(union_recall_at_k(prefix_targets, candidates_by_source, k), 5)
        uh = unique_hit_counts(prefix_targets, candidates_by_source, k)
        report["unique_hit_share"][k] = {
            name: round(count / len(prefix_targets), 5) for name, count in uh.items()
        }
        print(f"K={k}: single-source={report['single_source_recall'][k]} "
              f"union={report['union_recall'][k]} unique_hit_share={report['unique_hit_share'][k]}")

    rng = np.random.default_rng(OVERLAP_SEED)
    sample_users = rng.choice(list(prefix_targets), size=min(OVERLAP_SAMPLE_USERS, len(prefix_targets)),
                              replace=False).tolist()
    overlap = pairwise_jaccard_overlap(candidates_by_source, MAX_K, sample_users)
    report["pairwise_jaccard_overlap_at_max_k"] = {f"{a}|{b}": round(v, 5) for (a, b), v in overlap.items()}
    print(f"pairwise overlap @ K={MAX_K} (n={len(sample_users)} sampled users):")
    for pair, v in report["pairwise_jaccard_overlap_at_max_k"].items():
        print(f"  {pair}: {v}")

    strongest_single = max(report["single_source_recall"][MAX_K].values())
    union_at_max = report["union_recall"][MAX_K]
    report["verdict"] = {
        "strongest_single_source_recall_at_max_k": round(strongest_single, 5),
        "union_recall_at_max_k": round(union_at_max, 5),
        "absolute_gain_over_strongest_single": round(union_at_max - strongest_single, 5),
    }
    print(f"\nVERDICT: union recall@{MAX_K}={union_at_max:.5f} vs strongest single "
          f"source={strongest_single:.5f} (gain={union_at_max - strongest_single:+.5f})")

    out_path = ROOT / "experiments" / "results" / "ranking_v2_multi_source_preview.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
