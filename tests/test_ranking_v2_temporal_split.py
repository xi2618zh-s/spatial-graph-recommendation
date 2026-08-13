"""V2-0.1: correctness of the H_inner|v0|y_train|y_val split and of
SnapshotGowallaData as a drop-in for GowallaData (proven by actually running
the UNMODIFIED src/train/bpr_trainer.train_bpr against it, not just checking
attributes)."""

import numpy as np

from src.data.ranking_dataset_v2 import (
    SnapshotGowallaData, build_v2_layers, hcore_edges, hcore_plus_ytrain_edges, v0_targets,
)
from src.models.mf import MFBPR
from src.train import bpr_trainer as bpr_trainer_module
from src.train.bpr_trainer import train_bpr
from src.utils import common as common_module
from src.utils.common import set_seed


def test_split_positions_match_the_worked_example():
    # T=10 sequence, matches the worked example confirmed with the user:
    # [i1..i10] -> H_inner=i1..i7, v0=i8, y_train=i9, y_val=i10
    seq = {0: [(10 + i, 1_000_000 + i) for i in range(1, 11)]}  # items 11..20, ts 1000001..1000010
    layers = build_v2_layers(seq, min_h_inner=5)
    L = layers[0]
    assert L["h_inner_items"] == [11, 12, 13, 14, 15, 16, 17]
    assert L["v0_item"] == 18
    assert L["y_train_item"] == 19
    assert L["y_val_item"] == 20
    assert L["h_inner_ts"] == [1_000_001 + i for i in range(7)]
    assert L["v0_ts"] < L["y_train_ts"] < L["y_val_ts"]


def test_four_layers_never_overlap_and_edges_nest_correctly():
    rng = np.random.default_rng(0)
    seq = {}
    for u in range(20):
        length = rng.integers(8, 15)
        items = rng.choice(100, size=length, replace=False).tolist()
        ts = sorted(rng.integers(0, 10_000, size=length).tolist())
        seq[u] = list(zip(items, ts))
    layers = build_v2_layers(seq, min_h_inner=5)
    assert len(layers) == len(seq)  # every synthetic user has >= 8 interactions

    hcore = hcore_edges(layers)
    hcore_plus = hcore_plus_ytrain_edges(layers)
    for u, L in layers.items():
        four_layers = set(L["h_inner_items"]) | {L["v0_item"], L["y_train_item"], L["y_val_item"]}
        assert len(four_layers) == len(L["h_inner_items"]) + 3, "layers must not overlap within a user"
        assert L["y_train_item"] not in hcore[u], "R0's training edges must never include y_train"
        assert L["y_val_item"] not in hcore_plus[u], "R1's training edges must never include y_val"
        assert set(hcore[u]) == set(L["h_inner_items"]) | {L["v0_item"]}
        assert set(hcore_plus[u]) == set(hcore[u]) | {L["y_train_item"]}


def test_users_below_min_h_inner_are_dropped():
    seq = {0: [(i, i) for i in range(7)]}  # length 7 < 5+3=8
    assert build_v2_layers(seq, min_h_inner=5) == {}


def _tiny_v2_sequences(seed=0, n_users=12, n_items=25):
    rng = np.random.default_rng(seed)
    seq = {}
    for u in range(n_users):
        length = rng.integers(9, 14)
        items = rng.choice(n_items, size=length, replace=False).tolist()
        ts = sorted(rng.integers(0, 100_000, size=length).tolist())
        seq[u] = list(zip(items, ts))
    return seq, n_users, n_items


def test_snapshot_data_runs_through_unmodified_train_bpr(tmp_path, monkeypatch):
    """Proves SnapshotGowallaData is interface-compatible with the exact,
    unmodified train_bpr()/evaluate() the currently-running Colab ablation
    queue also imports -- not by inspecting attributes, but by actually
    training on it end to end."""
    monkeypatch.setattr(bpr_trainer_module, "ROOT", tmp_path)
    monkeypatch.setattr(common_module, "RESULTS_CSV", tmp_path / "summary.csv")

    seq, n_users, n_items = _tiny_v2_sequences()
    layers = build_v2_layers(seq, min_h_inner=5)
    assert len(layers) == n_users

    h_inner_edges = {u: L["h_inner_items"] for u, L in layers.items()}
    v0 = v0_targets(layers)

    # step 1: inner-validation run on H_inner, early-stopping against v0
    set_seed(777)
    model = MFBPR(n_users, n_items, dim=8)
    data = SnapshotGowallaData(h_inner_edges, v0, n_users, n_items)
    cfg = {
        "experiment_name": "v2_inner_validation", "seed": 777,
        "data": {"split_dir": "unused", "batch_size": 16},
        "model": {"name": "mf", "embedding_dim": 8},
        "train": {"lr": 0.05, "l2_reg": 1e-4, "epochs": 10, "eval_every": 1,
                 "early_stop_patience": 3},
        "eval": {"topk": [10, 20], "metrics": ["recall", "ndcg"]},
    }
    best = train_bpr(model, data, cfg, device="cpu", resume=False)
    e_star = int((tmp_path / "experiments" / "logs" / "v2_inner_validation" / "DONE")
                .read_text().strip().split("=")[1])
    assert e_star >= 1
    assert "recall@20" in best

    # step 2: R0 refit -- FRESH init, on Hcore (=H_inner+v0), fixed E* epochs,
    # no early stopping decision (eval_every=E* means exactly one save, at
    # epoch E*, regardless of the metric value)
    hcore = hcore_edges(layers)
    set_seed(778)  # a different seed is fine -- this is a fresh, independent init by design
    model_r0 = MFBPR(n_users, n_items, dim=8)
    data_r0 = SnapshotGowallaData(hcore, v0, n_users, n_items)  # v0 reused only as a harmless trigger
    cfg_r0 = dict(cfg, experiment_name="v2_r0",
                 train={"lr": 0.05, "l2_reg": 1e-4, "epochs": e_star,
                        "eval_every": e_star, "early_stop_patience": 10**6})
    train_bpr(model_r0, data_r0, cfg_r0, device="cpu", resume=False)
    r0_done = (tmp_path / "experiments" / "logs" / "v2_r0" / "DONE").read_text().strip()
    assert r0_done == f"best_epoch={e_star}", "R0's fixed-budget refit must land exactly at E*"
