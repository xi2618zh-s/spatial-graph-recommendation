"""M10: interrupted-then-resumed training must match an uninterrupted
reference run within a stated tolerance (PROJECT_HANDOFF_V2.md M10 -- "断点
续训与不中断参考 run 在容差内一致，逐位一致与统计一致分开说明"). Now that
checkpoints persist full RNG state (src/utils/common.py::rng_snapshot),
this test asserts BIT-EXACT agreement, not just "close enough" -- if RNG
persistence ever regresses, this is the test that should catch it.

Runs on a tiny synthetic dataset (not full Gowalla) so it's fast enough for
routine CPU test runs.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import GowallaData
from src.models.mf import MFBPR
from src.train import bpr_trainer as bpr_trainer_module
from src.train.bpr_trainer import train_bpr
from src.utils import common as common_module
from src.utils.common import set_seed


def _write_tiny_split(root):
    gowalla_dir = root / "data" / "gowalla"
    gowalla_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n_users, n_items = 15, 30
    with open(gowalla_dir / "train.txt", "w") as f:
        for u in range(n_users):
            items = sorted(rng.choice(n_items, size=rng.integers(3, 6), replace=False).tolist())
            f.write(f"{u} " + " ".join(map(str, items)) + "\n")
    with open(gowalla_dir / "test.txt", "w") as f:
        for u in range(n_users):
            items = sorted(rng.choice(n_items, size=2, replace=False).tolist())
            f.write(f"{u} " + " ".join(map(str, items)) + "\n")
    return gowalla_dir


def _tiny_cfg(run_name: str, epochs: int, seed: int = 777) -> dict:
    return {
        "experiment_name": run_name, "seed": seed,
        "data": {"split_dir": "unused", "batch_size": 16},
        "model": {"name": "mf", "embedding_dim": 8},
        "train": {"lr": 0.05, "l2_reg": 1e-4, "epochs": epochs, "eval_every": 1,
                 "early_stop_patience": 1000},
        "eval": {"topk": [10, 20], "metrics": ["recall", "ndcg"]},  # bpr_trainer tracks "recall@20" specifically
    }


@pytest.fixture
def tiny_data(tmp_path):
    gowalla_dir = _write_tiny_split(tmp_path)
    return GowallaData(gowalla_dir)


def _run(tmp_path, monkeypatch, tiny_data, run_name, epochs, resume, seed=777):
    monkeypatch.setattr(bpr_trainer_module, "ROOT", tmp_path)
    monkeypatch.setattr(common_module, "RESULTS_CSV", tmp_path / "summary.csv")
    set_seed(seed)
    model = MFBPR(tiny_data.n_users, tiny_data.n_items, dim=8)
    cfg = _tiny_cfg(run_name, epochs, seed)
    return train_bpr(model, tiny_data, cfg, device="cpu", resume=resume), model


def test_resumed_run_matches_uninterrupted_reference_bit_exact(tmp_path, monkeypatch, tiny_data):
    # Reference: 4 epochs, no interruption.
    best_ref, model_ref = _run(tmp_path, monkeypatch, tiny_data, "run_reference", epochs=4, resume=False)

    # Simulate an interruption after epoch 2, then resume to epoch 4.
    _run(tmp_path, monkeypatch, tiny_data, "run_resumed", epochs=2, resume=False)
    done_path = tmp_path / "experiments" / "logs" / "run_resumed" / "DONE"
    assert done_path.exists()
    done_path.unlink()  # a real crash never reaches this write; remove it to simulate that

    best_resumed, model_resumed = _run(
        tmp_path, monkeypatch, tiny_data, "run_resumed", epochs=4, resume=True
    )

    for k in best_ref:
        assert best_resumed[k] == pytest.approx(best_ref[k], abs=1e-9), \
            f"metric {k} diverged after resume: {best_resumed[k]} vs {best_ref[k]}"

    for p_ref, p_resumed in zip(model_ref.parameters(), model_resumed.parameters()):
        assert torch.equal(p_ref, p_resumed), "model weights diverged bit-exactly after resume"


def test_resume_without_saved_rng_warns_and_degrades_gracefully(tmp_path, monkeypatch, tiny_data, capsys):
    """Checkpoints written before RNG persistence existed must still resume
    (statistically, not bit-exactly) rather than crash."""
    monkeypatch.setattr(bpr_trainer_module, "ROOT", tmp_path)
    monkeypatch.setattr(common_module, "RESULTS_CSV", tmp_path / "summary.csv")
    set_seed(777)
    model = MFBPR(tiny_data.n_users, tiny_data.n_items, dim=8)
    train_bpr(model, tiny_data, _tiny_cfg("run_legacy", epochs=2, seed=777), device="cpu", resume=False)

    ckpt_path = tmp_path / "experiments" / "logs" / "run_legacy" / "last.ckpt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del ck["rng"]  # simulate a pre-RNG-persistence checkpoint
    torch.save(ck, ckpt_path)
    (tmp_path / "experiments" / "logs" / "run_legacy" / "DONE").unlink()

    model2 = MFBPR(tiny_data.n_users, tiny_data.n_items, dim=8)
    train_bpr(model2, tiny_data, _tiny_cfg("run_legacy", epochs=4, seed=777), device="cpu", resume=True)
    assert "predates RNG-state persistence" in capsys.readouterr().out
