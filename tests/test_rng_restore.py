"""Regression test for a real bug hit on Colab: resuming a GPU training run
crashed with `TypeError: RNG state must be a torch.ByteTensor` inside
src/utils/common.py::rng_restore.

Root cause: `torch.load(ckpt_path, map_location=device, ...)` remaps EVERY
tensor in the checkpoint to `device`, including the RNG-state ByteTensor
saved by rng_snapshot(). On a GPU resume (map_location="cuda"), the
resulting tensor is no longer an instance of the CPU-specific
`torch.ByteTensor` type `torch.set_rng_state`/`torch.cuda.set_rng_state_all`
require -- confirmed locally (no GPU needed to reproduce: a CPU tensor with
the wrong dtype fails the exact same isinstance check and raises the exact
same message, see test_wrong_dtype_reproduces_the_reported_error below).

Fix: rng_restore now force-converts every torch RNG-state tensor to a CPU
uint8 tensor before calling the setter, and wraps every restoration step so
a failure degrades to a printed warning instead of crashing the run -- RNG
continuity is an enhancement (bit-exact resume), not a correctness
requirement, and must never abort training.
"""

import pickle

import numpy as np
import pytest
import torch

from src.data.dataset import GowallaData
from src.models.mf import MFBPR
from src.train import bpr_trainer as bpr_trainer_module
from src.train.bpr_trainer import train_bpr
from src.utils import common as common_module
from src.utils.common import rng_restore, rng_snapshot, set_seed


def test_wrong_dtype_reproduces_the_reported_error():
    """Sanity-check that our reproduction actually matches the bug report,
    not a different error -- without this, the rest of the file could be
    testing the wrong failure mode."""
    with pytest.raises(TypeError, match="RNG state must be a torch.ByteTensor"):
        torch.set_rng_state(torch.zeros(5000, dtype=torch.float32))


def test_round_trip_through_torch_save_load_restores_exact_state(tmp_path):
    # snapshot taken immediately after seeding, before any draws
    set_seed(2020)
    rng = np.random.default_rng(2020)
    snap = rng_snapshot(rng)

    path = tmp_path / "rng.pt"
    torch.save(snap, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    # perturb every RNG source, then restore from the round-tripped snapshot
    torch.manual_seed(999)
    rng.integers(0, 100, size=5)
    rng_restore(loaded, rng)
    actual_numpy = list(rng.integers(0, 1000, size=10))
    actual_torch = torch.rand(10)

    # a completely independent, freshly-seeded reference (no save/load/perturb
    # involved) must draw identically, since the snapshot was taken right
    # after seeding with no draws in between
    set_seed(2020)
    ref_rng = np.random.default_rng(2020)
    expected_numpy = list(ref_rng.integers(0, 1000, size=10))
    expected_torch = torch.rand(10)

    assert actual_numpy == expected_numpy
    assert torch.equal(actual_torch, expected_torch)


def test_restore_degrades_to_warning_instead_of_crashing_on_bad_torch_state(capsys):
    """Directly reproduces the Colab crash: rng_restore must not raise, and
    must still restore the OTHER RNG sources despite the torch one failing."""
    good_snap = rng_snapshot()
    corrupted = dict(good_snap)
    corrupted["torch_cpu"] = torch.zeros(5000, dtype=torch.float32)  # wrong dtype -> reproduces the bug

    rng_restore(corrupted)  # must not raise

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "torch CPU RNG state" in out

    # python_random / numpy_legacy must still have been restored despite the torch failure
    random_state_after = __import__("random").getstate()
    assert random_state_after == good_snap["python_random"]


def test_restore_handles_a_gpu_shaped_torch_cuda_list_gracefully(capsys):
    """No GPU available to reproduce map_location="cuda" exactly, but the
    same coercion path handles a torch_cuda entry containing wrong-dtype
    tensors (standing in for what map_location remapping produces) without
    crashing, whether or not CUDA is actually available on this machine."""
    snap = {"torch_cuda": [torch.zeros(5000, dtype=torch.float32)]}
    rng_restore(snap)  # must not raise regardless of torch.cuda.is_available() here


def test_bpr_trainer_resume_survives_a_corrupted_rng_checkpoint(tmp_path, monkeypatch):
    """End-to-end through the REAL, unmodified train_bpr() resume path (the
    same one the currently-running SASRec/LightGCN/Spatial-LightGCN Colab
    jobs use via scripts/train.py) -- proves bpr_trainer.py needs no changes
    of its own because the fix is centralized in rng_restore()."""
    gowalla_dir = tmp_path / "data" / "gowalla"
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
    data = GowallaData(gowalla_dir)

    monkeypatch.setattr(bpr_trainer_module, "ROOT", tmp_path)
    monkeypatch.setattr(common_module, "RESULTS_CSV", tmp_path / "summary.csv")

    cfg = {
        "experiment_name": "corrupted_rng_resume", "seed": 777,
        "data": {"split_dir": "unused", "batch_size": 16},
        "model": {"name": "mf", "embedding_dim": 8},
        "train": {"lr": 0.05, "l2_reg": 1e-4, "epochs": 2, "eval_every": 1,
                 "early_stop_patience": 1000},
        "eval": {"topk": [10, 20], "metrics": ["recall", "ndcg"]},
    }
    set_seed(777)
    model = MFBPR(data.n_users, data.n_items, dim=8)
    train_bpr(model, data, cfg, device="cpu", resume=False)

    ckpt_path = tmp_path / "experiments" / "logs" / "corrupted_rng_resume" / "last.ckpt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck["rng"]["torch_cpu"] = torch.zeros(5000, dtype=torch.float32)  # simulate the Colab crash
    torch.save(ck, ckpt_path)
    (tmp_path / "experiments" / "logs" / "corrupted_rng_resume" / "DONE").unlink()

    cfg["train"]["epochs"] = 4
    model2 = MFBPR(data.n_users, data.n_items, dim=8)
    best = train_bpr(model2, data, cfg, device="cpu", resume=True)  # must not raise
    assert "recall@20" in best
