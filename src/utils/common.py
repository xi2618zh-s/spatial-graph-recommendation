"""Common utilities: seeding, config loading, result logging."""

import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = ROOT / "experiments" / "results" / "summary.csv"
RESULT_FIELDS = [
    "timestamp", "run_name", "model", "epoch",
    "recall@10", "recall@20", "recall@50",
    "ndcg@10", "ndcg@20", "ndcg@50",
    "config_hash", "notes",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_persistent_storage(log_dir: Path, allow_ephemeral: bool = False) -> None:
    """Refuse to start a long training run on Colab if experiments/ is not
    backed by a mounted Drive. Colab's local disk is wiped on disconnect —
    a checkpoint or result row written there is lost with the session.

    No-op outside Colab (local disks are already persistent).
    """
    if "google.colab" not in sys.modules:
        return
    if allow_ephemeral:
        return

    drive_mount = Path("/content/drive")
    resolved = log_dir.resolve()
    backed_by_drive = drive_mount.exists() and str(resolved).startswith(
        str(drive_mount.resolve())
    )
    if not backed_by_drive:
        raise SystemExit(
            "Refusing to train: running on Colab but "
            f"{log_dir} is not under a mounted Google Drive "
            f"({drive_mount}). Checkpoints and results would be lost when "
            "this session disconnects.\n"
            "Fix: mount Drive (from google.colab import drive; "
            "drive.mount('/content/drive')) and symlink experiments/ into "
            "MyDrive/sgr_experiments before training, e.g.:\n"
            "  ln -s /content/drive/MyDrive/sgr_experiments/logs experiments/logs\n"
            "Override (smoke tests only, results WILL be lost): "
            "pass --allow-ephemeral."
        )


def config_hash(cfg: dict) -> str:
    """Stable short hash of a run's config, recorded in summary.csv for
    manual config-drift auditing. NOT part of the dedup key in
    `append_result` below -- this project already gives every distinct
    experiment its own unique `run_name` (one YAML per run), so `run_name`
    alone is the natural identity key; config_hash is provenance, not
    uniqueness."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def append_result(run_name: str, model: str, metrics: dict, epoch: int = -1,
                  notes: str = "", cfg: dict | None = None) -> None:
    """(Re)write one row in the committed results table (source of truth).

    Idempotent by `run_name`: rerunning the same experiment (e.g. a retried
    or resumed queue entry) REPLACES its previous row instead of appending a
    duplicate, so a flaky Colab queue can never leave summary.csv with two
    conflicting rows for one run. Old rows written before the `config_hash`
    column existed are preserved with an empty value there, not dropped.
    """
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "run_name": run_name, "model": model, "epoch": epoch, "notes": notes,
        "config_hash": config_hash(cfg) if cfg is not None else "",
    }
    for k, v in metrics.items():
        row[k] = round(float(v), 5)

    existing = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r.get("run_name") != run_name]
    existing.append(row)

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(existing)


def rng_snapshot(numpy_generator: np.random.Generator | None = None) -> dict:
    """Full RNG state for exact resume, not just statistical continuation.
    `numpy_generator` is the `np.random.default_rng(seed)` instance trainers
    actually sample from -- distinct from (and not covered by) the legacy
    global `np.random` state, so it needs its own key."""
    snap = {"python_random": random.getstate(), "numpy_legacy": np.random.get_state()}
    if numpy_generator is not None:
        snap["numpy_generator_state"] = numpy_generator.bit_generator.state
    try:
        import torch
        snap["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return snap


def rng_restore(snap: dict, numpy_generator: np.random.Generator | None = None) -> None:
    """Inverse of rng_snapshot. Tolerates a partial/missing snapshot (e.g. a
    checkpoint written before this field existed) by restoring whatever keys
    are present and silently skipping the rest -- callers should print their
    own warning when `snap` is empty so a degraded resume isn't silent.

    Every individual restoration step is wrapped so a failure there degrades
    to a printed warning instead of raising: RNG continuity is an
    enhancement (bit-exact resume) on top of a training run that is already
    correct without it (statistically-continued resume), so it must never
    be the reason a resume aborts.

    The torch steps additionally force their tensors onto CPU (and to
    uint8) before calling the setter. `torch.load(ckpt_path,
    map_location=device, ...)` remaps EVERY tensor in a checkpoint to
    `device`, including the embedded RNG-state tensors -- but
    `torch.set_rng_state`/`torch.cuda.set_rng_state_all` both require a CPU
    ByteTensor regardless of what device the generator itself describes. On
    a GPU resume (`map_location="cuda"`) this previously raised `TypeError:
    RNG state must be a torch.ByteTensor` and crashed the whole resume; see
    tests/test_rng_restore.py for the reproduction.
    """
    if "python_random" in snap:
        try:
            random.setstate(snap["python_random"])
        except Exception as e:
            print(f"WARNING: failed to restore python random state ({e!r}); "
                  "continuing without it (resume will be statistically continued, not bit-exact)")
    if "numpy_legacy" in snap:
        try:
            np.random.set_state(snap["numpy_legacy"])
        except Exception as e:
            print(f"WARNING: failed to restore numpy legacy RNG state ({e!r}); continuing without it")
    if numpy_generator is not None and "numpy_generator_state" in snap:
        try:
            numpy_generator.bit_generator.state = snap["numpy_generator_state"]
        except Exception as e:
            print(f"WARNING: failed to restore numpy Generator state ({e!r}); continuing without it")
    try:
        import torch

        def _as_cpu_byte_tensor(state):
            if torch.is_tensor(state):
                return state.detach().to(device="cpu", dtype=torch.uint8)
            return state  # let the setter raise its own error if this truly isn't restorable

        if "torch_cpu" in snap:
            try:
                torch.set_rng_state(_as_cpu_byte_tensor(snap["torch_cpu"]))
            except Exception as e:
                print(f"WARNING: failed to restore torch CPU RNG state ({e!r}); continuing without it")
        if "torch_cuda" in snap and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(s) for s in snap["torch_cuda"]])
            except Exception as e:
                print(f"WARNING: failed to restore torch CUDA RNG state ({e!r}); continuing without it")
    except ImportError:
        pass


class Timer:
    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.msg}] ...", flush=True)
        return self

    def __exit__(self, *args):
        print(f"[{self.msg}] done in {time.time() - self.t0:.1f}s", flush=True)
