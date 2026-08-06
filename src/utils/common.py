"""Common utilities: seeding, config loading, result logging."""

import csv
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
    "notes",
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


def append_result(run_name: str, model: str, metrics: dict,
                  epoch: int = -1, notes: str = "") -> None:
    """Append one row to the committed results table (source of truth)."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "run_name": run_name, "model": model, "epoch": epoch, "notes": notes,
    }
    for k, v in metrics.items():
        row[k] = round(float(v), 5)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


class Timer:
    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.msg}] ...", flush=True)
        return self

    def __exit__(self, *args):
        print(f"[{self.msg}] done in {time.time() - self.t0:.1f}s", flush=True)
