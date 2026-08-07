"""M10: experiments/results/summary.csv must be idempotent by run_name --
retrying/resuming a Colab queue entry should never leave two conflicting
rows for the same run (PROJECT_HANDOFF_V2.md §1.2)."""

import csv

import src.utils.common as common


def test_append_result_replaces_same_run_name(tmp_path, monkeypatch):
    csv_path = tmp_path / "summary.csv"
    monkeypatch.setattr(common, "RESULTS_CSV", csv_path)

    common.append_result("run_a", "mf", {"recall@20": 0.10, "ndcg@20": 0.05}, epoch=10)
    common.append_result("run_a", "mf", {"recall@20": 0.15, "ndcg@20": 0.08}, epoch=20)  # retry/resume

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1, "rerunning the same run_name must replace, not duplicate, its row"
    assert rows[0]["epoch"] == "20"
    assert rows[0]["recall@20"] == "0.15"


def test_append_result_keeps_other_runs_and_old_schema_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "summary.csv"
    monkeypatch.setattr(common, "RESULTS_CSV", csv_path)

    # simulate a pre-config_hash-column legacy row written by an older version
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "run_name", "model", "epoch",
                                          "recall@10", "recall@20", "recall@50",
                                          "ndcg@10", "ndcg@20", "ndcg@50", "notes"])
        w.writeheader()
        w.writerow({"timestamp": "2026-01-01 00:00", "run_name": "legacy_run", "model": "mf",
                   "epoch": 5, "recall@20": 0.1, "notes": "pre-migration row"})

    common.append_result("run_b", "lightgcn", {"recall@20": 0.2}, epoch=1,
                         cfg={"seed": 2020, "model": {"name": "lightgcn"}})

    with open(csv_path, newline="") as f:
        rows = {r["run_name"]: r for r in csv.DictReader(f)}

    assert set(rows) == {"legacy_run", "run_b"}
    assert rows["legacy_run"]["config_hash"] == ""  # migrated in, not dropped
    assert rows["run_b"]["config_hash"] != ""


def test_config_hash_is_stable_and_order_independent():
    h1 = common.config_hash({"a": 1, "b": {"c": 2, "d": 3}})
    h2 = common.config_hash({"b": {"d": 3, "c": 2}, "a": 1})
    assert h1 == h2
