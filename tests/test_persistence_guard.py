"""M10: automated coverage for the Colab ephemeral-storage guard
(src/utils/common.py::check_persistent_storage) -- this was reimplemented in
the P0 reconciliation after being found missing from public `main`; this
test is what keeps it from silently regressing again."""

import sys
from pathlib import Path

import pytest

from src.utils.common import ROOT, check_persistent_storage


@pytest.fixture(autouse=True)
def _clean_colab_module():
    yield
    sys.modules.pop("google.colab", None)


def test_noop_outside_colab():
    # sys.modules has no "google.colab" entry in this test environment --
    # must not raise, regardless of allow_ephemeral.
    check_persistent_storage(ROOT / "experiments" / "logs" / "some_run", allow_ephemeral=False)


def test_raises_on_colab_without_drive_mount():
    sys.modules["google.colab"] = object()
    with pytest.raises(SystemExit):
        check_persistent_storage(ROOT / "experiments" / "logs" / "some_run", allow_ephemeral=False)


def test_allow_ephemeral_overrides_the_check():
    sys.modules["google.colab"] = object()
    check_persistent_storage(ROOT / "experiments" / "logs" / "some_run", allow_ephemeral=True)


def test_drive_backed_path_is_accepted(monkeypatch, tmp_path):
    sys.modules["google.colab"] = object()
    fake_drive = tmp_path / "content" / "drive"
    log_dir = fake_drive / "MyDrive" / "sgr_experiments" / "logs" / "some_run"
    log_dir.mkdir(parents=True)

    monkeypatch.setattr("src.utils.common.Path",
                        lambda p="/content/drive": fake_drive if p == "/content/drive" else Path(p))
    check_persistent_storage(log_dir, allow_ephemeral=False)
