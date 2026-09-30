"""Tests for scripts/setup_check.py, the session gate.

The check that keeps failing in practice is the CIFAR-10-C mount, and its error
message is the whole value of the script: the expected path is derived from a
Kaggle Dataset slug, so a Dataset named anything else mounts elsewhere and the
failure names a path the user never chose. "Does not exist" leaves them unable
to tell "I forgot to attach it" from "I named it differently" -- two very
different fixes. These tests pin that the message distinguishes them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "setup_check.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("setup_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corruption_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "labels.npy", np.zeros(10, dtype=np.int64))
    np.save(directory / "gaussian_noise.npy", np.zeros((10, 2), dtype=np.uint8))


def test_no_mount_root_at_all(check, tmp_path):
    """A local machine has no /kaggle/input; that is not a Kaggle problem."""
    message = check._diagnose_missing_mount(tmp_path / "absent")
    assert "--corrupt-root" in message
    assert "nothing is mounted" in message


def test_nothing_attached(check, tmp_path):
    (tmp_path / "input").mkdir()
    message = check._diagnose_missing_mount(tmp_path / "input")
    assert "nothing is attached" in message
    assert "Add Input" in message


def test_datasets_attached_but_none_hold_arrays(check, tmp_path):
    base = tmp_path / "input"
    (base / "shiftprofile-cache").mkdir(parents=True)
    (base / "shiftprofile-cache" / ".gitkeep").write_text("")

    message = check._diagnose_missing_mount(base)
    assert "shiftprofile-cache" in message, "must list what IS attached"
    assert "not attached to this notebook yet" in message


def test_dataset_attached_under_a_different_slug_is_suggested(check, tmp_path):
    """The case a bare 'does not exist' cannot distinguish."""
    base = tmp_path / "input"
    (base / "shiftprofile-cache").mkdir(parents=True)
    _corruption_dir(base / "cifar10-c-corrupted" / "CIFAR-10-C")

    message = check._diagnose_missing_mount(base)
    assert "cifar10-c-corrupted" in message
    assert "--corrupt-root" in message
    assert str(base / "cifar10-c-corrupted") in message


def test_flat_layout_is_also_recognised(check, tmp_path):
    base = tmp_path / "input"
    _corruption_dir(base / "cifar10c-flat")

    message = check._diagnose_missing_mount(base)
    assert "cifar10c-flat" in message
    assert "--corrupt-root" in message


def test_looks_like_cifar10c_rejects_an_unrelated_dataset(check, tmp_path):
    unrelated = tmp_path / "some-csv-dataset"
    unrelated.mkdir()
    (unrelated / "train.csv").write_text("a,b\n1,2\n")

    assert not check._looks_like_cifar10c(unrelated)


def test_looks_like_cifar10c_accepts_both_layouts(check, tmp_path):
    nested = tmp_path / "nested"
    _corruption_dir(nested / "CIFAR-10-C")
    flat = tmp_path / "flat"
    _corruption_dir(flat)

    assert check._looks_like_cifar10c(nested)
    assert check._looks_like_cifar10c(flat)


def test_expected_checksum_matches_the_library(check):
    """The script's pinned constant must track the actual index stream."""
    import hashlib

    from shiftprofile.data import fixed_eval_indices

    digest = hashlib.sha256(
        np.asarray(fixed_eval_indices(1000), dtype="<i8").tobytes()
    ).hexdigest()[:16]
    assert digest == check.EXPECTED_INDEX_CHECKSUM
