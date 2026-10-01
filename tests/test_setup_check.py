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


def test_the_message_shows_the_tree_not_just_the_dataset_names(check, tmp_path):
    """The gap a real session fell into.

    It reported "attached Datasets are: datasets. None of them contains .npy
    files" when there was a folder of CSVs inside. Naming a directory without
    showing its contents left the user nothing to act on.
    """
    base = tmp_path / "input"
    inner = base / "datasets" / "something-else"
    inner.mkdir(parents=True)
    (inner / "train.csv").write_text("a,b\n1,2\n")

    message = check._diagnose_missing_mount(base)
    assert "datasets/" in message
    assert "something-else/" in message
    assert "train.csv" in message


def test_dataset_attached_under_a_different_slug_is_suggested(check, tmp_path):
    """The case a bare 'does not exist' cannot distinguish."""
    base = tmp_path / "input"
    (base / "shiftprofile-cache").mkdir(parents=True)
    _corruption_dir(base / "cifar10-c-corrupted" / "CIFAR-10-C")

    message = check._diagnose_missing_mount(base)
    assert "--corrupt-root" in message
    assert "cifar10-c-corrupted" in message


def test_flat_layout_is_also_recognised(check, tmp_path):
    base = tmp_path / "input"
    _corruption_dir(base / "cifar10c-flat")

    message = check._diagnose_missing_mount(base)
    assert "cifar10c-flat" in message
    assert "--corrupt-root" in message


def test_arrays_nested_deep_are_found_and_the_path_given(check, tmp_path):
    """Uploading a folder puts the arrays three levels down, and that is fine."""
    base = tmp_path / "input"
    _corruption_dir(base / "datasets" / "cifar10c" / "CIFAR-10-C")

    message = check._diagnose_missing_mount(base)
    assert "--corrupt-root" in message
    suggested = Path(message.split("--corrupt-root")[1].strip())
    assert (suggested / "labels.npy").exists() or (
        suggested / "CIFAR-10-C" / "labels.npy"
    ).exists(), f"suggested {suggested} is not usable as a corruption root"


def test_arrays_present_but_labels_missing_names_that_specifically(check, tmp_path):
    """A distinguishable mistake, and not the same fix as a missing Dataset.

    Every corruption is scored against labels.npy, so an upload without it is
    unusable -- but reporting "not attached" would send the user looking for a
    Dataset that is right there.
    """
    base = tmp_path / "input"
    arrays = base / "cifar10c" / "CIFAR-10-C"
    arrays.mkdir(parents=True)
    np.save(arrays / "gaussian_noise.npy", np.zeros((10, 2), dtype=np.uint8))

    message = check._diagnose_missing_mount(base)
    assert "labels.npy" in message
    assert "not attached" not in message
    assert str(arrays) in message


def test_expected_checksum_matches_the_library(check):
    """The script's pinned constant must track the actual index stream."""
    import hashlib

    from shiftprofile.data import fixed_eval_indices

    digest = hashlib.sha256(
        np.asarray(fixed_eval_indices(1000), dtype="<i8").tobytes()
    ).hexdigest()[:16]
    assert digest == check.EXPECTED_INDEX_CHECKSUM
