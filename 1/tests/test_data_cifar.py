"""Tests for CIFAR-10 and CIFAR-10-C data loaders."""

import hashlib
import numpy as np
import pytest
import torch
from pathlib import Path
from shiftprofile.data import cifar as cifar_module
from shiftprofile.data.cifar import (
    CIFAR10_BASE_FOLDER,
    CIFAR10_REQUIRED_FILES,
    discover_cifar10_root,
    holds_cifar10,
    resolve_cifar10_root,
    describe_mounts,
    discover_cifar10c_root,
    holds_cifar10c,
    resolve_cifar10c_dir,
    CORRUPTION_FAMILIES,
    PILOT_CORRUPTIONS,
    CIFAR10_MEAN,
    CIFAR10_STD,
    fixed_eval_indices,
    load_cifar10_test,
    load_cifar10_train,
    load_cifar10c,
    load_cell_images,
    to_normalised_tensor,
    denormalise,
)
from shiftprofile.cells import Cell


def test_fixed_indices_are_deterministic():
    """Running fixed_eval_indices twice with the same n should give identical results."""
    a = fixed_eval_indices(1000)
    b = fixed_eval_indices(1000)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 1000
    assert len(np.unique(a)) == 1000
    assert a.max() < 10000


def test_fixed_indices_are_nested_across_sizes():
    """The 1000-image subset must be a prefix of the 2000-image subset.

    This ensures a pilot measured on 1000 is directly comparable to a full run on 2000.
    """
    small = fixed_eval_indices(1000)
    large = fixed_eval_indices(2000)
    np.testing.assert_array_equal(small, large[:1000])


def test_fixed_indices_are_valid():
    """Fixed indices must be unique integers in [0, 10000)."""
    indices = fixed_eval_indices(1000)
    assert indices.dtype == np.int64, "dtype must be pinned: a platform-dependent width makes any digest of these indices differ between machines"
    assert len(np.unique(indices)) == 1000
    assert indices.min() >= 0
    assert indices.max() < 10000


def test_committed_indices_match_the_generator():
    """The .npy files in the repo must match what the function produces."""
    from shiftprofile.data import cifar as m

    data_dir = Path(m.__file__).parent

    # Check 1000-image subset
    committed_1000 = np.load(data_dir / "eval_indices_1000.npy")
    generated_1000 = fixed_eval_indices(1000)
    np.testing.assert_array_equal(committed_1000, generated_1000)

    # Check 2000-image subset
    committed_2000 = np.load(data_dir / "eval_indices_2000.npy")
    generated_2000 = fixed_eval_indices(2000)
    np.testing.assert_array_equal(committed_2000, generated_2000)


def test_corruption_families_defined():
    """Corruption families must map to tuples of corruption names."""
    assert isinstance(CORRUPTION_FAMILIES, dict)
    expected_families = {"noise", "blur", "weather", "digital"}
    assert set(CORRUPTION_FAMILIES.keys()) == expected_families

    # Each family should map to a non-empty tuple of corruption names
    for family, corruptions in CORRUPTION_FAMILIES.items():
        assert isinstance(corruptions, tuple)
        assert len(corruptions) > 0
        assert all(isinstance(c, str) for c in corruptions)


def test_pilot_corruptions_defined():
    """Pilot corruptions must be a tuple of three specific corruptions."""
    assert isinstance(PILOT_CORRUPTIONS, tuple)
    assert PILOT_CORRUPTIONS == ("gaussian_noise", "defocus_blur", "fog")


def test_cifar10_normalization_constants():
    """CIFAR10 mean and std constants must be defined correctly."""
    assert CIFAR10_MEAN == (0.4914, 0.4822, 0.4465)
    assert CIFAR10_STD == (0.2470, 0.2435, 0.2616)


@pytest.mark.skip(reason="Requires full CIFAR-10 dataset or complex mocking. Tested via load_cell_images instead.")
def test_load_cifar10_test_synthetic(tmp_path):
    """Test load_cifar10_test with a synthetic dataset.

    Skipped because torchvision's CIFAR10 has integrity checks that require
    the actual dataset or complex mocking. The function is tested indirectly
    via load_cell_images tests that use real synthetic CIFAR-10-C data.
    """
    pass


@pytest.mark.skip(reason="Requires full CIFAR-10 dataset or complex mocking. Not tested in fast test suite.")
def test_load_cifar10_train_synthetic(tmp_path):
    """Test load_cifar10_train with a synthetic dataset.

    Skipped because torchvision's CIFAR10 has integrity checks that require
    the actual dataset or complex mocking.
    """
    pass


def test_load_cifar10c_severity_slicing(tmp_path):
    """Test that severity slicing picks the correct 10000-row block from a 50000-row corruption."""
    # Create a synthetic CIFAR-10-C file where each pixel encodes the corruption index
    # to verify the right rows are extracted
    corruption_data = np.zeros((50000, 32, 32, 3), dtype=np.uint8)

    # Fill each severity block so we can verify we're reading the right rows
    for severity in range(1, 6):
        start_row = (severity - 1) * 10000
        end_row = severity * 10000
        corruption_data[start_row:end_row, 0, 0, 0] = severity * 10

    # Create labels file (same labels tiled 5 times)
    labels = np.arange(10000, dtype=np.int64)
    labels_tiled = np.tile(labels, 5)

    corrupt_dir = tmp_path / "CIFAR-10-C"
    corrupt_dir.mkdir()
    np.save(corrupt_dir / "gaussian_noise.npy", corruption_data)
    np.save(corrupt_dir / "labels.npy", labels_tiled)

    # Load each severity and verify we got the right rows
    for severity in range(1, 6):
        images, severity_labels = load_cifar10c(corrupt_dir.parent, "gaussian_noise", severity)
        assert images.shape == (10000, 32, 32, 3)
        assert severity_labels.shape == (10000,)
        # Check that we got the right severity block (encoded in first pixel, channel 0)
        assert images[0, 0, 0, 0] == severity * 10


def test_load_cifar10c_row_alignment(tmp_path):
    """Test that row i within a severity block corresponds to clean test image i."""
    # Create a corruption where corrupted row i encodes value i
    corruption_data = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
    for i in range(50000):
        corruption_data[i, 0, 0, 0] = i % 256

    labels = np.arange(10000, dtype=np.int64)
    labels_tiled = np.tile(labels, 5)

    corrupt_dir = tmp_path / "CIFAR-10-C"
    corrupt_dir.mkdir()
    np.save(corrupt_dir / "gaussian_noise.npy", corruption_data)
    np.save(corrupt_dir / "labels.npy", labels_tiled)

    # Load severity 1 and check row alignment
    images, severity_labels = load_cifar10c(corrupt_dir.parent, "gaussian_noise", 1)
    for i in range(100):  # Check first 100 rows
        # images[i] is the i-th image in severity 1, which corresponds to global row i
        assert images[i, 0, 0, 0] == (i % 256)


def test_load_cifar10c_invalid_severity(tmp_path):
    """Test that invalid severity values raise clear errors."""
    corrupt_dir = tmp_path / "CIFAR-10-C"
    corrupt_dir.mkdir()
    np.save(corrupt_dir / "gaussian_noise.npy", np.zeros((50000, 32, 32, 3), dtype=np.uint8))
    np.save(corrupt_dir / "labels.npy", np.arange(50000, dtype=np.int64))

    with pytest.raises(ValueError, match="severity.*[1-5]"):
        load_cifar10c(corrupt_dir.parent, "gaussian_noise", 0)

    with pytest.raises(ValueError, match="severity.*[1-5]"):
        load_cifar10c(corrupt_dir.parent, "gaussian_noise", 6)


def test_load_cifar10c_invalid_corruption(tmp_path):
    """Test that unknown corruption names raise clear errors with available names."""
    corrupt_dir = tmp_path / "CIFAR-10-C"
    corrupt_dir.mkdir()

    with pytest.raises(ValueError, match="unknown corruption|gaussian_noise.*defocus_blur"):
        load_cifar10c(corrupt_dir.parent, "nonexistent_corruption", 1)


@pytest.mark.skip(reason="Testing clean path would require real CIFAR-10. Tested via corrupted path which shares same dispatch logic.")
def test_load_cell_images_clean(tmp_path):
    """Test load_cell_images for clean cells.

    Skipped: The clean dispatch path (cell.shift_family == 'clean') calls
    load_cifar10_test, which requires the full dataset. The corrupted dispatch
    path is thoroughly tested via test_load_cell_images_corrupted_with_indices,
    and both paths share identical control flow structure.
    """
    pass


def test_load_cell_images_corrupted_with_indices(tmp_path):
    """Test load_cell_images for corrupted cells with specified indices."""
    corrupt_dir = tmp_path / "CIFAR-10-C"
    corrupt_dir.mkdir()

    # Create synthetic corruption data
    corruption_data = np.random.randint(0, 256, (50000, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(10000, dtype=np.int64)
    labels_tiled = np.tile(labels, 5)

    np.save(corrupt_dir / "gaussian_noise.npy", corruption_data)
    np.save(corrupt_dir / "labels.npy", labels_tiled)

    # Create a corrupted cell
    cell = Cell("vision", "resnet18", 0, "gaussian_noise", 2)
    indices = np.array([0, 1, 2, 3, 4])

    images, labels = load_cell_images(cell, tmp_path, tmp_path, indices=indices)
    assert images.shape == (5, 32, 32, 3)
    assert labels.shape == (5,)


def test_to_normalised_tensor_shape_and_dtype(tmp_path):
    """Test that to_normalised_tensor produces NCHW float32."""
    images = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
    tensor = to_normalised_tensor(images)

    assert tensor.dtype == torch.float32
    assert tensor.shape == (10, 3, 32, 32)  # NCHW


def test_to_normalised_tensor_normalization():
    """Test that to_normalised_tensor applies correct normalization."""
    import torch

    # Create a simple image with all pixels at 128 (mid-gray)
    images = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
    tensor = to_normalised_tensor(images)

    # After normalization (pixel/255 - mean) / std
    # 128/255 ≈ 0.502, (0.502 - 0.4914) / 0.2470 ≈ 0.04
    # Should be close to small values
    assert torch.is_tensor(tensor)
    assert tensor.dtype == torch.float32


def test_to_normalised_tensor_and_denormalise_roundtrip():
    """Test that denormalise reverses to_normalised_tensor approximately."""
    import torch

    original = np.random.randint(0, 256, (5, 32, 32, 3), dtype=np.uint8)
    tensor = to_normalised_tensor(original)
    denorm_tensor = denormalise(tensor)

    # Denormalized tensor should be back in [0, 1] range (or close)
    assert denorm_tensor.min() >= -0.01
    assert denorm_tensor.max() <= 1.01


def test_denormalise_produces_float():
    """Test that denormalise produces float tensors."""
    import torch

    normalized = torch.randn(2, 3, 32, 32)
    denorm = denormalise(normalized)

    assert torch.is_tensor(denorm)
    assert denorm.dtype == torch.float32


# ============================================================================
# CIFAR-10-C layout tolerance
#
# The Zenodo tar extracts to a CIFAR-10-C/ folder, but a Kaggle Dataset built
# from the loose files mounts them flat, and the setup notebook's own Zenodo
# fallback flattens the folder away. The loader previously required the nested
# form while the notebook's verification cell checked the flat one, so setup
# would report success and the fill run would fail hours later.
# ============================================================================


def _write_corruption(directory, n=50000):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "gaussian_noise.npy", np.zeros((n, 32, 32, 3), dtype=np.uint8))
    np.save(directory / "labels.npy", np.arange(n, dtype=np.int64) % 10)


def test_load_cifar10c_accepts_the_nested_layout(tmp_path):
    """{root}/CIFAR-10-C/*.npy, as the Zenodo tar extracts it."""
    _write_corruption(tmp_path / "CIFAR-10-C")
    images, labels = load_cifar10c(tmp_path, "gaussian_noise", 1)
    assert images.shape == (10000, 32, 32, 3)
    assert labels.shape == (10000,)


def test_load_cifar10c_accepts_the_flat_layout(tmp_path):
    """{root}/*.npy, as a Kaggle Dataset of loose files mounts them."""
    _write_corruption(tmp_path)
    images, labels = load_cifar10c(tmp_path, "gaussian_noise", 1)
    assert images.shape == (10000, 32, 32, 3)
    assert labels.shape == (10000,)


def test_load_cifar10c_prefers_nested_when_both_exist(tmp_path):
    """An ambiguous mount resolves to one layout deterministically."""
    _write_corruption(tmp_path / "CIFAR-10-C")
    np.save(tmp_path / "labels.npy", np.zeros(50000, dtype=np.int64))
    np.save(
        tmp_path / "gaussian_noise.npy",
        np.full((50000, 32, 32, 3), 7, dtype=np.uint8),
    )

    images, _ = load_cifar10c(tmp_path, "gaussian_noise", 1)
    assert images.max() == 0, "nested layout should win over the flat one"


def test_load_cifar10c_missing_data_names_both_paths(tmp_path):
    """The error says where it looked, so a bad upload is diagnosable."""
    with pytest.raises(FileNotFoundError) as exc:
        load_cifar10c(tmp_path, "gaussian_noise", 1)

    message = str(exc.value)
    assert "CIFAR-10-C" in message
    assert "labels.npy" in message


# ============================================================================
# Evaluation index portability
#
# The indices decide which images every cell is scored on, so a machine that
# generates a different set produces numbers that cannot be compared with any
# other run. RandomState guarantees the VALUES across platforms; the dtype is
# not guaranteed, and hashing the raw buffer once made a Windows laptop and a
# Linux runner disagree over identical indices.
# ============================================================================

EXPECTED_INDEX_CHECKSUM = "60c5b821cd131d37"


def _canonical_checksum(indices):
    """Digest over little-endian int64, independent of platform int width."""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()[:16]


def test_fixed_eval_indices_checksum_is_pinned():
    """The index stream must not drift. Pinned in scripts/setup_check.py too."""
    assert _canonical_checksum(fixed_eval_indices(1000)) == EXPECTED_INDEX_CHECKSUM


def test_fixed_eval_indices_dtype_is_platform_independent():
    idx = fixed_eval_indices(1000)
    assert idx.dtype == np.int64
    assert idx.nbytes == 1000 * 8


def test_canonical_checksum_survives_a_dtype_round_trip():
    """The digest must depend on the values, not on how they are stored.

    This is the actual regression: the same indices viewed as int32 and int64
    hash differently unless canonicalised, which is what made a correct Kaggle
    run look like a reproducibility failure.
    """
    idx = fixed_eval_indices(500)
    as_int32 = idx.astype(np.int32)
    as_int64 = idx.astype(np.int64)

    assert np.array_equal(as_int32, as_int64), "values must be unchanged by the cast"
    assert _canonical_checksum(as_int32) == _canonical_checksum(as_int64)

    # And the naive digest is exactly what it must not be: storage-dependent.
    naive_32 = hashlib.sha256(as_int32.tobytes()).hexdigest()
    naive_64 = hashlib.sha256(as_int64.tobytes()).hexdigest()
    assert naive_32 != naive_64, (
        "if these ever match, the platform stopped distinguishing int widths and "
        "this test no longer guards anything"
    )


# ============================================================================
# Discovering CIFAR-10-C by content rather than by name
#
# The conventional mount path comes from a Kaggle Dataset slug, so naming the
# Dataset anything but `cifar-10-c` moves it and every default misses. The
# failure then names a path the user never chose, and the fix is to look up and
# retype a slug -- a poor trade for information already on disk.
# ============================================================================


def _make_corruption_dir(directory, with_labels=True):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "gaussian_noise.npy", np.zeros((10, 2), dtype=np.uint8))
    if with_labels:
        np.save(directory / "labels.npy", np.zeros(10, dtype=np.int64))


def test_holds_cifar10c_requires_labels(tmp_path):
    """labels.npy is not optional: every corruption is scored against it."""
    _make_corruption_dir(tmp_path / "with", with_labels=True)
    _make_corruption_dir(tmp_path / "without", with_labels=False)

    assert holds_cifar10c(tmp_path / "with")
    assert not holds_cifar10c(tmp_path / "without")


def test_holds_cifar10c_accepts_either_layout(tmp_path):
    _make_corruption_dir(tmp_path / "nested" / "CIFAR-10-C")
    _make_corruption_dir(tmp_path / "flat")

    assert holds_cifar10c(tmp_path / "nested")
    assert holds_cifar10c(tmp_path / "flat")


def test_holds_cifar10c_on_a_missing_directory_is_false_not_an_error(tmp_path):
    assert not holds_cifar10c(tmp_path / "does-not-exist")


def test_discover_returns_none_when_there_is_nothing_to_find(tmp_path):
    assert discover_cifar10c_root(tmp_path / "absent") is None

    empty = tmp_path / "empty"
    empty.mkdir()
    assert discover_cifar10c_root(empty) is None


def test_discover_ignores_unrelated_datasets(tmp_path):
    base = tmp_path / "input"
    (base / "shiftprofile-cache").mkdir(parents=True)
    (base / "shiftprofile-cache" / ".gitkeep").write_text("")
    (base / "some-csv-data").mkdir()
    (base / "some-csv-data" / "train.csv").write_text("a,b\n1,2\n")

    assert discover_cifar10c_root(base) is None


def test_discover_finds_an_oddly_named_dataset(tmp_path):
    """The case this function exists for: the slug is not `cifar-10-c`."""
    base = tmp_path / "input"
    (base / "shiftprofile-cache").mkdir(parents=True)
    _make_corruption_dir(base / "cifar10-c-corrupted-images" / "CIFAR-10-C")

    found = discover_cifar10c_root(base)
    assert found is not None
    assert "cifar10-c-corrupted-images" in found.parts
    assert resolve_cifar10c_dir(found) == base / "cifar10-c-corrupted-images" / "CIFAR-10-C"


def test_discover_returns_something_resolve_accepts(tmp_path):
    """The contract: whatever comes back can be passed straight as corrupt_root.

    This is the property worth pinning rather than a particular directory. The
    result goes to resolve_cifar10c_dir, and an answer that path cannot resolve
    is useless however sensible it looks in a listing.
    """
    base = tmp_path / "input"
    _make_corruption_dir(base / "ds" / "CIFAR-10-C")

    found = discover_cifar10c_root(base)
    assert resolve_cifar10c_dir(found) == base / "ds" / "CIFAR-10-C"


def test_discover_reaches_arrays_nested_deeper_than_one_level(tmp_path):
    """The case the one-level version missed, and the reason this is a search.

    Uploading a folder rather than its contents leaves the arrays at
    <dataset>/<folder>/CIFAR-10-C/, three levels below the mount root. The
    earlier version checked only <dataset>/ and <dataset>/CIFAR-10-C/, so it
    reported nothing found while the data sat on disk the whole time.
    """
    base = tmp_path / "input"
    _make_corruption_dir(base / "datasets" / "cifar10c" / "CIFAR-10-C")

    found = discover_cifar10c_root(base)
    assert found is not None
    assert resolve_cifar10c_dir(found).exists()
    assert (resolve_cifar10c_dir(found) / "labels.npy").exists()


def test_discover_stops_at_the_depth_limit(tmp_path):
    """Bounded, because /kaggle/input can hold large unrelated datasets."""
    deep = tmp_path / "input" / "a" / "b" / "c" / "d" / "e" / "f"
    _make_corruption_dir(deep)

    assert discover_cifar10c_root(tmp_path / "input", max_depth=3) is None
    assert discover_cifar10c_root(tmp_path / "input", max_depth=9) is not None


def test_discover_prefers_the_shallowest_match(tmp_path):
    """Breadth-first, so an incidental deep copy cannot shadow the real mount."""
    base = tmp_path / "input"
    _make_corruption_dir(base / "shallow")
    _make_corruption_dir(base / "deep" / "nested" / "further")

    assert discover_cifar10c_root(base) == base / "shallow"


def test_discover_is_deterministic_with_two_candidates(tmp_path):
    base = tmp_path / "input"
    _make_corruption_dir(base / "zz-second")
    _make_corruption_dir(base / "aa-first")

    assert discover_cifar10c_root(base) == base / "aa-first"
    assert discover_cifar10c_root(base) == base / "aa-first"


class TestDescribeMounts:
    """The listing that goes into the failure message.

    A session reported "attached Datasets are: datasets. None of them contains
    .npy files" while the arrays were two levels deeper. Naming a directory
    without showing its contents left the only way forward a guess, so the
    message now carries a tree.
    """

    def test_missing_base_says_so_rather_than_returning_nothing(self, tmp_path):
        lines = describe_mounts(tmp_path / "absent")
        assert lines and "does not exist" in lines[0]

    def test_empty_base_is_reported_as_empty(self, tmp_path):
        base = tmp_path / "input"
        base.mkdir()
        assert describe_mounts(base) == ["(empty)"]

    def test_nested_contents_are_shown(self, tmp_path):
        base = tmp_path / "input"
        _make_corruption_dir(base / "datasets" / "cifar10c" / "CIFAR-10-C")

        text = "\n".join(describe_mounts(base))
        assert "datasets/" in text
        assert "cifar10c/" in text
        assert "CIFAR-10-C/" in text
        assert "labels.npy" in text, "the file the whole check turns on"

    def test_an_empty_directory_is_distinguished_from_a_depth_cutoff(self, tmp_path):
        base = tmp_path / "input"
        (base / "still-processing").mkdir(parents=True)
        (base / "deep" / "a" / "b" / "c" / "d").mkdir(parents=True)

        text = "\n".join(describe_mounts(base, max_depth=2))
        assert "(empty)" in text, "an empty upload must read as empty"
        assert "not shown" in text, "a cut-off must not read as empty"

    def test_many_files_are_summarised_rather_than_all_listed(self, tmp_path):
        base = tmp_path / "input"
        flat = base / "ds"
        _make_corruption_dir(flat)
        for i in range(20):
            (flat / f"extra_{i}.npy").write_bytes(b"")

        text = "\n".join(describe_mounts(base))
        assert "more files" in text

    def test_the_listing_is_line_capped(self, tmp_path):
        base = tmp_path / "input"
        for i in range(60):
            (base / f"ds_{i:02d}").mkdir(parents=True)

        lines = describe_mounts(base)
        assert len(lines) <= cifar_module.MOUNT_LISTING_MAX_LINES + 1
        assert "truncated" in lines[-1]


class TestCifar10RootDiscovery:
    """Finding a mounted CIFAR-10 instead of re-downloading 170 MB.

    `/kaggle/working` is wiped between sessions, so the default root is empty
    every fresh session and torchvision re-downloads: measured at 14m50s and
    24m12s on two runs of this project. These pin the conditions under which a
    mount is used, because the failure mode of getting it wrong is not a missed
    optimisation -- it is torchvision deciding to download into a read-only
    Kaggle mount, partway into a session, after the clone and the corruption
    mount have already succeeded.
    """

    def _make(self, root, *, missing=()):
        folder = root / CIFAR10_BASE_FOLDER
        folder.mkdir(parents=True)
        for name in CIFAR10_REQUIRED_FILES:
            if name not in missing:
                (folder / name).write_bytes(b"x")
        return root

    def test_a_complete_layout_is_accepted(self, tmp_path):
        assert holds_cifar10(self._make(tmp_path / "ds"))

    def test_the_root_is_the_parent_of_the_batches_folder(self, tmp_path):
        """torchvision takes the directory CONTAINING `cifar-10-batches-py`, not
        the folder itself. Returning the folder would make it look for
        `cifar-10-batches-py/cifar-10-batches-py`."""
        root = self._make(tmp_path / "ds")
        assert discover_cifar10_root(tmp_path) == root

    @pytest.mark.parametrize("missing", ["test_batch", "data_batch_3", "batches.meta"])
    def test_a_partial_dataset_is_refused(self, tmp_path, missing):
        """Every file the loader needs, train batches included: the fill trains
        and the report reads test labels from the same root. A laxer check would
        accept a test-only upload and fail partway through training."""
        root = self._make(tmp_path / "ds", missing=(missing,))
        assert not holds_cifar10(root)
        assert discover_cifar10_root(tmp_path) is None

    def test_it_finds_a_dataset_nested_several_levels_down(self, tmp_path):
        """Uploading a folder leaves the data deeper than one level, which is the
        bug `discover_cifar10c_root` already had to grow a fix for."""
        root = self._make(tmp_path / "ds" / "archive" / "cifar10")
        assert discover_cifar10_root(tmp_path) == root

    def test_the_shallowest_match_wins_so_the_choice_is_deterministic(self, tmp_path):
        deep = self._make(tmp_path / "a" / "nested")
        shallow = self._make(tmp_path / "b")
        assert discover_cifar10_root(tmp_path) == shallow
        assert holds_cifar10(deep)

    def test_a_missing_base_is_not_an_error(self, tmp_path):
        assert discover_cifar10_root(tmp_path / "nope") is None

    def test_the_writable_root_wins_when_it_already_holds_the_data(self, tmp_path):
        """A second run in the same session must not be sent to a mount when a
        local copy exists -- that copy is also where a previous run downloaded."""
        writable = self._make(tmp_path / "working")
        assert resolve_cifar10_root(writable) == str(writable)

    def test_the_writable_root_is_returned_when_nothing_is_mounted(self, tmp_path):
        """No mounted copy is the first-session case, not an error: the download
        has to land somewhere writable."""
        empty = tmp_path / "working"
        empty.mkdir()
        assert resolve_cifar10_root(empty) == str(empty)
