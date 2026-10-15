import numpy as np
import pytest
from shiftprofile.cache import ArtifactCache, spec_key
from pathlib import Path


def test_key_is_order_independent():
    a = spec_key({"model": "resnet18", "seed": 0}, producer_version="v1")
    b = spec_key({"seed": 0, "model": "resnet18"}, producer_version="v1")
    assert a == b


def test_key_changes_with_producer_version():
    a = spec_key({"model": "resnet18"}, producer_version="v1")
    b = spec_key({"model": "resnet18"}, producer_version="v2")
    assert a != b


def test_key_changes_with_spec():
    a = spec_key({"seed": 0}, producer_version="v1")
    b = spec_key({"seed": 1}, producer_version="v1")
    assert a != b


def test_put_get_roundtrip_array(tmp_path):
    cache = ArtifactCache(tmp_path)
    spec = {"model": "resnet18", "seed": 0}
    payload = np.arange(12, dtype=np.float16).reshape(3, 4)

    cache.put_array(spec, "v1", payload)
    loaded = cache.get_array(spec, "v1")

    np.testing.assert_array_equal(loaded, payload)
    assert loaded.dtype == np.float16


def test_put_get_roundtrip_record(tmp_path):
    cache = ArtifactCache(tmp_path)
    spec = {"metric": "ece", "cell": "c0"}
    record = {"value": 0.1, "n": 1000}

    cache.put_record(spec, "v1", record)

    assert cache.get_record(spec, "v1") == record


def test_has_is_false_before_put(tmp_path):
    cache = ArtifactCache(tmp_path)
    spec = {"model": "resnet18"}
    assert not cache.has(spec, "v1", kind="array")
    cache.put_array(spec, "v1", np.zeros(3))
    assert cache.has(spec, "v1", kind="array")


def test_no_temp_files_remain(tmp_path):
    cache = ArtifactCache(tmp_path)
    cache.put_array({"a": 1}, "v1", np.zeros(3))
    assert list(tmp_path.rglob("*.tmp")) == []


def test_get_missing_raises(tmp_path):
    cache = ArtifactCache(tmp_path)
    with pytest.raises(KeyError, match="not cached"):
        cache.get_array({"nope": True}, "v1")


def test_put_is_idempotent(tmp_path):
    """Refilling an existing cell must not corrupt it — sessions get preempted
    and re-derive the same work-list on restart."""
    cache = ArtifactCache(tmp_path)
    spec = {"a": 1}
    payload = np.arange(5, dtype=np.float32)
    cache.put_array(spec, "v1", payload)
    cache.put_array(spec, "v1", payload)
    np.testing.assert_array_equal(cache.get_array(spec, "v1"), payload)


def test_key_is_stable_for_numpy_scalars():
    """Cell equality and cache identity must agree. A config built from
    np.arange or a pandas column yields np.int64, which must key identically
    to the Python int it compares equal to."""
    assert spec_key({"seed": np.int64(0)}, "v1") == spec_key({"seed": 0}, "v1")
    assert spec_key({"x": np.float64(0.5)}, "v1") == spec_key({"x": 0.5}, "v1")
    assert spec_key({"ok": np.bool_(True)}, "v1") == spec_key({"ok": True}, "v1")


def test_key_does_not_collide_across_types():
    """np.float32(0.1) and the string "0.1" must not share an artifact."""
    assert spec_key({"x": 0.1}, "v1") != spec_key({"x": "0.1"}, "v1")


def test_key_rejects_uncanonicalizable_values():
    """A key function that improvises on an unknown type is a key function
    that silently forks the cache."""
    class Opaque:
        pass

    with pytest.raises(TypeError, match="canonicaliz"):
        spec_key({"thing": Opaque()}, "v1")


def test_key_rejects_non_finite_floats():
    with pytest.raises(TypeError, match="non-finite"):
        spec_key({"x": float("nan")}, "v1")


def test_key_is_platform_stable_for_paths():
    from pathlib import PurePosixPath, PureWindowsPath
    assert spec_key({"p": PureWindowsPath("data/cifar")}, "v1") == spec_key(
        {"p": PurePosixPath("data/cifar")}, "v1"
    )


def test_crash_midwrite_preserves_previous_value(tmp_path):
    """The module claims a session killed mid-write leaves the cache
    consistent. Nothing tested that claim."""
    cache = ArtifactCache(tmp_path)
    spec, good = {"a": 1}, np.arange(5, dtype=np.float16)
    cache.put_array(spec, "v1", good)

    def boom(fh):
        fh.write(b"\x93NUMPY partial")
        raise KeyboardInterrupt("session preempted")

    with pytest.raises(KeyboardInterrupt):
        cache._atomic_write(cache._path(spec, "v1", ".npy"), boom)

    np.testing.assert_array_equal(cache.get_array(spec, "v1"), good)
    assert list(tmp_path.glob("*.tmp")) == []


def test_crash_on_first_write_creates_no_phantom_entry(tmp_path):
    """A crashed first write must not leave has() returning True for a cell
    that has no readable content."""
    cache = ArtifactCache(tmp_path)
    spec = {"a": 2}

    def boom(fh):
        fh.write(b"partial")
        raise KeyboardInterrupt("session preempted")

    with pytest.raises(KeyboardInterrupt):
        cache._atomic_write(cache._path(spec, "v1", ".npy"), boom)

    assert not cache.has(spec, "v1", kind="array")


def test_concurrent_writers_do_not_share_a_temp_name(tmp_path):
    """Two writers of the same cell must not contend on one temp file."""
    import tempfile
    cache = ArtifactCache(tmp_path)
    spec = {"a": 3}
    path = cache._path(spec, "v1", ".npy")

    # Patch mkstemp to record the temp names it generates
    original_mkstemp = tempfile.mkstemp
    temp_names = []

    def capture_mkstemp(*args, **kwargs):
        fd, tmp_name = original_mkstemp(*args, **kwargs)
        temp_names.append(tmp_name)
        return fd, tmp_name

    tempfile.mkstemp = capture_mkstemp
    try:
        def write_fn(fh):
            np.save(fh, np.zeros(3))

        cache._atomic_write(path, write_fn)
        cache._atomic_write(path, write_fn)

        assert len(temp_names) == 2, f"Expected 2 mkstemp calls, got {len(temp_names)}"
        assert temp_names[0] != temp_names[1], "temp names must be unique per writer"
    finally:
        tempfile.mkstemp = original_mkstemp


def test_has_distinguishes_artifact_kind(tmp_path):
    """A half-filled cell must not read as complete. Preemption between
    put_array and put_record is an ordinary event, not an exotic one."""
    cache = ArtifactCache(tmp_path)
    spec = {"cell": "c0"}

    cache.put_array(spec, "v1", np.zeros(3))

    assert cache.has(spec, "v1", kind="array")
    assert not cache.has(spec, "v1", kind="record")


def test_has_rejects_unknown_kind(tmp_path):
    cache = ArtifactCache(tmp_path)
    with pytest.raises(ValueError, match="kind"):
        cache.has({"a": 1}, "v1", kind="sideways")


def test_put_array_rejects_non_ndarray(tmp_path):
    cache = ArtifactCache(tmp_path)
    with pytest.raises(TypeError, match="ndarray"):
        cache.put_array({"a": 1}, "v1", [0.5, 0.25])


def test_put_array_rejects_object_dtype(tmp_path):
    cache = ArtifactCache(tmp_path)
    obj = np.array([{"a": 1}, {"b": 2}], dtype=object)
    with pytest.raises(TypeError, match="object arrays"):
        cache.put_array({"a": 1}, "v1", obj)


def test_reads_fall_back_to_a_read_only_root(tmp_path):
    """The Kaggle case: cache dataset mounted read-only, writes go elsewhere."""
    mounted = tmp_path / "input" / "cache"
    mounted.mkdir(parents=True)
    seed_cache = ArtifactCache(mounted)
    seed_cache.put_array({"a": 1}, "v1", np.arange(3, dtype=np.float16))

    working = tmp_path / "working" / "cache"
    cache = ArtifactCache(working, read_roots=[mounted])

    assert cache.has({"a": 1}, "v1", kind="array")
    np.testing.assert_array_equal(
        cache.get_array({"a": 1}, "v1"), np.arange(3, dtype=np.float16)
    )


def test_writes_never_touch_a_read_root(tmp_path):
    mounted = tmp_path / "input" / "cache"
    mounted.mkdir(parents=True)
    working = tmp_path / "working" / "cache"
    cache = ArtifactCache(working, read_roots=[mounted])

    cache.put_array({"b": 2}, "v1", np.zeros(3))

    assert list(mounted.glob("*.npy")) == []
    assert len(list(working.glob("*.npy"))) == 1


def test_write_root_shadows_read_root(tmp_path):
    """A freshly computed artifact must win over the stale mounted copy."""
    mounted = tmp_path / "input" / "cache"
    mounted.mkdir(parents=True)
    ArtifactCache(mounted).put_array({"c": 3}, "v1", np.array([1.0, 1.0]))

    working = tmp_path / "working" / "cache"
    cache = ArtifactCache(working, read_roots=[mounted])
    cache.put_array({"c": 3}, "v1", np.array([9.0, 9.0]))

    np.testing.assert_array_equal(cache.get_array({"c": 3}, "v1"), np.array([9.0, 9.0]))


def test_missing_read_root_is_not_an_error(tmp_path):
    """On the first ever run no cache dataset has been published yet."""
    cache = ArtifactCache(tmp_path / "working", read_roots=[tmp_path / "does_not_exist"])
    assert not cache.has({"d": 4}, "v1", kind="array")
    cache.put_array({"d": 4}, "v1", np.zeros(2))
    assert cache.has({"d": 4}, "v1", kind="array")


def test_read_roots_accepts_a_single_path(tmp_path):
    mounted = tmp_path / "input"
    mounted.mkdir()
    ArtifactCache(mounted).put_record({"e": 5}, "v1", {"ok": True})
    cache = ArtifactCache(tmp_path / "working", read_roots=mounted)
    assert cache.get_record({"e": 5}, "v1") == {"ok": True}


def test_resolve_reports_which_root_supplied_the_artifact(tmp_path):
    mounted = tmp_path / "input"
    mounted.mkdir()
    ArtifactCache(mounted).put_array({"f": 6}, "v1", np.zeros(2))
    working = tmp_path / "working"
    cache = ArtifactCache(working, read_roots=[mounted])

    assert cache.resolve({"f": 6}, "v1", kind="array").parent == mounted
    assert cache.resolve({"nope": 0}, "v1", kind="array") is None

    cache.put_array({"f": 6}, "v1", np.ones(2))
    assert cache.resolve({"f": 6}, "v1", kind="array").parent == working


def test_single_root_behaviour_is_unchanged(tmp_path):
    cache = ArtifactCache(tmp_path)
    cache.put_array({"g": 7}, "v1", np.zeros(3))
    assert cache.has({"g": 7}, "v1", kind="array")
    assert cache.resolve({"g": 7}, "v1", kind="array").parent == tmp_path
