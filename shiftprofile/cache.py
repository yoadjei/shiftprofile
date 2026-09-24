"""Content-hashed artifact store with atomic writes.

Keyed on a hash of the full spec dict plus the version of the *producing
function only*. A change to a plotting function must never invalidate a GPU
artifact that cost quota to produce.

Writes go to a temp file and are then renamed, so a session killed mid-write
leaves the cache consistent rather than corrupt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

import numpy as np


def _canonical(value):
    """Normalize a spec value to a JSON-native, platform-stable form.

    Raises on anything unrecognised. A key that quietly accepts a new type is a
    key that quietly forks the cache — and a forked cache costs GPU quota that
    this project does not have.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):          # np.int64, np.float32, np.bool_
        return _canonical(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"non-finite spec value: {value!r}")
        return float(value) + 0.0              # collapse -0.0 to 0.0
    if isinstance(value, Enum):
        return [type(value).__name__, _canonical(value.value)]
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"spec dict has non-string keys: {sorted(map(repr, value))}")
        return {k: _canonical(v) for k, v in value.items()}
    raise TypeError(
        f"spec value of type {type(value).__name__!r} is not canonicalizable: {value!r}"
    )


def spec_key(spec: dict[str, Any], producer_version: str) -> str:
    """Stable content hash of a spec. Key order must not matter, and neither
    must the incidental Python type a config loader happened to produce."""
    payload = json.dumps(
        _canonical({"spec": spec, "producer_version": producer_version}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ArtifactCache:
    """Flat content-addressed store of arrays and JSON records."""

    def __init__(
        self,
        write_root: Path | str,
        read_roots: Path | str | list[Path | str] | tuple[Path | str, ...] = (),
    ) -> None:
        """Initialize cache with a write root and optional read-only roots.

        Args:
            write_root: Directory where artifacts are written. Always created.
            read_roots: One or more read-only directories to search before write_root.
                       May contain paths that do not exist. Accepts a single path or iterable.
        """
        self.root = Path(write_root)
        self.root.mkdir(parents=True, exist_ok=True)

        # Normalize read_roots to a tuple of Paths
        if isinstance(read_roots, (str, Path)):
            self.read_roots = (Path(read_roots),)
        elif isinstance(read_roots, (list, tuple)):
            self.read_roots = tuple(Path(r) for r in read_roots)
        else:
            self.read_roots = ()

    def _path(self, spec: dict[str, Any], producer_version: str, suffix: str) -> Path:
        return self.root / f"{spec_key(spec, producer_version)}{suffix}"

    def has(self, spec: dict[str, Any], producer_version: str, kind: str) -> bool:
        """Whether a specific artifact kind exists for this spec.

        `kind` is required: a cell that has its array but not its record is not
        done, and defaulting would let that distinction be forgotten.
        """
        suffixes = {"array": ".npy", "record": ".json"}
        if kind not in suffixes:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(suffixes)}")
        return self.resolve(spec, producer_version, kind) is not None

    def resolve(
        self, spec: dict[str, Any], producer_version: str, kind: str
    ) -> Path | None:
        """Return the Path where the artifact was found, or None.

        Searches write_root first, then each read_root in order. First hit wins,
        so a freshly written artifact shadows a stale one in a read-only root.
        """
        suffixes = {"array": ".npy", "record": ".json"}
        if kind not in suffixes:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(suffixes)}")

        # Check write_root first
        path = self._path(spec, producer_version, suffixes[kind])
        if path.exists():
            return path

        # Then search read_roots in order
        for read_root in self.read_roots:
            # read_root may not exist; that's fine
            if not read_root.exists():
                continue
            read_path = read_root / path.name
            if read_path.exists():
                return read_path

        return None

    def _atomic_write(self, path: Path, write_fn) -> None:
        """Write via a private temp file, then rename.

        The temp name must be unique per writer: two Kaggle sessions filling the
        same cell would otherwise truncate each other's in-flight bytes and
        publish a torn artifact that has() reports as valid forever.
        """
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                write_fn(fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def put_array(
        self, spec: dict[str, Any], producer_version: str, array: np.ndarray
    ) -> Path:
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"put_array wants an ndarray, got {type(array).__name__}. "
                "A list would silently become float64 and blow the cache budget."
            )
        if array.dtype == object:
            raise TypeError(
                "object arrays cannot be read back (np.load defaults to allow_pickle=False)"
            )
        path = self._path(spec, producer_version, ".npy")
        self._atomic_write(path, lambda fh: np.save(fh, array))
        self._write_sidecar(spec, producer_version)
        return path

    def get_array(self, spec: dict[str, Any], producer_version: str) -> np.ndarray:
        path = self.resolve(spec, producer_version, kind="array")
        if path is None:
            raise KeyError(f"not cached: {spec} @ {producer_version}")
        return np.load(path)

    def put_record(
        self, spec: dict[str, Any], producer_version: str, record: dict[str, Any]
    ) -> Path:
        path = self._path(spec, producer_version, ".json")
        blob = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        self._atomic_write(path, lambda fh: fh.write(blob))
        self._write_sidecar(spec, producer_version)
        return path

    def get_record(
        self, spec: dict[str, Any], producer_version: str
    ) -> dict[str, Any]:
        path = self.resolve(spec, producer_version, kind="record")
        if path is None:
            raise KeyError(f"not cached: {spec} @ {producer_version}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_sidecar(self, spec: dict[str, Any], producer_version: str) -> None:
        """Human-readable index so a cache directory is auditable by hand."""
        path = self._path(spec, producer_version, ".spec.json")
        blob = json.dumps(
            {"spec": spec, "producer_version": producer_version},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        self._atomic_write(path, lambda fh: fh.write(blob))
