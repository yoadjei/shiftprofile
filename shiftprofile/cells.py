"""The cell: simultaneously the unit of computation and of statistical analysis.

A cell is (track, model_id, seed, shift_family, severity). Version A of the
vision grid is 3 models x 5 seeds x (1 clean + 5 corruptions x 3 severities)
= 240 cells.

Correlations are computed over cells *within* a model, never across the three
models — which is what makes the coupling claims defensible with so few models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .cache import ArtifactCache

CLEAN = "clean"


@dataclass(frozen=True, order=True)
class Cell:
    track: str
    model_id: str
    seed: int
    shift_family: str
    severity: int

    def spec(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "model_id": self.model_id,
            "seed": self.seed,
            "shift_family": self.shift_family,
            "severity": self.severity,
        }


def stage_spec(cell: Cell, stage: str, **extra: Any) -> dict[str, Any]:
    """The cache spec for one stage's artifact on one cell. THE single source.

    Every producer and every work-list check must build its key through this
    function. Constructing the dict by hand in more than one place is how a
    producer and its consumer silently disagree: `predict_cell` once wrote under
    `cell.spec()` while `fill.py` looked under `{**cell.spec(), "stage":
    "predict"}`, so the cache check could never hit and `skipped_cached` was
    always empty — a resumability guarantee that had quietly stopped holding.

    `extra` carries the axes a stage fans out over: `explainer` for attributions,
    `explainer` plus `imputation` plus `which` for removal curves. Those axes are
    what the E6 ablation varies, so results computed under different settings
    must never collide on one key.
    """
    return {**cell.spec(), "stage": stage, **extra}


def enumerate_cells(config: dict[str, Any], track: str) -> list[Cell]:
    """Expand a grid config into cells, one clean cell per (model, seed)."""
    cells: list[Cell] = []
    for model_id in config["models"]:
        for seed in config["seeds"]:
            cells.append(Cell(track, model_id, seed, CLEAN, 0))
            for family in config["shift_families"]:
                for severity in config["severities"]:
                    cells.append(Cell(track, model_id, seed, family, severity))
    return cells


def worklist(
    cells: list[Cell],
    cache: ArtifactCache,
    producer_version: str,
    kind: str = "record",
    spec_fn: Callable[[Cell], dict[str, Any]] = Cell.spec,
) -> list[Cell]:
    """Cells whose artifact is absent from the cache, in deterministic order.

    `spec_fn` lets a producer extend the cell's identity — attributions are
    keyed on (cell, explainer), not cell alone.

    Re-derived from scratch on every session start. A half-written cell does not
    exist as far as this function is concerned, because cache writes are atomic.

    Ordering is lexicographic by cell. Spec §5 calls for cost-ascending ordering
    so a short session completes whole units; that lands with the P3 runner, once
    there are measured per-cell costs to sort by.
    """
    return [
        c for c in sorted(cells)
        if not cache.has(spec_fn(c), producer_version, kind=kind)
    ]
