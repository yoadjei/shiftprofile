"""Budgeted workload filler for Kaggle sessions.

On Kaggle, sessions are 12 hours and can die at any moment. A run is not "do
the experiment" but "fill whatever cells are missing until time runs out,
then stop cleanly".

The filler works in cost-ascending order (predict → explain → curves), checks
the budget BEFORE starting each unit of work (never mid-work), records failures
and continues, and reports what was not reached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .cache import ArtifactCache
from .cells import Cell, enumerate_cells, worklist, CLEAN, stage_spec


@dataclass(frozen=True)
class StageSpec:
    """Specification for a pipeline stage and how it caches artifacts."""

    name: str
    version: str
    kind: str  # "array" or "record"
    per_explainer: bool  # does this stage fan out over explainers?
    per_imputation: bool  # does it also fan out over imputation schemes?


@dataclass
class FillReport:
    """Summary of what a fill() run completed, skipped, and failed on."""

    completed: list[str] = field(default_factory=list)
    """List of "stage:cell_id" strings for cells that finished successfully."""

    skipped_cached: list[str] = field(default_factory=list)
    """List of "stage:cell_id" strings for cells that were already cached."""

    failed: list[tuple[str, str]] = field(default_factory=list)
    """List of (cell_id, error_message) tuples for cells that raised."""

    not_implemented: list[str] = field(default_factory=list)
    """List of "stage:cell_id" strings for cells whose stage was not implemented."""

    elapsed_seconds: float = 0.0
    """Wall time consumed by the fill run."""

    budget_exhausted: bool = False
    """Whether the budget ran out before reaching the end of the work list."""

    def summary(self) -> str:
        """Human-readable summary of what happened and what was not reached.

        This is key: silent truncation reads as "covered everything" when it
        did not. This summary states explicitly what was NOT done and why.
        """
        lines = []
        lines.append(
            f"Completed {len(self.completed)} cells, "
            f"skipped {len(self.skipped_cached)} cached, "
            f"failed {len(self.failed)} cells"
        )

        if self.not_implemented:
            lines.append(
                f"Not implemented: {len(self.not_implemented)} cells "
                f"({', '.join(self.not_implemented[:3])}{'...' if len(self.not_implemented) > 3 else ''})"
            )

        if self.budget_exhausted:
            lines.append("Budget exhausted before reaching the end of the work list.")

        if self.failed:
            lines.append(
                f"Failures: {', '.join(cell_id for cell_id, _ in self.failed)}"
            )

        lines.append(f"Elapsed time: {self.elapsed_seconds:.1f}s")

        return "\n".join(lines)


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a config from a YAML file.

    Args:
        path: Path to a .yaml config file.

    Returns:
        Dictionary with keys like "name", "track", "models", "seeds", etc.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def fill(
    config: dict[str, Any],
    cache: ArtifactCache,
    *,
    budget_minutes: float,
    device: str = "cpu",
    stages: tuple[str, ...] = ("predict", "explain", "curves"),
    progress: Optional[Callable[[str, str], None]] = None,
    stages_impl: Optional[dict[str, Callable]] = None,
    now: Callable[[], float] = time.monotonic,
) -> FillReport:
    """Fill the work list until the budget runs out.

    Processes cells in cost-ascending order (predict → explain → curves).
    Checks the budget BEFORE starting each unit of work, never mid-work.
    Failures are recorded and do not stop the filler. Cached cells are
    skipped silently. The filler is idempotent: running twice completes
    nothing new the second time.

    Args:
        config: Config dict with keys "track", "models", "seeds",
                "shift_families", "severities", "explainers", etc.
        cache: ArtifactCache instance (write_root set, read_roots optional).
        budget_minutes: Time budget in minutes. Work begins only if there
                        is budget remaining; a unit is never started if
                        it would exceed the budget.
        device: Device to run on (default "cpu"). Passed to stage functions.
        stages: Tuple of stage names to run, in order. Default is
                ("predict", "explain", "curves").
        progress: Optional callback fn(stage, cell_id) for logging.
        stages_impl: Optional dict {stage_name: stage_function} for testing.
                     If not provided, uses the default implementations from
                     predict.py, explain.py, curves.py (which require real
                     data and models and are not available in pure CPU mode).
        now: Injected clock function for testing. Defaults to time.monotonic().

    Returns:
        FillReport with completed, skipped_cached, failed, elapsed_seconds,
        and budget_exhausted.
    """
    if stages_impl is None:
        stages_impl = {}

    # Define stage specifications
    stage_specs = {
        "predict": StageSpec("predict", "predict-v1", "array", per_explainer=False, per_imputation=False),
        "explain": StageSpec("explain", "explain-v1", "array", per_explainer=True, per_imputation=False),
        "curves": StageSpec("curves", "curves-v1", "array", per_explainer=True, per_imputation=True),
    }

    report = FillReport()
    start_time = now()
    budget_seconds = budget_minutes * 60

    # Enumerate cells from config
    all_cells = enumerate_cells(config, config["track"])
    explainers = config.get("explainers", [])
    imputations = config.get("imputations", ["mean"])  # Default to mean if not specified

    # Process each stage
    for stage_name in stages:
        if stage_name not in stage_specs:
            continue

        spec = stage_specs[stage_name]
        stage_fn = stages_impl.get(stage_name)

        # Build the work list for this stage, expanding by explainer and imputation
        for cell in all_cells:
            # Expand by explainer if needed
            explainer_list = explainers if spec.per_explainer else [None]

            for explainer in explainer_list:
                # Expand by imputation if needed
                imputation_list = imputations if spec.per_imputation else [None]

                for imputation in imputation_list:
                    # Build the spec for this work unit
                    cell_spec = stage_spec(cell, spec.name)
                    if spec.per_explainer:
                        cell_spec["explainer"] = explainer
                    if spec.per_imputation:
                        cell_spec["imputation"] = imputation

                    # Check cache
                    if cache.has(cell_spec, spec.version, kind=spec.kind):
                        # Build cell_id string
                        cell_id = f"{spec.name}:{cell.model_id}:{cell.seed}:{cell.shift_family}:{cell.severity}"
                        if spec.per_explainer:
                            cell_id += f":{explainer}"
                        if spec.per_imputation:
                            cell_id += f":{imputation}"
                        report.skipped_cached.append(cell_id)
                        continue

                    # Check budget before starting
                    elapsed = now() - start_time
                    if elapsed > budget_seconds:
                        report.budget_exhausted = True
                        report.elapsed_seconds = now() - start_time
                        return report

                    # Build cell_id string
                    cell_id = f"{spec.name}:{cell.model_id}:{cell.seed}:{cell.shift_family}:{cell.severity}"
                    if spec.per_explainer:
                        cell_id += f":{explainer}"
                    if spec.per_imputation:
                        cell_id += f":{imputation}"

                    # Handle missing implementation
                    if stage_fn is None:
                        report.not_implemented.append(cell_id)
                        continue

                    try:
                        if progress:
                            progress(spec.name, cell_id)

                        # Call stage function with appropriate kwargs
                        kw = {"device": device}
                        if spec.per_explainer:
                            kw["explainer"] = explainer
                        if spec.per_imputation:
                            kw["imputation"] = imputation

                        stage_fn(cell, cache, **kw)

                        report.completed.append(cell_id)
                    except Exception as e:
                        report.failed.append((cell_id, str(e)))

    report.elapsed_seconds = now() - start_time
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: python -m shiftprofile.fill [options]

    Example:
        python -m shiftprofile.fill \\
            --config configs/pilot.yaml \\
            --budget-minutes 660 \\
            --cache-write /kaggle/working/cache \\
            --cache-read /kaggle/input/shiftprofile-cache \\
            --device cuda
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fill work list until budget runs out."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        required=True,
        help="Time budget in minutes",
    )
    parser.add_argument(
        "--cache-write",
        type=Path,
        required=True,
        help="Cache directory for writes",
    )
    parser.add_argument(
        "--cache-read",
        type=Path,
        action="append",
        default=[],
        help="Read-only cache directory (repeatable)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["predict", "explain", "curves"],
        help="Stages to run (default: predict explain curves)",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    cache = ArtifactCache(args.cache_write, read_roots=args.cache_read)

    # Import the actual stage implementations
    from .predict import predict_cell
    from .explain import explain_cell
    from .curves import curves_cell

    # Determine clean_root and corrupt_root from config or environment
    # For now, use empty strings to trigger real data loading on Kaggle
    clean_root = config.get("clean_root", "")
    corrupt_root = config.get("corrupt_root", "")

    stages_impl = {
        "predict": lambda cell, cache, device="cpu", **kw: predict_cell(
            cell, None, clean_root, corrupt_root, cache, device=device, **kw
        ),
        "explain": lambda cell, cache, device="cpu", explainer=None, **kw: explain_cell(
            cell, None, clean_root, corrupt_root, cache,
            explainer=explainer, device=device, **kw
        ),
        "curves": lambda cell, cache, device="cpu", explainer=None, imputation=None, **kw: curves_cell(
            cell, None, clean_root, corrupt_root, cache,
            explainer=explainer, imputation=imputation, device=device, **kw
        ),
    }

    report = fill(
        config,
        cache,
        budget_minutes=args.budget_minutes,
        device=args.device,
        stages=tuple(args.stages),
        stages_impl=stages_impl,
    )

    print(report.summary())
    return 0 if not report.failed else 1
