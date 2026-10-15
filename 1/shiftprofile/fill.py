"""Budgeted workload filler for Kaggle sessions.

On Kaggle, sessions are 12 hours and can die at any moment. A run is not "do
the experiment" but "fill whatever cells are missing until time runs out,
then stop cleanly".

The filler works in cost-ascending order (predict → explain → curves), checks
the budget BEFORE starting each unit of work (never mid-work), records failures
and continues, and reports what was not reached.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .cache import ArtifactCache
from .cells import Cell, enumerate_cells


@dataclass(frozen=True)
class StageSpec:
    """How a stage fans out, and who to ask whether a unit is already done.

    `version` and `kind` used to live here, which meant the filler held a second
    copy of facts the stage module owned and could disagree with it. It did: the
    curves key built here omitted the `which` axis that `curves_cell` writes
    under, so the check could never hit. `is_cached` replaces both -- the stage
    module answers the question, because the stage module is what knows.
    """

    name: str
    per_explainer: bool  # does this stage fan out over explainers?
    per_imputation: bool  # does it also fan out over imputation schemes?
    is_cached: Optional[Callable[..., bool]] = None
    """`(cell, cache, **axes) -> bool`, from the stage module. None in tests that
    inject their own stages and want every unit treated as missing."""
    computes_attributions: bool = False
    """True for the stage that produces attributions. It receives the resolved
    options expanded as keywords, because that is how it is parameterised."""
    reads_attributions: bool = False
    """True for a stage that looks attributions up instead of computing them. It
    receives the same options as one dict, naming which attributions it wants.

    Two fields rather than one flag because the two stages need the same options
    in different shapes, and the translation belongs here -- resolved once -- not
    in each caller. curves_cell once looked attributions up under explain's
    DEFAULT options, so a run attributing at 4 IG steps had its curves hunt for
    the 32-step artifact and miss on every unit, forever."""


class RunManifest:
    """An append-only record of what a run did, written as it happens.

    Artifacts were already checkpointed -- each stage writes its own file
    atomically, so a resumed run skips what is on disk. What did not survive a
    preempted session was the account of the run itself: the FillReport existed
    only in memory and printed at the end, so a session killed at hour eleven
    left no record of what it had done or how long anything took.

    That second part turned out to matter more than it sounds. A pilot ran for
    six hours before anyone could tell that one attribution cell was taking
    fifty-five minutes, because nothing recorded per-cell timing. The cost of
    the full grid was derivable from the first cell and nobody had the number.

    One JSON object per line, flushed and fsynced as it is written, so a hard
    kill loses at most the line in flight. A unit is written twice: once when
    it starts and once when it ends. An entry with no matching end is the unit
    that was running when the session died, which is the thing you most want to
    know and the thing a summary printed at exit can never tell you.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def record(self, **fields: Any) -> None:
        self._handle.write(json.dumps(fields, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "RunManifest":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def previous_training_recipe(path: Path | str) -> Optional[dict[str, Any]]:
        """The training recipe the last run against this cache used, if recorded.

        Read back rather than merely written, because the recipe is the one
        thing that shapes every artifact in the cache without appearing in any
        artifact's key. See `check_training_recipe`.

        A malformed or truncated line is skipped rather than raised on: the
        manifest is append-only and fsynced per line, so the only line that can
        be broken is one from a session killed mid-write, and that is a thing to
        tolerate rather than a reason to refuse the run.
        """
        path = Path(path)
        if not path.exists():
            return None
        recipe = None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and entry.get("training") is not None:
                    recipe = entry["training"]
        return recipe


def restrict_grid(config: dict, *, seeds=None, models=None) -> dict:
    """A copy of `config` narrowed to these seeds and models.

    A 12-hour preemptible session does not hold the whole grid, so a run is
    split along the axes `enumerate_cells` builds from. The alternative in use
    until now was editing the config inside the session, which leaves the
    manifest naming a config path whose contents no longer match what ran, and
    the committed file no longer describing any actual run.

    A value not in the config is refused rather than silently producing an
    empty grid. `--seeds 1` against a config listing `[0]` is a typo every time,
    and a run that quietly does nothing looks exactly like a run that found
    everything cached.

    Args:
        config: Parsed config. Not mutated.
        seeds: Seeds to keep, or None to keep all.
        models: Model ids to keep, or None to keep all.

    Returns:
        A new config dict.

    Raises:
        ValueError: If any requested value is absent from the config.
    """
    out = dict(config)
    for key, requested in (("seeds", seeds), ("models", models)):
        if requested is None:
            continue
        available = list(config[key])
        missing = [v for v in requested if v not in available]
        if missing:
            raise ValueError(
                f"--{key} {missing} not in the config's {key} {available}. "
                f"Restricting the grid can only narrow it; it cannot add a "
                f"{key[:-1]} the config does not pre-register."
            )
        # Config order, not command-line order, so two sessions naming the same
        # subset in different orders enumerate cells identically.
        out[key] = [v for v in available if v in requested]
    return out


def check_training_recipe(
    previous: Optional[dict[str, Any]], current: dict[str, Any]
) -> Optional[str]:
    """Why this run must not write into this cache, or None if it may.

    The training hyperparameters are in the *checkpoint's* key, so a changed
    `epochs` correctly trains a new model. They are not in the keys of anything
    downstream: `predict`, `explain` and `curves` are keyed on the cell, which
    names `model_id` and `seed` and nothing about how the model was trained. So
    a cache that has seen two recipes holds logits from one model under keys a
    later run will happily reuse for another.

    The realistic way in is not a protocol change. It is a scratch config with
    `epochs: 2` run against the shared cache to check the pipeline works, which
    on a quota-limited project is a normal thing to do, followed by the real
    config. Nothing in the keys records that the first run happened.

    Refusing is the remedy rather than warning, and the fix offered is a
    separate `--cache-write` root rather than a flag, because a cache holding
    two training recipes is not a coherent artifact store and there is no
    reading of it that makes the mixed results meaningful.

    Args:
        previous: Recipe recorded by an earlier run, or None if there is none.
        current: Recipe this run would use.

    Returns:
        An explanatory message if the run must not proceed, else None.
    """
    if previous is None or previous == current:
        return None
    changed = sorted(
        k for k in set(previous) | set(current) if previous.get(k) != current.get(k)
    )
    lines = [
        "this cache was built with different training hyperparameters:",
        "",
    ]
    lines += [
        f"    {k}: {previous.get(k)!r} previously, {current.get(k)!r} now" for k in changed
    ]
    lines += [
        "",
        "They are in the checkpoint's cache key, so the model would be retrained,",
        "but predict/explain/curves are keyed on the cell alone. This run would",
        "reuse artifacts computed from the earlier model's weights and report",
        "them as this model's.",
        "",
        "Write to a different --cache-write root, or restore the earlier values.",
    ]
    return "\n".join(lines)


class ProgressReporter:
    """Rate-limited progress, because an unbounded one cost a two-hour run.

    The filler printed one line per work unit. At 510 units that overran
    Kaggle's notebook output buffer, and the buffer discards the OLDEST lines --
    so the three report tables, which print last, were gone, the cache went with
    the session, and the whole run had to be repeated.

    Printing nothing is not the alternative. A silent cell on a preemptible
    runner is indistinguishable from a hung one, which is how a healthy nine-hour
    pilot nearly got killed by hand, and per-unit timing is the number that
    diagnosed the 55-minute attribution cell nobody could see.

    So: every unit of a stage until `verbose_units` have gone by, then one
    counted line every `every` units carrying the elapsed time and the per-unit
    rate. A stage short enough to read in full still prints in full; a stage of
    four hundred prints about sixteen lines instead of four hundred, and each one
    says more than the line it replaces.
    """

    def __init__(
        self,
        *,
        verbose_units: int = 12,
        every: int = 25,
        now: Callable[[], float] = time.monotonic,
    ):
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self.verbose_units = verbose_units
        self.every = every
        self._now = now
        self._stage: Optional[str] = None
        self._seen = 0
        self._stage_start = 0.0

    def __call__(self, stage: str, cell_id: str) -> None:
        if stage != self._stage:
            self._stage = stage
            self._seen = 0
            self._stage_start = self._now()

        self._seen += 1
        if self._seen <= self.verbose_units:
            print(f"  {stage}: {cell_id}", flush=True)
            return

        if self._seen % self.every == 0:
            elapsed = self._now() - self._stage_start
            print(
                f"  {stage}: {self._seen} units, {elapsed:.0f}s "
                f"({elapsed / self._seen:.1f}s/unit), at {cell_id}",
                flush=True,
            )


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
    is_cached_impl: Optional[dict[str, Callable]] = None,
    manifest: Optional["RunManifest"] = None,
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
        is_cached_impl: Optional dict {stage_name: predicate} for testing, where
                     each predicate is `(cell, cache, **axes) -> bool`. Defaults
                     to the real `<stage>_is_cached` from each stage module. A
                     stage with no predicate has every unit treated as missing.
        manifest: Optional RunManifest. Each unit is recorded as it starts and
                  again as it ends, with its duration, so a preempted session
                  leaves a readable account of what it did and what cost what.
        now: Injected clock function for testing. Defaults to time.monotonic().

    Returns:
        FillReport with completed, skipped_cached, failed, elapsed_seconds,
        and budget_exhausted.
    """
    if stages_impl is None:
        stages_impl = {}

    # Each stage module answers "is this unit done?" itself. The filler used to
    # build the key and ask the cache directly, holding a second copy of facts the
    # module owned -- and disagreeing with it, because the curves key built here
    # omitted the `which` axis curves_cell writes under, so the check could never
    # hit and a resumed run re-reported every curve as freshly completed.
    # Imported inside the function because these modules pull in torch, which the
    # tests that inject their own stages should not have to pay for.
    if is_cached_impl is None:
        from .curves import curves_is_cached
        from .explain import explain_is_cached
        from .predict import predict_is_cached

        is_cached_impl = {
            "predict": predict_is_cached,
            "explain": explain_is_cached,
            "curves": curves_is_cached,
        }

    stage_specs = {
        "predict": StageSpec("predict", per_explainer=False, per_imputation=False,
                             is_cached=is_cached_impl.get("predict")),
        "explain": StageSpec("explain", per_explainer=True, per_imputation=False,
                             is_cached=is_cached_impl.get("explain"),
                             computes_attributions=True),
        "curves": StageSpec("curves", per_explainer=True, per_imputation=True,
                            is_cached=is_cached_impl.get("curves"),
                            reads_attributions=True),
    }

    report = FillReport()
    start_time = now()
    budget_seconds = budget_minutes * 60

    # Resolve the evaluation set ONCE, for every stage and every cell. Required,
    # never defaulted: `n_eval_images` sat in both configs and reached no producer
    # at all, so every run scored the whole 10,000-image test set while the
    # protocol asked for 1,000 and the cache key recorded neither. A default here
    # would restore exactly that silence, so a config without the key is an error.
    if "n_eval_images" not in config:
        raise ValueError(
            "config has no 'n_eval_images'. It decides how many images every "
            "artifact is computed over and is part of every cache key, so there "
            "is no safe default: guessing would silently produce artifacts that "
            "do not match the registered protocol."
        )
    from .data import fixed_eval_indices

    indices = fixed_eval_indices(config["n_eval_images"])

    # Resolved once, from the config, for every stage that computes attributions
    # and every stage that reads them. One source, so a curve cannot be looked up
    # under settings its attributions were not computed with.
    #
    # Built by the explain module, not here, and read by the report through the
    # same function. `baseline` is in it because it was NOT, and `explainer_options`
    # has carried it in the cache key all along on the strength of a comment saying
    # E6 varies it. E6 could not vary it: no config key reached this dict, so every
    # run attributed at the default and the ablation axis existed only in the key.
    from .explain import explain_options_from_config

    explain_options = explain_options_from_config(config)

    # Enumerate cells from config
    all_cells = enumerate_cells(config, config["track"])
    explainers = config.get("explainers", [])

    # Both configs spell this `imputation` (singular) while this read it as
    # `imputations`, so the key was never looked at: a config asking for `blur`
    # would have silently run `mean`. The values happened to agree, which is the
    # only reason it never showed. Both spellings are accepted now, singular as
    # the one scheme the frozen protocol uses and plural as E6's list.
    if "imputations" in config:
        imputations = config["imputations"]
    elif "imputation" in config:
        imputations = [config["imputation"]]
    else:
        imputations = ["mean"]

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
                    # The axes this unit fans out over. The same dict goes to the
                    # stage module's predicate and to the stage itself, so what is
                    # checked and what is computed cannot be different units.
                    axes = {"indices": indices}
                    if spec.per_explainer:
                        axes["explainer"] = explainer
                    if spec.per_imputation:
                        axes["imputation"] = imputation
                    # The producer is parameterised by these; the consumer is
                    # identified by them. Same dict, resolved once, shaped here so
                    # no stage or test stub has to remember which form it needs.
                    if spec.computes_attributions:
                        axes.update(explain_options)
                    if spec.reads_attributions:
                        axes["explain_options"] = explain_options

                    cell_id = f"{spec.name}:{cell.model_id}:{cell.seed}:{cell.shift_family}:{cell.severity}"
                    if spec.per_explainer:
                        cell_id += f":{explainer}"
                    if spec.per_imputation:
                        cell_id += f":{imputation}"

                    # Ask the stage module, which owns its key, rather than
                    # building one here and asking the cache.
                    if spec.is_cached is not None and spec.is_cached(cell, cache, **axes):
                        report.skipped_cached.append(cell_id)
                        if manifest:
                            manifest.record(
                                event="skipped_cached",
                                stage=spec.name,
                                cell_id=cell_id,
                            )
                        continue

                    # Check budget before starting
                    elapsed = now() - start_time
                    if elapsed > budget_seconds:
                        report.budget_exhausted = True
                        report.elapsed_seconds = now() - start_time
                        if manifest:
                            manifest.record(
                                event="budget_exhausted",
                                stage=spec.name,
                                elapsed_seconds=round(report.elapsed_seconds, 3),
                            )
                        return report

                    # Handle missing implementation
                    if stage_fn is None:
                        report.not_implemented.append(cell_id)
                        continue

                    # Written before the work, not after. A unit with a start
                    # and no end is the one that was running when the session
                    # died -- the single most useful line in the file, and one
                    # a record written on completion can never contain.
                    if manifest:
                        manifest.record(
                            event="started", stage=spec.name, cell_id=cell_id
                        )
                    unit_start = now()

                    try:
                        if progress:
                            progress(spec.name, cell_id)

                        # The same axes the predicate was asked about, so the unit
                        # computed is the unit that was found missing.
                        stage_fn(cell, cache, device=device, **axes)

                        report.completed.append(cell_id)
                        if manifest:
                            manifest.record(
                                event="completed",
                                stage=spec.name,
                                cell_id=cell_id,
                                seconds=round(now() - unit_start, 3),
                            )
                    except Exception as e:
                        report.failed.append((cell_id, str(e)))
                        if manifest:
                            manifest.record(
                                event="failed",
                                stage=spec.name,
                                cell_id=cell_id,
                                seconds=round(now() - unit_start, 3),
                                error=f"{type(e).__name__}: {e}",
                            )

    report.elapsed_seconds = now() - start_time
    if manifest:
        manifest.record(
            event="finished", elapsed_seconds=round(report.elapsed_seconds, 3)
        )
    return report


def _resolve_device(requested: str) -> str:
    """Turn "auto" into what this machine actually has.

    Neither fixed default is safe: "cpu" would silently run a GPU job on CPU for
    hours, and "cuda" would crash on a laptop. "auto" decides, and the caller
    prints the answer so the choice is on the record next to the timings.
    """
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: python -m shiftprofile.fill [options]

    Example:
        python -m shiftprofile.fill \
            --config configs/pilot.yaml \
            --budget-minutes 660 \
            --cache-write /kaggle/working/cache \
            --cache-read /kaggle/input/shiftprofile-cache \
            --data-root /kaggle/working/data \
            --corrupt-root /kaggle/input/cifar-10-c
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fill the work list until the budget runs out."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to a config YAML")
    parser.add_argument(
        "--budget-minutes", type=float, required=True, help="Wall-clock budget in minutes"
    )
    parser.add_argument("--cache-write", type=Path, required=True, help="Writable cache root")
    parser.add_argument(
        "--cache-read", type=Path, action="append", default=[],
        help="Read-only cache root; repeatable",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="CIFAR-10 root. Downloaded here if absent.",
    )
    parser.add_argument(
        "--corrupt-root", type=Path, default=Path("data"),
        help="CIFAR-10-C root, nested or flat",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto (default)")
    parser.add_argument(
        "--stages", nargs="+", default=["predict", "explain", "curves"],
        help="Stages to run, in cost-ascending order",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Run log, one JSON object per line. Defaults to "
             "run_manifest.jsonl inside the write cache, so it is saved "
             "along with the artifacts.",
    )
    # A 12-hour preemptible session cannot hold the whole grid, so a run is
    # split along the axes the grid is built from. Doing that by editing the
    # config in the session means the committed config no longer describes the
    # run, and the manifest records a config path whose contents have changed.
    # These narrow the grid without touching the file, and are recorded below.
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Run only these seeds. Must be a subset of the config's. "
             "Splits a grid across sessions without editing the config.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Run only these model ids. Must be a subset of the config's.",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = restrict_grid(config, seeds=args.seeds, models=args.models)
    cache = ArtifactCache(args.cache_write, read_roots=args.cache_read)
    device = _resolve_device(args.device)

    from .curves import curves_cell
    from .explain import explain_cell
    from .predict import predict_cell
    from .train import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_LR,
        load_or_train,
    )

    # One trained model per (model_id, seed), held for the life of the process.
    # Every stage of every shift cell sharing a model and seed needs the same
    # weights, so without this load_or_train would deserialise the same
    # checkpoint once per cell per stage.
    models: dict[tuple[str, int], Any] = {}

    def training_progress(epoch: int, total: int, metrics: dict) -> None:
        """Report during training, which is the long silent stretch of a session.

        Fifty epochs produce no output at all otherwise, and on a preemptible
        runner a silent cell is indistinguishable from a hung one — which invites
        killing a run by hand twenty minutes into work that was fine. Every fifth
        epoch is frequent enough to show liveness without burying the per-cell
        lines that follow.
        """
        if epoch % 5 == 0 or epoch == total or epoch == 1:
            loss = metrics.get("loss")
            suffix = f", loss {loss:.4f}" if isinstance(loss, (int, float)) else ""
            print(f"    epoch {epoch}/{total}{suffix}", flush=True)

    def model_for(cell: Cell):
        key = (cell.model_id, cell.seed)
        if key not in models:
            # Say so before the silence, not after it. A cache hit returns at once
            # and prints nothing more; a miss trains, and the reader needs to know
            # which of the two is happening.
            print(
                f"  model {cell.model_id} seed {cell.seed}: loading from cache or training",
                flush=True,
            )
            model, _ = load_or_train(
                cell.model_id,
                cell.seed,
                args.data_root,
                cache,
                device=device,
                # Defaults come from train.py rather than being repeated here.
                # Both numbers are in the training cache key, so a fallback that
                # drifted from the real default would key a checkpoint on
                # hyperparameters it was not trained with.
                epochs=config.get("epochs", DEFAULT_EPOCHS),
                batch_size=config.get("batch_size", DEFAULT_BATCH_SIZE),
                lr=config.get("lr", DEFAULT_LR),
                progress=training_progress,
            )
            models[key] = model
        return models[key]

    clean_root = str(args.data_root)
    corrupt_root = str(args.corrupt_root)

    def predict_stage(cell, cache, device="cpu", **kw):
        return predict_cell(
            cell, model_for(cell), clean_root, corrupt_root, cache, device=device, **kw
        )

    def explain_stage(cell, cache, device="cpu", explainer=None, **kw):
        # ig_steps and friends arrive in kw already, resolved once by fill from the
        # config and handed to curves as well, so the settings an attribution is
        # computed under are the settings curves looks it up by.
        return explain_cell(
            cell, model_for(cell), clean_root, corrupt_root, cache,
            explainer=explainer, device=device, **kw
        )

    def curves_stage(cell, cache, device="cpu", explainer=None, imputation=None, **kw):
        return curves_cell(
            cell, model_for(cell), clean_root, corrupt_root, cache,
            explainer=explainer, imputation=imputation, device=device, **kw
        )

    print(f"config      {args.config}")
    # Printed unconditionally, not only when narrowed: a log that names the
    # seeds only sometimes leaves the reader unable to tell a restricted run
    # from a full one by looking.
    print(f"models      {', '.join(map(str, config['models']))}")
    print(f"seeds       {', '.join(map(str, config['seeds']))}")
    print(f"device      {device}")
    print(f"budget      {args.budget_minutes:.0f} min")
    print(f"cache write {args.cache_write}")
    print(f"cache read  {', '.join(str(p) for p in args.cache_read) or '(none)'}")
    # Default it into the write cache so that whatever persists the artifacts
    # persists the account of how they were made, with no second thing to
    # remember to save.
    manifest_path = args.manifest or (args.cache_write / "run_manifest.jsonl")

    print(f"data        {clean_root}")
    print(f"corrupt     {corrupt_root}")
    print(f"stages      {', '.join(args.stages)}")
    print(f"manifest    {manifest_path}")
    print()

    training_recipe = {
        "epochs": config.get("epochs", DEFAULT_EPOCHS),
        "batch_size": config.get("batch_size", DEFAULT_BATCH_SIZE),
        "lr": config.get("lr", DEFAULT_LR),
    }
    refusal = check_training_recipe(
        RunManifest.previous_training_recipe(manifest_path), training_recipe
    )
    if refusal is not None:
        print(f"REFUSING: {refusal}")
        return 2

    with RunManifest(manifest_path) as manifest:
        manifest.record(
            event="run_start",
            # The recipe shapes every artifact in this cache and appears in no
            # artifact's key but the checkpoint's, so the manifest is the only
            # place it is recorded. `check_training_recipe` reads it back.
            training=training_recipe,
            config=str(args.config),
            # What this session actually ran, after --seeds/--models. The
            # config path alone stops identifying the run once a grid is split
            # across sessions, and these are the axes it is split on.
            seeds=list(config["seeds"]),
            models=list(config["models"]),
            device=device,
            budget_minutes=args.budget_minutes,
            stages=list(args.stages),
            # Recorded because it is part of every artifact's key: a manifest
            # spanning sessions must say which evaluation set each run used.
            n_eval_images=config.get("n_eval_images"),
            ig_steps=config.get("ig_steps"),
            # Wall-clock, unlike the monotonic clock used for durations. A
            # manifest spanning several preempted sessions is unreadable
            # without knowing which day each run happened.
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        report = fill(
            config,
            cache,
            budget_minutes=args.budget_minutes,
            device=device,
            stages=tuple(args.stages),
            stages_impl={
                "predict": predict_stage,
                "explain": explain_stage,
                "curves": curves_stage,
            },
            progress=ProgressReporter(),
            manifest=manifest,
        )

    print()
    print("=" * 60)
    print(report.summary())
    print(f"manifest written to {manifest_path}")
    print("=" * 60)
    return 0 if not report.failed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
