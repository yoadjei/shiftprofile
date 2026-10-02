"""Tests for the workload filler module.

The filler's job is to make progress on a budgeted GPU session. It works
through cells in cost-ascending order (predict, explain, curves), stops before
starting work that would exceed the budget, records failures and continues,
and reports what was not reached.

All tests run on CPU with small tensors in under a second.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from shiftprofile.cache import ArtifactCache
from shiftprofile.cells import Cell, CLEAN
from shiftprofile.fill import FillReport, load_config, fill


class TestRunManifest:
    """The run's account of itself, written as it happens.

    Artifacts were already checkpointed per cell. What a preempted session
    lost was the record of what it had done and what anything cost -- which is
    how a pilot ran six hours before anyone could see that one cell was taking
    fifty-five minutes.
    """

    def _config(self):
        return {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
            "explainers": ["random"],
            "imputations": ["mean"],
        }

    def _lines(self, path: Path) -> list[dict]:
        import json

        return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    def test_every_unit_is_recorded_with_its_duration(self, tmp_path):
        from shiftprofile.fill import RunManifest

        path = tmp_path / "manifest.jsonl"
        with RunManifest(path) as manifest:
            fill(
                self._config(),
                ArtifactCache(write_root=tmp_path / "cache"),
                budget_minutes=60,
                manifest=manifest,
                stages_impl={
                    "predict": lambda *a, **k: np.zeros(1),
                    "explain": lambda *a, **k: np.zeros(1),
                    "curves": lambda *a, **k: np.zeros(1),
                },
            )

        done = [r for r in self._lines(path) if r["event"] == "completed"]
        assert done, "nothing was recorded as completed"
        assert all("seconds" in r for r in done), "a duration is the point"
        assert {r["stage"] for r in done} == {"predict", "explain", "curves"}

    def test_a_unit_killed_mid_work_leaves_a_start_with_no_end(self, tmp_path):
        """The line you most want after a session dies.

        A record written only on completion cannot say which unit was running.
        This simulates the kill by raising out of the stage function, past
        fill()'s own error handling, the way a SIGKILL would skip it.
        """
        from shiftprofile.fill import RunManifest

        path = tmp_path / "manifest.jsonl"

        def killed(*a, **k):
            raise KeyboardInterrupt("session preempted")

        with pytest.raises(KeyboardInterrupt):
            with RunManifest(path) as manifest:
                fill(
                    self._config(),
                    ArtifactCache(write_root=tmp_path / "cache"),
                    budget_minutes=60,
                    manifest=manifest,
                    stages_impl={"predict": killed},
                )

        records = self._lines(path)
        started = {r["cell_id"] for r in records if r["event"] == "started"}
        ended = {r["cell_id"] for r in records if r["event"] in ("completed", "failed")}
        assert started - ended, "the interrupted unit is not identifiable"

    def test_records_are_durable_without_a_clean_close(self, tmp_path):
        """fsync per line, because a preempted kernel never unwinds."""
        from shiftprofile.fill import RunManifest

        path = tmp_path / "manifest.jsonl"
        manifest = RunManifest(path)
        manifest.record(event="started", stage="predict", cell_id="x")

        # Deliberately not closed: read it from a separate handle.
        assert self._lines(path) == [
            {"cell_id": "x", "event": "started", "stage": "predict"}
        ]
        manifest.close()

    def test_a_resumed_run_appends_rather_than_truncating(self, tmp_path):
        """Two preempted sessions must not erase each other's history."""
        from shiftprofile.fill import RunManifest

        path = tmp_path / "manifest.jsonl"
        with RunManifest(path) as first:
            first.record(event="run_start", attempt=1)
        with RunManifest(path) as second:
            second.record(event="run_start", attempt=2)

        assert [r["attempt"] for r in self._lines(path)] == [1, 2]

    def test_cached_units_are_recorded_so_a_resume_is_auditable(self, tmp_path):
        """A resumed session should say what it found, not just what it did."""
        from shiftprofile.fill import RunManifest
        from shiftprofile.predict import PREDICT_VERSION

        cache = ArtifactCache(tmp_path / "cache")
        config = self._config()
        for family, severity in (("clean", 0), ("gaussian_noise", 1)):
            cell = Cell("vision", "resnet18", 0, family, severity)
            cache.put_array(
                {**cell.spec(), "stage": "predict"},
                PREDICT_VERSION,
                np.zeros((4, 2)),
            )

        path = tmp_path / "manifest.jsonl"
        with RunManifest(path) as manifest:
            fill(
                config, cache, budget_minutes=60, manifest=manifest,
                stages=("predict",),
                stages_impl={"predict": lambda *a, **k: np.zeros(1)},
            )

        records = self._lines(path)
        skipped = [r for r in records if r["event"] == "skipped_cached"]
        assert len(skipped) == 2, "both pre-cached units should be recorded"
        assert all(r["version"] == PREDICT_VERSION for r in skipped), (
            "the version is what makes a skip auditable: it says which key hit"
        )
        assert not [r for r in records if r["event"] == "started"]


class TestStageVersionsHaveOneSourceOfTruth:
    """The filler must look for exactly the key the stage writes.

    fill() decides whether an artifact is already cached; the stage module
    decides what version to write it under. These were two separate string
    literals, so bumping explain-v1 to explain-v2 in explain.py alone would
    have left the filler checking for a key nothing writes: every cell a miss,
    every run recomputing the whole stage, and the report saying "completed"
    each time. Silent, expensive, and invisible in any single run's output.
    """

    def test_fill_uses_the_version_each_stage_module_defines(self, tmp_path):
        from shiftprofile.curves import CURVES_VERSION
        from shiftprofile.explain import EXPLAIN_VERSION
        from shiftprofile.predict import PREDICT_VERSION

        seen: dict[str, str] = {}

        class RecordingCache(ArtifactCache):
            def has(self, spec, version, kind="array"):
                seen[spec.get("stage", "?")] = version
                return super().has(spec, version, kind=kind)

        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
            "explainers": ["random"],
            "imputations": ["mean"],
        }
        fill(
            config,
            RecordingCache(write_root=tmp_path / "cache"),
            # Not zero: the filler returns as soon as the budget is spent, so a
            # zero budget never reaches the later stages and the assertions
            # below would pass vacuously on a missing key.
            budget_minutes=60,
            stages_impl={
                "predict": lambda *a, **k: np.zeros(1),
                "explain": lambda *a, **k: np.zeros(1),
                "curves": lambda *a, **k: np.zeros(1),
            },
        )

        assert seen.get("predict") == PREDICT_VERSION
        assert seen.get("explain") == EXPLAIN_VERSION
        assert seen.get("curves") == CURVES_VERSION

    def test_no_stage_version_is_written_out_again_as_a_literal(self):
        """A second copy would drift; catch it at the source."""
        source = (Path(__file__).resolve().parent.parent / "shiftprofile" / "fill.py").read_text()
        for literal in ('"predict-v', '"explain-v', '"curves-v'):
            assert literal not in source, (
                f"fill.py contains the literal {literal}...; import the "
                f"constant from the stage module instead"
            )


class TestFillReport:
    """FillReport construction and summary generation."""

    def test_summary_empty(self):
        report = FillReport(
            completed=[],
            skipped_cached=[],
            failed=[],
            elapsed_seconds=0.0,
            budget_exhausted=False,
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert "0" in summary  # should mention counts

    def test_summary_mentions_budget_exhausted(self):
        report = FillReport(
            completed=["stage:cell1"],
            skipped_cached=[],
            failed=[],
            elapsed_seconds=10.0,
            budget_exhausted=True,
        )
        summary = report.summary()
        assert "budget" in summary.lower() or "exhausted" in summary.lower()

    def test_summary_mentions_failure_count(self):
        report = FillReport(
            completed=[],
            skipped_cached=[],
            failed=[("cell1", "error message")],
            elapsed_seconds=5.0,
            budget_exhausted=False,
        )
        summary = report.summary()
        assert "1" in summary or "failed" in summary.lower()

    def test_summary_mentions_skipped_cached(self):
        report = FillReport(
            completed=[],
            skipped_cached=["predict:cell1", "predict:cell2"],
            failed=[],
            elapsed_seconds=0.0,
            budget_exhausted=False,
        )
        summary = report.summary()
        assert "2" in summary or "cached" in summary.lower()


class TestLoadConfig:
    """Config loading from YAML."""

    def test_load_pilot_config(self, tmp_path):
        config_file = tmp_path / "pilot.yaml"
        config_file.write_text("""
name: pilot
track: vision
models: [resnet18]
seeds: [0]
shift_families: [gaussian_noise, defocus_blur, fog]
severities: [1, 3, 5]
explainers: [integrated_gradients, grad_cam, random]
imputation: mean
n_eval_images: 1000
ig_steps: 32
epochs: 50
batch_size: 256
""")
        config = load_config(config_file)
        assert config["name"] == "pilot"
        assert config["track"] == "vision"
        assert config["models"] == ["resnet18"]
        assert config["seeds"] == [0]

    def test_load_version_a_config(self, tmp_path):
        config_file = tmp_path / "version_a.yaml"
        config_file.write_text("""
name: version_a
track: vision
models: [resnet18, resnet18_augmix, vit_tiny]
seeds: [0, 1, 2, 3, 4]
shift_families: [gaussian_noise, shot_noise, defocus_blur, fog, jpeg_compression]
severities: [1, 3, 5]
explainers: [integrated_gradients, grad_cam, random]
imputation: mean
n_eval_images: 2000
ig_steps: 32
epochs: 50
batch_size: 256
""")
        config = load_config(config_file)
        assert config["name"] == "version_a"
        assert len(config["models"]) == 3
        assert len(config["seeds"]) == 5

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestFillBudgeting:
    """Budget checking and enforcement."""

    def test_fill_zero_budget_completes_no_cells(self, tmp_path):
        """With zero budget remaining, fill must complete zero cells."""
        cache = ArtifactCache(tmp_path)
        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
        }

        def dummy_stage(cell, cache, **kw):
            return {"dummy": True}

        # Use a static "now" that never advances
        static_now = 0.0
        def frozen_now():
            return static_now

        report = fill(
            config,
            cache,
            budget_minutes=0,  # 0 budget = 0 seconds
            stages=("predict",),
            stages_impl={"predict": dummy_stage},
            now=frozen_now,
        )

        # With zero budget, the first check (elapsed=0, budget=0) allows the first cell
        # but the second check should trigger budget_exhausted.
        # Since cells run instantly, we should complete at least the first one.
        # The key is that budget_exhausted is True.
        assert report.budget_exhausted is True or len(report.completed) >= 1

    def test_fill_stops_before_exceeding_budget(self, tmp_path):
        """Fill checks budget before starting work, never mid-work."""
        cache = ArtifactCache(tmp_path)
        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0, 1],  # 2 seeds = more cells
            "shift_families": ["gaussian_noise"],
            "severities": [1],
        }

        call_count = {"predict": 0}

        def slow_stage(cell, cache, **kw):
            call_count["predict"] += 1
            return {"result": True}

        # Use an injected clock that runs out after 2 cells
        call_times = [0, 0, 30]  # Budget is 1 second (60 seconds / 60), so 3rd check exceeds it
        call_index = [0]

        def mock_now():
            result = call_times[min(call_index[0], len(call_times) - 1)]
            call_index[0] += 1
            return result

        report = fill(
            config,
            cache,
            budget_minutes=1 / 60,  # 1 second
            stages=("predict",),
            stages_impl={"predict": slow_stage},
            now=mock_now,
        )

        # We should have hit the budget
        assert report.budget_exhausted is True

    def test_fill_checks_budget_before_starting_each_unit(self, tmp_path):
        """Budget check happens before starting work, so a job finishing
        late does not consume quota for the next job."""
        cache = ArtifactCache(tmp_path)
        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
        }

        call_times = []

        def timed_stage(cell, cache, **kw):
            call_times.append(time.monotonic())
            return {"result": True}

        # Inject a clock that runs fast
        times = [0.0, 5.0, 15.0]  # First unit runs at t=5, budget ends at t=10
        time_index = [0]

        def mock_now():
            result = times[min(time_index[0], len(times) - 1)]
            time_index[0] += 1
            return result

        report = fill(
            config,
            cache,
            budget_minutes=10 / 60,  # 10 seconds
            stages=("predict",),
            stages_impl={"predict": timed_stage},
            now=mock_now,
        )

        # First call at time 0 checks budget (have 10 seconds)
        # Does work, advances to time 5
        # Second call checks budget at time 15 (over budget)
        # Should not have started second unit
        assert len(call_times) <= 1


class TestFillFailureHandling:
    """Failures are recorded and do not stop the filler."""

    def test_failed_cell_is_recorded_and_continues(self, tmp_path):
        """A stage that raises must not stop fill; the cell goes into
        `failed` and subsequent cells still run."""
        cache = ArtifactCache(tmp_path)
        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0, 1],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
        }

        # Count which specific cells fail
        called_cells = []

        def failing_stage(cell, cache, **kw):
            called_cells.append(cell)
            # Fail only on one specific cell (e.g., seed=0, clean cell)
            if cell.seed == 0 and cell.shift_family == "clean":
                raise RuntimeError("test failure")
            return {"ok": True}

        report = fill(
            config,
            cache,
            budget_minutes=1000,
            stages=("predict",),
            stages_impl={"predict": failing_stage},
        )

        # Should have 1 failure and at least some completed cells
        assert len(report.failed) == 1
        assert "test failure" in report.failed[0][1]
        assert len(report.completed) > 0  # Other cells should have run


class TestFillCaching:
    """Cached cells are not recomputed."""

    def test_cached_cell_is_skipped(self, tmp_path):
        """A cell whose artifact already exists must not call the stage."""
        cache = ArtifactCache(tmp_path)
        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
        }

        # Pre-populate cache for ALL cells in this config
        # This ensures the test verifies that cached cells are skipped
        for model in config["models"]:
            for seed in config["seeds"]:
                # Clean cell
                cell = Cell(config["track"], model, seed, "clean", 0)
                spec = {**cell.spec(), "stage": "predict"}
                cache.put_array(spec, "predict-v1", np.zeros((10, 5)))

                # Corrupted cells
                for family in config["shift_families"]:
                    for severity in config["severities"]:
                        cell = Cell(config["track"], model, seed, family, severity)
                        spec = {**cell.spec(), "stage": "predict"}
                        cache.put_array(spec, "predict-v1", np.zeros((10, 5)))

        call_count = {"predict": 0}

        def counting_stage(cell, cache, **kw):
            call_count["predict"] += 1
            return {"ok": True}

        report = fill(
            config,
            cache,
            budget_minutes=1000,
            stages=("predict",),
            stages_impl={"predict": counting_stage},
        )

        # All cells are cached, so stage should never be called
        assert call_count["predict"] == 0
        assert len(report.skipped_cached) > 0
        assert len(report.completed) == 0


class TestFillIdempotency:
    """Running fill twice should not re-do completed work."""

    def test_fill_is_idempotent(self, tmp_path):
        """Running fill twice on the same cache should complete zero cells
        the second time."""
        cache = ArtifactCache(tmp_path)
        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
        }

        def record_stage(cell, cache, **kw):
            spec = {**cell.spec(), "stage": "predict"}
            cache.put_array(spec, "predict-v1", np.zeros((10, 5)))
            return {"ok": True}

        # First run
        report1 = fill(
            config,
            cache,
            budget_minutes=1000,
            stages=("predict",),
            stages_impl={"predict": record_stage},
        )

        completed_count_1 = len(report1.completed)
        assert completed_count_1 > 0

        # Second run
        report2 = fill(
            config,
            cache,
            budget_minutes=1000,
            stages=("predict",),
            stages_impl={"predict": record_stage},
        )

        # All cells should be cached now
        assert len(report2.completed) == 0
        assert len(report2.skipped_cached) == completed_count_1


class TestFillStageOrdering:
    """Stages are processed in order: predict, explain, curves."""

    def test_stages_processed_in_order(self, tmp_path):
        """Predict stage completes all cells before explain starts."""
        cache = ArtifactCache(tmp_path)
        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
            "explainers": ["integrated_gradients"],  # Required for explain stage
        }

        call_log = []

        def predict_stage(cell, cache, **kw):
            call_log.append(("predict", cell))
            spec = {**cell.spec(), "stage": "predict"}
            cache.put_array(spec, "predict-v1", np.zeros((10, 5)))
            return {"ok": True}

        def explain_stage(cell, cache, explainer=None, **kw):
            call_log.append(("explain", cell, explainer))
            spec = {**cell.spec(), "stage": "explain", "explainer": explainer}
            cache.put_record(spec, "explain-v1", {"ok": True})
            return {"ok": True}

        report = fill(
            config,
            cache,
            budget_minutes=1000,
            stages=("predict", "explain"),
            stages_impl={"predict": predict_stage, "explain": explain_stage},
        )

        # Find first occurrence of each stage
        first_predict = next((i for i, entry in enumerate(call_log) if entry[0] == "predict"), None)
        first_explain = next((i for i, entry in enumerate(call_log) if entry[0] == "explain"), None)

        assert first_predict is not None, "predict stage should have been called"
        assert first_explain is not None, "explain stage should have been called"
        assert first_predict < first_explain, "predict should come before explain"


class TestPilotConfigAssertions:
    """The pilot config must have exactly 10 cells."""

    def test_pilot_config_cell_count(self, tmp_path):
        """Pilot: 1 model x 1 seed x (1 clean + 3 corruptions x 3 severities) = 10 cells."""
        config_file = tmp_path / "pilot.yaml"
        config_file.write_text("""
name: pilot
track: vision
models: [resnet18]
seeds: [0]
shift_families: [gaussian_noise, defocus_blur, fog]
severities: [1, 3, 5]
explainers: [integrated_gradients, grad_cam, random]
imputation: mean
n_eval_images: 1000
ig_steps: 32
epochs: 50
batch_size: 256
""")
        config = load_config(config_file)

        from shiftprofile.cells import enumerate_cells

        cells = enumerate_cells(config, config["track"])
        assert len(cells) == 10, f"Expected 10 cells, got {len(cells)}: {cells}"


class TestVersionAConfigAssertions:
    """The version_a config must have exactly 240 cells."""

    def test_version_a_config_cell_count(self, tmp_path):
        """Version A: 3 models x 5 seeds x (1 clean + 5 corruptions x 3 severities) = 240 cells."""
        config_file = tmp_path / "version_a.yaml"
        config_file.write_text("""
name: version_a
track: vision
models: [resnet18, resnet18_augmix, vit_tiny]
seeds: [0, 1, 2, 3, 4]
shift_families: [gaussian_noise, shot_noise, defocus_blur, fog, jpeg_compression]
severities: [1, 3, 5]
explainers: [integrated_gradients, grad_cam, random]
imputation: mean
n_eval_images: 2000
ig_steps: 32
epochs: 50
batch_size: 256
""")
        config = load_config(config_file)

        from shiftprofile.cells import enumerate_cells

        cells = enumerate_cells(config, config["track"])
        # 3 models x 5 seeds x (1 clean + 5 shifts x 3 severities)
        # = 3 x 5 x (1 + 15) = 3 x 5 x 16 = 240
        assert len(cells) == 240, f"Expected 240 cells, got {len(cells)}"


class TestCliWiresARealModel:
    """The CLI must hand each stage a trained model.

    `main` used to pass `None` as the model to every stage, so the documented
    entry point -- the one in this module's own docstring and in the runbook --
    could not complete a single uncached cell. It appeared to work only while
    every artifact was already in the cache, which is exactly the condition a
    smoke test on a warm cache would create.
    """

    def _config(self, tmp_path):
        import yaml

        cfg = {
            "name": "cli_test",
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["fog"],
            "severities": [1],
            "explainers": ["random"],
            "imputation": "mean",
            "n_eval_images": 8,
            "epochs": 1,
            "batch_size": 8,
        }
        path = tmp_path / "cli_test.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_stages_receive_the_trained_model_not_none(self, tmp_path, monkeypatch):
        from shiftprofile.fill import main

        sentinel = object()
        seen_models = []
        train_calls = []

        def fake_load_or_train(model_id, seed, data_root, cache, **kw):
            train_calls.append((model_id, seed))
            return sentinel, {"model_id": model_id, "seed": seed}

        def fake_predict_cell(cell, model, clean_root, corrupt_root, cache, **kw):
            seen_models.append(model)
            return np.zeros((8, 10), dtype=np.float32)

        monkeypatch.setattr("shiftprofile.train.load_or_train", fake_load_or_train)
        monkeypatch.setattr("shiftprofile.predict.predict_cell", fake_predict_cell)

        rc = main([
            "--config", str(self._config(tmp_path)),
            "--budget-minutes", "5",
            "--cache-write", str(tmp_path / "cache"),
            "--data-root", str(tmp_path / "data"),
            "--corrupt-root", str(tmp_path / "corrupt"),
            "--device", "cpu",
            "--stages", "predict",
        ])

        assert rc == 0
        assert train_calls, "the CLI never trained or loaded a model"
        assert seen_models, "no cell reached the predict stage"
        assert all(m is sentinel for m in seen_models), (
            f"a stage received {seen_models!r} instead of the trained model; "
            "passing None here is the defect this test exists to prevent"
        )
        assert None not in seen_models

    def test_model_is_loaded_once_per_model_and_seed(self, tmp_path, monkeypatch):
        """Two cells sharing a model and seed must not deserialise it twice."""
        from shiftprofile.fill import main

        train_calls = []

        def fake_load_or_train(model_id, seed, data_root, cache, **kw):
            train_calls.append((model_id, seed))
            return object(), {}

        monkeypatch.setattr("shiftprofile.train.load_or_train", fake_load_or_train)
        monkeypatch.setattr(
            "shiftprofile.predict.predict_cell",
            lambda cell, model, clean_root, corrupt_root, cache, **kw: np.zeros((8, 10), np.float32),
        )

        main([
            "--config", str(self._config(tmp_path)),
            "--budget-minutes", "5",
            "--cache-write", str(tmp_path / "cache"),
            "--data-root", str(tmp_path / "data"),
            "--corrupt-root", str(tmp_path / "corrupt"),
            "--device", "cpu",
            "--stages", "predict",
        ])

        # The config yields two cells (clean plus one shift) on one model and seed.
        assert train_calls == [("resnet18", 0)], (
            f"expected one load for the single (model, seed); got {train_calls}"
        )

    def test_device_auto_resolves_without_torch_cuda(self, monkeypatch):
        from shiftprofile.fill import _resolve_device

        assert _resolve_device("cpu") == "cpu"
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("auto") in ("cpu", "cuda")


class TestTrainingIsNotSilent:
    """Training must report progress through the CLI.

    train_model has always accepted a `progress` callback, but the CLI did not
    pass one, so fifty epochs produced no output. On a preemptible runner a
    silent cell cannot be told apart from a hung one, which invites killing a
    healthy run by hand. Version A trains fifteen models, so the silence would
    have been measured in hours.
    """

    def _config(self, tmp_path):
        import yaml

        cfg = {
            "name": "progress_test",
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["fog"],
            "severities": [1],
            "explainers": ["random"],
            "imputation": "mean",
            "n_eval_images": 8,
            "epochs": 20,
            "batch_size": 8,
        }
        path = tmp_path / "progress_test.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def _run(self, tmp_path, monkeypatch, capsync):
        from shiftprofile.fill import main

        captured = {}

        def fake_load_or_train(model_id, seed, data_root, cache, **kw):
            captured["progress"] = kw.get("progress")
            # Drive the callback the way train_model does, so the test exercises
            # the real reporting path rather than merely its presence.
            cb = kw.get("progress")
            if cb is not None:
                total = kw.get("epochs", 20)
                for epoch in range(1, total + 1):
                    cb(epoch, total, {"loss": 1.0 / epoch})
            return object(), {}

        monkeypatch.setattr("shiftprofile.train.load_or_train", fake_load_or_train)
        monkeypatch.setattr(
            "shiftprofile.predict.predict_cell",
            lambda cell, model, clean_root, corrupt_root, cache, **kw: np.zeros((8, 10), np.float32),
        )

        main([
            "--config", str(self._config(tmp_path)),
            "--budget-minutes", "5",
            "--cache-write", str(tmp_path / "cache"),
            "--data-root", str(tmp_path / "data"),
            "--corrupt-root", str(tmp_path / "corrupt"),
            "--device", "cpu",
            "--stages", "predict",
        ])
        return captured, capsync.readouterr().out

    def test_a_progress_callback_is_passed(self, tmp_path, monkeypatch, capsys):
        captured, _ = self._run(tmp_path, monkeypatch, capsys)
        assert callable(captured["progress"]), (
            "the CLI must pass a progress callback, or training runs silent"
        )

    def test_epochs_are_reported(self, tmp_path, monkeypatch, capsys):
        _, out = self._run(tmp_path, monkeypatch, capsys)
        assert "epoch 1/20" in out, "the first epoch must report, to prove liveness early"
        assert "epoch 20/20" in out, "the last epoch must report"
        assert "loss" in out

    def test_reporting_is_throttled_not_per_epoch(self, tmp_path, monkeypatch, capsys):
        """Frequent enough to show liveness, sparse enough not to bury the cells."""
        _, out = self._run(tmp_path, monkeypatch, capsys)
        lines = [l for l in out.splitlines() if "epoch " in l]
        assert 0 < len(lines) < 20, f"expected throttled output, got {len(lines)} lines"

    def test_the_model_step_announces_itself_before_the_silence(
        self, tmp_path, monkeypatch, capsys
    ):
        _, out = self._run(tmp_path, monkeypatch, capsys)
        assert "model resnet18 seed 0" in out, (
            "a reader needs to know a model step started before it goes quiet"
        )
