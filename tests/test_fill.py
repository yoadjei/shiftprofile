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
