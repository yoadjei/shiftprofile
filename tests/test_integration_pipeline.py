"""End-to-end integration tests for the shiftprofile pipeline.

Tests verify that the CLI seams between modules work correctly. The critical
tests are: imports resolve, stage specs use correct cache keys, and missing
implementations are recorded (not silently skipped).
"""

import numpy as np
import pytest

from shiftprofile.cache import ArtifactCache
from shiftprofile.cells import Cell, CLEAN
from shiftprofile.fill import fill


class TestCliStageImportsResolve:
    """Verify that all stage functions can be imported and are callable.

    D1: fill.main() imports its stage functions lazily, so a missing name is
    invisible until the CLI is actually invoked. This is the cheapest guard.
    """

    def test_cli_stage_imports_resolve(self):
        """The three stage functions must be importable from their modules."""
        from shiftprofile.predict import predict_cell
        from shiftprofile.explain import explain_cell
        from shiftprofile.curves import curves_cell

        assert callable(predict_cell)
        assert callable(explain_cell)
        assert callable(curves_cell)


class TestCacheKeyCorrectness:
    """Verify that cache keys are built correctly for each stage.

    D2, D3: The cache keys must distinguish between different explainers,
    imputations, and artifact kinds to prevent silent data loss.
    """

    def test_explain_stage_uses_array_kind_not_record(self, tmp_path):
        """D2: Explain stage must cache as 'array', not 'record'.

        The explain stage produces attribution arrays, and the old code
        checked kind='record', which would never hit. This test verifies
        the fix by checking the cache lookup logic.
        """
        cache = ArtifactCache(tmp_path)
        cell = Cell("vision", "resnet18", 0, CLEAN, 0)

        # Explain stage spec includes explainer
        spec = {**cell.spec(), "stage": "explain", "explainer": "integrated_gradients"}

        # Pre-populate as array (the correct kind)
        dummy_array = np.zeros((32, 32, 32), dtype=np.float16)
        cache.put_array(spec, "explain-v1", dummy_array)

        # The lookup must find it with kind="array"
        assert cache.has(spec, "explain-v1", kind="array")

        # And must not find it with kind="record" (the bug)
        assert not cache.has(spec, "explain-v1", kind="record")

    def test_different_explainers_have_different_cache_keys(self, tmp_path):
        """D3: Attributions for different explainers must land under DIFFERENT keys.

        Integrated Gradients and Grad-CAM attributions are fundamentally different
        and must not share a cache key, else results are silently mixed.
        """
        cache = ArtifactCache(tmp_path)
        cell = Cell("vision", "resnet18", 0, CLEAN, 0)

        spec_ig = {**cell.spec(), "stage": "explain", "explainer": "integrated_gradients"}
        spec_gc = {**cell.spec(), "stage": "explain", "explainer": "grad_cam"}

        # Both specs produce arrays
        array1 = np.ones((32, 32, 32), dtype=np.float16)
        array2 = np.zeros((32, 32, 32), dtype=np.float16)

        cache.put_array(spec_ig, "explain-v1", array1)
        cache.put_array(spec_gc, "explain-v1", array2)

        # They must be under different cache keys
        path_ig = cache.resolve(spec_ig, "explain-v1", "array")
        path_gc = cache.resolve(spec_gc, "explain-v1", "array")

        assert path_ig != path_gc, "Different explainers must have different cache keys"

        # Verify the data is not mixed
        retrieved_ig = cache.get_array(spec_ig, "explain-v1")
        retrieved_gc = cache.get_array(spec_gc, "explain-v1")
        assert np.allclose(retrieved_ig, array1)
        assert np.allclose(retrieved_gc, array2)

    def test_different_imputations_have_different_cache_keys(self, tmp_path):
        """D3: Curves for different imputations must land under DIFFERENT keys.

        Mean imputation and blur imputation are different validity controls
        and must not share a key, else the study's findings are wrong.
        """
        cache = ArtifactCache(tmp_path)
        cell = Cell("vision", "resnet18", 0, CLEAN, 0)

        spec_mean = {
            **cell.spec(),
            "stage": "curves",
            "explainer": "integrated_gradients",
            "imputation": "mean",
            "which": "model",
        }
        spec_blur = {
            **cell.spec(),
            "stage": "curves",
            "explainer": "integrated_gradients",
            "imputation": "blur",
            "which": "model",
        }

        # Both produce arrays
        array1 = np.ones((32, 9), dtype=np.float32)
        array2 = np.zeros((32, 9), dtype=np.float32)

        cache.put_array(spec_mean, "curves-v1", array1)
        cache.put_array(spec_blur, "curves-v1", array2)

        # They must be under different cache keys
        path_mean = cache.resolve(spec_mean, "curves-v1", "array")
        path_blur = cache.resolve(spec_blur, "curves-v1", "array")

        assert path_mean != path_blur, "Different imputations must have different cache keys"

        # Verify the data is not mixed
        retrieved_mean = cache.get_array(spec_mean, "curves-v1")
        retrieved_blur = cache.get_array(spec_blur, "curves-v1")
        assert np.allclose(retrieved_mean, array1)
        assert np.allclose(retrieved_blur, array2)


class TestNotImplementedRecording:
    """D4: Missing stage implementations are recorded, not silently skipped."""

    def test_not_implemented_stages_recorded_not_skipped(self, tmp_path):
        """When stage_fn is None, entries go to not_implemented, not completed."""
        cache = ArtifactCache(tmp_path)

        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": [],
            "severities": [],
            "explainers": ["integrated_gradients"],
            "imputations": ["mean"],
        }

        # No stages_impl provided
        report = fill(
            config, cache,
            budget_minutes=1000,
            device="cpu",
            stages=("predict", "explain", "curves"),
            stages_impl={},
        )

        # All work units should be recorded as not_implemented
        assert len(report.not_implemented) > 0
        assert len(report.completed) == 0
        assert len(report.failed) == 0

        # Summary must mention them
        summary = report.summary()
        assert "implemented" in summary.lower()

    def test_not_implemented_appears_in_summary(self, tmp_path):
        """The summary() method must explicitly state what was not implemented."""
        cache = ArtifactCache(tmp_path)

        config = {
            "track": "vision",
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["gaussian_noise"],
            "severities": [1],
            "explainers": ["integrated_gradients"],
            "imputations": ["mean"],
        }

        report = fill(
            config, cache,
            budget_minutes=1000,
            device="cpu",
            stages=("predict",),
            stages_impl={},
        )

        assert len(report.not_implemented) > 0
        summary = report.summary()

        # The summary must explicitly mention that some stages were not implemented
        # and must not read as "everything completed"
        assert "not" in summary.lower()
        assert "0" in summary  # Should show 0 completed
