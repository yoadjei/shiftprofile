import numpy as np
import pytest
from shiftprofile.registry import METRICS
import shiftprofile.metrics  # noqa: F401  (registers on import)


def test_expected_metrics_are_registered():
    expected = {
        "accuracy",
        "accuracy_at_coverage",
        "aurc",
        "brier",
        "ece",
        "ece_debiased",
        "nll",
    }
    assert expected <= set(METRICS.names())


def test_registered_metric_is_callable_by_name():
    probs = np.array([[0.7, 0.3]])
    labels = np.array([0])
    assert METRICS.get("brier")(probs, labels) == pytest.approx(0.18)


def test_registry_rejects_unknown_metric_name():
    with pytest.raises(KeyError, match="unknown metric"):
        METRICS.get("not_a_metric")


def test_importing_twice_does_not_double_register():
    """Duplicate registration raises by design, so a re-import must be safe."""
    import importlib
    import shiftprofile.metrics as m
    importlib.reload(m)
    assert METRICS.get("brier") is not None
