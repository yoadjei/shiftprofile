import numpy as np
import pandas as pd
import pytest
from shiftprofile.stats.mixed import fit_coupling_model, ConvergenceError


def _synthetic_cells(n_seeds=6, n_cells=40, accuracy_coef=0.5, extra=0.0, noise=0.01):
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(n_seeds):
        seed_offset = rng.normal(0, 0.05)
        for i in range(n_cells):
            acc_change = rng.uniform(-0.4, 0.0)
            metric_change = (
                accuracy_coef * acc_change
                + extra * (i / n_cells)
                + seed_offset
                + rng.normal(0, noise)
            )
            rows.append(
                {
                    "seed": seed,
                    "model": "resnet18" if i % 2 else "vit_tiny",
                    "shift_family": ["noise", "blur", "weather", "digital"][i % 4],
                    "accuracy_change": acc_change,
                    "metric_change": metric_change,
                }
            )
    return pd.DataFrame(rows)


def test_recovers_known_accuracy_coefficient():
    df = _synthetic_cells(accuracy_coef=0.5)
    result = fit_coupling_model(df, metric_column="metric_change")
    assert result.accuracy_coefficient == pytest.approx(0.5, abs=0.05)


def test_confidence_interval_brackets_the_true_coefficient():
    df = _synthetic_cells(accuracy_coef=0.5)
    result = fit_coupling_model(df, metric_column="metric_change")
    low, high = result.accuracy_ci
    assert low < 0.5 < high


def test_residual_variance_is_near_zero_when_accuracy_explains_everything():
    df = _synthetic_cells(accuracy_coef=0.5, extra=0.0, noise=0.001)
    result = fit_coupling_model(df, metric_column="metric_change")
    assert result.residual_std < 0.02


def test_residual_is_larger_when_a_second_factor_drives_the_metric():
    """Identical generator, identical seed; only `extra` differs. The residual
    after accuracy is the study's actual result, so this is the property that
    matters most. Absolute anchors are asserted, not just the inequality."""
    explained = fit_coupling_model(
        _synthetic_cells(accuracy_coef=0.5, extra=0.0), metric_column="metric_change"
    )
    unexplained = fit_coupling_model(
        _synthetic_cells(accuracy_coef=0.5, extra=1.0), metric_column="metric_change"
    )
    assert explained.residual_std < 0.05
    assert unexplained.residual_std > 0.15
    assert unexplained.residual_std > 5 * explained.residual_std


def test_reports_seed_variance_component():
    result = fit_coupling_model(_synthetic_cells(), metric_column="metric_change")
    assert result.seed_variance >= 0.0


def test_reports_cell_count():
    df = _synthetic_cells(n_seeds=6, n_cells=40)
    result = fit_coupling_model(df, metric_column="metric_change")
    assert result.n_cells == 240


def test_missing_column_raises_clearly():
    df = _synthetic_cells().drop(columns=["accuracy_change"])
    with pytest.raises(ValueError, match="accuracy_change"):
        fit_coupling_model(df, metric_column="metric_change")


def test_single_seed_raises_rather_than_silently_dropping_random_effect():
    """One seed means no seed variance to estimate. Pseudo-replication —
    treating correlated cells as independent — is the failure mode this
    guards against."""
    df = _synthetic_cells(n_seeds=1)
    with pytest.raises(ValueError, match="at least two seeds"):
        fit_coupling_model(df, metric_column="metric_change")
