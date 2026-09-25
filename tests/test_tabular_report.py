"""Tests for tabular report evaluation module.

Tests are comprehensive and deterministic. No test downloads data or accesses the network.
All calls to metrics and data loaders are mocked with synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from shiftprofile.tabular_report import GapRecord, evaluate_tabular_cell
from shiftprofile.cells import Cell
from shiftprofile.cache import ArtifactCache
from shiftprofile.data.acs import TabularSplit
from shiftprofile.metrics.subgroup import MIN_GROUP_SIZE


# ============================================================================
# Test GapRecord dataclass
# ============================================================================


def test_gap_record_frozen():
    """GapRecord should be frozen to prevent accidental modification."""
    rec = GapRecord(
        name="accuracy",
        value=0.95,
        interval=(0.93, 0.97),
        group_sizes={0: 100, 1: 150},
        n_total=250,
        reported=True,
        reason="",
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        rec.value = 0.90


def test_gap_record_reported_true():
    """Test creating a reported GapRecord."""
    rec = GapRecord(
        name="accuracy",
        value=0.95,
        interval=(0.93, 0.97),
        group_sizes={0: 100, 1: 150},
        n_total=250,
        reported=True,
        reason="",
    )
    assert rec.name == "accuracy"
    assert rec.value == 0.95
    assert rec.interval == (0.93, 0.97)
    assert rec.group_sizes == {0: 100, 1: 150}
    assert rec.n_total == 250
    assert rec.reported is True
    assert rec.reason == ""


def test_gap_record_reported_false_with_reason():
    """Test creating an unreported GapRecord with a reason."""
    rec = GapRecord(
        name="worst_group_accuracy",
        value=0.85,
        interval=(0.80, 0.90),
        group_sizes={0: 150, 1: 100},
        n_total=250,
        reported=False,
        reason="group 1 has 100 rows, below MIN_GROUP_SIZE (200)",
    )
    assert rec.reported is False
    assert "MIN_GROUP_SIZE" in rec.reason
    assert len(rec.reason) > 0


def test_gap_record_interval_is_tuple():
    """Interval must be stored as a tuple, not unpacked."""
    rec = GapRecord(
        name="brier",
        value=0.15,
        interval=(0.12, 0.18),
        group_sizes={0: 200},
        n_total=200,
        reported=True,
        reason="",
    )
    assert isinstance(rec.interval, tuple)
    assert len(rec.interval) == 2


# ============================================================================
# Test evaluate_tabular_cell: happy path with all metrics reported
# ============================================================================


def test_evaluate_tabular_cell_returns_list_of_gap_records():
    """evaluate_tabular_cell should return a list of GapRecord instances."""
    # Create synthetic data
    n_samples = 300
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    groups = rng.integers(0, 3, size=n_samples)

    # Synthetic probabilities
    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 300}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, GapRecord) for r in results)


def test_evaluate_tabular_cell_returns_all_six_metrics():
    """All six metrics must be included in the results."""
    n_samples = 400
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    # Use 4 groups, each large enough (100+ samples each should sum to 400)
    groups = np.array([i % 4 for i in range(n_samples)])
    # Shuffle to avoid groups being in order
    perm = rng.permutation(n_samples)
    groups = groups[perm]

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="xgb", seed=5, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 400}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    metric_names = {r.name for r in results}
    expected_metrics = {
        "accuracy", "brier", "ece_equal_mass",
        "worst_group_accuracy", "equal_opportunity_gap", "per_group_ece_gap"
    }
    assert metric_names == expected_metrics


def test_evaluate_tabular_cell_reported_records_have_valid_intervals():
    """All reported records must have non-degenerate intervals."""
    n_samples = 300
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    groups = np.array([0] * 150 + [1] * 150)

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 300}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    for record in results:
        if record.reported:
            low, high = record.interval
            assert low <= record.value <= high or np.isclose(low, record.value) or np.isclose(high, record.value)
            assert low <= high
            assert not np.isnan(low) and not np.isnan(high)
            assert np.isfinite(low) and np.isfinite(high)


def test_evaluate_tabular_cell_reported_records_have_group_sizes():
    """All reported records must have non-empty group_sizes dict."""
    n_samples = 300
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    groups = np.array([0] * 150 + [1] * 150)

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 300}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    for record in results:
        if record.reported:
            assert isinstance(record.group_sizes, dict)
            assert len(record.group_sizes) > 0
            # For non-group metrics (accuracy, brier, ece), group_sizes might include all groups
            # For group metrics, it should have multiple groups


def test_evaluate_tabular_cell_n_total_matches_sum_of_groups():
    """n_total must equal the sum of all group sizes."""
    n_samples = 300
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    groups = np.array([0] * 150 + [1] * 150)

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 300}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    for record in results:
        if record.reported:
            total_from_groups = sum(record.group_sizes.values())
            assert record.n_total == total_from_groups


# ============================================================================
# Test evaluate_tabular_cell: subgroup below MIN_GROUP_SIZE
# ============================================================================


def test_evaluate_tabular_cell_subgroup_below_threshold_not_reported():
    """A subgroup with 199 rows should be reported=False with a reason."""
    n_samples = 299
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    # Create two groups: one with 200, one with 99 (below threshold)
    groups = np.array([0] * 200 + [1] * 99)

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 299}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    subgroup_metrics = ["worst_group_accuracy", "equal_opportunity_gap", "per_group_ece_gap"]

    # Assert unconditionally. The earlier form of this test only checked
    # `reported is False` inside `if 1 in record.group_sizes and
    # record.group_sizes[1] == 99`, so it passed whenever the records were
    # absent entirely — which is the exact failure the gate is written against.
    # A vacuous assertion is worse than no assertion: it reads as coverage.
    by_name = {r.name: r for r in results}
    for name in subgroup_metrics:
        assert name in by_name, (
            f"{name} missing from the report. An under-powered subgroup must come "
            f"back flagged, never dropped: a vanished group reads as 'no disparity' "
            f"when it means 'we did not look'."
        )
        record = by_name[name]
        assert record.reported is False, (
            f"{name} was reported despite group 1 holding only 99 rows, under the "
            f"{MIN_GROUP_SIZE}-row floor"
        )
        assert "MIN_GROUP_SIZE" in record.reason or "200" in record.reason
        assert record.group_sizes, f"{name} must carry the group sizes behind it"


def test_evaluate_tabular_cell_subgroup_exactly_at_threshold_is_reported():
    """A subgroup with exactly 200 rows should be reported=True."""
    n_samples = 400
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    # Two groups with exactly 200 each
    groups = np.array([0] * 200 + [1] * 200)

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 400}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    subgroup_metrics = ["worst_group_accuracy", "equal_opportunity_gap", "per_group_ece_gap"]

    for record in results:
        if record.name in subgroup_metrics:
            # Should be reported because both groups have 200+ rows
            assert record.reported is True
            assert record.reason == ""


def test_evaluate_tabular_cell_unreported_record_still_in_list():
    """An unreported record must be in the returned list, never silently dropped."""
    n_samples = 299
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    groups = np.array([0] * 200 + [1] * 99)

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 299}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    metric_names = [r.name for r in results]
    # Even if a metric is unreported due to small groups, it should be in the list
    subgroup_metrics = ["worst_group_accuracy", "equal_opportunity_gap", "per_group_ece_gap"]
    for metric in subgroup_metrics:
        assert metric in metric_names


def test_evaluate_tabular_cell_unreported_record_has_reason():
    """An unreported record must have a non-empty reason field."""
    n_samples = 299
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)
    groups = np.array([0] * 200 + [1] * 99)

    proba = np.stack([1 - rng.uniform(0.4, 0.6, n_samples), rng.uniform(0.4, 0.6, n_samples)], axis=1)
    proba = proba / proba.sum(axis=1, keepdims=True)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 299}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    subgroup_metrics = ["worst_group_accuracy", "equal_opportunity_gap", "per_group_ece_gap"]

    for record in results:
        if record.name in subgroup_metrics and not record.reported:
            assert len(record.reason) > 0
            assert isinstance(record.reason, str)


# ============================================================================
# Test metric value agreement: ensure assembly, not reimplementation
# ============================================================================


def test_evaluate_tabular_cell_accuracy_matches_direct_call():
    """Accuracy value should match calling the metric directly."""
    n_samples = 300
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = np.array([0] * 150 + [1] * 150)
    groups = np.array([0] * 150 + [1] * 150)

    # Create probabilities where class 0 is predicted for first 150, class 1 for last 150
    proba = np.zeros((n_samples, 2))
    proba[:150, 0] = 0.8
    proba[:150, 1] = 0.2
    proba[150:, 0] = 0.2
    proba[150:, 1] = 0.8

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 300}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    # Find the accuracy record and verify its value
    accuracy_record = next(r for r in results if r.name == "accuracy")
    # With our perfect predictions, accuracy should be 1.0
    assert accuracy_record.value == pytest.approx(1.0)
    assert accuracy_record.reported is True


def test_evaluate_tabular_cell_brier_matches_direct_call():
    """Brier score should match direct calculation."""
    n_samples = 200
    n_features = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = np.array([0] * 100 + [1] * 100)
    groups = np.array([0] * 100 + [1] * 100)

    # Create specific probabilities for known Brier score
    proba = np.array([[0.6, 0.4]] * 100 + [[0.3, 0.7]] * 100, dtype=np.float64)

    data = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        domain="CA_2018", task="income"
    )

    cell = Cell(track="tabular", model_id="rf", seed=0, shift_family="clean", severity=0)
    config = {"task": "income", "source": "CA_2018", "n_eval": 200}

    with patch("shiftprofile.tabular_report.load_cell_table", return_value=data):
        with patch("shiftprofile.tabular_report.predict_tabular_cell", return_value=proba):
            cache = Mock()
            results = evaluate_tabular_cell(cell, config, cache, root=None)

    # Expected Brier:
    # First 100 (true label 0): (0.6-1)^2 + (0.4-0)^2 = 0.16 + 0.16 = 0.32 each -> sum 32
    # Last 100 (true label 1): (0.3-0)^2 + (0.7-1)^2 = 0.09 + 0.09 = 0.18 each -> sum 18
    # Total: 50, mean = 0.25
    brier_record = next(r for r in results if r.name == "brier")
    assert brier_record.value == pytest.approx(0.25, abs=0.01)
    assert brier_record.reported is True
