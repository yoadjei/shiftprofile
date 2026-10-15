import numpy as np
import pytest
from shiftprofile.metrics.faithfulness import (
    removal_auc,
    relative_faithfulness,
    rank_features,
)


def test_removal_auc_of_flat_curve_is_one():
    fractions = np.array([0.0, 0.5, 1.0])
    curve = np.array([1.0, 1.0, 1.0])
    assert removal_auc(curve, fractions) == pytest.approx(1.0)


def test_removal_auc_of_linear_decline_is_half():
    fractions = np.array([0.0, 0.5, 1.0])
    curve = np.array([1.0, 0.5, 0.0])
    assert removal_auc(curve, fractions) == pytest.approx(0.5)


def test_removal_auc_rewards_fast_collapse():
    """Same fractions, same endpoints; only the middle differs. Absolute
    anchors are asserted too, so a degenerate case where both sides are
    equal cannot hide behind the inequality."""
    fractions = np.array([0.0, 0.5, 1.0])
    fast = np.array([1.0, 0.1, 0.0])
    slow = np.array([1.0, 0.9, 0.0])
    assert removal_auc(fast, fractions) == pytest.approx(0.30)
    assert removal_auc(slow, fractions) == pytest.approx(0.70)
    assert removal_auc(fast, fractions) < removal_auc(slow, fractions)


def test_removal_auc_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        removal_auc(np.array([1.0, 0.5]), np.array([0.0, 0.5, 1.0]))


def test_removal_auc_rejects_unsorted_fractions():
    with pytest.raises(ValueError, match="ascending"):
        removal_auc(np.array([1.0, 0.5, 0.0]), np.array([0.0, 1.0, 0.5]))


def test_relative_faithfulness_is_positive_when_better_than_random():
    fractions = np.array([0.0, 0.5, 1.0])
    model_curve = np.array([1.0, 0.2, 0.0])   # AUC 0.35
    random_curve = np.array([1.0, 0.8, 0.0])  # AUC 0.65
    assert relative_faithfulness(
        model_curve, random_curve, fractions, imputation="mean"
    ) == pytest.approx(0.30)


def test_relative_faithfulness_is_zero_against_itself():
    fractions = np.array([0.0, 0.5, 1.0])
    curve = np.array([1.0, 0.4, 0.0])
    assert relative_faithfulness(
        curve, curve, fractions, imputation="mean"
    ) == pytest.approx(0.0)


def test_relative_faithfulness_refuses_missing_random_control():
    """Masking a corrupted image is doubly off-distribution, so raw removal
    AUC is not comparable across severities. The metric must be unable to
    report a number without its paired control."""
    fractions = np.array([0.0, 0.5, 1.0])
    curve = np.array([1.0, 0.4, 0.0])
    with pytest.raises(ValueError, match="random-attribution control"):
        relative_faithfulness(curve, None, fractions, imputation="mean")


def test_relative_faithfulness_requires_explicit_imputation():
    """The imputation scheme separates real degradation from an off-manifold
    artefact, so it must never have a default."""
    fractions = np.array([0.0, 0.5, 1.0])
    curve = np.array([1.0, 0.4, 0.0])
    with pytest.raises(TypeError):
        relative_faithfulness(curve, curve, fractions)


def test_relative_faithfulness_rejects_unknown_imputation():
    fractions = np.array([0.0, 0.5, 1.0])
    curve = np.array([1.0, 0.4, 0.0])
    with pytest.raises(ValueError, match="imputation"):
        relative_faithfulness(curve, curve, fractions, imputation="magic")


def test_rank_features_orders_by_absolute_attribution():
    attribution = np.array([0.1, -0.9, 0.5, 0.0])
    np.testing.assert_array_equal(rank_features(attribution), np.array([1, 2, 0, 3]))


def test_rank_features_is_deterministic_under_ties():
    attribution = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_equal(rank_features(attribution), np.array([0, 1, 2]))
