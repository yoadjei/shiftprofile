import numpy as np
import pytest
from shiftprofile.metrics.stability import attribution_stability


def test_identical_attributions_are_perfectly_stable():
    clean = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
    result = attribution_stability(
        clean, clean, np.array([0, 1]), np.array([0, 1]), top_k=2
    )
    assert result.spearman == pytest.approx(1.0)
    assert result.top_k_overlap == pytest.approx(1.0)
    assert result.n_conditioned == 2


def test_reversed_attributions_are_anticorrelated():
    clean = np.array([[0.1, 0.5, 0.9]])
    shifted = np.array([[0.9, 0.5, 0.1]])
    result = attribution_stability(
        clean, shifted, np.array([0]), np.array([0]), top_k=1
    )
    assert result.spearman == pytest.approx(-1.0)


def test_only_unchanged_predictions_are_counted():
    """A stability drop that merely reflects a changed prediction is not an
    explanation result. Samples whose prediction flipped are excluded."""
    clean = np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
    shifted = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    clean_preds = np.array([0, 0, 0])
    shifted_preds = np.array([0, 1, 0])  # middle sample flipped

    result = attribution_stability(
        clean, shifted, clean_preds, shifted_preds, top_k=1
    )

    assert result.n_conditioned == 2
    assert result.n_total == 3
    assert result.spearman == pytest.approx(1.0)


def test_no_unchanged_predictions_returns_nan_not_crash():
    """Under severe shift every prediction can flip. That is a reportable
    result, not an exception."""
    clean = np.array([[0.1, 0.9]])
    shifted = np.array([[0.9, 0.1]])
    result = attribution_stability(
        clean, shifted, np.array([0]), np.array([1]), top_k=1
    )
    assert np.isnan(result.spearman)
    assert result.n_conditioned == 0


def test_top_k_overlap_is_a_fraction():
    clean = np.array([[0.9, 0.8, 0.1, 0.0]])
    shifted = np.array([[0.9, 0.1, 0.8, 0.0]])
    result = attribution_stability(
        clean, shifted, np.array([0]), np.array([0]), top_k=2
    )
    # clean top-2 = {0, 1}; shifted top-2 = {0, 2}; overlap = 1/2
    assert result.top_k_overlap == pytest.approx(0.5)


def test_top_k_must_not_exceed_feature_count():
    clean = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="top_k"):
        attribution_stability(
            clean, clean, np.array([0]), np.array([0]), top_k=5
        )


def test_mismatched_attribution_shapes_are_rejected():
    with pytest.raises(ValueError, match="shapes differ"):
        attribution_stability(
            np.zeros((2, 3)), np.zeros((2, 4)),
            np.array([0, 0]), np.array([0, 0]), top_k=1
        )
