import numpy as np
import pytest
from scipy.stats import norm
from shiftprofile.metrics.robustness import (
    accuracy,
    accuracy_drop,
    effective_robustness,
)


def test_accuracy_counts_argmax_matches():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    labels = np.array([0, 1, 1])
    assert accuracy(probs, labels) == pytest.approx(2 / 3)


def test_accuracy_drop_is_clean_minus_shifted():
    assert accuracy_drop(clean_accuracy=0.95, shifted_accuracy=0.70) == pytest.approx(
        0.25
    )


def test_effective_robustness_is_zero_on_the_reference_line():
    # Reference models lie exactly on probit(ood) = 0.8 * probit(id) + 0.2
    z_id = np.array([0.5, 1.0, 1.5])
    z_ood = 0.8 * z_id + 0.2
    ref_id = norm.cdf(z_id)
    ref_ood = norm.cdf(z_ood)

    query_z_id = 2.0
    query_z_ood = 0.8 * query_z_id + 0.2

    er = effective_robustness(
        id_accuracy=float(norm.cdf(query_z_id)),
        ood_accuracy=float(norm.cdf(query_z_ood)),
        reference_id=ref_id,
        reference_ood=ref_ood,
    )
    assert er == pytest.approx(0.0, abs=1e-8)


def test_effective_robustness_is_positive_above_the_line():
    z_id = np.array([0.5, 1.0, 1.5])
    z_ood = 0.8 * z_id + 0.2
    er = effective_robustness(
        id_accuracy=float(norm.cdf(2.0)),
        ood_accuracy=float(norm.cdf(0.8 * 2.0 + 0.2 + 0.3)),
        reference_id=norm.cdf(z_id),
        reference_ood=norm.cdf(z_ood),
    )
    assert er == pytest.approx(0.3, abs=1e-8)


def test_effective_robustness_needs_at_least_two_reference_points():
    with pytest.raises(ValueError, match="at least two"):
        effective_robustness(
            id_accuracy=0.9,
            ood_accuracy=0.8,
            reference_id=np.array([0.85]),
            reference_ood=np.array([0.75]),
        )
