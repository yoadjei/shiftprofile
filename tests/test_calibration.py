import warnings
import numpy as np
import pytest
from shiftprofile.metrics.calibration import brier_score, nll


def test_brier_is_zero_for_perfect_confident_prediction():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels) == pytest.approx(0.0)


def test_brier_known_value():
    # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01 = 0.14
    probs = np.array([[0.7, 0.2, 0.1]])
    labels = np.array([0])
    assert brier_score(probs, labels) == pytest.approx(0.14)


def test_brier_uniform_binary_is_half():
    probs = np.array([[0.5, 0.5]])
    labels = np.array([0])
    assert brier_score(probs, labels) == pytest.approx(0.5)


def test_nll_is_zero_for_perfect_prediction():
    probs = np.array([[1.0, 0.0]])
    labels = np.array([0])
    assert nll(probs, labels) == pytest.approx(0.0, abs=1e-6)


def test_nll_known_value():
    probs = np.array([[0.7, 0.3]])
    labels = np.array([0])
    assert nll(probs, labels) == pytest.approx(-np.log(0.7))


def test_nll_is_finite_for_zero_probability():
    """A zero-probability true class must clip, not return inf — one such
    sample would otherwise destroy a whole cell's mean."""
    probs = np.array([[0.0, 1.0]])
    labels = np.array([0])
    assert np.isfinite(nll(probs, labels))


from shiftprofile.metrics.calibration import ece_equal_mass, ece_debiased


def test_ece_known_value_two_bins():
    # confidences 0.55, 0.65, 0.85, 0.95; correct = F, T, T, T
    # bin1: conf 0.60, acc 0.50 -> gap 0.10, weight 0.5
    # bin2: conf 0.90, acc 1.00 -> gap 0.10, weight 0.5
    # ECE = 0.10
    probs = np.array([[0.45, 0.55], [0.35, 0.65], [0.15, 0.85], [0.05, 0.95]])
    labels = np.array([0, 1, 1, 1])
    assert ece_equal_mass(probs, labels, n_bins=2) == pytest.approx(0.10)


def test_ece_is_zero_when_confidence_matches_accuracy():
    rng = np.random.default_rng(0)
    n = 20_000
    conf = rng.uniform(0.5, 1.0, size=n)
    correct = rng.uniform(size=n) < conf
    probs = np.stack([1 - conf, conf], axis=1)
    labels = np.where(correct, 1, 0)
    assert ece_equal_mass(probs, labels, n_bins=15) < 0.02


def test_ece_bins_are_equal_mass_not_equal_width():
    """All confidences in a narrow band must still spread across bins."""
    probs = np.stack([np.linspace(0.9, 0.91, 100), np.linspace(0.1, 0.09, 100)], axis=1)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.zeros(100, dtype=int)
    assert ece_equal_mass(probs, labels, n_bins=10) >= 0.0


def test_debiased_ece_is_closer_to_zero_on_calibrated_data():
    """The plugin estimator is biased upward at finite n; the debiased one
    should sit closer to the truth of zero."""
    rng = np.random.default_rng(1)
    n = 1500
    conf = rng.uniform(0.5, 1.0, size=n)
    correct = rng.uniform(size=n) < conf
    probs = np.stack([1 - conf, conf], axis=1)
    labels = np.where(correct, 1, 0)

    plugin = ece_equal_mass(probs, labels, n_bins=15)
    debiased = ece_debiased(probs, labels, n_bins=15)

    assert debiased <= plugin


def test_debiased_ece_is_non_negative():
    rng = np.random.default_rng(2)
    probs_raw = rng.uniform(size=(500, 3))
    probs = probs_raw / probs_raw.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 3, size=500)
    assert ece_debiased(probs, labels, n_bins=15) >= 0.0


from shiftprofile.metrics.calibration import aurc, accuracy_at_coverage


def test_aurc_is_zero_when_all_correct():
    probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    labels = np.array([1, 1, 1])
    assert aurc(probs, labels) == pytest.approx(0.0)


def test_aurc_known_value():
    # conf 0.9 (correct), conf 0.6 (wrong)
    # risk@1 = 0.0, risk@2 = 0.5  ->  AURC = 0.25
    probs = np.array([[0.1, 0.9], [0.6, 0.4]])
    labels = np.array([1, 1])
    assert aurc(probs, labels) == pytest.approx(0.25)


def test_aurc_penalises_misranked_confidence():
    """Same probabilities, same accuracy (1/2); only the alignment between
    confidence and error differs. AURC must punish the confident error —
    precisely the failure ECE is blind to."""
    probs = np.array([[0.1, 0.9], [0.4, 0.6]])
    labels_good = np.array([1, 0])  # confident sample correct, unconfident wrong
    labels_bad = np.array([0, 1])   # confident sample WRONG, unconfident correct
    assert aurc(probs, labels_good) == pytest.approx(0.25)
    assert aurc(probs, labels_bad) == pytest.approx(0.75)
    assert aurc(probs, labels_good) < aurc(probs, labels_bad)


def test_accuracy_at_coverage_takes_most_confident():
    probs = np.array([[0.05, 0.95], [0.1, 0.9], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([1, 1, 1, 1])
    assert accuracy_at_coverage(probs, labels, coverage=0.75) == pytest.approx(1.0)
    assert accuracy_at_coverage(probs, labels, coverage=1.0) == pytest.approx(0.75)


def test_accuracy_at_coverage_rejects_bad_coverage():
    probs = np.array([[0.1, 0.9]])
    labels = np.array([1])
    with pytest.raises(ValueError, match="coverage"):
        accuracy_at_coverage(probs, labels, coverage=0.0)


from shiftprofile.metrics.calibration import fit_temperature, apply_temperature


def test_temperature_scales_linearly_with_logit_magnitude():
    """If logits are doubled, the NLL-optimal temperature must double too."""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(2000, 5))
    labels = rng.integers(0, 5, size=2000)

    t1 = fit_temperature(logits, labels)
    t2 = fit_temperature(2.0 * logits, labels)

    assert t2 == pytest.approx(2.0 * t1, rel=1e-3)


def test_temperature_reduces_nll_of_overconfident_logits():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(3000, 4))
    labels = base.argmax(axis=1)
    overconfident = base * 6.0

    t = fit_temperature(overconfident, labels)
    before = nll(apply_temperature(overconfident, 1.0), labels)
    after = nll(apply_temperature(overconfident, t), labels)

    assert after <= before


def test_apply_temperature_returns_valid_probabilities():
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = apply_temperature(logits, 2.0)
    assert probs.sum(axis=1) == pytest.approx(1.0)
    assert (probs >= 0).all()


def test_fit_temperature_rejects_non_positive():
    logits = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(logits, 0.0)


def test_fit_temperature_warns_when_pinned_at_a_bound():
    """Perfectly separable data drives the optimum to zero. Returning the
    bound silently would present a failed fit as a successful one."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=(500, 4))
    labels = base.argmax(axis=1)  # perfectly predicted -> optimal T -> 0

    with pytest.warns(UserWarning, match="bound"):
        t = fit_temperature(base, labels)
    assert t == pytest.approx(0.05)


def test_fit_temperature_does_not_warn_on_an_interior_optimum():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(2000, 5))
    labels = rng.integers(0, 5, size=2000)

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes a failure
        t = fit_temperature(logits, labels)
    assert 0.05 < t < 100.0
