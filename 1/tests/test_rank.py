import numpy as np
import pytest
from shiftprofile.stats.rank import kendall_tau, bootstrap_rank_intervals


def test_identical_rankings_give_tau_one():
    assert kendall_tau([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_reversed_rankings_give_tau_minus_one():
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_tau_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        kendall_tau([1, 2, 3], [1, 2])


def test_rank_intervals_are_tight_for_well_separated_models():
    rng = np.random.default_rng(0)
    scores = {
        "model_a": rng.normal(10.0, 0.01, size=50),
        "model_b": rng.normal(5.0, 0.01, size=50),
        "model_c": rng.normal(1.0, 0.01, size=50),
    }
    result = bootstrap_rank_intervals(scores, n_resamples=200, seed=0)
    assert result["model_a"] == (1, 1)
    assert result["model_b"] == (2, 2)
    assert result["model_c"] == (3, 3)


def test_rank_intervals_are_wide_for_overlapping_models():
    """Same n, same seed; only the separation differs. Indistinguishable
    models must report an ambiguous rank rather than a confident ordering."""
    rng = np.random.default_rng(1)
    scores = {
        "model_a": rng.normal(5.0, 3.0, size=50),
        "model_b": rng.normal(5.0, 3.0, size=50),
    }
    result = bootstrap_rank_intervals(scores, n_resamples=300, seed=2)
    assert result["model_a"] == (1, 2)
    assert result["model_b"] == (1, 2)


def test_rank_intervals_need_at_least_two_models():
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_rank_intervals({"only": np.zeros(10)}, n_resamples=10, seed=0)


def test_lower_is_better_flips_the_ordering():
    """ECE and AURC are better when lower; the same helper must serve them."""
    rng = np.random.default_rng(3)
    scores = {
        "good": rng.normal(1.0, 0.01, size=50),
        "bad": rng.normal(9.0, 0.01, size=50),
    }
    higher = bootstrap_rank_intervals(scores, n_resamples=100, seed=0)
    lower = bootstrap_rank_intervals(
        scores, n_resamples=100, seed=0, higher_is_better=False
    )
    assert higher["bad"] == (1, 1)
    assert lower["good"] == (1, 1)
