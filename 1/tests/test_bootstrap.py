import numpy as np
import pytest
from shiftprofile.stats.bootstrap import bootstrap_interval, paired_bootstrap_difference


def test_constant_data_gives_degenerate_interval():
    values = np.full(100, 3.0)
    result = bootstrap_interval(values, np.mean, n_resamples=200, seed=0)
    assert result.point == pytest.approx(3.0)
    assert result.low == pytest.approx(3.0)
    assert result.high == pytest.approx(3.0)


def test_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=5.0, scale=1.0, size=500)
    result = bootstrap_interval(values, np.mean, n_resamples=500, seed=1)
    assert result.low < result.point < result.high


def test_same_seed_gives_identical_result():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    a = bootstrap_interval(values, np.mean, n_resamples=200, seed=7)
    b = bootstrap_interval(values, np.mean, n_resamples=200, seed=7)
    assert (a.point, a.low, a.high) == (b.point, b.low, b.high)


def test_different_seed_gives_different_interval():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    a = bootstrap_interval(values, np.mean, n_resamples=200, seed=7)
    b = bootstrap_interval(values, np.mean, n_resamples=200, seed=8)
    assert (a.low, a.high) != (b.low, b.high)


def test_wider_interval_for_noisier_data():
    """Interval width scales with sigma. Data differs only in scale (0.1 vs
    5.0, same n, same bootstrap seed), so the ratio should be roughly 50x;
    asserting >10x is robust while still being a real claim rather than a
    bare inequality that a degenerate case could satisfy."""
    rng = np.random.default_rng(2)
    tight = bootstrap_interval(
        rng.normal(scale=0.1, size=300), np.mean, n_resamples=400, seed=3
    )
    loose = bootstrap_interval(
        rng.normal(scale=5.0, size=300), np.mean, n_resamples=400, seed=3
    )
    tight_width = tight.high - tight.low
    loose_width = loose.high - loose.low
    assert loose_width > 10 * tight_width


def test_paired_bootstrap_resamples_the_same_indices():
    """Paired comparison on identical arrays must give a degenerate interval
    at zero — if the pairing were broken it would have spread."""
    rng = np.random.default_rng(4)
    values = rng.normal(size=300)
    result = paired_bootstrap_difference(values, values, n_resamples=300, seed=5)
    assert result.point == pytest.approx(0.0)
    assert result.low == pytest.approx(0.0)
    assert result.high == pytest.approx(0.0)


def test_paired_bootstrap_detects_a_constant_shift():
    rng = np.random.default_rng(6)
    a = rng.normal(size=500)
    b = a + 2.0
    result = paired_bootstrap_difference(b, a, n_resamples=400, seed=7)
    assert result.point == pytest.approx(2.0)
    assert result.low == pytest.approx(2.0)
    assert result.high == pytest.approx(2.0)


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_difference(np.zeros(5), np.zeros(6), n_resamples=10, seed=0)


def test_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_interval(np.array([]), np.mean, n_resamples=10, seed=0)


def test_half_width_helper():
    result = bootstrap_interval(np.arange(100.0), np.mean, n_resamples=200, seed=0)
    assert result.half_width == pytest.approx((result.high - result.low) / 2)
