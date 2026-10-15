"""Tests for domain classifier AUC shift distance metric.

The domain_classifier_auc function measures how distinguishable two domains are
by training a logistic regression classifier to predict domain membership. The AUC
must be out-of-fold (computed during cross-validation), and the result includes
a 95% confidence interval across folds.

Key design decisions:
- AUC of 0.5 means indistinguishable; 1.0 means trivially separable
- Subsampling to min(len(source), len(target), max_n) prevents class imbalance
  from inflating the AUC — test this directly by passing very unequal sizes
  from the SAME distribution and asserting AUC stays near 0.5
- Interval is required and computed across folds, not optional
- Raises if either side has fewer than n_splits rows
"""

import numpy as np
import pytest
from shiftprofile.metrics.shift_distance import ShiftDistance, domain_classifier_auc


class TestSameDistribution:
    """When both samples come from the same distribution, the classifier
    cannot tell them apart — AUC should be ~0.5 with a tight band."""

    def test_identical_gaussian_distributions_have_auc_near_0_5(self):
        """Two samples from N(0, I) should have AUC ~0.5."""
        np.random.seed(42)
        X_source = np.random.randn(1000, 10)
        X_target = np.random.randn(1000, 10)

        result = domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

        # Should be indistinguishable (AUC close to 0.5 within a tight band)
        assert 0.40 <= result.auc <= 0.60, f"AUC {result.auc} not in [0.40, 0.60]"
        # Interval should be tight around 0.5
        assert result.interval[0] >= 0.30
        assert result.interval[1] <= 0.70

    def test_same_distribution_unequal_sizes_subsampling_prevents_auc_inflation(self):
        """When sizes are very unequal but distributions identical, subsampling
        should keep AUC near 0.5. This tests the core reason for subsampling."""
        np.random.seed(123)
        # Very unequal: 10000 vs 100 from the same N(0, I)
        X_source = np.random.randn(10000, 10)
        X_target = np.random.randn(100, 10)

        result = domain_classifier_auc(X_source, X_target, seed=123, n_splits=5)

        # Despite the massive size difference, AUC should stay ~0.5 because
        # subsampling to min(10000, 100, 20000) = 100 per side equalizes them
        assert 0.35 <= result.auc <= 0.65, (
            f"AUC {result.auc} not in [0.35, 0.65] despite subsampling "
            "to equal class balance"
        )


class TestClearlySeparatedDistributions:
    """When the distributions are far apart, the classifier easily separates
    them — AUC should be ~1.0."""

    def test_well_separated_distributions_have_auc_near_1_0(self):
        """Source from N(0, I) vs Target from N(5, I) should have AUC ~1.0."""
        np.random.seed(456)
        X_source = np.random.randn(1000, 10)
        X_target = np.random.randn(1000, 10) + 5  # Shifted by 5 std

        result = domain_classifier_auc(X_source, X_target, seed=456, n_splits=5)

        # Should be very distinguishable (AUC close to 1.0)
        assert 0.90 <= result.auc <= 1.0, f"AUC {result.auc} not in [0.90, 1.0]"


class TestDeterminism:
    """The result must be deterministic for a fixed seed and change with
    different seeds."""

    def test_fixed_seed_gives_reproducible_auc(self):
        """Running twice with the same seed should give identical AUC."""
        np.random.seed(789)
        X_source = np.random.randn(500, 8)
        X_target = np.random.randn(500, 8) + 2

        result1 = domain_classifier_auc(X_source, X_target, seed=789, n_splits=5)
        result2 = domain_classifier_auc(X_source, X_target, seed=789, n_splits=5)

        assert result1.auc == result2.auc
        assert result1.interval == result2.interval



class TestTooFewRows:
    """Raise if either side has fewer than n_splits rows."""

    def test_source_too_small_raises(self):
        """If source has fewer than n_splits rows, raise ValueError."""
        X_source = np.random.randn(3, 5)  # Only 3 rows
        X_target = np.random.randn(100, 5)

        with pytest.raises(ValueError, match="fewer than n_splits"):
            domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

    def test_target_too_small_raises(self):
        """If target has fewer than n_splits rows, raise ValueError."""
        X_source = np.random.randn(100, 5)
        X_target = np.random.randn(2, 5)  # Only 2 rows

        with pytest.raises(ValueError, match="fewer than n_splits"):
            domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

    def test_both_exactly_n_splits_is_allowed(self):
        """Edge case: exactly n_splits rows on each side should work."""
        np.random.seed(111)
        X_source = np.random.randn(5, 3)
        X_target = np.random.randn(5, 3) + 2

        result = domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

        # Should complete without raising and have non-NaN AUC
        assert not np.isnan(result.auc)


class TestShiftDistanceDataclass:
    """Verify ShiftDistance dataclass structure and fields."""

    def test_shiftdistance_has_required_fields(self):
        """ShiftDistance must have auc, interval, n_source, n_target, n_splits."""
        np.random.seed(222)
        X_source = np.random.randn(100, 5)
        X_target = np.random.randn(100, 5)

        result = domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

        assert hasattr(result, "auc")
        assert hasattr(result, "interval")
        assert hasattr(result, "n_source")
        assert hasattr(result, "n_target")
        assert hasattr(result, "n_splits")

    def test_interval_is_tuple_of_two_floats(self):
        """The interval must be a tuple of exactly two floats."""
        np.random.seed(333)
        X_source = np.random.randn(100, 5)
        X_target = np.random.randn(100, 5)

        result = domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

        assert isinstance(result.interval, tuple)
        assert len(result.interval) == 2
        assert isinstance(result.interval[0], (float, np.floating))
        assert isinstance(result.interval[1], (float, np.floating))
        assert result.interval[0] <= result.interval[1]

    def test_n_source_and_n_target_reflect_subsampled_counts(self):
        """n_source and n_target should be the counts after subsampling."""
        np.random.seed(444)
        X_source = np.random.randn(500, 5)
        X_target = np.random.randn(200, 5)
        # max_n=20000, so min(500, 200, 20000) = 200
        # Both should be subsampled to 200

        result = domain_classifier_auc(
            X_source, X_target, seed=42, n_splits=5, max_n=20000
        )

        assert result.n_source == 200
        assert result.n_target == 200

    def test_max_n_cap_is_applied(self):
        """When both inputs are large, they should be capped at max_n."""
        np.random.seed(555)
        X_source = np.random.randn(10000, 5)
        X_target = np.random.randn(10000, 5)

        result = domain_classifier_auc(
            X_source, X_target, seed=42, n_splits=5, max_n=500
        )

        assert result.n_source == 500
        assert result.n_target == 500

    def test_n_splits_is_stored(self):
        """n_splits should be stored in the result."""
        np.random.seed(666)
        X_source = np.random.randn(100, 5)
        X_target = np.random.randn(100, 5)

        result = domain_classifier_auc(
            X_source, X_target, seed=42, n_splits=3
        )

        assert result.n_splits == 3


class TestOutOfFoldAUC:
    """Verify that AUC is computed out-of-fold during cross-validation."""

    def test_auc_is_out_of_fold(self):
        """The AUC should be the mean of per-fold out-of-fold predictions,
        not training-set AUC."""
        np.random.seed(777)
        X_source = np.random.randn(200, 5)
        X_target = np.random.randn(200, 5) + 1.5

        result = domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

        # With cross-validation, OOF AUC should be computed correctly.
        # We just check it's a valid probability.
        assert 0.5 <= result.auc <= 1.0


class TestIntervalAcrossFolds:
    """The interval must be computed across folds."""

    def test_interval_is_non_zero_width(self):
        """The interval should have non-zero width across different folds."""
        np.random.seed(888)
        X_source = np.random.randn(200, 5)
        X_target = np.random.randn(200, 5) + 1

        result = domain_classifier_auc(X_source, X_target, seed=42, n_splits=5)

        # The interval should not be a point (lower < upper in almost all cases)
        # Allow tiny tolerance for very stable results
        assert result.interval[1] - result.interval[0] > 0.001


class TestStandardization:
    """The features should be standardized before training the classifier."""

    def test_feature_scale_does_not_affect_auc_dramatically(self):
        """AUC should not depend on feature scaling (LogReg should handle it)."""
        np.random.seed(999)
        X_source_orig = np.random.randn(200, 5)
        X_target_orig = np.random.randn(200, 5) + 2

        # Scaled version
        X_source_scaled = X_source_orig * 1000
        X_target_scaled = X_target_orig * 1000

        result_orig = domain_classifier_auc(
            X_source_orig, X_target_orig, seed=42, n_splits=5
        )
        result_scaled = domain_classifier_auc(
            X_source_scaled, X_target_scaled, seed=42, n_splits=5
        )

        # Standardization should handle this, so AUCs should be close
        # (logistic regression + standardization should be scale-invariant)
        assert abs(result_orig.auc - result_scaled.auc) < 0.05
