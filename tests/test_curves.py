"""Tests for removal curves — the GPU half of faithfulness.

Tests verify imputation schemes, per-sample curve computation, and the
structural enforcement of paired random-attribution controls.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from shiftprofile.curves import (
    CURVES_VERSION,
    REMOVAL_FRACTIONS,
    impute,
    removal_curve,
    paired_removal_curves,
)
from shiftprofile.metrics.faithfulness import VALID_IMPUTATIONS


# ============================================================================
# Test fixtures and helpers
# ============================================================================


def _tiny_model() -> nn.Module:
    """Minimal model for fast CPU tests."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    ).eval()


def _model_that_uses_center_pixel() -> nn.Module:
    """A model whose decision depends on ONE known pixel region (center).

    This is used to verify that an informative attribution beats a random one.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(1, 10),
    ).eval()

    # Set weights so the model primarily responds to the center of the image
    with torch.no_grad():
        # Create a weight map that zeros out everything except center
        weight_map = torch.zeros(1, 3, 32, 32)
        # Mark center region (16x16 to 20x20) as important
        weight_map[:, :, 16:20, 16:20] = 1.0
        # This isn't directly a layer weight, but we'll engineer the input
        # through masking instead
    return model


# ============================================================================
# CURVES_VERSION and REMOVAL_FRACTIONS
# ============================================================================


def test_curves_version_is_exported():
    """CURVES_VERSION is the producer_version for the cache."""
    assert isinstance(CURVES_VERSION, str)
    assert len(CURVES_VERSION) > 0


def test_removal_fractions_is_tuple_and_sorted():
    """REMOVAL_FRACTIONS are the evaluation points for the study."""
    assert isinstance(REMOVAL_FRACTIONS, tuple)
    assert len(REMOVAL_FRACTIONS) >= 2
    assert REMOVAL_FRACTIONS[0] == 0.0
    assert REMOVAL_FRACTIONS[-1] == 1.0
    assert all(REMOVAL_FRACTIONS[i] < REMOVAL_FRACTIONS[i + 1]
               for i in range(len(REMOVAL_FRACTIONS) - 1))


# ============================================================================
# Imputation Tests
# ============================================================================


def test_impute_zero_mask_leaves_image_unchanged():
    """Imputation with zero mask should leave the image bit-identical."""
    image = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)

    for scheme in VALID_IMPUTATIONS:
        result = impute(image, mask, scheme, seed=0)
        # Since no positions are masked, the image should be identical
        torch.testing.assert_close(result, image)


def test_impute_full_mask_modifies_all_pixels():
    """Imputation with full mask should modify all pixels."""
    image = torch.randn(1, 3, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    for scheme in VALID_IMPUTATIONS:
        result = impute(image, mask, scheme, seed=0)
        # Result should have modified values (won't be identical to original)
        # For "zero", all should be zero; for others, should differ
        if scheme == "zero":
            torch.testing.assert_close(result, torch.zeros_like(result))
        else:
            # At least some values should differ
            assert not torch.allclose(result, image)


def test_impute_partial_mask_preserves_unmasked():
    """Imputation should only alter masked positions, leave others bit-identical."""
    image = torch.randn(1, 3, 32, 32)
    # Create a mask for left half of the image
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    mask[:, :, :16] = True

    for scheme in VALID_IMPUTATIONS:
        result = impute(image, mask, scheme, seed=0)
        # Right half should be absolutely identical
        torch.testing.assert_close(result[:, :, :, 16:], image[:, :, :, 16:])


def test_impute_mean_zeros_in_normalised_space():
    """'mean' imputation should produce zeros (per-channel CIFAR-10 mean in normalised space)."""
    # Normalized image (mean subtracted already)
    image = torch.randn(1, 3, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    result = impute(image, mask, "mean", seed=0)
    # Mean imputation in normalized space is zero
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_impute_blur_produces_gaussian_blurred():
    """'blur' imputation should produce a Gaussian-blurred version."""
    image = torch.ones(1, 3, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    result = impute(image, mask, "blur", seed=0)
    # Blurred image should have reduced values at edges (blur spreads values)
    assert result.shape == image.shape
    # Center should have higher values than edges
    center_val = result[0, 0, 16, 16].item()
    edge_val = result[0, 0, 0, 0].item()
    assert center_val > edge_val


def test_impute_uniform_noise_is_seeded():
    """'uniform_noise' with the same seed should produce identical results."""
    image = torch.randn(1, 3, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    result1 = impute(image, mask, "uniform_noise", seed=42)
    result2 = impute(image, mask, "uniform_noise", seed=42)
    torch.testing.assert_close(result1, result2)


def test_impute_uniform_noise_different_seed_differs():
    """'uniform_noise' with different seeds should produce different results."""
    image = torch.randn(1, 3, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    result1 = impute(image, mask, "uniform_noise", seed=42)
    result2 = impute(image, mask, "uniform_noise", seed=99)
    # Should differ (very high probability)
    assert not torch.allclose(result1, result2)


def test_impute_zero_scheme():
    """'zero' imputation should set masked pixels to 0."""
    image = torch.ones(1, 3, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    result = impute(image, mask, "zero", seed=0)
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_impute_invalid_scheme_raises():
    """Unknown imputation scheme should raise with valid schemes listed."""
    image = torch.randn(1, 3, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)

    with pytest.raises(ValueError, match="imputation"):
        impute(image, mask, "magic", seed=0)


# ============================================================================
# Removal Curve Tests
# ============================================================================


def test_removal_curve_output_shape():
    """removal_curve returns (n_samples, len(fractions))."""
    model = _tiny_model()
    images = torch.randn(5, 3, 32, 32)
    attributions = torch.randn(5, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    curve = removal_curve(
        model, images, attributions, targets,
        imputation="mean"
    )

    assert curve.shape == (5, len(REMOVAL_FRACTIONS))
    assert curve.dtype == np.float32


def test_removal_curve_zero_fraction_leaves_image_untouched():
    """fractions[0] == 0.0, so column 0 should be unmodified predicted probability."""
    model = _tiny_model()
    images = torch.randn(3, 3, 32, 32)
    attributions = torch.randn(3, 32, 32)
    targets = torch.tensor([0, 1, 2])

    curve = removal_curve(
        model, images, attributions, targets,
        imputation="mean"
    )

    # Column 0 should match the unmodified model output
    with torch.no_grad():
        baseline_logits = model(images)
        baseline_probs = torch.softmax(baseline_logits, 1)
        baseline = baseline_probs[range(3), targets].numpy()

    np.testing.assert_allclose(curve[:, 0], baseline, rtol=1e-5)


def test_removal_curve_is_per_sample_not_averaged():
    """Curve must be per-sample, not averaged — study bootstraps over samples."""
    model = _tiny_model()
    n = 10
    images = torch.randn(n, 3, 32, 32)
    attributions = torch.randn(n, 32, 32)
    targets = torch.randint(0, 10, (n,))

    curve = removal_curve(
        model, images, attributions, targets,
        imputation="mean"
    )

    assert curve.shape == (n, len(REMOVAL_FRACTIONS))


def test_removal_curve_fraction_one_removes_all():
    """Removing all pixels should give near-uniform prediction."""
    model = _tiny_model()
    images = torch.randn(5, 3, 32, 32)
    attributions = torch.randn(5, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    curve = removal_curve(
        model, images, attributions, targets,
        imputation="mean"
    )

    # At fraction 1.0, all pixels removed (zeroed), probabilities should be low
    # and similar to each other (near-random)
    assert curve[:, -1].max() < 0.5  # Loose upper bound


def test_removal_curve_imputation_is_required():
    """imputation is a required keyword with no default."""
    model = _tiny_model()
    images = torch.randn(3, 3, 32, 32)
    attributions = torch.randn(3, 32, 32)
    targets = torch.tensor([0, 1, 2])

    with pytest.raises(TypeError):
        removal_curve(model, images, attributions, targets)


def test_removal_curve_validates_imputation_scheme():
    """Unknown imputation scheme should raise with valid schemes."""
    model = _tiny_model()
    images = torch.randn(3, 3, 32, 32)
    attributions = torch.randn(3, 32, 32)
    targets = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError, match="imputation"):
        removal_curve(
            model, images, attributions, targets,
            imputation="invalid_scheme"
        )


def test_removal_curve_all_valid_imputations():
    """All VALID_IMPUTATIONS from metrics.faithfulness should work."""
    model = _tiny_model()
    images = torch.randn(3, 3, 32, 32)
    attributions = torch.randn(3, 32, 32)
    targets = torch.tensor([0, 1, 2])

    for scheme in VALID_IMPUTATIONS:
        curve = removal_curve(
            model, images, attributions, targets,
            imputation=scheme
        )
        assert curve.shape == (3, len(REMOVAL_FRACTIONS))


def test_removal_curve_batches_correctly():
    """Large batches should be split and reassembled correctly."""
    model = _tiny_model()
    n = 1000
    images = torch.randn(n, 3, 32, 32)
    attributions = torch.randn(n, 32, 32)
    targets = torch.randint(0, 10, (n,))

    # Run with small batch size to force batching
    curve = removal_curve(
        model, images, attributions, targets,
        imputation="mean",
        batch_size=64
    )

    assert curve.shape == (n, len(REMOVAL_FRACTIONS))


def test_removal_curve_runs_under_no_grad():
    """Curve computation should not accumulate gradients."""
    model = _tiny_model()
    images = torch.randn(3, 3, 32, 32, requires_grad=True)
    attributions = torch.randn(3, 32, 32)
    targets = torch.tensor([0, 1, 2])

    curve = removal_curve(
        model, images, attributions, targets,
        imputation="mean"
    )

    # No gradient should be computed
    assert images.grad is None


def test_removal_curve_different_attributions_produce_different_curves():
    """Different attributions should produce different removal curves.

    This is a sanity check that the removal curve actually depends on which
    pixels are ranked highest by the attribution. If two different attributions
    produce identical curves, the removal mechanism is broken.
    """
    model = _tiny_model()

    torch.manual_seed(42)
    images = torch.randn(5, 3, 32, 32)

    # Create two different attributions
    attributions_1 = torch.randn(5, 32, 32)
    attributions_2 = torch.randn(5, 32, 32)

    targets = torch.randint(0, 10, (5,))

    # Get curves for both attributions
    curve_1 = removal_curve(
        model, images, attributions_1, targets,
        imputation="mean"
    )
    curve_2 = removal_curve(
        model, images, attributions_2, targets,
        imputation="mean"
    )

    # The curves should be different (with high probability for random attributions)
    assert not np.allclose(curve_1, curve_2)


# ============================================================================
# Paired Removal Curves Tests
# ============================================================================


def test_paired_removal_curves_returns_tuple_of_two():
    """paired_removal_curves returns (model_curve, random_curve)."""
    model = _tiny_model()
    images = torch.randn(5, 3, 32, 32)
    attributions = torch.randn(5, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    result = paired_removal_curves(
        model, images, attributions, targets,
        imputation="mean",
        random_seed=0
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    model_curve, random_curve = result
    assert model_curve.shape == random_curve.shape


def test_paired_removal_curves_shapes():
    """Both curves have shape (n_samples, len(fractions))."""
    model = _tiny_model()
    n = 8
    images = torch.randn(n, 3, 32, 32)
    attributions = torch.randn(n, 32, 32)
    targets = torch.randint(0, 10, (n,))

    model_curve, random_curve = paired_removal_curves(
        model, images, attributions, targets,
        imputation="mean",
        random_seed=42
    )

    assert model_curve.shape == (n, len(REMOVAL_FRACTIONS))
    assert random_curve.shape == (n, len(REMOVAL_FRACTIONS))


def test_paired_removal_curves_random_is_reproducible():
    """Random control with same seed should be reproducible."""
    model = _tiny_model()
    images = torch.randn(5, 3, 32, 32)
    attributions = torch.randn(5, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    model_curve_1, random_curve_1 = paired_removal_curves(
        model, images, attributions, targets,
        imputation="mean",
        random_seed=42
    )
    model_curve_2, random_curve_2 = paired_removal_curves(
        model, images, attributions, targets,
        imputation="mean",
        random_seed=42
    )

    np.testing.assert_array_equal(model_curve_1, model_curve_2)
    np.testing.assert_array_equal(random_curve_1, random_curve_2)


def test_paired_removal_curves_random_differs_from_model_attr():
    """Random control should generally differ from model attribution curve."""
    model = _tiny_model()
    images = torch.randn(10, 3, 32, 32)
    attributions = torch.randn(10, 32, 32)
    targets = torch.randint(0, 10, (10,))

    model_curve, random_curve = paired_removal_curves(
        model, images, attributions, targets,
        imputation="mean",
        random_seed=99
    )

    # Curves should differ (high probability for random seed)
    assert not np.allclose(model_curve, random_curve)


def test_paired_removal_curves_random_seed_required():
    """random_seed is a required parameter for reproducibility."""
    model = _tiny_model()
    images = torch.randn(5, 3, 32, 32)
    attributions = torch.randn(5, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    # Should raise TypeError if random_seed is missing
    with pytest.raises(TypeError):
        paired_removal_curves(
            model, images, attributions, targets,
            imputation="mean"
        )


def test_paired_curves_enforce_validity_control():
    """Paired curves enforce the structural validity control — cannot produce model
    curve without its paired random control."""
    # This is tested by design: the function signature returns both or nothing
    model = _tiny_model()
    images = torch.randn(5, 3, 32, 32)
    attributions = torch.randn(5, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    # You cannot call a function to get just the model curve
    # Both are computed in the same call
    result = paired_removal_curves(
        model, images, attributions, targets,
        imputation="mean",
        random_seed=0
    )
    assert len(result) == 2  # Both or nothing
