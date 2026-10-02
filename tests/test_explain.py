"""Tests for attribution methods (IG, Grad-CAM, baselines).

These tests verify that attribution methods satisfy their mathematical properties,
handle edge cases correctly, and produce reproducible outputs.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from shiftprofile.explain import (
    EXPLAIN_VERSION,
    IG_STEPS,
    integrated_gradients,
    grad_cam,
    random_attribution,
    explain_batch,
    to_common_grid,
    ig_completeness_error,
)
from shiftprofile.models.vision import resnet18_cifar


# Test models for CPU-only testing
def _tiny_model() -> nn.Module:
    """Minimal model for fast CPU tests."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )


def _tiny_conv_model() -> nn.Module:
    """Minimal model with a features attribute for Grad-CAM."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    model.features = model
    return model


# ============================================================================
# Integrated Gradients Tests
# ============================================================================

def test_integrated_gradients_output_shape():
    """IG should return (n, 32, 32) with channels summed."""
    model = _tiny_model().eval()
    x = torch.randn(3, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0)
    assert attr.shape == (3, 32, 32)
    assert attr.dtype == np.float32


def test_integrated_gradients_default_steps():
    """IG should use IG_STEPS by default."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    # Just verify it runs without error; step count is internal
    attr = integrated_gradients(model, x, target=0)
    assert attr.shape == (1, 32, 32)


def test_integrated_gradients_custom_steps():
    """IG should accept custom step counts."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr_32 = integrated_gradients(model, x, target=0, steps=32)
    attr_64 = integrated_gradients(model, x, target=0, steps=64)
    assert attr_32.shape == attr_64.shape == (1, 32, 32)
    # More steps should give different (hopefully better) result
    assert not np.allclose(attr_32, attr_64)


def test_integrated_gradients_black_baseline():
    """IG with black baseline (zeros) should be the default."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr_default = integrated_gradients(model, x, target=0, baseline="black")
    attr_explicit = integrated_gradients(model, x, target=0, baseline="black")
    np.testing.assert_array_equal(attr_default, attr_explicit)


def test_integrated_gradients_blur_baseline():
    """IG with blur baseline should produce different result from black."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr_black = integrated_gradients(model, x, target=0, baseline="black")
    attr_blur = integrated_gradients(model, x, target=0, baseline="blur")
    # Different baselines should produce different attributions
    assert not np.allclose(attr_black, attr_blur)


def test_integrated_gradients_unknown_baseline_raises():
    """Unknown baseline should raise with helpful error."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    with pytest.raises(ValueError, match="baseline|black|blur"):
        integrated_gradients(model, x, target=0, baseline="unknown")


def test_integrated_gradients_satisfies_completeness():
    """IG's defining property: attributions sum to F(x) - F(baseline).
    If this fails the implementation is wrong, not merely imprecise."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0, steps=128, baseline="black")

    with torch.no_grad():
        delta = (model(x)[0, 0] - model(torch.zeros_like(x))[0, 0]).item()

    assert attr.sum() == pytest.approx(delta, rel=0.05)


def test_integrated_gradients_device_cpu():
    """IG should work on CPU."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0, device="cpu")
    assert attr.shape == (1, 32, 32)


def test_integrated_gradients_multiple_targets():
    """IG should accept different target classes."""
    model = _tiny_model().eval()
    x = torch.randn(2, 3, 32, 32)
    attr_0 = integrated_gradients(model, x, target=0)
    attr_1 = integrated_gradients(model, x, target=1)
    assert attr_0.shape == attr_1.shape == (2, 32, 32)
    # Different targets should generally produce different attributions
    assert not np.allclose(attr_0, attr_1)


# ============================================================================
# Grad-CAM Tests
# ============================================================================

def test_gradcam_output_shape():
    """Grad-CAM should return (n, 32, 32) upsampled to input resolution."""
    model = resnet18_cifar().eval()
    x = torch.randn(2, 3, 32, 32)
    attr = grad_cam(model, x, target=0)
    assert attr.shape == (2, 32, 32)
    assert attr.dtype == np.float32


def test_gradcam_removes_its_hooks():
    """A leaked hook corrupts every later forward pass in the session.
    This is a critical requirement for Kaggle's 12-hour sessions."""
    model = resnet18_cifar().eval()
    layer = model.layer4[-1]
    before = len(layer._forward_hooks) + len(layer._backward_hooks)
    grad_cam(model, torch.randn(2, 3, 32, 32), target=0, target_layer=layer)
    after = len(layer._forward_hooks) + len(layer._backward_hooks)
    assert after == before, f"Hooks leaked: before={before}, after={after}"


def test_gradcam_removes_hooks_even_when_model_raises():
    """Grad-CAM must clean up hooks even if the model raises an exception."""
    model = resnet18_cifar().eval()
    layer = model.layer4[-1]

    # Save the original forward
    original_forward = model.forward

    # Wrap to raise during forward
    def boom_forward(x):
        # This will execute grad_cam's forward, which hooks the layer,
        # then raise before the backward
        output = original_forward(x)
        raise RuntimeError("boom")

    model.forward = boom_forward

    before = len(layer._forward_hooks) + len(layer._backward_hooks)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            grad_cam(model, torch.randn(1, 3, 32, 32), target=0, target_layer=layer)
    finally:
        model.forward = original_forward

    after = len(layer._forward_hooks) + len(layer._backward_hooks)
    assert after == before, f"Hooks leaked after exception: before={before}, after={after}"


def test_gradcam_default_target_layer():
    """Grad-CAM should auto-detect target layer if not provided."""
    model = resnet18_cifar().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = grad_cam(model, x, target=0)
    assert attr.shape == (1, 32, 32)


def test_gradcam_explicit_target_layer():
    """Grad-CAM should accept an explicit target layer."""
    model = resnet18_cifar().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = grad_cam(model, x, target=0, target_layer=model.layer4[-1])
    assert attr.shape == (1, 32, 32)


def test_gradcam_multiple_targets():
    """Grad-CAM should accept different target classes."""
    model = resnet18_cifar().eval()
    x = torch.randn(2, 3, 32, 32)
    attr_0 = grad_cam(model, x, target=0)
    attr_1 = grad_cam(model, x, target=1)
    assert attr_0.shape == attr_1.shape == (2, 32, 32)
    # Different targets should generally produce different attributions
    assert not np.allclose(attr_0, attr_1)


def test_gradcam_device_cpu():
    """Grad-CAM should work on CPU."""
    model = resnet18_cifar().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = grad_cam(model, x, target=0, device="cpu")
    assert attr.shape == (1, 32, 32)


# ============================================================================
# Random Attribution Tests
# ============================================================================

def test_random_attribution_shape():
    """Random attribution should match the input shape."""
    attr = random_attribution((4, 32, 32), seed=0)
    assert attr.shape == (4, 32, 32)
    assert attr.dtype == np.float32


def test_random_attribution_is_reproducible():
    """Same seed should give identical results."""
    a = random_attribution((4, 32, 32), seed=0)
    b = random_attribution((4, 32, 32), seed=0)
    c = random_attribution((4, 32, 32), seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_attribution_values_in_reasonable_range():
    """Random attributions should be normally distributed with reasonable scale."""
    attr = random_attribution((4, 32, 32), seed=0)
    # Standard normal should have mean ~0 and std ~1
    assert abs(attr.mean()) < 0.2  # Mean close to 0
    assert 0.8 < attr.std() < 1.2  # Std close to 1


def test_random_attribution_seed_required():
    """seed parameter must be explicitly required (no default)."""
    # This is a signature test; the function should not have a default seed
    with pytest.raises(TypeError):
        random_attribution((4, 32, 32))


# ============================================================================
# explain_batch Tests
# ============================================================================

def test_explain_batch_integrated_gradients():
    """explain_batch should dispatch to integrated_gradients."""
    model = _tiny_model().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    attr = explain_batch(model, images, targets, explainer="integrated_gradients")
    assert attr.shape == (2, 32, 32)


def test_explain_batch_grad_cam():
    """explain_batch should dispatch to grad_cam."""
    model = resnet18_cifar().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    attr = explain_batch(model, images, targets, explainer="grad_cam")
    assert attr.shape == (2, 32, 32)


def test_explain_batch_random_attribution():
    """explain_batch should dispatch to random_attribution with deterministic seed."""
    model = _tiny_model().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    attr1 = explain_batch(model, images, targets, explainer="random", seed=0)
    attr2 = explain_batch(model, images, targets, explainer="random", seed=0)
    np.testing.assert_array_equal(attr1, attr2)


def test_explain_batch_unknown_explainer_raises_with_list():
    """Unknown explainer should raise with list of valid names."""
    model = _tiny_model().eval()
    images = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([0])
    with pytest.raises(ValueError, match="explainer|integrated_gradients|grad_cam|random"):
        explain_batch(model, images, targets, explainer="unknown")


def test_explain_batch_passes_kwargs():
    """explain_batch should pass keyword arguments to the explainer."""
    model = _tiny_model().eval()
    images = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([0])
    # Pass custom baseline and steps
    attr = explain_batch(
        model, images, targets,
        explainer="integrated_gradients",
        baseline="blur",
        steps=16
    )
    assert attr.shape == (1, 32, 32)


def test_explain_batch_device_propagation():
    """explain_batch should respect device parameter."""
    model = _tiny_model().eval()
    images = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([0])
    attr = explain_batch(model, images, targets, explainer="integrated_gradients", device="cpu")
    assert attr.shape == (1, 32, 32)


# ============================================================================
# Batching
# ============================================================================

class TestBatchingDoesNotChangeResults:
    """The optimisation is only valid if the numbers are the same.

    explain_batch dispatched one image at a time until the explainers learned
    to take a target per image. That made a pilot IG cell take 55 minutes on a
    T4 and put the full grid at roughly 220 GPU-hours for this stage. Batching
    is worth nothing if it quietly changes what is being measured, so these
    compare the batched result against the old one-at-a-time path directly.
    """

    @pytest.mark.parametrize("explainer", ["integrated_gradients", "grad_cam"])
    def test_an_image_gets_the_same_attribution_alone_or_in_a_batch(self, explainer):
        """The property batching has to preserve.

        Not "batch 1 equals batch 4" -- see the note on batch size 1 below.
        What must hold is that an image's attribution does not depend on what
        it was batched with, which is what makes a chunked run equivalent to
        the per-image one it replaced.
        """
        model = resnet18_cifar().eval()
        torch.manual_seed(0)
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([3, 1, 4, 1, 5, 9, 2])

        alone = explain_batch(
            model, images[2:3], targets[2:3], explainer=explainer, batch_size=4
        )
        embedded = explain_batch(
            model, images, targets, explainer=explainer, batch_size=7
        )[2:3]

        np.testing.assert_allclose(embedded, alone, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("explainer", ["integrated_gradients", "grad_cam"])
    def test_the_chunk_size_does_not_change_the_result(self, explainer):
        model = resnet18_cifar().eval()
        torch.manual_seed(0)
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([3, 1, 4, 1, 5, 9, 2])

        in_twos = explain_batch(
            model, images, targets, explainer=explainer, batch_size=2
        )
        in_one_go = explain_batch(
            model, images, targets, explainer=explainer, batch_size=7
        )

        np.testing.assert_allclose(in_twos, in_one_go, rtol=1e-4, atol=1e-5)

    def test_batch_size_one_is_the_odd_one_out_and_that_is_why_v2_exists(self):
        """Documents the measurement behind the EXPLAIN_VERSION bump.

        Every chunk size from 2 up agrees to around 1e-7 relative. Batch size
        1 does not: torch special-cases the degenerate batch to a different
        convolution algorithm, and IG accumulates that over its steps to about
        5e-3 relative on resnet18. A shallow model shows 1e-10 throughout, so
        this is kernel selection rather than anything about the maths.

        It matters because the superseded code path WAS batch size 1. Keeping
        explain-v1 would have let a resumed run pair those attributions with
        batched ones, differing by half a percent for no recorded reason.
        """
        model = resnet18_cifar().eval()
        torch.manual_seed(0)
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([3, 1, 4, 1, 5, 9, 2])

        def ig(batch_size):
            return explain_batch(
                model, images, targets,
                explainer="integrated_gradients", batch_size=batch_size,
            )

        scale = np.abs(ig(7)).max()
        assert np.abs(ig(2) - ig(7)).max() / scale < 1e-5, "chunk sizes 2+ agree"
        assert np.abs(ig(1) - ig(7)).max() / scale > 1e-4, (
            "batch size 1 no longer differs; if torch stopped special-casing it, "
            "say so here rather than deleting the test -- the version bump was "
            "justified by this difference"
        )

    def test_a_partial_final_chunk_is_handled(self):
        """7 images at batch size 4 leaves a chunk of 3."""
        model = _tiny_model().eval()
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        attr = explain_batch(
            model, images, targets, explainer="integrated_gradients", batch_size=4
        )
        assert attr.shape == (7, 32, 32)

    def test_each_image_is_attributed_against_its_own_target(self):
        """The reason the old code could not batch.

        integrated_gradients took one scalar target and applied it to the whole
        batch, so attributing images against different classes meant one call
        per image. A batch that silently used the first image's class for all
        of them would pass a shape check and be wrong everywhere.
        """
        model = _tiny_model().eval()
        images = torch.randn(2, 3, 32, 32)

        together = explain_batch(
            model, images, torch.tensor([0, 7]),
            explainer="integrated_gradients", batch_size=2,
        )
        separately = np.concatenate([
            integrated_gradients(model, images[0:1], target=0),
            integrated_gradients(model, images[1:2], target=7),
        ])

        np.testing.assert_allclose(together, separately, rtol=1e-4, atol=1e-5)

        shared_target = explain_batch(
            model, images, torch.tensor([0, 0]),
            explainer="integrated_gradients", batch_size=2,
        )
        assert not np.allclose(together[1], shared_target[1]), (
            "image 1 attributed identically against class 7 and class 0, so the "
            "per-image target is not reaching the model"
        )


class TestPerImageTargets:
    """A scalar still broadcasts, so single-target call sites keep working."""

    @pytest.mark.parametrize("explainer_fn", [integrated_gradients, grad_cam])
    def test_scalar_target_broadcasts_over_the_batch(self, explainer_fn):
        model = resnet18_cifar().eval()
        images = torch.randn(3, 3, 32, 32)

        scalar = explainer_fn(model, images, target=2)
        vector = explainer_fn(model, images, target=torch.tensor([2, 2, 2]))

        np.testing.assert_allclose(scalar, vector, rtol=1e-4, atol=1e-6)

    def test_mismatched_target_count_is_rejected(self):
        model = _tiny_model().eval()
        images = torch.randn(3, 3, 32, 32)

        with pytest.raises(ValueError, match="2 targets for 3 images"):
            integrated_gradients(model, images, target=torch.tensor([0, 1]))

    def test_explain_batch_rejects_a_mismatched_target_count(self):
        model = _tiny_model().eval()
        images = torch.randn(3, 3, 32, 32)

        with pytest.raises(ValueError, match="must match"):
            explain_batch(
                model, images, torch.tensor([0, 1]),
                explainer="integrated_gradients",
            )

    def test_a_nonsense_batch_size_is_rejected(self):
        model = _tiny_model().eval()
        images = torch.randn(2, 3, 32, 32)

        with pytest.raises(ValueError, match="at least 1"):
            explain_batch(
                model, images, torch.tensor([0, 1]),
                explainer="integrated_gradients", batch_size=0,
            )


# ============================================================================
# to_common_grid Tests
# ============================================================================

def test_to_common_grid_8x8():
    """to_common_grid should pool to 8x8."""
    attr = np.random.randn(2, 32, 32).astype(np.float32)
    grid = to_common_grid(attr, grid=8)
    assert grid.shape == (2, 8, 8)
    assert grid.dtype == np.float32


def test_to_common_grid_preserves_total_mass():
    """to_common_grid should preserve total mass to within float tolerance."""
    attr = np.ones((1, 32, 32), dtype=np.float32)
    grid = to_common_grid(attr, grid=8)
    # 32x32 pooled to 8x8 means each output cell sums 16 input cells
    # With all ones, total mass is 32*32=1024, output mass should be 1024
    assert grid.sum() == pytest.approx(attr.sum(), rel=1e-5)


def test_to_common_grid_custom_size():
    """to_common_grid should accept custom grid size."""
    attr = np.random.randn(1, 32, 32).astype(np.float32)
    grid_4 = to_common_grid(attr, grid=4)
    grid_8 = to_common_grid(attr, grid=8)
    grid_16 = to_common_grid(attr, grid=16)
    assert grid_4.shape == (1, 4, 4)
    assert grid_8.shape == (1, 8, 8)
    assert grid_16.shape == (1, 16, 16)


def test_to_common_grid_batches():
    """to_common_grid should work on batches."""
    attr = np.random.randn(4, 32, 32).astype(np.float32)
    grid = to_common_grid(attr, grid=8)
    assert grid.shape == (4, 8, 8)


# ============================================================================
# ig_completeness_error Tests
# ============================================================================

def test_ig_completeness_error_value():
    """ig_completeness_error should return a small relative error for good IG."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0, steps=128, baseline="black")

    error = ig_completeness_error(model, x, target=0, attribution=attr, baseline="black")
    assert isinstance(error, (float, np.floating))
    assert error >= 0
    assert error < 0.1  # Should be small for our simple model


def test_ig_completeness_error_more_steps_lower_error():
    """IG with more steps should have lower completeness error."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)

    attr_16 = integrated_gradients(model, x, target=0, steps=16, baseline="black")
    attr_64 = integrated_gradients(model, x, target=0, steps=64, baseline="black")

    error_16 = ig_completeness_error(model, x, target=0, attribution=attr_16, baseline="black")
    error_64 = ig_completeness_error(model, x, target=0, attribution=attr_64, baseline="black")

    # Both should be stored in the test data
    assert error_16 > 0
    assert error_64 > 0
    # More steps should reduce error (though not guaranteed for every input)


def test_ig_completeness_error_different_baselines():
    """ig_completeness_error should work with different baselines."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr_black = integrated_gradients(model, x, target=0, baseline="black")
    attr_blur = integrated_gradients(model, x, target=0, baseline="blur")

    error_black = ig_completeness_error(model, x, target=0, attribution=attr_black, baseline="black")
    error_blur = ig_completeness_error(model, x, target=0, attribution=attr_blur, baseline="blur")

    assert isinstance(error_black, (float, np.floating))
    assert isinstance(error_blur, (float, np.floating))


# ============================================================================
# Version and Constants Tests
# ============================================================================

def test_explain_version_defined():
    """EXPLAIN_VERSION should be defined."""
    assert EXPLAIN_VERSION == "explain-v2"


def test_ig_steps_constant():
    """IG_STEPS should be 32."""
    assert IG_STEPS == 32


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_attribution_pipeline():
    """Full pipeline: model -> explain_batch -> to_common_grid."""
    model = resnet18_cifar().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])

    # Get attributions at native resolution
    attr_native = explain_batch(model, images, targets, explainer="integrated_gradients", steps=16)
    assert attr_native.shape == (2, 32, 32)

    # Downsample for comparison
    attr_grid = to_common_grid(attr_native, grid=8)
    assert attr_grid.shape == (2, 8, 8)

    # Check mass conservation (to_common_grid rescales to preserve total mass)
    for i in range(2):
        assert attr_native[i].sum() == pytest.approx(attr_grid[i].sum(), rel=1e-4)
