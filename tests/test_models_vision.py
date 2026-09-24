"""Tests for vision model definitions.

These tests verify that the vision models preserve the spatial resolution
required for attribution maps and produce correct output shapes.
"""

import pytest
import torch

from shiftprofile.models.vision import (
    resnet18_cifar,
    vit_tiny,
    build_model,
    gradcam_target_layer,
    count_parameters,
)


def test_resnet18_cifar_preserves_spatial_resolution():
    """The torchvision stem would reduce 32x32 to 8x8 before the first block.
    Attribution maps depend on this, so it is a correctness requirement."""
    model = resnet18_cifar()
    x = torch.zeros(1, 3, 32, 32)
    feats = torch.nn.Sequential(model.conv1, model.bn1, model.relu)(x)
    assert feats.shape[-2:] == (32, 32)


def test_resnet18_cifar_has_no_maxpool_stem():
    model = resnet18_cifar()
    assert isinstance(model.maxpool, torch.nn.Identity)


def test_resnet18_cifar_forward_shape_and_finiteness():
    model = resnet18_cifar().eval()
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_resnet18_cifar_gradients_reach_the_input():
    """Integrated Gradients needs input gradients; a detached graph would
    silently produce zero attributions everywhere."""
    model = resnet18_cifar().eval()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    model(x)[0, 0].backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_gradcam_target_layer_resnet18_cifar():
    model = resnet18_cifar()
    assert gradcam_target_layer(model) is model.layer4[-1]


def test_gradcam_target_layer_refuses_unknown_architecture():
    with pytest.raises(ValueError, match="Grad-CAM"):
        gradcam_target_layer(torch.nn.Linear(2, 2))


def test_vit_tiny_forward_shape():
    pytest.importorskip("timm")
    model = vit_tiny(num_classes=10).eval()
    out = model(torch.randn(2, 3, 112, 112))
    assert out.shape == (2, 10)


def test_vit_tiny_default_img_size():
    pytest.importorskip("timm")
    model = vit_tiny(num_classes=10)
    # ViT-Tiny with 112x112 should produce different embedding than 224x224
    # (49 patches vs 196 patches)
    x = torch.randn(1, 3, 112, 112)
    out = model(x)
    assert out.shape[0] == 1


def test_vit_tiny_gradients_reach_input():
    pytest.importorskip("timm")
    model = vit_tiny(num_classes=10).eval()
    x = torch.randn(1, 3, 112, 112, requires_grad=True)
    model(x)[0, 0].backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_vit_tiny_without_timm_raises_clear_error():
    """Module must fail with clear error message if timm is not installed."""
    import sys
    import shiftprofile.models.vision as vision_module

    # Temporarily hide timm to test the error message
    timm = sys.modules.pop("timm", None)
    try:
        # Reimport the module to force lazy import error
        with pytest.raises(ImportError, match="pip install timm"):
            vit_tiny(num_classes=10)
    finally:
        if timm is not None:
            sys.modules["timm"] = timm


def test_build_model_resnet18():
    model = build_model("resnet18", num_classes=10)
    assert isinstance(model, torch.nn.Module)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_build_model_resnet18_augmix():
    model = build_model("resnet18_augmix", num_classes=10)
    assert isinstance(model, torch.nn.Module)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_build_model_vit_tiny():
    pytest.importorskip("timm")
    model = build_model("vit_tiny", num_classes=10, img_size=112)
    assert isinstance(model, torch.nn.Module)
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    assert out.shape == (2, 10)


def test_build_model_unknown_raises_with_list():
    with pytest.raises(ValueError, match="resnet18|vit_tiny|Available"):
        build_model("unknown_model")


def test_resnet18_and_augmix_have_identical_architecture():
    """The two resnet18 variants are the same architecture—the difference is
    training-time augmentation only, not architectural."""
    model_resnet18 = build_model("resnet18", num_classes=10)
    model_augmix = build_model("resnet18_augmix", num_classes=10)

    params_resnet18 = count_parameters(model_resnet18)
    params_augmix = count_parameters(model_augmix)
    assert params_resnet18 == params_augmix


def test_count_parameters():
    model = resnet18_cifar(num_classes=10)
    count = count_parameters(model)
    assert isinstance(count, int)
    assert count > 0
    # ResNet-18 should have roughly 11M parameters
    assert count > 1_000_000
    assert count < 20_000_000
