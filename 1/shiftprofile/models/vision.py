"""Vision model definitions for CIFAR-10 and attribution-friendly architectures.

This module provides vision models optimized for distribution shift studies.
The key focus is preserving spatial resolution for attribution maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-10 with spatial resolution preservation.

    The standard torchvision ResNet-18 uses a 7x7 stride-2 conv and maxpool,
    which on 32x32 inputs throws away almost all spatial resolution before the
    first residual block. This version replaces the stem with a 3x3 stride-1
    conv and replaces maxpool with an Identity layer, preserving 32x32
    resolution through the stem.

    This is a correctness requirement for attribution maps, not a performance
    tweak.

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10).

    Returns:
        A ResNet-18 module with CIFAR-adapted stem.
    """
    # Load the standard ResNet-18
    model = models.resnet18(weights=None, num_classes=num_classes)

    # Replace the standard stem with a CIFAR-adapted one
    # Standard stem: 7x7 stride-2 conv + 64 filters
    # CIFAR stem: 3x3 stride-1 conv + 64 filters
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model


def vit_tiny(
    img_size: int = 112, num_classes: int = 10, pretrained: bool = True
) -> nn.Module:
    """Vision Transformer Tiny using timm.

    Args:
        img_size: Input image size. Default 112 is deliberate: CIFAR-10 is 32x32,
                 upscaling to 224 adds no information, only cost. ViT-Tiny at 224
                 produces 196 patches (~80% of the pilot's attribution budget);
                 at 112 it produces 49 patches, a ~4x reduction.
        num_classes: Number of output classes (default: 10 for CIFAR-10).
        pretrained: Whether to load pretrained ImageNet weights (default: True).

    Returns:
        A ViT-Tiny module.

    Raises:
        ImportError: If timm is not installed.
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "vit_tiny requires timm to be installed. "
            "Install it with: pip install timm"
        )

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
    )
    return model


BUILDERS = {
    "resnet18": resnet18_cifar,
    "resnet18_augmix": resnet18_cifar,  # Same architecture; augmentation is training-time only
    "vit_tiny": vit_tiny,
}


def build_model(model_id: str, **kwargs) -> nn.Module:
    """Dispatch to a named model by ID.

    Args:
        model_id: A registered model name ("resnet18", "resnet18_augmix", "vit_tiny").
        **kwargs: Passed to the model constructor.

    Returns:
        A constructed model module.

    Raises:
        ValueError: If model_id is unknown.
    """
    if model_id not in BUILDERS:
        available = ", ".join(sorted(BUILDERS))
        raise ValueError(
            f"unknown model {model_id!r}. Available: {available}"
        )

    return BUILDERS[model_id](**kwargs)


# The model ids whose only difference from their base architecture is AugMix at
# training time. `build_model` maps `resnet18_augmix` to the same constructor as
# `resnet18`, so the id is the *only* thing that distinguishes them, and this set
# is where that meaning lives.
AUGMIX_MODELS = frozenset({"resnet18_augmix"})


def uses_augmix(model_id: str) -> bool:
    """Whether this model id means "trained with AugMix".

    Single source of truth, because the alternative was a free `augmix` flag on
    `train_model` that nothing ever set. `resnet18_augmix` therefore trained
    identically to `resnet18`, and the two checkpoints differed only in their
    cache key -- a pre-registered arm that silently measured nothing. A flag that
    can disagree with the id it is paired with will eventually disagree with it.

    Args:
        model_id: A registered model name.

    Returns:
        True if training this id applies AugMix augmentation.

    Raises:
        ValueError: If model_id is unknown. Returning False for an unrecognised
            id would make a typo in a config silently train the wrong thing.
    """
    if model_id not in BUILDERS:
        raise ValueError(
            f"unknown model {model_id!r}. Available: {', '.join(sorted(BUILDERS))}"
        )
    return model_id in AUGMIX_MODELS


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the module whose activations Grad-CAM should hook.

    For attribution methods like Grad-CAM, we need the last convolutional block.

    Args:
        model: A model module (must be a known architecture).

    Returns:
        The target layer for Grad-CAM (typically the last residual or conv block).

    Raises:
        ValueError: If the architecture is unknown.
    """
    # Check for ResNet-18 with CIFAR stem (has layer4 attribute)
    if hasattr(model, "layer4"):
        return model.layer4[-1]

    raise ValueError(
        f"Grad-CAM target layer not defined for architecture {type(model).__name__}. "
        "A silently wrong layer produces plausible-looking but meaningless maps."
    )


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: A model module.

    Returns:
        The total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())
