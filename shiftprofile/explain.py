"""Attribution methods: Integrated Gradients, Grad-CAM, random baseline.

All methods return attributions in normalized space with channels summed to
(n, 32, 32) shape. Attribution is per-pixel because removal operates on pixels,
not channels.

The design priorities here are:
  1. Correctness: IG satisfies its mathematical property (completeness)
  2. Reproducibility: Random baseline is exactly reproducible by seed
  3. Session robustness: Grad-CAM leaks no hooks, so Kaggle's 12-hour sessions
     do not slowly poison the model
  4. Cost efficiency: IG_STEPS = 32 is ~35% cheaper than the standard 50
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


EXPLAIN_VERSION = "explain-v1"
IG_STEPS = 32


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    *,
    steps: int = IG_STEPS,
    baseline: str = "black",
    device: str = "cpu",
) -> np.ndarray:
    """Integrated Gradients attribution method.

    Args:
        model: A PyTorch model in eval mode.
        x: Input images of shape (n, 3, 32, 32) in normalized space.
        target: Target class index.
        steps: Number of IG integration steps (default: 32).
        baseline: "black" (zeros) or "blur" (Gaussian-blurred input).
        device: Device to run on ("cpu" or "cuda").

    Returns:
        Attribution array of shape (n, 32, 32) with channels summed, dtype float32.

    Raises:
        ValueError: If baseline is not "black" or "blur".
    """
    if baseline not in ("black", "blur"):
        raise ValueError(f"baseline must be 'black' or 'blur', got {baseline!r}")

    x = x.to(device).detach()
    n = x.shape[0]

    # Compute baseline
    if baseline == "black":
        x_baseline = torch.zeros_like(x)
    elif baseline == "blur":
        # Gaussian blur in normalized space
        x_np = x.cpu().numpy()
        x_baseline_np = np.zeros_like(x_np)
        for i in range(n):
            for c in range(3):
                x_baseline_np[i, c] = ndimage.gaussian_filter(x_np[i, c], sigma=1.5)
        x_baseline = torch.from_numpy(x_baseline_np).to(device)

    model.eval()
    attributions = np.zeros((n, 32, 32), dtype=np.float32)

    for step in range(steps):
        # Interpolate between baseline and input
        alpha = step / steps
        x_interp = x_baseline + alpha * (x - x_baseline)
        x_interp.requires_grad = True

        # Forward pass
        with torch.enable_grad():
            out = model(x_interp)
            target_out = out[:, target]
            loss = target_out.sum()

        # Backward pass
        loss.backward()

        # Accumulate gradients (scaled by input difference)
        if x_interp.grad is not None:
            grad = x_interp.grad.detach()
            input_diff = x - x_baseline  # (n, 3, 32, 32)
            scaled_grad = grad * input_diff  # (n, 3, 32, 32)
            # Sum channels and accumulate
            attributions += scaled_grad.sum(dim=1).cpu().numpy() / steps

    return attributions


def grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    *,
    target_layer: Optional[nn.Module] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Grad-CAM attribution method.

    Args:
        model: A PyTorch model in eval mode.
        x: Input images of shape (n, 3, 32, 32).
        target: Target class index.
        target_layer: Layer to hook (default: auto-detect from model).
        device: Device to run on ("cpu" or "cuda").

    Returns:
        Attribution array of shape (n, 32, 32), dtype float32.

    Raises:
        ValueError: If target_layer cannot be auto-detected.

    Note:
        All hooks are removed in a finally block to prevent session pollution.
    """
    from shiftprofile.models.vision import gradcam_target_layer

    x = x.to(device).detach()
    n = x.shape[0]

    if target_layer is None:
        target_layer = gradcam_target_layer(model)

    model.eval()
    attributions = np.zeros((n, 32, 32), dtype=np.float32)

    # Storage for activations and gradients
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        x_input = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = model(x_input)
            target_out = out[:, target]
            loss = target_out.sum()

        loss.backward()

        # Compute Grad-CAM
        # gradients: (n, c, h, w)
        # activations: (n, c, h, w)
        if gradients is not None and activations is not None:
            grad_pooled = gradients.mean(dim=(2, 3), keepdim=True)  # (n, c, 1, 1)
            cam = (activations * grad_pooled).sum(dim=1)  # (n, h, w)
            cam = F.relu(cam)  # ReLU

            # Upsample to 32x32
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(32, 32),
                mode="bilinear",
                align_corners=False
            )  # (n, 1, 32, 32)
            attributions = cam.squeeze(1).cpu().numpy().astype(np.float32)

    finally:
        # CRITICAL: Remove hooks in finally block
        forward_handle.remove()
        backward_handle.remove()

    return attributions


def random_attribution(shape: tuple, *, seed: int) -> np.ndarray:
    """Random attribution baseline for faithfulness comparison.

    Args:
        shape: Shape of the output (n, 32, 32).
        seed: Random seed (required, no default).

    Returns:
        Random attribution array, dtype float32.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def explain_batch(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    explainer: str,
    *,
    device: str = "cpu",
    **kw,
) -> np.ndarray:
    """Batch attribution dispatcher.

    Args:
        model: A PyTorch model in eval mode.
        images: Batch of images (n, 3, 32, 32).
        targets: Target class indices (n,).
        explainer: Name of explainer ("integrated_gradients", "grad_cam", "random").
        device: Device to run on.
        **kw: Additional kwargs passed to the explainer.

    Returns:
        Attribution array of shape (n, 32, 32), dtype float32.

    Raises:
        ValueError: If explainer is unknown.
    """
    explainers = {
        "integrated_gradients": integrated_gradients,
        "grad_cam": grad_cam,
        "random": random_attribution,
    }

    if explainer not in explainers:
        names = ", ".join(sorted(explainers.keys()))
        raise ValueError(
            f"unknown explainer {explainer!r}. Available: {names}"
        )

    # Each explainer takes a DIFFERENT set of options, so kwargs are validated
    # per explainer rather than splatted into whichever function is dispatched.
    # Splatting meant a caller supplying options for all three explainers — the
    # natural thing to do from a config — crashed integrated_gradients with an
    # unexpected 'seed'. Silently dropping unknown keys would be worse: a
    # mistyped 'step' would leave IG on its default and nothing would say so.
    allowed = {
        "integrated_gradients": {"steps", "baseline"},
        "grad_cam": {"target_layer"},
        "random": {"seed"},
    }[explainer]
    unknown = set(kw) - allowed
    if unknown:
        raise TypeError(
            f"explainer {explainer!r} does not accept {sorted(unknown)}; "
            f"it accepts {sorted(allowed)}. Pass only the options this explainer "
            "understands rather than the union for all of them."
        )

    n = images.shape[0]
    targets_list = targets.tolist() if hasattr(targets, 'tolist') else list(targets)

    if explainer == "random":
        if "seed" not in kw:
            raise ValueError(
                "the random explainer requires an explicit 'seed': it is the "
                "control every faithfulness number is reported against, so it "
                "must be exactly reproducible."
            )
        return random_attribution((n, 32, 32), seed=kw["seed"])

    # For integrated_gradients and grad_cam, call per-image with its target
    fn = explainers[explainer]
    attributions = []
    for i in range(n):
        img = images[i:i+1]  # Keep batch dimension
        target = targets_list[i]
        attr = fn(model, img, target=target, device=device, **kw)
        attributions.append(attr)

    return np.concatenate(attributions, axis=0)


def to_common_grid(attribution: np.ndarray, grid: int = 8) -> np.ndarray:
    """Average-pool attribution to a coarser grid for cross-explainer comparison.

    This function is for visualization and aggregation only. Faithfulness metrics
    are ALWAYS computed at native (32, 32) resolution.

    Args:
        attribution: Attribution array of shape (n, 32, 32).
        grid: Output grid size (default: 8 for 8x8).

    Returns:
        Pooled attribution of shape (n, grid, grid), dtype float32, with total mass preserved.
    """
    n, h, w = attribution.shape
    assert h == w == 32, f"expected 32x32, got {h}x{w}"

    # Reshape and pool
    pool_size = h // grid
    pooled = attribution.reshape(n, grid, pool_size, grid, pool_size)
    pooled = pooled.mean(axis=(2, 4))  # Average pool
    # Rescale to preserve total mass
    pooled = pooled * (pool_size ** 2)

    return pooled.astype(np.float32)


def explain_cell(
    cell,  # Cell object
    model: nn.Module,
    clean_root: str,
    corrupt_root: str,
    cache,  # ArtifactCache
    *,
    explainer: str,
    device: str = "cpu",
    indices: Optional[np.ndarray] = None,
    ig_steps: int = IG_STEPS,
    baseline: str = "black",
    random_seed: int = 0,
) -> np.ndarray:
    """Compute attributions for a cell, checking cache first.

    Attributions are cached by (cell, explainer). The attribution target is
    the model's PREDICTED class, not the true label. Predictions must be cached
    in the predict stage first; raises if absent.

    Args:
        cell: A Cell object.
        model: A trained model.
        clean_root: Root directory for clean CIFAR-10 data.
        corrupt_root: Root directory for CIFAR-10-C data.
        cache: ArtifactCache for storing/loading attributions.
        explainer: Name of explainer ("integrated_gradients", "grad_cam", "random").
        device: Device to run on.
        indices: Optional array of indices to select from the data.
        ig_steps: Number of IG integration steps.
        baseline: Baseline for IG ("black" or "blur").
        random_seed: Seed for random attribution reproducibility.

    Returns:
        Attribution array of shape (n, 32, 32) with dtype float32.

    Raises:
        ValueError: If predictions are not cached.
    """
    from .data import load_cell_images, to_normalised_tensor

    from .cells import stage_spec

    spec = stage_spec(cell, "explain", explainer=explainer)

    # Check cache first
    if cache.has(spec, EXPLAIN_VERSION, kind="array"):
        return cache.get_array(spec, EXPLAIN_VERSION)

    # Cache miss: load images
    images, labels = load_cell_images(
        cell,
        clean_root,
        corrupt_root,
        indices=indices,
    )

    # Load predicted logits to determine target class
    # Predictions are keyed on cell alone (no explainer)
    from .predict import PREDICT_VERSION
    predict_spec = stage_spec(cell, "predict")
    if not cache.has(predict_spec, PREDICT_VERSION, kind="array"):
        raise ValueError(
            f"explain_cell requires cached predictions for cell {cell}. "
            f"Run predict stage first."
        )

    logits = cache.get_array(predict_spec, PREDICT_VERSION)
    predicted_classes = np.argmax(logits, axis=1)  # (n,)

    # Convert images to tensor
    tensor = to_normalised_tensor(images)
    tensor = tensor.to(device)

    # Compute attributions
    # Route only the options this explainer understands. explain_batch rejects
    # the rest rather than dropping them silently.
    explainer_kw = {
        "integrated_gradients": {"steps": ig_steps, "baseline": baseline},
        "grad_cam": {},
        "random": {"seed": random_seed},
    }[explainer]

    attributions = explain_batch(
        model,
        tensor,
        torch.from_numpy(predicted_classes).to(device),
        explainer,
        device=device,
        **explainer_kw,
    )

    # Attributions are the bulk of the cache (~1 GB at Version A), so they are
    # stored as float16.
    #
    # Return the ROUND-TRIPPED value, not the float32 original. Otherwise a
    # cache miss returns full precision while a cache hit returns the float16
    # version, so an uninterrupted run and a preempted-then-resumed run produce
    # different numbers from identical inputs — in a study whose whole design
    # rests on sessions being interchangeable.
    attributions_fp16 = attributions.astype(np.float16)
    cache.put_array(spec, EXPLAIN_VERSION, attributions_fp16)
    return attributions_fp16.astype(np.float32)


def ig_completeness_error(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    attribution: np.ndarray,
    baseline: str = "black",
    device: str = "cpu",
) -> float:
    """Relative completeness error of an IG approximation.

    The completeness property says: sum(attribution) ≈ f(x) - f(baseline).
    This function computes the relative error.

    Args:
        model: A PyTorch model in eval mode.
        x: Input images (n, 3, 32, 32) in normalized space.
        target: Target class index.
        attribution: IG attribution array of shape (n, 32, 32).
        baseline: "black" or "blur".
        device: Device to run on.

    Returns:
        Relative error as a float in [0, 1].
    """
    x = x.to(device)

    # Compute baseline
    if baseline == "black":
        x_baseline = torch.zeros_like(x)
    elif baseline == "blur":
        x_np = x.cpu().numpy()
        n = x.shape[0]
        x_baseline_np = np.zeros_like(x_np)
        for i in range(n):
            for c in range(3):
                x_baseline_np[i, c] = ndimage.gaussian_filter(x_np[i, c], sigma=1.5)
        x_baseline = torch.from_numpy(x_baseline_np).to(device)
    else:
        raise ValueError(f"baseline must be 'black' or 'blur', got {baseline!r}")

    model.eval()
    with torch.no_grad():
        out_x = model(x)[:, target]
        out_baseline = model(x_baseline)[:, target]

    delta = out_x - out_baseline  # (n,)
    attr_sum = attribution.sum(axis=(1, 2))  # (n,)

    # Relative error: |attr_sum - delta| / |delta|
    # Avoid division by zero
    delta_cpu = delta.cpu().numpy()
    abs_delta = np.abs(delta_cpu)
    abs_delta = np.maximum(abs_delta, 1e-8)

    error = np.abs(attr_sum - delta_cpu) / abs_delta
    return float(error.mean())
