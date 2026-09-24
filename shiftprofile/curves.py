"""Removal-based attribution faithfulness curves.

This module computes removal curves on GPU. Metrics that consume these curves
live in shiftprofile/metrics/faithfulness.py.

Key design decisions:

1. `paired_removal_curves` computes model and random-control curves in the
   same call, enforcing the structural property that a caller cannot produce
   one without the other. This is validity control #1 from the study.

2. `removal_curve` returns per-sample curves, not averaged. The bootstrap
   analysis requires sample-level data, not aggregates.

3. `imputation` is a required keyword with no default, so the scheme is
   always an explicit, recorded choice. This is validity control #2.

4. All computation runs under torch.no_grad() and in batches to minimize
   memory on Kaggle's preemptible GPU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from shiftprofile.metrics.faithfulness import VALID_IMPUTATIONS

CURVES_VERSION = "curves-v1"
REMOVAL_FRACTIONS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def impute(
    images: torch.Tensor,
    mask: torch.Tensor,
    scheme: str,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Replace masked pixels using an imputation scheme.

    Args:
        images: Tensor of shape (N, C, H, W).
        mask: Boolean tensor of shape (N, H, W); True means "impute this".
        scheme: One of ("mean", "blur", "uniform_noise", "zero").
        seed: Random seed for "uniform_noise" scheme.

    Returns:
        Tensor of same shape and dtype as images, with masked positions replaced
        and unmasked positions bit-identical to the input.

    Raises:
        ValueError: If scheme is unknown.
    """
    if scheme not in VALID_IMPUTATIONS:
        raise ValueError(
            f"unknown imputation {scheme!r}; expected one of {VALID_IMPUTATIONS}"
        )

    result = images.clone()
    mask_expanded = mask.unsqueeze(1)  # (N, 1, H, W)

    if scheme == "mean":
        # In normalized space, mean is zero. Replace with zeros.
        result = torch.where(mask_expanded, torch.tensor(0.0, dtype=images.dtype, device=images.device), result)

    elif scheme == "zero":
        # Explicitly set to zero
        result = torch.where(mask_expanded, torch.tensor(0.0, dtype=images.dtype, device=images.device), result)

    elif scheme == "blur":
        # Gaussian blur of the original image
        blurred = _gaussian_blur(images)
        result = torch.where(mask_expanded, blurred, result)

    elif scheme == "uniform_noise":
        # Seeded uniform noise in [-1, 1] (roughly the range of normalized images)
        rng = np.random.RandomState(seed)
        noise = torch.from_numpy(
            rng.uniform(-1.0, 1.0, size=images.shape)
        ).to(images.dtype).to(images.device)
        result = torch.where(mask_expanded, noise, result)

    return result


def _gaussian_blur(images: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur to a batch of images.

    Args:
        images: Tensor of shape (N, C, H, W).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred tensor of same shape.
    """
    # Create Gaussian kernel
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # Create 2D kernel
    kernel_1d = gauss.view(-1, 1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    kernel_2d = kernel_2d.to(images.device)

    # Apply separable convolution: blur in H, then blur in W
    N, C, H, W = images.shape
    # Reshape for grouped convolution
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    # Use padding to maintain size
    pad = kernel_size // 2
    blurred = F.conv2d(
        images,
        kernel,
        groups=C,
        padding=pad,
    )

    return blurred


def removal_curve(
    model: torch.nn.Module,
    images: torch.Tensor,
    attributions: torch.Tensor,
    targets: torch.Tensor,
    *,
    imputation: str,
    fractions: tuple[float, ...] = REMOVAL_FRACTIONS,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Compute the removal curve for a batch of images.

    The curve shows how predicted probability of the target class degrades as
    we remove pixels ranked by attribution magnitude.

    Args:
        model: A PyTorch model in eval mode.
        images: Tensor of shape (N, 3, H, W).
        attributions: Tensor of shape (N, H, W) with attribution scores.
        targets: Tensor of shape (N,) with target class indices.
        imputation: REQUIRED keyword. Imputation scheme for removed pixels.
                   One of ("mean", "blur", "uniform_noise", "zero").
        fractions: Tuple of removal fractions to evaluate (default: REMOVAL_FRACTIONS).
        batch_size: Batch size for forward passes (default: 256).
        device: Device to run on (default: "cpu").

    Returns:
        Array of shape (N, len(fractions)) with dtype float32. Each row is the
        predicted probability assigned to targets[i] after removing the top
        f fraction of pixels (ranked by |attribution|).

    Raises:
        ValueError: If imputation scheme is unknown.
    """
    if imputation not in VALID_IMPUTATIONS:
        raise ValueError(
            f"unknown imputation {imputation!r}; expected one of {VALID_IMPUTATIONS}"
        )

    model = model.to(device).eval()
    images = images.to(device)
    attributions = attributions.to(device)
    targets = targets.to(device)

    n_samples = images.shape[0]
    n_fractions = len(fractions)
    curves = np.zeros((n_samples, n_fractions), dtype=np.float32)

    # Sanity check: fractions[0] must be 0.0
    if fractions[0] != 0.0:
        raise ValueError(
            f"fractions[0] must be 0.0 (nothing removed), got {fractions[0]}"
        )

    with torch.no_grad():
        # Process in batches
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_size_actual = batch_end - batch_start

            images_batch = images[batch_start:batch_end]  # (B, 3, H, W)
            attr_batch = attributions[batch_start:batch_end]  # (B, H, W)
            targets_batch = targets[batch_start:batch_end]  # (B,)

            # For each removal fraction
            for frac_idx, frac in enumerate(fractions):
                if frac == 0.0:
                    # No removal: just get the baseline probability
                    logits = model(images_batch)
                    probs = torch.softmax(logits, dim=1)
                    batch_probs = probs[range(batch_size_actual), targets_batch]
                else:
                    # Create mask for top frac of pixels
                    mask = _get_removal_mask(attr_batch, frac)
                    # Impute masked pixels
                    modified_images = impute(
                        images_batch, mask, imputation, seed=0
                    )
                    # Forward pass
                    logits = model(modified_images)
                    probs = torch.softmax(logits, dim=1)
                    batch_probs = probs[range(batch_size_actual), targets_batch]

                curves[batch_start:batch_end, frac_idx] = batch_probs.cpu().numpy()

    return curves


def _get_removal_mask(
    attributions: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Create a boolean mask for the top fraction of pixels by |attribution|.

    Ties are broken deterministically by position (row-major order).

    Args:
        attributions: Tensor of shape (N, H, W).
        fraction: Fraction to remove, in [0, 1].

    Returns:
        Boolean tensor of shape (N, H, W); True means "remove this pixel".
    """
    N, H, W = attributions.shape
    abs_attr = torch.abs(attributions)

    # Flatten spatial dimensions
    flat_attr = abs_attr.view(N, H * W)  # (N, H*W)

    # Number of pixels to remove per sample
    n_remove = int(np.ceil(fraction * H * W))

    if n_remove == 0:
        return torch.zeros(N, H, W, dtype=torch.bool, device=attributions.device)

    if n_remove >= H * W:
        return torch.ones(N, H, W, dtype=torch.bool, device=attributions.device)

    # For each sample, find the threshold for the top n_remove pixels
    # Using topk with ties broken by position
    indices_flat = torch.arange(H * W, device=attributions.device).unsqueeze(0)
    indices_flat = indices_flat.expand(N, -1)

    # Create a compound key: (neg_attribution, position) for stable sorting
    # topk will sort by the first component, breaking ties by the second
    neg_attr_for_sort = -flat_attr
    # Offset position so that it's much smaller than attribution magnitudes
    # This ensures attribution is the primary key, position is the tiebreaker
    position_key = indices_flat.float() / (H * W)  # [0, 1)

    compound_key = neg_attr_for_sort + position_key / (H * W)

    # Get the top n_remove indices
    _, top_indices = torch.topk(compound_key, k=n_remove, dim=1, largest=True)

    # Create mask
    mask = torch.zeros(N, H * W, dtype=torch.bool, device=attributions.device)
    for i in range(N):
        mask[i, top_indices[i]] = True

    # Reshape back to (N, H, W)
    mask = mask.view(N, H, W)
    return mask


def paired_removal_curves(
    model: torch.nn.Module,
    images: torch.Tensor,
    attributions: torch.Tensor,
    targets: torch.Tensor,
    *,
    imputation: str,
    random_seed: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute paired model and random-attribution control curves.

    This function enforces the structural property that a caller cannot produce
    a model curve without its paired random-attribution control. Both are
    computed in the same call, refusing to return one without the other. This
    is validity control #1 from the study.

    Args:
        model: A PyTorch model in eval mode.
        images: Tensor of shape (N, 3, H, W).
        attributions: Tensor of shape (N, H, W) with attribution scores.
        targets: Tensor of shape (N,) with target class indices.
        imputation: REQUIRED keyword. Imputation scheme for removed pixels.
        random_seed: REQUIRED keyword. Seed for random attribution generation.
        **kwargs: Passed to removal_curve (batch_size, fractions, device).

    Returns:
        Tuple (model_curve, random_curve), each of shape (N, len(fractions)).

    Raises:
        ValueError: If imputation scheme is unknown.
        TypeError: If imputation or random_seed is missing.
    """
    if imputation not in VALID_IMPUTATIONS:
        raise ValueError(
            f"unknown imputation {imputation!r}; expected one of {VALID_IMPUTATIONS}"
        )

    # Compute model attribution curve
    model_curve = removal_curve(
        model, images, attributions, targets,
        imputation=imputation,
        **kwargs,
    )

    # Generate random attribution as a control
    rng = np.random.RandomState(random_seed)
    random_attributions = torch.from_numpy(
        rng.randn(*attributions.shape)
    ).to(attributions.dtype).to(attributions.device)

    # Compute random attribution curve
    random_curve = removal_curve(
        model, images, random_attributions, targets,
        imputation=imputation,
        **kwargs,
    )

    return model_curve, random_curve


def curves_cell(
    cell,  # Cell object
    model: torch.nn.Module,
    clean_root: str,
    corrupt_root: str,
    cache,  # ArtifactCache
    *,
    explainer: str,
    imputation: str,
    device: str = "cpu",
    indices: np.ndarray | None = None,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute removal curves for a cell, checking cache first.

    Curves are cached by (cell, explainer, imputation) to enforce the
    structural property that different explainers and imputation schemes
    produce different results under different keys.

    Reads attributions from the explain stage; raises if absent.

    Args:
        cell: A Cell object.
        model: A trained model.
        clean_root: Root directory for clean CIFAR-10 data.
        corrupt_root: Root directory for CIFAR-10-C data.
        cache: ArtifactCache for storing/loading curves.
        explainer: Name of explainer that produced the attributions.
        imputation: Imputation scheme for removed pixels.
        device: Device to run on.
        indices: Optional array of indices to select from the data.
        random_seed: Seed for random control generation.

    Returns:
        Tuple (model_curve, random_curve), each of shape (n_samples, n_fractions).

    Raises:
        ValueError: If attributions are not cached.
    """
    from .data import load_cell_images, to_normalised_tensor
    from .explain import EXPLAIN_VERSION

    # Build cache keys for both model and random control curves
    from .cells import stage_spec

    model_curve_spec = stage_spec(
        cell, "curves", explainer=explainer, imputation=imputation, which="model"
    )
    random_curve_spec = stage_spec(
        cell, "curves", explainer=explainer, imputation=imputation, which="random"
    )

    # Check cache for both curves
    model_cached = cache.has(model_curve_spec, CURVES_VERSION, kind="array")
    random_cached = cache.has(random_curve_spec, CURVES_VERSION, kind="array")

    if model_cached and random_cached:
        # Both cached: return them
        model_curve = cache.get_array(model_curve_spec, CURVES_VERSION)
        random_curve = cache.get_array(random_curve_spec, CURVES_VERSION)
        return model_curve, random_curve

    # Cache miss: load images
    images, labels = load_cell_images(
        cell,
        clean_root,
        corrupt_root,
        indices=indices,
    )

    # Load attributions from explain stage
    explain_spec = stage_spec(cell, "explain", explainer=explainer)
    if not cache.has(explain_spec, EXPLAIN_VERSION, kind="array"):
        raise ValueError(
            f"curves_cell requires cached attributions for cell {cell} and "
            f"explainer {explainer!r}. Run explain stage first."
        )

    # Load predictions to determine target class
    from .predict import PREDICT_VERSION
    predict_spec = stage_spec(cell, "predict")
    if not cache.has(predict_spec, PREDICT_VERSION, kind="array"):
        raise ValueError(
            f"curves_cell requires cached predictions for cell {cell}. "
            f"Run predict stage first."
        )

    logits = cache.get_array(predict_spec, PREDICT_VERSION)
    predicted_classes = np.argmax(logits, axis=1)

    # Load attributions (stored as float16, convert back to float32)
    attributions_fp16 = cache.get_array(explain_spec, EXPLAIN_VERSION)
    attributions = attributions_fp16.astype(np.float32)

    # Convert to tensors
    tensor = to_normalised_tensor(images)
    tensor = tensor.to(device)
    attributions_tensor = torch.from_numpy(attributions).to(device)
    targets_tensor = torch.from_numpy(predicted_classes).to(device)

    # Compute paired curves
    model_curve, random_curve = paired_removal_curves(
        model,
        tensor,
        attributions_tensor,
        targets_tensor,
        imputation=imputation,
        random_seed=random_seed,
        device=device,
    )

    # Store both curves (float32)
    cache.put_array(model_curve_spec, CURVES_VERSION, model_curve)
    cache.put_array(random_curve_spec, CURVES_VERSION, random_curve)

    return model_curve, random_curve
