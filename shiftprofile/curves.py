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

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from shiftprofile.data.cifar import CIFAR10_MEAN, CIFAR10_STD
from shiftprofile.metrics.faithfulness import (
    RETIRED_IMPUTATIONS,
    VALID_IMPUTATIONS,
)

# Bumped alongside explain-v2: a curve's key now names the evaluation index set,
# the two upstream versions it reads, and which attributions it was built from.
# Curves written under curves-v1 recorded none of that, so they cannot be told
# apart from curves over a different eval set or a different IG baseline.
CURVES_VERSION = "curves-v2"
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
    if scheme in RETIRED_IMPUTATIONS:
        raise ValueError(
            f"imputation {scheme!r} was retired: {RETIRED_IMPUTATIONS[scheme]}"
        )
    if scheme not in VALID_IMPUTATIONS:
        raise ValueError(
            f"unknown imputation {scheme!r}; expected one of {VALID_IMPUTATIONS}"
        )

    result = images.clone()
    mask_expanded = mask.unsqueeze(1)  # (N, 1, H, W)

    if scheme == "mean":
        # The dataset mean in pixel space is exactly 0 after normalisation, so
        # filling with zeros IS mean imputation here. Correct, and the reason
        # `zero` had to go: it ran this same line and silently duplicated it.
        result = torch.where(mask_expanded, torch.tensor(0.0, dtype=images.dtype, device=images.device), result)

    elif scheme == "black":
        # Black in PIXEL space, which normalisation maps to -mean/std, about
        # -1.99, -1.98, -1.71 per channel -- not zero. The scheme this replaces
        # was named `zero` and set normalised zero, which is the mean, so E6 ran
        # three distinct arms while reporting four and the duplicate inflated the
        # gate's denominator. Verified by the sweep: `mean` and `zero` agreed to
        # five decimals on all 120 curves, which two schemes cannot do by chance.
        black = torch.tensor(
            [-m / s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
            dtype=images.dtype, device=images.device,
        ).view(1, -1, 1, 1)
        result = torch.where(mask_expanded, black.expand_as(images), result)

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


def curves_spec(
    cell,
    *,
    explainer: str,
    imputation: str,
    which: str,
    indices: np.ndarray,
    control_seed: int = 0,
    explain_options: Optional[dict] = None,
) -> dict:
    """The cache key for one of this cell's two removal curves.

    A curve is identified by `which`: the model's attribution or the random
    control. Both are written per work unit, which is why `curves_is_cached`
    exists -- `fill` once built this key WITHOUT `which` and so could never find
    either artifact, reporting thirty curves "completed" every resumed run while
    `curves_cell` short-circuited and did nothing.

    `inputs` names both upstream versions read. Attributions are the input the
    curve is computed from, so an `explain` bump must invalidate the curve; it
    previously did not, which left curves derived from attributions that had been
    recomputed out from under them.

    `explain_options` is in `explain_spec`'s vocabulary (`ig_steps`, `baseline`,
    `random_seed`) and is normalised through `explainer_options` before going into
    the key, so two spellings of the same settings hash to one artifact and the
    dict recorded here is the same one `explain_spec` recorded.
    """
    from .cells import stage_spec
    from .data import eval_digest
    from .explain import EXPLAIN_VERSION, explainer_options
    from .predict import PREDICT_VERSION

    if which not in ("model", "random"):
        raise ValueError(f"which must be 'model' or 'random', got {which!r}")

    return stage_spec(
        cell,
        "curves",
        explainer=explainer,
        imputation=imputation,
        which=which,
        # The random control curve is drawn from this seed, so it determines the
        # contents of the artifact that `relative_faithfulness` subtracts. Named
        # `control_seed`, not `random_seed`: explain_cell has a `random_seed` too
        # and it means something else entirely -- the seed of the `random`
        # EXPLAINER. Two different quantities under one name in adjacent stages is
        # a mistake waiting to be made.
        control_seed=control_seed,
        # Which attributions this curve was computed from. A curve built on
        # black-baseline attributions is not the curve built on blur-baseline
        # ones, and E6 varies exactly that, so without this the ablation's curves
        # would collide with the frozen protocol's.
        explain_options=explainer_options(explainer, **(explain_options or {})),
        evaluated=eval_digest(indices),
        inputs={"explain": EXPLAIN_VERSION, "predict": PREDICT_VERSION},
    )


def curves_is_cached(
    cell,
    cache,
    *,
    explainer: str,
    imputation: str,
    indices: np.ndarray,
    control_seed: int = 0,
    explain_options: Optional[dict] = None,
) -> bool:
    """Whether BOTH of this unit's curves are on disk.

    One of the two is not a result. `relative_faithfulness` is the random
    control's AUC minus the model's, so a unit holding only one curve has nothing
    to report and must be recomputed.
    """
    return all(
        cache.has(
            curves_spec(
                cell,
                explainer=explainer,
                imputation=imputation,
                which=which,
                indices=indices,
                control_seed=control_seed,
                explain_options=explain_options,
            ),
            CURVES_VERSION,
            kind="array",
        )
        for which in ("model", "random")
    )


def curves_cell(
    cell,  # Cell object
    model: torch.nn.Module,
    clean_root: str,
    corrupt_root: str,
    cache,  # ArtifactCache
    *,
    explainer: str,
    imputation: str,
    indices: np.ndarray,
    device: str = "cpu",
    control_seed: int = 0,
    explain_options: Optional[dict] = None,
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
        indices: REQUIRED. The images to evaluate, from
                 `fixed_eval_indices(n_eval_images)`. Part of the cache key, and
                 must be the same set the attributions were computed over.
        device: Device to run on.
        random_seed: Seed for random control generation.

    Returns:
        Tuple (model_curve, random_curve), each of shape (len(indices), n_fractions).

    Raises:
        ValueError: If attributions are not cached.
    """
    from .data import load_cell_images, to_normalised_tensor
    from .explain import EXPLAIN_VERSION, explain_spec

    key = dict(
        explainer=explainer,
        imputation=imputation,
        indices=indices,
        control_seed=control_seed,
        explain_options=explain_options,
    )
    model_curve_spec = curves_spec(cell, which="model", **key)
    random_curve_spec = curves_spec(cell, which="random", **key)

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

    # Read both upstream artifacts through their OWN key builders. The versions
    # these two names resolve to are the same ones recorded in `inputs` above, so
    # what the curve was derived from and what its key claims cannot drift apart.
    from .predict import PREDICT_VERSION, predict_spec

    # Named explicitly rather than rebuilt from defaults. curves_cell once looked
    # up attributions with explain's DEFAULT options, so a run that attributed at
    # 4 IG steps had its curves look for the 32-step artifact and miss forever.
    attribution_key = explain_spec(
        cell, explainer=explainer, indices=indices, **(explain_options or {})
    )
    if not cache.has(attribution_key, EXPLAIN_VERSION, kind="array"):
        raise ValueError(
            f"curves_cell requires cached attributions for cell {cell}, explainer "
            f"{explainer!r}, over the same {len(indices)} evaluation images. Run "
            f"the explain stage first with the same n_eval_images."
        )

    prediction_key = predict_spec(cell, indices=indices)
    if not cache.has(prediction_key, PREDICT_VERSION, kind="array"):
        raise ValueError(
            f"curves_cell requires cached predictions for cell {cell} over the "
            f"same {len(indices)} evaluation images. Run the predict stage first "
            f"with the same n_eval_images."
        )

    logits = cache.get_array(prediction_key, PREDICT_VERSION)
    predicted_classes = np.argmax(logits, axis=1)

    # Load attributions (stored as float16, convert back to float32)
    attributions_fp16 = cache.get_array(attribution_key, EXPLAIN_VERSION)
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
        random_seed=control_seed,
        device=device,
    )

    # Store both curves (float32)
    cache.put_array(model_curve_spec, CURVES_VERSION, model_curve)
    cache.put_array(random_curve_spec, CURVES_VERSION, random_curve)

    return model_curve, random_curve
