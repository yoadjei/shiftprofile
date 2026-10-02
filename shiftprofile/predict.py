"""Prediction stage for evaluating models on CIFAR-10 and CIFAR-10-C.

Produces logits (not probabilities) so temperature scaling can be applied.
Caches logits by cell to avoid recomputation on Kaggle preemption.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .cache import ArtifactCache
from .cells import Cell
from .cells import stage_spec
from .data import load_cell_images, to_normalised_tensor

PREDICT_VERSION = "predict-v1"


def predict_logits(
    model: nn.Module,
    images: np.ndarray,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Run a model on images and return logits.

    Args:
        model: A trained model (nn.Module).
        images: Array of shape (n, 32, 32, 3) with dtype uint8.
        batch_size: Batch size for inference (default 512).
        device: Device to run on ("cpu" or "cuda").

    Returns:
        Logits array of shape (n, 10) with dtype float32.
        NOT probabilities — softmax is not applied.
    """
    model.eval()
    model = model.to(device)

    # Convert images to normalized tensors
    tensor = to_normalised_tensor(images)

    # Flatten if this is a Sequential model with Linear first layer (test stub)
    if isinstance(model, nn.Sequential) and isinstance(model[0], nn.Linear):
        tensor = tensor.flatten(1)

    # Convert entire tensor to device at once
    tensor = tensor.to(device)

    # Run inference in batches - all at once to avoid floating point ordering issues
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i : i + batch_size]
            logits = model(batch)
            all_logits.append(logits.detach().cpu().numpy().astype(np.float32))

    # Concatenate all batches
    logits = np.concatenate(all_logits, axis=0)

    return logits


def predict_spec(cell: Cell, *, indices: np.ndarray) -> dict:
    """The cache key for this cell's logits. The only place it is built.

    `fill` used to construct this itself, which is how a key and the artifact
    written under it drifted apart. Each stage module now owns its own key, so
    there is one definition to get right rather than two to keep in step.
    """
    from .data import eval_digest

    return stage_spec(cell, "predict", evaluated=eval_digest(indices))


def predict_is_cached(cell: Cell, cache: ArtifactCache, *, indices: np.ndarray) -> bool:
    """Whether this cell's logits are already on disk.

    `fill` asks this instead of building a key and calling `cache.has` itself.
    """
    return cache.has(predict_spec(cell, indices=indices), PREDICT_VERSION, kind="array")


def predict_cell(
    cell: Cell,
    model: nn.Module,
    clean_root: str,
    corrupt_root: str,
    cache: ArtifactCache,
    *,
    indices: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Predict logits for a cell, checking cache first.

    Args:
        cell: A Cell object (track, model_id, seed, shift_family, severity).
        model: A trained model.
        clean_root: Root directory for clean CIFAR-10 data.
        corrupt_root: Root directory for CIFAR-10-C data.
        cache: ArtifactCache for storing/loading logits.
        indices: REQUIRED. The images to evaluate, from
                 `fixed_eval_indices(n_eval_images)`. Part of the cache key, so
                 logits over different index sets cannot collide. It was once
                 optional and defaulted to None, which meant every run silently
                 scored the whole 10,000-image test set while the configs asked
                 for 1,000 and the key recorded neither.
        batch_size: Batch size for inference.
        device: Device to run on.

    Returns:
        Logits array of shape (len(indices), 10) with dtype float32.
    """
    spec = predict_spec(cell, indices=indices)

    # Check cache first
    if cache.has(spec, PREDICT_VERSION, kind="array"):
        return cache.get_array(spec, PREDICT_VERSION)

    # Cache miss: compute logits
    images, labels = load_cell_images(
        cell,
        clean_root,
        corrupt_root,
        indices=indices,
    )

    logits = predict_logits(
        model,
        images,
        batch_size=batch_size,
        device=device,
    )

    # Store in cache
    cache.put_array(spec, PREDICT_VERSION, logits)

    return logits
