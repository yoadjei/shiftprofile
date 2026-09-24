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


def predict_cell(
    cell: Cell,
    model: nn.Module,
    clean_root: str,
    corrupt_root: str,
    cache: ArtifactCache,
    *,
    batch_size: int = 512,
    device: str = "cpu",
    indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Predict logits for a cell, checking cache first.

    Args:
        cell: A Cell object (track, model_id, seed, shift_family, severity).
        model: A trained model.
        clean_root: Root directory for clean CIFAR-10 data.
        corrupt_root: Root directory for CIFAR-10-C data.
        cache: ArtifactCache for storing/loading logits.
        batch_size: Batch size for inference.
        device: Device to run on.
        indices: Optional array of indices to select from the data.

    Returns:
        Logits array of shape (n, 10) with dtype float32.
    """
    spec = stage_spec(cell, "predict")

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
