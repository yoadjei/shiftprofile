"""Training stage for vision models on CIFAR-10.

Produces trained models and metadata records cached by seed.
Implements deterministic training with seeded RNG, one-cycle LR schedule, and AMP.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR

from .cache import ArtifactCache
from .data import load_cifar10_train, load_cifar10_test, to_normalised_tensor
from .models import build_model
from .models.vision import uses_augmix

TRAIN_VERSION = "train-v2"

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 0.1


def train_spec(
    model_id: str,
    seed: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    """Cache key for a trained checkpoint.

    Every value that determines the weights is here. `model_id` and `seed` were
    the whole key through `train-v1`, which meant a config changing `epochs` from
    50 to 10 would silently be served the 50-epoch checkpoint -- `fill.main`
    reads `epochs` and `batch_size` straight out of the config, so that was one
    edit away at all times. This is the eighth instance of that defect class in
    this project and it is closed the same way as the other seven: name the
    value in the key.

    The hyperparameters are required rather than defaulted. A default here would
    let a caller that forgot to pass `epochs` file a 10-epoch checkpoint under a
    key asserting 50, which is the defect wearing a different hat.

    `augmix` is deliberately absent: it is a function of `model_id`
    (`models.vision.uses_augmix`), not an independent axis, so naming it would be
    naming the same thing twice and would admit keys that contradict themselves.

    Args:
        model_id: Name of the model (e.g., "resnet18", "resnet18_augmix", "vit_tiny").
        seed: Random seed used for training.
        epochs: Number of training epochs the checkpoint was trained for.
        batch_size: Training batch size used.
        lr: Peak learning rate of the one-cycle schedule.

    Returns:
        A dictionary suitable as a cache spec.
    """
    return {
        "model_id": model_id,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }


def train_model(
    model_id: str,
    seed: int,
    data_root: Path | str,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "cpu",
    progress: Optional[Callable[[int, int, dict], None]] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a vision model on CIFAR-10.

    AugMix is not a parameter. It is read off `model_id` via `uses_augmix`,
    because `resnet18_augmix` means exactly "resnet18 trained with AugMix" and
    nothing else -- `build_model` gives both ids the same architecture. It used
    to be a keyword defaulting to False that no caller ever set, so the
    pre-registered AugMix arm trained identically to plain `resnet18` and
    differed from it only in its cache key.

    Args:
        model_id: Name of the model to train. Also determines AugMix.
        seed: Random seed for deterministic training.
        data_root: Root directory for CIFAR-10 data.
        epochs: Number of training epochs (default 50).
        batch_size: Training batch size (default 256).
        lr: Learning rate (default 0.1).
        device: Device to train on ("cpu" or "cuda").
        progress: Optional callback for progress reporting.

    Returns:
        (model, metadata): Trained model and dict containing:
            - seed, epochs, batch_size, lr, augmix, final_train_loss
            - clean_test_accuracy (float in [0, 1])
            - wall_clock_seconds, torch_version
    """
    augmix = uses_augmix(model_id)

    # Seed all RNGs for determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()

    # Build model
    model = build_model(model_id, num_classes=10)
    model = model.to(device)

    # Load training data
    train_dataset = load_cifar10_train(data_root, augmix=augmix)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
    )

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    # One-cycle LR schedule
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=epochs * len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    final_train_loss = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision if on CUDA
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        # Track final loss
        if num_batches > 0:
            final_train_loss = total_loss / num_batches

        # Optional progress callback
        if progress is not None:
            progress(epoch + 1, epochs, {"loss": final_train_loss})

    # Compute clean test accuracy
    model.eval()

    test_images, test_labels = load_cifar10_test(data_root)
    test_tensor = to_normalised_tensor(test_images).to(device)
    test_labels_tensor = torch.from_numpy(test_labels).to(device)

    with torch.no_grad():
        test_logits = model(test_tensor)
        test_preds = test_logits.argmax(dim=1)
        correct = (test_preds == test_labels_tensor).sum().item()
        clean_test_accuracy = correct / len(test_labels)

    wall_clock_seconds = time.time() - start_time

    metadata = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "augmix": augmix,
        "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
        "clean_test_accuracy": float(clean_test_accuracy),
        "wall_clock_seconds": float(wall_clock_seconds),
        "torch_version": torch.__version__,
    }

    return model, metadata


def load_or_train(
    model_id: str,
    seed: int,
    data_root: Path | str,
    cache: ArtifactCache,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str = "cpu",
    **kw,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained model from cache or train it.

    Checks the cache first for a checkpoint. On miss, trains a new model,
    stores the checkpoint and metadata in the cache, and returns it.

    The hyperparameters are named here rather than swept up in `**kw` because
    this function is the one place the cache key and the training call have to
    agree. Passed through `**kw` they would reach `train_model` and not
    `train_spec`, so the checkpoint would be filed under a key describing
    different hyperparameters than the ones it was trained with.

    Args:
        model_id: Name of the model to train. Also determines AugMix.
        seed: Random seed for training.
        data_root: Root directory for CIFAR-10 data.
        cache: ArtifactCache instance for storing/loading checkpoints.
        epochs: Number of training epochs. In the key.
        batch_size: Training batch size. In the key.
        lr: Peak learning rate. In the key.
        device: Device to train on ("cpu" or "cuda"). Not in the key: it changes
            arithmetic order, not the training recipe.
        **kw: Forwarded to train_model (currently `progress`).

    Returns:
        (model, metadata): Trained model and metadata dict.
    """
    spec = train_spec(
        model_id, seed, epochs=epochs, batch_size=batch_size, lr=lr
    )

    # Check cache for existing checkpoint
    if cache.has(spec, TRAIN_VERSION, kind="record"):
        # Load from cache
        metadata = cache.get_record(spec, TRAIN_VERSION)
        checkpoint_path = cache.resolve(spec, TRAIN_VERSION, kind="record")
        # The checkpoint is stored alongside the metadata as .pt
        checkpoint_path_pt = checkpoint_path.parent / checkpoint_path.stem.replace(".json", ".pt")
        if checkpoint_path_pt.exists():
            model = build_model(model_id, num_classes=10)
            state_dict = torch.load(checkpoint_path_pt, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            return model, metadata

    # Cache miss: train new model
    model, metadata = train_model(
        model_id,
        seed,
        data_root,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        **kw,
    )

    # Store checkpoint and metadata in cache
    # Get the path where metadata will be stored, then save checkpoint alongside it
    from .cache import spec_key

    metadata_key = spec_key(spec, TRAIN_VERSION)
    checkpoint_path = cache.root / f"{metadata_key}.pt"

    # Save checkpoint and metadata
    torch.save(model.state_dict(), checkpoint_path)
    cache.put_record(spec, TRAIN_VERSION, metadata)

    return model, metadata
