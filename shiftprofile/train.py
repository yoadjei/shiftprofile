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

TRAIN_VERSION = "train-v1"


def train_spec(model_id: str, seed: int) -> dict[str, Any]:
    """Cache key for a trained checkpoint.

    Args:
        model_id: Name of the model (e.g., "resnet18", "resnet18_augmix", "vit_tiny").
        seed: Random seed used for training.

    Returns:
        A dictionary suitable as a cache spec.
    """
    return {
        "model_id": model_id,
        "seed": seed,
    }


def train_model(
    model_id: str,
    seed: int,
    data_root: Path | str,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.1,
    device: str = "cpu",
    augmix: bool = False,
    progress: Optional[Callable[[int, int, dict], None]] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a vision model on CIFAR-10.

    Args:
        model_id: Name of the model to train.
        seed: Random seed for deterministic training.
        data_root: Root directory for CIFAR-10 data.
        epochs: Number of training epochs (default 50).
        batch_size: Training batch size (default 256).
        lr: Learning rate (default 0.1).
        device: Device to train on ("cpu" or "cuda").
        augmix: Whether to use AugMix augmentation (default False).
        progress: Optional callback for progress reporting.

    Returns:
        (model, metadata): Trained model and dict containing:
            - seed, epochs, final_train_loss
            - clean_test_accuracy (float in [0, 1])
            - wall_clock_seconds, torch_version
    """
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
    device: str = "cpu",
    **kw,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained model from cache or train it.

    Checks the cache first for a checkpoint. On miss, trains a new model,
    stores the checkpoint and metadata in the cache, and returns it.

    Args:
        model_id: Name of the model to train.
        seed: Random seed for training.
        data_root: Root directory for CIFAR-10 data.
        cache: ArtifactCache instance for storing/loading checkpoints.
        device: Device to train on ("cpu" or "cuda").
        **kw: Additional kwargs passed to train_model (epochs, batch_size, lr, etc.).

    Returns:
        (model, metadata): Trained model and metadata dict.
    """
    spec = train_spec(model_id, seed)

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
