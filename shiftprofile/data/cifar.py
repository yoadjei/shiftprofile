"""CIFAR-10 and CIFAR-10-C data loaders with committed evaluation indices.

Why fixed indices matter: Every model must be scored on identical inputs at every
severity, or the comparison is confounded. The indices are generated once from a
fixed seed and committed to the repository, so they cannot drift between sessions
or machines.

CIFAR-10-C format:
  - One .npy per corruption, e.g. `gaussian_noise.npy`, shape (50000, 32, 32, 3), dtype uint8.
  - It is 5 severities stacked: severity s (1-indexed) occupies rows [(s-1)*10000 : s*10000].
  - `labels.npy` has shape (50000,) and is the same 10000 clean test labels tiled 5 times.
  - Row i within a severity block corresponds to clean test image i.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from torchvision import datasets, transforms

# Corruption families by type
CORRUPTION_FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightening"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}

# Pilot set: a small, representative subset of corruptions
PILOT_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog")

# CIFAR-10 normalization constants (empirical mean and std from train set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def fixed_eval_indices(n: int, total: int = 10000, seed: int = 20260923) -> np.ndarray:
    """Generate deterministic evaluation indices for CIFAR-10 test set.

    The same n and seed always produce identical results, ensuring reproducibility
    across machines and sessions. Indices for smaller n are prefixes of indices for
    larger n, so subsets remain comparable.

    Args:
        n: Number of indices to generate.
        total: Total size of the dataset to sample from (default 10000 for CIFAR-10 test set).
        seed: Random seed for reproducibility.

    Returns:
        Array of n unique indices in [0, total).
    """
    if n > total:
        raise ValueError(f"n ({n}) cannot exceed total ({total})")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(total)[:n]
    return indices


def load_cifar10_test(root: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 test set.

    Uses torchvision's CIFAR10 dataset with download support.

    Args:
        root: Root directory for dataset (torchvision will create cifar-10-batches-py subdirectory).

    Returns:
        (images, labels): uint8 array (10000, 32, 32, 3) and int64 array (10000,).
    """
    dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Extract raw images and labels
    images = []
    labels = []
    for img, label in dataset:
        # Convert back from tensor to numpy, undo ToTensor normalization
        img_np = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        images.append(img_np)
        labels.append(label)

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64)


def load_cifar10_train(root: Path | str, augmix: bool = False) -> torch.utils.data.Dataset:
    """Load CIFAR-10 training set with optional AugMix augmentation.

    Args:
        root: Root directory for dataset.
        augmix: Whether to apply AugMix augmentation.

    Returns:
        torch.utils.data.Dataset with CIFAR-10 training data.
    """
    if augmix:
        transform = transforms.Compose([
            transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    dataset = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=transform,
    )

    return dataset


def load_cifar10c(
    root: Path | str,
    corruption: str,
    severity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10-C corrupted test set at a specific severity.

    Args:
        root: Root directory containing the CIFAR-10-C directory.
        corruption: Name of the corruption (e.g., 'gaussian_noise', 'fog').
        severity: Severity level [1, 5].

    Returns:
        (images, labels): uint8 array (10000, 32, 32, 3) and int64 array (10000,).

    Raises:
        ValueError: If severity is not in [1, 5] or corruption is unknown.
    """
    if severity < 1 or severity > 5:
        raise ValueError(
            f"severity must be in [1, 5], got {severity}"
        )

    # Validate corruption name
    valid_corruptions = []
    for family_corruptions in CORRUPTION_FAMILIES.values():
        valid_corruptions.extend(family_corruptions)

    if corruption not in valid_corruptions:
        raise ValueError(
            f"unknown corruption {corruption!r}. Available: {', '.join(sorted(valid_corruptions))}"
        )

    root_path = Path(root)
    cifar10c_dir = root_path / "CIFAR-10-C"

    # Load the full corruption data (50000 rows, 5 severities stacked)
    corruption_file = cifar10c_dir / f"{corruption}.npy"
    labels_file = cifar10c_dir / "labels.npy"

    corruption_data = np.load(corruption_file)
    labels_data = np.load(labels_file)

    # Extract the 10000 images for this severity
    # Severity 1 -> rows [0:10000], severity 2 -> rows [10000:20000], etc.
    start_row = (severity - 1) * 10000
    end_row = severity * 10000

    images = corruption_data[start_row:end_row]
    labels = labels_data[start_row:end_row]

    return images, labels


def load_cell_images(
    cell,  # Cell object
    clean_root: Path | str,
    corrupt_root: Path | str,
    indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images for a cell, dispatching on shift family.

    This is the single entry point the pipeline uses, so a producer cannot
    accidentally load the wrong thing.

    Args:
        cell: Cell object with track, model_id, seed, shift_family, severity.
        clean_root: Root directory for clean CIFAR-10 data.
        corrupt_root: Root directory for CIFAR-10-C data.
        indices: Optional array of indices to select. If provided, only these
                 rows are returned.

    Returns:
        (images, labels): Arrays of images and labels for this cell.
    """
    from shiftprofile.cells import CLEAN

    if cell.shift_family == CLEAN:
        # Load clean test set
        images, labels = load_cifar10_test(clean_root)
    else:
        # Load corrupted set at the specified severity
        images, labels = load_cifar10c(corrupt_root, cell.shift_family, cell.severity)

    # Apply index selection if provided
    if indices is not None:
        images = images[indices]
        labels = labels[indices]

    return images, labels


def to_normalised_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC images to normalized float32 NCHW tensors.

    Args:
        images: Array of shape (N, 32, 32, 3) with dtype uint8.

    Returns:
        Tensor of shape (N, 3, 32, 32) with dtype float32, normalized by
        CIFAR10_MEAN and CIFAR10_STD.
    """
    # Convert to float in [0, 1]
    tensor = torch.from_numpy(images.astype(np.float32) / 255.0)

    # Transpose from NHWC to NCHW
    tensor = tensor.permute(0, 3, 1, 2)

    # Apply normalization
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor


def denormalise(t: torch.Tensor) -> torch.Tensor:
    """Reverse the normalization applied by to_normalised_tensor.

    Args:
        t: Normalized tensor in NCHW format.

    Returns:
        Tensor with normalization reversed, approximately in [0, 1] range.
    """
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return t * std + mean
