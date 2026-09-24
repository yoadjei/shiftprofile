"""Subgroup performance disparity — tabular track only.

Deliberately NOT called fairness. The ACS labels are survey outcomes, not
decisions, so there is no decision context in which a fairness claim would
mean anything. What is measured is whether performance gaps between
demographic groups widen under shift, and how fast relative to overall accuracy.

Groups below `min_group_size` are refused rather than silently reported, because
a gap computed on a handful of rows is noise presented as a finding.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .calibration import ece_equal_mass

MIN_GROUP_SIZE = 200


def group_sizes(groups: np.ndarray) -> dict:
    return dict(Counter(np.asarray(groups).tolist()))


def _validate(groups: np.ndarray, min_group_size: int) -> list:
    sizes = group_sizes(groups)
    too_small = {g: n for g, n in sizes.items() if n < min_group_size}
    if too_small:
        raise ValueError(
            f"groups below min_group_size={min_group_size}: {too_small}. "
            "Reporting a gap on this few rows would present noise as a finding. "
            "Collapse the groups or drop the cell."
        )
    return sorted(sizes)


def worst_group_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
) -> float:
    names = _validate(groups, min_group_size)
    preds = np.asarray(probs).argmax(axis=1)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    return float(min((preds[groups == g] == labels[groups == g]).mean() for g in names))


def equal_opportunity_gap(
    probs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    min_group_size: int = MIN_GROUP_SIZE,
    positive_class: int = 1,
) -> float:
    """Largest between-group difference in true positive rate."""
    names = _validate(groups, min_group_size)
    if len(names) < 2:
        raise ValueError(f"need at least two groups to compute a gap, got {names}")
    preds = np.asarray(probs).argmax(axis=1)
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    tprs = []
    for g in names:
        mask = (groups == g) & (labels == positive_class)
        if mask.sum() == 0:
            raise ValueError(f"group {g!r} has no positive-class samples")
        tprs.append((preds[mask] == positive_class).mean())
    return float(max(tprs) - min(tprs))


def per_group_ece_gap(
    probs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 15,
    min_group_size: int = MIN_GROUP_SIZE,
) -> float:
    """Largest between-group difference in top-label ECE."""
    names = _validate(groups, min_group_size)
    if len(names) < 2:
        raise ValueError(f"need at least two groups to compute a gap, got {names}")
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    eces = [
        ece_equal_mass(probs[groups == g], labels[groups == g], n_bins=n_bins)
        for g in names
    ]
    return float(max(eces) - min(eces))
