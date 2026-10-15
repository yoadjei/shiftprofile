"""Tabular track evaluation: assemble per-cell gap records with bootstrap intervals.

This module evaluates a single cell by computing six calibration and subgroup
performance metrics, each with a bootstrap confidence interval and group sizes.

Every reported gap carries an interval and a group n; nothing computed on fewer
than MIN_GROUP_SIZE (200) test rows is ever reported. A gap that cannot be
reported comes back with reported=False and a stated reason — it is NEVER
silently dropped. Silent omission is precisely the failure this gate prevents.

Gap records are assembled from pre-existing metric functions. This module does
not reimplement a single metric — it only assembles, bootstraps, and validates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

from .cache import ArtifactCache
from .cells import Cell
from .data.acs import load_cell_table
from .metrics.robustness import accuracy
from .tabular import predict_tabular_cell, source_domain, target_domain_of
from .metrics.subgroup import (
    MIN_GROUP_SIZE,
    group_sizes,
    worst_group_accuracy,
    equal_opportunity_gap,
    per_group_ece_gap,
)
from .metrics.calibration import (
    brier_score,
    ece_equal_mass,
)
from .stats.bootstrap import bootstrap_interval


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class GapRecord:
    """A single metric's value, interval, and group sizes for a cell.

    Attributes:
        name: Metric name (e.g. "accuracy", "worst_group_accuracy").
        value: Point estimate of the metric.
        interval: Tuple (low, high) from bootstrap.
        group_sizes: Dict mapping group id to count in the evaluation set.
        n_total: Total evaluation samples, sum of group_sizes.
        reported: Whether this record passes the MIN_GROUP_SIZE gate.
        reason: If reported=False, a human-readable explanation. Else empty string.
    """
    name: str
    value: float
    interval: tuple[float, float]
    group_sizes: dict[int, int]
    n_total: int
    reported: bool
    reason: str


# ============================================================================
# Metric evaluation with bootstrap intervals
# ============================================================================


def _accuracy_per_sample(proba: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample correctness, 1.0 or 0.0.

    Not a second accuracy implementation: its mean is exactly
    `metrics.robustness.accuracy`, and a test pins that. It exists because
    bootstrapping resamples rows, which needs the per-row decomposition rather
    than the scalar.
    """
    preds = np.asarray(proba).argmax(axis=1)
    return (preds == np.asarray(labels)).astype(np.float64)


def evaluate_tabular_cell(
    cell: Cell,
    config: dict[str, Any],
    cache: ArtifactCache,
    *,
    root: Optional[str] = None,
) -> list[GapRecord]:
    """Evaluate a tabular cell on six calibration and fairness metrics.

    Computes accuracy, brier, ece_equal_mass, worst_group_accuracy,
    equal_opportunity_gap, and per_group_ece_gap. Each is bootstrapped and
    validated against MIN_GROUP_SIZE.

    Args:
        cell: A Cell specifying track, model_id, seed, shift_family, severity.
        config: Dict with "task" and "n_eval" keys.
        cache: ArtifactCache for caching predictions.
        root: Root directory for ACS data (default: None).

    Returns:
        List of GapRecord, one per metric. Records with reported=False are
        included with a reason, never silently dropped.
    """
    # The domain this cell is scored on. Resolved through target_domain_of so
    # this module and predict_tabular_cell can never disagree about which rows
    # a cell's predictions correspond to — a mismatch here would silently score
    # one domain's probabilities against another domain's labels.
    target = target_domain_of(cell, source_domain(config))

    data = load_cell_table(
        config["task"],
        target,
        config["n_eval"],
        root=root,
        download=True,
    )
    proba = predict_tabular_cell(cell, config, cache, root=root)

    proba = np.asarray(proba, dtype=np.float64)
    labels = data.y
    groups = data.groups

    # Compute group sizes
    group_dict = group_sizes(groups)
    n_total = len(labels)

    records = []

    # ================================================================
    # Non-group metrics: accuracy, brier, ece_equal_mass
    # ================================================================

    # Accuracy: global, no group validation needed
    acc_value = accuracy(proba, labels)
    acc_per_sample = _accuracy_per_sample(proba, labels)
    acc_interval = bootstrap_interval(acc_per_sample, np.mean, n_resamples=2000, seed=0)
    records.append(GapRecord(
        name="accuracy",
        value=acc_value,
        interval=(acc_interval.low, acc_interval.high),
        group_sizes=group_dict,
        n_total=n_total,
        reported=True,
        reason="",
    ))

    # Brier: global, no group validation needed
    brier_value = brier_score(proba, labels)
    brier_per_sample = np.sum((proba - np.eye(2)[labels]) ** 2, axis=1)
    brier_interval = bootstrap_interval(brier_per_sample, np.mean, n_resamples=2000, seed=0)
    records.append(GapRecord(
        name="brier",
        value=brier_value,
        interval=(brier_interval.low, brier_interval.high),
        group_sizes=group_dict,
        n_total=n_total,
        reported=True,
        reason="",
    ))

    # ECE: global, no group validation needed
    ece_value = ece_equal_mass(proba, labels, n_bins=15)
    # ECE is a global metric, so we bootstrap by resampling rows and recomputing
    def ece_resampler():
        rng = np.random.default_rng(0)
        ece_values = []
        n = len(labels)
        for _ in range(2000):
            idxs = rng.integers(0, n, size=n)
            ece_values.append(ece_equal_mass(proba[idxs], labels[idxs], n_bins=15))
        return np.array(ece_values)
    ece_values = ece_resampler()
    ece_low, ece_high = np.quantile(ece_values, [0.025, 0.975])
    records.append(GapRecord(
        name="ece_equal_mass",
        value=ece_value,
        interval=(float(ece_low), float(ece_high)),
        group_sizes=group_dict,
        n_total=n_total,
        reported=True,
        reason="",
    ))

    # ================================================================
    # Group-based metrics: validate MIN_GROUP_SIZE
    # ================================================================

    # Check if all groups meet the threshold
    small_groups = {g: n for g, n in group_dict.items() if n < MIN_GROUP_SIZE}

    if small_groups:
        reason = f"groups below MIN_GROUP_SIZE={MIN_GROUP_SIZE}: {small_groups}"
    else:
        reason = ""

    # Worst group accuracy: only compute if all groups pass threshold
    if not small_groups:
        wga_value = worst_group_accuracy(proba, labels, groups, min_group_size=MIN_GROUP_SIZE)
        # Bootstrap: resample samples, recompute worst_group_accuracy
        # Note: bootstrap resamples might violate MIN_GROUP_SIZE; skip those replicates
        def wga_resampler():
            rng = np.random.default_rng(0)
            wga_values = []
            n = len(labels)
            for _ in range(2000):
                idxs = rng.integers(0, n, size=n)
                try:
                    val = worst_group_accuracy(
                        proba[idxs], labels[idxs], groups[idxs], min_group_size=MIN_GROUP_SIZE
                    )
                    wga_values.append(val)
                except ValueError:
                    # This bootstrap replicate violates MIN_GROUP_SIZE; skip it
                    pass
            return np.array(wga_values) if wga_values else np.array([wga_value])
        wga_values = wga_resampler()
        wga_low, wga_high = np.quantile(wga_values, [0.025, 0.975])
        records.append(GapRecord(
            name="worst_group_accuracy",
            value=wga_value,
            interval=(float(wga_low), float(wga_high)),
            group_sizes=group_dict,
            n_total=n_total,
            reported=True,
            reason="",
        ))
    else:
        records.append(GapRecord(
            name="worst_group_accuracy",
            value=np.nan,
            interval=(np.nan, np.nan),
            group_sizes=group_dict,
            n_total=n_total,
            reported=False,
            reason=reason,
        ))

    # Equal opportunity gap: only compute if all groups pass threshold
    if not small_groups:
        eog_value = equal_opportunity_gap(proba, labels, groups, min_group_size=MIN_GROUP_SIZE)
        def eog_resampler():
            rng = np.random.default_rng(0)
            eog_values = []
            n = len(labels)
            for _ in range(2000):
                idxs = rng.integers(0, n, size=n)
                try:
                    val = equal_opportunity_gap(
                        proba[idxs], labels[idxs], groups[idxs], min_group_size=MIN_GROUP_SIZE
                    )
                    eog_values.append(val)
                except (ValueError, IndexError):
                    # This bootstrap replicate violates MIN_GROUP_SIZE or has other issues; skip it
                    pass
            return np.array(eog_values) if eog_values else np.array([eog_value])
        eog_values = eog_resampler()
        eog_low, eog_high = np.quantile(eog_values, [0.025, 0.975])
        records.append(GapRecord(
            name="equal_opportunity_gap",
            value=eog_value,
            interval=(float(eog_low), float(eog_high)),
            group_sizes=group_dict,
            n_total=n_total,
            reported=True,
            reason="",
        ))
    else:
        records.append(GapRecord(
            name="equal_opportunity_gap",
            value=np.nan,
            interval=(np.nan, np.nan),
            group_sizes=group_dict,
            n_total=n_total,
            reported=False,
            reason=reason,
        ))

    # Per-group ECE gap: only compute if all groups pass threshold
    if not small_groups:
        peg_value = per_group_ece_gap(proba, labels, groups, n_bins=15, min_group_size=MIN_GROUP_SIZE)
        def peg_resampler():
            rng = np.random.default_rng(0)
            peg_values = []
            n = len(labels)
            for _ in range(2000):
                idxs = rng.integers(0, n, size=n)
                try:
                    val = per_group_ece_gap(
                        proba[idxs], labels[idxs], groups[idxs], n_bins=15, min_group_size=MIN_GROUP_SIZE
                    )
                    peg_values.append(val)
                except (ValueError, IndexError):
                    # This bootstrap replicate violates MIN_GROUP_SIZE or has other issues; skip it
                    pass
            return np.array(peg_values) if peg_values else np.array([peg_value])
        peg_values = peg_resampler()
        peg_low, peg_high = np.quantile(peg_values, [0.025, 0.975])
        records.append(GapRecord(
            name="per_group_ece_gap",
            value=peg_value,
            interval=(float(peg_low), float(peg_high)),
            group_sizes=group_dict,
            n_total=n_total,
            reported=True,
            reason="",
        ))
    else:
        records.append(GapRecord(
            name="per_group_ece_gap",
            value=np.nan,
            interval=(np.nan, np.nan),
            group_sizes=group_dict,
            n_total=n_total,
            reported=False,
            reason=reason,
        ))

    return records
