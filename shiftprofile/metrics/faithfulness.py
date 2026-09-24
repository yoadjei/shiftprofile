"""Removal-based attribution faithfulness.

Two of the study's three validity controls are enforced here by signature:

1. `relative_faithfulness` refuses to return a value without a paired
   random-attribution control measured in the same cell. Raw removal AUC is
   uninterpretable under shift, because masking an already-corrupted input
   pushes it doubly off-distribution — the drop would be reported as lost
   faithfulness when it is a masking artefact.

2. `imputation` is keyword-only with no default, so the scheme is always an
   explicit, recorded choice. It is the axis the E6 ablation varies.

This module computes metrics *from* removal curves. Producing the curves needs
forward passes and belongs to the GPU stage.
"""

from __future__ import annotations

import numpy as np

VALID_IMPUTATIONS = ("mean", "blur", "uniform_noise", "zero")


def rank_features(attribution: np.ndarray) -> np.ndarray:
    """Indices ordered by descending |attribution|, ties broken by position."""
    flat = np.abs(np.asarray(attribution, dtype=np.float64).ravel())
    return np.argsort(-flat, kind="stable")


def removal_auc(curve: np.ndarray, fractions: np.ndarray) -> float:
    """Area under the removal curve, normalised to the fraction range.

    `curve[i]` is the predicted-class probability after removing the top
    `fractions[i]` of features. Lower AUC means the attribution identified the
    features the model actually relied on.
    """
    curve = np.asarray(curve, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    if curve.shape != fractions.shape:
        raise ValueError(
            f"curve and fractions must be the same length, "
            f"got {curve.shape} and {fractions.shape}"
        )
    if curve.size < 2:
        raise ValueError("need at least two points to integrate a curve")
    if not np.all(np.diff(fractions) > 0):
        raise ValueError("fractions must be strictly ascending")
    span = fractions[-1] - fractions[0]
    return float(np.trapezoid(curve, fractions) / span)


def relative_faithfulness(
    model_curve: np.ndarray,
    random_curve: np.ndarray | None,
    fractions: np.ndarray,
    *,
    imputation: str,
) -> float:
    """Faithfulness of an attribution relative to a random-attribution control.

    Returns `random_auc - model_auc`: positive means the attribution beat random
    in this cell. Zero means the explainer carries no information the random
    baseline does not — the quantity whose severity onset the study tries to
    predict from in-distribution calibration.
    """
    if random_curve is None:
        raise ValueError(
            "a random-attribution control is required: raw removal AUC is not "
            "comparable across severities, because masking an already-shifted "
            "input is itself off-distribution. Measure the random control in "
            "the same cell and pass it here."
        )
    if imputation not in VALID_IMPUTATIONS:
        raise ValueError(
            f"unknown imputation {imputation!r}; expected one of {VALID_IMPUTATIONS}"
        )
    return removal_auc(random_curve, fractions) - removal_auc(model_curve, fractions)
