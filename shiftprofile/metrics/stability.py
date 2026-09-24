"""Attribution stability between a clean input and its corrupted counterpart.

The third validity control is enforced here: stability is computed ONLY over
samples whose prediction did not change. Without that conditioning, a stability
drop is confounded with an accuracy drop — the attribution changed because the
model now predicts a different class, which says nothing about the explainer.

`n_conditioned` is returned alongside every value and must be reported beside
it, because under severe shift the conditioning set shrinks and the estimate
gets noisier for a reason that has nothing to do with the explainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class StabilityResult:
    spearman: float
    top_k_overlap: float
    n_conditioned: int
    n_total: int


def attribution_stability(
    clean_attributions: np.ndarray,
    shifted_attributions: np.ndarray,
    clean_predictions: np.ndarray,
    shifted_predictions: np.ndarray,
    top_k: int,
) -> StabilityResult:
    """Rank correlation and top-k overlap over unchanged-prediction samples.

    Attributions are (n_samples, n_features), already flattened to a common
    grid by the caller. Comparing maps across explainers or model families is
    only valid on that common grid.
    """
    clean_attributions = np.asarray(clean_attributions, dtype=np.float64)
    shifted_attributions = np.asarray(shifted_attributions, dtype=np.float64)
    if clean_attributions.shape != shifted_attributions.shape:
        raise ValueError(
            f"attribution shapes differ: {clean_attributions.shape} vs "
            f"{shifted_attributions.shape}"
        )
    n_total, n_features = clean_attributions.shape
    if not 0 < top_k <= n_features:
        raise ValueError(f"top_k must be in (0, {n_features}], got {top_k}")

    unchanged = np.asarray(clean_predictions) == np.asarray(shifted_predictions)
    n_conditioned = int(unchanged.sum())
    if n_conditioned == 0:
        return StabilityResult(np.nan, np.nan, 0, n_total)

    clean_sub = clean_attributions[unchanged]
    shifted_sub = shifted_attributions[unchanged]

    correlations = []
    overlaps = []
    for clean_row, shifted_row in zip(clean_sub, shifted_sub):
        rho = spearmanr(clean_row, shifted_row).statistic
        correlations.append(0.0 if np.isnan(rho) else rho)

        clean_top = set(np.argsort(-np.abs(clean_row), kind="stable")[:top_k])
        shifted_top = set(np.argsort(-np.abs(shifted_row), kind="stable")[:top_k])
        overlaps.append(len(clean_top & shifted_top) / top_k)

    return StabilityResult(
        spearman=float(np.mean(correlations)),
        top_k_overlap=float(np.mean(overlaps)),
        n_conditioned=n_conditioned,
        n_total=n_total,
    )
