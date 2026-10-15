"""Calibration metrics.

Calibration is the study's central reliability axis: the cheapest measurement
and the candidate predictor of the expensive ones. ECE alone is insufficient —
it is bin-dependent, biased upward at small n, and blind to ranking — so it is
always reported beside Brier (a proper scoring rule) and AURC (ranking).

All functions take `probs` of shape (n, k) and integer `labels` of shape (n,).
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

EPS = 1e-12


def _check(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (n, k), got shape {probs.shape}")
    if labels.shape != (probs.shape[0],):
        raise ValueError(
            f"labels must be ({probs.shape[0]},), got shape {labels.shape}"
        )
    return probs, labels


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target.

    A strictly proper scoring rule, so unlike ECE it cannot be gamed by a model
    that is well-binned but uninformative.
    """
    probs, labels = _check(probs, labels)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative log likelihood of the true class, clipped to stay finite."""
    probs, labels = _check(probs, labels)
    true_p = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_p, EPS, 1.0))))


def _confidence_and_correctness(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probs, labels = _check(probs, labels)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    return confidence, correct


def _equal_mass_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split into bins holding (near) equal counts, not equal confidence width.

    Equal-width binning leaves most bins empty when confidences cluster near 1,
    which is the normal case for a trained classifier.
    """
    order = np.argsort(confidence, kind="stable")
    conf_sorted = confidence[order]
    corr_sorted = correct[order]
    n_bins = min(n_bins, len(conf_sorted))
    return list(
        zip(
            np.array_split(conf_sorted, n_bins),
            np.array_split(corr_sorted, n_bins),
        )
    )


def ece_equal_mass(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label expected calibration error with equal-mass bins (plugin)."""
    confidence, correct = _confidence_and_correctness(probs, labels)
    n = len(confidence)
    total = 0.0
    for conf_bin, corr_bin in _equal_mass_bins(confidence, correct, n_bins):
        if len(conf_bin) == 0:
            continue
        total += (len(conf_bin) / n) * abs(corr_bin.mean() - conf_bin.mean())
    return float(total)


def ece_debiased(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Bias-corrected squared-ECE estimator, reported on the ECE scale.

    The plugin estimator is biased upward because each bin's accuracy is a noisy
    binomial mean. Subtracting that within-bin variance removes the leading bias
    term. Reported as sqrt of the corrected squared ECE, floored at zero.
    """
    confidence, correct = _confidence_and_correctness(probs, labels)
    n = len(confidence)
    total = 0.0
    for conf_bin, corr_bin in _equal_mass_bins(confidence, correct, n_bins):
        n_b = len(conf_bin)
        if n_b < 2:
            continue
        acc_b = corr_bin.mean()
        gap_sq = (acc_b - conf_bin.mean()) ** 2
        bias = acc_b * (1.0 - acc_b) / (n_b - 1)
        total += (n_b / n) * (gap_sq - bias)
    return float(np.sqrt(max(total, 0.0)))


def aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area under the risk-coverage curve, using max-probability as the score.

    Measures whether the model's confidence *ranks* its errors correctly, which
    ECE is blind to: a model can be perfectly binned and still rank its errors
    at random.
    """
    confidence, correct = _confidence_and_correctness(probs, labels)
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(np.mean(cumulative_risk))


def accuracy_at_coverage(
    probs: np.ndarray, labels: np.ndarray, coverage: float = 0.8
) -> float:
    """Accuracy on the most-confident `coverage` fraction of samples."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    confidence, correct = _confidence_and_correctness(probs, labels)
    order = np.argsort(-confidence, kind="stable")
    k = max(1, int(round(coverage * len(confidence))))
    return float(correct[order][:k].mean())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax of logits divided by a scalar temperature."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (0.05, 100.0)
) -> float:
    """Fit a single scalar temperature by minimising NLL.

    MUST be fitted on a held-out in-distribution calibration split. Fitting on
    shifted data would leak the thing the study is trying to measure.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(labels))

    def objective(log_t: float) -> float:
        t = np.exp(log_t)
        return float(-np.mean(log_softmax(logits / t, axis=1)[rows, labels]))

    result = minimize_scalar(
        objective,
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": 1e-10},
    )
    temperature = float(np.exp(result.x))
    lo, hi = bounds
    if temperature <= lo * (1 + 1e-6) or temperature >= hi * (1 - 1e-6):
        warnings.warn(
            f"fitted temperature {temperature:.4g} is pinned at a bound {bounds}. "
            "The NLL optimum lies outside the search range, so this is a failed "
            "fit, not a calibrated one. Inspect the calibration split before "
            "using these probabilities.",
            UserWarning,
            stacklevel=2,
        )
    return temperature
