"""Robustness under natural covariate shift and common corruption.

Robustness means exactly one thing in this study: performance at graded
severity with labels unchanged. Adversarial perturbation is out of scope —
different threat model, and already covered by RobustBench.

Raw OOD accuracy is largely predicted by clean accuracy ("accuracy on the
line"), so effective robustness — the residual after that prediction — is what
gets reported.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

CLIP = 1e-6


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    return float((probs.argmax(axis=1) == labels).mean())


def accuracy_drop(clean_accuracy: float, shifted_accuracy: float) -> float:
    return float(clean_accuracy - shifted_accuracy)


def effective_robustness(
    id_accuracy: float,
    ood_accuracy: float,
    reference_id: np.ndarray,
    reference_ood: np.ndarray,
) -> float:
    """OOD accuracy beyond what ID accuracy predicts, in probit space.

    The linear fit is taken over a reference set of models. Returns the residual:
    zero means the model sits exactly on the accuracy-on-the-line trend.
    """
    reference_id = np.asarray(reference_id, dtype=np.float64)
    reference_ood = np.asarray(reference_ood, dtype=np.float64)
    if reference_id.size < 2:
        raise ValueError(
            f"need at least two reference points to fit a line, got {reference_id.size}"
        )

    z_ref_id = norm.ppf(np.clip(reference_id, CLIP, 1 - CLIP))
    z_ref_ood = norm.ppf(np.clip(reference_ood, CLIP, 1 - CLIP))
    slope, intercept = np.polyfit(z_ref_id, z_ref_ood, deg=1)

    z_id = norm.ppf(np.clip(id_accuracy, CLIP, 1 - CLIP))
    z_ood = norm.ppf(np.clip(ood_accuracy, CLIP, 1 - CLIP))
    return float(z_ood - (slope * z_id + intercept))
