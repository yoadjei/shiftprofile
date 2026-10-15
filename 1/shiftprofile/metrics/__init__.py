"""Metric registration.

Importing this package registers every probability-based metric under a stable
name, so configs and result records refer to metrics as strings.

Faithfulness, stability and subgroup metrics are deliberately NOT registered
here: they take different arguments (attribution maps, paired predictions,
group labels) and are called explicitly by the stages that have those inputs.
Forcing them behind a uniform (probs, labels) signature would hide exactly the
arguments the validity controls depend on — the random-attribution control, the
imputation scheme, and the unchanged-prediction conditioning.
"""

from ..registry import METRICS
from .calibration import (
    accuracy_at_coverage,
    aurc,
    brier_score,
    ece_debiased,
    ece_equal_mass,
    nll,
)
from .robustness import accuracy

_PROBABILITY_METRICS = {
    "accuracy": accuracy,
    "accuracy_at_coverage": accuracy_at_coverage,
    "aurc": aurc,
    "brier": brier_score,
    "ece": ece_equal_mass,
    "ece_debiased": ece_debiased,
    "nll": nll,
}

for _name, _fn in _PROBABILITY_METRICS.items():
    if _name not in METRICS:
        METRICS.register(_name)(_fn)

__all__ = [
    "accuracy",
    "accuracy_at_coverage",
    "aurc",
    "brier_score",
    "ece_debiased",
    "ece_equal_mass",
    "nll",
]
