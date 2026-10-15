"""Bootstrap intervals over evaluation samples.

Every reported number carries an interval. `half_width` exists because the
pilot gate is stated in those terms: the faithfulness interval half-width must
be under 10% of the clean-to-severe change, or the measurement cannot support
the claim and the protocol changes.

Comparisons are paired on the same evaluation samples — the fixed indices are
committed to the repository precisely so this pairing is possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for a statistic of per-sample values."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty array")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.array([statistic(values[row]) for row in idx])

    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return Interval(float(statistic(values)), float(low), float(high), n)


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Interval for mean(a) - mean(b), resampling both with the SAME indices.

    Breaking the pairing would inflate the interval by the between-sample
    variance that the paired design exists to remove.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be the same length, got {a.shape} and {b.shape}")
    return bootstrap_interval(
        a - b, np.mean, n_resamples=n_resamples, seed=seed, alpha=alpha
    )
