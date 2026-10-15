"""Rank stability of model orderings.

A ranking that flips when the faithfulness metric, explainer or seed changes is
not a finding. Bootstrap rank intervals make that visible instead of hiding it
behind a point ordering — and if rankings turn out to depend entirely on
arbitrary choices, that instability becomes the paper's headline rather than
its embarrassment.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def kendall_tau(ranking_a, ranking_b) -> float:
    """Kendall's tau-b between two orderings."""
    a = np.asarray(ranking_a, dtype=np.float64)
    b = np.asarray(ranking_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"rankings must be the same length, got {a.shape} and {b.shape}"
        )
    return float(kendalltau(a, b).statistic)


def bootstrap_rank_intervals(
    scores: dict[str, np.ndarray],
    n_resamples: int = 2000,
    seed: int = 0,
    higher_is_better: bool = True,
) -> dict[str, tuple[int, int]]:
    """Min and max rank each model attains across bootstrap resamples.

    Scores are per-sample arrays of equal length, resampled with shared indices
    so models are always compared on the same samples.
    """
    names = sorted(scores)
    if len(names) < 2:
        raise ValueError(f"need at least two models to rank, got {names}")

    matrix = np.stack([np.asarray(scores[n], dtype=np.float64) for n in names])
    n_samples = matrix.shape[1]
    rng = np.random.default_rng(seed)

    ranks = np.empty((n_resamples, len(names)), dtype=np.int64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_samples, size=n_samples)
        means = matrix[:, idx].mean(axis=1)
        order = np.argsort(-means if higher_is_better else means, kind="stable")
        placement = np.empty(len(names), dtype=np.int64)
        placement[order] = np.arange(1, len(names) + 1)
        ranks[i] = placement

    return {
        name: (int(ranks[:, j].min()), int(ranks[:, j].max()))
        for j, name in enumerate(names)
    }
