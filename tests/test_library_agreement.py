"""Pin our metric implementations against an independent library.

Characterisation tests, not correctness tests. If a dependency upgrade changes
a value, this fails and the discrepancy gets documented in
docs/metric_provenance.md rather than silently changing a published number.
"""

import numpy as np
import pytest
from shiftprofile.metrics.calibration import brier_score, ece_equal_mass

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def test_brier_agrees_with_sklearn_on_binary():
    rng = np.random.default_rng(0)
    p_positive = rng.uniform(size=500)
    probs = np.stack([1 - p_positive, p_positive], axis=1)
    labels = (rng.uniform(size=500) < p_positive).astype(int)

    ours = brier_score(probs, labels)
    # sklearn's binary Brier is the single-class MSE; the multiclass form
    # sums over both classes and is therefore exactly twice it.
    theirs = sklearn_metrics.brier_score_loss(labels, p_positive)

    assert ours == pytest.approx(2 * theirs, rel=1e-9)


def test_ece_is_stable_against_bin_count():
    """ECE is bin-dependent by construction. This records how much, so the
    sensitivity is a documented number rather than a surprise in review."""
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1.0, size=5000)
    probs = np.stack([1 - conf, conf], axis=1)
    labels = (rng.uniform(size=5000) < conf).astype(int)

    values = [ece_equal_mass(probs, labels, n_bins=b) for b in (5, 10, 15, 30)]

    assert max(values) < 0.05
    assert max(values) - min(values) < 0.03
