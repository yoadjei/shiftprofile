"""Domain classifier AUC — tabular track only.

Measures how distinguishable two domains are by training a logistic regression
classifier to predict domain membership from features alone. The AUC must be
out-of-fold (computed during cross-validation), not training accuracy.

Why this design:
- AUC of 0.5 means indistinguishable; 1.0 means trivially separable.
- Subsampling to min(len(source), len(target), max_n) prevents class imbalance
  from inflating AUC. Without it, a size-100 vs size-10000 comparison would show
  high AUC even if distributions were identical, because the classifier learns
  to predict the larger class.
- Features are standardized to ISO(0, 1) before training, so the metric depends
  only on distribution differences, not feature scale.
- The interval is required, not optional. It is computed as the quantile range
  across all folds, giving a 95% confidence band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ShiftDistance:
    """Out-of-fold AUC of domain classifier and 95% confidence interval."""

    auc: float  # 0.5 = indistinguishable, 1.0 = trivially separable
    interval: tuple[float, float]  # 95% confidence band across folds
    n_source: int  # Rows used (after subsampling)
    n_target: int  # Rows used (after subsampling)
    n_splits: int  # Number of folds


def domain_classifier_auc(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
    max_n: int = 20000,
) -> ShiftDistance:
    """Measure domain distinguishability via cross-validated logistic regression.

    Args:
        X_source: Source domain features, shape (n, d)
        X_target: Target domain features, shape (m, d)
        seed: Random seed for fold splitting and model
        n_splits: Number of CV folds (default 5)
        max_n: Maximum rows per domain after subsampling (default 20000)

    Returns:
        ShiftDistance with out-of-fold AUC, interval, and subsample counts

    Raises:
        ValueError: If either domain has fewer than n_splits rows
    """
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)

    if len(X_source) < n_splits or len(X_target) < n_splits:
        raise ValueError(
            f"Both domains have fewer than n_splits={n_splits} rows. "
            f"Got n_source={len(X_source)}, n_target={len(X_target)}"
        )

    # Subsample to balance class sizes and cap total compute
    subsample_size = min(len(X_source), len(X_target), max_n)

    rng = np.random.RandomState(seed)
    idx_source = rng.choice(len(X_source), subsample_size, replace=False)
    idx_target = rng.choice(len(X_target), subsample_size, replace=False)

    X_source_sub = X_source[idx_source]
    X_target_sub = X_target[idx_target]

    # Stack and create domain labels (source=0, target=1)
    X = np.vstack([X_source_sub, X_target_sub])
    y = np.hstack([np.zeros(subsample_size), np.ones(subsample_size)])

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Cross-validated logistic regression with out-of-fold AUC
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(X_train, y_train)

        # Out-of-fold prediction on test fold
        y_score = clf.predict_proba(X_test)[:, 1]
        fold_auc = roc_auc_score(y_test, y_score)
        fold_aucs.append(fold_auc)

    fold_aucs = np.array(fold_aucs)
    mean_auc = float(fold_aucs.mean())
    lower, upper = _confidence_interval_95(fold_aucs)

    return ShiftDistance(
        auc=mean_auc,
        interval=(float(lower), float(upper)),
        n_source=int(subsample_size),
        n_target=int(subsample_size),
        n_splits=int(n_splits),
    )


def _confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    """Compute 95% confidence interval from an array of fold values.

    Uses the quantile method (2.5th and 97.5th percentiles).
    """
    values = np.asarray(values)
    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))
    return lower, upper
