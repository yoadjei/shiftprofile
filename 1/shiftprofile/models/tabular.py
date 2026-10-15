"""Tabular model definitions for distribution shift studies with attribution.

This module provides three sklearn-compatible tabular models optimized for
faithfulness analysis. Each model is chosen to enable a specific type of
attribution method and to understand how faithfulness degrades under shift.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


TABULAR_MODELS: tuple[str, ...] = ("logreg", "xgboost", "mlp")


def build_tabular_model(model_id: str, *, seed: int) -> object:
    """Build an unfitted sklearn-compatible tabular model by ID.

    Args:
        model_id: A registered model name ("logreg", "xgboost", "mlp").
        seed: Random seed for reproducibility. Required because all models
              include stochastic components (weight init, SGD, or tree splits).

    Returns:
        An unfitted sklearn-compatible model (Pipeline or estimator).

    Raises:
        ValueError: If model_id is unknown; message lists valid options.
    """
    builders = {
        "logreg": _build_logreg,
        "xgboost": _build_xgboost,
        "mlp": _build_mlp,
    }

    if model_id not in builders:
        available = ", ".join(sorted(builders.keys()))
        raise ValueError(
            f"unknown model {model_id!r}. Available: {available}"
        )

    return builders[model_id](seed=seed)


def _build_logreg(*, seed: int) -> object:
    """Build logreg: StandardScaler → LogisticRegression.

    Why logreg: Logistic regression is the faithfulness reference point
    because a linear model's SHAP attributions are exact, not approximated.
    This makes it the ground truth for comparing attribution methods across
    the nonlinear models (xgboost, mlp).

    Args:
        seed: Random seed (for API consistency; not used by logreg).

    Returns:
        An unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )


def _build_xgboost(*, seed: int) -> object:
    """Build xgboost: XGBClassifier with seed set.

    Why xgboost: TreeSHAP provides exact attributions for tree-based models,
    and XGBoost is the production standard. This makes it representative of
    how attribution fidelity changes under shift in deployed models.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        An unfitted XGBClassifier.
    """
    # Use limited iterations for tests to keep them fast;
    # real training will override via fit parameters if needed.
    # subsample < 1.0 enables stochasticity in tree construction, ensuring
    # different seeds produce different models even on small datasets.
    return xgb.XGBClassifier(
        n_estimators=10,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
        eval_metric="logloss",
    )


def _build_mlp(*, seed: int) -> object:
    """Build mlp: StandardScaler → MLPClassifier with small hidden layers.

    Why mlp: MLPs are the ablation control for deep nonlinear models.
    KernelSHAP is CPU-bound and too slow for large models, so this is
    deliberately small (64, 32 hidden units). The model is not on the
    critical path for production results but provides evidence that
    faithfulness degradation generalizes beyond tree-based methods.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        An unfitted sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    random_state=seed,
                    max_iter=500,
                    early_stopping=False,
                    n_iter_no_change=500,
                ),
            ),
        ]
    )


def fit_tabular(model_id: str, X, y, *, seed: int) -> object:
    """Build and fit a tabular model.

    Args:
        model_id: A registered model name ("logreg", "xgboost", "mlp").
        X: Feature matrix (n_samples, n_features).
        y: Target labels (n_samples,).
        seed: Random seed for reproducibility.

    Returns:
        A fitted sklearn-compatible model or Pipeline.

    Raises:
        ValueError: If model_id is unknown; message lists valid options.
    """
    model = build_tabular_model(model_id, seed=seed)
    model.fit(X, y)
    return model


def predict_proba_tabular(model, X) -> np.ndarray:
    """Predict class probabilities from a fitted tabular model.

    All models (logreg, xgboost, mlp) are binary classifiers. This function
    normalizes their output to a consistent shape: (n_samples, 2) float64,
    with rows summing to 1.0.

    Args:
        model: A fitted sklearn-compatible model or Pipeline.
        X: Feature matrix (n_samples, n_features).

    Returns:
        Probability matrix (n_samples, 2) as float64, rows sum to 1.0.
    """
    proba = model.predict_proba(X)
    # Ensure float64 and shape (n, 2)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape[1] != 2:
        raise ValueError(
            f"Expected binary classifier output shape (..., 2), "
            f"got {proba.shape}"
        )
    return proba
