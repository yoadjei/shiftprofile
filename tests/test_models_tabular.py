"""Tests for tabular model definitions.

These tests verify that tabular models (logreg, xgboost, mlp) can be built,
fitted, and produce correct output shapes and probabilities.
All tests use synthetic data to ensure reproducibility and avoid network calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from shiftprofile.models.tabular import (
    TABULAR_MODELS,
    build_tabular_model,
    fit_tabular,
    predict_proba_tabular,
)


@pytest.fixture
def synthetic_binary_data():
    """Tiny synthetic binary classification dataset.

    Using a small dataset ensures tests complete in seconds and is sufficient
    to verify correctness. For testing purposes, we do not need large datasets.
    """
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_tabular_models_tuple_is_correct():
    """TABULAR_MODELS must be a tuple containing exactly three model names."""
    assert isinstance(TABULAR_MODELS, tuple)
    assert len(TABULAR_MODELS) == 3
    assert "logreg" in TABULAR_MODELS
    assert "xgboost" in TABULAR_MODELS
    assert "mlp" in TABULAR_MODELS


def test_build_tabular_model_logreg(synthetic_binary_data):
    """logreg builds and produces sklearn-compatible unfitted estimator."""
    model = build_tabular_model("logreg", seed=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_build_tabular_model_xgboost(synthetic_binary_data):
    """xgboost builds and produces sklearn-compatible unfitted estimator."""
    model = build_tabular_model("xgboost", seed=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_build_tabular_model_mlp(synthetic_binary_data):
    """mlp builds and produces sklearn-compatible unfitted estimator."""
    model = build_tabular_model("mlp", seed=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_build_tabular_model_unknown_raises_with_valid_names():
    """Unknown model_id raises ValueError with valid options listed."""
    with pytest.raises(ValueError, match="logreg|xgboost|mlp"):
        build_tabular_model("unknown_model", seed=42)


def test_fit_tabular_logreg(synthetic_binary_data):
    """fit_tabular builds and fits logreg successfully."""
    X, y = synthetic_binary_data
    model = fit_tabular("logreg", X, y, seed=42)
    assert hasattr(model, "predict_proba")
    # Verify it can make predictions (no error)
    out = model.predict_proba(X)
    assert out.shape[0] == X.shape[0]


def test_fit_tabular_xgboost(synthetic_binary_data):
    """fit_tabular builds and fits xgboost successfully."""
    X, y = synthetic_binary_data
    model = fit_tabular("xgboost", X, y, seed=42)
    assert hasattr(model, "predict_proba")
    # Verify it can make predictions (no error)
    out = model.predict_proba(X)
    assert out.shape[0] == X.shape[0]


def test_fit_tabular_mlp(synthetic_binary_data):
    """fit_tabular builds and fits mlp successfully."""
    X, y = synthetic_binary_data
    model = fit_tabular("mlp", X, y, seed=42)
    assert hasattr(model, "predict_proba")
    # Verify it can make predictions (no error)
    out = model.predict_proba(X)
    assert out.shape[0] == X.shape[0]


def test_predict_proba_tabular_shape_logreg(synthetic_binary_data):
    """predict_proba_tabular returns (n, 2) shaped array."""
    X, y = synthetic_binary_data
    model = fit_tabular("logreg", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    assert out.shape == (X.shape[0], 2)


def test_predict_proba_tabular_shape_xgboost(synthetic_binary_data):
    """predict_proba_tabular returns (n, 2) shaped array."""
    X, y = synthetic_binary_data
    model = fit_tabular("xgboost", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    assert out.shape == (X.shape[0], 2)


def test_predict_proba_tabular_shape_mlp(synthetic_binary_data):
    """predict_proba_tabular returns (n, 2) shaped array."""
    X, y = synthetic_binary_data
    model = fit_tabular("mlp", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    assert out.shape == (X.shape[0], 2)


def test_predict_proba_tabular_dtype_is_float64(synthetic_binary_data):
    """predict_proba_tabular returns float64 dtype."""
    X, y = synthetic_binary_data
    model = fit_tabular("logreg", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    assert out.dtype == np.float64


def test_predict_proba_tabular_rows_sum_to_one(synthetic_binary_data):
    """predict_proba_tabular rows sum to 1.0 within tolerance."""
    X, y = synthetic_binary_data
    model = fit_tabular("logreg", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    row_sums = out.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)


def test_predict_proba_tabular_rows_sum_to_one_xgboost(synthetic_binary_data):
    """predict_proba_tabular rows sum to 1.0 within tolerance for xgboost."""
    X, y = synthetic_binary_data
    model = fit_tabular("xgboost", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    row_sums = out.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)


def test_predict_proba_tabular_rows_sum_to_one_mlp(synthetic_binary_data):
    """predict_proba_tabular rows sum to 1.0 within tolerance for mlp."""
    X, y = synthetic_binary_data
    model = fit_tabular("mlp", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    row_sums = out.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)


def test_predict_proba_tabular_values_in_valid_range(synthetic_binary_data):
    """predict_proba_tabular values are in [0, 1]."""
    X, y = synthetic_binary_data
    model = fit_tabular("logreg", X, y, seed=42)
    out = predict_proba_tabular(model, X)
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_identical_seeds_logreg_produce_identical_predictions(synthetic_binary_data):
    """Two logreg models with identical seed produce identical predictions."""
    X, y = synthetic_binary_data
    model1 = fit_tabular("logreg", X, y, seed=42)
    model2 = fit_tabular("logreg", X, y, seed=42)
    out1 = predict_proba_tabular(model1, X)
    out2 = predict_proba_tabular(model2, X)
    np.testing.assert_array_equal(out1, out2)


def test_identical_seeds_xgboost_produce_identical_predictions(synthetic_binary_data):
    """Two xgboost models with identical seed produce identical predictions."""
    X, y = synthetic_binary_data
    model1 = fit_tabular("xgboost", X, y, seed=42)
    model2 = fit_tabular("xgboost", X, y, seed=42)
    out1 = predict_proba_tabular(model1, X)
    out2 = predict_proba_tabular(model2, X)
    np.testing.assert_array_equal(out1, out2)


def test_identical_seeds_mlp_produce_identical_predictions(synthetic_binary_data):
    """Two mlp models with identical seed produce identical predictions.

    MLPClassifier has multiple sources of randomness: weight initialization,
    SGD shuffle, and dropout. Setting seed controls all of them.
    """
    X, y = synthetic_binary_data
    model1 = fit_tabular("mlp", X, y, seed=42)
    model2 = fit_tabular("mlp", X, y, seed=42)
    out1 = predict_proba_tabular(model1, X)
    out2 = predict_proba_tabular(model2, X)
    np.testing.assert_array_equal(out1, out2)


def test_different_seeds_xgboost_produce_different_predictions(synthetic_binary_data):
    """Different seeds in xgboost produce different predictions (stochastic)."""
    X, y = synthetic_binary_data
    model_seed42 = fit_tabular("xgboost", X, y, seed=42)
    model_seed99 = fit_tabular("xgboost", X, y, seed=99)
    out_42 = predict_proba_tabular(model_seed42, X)
    out_99 = predict_proba_tabular(model_seed99, X)
    # At least some predictions should differ
    assert not np.allclose(out_42, out_99)


def test_different_seeds_mlp_produce_different_predictions(synthetic_binary_data):
    """Different seeds in mlp produce different predictions (stochastic)."""
    X, y = synthetic_binary_data
    model_seed42 = fit_tabular("mlp", X, y, seed=42)
    model_seed99 = fit_tabular("mlp", X, y, seed=99)
    out_42 = predict_proba_tabular(model_seed42, X)
    out_99 = predict_proba_tabular(model_seed99, X)
    # At least some predictions should differ
    assert not np.allclose(out_42, out_99)


def test_fit_tabular_unknown_raises_with_valid_names(synthetic_binary_data):
    """fit_tabular with unknown model_id raises ValueError with valid options."""
    X, y = synthetic_binary_data
    with pytest.raises(ValueError, match="logreg|xgboost|mlp"):
        fit_tabular("unknown_model", X, y, seed=42)
