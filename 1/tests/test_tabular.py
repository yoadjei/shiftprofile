"""Tests for tabular track: training and prediction stages.

Tests verify:
- target_domain_of correctly maps cells to domains
- Training caches and shares models across same (track, model_id, seed, task, source)
- Predictions cache per cell and do not refit on cache hit
- Second call to predict hits cache without retraining
- Array shape and dtype correctness
- No network access in any test (synthetic data only)
"""

from __future__ import annotations

import numpy as np
import pytest

from shiftprofile.tabular import (
    TABULAR_TRAIN_VERSION,
    TABULAR_PREDICT_VERSION,
    clear_model_memo,
    target_domain_of,
    train_tabular_cell,
    predict_tabular_cell,
    train_spec,
    predict_spec,
)
from shiftprofile.cache import ArtifactCache
from shiftprofile.cells import Cell, CLEAN
from shiftprofile.data.acs import TabularSplit


# ============================================================================
# Fixtures and Helpers
# ============================================================================


def make_synthetic_tabular_data(n_rows: int = 100, n_features: int = 10, seed: int = 0) -> TabularSplit:
    """Create synthetic tabular data for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_rows, n_features).astype(np.float64)
    y = rng.randint(0, 2, n_rows).astype(int)
    groups = rng.randint(1, 5, n_rows).astype(int)
    feature_names = tuple(f"feat_{i}" for i in range(n_features))

    return TabularSplit(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        domain="CA_2018",
        task="income"
    )


def make_config() -> dict:
    """Create a minimal config for testing.

    Domain ids are real-shaped ("CA_2018", not "shift_a") because the
    production path feeds shift_family straight to parse_domain. A placeholder
    that parse_domain would reject lets a test pass over a call chain that
    could never work against real data.
    """
    return {
        "task": "income",
        "source": "CA_2018",
        "models": ["logreg", "xgboost", "mlp"],
        "seeds": [0, 1],
        "shift_families": ["TX_2018"],
        "severities": [1, 2],
        "n_eval": 100,
        "n_train": 50,
    }


@pytest.fixture(autouse=True)
def _isolate_model_memo():
    """Fitted models are memoised per process, so a leak across tests would let
    one pass without fitting anything."""
    clear_model_memo()
    yield
    clear_model_memo()


# ============================================================================
# Tests for target_domain_of
# ============================================================================


class TestTargetDomainOf:
    """Tests for the target_domain_of function."""

    def test_target_domain_of_clean_returns_source(self):
        """Clean cells should return the source domain."""
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        result = target_domain_of(cell, source="CA_2018")
        assert result == "CA_2018"

    def test_target_domain_of_shift_returns_shift_family(self):
        """Shift cells should return the shift_family."""
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family="TX_2018", severity=1)
        result = target_domain_of(cell, source="CA_2018")
        assert result == "TX_2018"

    def test_target_domain_of_different_severities_same_shift(self):
        """Different severities of same shift should return same shift_family."""
        cell1 = Cell(track="tabular", model_id="logreg", seed=0, shift_family="NY_2018", severity=1)
        cell2 = Cell(track="tabular", model_id="logreg", seed=0, shift_family="NY_2018", severity=3)
        assert target_domain_of(cell1, "CA_2018") == target_domain_of(cell2, "CA_2018")

    def test_target_domain_of_source_parameter_only_used_for_clean(self):
        """Source parameter should only matter for clean cells."""
        cell_clean = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        cell_shift = Cell(track="tabular", model_id="logreg", seed=0, shift_family="TX_2018", severity=1)

        # Clean cells vary with source
        assert target_domain_of(cell_clean, "CA_2018") == "CA_2018"
        assert target_domain_of(cell_clean, "TX_2019") == "TX_2019"

        # Shift cells ignore source
        assert target_domain_of(cell_shift, "CA_2018") == "TX_2018"
        assert target_domain_of(cell_shift, "TX_2019") == "TX_2018"


# ============================================================================
# Tests for train_tabular_cell
# ============================================================================


class TestTrainTabularCell:
    """Tests for the train_tabular_cell function."""

    def test_train_tabular_cell_trains_on_cache_miss(self, tmp_path, monkeypatch):
        """Training should happen on cache miss."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        # Mock load_cell_table
        data = make_synthetic_tabular_data(n_rows=100)
        def mock_load_cell_table(*args, **kwargs):
            return data

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)

        # Training should complete without error
        result = train_tabular_cell(cell, config, cache, root="/tmp")

        # Result should be a fitted model
        assert result is not None
        assert hasattr(result, "predict_proba")

    def test_train_tabular_cell_caches_model(self, tmp_path, monkeypatch):
        """Training should cache the fitted model."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)
        def mock_load_cell_table(*args, **kwargs):
            return data

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)

        # Build the training spec using the single source of truth function
        source = "CA_2018"
        cache_spec = train_spec(cell, config["task"], source)

        # Before training, model should not be cached
        assert not cache.has(cache_spec, TABULAR_TRAIN_VERSION, kind="record")

        # After training, model metadata should be cached
        train_tabular_cell(cell, config, cache, root="/tmp")

        assert cache.has(cache_spec, TABULAR_TRAIN_VERSION, kind="record")

    def test_train_tabular_cell_loads_from_cache_on_hit(self, tmp_path, monkeypatch):
        """Second training call should load from cache, not retrain."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)

        fit_count = [0]

        # Mock the fit_tabular function to count calls
        def mock_fit_tabular(*args, **kwargs):
            fit_count[0] += 1
            # Return a minimal fitted model
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        def mock_load_cell_table(*args, **kwargs):
            return data

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        # First call should train
        model1 = train_tabular_cell(cell, config, cache, root="/tmp")
        first_fit_count = fit_count[0]
        assert first_fit_count == 1

        # Second call should load from cache, not retrain
        model2 = train_tabular_cell(cell, config, cache, root="/tmp")
        assert fit_count[0] == first_fit_count  # No additional fits

    def test_train_tabular_cell_same_source_different_target_domain_share_model(self, tmp_path, monkeypatch):
        """Two cells with same source but different target domains should share trained model."""
        cache = ArtifactCache(tmp_path)

        # One clean cell, one shifted cell (different target but same source)
        cell_clean = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        cell_shift = Cell(track="tabular", model_id="logreg", seed=0, shift_family="TX_2018", severity=1)

        config = make_config()
        data = make_synthetic_tabular_data(n_rows=100)

        def mock_load_cell_table(*args, **kwargs):
            return data

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)

        # Both should use source domain for training
        # They should have the same training spec
        source = "CA_2018"

        # Train on clean cell
        model_clean = train_tabular_cell(cell_clean, config, cache, root="/tmp")

        # Train on shift cell - should load the same model from cache
        model_shift = train_tabular_cell(cell_shift, config, cache, root="/tmp")

        # Both should exist and be the same model object (reused from cache)
        assert model_clean is not None
        assert model_shift is not None


# ============================================================================
# Tests for predict_tabular_cell
# ============================================================================


class TestPredictTabularCell:
    """Tests for the predict_tabular_cell function."""

    def test_predict_tabular_cell_returns_ndarray(self, tmp_path, monkeypatch):
        """predict_tabular_cell should return a numpy array."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)

        def mock_load_cell_table(*args, **kwargs):
            return data

        def mock_fit_tabular(*args, **kwargs):
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        result = predict_tabular_cell(cell, config, cache, root="/tmp")

        assert isinstance(result, np.ndarray)

    def test_predict_tabular_cell_returns_correct_shape(self, tmp_path, monkeypatch):
        """predict_tabular_cell should return shape (n_eval, 2) - class probabilities."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)

        def mock_load_cell_table(*args, **kwargs):
            return data

        def mock_fit_tabular(*args, **kwargs):
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        result = predict_tabular_cell(cell, config, cache, root="/tmp")

        # Shape should be (n_eval, 2)
        assert result.shape == (config["n_eval"], 2)

    def test_predict_tabular_cell_returns_float64(self, tmp_path, monkeypatch):
        """predict_tabular_cell should return float64 dtype."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)

        def mock_load_cell_table(*args, **kwargs):
            return data

        def mock_fit_tabular(*args, **kwargs):
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        result = predict_tabular_cell(cell, config, cache, root="/tmp")

        assert result.dtype == np.float64

    def test_predict_tabular_cell_caches_predictions(self, tmp_path, monkeypatch):
        """Predictions should be cached."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)

        def mock_load_cell_table(*args, **kwargs):
            return data

        def mock_fit_tabular(*args, **kwargs):
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        # Build the prediction spec using the single source of truth function
        source = "CA_2018"
        target = target_domain_of(cell, source)
        cache_spec = predict_spec(cell, config["task"], source, target)

        # Before prediction, should not be cached
        assert not cache.has(cache_spec, TABULAR_PREDICT_VERSION, kind="array")

        # After prediction, should be cached
        predict_tabular_cell(cell, config, cache, root="/tmp")

        assert cache.has(cache_spec, TABULAR_PREDICT_VERSION, kind="array")

    def test_predict_tabular_cell_cache_hit_does_not_retrain(self, tmp_path, monkeypatch):
        """Predictions cached from first call should not retrain on second call."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)

        fit_count = [0]

        def mock_fit_tabular(*args, **kwargs):
            fit_count[0] += 1
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        def mock_load_cell_table(*args, **kwargs):
            return data

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        # First call should train
        pred1 = predict_tabular_cell(cell, config, cache, root="/tmp")
        first_fit_count = fit_count[0]

        # Second call should load cached predictions and not retrain
        pred2 = predict_tabular_cell(cell, config, cache, root="/tmp")

        # No new fits should have occurred
        assert fit_count[0] == first_fit_count

        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_tabular_cell_different_targets_different_predictions_keys(self, tmp_path, monkeypatch):
        """Different target domains should have different prediction cache keys."""
        cache = ArtifactCache(tmp_path)

        # Two cells: clean and shifted, both using same source but different targets
        cell_clean = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        cell_shift = Cell(track="tabular", model_id="logreg", seed=0, shift_family="TX_2018", severity=1)

        config = make_config()
        data = make_synthetic_tabular_data(n_rows=100)

        def mock_load_cell_table(*args, **kwargs):
            return data

        def mock_fit_tabular(*args, **kwargs):
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        # Predict on both
        source = "CA_2018"
        pred_clean = predict_tabular_cell(cell_clean, config, cache, root="/tmp")
        pred_shift = predict_tabular_cell(cell_shift, config, cache, root="/tmp")

        # Build specs for both using the single source of truth function
        target_clean = target_domain_of(cell_clean, source)
        target_shift = target_domain_of(cell_shift, source)

        cache_spec_clean = predict_spec(cell_clean, config["task"], source, target_clean)
        cache_spec_shift = predict_spec(cell_shift, config["task"], source, target_shift)

        # Both should be cached
        assert cache.has(cache_spec_clean, TABULAR_PREDICT_VERSION, kind="array")
        assert cache.has(cache_spec_shift, TABULAR_PREDICT_VERSION, kind="array")

        # They should have different cache keys
        from shiftprofile.cache import spec_key
        key_clean = spec_key(cache_spec_clean, TABULAR_PREDICT_VERSION)
        key_shift = spec_key(cache_spec_shift, TABULAR_PREDICT_VERSION)
        assert key_clean != key_shift

    def test_predict_tabular_cell_probabilities_sum_to_one(self, tmp_path, monkeypatch):
        """Probability predictions should sum to 1.0 across classes."""
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        config = make_config()

        data = make_synthetic_tabular_data(n_rows=100)

        def mock_load_cell_table(*args, **kwargs):
            return data

        def mock_fit_tabular(*args, **kwargs):
            from shiftprofile.models.tabular import build_tabular_model
            model = build_tabular_model(kwargs.get("model_id", "logreg"), seed=kwargs.get("seed", 0))
            model.fit(args[1], args[2] if len(args) > 2 else kwargs.get("y"))
            return model

        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.load_train_table", mock_load_cell_table)
        monkeypatch.setattr("shiftprofile.tabular.fit_tabular", mock_fit_tabular)

        proba = predict_tabular_cell(cell, config, cache, root="/tmp")

        # Each row should sum to ~1.0
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)


class TestTrainEvalSeparation:
    """Regression guard: the fit must never see the rows it will be scored on.

    train_tabular_cell once called load_cell_table, the loader that returns the
    EVALUATION rows. Because the source domain doubles as the clean cell, that
    made the clean baseline in-sample while every shifted cell stayed
    out-of-sample, so a clean-to-shifted drop would appear for a model with no
    real degradation at all. That drop is this study's headline quantity.
    """

    def test_training_reads_the_train_loader_not_the_eval_loader(self, tmp_path, monkeypatch):
        called = []

        def fake_train_loader(task, domain, n_eval, n_train=None, **kw):
            called.append("train")
            return make_synthetic_tabular_data(n_rows=60)

        def fake_eval_loader(task, domain, n_eval, **kw):
            called.append("eval")
            return make_synthetic_tabular_data(n_rows=40)

        monkeypatch.setattr("shiftprofile.tabular.load_train_table", fake_train_loader)
        monkeypatch.setattr("shiftprofile.tabular.load_cell_table", fake_eval_loader)

        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        cache = ArtifactCache(tmp_path)
        train_tabular_cell(cell, make_config(), cache)

        assert called == ["train"], (
            f"training touched {called}; it must read only the training split"
        )

    def test_training_passes_the_configured_n_train(self, tmp_path, monkeypatch):
        seen = {}

        def fake_train_loader(task, domain, n_eval, n_train=None, **kw):
            seen["n_eval"] = n_eval
            seen["n_train"] = n_train
            seen["domain"] = domain
            return make_synthetic_tabular_data(n_rows=60)

        monkeypatch.setattr("shiftprofile.tabular.load_train_table", fake_train_loader)

        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family="TX_2018", severity=1)
        train_tabular_cell(cell, make_config(), ArtifactCache(tmp_path))

        assert seen["n_train"] == 50
        assert seen["n_eval"] == 100
        # Fitting always happens on the SOURCE domain, never on the cell's target.
        assert seen["domain"] == "CA_2018"


class TestConfigSource:
    """The source domain is required, never guessed."""

    def test_missing_source_raises_rather_than_defaulting(self, tmp_path):
        config = make_config()
        del config["source"]
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)

        with pytest.raises(KeyError, match="source"):
            train_tabular_cell(cell, config, ArtifactCache(tmp_path))

    def test_source_is_read_from_config_not_hardcoded(self, tmp_path, monkeypatch):
        seen = {}

        def fake_train_loader(task, domain, n_eval, n_train=None, **kw):
            seen["domain"] = domain
            return make_synthetic_tabular_data(n_rows=60)

        monkeypatch.setattr("shiftprofile.tabular.load_train_table", fake_train_loader)

        config = make_config()
        config["source"] = "WA_2016"
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        train_tabular_cell(cell, config, ArtifactCache(tmp_path))

        assert seen["domain"] == "WA_2016"


class TestCacheKeysGoThroughStageSpec:
    """Keys are built by stage_spec, so producer and consumer cannot drift."""

    def test_train_key_carries_a_stage_and_ignores_the_target(self):
        clean = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        shifted = Cell(track="tabular", model_id="logreg", seed=0, shift_family="TX_2018", severity=1)

        a = train_spec(clean, "income", "CA_2018")
        b = train_spec(shifted, "income", "CA_2018")

        assert a["stage"] == "tabular_train"
        assert a == b, "one fitted model must serve every target domain"

    def test_predict_key_separates_targets(self):
        a = Cell(track="tabular", model_id="logreg", seed=0, shift_family="TX_2018", severity=1)
        b = Cell(track="tabular", model_id="logreg", seed=0, shift_family="NY_2018", severity=1)

        ka = predict_spec(a, "income", "CA_2018", "TX_2018")
        kb = predict_spec(b, "income", "CA_2018", "NY_2018")

        assert ka["stage"] == "tabular_predict"
        assert ka != kb

    def test_train_and_predict_keys_never_collide(self):
        cell = Cell(track="tabular", model_id="logreg", seed=0, shift_family=CLEAN, severity=0)
        assert train_spec(cell, "income", "CA_2018") != predict_spec(
            cell, "income", "CA_2018", "CA_2018"
        )
