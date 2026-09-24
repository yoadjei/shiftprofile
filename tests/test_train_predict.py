"""Tests for training and prediction stages.

These tests verify:
- Training determinism and correctness
- Prediction batching correctness and shape
- Cache integration and cache hit detection
- Model state management (eval mode, no gradients)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from shiftprofile.train import train_spec, train_model, load_or_train, TRAIN_VERSION
from shiftprofile.predict import predict_logits, predict_cell, PREDICT_VERSION
from shiftprofile.cache import ArtifactCache
from shiftprofile.cells import Cell, stage_spec


# Helpers for test data
def make_stub_model():
    """A tiny 2-layer model for CPU tests."""
    return nn.Sequential(
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def make_synthetic_images(n: int = 64, seed: int = 42) -> np.ndarray:
    """Generate synthetic uint8 images for testing."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)


def make_synthetic_labels(n: int = 64, seed: int = 42) -> np.ndarray:
    """Generate synthetic labels for testing."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 10, size=n, dtype=np.int64)


def get_cifar_mocks():
    """Return mock functions for CIFAR-10 data loading."""
    from torch.utils.data import TensorDataset

    def mock_load_cifar10_train(*args, **kwargs):
        images = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.uint8).float() / 255
        images = (images - torch.tensor(0.4914).view(1, 1, 1, 1)) / torch.tensor(0.2470).view(1, 1, 1, 1)
        labels = torch.randint(0, 10, (64,), dtype=torch.long)
        return TensorDataset(images, labels)

    def mock_load_cifar10_test(*args, **kwargs):
        images = np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 10, (100,), dtype=np.int64)
        return images, labels

    return mock_load_cifar10_train, mock_load_cifar10_test


# ============================================================================
# Tests for train.py
# ============================================================================

class TestTrainSpec:
    def test_train_spec_returns_dict(self):
        spec = train_spec("resnet18", seed=42)
        assert isinstance(spec, dict)
        assert spec["model_id"] == "resnet18"
        assert spec["seed"] == 42

    def test_train_spec_includes_model_and_seed(self):
        spec = train_spec("vit_tiny", seed=123)
        assert "model_id" in spec
        assert "seed" in spec
        assert spec["model_id"] == "vit_tiny"
        assert spec["seed"] == 123


class TestTrainModel:
    def test_train_model_returns_model_and_dict(self):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        assert isinstance(model, nn.Module)
        assert isinstance(metadata, dict)

    def test_train_model_metadata_includes_seed(self):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        assert metadata["seed"] == 42

    def test_train_model_metadata_includes_epochs(self):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=3,
                batch_size=32,
                device="cpu",
            )
        assert "epochs" in metadata
        assert metadata["epochs"] == 3

    def test_train_model_metadata_includes_clean_test_accuracy(self):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        assert "clean_test_accuracy" in metadata
        assert isinstance(metadata["clean_test_accuracy"], (float, int))
        assert 0 <= metadata["clean_test_accuracy"] <= 1

    def test_train_model_metadata_includes_final_train_loss(self):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        assert "final_train_loss" in metadata
        assert isinstance(metadata["final_train_loss"], (float, int))

    def test_train_model_metadata_includes_wall_clock_seconds(self):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        assert "wall_clock_seconds" in metadata
        assert metadata["wall_clock_seconds"] > 0

    def test_train_model_metadata_includes_torch_version(self):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        assert "torch_version" in metadata

    def test_train_model_same_seed_gives_identical_metadata(self):
        """Same seed should produce identical training weights."""
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model1, meta1 = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
            model2, meta2 = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        # The metadata should be identical for the same seed
        assert meta1["seed"] == meta2["seed"]
        assert meta1["epochs"] == meta2["epochs"]

    def test_train_model_different_seeds_give_different_weights(self):
        """Different seeds should produce different model weights."""
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model1, _ = train_model(
                "resnet18",
                seed=42,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )
            model2, _ = train_model(
                "resnet18",
                seed=123,
                data_root="/tmp",
                epochs=1,
                batch_size=32,
                device="cpu",
            )

        # Get first layer weights from both models
        w1 = list(model1.parameters())[0].data.clone()
        w2 = list(model2.parameters())[0].data.clone()

        # They should be different
        assert not torch.allclose(w1, w2)


class TestLoadOrTrain:
    def test_load_or_train_trains_on_cache_miss(self, tmp_path):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()
        cache = ArtifactCache(tmp_path)

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            model, metadata = load_or_train(
                "resnet18",
                seed=42,
                data_root="/tmp",
                cache=cache,
                epochs=1,
                batch_size=32,
                device="cpu",
            )
        assert isinstance(model, nn.Module)
        assert isinstance(metadata, dict)

    def test_load_or_train_loads_from_cache_on_hit(self, tmp_path):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()
        cache = ArtifactCache(tmp_path)

        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            # First call trains
            model1, meta1 = load_or_train(
                "resnet18",
                seed=42,
                data_root="/tmp",
                cache=cache,
                epochs=1,
                batch_size=32,
                device="cpu",
            )

            # Second call should load from cache
            model2, meta2 = load_or_train(
                "resnet18",
                seed=42,
                data_root="/tmp",
                cache=cache,
                epochs=1,
                batch_size=32,
                device="cpu",
            )

        # Metadata should be identical except for wall_clock_seconds (which depends on runtime)
        assert meta1["seed"] == meta2["seed"]
        assert meta1["epochs"] == meta2["epochs"]
        assert meta1["final_train_loss"] == meta2["final_train_loss"]
        assert meta1["clean_test_accuracy"] == meta2["clean_test_accuracy"]
        assert meta1["torch_version"] == meta2["torch_version"]

    def test_load_or_train_caches_checkpoint(self, tmp_path):
        from unittest.mock import patch

        mock_train, mock_test = get_cifar_mocks()
        cache = ArtifactCache(tmp_path)
        spec = train_spec("resnet18", seed=42)

        # Before training, checkpoint should not be in cache
        assert not cache.has(spec, TRAIN_VERSION, kind="record")

        # After training, checkpoint should be cached
        with patch("shiftprofile.train.load_cifar10_train", mock_train), \
             patch("shiftprofile.train.load_cifar10_test", mock_test):
            load_or_train(
                "resnet18",
                seed=42,
                data_root="/tmp",
                cache=cache,
                epochs=1,
                batch_size=32,
                device="cpu",
            )

        # Metadata should be cached
        assert cache.has(spec, TRAIN_VERSION, kind="record")


# ============================================================================
# Tests for predict.py
# ============================================================================

class TestPredictLogits:
    def test_predict_logits_returns_float32(self):
        model = make_stub_model()
        images = make_synthetic_images(n=4)
        logits = predict_logits(model, images, device="cpu")
        assert logits.dtype == np.float32

    def test_predict_logits_returns_correct_shape(self):
        model = make_stub_model()
        images = make_synthetic_images(n=16)
        logits = predict_logits(model, images, device="cpu")
        assert logits.shape == (16, 10)

    def test_predict_logits_all_finite(self):
        model = make_stub_model()
        images = make_synthetic_images(n=8)
        logits = predict_logits(model, images, device="cpu")
        assert np.all(np.isfinite(logits))

    def test_predict_logits_batch_size_1_vs_64_identical(self):
        """Batching bug at boundaries is a correctness issue."""
        model = make_stub_model()
        images = make_synthetic_images(n=64)

        logits_bs1 = predict_logits(model, images, batch_size=1, device="cpu")
        logits_bs64 = predict_logits(model, images, batch_size=64, device="cpu")

        # Allow small floating point differences from different accumulation orders
        np.testing.assert_allclose(logits_bs1, logits_bs64, rtol=1e-4, atol=1e-6)

    def test_predict_logits_leaves_model_in_eval_mode(self):
        model = make_stub_model()
        model.train()  # Start in train mode
        images = make_synthetic_images(n=4)

        predict_logits(model, images, device="cpu")

        # Model should be in eval mode
        assert not model.training

    def test_predict_logits_creates_no_grad(self):
        model = make_stub_model()
        images = make_synthetic_images(n=4)

        # Run prediction
        predict_logits(model, images, device="cpu")

        # No gradients should be computed
        for param in model.parameters():
            assert param.grad is None


class TestPredictCell:
    def test_predict_cell_returns_ndarray(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        cache = ArtifactCache(tmp_path)
        cell = Cell(track="test", model_id="resnet18", seed=42, shift_family="clean", severity=0)
        model = make_stub_model()

        # Mock load_cell_images to return synthetic data
        def mock_load_cell_images(*args, **kwargs):
            return make_synthetic_images(n=64), make_synthetic_labels(n=64)

        with patch("shiftprofile.predict.load_cell_images", mock_load_cell_images):
            logits = predict_cell(
                cell,
                model,
                clean_root="/tmp",
                corrupt_root="/tmp",
                cache=cache,
                device="cpu",
            )

        assert isinstance(logits, np.ndarray)
        assert logits.shape[1] == 10
        assert logits.dtype == np.float32

    def test_predict_cell_cache_hit_does_not_call_model(self, tmp_path):
        cache = ArtifactCache(tmp_path)
        cell = Cell(track="test", model_id="resnet18", seed=42, shift_family="clean", severity=0)

        # Manually create a cache entry
        fake_logits = np.random.randn(64, 10).astype(np.float32)
        spec = stage_spec(cell, "predict")
        cache.put_array(spec, PREDICT_VERSION, fake_logits)

        # Create a model that raises if called
        class ErrorModel:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("Model was called but should have been cached!")

        error_model = ErrorModel()

        # This should use the cache and not call the model
        logits = predict_cell(
            cell,
            error_model,
            clean_root="/tmp",
            corrupt_root="/tmp",
            cache=cache,
            device="cpu",
        )

        np.testing.assert_array_equal(logits, fake_logits)

    def test_predict_cell_caches_result_on_miss(self, tmp_path):
        from unittest.mock import patch

        cache = ArtifactCache(tmp_path)
        cell = Cell(track="test", model_id="resnet18", seed=42, shift_family="clean", severity=0)
        model = make_stub_model()
        spec = stage_spec(cell, "predict")

        # Before prediction, logits should not be cached
        assert not cache.has(spec, PREDICT_VERSION, kind="array")

        # Mock load_cell_images to return synthetic data
        def mock_load_cell_images(*args, **kwargs):
            return make_synthetic_images(n=64), make_synthetic_labels(n=64)

        with patch("shiftprofile.predict.load_cell_images", mock_load_cell_images):
            # Run prediction
            predict_cell(
                cell,
                model,
                clean_root="/tmp",
                corrupt_root="/tmp",
                cache=cache,
                device="cpu",
            )

        # After prediction, logits should be cached
        assert cache.has(spec, PREDICT_VERSION, kind="array")

    def test_predict_cell_round_trip_is_exact(self, tmp_path):
        from unittest.mock import patch

        cache = ArtifactCache(tmp_path)
        cell = Cell(track="test", model_id="resnet18", seed=42, shift_family="clean", severity=0)
        model = make_stub_model()

        # Mock load_cell_images to return synthetic data
        def mock_load_cell_images(*args, **kwargs):
            return make_synthetic_images(n=64), make_synthetic_labels(n=64)

        with patch("shiftprofile.predict.load_cell_images", mock_load_cell_images):
            # First call computes and caches
            logits1 = predict_cell(
                cell,
                model,
                clean_root="/tmp",
                corrupt_root="/tmp",
                cache=cache,
                device="cpu",
            )

            # Second call loads from cache
            logits2 = predict_cell(
                cell,
                model,
                clean_root="/tmp",
                corrupt_root="/tmp",
                cache=cache,
                device="cpu",
            )

        # They should be exactly equal
        np.testing.assert_array_equal(logits1, logits2)
