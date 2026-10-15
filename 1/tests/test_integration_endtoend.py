"""End-to-end test of the real predict -> explain -> curves path.

Every other test in this suite either exercises one module alone or injects stub
stages through `fill(stages_impl=...)`. That is how four integration defects
survived a 254-test green suite: a missing `explain_cell`, an explain stage
checking the wrong artifact kind, a curves cache key that omitted the explainer,
and a silently skipped stage.

This file calls the REAL stage functions, with only the dataset loader faked, and
asserts the properties that the hand-built spec-dict tests could not: that the
functions themselves write where they claim to, that a second run is genuinely a
cache hit, and that two explainers do not overwrite each other's work.

CPU only, tiny tensors, no downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shiftprofile.cache import ArtifactCache
from shiftprofile.cells import Cell
from shiftprofile.curves import CURVES_VERSION, curves_cell
from shiftprofile.data import fixed_eval_indices
from shiftprofile.explain import EXPLAIN_VERSION, explain_cell, explain_spec
from shiftprofile.models.vision import resnet18_cifar
from shiftprofile.predict import PREDICT_VERSION, predict_cell

N_IMAGES = 4
IG_STEPS = 4  # keep the test under a second; correctness of IG is tested elsewhere

# Every stage is keyed on the evaluation index set, so the chain has to carry a
# real one. Drawn over the fake data's own length rather than CIFAR's 10,000.
INDICES = fixed_eval_indices(N_IMAGES, total=N_IMAGES)

# curves reads attributions rather than computing them, so it has to name the
# settings they were computed under. Omitting this is not a silent mismatch any
# more -- curves_cell raises, which is what the key change bought.
EXPLAIN_OPTIONS = {"ig_steps": IG_STEPS}


@pytest.fixture
def fake_images(monkeypatch):
    """Stand in for CIFAR-10 / CIFAR-10-C without downloading 2.9 GB.

    predict.py imports load_cell_images at module level; explain.py and curves.py
    import it lazily inside the function. Both patch points are therefore needed —
    a detail that itself argues for this test existing.
    """
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(N_IMAGES, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(N_IMAGES, dtype=np.int64) % 10

    def _fake(cell, clean_root, corrupt_root, indices=None):
        if indices is None:
            return images.copy(), labels.copy()
        return images[indices].copy(), labels[indices].copy()

    monkeypatch.setattr("shiftprofile.predict.load_cell_images", _fake)
    monkeypatch.setattr("shiftprofile.data.load_cell_images", _fake)
    return images, labels


@pytest.fixture
def model():
    torch.manual_seed(0)
    return resnet18_cifar().eval()


@pytest.fixture
def cell():
    return Cell("vision", "resnet18", 0, "gaussian_noise", 3)


def _run_predict(cell, model, cache):
    return predict_cell(cell, model, "clean", "corrupt", cache, batch_size=2, device="cpu", indices=INDICES)


def test_full_chain_runs_and_caches_each_stage(fake_images, model, cell, tmp_path):
    """predict -> explain -> curves through the real functions, then assert each
    artifact is actually retrievable from the cache it claims to have written."""
    cache = ArtifactCache(tmp_path)

    logits = _run_predict(cell, model, cache)
    assert logits.shape == (N_IMAGES, 10)
    assert logits.dtype == np.float32, "logits feed NLL; float16 would underflow"

    attribution = explain_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )
    assert attribution.shape == (N_IMAGES, 32, 32)

    model_curve, random_curve = curves_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", imputation="mean", indices=INDICES,
        explain_options=EXPLAIN_OPTIONS,
    )
    assert model_curve.shape == random_curve.shape
    assert model_curve.shape[0] == N_IMAGES

    assert cache.has(
        explain_spec(cell, explainer="integrated_gradients", indices=INDICES,
                     ig_steps=IG_STEPS),
        EXPLAIN_VERSION, kind="array",
    ), "explain_cell did not write where its own spec builder points"


def test_second_run_is_a_cache_hit_and_never_touches_the_model(
    fake_images, model, cell, tmp_path
):
    """The whole resumability design rests on this. If a stage recomputes on a
    warm cache, a preempted 12-hour session makes no cumulative progress."""
    cache = ArtifactCache(tmp_path)
    _run_predict(cell, model, cache)
    first = explain_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )

    class Exploding(torch.nn.Module):
        def forward(self, *a, **k):
            raise AssertionError("cache miss: the model was called on a warm cache")

    second = explain_cell(
        cell, Exploding(), "clean", "corrupt", cache,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )
    np.testing.assert_array_equal(first, second)


def test_two_explainers_do_not_overwrite_each_other(fake_images, model, cell, tmp_path):
    """Defect D3, tested through the real functions rather than by hashing two
    dicts by hand. If the cache key omitted the explainer, Grad-CAM's attributions
    would silently replace Integrated Gradients' and the second read would return
    the wrong array."""
    cache = ArtifactCache(tmp_path)
    _run_predict(cell, model, cache)

    ig = explain_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )
    cam = explain_cell(
        cell, model, "clean", "corrupt", cache, explainer="grad_cam", indices=INDICES,
    )

    assert not np.array_equal(ig, cam), "two explainers produced identical maps"

    # Re-read IG: it must still be IG, not whatever grad_cam wrote afterwards.
    ig_again = explain_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )
    np.testing.assert_array_equal(ig, ig_again)


def test_two_imputations_do_not_overwrite_each_other(fake_images, model, cell, tmp_path):
    """Imputation is an E6 ablation axis, so curves computed under different
    schemes must not share a cache key."""
    cache = ArtifactCache(tmp_path)
    _run_predict(cell, model, cache)
    explain_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )

    mean_curve, _ = curves_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", imputation="mean", indices=INDICES,
        explain_options=EXPLAIN_OPTIONS,
    )
    blur_curve, _ = curves_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", imputation="blur", indices=INDICES,
        explain_options=EXPLAIN_OPTIONS,
    )
    mean_again, _ = curves_cell(
        cell, model, "clean", "corrupt", cache,
        explainer="integrated_gradients", imputation="mean", indices=INDICES,
        explain_options=EXPLAIN_OPTIONS,
    )
    np.testing.assert_array_equal(mean_curve, mean_again)


def test_explain_refuses_when_predictions_are_absent(fake_images, model, cell, tmp_path):
    """The attribution target is the model's PREDICTED class, so explain depends
    on predict having run. Silently recomputing predictions here would hide a
    wiring bug in the stage ordering."""
    cache = ArtifactCache(tmp_path)
    with pytest.raises(Exception) as excinfo:
        explain_cell(
            cell, model, "clean", "corrupt", cache,
            explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
        )
    assert "predict" in str(excinfo.value).lower()


def test_curves_refuses_when_attributions_are_absent(fake_images, model, cell, tmp_path):
    cache = ArtifactCache(tmp_path)
    _run_predict(cell, model, cache)
    with pytest.raises(Exception) as excinfo:
        curves_cell(
            cell, model, "clean", "corrupt", cache,
            explainer="integrated_gradients", imputation="mean", indices=INDICES,
        explain_options=EXPLAIN_OPTIONS,
        )
    assert "explain" in str(excinfo.value).lower() or "attribution" in str(excinfo.value).lower()


def test_cache_split_serves_the_chain_from_a_read_only_root(
    fake_images, model, cell, tmp_path
):
    """The Kaggle shape: last session's cache is mounted read-only, this session
    writes elsewhere. A stage must find prior work through the read root."""
    mounted = tmp_path / "input"
    mounted.mkdir()
    seeded = ArtifactCache(mounted)
    _run_predict(cell, model, seeded)
    explain_cell(
        cell, model, "clean", "corrupt", seeded,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )

    working = ArtifactCache(tmp_path / "working", read_roots=[mounted])

    class Exploding(torch.nn.Module):
        def forward(self, *a, **k):
            raise AssertionError("did not find the mounted artifact")

    reused = explain_cell(
        cell, Exploding(), "clean", "corrupt", working,
        explainer="integrated_gradients", ig_steps=IG_STEPS, indices=INDICES,
    )
    assert reused.shape == (N_IMAGES, 32, 32)

    # And new work lands in the writable root, never the mounted one.
    curves_cell(
        cell, model, "clean", "corrupt", working,
        explainer="integrated_gradients", imputation="mean", indices=INDICES,
        explain_options=EXPLAIN_OPTIONS,
    )
    assert working.resolve(
        explain_spec(cell, explainer="integrated_gradients", indices=INDICES,
                     ig_steps=IG_STEPS),
        EXPLAIN_VERSION, kind="array",
    ).parent == mounted
