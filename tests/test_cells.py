import numpy as np
from shiftprofile.cache import ArtifactCache
from shiftprofile.cells import Cell, enumerate_cells, worklist

VERSION_A = {
    "models": ["resnet18", "resnet18_augmix", "vit_tiny"],
    "seeds": [0, 1, 2, 3, 4],
    "shift_families": ["gaussian_noise", "shot_noise", "defocus_blur", "fog", "jpeg"],
    "severities": [1, 3, 5],
}


def test_version_a_grid_is_240_cells():
    cells = enumerate_cells(VERSION_A, track="vision")
    assert len(cells) == 240


def test_grid_includes_one_clean_cell_per_model_seed():
    cells = enumerate_cells(VERSION_A, track="vision")
    clean = [c for c in cells if c.severity == 0]
    assert len(clean) == 15
    assert all(c.shift_family == "clean" for c in clean)


def test_cells_are_hashable_and_unique():
    cells = enumerate_cells(VERSION_A, track="vision")
    assert len(set(cells)) == len(cells)


def test_cell_spec_roundtrips_through_cache_key():
    cell = Cell("vision", "resnet18", 0, "fog", 3)
    assert cell.spec() == {
        "track": "vision",
        "model_id": "resnet18",
        "seed": 0,
        "shift_family": "fog",
        "severity": 3,
    }


def test_worklist_excludes_cached_cells(tmp_path):
    cache = ArtifactCache(tmp_path)
    cells = enumerate_cells(VERSION_A, track="vision")
    cache.put_record(cells[0].spec(), "predict-v1", {"done": True})

    pending = worklist(cells, cache, producer_version="predict-v1", kind="record")

    assert len(pending) == 239
    assert cells[0] not in pending


def test_worklist_is_stable_across_calls(tmp_path):
    """Preempted sessions re-derive the work-list; order must not drift."""
    cache = ArtifactCache(tmp_path)
    cells = enumerate_cells(VERSION_A, track="vision")
    assert worklist(cells, cache, "predict-v1", kind="record") == worklist(cells, cache, "predict-v1", kind="record")


def test_worklist_accepts_a_spec_extension(tmp_path):
    """Attributions are keyed on (cell, explainer), not cell alone."""
    cache = ArtifactCache(tmp_path)
    cells = enumerate_cells(VERSION_A, track="vision")
    ig = lambda c: c.spec() | {"explainer": "integrated_gradients"}
    gradcam = lambda c: c.spec() | {"explainer": "grad_cam"}

    cache.put_array(ig(cells[0]), "explain-v1", np.zeros(3))

    assert len(worklist(cells, cache, "explain-v1", kind="array", spec_fn=ig)) == 239
    # the same cell is still pending for a different explainer
    assert len(worklist(cells, cache, "explain-v1", kind="array", spec_fn=gradcam)) == 240
