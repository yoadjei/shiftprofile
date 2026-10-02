"""An artifact's key must name everything that determines its contents.

The same mistake was made four times in this codebase, and each instance was
invisible in any single run:

  1. `n_eval_images` sat in both configs and reached no producer, so every run
     scored all 10,000 test images while the protocol asked for 1,000 -- and the
     key recorded neither, so the two could not be told apart.
  2. Curves keyed on neither the attributions nor the predictions they were
     computed from, so bumping `explain-v1` to `explain-v2` recomputed the
     attributions and left the curves pointing at inputs that no longer existed.
  3. `fill` built the curves key without the `which` axis `curves_cell` writes
     under, so the check could never hit and a resumed run re-reported every
     curve as freshly completed.
  4. The IG baseline and step count were absent from the key, so E6 -- which
     exists to vary the baseline -- would have collided with the frozen protocol.

These tests are the properties, not the four bugs. A fifth instance of the same
mistake should fail here without anyone having to think of it as a new case.
"""

from __future__ import annotations

import numpy as np
import pytest

from shiftprofile.cache import ArtifactCache, spec_key
from shiftprofile.cells import Cell, stage_spec
from shiftprofile.curves import CURVES_VERSION, curves_is_cached, curves_spec
from shiftprofile.data import eval_digest, fixed_eval_indices
from shiftprofile.explain import EXPLAIN_VERSION, explain_spec, explainer_options
from shiftprofile.predict import PREDICT_VERSION, predict_spec


CELL = Cell("vision", "resnet18", 0, "fog", 3)
THOUSAND = fixed_eval_indices(1000)
TEN_THOUSAND = fixed_eval_indices(10000)


class TestEvalDigest:
    """The index set is part of the key, so it must be named and portable."""

    def test_none_is_refused_rather_than_standing_for_the_whole_test_set(self):
        """The single most important line in this file.

        Every design reviewed for this change proposed `digest(None) -> "none"`.
        That sentinel is the bug: it means a forgotten index set still produces a
        valid-looking key, which is exactly how 10,000-image artifacts came to
        share a key with the 1,000 the protocol asked for. Refusing makes the
        omission a crash.
        """
        with pytest.raises(TypeError, match="requires an index set"):
            eval_digest(None)

    def test_a_list_is_refused_because_its_width_is_not_portable(self):
        """`permutation` returns int32 on Windows and int64 on Linux.

        Coercing here would let two machines compute identical indices and hash
        them to different keys, which reads as a reproducibility failure.
        """
        with pytest.raises(TypeError, match="numpy array"):
            eval_digest([0, 1, 2])

    def test_a_non_int64_array_is_refused(self):
        with pytest.raises(TypeError, match="int64"):
            eval_digest(np.array([0, 1, 2], dtype=np.int32))

    def test_an_empty_index_set_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            eval_digest(np.array([], dtype=np.int64))

    def test_the_digest_is_pinned(self):
        """Pinned, not merely deterministic.

        A digest that drifted between numpy versions or between a laptop and a
        Kaggle runner would silently invalidate a cache that cost GPU quota to
        fill. The byte order is pinned to little-endian for the same reason, so
        these values do not depend on the host.
        """
        assert eval_digest(np.arange(4, dtype=np.int64)) == "a1e03200f1f82ad2"
        assert eval_digest(fixed_eval_indices(1000)) == "60c5b821cd131d37"

    def test_different_sizes_give_different_digests(self):
        assert eval_digest(THOUSAND) != eval_digest(TEN_THOUSAND)

    def test_a_prefix_is_not_the_same_set(self):
        """`fixed_eval_indices` makes smaller n a prefix of larger n, which keeps
        subsets comparable. It must not make them the same artifact."""
        assert eval_digest(fixed_eval_indices(500)) != eval_digest(THOUSAND)


class TestEvalSetIsInEveryKey:
    """Defect 1: artifacts over 1,000 images and over 10,000 shared one key."""

    def test_predict_keys_differ_by_eval_set(self):
        assert predict_spec(CELL, indices=THOUSAND) != predict_spec(
            CELL, indices=TEN_THOUSAND
        )

    def test_explain_keys_differ_by_eval_set(self):
        a = explain_spec(CELL, explainer="grad_cam", indices=THOUSAND)
        b = explain_spec(CELL, explainer="grad_cam", indices=TEN_THOUSAND)
        assert a != b

    def test_curves_keys_differ_by_eval_set(self):
        common = dict(explainer="grad_cam", imputation="mean", which="model")
        a = curves_spec(CELL, indices=THOUSAND, **common)
        b = curves_spec(CELL, indices=TEN_THOUSAND, **common)
        assert a != b

    @pytest.mark.parametrize("builder", [
        lambda i: predict_spec(CELL, indices=i),
        lambda i: explain_spec(CELL, explainer="grad_cam", indices=i),
        lambda i: curves_spec(CELL, explainer="grad_cam", imputation="mean",
                              which="model", indices=i),
    ])
    def test_every_stage_refuses_to_build_a_key_without_one(self, builder):
        with pytest.raises(TypeError):
            builder(None)


class TestUpstreamVersionsAreInTheKey:
    """Defect 2: a derived artifact outliving the inputs it was derived from."""

    def test_attributions_name_the_predict_version_they_read(self):
        spec = explain_spec(CELL, explainer="grad_cam", indices=THOUSAND)
        assert spec["inputs"] == {"predict": PREDICT_VERSION}

    def test_curves_name_both_versions_they_read(self):
        spec = curves_spec(
            CELL, explainer="grad_cam", imputation="mean", which="model",
            indices=THOUSAND,
        )
        assert spec["inputs"] == {
            "explain": EXPLAIN_VERSION,
            "predict": PREDICT_VERSION,
        }

    def test_bumping_an_upstream_version_changes_the_downstream_key(self, monkeypatch):
        """The property that makes the explain-v2 bump safe.

        Without it, recomputing attributions left the curves keyed as though
        nothing had changed, so a resumed run served curves built on attributions
        that were no longer on disk.
        """
        before = spec_key(
            curves_spec(CELL, explainer="grad_cam", imputation="mean",
                        which="model", indices=THOUSAND),
            CURVES_VERSION,
        )
        monkeypatch.setattr("shiftprofile.explain.EXPLAIN_VERSION", "explain-v99")
        after = spec_key(
            curves_spec(CELL, explainer="grad_cam", imputation="mean",
                        which="model", indices=THOUSAND),
            CURVES_VERSION,
        )
        assert before != after, (
            "an explain bump left the curves key unchanged, so stale curves would "
            "be served against attributions that had been recomputed"
        )


class TestCurvesNeedBothHalves:
    """Defect 3: the check that could never hit."""

    def _seed(self, cache, *which, indices=THOUSAND):
        for w in which:
            cache.put_array(
                curves_spec(CELL, explainer="grad_cam", imputation="mean",
                            which=w, indices=indices),
                CURVES_VERSION,
                np.zeros((len(indices), 9), dtype=np.float32),
            )

    def test_both_present_is_cached(self, tmp_path):
        cache = ArtifactCache(tmp_path)
        self._seed(cache, "model", "random")
        assert curves_is_cached(
            CELL, cache, explainer="grad_cam", imputation="mean", indices=THOUSAND
        )

    @pytest.mark.parametrize("present", ["model", "random"])
    def test_one_half_alone_is_not_a_result(self, tmp_path, present):
        """`relative_faithfulness` is the control's AUC minus the model's, so a
        unit holding one curve has nothing to report and must be recomputed."""
        cache = ArtifactCache(tmp_path)
        self._seed(cache, present)
        assert not curves_is_cached(
            CELL, cache, explainer="grad_cam", imputation="mean", indices=THOUSAND
        )

    def test_the_two_halves_do_not_share_a_key(self, tmp_path):
        model = curves_spec(CELL, explainer="grad_cam", imputation="mean",
                            which="model", indices=THOUSAND)
        random = curves_spec(CELL, explainer="grad_cam", imputation="mean",
                             which="random", indices=THOUSAND)
        assert spec_key(model, CURVES_VERSION) != spec_key(random, CURVES_VERSION)


class TestExplainerOptionsAreInTheKey:
    """Defect 4: E6 varies the IG baseline, which was absent from the key."""

    def test_the_baseline_changes_the_key(self):
        black = explain_spec(CELL, explainer="integrated_gradients",
                             indices=THOUSAND, baseline="black")
        blur = explain_spec(CELL, explainer="integrated_gradients",
                            indices=THOUSAND, baseline="blur")
        assert black != blur, "E6's varied baseline would collide with the protocol's"

    def test_the_step_count_changes_the_key(self):
        a = explain_spec(CELL, explainer="integrated_gradients",
                         indices=THOUSAND, ig_steps=32)
        b = explain_spec(CELL, explainer="integrated_gradients",
                         indices=THOUSAND, ig_steps=64)
        assert a != b

    def test_the_random_explainers_seed_changes_the_key(self):
        a = explain_spec(CELL, explainer="random", indices=THOUSAND, random_seed=0)
        b = explain_spec(CELL, explainer="random", indices=THOUSAND, random_seed=1)
        assert a != b

    def test_an_option_an_explainer_ignores_does_not_fork_its_cache(self):
        """Grad-CAM reads no options, so changing IG's step count must not
        invalidate a Grad-CAM artifact. Keying every explainer on every option
        would fork the cache for nothing, and quota is the binding constraint."""
        a = explain_spec(CELL, explainer="grad_cam", indices=THOUSAND, ig_steps=32)
        b = explain_spec(CELL, explainer="grad_cam", indices=THOUSAND, ig_steps=64)
        assert a == b
        assert explainer_options("grad_cam") == {}

    def test_curves_distinguish_the_attributions_they_were_built_from(self):
        """A curve over blur-baseline attributions is not the black-baseline one."""
        black = curves_spec(
            CELL, explainer="integrated_gradients", imputation="mean",
            which="model", indices=THOUSAND, explain_options={"baseline": "black"},
        )
        blur = curves_spec(
            CELL, explainer="integrated_gradients", imputation="mean",
            which="model", indices=THOUSAND, explain_options={"baseline": "blur"},
        )
        assert black != blur

    def test_curves_normalise_the_options_so_equivalent_spellings_agree(self):
        """Explicitly passing the defaults must name the same artifact as omitting
        them, or a caller who spells the protocol out gets a second cache."""
        implicit = curves_spec(
            CELL, explainer="integrated_gradients", imputation="mean",
            which="model", indices=THOUSAND,
        )
        explicit = curves_spec(
            CELL, explainer="integrated_gradients", imputation="mean",
            which="model", indices=THOUSAND,
            explain_options={"baseline": "black", "ig_steps": 32},
        )
        assert implicit == explicit


class TestControlSeedIsNamedForWhatItSeeds:
    """Two different quantities once shared the name `random_seed`.

    `explain_cell`'s seeds the `random` EXPLAINER's attribution map;
    `curves_cell`'s seeds the random CONTROL curve. Adjacent stages, same name,
    different meanings -- a mistake waiting to be made, so the curves one is now
    `control_seed`.
    """

    def test_the_control_seed_changes_the_curve_key(self):
        a = curves_spec(CELL, explainer="grad_cam", imputation="mean",
                        which="random", indices=THOUSAND, control_seed=0)
        b = curves_spec(CELL, explainer="grad_cam", imputation="mean",
                        which="random", indices=THOUSAND, control_seed=1)
        assert a != b

    def test_curves_spec_has_no_parameter_called_random_seed(self):
        import inspect

        params = inspect.signature(curves_spec).parameters
        assert "control_seed" in params
        assert "random_seed" not in params, (
            "the name is ambiguous next to explain_cell's random_seed, which "
            "seeds something else entirely"
        )


class TestStageSpecRefusesWhatItCannotHash:
    def test_an_empty_inputs_dict_is_refused(self):
        """Recording that a stage reads nothing is not the same as reading
        nothing; the distinction would otherwise be a silently empty axis."""
        with pytest.raises(ValueError, match="omit it entirely"):
            stage_spec(CELL, "predict", inputs={})

    def test_the_new_axes_are_absent_when_not_supplied(self):
        """A stage that evaluates no images and reads no artifact -- training --
        must not acquire empty axes that change its key. The 15 trained
        checkpoints are the expensive part of Version A and must survive."""
        spec = stage_spec(CELL, "train")
        assert "evaluated" not in spec
        assert "inputs" not in spec


class TestTheConfigReachesTheProducer:
    """The end-to-end property. Everything above is a key; this is the artifact.

    The migration reviewer named this the one mandatory test: if `fill` ever stops
    resolving the index set correctly, every key silently loses the axis that
    distinguishes 1,000 images from 10,000, the tests above still pass, and the
    run quietly produces artifacts that do not match the registered protocol. The
    only defence is asserting on the shape of what actually lands on disk.
    """

    def _run(self, tmp_path, n_eval_images, monkeypatch):
        import torch

        from shiftprofile.cache import ArtifactCache
        from shiftprofile.fill import fill
        from shiftprofile.models.vision import resnet18_cifar
        from shiftprofile.predict import predict_cell

        # Stand in for CIFAR without downloading it, honouring indices the way the
        # real loader does so the subsetting is actually exercised.
        rng = np.random.default_rng(0)
        images = rng.integers(0, 256, size=(64, 32, 32, 3), dtype=np.uint8)
        labels = (np.arange(64) % 10).astype(np.int64)

        def fake_load(cell, clean_root, corrupt_root, indices=None):
            if indices is None:
                return images.copy(), labels.copy()
            return images[indices].copy(), labels[indices].copy()

        monkeypatch.setattr("shiftprofile.predict.load_cell_images", fake_load)
        monkeypatch.setattr(
            "shiftprofile.data.cifar.fixed_eval_indices",
            lambda n, total=64, seed=20260923: fixed_eval_indices(n, total=64, seed=seed),
        )
        monkeypatch.setattr(
            "shiftprofile.data.fixed_eval_indices",
            lambda n, total=64, seed=20260923: fixed_eval_indices(n, total=64, seed=seed),
        )

        torch.manual_seed(0)
        model = resnet18_cifar().eval()
        cache = ArtifactCache(tmp_path / f"cache{n_eval_images}")
        config = {
            "track": "vision",
            "n_eval_images": n_eval_images,
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["fog"],
            "severities": [1],
            "explainers": ["random"],
            "imputations": ["mean"],
        }

        captured = {}

        def predict_stage(cell, cache, *, indices, device="cpu", **kw):
            out = predict_cell(
                cell, model, "clean", "corrupt", cache,
                indices=indices, device=device, batch_size=16,
            )
            captured.setdefault("rows", out.shape[0])
            captured.setdefault("indices", indices)
            return out

        report = fill(
            config, cache, budget_minutes=60, stages=("predict",),
            stages_impl={"predict": predict_stage},
        )
        return report, captured, cache, config

    @pytest.mark.parametrize("n_eval_images", [8, 32])
    def test_the_artifact_has_as_many_rows_as_the_config_asked_for(
        self, tmp_path, monkeypatch, n_eval_images
    ):
        _, captured, _, _ = self._run(tmp_path, n_eval_images, monkeypatch)
        assert captured["rows"] == n_eval_images, (
            f"config asked for {n_eval_images} images and the artifact has "
            f"{captured['rows']} rows; n_eval_images is not reaching the producer"
        )

    def test_two_eval_sizes_do_not_overwrite_each_other(self, tmp_path, monkeypatch):
        """The defect, stated as a test. For most of this project's life these two
        runs wrote to one key and the second silently replaced the first."""
        _, small, cache_small, cfg_small = self._run(tmp_path, 8, monkeypatch)
        _, large, cache_large, cfg_large = self._run(tmp_path, 32, monkeypatch)

        cell = Cell("vision", "resnet18", 0, "clean", 0)
        assert spec_key(
            predict_spec(cell, indices=small["indices"]), PREDICT_VERSION
        ) != spec_key(
            predict_spec(cell, indices=large["indices"]), PREDICT_VERSION
        )

    def test_a_config_without_n_eval_images_is_refused(self, tmp_path):
        """No default. A guess here is how the whole defect stayed invisible."""
        from shiftprofile.cache import ArtifactCache
        from shiftprofile.fill import fill

        with pytest.raises(ValueError, match="n_eval_images"):
            fill(
                {"track": "vision", "models": ["resnet18"], "seeds": [0],
                 "shift_families": ["fog"], "severities": [1]},
                ArtifactCache(tmp_path),
                budget_minutes=60,
                stages=("predict",),
                stages_impl={"predict": lambda *a, **k: np.zeros(1)},
            )

    def test_a_warm_second_run_skips_the_predict_stage(self, tmp_path, monkeypatch):
        """Resumability at the predict stage. The curves case, which is the one
        that was broken, is covered by the full chain below."""
        report1, captured, cache, config = self._run(tmp_path, 8, monkeypatch)
        assert report1.completed and not report1.skipped_cached

        from shiftprofile.fill import fill

        def should_not_run(cell, cache, **kw):
            raise AssertionError("a cached unit was recomputed on a warm cache")

        report2 = fill(
            config, cache, budget_minutes=60, stages=("predict",),
            stages_impl={"predict": should_not_run},
        )
        assert not report2.completed
        assert len(report2.skipped_cached) == len(report1.completed)


class TestAWarmRunSkipsTheWholeChain:
    """Defect 3 at the level it actually mattered: `fill`'s report.

    `fill` built the curves key without the `which` axis `curves_cell` writes
    under, so the check could never hit. A resumed session reported thirty curves
    "completed" every time while `curves_cell` short-circuited internally and did
    nothing -- cheap, but it meant the report could not be trusted to say what a
    preempted run had actually accomplished, which is the one thing it exists for.

    This runs the real predict, explain and curves functions, so it fails if any
    of the three disagrees with its own predicate.
    """

    def test_every_stage_reports_skipped_on_the_second_run(self, tmp_path, monkeypatch):
        import torch

        from shiftprofile.cache import ArtifactCache
        from shiftprofile.curves import curves_cell
        from shiftprofile.explain import explain_cell
        from shiftprofile.fill import fill
        from shiftprofile.models.vision import resnet18_cifar
        from shiftprofile.predict import predict_cell

        rng = np.random.default_rng(0)
        images = rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
        labels = (np.arange(8) % 10).astype(np.int64)

        def fake_load(cell, clean_root, corrupt_root, indices=None):
            if indices is None:
                return images.copy(), labels.copy()
            return images[indices].copy(), labels[indices].copy()

        for target in ("shiftprofile.predict.load_cell_images",
                       "shiftprofile.data.load_cell_images"):
            monkeypatch.setattr(target, fake_load)
        for target in ("shiftprofile.data.cifar.fixed_eval_indices",
                       "shiftprofile.data.fixed_eval_indices"):
            monkeypatch.setattr(
                target,
                lambda n, total=8, seed=20260923: fixed_eval_indices(n, total=8, seed=seed),
            )

        torch.manual_seed(0)
        model = resnet18_cifar().eval()
        cache = ArtifactCache(tmp_path / "cache")
        config = {
            "track": "vision",
            "n_eval_images": 8,
            "ig_steps": 2,  # keep it quick; the step count is keyed, not tested here
            "models": ["resnet18"],
            "seeds": [0],
            "shift_families": ["fog"],
            "severities": [1],
            "explainers": ["integrated_gradients"],
            "imputations": ["mean"],
        }

        def stages(forbidden=False):
            def guard(name):
                def run(cell, cache, **kw):
                    if forbidden:
                        raise AssertionError(
                            f"{name} recomputed a unit that was already cached"
                        )
                    kw.pop("device", None)
                    if name == "predict":
                        return predict_cell(cell, model, "c", "c", cache,
                                            batch_size=8, **kw)
                    if name == "explain":
                        return explain_cell(cell, model, "c", "c", cache, **kw)
                    return curves_cell(cell, model, "c", "c", cache, **kw)
                return run
            return {n: guard(n) for n in ("predict", "explain", "curves")}

        first = fill(config, cache, budget_minutes=60, stages_impl=stages())
        assert not first.failed, first.failed
        assert first.completed and not first.skipped_cached

        second = fill(config, cache, budget_minutes=60, stages_impl=stages(forbidden=True))
        assert not second.completed, (
            f"a warm run recomputed {second.completed}; resumability is broken"
        )
        assert sorted(second.skipped_cached) == sorted(first.completed)
        assert any(c.startswith("curves:") for c in second.skipped_cached), (
            "curves were never reported as cached, which is the defect itself"
        )
