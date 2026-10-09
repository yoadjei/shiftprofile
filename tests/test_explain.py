"""Tests for attribution methods (IG, Grad-CAM, baselines).

These tests verify that attribution methods satisfy their mathematical properties,
handle edge cases correctly, and produce reproducible outputs.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy import ndimage

from shiftprofile.explain import (
    BASE_EXPLAINERS,
    DEFAULT_IG_BASELINE,
    EXPLAIN_VERSION,
    IG_BASELINES,
    IG_STEPS,
    LOWRES_GRIDS,
    LOWRES_PREFIX,
    ROLLABLE,
    ROLLED_SUFFIX,
    ExplainerName,
    _ig_baseline,
    integrated_gradients,
    grad_cam,
    explain_options_from_config,
    parse_explainer,
    random_attribution,
    random_lowres_attribution,
    roll_attribution,
    explain_batch,
    to_common_grid,
    ig_completeness_error,
)
from shiftprofile.models.vision import resnet18_cifar


# Test models for CPU-only testing
def _tiny_model() -> nn.Module:
    """Minimal model for fast CPU tests."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )


def _tiny_conv_model() -> nn.Module:
    """Minimal model with a features attribute for Grad-CAM."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    model.features = model
    return model


# ============================================================================
# Integrated Gradients Tests
# ============================================================================

def test_integrated_gradients_output_shape():
    """IG should return (n, 32, 32) with channels summed."""
    model = _tiny_model().eval()
    x = torch.randn(3, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0)
    assert attr.shape == (3, 32, 32)
    assert attr.dtype == np.float32


def test_integrated_gradients_default_steps():
    """IG should use IG_STEPS by default."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    # Just verify it runs without error; step count is internal
    attr = integrated_gradients(model, x, target=0)
    assert attr.shape == (1, 32, 32)


def test_integrated_gradients_custom_steps():
    """IG should accept custom step counts."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr_32 = integrated_gradients(model, x, target=0, steps=32)
    attr_64 = integrated_gradients(model, x, target=0, steps=64)
    assert attr_32.shape == attr_64.shape == (1, 32, 32)
    # More steps should give different (hopefully better) result
    assert not np.allclose(attr_32, attr_64)


def test_the_default_baseline_is_the_mean_and_is_named_for_it():
    """Replaces a test whose docstring was "IG with black baseline (zeros) should
    be the default" -- which stated the defect as the requirement.

    `baseline="black"` set `torch.zeros_like(x)`, and normalisation maps the
    dataset MEAN to zero, so the string named a mean baseline. The old test
    asserted `black == black` and so could never have noticed. Same shape as
    `test_impute_full_mask_modifies_all_pixels`, which special-cased `zero` to
    assert the result was all zeros and thereby made the duplicate arm look
    correct; that one was replaced rather than patched, and so is this.

    The default behaviour is unchanged -- zeros, which is what has always run --
    so what is pinned here is that the default's NAME is now the one that
    describes it.
    """
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)

    assert DEFAULT_IG_BASELINE == "mean"
    np.testing.assert_array_equal(
        integrated_gradients(model, x, target=0),
        integrated_gradients(model, x, target=0, baseline="mean"),
    )
    np.testing.assert_array_equal(
        _ig_baseline(x, "mean", "cpu").numpy(), np.zeros_like(x.numpy())
    )


def test_mean_and_black_baselines_produce_different_attributions():
    """The test the `zero` imputation bug lacked, on the axis E6 varies.

    Without it an ablation arm that duplicates another is invisible: the `zero`
    scheme ran the same line as `mean` for the life of the project and was only
    caught when a 120-curve sweep showed the two agreeing to five decimals.
    Attribution is `grad * (x - baseline)`, so two baselines that are the same
    tensor give the same attribution, and "black vs blur" would have been
    "mean vs blur" with a third of the sweep spent re-measuring one arm.
    """
    model = _tiny_model().eval()
    x = torch.randn(4, 3, 32, 32)

    attr_mean = integrated_gradients(model, x, target=0, baseline="mean")
    attr_black = integrated_gradients(model, x, target=0, baseline="black")

    assert not np.allclose(attr_mean, attr_black), (
        "the mean and black baselines gave the same attribution, which is the "
        "degeneracy that made the retired `zero` imputation a duplicate arm"
    )


def test_the_black_baseline_is_black_in_pixel_space_not_normalised_zero():
    """What `black` has to be for the name to be true: -mean/std per channel.

    Normalised zero is the dataset mean. A baseline of zeros under the name
    `black` is the same conflation found in the `zero` imputation scheme, and it
    sat on a pre-registered ablation axis.
    """
    from shiftprofile.data.cifar import normalised_black

    x = torch.randn(2, 3, 32, 32)
    baseline = _ig_baseline(x, "black", "cpu")

    expected = torch.tensor(normalised_black()).view(1, 3, 1, 1).expand_as(x)
    torch.testing.assert_close(baseline, expected)
    assert not torch.allclose(baseline, torch.zeros_like(baseline))


def test_integrated_gradients_blur_baseline():
    """IG with blur baseline should produce different result from black."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr_black = integrated_gradients(model, x, target=0, baseline="black")
    attr_blur = integrated_gradients(model, x, target=0, baseline="blur")
    # Different baselines should produce different attributions
    assert not np.allclose(attr_black, attr_blur)


def test_integrated_gradients_unknown_baseline_raises():
    """Unknown baseline should raise with helpful error."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    with pytest.raises(ValueError, match="baseline|black|blur"):
        integrated_gradients(model, x, target=0, baseline="unknown")


@pytest.mark.parametrize("baseline", IG_BASELINES)
def test_integrated_gradients_satisfies_completeness(baseline):
    """IG's defining property: attributions sum to F(x) - F(baseline).
    If this fails the implementation is wrong, not merely imprecise.

    Parametrised over every baseline, and the reference is built by the same
    `_ig_baseline` the attribution was built from. It used to take `attr` with
    `baseline="black"` and `delta` from `model(torch.zeros_like(x))`, which was
    self-consistent only while "black" meant zeros: the two sides agreed about
    which baseline was in play because both spelled the bug the same way. Written
    this way the test would have caught the misnomer instead of hiding it -- a
    completeness check against a DIFFERENT baseline from the attribution's is a
    spurious error, and it reported none.

    Measured at this seed, steps=128: relative error 7.4e-3 (mean), 4.1e-3
    (black), 5.4e-3 (blur), so rel=0.02 leaves 2.7x headroom on the worst of the
    three. The black baseline needs no extra steps despite sitting 2.4x further
    from x (mean |x - baseline| 1.91 vs 0.80): the Riemann-sum error grows with
    the path length but F(x) - F(baseline) grows with it faster, so the RELATIVE
    error if anything improves.

    The model is seeded because the error is a ratio whose denominator can land
    near zero: over 64 random draws at steps=128 the median relative error is
    4.8e-3 but the worst is 1.8e-1 (mean) and 5.1e-1 (black). The old test left
    the model unseeded, which made it a lottery at any defensible tolerance.
    """
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0, steps=128, baseline=baseline)

    with torch.no_grad():
        reference = _ig_baseline(x, baseline, "cpu")
        delta = (model(x)[0, 0] - model(reference)[0, 0]).item()

    assert attr.sum() == pytest.approx(delta, rel=0.02)


def test_integrated_gradients_device_cpu():
    """IG should work on CPU."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0, device="cpu")
    assert attr.shape == (1, 32, 32)


def test_integrated_gradients_multiple_targets():
    """IG should accept different target classes."""
    model = _tiny_model().eval()
    x = torch.randn(2, 3, 32, 32)
    attr_0 = integrated_gradients(model, x, target=0)
    attr_1 = integrated_gradients(model, x, target=1)
    assert attr_0.shape == attr_1.shape == (2, 32, 32)
    # Different targets should generally produce different attributions
    assert not np.allclose(attr_0, attr_1)


# ============================================================================
# Grad-CAM Tests
# ============================================================================

def test_gradcam_output_shape():
    """Grad-CAM should return (n, 32, 32) upsampled to input resolution."""
    model = resnet18_cifar().eval()
    x = torch.randn(2, 3, 32, 32)
    attr = grad_cam(model, x, target=0)
    assert attr.shape == (2, 32, 32)
    assert attr.dtype == np.float32


def test_gradcam_removes_its_hooks():
    """A leaked hook corrupts every later forward pass in the session.
    This is a critical requirement for Kaggle's 12-hour sessions."""
    model = resnet18_cifar().eval()
    layer = model.layer4[-1]
    before = len(layer._forward_hooks) + len(layer._backward_hooks)
    grad_cam(model, torch.randn(2, 3, 32, 32), target=0, target_layer=layer)
    after = len(layer._forward_hooks) + len(layer._backward_hooks)
    assert after == before, f"Hooks leaked: before={before}, after={after}"


def test_gradcam_removes_hooks_even_when_model_raises():
    """Grad-CAM must clean up hooks even if the model raises an exception."""
    model = resnet18_cifar().eval()
    layer = model.layer4[-1]

    # Save the original forward
    original_forward = model.forward

    # Wrap to raise during forward
    def boom_forward(x):
        # This will execute grad_cam's forward, which hooks the layer,
        # then raise before the backward
        output = original_forward(x)
        raise RuntimeError("boom")

    model.forward = boom_forward

    before = len(layer._forward_hooks) + len(layer._backward_hooks)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            grad_cam(model, torch.randn(1, 3, 32, 32), target=0, target_layer=layer)
    finally:
        model.forward = original_forward

    after = len(layer._forward_hooks) + len(layer._backward_hooks)
    assert after == before, f"Hooks leaked after exception: before={before}, after={after}"


def test_gradcam_default_target_layer():
    """Grad-CAM should auto-detect target layer if not provided."""
    model = resnet18_cifar().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = grad_cam(model, x, target=0)
    assert attr.shape == (1, 32, 32)


def test_gradcam_explicit_target_layer():
    """Grad-CAM should accept an explicit target layer."""
    model = resnet18_cifar().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = grad_cam(model, x, target=0, target_layer=model.layer4[-1])
    assert attr.shape == (1, 32, 32)


def test_gradcam_multiple_targets():
    """Grad-CAM should accept different target classes.

    Seeded, and the maps are checked for content first. On an UNTRAINED resnet18
    the post-ReLU map is identically zero for a large share of weight draws -- 5
    of 12 consecutive seeds here produce an all-zero map for at least one target.
    Two all-zero maps are `allclose`, so this test failed outright whenever both
    draws landed there, and passed for the wrong reason whenever only one did.
    The model was built from whatever RNG state the previous test left and
    pytest-randomly reshuffles the order every run, so which of the three
    happened was decided by the ordering.
    """
    torch.manual_seed(0)
    model = resnet18_cifar().eval()
    x = torch.randn(2, 3, 32, 32)
    attr_0 = grad_cam(model, x, target=0)
    attr_1 = grad_cam(model, x, target=1)
    assert attr_0.shape == attr_1.shape == (2, 32, 32)
    assert np.abs(attr_0).max() > 0 and np.abs(attr_1).max() > 0, (
        "an all-zero map makes the comparison below vacuous rather than passing"
    )
    # Different targets should generally produce different attributions
    assert not np.allclose(attr_0, attr_1)


def test_gradcam_device_cpu():
    """Grad-CAM should work on CPU."""
    model = resnet18_cifar().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = grad_cam(model, x, target=0, device="cpu")
    assert attr.shape == (1, 32, 32)


# ============================================================================
# Random Attribution Tests
# ============================================================================

def test_random_attribution_shape():
    """Random attribution should match the input shape."""
    attr = random_attribution((4, 32, 32), seed=0)
    assert attr.shape == (4, 32, 32)
    assert attr.dtype == np.float32


def test_random_attribution_is_reproducible():
    """Same seed should give identical results."""
    a = random_attribution((4, 32, 32), seed=0)
    b = random_attribution((4, 32, 32), seed=0)
    c = random_attribution((4, 32, 32), seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_attribution_values_in_reasonable_range():
    """Random attributions should be normally distributed with reasonable scale."""
    attr = random_attribution((4, 32, 32), seed=0)
    # Standard normal should have mean ~0 and std ~1
    assert abs(attr.mean()) < 0.2  # Mean close to 0
    assert 0.8 < attr.std() < 1.2  # Std close to 1


def test_random_attribution_seed_required():
    """seed parameter must be explicitly required (no default)."""
    # This is a signature test; the function should not have a default seed
    with pytest.raises(TypeError):
        random_attribution((4, 32, 32))


# ============================================================================
# random_lowres: the geometry control
# ============================================================================

def _peaks(attribution: np.ndarray) -> int:
    """Distinct local maxima in a map: how many separate blobs it has.

    Connected components of the pixels that are the maximum of their 3x3
    neighbourhood and above zero. Components rather than pixels because bilinear
    upsampling produces plateaus, and a plateau is one blob however many pixels
    tie on it.
    """
    top = (attribution == ndimage.maximum_filter(attribution, size=3)) & (attribution > 0)
    return int(ndimage.label(top)[1])


class TestRandomLowresIsGradCamsGeometryWithoutGradCam:
    """The control the study was missing, so its pipeline has to match exactly.

    `random_lowres` exists to answer whether removal faithfulness scores an
    attribution or the shape of the mask it induces. That only works if its maps
    are the same KIND of object as Grad-CAM's: drawn coarse, ReLU'd, then
    bilinearly upsampled to 32x32. Ranking is by |attribution|, so a signed map
    and a ReLU'd one rank differently at identical smoothness -- which is also
    why the existing `random` explainer is not the grid=32 rung of this ladder
    but the pixel-i.i.d. control the protocol subtracts.
    """

    @pytest.mark.parametrize("grid", LOWRES_GRIDS)
    def test_the_map_is_non_negative(self, grid):
        """Post-ReLU, like Grad-CAM. A signed map would put its largest
        |attribution| on the most negative pixels and mask a different set."""
        attribution = random_lowres_attribution((4, 32, 32), grid=grid, seed=0)
        assert attribution.min() >= 0.0
        assert attribution.max() > 0.0

    @pytest.mark.parametrize("grid", LOWRES_GRIDS)
    def test_shape_and_dtype(self, grid):
        attribution = random_lowres_attribution((4, 32, 32), grid=grid, seed=0)
        assert attribution.shape == (4, 32, 32)
        assert attribution.dtype == np.float32

    def test_a_coarser_grid_gives_a_smoother_map(self):
        """The axis the ladder varies, asserted rather than assumed.

        If the grid did not change the blob size the whole ladder would measure
        one point five times and the geometry account would be untestable.
        Distinct local maxima over 8 images at seed 0, measured: grid 2 -> 51,
        4 -> 66, 8 -> 112, 16 -> 295, 32 -> 962. Strictly increasing, so a
        grid-2 map has far fewer separate blobs than a grid-16 one.
        """
        counts = [
            sum(
                _peaks(image)
                for image in random_lowres_attribution((8, 32, 32), grid=grid, seed=0)
            )
            for grid in LOWRES_GRIDS
        ]
        assert counts == sorted(counts) and len(set(counts)) == len(counts), (
            f"blob count is not monotone in the grid: {dict(zip(LOWRES_GRIDS, counts))}"
        )

    def test_it_is_reproducible_by_seed(self):
        a = random_lowres_attribution((4, 32, 32), grid=4, seed=0)
        b = random_lowres_attribution((4, 32, 32), grid=4, seed=0)
        c = random_lowres_attribution((4, 32, 32), grid=4, seed=1)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_the_top_rung_is_the_random_control_relud_not_the_control_itself(self):
        """Why `random` is not a point on this ladder.

        At grid=32 the upsample is the identity, so the arm is exactly the
        pixel-i.i.d. control passed through a ReLU -- which is a different object:
        half the control's values are negative and rank highest under
        |attribution|, where the ReLU'd map ranks them last. Taking grid=32 for
        free from `random` would have put a signed map on a curve of non-negative
        ones.
        """
        control = random_attribution((4, 32, 32), seed=3)
        rung = random_lowres_attribution((4, 32, 32), grid=32, seed=3)

        np.testing.assert_array_equal(rung, np.maximum(control, 0.0))
        assert not np.array_equal(rung, control)
        assert (control < 0).mean() > 0.4, "the control is signed; the rung is not"

    @pytest.mark.parametrize("grid", [1, 3, 0, -4])
    def test_a_grid_off_the_ladder_is_refused(self, grid):
        """1 is the one that matters: a single value ties every pixel, so
        `rank_features`' positional tie-break makes the mask the first n pixels
        in row-major order and the arm measures where sky sits in a CIFAR frame
        rather than blob size."""
        with pytest.raises(ValueError, match="grid"):
            random_lowres_attribution((4, 32, 32), grid=grid, seed=0)


# ============================================================================
# roll: the alignment control
# ============================================================================

class TestRollAttributionHoldsEverythingButAlignment:
    """A rolled map must differ from its original in alignment and nothing else.

    `faith(X) - faith(X_rolled)` is what X's localisation is worth once geometry
    is held fixed. That subtraction only isolates localisation if the roll leaves
    the value multiset and the spatial autocorrelation untouched -- if it changed
    the values, the difference would carry a magnitude effect too and the control
    would be measuring two things at once.
    """

    def test_each_image_keeps_its_exact_value_multiset(self):
        """A translation permutes pixels; it must not resample or interpolate
        them. np.roll wraps, so nothing falls off the edge either."""
        base = random_attribution((8, 32, 32), seed=0)
        rolled = roll_attribution(base, seed=0)

        for i in range(len(base)):
            np.testing.assert_array_equal(
                np.sort(base[i].ravel()), np.sort(rolled[i].ravel())
            )

    def test_no_image_is_left_where_it_was(self):
        """Offsets are drawn from [1, h*w), which excludes the identity. A
        control that was silently the attribution itself for some images would
        report their localisation as worth exactly nothing."""
        base = random_attribution((16, 32, 32), seed=0)
        rolled = roll_attribution(base, seed=0)

        for i in range(len(base)):
            assert not np.array_equal(base[i], rolled[i]), (
                f"image {i} was rolled onto itself, so it is its own control"
            )

    def test_the_offsets_are_per_image_not_one_for_the_batch(self):
        """A single shared offset would make the control a rigid translation of
        the whole cell, so any position-dependent effect in CIFAR would survive
        it intact."""
        base = np.tile(random_attribution((1, 32, 32), seed=0), (8, 1, 1))
        rolled = roll_attribution(base, seed=0)

        distinct = {image.tobytes() for image in rolled}
        assert len(distinct) > 1, "identical inputs were all rolled by one offset"

    def test_it_is_reproducible_by_seed(self):
        base = random_attribution((8, 32, 32), seed=0)
        np.testing.assert_array_equal(
            roll_attribution(base, seed=0), roll_attribution(base, seed=0)
        )
        assert not np.array_equal(
            roll_attribution(base, seed=0), roll_attribution(base, seed=1)
        )

    def test_the_shape_and_dtype_survive(self):
        base = random_attribution((8, 32, 32), seed=0)
        rolled = roll_attribution(base, seed=0)
        assert rolled.shape == base.shape
        assert rolled.dtype == base.dtype


# ============================================================================
# Explainer names
# ============================================================================

VALID_NAMES = (
    [("integrated_gradients", "integrated_gradients", False, None),
     ("grad_cam", "grad_cam", False, None),
     ("random", "random", False, None)]
    + [(f"{LOWRES_PREFIX}{g}", "random_lowres", False, g) for g in LOWRES_GRIDS]
    + [(f"{base}{ROLLED_SUFFIX}", base, True, None) for base in ROLLABLE]
)


class TestExplainerNamesAreParsedInExactlyOnePlace:
    """The grid and the roll are encoded in the NAME, so the parse is the key.

    `explainer_options` builds the cache key from a name and `explain_batch` does
    the work from the same name. If the two read it differently an artifact is
    keyed on settings it was not computed with, which is the defect class this
    whole module has been rewritten to close. One parse is the invariant, so the
    parse is what gets pinned.
    """

    @pytest.mark.parametrize("name,base,rolled,grid", VALID_NAMES)
    def test_a_valid_name_decomposes_and_recomposes(self, name, base, rolled, grid):
        parsed = parse_explainer(name)
        assert parsed == ExplainerName(base=base, rolled=rolled, grid=grid)
        assert parsed.base in BASE_EXPLAINERS

        stem = f"{LOWRES_PREFIX}{parsed.grid}" if parsed.grid else parsed.base
        assert stem + (ROLLED_SUFFIX if parsed.rolled else "") == name, (
            "the axes do not recompose to the name they came from, so the key "
            "and the dispatch can disagree about what was asked for"
        )

    def test_every_base_explainer_is_covered_by_a_valid_name(self):
        """A base reachable by no name would be dead code in the dispatch table
        and an arm the config could not request."""
        assert {base for _, base, _, _ in VALID_NAMES} == set(BASE_EXPLAINERS)

    def test_the_degenerate_grid_is_not_a_name(self):
        """grid=1 ties every pixel, so the mask falls to the positional
        tie-break and the arm measures image position, not geometry."""
        with pytest.raises(ValueError, match="grid"):
            parse_explainer(f"{LOWRES_PREFIX}1")

    def test_a_grid_that_is_not_on_the_ladder_is_not_a_name(self):
        """Refused rather than accepted-and-run: a config asking for a rung the
        ladder does not have would otherwise spend GPU on a point the report has
        no row for."""
        with pytest.raises(ValueError, match="grid"):
            parse_explainer(f"{LOWRES_PREFIX}3")

    @pytest.mark.parametrize("name", ["random_rolled", "random_lowres_4_rolled"])
    def test_rolling_an_already_random_map_is_refused(self, name):
        """Translating a map that carries no localisation gives another draw from
        the same distribution, so the arm would re-measure the one it rolled and
        burn a run slot doing it. Refused rather than allowed-and-pointless."""
        with pytest.raises(ValueError, match="roll"):
            parse_explainer(name)
        assert name.removesuffix(ROLLED_SUFFIX) not in ROLLABLE

    def test_an_unknown_name_is_refused_with_the_available_ones(self):
        with pytest.raises(ValueError, match="unknown explainer"):
            parse_explainer("gradient_shap")


# ============================================================================
# Config -> explainer options
# ============================================================================

class TestTheConfigIsReadInOnePlace:
    """`fill` and the report must map the config to options identically.

    They held separate copies for one commit: the filler decides what a curve is
    computed under and the report decides what key it is looked up by, so a
    one-entry difference makes a full cache read as an empty one -- the same
    two-copies-of-one-fact shape as every cache-key defect in this project.
    """

    def test_each_config_key_maps_to_its_explainer_options_keyword(self):
        """The config spells it `ig_baseline`, not `baseline`, so it cannot be
        confused with the removal `imputation` -- a different axis that happens
        to share the value names mean, black and blur."""
        assert explain_options_from_config({
            "ig_steps": 64, "ig_baseline": "black", "roll_seed": 3,
        }) == {"ig_steps": 64, "baseline": "black", "roll_seed": 3}

    def test_an_option_the_config_does_not_name_is_absent_not_defaulted(self):
        """`explainer_options` omits any option left at its default, so filling
        one in here would fork the key of every artifact already on disk."""
        assert explain_options_from_config({}) == {}
        assert explain_options_from_config({"ig_steps": 32}) == {"ig_steps": 32}

    def test_a_config_key_that_is_not_an_attribution_setting_is_ignored(self):
        """A config carries epochs, severities and imputations too. Splatting it
        would put the removal scheme in the attribution key and fork it per
        imputation, for attributions that do not depend on the imputation."""
        assert explain_options_from_config({
            "epochs": 50, "imputations": ["mean", "black"], "n_eval_images": 1000,
        }) == {}


# ============================================================================
# explain_batch Tests
# ============================================================================

def test_explain_batch_integrated_gradients():
    """explain_batch should dispatch to integrated_gradients."""
    model = _tiny_model().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    attr = explain_batch(model, images, targets, explainer="integrated_gradients")
    assert attr.shape == (2, 32, 32)


def test_explain_batch_grad_cam():
    """explain_batch should dispatch to grad_cam."""
    model = resnet18_cifar().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    attr = explain_batch(model, images, targets, explainer="grad_cam")
    assert attr.shape == (2, 32, 32)


def test_explain_batch_random_attribution():
    """explain_batch should dispatch to random_attribution with deterministic seed."""
    model = _tiny_model().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])
    attr1 = explain_batch(model, images, targets, explainer="random", seed=0)
    attr2 = explain_batch(model, images, targets, explainer="random", seed=0)
    np.testing.assert_array_equal(attr1, attr2)


def test_explain_batch_unknown_explainer_raises_with_list():
    """Unknown explainer should raise with list of valid names."""
    model = _tiny_model().eval()
    images = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([0])
    with pytest.raises(ValueError, match="explainer|integrated_gradients|grad_cam|random"):
        explain_batch(model, images, targets, explainer="unknown")


def test_explain_batch_passes_kwargs():
    """explain_batch should pass keyword arguments to the explainer."""
    model = _tiny_model().eval()
    images = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([0])
    # Pass custom baseline and steps
    attr = explain_batch(
        model, images, targets,
        explainer="integrated_gradients",
        baseline="blur",
        steps=16
    )
    assert attr.shape == (1, 32, 32)


def test_explain_batch_device_propagation():
    """explain_batch should respect device parameter."""
    model = _tiny_model().eval()
    images = torch.randn(1, 3, 32, 32)
    targets = torch.tensor([0])
    attr = explain_batch(model, images, targets, explainer="integrated_gradients", device="cpu")
    assert attr.shape == (1, 32, 32)


# ============================================================================
# Batching
# ============================================================================

class TestBatchingDoesNotChangeResults:
    """The optimisation is only valid if the numbers are the same.

    explain_batch dispatched one image at a time until the explainers learned
    to take a target per image. That made a pilot IG cell take 55 minutes on a
    T4 and put the full grid at roughly 220 GPU-hours for this stage. Batching
    is worth nothing if it quietly changes what is being measured, so these
    compare the batched result against the old one-at-a-time path directly.
    """

    @pytest.mark.parametrize("explainer", ["integrated_gradients", "grad_cam"])
    def test_an_image_gets_the_same_attribution_alone_or_in_a_batch(self, explainer):
        """The property batching has to preserve.

        Not "batch 1 equals batch 4" -- see the note on batch size 1 below.
        What must hold is that an image's attribution does not depend on what
        it was batched with, which is what makes a chunked run equivalent to
        the per-image one it replaced.

        The seed moved ABOVE the model. It used to sit between the model and the
        images, so the weights came from whatever RNG state the previous test
        left: pytest-randomly reshuffles the order every run and does not reseed
        torch, so the model was a different one each time. The difference this
        test measures is kernel selection, which depends on the weights, and
        across model draws it ranges over three orders of magnitude -- 9.7e-7
        relative at this seed, 1.4e-3 at another. An unseeded model made the test
        a lottery, and it is the lottery that failed, not the batching.

        Stated relative to the attribution's own scale rather than against an
        absolute epsilon. The absolute form is what let the lottery run so long:
        at atol=1e-5 the assertion held for any draw whose attributions were
        small enough, so it was measuring magnitude as much as agreement -- and
        magnitude is exactly what the IG baseline axis changes (|x - baseline| is
        2.4x larger under `black` than under `mean`).

        Measured at this seed: 9.7e-7 relative for IG, 1.2e-5 for Grad-CAM, so
        rtol=1e-4 leaves 8x headroom on the worse of the two while staying an
        order of magnitude under the 1e-3 batch-size-1 kernel difference the test
        below pins -- so a real batching bug still fails here.
        """
        torch.manual_seed(0)
        model = resnet18_cifar().eval()
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([3, 1, 4, 1, 5, 9, 2])

        alone = explain_batch(
            model, images[2:3], targets[2:3], explainer=explainer, batch_size=4
        )
        embedded = explain_batch(
            model, images, targets, explainer=explainer, batch_size=7
        )[2:3]

        scale = np.abs(embedded).max()
        assert scale > 0, (
            "the attribution is identically zero, so any agreement is vacuous"
        )
        assert np.abs(embedded - alone).max() / scale < 1e-4

    @pytest.mark.parametrize("explainer", ["integrated_gradients", "grad_cam"])
    def test_the_chunk_size_does_not_change_the_result(self, explainer):
        """Seeded above the model, for the reason given in the test above.

        This one failed 3 of 12 model draws, and the measurement says why: 7
        images at batch_size=2 leaves a trailing chunk of ONE, so `in_twos`
        silently includes the degenerate batch the test below pins. Image 6 --
        the trailing one -- carried the worst difference in every failing draw,
        at 3.1e-4 to 9.4e-4 relative, against 6.9e-7 for the draws that passed.
        """
        torch.manual_seed(0)
        model = resnet18_cifar().eval()
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([3, 1, 4, 1, 5, 9, 2])

        in_twos = explain_batch(
            model, images, targets, explainer=explainer, batch_size=2
        )
        in_one_go = explain_batch(
            model, images, targets, explainer=explainer, batch_size=7
        )

        np.testing.assert_allclose(in_twos, in_one_go, rtol=1e-4, atol=1e-5)

    def test_batch_size_one_is_the_odd_one_out_and_that_is_why_v2_exists(self):
        """Documents the measurement behind the EXPLAIN_VERSION bump.

        Every chunk size from 2 up agrees to around 1e-6 relative. Batch size
        1 does not: torch special-cases the degenerate batch to a different
        convolution algorithm, and IG accumulates that over its steps to a few
        times 1e-3 relative on resnet18. A shallow model shows 1e-10 throughout,
        so this is kernel selection rather than anything about the maths.

        It matters because the superseded code path WAS batch size 1. Keeping
        explain-v1 would have let a resumed run pair those attributions with
        batched ones, differing by tenths of a percent for no recorded reason.

        The model is now seeded, which is a fix and not a tidy-up: it was built
        from whatever RNG state the preceding test left, and both bounds here are
        weight-dependent, so this test failed 4 of 12 model draws. Three of those
        were the lower bound -- 7 images at batch_size=2 ends in a chunk of ONE,
        putting `ig(2)` on the very kernel path this test asserts it avoids, at
        up to 9.4e-4 relative. One was the upper bound: at that draw the batch-1
        difference was only 9.3e-7, so torch had not in fact diverged.

        Measured at this seed: chunk size 2 agrees to 9.0e-7 relative and batch
        size 1 differs by 2.4e-3. Both figures are recorded here because the
        version bump was justified by the second one.
        """
        torch.manual_seed(0)
        model = resnet18_cifar().eval()
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([3, 1, 4, 1, 5, 9, 2])

        def ig(batch_size):
            return explain_batch(
                model, images, targets,
                explainer="integrated_gradients", batch_size=batch_size,
            )

        scale = np.abs(ig(7)).max()
        assert np.abs(ig(2) - ig(7)).max() / scale < 1e-5, "chunk sizes 2+ agree"
        assert np.abs(ig(1) - ig(7)).max() / scale > 1e-4, (
            "batch size 1 no longer differs; if torch stopped special-casing it, "
            "say so here rather than deleting the test -- the version bump was "
            "justified by this difference"
        )

    def test_a_partial_final_chunk_is_handled(self):
        """7 images at batch size 4 leaves a chunk of 3."""
        model = _tiny_model().eval()
        images = torch.randn(7, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        attr = explain_batch(
            model, images, targets, explainer="integrated_gradients", batch_size=4
        )
        assert attr.shape == (7, 32, 32)

    def test_each_image_is_attributed_against_its_own_target(self):
        """The reason the old code could not batch.

        integrated_gradients took one scalar target and applied it to the whole
        batch, so attributing images against different classes meant one call
        per image. A batch that silently used the first image's class for all
        of them would pass a shape check and be wrong everywhere.
        """
        model = _tiny_model().eval()
        images = torch.randn(2, 3, 32, 32)

        together = explain_batch(
            model, images, torch.tensor([0, 7]),
            explainer="integrated_gradients", batch_size=2,
        )
        separately = np.concatenate([
            integrated_gradients(model, images[0:1], target=0),
            integrated_gradients(model, images[1:2], target=7),
        ])

        np.testing.assert_allclose(together, separately, rtol=1e-4, atol=1e-5)

        shared_target = explain_batch(
            model, images, torch.tensor([0, 0]),
            explainer="integrated_gradients", batch_size=2,
        )
        assert not np.allclose(together[1], shared_target[1]), (
            "image 1 attributed identically against class 7 and class 0, so the "
            "per-image target is not reaching the model"
        )


class TestPerImageTargets:
    """A scalar still broadcasts, so single-target call sites keep working."""

    @pytest.mark.parametrize("explainer_fn", [integrated_gradients, grad_cam])
    def test_scalar_target_broadcasts_over_the_batch(self, explainer_fn):
        model = resnet18_cifar().eval()
        images = torch.randn(3, 3, 32, 32)

        scalar = explainer_fn(model, images, target=2)
        vector = explainer_fn(model, images, target=torch.tensor([2, 2, 2]))

        np.testing.assert_allclose(scalar, vector, rtol=1e-4, atol=1e-6)

    def test_mismatched_target_count_is_rejected(self):
        model = _tiny_model().eval()
        images = torch.randn(3, 3, 32, 32)

        with pytest.raises(ValueError, match="2 targets for 3 images"):
            integrated_gradients(model, images, target=torch.tensor([0, 1]))

    def test_explain_batch_rejects_a_mismatched_target_count(self):
        model = _tiny_model().eval()
        images = torch.randn(3, 3, 32, 32)

        with pytest.raises(ValueError, match="must match"):
            explain_batch(
                model, images, torch.tensor([0, 1]),
                explainer="integrated_gradients",
            )

    def test_a_nonsense_batch_size_is_rejected(self):
        model = _tiny_model().eval()
        images = torch.randn(2, 3, 32, 32)

        with pytest.raises(ValueError, match="at least 1"):
            explain_batch(
                model, images, torch.tensor([0, 1]),
                explainer="integrated_gradients", batch_size=0,
            )


# ============================================================================
# to_common_grid Tests
# ============================================================================

def test_to_common_grid_8x8():
    """to_common_grid should pool to 8x8."""
    attr = np.random.randn(2, 32, 32).astype(np.float32)
    grid = to_common_grid(attr, grid=8)
    assert grid.shape == (2, 8, 8)
    assert grid.dtype == np.float32


def test_to_common_grid_preserves_total_mass():
    """to_common_grid should preserve total mass to within float tolerance."""
    attr = np.ones((1, 32, 32), dtype=np.float32)
    grid = to_common_grid(attr, grid=8)
    # 32x32 pooled to 8x8 means each output cell sums 16 input cells
    # With all ones, total mass is 32*32=1024, output mass should be 1024
    assert grid.sum() == pytest.approx(attr.sum(), rel=1e-5)


def test_to_common_grid_custom_size():
    """to_common_grid should accept custom grid size."""
    attr = np.random.randn(1, 32, 32).astype(np.float32)
    grid_4 = to_common_grid(attr, grid=4)
    grid_8 = to_common_grid(attr, grid=8)
    grid_16 = to_common_grid(attr, grid=16)
    assert grid_4.shape == (1, 4, 4)
    assert grid_8.shape == (1, 8, 8)
    assert grid_16.shape == (1, 16, 16)


def test_to_common_grid_batches():
    """to_common_grid should work on batches."""
    attr = np.random.randn(4, 32, 32).astype(np.float32)
    grid = to_common_grid(attr, grid=8)
    assert grid.shape == (4, 8, 8)


# ============================================================================
# ig_completeness_error Tests
# ============================================================================

def test_ig_completeness_error_value():
    """ig_completeness_error should return a small relative error for good IG.

    Seeded for the same reason as the completeness test above: the error is a
    ratio whose denominator F(x) - F(baseline) can land near zero on an unlucky
    model draw, and under the real `black` baseline the worst of 64 draws is
    5.1e-1 against this test's 0.1 bar. At this seed it is 4.1e-3.
    """
    torch.manual_seed(0)
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr = integrated_gradients(model, x, target=0, steps=128, baseline="black")

    error = ig_completeness_error(model, x, target=0, attribution=attr, baseline="black")
    assert isinstance(error, (float, np.floating))
    assert error >= 0
    assert error < 0.1  # Should be small for our simple model


def test_ig_completeness_error_more_steps_lower_error():
    """IG with more steps should have lower completeness error."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)

    attr_16 = integrated_gradients(model, x, target=0, steps=16, baseline="black")
    attr_64 = integrated_gradients(model, x, target=0, steps=64, baseline="black")

    error_16 = ig_completeness_error(model, x, target=0, attribution=attr_16, baseline="black")
    error_64 = ig_completeness_error(model, x, target=0, attribution=attr_64, baseline="black")

    # Both should be stored in the test data
    assert error_16 > 0
    assert error_64 > 0
    # More steps should reduce error (though not guaranteed for every input)


def test_ig_completeness_error_different_baselines():
    """ig_completeness_error should work with different baselines."""
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 32, 32)
    attr_black = integrated_gradients(model, x, target=0, baseline="black")
    attr_blur = integrated_gradients(model, x, target=0, baseline="blur")

    error_black = ig_completeness_error(model, x, target=0, attribution=attr_black, baseline="black")
    error_blur = ig_completeness_error(model, x, target=0, attribution=attr_blur, baseline="blur")

    assert isinstance(error_black, (float, np.floating))
    assert isinstance(error_blur, (float, np.floating))


# ============================================================================
# Version and Constants Tests
# ============================================================================

def test_explain_version_defined():
    """EXPLAIN_VERSION should be defined.

    v3 because `baseline="black"` names a different tensor than it did under v2.
    The version is what keeps the two apart: a cached v2 "black" attribution is a
    mean-baseline attribution filed under the wrong name, and bumping means
    nothing has to be reinterpreted by hand. Aliasing instead of bumping is what
    kept the retired `zero` imputation undetectable.
    """
    assert EXPLAIN_VERSION == "explain-v3"


def test_ig_steps_constant():
    """IG_STEPS should be 32."""
    assert IG_STEPS == 32


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_attribution_pipeline():
    """Full pipeline: model -> explain_batch -> to_common_grid."""
    model = resnet18_cifar().eval()
    images = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])

    # Get attributions at native resolution
    attr_native = explain_batch(model, images, targets, explainer="integrated_gradients", steps=16)
    assert attr_native.shape == (2, 32, 32)

    # Downsample for comparison
    attr_grid = to_common_grid(attr_native, grid=8)
    assert attr_grid.shape == (2, 8, 8)

    # Check mass conservation (to_common_grid rescales to preserve total mass)
    for i in range(2):
        assert attr_native[i].sum() == pytest.approx(attr_grid[i].sum(), rel=1e-4)
