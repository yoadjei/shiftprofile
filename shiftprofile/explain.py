"""Attribution methods: Integrated Gradients, Grad-CAM, and three controls.

All methods return attributions in normalized space with channels summed to
(n, 32, 32) shape. Attribution is per-pixel because removal operates on pixels,
not channels.

The design priorities here are:
  1. Correctness: IG satisfies its mathematical property (completeness)
  2. Reproducibility: every control is exactly reproducible by seed
  3. Session robustness: Grad-CAM leaks no hooks, so Kaggle's 12-hour sessions
     do not slowly poison the model
  4. Cost efficiency: IG_STEPS = 32 is ~35% cheaper than the standard 50

Three controls, not one, because removal-based faithfulness turned out to be
sensitive to things the single pixel-i.i.d. control cannot hold fixed:

  `random`            per-pixel i.i.d. noise. The control `paired_removal_curves`
                      subtracts, and the one the pilot's negative faithfulness
                      was measured against.
  `random_lowres_<k>` noise on a k x k lattice, upsampled exactly as Grad-CAM is.
                      Sees no model, so whatever it scores is blob size alone.
  `<name>_rolled`     the explainer's own map, translated. Same geometry and same
                      values, no alignment to the image it came from.

The pilot's finding is that an attribution's score tracks the geometry of the
mask it induces, and these two make that measurable instead of arguable: the
first holds the model out, the second holds the geometry in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from shiftprofile.data.cifar import normalised_black


# explain-v1 -> v2 when attribution moved from one image at a time to real
# batches. The two paths agree mathematically, but cuDNN picks different kernels
# at different batch sizes, so the float32 results are not guaranteed bitwise
# equal. Rather than let a resumed run mix attributions from both paths, the bump
# recomputed them -- which costs minutes now that it batches.
#
# v2 -> v3 because `baseline="black"` meant something else under v2. It set
# `torch.zeros_like(x)`, which in normalised space is the dataset MEAN, so the
# string "black" named a mean baseline. The name now means what it says, and the
# same spelling therefore identifies two different artifacts across the bump.
# The version is what tells them apart: a result record stores (spec, version),
# so `baseline="black"` under explain-v2 is unambiguously the old mean baseline
# and nothing has to be reinterpreted by hand. Aliasing instead of bumping is
# what made the retired `zero` imputation scheme undetectable for so long.
EXPLAIN_VERSION = "explain-v3"
IG_STEPS = 32

# "mean" is the former "black": zeros in normalised space, which IS the dataset
# mean. "black" is now black in pixel space. "blur" is unchanged. E6 varies this
# axis, so the names have to be true or the ablation contrasts an arm with
# itself -- which is exactly what happened on the imputation axis.
IG_BASELINES = ("mean", "black", "blur")

# The frozen protocol's baseline. Zeros, i.e. the dataset mean: the behaviour
# that has always run, now under a name that describes it.
DEFAULT_IG_BASELINE = "mean"

# Attribution runs one model pass per image per IG step, so the batch size is
# the difference between minutes and hours. Matches curves.py.
EXPLAIN_BATCH_SIZE = 256


def _per_image_targets(target, n: int, device: str) -> torch.Tensor:
    """Normalise a target spec to one class index per image.

    Attribution is taken against each image's own predicted class, so the
    batched path needs a vector. A scalar is broadcast, which keeps the
    single-target call sites and their tests working unchanged.
    """
    if isinstance(target, (int, np.integer)):
        return torch.full((n,), int(target), dtype=torch.long, device=device)

    targets = torch.as_tensor(target, dtype=torch.long, device=device).reshape(-1)
    if targets.shape[0] != n:
        raise ValueError(
            f"got {targets.shape[0]} targets for {n} images. Each image is "
            f"attributed against its own class, so the counts must match. Pass "
            f"a scalar only when every image genuinely shares a target."
        )
    return targets


def _ig_baseline(x: torch.Tensor, baseline: str, device: str) -> torch.Tensor:
    """The tensor IG integrates from. One place, so both callers agree.

    `integrated_gradients` and `ig_completeness_error` each built this
    independently and so each carried the same misnaming of `black`. The
    completeness check comparing an attribution against a DIFFERENT baseline
    from the one it was computed with would report a spurious error, so the two
    cannot be allowed to drift.
    """
    if baseline not in IG_BASELINES:
        raise ValueError(
            f"baseline must be one of {IG_BASELINES}, got {baseline!r}. Note "
            f"that 'black' means black in PIXEL space (about -1.99 per channel "
            f"once normalised); the all-zeros baseline is 'mean', because "
            f"normalisation maps the dataset mean to zero."
        )

    if baseline == "mean":
        # Zeros in normalised space. This is the dataset mean image, which is
        # what this baseline has always been; only the name has changed.
        return torch.zeros_like(x)

    if baseline == "black":
        return (
            torch.tensor(normalised_black(), dtype=x.dtype, device=device)
            .view(1, -1, 1, 1)
            .expand_as(x)
            .contiguous()
        )

    # Gaussian blur in normalized space
    x_np = x.cpu().numpy()
    x_baseline_np = np.zeros_like(x_np)
    for i in range(x.shape[0]):
        for c in range(3):
            x_baseline_np[i, c] = ndimage.gaussian_filter(x_np[i, c], sigma=1.5)
    return torch.from_numpy(x_baseline_np).to(device)


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target,
    *,
    steps: int = IG_STEPS,
    baseline: str = DEFAULT_IG_BASELINE,
    device: str = "cpu",
) -> np.ndarray:
    """Integrated Gradients attribution method.

    Args:
        model: A PyTorch model in eval mode.
        x: Input images of shape (n, 3, 32, 32) in normalized space.
        target: Class index per image, or one scalar shared by the whole batch.
        steps: Number of IG integration steps (default: 32).
        baseline: "mean" (zeros, i.e. the dataset mean), "black" (black in pixel
            space) or "blur" (Gaussian-blurred input).
        device: Device to run on ("cpu" or "cuda").

    Returns:
        Attribution array of shape (n, 32, 32) with channels summed, dtype float32.

    Raises:
        ValueError: If baseline is not one of IG_BASELINES.

    Note:
        Images in a batch are attributed independently, which holds because
        `model.eval()` puts BatchNorm on its running statistics. In train mode
        the batch statistics would couple the images and each attribution would
        depend on what it happened to be batched with.

        The baseline is not cosmetic: attribution is `grad * (x - baseline)`, so
        a pixel sitting at the baseline value scores zero however much the model
        depends on it. Under the "mean" baseline that is every pixel near the
        dataset mean, and "mean" removal imputation then moves a removed pixel to
        exactly that baseline -- so the choice interacts with the metric that
        consumes the attribution, which is why it is an E6 ablation axis.
    """
    x = x.to(device).detach()
    n = x.shape[0]
    x_baseline = _ig_baseline(x, baseline, device)

    model.eval()
    targets = _per_image_targets(target, n, device)
    attributions = np.zeros((n, 32, 32), dtype=np.float32)

    for step in range(steps):
        # Interpolate between baseline and input
        alpha = step / steps
        x_interp = x_baseline + alpha * (x - x_baseline)
        x_interp.requires_grad = True

        # Forward pass
        with torch.enable_grad():
            out = model(x_interp)
            # gather, not out[:, target]: each image scores against its own
            # class. Summing is safe because eval-mode BatchNorm keeps the
            # images independent, so d(sum)/d(x_i) is image i's own gradient.
            target_out = out.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = target_out.sum()

        # Backward pass
        loss.backward()

        # Accumulate gradients (scaled by input difference)
        if x_interp.grad is not None:
            grad = x_interp.grad.detach()
            input_diff = x - x_baseline  # (n, 3, 32, 32)
            scaled_grad = grad * input_diff  # (n, 3, 32, 32)
            # Sum channels and accumulate
            attributions += scaled_grad.sum(dim=1).cpu().numpy() / steps

    return attributions


def grad_cam(
    model: nn.Module,
    x: torch.Tensor,
    target,
    *,
    target_layer: Optional[nn.Module] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Grad-CAM attribution method.

    Args:
        model: A PyTorch model in eval mode.
        x: Input images of shape (n, 3, 32, 32).
        target: Class index per image, or one scalar shared by the whole batch.
        target_layer: Layer to hook (default: auto-detect from model).
        device: Device to run on ("cpu" or "cuda").

    Returns:
        Attribution array of shape (n, 32, 32), dtype float32.

    Raises:
        ValueError: If target_layer cannot be auto-detected.

    Note:
        All hooks are removed in a finally block to prevent session pollution.
        As with IG, batched images are independent only because `model.eval()`
        puts BatchNorm on its running statistics.
    """
    from shiftprofile.models.vision import gradcam_target_layer

    x = x.to(device).detach()
    n = x.shape[0]

    if target_layer is None:
        target_layer = gradcam_target_layer(model)

    model.eval()
    targets = _per_image_targets(target, n, device)
    attributions = np.zeros((n, 32, 32), dtype=np.float32)

    # Storage for activations and gradients
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        x_input = x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = model(x_input)
            target_out = out.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = target_out.sum()

        loss.backward()

        # Compute Grad-CAM
        # gradients: (n, c, h, w)
        # activations: (n, c, h, w)
        if gradients is not None and activations is not None:
            grad_pooled = gradients.mean(dim=(2, 3), keepdim=True)  # (n, c, 1, 1)
            cam = (activations * grad_pooled).sum(dim=1)  # (n, h, w)
            cam = F.relu(cam)  # ReLU

            # Upsample to 32x32
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(32, 32),
                mode="bilinear",
                align_corners=False
            )  # (n, 1, 32, 32)
            attributions = cam.squeeze(1).cpu().numpy().astype(np.float32)

    finally:
        # CRITICAL: Remove hooks in finally block
        forward_handle.remove()
        backward_handle.remove()

    return attributions


def random_attribution(shape: tuple, *, seed: int) -> np.ndarray:
    """Random attribution baseline for faithfulness comparison.

    Args:
        shape: Shape of the output (n, 32, 32).
        seed: Random seed (required, no default).

    Returns:
        Random attribution array, dtype float32.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


# The grids `random_lowres` may be drawn on. 4 is ResNet-18's final feature map
# on a 32x32 input, which is the resolution Grad-CAM actually has; the rest
# bracket it so the effect of blob size can be read as a curve rather than
# argued from one point.
#
# 1 is excluded deliberately. A single value makes every pixel tie, so the mask
# becomes the first n pixels in row-major order and the arm would measure image
# position (sky at the top of a CIFAR frame) rather than mask geometry.
LOWRES_GRIDS = (2, 4, 8, 16, 32)


def random_lowres_attribution(shape: tuple, *, grid: int, seed: int) -> np.ndarray:
    """A random attribution with a chosen spatial blob size, and no model input.

    This is the control the study was missing. `paired_removal_curves` always
    draws its control per-pixel i.i.d., so a method whose map is coarse is scored
    against a mask that is scattered, and the two differ in geometry before they
    differ in content: replacing one contiguous patch leaves a CIFAR image far
    more recognisable than replacing the same NUMBER of scattered pixels does.
    Any faithfulness this function registers is therefore attributable to blob
    size alone, because it never sees the model, the image or the label.

    Draws on a `grid` x `grid` lattice, applies ReLU, then bilinearly upsamples
    to 32x32 -- `grad_cam`'s exact pipeline, including the ReLU, so that the
    post-ReLU zeros and the interpolation ramps between them are the same shape
    of object. Matching all of it matters: ranking is by |attribution|, and a
    map that is non-negative with flat zero regions ranks differently from a
    signed one even at identical smoothness.

    Args:
        shape: Shape of the output (n, 32, 32).
        grid: Lattice size, one of LOWRES_GRIDS.
        seed: Random seed (required, no default).

    Returns:
        Random low-resolution attribution array, dtype float32.

    Raises:
        ValueError: If grid is not in LOWRES_GRIDS.
    """
    if grid not in LOWRES_GRIDS:
        raise ValueError(
            f"grid must be one of {LOWRES_GRIDS}, got {grid}. grid=1 is excluded "
            f"because a single value ties every pixel, leaving the mask to the "
            f"positional tie-break."
        )

    n, h, w = shape
    rng = np.random.RandomState(seed)
    coarse = torch.from_numpy(rng.randn(n, 1, grid, grid).astype(np.float32))
    coarse = F.relu(coarse)
    up = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    return up.squeeze(1).numpy().astype(np.float32)


def roll_attribution(attribution: np.ndarray, *, seed: int) -> np.ndarray:
    """Translate each attribution map by a seeded random offset, wrapping around.

    The per-explainer counterpart to `random_lowres_attribution`. A rolled map
    keeps the original's exact value multiset and exact spatial autocorrelation,
    so the mask it induces has the same geometry down to the pixel -- but it no
    longer sits over the part of the image it was computed from. The difference
    between an explainer's faithfulness and its rolled version is therefore what
    the explainer's LOCALISATION is worth once geometry is held fixed, which is
    the quantity a pixel-i.i.d. control cannot isolate.

    Offsets are drawn per image from the h*w - 1 non-identity translations, so
    the control is never silently the attribution itself.

    Args:
        attribution: Array of shape (n, h, w).
        seed: Random seed (required, no default).

    Returns:
        Rolled attribution array, same shape and dtype as the input.
    """
    n, h, w = attribution.shape
    rng = np.random.RandomState(seed)
    offsets = rng.randint(1, h * w, size=n)
    rolled = np.empty_like(attribution)
    for i, offset in enumerate(offsets):
        dy, dx = divmod(int(offset), w)
        rolled[i] = np.roll(attribution[i], shift=(dy, dx), axis=(0, 1))
    return rolled


LOWRES_PREFIX = "random_lowres_"
ROLLED_SUFFIX = "_rolled"

BASE_EXPLAINERS = ("integrated_gradients", "grad_cam", "random", "random_lowres")

# Rolling a map that carries no localisation changes nothing about it in
# distribution, so a rolled random arm would burn a run slot to re-measure the
# arm it was rolled from. Refused rather than allowed-and-pointless.
ROLLABLE = ("integrated_gradients", "grad_cam")


@dataclass(frozen=True)
class ExplainerName:
    """An explainer name decomposed into the axes encoded in it.

    The grid and the roll live in the NAME rather than in separate config keys so
    that `explainers:` stays the single list a config varies and `fill` needs no
    new axis to fan out over. The parse is shared by `explainer_options`, which
    builds the cache key, and `explain_batch`, which does the work -- the two
    reading one parse is the invariant that keeps an artifact from being keyed on
    settings it was not computed with.
    """

    base: str
    rolled: bool
    grid: Optional[int] = None


def parse_explainer(name: str) -> ExplainerName:
    """Decompose an explainer name. The only place the naming scheme is read.

    Accepts the four base names, `random_lowres_<grid>` for a grid in
    LOWRES_GRIDS, and a `_rolled` suffix on the explainers for which rolling
    means something.

    Raises:
        ValueError: If the name is not a recognised explainer.
    """
    rolled = name.endswith(ROLLED_SUFFIX)
    stem = name[: -len(ROLLED_SUFFIX)] if rolled else name

    grid = None
    if stem.startswith(LOWRES_PREFIX):
        suffix = stem[len(LOWRES_PREFIX):]
        if not suffix.isdigit():
            raise ValueError(
                f"unknown explainer {name!r}: expected {LOWRES_PREFIX}<grid> with "
                f"a grid in {LOWRES_GRIDS}, got suffix {suffix!r}"
            )
        grid = int(suffix)
        if grid not in LOWRES_GRIDS:
            raise ValueError(
                f"unknown explainer {name!r}: grid must be one of {LOWRES_GRIDS}"
            )
        base = "random_lowres"
    else:
        base = stem

    if base not in BASE_EXPLAINERS:
        raise ValueError(
            f"unknown explainer {name!r}. Available: "
            f"{', '.join(sorted(BASE_EXPLAINERS))}, "
            f"{LOWRES_PREFIX}<{'|'.join(str(g) for g in LOWRES_GRIDS)}>, and a "
            f"{ROLLED_SUFFIX!r} suffix on {', '.join(ROLLABLE)}"
        )

    if base == "random_lowres" and grid is None:
        raise ValueError(
            f"{name!r} needs its grid: spell it {LOWRES_PREFIX}<grid> with a grid "
            f"in {LOWRES_GRIDS}. The bare base name is an internal dispatch label "
            f"rather than a runnable arm, and accepting it built a cache key "
            f"carrying grid=None, deferring the failure to the attribution call -- "
            f"after the key for an artifact that cannot exist had been handed out."
        )

    if rolled and base not in ROLLABLE:
        raise ValueError(
            f"{name!r} is not available: rolling {base!r} produces another draw "
            f"from the same distribution, so the arm would re-measure the one it "
            f"was rolled from. Rollable: {', '.join(ROLLABLE)}."
        )

    return ExplainerName(base=base, rolled=rolled, grid=grid)


def explain_batch(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    explainer: str,
    *,
    device: str = "cpu",
    batch_size: int = EXPLAIN_BATCH_SIZE,
    **kw,
) -> np.ndarray:
    """Batch attribution dispatcher.

    Args:
        model: A PyTorch model in eval mode.
        images: Batch of images (n, 3, 32, 32).
        targets: Target class indices (n,).
        explainer: Any name `parse_explainer` accepts: a base explainer,
            `random_lowres_<grid>`, or either of those with a `_rolled` suffix.
        device: Device to run on.
        batch_size: Images per model pass (default: 256).
        **kw: Additional kwargs passed to the explainer.

    Returns:
        Attribution array of shape (n, 32, 32), dtype float32.

    Raises:
        ValueError: If explainer is unknown.

    Note:
        This dispatched one image at a time until the explainers learned to
        take a target per image. On a T4 that left the GPU almost entirely
        idle: IG at 32 steps over 1000 images is 32,000 model passes, and at
        batch size 1 the cost is kernel-launch overhead rather than compute.
        One pilot cell took about 55 minutes, which put the full grid at
        roughly 220 GPU-hours for this stage alone.
    """
    parsed = parse_explainer(explainer)

    # Each explainer takes a DIFFERENT set of options, so kwargs are validated
    # per explainer rather than splatted into whichever function is dispatched.
    # Splatting meant a caller supplying options for all three explainers — the
    # natural thing to do from a config — crashed integrated_gradients with an
    # unexpected 'seed'. Silently dropping unknown keys would be worse: a
    # mistyped 'step' would leave IG on its default and nothing would say so.
    allowed = {
        "integrated_gradients": {"steps", "baseline"},
        "grad_cam": {"target_layer"},
        "random": {"seed"},
        "random_lowres": {"grid", "seed"},
    }[parsed.base]
    if parsed.rolled:
        allowed = allowed | {"roll_seed"}
    unknown = set(kw) - allowed
    if unknown:
        raise TypeError(
            f"explainer {explainer!r} does not accept {sorted(unknown)}; "
            f"it accepts {sorted(allowed)}. Pass only the options this explainer "
            "understands rather than the union for all of them."
        )

    n = images.shape[0]
    options = dict(kw)
    roll_seed = options.pop("roll_seed", None)
    if parsed.rolled and roll_seed is None:
        raise ValueError(
            f"{explainer!r} requires an explicit 'roll_seed': the offsets are the "
            f"control the explainer is compared against, so they must be exactly "
            f"reproducible."
        )

    if parsed.base in ("random", "random_lowres"):
        if "seed" not in options:
            raise ValueError(
                f"the {parsed.base} explainer requires an explicit 'seed': it is "
                f"a control every faithfulness number is reported against, so it "
                f"must be exactly reproducible."
            )
        if parsed.base == "random":
            return random_attribution((n, 32, 32), seed=options["seed"])
        return random_lowres_attribution(
            (n, 32, 32), grid=options["grid"], seed=options["seed"]
        )

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    # Both remaining explainers take a target per image, so a chunk is one
    # model pass rather than one per image.
    fn = {
        "integrated_gradients": integrated_gradients,
        "grad_cam": grad_cam,
    }[parsed.base]
    targets = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
    if targets.shape[0] != n:
        raise ValueError(
            f"got {targets.shape[0]} targets for {n} images; they must match."
        )

    chunks = [
        fn(
            model,
            images[start:start + batch_size],
            target=targets[start:start + batch_size],
            device=device,
            **options,
        )
        for start in range(0, n, batch_size)
    ]
    attributions = np.concatenate(chunks, axis=0)

    # Rolled AFTER concatenation, not per chunk, so the offset an image gets
    # depends on its position in the cell and not on the batch size it happened
    # to be computed under. Otherwise a resumed run at a different batch size
    # would produce a different control from identical inputs.
    if parsed.rolled:
        attributions = roll_attribution(attributions, seed=roll_seed)

    return attributions


def to_common_grid(attribution: np.ndarray, grid: int = 8) -> np.ndarray:
    """Average-pool attribution to a coarser grid for cross-explainer comparison.

    This function is for visualization and aggregation only. Faithfulness metrics
    are ALWAYS computed at native (32, 32) resolution.

    Args:
        attribution: Attribution array of shape (n, 32, 32).
        grid: Output grid size (default: 8 for 8x8).

    Returns:
        Pooled attribution of shape (n, grid, grid), dtype float32, with total mass preserved.
    """
    n, h, w = attribution.shape
    assert h == w == 32, f"expected 32x32, got {h}x{w}"

    # Reshape and pool
    pool_size = h // grid
    pooled = attribution.reshape(n, grid, pool_size, grid, pool_size)
    pooled = pooled.mean(axis=(2, 4))  # Average pool
    # Rescale to preserve total mass
    pooled = pooled * (pool_size ** 2)

    return pooled.astype(np.float32)


def explainer_options(
    explainer: str,
    *,
    ig_steps: int = IG_STEPS,
    baseline: str = DEFAULT_IG_BASELINE,
    random_seed: int = 0,
    roll_seed: int = 0,
) -> dict:
    """The options this explainer actually consumes, and nothing else.

    One source for two callers that must agree: `explain_spec` puts this in the
    cache key, and `explain_cell` passes it to `explain_batch`. If they disagreed,
    an attribution would be keyed on options other than the ones it was computed
    with -- the defect this whole module's keying was rewritten to close.

    Only the options the dispatched explainer reads are returned. Grad-CAM reads
    none, so changing `ig_steps` must not invalidate a Grad-CAM artifact; keying
    every explainer on every option would fork the cache for no reason and cost
    quota this project does not have. `roll_seed` follows the same rule and is
    omitted unless the name actually asks for a roll.

    `baseline` matters beyond correctness: E6 varies the IG baseline, so it has to
    be part of the key or the ablation's attributions would collide with the
    frozen protocol's and silently serve the wrong ones.

    `grid` comes from the name rather than from an argument, so a config naming
    `random_lowres_4` cannot be run at a grid of 8 by a stale keyword.
    """
    parsed = parse_explainer(explainer)
    options = {
        "integrated_gradients": {"steps": ig_steps, "baseline": baseline},
        "grad_cam": {},
        "random": {"seed": random_seed},
        "random_lowres": {"grid": parsed.grid, "seed": random_seed},
    }[parsed.base]
    if parsed.rolled:
        options = {**options, "roll_seed": roll_seed}
    return options


# Config key -> `explainer_options` keyword. The config spells the IG baseline
# `ig_baseline` so it cannot be confused with the removal `imputation`, which is
# a different axis that happens to share the value names "mean", "black", "blur".
CONFIG_OPTION_KEYS = (
    ("ig_steps", "ig_steps"),
    ("ig_baseline", "baseline"),
    ("roll_seed", "roll_seed"),
)


def explain_options_from_config(config: dict) -> dict:
    """The attribution settings a config asks for, in `explainer_options` terms.

    Both the filler and the report need this, and they must agree exactly: the
    filler decides what a curve is computed under and the report decides what key
    it is looked up by, so a difference of one entry means a full cache reads as
    an empty one. They each held their own copy of this mapping for one commit,
    which is the same two-copies-of-one-fact shape as every cache-key defect this
    module has been rewritten to close.

    An option is included only when the config names it, so a config silent about
    an option produces the key it always produced.
    """
    return {name: config[key] for key, name in CONFIG_OPTION_KEYS if key in config}


def explain_spec(
    cell,
    *,
    explainer: str,
    indices: np.ndarray,
    ig_steps: int = IG_STEPS,
    baseline: str = DEFAULT_IG_BASELINE,
    random_seed: int = 0,
    roll_seed: int = 0,
) -> dict:
    """The cache key for this cell's attributions. The only place it is built.

    `inputs` names the predict version actually read, because attributions are
    taken against the model's PREDICTED class. Without that, bumping the predict
    version would leave attributions keyed as though nothing changed, pointing at
    target classes that no longer exist on disk.

    `options` carries the explainer's own settings, so that varying the IG
    baseline or step count -- which E6 does -- cannot collide with the frozen
    protocol's attributions.
    """
    from .cells import stage_spec
    from .data import eval_digest
    from .predict import PREDICT_VERSION

    return stage_spec(
        cell,
        "explain",
        explainer=explainer,
        options=explainer_options(
            explainer,
            ig_steps=ig_steps,
            baseline=baseline,
            random_seed=random_seed,
            roll_seed=roll_seed,
        ),
        evaluated=eval_digest(indices),
        inputs={"predict": PREDICT_VERSION},
    )


def explain_is_cached(
    cell, cache, *, explainer: str, indices: np.ndarray, **options
) -> bool:
    """Whether this cell's attributions for this explainer are already on disk.

    Takes the options as loose keywords, the same shape `explain_cell` takes them
    in, because `fill` hands this predicate and that producer the same arguments.
    A predicate whose signature differed from its producer's would be asked about
    one artifact while the producer computed another -- which is the whole class
    of defect this module's keying was rewritten to close.
    """
    return cache.has(
        explain_spec(cell, explainer=explainer, indices=indices, **options),
        EXPLAIN_VERSION,
        kind="array",
    )


def explain_cell(
    cell,  # Cell object
    model: nn.Module,
    clean_root: str,
    corrupt_root: str,
    cache,  # ArtifactCache
    *,
    explainer: str,
    indices: np.ndarray,
    device: str = "cpu",
    ig_steps: int = IG_STEPS,
    baseline: str = DEFAULT_IG_BASELINE,
    random_seed: int = 0,
    roll_seed: int = 0,
) -> np.ndarray:
    """Compute attributions for a cell, checking cache first.

    Attributions are cached by (cell, explainer). The attribution target is
    the model's PREDICTED class, not the true label. Predictions must be cached
    in the predict stage first; raises if absent.

    Args:
        cell: A Cell object.
        model: A trained model.
        clean_root: Root directory for clean CIFAR-10 data.
        corrupt_root: Root directory for CIFAR-10-C data.
        cache: ArtifactCache for storing/loading attributions.
        explainer: Name of explainer ("integrated_gradients", "grad_cam", "random").
        indices: REQUIRED. The images to attribute, from
                 `fixed_eval_indices(n_eval_images)`. Part of the cache key.
        device: Device to run on.
        ig_steps: Number of IG integration steps.
        baseline: Baseline for IG, one of IG_BASELINES.
        roll_seed: Seed for the `_rolled` offsets, where the name asks for them.
        random_seed: Seed for random attribution reproducibility.

    Returns:
        Attribution array of shape (len(indices), 32, 32) with dtype float32.

    Raises:
        ValueError: If predictions are not cached.
    """
    from .data import load_cell_images, to_normalised_tensor
    from .predict import PREDICT_VERSION, predict_spec as predict_key

    settings = dict(
        ig_steps=ig_steps,
        baseline=baseline,
        random_seed=random_seed,
        roll_seed=roll_seed,
    )
    options = explainer_options(explainer, **settings)
    spec = explain_spec(
        cell, explainer=explainer, indices=indices, **settings
    )

    # Check cache first
    if cache.has(spec, EXPLAIN_VERSION, kind="array"):
        return cache.get_array(spec, EXPLAIN_VERSION)

    # Cache miss: load images
    images, labels = load_cell_images(
        cell,
        clean_root,
        corrupt_root,
        indices=indices,
    )

    # Read the predictions through predict's own key builder. Writing the key out
    # here again is how a consumer and a producer come to disagree, and the same
    # index set must select the same images on both sides.
    upstream = predict_key(cell, indices=indices)
    if not cache.has(upstream, PREDICT_VERSION, kind="array"):
        raise ValueError(
            f"explain_cell requires cached predictions for cell {cell} over the "
            f"same {len(indices)} evaluation images. Run the predict stage first "
            f"with the same n_eval_images."
        )

    logits = cache.get_array(upstream, PREDICT_VERSION)
    predicted_classes = np.argmax(logits, axis=1)  # (n,)

    # Convert images to tensor
    tensor = to_normalised_tensor(images)
    tensor = tensor.to(device)

    # The same options that went into the cache key above, from the same
    # function, so the artifact cannot be keyed on settings it was not computed
    # with. explain_batch rejects options an explainer does not understand rather
    # than dropping them silently.
    attributions = explain_batch(
        model,
        tensor,
        torch.from_numpy(predicted_classes).to(device),
        explainer,
        device=device,
        **options,
    )

    # Attributions are the bulk of the cache (~1 GB at Version A), so they are
    # stored as float16.
    #
    # Return the ROUND-TRIPPED value, not the float32 original. Otherwise a
    # cache miss returns full precision while a cache hit returns the float16
    # version, so an uninterrupted run and a preempted-then-resumed run produce
    # different numbers from identical inputs — in a study whose whole design
    # rests on sessions being interchangeable.
    attributions_fp16 = attributions.astype(np.float16)
    cache.put_array(spec, EXPLAIN_VERSION, attributions_fp16)
    return attributions_fp16.astype(np.float32)


def ig_completeness_error(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    attribution: np.ndarray,
    baseline: str = DEFAULT_IG_BASELINE,
    device: str = "cpu",
) -> float:
    """Relative completeness error of an IG approximation.

    The completeness property says: sum(attribution) ≈ f(x) - f(baseline).
    This function computes the relative error.

    Args:
        model: A PyTorch model in eval mode.
        x: Input images (n, 3, 32, 32) in normalized space.
        target: Target class index.
        attribution: IG attribution array of shape (n, 32, 32).
        baseline: One of IG_BASELINES. Must be the baseline the attribution was
            computed with, or the error is measured against the wrong reference.
        device: Device to run on.

    Returns:
        Relative error as a float in [0, 1].
    """
    x = x.to(device)
    x_baseline = _ig_baseline(x, baseline, device)

    model.eval()
    with torch.no_grad():
        out_x = model(x)[:, target]
        out_baseline = model(x_baseline)[:, target]

    delta = out_x - out_baseline  # (n,)
    attr_sum = attribution.sum(axis=(1, 2))  # (n,)

    # Relative error: |attr_sum - delta| / |delta|
    # Avoid division by zero
    delta_cpu = delta.cpu().numpy()
    abs_delta = np.abs(delta_cpu)
    abs_delta = np.maximum(abs_delta, 1e-8)

    error = np.abs(attr_sum - delta_cpu) / abs_delta
    return float(error.mean())
