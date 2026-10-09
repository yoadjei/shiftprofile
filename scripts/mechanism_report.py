"""Is removal-based faithfulness still contaminated by mask geometry?

Written to settle a different question, and kept because the question it settles
now is the one that matters. The first validity run reported that translating
Grad-CAM's own map to an arbitrary location IMPROVED its faithfulness by 0.13 at
10 half-widths, and that a model-free blob of the same size scored better than
Grad-CAM. That was not a property of attributions. `_get_removal_mask` was
removing the pixels of LOWEST magnitude, so every curve was the removal curve
for deleting the background, and a map whose low region is a smooth contiguous
backdrop -- which is exactly Grad-CAM's -- scored worst of all. Rolling the map
decorrelated "lowest attribution" from "background" and the score moved toward
zero. See the curves-v3 note in `shiftprofile/curves.py`.

So the readings that run seemed to be choosing between are void, and these three
tests are repointed at what is actually open once the mask selects the right
pixels. They need no GPU and no new cells beyond a corrected fill: everything is
on disk, or recoverable from a seed.

TEST 1 -- displacement.  `roll_attribution` draws a per-image offset from
`RandomState(roll_seed).randint(1, h*w)`, which depends on nothing else, so
every image's displacement is recoverable exactly without reading the artifact.
The rolled arm is therefore already a position experiment that nobody had to
run. A real attribution should BEAT its own displaced copy, and by more as the
copy lands further from where the evidence is: a gain that stays flat in
displacement, or stays negative, says the metric is still not reading content.

TEST 2 -- geometry alone.  Per arm, two numbers computed from the cached
attributions: how far the top-5% mask's centroid sits from the image centre, and
how spread out that mask is. The `random_lowres` ladder varies the second on
purpose while holding attribution content at nothing, so it measures how much of
any arm's score is available from blob size alone. If real explainers and
model-free blobs still fall on one line through these two numbers, the metric
reads geometry and content adds nothing -- which was the P3b hypothesis and is
still untested, because the run that was meant to test it ran inverted.

TEST 3 -- where in the curve.  The per-image curves hold nine removal fractions.
Reports which fractions carry the gap between an explainer and its rolled
control, so a difference concentrated at one end of the removal range is not
reported as a property of the whole curve.

Invocation mirrors `scripts/pilot_report.py`.

    !python /kaggle/working/shiftprofile/scripts/mechanism_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from shiftprofile.cache import ArtifactCache  # noqa: E402
from shiftprofile.cells import enumerate_cells  # noqa: E402
from shiftprofile.curves import (  # noqa: E402
    CURVES_VERSION,
    REMOVAL_FRACTIONS,
    curves_spec,
)
from shiftprofile.data import fixed_eval_indices  # noqa: E402
from shiftprofile.explain import (  # noqa: E402
    EXPLAIN_VERSION,
    LOWRES_PREFIX,
    ROLLED_SUFFIX,
    explain_options_from_config,
    explain_spec,
)
from shiftprofile.metrics.faithfulness import relative_faithfulness  # noqa: E402
from shiftprofile.stats.bootstrap import bootstrap_interval  # noqa: E402

# The fraction of pixels whose mask geometry is measured. 0.05 is the first
# non-zero point on the removal curve, so it is the mask the metric reaches
# first and the one a reader pictures when they picture "the top 5%".
MASK_FRACTION = 0.05

# CIFAR images are 32x32, so the centre of the frame is (15.5, 15.5) and the
# farthest a centroid can sit from it is a corner, about 21.9 px.
IMAGE_CENTRE = 15.5


class MissingArtifacts(RuntimeError):
    """Named so the caller can tell "not run yet" from "computed wrong"."""


def roll_displacements(n: int, *, seed: int, height: int = 32, width: int = 32):
    """The per-image displacement `roll_attribution` applied, recovered.

    Reproduces that function's draw rather than reading it back from anywhere,
    because it depends only on the seed and the image count. Returns the
    distance on a torus: `np.roll` wraps, so a shift of 30 on a 32-wide axis is
    a displacement of 2, and treating it as 30 would put most images in the
    wrong bin.

    MUST stay in step with `explain.roll_attribution`. A test pins the two
    against each other; this is the sort of second copy of one fact that every
    cache-key defect in this project turned out to be.
    """
    offsets = np.random.RandomState(seed).randint(1, height * width, size=n)
    dy, dx = np.divmod(offsets, width)
    return np.hypot(np.minimum(dy, height - dy), np.minimum(dx, width - dx))


def mask_geometry(attributions: np.ndarray, *, fraction: float = MASK_FRACTION):
    """Where the top-`fraction` mask sits, and how spread out it is.

    Returns (centre_distance, spread), each of shape (n,):

    `centre_distance` is the mask centroid's distance from the middle of the
    frame. CIFAR-10 objects are roughly centred, so this is a usable proxy for
    "is the mask on the object" without bounding boxes the dataset does not have.

    `spread` is the mean distance of masked pixels from their own centroid, which
    is low for one contiguous blob and high for a scattered mask. It is the
    geometry axis the `random_lowres` ladder varies by construction.

    Ranking is by |attribution|, matching `_get_removal_mask`, so the mask
    measured here is the mask the curve was actually computed from.
    """
    n = attributions.shape[0]
    flat = np.abs(attributions.reshape(n, -1).astype(np.float64))
    n_pixels = flat.shape[1]
    side = int(round(np.sqrt(n_pixels)))
    k = int(np.ceil(fraction * n_pixels))

    # argpartition, not argsort: only the top k matter and n is 1000 per cell.
    top = np.argpartition(-flat, k - 1, axis=1)[:, :k]
    ys, xs = np.divmod(top, side)

    cy = ys.mean(axis=1)
    cx = xs.mean(axis=1)
    centre_distance = np.hypot(cy - IMAGE_CENTRE, cx - IMAGE_CENTRE)
    spread = np.hypot(ys - cy[:, None], xs - cx[:, None]).mean(axis=1)
    return centre_distance, spread


def per_image_faithfulness(cell, cache, *, explainer, imputation, indices,
                           explain_options):
    """One `relative_faithfulness` per image, read through curves' own key."""
    fractions = np.asarray(REMOVAL_FRACTIONS, dtype=np.float64)

    def curve(which):
        spec = curves_spec(
            cell, explainer=explainer, imputation=imputation, which=which,
            indices=indices, explain_options=explain_options,
        )
        try:
            return cache.get_array(spec, CURVES_VERSION)
        except KeyError:
            raise MissingArtifacts(
                f"no {which} curve for {cell} / {explainer} / {imputation}. "
                f"Run the curves stage for configs/pilot_validity.yaml first."
            )

    model_curve, random_curve = curve("model"), curve("random")
    return np.array([
        relative_faithfulness(m, r, fractions, imputation=imputation)
        for m, r in zip(model_curve, random_curve)
    ]), model_curve, random_curve


def displacement_test(cells, cache, *, pairs, imputations, indices,
                      explain_options, roll_seed, n_bins=5, n_resamples=2000):
    """TEST 1: does rolling further help more?

    Pools every cell, because the question is about images rather than cells,
    and bins by displacement quintile. A gain that grows with displacement is
    reading (b): a short roll still lands on the object. A flat gain is reading
    (a): the map was simply in the wrong place and anywhere else is as good.
    """
    rows = []
    for imputation in imputations:
        for explainer, rolled in pairs:
            gains, distances = [], []
            for cell in cells:
                base, _, _ = per_image_faithfulness(
                    cell, cache, explainer=explainer, imputation=imputation,
                    indices=indices, explain_options=explain_options,
                )
                moved, _, _ = per_image_faithfulness(
                    cell, cache, explainer=rolled, imputation=imputation,
                    indices=indices, explain_options=explain_options,
                )
                gains.append(moved - base)
                distances.append(roll_displacements(len(base), seed=roll_seed))

            gains = np.concatenate(gains)
            distances = np.concatenate(distances)
            edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
            for lo, hi in zip(edges[:-1], edges[1:]):
                in_bin = (distances >= lo) & (distances <= hi)
                interval = bootstrap_interval(
                    gains[in_bin], np.mean, n_resamples=n_resamples, seed=0
                )
                rows.append({
                    "imputation": imputation,
                    "explainer": explainer,
                    "displacement": float(distances[in_bin].mean()),
                    "gain": interval.point,
                    "half_width": interval.half_width,
                    "n": int(in_bin.sum()),
                })
    return rows


def geometry_test(cells, cache, *, explainers, imputations, indices,
                  explain_options):
    """TEST 2: is faithfulness a function of mask geometry across all arms?

    One row per (imputation, explainer): the arm's mean faithfulness beside the
    two geometry numbers of the masks it induced. Real explainers and model-free
    blobs are deliberately in the same table, because the question is whether
    anything distinguishes them once geometry is matched.
    """
    rows = []
    for imputation in imputations:
        for explainer in explainers:
            faith, centre, spread = [], [], []
            for cell in cells:
                values, _, _ = per_image_faithfulness(
                    cell, cache, explainer=explainer, imputation=imputation,
                    indices=indices, explain_options=explain_options,
                )
                spec = explain_spec(
                    cell, explainer=explainer, indices=indices, **explain_options
                )
                try:
                    attributions = cache.get_array(spec, EXPLAIN_VERSION)
                except KeyError:
                    raise MissingArtifacts(
                        f"no attributions for {cell} / {explainer}. The explain "
                        f"stage has not run for these options."
                    )
                c, s = mask_geometry(attributions.astype(np.float32))
                faith.append(values)
                centre.append(c)
                spread.append(s)

            rows.append({
                "imputation": imputation,
                "explainer": explainer,
                "faithfulness": float(np.mean(np.concatenate(faith))),
                "centre_distance": float(np.mean(np.concatenate(centre))),
                "spread": float(np.mean(np.concatenate(spread))),
                "model_free": explainer.startswith(LOWRES_PREFIX)
                or explainer == "random",
            })
    return rows


def fraction_profile(cells, cache, *, pairs, imputations, indices,
                     explain_options):
    """TEST 3: at which removal fractions does the gap live?

    The mean probability an explainer's mask leaves behind, minus the same for
    its rolled control, at each removal fraction. A gap concentrated at small
    fractions is a boundary artefact, reading (b). A gap tracking the object's
    extent is reading (a).
    """
    rows = []
    for imputation in imputations:
        for explainer, rolled in pairs:
            base_curves, moved_curves = [], []
            for cell in cells:
                _, base, _ = per_image_faithfulness(
                    cell, cache, explainer=explainer, imputation=imputation,
                    indices=indices, explain_options=explain_options,
                )
                _, moved, _ = per_image_faithfulness(
                    cell, cache, explainer=rolled, imputation=imputation,
                    indices=indices, explain_options=explain_options,
                )
                base_curves.append(base)
                moved_curves.append(moved)

            base = np.concatenate(base_curves).mean(axis=0)
            moved = np.concatenate(moved_curves).mean(axis=0)
            for fraction, b, m in zip(REMOVAL_FRACTIONS, base, moved):
                rows.append({
                    "imputation": imputation,
                    "explainer": explainer,
                    "fraction": fraction,
                    "kept_by_explainer": float(b),
                    "kept_by_rolled": float(m),
                    "gap": float(b - m),
                })
    return rows


def _print_report(displacement, geometry, profile) -> None:
    print()
    print("=" * 74)
    print("TEST 1   does rolling FURTHER help more?")
    print("         gain = faithfulness(rolled) - faithfulness(explainer)")
    print("         growing with displacement = reading (b); flat = reading (a)")
    print("=" * 74)
    print(f"{'imputation':<15} {'explainer':<22} {'displ_px':>8}  {'gain':>9}  "
          f"{'half_w':>8}  {'n':>5}")
    for r in displacement:
        print(f"{r['imputation']:<15} {r['explainer']:<22} "
              f"{r['displacement']:>8.2f}  {r['gain']:>+9.5f}  "
              f"{r['half_width']:>8.5f}  {r['n']:>5}")

    print()
    for imputation in sorted({r["imputation"] for r in displacement}):
        for explainer in sorted({r["explainer"] for r in displacement}):
            sub = [r for r in displacement
                   if r["imputation"] == imputation and r["explainer"] == explainer]
            if len(sub) < 2:
                continue
            first, last = sub[0], sub[-1]
            swing = last["gain"] - first["gain"]
            bar = max(first["half_width"], last["half_width"])
            verdict = (
                "GROWS with displacement -> (b)" if swing > bar
                else "SHRINKS with displacement" if swing < -bar
                else "FLAT in displacement -> (a)"
            )
            print(f"  {imputation:<15} {explainer:<22} "
                  f"{first['gain']:+.5f} -> {last['gain']:+.5f}  "
                  f"swing {swing:+.5f} vs half-width {bar:.5f}   {verdict}")

    print()
    print("=" * 74)
    print("TEST 2   faithfulness against mask geometry, all arms on one table")
    print("         centre_distance: mask centroid's distance from frame centre")
    print("         spread: how far masked pixels sit from their own centroid")
    print("=" * 74)
    for imputation in sorted({r["imputation"] for r in geometry}):
        sub = sorted(
            (r for r in geometry if r["imputation"] == imputation),
            key=lambda r: r["spread"],
        )
        print()
        print(f"  imputation: {imputation}   (ordered by mask spread)")
        print(f"    {'explainer':<30} {'spread':>7} {'centre':>7} "
              f"{'faithfulness':>13}  model-free")
        for r in sub:
            print(f"    {r['explainer']:<30} {r['spread']:>7.3f} "
                  f"{r['centre_distance']:>7.3f} {r['faithfulness']:>+13.5f}"
                  f"  {'yes' if r['model_free'] else ''}")

        model_free = [r for r in sub if r["model_free"]]
        real = [r for r in sub if not r["model_free"]]
        if len(model_free) >= 2 and real:
            print("    residual of each real arm against the model-free arms of "
                  "nearest spread:")
            for r in real:
                nearest = min(model_free,
                              key=lambda m: abs(m["spread"] - r["spread"]))
                print(f"      {r['explainer']:<28} vs {nearest['explainer']:<20} "
                      f"(spread {r['spread']:.2f} vs {nearest['spread']:.2f})  "
                      f"residual {r['faithfulness'] - nearest['faithfulness']:+.5f}")

    print()
    print("=" * 74)
    print("TEST 3   where in the removal curve the gap lives")
    print("         probability kept by the explainer's mask minus by a rolled one")
    print("=" * 74)
    for imputation in sorted({r["imputation"] for r in profile}):
        for explainer in sorted({r["explainer"] for r in profile}):
            sub = [r for r in profile
                   if r["imputation"] == imputation and r["explainer"] == explainer]
            if not sub:
                continue
            print()
            print(f"  {imputation} / {explainer}")
            print(f"    {'fraction':>8} {'explainer':>10} {'rolled':>10} {'gap':>9}")
            for r in sub:
                print(f"    {r['fraction']:>8.2f} {r['kept_by_explainer']:>10.4f} "
                      f"{r['kept_by_rolled']:>10.4f} {r['gap']:>+9.4f}")
            peak = max(sub, key=lambda r: abs(r["gap"]))
            print(f"    peak gap at fraction {peak['fraction']:.2f} "
                  f"({peak['gap']:+.4f})")
    print()
    print("=" * 74)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "pilot_validity.yaml")
    parser.add_argument("--cache-read", type=Path, action="append", default=[])
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument(
        "--imputation", action="append", default=[],
        help="Scheme to report; repeatable. Defaults to mean and blur, which "
             "are the two the sign of degradation reverses between.",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    explainers = config["explainers"]
    imputations = args.imputation or [
        s for s in ("mean", "blur") if s in config["imputations"]
    ]

    roots = args.cache_read or [
        p for p in (Path("/kaggle/working/cache"),
                    Path("/kaggle/input/shiftprofile-cache"))
        if p.exists()
    ]
    if not roots:
        print("no cache root found; pass --cache-read", file=sys.stderr)
        return 2
    cache = ArtifactCache(roots[0], read_roots=roots)

    indices = fixed_eval_indices(config["n_eval_images"])
    cells = enumerate_cells(config, config["track"])
    explain_options = explain_options_from_config(config)
    roll_seed = explain_options.get("roll_seed", 0)

    pairs = [
        (name, name + ROLLED_SUFFIX)
        for name in explainers
        if name + ROLLED_SUFFIX in explainers
    ]
    if not pairs:
        print("no rolled arm in this config, so the displacement and fraction "
              "tests have nothing to compare against", file=sys.stderr)
        return 2

    print(f"config      {args.config}")
    print(f"cache read  {', '.join(str(p) for p in roots)}")
    print(f"cells       {len(cells)}")
    print(f"imputation  {', '.join(imputations)}")
    print(f"pairs       {', '.join(f'{a} vs {b}' for a, b in pairs)}")
    print(f"roll seed   {roll_seed}")

    try:
        displacement = displacement_test(
            cells, cache, pairs=pairs, imputations=imputations, indices=indices,
            explain_options=explain_options, roll_seed=roll_seed,
            n_resamples=args.n_resamples,
        )
        geometry = geometry_test(
            cells, cache, explainers=explainers, imputations=imputations,
            indices=indices, explain_options=explain_options,
        )
        profile = fraction_profile(
            cells, cache, pairs=pairs, imputations=imputations, indices=indices,
            explain_options=explain_options,
        )
    except MissingArtifacts as exc:
        print(f"\nincomplete cache: {exc}", file=sys.stderr)
        return 1

    _print_report(displacement, geometry, profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
