"""Compute the pilot's calibration, faithfulness and P3 gate verdict from the cache.

Why a script rather than the notebook: notebook 02 read predictions under
`cell.spec()`, which carries no 'stage' and so matched nothing `predict_cell`
ever wrote, and read curves under a key missing `imputation` and `which` with
`get_record` where `curves_cell` writes two `put_array`s. Both misses were
swallowed, and the cell ended with "No faithfulness data found. This is expected
if notebook 01 did not complete." So a wrong cache key read as an unfinished run
for the whole life of the project, and a nine-hour pilot produced valid
artifacts and zero analysis.

None of that was caught because nothing ran it headlessly and nothing could test
it. A script can be imported, so `tests/test_pilot_report.py` exercises these
functions against a seeded cache. That is the actual fix; reading the cache
correctly is just the bug.

Invocation mirrors `scripts/run_fill.py`: it puts the repo root on `sys.path`
from its own location, so it does not depend on `pip install -e` having
registered against whichever interpreter a notebook `!` line resolves to.

    !python /kaggle/working/shiftprofile/scripts/pilot_report.py

Any flag wins over the default, so the same script runs locally:

    python scripts/pilot_report.py --config configs/pilot.yaml --cache-read ./cache
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.special import softmax  # noqa: E402

from shiftprofile.cache import ArtifactCache  # noqa: E402
from shiftprofile.cells import CLEAN, Cell, enumerate_cells  # noqa: E402
from shiftprofile.curves import CURVES_VERSION, REMOVAL_FRACTIONS, curves_spec  # noqa: E402
from shiftprofile.data import fixed_eval_indices, load_cifar10_test  # noqa: E402
from shiftprofile.explain import (  # noqa: E402
    LOWRES_PREFIX,
    ROLLED_SUFFIX,
    explain_options_from_config,
)
from shiftprofile.metrics.calibration import (  # noqa: E402
    aurc,
    brier_score,
    ece_debiased,
    ece_equal_mass,
)
from shiftprofile.metrics.faithfulness import relative_faithfulness  # noqa: E402
from shiftprofile.predict import PREDICT_VERSION, predict_spec  # noqa: E402
from shiftprofile.stats.bootstrap import bootstrap_interval  # noqa: E402

# The P3 gate, from the spec: the faithfulness interval half-width must be under
# this fraction of the clean-to-severity-5 change, or the measurement cannot
# support the claim and the protocol changes.
GATE_RATIO = 0.10

# The severity the gate compares clean against.
GATE_SEVERITY = 5


class MissingArtifacts(RuntimeError):
    """Named so the caller can tell "not run yet" from "computed wrong"."""


def explain_options_from(config: dict) -> dict:
    """The attribution settings the fill ran under, in `explain_spec` vocabulary.

    `curves_spec` needs these to name which attributions a curve was built from.
    Delegates to the explain module so the report and the filler cannot read the
    config differently: the filler decides what a curve is computed under and the
    report decides what key it is looked up by, so a one-entry difference makes a
    full cache read as an empty one.
    """
    return explain_options_from_config(config)


def imputations_from(config: dict) -> list[str]:
    """Every imputation scheme this config asks for, in order.

    Accepts both spellings for the same reason `fill` does: the configs say
    `imputation` and the filler long read `imputations`, so a config asking for
    blur would have run mean.

    A list rather than one scheme because the scheme is not incidental. The
    pilot found faithfulness NEGATIVE for every explainer under `mean` -- both
    Integrated Gradients and Grad-CAM scoring worse than the random control at
    every severity, with the ordering tracking how spatially contiguous each
    attribution map is rather than how good it is. Grad-CAM's map is 4x4,
    sixteen values upsampled to 1024 pixels, so its "top 5%" is one smooth blob,
    and replacing a contiguous patch with the channel mean disturbs the model far
    less than scattering mean-valued pixels does. Comparing schemes is how that
    is told apart from a real faithfulness signal, which is what E6 is for.
    """
    return list(config.get("imputations") or [config.get("imputation", "mean")])


def imputation_from(config: dict) -> str:
    """The first scheme, for callers that want the frozen protocol's single one."""
    return imputations_from(config)[0]


def calibration_rows(cells, cache, *, indices, labels) -> list[dict]:
    """Per-cell calibration metrics, read through predict's own key builder."""
    rows, missing = [], []
    for cell in cells:
        try:
            logits = cache.get_array(predict_spec(cell, indices=indices), PREDICT_VERSION)
        except KeyError:
            missing.append(cell)
            continue
        if logits.shape[0] != len(indices):
            raise MissingArtifacts(
                f"{cell}: logits have {logits.shape[0]} rows, expected {len(indices)}. "
                f"They were computed under a different n_eval_images than this "
                f"report was given."
            )
        probs = softmax(logits, axis=1)
        rows.append({
            "model_id": cell.model_id,
            "seed": cell.seed,
            "shift_family": cell.shift_family,
            "severity": cell.severity,
            "brier": brier_score(probs, labels),
            "ece_equal_mass": ece_equal_mass(probs, labels, n_bins=15),
            "ece_debiased": ece_debiased(probs, labels, n_bins=15),
            "aurc": aurc(probs, labels),
            "accuracy": float((probs.argmax(axis=1) == labels).mean()),
        })
    if missing:
        raise MissingArtifacts(
            f"{len(missing)} of {len(cells)} cells have no cached predictions for "
            f"{len(indices)} evaluation images, e.g. {missing[0]}. Either the fill "
            f"has not run for this evaluation set, or it ran for a different one -- "
            f"compare the run manifest's n_eval_images against this config."
        )
    return rows


def faithfulness_rows(cells, cache, *, explainers, indices, imputations,
                      explain_options, n_resamples=2000) -> list[dict]:
    """Per (cell, explainer, imputation) faithfulness, with the gate's interval.

    One `relative_faithfulness` per image, because the cell statistic is the mean
    over images -- which is why the interval half-width scales as 1/sqrt(n) and
    why `n_eval_images` is the lever the gate responds to.

    `imputations` is a list: the scheme is a first-class axis, not a setting.
    See `imputations_from`.
    """
    fractions = np.asarray(REMOVAL_FRACTIONS, dtype=np.float64)
    if isinstance(imputations, str):
        raise TypeError(
            f"imputations must be a list, got the string {imputations!r}. A bare "
            f"string would iterate character by character and look for curves "
            f"under an imputation named 'm'."
        )
    rows, missing = [], []
    for imputation in imputations:
        for cell in cells:
            for explainer in explainers:
                def curve(which):
                    return cache.get_array(
                        curves_spec(
                            cell, explainer=explainer, imputation=imputation,
                            which=which, indices=indices,
                            explain_options=explain_options,
                        ),
                        CURVES_VERSION,
                    )

                try:
                    model_curve, random_curve = curve("model"), curve("random")
                except KeyError:
                    missing.append((cell, explainer, imputation))
                    continue

                per_image = np.array([
                    relative_faithfulness(m, r, fractions, imputation=imputation)
                    for m, r in zip(model_curve, random_curve)
                ])
                interval = bootstrap_interval(
                    per_image, np.mean, n_resamples=n_resamples, seed=0
                )
                rows.append({
                    "model_id": cell.model_id,
                    "seed": cell.seed,
                    "shift_family": cell.shift_family,
                    "severity": cell.severity,
                    "explainer": explainer,
                    "imputation": imputation,
                    "faithfulness": interval.point,
                    "half_width": interval.half_width,
                    "low": interval.low,
                    "high": interval.high,
                    "n": interval.n,
                    # Kept, not just summarised, because `matched_rows` needs the
                    # per-image values of TWO explainers side by side to pair
                    # them. A difference of two cell means carries no interval.
                    "per_image": per_image,
                })
    if missing:
        cell, explainer, imputation = missing[0]
        raise MissingArtifacts(
            f"{len(missing)} cell-explainer-imputation triples have no cached "
            f"curves, e.g. {cell} / {explainer} / {imputation}. The curves stage "
            f"has not run for this evaluation set, imputation and explain "
            f"options. Skipping them is not an option: a partial grid is not the "
            f"pre-registered grid."
        )
    return rows


def matched_pairs(faith_rows) -> list[tuple[str, str, str]]:
    """Which explainers have a geometry-matched control present to compare against.

    Derived from what the run actually contains rather than hardcoded, so a
    config that omits an arm produces a shorter table instead of a crash.

    Two kinds of pair:

    `alignment`  X against `X_rolled` -- same map, translated. Holds geometry and
                 the value multiset exactly fixed and removes only the map's
                 registration to the image.
    `geometry`   grad_cam against `random_lowres_4` -- same blob size, no model.
                 4 is ResNet-18's final feature-map resolution, which is the
                 resolution Grad-CAM's map actually has.
    """
    present = {r["explainer"] for r in faith_rows}
    pairs = [
        (explainer, explainer + ROLLED_SUFFIX, "alignment")
        for explainer in sorted(present)
        if explainer + ROLLED_SUFFIX in present
    ]
    if "grad_cam" in present and f"{LOWRES_PREFIX}4" in present:
        pairs.append(("grad_cam", f"{LOWRES_PREFIX}4", "geometry"))
    return pairs


def matched_rows(faith_rows, *, pairs=None, n_resamples=2000) -> list[dict]:
    """Faithfulness measured against a control matched on mask geometry.

    The headline number this report exists to produce. `relative_faithfulness`
    compares an explainer against a control drawn per-pixel i.i.d., so a method
    whose attribution map is coarse is scored against a mask that is scattered,
    and the two masks differ in shape before they differ in content. On 32x32
    inputs that difference dominates: replacing one contiguous patch leaves far
    more of a CIFAR image intact than replacing the same NUMBER of scattered
    pixels, so the coarse method's probability falls less and it is recorded as
    less faithful -- whatever it actually pointed at.

    Subtracting one faithfulness from another removes it. Both are
    `auc(control) - auc(attribution)` over the same images with the same
    `control_seed`, the same imputation and the same model, so the control term
    is bitwise identical between them and cancels exactly:

        matched = rf(X) - rf(R) = auc(R) - auc(X)

    which is X scored directly against a control of its own geometry, and does
    not depend on the pixel-i.i.d. control at all. Done per image and then
    bootstrapped, because the pairing is what makes the interval tight: a
    difference of two cell means would have no interval.

    Raises:
        MissingArtifacts: If a row lacks the per-image values the pairing needs.
    """
    pairs = matched_pairs(faith_rows) if pairs is None else pairs
    by = {
        (r["explainer"], r.get("imputation"), r["shift_family"], r["severity"]): r
        for r in faith_rows
    }

    out = []
    for explainer, reference, kind in pairs:
        for key, row in by.items():
            if key[0] != explainer:
                continue
            ref = by.get((reference,) + key[1:])
            if ref is None:
                continue
            for name, candidate in ((explainer, row), (reference, ref)):
                if candidate.get("per_image") is None:
                    raise MissingArtifacts(
                        f"{name} has no per-image faithfulness values, so it "
                        f"cannot be paired against {reference}. These rows did "
                        f"not come from faithfulness_rows."
                    )
            # Element-wise: both arrays run over `fixed_eval_indices` in the same
            # order, so position i is the same image in both.
            interval = bootstrap_interval(
                row["per_image"] - ref["per_image"],
                np.mean, n_resamples=n_resamples, seed=0,
            )
            out.append({
                "explainer": explainer,
                "reference": reference,
                "kind": kind,
                "imputation": key[1],
                "shift_family": key[2],
                "severity": key[3],
                "matched": interval.point,
                "half_width": interval.half_width,
                "low": interval.low,
                "high": interval.high,
                "beats_control": interval.low > 0,
            })
    return out


def geometry_ladder(faith_rows) -> list[dict]:
    """Mean faithfulness per `random_lowres` grid, per imputation.

    None of these arms sees the model, the image or the label, so every value
    here is produced by blob size alone. Read as a curve against grid size: if
    Grad-CAM's score lands on it at grid=4, the score is its resolution.
    """
    rows = []
    for imputation in sorted({r.get("imputation") for r in faith_rows}, key=str):
        for explainer in sorted(
            {r["explainer"] for r in faith_rows
             if r["explainer"].startswith(LOWRES_PREFIX)},
            key=lambda name: int(name[len(LOWRES_PREFIX):]),
        ):
            sub = [r for r in faith_rows
                   if r["explainer"] == explainer
                   and r.get("imputation") == imputation]
            if not sub:
                continue
            rows.append({
                "imputation": imputation,
                "grid": int(explainer[len(LOWRES_PREFIX):]),
                "faithfulness": float(np.mean([r["faithfulness"] for r in sub])),
                "half_width": float(np.mean([r["half_width"] for r in sub])),
                "cells": len(sub),
            })
    return rows


def gate_verdicts(faith_rows, *, gate_ratio=GATE_RATIO, severity=GATE_SEVERITY) -> list[dict]:
    """The P3 gate, per (explainer, shift family).

    Half-width under `gate_ratio` of the clean-to-severity-5 change. The
    half-width taken is the LARGER of the two cells being compared: a change is
    only as well resolved as its blurrier endpoint, and taking the smaller would
    flatter the gate.

    A change no larger than its own half-width is reported as undefined rather
    than as a ratio. "The effect we are trying to resolve is absent" is a
    different finding from "our resolution is too coarse", and collapsing them
    loses the distinction that decides whether the protocol or the metric is
    what changes.

    The test is `abs(change) <= half_width`, not `change == 0`. An exact-zero
    test let the random control through with changes of 1e-5 against half-widths
    of 1.2e-4 and printed ratios of 67.9 and 10.1 as FAIL -- three rows of noise
    dressed as catastrophic failures, in the summary count as well. A change
    smaller than its uncertainty is not a change.
    """
    by = {}
    for r in faith_rows:
        key = (r["explainer"], r.get("imputation"), r["shift_family"], r["severity"])
        by[key] = r

    families = sorted({
        r["shift_family"] for r in faith_rows if r["shift_family"] != CLEAN
    })
    schemes = sorted({r.get("imputation") for r in faith_rows}, key=str)
    verdicts = []
    for imputation in schemes:
        for explainer in sorted({r["explainer"] for r in faith_rows}):
            clean = by.get((explainer, imputation, CLEAN, 0))
            if clean is None:
                continue
            for family in families:
                severe = by.get((explainer, imputation, family, severity))
                if severe is None:
                    continue
                change = severe["faithfulness"] - clean["faithfulness"]
                half_width = max(clean["half_width"], severe["half_width"])
                resolved = abs(change) > half_width
                ratio = half_width / abs(change) if resolved else None
                verdicts.append({
                    "explainer": explainer,
                    "imputation": imputation,
                    "shift_family": family,
                    "clean": clean["faithfulness"],
                    "severe": severe["faithfulness"],
                    "change": change,
                    "half_width": half_width,
                    "ratio": ratio,
                    "passes": ratio is not None and ratio < gate_ratio,
                })
    return verdicts


def beats_random(faith_rows) -> list[dict]:
    """Whether each explainer's interval excludes zero.

    Faithfulness IS the random control's AUC minus the model's, so zero means
    the explainer carries no information the random baseline does not. An
    explainer that never clears zero makes the gate moot: there is no effect
    whose onset could be predicted, and that is a more consequential finding
    than any timing result.
    """
    out = []
    for explainer in sorted({r["explainer"] for r in faith_rows}):
        rows = [r for r in faith_rows if r["explainer"] == explainer]
        above = [r for r in rows if r["low"] > 0]
        out.append({
            "explainer": explainer,
            "cells": len(rows),
            "cells_above_zero": len(above),
            "mean_faithfulness": float(np.mean([r["faithfulness"] for r in rows])),
        })
    return out


def _print_report(config, cal_rows, faith_rows) -> None:
    severities = sorted({r["severity"] for r in cal_rows})

    print()
    print("=" * 72)
    print(f"CALIBRATION  ({len(cal_rows)} cells, n={config['n_eval_images']} images)")
    print("=" * 72)
    print(f"{'sev':>4}  {'accuracy':>9}  {'brier':>8}  {'ece_em':>8}  {'ece_db':>8}  {'aurc':>8}")
    for sev in severities:
        sub = [r for r in cal_rows if r["severity"] == sev]
        def m(k):
            return float(np.mean([r[k] for r in sub]))
        print(f"{sev:>4}  {m('accuracy'):>9.4f}  {m('brier'):>8.4f}  "
              f"{m('ece_equal_mass'):>8.4f}  {m('ece_debiased'):>8.4f}  {m('aurc'):>8.4f}")

    print()
    print("=" * 72)
    print("FAITHFULNESS  (random control AUC minus model AUC; >0 beats random)")
    print("=" * 72)
    schemes = sorted({r.get("imputation") for r in faith_rows}, key=str)
    for imputation in schemes:
        subset = [r for r in faith_rows if r.get("imputation") == imputation]
        print()
        print(f"  imputation: {imputation}")
        for row in beats_random(subset):
            verdict = "NO CELL BEATS RANDOM" if row["cells_above_zero"] == 0 else ""
            print(f"    {row['explainer']:<22} mean {row['mean_faithfulness']:+.5f}   "
                  f"{row['cells_above_zero']}/{row['cells']} cells above 0  {verdict}")

    print()
    print(f"{'imputation':<15} {'explainer':<22} {'sev':>4}  "
          f"{'faithfulness':>13}  {'half_width':>11}")
    for imputation in schemes:
        for explainer in sorted({r["explainer"] for r in faith_rows}):
            for sev in severities:
                sub = [r for r in faith_rows
                       if r["explainer"] == explainer and r["severity"] == sev
                       and r.get("imputation") == imputation]
                if not sub:
                    continue
                print(f"{str(imputation):<15} {explainer:<22} {sev:>4}  "
                      f"{float(np.mean([r['faithfulness'] for r in sub])):>+13.5f}  "
                      f"{float(np.mean([r['half_width'] for r in sub])):>11.5f}")

    ladder = geometry_ladder(faith_rows)
    if ladder:
        print()
        print("=" * 72)
        print("GEOMETRY LADDER   random_lowres_<grid>: noise on a grid x grid")
        print("                  lattice, upsampled as Grad-CAM is. No model input.")
        print("=" * 72)
        print(f"{'imputation':<15} {'grid':>5}  {'faithfulness':>13}  {'half_width':>11}")
        for row in ladder:
            print(f"{str(row['imputation']):<15} {row['grid']:>5}  "
                  f"{row['faithfulness']:>+13.5f}  {row['half_width']:>11.5f}")
        print()
        print("Grad-CAM's map is 4x4 on ResNet-18. If its faithfulness sits on "
              "this\ncurve at grid=4, its score is its resolution and not its "
              "content.")

    matched = matched_rows(faith_rows)
    if matched:
        print()
        print("=" * 72)
        print("GEOMETRY-MATCHED FAITHFULNESS   explainer minus a control of the")
        print("                                same mask geometry; >0 beats it")
        print("=" * 72)
        print(f"{'imputation':<15} {'explainer':<24} {'vs':<20} {'kind':<10} "
              f"{'matched':>10}  {'half_w':>9}  verdict")
        for imputation in sorted({r["imputation"] for r in matched}, key=str):
            for explainer in sorted({r["explainer"] for r in matched}):
                sub = [r for r in matched
                       if r["imputation"] == imputation
                       and r["explainer"] == explainer]
                if not sub:
                    continue
                for reference in sorted({r["reference"] for r in sub}):
                    pair = [r for r in sub if r["reference"] == reference]
                    point = float(np.mean([r["matched"] for r in pair]))
                    half = float(np.mean([r["half_width"] for r in pair]))
                    above = sum(1 for r in pair if r["beats_control"])
                    verdict = (
                        f"{above}/{len(pair)} cells beat it"
                        if above else "NO CELL BEATS ITS MATCHED CONTROL"
                    )
                    print(f"{str(imputation):<15} {explainer:<24} {reference:<20} "
                          f"{pair[0]['kind']:<10} {point:>+10.5f}  {half:>9.5f}  "
                          f"{verdict}")
        print()
        print("The pixel-i.i.d. control cancels exactly in this difference: both "
              "terms\nuse the same control_seed over the same images, so what is "
              "left is the\nexplainer scored against its own geometry.")

    verdicts = gate_verdicts(faith_rows)
    print()
    print("=" * 72)
    print(f"P3 GATE   half-width < {GATE_RATIO:.0%} of the clean-to-severity-"
          f"{GATE_SEVERITY} change")
    print("=" * 72)
    print(f"{'imputation':<15} {'explainer':<22} {'family':<16} {'change':>10}  "
          f"{'half_w':>9}  {'ratio':>8}  verdict")
    for v in verdicts:
        if v["ratio"] is None:
            # Not FAIL. The change is no larger than its own half-width, so there
            # is nothing to resolve -- which for the `random` explainer is the
            # control behaving as a control, not a result failing a bar.
            ratio, verdict = "n/a", "no change"
        else:
            ratio, verdict = f"{v['ratio']:.3f}", "PASS" if v["passes"] else "FAIL"
        print(f"{str(v.get('imputation')):<15} {v['explainer']:<22} "
              f"{v['shift_family']:<16} {v['change']:>+10.5f}  "
              f"{v['half_width']:>9.5f}  {ratio:>8}  {verdict}")

    decided = [v for v in verdicts if v["ratio"] is not None]
    passed = [v for v in decided if v["passes"]]
    print()
    if not decided:
        print("GATE UNDECIDABLE: no clean-to-severe change exceeds its half-width.")
    else:
        print(f"GATE: {len(passed)}/{len(decided)} resolvable pairs pass.")
    if len(passed) != len(decided) or not decided:
        print(
            "Per the pre-registered failure branch, a failing gate means changing "
            "the faithfulness metric or pivoting to a measurement-validity paper. "
            "It does not mean raising n_eval_images: the half-width falls as "
            "1/sqrt(n), so passing by that route costs 100x the compute for a 10x "
            "narrowing and is a data-dependent protocol change besides."
        )
    print("=" * 72)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "pilot.yaml")
    parser.add_argument("--cache-read", type=Path, action="append", default=[],
                        help="Cache root to read; repeatable. Defaults to the "
                             "Kaggle working cache plus the mounted dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/working/data"))
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument(
        "--imputation", action="append", default=[],
        help="Imputation scheme to report; repeatable. Defaults to every scheme "
             "the config names. The pilot found faithfulness negative under "
             "'mean' for every explainer, so comparing schemes is how a real "
             "signal is told from an artefact of the masking.",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    imputations = args.imputation or imputations_from(config)

    roots = args.cache_read or [
        p for p in (Path("/kaggle/working/cache"),
                    Path("/kaggle/input/shiftprofile-cache"))
        if p.exists()
    ]
    if not roots:
        print("no cache root found; pass --cache-read", file=sys.stderr)
        return 2
    # Read-only: this script computes, it never writes an artifact. The first
    # root doubles as the write root the cache requires, and nothing writes.
    cache = ArtifactCache(roots[0], read_roots=roots)

    indices = fixed_eval_indices(config["n_eval_images"])
    _, labels = load_cifar10_test(args.data_root)
    labels = labels[indices]

    cells = enumerate_cells(config, config["track"])
    print(f"config      {args.config}")
    print(f"cache read  {', '.join(str(p) for p in roots)}")
    print(f"cells       {len(cells)}")
    print(f"eval images {config['n_eval_images']}")
    print(f"imputation  {', '.join(imputations)}")
    print(f"explainers  {', '.join(config['explainers'])}")

    try:
        cal_rows = calibration_rows(cells, cache, indices=indices, labels=labels)
        faith_rows = faithfulness_rows(
            cells, cache,
            explainers=config["explainers"],
            indices=indices,
            imputations=imputations,
            explain_options=explain_options_from(config),
            n_resamples=args.n_resamples,
        )
    except MissingArtifacts as exc:
        print(f"\nincomplete cache: {exc}", file=sys.stderr)
        return 1

    _print_report(config, cal_rows, faith_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
