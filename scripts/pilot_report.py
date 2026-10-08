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
    Derived from the same config key the fill reads, so the report cannot look
    for curves under settings the fill never used.
    """
    return {"ig_steps": config["ig_steps"]} if "ig_steps" in config else {}


def imputation_from(config: dict) -> str:
    """The one imputation scheme the frozen protocol uses.

    Accepts both spellings for the same reason `fill` does: the configs say
    `imputation` and the filler long read `imputations`, so a config asking for
    blur would have run mean.
    """
    schemes = config.get("imputations") or [config.get("imputation", "mean")]
    return schemes[0]


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


def faithfulness_rows(cells, cache, *, explainers, indices, imputation,
                      explain_options, n_resamples=2000) -> list[dict]:
    """Per (cell, explainer) faithfulness, with the interval the gate is stated in.

    One `relative_faithfulness` per image, because the cell statistic is the mean
    over images -- which is why the interval half-width scales as 1/sqrt(n) and
    why `n_eval_images` is the lever the gate responds to.
    """
    fractions = np.asarray(REMOVAL_FRACTIONS, dtype=np.float64)
    rows, missing = [], []
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
                missing.append((cell, explainer))
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
                "faithfulness": interval.point,
                "half_width": interval.half_width,
                "low": interval.low,
                "high": interval.high,
                "n": interval.n,
            })
    if missing:
        cell, explainer = missing[0]
        raise MissingArtifacts(
            f"{len(missing)} cell-explainer pairs have no cached curves, e.g. "
            f"{cell} / {explainer}. The curves stage has not run for this "
            f"evaluation set and these explain options. Skipping them is not an "
            f"option: a partial grid is not the pre-registered grid."
        )
    return rows


def gate_verdicts(faith_rows, *, gate_ratio=GATE_RATIO, severity=GATE_SEVERITY) -> list[dict]:
    """The P3 gate, per (explainer, shift family).

    Half-width under `gate_ratio` of the clean-to-severity-5 change. The
    half-width taken is the LARGER of the two cells being compared: a change is
    only as well resolved as its blurrier endpoint, and taking the smaller would
    flatter the gate.

    A change of zero is reported as undefined rather than as an infinite ratio.
    "The effect we are trying to resolve is absent" is a different finding from
    "our resolution is too coarse", and collapsing them loses the distinction
    that decides whether the protocol or the metric is what changes.
    """
    by = {}
    for r in faith_rows:
        by[(r["explainer"], r["shift_family"], r["severity"])] = r

    families = sorted({
        r["shift_family"] for r in faith_rows if r["shift_family"] != CLEAN
    })
    verdicts = []
    for explainer in sorted({r["explainer"] for r in faith_rows}):
        clean = by.get((explainer, CLEAN, 0))
        if clean is None:
            continue
        for family in families:
            severe = by.get((explainer, family, severity))
            if severe is None:
                continue
            change = severe["faithfulness"] - clean["faithfulness"]
            half_width = max(clean["half_width"], severe["half_width"])
            ratio = None if change == 0 else half_width / abs(change)
            verdicts.append({
                "explainer": explainer,
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
    for row in beats_random(faith_rows):
        verdict = "NO CELL BEATS RANDOM" if row["cells_above_zero"] == 0 else ""
        print(f"  {row['explainer']:<22} mean {row['mean_faithfulness']:+.5f}   "
              f"{row['cells_above_zero']}/{row['cells']} cells with interval above 0  {verdict}")

    print()
    print(f"{'explainer':<22} {'sev':>4}  {'faithfulness':>13}  {'half_width':>11}")
    for explainer in sorted({r["explainer"] for r in faith_rows}):
        for sev in severities:
            sub = [r for r in faith_rows
                   if r["explainer"] == explainer and r["severity"] == sev]
            if not sub:
                continue
            print(f"{explainer:<22} {sev:>4}  "
                  f"{float(np.mean([r['faithfulness'] for r in sub])):>+13.5f}  "
                  f"{float(np.mean([r['half_width'] for r in sub])):>11.5f}")

    verdicts = gate_verdicts(faith_rows)
    print()
    print("=" * 72)
    print(f"P3 GATE   half-width < {GATE_RATIO:.0%} of the clean-to-severity-"
          f"{GATE_SEVERITY} change")
    print("=" * 72)
    print(f"{'explainer':<22} {'family':<18} {'change':>10}  {'half_w':>9}  {'ratio':>8}  verdict")
    for v in verdicts:
        if v["ratio"] is None:
            # Not FAIL. An undefined ratio means the change is exactly zero, and
            # for the `random` explainer that is the control behaving as a
            # control rather than a result failing a bar.
            ratio, verdict = "n/a", "no change"
        else:
            ratio, verdict = f"{v['ratio']:.3f}", "PASS" if v["passes"] else "FAIL"
        print(f"{v['explainer']:<22} {v['shift_family']:<18} {v['change']:>+10.5f}  "
              f"{v['half_width']:>9.5f}  {ratio:>8}  {verdict}")

    decided = [v for v in verdicts if v["ratio"] is not None]
    passed = [v for v in decided if v["passes"]]
    print()
    if not decided:
        print("GATE UNDECIDABLE: every clean-to-severe change is exactly zero.")
    else:
        print(f"GATE: {len(passed)}/{len(decided)} explainer-family pairs pass.")
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
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())

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
    print(f"imputation  {imputation_from(config)}")
    print(f"explainers  {', '.join(config['explainers'])}")

    try:
        cal_rows = calibration_rows(cells, cache, indices=indices, labels=labels)
        faith_rows = faithfulness_rows(
            cells, cache,
            explainers=config["explainers"],
            indices=indices,
            imputation=imputation_from(config),
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
