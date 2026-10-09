"""Tests for the pilot report.

This exists because its predecessor could not have any. notebook 02 read
predictions under a key with no 'stage' and curves under a key missing
`imputation` and `which`, swallowed both misses, and signed off with "No
faithfulness data found. This is expected if notebook 01 did not complete." A
wrong cache key was indistinguishable from an unfinished run for the life of the
project, and a nine-hour pilot produced valid artifacts and no analysis at all.

So the first test here is not about statistics. It is that reading a correctly
populated cache yields rows, and that reading an incomplete one raises instead
of reporting zero findings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pilot_report  # noqa: E402
from pilot_report import (  # noqa: E402
    GATE_RATIO,
    MissingArtifacts,
    beats_random,
    calibration_rows,
    explain_options_from,
    faithfulness_rows,
    gate_verdicts,
    geometry_ladder,
    imputation_from,
    matched_pairs,
    matched_rows,
)

from shiftprofile.cache import ArtifactCache
from shiftprofile.cells import CLEAN, Cell
from shiftprofile.curves import CURVES_VERSION, REMOVAL_FRACTIONS, curves_spec
from shiftprofile.data import fixed_eval_indices
from shiftprofile.explain import explain_options_from_config, parse_explainer
from shiftprofile.metrics.faithfulness import relative_faithfulness, removal_auc
from shiftprofile.predict import PREDICT_VERSION, predict_spec

N = 16
INDICES = fixed_eval_indices(N, total=N)
EXPLAINERS = ["integrated_gradients", "random"]
CONFIG = {
    "track": "vision",
    "n_eval_images": N,
    "ig_steps": 4,
    "imputation": "mean",
    "models": ["resnet18"],
    "seeds": [0],
    "shift_families": ["fog"],
    "severities": [1, 5],
    "explainers": EXPLAINERS,
}


def _cells():
    return [
        Cell("vision", "resnet18", 0, CLEAN, 0),
        Cell("vision", "resnet18", 0, "fog", 1),
        Cell("vision", "resnet18", 0, "fog", 5),
    ]


def _seed_cache(tmp_path, *, faithfulness_by_severity=None, skip=()):
    """A cache populated the way the real producers populate it.

    Through the producers' own key builders, never a hand-built dict -- seeding
    by hand is how a test comes to assert a cache hit against a key nothing
    writes, which is the failure this whole area was rewritten to remove.
    """
    cache = ArtifactCache(tmp_path / "cache")
    rng = np.random.default_rng(0)
    fractions = np.asarray(REMOVAL_FRACTIONS)

    for cell in _cells():
        if ("predict", cell.shift_family, cell.severity) in skip:
            continue
        logits = rng.normal(size=(N, 10)).astype(np.float32)
        cache.put_array(predict_spec(cell, indices=INDICES), PREDICT_VERSION, logits)

    for cell in _cells():
        for explainer in EXPLAINERS:
            if ("curves", cell.shift_family, cell.severity, explainer) in skip:
                continue
            # A model curve that decays faster than the control gives positive
            # faithfulness; the gap sets its size.
            gap = 0.0
            if faithfulness_by_severity is not None:
                gap = faithfulness_by_severity.get(cell.severity, 0.0)
            model = np.clip(1.0 - (1.0 + gap) * fractions, 0.0, 1.0)
            control = np.clip(1.0 - fractions, 0.0, 1.0)
            for which, curve in (("model", model), ("random", control)):
                cache.put_array(
                    curves_spec(
                        cell, explainer=explainer, imputation="mean",
                        which=which, indices=INDICES,
                        explain_options=explain_options_from(CONFIG),
                    ),
                    CURVES_VERSION,
                    np.tile(curve, (N, 1)).astype(np.float32),
                )
    return cache


class TestItCanActuallyReadTheCache:
    """The regression that matters. Everything else is downstream of this."""

    def test_calibration_rows_are_produced(self, tmp_path):
        cache = _seed_cache(tmp_path)
        labels = np.arange(N, dtype=np.int64) % 10
        rows = calibration_rows(_cells(), cache, indices=INDICES, labels=labels)
        assert len(rows) == len(_cells())
        assert all("brier" in r and "accuracy" in r for r in rows)

    def test_faithfulness_rows_are_produced(self, tmp_path):
        cache = _seed_cache(tmp_path)
        rows = faithfulness_rows(
            _cells(), cache, explainers=EXPLAINERS, indices=INDICES,
            imputations=["mean"], explain_options=explain_options_from(CONFIG),
            n_resamples=200,
        )
        assert len(rows) == len(_cells()) * len(EXPLAINERS)
        assert all(r["half_width"] >= 0 for r in rows)

    def test_missing_predictions_raise_instead_of_reporting_nothing(self, tmp_path):
        """The exact failure mode of the notebook: a silent `continue` per miss,
        an empty result, and a reassuring message about an unfinished run."""
        cache = _seed_cache(tmp_path, skip={("predict", "fog", 5)})
        labels = np.arange(N, dtype=np.int64) % 10
        with pytest.raises(MissingArtifacts, match="no cached predictions"):
            calibration_rows(_cells(), cache, indices=INDICES, labels=labels)

    def test_missing_curves_raise_instead_of_reporting_nothing(self, tmp_path):
        cache = _seed_cache(
            tmp_path, skip={("curves", "fog", 5, "integrated_gradients")}
        )
        with pytest.raises(MissingArtifacts, match="no cached curves"):
            faithfulness_rows(
                _cells(), cache, explainers=EXPLAINERS, indices=INDICES,
                imputations=["mean"], explain_options=explain_options_from(CONFIG),
                n_resamples=200,
            )

    def test_the_wrong_eval_set_is_refused_not_silently_mismatched(self, tmp_path):
        """Asking for 1,000 images against artifacts over 16 must not half-work."""
        cache = _seed_cache(tmp_path)
        other = fixed_eval_indices(8, total=N)
        labels = np.arange(8, dtype=np.int64) % 10
        with pytest.raises(MissingArtifacts):
            calibration_rows(_cells(), cache, indices=other, labels=labels)

    def test_options_the_fill_did_not_use_are_refused(self, tmp_path):
        """Curves are keyed on which attributions they came from, so asking under
        the wrong IG step count must raise rather than return an empty table."""
        cache = _seed_cache(tmp_path)
        with pytest.raises(MissingArtifacts, match="no cached curves"):
            faithfulness_rows(
                _cells(), cache, explainers=EXPLAINERS, indices=INDICES,
                imputations=["mean"], explain_options={"ig_steps": 999},
                n_resamples=200,
            )


class TestTheGateStatistic:
    def _rows(self, clean, severe, *, half_width=0.0):
        return [
            {"model_id": "resnet18", "seed": 0, "shift_family": CLEAN, "severity": 0,
             "explainer": "ig", "imputation": "mean",
             "faithfulness": clean, "half_width": half_width,
             "low": clean - half_width, "high": clean + half_width, "n": 100},
            {"model_id": "resnet18", "seed": 0, "shift_family": "fog", "severity": 5,
             "explainer": "ig", "imputation": "mean",
             "faithfulness": severe, "half_width": half_width,
             "low": severe - half_width, "high": severe + half_width, "n": 100},
        ]

    def test_a_well_resolved_change_passes(self):
        # change 0.10, half-width 0.005 -> ratio 0.05, under the 10% bar
        v = gate_verdicts(self._rows(0.20, 0.10, half_width=0.005))[0]
        assert v["change"] == pytest.approx(-0.10)
        assert v["ratio"] == pytest.approx(0.05)
        assert v["passes"]

    def test_a_poorly_resolved_change_fails(self):
        v = gate_verdicts(self._rows(0.20, 0.10, half_width=0.05))[0]
        assert v["ratio"] == pytest.approx(0.5)
        assert not v["passes"]

    def test_the_bar_is_ten_percent(self):
        assert GATE_RATIO == 0.10
        just_under = gate_verdicts(self._rows(0.0, 0.10, half_width=0.0099))[0]
        just_over = gate_verdicts(self._rows(0.0, 0.10, half_width=0.0101))[0]
        assert just_under["passes"] and not just_over["passes"]

    def test_the_larger_half_width_is_the_one_used(self):
        """A change is only as well resolved as its blurrier endpoint; taking the
        smaller would flatter the gate."""
        rows = self._rows(0.20, 0.10)
        rows[0]["half_width"] = 0.002
        rows[1]["half_width"] = 0.009
        assert gate_verdicts(rows)[0]["half_width"] == pytest.approx(0.009)

    def test_a_change_no_larger_than_its_half_width_is_undefined(self):
        """The bug this replaced tested `change == 0` exactly, which let the
        random control through with a change of 1e-5 against a half-width of
        1.2e-4 and printed a ratio of 67.9 as FAIL -- noise dressed as a
        catastrophic failure, and counted in the summary."""
        v = gate_verdicts(self._rows(0.0, 0.00001, half_width=0.00012))[0]
        assert v["change"] != 0
        assert v["ratio"] is None, "a change under its own uncertainty is not a change"
        assert not v["passes"]

    def test_a_zero_change_is_undefined_not_failed(self):
        """"The effect is absent" and "our resolution is too coarse" are different
        findings, and they imply different pre-registered responses."""
        v = gate_verdicts(self._rows(0.10, 0.10, half_width=0.001))[0]
        assert v["change"] == 0
        assert v["ratio"] is None
        assert not v["passes"]

    def test_a_negative_change_is_judged_on_magnitude(self):
        """Faithfulness falling under shift is the study's hypothesis, so the
        gate must not treat the expected direction as unresolvable."""
        v = gate_verdicts(self._rows(0.30, 0.10, half_width=0.01))[0]
        assert v["change"] < 0
        assert v["ratio"] == pytest.approx(0.05)
        assert v["passes"]


class TestBeatsRandom:
    def test_an_explainer_that_never_clears_zero_is_reported(self, tmp_path):
        """Faithfulness is the control's AUC minus the model's, so zero means the
        explainer carries nothing the random baseline does not. If no cell clears
        it there is no effect whose onset could be predicted, which matters more
        than any gate ratio."""
        rows = [
            {"explainer": "ig", "faithfulness": 0.0, "half_width": 0.01,
             "low": -0.01, "high": 0.01, "shift_family": "fog", "severity": 5,
             "model_id": "resnet18", "seed": 0, "n": 100},
        ]
        [out] = beats_random(rows)
        assert out["cells_above_zero"] == 0

    def test_a_positive_interval_counts(self):
        rows = [
            {"explainer": "ig", "faithfulness": 0.05, "half_width": 0.01,
             "low": 0.04, "high": 0.06, "shift_family": "fog", "severity": 5,
             "model_id": "resnet18", "seed": 0, "n": 100},
        ]
        [out] = beats_random(rows)
        assert out["cells_above_zero"] == 1


class TestGeometryMatchedFaithfulness:
    """The headline number of the validity branch, and the arithmetic it rests on.

    `relative_faithfulness` scores an attribution against a control drawn
    per-pixel i.i.d., so a coarse map is compared with a scattered mask and the
    two differ in shape before they differ in content. The matched comparison
    removes that by subtracting one faithfulness from another instead of changing
    `paired_removal_curves`, which section 6 pins as validity control #1.
    """

    def _row(self, explainer, per_image, *, severity=5, imputation="mean"):
        per_image = np.asarray(per_image, dtype=np.float64)
        return {
            "model_id": "resnet18", "seed": 0, "shift_family": "fog",
            "severity": severity, "explainer": explainer, "imputation": imputation,
            "faithfulness": float(per_image.mean()), "half_width": 0.01,
            "low": float(per_image.mean()) - 0.01,
            "high": float(per_image.mean()) + 0.01,
            "n": len(per_image), "per_image": per_image,
        }

    def test_the_pixel_control_cancels_in_the_difference(self):
        """Why subtracting two reported numbers is a valid direct comparison.

        Both terms are `auc(control) - auc(attribution)` over the same images
        with the same control_seed, imputation and model, so the control term is
        bitwise identical in each and drops out:

            rf(X) - rf(R) = auc(R) - auc(X)

        which is X scored against a control of its own geometry and does not
        involve the pixel-i.i.d. control at all. If this identity did not hold,
        the matched table would be reporting a difference of two incommensurable
        quantities.
        """
        fractions = np.asarray(REMOVAL_FRACTIONS, dtype=np.float64)
        control = np.clip(1.0 - fractions, 0.0, 1.0)
        x_curve = np.clip(1.0 - 1.7 * fractions, 0.0, 1.0)
        r_curve = np.clip(1.0 - 1.2 * fractions, 0.0, 1.0)

        rf_x = relative_faithfulness(x_curve, control, fractions, imputation="mean")
        rf_r = relative_faithfulness(r_curve, control, fractions, imputation="mean")

        assert rf_x - rf_r == pytest.approx(
            removal_auc(r_curve, fractions) - removal_auc(x_curve, fractions)
        )

    def test_the_matched_value_is_the_per_image_mean_difference(self):
        """Paired per image, not a difference of two cell means.

        A difference of means would carry no interval, and the interval is the
        only thing that says whether an arm beat its matched control or landed
        inside the noise -- which is the entire verdict the table prints.
        """
        explainer = np.array([0.10, 0.20, 0.30, 0.05])
        reference = np.array([0.08, 0.25, 0.20, 0.05])
        rows = [
            self._row("grad_cam", explainer),
            self._row("grad_cam_rolled", reference),
        ]

        [out] = matched_rows(rows, n_resamples=200)

        assert out["explainer"] == "grad_cam"
        assert out["reference"] == "grad_cam_rolled"
        assert out["kind"] == "alignment"
        assert out["matched"] == pytest.approx(float((explainer - reference).mean()))
        assert out["half_width"] >= 0

    def test_a_row_without_its_per_image_values_raises(self):
        """`faithfulness_rows` keeps `per_image` precisely so the pairing is
        possible. Rows assembled any other way cannot be paired, and reporting a
        silently shorter table is the failure this whole script replaced."""
        rows = [
            self._row("grad_cam", [0.1, 0.2]),
            self._row("grad_cam_rolled", [0.1, 0.2]),
        ]
        rows[1]["per_image"] = None

        with pytest.raises(MissingArtifacts, match="per-image"):
            matched_rows(rows, n_resamples=200)

    def test_the_pairs_come_from_what_the_run_contains(self):
        """A config that omits an arm must produce a shorter table, not a crash:
        the geometry and alignment arms are new and will not be present in every
        cache this report is pointed at."""
        rolled_only = [
            self._row("integrated_gradients", [0.1, 0.2]),
            self._row("integrated_gradients_rolled", [0.1, 0.2]),
        ]
        assert matched_pairs(rolled_only) == [
            ("integrated_gradients", "integrated_gradients_rolled", "alignment")
        ]
        assert matched_pairs([self._row("grad_cam", [0.1])]) == []


class TestTheGeometryLadder:
    """Every rung sees no model, no image and no label, so it is blob size alone."""

    def _row(self, explainer, faithfulness, *, severity=5, imputation="mean"):
        return {
            "model_id": "resnet18", "seed": 0, "shift_family": "fog",
            "severity": severity, "explainer": explainer, "imputation": imputation,
            "faithfulness": faithfulness, "half_width": 0.01,
            "low": faithfulness - 0.01, "high": faithfulness + 0.01, "n": 4,
        }

    def _rows(self):
        return (
            [self._row(f"random_lowres_{g}", -0.3 + 0.01 * g) for g in (8, 2, 32, 4, 16)]
            + [self._row(name, -0.29) for name in
               ("grad_cam", "integrated_gradients", "random", "grad_cam_rolled")]
        )

    def test_the_rungs_are_ordered_by_grid_whatever_order_the_rows_arrive_in(self):
        """It is read as a curve against blob size, so the rows have to come out
        in that order. `sorted()` on the names would give 16 before 2."""
        ladder = geometry_ladder(self._rows())
        assert [row["grid"] for row in ladder] == [2, 4, 8, 16, 32]

    def test_only_the_lowres_arms_are_rungs(self):
        """`random` is the control the protocol subtracts, not the grid=32 rung:
        it is signed with no ReLU and no interpolation, so it ranks differently
        under |attribution| than any point on this curve. Grad-CAM is the thing
        being located ON the curve and cannot also be part of it.
        """
        ladder = geometry_ladder(self._rows())
        assert len(ladder) == 5
        assert all(row["cells"] == 1 for row in ladder)

    def test_each_imputation_gets_its_own_ladder(self):
        """The pilot's whole finding is that the scheme changes the ordering, so
        averaging the schemes together would hide the effect being measured."""
        rows = self._rows() + [
            self._row(f"random_lowres_{g}", -0.06, imputation="blur")
            for g in (2, 4, 8, 16, 32)
        ]
        ladder = geometry_ladder(rows)
        assert [(row["imputation"], row["grid"]) for row in ladder] == [
            ("blur", g) for g in (2, 4, 8, 16, 32)
        ] + [("mean", g) for g in (2, 4, 8, 16, 32)]


class TestConfigReading:
    def test_the_singular_imputation_spelling_is_honoured(self):
        """Both configs spell it singular and the filler long read it as plural,
        so a config asking for blur would have run mean."""
        assert imputation_from({"imputation": "blur"}) == "blur"

    def test_the_plural_spelling_wins_when_present(self):
        assert imputation_from({"imputations": ["zero"], "imputation": "mean"}) == "zero"

    def test_it_defaults_to_mean_when_absent(self):
        assert imputation_from({}) == "mean"

    def test_ig_steps_are_passed_through_so_curves_can_be_found(self):
        """The report must look for curves under the step count the fill used."""
        assert explain_options_from({"ig_steps": 4}) == {"ig_steps": 4}
        assert explain_options_from({}) == {}

    def test_the_ig_baseline_is_passed_through_too(self):
        """It was not, and `fill` did not forward it either, so a pre-registered
        ablation axis sat in the cache key with no way to set it."""
        assert explain_options_from({"ig_baseline": "black"}) == {"baseline": "black"}

    def test_the_report_reads_the_config_through_the_filler_s_own_function(self):
        """One source, asserted rather than maintained by hand. The filler decides
        what a curve is computed under and the report decides what key it is
        looked up by; a one-entry difference makes a full cache read as an empty
        one, and they held separate copies of this mapping for one commit."""
        config = {
            "ig_steps": 16, "ig_baseline": "black", "roll_seed": 2,
            "epochs": 50, "imputations": ["mean"],
        }
        assert explain_options_from(config) == explain_options_from_config(config)
        assert explain_options_from(config) == {
            "ig_steps": 16, "baseline": "black", "roll_seed": 2,
        }


def test_the_pilot_config_is_readable_by_this_script():
    """A guard against the config and the report drifting apart again."""
    import yaml

    config = yaml.safe_load(
        (Path(pilot_report.REPO_ROOT) / "configs" / "pilot.yaml").read_text()
    )
    assert config["n_eval_images"] == 1000
    assert imputation_from(config) == "mean"
    assert explain_options_from(config) == {"ig_steps": 32}


def test_the_validity_config_asks_for_arms_that_exist():
    """The same guard for the config that adds the new arms.

    Every rung and every control is requested by NAME, and the name carries the
    grid and the roll, so a typo is not a bad value in a field -- it is an
    explainer the dispatch has never heard of, found after the GPU time is spent.
    `parse_explainer` is the one place that knows, so the config is checked
    against it here rather than at the end of a run.
    """
    import yaml

    config = yaml.safe_load(
        (Path(pilot_report.REPO_ROOT) / "configs" / "pilot_validity.yaml").read_text()
    )
    for explainer in config["explainers"]:
        parse_explainer(explainer)

    assert explain_options_from(config) == {
        "ig_steps": 32, "baseline": "mean", "roll_seed": 0,
    }
    assert matched_pairs([{"explainer": name} for name in config["explainers"]]) == [
        ("grad_cam", "grad_cam_rolled", "alignment"),
        ("integrated_gradients", "integrated_gradients_rolled", "alignment"),
        ("grad_cam", "random_lowres_4", "geometry"),
    ]


class TestThePrintedReport:
    """A formatting crash here costs a Kaggle round trip, so it is covered."""

    def _rows(self):
        cal, faith = [], []
        for fam, sev in [(CLEAN, 0), ("fog", 1), ("fog", 5)]:
            cal.append(dict(
                model_id="resnet18", seed=0, shift_family=fam, severity=sev,
                brier=0.1, ece_equal_mass=0.05, ece_debiased=0.04,
                aurc=0.08, accuracy=0.9,
            ))
            for ex in ("integrated_gradients", "random"):
                pt = 0.05 - sev * 0.008 if ex != "random" else 0.0
                faith.append(dict(
                    model_id="resnet18", seed=0, shift_family=fam, severity=sev,
                    explainer=ex, imputation="mean",
                    faithfulness=pt, half_width=0.003,
                    low=pt - 0.003, high=pt + 0.003, n=1000,
                ))
        return cal, faith

    def test_it_prints_without_crashing(self, capsys):
        cal, faith = self._rows()
        pilot_report._print_report(
            {"n_eval_images": 1000, "explainers": ["integrated_gradients", "random"]},
            cal, faith,
        )
        out = capsys.readouterr().out
        assert "P3 GATE" in out
        assert "CALIBRATION" in out
        assert "FAITHFULNESS" in out

    def test_a_zero_change_prints_no_change_not_fail(self, capsys):
        """The random control has zero change by construction. Printing FAIL for
        it would contradict gate_verdicts' own distinction between an absent
        effect and a resolution that is too coarse."""
        cal, faith = self._rows()
        pilot_report._print_report(
            {"n_eval_images": 1000, "explainers": ["integrated_gradients", "random"]},
            cal, faith,
        )
        gate = capsys.readouterr().out.split("P3 GATE")[1]
        line = next(
            l for l in gate.splitlines()
            if "random" in l and "fog" in l
        )
        assert "no change" in line
        assert "FAIL" not in line

    def test_a_failing_gate_names_the_pre_registered_response(self, capsys):
        """It must not read as "collect more images": the half-width falls as
        1/sqrt(n), so that route costs 100x for a 10x narrowing and is a
        data-dependent protocol change."""
        cal, faith = self._rows()
        # Resolved but coarse: the clean-to-severe change is 0.04, so a
        # half-width of 0.02 clears the "is there a change" test and then fails
        # the 10% bar at a ratio of 0.5. A half-width of 0.5 would swamp the
        # change entirely and correctly report "no change" instead.
        for r in faith:
            r["half_width"] = 0.02
        pilot_report._print_report(
            {"n_eval_images": 1000, "explainers": ["integrated_gradients", "random"]},
            cal, faith,
        )
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "does not mean raising n_eval_images" in out
