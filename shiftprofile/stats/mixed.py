"""Mixed-effects coupling model.

Fits

    metric_change ~ accuracy_change + model + shift_family + (1 | seed)

The coefficient on accuracy_change is a nuisance parameter. The RESULT is what
is left after it: if accuracy change explains every metric change, the study's
answer is "cheap accuracy monitoring suffices". If a metric moves beyond what
accuracy predicts, that residual is the decoupling the study exists to find.

Cells within a seed are not independent, so seed enters as a random effect.
Treating them as independent would be pseudo-replication.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REQUIRED_COLUMNS = ("seed", "model", "shift_family", "accuracy_change")


class ConvergenceError(RuntimeError):
    """Every estimator in the ladder failed. Escalate — do not reinterpret."""


# Ordered estimator ladder, most preferred first.
#
# REML is preferred because it gives unbiased variance-component estimates, and
# this study has only five seeds — precisely the small-group regime where ML's
# downward bias on the seed variance matters. Measured on a synthetic 5-seed
# grid: REML 0.000714 vs ML 0.000572, about 20% lower.
#
# But convergence is an optimizer-by-data interaction, not a property of REML:
# lbfgs and powell each hit singular matrices on datasets the other handles.
# Hardcoding any single pair means silently failing on some real dataset, so the
# ladder falls through on failure and records which rung actually ran.
#
# The fixed effect and residual are robust to the choice (agreeing to three
# decimals across estimators in testing); only the variance component moves.
FIT_LADDER = (
    ("reml/default", {"reml": True}),  # statsmodels' own bfgs->lbfgs->cg chain
    ("reml/lbfgs", {"reml": True, "method": "lbfgs"}),
    ("reml/powell", {"reml": True, "method": "powell"}),
    ("reml/cg", {"reml": True, "method": "cg"}),
    ("ml/powell", {"reml": False, "method": "powell"}),
    ("ml/lbfgs", {"reml": False, "method": "lbfgs"}),
)


@dataclass(frozen=True)
class CouplingResult:
    accuracy_coefficient: float
    accuracy_ci: tuple[float, float]
    accuracy_pvalue: float
    residual_std: float
    seed_variance: float
    n_cells: int
    estimator: str
    attempts: tuple[str, ...]
    summary: str

    @property
    def used_reml(self) -> bool:
        return self.estimator.startswith("reml")

    @property
    def seed_variance_at_boundary(self) -> bool:
        """True when the random effect collapsed to exactly zero.

        A boundary solution means the optimiser converged in name only: it put
        all the variance in the residual and none in the seed grouping. Any
        inference resting on the variance component is then unreliable, and the
        fixed effects should be treated with suspicion too — observed biasing
        the accuracy coefficient from a true 0.5 to 0.74 on synthetic data
        where a large unmodelled factor drove the metric.

        Tested relative to the residual variance rather than against exact
        zero: optimisers park at ~1e-8, not 0.0, so an equality test never
        fires on a collapse that has plainly happened.
        """
        return self.seed_variance <= 1e-6 * max(self.residual_std**2, 1e-12)


def fit_coupling_model(cells: pd.DataFrame, metric_column: str) -> CouplingResult:
    """Fit the coupling model over a table of one row per cell.

    Walks `FIT_LADDER` and returns the first estimator that both fits and
    converges. The chosen estimator is recorded in `CouplingResult.estimator`
    and MUST be reported alongside any variance component it produced — a paper
    whose metrics were silently fit by different estimators is not comparable
    across its own rows.
    """
    missing = [c for c in (*REQUIRED_COLUMNS, metric_column) if c not in cells.columns]
    if missing:
        raise ValueError(f"cells table is missing required columns: {missing}")
    if cells["seed"].nunique() < 2:
        raise ValueError(
            "need at least two seeds to estimate a seed random effect; "
            "fitting with one seed would be pseudo-replication"
        )

    formula = f"{metric_column} ~ accuracy_change + C(model) + C(shift_family)"
    model = smf.mixedlm(formula, data=cells, groups=cells["seed"])

    attempts: list[str] = []
    for name, kwargs in FIT_LADDER:
        try:
            fit = model.fit(**kwargs)
        except Exception as exc:  # statsmodels raises several unrelated types
            attempts.append(f"{name}: {type(exc).__name__}")
            continue
        if not getattr(fit, "converged", True):
            attempts.append(f"{name}: did not converge")
            continue
        attempts.append(f"{name}: ok")
        break
    else:
        raise ConvergenceError(
            "every estimator in the ladder failed for "
            f"{metric_column!r} over {len(cells)} cells. Attempts: "
            + "; ".join(attempts)
            + ". Escalate rather than reinterpreting — a model that will not fit "
            "is not a null result."
        )

    if not name.startswith("reml"):
        warnings.warn(
            f"no REML estimator converged for {metric_column!r}; fell back to "
            f"{name}. ML biases variance components downward, worst with few "
            "groups — which is this study's regime at five seeds. Report the "
            f"estimator alongside any variance component. Attempts: "
            + "; ".join(attempts),
            UserWarning,
            stacklevel=2,
        )

    if float(np.asarray(fit.cov_re).ravel()[0]) <= 0.0:
        warnings.warn(
            f"seed variance collapsed to zero for {metric_column!r} (boundary "
            f"solution, estimator {name}). The fit converged in name only: all "
            "variance went to the residual and none to the seed grouping. Treat "
            "the fixed effects with suspicion, not just the variance component.",
            UserWarning,
            stacklevel=2,
        )

    ci = fit.conf_int().loc["accuracy_change"]
    return CouplingResult(
        accuracy_coefficient=float(fit.params["accuracy_change"]),
        accuracy_ci=(float(ci.iloc[0]), float(ci.iloc[1])),
        accuracy_pvalue=float(fit.pvalues["accuracy_change"]),
        residual_std=float(np.sqrt(fit.scale)),
        seed_variance=float(np.asarray(fit.cov_re).ravel()[0]),
        n_cells=len(cells),
        estimator=name,
        attempts=tuple(attempts),
        summary=str(fit.summary()),
    )
