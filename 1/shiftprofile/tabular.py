"""Tabular track: the cached train and predict stages.

The counterpart of `predict.py` for the tabular grid. A cell here is one
(model, seed, target domain) combination; the model is fitted once on the
source domain and scored against every target domain in turn.

Three invariants hold this module together:

1. **Fitting and scoring never share a row.** Training draws from
   `load_train_table`, scoring from `load_cell_table`, and those two are exact
   complements. The source domain doubles as the clean cell — the baseline every
   shifted cell is measured against — so an overlap would make the baseline
   in-sample while its comparators stay out-of-sample, producing a clean-to-
   shifted drop out of nothing at all.

2. **Training is keyed without the target domain.** One fitted model serves all
   nine domains; keying the fit by target would refit identical weights nine
   times over.

3. **Every cache key is built by `stage_spec`.** Never by hand. A producer in
   this codebase once wrote under one key while its consumer read another, so
   the cache could not hit and resumability had quietly stopped working.

The fitted estimator is held in a process-local memo rather than serialised
into the artifact cache. Pickling an estimator into a cache that round-trips
through a shared, versioned dataset buys very little and costs a lot: the
pickle is only loadable by the exact scikit-learn build that wrote it, and
`producer_version` does not change when scikit-learn is upgraded, so a stale
entry would deserialise into an error or, worse, into something subtly
different. Predictions are what resumability actually needs, and those are
cached as plain arrays. The most a lost memo costs is one refit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from .cache import ArtifactCache
from .cells import CLEAN, Cell, stage_spec
from .data.acs import load_cell_table, load_train_table
from .models.tabular import fit_tabular, predict_proba_tabular

TABULAR_TRAIN_VERSION = "tabular-train-v1"
TABULAR_PREDICT_VERSION = "tabular-predict-v1"

# Fitted estimators for the lifetime of this process, keyed by training identity.
# Deliberately not persisted; see the module docstring.
_MODEL_MEMO: dict[tuple, Any] = {}


def clear_model_memo() -> None:
    """Drop every memoised estimator. For tests and long-running sessions."""
    _MODEL_MEMO.clear()


def source_domain(config: dict[str, Any]) -> str:
    """The domain every model is fitted on.

    Required in the config rather than defaulted. A silent default here would
    be a fitted-on-the-wrong-data bug that still produces plausible numbers.
    """
    try:
        return config["source"]
    except KeyError:
        raise KeyError(
            "config is missing 'source', the domain models are fitted on "
            "(e.g. 'CA_2018'). It has no default: guessing it wrong would "
            "train on one population and report on another."
        ) from None


def target_domain_of(cell: Cell, source: str) -> str:
    """The domain this cell is scored on.

    A clean cell is scored on the source domain itself; a shift cell carries
    its target domain in `shift_family`.
    """
    if cell.shift_family == CLEAN:
        return source
    return cell.shift_family


def train_spec(cell: Cell, task: str, source: str) -> dict[str, Any]:
    """Cache spec for the fit. THE single source for training keys.

    The cell is normalised to its clean form before the key is built, so every
    target domain of a given (model, seed) maps to one key. That is what makes
    "fit once, score nine times" true of the cache and not just of the
    docstring.
    """
    fit_cell = replace(cell, shift_family=CLEAN, severity=0)
    return stage_spec(fit_cell, "tabular_train", task=task, source=source)


def predict_spec(cell: Cell, task: str, source: str, target: str) -> dict[str, Any]:
    """Cache spec for one cell's predictions. THE single source for those keys.

    Keyed per target domain, so two cells that share a fitted model still cache
    their predictions separately.
    """
    return stage_spec(
        cell, "tabular_predict", task=task, source=source, target=target
    )


def train_tabular_cell(
    cell: Cell,
    config: dict[str, Any],
    cache: ArtifactCache,
    *,
    root: Optional[str] = None,
) -> Any:
    """Fit this cell's model on the source domain's training rows.

    Returns the memoised estimator when one is already fitted for this
    (track, model_id, seed, task, source). The cache receives a provenance
    record — what was fitted, on how many rows — but never the estimator
    itself.
    """
    task = config["task"]
    source = source_domain(config)
    spec = train_spec(cell, task, source)

    memo_key = (cell.track, cell.model_id, cell.seed, task, source)
    if memo_key in _MODEL_MEMO:
        return _MODEL_MEMO[memo_key]

    data = load_train_table(
        task,
        source,
        config["n_eval"],
        config.get("n_train"),
        root=root,
        download=True,
    )

    model = fit_tabular(cell.model_id, data.X, data.y, seed=cell.seed)

    cache.put_record(
        spec,
        TABULAR_TRAIN_VERSION,
        {
            "model_id": cell.model_id,
            "seed": cell.seed,
            "task": task,
            "source": source,
            "n_train_rows": int(data.X.shape[0]),
            "n_features": int(data.X.shape[1]),
        },
    )

    _MODEL_MEMO[memo_key] = model
    return model


def predict_tabular_cell(
    cell: Cell,
    config: dict[str, Any],
    cache: ArtifactCache,
    *,
    root: Optional[str] = None,
) -> np.ndarray:
    """Class probabilities for one cell, checking the cache first.

    Returns an (n_eval, 2) float64 array. This is the artifact resumability
    depends on: once it is cached, the fitted model is never needed again.
    """
    task = config["task"]
    source = source_domain(config)
    target = target_domain_of(cell, source)
    spec = predict_spec(cell, task, source, target)

    if cache.has(spec, TABULAR_PREDICT_VERSION, kind="array"):
        return cache.get_array(spec, TABULAR_PREDICT_VERSION)

    model = train_tabular_cell(cell, config, cache, root=root)

    eval_data = load_cell_table(
        task, target, config["n_eval"], root=root, download=True
    )

    proba = np.asarray(predict_proba_tabular(model, eval_data.X), dtype=np.float64)
    cache.put_array(spec, TABULAR_PREDICT_VERSION, proba)
    return proba
