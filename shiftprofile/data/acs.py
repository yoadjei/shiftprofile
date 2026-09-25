"""American Community Survey (ACS) data loaders with domain shift.

Why this module exists: The tabular track measures whether calibration, attribution
faithfulness, and subgroup performance degrade together or separately under domain
shift. Domain shift is measured as the distance between training and deployment
states of the ACS (e.g. CA 2018 to CA 2019), not as synthetic corruption.

folktables is the standard loader for this task (Ding et al. 2021). Both ACSIncome
and ACSPublicCoverage are included. Both group on RAC1P (race code), which remains
IN the feature set — this is the folktables default and the standard setup in the
literature. Dropping it would silently change the task.

Evaluation indices are deterministic and nested: eval_indices(n, 500) is a subset
of eval_indices(n, 1000). This property lets pilot numbers stay comparable with a
full run without re-generating indices.
"""

from __future__ import annotations

import numpy as np
import os
from dataclasses import dataclass
from typing import Any, Optional
from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage


# ============================================================================
# Module-level constants
# ============================================================================

ACS_TASKS: dict[str, Any] = {
    "income": ACSIncome,
    "public_coverage": ACSPublicCoverage,
}

DEFAULT_ACS_ROOT: str = os.environ.get(
    "SHIFTPROFILE_ACS_ROOT",
    "data/acs"
)


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class TabularSplit:
    """A tabular dataset split with domain and task annotations.

    Attributes:
        X: Feature matrix of shape (n, d), dtype float64.
        y: Binary labels of shape (n,), dtype int {0, 1}.
        groups: Group membership (e.g. race codes RAC1P) of shape (n,), dtype int.
        feature_names: Tuple of feature names, length d.
        domain: Domain identifier, e.g. "CA_2018".
        task: Task name, e.g. "income" or "public_coverage".
    """
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: tuple[str, ...]
    domain: str
    task: str


# ============================================================================
# Domain utilities
# ============================================================================

def domain_id(state: str, year: int) -> str:
    """Construct a domain identifier from state and year.

    Args:
        state: Two-letter state code, e.g. "CA".
        year: Survey year as integer, e.g. 2018.

    Returns:
        Domain string in format "STATE_YEAR", e.g. "CA_2018".
    """
    return f"{state}_{year}"


def parse_domain(domain: str) -> tuple[str, int]:
    """Parse a domain identifier into state and year.

    Inverse of domain_id. Raises ValueError on malformed input.

    Args:
        domain: Domain string in format "STATE_YEAR", e.g. "CA_2018".

    Returns:
        Tuple of (state, year) where state is a string and year is an int.

    Raises:
        ValueError: If domain is malformed (no underscore, invalid year, etc.).
    """
    if "_" not in domain:
        raise ValueError(f"Malformed domain {domain!r}: missing underscore")

    parts = domain.split("_")
    if len(parts) != 2:
        raise ValueError(f"Malformed domain {domain!r}: multiple underscores or wrong format")

    state, year_str = parts

    if not state:
        raise ValueError(f"Malformed domain {domain!r}: empty state")

    if not year_str:
        raise ValueError(f"Malformed domain {domain!r}: empty year")

    try:
        year = int(year_str)
    except ValueError:
        raise ValueError(f"Malformed domain {domain!r}: year is not an integer")

    return state, year


# ============================================================================
# Evaluation indices
# ============================================================================

def eval_indices(n_rows: int, n_eval: int, *, seed: int = 0) -> np.ndarray:
    """Generate deterministic evaluation indices for a domain.

    The same n_eval and seed always produce identical results, ensuring
    reproducibility across machines and sessions. Indices for smaller n_eval
    are prefixes of indices for larger n_eval, so subsets remain comparable
    (the nesting property). This allows pilot numbers to stay directly comparable
    with full runs without regenerating indices.

    Args:
        n_rows: Total number of rows in the domain.
        n_eval: Number of evaluation indices to generate.
        seed: Random seed for reproducibility (default 0).

    Returns:
        Array of n_eval unique indices in [0, n_rows). The indices are ordered
        deterministically such that smaller n_eval calls return prefixes of
        larger calls, preserving the nesting property.

    Raises:
        ValueError: If n_eval > n_rows.
    """
    if n_eval > n_rows:
        raise ValueError(f"n_eval ({n_eval}) cannot exceed n_rows ({n_rows})")

    rng = np.random.RandomState(seed)
    # Generate a permutation of all indices, take the first n_eval.
    # The nesting property holds because the first n_eval elements of
    # permutation(seed) are always a prefix of the first m elements for m > n_eval
    indices = rng.permutation(n_rows)[:n_eval]
    return indices


def train_indices(
    n_rows: int,
    n_eval: int,
    n_train: Optional[int] = None,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Rows reserved for fitting: the complement of `eval_indices`.

    Training and evaluation must not share rows, and on this track that is not
    the usual hygiene argument. The source domain is evaluated as the *clean*
    cell — the baseline every shifted cell is compared against. Fit on the
    evaluation rows and the clean cell becomes an in-sample estimate while every
    shifted cell stays out-of-sample, so the comparison would show a drop from
    clean to shifted even if the model were perfectly robust. That is precisely
    the effect this study exists to measure, so an overlapping split would not
    merely inflate a number; it would manufacture the finding.

    Uses the same permutation as `eval_indices` at the same seed, so the two
    sets are exact complements by construction rather than by coincidence.

    Args:
        n_rows: Total number of rows in the domain.
        n_eval: Number of evaluation rows, held out from training.
        n_train: Cap on training rows. None means use every remaining row.
        seed: Random seed; must match the seed used for `eval_indices`.

    Returns:
        Array of indices in [0, n_rows), disjoint from `eval_indices(n_rows,
        n_eval, seed=seed)`.

    Raises:
        ValueError: If n_eval > n_rows, or if no rows remain for training.
    """
    if n_eval > n_rows:
        raise ValueError(f"n_eval ({n_eval}) cannot exceed n_rows ({n_rows})")

    rng = np.random.RandomState(seed)
    remaining = rng.permutation(n_rows)[n_eval:]

    if remaining.size == 0:
        raise ValueError(
            f"no rows left to train on: n_rows={n_rows} and n_eval={n_eval} "
            f"consume the whole domain. Lower n_eval or use a larger domain."
        )

    if n_train is not None:
        remaining = remaining[:n_train]

    return remaining


# ============================================================================
# Data loading
# ============================================================================

def load_domain(
    task: str,
    domain: str,
    *,
    root: Optional[str] = None,
    download: bool = True,
) -> TabularSplit:
    """Load a full domain without evaluation-set restriction.

    Used to compute in-distribution metrics (calibration, etc.) on the full domain
    before pilot gating.

    Args:
        task: Task name, e.g. "income" or "public_coverage".
        domain: Domain string, e.g. "CA_2018".
        root: Root directory for ACS data (default: DEFAULT_ACS_ROOT).
        download: Whether to download if not present (default True).

    Returns:
        TabularSplit with full domain data.
    """
    if root is None:
        root = DEFAULT_ACS_ROOT

    state, year = parse_domain(domain)
    task_class = ACS_TASKS[task]

    # Load data via folktables
    src = ACSDataSource(
        survey_year=str(year),
        horizon="1-Year",
        survey="person",
        root_dir=root
    )
    df = src.get_data(states=[state], download=download)

    # Extract features, labels, and groups via task-specific extractor
    X, y, groups = task_class.df_to_numpy(df)

    # Ensure correct dtypes
    X = X.astype(np.float64)
    y = y.astype(int)
    groups = groups.astype(int)

    # Get feature names from the task
    # folktables tasks define feature_names attribute
    feature_names = task_class.features

    return TabularSplit(
        X=X,
        y=y,
        groups=groups,
        feature_names=tuple(feature_names),
        domain=domain,
        task=task,
    )


def load_cell_table(
    task: str,
    domain: str,
    n_eval: int,
    *,
    root: Optional[str] = None,
    download: bool = True,
) -> TabularSplit:
    """Load a domain restricted to evaluation indices.

    This is the single entry point for filling the tabular grid, matching the
    pattern from load_cell_images in the vision track. A producer cannot
    accidentally load the wrong subset.

    Args:
        task: Task name, e.g. "income" or "public_coverage".
        domain: Domain string, e.g. "CA_2018".
        n_eval: Number of evaluation rows to select.
        root: Root directory for ACS data (default: DEFAULT_ACS_ROOT).
        download: Whether to download if not present (default True).

    Returns:
        TabularSplit with n_eval rows selected via eval_indices.
    """
    # First load the full domain to get metadata
    full_split = load_domain(task, domain, root=root, download=download)

    # Generate evaluation indices for this domain size
    indices = eval_indices(full_split.X.shape[0], n_eval, seed=0)

    # Restrict all arrays to evaluation set
    return TabularSplit(
        X=full_split.X[indices],
        y=full_split.y[indices],
        groups=full_split.groups[indices],
        feature_names=full_split.feature_names,
        domain=domain,
        task=task,
    )


def load_train_table(
    task: str,
    domain: str,
    n_eval: int,
    n_train: Optional[int] = None,
    *,
    root: Optional[str] = None,
    download: bool = True,
) -> TabularSplit:
    """Load a domain restricted to the training rows.

    The counterpart to `load_cell_table`. Anything fitted must come from here
    and anything scored must come from there; the two never overlap. See
    `train_indices` for why that separation is load-bearing on this track
    rather than routine.

    Args:
        task: Task name, e.g. "income" or "public_coverage".
        domain: Domain string, e.g. "CA_2018".
        n_eval: Number of evaluation rows to withhold.
        n_train: Cap on training rows. None means every remaining row.
        root: Root directory for ACS data (default: DEFAULT_ACS_ROOT).
        download: Whether to download if not present (default True).

    Returns:
        TabularSplit holding rows disjoint from `load_cell_table`'s.
    """
    full_split = load_domain(task, domain, root=root, download=download)

    indices = train_indices(full_split.X.shape[0], n_eval, n_train, seed=0)

    return TabularSplit(
        X=full_split.X[indices],
        y=full_split.y[indices],
        groups=full_split.groups[indices],
        feature_names=full_split.feature_names,
        domain=domain,
        task=task,
    )
