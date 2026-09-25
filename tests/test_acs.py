"""Tests for ACS (American Community Survey) data loaders with domain shift.

Tests are comprehensive and deterministic. No test downloads data or accesses the network.
All folktables calls are mocked with synthetic DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shiftprofile.data.acs import (
    ACS_TASKS,
    DEFAULT_ACS_ROOT,
    TabularSplit,
    domain_id,
    parse_domain,
    eval_indices,
    train_indices,
    load_domain,
    load_cell_table,
    load_train_table,
)


# ============================================================================
# Test domain_id and parse_domain round-trip
# ============================================================================


def test_domain_id_valid():
    """Test domain_id constructs correct domain strings."""
    assert domain_id("CA", 2018) == "CA_2018"
    assert domain_id("TX", 2019) == "TX_2019"
    assert domain_id("NY", 2017) == "NY_2017"


def test_domain_id_is_string():
    """Test that domain_id always returns a string."""
    result = domain_id("CA", 2018)
    assert isinstance(result, str)


def test_parse_domain_valid():
    """Test parse_domain extracts state and year correctly."""
    state, year = parse_domain("CA_2018")
    assert state == "CA"
    assert year == 2018
    assert isinstance(year, int)


def test_parse_domain_roundtrip():
    """Test that domain_id and parse_domain are inverses."""
    original_state = "TX"
    original_year = 2019
    domain_str = domain_id(original_state, original_year)
    parsed_state, parsed_year = parse_domain(domain_str)
    assert parsed_state == original_state
    assert parsed_year == original_year


def test_parse_domain_malformed_no_underscore():
    """Test that malformed domains without underscore raise ValueError."""
    with pytest.raises(ValueError):
        parse_domain("CA2018")


def test_parse_domain_malformed_no_year():
    """Test that malformed domains without year raise ValueError."""
    with pytest.raises(ValueError):
        parse_domain("CA_")


def test_parse_domain_malformed_non_int_year():
    """Test that domains with non-integer year raise ValueError."""
    with pytest.raises(ValueError):
        parse_domain("CA_abcd")


def test_parse_domain_malformed_empty_state():
    """Test that domains with empty state raise ValueError."""
    with pytest.raises(ValueError):
        parse_domain("_2018")


def test_parse_domain_malformed_multiple_underscores():
    """Test that parse_domain handles strings with multiple underscores correctly."""
    # Only the first underscore should be split point; this is a detail
    # but let's verify it handles the most common malformedness
    with pytest.raises(ValueError):
        parse_domain("CA_2018_extra")


# ============================================================================
# Test eval_indices determinism, sortedness, and nesting
# ============================================================================


def test_eval_indices_deterministic():
    """Running eval_indices twice with the same args must give identical results."""
    a = eval_indices(5000, 500, seed=0)
    b = eval_indices(5000, 500, seed=0)
    np.testing.assert_array_equal(a, b)


def test_eval_indices_default_seed():
    """eval_indices with default seed must be deterministic across calls."""
    a = eval_indices(5000, 100)
    b = eval_indices(5000, 100)
    np.testing.assert_array_equal(a, b)


def test_eval_indices_deterministically_ordered():
    """Returned indices are ordered deterministically by seed, not by value."""
    # The indices are not necessarily in sorted order, but they are deterministic
    indices1 = eval_indices(10000, 200, seed=0)
    indices2 = eval_indices(10000, 200, seed=0)
    # Same seed must produce identical order
    np.testing.assert_array_equal(indices1, indices2)


def test_eval_indices_unique():
    """Returned indices must all be unique."""
    indices = eval_indices(5000, 500, seed=0)
    assert len(indices) == len(np.unique(indices))


def test_eval_indices_in_bounds():
    """Returned indices must be in [0, n_rows)."""
    n_rows = 3000
    indices = eval_indices(n_rows, 500, seed=0)
    assert indices.min() >= 0
    assert indices.max() < n_rows


def test_eval_indices_nested_small_in_large():
    """eval_indices(n_rows, 500) must be a prefix of eval_indices(n_rows, 1000).

    This ensures pilot results remain comparable with full runs.
    """
    n_rows = 10000
    small = eval_indices(n_rows, 500, seed=0)
    large = eval_indices(n_rows, 1000, seed=0)
    np.testing.assert_array_equal(small, large[:500])


def test_eval_indices_nested_across_multiple_sizes():
    """Nesting must hold transitively: 100 ⊂ 200 ⊂ 400."""
    n_rows = 5000
    idx_100 = eval_indices(n_rows, 100, seed=0)
    idx_200 = eval_indices(n_rows, 200, seed=0)
    idx_400 = eval_indices(n_rows, 400, seed=0)

    np.testing.assert_array_equal(idx_100, idx_200[:100])
    np.testing.assert_array_equal(idx_200, idx_400[:200])
    np.testing.assert_array_equal(idx_100, idx_400[:100])


def test_eval_indices_different_seeds_produce_different_results():
    """Different seeds must produce different index sets."""
    indices_seed0 = eval_indices(5000, 100, seed=0)
    indices_seed1 = eval_indices(5000, 100, seed=1)
    # They should differ (with extremely high probability)
    assert not np.array_equal(indices_seed0, indices_seed1)


def test_eval_indices_n_eval_exceeds_n_rows():
    """eval_indices must raise ValueError if n_eval > n_rows."""
    with pytest.raises(ValueError):
        eval_indices(500, 1000, seed=0)


def test_eval_indices_returns_ndarray():
    """eval_indices must return numpy array."""
    result = eval_indices(1000, 100, seed=0)
    assert isinstance(result, np.ndarray)


def test_eval_indices_returns_integers():
    """eval_indices must return integer dtype."""
    result = eval_indices(1000, 100, seed=0)
    assert result.dtype in [np.int32, np.int64]


# ============================================================================
# Test TabularSplit dataclass
# ============================================================================


def test_tabular_split_creation():
    """TabularSplit must accept all required fields."""
    X = np.random.randn(100, 10)
    y = np.array([0, 1] * 50)
    groups = np.array([1] * 100)
    feature_names = tuple(f"feat_{i}" for i in range(10))

    split = TabularSplit(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        domain="CA_2018",
        task="income"
    )

    assert split.X is X
    assert split.y is y
    assert split.groups is groups
    assert split.feature_names == feature_names
    assert split.domain == "CA_2018"
    assert split.task == "income"


def test_tabular_split_is_frozen():
    """TabularSplit should be frozen (immutable)."""
    X = np.random.randn(10, 5)
    split = TabularSplit(
        X=X, y=np.array([0, 1] * 5), groups=np.array([1] * 10),
        feature_names=("a", "b", "c", "d", "e"),
        domain="CA_2018", task="income"
    )
    with pytest.raises((AttributeError, Exception)):
        split.domain = "TX_2019"


def test_tabular_split_expected_dtypes():
    """TabularSplit fields should have expected dtypes."""
    X = np.random.randn(10, 5).astype(np.float64)
    y = np.array([0, 1] * 5)
    groups = np.array([1] * 10)

    split = TabularSplit(
        X=X, y=y, groups=groups,
        feature_names=("a", "b", "c", "d", "e"),
        domain="CA_2018", task="income"
    )

    assert split.X.dtype == np.float64
    assert split.y.dtype in [np.int32, np.int64]
    assert split.groups.dtype in [np.int32, np.int64]


# ============================================================================
# Test ACS_TASKS constant
# ============================================================================


def test_acs_tasks_is_dict():
    """ACS_TASKS must be a dict."""
    assert isinstance(ACS_TASKS, dict)


def test_acs_tasks_has_required_keys():
    """ACS_TASKS must contain 'income' and 'public_coverage'."""
    assert "income" in ACS_TASKS
    assert "public_coverage" in ACS_TASKS


def test_acs_tasks_values_have_df_to_numpy():
    """ACS task classes must have a df_to_numpy method."""
    # Check that the task class objects have the expected interface
    # (they should be folktables task classes)
    for task_name, task_class in ACS_TASKS.items():
        assert hasattr(task_class, "df_to_numpy"), \
            f"ACS_TASKS[{task_name!r}] missing df_to_numpy method"


# ============================================================================
# Test DEFAULT_ACS_ROOT
# ============================================================================


def test_default_acs_root_is_string():
    """DEFAULT_ACS_ROOT must be a string."""
    assert isinstance(DEFAULT_ACS_ROOT, str)


def test_default_acs_root_not_empty():
    """DEFAULT_ACS_ROOT must not be empty."""
    assert len(DEFAULT_ACS_ROOT) > 0


# ============================================================================
# Test load_cell_table with synthetic data
# ============================================================================


def create_synthetic_acs_dataframe(n_rows: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic ACS DataFrame for testing.

    Mimics folktables ACS data structure with the required columns.
    """
    rng = np.random.RandomState(seed)

    # Minimal columns needed for ACSIncome/ACSPublicCoverage
    # RAC1P is the race code (1-9), stays in the feature set
    # PINCP is income, PUBCOV is public coverage status
    df = pd.DataFrame({
        "RAC1P": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], n_rows),
        "PINCP": rng.randint(0, 200000, n_rows),
        "PUBCOV": rng.randint(0, 2, n_rows),
        "AGE": rng.randint(18, 85, n_rows),
        "SCHL": rng.randint(1, 25, n_rows),
        "MAR": rng.choice([1, 2, 3, 4, 5], n_rows),
        "SEX": rng.choice([1, 2], n_rows),
        "WKHP": rng.randint(0, 99, n_rows),
    })
    return df


def test_load_cell_table_restricts_rows(monkeypatch, tmp_path):
    """load_cell_table must restrict rows using eval_indices."""
    # Create synthetic data
    full_df = create_synthetic_acs_dataframe(n_rows=1000, seed=0)

    # Mock the folktables ACSDataSource
    class MockACSDataSource:
        def __init__(self, survey_year="2018", horizon="1-Year", survey="person", root_dir=None):
            self.survey_year = survey_year
            self.root_dir = root_dir

        def get_data(self, states, download=True):
            return full_df

    def mock_acs_datasource(survey_year, horizon, survey, root_dir):
        return MockACSDataSource(survey_year, horizon, survey, root_dir)

    monkeypatch.setattr("shiftprofile.data.acs.ACSDataSource", mock_acs_datasource)

    # Mock the income task
    class MockACSIncome:
        features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

        @staticmethod
        def df_to_numpy(df):
            X = df.drop(columns=["PINCP", "PUBCOV"]).values.astype(np.float64)
            y = (df["PINCP"] > 50000).astype(int).values
            groups = df["RAC1P"].values
            return X, y, groups

    monkeypatch.setitem(ACS_TASKS, "income", MockACSIncome)

    # Test load_cell_table restricts to eval_indices
    n_eval = 200
    split = load_cell_table("income", "CA_2018", n_eval, root=str(tmp_path), download=False)

    # Check that we got exactly n_eval rows
    assert split.X.shape[0] == n_eval
    assert split.y.shape[0] == n_eval
    assert split.groups.shape[0] == n_eval


def test_load_cell_table_returns_tabular_split(monkeypatch, tmp_path):
    """load_cell_table must return a TabularSplit."""
    full_df = create_synthetic_acs_dataframe(n_rows=500, seed=0)

    class MockACSDataSource:
        def __init__(self, survey_year="2018", horizon="1-Year", survey="person", root_dir=None):
            pass

        def get_data(self, states, download=True):
            return full_df

    def mock_acs_datasource(survey_year, horizon, survey, root_dir):
        return MockACSDataSource()

    monkeypatch.setattr("shiftprofile.data.acs.ACSDataSource", mock_acs_datasource)

    class MockACSIncome:
        features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

        @staticmethod
        def df_to_numpy(df):
            X = df.drop(columns=["PINCP", "PUBCOV"]).values.astype(np.float64)
            y = (df["PINCP"] > 50000).astype(int).values
            groups = df["RAC1P"].values
            return X, y, groups

    monkeypatch.setitem(ACS_TASKS, "income", MockACSIncome)

    split = load_cell_table("income", "CA_2018", 100, root=str(tmp_path), download=False)

    assert isinstance(split, TabularSplit)
    assert split.task == "income"
    assert split.domain == "CA_2018"


def test_load_cell_table_preserves_shapes(monkeypatch, tmp_path):
    """load_cell_table must have aligned X, y, groups shapes."""
    full_df = create_synthetic_acs_dataframe(n_rows=500, seed=42)

    class MockACSDataSource:
        def get_data(self, states, download=True):
            return full_df

    def mock_acs_datasource(survey_year, horizon, survey, root_dir):
        return MockACSDataSource()

    monkeypatch.setattr("shiftprofile.data.acs.ACSDataSource", mock_acs_datasource)

    class MockACSIncome:
        features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

        @staticmethod
        def df_to_numpy(df):
            X = df.drop(columns=["PINCP", "PUBCOV"]).values.astype(np.float64)
            y = (df["PINCP"] > 50000).astype(int).values
            groups = df["RAC1P"].values
            return X, y, groups

    monkeypatch.setitem(ACS_TASKS, "income", MockACSIncome)

    split = load_cell_table("income", "CA_2018", 100, root=str(tmp_path), download=False)

    assert split.X.shape[0] == split.y.shape[0] == split.groups.shape[0] == 100


# ============================================================================
# Test load_domain
# ============================================================================


def test_load_domain_returns_tabular_split(monkeypatch, tmp_path):
    """load_domain must return a TabularSplit."""
    full_df = create_synthetic_acs_dataframe(n_rows=500, seed=0)

    class MockACSDataSource:
        def get_data(self, states, download=True):
            return full_df

    def mock_acs_datasource(survey_year, horizon, survey, root_dir):
        return MockACSDataSource()

    monkeypatch.setattr("shiftprofile.data.acs.ACSDataSource", mock_acs_datasource)

    class MockACSIncome:
        features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

        @staticmethod
        def df_to_numpy(df):
            X = df.drop(columns=["PINCP", "PUBCOV"]).values.astype(np.float64)
            y = (df["PINCP"] > 50000).astype(int).values
            groups = df["RAC1P"].values
            return X, y, groups

    monkeypatch.setitem(ACS_TASKS, "income", MockACSIncome)

    split = load_domain("income", "CA_2018", root=str(tmp_path), download=False)

    assert isinstance(split, TabularSplit)


def test_load_domain_uses_full_rows(monkeypatch, tmp_path):
    """load_domain must load the full domain without restricting via eval_indices."""
    full_df = create_synthetic_acs_dataframe(n_rows=500, seed=0)

    class MockACSDataSource:
        def get_data(self, states, download=True):
            return full_df

    def mock_acs_datasource(survey_year, horizon, survey, root_dir):
        return MockACSDataSource()

    monkeypatch.setattr("shiftprofile.data.acs.ACSDataSource", mock_acs_datasource)

    class MockACSIncome:
        features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

        @staticmethod
        def df_to_numpy(df):
            X = df.drop(columns=["PINCP", "PUBCOV"]).values.astype(np.float64)
            y = (df["PINCP"] > 50000).astype(int).values
            groups = df["RAC1P"].values
            return X, y, groups

    monkeypatch.setitem(ACS_TASKS, "income", MockACSIncome)

    split = load_domain("income", "CA_2018", root=str(tmp_path), download=False)

    # Should have all 500 rows, not restricted
    assert split.X.shape[0] == 500


def test_load_domain_public_coverage_task(monkeypatch, tmp_path):
    """load_domain must work with public_coverage task."""
    full_df = create_synthetic_acs_dataframe(n_rows=300, seed=1)

    class MockACSDataSource:
        def get_data(self, states, download=True):
            return full_df

    def mock_acs_datasource(survey_year, horizon, survey, root_dir):
        return MockACSDataSource()

    monkeypatch.setattr("shiftprofile.data.acs.ACSDataSource", mock_acs_datasource)

    class MockACSPublicCoverage:
        features = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']

        @staticmethod
        def df_to_numpy(df):
            X = df.drop(columns=["PINCP", "PUBCOV"]).values.astype(np.float64)
            y = df["PUBCOV"].astype(int).values
            groups = df["RAC1P"].values
            return X, y, groups

    monkeypatch.setitem(ACS_TASKS, "public_coverage", MockACSPublicCoverage)

    split = load_domain("public_coverage", "TX_2019", root=str(tmp_path), download=False)

    assert isinstance(split, TabularSplit)
    assert split.task == "public_coverage"
    assert split.domain == "TX_2019"
    assert split.X.shape[0] == 300


# ============================================================================
# Train/eval separation
#
# These are the regression guard for a contamination bug that was live in this
# module: training drew its rows from load_cell_table, the SAME rows the clean
# cell is scored on. Because the source domain doubles as the clean baseline,
# that made the baseline an in-sample estimate while every shifted cell stayed
# out-of-sample, so a clean-to-shifted drop would appear even for a perfectly
# robust model. That drop is the study's headline quantity, so the bug would
# not have inflated a number — it would have manufactured the finding.
# ============================================================================


def test_train_indices_disjoint_from_eval_indices():
    """No row may be both trained on and scored."""
    n_rows, n_eval = 1000, 200
    ev = set(eval_indices(n_rows, n_eval).tolist())
    tr = set(train_indices(n_rows, n_eval).tolist())

    assert len(ev) == n_eval
    assert ev & tr == set(), f"{len(ev & tr)} rows appear in both splits"


def test_train_and_eval_indices_partition_the_domain():
    """With no cap, the two splits cover every row exactly once."""
    n_rows, n_eval = 500, 120
    ev = eval_indices(n_rows, n_eval)
    tr = train_indices(n_rows, n_eval)

    assert len(tr) == n_rows - n_eval
    assert sorted(np.concatenate([ev, tr]).tolist()) == list(range(n_rows))


def test_train_indices_respects_the_cap():
    n_rows, n_eval, n_train = 1000, 100, 50
    tr = train_indices(n_rows, n_eval, n_train)

    assert len(tr) == n_train
    assert set(tr.tolist()) & set(eval_indices(n_rows, n_eval).tolist()) == set()


def test_train_indices_is_deterministic():
    a = train_indices(800, 100, 200)
    b = train_indices(800, 100, 200)
    assert np.array_equal(a, b)


def test_train_indices_raises_when_eval_consumes_the_domain():
    with pytest.raises(ValueError, match="no rows left to train on"):
        train_indices(100, 100)


def test_train_indices_raises_when_n_eval_exceeds_n_rows():
    with pytest.raises(ValueError, match="cannot exceed"):
        train_indices(100, 101)


def test_load_train_table_rows_are_absent_from_load_cell_table(monkeypatch):
    """End to end: the two loaders never hand back the same row.

    Asserted on the feature values rather than on indices, because it is the
    values that reach the estimator. An index-only check would still pass if
    the two loaders sliced different index sets out of different frames.
    """
    full_df = create_synthetic_acs_dataframe(n_rows=600, seed=3)

    class MockACSDataSource:
        def __init__(self, *args, **kwargs):
            pass

        def get_data(self, states, download=True):
            return full_df

    class MockACSIncome:
        features = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]

        @staticmethod
        def df_to_numpy(df):
            X = df.drop(columns=["PINCP", "PUBCOV"]).values.astype(np.float64)
            y = (df["PINCP"] > 50000).astype(int).values
            groups = df["RAC1P"].values
            return X, y, groups

    monkeypatch.setattr(
        "shiftprofile.data.acs.ACSDataSource",
        lambda survey_year, horizon, survey, root_dir: MockACSDataSource(),
    )
    monkeypatch.setattr("shiftprofile.data.acs.ACS_TASKS", {"income": MockACSIncome})

    ev = load_cell_table("income", "CA_2018", 100)
    tr = load_train_table("income", "CA_2018", 100, 200)

    assert ev.X.shape[0] == 100
    assert tr.X.shape[0] == 200

    eval_rows = {tuple(r) for r in ev.X.tolist()}
    train_rows = {tuple(r) for r in tr.X.tolist()}
    assert eval_rows & train_rows == set(), (
        "a row reached both the fit and the score. On this track that is not a "
        "hygiene problem: the source domain is the clean baseline, so overlap "
        "makes the baseline in-sample and fabricates a clean-to-shifted drop."
    )
