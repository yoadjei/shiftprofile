import numpy as np
import pytest
from shiftprofile.metrics.subgroup import (
    worst_group_accuracy,
    equal_opportunity_gap,
    per_group_ece_gap,
    group_sizes,
)

PROBS = np.array(
    [
        [0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.7, 0.3],
        [0.4, 0.6], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4],
    ]
)
LABELS = np.array([1, 1, 0, 0, 1, 1, 0, 1])
GROUPS = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])


def test_group_sizes_counts_each_group():
    assert group_sizes(GROUPS) == {"a": 4, "b": 4}


def test_worst_group_accuracy_takes_the_minimum():
    # group a: preds 1,1,0,0 vs labels 1,1,0,0 -> 4/4 = 1.00
    # group b: preds 1,1,0,0 vs labels 1,1,0,1 -> 3/4 = 0.75
    assert worst_group_accuracy(
        PROBS, LABELS, GROUPS, min_group_size=1
    ) == pytest.approx(0.75)


def test_equal_opportunity_gap_compares_tpr():
    # group a, y=1: samples 0,1 -> both predicted 1 -> TPR 1.00
    # group b, y=1: samples 4,5,7 -> predicted 1,1,0 -> TPR 2/3
    assert equal_opportunity_gap(
        PROBS, LABELS, GROUPS, min_group_size=1
    ) == pytest.approx(1 / 3)


def test_per_group_ece_gap_is_non_negative():
    assert per_group_ece_gap(PROBS, LABELS, GROUPS, n_bins=2, min_group_size=1) >= 0.0


def test_small_groups_are_excluded_and_reported():
    groups = np.array(["a"] * 7 + ["tiny"])
    with pytest.raises(ValueError, match="min_group_size"):
        worst_group_accuracy(PROBS, LABELS, groups, min_group_size=200)


def test_min_group_size_defaults_to_the_preregistered_threshold():
    """The protocol fixes 200 test rows as the floor for a reported subgroup,
    so the default must be that floor and not something permissive."""
    with pytest.raises(ValueError, match="200"):
        worst_group_accuracy(PROBS, LABELS, GROUPS)


def test_single_group_cannot_have_a_gap():
    groups = np.array(["a"] * 8)
    with pytest.raises(ValueError, match="two groups"):
        equal_opportunity_gap(PROBS, LABELS, groups, min_group_size=1)
