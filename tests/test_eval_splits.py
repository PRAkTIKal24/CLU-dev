"""Tests for leakage-safe split utilities (binding rule: unit-level splits)."""

import pytest

from chlu.eval.splits import (
    assert_no_unit_leakage,
    cross_condition_split,
    unit_split,
)

UNITS = [f"bearing{i:02d}" for i in range(10)]


def test_unit_split_disjoint_and_complete():
    train, test = unit_split(UNITS, test_fraction=0.3, seed=0)
    assert set(train) | set(test) == set(UNITS)
    assert not set(train) & set(test)
    assert len(test) == 3


def test_unit_split_deterministic_and_order_independent():
    a = unit_split(UNITS, test_fraction=0.5, seed=7)
    b = unit_split(list(reversed(UNITS)), test_fraction=0.5, seed=7)
    assert a == b
    c = unit_split(UNITS, test_fraction=0.5, seed=8)
    assert a != c  # different seed, different split (overwhelmingly likely)


def test_unit_split_stratified_covers_every_group():
    groups = {u: ("A" if i < 6 else "B") for i, u in enumerate(UNITS)}
    train, test = unit_split(UNITS, test_fraction=0.4, seed=1, stratify_by=groups)
    for side in (train, test):
        assert {"A", "B"} == {groups[u] for u in side}


def test_unit_split_rejects_bad_inputs():
    with pytest.raises(ValueError):
        unit_split(["only_one"], test_fraction=0.5)
    with pytest.raises(ValueError):
        unit_split(UNITS, test_fraction=1.5)
    with pytest.raises(ValueError):
        unit_split(UNITS, stratify_by={u: "A" for u in UNITS[:-1]})


def test_leakage_assert_raises_on_overlap():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_unit_leakage(["u1", "u2"], ["u2", "u3"])
    assert_no_unit_leakage(["u1"], ["u2"])  # disjoint passes


def test_cross_condition_split():
    conditions = {"u1": "1500rpm", "u2": "1500rpm", "u3": "900rpm", "u4": "900rpm"}
    train, test = cross_condition_split(conditions, test_conditions=["900rpm"])
    assert train == ("u1", "u2")
    assert test == ("u3", "u4")


def test_cross_condition_split_rejects_bad_conditions():
    conditions = {"u1": "a", "u2": "b"}
    with pytest.raises(ValueError, match="unknown"):
        cross_condition_split(conditions, test_conditions=["c"])
    with pytest.raises(ValueError, match="nothing left"):
        cross_condition_split(conditions, test_conditions=["a", "b"])
