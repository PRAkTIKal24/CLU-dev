"""Leakage-safe split utilities.

Binding rule (Hendriks et al. 2022, MSSP — the CWRU leakage lesson): split by
**physical unit / run / simulation seed**, never by window. These helpers are
the only sanctioned way to produce train/test partitions in the harness;
``evaluate_dataset`` re-checks disjointness on every run.
"""

from collections.abc import Iterable, Mapping

import numpy as np


def assert_no_unit_leakage(train_ids: Iterable[str], test_ids: Iterable[str]) -> None:
    """Raise ``ValueError`` if any unit appears in both partitions."""
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(
            f"unit leakage between train and test: {sorted(overlap)[:10]}"
            f"{' …' if len(overlap) > 10 else ''}"
        )


def unit_split(
    unit_ids: Iterable[str],
    test_fraction: float = 0.5,
    seed: int = 42,
    stratify_by: Mapping[str, str] | None = None,
) -> tuple:
    """Deterministic unit-level train/test split.

    Args:
        unit_ids: Unit identifiers (order-independent: they are sorted before
            shuffling, so the same set + seed always yields the same split).
        test_fraction: Fraction of units assigned to test (0 < f < 1).
        seed: RNG seed.
        stratify_by: Optional mapping unit_id -> group label (e.g. fault
            class); the split is then performed within each group so both
            partitions cover every group where possible.

    Returns:
        (train_ids, test_ids) as sorted tuples, disjoint by construction.
    """
    ids = sorted(set(unit_ids))
    if len(ids) < 2:
        raise ValueError(f"need >= 2 units to split, got {len(ids)}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    rng = np.random.default_rng(seed)
    if stratify_by is None:
        groups = {"__all__": ids}
    else:
        missing = [u for u in ids if u not in stratify_by]
        if missing:
            raise ValueError(f"stratify_by missing units: {missing[:5]}")
        groups = {}
        for uid in ids:
            groups.setdefault(str(stratify_by[uid]), []).append(uid)

    train, test = [], []
    for _, members in sorted(groups.items()):
        members = sorted(members)
        perm = rng.permutation(len(members))
        n_test = int(round(test_fraction * len(members)))
        # keep both sides non-empty within a group when it has >= 2 members
        if len(members) >= 2:
            n_test = min(max(n_test, 1), len(members) - 1)
        test.extend(members[i] for i in perm[:n_test])
        train.extend(members[i] for i in perm[n_test:])

    train_ids, test_ids = tuple(sorted(train)), tuple(sorted(test))
    assert_no_unit_leakage(train_ids, test_ids)
    return train_ids, test_ids


def cross_condition_split(
    unit_conditions: Mapping[str, str],
    test_conditions: Iterable[str],
) -> tuple:
    """Cross-operating-condition split (Paderborn-style generalization test).

    Train on units from all conditions *except* ``test_conditions``; test on
    units from ``test_conditions`` only.

    Args:
        unit_conditions: Mapping unit_id -> operating-condition label.
        test_conditions: Condition labels held out for testing.

    Returns:
        (train_ids, test_ids) sorted tuples, disjoint by construction.
    """
    test_conditions = {str(c) for c in test_conditions}
    known = {str(c) for c in unit_conditions.values()}
    unknown = test_conditions - known
    if unknown:
        raise ValueError(f"unknown test conditions: {sorted(unknown)}")
    if test_conditions == known:
        raise ValueError("all conditions held out — nothing left to train on")

    train = tuple(
        sorted(u for u, c in unit_conditions.items() if str(c) not in test_conditions)
    )
    test = tuple(
        sorted(u for u, c in unit_conditions.items() if str(c) in test_conditions)
    )
    assert_no_unit_leakage(train, test)
    return train, test
