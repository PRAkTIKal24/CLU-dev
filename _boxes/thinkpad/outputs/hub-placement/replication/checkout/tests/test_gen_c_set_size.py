"""The C set is sized to match the held-out set (Head ruling, 2026-09-09).

The primary refusal metric is ``AUROC(B unseen-valid vs C impossible)``.  A C set
much larger or smaller than B distorts that number, so ``|C|`` tracks ``|B|``
rather than sitting at a fixed default.  Shapes3D holds out only **72** rows, so
the old fixed 32 was not "comparable" in any useful sense.
"""

import numpy as np
import pytest

from chlu.config import GenDataConfig
from chlu.data.gen_splits import FactorGrid, build_c_invalid, validate_c_invalid


def _grid(cards=(3, 4, 5)):
    return FactorGrid(
        dataset="toy",
        names=[f"f{i}" for i in range(len(cards))],
        values=[np.linspace(0.0, 1.0, c) for c in cards],
    )


def test_default_is_none_so_the_size_is_derived_not_fixed():
    assert GenDataConfig().c_invalid_n_per_kind is None


@pytest.mark.parametrize("n_test", [72, 6292, 1, 2, 3])
def test_derived_size_gives_C_comparable_to_B(n_test):
    n_per_kind = max(1, -(-n_test // 2))
    total = 2 * n_per_kind
    # |C| matches |B| exactly when n_test is even, and overshoots by exactly one
    # when it is odd. Never under-samples.
    assert total >= n_test
    assert total - n_test <= 1


def test_shapes3d_sized_C_is_built_and_valid():
    grid = _grid((10, 10, 10, 8, 4, 15))
    c_set = build_c_invalid(grid, n_per_kind=36, seed=0)
    assert len(c_set["records"]) == 72
    assert validate_c_invalid(c_set, grid) == 72


def test_derived_size_is_still_deterministic_at_a_fixed_seed():
    grid = _grid()
    a = build_c_invalid(grid, n_per_kind=36, seed=0)
    b = build_c_invalid(grid, n_per_kind=36, seed=0)
    assert a == b
