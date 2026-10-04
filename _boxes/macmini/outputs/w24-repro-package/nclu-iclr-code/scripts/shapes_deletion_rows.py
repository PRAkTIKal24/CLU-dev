"""shapes deletion rows."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RED_FACTOR, RED_INDEX = (2, 0)
SHAPE_FACTOR, CUBE_INDEX = (4, 0)
SPHERE_INDEX = 2
ITEM_FRACTION = 0.01


def factor_indices(pool: np.ndarray, cards) -> np.ndarray:
    """row id -> (n, F) factor indices, odometer order (last factor fastest)."""
    c = np.asarray([int(x) for x in cards], np.int64)
    strides = np.ones(c.size, np.int64)
    for i in range(c.size - 2, -1, -1):
        strides[i] = strides[i + 1] * c[i + 1]
    return np.asarray(pool, np.int64)[:, None] // strides[None, :] % c[None, :]


def corner_requests_value(corner, grid, factor: int, index: int) -> bool:
    """Does this corner ask for that factor value anywhere in its request set?"""
    from nclu.experiments import factor_wells as fw

    vals = fw._request_values_from_rows(grid, corner["rows"])
    gv = np.asarray(grid.values[factor], np.float64)
    target = float(gv[index])
    col = np.asarray(vals, np.float64)[:, factor]
    ok = ~np.isnan(col)
    return bool(np.isclose(col[ok], target).any())
