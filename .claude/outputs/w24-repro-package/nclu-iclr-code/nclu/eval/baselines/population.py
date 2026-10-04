"""population."""

from __future__ import annotations
from typing import Optional
import numpy as np

FIT_POPULATION = "montero_combination"
FIT_POPULATION_NOTE = "These rivals are fitted on the MONTERO COMBINATION-SPLIT pool, which contains every factor VALUE and holds out only combinations. A value-holdout arm is fitted on a pool with a whole value carved out. On a value-holdout row (rows 2 and 5) the two are NOT like-for-like: this rival saw the value, that arm did not, and the comparison flatters whichever one saw more. The mechanism can enter the row; the FIT cannot. Use n_per_value to check mechanically rather than trusting this sentence."


def fit_population_record(
    grid, fit_rows, population: str = FIT_POPULATION, note: Optional[str] = None
) -> dict:
    """What this rival's fit pool actually contained, per factor value.

    Recorded as counts rather than as a claim, so a consumer holding out any value
    can check whether the rival saw it without knowing anything about how the pool
    was built. ``0`` for a value means the rival never fitted a row carrying it.
    """
    fi = grid.factor_indices_of(np.asarray(fit_rows, np.int64))
    per_value, zeros = ({}, {})
    for k, name in enumerate(grid.names):
        counts = np.bincount(fi[:, k], minlength=int(grid.cardinalities[k]))
        per_value[name] = [int(c) for c in counts]
        z = [int(i) for i in np.nonzero(counts == 0)[0]]
        if z:
            zeros[name] = z
    return {
        "fit_population": population,
        "fit_population_note": note
        or (
            FIT_POPULATION_NOTE
            if population == FIT_POPULATION
            else "fitted on a pool with whole VALUES carved out; see values_never_fitted"
        ),
        "n_rows_fit": int(fi.shape[0]),
        "n_per_value": per_value,
        "grid_values": {
            n: [float(v) for v in np.asarray(grid.values[k])] for k, n in enumerate(grid.names)
        },
        "values_never_fitted": zeros,
        "holds_out_any_value": bool(zeros),
        "not_like_for_like_with": "any arm fitted on a value-holdout pool, on rows 2 and 5",
    }


def assert_not_fitted_on(record: dict, held_out: dict) -> dict:
    seen = {}
    for factor, values in (held_out or {}).items():
        gv = record["grid_values"].get(factor)
        counts = record["n_per_value"].get(factor)
        if gv is None or counts is None:
            continue
        for v in values:
            idx = int(np.argmin(np.abs(np.asarray(gv, float) - float(v))))
            if counts[idx] > 0:
                seen.setdefault(factor, []).append(
                    {
                        "value": float(v),
                        "grid_index": idx,
                        "rows_fitted_carrying_it": int(counts[idx]),
                    }
                )
    return {
        "held_out_by_caller": held_out,
        "values_this_rival_DID_fit": seen,
        "like_for_like": not seen,
        "verdict": "like-for-like: this rival never fitted a row carrying any value you hold out"
        if not seen
        else "NOT like-for-like: this rival fitted rows carrying a value you hold out, so a row-2/row-5 comparison against a value-holdout arm flatters this rival. Report it with a population footnote or refit on your own pool.",
    }
