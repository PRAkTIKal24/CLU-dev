"""WHICH POPULATION each rival number is on, measured rather than asserted.

hub/read measured that the value-holdout test index is ~80 percent rows carrying
a value no training row holds, while every one of its corner sets is
retained-values-only. Those are different populations, and a number quoted from
one against a number from the other is not a comparison. It cost that hub an
hour: a read scoring R^2 = -7.08 on random test rows was the best read it had on
the corner population.

This spoke's rival numbers are on a THIRD population again - the Montero
recombination-to-element corner, which holds out a COMBINATION rather than a
value. So before four hubs quote them, the population is measured here and
stated in the readiness file:

* is every factor value appearing in the scored set also present in the fit pool?
  (if yes, the set is retained-values-only and nothing on it is row-2 territory)
* how many rows, and what does the compositional factor realise?

Nothing here is an assertion about what the split *ought* to contain. It reads
the realised rows and reports what is in them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def coverage(grid, fit_rows: np.ndarray, rows: np.ndarray) -> dict:
    """Per factor: which values this set realises, and whether the fit pool has them."""
    fi_fit = grid.factor_indices_of(np.asarray(fit_rows, np.int64))
    fi = grid.factor_indices_of(np.asarray(rows, np.int64))
    out, novel_any = {}, 0
    novel_rows = np.zeros(fi.shape[0], bool)
    for k, name in enumerate(grid.names):
        have = set(int(v) for v in np.unique(fi_fit[:, k]))
        here = sorted(int(v) for v in np.unique(fi[:, k]))
        novel = [v for v in here if v not in have]
        out[name] = {
            "cardinality": int(grid.cardinalities[k]),
            "values_realised": here,
            "values_absent_from_fit_pool": novel,
            "retained_values_only": not novel,
        }
        if novel:
            novel_any += 1
            novel_rows |= np.isin(fi[:, k], np.asarray(novel, np.int64))
    return {
        "n_rows": int(fi.shape[0]),
        "per_factor": out,
        "n_factors_with_a_novel_value": int(novel_any),
        "n_rows_carrying_a_novel_value": int(novel_rows.sum()),
        "frac_rows_carrying_a_novel_value": float(novel_rows.mean())
        if fi.shape[0] else 0.0,
        "retained_values_only": bool(novel_any == 0),
    }


def describe(ctx: dict) -> dict:
    """The population every rival number in this spoke is measured on."""
    grid = ctx["grid"]
    fit = ctx["row_sets"]["fit"]
    sets = {}
    for name in ("B", "MS", "fit", "val"):
        sets[name] = coverage(grid, fit, ctx["row_sets"][name])
    retained = all(v["retained_values_only"] for v in sets.values())
    return {
        "split": "Montero recombination-to-element, leave1out (a COMBINATION hold-out)",
        "note": (
            "This is NOT the value hold-out population. The scored set B is the "
            "Montero corner: every factor VALUE in it also appears in the fit pool, "
            "in a CONJUNCTION the pool never carries. MS is its photometrically "
            "matched-seen control. So no rival number from this spoke is on row-2 "
            "territory, and none of them should be read as evidence about a value "
            "the arm never held - that is what the deletion comparator measures."),
        "sets": sets,
        "all_scored_sets_retained_values_only": bool(retained),
        "what_each_statistic_is_on": {
            "judge_in_corner": "B (positives) against MS (negatives), both "
                               "retained-values-only",
            "judge_composition_B": "B only, retained-values-only",
            "distinct_factor_tuples_B": "B only, retained-values-only",
            "val_mse_full_schedule": "the stratified validation rows, "
                                     "retained-values-only, fully specified requests",
            "val_mse_partial_requests": "the same rows with a random subset of "
                                        "factors requested",
            "deletion_comparator": "forget/retain sets defined by a value REMOVED "
                                   "from the arm's fit pool - the only numbers in "
                                   "this spoke that are about an unheld value",
        },
    }


def emit(ctx: dict, path: str | Path, log=lambda m: None) -> Path:
    doc = describe(ctx)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log(f"  population: all scored sets retained-values-only = "
        f"{doc['all_scored_sets_retained_values_only']}; "
        f"B={doc['sets']['B']['n_rows']} rows, MS={doc['sets']['MS']['n_rows']} rows")
    return p


# ---------------------------------------------------------------------------
# The fit population, carried ON the fitted object
# ---------------------------------------------------------------------------

#: The pool every rival in this spoke is fitted on.
FIT_POPULATION = "montero_combination"

FIT_POPULATION_NOTE = (
    "These rivals are fitted on the MONTERO COMBINATION-SPLIT pool, which contains "
    "every factor VALUE and holds out only combinations. A value-holdout arm is "
    "fitted on a pool with a whole value carved out. On a value-holdout row (rows 2 "
    "and 5) the two are NOT like-for-like: this rival saw the value, that arm did "
    "not, and the comparison flatters whichever one saw more. The mechanism can "
    "enter the row; the FIT cannot. Use n_per_value to check mechanically rather "
    "than trusting this sentence.")


def fit_population_record(grid, fit_rows, population: str = FIT_POPULATION,
                          note: Optional[str] = None) -> dict:
    """What this rival's fit pool actually contained, per factor value.

    Recorded as counts rather than as a claim, so a consumer holding out any value
    can check whether the rival saw it without knowing anything about how the pool
    was built. ``0`` for a value means the rival never fitted a row carrying it.
    """
    fi = grid.factor_indices_of(np.asarray(fit_rows, np.int64))
    per_value, zeros = {}, {}
    for k, name in enumerate(grid.names):
        counts = np.bincount(fi[:, k], minlength=int(grid.cardinalities[k]))
        per_value[name] = [int(c) for c in counts]
        z = [int(i) for i in np.nonzero(counts == 0)[0]]
        if z:
            zeros[name] = z
    return {
        "fit_population": population,
        "fit_population_note": note or (
            FIT_POPULATION_NOTE if population == FIT_POPULATION else
            "fitted on a pool with whole VALUES carved out; see values_never_fitted"),
        "n_rows_fit": int(fi.shape[0]),
        "n_per_value": per_value,
        "grid_values": {n: [float(v) for v in np.asarray(grid.values[k])]
                        for k, n in enumerate(grid.names)},
        "values_never_fitted": zeros,
        "holds_out_any_value": bool(zeros),
        "not_like_for_like_with": (
            "any arm fitted on a value-holdout pool, on rows 2 and 5"),
    }


def assert_not_fitted_on(record: dict, held_out: dict) -> dict:
    """Did this rival's fit pool contain a value the caller is holding out?

    Args:
        record: a :func:`fit_population_record`.
        held_out: ``{factor_name: [values]}`` the caller treats as held out.

    Returns:
        A verdict dict. ``like_for_like`` is False when the rival fitted rows
        carrying a value the caller holds out - which makes any comparison on that
        row a comparison between a rival that saw the value and an arm that did
        not.
    """
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
                    {"value": float(v), "grid_index": idx,
                     "rows_fitted_carrying_it": int(counts[idx])})
    return {
        "held_out_by_caller": held_out,
        "values_this_rival_DID_fit": seen,
        "like_for_like": not seen,
        "verdict": ("like-for-like: this rival never fitted a row carrying any "
                    "value you hold out" if not seen else
                    "NOT like-for-like: this rival fitted rows carrying a value you "
                    "hold out, so a row-2/row-5 comparison against a value-holdout "
                    "arm flatters this rival. Report it with a population footnote "
                    "or refit on your own pool."),
    }
