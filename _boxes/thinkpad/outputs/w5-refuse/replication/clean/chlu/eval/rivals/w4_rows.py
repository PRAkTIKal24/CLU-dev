"""Which rival can enter which row, and the footnote when it cannot.

"A rival that cannot enter a row gets a dash and a footnote, and a dash is never
a win for us." That rule only works if the dashes are decided in advance, on
structural grounds, and written down where a hub reads them - not discovered
later by whoever is assembling a table and tempted to leave a gap unexplained.

So the eligibility is declared here, per rival per row, with the reason, and it
is emitted as an artifact beside the numbers.

The three structural capabilities a row can demand
---------------------------------------------------
``offgrid``   the request names a value the grid does not contain (row 2). A
              rival whose input is a class index or a one-hot has a CLOSED
              vocabulary over the realised grid and cannot express such a value
              at all.
``full``      the request names all six factors on the grid. This is what the
              pair and n-way corners of rows 3 and 4 actually ask:
              ``exp_gen_factor_wells.run_arm`` builds them with
              ``grid.values_of(rows)``, so the cell may constrain two factors
              while the REQUEST still names all six. Every rival can do this.
``partial``   the request leaves some factors unconstrained, spelled ``nan`` by
              ``arm_features``. This reaches the learners through the QUERY SETS
              on rows 1, 2, 5 and 6 - not through rows 3 and 4.
``deletable`` the rival can be refitted from scratch on a pool with a whole
              value removed (rows 5-7, the value hold-out). Every rival here is
              refittable; what differs is the cost, which the deletion
              comparator measures rather than assumes.

A dash on ``offgrid`` for LACE and the energy-based arm is the METHOD's limit,
documented at ``chlu.core.gen_request.DenseRequests``, not a limit this spoke
imposed and not a weakness introduced by our code. Recording whose limit it is
matters: a referee reading an unexplained dash assumes we chose it.
"""

from __future__ import annotations

import json
from pathlib import Path

CLOSED_VOCAB = (
    "closed vocabulary: this rival's input is a class index over the realised "
    "grid, so a value between or outside the grid points cannot be expressed at "
    "all. Snapping it to the nearest grid value would score the rival on a "
    "request it never received, so it is refused instead.")

SAMPLER_DRAW = (
    "this arm's output is a stochastic draw, not a function of the request: two "
    "calls on the same request return different features by construction.")

# The full-only PRIMARY variants refuse a partial request: they can encode one but
# were never trained on one, and the _dropout sibling was. Refusing and naming the
# sibling is how a hub avoids quoting an out-of-distribution number when an
# in-distribution one exists.
ELIGIBILITY = {
    "table": {
        "offgrid": (True, "FactorPrototypes.prototype interpolates between grid "
                          "points and extrapolates outside them"),
        "partial": (True, "the composition rule sums one term per REQUESTED "
                          "factor; an unrequested factor contributes nothing"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "closed-form refit from the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "cont_mlp": {
        "offgrid": (True, "its input is the min-max normalised VALUE, so an "
                          "off-grid hue is simply a number no training row carried"),
        "partial": (False, "PRIMARY variant, full requests only; load "
                           "cont_mlp_dropout for a partial request"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "mlp": {
        "offgrid": (False, CLOSED_VOCAB + " (one-hot over the grid values)"),
        "partial": (False, "the PRIMARY variant is trained on fully specified "
                           "requests only, which is what rows 3 and 4 ask. For a "
                           "partial request load mlp_dropout, which is the same "
                           "architecture at the same matched count."),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "transformer": {
        "offgrid": (False, CLOSED_VOCAB + " (token over the grid values)"),
        "partial": (False, "PRIMARY variant, full requests only; load "
                           "transformer_dropout for a partial request"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "lstm": {
        "offgrid": (False, CLOSED_VOCAB + " (token over the grid values)"),
        "partial": (False, "PRIMARY variant, full requests only; load "
                           "lstm_dropout for a partial request"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "mlp_180k": {
        "offgrid": (False, CLOSED_VOCAB + " (one-hot over the grid values)"),
        "partial": (False, "PRIMARY variant, full requests only; load "
                           "mlp_180k_dropout for a partial request"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "*_dropout (every learner)": {
        "offgrid": (False, "same as the primary variant of that family: only "
                           "cont_mlp_dropout accepts an off-grid value"),
        "partial": (True, "trained with half its requests partially specified, so "
                          "an absent factor is in distribution"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "lace_repaired": {
        "offgrid": (False, "LACE's categorical head has a closed vocabulary over "
                           "the realised grid and cannot express an off-grid value "
                           "(chlu.core.gen_request.DenseRequests). THE METHOD'S "
                           "limit, not this spoke's."),
        "partial": (True, "on-grid partial requests only; the dense request form "
                          "carries a target per factor"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "refit the whitening and classifier on the reduced pool"),
        "callable": (False, SAMPLER_DRAW + " Ask this spoke to score your corners."),
    },
    "poe_ebm": {
        "offgrid": (False, "the per-factor experts are categorical over the "
                           "realised grid. THE METHOD'S limit, not this spoke's."),
        "partial": (False, "the experts are composed by summing one energy per "
                           "factor over a FULL assignment; an absent factor has no "
                           "expert term and the product of experts is undefined "
                           "for it"),
        "full": (True, "a fully specified on-grid request is what "
                       "every rival in this set is built for"),
        "deletable": (True, "retrain the experts on the reduced pool"),
        "callable": (False, SAMPLER_DRAW + " Ask this spoke to score your corners."),
    },
}

#: ⚠ POPULATION, which is separate from capability and is easier to miss.
#: Every rival in this set is fitted on the MONTERO COMBINATION pool, which
#: contains every factor value. A value-holdout arm is fitted on a pool with a
#: whole value carved out. On rows 2 and 5 the mechanism can enter but the FIT
#: cannot: this rival saw the value, that arm did not, so the comparison is not
#: like-for-like and flatters this rival.
#:
#: Measured, not asserted: the fitted table's n_per_value[object_hue] reads
#: [3236, 3223, 3164, 3243, 3312, 3157, 3324, 3114, 3272, 3211] - 3,157 rows at
#: the held-out hue 0.5, against 0 for a value-holdout arm.
POPULATION_CAVEAT_ROWS = ("row 2 (an off-grid value)",
                          "rows 5-7 (a whole value the arm never fitted)")
POPULATION_CAVEAT = (
    "ELIGIBLE BUT WRONG POPULATION. This rival is fitted on the Montero "
    "combination pool, which contains every factor value, so on a value-holdout "
    "row it saw the value the arm under test never did. The mechanism can enter "
    "the row; the fit cannot. Either report it with a population footnote and do "
    "NOT count it as beating a value-holdout arm, or refit it on your own pool. "
    "Check mechanically with load_rival(seed, name).like_for_like({factor: "
    "[values]}) rather than relying on this note.")

#: Which structural capability each row of the taxonomy demands.
ROW_DEMANDS = {
    "row 2 (an off-grid value)": "offgrid",
    "rows 3-4 (pair and n-way corners; FULL six-factor requests)": "full",
    "rows 1/2/5/6 via the query sets (partially specified requests)": "partial",
    "rows 5-7 (a whole value the arm never fitted)": "deletable",
}


def eligibility_table() -> dict:
    """Per rival per row: can it enter, and if not, whose limit is it."""
    out = {}
    for rival, caps in ELIGIBILITY.items():
        rows = {}
        for row, cap in ROW_DEMANDS.items():
            ok, why = caps[cap]
            entry = {"can_enter": bool(ok),
                     "reason" if ok else "footnote": why}
            if row in POPULATION_CAVEAT_ROWS:
                entry["population"] = "montero_combination"
                entry["like_for_like_with_a_value_holdout_arm"] = False
                entry["POPULATION_CAVEAT"] = POPULATION_CAVEAT
            rows[row] = entry
        out[rival] = {
            "rows": rows,
            "capabilities": {k: {"supported": v[0], "note": v[1]}
                             for k, v in caps.items()},
        }
    return out


def emit(path: str | Path) -> Path:
    doc = {
        "rule": ("A rival that cannot enter a row gets a DASH and a FOOTNOTE. "
                 "A dash is never a win for us, and every dash here names whose "
                 "limit it is."),
        "generation": 2,
        "rows": list(ROW_DEMANDS),
        "eligibility": eligibility_table(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p


def render() -> str:
    """The eligibility table as markdown."""
    L = ["### Which rival can enter which row", "",
         "A dash below is a structural fact about the rival, decided in advance "
         "and footnoted. It is never a win for us.", "",
         "| rival | rows 3-4 (FULL requests) | row 2 (off-grid value) | "
         "partial requests (query sets) | rows 5-7 (value hold-out) | callable |",
         "|---|:--:|:--:|:--:|:--:|:--:|"]
    tick = {True: "yes", False: "—"}
    for rival, caps in ELIGIBILITY.items():
        L.append(f"| {rival} | {tick[caps['full'][0]]} | "
                 f"{tick[caps['offgrid'][0]]} | {tick[caps['partial'][0]]} | "
                 f"{tick[caps['deletable'][0]]} | {tick[caps['callable'][0]]} |")
    L.append("")
    L.append("**Population caveat on rows 2 and 5-7.** Every rival here is fitted on "
             "the Montero combination pool, which contains every factor value. On a "
             "value-holdout row it saw the value the arm under test never did - the "
             "fitted table has 3,157 rows at `object_hue` 0.5 against a value-holdout "
             "arm's 0. Eligible, but NOT like-for-like: report with a population "
             "footnote, or refit on your own pool. Check with "
             "`load_rival(seed, name).like_for_like({factor: [values]})`.")
    L.append("")
    L.append("Footnotes for every dash:")
    L.append("")
    for rival, caps in ELIGIBILITY.items():
        for cap, (ok, why) in caps.items():
            if not ok:
                L.append(f"- **{rival} / {cap}** — {why}")
    L.append("")
    return "\n".join(L)
