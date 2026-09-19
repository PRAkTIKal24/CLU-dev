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
``partial``   the request leaves some factors unconstrained (rows 3 and 4, the
              pair and n-way corners). ``arm_features`` spells this as ``nan``.
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

ELIGIBILITY = {
    "table": {
        "offgrid": (True, "FactorPrototypes.prototype interpolates between grid "
                          "points and extrapolates outside them"),
        "partial": (True, "the composition rule sums one term per REQUESTED "
                          "factor; an unrequested factor contributes nothing"),
        "deletable": (True, "closed-form refit from the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "cont_mlp": {
        "offgrid": (True, "its input is the min-max normalised VALUE, so an "
                          "off-grid hue is simply a number no training row carried"),
        "partial": (True, "a requested-flag channel per factor, trained with "
                          "request dropout"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "mlp": {
        "offgrid": (False, CLOSED_VOCAB + " (one-hot over the grid values)"),
        "partial": (True, "an absent factor is the all-zero one-hot block, "
                          "trained with request dropout"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "transformer": {
        "offgrid": (False, CLOSED_VOCAB + " (token over the grid values)"),
        "partial": (True, "a reserved absent token per factor, trained with "
                          "request dropout"),
        "deletable": (True, "retrain from scratch on the reduced pool"),
        "callable": (True, "exposed through w4_api.load_rival"),
    },
    "lstm": {
        "offgrid": (False, CLOSED_VOCAB + " (token over the grid values)"),
        "partial": (True, "a reserved absent token per factor, trained with "
                          "request dropout"),
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
        "deletable": (True, "retrain the experts on the reduced pool"),
        "callable": (False, SAMPLER_DRAW + " Ask this spoke to score your corners."),
    },
}

#: Which structural capability each row of the taxonomy demands.
ROW_DEMANDS = {
    "row 2 (an off-grid value)": "offgrid",
    "rows 3-4 (pair and n-way corners, partially specified)": "partial",
    "rows 5-7 (a whole value the arm never fitted)": "deletable",
}


def eligibility_table() -> dict:
    """Per rival per row: can it enter, and if not, whose limit is it."""
    out = {}
    for rival, caps in ELIGIBILITY.items():
        rows = {}
        for row, cap in ROW_DEMANDS.items():
            ok, why = caps[cap]
            rows[row] = {"can_enter": bool(ok),
                         "reason" if ok else "footnote": why}
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
         "| rival | row 2 (off-grid value) | rows 3-4 (partial requests) | "
         "rows 5-7 (value hold-out) | callable |",
         "|---|:--:|:--:|:--:|:--:|"]
    tick = {True: "yes", False: "—"}
    for rival, caps in ELIGIBILITY.items():
        L.append(f"| {rival} | {tick[caps['offgrid'][0]]} | "
                 f"{tick[caps['partial'][0]]} | {tick[caps['deletable'][0]]} | "
                 f"{tick[caps['callable'][0]]} |")
    L.append("")
    L.append("Footnotes for every dash:")
    L.append("")
    for rival, caps in ELIGIBILITY.items():
        for cap, (ok, why) in caps.items():
            if not ok:
                L.append(f"- **{rival} / {cap}** — {why}")
    L.append("")
    return "\n".join(L)
