"""The rebased ABSOLUTE budget grid, by a rule that refers to route lengths only.

The second defect the repair exposes
------------------------------------
A two-key route is longer than a one-key route, and CMAZE-2's budget words are ABSOLUTE
integers (30, 45, 60, with 38 and 53 as the untrained interpolation words).  On the fixed
bank the fraction of kept pairs that can be WRITTEN at each of those words collapses --
0.0268 / 0.2007 / 0.4425 -- because a pair is only ever written at a budget word its
optimum fits.  The split then carries roughly a third of w9's training requests.

The rule, fixed before it was applied and not moved
---------------------------------------------------
Take the percentile that each of w9's words occupies in the BEFORE bank's optimum
distribution; read the AFTER bank's optimum distribution at those same percentiles; round
to the nearest multiple of 5.  The interpolation words are the midpoints of consecutive
words, exactly as 38 and 53 are the midpoints of 30-45 and 45-60.

The rule mentions route lengths and nothing else.  No planner, no validity, no blind
column enters it, so it cannot be tuned toward a number.  Applying it with the BEFORE bank
as both source and target returns w9's own grid, which is the check that it is a
generalisation of w9's choice rather than a replacement for it.
"""

from __future__ import annotations

import numpy as np

W9_GRID = (30, 45, 60)
W9_INTERP = (38, 53)


def optima(bank):
    return np.array([s.opt[p] for s in bank for p in s.pairs], dtype=float)


def percentiles_of(values, words):
    """The percentile each word occupies in ``values`` (fraction at or below it, x100)."""
    v = np.asarray(values, float)
    return [100.0 * float((v <= w).mean()) for w in words]


def rebase(before_bank, after_bank, grid=W9_GRID, round_to=5):
    """``(grid, interp, diagnostics)`` for ``after_bank``."""
    ob, oa = optima(before_bank), optima(after_bank)
    pct = percentiles_of(ob, grid)
    raw = [float(np.percentile(oa, p)) for p in pct]
    words = [int(round_to * round(r / round_to)) for r in raw]
    words = sorted(set(words))
    interp = [int(round((words[i] + words[i + 1]) / 2.0))
              for i in range(len(words) - 1)]
    diag = {"w9_grid": list(grid), "w9_interp": list(W9_INTERP),
            "percentiles_of_w9_grid_in_before_bank": pct,
            "raw_quantiles_of_after_bank": raw,
            "rebased_grid": words, "rebased_interp": interp,
            "before_opt": {"n": int(ob.size), "mean": float(ob.mean()),
                           "median": float(np.median(ob)), "max": float(ob.max())},
            "after_opt": {"n": int(oa.size), "mean": float(oa.mean()),
                          "median": float(np.median(oa)), "max": float(oa.max())},
            "writable_fraction_before_at_w9_grid":
                {str(w): float((ob <= w).mean()) for w in grid},
            "writable_fraction_after_at_w9_grid":
                {str(w): float((oa <= w).mean()) for w in grid},
            "writable_fraction_after_at_rebased_grid":
                {str(w): float((oa <= w).mean()) for w in words}}
    return tuple(words), tuple(interp), diag


def self_check(before_bank):
    """The rule applied with BEFORE as both source and target must return w9's own grid."""
    g, i, _d = rebase(before_bank, before_bank)
    return {"grid": list(g), "interp": list(i), "w9_grid": list(W9_GRID),
            "w9_interp": list(W9_INTERP), "grid_matches_w9": list(g) == list(W9_GRID),
            "interp_matches_w9": list(i) == list(W9_INTERP)}
