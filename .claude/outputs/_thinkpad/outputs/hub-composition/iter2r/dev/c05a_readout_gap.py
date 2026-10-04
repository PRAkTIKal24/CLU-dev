"""C05a — ⭐⭐ WHERE THE OBJECT ACTUALLY LIVES, and what that costs dreaming.

    python c05a_readout_gap.py <seed> <corner,corner,...>

⛔ SELECTION ONLY.

WHY
---
C05's zero-step control produced a fact that has to be understood before any dream number
means anything: taking the object's OWN launch `x = clu_wide_vector(store, a)` and re-reading
it through the store's soft occupancy `π(x)` — the only readout a *state-space* dream admits,
because a dream state is a point in feature space and carries no coefficients — took the
headline from **38 valid novel tuples to 0**, and distinct from 61 to 15, on corner 0.

This unit asks two precise questions and answers them with numbers:
 1. **Are the head's coefficients inside the mode simplex?**  `clu_wide_vector` computes
    `x̂_k = Σ_m a_km · mode_km` and nothing constrains `a` to be a probability vector.  If
    `a` has negative entries or rows that do not sum to 1, the object's output lies OUTSIDE
    the convex hull of the stored modes, and ⛔ no occupancy — at any temperature — can
    reproduce it.
 2. **Is there ANY τ at which the re-read recovers the object?**  A τ sweep over five decades
    settles it.  τ → 0 is the argmax read; τ → ∞ is the well-mean.  ⭐ If nothing in between
    recovers the headline, the gap is structural, ⛔ not a bad choice of τ.

⭐ This is the mechanism behind whatever C05's verdict turns out to be, and it is measured
on the object itself, not argued.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp  # noqa: E402
import ctx  # noqa: E402
import dev  # noqa: E402
import fast  # noqa: E402
import stores  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CORNERS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,2,4").split(",")]
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
OUTP = stores.CACHE.parent / f"c05a_readout_gap_seed{SEED}.json"

T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import (DreamLandscape, fit_code_coupling,  # noqa: E402
                                 mode_distance_scale, requests_from_labels,
                                 well_energies)
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_store import read_from_codes  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
res = json.load(open(OUTP, encoding="utf-8")) if OUTP.exists() else {"seed": SEED,
                                                                     "corners": {}}

for ci in CORNERS:
    if str(ci) in res["corners"]:
        ctx.log(f"[c{ci}] already done — skipped")
        continue
    t0 = time.time()
    s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    perm, modes = stores.build(c, ci, s["pool"], X_store, K, M, SEED)
    st = stores.to_gen_store(perm, modes, c["feat_shape"])
    pq = fast.EqualPQ(perm, np.asarray(modes, np.float32), (D,))
    head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
    a = np.asarray(head.coefficients(requests_from_labels(lab_dev, "full", 0, SEED)),
                   np.float64)                                          # (n, K, M)
    land = DreamLandscape(st, fit_code_coupling(pq.codes(Zs), M), 1.0)
    tau0 = float(mode_distance_scale(st)["scale"])

    # ---- Q1: where do the coefficients live? -------------------------------
    rs = a.sum(-1)                                                      # (n, K)
    q1 = {
        "row_sums_over_modes": {"mean": float(rs.mean()), "min": float(rs.min()),
                                "max": float(rs.max()),
                                "frac_within_1pct_of_1": float(
                                    (np.abs(rs - 1.0) < 0.01).mean())},
        "frac_coefficients_negative": float((a < 0).mean()),
        "min_coefficient": float(a.min()), "max_coefficient": float(a.max()),
        "L1_norm_per_well": {"mean": float(np.abs(a).sum(-1).mean()),
                             "max": float(np.abs(a).sum(-1).max())},
        "frac_wells_that_are_a_probability_vector": float(
            ((a >= -1e-6).all(-1) & (np.abs(rs - 1.0) < 0.01)).mean()),
    }
    ctx.log(f"[c{ci}] Q1 coefficients: row-sum mean {q1['row_sums_over_modes']['mean']:.4f} "
            f"(min {q1['row_sums_over_modes']['min']:.3f}, "
            f"max {q1['row_sums_over_modes']['max']:.3f}); "
            f"{100 * q1['frac_coefficients_negative']:.1f}% negative; "
            f"mean L1 per well {q1['L1_norm_per_well']['mean']:.3f}; "
            f"wells that ARE a probability vector: "
            f"{100 * q1['frac_wells_that_are_a_probability_vector']:.2f}%")

    # ---- the object, and the projection of it onto the mode simplex --------
    x = np.asarray(clu_wide_vector(st, a.astype(np.float32)), np.float32).reshape((-1, D))
    h_obj = comp.headline(c, x.reshape((-1,) + tuple(c["feat_shape"])), lab_dev, pool_tup)
    ctx.log(f"[c{ci}] THE OBJECT (coefficients straight to the read): "
            f"tuples_valid_novel={h_obj['tuples_valid_novel']}  distinct={h_obj['distinct']}")

    # ---- Q2: the tau sweep of the state-space re-read ----------------------
    sweep = []
    for mult in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        tau = tau0 * mult
        _e, pis = well_energies(land, x, tau)
        pi = np.stack([np.asarray(p, np.float32) for p in pis], axis=1)
        h = comp.headline(c, np.asarray(clu_wide_vector(st, pi), np.float32),
                          lab_dev, pool_tup)
        # how far is the re-read from the object it is trying to reproduce?
        xr = np.asarray(clu_wide_vector(st, pi), np.float32).reshape((-1, D))
        rel = float(np.linalg.norm(xr - x) / max(np.linalg.norm(x), 1e-30))
        sweep.append({"tau_multiple_of_store_scale": mult, "tau": tau,
                      "tuples_valid_novel": h["tuples_valid_novel"],
                      "distinct": h["distinct"], "aggregate": h["aggregate"],
                      "rel_error_vs_the_object": rel,
                      "mean_max_occupancy": float(pi.max(-1).mean())})
        ctx.log(f"[c{ci}]   tau={mult:>5g}x ({tau:7.3f})  re-read "
                f"tuples_valid_novel={h['tuples_valid_novel']:3d}  "
                f"distinct={h['distinct']:3d}  rel-err vs object={rel:.4f}  "
                f"mean max pi={pi.max(-1).mean():.4f}")
    h_arg = comp.headline(c, np.asarray(read_from_codes(st, pq.codes(x)), np.float32),
                          lab_dev, pool_tup)
    ctx.log(f"[c{ci}]   tau -> 0 (the ARGMAX read of the same state): "
            f"tuples_valid_novel={h_arg['tuples_valid_novel']}  distinct={h_arg['distinct']}")

    res["corners"][str(ci)] = {
        "corner": comp.CORNERS[ci], "tau_store_scale": tau0,
        "Q1_where_the_coefficients_live": q1,
        "THE_OBJECT": h_obj,
        "Q2_tau_sweep_of_the_state_space_re_read": sweep,
        "tau_to_zero_argmax_read_of_the_same_state": h_arg,
        "seconds": round(time.time() - t0, 1)}
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    ctx.log(f"[c{ci}] done in {time.time() - t0:.1f}s -> {OUTP}")

ctx.log(f"C05a done in {time.time() - T0:.1f}s")
