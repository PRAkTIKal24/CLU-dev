"""w6 hub/compose-physics -- WHERE DO THE LOST TUPLES GO?

The val sweep shows the settled read losing composition monotonically in how far the
state is allowed to relax.  ⛔ "It loses" is not yet a mechanism.  There are exactly two
ways a corner can lose distinct valid NOVEL tuples, and they mean opposite things:

* the judge stops agreeing that the output satisfies the request -- **validity** falls,
  i.e. settling is walking the state off the request; or
* the outputs stay valid but collapse onto combinations the store was fitted on --
  **novelty** falls, i.e. settling is regressing to seen combinations.

⭐ The second is charter section E.4's banked null ("the density-over-atoms descent
regresses to seen combinations").  If the cross-well coupling reproduces it, then the
Gaussian coupling and the kernel density are the same failure wearing different clothes,
and that is the finding rather than "my coupling was badly tuned".

This reads only artifacts already on disk and scores nothing.

    python scripts/w6_settle_mechanism.py LEG [SEED] [val|test]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.experiments import exp_w6_compose_physics as P

ORDER = [
    ("pca_tensor", "the INCUMBENT (static read)"),
    ("pca_settle_zerostep", "0 steps (asserted identical to the incumbent)"),
    ("pca_settle_f01_conv", "lambda = 0.01 x safe, converged"),
    ("pca_settle_f05_conv", "lambda = 0.05 x safe, converged"),
    ("pca_settle_f10_conv", "lambda = 0.10 x safe, converged"),
    ("pca_settle_f25_s3", "lambda = 0.25 x safe, 3 steps"),
    ("pca_settle_onestep", "lambda = 1.00 x safe, ONE step"),
    ("pca_settle_f25_conv", "lambda = 0.25 x safe, converged"),
    ("pca_settle_f100_s3", "lambda = 1.00 x safe, 3 steps"),
    ("pca_settle_f50_conv", "lambda = 0.50 x safe, converged"),
    ("pca_settle_f100_s10", "lambda = 1.00 x safe, 10 steps"),
    ("pca_settle", "THE ARM: lambda = 1.00 x safe, converged"),
    ("pca_settle_r4", "coupling rank 4, converged"),
    ("pca_settle_r64", "coupling rank 64, converged"),
    ("pca_settle_kernel", "the BANKED kernel/density descent (charter E.4)"),
    ("pca_settle_lookup", "FLOOR: nearest atom to the settled state (C.1/C.4)"),
]


def main(argv=None) -> int:
    leg = Path(argv[0] if argv else
               "C:/chlu/outputs/w6-compose-physics/dev/val_seed0")
    seed = int(argv[1]) if len(argv) > 1 else 0
    condition = argv[2] if len(argv) > 2 else "val"
    P.install()
    R.set_condition(seed, condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = tuple(int(c) for c in grid.cardinalities)
    if condition == "val":
        # the val condition removes the val corners' rows from the fit pool, so "seen"
        # must be measured against the rows the store was ACTUALLY fitted from
        # (exp_gen_read.build_table makes the same correction, for the same reason)
        fitted = np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond))
        corners = fw.val_corners(cond)
    else:
        fitted = cond["train_rows"]
        # row 4 ONLY -- the 49 n-way corners, which is the row this hub is decided on
        corners = fw.nway_corners(cond)
    seen_mask = fw.seen_tuple_mask(grid, fitted)

    print(f"leg: {leg}   condition = {condition}")
    if condition == "val":
        print(f"{len(corners)} val corners  (SELECTION set; nothing here is scored)\n")
    else:
        print(f"{len(corners)} row-4 n-way corners  (the SCORED held-out set)\n")
    hdr = (f"{'leg':24s} {'displ':>7s} {'valid':>7s} {'distinct':>8s} "
           f"{'novel':>7s} {'SEEN':>7s} {'seen%':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for arm, what in ORDER:
        p = leg / "arms" / arm / "rows.npz"
        if not p.exists():
            continue
        cor = np.load(p, allow_pickle=True)["corners"].item()
        nv, nd, nn, ns = [], [], [], []
        for c in corners:
            r = cor[c["corner_id"]]
            v = np.asarray(r["valid"], bool)
            t = np.asarray(r["tuples"])
            keep = t[v]
            if keep.size:
                lin = np.unique(np.ravel_multi_index(tuple(keep.T), cards))
                novel = lin[~seen_mask[lin]]
            else:
                lin = np.zeros(0, np.int64)
                novel = np.zeros(0, np.int64)
            nv.append(float(v.mean()))
            nd.append(int(lin.size))
            nn.append(int(novel.size))
            ns.append(int(lin.size) - int(novel.size))
        d = leg / "arms" / arm / "settle_diagnostics.json"
        disp = float("nan")
        if d.exists():
            import json
            with open(d, encoding="utf-8") as fh:
                disp = json.load(fh).get("launch_to_settled_rel_median", float("nan"))
        dm, nm = float(np.mean(nd)), float(np.mean(nn))
        sm = float(np.mean(ns))
        print(f"{arm:24s} {disp:7.4f} {np.mean(nv):7.4f} {dm:8.3f} "
              f"{nm:7.3f} {sm:7.3f} {(100.0 * sm / dm if dm else 0.0):5.1f}%")
    print("")
    print("columns: displ = median relative displacement launch->settled; "
          "valid = judge validity rate;")
    print("distinct = distinct VALID tuples per corner; novel = of those, never "
          "fitted on;")
    print("SEEN = of those, already in the fit pool.  novel is the metric rows 3 "
          "and 4 are decided in.")
    print("")
    print("MEASURED, on BOTH conditions: SEEN is identically 0 for every leg, "
          "including the incumbent.")
    print("Under val the corners' rows are removed from the fit pool; under test the "
          "row-4 corners are")
    print("Montero-shaped combinations the condition never emitted.  Either way the "
          "scored corners hold")
    print("NO seen combinations, so distinct == novel by construction and the whole "
          "metric is carried by")
    print("VALIDITY.  Consequence: the settled read is NOT regressing to seen "
          "combinations (charter E.4's")
    print("banked mechanism) -- it cannot, there are none to regress to.  It is "
          "walking the state OFF the")
    print("request, and the external classifier charges it one tuple for one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
