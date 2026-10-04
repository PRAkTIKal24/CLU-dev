"""hub/refusal-2 -- is the factor DECODABLE from the well's features at all?  (the honest ceiling)

MECHANISM ONLY.  fit rows and their labels only -- split in half, fitted on one half, scored
on the other.  No val, no C record, no a7_redundant.json, no held-out row.  NOT scored,
NOT adopted.

WHY THIS EXISTS.  ``addressability_locus`` measured a Fisher between/within ratio and found
the three hues 10-38x more separable in the frozen encoder's features than scale, shape and
orientation.  ⛔ Fisher is a LINEAR measure, and a reviewer's first move is "your factor is
encoded non-linearly and your conclusion is too strong."  So the same question is asked
again by two decoders that do not assume a linear class structure:

  * NEAREST CLASS CENTROID -- fit centroids on fit-half A, classify fit-half B;
  * 1-NEAREST NEIGHBOUR    -- classify fit-half B against fit-half A directly.

Both are reported against the factor's own chance rate (1 / cardinality) and against the
majority-class rate, so an imbalanced factor cannot look good for free.  ⭐ If a factor is
at chance under BOTH decoders in EVERY well, it is not in the representation, and no address
book over atoms drawn from that representation can address it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))

import numpy as np  # noqa: E402

import refusal_lib as R  # noqa: E402


def nearest_centroid(Za, ya, Zb, card: int):
    cen, present = [], []
    for v in range(card):
        sel = ya == v
        if sel.any():
            cen.append(Za[sel].mean(0))
            present.append(v)
    if not cen:
        return None
    C = np.stack(cen)
    d = ((Zb * Zb).sum(1)[:, None] - 2.0 * (Zb @ C.T) + (C * C).sum(1)[None, :])
    return np.asarray(present, np.int64)[d.argmin(1)]


def one_nn(Za, ya, Zb, chunk: int = 512):
    out = np.zeros(Zb.shape[0], np.int64)
    aa = (Za * Za).sum(1)[None, :]
    for i in range(0, Zb.shape[0], chunk):
        z = Zb[i: i + chunk]
        d = (z * z).sum(1)[:, None] - 2.0 * (z @ Za.T) + aa
        out[i: i + z.shape[0]] = ya[d.argmin(1)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="A")
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()

    ctx = R.build_context(int(a.seed))
    built = R.build_land(ctx, str(a.budget))
    store = built["land"].store
    grid = ctx["grid"]
    names = list(grid.names)
    cards = [int(c) for c in grid.cardinalities]

    X = np.asarray(ctx["X_store"], np.float32)
    X = X.reshape((X.shape[0], -1)).astype(np.float64)
    Y = np.asarray(ctx["lab_store"], np.int64)
    rng = np.random.default_rng(20260917)
    n = min(int(a.n), X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    half = n // 2
    ia, ib = idx[:half], idx[half:]

    rows = {}
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        Za, Zb = X[ia][:, cols], X[ib][:, cols]
        for f, nm in enumerate(names):
            ya, yb = Y[ia][:, f], Y[ib][:, f]
            pc = nearest_centroid(Za, ya, Zb, cards[f])
            pn = one_nn(Za, ya, Zb)
            counts = np.bincount(ya, minlength=cards[f])
            rows.setdefault(nm, {})[f"well_{k}"] = {
                "nearest_centroid_acc": float((pc == yb).mean()) if pc is not None else None,
                "one_nn_acc": float((pn == yb).mean()),
                "chance_1_over_card": float(1.0 / cards[f]),
                "majority_class_rate": float(counts.max() / max(counts.sum(), 1)),
            }
        R.tlog(f"  well {k} done")

    summary = {}
    for nm in names:
        f = names.index(nm)
        nc = max(v["nearest_centroid_acc"] or 0.0 for v in rows[nm].values())
        nn = max(v["one_nn_acc"] for v in rows[nm].values())
        summary[nm] = {
            "best_nearest_centroid_acc_over_wells": nc,
            "best_1nn_acc_over_wells": nn,
            "chance": 1.0 / cards[f],
            "lift_over_chance_centroid": nc * cards[f],
            "lift_over_chance_1nn": nn * cards[f],
        }

    rep = {"schema": "hub-refusal-2/feature-decodability/1",
           "DECLARED": ("MECHANISM ONLY. fit rows only, split in half; fitted on one half, "
                        "scored on the other. No val, no C, no held-out row. NOT scored, "
                        "NOT adopted."),
           "seed": int(a.seed), "budget": str(a.budget),
           "n_fit_rows_used": n, "n_wells": int(store.n_wells),
           "per_factor_per_well": rows,
           "BEST_OVER_WELLS": summary,
           "reading": ("a factor at chance under BOTH decoders in EVERY well is not in the "
                       "representation, and no address book over atoms drawn from that "
                       "representation can address it.")}
    p = R.OUT / f"FEATURE_DECODABILITY_seed{a.seed}_{a.budget}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("\n%-13s %9s %9s %9s %9s %9s" % (
        "factor", "chance", "centroid", "lift", "1-NN", "lift"))
    for nm, s in summary.items():
        print("%-13s %9.4f %9.4f %9.2fx %9.4f %9.2fx" % (
            nm, s["chance"], s["best_nearest_centroid_acc_over_wells"],
            s["lift_over_chance_centroid"], s["best_1nn_acc_over_wells"],
            s["lift_over_chance_1nn"]))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
