"""w6.1 retry -- IF NOT THE SUBSPACE, THEN WHAT?  The ANSWER LATTICE.

`w61_settle_subspace.py` kills the declared hypothesis: the settled state never leaves
`span(G_k)` -- it cannot, the settle is parameterised in span coordinates and emits
through the same `render_span` the incumbent uses.  So the loss has to come from motion
*inside* the subspace, and this script measures what that motion does.

The read is an address book: every grid tuple has an answer, and the answers of two
requests differing in ONE factor value are a finite distance apart.  Those answers are a
LATTICE inside `span(G_k)`, and the judge's job is to land the state in the right cell.
Per request this script builds the answers of all Hamming-1 neighbours (every other value
of every factor, 51 of them on this grid) from the same address book and measures:

  1. **the local spacing** -- how far the nearest neighbouring answer is;
  2. **the tangent split** -- how much of the displacement lies in the span of the
     neighbour-difference directions (motion ALONG the lattice, i.e. toward other valid
     answers) versus orthogonal to it (motion off every answer at once);
  3. **the landing cell** -- which lattice answer the settled state is nearest to, and
     whether the judge's read-back is the tuple of THAT answer.

(3) is the causal form of the claim: if the lost tuples land on the neighbour the state
drifted into, then the loss is a resolution failure, not a representation failure.

Reads only the parent's artifacts.  No judge, nothing scored.

    python scripts/w61_answer_lattice.py --leg DIR --seed S --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.experiments import exp_w6_compose_physics as P

LEGS = ("pca_tensor", "pca_settle_zerostep", "pca_settle_f05_conv",
        "pca_settle_onestep", "pca_settle")


def load_arm(leg: Path, arm: str) -> dict:
    with open(leg / "arms" / arm / "arm.pkl", "rb") as fh:
        return pickle.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tangent-every", type=int, default=6,
                    help="do the tangent split on every Nth request (it is an "
                         "O(51^2) solve per request and the estimate is a median)")
    a = ap.parse_args(argv)
    leg = Path(a.leg)
    t0 = time.time()

    P.install()
    R.set_condition(a.seed, a.condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    corners = (fw.val_corners(cond) if a.condition == "val"
               else fw.nway_corners(cond))
    cards = [int(c) for c in grid.cardinalities]
    gv = [np.asarray(v, np.float64) for v in grid.values]
    F = len(cards)
    #: the shell is laid out on a FIXED grid of (factor, value) columns, so column
    #: (f, v) always means "the same request with factor f set to value v".  The six
    #: columns where v is the request's own value are the request itself and are
    #: dropped by the zero-distance test below; that keeps the array rectangular.
    cols = [(f, v) for f in range(F) for v in range(cards[f])]
    nnb = len(cols)

    arms = {arm: load_arm(leg, arm) for arm in LEGS
            if (leg / "arms" / arm / "arm.pkl").exists()}
    rows = {arm: np.load(leg / "arms" / arm / "rows.npz",
                         allow_pickle=True)["corners"].item() for arm in arms}
    print(f"[seed {a.seed}] {len(corners)} corners, {nnb} Hamming-1 neighbours each",
          flush=True)

    acc = {arm: {"wrong": [], "near_self": [], "near_is_judged": [],
                 "near_factor_hit": [], "valid": [], "tan": [], "orth": [],
                 "d_self": [], "d_near": [], "ferr": [], "prog": []}
           for arm in arms}
    #: ⭐ the lattice spacing FACTOR BY FACTOR -- the distance from a request's
    #: answer to the nearest answer that differs only in that factor.  This is the
    #: resolution the judge has to have, per factor, and it is not the same in each.
    fspace = [[] for _ in range(F)]

    for ci, c in enumerate(corners):
        vals = fw._request_values_from_rows(grid, c["rows"])
        n = vals.shape[0]
        I = np.zeros((n, F), np.int64)
        for f in range(F):
            I[:, f] = np.abs(vals[:, f][:, None] - gv[f][None, :]).argmin(1)
        # the Hamming-1 shell of every request, as VALUES the address book can read
        nb_vals = np.repeat(vals[:, None, :], nnb, axis=1)
        nb_idx = np.repeat(I[:, None, :], nnb, axis=1)
        selfcol = np.zeros((n, nnb), bool)
        for j, (f, v) in enumerate(cols):
            nb_vals[:, j, f] = gv[f][v]
            nb_idx[:, j, f] = v
            selfcol[:, j] = I[:, f] == v
        obj_t = arms["pca_tensor"]
        Xreq, _ = R.arm_features(obj_t, vals, grid)
        Xreq = np.asarray(Xreq, np.float64)
        flat = nb_vals.reshape(-1, F)
        Xnb, _ = R.arm_features(obj_t, flat, grid)
        Xnb = np.asarray(Xnb, np.float64).reshape(n, nnb, -1)
        Dnb = Xnb - Xreq[:, None, :]                     # (n, nnb, D)
        dnb = np.linalg.norm(Dnb, axis=2)
        # ⚠ The six columns where v is the request's own value are the request
        # itself and are masked BY INDEX, not by distance: `grid.values_of` hands back
        # float32-rounded values while `grid.values` holds the float64 grid, so a
        # self column reads back ~1e-6 away from the request rather than exactly 0
        # (against a real neighbour spacing of ~1.4, so it changes nothing else).
        same = selfcol
        dnb_m = np.where(same, np.inf, dnb)
        fmask = [np.array([cf == f for cf, _cv in cols]) for f in range(F)]
        fnear = np.zeros((n, F), np.int64)          # the nearest same-factor answer
        for f in range(F):
            sub = np.flatnonzero(fmask[f])
            j = sub[np.argmin(dnb_m[:, sub], axis=1)]
            fnear[:, f] = j
            fspace[f].append(dnb[np.arange(n), j])

        for arm, obj in arms.items():
            X, _ = R.arm_features(obj, vals, grid)
            X = np.asarray(X, np.float64)
            Delta = X - Xreq
            dself = np.linalg.norm(Delta, axis=1)
            # distance from the settled state to each neighbour's answer
            dto = np.linalg.norm(Xnb - X[:, None, :], axis=2)
            dto = np.where(same, np.inf, dto)
            jmin = dto.argmin(1)
            dmin = dto[np.arange(n), jmin]
            near_self = dself <= dmin
            r = rows[arm][c["corner_id"]]
            tj = np.asarray(r["tuples"], np.int64)
            val = np.asarray(r["valid"], bool)
            nb_t = nb_idx[np.arange(n), jmin]            # the landing cell's tuple
            acc[arm]["wrong"].append((tj != I).sum(1))
            acc[arm]["near_self"].append(near_self)
            acc[arm]["near_is_judged"].append(
                np.where(near_self, (tj == I).all(1), (tj == nb_t).all(1)))
            fdiff = np.argmax(nb_t != I, axis=1)
            acc[arm]["near_factor_hit"].append(
                np.where(near_self, (tj == I).all(1),
                         tj[np.arange(n), fdiff] == nb_t[np.arange(n), fdiff]))
            acc[arm]["valid"].append(val)
            acc[arm]["ferr"].append(tj != I)
            # ⭐ the FRACTIONAL PROGRESS of the displacement toward the nearest
            # answer that differs only in factor f: 0 = no motion that way, 0.5 = the
            # midpoint between this factor's answer and its neighbour's, 1 = onto it.
            pr = np.zeros((n, F), np.float64)
            for f in range(F):
                V = Dnb[np.arange(n), fnear[:, f]]           # (n, D)
                q = np.einsum("ij,ij->i", V, V)
                pr[:, f] = np.einsum("ij,ij->i", Delta, V) / np.maximum(q, 1e-30)
            acc[arm]["prog"].append(pr)
            acc[arm]["d_self"].append(dself)
            acc[arm]["d_near"].append(np.where(np.isfinite(dnb_m).any(1),
                                               dnb_m.min(1), np.nan))
            # tangent split on a subsample
            sub = np.arange(0, n, int(a.tangent_every))
            tan, orth = [], []
            for i in sub:
                M = Dnb[i][~same[i]]                     # (m, D) lattice directions
                G = M @ M.T
                y = M @ Delta[i]
                cf = np.linalg.lstsq(G, y, rcond=1e-10)[0]
                pn = float(np.sqrt(max(float(cf @ y), 0.0)))
                dn = float(dself[i])
                tan.append(pn / max(dn, 1e-12))
                orth.append(np.sqrt(max(dn * dn - pn * pn, 0.0)) / max(dn, 1e-12))
            acc[arm]["tan"].append(np.asarray(tan))
            acc[arm]["orth"].append(np.asarray(orth))
        if ci % 10 == 0:
            print(f"  corner {ci}/{len(corners)}  {time.time() - t0:.0f}s", flush=True)

    fspace = [np.concatenate(v) for v in fspace]
    out = {"leg": str(leg), "seed": a.seed, "n_corners": len(corners),
           "n_hamming1_neighbours": nnb,
           "factor_cardinalities": cards,
           "median_lattice_spacing_per_factor": [float(np.median(v)) for v in fspace],
           "legs": {}}
    for arm in arms:
        d = {k: np.concatenate(v) for k, v in acc[arm].items()}
        w = d["wrong"]
        ferr = d["ferr"]
        moved = ~d["near_self"]
        blk = {
            "validity": float(d["valid"].mean()),
            "median_displacement": float(np.median(d["d_self"])),
            "median_nearest_neighbour_answer_distance": float(np.nanmedian(d["d_near"])),
            "mean_wrong_factors": float(w.mean()),
            "wrong_factor_histogram": [int((w == k).sum()) for k in range(7)],
            "frac_nearest_answer_is_the_request": float(d["near_self"].mean()),
            "judged_tuple_is_the_nearest_answer": float(d["near_is_judged"].mean()),
            "judged_tuple_is_nearest_answer_given_moved":
                (float(d["near_is_judged"][moved].mean()) if moved.any() else None),
            "judged_factor_matches_landing_cell_given_moved":
                (float(d["near_factor_hit"][moved].mean()) if moved.any() else None),
            "validity_given_nearest_is_self":
                (float(d["valid"][d["near_self"]].mean())
                 if d["near_self"].any() else None),
            "validity_given_nearest_is_a_neighbour":
                (float(d["valid"][moved].mean()) if moved.any() else None),
            "median_tangent_fraction_of_displacement": float(np.median(d["tan"])),
            "median_orthogonal_fraction_of_displacement": float(np.median(d["orth"])),
            "n_tangent_sampled": int(d["tan"].size),
            "error_rate_per_factor": [float(ferr[:, f].mean()) for f in range(F)],
            "median_abs_progress_per_factor":
                [float(np.median(np.abs(d["prog"][:, f]))) for f in range(F)],
            "frac_progress_over_half_per_factor":
                [float((np.abs(d["prog"][:, f]) > 0.5).mean()) for f in range(F)],
        }
        out["legs"][arm] = blk
    np.savez_compressed(
        Path(a.out).with_suffix(".npz"),
        spacing=np.stack(fspace, 1),
        **{f"prog__{arm}": np.concatenate(acc[arm]["prog"]) for arm in arms},
        **{f"ferr__{arm}": np.concatenate(acc[arm]["ferr"]) for arm in arms},
        **{f"dself__{arm}": np.concatenate(acc[arm]["d_self"]) for arm in arms})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
