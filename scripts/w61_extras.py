"""w6.1 retry -- the supporting measurements, in one rerunnable place.

Six checks that the mechanism argument leans on.  Each was run once ad hoc while the
argument was being built; they live here so every number in the report can be
reproduced by one command.  ⛔ Re-analysis of the parent's paid artifacts plus
arithmetic; no judge, nothing scored.

  A. **Can ANY leg leave span(G_k)?**  Including `pca_settle_kernel`, the banked
     density descent, which runs in CHANNEL space and is the one leg with a chance of
     leaving.  (`val` only -- that is the only place the kernel leg was ever run.)
  B. **Standardisation round trip.**  Standardise and un-standardise the address
     book's coefficients with no steps taken; how big is the residue?
  C. **The lattice tangent's rank**, and what an isotropic step would put in it.
  D. **Row 3 against row 4.**  Is the scored metric on both rows just the per-request
     validity rate times the corner ceiling?
  E. **The motion budget.**  At what displacement does the loss reach the
     pre-registered +-1.0 novel tuples per corner?
  F. **Regression to seen combinations (charter E.4), tested on the ERRORS**, against
     a locality-matched null.

    python scripts/w61_extras.py --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.experiments import exp_w6_compose_physics as P

SCORED = "C:/chlu/outputs/w6-compose-physics/scored"
VAL = "C:/chlu/outputs/w6-compose-physics/dev/val_seed0"
LEGS = ("pca_tensor", "pca_settle_zerostep", "pca_settle_f05_conv",
        "pca_settle_onestep", "pca_settle")


def in_span_split(store, basis, X):
    ins = np.zeros(X.shape[0])
    out = np.zeros(X.shape[0])
    for idx, U in zip(store.well_channels, basis):
        idx = np.asarray(idx, np.int64)
        Un = np.asarray(U, np.float64)
        xb = np.asarray(X[:, idx], np.float64)
        pr = (xb @ Un) @ Un.T
        ins += (pr ** 2).sum(1)
        out += ((xb - pr) ** 2).sum(1)
    return np.sqrt(ins), np.sqrt(out)


def check_a(res):
    """Can any leg -- including the channel-space kernel descent -- leave the span?"""
    R.set_condition(0, "val")
    cond = fw.load_condition()
    grid = cond["grid"]
    arms = ("pca_tensor", "pca_settle", "pca_settle_kernel", "pca_settle_lookup")
    objs, rows = {}, {}
    for a in arms:
        with open(f"{VAL}/arms/{a}/arm.pkl", "rb") as fh:
            objs[a] = pickle.load(fh)
        rows[a] = np.load(f"{VAL}/arms/{a}/rows.npz",
                          allow_pickle=True)["corners"].item()
    store, basis = objs["pca_tensor"]["store"], objs["pca_tensor"]["basis"]
    acc = {a: {"off": [], "disp": [], "valid": []} for a in arms}
    for c in fw.val_corners(cond):
        vals = fw._request_values_from_rows(grid, c["rows"])
        Xt = np.asarray(R.arm_features(objs["pca_tensor"], vals, grid)[0], np.float64)
        for a in arms:
            X = np.asarray(R.arm_features(objs[a], vals, grid)[0], np.float64)
            i, o = in_span_split(store, basis, X)
            acc[a]["off"].append(o / np.maximum(i, 1e-12))
            acc[a]["disp"].append(np.linalg.norm(X - Xt, axis=1))
            acc[a]["valid"].append(
                np.asarray(rows[a][c["corner_id"]]["valid"], bool))
    res["A_can_any_leg_leave_the_span"] = {
        a: {"median_off_span_fraction": float(np.median(np.concatenate(d["off"]))),
            "max_off_span_fraction": float(np.max(np.concatenate(d["off"]))),
            "median_displacement": float(np.median(np.concatenate(d["disp"]))),
            "validity": float(np.concatenate(d["valid"]).mean())}
        for a, d in acc.items()}
    res["A_note"] = (
        "The kernel descent's force is x - sum_i p_i a_i: both terms lie in span(G_k), "
        "so span(G_k) is an INVARIANT subspace of it, exactly as it is of the "
        "coefficient-space settle. No dynamics built from these atoms can leave it.")


def check_b_c(res):
    """The standardisation round trip, and the rank of the lattice tangent."""
    R.set_condition(0, "test")
    cond = fw.load_condition()
    grid = cond["grid"]
    with open(f"{SCORED}/test_seed0/arms/pca_settle/arm.pkl", "rb") as fh:
        o = pickle.load(fh)
    with open(f"{SCORED}/test_seed0/arms/pca_tensor/arm.pkl", "rb") as fh:
        t = pickle.load(fh)
    cpl, addr = o["coupling"], o["address"]
    c = fw.nway_corners(cond)[0]
    vals = fw._request_values_from_rows(grid, c["rows"])
    Z, _ = addr.coefficients(vals, grid.values)
    cen = [np.asarray(x, np.float32) for x in cpl.centres]
    rad = [np.asarray(x, np.float32) for x in cpl.radii]
    rt = [((np.asarray(z, np.float32) - cc[None, :]) / ss[None, :]) * ss[None, :]
          + cc[None, :] for z, cc, ss in zip(Z, cen, rad)]
    num = float(np.sqrt(sum(float(((np.asarray(a, np.float32) - b) ** 2).sum())
                            for a, b in zip(rt, Z))))
    den = float(np.sqrt(sum(float((np.asarray(b, np.float32) ** 2).sum())
                            for b in Z)))
    res["B_standardisation_round_trip_relative"] = num / den

    cards = [int(x) for x in grid.cardinalities]
    gv = [np.asarray(v, np.float64) for v in grid.values]
    F = 6
    cols = [(f, v) for f in range(F) for v in range(cards[f])]
    n = vals.shape[0]
    I = np.zeros((n, F), np.int64)
    for f in range(F):
        I[:, f] = np.abs(vals[:, f][:, None] - gv[f][None, :]).argmin(1)
    nb = np.repeat(vals[:, None, :], len(cols), axis=1)
    selfc = np.zeros((n, len(cols)), bool)
    for j, (f, v) in enumerate(cols):
        nb[:, j, f] = gv[f][v]
        selfc[:, j] = I[:, f] == v
    Xr = np.asarray(R.arm_features(t, vals, grid)[0], np.float64)
    Xn = np.asarray(R.arm_features(t, nb.reshape(-1, F), grid)[0],
                    np.float64).reshape(n, len(cols), -1)
    D = Xn - Xr[:, None, :]
    rk = []
    for i in range(0, n, 6):
        s = np.linalg.svd(D[i][~selfc[i]], compute_uv=False)
        rk.append(int((s > s[0] * 1e-8).sum()))
    state_dim = int(sum(np.asarray(U).shape[1] for U in t["basis"]))
    res["C_lattice_tangent"] = {
        "median_rank": int(np.median(rk)), "n_directions": int((~selfc[0]).sum()),
        "state_dimension": state_dim,
        "isotropic_expected_tangent_fraction":
            float(np.sqrt(np.median(rk) / state_dim)),
    }


def check_d_e(res):
    """Row 3 against row 4, and the motion budget."""
    rows_out, budget = {}, {}
    for seed in (0, 1, 2):
        R.set_condition(seed, "test")
        cond = fw.load_condition()
        grid = cond["grid"]
        cards = tuple(int(c) for c in grid.cardinalities)
        seen = fw.seen_tuple_mask(grid, cond["train_rows"])
        sets = {"row3": (fw.pair_subcorners(cond), 36.0),
                "row4": (fw.nway_corners(cond), 72.0)}
        base = {}
        per = {}
        for arm in LEGS:
            cor = np.load(f"{SCORED}/test_seed{seed}/arms/{arm}/rows.npz",
                          allow_pickle=True)["corners"].item()
            blk = {}
            for rn, (cs, ceil) in sets.items():
                nov, val = [], []
                for c in cs:
                    r = cor[c["corner_id"]]
                    v = np.asarray(r["valid"], bool)
                    t = np.asarray(r["tuples"])
                    k = t[v]
                    nov.append(int((~seen[np.unique(np.ravel_multi_index(
                        tuple(k.T), cards))]).sum()) if k.size else 0)
                    val.append(v)
                nv = float(np.mean(nov))
                vv = float(np.concatenate(val).mean())
                if arm == "pca_tensor":
                    base[rn] = (nv, vv)
                blk[rn] = {
                    "novel_per_corner": nv, "validity": vv,
                    "delta_novel": nv - base[rn][0],
                    "delta_validity": vv - base[rn][1],
                    "delta_validity_times_ceiling": (vv - base[rn][1]) * ceil,
                }
            per[arm] = blk
        rows_out[str(seed)] = per
        # the motion budget: the local slope of validity loss in lattice units, from
        # the two legs that stay inside one lattice step
        sub = json.loads(Path(
            f"C:/chlu/outputs/w61-compose-physics/mech/subspace_seed{seed}.json"
        ).read_text("utf-8"))
        L = sub["legs"]
        x = np.array([L["pca_settle_f05_conv"]["median_displacement_over_rho1"],
                      L["pca_settle_onestep"]["median_displacement_over_rho1"]])
        y = np.array([L["pca_tensor"]["validity"] - L["pca_settle_f05_conv"]["validity"],
                      L["pca_tensor"]["validity"] - L["pca_settle_onestep"]["validity"]])
        sl = float(x @ y) / float(x @ x)
        dose = (1.0 / 72.0) / sl
        budget[str(seed)] = {
            "slope_validity_per_unit_displacement_over_rho1": sl,
            "budget_in_rho1_units": dose,
            "budget_absolute": dose * sub["median_rho1"],
            "budget_as_fraction_of_state_norm":
                dose * sub["median_rho1"] / sub["median_state_norm"],
        }
    res["D_row3_vs_row4"] = rows_out
    res["D_note"] = ("delta_novel equals delta_validity times the corner ceiling "
                     "exactly, on every leg, row and seed: with SEEN = 0 and every "
                     "valid tuple distinct, the scored metric IS the validity rate "
                     "rescaled. Explaining the per-request validity explains the "
                     "whole metric.")
    res["E_motion_budget"] = budget


def check_f(res, reps: int = 20):
    """Do the WRONG read-backs land on fitted combinations, against a local null?"""
    rng = np.random.default_rng(1234)
    out = {}
    for seed in (0, 1, 2):
        R.set_condition(seed, "test")
        cond = fw.load_condition()
        grid = cond["grid"]
        cards = np.array([int(c) for c in grid.cardinalities])
        gv = [np.asarray(v, np.float64) for v in grid.values]
        seen = fw.seen_tuple_mask(grid, cond["train_rows"])
        corners = fw.nway_corners(cond)
        per = {"grid_seen_fraction": float(seen.mean())}
        for arm in LEGS:
            cor = np.load(f"{SCORED}/test_seed{seed}/arms/{arm}/rows.npz",
                          allow_pickle=True)["corners"].item()
            T, V, I = [], [], []
            for c in corners:
                r = cor[c["corner_id"]]
                T.append(np.asarray(r["tuples"]))
                V.append(np.asarray(r["valid"], bool))
                vals = fw._request_values_from_rows(grid, c["rows"])
                ii = np.zeros(vals.shape, np.int64)
                for f in range(6):
                    ii[:, f] = np.abs(vals[:, f][:, None] - gv[f][None, :]).argmin(1)
                I.append(ii)
            T, V, I = np.concatenate(T), np.concatenate(V), np.concatenate(I)
            bad = ~V
            if not bad.any():
                continue
            Tb, Ib = T[bad], I[bad]
            h = (Tb != Ib).sum(1)
            obs = float(seen[np.ravel_multi_index(tuple(Tb.T), tuple(cards))].mean())
            acc = []
            for _ in range(reps):
                Tn = Ib.copy()
                for i in range(Tb.shape[0]):
                    k = int(h[i])
                    if k == 0:
                        continue
                    for f in rng.choice(6, size=k, replace=False):
                        ch = [v for v in range(cards[f]) if v != Ib[i, f]]
                        Tn[i, f] = ch[rng.integers(len(ch))]
                acc.append(float(
                    seen[np.ravel_multi_index(tuple(Tn.T), tuple(cards))].mean()))
            null = float(np.mean(acc))
            per[arm] = {"n_wrong": int(bad.sum()),
                        "mean_hamming_of_a_wrong_readback": float(h.mean()),
                        "seen_fraction_observed": obs,
                        "seen_fraction_locality_matched_null": null,
                        "ratio": obs / max(null, 1e-9)}
        out[str(seed)] = per
    res["F_regression_to_seen_combinations"] = out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P.install()
    res: dict = {}
    for fn in (check_a, check_b_c, check_d_e, check_f):
        fn(res)
        print(f"  {fn.__name__} done", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
