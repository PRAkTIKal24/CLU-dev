"""w6.1 retry -- IS THE COST OF SETTLING A SUBSPACE EXIT?

The parent measured that the settled read loses composition monotonically in how far
the state is allowed to move, and the retry's declared hypothesis is:

    settling moves the state OFF the per-well principal subspace span(G_k) that the
    content spans, while the read can only answer from inside that subspace, so every
    unit of displacement is a unit of signal thrown away.

This script TESTS that rather than asserting it.  For every request in the SCORED row-4
set it recomputes each leg's emitted state from the artifacts already on disk -- no
judge, no scoring, nothing new is run -- and decomposes it into

  * the component inside  span(G_k),
  * the component orthogonal to span(G_k),

and then asks whether the orthogonal fraction predicts the per-request loss.  It also
carries the alternative the hypothesis has to beat: WITHIN-span motion across the
ANSWER LATTICE, i.e. the displacement measured against the distance to the nearest
answer that differs in exactly one factor -- the resolution the judge has to resolve.

Everything it reads is an artifact of the parent's already-paid legs.  No judge is
constructed and nothing is scored.

    python scripts/w61_settle_subspace.py --leg DIR --seed S --out FILE.json
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

#: the legs that were SCORED on the held-out set, in displacement order
LEGS = ("pca_tensor", "pca_settle_zerostep", "pca_settle_f05_conv",
        "pca_settle_onestep", "pca_settle")


def load_arm(leg: Path, arm: str) -> dict:
    with open(leg / "arms" / arm / "arm.pkl", "rb") as fh:
        return pickle.load(fh)


def in_span_split(store, basis, X: np.ndarray) -> tuple:
    """(||P x||, ||(I-P) x||) per request; P projects onto span(G_k), well by well."""
    ins = np.zeros(X.shape[0], np.float64)
    out = np.zeros(X.shape[0], np.float64)
    for idx, U in zip(store.well_channels, basis):
        idx = np.asarray(idx, np.int64)
        Un = np.asarray(U, np.float64)
        xb = np.asarray(X[:, idx], np.float64)
        pr = (xb @ Un) @ Un.T
        ins += (pr ** 2).sum(1)
        out += ((xb - pr) ** 2).sum(1)
    return np.sqrt(ins), np.sqrt(out)


def factor_indices(vals: np.ndarray, grid) -> np.ndarray:
    out = np.zeros(vals.shape, np.int64)
    for f in range(vals.shape[1]):
        gv = np.asarray(grid.values[f], np.float64)
        out[:, f] = np.abs(vals[:, f][:, None] - gv[None, :]).argmin(1)
    return out


def hamming1_spacing(X: np.ndarray, I: np.ndarray) -> tuple:
    """Per request: the distance to the nearest answer differing in EXACTLY one factor,
    and to the nearest answer of any kind -- both on the INCUMBENT's own answers."""
    n, F = I.shape
    rho1 = np.full(n, np.inf)
    for f in range(F):
        key = np.delete(I, f, axis=1)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        for g in np.unique(inv):
            m = np.flatnonzero(inv == g)
            if m.size < 2:
                continue
            B = X[m].astype(np.float64)
            sq = np.sum(B * B, 1)
            d2 = sq[:, None] - 2.0 * (B @ B.T) + sq[None, :]
            np.fill_diagonal(d2, np.inf)
            rho1[m] = np.minimum(rho1[m], np.sqrt(np.maximum(d2.min(1), 0.0)))
    A = X.astype(np.float64)
    sq = np.sum(A * A, 1)
    d2 = sq[:, None] - 2.0 * (A @ A.T) + sq[None, :]
    np.fill_diagonal(d2, np.inf)
    rho_any = np.sqrt(np.maximum(d2.min(1), 0.0))
    return rho1, rho_any


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(score is higher on a positive than on a negative), ties counted at 0.5."""
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    o = np.argsort(score, kind="mergesort")
    s = np.asarray(score, np.float64)[o]
    rr = np.empty(s.size, np.float64)
    rr[o] = np.arange(1, s.size + 1, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            rr[o[i:j + 1]] = np.mean(rr[o[i:j + 1]])
        i = j + 1
    n1 = int(y.sum())
    n0 = int((~y).sum())
    return float((rr[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    leg = Path(a.leg)
    t0 = time.time()

    P.install()
    R.set_condition(a.seed, a.condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    corners = (fw.nway_corners(cond) if a.condition == "test"
               else fw.val_corners(cond))
    print(f"[{a.condition} seed {a.seed}] {len(corners)} corners", flush=True)

    arms = {}
    for arm in LEGS:
        if (leg / "arms" / arm / "arm.pkl").exists():
            arms[arm] = load_arm(leg, arm)
    store = arms["pca_tensor"]["store"]
    basis = arms["pca_tensor"]["basis"]
    print("  legs: " + ", ".join(arms), flush=True)

    rows = {arm: np.load(leg / "arms" / arm / "rows.npz",
                         allow_pickle=True)["corners"].item() for arm in arms}

    keys = ("disp", "off_state", "norm", "off_disp", "valid", "accept")
    rec = {arm: {k: [] for k in keys} for arm in arms}
    tup = {arm: [] for arm in arms}
    Xt_all, I_all, cid_all = [], [], []
    bit_identical = {}

    for ci, c in enumerate(corners):
        vals = fw._request_values_from_rows(grid, c["rows"])
        I = factor_indices(vals, grid)
        Xs = {}
        for arm, obj in arms.items():
            X, _oor = R.arm_features(obj, vals, grid)
            Xs[arm] = np.asarray(X, np.float32)
        Xt = Xs["pca_tensor"]
        nt = np.linalg.norm(Xt.astype(np.float64), axis=1)
        for arm in arms:
            X = Xs[arm]
            if arm != "pca_tensor":
                same = bool(np.array_equal(X, Xt))
                bit_identical[arm] = bit_identical.get(arm, True) and same
            ins, off = in_span_split(store, basis, X)
            D = X.astype(np.float64) - Xt.astype(np.float64)
            dn = np.linalg.norm(D, axis=1)
            _di, doff = in_span_split(store, basis, D)
            r = rows[arm][c["corner_id"]]
            rec[arm]["disp"].append(dn)
            rec[arm]["norm"].append(nt)
            rec[arm]["off_state"].append(off / np.maximum(ins, 1e-12))
            rec[arm]["off_disp"].append(doff / np.maximum(dn, 1e-12))
            rec[arm]["valid"].append(np.asarray(r["valid"], bool))
            rec[arm]["accept"].append(np.asarray(r["accept"], bool))
            tup[arm].append(np.asarray(r["tuples"], np.int64))
        Xt_all.append(Xt)
        I_all.append(I)
        cid_all.append(np.full(vals.shape[0], ci, np.int64))
        if ci % 10 == 0:
            print(f"  corner {ci}/{len(corners)}  {time.time() - t0:.0f}s", flush=True)

    Xt_all = np.concatenate(Xt_all, 0)
    I_all = np.concatenate(I_all, 0)
    cid = np.concatenate(cid_all, 0)
    for arm in rec:
        for k in keys:
            rec[arm][k] = np.concatenate(rec[arm][k], 0)
        tup[arm] = np.concatenate(tup[arm], 0)
    n = Xt_all.shape[0]
    print(f"  {n} requests; answer spacing ...", flush=True)
    rho1, rho_any = hamming1_spacing(Xt_all, I_all)
    print(f"  spacing done {time.time() - t0:.0f}s", flush=True)

    acc = rec["pca_tensor"]["accept"]
    vt = rec["pca_tensor"]["valid"]
    keep = acc & np.isfinite(rho1)
    out = {
        "leg": str(leg), "seed": a.seed, "condition": a.condition,
        "n_requests": int(n), "n_corners": len(corners),
        "n_analysed": int(keep.sum()), "accept_rate": float(acc.mean()),
        "bit_identical_to_pca_tensor": bit_identical,
        "rho1_finite_frac": float(np.isfinite(rho1).mean()),
        "median_rho1": float(np.median(rho1[np.isfinite(rho1)])),
        "median_rho_any": float(np.median(rho_any)),
        "median_state_norm": float(np.median(rec["pca_tensor"]["norm"])),
        "legs": {},
    }

    edges = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 1e9])
    for arm in arms:
        d = rec[arm]
        ratio = d["disp"] / np.maximum(rho1, 1e-12)
        rel = d["disp"] / np.maximum(d["norm"], 1e-12)
        inv = ~d["valid"]
        blk = {
            "validity": float(d["valid"][keep].mean()),
            "median_rel_displacement": float(np.median(rel[keep])),
            "median_abs_displacement": float(np.median(d["disp"][keep])),
            "median_off_span_fraction_of_state": float(np.median(d["off_state"][keep])),
            "max_off_span_fraction_of_state": float(np.max(d["off_state"][keep])),
            "median_off_span_fraction_of_displacement":
                float(np.median(d["off_disp"][keep])),
            "max_off_span_fraction_of_displacement": float(np.max(d["off_disp"][keep])),
            "median_displacement_over_rho1": float(np.median(ratio[keep])),
            "frac_displacement_over_half_rho1": float((ratio[keep] > 0.5).mean()),
            "auc_off_span_predicts_invalid": auc(d["off_state"][keep], inv[keep]),
            "auc_displacement_predicts_invalid": auc(d["disp"][keep], inv[keep]),
            "auc_ratio_predicts_invalid": auc(ratio[keep], inv[keep]),
            "flipped_valid_to_invalid": int((vt & ~d["valid"])[keep].sum()),
            "flipped_invalid_to_valid": int((~vt & d["valid"])[keep].sum()),
        }
        b = np.digitize(ratio[keep], edges) - 1
        vv = d["valid"][keep]
        blk["validity_by_ratio_bin"] = [
            {"lo": float(edges[i]), "hi": float(edges[i + 1]),
             "n": int((b == i).sum()),
             "validity": (float(vv[b == i].mean()) if (b == i).any() else None)}
            for i in range(len(edges) - 1)]
        out["legs"][arm] = blk

    percorner = {}
    for arm in arms:
        d = rec[arm]
        ratio = d["disp"] / np.maximum(rho1, 1e-12)
        lo, rr, oo = [], [], []
        for ci in range(len(corners)):
            m = keep & (cid == ci)
            lo.append(float(vt[m].mean() - d["valid"][m].mean()))
            rr.append(float(np.median(ratio[m])))
            oo.append(float(np.median(d["off_state"][m])))
        percorner[arm] = {"loss": lo, "median_ratio": rr, "median_off_span": oo}
    out["per_corner"] = percorner

    np.savez_compressed(
        Path(a.out).with_suffix(".npz"),
        cid=cid, rho1=rho1, rho_any=rho_any, I=I_all, accept=acc, valid_tensor=vt,
        **{f"{k}__{arm}": rec[arm][k] for arm in arms
           for k in ("disp", "off_state", "off_disp", "norm", "valid")},
        **{f"tuples__{arm}": tup[arm] for arm in arms})
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
