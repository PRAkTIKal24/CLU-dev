"""ADVERSARIAL VERIFICATION -- claim 4: is the factored kNN rival FAIRLY treated by the
"refuse if ANY of 5 wells exceeds its own q = 0.95 column quantile" reduction?

The reduction makes the rival's realised refusal rate on VALID data ~21 %, not 5 %.  Two
questions, both answered with numbers and neither with an opinion:

  (A) OPERATING POINT.  Is q = 0.95 per column near the rival's best score, or is the
      rival being run at a handicapped point?  Swept over q, and over a family-wise
      calibration q* = 0.95^(1/K) that makes the ANY-of-K refusal rate 5 %.
  (B) OPERATING-POINT-FREE.  The maximum over ALL thresholds of
      score = 1/2 (refuse_oor + accept_val), for every statistic -- the best any gate of
      that family could do.  This removes the threshold from the comparison entirely and
      is the right way to ask whether the CLU statistic loses because of its RADIUS or
      because of its SIGNAL.

WRITES: VERIFY_FAIRNESS.json under $HUB_OUT.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-admission"
sys.path.insert(0, str(OUT / "scorer"))
sys.path.insert(0, os.path.join(os.environ["CHLU_HUB_OUTPUTS"], "hub-refusal", "scorer"))

import refusal_lib as R  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402

T0 = time.time()


def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


SEED, BUDGET = 0, "A"
SEVERITIES = [2.0, 3.0, 4.0, 7.0, 13.0]

ctx = R.build_context(SEED)
built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, grid = built["land"], built["store"], ctx["grid"]
names = list(grid.names)
pool = np.asarray(ctx["pool"], np.int64)
WC = [np.asarray(w, np.int64) for w in store.well_channels]
ATOMS = [np.asarray(a, np.float64) for a in store.atoms]
SIG = np.asarray(land.sigma, np.float64)
K = len(ATOMS)
Z = np.zeros((store.n_atoms, store.dim), np.float64)
for k, cols in enumerate(WC):
    Z[:, cols] = ATOMS[k]
protos = FactorPrototypes.fit(grid, ctx["X_store"], ctx["pool"])


def my_E_and_d(X, chunk=4096):
    X = np.asarray(X, np.float64)
    n = X.shape[0]
    E = np.empty((n, K))
    D = np.empty((n, K))
    for k in range(K):
        A, cols, s2 = ATOMS[k], WC[k], float(SIG[k]) ** 2
        aa = (A * A).sum(1)[None, :]
        for i in range(0, n, chunk):
            z = X[i:i + chunk][:, cols]
            d2 = np.maximum((z * z).sum(1)[:, None] - 2.0 * (z @ A.T) + aa, 0.0)
            m = d2.min(1)
            S = np.exp(-(d2 - m[:, None]) / (2.0 * s2)).sum(1)
            E[i:i + chunk, k] = 0.5 * m - s2 * np.log(S)
            D[i:i + chunk, k] = np.sqrt(m)
    return E, D


def whole_d(X, chunk=4096):
    X = np.asarray(X, np.float64)
    out = np.empty(X.shape[0])
    aa = (Z * Z).sum(1)[None, :]
    for i in range(0, X.shape[0], chunk):
        q = X[i:i + chunk]
        out[i:i + q.shape[0]] = np.sqrt(np.maximum(
            ((q * q).sum(1)[:, None] - 2.0 * (q @ Z.T) + aa).min(1), 0.0))
    return out


def offgrid(f, w):
    v = np.asarray(grid.values[f], np.float64)
    return float(v[-2] + w * (v[-1] - v[-2]))


def perturb(reqs, f, val):
    out = []
    for r in reqs:
        q = copy.deepcopy(r)
        for tt in q["request"]:
            if int(tt["factor_index"]) == f:
                tt["value"] = float(val)
        q.pop("class_indices", None)
        out.append(q)
    return out


val_rows = np.asarray(ctx["val_rows"], np.int64)
req_val = requests_from_rows(grid, val_rows)
cal_rows = np.asarray(ctx["fit_rows"], np.int64)
cal_rows = cal_rows[np.random.default_rng(20260917).permutation(cal_rows.size)[:4096]]
X_val = np.asarray(protos.embed_many(req_val), np.float64).reshape(-1, store.dim)
X_cal = np.asarray(protos.embed_many(requests_from_rows(grid, cal_rows)),
                   np.float64).reshape(-1, store.dim)
X_oor = np.concatenate([np.asarray(protos.embed_many(perturb(req_val, f, offgrid(f, w))),
                                   np.float64).reshape(-1, store.dim)
                        for f in range(len(names)) for w in SEVERITIES], axis=0)
X_a7 = np.asarray(protos.embed_many(R.request_sets(ctx, {"A7"})["A7"]),
                  np.float64).reshape(-1, store.dim)
log(f"val={X_val.shape[0]} cal={X_cal.shape[0]} oor={X_oor.shape[0]} a7={X_a7.shape[0]}")

E_val, D_val = my_E_and_d(X_val)
E_oor, D_oor = my_E_and_d(X_oor)
E_cal, D_cal = my_E_and_d(X_cal)
E_a7, D_a7 = my_E_and_d(X_a7)
dw_val, dw_oor, dw_cal = whole_d(X_val), whole_d(X_oor), whole_d(X_cal)

V = {"schema": "verify-admission-fairness/1", "K": K}

# ---------------------------------------------------------------------------
# (A) the operating point
# ---------------------------------------------------------------------------
log("")
log("=== (A) the factored rival's operating point ===")
rows = []
qstar = 0.95 ** (1.0 / K)
for qq in sorted({0.90, 0.95, 0.97, 0.98, 0.985, qstar, 0.99, 0.995, 0.999}):
    thr = np.quantile(D_cal, qq, axis=0)
    av = float((~(D_val > thr[None, :]).any(1)).mean())
    ro = float((D_oor > thr[None, :]).any(1).mean())
    a7 = float((~(D_a7 > thr[None, :]).any(1)).mean())
    rows.append({"q_per_column": float(qq), "familywise_refuse_on_cal":
                 float((D_cal > thr[None, :]).any(1).mean()),
                 "accept_val": av, "refuse_oor": ro, "accept_A7": a7,
                 "score": 0.5 * (ro + av),
                 "is_the_hubs_choice": bool(abs(qq - 0.95) < 1e-12),
                 "is_familywise_5pct": bool(abs(qq - qstar) < 1e-12)})
    log(f"  q={qq:<8.5f} fw_refuse_cal={rows[-1]['familywise_refuse_on_cal']:.4f} "
        f"accept_val={av:.4f} refuse_oor={ro:.4f} accept_A7={a7:.4f} "
        f"score={rows[-1]['score']:.4f}")
V["rival_q_sweep"] = rows
best = max(rows, key=lambda r: r["score"])
hub = [r for r in rows if r["is_the_hubs_choice"]][0]
fw5 = [r for r in rows if r["is_familywise_5pct"]][0]
V["rival_at_hubs_q095"] = hub
V["rival_best_over_q"] = best
V["rival_at_familywise_5pct"] = fw5
log(f"  hub's q=0.95 score={hub['score']:.4f}   best over q = {best['score']:.4f} "
    f"at q={best['q_per_column']:.5f}   family-wise-5% q*={qstar:.5f} "
    f"score={fw5['score']:.4f}")

# ---------------------------------------------------------------------------
# (B) operating-point-free: the BEST score any threshold of each family can reach
# ---------------------------------------------------------------------------
log("")
log("=== (B) the best score reachable by ANY threshold, per statistic ===")


def best_score(s_val, s_oor):
    """max over t of 1/2[ P(s_oor > t) + P(s_val <= t) ]."""
    s_val = np.asarray(s_val, np.float64)
    s_oor = np.asarray(s_oor, np.float64)
    cand = np.unique(np.concatenate([s_val, s_oor]))
    if cand.size > 20000:
        cand = np.quantile(cand, np.linspace(0, 1, 20000))
    sv = np.sort(s_val)
    so = np.sort(s_oor)
    acc = np.searchsorted(sv, cand, side="right") / sv.size
    ref = 1.0 - np.searchsorted(so, cand, side="right") / so.size
    sc = 0.5 * (acc + ref)
    i = int(np.argmax(sc))
    return {"best_score": float(sc[i]), "threshold": float(cand[i]),
            "accept_val_at_best": float(acc[i]), "refuse_oor_at_best": float(ref[i])}


thr95 = np.quantile(D_cal, 0.95, axis=0)
stats = {
    "CLU max_k E_k  (the arm's statistic)": (E_val.max(1), E_oor.max(1)),
    "CLU max_k d_k/sigma_k": ((D_val / SIG).max(1), (D_oor / SIG).max(1)),
    "knn_factored max_k d_k/thr_k (the rival)": ((D_val / thr95).max(1), (D_oor / thr95).max(1)),
    "knn_whole latent d": (dw_val, dw_oor),
}
V["best_achievable_score"] = {}
for nm, (a, b) in stats.items():
    r = best_score(a, b)
    V["best_achievable_score"][nm] = r
    log(f"  {nm:42s} best score = {r['best_score']:.4f} "
        f"(accept_val {r['accept_val_at_best']:.4f}, refuse_oor {r['refuse_oor_at_best']:.4f})")

V["arm_at_the_DERIVED_zero"] = {
    "accept_val": float((E_val.max(1) <= 0).mean()),
    "refuse_oor": float((E_oor.max(1) > 0).mean()),
    "accept_A7": float((E_a7.max(1) <= 0).mean()),
    "score": 0.5 * (float((E_oor.max(1) > 0).mean()) + float((E_val.max(1) <= 0).mean()))}
log(f"  the ARM at the derived zero: {V['arm_at_the_DERIVED_zero']}")

# how much of the arm's loss is the RADIUS and how much is the SIGNAL?
V["decomposition_of_the_loss"] = {
    "arm_score_at_derived_zero": V["arm_at_the_DERIVED_zero"]["score"],
    "arm_best_score_same_statistic": V["best_achievable_score"]["CLU max_k E_k  (the arm's statistic)"]["best_score"],
    "rival_score_at_hubs_q095": hub["score"],
    "rival_best_score": V["best_achievable_score"]["knn_factored max_k d_k/thr_k (the rival)"]["best_score"],
    "loss_attributable_to_the_RADIUS":
        V["best_achievable_score"]["CLU max_k E_k  (the arm's statistic)"]["best_score"]
        - V["arm_at_the_DERIVED_zero"]["score"],
    "loss_attributable_to_the_SIGNAL":
        V["best_achievable_score"]["knn_factored max_k d_k/thr_k (the rival)"]["best_score"]
        - V["best_achievable_score"]["CLU max_k E_k  (the arm's statistic)"]["best_score"],
    "hub_reported_delta": -0.1383}
log("")
log(f"  loss decomposition: {json.dumps(V['decomposition_of_the_loss'], indent=2)}")

json.dump(V, open(OUT / "VERIFY_FAIRNESS.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'VERIFY_FAIRNESS.json'}")
