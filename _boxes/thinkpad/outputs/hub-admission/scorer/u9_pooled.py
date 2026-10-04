"""UNIT 9 -- the BAR's ACTUAL definition, and the last unmeasured query mode.

(a) ``out_of_support`` POOLED.  SS5 defines the pair on ``refuse_out_of_support`` where
    "out-of-support pools out-of-range AND out-of-vocabulary".  UNIT 3 reported
    ``refuse_oor`` alone.  The pooled number is the one the bar actually asks for, and
    because the CLU refuses 0.0000 of out-of-vocabulary it can only be WORSE for the arm.
    ⛔ Reported anyway, at full strength -- a bar evaluated on a favourable subset is not
    the bar.

(b) FREE MODE.  SS5.9 says "exemplar refusal dead, symbol and free modes continue", but
    free mode was never defined in a query set.  Taken here as its only sensible reading:
    a request that names only SOME of the six factors and leaves the rest free.  It is a
    VALID request -- an under-specified one -- so it must be ACCEPTED.
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

import admission_ctrl as AC         # noqa: E402
import admission_lib as AL          # noqa: E402
import refusal_lib as R             # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import factor_labels  # noqa: E402
from chlu.experiments.exp_gen_dynamical_read import build_landscape  # noqa: E402

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED, BUDGET, Q = 0, "A", 0.95
SEVERITIES = [2.0, 3.0, 4.0, 7.0, 13.0]
RNG = np.random.default_rng(20260917)

ctx = R.build_context(SEED)
b = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, book, grid = b["land"], b["store"], b["book"], ctx["grid"]
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)
F_pool = np.asarray(grid.factor_indices_of(pool), np.int64)
wc = ctx["well_channels"]
protos = FactorPrototypes.fit(grid, X_pool, pool)
req_val = requests_from_rows(grid, np.asarray(ctx["val_rows"], np.int64))
cal_rows = np.asarray(ctx["fit_rows"], np.int64)
X_cal = protos.embed_many(requests_from_rows(
    grid, cal_rows[RNG.permutation(cal_rows.size)[:4096]])).reshape(-1, store.dim)
g_f = AC.KnnGate.fit(store, X_cal, "factored", Q)
g_w = AC.KnnGate.fit(store, X_cal, "whole", Q)
fw = json.load(open(OUT / "FACTOR_WELL_MAP_seed0.json"))
rec = {"schema": "admission-pooled/1", "seed": SEED, "budget": BUDGET, "q": Q}


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


# ---------------------------------------------------------------------------
# (b) FREE MODE -- drop n of the six terms.  A valid, under-specified request.
# ---------------------------------------------------------------------------
log("=== (b) FREE MODE: a request naming only some factors (valid, under-specified) ===")
rec["free_mode"] = []
for n_free in (1, 2, 3):
    out = []
    for i, r in enumerate(req_val):
        q = copy.deepcopy(r)
        drop = RNG.permutation(len(names))[:n_free]
        q["request"] = [t for t in q["request"] if int(t["factor_index"]) not in set(drop.tolist())]
        q.pop("class_indices", None)
        out.append(q)
    Xf = protos.embed_many(out).reshape(-1, store.dim)
    g = AL.gate(land, Xf)
    row = {"n_factors_free": n_free, "n_terms": len(names) - n_free,
           "CLU_accept": float(g["accept"].mean()),
           "CLU_d_over_sigma": float(g["d_over_sigma"].min(1).mean()),
           "knn_factored_accept": float(g_f.decide(store, Xf)["accept"].mean()),
           "knn_whole_accept": float(g_w.decide(store, Xf)["accept"].mean())}
    rec["free_mode"].append(row)
    log(f"  {n_free} free ({row['n_terms']} terms): CLU accept={row['CLU_accept']:.4f} "
        f"d/σ={row['CLU_d_over_sigma']:.3f} | knn_factored accept={row['knn_factored_accept']:.4f} "
        f"| knn_whole accept={row['knn_whole_accept']:.4f}")

# ---------------------------------------------------------------------------
# (a) out_of_support POOLED = out_of_range  U  out_of_vocabulary
# ---------------------------------------------------------------------------
log("")
log("=== (a) out_of_support POOLED (out_of_range ∪ out_of_vocabulary), the BAR's definition ===")
req_oor = [q for f in range(len(names)) for w in SEVERITIES for q in perturb(req_val, f, offgrid(f, w))]
X_oor = protos.embed_many(req_oor).reshape(-1, store.dim)
ref = {"CLU": AL.gate(land, X_oor)["refuse"],
       "knn_factored": g_f.decide(store, X_oor)["refuse"],
       "knn_whole": g_w.decide(store, X_oor)["refuse"]}
log(f"  out_of_range  n={X_oor.shape[0]:6d}  CLU={ref['CLU'].mean():.4f} "
    f"knn_factored={ref['knn_factored'].mean():.4f} knn_whole={ref['knn_whole'].mean():.4f}")

# OOV, symbol + exemplar, on the six held-out-value stores (rebuilt; shortfall must be 0)
oov_sym, oov_ex = {k: [] for k in ref}, {k: [] for k in ref}
for f, nm in enumerate(names):
    v_idx = cards[f] // 2
    v_val = float(np.asarray(grid.values[f], np.float64)[v_idx])
    wells = list(fw["map"][nm]["assigned_wells"])
    keep = F_pool[:, f] != v_idx
    Xk, Pk = X_pool[keep], pool[keep]
    Lr = build_landscape(ctx["config"], BUDGET, Xk, factor_labels(grid, Pk), wc,
                         ctx["feat_shape"], cards, SEED, log=lambda *a, **k: None)["land"]
    pr = FactorPrototypes.fit(grid, Xk, Pk)
    calr = pr.embed_many(requests_from_rows(grid, Pk[RNG.permutation(Pk.size)[:2048]])).reshape(-1, Lr.store.dim)
    gfr, gwr = AC.KnnGate.fit(Lr.store, calr, "factored", Q), AC.KnnGate.fit(Lr.store, calr, "whole", Q)
    # symbol
    Xs = pr.embed_many(perturb(req_val, f, v_val)).reshape(-1, Lr.store.dim)
    # exemplar: real cached features of held-out images in the factor's wells
    hi = np.flatnonzero(~keep)
    hi = hi[RNG.permutation(hi.size)[:512]]
    rq = requests_from_rows(grid, pool[hi])
    Xe = pr.embed_many(rq).reshape(-1, Lr.store.dim).copy()
    for k in wells:
        c = np.asarray(wc[k], np.int64)
        Xe[:, c] = X_pool[hi][:, c]
    for tag, Xq, acc in (("sym", Xs, oov_sym), ("ex", Xe, oov_ex)):
        acc["CLU"].append(AL.gate(Lr, Xq)["refuse"])
        acc["knn_factored"].append(gfr.decide(Lr.store, Xq)["refuse"])
        acc["knn_whole"].append(gwr.decide(Lr.store, Xq)["refuse"])
    log(f"  {nm:12s} oov_symbol CLU={oov_sym['CLU'][-1].mean():.4f} "
        f"knnF={oov_sym['knn_factored'][-1].mean():.4f} | oov_exemplar CLU={oov_ex['CLU'][-1].mean():.4f} "
        f"knnF={oov_ex['knn_factored'][-1].mean():.4f}")

rec["pooled"] = {}
log("")
log(f"{'gate':16s} {'oor':>9s} {'oov_sym':>9s} {'oov_exem':>9s} {'POOLED out_of_support':>23s}")
for k in ref:
    a = ref[k]
    s = np.concatenate(oov_sym[k]); e = np.concatenate(oov_ex[k])
    p = np.concatenate([a, s, e])
    rec["pooled"][k] = {"refuse_oor": float(a.mean()), "n_oor": int(a.size),
                        "refuse_oov_symbol": float(s.mean()), "n_oov_symbol": int(s.size),
                        "refuse_oov_exemplar": float(e.mean()), "n_oov_exemplar": int(e.size),
                        "refuse_out_of_support_POOLED": float(p.mean()), "n_pooled": int(p.size)}
    log(f"{k:16s} {a.mean():9.4f} {s.mean():9.4f} {e.mean():9.4f} {p.mean():23.4f}")

# the pair, on the BAR's own pooled definition (accept population = val, DEV proxy for B)
acc_arm = AL.gate(land, protos.embed_many(req_val).reshape(-1, store.dim))["accept"]
X_val = protos.embed_many(req_val).reshape(-1, store.dim)
rec["R1_pooled"] = {}
for riv in ("knn_factored", "knn_whole"):
    gk = g_f if riv == "knn_factored" else g_w
    p_arm = np.concatenate([ref["CLU"], np.concatenate(oov_sym["CLU"]), np.concatenate(oov_ex["CLU"])])
    p_riv = np.concatenate([ref[riv], np.concatenate(oov_sym[riv]), np.concatenate(oov_ex[riv])])
    pr_ = AC.paired_bootstrap(acc_arm, gk.decide(store, X_val)["accept"], p_arm, p_riv)
    rec["R1_pooled"][riv] = pr_
    log(f"  R1 on the POOLED pair vs {riv}: Δ={pr_['delta']:+.4f} "
        f"CI95=[{pr_['ci95'][0]:+.4f},{pr_['ci95'][1]:+.4f}] -> {pr_['verdict_R1'].upper()}")

json.dump(rec, open(OUT / "POOLED_SUPPORT_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'POOLED_SUPPORT_seed0_A.json'}")
