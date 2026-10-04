"""UNIT 5 -- THE RADIUS ATLAS.  Why the derived gate behaves as it does, in one table.

Three things, all DEV, none of them a new gate:
  1. THE ATLAS.  Where every population sits, in the gate's OWN units (d/sigma_k), against
     the three radii that matter: the empirical ZERO LEVEL SET, the CERTIFIED radius
     sigma_k*sqrt(2 ln N), and the rivals' FITTED q = 0.95 quantile.
  2. THRESHOLD-FREE AUROC.  Is the physics SIGNAL better than a standard readout margin?
     That is the paper's own negative finding and it is separable from where the threshold
     sits.  AUROC has no threshold, so it answers exactly that and nothing else.
  3. THE FITTED ALTERNATIVE, as SS5 permits: "A fitted alternative may be reported beside it,
     labelled, never adopted."  Threshold the SAME statistic max_k E_k at the q-quantile of
     val instead of at zero.  ⛔ REPORTED BESIDE. ⛔ NEVER ADOPTED. The gate remains E_k > 0.
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

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED, BUDGET, Q = 0, "A", 0.95
SEVERITIES = [2.0, 3.0, 4.0, 7.0, 13.0]
RNG = np.random.default_rng(20260917)

ctx = R.build_context(SEED)
built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, book, grid = built["land"], built["store"], built["book"], ctx["grid"]
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)
sig = np.asarray(land.sigma, np.float64)
rad = AL.certified_radius(land)
protos = FactorPrototypes.fit(grid, X_pool, pool)
log(f"K={store.n_wells} N={store.n_atoms} sigma={np.round(sig,4).tolist()}")

rec = {"schema": "admission-atlas/1", "seed": SEED, "budget": BUDGET,
       "sigma": [float(s) for s in sig],
       "certified_radius": [float(x) for x in rad],
       "certified_radius_over_sigma": [float(x) for x in rad / sig],
       "atlas": [], "zero_level_set": {}, "auroc": {}, "fitted_beside": {}}

# ---------------------------------------------------------------------------
# 1. THE EMPIRICAL ZERO LEVEL SET, per well: walk out from a stored atom
# ---------------------------------------------------------------------------
log("")
log("=== 1. THE EMPIRICAL ZERO LEVEL SET (walk out from each stored atom) ===")
N_DIR, N_T = 4, 400
zl = []
for k, a in enumerate(store.atoms):
    A = np.asarray(a, np.float64)
    N, dk = A.shape
    t = np.linspace(0.0, 5.0 * sig[k], N_T)
    r0 = []
    for i in range(N):
        for _ in range(N_DIR):
            u = RNG.normal(size=dk)
            u /= np.linalg.norm(u)
            P = A[i][None, :] + t[:, None] * u[None, :]
            d2 = ((P * P).sum(1)[:, None] - 2.0 * (P @ A.T) + (A * A).sum(1)[None, :])
            lse = np.log(np.exp(-np.maximum(d2, 0.0) / (2.0 * sig[k] ** 2)).sum(1))
            E = -(sig[k] ** 2) * lse
            c = np.flatnonzero(E > 0.0)
            r0.append(t[c[0]] / sig[k] if c.size else np.nan)
    r0 = np.asarray(r0, np.float64)
    row = {"well": k, "sigma": float(sig[k]),
           "zero_crossing_over_sigma_median": float(np.nanmedian(r0)),
           "zero_crossing_over_sigma_p05": float(np.nanpercentile(r0, 5)),
           "zero_crossing_over_sigma_p95": float(np.nanpercentile(r0, 95)),
           "certified_over_sigma": float(rad[k] / sig[k]),
           "n_rays": int(r0.size), "n_never_crossed": int(np.isnan(r0).sum())}
    zl.append(row)
    log(f"  well {k}: zero level set at d/σ = {row['zero_crossing_over_sigma_median']:.3f} "
        f"[p05 {row['zero_crossing_over_sigma_p05']:.3f}, p95 {row['zero_crossing_over_sigma_p95']:.3f}]"
        f"   certified at {row['certified_over_sigma']:.3f}")
rec["zero_level_set"] = {"per_well": zl,
                         "median_over_wells": float(np.median([r["zero_crossing_over_sigma_median"] for r in zl]))}

# ---------------------------------------------------------------------------
# 2. THE POPULATIONS
# ---------------------------------------------------------------------------
val_rows = np.asarray(ctx["val_rows"], np.int64)
order = np.argsort(pool); posv = order[np.searchsorted(pool[order], val_rows)]
req_val = requests_from_rows(grid, val_rows)
req_a7 = R.request_sets(ctx, {"A7"})["A7"]
cal_rows = np.asarray(ctx["fit_rows"], np.int64)
cal_rows = cal_rows[RNG.permutation(cal_rows.size)[:4096]]
X_cal = protos.embed_many(requests_from_rows(grid, cal_rows)).reshape(-1, store.dim)


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


pops = {}
pops["atoms (stored)"] = np.asarray(AC.atom_matrix(store), np.float32)
pops["real features: all store_train"] = X_pool[RNG.permutation(pool.size)[:4096]]
pops["real features: val rows"] = X_pool[posv]
lw_v, nt_v, _ = R.term_experts(book, grid, req_val)
pops["CLU launch: clamp_atom"] = AL.launch_from_terms_mode(land, book, lw_v, nt_v, "clamp_atom")["x0"]
pops["CLU launch: posterior_mean"] = AL.launch_from_terms_mode(land, book, lw_v, nt_v, "posterior_mean")["x0"]
pops["R8 affine: valid (val)"] = protos.embed_many(req_val).reshape(-1, store.dim)
pops["R8 affine: A7 (valid, redundant)"] = protos.embed_many(req_a7).reshape(-1, store.dim)
oor_by_sev = {}
for w in SEVERITIES:
    Xs = np.concatenate([protos.embed_many(perturb(req_val, f, offgrid(f, w))).reshape(-1, store.dim)
                         for f in range(len(names))], axis=0)
    oor_by_sev[w] = Xs
    pops[f"R8 affine: out_of_range w={w:g}"] = Xs

log("")
log("=== 2. THE RADIUS ATLAS -- everything in the gate's own units ===")
log(f"{'population':40s} {'n':>6s} {'min_k d/σ':>12s} {'max_k E_k':>12s} {'refuse':>8s}")
for nm, X in pops.items():
    g = AL.gate(land, X)
    ds = g["d_over_sigma"].min(1)
    row = {"population": nm, "n": int(X.shape[0]),
           "min_k_d_over_sigma_mean": float(ds.mean()),
           "min_k_d_over_sigma_median": float(np.median(ds)),
           "min_k_d_over_sigma_p95": float(np.percentile(ds, 95)),
           "max_k_E_mean": float(g["E"].max(1).mean()),
           "max_k_E_median": float(np.median(g["E"].max(1))),
           "refuse_rate": float(g["refuse"].mean()),
           "certified_fraction_of_refusals": (
               float(g["certified_refusal"].sum()) / max(int(g["refuse"].sum()), 1)
               if int(g["refuse"].sum()) else None)}
    rec["atlas"].append(row)
    log(f"{nm:40s} {row['n']:6d} {row['min_k_d_over_sigma_mean']:12.3f} "
        f"{row['max_k_E_mean']:12.4g} {row['refuse_rate']:8.4f}")

g_f = AC.KnnGate.fit(store, X_cal, "factored", Q)
g_w = AC.KnnGate.fit(store, X_cal, "whole", Q)
rec["fitted_q95_over_sigma"] = [float(t / s) for t, s in zip(g_f.thresholds, sig)]
log("")
log(f"  fitted factored-kNN q=0.95 thresholds / σ_k = "
    f"{[round(x,3) for x in rec['fitted_q95_over_sigma']]}")
log(f"  certified radius / σ_k                    = {[round(x,3) for x in rec['certified_radius_over_sigma']]}")
log(f"  empirical ZERO LEVEL SET / σ_k (median)   = "
    f"{[round(r['zero_crossing_over_sigma_median'],3) for r in zl]}")

# ---------------------------------------------------------------------------
# 3. THRESHOLD-FREE AUROC -- is the PHYSICS SIGNAL better than a readout margin?
# ---------------------------------------------------------------------------
def auroc(pos, neg):
    """P(score(pos) > score(neg)), ties at 0.5.  Higher score = more invalid."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    a = np.concatenate([pos, neg])
    r = np.empty(a.size, np.float64)
    o = np.argsort(a, kind="mergesort")
    s = a[o]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


X_oor = np.concatenate([oor_by_sev[w] for w in SEVERITIES], axis=0)
X_val_r8 = pops["R8 affine: valid (val)"]
X_a7_r8 = pops["R8 affine: A7 (valid, redundant)"]


def stats_of(X, reqs=None):
    g = AL.gate(land, X)
    out = {"CLU max_k E_k": g["E"].max(1),
           "kNN factored max_k d_k/thr_k": (g["d"] / g_f.thresholds[None, :]).max(1),
           "kNN whole-latent d": AC.whole_latent_dist(store, X)}
    return out


S_val, S_oor, S_a7 = stats_of(X_val_r8), stats_of(X_oor), stats_of(X_a7_r8)
lw_o, nt_o, _ = R.term_experts(book, grid, [q for f in range(len(names)) for w in SEVERITIES
                                            for q in perturb(req_val, f, offgrid(f, w))])
lw_a, nt_a, _ = R.term_experts(book, grid, req_a7)
P_val = AL.launch_from_terms_mode(land, book, lw_v, nt_v, "posterior_mean")["posterior"]
P_oor = AL.launch_from_terms_mode(land, book, lw_o, nt_o, "posterior_mean")["posterior"]
P_a7 = AL.launch_from_terms_mode(land, book, lw_a, nt_a, "posterior_mean")["posterior"]
S_val["margin M (physics-free)"] = -AC.head_margin(P_val)
S_oor["margin M (physics-free)"] = -AC.head_margin(P_oor)
S_a7["margin M (physics-free)"] = -AC.head_margin(P_a7)

log("")
log("=== 3. THRESHOLD-FREE AUROC (no threshold anywhere; higher = separates better) ===")
log(f"{'statistic':34s} {'oor vs val':>12s} {'oor vs A7':>12s}   Δ over margin M")
mbase = auroc(S_oor["margin M (physics-free)"], S_val["margin M (physics-free)"])
for s in ("CLU max_k E_k", "kNN factored max_k d_k/thr_k", "kNN whole-latent d", "margin M (physics-free)"):
    a1 = auroc(S_oor[s], S_val[s])
    a2 = auroc(S_oor[s], S_a7[s])
    rec["auroc"][s] = {"oor_vs_val": a1, "oor_vs_A7": a2, "delta_over_margin_M": a1 - mbase}
    log(f"{s:34s} {a1:12.4f} {a2:12.4f}   {a1-mbase:+.4f}")
rec["auroc_note"] = ("the paper's own negative finding is that the raw energy signal adds at most "
                     "+0.024 AUROC over a standard readout margin; this row is the same comparison "
                     "on this object")

# ---------------------------------------------------------------------------
# 4. THE FITTED ALTERNATIVE -- reported beside, ⛔ NEVER ADOPTED
# ---------------------------------------------------------------------------
log("")
log("=== 4. THE FITTED ALTERNATIVE (⛔ reported beside, NEVER adopted; the gate stays E_k > 0) ===")
thr = float(np.quantile(S_val["CLU max_k E_k"], Q))
acc = {"val": float((S_val["CLU max_k E_k"] <= thr).mean()),
       "A7": float((S_a7["CLU max_k E_k"] <= thr).mean()),
       "oor_refuse": float((S_oor["CLU max_k E_k"] > thr).mean())}
rec["fitted_beside"] = {"statistic": "max_k E_k", "threshold_rule": f"q={Q} quantile of val",
                        "threshold": thr, "derived_threshold": 0.0, **acc,
                        "score": 0.5 * (acc["oor_refuse"] + acc["val"])}
log(f"  fitted threshold on max_k E_k = {thr:.4f}   (⭐ the DERIVED threshold is 0.0)")
log(f"  accept_val={acc['val']:.4f}  accept_A7={acc['A7']:.4f}  refuse_oor={acc['oor_refuse']:.4f}  "
    f"score={rec['fitted_beside']['score']:.4f}")
log(f"  ⭐ for comparison the DERIVED gate gives accept_val=1.0000 accept_A7=0.8056 "
    f"refuse_oor=0.3714 score=0.6857")

json.dump(rec, open(OUT / "RADIUS_ATLAS_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'RADIUS_ATLAS_seed0_A.json'}")
