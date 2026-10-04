"""ADVERSARIAL VERIFICATION -- claims 2, 3, 4, 5.

Everything here is written from the FORMULA in PREREG-1 SS1, in float64 numpy, WITHOUT
calling admission_lib.  The shipped landscape object is used only as a container for the
atoms and the channel partition; every number is recomputed here.

    E_k(x) = -sigma_k^2 * log sum_i exp(-||x_k - a_ki||^2 / (2 sigma_k^2))

Stable form used here (algebraically identical, no library):
    let m = min_i d2_i;  S = sum_i exp(-(d2_i - m)/(2 s^2))  (>= 1)
    E_k = m/2 - s^2 * log(S)
so E_k > 0  <=>  m/(2 s^2) > log S,  and log S <= log N gives the certified radius
sqrt(2 log N) * sigma.  The sqrt, the factor 2 and the sign are all visible here.

WRITES: VERIFY_CORE.json under $HUB_OUT.  READ-ONLY everywhere else.
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

import admission_ctrl as AC          # noqa: E402  (only for the RIVAL, claim 4)
import admission_lib as AL           # noqa: E402  (only to CROSS-CHECK against)
import refusal_lib as R              # noqa: E402
from chlu.core.gen_landscape import store_energy  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402

T0 = time.time()


def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


SEED, BUDGET, Q = 0, "A", 0.95
SEVERITIES = [2.0, 3.0, 4.0, 7.0, 13.0]
RNG = np.random.default_rng(777001)

V = {"schema": "verify-admission-core/1", "seed": SEED, "budget": BUDGET}

ctx = R.build_context(SEED)
built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, book, grid = built["land"], built["store"], built["book"], ctx["grid"]
names = list(grid.names)
pool = np.asarray(ctx["pool"], np.int64)
X_raw = ctx["X_store"]
V["X_store_dtype"] = str(np.asarray(X_raw).dtype)
V["cache_meta"] = ctx["cache_meta"]
X_pool = np.asarray(X_raw, np.float64).reshape(pool.size, -1)      # float64 from here
WC = [np.asarray(w, np.int64) for w in store.well_channels]
ATOMS = [np.asarray(a, np.float64) for a in store.atoms]
K = len(ATOMS)
log(f"K={K} N={store.n_atoms} dim={store.dim} X dtype={V['X_store_dtype']}")


# ---------------------------------------------------------------------------
# 0. sigma_k, recomputed from the atoms in float64 (nn_median, frac 1.0)
# ---------------------------------------------------------------------------
def my_nn_median(A: np.ndarray) -> float:
    g = (A * A).sum(1)
    d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * (A @ A.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(1))))


sig_mine = np.asarray([my_nn_median(A) for A in ATOMS], np.float64)
sig_his = np.asarray(land.sigma, np.float64)
V["sigma_mine"] = sig_mine.tolist()
V["sigma_his"] = sig_his.tolist()
V["sigma_max_rel_delta"] = float(np.max(np.abs(sig_mine - sig_his) / sig_his))
log(f"sigma mine = {np.round(sig_mine,4).tolist()}")
log(f"sigma his  = {np.round(sig_his,4).tolist()}  max rel delta = {V['sigma_max_rel_delta']:.3e}")
# the gate is his sigma; use HIS so any disagreement below is not a sigma disagreement
SIG = sig_his


# ---------------------------------------------------------------------------
# 1. my own E_k and d_k, float64, from the formula
# ---------------------------------------------------------------------------
def my_E_and_d(X, atoms=ATOMS, wc=WC, sig=SIG, chunk=2048):
    X = np.asarray(X, np.float64)
    n = X.shape[0]
    E = np.empty((n, len(atoms)), np.float64)
    D = np.empty((n, len(atoms)), np.float64)
    for k, (A, cols) in enumerate(zip(atoms, wc)):
        s2 = float(sig[k]) ** 2
        aa = (A * A).sum(1)[None, :]
        for i in range(0, n, chunk):
            z = X[i:i + chunk][:, cols]
            d2 = np.maximum((z * z).sum(1)[:, None] - 2.0 * (z @ A.T) + aa, 0.0)
            m = d2.min(1)
            S = np.exp(-(d2 - m[:, None]) / (2.0 * s2)).sum(1)
            E[i:i + chunk, k] = 0.5 * m - s2 * np.log(S)
            D[i:i + chunk, k] = np.sqrt(m)
    return E, D


def my_gate(X):
    E, D = my_E_and_d(X)
    return {"E": E, "d": D, "dos": D / SIG[None, :],
            "refuse": (E > 0.0).any(1), "accept": ~(E > 0.0).any(1)}


# cross-check mine vs his vs the shipped store_energy
probe = X_pool[RNG.permutation(pool.size)[:1024]]
E_mine, D_mine = my_E_and_d(probe)
E_his = AL.per_well_energy(land, probe.astype(np.float32))
D_his = AL.per_well_nearest(land, probe.astype(np.float32))
ref = np.asarray(store_energy(land, np.asarray(probe, np.float32)), np.float64)
V["crosscheck"] = {
    "max_abs_delta_E_mine_vs_his": float(np.max(np.abs(E_mine - E_his))),
    "max_rel_delta_E_mine_vs_his": float(np.max(np.abs(E_mine - E_his) / np.maximum(np.abs(E_his), 1.0))),
    "max_abs_delta_d_mine_vs_his": float(np.max(np.abs(D_mine - D_his))),
    "max_rel_delta_sumE_mine_vs_shipped_store_energy":
        float(np.max(np.abs(E_mine.sum(1) - ref) / np.maximum(np.abs(ref), 1.0))),
    "refuse_disagreements": int(((E_mine > 0).any(1) != (E_his > 0).any(1)).sum()),
}
log(f"cross-check: {V['crosscheck']}")

# the identity, my own arithmetic
ident = []
for k, A in enumerate(ATOMS):
    s2 = float(SIG[k]) ** 2
    g = (A * A).sum(1)
    d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * (A @ A.T), 0.0)
    m = d2.min(1)          # 0, self
    S = np.exp(-(d2 - m[:, None]) / (2.0 * s2)).sum(1)
    Ek = 0.5 * m - s2 * np.log(S)
    ident.append({"well": k, "max_E_at_atoms": float(Ek.max()),
                  "certified_radius_over_sigma": float(np.sqrt(2.0 * np.log(A.shape[0])))})
V["identity_mine"] = {"per_well": ident,
                      "max_over_wells": float(max(r["max_E_at_atoms"] for r in ident)),
                      "holds": bool(max(r["max_E_at_atoms"] for r in ident) <= 0.0)}
log(f"identity (mine): worst max_i E_k(a_ki) = {V['identity_mine']['max_over_wells']:.4f}")


# ---------------------------------------------------------------------------
# 2. CLAIM 2 -- where the data sits, and where the zero level set sits
# ---------------------------------------------------------------------------
protos = FactorPrototypes.fit(grid, ctx["X_store"], ctx["pool"])
val_rows = np.asarray(ctx["val_rows"], np.int64)
order = np.argsort(pool)
posv = order[np.searchsorted(pool[order], val_rows)]
req_val = requests_from_rows(grid, val_rows)
req_a7 = R.request_sets(ctx, {"A7"})["A7"]


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
Z = np.zeros((store.n_atoms, store.dim), np.float64)
for k, cols in enumerate(WC):
    Z[:, cols] = ATOMS[k]
pops["stored atoms"] = Z
pops["real features: store_train (4096)"] = X_pool[RNG.permutation(pool.size)[:4096]]
pops["real features: val rows"] = X_pool[posv]
pops["R8 affine: val"] = np.asarray(protos.embed_many(req_val), np.float64).reshape(-1, store.dim)
pops["R8 affine: A7"] = np.asarray(protos.embed_many(req_a7), np.float64).reshape(-1, store.dim)
oor = {}
for w in SEVERITIES:
    Xs = np.concatenate([np.asarray(protos.embed_many(perturb(req_val, f, offgrid(f, w))),
                                    np.float64).reshape(-1, store.dim)
                         for f in range(len(names))], axis=0)
    oor[w] = Xs
    pops[f"R8 affine: oor w={w:g}"] = Xs

log("")
log("=== CLAIM 2 -- the atlas, recomputed independently (float64) ===")
V["atlas_mine"] = []
for nm, X in pops.items():
    g = my_gate(X)
    row = {"population": nm, "n": int(X.shape[0]),
           "min_k_d_over_sigma_mean": float(g["dos"].min(1).mean()),
           "min_k_d_over_sigma_median": float(np.median(g["dos"].min(1))),
           "mean_d_over_sigma_all_wells": float(g["dos"].mean()),
           "max_k_E_mean": float(g["E"].max(1).mean()),
           "refuse_rate": float(g["refuse"].mean())}
    V["atlas_mine"].append(row)
    log(f"{nm:38s} n={row['n']:6d}  min_k d/s = {row['min_k_d_over_sigma_mean']:.4f}  "
        f"max_k E = {row['max_k_E_mean']:10.4g}  refuse = {row['refuse_rate']:.4f}")

# --- the zero level set, my own bisection along random rays from stored atoms
log("")
log("=== CLAIM 2 -- the zero level set, my own bisection (256 rays/well) ===")


def E_of_points(P, A, s):
    s2 = float(s) ** 2
    aa = (A * A).sum(1)[None, :]
    d2 = np.maximum((P * P).sum(1)[:, None] - 2.0 * (P @ A.T) + aa, 0.0)
    m = d2.min(1)
    S = np.exp(-(d2 - m[:, None]) / (2.0 * s2)).sum(1)
    return 0.5 * m - s2 * np.log(S), np.sqrt(m)


zl = []
rng2 = np.random.default_rng(424242)
for k, A in enumerate(ATOMS):
    s = float(SIG[k])
    cross = []
    for i in range(A.shape[0]):
        for _ in range(4):
            u = rng2.normal(size=A.shape[1])
            u /= np.linalg.norm(u)
            lo, hi = 0.0, 6.0 * s
            Ehi, _ = E_of_points((A[i] + hi * u)[None, :], A, s)
            if Ehi[0] <= 0.0:
                cross.append(np.nan)
                continue
            for _ in range(60):                    # pure bisection, no grid
                mid = 0.5 * (lo + hi)
                Em, _ = E_of_points((A[i] + mid * u)[None, :], A, s)
                if Em[0] > 0.0:
                    hi = mid
                else:
                    lo = mid
            # d/sigma at the crossing = distance to the NEAREST atom, not to a[i]
            _, dmin = E_of_points((A[i] + hi * u)[None, :], A, s)
            cross.append(float(dmin[0]) / s)
    cross = np.asarray(cross, np.float64)
    zl.append({"well": k, "zero_crossing_over_sigma_median": float(np.nanmedian(cross)),
               "p05": float(np.nanpercentile(cross, 5)), "p95": float(np.nanpercentile(cross, 95)),
               "n_rays": int(cross.size), "n_never_crossed": int(np.isnan(cross).sum()),
               "certified_over_sigma": float(np.sqrt(2.0 * np.log(A.shape[0])))})
    log(f"  well {k}: zero level set d/sigma = {zl[-1]['zero_crossing_over_sigma_median']:.3f} "
        f"[p05 {zl[-1]['p05']:.3f} p95 {zl[-1]['p95']:.3f}]  certified {zl[-1]['certified_over_sigma']:.3f}")
V["zero_level_set_mine"] = zl

# --- and the crossing IN THE DIRECTION OF THE DATA (the fair version of the same test)
log("")
log("=== CLAIM 2b -- zero crossing along the ray atom -> REAL val feature (data direction) ===")
Xv = X_pool[posv]
dat = []
for k, A in enumerate(ATOMS):
    s = float(SIG[k])
    zq = Xv[:, WC[k]]
    aa = (A * A).sum(1)[None, :]
    d2 = np.maximum((zq * zq).sum(1)[:, None] - 2.0 * (zq @ A.T) + aa, 0.0)
    nn = d2.argmin(1)
    fac = []
    for i in range(0, zq.shape[0], 64):
        sl = slice(i, min(i + 64, zq.shape[0]))
        a0 = A[nn[sl]]
        dirv = zq[sl] - a0
        nrm = np.linalg.norm(dirv, axis=1, keepdims=True)
        dirv = dirv / np.maximum(nrm, 1e-12)
        lo = np.zeros(dirv.shape[0])
        hi = np.full(dirv.shape[0], 8.0 * s)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            P = a0 + mid[:, None] * dirv
            Em, _ = E_of_points(P, A, s)
            pos = Em > 0.0
            hi = np.where(pos, mid, hi)
            lo = np.where(pos, lo, mid)
        P = a0 + hi[:, None] * dirv
        _, dmin = E_of_points(P, A, s)
        ok = hi < 8.0 * s - 1e-6
        fac.append(np.where(ok, dmin / s, np.nan))
    fac = np.concatenate(fac)
    dat.append({"well": k, "median_zero_crossing_over_sigma_data_direction": float(np.nanmedian(fac)),
                "n_never_crossed": int(np.isnan(fac).sum())})
    log(f"  well {k}: data-direction crossing d/sigma = "
        f"{dat[-1]['median_zero_crossing_over_sigma_data_direction']:.3f} "
        f"(never crossed: {dat[-1]['n_never_crossed']}/{fac.size})")
V["zero_level_set_data_direction"] = dat

# --- float16 sensitivity: does casting the cache differently move anything?
g32 = my_gate(np.asarray(X_pool[posv], np.float32).astype(np.float64))
g16 = my_gate(np.asarray(np.asarray(X_raw).reshape(pool.size, -1)[posv], np.float16).astype(np.float64))
V["float_cast_check"] = {
    "val_min_k_dos_from_float32": float(g32["dos"].min(1).mean()),
    "val_min_k_dos_from_float16": float(g16["dos"].min(1).mean()),
    "refuse_f32": float(g32["refuse"].mean()), "refuse_f16": float(g16["refuse"].mean())}
log(f"float cast check: {V['float_cast_check']}")

# ---------------------------------------------------------------------------
# 3. CLAIM 3 -- A7 vs A7D
# ---------------------------------------------------------------------------
log("")
log("=== CLAIM 3 -- A7 vs A7D ===")
sets = R.request_sets(ctx, {"A7", "A7D"})
A7, A7D = sets["A7"], sets["A7D"]
struct = {"n_A7": len(A7), "n_A7D": len(A7D),
          "terms_A7": sorted({len(r["request"]) for r in A7}),
          "terms_A7D": sorted({len(r["request"]) for r in A7D}),
          "ids_match_after_suffix": bool(all(d["id"] == a["id"] + "-dedup"
                                             for a, d in zip(A7, A7D))),
          "same_row": bool(all(a.get("row") == d.get("row") for a, d in zip(A7, A7D)))}
# is the removed term EXACTLY a duplicate of an earlier term at the SAME value?
dups = []
for a, d in zip(A7, A7D):
    ta, td = a["request"], d["request"]
    seen, extra = {}, []
    for t in ta:
        f = int(t["factor_index"])
        if f in seen:
            extra.append((f, float(t["value"]), float(seen[f])))
        else:
            seen[f] = float(t["value"])
    dups.append(extra)
struct["n_with_exactly_one_duplicate_term"] = int(sum(len(e) == 1 for e in dups))
struct["duplicate_same_value"] = int(sum(1 for e in dups if len(e) == 1 and e[0][1] == e[0][2]))
struct["duplicate_factors"] = sorted({names[e[0][0]] for e in dups if e})
struct["declared_redundant_factor_matches"] = int(sum(
    1 for a, e in zip(A7, dups)
    if e and (a.get("redundant_factor") in (None, names[e[0][0]], e[0][0]))))
log(f"  structure: {struct}")
V["a7_structure"] = struct

X_a7 = np.asarray(protos.embed_many(A7), np.float64).reshape(-1, store.dim)
X_a7d = np.asarray(protos.embed_many(A7D), np.float64).reshape(-1, store.dim)
# the embedding difference must be exactly ONE (mu_{k,v} - mu_global) deviation
mu = np.asarray(protos.mu_global, np.float64)
dev_expected = np.empty_like(X_a7)
for i, e in enumerate(dups):
    f, v, _ = e[0]
    dev_expected[i] = np.asarray(protos.prototype(f, v), np.float64) - mu
diff = X_a7 - X_a7d
V["a7_embedding_difference"] = {
    "max_abs_delta_vs_one_deviation": float(np.max(np.abs(diff - dev_expected))),
    "max_rel_delta": float(np.max(np.abs(diff - dev_expected)) /
                           max(float(np.max(np.abs(dev_expected))), 1e-12)),
    "mean_norm_of_the_extra_deviation": float(np.linalg.norm(dev_expected, axis=1).mean()),
    "mean_norm_of_the_embedding": float(np.linalg.norm(X_a7, axis=1).mean())}
log(f"  embedding difference == one (mu_kv - mu_global)? {V['a7_embedding_difference']}")

ga7, ga7d = my_gate(X_a7), my_gate(X_a7d)
V["a7_gate_mine"] = {"accept_A7": float(ga7["accept"].mean()),
                     "accept_A7D": float(ga7d["accept"].mean()),
                     "d_over_sigma_A7": float(ga7["dos"].min(1).mean()),
                     "d_over_sigma_A7D": float(ga7d["dos"].min(1).mean()),
                     "n_refused_A7": int(ga7["refuse"].sum()),
                     "n_refused_A7D": int(ga7d["refuse"].sum()),
                     "refused_A7_set_is_superset_of_A7D":
                         bool(np.all(ga7d["refuse"] <= ga7["refuse"]))}
log(f"  gate: {V['a7_gate_mine']}")

# ---------------------------------------------------------------------------
# 4. CLAIM 4 -- fairness of the factored kNN rival
# ---------------------------------------------------------------------------
log("")
log("=== CLAIM 4 -- the rival's operating point ===")
cal_rows = np.asarray(ctx["fit_rows"], np.int64)
cal_rows = cal_rows[np.random.default_rng(20260917).permutation(cal_rows.size)[:4096]]
X_cal = np.asarray(protos.embed_many(requests_from_rows(grid, cal_rows)),
                   np.float64).reshape(-1, store.dim)
X_val = pops["R8 affine: val"]
X_oor = np.concatenate([oor[w] for w in SEVERITIES], axis=0)


def factored_d(X):
    _, D = my_E_and_d(X)
    return D


D_cal, D_val, D_oor, D_a7 = factored_d(X_cal), factored_d(X_val), factored_d(X_oor), factored_d(X_a7)


def whole_d(X, chunk=2048):
    out = np.empty(X.shape[0])
    aa = (Z * Z).sum(1)[None, :]
    for i in range(0, X.shape[0], chunk):
        q = X[i:i + chunk]
        d2 = np.maximum((q * q).sum(1)[:, None] - 2.0 * (q @ Z.T) + aa, 0.0)
        out[i:i + q.shape[0]] = np.sqrt(d2.min(1))
    return out


sweep = []
for qq in (0.90, 0.95, 0.98, 0.98979, 0.99, 0.995, 0.999, 1.0):
    thr = np.quantile(D_cal, qq, axis=0)
    acc_val = float((~(D_val > thr[None, :]).any(1)).mean())
    ref_oor = float((D_oor > thr[None, :]).any(1).mean())
    acc_a7 = float((~(D_a7 > thr[None, :]).any(1)).mean())
    acc_cal = float((~(D_cal > thr[None, :]).any(1)).mean())
    sweep.append({"q_per_column": qq, "familywise_accept_on_cal": acc_cal,
                  "accept_val": acc_val, "refuse_oor": ref_oor, "accept_A7": acc_a7,
                  "score": 0.5 * (ref_oor + acc_val)})
    log(f"  q={qq:<8g} accept_cal={acc_cal:.4f} accept_val={acc_val:.4f} "
        f"refuse_oor={ref_oor:.4f} score={0.5*(ref_oor+acc_val):.4f}")
V["knn_factored_q_sweep"] = sweep
# whole-latent rival, same sweep
dw_cal, dw_val, dw_oor = whole_d(X_cal), whole_d(X_val), whole_d(X_oor)
sweep_w = []
for qq in (0.90, 0.95, 0.99, 0.999, 1.0):
    t = float(np.quantile(dw_cal, qq))
    sweep_w.append({"q": qq, "accept_val": float((dw_val <= t).mean()),
                    "refuse_oor": float((dw_oor > t).mean()),
                    "score": 0.5 * (float((dw_oor > t).mean()) + float((dw_val <= t).mean()))})
V["knn_whole_q_sweep"] = sweep_w
log(f"  knn_whole sweep: {sweep_w}")

# the CLU gate on the same populations, my arithmetic
g_val, g_oor = my_gate(X_val), my_gate(X_oor)
V["clu_arm_mine"] = {"accept_val": float(g_val["accept"].mean()),
                     "refuse_oor": float(g_oor["refuse"].mean()),
                     "accept_A7": float(ga7["accept"].mean()),
                     "score": 0.5 * (float(g_oor["refuse"].mean()) + float(g_val["accept"].mean()))}
log(f"  CLU arm (mine): {V['clu_arm_mine']}")


# --- threshold-free: AUROC of each statistic, my own implementation
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    a = np.concatenate([pos, neg])
    o = np.argsort(a, kind="mergesort")
    s = a[o]
    r = np.empty(a.size)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


thr95 = np.quantile(D_cal, 0.95, axis=0)
V["auroc_mine"] = {
    "CLU max_k E_k (oor vs val)": auroc(g_oor["E"].max(1), g_val["E"].max(1)),
    "CLU max_k d/sigma (oor vs val)": auroc((D_oor / SIG).max(1), (D_val / SIG).max(1)),
    "knn_factored max_k d/thr (oor vs val)": auroc((D_oor / thr95).max(1), (D_val / thr95).max(1)),
    "knn_whole d (oor vs val)": auroc(dw_oor, dw_val),
    "CLU max_k E_k (oor vs A7)": auroc(g_oor["E"].max(1), ga7["E"].max(1)),
    "knn_factored max_k d/thr (oor vs A7)": auroc((D_oor / thr95).max(1), (D_a7 / thr95).max(1)),
}
log(f"  AUROC (mine): {json.dumps(V['auroc_mine'], indent=2)}")

# ---------------------------------------------------------------------------
# 5. CLAIM 5 -- the paired bootstrap, re-implemented
# ---------------------------------------------------------------------------
log("")
log("=== CLAIM 5 -- paired bootstrap, my own ===")
acc_arm = g_val["accept"]
ref_arm = g_oor["refuse"]
thr_f = np.quantile(D_cal, 0.95, axis=0)
acc_riv = ~(D_val > thr_f[None, :]).any(1)
ref_riv = (D_oor > thr_f[None, :]).any(1)


def my_paired(a_arm, a_riv, r_arm, r_riv, n_boot=4000, seed=99):
    rng = np.random.default_rng(seed)
    na, nr = a_arm.size, r_arm.size
    d = np.empty(n_boot)
    for b in range(n_boot):
        ia = rng.integers(0, na, na)
        ir = rng.integers(0, nr, nr)
        d[b] = (0.5 * (r_arm[ir].mean() + a_arm[ia].mean())
                - 0.5 * (r_riv[ir].mean() + a_riv[ia].mean()))
    delta = 0.5 * (r_arm.mean() + a_arm.mean()) - 0.5 * (r_riv.mean() + a_riv.mean())
    return {"delta": float(delta), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "boot_mean": float(d.mean()), "boot_sd": float(d.std(ddof=1))}


def my_paired_unpaired(a_arm, a_riv, r_arm, r_riv, n_boot=4000, seed=99):
    """the SAME thing with INDEPENDENT resamples -- to show the pairing matters."""
    rng = np.random.default_rng(seed)
    na, nr = a_arm.size, r_arm.size
    d = np.empty(n_boot)
    for b in range(n_boot):
        d[b] = (0.5 * (r_arm[rng.integers(0, nr, nr)].mean() + a_arm[rng.integers(0, na, na)].mean())
                - 0.5 * (r_riv[rng.integers(0, nr, nr)].mean() + a_riv[rng.integers(0, na, na)].mean()))
    return {"ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "boot_sd": float(d.std(ddof=1))}


V["paired_bootstrap_mine"] = my_paired(acc_arm, acc_riv, ref_arm, ref_riv)
V["unpaired_bootstrap_mine"] = my_paired_unpaired(acc_arm, acc_riv, ref_arm, ref_riv)
his = AC.paired_bootstrap(acc_arm, acc_riv, ref_arm, ref_riv)
V["paired_bootstrap_his_on_my_vectors"] = {"delta": his["delta"], "ci95": his["ci95"],
                                           "verdict": his["verdict_R1"]}
log(f"  mine    : {V['paired_bootstrap_mine']}")
log(f"  unpaired: {V['unpaired_bootstrap_mine']}")
log(f"  his     : {V['paired_bootstrap_his_on_my_vectors']}")

json.dump(V, open(OUT / "VERIFY_CORE.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'VERIFY_CORE.json'}")
