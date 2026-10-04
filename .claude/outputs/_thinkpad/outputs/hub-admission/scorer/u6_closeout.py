"""UNIT 6 -- the remaining measurement gaps, closed.

  (i)   CAPACITY CONTROL: rung B (N = 512 atoms).  Does MORE STORE move the radius?
        The bandwidth rule is nn_median, so the prediction is NO -- 8x the atoms tile the
        same manifold 8x more finely, sigma_k shrinks with the spacing, and d/sigma is
        scale-free.  Measured rather than asserted, because it is the obvious objection.
  (ii)  OUT-OF-VOCABULARY, SYMBOL: a request naming a value the store never wrote.  Needs
        a held-out-value store, so it could only be measured after UNIT 4.  It is half of
        ``out_of_support`` and the bar depends on it.
  (iii) A7 vs A7D: is the CLU's accept_A7 deficit the R8 head's additive DOUBLE-COUNT of a
        duplicated term, or the store's geometry?  ``gen_request.deduplicate_terms`` is the
        shipped answer key.
  (iv)  LOCALISATION ANATOMY: where the pre-registered rule works and where it does not.
  (v)   THE CERTIFIED FRACTION OF REFUSALS, as SS0 requires it be reported.
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
from chlu.core.gen_request import (  # noqa: E402
    FactorPrototypes, deduplicate_terms, requests_from_rows,
)
from chlu.eval.gen_probe import factor_labels  # noqa: E402
from chlu.experiments.exp_gen_dynamical_read import build_landscape  # noqa: E402

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED, Q = 0, 0.95
SEVERITIES = [2.0, 3.0, 4.0, 7.0, 13.0]
RNG = np.random.default_rng(20260917)

ctx = R.build_context(SEED)
grid = ctx["grid"]
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)
F_pool = np.asarray(grid.factor_indices_of(pool), np.int64)
wc = ctx["well_channels"]
val_rows = np.asarray(ctx["val_rows"], np.int64)
req_val = requests_from_rows(grid, val_rows)
protos = FactorPrototypes.fit(grid, X_pool, pool)
rec = {"schema": "admission-closeout/1", "seed": SEED}


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
# (i) CAPACITY CONTROL -- rung B
# ---------------------------------------------------------------------------
log("=== (i) CAPACITY CONTROL: rung A vs rung B ===")
rec["capacity"] = []
for budget in ("A", "B"):
    b = R.build_land(ctx, budget=budget, log=lambda *a, **k: None)
    L, S = b["land"], b["store"]
    sig = np.asarray(L.sigma, np.float64)
    rad = AL.certified_radius(L)
    # empirical zero level set, 2 rays per atom (cheap, same estimator as UNIT 5)
    zc = []
    for k, a in enumerate(S.atoms):
        A = np.asarray(a, np.float64)
        t = np.linspace(0.0, 5.0 * sig[k], 300)
        r0 = []
        for i in RNG.permutation(A.shape[0])[:32]:
            for _ in range(2):
                u = RNG.normal(size=A.shape[1]); u /= np.linalg.norm(u)
                P = A[i][None, :] + t[:, None] * u[None, :]
                d2 = ((P * P).sum(1)[:, None] - 2.0 * (P @ A.T) + (A * A).sum(1)[None, :])
                E = -(sig[k] ** 2) * np.log(np.exp(-np.maximum(d2, 0.0) / (2 * sig[k] ** 2)).sum(1))
                c = np.flatnonzero(E > 0.0)
                r0.append(t[c[0]] / sig[k] if c.size else np.nan)
        zc.append(float(np.nanmedian(r0)))
    Xr = X_pool[RNG.permutation(pool.size)[:2048]]
    g_real = AL.gate(L, Xr)
    Xv = protos.embed_many(req_val).reshape(-1, S.dim)
    g_val = AL.gate(L, Xv)
    Xo = np.concatenate([protos.embed_many(perturb(req_val, f, offgrid(f, w))).reshape(-1, S.dim)
                         for f in range(len(names)) for w in SEVERITIES], axis=0)
    g_oor = AL.gate(L, Xo)
    row = {"budget": budget, "n_atoms": int(S.n_atoms), "store_bytes": int(S.n_bytes()),
           "sigma": [float(s) for s in sig],
           "certified_over_sigma": float(rad[0] / sig[0]),
           "zero_level_set_over_sigma": zc,
           "real_features_d_over_sigma_mean": float(g_real["d_over_sigma"].min(1).mean()),
           "real_features_refuse": float(g_real["refuse"].mean()),
           "accept_val": float(g_val["accept"].mean()),
           "refuse_oor": float(g_oor["refuse"].mean()),
           "certified_fraction_of_refusals_oor": (
               float(g_oor["certified_refusal"].sum()) / max(int(g_oor["refuse"].sum()), 1))}
    rec["capacity"].append(row)
    log(f"  rung {budget}: N={row['n_atoms']} bytes={row['store_bytes']:,} "
        f"sigma={np.round(sig,3).tolist()}")
    log(f"      zero level set / sigma = {[round(x,3) for x in zc]}   certified/sigma = "
        f"{row['certified_over_sigma']:.3f}")
    log(f"      REAL features sit at d/sigma = {row['real_features_d_over_sigma_mean']:.3f} "
        f"(refuse {row['real_features_refuse']:.4f})")
    log(f"      accept_val={row['accept_val']:.4f}  refuse_oor={row['refuse_oor']:.4f}  "
        f"certified frac of refusals={row['certified_fraction_of_refusals_oor']:.4f}")
    if budget == "A":
        landA, storeA, bookA = L, S, b["book"]

# ---------------------------------------------------------------------------
# (ii) OUT-OF-VOCABULARY, SYMBOL -- on the six held-out-value stores
# ---------------------------------------------------------------------------
log("")
log("=== (ii) OUT-OF-VOCABULARY, SYMBOL (a value the store never wrote) ===")
rec["oov_symbol"] = []
for f, nm in enumerate(names):
    v_idx = cards[f] // 2
    v_val = float(np.asarray(grid.values[f], np.float64)[v_idx])
    keep = F_pool[:, f] != v_idx
    Xk, Pk = X_pool[keep], pool[keep]
    Lr = build_landscape(ctx["config"], "A", Xk, factor_labels(grid, Pk), wc,
                         ctx["feat_shape"], cards, SEED, log=lambda *a, **k: None)["land"]
    pr = FactorPrototypes.fit(grid, Xk, Pk)
    n_for_v = pr.n_per_value[f][v_idx]
    req_oov = perturb(req_val, f, v_val)              # symbolically name the deleted value
    X_oov = pr.embed_many(req_oov).reshape(-1, Lr.store.dim)
    X_val_r = pr.embed_many(req_val).reshape(-1, Lr.store.dim)
    g = AL.gate(Lr, X_oov)
    gv = AL.gate(Lr, X_val_r)
    cal = pr.embed_many(requests_from_rows(grid, Pk[RNG.permutation(Pk.size)[:2048]])).reshape(-1, Lr.store.dim)
    gf = AC.KnnGate.fit(Lr.store, cal, "factored", Q)
    row = {"factor": nm, "held_value": v_val, "train_rows_for_that_value_after_delete": int(n_for_v),
           "CLU_refuse_oov_symbol": float(g["refuse"].mean()),
           "CLU_accept_val": float(gv["accept"].mean()),
           "CLU_d_over_sigma": float(g["d_over_sigma"].min(1).mean()),
           "knn_factored_refuse_oov_symbol": float(gf.decide(Lr.store, X_oov)["refuse"].mean()),
           "knn_factored_accept_val": float(gf.decide(Lr.store, X_val_r)["accept"].mean())}
    rec["oov_symbol"].append(row)
    log(f"  {nm:12s} rows for the deleted value = {n_for_v} (prototype falls back to mu_global) "
        f"| CLU refuse={row['CLU_refuse_oov_symbol']:.4f} d/σ={row['CLU_d_over_sigma']:.3f} "
        f"| knn_factored refuse={row['knn_factored_refuse_oov_symbol']:.4f}")
rec["oov_symbol_mean"] = {
    "CLU_refuse": float(np.mean([r["CLU_refuse_oov_symbol"] for r in rec["oov_symbol"]])),
    "knn_factored_refuse": float(np.mean([r["knn_factored_refuse_oov_symbol"] for r in rec["oov_symbol"]]))}
log(f"  MEAN: CLU refuse={rec['oov_symbol_mean']['CLU_refuse']:.4f}  "
    f"knn_factored refuse={rec['oov_symbol_mean']['knn_factored_refuse']:.4f}")

# ---------------------------------------------------------------------------
# (iii) A7 vs A7D
# ---------------------------------------------------------------------------
log("")
log("=== (iii) A7 vs A7D -- is the accept_A7 deficit the R8 head's double-count? ===")
sets = R.request_sets(ctx, {"A7", "A7D"})
X_a7 = protos.embed_many(sets["A7"]).reshape(-1, storeA.dim)
X_a7d = protos.embed_many(sets["A7D"]).reshape(-1, storeA.dim)
cal = protos.embed_many(requests_from_rows(
    grid, np.asarray(ctx["fit_rows"], np.int64)[RNG.permutation(ctx["fit_rows"].size)[:4096]])
).reshape(-1, storeA.dim)
gfA = AC.KnnGate.fit(storeA, cal, "factored", Q)
rec["a7"] = {
    "n_A7": len(sets["A7"]), "n_A7D": len(sets["A7D"]),
    "terms_A7": sorted({len(r["request"]) for r in sets["A7"]}),
    "terms_A7D": sorted({len(r["request"]) for r in sets["A7D"]}),
    "CLU_accept_A7": float(AL.gate(landA, X_a7)["accept"].mean()),
    "CLU_accept_A7D": float(AL.gate(landA, X_a7d)["accept"].mean()),
    "CLU_d_over_sigma_A7": float(AL.gate(landA, X_a7)["d_over_sigma"].min(1).mean()),
    "CLU_d_over_sigma_A7D": float(AL.gate(landA, X_a7d)["d_over_sigma"].min(1).mean()),
    "knn_factored_accept_A7": float(gfA.decide(storeA, X_a7)["accept"].mean()),
    "knn_factored_accept_A7D": float(gfA.decide(storeA, X_a7d)["accept"].mean())}
for k, v in rec["a7"].items():
    log(f"  {k} = {v}")

# ---------------------------------------------------------------------------
# (iv) LOCALISATION ANATOMY
# ---------------------------------------------------------------------------
log("")
log("=== (iv) LOCALISATION ANATOMY (pre-registered rule, by severity) ===")
rec["localisation_by_severity"] = []
for w in SEVERITIES:
    reqs, fac = [], []
    for f in range(len(names)):
        qs = perturb(req_val, f, offgrid(f, w))
        reqs += qs
        fac += [f] * len(qs)
    fac = np.asarray(fac, np.int64)
    lw, nt, _ = R.term_experts(bookA, grid, reqs)
    X = protos.embed_many(reqs).reshape(-1, storeA.dim)
    g = AL.gate(landA, X)
    pred = AC.localise(landA, X, lw, nt, g)
    row = {"severity": w, "accuracy": float((pred == fac).mean()),
           "refuse_rate": float(g["refuse"].mean()),
           "accuracy_on_refused": float((pred[g["refuse"]] == fac[g["refuse"]]).mean())
           if g["refuse"].any() else None,
           "accuracy_on_accepted": float((pred[g["accept"]] == fac[g["accept"]]).mean())
           if g["accept"].any() else None}
    rec["localisation_by_severity"].append(row)
    log(f"  w={w:5g}  acc={row['accuracy']:.4f}  refuse={row['refuse_rate']:.4f}  "
        f"acc|refused={row['accuracy_on_refused']}  acc|accepted={row['accuracy_on_accepted']}")

json.dump(rec, open(OUT / "CLOSEOUT_seed0.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'CLOSEOUT_seed0.json'}")
