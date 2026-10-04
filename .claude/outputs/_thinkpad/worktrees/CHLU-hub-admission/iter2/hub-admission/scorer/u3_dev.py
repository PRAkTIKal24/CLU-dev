"""UNIT 3 -- THE DEV TABLE.  Every SS4 control, on val corners.  ⛔ B is NOT touched.

Everything here is DEV and costs nothing from the scored budget.  It exists to decide
whether the SS5 bar is reachable BEFORE PREREG-2 and the single scored run.
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
N_CAL = int(os.environ.get("N_CAL", "4096"))

ctx = R.build_context(SEED)
built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, book, grid = built["land"], built["store"], built["book"], ctx["grid"]
names = list(grid.names)
protos = FactorPrototypes.fit(grid, ctx["X_store"], ctx["pool"])
log(f"land K={store.n_wells} N={store.n_atoms}; R8 head bytes={protos.n_bytes:,}")

rec = {"schema": "admission-dev/1", "seed": SEED, "budget": BUDGET, "q": Q,
       "K_wells": int(store.n_wells), "n_atoms": int(store.n_atoms),
       "sigma": [float(s) for s in np.asarray(land.sigma)],
       "certified_radius": [float(x) for x in AL.certified_radius(land)],
       "r8_head_bytes": int(protos.n_bytes), "store_bytes": int(store.n_bytes()),
       "names": names, "sets": {}, "rows": [], "pairs": {}, "localisation": {}}


def embed(reqs):
    return protos.embed_many(reqs).reshape(len(reqs), -1)


def experts(reqs):
    return R.term_experts(book, grid, reqs)


# ---------------------------------------------------------------------------
# the query sets.  ⛔ val corners and fit corners only; B is held out.
# ---------------------------------------------------------------------------
val_rows = np.asarray(ctx["val_rows"], np.int64)
cal_rows = np.asarray(ctx["fit_rows"], np.int64)
cal_rows = cal_rows[np.random.default_rng(20260917).permutation(cal_rows.size)[:N_CAL]]
req_val = requests_from_rows(grid, val_rows)
req_cal = requests_from_rows(grid, cal_rows)
log(f"val corners = {len(req_val)}   calibration corners (from fit_rows) = {len(req_cal)}")

sets = R.request_sets(ctx, {"A7"})
req_a7 = sets["A7"]
log(f"A7 (redundant-term valid control) = {len(req_a7)} requests, "
    f"terms per request = {sorted({len(r['request']) for r in req_a7})}")


def offgrid_value(f, w):
    v = np.asarray(grid.values[f], np.float64)
    return float(v[-2] + w * (v[-1] - v[-2]))


def perturb(reqs, f, val):
    out = []
    for r in reqs:
        q = copy.deepcopy(r)
        for t in q["request"]:
            if int(t["factor_index"]) == f:
                t["value"] = float(val)
        q.pop("class_indices", None)
        out.append(q)
    return out


def midgrid(reqs, f):
    v = np.asarray(grid.values[f], np.float64)
    out = []
    for r in reqs:
        q = copy.deepcopy(r)
        for t in q["request"]:
            if int(t["factor_index"]) == f:
                cur = float(t["value"])
                i = int(np.argmin(np.abs(v - cur)))
                j = i + 1 if i + 1 < len(v) else i - 1
                t["value"] = float(0.5 * (v[i] + v[j]))
        q.pop("class_indices", None)
        out.append(q)
    return out


req_oor, oor_factor, oor_sev = [], [], []
for f in range(len(names)):
    for w in SEVERITIES:
        qs = perturb(req_val, f, offgrid_value(f, w))
        req_oor += qs
        oor_factor += [f] * len(qs)
        oor_sev += [w] * len(qs)
oor_factor = np.asarray(oor_factor, np.int64)
oor_sev = np.asarray(oor_sev, np.float64)

req_unseen = [q for f in range(len(names)) for q in midgrid(req_val, f)]
log(f"oor pool = {len(req_oor)}   unseen_value = {len(req_unseen)}")
rec["sets"] = {"val": len(req_val), "cal": len(req_cal), "A7": len(req_a7),
               "oor": len(req_oor), "unseen_value": len(req_unseen),
               "severities": SEVERITIES,
               "calibration_source": "fit_rows (disjoint from val and from B)"}

# ---------------------------------------------------------------------------
# embeddings + posteriors
# ---------------------------------------------------------------------------
log("embedding ...")
X = {"val": embed(req_val), "cal": embed(req_cal), "A7": embed(req_a7),
     "oor": embed(req_oor), "unseen_value": embed(req_unseen)}
LW = {k: experts(v) for k, v in
      (("val", req_val), ("cal", req_cal), ("A7", req_a7),
       ("oor", req_oor), ("unseen_value", req_unseen))}
P = {k: AL.launch_from_terms_mode(land, book, LW[k][0], LW[k][1], "posterior_mean")["posterior"]
     for k in LW}
log("embedded.")

# ---------------------------------------------------------------------------
# self-check: the (term, well) residual reduction reproduces the shipped scalar
# ---------------------------------------------------------------------------
chk = AC.assert_matches_shipped_request_energy(land, X["val"], LW["val"][0], LW["val"][1])
rec["request_energy_agreement"] = chk
log(f"SELF-CHECK request_energy: {chk}")

# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------
land_K1 = AC.build_K1(land, np.asarray(ctx["X_store"], np.float32))
rec["K1"] = {"sigma": [float(s) for s in np.asarray(land_K1.sigma)],
             "certified_radius": [float(x) for x in AL.certified_radius(land_K1)],
             "store_bytes": int(land_K1.store.n_bytes()),
             "identity": AL.identity_E_le_0_on_live_store(land_K1)}
log(f"K=1 control: sigma={rec['K1']['sigma']} bytes={rec['K1']['store_bytes']:,} "
    f"identity_holds={rec['K1']['identity']['identity_E_le_0_holds']}")

g_whole = AC.KnnGate.fit(store, X["cal"], "whole", Q)
g_fact = AC.KnnGate.fit(store, X["cal"], "factored", Q)
g_marg = AC.MarginGate.fit(P["cal"], Q)
rec["fitted_thresholds"] = {"knn_whole": g_whole.summary(store, X["cal"])["thresholds"],
                            "knn_factored": g_fact.summary(store, X["cal"])["thresholds"],
                            "margin_M": [g_marg.thr]}
log(f"fitted on {len(req_cal)} calibration corners: {rec['fitted_thresholds']}")

ARMS = ["CLU_gate@r8", "CLU_K1@r8", "knn_whole", "knn_factored", "margin_M",
        "CLU_gate@posterior_mean", "CLU_gate@clamp_atom"]


def decide(arm, setname):
    if arm == "CLU_gate@r8":
        return AL.gate(land, X[setname])["accept"]
    if arm == "CLU_K1@r8":
        return AL.gate(land_K1, X[setname])["accept"]
    if arm == "knn_whole":
        return g_whole.decide(store, X[setname])["accept"]
    if arm == "knn_factored":
        return g_fact.decide(store, X[setname])["accept"]
    if arm == "margin_M":
        return g_marg.decide(P[setname])["accept"]
    if arm == "CLU_gate@posterior_mean":
        lw, nt = LW[setname][0], LW[setname][1]
        return AL.gate(land, AL.launch_from_terms_mode(land, book, lw, nt, "posterior_mean")["x0"])["accept"]
    if arm == "CLU_gate@clamp_atom":
        lw, nt = LW[setname][0], LW[setname][1]
        return AL.gate(land, AL.launch_from_terms_mode(land, book, lw, nt, "clamp_atom")["x0"])["accept"]
    raise ValueError(arm)


ACC = {a: {s: decide(a, s) for s in ("val", "A7", "oor", "unseen_value")} for a in ARMS}

log("")
log("=== THE DEV TABLE (seed 0, rung A, val corners; B UNTOUCHED) ===")
log(f"{'arm':26s} {'accept_val':>11s} {'accept_A7':>10s} {'refuse_oor':>11s} "
    f"{'accept_unseen':>14s} {'score':>8s}  fitted?")
for a in ARMS:
    av = float(ACC[a]["val"].mean()); a7 = float(ACC[a]["A7"].mean())
    ro = float((~ACC[a]["oor"]).mean()); au = float(ACC[a]["unseen_value"].mean())
    fitted = a in ("knn_whole", "knn_factored", "margin_M")
    row = {"arm": a, "accept_val": av, "accept_A7": a7, "refuse_oor": ro,
           "accept_unseen_value": au, "score_dev": 0.5 * (ro + av),
           "A7_minus_val": a7 - av, "has_fitted_threshold": fitted}
    rec["rows"].append(row)
    log(f"{a:26s} {av:11.4f} {a7:10.4f} {ro:11.4f} {au:14.4f} "
        f"{0.5*(ro+av):8.4f}  {'FITTED q=0.95' if fitted else 'DERIVED (no free parameter)'}")

# severity-resolved refusal, arm vs the fair rival
log("")
log("=== refuse_oor by severity ===")
rec["by_severity"] = []
for w in SEVERITIES:
    m = oor_sev == w
    r = {"severity": w,
         **{a: float((~ACC[a]["oor"][m]).mean()) for a in ARMS}}
    rec["by_severity"].append(r)
    log(f"  w={w:5g}  " + "  ".join(f"{a.split('@')[0][:12]}={r[a]:.3f}" for a in ARMS[:5]))

log("")
log("=== refuse_oor by factor (pooled over severities) ===")
rec["by_factor"] = []
for f, nm in enumerate(names):
    m = oor_factor == f
    r = {"factor": nm, **{a: float((~ACC[a]["oor"][m]).mean()) for a in ARMS}}
    rec["by_factor"].append(r)
    log(f"  {nm:12s}  " + "  ".join(f"{a.split('@')[0][:12]}={r[a]:.3f}" for a in ARMS[:5]))

# ---------------------------------------------------------------------------
# R1 pairs: arm vs each rival
# ---------------------------------------------------------------------------
log("")
log("=== R1 PAIRED COMPARISON (DEV proxy: accept_val in place of accept_B) ===")
for riv in ("knn_factored", "knn_whole", "margin_M", "CLU_K1@r8"):
    p = AC.paired_bootstrap(ACC["CLU_gate@r8"]["val"], ACC[riv]["val"],
                            ~ACC["CLU_gate@r8"]["oor"], ~ACC[riv]["oor"])
    rec["pairs"][riv] = p
    log(f"  vs {riv:16s} delta={p['delta']:+.4f} CI95=[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] "
        f"-> {p['verdict_R1'].upper()}")

# ---------------------------------------------------------------------------
# localisation: single-factor perturbations
# ---------------------------------------------------------------------------
log("")
log("=== TERM LOCALISATION (single-factor out_of_range; chance = 1/6 = 0.1667) ===")
lw_o, nt_o = LW["oor"][0], LW["oor"][1]
for tag, L in (("CLU_gate@r8", land), ("CLU_K1@r8", land_K1)):
    g = AL.gate(L, X["oor"])
    pred = AC.localise(L, X["oor"], lw_o, nt_o, g)
    acc = float((pred == oor_factor).mean())
    per_f = {names[f]: float((pred[oor_factor == f] == f).mean()) for f in range(len(names))}
    ref_only = g["refuse"]
    acc_ref = float((pred[ref_only] == oor_factor[ref_only]).mean()) if ref_only.any() else None
    rec["localisation"][tag] = {"accuracy": acc, "per_factor": per_f,
                                "accuracy_on_refused_only": acc_ref,
                                "n_refused": int(ref_only.sum())}
    log(f"  {tag:16s} acc={acc:.4f}  on-refused={acc_ref if acc_ref is None else round(acc_ref,4)} "
        f"(n={int(ref_only.sum())})")
    log(f"      per factor: " + "  ".join(f"{k}={v:.3f}" for k, v in per_f.items()))

json.dump(rec, open(OUT / "DEV_TABLE_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'DEV_TABLE_seed0_A.json'}")
