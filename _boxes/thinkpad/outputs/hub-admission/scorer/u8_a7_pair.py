"""UNIT 8 -- THE A7-CONTROLLED PAIR.  What SS7b's rule implies, quantified.

SS7b (programme-wide, 2026-09-17): "three of five published rivals falsely refuse ~60% of the
valid A7 redundant-term control, and the published table does not show it.  A refusal
number without its A7 companion overstates the rival and understates the arm."

That is a claim about a NUMBER, so measure the number.  Swap the accept population from
`val` to `A7` and re-run the same pair across the rival's whole operating curve:
    score_A7 = 0.5 * (refuse_oor + accept_A7)

⛔ This is NOT the pre-registered bar and it does NOT change the verdict.  It is reported
BESIDE, labelled, because SS7b makes it mandatory to show accept_A7 next to every refusal
number, and this is what that looks like when it is carried all the way through.
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

SEED, BUDGET = 0, "A"
SEVERITIES = [2.0, 3.0, 4.0, 7.0, 13.0]
RNG = np.random.default_rng(20260917)

ctx = R.build_context(SEED)
b = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, book, grid = b["land"], b["store"], b["book"], ctx["grid"]
names = list(grid.names)
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)
protos = FactorPrototypes.fit(grid, X_pool, pool)
req_val = requests_from_rows(grid, np.asarray(ctx["val_rows"], np.int64))
sets = R.request_sets(ctx, {"A7", "A7D"})


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


req_oor = [q for f in range(len(names)) for w in SEVERITIES for q in perturb(req_val, f, offgrid(f, w))]
X = {"val": protos.embed_many(req_val).reshape(-1, store.dim),
     "A7": protos.embed_many(sets["A7"]).reshape(-1, store.dim),
     "A7D": protos.embed_many(sets["A7D"]).reshape(-1, store.dim),
     "oor": protos.embed_many(req_oor).reshape(-1, store.dim)}
cal_rows = np.asarray(ctx["fit_rows"], np.int64)
X_cal = protos.embed_many(requests_from_rows(
    grid, cal_rows[RNG.permutation(cal_rows.size)[:4096]])).reshape(-1, store.dim)

rec = {"schema": "admission-a7-pair/1", "seed": SEED, "budget": BUDGET,
       "note": "⛔ reported BESIDE the pre-registered bar; it does not change the verdict",
       "rows": []}

g = AL.gate(land, X["val"])
arm = {"arm": "CLU_gate@r8 (DERIVED, no free parameter)", "q": None,
       "accept_val": float(g["accept"].mean()),
       "accept_A7": float(AL.gate(land, X["A7"])["accept"].mean()),
       "accept_A7D": float(AL.gate(land, X["A7D"])["accept"].mean()),
       "refuse_oor": float(AL.gate(land, X["oor"])["refuse"].mean())}
arm["score_val"] = 0.5 * (arm["refuse_oor"] + arm["accept_val"])
arm["score_A7"] = 0.5 * (arm["refuse_oor"] + arm["accept_A7"])
arm["score_A7D"] = 0.5 * (arm["refuse_oor"] + arm["accept_A7D"])
rec["rows"].append(arm)

for kind in ("factored", "whole"):
    for q in (0.95, 0.99, 0.999, 0.9999, 0.999991, 1.0):
        gk = AC.KnnGate.fit(store, X_cal, kind, q)
        r = {"arm": f"knn_{kind} (FITTED q)", "q": q,
             "accept_val": float(gk.decide(store, X["val"])["accept"].mean()),
             "accept_A7": float(gk.decide(store, X["A7"])["accept"].mean()),
             "accept_A7D": float(gk.decide(store, X["A7D"])["accept"].mean()),
             "refuse_oor": float(gk.decide(store, X["oor"])["refuse"].mean())}
        r["score_val"] = 0.5 * (r["refuse_oor"] + r["accept_val"])
        r["score_A7"] = 0.5 * (r["refuse_oor"] + r["accept_A7"])
        r["score_A7D"] = 0.5 * (r["refuse_oor"] + r["accept_A7D"])
        rec["rows"].append(r)

log("")
log("=== THE A7-CONTROLLED PAIR  (score_X = ½(refuse_oor + accept_X)) ===")
log(f"{'arm':38s} {'q':>9s} {'acc_val':>8s} {'acc_A7':>8s} {'acc_A7D':>8s} {'ref_oor':>8s} "
    f"{'sc_val':>7s} {'sc_A7':>7s} {'sc_A7D':>7s}")
for r in rec["rows"]:
    log(f"{r['arm']:38s} {str(r['q']):>9s} {r['accept_val']:8.4f} {r['accept_A7']:8.4f} "
        f"{r['accept_A7D']:8.4f} {r['refuse_oor']:8.4f} {r['score_val']:7.4f} "
        f"{r['score_A7']:7.4f} {r['score_A7D']:7.4f}")

best = {k: max(rec["rows"][1:], key=lambda r: r[k]) for k in ("score_val", "score_A7", "score_A7D")}
rec["best_rival"] = {k: {"arm": v["arm"], "q": v["q"], k: v[k]} for k, v in best.items()}
rec["arm_vs_best_rival"] = {k: float(arm[k] - best[k][k]) for k in best}
log("")
for k in ("score_val", "score_A7", "score_A7D"):
    log(f"  {k:9s}: arm={arm[k]:.4f}  best rival={best[k][k]:.4f} "
        f"({best[k]['arm']}, q={best[k]['q']})  Δ={arm[k]-best[k][k]:+.4f}")

# paired R1 on the A7-controlled pair vs the rival at its own pre-registered q = 0.95
gk = AC.KnnGate.fit(store, X_cal, "factored", 0.95)
p = AC.paired_bootstrap(AL.gate(land, X["A7"])["accept"], gk.decide(store, X["A7"])["accept"],
                        AL.gate(land, X["oor"])["refuse"], gk.decide(store, X["oor"])["refuse"])
rec["R1_A7_controlled_vs_knn_factored_q095"] = p
log("")
log(f"  R1 on the A7-controlled pair vs knn_factored q=0.95: Δ={p['delta']:+.4f} "
    f"CI95=[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] -> {p['verdict_R1'].upper()}")

json.dump(rec, open(OUT / "A7_PAIR_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'A7_PAIR_seed0_A.json'}")
