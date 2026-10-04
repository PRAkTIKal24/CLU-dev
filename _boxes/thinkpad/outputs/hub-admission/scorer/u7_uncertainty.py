"""UNIT 7 -- uncertainty, and the matched-operating-point comparison.

(a) SEED SPREAD IS UNMEASURABLE ON THIS BOX and the report must say so rather than imply
    an interval it does not have.  Seeds 1 and 2 have caches with ``scope="used"`` (9,288
    rows); covering the cell would need ~32,200 new rows, which is a RE-ENCODE and is
    forbidden.  ⛔ So no seed-to-seed spread exists here.  What CAN be computed is a
    WITHIN-SEED bootstrap over requests, and it is labelled as exactly that -- it bounds
    sampling noise, ⛔ not seed variation.

(b) THE MATCHED-OPERATING-POINT COMPARISON, reported BESIDE the pre-registered pair.
    The pre-registered score rewards the rival's own choice of q, which lands it at a 21%
    refusal rate on VALID data.  The fairest single comparison is at MATCHED accept: give
    the rival whatever q makes its accept_val equal the arm's 1.0000, and read off its
    refuse_oor there.  ⛔ This does NOT replace the bar and does NOT change the verdict --
    the bar is the bar.  It is reported so a reviewer can see the whole picture.
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
b = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, book, grid = b["land"], b["store"], b["book"], ctx["grid"]
names = list(grid.names)
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)
protos = FactorPrototypes.fit(grid, X_pool, pool)
val_rows = np.asarray(ctx["val_rows"], np.int64)
req_val = requests_from_rows(grid, val_rows)
rec = {"schema": "admission-uncertainty/1", "seed": SEED, "budget": BUDGET,
       "seed_spread_measurable": False,
       "seed_spread_reason": ("seeds 1 and 2 have feature caches with scope='used' (9,288 rows); "
                              "covering the cell needs ~32,200 new rows, which is a RE-ENCODE and is "
                              "forbidden. No seed-to-seed spread can be computed on this box.")}
log(rec["seed_spread_reason"])


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


req_oor, fac = [], []
for f in range(len(names)):
    for w in SEVERITIES:
        qs = perturb(req_val, f, offgrid(f, w))
        req_oor += qs
        fac += [f] * len(qs)
fac = np.asarray(fac, np.int64)
X_val = protos.embed_many(req_val).reshape(-1, store.dim)
X_oor = protos.embed_many(req_oor).reshape(-1, store.dim)
cal_rows = np.asarray(ctx["fit_rows"], np.int64)
cal_rows = cal_rows[RNG.permutation(cal_rows.size)[:4096]]
X_cal = protos.embed_many(requests_from_rows(grid, cal_rows)).reshape(-1, store.dim)

# ---------------------------------------------------------------------------
# (a) within-seed bootstrap on the LOCALISATION GAP over the single-well control
# ---------------------------------------------------------------------------
log("")
log("=== (a) WITHIN-SEED bootstrap (⛔ NOT seed spread) ===")
land_K1 = AC.build_K1(land, X_pool)
lw, nt, _ = R.term_experts(book, grid, req_oor)
pred_A = AC.localise(land, X_oor, lw, nt, AL.gate(land, X_oor))
pred_1 = AC.localise(land_K1, X_oor, lw, nt, AL.gate(land_K1, X_oor))
hit_A, hit_1 = (pred_A == fac), (pred_1 == fac)
n = hit_A.size
d = np.empty(4000, np.float64)
for i in range(4000):
    j = RNG.integers(0, n, n)
    d[i] = hit_A[j].mean() - hit_1[j].mean()
rec["localisation"] = {
    "accuracy_arm": float(hit_A.mean()), "accuracy_K1": float(hit_1.mean()),
    "gap": float(hit_A.mean() - hit_1.mean()),
    "gap_ci95_within_seed": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
    "bar": 0.50, "bar_met": bool(hit_A.mean() >= 0.50),
    "beats_single_well": bool(np.quantile(d, 0.025) > 0.0),
    "note": "paired within-seed bootstrap over the 15,360 oor requests; ⛔ not a seed interval"}
log(f"  localisation arm={hit_A.mean():.4f}  K1={hit_1.mean():.4f}  gap={rec['localisation']['gap']:+.4f} "
    f"CI95(within-seed)=[{rec['localisation']['gap_ci95_within_seed'][0]:+.4f},"
    f"{rec['localisation']['gap_ci95_within_seed'][1]:+.4f}]  bar>=0.50 met? {rec['localisation']['bar_met']}")

# ---------------------------------------------------------------------------
# (b) the matched-operating-point comparison, ⛔ reported beside, not the bar
# ---------------------------------------------------------------------------
log("")
log("=== (b) MATCHED OPERATING POINT (⛔ reported beside; the BAR is unchanged) ===")
g_arm_val = AL.gate(land, X_val)
g_arm_oor = AL.gate(land, X_oor)
arm_acc, arm_ref = float(g_arm_val["accept"].mean()), float(g_arm_oor["refuse"].mean())
log(f"  the arm (derived, no free parameter): accept_val={arm_acc:.4f}  refuse_oor={arm_ref:.4f}")

rows = []
for kind in ("factored", "whole"):
    for q in (0.95, 0.99, 0.999, 0.9999, 1.0):
        g = AC.KnnGate.fit(store, X_cal, kind, q)
        av = float(g.decide(store, X_val)["accept"].mean())
        ro = float(g.decide(store, X_oor)["refuse"].mean())
        rows.append({"kind": kind, "q": q, "accept_val": av, "refuse_oor": ro,
                     "score": 0.5 * (ro + av)})
        log(f"  knn_{kind:9s} q={q:<7g} accept_val={av:.4f}  refuse_oor={ro:.4f}  "
            f"score={0.5*(ro+av):.4f}")
rec["rival_operating_curve"] = rows

# the rival's refuse_oor at the q that first reaches the arm's accept_val
matched = {}
for kind in ("factored", "whole"):
    lo, hi = 0.90, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        g = AC.KnnGate.fit(store, X_cal, kind, mid)
        if float(g.decide(store, X_val)["accept"].mean()) >= arm_acc:
            hi = mid
        else:
            lo = mid
    g = AC.KnnGate.fit(store, X_cal, kind, hi)
    matched[kind] = {"q": float(hi),
                     "accept_val": float(g.decide(store, X_val)["accept"].mean()),
                     "refuse_oor": float(g.decide(store, X_oor)["refuse"].mean()),
                     "thresholds_over_sigma": [float(t / s) for t, s in
                                               zip(g.thresholds, np.asarray(land.sigma, np.float64))]
                     if kind == "factored" else None}
    log(f"  MATCHED {kind}: q={hi:.6f} accept_val={matched[kind]['accept_val']:.4f} "
        f"refuse_oor={matched[kind]['refuse_oor']:.4f}  (arm: {arm_ref:.4f})")
rec["matched_operating_point"] = {"arm_accept_val": arm_acc, "arm_refuse_oor": arm_ref, **matched}
rec["arm_wins_at_matched_accept"] = {
    k: bool(arm_ref > matched[k]["refuse_oor"]) for k in matched}
log(f"  ⇒ arm beats the rival at MATCHED accept? {rec['arm_wins_at_matched_accept']}")
log("  ⛔ This does not change the pre-registered verdict. The bar is the bar.")

json.dump(rec, open(OUT / "UNCERTAINTY_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'UNCERTAINTY_seed0_A.json'}")
