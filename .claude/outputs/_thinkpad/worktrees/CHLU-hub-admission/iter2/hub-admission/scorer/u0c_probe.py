"""UNIT 0c -- SS0c of the addendum, plus step 1/step 2 ground truth, in one pass.

The one question: where does a VALID launch sit relative to the zero level set?
Measured at EVERY candidate launch point, because "the launch point" is ambiguous and
the honest thing is to report all of them:
    clamp_atom      the SHIPPED default (gen_landscape LAUNCH_MODES / config launch_mode)
    posterior_mean  the LINEAR read -- hub/refusal-2's "21 atom-spreads" point
    prior_atom      the launch-off control
    r8_affine       the R8 request embedding (FactorPrototypes.embed) -- "the affine head"
    true_feature    the row's OWN cached encoder feature (the exemplar reference)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-admission"
sys.path.insert(0, str(OUT / "scorer"))
sys.path.insert(0, os.path.join(os.environ["CHLU_HUB_OUTPUTS"], "hub-refusal", "scorer"))

import admission_lib as AL          # noqa: E402
import refusal_lib as R             # noqa: E402
from chlu.core.gen_landscape import launch_state  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)

SEED = 0
BUDGET = "A"
N_PROBE = int(os.environ.get("N_PROBE", "256"))

log("building context (seed 0) ...")
ctx = R.build_context(SEED)
log(f"context built in {ctx['build_s']:.1f}s")
log(f"feat_shape={ctx['feat_shape']}  pool={ctx['pool'].size}  fit={ctx['fit_rows'].size} "
    f"val={ctx['val_rows'].size}  X_store={ctx['X_store'].shape}")
log(f"cache_meta={ctx['cache_meta']}")
log(f"K(well_channels)={len(ctx['well_channels'])} sizes={[len(w) for w in ctx['well_channels']]}")
log(f"names={ctx['names']} cards={ctx['cards']}")
log(f"grouping={ctx['grouping']}")
log(f"reencode: ulp={ctx['reencode_check'].get('max_ulp_delta')} "
    f"frac={ctx['reencode_check'].get('frac_differing')} "
    f"enforced={ctx['reencode_check'].get('enforced')}")

log("building landscape (budget A) ...")
built = R.build_land(ctx, budget=BUDGET, log=log)
land, store, book = built["land"], built["store"], built["book"]
log(f"n_wells={store.n_wells} n_atoms={store.n_atoms} dim={store.dim} sigma={np.asarray(land.sigma)}")

rec = {"schema": "admission-u0c/1", "seed": SEED, "budget": BUDGET,
       "feat_shape": list(ctx["feat_shape"]),
       "K_wells": int(store.n_wells), "n_atoms": int(store.n_atoms), "dim": int(store.dim),
       "well_sizes": [int(len(w)) for w in ctx["well_channels"]],
       "sigma": [float(s) for s in np.asarray(land.sigma)],
       "cache_meta": ctx["cache_meta"], "grouping": ctx["grouping"],
       "names": ctx["names"], "cards": [int(c) for c in ctx["cards"]],
       "reencode_check": {k: v for k, v in ctx["reencode_check"].items()}}

# ---- PREREG-1 self-checks -------------------------------------------------
log("SELF-CHECK 1: identity E_k(a_ki) <= 0 on the LIVE store ...")
ident = AL.identity_E_le_0_on_live_store(land)
rec["identity"] = ident
log(f"  identity_E_le_0_holds={ident['identity_E_le_0_holds']} "
    f"worst={ident['max_over_all_wells']:.4f}")
for w in ident["wells"]:
    log(f"   well {w['well']:2d} N={w['n_atoms']} sigma={w['sigma']:.4f} "
        f"max_E={w['max_E_at_atoms']:.3f} cert_radius={w['certified_radius']:.3f}")

# ---- the probe sets -------------------------------------------------------
val_rows = np.asarray(ctx["val_rows"], np.int64)[:N_PROBE]
grid = ctx["grid"]
reqs_val = requests_from_rows(grid, val_rows)
log(f"val probe requests: {len(reqs_val)}")

log("SELF-CHECK 2: term_experts reproduces request_experts on valid rows ...")
log_w, n_terms, off = R.term_experts(book, grid, reqs_val)
rec["term_experts_offgrid"] = off
rec["reproduces_shipped_experts"] = R.assert_reproduces_shipped_experts(book, reqs_val, log_w)
log(f"  {rec['reproduces_shipped_experts']}")

# R8 affine head, fitted on the store-train pool (the arm's own addressing head)
log("fitting R8 FactorPrototypes on the store-train pool ...")
protos = FactorPrototypes.fit(grid, ctx["X_store"], ctx["pool"])
rec["r8_bytes"] = protos.byte_ledger()

# positions of val_rows inside pool -> their OWN cached features
pool = np.asarray(ctx["pool"], np.int64)
order = np.argsort(pool)
pos = order[np.searchsorted(pool[order], val_rows)]
assert np.array_equal(pool[pos], val_rows), "val rows are not all inside the store pool"
X_true = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)[pos]

R_int = np.asarray([[int(c) for c in r["class_indices"]] for r in reqs_val], np.int64)
launches = {}
for mode in ("clamp_atom", "posterior_mean", "prior_atom"):
    launches[mode] = np.asarray(launch_state(land, book, R_int, launch_mode=mode)["x0"], np.float32)
launches["r8_affine"] = protos.embed_many(reqs_val).reshape(len(reqs_val), -1)
launches["true_feature"] = X_true

log("SELF-CHECK 3: sum_k E_k == shipped store_energy ...")
rec["store_energy_agreement"] = AL.assert_matches_shipped_store_energy(land, launches["clamp_atom"])
log(f"  {rec['store_energy_agreement']}")

rec["launch_points_valid"] = {}
log("")
log("=== SS0c: WHERE DO VALID val LAUNCHES SIT? (gate: refuse well k iff E_k > 0) ===")
for mode, X in launches.items():
    s = AL.gate_summary(land, X)
    rec["launch_points_valid"][mode] = s
    log(f"  {mode:15s} accept={s['accept_rate']:.4f} refuse={s['refuse_rate']:.4f} "
        f"meanWellsRef={s['mean_wells_refusing']:.2f} "
        f"max_k E_k (mean)={s['E_max_over_wells_mean']:.4g} "
        f"min_k d/sigma (mean)={s['d_over_sigma_min_mean']:.3f} "
        f"d/sigma (mean)={s['d_over_sigma_mean']:.3f}")

json.dump(rec, open(OUT / "U0C_PROBE_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'U0C_PROBE_seed0_A.json'}")
