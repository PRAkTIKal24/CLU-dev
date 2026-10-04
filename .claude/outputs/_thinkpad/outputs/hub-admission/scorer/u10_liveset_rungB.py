"""UNIT 10 -- the CAPACITY CONTROL on the HEADLINE claim.

UNIT 6 ran the capacity control (rung B, 512 atoms, 8x the bytes) on ``refuse_oor``.  It was
never run on the LIVE-SET claim, which is the headline.  The obvious objection --
"a bigger store would notice the deletion" -- deserves a measurement and not just the
scale-free argument.

Two factors, chosen BEFORE running for a stated reason: ``floor_hue`` (the only
singly-assigned factor in the map, and the one where the fitted rival scored a perfect
1.0000 separation) and ``wall_hue`` (the rival's second best, 1.0000 / 0.5820).  If the
derived gate is going to notice a deletion anywhere at any capacity, it is here.
⛔ From cached features only.  ⛔ eval_b untouched.
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

import admission_ctrl as AC         # noqa: E402
import admission_lib as AL          # noqa: E402
import refusal_lib as R             # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import factor_labels  # noqa: E402
from chlu.experiments.exp_gen_dynamical_read import build_landscape  # noqa: E402

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED, Q = 0, 0.95
FACTORS = ["floor_hue", "wall_hue"]
N_QUERY = 256
RNG = np.random.default_rng(20260917)

ctx = R.build_context(SEED)
grid = ctx["grid"]
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)
F_pool = np.asarray(grid.factor_indices_of(pool), np.int64)
wc = ctx["well_channels"]
fw = json.load(open(OUT / "FACTOR_WELL_MAP_seed0.json"))
rec = {"schema": "admission-liveset-rungB/1", "seed": SEED,
       "factors_chosen_before_running": FACTORS,
       "reason": ("floor_hue is the only singly-assigned factor in the map and the one where the "
                  "fitted rival scored a perfect 1.0000 separation; wall_hue is its second best. "
                  "If the derived gate notices a deletion at any capacity, it is here."),
       "rows": []}

full_by_budget = {bud: R.build_land(ctx, budget=bud, log=lambda *a, **k: None) for bud in ("A", "B")}
for bud, fb in full_by_budget.items():
    log(f"full store rung {bud}: N={fb['store'].n_atoms} bytes={fb['store'].n_bytes():,} "
        f"sigma={np.round(np.asarray(fb['land'].sigma),3).tolist()}")

protos_full = FactorPrototypes.fit(grid, X_pool, pool)

for nm in FACTORS:
    f = names.index(nm)
    v_idx = cards[f] // 2
    wells = list(fw["map"][nm]["assigned_wells"])
    keep = F_pool[:, f] != v_idx
    Xk, Pk = X_pool[keep], pool[keep]
    pr = FactorPrototypes.fit(grid, Xk, Pk)
    held = np.flatnonzero(~keep)
    held = held[RNG.permutation(held.size)[:N_QUERY]]
    other = np.flatnonzero(F_pool[:, f] == (v_idx + 1) % cards[f])
    other = other[RNG.permutation(other.size)[:N_QUERY]]

    def hybrid(idx, protos):
        rq = requests_from_rows(grid, pool[idx])
        x = protos.embed_many(rq).reshape(len(rq), -1).copy()
        for k in wells:
            c = np.asarray(wc[k], np.int64)
            x[:, c] = X_pool[idx][:, c]
        return x

    for bud in ("A", "B"):
        log("")
        log(f"=== {nm}, rung {bud}: removing {int((~keep).sum())} rows, rebuilding ===")
        t = time.time()
        Lr = build_landscape(ctx["config"], bud, Xk, factor_labels(grid, Pk), wc,
                             ctx["feat_shape"], cards, SEED, log=lambda *a, **k: None)["land"]
        log(f"  rebuilt in {time.time()-t:.0f}s  N={Lr.store.n_atoms} "
            f"sigma={np.round(np.asarray(Lr.sigma),3).tolist()}")
        leak = int((np.asarray(grid.factor_indices_of(Pk[np.asarray(Lr.store.atom_rows, np.int64)]),
                               np.int64)[:, f] == v_idx).sum())
        if leak:
            raise AssertionError(f"⛔ rebuilt rung {bud} store leaks {leak} held-out-value atoms")
        Xh, Xi = hybrid(held, pr), hybrid(other, pr)
        gh, gi = AL.gate(Lr, Xh), AL.gate(Lr, Xi)
        # write-back against the FULL store at the same rung
        Lf = full_by_budget[bud]["land"]
        gw = AL.gate(Lf, hybrid(held, protos_full))
        calr = pr.embed_many(requests_from_rows(grid, Pk[RNG.permutation(Pk.size)[:2048]])).reshape(-1, Lr.store.dim)
        gk = AC.KnnGate.fit(Lr.store, calr, "factored", Q)
        row = {"factor": nm, "budget": bud, "n_atoms": int(Lr.store.n_atoms),
               "store_bytes": int(Lr.store.n_bytes()), "atoms_leaking": leak,
               "sigma": [float(s) for s in np.asarray(Lr.sigma)],
               "certified_radius_over_sigma": float(AL.certified_radius(Lr)[0] / np.asarray(Lr.sigma)[0]),
               "CLU_refuse_heldout": float(gh["refuse"].mean()),
               "CLU_refuse_invocab": float(gi["refuse"].mean()),
               "CLU_writeback_accept": float(gw["accept"].mean()),
               "CLU_heldout_d_over_sigma": float(gh["d_over_sigma"].min(1).mean()),
               "CLU_heldout_max_E_mean": float(gh["E"].max(1).mean()),
               "CLU_heldout_max_E_closest": float(gh["E"].max(1).max()),
               "knn_factored_refuse_heldout": float(gk.decide(Lr.store, Xh)["refuse"].mean()),
               "knn_factored_refuse_invocab": float(gk.decide(Lr.store, Xi)["refuse"].mean())}
        row["CLU_separation"] = row["CLU_refuse_heldout"] - row["CLU_refuse_invocab"]
        row["knn_separation"] = row["knn_factored_refuse_heldout"] - row["knn_factored_refuse_invocab"]
        rec["rows"].append(row)
        log(f"  ⭐ CLU refuse(held-out)={row['CLU_refuse_heldout']:.4f} "
            f"refuse(in-vocab)={row['CLU_refuse_invocab']:.4f} sep={row['CLU_separation']:+.4f}")
        log(f"     held-out queries sit at d/σ={row['CLU_heldout_d_over_sigma']:.3f}; "
            f"mean max_k E_k={row['CLU_heldout_max_E_mean']:.2f}; "
            f"CLOSEST any query came to the gate = {row['CLU_heldout_max_E_closest']:.2f} "
            f"(the gate is at 0)")
        log(f"     knn_factored sep={row['knn_separation']:+.4f} "
            f"(held {row['knn_factored_refuse_heldout']:.4f} / in-vocab "
            f"{row['knn_factored_refuse_invocab']:.4f})")

log("")
log("=== DOES CAPACITY RESCUE THE LIVE-SET CLAIM? ===")
for r in rec["rows"]:
    log(f"  {r['factor']:10s} rung {r['budget']} (N={r['n_atoms']:4d}, {r['store_bytes']:,} B): "
        f"CLU sep={r['CLU_separation']:+.4f}   knn sep={r['knn_separation']:+.4f}")
rec["capacity_rescues_liveset"] = bool(any(r["CLU_separation"] >= 0.05 for r in rec["rows"]))
log(f"  ⇒ capacity rescues the live-set claim? {rec['capacity_rescues_liveset']}")

json.dump(rec, open(OUT / "LIVESET_RUNGB_seed0.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'LIVESET_RUNGB_seed0.json'}")
