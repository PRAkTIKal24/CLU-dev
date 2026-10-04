"""ADVERSARIAL VERIFICATION -- claim 1: "the gate refuses 0.0000 of real encoded images,
even after every atom carrying that value is deleted from the store".

Everything recomputed from the formula in float64; admission_lib is NOT used for any
gate number here.  Four independent checks per factor:

  (a) the rebuilt store really carries no atom with the held-out value -- re-derived from
      the store's own ``atom_rows`` mapped back to global row ids and re-labelled;
      plus: the rebuilt atoms really DIFFER from the full store's atoms;
  (b) the hybrid query really is the cached encoder feature in the assigned wells and the
      R8 embedding everywhere else -- asserted channel-by-channel, bit-exactly;
  (c) the refusal really is 0, and HOW FAR from 0 the worst query is;
  (d) ⭐ THE STRONGEST VARIANT: the FULL, unmodified 2048-dim cached feature vector of a
      held-out-value image, gated against the rebuilt store.  No hybrid, no R8.

WRITES: VERIFY_LIVESET.json under $HUB_OUT.
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

import admission_ctrl as AC          # noqa: E402  (the RIVAL only)
import refusal_lib as R              # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import factor_labels  # noqa: E402
from chlu.experiments.exp_gen_dynamical_read import build_landscape  # noqa: E402

T0 = time.time()


def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


SEED, BUDGET, Q = 0, "A", 0.95
N_QUERY = 256
RNG = np.random.default_rng(20260917)          # ⭐ the hub's own seed, same rows

ctx = R.build_context(SEED)
full = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land_full, store_full, grid = full["land"], full["store"], ctx["grid"]
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float64).reshape(pool.size, -1)
F_pool = np.asarray(grid.factor_indices_of(pool), np.int64)
wc = ctx["well_channels"]
WC = [np.asarray(w, np.int64) for w in wc]
fw = json.load(open(OUT / "FACTOR_WELL_MAP_seed0.json"))
protos_full = FactorPrototypes.fit(grid, ctx["X_store"], ctx["pool"])
log(f"full store K={store_full.n_wells} N={store_full.n_atoms} pool={pool.size}")


def my_E_and_d(X, atoms, wc_, sig, chunk=1024):
    """E_k(x) = -s^2 log sum_i exp(-d2/(2 s^2)),  stable form  E_k = m/2 - s^2 log S."""
    X = np.asarray(X, np.float64)
    n = X.shape[0]
    E = np.empty((n, len(atoms)), np.float64)
    D = np.empty((n, len(atoms)), np.float64)
    for k, (A, cols) in enumerate(zip(atoms, wc_)):
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


def gate_of(L, X):
    atoms = [np.asarray(a, np.float64) for a in L.store.atoms]
    wcl = [np.asarray(w, np.int64) for w in L.store.well_channels]
    sig = np.asarray(L.sigma, np.float64)
    E, D = my_E_and_d(X, atoms, wcl, sig)
    ref = (E > 0.0).any(1)
    return {"E": E, "d": D, "dos": D / sig[None, :], "refuse": ref, "accept": ~ref,
            "max_k_E": E.max(1)}


V = {"schema": "verify-admission-liveset/1", "seed": SEED, "budget": BUDGET,
     "n_query": N_QUERY, "factors": []}

for f, nm in enumerate(names):
    v_idx = cards[f] // 2
    wells = list(fw["map"][nm]["assigned_wells"])
    held_mask = F_pool[:, f] == v_idx
    keep = ~held_mask
    Xk, Pk = X_pool[keep], pool[keep]
    log("")
    log(f"=== {nm}: hold out value {v_idx}; wells {wells}; removing {int(held_mask.sum())} rows ===")

    land_r = build_landscape(ctx["config"], BUDGET, np.asarray(Xk, np.float32),
                             factor_labels(grid, Pk), wc, ctx["feat_shape"], cards, SEED,
                             log=lambda *a, **k: None)["land"]
    store_r = land_r.store

    # ---- (a) the hold-out is real, re-derived independently -----------------
    atom_rows_global = Pk[np.asarray(store_r.atom_rows, np.int64)]
    lab = np.asarray(grid.factor_indices_of(atom_rows_global), np.int64)
    leak = int((lab[:, f] == v_idx).sum())
    full_rows = pool[np.asarray(store_full.atom_rows, np.int64)]
    n_shared = int(np.isin(atom_rows_global, full_rows).sum())
    # and: no atom VECTOR of the rebuilt store equals any deleted row's feature
    Zr = np.zeros((store_r.n_atoms, store_r.dim), np.float64)
    for k, cols in enumerate(WC):
        Zr[:, cols] = np.asarray(store_r.atoms[k], np.float64)
    Xheld_all = X_pool[held_mask]
    sub = Xheld_all[RNG.permutation(Xheld_all.shape[0])[:2048]]
    dmin_atom_to_held = float(np.sqrt(np.maximum(
        ((Zr * Zr).sum(1)[:, None] - 2.0 * (Zr @ sub.T) + (sub * sub).sum(1)[None, :]).min(), 0.0)))
    log(f"  (a) atoms carrying the held-out value = {leak} (mine); "
        f"atoms shared with the FULL store = {n_shared}/{store_r.n_atoms}; "
        f"min ||atom - any deleted row|| = {dmin_atom_to_held:.4f}")

    # ---- the queries, the hub's own rows ------------------------------------
    held_idx = np.flatnonzero(held_mask)
    held_idx = held_idx[RNG.permutation(held_idx.size)[:N_QUERY]]
    other_v = (v_idx + 1) % cards[f]
    in_idx = np.flatnonzero(F_pool[:, f] == other_v)
    in_idx = in_idx[RNG.permutation(in_idx.size)[:N_QUERY]]
    req_held = requests_from_rows(grid, pool[held_idx])
    req_in = requests_from_rows(grid, pool[in_idx])
    protos_r = FactorPrototypes.fit(grid, np.asarray(Xk, np.float32), Pk)

    def hybrid(Xreal, reqs, protos):
        x = np.asarray(protos.embed_many(reqs), np.float64).reshape(len(reqs), -1).copy()
        for k in wells:
            x[:, WC[k]] = Xreal[:, WC[k]]
        return x

    Xr8_held = np.asarray(protos_r.embed_many(req_held), np.float64).reshape(len(req_held), -1)
    X_held_h = hybrid(X_pool[held_idx], req_held, protos_r)
    X_in_h = hybrid(X_pool[in_idx], req_in, protos_r)

    # ---- (b) the hybrid really is the feature in exactly the assigned wells --
    assigned = np.concatenate([WC[k] for k in wells])
    other = np.setdiff1d(np.arange(store_r.dim), assigned)
    exact_in_wells = bool(np.array_equal(X_held_h[:, assigned], X_pool[held_idx][:, assigned]))
    untouched_elsewhere = bool(np.array_equal(X_held_h[:, other], Xr8_held[:, other]))
    n_ch_diff = int((np.abs(X_held_h - Xr8_held) > 0).any(0).sum())
    frac_dims_real = float(assigned.size) / float(store_r.dim)
    log(f"  (b) hybrid: real feature present bit-exactly in the assigned wells = "
        f"{exact_in_wells}; R8 untouched elsewhere = {untouched_elsewhere}; "
        f"channels that differ from pure R8 = {n_ch_diff} (assigned = {assigned.size}, "
        f"{frac_dims_real:.1%} of 2048)")

    # ---- (c) the refusal, and how far from zero ------------------------------
    g_h = gate_of(land_r, X_held_h)
    g_i = gate_of(land_r, X_in_h)
    log(f"  (c) CLU refuse(held-out hybrid) = {g_h['refuse'].mean():.4f}   "
        f"refuse(in-vocab hybrid) = {g_i['refuse'].mean():.4f}")
    log(f"      max_k E_k: held-out mean {g_h['max_k_E'].mean():.2f}  "
        f"WORST (closest to the gate) {g_h['max_k_E'].max():.2f}   "
        f"in-vocab worst {g_i['max_k_E'].max():.2f}")

    # ---- (d) ⭐ THE FULL CACHED FEATURE, ALL 2048 DIMS, NO HYBRID ------------
    X_held_full = X_pool[held_idx]
    X_in_full = X_pool[in_idx]
    gF_h = gate_of(land_r, X_held_full)
    gF_i = gate_of(land_r, X_in_full)
    gFull_h = gate_of(land_full, X_held_full)          # same images vs the FULL store
    log(f"  (d) FULL 2048-dim cached feature vs the REBUILT store: "
        f"refuse(held-out) = {gF_h['refuse'].mean():.4f}  "
        f"refuse(in-vocab) = {gF_i['refuse'].mean():.4f}  "
        f"[vs the FULL store: {gFull_h['refuse'].mean():.4f}]")
    log(f"      d/sigma min_k: held-out {gF_h['dos'].min(1).mean():.4f} vs "
        f"in-vocab {gF_i['dos'].min(1).mean():.4f}; "
        f"worst max_k E_k held-out = {gF_h['max_k_E'].max():.2f}")

    # the rival, on the same FULL feature vectors, for scale
    cal_idx = RNG.permutation(Pk.size)[:2048]
    X_cal = np.asarray(protos_r.embed_many(requests_from_rows(grid, Pk[cal_idx])),
                       np.float32).reshape(-1, store_r.dim)
    gf = AC.KnnGate.fit(store_r, X_cal, "factored", Q)
    riv_full_h = float(gf.decide(store_r, np.asarray(X_held_full, np.float32))["refuse"].mean())
    riv_full_i = float(gf.decide(store_r, np.asarray(X_in_full, np.float32))["refuse"].mean())
    riv_hyb_h = float(gf.decide(store_r, np.asarray(X_held_h, np.float32))["refuse"].mean())
    log(f"      knn_factored on the FULL feature: refuse(held)={riv_full_h:.4f} "
        f"refuse(in-vocab)={riv_full_i:.4f}   (on the hybrid: {riv_hyb_h:.4f})")

    V["factors"].append({
        "factor": nm, "held_value_index": v_idx, "wells_assigned": wells,
        "n_rows_removed": int(held_mask.sum()),
        "a_atoms_leaking_held_value_mine": leak,
        "a_atoms_shared_with_full_store": n_shared, "a_n_atoms": int(store_r.n_atoms),
        "a_min_dist_rebuilt_atom_to_any_deleted_row": dmin_atom_to_held,
        "b_hybrid_exact_in_assigned_wells": exact_in_wells,
        "b_hybrid_untouched_elsewhere": untouched_elsewhere,
        "b_n_channels_differing_from_pure_r8": n_ch_diff,
        "b_assigned_channels": int(assigned.size),
        "b_frac_of_dims_carrying_the_real_image": frac_dims_real,
        "c_refuse_heldout_hybrid": float(g_h["refuse"].mean()),
        "c_refuse_invocab_hybrid": float(g_i["refuse"].mean()),
        "c_max_k_E_heldout_mean": float(g_h["max_k_E"].mean()),
        "c_max_k_E_heldout_worst": float(g_h["max_k_E"].max()),
        "c_max_k_E_invocab_worst": float(g_i["max_k_E"].max()),
        "c_dos_heldout": float(g_h["dos"].min(1).mean()),
        "c_dos_invocab": float(g_i["dos"].min(1).mean()),
        "d_refuse_FULLFEATURE_heldout_rebuilt": float(gF_h["refuse"].mean()),
        "d_refuse_FULLFEATURE_invocab_rebuilt": float(gF_i["refuse"].mean()),
        "d_refuse_FULLFEATURE_heldout_fullstore": float(gFull_h["refuse"].mean()),
        "d_max_k_E_FULLFEATURE_heldout_worst": float(gF_h["max_k_E"].max()),
        "d_max_k_E_FULLFEATURE_heldout_mean": float(gF_h["max_k_E"].mean()),
        "d_dos_FULLFEATURE_heldout": float(gF_h["dos"].min(1).mean()),
        "d_dos_FULLFEATURE_invocab": float(gF_i["dos"].min(1).mean()),
        "rival_refuse_FULLFEATURE_heldout": riv_full_h,
        "rival_refuse_FULLFEATURE_invocab": riv_full_i,
        "rival_refuse_hybrid_heldout": riv_hyb_h,
    })

agg = {}
for k in ("c_refuse_heldout_hybrid", "c_refuse_invocab_hybrid",
          "d_refuse_FULLFEATURE_heldout_rebuilt", "d_refuse_FULLFEATURE_invocab_rebuilt",
          "d_refuse_FULLFEATURE_heldout_fullstore",
          "rival_refuse_FULLFEATURE_heldout", "rival_refuse_FULLFEATURE_invocab",
          "c_max_k_E_heldout_worst", "d_max_k_E_FULLFEATURE_heldout_worst",
          "d_dos_FULLFEATURE_heldout", "d_dos_FULLFEATURE_invocab"):
    agg[k] = float(np.mean([r[k] for r in V["factors"]]))
V["mean_over_factors"] = agg
V["worst_max_k_E_over_everything"] = float(max(
    max(r["c_max_k_E_heldout_worst"], r["d_max_k_E_FULLFEATURE_heldout_worst"])
    for r in V["factors"]))

log("")
log("=== VERDICT SUMMARY (mean over the six factors) ===")
for k, v in agg.items():
    log(f"  {k:48s} = {v:.4f}")
log(f"  worst max_k E_k over EVERY real-image query anywhere = "
    f"{V['worst_max_k_E_over_everything']:.2f}  (the gate is at 0.0)")

json.dump(V, open(OUT / "VERIFY_LIVESET.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'VERIFY_LIVESET.json'}")
