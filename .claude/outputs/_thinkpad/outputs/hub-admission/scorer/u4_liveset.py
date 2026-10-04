"""UNIT 4 -- CONTROL (d): THE LIVE SET.  Hold out a VALUE, refuse it; write it back, accept it.

This is the claim no distance threshold can fake by construction: the gate is a functional
of the atoms that are ACTUALLY IN THE STORE, so deleting every row carrying a value must
move the decision for exactly that value and for nothing else.

SIX stores, one per factor: rebuilt from ``store_train \\ rows(v)`` for one value v.
⛔ FROM CACHED FEATURES ONLY.  The cache is ``scope="full"`` (480,000 rows = the whole
split), so every row this unit touches is already in it and the SHORTFALL IS ZERO.  That
is asserted, not assumed -- a shortfall > 0 is falsifier SS5.6 and stops the unit.

QUERIES (the "black cube" set, on REAL images):
    exemplar_heldout : real cached features of images carrying the held-out value v,
                       contributing ONLY the channels the factor-to-well map assigns to
                       that factor; the other wells' channels come from the symbol-mode
                       R8 embedding of the same row's request.
    exemplar_invocab : the SS5.9 control -- the identical construction on images carrying a
                       value that is STILL in the store.  If these two refuse at within
                       0.05 of each other the encoder does not separate and exemplar
                       refusal is dead.
    writeback        : exemplar_heldout's images against the FULL store -- must be ACCEPTED.
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

SEED, BUDGET, Q = 0, "A", 0.95
N_QUERY = int(os.environ.get("N_QUERY", "256"))
N_CAL = int(os.environ.get("N_CAL", "2048"))

ctx = R.build_context(SEED)
full = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land_full, store_full, grid = full["land"], full["store"], ctx["grid"]
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float32).reshape(pool.size, -1)
F_pool = np.asarray(grid.factor_indices_of(pool), np.int64)
wc = ctx["well_channels"]
log(f"full store: K={store_full.n_wells} N={store_full.n_atoms}  pool={pool.size}")

# the factor -> well map, from UNIT 2
fw = json.load(open(OUT / "FACTOR_WELL_MAP_seed0.json"))
log("factor -> well map (UNIT 2): " +
    ", ".join(f"{n}->{fw['map'][n]['assigned_wells']}" for n in names))

rng = np.random.default_rng(20260917)
rec = {"schema": "admission-liveset/1", "seed": SEED, "budget": BUDGET, "q": Q,
       "n_query": N_QUERY, "held_out_value_rule": "v = cards[f] // 2 (the middle grid value)",
       "cache_scope": ctx["cache_meta"].get("scope"),
       "factor_well_map": {n: fw["map"][n]["assigned_wells"] for n in names},
       "factors": [], "shortfall_total": 0}

protos_full = FactorPrototypes.fit(grid, X_pool, pool)


def hybrid(X_real_rows, reqs, wells, protos):
    """The exemplar launch: ``wells``' channels from the REAL cached feature, the rest
    from the symbol-mode R8 embedding of the same request."""
    x = protos.embed_many(reqs).reshape(len(reqs), -1).copy()
    for k in wells:
        cols = np.asarray(wc[k], np.int64)
        x[:, cols] = X_real_rows[:, cols]
    return x


for f, nm in enumerate(names):
    v_idx = cards[f] // 2
    v_val = float(np.asarray(grid.values[f], np.float64)[v_idx])
    wells = list(fw["map"][nm]["assigned_wells"])
    held_mask = F_pool[:, f] == v_idx
    keep = ~held_mask
    n_held, n_keep = int(held_mask.sum()), int(keep.sum())
    log("")
    log(f"=== {nm}: hold out value index {v_idx} (={v_val:g}); wells {wells}; "
        f"removing {n_held} of {pool.size} rows, {n_keep} remain ===")

    # -- SS5.6: the shortfall.  Every row used is a store_train row, already cached.
    shortfall = int(np.setdiff1d(pool[keep], pool).size) + int(np.setdiff1d(pool[held_mask], pool).size)
    if shortfall != 0:
        raise AssertionError(f"⛔ FALSIFIER §5.6: shortfall = {shortfall} rows on {nm}")

    Xk, Pk = X_pool[keep], pool[keep]
    land_r = build_landscape(ctx["config"], BUDGET, Xk, factor_labels(grid, Pk), wc,
                             ctx["feat_shape"], cards, SEED, log=lambda *a, **k: None)["land"]
    store_r = land_r.store
    ident = AL.identity_E_le_0_on_live_store(land_r)
    # ⛔ no atom of the rebuilt store may carry the held-out value
    atom_rows_global = Pk[np.asarray(store_r.atom_rows, np.int64)]
    leak = int((np.asarray(grid.factor_indices_of(atom_rows_global), np.int64)[:, f] == v_idx).sum())
    log(f"  rebuilt: sigma={np.round(np.asarray(land_r.sigma),4).tolist()} "
        f"identity={ident['identity_E_le_0_holds']} worst={ident['max_over_all_wells']:.2f} "
        f"atoms carrying the held-out value = {leak} (must be 0)")
    if leak:
        raise AssertionError(f"⛔ the rebuilt store still holds {leak} atoms with the held-out value")

    # -- query rows
    held_idx = np.flatnonzero(held_mask)
    held_idx = held_idx[rng.permutation(held_idx.size)[:N_QUERY]]
    other_v = (v_idx + 1) % cards[f]
    in_idx = np.flatnonzero(F_pool[:, f] == other_v)
    in_idx = in_idx[rng.permutation(in_idx.size)[:N_QUERY]]

    req_held = requests_from_rows(grid, pool[held_idx])
    req_in = requests_from_rows(grid, pool[in_idx])
    protos_r = FactorPrototypes.fit(grid, Xk, Pk)

    X_held_r = hybrid(X_pool[held_idx], req_held, wells, protos_r)      # vs REBUILT store
    X_in_r = hybrid(X_pool[in_idx], req_in, wells, protos_r)            # vs REBUILT store
    X_held_f = hybrid(X_pool[held_idx], req_held, wells, protos_full)   # WRITE-BACK, full store

    # -- accept companions on the rebuilt store (§7b: A7 beside every row)
    cal_idx = rng.permutation(n_keep)[:N_CAL]
    req_cal = requests_from_rows(grid, Pk[cal_idx])
    X_cal = protos_r.embed_many(req_cal).reshape(len(req_cal), -1)
    val_rows = np.asarray(ctx["val_rows"], np.int64)
    val_rows = val_rows[np.isin(val_rows, Pk)]                          # val corners still in the store
    req_val = requests_from_rows(grid, val_rows)
    X_val = protos_r.embed_many(req_val).reshape(len(req_val), -1)
    req_a7 = R.request_sets(ctx, {"A7"})["A7"]
    X_a7 = protos_r.embed_many(req_a7).reshape(len(req_a7), -1)

    g_w = AC.KnnGate.fit(store_r, X_cal, "whole", Q)
    g_f = AC.KnnGate.fit(store_r, X_cal, "factored", Q)

    def rates(L, S, X):
        g = AL.gate(L, X)
        return {"accept": float(g["accept"].mean()), "refuse": float(g["refuse"].mean()),
                "certified_fraction_of_refusals": (
                    float(g["certified_refusal"].sum()) / max(int(g["refuse"].sum()), 1)
                    if int(g["refuse"].sum()) else None),
                "mean_max_E": float(g["E"].max(1).mean()),
                "per_well_refuse": [float(x) for x in g["refuse_well"].mean(0)]}

    row = {"factor": nm, "factor_index": f, "held_value_index": v_idx, "held_value": v_val,
           "wells_assigned": wells, "n_rows_removed": n_held, "n_rows_remaining": n_keep,
           "shortfall": shortfall, "atoms_leaking_held_value": leak,
           "sigma_rebuilt": [float(s) for s in np.asarray(land_r.sigma)],
           "identity_holds_rebuilt": bool(ident["identity_E_le_0_holds"]),
           "CLU": {
               "exemplar_heldout_refuse": rates(land_r, store_r, X_held_r)["refuse"],
               "exemplar_invocab_refuse": rates(land_r, store_r, X_in_r)["refuse"],
               "writeback_accept": rates(land_full, store_full, X_held_f)["accept"],
               "accept_val_rebuilt": rates(land_r, store_r, X_val)["accept"],
               "accept_A7_rebuilt": rates(land_r, store_r, X_a7)["accept"],
               "detail_heldout": rates(land_r, store_r, X_held_r),
               "detail_invocab": rates(land_r, store_r, X_in_r),
               "detail_writeback": rates(land_full, store_full, X_held_f)},
           "knn_factored": {
               "exemplar_heldout_refuse": float(g_f.decide(store_r, X_held_r)["refuse"].mean()),
               "exemplar_invocab_refuse": float(g_f.decide(store_r, X_in_r)["refuse"].mean()),
               "accept_val_rebuilt": float(g_f.decide(store_r, X_val)["accept"].mean()),
               "accept_A7_rebuilt": float(g_f.decide(store_r, X_a7)["accept"].mean())},
           "knn_whole": {
               "exemplar_heldout_refuse": float(g_w.decide(store_r, X_held_r)["refuse"].mean()),
               "exemplar_invocab_refuse": float(g_w.decide(store_r, X_in_r)["refuse"].mean()),
               "accept_val_rebuilt": float(g_w.decide(store_r, X_val)["accept"].mean()),
               "accept_A7_rebuilt": float(g_w.decide(store_r, X_a7)["accept"].mean())}}
    c = row["CLU"]
    row["separation_heldout_minus_invocab"] = c["exemplar_heldout_refuse"] - c["exemplar_invocab_refuse"]
    row["F5_9_fires"] = bool(abs(row["separation_heldout_minus_invocab"]) < 0.05)
    rec["factors"].append(row)
    log(f"  ⭐ CLU   refuse(held-out)={c['exemplar_heldout_refuse']:.4f}  "
        f"refuse(in-vocab ctrl)={c['exemplar_invocab_refuse']:.4f}  "
        f"sep={row['separation_heldout_minus_invocab']:+.4f}  "
        f"write-back accept={c['writeback_accept']:.4f}")
    log(f"     accept_val={c['accept_val_rebuilt']:.4f}  accept_A7={c['accept_A7_rebuilt']:.4f}  "
        f"certified frac of refusals={c['detail_heldout']['certified_fraction_of_refusals']}")
    log(f"     knn_factored refuse(held)={row['knn_factored']['exemplar_heldout_refuse']:.4f} "
        f"refuse(in-vocab)={row['knn_factored']['exemplar_invocab_refuse']:.4f} "
        f"accept_val={row['knn_factored']['accept_val_rebuilt']:.4f} "
        f"accept_A7={row['knn_factored']['accept_A7_rebuilt']:.4f}")
    log(f"     §5.9 fires (|sep| < 0.05)? {row['F5_9_fires']}")

# ---------------------------------------------------------------------------
agg = {k: float(np.mean([r["CLU"][k] for r in rec["factors"]]))
       for k in ("exemplar_heldout_refuse", "exemplar_invocab_refuse", "writeback_accept",
                 "accept_val_rebuilt", "accept_A7_rebuilt")}
rec["mean_over_factors_CLU"] = agg
rec["mean_over_factors_knn_factored"] = {
    k: float(np.mean([r["knn_factored"][k] for r in rec["factors"]]))
    for k in ("exemplar_heldout_refuse", "exemplar_invocab_refuse",
              "accept_val_rebuilt", "accept_A7_rebuilt")}
rec["bar_liveset_refuse_ge_0.90"] = bool(agg["exemplar_heldout_refuse"] >= 0.90)
rec["bar_liveset_writeback_ge_0.90"] = bool(agg["writeback_accept"] >= 0.90)
rec["F5_6_fires"] = bool(rec["shortfall_total"] != 0)
rec["F5_9_fires_any"] = bool(any(r["F5_9_fires"] for r in rec["factors"]))

log("")
log("=== LIVE-SET SUMMARY (mean over the six factors) ===")
log(f"  CLU  refuse(held-out exemplar) = {agg['exemplar_heldout_refuse']:.4f}   "
    f"bar >= 0.90 -> {rec['bar_liveset_refuse_ge_0.90']}")
log(f"  CLU  accept(write-back)        = {agg['writeback_accept']:.4f}   "
    f"bar >= 0.90 -> {rec['bar_liveset_writeback_ge_0.90']}")
log(f"  CLU  refuse(in-vocab control)  = {agg['exemplar_invocab_refuse']:.4f}")
log(f"  CLU  accept_val = {agg['accept_val_rebuilt']:.4f}   accept_A7 = {agg['accept_A7_rebuilt']:.4f}")
log(f"  knn_factored: {rec['mean_over_factors_knn_factored']}")
log(f"  §5.6 fires? {rec['F5_6_fires']}   §5.9 fires on any factor? {rec['F5_9_fires_any']}")

json.dump(rec, open(OUT / "LIVESET_seed0_A.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'LIVESET_seed0_A.json'}")
