"""ADVERSARIAL VERIFICATION -- item 6: is there a launch point the hub MISSED at which
``E_k > 0`` is LIVE-SET sensitive?

The hub tested ``clamp_atom``, ``prior_atom``, ``posterior_mean``, the R8 affine head and
raw cached features.  ⭐ It did NOT test the one point the shipped read actually produces:
**the SETTLED state** ``settle_chain(land, x0, log_w, cfg)`` -- 1000 MALA steps down
``store_energy + beta*request_energy + lambda*coupling_energy``, with sigma annealed
4 -> 1.  That state is:
  * SHIPPED            (``gen_landscape.settle_chain``, the experiment's own call);
  * REQUEST-ONLY       (``assert_request_only`` covers the whole path; no feature can
                        enter through ``log_w``);
  * NOT FITTED         (no threshold, no quantile, no example of an invalid request);
  * and it is NOT in the atoms' convex hull -- the coupling term is an INDEFINITE
    bilinear form, so the minimiser can and does leave the hull.
⛔ The gate stays at ``E_k > 0``.  ⛔ sigma_k is not scaled.  Nothing is fitted.

THREE probes, all on the SETTLED state:
  P1  symbol mode, full store: valid ``val`` vs ``out_of_range`` w in {2,3,4,7,13}.
  P2  ⭐ LIVE SET, exemplar: the rebuilt held-out-value store, settled FROM THE HYBRID
      launch (the hub's own exemplar construction) -- held-out vs in-vocabulary.
  P3  ⭐ LIVE SET, symbol: the rebuilt store, settled from the shipped ``clamp_atom``
      launch -- a request naming the DELETED value vs one naming a live value.

WRITES: VERIFY_LAUNCH.json under $HUB_OUT.
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

import jax  # noqa: E402
import refusal_lib as R  # noqa: E402
from chlu.config import gen_dynamical_read_config  # noqa: E402
from chlu.core.gen_landscape import SettleCfg, read_settled, settle_chain  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.eval.gen_probe import factor_labels  # noqa: E402
from chlu.experiments.exp_gen_dynamical_read import build_landscape  # noqa: E402

T0 = time.time()


def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


SEED, BUDGET = 0, "A"
STEPS = int(os.environ.get("STEPS", "1000"))
NQ = int(os.environ.get("NQ", "128"))
RNG = np.random.default_rng(31337)

ctx = R.build_context(SEED)
grid = ctx["grid"]
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
pool = np.asarray(ctx["pool"], np.int64)
X_pool = np.asarray(ctx["X_store"], np.float64).reshape(pool.size, -1)
F_pool = np.asarray(grid.factor_indices_of(pool), np.int64)
wc = ctx["well_channels"]
WC = [np.asarray(w, np.int64) for w in wc]
c = gen_dynamical_read_config(ctx["config"])
CFG = SettleCfg(steps=STEPS, step_size=float(c.step_size),
                sigma_scale_start=float(c.sigma_scale_start),
                sigma_scale_end=float(c.sigma_scale_end),
                temp_start=float(c.temp_start), temp_end=float(c.temp_end),
                metropolis=bool(c.metropolis))
log(f"SettleCfg = {CFG.as_dict()}  (STEPS={STEPS}, NQ={NQ})")

full = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land_full, store_full, book_full = full["land"], full["store"], full["book"]
protos_full = FactorPrototypes.fit(grid, ctx["X_store"], ctx["pool"])
log(f"full store: coupling_strength={float(land_full.coupling_strength):.4g} "
    f"request_strength={float(land_full.request_strength):.4g}")

V = {"schema": "verify-admission-launch/1", "seed": SEED, "budget": BUDGET,
     "steps": STEPS, "n_query": NQ, "settle_cfg": CFG.as_dict(),
     "coupling_strength_used": float(land_full.coupling_strength),
     "probes": {}}


def gate_of(L, X):
    atoms = [np.asarray(a, np.float64) for a in L.store.atoms]
    wcl = [np.asarray(w, np.int64) for w in L.store.well_channels]
    sig = np.asarray(L.sigma, np.float64)
    X = np.asarray(X, np.float64)
    n = X.shape[0]
    E = np.empty((n, len(atoms)))
    D = np.empty((n, len(atoms)))
    for k, (A, cols) in enumerate(zip(atoms, wcl)):
        s2 = float(sig[k]) ** 2
        aa = (A * A).sum(1)[None, :]
        z = X[:, cols]
        d2 = np.maximum((z * z).sum(1)[:, None] - 2.0 * (z @ A.T) + aa, 0.0)
        m = d2.min(1)
        S = np.exp(-(d2 - m[:, None]) / (2.0 * s2)).sum(1)
        E[:, k] = 0.5 * m - s2 * np.log(S)
        D[:, k] = np.sqrt(m)
    ref = (E > 0.0).any(1)
    return {"E": E, "dos": D / sig[None, :], "refuse": ref,
            "refuse_rate": float(ref.mean()), "max_k_E": E.max(1),
            "mean_max_E": float(E.max(1).mean()), "dos_min": float((D / sig).min(1).mean())}


def settle(L, book, reqs, x0=None, tag=""):
    lw, nt, _ = R.term_experts(book, grid, reqs)
    if x0 is None:
        lau = R.launch_from_terms(L, book, lw, nt, float(c.clamp_threshold))
        x0 = lau["x0"]
    key = jax.random.PRNGKey(int(c.row_seed_offset) + SEED)
    t = time.time()
    out = settle_chain(L, np.asarray(x0, np.float32), np.ascontiguousarray(lw), CFG, key)
    xs = np.asarray(out["x"], np.float32)
    E = np.asarray(out["trace"]["energy_ref"], np.float64)
    disp = np.asarray(out["trace"]["displacement"], np.float64)
    log(f"    settled [{tag}] n={len(reqs)} in {time.time()-t:.1f}s  "
        f"E_ref {E[0].mean():.1f} -> {E[-1].mean():.1f}  "
        f"total displacement {disp.sum(0).mean():.3f}")
    return xs, x0, {"E_ref_first": float(E[0].mean()), "E_ref_last": float(E[-1].mean()),
                    "total_displacement_mean": float(disp.sum(0).mean())}


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
# P1 -- symbol mode on the FULL store: does the SETTLED state cross the level set?
# ---------------------------------------------------------------------------
log("")
log("=== P1: the SETTLED state, symbol mode, full store ===")
val_rows = np.asarray(ctx["val_rows"], np.int64)[:NQ]
req_val = requests_from_rows(grid, val_rows)
P1 = {}
xs_v, x0_v, tr_v = settle(land_full, book_full, req_val, tag="val")
g_x0 = gate_of(land_full, x0_v)
g_xs = gate_of(land_full, xs_v)
g_mix = gate_of(land_full, read_settled(land_full, xs_v, "atom_mixture").reshape(len(req_val), -1))
P1["val"] = {"launch_clamp_atom": {k: g_x0[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
             "settled_state": {k: g_xs[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
             "settled_atom_mixture": {k: g_mix[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
             "trace": tr_v}
log(f"  val   : launch refuse={g_x0['refuse_rate']:.4f} (E={g_x0['mean_max_E']:.1f})  "
    f"SETTLED refuse={g_xs['refuse_rate']:.4f} (E={g_xs['mean_max_E']:.1f}, "
    f"d/s={g_xs['dos_min']:.3f})  atom_mixture refuse={g_mix['refuse_rate']:.4f}")
for w in (2.0, 3.0, 4.0, 7.0, 13.0):
    req_o = perturb(req_val, 0, offgrid(0, w))      # floor_hue, the cleanest factor
    xs_o, x0_o, tr_o = settle(land_full, book_full, req_o, tag=f"oor w={w:g}")
    go0, gos = gate_of(land_full, x0_o), gate_of(land_full, xs_o)
    P1[f"oor_floor_hue_w{w:g}"] = {
        "launch_clamp_atom": {k: go0[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
        "settled_state": {k: gos[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
        "trace": tr_o}
    log(f"  oor w={w:<4g}: launch refuse={go0['refuse_rate']:.4f}  "
        f"SETTLED refuse={gos['refuse_rate']:.4f} (E={gos['mean_max_E']:.1f}, "
        f"d/s={gos['dos_min']:.3f})")
V["probes"]["P1_settled_symbol_full_store"] = P1

# ---------------------------------------------------------------------------
# P2 / P3 -- the LIVE SET, on the rebuilt floor_hue store (single assigned well)
# ---------------------------------------------------------------------------
fw = json.load(open(OUT / "FACTOR_WELL_MAP_seed0.json"))
for fname in ("floor_hue", "wall_hue"):
    f = names.index(fname)
    v_idx = cards[f] // 2
    v_val = float(np.asarray(grid.values[f], np.float64)[v_idx])
    wells = list(fw["map"][fname]["assigned_wells"])
    held_mask = F_pool[:, f] == v_idx
    keep = ~held_mask
    Xk, Pk = X_pool[keep], pool[keep]
    log("")
    log(f"=== P2/P3 on the REBUILT {fname} store (held-out value {v_idx}, wells {wells}) ===")
    br = build_landscape(ctx["config"], BUDGET, np.asarray(Xk, np.float32),
                         factor_labels(grid, Pk), wc, ctx["feat_shape"], cards, SEED,
                         log=lambda *a, **k: None)
    land_r, book_r = br["land"], br["book"]
    protos_r = FactorPrototypes.fit(grid, np.asarray(Xk, np.float32), Pk)

    held_idx = np.flatnonzero(held_mask)
    held_idx = held_idx[RNG.permutation(held_idx.size)[:NQ]]
    other_v = (v_idx + 1) % cards[f]
    in_idx = np.flatnonzero(F_pool[:, f] == other_v)
    in_idx = in_idx[RNG.permutation(in_idx.size)[:NQ]]
    req_held = requests_from_rows(grid, pool[held_idx])
    req_in = requests_from_rows(grid, pool[in_idx])

    def hybrid(Xreal, reqs):
        x = np.asarray(protos_r.embed_many(reqs), np.float64).reshape(len(reqs), -1).copy()
        for k in wells:
            x[:, WC[k]] = Xreal[:, WC[k]]
        return x

    # -- P2: settle FROM the exemplar hybrid launch
    P2 = {}
    for tag, rq, Xr in (("heldout", req_held, X_pool[held_idx]),
                        ("invocab", req_in, X_pool[in_idx])):
        x0h = hybrid(Xr, rq)
        xs, _, tr = settle(land_r, book_r, rq, x0=x0h, tag=f"P2 {fname}/{tag}")
        g0, gs = gate_of(land_r, x0h), gate_of(land_r, xs)
        P2[tag] = {"launch_hybrid": {k: g0[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
                   "settled_state": {k: gs[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
                   "trace": tr}
        log(f"  P2 {tag:8s}: launch refuse={g0['refuse_rate']:.4f} (E={g0['mean_max_E']:.1f}) "
            f"-> SETTLED refuse={gs['refuse_rate']:.4f} (E={gs['mean_max_E']:.1f}, "
            f"d/s={gs['dos_min']:.3f})")
    P2["separation_settled"] = (P2["heldout"]["settled_state"]["refuse_rate"]
                                - P2["invocab"]["settled_state"]["refuse_rate"])
    log(f"  ⭐ P2 separation at the SETTLED state = {P2['separation_settled']:+.4f}")

    # -- P3: OOV symbol, settle from the shipped clamp_atom launch
    P3 = {}
    req_val_r = requests_from_rows(grid, np.asarray(ctx["val_rows"], np.int64)[:NQ])
    for tag, rq in (("oov_symbol", perturb(req_val_r, f, v_val)),
                    ("in_vocab_symbol", req_val_r)):
        xs, x0, tr = settle(land_r, book_r, rq, tag=f"P3 {fname}/{tag}")
        g0, gs = gate_of(land_r, x0), gate_of(land_r, xs)
        P3[tag] = {"launch_clamp_atom": {k: g0[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
                   "settled_state": {k: gs[k] for k in ("refuse_rate", "mean_max_E", "dos_min")},
                   "trace": tr}
        log(f"  P3 {tag:16s}: launch refuse={g0['refuse_rate']:.4f} -> SETTLED "
            f"refuse={gs['refuse_rate']:.4f} (E={gs['mean_max_E']:.1f}, d/s={gs['dos_min']:.3f})")
    P3["separation_settled"] = (P3["oov_symbol"]["settled_state"]["refuse_rate"]
                                - P3["in_vocab_symbol"]["settled_state"]["refuse_rate"])
    log(f"  ⭐ P3 separation at the SETTLED state = {P3['separation_settled']:+.4f}")
    V["probes"][f"P2_settled_exemplar_{fname}"] = P2
    V["probes"][f"P3_settled_oov_symbol_{fname}"] = P3

json.dump(V, open(OUT / "VERIFY_LAUNCH.json", "w"), indent=2, default=str)
log(f"wrote {OUT/'VERIFY_LAUNCH.json'}")
