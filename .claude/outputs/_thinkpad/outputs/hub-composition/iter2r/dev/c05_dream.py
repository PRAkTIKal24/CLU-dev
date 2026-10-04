"""C05 — ⭐ DREAMING ON THE **LINEAR** READ, at the pre-registered finite-T grid.

    python c05_dream.py <tag> <seed> <corner,corner,...> [steps]

⛔ SELECTION ONLY.  The corner never touches eval_a / eval_b (`dev.dev_setup`'s guard).

WHAT IS NEW HERE, AND WHY IT IS NOT A REDISCOVERY
-------------------------------------------------
Iteration 1 measured dreaming on the **argmax** read and found it **INERT**: at T = 0.5 the
settled state gave 463 distinct / 4 valid-novel of 512, but the **zero-step launch gave the
same**, and β = 0 agreed with β = 1 at every grid temperature.  ⛔ Inherited fact, ⛔ not
re-derived.

The addendum's live question is different: *the argmax read is pinned to a vertex and the
LINEAR read is not, so the same rollout may not be inert under it.*  Concretely,
    argmax read:  x -> code_k = argmin_m d²_km(x) -> gather
is a step function of the state — the whole interior of a Voronoi cell decodes identically,
so motion that does not cross a cell boundary is invisible **by construction**.
    linear read:  π_k(x) = softmax_m(−d²_km(x)/2τ²),  x̂ = Σ_m π_km · mode_km
is continuous in x, and the argmax read is exactly its τ → 0 limit.  ⇒ if the rollout moves
the state at all, the linear read can see it.  ⭐ That is the only reason this unit exists.

THE PROTOCOL, DECLARED BEFORE IT RAN
------------------------------------
⛔ `temp ∈ {0.0, 0.005, 0.05, 0.5}` — the PRE-REGISTERED grid, ⛔ not widened, before or
after seeing a result.
β, the analogue of iteration 1's `request_strength`, on MY object:
  **β = 0 — DREAMING.** The launch carries **no request information at all**: `init_state`
  with nothing clamped and `init_mode="prior_sample"`, i.e. a mode drawn from the store's
  own empirical marginal P(code_k).  ⛔ No request, no head, no feature.
  **β = 1 — the steered control.** The launch is the request-addressed linear read
  `clu_wide_vector(store, head.coefficients(request))` — the object itself.
Both then run the SAME rollout at the SAME temperature and are read the SAME way.

THE THREE CONTROLS THAT DECIDE THE VERDICT
------------------------------------------
⭐ **A — the zero-step launch**, decoded through the identical readout.  If the settled
state does not beat it, the number is the launch's and ⛔ not dreaming's.
⭐ **B — β = 0 vs β = 1.**  If they agree, the request is not steering the chain.
⭐ **C — state motion.**  ‖x_settled − x_launch‖/‖x_launch‖ and the fraction of wells whose
argmax code changed.  ⭐ This separates *"the dynamics did nothing"* from *"the dynamics did
something the ARGMAX read could not see"* — iteration 1 could not tell those apart, and
that distinction is the whole point of running this on the linear read.

⭐ If the settled readout does not beat the launch at any grid temperature, the verdict is
**DEAD, in those words**.  ⛔ Never reported as a null.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp  # noqa: E402
import ctx  # noqa: E402
import dev  # noqa: E402
import fast  # noqa: E402
import stores  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "dream"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CORNERS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,2,4").split(",")]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]

TEMP_GRID = [0.0, 0.005, 0.05, 0.5]      # ⛔ PRE-REGISTERED. Never widened.
BETAS = [0.0, 1.0]                        # 0 = dreaming, 1 = the steered control
OUTP = stores.CACHE.parent / f"c05_dream_{TAG}_seed{SEED}.json"


def soft_posterior(land, X, tau: float) -> np.ndarray:
    """π_k(x) = softmax_m(−d²_km(x)/2τ²) — ⭐ straight out of the PINNED
    `gen_dream.well_energies`, which is the same occupancy the landscape's own energy
    uses.  ⛔ Not a re-spelling of it: it IS it.

    ⛔ `X` is the DREAM STATE — the store's own composed state — ⛔ never a target
    feature.  Nothing on this path takes a feature argument.
    """
    from chlu.core.gen_dream import well_energies
    _e, pis = well_energies(land, np.asarray(X, np.float32), float(tau))
    return np.stack([np.asarray(p, np.float32) for p in pis], axis=1)   # (n, K, M)


T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]
res = {"seed": SEED, "temp_grid": TEMP_GRID, "betas": BETAS, "steps": STEPS,
       "config": {"K": K, "M": M, "rank": RANK, "ridge": RIDGE, "pairs": PAIR_NAMES},
       "beta_definition": {
           "0": "DREAMING — init_state, nothing clamped, init_mode='prior_sample' "
                "(a mode drawn from the store's own empirical marginal). NO request.",
           "1": "the steered control — the request-addressed LINEAR read as the launch"},
       "corners": {}}
if OUTP.exists():
    res = json.load(open(OUTP, encoding="utf-8"))

import jax  # noqa: E402
from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import (DreamLandscape, RolloutCfg, dream_rollout,  # noqa: E402
                                 fit_code_coupling, init_state,
                                 mode_distance_scale, requests_from_labels)
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_store import read_from_codes  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))

for ci in CORNERS:
    if str(ci) in res["corners"]:
        ctx.log(f"[c{ci}] already in {OUTP.name} — skipped")
        continue
    t0 = time.time()
    s = dev.dev_setup(c, comp.CORNERS[ci], log=ctx.log)
    X_store = c["gather"](s["pool"])
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    lab_store = c["factor_labels"](grid, s["pool"])
    lab_dev = s["lab_dev"]
    n = int(lab_dev.shape[0])
    pool_tup = comp.pool_tuple_array(grid, s["pool"], c["factor_labels"])
    perm, modes = stores.build(c, ci, s["pool"], X_store, K, M, SEED)
    st = stores.to_gen_store(perm, modes, c["feat_shape"])
    pq = fast.EqualPQ(perm, np.asarray(modes, np.float32), (D,))
    head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
    a_req = head.coefficients(requests_from_labels(lab_dev, "full", 0, SEED))

    codes_pool = pq.codes(Zs)
    J = fit_code_coupling(codes_pool, M)
    land = DreamLandscape(st, J, 1.0)
    tau = float(mode_distance_scale(st)["scale"])
    prior = np.stack([np.bincount(codes_pool[:, k], minlength=M) + 1.0
                      for k in range(K)])
    prior = prior / prior.sum(1, keepdims=True)

    # ---- the two launches ---------------------------------------------------
    x_steer = np.asarray(clu_wide_vector(st, a_req), np.float32).reshape((-1, D))
    no_clamp = {"code": np.zeros((n, K), np.int64), "clamped": np.zeros((n, K), bool)}
    init = init_state(st, no_clamp, prior, "prior_sample", SEED)
    x_dream = np.asarray(init["x0"], np.float32).reshape((-1, D))
    coord_free = np.asarray(init["coord_clamped"], bool)      # all False -> all free
    assert not coord_free.any(), "⛔ nothing may be clamped: β enters through the LAUNCH"
    launches = {0.0: x_dream, 1.0: x_steer}

    ctx.log(f"[c{ci}] tau={tau:.4f} (store's own mode-distance scale); "
            f"n={n}; steps={STEPS}")
    cA = {}
    for beta, x0 in launches.items():
        pi0 = soft_posterior(land, x0, tau)
        cA[f"beta{beta:g}"] = {
            "LINEAR": comp.headline(c, np.asarray(clu_wide_vector(st, pi0), np.float32),
                                    lab_dev, pool_tup),
            "ARGMAX": comp.headline(c, np.asarray(read_from_codes(st, pq.codes(x0)),
                                                  np.float32), lab_dev, pool_tup)}
        ctx.log(f"[c{ci}] CONTROL A zero-step launch β={beta:g}: "
                f"LINEAR={cA[f'beta{beta:g}']['LINEAR']['tuples_valid_novel']:3d} "
                f"(distinct {cA[f'beta{beta:g}']['LINEAR']['distinct']:3d})  "
                f"ARGMAX={cA[f'beta{beta:g}']['ARGMAX']['tuples_valid_novel']:3d}")

    rows = []
    for beta in BETAS:
        x0 = launches[beta]
        n0 = float(np.linalg.norm(x0))
        c0 = pq.codes(x0)
        for T in TEMP_GRID:
            key = jax.random.PRNGKey(int(SEED) * 7919 + int(T * 1e6) * 31
                                     + int(beta * 13) + ci)
            if T > 0.0:
                cfg = RolloutCfg(sampler="mala", steps=STEPS, step_size=1e-3,
                                 tau_start=tau, tau_end=tau, temp_start=T, temp_end=T,
                                 uld_dt=1e-2, uld_gamma=1.0)
                out = dream_rollout(land, x0, coord_free, cfg, key, steps=STEPS,
                                    sampler="mala")
                xs = np.asarray(out["x"], np.float32).reshape((-1, D))
            else:
                # ⚠ DECLARED (LOG 12:45Z): the pinned `_geom` anneal computes a·(b/a)^t and
                # cannot represent a ZERO endpoint. The T = 0 leg is the exact noiseless
                # limit — the SAME pinned `_mala_step_jit` at temp = 0.0, with a constant
                # schedule. ⛔ The temperature is 0.0, not a small positive stand-in.
                import jax.numpy as jnp
                from chlu.core.gen_dream import _mala_step_jit
                dt_ = land.store.modes[0].dtype
                xj = jnp.asarray(np.asarray(x0, np.float32), dt_)
                freej = jnp.asarray((~np.asarray(coord_free, bool)).astype(np.float32), dt_)
                eps = jnp.asarray(1e-3, dt_)
                tz = jnp.asarray(0.0, dt_)
                tj = jnp.asarray(tau, dt_)
                for _i in range(STEPS):
                    key, sk = jax.random.split(key)
                    xj, _a, _acc, _p, _k, _e, _er = _mala_step_jit(
                        land, xj, freej, sk, eps, tz, tj, tj, True)
                xs = np.asarray(xj, np.float32).reshape((-1, D))
            moved = float(np.linalg.norm(xs - x0) / max(n0, 1e-30))
            cs = pq.codes(xs)
            frac_code = float((c0 != cs).mean())
            h_lin = comp.headline(
                c, np.asarray(clu_wide_vector(st, soft_posterior(land, xs, tau)), np.float32),
                lab_dev, pool_tup)
            h_arg = comp.headline(c, np.asarray(read_from_codes(st, cs), np.float32),
                                  lab_dev, pool_tup)
            rows.append({"beta": beta, "temp": T,
                         "settled_LINEAR": h_lin, "settled_ARGMAX": h_arg,
                         "C_state_motion_rel": moved,
                         "C_frac_wells_code_changed": frac_code})
            ctx.log(f"[c{ci}] β={beta:g} T={T:<6g} LINEAR={h_lin['tuples_valid_novel']:3d} "
                    f"(distinct {h_lin['distinct']:3d})  "
                    f"ARGMAX={h_arg['tuples_valid_novel']:3d}  "
                    f"|Δx|/|x|={moved:.5f}  codes moved={frac_code:.4f}")
            res["corners"][str(ci)] = {
                "corner": comp.CORNERS[ci], "tau": tau, "n": n,
                "CONTROL_A_zero_step_launch": cA, "grid": rows,
                "seconds": round(time.time() - t0, 1)}
            with open(OUTP, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2, default=str)
    ctx.log(f"[c{ci}] done in {time.time() - t0:.1f}s -> {OUTP}")

ctx.log(f"C05 done in {time.time() - T0:.1f}s")
