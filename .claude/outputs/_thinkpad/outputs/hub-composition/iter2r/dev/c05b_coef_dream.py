"""C05b — ⭐ DREAMING IN **COEFFICIENT SPACE**, at the same pre-registered grid.
Plus the corrected **T = 0** legs of C05, which the MALA acceptance ratio made degenerate.

    python c05b_coef_dream.py <seed> <corner,corner,...> [steps]

⛔ SELECTION ONLY. ⛔ The temperature grid is the pre-registered `{0.0, 0.005, 0.05, 0.5}`
and is NOT widened. Declared in LOG 12:45Z **before** either attempt was run.

PART 1 — THE COEFFICIENT-SPACE DREAM
------------------------------------
C05a measured that the object IS its coefficients: 47 % of `a` is negative, L1 ≈ 8.5 per
well, and **0.00 %** of wells are a probability vector, so the read lies outside the convex
hull of the stored modes and ⛔ no state-space dream can reach it.  A dream on THIS object
must therefore perturb `a`:

    a(T) = a(r) + T · σ_a ⊙ ξ ,   ξ ~ N(0, I) ,
    σ_a = the per-(well, mode) standard deviation of `a` across the POOL's own request tuples

**β = 1** takes `r` = the target request.  **β = 0** takes `r` = a uniformly random grid
tuple, so ⛔ no information about the target enters the launch at all.
⭐ T = 0 is then exactly the zero-step launch, which is CONTROL A, exact and free.

PART 2 — THE CORRECTED T = 0 STATE-SPACE LEG
--------------------------------------------
⚠ **A defect in my own C05 T = 0 leg, found by reading the pinned code after the run and
reported rather than quietly re-run.**  `_mala_step` forms
`log_alpha = −ΔE/temp + logq_rev − logq_fwd` with `logq_* = −‖·‖²/(4·eps·temp)`.  At
`temp = 0` both terms are `0/0` ⇒ `log_alpha` is **NaN** ⇒ `log(u) < NaN` is False ⇒ **every
proposal is rejected**.  ⭐ That is exactly what C05 measured: `|Δx|/|x| = 0.00000` at T = 0.
⛔ So C05's T = 0 row did NOT measure zero-temperature descent; it measured a degenerate
acceptance ratio.  The honest T = 0 dynamics is the **unadjusted** chain (`correct=False`),
whose step at `temp = 0` is `x ← x − eps·∇E`: pure noiseless gradient descent, which is what
§E.4 means by *"the settle runs at temp = 0.0"*.  Both are reported.
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

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CORNERS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,2,4").split(",")]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
TEMP_GRID = [0.0, 0.005, 0.05, 0.5]      # ⛔ PRE-REGISTERED. Never widened.
N_DRAWS = 3                               # independent ξ draws per (β, T), for a sd
OUTP = stores.CACHE.parent / f"c05b_coef_dream_seed{SEED}.json"

T0 = time.time()
c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards, grid = c["cards"], c["grid"]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import (DreamLandscape, _mala_step_jit,  # noqa: E402
                                 fit_code_coupling, init_state,
                                 mode_distance_scale, requests_from_labels)
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_store import read_from_codes  # noqa: E402

PAIRS = list(default_pairs(c["names"], PAIR_NAMES))
res = json.load(open(OUTP, encoding="utf-8")) if OUTP.exists() else {
    "seed": SEED, "temp_grid": TEMP_GRID, "steps": STEPS, "n_draws": N_DRAWS,
    "declared": "LOG 2026-09-17T12:45Z, before either dreaming attempt was run",
    "corners": {}}

for ci in CORNERS:
    if str(ci) in res["corners"]:
        ctx.log(f"[c{ci}] already done — skipped")
        continue
    t0 = time.time()
    s = dev.dev_setup(c, comp.CORNERS[ci], log=lambda *_a: None)
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
    rng = np.random.default_rng(int(SEED) * 999_331 + ci)

    a_target = np.asarray(head.coefficients(
        requests_from_labels(lab_dev, "full", 0, SEED)), np.float32)     # (n, K, M)
    # σ_a from the POOL's own request tuples — store-side only, ⛔ no eval feature
    sub = rng.choice(lab_store.shape[0], size=min(4096, lab_store.shape[0]), replace=False)
    a_pool = np.asarray(head.coefficients(np.asarray(lab_store, np.int64)[sub]), np.float32)
    sigma_a = a_pool.std(0)                                              # (K, M)
    # β = 0: a uniformly random grid tuple per item — ⛔ no target information
    lab_rand = np.stack([rng.integers(0, int(cv), size=n) for cv in cards], 1)
    a_rand = np.asarray(head.coefficients(lab_rand.astype(np.int64)), np.float32)
    ctx.log(f"[c{ci}] sigma_a: mean {sigma_a.mean():.4f} max {sigma_a.max():.4f}; "
            f"|a_target| mean {np.abs(a_target).mean():.4f}")

    def score(a):
        return comp.headline(c, np.asarray(clu_wide_vector(st, np.asarray(a, np.float32)),
                                           np.float32), lab_dev, pool_tup)

    rows = []
    for beta, a0 in ((0.0, a_rand), (1.0, a_target)):
        for T in TEMP_GRID:
            vals, ds = [], []
            reps = 1 if T == 0.0 else N_DRAWS     # T=0 is deterministic: it IS the launch
            for j in range(reps):
                xi = rng.standard_normal(a0.shape).astype(np.float32)
                h = score(a0 + float(T) * sigma_a[None] * xi)
                vals.append(h["tuples_valid_novel"])
                ds.append(h["distinct"])
            rows.append({"beta": beta, "temp": T, "n_draws": reps,
                         "tuples_valid_novel_mean": float(np.mean(vals)),
                         "tuples_valid_novel_sd": float(np.std(vals, ddof=1)) if reps > 1
                         else 0.0,
                         "tuples_valid_novel_all": [int(v) for v in vals],
                         "distinct_mean": float(np.mean(ds)),
                         "is_CONTROL_A_zero_step_launch": bool(T == 0.0)})
            ctx.log(f"[c{ci}] COEF β={beta:g} T={T:<6g} "
                    f"tuples_valid_novel={np.mean(vals):6.2f} "
                    f"(draws {vals})  distinct={np.mean(ds):5.1f}"
                    f"{'   <- CONTROL A (the launch itself)' if T == 0.0 else ''}")

    # ---- PART 2: the corrected T = 0 state-space legs ----------------------
    land = DreamLandscape(st, fit_code_coupling(pq.codes(Zs), M), 1.0)
    tau = float(mode_distance_scale(st)["scale"])
    prior = np.stack([np.bincount(pq.codes(Zs)[:, k], minlength=M) + 1.0
                      for k in range(K)])
    prior = prior / prior.sum(1, keepdims=True)
    x_steer = np.asarray(clu_wide_vector(st, a_target), np.float32).reshape((-1, D))
    init = init_state(st, {"code": np.zeros((n, K), np.int64),
                           "clamped": np.zeros((n, K), bool)}, prior, "prior_sample", SEED)
    x_dream = np.asarray(init["x0"], np.float32).reshape((-1, D))
    dt_ = st.modes[0].dtype
    freej = jnp.ones((n, D), dt_)
    t0rows = []
    from chlu.core.gen_dream import well_energies
    for beta, x0 in ((0.0, x_dream), (1.0, x_steer)):
        for correct in (True, False):
            key = jax.random.PRNGKey(int(SEED) * 41 + ci * 7 + int(beta) * 3 + int(correct))
            xj = jnp.asarray(np.asarray(x0, np.float32), dt_)
            for _i in range(STEPS):
                key, sk = jax.random.split(key)
                xj, _al, _ac, _p, _k, _e, _er = _mala_step_jit(
                    land, xj, freej, sk, jnp.asarray(1e-3, dt_), jnp.asarray(0.0, dt_),
                    jnp.asarray(tau, dt_), jnp.asarray(tau, dt_), bool(correct))
            xs = np.asarray(xj, np.float32).reshape((-1, D))
            _e2, pis = well_energies(land, xs, tau)
            pi = np.stack([np.asarray(p, np.float32) for p in pis], 1)
            h_lin = comp.headline(c, np.asarray(clu_wide_vector(st, pi), np.float32),
                                  lab_dev, pool_tup)
            h_arg = comp.headline(c, np.asarray(read_from_codes(st, pq.codes(xs)),
                                                np.float32), lab_dev, pool_tup)
            moved = float(np.linalg.norm(xs - x0) / max(float(np.linalg.norm(x0)), 1e-30))
            t0rows.append({"beta": beta, "temp": 0.0,
                           "metropolis_correction": bool(correct),
                           "what": ("MALA-corrected: ⚠ log_alpha is 0/0 = NaN at temp 0, "
                                    "so EVERY proposal is rejected — degenerate, reported "
                                    "to show why") if correct else
                                   ("UNADJUSTED: x <- x - eps*grad E, i.e. TRUE noiseless "
                                    "zero-temperature descent"),
                           "settled_LINEAR": h_lin, "settled_ARGMAX": h_arg,
                           "C_state_motion_rel": moved})
            ctx.log(f"[c{ci}] T=0 STATE β={beta:g} correct={correct}  "
                    f"LINEAR={h_lin['tuples_valid_novel']:3d} "
                    f"(distinct {h_lin['distinct']:3d})  "
                    f"ARGMAX={h_arg['tuples_valid_novel']:3d}  |Δx|/|x|={moved:.5f}")

    res["corners"][str(ci)] = {
        "corner": comp.CORNERS[ci], "n": n, "tau": tau,
        "sigma_a": {"mean": float(sigma_a.mean()), "max": float(sigma_a.max())},
        "PART1_coefficient_space_dream": rows,
        "PART2_corrected_T0_state_space": t0rows,
        "seconds": round(time.time() - t0, 1)}
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    ctx.log(f"[c{ci}] done in {time.time() - t0:.1f}s -> {OUTP}")

ctx.log(f"C05b done in {time.time() - T0:.1f}s")
