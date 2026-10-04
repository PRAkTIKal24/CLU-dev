"""The three specialist rivals: the prototype table, LACE, and the PoE-EBM.

Nothing here reimplements a rival. The prototype table is
:class:`chlu.core.gen_request.FactorPrototypes`, the LACE arm is
``exp_gen_cell.run_lace_arm`` and the energy-based arm is
``exp_gen_cell.run_ebm_arm``, all called as shipped. Rewriting a competitor's
method inside our own module is the surest way to weaken it by accident.

What this module adds is one thing: **the LACE selection repair**.

The LACE repair
---------------
``GenCellConfig.lace_selection_source`` points at
``gen-arms-a7-wiring/runs/historical_688128``, and ``exp_gen_cell.lace_selection``
INHERITS ``(sigma, form, shrinkage)`` from that manifest instead of re-selecting.
That inherited configuration was selected under the old head/tail cut, but the
cell scores it on the STRATIFIED cut that ``split_fit_val`` now makes. The
configuration is therefore selected for a different fit set than the one it is
scored on, and the harness flags ``lace`` as regressed - 0.6136 measured against
a published 0.7077.

The repair is to re-run the SAME 36-trial selection - the same grid, the same
short schedule, the same criterion (validation aggregate probe accuracy, C never
enters) - on the stratified rows the arm is actually scored on.

The repair lives HERE. ``chlu/config.py`` is not edited: the pinned
``lace_selection_source`` stays exactly as it is, the historical leg is still run
from it, and BOTH numbers are printed side by side so the change is auditable.
Repairing a rival by editing a pinned config would make the two legs
unreproducible against each other.

The PoE-EBM
-----------
Runs at its OWN budget (independent per-factor experts, hidden 256, depth 2), not
at the 524,288 B the store is held to, and the byte multiple is printed beside
every row it produces. It is run TWICE on independent sampler keys, because a
72-item corner read from a stochastic sampler cannot carry a confident claim from
one draw. If it wins outright it goes in the appendix WITH its numbers.
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import numpy as np

from ...core.gen_compose import COMPOSITIONAL_FACTOR
from ...eval.rivals import gen_lace as lace
from ...experiments.exp_gen_baselines import _slice_dense, score_composition
from ...experiments.exp_gen_cell import (
    lace_selection,
    run_ebm_arm,
    run_lace_arm,
)

REQUEST_SETS = ("B", "MS")


# ---------------------------------------------------------------------------
# 1. The prototype table - the incumbent
# ---------------------------------------------------------------------------


def run_table(ctx: dict, log: Callable = print) -> dict:
    """Per-value means, composed additively. The incumbent specialist.

    ``x_hat(request) = mu_global + sum_terms (mu_{k,v} - mu_global)``, fitted on
    the fit rows only. It is the closed-form solution to the same request ->
    feature regression the generic learners are trained for, which is why it is
    the parameter-matching reference.

    It interpolates and extrapolates off-grid values through
    ``FactorPrototypes.prototype``, so - like the continuous-input MLP and unlike
    LACE's closed categorical vocabulary - it CAN enter row 2.
    """
    t0 = time.time()
    protos = ctx["protos"]
    from ...core.gen_request import requests_from_rows

    req_dicts = {k: requests_from_rows(ctx["grid"], ctx["row_sets"][k])
                 for k in REQUEST_SETS}
    feats = {k: np.asarray(protos.embed_many(req_dicts[k]), np.float32)
             for k in REQUEST_SETS}
    led = protos.byte_ledger()
    log(f"  [table] {led['n_prototypes']} prototypes x D={led['dim']} = "
        f"{led['bytes']} B ({led['bytes'] / 4:.0f} float32 parameters)")
    return {
        "arm": "table",
        "features": feats,
        "byte_ledger": led,
        "param_count": int(led["bytes"] // 4),
        "n_rows_fit": int(protos.n_rows_fit),
        "thin_values": {
            ctx["names"][k]: [int(c) for c in counts]
            for k, counts in enumerate(protos.n_per_value)
            if min(counts) < 8
        },
        "offgrid_capable": True,
        "offgrid_note": ("FactorPrototypes.prototype interpolates and extrapolates "
                         "off-grid values, so the table can enter row 2"),
        "wall_clock_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# 2. LACE - the historical selection, and the repaired one
# ---------------------------------------------------------------------------


def lace_selection_repaired(ctx: dict, config, seed: int,
                            log: Callable = print) -> dict:
    """Re-run the 36-trial LACE selection ON THE STRATIFIED SPLIT.

    The grid, the short schedule, the sampler and the criterion are
    ``exp_gen_baselines.run_lace_arm``'s, unchanged. The only thing that differs
    from the inherited leg is the rows the selection is made on: the stratified
    fit/val cut the cell actually scores, rather than the head/tail cut the
    historical manifest was selected under.
    """
    g = config.gen_baselines
    t0 = time.time()
    Xfit = ctx["X_fit"]
    cards = ctx["cardinalities"]
    key = jax.random.PRNGKey(int(seed) * 10_009 + 20_260_911)

    whitenings, wchecks = {}, {}
    for sh in list(g.lace_shrinkages):
        shv = sh if sh in ("none", "ledoit_wolf") else float(sh)
        w = lace.fit_whitening(Xfit, shv)
        whitenings[str(sh)] = w
        chk = lace.whitening_check(w, Xfit)
        chk["prior_identity_near_exact"] = bool(
            float(chk["mean_abs_diag_minus_one"]) <= float(g.lace_whitening_exact_tol))
        wchecks[str(sh)] = chk
        log(f"    [lace/repair] whitening shrinkage={sh}: "
            f"|diag-1|={chk['mean_abs_diag_minus_one']:.4g} "
            f"exact={chk['prior_identity_near_exact']}")

    tune_models = {}
    for sh in list(g.lace_shrinkages):
        key, tk = jax.random.split(key)
        m, info, key = lace.train_classifier(
            whitenings[str(sh)].forward(Xfit), ctx["lab_fit"], ctx["cont_fit"], cards,
            tk, n_epochs=int(g.lace_tune_epochs), batch=int(g.lace_batch),
            lr=float(g.lace_lr), warmup=int(g.lace_warmup_iters), log=None)
        tune_models[str(sh)] = m

    grid = lace.tuning_grid(g.lace_sigmas, g.lace_forms,
                            [str(s) for s in g.lace_shrinkages])
    n_tune = min(int(g.lace_tune_n), ctx["req_val"].n)
    req_tune = _slice_dense(ctx["req_val"], n_tune)
    lab_tune = ctx["lab_val"][:n_tune]
    log(f"    [lace/repair] tuning budget: {len(grid)} trials on the STRATIFIED split "
        f"(short schedule, {g.lace_tune_epochs} epochs, {n_tune} tune requests)")

    trials = []
    for t_i, trial in enumerate(grid):
        m = tune_models[str(trial["shrinkage"])]
        key, sk = jax.random.split(key)
        U = lace.sample_ld(
            m, req_tune, sk, std=lace.LACE_SGLD_STD_CORRECTED, lr=lace.LACE_SGLD_LR,
            steps=int(g.lace_tune_ld_steps), form=trial["form"],
            sigma=float(trial["sigma"]), batch=int(ctx["eval_batch"]))
        X = whitenings[str(trial["shrinkage"])].inverse(U)
        comp = score_composition(ctx["decoder"], ctx["probe"], X, lab_tune,
                                 ctx["names"], ctx["feat_shape"], ctx["eval_batch"])
        trials.append({**trial, "shrinkage": str(trial["shrinkage"]),
                       "val_aggregate": comp["aggregate"],
                       "val_compositional": comp["per_factor"][COMPOSITIONAL_FACTOR]})
        if (t_i + 1) % 9 == 0 or t_i == 0:
            log(f"    [lace/repair] trial {t_i + 1}/{len(grid)} "
                f"sigma={trial['sigma']} {trial['form']} shrink={trial['shrinkage']}: "
                f"val agg={comp['aggregate']:.4f}")

    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    best_comp = trials[int(np.argmax([t["val_compositional"] for t in trials]))]
    sel = {k: best[k] for k in ("sigma", "form", "shrinkage")}
    log(f"    [lace/repair] SELECTED on the stratified split: sigma={sel['sigma']} "
        f"form={sel['form']} shrinkage={sel['shrinkage']} "
        f"(val agg {best['val_aggregate']:.4f})")
    return {
        "inherited": False,
        "repaired": True,
        "selected": sel,
        "selection_criterion": ("validation aggregate probe accuracy on the "
                                "STRATIFIED split; C never enters"),
        "second_rule_selected": {k: best_comp[k] for k in
                                 ("sigma", "form", "shrinkage")},
        "second_rule_agrees": bool(all(best[k] == best_comp[k]
                                       for k in ("sigma", "form", "shrinkage"))),
        "tuning_trials": len(grid),
        "trials": trials,
        "whitening_checks": {k: {kk: float(vv) for kk, vv in v.items()
                                 if isinstance(vv, (int, float))}
                             for k, v in wchecks.items()},
        "selected_is_whitening_exact": bool(
            wchecks[str(sel["shrinkage"])]["prior_identity_near_exact"]),
        "repair_note": (
            "The pinned GenCellConfig.lace_selection_source was NOT edited. The "
            "historical leg is still run from it and both numbers are reported."),
        "wall_clock_s": float(time.time() - t0),
    }


def run_lace_both_legs(ctx: dict, config, seed: int, log: Callable = print) -> dict:
    """Run LACE twice: on the inherited selection, and on the repaired one.

    Both legs go through the SAME shipped ``exp_gen_cell.run_lace_arm``, so the
    only difference between them is the configuration that was selected.
    """
    out = {}
    try:
        hist = lace_selection(config, int(seed), log=log)
    except Exception as exc:  # the historical manifest may be absent for a seed
        hist = {"inherited": False, "NOT_RUN": f"{type(exc).__name__}: {exc}"}
    out["selection_historical"] = hist

    rep = lace_selection_repaired(ctx, config, seed, log=log)
    out["selection_repaired"] = rep

    legs = {}
    if hist.get("inherited"):
        log("  [lace] leg 1/2: the INHERITED (historical_688128) selection")
        st = run_lace_arm(ctx, config, seed, hist, log=log)
        legs["historical"] = {"selected": hist["selected"],
                              "features": st["samplers"],
                              "whitening": st["whitening"],
                              "classifier": st["classifier"],
                              "wall_clock_s": st["wall_clock_s"]}
    else:
        legs["historical"] = {"NOT_RUN": hist.get("NOT_RUN",
                                                  "no inherited selection available")}

    log("  [lace] leg 2/2: the REPAIRED (stratified-split) selection")
    st = run_lace_arm(ctx, config, seed, rep, log=log)
    legs["repaired"] = {"selected": rep["selected"],
                        "features": st["samplers"],
                        "whitening": st["whitening"],
                        "classifier": st["classifier"],
                        "wall_clock_s": st["wall_clock_s"]}
    out["legs"] = legs
    out["primary_sampler"] = str(config.gen_cell.lace_primary_sampler)
    return out


# ---------------------------------------------------------------------------
# 3. The product-of-experts EBM, at its OWN budget, run twice
# ---------------------------------------------------------------------------


def run_poe_ebm(ctx: dict, config, seed: int, repeats: int = 2,
                log: Callable = print) -> dict:
    """The energy-based composition arm, at its own budget, on independent keys.

    ``exp_gen_cell.run_ebm_arm`` is called as shipped. It is run ``repeats``
    times; each repeat re-trains its experts and re-samples on its own key
    stream, so the spread across repeats is a real sampler-plus-fit noise floor
    rather than a re-draw from one fit.

    The byte multiple - the arm's deployed bytes over the store's 524,288 B - is
    computed here and belongs beside every number this arm produces.
    """
    store_bytes = int(config.gen_cell.expect_deployed_store_bytes or 524_288)
    runs = []
    for r in range(int(repeats)):
        log(f"  [poe_ebm] run {r + 1}/{repeats} (independent expert fit and sampler keys)")
        t0 = time.time()
        # A distinct seed offset per repeat: run_ebm_arm keys off `seed`, so two
        # calls with the same seed would be the same draw, not a second one.
        res = run_ebm_arm(ctx, config, int(seed) + 1000 * r, log=log)
        nb = int(res["experts"]["deployed_bytes"])
        res["byte_ledger"] = {
            "deployed_bytes": nb,
            "param_count": int(res["experts"]["param_count"]),
            "store_reference_bytes": store_bytes,
            "byte_multiple": float(nb / store_bytes),
            "note": ("this arm runs at its OWN budget, not the store's; the multiple "
                     "belongs beside every number it produces"),
        }
        res["repeat_index"] = r
        res["wall_clock_s"] = float(time.time() - t0)
        log(f"  [poe_ebm] run {r + 1}: {res['experts']['param_count']} params = {nb} B "
            f"= {res['byte_ledger']['byte_multiple']:.1f}x the {store_bytes} B store")
        runs.append(res)
    return {
        "arm": "poe_ebm",
        "repeats": int(repeats),
        "runs": runs,
        "byte_multiple": runs[0]["byte_ledger"]["byte_multiple"],
        "byte_ledger": runs[0]["byte_ledger"],
    }
