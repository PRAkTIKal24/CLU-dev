"""specialists."""

from __future__ import annotations
import time
from typing import Callable
import jax
import numpy as np
from nclu.core.gen_compose import COMPOSITIONAL_FACTOR
from nclu.eval.baselines import gen_lace as lace
from nclu.experiments.exp_gen_baselines import _slice_dense, score_composition
from nclu.experiments.exp_gen_cell import lace_selection, run_ebm_arm, run_lace_arm

REQUEST_SETS = ("B", "MS")


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
    from nclu.core.gen_request import requests_from_rows

    req_dicts = {k: requests_from_rows(ctx["grid"], ctx["row_sets"][k]) for k in REQUEST_SETS}
    feats = {k: np.asarray(protos.embed_many(req_dicts[k]), np.float32) for k in REQUEST_SETS}
    led = protos.byte_footprint()
    log(
        f"  [table] {led['n_prototypes']} prototypes x D={led['dim']} = {led['bytes']} B ({led['bytes'] / 4:.0f} float32 parameters)"
    )
    return {
        "arm": "table",
        "features": feats,
        "byte_footprint": led,
        "param_count": int(led["bytes"] // 4),
        "n_rows_fit": int(protos.n_rows_fit),
        "thin_values": {
            ctx["names"][k]: [int(c) for c in counts]
            for k, counts in enumerate(protos.n_per_value)
            if min(counts) < 8
        },
        "offgrid_capable": True,
        "offgrid_note": "FactorPrototypes.prototype interpolates and extrapolates off-grid values, so the table can enter row 2",
        "wall_clock_s": float(time.time() - t0),
    }


def lace_selection_repaired(ctx: dict, config, seed: int, log: Callable = print) -> dict:
    g = config.gen_baselines
    t0 = time.time()
    Xfit = ctx["X_fit"]
    cards = ctx["cardinalities"]
    key = jax.random.PRNGKey(int(seed) * 10009 + 20260911)
    whitenings, wchecks = ({}, {})
    for sh in list(g.lace_shrinkages):
        shv = sh if sh in ("none", "ledoit_wolf") else float(sh)
        w = lace.fit_whitening(Xfit, shv)
        whitenings[str(sh)] = w
        chk = lace.whitening_check(w, Xfit)
        chk["prior_identity_near_exact"] = bool(
            float(chk["mean_abs_diag_minus_one"]) <= float(g.lace_whitening_exact_tol)
        )
        wchecks[str(sh)] = chk
        log(
            f"    [lace/repair] whitening shrinkage={sh}: |diag-1|={chk['mean_abs_diag_minus_one']:.4g} exact={chk['prior_identity_near_exact']}"
        )
    tune_models = {}
    for sh in list(g.lace_shrinkages):
        key, tk = jax.random.split(key)
        m, info, key = lace.train_classifier(
            whitenings[str(sh)].forward(Xfit),
            ctx["lab_fit"],
            ctx["cont_fit"],
            cards,
            tk,
            n_epochs=int(g.lace_tune_epochs),
            batch=int(g.lace_batch),
            lr=float(g.lace_lr),
            warmup=int(g.lace_warmup_iters),
            log=None,
        )
        tune_models[str(sh)] = m
    grid = lace.tuning_grid(g.lace_sigmas, g.lace_forms, [str(s) for s in g.lace_shrinkages])
    n_tune = min(int(g.lace_tune_n), ctx["req_val"].n)
    req_tune = _slice_dense(ctx["req_val"], n_tune)
    lab_tune = ctx["lab_val"][:n_tune]
    log(
        f"    [lace/repair] tuning budget: {len(grid)} trials on the STRATIFIED split (short schedule, {g.lace_tune_epochs} epochs, {n_tune} tune requests)"
    )
    trials = []
    for t_i, trial in enumerate(grid):
        m = tune_models[str(trial["shrinkage"])]
        key, sk = jax.random.split(key)
        U = lace.sample_ld(
            m,
            req_tune,
            sk,
            std=lace.LACE_SGLD_STD_CORRECTED,
            lr=lace.LACE_SGLD_LR,
            steps=int(g.lace_tune_ld_steps),
            form=trial["form"],
            sigma=float(trial["sigma"]),
            batch=int(ctx["eval_batch"]),
        )
        X = whitenings[str(trial["shrinkage"])].inverse(U)
        comp = score_composition(
            ctx["decoder"],
            ctx["probe"],
            X,
            lab_tune,
            ctx["names"],
            ctx["feat_shape"],
            ctx["eval_batch"],
        )
        trials.append(
            {
                **trial,
                "shrinkage": str(trial["shrinkage"]),
                "val_aggregate": comp["aggregate"],
                "val_compositional": comp["per_factor"][COMPOSITIONAL_FACTOR],
            }
        )
        if (t_i + 1) % 9 == 0 or t_i == 0:
            log(
                f"    [lace/repair] trial {t_i + 1}/{len(grid)} sigma={trial['sigma']} {trial['form']} shrink={trial['shrinkage']}: val agg={comp['aggregate']:.4f}"
            )
    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    best_comp = trials[int(np.argmax([t["val_compositional"] for t in trials]))]
    sel = {k: best[k] for k in ("sigma", "form", "shrinkage")}
    log(
        f"    [lace/repair] SELECTED on the stratified split: sigma={sel['sigma']} form={sel['form']} shrinkage={sel['shrinkage']} (val agg {best['val_aggregate']:.4f})"
    )
    return {
        "inherited": False,
        "repaired": True,
        "selected": sel,
        "selection_criterion": "validation aggregate probe accuracy on the STRATIFIED split; C never enters",
        "second_rule_selected": {k: best_comp[k] for k in ("sigma", "form", "shrinkage")},
        "second_rule_agrees": bool(
            all((best[k] == best_comp[k] for k in ("sigma", "form", "shrinkage")))
        ),
        "tuning_trials": len(grid),
        "trials": trials,
        "whitening_checks": {
            k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
            for k, v in wchecks.items()
        },
        "selected_is_whitening_exact": bool(
            wchecks[str(sel["shrinkage"])]["prior_identity_near_exact"]
        ),
        "repair_note": "The pinned GenCellConfig.lace_selection_source was NOT edited. The historical leg is still run from it and both numbers are reported.",
        "wall_clock_s": float(time.time() - t0),
    }


def run_lace_both_legs(ctx: dict, config, seed: int, log: Callable = print) -> dict:
    out = {}
    try:
        hist = lace_selection(config, int(seed), log=log)
    except Exception as exc:
        hist = {"inherited": False, "NOT_RUN": f"{type(exc).__name__}: {exc}"}
    out["selection_historical"] = hist
    rep = lace_selection_repaired(ctx, config, seed, log=log)
    out["selection_repaired"] = rep
    legs = {}
    if hist.get("inherited"):
        log("  [lace] leg 1/2: the INHERITED (historical_688128) selection")
        st = run_lace_arm(ctx, config, seed, hist, log=log)
        legs["historical"] = {
            "selected": hist["selected"],
            "features": st["samplers"],
            "whitening": st["whitening"],
            "classifier": st["classifier"],
            "wall_clock_s": st["wall_clock_s"],
        }
    else:
        legs["historical"] = {"NOT_RUN": hist.get("NOT_RUN", "no inherited selection available")}
    log("  [lace] leg 2/2: the REPAIRED (stratified-split) selection")
    st = run_lace_arm(ctx, config, seed, rep, log=log)
    legs["repaired"] = {
        "selected": rep["selected"],
        "features": st["samplers"],
        "whitening": st["whitening"],
        "classifier": st["classifier"],
        "wall_clock_s": st["wall_clock_s"],
    }
    out["legs"] = legs
    out["primary_sampler"] = str(config.gen_cell.lace_primary_sampler)
    return out


def run_poe_ebm(ctx: dict, config, seed: int, repeats: int = 2, log: Callable = print) -> dict:
    store_bytes = int(config.gen_cell.expect_deployed_store_bytes or 524288)
    runs = []
    for r in range(int(repeats)):
        log(f"  [poe_ebm] run {r + 1}/{repeats} (independent expert fit and sampler keys)")
        t0 = time.time()
        res = run_ebm_arm(ctx, config, int(seed) + 1000 * r, log=log)
        nb = int(res["experts"]["deployed_bytes"])
        res["byte_footprint"] = {
            "deployed_bytes": nb,
            "param_count": int(res["experts"]["param_count"]),
            "store_reference_bytes": store_bytes,
            "byte_multiple": float(nb / store_bytes),
            "note": "this arm runs at its OWN budget, not the store's; the multiple belongs beside every number it produces",
        }
        res["repeat_index"] = r
        res["wall_clock_s"] = float(time.time() - t0)
        log(
            f"  [poe_ebm] run {r + 1}: {res['experts']['param_count']} params = {nb} B = {res['byte_footprint']['byte_multiple']:.1f}x the {store_bytes} B store"
        )
        runs.append(res)
    return {
        "arm": "poe_ebm",
        "repeats": int(repeats),
        "runs": runs,
        "byte_multiple": runs[0]["byte_footprint"]["byte_multiple"],
        "byte_footprint": runs[0]["byte_footprint"],
    }
