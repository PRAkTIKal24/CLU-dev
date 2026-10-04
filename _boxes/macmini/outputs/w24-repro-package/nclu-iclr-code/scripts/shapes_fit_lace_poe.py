"""LACE and PoE at their own settings in the frozen feature space, on rows 3-4 or a ladder rung.

python scripts/shapes_fit_lace_poe.py --rung shipped --seed 0 --nrows 131072
python scripts/shapes_fit_lace_poe.py --rung 3 --seed 0 --nrows 32256
"""

from __future__ import annotations
from nclu.paths import run_dir
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nclu

_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in Path(nclu.__file__).resolve().parents:
    raise SystemExit("nclu resolves to %s, OUTSIDE %s -- refusing." % (nclu.__file__, _ROOT))
import jax
import jax.numpy as jnp
from nclu.config import get_default_config
from nclu.core.gen_request import dense_request_arrays
from nclu.data.gen_rowsplit import split_fit_val
from nclu.eval.baselines import gen_lace as lace
from nclu.experiments import gen_deletion as D7
from nclu.experiments import factor_wells as fw
from nclu.experiments import exp_gen_read as R
from nclu.experiments import composition_cell as CC
from nclu.experiments.exp_gen_cell import ebm_sample, train_expert
import shapes_ladder_lib as LEG
from shapes_fast_judge import FastJudge, assert_identical

T0 = time.time()
LADDER_RUNS = Path(str(run_dir("shapes3d/ladder_earlier")))
N_ROWS = 32768
N_FIT, N_VAL = (32256, 512)
STORE_REF = 524288


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


def n_params(tree) -> int:
    return int(
        sum((np.asarray(x).size for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "shape")))
    )


def val_aggregate(fast, X, lab) -> float:
    tup = fast.read_back(np.asarray(X, np.float32), int(fw.PINS["eval_batch"]))
    return float(np.mean([(tup[:, k] == lab[:, k]).mean() for k in range(lab.shape[1])]))


def dense_from_labels(grid, lab):
    reqs = [
        {
            "request": [
                {"factor_index": k, "value": float(np.asarray(grid.values[k])[int(lab[i, k])])}
                for k in range(lab.shape[1])
            ]
        }
        for i in range(lab.shape[0])
    ]
    return dense_request_arrays(grid, reqs)


def values_to_labels(grid, V):
    """(N, F) requested VALUES -> nearest class per factor + the off-grid mask."""
    V = np.asarray(V, np.float64)
    lab = np.zeros(V.shape, np.int64)
    oor = np.zeros(V.shape[0], bool)
    for k in range(V.shape[1]):
        vals = np.asarray(grid.values[k], np.float64)
        d = np.abs(V[:, k][:, None] - vals[None, :])
        lab[:, k] = np.argmin(d, axis=1)
        oor |= d.min(axis=1) > 1e-06 * max(1.0, float(np.abs(vals).max()))
    return (lab, oor)


def fit_lace(ctx, seed: int, fast) -> dict:
    g = get_default_config().gen_baselines
    c = get_default_config().gen_cell
    t0 = time.time()
    key = jax.random.PRNGKey(int(seed) * 10009 + 20260923)
    whit, wchk = ({}, {})
    for sh in list(g.lace_shrinkages):
        shv = sh if sh in ("none", "ledoit_wolf") else float(sh)
        whit[str(sh)] = lace.fit_whitening(ctx["X_fit"], shv)
        wchk[str(sh)] = {
            k: float(v)
            for k, v in lace.whitening_check(whit[str(sh)], ctx["X_fit"]).items()
            if isinstance(v, (int, float))
        }
    tune_models = {}
    for sh in list(g.lace_shrinkages):
        key, tk = jax.random.split(key)
        m, _info, key = lace.train_classifier(
            whit[str(sh)].forward(ctx["X_fit"]),
            ctx["lab_fit"],
            ctx["cont_fit"],
            ctx["cards"],
            tk,
            n_epochs=int(g.lace_tune_epochs),
            batch=int(g.lace_batch),
            lr=float(g.lace_lr),
            warmup=int(g.lace_warmup_iters),
            log=None,
        )
        tune_models[str(sh)] = m
    grid_t = lace.tuning_grid(g.lace_sigmas, g.lace_forms, [str(s) for s in g.lace_shrinkages])
    n_tune = min(int(g.lace_tune_n), ctx["lab_val"].shape[0])
    req_t = dense_from_labels(ctx["grid"], ctx["lab_val"][:n_tune])
    trials = []
    for tr in grid_t:
        key, sk = jax.random.split(key)
        U = lace.sample_ld(
            tune_models[str(tr["shrinkage"])],
            req_t,
            sk,
            std=lace.LACE_SGLD_STD_CORRECTED,
            lr=lace.LACE_SGLD_LR,
            steps=int(g.lace_tune_ld_steps),
            form=tr["form"],
            sigma=float(tr["sigma"]),
            batch=256,
        )
        X = whit[str(tr["shrinkage"])].inverse(U)
        trials.append(
            {
                "sigma": float(tr["sigma"]),
                "form": tr["form"],
                "shrinkage": str(tr["shrinkage"]),
                "val_aggregate": val_aggregate(fast, X, ctx["lab_val"][:n_tune]),
            }
        )
    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    log("  [lace] 36-trial selection done in %.0fs: %s" % (time.time() - t0, best))
    sh = best["shrinkage"]
    wh = whit[sh]
    key, fk = jax.random.split(key)
    model, fit_info, key = lace.train_classifier(
        wh.forward(ctx["X_fit"]),
        ctx["lab_fit"],
        ctx["cont_fit"],
        ctx["cards"],
        fk,
        n_epochs=int(c.lace_n_epochs),
        batch=int(c.lace_batch),
        lr=float(c.lace_lr),
        warmup=int(c.lace_warmup_iters),
        log=None,
    )
    own = lace.classifier_accuracy(model, wh.forward(ctx["X_val"]), ctx["lab_val"])
    D = int(ctx["X_fit"].shape[1])
    p_clf = n_params(model)
    budget = {
        "classifier_params": p_clf,
        "whitening_floats": int(D * D + D),
        "deployed_bytes_classifier_only": int(4 * p_clf),
        "deployed_bytes_classifier_plus_whitening": int(4 * (p_clf + D * D + D)),
        "x_store_reference_classifier_only": round(4 * p_clf / STORE_REF, 3),
        "x_store_reference_with_whitening": round(4 * (p_clf + D * D + D) / STORE_REF, 3),
        "fit_rows": int(ctx["X_fit"].shape[0]),
        "val_rows": int(ctx["X_val"].shape[0]),
        "epochs": int(c.lace_n_epochs),
        "steps": int(fit_info["steps"]),
        "tuning_trials": len(trials),
        "tune_epochs": int(g.lace_tune_epochs),
        "tune_ld_steps": int(g.lace_tune_ld_steps),
        "tune_n": int(n_tune),
        "sampler": "ode (dopri5, the PRIMARY sampler)",
        "convention": "float32 parameters x 4 B; the classifier alone and classifier + whitening are BOTH printed; ⛔ never matched to the store",
    }
    log(
        "  [lace] classifier %d params, own val mean %.4f; budget %s" % (p_clf, own["mean"], budget)
    )

    def rival(V):
        labs, oor = values_to_labels(ctx["grid"], V)
        req = dense_from_labels(ctx["grid"], labs)
        nonlocal key
        key, rk = jax.random.split(key)
        U = lace.sample_ode(
            model, req, rk, form=best["form"], sigma=float(best["sigma"]), batch=256
        )
        return (np.asarray(wh.inverse(U), np.float32), oor)

    return {
        "rival": rival,
        "budget": budget,
        "selected": best,
        "trials": trials,
        "whitening_checks": wchk,
        "classifier": {k: v for k, v in fit_info.items() if isinstance(v, (int, float, str))},
        "own_val_accuracy": own,
        "seconds_fit": round(time.time() - t0, 1),
    }


def fit_poe(ctx, seed: int, fast) -> dict:
    cfg = get_default_config().gen_cell
    t0 = time.time()
    mu = np.mean(ctx["X_fit"], axis=0)
    sd = np.maximum(np.std(ctx["X_fit"], axis=0), 0.001)
    std_fit = ((ctx["X_fit"] - mu) / sd).astype(np.float32)
    std_val = ((ctx["X_val"] - mu) / sd).astype(np.float32)
    key = jax.random.PRNGKey(int(seed) * 10010 + 20260923)
    experts, fits = ([], [])
    for j, cc in enumerate(ctx["cards"]):
        m, info, key = train_expert(std_fit, ctx["lab_fit"][:, j], int(cc), cfg, key, log)
        experts.append(m)
        fits.append({"factor": ctx["names"][j], **info})
    own = [
        float(
            (
                np.argmax(np.asarray(jax.vmap(ex)(jnp.asarray(std_val))), axis=1)
                == ctx["lab_val"][:, j]
            ).mean()
        )
        for j, ex in enumerate(experts)
    ]
    n_tune = int(min(int(cfg.ebm_tune_n), ctx["lab_val"].shape[0]))
    tune_T = ctx["lab_val"][:n_tune]
    grid_t = [
        {"step_size": float(lr), "prior_sigma": float(ps), "n_steps": int(ns), "clip": bool(cl)}
        for lr in list(cfg.ebm_step_sizes)
        for ps in list(cfg.ebm_prior_sigmas)
        for ns in list(cfg.ebm_steps_grid)
        for cl in list(cfg.ebm_clip_grid)
    ]
    trials = []
    for tr in grid_t:
        key, sk = jax.random.split(key)
        U = ebm_sample(experts, tune_T, sk, tr["step_size"], tr["prior_sigma"], tr["n_steps"], 256)
        X = U * sd + mu
        if tr["clip"]:
            X = np.maximum(X, 0.0)
        trials.append({**tr, "val_aggregate": val_aggregate(fast, X, tune_T)})
    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    p = n_params(experts)
    D = int(ctx["X_fit"].shape[1])
    budget = {
        "expert_params": p,
        "standardiser_floats": int(2 * D),
        "deployed_bytes": int(4 * p),
        "x_store_reference": round(4 * p / STORE_REF, 3),
        "fit_rows": int(ctx["X_fit"].shape[0]),
        "val_rows": int(ctx["X_val"].shape[0]),
        "n_experts": len(experts),
        "hidden": int(cfg.ebm_hidden),
        "depth": int(cfg.ebm_depth),
        "epochs": int(cfg.ebm_epochs),
        "tuning_trials": len(trials),
        "tune_n": n_tune,
        "sampler": "SGLD from noise, Langevin-consistent sqrt(2 lr) noise, prior explicit",
        "convention": "float32 parameters x 4 B; ⛔ never matched to the store",
    }
    log(
        "  [poe] %d experts, %d params, own val %s; selected %s (%.0fs)"
        % (len(experts), p, [round(a, 3) for a in own], best, time.time() - t0)
    )

    def rival(V):
        labs, oor = values_to_labels(ctx["grid"], V)
        nonlocal key
        key, sk = jax.random.split(key)
        U = ebm_sample(
            experts, labs, sk, best["step_size"], best["prior_sigma"], best["n_steps"], 256
        )
        X = U * sd + mu
        if best["clip"]:
            X = np.maximum(X, 0.0)
        return (np.asarray(X, np.float32), oor)

    return {
        "rival": rival,
        "budget": budget,
        "selected": best,
        "trials": trials,
        "experts_fit": fits,
        "own_val_accuracy_per_factor": own,
        "seconds_fit": round(time.time() - t0, 1),
    }


def summarise(arm, per, extra):
    dv = np.array([r["distinct_valid_novel_tuples"] for r in per], float)
    ja = np.array([r["judge_accuracy"] for r in per], float)
    s = {
        "arm": arm,
        "n_corners": len(per),
        "distinct_valid_novel_tuples_mean": float(dv.mean()),
        "distinct_valid_novel_tuples_sd": float(dv.std(ddof=1)) if dv.size > 1 else None,
        "judge_accuracy_mean": float(ja.mean()),
    }
    s.update(extra)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, help="2|3|4|5|6|shipped")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=str(run_dir("shapes3d/lacepoe")))
    ap.add_argument("--arms", default="lace,poe")
    ap.add_argument("--nrows", type=int, default=131072)
    ap.add_argument("--max-corners", type=int, default=0, help="DEV ONLY: truncate corners")
    a = ap.parse_args(argv)
    leg = Path(a.out) / ("%s_seed%d" % (a.rung, a.seed))
    leg.mkdir(parents=True, exist_ok=True)
    global N_ROWS, N_FIT
    N_ROWS, N_FIT = (int(a.nrows), int(a.nrows) - N_VAL)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from shapes_common import load_pool

    fw.PINS["condition"] = "val" if a.rung == "val" else "test"
    fw.PINS["seed"] = int(a.seed)
    cache = D7.assert_generation2(int(a.seed))
    log("nclu.__file__ = %s" % nclu.__file__)
    log(
        "rung=%s seed=%d nrows=%d cache gen %s scope %s"
        % (a.rung, a.seed, N_ROWS, cache["generation"], cache["scope"])
    )
    if a.rung == "val":
        cond = fw.load_condition()
        cond["corners"] = fw.val_corners(cond)
        grid = cond["grid"]
        pool, _dead, _pf = load_pool("val")
        fitted = np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond))
        if np.isin(pool, fw.val_corner_rows(cond)).any():
            raise SystemExit("val pool intersects the val corners")
        bitmap = fw.value_presence_bitmap(grid, fitted)
        seen_mask = fw.seen_tuple_mask(grid, fitted)
        chk = {"val": True, "n_corners": len(cond["corners"])}
    else:
        src = LADDER_RUNS / ("%s_seed%d" % (a.rung, a.seed))
        for name in ("partitions.npz", "partitions.json"):
            if not (leg / name).exists():
                shutil.copyfile(src / name, leg / name)
        cond = LEG.make_condition(a.rung)
        LEG._COND["cond"] = cond
        fw.load_condition = LEG._patched_load_condition
        CC.fw.load_condition = LEG._patched_load_condition
        R.fw.load_condition = LEG._patched_load_condition
        grid = cond["grid"]
        chk = fw.assert_corners_disjoint_and_heldout(cond["corners"], cond["test_rows"], a.rung)
        z = dict(np.load(leg / "partitions.npz", allow_pickle=True))
        pool = np.asarray(z["pool"], np.int64)
        if np.isin(pool, cond["test_rows"]).any():
            raise SystemExit("⛔ the fit pool intersects the held-out rows")
        bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
        seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    if a.max_corners:
        cond["corners"] = cond["corners"][: int(a.max_corners)]
        log("⚠ DEV: corners truncated to %d -- NOT A TABLE NUMBER" % a.max_corners)
    log("corners: %s" % chk)
    rows = pool[:N_ROWS]
    fit_rows, val_rows, split_info = split_fit_val(
        rows, grid.factor_indices_of(rows), N_FIT, N_VAL, int(a.seed), mode="stratified"
    )
    F = fw.features_mmap(int(a.seed))
    X_fit = np.asarray(fw.gather(F, fit_rows), np.float32).reshape((fit_rows.size, -1))
    X_val = np.asarray(fw.gather(F, val_rows), np.float32).reshape((val_rows.size, -1))
    lab_fit = grid.factor_indices_of(fit_rows)
    lab_val = grid.factor_indices_of(val_rows)
    cont = []
    for k in range(grid.n_factors):
        vk = np.asarray(grid.values[k], np.float64)
        span = max(float(vk[-1] - vk[0]), 1e-12)
        cont.append((vk[lab_fit[:, k]] - vk[0]) / span)
    ctx = {
        "grid": grid,
        "names": list(grid.names),
        "cards": [int(c) for c in grid.cardinalities],
        "X_fit": X_fit,
        "X_val": X_val,
        "lab_fit": lab_fit,
        "lab_val": lab_val,
        "cont_fit": np.stack(cont, axis=1).astype(np.float32),
    }
    log(
        "fit %d / val %d rows (stratified, from the ladder pool's first %d)"
        % (fit_rows.size, val_rows.size, N_ROWS)
    )
    judge = fw.Judge(int(a.seed), log=log)
    fast = FastJudge(judge)
    _chk = assert_identical(judge, fast, fw.gather(F, pool[:256]))
    if not _chk["identical"]:
        raise SystemExit("FastJudge is NOT identical to the shipped Judge -- refusing.")
    meta = {
        "rung": a.rung,
        "seed": int(a.seed),
        "package_file": str(nclu.__file__),
        "cache": cache,
        "corner_check": chk,
        "n_corners": len(cond["corners"]),
        "DEV_truncated": bool(a.max_corners),
        "fastjudge_positive_control": _chk,
        "fit_split": {
            k: v for k, v in split_info.items() if isinstance(v, (int, float, str, bool))
        },
        "estimand": {
            "metric": "distinct_valid_novel_tuples",
            "unit": "corner",
            "function": "exp_gen_factor_wells.per_corner_scores",
        },
        "arms": {},
        "failed_arms": [],
    }
    for arm in [x for x in a.arms.split(",") if x]:
        done = leg / ("rows_%s.json" % arm)
        if done.exists():
            log("  %s: already on disk, skipping" % arm)
            meta["arms"][arm] = json.load(open(done, encoding="utf-8"))["summary"]
            continue
        t = time.time()
        try:
            fitted = (fit_lace if arm == "lace" else fit_poe)(ctx, int(a.seed), fast)
            obj = {
                "arm": arm,
                "read": "rival",
                "kind": "rival",
                "rival": fitted["rival"],
                "rival_partial": None,
                "store": None,
                "protos": None,
                "byte_footprint": {
                    "deployed_bytes": fitted["budget"].get(
                        "deployed_bytes",
                        fitted["budget"].get("deployed_bytes_classifier_plus_whitening"),
                    )
                },
                "head_bytes": {"head_bytes": 0},
            }
            ts = time.time()
            per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
            extra = {k: v for k, v in fitted.items() if k != "rival"}
            extra.update(
                {
                    "seconds_score": round(time.time() - ts, 1),
                    "seconds_total": round(time.time() - t, 1),
                    "OWN_BUDGET": True,
                    "refusals": R._REFUSALS.get(arm, {}),
                }
            )
            s = summarise(arm, per, extra)
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": s, "per_corner": per}, fh, indent=1, default=str)
            meta["arms"][arm] = {k: v for k, v in s.items() if k != "trials"}
            log(
                "  %-6s novel %.4f  judge_acc %.4f  in %.0fs (score %.0fs)"
                % (
                    arm,
                    s["distinct_valid_novel_tuples_mean"],
                    s["judge_accuracy_mean"],
                    time.time() - t,
                    extra["seconds_score"],
                )
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            meta["failed_arms"].append(arm)
            with open(leg / ("FAILED_%s.json" % arm), "w", encoding="utf-8") as fh:
                json.dump({"arm": arm, "error": "%s: %s" % (type(exc).__name__, exc)}, fh, indent=1)
            log("  !!!! %s FAILED: %s: %s" % (arm, type(exc).__name__, exc))
    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s failed=%s" % (leg, meta["failed_arms"]))
    return 3 if meta["failed_arms"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
