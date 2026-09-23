"""w15 spoke/lace-poe-rows -- LACE and PoE as ADDED ROWS on Shapes rows 3-4 and every ladder rung.

⛔ LACE and PoE NEVER replace the learned-rival columns (MLP / LSTM / transformer /
continuous MLP).  Those columns are read, unchanged, from the ladder's own legs and are
printed beside these rows by ``scripts/w15_lace_poe_table.py``.

WHAT IS HELD FIXED -- the ladder's instrument, byte for byte
------------------------------------------------------------
* the rungs, the corners and the condition swap: ``w12b_ladder_leg.make_condition``
  (``shipped`` = rows 3 + 4: the 60 pair sub-corners and the 49 n-way corners);
* the fit pool: the ladder leg's OWN ``partitions.npz``
  (``outputs/w12-composition-ladder/runs/<rung>_seed<s>/``, READ ONLY), so the rivals
  see the identical pool the store arms and ``learned_1x`` saw -- disjoint from every
  corner by the ladder's own assertion;
* the estimand: ``distinct_valid_novel_tuples`` per corner via ``fw.per_corner_scores``
  through ``w12b_ladder_leg.score_arm``, unchanged;
* the judge: ``fw.Judge`` + ``FastJudge``, asserted identical on real features first.

THEIR OWN BUDGETS -- DECLARED, PRINTED IN EVERY ROW
---------------------------------------------------
Each rival runs at the budget its shipped implementation carries (``GenBaselinesConfig`` /
``GenCellConfig``), ⛔ not at the store's bytes:

* data: the first 32,768 rows of the ladder pool (= ``learned_1x``'s ``N_FIT_ROWS``),
  stratified into 32,256 fit + 512 val by the harness's own ``split_fit_val`` -- the
  generation table's own ``n_fit`` / ``n_val``;
* LACE: the 36-trial selection (sigma x form x shrinkage, 6-epoch tune classifiers, 30-step
  LD on 256 val requests) scored by val aggregate accuracy, then the 50-epoch classifier
  on the selected whitening, sampled by the PRIMARY ``ode`` sampler; parameters =
  classifier + whitening (mean + D x D transform), both printed;
* PoE (Du, Li & Mordatch 2020; ``exp_gen_cell.run_ebm_arm``'s method): six independent
  experts (hidden 256, depth 2, 20 epochs), the 36-trial SGLD grid scored by val aggregate
  accuracy, one run per (rung, seed) -- the three seeds carry the spread.

⭐ The selection criterion reads decoded val renders through THE SAME judge the corners
are scored by.  That is the never-weaken-a-competitor direction (the rival is tuned on the
scorer's own view of in-distribution rows) and it is declared, not hidden.  ⛔ No corner
row, and no held-out value, enters any selection.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chlu  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in Path(chlu.__file__).resolve().parents:
    raise SystemExit("chlu resolves to %s, OUTSIDE %s -- refusing." % (chlu.__file__, _ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from chlu.config import get_default_config  # noqa: E402
from chlu.core.gen_request import dense_request_arrays  # noqa: E402
from chlu.data.gen_rowsplit import split_fit_val  # noqa: E402
from chlu.eval.rivals import gen_lace as lace  # noqa: E402
from chlu.experiments import exp_gen_delete_row7 as D7  # noqa: E402
from chlu.experiments import exp_gen_factor_wells as fw  # noqa: E402
from chlu.experiments import exp_gen_read as R  # noqa: E402
from chlu.experiments import exp_w5_compose_field as W5  # noqa: E402
from chlu.experiments.exp_gen_cell import ebm_sample, train_expert  # noqa: E402

import w12b_ladder_leg as LEG  # noqa: E402
from w12_fastjudge import FastJudge, assert_identical  # noqa: E402

T0 = time.time()
LADDER_RUNS = Path("C:/chlu/outputs/w12-composition-ladder/runs")   # READ ONLY
N_ROWS = 32_768
N_FIT, N_VAL = 32_256, 512
STORE_REF = 524_288


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


def n_params(tree) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)
                   if hasattr(x, "shape")))


def val_aggregate(fast, X, lab) -> float:
    tup = fast.read_back(np.asarray(X, np.float32), int(fw.PINS["eval_batch"]))
    return float(np.mean([(tup[:, k] == lab[:, k]).mean() for k in range(lab.shape[1])]))


def dense_from_labels(grid, lab):
    reqs = [{"request": [{"factor_index": k,
                          "value": float(np.asarray(grid.values[k])[int(lab[i, k])])}
                         for k in range(lab.shape[1])]} for i in range(lab.shape[0])]
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
        oor |= d.min(axis=1) > 1e-6 * max(1.0, float(np.abs(vals).max()))
    return lab, oor


# ---------------------------------------------------------------------------
# LACE
# ---------------------------------------------------------------------------

def fit_lace(ctx, seed: int, fast) -> dict:
    g = get_default_config().gen_baselines
    c = get_default_config().gen_cell
    t0 = time.time()
    key = jax.random.PRNGKey(int(seed) * 10_009 + 20_260_923)
    whit, wchk = {}, {}
    for sh in list(g.lace_shrinkages):
        shv = sh if sh in ("none", "ledoit_wolf") else float(sh)
        whit[str(sh)] = lace.fit_whitening(ctx["X_fit"], shv)
        wchk[str(sh)] = {k: float(v) for k, v in
                         lace.whitening_check(whit[str(sh)], ctx["X_fit"]).items()
                         if isinstance(v, (int, float))}
    tune_models = {}
    for sh in list(g.lace_shrinkages):
        key, tk = jax.random.split(key)
        m, _info, key = lace.train_classifier(
            whit[str(sh)].forward(ctx["X_fit"]), ctx["lab_fit"], ctx["cont_fit"],
            ctx["cards"], tk, n_epochs=int(g.lace_tune_epochs), batch=int(g.lace_batch),
            lr=float(g.lace_lr), warmup=int(g.lace_warmup_iters), log=None)
        tune_models[str(sh)] = m
    grid_t = lace.tuning_grid(g.lace_sigmas, g.lace_forms, [str(s) for s in g.lace_shrinkages])
    n_tune = min(int(g.lace_tune_n), ctx["lab_val"].shape[0])
    req_t = dense_from_labels(ctx["grid"], ctx["lab_val"][:n_tune])
    trials = []
    for tr in grid_t:
        key, sk = jax.random.split(key)
        U = lace.sample_ld(tune_models[str(tr["shrinkage"])], req_t, sk,
                           std=lace.LACE_SGLD_STD_CORRECTED, lr=lace.LACE_SGLD_LR,
                           steps=int(g.lace_tune_ld_steps), form=tr["form"],
                           sigma=float(tr["sigma"]), batch=256)
        X = whit[str(tr["shrinkage"])].inverse(U)
        trials.append({"sigma": float(tr["sigma"]), "form": tr["form"],
                       "shrinkage": str(tr["shrinkage"]),
                       "val_aggregate": val_aggregate(fast, X, ctx["lab_val"][:n_tune])})
    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    log("  [lace] 36-trial selection done in %.0fs: %s" % (time.time() - t0, best))
    sh = best["shrinkage"]
    wh = whit[sh]
    key, fk = jax.random.split(key)
    model, fit_info, key = lace.train_classifier(
        wh.forward(ctx["X_fit"]), ctx["lab_fit"], ctx["cont_fit"], ctx["cards"], fk,
        n_epochs=int(c.lace_n_epochs), batch=int(c.lace_batch), lr=float(c.lace_lr),
        warmup=int(c.lace_warmup_iters), log=None)
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
        "fit_rows": int(ctx["X_fit"].shape[0]), "val_rows": int(ctx["X_val"].shape[0]),
        "epochs": int(c.lace_n_epochs), "steps": int(fit_info["steps"]),
        "tuning_trials": len(trials), "tune_epochs": int(g.lace_tune_epochs),
        "tune_ld_steps": int(g.lace_tune_ld_steps), "tune_n": int(n_tune),
        "sampler": "ode (dopri5, the PRIMARY sampler)",
        "convention": ("float32 parameters x 4 B; the classifier alone and classifier + "
                       "whitening are BOTH printed; ⛔ never matched to the store"),
    }
    log("  [lace] classifier %d params, own val mean %.4f; budget %s"
        % (p_clf, own["mean"], budget))

    def rival(V):
        labs, oor = values_to_labels(ctx["grid"], V)
        req = dense_from_labels(ctx["grid"], labs)
        nonlocal key
        key, rk = jax.random.split(key)
        U = lace.sample_ode(model, req, rk, form=best["form"], sigma=float(best["sigma"]),
                            batch=256)
        return np.asarray(wh.inverse(U), np.float32), oor

    return {"rival": rival, "budget": budget, "selected": best, "trials": trials,
            "whitening_checks": wchk,
            "classifier": {k: v for k, v in fit_info.items()
                           if isinstance(v, (int, float, str))},
            "own_val_accuracy": own, "seconds_fit": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# PoE
# ---------------------------------------------------------------------------

def fit_poe(ctx, seed: int, fast) -> dict:
    cfg = get_default_config().gen_cell
    t0 = time.time()
    mu = np.mean(ctx["X_fit"], axis=0)
    sd = np.maximum(np.std(ctx["X_fit"], axis=0), 1e-3)
    std_fit = ((ctx["X_fit"] - mu) / sd).astype(np.float32)
    std_val = ((ctx["X_val"] - mu) / sd).astype(np.float32)
    key = jax.random.PRNGKey(int(seed) * 10_010 + 20_260_923)
    experts, fits = [], []
    for j, cc in enumerate(ctx["cards"]):
        m, info, key = train_expert(std_fit, ctx["lab_fit"][:, j], int(cc), cfg, key, log)
        experts.append(m)
        fits.append({"factor": ctx["names"][j], **info})
    own = [float((np.argmax(np.asarray(jax.vmap(ex)(jnp.asarray(std_val))), axis=1)
                  == ctx["lab_val"][:, j]).mean()) for j, ex in enumerate(experts)]
    n_tune = int(min(int(cfg.ebm_tune_n), ctx["lab_val"].shape[0]))
    tune_T = ctx["lab_val"][:n_tune]
    grid_t = [{"step_size": float(lr), "prior_sigma": float(ps), "n_steps": int(ns),
               "clip": bool(cl)}
              for lr in list(cfg.ebm_step_sizes) for ps in list(cfg.ebm_prior_sigmas)
              for ns in list(cfg.ebm_steps_grid) for cl in list(cfg.ebm_clip_grid)]
    trials = []
    for tr in grid_t:
        key, sk = jax.random.split(key)
        U = ebm_sample(experts, tune_T, sk, tr["step_size"], tr["prior_sigma"],
                       tr["n_steps"], 256)
        X = U * sd + mu
        if tr["clip"]:
            X = np.maximum(X, 0.0)
        trials.append({**tr, "val_aggregate": val_aggregate(fast, X, tune_T)})
    best = trials[int(np.argmax([t["val_aggregate"] for t in trials]))]
    p = n_params(experts)
    D = int(ctx["X_fit"].shape[1])
    budget = {
        "expert_params": p, "standardiser_floats": int(2 * D),
        "deployed_bytes": int(4 * p), "x_store_reference": round(4 * p / STORE_REF, 3),
        "fit_rows": int(ctx["X_fit"].shape[0]), "val_rows": int(ctx["X_val"].shape[0]),
        "n_experts": len(experts), "hidden": int(cfg.ebm_hidden),
        "depth": int(cfg.ebm_depth), "epochs": int(cfg.ebm_epochs),
        "tuning_trials": len(trials), "tune_n": n_tune,
        "sampler": "SGLD from noise, Langevin-consistent sqrt(2 lr) noise, prior explicit",
        "convention": "float32 parameters x 4 B; ⛔ never matched to the store",
    }
    log("  [poe] %d experts, %d params, own val %s; selected %s (%.0fs)"
        % (len(experts), p, [round(a, 3) for a in own], best, time.time() - t0))

    def rival(V):
        labs, oor = values_to_labels(ctx["grid"], V)
        nonlocal key
        key, sk = jax.random.split(key)
        U = ebm_sample(experts, labs, sk, best["step_size"], best["prior_sigma"],
                       best["n_steps"], 256)
        X = U * sd + mu
        if best["clip"]:
            X = np.maximum(X, 0.0)
        return np.asarray(X, np.float32), oor

    return {"rival": rival, "budget": budget, "selected": best, "trials": trials,
            "experts_fit": fits, "own_val_accuracy_per_factor": own,
            "seconds_fit": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------

def summarise(arm, per, extra):
    dv = np.array([r["distinct_valid_novel_tuples"] for r in per], float)
    ja = np.array([r["judge_accuracy"] for r in per], float)
    s = {"arm": arm, "n_corners": len(per),
         "distinct_valid_novel_tuples_mean": float(dv.mean()),
         "distinct_valid_novel_tuples_sd": float(dv.std(ddof=1)) if dv.size > 1 else None,
         "judge_accuracy_mean": float(ja.mean())}
    s.update(extra)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, help="2|3|4|5|6|shipped")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="lace,poe")
    ap.add_argument("--max-corners", type=int, default=0, help="DEV ONLY: truncate corners")
    a = ap.parse_args(argv)

    leg = Path(a.out) / ("%s_seed%d" % (a.rung, a.seed))
    leg.mkdir(parents=True, exist_ok=True)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = int(a.seed)
    cache = D7.assert_generation2(int(a.seed))
    log("chlu.__file__ = %s" % chlu.__file__)
    log("rung=%s seed=%d cache gen %s scope %s" % (a.rung, a.seed, cache["generation"],
                                                  cache["scope"]))
    src = LADDER_RUNS / ("%s_seed%d" % (a.rung, a.seed))
    for name in ("partitions.npz", "partitions.json"):
        if not (leg / name).exists():
            shutil.copyfile(src / name, leg / name)

    cond = LEG.make_condition(a.rung)
    LEG._COND["cond"] = cond
    fw.load_condition = LEG._patched_load_condition
    W5.fw.load_condition = LEG._patched_load_condition
    R.fw.load_condition = LEG._patched_load_condition
    grid = cond["grid"]
    if a.max_corners:
        cond["corners"] = cond["corners"][: int(a.max_corners)]
        log("⚠ DEV: corners truncated to %d -- NOT A TABLE NUMBER" % a.max_corners)
    chk = fw.assert_corners_disjoint_and_heldout(cond["corners"], cond["test_rows"], a.rung)
    log("corners: %s" % chk)
    z = dict(np.load(leg / "partitions.npz", allow_pickle=True))
    pool = np.asarray(z["pool"], np.int64)
    if np.isin(pool, cond["test_rows"]).any():
        raise SystemExit("⛔ the fit pool intersects the held-out rows")
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])

    rows = pool[:N_ROWS]
    fit_rows, val_rows, split_info = split_fit_val(
        rows, grid.factor_indices_of(rows), N_FIT, N_VAL, int(a.seed), mode="stratified")
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
    ctx = {"grid": grid, "names": list(grid.names),
           "cards": [int(c) for c in grid.cardinalities],
           "X_fit": X_fit, "X_val": X_val, "lab_fit": lab_fit, "lab_val": lab_val,
           "cont_fit": np.stack(cont, axis=1).astype(np.float32)}
    log("fit %d / val %d rows (stratified, from the ladder pool's first %d)"
        % (fit_rows.size, val_rows.size, N_ROWS))

    judge = fw.Judge(int(a.seed), log=log)
    fast = FastJudge(judge)
    _chk = assert_identical(judge, fast, fw.gather(F, pool[:256]))
    if not _chk["identical"]:
        raise SystemExit("FastJudge is NOT identical to the shipped Judge -- refusing.")

    meta = {"rung": a.rung, "seed": int(a.seed), "chlu_file": str(chlu.__file__),
            "cache": cache, "corner_check": chk, "n_corners": len(cond["corners"]),
            "DEV_truncated": bool(a.max_corners), "fastjudge_positive_control": _chk,
            "fit_split": {k: v for k, v in split_info.items()
                          if isinstance(v, (int, float, str, bool))},
            "estimand": {"metric": "distinct_valid_novel_tuples", "unit": "corner",
                         "function": "exp_gen_factor_wells.per_corner_scores"},
            "arms": {}, "failed_arms": []}
    for arm in [x for x in a.arms.split(",") if x]:
        done = leg / ("rows_%s.json" % arm)
        if done.exists():
            log("  %s: already on disk, skipping" % arm)
            meta["arms"][arm] = json.load(open(done, encoding="utf-8"))["summary"]
            continue
        t = time.time()
        try:
            fitted = (fit_lace if arm == "lace" else fit_poe)(ctx, int(a.seed), fast)
            obj = {"arm": arm, "read": "rival", "kind": "rival", "rival": fitted["rival"],
                   "rival_partial": None, "store": None, "protos": None,
                   "byte_ledger": {"deployed_bytes": fitted["budget"].get(
                       "deployed_bytes", fitted["budget"].get(
                           "deployed_bytes_classifier_plus_whitening"))},
                   "head_bytes": {"head_bytes": 0}}
            ts = time.time()
            per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
            extra = {k: v for k, v in fitted.items() if k != "rival"}
            extra.update({"seconds_score": round(time.time() - ts, 1),
                          "seconds_total": round(time.time() - t, 1),
                          "OWN_BUDGET": True,
                          "refusals": R._REFUSALS.get(arm, {})})
            s = summarise(arm, per, extra)
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": s, "per_corner": per}, fh, indent=1, default=str)
            meta["arms"][arm] = {k: v for k, v in s.items() if k != "trials"}
            log("  %-6s novel %.4f  judge_acc %.4f  in %.0fs (score %.0fs)"
                % (arm, s["distinct_valid_novel_tuples_mean"], s["judge_accuracy_mean"],
                   time.time() - t, extra["seconds_score"]))
        except Exception as exc:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            meta["failed_arms"].append(arm)
            with open(leg / ("FAILED_%s.json" % arm), "w", encoding="utf-8") as fh:
                json.dump({"arm": arm, "error": "%s: %s" % (type(exc).__name__, exc)},
                          fh, indent=1)
            log("  !!!! %s FAILED: %s: %s" % (arm, type(exc).__name__, exc))
    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s failed=%s" % (leg, meta["failed_arms"]))
    return 3 if meta["failed_arms"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
