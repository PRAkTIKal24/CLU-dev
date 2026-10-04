"""deletion."""

from __future__ import annotations
import time
from typing import Callable, Optional
import jax
import numpy as np
from nclu.core.gen_request import FactorPrototypes, requests_from_rows
from nclu.eval.gen_probe import factor_labels
from nclu.eval.baselines import learners as L
from nclu.eval.baselines.common import render, score_images

DELETED_FACTOR = "object_hue"
DELETED_VALUE = 0.5
SCOPE = "SCOPE OF THE DELETION COMPARATOR. The value is held out of the ARM'S FIT POOL only, not from the encoder, the decoder or the judge, which are frozen and were fitted on a split containing every hue. 'Never trained on' means the arm never fitted a row carrying that value. The decoder can still render it and the judge can still recognise it, which is what makes the comparison scorable at all."


def _value_index(grid, factor: str, value: float) -> int:
    k = list(grid.names).index(factor)
    vals = np.asarray(grid.values[k], np.float64)
    i = int(np.argmin(np.abs(vals - float(value))))
    if not np.isclose(vals[i], float(value)):
        raise AssertionError(f"{factor}={value} is not a grid value; nearest is {vals[i]}")
    return (k, i)


def partition_rows(
    ctx: dict,
    factor: str = DELETED_FACTOR,
    value: float = DELETED_VALUE,
    n_eval: int = 1024,
    log: Callable = print,
) -> dict:
    """Split fit rows into kept/dropped, and build the forget/retain eval sets."""
    grid = ctx["grid"]
    k, vi = _value_index(grid, factor, value)
    fit_rows = np.asarray(ctx["row_sets"]["fit"], np.int64)
    fi = grid.factor_indices_of(fit_rows)
    drop = fi[:, k] == vi
    kept_rows = fit_rows[~drop]
    dropped_rows = fit_rows[drop]
    val_rows = np.asarray(ctx["row_sets"]["val"], np.int64)
    vfi = grid.factor_indices_of(val_rows)
    forget = val_rows[vfi[:, k] == vi]
    retain = val_rows[vfi[:, k] != vi]
    if forget.size == 0:
        pool = np.setdiff1d(np.asarray(ctx["pool"], np.int64), fit_rows)
        pfi = grid.factor_indices_of(pool)
        forget = pool[pfi[:, k] == vi][:n_eval]
        retain = pool[pfi[:, k] != vi][:n_eval]
        src = "store pool minus every fitted row (the 512-row validation cut carried no instance of the deleted value)"
    else:
        retain = retain[:n_eval]
        forget = forget[:n_eval]
        src = "the validation rows"
    log(
        f"  [deletion] {factor}={value} (grid index {vi}): dropping {int(drop.sum())}/{fit_rows.size} fit rows; forget set {forget.size}, retain set {retain.size}, from {src}"
    )
    return {
        "factor": factor,
        "value": float(value),
        "factor_index": int(k),
        "value_index": int(vi),
        "kept_rows": kept_rows,
        "dropped_rows": dropped_rows,
        "forget_rows": forget,
        "retain_rows": retain,
        "eval_source": src,
        "n_fit_before": int(fit_rows.size),
        "n_fit_after": int(kept_rows.size),
        "n_dropped": int(dropped_rows.size),
    }


def _score_sets(ctx: dict, feats_fn: Callable, part: dict, truth: dict) -> dict:
    """Judge accuracy and feature error on the forget and retain sets."""
    out = {}
    for name in ("forget", "retain"):
        rows = part[f"{name}_rows"]
        if rows.size == 0:
            out[name] = {"NOT_RUN": "empty set"}
            continue
        reqs = requests_from_rows(ctx["grid"], rows)
        F = feats_fn(reqs, rows)
        imgs = render(ctx["decoder"], F, ctx["feat_shape"], ctx["eval_batch"])
        lab = factor_labels(ctx["grid"], rows)
        sc = score_images(ctx["external"], imgs, lab, ctx["names"], ctx["eval_batch"])
        out[name] = {
            "n": int(rows.size),
            "judge_aggregate": float(sc["aggregate"]),
            "judge_per_factor": {k: float(v) for k, v in sc["per_factor"].items()},
            "judge_on_deleted_factor": float(sc["per_factor"][part["factor"]]),
            "feature_mse": float(np.mean((F - truth[name]) ** 2)),
        }
    return out


def run_deletion_comparator(
    ctx: dict,
    factor: str = DELETED_FACTOR,
    value: float = DELETED_VALUE,
    families: Optional[list] = None,
    param_target: Optional[int] = None,
    log: Callable = print,
) -> dict:
    """Retrain-from-scratch-without-the-value, against the cheap edit, with costs."""
    from nclu.eval.baselines.common import gather_gen2, load_gen2_cache

    part = partition_rows(ctx, factor, value, log=log)
    grid = ctx["grid"]
    truth = {}
    for name in ("forget", "retain"):
        rows = part[f"{name}_rows"]
        truth[name] = (
            np.asarray(ctx["_gather"](rows), np.float32).reshape((rows.size, -1))
            if rows.size
            else np.zeros((0, ctx["D"]), np.float32)
        )
    out = {
        "scope": SCOPE,
        "partition": {k: v for k, v in part.items() if not isinstance(v, np.ndarray)},
        "methods": {},
    }
    full = ctx["protos"]
    t0 = time.time()
    out["methods"]["table_full_pool"] = {
        "role": "the arm before any deletion - the reference both methods move from",
        "scores": _score_sets(ctx, lambda r, rows: full.embed_many(r), part, truth),
        "wall_clock_s": float(time.time() - t0),
        "refit_rows": int(part["n_fit_before"]),
    }
    t0 = time.time()
    Xkept = ctx["_gather"](part["kept_rows"])
    retrained = FactorPrototypes.fit(grid, Xkept, part["kept_rows"])
    cost_retrain = float(time.time() - t0)
    out["methods"]["table_retrain_from_scratch"] = {
        "role": "GOLD STANDARD: the value was never in the pool",
        "scores": _score_sets(ctx, lambda r, rows: retrained.embed_many(r), part, truth),
        "wall_clock_s": cost_retrain,
        "refit_rows": int(part["n_fit_after"]),
        "n_per_value_after": [int(c) for c in retrained.n_per_value[part["factor_index"]]],
    }
    t0 = time.time()
    edited_protos = list(full.protos)
    P = np.array(edited_protos[part["factor_index"]], copy=True)
    P[part["value_index"]] = full.mu_global
    edited_protos[part["factor_index"]] = P
    edited = FactorPrototypes(
        names=full.names,
        values=full.values,
        protos=tuple(edited_protos),
        mu_global=full.mu_global,
        n_rows_fit=full.n_rows_fit,
        n_per_value=full.n_per_value,
    )
    cost_edit = float(time.time() - t0)
    out["methods"]["table_edit_mean_substitution"] = {
        "role": "the cheap O(D) alternative: replace the deleted value's prototype with the global mean, touching nothing else",
        "scores": _score_sets(ctx, lambda r, rows: edited.embed_many(r), part, truth),
        "wall_clock_s": cost_edit,
        "refit_rows": 0,
    }
    if families:
        key = jax.random.PRNGKey(int(ctx["seed"]) * 7717 + 20260919)
        target = int(param_target or full.n_bytes // 4)
        bs = L.builders(ctx["cardinalities"], ctx["D"], key)
        for fam in families:
            build, kind = bs[fam]
            w, n_params, err = L.match_width(build, target, log=log)
            Xin_full = L.encode_inputs(kind, ctx["req_fit"], ctx["cardinalities"])
            keep_mask = np.isin(ctx["row_sets"]["fit"], part["kept_rows"])
            Yfull = np.asarray(ctx["X_fit"], np.float32).reshape((Xin_full.shape[0], -1))
            t0 = time.time()
            key, fk = jax.random.split(key)
            m, info = L.train(
                build(w), Xin_full[keep_mask], Yfull[keep_mask], fk, epochs=20, lr=0.001, wd=0.0
            )
            cost = float(time.time() - t0)

            def _f(reqs, rows, _m=m, _kind=kind):
                from nclu.core.gen_request import dense_request_arrays

                dr = dense_request_arrays(ctx["grid"], reqs)
                return L.predict(_m, L.encode_inputs(_kind, dr, ctx["cardinalities"]))

            out["methods"][f"{fam}_retrain_from_scratch"] = {
                "role": f"GOLD STANDARD for the {fam} rival",
                "param_count": int(n_params),
                "param_error": float(err),
                "scores": _score_sets(ctx, _f, part, truth),
                "wall_clock_s": cost,
                "refit_rows": int(keep_mask.sum()),
                "fit": info,
            }
    out["cost_line"] = cost_line(out["methods"], part)
    log("  [deletion] COST LINE: " + out["cost_line"]["one_line"])
    return out


def cost_line(methods: dict, part: dict) -> dict:
    gold = methods.get("table_retrain_from_scratch", {})
    cheap = methods.get("table_edit_mean_substitution", {})
    tg, tc = (float(gold.get("wall_clock_s", 0.0)), float(cheap.get("wall_clock_s", 0.0)))
    TIMER_FLOOR = 0.0001
    below_resolution = tc < TIMER_FLOOR
    ratio = None if below_resolution else tg / tc
    gf = (gold.get("scores", {}).get("forget") or {}).get("judge_on_deleted_factor")
    cf = (cheap.get("scores", {}).get("forget") or {}).get("judge_on_deleted_factor")
    gr = (gold.get("scores", {}).get("retain") or {}).get("judge_on_deleted_factor")
    cr = (cheap.get("scores", {}).get("retain") or {}).get("judge_on_deleted_factor")
    return {
        "deleted": f"{part['factor']}={part['value']} (grid index {part['value_index']})",
        "rows_dropped": int(part["n_dropped"]),
        "rows_refitted_by_gold_standard": int(part["n_fit_after"]),
        "rows_refitted_by_edit": 0,
        "wall_clock_gold_standard_s": tg,
        "wall_clock_edit_s": tc,
        "edit_below_timer_resolution": bool(below_resolution),
        "cost_ratio_gold_over_edit": ratio,
        "cost_ratio_note": "not reported as a ratio: the edit is a single O(D) row copy and falls below the clock's resolution, so a ratio against it would be an artifact of the timer"
        if below_resolution
        else "gold-standard wall clock over edit wall clock",
        "judge_on_deleted_factor": {
            "gold_forget": gf,
            "edit_forget": cf,
            "gold_retain": gr,
            "edit_retain": cr,
        },
        "one_line": f"deleting {part['factor']}={part['value']} costs {tg:.2f}s and a refit of {part['n_fit_after']:,} rows by the gold standard, against "
        + (
            "an O(D) copy of one prototype row and 0 refitted rows by mean-substitution (below timer resolution, so no ratio is quoted)"
            if below_resolution
            else f"{tc:.4f}s and 0 rows by mean-substitution ({ratio:.0f}x)"
        )
        + f"; on the forget set the judge reads the deleted factor at {(gf if gf is None else round(gf, 4))} (gold) vs {(cf if cf is None else round(cf, 4))} (edit), and on the retain set {(gr if gr is None else round(gr, 4))} vs {(cr if cr is None else round(cr, 4))}",
        "scope": SCOPE,
    }
