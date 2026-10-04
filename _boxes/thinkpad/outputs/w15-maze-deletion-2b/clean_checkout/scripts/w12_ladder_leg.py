"""w12 hub/composition-ladder -- ONE LEG = one (rung, seed).

The k-way families are the HARNESS's (``chlu.data.gen_kway_ladder``).  This runner
builds no family of its own: it swaps the condition under
``exp_gen_factor_wells.load_condition`` and drives the SHIPPED arm/judge path unchanged.

Rungs:  k = 2, 3, 4, 5, 6  (the ladder) and ``shipped`` (the value-hold-out condition on
disk, which carries row 3 and row 4 as they are published -- the reproduction reference).

Arms:
  store   single_design_K5             the w8 DowndatableStore.  K* UNPUBLISHED => K=5, DECLARED.
          pca_tensor                   the wave-5 subspace store.
          <store>__landscape_shuffled  charter C.2, POST-fit: atoms permuted, book kept.
          <store>__random_atoms        charter C.2 strong form: random atoms, book REFIT.
  learned rival_{mlp,lstm,transformer,cont_mlp,table}  @ the 524,288 B REFERENCE pin.

Every arm is scored on the rung's own corners with the harness's pre-declared estimand:
``exp_gen_factor_wells.per_corner_scores`` -> ``distinct_valid_novel_tuples``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chlu

if "CHLU-w12-composition-ladder" not in str(chlu.__file__):
    raise AssertionError("chlu resolves OUTSIDE this worktree: %s" % chlu.__file__)

from chlu.core.gen_read import TensorAddress, span_basis
from chlu.data import gen_kway_ladder as L
from chlu.experiments import exp_gen_delete_row7 as D7
from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.experiments import exp_w5_compose_field as W5
from chlu.experiments import exp_w8_downdate as W8
from w12_fastjudge import FastJudge, assert_identical

T0 = time.time()


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


STORE_ARMS = ("single_design_K5", "pca_tensor")
CONTROLS = ("landscape_shuffled", "random_atoms")
#: charter C.2 at this hub's interface.  ``landscape_shuffled`` = the atom VALUES
#: permuted with the fitted book kept (``perturb_modes('value_permute')``, POST-fit);
#: ``random_atoms`` = random atoms at the same Frobenius norm with the book REFIT
#: against them (``perturb_modes('random_span')``, PRE-fit) -- the strong form, the same
#: object ``exp_w5_compose_field`` calls ``pca_random_refit`` and ``w9_maze/arms.py``
#: calls ``random_atoms``.
CONTROL_KIND = {"landscape_shuffled": ("value_permute", "post"),
                "random_atoms": ("random_span", "pre")}
RIVALS = ("mlp", "lstm", "transformer", "cont_mlp", "table")


# ---------------------------------------------------------------------------
# the condition swap
# ---------------------------------------------------------------------------

def make_condition(rung: str) -> dict:
    """The rung's condition, in ``load_condition``'s own shape."""
    base = fw.load_condition()          # the shipped value-hold-out condition
    if rung == "shipped":
        base["rung"] = "shipped"
        base["corners"] = fw.pair_subcorners(base) + fw.nway_corners(base)
        return base
    k = int(rung)
    grid = base["grid"]
    cfg = L.KwayLadderConfig()
    split, cells, corners = L.build_kway_split(grid, cfg, k)
    cell_json = [{"cell_id": c.cell_id,
                  "values": {kk: [float(x) for x in vv] for kk, vv in c.values.items()},
                  "meta": c.meta,
                  "n_rows_realised": int(c.rows(grid).size)} for c in cells]
    return {
        "grid": grid,
        "train_rows": np.asarray(split.train_rows, np.int64),
        "test_rows": np.asarray(split.test_rows, np.int64),
        "cells": {"cells": cell_json, "config": base["cells"]["config"]},
        "query_sets": base["query_sets"],
        "scope": base["scope"],
        "rung": "k%d" % k,
        "corners": [{"corner_id": c.cell_id,
                     "family": c.meta.get("family", "kway%d" % k),
                     "rows": np.asarray(c.rows(grid), np.int64),
                     "meta": c.meta} for c in corners],
    }


_COND = {}


def _patched_load_condition():
    return _COND["cond"]


# ---------------------------------------------------------------------------
# the store arms
# ---------------------------------------------------------------------------

def _rank_width(store) -> dict:
    """rank/width on EVERY store row (wave 9's saturation trap)."""
    try:
        B = span_basis(store)
        ranks = [int(np.asarray(U).shape[1]) for U in B]
    except Exception:
        ranks = [int(np.asarray(M).shape[0]) for M in store.modes]
    widths = [int(np.asarray(c).size) for c in store.well_channels]
    rw = [r / w for r, w in zip(ranks, widths)]
    return {"rank_per_well": ranks, "width_per_well": widths,
            "rank_over_width_per_well": [round(x, 4) for x in rw],
            "rank_over_width_max": round(max(rw), 4),
            "rank_over_width_mean": round(float(np.mean(rw)), 4)}


def _address_book(leg: Path, store, basis, tag: str):
    cache = leg / ("address_%s_R%d_sw%d.pkl"
                   % (tag, int(R.READ_PINS["rank"]), int(R.READ_PINS["sweeps"])))
    if cache.exists():
        with open(cache, "rb") as fh:
            log("    address book: reused %s" % cache.name)
            return pickle.load(fh)
    grid, _ = fw.build_grid()
    pool = np.asarray(np.load(leg / "partitions.npz")["pool"], np.int64)
    pool = pool[: int(R.READ_PINS["n_addr_train"])]
    t = time.time()
    Y, T = R._pool_span_targets(store, pool, basis, log=log)
    ntr = int(0.9 * Y.shape[0])
    addr = TensorAddress([int(c) for c in grid.cardinalities],
                         rank=int(R.READ_PINS["rank"]),
                         ridge=float(R.READ_PINS["ridge"]),
                         seed=int(R.READ_PINS["address_seed"]))
    addr.multiplicative = True
    addr.fit(Y[:ntr], [t_[:ntr] for t_ in T], sweeps=int(R.READ_PINS["sweeps"]),
             log=log, Y_val=Y[ntr:], targets_val=[t_[ntr:] for t_ in T])
    with open(cache, "wb") as fh:
        pickle.dump(addr, fh)
    log("    address book %s: fitted in %.1fs" % (tag, time.time() - t))
    return addr


def build_store_obj(name: str, leg: Path, control) -> dict:
    """One store arm (or one of its two charter-C controls), on this rung's own pool."""
    kind, when = (None, None) if control is None else CONTROL_KIND[control]
    if name == "pca_tensor":
        base = W5.build_pca_content(leg, log=log)
        store = base["store"]
        led = dict(base["byte_ledger"])
        partition = base["partition"]
    elif name == "single_design_K5":
        cache = leg / "single_design_store.pkl"
        if cache.exists():
            with open(cache, "rb") as fh:
                store = pickle.load(fh)
        else:
            fw.build_partitions(leg, log=log)
            pool, wc, dead = W8._pool_and_wells(fw, leg)
            grid, _ = fw.build_grid()
            cards = [int(c) for c in grid.cardinalities]
            store = W8.build_store(fw, leg, pool, wc, cards, log=log)
            with open(cache, "wb") as fh:
                pickle.dump(store, fh)
        led = store.byte_ledger(int(W8.W8_PINS["store_bytes"]))
        partition = W8.W8_PINS["partition"]
    else:
        raise ValueError(name)

    if kind is not None and when == "pre":
        store = W5.perturb_modes(store, kind, int(fw.PINS["seed"]))
    basis = span_basis(store)
    tag = name if (control is None or when == "post") else "%s_%s" % (name, control)
    addr = _address_book(leg, store, basis, tag)
    obj = {"arm": name if control is None else "%s__%s" % (name, control),
           "read": "tensor", "kind": "span", "store": store, "basis": basis,
           "address": addr, "byte_ledger": led, "address_bytes": addr.byte_ledger(),
           "head_bytes": {"head_bytes": 0, "note": "the address book IS the head"},
           "partition": partition, "protos": None,
           "K_star_note": ("K* is NOT published (w12-single-design-crossover/"
                           "CROSSOVER-val.json: K_star=null). DECLARED: K = 5.")}
    if kind is not None and when == "post":
        obj["store"] = W5.perturb_modes(store, kind, int(fw.PINS["seed"]))
        obj["basis"] = W5.basis_at_fixed_rank(
            obj["store"], [np.asarray(U).shape[1] for U in basis])
        obj["perturbation"] = {"kind": kind, "when": "AFTER the address book was fitted"}
    elif kind is not None:
        obj["perturbation"] = {"kind": kind, "when": "BEFORE the address book was fitted"}
    obj["rank_width"] = _rank_width(obj["store"])
    return obj


def build_rival_obj(name: str, seed: int) -> dict:
    from chlu.eval.rivals import w4_api
    rv = w4_api.load_rival(int(seed), name)
    return {"arm": "rival_%s" % name, "read": "rival", "kind": "rival", "rival": rv,
            "rival_partial": None, "store": None, "protos": None,
            "byte_ledger": {"deployed_bytes": None,
                            "param_count": rv.meta.get("param_count"),
                            "budget_bytes": 524288,
                            "what": ("the 524,288 B REFERENCE pin: this rival set is "
                                     "matched to 131,072 float32 parameters")},
            "head_bytes": {"head_bytes": 0},
            "population": rv.population.get("fit_population"),
            "param_count": rv.meta.get("param_count")}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_arm(obj: dict, cond: dict, judge, bitmap, seen_mask) -> list:
    """Every corner of the rung, through the shipped acceptance / validity / scoring path.

    ⭐ The corners are answered in ONE call rather than one call each. MEASURED
    (`logs/batch_check.log`): `exp_gen_read.arm_features` returns `IDENTICAL: True,
    max|diff| 0.0` batched vs per-corner, and the judge's tuples are `IDENTICAL: True`
    at the pinned `eval_batch = 256` vs 72-row fragments. ⛔ Nothing is approximated:
    the reads and the judge are row-wise, the offsets slice the same rows back into the
    same corners, and `per_corner_scores` is the shipped function, unchanged.
    """
    grid = cond["grid"]
    corners = cond["corners"]
    vals_cat, offs = [], []
    p = 0
    for c in corners:
        vals = fw._request_values_from_rows(grid, c["rows"])
        vals_cat.append(vals)
        offs.append((p, p + vals.shape[0]))
        p += vals.shape[0]
    V = np.concatenate(vals_cat)
    feats, oor = R.arm_features(obj, V, grid)
    acc = fw.accept_rule(V, oor, np.zeros(V.shape[0], bool), bitmap, grid)
    tup = judge.read_back(feats, int(fw.PINS["eval_batch"]))
    valid = fw.judge_valid(V, tup, grid) & acc
    return fw.per_corner_scores(corners, V, tup, valid, acc, offs, grid, seen_mask)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="")
    a = ap.parse_args(argv)

    leg = Path(a.out) / ("%s_seed%d" % (a.rung, a.seed))
    leg.mkdir(parents=True, exist_ok=True)
    fw.PINS["condition"] = "test"       # the scored geometry; no val-corner exclusion
    fw.PINS["seed"] = int(a.seed)
    cache = D7.assert_generation2(int(a.seed))
    log("rung=%s seed=%d cache gen %s scope %s n_rows %s root %s"
        % (a.rung, a.seed, cache["generation"], cache["scope"], cache["n_rows"],
           cache["root"]))
    log("chlu.__file__ = %s" % chlu.__file__)

    cond = make_condition(a.rung)
    _COND["cond"] = cond
    fw.load_condition = _patched_load_condition
    W5.fw.load_condition = _patched_load_condition
    R.fw.load_condition = _patched_load_condition
    grid = cond["grid"]
    corners = cond["corners"]
    chk = fw.assert_corners_disjoint_and_heldout(corners, cond["test_rows"], a.rung)
    log("corners: %s" % chk)

    fw.build_partitions(leg, log=log)
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    judge = fw.Judge(int(a.seed), log=log)
    # the read-back path, with the jit hoisted out of the per-call body.  Proven
    # element-for-element identical to the shipped Judge on REAL features before use;
    # the leg refuses to produce a number if it is not.  See scripts/w12_fastjudge.py.
    fast = FastJudge(judge)
    _pool = np.asarray(np.load(leg / "partitions.npz")["pool"], np.int64)[:256]
    _t = time.time()
    _chk = assert_identical(judge, fast, fw.gather(fw.features_mmap(int(a.seed)), _pool))
    log("FastJudge positive control: %s (%d requests, %d elements) in %.1fs"
        % (_chk["identical"], _chk["n_requests"], _chk["n_elements_compared"],
           time.time() - _t))

    want = [x for x in a.arms.split(",") if x] or (
        list(STORE_ARMS)
        + ["%s__%s" % (s, c) for s in STORE_ARMS for c in CONTROLS]
        + ["rival_%s" % r for r in RIVALS])

    meta = {"rung": a.rung, "seed": int(a.seed), "chlu_file": str(chlu.__file__),
            "cache": cache, "corner_check": chk,
            "n_train": int(cond["train_rows"].size),
            "n_test": int(cond["test_rows"].size),
            "n_corners": len(corners),
            "fastjudge_positive_control": _chk,
            "corner_ids": [c["corner_id"] for c in corners],
            "estimand": {
                "function": "exp_gen_factor_wells.paired_estimand",
                "metric": "distinct_valid_novel_tuples",
                "margin_name": "DECORATION_MARGIN_NOVELTY",
                "margin": float(fw.DECORATION_MARGIN_NOVELTY),
                "unit": "corner",
                "source": ("LADDER-READY.json:estimand -- pre-declared by "
                           "w12/kway-ladder-harness, not restated here")},
            "arms": {}}
    failed = []
    for arm in want:
        done = leg / ("rows_%s.json" % arm)
        if done.exists():
            log("  %s: already on disk, skipping" % arm)
            with open(done, encoding="utf-8") as fh:
                meta["arms"][arm] = json.load(fh)["summary"]
            continue
        t = time.time()
        log("---- %s ----" % arm)
        try:
            if arm.startswith("rival_"):
                obj = build_rival_obj(arm[len("rival_"):], int(a.seed))
                extra = {"param_count": obj["param_count"],
                         "population": obj["population"],
                         "byte_matching": "524,288 B REFERENCE pin"}
            else:
                base, _, ctl = arm.partition("__")
                obj = build_store_obj(base, leg, ctl or None)
                extra = {"rank_width": obj["rank_width"],
                         "byte_ledger": {k: v for k, v in obj["byte_ledger"].items()
                                         if k != "what"},
                         "address_bytes": obj["address_bytes"],
                         "perturbation": obj.get("perturbation"),
                         "byte_matching": "1x the arm's own deployed total (store+book)"}
            per = score_arm(obj, cond, fast, bitmap, seen_mask)
            dv = np.array([r["distinct_valid_novel_tuples"] for r in per], float)
            ja = np.array([r["judge_accuracy"] for r in per], float)
            summ = {"arm": arm, "n_corners": len(per),
                    "distinct_valid_novel_tuples_mean": float(dv.mean()),
                    "distinct_valid_novel_tuples_sd": float(dv.std(ddof=1)),
                    "judge_accuracy_mean": float(ja.mean()),
                    "judge_accuracy_sd": float(ja.std(ddof=1)),
                    "seconds": time.time() - t}
            summ.update(extra)
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": summ, "per_corner": per}, fh, indent=1,
                          default=str)
            meta["arms"][arm] = summ
            log("  %s: novel %.4f +- %.4f  judge_acc %.4f  in %.1fs"
                % (arm, dv.mean(), dv.std(ddof=1), ja.mean(), time.time() - t))
            del obj
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failed.append(arm)
            with open(leg / ("FAILED_%s.json" % arm), "w", encoding="utf-8") as fh:
                json.dump({"arm": arm, "error": "%s: %s" % (type(exc).__name__, exc)},
                          fh, indent=1)
            log("  !!!! %s FAILED: %s: %s" % (arm, type(exc).__name__, exc))
    meta["failed_arms"] = failed
    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s  failed=%s" % (leg, failed))
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
