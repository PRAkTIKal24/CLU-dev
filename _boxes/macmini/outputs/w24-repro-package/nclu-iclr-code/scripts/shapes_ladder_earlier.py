"""One rung of the n-way ladder: builds the rung's hold-out and fit pool (writes its partitions.npz),
the K=5 label-derived n-CLU, and scores the earlier baseline set and the table floor on the rung.

    python scripts/shapes_ladder_earlier.py --rung 3 --seed 0 --out runs/shapes3d/ladder_earlier
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
import nclu
from pathlib import Path as _P

if _P(__file__).resolve().parents[1] not in _P(nclu.__file__).resolve().parents:
    raise AssertionError("the package resolves outside this checkout: %s" % nclu.__file__)
from nclu.core.gen_read import TensorAddress, span_basis
from nclu.data import gen_kway_ladder as L
from nclu.experiments import gen_deletion as D7
from nclu.experiments import factor_wells as fw
from nclu.experiments import exp_gen_read as R
from nclu.experiments import composition_cell as CC
from nclu.experiments import deletion_full_square as FSQ
from shapes_fast_judge import FastJudge, assert_identical

T0 = time.time()


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


STORE_ARMS = ("single_design_K5", "pca_tensor")
CONTROLS = ("landscape_shuffled", "random_atoms")
CONTROL_KIND = {
    "landscape_shuffled": ("value_permute", "post"),
    "random_atoms": ("random_span", "pre"),
}
RIVALS = ("mlp", "lstm", "transformer", "cont_mlp", "table")


def make_condition(rung: str) -> dict:
    """The rung's condition, in ``load_condition``'s own shape."""
    base = fw.load_condition()
    if rung == "shipped":
        base["rung"] = "shipped"
        base["corners"] = fw.pair_subcorners(base) + fw.nway_corners(base)
        return base
    k = int(rung)
    grid = base["grid"]
    cfg = L.KwayLadderConfig()
    split, cells, corners = L.build_kway_split(grid, cfg, k)
    cell_json = [
        {
            "cell_id": c.cell_id,
            "values": {kk: [float(x) for x in vv] for kk, vv in c.values.items()},
            "meta": c.meta,
            "n_rows_realised": int(c.rows(grid).size),
        }
        for c in cells
    ]
    return {
        "grid": grid,
        "train_rows": np.asarray(split.train_rows, np.int64),
        "test_rows": np.asarray(split.test_rows, np.int64),
        "cells": {"cells": cell_json, "config": base["cells"]["config"]},
        "query_sets": base["query_sets"],
        "scope": base["scope"],
        "rung": "k%d" % k,
        "corners": [
            {
                "corner_id": c.cell_id,
                "family": c.meta.get("family", "kway%d" % k),
                "rows": np.asarray(c.rows(grid), np.int64),
                "meta": c.meta,
            }
            for c in corners
        ],
    }


_COND = {}


def _patched_load_condition():
    return _COND["cond"]


def _rank_width(store) -> dict:
    try:
        B = span_basis(store)
        ranks = [int(np.asarray(U).shape[1]) for U in B]
    except Exception:
        ranks = [int(np.asarray(M).shape[0]) for M in store.modes]
    widths = [int(np.asarray(c).size) for c in store.well_channels]
    rw = [r / w for r, w in zip(ranks, widths)]
    return {
        "rank_per_well": ranks,
        "width_per_well": widths,
        "rank_over_width_per_well": [round(x, 4) for x in rw],
        "rank_over_width_max": round(max(rw), 4),
        "rank_over_width_mean": round(float(np.mean(rw)), 4),
    }


def _address_book(leg: Path, store, basis, tag: str):
    cache = leg / (
        "address_%s_R%d_sw%d.pkl" % (tag, int(R.READ_PINS["rank"]), int(R.READ_PINS["sweeps"]))
    )
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
    addr = TensorAddress(
        [int(c) for c in grid.cardinalities],
        rank=int(R.READ_PINS["rank"]),
        ridge=float(R.READ_PINS["ridge"]),
        seed=int(R.READ_PINS["address_seed"]),
    )
    addr.multiplicative = True
    addr.fit(
        Y[:ntr],
        [t_[:ntr] for t_ in T],
        sweeps=int(R.READ_PINS["sweeps"]),
        log=log,
        Y_val=Y[ntr:],
        targets_val=[t_[ntr:] for t_ in T],
    )
    with open(cache, "wb") as fh:
        pickle.dump(addr, fh)
    log("    address book %s: fitted in %.1fs" % (tag, time.time() - t))
    return addr


def build_store_obj(name: str, leg: Path, control) -> dict:
    kind, when = (None, None) if control is None else CONTROL_KIND[control]
    if name == "pca_tensor":
        base = CC.build_pca_content(leg, log=log)
        store = base["store"]
        led = dict(base["byte_footprint"])
        partition = base["partition"]
    elif name == "single_design_K5":
        cache = leg / "single_design_store.pkl"
        if cache.exists():
            with open(cache, "rb") as fh:
                store = pickle.load(fh)
        else:
            fw.build_partitions(leg, log=log)
            pool, wc, dead = FSQ._pool_and_wells(fw, leg)
            grid, _ = fw.build_grid()
            cards = [int(c) for c in grid.cardinalities]
            store = FSQ.build_store(fw, leg, pool, wc, cards, log=log)
            with open(cache, "wb") as fh:
                pickle.dump(store, fh)
        led = store.byte_footprint(int(FSQ.FSQ_PINS["store_bytes"]))
        partition = FSQ.FSQ_PINS["partition"]
    else:
        raise ValueError(name)
    if kind is not None and when == "pre":
        store = CC.perturb_modes(store, kind, int(fw.PINS["seed"]))
    basis = span_basis(store)
    tag = name if control is None or when == "post" else "%s_%s" % (name, control)
    addr = _address_book(leg, store, basis, tag)
    obj = {
        "arm": name if control is None else "%s__%s" % (name, control),
        "read": "tensor",
        "kind": "span",
        "store": store,
        "basis": basis,
        "address": addr,
        "byte_footprint": led,
        "address_bytes": addr.byte_footprint(),
        "head_bytes": {"head_bytes": 0, "note": "the address book IS the head"},
        "partition": partition,
        "protos": None,
        "K_star_note": "K* is NOT published (shapes3d/k_sweep/CROSSOVER-val.json: K_star=null). DECLARED: K = 5.",
    }
    if kind is not None and when == "post":
        obj["store"] = CC.perturb_modes(store, kind, int(fw.PINS["seed"]))
        obj["basis"] = CC.basis_at_fixed_rank(obj["store"], [np.asarray(U).shape[1] for U in basis])
        obj["perturbation"] = {"kind": kind, "when": "AFTER the address book was fitted"}
    elif kind is not None:
        obj["perturbation"] = {"kind": kind, "when": "BEFORE the address book was fitted"}
    obj["rank_width"] = _rank_width(obj["store"])
    return obj


def build_rival_obj(name: str, seed: int) -> dict:
    from nclu.eval.baselines import api as baseline_api

    rv = baseline_api.load_rival(int(seed), name)
    return {
        "arm": "rival_%s" % name,
        "read": "rival",
        "kind": "rival",
        "rival": rv,
        "rival_partial": None,
        "store": None,
        "protos": None,
        "byte_footprint": {
            "deployed_bytes": None,
            "param_count": rv.meta.get("param_count"),
            "budget_bytes": 524288,
            "what": "the 524,288 B REFERENCE pin: this rival set is matched to 131,072 float32 parameters",
        },
        "head_bytes": {"head_bytes": 0},
        "population": rv.population.get("fit_population"),
        "param_count": rv.meta.get("param_count"),
    }


def score_arm(obj: dict, cond: dict, judge, bitmap, seen_mask) -> list:
    grid = cond["grid"]
    corners = cond["corners"]
    vals_cat, offs = ([], [])
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
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = int(a.seed)
    cache = D7.assert_generation2(int(a.seed))
    log(
        "rung=%s seed=%d cache gen %s scope %s n_rows %s root %s"
        % (a.rung, a.seed, cache["generation"], cache["scope"], cache["n_rows"], cache["root"])
    )
    log("nclu.__file__ = %s" % nclu.__file__)
    cond = make_condition(a.rung)
    _COND["cond"] = cond
    fw.load_condition = _patched_load_condition
    CC.fw.load_condition = _patched_load_condition
    R.fw.load_condition = _patched_load_condition
    grid = cond["grid"]
    corners = cond["corners"]
    chk = fw.assert_corners_disjoint_and_heldout(corners, cond["test_rows"], a.rung)
    log("corners: %s" % chk)
    fw.build_partitions(leg, log=log)
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    judge = fw.Judge(int(a.seed), log=log)
    fast = FastJudge(judge)
    _pool = np.asarray(np.load(leg / "partitions.npz")["pool"], np.int64)[:256]
    _t = time.time()
    _chk = assert_identical(judge, fast, fw.gather(fw.features_mmap(int(a.seed)), _pool))
    log(
        "FastJudge positive control: %s (%d requests, %d elements) in %.1fs"
        % (_chk["identical"], _chk["n_requests"], _chk["n_elements_compared"], time.time() - _t)
    )
    want = [x for x in a.arms.split(",") if x] or list(STORE_ARMS) + [
        "%s__%s" % (s, c) for s in STORE_ARMS for c in CONTROLS
    ] + ["rival_%s" % r for r in RIVALS]
    meta = {
        "rung": a.rung,
        "seed": int(a.seed),
        "package_file": str(nclu.__file__),
        "cache": cache,
        "corner_check": chk,
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
            "source": "LADDER-READY.json:estimand -- pre-declared by kway-ladder-harness, not restated here",
        },
        "arms": {},
    }
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
                obj = build_rival_obj(arm[len("rival_") :], int(a.seed))
                extra = {
                    "param_count": obj["param_count"],
                    "population": obj["population"],
                    "byte_matching": "524,288 B REFERENCE pin",
                }
            else:
                base, _, ctl = arm.partition("__")
                obj = build_store_obj(base, leg, ctl or None)
                extra = {
                    "rank_width": obj["rank_width"],
                    "byte_footprint": {
                        k: v for k, v in obj["byte_footprint"].items() if k != "what"
                    },
                    "address_bytes": obj["address_bytes"],
                    "perturbation": obj.get("perturbation"),
                    "byte_matching": "1x the arm's own deployed total (store+book)",
                }
            per = score_arm(obj, cond, fast, bitmap, seen_mask)
            dv = np.array([r["distinct_valid_novel_tuples"] for r in per], float)
            ja = np.array([r["judge_accuracy"] for r in per], float)
            summ = {
                "arm": arm,
                "n_corners": len(per),
                "distinct_valid_novel_tuples_mean": float(dv.mean()),
                "distinct_valid_novel_tuples_sd": float(dv.std(ddof=1)),
                "judge_accuracy_mean": float(ja.mean()),
                "judge_accuracy_sd": float(ja.std(ddof=1)),
                "seconds": time.time() - t,
            }
            summ.update(extra)
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": summ, "per_corner": per}, fh, indent=1, default=str)
            meta["arms"][arm] = summ
            log(
                "  %s: novel %.4f +- %.4f  judge_acc %.4f  in %.1fs"
                % (arm, dv.mean(), dv.std(ddof=1), ja.mean(), time.time() - t)
            )
            del obj
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failed.append(arm)
            with open(leg / ("FAILED_%s.json" % arm), "w", encoding="utf-8") as fh:
                json.dump({"arm": arm, "error": "%s: %s" % (type(exc).__name__, exc)}, fh, indent=1)
            log("  !!!! %s FAILED: %s: %s" % (arm, type(exc).__name__, exc))
    meta["failed_arms"] = failed
    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s  failed=%s" % (leg, failed))
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
