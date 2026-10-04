"""shapes coverage control."""

from __future__ import annotations
from nclu.paths import run_dir
import argparse
import json
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
from nclu.core.gen_store import dead_channels
from nclu.data import gen_kway_ladder as L
from nclu.experiments import gen_deletion as D7
from nclu.experiments import factor_wells as fw
from nclu.experiments import exp_gen_read as R
from nclu.experiments import composition_cell as CC
import shapes_ladder_learned as ONEX
import shapes_ladder_nclu as KS
import shapes_ladder_lib as LEG
from shapes_nclu_lib import ShimStore, four_part_footprint, perturb_modes, qr_basis
from shapes_fast_judge import FastJudge, assert_identical

T0 = time.time()
EARLIER_LADDER_RUNS = Path(str(run_dir("shapes3d/ladder_earlier")))
MID_SCALE_IDX = (3,)
RUNGS = (3, 6)


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


def _val_options(grid, cfg, pair_cells):
    opts = _ORIG_OPTS(grid, cfg, pair_cells)
    sv = grid.values[grid.index_of("scale")]
    opts["scale"] = [
        {
            "label": "m",
            "values": {"scale": tuple((float(sv[i]) for i in MID_SCALE_IDX))},
            "meta": {"scale_corner": "middle (VAL)"},
        }
    ]
    return opts


_ORIG_OPTS = L.coordinate_options


def val_cells_and_corners(grid, cfg, k, pair):
    L.coordinate_options = _val_options
    try:
        cells = L.build_kway_cells(grid, cfg, k, pair_cells=pair)
        corners = L.kway_corners(grid, cfg, k, cells, pair_cells=pair)
    finally:
        L.coordinate_options = _ORIG_OPTS
    return (cells, corners)


def build_conditions(log=log) -> dict:
    base = fw.load_condition()
    grid = base["grid"]
    cfg = L.KwayLadderConfig()
    from nclu.data.gen_value_holdout import build_pair_cells

    pair = build_pair_cells(grid, cfg.value_cfg)
    out, test_tr, test_te = ({}, {}, {})
    for k in RUNGS:
        split, _, _ = L.build_kway_split(grid, cfg, k)
        test_tr[k] = np.asarray(split.train_rows, np.int64)
        test_te[k] = np.asarray(split.test_rows, np.int64)
    audit = {
        "train3_subset_of_train6": bool(np.isin(test_tr[3], test_tr[6]).all()),
        "test_rung_train_rows": {str(k): int(test_tr[k].size) for k in RUNGS},
        "test_rung_test_rows": {str(k): int(test_te[k].size) for k in RUNGS},
    }
    corner_rowsets = {}
    for k in RUNGS:
        cells, corners = val_cells_and_corners(grid, cfg, k, pair)
        cell_rows = np.unique(np.concatenate([c.rows(grid) for c in cells]))
        crn = [
            {
                "corner_id": "val-" + c.cell_id,
                "family": "val_kway%d" % k,
                "rows": np.asarray(c.rows(grid), np.int64),
                "meta": c.meta,
            }
            for c in corners
        ]
        allrows = np.concatenate([c["rows"] for c in crn])
        for kk in RUNGS:
            bad = np.intersect1d(allrows, test_te[kk]).size
            if bad:
                raise AssertionError(
                    "val rung %d: %d corner rows are TEST rows of rung %d" % (k, bad, kk)
                )
            bad = np.intersect1d(cell_rows, test_te[kk]).size
            if bad:
                raise AssertionError(
                    "val rung %d: %d cell rows are TEST rows of rung %d" % (k, bad, kk)
                )
        train = np.setdiff1d(test_tr[k], cell_rows)
        held = np.union1d(test_te[k], cell_rows)
        cond = {
            "grid": grid,
            "train_rows": train,
            "test_rows": held,
            "cells": base["cells"],
            "query_sets": base["query_sets"],
            "scope": base["scope"],
            "rung": "val_k%d" % k,
            "corners": crn,
        }
        chk = fw.assert_corners_disjoint_and_heldout(crn, held, "val_k%d" % k)
        bm = fw.value_presence_bitmap(grid, train)
        fi = grid.factor_indices_of(allrows)
        missing = {
            grid.names[f]: sorted(
                set(np.unique(fi[:, f]).tolist()) - set(np.flatnonzero(bm[f]).tolist())
            )
            for f in range(fi.shape[1])
        }
        missing = {f: v for f, v in missing.items() if v}
        if missing:
            raise AssertionError(
                "val rung %d: requested values absent from train: %s" % (k, missing)
            )
        corner_rowsets[k] = np.sort(allrows)
        out[k] = cond
        audit["val_k%d" % k] = {
            "n_cells": len(cells),
            "n_cell_rows": int(cell_rows.size),
            "n_corners": len(crn),
            "rows_per_corner": sorted({int(c["rows"].size) for c in crn}),
            "train_rows": int(train.size),
            "held_rows": int(held.size),
            "corner_check": chk,
            "pool_size_pin_served": bool(train.size >= fw.PINS["n_pool"]),
        }
        log(
            "val k=%d: %d cells (%d rows), %d corners, train %d"
            % (k, len(cells), cell_rows.size, len(crn), train.size)
        )
    same = bool(np.array_equal(corner_rowsets[3], corner_rowsets[6]))
    if not same:
        raise AssertionError("val k=3 and val k=6 do not probe the identical rows")
    audit["val_rungs_probe_identical_rows"] = same
    key = lambda c: tuple(np.sort(c["rows"])[:4].tolist()) + (int(c["rows"].size),)
    k3 = {key(c): c["corner_id"] for c in out[3]["corners"]}
    audit["corner_pairing_k6_to_k3"] = {c["corner_id"]: k3[key(c)] for c in out[6]["corners"]}
    return {"cond": out, "audit": audit, "test_train": test_tr}


def pool_and_dead(train: np.ndarray, seed: int):
    """``fw.build_partitions``'s pool and dead mask, by the same two functions."""
    pool = fw.fit_pool(train, fw.PINS["n_pool"], fw.PINS["pool_rng_seed"], exclude=None)
    X = fw.gather(fw.features_mmap(int(seed)), pool[: int(fw.PINS["n_corr"])])
    dead = dead_channels(X, eps=float(fw.PINS["dead_channel_eps"]))
    return (pool, np.asarray(dead).ravel().astype(bool))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--out", default=str(run_dir("shapes3d/coverage_control/val")))
    ap.add_argument("--learned", type=int, default=1)
    a = ap.parse_args(argv)
    seed = int(a.seed)
    leg = Path(a.out) / ("seed%d_K%d" % (seed, a.K))
    leg.mkdir(parents=True, exist_ok=True)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = seed
    cache = D7.assert_generation2(seed)
    log("nclu.__file__ = %s" % nclu.__file__)
    log("seed=%d K=%d cache gen %s scope %s" % (seed, a.K, cache["generation"], cache["scope"]))
    C = build_conditions()
    conds, audit = (C["cond"], C["audit"])
    grid = conds[3]["grid"]
    pc = {}
    for k in RUNGS:
        src = EARLIER_LADDER_RUNS / ("%d_seed%d" % (k, seed)) / "partitions.npz"
        if src.exists():
            ref = np.asarray(np.load(src, allow_pickle=True)["pool"], np.int64)
            mine = fw.fit_pool(
                C["test_train"][k], fw.PINS["n_pool"], fw.PINS["pool_rng_seed"], exclude=None
            )
            pc[str(k)] = bool(np.array_equal(ref, mine))
        else:
            pc[str(k)] = "reference absent"
    audit["pool_recompute_equals_test_pool"] = pc
    log("pool recompute == test pool: %s" % pc)
    if any((v is False for v in pc.values())):
        raise SystemExit("pool recomputation does NOT reproduce the ladder's pool -- refusing")
    judge = fw.Judge(seed, log=log)
    fast = FastJudge(judge)
    masks = {
        k: (
            fw.value_presence_bitmap(grid, conds[k]["train_rows"]),
            fw.seen_tuple_mask(grid, conds[k]["train_rows"]),
        )
        for k in RUNGS
    }
    meta = {
        "VAL_LEG_ONLY": True,
        "scored_target_composition": "DECLARED EXHAUSTED",
        "seed": seed,
        "K": int(a.K),
        "package_file": str(nclu.__file__),
        "cache": cache,
        "audit": audit,
        "stores": {},
        "cells": {},
        "failed": [],
    }

    def run(name, obj, k_est, extra):
        done = leg / ("rows_%s.json" % name)
        if done.exists():
            with open(done, encoding="utf-8") as fh:
                meta["cells"][name] = json.load(fh)["summary"]
            log("  %s: on disk" % name)
            return
        t = time.time()
        try:
            bm, sm = masks[k_est]
            per = LEG.score_arm(obj, conds[k_est], fast, bm, sm)
            s = KS.summarise(
                name, per, dict(extra, estimand_rung=k_est, seconds=round(time.time() - t, 1))
            )
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": s, "per_corner": per}, fh, indent=1, default=str)
            meta["cells"][name] = s
            log(
                "  %-44s novel %.4f judge_acc %.4f (%.0fs)"
                % (
                    name,
                    s["distinct_valid_novel_tuples_mean"],
                    s["judge_accuracy_mean"],
                    time.time() - t,
                )
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            meta["failed"].append(name)
            log("  !!!! %s FAILED %s: %s" % (name, type(exc).__name__, exc))

    for kp in RUNGS:
        sub = leg / ("pool%d" % kp)
        sub.mkdir(exist_ok=True)
        pool, dead = pool_and_dead(conds[kp]["train_rows"], seed)
        np.savez(sub / "pool.npz", pool=pool, dead=dead)
        if kp == 3:
            _chk = assert_identical(judge, fast, fw.gather(fw.features_mmap(seed), pool[:256]))
            meta["fastjudge_positive_control"] = _chk
            log("FastJudge positive control: %s" % _chk["identical"])
            if not _chk["identical"]:
                raise SystemExit("FastJudge not identical -- refusing")
        st, basis, sinfo = KS.build_cov_store(seed, int(a.K), pool, dead)
        addr, addr_s = KS.address_book(sub, st, basis, int(a.K), pool)
        led = four_part_footprint(st, int(addr.byte_footprint()["address_bytes"]))
        meta["stores"]["pool%d" % kp] = {
            "store_info": sinfo,
            "footprint_four_part": led,
            "address_fit_seconds": round(addr_s, 1),
            "pool_n": int(pool.size),
            "pool_first8": pool[:8].tolist(),
            "train_rows_drawn_from": int(conds[kp]["train_rows"].size),
        }
        obj = {
            "arm": "cov_tensor_K%d" % a.K,
            "read": "tensor",
            "kind": "span",
            "store": st,
            "basis": basis,
            "address": addr,
            "protos": None,
            "head_bytes": {"head_bytes": 0},
        }
        for ke in RUNGS:
            cell = {(3, 3): "A", (6, 6): "B", (3, 6): "C", (6, 3): "D"}[kp, ke]
            run(
                "%s_store_pool%d_est%d" % (cell, kp, ke),
                obj,
                ke,
                {"cell": cell, "store_pool_rung": kp},
            )
            for cname, kind in KS.CONTROLS:
                mods = perturb_modes(st.modes, kind, seed)
                sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
                if kind == "zero":
                    cobj = dict(obj, store=sh, basis=basis, zeroed=True, arm=cname)
                else:
                    cobj = dict(
                        obj,
                        store=sh,
                        arm=cname,
                        basis=qr_basis(mods, st.deployed_rank, widths=[b.shape[1] for b in basis]),
                    )
                run(
                    "%s_store_pool%d_est%d__%s" % (cell, kp, ke, cname),
                    cobj,
                    ke,
                    {"cell": cell, "control": kind, "store_pool_rung": kp},
                )
        if a.learned:
            tgt = int(led["honest_total_bytes"] // 4)
            f = ONEX.fit_one(tgt, pool, grid, seed, "learned_1x_pool%d" % kp)
            lobj = {
                "arm": "learned_1x",
                "read": "rival",
                "kind": "rival",
                "rival": f["fn"],
                "rival_partial": None,
                "store": None,
                "protos": None,
                "byte_footprint": {"param_count": f["param_count"]},
                "head_bytes": {"head_bytes": 0},
            }
            for ke in RUNGS:
                cell = {(3, 3): "A", (6, 6): "B", (3, 6): "C", (6, 3): "D"}[kp, ke]
                run(
                    "%s_learned1x_pool%d_est%d" % (cell, kp, ke),
                    lobj,
                    ke,
                    {
                        "cell": cell,
                        "param_count": f["param_count"],
                        "target_params": tgt,
                        "store_pool_rung": kp,
                    },
                )
        del st, basis, addr, obj
    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s failed=%s" % (leg, meta["failed"]))
    return 3 if meta["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
