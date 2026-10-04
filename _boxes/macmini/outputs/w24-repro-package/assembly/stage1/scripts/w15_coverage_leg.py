"""w15 spoke/coverage-control -- IS THE LADDER'S n=3 -> n=6 RISE DEPTH, OR COVERAGE?

⛔ VAL LEG ONLY.  The `composition` scored target is DECLARED EXHAUSTED (see the spoke's
LOG.md / FINDINGS.md): no test-split number of a new configuration is produced here.

THE CONTROL (w12-kway-ladder-harness FINDINGS §4.3)
---------------------------------------------------
Re-run n = 6 with its fit pool restricted to n = 3's train index -- same probe, n = 3's
coverage.  Because ``fw.fit_pool`` is ONE ordered draw ``rng(pool_rng_seed)`` from the
train index and ``pool_rng_seed`` is a single pin, "n = 6 with n = 3's train index" draws
EXACTLY n = 3's pool, hence n = 3's store and address book bit for bit.  What remains to
separate it from n = 3 is the ESTIMAND's train-derived masks (``seen_tuple_mask``,
``value_presence_bitmap``).  So per seed this runner fits TWO stores and reads a 2 x 2:

                         estimand n=3 (masks of train_3)   estimand n=6 (masks of train_6)
    store on pool(train_3)        A  = rung n=3                 C  = n=6 at n=3's coverage
    store on pool(train_6)        D  (diagnostic)               B  = rung n=6

    B - C = the STORE-COVERAGE effect at fixed n=6 estimand
    C - A = the SCORING-MASK effect at fixed store

THE VAL GEOMETRY (no val set existed for the ladder -- the full sweep uses every option)
-------------------------------------------------------------------------------------
The ladder's own builder (``chlu.data.gen_kway_ladder``) with ONE option set swapped:
``scale`` = the MIDDLE value 0.964 (index 3; AMENDMENT 1), a scale value no test cell or corner
touches.  Everything else -- hue bands, shapes, floorwall bands, orientation corners, the
nesting and tiling -- is the builder's own.  Val rung-n condition:
    train  = test-rung-n train  \\ val-rung-n cells
    held   = test-rung-n test   U  val-rung-n cells
Every val corner row is asserted to lie in the TRAIN index of BOTH test rungs (never a
test row).  Corners: 60 x 24 rows (test: 120 x 72) -- DECLARED.

Store = the main-text K* = 32 ``CovContentStore`` on the label-free ``ward_balanced`` cut,
built and read by ``w12b_ladder_kstar``'s own functions, unchanged.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chlu
_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in Path(chlu.__file__).resolve().parents:
    raise SystemExit("chlu resolves to %s, OUTSIDE %s -- refusing." % (chlu.__file__, _ROOT))
from chlu.core.gen_store import dead_channels                       # noqa: E402
from chlu.data import gen_kway_ladder as L                          # noqa: E402
from chlu.experiments import exp_gen_delete_row7 as D7              # noqa: E402
from chlu.experiments import exp_gen_factor_wells as fw             # noqa: E402
from chlu.experiments import exp_gen_read as R                      # noqa: E402
from chlu.experiments import exp_w5_compose_field as W5             # noqa: E402
import w12b_ladder_1x as ONEX                                       # noqa: E402
import w12b_ladder_kstar as KS                                      # noqa: E402
import w12b_ladder_leg as LEG                                       # noqa: E402
from w12_crossover_lib import ShimStore, four_part_ledger, perturb_modes, qr_basis  # noqa: E402
from w12_fastjudge import FastJudge, assert_identical               # noqa: E402
T0 = time.time()
W12_RUNS = Path("C:/chlu/outputs/w12-composition-ladder/runs")      # READ ONLY
MID_SCALE_IDX = (3,)
RUNGS = (3, 6)
def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


def _val_options(grid, cfg, pair_cells):
    opts = _ORIG_OPTS(grid, cfg, pair_cells)
    sv = grid.values[grid.index_of("scale")]
    opts["scale"] = [{"label": "m",
                      "values": {"scale": tuple(float(sv[i]) for i in MID_SCALE_IDX)},
                      "meta": {"scale_corner": "middle (VAL)"}}]
    return opts


_ORIG_OPTS = L.coordinate_options
def val_cells_and_corners(grid, cfg, k, pair):
    L.coordinate_options = _val_options
    try:
        cells = L.build_kway_cells(grid, cfg, k, pair_cells=pair)
        corners = L.kway_corners(grid, cfg, k, cells, pair_cells=pair)
    finally:
        L.coordinate_options = _ORIG_OPTS
    return cells, corners


def build_conditions(log=log) -> dict:
    base = fw.load_condition()
    grid = base["grid"]
    cfg = L.KwayLadderConfig()
    from chlu.data.gen_value_holdout import build_pair_cells
    pair = build_pair_cells(grid, cfg.value_cfg)
    out, test_tr, test_te = {}, {}, {}
    for k in RUNGS:
        split, _, _ = L.build_kway_split(grid, cfg, k)
        test_tr[k] = np.asarray(split.train_rows, np.int64)
        test_te[k] = np.asarray(split.test_rows, np.int64)
    audit = {"train3_subset_of_train6": bool(np.isin(test_tr[3], test_tr[6]).all()),
             "test_rung_train_rows": {str(k): int(test_tr[k].size) for k in RUNGS},
             "test_rung_test_rows": {str(k): int(test_te[k].size) for k in RUNGS}}
    corner_rowsets = {}
    for k in RUNGS:
        cells, corners = val_cells_and_corners(grid, cfg, k, pair)
        cell_rows = np.unique(np.concatenate([c.rows(grid) for c in cells]))
        crn = [{"corner_id": "val-" + c.cell_id, "family": "val_kway%d" % k,
                "rows": np.asarray(c.rows(grid), np.int64), "meta": c.meta}
               for c in corners]
        allrows = np.concatenate([c["rows"] for c in crn])
        # ⛔ a val row must be in the TRAIN index of BOTH test rungs, never a test row
        for kk in RUNGS:
            bad = np.intersect1d(allrows, test_te[kk]).size
            if bad:
                raise AssertionError("val rung %d: %d corner rows are TEST rows of rung %d"
                                     % (k, bad, kk))
            bad = np.intersect1d(cell_rows, test_te[kk]).size
            if bad:
                raise AssertionError("val rung %d: %d cell rows are TEST rows of rung %d"
                                     % (k, bad, kk))
        train = np.setdiff1d(test_tr[k], cell_rows)
        held = np.union1d(test_te[k], cell_rows)
        cond = {"grid": grid, "train_rows": train, "test_rows": held,
                "cells": base["cells"], "query_sets": base["query_sets"],
                "scope": base["scope"], "rung": "val_k%d" % k, "corners": crn}
        chk = fw.assert_corners_disjoint_and_heldout(crn, held, "val_k%d" % k)
        # ⛔ no factor value a val corner requests may be ABSENT from the train index --
        # otherwise the acceptance rule refuses the request and the leg measures refusal,
        # not composition (the amendment-1 failure)
        bm = fw.value_presence_bitmap(grid, train)
        fi = grid.factor_indices_of(allrows)
        missing = {grid.names[f]: sorted(set(np.unique(fi[:, f]).tolist())
                                         - set(np.flatnonzero(bm[f]).tolist()))
                   for f in range(fi.shape[1])}
        missing = {f: v for f, v in missing.items() if v}
        if missing:
            raise AssertionError("val rung %d: requested values absent from train: %s"
                                 % (k, missing))
        corner_rowsets[k] = np.sort(allrows)
        out[k] = cond
        audit["val_k%d" % k] = {"n_cells": len(cells), "n_cell_rows": int(cell_rows.size),
                                "n_corners": len(crn),
                                "rows_per_corner": sorted({int(c["rows"].size) for c in crn}),
                                "train_rows": int(train.size), "held_rows": int(held.size),
                                "corner_check": chk,
                                "pool_size_pin_served": bool(train.size >= fw.PINS["n_pool"])}
        log("val k=%d: %d cells (%d rows), %d corners, train %d"
            % (k, len(cells), cell_rows.size, len(crn), train.size))
    same = bool(np.array_equal(corner_rowsets[3], corner_rowsets[6]))
    if not same:
        raise AssertionError("val k=3 and val k=6 do not probe the identical rows")
    audit["val_rungs_probe_identical_rows"] = same
    # same corner ORDER is not required (scores are compared as means and by row-set
    # matched corners); record the mapping for paired reads
    key = lambda c: tuple(np.sort(c["rows"])[:4].tolist()) + (int(c["rows"].size),)
    k3 = {key(c): c["corner_id"] for c in out[3]["corners"]}
    audit["corner_pairing_k6_to_k3"] = {c["corner_id"]: k3[key(c)] for c in out[6]["corners"]}
    return {"cond": out, "audit": audit, "test_train": test_tr}


def pool_and_dead(train: np.ndarray, seed: int):
    """``fw.build_partitions``'s pool and dead mask, by the same two functions."""
    pool = fw.fit_pool(train, fw.PINS["n_pool"], fw.PINS["pool_rng_seed"], exclude=None)
    X = fw.gather(fw.features_mmap(int(seed)), pool[: int(fw.PINS["n_corr"])])
    dead = dead_channels(X, eps=float(fw.PINS["dead_channel_eps"]))
    return pool, np.asarray(dead).ravel().astype(bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--out", default="C:/chlu/outputs/w15-coverage-control/val")
    ap.add_argument("--learned", type=int, default=1)
    a = ap.parse_args(argv)
    seed = int(a.seed)
    leg = Path(a.out) / ("seed%d_K%d" % (seed, a.K))
    leg.mkdir(parents=True, exist_ok=True)
    fw.PINS["condition"] = "test"   # mechanical: no extra fw.val_corners exclusion; the
    fw.PINS["seed"] = seed          # val hold-out is carried in the condition dicts below
    cache = D7.assert_generation2(seed)
    log("chlu.__file__ = %s" % chlu.__file__)
    log("seed=%d K=%d cache gen %s scope %s" % (seed, a.K, cache["generation"], cache["scope"]))

    C = build_conditions()
    conds, audit = C["cond"], C["audit"]
    grid = conds[3]["grid"]

    # ⭐ positive control on the instrument: my pool recomputation == the w12 ladder's
    # own test-rung pool (partitions.npz) for rung 3 and rung 6
    pc = {}
    for k in RUNGS:
        src = W12_RUNS / ("%d_seed%d" % (k, seed)) / "partitions.npz"
        if src.exists():
            ref = np.asarray(np.load(src, allow_pickle=True)["pool"], np.int64)
            mine = fw.fit_pool(C["test_train"][k], fw.PINS["n_pool"],
                               fw.PINS["pool_rng_seed"], exclude=None)
            pc[str(k)] = bool(np.array_equal(ref, mine))
        else:
            pc[str(k)] = "reference absent"
    audit["pool_recompute_equals_w12_test_pool"] = pc
    log("pool recompute == w12 test pool: %s" % pc)
    if any(v is False for v in pc.values()):
        raise SystemExit("pool recomputation does NOT reproduce the ladder's pool -- refusing")

    judge = fw.Judge(seed, log=log)
    fast = FastJudge(judge)
    masks = {k: (fw.value_presence_bitmap(grid, conds[k]["train_rows"]),
                 fw.seen_tuple_mask(grid, conds[k]["train_rows"])) for k in RUNGS}

    meta = {"VAL_LEG_ONLY": True, "scored_target_composition": "DECLARED EXHAUSTED",
            "seed": seed, "K": int(a.K), "chlu_file": str(chlu.__file__), "cache": cache,
            "audit": audit, "stores": {}, "cells": {}, "failed": []}

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
            s = KS.summarise(name, per, dict(extra, estimand_rung=k_est,
                                             seconds=round(time.time() - t, 1)))
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": s, "per_corner": per}, fh, indent=1, default=str)
            meta["cells"][name] = s
            log("  %-44s novel %.4f judge_acc %.4f (%.0fs)"
                % (name, s["distinct_valid_novel_tuples_mean"], s["judge_accuracy_mean"],
                   time.time() - t))
        except Exception as exc:                                     # noqa: BLE001
            import traceback
            traceback.print_exc()
            meta["failed"].append(name)
            log("  !!!! %s FAILED %s: %s" % (name, type(exc).__name__, exc))

    for kp in RUNGS:                      # the store's pool coverage
        sub = leg / ("pool%d" % kp)
        sub.mkdir(exist_ok=True)
        pool, dead = pool_and_dead(conds[kp]["train_rows"], seed)
        np.savez(sub / "pool.npz", pool=pool, dead=dead)
        if kp == 3:
            _chk = assert_identical(judge, fast,
                                    fw.gather(fw.features_mmap(seed), pool[:256]))
            meta["fastjudge_positive_control"] = _chk
            log("FastJudge positive control: %s" % _chk["identical"])
            if not _chk["identical"]:
                raise SystemExit("FastJudge not identical -- refusing")
        st, basis, sinfo = KS.build_cov_store(seed, int(a.K), pool, dead)
        addr, addr_s = KS.address_book(sub, st, basis, int(a.K), pool)
        led = four_part_ledger(st, int(addr.byte_ledger()["address_bytes"]))
        meta["stores"]["pool%d" % kp] = {"store_info": sinfo, "ledger_four_part": led,
                                         "address_fit_seconds": round(addr_s, 1),
                                         "pool_n": int(pool.size),
                                         "pool_first8": pool[:8].tolist(),
                                         "train_rows_drawn_from": int(conds[kp]["train_rows"].size)}
        obj = {"arm": "cov_tensor_K%d" % a.K, "read": "tensor", "kind": "span",
               "store": st, "basis": basis, "address": addr, "protos": None,
               "head_bytes": {"head_bytes": 0}}
        for ke in RUNGS:                  # the estimand's masks
            cell = {(3, 3): "A", (6, 6): "B", (3, 6): "C", (6, 3): "D"}[(kp, ke)]
            run("%s_store_pool%d_est%d" % (cell, kp, ke), obj, ke,
                {"cell": cell, "store_pool_rung": kp})
            # charter-C controls, post-fit form (w12b's gate), on every cell
            for cname, kind in KS.CONTROLS:
                mods = perturb_modes(st.modes, kind, seed)
                sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
                if kind == "zero":
                    cobj = dict(obj, store=sh, basis=basis, zeroed=True, arm=cname)
                else:
                    cobj = dict(obj, store=sh, arm=cname,
                                basis=qr_basis(mods, st.deployed_rank,
                                               widths=[b.shape[1] for b in basis]))
                run("%s_store_pool%d_est%d__%s" % (cell, kp, ke, cname), cobj, ke,
                    {"cell": cell, "control": kind, "store_pool_rung": kp})
        if a.learned:
            tgt = int(led["honest_total_bytes"] // 4)
            f = ONEX.fit_one(tgt, pool, grid, seed, "learned_1x_pool%d" % kp)
            lobj = {"arm": "learned_1x", "read": "rival", "kind": "rival", "rival": f["fn"],
                    "rival_partial": None, "store": None, "protos": None,
                    "byte_ledger": {"param_count": f["param_count"]},
                    "head_bytes": {"head_bytes": 0}}
            for ke in RUNGS:
                cell = {(3, 3): "A", (6, 6): "B", (3, 6): "C", (6, 3): "D"}[(kp, ke)]
                run("%s_learned1x_pool%d_est%d" % (cell, kp, ke), lobj, ke,
                    {"cell": cell, "param_count": f["param_count"],
                     "target_params": tgt, "store_pool_rung": kp})
        del st, basis, addr, obj

    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s failed=%s" % (leg, meta["failed"]))
    return 3 if meta["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
