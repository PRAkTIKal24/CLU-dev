"""w12 hub/composition-ladder -- AMENDMENT 1: the byte-matched learners, fitted by THIS hub.

Two learned arms per leg, both `mlp` (one-hot request -> 2048-d feature), both fitted on
**that rung's own fit pool** so the population caveat that hangs over spoke A's rival set
(`fit_population = montero_combination`) does not hang over these:

  learned_ref_ownpool   131,072 parameters  = the 524,288 B REFERENCE pin
  learned_1x            the MEASURED total bytes of `single_design_K5` on that leg / 4
                        (atoms + sufficient statistics + address book, never netted)

Width comes from `w4_learners.match_width`, which measures the built model rather than a
formula and REFUSES outside its own +-10% band -- a refusal is reported as the result.
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

# ⚠ w12b hub/nr6-and-kstar: the ONLY edit to this file is the worktree name in the
# assertion below.  The w12 ladder asserted "CHLU-w12-composition-ladder"; this copy runs
# in "CHLU-w12b-nr6-and-kstar" and asserts THAT, by the same mechanism and for the same
# reason.  ⛔ Nothing else in this module is touched, so the K* replication drives the
# ladder's own condition swap, its own store builder and its own scorer, unchanged.
from pathlib import Path as _P
if _P(__file__).resolve().parents[1] not in _P(chlu.__file__).resolve().parents:
    raise AssertionError("chlu resolves OUTSIDE this worktree: %s" % chlu.__file__)

from chlu.experiments import exp_gen_delete_row7 as D7
from chlu.experiments import exp_gen_factor_wells as fw
from chlu.experiments import exp_gen_read as R
from chlu.eval.rivals import w4_learners as WL
from w12_fastjudge import FastJudge, assert_identical

import w12b_ladder_leg as LEG  # the condition swap, verbatim

T0 = time.time()


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


REF_PIN_PARAMS = 131072
N_FIT_ROWS = 32768          # = exp_gen_factor_wells.PINS['n_store_train']
EPOCHS = 20
LR = 1e-3
WD = 0.0


def one_x_target(leg: Path) -> dict:
    """The single design's MEASURED total on this leg, in bytes and in float32 params."""
    p = leg / "single_design_store.pkl"
    if not p.exists():
        return {"status": "DECLARED NOT-RUN",
                "reason": "no single_design_store.pkl on this leg -- nothing to match"}
    with open(p, "rb") as fh:
        st = pickle.load(fh)
    b1 = st.byte_ledger(524288)
    b2 = st.covariance_ledger()
    books = sorted(leg.glob("address_single_design_K5_R*.pkl"))
    ab = None
    if books:
        with open(books[0], "rb") as fh:
            ab = int(pickle.load(fh).byte_ledger().get("address_bytes", 0))
    total = int(b1["deployed_bytes"]) + int(b2["total_statistics_bytes"]) + int(ab or 0)
    return {"status": "RUN",
            "p1_atom_bytes": int(b1["deployed_bytes"]),
            "p2_statistics_bytes": int(b2["total_statistics_bytes"]),
            "p2_if_symmetric_triangle_float32": int(b2["if_symmetric_triangle_float32"]),
            "p3_address_book_bytes": ab,
            "p3_status": "measured" if ab is not None else
                         "DECLARED NOT-RUN -- no address book on disk; the total EXCLUDES it",
            "well_sizes": [int(x) for x in st.well_sizes],
            "total_bytes": total,
            "times_reference_pin": round(total / 524288.0, 4),
            "target_params_float32": total // 4,
            "never_netted": True}


def fit_one(target_params: int, pool: np.ndarray, grid, seed: int, tag: str) -> dict:
    import jax
    cards = [int(c) for c in grid.cardinalities]
    build, kind = WL.builders(cards, int(fw.PINS["D"]),
                              jax.random.PRNGKey(int(seed)))["mlp"]
    w, npar, err = WL.match_width(build, int(target_params), lo=1, hi=8192, log=log)
    model = build(w)
    rows = np.asarray(pool, np.int64)[:N_FIT_ROWS]
    Y = fw.gather(fw.features_mmap(int(seed)), rows)
    cls = grid.factor_indices_of(rows)
    Xin = WL.onehot_inputs(cls, cards)
    t = time.time()
    model, info = WL.train(model, Xin, Y, jax.random.PRNGKey(int(seed) + 1),
                           epochs=EPOCHS, lr=LR, wd=WD, batch=WL.BATCH, log=log,
                           rng=np.random.default_rng(int(seed)))
    del Y, Xin
    log("    %s: width %d, %d params (%.2f%% of target), fitted in %.1fs"
        % (tag, w, npar, 100.0 * err, time.time() - t))

    gv = [np.asarray(v, np.float64) for v in grid.values]

    def fn(values: np.ndarray):
        V = np.asarray(values, np.float64)
        idx = np.empty(V.shape, np.int64)
        oor = np.zeros(V.shape[0], bool)
        for k in range(V.shape[1]):
            g = gv[k]
            idx[:, k] = np.argmin(np.abs(V[:, k, None] - g[None, :]), axis=1)
            oor |= (V[:, k] < g[0]) | (V[:, k] > g[-1])
        X = WL.onehot_inputs(idx, cards)
        return WL.predict(model, X), oor

    return {"fn": fn, "width": int(w), "param_count": int(npar),
            "param_err_frac": float(err), "train_info": info,
            "n_fit_rows": int(rows.size), "fit_pool": "this rung's own pool",
            "target_params": int(target_params)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="ref,1x")
    a = ap.parse_args(argv)

    leg = Path(a.out) / ("%s_seed%d" % (a.rung, a.seed))
    if not (leg / "partitions.npz").exists():
        raise SystemExit("leg %s has no partitions.npz -- run w12_ladder_leg.py first" % leg)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = int(a.seed)
    cache = D7.assert_generation2(int(a.seed))
    log("rung=%s seed=%d cache gen %s scope %s" % (a.rung, a.seed, cache["generation"],
                                                   cache["scope"]))
    cond = LEG.make_condition(a.rung)
    LEG._COND["cond"] = cond
    fw.load_condition = LEG._patched_load_condition
    R.fw.load_condition = LEG._patched_load_condition
    grid = cond["grid"]
    pool = np.asarray(np.load(leg / "partitions.npz")["pool"], np.int64)
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    judge = fw.Judge(int(a.seed), log=log)
    fast = FastJudge(judge)
    _chk = assert_identical(judge, fast,
                            fw.gather(fw.features_mmap(int(a.seed)), pool[:256]))
    log("FastJudge positive control: %s (%d requests, %d elements)"
        % (_chk["identical"], _chk["n_requests"], _chk["n_elements_compared"]))

    tgt = one_x_target(leg)
    with open(leg / "ONEX_TARGET.json", "w", encoding="utf-8") as fh:
        json.dump(tgt, fh, indent=1, default=str)
    log("1x target: %s" % json.dumps({k: v for k, v in tgt.items()
                                      if k != "well_sizes"}))

    jobs = []
    if "ref" in a.which:
        jobs.append(("learned_ref_ownpool", REF_PIN_PARAMS,
                     "524,288 B REFERENCE pin, fitted on this rung's OWN pool"))
    if "1x" in a.which:
        if tgt["status"] != "RUN":
            with open(leg / "FAILED_learned_1x.json", "w", encoding="utf-8") as fh:
                json.dump({"arm": "learned_1x", "error": tgt["reason"]}, fh, indent=1)
            log("  learned_1x: DECLARED NOT-RUN -- %s" % tgt["reason"])
        else:
            jobs.append(("learned_1x", int(tgt["target_params_float32"]),
                         "1x the single design's own MEASURED total (%d B = %.2fx the "
                         "524,288 B reference pin)" % (tgt["total_bytes"],
                                                       tgt["times_reference_pin"])))

    for arm, target, matching in jobs:
        done = leg / ("rows_%s.json" % arm)
        if done.exists():
            log("  %s: already on disk, skipping" % arm)
            continue
        t = time.time()
        log("---- %s (target %d params) ----" % (arm, target))
        try:
            f = fit_one(target, pool, grid, int(a.seed), arm)
            obj = {"arm": arm, "read": "rival", "kind": "rival", "rival": f["fn"],
                   "rival_partial": None, "store": None, "protos": None,
                   "byte_ledger": {"param_count": f["param_count"],
                                   "deployed_bytes": f["param_count"] * 4},
                   "head_bytes": {"head_bytes": 0}}
            per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
            dv = np.array([r["distinct_valid_novel_tuples"] for r in per], float)
            ja = np.array([r["judge_accuracy"] for r in per], float)
            summ = {"arm": arm, "n_corners": len(per),
                    "distinct_valid_novel_tuples_mean": float(dv.mean()),
                    "distinct_valid_novel_tuples_sd": float(dv.std(ddof=1)),
                    "judge_accuracy_mean": float(ja.mean()),
                    "judge_accuracy_sd": float(ja.std(ddof=1)),
                    "param_count": f["param_count"], "width": f["width"],
                    "param_err_frac": f["param_err_frac"],
                    "target_params": f["target_params"],
                    "n_fit_rows": f["n_fit_rows"],
                    "population": "this rung's OWN fit pool (NOT montero_combination)",
                    "byte_matching": matching,
                    "train_info": f["train_info"],
                    "fitted_by": "w12 hub/composition-ladder, AMENDMENT-1",
                    "fastjudge_positive_control": _chk,
                    "seconds": time.time() - t}
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": summ, "per_corner": per}, fh, indent=1, default=str)
            log("  %s: novel %.4f +- %.4f  judge_acc %.4f  in %.1fs"
                % (arm, dv.mean(), dv.std(ddof=1), ja.mean(), time.time() - t))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with open(leg / ("FAILED_%s.json" % arm), "w", encoding="utf-8") as fh:
                json.dump({"arm": arm, "error": "%s: %s" % (type(exc).__name__, exc)},
                          fh, indent=1)
            log("  !!!! %s FAILED: %s: %s" % (arm, type(exc).__name__, exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
