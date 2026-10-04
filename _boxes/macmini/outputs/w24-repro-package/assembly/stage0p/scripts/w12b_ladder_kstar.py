"""w12b hub/nr6-and-kstar -- THE K* REPLICATION OF THE COMPOSITION LADDER.

⛔ FREE.  A REPLICATION.  No ledger row, no scored budget, no new design, no moved bar.

WHY THIS EXISTS
---------------
The n-way ladder (`w12 hub/composition-ladder`) ran its store arm at **K = 5**, because
`CROSSOVER-val.json` still read `K_star: null` when that hub started.  Its opponent
`learned_1x` was therefore byte-matched to the K = 5 object's own measured total.  The
Head's ruling (`HEAD-RULING-K.md`, 2026-09-21T12:38:07Z) is that the main text carries
**ONE K -- K*** and the K = 5 ladder goes to the appendix as the sweep's endpoint.
⛔ So the ladder's published verdict is measured against an opponent the main text will
never use, and the sign is NOT assumed to carry: it is re-measured here.

WHAT IS HELD FIXED -- everything except K
-----------------------------------------
* the rungs, the corners and the condition swap: `w12b_ladder_leg.make_condition`, a copy
  of the ladder's own module whose ONLY edit is the worktree name in its import assertion;
* the fit pool and the dead-channel mask: the ladder leg's OWN `partitions.npz`, copied in
  byte-for-byte, so the two K's see the identical rows;
* the estimand: `distinct_valid_novel_tuples` per corner via `fw.per_corner_scores`, the
  harness's pre-declared metric (`LADDER-READY.json:estimand`);
* the judge: `fw.Judge` + `FastJudge`, asserted element-for-element identical on real
  features before any number is produced;
* the learner: `w12b_ladder_1x.fit_one` -- the ladder's own `mlp`, its own epochs / lr /
  batch, matched by `w4_learners.match_width`, fitted on the rung's OWN pool.

⚠⚠ THE ONE CONFOUND, DECLARED HERE AND NOT BURIED
-------------------------------------------------
The ladder's `single_design_K5` is a `DowndatableStore` on the **label-MI `factor`**
partition.  K* is defined by the crossover hub's frozen selection rule on the **label-free
`ward_balanced`** cut of a `CovContentStore`.  Going from the ladder's K = 5 arm to the K*
arm therefore changes TWO things at once -- the well count AND the construction (partition
rule, store class, covariance convention).

⭐ SO THIS RUNNER ALSO BUILDS A **BRIDGE** ARM: the SAME `CovContentStore` /
`ward_balanced` construction at **K = 5**, on the same rung and the same pool (`--K 5`).

    K* arm   vs  bridge arm                        = the effect of K alone
    bridge arm  vs  the ladder's single_design_K5  = the effect of the construction alone

⛔ Without the bridge a sign change could not be attributed, and this hub would have
replaced one confounded number with another.

THE CONTROLS
------------
`landscape_removed`, `landscape_shuffled` and `random_atoms` are run in the **crossover
hub's** form: the perturbation is applied AFTER the address book is fitted and the control
is read through a basis of its OWN atoms at the book's own widths (`qr_basis`).  That is
the form the reused PREREG's control gate names, and it is the gate the task file names.
⛔ The ladder's STRONGER `random_atoms` (random atoms with the address book REFIT against
them) is a **DECLARED NOT-RUN** here: at K* the book costs roughly an hour of
single-thread CPU per leg and refitting it per control per leg was not affordable beside
the scored runs.  The weaker form is the conservative direction to report only if it
collapses anyway -- and if it does not collapse, no row is claimed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chlu

_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in Path(chlu.__file__).resolve().parents:
    raise SystemExit("chlu resolves to %s, OUTSIDE %s -- refusing to continue."
                     % (chlu.__file__, _ROOT))

from chlu.core.gen_read import TensorAddress                       # noqa: E402
from chlu.core.gen_store import channel_correlation                # noqa: E402
from chlu.experiments import exp_gen_delete_row7 as D7             # noqa: E402
from chlu.experiments import exp_gen_factor_wells as fw            # noqa: E402
from chlu.experiments import exp_gen_read as R                     # noqa: E402
from chlu.experiments import exp_w5_compose_field as W5            # noqa: E402
from chlu.experiments.exp_w8_single_design import CovContentStore  # noqa: E402

import w12b_ladder_leg as LEG                                      # noqa: E402
import w12b_ladder_1x as ONEX                                      # noqa: E402
from w12_crossover_lib import (                                    # noqa: E402
    D, ShimStore, atoms_for, four_part_ledger, perturb_modes, pinned_basis, qr_basis,
    wells_for,
)
from w12_fastjudge import FastJudge, assert_identical              # noqa: E402

T0 = time.time()
LADDER_RUNS = Path("C:/chlu/outputs/w12-composition-ladder/runs")   # READ ONLY

#: the crossover hub's control gate, in its own form and its own order
CONTROLS = (("landscape_removed", "zero"),
            ("landscape_shuffled", "value_permute"),
            ("random_atoms", "random_span"))


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


def seed_partitions(leg: Path, rung: str, seed: int) -> str:
    """The ladder leg's OWN pool and dead mask, copied in so the two K's see one pool."""
    src = LADDER_RUNS / ("%s_seed%d" % (rung, seed))
    leg.mkdir(parents=True, exist_ok=True)
    got = []
    for name in ("partitions.npz", "partitions.json"):
        s, d = src / name, leg / name
        if d.exists():
            got.append("%s already here" % name)
        elif s.exists():
            shutil.copyfile(s, d)
            got.append("%s copied from the w12 ladder leg" % name)
        else:
            got.append("%s ABSENT at source -- it will be rebuilt here" % name)
    return "; ".join(got)


def build_cov_store(seed: int, K: int, pool: np.ndarray, dead: np.ndarray):
    """The crossover hub's object, at K, on THIS rung's pool.  Nothing new is invented."""
    t = time.time()
    F = fw.features_mmap(int(seed))
    Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
    corr = channel_correlation(Xc)
    del Xc
    X = np.asarray(fw.gather(F, pool), np.float32)
    wc = wells_for(corr, int(K), dead)
    sizes = [int(c.size) for c in wc]
    A = atoms_for(sizes)
    st = CovContentStore(X, wc, A, D)
    del X
    basis = pinned_basis(st)
    rw = [round(int(b.shape[1]) / s, 4) for b, s in zip(basis, sizes)]
    secs = time.time() - t
    log("  store K=%d A=%d built in %.1fs  wells %d..%d  RANK/WIDTH %.4f..%.4f"
        % (K, A, secs, min(sizes), max(sizes), min(rw), max(rw)))
    return st, basis, {"K": int(K), "A_requested": int(A), "well_sizes": sizes,
                       "deployed_rank": [int(r) for r in st.deployed_rank],
                       "rank_over_width": rw,
                       "rank_over_width_max": float(max(rw)),
                       "build_seconds": round(secs, 1)}


def address_book(leg: Path, st, basis, K: int, pool: np.ndarray):
    """`R.READ_PINS` unchanged -- rank 128, ridge 1e-2, 40 sweeps, the whole pool."""
    cache = leg / ("address_covstore_K%d_R%d_sw%d.pkl"
                   % (K, int(R.READ_PINS["rank"]), int(R.READ_PINS["sweeps"])))
    if cache.exists():
        with open(cache, "rb") as fh:
            log("  address book: reused %s" % cache.name)
            return pickle.load(fh), 0.0
    grid, _ = fw.build_grid()
    t = time.time()
    Y, T = R._pool_span_targets(st, pool, basis, log=lambda *_: None)
    ntr = int(0.9 * Y.shape[0])
    addr = TensorAddress([int(c) for c in grid.cardinalities],
                         rank=int(R.READ_PINS["rank"]),
                         ridge=float(R.READ_PINS["ridge"]),
                         seed=int(R.READ_PINS["address_seed"]))
    addr.multiplicative = True
    addr.fit(Y[:ntr], [t_[:ntr] for t_ in T], sweeps=int(R.READ_PINS["sweeps"]),
             log=lambda *_: None, Y_val=Y[ntr:], targets_val=[t_[ntr:] for t_ in T])
    with open(cache, "wb") as fh:
        pickle.dump(addr, fh)
    secs = time.time() - t
    log("  address book K=%d fitted in %.1fs" % (K, secs))
    return addr, secs


def summarise(arm: str, per: list, extra: dict) -> dict:
    dv = np.array([r["distinct_valid_novel_tuples"] for r in per], float)
    ja = np.array([r["judge_accuracy"] for r in per], float)
    s = {"arm": arm, "n_corners": len(per),
         "distinct_valid_novel_tuples_mean": float(dv.mean()),
         "distinct_valid_novel_tuples_sd": (float(dv.std(ddof=1)) if dv.size > 1
                                            else None),
         "judge_accuracy_mean": float(ja.mean())}
    s.update(extra)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, help="2|3|4|5|6|shipped")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=32, help="32 = K*; 5 = the BRIDGE arm")
    ap.add_argument("--out", default="C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")
    ap.add_argument("--arms", default="store,controls")
    a = ap.parse_args(argv)

    leg = Path(a.out) / ("%s_seed%d_K%d" % (a.rung, a.seed, a.K))
    leg.mkdir(parents=True, exist_ok=True)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = int(a.seed)
    cache = D7.assert_generation2(int(a.seed))
    log("chlu.__file__ = %s" % chlu.__file__)
    log("rung=%s seed=%d K=%d  cache gen %s scope %s root %s"
        % (a.rung, a.seed, a.K, cache["generation"], cache["scope"], cache["root"]))
    log("partitions: %s" % seed_partitions(leg, a.rung, int(a.seed)))

    cond = LEG.make_condition(a.rung)
    LEG._COND["cond"] = cond
    fw.load_condition = LEG._patched_load_condition
    W5.fw.load_condition = LEG._patched_load_condition
    R.fw.load_condition = LEG._patched_load_condition
    ONEX.fw.load_condition = LEG._patched_load_condition
    grid = cond["grid"]
    corners = cond["corners"]
    chk = fw.assert_corners_disjoint_and_heldout(corners, cond["test_rows"], a.rung)
    log("corners: %s" % chk)

    fw.build_partitions(leg, log=log)
    z = dict(np.load(leg / "partitions.npz", allow_pickle=True))
    pool = np.asarray(z["pool"], np.int64)
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])

    judge = fw.Judge(int(a.seed), log=log)
    fast = FastJudge(judge)
    _chk = assert_identical(judge, fast,
                            fw.gather(fw.features_mmap(int(a.seed)), pool[:256]))
    log("FastJudge positive control: %s (%d requests, %d elements)"
        % (_chk["identical"], _chk["n_requests"], _chk["n_elements_compared"]))
    if not _chk["identical"]:
        raise SystemExit("FastJudge is NOT identical to the shipped Judge -- refusing.")

    meta = {"REPLICATION_NOT_SCORED": True, "rung": a.rung, "seed": int(a.seed),
            "K": int(a.K), "chlu_file": str(chlu.__file__), "cache": cache,
            "corner_check": chk, "n_corners": len(corners),
            "n_train": int(cond["train_rows"].size),
            "n_test": int(cond["test_rows"].size),
            "fastjudge_positive_control": _chk,
            "estimand": {"metric": "distinct_valid_novel_tuples", "unit": "corner",
                         "function": "exp_gen_factor_wells.per_corner_scores",
                         "margin": float(fw.DECORATION_MARGIN_NOVELTY),
                         "source": ("LADDER-READY.json:estimand -- pre-declared by "
                                    "w12/kway-ladder-harness, not restated here")},
            "arms": {}, "failed_arms": []}

    st, basis, sinfo = build_cov_store(int(a.seed), int(a.K), pool, dead)
    meta["store_info"] = sinfo
    addr, addr_s = address_book(leg, st, basis, int(a.K), pool)
    abytes = int(addr.byte_ledger()["address_bytes"])
    led = four_part_ledger(st, abytes)
    led["address_fit_seconds"] = round(addr_s, 1)
    meta["ledger_four_part"] = led
    tgt_params = int(led["honest_total_bytes"] // 4)
    meta["one_x_target"] = {
        "target_params_float32": tgt_params,
        "honest_total_bytes": int(led["honest_total_bytes"]),
        "times_reference_pin": led["times_reference_pin"],
        "composition": ("atoms + covariance/count + address book + coupling/hole, "
                        "summed and NEVER netted"),
        "note": ("this is the CROSSOVER hub's four-part convention -- the one the main "
                 "text's K* object carries. The w12 ladder's own `one_x_target` used a "
                 "DIFFERENT composition (DowndatableStore deployed bytes + FULL-SQUARE "
                 "float64 statistics + book) and the two are NOT comparable line for "
                 "line; both are printed so a reader can see which is which."),
    }
    log("FOUR-PART LEDGER: atoms %d | cov+count %d | address %d | coupling %d = %d B "
        "[%sx the 524,288 B REFERENCE column] -> 1x target %d params"
        % (led["p1_atom_bytes"], led["p2_covariance_plus_count_bytes"], abytes,
           led["p4_coupling_hole_table_bytes"], led["honest_total_bytes"],
           led["times_reference_pin"], tgt_params))

    want = [x for x in a.arms.split(",") if x]
    base_obj = {"arm": "cov_tensor_K%d" % a.K, "read": "tensor", "kind": "span",
                "store": st, "basis": basis, "address": addr, "protos": None,
                "head_bytes": {"head_bytes": 0}}

    def run(arm_name, obj, extra):
        done = leg / ("rows_%s.json" % arm_name)
        if done.exists():
            log("  %s: already on disk, skipping" % arm_name)
            with open(done, encoding="utf-8") as fh:
                meta["arms"][arm_name] = json.load(fh)["summary"]
            return
        t = time.time()
        try:
            per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
            s = summarise(arm_name, per, dict(extra, seconds=round(time.time() - t, 1)))
            with open(done, "w", encoding="utf-8") as fh:
                json.dump({"summary": s, "per_corner": per}, fh, indent=1, default=str)
            meta["arms"][arm_name] = s
            log("  %-34s novel %.4f  judge_acc %.4f  in %.1fs"
                % (arm_name, s["distinct_valid_novel_tuples_mean"],
                   s["judge_accuracy_mean"], time.time() - t))
        except Exception as exc:                                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            meta["failed_arms"].append(arm_name)
            with open(leg / ("FAILED_%s.json" % arm_name), "w", encoding="utf-8") as fh:
                json.dump({"arm": arm_name,
                           "error": "%s: %s" % (type(exc).__name__, exc)}, fh, indent=1)
            log("  !!!! %s FAILED: %s: %s" % (arm_name, type(exc).__name__, exc))

    store_arm = "single_design_K%d" % a.K
    if "store" in want:
        run(store_arm, base_obj,
            {"byte_matching": "1x its own four-part total", "ledger": led,
             "rank_over_width_max": sinfo["rank_over_width_max"],
             "construction": ("CovContentStore on the LABEL-FREE ward_balanced "
                              "correlation cut, A = min(64, min_k d_k - 1) -- the "
                              "crossover hub's object, built on this rung's own pool")})

    if "controls" in want:
        for name, kind in CONTROLS:
            mods = perturb_modes(st.modes, kind, int(a.seed))
            sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
            if kind == "zero":
                cobj = dict(base_obj, store=sh, basis=basis, zeroed=True, arm=name)
            else:
                cobj = dict(base_obj, store=sh, arm=name,
                            basis=qr_basis(mods, st.deployed_rank,
                                           widths=[b.shape[1] for b in basis]))
            run("%s__%s" % (store_arm, name), cobj,
                {"control": kind, "when": "AFTER the address book was fitted",
                 "form": ("the crossover hub's control gate. The ladder's STRONGER "
                          "random_atoms -- random atoms with the address book REFIT "
                          "against them -- is a DECLARED NOT-RUN here: the book costs "
                          "about an hour of single-thread CPU per leg at K*."),
                 "must": ("COLLAPSE. If it does not, the store is decorative here and "
                          "NO row is claimed -- that is the result.")})

    if "learned_1x" in want:
        done = leg / "rows_learned_1x.json"
        if done.exists():
            log("  learned_1x: already on disk, skipping")
            with open(done, encoding="utf-8") as fh:
                meta["arms"]["learned_1x"] = json.load(fh)["summary"]
        else:
            t = time.time()
            try:
                f = ONEX.fit_one(tgt_params, pool, grid, int(a.seed), "learned_1x")
                obj = {"arm": "learned_1x", "read": "rival", "kind": "rival",
                       "rival": f["fn"], "rival_partial": None, "store": None,
                       "protos": None,
                       "byte_ledger": {"param_count": f["param_count"],
                                       "deployed_bytes": f["param_count"] * 4},
                       "head_bytes": {"head_bytes": 0}}
                per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
                s = summarise("learned_1x", per, {
                    "param_count": f["param_count"], "width": f["width"],
                    "param_err_frac": f["param_err_frac"],
                    "target_params": f["target_params"],
                    "n_fit_rows": f["n_fit_rows"],
                    "population": "this rung's OWN fit pool (NOT montero_combination)",
                    "byte_matching": ("1x the K=%d object's own four-part total (%d B)"
                                      % (a.K, led["honest_total_bytes"])),
                    "train_info": f["train_info"],
                    "seconds": round(time.time() - t, 1)})
                with open(done, "w", encoding="utf-8") as fh:
                    json.dump({"summary": s, "per_corner": per}, fh, indent=1,
                              default=str)
                meta["arms"]["learned_1x"] = s
                log("  %-34s novel %.4f  (%d params, width %d)  in %.1fs"
                    % ("learned_1x", s["distinct_valid_novel_tuples_mean"],
                       f["param_count"], f["width"], time.time() - t))
            except Exception as exc:                                # noqa: BLE001
                import traceback
                traceback.print_exc()
                meta["failed_arms"].append("learned_1x")
                with open(leg / "FAILED_learned_1x.json", "w", encoding="utf-8") as fh:
                    json.dump({"arm": "learned_1x",
                               "error": "%s: %s" % (type(exc).__name__, exc)}, fh,
                              indent=1)
                log("  !!!! learned_1x FAILED: %s: %s" % (type(exc).__name__, exc))

    # -- THE TWO TOKEN FAMILIES, RE-FITTED POST THE DTYPE REPAIR -------------------
    # ⛔ The ladder's published table carries `rival_transformer` and `rival_lstm` from
    # SPOKE A's wave-4 set, which PRE-DATES the `w4_learners.train` dtype repair. Before
    # that repair `eqx.nn.Embedding` refused to index a float32 array, so neither family
    # could be FITTED at all, and their flat scores (0.489-0.531 and 0.579-0.604 of the
    # per-corner ceiling) may be artefacts of a rival that could not train.
    # ⭐ These arms re-fit both families HERE, at the ladder's own 524,288 B REFERENCE pin
    # (131,072 float32 parameters), on THIS rung's own fit pool.
    # ⚠ DECLARED: spoke A's set was fitted on the `montero_combination` population, not on
    # the rung's pool. The populations DIFFER, so these are not drop-in replacements for
    # that row -- they are a POST-FIX reading of the same families, and the direction of
    # the difference runs AGAINST this hub, because a rival fitted on the rung's own pool
    # is the STRONGER of the two.
    if "rivals_refit" in want:
        import jax
        from chlu.eval.rivals import w12_deletion_baselines as W12B
        cards_r = [int(c) for c in grid.cardinalities]
        key0 = jax.random.PRNGKey(int(a.seed) * 7717 + 20_260_921)
        rows_fit = np.asarray(pool, np.int64)[: int(ONEX.N_FIT_ROWS)]
        Yf = fw.gather(fw.features_mmap(int(a.seed)), rows_fit)
        for fam in ("transformer", "lstm"):
            nm = "rival_%s__refit_postfix" % fam
            done = leg / ("rows_%s.json" % nm)
            if done.exists():
                log("  %s: already on disk, skipping" % nm)
                with open(done, encoding="utf-8") as fh:
                    meta["arms"][nm] = json.load(fh)["summary"]
                continue
            t = time.time()
            try:
                key0, k1 = jax.random.split(key0)
                armspec = W12B.matched_arm(fam, cards_r, D, k1, log=log)
                key0, k2 = jax.random.split(key0)
                res = W12B.fit_as_trained(armspec, grid, rows_fit, Yf, cards_r, k2,
                                          log=log)
                # ⛔ `cache` here is `assert_generation2`'s dict, which carries NO
                # `features_sha256`; `save_learner` requires it. The full manifest is
                # `fw.feature_cache_manifest`.
                W12B.save_comparator(res, int(a.seed), grid,
                                     fw.feature_cache_manifest(int(a.seed)),
                                     root=str(leg / "rivals_refit"), spec_label="")
                rv = W12B.load_comparator(int(a.seed), "%s__as_trained" % fam,
                                          root=str(leg / "rivals_refit"))
                obj = {"arm": nm, "read": "rival", "kind": "rival", "rival": rv,
                       "rival_partial": None, "store": None, "protos": None,
                       "byte_ledger": {"param_count": armspec["param_count"],
                                       "deployed_bytes": armspec["param_count"] * 4,
                                       "budget_bytes": 524288},
                       "head_bytes": {"head_bytes": 0}}
                per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
                s_ = summarise(nm, per, {
                    "param_count": armspec["param_count"],
                    "param_target": armspec["param_target"],
                    "width": armspec["width"], "in_kind": armspec["in_kind"],
                    "byte_matching": "the 524,288 B REFERENCE pin (131,072 params)",
                    "population": ("THIS rung's own fit pool -- ⚠ NOT spoke A's "
                                   "montero_combination, so this is not a drop-in "
                                   "replacement for the ladder's rival_%s row" % fam),
                    "n_fit_rows": int(rows_fit.size),
                    "POST_DATES_THE_DTYPE_FIX": True,
                    "train_info": res.get("fit"), "cost_line": res.get("cost"),
                    "seconds": round(time.time() - t, 1)})
                with open(done, "w", encoding="utf-8") as fh:
                    json.dump({"summary": s_, "per_corner": per}, fh, indent=1,
                              default=str)
                meta["arms"][nm] = s_
                log("  %-34s novel %.4f  (%d params)  in %.1fs"
                    % (nm, s_["distinct_valid_novel_tuples_mean"],
                       armspec["param_count"], time.time() - t))
            except Exception as exc:                                # noqa: BLE001
                import traceback
                traceback.print_exc()
                meta["failed_arms"].append(nm)
                with open(leg / ("FAILED_%s.json" % nm), "w", encoding="utf-8") as fh:
                    json.dump({"arm": nm,
                               "error": "%s: %s" % (type(exc).__name__, exc)}, fh,
                              indent=1)
                log("  !!!! %s FAILED: %s: %s" % (nm, type(exc).__name__, exc))
        del Yf

    if store_arm in meta["arms"] and "learned_1x" in meta["arms"]:
        m = (meta["arms"][store_arm]["distinct_valid_novel_tuples_mean"]
             - meta["arms"]["learned_1x"]["distinct_valid_novel_tuples_mean"])
        meta["margin_store_minus_learned_1x"] = float(m)
        log("  MARGIN store - learned_1x = %+.4f" % m)

    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s  failed=%s" % (leg, meta["failed_arms"]))
    return 3 if meta["failed_arms"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
