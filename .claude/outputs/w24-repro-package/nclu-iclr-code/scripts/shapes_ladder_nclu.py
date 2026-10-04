"""The K* n-CLU on one ladder rung (fit pool and dead-channel mask of that rung), with its controls.

python scripts/shapes_ladder_nclu.py --rung 3 --seed 0 --arms store,controls
"""

from __future__ import annotations
from nclu.paths import run_dir
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
import nclu

_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in Path(nclu.__file__).resolve().parents:
    raise SystemExit(
        "nclu resolves to %s, OUTSIDE %s -- refusing to continue." % (nclu.__file__, _ROOT)
    )
from nclu.core.gen_read import TensorAddress
from nclu.core.gen_store import channel_correlation
from nclu.experiments import gen_deletion as D7
from nclu.experiments import factor_wells as fw
from nclu.experiments import exp_gen_read as R
from nclu.experiments import composition_cell as CC
from nclu.experiments.covariance_store import CovContentStore
import shapes_ladder_lib as LEG
import shapes_ladder_learned as ONEX
from shapes_nclu_lib import (
    D,
    ShimStore,
    atoms_for,
    four_part_footprint,
    perturb_modes,
    pinned_basis,
    qr_basis,
    wells_for,
)
from shapes_fast_judge import FastJudge, assert_identical

T0 = time.time()
LADDER_RUNS = Path(str(run_dir("shapes3d/ladder_earlier")))
CONTROLS = (
    ("landscape_removed", "zero"),
    ("landscape_shuffled", "value_permute"),
    ("random_atoms", "random_span"),
)


def log(m=""):
    print("[%8.1fs] %s" % (time.time() - T0, m), flush=True)


def seed_partitions(leg: Path, rung: str, seed: int) -> str:
    """The ladder leg's OWN pool and dead mask, copied in so the two K's see one pool."""
    src = LADDER_RUNS / ("%s_seed%d" % (rung, seed))
    leg.mkdir(parents=True, exist_ok=True)
    got = []
    for name in ("partitions.npz", "partitions.json"):
        s, d = (src / name, leg / name)
        if d.exists():
            got.append("%s already here" % name)
        elif s.exists():
            shutil.copyfile(s, d)
            got.append("%s copied from the ladder leg" % name)
        else:
            got.append("%s ABSENT at source -- it will be rebuilt here" % name)
    return "; ".join(got)


def build_cov_store(seed: int, K: int, pool: np.ndarray, dead: np.ndarray):
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
    log(
        "  store K=%d A=%d built in %.1fs  wells %d..%d  RANK/WIDTH %.4f..%.4f"
        % (K, A, secs, min(sizes), max(sizes), min(rw), max(rw))
    )
    return (
        st,
        basis,
        {
            "K": int(K),
            "A_requested": int(A),
            "well_sizes": sizes,
            "deployed_rank": [int(r) for r in st.deployed_rank],
            "rank_over_width": rw,
            "rank_over_width_max": float(max(rw)),
            "build_seconds": round(secs, 1),
        },
    )


def address_book(leg: Path, st, basis, K: int, pool: np.ndarray):
    cache = leg / (
        "address_covstore_K%d_R%d_sw%d.pkl"
        % (K, int(R.READ_PINS["rank"]), int(R.READ_PINS["sweeps"]))
    )
    if cache.exists():
        with open(cache, "rb") as fh:
            log("  address book: reused %s" % cache.name)
            return (pickle.load(fh), 0.0)
    grid, _ = fw.build_grid()
    t = time.time()
    Y, T = R._pool_span_targets(st, pool, basis, log=lambda *_: None)
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
        log=lambda *_: None,
        Y_val=Y[ntr:],
        targets_val=[t_[ntr:] for t_ in T],
    )
    with open(cache, "wb") as fh:
        pickle.dump(addr, fh)
    secs = time.time() - t
    log("  address book K=%d fitted in %.1fs" % (K, secs))
    return (addr, secs)


def summarise(arm: str, per: list, extra: dict) -> dict:
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
    ap.add_argument("--K", type=int, default=32, help="32 = K*; 5 = the BRIDGE arm")
    ap.add_argument("--out", default=str(run_dir("shapes3d/ladder_nclu")))
    ap.add_argument("--arms", default="store,controls")
    a = ap.parse_args(argv)
    leg = Path(a.out) / ("%s_seed%d_K%d" % (a.rung, a.seed, a.K))
    leg.mkdir(parents=True, exist_ok=True)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = int(a.seed)
    cache = D7.assert_generation2(int(a.seed))
    log("nclu.__file__ = %s" % nclu.__file__)
    log(
        "rung=%s seed=%d K=%d  cache gen %s scope %s root %s"
        % (a.rung, a.seed, a.K, cache["generation"], cache["scope"], cache["root"])
    )
    log("partitions: %s" % seed_partitions(leg, a.rung, int(a.seed)))
    cond = LEG.make_condition(a.rung)
    LEG._COND["cond"] = cond
    fw.load_condition = LEG._patched_load_condition
    CC.fw.load_condition = LEG._patched_load_condition
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
    _chk = assert_identical(judge, fast, fw.gather(fw.features_mmap(int(a.seed)), pool[:256]))
    log(
        "FastJudge positive control: %s (%d requests, %d elements)"
        % (_chk["identical"], _chk["n_requests"], _chk["n_elements_compared"])
    )
    if not _chk["identical"]:
        raise SystemExit("FastJudge is NOT identical to the shipped Judge -- refusing.")
    meta = {
        "REPLICATION_NOT_SCORED": True,
        "rung": a.rung,
        "seed": int(a.seed),
        "K": int(a.K),
        "package_file": str(nclu.__file__),
        "cache": cache,
        "corner_check": chk,
        "n_corners": len(corners),
        "n_train": int(cond["train_rows"].size),
        "n_test": int(cond["test_rows"].size),
        "fastjudge_positive_control": _chk,
        "estimand": {
            "metric": "distinct_valid_novel_tuples",
            "unit": "corner",
            "function": "exp_gen_factor_wells.per_corner_scores",
            "margin": float(fw.DECORATION_MARGIN_NOVELTY),
            "source": "LADDER-READY.json:estimand -- pre-declared by kway-ladder-harness, not restated here",
        },
        "arms": {},
        "failed_arms": [],
    }
    st, basis, sinfo = build_cov_store(int(a.seed), int(a.K), pool, dead)
    meta["store_info"] = sinfo
    addr, addr_s = address_book(leg, st, basis, int(a.K), pool)
    abytes = int(addr.byte_footprint()["address_bytes"])
    led = four_part_footprint(st, abytes)
    led["address_fit_seconds"] = round(addr_s, 1)
    meta["footprint_four_part"] = led
    tgt_params = int(led["honest_total_bytes"] // 4)
    meta["one_x_target"] = {
        "target_params_float32": tgt_params,
        "honest_total_bytes": int(led["honest_total_bytes"]),
        "times_reference_pin": led["times_reference_pin"],
        "composition": "atoms + covariance/count + address book + coupling/hole, summed and NEVER netted",
        "note": "this is the CROSSOVER study's four-part convention -- the one the main text's K* object carries. The ladder's own `one_x_target` used a DIFFERENT composition (DowndatableStore deployed bytes + FULL-SQUARE float64 statistics + book) and the two are NOT comparable line for line; both are printed so a reader can see which is which.",
    }
    log(
        "FOUR-PART FOOTPRINT: atoms %d | cov+count %d | address %d | coupling %d = %d B [%sx the 524,288 B REFERENCE column] -> 1x target %d params"
        % (
            led["p1_atom_bytes"],
            led["p2_covariance_plus_count_bytes"],
            abytes,
            led["p4_coupling_hole_table_bytes"],
            led["honest_total_bytes"],
            led["times_reference_pin"],
            tgt_params,
        )
    )
    want = [x for x in a.arms.split(",") if x]
    base_obj = {
        "arm": "cov_tensor_K%d" % a.K,
        "read": "tensor",
        "kind": "span",
        "store": st,
        "basis": basis,
        "address": addr,
        "protos": None,
        "head_bytes": {"head_bytes": 0},
    }

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
            log(
                "  %-34s novel %.4f  judge_acc %.4f  in %.1fs"
                % (
                    arm_name,
                    s["distinct_valid_novel_tuples_mean"],
                    s["judge_accuracy_mean"],
                    time.time() - t,
                )
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            meta["failed_arms"].append(arm_name)
            with open(leg / ("FAILED_%s.json" % arm_name), "w", encoding="utf-8") as fh:
                json.dump(
                    {"arm": arm_name, "error": "%s: %s" % (type(exc).__name__, exc)}, fh, indent=1
                )
            log("  !!!! %s FAILED: %s: %s" % (arm_name, type(exc).__name__, exc))

    store_arm = "single_design_K%d" % a.K
    if "store" in want:
        run(
            store_arm,
            base_obj,
            {
                "byte_matching": "1x its own four-part total",
                "footprint": led,
                "rank_over_width_max": sinfo["rank_over_width_max"],
                "construction": "CovContentStore on the LABEL-FREE ward_balanced correlation cut, A = min(64, min_k d_k - 1) -- the crossover study's object, built on this rung's own pool",
            },
        )
    if "controls" in want:
        for name, kind in CONTROLS:
            mods = perturb_modes(st.modes, kind, int(a.seed))
            sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
            if kind == "zero":
                cobj = dict(base_obj, store=sh, basis=basis, zeroed=True, arm=name)
            else:
                cobj = dict(
                    base_obj,
                    store=sh,
                    arm=name,
                    basis=qr_basis(mods, st.deployed_rank, widths=[b.shape[1] for b in basis]),
                )
            run(
                "%s__%s" % (store_arm, name),
                cobj,
                {
                    "control": kind,
                    "when": "AFTER the address book was fitted",
                    "form": "the crossover study's control gate. The ladder's STRONGER random_atoms -- random atoms with the address book REFIT against them -- is a DECLARED NOT-RUN here: the book costs about an hour of single-thread CPU per leg at K*.",
                    "must": "COLLAPSE. If it does not, the store is decorative here and NO row is claimed -- that is the result.",
                },
            )
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
                obj = {
                    "arm": "learned_1x",
                    "read": "rival",
                    "kind": "rival",
                    "rival": f["fn"],
                    "rival_partial": None,
                    "store": None,
                    "protos": None,
                    "byte_footprint": {
                        "param_count": f["param_count"],
                        "deployed_bytes": f["param_count"] * 4,
                    },
                    "head_bytes": {"head_bytes": 0},
                }
                per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
                s = summarise(
                    "learned_1x",
                    per,
                    {
                        "param_count": f["param_count"],
                        "width": f["width"],
                        "param_err_frac": f["param_err_frac"],
                        "target_params": f["target_params"],
                        "n_fit_rows": f["n_fit_rows"],
                        "population": "this rung's OWN fit pool (NOT montero_combination)",
                        "byte_matching": "1x the K=%d object's own four-part total (%d B)"
                        % (a.K, led["honest_total_bytes"]),
                        "train_info": f["train_info"],
                        "seconds": round(time.time() - t, 1),
                    },
                )
                with open(done, "w", encoding="utf-8") as fh:
                    json.dump({"summary": s, "per_corner": per}, fh, indent=1, default=str)
                meta["arms"]["learned_1x"] = s
                log(
                    "  %-34s novel %.4f  (%d params, width %d)  in %.1fs"
                    % (
                        "learned_1x",
                        s["distinct_valid_novel_tuples_mean"],
                        f["param_count"],
                        f["width"],
                        time.time() - t,
                    )
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                meta["failed_arms"].append("learned_1x")
                with open(leg / "FAILED_learned_1x.json", "w", encoding="utf-8") as fh:
                    json.dump(
                        {"arm": "learned_1x", "error": "%s: %s" % (type(exc).__name__, exc)},
                        fh,
                        indent=1,
                    )
                log("  !!!! learned_1x FAILED: %s: %s" % (type(exc).__name__, exc))
    if "rivals_refit" in want:
        import jax
        from nclu.eval.baselines import refusal_comparator as RCB

        cards_r = [int(c) for c in grid.cardinalities]
        key0 = jax.random.PRNGKey(int(a.seed) * 7717 + 20260921)
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
                armspec = RCB.matched_arm(fam, cards_r, D, k1, log=log)
                key0, k2 = jax.random.split(key0)
                res = RCB.fit_as_trained(armspec, grid, rows_fit, Yf, cards_r, k2, log=log)
                RCB.save_comparator(
                    res,
                    int(a.seed),
                    grid,
                    fw.feature_cache_manifest(int(a.seed)),
                    root=str(leg / "rivals_refit"),
                    spec_label="",
                )
                rv = RCB.load_comparator(
                    int(a.seed), "%s__as_trained" % fam, root=str(leg / "rivals_refit")
                )
                obj = {
                    "arm": nm,
                    "read": "rival",
                    "kind": "rival",
                    "rival": rv,
                    "rival_partial": None,
                    "store": None,
                    "protos": None,
                    "byte_footprint": {
                        "param_count": armspec["param_count"],
                        "deployed_bytes": armspec["param_count"] * 4,
                        "budget_bytes": 524288,
                    },
                    "head_bytes": {"head_bytes": 0},
                }
                per = LEG.score_arm(obj, cond, fast, bitmap, seen_mask)
                s_ = summarise(
                    nm,
                    per,
                    {
                        "param_count": armspec["param_count"],
                        "param_target": armspec["param_target"],
                        "width": armspec["width"],
                        "in_kind": armspec["in_kind"],
                        "byte_matching": "the 524,288 B REFERENCE pin (131,072 params)",
                        "population": "THIS rung's own fit pool -- ⚠ NOT the earlier baseline set's montero_combination, so this is not a drop-in replacement for the ladder's rival_%s row"
                        % fam,
                        "n_fit_rows": int(rows_fit.size),
                        "POST_DATES_THE_DTYPE_FIX": True,
                        "train_info": res.get("fit"),
                        "cost_line": res.get("cost"),
                        "seconds": round(time.time() - t, 1),
                    },
                )
                with open(done, "w", encoding="utf-8") as fh:
                    json.dump({"summary": s_, "per_corner": per}, fh, indent=1, default=str)
                meta["arms"][nm] = s_
                log(
                    "  %-34s novel %.4f  (%d params)  in %.1fs"
                    % (
                        nm,
                        s_["distinct_valid_novel_tuples_mean"],
                        armspec["param_count"],
                        time.time() - t,
                    )
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                meta["failed_arms"].append(nm)
                with open(leg / ("FAILED_%s.json" % nm), "w", encoding="utf-8") as fh:
                    json.dump(
                        {"arm": nm, "error": "%s: %s" % (type(exc).__name__, exc)}, fh, indent=1
                    )
                log("  !!!! %s FAILED: %s: %s" % (nm, type(exc).__name__, exc))
        del Yf
    if store_arm in meta["arms"] and "learned_1x" in meta["arms"]:
        m = (
            meta["arms"][store_arm]["distinct_valid_novel_tuples_mean"]
            - meta["arms"]["learned_1x"]["distinct_valid_novel_tuples_mean"]
        )
        meta["margin_store_minus_learned_1x"] = float(m)
        log("  MARGIN store - learned_1x = %+.4f" % m)
    with open(leg / "LEG.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    log("leg done -> %s  failed=%s" % (leg, meta["failed_arms"]))
    return 3 if meta["failed_arms"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
