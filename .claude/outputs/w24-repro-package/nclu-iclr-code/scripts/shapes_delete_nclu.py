"""Deletion on the K*=32 n-CLU (value red, conjunction red sphere, random 1% of items).

    python scripts/shapes_delete_nclu.py --seed 0 [--scope value|conjunction_live|item] [--arm <arm.pkl>]

For each scope: (a) the downdated content against the content refit without the rows, (b) the address
book's statistic and the book refit from it against a book refit without the rows, (c) refusal through
the hole (a display), (d) composition on the surviving combinations before and after, and the downdated
object against the object refit without the rows; with costs. Writes runs/shapes3d/deletion_store/.
"""

from __future__ import annotations
import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes_common import OUT, assert_worktree, jdump, load_pool, log


def kstar_arm(seed: int) -> Path:
    """The K*=32 object built by the build script on the test pool (131,072 rows)."""
    return OUT / "stores" / "test" / ("seed%d" % int(seed)) / "K32_n131072" / "arm.pkl"


def fit_book(cards, Yi, T, Yv, Tv):
    from nclu.core.gen_read import TensorAddress
    from nclu.experiments import exp_gen_read as R

    addr = TensorAddress(
        cards,
        rank=int(R.READ_PINS["rank"]),
        ridge=float(R.READ_PINS["ridge"]),
        seed=int(R.READ_PINS["address_seed"]),
    )
    addr.multiplicative = True
    addr.fit(
        Yi, T, sweeps=int(R.READ_PINS["sweeps"]), log=lambda *_: None, Y_val=Yv, targets_val=Tv
    )
    return addr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scope", action="append", default=None)
    ap.add_argument("--arm", default=None, help="object to delete from (default: the K* build)")
    a = ap.parse_args(argv)
    wt = assert_worktree()
    import shapes_deletion_rows as DR
    import shapes_ladder_lib as LEG
    from nclu.core.gen_read import span_coefficients
    from nclu.experiments import factor_wells as fw
    from nclu.experiments.gen_deletion import assert_generation2
    from nclu.experiments.covariance_store import CovContentStore
    from shapes_nclu_lib import D, pinned_basis
    from shapes_fast_judge import FastJudge, assert_identical

    seed = int(a.seed)
    fw.PINS["seed"], fw.PINS["condition"] = (seed, "test")
    man = assert_generation2(seed)
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    corners = fw.pair_subcorners(cond) + fw.nway_corners(cond)
    seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    bitmap0 = fw.value_presence_bitmap(grid, cond["train_rows"])
    judge = fw.Judge(seed, log=lambda *_: None)
    fast = FastJudge(judge)
    pool, _dead, _pf = load_pool("test")
    F = fw.features_mmap(seed)
    chk = assert_identical(judge, fast, fw.gather(F, pool[:256]))
    log("seed %d  FastJudge identical %s" % (seed, chk["identical"]))
    with open(a.arm or kstar_arm(seed), "rb") as fh:
        saved = pickle.load(fh)
    st, addr0, basis0 = (saved["store"], saved["address"], saved["basis"])
    wc = [np.asarray(c, np.int64) for c in saved["well_channels"]]
    A = int(saved["A"])
    idx = DR.factor_indices(pool, cards)
    red = idx[:, DR.RED_FACTOR] == DR.RED_INDEX
    masks = {"value": red, "conjunction_live": red & (idx[:, DR.SHAPE_FACTOR] == DR.SPHERE_INDEX)}
    rng = np.random.default_rng(20260921 + seed)
    m_item = np.zeros(pool.size, bool)
    m_item[rng.choice(pool.size, int(round(DR.ITEM_FRACTION * pool.size)), replace=False)] = True
    masks["item"] = m_item
    ntr = int(0.9 * pool.size)
    fitpos = np.arange(pool.size) < ntr
    Yall = grid.factor_indices_of(pool)

    def score(obj, bitmap, sel_ids):
        cc = [c for c in corners if c["corner_id"] in sel_ids]
        per = LEG.score_arm(obj, dict(cond, corners=cc), fast, bitmap, seen_mask)
        return per

    outd = OUT / "deletion_store" / ("seed%d" % seed)
    outd.mkdir(parents=True, exist_ok=True)
    for scope in a.scope or ["value", "conjunction_live", "item"]:
        p = outd / ("%s.json" % scope)
        if p.exists():
            log("%s already on disk" % scope)
            continue
        m = masks[scope]
        X = np.asarray(fw.gather(F, pool), np.float32)
        rec = {
            "scope": scope,
            "seed": seed,
            "n_deleted": int(m.sum()),
            "n_survivors": int((~m).sum()),
            "cache": man,
            "package_file": wt,
            "pool_rows": int(pool.size),
        }
        log("===== %s: %d rows deleted =====" % (scope, int(m.sum())))
        st2 = copy.deepcopy(st)
        t0 = time.time()
        receipt = st2.delete_rows(X[m])
        U_post = pinned_basis(st2)
        t_store = time.time() - t0
        t0 = time.time()
        ref = CovContentStore(X[~m], wc, A, D)
        t_ref = time.time() - t0
        U_ref = pinned_basis(ref)
        devs = [float(np.max(np.abs(st2.modes[k] - ref.modes[k]))) for k in range(len(wc))]
        rec["a_store_equals_refit_without"] = {
            "RESULT": True,
            "max_abs_mode_dev": max(devs),
            "ranks_agree": list(st2.deployed_rank) == list(ref.deployed_rank),
            "downdate_seconds": round(t_store, 3),
            "downdate_data_rows_read": 0,
            "refit_without_seconds": round(t_ref, 2),
            "refit_without_data_rows_read": int((~m).sum()),
        }
        log(
            "  (a) store max|mode dev| %.3e ranks agree %s"
            % (max(devs), rec["a_store_equals_refit_without"]["ranks_agree"])
        )
        fit_all = fitpos
        stat_n = fit_all.astype(np.int64)
        stat_s = np.zeros((pool.size, X.shape[1]), np.float32)
        stat_s[fit_all] = X[fit_all]
        t0 = time.time()
        stat_n[m] = 0
        stat_s[m] = 0.0
        t_bdown = time.time() - t0
        keep_fit = fit_all & ~m
        dn = int(np.max(np.abs(stat_n - keep_fit.astype(np.int64))))
        dm = 0.0
        for s0 in range(0, pool.size, 8192):
            e0 = min(s0 + 8192, pool.size)
            scr = np.where(keep_fit[s0:e0, None], X[s0:e0], 0).astype(np.float32)
            dm = max(dm, float(np.max(np.abs(stat_s[s0:e0] - scr))))
        b1 = {
            "max_abs_count_dev": dn,
            "max_abs_first_moment_dev": dm,
            "statistic": "per-tuple (count, raw-feature first moment) over the book's fit rows",
            "statistic_bytes": int(fit_all.sum()) * (D * 4 + 8),
            "statistic_bytes_note": "= the fit data itself: every Shapes3D tuple is ONE row, and the ALS book has no smaller exact statistic",
            "downdate_seconds": round(t_bdown, 3),
        }
        log(
            "  (b1) stat dev count %d moment %.3e  (%.2f GB statistic)"
            % (dn, dm, b1["statistic_bytes"] / 1000000000.0)
        )
        down_n, down_s = (stat_n, stat_s)
        t0 = time.time()
        sel = down_n > 0
        Tfit = span_coefficients(st2, (down_s[sel] / down_n[sel, None]).astype(np.float32), U_post)
        vsel = ~fitpos & ~m
        Tval = span_coefficients(st2, X[vsel], U_post)
        del down_s, stat_s, X
        book_A = fit_book(cards, Yall[sel], Tfit, Yall[vsel], Tval)
        t_bookA = time.time() - t0
        del Tfit, Tval
        from nclu.experiments import exp_gen_read as R

        t0 = time.time()
        surv = pool[~m]
        Ys, Ts = R._pool_span_targets(ref, surv, U_ref, log=lambda *_: None)
        fit_s = fitpos[~m]
        book_B = fit_book(
            cards, Ys[fit_s], [t[fit_s] for t in Ts], Ys[~fit_s], [t[~fit_s] for t in Ts]
        )
        t_bookB = time.time() - t0
        del Ts
        eA, eB = (
            np.concatenate([np.ravel(e) for e in book_A.E]),
            np.concatenate([np.ravel(e) for e in book_B.E]),
        )
        wA, wB = (
            np.concatenate([np.ravel(w) for w in book_A.W]),
            np.concatenate([np.ravel(w) for w in book_B.W]),
        )
        if scope == "value":
            dele_ids = {
                c["corner_id"]
                for c in corners
                if DR.corner_requests_value(c, grid, DR.RED_FACTOR, DR.RED_INDEX)
            }
        elif scope == "conjunction_live":
            dele_ids = {
                c["corner_id"]
                for c in corners
                if DR.corner_requests_value(c, grid, DR.RED_FACTOR, DR.RED_INDEX)
                and DR.corner_requests_value(c, grid, DR.SHAPE_FACTOR, DR.SPHERE_INDEX)
            }
        else:
            dele_ids = set()
        surv_ids = {c["corner_id"] for c in corners} - dele_ids
        cc = [c for c in corners if c["corner_id"] in surv_ids]
        V = np.concatenate([fw._request_values_from_rows(grid, c["rows"]) for c in cc])
        xa, _ = book_A.read(st2, V, grid.values, U_post)
        xb, _ = book_B.read(ref, V, grid.values, U_ref)
        rel = float(np.linalg.norm(xa - xb) / (np.linalg.norm(xb) + 1e-12))
        rec["b_address_book_statistics_equal_refit_without"] = {
            "RESULT": True,
            "b1_statistic": b1,
            "b2_book": {
                "max_abs_E_dev": float(np.max(np.abs(eA - eB))),
                "max_abs_W_dev": float(np.max(np.abs(wA - wB))),
                "rel_E_dev": float(np.linalg.norm(eA - eB) / np.linalg.norm(eB)),
                "rel_W_dev": float(np.linalg.norm(wA - wB) / np.linalg.norm(wB)),
                "rendered_features_rel_dev_on_survivor_requests": rel,
                "n_requests": int(V.shape[0]),
                "book_refit_from_downdated_statistic_seconds": round(t_bookA, 1),
                "book_refit_without_pipeline_seconds": round(t_bookB, 1),
                "book_refit_without_data_rows_read": int(surv.size),
                "book_from_statistic_pool_rows_read": 0,
                "membership": "positional 90/10 membership preserved for survivors",
            },
        }
        log(
            "  (b2) book E rel dev %.3e  W rel dev %.3e  rendered rel dev %.3e  (%.0fs / %.0fs)"
            % (
                rec["b_address_book_statistics_equal_refit_without"]["b2_book"]["rel_E_dev"],
                rec["b_address_book_statistics_equal_refit_without"]["b2_book"]["rel_W_dev"],
                rel,
                t_bookA,
                t_bookB,
            )
        )
        bitmap1 = [np.array(b, copy=True) for b in bitmap0]
        c_rec = {"DISPLAY_NEVER_A_RESULT": True}
        if scope == "value":
            bitmap1[DR.RED_FACTOR][DR.RED_INDEX] = False
            c_rec["hole"] = (
                "value-presence bit object_hue=0.0 (red) cleared: 1 bit rewritten, 0 rows read"
            )
        elif scope == "conjunction_live":
            c_rec["hole"] = (
                "coupling hole (object_hue=0.0, shape=2.0) in the 1,322 B coupling table (T7 display_refusal index 922): 1 byte rewritten, 0 rows read"
            )
        else:
            c_rec["DECLARED_NA"] = (
                "an item deletion clears no value bit and punches no conjunction hole: 0 certifiably-impossible requests (shapes3d/refusal_comparator FINDINGS §3)"
            )

        def refused(Vr):
            if scope == "value":
                acc = fw.accept_rule(
                    Vr, np.zeros(len(Vr), bool), np.zeros(len(Vr), bool), bitmap1, grid
                )
                return ~acc
            if scope == "conjunction_live":
                hv = np.isclose(
                    Vr[:, DR.RED_FACTOR], grid.values[DR.RED_FACTOR][DR.RED_INDEX]
                ) & np.isclose(
                    Vr[:, DR.SHAPE_FACTOR], grid.values[DR.SHAPE_FACTOR][DR.SPHERE_INDEX]
                )
                return hv
            return np.zeros(len(Vr), bool)

        if scope != "item":
            dc = [c for c in corners if c["corner_id"] in dele_ids]
            Vd = (
                np.concatenate([fw._request_values_from_rows(grid, c["rows"]) for c in dc])
                if dc
                else np.zeros((0, len(cards)))
            )
            c_rec.update(
                {
                    "refusal_rate_on_deleted_corner_requests": float(refused(Vd).mean())
                    if len(Vd)
                    else None,
                    "n_deleted_corner_requests": int(len(Vd)),
                    "false_refusal_rate_on_survivor_requests": float(refused(V).mean()),
                    "n_survivor_requests": int(len(V)),
                }
            )
        rec["c_hole_DISPLAY"] = c_rec
        objA = {
            "arm": "after_%s" % scope,
            "read": "tensor",
            "kind": "span",
            "store": st2,
            "basis": U_post,
            "address": book_A,
            "protos": None,
            "head_bytes": {"head_bytes": 0},
        }
        perA = score(objA, bitmap1, surv_ids)
        objB = dict(objA, store=ref, basis=U_ref, address=book_B, arm="refit_without_%s" % scope)
        perB = score(objB, bitmap1, surv_ids)
        obj0 = dict(objA, store=st, basis=basis0, address=addr0, arm="before")
        by0 = {r["corner_id"]: r for r in score(obj0, bitmap0, surv_ids)}
        befo = [by0[r["corner_id"]] for r in perA]
        pd = fw.paired_estimand(perA, befo, "distinct_valid_novel_tuples", 1.0, "after", "before")
        pAB = fw.paired_estimand(
            perA, perB, "distinct_valid_novel_tuples", 1.0, "downdate", "refit_without"
        )
        rec["d_composition_other_values"] = {
            "RESULT": True,
            "n_survivor_corners": len(perA),
            "before_mean": pd["control_mean"],
            "after_mean": pd["arm_mean"],
            "paired_after_minus_before": {
                k: pd[k]
                for k in (
                    "mean_paired_delta",
                    "bootstrap_ci95",
                    "corners_arm_wins",
                    "corners_tied",
                    "corners_control_wins",
                )
            },
            "downdate_minus_refit_without_pipeline": {
                k: pAB[k]
                for k in (
                    "mean_paired_delta",
                    "bootstrap_ci95",
                    "corners_arm_wins",
                    "corners_tied",
                    "corners_control_wins",
                )
            },
            "per_corner_after": perA,
            "per_corner_refit_without": perB,
        }
        rec["receipt_keys"] = sorted(receipt.keys()) if isinstance(receipt, dict) else None
        log(
            "  (d) survivors %d: before %.3f after %.3f  delta %+.3f  | vs refit-without %+.3f"
            % (
                len(perA),
                pd["control_mean"],
                pd["arm_mean"],
                pd["mean_paired_delta"],
                pAB["mean_paired_delta"],
            )
        )
        jdump(rec, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
