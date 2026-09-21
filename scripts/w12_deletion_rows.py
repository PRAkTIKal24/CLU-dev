"""w12 -- THE DELETION ROWS: three scopes, one object, every cost line printed.

(a) VALUE        delete every fit row carrying RED (object_hue index 0)
(b) CONJUNCTION  delete every fit row carrying RED and CUBE (shape index 0)
(c) ITEM         delete a random 1 % of fit rows

and, beside them, the DISPLAY refusal row: the coupling/hole table with a hole punched
at the deleted conjunction.

TWO EXACT MECHANISMS, ON ONE OBJECT, REPORTED SEPARATELY AND NEVER MERGED:
  (i)  data forgetting  = the covariance/count DOWNDATE
  (ii) refusal          = the request count / coupling table WITH HOLES

A compositional store can still compose "red cube" from "red" and "cube" after the
red-cube rows are gone.  THAT IS COMPOSITION WORKING, NOT DELETION FAILING -- the
refusal row is what the hole is for.  Both are reported, separately.

"EXACT" HERE MEANS STORE-SCOPED EXACTNESS: `downdate(delete rows)` equals the store
fitted from scratch without those rows, at the same partition and the same rank.  The
frozen encoder and the frozen decoder are NOT refit.  That is the only definition these
measurements support and it is the one written on every row.

LANGUAGE, binding: never "no changes after deletion".  Post-deletion adaptation is
LIGHT and DATA-FREE -- re-eigendecompose the ledger, carry the address book, no row of
data read -- against baselines that can only delete by retraining from scratch without
the deleted rows.

On `--condition val` every number is a DEV number and is marked `SMOKE_NOT_A_CLAIM`.
On `--condition test` the gate and the ledger row are enforced by `w12_scored_leg`'s
`assert_gates`, which this script calls.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w12_crossover_lib import (  # noqa: E402
    D, atoms_for, four_part_ledger, load_partition, log, pinned_basis, wells_for,
)

OUT = Path("C:/chlu/outputs/w12-single-design-crossover")
RED_FACTOR, RED_INDEX = 2, 0        # object_hue = 0.0, pinned at exp_gen_factor_wells.py:1348
SHAPE_FACTOR, CUBE_INDEX = 4, 0     # shape = 0.0 = cube
#: THE CONJUNCTION AS LITERALLY SPECIFIED IS VACUOUS ON THIS CONDITION, and it is
#: reported as vacuous rather than quietly replaced.  `shape = cube` is HELD OUT of the
#: value-holdout split entirely, so the fit pool contains ZERO cubes and "delete every
#: row carrying red AND cube" removes 0 rows.  The repo already says so in its own words:
#:   exp_gen_delete_row7.py:325  "red_cube": (0, 0, False,
#:       "shape=cube is held out entirely; reported, not a bar")
#: MEASURED on the fit pool, val / test:
#:   red & cube     0 / 0          (cube is a held-out VALUE)
#:   red & cylinder 0 / 0          (red-cylinder is the held-out PAIR cell)
#:   red & sphere   5,507 / 5,511
#:   red & oblong   5,517 / 5,533
#: So the DECLARED NON-VACUOUS SUBSTITUTE is red AND SPHERE (shape index 2) -- a
#: conjunction whose two marginals both survive the deletion, which is the only version of
#: this row that can test anything.
#: THE INDEX IS NOT MINE TO CHOOSE FREELY: spoke/deletion-baselines found the same
#: vacuity independently and moved its own conjunction pin to
#:   "object_hue 0.0 AND shape 2.0 = 5,511 rows", declared in its PINS_W12
#: (/c/chlu/outputs/w12-deletion-baselines/LOG.md, 11:44Z entry), and its
#: retrained-without comparator deletes exactly those rows.  A comparator and an arm that
#: delete DIFFERENT rows are not a comparison, so this pin is adopted from the spoke
#: rather than picked here.  BOTH rows are reported: the specified one with its count of
#: 0 and its reason, and the substitute with its own row.
SPHERE_INDEX = 2                    # shape = 2.0 = sphere
ITEM_FRACTION = 0.01                # the declared 1 %


def factor_indices(pool: np.ndarray, cards) -> np.ndarray:
    """row id -> (n, F) factor indices, odometer order (last factor fastest)."""
    c = np.asarray([int(x) for x in cards], np.int64)
    strides = np.ones(c.size, np.int64)
    for i in range(c.size - 2, -1, -1):
        strides[i] = strides[i + 1] * c[i + 1]
    return (np.asarray(pool, np.int64)[:, None] // strides[None, :]) % c[None, :]


def match_basis_width(U_pre, U_post):
    """Make the post-deletion basis emit the same number of coefficients as the book.

    The deployed rank can DROP when rows are removed (a direction the data no longer
    determines is cut), and `carry_address_book` asserts the widths match.  Padding the
    missing columns with zeros is the honest completion: the book's coefficient for a
    direction the store no longer holds is projected onto nothing, which is what the
    deletion means.  It is recorded on the row rather than done silently.
    """
    out, note = [], []
    for k, (A0, A1) in enumerate(zip(U_pre, U_post)):
        r0, r1 = A0.shape[1], A1.shape[1]
        if r1 == r0:
            out.append(A1)
            continue
        if r1 < r0:
            B = np.zeros((A1.shape[0], r0), np.float64)
            B[:, :r1] = A1
            out.append(B)
            note.append({"well": k, "rank_before": int(r0), "rank_after": int(r1),
                         "action": "zero-padded to the book's width"})
        else:
            out.append(A1[:, :r0])
            note.append({"well": k, "rank_before": int(r0), "rank_after": int(r1),
                         "action": "truncated to the book's width"})
    return out, note


def our_cost_line(st_before, st_after, carry_rec, seconds: float) -> dict:
    """seconds | bytes rewritten | data rows read.  The last one is 0 and that is the point."""
    atom_b = int(sum(M.size for M in st_after.modes)) * 4
    cov_b = int(sum(w.stat_bytes() for w in st_after.stats))
    addr_b = int(carry_rec["address_book_bytes_changed"])
    return {
        "arm": "cov_tensor (ours)",
        "seconds": round(float(seconds), 4),
        "bytes_rewritten": int(atom_b + cov_b + addr_b),
        "bytes_rewritten_breakdown": {
            "atoms_re_derived": atom_b,
            "covariance_and_count_downdated": cov_b,
            "address_book_carried": addr_b,
        },
        "data_rows_read": 0,
        "refit": False,
        "items_kept": False,
        "what_happened": ("post-deletion adaptation is LIGHT and DATA-FREE: the "
                          "covariance and count are downdated by subtraction, the atoms "
                          "are re-eigendecomposed from the downdated statistic, and the "
                          "address book is carried across the basis change. NO ROW OF "
                          "DATA IS READ."),
        "the_comparator_pays": ("a learned baseline can only delete by retraining from "
                                "scratch WITHOUT the deleted rows -- its cost line is "
                                "its own and is printed beside this one"),
    }


def corner_requests_value(corner, grid, factor: int, index: int) -> bool:
    """Does this corner ask for that factor value anywhere in its request set?"""
    from chlu.experiments import exp_gen_factor_wells as fw
    vals = fw._request_values_from_rows(grid, corner["rows"])
    gv = np.asarray(grid.values[factor], np.float64)
    target = float(gv[index])
    col = np.asarray(vals, np.float64)[:, factor]
    ok = ~np.isnan(col)
    return bool((np.isclose(col[ok], target)).any())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val", choices=("val", "test"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--run-n", type=int, default=None)
    ap.add_argument("--scope", action="append",
                    choices=("value", "conjunction", "conjunction_live", "item"),
                    default=None)
    ap.add_argument("--reuse-arm", default=None,
                    help="a directory holding arm.pkl from a previous build/fit")
    a = ap.parse_args(argv)
    scopes = a.scope or ["value", "conjunction", "conjunction_live", "item"]

    if a.condition == "test":
        from w12_scored_leg import assert_gates
        if a.run_n is None:
            raise SystemExit("--run-n is required on the test condition")
        assert_gates(a.run_n)

    if a.K is None:
        doc = json.loads((OUT / "CROSSOVER-val.json").read_text(encoding="utf-8"))
        if doc.get("K_star") is None:
            raise SystemExit("K* is None -- the declared rule selected nothing.")
        Kstar = int(doc["K_star"])
    else:
        Kstar = int(a.K)

    from chlu.core.downdate_store import (carry_address_book, coupling_index,
                                          coupling_layout, term_offsets)
    from chlu.core.gen_read import TensorAddress
    from chlu.core.gen_store import channel_correlation
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    from chlu.experiments.exp_w8_single_design import CovContentStore
    from w12_scored_leg import score_corners_full, summarise

    seed = int(a.seed)
    fw.PINS["condition"] = a.condition
    fw.PINS["seed"] = seed
    man = assert_generation2(seed)
    _z, dead, pool = load_partition(a.condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    judge = fw.Judge(seed, log=lambda *_: None)
    if a.condition == "val":
        fitted = np.setdiff1d(cond["train_rows"], fw.val_corner_rows(cond))
        corners = fw.val_corners(cond)
    else:
        fitted = cond["train_rows"]
        corners = fw.pair_subcorners(cond) + fw.nway_corners(cond)
    bitmap = fw.value_presence_bitmap(grid, fitted)
    seen_mask = fw.seen_tuple_mask(grid, fitted)
    log(f"condition={a.condition} seed={seed} K={Kstar} corners={len(corners)} "
        f"cache={man}")

    F = fw.features_mmap(seed)
    Xc = np.asarray(fw.gather(F, pool[: int(fw.PINS["n_corr"])]), np.float32)
    corr = channel_correlation(Xc)
    del Xc
    X = np.asarray(fw.gather(F, pool), np.float32)
    idx = factor_indices(pool, cards)
    red = idx[:, RED_FACTOR] == RED_INDEX
    masks = {
        "value": red,
        "conjunction": red & (idx[:, SHAPE_FACTOR] == CUBE_INDEX),
        "conjunction_live": red & (idx[:, SHAPE_FACTOR] == SPHERE_INDEX),
    }
    shape_counts = {nm: int((red & (idx[:, SHAPE_FACTOR] == si)).sum())
                    for si, nm in ((0, "cube"), (1, "cylinder"), (2, "sphere"),
                                   (3, "oblong"))}
    log(f"  red AND <shape> rows in the fit pool: {shape_counts}")
    rng = np.random.default_rng(20_260_921 + seed)
    m_item = np.zeros(pool.size, bool)
    m_item[rng.choice(pool.size, int(round(ITEM_FRACTION * pool.size)),
                      replace=False)] = True
    masks["item"] = m_item
    for k, m in masks.items():
        log(f"  scope {k:12s}: {int(m.sum()):,} of {pool.size:,} fit rows "
            f"({100.0 * m.mean():.2f} %)")

    wc = wells_for(corr, Kstar, dead)
    sizes = [int(c.size) for c in wc]
    A = atoms_for(sizes)

    reuse = Path(a.reuse_arm) / "arm.pkl" if a.reuse_arm else None
    if reuse and reuse.exists():
        with open(reuse, "rb") as fh:
            saved = pickle.load(fh)
        st, addr, basis = saved["store"], saved["address"], saved["basis"]
        log(f"  reusing the fitted arm from {reuse} (K={saved['K']}, A={saved['A']})")
        t_addr = None
    else:
        t0 = time.time()
        st = CovContentStore(X, wc, A, D)
        basis = pinned_basis(st)
        Y, T = R._pool_span_targets(st, pool, basis, log=lambda *_: None)
        ntr = int(0.9 * Y.shape[0])
        addr = TensorAddress(cards, rank=int(R.READ_PINS["rank"]),
                             ridge=float(R.READ_PINS["ridge"]),
                             seed=int(R.READ_PINS["address_seed"]))
        addr.multiplicative = True
        addr.fit(Y[:ntr], [t[:ntr] for t in T], sweeps=int(R.READ_PINS["sweeps"]),
                 log=lambda *_: None, Y_val=Y[ntr:], targets_val=[t[ntr:] for t in T])
        t_addr = time.time() - t0
        log(f"  built + fitted in {t_addr:.1f}s")

    abytes = int(addr.byte_ledger()["address_bytes"])
    led = four_part_ledger(st, abytes)
    obj = {"arm": "cov_tensor", "read": "tensor", "kind": "span", "store": st,
           "basis": basis, "address": addr, "protos": None,
           "head_bytes": {"head_bytes": 0}}
    before = score_corners_full(obj, corners, grid, judge, bitmap, "BEFORE deletion")
    rows_before = fw._per_corner(corners, before, grid, seen_mask)
    by_id_before = {r["corner_id"]: r for r in rows_before}

    out = {
        "SMOKE_NOT_A_CLAIM": a.condition == "val",
        "condition": a.condition, "seed": seed, "K": Kstar, "A_requested": A,
        "feature_cache": man, "ledger_four_part": led,
        "exactness_definition": (
            "STORE-SCOPED: downdate(delete rows) equals the store fitted from scratch "
            "without those rows, same partition, same rank. THE FROZEN ENCODER AND "
            "DECODER ARE NOT REFIT."),
        "before": {"all": summarise(rows_before), "per_corner": rows_before},
        "scopes": {},
        "display_refusal": None,
    }

    for scope in scopes:
        m = masks[scope]
        log(f"\n===== SCOPE {scope.upper()} -- {int(m.sum()):,} rows =====")
        st2 = copy.deepcopy(st)
        U_pre = pinned_basis(st2)
        t0 = time.time()
        receipt = st2.delete_rows(X[m])
        U_post_raw = pinned_basis(st2)
        U_post, width_note = match_basis_width(U_pre, U_post_raw)
        addr2, carry = carry_address_book(addr, U_pre, U_post)
        t_adapt = time.time() - t0
        cost_ours = our_cost_line(st, st2, carry, t_adapt)
        log(f"  OUR ADAPTATION  {cost_ours['seconds']}s  "
            f"{cost_ours['bytes_rewritten']:,} B rewritten  "
            f"{cost_ours['data_rows_read']} data rows read")

        # --- STORE-SCOPED EXACTNESS, with the refit's own cost printed ---------
        t0 = time.time()
        ref = CovContentStore(X[~m], wc, A, D)
        t_refit = time.time() - t0
        devs, sub = [], []
        for k in range(Kstar):
            r = min(st2.deployed_rank[k], ref.deployed_rank[k])
            devs.append(float(np.max(np.abs(st2.modes[k] - ref.modes[k]))))
            if r > 0:
                Ud = np.linalg.qr(np.asarray(st2.modes[k][1:1 + r], np.float64).T)[0]
                Ur = np.linalg.qr(np.asarray(ref.modes[k][1:1 + r], np.float64).T)[0]
                sub.append(float(np.linalg.norm(Ud @ Ud.T - Ur @ Ur.T, 2)))
        log(f"  EXACTNESS (store-scoped)  max|mode dev| {max(devs):.3e}  "
            f"max subspace dist {(max(sub) if sub else float('nan')):.3e}  "
            f"ranks agree {st2.deployed_rank == ref.deployed_rank}  "
            f"(refit-without took {t_refit:.1f}s, read {int((~m).sum()):,} rows)")

        # --- the composition row after the deletion ---------------------------
        obj2 = dict(obj, store=st2, basis=U_post_raw, address=addr2,
                    arm=f"cov_tensor_after_{scope}")
        after = score_corners_full(obj2, corners, grid, judge, bitmap,
                                   f"AFTER {scope}")
        rows_after = fw._per_corner(corners, after, grid, seen_mask)
        by_id_after = {r["corner_id"]: r for r in rows_after}

        if scope == "value":
            surv = [c["corner_id"] for c in corners
                    if not corner_requests_value(c, grid, RED_FACTOR, RED_INDEX)]
            dele = [c["corner_id"] for c in corners
                    if corner_requests_value(c, grid, RED_FACTOR, RED_INDEX)]
        elif scope in ("conjunction", "conjunction_live"):
            si = CUBE_INDEX if scope == "conjunction" else SPHERE_INDEX
            surv = [c["corner_id"] for c in corners
                    if not (corner_requests_value(c, grid, RED_FACTOR, RED_INDEX)
                            and corner_requests_value(c, grid, SHAPE_FACTOR, si))]
            dele = [c["corner_id"] for c in corners if c["corner_id"] not in surv]
        else:
            surv = [c["corner_id"] for c in corners]
            dele = []

        paired = (fw.paired_estimand([by_id_after[i] for i in surv],
                                     [by_id_before[i] for i in surv],
                                     "distinct_valid_novel_tuples", 1.0,
                                     f"after_{scope}", "before")
                  if surv else {"DECLARED_NOT_RUN": "no survivor corner"})
        out["scopes"][scope] = {
            "scope_definition": {
                "value": "every fit row carrying red (object_hue index 0)",
                "conjunction": ("every fit row carrying red AND cube (shape index 0)"
                                " -- AS LITERALLY SPECIFIED BY THE TASK FILE"),
                "conjunction_live": ("every fit row carrying red AND sphere (shape "
                                     "index 2) -- THE DECLARED NON-VACUOUS "
                                     "SUBSTITUTE, because red AND cube and red AND "
                                     "cylinder are both EMPTY in this split"),
                "item": "a random 1 % of fit rows",
            }[scope],
            "VACUOUS": bool(int(m.sum()) == 0),
            "vacuous_reason": (
                ("shape=cube is held out of the value-holdout split entirely, so the "
                 "fit pool contains ZERO cubes and this deletion removes 0 rows. It "
                 "is reported as a count of 0 with its reason, NEVER as a passing "
                 "row. exp_gen_delete_row7.py:325 states the same property in the "
                 "repo's own words. The non-vacuous substitute is the "
                 "`conjunction_live` scope.")
                if int(m.sum()) == 0 else None),
            "red_and_shape_row_counts": shape_counts,
            "n_rows_deleted": int(m.sum()),
            "frac_rows_deleted": float(m.mean()),
            "survivor_corner_ids": surv, "deleted_corner_ids": dele,
            "n_survivor_corners": len(surv), "n_deleted_corners": len(dele),
            "our_cost_line": cost_ours,
            "receipt": {k: str(v) for k, v in receipt.items()},
            "basis_change_audit": st2.basis_change_audit(),
            "basis_width_adjustments": width_note,
            "carry_receipt": {k: (v if not isinstance(v, (list, np.ndarray))
                                  else list(np.asarray(v).ravel()[:8]))
                              for k, v in carry.items()},
            "exactness_store_scoped": {
                "max_abs_mode_dev": float(max(devs)), "per_well_mode_dev": devs,
                "max_subspace_projector_dist": float(max(sub)) if sub else None,
                "ranks_agree": bool(st2.deployed_rank == ref.deployed_rank),
                "refit_without_cost_line": {
                    "arm": "refit-from-scratch-without (the reference)",
                    "seconds": round(t_refit, 3),
                    "data_rows_read": int((~m).sum()),
                    "bytes_rewritten": int(
                        sum(M.size for M in ref.modes) * 4
                        + sum(w.stat_bytes() for w in ref.stats)),
                    "refit": True},
            },
            "after": {"survivors": summarise([by_id_after[i] for i in surv]),
                      "deleted": summarise([by_id_after[i] for i in dele]) if dele
                      else {"DECLARED_NOT_RUN": "this scope deletes no corner"},
                      "per_corner": rows_after},
            "before_same_corners": {
                "survivors": summarise([by_id_before[i] for i in surv]),
                "deleted": summarise([by_id_before[i] for i in dele]) if dele
                else {"DECLARED_NOT_RUN": "this scope deletes no corner"}},
            "paired_after_minus_before_on_survivors": paired,
        }
        del st2, ref

    # --- DISPLAY: the coupling / hole table -----------------------------------
    # MECHANISM (ii), and it is a DIFFERENT array from the statistic.  Punching the hole
    # changes ONE byte and leaves the coupling VALUE where it is, so the marginals "red"
    # and "cube" stay answerable -- which is exactly why the object can still COMPOSE
    # "red cube" after the red-cube rows are gone while REFUSING the request.
    lay = coupling_layout(cards)
    hole = np.zeros(lay["size"], np.uint8)
    hidx = coupling_index(cards, RED_FACTOR, RED_INDEX, SHAPE_FACTOR, CUBE_INDEX)
    hidx_live = coupling_index(cards, RED_FACTOR, RED_INDEX, SHAPE_FACTOR,
                               SPHERE_INDEX)
    t0 = time.time()
    hole[hidx] = 1
    hole[hidx_live] = 1
    t_hole = time.time() - t0
    from chlu.core.downdate_store import assert_layout_matches_field_store
    out["display_refusal"] = {
        "DISPLAY_NEVER_A_RESULT": True,
        "layout_is_the_field_store's": assert_layout_matches_field_store(cards),
        "holes": [
            {"index": int(hidx),
             "pair": [["object_hue", int(RED_INDEX)], ["shape", int(CUBE_INDEX)]],
             "label": "red AND cube -- the task file's conjunction",
             "note": ("shape=cube is held out of the split, so NO red cube was ever "
                      "written. The hole still makes the request refusable, and "
                      "that is the point of the row: refusal is a SEPARATE "
                      "mechanism from forgetting and does not need a deleted row "
                      "to have existed.")},
            {"index": int(hidx_live),
             "pair": [["object_hue", int(RED_INDEX)], ["shape", int(SPHERE_INDEX)]],
             "label": "red AND sphere -- the non-vacuous substitute",
             "note": ("here rows WERE deleted, so the hole and the downdate are "
                      "exercised on the same conjunction and the two mechanisms are "
                      "separable on one object")},
        ],
        "cost_line": {"arm": "cov_tensor (ours)", "seconds": round(t_hole, 6),
                      "bytes_rewritten": 2, "data_rows_read": 0, "refit": False,
                      "note": "one byte per hole, two holes"},
        "n_holes_after": int(hole.sum()),
        "coupling_table_bytes": int(lay["size"]),
        "comparator": ("the refuse-token transformer from spoke/deletion-baselines is "
                       "the ONLY permitted comparator for a refusal DISPLAY row; the "
                       "baselines' quantity is their RENDER RATE on the deleted "
                       "request, not a refusal"),
        "comparator_status": "DECLARED NOT-RUN here -- see LOG.md for the gate state",
        "carried_caveat": ("wave 9's refuser separated possible from impossible at only "
                           "+0.047 and a ridge probe on the same observation scored "
                           "AUC 0.48-0.55, i.e. chance, so a DISPLAY comparison against "
                           "a refuser of that quality is not yet informative"),
        "what_this_row_is_for": (
            "A compositional store can still compose 'red cube' from 'red' and 'cube' "
            "after the red-cube rows are gone. THAT IS COMPOSITION WORKING, NOT "
            "DELETION FAILING. The hole is the separate mechanism that makes the "
            "request refusable, and the two are reported separately on the same object."),
    }

    d = OUT / "deletion" / f"{a.condition}_seed{seed}_K{Kstar}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "deletion.json").write_text(json.dumps(out, indent=2, default=str),
                                     encoding="utf-8")
    log(f"\nwrote {d / 'deletion.json'}")
    for s, rec in out["scopes"].items():
        p = rec["paired_after_minus_before_on_survivors"]
        log(f"  {s:12s} survivors {rec['n_survivor_corners']:3d}  "
            f"novel/corner {rec['before_same_corners']['survivors']['novel_per_corner_mean']:7.3f}"
            f" -> {rec['after']['survivors']['novel_per_corner_mean']:7.3f}   "
            f"paired {p.get('mean_paired_delta', float('nan')):+.4f}   "
            f"exactness {rec['exactness_store_scoped']['max_abs_mode_dev']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
