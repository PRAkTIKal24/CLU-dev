"""w15 -- THE DELETION ROWS AT THREE SEEDS, with the RIVAL deletion column.

THIS SCRIPT SPENDS NO SCORED RUN.  `single-design` is EXHAUSTED at 3/3:
/c/chlu/SCORED-LEDGER.md rows 43 and 44 (both 2026-09-21T21:33:51Z) charge
`w12b hub/nr6-and-kstar | single-design | 2/3` and `3/3`, and row 3/3 says in its own
words that it "adds seeds 1 and 2 at K* = 32 on TEST".  Those artefacts exist and are
READ, not re-produced.  What did NOT get produced there is the PREREG's RESULT 3 -- the
deletion rows -- which exist at SEED 0 ONLY.  This script completes RESULT 3 at seeds 1
and 2 on the SAME already-fitted arms (`arm.pkl`), the SAME K* = 32, the SAME TEST
condition, under the SAME PREREG + AMENDMENT 1.  No bar moves, no K is selected, no
design changes.  A ledger NOTE (not a run row) was appended at 2026-09-22T21:31:47Z
before any number here.

WHAT IS GENUINELY NEW: the RIVAL DELETION COLUMN, absent at every seed including 0.
Rival deletes ONLY by retrain-without, cost printed; rival as-trained printed beside it.
Families mlp / lstm / transformer / cont_mlp from
C:/chlu/outputs/w12-deletion-baselines/comparators/seed_{0,1,2}/, entry point
`chlu.eval.rivals.w12_deletion_baselines.load_comparator` as named in
BASELINES-READY.json.

Every helper on the arm side is IMPORTED from `w12_deletion_rows`, not re-implemented,
so the arm leg is byte-for-byte run 1/3's configuration.

"EXACT" = STORE-SCOPED exactness: downdate(delete rows) equals the store fitted from
scratch without those rows, at the same partition and the same rank.  THE FROZEN ENCODER
AND THE FROZEN DECODER ARE DECLARED OUTSIDE THE DELETABLE OBJECT AND ARE NOT REFIT.
Post-deletion adaptation is LIGHT and DATA-FREE -- never "no changes".
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
    D, four_part_ledger, load_partition, log, pinned_basis,
)
import w12_deletion_rows as W12D  # noqa: E402

OUT = Path("C:/chlu/outputs/w15-single-design-seeds")
W12_OUT = Path("C:/chlu/outputs/w12-single-design-crossover")
W12B_OUT = Path("C:/chlu/outputs/w12b-nr6-and-kstar")
COMPARATORS_ROOT = "C:/chlu/outputs/w12-deletion-baselines/comparators"
BASELINES_READY = Path("C:/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json")

RIVALS = ("mlp", "lstm", "transformer", "cont_mlp")

#: scope -> the comparator's own spec LABEL on disk.  `conjunction` (red AND cube) has
#: NO comparator and never will: it deletes 0 rows and `partition_fit_rows` REFUSES a
#: no-op spec by construction.  Reported as a declared NOT-RUN with that reason.
SCOPE_TO_SPEC = {
    "value": "value-object_hue-0",
    "conjunction": None,
    "conjunction_live": "conj-object_hue0-shape2",
    "item": "item-0.01",
}

#: The scored artefact that carries the ALREADY-CHARGED T7 row for each seed: the
#: before-deletion `cov_tensor` arm, all four charter-C controls, the rival as-trained
#: column, the `table` floor and the four-part byte ledger.
SCORED = {
    0: W12_OUT / "scored" / "run1_seed0_K32",
    1: W12B_OUT / "scored" / "run3_seed1_K32",
    2: W12B_OUT / "scored" / "run3_seed2_K32",
}

ITEM_MASK_CAVEAT = (
    "THE ARM AND THE COMPARATOR DELETE DIFFERENT ROWS ON THIS SCOPE, at every seed "
    "including 0, and it is structural rather than a seed-offset slip. The arm's 1 % "
    "mask indexes the 131,072-row partition pool (w12_deletion_rows.py: "
    "np.random.default_rng(20_260_921 + seed)); the comparator's indexes its own "
    "16,384-row `fit_rows` (w12_deletion_baselines.partition_fit_rows: "
    "np.random.default_rng(20260921)). Two different index spaces. w12's mask is kept "
    "BYTE-FOR-BYTE because this leg is a pure replication of run 1/3; the mismatch is "
    "printed here rather than silently re-pinned. Both are 'a random 1 % of the fit "
    "pool'; neither is the other's rows.")

RIVAL_SCALE_CAVEAT = (
    "AGAINST THIS HUB, and printed on every rival row: the comparators were fitted at "
    "BASELINES-READY.json -> per_seed.<s>.build_scale.n_fit_rows_used = 16,384 while "
    "the store is fitted on the full 131,072-row pool. A scale-matched rival would be "
    "STRONGER. This sits on top of the already-declared 1x problem (AMENDMENT 1 A1.2): "
    "the comparator set is matched to param_target = 131,072, i.e. the 524,288 B "
    "REFERENCE column, NEVER to this object's own four-part total. No row here is "
    "reported as a 1x result.")


def rival_cost_lines(seed: int) -> dict:
    """Every comparator's own cost line, quoted from the spoke's gate file."""
    doc = json.loads(BASELINES_READY.read_text(encoding="utf-8"))
    recs = doc["per_seed"][str(seed)]["records"]
    out = {}
    for key, rec in recs.items():
        out[key] = {"kind": rec.get("kind"), "arm": rec.get("arm"),
                    "param_count": rec.get("param_count"),
                    "param_error": rec.get("param_error"),
                    "spec": rec.get("spec"), "cost": rec.get("cost"),
                    "smoke": rec.get("smoke"),
                    "checkpoint": rec.get("checkpoint")}
    return out


def _load_scored(seed: int) -> dict:
    return json.loads((SCORED[seed] / "scored.json").read_text(encoding="utf-8"))


def _sink(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")


def score_rivals(seed, spec_label, corner_ids, corners, grid, judge, bitmap,
                 seen_mask, out_path, out_doc):
    """The RIVAL deletion column for one scope: retrain-without, scored on the same
    survivor corners as the arm.  Retrain-without is the ONLY deletion mechanism a
    learned baseline gets here; its cost line is its own and is printed with it."""
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.eval.rivals.w12_deletion_baselines import load_comparator
    from w12_scored_leg import score_corners_full, summarise

    keep = set(corner_ids)
    sel = [c for c in corners if c["corner_id"] in keep]
    res = {}
    for fam in RIVALS:
        name = f"{fam}__retrain_without__{spec_label}"
        t0 = time.time()
        try:
            rv = load_comparator(seed, name, root=COMPARATORS_ROOT)
        except Exception as exc:                                   # noqa: BLE001
            res[fam] = {"DECLARED_NOT_RUN": f"{type(exc).__name__}: {exc}",
                        "checkpoint_name": name}
            log(f"    {name:52s} DECLARED NOT-RUN: {exc}")
            _sink(out_path, out_doc)
            continue
        robj = {"arm": name, "read": "rival", "rival": rv, "rival_partial": None,
                "store": None, "head_bytes": {"head_bytes": 0}}
        done = score_corners_full(robj, sel, grid, judge, bitmap, name)
        rows = fw._per_corner(sel, done, grid, seen_mask)
        meta = getattr(rv, "meta", None) or {}
        res[fam] = {
            "checkpoint_name": name,
            "deletion_mechanism": ("RETRAIN-WITHOUT -- refit from scratch on the pool "
                                   "minus the deleted rows. The only deletion mechanism "
                                   "a learned baseline gets on this table."),
            "param_count": meta.get("param_count"),
            "survivors": summarise(rows),
            "per_corner": rows,
            "score_seconds": round(time.time() - t0, 2),
        }
        log(f"    {name:52s} novel/corner "
            f"{res[fam]['survivors']['novel_per_corner_mean']:8.3f}")
        _sink(out_path, out_doc)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--stage", default="full", choices=("full", "rivals"),
                    help="full = arm downdate legs + rival column; rivals = rival "
                         "column only, reusing an existing deletion.json for the arm")
    ap.add_argument("--reuse-arm", default=None)
    ap.add_argument("--arm-json", default=None,
                    help="--stage rivals: the deletion.json whose arm legs are reused")
    a = ap.parse_args(argv)

    from chlu.core.downdate_store import (carry_address_book, coupling_index,
                                          coupling_layout)
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    from chlu.experiments.exp_w8_single_design import CovContentStore
    from w12_scored_leg import score_corners_full, summarise

    seed, Kstar = int(a.seed), int(a.K)
    fw.PINS["condition"] = "test"
    fw.PINS["seed"] = seed
    man = assert_generation2(seed)
    _z, dead, pool = load_partition("test")
    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    judge = fw.Judge(seed, log=lambda *_: None)
    fitted = cond["train_rows"]
    corners = fw.pair_subcorners(cond) + fw.nway_corners(cond)
    bitmap = fw.value_presence_bitmap(grid, fitted)
    seen_mask = fw.seen_tuple_mask(grid, fitted)
    log(f"condition=test seed={seed} K={Kstar} corners={len(corners)} "
        f"stage={a.stage} cache={man}")

    scored = _load_scored(seed)
    costs = rival_cost_lines(seed)
    out_path = OUT / "deletion" / f"test_seed{seed}_K{Kstar}" / (
        "deletion.json" if a.stage == "full" else "rivals.json")

    doc = {
        "SCORED_RUN_SPENT": 0,
        "NOT_A_NEW_SCORED_RUN": (
            "single-design is EXHAUSTED at 3/3. /c/chlu/SCORED-LEDGER.md rows 43 and 44 "
            "(2026-09-21T21:33:51Z) charge w12b hub/nr6-and-kstar with single-design "
            "2/3 and 3/3, and 3/3 states it 'adds seeds 1 and 2 at K* = 32 on TEST'. "
            "This leg completes RESULT 3 (the deletion rows) of runs ALREADY CHARGED, "
            "on the same fitted arms, same K, same condition, same PREREG. A ledger "
            "NOTE -- not a run row -- was appended at 2026-09-22T21:31:47Z before any "
            "number here."),
        "prereg": ("outputs/w12-single-design-crossover/"
                   "PREREG-w12-single-design-crossover-1.md + AMENDMENT 1"),
        "condition": "test", "seed": seed, "K": Kstar, "stage": a.stage,
        "feature_cache": man,
        "carried_scored_artefact": str(SCORED[seed] / "scored.json"),
        "exactness_definition": (
            "STORE-SCOPED: downdate(delete rows) equals the store fitted from scratch "
            "without those rows, same partition, same rank. THE FROZEN ENCODER AND THE "
            "FROZEN DECODER ARE DECLARED OUTSIDE THE DELETABLE OBJECT AND ARE NOT "
            "REFIT."),
        "adaptation_language": (
            "post-deletion adaptation is LIGHT and DATA-FREE -- the covariance and "
            "count are downdated by subtraction, the atoms are re-eigendecomposed from "
            "the downdated statistic, the address book is carried across the basis "
            "change, and NO ROW OF DATA IS READ. It is never 'no changes'."),
        "rival_scale_caveat": RIVAL_SCALE_CAVEAT,
        "control_gate_source": {
            "where": str(SCORED[seed] / "scored.json"),
            "note": ("the four charter-C controls are MEASURED at this seed in the "
                     "scored artefact above -- not assumed from seed 0 -- and the "
                     "degraded arms are re-scored in-process here for the sphere "
                     "caveat column"),
            "controls_novel_per_corner": {
                nm: {"row3_pair": scored["per_arm"][nm]["row3_pair"][
                         "novel_per_corner_mean"],
                     "row4_nway": scored["per_arm"][nm]["row4_nway"][
                         "novel_per_corner_mean"],
                     "all": scored["per_arm"][nm]["all"]["novel_per_corner_mean"]}
                for nm in ("landscape_removed", "landscape_shuffled",
                           "landscape_shuffled_across_wells", "random_atoms")},
            "cov_tensor_novel_per_corner_all": scored["per_arm"]["cov_tensor"]["all"][
                "novel_per_corner_mean"],
        },
        "rival_as_trained_carried": {
            f"{f}__as_trained": {
                "all": scored["per_arm"][f"{f}__as_trained"]["all"],
                "role": ("the learned arm BEFORE any deletion -- printed beside the "
                         "retrain-without column, never as a deletion mechanism"),
                "cost": (costs.get(f"{f}__as_trained@value-object_hue-0") or {}
                         ).get("cost"),
            } for f in RIVALS if f"{f}__as_trained" in scored["per_arm"]},
        "scopes": {},
    }
    _sink(out_path, doc)

    # ---- the arm side --------------------------------------------------------
    if a.stage == "rivals":
        arm = json.loads(Path(a.arm_json).read_text(encoding="utf-8"))
        doc["arm_legs_reused_from"] = str(a.arm_json)
        doc["ledger_four_part"] = arm["ledger_four_part"]
        doc["before"] = {"all": arm["before"]["all"],
                         "shape_report": arm["before"]["shape_report"]}
        doc["sphere_caveat"] = arm["sphere_caveat"]
        scope_info = {}
        for s, v in arm["scopes"].items():
            doc["scopes"][s] = {
                "carried_from": str(a.arm_json),
                "scope_definition": v["scope_definition"],
                "VACUOUS": v["VACUOUS"], "vacuous_reason": v.get("vacuous_reason"),
                "n_rows_deleted": v["n_rows_deleted"],
                "n_survivor_corners": v["n_survivor_corners"],
                "n_deleted_corners": v["n_deleted_corners"],
                "our_cost_line": v["our_cost_line"],
                "exactness_store_scoped": {
                    k2: v["exactness_store_scoped"].get(k2) for k2 in
                    ("max_abs_mode_dev", "max_subspace_projector_dist", "ranks_agree",
                     "refit_without_cost_line")},
                "after": {"survivors": v["after"]["survivors"],
                          "deleted": v["after"]["deleted"]},
                "before_same_corners": v["before_same_corners"],
                "paired_after_minus_before_on_survivors":
                    v["paired_after_minus_before_on_survivors"],
            }
            scope_info[s] = {"survivor_corner_ids": v["survivor_corner_ids"],
                             "after_per_corner": v["after"]["per_corner"]}
        _sink(out_path, doc)
    else:
        F = fw.features_mmap(seed)
        X = np.asarray(fw.gather(F, pool), np.float32)
        idx = W12D.factor_indices(pool, cards)
        red = idx[:, W12D.RED_FACTOR] == W12D.RED_INDEX
        masks = {
            "value": red,
            "conjunction": red & (idx[:, W12D.SHAPE_FACTOR] == W12D.CUBE_INDEX),
            "conjunction_live": red & (idx[:, W12D.SHAPE_FACTOR] == W12D.SPHERE_INDEX),
        }
        shape_counts = {nm: int((red & (idx[:, W12D.SHAPE_FACTOR] == si)).sum())
                        for si, nm in ((0, "cube"), (1, "cylinder"), (2, "sphere"),
                                       (3, "oblong"))}
        log(f"  red AND <shape> rows in the fit pool: {shape_counts}")
        rng = np.random.default_rng(20_260_921 + seed)   # w12's mask, byte-for-byte
        m_item = np.zeros(pool.size, bool)
        m_item[rng.choice(pool.size, int(round(W12D.ITEM_FRACTION * pool.size)),
                          replace=False)] = True
        masks["item"] = m_item
        for k, m in masks.items():
            log(f"  scope {k:17s}: {int(m.sum()):,} of {pool.size:,} fit rows "
                f"({100.0 * m.mean():.2f} %)")

        reuse = Path(a.reuse_arm or str(SCORED[seed])) / "arm.pkl"
        with open(reuse, "rb") as fh:
            saved = pickle.load(fh)
        st, addr, basis = saved["store"], saved["address"], saved["basis"]
        wc = [np.asarray(c, np.int64) for c in saved["well_channels"]]
        A = int(saved["A"])
        if int(saved["K"]) != Kstar:
            raise SystemExit(f"--reuse-arm holds K={saved['K']}, not K={Kstar}")
        log(f"  reusing the fitted arm from {reuse} (K={saved['K']}, A={A})")

        abytes = int(addr.byte_ledger()["address_bytes"])
        doc["ledger_four_part"] = four_part_ledger(st, abytes)
        doc["A_requested"] = A
        obj = {"arm": "cov_tensor", "read": "tensor", "kind": "span", "store": st,
               "basis": basis, "address": addr, "protos": None,
               "head_bytes": {"head_bytes": 0}}
        before = score_corners_full(obj, corners, grid, judge, bitmap,
                                    "BEFORE deletion")
        rows_before = fw._per_corner(corners, before, grid, seen_mask)
        by_id_before = {r["corner_id"]: r for r in rows_before}
        shape_before = W12D.shape_report(before, corners, grid)

        # INTEGRITY: the reused arm must reproduce the already-charged T7 row exactly.
        ref = {r["corner_id"]: r for r in scored["per_arm"]["cov_tensor"]["per_corner"]}
        dev = max(abs(float(r["distinct_valid_novel_tuples"])
                      - float(ref[r["corner_id"]]["distinct_valid_novel_tuples"]))
                  for r in rows_before)
        doc["before_reproduces_the_scored_row"] = {
            "max_abs_per_corner_deviation": dev,
            "identical": bool(dev == 0.0),
            "against": str(SCORED[seed] / "scored.json"),
            "what": ("the before-deletion arm is re-scored here from the SAME arm.pkl "
                     "and must reproduce the already-charged T7 per-corner row exactly; "
                     "a non-zero deviation would mean the deletion legs below are not "
                     "measured against the published row")}
        log(f"  BEFORE reproduces the scored row: max dev {dev}")
        doc["before"] = {"all": summarise(rows_before), "shape_report": shape_before,
                         "per_corner": rows_before}
        _sink(out_path, doc)

        from w12_crossover_lib import ShimStore, perturb_modes, qr_basis
        degraded = {}
        for nm, kind in (("landscape_removed", "zero"),
                         ("random_atoms", "random_span")):
            mods = perturb_modes(st.modes, kind, seed)
            sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
            cobj = (dict(obj, store=sh, basis=basis, zeroed=True, arm=nm)
                    if kind == "zero" else
                    dict(obj, store=sh,
                         basis=qr_basis(mods, st.deployed_rank,
                                        widths=[b.shape[1] for b in basis]), arm=nm))
            dd = score_corners_full(cobj, corners, grid, judge, bitmap,
                                    f"DEGRADED {nm}")
            degraded[nm] = W12D.shape_report(dd, corners, grid)
            log(f"  SHAPE ({nm}) sphere excess "
                f"{degraded[nm]['judged_sphere_excess']:+.4f}")
        doc["sphere_caveat"] = {
            "HEAD_RULING": ("the conjunction pin STAYS object_hue = 0.0 AND shape = 2.0 "
                            "-- the RED SPHERE -- and this caveat column is printed "
                            "with it"),
            "source": "chlu/data/gen_value_holdout.py:299",
            "what_it_threatens": ("an arm DAMAGED by the red-sphere deletion could be "
                                  "judged sphere for rendering reasons and score "
                                  "spuriously well on exactly the deleted conjunction"),
            "measured_upper_bound_degraded_arms": degraded,
        }
        _sink(out_path, doc)

        scope_info = {}
        for scope in ("value", "conjunction", "conjunction_live", "item"):
            m = masks[scope]
            log(f"\n===== SCOPE {scope.upper()} -- {int(m.sum()):,} rows =====")
            st2 = copy.deepcopy(st)
            U_pre = pinned_basis(st2)
            t0 = time.time()
            receipt = st2.delete_rows(X[m])
            U_post_raw = pinned_basis(st2)
            U_post, width_note = W12D.match_basis_width(U_pre, U_post_raw)
            addr2, carry = carry_address_book(addr, U_pre, U_post)
            t_adapt = time.time() - t0
            cost_ours = W12D.our_cost_line(st, st2, carry, t_adapt)
            log(f"  OUR ADAPTATION  {cost_ours['seconds']}s  "
                f"{cost_ours['bytes_rewritten']:,} B rewritten  "
                f"{cost_ours['data_rows_read']} data rows read")

            t0 = time.time()
            refst = CovContentStore(X[~m], wc, A, D)
            t_refit = time.time() - t0
            devs, sub = [], []
            for k in range(Kstar):
                r = min(st2.deployed_rank[k], refst.deployed_rank[k])
                devs.append(float(np.max(np.abs(st2.modes[k] - refst.modes[k]))))
                if r > 0:
                    Ud = np.linalg.qr(
                        np.asarray(st2.modes[k][1:1 + r], np.float64).T)[0]
                    Ur = np.linalg.qr(
                        np.asarray(refst.modes[k][1:1 + r], np.float64).T)[0]
                    sub.append(float(np.linalg.norm(Ud @ Ud.T - Ur @ Ur.T, 2)))
            log(f"  EXACTNESS (store-scoped)  max|mode dev| {max(devs):.3e}  "
                f"max subspace dist {(max(sub) if sub else float('nan')):.3e}  "
                f"ranks agree {st2.deployed_rank == refst.deployed_rank}  "
                f"(refit-without took {t_refit:.1f}s, read {int((~m).sum()):,} rows)")

            # ⛔ A DEFECT OF w12's HARNESS, FOUND HERE AND FIXED BEFORE ANY NUMBER.
            # w12_deletion_rows.py read the post-deletion store through `U_post_raw`
            # while it carried the address book onto the WIDTH-MATCHED `U_post`. Where
            # the two widths agree -- seed 0, every scope, `basis_width_adjustments`
            # `[]` -- those are the same array and nothing changes. At seed 1 they do
            # NOT agree: after the red rows go, the mean's perpendicular part falls
            # below `pinned_basis`'s 1e-4 admission threshold in one well, so the raw
            # post basis is 43 columns wide against the book's 44, and `render_span`
            # dies in `C @ Uk.T` ("size 43 is different from 44"). The book must be
            # read through the basis it was carried onto, so `U_post` is what the read
            # gets. The padding column is exactly zero, so the book's coefficient for a
            # direction the store no longer holds is projected onto nothing -- which is
            # what the deletion MEANS, and it is recorded on the row in
            # `basis_width_adjustments` rather than done silently.
            obj2 = dict(obj, store=st2, basis=U_post, address=addr2,
                        arm=f"cov_tensor_after_{scope}")
            after = score_corners_full(obj2, corners, grid, judge, bitmap,
                                       f"AFTER {scope}")
            rows_after = fw._per_corner(corners, after, grid, seen_mask)
            by_id_after = {r["corner_id"]: r for r in rows_after}

            if scope == "value":
                surv = [c["corner_id"] for c in corners
                        if not W12D.corner_requests_value(c, grid, W12D.RED_FACTOR,
                                                          W12D.RED_INDEX)]
            elif scope in ("conjunction", "conjunction_live"):
                si = W12D.CUBE_INDEX if scope == "conjunction" else W12D.SPHERE_INDEX
                surv = [c["corner_id"] for c in corners
                        if not (W12D.corner_requests_value(c, grid, W12D.RED_FACTOR,
                                                           W12D.RED_INDEX)
                                and W12D.corner_requests_value(c, grid,
                                                               W12D.SHAPE_FACTOR, si))]
            else:
                surv = [c["corner_id"] for c in corners]
            keep = set(surv)
            dele = [c["corner_id"] for c in corners if c["corner_id"] not in keep]

            paired = (fw.paired_estimand([by_id_after[i] for i in surv],
                                         [by_id_before[i] for i in surv],
                                         "distinct_valid_novel_tuples", 1.0,
                                         f"after_{scope}", "before")
                      if surv else {"DECLARED_NOT_RUN": "no survivor corner"})
            rec = {
                "scope_definition": {
                    "value": "every fit row carrying red (object_hue index 0)",
                    "conjunction": ("every fit row carrying red AND cube (shape index "
                                    "0) -- AS LITERALLY SPECIFIED"),
                    "conjunction_live": ("every fit row carrying a RED SPHERE "
                                         "(object_hue index 0 AND shape index 2) -- the "
                                         "declared non-vacuous pin, named 'red sphere' "
                                         "everywhere"),
                    "item": "a random 1 % of fit rows",
                }[scope],
                "VACUOUS": bool(int(m.sum()) == 0),
                "vacuous_reason": (
                    ("shape=cube is held out of the value-holdout split entirely, so "
                     "the fit pool contains ZERO cubes and this deletion removes 0 "
                     "rows. Reported as a count of 0 with its reason, NEVER as a "
                     "passing row. The non-vacuous pin is `conjunction_live` = the RED "
                     "SPHERE.") if int(m.sum()) == 0 else None),
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
                    "ranks_agree": bool(st2.deployed_rank == refst.deployed_rank),
                    "refit_without_cost_line": {
                        "arm": "refit-from-scratch-without (the reference)",
                        "seconds": round(t_refit, 3),
                        "data_rows_read": int((~m).sum()),
                        "bytes_rewritten": int(
                            sum(M.size for M in refst.modes) * 4
                            + sum(w.stat_bytes() for w in refst.stats)),
                        "refit": True},
                },
                "shape_report_after": W12D.shape_report(after, corners, grid),
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
            if scope == "item":
                rec["item_mask_caveat"] = ITEM_MASK_CAVEAT
            doc["scopes"][scope] = rec
            scope_info[scope] = {"survivor_corner_ids": surv,
                                 "after_per_corner": rows_after}
            _sink(out_path, doc)
            del st2, refst

        # DISPLAY: the coupling / hole table.  MECHANISM (ii), a DIFFERENT array.
        lay = coupling_layout(cards)
        hole = np.zeros(lay["size"], np.uint8)
        hidx = coupling_index(cards, W12D.RED_FACTOR, W12D.RED_INDEX,
                              W12D.SHAPE_FACTOR, W12D.CUBE_INDEX)
        hidx_live = coupling_index(cards, W12D.RED_FACTOR, W12D.RED_INDEX,
                                   W12D.SHAPE_FACTOR, W12D.SPHERE_INDEX)
        t0 = time.time()
        hole[hidx] = 1
        hole[hidx_live] = 1
        t_hole = time.time() - t0
        from chlu.core.downdate_store import assert_layout_matches_field_store
        doc["display_refusal"] = {
            "DISPLAY_NEVER_A_RESULT": True,
            "layout_is_the_field_stores": assert_layout_matches_field_store(cards),
            "holes": [
                {"index": int(hidx), "label": "red AND cube",
                 "pair": [["object_hue", 0], ["shape", 0]]},
                {"index": int(hidx_live), "label": "red sphere -- the live pin",
                 "pair": [["object_hue", 0], ["shape", 2]]}],
            "cost_line": {"arm": "cov_tensor (ours)", "seconds": round(t_hole, 6),
                          "bytes_rewritten": 2, "data_rows_read": 0, "refit": False},
            "n_holes_after": int(hole.sum()),
            "coupling_table_bytes": int(lay["size"]),
            "comparator_status": ("DECLARED NOT-RUN -- the only permitted comparator is "
                                  "the refuse-token transformer, and wave 9 measured a "
                                  "refuser of this quality at +0.047 separation with a "
                                  "ridge probe at AUC 0.48-0.55 (chance), so the "
                                  "comparison is not yet informative"),
            "what_this_row_is_for": (
                "A compositional store can still compose 'red sphere' from 'red' and "
                "'sphere' after the red-sphere rows are gone. THAT IS COMPOSITION "
                "WORKING, NOT DELETION FAILING. The hole is the SEPARATE mechanism that "
                "makes the request refusable. Both are reported, separately, on one "
                "object."),
        }
        _sink(out_path, doc)

    # ---- the RIVAL deletion column ------------------------------------------
    doc["rival_deletion_column"] = {}
    for scope, spec in SCOPE_TO_SPEC.items():
        if spec is None:
            doc["rival_deletion_column"][scope] = {
                "DECLARED_NOT_RUN": (
                    "red AND cube deletes 0 rows, and "
                    "w12_deletion_baselines.partition_fit_rows REFUSES a no-op spec by "
                    "construction ('a deletion that removes zero rows is not a "
                    "deletion'). No comparator checkpoint exists or can exist for this "
                    "scope. Reported as a declared NOT-RUN, never a zero.")}
            _sink(out_path, doc)
            continue
        surv = scope_info[scope]["survivor_corner_ids"]
        log(f"\n===== RIVAL COLUMN, SCOPE {scope.upper()} "
            f"({len(surv)} survivor corners) =====")
        entry = {"comparator_spec_label": spec,
                 "n_survivor_corners": len(surv),
                 "deletion_mechanism_allowed": ("RETRAIN-WITHOUT ONLY. A learned "
                                                "baseline deletes by refitting from "
                                                "scratch on the pool minus the deleted "
                                                "rows; its cost line is printed."),
                 "scale_caveat": RIVAL_SCALE_CAVEAT}
        if scope == "item":
            entry["item_mask_caveat"] = ITEM_MASK_CAVEAT
        doc["rival_deletion_column"][scope] = entry
        _sink(out_path, doc)
        entry["retrain_without"] = score_rivals(
            seed, spec, surv, corners, grid, judge, bitmap, seen_mask, out_path, doc)
        entry["retrain_without_cost_lines"] = {
            f: (costs.get(f"{f}__retrain_without__{spec}") or {}).get("cost")
            for f in RIVALS}
        entry["as_trained_cost_lines"] = {
            f: (costs.get(f"{f}__as_trained@{spec}") or {}).get("cost")
            for f in RIVALS}
        af = {r["corner_id"]: r for r in scope_info[scope]["after_per_corner"]}
        ours = [af[i] for i in surv]
        entry["ours_after_deletion_on_the_same_survivors"] = summarise(ours)
        entry["paired_ours_minus_rival"] = {}
        for f in RIVALS:
            rr = entry["retrain_without"].get(f, {})
            if "per_corner" not in rr:
                entry["paired_ours_minus_rival"][f] = {
                    "DECLARED_NOT_RUN": rr.get("DECLARED_NOT_RUN")}
                continue
            entry["paired_ours_minus_rival"][f] = fw.paired_estimand(
                ours, rr["per_corner"], "distinct_valid_novel_tuples", 1.0,
                f"cov_tensor_after_{scope}", f"{f}__retrain_without__{spec}")
        _sink(out_path, doc)

    doc["DONE"] = True
    _sink(out_path, doc)
    log(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
