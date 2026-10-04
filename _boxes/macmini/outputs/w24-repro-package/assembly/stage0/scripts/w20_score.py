"""w20 hub/shapes-data-matched -- SCORE arms on one corner set, for one seed.

Corner sets
  val        the 24 val corners (free; the store's val pool excludes them)
  rows34     T7's TEST corners: 60 `pair-*` sub-corners (row 3) + 49 `nway-*` (row 4).
             ⛔ GATED: a ledger row `w20 hub/shapes-data-matched | shapes-matched | n/3`
             must already exist for `--run-n`.
  rung:<k>   the composition-ladder rung's corners (a replication, as in w12b)

Arm specs (`--arms`, comma-separated)
  store:<arm.pkl>:<label>     a fitted store; its four controls are scored beside it
                              unless --no-controls.  `landscape_shuffled_across_wells`
                              is a DERANGEMENT and its permutation is PRINTED
                              (`w20_lib.perturb_modes_w20`).
  rival:<tag>:<name>          a w20 data-matched rival (OUT/rivals/<tag>/ckpt)
  w12comp:<name>              an EXISTING w12 deletion-baselines comparator (16,384 rows)
  table                       the prototype table -- a FLOOR, never a rival

Metric: `distinct_valid_novel_tuples` per corner (the programme's estimand), plus the
per-corner CEILING (distinct requested tuples the fit split never holds).
Scoring path: rows34/val use `w12b_scored_leg.score_corners_full` + `fw._per_corner`
(T7's own path, so a re-scored T7 store must reproduce T7's rows exactly); rungs use
`w12b_ladder_leg.score_arm` + FastJudge (asserted identical), the ladder's own path.
Resumable per arm.
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
from w20_lib import (OUT, assert_worktree, jdump, ledger_has, log,  # noqa: E402
                     perturb_modes_w20)

CONTROLS = (("landscape_removed", "zero"),
            ("landscape_shuffled", "value_permute"),
            ("landscape_shuffled_across_wells", "well_permute"),
            ("random_atoms", "random_span"))


def summarise(rows, family=None):
    sel = [r for r in rows if family is None or r["family"] == family]
    if not sel:
        return {"n_corners": 0}
    nov = np.array([r["distinct_valid_novel_tuples"] for r in sel], float)
    cei = np.array([r.get("ceiling", np.nan) for r in sel], float)
    return {"n_corners": len(sel), "novel_per_corner_mean": float(nov.mean()),
            "ceiling_per_corner_mean": float(np.nanmean(cei)),
            "judge_accuracy_mean": float(np.mean([r["judge_accuracy"] for r in sel]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--run-n", type=int, default=0)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--only-controls", default="", help="comma list: score only these controls")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--slow", action="store_true", help="T7's per-corner path + shipped Judge")
    a = ap.parse_args(argv)
    wt = assert_worktree()
    log("chlu.__file__ = %s" % wt)

    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R
    from chlu.experiments.exp_gen_delete_row7 import assert_generation2
    from w12_crossover_lib import ShimStore, qr_basis

    seed = int(a.seed)
    fw.PINS["seed"] = seed
    fw.PINS["condition"] = "val" if a.corners == "val" else "test"
    man = assert_generation2(seed)

    if a.corners == "rows34":
        if not ledger_has(int(a.run_n)):
            raise SystemExit("LEDGER ROW MISSING for run %d/3 -- refusing to score TEST "
                             "corners. The row is written BEFORE the run." % a.run_n)
        log("LEDGER GATE OPEN: run %d/3" % a.run_n)

    rung = a.corners.split(":", 1)[1] if a.corners.startswith("rung:") else None
    if rung is not None:
        import w12b_ladder_leg as LEG
        from chlu.experiments import exp_w5_compose_field as W5
        cond = LEG.make_condition(rung)
        LEG._COND["cond"] = cond
        fw.load_condition = LEG._patched_load_condition
        W5.fw.load_condition = LEG._patched_load_condition
        R.fw.load_condition = LEG._patched_load_condition
        corners = cond["corners"]
        fw.assert_corners_disjoint_and_heldout(corners, cond["test_rows"], rung)
    else:
        cond = fw.load_condition()
        corners = (fw.val_corners(cond) if a.corners == "val"
                   else fw.pair_subcorners(cond) + fw.nway_corners(cond))
    grid = cond["grid"]
    bitmap = fw.value_presence_bitmap(grid, cond["train_rows"])
    if a.corners == "val":
        # the val corners live in the value-holdout TRAIN index and are removed from the
        # val fit pool (`fw.fit_pool(exclude=val_corner_rows)`), so "novel" on val is
        # relative to train_rows MINUS the val corners' rows -- otherwise every val
        # tuple reads as seen and the ceiling is 0 (measured on the first val pass).
        seen_mask = fw.seen_tuple_mask(grid, np.setdiff1d(cond["train_rows"],
                                                          fw.val_corner_rows(cond)))
    else:
        seen_mask = fw.seen_tuple_mask(grid, cond["train_rows"])
    ceiling = {c["corner_id"]: int((~seen_mask[np.unique(np.asarray(c["rows"], np.int64))]).sum())
               for c in corners}
    judge = fw.Judge(seed, log=lambda *_: None)
    if rung is not None or not a.slow:
        # ⭐ rows34/val also use the ladder's batched path + FastJudge unless --slow: w12b
        # measured the batched read IDENTICAL to per-corner reads, and FastJudge is asserted
        # element-for-element identical to the shipped Judge right here, in this process.
        # The T7 reproduction check in w20_tables.py is the end-to-end proof.
        import w12b_ladder_leg as LEG
        from w12_fastjudge import FastJudge, assert_identical
        fast = FastJudge(judge)
        pool0 = (np.load(str(Path("C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")
                             / ("%s_seed0_K32" % rung) / "partitions.npz"))["pool"]
                 if rung is not None else
                 np.load("C:/chlu/outputs/hub-factor-wells/run/partitions.npz")["pool"])
        chk = assert_identical(judge, fast, fw.gather(fw.features_mmap(seed), pool0[:256]))
        log("FastJudge positive control: %s" % chk["identical"])
    log("corners=%s seed=%d: %d corners" % (a.corners, seed, len(corners)))

    def score(obj):
        if rung is not None or not a.slow:
            cond_ = dict(cond, corners=corners)
            per = LEG.score_arm(obj, cond_, fast, bitmap, seen_mask)
        else:
            from w12b_scored_leg import score_corners_full
            done = score_corners_full(obj, corners, grid, judge, bitmap, obj["arm"])
            per = fw._per_corner(corners, done, grid, seen_mask)
        for r in per:
            r["ceiling"] = ceiling[r["corner_id"]]
        return per

    outd = OUT / "scores" / (a.corners.replace(":", "") + a.out_tag) / ("seed%d" % seed)
    outd.mkdir(parents=True, exist_ok=True)

    def emit(label, obj, extra):
        p = outd / ("%s.json" % label)
        if p.exists():
            log("  %-44s already on disk" % label)
            return
        t = time.time()
        per = score(obj)
        s = {"all": summarise(per), "row3_pair": summarise(per, "pair"),
             "row4_nway": summarise(per, "nway")}
        jdump({"label": label, "corners": a.corners, "seed": seed, "cache": man,
               "summary": s, "extra": extra, "per_corner": per,
               "seconds": round(time.time() - t, 1), "chlu_file": wt}, p)
        log("  %-44s novel/corner %.3f (ceiling %.2f)  in %.0fs"
            % (label, s["all"]["novel_per_corner_mean"],
               s["all"]["ceiling_per_corner_mean"], time.time() - t))

    only = [x for x in a.only_controls.split(",") if x]
    for spec in [x for x in a.arms.split(",") if x]:
        kind = spec.split(":", 1)[0]
        if kind == "store":
            path = spec[len("store:"):spec.rfind(":")]
            label = spec[spec.rfind(":") + 1:]
            with open(path, "rb") as fh:
                arm = pickle.load(fh)
            st, addr, basis = arm["store"], arm["address"], arm["basis"]
            obj = {"arm": label, "read": "tensor", "kind": "span", "store": st,
                   "basis": basis, "address": addr, "protos": None,
                   "head_bytes": {"head_bytes": 0}}
            if not only:
                emit(label, obj, {"source": path, "K": len(st.modes)})
            if a.no_controls:
                continue
            for name, ck in CONTROLS:
                if only and name not in only:
                    continue
                mods, info = perturb_modes_w20(st.modes, ck, seed)
                sh = ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
                if ck == "zero":
                    cobj = dict(obj, store=sh, basis=basis, zeroed=True, arm=name)
                else:
                    cobj = dict(obj, store=sh, arm=name,
                                basis=qr_basis(mods, st.deployed_rank,
                                               widths=[b.shape[1] for b in basis]))
                if ck == "well_permute":
                    log("  across-wells DERANGEMENT perm=%s fixed_points=%d | historical "
                        "T7 perm fixed_points=%d" % (info["perm"], info["fixed_points"],
                                                     info["historical_T7_fixed_points"]))
                emit("%s__%s" % (label, name), cobj,
                     {"control": name, "perturbation": info, "source": path,
                      "when": "AFTER the address book was fitted; read through a basis "
                              "of its own atoms at the book's widths (qr_basis)"})
        elif kind in ("rival", "w12comp", "table"):
            from chlu.eval.rivals import w4_api
            from chlu.eval.rivals.w12_deletion_baselines import load_comparator
            if kind == "rival":
                _, tag, name = spec.split(":")
                rv = w4_api.load_rival(seed, name, root=str(OUT / "rivals" / tag / "ckpt"))
                label = "%s@%s" % (name, tag)
            elif kind == "w12comp":
                name = spec.split(":")[1]
                rv = load_comparator(seed, name, root="C:/chlu/outputs/w12-deletion-baselines/comparators")
                label = "w12comp_%s" % name
            else:
                rv = w4_api.load_rival(seed, "table")
                label = "table_FLOOR"
            obj = {"arm": label, "read": "rival", "kind": "rival", "rival": rv,
                   "rival_partial": None, "store": None, "protos": None,
                   "head_bytes": {"head_bytes": 0}}
            emit(label, obj, {"spec": spec, "param_count": (rv.meta or {}).get("param_count"),
                              "meta": {k: v for k, v in (rv.meta or {}).items()
                                       if k not in ("grid_values",)}})
        else:
            raise SystemExit("unknown arm spec %r" % spec)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
