"""hub/store-content — fill all SEVEN scoreboard rows for every arm.

⛔ Rows 1-4 are this hub's target.  Rows 5, 6 and 7 are filled because w3-hub-common
§2 makes the four wave-3 hubs comparable only if every hub fills all seven, and they
are **reported, never claimed** — w3-hub-common §7 forbids quoting a row-5 or row-6
number as evidence for the store, and `build_scoreboard` itself refuses to phrase a
row-6 verdict as evidence.

⛔ The scoring is `chlu.experiments.gen_scoreboard`, called, never re-implemented:
`make_outcome` -> `build_scoreboard` -> `write_scoreboard`.  The laundering control
`grid_syntax_rule` is supplied because the module refuses a table without it.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import SPECS, LEGAL, open_sources                      # noqa
from arm import build_arm, read_arm, penult_of                      # noqa
from instrument import load_instrument, render, judge, FACTORS      # noqa
from run_content import build_world, request_matrix, seen_tuples, eval_rows  # noqa
from rows567 import fit_threshold, accept_mask, evidence, delete_value        # noqa
from controls import BasisHead, KnnProto, onehot                    # noqa

from chlu.experiments import gen_scoreboard as SB                   # noqa
from chlu.core.gen_request import load_c_requests                   # noqa

OUT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "hub-store-content"
VH = Path(os.environ.get("CHLU_GEN_DATA_ROOT", "C:/chlu/data/gen")) / "shapes3d" / "value_holdout"


def _terms_to_indices(records, grid) -> tuple:
    """Query-set records -> (n, F) factor-index matrix + an off-grid mask.

    ⛔ A term whose value is not on the grid gets ``-1`` and is counted: rows 2 and 6
    are off-grid BY DEFINITION, and silently snapping them to the nearest grid value
    would convert an unsatisfiable request into a satisfiable one.
    """
    F = grid.n_factors
    R = -np.ones((len(records), F), np.int64)
    offgrid = np.zeros(len(records), np.int64)
    for i, rec in enumerate(records):
        for t in rec["request"]:
            k = int(t["factor_index"])
            hit = np.flatnonzero(grid.values[k] == float(t["value"]))
            if hit.size:
                R[i, k] = int(hit[0])
            else:
                offgrid[i] += 1
    return R, offgrid


def load_rowsets(grid) -> dict:
    out = {}
    for key, fn in (("row1_seen", "row1_seen.json"), ("row2_unseen", "row2_unseen.json"),
                    ("row5_held_out", "row5_held_out.json"), ("c_invalid", "c_invalid.json")):
        d = json.load(open(VH / fn))
        R, off = _terms_to_indices(d["records"], grid)
        out[key] = {"records": d["records"], "R": R, "n_offgrid": off}
    return out


def ledger_for(row_id, base_ledger, deleted_ledger=None):
    return dict(deleted_ledger if (row_id == "r7_deletion" and deleted_ledger)
                else base_ledger)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="P,WIDEN,WIDEN_SHUF")
    ap.add_argument("--primary", default="WIDEN")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--tag", default="scoreboard")
    ap.add_argument("--max-per-corner", type=int, default=0,
                    help="PLUMBING TEST ONLY: cap requests per corner. A capped run "
                         "is written with plumbing_test=true and is never a result.")
    a = ap.parse_args(argv)
    if not Path(a.prereg).exists():
        print("REFUSING: --prereg must point at an existing PREREG file")
        return 3

    rdir = OUT / "runs" / a.tag
    rdir.mkdir(parents=True, exist_ok=True)
    inst = load_instrument(a.seed)
    world = build_world(a.seed)
    grid = world["grid"]
    src = open_sources(a.seed)
    pool = world["fit_rows"]
    seen = seen_tuples(grid, pool)
    rs = load_rowsets(grid)

    r3_rows, r3_cid, _ = eval_rows(world, "pairs")
    r4_rows, r4_cid, _ = eval_rows(world, "corners")
    if a.max_per_corner:
        def _cap(rows, cid, k):
            keep = []
            rng = np.random.default_rng(20_260_919)
            for c in dict.fromkeys(cid.tolist()):
                idx = np.flatnonzero(cid == c)
                keep.append(idx if idx.size <= k else rng.choice(idx, k, replace=False))
            sel = np.sort(np.concatenate(keep))
            return rows[sel], cid[sel]
        r3_rows, r3_cid = _cap(r3_rows, r3_cid, a.max_per_corner)
        r4_rows, r4_cid = _cap(r4_rows, r4_cid, a.max_per_corner)
        print(f"  PLUMBING TEST: rows 3/4 capped to {a.max_per_corner}/corner "
              f"-> {r3_rows.size} / {r4_rows.size}")
    R3, R4 = request_matrix(grid, r3_rows), request_matrix(grid, r4_rows)
    ROWREQ = {"r1_seen": rs["row1_seen"]["R"], "r2_unseen_valid": rs["row2_unseen"]["R"],
              "r3_comp_2way": R3, "r4_comp_nway": R4,
              "r5_comp_refusal": rs["row5_held_out"]["R"],
              "r6_invalid_refusal": rs["c_invalid"]["R"]}

    outcomes, deletion, extra = {}, {}, {}

    def score(name, reader, ledger, thr, setting, accept_fn, deleted=None):
        rows = {}
        for rid, R in ROWREQ.items():
            x = reader(R)
            imgs = render(inst["decoder"], x)
            pred = judge(inst["classifier"], imgs)
            acc = accept_fn(R)
            valid = acc & (pred == np.where(R >= 0, R, pred)).all(axis=1)
            rows[rid] = SB.make_outcome(
                accept=acc, valid=valid, tuples=np.where(acc[:, None], pred, -1),
                threshold=float(thr), setting=setting,
                byte_ledger=ledger_for(rid, ledger))
            del imgs
        # row 7: the SAME object after deletion, no retraining.
        # ⛔ `score_deletion_row` takes full OUTCOME objects, not summaries: the
        # survivor half is rows 3-4's novelty metric re-measured post-deletion, the
        # deleted half is row 5's refusal pair re-measured post-deletion, and `pre`
        # is row 5 BEFORE deletion so `delta_reject` can be formed. `delta_reject`
        # is the one quantity a no-model grid rule cannot produce, because deleting
        # a value from the store does not move the factor grid.
        if deleted is not None:
            dread, dledger, dacc = deleted
            surv_pred = judge(inst["classifier"], render(inst["decoder"], dread(R4)))
            av = dacc(R4)
            vv = av & (surv_pred == R4).all(axis=1)
            survivor = SB.make_outcome(
                accept=av, valid=vv, tuples=np.where(av[:, None], surv_pred, -1),
                threshold=float(thr), setting=setting,
                byte_ledger=ledger_for("r7_deletion", ledger, dledger))
            Rd = ROWREQ["r5_comp_refusal"]
            del_pred = judge(inst["classifier"], render(inst["decoder"], dread(Rd)))
            ad = dacc(Rd)
            vd = ad & (del_pred == np.where(Rd >= 0, Rd, del_pred)).all(axis=1)
            deleted_out = SB.make_outcome(
                accept=ad, valid=vd, tuples=np.where(ad[:, None], del_pred, -1),
                threshold=float(thr), setting=setting,
                byte_ledger=ledger_for("r7_deletion", ledger, dledger))
            rows["r7_deletion"] = survivor
            deletion[name] = {
                "survivor": survivor, "deleted": deleted_out,
                "pre": rows["r5_comp_refusal"],
                # ⭐ EQUAL by construction: nothing is refitted between the two
                # measurements, and score_deletion_row checks that equality rather
                # than believing the prose.
                "fit_fingerprint": (json.dumps(setting, sort_keys=True),
                                    json.dumps(setting, sort_keys=True))}
        else:
            n4 = ROWREQ["r4_comp_nway"].shape[0]
            rows["r7_deletion"] = SB.make_outcome(
                accept=np.zeros(n4, bool), valid=np.zeros(n4, bool),
                tuples=-np.ones((n4, grid.n_factors), np.int64),
                threshold=float(thr), setting=setting,
                byte_ledger=ledger_for("r7_deletion", ledger))
        outcomes[name] = rows

    # ---- the content arms -------------------------------------------------
    vcfg = world["vcfg"]
    del_factor, del_value = "object_hue", vcfg.held_out_hue_value
    del_idx = int(np.flatnonzero(grid.values[grid.index_of(del_factor)]
                                 == float(vcfg.row1_seen_hue_value))[0])
    for name in [s.strip() for s in a.arms.split(",") if s.strip()]:
        t0 = time.time()
        spec = SPECS[name]
        armo = build_arm(spec, pool, request_matrix(grid, pool),
                         list(grid.cardinalities), seed=a.seed, src=src)
        th = fit_threshold(armo.book, request_matrix(grid, pool))
        setting = {"arm": name, "launch_mode": "clamp_atom", "settling_steps": 0,
                   "threshold_quantile": th["quantile"], "D": spec.D, "N": spec.n_atoms}
        darm, dinfo = delete_value(armo, grid, pool, del_factor, del_idx)
        score(name,
              reader=lambda R, A=armo: penult_of(A, read_arm(A, R, "full")["x"]),
              ledger=armo.ledger, thr=th["threshold"], setting=setting,
              accept_fn=lambda R, A=armo, t=th: accept_mask(A.book, R, t["threshold"]),
              deleted=(lambda R, A=darm: penult_of(A, read_arm(A, R, "full")["x"]),
                       darm.ledger,
                       lambda R, A=darm, t=th: accept_mask(A.book, R, t["threshold"])))
        extra[name] = {"threshold": th, "deletion": dinfo,
                       "grouping": armo.grouping, "spec": spec.assert_matched(),
                       "wall_seconds": float(time.time() - t0)}
        print(f"  [{name}] seven rows done ({time.time()-t0:.0f}s)")

    # ---- the two mandatory controls ---------------------------------------
    from run_content import build_Z_for
    Zp, scales = build_Z_for(SPECS["P"], pool, a.seed, src)
    Rfit = request_matrix(grid, pool)

    bh = BasisHead(Zp, Rfit, list(grid.cardinalities), SPECS["P"].n_atoms, seed=a.seed)
    bh_s = np.linalg.norm(onehot(Rfit, [int(c) for c in grid.cardinalities]) , axis=1)  # noqa
    bh_thr = 0.0   # basis_head has no refusal mechanism: it answers everything
    score("basis_head",
          reader=lambda R: (bh.read(R)[:, :2048] * np.float32(scales["p"])
                            ).reshape((-1, 128, 4, 4)),
          ledger=bh.ledger, thr=bh_thr,
          setting={"arm": "basis_head", "threshold_rule": "none - answers everything",
                   "ridge_proj": 1e-2, "ridge_head": 1e-1},
          accept_fn=lambda R: np.ones(R.shape[0], bool))

    kp = KnnProto(grid, Zp, pool)
    score("knn_proto",
          reader=lambda R: (kp.read(R)[:, :2048] * np.float32(scales["p"])
                            ).reshape((-1, 128, 4, 4)),
          ledger=kp.ledger, thr=bh_thr,
          setting={"arm": "knn_proto", "threshold_rule": "none - answers everything"},
          accept_fn=lambda R: np.ones(R.shape[0], bool))

    # ---- the laundering control: the no-model syntactic rule --------------
    lr = {}
    for rid, R in ROWREQ.items():
        recs = {"r1_seen": rs["row1_seen"]["records"],
                "r2_unseen_valid": rs["row2_unseen"]["records"],
                "r5_comp_refusal": rs["row5_held_out"]["records"],
                "r6_invalid_refusal": rs["c_invalid"]["records"]}.get(rid)
        if recs is not None:
            acc = np.array([SB.grid_syntax_accept(r["request"], grid) for r in recs], bool)
        else:
            acc = np.array([SB.grid_syntax_accept(
                [{"factor": grid.names[k], "factor_index": k,
                  "value": float(grid.values[k][R[i, k]])} for k in range(grid.n_factors)],
                grid) for i in range(R.shape[0])], bool)
        lr[rid] = SB.make_outcome(
            accept=acc, valid=np.zeros(acc.size, bool),
            tuples=-np.ones((acc.size, grid.n_factors), np.int64),
            threshold=0.0, setting={"arm": SB.LAUNDERING_ARM},
            byte_ledger={"deployed_bytes": 0, "budget_bytes": SB.ScoreboardConfig().byte_budget,
                         "what": "no model at all - the request and the grid",
                         "budget_note": "the no-model rule deploys ZERO bytes by construction"})
    lr["r7_deletion"] = lr["r5_comp_refusal"]
    outcomes[SB.LAUNDERING_ARM] = lr

    cfg = SB.ScoreboardConfig(decoration_margin=0.02)
    table = SB.build_scoreboard(a.primary, outcomes, sorted(seen), cfg=cfg,
                                deletion=deletion,
                                provenance={"hub": "hub/store-content",
                                            "plumbing_test": bool(a.max_per_corner),
                                            "n_row3": int(r3_rows.size),
                                            "n_row4": int(r4_rows.size),
                                            "prereg": str(a.prereg),
                                            "instrument_hashes": inst["hashes"],
                                            "arms": extra})
    fn = "scoreboard_PLUMBING_TEST.json" if a.max_per_corner else "scoreboard.json"
    json.dump(table, open(rdir / fn, "w"), indent=1, default=str)
    print(f"  -> {rdir / fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
