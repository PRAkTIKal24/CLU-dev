"""w15 spoke/coverage-control -- aggregate the val legs into COVERAGE-CONTROL.json.

Reads ``<out>/val/seed<s>_K32/rows_*.json`` (this spoke) and, READ ONLY, the test-split
rung n=3 / n=6 K* legs of ``w12b-nr6-and-kstar`` for context.  Applies the PREREG bars
verbatim.  Writes nothing but ``COVERAGE-CONTROL.json`` in the output dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

OUT = Path("C:/chlu/outputs/w15-coverage-control")
W12B = Path("C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")
SEEDS = (0, 1, 2)
MARGIN = 1.0
NOTE = ("We beat the naive fair rivals at every n; the CLU's own 2-way \u2192 6-way rise is "
        "not to be inferred, because rungs 3\u20136 share corners and differ in coverage.")
CELLS = {"A": (3, 3), "B": (6, 6), "C": (3, 6), "D": (6, 3)}


def load(p: Path):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def per_corner(rows, key="distinct_valid_novel_tuples"):
    return {r["corner_id"]: float(r[key]) for r in rows["per_corner"]}


def main() -> int:
    global SEEDS
    present = tuple(s for s in SEEDS if (OUT / "val" / ("seed%d_K32" % s) / "LEG.json").exists())
    missing = [s for s in SEEDS if s not in present]
    if not present:
        print("no coverage-control leg under %s: run scripts/shapes_coverage_control.py first" % (OUT / "val"))
        return 0
    if missing:
        print("seeds without a coverage-control leg, not aggregated: %s" % missing)
    SEEDS = present
    res = {"what": "n=6 at n=3's coverage vs n=6 vs n=3 -- VAL LEG ONLY",
           "scored_runs_spent": 0,
           "composition_target": "DECLARED EXHAUSTED (ledger L8, L11/12, L15, L27, L30/32: "
                                 "5 spends since 2026-09-16 vs a cap of 3)",
           "prereg": str(OUT / "PREREG.md"),
           "inference_note_verbatim": NOTE,
           "estimand": {"metric": "distinct_valid_novel_tuples", "unit": "corner",
                        "function": "exp_gen_factor_wells.per_corner_scores",
                        "margin": MARGIN},
           "store": "K*=32 CovContentStore, ward_balanced (w12b_ladder_kstar functions)",
           "n_seeds": len(SEEDS), "seeds": list(SEEDS), "seeds_missing": missing,
           "val": {}, "test_context": {}}
    val = {}
    for s in SEEDS:
        leg = OUT / "val" / ("seed%d_K32" % s)
        lj = load(leg / "LEG.json")
        pair = lj["audit"]["corner_pairing_k6_to_k3"]
        row = {"LEG": str(leg / "LEG.json"), "failed": lj["failed"], "store": {},
               "learned_1x": {}, "controls": {}, "store_bytes": {}}
        pcs = {}
        for c, (kp, ke) in CELLS.items():
            f = leg / ("rows_%s_store_pool%d_est%d.json" % (c, kp, ke))
            d = load(f)
            row["store"][c] = {"mean": d["summary"]["distinct_valid_novel_tuples_mean"],
                               "judge_acc": d["summary"]["judge_accuracy_mean"],
                               "n_corners": d["summary"]["n_corners"], "file": str(f)}
            pc = per_corner(d)
            # key every per-corner vector by the est-6 corner id so A/D pair with B/C
            if ke == 3:
                pc = {k6: pc[k3] for k6, k3 in pair.items()}
            pcs[c] = pc
            ctl = {}
            for cn in ("landscape_removed", "landscape_shuffled", "random_atoms"):
                g = leg / ("rows_%s_store_pool%d_est%d__%s.json" % (c, kp, ke, cn))
                ctl[cn] = load(g)["summary"]["distinct_valid_novel_tuples_mean"] if g.exists() else None
            ok = all(v is not None and v < row["store"][c]["mean"] - MARGIN for v in ctl.values())
            row["controls"][c] = dict(ctl, collapses=bool(ok))
            g = leg / ("rows_%s_learned1x_pool%d_est%d.json" % (c, kp, ke))
            if g.exists():
                dd = load(g)["summary"]
                row["learned_1x"][c] = {"mean": dd["distinct_valid_novel_tuples_mean"],
                                        "param_count": dd.get("param_count"), "file": str(g)}
        for p in ("pool3", "pool6"):
            st = lj["stores"][p]
            row["store_bytes"][p] = {"honest_total_bytes": st["ledger_four_part"]["honest_total_bytes"],
                                     "convention": "crossover hub four-part: atoms + cov/count + "
                                                   "address book + coupling/hole, never netted",
                                     "train_rows_drawn_from": st["train_rows_drawn_from"],
                                     "pool_n": st["pool_n"]}
        m = {c: row["store"][c]["mean"] for c in CELLS}
        rise = m["B"] - m["A"]
        cov = m["B"] - m["C"]
        mask = m["C"] - m["A"]
        ids = sorted(pcs["B"])
        bc = np.array([pcs["B"][i] - pcs["C"][i] for i in ids])
        ba = np.array([pcs["B"][i] - pcs["A"][i] for i in ids])
        nz = bc[bc != 0]
        row["decomposition"] = {
            "rise_B_minus_A": rise, "coverage_B_minus_C": cov, "mask_C_minus_A": mask,
            "store_at_n3_estimand_D_minus_A": m["D"] - m["A"],
            "coverage_share_s": (cov / rise) if rise != 0 else None,
            "paired_B_minus_C": {"n_corners": int(bc.size), "mean": float(bc.mean()),
                                 "n_pos": int((bc > 0).sum()), "n_neg": int((bc < 0).sum()),
                                 "sign_test_p": (float(binomtest(int((nz > 0).sum()), int(nz.size)).pvalue)
                                                 if nz.size else None)},
            "paired_B_minus_A": {"mean": float(ba.mean()), "n_pos": int((ba > 0).sum()),
                                 "n_neg": int((ba < 0).sum())}}
        if row["learned_1x"]:
            lm = {c: row["learned_1x"][c]["mean"] for c in row["learned_1x"]}
            if all(c in lm for c in CELLS):
                row["decomposition"]["learned_1x"] = {
                    "rise_B_minus_A": lm["B"] - lm["A"], "coverage_B_minus_C": lm["B"] - lm["C"],
                    "mask_C_minus_A": lm["C"] - lm["A"]}
                row["decomposition"]["margin_store_minus_learned_1x"] = {
                    c: m[c] - lm[c] for c in CELLS}
        val[s] = row
    res["val"] = {str(s): v for s, v in val.items()}

    # ---- the PREREG bars, verbatim ------------------------------------------------
    D = [val[s]["decomposition"] for s in SEEDS]
    gate = {c: all(val[s]["controls"][c]["collapses"] for s in SEEDS) for c in CELLS}
    rise_ok = all(d["rise_B_minus_A"] > MARGIN for d in D)
    cov_verdict = all((d["coverage_share_s"] or 0) >= 0.5 and d["coverage_B_minus_C"] > MARGIN
                      for d in D)
    depth_verdict = all(abs(d["coverage_B_minus_C"]) <= MARGIN for d in D)
    if not all(gate.values()):
        verdict = "NO CLAIM -- a control did not collapse (see control_gate)"
    elif not rise_ok:
        verdict = "NO RISE TO DECOMPOSE ON VAL (B - A <= 1.0 on some seed)"
    elif cov_verdict:
        verdict = "COVERAGE -- the note stands"
    elif depth_verdict:
        verdict = "DEPTH -- the caveat is too strong"
    else:
        verdict = "MIXED"
    res["bars"] = {"control_gate_every_seed": gate, "rise_exists_every_seed": rise_ok,
                   "coverage_bar_every_seed": cov_verdict, "depth_bar_every_seed": depth_verdict,
                   "VERDICT": verdict}

    def ms(xs):
        a = np.array(xs, float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "per_seed": a.tolist()}

    res["summary_val"] = {
        "n_seeds": 3,
        "store": {c: ms([val[s]["store"][c]["mean"] for s in SEEDS]) for c in CELLS},
        "rise_B_minus_A": ms([d["rise_B_minus_A"] for d in D]),
        "coverage_B_minus_C": ms([d["coverage_B_minus_C"] for d in D]),
        "mask_C_minus_A": ms([d["mask_C_minus_A"] for d in D]),
        "coverage_share_s": ms([d["coverage_share_s"] for d in D]),
    }
    if all("learned_1x" in d for d in D):
        res["summary_val"]["learned_1x"] = {
            c: ms([val[s]["learned_1x"][c]["mean"] for s in SEEDS]) for c in CELLS}
        res["summary_val"]["margin_store_minus_learned_1x"] = {
            c: ms([d["margin_store_minus_learned_1x"][c] for d in D]) for c in CELLS}

    # ---- test context (READ ONLY, already published legs) --------------------------
    t = {}
    for r in (3, 6):
        for s in SEEDS:
            f = W12B / ("%d_seed%d_K32" % (r, s)) / "rows_single_design_K32.json"
            g = W12B / ("%d_seed%d_K32" % (r, s)) / "rows_learned_1x.json"
            if not f.exists():
                continue
            t["n%d_seed%d" % (r, s)] = {
                "store": load(f)["summary"]["distinct_valid_novel_tuples_mean"],
                "learned_1x": (load(g)["summary"]["distinct_valid_novel_tuples_mean"]
                               if g.exists() else None),
                "file": str(f)}
    need = ["n%d_seed%d" % (r, s) for r in (3, 6) for s in SEEDS]
    if not all(k in t for k in need):
        res["test_context"] = {"not_present": [k for k in need if k not in t],
                               "note": "the K* ladder legs of rungs 3 and 6 (scripts/shapes_ladder_nclu.py) are read here"}
    else:
     res["test_context"] = {
        "source": str(W12B), "n_seeds": 3,
        "A_test_n3": ms([t["n3_seed%d" % s]["store"] for s in SEEDS]),
        "B_test_n6": ms([t["n6_seed%d" % s]["store"] for s in SEEDS]),
        "rise_test_B_minus_A": ms([t["n6_seed%d" % s]["store"] - t["n3_seed%d" % s]["store"]
                                   for s in SEEDS]),
        "C_test": ("DECLARED NOT-RUN -- a new configuration on the test split is a scored "
                   "run and `composition` is exhausted. NB by construction C_test's store is "
                   "bit-identical to A_test's (same pool draw from the same train index; the "
                   "pool recomputation reproduces w12's test pools, asserted per seed), so "
                   "C_test - A_test would be the scoring-mask effect alone."),
        "C_test_DERIVED_not_measured": {
            "value": ms([t["n3_seed%d" % s]["store"] for s in SEEDS]),
            "why": ("identity, IDENTITY-CHECK.txt: (1) pool(train_3) is rung-3's pool; "
                    "(2) acceptance 1.0 under either rung's bitmap; (3) judge_valid demands "
                    "all six factors == the request, so every valid tuple is a held-out "
                    "corner tuple and novel under either mask (0 mismatches in 3,600 test "
                    "rows); (4) the corner rows are identical. Confirmed empirically on val: "
                    "C == A on every seed."),
            "coverage_B_minus_C_test": ms([t["n6_seed%d" % s]["store"] - t["n3_seed%d" % s]["store"]
                                           for s in SEEDS]),
            "coverage_share_test": 1.0},
        "per_leg": t}
    with open(OUT / "COVERAGE-CONTROL.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"bars": res["bars"], "summary_val": res["summary_val"],
                      "test": {k: res["test_context"].get(k) for k in
                               ("A_test_n3", "B_test_n6", "rise_test_B_minus_A")}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
