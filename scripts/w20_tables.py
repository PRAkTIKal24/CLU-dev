"""w20 hub/shapes-data-matched -- BUILD SHAPES-MATCHED.json / .md from artefacts on disk.

Reads only score files (per-corner rows) and fit artefacts; computes nothing new about
any arm.  Every number carries the file it came from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w20_lib import FAMILIES, OUT, RUNGS, jdump  # noqa: E402

SEEDS = (0, 1, 2)
MARGIN = 1.0
DRAFT = (39.0, 49.0)
POOLS = {"n131072": "131,072", "n32768": "32,768", "n16384": "16,384"}
CONTROLS = ("landscape_removed", "landscape_shuffled", "landscape_shuffled_across_wells",
            "random_atoms")


def rd(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def paired(a_rows, b_rows):
    from chlu.experiments import exp_gen_factor_wells as fw
    return fw.paired_estimand(a_rows, b_rows, "distinct_valid_novel_tuples", MARGIN, "a", "b")


def mean_of(rows, fam=None):
    sel = [r for r in rows if fam is None or r["family"] == fam]
    return float(np.mean([r["distinct_valid_novel_tuples"] for r in sel])) if sel else None


def ceil_of(rows, fam=None):
    sel = [r for r in rows if (fam is None or r["family"] == fam) and "ceiling" in r]
    return float(np.mean([r["ceiling"] for r in sel])) if sel else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None,
            "per_seed": xs, "n_seeds": len(xs)}


def curve_evidence(where, seed, fam):
    d = OUT / "rivals" / where / ("seed%d" % seed) / fam
    out = {}
    sel = rd(d / "selection.json")
    if sel:
        out["selected"] = sel.get("selected")
        out["selection_source"] = sel.get("TRANSFERRED_FROM", str(d / "selection.json"))
        if "full" in sel:
            out["selection_heldout_curve"] = {
                str(f["epochs"]): [round(r["val_mse"], 6) for r in f["curve"] if "val_mse" in r]
                for f in sel["full"]}
    for st in ("declared", "tenx"):
        j = rd(d / ("%s.json" % st))
        if j:
            c = j["curve"]
            step = max(1, len(c) // 20)
            out[st] = {"epochs": j["epochs"], "optimizer_steps": j["optimizer_steps"],
                       "n_fit_rows": j["n_fit_rows"], "param_count": j["param_count"],
                       "train_mse_first": c[0]["train_mse"], "train_mse_final": c[-1]["train_mse"],
                       "last_decile_rel_change": j["last_decile_rel_change"],
                       "train_mse_curve_every_%d_epochs" % step:
                           [round(r["train_mse"], 6) for r in c[::step]] + [round(c[-1]["train_mse"], 6)],
                       "fit_seconds": j["train_info"]["seconds"], "source": str(d / ("%s.json" % st))}
    return out


def table_rows34():
    base = OUT / "scores" / "rows34"
    T7 = {0: "C:/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json",
          1: "C:/chlu/outputs/w12b-nr6-and-kstar/scored/run3_seed1_K32/scored.json",
          2: "C:/chlu/outputs/w12b-nr6-and-kstar/scored/run3_seed2_K32/scored.json"}
    res = {"corners": "rows 3-4 TEST: 60 pair sub-corners (row 3) + 49 n-way corners (row 4)",
           "per_seed": {}, "integrity_T7_reproduced": {}}
    for s in SEEDS:
        d = base / ("seed%d" % s)
        if not d.exists():
            continue
        arms = {}
        for f in sorted(d.glob("*.json")):
            j = rd(f)
            arms[j["label"]] = {"rows": j["per_corner"], "src": str(f), "extra": j.get("extra")}
        ref = None
        for k in arms:
            ref = [r["corner_id"] for r in arms[k]["rows"]]
            break
        lr = Path("C:/chlu/outputs/w12-composition-ladder/runs") / ("shipped_seed%d" % s)
        for fn, lab in (("rows_learned_ref_ownpool.json", "EXISTING_learned_ref_ownpool@130514_32768rows_2560steps"),):
            j = rd(lr / fn)
            if j and ref:
                m = {r["corner_id"]: r for r in j["per_corner"]}
                if set(m) == set(ref):
                    arms[lab] = {"rows": [m[c] for c in ref], "src": str(lr / fn), "extra": None}
                else:
                    res.setdefault("reuse_refused", []).append({"file": str(lr / fn), "why": "corner ids differ"})
        res["per_seed"][s] = arms
        t7 = rd(T7[s])
        if t7 and "store_K32_n131072" in arms:
            a = [r["distinct_valid_novel_tuples"] for r in arms["store_K32_n131072"]["rows"]]
            b = [r["distinct_valid_novel_tuples"] for r in t7["per_arm"]["cov_tensor"]["per_corner"]]
            res["integrity_T7_reproduced"][s] = {"identical": a == b, "T7": T7[s],
                                                 "max_abs_dev": float(np.max(np.abs(np.array(a) - np.array(b))))}
    return res


def summarise_block(arms_by_seed, store_labels, rival_where, fams=FAMILIES, has_rows=True,
                    existing_rivals=None):
    """Per-arm means per seed (all / row3 / row4), controls, P1, P3."""
    out = {"arms": {}, "P1_convergence": {}, "P3_margin": {}, "P2_controls": {}}
    fams_rows = [None, "pair", "nway"] if has_rows else [None]
    key = {None: "all", "pair": "row3_pair", "nway": "row4_nway"}
    labels = set()
    for s, arms in arms_by_seed.items():
        labels |= set(arms)
    for lab in sorted(labels):
        e = {"n_seeds": 0, "sources": {}}
        for fr in fams_rows:
            e[key[fr]] = ms([mean_of(arms_by_seed[s][lab]["rows"], fr)
                             for s in arms_by_seed if lab in arms_by_seed[s]])
            e["ceiling_" + key[fr]] = ms([ceil_of(arms_by_seed[s][lab]["rows"], fr)
                                         for s in arms_by_seed if lab in arms_by_seed[s]])
        e["n_seeds"] = sum(1 for s in arms_by_seed if lab in arms_by_seed[s])
        e["sources"] = {s: arms_by_seed[s][lab]["src"] for s in arms_by_seed if lab in arms_by_seed[s]}
        ex = [arms_by_seed[s][lab].get("extra") for s in arms_by_seed if lab in arms_by_seed[s]]
        pert = [x.get("perturbation") for x in ex if x and x.get("perturbation")]
        if pert and pert[0].get("kind") == "well_permute":
            e["derangement_printed"] = [{"perm": p["perm"], "fixed_points": p["fixed_points"],
                                         "historical_T7_perm_fixed_points": p["historical_T7_fixed_points"]}
                                        for p in pert]
        out["arms"][lab] = e
    # P2
    for st in store_labels:
        g = {}
        for c in CONTROLS:
            lab = "%s__%s" % (st, c)
            if lab in out["arms"] and out["arms"][lab]["all"]:
                ps = out["arms"][lab]["all"]["per_seed"]
                g[c] = {"per_seed": ps, "collapses_every_seed": all(x <= MARGIN for x in ps)}
            else:
                g[c] = {"DECLARED_NOT_RUN_or_missing": True}
        g["ALL_COLLAPSE"] = all(v.get("collapses_every_seed") for v in g.values() if isinstance(v, dict))
        out["P2_controls"][st] = g
    # P1 and carried rival per family per seed
    carried = {}
    for fam in fams:
        dl, tl = "%s__declared@%s" % (fam, rival_where), "%s__tenx@%s" % (fam, rival_where)
        per = {}
        for s, arms in arms_by_seed.items():
            if dl not in arms:
                continue
            dm = mean_of(arms[dl]["rows"])
            tm = mean_of(arms[tl]["rows"]) if tl in arms else None
            if tm is None:
                per[s] = {"declared": dm, "tenx": None, "P1": "TENX NOT RUN -- declared carried, "
                          "UNVERIFIED convergence", "carried": "declared"}
                carried[(fam, s)] = (dl, dm)
                continue
            under = (tm - dm) > MARGIN
            best = tl if tm >= dm else dl
            per[s] = {"declared": dm, "tenx": tm, "tenx_minus_declared": tm - dm,
                      "P1": "UNDER-TRAINED: TENX carries" if under else "PASS",
                      "carried": "tenx" if under else "declared",
                      "max_of_two": max(dm, tm)}
            carried[(fam, s)] = (tl if under else dl, tm if under else dm)
            carried[(fam, s, "max")] = (best, max(dm, tm))
        out["P1_convergence"][fam] = per
    # P3 for each store label vs each family's carried rival and vs strongest
    for st in store_labels:
        blk = {"per_family": {}, "vs_strongest": {}}
        for fam in fams:
            pf = {}
            for s, arms in arms_by_seed.items():
                if st not in arms or (fam, s) not in carried:
                    continue
                lab, _ = carried[(fam, s)]
                pe = paired(arms[st]["rows"], arms[lab]["rows"])
                labm, _ = carried[(fam, s, "max")] if (fam, s, "max") in carried else carried[(fam, s)]
                pm = paired(arms[st]["rows"], arms[labm]["rows"])
                pf[s] = {"carried_rival": lab, "delta": pe["mean_paired_delta"],
                         "ci95": pe["bootstrap_ci95"],
                         "won_tied_lost": [pe["corners_arm_wins"], pe["corners_tied"], pe["corners_control_wins"]],
                         "delta_vs_max_of_declared_tenx": pm["mean_paired_delta"],
                         "rows": ({"row3": mean_of(arms[st]["rows"], "pair") - mean_of(arms[lab]["rows"], "pair"),
                                   "row4": mean_of(arms[st]["rows"], "nway") - mean_of(arms[lab]["rows"], "nway")}
                                  if has_rows else None)}
            if pf:
                pf["mean_delta"] = float(np.mean([v["delta"] for k, v in pf.items() if isinstance(k, int)]))
            blk["per_family"][fam] = pf
        for s, arms in arms_by_seed.items():
            if st not in arms:
                continue
            cand = [(carried[(f, s, "max")] if (f, s, "max") in carried else carried[(f, s)])
                    for f in fams if (f, s) in carried]
            if not cand:
                continue
            lab, _ = max(cand, key=lambda x: x[1])
            pe = paired(arms[st]["rows"], arms[lab]["rows"])
            blk["vs_strongest"][s] = {"strongest_rival (max of declared/tenx)": lab,
                                      "store": pe["arm_mean"], "rival": pe["control_mean"],
                                      "delta": pe["mean_paired_delta"], "ci95": pe["bootstrap_ci95"],
                                      "won_tied_lost": [pe["corners_arm_wins"], pe["corners_tied"],
                                                        pe["corners_control_wins"]],
                                      "beats_beyond_margin": pe["beats_beyond_margin"]}
        ds = [v["delta"] for v in blk["vs_strongest"].values()]
        if ds:
            m = float(np.mean(ds))
            blk["vs_strongest_mean"] = m
            blk["verdict"] = ("HOLDS" if m >= DRAFT[0] else "COLLAPSES" if m <= MARGIN
                              else "SHRINKS")
        if existing_rivals:
            ex = {}
            for er in existing_rivals:
                for s, arms in arms_by_seed.items():
                    if st in arms and er in arms:
                        pe = paired(arms[st]["rows"], arms[er]["rows"])
                        ex.setdefault(er, {})[s] = {"delta": pe["mean_paired_delta"],
                                                    "ci95": pe["bootstrap_ci95"]}
            blk["vs_existing_rivals"] = ex
        out["P3_margin"][st] = blk
    return out


def load_dir(base: Path):
    by = {}
    for s in SEEDS:
        d = base / ("seed%d" % s)
        if not d.exists():
            continue
        arms = {}
        for f in sorted(d.glob("*.json")):
            j = rd(f)
            # keyed by FILE NAME: the ladder's across-wells control for the K*=32 store is scored on
            # the REBUILT object and copied to the store's name (ladderscore.sh)
            arms[f.stem] = {"rows": j["per_corner"], "src": str(f), "extra": j.get("extra")}
        by[s] = arms
    return by


def add_existing_ladder(by, rung):
    """w12b K*=32 store + its controls + learned_1x, and the w12 ladder K=5 arm."""
    kl = Path("C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")
    for s in list(by):
        leg = kl / ("%s_seed%d_K32" % (rung, s))
        for fn, lab in (("rows_single_design_K32.json", "store_K32_n131072"),
                        ("rows_single_design_K32__landscape_removed.json", "store_K32_n131072__landscape_removed"),
                        ("rows_single_design_K32__landscape_shuffled.json", "store_K32_n131072__landscape_shuffled"),
                        ("rows_single_design_K32__random_atoms.json", "store_K32_n131072__random_atoms")):
            j = rd(leg / fn)
            if j and lab not in by[s]:
                by[s][lab] = {"rows": j["per_corner"], "src": str(leg / fn), "extra": None}
        lr = Path("C:/chlu/outputs/w12-composition-ladder/runs") / ("%s_seed%d" % (rung, s))
        for fn, lab in (("rows_single_design_K5.json", "EXISTING_ladder_store_K5_n131072"),
                        ("rows_rival_mlp.json", "EXISTING_spokeA_mlp@118k_32256montero"),
                        ("rows_rival_lstm.json", "EXISTING_spokeA_lstm@118k_32256montero"),
                        ("rows_rival_transformer.json", "EXISTING_spokeA_transformer@118k_32256montero"),
                        ("rows_rival_cont_mlp.json", "EXISTING_spokeA_cont_mlp@118k_32256montero"),
                        ("rows_single_design_K5__landscape_shuffled.json", "EXISTING_ladder_store_K5_n131072__landscape_shuffled"),
                        ("rows_single_design_K5__random_atoms.json", "EXISTING_ladder_store_K5_n131072__random_atoms"),
                        ("rows_learned_ref_ownpool.json", "EXISTING_learned_ref_ownpool@130514_32768rows_2560steps"),
                        # learned_1x (536k / ~3.2M params, matched to the store's MEMORY ledger)
                        # is NOT printed: COMMON rule -- rivals are matched on read-path
                        # parameters; "the inflated-rival row does not exist".,
                        ("rows_rival_table.json", "table_FLOOR")):
            j = rd(lr / fn)
            if j and lab not in by[s]:
                by[s][lab] = {"rows": j["per_corner"], "src": str(lr / fn), "extra": None}
    return by


def deletion_block():
    """Store (a)-(d) from deletion_store/, rivals' retrain-without from scores/rows34 (run 3/3)."""
    out = {"definition": ("Head AMENDMENT-2: forget c = downdate store + downdate address-book "
                          "statistics (book refit from them) + punch hole. (a)(b)(d) RESULT, (c) DISPLAY."),
           "RETIRED": "'c still composes after forgetting c' -- not computed, not printed",
           "per_scope": {}}
    for scope in ("value", "conjunction_live", "item"):
        blk = {"store": {}, "rivals": {}}
        for sd in SEEDS:
            j = rd(OUT / "deletion_store" / ("seed%d" % sd) / ("%s.json" % scope))
            if not j:
                continue
            d = j["d_composition_other_values"]
            blk["store"][sd] = {"a": j["a_store_equals_refit_without"],
                                "b": j["b_address_book_statistics_equal_refit_without"],
                                "c_DISPLAY": j["c_hole_DISPLAY"],
                                "d": {k: v for k, v in d.items() if not k.startswith("per_corner")},
                                "n_deleted": j["n_deleted"], "n_survivors": j["n_survivors"],
                                "source": str(OUT / "deletion_store" / ("seed%d" % sd) / ("%s.json" % scope))}
            after = {r["corner_id"]: r for r in d["per_corner_after"]}
            surv = list(after)
            base = OUT / "scores" / "rows34" / ("seed%d" % sd)
            for fam in FAMILIES:
                bt = rd(base / ("%s__declared@test.json" % fam))
                # the rival's pre-deletion reference is its DECLARED fit (the schedule the
                # retrain-without used); P1 flags travel from rows34.P1_convergence
                rw = rd(base / ("%s__declared@test_del_%s.json" % (fam, scope)))
                if not (bt and rw):
                    continue
                bm = {r["corner_id"]: r for r in bt["per_corner"]}
                rm = {r["corner_id"]: r for r in rw["per_corner"]}
                A = [after[c] for c in surv]
                Rw = [rm[c] for c in surv]
                Bt = [bm[c] for c in surv]
                pe = paired(A, Rw)
                pr = paired(Rw, Bt)
                cl = rd(OUT / "rivals" / ("test_del_%s" % scope) / ("seed%d" % sd) / fam / "declared.json")
                blk["rivals"].setdefault(fam, {})[sd] = {
                    "rival_as_trained_on_survivors": pr["control_mean"],
                    "rival_retrain_without_on_survivors": pr["arm_mean"],
                    "rival_after_minus_before": pr["mean_paired_delta"],
                    "ours_after": pe["arm_mean"], "ours_minus_rival_retrain_without": pe["mean_paired_delta"],
                    "ci95": pe["bootstrap_ci95"],
                    "won_tied_lost": [pe["corners_arm_wins"], pe["corners_tied"], pe["corners_control_wins"]],
                    "rival_cost_line": (cl or {}).get("cost_line"),
                    "rival_deletion": (cl or {}).get("deletion")}
        out["per_scope"][scope] = blk
    return out


def lacepoe_block():
    out = {}
    for leg in sorted((OUT / "lacepoe").glob("*_seed*")) if (OUT / "lacepoe").exists() else []:
        for arm in ("lace", "poe"):
            j = rd(leg / ("rows_%s.json" % arm))
            if j:
                sm = j["summary"]
                out.setdefault(leg.name.rsplit("_seed", 1)[0], {}).setdefault(arm, {})[leg.name.rsplit("_seed", 1)[1]] = {
                    "novel_per_corner": sm.get("distinct_valid_novel_tuples_mean"),
                    "budget": (sm.get("extra") or sm).get("budget") if isinstance(sm, dict) else None,
                    "source": str(leg / ("rows_%s.json" % arm))}
    return out


def main():
    sys.path.insert(0, "C:/chlu/worktrees/CHLU-w20-shapes-data-matched")
    doc = {"schema": "w20 SHAPES-MATCHED v1", "margin": MARGIN, "draft_margin_range": DRAFT,
           "seeds": list(SEEDS)}
    # ---- rows 3-4 ----
    r34 = table_rows34()
    by = {s: v for s, v in r34["per_seed"].items()}
    ex16 = ["w12comp_%s__as_trained" % f for f in FAMILIES] + ["EXISTING_learned_ref_ownpool@130514_32768rows_2560steps", "mlp__declared@test16k_stepmatched", "mlp__declared@test32k_stepmatched"]
    stores = [x for x in ["store_K32_n131072", "store_K5_n131072", "store_K32_n16384",
                          "store_K5_n16384", "store_K32_n32768", "store_K5_n32768"]
              if any(x in by[s] for s in by)]
    doc["rows34"] = {"integrity_T7_reproduced": r34["integrity_T7_reproduced"],
                     **summarise_block(by, stores, "test", existing_rivals=ex16 + ["table_FLOOR"])}
    # P3 against the rival at its best over EVERY fit that ran (PREREG A5.3/A5.4): declared, tenx, 30x
    best_all = {}
    for sd, arms in by.items():
        if "store_K32_n131072" not in arms:
            continue
        cands = [l for l in arms if any(l.startswith(f + "__") for f in FAMILIES)
                 and l.split("@")[1] in ("test", "test30x")]
        if not cands:
            continue
        lab = max(cands, key=lambda l: mean_of(arms[l]["rows"]))
        pe = paired(arms["store_K32_n131072"]["rows"], arms[lab]["rows"])
        best_all[sd] = {"best_rival_over_all_fits": lab, "rival": pe["control_mean"],
                        "store": pe["arm_mean"], "delta": pe["mean_paired_delta"],
                        "ci95": pe["bootstrap_ci95"],
                        "won_tied_lost": [pe["corners_arm_wins"], pe["corners_tied"], pe["corners_control_wins"]]}
    if best_all:
        m = float(np.mean([v["delta"] for v in best_all.values()]))
        best_all["mean"] = m
        best_all["verdict"] = "HOLDS" if m >= DRAFT[0] else "COLLAPSES" if m <= MARGIN else "SHRINKS"
    doc["rows34"]["P3_vs_best_over_all_fits_incl_30x"] = best_all
    # the draft's margin, recomputed on the same files: store − each 16k comparator, per seed
    draft = {}
    for sd, arms in by.items():
        for f in FAMILIES:
            l = "w12comp_%s__as_trained" % f
            if l in arms and "store_K32_n131072" in arms:
                draft.setdefault(f, {})[sd] = paired(arms["store_K32_n131072"]["rows"], arms[l]["rows"])["mean_paired_delta"]
    doc["rows34"]["draft_margin_recomputed_vs_16k_comparators"] = draft
    doc["rows34"]["convergence_evidence"] = {
        f: {s: curve_evidence("test", s, f) for s in SEEDS} for f in FAMILIES}
    # ---- ladder ----
    lad = {}
    for k in RUNGS:
        byk = add_existing_ladder(load_dir(OUT / "scores" / ("rung%s" % k)), k)
        if not byk:
            continue
        st = [x for x in ["store_K32_n131072", "EXISTING_ladder_store_K5_n131072",
                          "store_K32_n16384", "store_K5_n16384", "store_K32_n32768", "store_K5_n32768"]
              if any(x in byk[s] for s in byk)]
        exl = sorted({l for s in byk for l in byk[s] if l.startswith("EXISTING_") and "store" not in l}
                     | {"table_FLOOR"})
        lad[k] = summarise_block(byk, st, "rung%s" % k, has_rows=False, existing_rivals=exl)
        integ = {}
        for sd, arms in byk.items():
            if "store_K32_n131072" in arms and "store_K32_n131072_REBUILT" in arms:
                a_ = {r["corner_id"]: r["distinct_valid_novel_tuples"] for r in arms["store_K32_n131072"]["rows"]}
                b_ = {r["corner_id"]: r["distinct_valid_novel_tuples"] for r in arms["store_K32_n131072_REBUILT"]["rows"]}
                integ[sd] = {"identical_per_corner": a_ == b_,
                             "n_corners_differ": sum(1 for c in a_ if a_[c] != b_.get(c))}
        lad[k]["integrity_rebuilt_vs_w12b"] = integ
        lad[k]["convergence_evidence"] = {f: {s: curve_evidence("rung%s" % k, s, f) for s in SEEDS}
                                          for f in FAMILIES}
    doc["ladder"] = lad
    g = rd("C:/chlu/outputs/w15-lace-poe-rows/GENTABLE-KSTAR.json")
    if g:
        doc["LACE_PoE_from_w15"] = {
            "scoping_VERBATIM": ("LACE/PoE as instantiated in this feature space at their own "
                                 "declared budgets -- NOT a refutation of LACE in general"),
            "source": "C:/chlu/outputs/w15-lace-poe-rows/GENTABLE-KSTAR.json",
            "rows34": {a: g["PART_B_rows_3_4"]["arms"][a] for a in ("LACE", "PoE")
                       if a in g["PART_B_rows_3_4"].get("arms", {})},
            "ladder": {k: {a: g["PART_B_ladder"][k]["arms"][a] for a in ("LACE", "PoE")
                           if a in g["PART_B_ladder"][k]["arms"]} for k in g["PART_B_ladder"]}}
    doc["deletion_matched"] = deletion_block()
    doc["LACE_PoE_at_131072"] = {"scoping_VERBATIM": ("LACE/PoE as instantiated in this feature space at "
                                                      "their own declared budgets -- NOT a refutation of LACE in general"),
                                 "fit_rows": "130,560 fit + 512 val (w15: 32,256 + 512)",
                                 "legs": lacepoe_block()}
    jdump(doc, OUT / "SHAPES-MATCHED.json")
    print("wrote", OUT / "SHAPES-MATCHED.json")


if __name__ == "__main__":
    main()
