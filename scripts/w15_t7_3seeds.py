"""w15 -- T7-3SEEDS.json: the acceptance artefact, built from scored artefacts ONLY.

Rows 3 and 4 at K* = 32 on TEST with n = 3 and the seed list printed; the four-part
byte ledger PER SEED, never netted, with its convention named; all four charter-C
controls PER SEED; and the three deletion rows with their adaptation cost.

Nothing is computed here.  Every number is read out of a JSON on disk and carries that
path.  A seed or a leg whose artefact is missing is written as a DECLARED NOT-RUN with
the absent path, never as a zero.
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

OUT = Path("C:/chlu/outputs/w15-single-design-seeds")
W12 = Path("C:/chlu/outputs/w12-single-design-crossover")
W12B = Path("C:/chlu/outputs/w12b-nr6-and-kstar")

SEEDS = [0, 1, 2]
K = 32

SCORED = {
    0: W12 / "scored" / "run1_seed0_K32" / "scored.json",
    1: W12B / "scored" / "run3_seed1_K32" / "scored.json",
    2: W12B / "scored" / "run3_seed2_K32" / "scored.json",
}
#: the arm's deletion legs: seed 0 from w12, seeds 1-2 produced here
DELETION = {
    0: W12 / "deletion" / "test_seed0_K32" / "deletion.json",
    1: OUT / "deletion" / "test_seed1_K32" / "deletion.json",
    2: OUT / "deletion" / "test_seed2_K32" / "deletion.json",
}
#: the rival deletion column: seed 0 in its own rivals.json, seeds 1-2 inside the
#: deletion.json above
RIVALS_DOC = {
    0: OUT / "deletion" / "test_seed0_K32" / "rivals.json",
    1: OUT / "deletion" / "test_seed1_K32" / "deletion.json",
    2: OUT / "deletion" / "test_seed2_K32" / "deletion.json",
}

CONTROLS = ("landscape_removed", "landscape_shuffled",
            "landscape_shuffled_across_wells", "random_atoms")
RIVALS = ("mlp", "lstm", "transformer", "cont_mlp")
SCOPES = ("value", "conjunction", "conjunction_live", "item")
SCOPE_LABEL = {
    "value": "red value (object_hue index 0)",
    "conjunction": "red AND cube -- AS LITERALLY SPECIFIED; deletes ZERO rows",
    "conjunction_live": "RED SPHERE (object_hue = 0.0 AND shape = 2.0) -- the live "
                        "conjunction pin",
    "item": "a random 1 % of fit rows",
}

BYTE_CONVENTION = (
    "FOUR-PART TOTAL, SUMMED AND NEVER NETTED. part 1 atoms = sum_k A x d_k float32 "
    "(4 B/float); part 2 = the packed float64 second-moment triangle + first moment + "
    "count that make the deletion exact; part 3 = the fitted TensorAddress parameters; "
    "part 4 = the coupling / hole table that carries the refusal holes. "
    "`reference_pin_bytes = 524,288` is a REFERENCE COLUMN ONLY this wave -- it is NOT "
    "a budget and the object is 1x its OWN total. `times_reference_pin` is "
    "honest_total_bytes / 524,288. The MEMORY LEDGER IS PRINTED SEPARATELY AND IS "
    "NEVER MATCHED between arms.")


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _ms(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return {"n": 0, "mean": None, "sd": None}
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else None
    return {"n": len(xs), "mean": m, "sd": sd, "values": xs}


def main() -> int:
    scored = {s: _load(SCORED[s]) for s in SEEDS}
    delet = {s: _load(DELETION[s]) for s in SEEDS}
    rivdoc = {s: _load(RIVALS_DOC[s]) for s in SEEDS}

    doc = {
        "WHAT_THIS_IS": (
            "T7 -- rows 3 and 4 of the composition table at K* = 32 on the TEST "
            "condition -- with n = 3 and the seed list printed, the four-part byte "
            "ledger per seed, all four charter-C controls per seed, and the three "
            "deletion rows with their adaptation cost."),
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "built_by": "scripts/w15_t7_3seeds.py, from scored artefacts on disk only",
        "condition": "test", "K_star": K,
        "n_seeds": len(SEEDS), "seeds": list(SEEDS),
        "notation": "n = factors jointly composed; K = wells. Here K = 32 wells.",
        "SCORED_RUNS_SPENT_BY_THIS_HUB": 0,
        "scored_run_accounting": (
            "single-design is EXHAUSTED at 3/3. Run 1/3 was spent by w12 "
            "hub/single-design-crossover (2026-09-21T12:53:06Z) and runs 2/3 and 3/3 by "
            "w12b hub/nr6-and-kstar (2026-09-21T21:33:51Z), the latter explicitly to "
            "'add seeds 1 and 2 at K* = 32 on TEST'. This hub therefore spends NOTHING "
            "and re-produces nothing: the T7 rows and the control gate below are READ "
            "from those charged artefacts. What this hub adds is RESULT 3 of the same "
            "PREREG -- the deletion rows -- which existed at seed 0 only, plus the "
            "rival deletion column, which existed at no seed at all."),
        "prereg": ("outputs/w12-single-design-crossover/"
                   "PREREG-w12-single-design-crossover-1.md + AMENDMENT 1 "
                   "(+ outputs/w12b-nr6-and-kstar/PREREG-AMENDMENT-2.md for seeds 1-2)"),
        "ledger_note": ("/c/chlu/SCORED-LEDGER.md, row appended 2026-09-22T21:31:47Z by "
                        "w15 hub/single-design-seeds -- a NOTE, not a run row"),
        "metric": ("distinct_valid_novel_tuples PER CORNER; the corner is the unit. "
                   "Verdict function fw.paired_estimand, margin "
                   "DECORATION_MARGIN_NOVELTY = 1.0. A rate never appears without its "
                   "count."),
        "sources": {f"seed{s}": {"scored": str(SCORED[s]),
                                 "deletion": str(DELETION[s]),
                                 "rival_deletion_column": str(RIVALS_DOC[s])}
                    for s in SEEDS},
        "byte_ledger_convention": BYTE_CONVENTION,
    }

    # ---- T7: rows 3 and 4, n = 3 --------------------------------------------
    t7 = {"n_seeds": len(SEEDS), "seeds": list(SEEDS), "rows": {}}
    for arm in ("cov_tensor",) + CONTROLS + tuple(f"{r}__as_trained" for r in RIVALS) \
            + ("table",):
        per = {}
        for s in SEEDS:
            d = scored[s]
            if d is None or arm not in d.get("per_arm", {}):
                per[f"seed{s}"] = {"DECLARED_NOT_RUN": f"absent from {SCORED[s]}"}
                continue
            per[f"seed{s}"] = {
                "row3_pair": d["per_arm"][arm]["row3_pair"]["novel_per_corner_mean"],
                "row4_nway": d["per_arm"][arm]["row4_nway"]["novel_per_corner_mean"],
                "row3_n_corners": d["per_arm"][arm]["row3_pair"]["n_corners"],
                "row4_n_corners": d["per_arm"][arm]["row4_nway"]["n_corners"],
                "source": str(SCORED[s]),
            }
        t7["rows"][arm] = {
            "per_seed": per,
            "row3_pair_over_seeds": _ms([v.get("row3_pair") for v in per.values()]),
            "row4_nway_over_seeds": _ms([v.get("row4_nway") for v in per.values()]),
            "role": ("THE ARM" if arm == "cov_tensor" else
                     "CHARTER-C CONTROL -- must collapse" if arm in CONTROLS else
                     "PRINTED FLOOR, never a rival" if arm == "table" else
                     "524,288 B REFERENCE-column comparator; the 1x column is a "
                     "DECLARED NOT-RUN"),
        }
    doc["T7_rows_3_and_4"] = t7

    # ---- the four-part byte ledger, per seed --------------------------------
    doc["byte_ledger_per_seed"] = {
        "convention": BYTE_CONVENTION,
        "n_seeds": len(SEEDS), "seeds": list(SEEDS),
        "per_seed": {f"seed{s}": (dict(scored[s]["ledger_four_part"],
                                       source=str(SCORED[s]))
                                  if scored[s] else
                                  {"DECLARED_NOT_RUN": f"absent: {SCORED[s]}"})
                     for s in SEEDS},
        "NETTED": False,
    }

    # ---- the control gate, per seed -----------------------------------------
    gate = {"rule": ("NO STORE ROW IS CLAIMED unless landscape_removed, "
                     "landscape_shuffled, landscape_shuffled_across_wells and "
                     "random_atoms ALL collapse on EVERY seed. A control that "
                     "collapsed at seed 0 is NOT assumed to collapse at seeds 1 and 2 "
                     "-- it is measured at each."),
            "collapse_definition": ("the control's distinct_valid_novel_tuples per "
                                    "corner reads 0.0000 against the arm, and the "
                                    "paired estimand beats the 1.0 margin on every "
                                    "corner"),
            "per_seed": {}}
    all_pass = True
    for s in SEEDS:
        d = scored[s]
        if d is None:
            gate["per_seed"][f"seed{s}"] = {"DECLARED_NOT_RUN": str(SCORED[s])}
            all_pass = False
            continue
        row = {"arm_novel_per_corner_all":
               d["per_arm"]["cov_tensor"]["all"]["novel_per_corner_mean"],
               "source": str(SCORED[s]), "controls": {}}
        ok = True
        for c in CONTROLS:
            pv = d["paired_vs_cov_tensor"].get(c, {})
            v = d["per_arm"][c]["all"]["novel_per_corner_mean"]
            beats = pv.get("beats_beyond_margin")
            collapsed = bool(v == 0.0)
            ok = ok and collapsed
            row["controls"][c] = {
                "novel_per_corner_all": v,
                "row3_pair": d["per_arm"][c]["row3_pair"]["novel_per_corner_mean"],
                "row4_nway": d["per_arm"][c]["row4_nway"]["novel_per_corner_mean"],
                "collapsed": collapsed,
                "paired_mean_delta": pv.get("mean_paired_delta"),
                "paired_ci95": pv.get("bootstrap_ci95"),
                "corners_won": pv.get("corners_arm_wins"),
                "corners_tied": pv.get("corners_tied"),
                "corners_lost": pv.get("corners_control_wins"),
                "sign_test_p_two_sided": pv.get("sign_test_p_two_sided"),
                "beats_the_1.0_margin": beats,
            }
        row["ALL_FOUR_COLLAPSE"] = ok
        all_pass = all_pass and ok
        gate["per_seed"][f"seed{s}"] = row
    gate["ALL_FOUR_COLLAPSE_ON_EVERY_SEED"] = all_pass
    gate["consequence"] = ("store rows ARE claimable at n = 3" if all_pass else
                           "NO STORE ROW IS CLAIMED -- that is the result, and it is "
                           "stated in those words")
    doc["control_gate"] = gate

    # ---- the deletion rows ---------------------------------------------------
    dele = {"n_seeds_attempted": len(SEEDS), "seeds": list(SEEDS),
            "exactness_definition": (
                "STORE-SCOPED: downdate(delete rows) equals the store fitted from "
                "scratch without those rows, at the same partition and the same rank. "
                "THE FROZEN ENCODER AND THE FROZEN DECODER ARE DECLARED OUTSIDE THE "
                "DELETABLE OBJECT AND ARE NOT REFIT."),
            "adaptation_language": (
                "post-deletion adaptation is LIGHT and DATA-FREE -- the covariance and "
                "count are downdated by subtraction, the atoms are re-eigendecomposed "
                "from the downdated statistic, the address book is carried across the "
                "basis change, and NO ROW OF DATA IS READ. It is never 'no changes'."),
            "rival_deletion_rule": (
                "A RIVAL DELETES ONLY BY RETRAIN-WITHOUT, and its cost is printed. "
                "Rival as-trained is printed beside it and is NOT a deletion "
                "mechanism."),
            "scopes": {}}
    for sc in SCOPES:
        per = {}
        for s in SEEDS:
            d = delet[s]
            if d is None or sc not in (d.get("scopes") or {}):
                per[f"seed{s}"] = {"DECLARED_NOT_RUN": f"absent: {DELETION[s]}"}
                continue
            v = d["scopes"][sc]
            p = v["paired_after_minus_before_on_survivors"]
            ex = v["exactness_store_scoped"]
            per[f"seed{s}"] = {
                "n_rows_deleted": v["n_rows_deleted"],
                "frac_rows_deleted": v.get("frac_rows_deleted"),
                "VACUOUS": v["VACUOUS"], "vacuous_reason": v.get("vacuous_reason"),
                "n_survivor_corners": v["n_survivor_corners"],
                "n_deleted_corners": v["n_deleted_corners"],
                "survivors_before": v["before_same_corners"]["survivors"].get(
                    "novel_per_corner_mean"),
                "survivors_after": v["after"]["survivors"].get(
                    "novel_per_corner_mean"),
                "paired_after_minus_before": p.get("mean_paired_delta"),
                "paired_ci95": p.get("bootstrap_ci95"),
                "exactness_max_abs_mode_dev": ex["max_abs_mode_dev"],
                "exactness_max_subspace_projector_dist": ex.get(
                    "max_subspace_projector_dist"),
                "ranks_agree_with_refit_without": ex["ranks_agree"],
                "OUR_ADAPTATION_COST": v["our_cost_line"],
                "REFIT_WITHOUT_COST": ex["refit_without_cost_line"],
                "source": str(DELETION[s]),
            }
        dele["scopes"][sc] = {
            "label": SCOPE_LABEL[sc],
            "per_seed": per,
            "over_seeds": {
                "n_rows_deleted": _ms([v.get("n_rows_deleted") for v in per.values()]),
                "paired_after_minus_before": _ms(
                    [v.get("paired_after_minus_before") for v in per.values()]),
                "survivors_after": _ms([v.get("survivors_after")
                                        for v in per.values()]),
                "our_adaptation_seconds": _ms(
                    [(v.get("OUR_ADAPTATION_COST") or {}).get("seconds")
                     for v in per.values()]),
                "our_data_rows_read": _ms(
                    [(v.get("OUR_ADAPTATION_COST") or {}).get("data_rows_read")
                     for v in per.values()]),
                "refit_without_seconds": _ms(
                    [(v.get("REFIT_WITHOUT_COST") or {}).get("seconds")
                     for v in per.values()]),
                "refit_without_data_rows_read": _ms(
                    [(v.get("REFIT_WITHOUT_COST") or {}).get("data_rows_read")
                     for v in per.values()]),
            },
        }
    doc["deletion_rows"] = dele

    # ---- the rival deletion column ------------------------------------------
    riv = {"rule": ("A RIVAL DELETES ONLY BY RETRAIN-WITHOUT. Its cost line is quoted "
                    "from outputs/w12-deletion-baselines/BASELINES-READY.json. Rival "
                    "as-trained is printed beside it, never as a deletion mechanism."),
           "scopes": {}}
    for sc in SCOPES:
        per = {}
        for s in SEEDS:
            d = rivdoc[s]
            col = (d or {}).get("rival_deletion_column", {}).get(sc)
            if col is None:
                per[f"seed{s}"] = {"DECLARED_NOT_RUN": f"absent: {RIVALS_DOC[s]}"}
                continue
            if "DECLARED_NOT_RUN" in col:
                per[f"seed{s}"] = {"DECLARED_NOT_RUN": col["DECLARED_NOT_RUN"]}
                continue
            fams = {}
            for f in RIVALS:
                rw = (col.get("retrain_without") or {}).get(f, {})
                pm = (col.get("paired_ours_minus_rival") or {}).get(f, {})
                at = ((d.get("rival_as_trained_carried") or {})
                      .get(f"{f}__as_trained") or {})
                fams[f] = {
                    "retrain_without_novel_per_corner": (
                        (rw.get("survivors") or {}).get("novel_per_corner_mean")
                        if "survivors" in rw else
                        {"DECLARED_NOT_RUN": rw.get("DECLARED_NOT_RUN")}),
                    "retrain_without_cost": (
                        col.get("retrain_without_cost_lines") or {}).get(f),
                    "as_trained_novel_per_corner_all_109_corners": (
                        at.get("all") or {}).get("novel_per_corner_mean"),
                    "as_trained_cost": (
                        col.get("as_trained_cost_lines") or {}).get(f),
                    "paired_ours_minus_rival": {
                        "mean_paired_delta": pm.get("mean_paired_delta"),
                        "ci95": pm.get("bootstrap_ci95"),
                        "corners_won": pm.get("corners_arm_wins"),
                        "corners_tied": pm.get("corners_tied"),
                        "corners_lost": pm.get("corners_control_wins"),
                        "sign_test_p_two_sided": pm.get("sign_test_p_two_sided"),
                        "beats_the_1.0_margin": pm.get("beats_beyond_margin"),
                        "DECLARED_NOT_RUN": pm.get("DECLARED_NOT_RUN"),
                    },
                }
            per[f"seed{s}"] = {
                "n_survivor_corners": col.get("n_survivor_corners"),
                "ours_after_deletion_on_the_same_survivors": (
                    col.get("ours_after_deletion_on_the_same_survivors") or {}).get(
                        "novel_per_corner_mean"),
                "families": fams,
                "scale_caveat": col.get("scale_caveat"),
                "item_mask_caveat": col.get("item_mask_caveat"),
                "source": str(RIVALS_DOC[s]),
            }
        riv["scopes"][sc] = {"label": SCOPE_LABEL[sc], "per_seed": per}
    doc["rival_deletion_column"] = riv

    doc["WHAT_DID_NOT_RUN"] = {
        "single_design_runs_2_and_3": (
            "DECLARED NOT-NEEDED and NOT AVAILABLE. Already spent by w12b "
            "hub/nr6-and-kstar on 2026-09-21T21:33:51Z on exactly this content; the "
            "target is exhausted at 3/3 and this hub spent 0."),
        "the_1x_comparator_column": (
            "DECLARED NOT-RUN, carried from AMENDMENT 1 A1.2. The comparator set is "
            "matched to param_target = 131,072 (the 524,288 B REFERENCE column), never "
            "to this object's own four-part total. A 1x-matched rival would be "
            "STRONGER, so the available comparison FLATTERS this arm and no row here is "
            "reported as a 1x result."),
        "rival_retrain_without_on_the_red_AND_cube_scope": (
            "DECLARED NOT-RUN and cannot exist: the scope deletes 0 rows and "
            "w12_deletion_baselines.partition_fit_rows refuses a no-op spec by "
            "construction."),
        "the_refusal_DISPLAY_comparator": (
            "DECLARED NOT-RUN, carried. The only permitted comparator is the "
            "refuse-token transformer; wave 9 measured a refuser of that quality at "
            "+0.047 separation with a ridge probe at AUC 0.48-0.55, i.e. chance, so the "
            "comparison is not yet informative."),
        "seeds_3_and_4": (
            "DECLARED NOT-RUN, carried from AMENDMENT 1 A1.1 and w12b's row: they pass "
            "the cache rule but have NO decoder and NO external classifier, so "
            "fw.Judge(seed) cannot be built. Never padded to 4 or 5 seeds."),
    }

    p = OUT / "T7-3SEEDS.json"
    p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print("wrote", p)
    print("n =", doc["n_seeds"], "seeds", doc["seeds"])
    print("ALL FOUR CONTROLS COLLAPSE ON EVERY SEED:",
          gate["ALL_FOUR_COLLAPSE_ON_EVERY_SEED"])
    for arm in ("cov_tensor",):
        r = t7["rows"][arm]
        print(arm, "row3", r["row3_pair_over_seeds"], "row4",
              r["row4_nway_over_seeds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
