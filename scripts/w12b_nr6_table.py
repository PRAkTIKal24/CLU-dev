"""w12b hub/nr6-and-kstar -- NR6's table, built from the scored artefacts and nothing else.

⛔ This module RE-DERIVES nothing from memory.  Every cell names the file it came from,
and a cell with no file is printed as a DECLARED NOT-RUN, never as a zero and never as a
blank.  The w12 hub's directory is READ; it is never written.

Three things are assembled here:

1. **T7 across seeds.** Rows 3 and 4 at K* = 32 on TEST.  Seed 0 is the w12 hub's own
   `run1_seed0_K32/scored.json`; seeds 1 and 2 are this hub's run 3/3.  The seed count is
   printed on the table, because T7 printed `n = 1` until now.

2. **The control gate.** All four charter-C controls, on every seed of every row.
   ⛔ If any control fails to collapse on any seed, the row is NOT claimed -- that is the
   result, and this script prints it as the verdict rather than footnoting it.

3. **The val/test gap at the selection rule's own quantity.**  The rule that chose K* was
   `K* = max{K : v(5) - mean_seeds v(K) <= 0.02}` where `v` is the mean over corners of the
   per-corner valid fraction.  On TEST that same statistic is the mean over corners of
   `judge_accuracy` (`fw._per_corner`: `judge_accuracy = valid.mean()` per corner, and the
   val sweep's `valid_mean` is the mean of exactly that over its corners -- the same
   number computed by the same expression).  ⚠ The CORNER SETS DIFFER (24 val corners vs
   60 + 49 test corners), so the LEVELS are not comparable and are never subtracted across
   conditions.  What IS comparable, and what the rule actually used, is the DROP from
   K = 5 to K = 32 within one condition.  That is the gap this script reports.
"""
from __future__ import annotations

import argparse
import json
import statistics as S
from pathlib import Path

W12 = Path("C:/chlu/outputs/w12-single-design-crossover")          # READ ONLY
MY = Path("C:/chlu/outputs/w12b-nr6-and-kstar")
CONTROLS = ("landscape_removed", "landscape_shuffled",
            "landscape_shuffled_across_wells", "random_atoms")
NOT_RUN = "DECLARED NOT-RUN"


def load(p: Path):
    if not p.exists():
        return None, f"{NOT_RUN}: {p} does not exist"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d, str(p)


def corner_mean(d: dict, arm: str, field: str, family=None):
    per = ((d.get("per_arm") or {}).get(arm) or {}).get("per_corner")
    if not per:
        return None
    sel = [r for r in per if family is None or r["family"] == family]
    if not sel:
        return None
    return float(S.fmean(float(r[field]) for r in sel))


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MY))
    a = ap.parse_args(argv)
    out = Path(a.out)

    sources = {
        ("K32", 0): W12 / "scored" / "run1_seed0_K32" / "scored.json",
        ("K32", 1): MY / "scored" / "run3_seed1_K32" / "scored.json",
        ("K32", 2): MY / "scored" / "run3_seed2_K32" / "scored.json",
        ("K5", 0): MY / "scored" / "run2_seed0_K5" / "scored.json",
        ("K5", 1): MY / "scored" / "run2_seed1_K5" / "scored.json",
        ("K5", 2): MY / "scored" / "run2_seed2_K5" / "scored.json",
    }
    docs, where, missing = {}, {}, {}
    for k, p in sources.items():
        d, w = load(p)
        if d is None:
            missing[f"{k[0]}_seed{k[1]}"] = w
        else:
            docs[k], where[f"{k[0]}_seed{k[1]}"] = d, w

    report = {"SCHEMA": "chlu-w12b-nr6/1",
              "hub": "w12b hub/nr6-and-kstar",
              "sources": where, "DECLARED_NOT_RUN": missing,
              "arms": {}, "controls": {}, "verdict": {}}

    # ---- 1. every arm, every K, every seed -------------------------------------
    arm_names = []
    for d in docs.values():
        for nm in (d.get("per_arm") or {}):
            if nm not in arm_names:
                arm_names.append(nm)
    for nm in arm_names:
        rec = {}
        for (K, seed), d in sorted(docs.items()):
            pa = (d.get("per_arm") or {}).get(nm)
            if pa is None:
                rec[f"{K}_seed{seed}"] = NOT_RUN + ": arm absent from this leg"
                continue
            rec[f"{K}_seed{seed}"] = {
                "row3_pair_novel_mean": pa["row3_pair"].get("novel_per_corner_mean"),
                "row4_nway_novel_mean": pa["row4_nway"].get("novel_per_corner_mean"),
                "all_novel_mean": pa["all"].get("novel_per_corner_mean"),
                "judge_accuracy_mean": corner_mean(d, nm, "judge_accuracy"),
                "n_corners": pa["all"].get("n_corners"),
            }
        report["arms"][nm] = rec

    # ---- 2. the control gate ----------------------------------------------------
    gate_ok, gate_rows = True, {}
    for (K, seed), d in sorted(docs.items()):
        arm = corner_mean(d, "cov_tensor", "distinct_valid_novel_tuples")
        row = {"arm_novel_mean": arm}
        for c in CONTROLS:
            v = corner_mean(d, c, "distinct_valid_novel_tuples")
            row[c] = v
            if v is None:
                row[c] = NOT_RUN + ": control absent from this leg"
                gate_ok = False
            elif v != 0.0:
                gate_ok = False
        row["all_four_collapse_to_exactly_zero"] = all(
            isinstance(row[c], float) and row[c] == 0.0 for c in CONTROLS)
        gate_rows[f"{K}_seed{seed}"] = row
    report["controls"] = {
        "per_leg": gate_rows,
        "rule": ("charter C.2. NO ROW IS CLAIMED unless every control collapses on "
                 "every seed of that row. A control that collapsed at seed 0 is not "
                 "assumed to collapse at seeds 1 and 2."),
        "ALL_LEGS_PASS": bool(gate_ok),
    }

    # ---- 3. T7, the headline row, with its seed count ----------------------------
    for K in ("K32", "K5"):
        seeds = sorted(s for (kk, s) in docs if kk == K)
        if not seeds:
            report["verdict"][K] = NOT_RUN + ": no leg at this K"
            continue
        r3 = [docs[(K, s)]["per_arm"]["cov_tensor"]["row3_pair"]["novel_per_corner_mean"]
              for s in seeds]
        r4 = [docs[(K, s)]["per_arm"]["cov_tensor"]["row4_nway"]["novel_per_corner_mean"]
              for s in seeds]
        ja = [corner_mean(docs[(K, s)], "cov_tensor", "judge_accuracy") for s in seeds]
        report["verdict"][K] = {
            "n_seeds": len(seeds), "seeds": seeds,
            "row3_pair_per_seed": r3,
            "row3_pair_mean": float(S.fmean(r3)),
            "row3_pair_sd": float(S.stdev(r3)) if len(r3) > 1 else None,
            "row4_nway_per_seed": r4,
            "row4_nway_mean": float(S.fmean(r4)),
            "row4_nway_sd": float(S.stdev(r4)) if len(r4) > 1 else None,
            "judge_accuracy_per_seed": ja,
            "judge_accuracy_mean": float(S.fmean(ja)),
            "SEED_COUNT_IS_PRINTED": True,
            "NEVER_PADDED": ("seeds 3 and 4 pass the cache rule and have NO decoder and "
                             "NO external classifier; neither can be judged on the one "
                             "instrument every other number used"),
        }

    # ---- 4. the val/test gap at the rule's own quantity --------------------------
    val, vw = load(W12 / "CROSSOVER-val.json")
    gap = {"quantity": ("mean over corners of the per-corner VALID FRACTION "
                        "(val: `valid_mean`; test: mean of `judge_accuracy`) -- the same "
                        "expression, `valid.mean()` per corner, averaged over corners"),
           "WARNING": ("the corner SETS differ (24 val corners vs 60 + 49 test corners), "
                       "so levels are NOT comparable across conditions and are never "
                       "subtracted across them. The DROP from K = 5 to K = 32 within one "
                       "condition IS the quantity the frozen selection rule used, and "
                       "that is what is compared."),
           "rule": "K* = max{K : v(5) - mean_seeds v(K) <= 0.02}, frozen 11:10:24Z",
           "band": 0.02}
    if val:
        gap["val_source"] = vw
        pk = val["per_K"]
        gap["val"] = {
            "K5_mean": pk["5"]["val_composition_mean"],
            "K32_mean": pk["32"]["val_composition_mean"],
            "K5_per_seed": pk["5"]["val_composition_per_seed"],
            "K32_per_seed": pk["32"]["val_composition_per_seed"],
            "drop_K5_minus_K32": (pk["5"]["val_composition_mean"]
                                  - pk["32"]["val_composition_mean"]),
        }
    t5 = [corner_mean(docs[("K5", s)], "cov_tensor", "judge_accuracy")
          for s in sorted(s for (k, s) in docs if k == "K5")]
    t32 = [corner_mean(docs[("K32", s)], "cov_tensor", "judge_accuracy")
           for s in sorted(s for (k, s) in docs if k == "K32")]
    if t5 and t32:
        common = sorted(set(s for (k, s) in docs if k == "K5")
                        & set(s for (k, s) in docs if k == "K32"))
        gap["test"] = {
            "K5_per_seed": {str(s): corner_mean(docs[("K5", s)], "cov_tensor",
                                                "judge_accuracy")
                            for s in sorted(s for (k, s) in docs if k == "K5")},
            "K32_per_seed": {str(s): corner_mean(docs[("K32", s)], "cov_tensor",
                                                 "judge_accuracy")
                             for s in sorted(s for (k, s) in docs if k == "K32")},
            "K5_mean": float(S.fmean(t5)), "K32_mean": float(S.fmean(t32)),
            "drop_K5_minus_K32_all_available_seeds": float(S.fmean(t5) - S.fmean(t32)),
            "paired_seeds_used_for_the_drop": common,
            "drop_K5_minus_K32_paired": (
                float(S.fmean([corner_mean(docs[("K5", s)], "cov_tensor",
                                           "judge_accuracy")
                               - corner_mean(docs[("K32", s)], "cov_tensor",
                                             "judge_accuracy") for s in common]))
                if common else NOT_RUN + ": no seed has both K"),
        }
        d_ = gap["test"]["drop_K5_minus_K32_paired"]
        if isinstance(d_, float):
            gap["TEST_DROP_WITHIN_THE_0.02_BAND"] = bool(d_ <= 0.02)
            gap["reading"] = (
                "the selection rule's band HOLDS on TEST" if d_ <= 0.02 else
                "⛔ the selection rule's band FAILS on TEST: K* was chosen on a "
                "quantity that does not transfer, and K*'s provenance is damaged")
    else:
        gap["test"] = NOT_RUN + ": one of the two K's has no TEST leg"
    report["val_test_gap"] = gap

    # ---- 5. the two token families, re-fitted post the dtype fix ------------------
    tok = {}
    for (K, seed), d in sorted(docs.items()):
        rec = {"refit_meta": d.get("token_rivals_refit_postfix")}
        for fam in ("transformer", "lstm"):
            loaded = corner_mean(d, f"{fam}__as_trained",
                                 "distinct_valid_novel_tuples")
            refit = corner_mean(d, f"{fam}__refit_postfix",
                                "distinct_valid_novel_tuples")
            rec[fam] = {
                "loaded_comparator_novel_mean": (
                    loaded if loaded is not None
                    else NOT_RUN + ": not in this leg"),
                "refitted_here_novel_mean": (
                    refit if refit is not None
                    else NOT_RUN + ": --refit-tokens produced no arm"),
                "difference_refit_minus_loaded": (
                    (refit - loaded) if (refit is not None and loaded is not None)
                    else NOT_RUN),
            }
        tok[f"{K}_seed{seed}"] = rec
    report["token_families_pre_vs_post_fix"] = {
        "per_leg": tok,
        "provenance": ("the dtype repair `_as_model_input` is at 1635e0a, an ANCESTOR of "
                       "this base e58ab05, so both the loaded comparators (fitted by w12 "
                       "spoke/deletion-baselines on that same commit) and the arms "
                       "re-fitted in-process POST-DATE the fix. The second, functionally "
                       "equivalent repair 8e5f202 is on w13/encoder-adapt and is NOT in "
                       "this base."),
        "why_it_matters": ("w12 measured that before the repair `eqx.nn.Embedding` "
                           "refused to index a float32 array, so `transformer` and "
                           "`lstm` could not be FITTED at all. Spoke A's wave-4 rival "
                           "set -- the one the LADDER scores, `rival_transformer` and "
                           "`rival_lstm` -- pre-dates the repair and its flat scores may "
                           "be artefacts. That set is NOT used on these rows."),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "NR6-T7.json").write_text(json.dumps(report, indent=1, default=str),
                                     encoding="utf-8")

    # ---- the printable table -----------------------------------------------------
    L = []
    L.append("# NR6 — T7 with its seed count printed\n")
    L.append("Built by `scripts/w12b_nr6_table.py` from the scored artefacts only.\n")
    L.append("## Sources\n")
    for k, v in where.items():
        L.append(f"- `{k}` → `{v}`")
    for k, v in missing.items():
        L.append(f"- `{k}` → **{v}**")
    L.append("\n## The control gate (charter C.2) — checked BEFORE any margin\n")
    L.append("| leg | arm | " + " | ".join(f"`{c}`" for c in CONTROLS) + " | all collapse? |")
    L.append("|---|---|" + "---|" * (len(CONTROLS) + 1))
    for leg, row in gate_rows.items():
        L.append("| `%s` | %s | %s | **%s** |"
                 % (leg, fmt(row["arm_novel_mean"]),
                    " | ".join(fmt(row[c]) for c in CONTROLS),
                    "YES" if row["all_four_collapse_to_exactly_zero"] else "⛔ NO"))
    L.append(f"\n**ALL LEGS PASS: {report['controls']['ALL_LEGS_PASS']}**  "
             "⛔ If this is False no row is claimed — that is the result.\n")
    L.append("## Every arm, every K, every seed — `distinct_valid_novel_tuples` per corner\n")
    legs = sorted(where)
    L.append("| arm | row | " + " | ".join(f"`{l}`" for l in legs) + " |")
    L.append("|---|---|" + "---|" * len(legs))
    for nm in arm_names:
        for rowkey, label in (("row3_pair_novel_mean", "row 3 (pair)"),
                              ("row4_nway_novel_mean", "row 4 (n-way)")):
            cells = []
            for l in legs:
                r = report["arms"][nm].get(l)
                cells.append(fmt(r[rowkey]) if isinstance(r, dict) else NOT_RUN)
            L.append(f"| `{nm}` | {label} | " + " | ".join(cells) + " |")
    L.append("\n## T7, with the count\n")
    for K, v in report["verdict"].items():
        if not isinstance(v, dict):
            L.append(f"- **{K}** — {v}")
            continue
        L.append(f"- **{K}: n = {v['n_seeds']} seeds {v['seeds']}** — "
                 f"row 3 {fmt(v['row3_pair_mean'], 3)} ± "
                 f"{fmt(v['row3_pair_sd'], 3) if v['row3_pair_sd'] else 'n/a'}, "
                 f"row 4 {fmt(v['row4_nway_mean'], 3)} ± "
                 f"{fmt(v['row4_nway_sd'], 3) if v['row4_nway_sd'] else 'n/a'}; "
                 f"per seed row 3 {[round(x, 3) for x in v['row3_pair_per_seed']]}, "
                 f"row 4 {[round(x, 3) for x in v['row4_nway_per_seed']]}")
    L.append("\n## The val/test gap at the selection rule's own quantity\n")
    L.append("```\n" + json.dumps(gap, indent=1, default=str) + "\n```")
    (out / "NR6-T7.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out / 'NR6-T7.json'} and {out / 'NR6-T7.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
