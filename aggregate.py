"""Aggregate the seed reports into RESULTS.md.

Every number is mean over seeds with the per-seed min/max in brackets, so a reader
can see the spread rather than a mean that hides it.  Nothing is smoothed and no
seed is dropped.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os

import numpy as np


def load(outdir, suffix):
    reps = []
    for path in sorted(glob.glob(os.path.join(outdir, "harness_seed*%s.json" % suffix))):
        if suffix == "" and "_easy" in path:
            continue
        with open(path) as f:
            reps.append(json.load(f))
    return sorted(reps, key=lambda r: r["seed"])


def ms(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return "n/a"
    if v.size == 1:
        return "%.4f" % v[0]
    return "%.4f [%.4f, %.4f]" % (v.mean(), v.min(), v.max())


def row(reps, getter):
    return ms([getter(r) for r in reps])


def table(reps, title):
    if not reps:
        return "_(NOT RUN)_\n"
    out = ["### %s - %d seed(s): %s\n" % (title, len(reps),
                                          ", ".join(str(r["seed"]) for r in reps)),
           "tier `%s`, generator %s, %d train / %d eval instances per seed\n"
           % (reps[0].get("tier", "cmaze1"),
              reps[0].get("generator_kwargs") or "(env-scout adopted defaults)",
              reps[0]["instances"]["n_train"], reps[0]["instances"]["n_eval"]),
           "| arm / column | params | validity (exact judge) | score | goal, budget lifted | legal prefix / opt | per-step acc |",
           "|---|---|---|---|---|---|---|"]
    named = [("mlp", "MLP"), ("lstm", "LSTM"), ("transformer", "Transformer"),
             ("table", "Table floor (raw state)"),
             ("table_enc", "Table-enc (encoder bits)")]
    for key, label in named:
        if key not in reps[0]["results"]:
            continue
        p = (reps[0]["training"][key]["params"]
             if key in reps[0].get("training", {}) else "-")
        out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            label, p,
            row(reps, lambda r: r["results"][key]["validity"]),
            row(reps, lambda r: r["results"][key]["score"]),
            row(reps, lambda r: r["results"][key]["goal_rate_budget_lifted"]),
            row(reps, lambda r: r["results"][key]["mean_legal_prefix_over_opt"]),
            row(reps, lambda r: r["results"][key].get("teacher_forced_action_acc"))))
    out.append("| | | | | | | |")
    for name in reps[0]["columns"]:
        tag = "**reported column**" if name.startswith(("blind", "fragment")) else ""
        out.append("| %s %s | - | %s | %s | %s | %s | - |" % (
            name, tag,
            row(reps, lambda r: r["columns"][name]["validity"]),
            row(reps, lambda r: r["columns"][name]["score"]),
            row(reps, lambda r: r["columns"][name]["goal_rate_budget_lifted"]),
            row(reps, lambda r: r["columns"][name]["mean_legal_prefix_over_opt"])))
    out.append("")
    if "refuser_headline" in reps[0]:
        out += ["**Supervised refuser** (the only comparator for DISPLAY rows; "
                "refusal read as the decision at t=0):\n",
                "| | refusal rate | validity when it acts |", "|---|---|---|",
                "| on IMPOSSIBLE (correct refusals) | %s | %s |" % (
                    row(reps, lambda r: r["refuser_headline"]["refusal_rate_on_impossible"]),
                    row(reps, lambda r: r["results"]["refuser_impossible"]["validity"])),
                "| on POSSIBLE (false refusals) | %s | %s |" % (
                    row(reps, lambda r: r["refuser_headline"]["refusal_rate_on_possible"]),
                    row(reps, lambda r: r["results"]["refuser_possible"]["validity"])),
                "| **separation** | %s | |" % row(
                    reps, lambda r: r["refuser_headline"]["separation"]),
                ""]
        anystep = [k for k in reps[0]["results"] if k.startswith("refuser_possible_any")]
        if anystep:
            out += ["Alternative reading, refusal allowed at any step: impossible %s, "
                    "possible %s.\n" % (
                        row(reps, lambda r: r["results"]["refuser_impossible_any_step"]["refusal_rate"]),
                        row(reps, lambda r: r["results"]["refuser_possible_any_step"]["refusal_rate"]))]
    return "\n".join(out) + "\n"


def audits(reps):
    if not reps:
        return "_(NOT RUN)_\n"
    ja = [r["judge_audit"] for r in reps]
    sa = [r["stepper_audit"] for r in reps]
    lines = ["| check | result |", "|---|---|",
             "| judge exact (all four parts) | **%s** |"
             % all(a["exact"] for a in ja),
             "| tolerance used anywhere | **None** |",
             "| A source: float machinery in judge | %s |"
             % ("none" if not any(a["source_findings"] for a in ja) else "FOUND"),
             "| B types: float contact under tracing ints | %s |"
             % ("none" if not any(a["type_findings"] for a in ja) else "FOUND"),
             "| C redundant: disagreements with independent replay | %d / %d routes |"
             % (sum(a["redundant_disagreements"] for a in ja),
                sum(a["n_route_pairs"] for a in ja)),
             "| D oracle: optimum / truncation / budget / mutants | %d failures |"
             % sum(a["oracle_failures"] for a in ja),
             "| stepper disagreements with the judge | %d / %d routes |"
             % (sum(s["disagreements"] for s in sa), sum(s["routes"] for s in sa)),
             "| encoder loss sees actions or judge | %s |"
             % any(r["encoder_independence"]["sees_actions"]
                   or r["encoder_independence"]["sees_judge"] for r in reps),
             "| action head identical shape across arms | %s |"
             % all(r["head_identity"]["identical_shape_across_arms"] for r in reps),
             "| impossible-set certification failures | %d |"
             % sum(r.get("impossible", {}).get("n_certification_failures", 0)
                   for r in reps),
             "| impossible examples seen by a non-refuser arm | %d |"
             % sum(r.get("impossible", {}).get("unseen_by_other_arms", {})
                   .get("impossible_in_arm_data", 0) for r in reps),
             ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/c/chlu/outputs/w9-harness")
    a = ap.parse_args()
    main_reps = load(a.out, "")
    easy_reps = load(a.out, "_easy")
    doc = ["# w9 maze harness - results\n",
           "Scored by `w9_harness.judge_maze.JUDGE` - env-scout's exact integer "
           "judge, re-exported unwrapped. No tolerance anywhere.\n",
           "## Audits\n", audits(main_reps or easy_reps),
           "## CMAZE-1, env-scout adopted defaults\n", table(main_reps, "Main tier"),
           "## CMAZE-1-EASY (declared tier: h=w=4, k=2, same generator, same judge)\n",
           table(easy_reps, "Easy tier")]
    path = os.path.join(a.out, "RESULTS.md")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print("\n".join(doc))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
