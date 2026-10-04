"""Assemble VAL-SMOKE.json and HARNESS-READY.json from the per-seed artefacts.

    python scripts/w15_cmaze_t_ready.py <out_dir>

⛔ Nothing is computed here that ``w15_cmaze_t_run.py`` did not measure.  This file
collates, checks the six acceptance conditions of the task file against the bars
PREREG.md declared, decides the instrument warnings, and states the not-runs.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
CLU = "CLU (single design)"
BLIND_PLANNERS = ["column: BLIND-COLOUR (planner)",
                  "column: BLIND-SEQUENTIAL (planner)"]
BLIND_LEARNED = ["arm: BLIND-COLOUR (learned: colour identity removed from the request)",
                 "arm: BLIND-SEQUENTIAL (learned: main effects only, "
                 "no interaction terms)"]
CONTROLS = ["control: landscape_removed", "control: landscape_shuffled",
            "control: landscape_shuffled_across_wells", "control: random_atoms"]
FLOORS = ["floor: lookup cross-maze WORDS", "floor: lookup cross-maze raw r1",
          "floor: fragment cross-maze r1", "floor: lookup WITHIN-maze raw r1"]
RIVALS = ["rival: mlp", "rival: lstm", "rival: transformer"]
ROWS = ("seen", "pair", "nway", "value")
ROW_TITLE = {"seen": "seen", "pair": "unseen 2-way", "nway": "unseen n-way",
             "value": "value hold-out (never-seen colour)"}
REPLAY_BAR = 50000
IMPOSSIBLE_BAR = 50000
CONTROL_COLLAPSE_MARGIN = 0.15
BLIND_NEAR_MARGIN = 0.05
def _git(*a):
    try:
        return subprocess.check_output(("git",) + a, text=True).strip()
    except Exception as exc:                                    # pragma: no cover
        return "unavailable: %s" % exc


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sd(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def arm_names(seeds):
    names = []
    for d in seeds:
        for row in ROWS:
            blk = d["ROWS"].get(row, {})
            for k in blk:
                if k not in names and isinstance(blk[k], dict) and "exact" in blk[k]:
                    names.append(k)
    order = [CLU] + BLIND_LEARNED + RIVALS + FLOORS + CONTROLS + BLIND_PLANNERS
    return [n for n in order if n in names] + [n for n in names if n not in order]


def collect(out_dir):
    seeds = []
    for p in sorted(glob.glob(os.path.join(out_dir, "val_seed*.json"))):
        seeds.append(json.load(open(p)))
    seeds.sort(key=lambda d: d["seed_index"])
    return seeds


def val_smoke(seeds):
    names = arm_names(seeds)
    table = {}
    for row in ROWS:
        rblk = {}
        ns = [d["ROWS"].get(row, {}).get(CLU, {}).get("n") for d in seeds]
        for nm in names:
            vals = [d["ROWS"].get(row, {}).get(nm, {}).get("exact") for d in seeds]
            rblk[nm] = {"per_seed": vals, "mean": _mean(vals), "sd": _sd(vals),
                        "n_seeds": len([v for v in vals if v is not None])}
        ceil = [d["ROWS"].get(row, {}).get("column: CODEC CEILING", {}).get("exact")
                for d in seeds]
        rblk["column: CODEC CEILING (pipeline ceiling)"] = {
            "per_seed": ceil, "mean": _mean(ceil), "sd": _sd(ceil),
            "n_seeds": len([v for v in ceil if v is not None])}
        table[row] = {"title": ROW_TITLE[row], "n_rows_scored_per_seed": ns,
                      "arms": rblk}
    roll_names, roll = [], {}
    for d in seeds:
        for k in d["ROLLOUT"]["arms"]:
            if k not in roll_names:
                roll_names.append(k)
    for nm in roll_names:
        for field in ("validity", "per_step_exactness", "all_steps_exact_rate"):
            vals = [d["ROLLOUT"]["arms"].get(nm, {}).get(field) for d in seeds]
            roll.setdefault(nm, {})[field] = {"per_seed": vals, "mean": _mean(vals),
                                              "sd": _sd(vals)}
    return {
        "n_seeds": len(seeds),
        "seed_indices": [d["seed_index"] for d in seeds],
        "tolerance_used": None,
        "scored_runs_spent": 0,
        "scored_column": ("`exact` -- the fraction of transitions whose generated next "
                          "state the exact transition judge accepts (legality, "
                          "displacement, keystate and pocket, all by integer "
                          "comparison)"),
        "rows": table,
        "rollout": {
            "definition": seeds[0]["ROLLOUT"]["chaining"],
            "route_length": seeds[0]["ROLLOUT"]["route_length"],
            "n_rollouts_per_seed": [d["ROLLOUT"]["n_rollouts"] for d in seeds],
            "arms": roll},
        "config_A_per_seed": [d["config_A"] for d in seeds],
        "clu_ledger_per_seed": [d["clu_ledger"] for d in seeds],
        "matched_parameters_per_seed": [d["matched_parameters"] for d in seeds],
        "deletion_per_seed": [d["DELETION"] for d in seeds],
        "holdout_per_seed": [d["holdout"] for d in seeds],
        "banks_per_seed": [d["banks"] for d in seeds],
        "rule_b_per_seed": [d["ACCEPT_4_rule_b_bites"] for d in seeds],
        "frozen_codec_per_seed": [d["frozen_codec"] for d in seeds],
        "frozen_state_encoder_per_seed": [d["frozen_state_encoder"] for d in seeds],
        "clu_val_sweep_per_seed": [d["clu_val_sweep"] for d in seeds],
    }


def build(out_dir):
    seeds = collect(out_dir)
    assert seeds, "no val_seed*.json found in %s" % out_dir
    vs = val_smoke(seeds)

    # ---- acceptance 1 --------------------------------------------------
    replayed = sum(d["ACCEPT_1_independent_replay"]["transitions_replayed"]
                   for d in seeds)
    disagree = sum(d["ACCEPT_1_independent_replay"]["disagreements"] for d in seeds)
    a1 = {"pass": bool(disagree == 0 and replayed >= REPLAY_BAR),
          "transitions_replayed": replayed, "disagreements": disagree,
          "bar": "0 disagreements over >= %d transitions" % REPLAY_BAR,
          "per_seed": [{"seed": d["seed_index"],
                        "replayed": d["ACCEPT_1_independent_replay"]
                        ["transitions_replayed"],
                        "disagreements": d["ACCEPT_1_independent_replay"]
                        ["disagreements"]} for d in seeds],
          "drivers": seeds[0]["ACCEPT_1_independent_replay"]["drivers"]}

    # ---- acceptance 2 --------------------------------------------------
    checked = sum(d["ACCEPT_2_no_impossible_example"]["rows_checked"] for d in seeds)
    bad = sum(d["ACCEPT_2_no_impossible_example"]["rows_with_no_accepting_emission"]
              for d in seeds)
    a2 = {"pass": bool(bad == 0 and checked >= IMPOSSIBLE_BAR),
          "rows_checked": checked, "rows_with_no_accepting_emission": bad,
          "bar": "0 impossible rows over >= %d rows checked" % IMPOSSIBLE_BAR,
          "definition": seeds[0]["ACCEPT_2_no_impossible_example"]["definition"],
          "refusal_display": ("RETIRED.  The value hold-out is SCORED, not displayed; "
                              "no arm in this harness has a refuse token.")}

    # ---- acceptance 3 --------------------------------------------------
    a3 = {"per_seed": [d["ACCEPT_3_holdout_disjointness"] for d in seeds],
          "words_determine_transition":
              [d["ACCEPT_3b_words_determine_transition"] for d in seeds],
          "pass": bool(all(d["ACCEPT_3_holdout_disjointness"]["pass"] for d in seeds)
                       and all(d["ACCEPT_3b_words_determine_transition"]["pass"]
                               for d in seeds)),
          "bar": ("the value hold-out colour never appears in a train rule; no held-out "
                  "n-way or 2-way combination occurs in train; the five words determine "
                  "the transition")}

    # ---- acceptance 4 (REPORTED, never a gate) --------------------------
    a4 = {"REPORTED_COLUMN_NEVER_AN_ABORT_CONDITION": True,
          "rule_b_renders_the_action_ILLEGAL": {
              "stratified_bank_train_fraction":
                  [d["ACCEPT_4_rule_b_bites"]["stratified_bank_train"]["fraction"]
                   for d in seeds],
              "stratified_bank_val_fraction":
                  [d["ACCEPT_4_rule_b_bites"]["stratified_bank_val"]["fraction"]
                   for d in seeds],
              "natural_distribution_train_fraction":
                  [d["ACCEPT_4_rule_b_bites"]["natural_distribution_train"]["fraction"]
                   for d in seeds],
              "natural_distribution_val_fraction":
                  [d["ACCEPT_4_rule_b_bites"]["natural_distribution_val"]["fraction"]
                   for d in seeds]},
          "definition": seeds[0]["ACCEPT_4_rule_b_bites"]["stratified_bank_train"]
          ["note"],
          "which_is_which": seeds[0]["ACCEPT_4_rule_b_bites"]["which_is_which"],
          "three_way_configuration_count_per_seed_val_bank":
              [d["ACCEPT_4_rule_b_bites"]["stratified_bank_val"]
               ["n_door_eq_pocket_eq_tint"] for d in seeds]}

    # ---- acceptance 5, 6 ------------------------------------------------
    a5 = {"pass": bool(len(seeds) == 3),
          "n_seeds": len(seeds), "seed_indices": [d["seed_index"] for d in seeds],
          "rows_present": sorted(vs["rows"].keys()) + ["rollout"],
          "arms_present": arm_names(seeds),
          "artefact": "VAL-SMOKE.json",
          "bar": "3 seeds, all or none; all rows; every rival"}
    a6 = {"pass": os.path.exists(os.path.join(out_dir, "PREREG.md")),
          "artefact": "PREREG.md",
          "sha256": _sha(os.path.join(out_dir, "PREREG.md"))
          if os.path.exists(os.path.join(out_dir, "PREREG.md")) else None}

    # ---- instrument warnings -------------------------------------------
    warnings = []
    nway = vs["rows"]["nway"]["arms"]
    clu_nway = nway[CLU]["mean"]
    for nm in BLIND_LEARNED:
        v = nway.get(nm, {}).get("mean")
        if v is not None and clu_nway is not None and v >= clu_nway - BLIND_NEAR_MARGIN:
            warnings.append({
                "warning": "composition_blind_scores_near_the_clu_on_unseen_nway",
                "arm": nm, "blind_mean": v, "clu_mean": clu_nway,
                "margin_used": BLIND_NEAR_MARGIN,
                "reading": ("a LEARNED composition-blind policy matches the CLU on the "
                            "unseen n-way row.  On this row the instrument is not "
                            "separating composition from the blind policy's own "
                            "generalisation, and that is said plainly here rather than "
                            "hidden."),
                "SHIPPED_ANYWAY": True})
    for nm in BLIND_PLANNERS:
        v = nway.get(nm, {}).get("mean")
        if v is not None and clu_nway is not None and v >= clu_nway - BLIND_NEAR_MARGIN:
            warnings.append({
                "warning": "composition_blind_PLANNER_scores_near_the_clu_on_unseen_nway",
                "arm": nm, "blind_mean": v, "clu_mean": clu_nway,
                "margin_used": BLIND_NEAR_MARGIN,
                "reading": ("⚠ READ THIS ONE CAREFULLY.  The BLIND-* PLANNER columns "
                            "are rule-KNOWING relaxed oracles: they read the grid, they "
                            "are never trained, and so they never have to generalise to "
                            "a held-out combination at all.  They are an UPPER BOUND on "
                            "what a composition-blind policy could do, not a policy.  "
                            "The falsifier the task file names is the LEARNED "
                            "BLIND-COLOUR / BLIND-SEQUENTIAL arm, reported separately."),
                "SHIPPED_ANYWAY": True})
    for row in ROWS:
        blk = vs["rows"][row]["arms"]
        c = blk[CLU]["mean"]
        for ctl in CONTROLS:
            v = blk.get(ctl, {}).get("mean")
            if v is None or c is None:
                continue
            if v > c - CONTROL_COLLAPSE_MARGIN:
                warnings.append({
                    "warning": "charter_C_control_did_not_collapse",
                    "row": row, "control": ctl, "control_mean": v, "clu_mean": c,
                    "margin_used": CONTROL_COLLAPSE_MARGIN,
                    "reading": ("on this row no STORE claim is available; it is an "
                                "INTERFACE claim or nothing.")})
    for row in ROWS:
        blk = vs["rows"][row]["arms"]
        c = blk[CLU]["mean"]
        v = blk.get("floor: lookup cross-maze WORDS", {}).get("mean")
        if v is not None and c is not None and v >= c - BLIND_NEAR_MARGIN:
            warnings.append({
                "warning": "cross_maze_lookup_floor_scores_near_the_clu",
                "row": row, "floor_mean": v, "clu_mean": c,
                "reading": "on this row the result is retrieval, not composition."})

    status_ok = a1["pass"] and a2["pass"] and a3["pass"] and a5["pass"] and a6["pass"]

    res = {
        "artefact": "HARNESS-READY.json",
        "spoke": "w15/cmaze-t-harness",
        "stamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ready" if status_ok else "NOT-READY",
        "tolerance_used": None,
        "scored_runs_spent": 0,
        "one_line": ("CMAZE-T -- the compositional TRANSITION instrument over CMAZE-1.  "
                     "Given (grid state, action) an arm generates the next state and the "
                     "ENGINE judges it exactly: legality under RULES A+B, displacement, "
                     "keystate and pocket, all by integer comparison.  Engine, solver "
                     "and exact judge are w8_envscout.maze_colour, imported unmodified."),

        "provenance": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "base": "148f3b7 [hub] base/iter8: fold w13/referee-2",
            "worktree": os.getcwd().replace("\\", "/"),
            "python": sys.version.split()[0],
            "engine": {
                "module": "w8_envscout.maze_colour",
                "generator": ("generate() at ITS OWN DEFAULTS (h=w=7, n_colours=4, "
                              "braid=0.40, p_tint=0.22, adversarial=2, no acceptance "
                              "filter)"),
                "sha256": _sha(os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "w8_envscout", "maze_colour.py")),
                "modified": False},
            "stepper": {
                "module": "w9_harness.dynamics",
                "sha256": _sha(os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "w9_harness", "dynamics.py")),
                "modified": False},
            "second_replay": {
                "module": "w9_harness.judge_maze (judge_ref)",
                "sha256": _sha(os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "w9_harness", "judge_maze.py")),
                "modified": False},
            "cmaze_t_layer": {
                "modules": ["chlu/exp_w15_cmaze_t.py", "chlu/exp_w15_cmaze_t_arms.py",
                            "scripts/w15_cmaze_t_run.py",
                            "scripts/w15_cmaze_t_ready.py"],
                "sha256": {f: _sha(os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), f))
                    for f in ("chlu/exp_w15_cmaze_t.py",
                              "chlu/exp_w15_cmaze_t_arms.py",
                              "scripts/w15_cmaze_t_run.py")}},
            "THE_ONE_DISTRIBUTIONAL_CHANGE": {
                "what": ("the request layer tints a fraction of DOOR cells, then hands "
                         "the instance to the unchanged engine, solver and judge"),
                "why": ("maze_colour.generate zeroes the tint on start, goal, door and "
                        "key cells, so a door cell is never tinted (measured: 160 door "
                        "cells over 40 generator seeds, 0 tinted).  RULES A and B can "
                        "then never interact WITHIN one transition, BLIND-SEQUENTIAL is "
                        "vacuous at the transition level, and a transition instrument "
                        "with no within-step interaction is not a composition "
                        "instrument."),
                "legitimacy": ("the RULES already specify the case -- judge, step, "
                               "judge_ref and solve all resolve the door first and then "
                               "test the tint against the UPDATED pocket.  Only the "
                               "generator declines to emit it.  Same class of move as "
                               "w9_maze.task._poison_pairs and _seal."),
                "safety": ("every tinted instance is re-solved with the UNMODIFIED "
                           "solver; a tint that kills the oracle is reverted"),
                "receipts_per_seed": [d["banks"]["train"]["door_tint"] for d in seeds]},
        },

        "ACCEPTANCE": {
            "1__independent_replay_zero_disagreements": a1,
            "2__no_impossible_example_shown_to_a_non_refuser_arm": a2,
            "3__holdouts_disjoint_from_train_by_construction": a3,
            "4__rule_B_bites_REPORTED_COLUMN": a4,
            "5__seeded_VAL_smoke_3_seeds_all_rows_every_rival": a5,
            "6__PREREG_before_any_number": a6,
        },

        "instrument_warnings": warnings if warnings else [
            {"warning": None,
             "reading": "no declared instrument warning fired"}],

        "VAL_SMOKE_SUMMARY": {
            row: {"title": ROW_TITLE[row],
                  "n_rows_scored_per_seed": vs["rows"][row]["n_rows_scored_per_seed"],
                  "arms": {nm: {"mean": vs["rows"][row]["arms"][nm]["mean"],
                                "sd": vs["rows"][row]["arms"][nm]["sd"],
                                "per_seed": vs["rows"][row]["arms"][nm]["per_seed"]}
                           for nm in vs["rows"][row]["arms"]}}
            for row in ROWS},
        "ROLLOUT_SUMMARY": vs["rollout"],

        "DECLARED_NOT_RUN": [
            ("Any SCORED run.  The task file budgets ZERO for this spoke; the val smoke "
             "is free.  No row of /c/chlu/SCORED-LEDGER.md is claimed."),
            ("The TEST maze block.  Its generator seed range is recorded per seed in "
             "banks.test_RESERVED_NOT_EVALUATED and it is never built or evaluated -- "
             "it is wave 16's to spend."),
            ("A refusal display.  Refusal displays are retired; the value hold-out is "
             "SCORED and no arm carries a refuse token."),
            ("Any change to w8_envscout.maze_colour, w9_harness or w9_maze.  They are "
             "imported; their file sha256 values are in provenance so the claim is "
             "checkable rather than asserted."),
            ("GPU and cluster.  Local CPU only, three concurrent seed processes, "
             "OMP/MKL threads pinned to 4."),
        ],
    }
    return vs, res


def main(out_dir):
    vs, res = build(out_dir)
    with open(os.path.join(out_dir, "VAL-SMOKE.json"), "w") as fh:
        json.dump(vs, fh, indent=1)
    with open(os.path.join(out_dir, "HARNESS-READY.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote VAL-SMOKE.json and HARNESS-READY.json; status =", res["status"])
    for k, v in res["ACCEPTANCE"].items():
        p = v.get("pass")
        print("  %-62s %s" % (k, "PASS" if p else ("REPORTED" if p is None else "FAIL")))
    for w in res["instrument_warnings"]:
        print("  WARNING:", w.get("warning"))


if __name__ == "__main__":
    main(sys.argv[1])
