"""Assemble REQUEST-LAYER-READY.json from the measured artefacts.

Nothing is computed here that was not measured by ``w14_request_layer.measure`` or
``.probes``; this file collates, checks the four acceptance conditions, and states the
not-runs.  A published file is not a finished file: the checks below are what make it one.

Usage:  python -m w14_request_layer.readiness <out_dir>
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

SEQ = "BLIND-SEQUENTIAL(key order from A)"
ORACLE = "ORACLE          (full composition)"
BLIND = ["BLIND-A         (rule B dropped)", "BLIND-B         (rule A dropped)",
         "BLIND-COLOUR    (colours collapsed)", "BLIND-FACTOR    (A x B local only)",
         SEQ]
ROWS = ("seen", "unseen_pair", "nway", "budget_interp")
HELD = ("unseen_pair", "nway", "budget_interp")
PUBLISHED_BLIND_SEQUENTIAL = 0.4980
PUBLISHED_FRAG_CROSS = (0.0006, 0.0000)
W9_CMAZE2_BLIND_SEQUENTIAL = [0.9077, 0.8905, 0.9130]


def _git(*a):
    try:
        return subprocess.check_output(("git",) + a, text=True).strip()
    except Exception as exc:                                    # pragma: no cover
        return "unavailable: %s" % exc


def _short(name):
    return name.split("(")[0].strip()


def _hist_summary(hist):
    tot = sum(hist.values())
    multi = sum(v for k, v in hist.items() if int(k) >= 2)
    return {"histogram": {str(k): hist[k]
                          for k in sorted(hist, key=lambda x: int(x))},
            "n_routes": tot, "n_routes_multi_key": multi,
            "fraction_multi_key": (multi / tot) if tot else None}


def collect(out_dir):
    art = os.path.join(out_dir, "artefacts")
    anchor = json.load(open(os.path.join(art, "anchor_cmaze1.json")))
    runs = {}
    for p in sorted(glob.glob(os.path.join(art, "measure_*.json"))):
        d = json.load(open(p))
        runs["%s_seed%d" % (d["which"], d["seed_index"])] = d
    return anchor, runs


def _row_block(m, r):
    if "blind" not in m.get(r, {}):
        return {"n": 0, "note": m.get(r, {}).get("note", "empty row set")}
    b = m[r]["blind"]
    out = {"n": b[SEQ]["n"], "ORACLE": b[ORACLE]["validity"]}
    out.update({_short(k): b[k]["validity"] for k in BLIND})
    out.update({"BLIND-SEQUENTIAL_bend": b[SEQ]["validity_bend"],
                "BLIND-SEQUENTIAL_nobend": b[SEQ]["validity_nobend"],
                "rule_b_bend_fraction": m[r]["bend"]["bend_fraction"],
                "detour_steps": m[r]["bend"]["detour_steps"],
                "opt_mean": m[r]["bend"]["opt_mean"]})
    return out


def blind_table(runs):
    out = {}
    for name, d in sorted(runs.items()):
        out[name] = {"SLACK10_all_pairs": {
            "n": d["all_pairs_slack10"]["n"],
            "ORACLE": d["all_pairs_slack10"]["blind"][ORACLE]["validity"],
            **{_short(k): d["all_pairs_slack10"]["blind"][k]["validity"] for k in BLIND},
            "rule_b_bend_fraction": d["all_pairs_slack10"]["bend"]["bend_fraction"],
            "detour_steps": d["all_pairs_slack10"]["bend"]["detour_steps"],
            "opt_mean": d["all_pairs_slack10"]["bend"]["opt_mean"]}}
        for key, m in d["measurements"].items():
            out[name][key] = {"n_train_requests": m["n_train_requests"],
                              "budget_words": m["budgets"], "interp_words": m["interp"]}
            out[name][key].update({r: _row_block(m, r) for r in ROWS})
    return out


def frag_table(runs):
    out = {}
    for name, d in sorted(runs.items()):
        out[name] = {}
        for key, m in d["measurements"].items():
            sm, cm = m.get("fragment_same_maze", {}), m.get("fragment_cross_maze", {})
            if "DECLARED_NOT_RUN" in sm or "DECLARED_NOT_RUN" in cm:
                out[name][key] = {"same_maze": sm, "cross_maze": cm}
                continue
            out[name][key] = {
                "n_eval_rows": m["n_fragment_eval_rows"],
                "CROSS_MAZE__the_published_quantity": {
                    "r1": cm["FRAG-cross-r1"]["validity"],
                    "r2": cm["FRAG-cross-r2"]["validity"],
                    "pool_routes": cm["FRAG-cross-r1"]["routes"],
                    "signatures_r1": cm["FRAG-cross-r1"]["signatures"],
                    "signatures_r2": cm["FRAG-cross-r2"]["signatures"],
                    "published_CMAZE1_reference": list(PUBLISHED_FRAG_CROSS)},
                "SAME_MAZE__a_DIFFERENT_quantity": {
                    "r1": sm["FRAG-r1"]["validity"], "r2": sm["FRAG-r2"]["validity"],
                    "state_coverage_r1": sm["FRAG-r1"]["state_coverage"],
                    "state_coverage_r2": sm["FRAG-r2"]["state_coverage"],
                    "signatures_r1": sm["FRAG-r1"]["signatures"],
                    "signatures_r2": sm["FRAG-r2"]["signatures"],
                    "w9_CMAZE2_reference": [0.0741, 0.5105],
                    "note": ("the table contains routes through the very maze being "
                             "evaluated; NOT comparable with the cross-maze 0.0006/0.0000")}}
    return out


def build(out_dir):
    anchor, runs = collect(out_dir)
    before = {k: v for k, v in runs.items() if v["which"] == "before"}
    after = {k: v for k, v in runs.items() if v["which"] == "after"}
    assert after, "no AFTER measurement found"

    def merged_hist(rs):
        h = {}
        for d in rs.values():
            for k, v in d["key_count_histogram"].items():
                h[k] = h.get(k, 0) + v
        return h

    hb, ha = _hist_summary(merged_hist(before)), _hist_summary(merged_hist(after))

    oracle_vals, held_seq = [], {}
    for name, d in after.items():
        oracle_vals.append(d["all_pairs_slack10"]["blind"][ORACLE]["validity"])
        for key, m in d["measurements"].items():
            for r in ROWS:
                if "blind" in m.get(r, {}):
                    oracle_vals.append(m[r]["blind"][ORACLE]["validity"])
                    if r in HELD:
                        held_seq.setdefault(key, []).append(
                            m[r]["blind"][SEQ]["validity"])
    held_seq["SLACK10_all_pairs"] = [d["all_pairs_slack10"]["blind"][SEQ]["validity"]
                                     for d in after.values()]
    worst = {k: max(v) for k, v in sorted(held_seq.items())}

    c1 = ha["fraction_multi_key"] is not None and ha["fraction_multi_key"] >= 0.90
    c2 = bool(worst) and all(v <= PUBLISHED_BLIND_SEQUENTIAL for v in worst.values())
    c3 = bool(oracle_vals) and all(v == 1.0 for v in oracle_vals)
    c4 = all(d["tolerance_used"] is None for d in runs.values())
    reb = sorted(after.values(), key=lambda d: d["seed_index"])[0].get("rebase", {})

    res = {
        "artefact": "REQUEST-LAYER-READY.json",
        "spoke": "w14/request-layer-fix",
        "stamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ready" if (c1 and c2 and c3 and c4) else "NOT-READY",
        "tolerance_used": None,
        "scored_runs_spent": 0,
        "one_line": ("CMAZE-2b -- the CMAZE-2 request layer with ONE acceptance rule "
                     "added: a pair is kept iff its RULES A+B optimal route picks up "
                     ">= 2 keys.  Engine, solver and exact integer judge are "
                     "w8_envscout.maze_colour, imported unmodified."),

        "provenance": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "base": "6699665 [hub] base/iter8: fold w13/encoder-adapt",
            "worktree": os.getcwd().replace("\\", "/"),
            "request_layer_before": "w9_maze.task.build_site",
            "request_layer_after": "w14_request_layer.task2.build_site_v2",
            "generator": ("w8_envscout.maze_colour.generate at ITS OWN DEFAULTS "
                          "(h=w=7, n_colours=4, braid=0.40, p_tint=0.22, adversarial=2, "
                          "no acceptance filter)"),
            "judge": sorted(after.values(), key=lambda d: d["seed_index"])[0]["judge"],
            "banks": {k: {"path": v["bank"].replace("\\", "/"),
                          "seed0": v["bank_seed0"],
                          "n_sites": len(v["key_count_histogram"]) and
                                     v["measurements"][list(v["measurements"])[0]]["n_mazes"],
                          "n_routes": v["n_routes_in_bank"],
                          "generator_seeds_tried": v["bank_n_tried"],
                          "acceptance": v["bank_acceptance"],
                          "cross_maze_pool_bank": v["pool_bank"].replace("\\", "/")}
                      for k, v in sorted(runs.items())}},

        "ACCEPTANCE": {
            "1__a_large_majority_of_routes_require_2_or_more_keys": {
                "pass": bool(c1),
                "fraction_multi_key_AFTER": ha["fraction_multi_key"],
                "fraction_multi_key_BEFORE": hb["fraction_multi_key"],
                "threshold_used": 0.90},
            "2__blind_sequential_falls_back_toward_the_published_0.4980": {
                "pass": bool(c2),
                "worst_held_out_row_per_budget_condition": worst,
                "published_CMAZE1": PUBLISHED_BLIND_SEQUENTIAL,
                "w9_CMAZE2_scored_seeds": W9_CMAZE2_BLIND_SEQUENTIAL,
                "note": ("it lands BELOW 0.4980, not at it, and that is expected: 0.4980 "
                         "is a MIXTURE over CMAZE-1's own key counts (73.6 % one-key at "
                         "0.6532, the rest at 0.1351/0.0000).  CMAZE-2b is 100 % "
                         "multi-key, so it sits at CMAZE-1's multi-key BAND, not at "
                         "CMAZE-1's marginal.")},
            "3__ORACLE_stays_1.0000": {"pass": bool(c3),
                                       "distinct_values_seen": sorted(set(oracle_vals)),
                                       "n_row_x_condition_checked": len(oracle_vals)},
            "4__the_exact_integer_judge_does_not_move": {
                "pass": bool(c4), "tolerance_used": None,
                "note": ("w8_envscout.maze_colour.judge is imported, not wrapped and not "
                         "reimplemented; its file sha256 is in provenance.judge")}},

        "KEY_COUNT_HISTOGRAM": {
            "AFTER__w14_request_layer": ha,
            "BEFORE__w9_request_layer": hb,
            "w9_published_figure": {
                "histogram": {"1": 1242, "2": 198, "3": 9}, "n_routes": 1449,
                "source": ("/c/chlu/evidence/outputs/_thinkpad/w9/w9-maze/LOG.md "
                           "lines 92 and 413")},
            "CMAZE1_proper_for_reference": {
                "histogram": anchor["key_count_hist"], "n_routes": anchor["n"],
                "note": ("CMAZE-1's OWN routes are 73.6 % single-key, so a single-key "
                         "majority is not by itself what separates 0.4980 from 0.9037; "
                         "the mechanism is in artefacts/anchor_breakdown.txt")}},

        "BLIND_COLUMNS": blind_table(runs),
        "FRAGMENT_LOOKUP": frag_table(runs),
        "BUDGET_GRID": {
            "primary": "GRID_W9 -- w9's unchanged absolute words 30/45/60, interp 38/53",
            "rebased": reb,
            "rule": ("the rebased words sit at the same percentiles of the AFTER bank's "
                     "route-length distribution that w9's words occupy in the BEFORE "
                     "bank's, rounded to the nearest 5; the interpolation words are the "
                     "midpoints.  The rule mentions route lengths and nothing else."),
            "self_check": ("applying the rule with BEFORE as both source and target "
                           "returns w9's own grid [30, 45, 60]; the interpolation words "
                           "come back [38, 52] against w9's [38, 53] because Python "
                           "rounds 52.5 to even.  Declared, not silently fixed."),
            "writable_fraction": {k: v["budget_fit"] for k, v in sorted(runs.items())}},

        "ANCHOR_CMAZE1_proper": {
            "n": anchor["n"], "seeds": "0..499, zero rejections",
            "bend_fraction": anchor["bend_fraction"],
            "key_count_hist": anchor["key_count_hist"],
            "blind": {_short(k): v["validity"] for k, v in anchor["blind"].items()},
            "published_reference": {"blind_sequential": 0.4980,
                                    "w9_harness_remeasurement": 0.5317,
                                    "rule_b_bends": 0.816,
                                    "fragment_lookup_cross_maze": [0.0000, 0.0000]}},

        "MECHANISM": {
            "source": "artefacts/anchor_breakdown.txt (CMAZE-1 proper, n=400)",
            "blind_sequential_by_key_count_of_the_A_plus_B_optimum": {
                "1": [194, 297, 0.6532], "2": [10, 74, 0.1351],
                "3": [0, 26, 0.0000], "4": [0, 3, 0.0000]},
            "blind_sequential_when_ruleA_key_order_EQUALS_A_plus_B": [201, 201, 1.0000],
            "blind_sequential_when_it_DIFFERS": [3, 199, 0.0151],
            "why_one_key_is_fatal": (
                "with ONE key the constraint key_order=(c,) admits exactly one pickup of "
                "exactly colour c -- which is what the oracle does too -- so "
                "BLIND-SEQUENTIAL *is* the oracle on a one-key route"),
            "root_cause_of_the_w9_defect": (
                "the >= 2-key property is all-or-nothing per maze and DIRECTIONAL: where "
                "it holds, 18/18 cross pairs need >= 2 keys and the REVERSE direction is "
                "0/18 FEASIBLE, because the cut door's key lives on one side of the cut. "
                "w9's layer required cross pairs in BOTH directions, so its own symmetry "
                "requirement selected for single-key cuts.  See artefacts/"
                "probe_direction.txt (13 of 40 mazes) and artefacts/probe_density.txt.")},

        "REFUSAL_MATERIAL_AND_CONTROLS": (
            json.load(open(os.path.join(out_dir, "artefacts",
                                        "refusal_material.json")))
            if os.path.exists(os.path.join(out_dir, "artefacts",
                                           "refusal_material.json"))
            else {"DECLARED_NOT_RUN": "refusal_material.json absent"}),

        "DECLARED_DIFFERENCES_FROM_CMAZE2": [
            ("The kept set is no longer symmetric in direction: all six agent words sit "
             "in the source door-free component and all six target words in the "
             "destination one.  Direction is a CONSTANT of the split, not a balanced "
             "factor; it cannot confound a held-out comparison because it does not vary, "
             "but it is a difference and it is stated."),
            ("The >= 2-key screen is applied to the UNSEALED maze, before the walled "
             "target is chosen, so the refusal walls cannot be what creates the "
             "multi-key structure."),
            ("Generator-seed acceptance falls from 0.2000 (before) to ~0.165 (after): "
             "the cost of the repair is generator seeds, not code."),
            ("Routes are longer -- mean optimum 37.4 -> 65.8 steps -- which is why "
             "GRID_REBASED exists.  At w9's unchanged words the split carries 454 "
             "training requests against w9's 1208; at the rebased words it carries "
             "about 1442."),
            ("The WALLED target -- the second refusal condition -- is harder to place: "
             "all six target words now sit in one component, so sealing one without "
             "breaking the other pairs' >= 2-key routes succeeds on 56 of 96 sites "
             "against 93 of 96 before.  504 impossible requests remain (837 before)."),
        ],

        "DECLARED_NOT_RUN": [
            ("Any SCORED run.  This spoke is 0/0 by its own task file; no arm is fitted, "
             "no store exists, no RESULT row is claimed."),
            ("The learned arms, the CLU, the encoder and the pipeline ceiling on the "
             "fixed split -- that is hub/read-verb's work, and this file is its "
             "precondition, not its result."),
            ("A BEFORE bank at seed indices 1 and 2.  Only `before 0` (96 sites) and a "
             "48-site `before 1` pool were built; the BEFORE column is one bank, which "
             "is enough because it reproduces w9's published numbers exactly."),
            ("Any change to w8_envscout.maze_colour.  The engine, the solver and the "
             "exact integer judge are imported, and the judge's file sha256 is recorded "
             "so that claim is checkable rather than asserted."),
            ("GPU and cluster.  Local CPU only, three concurrent processes."),
        ],
    }
    return res


def main(out_dir):
    res = build(out_dir)
    p = os.path.join(out_dir, "REQUEST-LAYER-READY.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", p, "status =", res["status"])
    for k, v in res["ACCEPTANCE"].items():
        print("  %-62s %s" % (k, "PASS" if v["pass"] else "FAIL"))


if __name__ == "__main__":
    main(sys.argv[1])
