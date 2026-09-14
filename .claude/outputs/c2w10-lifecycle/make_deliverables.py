"""Build C2W10's two deliverables MECHANICALLY from the tests and the run.

    LIFECYCLE-MECHANICS-DONE.json   per-leg L1..L7 booleans + the AND
    USAGE-TELEMETRY.json            the I2 spoke's input

⛔ Nothing here decides anything: every leg boolean is a pytest exit status, and
``lifecycle_mechanics_done`` is the AND over the legs, computed in code.
Run from the worktree root with the MAIN venv (protocol §4).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

WT = "/Users/user/Desktop/CHLU-c2w10"
PY = "/Users/user/Desktop/CHLU/.venv/bin/python"
OUT = "/Users/user/Desktop/CHLU/.claude/outputs/c2w10-lifecycle"
RUN = os.path.join(OUT, "run2", "persistent_store.json")  # d_safe_frac = 0.60

# leg -> (test file, -k selector, the designed negatives with their can-fail twin)
LEGS = {
    "L1": ("tests/test_store_lifecycle.py", "l1_", [
        ("a single burst reaching h_hi does NOT promote",
         "test_l1_single_burst_does_not_promote",
         "test_l1_single_burst_negative_can_fail",
         "d_dwell := window (the hysteresis stops binding)"),
        ("a well below h_lo never promotes",
         "test_l1_below_h_lo_never_promotes",
         "test_l1_below_h_lo_negative_can_fail",
         "h_hi := 0 (the threshold stops binding)"),
    ]),
    "L2": ("tests/test_store_lifecycle.py", "l2_", [
        ("an early-popular-then-abandoned well MUST demote, and must NOT be trashed",
         "test_l2_early_popular_then_abandoned_well_demotes",
         "test_l2_demotion_negative_can_fail",
         "demote := False (the verb is switched off)"),
    ]),
    "L3": ("tests/test_store_lifecycle.py", "l3_", [
        ("(a) useful in stream 1 only => trashed at k",
         "test_l3_useful_in_stream_one_only_is_trashed_at_k",
         "test_l3_the_two_readings_of_the_criterion_differ_and_both_ship",
         "trash_criterion := since_first_seen (the literal reading does not fire)"),
        ("(b) useful in EVERY stream => never trashed",
         "test_l3_useful_in_every_stream_is_never_trashed",
         "test_l3_useful_in_every_stream_negative_can_fail",
         "the discriminating input flipped: the same item with its hits zeroed IS trashed"),
        ("(c) the censoring guard: a well admitted in the last stream is never trashed",
         "test_l3_censoring_guard_spares_a_young_well",
         "test_l3_censoring_guard_can_fail",
         "censoring_guard := False (the young well IS trashed)"),
    ]),
    "L4": ("tests/test_store_lifecycle.py", "l4_", [
        ("forcing every item's usage high trips the monitor and REFUSES",
         "test_l4_forcing_every_item_high_trips_the_monitor_and_refuses",
         "test_l4_negative_can_fail",
         "f_max := 1.0 (the bound stops binding; everything protects, no trip)"),
    ]),
    "L5": ("tests/test_store_lifecycle.py", "l5_ or l5b_", [
        ("with the guard OFF a planted destructive rewrite reduces the depth",
         "test_l5_guard_off_a_destructive_rewrite_reduces_depth",
         "test_l5_guard_off_a_destructive_rewrite_reduces_depth",
         "refresh_monotonic := False vs True on the same event (both asserted)"),
        ("L5-b: the store-level guard matches blocks.py on C2W6's OWN events",
         "test_l5b_cross_implementation_against_c2w6_recorded_events",
         "test_l5b_cross_implementation_against_c2w6_recorded_events",
         "guard OFF on the same recorded events reproduces 16/2/3 and 6/0/0 "
         "post-guard violations"),
    ]),
    "L6": ("tests/test_store_lifecycle.py", "l6_", [
        ("netted == raw BITWISE at leak = 0", "test_l6_netted_is_bitwise_raw_at_leak_zero",
         "test_l6_netted_exceeds_raw_under_decay",
         "leak > 0 => netted > raw strictly (the same function, both branches)"),
        ("a well with no writes nets to the analytic exp(-leak dt)",
         "test_l6_a_well_with_no_writes_nets_to_the_analytic_law_x64",
         "test_l6_a_well_with_no_writes_nets_to_the_analytic_law_float32_floor",
         "the shipped float32 store misses 1e-9 by ~5e-8; both bounds asserted"),
    ]),
    "L7": ("tests/test_persistent_store.py", "l7_", [
        ("OFF is bit-identical AND parameter-count-identical",
         "test_l7_off_is_bit_identical_and_parameter_count_identical",
         "test_l7_the_store_config_only_attaches_the_trash_region_when_the_verb_is_on",
         "the trash region is attached only when the verb is ON; OFF attaches no "
         "field at all (an empty field is not bit-identical)"),
    ]),
}

EXTRA = {
    "gamma_phi_off_regressions": ("tests/test_well_lifecycle.py", "gamma_phi"),
    "stream_sources": ("tests/test_stream_sources.py", ""),
    "rig": ("tests/test_persistent_store.py", ""),
}


def run_pytest(path: str, selector: str):
    cmd = [PY, "-m", "pytest", path, "-q", "--no-cov", "-p", "no:cacheprovider"]
    if selector:
        cmd += ["-k", selector]
    env = dict(os.environ, PYTHONPATH=WT)
    r = subprocess.run(cmd, cwd=WT, capture_output=True, text=True, env=env)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
    return {"cmd": " ".join(cmd[3:]), "returncode": r.returncode,
            "summary": (tail[0] if tail else ""), "passed": r.returncode == 0}


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=WT,
                            capture_output=True, text=True).stdout.strip()
    legs = {}
    for leg, (path, sel, negs) in LEGS.items():
        res = run_pytest(path, sel)
        legs[leg] = {
            "landed": bool(res["passed"]),
            "reason": ("" if res["passed"] else
                       f"pytest FAILED: {res['summary']}"),
            "pytest": res,
            "designed_negatives": [
                {"statement": n[0], "test": n[1], "can_fail_test": n[2],
                 "mutation_that_makes_it_fail": n[3], "result": "pass" if res["passed"]
                 else "see pytest summary"}
                for n in negs],
        }
    extra = {k: run_pytest(v[0], v[1]) for k, v in EXTRA.items()}

    run = json.load(open(RUN)) if os.path.exists(RUN) else None
    exercised = {}
    if run:
        ev = [c["controller_events"] for c in run["cells"]]
        exercised = {
            "L1_promote": int(sum(e["promote"] for e in ev)),
            "L2_demote": int(sum(e["demote"] for e in ev)),
            "L3_trash": int(sum(e["trash"] for e in ev)),
            "L4_promote_refused": int(sum(e["promote_refused"] for e in ev)),
        }
        for leg, key in (("L1", "L1_promote"), ("L2", "L2_demote"),
                         ("L3", "L3_trash"), ("L4", "L4_promote_refused")):
            legs[leg]["exercised_on_stream"] = bool(exercised[key] > 0)
            legs[leg]["n_events_on_stream"] = exercised[key]
            if exercised[key] == 0:
                legs[leg]["k_c_verdict"] = (
                    "UNEXERCISED at the measured operating point — the verb's "
                    "designed negatives are green on the planted population, and "
                    "its target population on the stream was empty. Not working, "
                    "not broken (K-C, registered in advance).")

    done = all(v["landed"] for v in legs.values())
    out = {
        "wave": "C2W10", "spoke": "lifecycle-mechanics",
        "label": ("MECHANICS — no VALUE cell, no performance claim, no verdict "
                  "(§A33.1). Any launder margin would be a DIAGNOSTIC column."),
        "commit": commit, "base": "9e0bb25",
        "branch": "agent/experiment-engineer/c2w10-lifecycle-mechanics",
        "legs": legs,
        "lifecycle_mechanics_done": bool(done),
        "how_computed": ("the AND over legs[L1..L7]['landed'], each of which is a "
                         "pytest exit status; nothing here is a judgement call"),
        "supporting_suites": extra,
        "exercised_on_stream": exercised,
        "declared_not_runs": [
            "the real INSECTS stream: BENCHMARK-GATE.json does not exist, so the "
            "real-stream legs are a DECLARED NOT-RUN, never a null. The loader, its "
            "sha256 reproduction gate, decimation and the structure-preservation "
            "assertion all ship and are pytest-pinned on the synthetic.",
            "d = 16 (measured inert at 131 072 atoms)",
            "merge verbs / K9 re-registration; prune-below-budget by depth",
            "the anytime / compute-adaptive curve (C2W9's)",
            "any VALUE cell, tier-ii verdict, full-CLU verdict or CSF3 claim",
        ],
        "venue_note": ("⛔ the synthetic regime-switcher is a regression / mechanics "
                       "instrument and NEVER a claim venue (§A14.8)"),
    }
    if run:
        out["run"] = {
            "pricing": run["pricing"],
            "off_identity": run["off_identity"],
            "real_stream": run["real_stream"],
            "n_live_max": run["n_live_max"],
            "seeds": run["seeds"],
            "wall_s": run["wall_s"],
            "byte_ledger_by_seed": [c["bytes"] for c in run["cells"]],
            "monitor_trips_by_seed": [
                {k: v for k, v in c["monitor_trips"].items() if v.get("tripped")}
                for c in run["cells"]],
            "protected_saturation_by_seed": [c["lifecycle"]["monitor"]
                                             for c in run["cells"]],
            "read_coverage_by_seed": [c["read_coverage"] for c in run["cells"]],
            "geometry_by_seed": [c["geometry"] for c in run["cells"]],
            "flags": run["cells"][0]["flags"],
            "drift_free_control": (
                {k: run["controls"]["drift_free"][k] for k in
                 ("controller_events", "lifecycle", "n_live_max", "read_coverage")}
                if run["controls"].get("drift_free") else None),
        }
    with open(os.path.join(OUT, "LIFECYCLE-MECHANICS-DONE.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("lifecycle_mechanics_done =", done)
    for k, v in legs.items():
        print(f"  {k}: landed={v['landed']} exercised_on_stream="
              f"{v.get('exercised_on_stream')} {v['pytest']['summary']}")

    # ---------------- USAGE-TELEMETRY.json ----------------
    if not run:
        print("no run artifact; USAGE-TELEMETRY.json not written")
        return 1
    cells = run["cells"]
    n_live_max = int(max(c["n_live_max"] for c in cells))
    tele = {
        "wave": "C2W10", "produced_by": "exp_persistent_store (MECHANICS)",
        "commit": commit,
        "proxy": "read_hits (item-id keyed; depth NEVER enters U — §A28.3(ii))",
        "n_seeds": len(cells),
        "seeds": [c["seed"] for c in cells],
        "n_live_max": n_live_max,
        "n_live_max_ge_64": bool(n_live_max >= 64),
        "n_live_max_per_seed": {int(c["seed"]): int(c["n_live_max"]) for c in cells},
        "n_seeds_meeting_64": int(sum(c["n_live_max"] >= 64 for c in cells)),
        "i2_power_note": (
            ("⚠ n_live_max >= 64 is met on %d of %d seeds "
             "(%s). PREREG-C2W10 §5's I2-a asks for n_live >= 64 PER SEED on >= 3 "
             "seeds: on this run that is NOT met on every seed, and the shortfall "
             "is an ADMISSION-GEOMETRY fact, not a null — a seed whose distinct "
             "anchors happen to pack tighter than d_safe = %.2f x the measured "
             "item spacing has fewer admissible addresses, so the store cannot "
             "fill its budget. The per-seed numbers are above; the I2 spoke "
             "decides whether to run on the qualifying seeds or to ask for a "
             "re-priced operating point.")
            % (int(sum(c["n_live_max"] >= 64 for c in cells)), len(cells),
               ", ".join(f"seed {c['seed']}: {c['n_live_max']}" for c in cells),
               float(cells[0]["flags"]["clu_system"].get("d_safe_override", 0.0)
                     and cells[0]["flags"].get("d_safe_frac", 0.88) or 0.88))
            if not all(c["n_live_max"] >= 64 for c in cells) else
            "n_live >= 64 on every seed: PREREG-C2W10 §5's I2-a power "
            "precondition is met at this operating point."),
        "per_seed": [{
            "seed": c["seed"],
            "n_live_at_each_measurement_point": c["n_live_points"],
            "n_live_end": c["n_live_end"],
            "n_live_max": c["n_live_max"],
            "hits_by_stream": c["cross_stream"]["hits_by_stream"],
            "first_seen_stream": c["cross_stream"]["first_seen_stream"],
            "depth_curves_raw_and_netted": c["depth_curves"],
            "read_coverage": c["read_coverage"],
            "usage_summary": c["usage"],
            "lifecycle_states": c["lifecycle"]["state_ids"],
            "controller_events": c["controller_events"],
            "bytes": c["bytes"],
        } for c in cells],
        "netting": ("every depth curve carries depth_raw AND depth_netted with the "
                    "cumulative designed-decay factor it was netted by; the netting "
                    "replays chlu.core.well_lifecycle.designed_decay_factors "
                    "(imported read-only)"),
        "caveats": [
            "⛔ depth is NOT feature importance (§A23.5 is ACTIVE and only the "
            "Advisor may amend it)",
            "⛔ the synthetic regime-switcher is a mechanics instrument, never a "
            "claim venue (§A14.8)",
            "⚠ read_coverage is the LAUNCH-POINT coverage rate (min_j |q0 - c_j| "
            "<= 1/2 min-sep). It is the ceiling on the usage proxy's resolution "
            "and must be quoted beside any statement about read_hits.",
        ],
        "flags": cells[0]["flags"],
    }
    with open(os.path.join(OUT, "USAGE-TELEMETRY.json"), "w") as f:
        json.dump(tele, f, indent=2)
    print(f"USAGE-TELEMETRY.json: n_seeds={len(cells)} n_live_max={n_live_max} "
          f"(>=64: {n_live_max >= 64})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
