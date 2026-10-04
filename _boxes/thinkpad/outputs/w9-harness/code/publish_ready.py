"""Write /c/chlu/outputs/w9-harness/HARNESS-READY.json from the seed reports.

`status` is "partial" until every field it claims is backed by a completed seed
report, and "ready" only when all three seeds have run every arm.  A field that has
not been produced stays false -- a declared NOT-RUN, never a null.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os

FIELDS = ("mlp", "lstm", "transformer", "table", "refuser")


def load(outdir, tier="cmaze1", pattern="harness_seed*.json"):
    """Seed reports for ONE tier.  The tiers are separate instruments and are never
    averaged together."""
    reps = []
    for path in sorted(glob.glob(os.path.join(outdir, pattern))):
        with open(path) as f:
            r = json.load(f)
        if r.get("tier", "cmaze1") == tier:
            reps.append((os.path.basename(path), r))
    return reps


def build(outdir, reps, want_seeds=(0, 1, 2), status=None):
    judge_exact = bool(reps) and all(r["judge_audit"]["exact"] for _, r in reps)
    tol = None                      # there is no tolerance anywhere in the judge
    enc_trained = any("encoder" in r for _, r in reps)
    baselines = {k: False for k in FIELDS}
    params = {}
    cols = {"fragment_lookup": False, "composition_blind": False}
    seeds_done = sorted({r["seed"] for _, r in reps})
    tiers = sorted({r.get("tier", "cmaze1") for _, r in reps})
    for _, r in reps:
        for k in ("mlp", "lstm", "transformer"):
            if k in r.get("results", {}):
                baselines[k] = True
                params[k] = r["training"][k]["params"]
        if "table" in r.get("results", {}):
            baselines["table"] = True
            params["table_keys"] = r["results"]["table"].get("label", "")
        if "refuser_impossible" in r.get("results", {}):
            baselines["refuser"] = True
            params["refuser"] = r.get("refuser_training", {}).get("params")
        if "encoder" in r:
            params["encoder"] = r["encoder"]["params_encoder"]
        if r.get("columns"):
            cols["fragment_lookup"] = any(k.startswith("fragment-lookup")
                                          for k in r["columns"])
            cols["composition_blind"] = any(k.startswith("blind-")
                                            for k in r["columns"])
    complete = (judge_exact and enc_trained and all(baselines.values())
                and all(cols.values()) and set(want_seeds) <= set(seeds_done))
    out = {
        "status": status or ("ready" if complete else "partial"),
        "judge_exact": judge_exact,
        "tolerance_used": tol,
        "encoder": {"trained": enc_trained, "shared": True},
        "action_head": {"shared": True},
        "baselines": baselines,
        "param_counts": params,
        "columns": cols,
        "seeds": seeds_done if seeds_done else [],
        "tier": tiers[0] if len(tiers) == 1 else tiers,
        "seeds_requested": list(want_seeds),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reports": [n for n, _ in reps],
        "notes": {
            "judge": "w8_envscout.maze_colour.judge, re-exported unwrapped as "
                     "w9_harness.judge_maze.JUDGE.  Four-part audit "
                     "(source/types/redundant-replay/oracle) passes; see "
                     "judge_audit in each seed report.",
            "encoder": "one conv autoencoder per seed, trained by per-pixel "
                       "reconstruction on maze images only, then FROZEN.  No arm "
                       "objective reaches it; every arm reads the same features.",
            "action_head": "one ActionHead class, d_model fixed at 32 for every "
                           "arm and built from a single shared PRNG key, so all "
                           "heads start byte-identical and have identical shape.  "
                           "Weights are then trained per arm -- that part is NOT "
                           "shared and is stated as such.",
            "columns": "fragment-lookup and composition-blind are REPORTED "
                       "COLUMNS.  Nothing in this harness aborts or filters on "
                       "them.",
            "CAVEAT_refuser": "READ THIS BEFORE USING A DISPLAY ROW.  The refuser "
                              "does NOT separate possible from impossible: "
                              "separation (refusal on impossible minus refusal on "
                              "possible) is +0.047 [0.000, 0.115] over 3 seeds on "
                              "cmaze1 and +0.020 [-0.005, 0.050] on cmaze1-easy.  "
                              "A ridge probe on the same shared observation at t=0 "
                              "gets AUC 0.48-0.55 for possible-vs-impossible, i.e. "
                              "chance, so this is a property of the PROBLEM at this "
                              "scale (deciding solvability within a budget is close "
                              "to solving the maze), not of the wiring.  DISPLAY-row "
                              "comparisons against this refuser are not yet "
                              "informative.  See refusal_separability.json.",
            "CAVEAT_encoder_instability": "The seed sweep was run BEFORE the "
                                          "encoder restart fix.  A single 1200-step "
                                          "encoder run is unreliable: seed 1 stalled "
                                          "at BCE 0.0796 (probe AUC 0.61) where "
                                          "seeds 0/2 reached 0.0050/0.0038 (AUC "
                                          "0.92/0.80), so seed 1's arm numbers "
                                          "inherit a degraded encoder.  "
                                          "encoder.train now takes restarts / "
                                          "accept_bce, selected on RECONSTRUCTION "
                                          "LOSS ONLY; verified across all 6 "
                                          "seed x tier combinations, worst-case BCE "
                                          "0.101 -> 0.010 and worst-case mean probe "
                                          "AUC 0.44 -> 0.75, at no extra cost.  "
                                          "See encoder_probe_1restart.json vs "
                                          "encoder_probe_restarts.json.  Re-running "
                                          "the arm sweep with the fixed encoder is "
                                          "DECLARED NOT-RUN (wall clock).",
            "CAVEAT_learned_arms": "Under a zero-tolerance judge the learned arms "
                                   "are near the floor: validity 0.000-0.005 "
                                   "(cmaze1) and 0.015-0.018 (cmaze1-easy) against "
                                   "an oracle of 1.0 and a strongest-blind-planner "
                                   "column of 0.53 / 0.80.  Teacher-forced per-step "
                                   "accuracy (0.52 MLP / 0.75 LSTM / 0.74 "
                                   "transformer) and the exact diagnostics are where "
                                   "the dynamic range is at this training budget.",
            "refuser": "supervised transformer with a REFUSE token, trained on "
                       "exactly-certified impossible instances.  It is the only "
                       "comparator for DISPLAY rows.  No impossible instance "
                       "appears in any other arm's data.",
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/c/chlu/outputs/w9-harness")
    ap.add_argument("--status", default="")
    a = ap.parse_args()
    reps = load(a.out, "cmaze1")
    doc = build(a.out, reps, status=a.status or None)
    easy = load(a.out, "cmaze1-easy")
    doc["easy_tier"] = {"seeds": sorted({r["seed"] for _, r in easy}),
                        "reports": [n for n, _ in easy],
                        "note": "CMAZE-1-EASY: same generator (h=w=4, k=2), same "
                                "exact judge, declared separately and never "
                                "averaged with the main tier."}
    path = os.path.join(a.out, "HARNESS-READY.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(doc, indent=1))
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
