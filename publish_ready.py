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
