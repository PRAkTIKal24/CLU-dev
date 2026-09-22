"""Drive the CMAZE-O seeded VAL smoke: 3 seeds, all rows, every arm, then verify.

Usage:  python scripts/w15_cmaze_o_valsmoke.py <out_dir> [--seeds 0,1,2] [--tiny]

Writes, into ``<out_dir>``:
  seed<N>.json          the full per-seed record
  states_seed<N>.npz    every judged state, packed, with the pipeline's verdict bits
  bank_seed<N>.npz      the layout bank that seed used
  VERIFY.json           the independent checker's replay of every judged state
  VAL-SMOKE.json        the assembled 3-seed table
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chlu.exp_w15_cmaze_o_deletion import deletion_study          # noqa: E402
from chlu.exp_w15_cmaze_o_pipeline import CONFIG_A, ROWS, run_seed  # noqa: E402
from chlu.exp_w15_cmaze_o_verify import verify_seed               # noqa: E402


def main(argv):
    out_dir = argv[1]
    seeds = [0, 1, 2]
    tiny = "--tiny" in argv
    for a in argv:
        if a.startswith("--seeds"):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(n_layouts=12, n_train=8000, n_row=400, n_seen=600,
              ae_steps=6000, rival_steps=3000)
    if tiny:
        kw = dict(n_layouts=6, n_train=500, n_row=60, n_seen=60, ae_steps=40,
                  rival_steps=60)

    t0 = time.time()
    per_seed, verify, deletions = {}, {}, {}
    for s in seeds:
        res = run_seed(s, out_dir, **kw)
        per_seed[s] = res
        v = verify_seed(s, out_dir)
        verify[s] = v
        print("[seed %d] independent replay: %d states, %d disagreements, bank "
              "mismatches %s" % (s, v["states_replayed"], v["disagreements"],
                                 v["layout_bank_rebuild_mismatches"]), flush=True)
        # the deletion rows were run inside run_seed on this seed's frozen AE
        deletions[s] = res.get("deletion")
        with open(os.path.join(out_dir, "VERIFY.json"), "w", encoding="utf-8") as f:
            json.dump(verify, f, indent=1, default=str)

    # ------------------------------------------------------------------ assemble
    arms = sorted({a for s in seeds for a in per_seed[s]["results"]})
    table = {}
    for a in arms:
        table[a] = {}
        for row in ROWS:
            vals = {k: [] for k in ("valid", "exact_objects", "exact_layout", "legal")}
            ns = []
            for s in seeds:
                r = per_seed[s]["results"].get(a, {}).get(row)
                if not r or r.get("n", 0) == 0:
                    continue
                ns.append(r["n"])
                for k in vals:
                    vals[k].append(r[k])
            if not ns:
                continue
            table[a][row] = {
                "seeds": len(ns), "n_per_seed": ns,
                **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))
                       if len(v) > 1 else 0.0, "per_seed": [float(x) for x in v]}
                   for k, v in vals.items()},
                "tolerance_used": None,
            }

    smoke = {
        "artefact": "VAL-SMOKE.json",
        "spoke": "w15/cmaze-o-harness",
        "split": "VAL",
        "scored_runs_spent": 0,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "config_A": dict(CONFIG_A),
        "rows": list(ROWS),
        "tolerance_used": None,
        "table": table,
        "per_seed_ledgers": {
            str(s): {"clu": per_seed[s]["clu_ledger"],
                     "lookup": per_seed[s]["lookup_ledger"],
                     "rivals": per_seed[s]["rivals"],
                     "ae": {k: v for k, v in per_seed[s]["ae"].items()
                            if k != "loss_history"},
                     "pool": per_seed[s]["pool_stats"],
                     "impossible_request_check":
                         per_seed[s]["impossible_request_check"],
                     "disjointness_pass": per_seed[s]["disjointness"]["pass"],
                     "nearest_item_audit":
                         per_seed[s]["nearest_item_audit_unseen_nway"]}
            for s in seeds},
        "verify": verify,
        "deletion": deletions,
        "wall_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "VAL-SMOKE.json"), "w", encoding="utf-8") as f:
        json.dump(smoke, f, indent=1, default=str)
    print("VAL-SMOKE.json written; total %.0fs" % smoke["wall_seconds"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
