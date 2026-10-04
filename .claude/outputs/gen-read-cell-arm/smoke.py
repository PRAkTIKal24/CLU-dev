"""⚠ PLUMBING SMOKE ONLY — ⛔ no number from this run is ever reported.

Exercises the real cell path (real store, real split, real decoder, real classifier)
with the settling budget cut to a handful of steps, into a THROWAWAY out dir, so a
plumbing defect is found in minutes rather than after the scored run.
"""
import os, sys, time
from pathlib import Path

import chlu.experiments.exp_gen_cell as C
from chlu.config import load_config

C.DYNREAD_STEPS = {"A": 8, "B": 8}
C.DYNREAD_BUDGETS = ("A",)
C.DYNREAD_CONVERGENCE_CHECKPOINTS = (4, 8, 1000)

cfg = load_config(Path(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
C.check_preconditions(cfg)
man = C.run_seed("shapes3d", "recon", 0, out, cfg, arms=["dynread"])
print("=== smoke done in %.0fs ===" % (time.time() - t0))
for a, r in man["arms"].items():
    if a.split("/")[0] != "dynread":
        continue
    d = r["dynread"]
    print(f"{a:26s} B_agg={r['external']['composition_B']['aggregate']:.4f} "
          f"tuples={r['distinct']['B']['factor_tuples']}/{r['distinct']['B']['n']} "
          f"bytes={r['byte_ledger']['deployed_bytes']} "
          f"lam={d['coupling_certificate']['coupling_strength_used']:.4g} "
          f"quotable={d['QUOTABLE']}")
    print("   artefacts:", d["artefacts"]["B"])
