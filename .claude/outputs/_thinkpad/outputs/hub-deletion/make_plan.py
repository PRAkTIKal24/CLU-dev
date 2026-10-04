#!/usr/bin/env python
"""Emit the PREREG-01 §2/§3 plan as JSONL, with each placebo sized to match its
deletion EXACTLY (n rows).  Deterministic; no RNG here beyond build_unit's own.
"""
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(sys.argv[0]).resolve()
sys.path.insert(0, r"C:\Users\PratikJawahar\chlu\CHLU-hub-deletion-2")

from chlu.config import CHLUConfig
from chlu.core.gen_bridge import dataset_module
from chlu.experiments.exp_gen_deletion import factor_tuples, row_predicate
from chlu.experiments.exp_gen_store import select_store_rows

cfg = CHLUConfig()
cfg.gen_store.dataset, cfg.gen_store.pretext = "shapes3d", "recon"
mod = dataset_module("shapes3d")
split, _ = mod.build_split(cfg.gen_data.data_root)
grid = split.grid
names, cards = list(grid.names), [int(c) for c in grid.cardinalities]
rows = select_store_rows(split, cfg, 0)
pool = np.sort(np.asarray(rows["store_train"], np.int64))
tup = factor_tuples(grid, pool)

print(f"pool = {pool.size} rows; factors {list(zip(names, cards))}", file=sys.stderr)
plan, meta = [], []
# --- PREREG-01 §2: the middle value of every factor ------------------------
for f, (nm, c) in enumerate(zip(names, cards)):
    v = c // 2
    n = int(row_predicate(tup, [(f, v)]).sum())
    tag = f"del_f{f}v{v}"
    plan.append({"tag": tag, "spec": {"kind": "delete", "spec": [[f, v]]}})
    plan.append({"tag": f"plc_f{f}v{v}", "spec": {"kind": "placebo", "n": n, "seed": 100 + f}})
    meta.append({"tag": tag, "factor": nm, "f": f, "value": v, "card": c, "n_deleted": n,
                 "frac_of_pool": n / pool.size, "placebo": f"plc_f{f}v{v}"})
    print(f"  f{f} {nm:12s} card={c:2d} v={v:2d} -> {n:6d} rows ({n/pool.size:.3%})", file=sys.stderr)

# --- the conjunction (gs task file item 3), DEV only -----------------------
CONJ = [[2, 5], [4, 2]]
nc = int(row_predicate(tup, [(f, v) for f, v in CONJ]).sum())
plan.append({"tag": "del_conj_oh5_sh2", "spec": {"kind": "delete", "spec": CONJ}})
plan.append({"tag": "plc_conj_oh5_sh2", "spec": {"kind": "placebo", "n": nc, "seed": 200}})
meta.append({"tag": "del_conj_oh5_sh2", "factor": "object_hue AND shape", "f": None,
             "value": "oh=5 & shape=2", "card": None, "n_deleted": nc,
             "frac_of_pool": nc / pool.size, "placebo": "plc_conj_oh5_sh2"})
print(f"  conj oh=5 & shape=2 -> {nc} rows ({nc/pool.size:.3%})", file=sys.stderr)

# --- churn-spread repeats at the LARGEST deletion size ---------------------
nmax = max(m["n_deleted"] for m in meta if m["f"] is not None)
for r in (1, 2, 3):
    plan.append({"tag": f"plc_churn_r{r}", "spec": {"kind": "placebo", "n": nmax, "seed": 300 + r}})
print(f"  churn repeats at n={nmax}: plc_churn_r1..r3", file=sys.stderr)

outdir = Path(sys.argv[1])
with open(outdir / "plan_dev.jsonl", "w") as fh:
    for p in plan:
        fh.write(json.dumps(p) + "\n")
with open(outdir / "plan_meta.json", "w") as fh:
    json.dump({"pool_size": int(pool.size), "factor_names": names, "cardinalities": cards,
               "deletions": meta, "churn_n": int(nmax),
               "churn_tags": [f"plc_churn_r{r}" for r in (1, 2, 3)]},
              fh, indent=2, sort_keys=True)
print(f"wrote {outdir/'plan_dev.jsonl'} ({len(plan)} units)", file=sys.stderr)
