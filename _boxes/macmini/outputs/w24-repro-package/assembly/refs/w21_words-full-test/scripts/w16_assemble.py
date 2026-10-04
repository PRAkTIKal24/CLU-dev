"""Assemble MAZE-WM.json from the w16 artefacts.  Every number is read from a file.

    python scripts/w16_assemble.py <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np

O = sys.argv[1]
J = lambda p: json.load(open(os.path.join(O, p)))
ex = lambda p: os.path.exists(os.path.join(O, p))


def diag_t():
    out = {}
    for s in range(3):
        p = "t1/A1_rivaldiag_val_seed%d.json" % s
        if ex(p):
            for a, e in J(p)["arms"].items():
                out.setdefault(a, {}).setdefault("nway_per_seed", {})[s] = e["rows"]["nway"]
    for a in out:
        v = list(out[a]["nway_per_seed"].values())
        out[a]["nway_mean_over_available_seeds"] = float(np.mean(v))
        out[a]["n_seeds"] = len(v)
    return out


def diag_o():
    out = {}
    for s in range(3):
        p = "t2/C_rivaldiag_val_seed%d.json" % s
        if ex(p):
            for k, e in J(p)["rivals"].items():
                out.setdefault(k, {}).setdefault("nway_per_seed", {})[s] = \
                    e["rows"]["unseen_nway"]["valid"]
    for a in out:
        v = list(out[a]["nway_per_seed"].values())
        out[a]["nway_mean_over_available_seeds"] = float(np.mean(v))
        out[a]["n_seeds"] = len(v)
    return out


def sel(p, keys):
    if not ex(p):
        return None
    d = J(p)
    c = d["CONFIG"]
    return {"K_star": d["K_star"], "config": {k: c.get(k) for k in keys},
            "eligibility": d.get("eligibility")}


def bars(p):
    if not ex(p):
        return None
    d = J(p)
    b = d.get("bars_nway") or d.get("bars_unseen_nway")
    return {"verdict_by_prereg_rule": d["verdict"], "ALL_PASS": d["ALL_PASS"],
            "bars": {k: v["pass"] for k, v in b.items()}, "artefact": p}


doc = {
    "artefact": "MAZE-WM.json", "hub": "w16 hub/maze-wm", "branch": "w16/maze-wm",
    "tolerance_used": None, "n_seeds_every_row": 3,
    "budget": {"total": 18, "spent": [], "declared_not_run": []},
    "configs": [],
    "training_budget_diagnostics_VAL": {"CMAZE-T A1": diag_t(), "CMAZE-O C-field": diag_o()},
}
C = doc["configs"]
C.append({"task": "T1 CMAZE-T", "config": "A (pre-amendment)", "selection": sel("t1/SELECT-none.json", ("latent", "K", "n_atoms", "span", "read_floats")),
          "val": bars("t1/BARS-A-val.json"), "verdict": "SUPERSEDED by AMENDMENT 1 (across-wells control was the identity on seeds 0,1)", "scored_runs": 0})
C.append({"task": "T1 CMAZE-T", "config": "A1", "selection": sel("t1/SELECT-none_am1.json", ("latent", "K", "n_atoms", "span", "read_floats")),
          "val": bars("t1/BARS-A1-val.json"), "test": bars("t1/BARS-A1-test.json"), "scored_runs": 3, "ledger_rows": ["1/18", "2/18", "3/18"],
          "prereg": "PREREG-w16-maze-wm-T1.md (+AM1)"})
C.append({"task": "T1 CMAZE-T", "config": "B (match-bit interface)", "selection": sel("t1/SELECT-match_am1.json", ("latent", "K", "n_atoms", "span", "read_floats")),
          "val": bars("t1/BARS-B-val.json"), "verdict": "FAILS VAL (VB3)", "scored_runs": 0})
C.append({"task": "T1 CMAZE-T", "config": "C (C-code item store + latent-512 re-sweep)", "selection": sel("t1/SELECT-none_am1_C.json", ("latent", "K", "n_atoms", "span", "read_floats")),
          "scored_runs": 0})
C.append({"task": "T2 CMAZE-O", "config": "A", "selection": sel("t2/SELECT-A.json", ("K", "n_atoms", "span", "read_floats")),
          "verdict": "FAILS VAL (VB0 viability)", "scored_runs": 0})
C.append({"task": "T2 CMAZE-O", "config": "C (C-field)", "selection": sel("t2/SELECT-C-field.json", ("K", "n_atoms", "span", "read_floats")),
          "val": bars("t2/BARS-C-val.json"), "test": bars("t2/BARS-C-test.json"), "scored_runs": 3, "ledger_rows": ["4/18", "5/18", "6/18"],
          "prereg": "PREREG-w16-maze-wm-T2.md (+AM1, AM2)"})
C.append({"task": "T2 CMAZE-O", "config": "C2 (latent above 0.30), recipe rivals", "selection": sel("t2/SELECT-C-rank.json", ("K", "n_atoms", "span", "read_floats")),
          "val": bars("t2/BARS-C2-val.json"), "verdict": "not accepted (AMENDMENT 4: rivals under-trained)", "scored_runs": 0})
C.append({"task": "T2 CMAZE-O", "config": "C2r (C2 vs 30k-step rivals)", "val": bars("t2/BARS-C2r-val.json"), "scored_runs": 0})
for extra in glob.glob(os.path.join(O, "EXTRA-*.json")):
    doc.setdefault("extra", []).append(json.load(open(extra)))
json.dump(doc, open(os.path.join(O, "MAZE-WM.json"), "w"), indent=1, default=str)
print("wrote MAZE-WM.json")
