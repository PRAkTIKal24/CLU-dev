"""PREREG-T2 selection rule over a 3-seed CMAZE-O VAL sweep.  Writes SELECT-<name>.json.

    python scripts/w16_select_o.py <t2_dir> <file_prefix> <space_key> <name>
      e.g.  t2 sweepC field C-field   |   t2 sweep grid A
"""
import json
import os
import sys

import numpy as np

d, pref, space, name = sys.argv[1:5]
S = [json.load(open(os.path.join(d, "%s_seed%d.json" % (pref, s)))) for s in range(3)]
rows = ("seen", "unseen_2way", "unseen_nway")
table = []
for i, g in enumerate(S[0][space]):
    if "CLU" not in g:
        continue
    if space == "grid" and not g.get("pair_block", True):
        continue
    gs = [s[space][i] for s in S]
    assert all((x["K"], x["n_atoms"]) == (g["K"], g["n_atoms"]) for x in gs)
    m = lambda f: float(np.mean([f(x) for x in gs]))
    e = {"K": g["K"], "n_atoms": g["n_atoms"], "span": g["span"],
         "read_floats": g["read_floats"],
         "CLU": {r: m(lambda x, r=r: x["CLU"][r]["valid"]) for r in rows},
         "CLU_nway_per_seed": [x["CLU"]["unseen_nway"]["valid"] for x in gs],
         "controls_nway": {c: m(lambda x, c=c: x["controls"][c]["unseen_nway"]["valid"])
                           for c in g["controls"]}}
    e["eligible"] = bool(all(v <= 0.5 * e["CLU"]["unseen_nway"]
                             for v in e["controls_nway"].values())
                         and e["CLU"]["unseen_nway"] > 0)
    table.append(e)
elig = [e for e in table if e["eligible"]]
label = "STORE-gate eligible"
if not elig:
    elig, label = table, "none eligible -> INTERFACE"
best = {}
for e in elig:
    if e["K"] not in best or e["CLU"]["unseen_nway"] > best[e["K"]]["CLU"]["unseen_nway"]:
        best[e["K"]] = e
mx = max(v["CLU"]["unseen_nway"] for v in best.values())
Ks = min(K for K, v in best.items() if v["CLU"]["unseen_nway"] >= mx - 0.01)
c = [e for e in elig if e["K"] == Ks]
top = max(e["CLU"]["unseen_nway"] for e in c)
cfg = min([e for e in c if e["CLU"]["unseen_nway"] >= top - 1e-12], key=lambda e: e["read_floats"])
json.dump({"name": name, "n_seeds": 3, "eligibility": label, "K_star": Ks, "CONFIG": cfg,
           "best_per_K": {K: v for K, v in sorted(best.items())}, "table": table},
          open(os.path.join(d, "SELECT-%s.json" % name), "w"), indent=1)
for e in table:
    print("K %2d M %2d span %.3f rf %8d | CLU %s | ctl %s %s" % (
        e["K"], e["n_atoms"], e["span"], e["read_floats"],
        " ".join("%.4f" % e["CLU"][r] for r in rows),
        " ".join("%.4f" % v for v in e["controls_nway"].values()), "ELIG" if e["eligible"] else ""))
print(label, "K*", Ks, "CONFIG", {k: cfg[k] for k in ("K", "n_atoms", "span", "read_floats")},
      "nway per seed", [round(x, 4) for x in cfg["CLU_nway_per_seed"]])
