"""Apply PREREG-T1's crossover rule to the 3-seed VAL sweep.  Writes SELECT-<iface>.json.

    python scripts/w16_select_t.py <t1_dir> <iface>
"""
import json
import os
import sys

import numpy as np

d, iface = sys.argv[1], sys.argv[2]
S = [json.load(open(os.path.join(d, "sweep_%s_seed%d.json" % (iface, s)))) for s in range(3)]
key = lambda r: (r["latent"], r["K"], r["n_atoms"])
rows = {}
for s, doc in enumerate(S):
    for r in doc["grid"]:
        if "CLU" in r:
            rows.setdefault(key(r), []).append(r)
table = []
for k, rs in sorted(rows.items(), key=lambda kv: str(kv[0])):
    if len(rs) != 3:
        continue
    m = lambda f: float(np.mean([f(r) for r in rs]))
    e = {"latent": k[0], "K": k[1], "n_atoms": k[2], "span": rs[0]["span"],
         "read_floats": m(lambda r: r["read_floats"])}
    for row in ("seen", "pair", "nway", "value"):
        e["CLU_" + row] = m(lambda r: r["CLU"][row])
        e["CLU_" + row + "_per_seed"] = [r["CLU"][row] for r in rs]
        e["worst_ctl_" + row] = max(m(lambda r, c=c: r["controls"][c][row])
                                    for c in rs[0]["controls"])
        e["blind_seq_" + row] = m(lambda r: r["blind_seq"][row])
        e["blind_colour_" + row] = m(lambda r: r["blind_colour"][row])
    e["controls_nway"] = {c: m(lambda r, c=c: r["controls"][c]["nway"])
                          for c in rs[0]["controls"]}
    e["eligible_store_gate_nway"] = all(v <= e["CLU_nway"] - 0.15
                                        for v in e["controls_nway"].values())
    table.append(e)
elig = [e for e in table if e["eligible_store_gate_nway"]]
label = "STORE-gate eligible"
if not elig:
    elig, label = table, "NO config eligible -> all eligible, row labelled INTERFACE"
bestK = {}
for e in elig:
    if e["K"] not in bestK or e["CLU_nway"] > bestK[e["K"]]["CLU_nway"]:
        bestK[e["K"]] = e
mx = max(v["CLU_nway"] for v in bestK.values())
Kstar = min(K for K, v in bestK.items() if v["CLU_nway"] >= mx - 0.02)
cands = [e for e in elig if e["K"] == Kstar]
top = max(e["CLU_nway"] for e in cands)
cfg = min([e for e in cands if e["CLU_nway"] >= top - 1e-12], key=lambda e: e["read_floats"])
out = {"iface": iface, "n_seeds": 3, "eligibility": label, "n_configs": len(table),
       "n_eligible": len(elig),
       "best_per_K": {K: {k: v[k] for k in ("latent", "n_atoms", "CLU_nway", "span")}
                      for K, v in sorted(bestK.items())},
       "K_star": Kstar, "CONFIG": cfg, "table": table}
json.dump(out, open(os.path.join(d, "SELECT-%s.json" % iface), "w"), indent=1)
print("eligible %d/%d (%s)" % (len(elig), len(table), label))
for K, v in sorted(bestK.items()):
    print("  K %2d best: lat %d atoms %d nway %.4f" % (K, v["latent"], v["n_atoms"], v["CLU_nway"]))
print("K* =", Kstar, "CONFIG:", {k: cfg[k] for k in ("latent", "K", "n_atoms", "span", "read_floats")})
print("  CLU nway %.4f per seed %s ; ctl %s ; bseq %.4f bcol %.4f ; seen %.4f pair %.4f value %.4f"
      % (cfg["CLU_nway"], cfg["CLU_nway_per_seed"], {k: round(v, 4) for k, v in cfg["controls_nway"].items()},
         cfg["blind_seq_nway"], cfg["blind_colour_nway"], cfg["CLU_seen"], cfg["CLU_pair"], cfg["CLU_value"]))
