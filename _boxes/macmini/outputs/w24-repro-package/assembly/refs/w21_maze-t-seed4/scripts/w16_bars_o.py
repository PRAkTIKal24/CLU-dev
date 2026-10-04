"""PREREG-T2 bars (VB/SB 0-4) from three CMAZE-O `full` JSONs.  Writes BARS-<tag>-<leg>.json.

    python scripts/w16_bars_o.py <t2_dir> <tag> <leg>
"""
import json
import os
import sys

import numpy as np

d, tag, leg = sys.argv[1:4]
S = [json.load(open(os.path.join(d, "%s_%s_seed%d.json" % (tag, leg, s)))) for s in range(3)]
CTL = ("CTRL:landscape_removed", "CTRL:landscape_shuffled_channels",
       "CTRL:landscape_shuffled_across_wells", "CTRL:random_atoms")
ROWS = ("seen", "unseen_2way", "unseen_nway", "value_holdout", "value_slot_holdout")
v = lambda s, row, a: s["ROWS"][row][a]["valid"]
m = lambda row, a: float(np.mean([v(s, row, a) for s in S]))
out = {"tag": tag, "leg": leg, "n_seeds": 3, "config": S[0]["config"], "rows": {}}
for row in ROWS:
    arms = list(S[0]["ROWS"][row])
    out["rows"][row] = {"n_per_seed": [s["ROWS"][row]["CLU"]["n"] for s in S],
                        **{a: {"mean": m(row, a), "per_seed": [v(s, row, a) for s in S],
                               "sd": float(np.std([v(s, row, a) for s in S], ddof=1))}
                           for a in arms}}
    clu = m(row, "CLU")
    out["rows"][row]["claim_label"] = (
        "STORE (controls collapse <= 0.5 x CLU)" if clu > 0 and all(m(row, c) <= 0.5 * clu for c in CTL)
        else "INTERFACE or NOTHING")
nw = "unseen_nway"
clu = m(nw, "CLU")
riv = [a for a in S[0]["ROWS"][nw] if a.startswith("RIVAL:")]
blind = [a for a in S[0]["ROWS"][nw] if a.startswith("BLIND")]
br = max(riv, key=lambda a: m(nw, a))
every = all(v(s, nw, "CLU") > max(v(s, nw, a) for a in riv) for s in S)
bars = {
    "0_viability": {"CLU": clu, "pass": bool(clu >= 0.05)},
    "1_composition": {"best_rival": br, "best_rival_mean": m(nw, br), "margin": clu - m(nw, br),
                      "every_seed_ahead": every, "pass": bool(clu >= m(nw, br) + 0.03 and every)},
    "2_store_gate": {"controls": {c: m(nw, c) for c in CTL}, "bar": 0.5 * clu,
                     "pass": bool(all(m(nw, c) <= 0.5 * clu for c in CTL))},
    "3_blind": {"blind": {b: m(nw, b) for b in blind}, "bar": clu - 0.03,
                "pass": bool(max(m(nw, b) for b in blind) <= clu - 0.03)},
    "4_floor": {"lookup": m(nw, "LOOKUP (printed floor)"), "bar": clu - 0.03,
                "pass": bool(m(nw, "LOOKUP (printed floor)") <= clu - 0.03)},
}
out["bars_unseen_nway"] = bars
out["ALL_PASS"] = all(b["pass"] for b in bars.values())
out["verdict"] = ("WIN: STORE row on unseen_nway" if out["ALL_PASS"] else
                  "INTERFACE row" if all(bars[k]["pass"] for k in bars if k != "2_store_gate")
                  else "FAIL")
out["ledger"] = [s["clu_ledger"] for s in S]
out["rivals"] = [s["rivals"] for s in S]
out["lookup_ledger"] = [s["lookup_ledger"] for s in S]
json.dump(out, open(os.path.join(d, "BARS-%s-%s.json" % (tag, leg)), "w"), indent=1, default=str)
for k, b in bars.items():
    print(k, "PASS" if b["pass"] else "FAIL",
          {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in b.items() if kk != "pass"})
for row in ROWS:
    r = out["rows"][row]
    print("%-18s n=%s CLU %.4f %s | best rival %.4f | lookup %.4f | ctl %s | %s" % (
        row, r["n_per_seed"], r["CLU"]["mean"], [round(x, 3) for x in r["CLU"]["per_seed"]],
        max(r[a]["mean"] for a in riv), r["LOOKUP (printed floor)"]["mean"],
        [round(r[c]["mean"], 3) for c in CTL], r["claim_label"]))
print("VERDICT:", out["verdict"])
