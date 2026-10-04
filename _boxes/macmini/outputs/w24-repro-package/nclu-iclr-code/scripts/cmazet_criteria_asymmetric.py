"""Criteria of the asymmetric-input runs.   python scripts/cmazet_criteria_asymmetric.py <dir> <tag> <leg>"""

import json
import os
import sys
import numpy as np

d, tag, leg = sys.argv[1:4]
S = [json.load(open(os.path.join(d, "%s_%s_seed%d.json" % (tag, leg, s)))) for s in range(3)]
RIV = ("rival: mlp", "rival: lstm", "rival: transformer")
CTL = (
    "control: landscape_removed",
    "control: landscape_shuffled",
    "control: landscape_shuffled_across_wells",
    "control: random_atoms",
)
BL = (
    "arm: BLIND-COLOUR (learned: colour identity removed from the request)",
    "arm: BLIND-SEQUENTIAL (learned: main effects only, no interaction terms)",
)


def m(row, arm):
    return float(np.mean([s["ROWS"][row][arm] for s in S]))


def per(row, arm):
    return [s["ROWS"][row][arm] for s in S]


out = {
    "tag": tag,
    "leg": leg,
    "n_seeds": 3,
    "config": S[0]["config"],
    "iface": S[0]["iface"],
    "rows": {},
}
for row in ("seen", "pair", "nway", "value"):
    arms = [k for k in S[0]["ROWS"][row] if k != "n"]
    out["rows"][row] = {
        "n_per_seed": [s["ROWS"][row]["n"] for s in S],
        **{
            a: {
                "mean": m(row, a),
                "sd": float(np.std(per(row, a), ddof=1)),
                "per_seed": per(row, a),
            }
            for a in arms
        },
    }
    clu = m(row, "CLU")
    gate = all((m(row, c) <= clu - 0.15 for c in CTL))
    out["rows"][row]["claim_label"] = (
        "STORE (controls collapse)"
        if gate
        else "INTERFACE or NOTHING (a control does not collapse)"
    )
nw = "nway"
clu = m(nw, "CLU")
best_riv = max(RIV, key=lambda a: m(nw, a))
every = all((S[i]["ROWS"][nw]["CLU"] > max((S[i]["ROWS"][nw][a] for a in RIV)) for i in range(3)))
bars = {
    "1_composition": {
        "CLU": clu,
        "best_rival": best_riv,
        "best_rival_mean": m(nw, best_riv),
        "margin": clu - m(nw, best_riv),
        "every_seed_ahead": every,
        "pass": bool(clu >= m(nw, best_riv) + 0.1 and every),
    },
    "2_store_gate": {
        "controls": {c: m(nw, c) for c in CTL},
        "bar": clu - 0.15,
        "pass": bool(all((m(nw, c) <= clu - 0.15 for c in CTL))),
    },
    "3_falsifier": {
        "blind": {b: m(nw, b) for b in BL},
        "bar": clu - 0.05,
        "pass": bool(max((m(nw, b) for b in BL)) <= clu - 0.05),
    },
    "4_floor": {
        "lookup_words": m(nw, "floor: lookup cross-maze WORDS"),
        "bar": clu - 0.1,
        "pass": bool(m(nw, "floor: lookup cross-maze WORDS") <= clu - 0.1),
    },
}
out["bars_nway"] = bars
out["ALL_PASS"] = all((b["pass"] for b in bars.values()))
out["verdict"] = (
    "WIN: STORE row on nway"
    if out["ALL_PASS"]
    else "INTERFACE row on nway (store gate fails, others pass)"
    if all((bars[k]["pass"] for k in ("1_composition", "3_falsifier", "4_floor")))
    else "FAIL"
)
out["rollout"] = {
    a: {
        "validity": float(np.mean([s["ROLLOUT"]["arms"][a]["validity"] for s in S])),
        "per_step_exactness": float(
            np.mean([s["ROLLOUT"]["arms"][a]["per_step_exactness"] for s in S])
        ),
    }
    for a in S[0]["ROLLOUT"]["arms"]
}
out["footprint"] = [s["clu_footprint"] for s in S]
out["rank_over_width"] = [s["rank_over_width_printed"] for s in S]
out["matched_parameters"] = [s["matched_parameters"] for s in S]
json.dump(out, open(os.path.join(d, "BARS-%s-%s.json" % (tag, leg)), "w"), indent=1)
for k, b in bars.items():
    print(
        k,
        "PASS" if b["pass"] else "FAIL",
        {kk: round(v, 4) if isinstance(v, float) else v for kk, v in b.items() if kk != "pass"},
    )
for row in ("seen", "pair", "nway", "value"):
    r = out["rows"][row]
    print(
        "%-5s n=%s CLU %.4f %s | best rival %.4f | ctl %s"
        % (
            row,
            r["n_per_seed"],
            r["CLU"]["mean"],
            [round(x, 3) for x in r["CLU"]["per_seed"]],
            max((r[a]["mean"] for a in RIV)),
            [round(r[c]["mean"], 3) for c in CTL],
        )
    )
print("VERDICT:", out["verdict"])
