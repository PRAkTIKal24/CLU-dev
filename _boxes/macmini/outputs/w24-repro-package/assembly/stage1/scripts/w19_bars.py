"""w19 bars: VB1–VB4 (VAL) / SB1–SB4 (TEST) per condition, exactly as PREREG-w19.

    python scripts/w19_bars.py <out_dir> <cond> <leg>

R* (VB1) = best MATCHED rival over every kind / input variant of the condition AND over
both budgets (8,000-step row and 80,000-step 10x column).  The 10x-read-floats column is
printed, never a bar.  WORDS-MAIN: VB3 on BLIND-COLOUR only (BLIND-SEQUENTIAL IS the CLU).
"""
import json
import os
import sys
import numpy as np
BSEQ = "arm: BLIND-SEQUENTIAL (learned: main effects only, no interaction terms)"
BCOL = "arm: BLIND-COLOUR (learned: colour identity removed from the request)"
SEEDS = (0, 1, 2)
def load(out, cond, leg):
    return [json.load(open(os.path.join(out, "%s_%s_seed%d.json" % (cond, leg, s))))
            for s in SEEDS]


def row_stats(runs, row):
    B = [r["ROWS"][row] for r in runs]
    keys = [k for k in B[0] if isinstance(B[0][k], (int, float)) and k != "n"
            and not k.endswith(":coverage")]
    per = {k: [float(b[k]) for b in B] for k in keys}
    mean = {k: float(np.mean(v)) for k, v in per.items()}
    return B, per, mean


def verdict_row(runs, row, cond):
    B, per, mean = row_stats(runs, row)
    clu = mean["CLU"]
    matched = [k for k in per if k.startswith("rival: ") or k.startswith("rival@10x-steps: ")]
    best_k = max(matched, key=lambda k: mean[k])
    best_seed = [max(per[k][i] for k in matched) for i in range(len(runs))]
    ctl = {k: mean[k] for k in per if k.startswith("control: ")}
    blind_keys = [BCOL] if cond == "WORDS-MAIN" else [BCOL, BSEQ]
    blind = max(mean[k] for k in blind_keys)
    wfloor = [k for k in per if k.startswith("floor: lookup cross-maze WORDS")][0]
    b1 = clu >= mean[best_k] + 0.10 and all(per["CLU"][i] > best_seed[i]
                                             for i in range(len(runs)))
    b2 = all(v <= clu - 0.15 for v in ctl.values())
    b3 = blind <= clu - 0.05
    b4 = mean[wfloor] <= clu - 0.10
    if b1 and b2 and b3 and b4:
        label = "STORE"
    elif b1 and b3 and b4:
        label = "INTERFACE"
    else:
        label = "LOSS" if not b1 else "NOTHING"
    return {"n_seeds": len(runs), "n_rows_per_seed": [b["n"] for b in B],
            "CLU_mean": clu, "CLU_per_seed": per["CLU"],
            "R_star": best_k, "R_star_mean": mean[best_k],
            "best_matched_rival_per_seed": best_seed,
            "best_matched_8k": max((k for k in matched if k.startswith("rival: ")),
                                   key=lambda k: mean[k]),
            "margin": clu - mean[best_k],
            "controls_mean": ctl, "control_bar": clu - 0.15,
            "blind_max": blind, "blind_arms_in_bar": blind_keys, "blind_bar": clu - 0.05,
            "words_floor": wfloor, "words_floor_mean": mean[wfloor],
            "B1_composition": bool(b1), "B2_store_gate": bool(b2),
            "B3_falsifier": bool(b3), "B4_floor": bool(b4),
            "PASS_all": bool(b1 and b2 and b3 and b4), "label": label,
            "means": mean, "per_seed": per}


def main(out, cond, leg):
    runs = load(out, cond, leg)
    res = {"condition": cond, "leg": leg, "n_seeds": len(runs),
           "prereg": "PREREG-w19-maze-t-symmetric.md",
           "across_wells_derangement_per_seed": [r["across_wells_derangement"] for r in runs],
           "rows": {}}
    for row in ("nway", "seen", "pair", "value"):
        if all("CLU" in r["ROWS"].get(row, {}) for r in runs):
            res["rows"][row] = verdict_row(runs, row, cond)
    p = res["rows"]["nway"]
    pre = "VB" if leg == "val" else "SB"
    res["PRIMARY_nway"] = {pre + "1": p["B1_composition"], pre + "2": p["B2_store_gate"],
                           pre + "3": p["B3_falsifier"], pre + "4": p["B4_floor"],
                           "PASS": p["PASS_all"], "label": p["label"]}
    if cond.startswith("OBS"):
        res["word_extractor_accuracy"] = {row: [r["ROWS"][row].get("word_extractor_accuracy")
                                                for r in runs] for row in res["rows"]}
    res["rollout"] = {k: [r["ROLLOUT"]["arms"][k]["validity"] for r in runs]
                      for k in runs[0]["ROLLOUT"]["arms"]}
    res["per_step_exactness"] = {k: [r["ROLLOUT"]["arms"][k]["per_step_exactness"]
                                     for r in runs] for k in runs[0]["ROLLOUT"]["arms"]}
    fp = os.path.join(out, "BARS-%s-%s.json" % (cond, leg))
    json.dump(res, open(fp, "w"), indent=1)
    print("%s %s nway: CLU %.4f %s | R* %s %.4f (margin %+.4f) | ctl max %.4f (bar %.4f) | "
          "blind %.4f | floor %.4f | %s -> %s"
          % (cond, leg, p["CLU_mean"], [round(x, 4) for x in p["CLU_per_seed"]],
             p["R_star"], p["R_star_mean"], p["margin"], max(p["controls_mean"].values()),
             p["control_bar"], p["blind_max"], p["words_floor_mean"],
             res["PRIMARY_nway"], p["label"]))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
