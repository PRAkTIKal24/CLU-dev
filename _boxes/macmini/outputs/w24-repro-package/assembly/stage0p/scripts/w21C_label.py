"""w21 label: the Head's WIN / COMPARABLE / LOSS rule (PREREG-w21 §4), per condition/leg.

    python scripts/w21_label.py <out_dir> <cond> <leg>      (reads <out>/<legdir>/<cond>_<leg>_seed{0,1,2}.json)

R* = best MATCHED rival over every kind AND both budgets ('rival: ' 8k, 'rival@10x-steps: '
80k), by mean.  best-per-seed = max over the same set on each seed.
WIN        CLU mean > R* mean AND CLU > best-per-seed on >= 2 of 3 seeds AND store gate.
LOSS       CLU mean < R* mean - s.
COMPARABLE otherwise.   s = sample std over seeds of (CLU - best-per-seed).
Store gate = every control mean <= CLU - 0.15.  Printed, never bars: blind (CLU-0.05),
WORDS floor (CLU-0.10), 10x-read-floats diagnostic, '+rel' diagnostic.
"""
import json, os, sys
import numpy as np

SEEDS = (0, 1, 2)
LEGDIR = {"val": "val_full", "test": "test"}


def load(out, cond, leg):
    return [json.load(open(os.path.join(out, LEGDIR[leg], "%s_%s_seed%d.json" % (cond, leg, s))))
            for s in SEEDS]


def label_row(runs, row):
    B = [r["ROWS"][row] for r in runs]
    per = {k: [float(b[k]) for b in B] for k in B[0]
           if isinstance(B[0][k], (int, float)) and k != "n" and not k.endswith(":coverage")}
    mean = {k: float(np.mean(v)) for k, v in per.items()}
    clu = per["CLU"]
    matched = [k for k in per if k.startswith("rival: ") or k.startswith("rival@10x-steps: ")]
    rstar = max(matched, key=lambda k: mean[k])
    best_seed = [max(per[k][i] for k in matched) for i in range(len(runs))]
    marg = [c - b for c, b in zip(clu, best_seed)]
    s = float(np.std(marg, ddof=1))
    ahead = sum(m > 0 for m in marg)
    ctl = {k: mean[k] for k in per if k.startswith("control: ")}
    gate = all(v <= mean["CLU"] - 0.15 for v in ctl.values())
    if mean["CLU"] > mean[rstar] and ahead >= 2 and gate:
        lab = "WIN"
    elif mean["CLU"] < mean[rstar] - s:
        lab = "LOSS"
    else:
        lab = "COMPARABLE"
    blind = {k: mean[k] for k in per if k.startswith("arm: ")}
    wf = [k for k in per if k.startswith("floor: lookup cross-maze WORDS")][0]
    diag10 = {k: mean[k] for k in per if k.startswith("diagnostic@10x")}
    rel = {k: per[k] for k in per if k.startswith("diagnostic+rel")}
    relbest = max(rel, key=lambda k: np.mean(rel[k])) if rel else None
    return {"n_seeds": len(runs), "n_rows_per_seed": [b["n"] for b in B],
            "CLU_mean": mean["CLU"], "CLU_per_seed": clu,
            "R_star": rstar, "R_star_mean": mean[rstar], "R_star_per_seed": per[rstar],
            "best_matched_rival_per_seed": best_seed, "margin_mean_vs_Rstar": mean["CLU"] - mean[rstar],
            "per_seed_margin_vs_best": marg, "seeds_ahead": ahead, "spread_s": s,
            "best_8k": max((k for k in matched if k.startswith("rival: ")), key=lambda k: mean[k]),
            "best_80k": max((k for k in matched if k.startswith("rival@")), key=lambda k: mean[k]),
            "controls_mean": ctl, "store_gate_bar": mean["CLU"] - 0.15, "store_gate": bool(gate),
            "blind_mean": blind, "blind_max": max(blind.values()), "blind_bar": mean["CLU"] - 0.05,
            "words_floor": wf, "words_floor_mean": mean[wf],
            "diag_10x_read_floats_mean": diag10,
            "diag_rel_best": relbest, "diag_rel_best_per_seed": rel.get(relbest),
            "diag_rel_best_mean": float(np.mean(rel[relbest])) if relbest else None,
            "diag_rel_all_mean": {k: float(np.mean(v)) for k, v in rel.items()},
            "label": lab, "means": mean, "per_seed": per}


def main(out, cond, leg):
    runs = load(out, cond, leg)
    res = {"condition": cond, "leg": leg, "n_seeds": len(runs), "prereg": "PREREG-w21-maze-t-clu-improve.md",
           "clu_cfg": runs[0].get("w21_clu_cfg"),
           "across_wells_derangement_per_seed": [r["across_wells_derangement"] for r in runs],
           "clu_read_floats": runs[0]["clu_ledger"]["read_floats"],
           "matched_parameters": runs[0]["matched_parameters"],
           "rel_diagnostic_parameters": runs[0].get("rel_diagnostic_parameters"),
           "rows": {row: label_row(runs, row) for row in ("nway", "seen", "pair", "value")
                    if all("CLU" in r["ROWS"].get(row, {}) for r in runs)},
           "per_step_exactness": {k: [r["ROLLOUT"]["arms"][k]["per_step_exactness"] for r in runs]
                                  for k in runs[0]["ROLLOUT"]["arms"]},
           "rollout": {k: [r["ROLLOUT"]["arms"][k]["validity"] for r in runs]
                       for k in runs[0]["ROLLOUT"]["arms"]}}
    p = res["rows"]["nway"]
    fp = os.path.join(out, LEGDIR[leg], "LABEL-%s-%s.json" % (cond, leg))
    json.dump(res, open(fp, "w"), indent=1)
    print("%s %s nway: CLU %.4f %s | R* %s %.4f %s | best/seed %s | ahead %d/3 s %.4f | ctl max %.4f (bar %.4f) "
          "| blind %.4f | floor %.4f | +rel best %s %s | -> %s"
          % (cond, leg, p["CLU_mean"], [round(x, 4) for x in p["CLU_per_seed"]], p["R_star"], p["R_star_mean"],
             [round(x, 4) for x in p["R_star_per_seed"]], [round(x, 4) for x in p["best_matched_rival_per_seed"]],
             p["seeds_ahead"], p["spread_s"], max(p["controls_mean"].values()), p["store_gate_bar"],
             p["blind_max"], p["words_floor_mean"], p["diag_rel_best"],
             None if p["diag_rel_best_mean"] is None else round(p["diag_rel_best_mean"], 4), p["label"]))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
