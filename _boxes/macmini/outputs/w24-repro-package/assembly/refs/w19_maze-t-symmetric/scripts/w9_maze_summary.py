"""The estimand: paired corners across the three scored runs.

PREREG-w9-maze-1 section 3 declares the estimand as the paired difference in
exact-judge VALIDITY between the CLU and each comparator, paired by
(row, seed, rule-B-bend class), with a declared margin of 0.05 absolute and the
direction CLU > comparator.  This script computes exactly that from the scored JSONs
and prints every corner, not a selection.

    python scripts/w9_maze_summary.py scored1_seed0_test scored2_seed1_test ...
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "C:/chlu/outputs/w9-maze/runs"
ROWS = ("seen", "unseen_pair", "nway", "budget_interp")
COMPARATORS = ("mlp", "lstm", "transformer", "lookup_matched", "landscape_shuffled",
               "random_atoms", "landscape_removed", "clu_lowrank", "clu_nopair",
               "lookup_full")
MARGIN = 0.05


def load(tags):
    out = []
    for t in tags:
        with open(os.path.join(OUT, t + ".json")) as fh:
            out.append(json.load(fh))
    return out


def main(tags):
    res = load(tags)
    print("SCORED RUNS: %s   (scored %d/3)" % (", ".join(tags), len(tags)))
    print("\n== CEILINGS (mean over seeds) ==")
    for r in ROWS:
        pc = np.mean([x["pipeline_ceiling_rows"][r]["validity"] for x in res])
        sc = np.mean([x["span_ceiling_rows"][r]["validity"] for x in res])
        rc = np.mean([x["representation_ceiling"][r]["validity"] for x in res])
        flag = "  <-- CEILING-BOUND (BAR 5)" if pc < 0.50 else ""
        print("  %-14s representation %.4f  pipeline %.4f  store-span %.4f%s"
              % (r, rc, pc, sc, flag))

    print("\n== ENVIRONMENT COLUMNS on this hub's split (mean over seeds) ==")
    seq = [v for x in res for k, v in
           x["env_columns"]["unseen_pair"]["blind"].items()
           if k.startswith("BLIND-SEQUENTIAL")]
    print("  BLIND-SEQUENTIAL validity %.4f   [published CMAZE-1: 0.4980]"
          % np.mean([v["validity"] for v in seq]))
    for rad in (1, 2):
        same = np.mean([x["env_columns"]["fragment"]["FRAG-r%d" % rad]["score"]
                        for x in res])
        cross = np.mean([x["env_columns"].get("fragment_cross_maze", {})
                         .get("FRAG-cross-r%d" % rad, {}).get("score", np.nan)
                         for x in res])
        print("  FRAGMENT-LOOKUP r=%d  same-maze %.4f   cross-maze %.4f"
              "   [published CMAZE-1 cross-maze: 0.0000]" % (rad, same, cross))
    print("  rule B bends %.4f of these routes   [published CMAZE-1: 0.816]"
          % np.mean([x["env_columns"]["unseen_pair"]["bend"]["bend_fraction"]
                     for x in res]))

    print("\n== RESULT ROWS, validity, every seed (the best is never reported alone) ==")
    arms = ["clu"] + list(COMPARATORS)
    for r in ROWS:
        print("  ROW %s" % r)
        print("    %-20s %s   mean" % ("arm", "  ".join("seed%d" % x["seed"]
                                                        for x in res)))
        for a in arms:
            vals = [x["table"][r].get(a, {}).get("validity", float("nan"))
                    for x in res]
            print("    %-20s %s   %.4f"
                  % (a, "  ".join("%.4f" % v for v in vals), float(np.mean(vals))))

    print("\n== THE ESTIMAND: paired corners, clu minus comparator, by bend class ==")
    print("   margin declared in PREREG-w9-maze-1: %.2f absolute validity, "
          "direction clu > comparator" % MARGIN)
    corners, wins, losses = [], 0, 0
    for r in ROWS:
        for x in res:
            for bend in ("bend", "nobend"):
                c = x["table"][r]["clu"][bend]["validity"]
                for a in ("mlp", "lstm", "transformer", "lookup_matched",
                          "landscape_shuffled"):
                    if a not in x["table"][r]:
                        continue
                    v = x["table"][r][a][bend]["validity"]
                    if np.isnan(c) or np.isnan(v):
                        continue
                    d = c - v
                    corners.append({"row": r, "seed": x["seed"], "bend": bend,
                                    "comparator": a, "delta": float(d)})
                    wins += d >= MARGIN
                    losses += d <= -MARGIN
    d = np.array([q["delta"] for q in corners])
    print("   corners: %d   (>= 49 required)" % len(corners))
    print("   clu ahead by the margin: %d    behind by the margin: %d    "
          "within the margin: %d" % (wins, losses, len(d) - wins - losses))
    print("   mean paired delta %.4f   median %.4f   [negative = the CLU loses]"
          % (d.mean(), float(np.median(d))))
    for a in ("mlp", "lstm", "transformer", "lookup_matched", "landscape_shuffled"):
        da = np.array([q["delta"] for q in corners if q["comparator"] == a])
        print("     vs %-20s n=%3d  mean %+.4f  ahead %2d  behind %2d"
              % (a, da.size, da.mean(), int((da >= MARGIN).sum()),
                 int((da <= -MARGIN).sum())))
    for bend in ("bend", "nobend"):
        db = np.array([q["delta"] for q in corners if q["bend"] == bend])
        print("     in the %-7s class      n=%3d  mean %+.4f" % (bend, db.size,
                                                                 db.mean()))

    print("\n== BARS ==")
    b1 = []
    for r in ("unseen_pair", "nway", "budget_interp"):
        c = np.mean([x["table"][r]["clu"]["validity"] for x in res])
        best = max(np.mean([x["table"][r][a]["validity"] for x in res])
                   for a in ("mlp", "lstm", "transformer"))
        b1.append((r, c, best, c - best))
        print("  BAR 1 %-14s clu %.4f  best learned baseline %.4f  delta %+.4f  %s"
              % (r, c, best, c - best, "PASS" if c - best >= MARGIN else "FAIL"))
    for r in ("unseen_pair", "nway", "budget_interp"):
        c = np.mean([x["table"][r]["clu"]["validity"] for x in res])
        lk = np.mean([x["table"][r]["lookup_matched"]["validity"] for x in res])
        fr = np.mean([x["env_columns"]["fragment"]["FRAG-r2"]["score"] for x in res])
        print("  BAR 2 %-14s clu %.4f  lookup %.4f (delta %+.4f)  frag-same-maze r2 "
              "%.4f (delta %+.4f)  %s"
              % (r, c, lk, c - lk, fr, c - fr,
                 "PASS" if (c - lk >= MARGIN and c - fr >= MARGIN) else "FAIL"))
    lr = np.mean([x["table"]["seen"]["landscape_removed"]["validity"] for x in res])
    ls = np.mean([x["table"]["seen"]["landscape_shuffled"]["validity"] for x in res])
    ra = np.mean([x["table"]["seen"]["random_atoms"]["validity"] for x in res])
    print("  BAR 3 controls on seen: landscape_removed %.4f  landscape_shuffled %.4f  "
          "random_atoms %.4f  %s" % (lr, ls, ra,
                                     "PASS" if max(lr, ls) <= 0.05 else "FAIL"))
    print("  BAR 4 easy half: mean paired delta in bend %+.4f vs nobend %+.4f"
          % (np.mean([q["delta"] for q in corners if q["bend"] == "bend"]),
             np.mean([q["delta"] for q in corners if q["bend"] == "nobend"])))
    ex = [x["display"]["delete_green_exactness"] for x in res if "display" in x]
    if ex:
        print("  BAR 6 deletion exactness: statistic max %.3e  subspace max %.3e  %s"
              % (max(e["statistic_max_rel_residual"] for e in ex),
                 max(e["subspace_max_projector_residual"] for e in ex),
                 "PASS" if (max(e["statistic_max_rel_residual"] for e in ex) <= 1e-10
                            and max(e["subspace_max_projector_residual"]
                                    for e in ex) <= 1e-6) else "FAIL"))
        print("  BAR 6 refusal: impossible(budget) %.4f  impossible(walled) %.4f  "
              "possible(mt_unseen) %.4f  FALSE REFUSAL(grounded) %.4f  %s"
              % (np.mean([x["display"]["refusal"]["budget_infeasible"]["refusal"]
                          for x in res]),
                 np.mean([x["display"]["refusal"]["walled"]["refusal"] for x in res]),
                 np.mean([x["display"]["refusal"]["mt_unseen"]["refusal"]
                          for x in res]),
                 np.mean([x["display"]["refusal"]["grounded_possible"]["refusal"]
                          for x in res]),
                 "PASS" if np.mean([x["display"]["refusal"]["grounded_possible"]
                                    ["refusal"] for x in res]) <= 0.20 else "FAIL"))
    c4 = [x.get("c4_nearest_item", {}).get("unseen_pair") for x in res]
    c4 = [q for q in c4 if q]
    if c4:
        print("  C.4 nearest-item ratio on unseen_pair: %.4f  (%s)"
              % (np.mean([q["ratio"] for q in c4]), c4[0]["verdict"]))
    with open(os.path.join(OUT, "summary_corners.json"), "w") as fh:
        json.dump(corners, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
