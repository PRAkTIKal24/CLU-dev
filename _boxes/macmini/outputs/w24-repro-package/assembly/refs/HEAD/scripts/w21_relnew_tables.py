"""w21 spoke/rel-newseeds tables: CLU-REL-A10 and A1 on BOTH TEST, six seeds, one table set.

    python scripts/w21_relnew_tables.py <out_dir>        (writes TABLES.md + REL-NEWSEEDS.json)

⛔ CMAZE-T only.  No bar, no label, no verdict.  Every cell is generated from a run JSON (or a
zero-fit replay JSON), whose path is printed under the table.  The arm tables, the ceiling
definition, deletion/rollout/training tables are ``scripts/w21_tables.py``'s functions,
imported UNCHANGED, and applied identically to both configs.

Sources
  CLU-REL-A10 BOTH TEST  seeds 0-2  outputs/w21-maze-t-clu-improve/test   (peer scored 7-9/9)
                         seeds 3-5  <out>/test                              (this spoke 1-3/3)
  A1          BOTH TEST  seeds 0-2  outputs/w19-maze-t-symmetric/test
                         seeds 3-5  outputs/w21-maze-t-seed4/test
  CLU-REL-A10 BOTH VAL   seeds 0-2  outputs/w21-maze-t-clu-improve/val_full (free)
                         seeds 3-5  outputs/w21-maze-t-seed4/val_cluimprove (free)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w21_tables as WT                                              # noqa: E402

OUT_PEER = "C:/chlu/outputs/w21-maze-t-clu-improve"
OUT_W19 = "C:/chlu/outputs/w19-maze-t-symmetric"
OUT_S4 = "C:/chlu/outputs/w21-maze-t-seed4"
ALL = (0, 1, 2, 3, 4, 5)
ROWS = ("nway", "seen", "pair", "value")
CONTROLS = ("landscape_removed", "landscape_shuffled", "landscape_shuffled_across_wells",
            "random_atoms")
PLANNERS = ("ORACLE", "BLIND-A", "BLIND-B", "BLIND-COLOUR", "BLIND-SEQUENTIAL", "BLIND-FACTOR")
REL_PREFIX = WT.CEIL_PREFIX + ("diagnostic+rel@",)
#: printing convention for rival convergence (raw losses always printed beside it)
NC_HIGH = 1e-2
CAPTION = ("CMAZE-T, condition BOTH (words-plus-observation), TEST leg. Six seeds (0-5), all "
           "run, all printed, none discarded. CLU-REL-A10 = the peer's frozen config "
           "(sha256 0d4ec5b9…, K 2, 10 atoms, ridge 1, 9,200 read floats); A1 = latent 128, K 2, "
           "6 atoms, ridge 1 (4,584 read floats). Rivals are re-matched to each config's own "
           "read-path floats by the standing rule and trained by the frozen 8k / 80k recipe. "
           "Seed count is a column on every table.")


def f(x):
    return "—" if x is None else "%.3f" % x


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def sources(out):
    S = {"REL": {}, "A1": {}, "REL-VAL": {}}
    for s in ALL:
        c = {"REL": os.path.join(OUT_PEER if s < 3 else out, "test", "BOTH_test_seed%d.json" % s),
             "A1": os.path.join(OUT_W19 if s < 3 else OUT_S4, "test", "BOTH_test_seed%d.json" % s),
             "REL-VAL": os.path.join(OUT_PEER if s < 3 else OUT_S4,
                                     "val_full" if s < 3 else "val_cluimprove",
                                     "BOTH_val_seed%d.json" % s)}
        for t, p in c.items():
            j = load(p)
            if j is not None and not j.get("smoke"):
                S[t][s] = (j, p)
    for s, (j, p) in list(S["REL"].items()) + list(S["REL-VAL"].items()):
        assert j["w21_clu_cfg"]["sha256"].startswith("0d4ec5b9fba55bd6"), p
        assert j["w21_clu_cfg"]["name"] == "CLU-REL-A10", p
    for s, (j, p) in S["A1"].items():
        assert j.get("w21_clu_cfg") is None and j["config"]["n_atoms"] == 6, p
    return S


def num(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and k != "n"
            and not k.endswith(":coverage")}


def best_of(d, pred):
    c = {k: v for k, v in num(d).items() if pred(k)}
    if not c:
        return None, []
    b = max(c.values())
    return b, sorted(k for k, v in c.items() if v == b)


def is_matched_rival(k):
    return k.startswith(("rival: ", "rival@10x-steps: ")) and "+rel" not in k


def is_rel(k):
    return k.startswith("diagnostic+rel@")


def per_tuple(out, cfg, s, J):
    if J.get("PER_TUPLE_NWAY"):
        return J["PER_TUPLE_NWAY"], "run JSON PER_TUPLE_NWAY"
    p = (os.path.join(out, "replay", "BOTH_test_seed%d_PERTUPLE.json" % s) if cfg == "REL"
         else os.path.join(OUT_S4, "replay", "BOTH_test_seed%d_PERTUPLE.json" % s))
    j = load(p)
    if j is None:
        return None, None
    assert j["all_param_hashes_match"] and j["all_reproduce"], p
    return j["PER_TUPLE_NWAY"], p


# ------------------------------------------------------------------ section builders
def headline(S, cls, flags=None):
    flagged = {(c, d["seed"], d["arm"]) for c, v in (flags or {}).items() for d in v}
    L = ["", "## 0. Six-seed side-by-side, n-way row (exact) — both configs, same rows, same "
         "arms, same ceiling definition", "",
         "Ceiling = best n-way over every arm FITTED TO THE TRAIN DATA (`w21_tables.CEIL_PREFIX`: "
         "CLU, matched rivals 8k and 80k, 10x-read-floats diagnostic, controls, learned blind "
         "arms, lookup floors) — the definition used for A1 in maze-t-seed4. Planner columns "
         "(GIVEN the engine rules; ORACLE = 1.000 by construction) are EXCLUDED and printed "
         "beside. The `+rel` diagnostic arms exist only in CLU-REL-A10 runs and are NOT in that "
         "ceiling; a second CLU-REL-A10 ceiling that includes them is printed as its own row.",
         "", "| seed | class (A1 ceiling, maze-t-seed4) | n rows | A1 CLU | A1 ceiling | A1 ceiling arm | "
         "A1 best matched rival (arm) | REL CLU | REL ceiling | REL ceiling arm | REL ceiling incl. "
         "+rel (arm) | REL best matched rival (arm) | REL best +rel diagnostic (arm) | REL max "
         "control (which) | planners ORACLE / BLIND-SEQ / BLIND-B |",
         "|" + "---|" * 15]
    agg = {}
    for s in ALL:
        a = S["A1"].get(s)
        r = S["REL"].get(s)
        if r is None:
            L.append("| %d | %s | NOT RUN / NOT FINISHED |" % (s, cls.get(s, "?")))
            continue
        rn, an = r[0]["ROWS"]["nway"], a[0]["ROWS"]["nway"]
        assert rn["n"] == an["n"], s
        ac, aw = WT.ceiling(an)
        rc, rw = WT.ceiling(rn)
        rc2, rw2 = WT.ceiling(rn, REL_PREFIX)
        am, amw = best_of(an, is_matched_rival)
        rm, rmw = best_of(rn, is_matched_rival)
        rr, rrw = best_of(rn, is_rel)
        mc, mcw = best_of(rn, lambda k: k.startswith("control: "))
        pl = " / ".join(f(rn.get("column: %s (planner)" % p)) for p in
                        ("ORACLE", "BLIND-SEQUENTIAL", "BLIND-B"))
        L.append("| %d | %s | %d | %s | %s | %s | %s (%s) | **%s** | %s | %s | %s (%s) | %s (%s) | "
                 "%s (%s) | %s (%s) | %s |"
                 % (s, cls.get(s, "?"), rn["n"], f(an["CLU"]), f(ac), "; ".join(aw), f(am),
                    "; ".join(amw), f(rn["CLU"]), f(rc), "; ".join(rw), f(rc2), "; ".join(rw2),
                    f(rm), "; ".join(rmw), f(rr), "; ".join(rrw), f(mc),
                    "; ".join(x.replace("control: ", "") for x in mcw), pl))
        for k, v in (("A1 CLU", an["CLU"]), ("A1 ceiling", ac), ("A1 best matched rival", am),
                     ("REL CLU", rn["CLU"]), ("REL ceiling", rc), ("REL ceiling incl +rel", rc2),
                     ("REL best matched rival", rm), ("REL best +rel", rr),
                     ("REL max control", mc)):
            agg.setdefault(k, {})[s] = v
    seeds = sorted(S["REL"])
    L += ["", "Means over seed sets (seed count in the header; a mean is printed only over "
          "seeds where every cell exists):", "",
          "| quantity | all (%d) %s | orig 0-2 | added 3-5 | easy %s | hard %s |"
          % (len(seeds), seeds, sorted(s for s in seeds if cls.get(s) == "easy"),
             sorted(s for s in seeds if cls.get(s) == "hard")), "|---|---|---|---|---|---|"]
    for k, d in agg.items():
        def m(ss):
            ss = [s for s in ss if s in d and d[s] is not None]
            return "%s (%d)" % (f(np.mean([d[s] for s in ss])), len(ss)) if ss else "—"
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            k, m(seeds), m([s for s in seeds if s < 3]), m([s for s in seeds if s >= 3]),
            m([s for s in seeds if cls.get(s) == "easy"]),
            m([s for s in seeds if cls.get(s) == "hard"])))
    L += ["", "Per-seed differences (REL − A1) and margins, n-way:", "",
          "| seed | REL CLU − A1 CLU | REL CLU − REL best matched rival | REL CLU − REL best +rel | "
          "REL best +rel − REL best matched rival (the '+rel' lift) | REL CLU − REL max control | "
          "A1 CLU − A1 best matched rival | REL CLU − REL best UNFLAGGED matched rival (arm) | "
          "A1 CLU − A1 best UNFLAGGED matched rival (arm) |", "|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        g = lambda k: agg[k].get(s)                                      # noqa: E731
        uf = {}
        for tag, cn in (("REL", "CLU-REL-A10"), ("A1", "A1")):
            d = S[tag][s][0]["ROWS"]["nway"]
            uf[tag] = best_of(d, lambda k: is_matched_rival(k) and (cn, s, k) not in flagged)
        L.append("| %d | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f | %+.3f (%s) | %+.3f (%s) |" % (
            s, g("REL CLU") - g("A1 CLU"), g("REL CLU") - g("REL best matched rival"),
            g("REL CLU") - g("REL best +rel"), g("REL best +rel") - g("REL best matched rival"),
            g("REL CLU") - g("REL max control"), g("A1 CLU") - g("A1 best matched rival"),
            g("REL CLU") - uf["REL"][0], "; ".join(uf["REL"][1]),
            g("A1 CLU") - uf["A1"][0], "; ".join(uf["A1"][1])))
    L += ["", "Sources: " + "; ".join("REL seed %d `%s`" % (s, S["REL"][s][1]) for s in seeds)
          + "; " + "; ".join("A1 seed %d `%s`" % (s, S["A1"][s][1]) for s in sorted(S["A1"]))]
    return L, agg


def rel_table(S):
    runs = S["REL"]
    seeds = sorted(runs)
    L = ["", "## 2. ⭐⭐ The `+rel` DIAGNOSTIC — rivals handed the CLU's relational columns, per "
         "seed and row (CLU-REL-A10 runs only)", "",
         "`+rel` = a copy of each rival's shared input (obs ⊕ 318 design) with CLU-REL-A10's "
         "colour-relational columns appended (`w21_clu.CluFactory.relations_block`), re-matched "
         "to the same read floats, frozen recipe at 8k and 80k. A DIAGNOSTIC, never a rival row "
         "and never a label: it separates the CODING (the relational columns) from the STORE.", ""]
    for row in ROWS:
        L += ["", "#### row `%s` · seeds = %d %s" % (row, len(seeds), seeds), "",
              "| arm | " + " | ".join("seed %d" % s for s in seeds) + " | mean (%d) |" % len(seeds),
              "|---|" + "---|" * (len(seeds) + 1)]
        keys = ["CLU"] + [k for k in WT.arm_keys(runs[seeds[0]][0]["ROWS"][row])
                          if is_matched_rival(k) or is_rel(k)]
        for k in keys:
            v = [runs[s][0]["ROWS"][row].get(k) for s in seeds]
            if any(x is None for x in v):
                continue
            L.append("| %s | %s | %s |" % (WT.pretty(k), " | ".join(f(x) for x in v), f(np.mean(v))))
        bm = [best_of(runs[s][0]["ROWS"][row], is_matched_rival) for s in seeds]
        br = [best_of(runs[s][0]["ROWS"][row], is_rel) for s in seeds]
        cl = [runs[s][0]["ROWS"][row]["CLU"] for s in seeds]
        L.append("| **best matched rival (frozen input, 8k/80k)** | %s | %s |" % (
            " | ".join("%s (%s)" % (f(b[0]), "; ".join(b[1])) for b in bm), f(np.mean([b[0] for b in bm]))))
        L.append("| **best +rel diagnostic (8k/80k)** | %s | %s |" % (
            " | ".join("%s (%s)" % (f(b[0]), "; ".join(b[1])) for b in br), f(np.mean([b[0] for b in br]))))
        L.append("| +rel lift = best +rel − best matched rival | %s | %+.3f |" % (
            " | ".join("%+.3f" % (b[0] - a[0]) for a, b in zip(bm, br)),
            np.mean([b[0] - a[0] for a, b in zip(bm, br)])))
        L.append("| CLU − best matched rival | %s | %+.3f |" % (
            " | ".join("%+.3f" % (c - a[0]) for c, a in zip(cl, bm)),
            np.mean([c - a[0] for c, a in zip(cl, bm)])))
        L.append("| CLU − best +rel | %s | %+.3f |" % (
            " | ".join("%+.3f" % (c - b[0]) for c, b in zip(cl, br)),
            np.mean([c - b[0] for c, b in zip(cl, br)])))
        L.append("| share of the CLU's margin over the best matched rival that the +rel lift "
                 "recovers = lift / (CLU − best matched rival) | %s | — |" % " | ".join(
                     ("%.2f" % ((b[0] - a[0]) / (c - a[0])) if abs(c - a[0]) > 1e-9 else "n/a (margin 0)")
                     for c, a, b in zip(cl, bm, br)))
    L += ["", "+rel parameter counts per seed: " + "; ".join(
        "seed %d %s" % (s, {k: (v["params"], v["input_dim"]) for k, v in
                            runs[s][0].get("rel_diagnostic_parameters", {}).items()}) for s in seeds),
          "", "Sources: " + "; ".join("seed %d `%s`" % (s, runs[s][1]) for s in seeds)]
    return L


def controls_table(S):
    L = ["", "## 3. ⭐⭐ The four charter-C controls against the CLU, per seed (CLU-REL-A10; TEST "
         "and the free VAL leg side by side; A1 TEST beside)", "",
         "`control ≥ CLU` is printed as **TIE/ABOVE** wherever it happens, on any row. The "
         "across-wells derangement is asserted in the runner (no fixed point) and printed.", ""]
    for row in ROWS:
        L += ["", "#### row `%s`" % row, "",
              "| seed | leg | config | n | CLU | " + " | ".join(CONTROLS) +
              " | max control | CLU − max control | control ≥ CLU? | derangement |",
              "|" + "---|" * (11)]
        for s in ALL:
            for tag, leg, cfg in (("REL", "TEST", "CLU-REL-A10"), ("REL-VAL", "VAL (free)",
                                                                   "CLU-REL-A10"),
                                  ("A1", "TEST", "A1")):
                x = S[tag].get(s)
                if x is None:
                    L.append("| %d | %s | %s | NOT AVAILABLE |||||||||" % (s, leg, cfg))
                    continue
                d = x[0]["ROWS"].get(row)
                if not isinstance(d, dict) or "CLU" not in d:
                    L.append("| %d | %s | %s | row not run |||||||||" % (s, leg, cfg))
                    continue
                cv = [d.get("control: " + c) for c in CONTROLS]
                mx = max(v for v in cv if v is not None)
                who = [c for c, v in zip(CONTROLS, cv) if v == mx]
                tie = [c for c, v in zip(CONTROLS, cv) if v is not None and v >= d["CLU"]]
                L.append("| %d | %s | %s | %d | %s | %s | %s (%s) | %+.4f | %s | %s |" % (
                    s, leg, cfg, d["n"], "%.4f" % d["CLU"], " | ".join(
                        "%.4f" % v if v is not None else "—" for v in cv), "%.4f" % mx,
                    "; ".join(who), d["CLU"] - mx,
                    ("**TIE/ABOVE: %s**" % "; ".join(tie)) if tie else "no",
                    x[0].get("across_wells_derangement")))
    L += ["", "Sources: the BOTH run JSONs listed in §0 (TEST) and, for VAL, " + "; ".join(
        "seed %d `%s`" % (s, S["REL-VAL"][s][1]) for s in sorted(S["REL-VAL"]))]
    return L


def convergence_table(S):
    L = ["", "## 4. ⭐ Rival convergence — every rival's final training loss, both configs, "
         "non-convergence flagged", "",
         "Printing convention (raw losses always shown): **WORSE@80k** = the 80k run of the same "
         "rival ends at a final loss ≥ its 8k run's; **HIGH** = final loss > %.0e. The `seen` "
         "column is the arm's exact score on held-in (train-distribution) rows — a rival that "
         "cannot fit what it was trained on is not a rival. A margin over a flagged rival is not "
         "a margin." % NC_HIGH, ""]
    pairs = (("rival: ", "rival@10x-steps: "), ("diagnostic+rel@8k: ", "diagnostic+rel@80k: "))
    flags = {}
    for tag, cfgname in (("REL", "CLU-REL-A10"), ("A1", "A1")):
        runs = S[tag]
        L += ["", "#### %s · seeds = %d %s" % (cfgname, len(runs), sorted(runs)), "",
              "| seed | rival | params | steps | final loss | seen (exact) | n-way (exact) | flag |",
              "|---|---|---|---|---|---|---|---|"]
        for s in sorted(runs):
            J = runs[s][0]
            RT = J.get("rival_training", {})
            for nm in sorted(RT, key=lambda k: (k.split(": ")[1], k)):
                r = RT[nm]
                fl = []
                if r.get("final_loss") is not None and r["final_loss"] > NC_HIGH:
                    fl.append("HIGH")
                for a, b in pairs:
                    if nm.startswith(b):
                        r8 = RT.get(a + nm[len(b):])
                        if r8 and r["final_loss"] >= r8["final_loss"]:
                            fl.append("WORSE@80k (8k %.3g)" % r8["final_loss"])
                if fl:
                    flags.setdefault(cfgname, []).append({"seed": s, "arm": nm, "flags": fl,
                                                          "final_loss": r["final_loss"],
                                                          "nway": J["ROWS"]["nway"].get(nm)})
                L.append("| %d | %s | %s | %s | %.3e | %s | %s | %s |" % (
                    s, WT.pretty(nm), r.get("params"), r.get("steps"), r["final_loss"],
                    f(J["ROWS"]["seen"].get(nm)), f(J["ROWS"]["nway"].get(nm)),
                    ("**" + ", ".join(fl) + "**") if fl else ""))
        L += ["", "Matched targets: " + "; ".join("seed %d %s" % (s, runs[s][0]["matched_target"])
                                                  for s in sorted(runs))]
    L += ["", "Flagged rivals (summary): " + json.dumps(flags)]
    return L, flags


def tuple_side(out, S, s):
    r, a = S["REL"].get(s), S["A1"].get(s)
    L = ["", "#### seed %d · per held-out n-way tuple (w_agent, w_door, w_tint, w_dir)" % s]
    if r is None:
        return L + ["", "CLU-REL-A10 run not available"]
    PR, pr = per_tuple(out, "REL", s, r[0])
    PA, pa = per_tuple(out, "A1", s, a[0]) if a else (None, None)
    if PR is None:
        return L + ["", "per-tuple NOT AVAILABLE (replay not run / not finished)"]
    arms_r = [k for k in PR if k.startswith(WT.CEIL_PREFIX)]
    L += ["", "| tuple | n | A1 CLU | REL CLU | REL best matched rival | REL best +rel | REL tuple "
          "ceiling (arm) | REL max control |", "|---|---|---|---|---|---|---|---|"]
    for t in sorted(PR["CLU"], key=lambda t: [int(x) for x in t.split(",")]):
        n = PR["CLU"][t][0]
        sc = lambda P, k: P[k][t][1] / P[k][t][0] if (P and k in P and t in P[k]) else None  # noqa
        ce = {k: sc(PR, k) for k in arms_r}
        b = max(ce.values())
        who = sorted(k for k, v in ce.items() if v == b)
        bm = max(sc(PR, k) for k in PR if is_matched_rival(k))
        br = max((sc(PR, k) for k in PR if is_rel(k)), default=None)
        mc = max(sc(PR, k) for k in PR if k.startswith("control: "))
        L.append("| (%s) | %d | %s | %s | %s | %s | %s (%s) | %s |" % (
            t, n, f(sc(PA, "CLU")), f(sc(PR, "CLU")), f(bm), f(br), f(b),
            "; ".join(who) if len(who) <= 3 else "%d arms incl. %s" % (len(who), who[0]), f(mc)))
    L += ["", "Per-tuple sources: REL `%s`; A1 `%s`." % (pr, pa)]
    return L


def main(out):
    S = sources(out)
    M4 = load(os.path.join(OUT_S4, "MAZE-T-SEED4.json"))
    cls = {int(k): v for k, v in M4["classification"]["class"].items()}
    L = ["# w21 rel-newseeds — TABLES (generated by scripts/w21_relnew_tables.py)", "",
         "⛔ CMAZE-T only. No PREREG, no bar, no label, no verdict: every arm, seed and row is "
         "printed; the Head decides what is quotable.", "", "**Caption (every table):** " + CAPTION,
         "", "Easy/hard labels are maze-t-seed4's (largest gap in A1's BOTH TEST per-seed "
         "ceilings; `%s`), carried over UNCHANGED so the two configs are read with the same "
         "column split; the same rule applied to CLU-REL-A10's own ceilings is printed in §1."
         % os.path.join(OUT_S4, "MAZE-T-SEED4.json")]
    C, flags = convergence_table(S)
    H, agg = headline(S, cls, flags)
    L += H
    # REL-ceiling classification by the same largest-gap rule (sensitivity only)
    rc = {s: WT.ceiling(S["REL"][s][0]["ROWS"]["nway"])[0] for s in S["REL"]}
    xs = sorted(rc.items(), key=lambda kv: kv[1])
    L += ["", "## 1. Largest-gap rule applied to CLU-REL-A10's own BOTH TEST ceilings (printed "
          "beside A1's; the column split below stays A1's)", ""]
    if len(xs) > 1:
        i = max(range(len(xs) - 1), key=lambda j: xs[j + 1][1] - xs[j][1])
        cut = (xs[i][1] + xs[i + 1][1]) / 2
        L += ["sorted REL ceilings %s → largest gap %.4f between %.4f and %.4f → below-cut seeds %s"
              % ([(s, round(v, 4)) for s, v in xs], xs[i + 1][1] - xs[i][1], xs[i][1], xs[i + 1][1],
                 sorted(s for s, v in xs if v < cut))]
    L += rel_table(S)
    L += controls_table(S)
    L += C
    for tag, name in (("REL", "CLU-REL-A10"), ("A1", "A1")):
        runs = S[tag]
        title = "CMAZE-T · BOTH · TEST · CLU = **%s**" % name
        L += ["", "## %s. Full arm tables — %s (every arm, every row; `w21_tables.arm_table`)"
              % ("5" if tag == "REL" else "6", name)]
        for row in ROWS:
            L += WT.arm_table(title, runs, row, cls)
        L += WT.training_table(title, runs)
        L += WT.deletion_table(title, runs)
        L += WT.rollout_table(title, runs)
    L += ["", "## 7. Per-tuple n-way, CLU-REL-A10 beside A1 (same held-out tuples and rows: banks "
          "and hold-outs depend only on seed and leg)", ""]
    for s in ALL:
        L += tuple_side(out, S, s)
    rep = {}
    for s in ALL:
        j = load(os.path.join(out, "replay", "BOTH_test_seed%d_PERTUPLE.json" % s))
        if j:
            rep[s] = {"objects": j["n_objects_checked"], "hashes_match": j["all_param_hashes_match"],
                      "reproduce_all": j["all_reproduce"], "not_replayed": j["run_json_arms_not_replayed"],
                      "path": os.path.join(out, "replay", "BOTH_test_seed%d_PERTUPLE.json" % s)}
    L += ["", "## 8. Zero-fit replay status (CLU-REL-A10 BOTH TEST; `scripts/w21_relnew_replay.py`)",
          "", "| seed | objects hash-checked (param + file sha256) | all match | n-way numbers "
          "reproduce | arms not replayed | path |", "|---|---|---|---|---|---|"]
    for s in ALL:
        x = rep.get(s)
        L.append("| %d | %s |" % (s, " | ".join(str(v) for v in x.values()) if x else "NOT RUN"))
    dg = load(os.path.join(OUT_S4, "diagnosis", "SEED-DIAGNOSIS-v2.json"))
    L += ["", "## 9. CLU-REL-A10 TEST misses vs the zero-fit relation-unreachable rows "
          "(maze-t-seed4 `diagnosis/SEED-DIAGNOSIS-v2.json`, TEST leg; zero-fit, banks + hold-outs "
          "only)", "", "| seed | n | REL CLU exact rows | REL CLU missed rows | unidentified-for-colour-"
          "identity rows | …also unreachable by colour relation (rows) | 1 − unreachable/n | REL CLU "
          "− (1 − unreachable/n) | REL ceiling incl +rel |", "|---|---|---|---|---|---|---|---|---|"]
    for s in ALL:
        d = [x for x in (dg or {}).get("runs", []) if x["seed"] == s and x["leg"] == "test"]
        if not d or s not in S["REL"]:
            L.append("| %d | not available |" % s)
            continue
        d = d[0]
        rn = S["REL"][s][0]["ROWS"]["nway"]
        assert d["n_eval_nway_rows"] == rn["n"], s
        ex = int(round(rn["CLU"] * rn["n"]))
        u = d["unidentified_rows_ALSO_unreachable_by_colour_relation"]
        L.append("| %d | %d | %d | %d | %d | %d | %.4f | %+.4f | %s |" % (
            s, rn["n"], ex, rn["n"] - ex, d["unreachable_eval_rows"], u, 1 - u / rn["n"],
            rn["CLU"] - (1 - u / rn["n"]), f(WT.ceiling(rn, REL_PREFIX)[0])))
    L += ["", "Source: `%s`." % os.path.join(OUT_S4, "diagnosis", "SEED-DIAGNOSIS-v2.json")]
    with open(os.path.join(out, "TABLES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    M = {"caption": CAPTION, "ceiling_definition": WT.CEIL_PREFIX,
         "ceiling_incl_rel_definition": REL_PREFIX, "class_from_maze_t_seed4": cls,
         "sources": {t: {s: v[1] for s, v in d.items()} for t, d in S.items()},
         "nway_summary": agg, "rival_nonconvergence_flags": flags, "replay": rep,
         "nway_all_arms": {t: {s: num(v[0]["ROWS"]["nway"]) for s, v in d.items()}
                           for t, d in S.items()}}
    with open(os.path.join(out, "REL-NEWSEEDS.json"), "w") as fh:
        json.dump(M, fh, indent=1)
    print("wrote TABLES.md + REL-NEWSEEDS.json; REL seeds %s A1 %s REL-VAL %s replay %s"
          % (sorted(S["REL"]), sorted(S["A1"]), sorted(S["REL-VAL"]), sorted(rep)))


if __name__ == "__main__":
    main(sys.argv[1])
