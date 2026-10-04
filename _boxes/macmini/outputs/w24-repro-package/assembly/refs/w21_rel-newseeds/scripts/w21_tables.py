"""w21 spoke/maze-t-seed4 tables: every cell generated from a JSON.

    python scripts/w21_tables.py <out_dir>          (writes TABLES.md + MAZE-T-SEED4.json)

⛔ CMAZE-T only.  No bar, no label, no verdict: every arm, every seed, every row is printed and
the Head decides what is quotable.  Seed count is a column on every table; means are printed
per seed SET and the set is named (all / easy / hard / original 0-2); rows with different
seed sets are never averaged together.  Every table carries the PER-SEED CEILING row (best
score any TRAINED arm reaches on that seed, arm named).
"""
import json
import os
import sys

import numpy as np

OUT_W19 = "C:/chlu/outputs/w19-maze-t-symmetric"
OUT_WF = "C:/chlu/outputs/w21-words-full-test"
OUT_PEER = "C:/chlu/outputs/w21-maze-t-clu-improve"
SEEDS_ORIG, SEEDS_NEW = (0, 1, 2), (3, 4, 5)
ALL = SEEDS_ORIG + SEEDS_NEW
ROWS = ("nway", "seen", "pair", "value")
CAPTION = ("Three original seeds (0, 1, 2) plus three added seeds (3, 4, 5), ALL run and ALL "
           "printed; easy seeds in the main text and hard seeds in the appendix, classified by "
           "the measured per-seed CEILING of the words-plus-observation (BOTH) TEST n-way row "
           "(never by the CLU's own score). 6 seeds tried, 6 kept, 0 discarded.")
CEIL_PREFIX = ("CLU", "rival: ", "rival@10x-steps: ", "diagnostic@", "control: ", "arm: ",
               "floor: ")


def f(x):
    return "—" if x is None else "%.3f" % x


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def sources(out):
    """{table: {seed: (json, path)}} -- where every cell comes from."""
    S = {"BOTH-TEST": {}, "OBS-VAL": {}, "WORDS-FULL-TEST": {}, "CLU-REL-A10-BOTH-VAL": {}}
    for s in ALL:
        cands = {
            "BOTH-TEST": [os.path.join(OUT_W19 if s < 3 else out, "test",
                                       "BOTH_test_seed%d.json" % s)],
            "OBS-VAL": [os.path.join(OUT_W19 if s < 3 else out, "val", "OBS_val_seed%d.json" % s)],
            "WORDS-FULL-TEST": [os.path.join(OUT_WF, "test", "WORDS-FULL_test_seed%d.json" % s)]
            if s < 3 else [],
            "CLU-REL-A10-BOTH-VAL": [os.path.join(OUT_PEER if s < 3 else out,
                                                  "val_full" if s < 3 else "val_cluimprove",
                                                  "BOTH_val_seed%d.json" % s)]}
        for t, ps in cands.items():
            for p in ps:
                j = load(p)
                if j is not None:
                    S[t][s] = (j, p)
    return S


def per_tuple(out, table, s, J):
    """Per-tuple n-way counts: in the run JSON (w21 runner) or in the zero-fit replay."""
    if J is not None and J.get("PER_TUPLE_NWAY"):
        return J["PER_TUPLE_NWAY"], "run JSON PER_TUPLE_NWAY"
    nm = {"BOTH-TEST": "BOTH_test", "OBS-VAL": "OBS_val",
          "WORDS-FULL-TEST": "WORDS-FULL_test"}.get(table)
    if nm is None:
        return None, None
    p = os.path.join(out, "replay", "%s_seed%d_PERTUPLE.json" % (nm, s))
    j = load(p)
    if j is None:
        return None, None
    assert j["all_param_hashes_match"] and j["all_reproduce"], p
    return j["PER_TUPLE_NWAY"], p


def arm_keys(rows):
    keys = [k for k in rows if isinstance(rows[k], (int, float)) and k != "n"
            and not k.endswith(":coverage")]
    order = []
    for pre in ("CLU", "rival: ", "rival@10x-steps: ", "diagnostic@", "control: ", "arm: ",
                "floor: ", "column: "):
        order += sorted(k for k in keys if k.startswith(pre) and k not in order)
    order += sorted(k for k in keys if k not in order)
    return order


MATCHED_PREFIX = ("CLU", "rival: ", "rival@10x-steps: ")


def ceiling(rowd, prefix=CEIL_PREFIX):
    c = {k: v for k, v in rowd.items() if isinstance(v, (int, float)) and k != "n"
         and not k.endswith(":coverage") and k.startswith(prefix)
         and not (prefix is MATCHED_PREFIX and "+rel" in k)}
    best = max(c.values())
    arms = sorted(k for k, v in c.items() if v == best)
    return best, arms


def pretty(k):
    if k.startswith("rival: "):
        return k.replace("rival: ", "rival (matched, 8,000 steps): ")
    if k.startswith("rival@10x-steps: "):
        return k.replace("rival@10x-steps: ", "rival (matched, 80,000 steps): ")
    if k.startswith("diagnostic@10x-read-floats: "):
        return k + " (8,000 steps, 10x read floats; NOT matched)"
    if k.startswith("column: "):
        return k + " [GIVEN the engine rules; not in the ceiling]"
    return k


def classify(S):
    """Easy/hard from the BOTH TEST n-way CEILING: the cut is the LARGEST GAP between
    consecutive sorted ceilings; every other possible cut is printed as sensitivity."""
    ce = {s: ceiling(S["BOTH-TEST"][s][0]["ROWS"]["nway"])[0] for s in S["BOTH-TEST"]}
    xs = sorted(ce.items(), key=lambda kv: kv[1])
    if len(xs) < 2:
        return ce, None, {}, []
    gaps = [(xs[i + 1][1] - xs[i][1], i) for i in range(len(xs) - 1)]
    g, i = max(gaps)
    cut = (xs[i][1] + xs[i + 1][1]) / 2
    cls = {s: ("hard" if v < cut else "easy") for s, v in ce.items()}
    sens = [{"cut_between": [xs[j][1], xs[j + 1][1]], "gap": xs[j + 1][1] - xs[j][1],
             "hard_seeds": sorted(s for s, _ in xs[:j + 1])} for j in range(len(xs) - 1)]
    return ce, {"cut": cut, "gap": g, "between": [xs[i][1], xs[i + 1][1]],
                "rule": "largest gap between consecutive sorted per-seed ceilings (BOTH TEST n-way)"}, cls, sens


def arm_table(title, runs, row, cls, note=""):
    seeds = sorted(runs)
    L = ["", "#### %s · row `%s` (exact) · seeds = %d %s · rows/seed %s"
         % (title, row, len(seeds), seeds, [runs[s][0]["ROWS"][row]["n"] for s in seeds])]
    if note:
        L += ["", note]
    easy = [s for s in seeds if cls.get(s) == "easy"]
    hard = [s for s in seeds if cls.get(s) == "hard"]
    orig = [s for s in seeds if s in SEEDS_ORIG]
    hdr = ["arm", "mean all (%d)" % len(seeds)]
    if cls:
        hdr += ["mean easy %s" % easy, "mean hard %s" % hard]
    hdr += ["mean orig %s" % orig] + ["seed %d%s" % (s, (" (%s)" % cls[s]) if s in cls else "")
                                       for s in seeds]
    L += ["", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    keys = arm_keys(runs[seeds[0]][0]["ROWS"][row])
    for k in keys:
        v = {s: runs[s][0]["ROWS"][row].get(k) for s in seeds}
        if any(x is None for x in v.values()):
            continue

        def m(ss):
            return f(np.mean([v[s] for s in ss])) if ss else "—"
        name = pretty(k)
        if k.startswith("control: landscape_shuffled_across_wells"):
            name += " (DERANGED %s)" % [runs[s][0]["across_wells_derangement"] for s in seeds]
        cells = [name, m(seeds)] + ([m(easy), m(hard)] if cls else []) + [m(orig)] + \
                [f(v[s]) for s in seeds]
        L.append("| " + " | ".join(cells) + " |")
    ce = {s: ceiling(runs[s][0]["ROWS"][row]) for s in seeds}
    cells = ["**PER-SEED CEILING** (best trained arm)", f(np.mean([ce[s][0] for s in seeds]))]
    if cls:
        cells += [f(np.mean([ce[s][0] for s in easy])) if easy else "—",
                  f(np.mean([ce[s][0] for s in hard])) if hard else "—"]
    cells += [f(np.mean([ce[s][0] for s in orig])) if orig else "—"]
    cells += ["**%s**" % f(ce[s][0]) for s in seeds]
    L.append("| " + " | ".join(cells) + " |")
    L.append("| ceiling reached by | | " + ("| | " if cls else "") + "| "
             + " | ".join("; ".join(ce[s][1]) for s in seeds) + " |")
    cm = {s: ceiling(runs[s][0]["ROWS"][row], MATCHED_PREFIX) for s in seeds}
    L.append("| ceiling among MATCHED arms only (CLU + matched rivals 8k/80k) | %s | " % f(
        np.mean([cm[s][0] for s in seeds])) + ("| | " if cls else "") + "| "
             + " | ".join("%s (%s)" % (f(cm[s][0]), "; ".join(cm[s][1])) for s in seeds) + " |")
    rp = ("rival: ", "rival@10x-steps: ")
    rb = {s: (lambda d: (max(d.values()), sorted(k for k, v in d.items() if v == max(d.values()))))(
        {k: v for k, v in runs[s][0]["ROWS"][row].items() if isinstance(v, (int, float))
         and k.startswith(rp) and "+rel" not in k}) for s in seeds}
    L.append("| best MATCHED RIVAL per seed (over kinds, input variants, 8k AND 80k) | %s | "
             % f(np.mean([rb[s][0] for s in seeds])) + ("| | " if cls else "") + "| "
             + " | ".join("%s (%s)" % (f(rb[s][0]), "; ".join(rb[s][1])) for s in seeds) + " |")
    L.append("| CLU − best matched rival (per seed) | %+.3f | " % np.mean(
        [runs[s][0]["ROWS"][row]["CLU"] - rb[s][0] for s in seeds]) + ("| | " if cls else "")
             + "| " + " | ".join("%+.3f" % (runs[s][0]["ROWS"][row]["CLU"] - rb[s][0])
                                 for s in seeds) + " |")
    L.append("| CLU − ceiling | | " + ("| | " if cls else "") + "| "
             + " | ".join("%+.3f" % (runs[s][0]["ROWS"][row]["CLU"] - ce[s][0]) for s in seeds)
             + " |")
    if "word_extractor_accuracy" in runs[seeds[0]][0]["ROWS"][row]:
        L.append("| word extractor accuracy, whole tuple | | " + ("| | " if cls else "") + "| "
                 + " | ".join(f(runs[s][0]["ROWS"][row]["word_extractor_accuracy"]["whole_tuple"])
                              for s in seeds) + " |")
    L += ["", "Sources: " + "; ".join("seed %d `%s`" % (s, runs[s][1]) for s in seeds)]
    return L


def training_table(title, runs):
    seeds = sorted(runs)
    L = ["", "#### %s · rival training record (final loss / params / steps) · seeds = %d"
         % (title, len(seeds)), "",
         "| rival | params | steps | " + " | ".join("seed %d final loss" % s for s in seeds) + " |",
         "|---|---|---|" + "---|" * len(seeds)]
    names = sorted(runs[seeds[0]][0].get("rival_training", {}))
    for n in names:
        r0 = runs[seeds[0]][0]["rival_training"][n]
        L.append("| %s | %s | %s | %s |" % (n, r0.get("params"), r0.get("steps"), " | ".join(
            "%.2e" % runs[s][0]["rival_training"].get(n, {}).get("final_loss", float("nan"))
            for s in seeds)))
    mp = runs[seeds[0]][0].get("matched_target")
    L += ["", "Matched target (read-path floats): %s" % mp]
    return L


def deletion_table(title, runs):
    seeds = sorted(runs)
    L = ["", "#### %s · DELETION (colour 1) · seeds = %d" % (title, len(seeds)), "",
         "Store-scoped exactness: store after deletion == store FITTED WITHOUT the rows. Frozen "
         "codec, state encoder (and extractor) are OUTSIDE the deletable object. Post-deletion "
         "adaptation = downdate of the statistics + the same closed-form re-solve: light and "
         "data-free, NOT 'no changes'. The rival deletes only by retrain-without (cost printed).",
         "", "| seed | mode | rows deleted | max rel ΔS | atoms max abs Δ | CLU after (n-way) | CLU "
         "fitted-without (n-way) | reads identical | CLU cost s | rival retrain-without (n-way) | "
         "rival as-trained (n-way) | rival retrain s / grad steps / examples |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        D = runs[s][0].get("DELETION", {})
        for mode in ("value", "conjunction_colour_AND_tinted_floor"):
            b = D.get(mode)
            if not isinstance(b, dict) or "store_scoped_exactness" not in b:
                L.append("| %d | %s | %s |||||||||| " % (s, mode, b))
                continue
            e = b["store_scoped_exactness"]
            c = b["composition_on_surviving_colours"].get("nway", {})
            r = b["rival_composition_on_surviving_colours"].get("nway", {})
            rc = b["rival_retrain_without_cost"]
            L.append("| %d | %s | %d | %.2e | %.2e | %s | %s | %s | %.2f | %s | %s | %.0f / %d / %d |"
                     % (s, mode, b["n_rows_deleted"], e["max_relative_delta_S"],
                        e["deployed_atoms_max_abs_delta"],
                        f(c.get("CLU after deletion (downdate)")),
                        f(c.get("CLU fitted WITHOUT the rows")), f(c.get("reads_identical_rows_frac")),
                        b["clu_adaptation_cost"]["seconds"], f(r.get("rival retrain-without")),
                        f(r.get("rival as-trained")), rc["seconds"], rc["gradient_steps"],
                        rc["examples_seen"]))
    return L


def rollout_table(title, runs):
    seeds = sorted(runs)
    L = ["", "#### %s · per-step exactness and rollout validity (limitation) · seeds = %d"
         % (title, len(seeds)), "",
         "| arm | " + " | ".join("seed %d per-step / validity" % s for s in seeds) + " |",
         "|---|" + "---|" * len(seeds)]
    names = list(runs[seeds[0]][0]["ROLLOUT"]["arms"])
    for n in names:
        L.append("| %s | %s |" % (pretty(n), " | ".join(
            "%s / %s" % (f(runs[s][0]["ROLLOUT"]["arms"].get(n, {}).get("per_step_exactness")),
                         f(runs[s][0]["ROLLOUT"]["arms"].get(n, {}).get("validity")))
            for s in seeds)))
    return L


def tuple_table(out, table, S, diag, s, cls):
    J, p = S[table][s]
    PT, src = per_tuple(out, table, s, J)
    dg = None
    for d in diag.get("runs", []) if diag else []:
        if d["seed"] == s and d["leg"] == ("test" if table.endswith("TEST") else "val"):
            dg = d
    L = ["", "#### %s · seed %d (%s) · per held-out n-way tuple (w_agent, w_door, w_tint, w_dir)"
         % (table, s, cls.get(s, "?"))]
    if PT is None:
        return L + ["", "per-tuple scores NOT AVAILABLE (no per-tuple source)"]
    arms = [a for a in PT if a.startswith(CEIL_PREFIX)]
    tuples = sorted(PT["CLU"], key=lambda t: [int(x) for x in t.split(",")])
    dmap = {",".join(str(x) for x in r["tuple"]): r for r in (dg["tuples"] if dg else [])}
    L += ["", "| tuple | family | n | support class (zero-fit) | blocked by held-out 2-way pair | "
          "reachable by colour relation | CLU | tuple ceiling | ceiling arm(s) | #trained arms "
          "exact > 0.5 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tuples:
        n = PT["CLU"][t][0]
        sc = {a: PT[a][t][1] / PT[a][t][0] for a in arms if t in PT[a]}
        best = max(sc.values())
        who = sorted(a for a, v in sc.items() if v == best)
        r = dmap.get(t, {})
        rel = r.get("relation") or {}
        L.append("| (%s) | %s | %d | %s | %s | %s | %s | %s | %s | %d/%d |"
                 % (t, ",".join(r.get("family", [])), n, r.get("class", "?"),
                    r.get("blocked_by_heldout_2way_pair", "?"),
                    {True: "yes", False: "NO"}.get(rel.get("reachable_by_relation"), "?"),
                    f(sc["CLU"]), f(best), ("; ".join(who) if len(who) <= 3 else
                                             "%d arms incl. %s" % (len(who), "; ".join(who[:2]))),
                    sum(v > 0.5 for v in sc.values()), len(sc)))
    L += ["", "Per-tuple source: `%s`; support classes: `diagnosis/SEED-DIAGNOSIS-v2.json`." % src]
    return L


def main(out):
    S = sources(out)
    diag = load(os.path.join(out, "diagnosis", "SEED-DIAGNOSIS-v2.json"))
    ce, cutinfo, cls, sens = classify(S)
    easy = sorted(s for s, c in cls.items() if c == "easy")
    hard = sorted(s for s, c in cls.items() if c == "hard")
    L = ["# w21 maze-t-seed4 — TABLES (generated by scripts/w21_tables.py)", "",
         "⛔ CMAZE-T only. No PREREG, no bar, no label, no verdict anywhere: every arm, seed and "
         "row is printed; the Head decides what is quotable.", "", "**Caption (every table):** "
         + CAPTION, ""]
    # ---- seed classification
    L += ["## 0. Seed classification by measured CEILING (BOTH TEST n-way)", "",
          "Ceiling = best n-way exact over every arm FITTED TO THE TRAIN DATA (CLU, every "
          "matched rival at 8k and 80k, the 10x-read-floats diagnostic, learned blind arms, "
          "controls, lookup floors). The planner columns are GIVEN the engine rules (ORACLE = "
          "1.000 by construction) and are excluded.", ""]
    if cutinfo:
        L += ["**Cut:** %s → cut = %.4f (gap %.4f between %.4f and %.4f). Easy = %s; hard = %s."
              % (cutinfo["rule"], cutinfo["cut"], cutinfo["gap"], cutinfo["between"][0],
                 cutinfo["between"][1], easy, hard), ""]
    L += ["**Main text / appendix split:** in every arm table below, the columns marked `(easy)` and "
          "the `mean easy` column are the MAIN-TEXT view; the columns marked `(hard)` and `mean hard` "
          "are the APPENDIX view. Both are printed in full in every table; nothing is dropped.", ""]
    L += ["| seed | origin | BOTH TEST ceiling | reached by | CLU | CLU − ceiling | class | "
          "OBS VAL ceiling | WORDS-FULL TEST ceiling | CLU-REL-A10 BOTH VAL (CLU / ceiling) | "
          "zero-fit: unidentified n-way rows (TEST) | …also unreachable by colour relation |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ALL:
        if s not in S["BOTH-TEST"]:
            L.append("| %d | %s | NOT YET RUN |||||||||| " % (s, "added" if s > 2 else "original"))
            continue
        J = S["BOTH-TEST"][s][0]
        c, who = ceiling(J["ROWS"]["nway"])
        oc = ceiling(S["OBS-VAL"][s][0]["ROWS"]["nway"])[0] if s in S["OBS-VAL"] else None
        wc = ceiling(S["WORDS-FULL-TEST"][s][0]["ROWS"]["nway"])[0] if s in S["WORDS-FULL-TEST"] else None
        pr = S["CLU-REL-A10-BOTH-VAL"].get(s)
        prs = ("%s / %s" % (f(pr[0]["ROWS"]["nway"]["CLU"]), f(ceiling(pr[0]["ROWS"]["nway"])[0]))
               if pr else "not run")
        dg = [d for d in (diag or {}).get("runs", []) if d["seed"] == s and d["leg"] == "test"]
        ds = ("%d / %d (%.3f)" % (dg[0]["unreachable_eval_rows"], dg[0]["n_eval_nway_rows"],
                                  dg[0]["unreachable_row_fraction"]) if dg else "—")
        dr = ("%d rows" % dg[0]["unidentified_rows_ALSO_unreachable_by_colour_relation"]
              if dg and "unidentified_rows_ALSO_unreachable_by_colour_relation" in dg[0] else "—")
        L.append("| %d | %s | %s | %s | %s | %+.3f | **%s** | %s | %s | %s | %s | %s |"
                 % (s, "added" if s > 2 else "original", f(c), "; ".join(who),
                    f(J["ROWS"]["nway"]["CLU"]), J["ROWS"]["nway"]["CLU"] - c, cls.get(s, "?"),
                    f(oc), f(wc) if wc is not None else "— (not run on this seed)", prs, ds, dr))
    cmx = {s: ceiling(S["BOTH-TEST"][s][0]["ROWS"]["nway"], MATCHED_PREFIX)[0] for s in S["BOTH-TEST"]}
    L += ["", "Ceiling among MATCHED arms only (CLU + matched rivals, 8k/80k; excludes the unmatched "
          "10x-read-floats diagnostic): " + "; ".join("seed %d %s" % (s, f(v)) for s, v in sorted(cmx.items()))
          + ". Largest-gap classification under THIS ceiling: hard = %s."
          % (lambda xs: sorted(s for s, v in xs if v < ((lambda g: (xs[g][1] + xs[g + 1][1]) / 2)(
              max(range(len(xs) - 1), key=lambda i: xs[i + 1][1] - xs[i][1])))))(
              sorted(cmx.items(), key=lambda kv: kv[1])) if len(cmx) > 1 else "—"]
    L += ["", "Sensitivity — every possible cut between consecutive sorted ceilings:", "",
          "| cut between | gap | hard seeds |", "|---|---|---|"]
    for x in sens:
        L.append("| %.4f – %.4f | %.4f | %s |" % (x["cut_between"][0], x["cut_between"][1],
                                                  x["gap"], x["hard_seeds"]))
    # ---- main arm tables
    secs = [("1", "BOTH-TEST", "CMAZE-T · condition **BOTH** (words-plus-observation) · TEST "
             "(the winner row; seeds 0-2 = w19 scored runs, 3-5 = maze-t-seed4 scored runs 1-3/3)"),
            ("2", "OBS-VAL", "CMAZE-T · condition **OBS** · VAL (the comparable row; free)"),
            ("3", "WORDS-FULL-TEST", "CMAZE-T · condition **WORDS-FULL** · TEST (LOSS row; NO new "
             "seeds per AMENDMENT-1 → 3 seeds)"),
            ("4", "CLU-REL-A10-BOTH-VAL", "CMAZE-T · condition **BOTH** · VAL · CLU = peer's "
             "improved **CLU-REL-A10** (FREE VAL leg; rivals re-matched to its 9,200 read "
             "floats; '+rel' = rivals handed the CLU's relational columns, a diagnostic)")]
    for num, key, title in secs:
        runs = S[key]
        L += ["", "## %s. %s" % (num, title)]
        if not runs:
            L += ["", "NOT RUN / NOT AVAILABLE"]
            continue
        missing = [s for s in ALL if s not in runs]
        if missing:
            L += ["", "Seeds without a JSON (not run or not finished): %s" % missing]
        for row in ROWS:
            if all(row in runs[s][0]["ROWS"] and isinstance(runs[s][0]["ROWS"][row], dict)
                   and "CLU" in runs[s][0]["ROWS"][row] for s in runs):
                L += arm_table(title, runs, row, cls)
        L += training_table(title, runs)
        if any("store_scoped_exactness" in str(runs[s][0].get("DELETION", ""))[:10000]
               for s in runs):
            L += deletion_table(title, runs)
        L += rollout_table(title, runs)
    # ---- per-tuple / hard-seed diagnosis
    L += ["", "## 5. Zero-run diagnosis: which held-out n-way tuples are unreachable from the "
          "words, and why", "",
          "Zero-fit (`scripts/w21_seed_diagnosis.py`): banks + hold-out design only. Class A = the "
          "(agent,door,tint) triple is in train with the same outcome (only the direction is "
          "new); B = no triple but all three colour pairs present and agreeing; C = pairs present "
          "but conflicting (needs an unseen 3-way interaction); D = a colour pair has ZERO train "
          "rows. 'Unidentified' = C or D. 'Reachable by colour relation' = train shows the same "
          "colour-EQUALITY pattern (pocket==door / pocket==tint / door==tint, colour-free) with "
          "the same colour-free outcome under other colours.", ""]
    if diag:
        L += ["| seed | leg | tuples | tuples w/o eval rows | eval rows | unidentified tuples | "
              "unidentified rows (frac) | …of which unreachable by colour relation (rows) | "
              "held-out 2-way pairs |", "|---|---|---|---|---|---|---|---|---|"]
        for d in diag["runs"]:
            L.append("| %d | %s | %d | %d | %d | %d | %d (%.3f) | %s | %s |"
                     % (d["seed"], d["leg"].upper(), d["n_tuples"],
                        d["class_count_tuples"].get("NO-EVAL-ROWS", 0), d["n_eval_nway_rows"],
                        d["n_unreachable_tuples"], d["unreachable_eval_rows"],
                        d["unreachable_row_fraction"],
                        "%d tuples (%d rows)" % (
                            d["n_unidentified_tuples_ALSO_unreachable_by_colour_relation"],
                            d["unidentified_rows_ALSO_unreachable_by_colour_relation"]),
                        d["heldout_2way_pairs"]))
        L += ["", "### 5a. The list — every unidentified tuple, every seed (TEST)", "",
              "| seed | tuple | family | n eval | class | why |", "|---|---|---|---|---|---|"]
        for d in diag["runs"]:
            if d["leg"] != "test":
                continue
            for r in d["tuples"]:
                if r["class"][0] in "CD":
                    L.append("| %d | (%s) | %s | %d | %s | %s |"
                             % (d["seed"], ",".join(str(x) for x in r["tuple"]),
                                ",".join(r["family"]), r["n_eval"], r["class"], r["why"]))
    else:
        L += ["diagnosis JSON not available"]
    L += ["", "### 5b. Per-tuple scores and per-tuple ceiling (BOTH TEST), every seed", ""]
    for s in ALL:
        if s in S["BOTH-TEST"]:
            L += tuple_table(out, "BOTH-TEST", S, diag, s, cls)
    with open(os.path.join(out, "TABLES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    M = {"caption": CAPTION, "ceiling_definition": CEIL_PREFIX, "classification": {
        "cut": cutinfo, "class": cls, "ceilings_BOTH_TEST_nway": ce, "sensitivity": sens},
        "sources": {t: {s: v[1] for s, v in d.items()} for t, d in S.items()},
        "nway": {t: {s: {"CLU": v[0]["ROWS"]["nway"]["CLU"],
                         "ceiling": ceiling(v[0]["ROWS"]["nway"])[0],
                         "ceiling_arms": ceiling(v[0]["ROWS"]["nway"])[1],
                         "all_arms": {k: x for k, x in v[0]["ROWS"]["nway"].items()
                                      if isinstance(x, (int, float))}}
                     for s, v in d.items()} for t, d in S.items()},
        "diagnosis": "diagnosis/SEED-DIAGNOSIS-v2.json"}
    with open(os.path.join(out, "MAZE-T-SEED4.json"), "w") as fh:
        json.dump(M, fh, indent=1)
    print("wrote TABLES.md + MAZE-T-SEED4.json; BOTH-TEST seeds %s OBS-VAL %s WF %s REL %s; class %s"
          % (sorted(S["BOTH-TEST"]), sorted(S["OBS-VAL"]), sorted(S["WORDS-FULL-TEST"]),
             sorted(S["CLU-REL-A10-BOTH-VAL"]), cls))


if __name__ == "__main__":
    main(sys.argv[1])
