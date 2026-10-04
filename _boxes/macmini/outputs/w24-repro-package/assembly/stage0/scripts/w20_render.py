"""Render SHAPES-MATCHED.md from SHAPES-MATCHED.json (+ deletion_store/, lacepoe/). No new numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w20_lib import FAMILIES, OUT, RUNGS  # noqa: E402

SEEDS = (0, 1, 2)


def f3(x):
    return "—" if x is None else "%.3f" % x


def cell(e, key="all"):
    if not e or not e.get(key):
        return "—", "—", 0
    v = e[key]
    ps = " / ".join("%.3f" % x for x in v["per_seed"])
    sd = ("±%.3f" % v["sd"]) if v.get("sd") is not None else ""
    return "%.3f %s" % (v["mean"], sd), ps, v["n_seeds"]


ROLE = {
    "store_K32_n131072": ("store K*=32, fit 131,072 rows", "THE ARM"),
    "store_K5_n131072": ("store K=5, fit 131,072 rows", "THE ARM (K=5)"),
    "store_K32_n16384": ("store K*=32, fit 16,384 rows", "M-16k arm"),
    "store_K5_n16384": ("store K=5, fit 16,384 rows", "M-16k arm"),
    "store_K32_n32768": ("store K*=32, fit 32,768 rows", "M-16k arm"),
    "store_K5_n32768": ("store K=5, fit 32,768 rows", "M-16k arm"),
    "table_FLOOR": ("prototype table", "FLOOR, never a rival"),
    "EXISTING_learned_ref_ownpool@130514_32768rows_2560steps":
        ("MLP 130,514 p, 32,768 rows, 20 ep lr1e-3 (existing learned_ref_ownpool)",
         "EXISTING data-matched-at-32,768 rival — NOT 131,072"),
    "mlp__declared@test16k_stepmatched": ("MLP, 16,384 rows, 20,480 steps (A1.1)", "DIAGNOSTIC"),
    "mlp__declared@test32k_stepmatched": ("MLP, 32,768 rows, 20,480 steps (A2.3)", "DIAGNOSTIC"),
}
for f in FAMILIES:
    ROLE["%s__declared@test" % f] = ("%s, 131,072 rows, DECLARED (spoke-A protocol)" % f,
                                      "DATA-MATCHED RIVAL")
    ROLE["%s__tenx@test" % f] = ("%s, 131,072 rows, 10x steps" % f, "10x DIAGNOSTIC")
    ROLE["w12comp_%s__as_trained" % f] = ("%s, 16,384 rows, 20 ep (T7's comparator)" % f,
                                          "EXISTING 16k rival (draft)")


def main():
    d = json.loads((OUT / "SHAPES-MATCHED.json").read_text(encoding="utf-8"))
    L = ["# SHAPES-MATCHED — the one-to-one rival tables (w20 hub/shapes-data-matched)", "",
         "Every number is read from `SHAPES-MATCHED.json`, which reads only per-corner score files "
         "(`scores/…`, paths in the JSON's `sources`). Metric: distinct valid novel tuples per corner "
         "(the corner is the unit). Seeds 0/1/2; the seeds column is printed on every row. "
         "Margin = 1.0 novel tuple/corner (pre-declared).", ""]
    r = d["rows34"]
    L += ["## Table M-131k (and M-16k) — rows 3–4 TEST (60 pair + 49 n-way corners)", "",
          "**Fit pools:** store 131,072 rows (its exact pool) · data-matched rivals 131,072 rows (the SAME "
          "pool) · T7's comparators 16,384 rows (= the pool's first 16,384) · `learned_ref_ownpool` "
          "32,768 rows (= the pool's first 32,768) · M-16k stores 16,384 / 32,768 rows (nested prefixes). "
          "Rival parameters: 130,433–131,891 (the 131,072 read-path pin); the store's four-part ledger is "
          "printed separately and never matched. Ceiling = distinct requested tuples the split never holds.",
          "",
          "| arm | role | all corners mean ± sd | per seed | row 3 (pair) | row 4 (n-way) | ceiling (all) | n seeds |",
          "|---|---|---|---|---|---|---|---|"]
    arms = r["arms"]
    order = [k for k in ROLE if k in arms] + sorted(k for k in arms if k not in ROLE)
    for k in order:
        if "__landscape" in k or "__random_atoms" in k:
            continue
        a, ps, n = cell(arms[k])
        r3, _, _ = cell(arms[k], "row3_pair")
        r4, _, _ = cell(arms[k], "row4_nway")
        ce = arms[k].get("ceiling_all")
        L.append("| %s | %s | %s | %s | %s | %s | %s | %d |" % (
            ROLE.get(k, (k, ""))[0], ROLE.get(k, ("", k))[1], a, ps, r3, r4,
            f3(ce["mean"]) if ce else "—", n))
    L += ["", "### Controls (P2) — must read ≤ 1.0 on every seed", "",
          "| store | landscape_removed | landscape_shuffled | across_wells (DERANGEMENT) | random_atoms | all collapse |",
          "|---|---|---|---|---|---|"]
    for st, g in r["P2_controls"].items():
        def c(x):
            v = g.get(x, {})
            return " / ".join("%.3f" % y for y in v.get("per_seed", [])) or "—"
        L.append("| %s | %s | %s | %s | %s | %s |" % (st, c("landscape_removed"), c("landscape_shuffled"),
                                                   c("landscape_shuffled_across_wells"), c("random_atoms"),
                                                   "YES" if g.get("ALL_COLLAPSE") else "NO"))
    for st in ("store_K32_n131072",):
        e = arms.get("%s__landscape_shuffled_across_wells" % st, {})
        for i, p in enumerate(e.get("derangement_printed", [])):
            L.append("")
            L.append("seed %d derangement perm = `%s` (fixed points %d; the historical T7 permutation "
                     "had %d fixed point(s))" % (i, p["perm"], p["fixed_points"],
                                                 p["historical_T7_perm_fixed_points"]))
    L += ["", "### P1 — convergence (DECLARED vs 10× steps), per family per seed", "",
          "| family | seed | DECLARED | TENX | TENX − DECLARED | P1 | carried |", "|---|---|---|---|---|---|---|"]
    for f, per in r["P1_convergence"].items():
        for s, v in sorted(per.items()):
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (f, s, f3(v.get("declared")), f3(v.get("tenx")),
                                                           f3(v.get("tenx_minus_declared")), v["P1"], v["carried"]))
    L += ["", "### P3 — THE DATA-MATCHED MARGIN (store − strongest data-matched rival, paired per corner)", "",
          "| store | seed | strongest rival | store | rival | Δ | CI95 | won/tied/lost |", "|---|---|---|---|---|---|---|---|"]
    for st, blk in r["P3_margin"].items():
        for s, v in sorted(blk.get("vs_strongest", {}).items()):
            L.append("| %s | %s | %s | %.3f | %.3f | **%+.3f** | [%.2f, %.2f] | %s |" % (
                st, s, v["strongest_rival (max of declared/tenx)"], v["store"], v["rival"], v["delta"],
                v["ci95"][0], v["ci95"][1], "/".join(map(str, v["won_tied_lost"]))))
        if "vs_strongest_mean" in blk:
            L.append("| %s | **mean** | | | | **%+.3f** | | verdict **%s** (draft: +39…+49) |" % (
                st, blk["vs_strongest_mean"], blk["verdict"]))
    L += ["", "### Margins against every rival column (paired Δ per seed)", "",
          "| store | rival | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|---|"]
    for st, blk in r["P3_margin"].items():
        for f, pf in blk.get("per_family", {}).items():
            vals = [("%+.3f" % pf[s]["delta"]) if s in pf else "—" for s in ("0", "1", "2")]
            L.append("| %s | %s (data-matched, carried) | %s |" % (st, f, " | ".join(vals)))
        for er, pe in blk.get("vs_existing_rivals", {}).items():
            vals = [("%+.3f" % pe[s]["delta"]) if s in pe else "—" for s in ("0", "1", "2")]
            L.append("| %s | %s | %s |" % (st, er, " | ".join(vals)))
    ba = r.get("P3_vs_best_over_all_fits_incl_30x") or {}
    if ba:
        L += ["", "### P3 against the rival at its best over EVERY fit that ran (DECLARED, 10×, 30×)", "",
              "| seed | best rival fit | store | rival | Δ | CI95 | won/tied/lost |", "|---|---|---|---|---|---|---|"]
        for k_, v in sorted((k, v) for k, v in ba.items() if k not in ("mean", "verdict")):
            L.append("| %s | %s | %.3f | %.3f | **%+.3f** | [%.2f, %.2f] | %s |" % (
                k_, v["best_rival_over_all_fits"], v["store"], v["rival"], v["delta"], v["ci95"][0],
                v["ci95"][1], "/".join(map(str, v["won_tied_lost"]))))
        L.append("| **mean** | | | | **%+.3f** | | **%s** |" % (ba["mean"], ba["verdict"]))
    dr = r.get("draft_margin_recomputed_vs_16k_comparators") or {}
    if dr:
        L += ["", "### The draft's margin, recomputed from the same score files (store K*=32 − 16k comparator)", "",
              "| comparator (16,384 rows) | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|"]
        for f, pv in dr.items():
            L.append("| %s | %s |" % (f, " | ".join(("%+.3f" % pv[s_]) if s_ in pv else "—" for s_ in ("0", "1", "2"))))
    # deletion
    de = d.get("deletion_matched")
    if de:
        L += ["", "## Matched deletion (Head AMENDMENT-2 definition; rows 3–4 TEST survivor corners; pool 131,072)", "",
              "⛔ " + de["RETIRED"] + ".", "",
              "| scope | seed | deleted rows | (a) store max|mode dev| | (b1) statistic dev | (b1) statistic bytes | (b2) rendered rel dev | (c) DISPLAY refusal (deleted / false on survivors) | (d) survivors: before → after (Δ) | (d) downdate − refit-without |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sc, blk in de["per_scope"].items():
            for sd, v in sorted(blk["store"].items()):
                c = v["c_DISPLAY"]
                cs = ("N/A (no hole exists)" if "DECLARED_NA" in c else "%s / %s" % (
                    f3(c.get("refusal_rate_on_deleted_corner_requests")), f3(c.get("false_refusal_rate_on_survivor_requests"))))
                dd = v["d"]
                L.append("| %s | %s | %d | %.2e | %s | %.2f GB | %.2e | %s | %.3f → %.3f (%+.3f) | %+.3f |" % (
                    sc, sd, v["n_deleted"], v["a"]["max_abs_mode_dev"],
                    "%.1e" % v["b"]["b1_statistic"]["max_abs_first_moment_dev"],
                    v["b"]["b1_statistic"]["statistic_bytes"] / 1e9,
                    v["b"]["b2_book"]["rendered_features_rel_dev_on_survivor_requests"], cs,
                    dd["before_mean"], dd["after_mean"], dd["paired_after_minus_before"]["mean_paired_delta"],
                    dd["downdate_minus_refit_without_pipeline"]["mean_paired_delta"]))
        L += ["", "| scope | seed | family | rival as-trained → retrain-without (survivors) | ours after | ours − rival retrain-without | CI95 | rival s | rival rows read | rival bytes rewritten |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sc, blk in de["per_scope"].items():
            for f, per in blk["rivals"].items():
                for sd, v in sorted(per.items()):
                    cl = v.get("rival_cost_line") or {}
                    L.append("| %s | %s | %s | %.3f → %.3f | %.3f | **%+.3f** | [%.2f, %.2f] | %s | %s | %s |" % (
                        sc, sd, f, v["rival_as_trained_on_survivors"], v["rival_retrain_without_on_survivors"],
                        v["ours_after"], v["ours_minus_rival_retrain_without"], v["ci95"][0], v["ci95"][1],
                        cl.get("seconds", "—"), cl.get("data_rows_read", "—"), cl.get("bytes_rewritten", "—")))
    l1 = d.get("LACE_PoE_at_131072")
    if l1 and l1.get("legs"):
        L += ["", "### LACE / PoE re-fit at 131,072 rows (Head AMENDMENT-1 i)", "",
              "⛔ Scope, verbatim: **%s**. Fit rows: %s." % (l1["scoping_VERBATIM"], l1["fit_rows"]), "",
              "| leg | arm | per seed (novel/corner) |", "|---|---|---|"]
        for leg, arms_ in l1["legs"].items():
            for arm, per in arms_.items():
                L.append("| %s | %s | %s |" % (leg, arm, " / ".join("s%s %s" % (k, f3(v["novel_per_corner"])) for k, v in sorted(per.items()))))
    # LACE/PoE
    lp = d.get("LACE_PoE_from_w15")
    if lp:
        L += ["", "### LACE / PoE", "", "⛔ Scope, verbatim: **%s**." % lp["scoping_VERBATIM"], ""]
        L.append("w15 (32,256 fit rows): " + json.dumps({k: v.get("per_seed") for k, v in lp["rows34"].items()}))
    L += ["", "## The ladder (free replication; rung pools 131,072 rows)", ""]
    for k, blk in d.get("ladder", {}).items():
        L += ["### rung n=%s" % k, "",
              "| arm | mean ± sd | per seed | n seeds |", "|---|---|---|---|"]
        for a_, e in sorted(blk["arms"].items()):
            if "__landscape" in a_ or "__random_atoms" in a_:
                continue
            m_, ps, n = cell(e)
            L.append("| %s | %s | %s | %d |" % (a_, m_, ps, n))
        for st, b in blk["P3_margin"].items():
            if "vs_strongest_mean" in b:
                ex = b.get("vs_existing_rivals", {})
                sa = [k_ for k_ in ex if k_.startswith("EXISTING_spokeA_")]
                draft = []
                for sd in ("0", "1", "2"):
                    v_ = [ex[k_][sd]["delta"] for k_ in sa if sd in ex[k_]]
                    draft.append(("%+.3f" % min(v_)) if v_ else "—")
                L.append("")
                L.append("%s − strongest DATA-MATCHED rival (131,072 rung rows): per seed %s, mean **%+.3f** · "
                         "draft-style margin (− strongest spoke-A `@ref` rival, 32,256 Montero rows): %s" % (
                    st, ["%+.3f" % v["delta"] for _, v in sorted(b["vs_strongest"].items())],
                    b["vs_strongest_mean"], draft))
        L.append("")
    (OUT / "SHAPES-MATCHED.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT / "SHAPES-MATCHED.md")


if __name__ == "__main__":
    main()
