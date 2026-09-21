"""w12 hub/composition-ladder -- LADDER.md, rendered from LADDER.json + BYTES.json only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNGS = ["2", "3", "4", "5", "6"]
STORE = ["single_design_K5", "pca_tensor"]
CONTROLS = ["landscape_shuffled", "random_atoms"]
LEARNED = ["rival_mlp", "rival_lstm", "rival_transformer", "rival_cont_mlp", "rival_table"]
OWNPOOL = ["learned_ref_ownpool", "learned_1x"]
ORDER = (STORE + ["%s__%s" % (s, c) for s in STORE for c in CONTROLS]
         + LEARNED + OWNPOOL)
LABEL = {
    "single_design_K5": "**single design** (K=5, DECLARED)",
    "pca_tensor": "**pca_tensor** (wave-5 subspace store)",
    "single_design_K5__landscape_shuffled": "&nbsp;&nbsp;· control `landscape_shuffled`",
    "single_design_K5__random_atoms": "&nbsp;&nbsp;· control `random_atoms`",
    "pca_tensor__landscape_shuffled": "&nbsp;&nbsp;· control `landscape_shuffled`",
    "pca_tensor__random_atoms": "&nbsp;&nbsp;· control `random_atoms`",
    "rival_mlp": "MLP @ 524,288 B ref", "rival_lstm": "LSTM @ 524,288 B ref",
    "rival_transformer": "transformer @ 524,288 B ref",
    "rival_cont_mlp": "continuous MLP @ 524,288 B ref",
    "rival_table": "table FLOOR @ 524,288 B ref",
    "learned_ref_ownpool": "MLP @ 524,288 B ref, **own pool** (AMENDMENT-1)",
    "learned_1x": "MLP @ **1x** the single design's own total (AMENDMENT-1)",
}


def cell(A, key="across_seed_mean", per=None):
    if A.get("status") != "RUN":
        return "**DECLARED NOT-RUN** — " + A.get("reason", "no reason recorded")
    v = A[key]
    seeds = sorted(A["per_seed"])
    s = " / ".join("%.3f" % A["per_seed"][x]["mean"] for x in seeds)
    return "%.3f (%s)" % (v, s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--bytes", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = json.load(open(a.ladder, encoding="utf-8"))
    by = json.load(open(a.bytes, encoding="utf-8")) if a.bytes and Path(a.bytes).exists() else None
    L = []
    W = L.append

    W("# w12 hub/composition-ladder — THE LADDER TABLE")
    W("")
    W("Every number below is `distinct_valid_novel_tuples` per corner, the metric and the "
      "margin the harness pre-declared (`LADDER-READY.json:estimand`). Unit = **corner**. "
      "Cells read `mean over seeds (seed0 / seed1 / seed2)`.")
    W("")
    W("⛔ **Seed count and configuration count are separate numbers and are never merged.** "
      "`n_seeds = %d` (seeds %s). Configurations per rung = the rung's corner count, "
      "printed in the header of each column."
      % (rep["n_seeds_declared"], rep["seeds"]))
    W("")
    hdr = []
    for r in RUNGS:
        R = rep["rungs"].get(r, {})
        n = sorted(set((R.get("n_corners_per_seed") or {}).values()))
        hdr.append("k=%s<br><sub>%s corners · ceiling %s</sub>"
                   % (r, n or "—", R.get("ceiling_per_corner") or "—"))
    R = rep["rungs"].get("shipped", {})
    n = sorted(set((R.get("n_corners_per_seed") or {}).values()))
    hdr.append("shipped<br><sub>%s corners (row3+row4)</sub>" % (n or "—"))

    W("## 1. The ladder — score per corner")
    W("")
    W("| arm | " + " | ".join(hdr) + " |")
    W("|---|" + "---|" * (len(RUNGS) + 1))
    for arm in ORDER:
        row = [LABEL.get(arm, arm)]
        for r in RUNGS + ["shipped"]:
            row.append(cell(rep["rungs"].get(r, {}).get("arms", {}).get(arm,
                            {"status": "DECLARED NOT-RUN", "reason": "rung absent"})))
        W("| " + " | ".join(row) + " |")
    W("")
    W("⚠ **The rungs are NOT on one absolute scale.** The metric's ceiling is the corner's "
      "own row count: 36 at k=2 and 72 at k≥3. A k=2 number and a k=3 number are not "
      "comparable as absolutes; the ceiling is printed on every column for that reason.")
    W("")

    W("## 2. rank / width on every store row (wave 9's saturation trap)")
    W("")
    W("| arm | " + " | ".join("k=%s" % r for r in RUNGS) + " | shipped |")
    W("|---|" + "---|" * (len(RUNGS) + 1))
    for arm in STORE + ["%s__%s" % (s, c) for s in STORE for c in CONTROLS]:
        row = [LABEL.get(arm, arm)]
        for r in RUNGS + ["shipped"]:
            A = rep["rungs"].get(r, {}).get("arms", {}).get(arm, {})
            if A.get("status") != "RUN":
                row.append("—")
                continue
            vals = [(A["per_seed"][s].get("rank_width") or {}).get("rank_over_width_max")
                    for s in sorted(A["per_seed"])]
            vals = [v for v in vals if v is not None]
            row.append(" / ".join("%.3f" % v for v in vals) if vals else "—")
        W("| " + " | ".join(row) + " |")
    W("")
    W("> wave 9 measured its maze arm DECORATIVE at `rank/d_k = 0.938`. Any cell at or above "
      "that is in the saturation regime and is flagged in FINDINGS.")
    W("")

    W("## 3. ⛔ THE CONTROLS — the store is DECORATIVE at any rung where they do not collapse")
    W("")
    W("| store arm | control | " + " | ".join("k=%s" % r for r in RUNGS) + " | shipped |")
    W("|---|---|" + "---|" * (len(RUNGS) + 1))
    for base in STORE:
        for ctl in CONTROLS:
            row = [LABEL.get(base, base), "`%s`" % ctl]
            for r in RUNGS + ["shipped"]:
                v = rep.get("control_verdicts", {}).get(r, {}).get(base, {}).get(ctl, {})
                if v.get("status") != "RUN":
                    row.append("**NOT-RUN**")
                    continue
                d = " / ".join("%+.2f" % p["mean_paired_delta"] for p in v["per_seed"])
                row.append(("collapses<br><sub>Δ %s</sub>" % d)
                           if v["control_collapses_on_every_seed"]
                           else ("⛔ **DOES NOT COLLAPSE**<br><sub>Δ %s</sub>" % d))
            W("| " + " | ".join(row) + " |")
    W("")
    W("| store arm | " + " | ".join("k=%s" % r for r in RUNGS) + " | shipped |")
    W("|---|" + "---|" * (len(RUNGS) + 1))
    for base in STORE:
        row = [LABEL.get(base, base)]
        for r in RUNGS + ["shipped"]:
            row.append(rep.get("control_verdicts", {}).get(r, {}).get(base, {})
                       .get("VERDICT", "—"))
        W("| " + " | ".join(row) + " |")
    W("")

    W("## 4. Paired margin against the BEST learned baseline (PREREG §5 rule)")
    W("")
    W("The rule was fixed at 2026-09-21T11:38:41Z, before any arm number existed: per "
      "(rung, seed) the learned arm with the **largest** mean, among the five pre-existing "
      "learned arms. The winner's identity is printed.")
    W("")
    W("| rung | best learned baseline (seed0 / seed1 / seed2) | " +
      " | ".join("Δ %s" % LABEL.get(s, s).replace("**", "") for s in STORE) + " |")
    W("|---|---|" + "---|" * len(STORE))
    for r in RUNGS + ["shipped"]:
        R = rep["rungs"].get(r, {})
        bl = R.get("best_learned_per_seed") or {}
        who = " / ".join("%s (%.3f)" % (bl[s]["arm"], bl[s]["mean"])
                         for s in sorted(bl) if bl[s]["arm"]) or "—"
        row = ["k=%s" % r if r != "shipped" else "shipped", who]
        for arm in STORE:
            A = R.get("arms", {}).get(arm, {})
            if A.get("status") != "RUN":
                row.append("—")
                continue
            ds = [A["per_seed"][s].get("paired_vs_best_learned") for s in sorted(A["per_seed"])]
            ds = [d for d in ds if d]
            row.append(" / ".join("%+.2f [%+.2f, %+.2f]"
                                  % (d["mean_paired_delta"], d["bootstrap_ci95"][0],
                                     d["bootstrap_ci95"][1]) for d in ds) or "—")
        W("| " + " | ".join(row) + " |")
    W("")

    W("## 5. ⛔ END REPRODUCTION — the non-negotiable")
    W("")
    for tag, title in (("k2_vs_shipped_row3", "k = 2 vs the shipped ROW 3"),
                       ("k6_vs_shipped_row4", "k = 6 (first 49 corners) vs the shipped ROW 4")):
        E = rep.get("end_reproduction", {}).get(tag, {})
        W("### %s — **%s**" % (title, E.get("VERDICT", "—")))
        W("")
        W("| arm | rung mean (s0/s1/s2) | shipped mean (s0/s1/s2) | |Δ| | within the shipped "
          "corner sd on every seed |")
        W("|---|---|---|---|---|")
        for arm in ORDER:
            A = E.get("per_arm", {}).get(arm, {})
            if A.get("status") != "RUN":
                W("| %s | — | — | — | **NOT-RUN**: %s |"
                  % (LABEL.get(arm, arm), A.get("reason", "absent")))
                continue
            W("| %s | %s | %s | %.3f | %s |" % (
                LABEL.get(arm, arm),
                " / ".join("%.3f" % x["rung_mean"] for x in A["per_seed"]),
                " / ".join("%.3f" % x["shipped_mean"] for x in A["per_seed"]),
                A["max_abs_diff"],
                "yes" if A["reproduces_on_every_seed"] else "⛔ **no**"))
        W("")

    if by:
        W("## 6. The byte ledgers — measured on the objects this hub fitted, never netted")
        W("")
        W("| arm | atoms | sufficient statistics | address book | total | × 524,288 B |")
        W("|---|---|---|---|---|---|")
        for arm, rows in (by.get("per_arm") or {}).items():
            for rr in rows[:1]:
                W("| %s (leg `%s`) | %s B | %s B | %s | %s B | %.2f× |" % (
                    arm, rr["leg"], "{:,}".format(rr["p1_atoms"]),
                    "{:,}".format(rr["p2_statistics"]),
                    ("{:,} B".format(rr["p3_address_book"])
                     if rr["p3_address_book"] is not None else rr["p3_status"]),
                    "{:,}".format(rr["total_bytes"]), rr["times_reference_pin"]))
        s = by.get("summary") or {}
        for arm, v in s.items():
            W("")
            W("> `%s` across %d legs: total **%s – %s B** = **%.2f× – %.2f×** the "
              "524,288 B reference pin." % (arm, v["n_legs"],
                                            "{:,}".format(v["total_bytes_min"]),
                                            "{:,}".format(v["total_bytes_max"]),
                                            v["times_reference_pin_min"],
                                            v["times_reference_pin_max"]))
        W("")

    W("## 7. ⛔ DECLARED NOT-RUN")
    W("")
    nr = rep.get("not_run") or []
    if not nr:
        W("None — every (rung, seed, arm) cell in the matrix was scored.")
    else:
        agg = {}
        for x in nr:
            agg.setdefault((x["arm"], x["reason"]), []).append("%s:s%d" % (x["rung"], x["seed"]))
        W("| arm | reason | cells |")
        W("|---|---|---|")
        for (arm, why), cells in sorted(agg.items()):
            W("| `%s` | %s | %d — %s |" % (arm, why, len(cells), ", ".join(cells)))
    W("")
    p = Path(a.out) / "LADDER.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
