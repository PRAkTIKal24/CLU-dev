"""w21 spoke/words-full-test -- the Head's WIN / COMPARABLE / LOSS label, TABLES.md and
WORDS-FULL-TEST.json, all generated from the per-seed JSONs.

    python scripts/w21_report.py <out_dir>

Reads ``<out>/test/WORDS-FULL_test_seed{0,1,2}.json`` and ``<out>/test/BARS-WORDS-FULL-test.json``
(the latter written by the UNCHANGED ``scripts/w19_bars.py``), adds ``HEAD_LABEL`` to the BARS
file (the w19 B1..B4 / ``label`` fields stay as the w19 record and are marked superseded), and
writes TABLES.md + WORDS-FULL-TEST.json.  The VAL leg is read from w19's record
(``/c/chlu/outputs/w19-maze-t-symmetric/val``) and printed beside TEST under the same rule.
Label rule: PREREG-w21-words-full-test.md.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w19_tables as WT                                              # noqa: E402

COND = "WORDS-FULL"
SEEDS = (0, 1, 2)
W19_VAL = os.environ.get("W19_VAL_DIR", "C:/chlu/outputs/w19-maze-t-symmetric/val")
f = WT.f


def load(d, leg):
    return [json.load(open(os.path.join(d, "%s_%s_seed%d.json" % (COND, leg, s))))
            for s in SEEDS]


def head_label(runs, row):
    """PREREG-w21: R* over every matched kind x both budgets; sigma = sd(ddof 1) of the
    per-seed paired difference; WIN / COMPARABLE / LOSS."""
    B = [r["ROWS"][row] for r in runs]
    matched = [k for k in B[0] if k.startswith("rival: ") or k.startswith("rival@10x-steps: ")]
    per = {k: np.array([float(b[k]) for b in B]) for k in matched + ["CLU"]}
    ctl = {k: float(np.mean([b[k] for b in B])) for k in B[0] if k.startswith("control: ")}
    diag = {k: [float(b[k]) for b in B] for k in B[0] if k.startswith("diagnostic@")}
    clu = per["CLU"]
    rstar = max(matched, key=lambda k: per[k].mean())
    d = clu - per[rstar]
    dbar, sig = float(d.mean()), float(d.std(ddof=1))
    n_ahead = int((d > 0).sum())
    gate = all(v <= clu.mean() - 0.15 for v in ctl.values())
    if dbar > 0 and n_ahead >= 2 and gate:
        lab, why = "WIN", "ahead on the mean and on %d/3 seeds; store gate holds" % n_ahead
    elif abs(dbar) <= sig:
        lab, why = "COMPARABLE", "|mean diff| %.4f <= one seed's spread %.4f" % (abs(dbar), sig)
    elif dbar < 0:
        lab, why = "LOSS", "behind by %.4f > one seed's spread %.4f" % (-dbar, sig)
    else:
        fail = [x for x, ok in (("majority-of-seeds", n_ahead >= 2), ("store gate", gate))
                if not ok]
        lab, why = "COMPARABLE", "ahead beyond the spread but %s fails" % " and ".join(fail)
    best_seed = np.max(np.stack([per[k] for k in matched]), axis=0)
    best_seed_arm = [max(matched, key=lambda k: per[k][i]) for i in range(len(runs))]
    return {"row": row, "n_seeds": len(runs), "n_rows_per_seed": [b["n"] for b in B],
            "CLU_per_seed": clu.tolist(), "CLU_mean": float(clu.mean()),
            "R_star": rstar, "R_star_per_seed": per[rstar].tolist(),
            "R_star_mean": float(per[rstar].mean()),
            "paired_diff_per_seed": d.tolist(), "mean_diff": dbar,
            "one_seed_spread_sd_paired_diff": sig, "seeds_CLU_ahead": n_ahead,
            "store_gate_controls_le_CLU_minus_0.15": bool(gate), "controls_mean": ctl,
            "LABEL": lab, "why": why,
            "printed_never_the_label": {
                "sd_CLU_per_seed": float(clu.std(ddof=1)),
                "sd_R_star_per_seed": float(per[rstar].std(ddof=1)),
                "best_matched_rival_per_seed": best_seed.tolist(),
                "best_matched_rival_per_seed_arm": best_seed_arm,
                "CLU_minus_best_per_seed": (clu - best_seed).tolist(),
                "diagnostic_10x_read_floats": diag},
            "matched_means": {k: float(per[k].mean()) for k in matched}}


def label_table(title, labs):
    L = ["", title, "",
         "| leg | row | n_seeds | CLU (s0 / s1 / s2) | CLU mean | R* (arm) | R* (s0 / s1 / s2) | R* mean | "
         "Δ per seed | Δ̄ | one seed's spread σ | seeds CLU ahead | store gate | LABEL |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for leg, row, h in labs:
        L.append("| %s | `%s` | %d | %s | %s | %s | %s | %s | %s | %+.4f | %.4f | %d/3 | %s | **%s** |"
                 % (leg, row, h["n_seeds"], " / ".join("%.4f" % x for x in h["CLU_per_seed"]),
                    "%.4f" % h["CLU_mean"], h["R_star"],
                    " / ".join("%.4f" % x for x in h["R_star_per_seed"]), "%.4f" % h["R_star_mean"],
                    " / ".join("%+.4f" % x for x in h["paired_diff_per_seed"]), h["mean_diff"],
                    h["one_seed_spread_sd_paired_diff"], h["seeds_CLU_ahead"],
                    "holds" if h["store_gate_controls_le_CLU_minus_0.15"] else "FAILS", h["LABEL"]))
    return L


def main(out):
    tdir = os.path.join(out, "test")
    test = load(tdir, "test")
    val = load(W19_VAL, "val")
    bp = os.path.join(tdir, "BARS-%s-test.json" % COND)
    bars = json.load(open(bp))
    rows = ("nway", "seen", "pair", "value")
    HT = {row: head_label(test, row) for row in rows}
    HV = {row: head_label(val, row) for row in rows}
    bars["HEAD_LABEL"] = {"rule": "PREREG-w21-words-full-test.md (the Head's WIN/COMPARABLE/LOSS; "
                                  "no numeric margin bar)",
                          "primary_row": "nway", "PRIMARY": HT["nway"]["LABEL"],
                          "rows": HT}
    bars["w19_prereg_fields_SUPERSEDED"] = ("rows.*.B1..B4 / label / PRIMARY_nway are the w19 "
                                            "PREREG margin-bar record, printed only; HEAD_LABEL "
                                            "is the label")
    json.dump(bars, open(bp, "w"), indent=1)

    r0 = test[0]
    L = ["# w21 words-full-test — TABLES (generated by scripts/w21_report.py)", "",
         "⛔ CMAZE-T only, condition **WORDS-FULL** only (the 318-wide `[bias|main|pair]` design to "
         "EVERY arm). No wave-16 row; no other condition. n_seeds = 3 (0, 1, 2) on every row. "
         "TEST = runs 4–6/9 on target `maze-t-symmetric` (this wave); VAL = w19's record "
         "(`%s`), reproduced bit-for-bit for the CLU + controls (`repro/REPRO_COMPARE.txt`)." % W19_VAL,
         ""]
    L += label_table("## 1. The label (Head's rule, PREREG-w21) — primary row `nway`, TEST "
                     "(and VAL for reference)",
                     [("TEST", "nway", HT["nway"]), ("VAL (w19)", "nway", HV["nway"])])
    h = HT["nway"]["printed_never_the_label"]
    L += ["", "Printed, never the label (TEST nway): σ from the CLU's per-seed spread %.4f; from "
          "R*'s %.4f. Per-seed best matched rival %s (%s); CLU − best per seed %s. 10×-read-floats "
          "diagnostic (8,000 steps; not matched, never a bar): %s."
          % (h["sd_CLU_per_seed"], h["sd_R_star_per_seed"],
             " / ".join("%.4f" % x for x in h["best_matched_rival_per_seed"]),
             ", ".join(h["best_matched_rival_per_seed_arm"]),
             " / ".join("%+.4f" % x for x in h["CLU_minus_best_per_seed"]),
             "; ".join("%s %s" % (k, " / ".join("%.4f" % x for x in v))
                       for k, v in h["diagnostic_10x_read_floats"].items())), ""]
    L += label_table("## 2. Other rows (same rule, secondary; seen = sanity, pair = blind "
                     "falsifier row, value = extrapolation)",
                     [("TEST", r, HT[r]) for r in rows[1:]] + [("VAL (w19)", r, HV[r])
                                                               for r in rows[1:]])
    # matched parameters: width beside float count
    L += ["", "## 3. Matched rivals — hidden width beside matched read-path float count "
          "(target = CLU read floats %d; convention: trainable float count)"
          % r0["matched_target"]["target_floats"], "",
          "| rival | input dim | hidden width | matched floats | residual vs target | "
          "10×-read-floats diag width | diag floats | seeds |", "|---|---|---|---|---|---|---|---|"]
    for k, v in r0["matched_parameters"].items():
        same = all(r["matched_parameters"][k] == v for r in test)
        cd = r0["capacity_diagnostic"][k]
        L.append("| %s | %d | %d | %d | %+.2f%% | %d | %d | 3 (%s) |"
                 % (k, v["input_dim"], v["width"], v["params"], v["residual_pct"], cd["width"],
                    cd["params"], "identical on all seeds" if same else "PER-SEED DIFFERS"))
    led = r0["clu_ledger"]
    L += ["", "CLU A1 read path: %d floats (atoms %d + address book %d) → %d B deployed "
          "(convention: 4 × (atom + address-book floats), float32); float64 sufficient statistic "
          "%d B printed beside, never netted, never matched."
          % (led["read_floats"], led["store_atom_floats"], led["head_floats"],
             led["deployed_bytes_float32"], led["statistic_bytes_float64"])]
    # per-step + rollout
    L += ["", "## 4. Per-step exactness, mean first-error step, rollout validity (TEST)", "",
          "| arm | n_seeds | per-step exactness (s0 / s1 / s2) | mean | mean first-error step "
          "(s0 / s1 / s2) | rollout validity (s0 / s1 / s2) |", "|---|---|---|---|---|---|"]
    arms = test[0]["ROLLOUT"]["arms"]
    rk = [k for k in arms if k.startswith("rival")]
    best = max(rk, key=lambda k: np.mean([r["ROLLOUT"]["arms"][k]["per_step_exactness"]
                                          for r in test]))
    for k in ["CLU", best] + [x for x in rk if x != best] + \
            [x for x in arms if x != "CLU" and x not in rk]:
        A = [r["ROLLOUT"]["arms"][k] for r in test]
        fe = [a.get("mean_first_error_step") for a in A]
        L.append("| %s%s | 3 | %s | %s | %s | %s |"
                 % (k, " (= best rival per-step)" if k == best else "",
                    " / ".join(f(a["per_step_exactness"]) for a in A),
                    f(np.mean([a["per_step_exactness"] for a in A])),
                    " / ".join("—" if x is None else "%.2f" % x for x in fe),
                    " / ".join(f(a["validity"]) for a in A)))
    L += ["", "⚠ **No arm validates a rollout**: validity = every step of the ~%.0f-step oracle "
          "route exact AND goal within budget, chained on the arm's own states; per-step error "
          "compounds. Printed once, as a limitation." % test[0]["ROLLOUT"]["mean_route_len"]]
    # full arm tables
    L += ["", "## 5. Full arm tables (every arm, every seed)"]
    for leg, runs in (("test", test), ("val", val)):
        for row in rows:
            L += WT.arm_table(runs, row, COND, leg)
    # hidden widths inline note
    # deletion
    L += ["", "## 6. Deletion (TEST, both modes, every seed) — Advisor ruling (d): PRE-deletion "
          "composition on the SAME surviving-colour rows printed beside post-deletion", "",
          "| seed | mode | rows deleted / kept | store rel. ΔS (bar 1e-5) | atoms max abs Δ | "
          "downdate == fitted-without (surviving n-way rows identical) | CLU n-way BEFORE → AFTER "
          "(same rows, n) | CLU fitted-without | CLU adapt s (data-free) | rival retrain-without s "
          "(steps, examples) | rival n-way as-trained → retrain-without |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in test:
        for mode in ("value", "conjunction_colour_AND_tinted_floor"):
            b = r["DELETION"].get(mode, {})
            if "store_scoped_exactness" not in b:
                L.append("| %d | %s | %s |" % (r["seed_index"], mode,
                                              b.get("DECLARED_NOT_RUN", "missing")))
                continue
            s, c = b["store_scoped_exactness"], b["composition_on_surviving_colours"]["nway"]
            rv = b["rival_composition_on_surviving_colours"]["nway"]
            rc = b["rival_retrain_without_cost"]
            L.append("| %d | %s | %d / %d | %.2e (%s) | %.2e | %s | %s → %s (n=%d) | %s | %.2f | "
                     "%.1f (%d, %s) | %s → %s |"
                     % (r["seed_index"], mode, b["n_rows_deleted"], b["n_rows_kept"],
                        s["max_relative_delta_S"], "PASS" if s["pass"] else "FAIL",
                        s["deployed_atoms_max_abs_delta"], f(c["reads_identical_rows_frac"]),
                        f(c["CLU BEFORE deletion (as fitted, same rows)"]),
                        f(c["CLU after deletion (downdate)"]), c["n"],
                        f(c["CLU fitted WITHOUT the rows"]),
                        b["clu_adaptation_cost"]["seconds"], rc["seconds"],
                        rc["gradient_steps"], rc["examples_seen"],
                        f(rv["rival as-trained"]), f(rv["rival retrain-without"])))
    L += ["", "Frozen codec and state encoder are declared OUTSIDE the deletable object. CLU "
          "post-deletion adaptation is light and data-free (A, Mx, S_k, t_k, n_k downdated; the "
          "same closed form re-solved) — not 'no changes'. The rival deletes only by "
          "retrain-without (mlp, the condition's shared 318 input, 8,000 steps)."]
    L += ["", "Sources: per-seed `test/WORDS-FULL_test_seed{0,1,2}.json`; labels "
          "`test/BARS-WORDS-FULL-test.json` → `HEAD_LABEL`; VAL `%s/WORDS-FULL_val_seed*.json`."
          % W19_VAL]
    open(os.path.join(out, "TABLES.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    res = {"artefact": "WORDS-FULL-TEST.json", "spoke": "w21 spoke/words-full-test",
           "branch": "w21/words-full-test", "base": "w19/maze-t-symmetric 0455961",
           "target": "maze-t-symmetric", "scored_runs": "4/9, 5/9, 6/9 (seeds 0, 1, 2)",
           "prereg": "PREREG-w21-words-full-test.md",
           "condition": COND, "store_config": {"latent": 128, "K": 2, "n_atoms": 6, "ridge": 1.0,
                                               "name": "A1 (UNCHANGED)"},
           "n_seeds": 3, "tolerance_used": None,
           "instrument_check": {"artefact": "repro/REPRO_COMPARE.txt",
                                "w19_val_nway_triple": HV["nway"]["CLU_per_seed"],
                                "reproduced": True, "max_abs_delta": 0.0},
           "derangements_test": [r["across_wells_derangement"] for r in test],
           "LABEL_TEST_nway": HT["nway"]["LABEL"], "head_label_test": HT, "head_label_val": HV,
           "matched_parameters": r0["matched_parameters"],
           "capacity_diagnostic": r0["capacity_diagnostic"], "clu_ledger": r0["clu_ledger"],
           "per_step_and_rollout": {k: [r["ROLLOUT"]["arms"][k] for r in test]
                                    for k in ["CLU", best]},
           "best_rival_per_step": best,
           "deletion": {r["seed_index"]: r["DELETION"] for r in test},
           "artefacts": {"bars": "test/BARS-WORDS-FULL-test.json",
                         "per_seed": ["test/WORDS-FULL_test_seed%d.json" % s for s in SEEDS],
                         "tables": "TABLES.md"}}
    json.dump(res, open(os.path.join(out, "WORDS-FULL-TEST.json"), "w"), indent=1)
    print("TEST nway: %s (%s)  |  VAL nway: %s (%s)" % (HT["nway"]["LABEL"], HT["nway"]["why"],
                                                       HV["nway"]["LABEL"], HV["nway"]["why"]))
    for row in rows:
        x = HT[row]
        print("  %-5s CLU %.4f R* %s %.4f diff %+.4f sd %.4f ahead %d/3 gate %s -> %s"
              % (row, x["CLU_mean"], x["R_star"], x["R_star_mean"], x["mean_diff"],
                 x["one_seed_spread_sd_paired_diff"], x["seeds_CLU_ahead"],
                 x["store_gate_controls_le_CLU_minus_0.15"], x["LABEL"]))


if __name__ == "__main__":
    main(sys.argv[1])
