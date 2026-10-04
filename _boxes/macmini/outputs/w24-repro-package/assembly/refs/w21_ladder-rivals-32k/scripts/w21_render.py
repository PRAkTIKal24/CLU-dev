"""w21 spoke/ladder-rivals-32k -- RENDER TABLES.md, CURVES.md, CAPTION.md, caption.json.

Renders LADDER-RIVALS-32K.json only; computes nothing about any arm.  Every cell names the
JSON key it came from (TABLES.md section headers name the key path).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w21_lib import FAMILIES, OUT, RUNGS  # noqa: E402

SEEDS = ("0", "1", "2")
STEPMATCH = {"mlp": 204_800, "cont_mlp": 204_800, "lstm": 20_480, "transformer": 20_480}

CAPTION = (
    "**Ladder rivals, three column sets — pool AND optimiser steps named for each.** "
    "Store: K\\*=32, content fitted on the rung's first **32,768** rows (`n_store_train`, read from "
    "w12b `kstar_ladder/*/LEG.json` `n_fit_rows` and `partitions.json`), from a 131,072-row rung pool. "
    "**Set A** (the draft's ladder rivals, spoke-A `@ref`): **32,256** Montero combination-split rows "
    "(⚠ this pool holds the value-holdout's held-out values), 117,878–119,858 params, **5,040** steps. "
    "**Set B** (w20 data-matched): the rung's **131,072**-row pool, 130,433–131,891 params; DECLARED "
    "**20,480** steps, TENX **204,800** steps (mlp and cont_mlp carried at TENX, lstm and transformer at DECLARED). "
    "**Set C** (w21, this table): the store's own **32,768** content rows, the same four families at the same "
    "read-path float match (mlp 130,514 / lstm 130,721 / transformer 130,433 / cont_mlp 131,891 params, as set B), "
    "spoke-A selection re-run on these rows (lr 3e-3, 40 epochs on all 60 legs); DECLARED **5,120** steps, TENX **51,200** steps, and a step-matched column at set B's "
    "carried budget (mlp and cont_mlp **204,800**, lstm and transformer **20,480**). "
    "`learned_ref_ownpool`: the same **32,768** rows, 130,514 params, **2,560** steps. "
    "Margin = store − the strongest rival per seed (max over families of max(DECLARED, TENX); w20's convention), "
    "paired per corner, bootstrap CI95. Seeds = 3 in every cell. Rungs n = 3–6 probe the same 120 corners "
    "(ceiling 72); n = 2 has 60 corners (ceiling 36).")


def f2(x, nd=2, sign=False):
    if x is None:
        return "—"
    return ("%+.*f" if sign else "%.*f") % (nd, x)


def seeds_str(per, key="delta", sign=True):
    if not per:
        return "NOT RUN"
    vals = [per.get(s, per.get(int(s)) if isinstance(per, dict) else None) for s in SEEDS]
    return " / ".join(f2(v[key], sign=sign) if v else "—" for v in vals)


def mean_of(ms_):
    return None if not ms_ else ms_.get("mean")


def main():
    J = json.loads((OUT / "LADDER-RIVALS-32K.json").read_text(encoding="utf-8"))
    R = J["rungs"]
    L = []
    w = L.append
    w("# w21 spoke/ladder-rivals-32k — TABLES")
    w("")
    w("Every cell is read from `LADDER-RIVALS-32K.json` (key path in each section header); that JSON names "
      "the per-corner score file or fit artefact behind every number. FREE REPLICATION: no ledger row, "
      "0 scored runs. Seeds 0/1/2 in every cell.")
    w("")
    w("## Table L3 — the ladder with THREE rival column sets")
    w("")
    w("**Caption.** " + CAPTION)
    w("")
    w("Key: `rungs.<n>.margins_K32_store_minus_rival.<set>.per_seed.<s>.delta` (CI95 in the JSON), "
      "`rungs.<n>.store.summary`, `rungs.<n>.crosscheck.X1_same_rows_same_steps.ref`.")
    w("")
    w("| rung | store K\\*=32 (content 32,768 rows) | **A**: K\\* − strongest spoke-A `@ref` (32,256 rows · 5,040 steps) s0/s1/s2 [mean] "
      "| **B**: K\\* − strongest w20 (131,072 rows · 20,480 / 204,800 steps) s0/s1/s2 [mean] "
      "| **C**: K\\* − strongest w21 (32,768 rows · 5,120 / 51,200 steps) s0/s1/s2 [mean] "
      "| **C-sm**: K\\* − strongest w21 step-matched to B (32,768 rows · 20,480 / 204,800 steps) [mean] "
      "| `learned_ref_ownpool` (32,768 rows · 2,560 steps) | n |")
    w("|---|---|---|---|---|---|---|---|")
    for k in RUNGS:
        r = R[k]
        m = r["margins_K32_store_minus_rival"]
        cells = []
        for key in ("setA_vs_strongest", "setB_vs_strongest_declared_or_tenx_RECOMPUTED",
                    "setC_vs_strongest_declared_or_tenx", "setC_vs_strongest_stepmatched"):
            b = m.get(key, {})
            md = mean_of(b.get("mean_delta"))
            cells.append("%s [**%s**]" % (seeds_str(b.get("per_seed", {})), f2(md, sign=True))
                         + ("" if b.get("complete_3_seeds") else " ⚠ incomplete"))
        ref = r["crosscheck"]["X1_same_rows_same_steps"]["ref"]
        w("| n=%s | %s | %s | %s | %s | %s | %s | 3 |" % (
            k, f2(mean_of(r["store"]["summary"])), cells[0], cells[1], cells[2], cells[3], f2(mean_of(ref))))
    w("")
    w("Strongest rival per seed (the label the margin was taken against):")
    w("")
    w("| rung | A | B | C | C-sm |")
    w("|---|---|---|---|---|")
    for k in RUNGS:
        m = R[k]["margins_K32_store_minus_rival"]
        cs = []
        for key in ("setA_vs_strongest", "setB_vs_strongest_declared_or_tenx_RECOMPUTED",
                    "setC_vs_strongest_declared_or_tenx", "setC_vs_strongest_stepmatched"):
            per = m.get(key, {}).get("per_seed", {})
            cs.append(" / ".join("%s (%s)" % (per[s]["rival"], f2(per[s]["rival_mean"])) if s in per else "—"
                                 for s in SEEDS))
        w("| n=%s | %s | %s | %s | %s |" % (k, *cs))
    w("")
    w("Set C, rival's best case (max over every set-C fit per seed; `setC_vs_best_of_every_fit`): " +
      "; ".join("n=%s %s [%s]" % (k, seeds_str(R[k]["margins_K32_store_minus_rival"]["setC_vs_best_of_every_fit"]["per_seed"]),
                                   f2(mean_of(R[k]["margins_K32_store_minus_rival"]["setC_vs_best_of_every_fit"]["mean_delta"]), sign=True))
                for k in RUNGS) + ".")
    w("")
    w("Integrity: set B recomputed from w20's score files equals w20's printed ladder P3 — " +
      ", ".join("n=%s %s" % (k, R[k]["margins_K32_store_minus_rival"]["setB_integrity_vs_w20_printed"].get("identical"))
                for k in RUNGS) + " (`setB_integrity_vs_w20_printed`).")
    w("")

    # ---- per family ----
    w("## Table F — every family, every set, per rung (3-seed mean; per-seed values in the JSON)")
    w("")
    w("Keys: `rungs.<n>.setA_spokeA_32256.<fam>.summary`, `rungs.<n>.setB_w20_131072.<fam>.<stage>.summary`, "
      "`rungs.<n>.setC_w21_32768.<fam>.<stage>.summary`. Column headers give POOL · STEPS.")
    w("")
    w("| rung | family | A: 32,256 · 5,040 | B DECLARED: 131,072 · 20,480 | B TENX: 131,072 · 204,800 "
      "| **C DECLARED: 32,768 · 5,120** | **C TENX: 32,768 · 51,200** | **C step-matched: 32,768 · (fam)** |")
    w("|---|---|---|---|---|---|---|---|")
    for k in RUNGS:
        r = R[k]
        for fam in FAMILIES:
            a = mean_of(r["setA_spokeA_32256"][fam]["summary"])
            bd = r["setB_w20_131072"][fam]["declared"]["summary"]
            bt = r["setB_w20_131072"][fam]["tenx"]["summary"]
            c = r["setC_w21_32768"][fam]
            sm = c["steps%d" % STEPMATCH[fam]]["summary"]

            def cell(x):
                if not x:
                    return "NOT RUN"
                return f2(x["mean"]) + ("" if x["n_seeds"] == 3 else " (%d seed%s)" % (x["n_seeds"], "" if x["n_seeds"] == 1 else "s"))
            w("| n=%s | %s | %s | %s | %s | %s | %s | %s · %s |" % (
                k, fam, f2(a), cell(bd), cell(bt), cell(c["declared"]["summary"]), cell(c["tenx"]["summary"]),
                cell(sm), "{:,}".format(STEPMATCH[fam])))
    w("")
    w("B TENX was run on every rung/seed only for mlp and cont_mlp; for lstm/transformer only as w20's "
      "spot check (rungs 2 and 6, seed 0) — shown with its seed count.")
    w("")

    # ---- P1 ----
    w("## Table P1 — convergence of set C (TENX − DECLARED, 10× the steps; bar 1.0)")
    w("")
    w("Key: `rungs.<n>.setC_w21_32768.<fam>.P1.<s>`. Last column: step-matched − TENX.")
    w("")
    w("| rung | family | DECLARED s0/s1/s2 | TENX s0/s1/s2 | TENX − DECLARED | P1 | step-matched − TENX |")
    w("|---|---|---|---|---|---|---|")
    for k in RUNGS:
        for fam in FAMILIES:
            P = R[k]["setC_w21_32768"][fam]["P1"]
            g = [P.get(s, {}) for s in SEEDS]
            w("| n=%s | %s | %s | %s | %s | %s | %s |" % (
                k, fam, " / ".join(f2(x.get("declared")) for x in g), " / ".join(f2(x.get("tenx")) for x in g),
                " / ".join(f2(x.get("tenx_minus_declared"), sign=True) for x in g),
                " / ".join({"PASS": "PASS", "UNDER-TRAINED: TENX carries": "FAIL"}.get(x.get("P1"), x.get("P1", "—")) for x in g),
                " / ".join(f2(x.get("stepmatched_minus_tenx"), sign=True) for x in g)))
    w("")

    # ---- cross-check ----
    w("## Table X1 — THE CROSS-CHECK: same rows, same steps (`crosscheck.X1_same_rows_same_steps`)")
    w("")
    w("`learned_ref_ownpool` (EXISTING, w12 composition-ladder AMENDMENT 1) vs this spoke's `mlp__ownpoolsched`: "
      "both mlp, width 61, 130,514 params, the rung's first 32,768 rows, 20 epochs, lr 1e-3, wd 0, batch 256 = "
      "2,560 steps. Pre-declared bar (LOG 13:58Z): |Δ of 3-seed means| ≤ the larger across-seed SD.")
    w("")
    w("| rung | ref s0/s1/s2 [mean ± sd] | mine s0/s1/s2 [mean ± sd] | Δ mine − ref | bar (max sd) | per-seed paired Δ [CI95] | verdict |")
    w("|---|---|---|---|---|---|---|")
    for k in RUNGS:
        X = R[k]["crosscheck"]["X1_same_rows_same_steps"]
        rf, mn = X.get("ref"), X.get("mine")

        def msd(z):
            if not z:
                return "NOT RUN"
            return "%s [%s ± %s]" % (" / ".join(f2(v) for v in z["per_seed"]), f2(z["mean"]), f2(z["sd"]))
        pp = []
        for s in SEEDS:
            q = X["per_seed"].get(s, {}).get("paired_mine_minus_ref")
            pp.append("%s [%s, %s]" % (f2(q["delta"], sign=True), f2(q["ci95"][0], sign=True), f2(q["ci95"][1], sign=True)) if q else "—")
        w("| n=%s | %s | %s | %s | %s | %s | %s |" % (
            k, msd(rf), msd(mn), f2(X.get("delta_mean_mine_minus_ref"), sign=True), f2(X.get("seed_spread_bar")),
            "; ".join(pp), X.get("verdict", "NOT RUN")))
    w("")
    w("Diagnostic, NOT the verdict (`crosscheck.X1_diagnostic_ref_recipe_replay`): `learned_ref_ownpool`'s own "
      "recipe replayed with its own PRNG keys and trainer (`scripts/w21_x1_replay.py`), compared per corner to the "
      "EXISTING rows file:")
    w("")
    w("| rung | replay s0/s1/s2 | ref s0/s1/s2 | corners differing s0/s1/s2 | identical per corner |")
    w("|---|---|---|---|---|")
    for k in RUNGS:
        dg = R[k]["crosscheck"].get("X1_diagnostic_ref_recipe_replay", {}).get("per_seed", {})
        g = [dg.get(s, {}) for s in SEEDS]
        w("| n=%s | %s | %s | %s | %s |" % (
            k, " / ".join(f2(x.get("replay_mean")) for x in g), " / ".join(f2(x.get("ref_mean")) for x in g),
            " / ".join("%s/%s" % (x.get("n_corners_differing", "—"), x.get("n_corners", "—")) for x in g),
            " / ".join(str(x.get("identical_per_corner", "NOT RUN")) for x in g)))
    w("")
    w("## Table X2 — same rows, DIFFERENT steps (expected to differ: the measured steps effect)")
    w("")
    w("Key: `crosscheck.X2_same_rows_different_steps`. All columns: mlp, the same 32,768 rows.")
    w("")
    w("| rung | `learned_ref_ownpool` · 2,560 steps | mine `ownpoolsched` · 2,560 | mine DECLARED · 5,120 | mine TENX · 51,200 | mine step-matched · 204,800 | w20 mlp TENX · 131,072 rows · 204,800 |")
    w("|---|---|---|---|---|---|---|")
    for k in RUNGS:
        X2 = R[k]["crosscheck"]["X2_same_rows_different_steps"]
        X1 = R[k]["crosscheck"]["X1_same_rows_same_steps"]
        w("| n=%s | %s | %s | %s | %s | %s | %s |" % (
            k, f2(mean_of(X2["ref_2560_steps"])), f2(mean_of(X1.get("mine"))), f2(mean_of(X2["mine_declared"])),
            f2(mean_of(X2["mine_tenx"])), f2(mean_of(X2["mine_steps204800"])),
            f2(mean_of(R[k]["setB_w20_131072"]["mlp"]["tenx"]["summary"]))))
    w("")
    (OUT / "TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- curves ----
    C = ["# w21 — LOSS CURVES (set C, every rung × seed × family × stage)", "",
         "Per-epoch mean train MSE, subsampled to ≈20 points (every k epochs, final epoch appended). Full per-epoch "
         "curves: `rivals/rung<n>_n32768/seed<s>/<fam>/<stage>.json` `curve`. Selection held-out curves: "
         "`.../selection.json` `full[].curve`. Key here: `rungs.<n>.convergence_evidence.<fam>.<s>.<stage>`.", ""]
    for k in RUNGS:
        for fam in FAMILIES:
            for s in SEEDS:
                ev = R[k]["convergence_evidence"].get(fam, {}).get(s, {})
                sel = ev.get("selection")
                C.append("### n=%s · %s · seed %s" % (k, fam, s))
                if sel:
                    C.append("selection: %s; held-out MSE curve (40-epoch trial, final 3): %s" % (
                        json.dumps(sel["selected"]), sel["heldout_curve_by_epochs"].get("40", [])[-3:]))
                for st, fm in ev.items():
                    if st == "selection" or not fm:
                        continue
                    key = [x for x in fm if x.startswith("train_mse_curve_every_")][0]
                    C.append("- **%s** · %d rows · %d epochs · %s steps · lr %g wd %g · %s: %s (last-decile rel. change %.4f)" % (
                        st, fm["n_fit_rows"], fm["epochs"], "{:,}".format(fm["optimizer_steps"]), fm["lr"],
                        fm["weight_decay"], key.replace("train_mse_curve_", ""), fm[key], fm["last_decile_rel_change"]))
                C.append("")
    (OUT / "CURVES.md").write_text("\n".join(C) + "\n", encoding="utf-8")

    (OUT / "CAPTION.md").write_text(CAPTION + "\n", encoding="utf-8")
    cap = {"caption_markdown": CAPTION,
           "pools": {"store_content": 32768, "set_A_spokeA": 32256, "set_B_w20": 131072, "set_C_w21": 32768,
                     "learned_ref_ownpool": 32768},
           "steps": {"set_A_spokeA": 5040, "set_B_declared": 20480, "set_B_tenx": 204800,
                     "set_C_declared": 5120, "set_C_tenx": 51200,
                     "set_C_stepmatched": STEPMATCH, "learned_ref_ownpool": 2560},
           "source_json": str(OUT / "LADDER-RIVALS-32K.json")}
    (OUT / "caption.json").write_text(json.dumps(cap, indent=1), encoding="utf-8")
    print("wrote TABLES.md, CURVES.md, CAPTION.md, caption.json")


if __name__ == "__main__":
    main()
