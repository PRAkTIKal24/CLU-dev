"""Generate TABLES.md from the w16 JSONs (no hand-copied cell).

    python scripts/w16_tables.py <out_dir>
"""
import json
import os
import sys

import numpy as np

O = sys.argv[1]
L = []
P = L.append


def load3(pat):
    fs = [os.path.join(O, pat % s) for s in range(3)]
    return [json.load(open(f)) for f in fs] if all(os.path.exists(f) for f in fs) else None


def fmt(v):
    return "%.4f ± %.4f | %s" % (np.mean(v), np.std(v, ddof=1), " / ".join("%.3f" % x for x in v))


def t_table(tag, leg, title):
    S = load3("t1/%s_%s_seed%%d.json" % (tag, leg))
    if S is None:
        P("\n### %s — NOT AVAILABLE\n" % title)
        return
    P("\n### %s\nSource: `t1/%s_%s_seed{0,1,2}.json`. Column `exact` (exact transition judge). "
      "n_seeds = 3. Config %s, iface `%s`, matched-rival steps %s.\n"
      % (title, tag, leg, S[0]["config"], S[0]["iface"], S[0].get("rival_steps_matched", 8000)))
    for row in ("nway", "pair", "value", "seen"):
        P("\n**row `%s`** — n per seed %s — claim label: see BARS\n" % (row, [s["ROWS"][row]["n"] for s in S]))
        P("| arm | mean ± sd | per seed |\n|---|---|---|")
        for a in S[0]["ROWS"][row]:
            if a == "n":
                continue
            P("| %s | %s |" % (a, fmt([s["ROWS"][row][a] for s in S])))
    P("\n**rollout (full optimal routes, chained generated states)** — %s rollouts/seed\n"
      % [s["ROLLOUT"]["n_rollouts"] for s in S])
    P("| arm | validity | per-step exactness |\n|---|---|---|")
    for a in S[0]["ROLLOUT"]["arms"]:
        P("| %s | %s | %s |" % (a, fmt([s["ROLLOUT"]["arms"][a]["validity"] for s in S]),
                                fmt([s["ROLLOUT"]["arms"][a]["per_step_exactness"] for s in S])))
    if "value" in S[0].get("DELETION", {}):
        P("\n**deletion** (deleted colour 1; store-scoped exactness; composition on surviving colours, nway)\n")
        P("| seed | deletion | rows del/kept | S_k rel Δ | atoms rel Δ | CLU downdate | CLU fitted-without | rival retrain-without | rival as-trained | CLU adapt s | rival retrain s |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, s in enumerate(S):
            for m in ("value", "conjunction_colour_AND_tinted_floor"):
                b = s["DELETION"][m]
                c = b["composition_on_surviving_colours"].get("nway", {})
                r = b["rival_composition_on_surviving_colours"].get("nway", {})
                P("| %d | %s | %d / %d | %.2e | %.2e | %.4f | %.4f | %.4f | %.4f | %.3f | %.1f |" % (
                    i, m.split("_")[0], b["n_rows_deleted"], b["n_rows_kept"],
                    b["store_scoped_exactness"]["max_relative_delta"],
                    b["store_scoped_exactness"]["deployed_atoms_max_relative_delta"],
                    c.get("CLU after deletion (downdate)", float("nan")),
                    c.get("CLU fitted WITHOUT the rows", float("nan")),
                    r.get("rival retrain-without", float("nan")), r.get("rival as-trained", float("nan")),
                    b["clu_adaptation_cost"]["seconds"], b["rival_retrain_without_cost"]["seconds"]))
    P("\n**ledger** (deployed bytes = 4 × (atom + address-book floats), float32; float64 sufficient statistic beside, never netted; rivals matched on read floats)\n")
    P("| seed | read floats | atoms | head | deployed B (f32) | stat B (f64) | rank/width per well | matched rivals (params, residual %) |")
    P("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(S):
        l = s["clu_ledger"]
        mp = s["matched_parameters"]["achieved"]
        P("| %d | %d | %d | %d | %d | %d | %s | %s |" % (
            i, l["read_floats"], l["store_atom_floats"], l["head_floats"], l["deployed_bytes_float32"],
            l["statistic_bytes_float64"], [round(x, 3) for x in l["span_fraction_n_atoms_over_d_k"]],
            ", ".join("%s %d (%+.1f)" % (k, v["params"], v["residual_pct"]) for k, v in mp.items())))


def o_table(tag, leg, title):
    S = load3("t2/%s_%s_seed%%d.json" % (tag, leg))
    if S is None:
        P("\n### %s — NOT AVAILABLE\n" % title)
        return
    P("\n### %s\nSource: `t2/%s_%s_seed{0,1,2}.json`. Column `valid` (exact objects AND legal). "
      "n_seeds = 3. Config %s, rival steps %s.\n" % (title, tag, leg, S[0]["config"], S[0].get("rival_steps", 3000)))
    for row in ("unseen_nway", "unseen_2way", "seen", "value_holdout", "value_slot_holdout"):
        P("\n**row `%s`** — n per seed %s\n" % (row, [s["ROWS"][row]["CLU"]["n"] for s in S]))
        P("| arm | mean ± sd | per seed |\n|---|---|---|")
        for a in S[0]["ROWS"][row]:
            P("| %s | %s |" % (a, fmt([s["ROWS"][row][a]["valid"] for s in S])))
    if S[0].get("DELETION"):
        P("\n**deletion** (colour 2; store-scoped; unseen_nway)\n")
        P("| seed | deletion | n deleted (frac) | rel ΔS | side | CLU as-trained | CLU after | CLU refit-without | MLP as-trained | MLP retrain-without | CLU adapt s | MLP retrain s |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, s in enumerate(S):
            for k, x in s["DELETION"]["deletions"].items():
                for side in ("other_colours", "deleted_colour"):
                    t = x["post_deletion_composition"]["unseen_nway"][side]
                    if "CLU_as_trained" not in t:
                        continue
                    P("| %d | %s | %d (%.3f) | %.2e | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.2f | %.1f |" % (
                        i, k, x["n_deleted"], x["fraction_of_train"],
                        x["store_scoped_exactness"]["relative_S_gap"], side,
                        t["CLU_as_trained"]["valid"], t["CLU_after_deletion"]["valid"],
                        t["CLU_refit_without"]["valid"], t["MLP_as_trained"]["valid"],
                        t["MLP_retrain_without"]["valid"], x["adaptation"]["CLU_adaptation_seconds"],
                        x["adaptation"]["rival_retrain_without_seconds"]))
    P("\n**ledger**: " + "; ".join("seed %d read floats %d (atoms %d + head %d), rank/width %s, lookup items %s, rivals %s" % (
        i, s["clu_ledger"]["read_floats"], s["clu_ledger"]["store_atom_floats"], s["clu_ledger"]["head_floats"],
        s["clu_ledger"].get("span_fraction"), s["lookup_ledger"].get("items_held"),
        {k: v.get("n_params", v.get("params")) for k, v in s["rivals"].items()}) for i, s in enumerate(S)))


def diag_tables():
    P("\n### TRAINING-BUDGET DIAGNOSTICS (VAL; NOT rival rows, NOT bars)\n")
    S = load3("t1/A1_rivaldiag_val_seed%d.json")
    if S:
        P("CMAZE-T, A1 rows, rivals at 80,000 steps (10× recipe). Source `t1/A1_rivaldiag_val_seed*.json`. CLU A1 VAL nway 0.802 / 0.569 / 0.803.\n")
        P("| arm | nway | pair | seen | value |\n|---|---|---|---|---|")
        for a in S[0]["arms"]:
            P("| %s | %s | %s | %s | %s |" % (a, *[fmt([s["arms"][a]["rows"][r] for s in S]) for r in ("nway", "pair", "seen", "value")]))
    S = load3("t2/C_rivaldiag_val_seed%d.json")
    if S:
        P("\nCMAZE-O, C-field rows, matched rivals at 30,000 steps (10× recipe). Source `t2/C_rivaldiag_val_seed*.json`. CLU C-field VAL unseen_nway 0.640 / 0.730 / 0.730.\n")
        P("| rival | unseen_nway | unseen_2way | seen | params | held-in MSE |\n|---|---|---|---|---|---|")
        for k in S[0]["rivals"]:
            P("| %s | %s | %s | %s | %d | %s |" % (k, *[fmt([s["rivals"][k]["rows"][r]["valid"] for s in S]) for r in ("unseen_nway", "unseen_2way", "seen")],
                                                  S[0]["rivals"][k]["params"], [round(s["rivals"][k]["heldin_val_mse"], 5) for s in S]))


P("# w16 hub/maze-wm — TABLES (generated by `scripts/w16_tables.py`; no hand-copied cell)\n")
P("⛔ CMAZE-T and CMAZE-O never share a table. n_seeds = 3 on every row. tolerance_used = null.\n")
P("\n## CMAZE-T (Task 1)")
t_table("A1", "test", "T1 · Config A1 · TEST (SCORED runs 1–3/18)")
t_table("C512", "test", "T1 · Config C-512 · TEST (SCORED runs 7–9/18)")
t_table("A1", "val", "T1 · Config A1 · VAL")
t_table("B", "val", "T1 · Config B (match-bit interface) · VAL — FAILED VB3")
t_table("C512", "val", "T1 · Config C-512 · VAL")
P("\n## CMAZE-O (Task 2)")
o_table("C", "test", "T2 · Config C (C-field) · TEST (SCORED runs 4–6/18)")
o_table("C", "val", "T2 · Config C (C-field) · VAL")
o_table("C2", "val", "T2 · Config C2 (latent, span 0.5) · VAL, recipe rivals")
o_table("C2r", "val", "T2 · Config C2 · VAL, 30k-step rivals — FAILED VB1")
diag_tables()
open(os.path.join(O, "TABLES.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("TABLES.md", len(L), "lines")
