"""Emit the report's markdown tables from the scored result JSONs.

    python -m w14_read_verb.tables <out_dir> > TABLES.md

Every number the final report prints comes from here, so no cell is ever hand-copied
and a regenerated report is byte-identical to the one that shipped.
"""

from __future__ import annotations

import glob
import json
import os
import sys

ARMS = [
    ("clu", "**CLU**"),
    ("clu_lowrank", "`clu_lowrank` (rank/width 0.1875)"),
    ("clu_nopair", "`clu_nopair` (pair block off)"),
    ("landscape_removed", "*`landscape_removed`*"),
    ("landscape_shuffled", "*`landscape_shuffled`*"),
    ("random_atoms", "*`random_atoms`*"),
    ("lookup_matched", "`lookup_matched` (the floor)"),
    ("mlp", "`mlp` @ matched params"),
    ("lstm", "`lstm` @ matched params"),
    ("transformer", "`transformer` @ matched params"),
]
ROWS = ("unseen_pair", "nway", "seen", "budget_interp")


def load(out_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "runs", "scored*_test.json"))):
        with open(p) as fh:
            runs.append(json.load(fh))
    return sorted(runs, key=lambda r: r["seed"])


def v(r, row, arm, verb):
    try:
        return r["table"][row][arm][verb]["validity"]
    except KeyError:
        return None


def fmt(x, bold=False):
    if x is None:
        return "—"
    s = "%.4f" % x
    return ("**%s**" % s) if bold else s


def verb_table(runs, row):
    out = ["", "#### row `%s` — validity under the exact integer judge, TEST, 3 seeds"
           % row, ""]
    out.append("| arm | verb | seed 0 | seed 1 | seed 2 |")
    out.append("|---|---|---|---|---|")
    for arm, label in ARMS:
        if arm not in runs[0]["table"].get(row, {}):
            continue
        for verb, vlabel in (("FULL", "`FULL` (w9's read)"),
                             ("AVG", "`AVG` (**the ablation**)"),
                             ("MERGE_target_wins", "**`MERGE` (the arm)**")):
            cells = [fmt(v(r, row, arm, verb), verb.startswith("MERGE")) for r in runs]
            out.append("| %s | %s | %s |" % (label, vlabel, " | ".join(cells)))
    return out


def delta_table(runs, row):
    out = ["", "#### row `%s` — ⭐ `MERGE` − `AVG`, EVERY arm, TEST, 3 seeds" % row, ""]
    out.append("| arm | seed 0 | seed 1 | seed 2 | mean |")
    out.append("|---|---|---|---|---|")
    for arm, label in ARMS:
        if arm not in runs[0]["table"].get(row, {}):
            continue
        ds = []
        for r in runs:
            m, a = v(r, row, arm, "MERGE_target_wins"), v(r, row, arm, "AVG")
            ds.append(None if None in (m, a) else m - a)
        good = [d for d in ds if d is not None]
        out.append("| %s | %s | %s |" % (
            label, " | ".join(("%+.4f" % d) if d is not None else "—" for d in ds),
            ("**%+.4f**" % (sum(good) / len(good))) if good else "—"))
    return out


def merge_controls(runs, row):
    out = ["", "#### row `%s` — the MERGE RULE's own controls, CLU, identical two plans"
           % row, ""]
    out.append("| overlay rule | seed 0 | seed 1 | seed 2 |")
    out.append("|---|---|---|---|")
    for verb, label in (("MERGE_target_wins", "**target's plan wins conflicts** (deployed)"),
                        ("MERGE_agent_wins", "agent's plan wins conflicts"),
                        ("MERGE_agent_only", "the agent half alone"),
                        ("MERGE_target_only", "the target half alone"),
                        ("AVG", "*latent averaging (the ablation)*")):
        out.append("| %s | %s |" % (label, " | ".join(
            fmt(v(r, row, "clu", verb), verb == "MERGE_target_wins") for r in runs)))
    return out


def context(runs):
    out = ["", "#### the leg, the ceilings and the environment's own columns", ""]
    out.append("| quantity | seed 0 | seed 1 | seed 2 |")
    out.append("|---|---|---|---|")
    def rowline(label, f):
        out.append("| %s | %s |" % (label, " | ".join(str(f(r)) for r in runs)))
    rowline("budget words (`GRID_REBASED`)", lambda r: "/".join(map(str, r["budgets"])))
    rowline("training requests", lambda r: r["n_train_requests"])
    rowline("n `unseen_pair` / `nway`",
            lambda r: "%d / %d" % (r["n_rows"]["unseen_pair"], r["n_rows"]["nway"]))
    rowline("pipeline ceiling `unseen_pair`",
            lambda r: "%.4f" % r["pipeline_ceiling_rows"]["unseen_pair"]["validity"])
    rowline("pipeline ceiling `nway`",
            lambda r: "%.4f" % r["pipeline_ceiling_rows"]["nway"]["validity"])
    rowline("store-span ceiling `unseen_pair`",
            lambda r: "%.4f" % r["span_ceiling_rows"]["unseen_pair"]["validity"])
    rowline("**rank/width** (every store row)",
            lambda r: "%.4f" % r["ledger"]["rank_over_width"][0])
    rowline("read floats (atoms + address book)",
            lambda r: "%d (%d + %d)" % (r["ledger"]["read_floats"],
                                        r["ledger"]["store_atom_floats"],
                                        r["ledger"]["head_floats"]))
    rowline("`tolerance_used`", lambda r: str(r["tolerance_used"]))
    for rw in ("unseen_pair", "nway"):
        rowline("**BLIND-SEQUENTIAL** `%s` (OWN split)" % rw,
                lambda r, rw=rw: "%.4f" % [
                    x["validity"] for k, x in r["env_columns"][rw]["blind"].items()
                    if k.startswith("BLIND-SEQUENTIAL")][0])
        rowline("ORACLE `%s`" % rw,
                lambda r, rw=rw: "%.4f" % [
                    x["validity"] for k, x in r["env_columns"][rw]["blind"].items()
                    if k.startswith("ORACLE")][0])
        rowline("rule-B bend fraction `%s`" % rw,
                lambda r, rw=rw: "%.4f" % r["env_columns"][rw]["bend"]["bend_fraction"])
    return out


def bars(runs, row="unseen_pair"):
    out = ["", "#### the PRE-REGISTERED bars, recomputed from the JSONs", ""]
    out.append("| bar | seed 0 | seed 1 | seed 2 | verdict |")
    out.append("|---|---|---|---|---|")
    d_clu = [v(r, row, "clu", "MERGE_target_wins") - v(r, row, "clu", "AVG")
             for r in runs]
    d_tr = [v(r, row, "transformer", "MERGE_target_wins")
            - v(r, row, "transformer", "AVG") for r in runs]
    out.append("| **BAR 1** CLU `MERGE`−`AVG` ≥ +0.10 | %s | %s |" % (
        " | ".join("%+.4f" % d for d in d_clu),
        "⭐ **PASS**" if all(d >= 0.10 for d in d_clu) else "⛔ **FAIL**"))
    out.append("| **BAR 2** CLU delta − transformer delta ≥ +0.05 | %s | %s |" % (
        " | ".join("%+.4f" % (a - b) for a, b in zip(d_clu, d_tr)),
        "PASS" if all(a - b >= 0.05 for a, b in zip(d_clu, d_tr))
        else "⛔ **FAIL**"))
    cs = [v(r, "seen", "clu", "MERGE_target_wins") for r in runs]
    for ctl in ("landscape_shuffled", "random_atoms"):
        cv = [v(r, "seen", ctl, "MERGE_target_wins") for r in runs]
        out.append("| **BAR 3** `%s` ≤ 0.5 × CLU on `seen` | %s | %s |" % (
            ctl, " | ".join("%.4f vs %.4f" % (a, 0.5 * b) for a, b in zip(cv, cs)),
            "PASS" if all(a <= 0.5 * b for a, b in zip(cv, cs)) else "⛔ **FAIL**"))
    m = [v(r, row, "clu", "MERGE_target_wins") for r in runs]
    lk = [v(r, row, "lookup_matched", "MERGE_target_wins") for r in runs]
    out.append("| **BAR 5** CLU `MERGE` > `lookup_matched` `MERGE` | %s | %s |" % (
        " | ".join("%.4f vs %.4f" % (a, b) for a, b in zip(m, lk)),
        "PASS" if all(a > b for a, b in zip(m, lk)) else "⛔ **FAIL**"))
    return out


def latent(runs, row="unseen_pair"):
    out = ["", "#### ⭐⭐ WHY `AVG` CANNOT WORK — the latent geometry, row `%s`" % row, ""]
    out.append("| arm | quantity | seed 0 | seed 1 | seed 2 |")
    out.append("|---|---|---|---|---|")
    for arm in ("clu", "lookup_matched", "transformer"):
        for k, lab in (("cos_A_B", "**cos(Z_A, Z_B)**"),
                       ("cos_AVG_FULL", "cos(Z_AVG, Z_FULL)")):
            out.append("| `%s` | %s | %s |" % (arm, lab, " | ".join(
                "%+.4f" % r["table"][row][arm]["_latent"][k] for r in runs)))
    return out


def main(out_dir):
    runs = load(out_dir)
    if not runs:
        print("no scored runs in", out_dir)
        return
    lines = ["<!-- generated by w14_read_verb.tables -- do not hand-edit -->"]
    lines += context(runs)
    for row in ("unseen_pair", "nway", "seen"):
        lines += verb_table(runs, row)
        lines += delta_table(runs, row)
    lines += merge_controls(runs, "unseen_pair")
    lines += merge_controls(runs, "nway")
    lines += latent(runs, "unseen_pair")
    lines += bars(runs, "unseen_pair")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/chlu/outputs/w14-read-verb")
