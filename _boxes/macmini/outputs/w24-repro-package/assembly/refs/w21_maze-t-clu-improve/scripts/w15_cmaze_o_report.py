"""Print the CMAZE-O VAL-smoke tables as markdown, every number with its artefact path.

Usage:  python scripts/w15_cmaze_o_report.py <out_dir>
        (out_dir is the spoke output dir, which must contain valsmoke/VAL-SMOKE.json)
"""

from __future__ import annotations

import io
import json
import os
import sys

ARM_ORDER = [
    ("AE-CEILING", "AE ceiling (encode the true picture, decode, judge)"),
    ("CLU", "**CLU — Config-A single design**"),
    ("RIVAL:mlp", "MLP, matched params"),
    ("RIVAL:cont_mlp", "continuous MLP, matched params"),
    ("RIVAL:lstm", "LSTM, matched params"),
    ("RIVAL:transformer", "transformer, matched params"),
    ("LOOKUP", "matched-bytes LOOKUP **floor** (never a rival)"),
    ("CTRL:landscape_removed", "control: landscape_removed"),
    ("CTRL:landscape_shuffled", "control: landscape_shuffled"),
    ("CTRL:landscape_shuffled_across_wells",
     "control: landscape_shuffled_across_wells"),
    ("CTRL:random_atoms", "control: random_atoms"),
    ("BLIND-layout-free-cells", "BLIND: layout-free cells"),
    ("BLIND-colour-marginal", "BLIND: colour marginal"),
    ("BLIND-marginal", "BLIND: fully factorised marginal"),
    ("BLIND-no-layout", "BLIND: no layout word"),
    ("RANDOM", "RANDOM state"),
]
ROW_LABEL = {"seen": "seen", "unseen_2way": "unseen 2-way",
             "unseen_nway": "unseen n-way (n=4)",
             "value_holdout": "value hold-out (strict)",
             "value_slot_holdout": "value hold-out (slot)"}


def fmt(cell, key="valid"):
    if cell is None:
        return "—"
    m, s = cell[key]["mean"], cell[key]["sd"]
    return "%.4f ± %.4f" % (m, s)


def main(argv):
    out_dir = argv[1]
    sp = os.path.join(out_dir, "valsmoke", "VAL-SMOKE.json")
    d = json.load(open(sp, encoding="utf-8"))
    t, rows, seeds = d["table"], d["rows"], d["seeds"]
    L = []
    A = L.append

    for key in ("valid", "exact_objects", "legal", "exact_layout"):
        A("")
        A("### `%s` — VAL, %d seeds, mean ± sd over seeds" % (key, len(seeds)))
        A("")
        A("Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.%s`. "
          "`tolerance_used` is `null` on every cell." % key)
        A("")
        A("| arm | " + " | ".join(ROW_LABEL[r] for r in rows) + " |")
        A("|---|" + "---|" * len(rows))
        for arm, label in ARM_ORDER:
            if arm not in t:
                continue
            A("| %s | %s |" % (label, " | ".join(
                fmt(t[arm].get(r), key) for r in rows)))
        A("")
        A("Row sizes per seed: " + ", ".join(
            "%s n=%s" % (ROW_LABEL[r], t["CLU"][r]["n_per_seed"])
            for r in rows if r in t.get("CLU", {})) + ".")

    # ------------------------------------------------------------------ ledgers
    A("")
    A("### Memory ledger — printed separately, never matched (COMMON.md)")
    A("")
    A("Source: `valsmoke/VAL-SMOKE.json` → `per_seed_ledgers.<seed>`. "
      "Seed count is a column. Byte convention is named on every figure.")
    A("")
    led = d["per_seed_ledgers"]
    ss = [str(s) for s in seeds]
    A("| object | " + " | ".join("seed %s" % s for s in ss)
      + " | seeds | byte convention |")
    A("|---|" + "---|" * (len(ss) + 2))

    def row(label, fn, conv):
        A("| %s | %s | %d | %s |"
          % (label, " | ".join(fn(led[s]) for s in ss), len(ss), conv))

    row("CLU deployed read floats (atoms + address book)",
        lambda L_: "{:,}".format(L_["clu"]["read_floats"]),
        "4 bytes/float, float32")
    row("… of which atoms", lambda L_: "{:,}".format(L_["clu"]["store_atom_floats"]),
        "float32")
    row("… of which address book", lambda L_: "{:,}".format(L_["clu"]["head_floats"]),
        "float32")
    row("CLU deployed bytes",
        lambda L_: "{:,}".format(L_["clu"]["deployed_bytes_float32"]), "float32")
    row("CLU float64 sufficient statistic (bytes)",
        lambda L_: "{:,}".format(L_["clu"]["statistic_bytes_float64"]),
        "float64 — buys exact deletion, **never netted in, never matched**")
    row("CLU span fraction (n_atoms / d_k)",
        lambda L_: "%.3f" % L_["clu"]["span_fraction"], "—")
    row("LOOKUP floor items held",
        lambda L_: "{:,}".format(L_["lookup"]["items_held"]),
        "matched to the CLU's read floats")
    for k in ("mlp", "cont_mlp", "lstm", "transformer"):
        row("rival %s params (matched)" % k,
            lambda L_, k=k: "{:,} ({})".format(
                L_["rivals"][k]["params"],
                "matched" if L_["rivals"][k]["matched_within_2pct"] else "NOT matched"),
            "params × 4 bytes, float32")
    row("shared frozen AE params (encoder + decoder)",
        lambda L_: "{:,}".format(L_["ae"]["total"]),
        "declared OUTSIDE the deletable object; identical for every arm")

    # ------------------------------------------------------------------ deletion
    A("")
    A("### Deletion rows — VAL, colour c = 2")
    A("")
    A("Source: `valsmoke/VAL-SMOKE.json` → `deletion.<seed>`; "
      "`valsmoke/seed<N>.json` → `deletion`.")
    A("")
    dele = d.get("deletion") or {}
    s0 = ss[0]
    if dele.get(s0):
        A("| deletion | seed | rows deleted | store-scoped max abs ΔS | rel. ΔS | Δn | "
          "CLU adaptation (s) | rival retrain-without (s) | speed-up |")
        A("|---|---|---|---|---|---|---|---|---|")
        for kind in ("value", "conjunction"):
            for s in ss:
                if not dele.get(s):
                    continue
                e = dele[s]["deletions"][kind]
                g, ad = e["store_scoped_exactness"], e["adaptation"]
                A("| %s | %s | %d (%.1f%%) | %.3e | %.3e | %d | %.2f | %.0f | %.0f× |"
                  % (kind, s, e["n_deleted"], 100 * e["fraction_of_train"],
                     g["max_abs_diff_S"], g["relative_S_gap"], g["max_abs_diff_n"],
                     ad["CLU_adaptation_seconds"], ad["rival_retrain_without_seconds"],
                     ad["speedup_vs_rival_retrain"]))
        A("")
        A("#### Post-deletion composition on the OTHER colours (`valid`)")
        A("")
        A("| deletion | row | side | CLU as-trained | CLU after deletion | "
          "CLU refit-without | MLP as-trained | MLP retrain-without | n per seed |")
        A("|---|---|---|---|---|---|---|---|---|")
        for kind in ("value", "conjunction"):
            for rowname in ("seen", "unseen_2way", "unseen_nway"):
                for side in ("other_colours", "deleted_colour"):
                    cells, ns = [], []
                    for arm in ("CLU_as_trained", "CLU_after_deletion",
                                "CLU_refit_without", "MLP_as_trained",
                                "MLP_retrain_without"):
                        vals = []
                        for s in ss:
                            if not dele.get(s):
                                continue
                            e = dele[s]["deletions"][kind][
                                "post_deletion_composition"].get(rowname, {}).get(side)
                            if e and arm in e:
                                vals.append(e[arm]["valid"])
                                if arm == "CLU_as_trained":
                                    ns.append(e[arm]["n"])
                        cells.append("%.4f" % (sum(vals) / len(vals)) if vals else "—")
                    A("| %s | %s | %s | %s | %s |"
                      % (kind, ROW_LABEL[rowname], side, " | ".join(cells),
                         "+".join(str(x) for x in ns) if ns else "—"))
    else:
        A("**DECLARED NOT-RUN** — no deletion record in the smoke artefact.")

    dest = os.path.join(out_dir, "TABLES.md")
    with io.open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (dest, len(L)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
