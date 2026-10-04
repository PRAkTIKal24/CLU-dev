"""Generate cells_pass3.py (run once; the build never runs this, it re-verifies the static register).

Spec entries:
  ("row", id, alias, line)          -> the whole markdown table row on that line, outer pipes stripped
  ("q", id, alias, value)           -> the line is found; the value must occur on exactly one line
  ("ql", id, alias, value, line)    -> the value must occur on the given line (for repeated text)
"""
import pprint
from pathlib import Path

EV = Path("/Users/batmac/clu/evidence/outputs")
HERE = Path(__file__).resolve().parent

SRC3 = {
    "W19F": "w19-maze-t-symmetric/FINDINGS.md",
    "W19T": "w19-maze-t-symmetric/TABLES.md",
    "W19P": "w19-maze-t-symmetric/PREREG-w19-maze-t-symmetric.md",
    "W19L": "w19-maze-t-symmetric/LOG.md",
    "W19E": "w19-maze-t-symmetric/END_UTC.txt",
    # the live drafts AFTER this pass's edits
    "DNOW": "/Users/batmac/clu/outputs/w17-paper/DRAFT-v2.tex",
    "DSTY": "/Users/batmac/clu/outputs/w17-paper/DRAFT-v2-style.tex",
    # the Hub-corrected report, current text
    "W16NOW": "w16-advisor-report.md",
    "CAB": "/Users/batmac/clu/outputs/w17-paper/CORRECTIONS-ABSORBED.tex",
}

SPEC = [
    # ---- w19 condition table IN FULL (TABLES §1), every condition, win or lose
    ("q", "s_rule", "W19T", "⛔ CMAZE-T only. No wave-16 row appears in any table."),
    ("row", "s_hdr", "W19T", 9),
    ("q", "s_r_def", "W19T", "R* = best MATCHED rival over every kind/input variant AND both budgets (8,000 / 80,000 steps). n_seeds = 3 on every row."),
    ("row", "s_wm", "W19T", 11),
    ("row", "s_wf", "W19T", 12),
    ("row", "s_obs", "W19T", 13),
    ("row", "s_bothv", "W19T", 14),
    ("row", "s_botht", "W19T", 15),
    ("row", "s_x10", "W19T", 16),
    ("q", "s_bkey", "W19T", "B1..B4 = VB1..VB4 on VAL, SB1..SB4 on TEST (PREREG-w19). WORDS-MAIN: B3 on BLIND-COLOUR only"),
    # the record's own verdict table (FINDINGS §0)
    ("q", "f_answer", "W19F", "**The w16 CMAZE-T store win does NOT survive symmetric inputs.**"),
    ("q", "f_norule", "W19F", "No condition wins by the"),
    ("q", "f_wm", "W19F", "rivals given the words compose far better than a main-effects CLU: 0.747 vs 0.334"),
    ("q", "f_wf", "W19F", "⭐ **the pair block, not the store, carries the w16 margin**: a matched transformer given the CLU's own design reads 0.742 vs CLU 0.725"),
    ("q", "f_obs", "W19F", "CLU 0.351 vs rivals 0.457; the extractor recovers the whole word tuple on only 6–22 % of held-out n-way rows"),
    ("q", "f_both", "W19F", "margin +0.099 < 0.10, and the CLU is behind the matched 8k LSTM on seed 1 (0.566 vs 0.567)"),
    ("q", "f_x10", "W19F", "even a 10× extractor leaves the CLU at 0.447 vs rivals 0.461"),
    ("q", "f_budget", "W19F", "**Budget: 3 of 9 scored runs spent (BOTH, TEST), 6 DECLARED NOT-RUN**"),
    ("q", "f_survives", "W19F", "⭐ **What survives is the store GATE, not a rival margin.**"),
    ("q", "f_removed0", "W19F", "four charter-C controls collapse relative to the CLU (landscape_removed = 0.000"),
    ("q", "f_notbetter", "W19F", "but on this task it is not better than a matched network given the same"),
    ("q", "f_asym", "W19F", "In short: w16's rivals read a 636-dim conv-feature observation, while the CLU read the five"),
    ("q", "f_asym2", "W19F", "words through a 318-wide `[bias|main|pair]` design. Give the rivals the same input and the"),
    ("q", "f_bothwhy", "W19F", "954 input dims leave the rivals a 4-unit hidden layer"),
    ("q", "f_instr", "W19F", "triples match w16 A1 to every printed digit (`w16-maze-wm/t1/A1_*`). Only the rivals'"),
    ("q", "f_instr_t", "W19F", "on TEST n-way = 0.800 / 0.566 / 0.808 (controls 0.000 / 0.225 / 0.244 / 0.166)"),
    ("q", "f_derange", "W19F", "**`landscape_shuffled_across_wells` is a DERANGEMENT, asserted and printed:** `[1, 0]` on"),
    ("q", "f_wfseeds", "W19F", "The 8k transformer reads 0.832 / 0.569 / 0.825 against the CLU's 0.802 / 0.569 / 0.803."),
    ("q", "f_wf80k", "W19F", "The 80k column is lower (0.799 / 0.569 / 0.713). More steps do not help the rival here,"),
    ("q", "f_seed1cap", "W19F", "Seed 1 caps every strong arm at **0.5689** (CLU, rivals, 10×-float diagnostics) on"),
    ("q", "f_datacap", "W19F", "VAL: a data ceiling of that seed's hold-out, not a property of the store."),
    ("q", "f_sb1", "W19F", "**SB1 FAIL.** Margin +0.0988 < 0.10, and the every-seed clause fails on seed 1."),
    ("q", "f_s1lstm", "W19F", "the best matched rival is lstm at 8k, 0.5674 > CLU 0.5660"),
    ("q", "f_loss", "W19F", "**→ LOSS by the pre-registered rule.**"),
    ("q", "f_c1", "W19F", "The BOTH rivals are matched to 4,584 floats on a 954-dim input, which leaves a 4-unit"),
    ("q", "f_c2", "W19F", "The 10×-read-floats transformer passes the CLU on seed 0 (0.830 vs 0.800)."),
    # BOTH TEST per-arm table (FINDINGS §3)
    ("q", "bt_n", "W19F", "Unseen n-way, `exact`, n = 761 / 742 / 686:"),
    ("row", "bt_clu", "W19F", 107),
    ("row", "bt_mlp8", "W19F", 108),
    ("row", "bt_lstm8", "W19F", 109),
    ("row", "bt_tf8", "W19F", 110),
    ("row", "bt_mlp80", "W19F", 111),
    ("row", "bt_lstm80", "W19F", 112),
    ("row", "bt_tf80", "W19F", 113),
    ("row", "bt_diag", "W19F", 114),
    ("row", "bt_rem", "W19F", 115),
    ("row", "bt_shuf", "W19F", 116),
    ("row", "bt_acr", "W19F", 117),
    ("row", "bt_ra", "W19F", 118),
    ("row", "bt_bs", "W19F", 119),
    ("row", "bt_bc", "W19F", 120),
    ("row", "bt_words", "W19F", 121),
    # other rows, labelled (TABLES §2)
    ("row", "o_hdr", "W19T", 22),
] + [("row", f"o_{i}", "W19T", i) for i in range(24, 42)] + [
    # per-step / rollout (TABLES §3)
    ("row", "p_hdr", "W19T", 45),
] + [("row", f"p_{i}", "W19T", i) for i in range(47, 53)] + [
    ("q", "p_lim", "W19T", "⚠ Limitation: per-step error compounds over ~40 steps (0.93^40 ≈ 0.05), so it is ~0 for every learned arm."),
    # ledger (TABLES §4), with the convention IN the table
    ("q", "l_conv", "W19T", "Convention: deployed bytes = 4 × (atom floats + address-book floats), float32; the float64 sufficient statistic is printed beside, never netted, never matched. In OBS the extractor floats are ADDED to the matched target (declared)."),
    ("row", "l_hdr", "W19T", 60),
] + [("row", f"l_{i}", "W19T", i) for i in range(62, 67)] + [
    # deletion (TABLES §6)
    ("row", "d_hdr", "W19T", 912),
] + [("row", f"d_{i}", "W19T", i) for i in range(914, 920)] + [
    ("q", "d_note", "W19T", "Frozen codec, state encoder and (OBS) word extractor are declared OUTSIDE the deletable object."),
    ("q", "d_adapt", "W19T", "CLU post-deletion adaptation is light and data-free (A, Mx, S_k, t_k, n_k downdated; the same closed form re-solved) — not 'no changes'."),
    ("q", "d_led", "W19F", "34,320 B statistic (f64). WORDS-MAIN (main-only head): 1,104 → 4,416 B."),
    ("q", "d_led2", "W19F", "A1: 4,584 read floats → **18,336 B deployed (f32)**"),
    ("q", "d_notmargin", "W19F", "Deletion is a property of the CLU's object; it is not a composition margin. It still stands."),
    # extractor accuracy (FINDINGS §4)
    ("q", "x_decl", "W19F", "Declared extractor: 636 → 32 → 27, 6,000 steps, TRAIN (seen) rows only, 21,275 floats."),
    ("row", "x_hdr", "W19F", 156),
] + [("row", f"x_{i}", "W19F", i) for i in range(158, 163)] + [
    ("q", "x10_train", "W19F", "TRAIN 0.483 / 0.847 / 0.922."),
    ("q", "x10_nway", "W19F", "n-way 0.148 / 0.296 / 0.328."),
    ("q", "x_link", "W19F", "**Named link: READ (the"),
    ("q", "x_nochan", "W19F", "(`+extractor`) do no better than on the raw observation (0.452 vs 0.457 at 80k), so the"),
    # budget (FINDINGS §6)
    ("row", "b_hdr", "W19F", 198),
    ("row", "b_13", "W19F", 200),
    ("row", "b_49", "W19F", 201),
    ("q", "nr_c512", "W19F", "**Config C-512 was not re-run** under any condition."),
    ("q", "nr_c512b", "W19F", "declared not-run, not a loss."),
    ("q", "nr_del", "W19F", "Deletion rows on VAL legs and on non-scored conditions (`DELETION: DECLARED_NOT_RUN` in"),
    ("q", "nr_gpu", "W19F", "GPU; any seed beyond 3; any edit to w15/w16 modules (imported only)."),
    # PREREG + AMENDMENT 1
    ("q", "pr_full", "W19P", "Otherwise → LOSS row. Every condition is printed in the condition table whatever its outcome."),
    ("q", "pr_notune", "W19P", "does not consume a run. ⛔ No re-tuning to rescue a condition."),
    ("q", "pr_proof", "W19P", "WORDS-FULL: a matched rival given the CLU's exact 318-wide design reaching the CLU on `nway`"),
    ("q", "pr_proof2", "W19P", "(the pair block, not the store, carries the margin)."),
    ("q", "pr_a1", "W19P", "## AMENDMENT 1 — 2026-09-23 ~17:40Z, during the OBS VAL runs, BEFORE any OBS row number exists"),
    ("q", "pr_a1bar", "W19P", "* ⛔ **The OBS verdict (VB1–VB4) is computed on the DECLARED extractor only.** The X10 column"),
    ("q", "pr_a1bar2", "W19P", "is printed beside it and cannot pass the condition or earn a scored run. Nothing else moves."),
    ("q", "pr_obs634", "W19P", "(a) raw obs 634 = conv-AE patch+global ⊕ 29 dyn ⊕ action"),
    ("q", "f_obs636", "W19F", "The observation is used unchanged (636 = 600 + 31 + 5)."),
    ("q", "f_c512why", "W19P", "(C-512 is NOT re-run: it is A1's replication and the brief"),
    # persistence
    ("q", "ps_384", "W19F", "`persist/<cond>_seed<N>/<leg>/` holds 384 objects, each with a `<file>.sha256` (param hash +"),
    ("q", "ps_manifest", "W19F", "file hash) and `MANIFEST.json` (rebuild args):"),
    ("q", "ps_ok", "W19F", "All 384 file hashes re-verified: 0 mismatches."),
    ("q", "ps_path", "W19F", "The `persist_dir` field inside each result JSON names the pre-move path"),
    ("q", "w19_end", "W19E", "2026-09-23T20:21:58Z"),
    # ---- the drafts AFTER pass 3 (both files)
    ("q", "dn_mf2", "DNOW", "the statistic alone is 23.5121 times"),
    ("q", "ds_mf2", "DSTY", "the statistic alone is 23.5121 times"),
    ("q", "dn_k5", "DNOW", "$n=2$: 60 corners; $n\\ge3$: the identical 120 corners. \\emph{Inference note, as for"),
    ("q", "dn_k5b", "DNOW", "depth, so the store's own rise from $n=2$ to $n=6$ is not to be inferred.}"),
    ("q", "dn_cap", "DNOW", "wins: a matched transformer given the store's own pair design matches or beats the store, so the"),
    ("q", "dn_cap2", "DNOW", "under this asymmetric input, never an unqualified win. Config A1:"),
    ("q", "dn_nway", "DNOW", "\\textbf{STORE row under asymmetric inputs only} (the margin does not survive symmetric inputs, Table~\\ref{tab:cmazet-sym})"),
    ("q", "dn_sec", "DNOW", "receives the identical input, has closed with no condition winning (Table~\\ref{tab:cmazet-sym},"),
    ("q", "dn_stand", "DNOW", "What stands on CMAZE-T is the store gate (every landscape control collapses) and exact deletion,"),
    ("q", "dn_symcap", "DNOW", "\\caption{\\textbf{CMAZE-T with symmetric inputs: pending.}"),
    ("q", "dn_w20", "DNOW", "\\PLACEHOLDER{w20 \\texttt{shapes-data-matched}: the one data-matched rival set that replaces both"),
    ("q", "dn_ph_wf", "DNOW", "WORDS-FULL & 3 & \\PLACEHOLDER{w19 \\texttt{maze-t-symmetric} WORDS-FULL:"),
    ("q", "cab_mf2", "CAB", "its statistic alone is 23.5117 times"),
    # ---- the Hub's corrected report
    ("q", "w16now_corr", "W16NOW", "⚠ **CORRECTED 2026-09-23 against the primary record** (`w16-maze-wm/`) by `w18/seeds-and-tables` pass 2."),
    ("q", "w16now_eight", "W16NOW", "The Hub's first version of this section paraphrased and got eight things wrong; **the artefact wins in every one.**"),
]


def lines_of(alias):
    rel = SRC3[alias]
    p = Path(rel) if rel.startswith("/") else EV / rel
    return p.read_text(encoding="utf-8").splitlines()


cells = {}
for e in SPEC:
    kind, cid, alias = e[0], e[1], e[2]
    L = lines_of(alias)
    if kind == "row":
        ln = e[3]
        row = L[ln - 1].strip()
        assert row.startswith("|") and row.endswith("|"), (cid, row)
        cells[cid] = (alias, ln, row[1:-1].strip())
    elif kind == "q":
        hits = [i + 1 for i, t in enumerate(L) if e[3] in t]
        assert len(hits) == 1, (cid, hits)
        cells[cid] = (alias, hits[0], e[3])
    else:
        assert e[3] in L[e[4] - 1], cid
        cells[cid] = (alias, e[4], e[3])
    if alias.startswith("W19"):
        cells["y_" + cid] = cells.pop(cid)

out = HERE / "cells_pass3.py"
out.write_text(
    '"""Pass-3 cell register (static; generated once by gen_cells_pass3.py, then re-verified by every build).\n\n'
    "w19 `maze-t-symmetric` (its own tables, never a w16 row), the live drafts after pass 3, and the Hub's\n"
    'corrected w16 report. Each cell is (source alias, 1-indexed line, value)."""\n\n'
    "SRC3 = " + pprint.pformat(SRC3, width=110, sort_dicts=False) + "\n\n"
    "CELLS3 = " + pprint.pformat(cells, width=110, sort_dicts=False) + "\n",
    encoding="utf-8",
)
print(f"wrote {out}: {len(cells)} cells")
