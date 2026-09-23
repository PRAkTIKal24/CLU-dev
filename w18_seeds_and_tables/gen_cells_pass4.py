"""Generate cells_pass4.py once (kept for provenance; build_tables.py does not run it).

Pass 4 = the Head's DELETION-RULE-AMENDMENT (2026-09-23 22:05). Every cell is (alias, line, value), and the
build re-verifies that the value is verbatim on that line. The per-seed deletion rows are read from
w15-single-design-seeds/T7-3SEEDS.md l.85-96 (one table, all three seeds, all four scopes), split on '|'.
"""
import pprint
from pathlib import Path

EV = Path("/Users/batmac/clu/evidence/outputs")
HERE = Path(__file__).resolve().parent

SRC4 = {
    "AMD": "@here/draft_pass4/DELETION-RULE-AMENDMENT.md",
    "D4V": "/Users/batmac/clu/outputs/w17-paper/DRAFT-v2.tex",
    "D4S": "/Users/batmac/clu/outputs/w17-paper/DRAFT-v2-style.tex",
    "BDF": "w15-bar-d-score/FINDINGS.md",
    "W15A": "w15-advisor-report.md",
}

cells = {}
t7 = (EV / "w15-single-design-seeds/T7-3SEEDS.md").read_text(encoding="utf-8").splitlines()
tag = {"`value`": "v", "`conjunction`": "c", "`conjunction_live`": "cl", "`item`": "i"}
for ln in range(85, 97):
    f = [x.strip() for x in t7[ln - 1].strip().strip("|").split("|")]
    scope, seed, rows, surv, ba, d, ci, ex, ranks, secs, byts, read = f
    k = f"p4_{tag[scope]}{seed}"
    for suffix, val in (("rows", rows), ("surv", surv), ("ba", ba), ("d", d), ("ex", ex),
                        ("s", secs), ("b", byts)):
        cells[f"{k}_{suffix}"] = ("SD", ln, val)

cells.update({
    # the amendment itself (pinned copy)
    "a_def": ("AMD", 6, "**\"Forget colour c\"** = **downdate the store** + **downdate the ADDRESS BOOK's statistics** + **punch the hole**."),
    "a_scope": ("AMD", 7, "For **value c**, **c ∧ sphere**, **1 % items**, on the **131,072 pool**, **3 seeds**:"),
    "a_meas": ("AMD", 8, "**(a) store == refit-without** — RESULT · **(b) address-book statistics == refit-without** — RESULT"),
    "a_c": ("AMD", 8, "**(c) requests for c REFUSED via the hole** — ⚠ **DISPLAY row** (the hole is a flag; symbolic refusal is **closed as a claim**)"),
    "a_d": ("AMD", 8, "**(d) composition on the other values unchanged** — RESULT."),
    "a_w20": ("AMD", 9, "⚠ w20's hub is measuring the matched version now. Where a cell is not yet in the artefacts, **placeholder it and name it** — ⛔ do not invent it."),
    "a_ret": ("AMD", 12, "⛔ **\"red still composes after deleting red\" DOES NOT PRINT.**"),
    "a_bard": ("AMD", 15, "*the GPU leg is MIXED; the CPU leg is UNMEASURED after the scorer defect (`N359`).*"),
    "a_smoke": ("AMD", 18, "⚠ They used the identity-prone `rng.permutation(K)` control (`N350`) and were never re-measured."),
    "a_plateau": ("AMD", 21, "⭐ **K = 5…32 plateau, with the byte minimum near K = 24.** ⛔ Not as a clean K\\* selection."),
    # what the artefacts say about the address book: CARRIED, not downdated
    "p4_carry": ("SDF", 189, "the address book is carried across the basis change, and **no row of data is read**."),
    # (c): the item scope has no hole
    "p4_item_nohole": ("DB", 91, "**0 of 164** requests built from the deleted rows are certifiably impossible"),
    # BAR D, one line
    "p4_bard_gpu": ("BDF", 16, "**BAR D-GPU = MIXED on row 4 and on row 3** by the pre-declared rule"),
    "p4_bard_cpu": ("BDF", 156, "is **unmeasured**, not \"VOID because the baseline scores 0\""),
    "p4_n359": ("W15A", 51, "**The defect (registry `N359`):** the w8 scoring path **never loaded the refit baseline**."),
    # N350: the w15 smoke control
    "p4_n350": ("W15A", 61, "registry `N350` — the shipped `landscape_shuffled_across_wells` used `rng.permutation(K)`, **which at K = 2 is the identity half the time**."),
    "p4_n350b": ("W15A", 61, "The w15 harness smoke columns used the OLD control and were never re-measured."),
    # the plateau (the byte minimum)
    "p4_xs_claim": ("XS", 85, "the honest four-part total is U-shaped in K with its minimum near K = 24."),
    # the live drafts after pass 4
    "d4_abs_plateau": ("D4V", 100, "the finest partition on a validation plateau"),
    "d4_cap": ("D4V", 695, "four measurements per scope.} Forgetting $c$ is: downdate the"),
    "d4_b_v": ("D4V", 729, "\\PLACEHOLDER{w20 matched deletion, measurement (b) on the value red:"),
    "d4_c_v": ("D4V", 734, "\\PLACEHOLDER{w20 matched deletion, measurement (c) on the value red:"),
    "d4_c_cl": ("D4V", 735, "hole punched: 2 B rewritten for two holes (red $\\wedge$ cube and the red sphere) in the 1,322 B coupling table, 0 rows read, no refit, every seed"),
    "d4_c_i": ("D4V", 736, "no hole: an item deletion makes no request impossible (0 of 164 on the comparator's 16,384-row pool, every seed)"),
    "d4_kpl": ("D4V", 1668, "\\textbf{$K=5\\ldots32$ is a plateau, not a $\\Kstar$ selection}"),
    "d4_fig": ("D4V", 1827, "Survivor composition before and after deletion (measurement (d) of Table~\\ref{tab:del})"),
    "d4_bard": ("D4V", 2014, "Criterion D, the pre-registered success bar, in one line: the GPU (longer, full-decoder) leg is \\textsc{mixed}; the CPU (shortened-decoder) leg is \\textsc{unmeasured} after the scorer defect (registry N359)."),
    "d4_label": ("D4V", 376, "(for example, criterion S1 or criterion S2)."),
    "d4_index": ("D4V", 2859, "Fifteen rows remain pending"),
    "d4s_cap": ("D4S", 696, "four measurements per scope.} Forgetting $c$ is: downdate the"),
    "d4s_b_v": ("D4S", 730, "\\PLACEHOLDER{w20 matched deletion, measurement (b) on the value red:"),
    "d4s_bard": ("D4S", 2000, "Criterion D, the pre-registered success bar, in one line:"),
    "d4s_kpl": ("D4S", 1654, "\\textbf{$K=5\\ldots32$ is a plateau, not a $\\Kstar$ selection}"),
})

out = ('"""Pass-4 cell register (static; generated once by gen_cells_pass4.py, then re-verified by every build).\n\n'
       "The Head's DELETION-RULE-AMENDMENT (2026-09-23 22:05): the four-measurement deletion table, the retired\n"
       "row, BAR D's one line, the w15 smoke controls (N350) and the K plateau; plus the drafts after pass 4.\n"
       'Each cell is (source alias, 1-indexed line, value)."""\n\n'
       f"SRC4 = {pprint.pformat(SRC4, width=110)}\n\nCELLS4 = {pprint.pformat(cells, width=110)}\n")
(HERE / "cells_pass4.py").write_text(out, encoding="utf-8")
print(len(cells), "cells")
