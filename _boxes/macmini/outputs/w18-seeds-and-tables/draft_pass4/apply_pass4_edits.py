"""w18 spoke/seeds-and-tables, PASS 4: the Head's DELETION-RULE-AMENDMENT (2026-09-23 22:05), in both drafts.

Guarded, idempotent edits to DRAFT-v2.tex and DRAFT-v2-style.tex. Five items:
  DEL   (item 1) "Forget colour c" = downdate the store + downdate the ADDRESS BOOK's statistics + punch
        the hole. tab:del prints four measurements per scope: (a) store == refit-without, RESULT;
        (b) address-book statistics == refit-without, RESULT; (c) requests for c refused via the hole,
        DISPLAY; (d) composition on the other values unchanged, RESULT. (b) is in no artefact on batmac
        (w20's hub is measuring it), nor is a K* hole for the value red: those cells are NAMED
        \\PLACEHOLDERs, added identically to both drafts. No number is invented.
  RET   (item 2) "red still composes after deleting red" leaves every table and figure (and the one
        prose hit, with the probe sentence that restates it). The evidence stays in the artefacts.
  BARD  (item 3) BAR D leaves the main text; one appendix line: the GPU leg is MIXED, the CPU leg is
        UNMEASURED after the scorer defect (N359). The unfreezing decomposition (A', A'') stays.
  (item 4) No draft TABLE carries a w15 harness smoke control column: nothing to edit (FINDINGS).
  KPL   (item 5) The K-curve tables print K = 5...32 as a plateau with the byte minimum near K = 24.

Refuses to write a file if any anchor is missing or ambiguous, if the pre-existing ordered
\\PLACEHOLDER{} sequence changes (the only change allowed is the insertion of the NEW list below, in
order, in tab:del), or if a retired phrase survives. Usage: python apply_pass4_edits.py DRAFT.tex [...]
"""
import re
import sys

SD = "% SRC: w15-single-design-seeds/FINDINGS.md"
W13 = "% SRC: w13-seeds-and-tables/TABLES.md"
XS = "% SRC: w15-crossover-seeds/CROSSOVER-3SEED.md"

NEW_PLACEHOLDERS = [
    "w20 matched deletion, measurement (b) on the value red: the address book's statistics after the "
    "address-book downdate against those refit without the red rows, on the store's fit pool, seeds 0, "
    "1 and 2, with measurement (d) re-read with the address book downdated rather than carried; may "
    "land as a negative",
    "w20 matched deletion, measurement (b) on the red sphere: as for the value red, seeds 0, 1 and 2; "
    "may land as a negative",
    "w20 matched deletion, measurement (b) on the random 1\\% of items: as for the value red, seeds 0, "
    "1 and 2; may land as a negative",
    "w20 matched deletion, measurement (c) on the value red: a hole for the value red on the "
    "$\\Kstar$ object; the holes punched there are red $\\wedge$ cube and the red sphere, and none for "
    "the value red is in the artefacts",
]
P = ["\\PLACEHOLDER{" + p + "}" for p in NEW_PLACEHOLDERS]

DEL_BLOCK_OLD = (
    "item: random 1\\%, seed 2 & 1 & 1,311 & 0.3837 & 0 & 18.649 & 129,761 & \\sci{3.162}{-08} & 0.0000 (109) & 0.18117547035211387 \\\\\n"
    "% ENDSRC\n"
    "\\bottomrule\n")
DEL_BLOCK_NEW = (
    "item: random 1\\%, seed 2 & 1 & 1,311 & 0.3837 & 0 & 18.649 & 129,761 & \\sci{3.162}{-08} & 0.0000 (109) & 0.18117547035211387 \\\\\n"
    "% ENDSRC\n"
    "\\midrule\n"
    "\\multicolumn{10}{l}{\\textbf{(b) address-book statistics $=$ refit-without}: a result; the address-book downdate is new} \\\\\n"
    "value: red & 3 & \\multicolumn{8}{p{12cm}}{" + P[0] + "} \\\\\n"
    "conj.: red sphere & 3 & \\multicolumn{8}{p{12cm}}{" + P[1] + "} \\\\\n"
    "item: random 1\\% & 3 & \\multicolumn{8}{p{12cm}}{" + P[2] + "} \\\\\n"
    "\\midrule\n"
    "\\multicolumn{10}{l}{\\textbf{(c) requests for $c$ refused through the hole}: a \\textsc{display} row, never a result; the hole is a flag} \\\\\n"
    "value: red & 3 & \\multicolumn{8}{p{12cm}}{" + P[3] + "} \\\\\n"
    "conj.: red sphere & 3 & \\multicolumn{8}{l}{hole punched: 2 B rewritten for two holes (red $\\wedge$ cube and the red sphere) in the 1,322 B coupling table, 0 rows read, no refit, every seed} \\\\ " + SD + "\n"
    "item: random 1\\% & 3 & \\multicolumn{8}{l}{no hole: an item deletion makes no request impossible (0 of 164 on the comparator's 16,384-row pool, every seed)} \\\\ % SRC: w12-deletion-baselines/FINDINGS.md\n"
    "\\bottomrule\n")

COMMON = [
    # ---------------- pass 4b migrations (the one appendix line is the only Criterion D mention left)
    ("MIG-bard-caption",
     "row 4); store: the \\texttt{pca\\_tensor} atom block (524,288 B, float32). Criterion D is not a row % SRC: w15-bar-d-score/FINDINGS.md\n"
     "of this table (one line, above).}\n",
     "row 4); store: the \\texttt{pca\\_tensor} atom block (524,288 B, float32).} % SRC: w15-bar-d-score/FINDINGS.md\n"),
    ("BARD-main-label",
     "pre-registration (for example, criterion D or criterion S1).\n",
     "pre-registration (for example, criterion S1 or criterion S2).\n"),
    # ---------------- item 1: tab:del, four measurements
    ("DEL-caption",
     "\\caption{\\textbf{Exact (store-scoped) deletion on the single design, $\\Kstar=32$, TEST, seeds 0, 1 " + W13 + "\n"
     "and 2, one row per seed ($n=1$ each).} ``Exactness'' is the store's maximum absolute mode deviation against the store refit\n"
     "without the rows. Survivor $\\Delta$ is the paired composition change on corners that do not\n"
     "request the deleted scope. Seeds 1 and 2",
     "\\caption{\\textbf{Exact (store-scoped) deletion on the single design, $\\Kstar=32$, TEST, fit pool 131,072 rows, seeds 0, 1 " + W13 + "\n"
     "and 2, one row per seed ($n=1$ each): four measurements per scope.} Forgetting $c$ is: downdate the\n"
     "store, downdate the address book's statistics, punch the hole. \\textbf{(a) A result}: the store\n"
     "equals the store refit without the rows (``exactness'', the store's maximum absolute mode deviation\n"
     "against it). \\textbf{(b) A result}: the address book's statistics equal those refit without the\n"
     "rows; the address-book downdate is new, and every (b) cell is a named placeholder.\n"
     "\\textbf{(c) A display, never a result}: requests for $c$ refused through the hole, a flag in the\n"
     "coupling table; symbolic refusal is not claimed. \\textbf{(d) A result}: survivor $\\Delta$, the\n"
     "paired composition change on corners that do not request the deleted scope, measured with the\n"
     "address book carried across the basis change, not downdated. Seeds 1 and 2"),
    ("DEL-header",
     "scope & seeds & rows del. & store s & store rows read & ref. refit s & ref. rows read & exactness & survivor $\\Delta$ & cos$_{\\min}$ \\\\\n",
     "scope & seeds & rows del. & store s & store rows read & ref. refit s & ref. rows read & (a) exactness & (d) survivor $\\Delta$ & cos$_{\\min}$ \\\\\n"),
    ("DEL-blocks", DEL_BLOCK_OLD, DEL_BLOCK_NEW),
    # ---------------- item 2: retire "red still composes after deleting red"
    ("RET-paragraph",
     "\\paragraph{Deletion is not capability erasure, at three seeds.} After every red row is deleted,\n"
     "the 20 deleted (red-requesting) corners still compose red on all three seeds: 32.250 $\\to$ 32.250 " + SD + "\n"
     "novel valid tuples per corner (judge accuracy 0.8958 $\\to$ 0.8958) on seed 0, 33.500 $\\to$ 33.500 " + SD + "\n"
     "(0.9306 $\\to$ 0.9306) on seed 1, and 33.600 $\\to$ 33.550 (0.9333 $\\to$ 0.9319) on seed 2. " + SD + "\n"
     "The deletion claim on this object is store-scoped exactness plus a data-free adaptation cost,\n"
     "not capability erasure. Removing the capability is the hole's job (\\S\\ref{sec:hole}), and the\n"
     "hole is a display row with a declared not-run comparator, so it is not claimed.\n"
     "\n",
     ""),
    ("RET-figcaption",
     "\\figslot{fig_deletion}{\\textbf{Deletion on the single design at $\\Kstar$.} Post-deletion composition and adaptation cost per arm",
     "\\figslot{fig_deletion}{\\textbf{Deletion on the single design at $\\Kstar$.} Survivor composition before and after deletion (measurement (d) of Table~\\ref{tab:del}) and adaptation cost per arm"),
    # ---------------- item 3: BAR D out of the main text; one appendix line
    ("BARD-main",
     "Training the encoder through the store licenses no store claim. Pre-registered criterion D, whether\n"
     "the store's margin over a matched learner grows, is mixed on the longer leg, with no claim in either\n"
     "direction, and unmeasured on the shortened leg, whose learner rows are a rival fitted on the frozen\n"
     "encoder's features; the swap loss helps on both legs, and ``unfreezing hurts'' does not survive the\n"
     "longer leg (Appendices~\\ref{app:encoder-main} and~\\ref{app:encloop}).\n",
     "Training the encoder through the store licenses no store claim: the swap loss helps on both legs,\n"
     "and ``unfreezing hurts'' does not survive the longer leg (Appendices~\\ref{app:encoder-main}\n"
     "and~\\ref{app:encloop}).\n"),
    ("BARD-limits",
     "fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no rival; " + W13 + "\n"
     "  criterion D, growth of the store's margin over a matched learner, is mixed on the longer leg and\n"
     "  unmeasured on the shortened one (\\S\\ref{sec:bard}).\n",
     "fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no rival " + W13 + "\n"
     "  (Appendix~\\ref{app:encloop}).\n"),
    ("BARD-encloop-row",
     "margin over the frozen-feature rival (not a measurement of the refitted learner), alone $\\to$ arm & 3 & +12.078 $\\to$ +10.311 & +24.687 $\\to$ +26.048 \\\\\n",
     ""),
    ("BARD-encloop-text",
     "row 3. The pre-declared success bar, growth of the store's margin over the matched learner, is not\n"
     "assessed on this leg: the learner rows its scorer loaded are the frozen-feature rival (below), which\n"
     "sits at or within 0.02 of zero (0.006 on row 3), so criterion D on this leg is unmeasured.\n",
     "row 3.\n"),
    ("BARD-one-line", None, None),  # filled below: the scored-leg paragraph becomes one line
    ("BARD-table",
     "\\caption{\\textbf{Criterion D and the decomposition on the longer leg only (4,000 joint steps, % SRC: w15-bar-d-score/FINDINGS.md\n",
     "\\caption{\\textbf{The unfreezing decomposition on the longer leg only (4,000 joint steps, % SRC: w15-bar-d-score/FINDINGS.md\n"),
    ("BARD-table-caption2",
     "row 4); store: the \\texttt{pca\\_tensor} atom block (524,288 B, float32); learner: the % SRC: w15-bar-d-score/FINDINGS.md\n"
     "matched-parameter MLP refitted on each arm's features (178,952 parameters). Paired intervals are % SRC: w15-bar-d-score/FINDINGS.md\n"
     "10,000-draw bootstraps over corners.} % SRC: w15-bar-d-score/FINDINGS.md\n",
     "row 4); store: the \\texttt{pca\\_tensor} atom block (524,288 B, float32).} % SRC: w15-bar-d-score/FINDINGS.md\n"),
    ("BARD-table-rows",
     "% SRCBLOCK: w15-bar-d-score/FINDINGS.md\n"
     "$M(\\texttt{alone})$, store $-$ learner & 3 & $-0.517$ / +0.867 / $-2.783$ & +0.918 / $-0.061$ / $-0.837$ \\\\\n"
     "$M(\\texttt{loop\\_swap})$ & 3 & +0.217 / $-0.017$ / +1.183 & +0.000 / +0.286 / +0.082 \\\\\n"
     "growth & 3 & +0.733 / $-0.883$ / +3.967 & $-0.918$ / +0.347 / +0.918 \\\\\n"
     "growth, CI95 & 3 & [$-0.950$, +2.450] / [$-2.133$, +0.400] / [+1.983, +6.117] & [$-3.245$, +1.694] / [$-1.163$, +1.898] / [$-2.163$, +4.000] \\\\\n"
     "\\textbf{Criterion D (pre-declared rule)} & 3 & \\textsc{mixed}: no claim & \\textsc{mixed}: no claim \\\\\n"
     "store, \\texttt{alone} / \\texttt{loop\\_swap} & 3 & 14.650 / 30.783 / 16.883; 35.817 / 35.767 / 35.650 & 38.531 / 64.449 / 39.286; 72.000 / 71.633 / 71.878 \\\\\n"
     "learner, \\texttt{alone} / \\texttt{loop\\_swap} & 3 & 15.167 / 29.917 / 19.667; 35.600 / 35.783 / 34.467 & 37.612 / 64.510 / 40.122; 72.000 / 71.347 / 71.796 \\\\\n"
     "three landscape controls & 3 & \\multicolumn{2}{c}{collapse on every scored leg, both rows (at most 0.490 per corner; bar $\\le1.0$)} \\\\\n"
     "\\midrule\n"
     "$A'$",
     "% SRCBLOCK: w15-bar-d-score/FINDINGS.md\n"
     "$A'$"),
    ("BARD-encoder-main-1",
     "of them at or below 0.0003). On this leg the store's margin\n"
     "over the matched-parameter learner is not measured: the printed $+12.078\\to+10.311$ (row 3) and\n"
     "$+24.687\\to+26.048$ (row 4) are margins over the frozen-feature rival, not over the refitted learner\n"
     "(Appendix~\\ref{app:encloop}), so criterion D here is unmeasured, and on the longer leg, where it was\n"
     "measured, it is mixed with no claim either way. This is a finding about the encoder, and the claim\n",
     "of them at or below 0.0003). This is a finding about the encoder, and the claim\n"),
    ("BARD-encoder-main-2",
     "support is that the swap loss helps. Criterion D is \\textsc{mixed} on both rows, so the margin\n"
     "does not grow in the pre-declared sense and no claim is made either way; the features improved\n"
     "for every arm, \\texttt{loop\\_swap} lifting the matched learner as much as the store, both ending at\n"
     "0.99--1.00 of the 72-tuple ceiling, which is a finding about the encoder, not about the store. % SRC: w15-bar-d-score/FINDINGS.md\n",
     "support is that the swap loss helps.\n"),
    ("BARD-index",
     "Criterion D and its decomposition paragraph (Appendices~\\ref{app:encloop}\n"
     "and~\\ref{app:encoder-main}, Table~\\ref{tab:bard-gpu}); and the coverage control\n",
     "the decomposition paragraph (Appendix~\\ref{app:encoder-main}, Table~\\ref{tab:bard-gpu}); and the\n"
     "coverage control\n"),
    ("DEL-index",
     "Eleven rows remain pending, each marked in the text by a red \\textsc{placeholder} whose argument names\n"
     "the row and states that it may land as a negative or not at all, and each awaits a further run: the\n"
     "data-matched rival set that replaces both rival sets of \\S\\ref{sec:shapes-t7} (one); the\n",
     "Fifteen rows remain pending, each marked in the text by a red \\textsc{placeholder} whose argument names\n"
     "the row and states that it may land as a negative or not at all, and each awaits a further run: the\n"
     "data-matched rival set that replaces both rival sets of \\S\\ref{sec:shapes-t7} (one); the new\n"
     "deletion cells of Table~\\ref{tab:del}, measurement (b) on each scope and measurement (c) for the\n"
     "value red, which the matched deletion measures (four); the\n"),
    # ---------------- item 5: the K-curve tables print the plateau
    ("KPL-kcurve",
     "honest four-part total is U-shaped.}\n"
     "\\label{tab:kcurve}\n",
     "honest four-part total is U-shaped. The $n=1$ middle rows are superseded by Table~\\ref{tab:kcurve3},\n"
     "which prints $K=5\\ldots32$ as a plateau, not a selection. " + XS + "\n"
     "}\n"
     "\\label{tab:kcurve}\n"),
    ("KPL-kcurve3",
     "reference, not a budget. The seed sd is printed beside every mean. $K=20$ fails the strict control gate (text).} " + XS + "\n",
     "reference, not a budget. The seed sd is printed beside every mean. $K=20$ fails the strict control gate (text). " + XS + "\n"
     "\\textbf{$K=5\\ldots32$ is a plateau, not a $\\Kstar$ selection}: three seeds do not resolve the validation " + XS + "\n"
     "composition across it, $\\Kstar=32$ is what the frozen rule returns, and the byte minimum is near $K=24$.} " + XS + "\n"),
]

BARD_ONE_LINE_START = "\\paragraph{Criterion D on the longer leg, scored.} One\n"
BARD_ONE_LINE_END = ("re-scoring it (its remaining pre-registered run) is a declared not-run. "
                     "% SRC: w15-bar-d-score/FINDINGS.md\n")
BARD_ONE_LINE = ("Criterion D, the pre-registered success bar, in one line: the GPU (longer, full-decoder) leg is "
                 "\\textsc{mixed}; the CPU (shortened-decoder) leg is \\textsc{unmeasured} after the scorer defect "
                 "(registry N359). % SRC: w15-bar-d-score/FINDINGS.md; w15-advisor-report.md\n")

PER_FILE = {
    "DRAFT-v2.tex": [
        ("DEL-definition",
         "a deletion result. \\emph{Item}: a random 1\\% of fit rows. Table~\\ref{tab:del} gives each with\n"
         "its cost. Post-deletion adaptation of the store is \\emph{light and data-free}: the covariance\n"
         "and count are downdated by subtraction, the atoms are re-eigendecomposed from the downdated\n"
         "statistic and the address book is carried across the basis change, with no row of data read.\n",
         "a deletion result. \\emph{Item}: a random 1\\% of fit rows. Table~\\ref{tab:del} gives each with\n"
         "its cost. \\emph{Forgetting a value $c$} is three operations on one object: downdate the store,\n"
         "downdate the address book's statistics, and punch the hole for $c$ (\\S\\ref{sec:hole}). Every\n"
         "deletion row therefore prints four measurements: (a) the store equals the store refit without the\n"
         "rows, a result; (b) the address book's statistics equal those refit without the rows, a result;\n"
         "(c) requests for $c$ are refused through the hole, a display row only, because the hole is a flag\n"
         "and symbolic refusal is not claimed; and (d) composition on the other values is unchanged, a\n"
         "result. Post-deletion adaptation of the store is \\emph{light and data-free}: the covariance\n"
         "and count are downdated by subtraction and the atoms are re-eigendecomposed from the downdated\n"
         "statistic, with no row of data read. The (a) and (d) rows printed here were measured with the\n"
         "address book carried across the basis change, not downdated; the address-book downdate is new,\n"
         "and its cells are named placeholders in Table~\\ref{tab:del}.\n"),
        ("RET-probe",
         "\\paragraph{The probe that bounds the claim.} Exactness is store-scoped in a way that is\n"
         "measurable, not only definitional. Deleting all 11,044 red rows leaves the 20 red-requesting\n"
         "corners at 32.2500 novel valid tuples per corner before and after, and judge accuracy at\n"
         "0.8958 before and after. " + W13 + "\n",
         "\\paragraph{The probe that bounds the claim.} Exactness is store-scoped in a way that is\n"
         "measurable, not only definitional.\n"),
        ("RET-emph",
         "\\emph{``The store has forgotten red'' and ``the object no longer answers red'' are different\n"
         "statements, and at $\\Kstar$ only the first is demonstrated.} At $\\Kstar$ the value identity\n"
         "lives in the address book, which the deletion carries and does not refit. The trade-off moved\n",
         "At $\\Kstar$ the value identity\n"
         "lives in the address book, which the carry rotates and does not refit; this is why the deletion\n"
         "now also downdates the address book's statistics (measurement (b) of Table~\\ref{tab:del}). The trade-off moved\n"),
        ("RET-hole",
         "Because the downdate does not stop the object answering red (the probe above), the hole is the\n"
         "only mechanism on this object that can. An item deletion makes nothing impossible: 0 of 164\n",
         "The hole is the only mechanism on this object that refuses, and it is a flag: a request for a\n"
         "holed value is refused by construction, so refusal is measurement (c) of Table~\\ref{tab:del}, a\n"
         "display row, and symbolic refusal is not claimed. An item deletion makes nothing impossible: 0 of 164\n"),
    ],
    "DRAFT-v2-style.tex": [
        ("DEL-definition",
         "Table~\\ref{tab:del} gives each deletion with its cost. Post-deletion adaptation of the store is\n"
         "\\emph{light and data-free}: the covariance and count are downdated by subtraction, the atoms are\n"
         "re-eigendecomposed from the downdated statistic and the address book is carried across the basis\n"
         "change, with no row of data read.",
         "Table~\\ref{tab:del} gives each deletion with its cost. \\emph{Forgetting a value $c$} consists of\n"
         "three operations on one object: the store is downdated, the address book's statistics are\n"
         "downdated, and the hole for $c$ is punched (\\S\\ref{sec:hole}). Each deletion row therefore prints\n"
         "four measurements: (a) the store equals the store refit without the rows, a result; (b) the\n"
         "address book's statistics equal those refit without the rows, a result; (c) requests for $c$ are\n"
         "refused through the hole, only a display row, since the hole is a flag and symbolic refusal is\n"
         "not claimed; and (d) composition on the other values is unchanged, a result. Post-deletion\n"
         "adaptation of the store is \\emph{light and data-free}: the covariance and count are downdated by\n"
         "subtraction and the atoms are re-eigendecomposed from the downdated statistic, with no row of data\n"
         "read. The (a) and (d) rows printed here were measured with the address book carried across the\n"
         "basis change rather than downdated; the address-book downdate is new, and its cells are named\n"
         "placeholders in Table~\\ref{tab:del}."),
        ("RET-probe",
         "\\paragraph{The probe that bounds the claim.} The store-scoping of exactness is measurable, not\n"
         "only definitional. Deleting all 11,044 red rows leaves the 20 red-requesting corners at 32.2500\n"
         "novel valid tuples per corner before and after, and judge accuracy at 0.8958 before and after.\n"
         + W13 + "\n",
         "\\paragraph{The probe that bounds the claim.} The store-scoping of exactness is measurable, not\n"
         "only definitional.\n"),
        ("RET-emph",
         "\\emph{``The store has forgotten red'' and ``the object no longer answers red'' are different\n"
         "statements, and at $\\Kstar$ only the first is demonstrated.} At $\\Kstar$ the value identity\n"
         "resides in the address book, which the deletion carries rather than refits. The trade-off moved\n",
         "At $\\Kstar$ the value identity\n"
         "resides in the address book, which the carry rotates rather than refits; this is why the deletion\n"
         "now also downdates the address book's statistics (measurement (b) of Table~\\ref{tab:del}). The trade-off moved\n"),
        ("RET-hole",
         "Because the downdate does not stop the object answering red (the probe above), the hole is the\n"
         "only mechanism on this object that can. An item deletion, in contrast, makes nothing impossible:\n",
         "The hole is the only mechanism on this object that refuses, and it is a flag: a request for a\n"
         "holed value is refused by construction, so refusal is measurement (c) of Table~\\ref{tab:del}, a\n"
         "display row, and symbolic refusal is not claimed. An item deletion, in contrast, makes nothing impossible:\n"),
    ],
}

RETIRED = [r"criterion D or criterion S1", r"Criterion D is not a row", r"still compose", r"capability erasure", r"answering red", r"no longer answers red",
           r"Criterion D on the longer leg", r"margin over the frozen-feature rival"]


def placeholders(text):
    out, i = [], 0
    while True:
        j = text.find("\\PLACEHOLDER{", i)
        if j < 0:
            return out
        k, depth = j + len("\\PLACEHOLDER{"), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j:k])
        i = k


def apply(path):
    name = path.rsplit("/", 1)[-1]
    text = open(path, encoding="utf-8").read()
    before_ph = [p for p in placeholders(text) if p not in P]
    edits = []
    for tag, old, new in COMMON + PER_FILE[name]:
        if tag == "BARD-one-line":
            a, b = text.count(BARD_ONE_LINE_START), text.count(BARD_ONE_LINE_END)
            if a == 1 and b == 1:
                s = text.index(BARD_ONE_LINE_START)
                e = text.index(BARD_ONE_LINE_END) + len(BARD_ONE_LINE_END)
                if e <= s:
                    raise SystemExit(f"{name}: {tag}: end before start")
                text = text[:s] + BARD_ONE_LINE + text[e:]
                edits.append(tag)
            elif a == 0 and b == 0 and text.count(BARD_ONE_LINE) == 1:
                edits.append(tag + " (already applied)")
            else:
                raise SystemExit(f"{name}: {tag}: start {a}, end {b}: refusing")
            continue
        n_old, n_new = text.count(old), (text.count(new) if new else 0)
        if tag.startswith("MIG-") and n_old == 0:
            edits.append(tag + " (nothing to migrate)")
            continue
        if n_old == 1:
            text = text.replace(old, new)
            edits.append(tag)
        elif n_old == 0 and (new == "" or n_new >= 1):
            edits.append(tag + " (already applied)")
        else:
            raise SystemExit(f"{name}: {tag}: anchor found {n_old} times: refusing")
    after_all = placeholders(text)
    after_old = [p for p in after_all if p not in P]
    new_seen = [p for p in after_all if p in P]
    if after_old != before_ph:
        raise SystemExit(f"{name}: pre-existing \\PLACEHOLDER sequence changed: refusing")
    if new_seen != P:
        raise SystemExit(f"{name}: new placeholders not exactly the declared four, in order: refusing")
    for r in RETIRED:
        if re.search(r, text):
            raise SystemExit(f"{name}: retired phrase survives: {r!r}: refusing")
    open(path, "w", encoding="utf-8").write(text)
    print(f"{name}: {len(edits)} edits: " + ", ".join(edits))
    print(f"{name}: pre-existing PLACEHOLDERs {len(after_old)} unchanged; new {len(new_seen)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        apply(p)
