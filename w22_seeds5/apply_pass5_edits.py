#!/usr/bin/env python3
"""w22 seeds-and-tables PASS 5: guarded, idempotent draft edits (both drafts, identical text).

  E1  tab:ladder caption: every rival set named with its pool AND its step budget; the "third set is pending"
      sentence becomes the store-content set's description.
  E2  tab:ladder block headers: pool and steps in each header (sets A and B).
  E3  tab:ladder: the 32,768-row \\PLACEHOLDER row is FILLED from w21-ladder-rivals-32k/TABLES.md (rows copied
      verbatim from its Table P1 and Table L3 lines, never retyped).
  E4  the ladder paragraph: the store-content set's reading, the over-rung means, and the X1 single-fit variance.
  E5  the placeholder index: three remain, and they are PERMANENTLY BLOCKED (w16 persisted no arm objects).
  E6  the declared not-runs: the third set is no longer pending.
  E7  PoE's multiple 27.13 -> 27.135 (GENTABLE-KSTAR.json x_store_reference; 14,226,660 / 524,288 = 27.1350).

Refuses to write if an anchor is missing or not unique, if the \\PLACEHOLDER sequence changes other than by the
one declared removal, or if a retired phrase survives. usage: apply_pass5_edits.py IN.tex OUT.tex"""
import re
import sys
from pathlib import Path

EV = Path("/Users/batmac/clu/evidence/outputs")
W21 = EV / "w21-ladder-rivals-32k" / "TABLES.md"
SRC_T = "% SRC: w21-ladder-rivals-32k/TABLES.md"
SRC_C = "w21-ladder-rivals-32k/CAPTION.md"
SRC_F = "w21-ladder-rivals-32k/FINDINGS.md"


def w21_rows():
    """Per-seed cells for set C, copied verbatim from w21 TABLES.md (Table P1 and Table L3)."""
    lines = W21.read_text(encoding="utf-8").splitlines()
    p1, l3 = {}, {}
    for ln in lines:
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if (len(c) == 7 and re.fullmatch(r"n=[2-6]", c[0]) and c[1] in ("mlp", "lstm", "transformer", "cont_mlp")
                and re.fullmatch(r"(PASS|FAIL)( / (PASS|FAIL)){2}", c[5])):
            p1[(int(c[0][2]), c[1])] = (c[2], c[3])
        if len(c) == 8 and re.fullmatch(r"n=[2-6]", c[0]) and c[2].startswith("+") and "[**" in c[4]:
            l3[int(c[0][2])] = (c[4], c[5])
    assert len(p1) == 20 and len(l3) == 5, (len(p1), len(l3))

    def trip(cell):  # "+6.65 / +7.37 / +4.22 [**+6.08**]" -> ("+6.65 / +7.37 / +4.22", "+6.08")
        m = re.fullmatch(r"(\S+ / \S+ / \S+) \[\*\*(\S+)\*\*\]", cell)
        assert m, cell
        return m.group(1), m.group(2)
    return p1, {k: (trip(a), trip(b)) for k, (a, b) in l3.items()}


def tex_signed(s):
    return " / ".join(("$-" + x[1:] + "$") if x.startswith("-") else x for x in s.split(" / "))


def block_c():
    p1, l3 = w21_rows()
    fam = [("mlp", "MLP"), ("lstm", "LSTM"), ("transformer", "transformer"), ("cont_mlp", "continuous MLP")]
    out = [r"\multicolumn{7}{l}{\emph{store-content set: the rung's first 32,768 rows, exactly the store's content rows; 5,120 steps, ten times 51,200}} \\ % SRC: " + SRC_C,
           "% SRCBLOCK: w21-ladder-rivals-32k/TABLES.md"]
    for key, name in fam:
        for i, steps in ((0, "5,120"), (1, "51,200")):
            out.append(f"{name} ({steps}) & 3 & " + " & ".join(p1[(n, key)][i] for n in range(2, 7)) + r" \\")
    out.append(r"\textbf{store $-$ strongest, paired} & 3 & " + " & ".join(tex_signed(l3[n][0][0]) for n in range(2, 7)) + r" \\")
    out.append(r"\quad 3-seed mean & 3 & " + " & ".join(l3[n][0][1] for n in range(2, 7)) + r" \\")
    out.append(r"store $-$ strongest step-matched, paired & 3 & " + " & ".join(tex_signed(l3[n][1][0]) for n in range(2, 7)) + r" \\")
    out.append(r"\quad 3-seed mean & 3 & " + " & ".join(l3[n][1][1] for n in range(2, 7)) + r" \\")
    out.append("% ENDSRC")
    return "\n".join(out)


PH_ROW = (r"32,768-row set & 3 & \multicolumn{5}{l}{\PLACEHOLDER{ladder rival set fitted on 32,768 rows of each rung's pool, four families, 3 seeds: the column slot, pending; may land as a negative}} \\ % SRC: w20-shapes-data-matched/FINDINGS.md")
PH_ARG = "ladder rival set fitted on 32,768 rows of each rung's pool, four families, 3 seeds: the column slot, pending; may land as a negative"

EDITS = [
    ("E1", r"""combination-split pool that carries the held-out values (\S\ref{sec:shapes-t7}); a rival of that set
does not depend on the rung. A third set, fitted on 32,768 rows of each rung's pool, is pending. LACE % SRC: w20-shapes-data-matched/FINDINGS.md
""", r"""combination-split pool that carries the held-out values (\S\ref{sec:shapes-t7}), at 5,040 optimiser % SRC: """ + SRC_C + r"""
steps; a rival of that set does not depend on the rung. \emph{Store-content set}: the same four
families fitted on the rung's first 32,768 rows, exactly the store's content rows, at the selected % SRC: """ + SRC_C + r"""
schedule (5,120 steps) and at ten times it (51,200); its margin row is the store minus the strongest % SRC: """ + SRC_C + r"""
of its selected and ten-times fits, and a second margin row takes the strongest fit step-matched to the
data-matched set (20,480 steps for the LSTM and the transformer, 204,800 for the MLP and the % SRC: """ + SRC_C + r"""
continuous MLP), both paired per corner. Every header names its pool and its steps. LACE
"""),
    ("E2a", r"""\multicolumn{7}{l}{\emph{data-matched set: each rung's own 131,072 rows}} \\ % SRC: w20-shapes-data-matched/FINDINGS.md
""", r"""\multicolumn{7}{l}{\emph{data-matched set: each rung's own 131,072 rows; 20,480 steps, and 204,800 for the MLP and the continuous MLP}} \\ % SRC: w20-shapes-data-matched/FINDINGS.md; """ + SRC_C + "\n"),
    ("E2b", r"""\multicolumn{7}{l}{\emph{earlier set: 32,256 rows of the combination-split pool}} \\ % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json
""", r"""\multicolumn{7}{l}{\emph{earlier set: 32,256 rows of the combination-split pool; 5,040 steps}} \\ % SRC: _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; """ + SRC_C + "\n"),
    ("E3", PH_ROW + "\n", None),  # filled by block_c()
    ("E4", r"""composition-depth result.
% SRC: w12b-nr6-and-kstar/FINDINGS.md
""", r"""composition-depth result.
% SRC: w12b-nr6-and-kstar/FINDINGS.md
The store-content set asks whether the data-matched set's 131,072 rows, four times the store's
32,768 content rows, favour the rivals. They do not: fitted on exactly the store's content rows, % SRC: """ + SRC_F + r"""
the rivals leave the margin where the data-matched set left it, +6.08 / +11.48 / +14.00 / +9.47 / % SRC: """ + SRC_F + r"""
+10.36 at $n=2,\ldots,6$ (mean over rungs +10.28, and +10.75 step-matched, against +9.97 for the % SRC: """ + SRC_F + r"""
data-matched set and +17.53 for the earlier set), and every 95\% interval of the store-content % SRC: """ + SRC_F + r"""
margins lies above zero (lowest lower bound +1.98, at $n=2$, seed 2, against the step-matched MLP). % SRC: """ + SRC_F + r"""
The MLP is the strongest store-content rival on every rung and seed and is flat from 51,200 to % SRC: """ + SRC_F + r"""
204,800 steps; the continuous MLP is still gaining at 204,800 steps and never becomes the strongest % SRC: """ + SRC_F + r"""
rival. One more caveat bears on every single-fit rival number in this paper: two MLP fits with the
same rows, width, schedule and 2,560 steps, differing only in the random draw, differ per seed by % SRC: """ + SRC_F + r"""
$-2.15$ to $+4.13$ novel tuples per corner, and their per-corner interval excludes zero on 12 of 15 % SRC: """ + SRC_F + r"""
rung--seed legs. Family differences smaller than that on a single seed are not read as real.
"""),
    ("E5", r"""Four rows remain pending, each marked in the text by a red \textsc{placeholder} whose argument names
the row and states that it may land as a negative or not at all: the third ladder rival set, fitted
on 32,768 rows of each rung's pool (one; Table~\ref{tab:ladder}); and the per-arm generated states of % SRC: w20-shapes-data-matched/FINDINGS.md
the arms of Table~\ref{tab:cmazet} in Figures~\ref{fig:cmazet-families}, \ref{fig:cmazet-door-tint}
and~\ref{fig:cmazet-deletion} (three), which were never persisted and need figure work. Rows that
were pending earlier are now written in from their artefacts: the data-matched rival set
(Tables~\ref{tab:t7} and~\ref{tab:ladder}); the four deletion measurements on every scope
""", r"""Three rows remain, each marked in the text by a red \textsc{placeholder} whose argument names
the row: the per-arm generated states of the arms of Table~\ref{tab:cmazet} in
Figures~\ref{fig:cmazet-families}, \ref{fig:cmazet-door-tint} and~\ref{fig:cmazet-deletion}. They are
permanently blocked, not pending: the run behind Table~\ref{tab:cmazet} persisted no arm object and
no per-transition prediction (its run records hold aggregates only), so these panels cannot be drawn
from that run, and we keep them as named placeholders rather than draw them from any other run. Rows
that were pending earlier are now written in from their artefacts: the data-matched rival set
(Tables~\ref{tab:t7} and~\ref{tab:ladder}); the store-content ladder rival set, fitted on each rung's
32,768 content rows (Table~\ref{tab:ladder}); the four deletion measurements on every scope % SRC: """ + SRC_C + "\n"),
    ("E6", r"""\item The third ladder rival set on 32,768 rows of each rung's pool (pending); LACE and PoE refit % SRC: w20-shapes-data-matched/FINDINGS.md
  on 131,072 rows on the ladder rungs (rows 3--4 only were run); the ten-times fits on the ladder % SRC: w20-shapes-data-matched/FINDINGS.md
""", r"""\item LACE and PoE refit on 32,768 or 131,072 rows on the ladder rungs (rows 3--4 only were run, at % SRC: w20-shapes-data-matched/FINDINGS.md; """ + SRC_F + r"""
  131,072 rows); the store-content set's continuous MLP past 204,800 steps; the data-matched set's ten-times fits on the ladder % SRC: w20-shapes-data-matched/FINDINGS.md; """ + SRC_F + "\n"),
]

RETIRED = ["A third set, fitted on 32,768 rows of each rung's pool, is pending",
           "The third ladder rival set on 32,768 rows of each rung's pool (pending)",
           "Four rows remain pending", "which were never persisted and need figure work", "27.13$\\times$"]


def placeholders(text):
    from texnum import macro_args
    return macro_args(text, "PLACEHOLDER")


def apply(text):
    before = placeholders(text)
    for name, old, new in EDITS:
        if name == "E3":
            new = block_c() + "\n"
        if text.count(new) == 1:
            continue  # idempotent: already applied
        n_old = text.count(old)
        if n_old != 1:
            sys.exit(f"REFUSE {name}: anchor found {n_old} times")
        text = text.replace(old, new)
    # E7: PoE's multiple, two places, each on a line whose SRC must also name the JSON
    poe_lines = [i for i, ln in enumerate(text.split("\n")) if "27.13$\\times$" in ln or "27.135$\\times$" in ln]
    lines = text.split("\n")
    for i in poe_lines:
        ln = lines[i].replace("27.13$\\times$", "27.135$\\times$")
        if "GENTABLE-KSTAR.json" not in ln:
            if "% SRC:" in ln:
                ln = ln + "; w15-lace-poe-rows/GENTABLE-KSTAR.json"
            else:
                ln = ln + " % SRC: w15-lace-poe-rows/FINDINGS.md; w15-lace-poe-rows/GENTABLE-KSTAR.json"
        lines[i] = ln
    if len(poe_lines) != 2:
        sys.exit(f"REFUSE E7: {len(poe_lines)} PoE-multiple lines, expected 2")
    text = "\n".join(lines)
    after = placeholders(text)
    exp = [p for p in before if p != PH_ARG]
    if after != exp:
        sys.exit(f"REFUSE: PLACEHOLDER sequence changed beyond the declared removal: {len(before)} -> {len(after)}")
    for r in RETIRED:
        if r in text:
            sys.exit(f"REFUSE: retired phrase survives: {r}")
    return text


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    out = apply(src.read_text(encoding="utf-8"))
    dst.write_text(out, encoding="utf-8")
    print(f"OK {src.name} -> {dst}: PLACEHOLDER {len(placeholders(src.read_text(encoding='utf-8')))} -> {len(placeholders(out))}")
