"""w18 spoke/seeds-and-tables, PASS 3: guarded, idempotent edits to DRAFT-v2.tex and DRAFT-v2-style.tex.

Three things, each an exact-anchor replacement that must match exactly once (or already be applied):
  MF2   the referee's rounding slip 23.5117 -> 23.5121 (w15-table-fixes/TABLE-FIXES.md:83-84).
  K5    the ladder's n >= 3 training-data (coverage) inference note now travels with tab:k5 too.
  W19   the w16 CMAZE-T n-way margin is qualified wherever the draft said the symmetric-input test
        was "pending": w19 closed with every condition a LOSS, and in its own words "the pair block,
        not the store, carries the w16 margin" (w19-maze-t-symmetric/FINDINGS.md:22, 217-218).
        Qualitative only: no w19 number enters the draft here. The w19 rows are the \\PLACEHOLDERs of
        tab:cmazet-sym, which placeholder-fill owns and which this script never touches.

The script refuses to write a file if any anchor is missing or ambiguous, or if the ordered
\\PLACEHOLDER{} argument sequence changes. Usage: python apply_pass3_edits.py DRAFT.tex [DRAFT2.tex ...]
"""
import re
import sys

W19 = "% SRC: w19-maze-t-symmetric/FINDINGS.md"

COMMON = [
    ("W19-abstract",
     "rivals, which read the observation, do not receive, so the row is scoped to that request until a\n"
     "symmetric-input test is run, and no matched arm, the store included, yields a valid chained\n"
     "rollout.\n",
     "rivals, which read the observation, do not receive; when every arm is given the identical input\n"
     "the margin does not survive, because a matched transformer given the store's own pair design\n"
     "matches or beats it, so the pair block, not the store, carries it; and no matched arm, the store " + W19 + "\n"
     "included, yields a valid chained rollout.\n"),
    ("W19-intro",
     "pair block that its rivals do not receive (a symmetric-input test is pending); on validation it\n",
     "pair block that its rivals do not receive (given the identical input, matched rivals match or\n"
     "beat it, so that margin belongs to the pair block, not the store); on validation it " + W19 + "\n"),
    ("W19-section",
     "receives the identical input, is pending (Table~\\ref{tab:cmazet-sym}, Appendix~\\ref{app:mazewm}).\n",
     "receives the identical input, has closed with no condition winning (Table~\\ref{tab:cmazet-sym},\n"
     "Appendix~\\ref{app:mazewm}): given the store's own pair design, a matched transformer matches or\n"
     "beats the store, so the pair block, not the store, carries this row's margin over the rivals. " + W19 + "\n"
     "What stands on CMAZE-T is the store gate (every landscape control collapses) and exact deletion,\n"
     "not a margin over matched rivals.\n"),
    ("W19-summary",
     "interface result (CMAZE-2b), the CMAZE-T store row, scoped to its pair-design request, with its declared negatives, and the CMAZE-O store\n",
     "interface result (CMAZE-2b), the CMAZE-T store gate and exact deletion, whose margin over matched\n"
     "rivals belongs to the pair-design request and does not survive symmetric inputs, with its " + W19 + "\n"
     "declared negatives, and the CMAZE-O store\n"),
    ("W19-limits",
     "  On CMAZE-T the store row is scoped to a word request with a pair block that the rivals do not\n"
     "  receive, and a symmetric-input test is pending (\\S\\ref{sec:maze-wm}).\n",
     "  On CMAZE-T the store row is scoped to a word request with a pair block that the rivals do not\n"
     "  receive; given the identical input no condition wins, and the pair block, not the store, carries\n"
     "  the margin (\\S\\ref{sec:maze-wm}, Table~\\ref{tab:cmazet-sym}). " + W19 + "\n"),
    ("W19-tab-cmazet-caption",
     "(\\S\\ref{sec:maze-wm}); the symmetric-input rows are Table~\\ref{tab:cmazet-sym}. Config A1:",
     "(\\S\\ref{sec:maze-wm}); the symmetric-input rows are Table~\\ref{tab:cmazet-sym}, where no condition\n"
     "wins: a matched transformer given the store's own pair design matches or beats the store, so the " + W19 + "\n"
     "pair block, not the store, carries this table's $n$-way margin, and that row is a store row only\n"
     "under this asymmetric input, never an unqualified win. Config A1:"),
    ("W19-tab-cmazet-nway",
     "unseen $n$-way & 3 & \\textbf{STORE row}: store 0.7246",
     "unseen $n$-way & 3 & \\textbf{STORE row under asymmetric inputs only} (the margin does not survive symmetric inputs, Table~\\ref{tab:cmazet-sym}): store 0.7246"),
    ("K5-note",
     "read-path rival with the rung's own corners, and the two pre-declared landscape controls as paired deltas.}\n"
     "\\label{tab:k5}\n",
     "read-path rival with the rung's own corners, and the two pre-declared landscape controls as paired deltas.\n"
     "$n=2$: 60 corners; $n\\ge3$: the identical 120 corners. \\emph{Inference note, as for % SRC: w12-composition-ladder/FINDINGS.md\n"
     "Table~\\ref{tab:ladder}:} for $n\\ge3$ the rungs differ in fit-pool coverage, not in composition\n"
     "depth, so the store's own rise from $n=2$ to $n=6$ is not to be inferred.}\n"
     "\\label{tab:k5}\n"),
]

MF2 = {
    "DRAFT-v2.tex": (
        "B, 24.5466 / 23.8422 / 23.6697 times the pin; the statistic alone is 23.5117 times; a residual of\n",
        "B, 24.5466 / 23.8422 / 23.6697 times the pin; the statistic alone is 23.5121 times; a residual of % SRC: w15-table-fixes/TABLE-FIXES.md\n"),
    "DRAFT-v2-style.tex": (
        "and the statistic alone is 23.5117 times; a residual of exactly 16,384 B between the serialised\n",
        "and the statistic alone is 23.5121 times; a residual of exactly 16,384 B between the serialised % SRC: w15-table-fixes/TABLE-FIXES.md\n"),
}


def placeholders(text):
    out, i = [], 0
    while (j := text.find("\\PLACEHOLDER{", i)) >= 0:
        k, depth = j + len("\\PLACEHOLDER{"), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j:k])
        i = k
    return out


def apply(path):
    name = path.rsplit("/", 1)[-1]
    if name not in MF2:
        raise SystemExit(f"{path}: not a known draft (expected {sorted(MF2)})")
    text = open(path, encoding="utf-8").read()
    before = placeholders(text)
    edits = [("MF2",) + MF2[name]] + COMMON
    log = []
    for tag, old, new in edits:
        n_old, n_new = text.count(old), text.count(new)
        if n_old == 0 and n_new == 1:
            log.append(f"{tag}: already applied")
        elif n_old == 1 and n_new == 0:
            text = text.replace(old, new)
            log.append(f"{tag}: applied")
        else:
            raise SystemExit(f"{path}: {tag}: anchor count {n_old}, replacement count {n_new}; refusing to write")
    if placeholders(text) != before:
        raise SystemExit(f"{path}: PLACEHOLDER sequence would change; refusing to write")
    if "23.5117" in text:
        raise SystemExit(f"{path}: 23.5117 still present; refusing to write")
    open(path, "w", encoding="utf-8").write(text)
    print(path, "|", "; ".join(log), f"| placeholders unchanged: {len(before)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        apply(p)
