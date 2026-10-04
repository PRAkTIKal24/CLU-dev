# pw1-draft-template — apply the latest ICLR 2027 main-track (double-blind) template to BOTH drafts; zero content change
Session `pw1-draft-template` · worktree `~/clu/worktrees/CHLU-pw1-draft-template` · branch `pw1/draft-template` (from `pw1/draft-final`: its audit, index and page-count tooling are in the tree) · out `~/clu/outputs/pw1-draft-template/`
**You hold the `.tex` lock.** Added by the Head 2026-09-25 as PW-1's final task. Read `COMMON.md`, `~/clu/outputs/w17-paper/VENUE.md` §3–§4 (the template record and the double-blind rules), and `~/clu/outputs/pw1-draft-final/FINDINGS.md` + `FINAL-INDEX.md`.

## The one job
Make both drafts conform exactly to the current official ICLR 2027 conference template, in submission (double-blind) mode, **changing no content**: no sentence, number, table cell, caption, figure, citation or section order.
1. **Latest template.** Download the official zip again (`https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`, per VENUE.md) into a fresh timestamped dir. Record its sha256 and the sha256 of every member. Compare member by member with `w17-paper/iclr2027/` and with VENUE.md's recorded hashes.
   - If they are identical, install nothing and say so.
   - If they differ, move the old `iclr2027/` to `iclr2027-retired-pw1/` (move, never delete), install the new members, and print a diff of every changed file.
   - If the download fails, say so, keep the on-disk copy, and continue with steps 2–4.
2. **Preamble conformity.** Compare both drafts' preamble line by line against the template's sample `iclr2027_conference.tex`: `\documentclass{article}`, `\usepackage{iclr2027_conference,times}`, the template's own package lines, `\iclrfinalcopy` COMMENTED OUT, `\author{Anonymous authors}`, `\bibliographystyle` resolving to the template's `.bst`. The template says "Do not change any aspects of the formatting parameters in the style files".
   - List every command in the drafts that overrides a style parameter: geometry or margins, `\textwidth`/`\textheight`, font size, `\baselinestretch`/`\linespread`, `\parskip`/`\parindent`, caption or section spacing, float parameters (`\topfraction`, `\floatsep`, …), and `\vspace{-…}` hacks outside figures.
   - Remove each override that changes a template parameter. Print it with its line and its effect on the page count.
   - Keep the draft's own content macros (`\PLACEHOLDER`, `\TODO`, `\figslot`, `\sci`, `\Kstar`, …) and standard packages that change no layout parameter (e.g. `booktabs`, `amsmath`, `hyperref` if the template allows it).
   - If you cannot tell whether a command is a layout override, list it and leave it in.
3. **Double-blind audit (report, and fix only formatting-level leaks):**
   - PDF metadata: `/Author`, `/Title`, `/Subject`, `/Creator`, `/Keywords`.
   - Rendered text: names, institutions, e-mails, URLs that identify, machine paths, "our previous work" self-reference.
   - No acknowledgements section. The running header reads "Under review as a conference paper at ICLR 2027".
   - Fix only metadata or hyperref settings that leak identity. For anything in the prose, print it and do NOT edit it. Prose is the Head's.
   - Venue statements (AI use, ethics, reproducibility; their order and presence) are MF-12, HELD by the Head: report where they are, change nothing.
   - The `.tex` comments carry machine paths. Report their count; do not strip them (that is PW-2's anonymised-upload step).
4. **Rebuild and re-audit:** both drafts compile from the w17-paper layout (exit 0, 0 undefined citations/references, overfull boxes listed), style diff-check EMPTY, provenance identical, `\PLACEHOLDER` / `\TODO` byte-identical. Re-run pw1-draft-final's index and per-section page-count tooling, and write an updated `FINAL-INDEX-template.md` (figure/table index, per-section pages, main text against 9 pp) with a before → after column for every page number that moved.

## Acceptance
- A word-level diff of the rendered PDF text before → after is EMPTY, apart from layout artefacts (hyphenation or line breaks). Print the diff tool and its result. This is the "no content change" proof.
- A template hash table (downloaded vs on-disk vs VENUE.md), and the list of overrides removed or kept, with reasons.
- Installed byte-identically into both drafts and PDFs.
**What would prove this wrong:** any rendered word changes, or a template parameter override survives without being listed.
