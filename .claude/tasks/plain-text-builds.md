# plain-text-builds — paper-writer (V2 and V5 as PLAIN, unformatted documents: no page-fitting typography, no "Mo", all banked figures restored)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 45; Head directives 2026-08-21, items 1–3).** Read `.claude/AGENT_PROTOCOL.md`, then this file.

**Output: a NEW folder `.claude/papers/plain/` with two subfolders — `plain/v2/` and `plain/v5/`** (each: `submission.tex` · `submission.pdf` · `figs/` · `BUILD-NOTE.md`). ⛔ **Every existing paper folder is read-only for this pass** — `.claude/papers/{v2-short,v5-short,neurreps-variants,palm-variant,v2-neurreps-descoped}/**` are sources only. (⚠ Note the standing lesson: do not run `pdflatex` inside a source folder; build only in `plain/`.) `pdflatex` is at `/Library/TeX/texbin/pdflatex`.

**Sources:** V2 ← `.claude/papers/v2-neurreps-descoped/submission.tex` (the current narrative: one contribution, 2026 audience, corrected novelty scope). V5 ← `.claude/papers/palm-variant/v5/submission.tex`.

**DIAL DECLARATION: none — typesetting/editorial pass; zero number changes; the only mandated wording change is item 2 below.**

## 1 — Plain typography: strip every page-fitting device (Head: "strictly ignore the page budget")
⛔ **Page count is irrelevant for this pass. Do not condense and do not expand any text.** Remove, everywhere:
- every font-size command on body text, captions, references, tables and appendices (`\tiny`, `\scriptsize`, `\footnotesize`, `\small`) — set everything at the document's default size;
- `multicol` reference lists → a normal single-column reference list;
- every tightened skip and layout hack (`\@startsection` redefinitions, altered `\textfloatsep` / `\intextsep` / `\abovecaptionskip` / `\belowcaptionskip`, `\raggedbottom`, narrowed table column specs introduced to clear overfull boxes);
- every reduced figure width — each figure is placed at its natural width (`\linewidth`, or the width its aspect ratio wants), never a fraction chosen to save a page.
**Permitted exception, and it must be listed:** a `\small` on a table that would otherwise physically overflow the text block. List every instance in the build note; there should be very few.

**Formatting minimalism (Head):** plain text throughout — ⛔ no bold outside structural headers · italics sparingly (emphasis where a sentence genuinely needs it) · `\texttt{}` for software, tool, flag and file names (e.g. `\texttt{langevin\_noise="fdt"}`) · standard sectioning, standard captions. Let overfull boxes and page breaks fall where they fall; report them, do not fight them.

## 2 — ⛔ The word "Mo" appears nowhere (Head directive, strict)
The name must not appear in **body text, section text, figure captions, figure labels, filenames, or in-text mentions of any kind.** Every reference to that work is phrased as a citation: *"…the single-exponential lifetime law reported in~\cite{key}"*, *"…reproduced under the protocol of~\cite{key}"*. Rewrite each sentence so it reads naturally with the citation carrying the reference — this is the one mandated wording change in this pass.
- **Rename the figure files and labels that embed the name** (`fig1_mo_headtohead.png`, `fig2_mo.png`, `sf1_mo_estimator_overlay.png`, label `fig:mo_own`) to neutral, content-describing names (e.g. `fig_lifetime_headtohead.png`, `fig:lifetime_headtohead`) and update every reference. Copy the renamed files into `plain/*/figs/`; ⛔ do not rename anything inside the source folders.
- ⚠ **Trap — do not sed blindly:** the source contains **"Morse"** and **"Moser"**, which are different words and must survive untouched. Match on the author token only.
- ✅ **The bibliography entry keeps its authors** — that is what a citation is. The instruction governs prose, captions and labels, not the reference list.
- State the check in the build note: occurrences of the standalone author token in `submission.tex` = 0, with the Morse/Moser false positives shown as excluded.

## 3 — Restore the banked figures (Head: "add as many plots as we have, at least to the appendix")
Both papers ship a small subset of what is banked. Restore the rest **where they serve the narrative**; the appendix has no page limit, so the default is *include*.
- **V2 — three banked figures are currently unused**, all present in `.claude/papers/v2-short/figs/`: ⭐ **`fig3_retention_overlay.png`** (the Head named this one: retention of the trained CLU against the LSTM baseline — it shows the baseline-collapse result directly), `fig1_gmor.png`, and `sf1_mo_estimator_overlay.png` (the estimator overlay — ⚠ rename per item 2). The canonical `v2-short/draft.tex` uses all six and is your reference for what each shows and how it was captioned.
- **V5 — seven banked figures are currently unused**, in `.claude/papers/v5-short/figs/`: `fig2_vault.png`, `figB_dlaw.png`, `figB_massive_vs_flat.png`, `figB_signflip.png`, `figC_Tstar.png`, `figC_lambda_coset.png`, `figC_register_capacity.png`.
- **Placement rule:** a figure whose result is in main text may go to main text if it is the clearest presentation of that result; **everything else goes to the appendix**, grouped with the result it evidences — including figures whose result was demoted (demotion moved the claim out of the contributions list, not the evidence out of the paper).
- ⛔ **Judgment required, not bulk inclusion:** include a figure only if it evidences a statement the paper actually makes, and give each one a caption that says what it shows, its scope (seeds, dimension, budget) and its source-report provenance in the paper's own voice. **If a banked figure has no home in the narrative, leave it out and say why in the build note** — one line each.
- ⚠ **Multi-seed status must be visible:** where a figure shows a single seed or an n < 3 cell, the caption says so plainly (the Head's own instruction: single-seed material is appendix material).

## Boundaries
1. ⛔ Approved wordings, mandatory riders, scope qualifiers and fine print stay **verbatim** and beside their claims; the item-2 rewrites may not alter any claim's scope.
2. ⛔ Zero number changes; zero findings added or dropped. Two-way numeric-token check against each source, printed.
3. ⛔ All sweeps (never-quote · internal-apparatus · semantic hermeticity) per-file, positive-controlled, printed. Anonymization posture unchanged from the sources.
4. New captions are new prose: they must obey `PJ_Writing_Style_Context.md` (direct, plain terms, no weasel words) and carry no unsupported claim.

## Acceptance criteria
1. Two plain builds that compile; page counts reported and explicitly **not** optimised; the list of permitted `\small` exceptions printed (expected: few).
2. Standalone author-token count = 0 in both `submission.tex` files, with Morse/Moser shown as excluded false positives; no figure filename or label carries the name.
3. Every banked figure either included with a provenance-bearing caption, or excluded with a one-line reason; single-seed figures labelled as such.
4. Numeric checks and sweeps printed; all source folders byte-untouched (state the check, per folder).
