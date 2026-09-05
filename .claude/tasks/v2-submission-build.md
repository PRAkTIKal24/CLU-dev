# v2-submission-build — paper-writer (V2 → the clean submission artifact; Head ruling 2026-08-19)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 29).** Read `.claude/AGENT_PROTOCOL.md`, then this file. You create `papers/v2-short/submission/` (submission.tex + submission.pdf + a build note) and add a CHANGELOG entry. ⛔ **`draft.md`/`draft.tex` (the internal canonical, v0.8) are NOT edited** — the canonical keeps its provenance apparatus by design; the submission build is a derived artifact.

**DIAL DECLARATION: none — editorial build; zero content/number changes.**

## The Head's three instructions (verbatim substance, binding)
1. **No internal apparatus anywhere:** git commit tags/hashes, branch names, flag-provenance tables, registry tokens, draft-status blocks, venue-class headers, "reading order" notes, inline HTML comments, cross-reference-convention notes — ALL stripped from the submission artifact.
2. **Appendix triage: keep only appendices that explain VISUAL/MEASURED results (tables, plots); remove the rest.**
3. **Remove all parenthetical editorial comments.**

## Executing instruction 2 — the triage, with three ruled exceptions
- **REMOVE outright:** the flag-provenance appendix (internal) · **Appendix M in its entirety** (the retained long-form — internal C-10 insurance, pure duplication for a reviewer) · any appendix that is prose-only commentary.
- **KEEP (the criterion):** every appendix whose substance is figures/tables/measured results (the erosion study, the kick-amplitude probe, the exceptional-point + damping-corollary material, the T=0/T>0 budget faces, GMOR) — each trimmed to: the result, its figure/table, the mandatory fine print. Any long derivation inside them compresses to statement + pointer-to-figure.
- **Exception 1 — Appendix A (the colleague's primer): KEPT** (a deliberate standing Head ruling, Add.11; a general cleanup does not silently override it — the Head can strike it with one word if intended). Trim its boxed scope note to the compact form; keep the corrected A.5 sentence (strip the HTML flag comment — the comment, not the sentence).
- **Exception 2 — the negatives appendix: KEPT AS A COMPACT TABLE** (the Q4/C-9 ruling: negatives distributed in appendices is program identity, and every entry carries a measurement — it satisfies the Head's "measured results" criterion in table form). One row per negative: claim tested · result · number.
- **Exception 3 — the CM-21 retirements + prior-art appendix: KEPT COMPRESSED** (the four retirements are a claims-discipline obligation; a submission without them re-opens retired claims). Half a page maximum.

## Executing instruction 3 — the parentheses purge, with the one distinction that must not be lost
- **STRIP:** editorial/meta parentheses — "(see App …; also …)" chains, "(placement pending)", "(labelled evidence)", "(flagged to the Hub)", drafting asides, anything addressed to us rather than the reader.
- ⛔ **DO NOT STRIP (rewrite into prose instead where they read awkwardly):** C-5 scale qualifiers ("dim 4, 5 seeds, laptop-CPU"), C-6 fine print, statistical qualifiers (±, SE, seed counts), and the mandatory riders — these are binding claim-scope, not comments. A qualifier may move from parentheses into the sentence; it may never disappear.

## Submission mechanics
1. **Template:** the NeurReps EA venue template if obtainable locally; else the NeurIPS 2026 style file as the closest approximation, with a build-note line saying which was used. Report the true main-text page count against the 4-pp limit.
2. **If main text still exceeds 4 pp in-template:** apply the Add.25 lean — GMOR-proper compresses to one sentence + appendix pointer — and report what that bought; ⛔ nothing else moves without flagging in the build note.
3. **Anonymization (the full Add.2 checklist + Add.28):** `\author{Anonymous}` per template · no acknowledgments/funding · metadata scrubbed (check the PDF's Producer/Author fields) · third-person self-citation intact ("introduced as CHLU in Jawahar & Pierini (2026)") · ⛔ the theory note cited as **"Anonymous (2026), provided in the supplementary material"** per the ratified Option B — and **produce the anonymized F5 supplementary PDF** (from `papers/f5-note/f5-note.tex`: names/acknowledgments/identifying strings stripped, compiled; placed at `papers/v2-short/submission/supplementary-theory-note.pdf`). ⛔ f5-note source files untouched.
4. **Final sweep on the submission artifact itself** (per-file, positive controls): the never-quote list + the semantic hermeticity class + a new class for this build: `commit`, `branch`, `agent/`, `chlu/`, `.claude/`, `CHLU`, `tectonic`, `draft.md` — all must be 0 in submission.tex (the J&P citation is the one sanctioned "CHLU" instance).

## Writing style (Head directive, 2026-08-19 — BINDING on this and every future rewrite/build)
Read `.claude/PJ_Writing_Style_Context.md` and apply its Core Writing Philosophy + Formatting rules to every sentence you touch (the thesis-persona framing in that file does not apply; the style rules do): ABT narrative structure for section openings · macro-to-micro information flow · objective tone, zero weasel words, magnitude descriptors only when the data supports them · succinct phrasing with syntactic variety · explicit signposting · assumptions as bullets · "we" for research actions, passive for established facts · bold sparingly, for new-concept definitions only. **Plus the Head's rider: brevity, and simple basic technical terms — never complicate an explanation for no reason.**
⛔ **One boundary the style pass must not cross:** matrix-approved wordings, mandatory riders, and scope qualifiers are BINDING VERBATIM — style may reposition them, never paraphrase or "simplify" them. Style governs everything else.

## Acceptance criteria
1. `submission.pdf` builds clean; page count stated; a reviewer sees zero internal apparatus (criterion: nothing in the PDF names a tool, file, branch, commit, registry row, or process).
2. The build note maps every removed appendix/block to its canonical home (nothing is lost, everything is findable).
3. The parentheses purge log: counts stripped vs converted-to-prose vs kept-as-mandatory.
4. Canonical `draft.md`/`draft.tex` byte-untouched; f5-note sources byte-untouched.
