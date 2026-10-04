# FINDINGS — pw2-draft-methods (PW-2 item 3: the Methods opening, BOTH drafts)

Session `pw2-draft-methods`, branch `pw2/draft-methods` (from `pw2/draft-sections` 6b431af), batmac. Stage `stage-20260925T155839Z`, snapshot `snap-20260925T155839Z` (sha256 in `snap-*/SHA256`; it equals pw2-draft-sections' `INSTALLED-SHA256.txt`, so I started from their install).

## Status

| brief point | status |
|---|---|
| 1. The Head's text at `% PW2-ITEM3-SLOT`, verbatim, both drafts | **DONE** |
| 2. Citation-scope rule, every CHLU mention listed with a verdict; `~\citep` at the first allowed mention outside Methods | **DONE**: 27 mentions per draft; 0 VIOLATION; 1 citation added; 1 flag and 3 notes for the Head (below) |
| 3. `HEAD-HELD.md` updated with the new Methods first paragraph | **DONE** (`HEAD-HELD.md` here) |

## Acceptance (both drafts)

| check | before (live, item 2) | after (installed) |
|---|---|---|
| pdflatex / bibtex exit | 0 / 0 | 0 / 0, no Rerun, 4 passes |
| undefined citations / references / errors / multiply-defined / overfull | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| bibitems (entries cited) | 55 of 56 (`jawahar2026chlu` uncited) | **56 of 56** |
| pages v2 / style | 43 / 44 | 43 / 44 |
| underfull boxes v2 / style | 20 / 20 | 21 / 17. All the differing ones are `\vbox ... while \output is active`, meaning page breaks. No `\hbox` changed. |
| figure / table / section numbers | — | 77 of 77 labels, 0 renumbered. Pages moved: v2 12 labels, style 7 (by one or two pages, because the new paragraph is 10 lines) |
| style diff-check (numbers multiset, placeholders, TODO, artefact paths, SRC strings, heading and label sequence, floats, preamble) | EMPTY | **EMPTY** (`STYLE-DIFF-CHECK.md`) |
| provenance findings | v2 16 lines, style 17 | identical, with line numbers masked (every finding is shifted by +9 lines) |
| `\TODO` / `\PLACEHOLDER` | 6 / 3 lines per draft | unchanged, byte-identical arguments |
| Head-held paragraphs | — | 6 of 7 rows byte-identical to item 2's list; the Methods row is the new paragraph, sha256 `c100fed5…2988` (the same in both drafts) |
| bib `refs_clu_iclr.bib` | b2cf715f… both | b2cf715f… both, at the start and the end; never written |
| installed | — | `.tex` + PDF in `~/clu/pw2/live/{v2,style}`, cmp-equal to the stage. The live files were cmp-equal to the snapshot just before the install, and the lock guard was clean. A fresh build from a copy of the live folders gives identical `\newlabel`s (`INSTALLED-SHA256.txt`, `verify-live-*`) |

## 1. The Methods opening

Source, both drafts, lines 273–282 (the first chunk after `\section{Methods}\label{sec:object}`; in the style twin its own signpost "A claim about any of the three properties …" now follows as the second paragraph):
```
% PW2 item 3 (PLAN.md sec. 2): the Head's Methods opening, verbatim; LaTeX only (\citet = "Jawahar & Pierini (2026)").
\citet{jawahar2026chlu} introduced the CHLU as a latent dynamics model (defined here as an algorithm
... (wrapped at 100 columns) ...
as a virtue of the design itself.
```
LaTeX set: each "Jawahar \& Pierini (2026)" → `\citet{jawahar2026chlu}`. Both occurrences are at the start of a sentence, so no `~` is needed. The text has no quotes or dashes to typeset. Every word, punctuation mark and spelling ("factorized", "virtue", "naive-CLU", "towards") is the plan's own.

**Rendered vs plan** (`RENDERED.txt`, `pw2_draft_methods/rendered.py`; the plan's text is read out of PLAN.md, not retyped):
- plan: `Jawahar & Pierini (2026) introduced the CHLU as a latent dynamics model (defined here as … such as deletion as a virtue of the design itself.`
- rendered, v2 and style: identical. 135 of 135 words, **WORD DIFF: EMPTY**, and the exact string equals the plan's. natbib prints `\citet` as "Jawahar & Pierini (2026)", which is the plan's text character for character, so the diff is empty with no exception needed for the citation.
- Whole document (`CHARPROOF.txt`): the character multiset of the after-PDF equals the before-PDF plus exactly three insertions: the paragraph, "(Jawahar & Pierini, 2026)", and the bibliography entry. Nothing is lost and nothing else is added, in either draft. This holds even with floats moving between pages.
- Positive controls: one planted word (`virtue` → `vertue`) → word diff 1 NON-EMPTY; one planted character → CHAR PROOF FAIL; a planted `~\citep{jawahar2026chlu}` after "retention law" → verdicts.py VIOLATION, exit 1.

**Choice logged:** `PLAN.md` §2's order-of-work line says `\citep{jawahar2026chlu}`, while the brief says `\citet`. I used `\citet`. The item-3 text is binding verbatim, and "Jawahar & Pierini (2026) introduced …" is the author-as-noun form; `\citep` would print "(Jawahar & Pierini, 2026) introduced …" and change the Head's text. The `~\citep` form is used for the one addition outside Methods.

**Locator verified:** arxiv.org/abs/2603.01768 was fetched 2026-09-25. Title "CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning", authors Pratik Jawahar, Maurizio Pierini, v1 2 Mar 2026; this matches the bib entry. The abstract names the unit "(pronounced: clue)", a relativistic Hamiltonian structure and symplectic integration. It does not mention deletion, retention, test-time compute or GMOR. It does mention a "memory-stability trade-off" (in the stability sense); the drafts do not cite it for memory.

## 2. Every CHLU mention, with its verdict (`CHLU-MENTIONS.md`, full sentences)

The drafts name the parent unit "the CLU" (and "a settling unit", "the settling (physics) CLU"). "CHLU" appears only in the new Methods opening. The scan (`chlu_scan.py`, code only) covers the following terms: CHLU; bare CLU; Hamiltonian; symplectic; relativistic; governor; settling; field-mode; zero-dynamics; shorts / app:shorts; GMOR; retention law; dissipative channel; `jawahar2026chlu`. The rows and verdicts are identical in both drafts (line numbers below are v2's, after install; style's are within ±9).

| v2 line | mention | verdict |
|---|---|---|
| 132 | abstract: "the n-CLU is exactly the zero-dynamics limit of a settling unit" | do not cite (abstract is Head-owned and carries no citations; the identity is this study's) |
| 197 | Hamiltonian neural networks~\citep{greydanus…} | not the CHLU |
| 257 | "its measured zero-dynamics identity" (signpost) | do not cite |
| 274 | `\citet{jawahar2026chlu} introduced the CHLU as a latent dynamics model …` | cite (item-3 text): the unit |
| 278 | `\citet{jawahar2026chlu} envision their unit …` | cite (item-3 text): the unit |
| 280 | "our naive-CLU as a special case of the CHLU … factorized memory unit … deletion" | already fine: memory and deletion are attributed to "we" |
| 916 | refusal holes, "the field-mode n-CLU's" coupling layout | do not cite |
| 1172 | heading "The zero-dynamics clue" | do not cite |
| **1175** | "the zero-dynamics limit of the CLU~\citep{jawahar2026chlu}: the same content, the same wells, …" | **cite (added here)**: first mention of the unit and its dynamics outside Methods. The citation follows "the CLU" only; the identity is this study's |
| 1199 | "Its laws are this paper's own theory … (further laws in App. Prior derivations)" | already fine: says it outright |
| 1202, 1214 | field-mode n-CLU relaxation law; "no arm of ours fails the retention law" | do not cite (retention) |
| 1235 | "the settling (physics) CLU" (follow-up) | already fine: allowed, but not re-cited (the first mention carries it) |
| 1883, 1914, 1933, 1938, 1983, 2000, 2034, 2395 | zero-dynamics appendix, retention-law appendix and table, settling arms, field-mode refusal, rollout, CRR | do not cite |
| 2006 | App. "Prior derivations" (`app:shorts`) | already fine (**FLAG**, see below) |
| 2065, 2078, 2079 | GMOR (cited to Gell-Mann 1968), dissipative channel, retention law | do not cite |
| 2502, 2504 | retention-law items in the not-run list | do not cite |

Counts per draft: do not cite 19; cite (item-3 text) 2; cite (added here) 1; already fine 3; already fine (flag) 1; not the CHLU 1; **VIOLATION 0**. The shorts are cited nowhere (the `shorts-evidence-inventory.md` strings are `% SRC` comments, not citations).

**For the Head (reported, not rewritten; item 1 and the Head own the prose):**
1. **FLAG, adjacency:** App. "Prior derivations" says its laws were "derived and validated on designed or small trained testbeds before this study" (memory-level deletion, retention triad, GMOR, dissipative channel, …). Methods now cites Jawahar & Pierini (2026) for the CHLU, so a reader could take "prior derivations" to mean that paper. No sentence says so, and the section opens "The laws below are this paper's own theory", so there is no VIOLATION. A heading such as "Further derivations of this paper's theory" would remove the ambiguity.
2. **Coinage attribution:** Methods now says Jawahar & Pierini "introduced the CHLU as a latent dynamics model (defined here as …)", while Sec. 5.4 opens "We call the class of objects … a *Latent Dynamics Model*". The two sentences disagree about who coined the term, and 5.4 gives a different definition (a memory read by relaxing a latent state under a designed energy).
3. **Naming:** "CHLU" (Methods) and "the CLU" (Secs. 5.3 and 5.6) name the same unit. The arXiv abstract says "pronounced: clue", but the paper never states CLU = CHLU. "a settling unit" (abstract) is a third name.
4. **Bib rendering:** the entry prints "Chlu: The causal hamiltonian learning unit …" because the bib title has no braces around {CHLU} and {Hamiltonian}, so the .bst lower-cases them. The bib is the Head's and was not touched.

## 3. HEAD-HELD.md
`HEAD-HELD.md` here replaces item 2's list for later spokes. The Methods row is now `\citet{jawahar2026chlu} introduced the CHLU as a latent dynamics`, lines 273–282, sha256 `c100fed55e462eb5fa87694eeb063198605539b44cb28a2eeab53e2ab81e2988` in both drafts. The other six rows per draft are unchanged; `headheld.py` passes all 7 on both installed drafts against the new list. The old list fails only on the Methods row, as expected.

## What did NOT run
- No other PW-2 item (1, 4, 5) was touched. No prose outside the slot changed, apart from the one added `~\citep`. The four notes above are reported, not fixed.
- The arXiv paper's full text was not read; only its abs page (title, authors, dates, abstract) was. Whether the arXiv CHLU has a "settling loop" is not checked; the added citation is attached to "the CLU" (the unit) only.
- No page-by-page visual review of the PDFs; the proofs are text-extracted (pypdf). Page markers and section page counts were not regenerated (the page count is unchanged at 43 / 44; the Methods paragraph adds about 10 lines to Sec. 2).
- No compute, no fits, no GPU. The bib, `outputs/w17-paper`, `outputs/pw1-*`, `outputs/w*/`, TABLES.md, the registry and the claims matrix were not written.
- An install attempt first failed harmlessly (zsh did not split a loop variable, so `cp` failed and the live files stayed untouched). It was re-run and verified; both attempts are logged in `LOG.md`.

## Branch and bundle
Tooling commit 2e52807 on `pw2/draft-methods` (`pw2_draft_methods/`). Bundle `pw2-draft-methods.bundle`, sha256 7a39a165…8b40; `git bundle verify` is okay, the history is complete, and the head is refs/heads/pw2/draft-methods = 2e52807. Outputs are not committed. At the end: bib b2cf715f… in both folders, and the live files still match `INSTALLED-SHA256.txt`.
