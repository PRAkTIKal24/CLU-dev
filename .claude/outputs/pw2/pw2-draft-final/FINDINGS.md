# FINDINGS — pw2-draft-final (the closing audit of PW-2, BOTH drafts)

Session `pw2-draft-final`, branch `pw2/draft-final` (from `pw2/draft-leftovers` 5594b5e), batmac. Start 2026-09-25T17:51:36Z.
The deliverable is **`FINAL-INDEX.md`**. It has eleven sections, one per brief point, plus the positive controls. The live drafts were audited read-only.
- Reproducible stage: `final-20260925T180521Z/`, built by `pw2_draft_final/run_final.sh`.
- A first, hand-driven pass is kept in `final-20260925T175221Z/`. It reached the same results.
- Full per-token number lists: `NUMBERS-NOW-DRAFT-v2.md` and `NUMBERS-NOW-DRAFT-v2-style.md`.
- All tells and all sections: `TELLS-NOW.md`.

## Status: all eleven points DONE, both drafts. No content edit. No mechanical defect was found, so nothing was fixed or installed.

| # | brief point | status | result |
|---|---|---|---|
| 1 | Full compile from a clean copy | done | Both drafts: exit 0, 0 undefined citations, 0 undefined references, 0 errors, **0 overfull**. Underfull: v2 16, style 20 (every one listed). One bibtex warning, verbatim: `Warning--empty journal in ha_world_2018`. Pages: v2 **42**, style **43**. The installed PDFs extract to the same text as the fresh builds, page by page. |
| 2 | Style diff-check | done | **ALL EMPTY.** Number multiset **3,362 = 3,362** (1,758 distinct), lost ∅, added ∅. Placeholders 0 = 0. Artefact paths 607 = 607. 100 headings and 67 labels in identical order. |
| 3 | Provenance vs PW-1 final (`pw1-draft-citations`) | done | **IDENTICAL** in both drafts, line numbers masked: v2 has 3 unsourced lines and 9 not-found tokens, style 3 and 10, all present at PW-1 final. There were no regressions and none were resolved. Checked tokens fell 2,597 → 2,342 (item 5). |
| 4 | Figure/table index | done | 13 figures and 20 tables in each draft, 34 labels (Tab. 19 carries both `tab:cmazet` and `tab:cmaze-del`). Every label agrees with its lof/lot entry. Numbers are identical between the drafts; 31 floats sit on a different page in the style twin. |
| 5 | Page counts per section (method C) | done | Main text (p. 1 → AI use statement): **v2 17.95 pp (+8.95 over 9)**, **style 18.22 pp (+9.22)**. Results alone: 8.57 / 8.69 pp. Every section, the statements, the references and each appendix are listed in §5. |
| 6 | Every number left in the main text, re-derived independently | done | New scanner and classifier (no code shared with pw2_draft_numbers): v2 383 tokens, style 381. **0 UNCLASSIFIED.** 32 rounded phrases re-checked verbatim against their records, **0 FAIL**. The abstract holds 5 quantities. **One disagreement of substance** with `NUMBERS-LEFT.md`: "more than **eighteen** tuples" (Results, Item memory) is not in that list; its record holds (18.56 > 18). Every other difference is tokeniser or taxonomy (§6). |
| 7 | Per-tell counts per section, now | done | Totals T1…T7, now: v2 5·19·5·37·0·7·52, style 5·18·4·37·0·7·54. pw2-draft-tells' after: v2 5·20·5·48·0·7·51, style 5·19·4·48·0·7·53. Three sections changed since that spoke, all through pw2-draft-leftovers: Discussion T4 12 → 1; App R T2 1 → 0 and T7 0 → 1. |
| 8 | Head-held paragraphs | done | **7 of 7 byte-equal in both drafts**, by two independent checks: sha256 (`headheld.py`) and raw-line bytes against the draft that HEAD-HELD.md was written from. |
| 9 | Remaining `\TODO` / `\PLACEHOLDER` | done | `\TODO` 3, `\PLACEHOLDER` 0 per draft, byte-identical arguments: the abstract's maze sentence (Head-held), the AI use statement, and the anonymised code archive. |
| 10 | `jawahar2026chlu` citation scope | done | 3 citations per draft: Methods ×2 for the unit, Discussion 5.3 for the unit's zero-dynamics limit. **0 VIOLATION.** The Methods sentence on "memory unit … deletion" carries no citation and attributes both to "we". |
| 11 | Bib sha256 | done | `b2cf715f…` in both live folders at start and end, 56 entries. Never written. |
| — | Positive controls | done | **11 of 11 CAUGHT.** They cover: a planted undefined ref and cite (compile), a style-twin number change, an unsourced number, an unclassified number, a wrong rounding, one space in Head-held text, a planted `\PLACEHOLDER`, a retention-law citation, the seven tells, a planted path and doubled word, and a one-byte bib change. |

## Before → after (the audit's own comparisons)

| measure | reference state | now (v2 / style) |
|---|---|---|
| provenance: unsourced lines / not-found tokens | PW-1 final: 3 / 9 (v2), 3 / 10 (style) | 3 / 9, 3 / 10 (identical findings) |
| number multiset between the drafts | before item 5 (pw2-draft-numbers' snapshot): 3,631 = 3,631 | 3,362 = 3,362 |
| `\TODO` / `\PLACEHOLDER` uses per draft | before item 5 (pw2-draft-numbers' snapshot): 4 / 2 | 3 / 0 |
| pages (total / main text) | pw2-draft-sections' after: 43 / 44 total | 42 / 43; main text 17.95 / 18.22 pp |
| main-text digit tokens (pw2-draft-numbers' scanner / this audit's) | before item 5: 586 per draft | 174 per draft (theirs, leftovers' list) / 198 and 199 (this audit; the extra ones are digits glued to names) |
| tells totals T1…T7 (v2) | before PW-2 item 1: 6·49·38·49·17·12·97 | 5·19·5·37·0·7·52 |

## For the Head (reported, not edited: no spoke may edit these)
1. **Length.** The main text runs 17.95 pp (v2) and 18.22 pp (style) against 9, and Results alone is 8.57 / 8.69 pp. This is the Head's nine-page pass.
2. **Head-held paragraphs still carry per-seed numbers.** The Experiments first paragraph has 32 digit tokens in v2 (33 in style) and the Results first paragraph 38. They include 30 of the 37 remaining T4 hits and 5 of the 5 (v2) / 4 of the 4 (style) T3 hits. Only the Head can move them.
3. **Head-held abstract `\TODO` uses `\S\ref` and `Table~\ref`.** The vocabulary rule asks for `Sec.~\ref` and `Tab.~\ref`. This is the only non-conforming cross-reference in either draft.
4. **Figs. 6–8 are cited from the main text but print on appendix pages** (pp. 27, 29, 31 in v2), because their source sits in Apps. D, F and H. Fig. 5 prints in Results.
5. **App. R "Placeholder index" indexes no placeholder.** Its one paragraph now fails T7, since the cut removed its short sentence (pw2-draft-leftovers' Head note 2).
6. **Seed tallies in prose.** Examples: "ahead on three of six seeds, level on two, behind on one" and "on two of three seeds". They are classed COUNT (high-level) here and in NUMBERS-LEFT. Each can be read off its cited table, so the Head may prefer them in captions.
7. **Rounded phrases that NUMBERS-LEFT classed as counts:** "about four thousand", "about three months", "four times the n-CLU's content rows", and in the style twin "more than four of the six claim atoms". Each is now checked against its record: all PASS.

## Choices logged
- **Main text** runs from the top of p. 1 (title and abstract) to the "AI use statement" heading, as in pw1-draft-final. Introduction → AI use is also printed: 17.10 / 17.36 pp.
- **Latest HEAD-HELD.md** is pw2-draft-leftovers' copy (identical to pw2-draft-tells'). Its raw-line check compares against leftovers' `snap-start/`, which is cmp-equal to pw2-draft-tells' install (sha256 15161cfa… for v2).
- **"PW-1 final state"** means pw1-draft-citations' installed drafts (`stage-145204/post-install/`, equal to `outputs/w17-paper/`). The current `check_provenance.py` was run on both.
- **Number classes** are this audit's own taxonomy. For comparison, both taxonomies are mapped onto five coarse classes, and "high-level seed count" is treated as a count.
- **Doubled words in the rendered PDF text** (17 in each draft) all come from figure panels and adjacent table cells. The source prose has 0, so they are not defects.

## What did NOT run
- No edit of any kind to the live drafts, the bib, the figures or any other spoke's outputs. No mechanical defect was found, so the lock guard (checked: no other draft/figures session live) and the install method were not needed.
- No page-by-page visual review of the PDFs. The checks are text extraction (pypdf), build logs, aux labels and shipout markers. The title was checked as 2 rendered lines by text only, not by image.
- No re-verification of numbers inside tables and captions against their records beyond what `check_provenance.py` does. The rounded-phrase check covers the main text only.
- The tells detectors were re-used, not re-derived. Only the numbers were re-derived independently, as the brief asks.
- The arXiv CHLU paper was not re-read. The citation-scope audit is sentence-level on the drafts (the topic rule of PLAN §2 item 3).
- No compute, fits or GPU. Nothing was written to `outputs/w17-paper`, `outputs/pw1-*`, `outputs/paper-visuals`, `outputs/w*/`, TABLES.md, the registry or the claims matrix.

## Branch, bundle, end state
- Tooling commit **9da842a** on `pw2/draft-final`, in `pw2_draft_final/`. Outputs are not committed.
- Bundle `pw2-draft-final.bundle`, sha256 ae06f864…2c0e. `git bundle verify` exits 0 with complete history; head refs/heads/pw2/draft-final = 9da842a.
- At the end, the live `.tex`, PDFs and bibs are byte-identical to the start (`SHA256-live-end.txt` against the stage's `snap/SHA256-live-start.txt`). The bib is b2cf715f… in both folders.
