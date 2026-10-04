# PW-2 — REPORT FOR THE HEAD AND THE ADVISOR
**Paper lead Hub, 2026-09-25. Machine: batmac (spokes), Mac (the Head's folders). Six spokes, all finished and reaped. Zero runs, zero fits, zero GPU.**

## The two folders (edited in place)
- `/Users/user/Desktop/naive-CLU-final-v2fina_subF/`: `DRAFT-v2.tex` (sha256 4640c986…), `DRAFT-v2.pdf`, `refs_clu_iclr.bib`, `figures/`, `iclr2027/`
- `/Users/user/Desktop/naive-CLU-iclr-v2fina-stylesubF/`: `DRAFT-v2-style.tex` (sha256 feecbbf9…), `DRAFT-v2-style.pdf`, the same bib, `figures/`, `iclr2027/`

Both folders are sha256-equal to the batmac working copy the audit checked. The only file added to each is the built PDF, so each holds 21 files. The Head's bib was never written: it is b2cf715f… (56 entries) from start to end.
**Snapshots:** a dated before/after pair with sha256 for every install, in `.claude/outputs/pw2-desktop-snapshots/`: base 17:36, then sections, methods, numbers, tells and leftovers. Any earlier state can be restored from these.
Nothing was written to `outputs/pw1-paper/`, `outputs/w17-paper/`, `outputs/paper-visuals/`, `outputs/w*/`, TABLES.md, the registry or the matrix.

## 0. The answer
All five items are done in both drafts, in lock-step:
- both drafts compile with 0 undefined citations, 0 undefined references, 0 errors and 0 overfull boxes;
- the style diff-check is EMPTY: the number multiset is identical between the two drafts (3,362 = 3,362);
- provenance findings are identical to PW-1 final;
- all 56 bib entries are cited;
- `\TODO` is 3 and `\PLACEHOLDER` 0 per draft;
- all 7 Head-held paragraphs are byte-identical, apart from the borrowings you ordered.

**Pages: v2 42, style 43. Main text: 17.95 pp / 18.22 pp against 9.** Results alone is 8.57 / 8.69 pp. The nine-page cut is yours; no spoke shortened anything for length.

## 1. Item by item
| item | status | what happened |
|---|---|---|
| **2 re-sectioning** | **done** | Introduction → Methods → Experiments → Results → Discussion, then references and appendices. 62 (v2) / 64 (style) paragraphs were moved; none was lost, duplicated or rewritten, and the multiset of paragraph hashes is proved equal. The only new text is headings. Appendix headings and order are unchanged. Paragraphs mixing method and result were kept whole where their main content belongs (each logged). The maze figure and two maze tables stay in App. N: moving floats between main text and appendix is your page decision, and Results 4.8 cites them there. |
| **3 Methods opening** | **done** | Your text opens Methods character for character in both drafts; only `\citet{jawahar2026chlu}` and LaTeX typography were set. Scope audit: 27 CHLU / CLU / dynamics mentions per draft; 3 citations of `jawahar2026chlu` (Methods ×2, and Discussion's "zero-dynamics limit of the CLU"), all for the unit and its dynamics; **0 violations**. The shorts are cited nowhere. The Methods sentence naming "a factorized memory unit … deletion" carries no citation and attributes both to "we". |
| **5 numbers** | **done** (outside Head-held text) | Main-text digit tokens went from 586 to about 200 per draft, and 70 of those remaining sit in Head-held paragraphs. Abstract: 49 numeral tokens → **5 quantities** (2 are Head-held: "three things at once" and the maze `\TODO`'s "six-seed"). The other three are "all three properties", "about seven more … than the strongest, an MLP" (record +7.034) and "about six tuples" (records 6.13 / 6.43 / 6.30). 97 numbers moved: 71 into tables or captions that already carried them, 19 added to existing captions or tables, 7 kept only as `%` pointers. Rounded phrases: 51 checked, then re-checked independently by the audit (32 in its taxonomy), **0 FAIL**. No rounding goes up past its record; "nearly" is used where a record falls just short (e.g. "nearly a gigabyte" for 0.967 GB). The limitations list became editable only after the item-1 borrowing, so it was done in a follow-up: 27 → 10 digit tokens, all identifiers. |
| **1 prose tells** | **done** (outside Head-held text) | Subtractive only: v2 −366 words, style −427, and no section grew except v2 Methods (+2), because of the borrowed signpost you ordered. No connective was added. Counts are in §3. The three borrowings from the style twin were folded into v2: the abstract's first sentence, the Methods and Discussion signposts, and "We claim neither … nor". |
| **4 leftovers** | **done** | (a) Ladder legend now "seed at $10\times$ training steps". This is a one-string change: rebuilt twice byte-identically, installed into both folders' `figures/`, with `paper-visuals/` untouched. (b) Figs. 5–8 are each cited from the sentence that states their result; none was moved. (c) The byte-ratio `\TODO` sentence was dropped, with a pointer, taking `\TODO` from 4 to 3. (d) Both observation-only `\PLACEHOLDER`s were cut (2 → 0), and one appendix sentence now says those models were not persisted. (e) Absolute machine paths in comments: 2 → 0 per draft, and the provenance checker still resolves every source. (f) The deletion-scope quote is now at body size: 8.97 pt → 9.96 pt. |

## 2. Decisions only the Head can make (nothing here was changed)
1. **Head-held paragraphs still carry per-seed numbers and tells.** Numbers: Experiments' first paragraph has 32 digit tokens (33 in style) and Results' first paragraph 38. Tells, by paragraph:
   - abstract opening: "three things at once" (triad);
   - Introduction: "each is cited in one sentence below …" (self-narration) and the borrowed "We claim neither … nor" (contrast);
   - Experiments: "was not extended to manufacture a crossover";
   - Results: "measured, not assumed" and "the grids were not widened after the fact";
   - Discussion, the borrowed signpost: "naive unit", "the one measured clue" and the opener "We now".
2. **Rounding to check:** 32.1 became "about thirty" (limitations list, red with other shapes at seed 0).
3. **Coinage conflict:** Methods (your text) says Jawahar & Pierini introduced the CHLU "as a latent dynamics model (defined here as …)". Sec. 5.4 says "We call the class … a *Latent Dynamics Model*" and defines it differently.
4. **Names:** "CHLU" (Methods), "the CLU" (Secs. 5.3, 5.6) and "a settling unit" (abstract) name the same unit, and the paper never states that CLU = CHLU.
5. **App. L "Prior derivations":** with Methods now citing your arXiv paper, a reader could take "prior derivations" to mean that paper. The section does open "The laws below are this paper's own theory", so it is not a violation. Renaming the heading is yours, since appendix headings were excluded.
6. **Bib rendering:** your entry prints "Chlu: The causal hamiltonian learning unit …". The title needs `{CHLU}` and `{Hamiltonian}` braced. `ha_world_2018` still has no journal (the one bibtex warning).
7. **Cross-references in Head-held text:**
   - The abstract `\TODO` uses `\S\ref` and `Table~\ref`, the only non-conforming references in either draft.
   - The Introduction's first paragraph points "matched on the read path" at Sec. 2 (now Methods). The matching rule now sits in Experiments 3.2.
8. **Placement:** Figs. 6–8 are cited from Results but print on appendix pages (27, 29, 31 in v2), because their source sits in Apps. D, F and H. App. R "Placeholder index" now indexes no placeholder.
9. **Seed tallies in prose,** e.g. "ahead on three of six seeds, level on two, behind on one", were classed as high-level counts. You may prefer them in captions.

## 3. Per-tell counts (v2, before → after, whole main text; style in brackets; `[h]` = inside Head-held text)
| tell | before → now (v2) | style | left in Head-held text |
|---|---|---|---|
| T1 triads | 6 → 5 | 5 → 5 | 1 (the "three things at once" frame). The other 4 are genuine lists of three or four |
| T2 contrasts | 49 → 19 | 49 → 18 | 2; at most one editable contrast per section |
| T3 self-narration | 38 → 5 | 37 → 4 | all 5 / 4 |
| T4 numbers in prose | 49 → 37 | 49 → 37 | 30 |
| T5 headings | 17 → 0 | 17 → 0 | 0 |
| T6 coinages | 12 → 7 | 16 → 7 | 3 |
| T7 rhythm (paragraphs with no short sentence after a long one) | 97 → 52 | 98 → 54 | 5 / 6 |

Per section, main text (v2 before → now):

| section | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| Abstract | 1→1 | 1→1 | 2→0 | 0→0 | 0→0 | 2→1 | 1→0 |
| Introduction | 2→2 | 5→2 | 7→2 | 1→1 | 0→0 | 4→1 | 9→1 |
| Methods | 0→0 | 0→0 | 1→0 | 2→2 | 2→0 | 0→0 | 6→3 |
| Experiments | 1→1 | 4→1 | 6→1 | 12→12 | 2→0 | 0→0 | 6→2 |
| Results | 0→0 | 14→2 | 12→2 | 22→21 | 11→0 | 5→3 | 19→9 |
| Discussion | 1→0 | 5→1 | 0→0 | 12→1 | 2→0 | 1→2 | 12→11 |
| Statements | 0→0 | 1→1 | 0→0 | 0→0 | 0→0 | 0→0 | 3→3 |

Every appendix row, and the style twin's table: `.claude/outputs/pw2-tells-per-section.md`. Detectors and positive controls: `pw2-draft-tells/TELLS-COUNT.md`.
Discussion's T6 rose (1 → 2) only because of the borrowed signpost ("naive unit", "the one measured clue"), which is Head-held.

## 4. Every number left in the main text
The full list, token by token, with line, sentence and justification, is in `.claude/outputs/pw2-numbers-left-DRAFT-v2.md` and `…-DRAFT-v2-style.md`. It was re-derived independently by the final audit, with **0 unclassified**.

| section (v2) | digits | words | Head-held | identifier | protocol / definition | rounded | count |
|---|---|---|---|---|---|---|---|
| Abstract | 1 | 8 | 2 | 3 | 0 | 2 | 1 |
| Introduction | 8 | 30 | 2 | 9 | 0 | 4 | 22 |
| Methods | 13 | 5 | 0 | 9 | 4 | 0 | 4 |
| Experiments | 37 | 19 | 36 | 5 | 0 | 0 | 15 |
| Results | 121 | 85 | 50 | 77 | 10 | 18 | 50 |
| Discussion | 18 | 38 | 1 | 22 | 4 | 7 | 20 |
| **total** | **198** | **185** | **91** | **125** | **18** | **31** | **112** |

(6 more tokens are not numbers, e.g. the pronoun "one".)
- **Identifiers** are names, not results: seed indices, rung names $n=2…6$, $K^\star=32$, $K=5$, "6-atom / 10-atom", "Shapes3D", row 3 / row 4, list enumerators, citation years, dtype names.
- **Protocol / definition** covers the declared margin 1.0, the reference definition 524,288 B / 131,072 float32, "64 checks", "256 requests", "coverage share is 1 by construction".
- **Rounded** values are each checked against their record. Example: "about seven more … than the strongest matched baseline, an MLP" (+7.034).
- **Counts** are the paper's frame ("three properties", "six seeds") and high-level seed tallies.

**Abstract (5 quantities):**
1. "three things at once" (Head-held);
2. "all three properties";
3. "about seven" (+7.034);
4. "about six" (6.13 / 6.43 / 6.30);
5. "six-seed" (Head-held `\TODO`).

## 5. Figure and table index (v2; style numbers identical)
| # | label | caption lead | p. |
|---|---|---|---|
| Fig. 1 | fig:task-shapes | The Shapes3D task. | 5 |
| Fig. 2 | fig:task-cmazet | The CMAZE-T task. | 6 |
| Fig. 3 | fig:shapes-per-model | Each model succeeding and failing on held-out combinations (Shapes3D, seed 0, K*=32). | 9 |
| Fig. 4 | fig:shapes-baseline-wins | Held-out requests on which a baseline succeeds and the n-CLU fails (seed 0). | 10 |
| Fig. 5 | fig:ladder | The ladder at K* against data-matched baselines, 3 seeds. | 12 |
| Fig. 6 | fig:crossover | The K-curve, validation, 3 seeds at every K. | 27 (App. D) |
| Fig. 7 | fig:deletion | Deletion on the single design at K*, rows 3–4, test, fit pool 131,072 rows. | 29 (App. F) |
| Fig. 8 | fig:adapt | Encoder adaptation margins. | 31 (App. H) |
| Fig. 9 | fig:maze-wm | CMAZE-T: the n-CLU against the best matched baseline under four input choices … | 36 (App. N) |
| Fig. 10 | fig:cmazet-families | Held-out CMAZE-T transitions, one per rule family … | 38 |
| Fig. 11 | fig:cmazet-per-arm | Each CMAZE-T model succeeding and failing (seed 0). | 39 |
| Fig. 12 | fig:cmazet-rel-per-arm | … with relational coding. | 40 |
| Fig. 13 | fig:cmazet-door-tint | Door-then-tint transitions. | 41 |
| Tab. 1 | tab:t7 | Composition at K*=32 against data-matched baselines, TEST, 3 seeds. | 8 |
| Tab. 2 | tab:ledger-kstar | The footprint breakdown of the K*=32 object. | 10 |
| Tab. 3 | tab:ladder | The ladder at K*=32, per seed. | 11 |
| Tab. 4 | tab:del | Deletion on the single design, the four measurements. | 13 |
| Tab. 5 | tab:hole | Refusal DISPLAY rows. | 14 |
| Tab. 6 | tab:gentable-kstar | The generation table, K* single design. | 25 |
| Tab. 7 | tab:lace-poe | LACE and PoE at their own settings. | 25 |
| Tab. 8 | tab:k5 | Ladder at K=5. | 26 |
| Tab. 9 | tab:kcurve | The K-curve, validation. | 26 |
| Tab. 10 | tab:kcurve3 | The K-curve at 3 seeds on every row. | 27 |
| Tab. 11 | tab:impl | Deletion on Shapes3D, implementations 1 and 2. | 28 |
| Tab. 12 | tab:ablations | The two ablations, earlier composition cell. | 29 |
| Tab. 13 | tab:rivals-del | Deletion cost for the data-matched baselines. | 30 |
| Tab. 14 | tab:rivals-del-seeds | The n-CLU after deletion against each retrained baseline. | 30 |
| Tab. 15 | tab:adapt | Encoder adaptation at K*. | 30 |
| Tab. 16 | tab:encloop | Encoder loop, shortened decoder. | 31 |
| Tab. 17 | tab:bard-gpu | The unfreezing decomposition on the longer leg. | 32 |
| Tab. 18 | tab:crr | Declared criteria of the retention law on the field-mode n-CLU. | 33 |
| Tab. 19 | tab:cmazet (+tab:cmaze-del) | Maze world-model rows: CMAZE-T, base n-CLU, 3 seeds. | 37 |
| Tab. 20 | tab:cmazet-sym | CMAZE-T with symmetric inputs. | 42 |

The style twin's pages are in `.claude/outputs/pw2-final-index.md` §4.

## 6. Page counts per section (start page, length in pp; v2 / style)
| section | v2 | style |
|---|---|---|
| title + abstract | 1, 0.86 | 1, 0.86 |
| 1 Introduction | 1, 1.93 | 1, 1.97 |
| 2 Methods | 3, 2.03 | 3, 2.00 |
| 3 Experiments | 5, 2.58 | 5, 2.64 |
| 4 Results | 8, 8.57 | 8, 8.69 |
| 5 Discussion | 16, 1.99 | 17, 2.05 |
| AI use / ethics / reproducibility statements | 18–19, 0.46 | 19, 0.48 |
| References | 19, 4.60 | 19, 5.32 |
| Appendices A–S | 24–42 | 25–43 |
| **Main text (to AI use statement)** | **17.95 (+8.95 over 9)** | **18.22 (+9.22)** |
| **Total** | **42** | **43** |

Subsections and every appendix: `.claude/outputs/pw2-final-index.md` §5.

## 7. Every remaining `\TODO` and `\PLACEHOLDER` (identical in both drafts)
| # | kind | where | argument |
|---|---|---|---|
| 1 | TODO | abstract (Head-held) | CMAZE-T sentence reserved for the Head … |
| 2 | TODO | AI use statement (MF-12, Head-held) | the authors to confirm which of the venue's required disclosure categories apply … |
| 3 | TODO | Reproducibility statement | anonymised code archive not yet built. |

`\PLACEHOLDER`: **0**.

## 8. What did NOT run
- No length cut: that is your pass.
- No edit to any Head-held paragraph, except the ordered borrowings.
- No edit to the bib, appendix headings or appendix order.
- No edit to `paper-visuals/` or `w17-paper/`. The rebuilt `fig_ladder.pdf` lives only in the two Desktop folders and `outputs/pw2-draft-leftovers/`.
- No page-by-page visual review of the PDFs. The checks were text extraction, build logs, `.aux` labels and page markers.
- No re-verification of numbers inside tables and captions beyond the provenance checker.
- The arXiv CHLU paper was not re-read; the scope audit works sentence by sentence on the drafts.
- No compute of any kind.

## 9. Record
| spoke | branch @ commit | item |
|---|---|---|
| pw2-draft-sections | 6b431af | 2 |
| pw2-draft-methods | 2e52807 | 3 |
| pw2-draft-numbers | 83596f2 | 5 |
| pw2-draft-tells | 5a3d1a2 | 1 + borrowings |
| pw2-draft-leftovers | 5594b5e | 4 + the item-5 follow-up on the limitations list |
| pw2-draft-final | 9da842a | audit (11/11 points, 11/11 positive controls caught; no defect, no install) |

Spoke records: `~/clu/outputs/pw2-<spoke>/` on batmac, mirrored at `.claude/outputs/pw2/`. Hub log: `.claude/outputs/pw2-hub-log.md`. Tasks: `.claude/tasks/pw2/`.
**PW-2 is closed.**
