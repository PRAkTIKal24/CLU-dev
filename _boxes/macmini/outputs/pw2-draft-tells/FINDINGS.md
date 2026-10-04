# FINDINGS — pw2-draft-tells (PW-2 item 1: the seven machine-prose tells, subtractive, BOTH drafts, + the item-4 borrowings)

Session `pw2-draft-tells`, branch `pw2/draft-tells` (from `pw2/draft-numbers` 83596f2), batmac. Snapshot `snap-start/` (sha256 in `snap-start/SHA256`; equal to pw2-draft-numbers' `INSTALLED-SHA256.txt`). Stage `stage-*/` (path in `CURRENT_STAGE`); install verified in `verify-live-*/`.

## Status: item 1 DONE in both drafts, installed; the three borrowings folded into v2. Residuals are listed below, each with its reason.

| brief point | result |
|---|---|
| T1 triads | Detector hits 6 → 5 (v2), 5 → 5 (style). Two rhetorical triads of one claim were cut to the claim: "they occupy no row …, they beat no model, and we make no capability claim" in Discussion 5.4 (D10) and App P (P31). The anaphoric "the same content, the same wells, the same read" became "the same content, wells and read" in v2 (D5); the style twin already read that way. The remaining hits are lists whose content really is three or four (three prior-work examples; four screen sources; four named controls; "different code, different authors and different sign conventions"), plus the Head-held "three things at once" (below). |
| T2 contrasts, ≤ 1 per section | 49 → 20 (v2), 49 → 19 (style). Every main section and every appendix now has at most one editable contrast; the two Head-held ones are extra (table below). |
| T3 self-narration | 38 → 5 (v2), 37 → 4 (style). **All remaining hits are Head-held** (below). Every one of the brief's seven examples is gone from editable text. |
| T4 numbers (re-check only) | Main-text hits 49 → 48 in both drafts. Of the 48, 30 are Head-held. The others are identifiers or definitions (2024, the 524,288 / 131,072 reference definition, rung names "3--6", "rows 3--4", "256 requests"), plus the limitations-list numbers (next line). |
| T4, finding for the hub/Head | v2's limitations list was Head-held during item 5 (it was v2's Discussion first paragraph), and item 5 held the style copy byte-identical for lock-step. The B3 borrowing makes the style signpost v2's Discussion first paragraph too, so **the list is no longer Head-held in either draft**. It still carries 6.13 / 6.43 / 6.30, 32.1, 2,367 to 5,913, 4.0768 to 4.0897, 3.9942, 0.991, 0.0648, 524,288 and 1.47. The brief confined this pass to a re-check, so none of these was moved. A small item-5 follow-up with `pw2_draft_numbers` (rounded values verified against their records) would clear them. |
| T5 headings (main text) | 17 → 0 in both drafts. See EDITS.md M3, M6, X1, X7, R2, R4, R13, R22, R23, R25, R27, R30, R32, R35, R37, D4, D8. Appendix headings were not touched. |
| T6 coinages | 12 → 8 (v2), 16 → 8 (style). "leaving note" is 0 and "Thickness" is 0 in both drafts. "naive unit" after its first mention: the non-Head-held ones became n-CLU (A2, I15, D11); the remaining ones are Head-held. Paragraphs with ≥ 5 n-CLU tokens: 6 → 4 in each draft (abstract 8, inference note 8, relational coding 10, CMAZE-T claims 10). The rest of those tokens are set names ("n-CLU-content set", "n-CLU-REL"), the Head-ruled phrase "matches an n-CLU that was never fitted", or places where a pronoun would be ambiguous. |
| T7 rhythm | Paragraphs with no short sentence after a long one: 96 → 50 (v2), 94 → 51 (style). "This section" openers: v2 1 → 0; style 2 → 1 (the remaining one is Head-held). Connective openers: v2 0 → 1, style 2 → 1 (both are the Head-held "We now", see Rises). **Every split is a split of an existing sentence.** Residual paragraphs are listed below. |
| Head-held paragraphs | byte-identical, except the borrowings. Style: `headheld.py` PASS 7/7 against the old list, and 7/7 against the new `HEAD-HELD.md`. v2: exactly 3 rows changed (the abstract opening, the Introduction first paragraph and the Discussion first paragraph, all by borrowings); PASS 7/7 against the new list |
| The three borrowings (plan item 4) | done in v2 (4 edits; the abstract's first sentence counts as the Head-held opening): printed below |
| Word count ≤ 0 per section | Every section fell or stayed equal, except **v2 Methods +2 words** (997 → 999). The borrowing B2 adds the style twin's 29-word Methods signpost there, and the tell edits in Methods removed 27 words. Whole document: v2 25,817 → 25,451 words (−366); style 25,981 → 25,554 (−427). `WORDCOUNT.md` |
| No clause or connective added | Source audit (`audit.json`, `EDITS.md`): in both drafts, the whole-document net word change **equals** the sum of the listed edits' changes. Outside the borrowings, the only added words are:<br>• heading restatements: "Colour relations under the …", "The counts as images", "Conjunction deletion and the …", "Claims not made on CMAZE-T";<br>• "The task" (replacing "This section");<br>• "apply to" (replacing "travel with");<br>• the pronouns it/its (4);<br>• "n-CLU" (replacing "naive unit");<br>• "and" in "content, wells and read";<br>• "not … or" (replacing "neither … nor", 1 in each draft) and "no … and no" (style App L).<br>No connective (therefore, hence, thus, however, so, moreover) was added. The borrowed text carries "therefore" and "now", as the Head chose it. |
| Compile (COMMON) | Both drafts: exit 0; 0 undefined citations; 0 undefined references; 0 errors; 0 multiply defined; **0 overfull**. Underfull: v2 17 → 18, style 21 → 18, all `\vbox … while \output is active`. 56 bibitems. No Rerun. Pages: v2 43 → 42, style 44 → 44 (`COMPILE.txt`) |
| Style diff-check | **ALL DIFFS EMPTY**:<br>• number multiset 3,377 = 3,377 (1,761 distinct). Before this pass it was 3,381. The 4 tokens that left both drafts are the repeated quotation of the inference note ("2-way → 6-way … rungs 3--6"), deleted with its "travels with" announcement (R11); the note itself opens the paragraph before.<br>• placeholders, artefact paths, SRC strings: equal.<br>• heading sequence: 100 = 100 and identical (101 before; "Toward a CMAZE suite." removed, D4). |
| Provenance | Identical to before in both drafts, with line numbers masked (v2: 3 unsourced lines, 9 not-found tokens; style: 3 and 10, all pre-existing) |
| `\TODO` / `\PLACEHOLDER` | Unchanged: 6 / 3 macro occurrences per draft, arguments byte-identical |
| Labels | 77 of 77 in each draft; 0 missing, 0 renumbered. Pages moved: v2 56 labels (the draft is one page shorter), style 14 (`LABELS.txt`) |
| Installed | `.tex` and PDF in `~/clu/pw2/live/{v2,style}`, cmp-equal to the stage. The lock guard was clean, and the live files were cmp-equal to the snapshot just before install. A fresh build from a copy of the live folders gives identical `\newlabel`s and the same metrics. sha256 in `INSTALLED-SHA256.txt` |
| Bib | `refs_clu_iclr.bib` is b2cf715f… in both folders at the start and at the end; it was never written |

## The borrowings: style source → v2 before → v2 after
1. **Abstract, first sentence (B1)**
   - style source: "A world model trusted with requests it has never seen needs three things at once: its outputs are *valid* under the task's judge, it *composes* held-out combinations of factors from factors it has seen, and it can *forget* specific training rows."
   - v2 before: "A world model that is to be trusted with requests it has never seen needs three things at once: its outputs must be *valid* under the task's judge, it must *compose* held-out combinations of factors from factors it has seen, and it must be able to *forget* specific training rows."
   - v2 after: the style source, byte for byte including its line breaks. Its sha256 is 95017d4b…, the style row.
2. **Sec. 2 (Methods) signpost (B2)**. Style source (its second Methods paragraph, after the Head's item-3 text): "A claim about any of the three properties is only as precise as the definitions of the object, its judge and its memory footprint; we therefore fix these here, before any result."
   - v2 before: absent.
   - v2 after: inserted verbatim at the same place.
   - Then, in both drafts, split at its semicolon for rhythm (M1): "… memory footprint. We therefore fix these here, before any result."
3. **Sec. 5 (Discussion) signpost (B3)**. Style source: "We now state what the naive unit does not do, the one measured clue, and the theory we carry."
   - v2 before: absent. v2's Discussion opened with the limitations list.
   - v2 after: inserted verbatim at the same place. It is now v2's Head-held Discussion first paragraph (sha256 2eb7da66…, the style row). Not edited, because it is Head-held in the style twin.
4. **"we claim neither … nor" (B4)**. Style source: "We claim neither the best compositional-generation system nor the best unlearning system."
   - v2 before: "We do not claim the best compositional-generation system or the best unlearning system."
   - v2 after: the style source. The Head-held Introduction paragraph differs from before in exactly this line (diff checked).

Mapping by content after item 2's renaming: "Sec. 2" is now Methods (the style signpost sits after the Head's item-3 text), and "Sec. 5" is now Discussion.

## Rises (no count rose in editable text)
- v2 Discussion, T6 1 → 2 and T7 connective opener 0 → 1. Both come from B3: the borrowed signpost carries "naive unit", "the one measured clue" and the opener "We now". It is Head-held in both drafts after the borrowing, so it was left as the Head chose it. v2 Discussion's own pre-existing T6 hit was fixed (D11).
- Per paragraph, two paragraphs that passed T7 before fail after. In each, a T3 deletion removed the paragraph's only short sentence, and no subtractive split restores one:
  - Methods, "The judge and the estimand": "Seed count is a column of every table." was deleted (M8, the brief's "every row prints its seed count").
  - v2 App L opener: "Status discipline is part of the section." was deleted (P20).
  - Their sections' T7 counts still fell: Methods 6 → 3, App L 13 → 7 (v2).

## For the Head: tells inside Head-held text (reported, not edited)
- Abstract opening, both drafts: the triad frame "three things at once" (T1). The brief says never write it; it is the Head's sentence, and B1 makes the two drafts carry the same one.
- Introduction first paragraph:
  - T3: v2 "each is cited in one sentence below, with the unit/system distinction stated"; style "we cite each in one sentence below and state the unit/system distinction".
  - T2: the borrowed "We claim neither … nor".
- Experiments first paragraph, both drafts:
  - T3: "the sweep range was not extended to manufacture a crossover";
  - T4: 12 numbers;
  - style only: T7c "This section measures the three properties on Shapes3D."
- Results first paragraph, both drafts:
  - T3 and T2: "was measured, not assumed";
  - T3: "the grids were not widened after the fact";
  - T4: 18 numbers.
- Discussion first paragraph (the borrowed signpost): T6 "naive unit" and "one measured clue"; T7b "We now".

## T2: the one contrast kept per section (editable text)
| section | kept |
|---|---|
| Abstract | "Deleting rows downdates the statistic, not the truncated factors" |
| Introduction | "\emph{Downdate the statistic, not the truncated factors}" (+ the Head-held "We claim neither … nor") |
| Methods | none |
| Experiments | "A lookup table of written items is a printed floor, never a baseline" |
| Results | "so the loss is the deleted content, not instability of the refit" (+ the Head-held "measured, not assumed") |
| Discussion | "so far a *law about the n-CLU*, not a *lever on the task*" |
| Statements | "it checks that a number is present in its record, not what the number means" |
| App A, B, C, D, E, J (v2), K, L, N, P, R, S | one each (TELLS-COUNT.md): "a declared absence, not a zero"; the LACE record quotation; "a property of K=5, not of the n-CLU"; "what the rule returns, not a K …"; "a property of the content, not of the downdate"; "not a grid point"; "(analysis, not a row)"; "parameter separation, not block structure"; "(validation only, not a winner)"; "on this memory, not the law"; "permanently blocked, not pending"; the quoted "deletion, not composition" |

The contrasts removed were each restated as their positive half, e.g. "X, not Y" → "X" and "neither A nor B" → "not A or B". Every one is in EDITS.md, tagged T2.

## T7 residual paragraphs (no split point gives ≤ 8 words without adding words)
v2: 50 (5 Head-held); style: 51 (6 Head-held). The full list, with sentence lengths, is in `T7-residual.json`. They fall into four groups:
- (a) paragraphs whose only split points leave a 9–17-word sentence, e.g. "Our route …" was fixed, but not "The comparator's job …", "Tab. 5 (App. F) prints both ablations per seed", "Whether the equality survives at K* is not measured" (9 words);
- (b) one- or two-sentence paragraphs whose long sentence comes last (App A seeds, App C, the CMAZE suite paragraph, the follow-up paragraph, the statements);
- (c) long list items and noun-phrase lists (the limitations items, App L's bullets, App S's not-run list) that do not split into sentences without adding a verb or subject;
- (d) data dumps (App E's K=5 smoke, App K).

The style item-memory split (R24) yields a 9-word sentence ("This n-CLU is a ceiling and not an arm."), one word over; in v2 the same split gives 8.

**Choice logged:** at a split I never added a subject, verb or connective. The only changes at a split are the capital and, where the joint was ", and"/", so"/", because"/", while", deleting that word. Many more paragraphs would pass with a pronoun subject added ("It is …"), but the brief forbids adding words at a split.

## Choices logged
1. **"At most ONE per section" (T2)**: the sections are the five main sections, the abstract, the statements, and each appendix (the brief's deliverable granularity). Head-held contrasts are counted but not charged against the one editable contrast; the kept contrast is the most load-bearing in each section.
2. **"X, never Y"** is counted with "X, not Y" (the brief lists "printed, never netted"). T2 and T3 are scanned in prose, list items and the quote, and not in captions or tables; T6 also scans captions and tables, as the brief asks.
3. **T5** covers main-text `\section`, `\subsection` and `\paragraph` run-in titles, since item 2 left the run-in titles to item 1. The limitation and contribution bold item leads are sentences inside list items, not headings.
4. **T7 thresholds**: short ≤ 8 words, as the brief says; long ≥ 18 words. The short sentence must directly follow the long one. List items are held to the rule only when they carry ≥ 40 words. `\TODO`/`\PLACEHOLDER` arguments are not prose, and `\paragraph` titles are not counted as sentences.
5. **"naive unit"**: kept where it names the class ("other naive units"); replaced by n-CLU where it names the n-CLU after its first mention. The Head-ruled deletion phrase "matches an n-CLU that was never fitted" was never pronominalised.
6. **"Thickness, stated in the introduction."** was dropped with its run-in label; the paragraph keeps its facts.
7. **"This section asks …" → "The task asks …"**, a plain restatement of two words. The style twin's Head-held "This section measures …" was not touched.
8. **"Two caveats travel with these rows: … travel with them"** → "The baselines retrain … so the … convergence flags … apply to them. Approximate unlearning …". This is the only verb substitution outside the headings.
9. **Style-only edits** (the style twin's own wording at the same places):
   - I7 "However," opener;
   - D12 "The leaving note states what the naive unit does not provide; the follow-up is therefore …" → "The follow-up is …";
   - P35 "neither … nor";
   - the style variants of each shared edit.
   There are no v2-only tell edits except where the style twin already had the fix (R15 split, P15 split, D5 triad).

## What did NOT run
- Item 5 was not re-run on the limitations list, whose numbers were released from Head-held by B3 (T4 finding above). No number was moved, rounded or added.
- No tell was fixed inside Head-held text, and the borrowed signposts were not reworded. The only change to B2 is the M1 split, made in both drafts.
- No appendix heading was changed (brief). Captions and tables were scanned for T6 only, and none had a hit.
- No paraphrase pass. No sentence was rewritten for style, and no sentence was added apart from the borrowings.
- The 50 / 51 T7 residual paragraphs were not fixed (reason above). The n-CLU-density residuals in 4 paragraphs per draft were not fixed where a pronoun would be ambiguous.
- No page-by-page visual review of the PDFs. Checks are source-level (detectors, word audit, style diff-check, provenance) plus the build logs, aux labels and a rendered spot check of 12 edited sentences (pypdf text). The rendered word diff (`WORDDIFF-*.txt`) is dominated by float reflow and was used only for spot checks.
- No page counts per section were regenerated (v2 is now 42 pages, style 44).
- No compute, fits or GPU. The bib, `outputs/w17-paper`, `outputs/pw1-*`, `outputs/w*/`, TABLES.md, the registry and the claims matrix were not written.

## Deliverables in this directory
- The brief's deliverables:
  - `TELLS-COUNT.md`: every tell × section × draft, before → after, with Head-held counts, the detectors and the controls;
  - `EDITS.md`: 127 edits, before → after per draft, tagged, with the words each adds;
  - `WORDCOUNT.md`;
  - `HEAD-HELD.md` (updated).
- Checks:
  - `STYLE-DIFF-CHECK.md`, `prov-{before,after}-*.txt`, `COMPILE.txt`, `LABELS.txt`;
  - `CONTROLS.txt` (planted tells: every detector CAUGHT; the brief's seven examples MATCHED; ALL CONTROLS PASS), with the planted draft in `controls/`.
- Supporting records:
  - `audit.json`, `T7-residual.json`, `INSTALLED-SHA256.txt`;
  - `WORDDIFF-*.txt`, `*-before.txt` / `*-after.txt` (rendered text);
  - `view/` (reading aids: comment-stripped v2 source, split proposals);
  - `snap-start/`, `work/`, `stage-*/`, `verify-live-*/`, `LOG.md`.

## Branch and bundle
Tooling commit 5a3d1a2 on `pw2/draft-tells` (`pw2_draft_tells/`). Bundle `pw2-draft-tells.bundle`, sha256 729c909b…d791. `git bundle verify` exit 0, complete history, head refs/heads/pw2/draft-tells = 5a3d1a2. Outputs are not committed.
