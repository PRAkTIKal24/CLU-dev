# pw2-draft-tells — PW-2 item 1: the seven machine-prose tells, SUBTRACTIVE, BOTH drafts (+ the three borrowings into v2)
Session `pw2-draft-tells` · branch `pw2/draft-tells` (from `pw2/draft-numbers`) · out `~/clu/outputs/pw2-draft-tells/`
**You hold the `.tex` lock.** Read COMMON.md, PLAN.md §2 item 1 and item 4's last bullet IN FULL, and `~/clu/outputs/pw2-draft-{sections,methods,numbers}/FINDINGS.md` + the latest `HEAD-HELD.md`.

## The one job
Apply the seven rules of item 1 **subtractively**: delete, split or plainly restate. ⛔ **Never paraphrase for style and never add connective tissue.** The Head ruled that a paraphrase pass made the style twin worse.
1. **Triads:** no list of three where the content is not three. Drop to the real count, and never write "three things at once".
2. **Contrast constructions** ("not X, but Y", "X, not Y", "neither … nor"): at most ONE per section; the rest become a plain statement of Y.
3. **Self-narration of discipline:** delete the announcement and keep the fact. Examples: "which we print", "every row prints its seed count", "cited in one sentence below with the distinction stated", "was not extended to manufacture", "measured, not hedged", "printed, never netted", "we state plainly".
4. **Numbers:** already handled by item 5. Only re-check that no number with more than two significant figures and no range remains in running text.
5. **Headings:** noun phrases, no colons-as-sentences, no parentheticals. Main-text subsections only; appendix headings are NOT changed.
6. **Coinages:** "naive unit" at first mention, then n-CLU (not five per paragraph); "the one measured clue", "leaving note" and "thickness" replaced by plain words or dropped.
7. **Rhythm:** within each paragraph, at least one short sentence (≤ 8 words) after a long one, made by SPLITTING an existing sentence, never by adding a new one. No paragraph opens with a connective (therefore, hence, thus, however, we now). No sentence opens with "This section".
- **Head-held paragraphs** (latest HEAD-HELD.md: the abstract's opening and the first paragraph of each main section) stay byte-identical. The one exception is the borrowing below, if it lands in one of them.
- **The three borrowings from the style twin, folded into v2** (plan item 4, last bullet): the abstract's first sentence, the Sec. 2 and Sec. 5 signposts (locate them in the style draft; after item 2 their sections are renamed, so map by content), and the "we claim neither … nor" sentence. Print each: style source → v2 before → v2 after. The abstract's first sentence counts as part of the Head-held abstract opening. Update HEAD-HELD.md.
- Scope: main text AND appendices for tells 2, 3, 6 and 7. Tell 5 on main-text headings only.
- **Lock-step:** apply the same fixes at the same places in both drafts. The style twin keeps its own wording where it already differs.
- Tell 6 in captions and tables too.

## Deliverables
- `TELLS-COUNT.md`: for EACH tell × EACH section (main sections and each appendix) × each draft, the count before → after. For each tell, give the detector (regex or rule) and its positive control.
- `EDITS.md`: every edit, before → after sentence, tagged with its tell.
- A word-count delta per section. It should be ≤ 0 everywhere: this pass only removes. Print any section that grew, with the reason.
- Compile checks per COMMON; installed.
**What would prove this wrong:** a new clause or connective added; a Head-held paragraph changed outside the borrowings; any tell count that rose.
