# pj-minimal-v5 — paper-writer report

**Task + acceptance criterion:** the HEAD-APPROVED minimal pass on `v5-palm/pj_sub.tex` — exactly the 12 approved text items + Part-1 figures + Part-3 appendix-only restorations + the `§`-symbol rule, **zero other diffs**; riders verbatim; orphan list empty; 11/11 figures; `submission.tex` untouched; appendices-changed table present.
**Status: done** — with **one honest miss, declared, not fixed**: a **third, pre-existing, non-enumerated overfull table** survives (§"Open items" A). Everything else passes.

⚠ **RECONCILIATION LIST — NEEDS AN OWNER (protocol §5 corollary, stated in the first 10 lines).** Seven items below need a Hub/Head decision, three of which are content decisions the Head owns: **(A)** the third overfull table (App D, pre-existing, patch staged unapplied); **(5a)** the App-E *lifecycle* row whose host claim is not in the paper; **(F)** the restored J&P bibliography entry as a de-anonymization surface in a blank-`\author{}` build. None was resolved unilaterally.

**DIAL DECLARATION (echoed): none — editorial; zero new numbers without ancestors.** Laundering control: n/a (no performance number produced). Falsifies: an added token with no ancestor line. Does NOT falsify: page growth, or losing a page-limit comparison — the task says page split is *reported, not optimized*.

**Deliverable #1:** `.claude/NIPSsubmission/v5-palm/BUILD-NOTE-R4.md` (10 sections; the Head's appendices-ADDED/CHANGED table is §5).

## What I did
- Read `AGENT_PROTOCOL.md`, the Positioning Charter (`philosophy-synthesis.md` L581–600, **C-1 in its REVERSED form**), `claims_matrix.md`, then the task file. Confirmed object md5 `6c1902f74ee9611d718cc65b9fd1a031` before touching anything.
- Wrote `.claude/scratch/pj-minimal-v5/minimal.py`: **30 exact-match, single-occurrence, asserted replacements**. Every assert passed first run. Blocks were **extracted programmatically** from the two authorised sources (base `submission.tex`; `pj-restore-R3-preserved/pj_sub_v5_R3.tex`) rather than retyped, so "verbatim" is mechanical, not eyeballed.
- Rebuilt ×2 with `/Library/TeX/texbin/pdflatex`, in-folder.
- Ran: two-way numcheck (forward token + forward **verbatim-line** + reverse loss), the A.4 never-quote sweep (35 patterns, positive-controlled), the `certified` per-occurrence table, the author-token sweep, the honest-scope count, the `§` count, the figure/placeholder inventory, the negatives-row count, the folder byte-untouched diff, and a `pdftotext` page-split walk.
- Wrote `BUILD-NOTE-R4.md`. Preserved `pj_sub.PRE.tex` + the 14-hunk `pass.diff` in scratch. Staged (**unapplied**) `appD-overfull.patch`.

## How I verified — commands and observed output
| check | command | observed |
|---|---|---|
| edits | `python3 minimal.py` | `edits applied: 30` (all asserts single-occurrence) |
| build | `pdflatex … pj_sub.tex` ×2 | `Output written on pj_sub.pdf (18 pages, 1743808 bytes)` |
| errors / undefined | `grep -c '^!'` · `grep -c 'ndefined'` | **0** · **0** |
| overfull vbox | `grep -c 'Overfull \vbox'` | **0** |
| overfull hbox | `grep -c 'Overfull \hbox'` | **1** — `lines 371--380`, 49.1588pt (App D; **pre-existing**, see below) |
| PRE-file control build | rebuilt untouched PRE in scratch | **3** overfull: `192–200` 177.50pt · `226–235` 152.59pt · `271–280` **49.1588pt**. The two enumerated tables are fixed; the third is the same box, untouched. |
| numcheck forward | `python3 numcheck.py` | 166 added lines, 264 distinct numeric tokens, **ORPHANS: 0** |
| numcheck verbatim | `python3 numcheck2.py` | **153 of 166 added lines appear verbatim in an ancestor**; the 13 hand-composed lines are exactly the 12 items + the 4 `§` conversions |
| numcheck reverse | same | tokens lost PRE→POST: **1**, and it is `0.8` from a deleted `\parbox{0.8\textwidth}` |
| A.4 zero list | 35-pattern loop incl. `13.9` | **0 hits on all 35**; positive control `107.77\|Blelloch\|certified\|…` → **20 lines** ⇒ live |
| `certified` | per-occurrence | **3**: l.80 literature-description (now correct), **l.124 the restored DENIAL**, l.156 Guo bib title. **0 affirmative.** |
| honest-scope | `grep -c 'store-level guarantee only'` | **1** |
| `§` | `grep -c '\\S'` · `grep -c '§'` | **0** · **0** (was 4) |
| figures | `grep -n includegraphics` · `grep -c framebox\|Placeholder` | **11/11** · **0** |
| negatives | row parse | **20** (11 + 9) |
| folder | `md5 * figs/*` PRE vs POST | only `pj_sub.{tex,aux,log,pdf}` differ; **`submission.tex` = `1d09…332`, untouched**; all 11 PNGs untouched |
| final md5 | `md5 pj_sub.tex` | **`594a6919d773b5e9e82b68af07816d0e`** |
| visual | `mutool draw` pp. 4, 16 | Fig 1 renders; the App-D table visibly crosses the right margin |

## Findings / results
**All 30 edits landed and are classified in `BUILD-NOTE-R4.md` §§1–3, 5.**
- **§-rule:** 4 conversions to `Sec.~\ref{}`; **no `\label` had to be added** — all four targets already had one. Added sentences use the same form (item 1 reads *"Guo et al.'s (2020) Sec.~2, Eq.~(1)"*).
- **Figures 11/11.** Main text gets the two referee MUST-restores (`fig1_damping_optimum`, `fig2_vault`); the two remaining placeholders became real figures with **their existing captions untouched**; the other 7 wired to their **base** appendix homes with **base captions verbatim**, including the four whose claim text is not in the paper (Head's rule: the caption states the result).
- **Two truncated tables fixed from the R3 bank, `\small` NOT used as the fix.** File-wide `\small` went **6 → 5** (three placeholder captions removed; the two restored tables carry the `\small` their *base* versions carry). The transposed App-B table carries **all 39 values unchanged** plus the restored `Mean ± sd` column.
- **12/12 text items executed**, incl. the restored **certified-`(ε,δ)`-unlearning denial** and the **encoder-exclusion sentence** — this closes the standing `pj-fidelity-v5-r2` finding B-2 ("inversion removed, denial not restored") and the A.3 Guo mis-statement in one edit each. **C-6 is now satisfied at both headline sites**: the `ℓ_θ/Δ<0.05` rider sits beside `3.77±0.23×`, and the `86.97±2.94×` first-passage counterpart sits beside the `107.77×` headline with its estimator's name.
- **⛔ struck item — branch executed: ROWS-RESTORE.** 20/20 negatives rows restored, so App-E's *"every negative result … is documented below"* is true as written and was **not** softened.
- **Part 3 complete:** App E 5 → **20** rows · App C **+2 tables** (the confinement/hop table is where the confinement control numbers live) · App B **+1 table** + the mean±sd · App D **+prior-art paragraph** (no-priority clause + the 4th Blelloch–Golovin attribution) **+R50/TTL** (`1.146→0.752`, `1.52×`, `0.75–0.77`) — ⛔ **not** in main text.
- **Appendices ADDED or CHANGED vs the Head's rewrite:** **A, B, C, D, E all CHANGED; NONE newly created.** Stated explicitly, as required.
- **Not restored** (base home = main text, or on the ⛔ list): seam sentence · Titans one-liner · abstract restorations beyond item 8b · trilemma dial · status headers · anonymization note · fdt-fine-print block · `T*` claim text · designed-symmetry §3.2 block · all A2 vocabulary fixes · all intensifier/garble edits. Full list in `BUILD-NOTE-R4.md` §6.
- **Page split (reported, not optimized):** 11 → **18** pages. Main text is **6 pages before and after** — every added page is figures or appendix tables/rows.

## Charter / matrix compliance
C-1 (reversed): **no audit-confession paragraph exists or was added**; J&P 2026 is cited for the primitive's introduction only, in third person. C-2: base's verification/evidence labels were left exactly as the Head's rewrite has them (relabelling was on the ⛔ list). C-5: item 8b adds the missing scale qualifier **in-sentence** in the abstract. C-6: see above. C-9/C-10: 20/20 negatives, appendix-maximalist restorations, nothing self-pruned. Claims matrix: the score sentence is restored in the **canonical wording** (*"external benchmarks won on their own headline metric = ZERO"*); no forbidden claim appears; no constant was altered.
**Related-work prose:** none was written by me this pass — Related Work changes are two ancestor-traced corrections (Guo, Mo), not new positioning.

## Git footprint
**None.** No tracked file was touched; all work is under `.claude/**` (gitignored). No branch, no commit.

## Open questions / follow-ups / risks (the reconciliation list — Hub to assign an owner)
1. **⚠⚠ A (blocking-ish, Head/Hub).** A **third** overfull table exists — App D laundering control, **49.16pt (~12%) into the right margin, page 16** — and it is **pre-existing** (proved by rebuilding the untouched PRE file) and **not enumerated**. I left it alone because the task forbids anything not enumerated *"however beneficial it looks"*, which is exactly the rule R3 broke. So *"0 overfull table boxes"* holds **for the two enumerated tables, not file-wide**. A **content-free** one-line column-spec fix (no caption, no values, no `\small`) is staged unapplied at `.claude/scratch/pj-minimal-v5/appD-overfull.patch`. **Please rule.**
2. **B.** Item 10's *"l.116 area"* contains no `3.77×`; the rider went to the real `3.77±0.23×` site (post-l.185). Reported, not guessed.
3. **C.** Item 11's *"first 107.77× site"* — the first *textual* occurrence is the abstract; I read it as the first **body** site (Sec. 3.2, post-l.112), matching the R3 bank's placement, since the abstract is governed by item 8b. Confirm.
4. **D.** The score sentence now appears **twice** (Limitations bullet + App-E row). This matches the base, which also carries it twice, but the *"Stated once:"* prefix now sits beside a second instance. Head may want the prefix on one only.
5. **E.** Author-token sweep returns **2**, not bibliography-only: l.158 (bibliography) and **l.56 (body)** — the Charter-mandated CLU continuity sentence *"introduced as CHLU by Jawahar & Pierini (2026)"*. **"Bibliography-only" is not achievable while the naming rule stands.** Reported rather than resolved.
6. **F (Head).** Item 5 puts real author names into a build with blank `\author{}` and scrubbed PDF metadata. Advisor-ratified (Add. 60), third person throughout — but it is a de-anonymization surface for an anonymized build.
7. **G (Head).** The App-E **lifecycle row** is restored per Part 3(a) but **its host claim is not in the paper** (the PROTECTED⇄ACTIVE→TRASH contribution was cut in the Head's rewrite). It reads as a negative about a mechanism the reader never meets. Keep → leave as is; cut → one-line delete, count drops to 19/20.

## Proposed handover updates (for the Hub)
- `v5-palm/pj_sub.tex` is at **md5 `594a6919d773b5e9e82b68af07816d0e`**, 426 lines, **18 pages**, 11/11 figures, 20/20 negatives rows, 0 placeholders, 0 errors, 0 undefined refs, 0 overfull `\vbox`, **1 pre-existing overfull `\hbox`** (App D, not enumerated, patch staged). `BUILD-NOTE-R4.md` is the pass record; `BUILD-NOTE.md` and `BUILD-NOTE-R3.md` are untouched.
- **Two standing fidelity findings are now closed by this pass:** `pj-fidelity-v5-r2` **A.3** (the Guo mis-statement) and **B-2** (the missing certified-`(ε,δ)` denial). The registers can be updated.
- **R3 bank status:** consumed only where authorised (the two truncated tables, the four appendix tables, the prior-art paragraph, the negatives rows). `pj_sub_v5_R3.tex` remains untouched in `.claude/scratch/pj-restore-R3-preserved/`.
- **The seven open items above need an owner at the review that accepts this report** — item 1 is the only one that changes the compiled artifact.
