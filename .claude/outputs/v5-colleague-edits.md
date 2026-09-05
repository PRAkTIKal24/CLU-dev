# v5-colleague-edits — paper-writer report

**Task + acceptance criterion:** execute the 13 Head-approved items from the colleague's V5 review (`.claude/outputs/v5-colleague-edit-list.md`) inside `pj_sub.tex`, with zero unattributable hunks, an empty orphan list, and main-text standalone-ness that improves.
**Status: done.**
**DIAL DECLARATION (echoed): Dials touched — NONE.** No experiment, config, registry or charter. One `.tex` file edited within an enumerated set; one build note; one report.

> ⚠ **RECONCILIATION LIST — THIS REPORT CONTAINS ONE, AND IT NEEDS AN OWNER (protocol §5 corollary).** See **§6.1 (decision needed, one character)** and **§6.3 (two pre-existing numeric mismatches now adjacent in one table, needing whoever owns the emergent-arm numbers)**. Both are OUTSIDE the approved set and were **not** touched.

---

## 1. Object, integrity, footprint

| | |
|---|---|
| Object | `.claude/NIPSsubmission/v5-palm/pj_sub.tex` |
| Boot `md5` | `c63a57fc910663dfa1e644b9b349ce6f` — **matched on disk before the first edit** ✅ |
| Boot `mtime` | `1787760290` — **re-checked immediately before applying, unmoved** ✅ (no concurrent Head edit) |
| **Final `md5`** | **`ca56fef3b86a2d5d17314f84e130df3a`** |
| Files written | `pj_sub.tex`; `BUILD-NOTE-COLLEAGUE.md` (deliverable #1, written **before** the file shipped) |
| Folder integrity | aggregate `md5` of every other file in `v5-palm/` = `96ebe09d1b08dd7a1e9fd8add60b1e76` **before and after** — `figs/`, `submission.tex`, `neurips_2025_ml4ps.sty`, `pj_sub_preEdit_stable.tex`, `pj_sub_buildcopy.*`, prior BUILD-NOTEs all byte-untouched ✅ |
| `~/Desktop/V5_PALM_Submission/**` | untouched — `paper.tex` still at the boot `md5`, `refs.bib` unchanged ✅ (read-only source of `refs.bib`/`neurips_2026.sty`/`figs` for the scratch build) |
| Live `pj_sub.pdf/.aux/.log/.out` | **not overwritten** (§7: `pj_sub.tex` is the only writable file) ⇒ ⚠ **the live PDF is now stale**; the verified rebuild is banked at `.claude/outputs/v5-colleague-edits/pj_sub.pdf` |
| Git footprint | **none** — no tracked file touched (all work under `.claude/**`) |

**Scope:** 13 items executed. **T.3 DECLINED by the Head → not executed**: l.52 is byte-identical (line `md5` before = after), so `atoms` and `superposed` are untouched, and `superposed` still occurs exactly 2× file-wide. **Hunks attributable to no approved item: ZERO.**

---

## 2. What I did (18 changed lines + 1 insert; nothing else differs)

Changed pre-pass lines: `35, 40, 42, 44, 57, 59, 74, 76, 78, 80, 89, 93, 95, 97, 107, 119, 120, 123`, plus the R.0 insert after `134`.

- **T.1** (l.42) `after the fact` → `even after the entry is nominally deleted`; `\citep` and surrounding words byte-identical.
- **T.2** (l.44) framing sentence replaced with the Head's wording verbatim. ⛔ **Guard held: `grep -ci interpret` → 0** (control `prescribed` → 1).
- **T.4** (l.59) `\item \textbf{Nomenclature borrowing from and building on CHLU~\cite{jawahar_chlu_2026}:}`; block **not** relocated, sub-items byte-identical; the key resolves (0 undefined citations). ⚠ **§6.1.**
- **T.5 + R.1 + R.2** (l.74, l.78, l.80) §2.1 opens with its thesis; the l.78 paragraph is inverted (meaning first, trained-`SO(2)` mechanics second); both half-lives are named `retention half-life`; one clause distinguishes the **near-flat stored coset** from the **stiff radial mode** and says one law governs both — built only from l.63, l.78 and Fig. 1's caption (l.86). ⛔ **No physics composed; the STOP clause never fired.** ⛔ l.78's own numbers were **not** moved (R.3 enumerates only l.80's).
- **R.3 / R.4 / R.5** numbers moved to the new table; **every mandatory rider stayed in the body** (§4).
- **R.3b** two intuition lines (of the permitted three); R.4's block got none — it already ends in plain words. ⛔ Zero numbers, zero capability/payoff/comparison/forward claim, zero programme-vision content.
- **P.1** (l.97) colleague's brake/refrigerator sentence adopted verbatim. **P.2** (l.107) one clause: *layout depends only on which items are stored, not on write order/history*. ⛔ l.109 untouched — it does not appear in the diff.
- **R.0** new appendix section `Values Quoted in the Main Text` inserted immediately after `\appendix` ⇒ `Table~\ref{tab:numbers}` is the **first table of the appendix** (prints as Table 1), 12 rows, `Result | Quantity | Value | Arm | Scope / rider`, with the **Arm legend** (`verification` = designed testbed, `evidence` = learned system) in both the lead paragraph and the caption — ⭐ this is the first time the paper defines a taxonomy it already uses in six figure captions.
- **X** 26 instances audited: **18 deleted, 8 retained with reasons** (§5). ⛔ `remarkable` is gone.

---

## 3. How I verified (commands + observed output)

**Application method.** Every edit applied by exact-string replacement with an assertion that the match count is **exactly 1** (`apply_edits.py`); a fuzzy or duplicated match aborts the pass. All 21 replacements printed `ok`.

**Build** (`/Library/TeX/texbin`, `pdflatex → bibtex → pdflatex → pdflatex`, scratch dir `/tmp/v5edit`, never the live folder):

| metric | pre-pass baseline | after |
|---|---|---|
| errors | 0 | **0** |
| undefined citations | 0 | **0** |
| undefined references | 0 | **0** |
| overfull boxes | 0 | **0** |
| total pages | 18 | **19** |

**Numeric two-way check** (`numcheck.py`, token-level multiset diff of main text lines 27–133):
- numbers that **left** main text: 37 tokens (`0.902`, `1.0020`, `1.116`, `1.7`, `0.9001`, `0.9032`, `4.9`, `1.0068`, `0.0219`, `1.26`, `107.77`×2, `13.28`, `86.97`, `4.78`, `2.94`, `0.12`, `25`, `200`, …);
- **ORPHAN LIST: `[]` — EMPTY** ✅;
- **table numbers with no ancestor in the pre-pass file: `[]` — EMPTY** ✅ (only non-ancestor numerals in the block are `p{0.165\linewidth}`-style column widths and `\tabcolsep`);
- numbers **added** to main text: exactly one token, **`2026`**, from the new `\cite{jawahar_chlu_2026}` key. **No numeric claim was added anywhere.**

**Main-text page count** — measured with `pagefrac.py` (full pages + the fraction of page 5's text block above `References`, from `pdftotext -bbox` word boxes, using page 4's body box as the reference block):

| | before | after | Δ |
|---|---|---|---|
| main text | **4.28 pp** | **4.32 pp** | **+0.04 pp** (≈ 26 pt ≈ 2.4 lines) |

⚠ The task's stated baseline is **4.30 pp**; my instrument reads the same pre-pass PDF as **4.28 pp** — a metric-definition difference, not a content one. The **Δ** is measured with one instrument on both ends and is sound. ⛔ **Nothing was cut to hit a number.** The **main text grew**: Group R removed ~37 numeric tokens, but P.1's explanatory sentence, P.2's clause, R.2's distinguishing clause, two R.3b lines, four `Table~\ref` pointers and the R.0a riders that must stay in the body together exceed them. **The paper is now +0.04 pp further over the 4-pp limit.**

**Grep hazard (memory: directory-level grep over `.claude/` returns false negatives).** Every negative sweep carries a positive control: `interpret` → 0 / `prescribed` → 1 · `after the fact` → 0 / `even after the entry` → 1 · hard-coded appendix letters → 0 / `App.~\ref` → 17 · line-`md5` identity checks for l.52 and l.109.

**Flag provenance.** ⛔ This pass measured nothing — **zero new numbers**, so there is no new result to carry a flag-provenance table. Every number in the new table inherits the flags of its ancestor line, and the table's caption carries them explicitly: `dim 4, hidden 64, ε=0.05, single-CPU`; every `T>0` cell `langevin_noise="fdt"` + Newtonian kinetic mode; per-row `seeds`, `T`, `Δ`, `γ`, `γ_φ`, and the arm. Ancestor line for every cell is printed in the build note §2a.

---

## 4. The R.0a test — no claim lost its rider

Full table in the build note §3. Summary: **8 claims audited, 0 blocking failures.** The designed-vs-emergent distinction survives at every result (l.80 *"emergent MLP checkpoints"*, *"both architecture families"*, table `Arm` column). The **probe-resolution rider is kept verbatim in the body** and repeated in bold in the table. The **`D̂_θ` estimator name is in the body sentence** beside the vault claim, with the scalar control and the *"first-passage reading is **not** the quoted vault"* rider, and `ℓ_θ/Δ=0.079` kept as a body number. l.109's deletion conditions/recency exclusion/encoder scope: untouched.

**One place needs the Advisor's eye (§3a(a) of the build note):** at l.93 the rider *was* a number (`γ∈[0.002,0.5]`). R.0a forbids splitting a number from its rider, so **both moved together** and the body now reads *"latches indefinitely across **every friction setting we evaluated**"* with the range one pointer away in the same parenthesis — a scope qualifier, not an unqualified *"any friction"*. One-line restore if you prefer the explicit range in the body (it would then be duplicated, never orphaned).

**Standalone-ness improves** — mechanisms named: P.1 gives both metaphors their physical content; P.2 makes the deletion claim operational for an ML reader; R.2's clause resolves a genuine ambiguity (two different half-lives quoted in adjacent paragraphs with no statement of which mode each belonged to); T.5+R.1 put the claim before the apparatus; the table gives every moved number one addressable home **with its arm and rider**, and defines `verification`/`evidence` for the first time. Counterweight recorded honestly: exact magnitudes now require the appendix — the trade the colleague asked for and the Head approved.

---

## 5. Group X ledger (26 = 18 deleted + 8 retained)

**Deleted (18):** l.35 `explicitly` · l.40 `distinct`, `intrinsically` · l.57 `precisely` · l.74 `fundamentally` · l.76 `distinct` · l.78 `distinct`, `precisely`, `singular` · l.80 `distinct` · l.93 `physically` · l.95 `successfully` · l.97 `explicitly`, **`remarkable`** · l.107 `explicitly` · l.119 `explicitly` · l.120 `intrinsically` · l.123 `explicitly`.
⭐ l.78's `precisely` is a substantive win: *"tracks the μ⁻² law **precisely** until εμ≈γ/2"* was ambiguous between the tracking's fidelity and the crossover's location — exactly the class of intensifier the sibling-paper referee caught flipping statements. The fidelity is stated quantitatively by the log-slopes in the next sentence.
⚠ Closest call, flagged not hidden: l.40 `distinct` (`operate under distinct retention/deletion mechanisms:`) — the colon-list that follows carries the heterogeneity, so I judged no meaning change.

**Retained (8), each with its load-bearing reason:** l.67 / l.118 / l.121 `strictly` (the three the Advisor expected to survive — real requirement, C-5 scale qualifier, scope narrowing) · l.52 / l.93 / l.107 `singular` (⛔ deletion is ungrammatical; the word means *a single*, and substitution is outside a deletion-only sweep — see §6.2) · l.109 and l.121 `intrinsically` (C-6 fine print: marks the recency-eviction breakage as **structural**, hence excluded rather than repairable; l.109 is additionally the protected passage).

---

## 6. Findings — defects OUTSIDE the approved set. ⛔ NONE was touched.

### 6.1 ⚠ DECISION NEEDED (one character) — the T.4 citation renders as running text
Executed verbatim, including `\cite`. Under `natbib`+`plainnat` **`\cite` is textual**, so the bullet prints:
> **Nomenclature borrowing from and building on CHLU Jawahar and Pierini [2026]:**
The intended reading is `CHLU [Jawahar and Pierini, 2026]` = **`\citep`**. ⛔ I did not make the change: the task supplied the literal LaTeX as the Head's wording, and silently switching the citation command is the kind of unenumerated "improvement" the scope rule forbids. **Ready patch: `\cite` → `\citep` on l.59 only.** This is the only thing in the pass that is visibly wrong on the page.

### 6.2 `singular` used to mean *"a single"* in three places (l.52, l.93, l.107)
Retained under the deletion-only rule (§5). Correct repair is a word substitution, outside this pass's licence — and at l.52/l.107 it sits beside `superposed`, which the DECLINED T.3 protects. **Recommend a separate three-token item.**

### 6.3 ⚠ Two pre-existing numeric mismatches, now adjacent in one table — NEEDS AN OWNER
The pass moved l.80's `0.902±0.003` and l.89's `0.9032±0.0027` into the same table. These are **the same quantity on the same instrument** (emergent argmin, one-step Jacobian, 3 seeds) at two precisions, and **`0.9032` does not round to `0.902`**. Same for the underdamped slope: body l.80 `+1.116±0.011` vs App. D's I-J row `+1.1182±0.0107`. **Both pre-date this pass** (present in the boot file) and ⛔ **I adjusted, rounded and reconciled nothing** — an unsourceable number is a missing-experiment note, not an improvisation. ⚠ But the table now puts them ~6 rows apart where a referee sees both at once. Rows are labelled by instrument and result-block so they are at least distinguishable. **Route to whoever owns the emergent-arm numbers:** either one reading is stale, or they are different fits and the body should say which.

### 6.4 The abstract invokes the vault factor without its estimator
l.35: *"demonstrating a `107.77±4.78×` retention factor on designed architectures"* — arm rider present, **`D̂_θ` estimator name absent**, while R.0a's rule is *"wherever it is invoked."* The abstract's number was not in R.5's move list, so l.35 was left alone apart from the approved `X` deletion. Not a blocking failure under deliverable 4 (no number moved out of the abstract), but it is the one remaining unestimated invocation. **Recommend a scoped one-clause item.**

### 6.5 Two typographic residues
(a) l.42 ends `App.~\ref{app:related}` with **no full stop** — untouched. (b) The pre-pass l.80 carried a stray doubled parenthesis `(Figure~\ref{fig:lambdacoset}))`; the second `)` closed nothing, and R.3's approved rewrite necessarily re-flowed that parenthetical, so the shipped line has a single paren. **Consequence of an approved hunk, not an independent edit** — recorded so the diff reader is not surprised.

### 6.6 Craft note for whoever re-opens this file
**Inside a `p{}` column, `\linewidth` is the column width**, so `\multirow{n}{0.145\linewidth}{…}` builds a parbox ~7× too narrow and floods the log with overfull boxes. The shipped table uses `\multirow{n}{\linewidth}{…}` + a table-local `\setlength{\tabcolsep}{4pt}` (widths then sum to 0.82 against 0.879 available). One label (`Cross-instrument check`) keeps only its `App.~\ref` pointer because the 2-row span could not hold five lines of label without an overfull `\vbox`.

---

## 7. Open questions / risks for the Hub & Advisor

1. **§6.1 — rule on `\cite` vs `\citep`.** Blocking for a clean ship; one character.
2. **Instruction conflict I resolved, please confirm.** T.5 (approved `yes`) names `l.72–74`; R.2's consistency note says *"⛔ Do not edit l.35 or l.74."* I read R.2's prohibition as scoped to **R.2's subject — the naming of the quantity** — since T.5 cannot be executed otherwise. **Mitigation: `retention half-life` at l.74 is preserved verbatim**, and l.35's only change is the `X` deletion of `explicitly` (l.35 is on X's own approved line list). If the Advisor meant l.74 frozen outright, that single hunk is the one to revert; nothing depends on it.
3. **Page count went the wrong way (+0.04 pp).** Measured, not assumed, and reported rather than fixed — pruning is not this pass's job. The Advisor now has a number to scope one with.
4. **Appendix letters all shift by one** (A→B … F→G) because R.0's section is first. Every in-file pointer is a `\ref` and resolves; but do not quote appendix letters from the old PDF in correspondence.
5. **Build copy refresh.** `~/Desktop/V5_PALM_Submission/paper.tex` was deliberately not touched; refresh it from `pj_sub.tex` at acceptance, and decide whether the stale live `pj_sub.pdf` is refreshed now or then.
6. **§6.3 needs an owner at this review** — per the protocol's reconciliation-list corollary, a finding buried without an owner is how the "2.6" retraction sat live for two waves.

**Artifacts banked** (`.claude/outputs/v5-colleague-edits/`): `pj_sub.pdf` (the verified 19-pp build), `pj_sub.log`, `pj_sub_BEFORE.tex` (the boot bytes, for a byte-exact revert), and the four scripts (`apply_edits.py`, `table_fix.py`, `numcheck.py`, `pagefrac.py`) so every check in this report is re-runnable.

## Proposed handover updates (for the Hub)
- **V5 re-opened and re-closed on 2026-08-27** for the colleague's edit set: `pj_sub.tex` now at **`ca56fef3b86a2d5d17314f84e130df3a`** (was `c63a57fc…`); 19 pp; main text **4.32 pp** (was 4.28 pp on this instrument, 4.30 pp as previously recorded); build clean (0 errors / 0 undefined / 0 overfull).
- **New standing asset:** the paper now defines the **`verification` (designed testbed) vs `evidence` (learned system)** taxonomy in App. A's table legend — the first time C-2's labels are defined inside a shipped draft. Reusable wording for the other shorts.
- **Three items owed** and not executable under this pass's scope: **`\cite`→`\citep`** (l.59), the **`singular` → `single`** three-token fix (l.52/93/107), the **abstract's missing estimator name** (l.35). Plus **one reconciliation item with no owner yet**: the `0.902` vs `0.9032` and `+1.116` vs `+1.1182` mismatches, both pre-existing.
- **Craft lesson worth carrying to the other shorts' build notes:** `\linewidth` inside a `p{}` column is the *column* width (§6.6).
