# `v5-colleague-edits` — the Head-approved edit set from the colleague's review

**Agent:** `paper-writer`
**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-27 (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 95).**
**Object:** `.claude/NIPSsubmission/v5-palm/pj_sub.tex` — **md5 `c63a57fc910663dfa1e644b9b349ce6f`**, 18 pp, main text 4.30 pp.
**Report:** `.claude/outputs/v5-colleague-edits.md` · **Build note:** `BUILD-NOTE-COLLEAGUE.md` in the paper's folder, **deliverable #1**.
**Decision sheet of record:** `.claude/outputs/v5-colleague-edit-list.md` — the Head ruled every item there; this file is its execution form.

---

## ⛔⛔ THE SCOPE RULE — read before anything else

**Thirteen items are approved. One is DECLINED. Nothing else exists.**

⛔ **Anything not enumerated in §2–§5 is forbidden, however beneficial it looks.** This program lost a full pass to a writer executing faithfully against a task file that had quietly expanded "add the missing pieces" into "restore everything the reports list" — the work was reverted and the fault was the task file's, not the writer's. **If you notice a defect outside the approved set, it goes in the findings list. It never goes in the file.**

⛔ **V5 is a CLOSED paper being re-opened for exactly this set.** Every re-open in this program's history has introduced new regressions alongside its fixes. Your footprint discipline is the whole safeguard.

### ⛔ DECLINED — do not do this
**T.3 (l.52, "atoms")** — the Head ruled **NO**: *"the point was about the superposed term and that is correctly used in both senses."* ⛔ **Do not add a disambiguating clause to `atoms`, and do not touch `superposed`.**

---

## 1. Boot

1. Read this file fully, then `.claude/outputs/v5-colleague-edit-list.md` for the Head's rulings verbatim.
2. **Verify the object on disk:** `md5 -q pj_sub.tex` must be `c63a57fc910663dfa1e644b9b349ce6f`. ⛔ **If it differs, the Head has edited since scoping — STOP and report; do not guess which lines moved.** Line numbers in this file are pinned to that hash.
3. Record `pj_sub.tex`'s mtime at boot and again at the end. ⛔ **If it moved mid-pass, stop and report** — the Head may be editing.
4. Read `submission.tex` in the same folder — the ancestor base, and the authority for any wording that must be verbatim.

---

## 2. Group T — terminology & framing

**T.1 · l.42** — replace `after the fact` with `even after the entry is nominally deleted`. Surrounding words and the `\citep` unchanged.

**T.2 · l.44** — replace the sentence with:
> `In this work, we propose and analyze a memory framework where forgetting is a prescribed dynamical property of the store, governed by parameters that can be set, targeted, and analyzed in closed form, rather than an emergent side effect or an external bookkeeping rule.`

⛔ **Guard, Head-ratified:** the word **"interpretability" must NOT enter**, nor any synonym asserting the store is interpretable. It is not established for this store, and capabilities stay in the "and also" position under the genuine-win bar. *prescribed · set · targeted · analyzed in closed form* are each measured; nothing beyond them is licensed.

**T.4 · l.59** — the Nomenclature bullet gains its provenance. Head's wording:
> `\item \textbf{Nomenclature borrowing from and building on CHLU~\cite{jawahar_chlu_2026}:}`

(⚠ `jawahar_chlu_2026` is the only CHLU key in `refs.bib` — verify it resolves.) ⛔ **The block is NOT relocated** — the Head ruled the clause only.

**T.5 · l.72–74** — §2.1 opens with its own thesis instead of its method: **friction is not a monotone retention knob; there is an optimal friction, set by the mode's spectral mass** — then the existing mechanics. ⭐ **Execute together with R.1**; they are one rewrite of the same opening.

---

## 3. Group R — the results restructure

### R.0 — build ONE new table, first table of the appendix

**Schema:** `Result | Quantity | Value | Arm | Scope / rider`

It absorbs every number moved by R.3, R.4 and R.5. ⛔ **ZERO new numbers** — every cell must have an ancestor already in `pj_sub.tex`; print the ancestor line for each row in the build note.

⭐ **The `Arm` column carries a legend defining the taxonomy**: *verification* = designed testbed · *evidence* = learned system. This is deliberate — the terms are currently used in main-text figure captions and defined nowhere in the paper, and the legend closes that.

### ⛔ R.0a — THE BINDING CONSTRAINT ON ALL OF GROUP R
**A number and its mandatory rider move together, or not at all.**
⛔ **The inverse is never permitted:** a claim may not stay in main text while its qualification moves to the appendix. This program has measured that exact failure once — *"a materially less-qualified paper without a single number having been edited"* — and no diff can see it.
⚠ **V5's main text was already judged NOT to stand alone by its final audit. The acceptance test for this pass is that standalone-ness IMPROVES.**

**These stay in main text, beside the claims they qualify:**
- the **designed-vs-emergent** distinction wherever a result is stated;
- the **probe-resolution rider** wherever the μ² span is invoked (the low endpoint is instrument resolution, ⛔ *not* a measured spectral mass);
- the **estimator name** wherever the vault factor is invoked;
- the deletion conditions + recency exclusion + encoder scope (already one passage at l.109 — ⛔ do not touch it).

### R.1 · l.78 — result before method
Invert the paragraph: what the V-curve **means** first, then the trained-SO(2) testbed mechanics. Merges with T.5.

### R.2 · l.78 and l.80 — name the quantity, keep the mode, and distinguish the two
**Head ruling:** *"keep the mode but specify that this is different from the other retention half life, and how, briefly."*

- **l.78** → `the massive radial mode's retention half-life`
- **l.80** → keeps its `coset-tracked` identifier, likewise named `retention half-life`
- ⭐ **Add one brief clause distinguishing them**, drawn from the paper's own material — ⛔ **do not invent the physics**:
  - the **coset** is the **near-flat stored direction** — the Nomenclature already says *"a written value occupies a near-flat coset direction"* (l.63) — with a very small spectral mass (`μ²_soft = 2.0–5.4×10⁻²` on the emergent arm, per Fig. 1's caption);
  - the **massive radial mode** is a **stiff** direction, `μ²_rad = 0.670–1.348` (Fig. 1's caption), and is not where a value is stored;
  - ⭐ the point of measuring both is that **the same law governs them**, which is what makes the μ² span meaningful.
- ⚠ **Consistency note:** l.35 (abstract) and l.74 already say *"retention half-life"*; this makes l.78/l.80 agree with them. ⛔ Do not edit l.35 or l.74.
- ⛔ **If the brief clause cannot be written from the material above without asserting something the paper does not state, STOP and report** rather than composing physics.

### R.3 · l.80 — emergent numbers → table
**Move:** argmin `0.902±0.003×γ_crit` · log-slopes `−1.0020±0.0003` and `+1.116±0.011` · span `μ² ∈ [1.7×10⁻¹², 7×10⁻²]`.
**Main text keeps:** the qualitative law, the designed-vs-emergent distinction, and ⛔ **the probe-resolution rider**.

### R.4 · l.89 — cross-instrument numbers → table
**Move:** rollout argmin `0.9001±0.0052` vs Jacobian `0.9032±0.0027`.
**Main text keeps:** that the shape reproduces on a second instrument, with a pointer to the table.

### R.5 · l.93, l.95, l.97 — latch & vault numbers → table
**Move:** latch drift `≤4.9×10⁻¹² rad / 200k steps, γ∈[0.002,0.5]` · `D̂/D_pred = 1.0068±0.0219` over 25 (γ,T) cells · `T_local = 1.26×10⁻⁴` vs `10⁻³` outside · vault `107.77±4.78×` **with** control `13.28±0.12×` **and** raw first-passage `86.97±2.94×`.
⛔ **The estimator name travels with the vault factor wherever it is invoked** — the quoted number is the `D̂_θ` estimator's, and the raw first-passage reading is explicitly not the quoted vault.

### R.3b — the descriptive intuition lines (Head addition)
**Head ruling: DESCRIPTIVE ONLY.** Add **at most three** plain-language one-liners — **one per result block that loses its numbers (R.3, R.4, R.5)** — saying what the result means in ordinary words. Fewer is fine if a block does not need one.

⛔ **Hard boundary:** these lines say **what the result is**, never **what it implies for the programme, the CLU vision, or a memory system's capabilities.** The Head has ruled that the vision belongs to the ICLR long and not to this short. ⛔ No line may add a capability, a payoff, a comparison, or a forward claim. ⛔ Zero new numbers.
*Register example — this shape, not this content: "friction has an optimum; past it, more friction costs retention rather than buying it."*

---

## 4. Group P — physics exposition

**P.1 · l.97** — replace with the colleague's clearer sentence (the number moves to the table under R.5):
> `A localized spatial hole within this field acts simultaneously as a brake, increasing dissipation, and a refrigerator, reducing the local effective temperature.`

**P.2 · l.107** — after `…a pure function of its live set alone`, add one clause on what that buys: **the layout depends only on which items are currently stored, not on the order or history of writes.** ⛔ One clause; ⛔ do not touch the deletion-conditions passage at l.109.

⚠ **P.1 and P.2 ADD words while Group R removes them.** Net effect is expected negative but ⛔ **must be measured, not assumed** (§6).

---

## 5. Group X — the intensifier sweep

**26 occurrences across 19 main-text body lines** (⛔ figure captions are out of scope): `l.35 · 40 · 52 · 57 · 67 · 74 · 76 · 78 · 80 · 93 · 95 · 97 · 107 · 109 · 118 · 119 · 120 · 121 · 123`
Words: `explicitly`×5 · `distinct`×4 · `intrinsically`×4 · `singular`×4 · `strictly`×3 · `precisely`×2 · `fundamentally` · `physically` · `successfully` · **`remarkable`**×1.

⛔ **l.97 reads *"generates a remarkable retention vault factor"*** — `remarkable` is named in the standing forbidden class. The rule is not cosmetic: a blind referee caught intensifiers **flipping two statements false** in a sibling paper.

**The rule: delete ONLY where deletion changes no meaning. ⛔ Where the adverb is load-bearing, REPORT it — never delete it.** The Head listed no exclusions, so judgment is yours under this rule, and every retained instance must appear in the build note with its reason.

⚠ **Three the Advisor expects to survive** — flagged so they are not stripped mechanically:
- **l.121** `applies **strictly** at the isolated store-level` — narrows scope;
- **l.67** `$T>0$ **strictly** requires FDT-consistent noise` — the requirement is real;
- **l.118** `**strictly** constrained to a dimension of 4` — a C-5 scale qualifier.

⛔ **This sweep touches adverbs and adjectives only.** It may not re-flow a sentence, re-order a clause, or change any noun, verb, number, label or citation.

---

## 6. Deliverables and acceptance

1. **`BUILD-NOTE-COLLEAGUE.md`** — deliverable #1, written before the PDF ships.
2. **Diff contract:** every changed hunk labelled with its item ID (`T.1`, `R.3`, `X`, …). ⛔ **Hunks attributable to no approved item = ZERO.** For T.1/T.2/T.4/P.1/P.2, print before→after and confirm the surrounding words are byte-identical.
3. ⛔ **Numeric two-way check:** every number that leaves main text must appear in the new table, and every table cell must have an ancestor line in the pre-pass file. **Print both lists. The orphan list must be EMPTY.**
4. **The R.0a test, stated explicitly:** list every main-text claim whose number moved, and name the qualification that remains beside it. ⛔ Any claim left in main text without its rider is a blocking failure — stop and report rather than shipping it.
5. **Group X ledger:** every deletion, and every retained instance with its load-bearing reason.
6. **Build:** `pdflatex → bibtex → pdflatex → pdflatex`; report `0 errors · 0 undefined citations · 0 undefined references · 0 overfull boxes`.
7. **Measure:** main-text page count **before and after** (before = **4.30 pp** against a 4-pp limit). ⛔ Report it; ⛔ **do not cut anything to hit a number** — the page limit is not this pass's job.
8. Final `md5` of `pj_sub.tex`, and confirmation that `figs/`, `refs.bib`, `submission.tex` and every other file in the folder are byte-untouched.

## 7. Boundaries

- ⛔ **`pj_sub.tex` is the ONLY file you may write.** `.claude/papers/**`, `submission.tex`, `figs/`, `refs.bib` and `~/Desktop/V5_PALM_Submission/**` are byte-untouched. The Advisor refreshes the build copy after acceptance.
- **Build in a scratch copy** (`/tmp`), never in the live folder while iterating; apply only the verified result.
- `pdflatex`/`bibtex` are **not on `PATH`**: use `/Library/TeX/texbin/`.
- ⛔ **The Head's prose is the Head's.** Outside the approved edits you change nothing — not a typo, not a comma.
- ⛔ **Zero new numbers, zero new claims** anywhere in this pass, including the table and the intuition lines.
- ⚠ **Grep hazard:** directory-level grep over `.claude/` silently returns nothing (gitignored). Sweep per-file, and **positive-control every negative** before reporting "zero occurrences".

## DIAL DECLARATION
**Dials touched: NONE.** No experiment, no config, no registry, no charter. This pass edits one `.tex` file within an enumerated set and writes one build note and one report.
