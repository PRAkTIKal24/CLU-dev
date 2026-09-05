# V5 — edit list from the colleague's review · **DECISION SHEET**

**Prepared by the Shorts Advisor, 2026-08-27.** Edit this file directly — fill in each `RULING:` line.
**Object:** `.claude/NIPSsubmission/v5-palm/pj_sub.tex` · **md5 at preparation `c63a57fc910663dfa1e644b9b349ce6f`** · 18 pp, main text 4.30 pp.
**Source:** `~/Downloads/F_v5_PaLM_Subm3 (1).pdf` — 13 annotations by **RiccardoMaggioni**, 2026-08-27 16:43.
⭐ **Version reviewed = the CLOSED V5** (verified by content probe: the "We also note" fix present, negatives appendix absent, 18 pp) ⇒ **nothing here is already-addressed.**

**How to rule:** put `YES` / `NO` / `AMEND: <what>` on each `RULING:` line. Anything left blank is treated as **NOT approved** and will not be touched.

> ⛔ **A spoke will be scoped against exactly what is approved here.** Anything not enumerated in this file is forbidden to the executing pass, however beneficial it looks — the standing rule earned when an over-scoped restoration cost this program a full pass.

---

## Group T — terminology & framing *(his p.1–2 comments)*

### T.1 · line 42 — "after the fact" reads informal
> *"even after the entry is nominally deleted. // after the fact seems to informal"*

**Now:** `…which leaves measurable residue within the network architecture **after the fact** \citep{chakraborttii_ghost_2026,wang_memleak_2026}.`
**Proposed:** `…which leaves measurable residue within the network architecture **even after the entry is nominally deleted** \citep{…}.` *(his wording)*
**Grade:** ✅ free editorial
**RULING:** yes

---

### T.2 · line 44 — the framing sentence ⚠ *highest-value item in the review*
> *"current all the NN have this dynamic property no? the main selling point is the controllability and interpretability of the method."*

**Now:** `In this work, we propose and analyze a memory framework where forgetting is an **intrinsic** dynamical property of the store itself, rather than an external bookkeeping rule.`
**Proposed (his structure, Advisor-guarded):** `In this work, we propose and analyze a memory framework where forgetting is a prescribed dynamical property of the store, governed by parameters that can be set, targeted, and analyzed in closed form, rather than an emergent side effect or an external bookkeeping rule.`
**Grade:** ⚠ **claim-shaped.** His objection is correct — every recurrent net forgets; what is ours is that it is *specified*.
⛔ **Advisor guard:** adopt the structure, ⛔ **do NOT import "interpretability"** — it is not established for this store, and capabilities stay in the "and also" position under the genuine-win bar. *prescribed · set · targeted · analyzed in closed form* are all measured.
**RULING:** yes

---

### T.3 · line 52 — "atoms" collides with an ML sense
> *"this seems the physics term. I know that there is also a ML definition of it, it's the same? just to avoid confusion in the reader"*

**Now:** `Memory items can optionally be written as \emph{atoms}: localized energy wells superposed into a singular energy function that is subsequently read by dynamical relaxation.`
**Proposed:** one disambiguating clause at first use, e.g. `…written as \emph{atoms} (energy wells in the store's potential, not the ML sense of atomic units): localized energy wells superposed…`
**Grade:** ✅ free editorial
**RULING:** No, the point was about the superposed term and that is correctly used in both senses.

---

### T.4 · lines 59–63 — Nomenclature appears without provenance
> *"is this the best place to have it? in addition I would add a explicit reminder to the fact that they are defined in the clue paper. Because they seems out of nowhere"*

**Now:** `\item \textbf{Nomenclature:}` + four sub-items (inertial mass · spectral mass · stored direction · integration step), no source given.
**Proposed:** one clause on the Nomenclature bullet attributing the definitions to the CLU paper.
⛔ **Advisor note:** his *first* question (is this the right *place*) implies relocating the block — a structural move with page consequences. **Recommend the clause only, not relocation.** Say `AMEND: relocate too` if you want the bigger move.
**Grade:** ✅ free editorial
**RULING:** Amend: the Nomenclature bullet item reads "Nomenclature borrowing from and building on CHLU~\cite{.." citing the CHLU paper.

---

### T.5 · lines 72–74 — §2.1 does not open with its own thesis
> *"Here the point is: Friction is not simply a stronger/weaker retention knob. There is an optimal friction, determined by the mode's spectral mass, right?"*

**Now:** §2.1 opens at l.74: `To understand the macro-dynamics of memory decay, we first analyze damping as a retention dial.`
**Proposed:** open with the claim — friction is not a monotone knob; there is an **optimal** friction set by the mode's spectral mass — then the mechanics.
**Grade:** ✅ **he is right; this is the section's actual contribution.** Executes together with **R.1**.
**RULING:** yes

---

## Group R — results restructure *(his five p.3 comments)*

His five comments reduce to one request: **state the result, move the numbers out.**

**Measured load:** ≈**39 numbers in 7 body paragraphs** of a 4.30-pp paper (figure captions excluded — those stay with their figures): l.78 (4) · l.80 (10) · l.89 (5) · l.93 (4) · l.95 (2) · l.97 (12) · l.109 (2).

### R.0 · NEW — one appendix table absorbing all of it
**Schema:** `Result | Quantity | Value | Arm | Scope / rider` — placed as the first table of the appendix.
⭐ **The `Arm` column carries a legend defining *verification* (designed testbed) vs *evidence* (learned system)** — which also closes the separate finding that this taxonomy is used in main-text captions and defined nowhere.
⛔ **Zero new numbers.** Every cell has an ancestor already in the paper.
**RULING:** yes

---

### ⛔ R.0a — THE BINDING CONSTRAINT ON GROUP R *(not a rulable item; it governs whatever is approved)*
**A number and its mandatory rider move together, or not at all.**
⛔ The inverse — a claim staying in main text while its qualification moves to the appendix — is **never** an option. This program has measured that exact failure once: *"a materially less-qualified paper without a single number having been edited."*
⚠ V5's main text was **already judged not to stand alone** by the final audit. **Acceptance test for this pass: standalone-ness must improve, not degrade.**
**Must remain in main text beside their claims:** the designed-vs-emergent distinction · the probe-resolution rider wherever the μ² span is invoked · the estimator name wherever the vault factor is invoked · the deletion conditions + recency exclusion + encoder scope (already one passage at l.109).

---

### R.1 · line 78 — result before experiment
> *"I would first explain the result and then the details of the experiment. lot to keep in mind before reaching the important part"*

**Proposed:** invert the paragraph — what the V-curve *means* first, then the trained-SO(2) testbed mechanics. Merges with **T.5**.
**RULING:** yes

---

### R.2 · lines 78, 80 — naming
> *"retention half life, seems more clear. Similar comment for the previous ones"*

**Proposed:** use **"retention half-life"** at first use in each subsection instead of the bare symbol.
**RULING:** yes, but note that this was for the "massive radial mode's half life" terminology. if unsure hold and ask me.

---

### R.3 · line 80 — emergent numbers → table
> *"I'll leave all the numbers to the appendix and here I would pass the main concept."*

**Moves:** argmin `0.902±0.003×γ_crit` · log-slopes `−1.0020±0.0003` / `+1.116±0.011` · span `μ² ∈ [1.7e−12, 7e−2]`.
⛔ **Main text keeps the qualitative law AND the probe-resolution rider** (the low endpoint is instrument resolution, not a measured spectral mass).
**RULING:** yes, lets add very simple one line intuitions on the results and what it means for the overall vision of the paper.

---

### R.4 · line 89 — cross-instrument numbers → table
> *"make it clear that is in the appendix"*

**Moves:** rollout argmin `0.9001±0.0052` vs Jacobian `0.9032±0.0027`.
**Main text keeps:** "the shape reproduces on a second instrument (Table N)".
**RULING:** yes

---

### R.5 · lines 93, 95, 97 — vault & latch numbers → table
> *"same here, I think we can leave the exact numbers for the section and pass the main concepts."*

**Moves:** latch drift `≤4.9e−12 rad / 200k steps, γ∈[0.002,0.5]` · `D̂/D_pred = 1.0068±0.0219` over 25 cells · `T_local = 1.26e−4` vs `1e−3` · vault `107.77±4.78×` with control `13.28±0.12×` and raw first-passage `86.97±2.94×`.
⛔ **The estimator name travels with the vault factor wherever it is invoked.**
**RULING:** yes

---

## Group P — physics exposition

### P.1 · line 97 — brake / refrigerator unexplained
> *"I would explain, I think non obvious for non physics people"*

**Now:** `A localized spatial hole within this field functions concurrently as a brake and a refrigerator ($T_{\rm local}=1.26\times10^{-4}$ versus $10^{-3}$ externally).`
**Proposed (his sentence):** `A localized spatial hole within this field acts simultaneously as a **brake, increasing dissipation**, and a **refrigerator, reducing the local effective temperature**.` *(number to the table under R.5)*
**Grade:** ✅ clearer than ours and supplies the *why* for both metaphors
**RULING:** yes

---

### P.2 · line 107 — "pure function of its live set" unexplained
> *"same"*

**Now:** `…physical state can be explicitly reduced to a pure function of its live set alone.`
**Proposed:** one clause on what that buys — the layout depends only on which items are currently stored, not on the order or history of writes.
**Grade:** ✅ free editorial
**RULING:** yes

---

⚠ **P.1 and P.2 ADD words while Group R removes them.** Net expected negative; ⛔ to be **measured**, not assumed.

---

## Group X — intensifier sweep *(Advisor find; the mechanical half of his "show off of terminology" verdict)*

**26 occurrences across 19 main-text body lines** (figure captions excluded):
`l.35 · 40 · 52 · 57 · 67 · 74 · 76 · 78 · 80 · 93 · 95 · 97 · 107 · 109 · 118 · 119 · 120 · 121 · 123`

| word | count |
|---|---|
| explicitly | 5 |
| distinct | 4 |
| intrinsically | 4 |
| singular | 4 |
| strictly | 3 |
| precisely | 2 |
| fundamentally · physically · successfully · **remarkable** | 1 each |

⛔ **l.97 currently reads "generates a *remarkable* retention vault factor."** `remarkable` is named in the standing forbidden class, and the rule is not cosmetic — a blind referee caught intensifiers **flipping two statements false** in a sibling paper.

**Proposed rule:** delete only where deletion changes no meaning. ⛔ **Where the adverb is load-bearing, report it — never delete it.**

⚠ **Three the Advisor expects to survive**, flagged so they are not stripped mechanically:
- **l.121** — "applies **strictly** at the isolated store-level" *(narrows scope)*
- **l.67** — "$T>0$ **strictly** requires FDT-consistent noise" *(the requirement is real)*
- **l.118** — "**strictly** constrained to a dimension of 4" *(C-5 scale qualifier)*

**RULING:** yes
**Lines to exclude from the sweep (add any):**

---

## Execution constraints *(bind whatever is approved; not rulable)*

1. ⛔ **Anything not enumerated above is forbidden**, however beneficial it looks.
2. ⛔ **The Head's prose is the Head's** — a defect noticed outside the approved set goes in a findings list, never into the file.
3. **Diff contract:** every hunk labelled against its item ID; ⛔ unattributable hunks = **ZERO**.
4. **Numeric two-way check:** every number leaving main text must appear in the new table — ⛔ **orphan list must be empty**.
5. **File-chain rule:** `pj_sub.tex` is canonical; `~/Desktop/V5_PALM_Submission/paper.tex` is a build copy refreshed before the pass, never authored. ⛔ No spoke writes while the Head is editing.
6. **Measure, don't assume:** main-text page count reported **before and after** (currently **4.30 pp** against a 4-pp limit).
7. Snapshot the accepted state off the live path at acceptance — **the bytes, not just the hash.**

⚠ **V5 is currently CLOSED; this pass re-opens it.** The colleague's structural read is independently corroborated by both referees, so the upside is real — but every re-open in this program has introduced new regressions alongside its fixes.
