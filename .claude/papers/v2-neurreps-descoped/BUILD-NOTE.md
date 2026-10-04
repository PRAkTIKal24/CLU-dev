# V2 — the de-scoped extended abstract (THIRD, distinct V2 artifact) — build note

**A THIRD, SEPARATE V2 ARTIFACT.** Source = `papers/neurreps-variants/v2/submission.tex` (the first reframe).
Three V2 artifacts now coexist by the Head's design and this one is independent of both others:

| folder | what it is |
|---|---|
| `papers/v2-short/**` | the live r9 build — **byte-untouched, verified** (§9) |
| `papers/neurreps-variants/v2/**` | the first reframe, 13 pp, five contributions — ⚠ **NOT byte-untouched; see §9, and the cause was not this pass alone** |
| `papers/v2-neurreps-descoped/**` | **this one** — one contribution, current audience, corrected novelty scope |

Same results, same numbers, zero retractions. What changed is **scope** (one contribution, not five),
**audience** (§2 written to the current call, not a four-year-old census) and **novelty framing**
(the destroy-and-restore pattern is prior art here and is printed as such).

---

## 1. Files

| file | what it is |
|---|---|
| `submission.tex` | the de-scoped source |
| `submission.pdf` | **14 pp total**: main **6.19 pp** · references **1.66 pp** (47 entries) · appendices **6.16 pp** |
| `supplementary-theory-note.tex` / `.pdf` | copied unchanged from the reframe (anonymized theory note, 12 pp) |
| `figs/` | the three figures, copied from the reframe **before** the concurrent re-render (§10 — hashes recorded) |
| `neurips_2025_ml4ps.sty` | the template actually used (§2) |

Build: `pdflatex` ×3 (TeX Live 2026, `/Library/TeX/texbin`). **0 errors, 0 undefined references, 0 overfull boxes.**

## 2. Template

Identical to both other builds and for the same reason: neither the target venue's template nor the
NeurIPS 2026 style file is on this machine, so the closest genuine NeurIPS-family style file present
(`neurips_2025_ml4ps.sty`, textwidth 5.5 in × textheight 9 in, 10 pt, submission mode with line numbers and
the `Anonymous Author(s)` block) is used, with the notice box suppressed so the artifact is venue-neutral.
⚠ **The page count must be re-measured in the real venue template before submission.**

---

## 3. ⭐ THE DE-SCOPE, EXECUTED

**Main text carries ONE contribution: the transverse-curvature price list of a trained recurrent memory**,
in three parts — what a given curvature buys (`n₁/₂ ∝ μ⁻²`), where the law stops (crossover + floor), and
whether it survives the training-time correction. Its **evidence** is the published-law head-to-head (§4.2);
its **honest negative** is the boundary (§4.3). Abstract and the contributions paragraph claim nothing else.

### 3.1 Structural change against the source reframe

| | source reframe | this build |
|---|---|---|
| contributions claimed | 5 (budget · head-to-head · GMOR · designed-vs-emergent + price · taxonomy) | **1** (the price list) |
| the price list's own numbers | Appendix C only | **promoted to main text (§4.1)** — a contribution cannot have its evidence only in an appendix |
| GMOR proper | main-text contribution (ii) + App C | **appendix only** (App F.3–F.6 + Fig 3), marked demoted in its own opening line |
| realization taxonomy | §3 "Axis 1 / Axis 2", contribution (3) | **canonical-only** (prose-only demotion); a 183-word definitional gloss is retained — see 3.3 |
| price of the prior | §4.2 main text | **appendix only** (App B), marked demoted in its own opening line |
| §2 | written to a 2022 census | rebuilt for the current call, +2 verified citations |

### 3.2 Demotion list (⛔ demoted, never retracted — nothing left the record)

| demoted item | new home | has plot/table? |
|---|---|---|
| GMOR proper (linear ambient spurion, Σ measured three ways, the LEC ratio) | **App F.3–F.6 + Fig 3 + the 10-row δ table** | yes (both) |
| The price of the prior (param-matched Pareto table, loan curve, recovery ladder) | **App B + two tables** | yes (both) |
| The symmetry-realization taxonomy as a labelled contribution | **canonical-only** — it is prose with no plot and no results table | no |

### 3.3 ⚠ The one judgement call inside the taxonomy demotion, stated so the Head can overrule it

The taxonomy is demoted **as a contribution and as a section**: the `Axis 1 / Axis 2` framing, the three
cell headings (`Wigner–Weyl` / `Nambu–Goldstone` / `Pseudo-Goldstone` as a labelled partition) and
contributions item (3) are **absent from this submission** and live on in the two canonical artifacts.
A **183-word definitional gloss** is retained in §3 because **three verbatim-protected passages point at it**:
fine print (c) says *"the self-broken MLP being an observed instance of **the cell**"*; App D's mass-tying rule
says *"a degenerate **pseudo-Goldstone** multiplet"*; and §4.4's erosion result predicts a degenerate `μ²` pair
that only Schur's lemma explains. Deleting the gloss would leave three protected sentences without a referent,
which C-6 forbids more strongly than the demotion requires. The gloss claims nothing and appears in no
contributions list. **Measured cost: 0.269 pp** (row C6 below) — the Head can delete it *if* fine print (c) and
App D's rule are re-anchored at the same time. ⛔ Deleting it alone is a C-6 violation.

### 3.4 Submission-absent / canonical-present list (prose only; **zero numbers**, see §8)

Present in `papers/v2-short/**` and `papers/neurreps-variants/v2/**`, absent from this submission:
1. `\paragraph{Axis 1: which symmetry the trained potential realizes.}` and its three cell definitions
   (`Wigner–Weyl: unbroken, invariant vacuum…` / `Nambu–Goldstone: spontaneously broken…` /
   `Pseudo-Goldstone: explicitly broken, tilted orbit, μ²=δΣ/F² small…`), and the `Axis 2 is the map:` sentence.
2. Contributions items (1)–(4) of the source, replaced by a single-contribution paragraph.
3. Abstract items (i)–(iii) of the source, replaced by a single-contribution abstract.
4. The section titles `Setup, and the two axes`, `Designed versus emergent protection, and the price of the
   physics`, `Learned baselines, and the recipe that keeps the orbit intact`.
5. The clause *"a symmetry-restoration transition in the sense of §2, the potential landing in the Wigner–Weyl
   cell"* — reworded to *"the minimiser becomes invariant, with the degenerate μ² pair predicted in §3"*
   (same content, no taxonomy dependency).
6. The §4 preamble sentence *"…which establishes that 2×2 block, to machine precision, as the trained map's own
   angular block"* — the identity it asserts is now stated with its number in §4.1.
7. Four sentences of §2 replaced by the longer current-audience treatment (§4 below).
8. The word *"budget"* as the object's name; the object is called **the price list** throughout,
   per the ruling's own wording. One instance survived deliberately nowhere — the CM-4 evidence clause now
   reads *"the price-list law to +12–15%"*.

⛔ **No number, no finding, no negative and no qualifier is on this list.** The two-way numeric check (§8) is
exact in the "absent" direction: **0 numeric tokens in the source are missing here.**

---

## 4. The audience corrections, executed

1. **§2 is written to the current call, not to the 2022 census.** Its opening sentence names the two topic
   areas the paper actually sits in (representation *dynamics*; symmetry + dynamical systems + learning)
   without naming a venue — the venue-neutrality sweep (§7) requires that. Structure: the object and its
   established properties → the five facts that are **not ours** → the audience's own nearest work →
   the comparators and the drift disclaimer → the retirements.
2. **The novelty retraction is printed in the scout's scoped form**, twice over:
   - §2: *"A homeostatic mechanism that restores a destroyed flat direction is 23 years old (Renart, Song &
     Wang 2003), so a corrective term is not our idea, **and nothing here is a first report of the
     destroy-and-restore pattern**. A local learning rule can **produce** accurate ring-attractor tuning
     (Vafidis et al. 2022), so nothing we measure implies that learning cannot build a flat direction."*
   - §4.4: *"**That a corrective term can keep a flat direction alive is not new** (Renart, Song & Wang 2003).
     What we add is the third part of the price list…"*
   - §4.3 carries the N46 rider **verbatim**: *"This is a measurement on our architecture class and training
     recipe, not a general statement that learning cannot produce a tuned flat direction — a local learning
     rule that does produce one is published (Vafidis et al. 2022)."*
3. **Citations, all scout-verified.** 47 entries; **all 45 of the reframe's entries retained and every one
   still cited from the surviving body** (checked mechanically, §8). **Two added**, both from the scout's
   v228 record: **Vastola (2024)**, optimal packing of attractor states — the audience's own capacity
   question; **Dönmez (2024)**, memory modification through symmetry and geometry.
   ⚠ **Dönmez 2024 prints without a page range** — the scout retrieved title/author/year/volume but not
   pages (its §1.3b note). Flagged as an open item rather than invented.
   **Xu et al. (2023) is given the position the scout asked for**: a full sentence naming it as the closest
   published instance of this paper's object, followed by what we add ("a price list for time").
   Scout traps honoured: Seung sole author · Jude first author · Burak & Fiete's **Eq. 2 not reproduced** ·
   Dinc et al. cited as a preprint with no PRX volume · PMLR page ranges reproduced as PMLR gives them
   (including the Akhtiamov/Aslan p. 181 overlap) · PMLR items cited with the **2023** year, one convention.
   ⚠ **Scout F8 (the "conformal" collision) handled by avoidance**, as in the reframe: the body never uses
   "conformal" for Xu et al.; the collision survives only in their reference-list title, which cannot be altered.
4. **Negatives are foregrounded**, per the track's stated purpose:
   - the **abstract's THEREFORE clause** names four: the designed-vs-emergent gap *with its scope*, the
     designed-only register, the retired compute claim, the four retired positioning claims;
   - the **contributions paragraph** ends on *"Its honest negative is the boundary of §4.3"*;
   - §4.3 is titled **"The honest negative: where the price list does not extend"**.
5. **CM-21's four retirements** are stated compressed in §2 (one sentence each) with the approved replacement
   wording verbatim, and elaborated with their numbers in App E. None bounds a demoted contribution, so none
   moved. ⛔ None is re-asserted anywhere (sweep §7).
6. **Bridge vocabulary** used only where the scout marked it exact: *marginally stable tangent / stable normal*
   (quoted), *fine-tuning problem* (scoped in-sentence), *fast normal flow / slow tangent flow*, *exact versus
   relaxed equivariance*. Kept ours where there is no equivalent: **coset coordinate/register**, **exceptional
   point** (defined on first use), **half-life `n₁/₂`** (with the units conversion printed once).
   **"Continuous attractor" is never applied to our unit** — its 3 body uses are the literature's definition,
   other people's task-trained networks, and Xu et al.'s object. **The no-biological-claim sentence appears
   exactly once** (§2, immediately after the ring/torus sentence).

---

## 5. Page split — measured, and the 4-pp target MISSED

Measured from PDF word bounding boxes against the text block (top 72 pt, bottom 720 pt, page 792 pt) —
the same instrument both other builds used, so the numbers are directly comparable.

| block | r9 | reframe | **this build** |
|---|---|---|---|
| title + abstract | 0.71 | 0.70 | **0.85** |
| §1 Introduction | 0.49 | 0.63 | **0.63** |
| §2 Related work | 0.88 | 0.95 | **1.20** |
| §3 Setup | 0.79 | 1.24 | **0.95** |
| §4 Results | 1.92 | 1.63 | **2.00** |
| §5 Discussion | 0.94 | 0.55 | **0.55** |
| **MAIN TOTAL** | **5.72** | **5.69** | **6.19 pp** ⚠ against a 4 pp aim — **MISSED** |
| references | 1.01 (28) | 1.63 (45) | **1.66 pp (47)** |
| appendices | 4.26 | 5.68 | **6.16 pp** |
| **TOTAL** | **11** | **13** | **14 pp** |

Main text is **4,314 words** plus one figure.

### 5.1 ⛔ The finding the Head asked for: the de-scope did not buy the page, and here is the arithmetic

The de-scope removed two main-text blocks and forced one promotion:

| move | Δ main |
|---|---|
| taxonomy out of §3 (contribution → canonical-only, gloss retained) | −0.29 |
| price of the prior out of §4 (→ App B) | −0.25 |
| **price list promoted INTO §4.1** — the retained contribution's own evidence | **+0.75** |
| §2 rebuilt for the current audience (the mandated correction) | +0.25 |
| abstract rewritten to carry the negatives (the mandated correction) | +0.15 |
| net | **+0.50** |

**Add.40's projection that the de-scope reaches 4 pp assumed the price list stays in an appendix.** It cannot:
Head policy is *main text = main results only*, and the paper's single contribution is the price list. With it
in main text, **4 pp is still unreachable without C-6 trades** — measured below, taking every free block lands
at **5.16 pp**, and additionally removing the headline figure lands at **4.81 pp**. ⛔ Per the standing
boundary the writer did not trade any fine print for space; this is reported, not decided.

---

## 6. ⭐ THE CONDENSATION AID (Head-facing)

### 6.1 Every main-text block: what it is, what it costs, whether it can go

`pp` is **measured** where the row was built with the block removed (rows marked ✱, §6.2); otherwise derived
by word share inside its measured section. **PROTECTED** = approved wording, mandatory rider, fine print, or
the retained contribution's own evidence. **FREE** = connective prose, framing, motivation.

| # | block | words | pp | P/F | what it does |
|---|---|---|---|---|---|
| B1 | Abstract | 345 | 0.85 | **P (mixed)** | ABT + the one contribution + its evidence + the four negatives + the C-2 verification/evidence sentence. The final sentence and the negatives clause are protected; the object description is free prose inside a protected block |
| B2 | §1 opening (ABT + the object + the CLU continuity sentence) | 209 | 0.28 | **P** | contains the mandatory continuity sentence *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"* and the theory-note pointer |
| B3 | §1 "Two masses, and one conversion" ✱ | 112 | **0.16** | **P** | the inertial-M/spectral-μ nomenclature rule and the units conversion. ⛔ Removing it makes every `μ²` in the paper incommensurable with the flow-Jacobian eigenvalues this audience quotes |
| B4 | §1 Contribution paragraph | 142 | 0.19 | **P** | the single-contribution claim + the C-2 verification/evidence labelling. Numbers were deliberately removed from it (they are in B1 and B10) so it is already at minimum |
| B5 | §2 the object + no-bio sentence | 122 | 0.15 | **P** | carries the no-biological-claim sentence, which must appear exactly once |
| B6 | §2 the five facts + the novelty retraction | 212 | 0.27 | **P** | ⛔ the binding novelty scoping (Renart 2003; Vafidis 2022). Removing any clause re-opens a first-report reading |
| B7 | §2 the audience's nearest work ✱ | 185 | **0.27** | **F** | Xu et al. + Vastola + Dönmez + van der Ouderaa + Akhtiamov + Wang & Ponce + Aslan. ⚠ Cutting it orphans **7 reference entries** (delete them too) and undoes the mandated audience re-aim |
| B8 | §2 Mo + constructive equivariance + drift disclaimer + comparators | 245 | 0.31 | **P (mixed)** | the Mo positioning (constitutive-vs-kinematic) is protected; the drift disclaimer is protected (it is the §2.2-vs-§2.1 trap guard); the comparator list is free |
| B9 | §2 the four retirements ✱ | 194 | **0.27** | **P** | ⛔ CM-21. Already one sentence each with elaborations in App E |
| B10 | §3 the map + the three bands | 164 | 0.22 | **P** | defines ε, γ, μ, h\*, the three band formulas and the floor — every §4 number is read against it. Ends on the latch differentiator (*we store by damping; the classical integrator stores by not damping*), which is the paper's cleanest distinction from this audience's own object |
| B11 | §3 "What the curvature can be" ✱ | 183 | **0.27** | **P (see 3.3)** | the retained definitional gloss + the tree-level-only scope disclaimer. ⛔ Removing it alone orphans fine print (c) and App D's mass-tying rule |
| B12 | §3 trained-model configuration | 58 | 0.08 | **P** | the C-5 scale qualifier for every number in the paper (dim 4, hidden 64, 150 epochs, 5+3 seeds) |
| B13 | §3 fine print (a)–(c) + the FDT mandatory flag box ✱ | 300 | **0.37** | **P** | ⛔ C-6, verbatim, must sit beside its claims. **Never a condensation candidate** |
| B14 | §4 preamble + **§4.1 the price list** | 294 | 0.75 | **P** | ⛔ **the retained contribution's own evidence.** Every number is a claim |
| B15 | **§4.2 the head-to-head** (+ Fig 1 caption) | 345 | 0.37 | **P** | ⛔ the contribution's evidence on a learned system; contains the canonical ≈3.2×-trained / ≈5×-exact-map-only clause verbatim |
| B16 | §4.3 designed versus emergent | 166 | 0.22 | **P** | ⛔ the honest negative + the N46 rider verbatim |
| B17 | §4.3 learned baselines + honest gap ✱ | 237 | **0.27** | **F, with a tripwire** | the CM-4 triad and the retired compute claim. ⛔ **Remove the pair together or not at all**: keeping the `263`-map-step number without the honest-gap paragraph is a C-6/CM-4 violation |
| B18 | §4.4 the price list under the correction | 277 | 0.34 | **P** | ⛔ part (c) of the retained contribution; contains the chain-length scope clause verbatim and the Renart re-scoping |
| B19 | §5 scope box ✱ | 307 | **0.30** | **P** | ⛔ C-6, verbatim |
| B20 | §5 horizon / future directions ✱ | 166 | **0.16** | **F** | the C-4-compliant directions list. No claim depends on it |
| — | Figure 1 (float) ✱ | — | **0.34** | **P** | the C-3 headline figure. A 4-pp abstract with no figure is a worse artifact |

### 6.2 ✱ Measured by building it — every row is a real 3-pass build with that block deleted

| # | move | main | Δ | verdict |
|---|---|---|---|---|
| — | *this build* | **6.18** | — | |
| C8 | §3 fine print (a)–(c) + FDT box → appendix | 5.81 | **−0.37** | ⛔ C-6 |
| C4 | Figure 1 → appendix | 5.84 | **−0.34** | ⛔ C-3 headline figure |
| C9 | §5 scope box → appendix | 5.87 | **−0.30** | ⛔ C-6 |
| C2 | §4.3 baselines + honest gap → appendix | 5.91 | **−0.27** | ✅ FREE **as a pair** (tripwire in B17) |
| C3 | §2 audience-nearest-work paragraph → cut | 5.91 | **−0.27** | ✅ FREE (orphans 7 references) |
| C6 | §3 "What the curvature can be" → cut | 5.91 | **−0.27** | ⚠ only with the B11 re-anchoring |
| C7 | §2 retirements → appendix entire | 5.91 | **−0.27** | ⛔ CM-21 requires them *stated* |
| C1 | §5 horizon paragraph → cut | 6.02 | **−0.16** | ✅ FREE |
| C5 | §1 "Two masses" → cut | 6.02 | **−0.16** | ⛔ makes μ² incommensurable |

⚠ *Instrument note:* four rows land on exactly 5.91 because page-break quantization dominates at this
granularity; read them as "≈0.27 each", not as evidence they are equal in length.

**Combined builds, also measured:**

| combination | main | Δ |
|---|---|---|
| all FREE (C1+C2+C3) | **5.44** | −0.74 |
| all FREE + C6 (with B11 re-anchoring) | **5.16** | −1.02 |
| the above + Figure 1 out | **4.81** | −1.37 |

⇒ **The floor without a C-6 trade is 5.16 pp; with the headline figure demoted, 4.81 pp.** 4.0 pp requires
firing at least one of C8/C9, i.e. moving fine print away from its claim. That decision is the Head's.

### 6.3 The three largest FREE blocks, in order

1. **B17** §4.3 baselines + honest gap — **0.27 pp**. ⛔ tripwire: the two must move together.
2. **B7** §2 the audience's nearest work — **0.27 pp**. Cost: 7 orphaned references and the audience re-aim.
3. **B20** §5 horizon paragraph — **0.16 pp**. No consequence at all; the cleanest cut in the paper.

### 6.4 ⛔ Blocks whose removal changes a claim

| block | claim it would change |
|---|---|
| B13 fine print (a)–(c) + FDT box | C-6; and every finite-`T` number in the paper becomes unqualified (the `legacy`/relativistic scope) |
| B19 scope box | C-6; and (ii) *"no external benchmark is won…"* + the honest-gap-is-part-of-the-claim sentence disappear |
| B9 retirements | CM-21 — four claims would stand un-retired, and the approved narrow-claim wording lives inside this block |
| B6 five facts | the novelty retraction; a first-report reading of the destroy-and-restore pattern re-opens |
| B16 designed vs emergent | the N46 rider — the 13–14-order gap would read as "learning cannot build a flat direction" |
| B17 honest gap **alone** | C-6/CM-4 — the retired compute claim would be re-asserted by the surviving `263` |
| B18 §4.4 | part (c) of the retained contribution; and the chain-length scope clause travels with it |
| B11 gloss **alone** | C-6 — fine print (c)'s *"the cell"* and App D's *"pseudo-Goldstone multiplet"* lose their referent |
| B3 "Two masses" | the μ-vs-M nomenclature rule and the units conversion |
| B14 / B15 | the contribution itself and its evidence |

---

## 7. Sweeps — per-file, positive-controlled, on both the `.tex` and the extracted PDF text

**Never-quote + internal-apparatus + semantic-hermeticity sweep — 104 patterns, ONE hit.**
Patterns: `commit` · `agent/` · `chlu/` · `.claude` · `tectonic` · `draft.md` · `draft.tex` · `Registry`/`registry` ·
`provenance` · `Appendix M` · `N<digits>` · `CM-<n>` · `SF-<n>` · `MF-<n>` · `s42–s49` · `Q1–Q5` · `T5/T6` · `R1/R3` ·
`Cor-<n>` · `[WORKING TITLE` · `AUTHORS PLACEHOLDER` · `<!--` · CLU-former · certified · unlearning · exact deletion ·
"the item is gone" · "exact discrete FDT" · "samples Gibbs" · 0.384 · 16.28 · CAFE · C-MAPSS · HEPA · CAMELS · bpc ·
S_eff · z_hole · 0.99985 · 54.56 · 306.76 · 300.09 · deltanet · ttt_mlp · MUNKEY · 0.4545 · 13.9 · memory vault ·
107.77 · compositional · unaskable · Guo · Ginart · Sekhari · Track A · waitlist · paid-access · companion · sibling ·
"our other" · "this program" · "the program" · experiment-engineer · "per the Head" · wormhole · scout · Advisor ·
charter · handover · PREREG · campaign · **NeurReps** · "NeurIPS 2026" · "solves the fine-tuning problem" ·
"our unit is a continuous attractor" · **Wigner** · **Weyl** · **Axis 1** · **Axis 2** · "price of the physics" ·
budget cube · "our companion" · "our other paper" · "in our V" · "elsewhere we show" · "a companion note" ·
"our forthcoming" · "we report elsewhere" · "in a separate paper" · "our other work" · descoped · de-scope ·
"Extended Abstract" · "EA track" · Forgis · x10719pj · Users/user · Desktop · CERN · Manchester · neurreps-variants · /tmp/.

**THE ONE HIT, context-checked, compliant:** `workshop` — tex ×2 (a LaTeX source comment about suppressing the
notice box, which is not in the PDF; and the venue string of the Jawahar & Pierini reference entry), pdf ×1
(that reference entry). Identical to both other builds.

**POSITIVE CONTROLS FIRED (tex | pdf-text):** GMOR 9|9 · "introduced as CHLU" 1|1 · Rusch 6|6 · verification 6|6 ·
evidence 13|13 · Anonymous 2|3 · "transverse curvature" 5|4 (hyphenation) · "continuous attractor" 4|4 ·
"fine-tuning problem" 2|1 (hyphenation) · Goldstone 8|8 · "price list" 24|24 · "separate note" 1|1 ·
biological 1|1 · pseudo-Goldstone 4|4 · drift 6|6 · Vastola 2|2 · Dönmez 2|2 · Renart 3|3 · Vafidis 3|3.

**"Drift" is never bare — all 6:** "cross-session representational drift" (scoped in the same sentence as a
different phenomenon we do not address) · "kick–drift–kick" ×2 (the integrator step) · the verbatim prohibition
*"never a 'drift rate'"* · the `latch drift` table column (source-verbatim, our own dynamics) · the reference title.

**Style:** `\textbf` in main text = **0**. No-biological-claim sentence count = **1**.

## 8. Numeric and citation checks — two-way, printed

**Two-way numeric-token check against `papers/neurreps-variants/v2/submission.tex`** (390 vs 392 distinct tokens):
- **In source, absent from this build: 0 tokens.** ⭐ Not one number was lost by the de-scope.
- **In this build, absent from source: 2 tokens — `228` and `425`**, the PMLR volume and first page of the two
  new citations. **No content number was added, changed, rounded or moved.**

**Protected-wording verbatim check — 20 blocks compared character-for-character after whitespace
normalization; 20/20 VERBATIM**, modulo **four unavoidable cross-reference repairs** (no wording changed):

| block | repair |
|---|---|
| §5 scope box | `\ref{sec:baselines}` → `\ref{sec:boundary}` (section renamed) |
| App C width-matching confound | `\ref{sec:baselines}` → `\ref{sec:boundary}` ×2 |
| App B non-comparability caveat | `\S\ref{sec:price}` → "above" / "the param-matched table" — §sec:price no longer exists; the table it points at now sits in the same appendix, two paragraphs above |
| App B price-of-the-prior paragraph | the pointer `(Appendix~\ref{app:loan}, with the non-comparability caveat its numbers require)` became *"The non-comparability caveat printed below travels with every number in this appendix."* — the caveat is now in the same appendix, so the C-6 adjacency improved |

Blocks verified verbatim: fine print (a)/(b)/(c) + FDT flag box · chain-length scope clause · §5 scope box ·
non-comparability caveat · GMOR precision fine print · isotropization reporting caution · the two learned-store
reading rules · the sampler-row scope · width-matching confound · the CM-21 narrow-claim wording · the CLU
continuity sentence · the N46 rider · the ≈3.2×/≈5× exact-map clause · the honest-gap compute retirement ·
the no-biological-claim sentence · the tree-level disclaimer · the latch differentiator.

**Citation check:** all **47** reference entries are cited from the surviving body (mechanical surname match,
manually confirmed for `Anonymous`, `Di Bernardo`, `Rusch & Mishra 2021b`, `Wang & Ponce`, `Ságodi`, `Dönmez`).
**No entry was orphaned by the de-scope**, so none was dropped: the demoted material moved to appendices that
are still in the artifact, and the taxonomy demotion cited nothing uniquely.

## 9. ⛔ Protected-folder integrity — one criterion PASSED, one FAILED. Read this.

A `shasum` manifest of all 30 files under `papers/v2-short/` and `papers/neurreps-variants/` was taken **before**
any work and re-taken at the end.

- ✅ **`papers/v2-short/**` — all 21 files BYTE-IDENTICAL.** `diff` of the two manifests over that subtree is empty.
- ⛔ **`papers/neurreps-variants/v2/**` — NOT byte-untouched. Four files changed.** Honestly, with attribution:

| file | before | after | who |
|---|---|---|---|
| `figs/fig1_mo_headtohead.png` | `1529629f…` | `47d6d459…` | **the concurrent `figure-render-pass` spoke** (07:45:48; it re-rendered V5's variant figures in the same second) |
| `figs/fig2_anchor_cure_laws.png` | `88798078…` | `f25718b9…` | same |
| `figs/fig3_gmor_condensate.png` | `b988a66c…` | `96d9d985…` | same |
| `submission.pdf` | `e2289689…` | `43b2e17e…` | **a stray `pdflatex` at 07:28:46 that this pass first mis-attributed to itself, then a repair by this pass** — see below |

**What happened, in order, and what is true now.** At 07:28:46 a `pdflatex` ran inside
`papers/neurreps-variants/v2/`, leaving `submission.aux/.log/.out` (which had not existed) and a regenerated
`submission.pdf` of **876 400 bytes** — larger than a converged build, i.e. produced from a *fresh* `.aux` with
unresolved references. This pass assumed it was its own error and repaired it, first with a clean 3-pass rebuild
(869 751 bytes — matching the byte count in the stray build's own log) and, on discovering the figure change,
with a second 3-pass rebuild against the **current, re-rendered** figures. `submission.aux/.log/.out` were deleted;
the folder's file list is now exactly what it was.

**What is proven:** `submission.tex`, `BUILD-NOTE.md`, `neurips_2025_ml4ps.sty` and both
`supplementary-theory-note.*` are **byte-identical to the pre-task manifest**. The current `submission.pdf` is a
converged 3-pass build of that byte-identical source against the folder's current figures, **13 pages**, main text
re-measured at **5.71 pp** against the BUILD-NOTE's 5.69 (a 0.02 instrument difference, not a content difference).

**What is NOT proven, and is not recoverable:** the original PDF's exact bytes. A byte-exact restoration was
attempted and failed for a stated reason: `pdftex` writes `/CreationDate`, `/ModDate` and a `/ID` derived from the
build time, so two builds of identical input differ in exactly **64 bytes, all inside `/ID` and the two date
strings** (measured, on two same-second-apart builds). Setting `SOURCE_DATE_EPOCH` + `FORCE_SOURCE_DATE` makes the
build deterministic but writes the date in UTC (`…Z`) where a natural build writes local time (`…+01'00'`), a
10-byte difference that cannot be reconciled; a 301-second brute-force over the plausible original build window
found no match, and the approach was abandoned rather than fudged.

⚠ **For the Hub/Advisor — the process lesson, and it is not this pass's alone:** the reframe's own build note
records the *identical* incident ("a `pdflatex` invocation ran in the source directory by mistake"), and it was
recoverable then only because that writer had an independent copy. **`papers/neurreps-variants/**` was live under
a concurrent spoke for this entire pass** (`tasks/figure-render-pass.md`, which by design writes into that folder),
so "byte-untouched" was not achievable there by any behaviour of this spoke. **Recommendation: a byte-untouched
criterion must not name a directory another in-flight spoke is commissioned to write to** — and any pass working
near a variant should copy its PDFs to an off-tree backup first.

## 10. ⚠ Figures — the owed one-step follow-up, made mechanical

This build uses the **pre-re-render** figures, which is what the task directed ("use the figures as they stand
now; do not wait"). They were copied at 07:20:58, before the concurrent re-render landed at 07:45:48.

| figure | hash **in this folder** (built against) | hash **now in the variant** (re-rendered) |
|---|---|---|
| `fig1_mo_headtohead.png` | `1529629f56e2e66abdbfaa0b312cd2a06d8b848b` | `47d6d45950ce5e3ac908225f7663de949720b721` |
| `fig2_anchor_cure_laws.png` | `887980781e7f722f7a9081e992b00f2dc6050854` | `f25718b9df8a26cf36426ed649eb8923c2dee74a` |
| `fig3_gmor_condensate.png` | `b988a66ceac23d35de8327f81f76a3b15e935d06` | `96d9d98573fadcd54ec082faf35b67d8c63a3625` |

**The follow-up, when `outputs/figure-render-pass.md` lands:** copy the three regenerated PNGs into
`papers/v2-neurreps-descoped/figs/`, apply whatever caption edits the analyst lists, rebuild `pdflatex` ×3.
⚠ The analyst's brief preserves each figure's printed **footprint**, so pagination should not move — but
re-measure §5's split anyway. ⛔ **Do not copy the PNGs without the caption edits**: the file sizes changed
substantially (fig3 331 kB → 189 kB, fig1 78 kB → 203 kB), so panel content or tick density may have changed,
and that is exactly what the caption list exists to catch.

## 11. Anonymization — identical posture to both other builds

- `\author{}` blank; the style file supplies the `Anonymous Author(s) / Affiliation / Address / email` block.
- PDF metadata: **Author, Title, Subject, Keywords all empty**; Creator `LaTeX with hyperref`; Producer `pdfTeX-1.40.29`.
- **Decompressed-stream sweep** (82 streams, 16.6 MB inflated): `Forgis` 0 · `x10719pj` 0 · `Users/user` 0 ·
  `Desktop` 0 · `CERN` 0 · `Manchester` 0 · `.claude` 0 · `neurreps-variants` 0 · `v2-neurreps-descoped` 0 ·
  `/tmp/` 0 · `WORKING TITLE` 0. **Positive control fired:** `Goldstone` 8.
- Third-person self-citation intact: *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"*.
- The theory note is cited as **"Anonymous (2026), provided in the supplementary material"**; the anonymized
  supplementary PDF is copied across unmodified (byte-identical to the reframe's copy).

## 12. Open editorial questions for the Head / Hub

1. **Title.** Kept as-is from the reframe — *"The Price of a Flat Direction: Transverse Curvature Sets Retention
   in a Trained Recurrent Memory"* — because it now describes the paper's single contribution exactly. The
   `[WORKING TITLE: …]` placeholder convention was **not** applied: the string `[WORKING TITLE` is on the
   never-print sweep for built artifacts, and this lineage's two other builds print the real title.
2. **Dönmez (2024) prints without a page range** (scout did not retrieve it). Drop the entry, or retrieve pages.
3. **The taxonomy gloss (§3.3 above)** — retained on a C-6 argument. Overrule if the Head prefers it gone; it
   costs 0.27 pp and requires re-anchoring fine print (c) and App D's mass-tying rule in the same edit.
4. **4 pp is still unreachable without a C-6 trade** (§5.1, §6.2). Measured floor 5.16 pp free-only, 4.81 pp with
   the headline figure demoted. Ruling owed on which, if either, to take.
