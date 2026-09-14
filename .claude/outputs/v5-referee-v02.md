# v5-referee-v02 — paper-referee report

**Task + acceptance criterion:** adversarial referee pass on `papers/v5-short/draft.md` v0.2 (V5's first-ever referee pass); MUST/SHOULD/NICE triage with citations, simulated PALM-short verdict, both Add.21/Add.23 carry-items explicitly discharged, positive-controlled sweep printed, zero edits outside this report.
**Status:** done.
**DIAL DECLARATION (echoed):** none — adversarial review; no performance claim; no laundering control applies to *this* report (but see MF-9: a laundering control that fired is missing from the *draft*).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — READ FIRST (protocol §5 corollary, needs a named owner):**
> 1. **MF-13** — `venue-follow-up.md` §3 claim-(a) basis is still uncorrected (Add.24 routing item 4, undischarged), and **V5 inherited the error verbatim into App H(a) and K.2**. Owner: Hub (source) + writer (draft).
> 2. **MF-1/MF-2** — the "≈11 decades" instrument note exists in **V2 App K.4** and is **absent from V5**. A lockstep divergence: fixing it in V5 only is correct, but the Hub must confirm V2's note is not itself edited out at `v2-revision-7`.
> 3. **MF-14** — matrix §3 says the F5 note may be cited "once live"; it is not live and V5 leans on it in §1. Head decision, not a writer fix.

---

## VERDICT (simulated PALM short track): **BORDERLINE — reject-leaning as submitted; weak-accept after the 14 MUST-FIX.**

**Meta-review.** This is an unusually disciplined draft: the never-quote sweep is clean (0/25 forbidden patterns, positive controls live), the Blelloch–Golovin attribution is present at all seven exact-deletion sites, "certified" appears only in denial or literature-description form, N108's required sentence is present and unsoftened, N112's overflow clause and N127's capacity removal both travel, the §0.13 lifecycle wording is verbatim with both riders, the score sentence is stated, the substrate-scope sentence is in the paper's own voice, and the 4-pp claim is *measured* (tectonic: `maincount.xdv, 4 pages`), not asserted. That is the best charter/matrix compliance the program has produced. **It is nonetheless not submittable, for three reasons a PALM reviewer will reach before they reach any of the above.** (1) **The headline figure does not show the headline claim** — Fig. 1(a) plots three emergent seeds spanning <½ a decade of μ² under a caption asserting eleven, with the designed arm present only as an unplotted legend entry, and both panels carry internal source-report labels ("Cor-13", and "T6"/"T5" in Fig. D.1) — the exact V1-referee F1/F7 failure class, repeated. (2) **The "≈11 orders of magnitude" number is fragile in a way V2 already knows how to defuse and V5 does not**: the low endpoint 1.7×10⁻¹² is the *Jacobian ring-profile instrument floor* on the designed checkpoint (`v5-gate` §3.1 "the ring-profile machinery validates on designed (ripple 1.9e-16, μ²_ring = 1.7e-12)"), while §3.1 in the same paragraph quotes the designed flat coset at μ²≈10⁻¹⁵ — V2 App K.4 carries an explicit instrument note precisely so "the spans quoted in this paper cannot be read as inconsistent", and V5 dropped it. On top of that the whole V-curve is a **one-step-Jacobian** measurement whose own source reports a **19–43% disagreement with direct rollouts** (`v5-gate` §3.2: 233.6/653.3/249.0 Jacobian vs 190/370/150 rollout) that appears nowhere in the paper. (3) **Mechanical defects that make the compiled artifact fail inspection**: nine dangling `§3.5` cross-references to a section that does not exist; ~39 hard-coded `§3.x` references that point at *Related work* in the built PDF because `\section{Results}` compiles as §2; a limitations list that skips (iv); five appendix figures cited as raw PNG filenames and never embedded; **and no bibliography at all** — every one of the ~30 external citations (Blelloch–Golovin, Guo §3 Eq. (1), Ginart Def. A.5, Sekhari Def. 3, Mo 2026, Minami & Hidaka, …) is unresolvable. Add one arithmetic error inside the appendix that defends the headline vault (D.1's "γ_eff/(2−γ_eff) = 13.88×" evaluates to 0.356), one uncorrected mis-citation the program has already corrected on the record (arXiv:2503.21536 attributed to CD bias), one missing mandatory scope clause (the k-regime clause vs Decelle/Agoritsas), and one omitted laundering control that *fired* (CLU-with-a-TTL-flag is within 0.017 AUC of physical decay against the exact adversary — measured, in `mia-decay-measurement` §item-4, absent from the draft), and the review writes itself. **None of the MUST-FIX requires a new experiment.** Fixed, this is a weak-accept: honest, dense, well-scoped, with a genuinely unusual amount of self-falsification on the record — and with a scale story (dim 4 / dim 3 / capacity 8 / 384 synthetic instances) that will keep it out of accept territory at a venue whose remit is long-term memory for *AI systems*.

---

## A. Findings — MUST-FIX (block submission)

### MF-1 · §3.1 line 35 + line 37 + abstract line 13 — the "≈11 decades" span is internally contradictory, and V2 carries the fix V5 dropped
**Attack.** §3.1 states in one paragraph that the designed flat coset has "μ²≈10⁻¹⁵" and that "across both families the curve spans μ²∈[1.7×10⁻¹², 7×10⁻²] — **eleven orders of magnitude**." A reviewer computes: if the designed coset is at 10⁻¹⁵ and it is "the same curve at μ→0", the span is ~13.8 decades, not 11; if the span is 11, the designed coset is *not* on it. Either way one of the two sentences is wrong as printed.
**Evidence.** `v5-gate.md` §3.1 line 194: *"The ring-profile machinery validates on designed (ripple 1.9e-16, `μ²_ring = 1.7e-12`)"* — 1.7e-12 is the **Jacobian ring-profile instrument floor**, not a physical spectral mass. **V2 `draft.md` App K.4 states this explicitly**: *"the designed endpoint here is the smallest μ² the **Jacobian-derived** γ-grid probe of B.8 can resolve on this curve — not the **Hessian** μ² ≤ 2.4×10⁻¹⁵ of §3.3/B.1 … eleven orders is the span of one measured curve, 13–14 orders is the Hessian gap."* V5 has no such note anywhere (grep: 0 hits for "Hessian" in §3.1/App C).
**Reviewer-failure scenario.** "The authors' headline is a span of eleven decades. Two sentences later they quote the low-mass endpoint as 10⁻¹⁵. The span is either 11 or 14 and I cannot tell which measurement it is, and the low endpoint appears to be a numerical resolution floor rather than a mode."
**Triage: MUST-FIX.** Import V2 K.4's instrument note verbatim (≈35 words, main text, C-7).

### MF-2 · §3.1 line 35 — "one curve at **three** values of μ" contradicts the matrix-approved wording and is physically wrong
**Attack.** The approved unification (CM-16b; `t-lever-forgetting` §8.3; V2 App K.4) is **"one damping-optimum curve evaluated at *two* values of μ, not two laws."** V5 writes "latch, overdamped register and working memory are that *one curve at three values of μ*". But the underdamped "working memory" branch is by the paper's own A.0 the **mass-independent** envelope `2ln2/(−ln(1−γ))` — it is not a third value of μ, it is the same mode at a different γ.
**Evidence.** Draft A.0: *"an **underdamped working memory** at the mass-independent envelope"*; matrix CM-16b + V2 K.4: "two values of μ".
**Triage: MUST-FIX.** Charter C-8/M2 (one coherent program, zero contradictions between shorts) + internal physics error. Restore "two values of μ", or rewrite as "three regimes of one curve".

### MF-3 · §3.1 — the headline law is measured on a Jacobian instrument whose rollout cross-check disagrees by 19–43%, undisclosed
**Attack.** Both arms of the V-curve are read off the one-step Jacobian ("from the model's own one-step Jacobian"; figure y-axis "n₁/₂ = ln2/gap"). The source report gives the direct-rollout comparison and it does not agree.
**Evidence.** `v5-gate.md` line 207: *"`n₁/₂(T=0, γ=0.05) = 233.6 / 653.3 / 249.0` steps (Jacobian; rollout-measured 190 / 370 / 150)"* → ratios 1.23 / 1.77 / 1.66. Zero occurrences of "rollout" in the draft's §3.1/App C.
**Reviewer-failure scenario.** A reviewer who runs the code (or reads the anonymized supplement) finds the model forgets 20–45% faster than the paper's instrument says, and the paper never mentions it. This is precisely the N51-class "an instrument's validity is a separate empirical question" rule the program invented — applied to everything except its own headline.
**Triage: MUST-FIX.** One sentence in §3.1 or App C.3 stating the Jacobian/rollout ratio and why the Jacobian is the quoted instrument (linear-response gap vs finite-amplitude write), plus the numbers.

### MF-4 · §3.1 / App B.6 / App C.3 — the two arms disagree on the overdamped exponent and neither matches the prediction; no explanation given
**Attack.** The paper asserts one law and prints, for the same branch, designed **+1.23 … +1.27** and emergent **+1.116 ± 0.011** — non-overlapping, ~10–14% apart, against a theoretical **+1**. The argmin deficit gets an explanation ("the known discrete-map correction"); the slope deviation gets none.
**Evidence.** `t-lever-forgetting.md` §8.3 (slope "+1.23…+1.27 for γ > 2γ_crit"; note the *window* qualifier `γ > 2γ_crit`, which the draft also drops); `v5-gate.md` §3.4 table (+1.1236 / +1.1010 / +1.1226).
**Reviewer-failure scenario.** "The authors claim a single law. Their two families give overdamped exponents whose error bars exclude each other and exclude the predicted value of 1. Why?"
**Triage: MUST-FIX.** Either (a) restore the asymptotic-window qualifiers ("slope measured over γ < γ_crit/2 and γ > 2γ_crit") and state that the residual excess is the finite-window/discrete-map correction with its size, or (b) report the exponents in the collapsed variable so the two agree.

### MF-5 · Figure 1 (headline) — the figure does not show the claim, and it leaks an internal label
**Attack, four parts.** (a) Panel (a) plots **only the three emergent seeds** — μ²∈[2.0e-2, 5.4e-2], i.e. **less than half a decade** — under a caption asserting eleven decades and asserting the emergent coset "traces the same curve **as the designed massive mode**", which is *not drawn*. (b) The legend entry "designed s44 (n₁/₂ = ∞)" has **no visible line** — an empty legend key. (c) Panel (b)'s title reads "**Cor-13** holds only for the DESIGNED symmetry" — "Cor-13" is an internal theory-note/source-report label that appears **nowhere in the paper**. (d) Panel (b) is the *designed-only negative* (N46) and consumes 50% of the headline figure's area; the unification consumes none of it at scale.
**Evidence.** `figs/fig1_vcurve.png` (inspected); `v5-gate.md` §3.4 table row "designed s44 | γ_crit = 0 | — (n₁/₂ = ∞ ∀γ)" — the designed arm contributes no point to this curve by construction. Precedent: V1 referee F1 ("the headline figure draws … three indistinguishable flat lines — the visual opposite of the thesis") and F7 ("an internal source-report section number leaked into a figure title"), `philosophy-synthesis.md` line 771.
**Triage: MUST-FIX.** Replot as a **collapse**: n₁/₂/n_min vs γ/γ_crit for designed-massive (5 seeds) + emergent (3 seeds) + the designed flat corner, with a μ² axis or colourbar that *shows* the decade span; strip "Cor-13"; move the pseudo-Goldstone panel to App C (where `figC_register_capacity.png` already lives).

### MF-6 · Whole document — no bibliography; ~30 citations are unresolvable
**Attack.** The draft cites Blelloch & Golovin FOCS'07, Guo et al. 2020 §3 Eq. (1), Ginart et al. Def. A.5, Sekhari et al. Def. 3, SISA/Bourtoule 2021, SILO 2024, PALL 2025, Ticketed L-U COLT'23, MUSE, CURE4Rec, Micciancio STOC'97, Naor & Teague STOC'01, Hartline et al. 2005, Buchbinder & Petrank CRYPTO'03, Blelloch–Golovin–Vassilevska SWAT 2008, Minami & Hidaka 2018/2020, Mo 2026, Di Bernardo et al. 2025, Iqbal et al. 2026, Golubitsky et al. 1988, Krupa 1990, Fischer & Igel 2011, Nijkamp et al. 2020, arXiv:2503.21536, Du & Mordatch 2019, Hinton et al. 1995, Tieleman 2008, Hochreiter & Schmidhuber 1997, Rusch et al. 2022, Rusch & Mishra 2021, Hairer–Lubich–Wanner, Jawahar & Pierini 2026, Anonymous 2026 — and has **no `## References` section**. V2's draft has one at line 70.
**Evidence.** `grep -c "^## References" v5-short/draft.md` = 0. Add.21 flagged exactly this for V2 ("not mechanically submittable (5 pp / no bib)"); the finding was never transferred to V5.
**Triage: MUST-FIX.** References are excluded from the 4-pp limit (Add.20 venue quote), so there is no page cost.

### MF-7 · Whole document — nine dangling `§3.5` references and ~39 `§3.x` references that are wrong in the compiled PDF
**Attack.** (a) `§3.5` is referenced 9 times (A.3, App C.7, App H rider, App J items 1/11/12, K.2, K.3) — **there is no §3.5**; the material moved into §3.2. (b) The markdown numbers Results as "## 3." but `draft.tex` compiles `\section{Introduction}`, `\section{Results}` ⇒ **Results is §2 in the PDF**, so all 39 hard-coded `\S 3.x` refs point at *Related work* or nothing. (c) The main text has no §2 at all (jumps 1 → 3). (d) The Limitations list runs (i)(ii)(iii)**(v)**(vi) — (iv) is missing, so a reviewer sees a deleted limitation.
**Evidence.** `grep -c "§3.5" draft.md` = 9; `grep -n "^\\section{" draft.tex` → Introduction, Results, Related work, Limitations; `grep -o "3\.[0-9]" draft.tex | sort | uniq -c` → 39 hits. Matrix §3: *"No cross-short citations, **no dangling refs**."*
**Triage: MUST-FIX.** Convert to `\label`/`\ref`, or hand-renumber; restore (iv) or renumber.

### MF-8 · App D.1 — arithmetic error in the sentence that defends the headline vault
**Attack.** D.1 prints *"A coupled bath would give D_θ ∝ γ_eff⁻¹ and a vault of **γ_eff/(2−γ_eff) = 13.88×**."* With γ_eff = 0.525 that expression equals **0.356**, not 13.88. The correct vault ratio is [γ_eff(2−γ)]/[γ(2−γ_eff)] = (0.525×1.95)/(0.05×1.475) = 13.88.
**Evidence.** `v5-gate.md` §2.1 writes the *scaling* correctly (`n₁/₂ ∝ γ_eff/(2−γ_eff)`) and the ratio separately; the draft collapsed a scaling into an equality. Everything downstream (7.942, 8.11 ± 0.37, 110.25) is arithmetically consistent — this is a transcription defect, not a result defect.
**Triage: MUST-FIX.** A physics reviewer checks this in ten seconds and it sits inside the discriminator that makes the 107.77× credible.

### MF-9 · §3.3 / App E.5 — the laundering control that FIRED is omitted from the paper
**Attack.** The draft's lifetimes claim leads with retrieval geometry vs a **TTL vector-store** (R₅₀ 1.146→0.752 vs constant ≈0.77). It never reports the *tighter* control from the same report: **CLU-with-a-TTL-flag** (identical store/adversary, `amp ≡ 1` until expiry), which against the exact adversary is **within 0.017 AUC of physical decay (0.983 vs 1.000) — "no measurable differentiator"**, and the source explicitly records "⛔ **Against an exact adversary the laundering control FIRES**".
**Evidence.** `mia-decay-measurement.md` items 4 and §3(a), line 127. Charter §4.1 genuine-win bar; AGENT_PROTOCOL §7 laundering-control rule.
**Reviewer-failure scenario.** "The authors' own supplementary reports a control in which a boolean expiry flag is indistinguishable from their physical decay. It is not in the paper."
**Triage: MUST-FIX (wiring, not a new run).** One clause in §3.3 + a row in E.5. This *strengthens* the paper: it makes the retrieval-geometry differentiator the honest survivor rather than an unexplained choice.

### MF-10 · App H(a) + K.2 — an uncorrected mis-citation the program has already corrected on the record (**CARRY-ITEM 2, part 1**)
**Attack.** Both sites state *"that CD induces spurious symmetry breaking between degenerate sectors is documented in the RBM literature (arXiv:2503.21536)"*. That paper (Toledo-Marin, Maiti, Fox & Melko, MLST 6:035030, 2025) attributes symmetry breaking to **hierarchical feature learning, not to CD**; the symmetry broken is an *initialization* symmetry of the weight spectrum; CD/PCD appear only as sampling implementations.
**Evidence.** `outputs/v2-cite-check.md` §Part 2 line 328 + reconciliation item 6 + line 390: *"The 'CD induces spurious symmetry breaking in Boltzmann machines is documented' basis does **not** survive; the defensible substrate cites for claim (a) are Fischer–Igel 2010/2011 + Nijkamp et al. 2020."* Add.24 routing item 4 records the source correction as still **owed** — V5 inherited the pre-correction text.
**Triage: MUST-FIX.** Delete the 2503.21536-as-CD sentence at both sites; cite Fischer & Igel **2010, 2011** (the draft has 2011 only) + Nijkamp 2020; keep 2503.21536 only if described accurately, as a *different* mechanism.

### MF-11 · App H(c) — the mandatory k-regime scope clause is absent (**CARRY-ITEM 2, part 2**)
**Attack.** H(c) states the erosion horizon is "set by sleep-update *frequency* … **independent of chain length**: … the number of sleep sub-steps (50 vs 500) is irrelevant to ±1 epoch" — flatly, against a literature in which *k* vs mixing time **defines the learning regime**.
**Evidence.** `v2-cite-check.md` line 337: *"the draft/V5 cannot say 'chain length is irrelevant' flatly against a literature in which k vs mixing time defines the learning regime"*; Add.23 makes the clause binding. Recommended wording is on record (cite-check §Ship-guidance): *"within our sweep both chain lengths (k=50 and 500) sit on the same side of the model's mixing time, so the frequency-decisive/steps-irrelevant finding is a statement about that regime, not a contradiction of the k-dependence documented for RBM learning (Decelle et al. 2021; Agoritsas et al. 2023)."*
**Triage: MUST-FIX.** Paste the approved clause; add both references to the bib (MF-6).

### MF-12 · App H — a stale "pending a targeted follow-up" marker for a scout that already ran
**Attack.** H closes with *"Novelty of (b)/(c) is single-sourced by absence and flagged for a targeted follow-up (continual-EBM, equilibrium-propagation and RBM-symmetry-breaking literature) **before camera-ready**."* That follow-up ran twice: the Jul-20 novelty scout (Add.21 MF-5 resolution) and `v2-cite-check` Part 2 (Add.23), which returned **(b) CONFIRMED-NOVEL** and (c) novel-as-stated. Shipping a to-do note in a submitted appendix invites "the authors have not checked their own novelty claim."
**Evidence.** `v2-cite-check.md` line 377/389.
**Triage: MUST-FIX.** Replace with the ship-guidance sentence (which is *stronger*: an executed targeted search that found nothing).

### MF-13 · Abstract line 13 — three scope-widening omissions in the one paragraph everyone reads
**Attack.** (a) *"a localized friction hole is a 107.77 ± 4.78× memory vault"* — no designed-only qualifier, in an abstract that carefully qualifies the V-curve with "3/3 seeds (dim 4, hidden 64, laptop-CPU)". The vault is measured on **3 designed SO(2) seeds** on precisely the coset register §3.2 shows **does not emerge**; a reader infers the vault generalizes. (b) *"store-level deletion is exact, the post-deletion store being byte-identical at every load measured"* — the **three stated conditions** (`budget ≥ n_cells`, `leak = 0`, priority/attribute eviction) and the **recency exclusion** are dropped, which is the "unqualified exact deletion" N118 explicitly bans and CM-25(f) requires. (c) No scale qualifier anywhere on the deletion claim (dim 3, capacity 8–64, no learning).
**Evidence.** N118 (⛔ unqualified "exact deletion"); CM-25(f) verbatim; charter C-5; A.4 provenance (3 designed seeds).
**Triage: MUST-FIX.** ≈15 words; C-5/C-6 are draft-review checklist items for exactly this.

### MF-14 · §1 line 21 — "a **companion** theory note (Anonymous, 2026)": C-8 semantic hermeticity + an uncitable load-bearing reference
**Attack.** (a) Matrix §3 permits the F5 note *"(third person, **once live**)"*. It is not live (handover line 752/3755/3773/3790 track the arXiv push as the standing #1 blocker; Add.24 does not clear it). (b) "a **companion** theory note" is not third person — it asserts co-authorship with an anonymous artifact, which under PALM double-blind announces a coordinated submission set (M2/M3), and it is the exact "companion/program/sibling language" the Add.21 semantic sweep targets. (c) The sentence is load-bearing: it is where "the exactly-solvable core" is deferred to, so a reviewer cannot check the derivation the V-curve rests on.
**Evidence.** `grep -oi companion draft.md` = 2 (this + a benign use in C.6); `claims_matrix.md` §3; charter C-8.
**Triage: MUST-FIX + Head decision.** Either the note is public before submission and is cited in third person without "companion", or §3.1/A.0 must carry the 2×2-matrix reduction self-containedly (it nearly does — A.0 is one sentence away).

---

## B. Findings — SHOULD-FIX

- **SF-1 · App B.5 vs C.2/C.7 — two different "designed" flatness numbers, unlabelled.** B.5 gives `||λ_flat|−1| ≤ 1.7×10⁻¹⁴` (t-lever, A.2); C.2/§3.2 give "designed ≤ 1.1×10⁻¹⁵" (v5-gate, A.3). Both are true on their own grid, but the draft never says so, and C-7's standard is that *"apparent contradictions between differently-flagged runs must be impossible to construct."* One parenthetical fixes it.
- **SF-2 · §3.3 — "the packing price of exactness is negative" is asserted with no number in the paper.** The backing numbers exist and are matrix-approved: *"61/64 admitted, per-admitted 1.0000, per-offered 0.9531 … against refuse-and-relocate's 43/64"* (CM-23(v); `placement-landing.md` §3, line 99/197). Print them in E.3. A comparative claim with no number is the first thing a reviewer marks.
- **SF-3 · §3.3 — retention 0.832 is quoted without its placement scope.** The standing caveat: *"every retention magnitude here is placement-dependent: on a max-radius ring the same store gives **0.500** at A = 0.06 where the controller-placed disk gives **0.886** ⇒ state the |c| distribution with any retention number"* (N119 / N108 w26 block, `HEP_primers.md` Record 5). Add the |c| clause once in E.5.
- **SF-4 · Fig. D.1 panel (c) — internal labels "T6"/"T5" in a figure title and inset, and the bars show 84×/86×/91× (raw FPT) under a caption whose headline is 107.77× (D̂).** The paper argues raw FPT is boundary-layer-biased (D.3) — correct — but the *visual* still shows the number the text calls biased. Annotate the D̂ value on the bars; strip T5/T6.
- **SF-5 · Five appendix figures cited as filenames, never embedded.** B.6 ("Figures: `figB_dlaw.png`, `figB_signflip.png`, `figB_massive_vs_flat.png`"), C.2 ("Figure `figC_register_capacity.png`"), C.4 ("Figure `figC_Tstar.png`"). All five exist in `figs/`. As printed these are drafting artifacts in the submitted supplement.
- **SF-6 · §3.1 — the asymptotic-window qualifiers on the slopes are dropped.** Source: "−1.006 for γ < γ_crit/2, +1.23…+1.27 for γ > 2γ_crit". Draft: "below"/"above". Related to MF-4; cheap to restore.
- **SF-7 · Related work / §4 — zero contact with the venue's own literature.** §4 covers dissipative Goldstone physics, equivariant RNNs, history-independent data structures, machine unlearning and gated RNNs. PALM is *long-term memory for AI systems*; there is no citation to agent/LLM long-term-memory work (external-memory agents, retrieval-augmented long-term stores, memory-consolidation/forgetting policies in deployed systems), which is the literature the reviewers are drawn from. Note that **PALM's topic scope was never scouted** (charter Add.7/Add.8 Q7: *"PALM's topic scope (a known gap)"*, sweep deferred by the Head). This is the largest fit risk after scale.
- **SF-8 · §3.4 buys nothing in a hard 4-pp budget.** 124 words of main text that state, correctly, that no value or benchmark number is claimed and that the surface is a declared not-run. It is compliant and it is content-free to a reviewer. Moving it to App F (it is already there in full) buys ≈0.19 pp — very close to what the MUST-FIX additions cost (see §D).
- **SF-9 · Three contributions in a 4-pp short.** The advisor's own standing test (charter §5 J2, quoted in Add.19): *"a workshop short earns its page count with one honest contribution + its controls."* §3.3 gets one column for an entire deletion estate and §3.4 gets six lines. A reviewer will read this as three abstracts. Recommend: contribution (1) leads and is defended; (2) is compressed to the CM-25(f) sentence + attribution + N108; (3) goes to the appendix.
- **SF-10 · E.7 / K.2 — the closest competing work is cited as "a recent preprint" with no name, no venue and no identifier.** N168 requires the form *"an ICLR-2026 workshop paper (oral)"* with the workshop name quarantined; the draft under-cites it as a preprint and never names it, so the reader cannot check the one paper that most narrows the novelty claim. Give the arXiv id + "an ICLR-2026 workshop paper (oral)"; the "not exact, gap 0.56 ± 0.21" narrowing is already correct.
- **SF-11 · Missing falsifier statement for contribution (1).** Nowhere does the main text say what would have falsified the V-curve. A reviewer's reflex — "this is the textbook underdamped/overdamped crossover of a 2×2 linear map; measuring that a trained model's Jacobian obeys linear response is not a finding" — is answerable (mode mixing / anharmonicity / non-quadratic soft directions could have broken the single-mode reduction; the ±25% agreement with the exact-Goldstone law above T\* is evidence it did not) but is never answered. One sentence.

## C. Findings — NICE

- **N-1** · §3.3 quotes the TTL lookup radius as "≈0.77" where E.5 and the source give "0.75–0.77" / "0.75 constant". Quote the range or the P5 value.
- **N-2** · "Venue class" header block and the "Draft status" block are drafting furniture; strip before submission.
- **N-3** · Fig. 1(a) y-axis "n₁/₂ = ln2/gap (steps)" is an instrument formula, not a reader label.
- **N-4** · The three emergent curves in Fig. 1(a) are visibly *not* collapsed above the minimum (s43 sits ≈2.5× above s42/s44); normalizing by n_min would make the "3/3 seeds, one curve" claim visually true as well as numerically true.
- **N-5** · §3.2's "⚠ The designed-symmetry precondition" is the paper's strongest honest content and is currently a mid-paragraph aside inside the vault subsection. It deserves its own bolded lead-in.
- **N-6** · C.2 quotes emergent `1−|λ_coset|` at γ≈0.048 but omits the grid minimum (7.6e-5 … 2.0e-4, `v5-gate` §3.2 table) — the more conservative number, and quoting it strengthens the honesty posture.

---

## D. 4-pp compliance

**Measured, and the measurement is sound.** `scratch/v5-rebuild/maincount.log`: `Output written on maincount.xdv (4 pages)`; `draft.log`: 17 pages full. Main text = 2,617 words, ≈654 words/page in `article`/10pt/1in margins/single column, headline figure included. References excluded per the Head-verified venue text (Add.20). **Two caveats:**
1. **The venue style is not the measured style.** `maincount.tex` line 1 is `\documentclass{article}` with `[margin=1in]{geometry}`. A PALM style file with a narrower text block or a smaller body font moves this either way; the 4 pp is a *generic-article* 4 pp. Re-measure in the venue class before freezing — the draft's own comment line 6 already says so.
2. **The MUST-FIX additions cost ≈100–120 words ≈ 0.17 pp** (MF-1 instrument note ≈35 w; MF-3 rollout caveat ≈25 w; MF-4 window qualifiers ≈15 w; MF-9 laundering clause ≈30 w; MF-13 abstract qualifiers ≈15 w), plus `\cite` keys which perturb line breaks.

**Move-to-supplementary menu (in preference order; the first two suffice):**
| # | move | main-text saving |
|---|---|---|
| 1 | **§3.4 lifecycle → App F** (already there in full); leave one sentence in §1 contributions | ≈124 w ≈ 0.19 pp |
| 2 | §3.3's second paragraph ("Attribution, in the same breath…") → keep the one-clause form in-line, full block to E.2 (already there verbatim) | ≈95 w ≈ 0.15 pp |
| 3 | §3.3's trilemma paragraph → E.6 (already there in full); keep the compute-adaptive-read corner sentence | ≈110 w ≈ 0.17 pp |
| 4 | §3.2's gated-stiffness `R₅₀` numbers → E.6 | ≈40 w ≈ 0.06 pp |
| ⛔ | Do **not** cut: N108's sentence, the CM-25(f) verbatim, the BG attribution clause, the score sentence, the §A20.5 substrate sentence, the designed-symmetry precondition, the `fdt`+Newtonian fine print | — |

---

## E. CARRY-ITEM DISCHARGE

### ⭐ Carry-item 1 — the V2↔V5 lockstep table (Add.21). **DISCHARGED. Two divergences, both in V5's disfavour.**

| shared quantity | V2 v0.7 (App K / §3) | V5 v0.2 | verdict |
|---|---|---|---|
| diffusion law ratio | `1.0068 ± 0.0219`, 25 cells, min 0.9644 max 1.0484 (K.2) | identical (§3.2, B.2) | ✅ identical |
| sign-flip slope | `+0.955 ± 0.042`, T=1e-3, 5 seeds, 10/10 (K.3) | identical (§3.2, B.3) | ✅ |
| γ:0.05→0.2 effect | `3.77 ± 0.23×`, per-seed 3.55/3.64/3.61/3.89/4.15 (K.3) | identical (§3.2, B.3) | ✅ |
| dlog n/dlog T | `−0.956 … −0.979` (K.3) | identical (B.3) | ✅ |
| absolute law | `0.378748 Δ²F²γ/(ε²T(2−γ))`; 1.054±0.057 at ℓ_θ/Δ<0.05 (K.3) | identical (B.4) | ✅ |
| trained/toy coset | `1.0020 ± 0.0495`, 20 cells (K.5) | identical (B.4) | ✅ |
| raw designed ratio | `1.257 ± 0.261` (K.6) | identical (B.4) | ✅ |
| designed latch | `||λ_flat|−1| ≤ 1.7e-14`, drift ≤4.9e-12 rad/200k, 30/30 (K.1) | identical (B.5) | ✅ |
| emergent latch failure | `1−|λ_coset| ≈ 1e-3` at γ≈0.05, **min ≈8e-5 over the grid**; designed ≤1.1e-15; ~12 orders (K.1) | `1.06e-3–2.96e-3` at γ≈0.048; designed ≤1.1e-15 — **grid minimum omitted** (C.2) | ⚠ compatible; V5 omits the more conservative figure (NICE N-6) |
| emergent capacity | `≈1–1.6 bits`, 2–3 minima (K.1) | identical (C.2) | ✅ |
| emergent V-curve | argmin `0.902±0.003`, slopes `−1.0020±0.0003` / `+1.116±0.011`, 3/3 (K.4) | identical (§3.1, C.3) | ✅ |
| designed V-curve | argmin, slopes `−1.006` / `+1.23…+1.27` (K.4) | identical (§3.1, B.6) | ✅ |
| T\* | `≈3e-3`, predicted 2.7–3.7e-3, barrier 2.3–3.6e-2 (K.4) | `≈3e-3`, predicted 2.72–3.66e-3, barrier 2.29–3.57e-2 (C.4) | ✅ same to rounding |
| matched-designed raw exponents | `−0.53,−0.60,−1.04` / `+0.78,+0.63,+0.55`; ℓ_θ/Δ up to 2.0 (K.6) | identical; ℓ_θ/Δ = 2.03 (C.5) | ✅ |
| **the unification's shape** | **"One damping-optimum curve evaluated at TWO values of μ, not two laws."** (K.4) | **"that one curve at THREE values of μ"** (§3.1) | ⛔ **DIVERGENT — MF-2** |
| **the 11-decade span** | `1.7e-12 → 7e-2`, **with the instrument note** distinguishing the Jacobian endpoint from the Hessian μ²≤2.4e-15 "so the spans quoted in this paper cannot be read as inconsistent" (K.4) | `1.7e-12 → 7e-2`, **note absent**, and μ²≈10⁻¹⁵ quoted in the same paragraph (§3.1) | ⛔ **DIVERGENT — MF-1** |
| vault / refrigerator / 107.77× | **not carried by V2** (0 hits for 107.77 / 13.28 / 110.25 / 1.26e-4) | §3.2 + App D | ✅ no conflict; V5 owns it (Q11/Add.8) |
| mass-independent floor 27.03 | 7 sites (§3, App F) | 1 site (§3.1) | ✅ consistent |
| erosion study | V2 App C (law + cure) | V5 App H | ⚠ **duplicated content across two submissions** — see M3 note below |

**M3/salami note (Head-facing, not a fix).** V5's §3.1/§3.2/App B/App C and V2's App K are the *same measurements* — V5 leads with them, V2 appendixes them — and both drafts also carry the erosion study. Q11 (Add.8) partitions the *claim* correctly, but neither paper may cite the other (C-8), so nothing in either submission discloses the overlap. If both go out in the same cycle to overlapping reviewer pools, this is the program's highest-exposure M3 configuration. Decision, not a defect.

### ⭐ Carry-item 2 — erosion wording vs cite-check Part 2 (Add.23). **DISCHARGED: V5 FAILS on both required items.**

| requirement (Add.23 / `v2-cite-check` §Ship-guidance) | V5 App H | verdict |
|---|---|---|
| claim (a) substrate cites = **Fischer–Igel 2010/2011 + Nijkamp 2020** | cites "Fischer & Igel 2011; Nijkamp et al. 2020" | ⚠ partial — **2010 missing** |
| ⛔ **arXiv:2503.21536 must NOT be cited as CD-induced symmetry breaking** (it attributes it to hierarchical feature learning) | cited exactly that way, **twice** (H(a) line 230; K.2 line 281) | ⛔ **FAIL — MF-10** |
| claim (c) must carry the **k-regime scope clause** vs Decelle 2021 / Agoritsas 2023 | no clause; "independent of chain length" stated flatly | ⛔ **FAIL — MF-11** |
| (b) may be stated as novel | stated as "novel to our knowledge, single-sourced" — correct, but with a stale pending-scout marker | ⚠ **MF-12** |
| Decelle 2021 / Agoritsas 2023 in the bibliography | no bibliography exists | ⛔ **FAIL — MF-6** |
| **V5's own bibliography state** (in scope per the task) | **absent entirely** (V2 has one at line 70) | ⛔ **FAIL — MF-6** |

---

## F. Compliance sweep (per-file, positive-controlled, zero-hit list printed)

**File:** `.claude/papers/v5-short/draft.md` — 296 lines, 10,914 words. **Instrument LIVE** (positive controls: `deletion` 25, `gamma` 142, `Blelloch` 11).

**Zero-hit list (25 patterns, all 0):** `13.9` · `≈14×` · `14\times` (as vault) · `is certified` / `we certify` / `certified removal of <ours>` · `we alone` · `CLU-former` · `0 of 5` · `CSF3` · `prior mismatch` · `P = 4` / `P=4` · `compositional family` · `residual protects` · `watch stayed green` · `null*` · `state-of-the-art` · `SOTA` · `best-in-class` · `benchmark win` · `we outperform` · `outperform` · `beats` · `wins` · `our fix-up cascade` · `deletion-compliant` · `0.272` · `Guo Def. 1` / `Def. 2`.

**Non-zero but COMPLIANT (inspected individually):**
| pattern | n | disposition |
|---|---|---|
| `certified` | 5 | ✅ all denial or literature-description: §3.3 "we do not claim certified (ε,δ) unlearning"; §4 + E.7 "*certified* removal in this literature is an (ε,δ) notion"; K.2 "not … about certified (ε,δ) unlearning". **N118 / CM-25(f) satisfied.** |
| `unlearning` | 7 | ✅ never applied to the CLU mechanism; §4/K.2/E.7/App-J all describe the literature or deny. **N99 naming hazard satisfied.** |
| `13.88` | 1 | ✅ labelled a refuted coupled-bath prediction (D.1) — but see **MF-8** (the formula printed beside it is wrong). |
| `novel` | 5 | ✅ hedged ("novel to our knowledge, single-sourced"); one explicit no-priority disclaimer in E.2. ⚠ **MF-12** (stale pending marker). |
| `guarantee` | 1 | ✅ "This is a **store-level** guarantee only…" — the scoped form. |
| `companion` | 2 | ⚠ one is a C-8 semantic hit — **MF-14**; the other (C.6) is benign. |
| `Anonymous` | 2 | ⚠ **MF-14** (F5 note not live). |

**Positive charter/matrix checks that PASS:**
- **Blelloch–Golovin attribution present at every exact-deletion site** (abstract; §1 contributions; §3.3 "Attribution, in the same breath as the theorems"; §4; E.2 ×2; K.2; App J #8 — 11 mentions, 7 attributional). **N118 satisfied.**
- **CM-25(f) verbatim** in §3.3 — checked word-for-word against the matrix row. ✅
- **CM-25(g) verbatim** (`R₅₀` 1.135→0.771 / baseline 1.146→0.752; effect size not correlation) + **N129** γ_read=0 oscillatory caveat. ✅
- **N108's required sentence** "the store stops answering before it stops leaking" — 3 sites (abstract, §3.3, E.5), unsoftened, with the no-distinguishability-per-se denial. ✅
- **N112** overflow scope clause in §3.3 and E.4, including the failed-prereg disclosure and the harness-artefact correction. ✅
- **N127** capacity qualifier removed with the waitlist, 0.29×–1.71× stated. ✅
- **N131 citation fence:** Guo **§3 Eq. (1)** at both sites; **no "Def. 1/2"**; Ginart **Def. A.5** used only as adopted *vocabulary* alongside an explicit *"No deletion-cost statement is made here"* and App J #9. ✅ (M1 correctly treated as unavailable.)
- **N163/N168 "we alone delete" ban** discharged by leading §3.3 with the flat-table trivial substitute. ✅ (⚠ under-citation of the competing work — SF-10.)
- **§0.13 lifecycle wording verbatim**, both riders present (demotion = re-exposure, never trash; trash keyed on `read_hits`, never depth), L4 UNEXERCISED / 0 refusals, **zero VALUE numbers**, value surface labelled a **declared not-run, never a null**, substrate labelled a mechanics instrument. **No C2W8 cell number present.** ✅
- **Score sentence** stated in §3.3 and repeated in App J #14. ✅
- **§A20.5 substrate-scope sentence** once, in the paper's voice, §1. ✅
- **Q5/Q9 scale-as-scope-choice line** in §5 with named future experiments and **no C3 number**. ✅
- **C-9:** 15 negatives in App J, each with its measurement and a future-work anchor. Best in the portfolio. ✅
- **C-2:** designed = "verification", learned = "evidence", labelled per section and in every figure caption. ✅
- **C-6:** the `fdt`+Newtonian fine print sits *beside* the T>0 claim in §3.2, not only in App G. ✅
- **C-7:** flag-provenance tables A.0–A.9 with commits, seeds, env, and the two honest commit-provenance caveats (A.6's cwd/PYTHONPATH note; A.8's dtype no-op). Exemplary. ✅
- **N149/N150 blast radius:** the tilt refutation-in-sign and `τ_max = Γ/2α` travel in §3.2's tail, C.7, K.3, App H's rider and App J #11/#12; the ε-notation collision is stated in §1 and K.1. **No unfenced ε/tilt/lifetime sentence found** (grep over all tilt/lifetime sentences). ✅
- **PALM anonymization-extends-to-code** note present in the header and the closing anonymization note; **no repository, institution or acknowledgement string anywhere** (grep: `github`, `gitlab`, `zenodo`, `huggingface`, institution names → 0). ✅ ⚠ The only artifact strings are internal harness paths inside A.5–A.9 (`.claude/scratch/...`, `chlu/core/placement.py`, `../CHLU-waitlist`) — **these must be neutralized before the supplement ships**: `chlu/` and worktree names are project-identifying. Filed as **SF-12** below.

- **SF-12 · Provenance tables leak project-identifying paths.** A.5 (`.claude/scratch/order-independent-placement/{pgcp.py,…}`), A.6/A.8 (`../CHLU-waitlist`), A.9 (`../CHLU-c2w10`), E.3 (`chlu/core/placement.py`), plus repo commit hashes. Under PALM's *"any supplementary or linked material as well, including code"* these are de-anonymizing once the anonymized snapshot is diffed. Rename to neutral paths in the appendix, keep the hashes (they are meaningless without the repo).

---

## G. Reviewer-hat attack pass (register composites against THIS draft)

- **G1 (unit test on a testbed built to satisfy the theory).** Partly answered by C-2 labelling and by the emergent arm — but **contribution (2) is fully exposed**: the deletion estate runs on a designed atom store, dim 3, capacity 8–64, **no learning anywhere**, and the paper concedes the algorithm is Blelloch–Golovin's. A reviewer will read §3.3 as "a correctness test of a 2007 data structure on an 8-slot toy". The defensible composition claim (packing certificate + decaying content + delete/decay commutation + energy function) is real and *is* stated — but it is stated in the appendix and in an attribution paragraph, while the p.1 contributions list sells "Deletion and lifetimes as structural properties of the store." **Recommendation:** rewrite contribution (2) as the *composition* claim, not the exactness claim.
- **G2 (which component buys what).** **Unanswered in V5, and this is the largest genuinely-missing control.** There is no minus-the-physics arm anywhere: no non-symplectic twin, no unconstrained damped update, nothing that shows the conformal-symplectic structure is load-bearing for the V-curve rather than a restatement of a 2×2 damped map. The paper's own §3 preamble hands the reviewer the objection ("so per normal mode memory reduces to one 2×2 matrix"). See ME-6.
- **G3 (toy scale).** Declared honestly in §5 and unfixable by writing. At a venue about long-term memory for AI systems the entire evidence base is dim 4 / hidden 64 / dim-3 store / capacity 8–64 / 384 synthetic instances / laptop CPU. This alone caps the ceiling at weak-accept.
- **G5 (certificate fine print).** **Best-handled item in the draft.** The deletion scope, the recency exclusion, the overflow failure, the `fdt`+Newtonian requirement, the γ_read=0 oscillation, the adversary-relativity, the σ_obs-is-our-choice admission — all sit next to their claims. The only inversion risk left is the **abstract** (MF-13).
- **G6 (foundational-paper falsifications).** C-1 honoured: no audit paragraph; J&P 2026 cited only for the primitive's introduction; no legacy number is load-bearing. ✅
- **M2/M3 (salami / de-anon optics).** See the lockstep table's closing note. Additional exposure: "the CLU … introduced as CHLU in Jawahar & Pierini (2026)" + "a **companion** theory note (Anonymous, 2026)" is a two-step de-anonymization for any reviewer who searches the first citation. C-1(d) sanctions the first; the second is MF-14.
- **NEW — A1: the headline law may be a linearization tautology.** The V-curve is the spectral-radius formula for a damped 2×2 map, measured *from the Jacobian*. The paper never states what could have falsified it. See SF-11 and ME-1.
- **NEW — A2: the vault is measured on a register the paper proves does not emerge.** 107.77× is the paper's most quotable number and it lives entirely on designed SO(2), 3 seeds. MF-13(a).
- **NEW — A3: "the price of exactness is negative" is a comparative with no number.** SF-2.
- **NEW — A4: the leakage story's tight control is missing.** MF-9.

---

## H. Missing-experiment list (for the Hub; quality-first posture — "this claim needs a run" is welcome)

| # | experiment | why a reviewer demands it | cost | class |
|---|---|---|---|---|
| **ME-1** | **Rollout-validated V-curve.** Direct T=0 rollout n₁/₂(γ) on the dense grid, designed + emergent, against the Jacobian instrument; report the ratio curve. | The one cross-check that exists disagrees by 19–43% (`v5-gate` §3.2) and is unpublished. Closes MF-3 with data instead of a caveat. | laptop, hours | **genuinely missing** |
| **ME-2** | **Minus-the-physics twin for the V-curve.** Identical dims/params, non-symplectic (or volume-broken) damped update; does a single-mode V-curve with argmin at 2εμ still appear? | The G2 control the program ran for V1/V2 (`minus-the-physics`) and never ran for the forgetting law. Decides whether the headline is a CLU property or a linear-map property. | laptop | **genuinely missing** |
| **ME-3** | **Vault on an emergent checkpoint above T\*.** γ_φ hole + D̂ estimator on `emergent150_s{42,43,44}` at T > 3e-3. | Converts the paper's most quotable number from designed-only to a generalizing claim, or produces a first-class negative. Directly answers MF-13(a). | laptop; harness exists (`v5-gate` R3) | **genuinely missing** |
| **ME-4** | **M1 — amortized per-delete cost vs the flat-datastore substitute** (as a function of n). | §5 and E.7 both name it as unrun; N131 blocks any Ginart-A.5 instantiation until it exists; it is the single most-asked question about an exact-deletion mechanism. | laptop | **genuinely missing** (already named) |
| **ME-5** | **Occupancy sweep for the hole/leak statistics + `R₅₀` under a placement distribution.** | E.5's own rider: *"an occupancy sweep is owed before any such number is generalized"*; and N119/N108-w26's |c| dependence (0.500 vs 0.886). Closes SF-3 and the "quote-the-curve" exposure on E.5. | laptop; harness exists | **genuinely missing** |
| **ME-6** | **Deletion at 10³-item stores** (and dim > 3). | §5 names it; it is the only route out of the "8-slot toy" reading of contribution (2). | laptop→small | **genuinely missing** (already named) |
| **ME-7** | **PALM topic-scope + agent/LLM long-term-memory literature scout.** | Charter Add.7/Add.8 Q7 records PALM's topic scope as *"a known gap"*; §4 has zero venue-native citations (SF-7). Not an experiment but a blocking spoke. | web scout | **genuinely missing** |
| **W-1** | **Wire the TTL-flag laundering control** into §3.3/E.5 (0.983 vs 1.000 exact; 0.559 vs 1.000 at σ=0.1). | MF-9. | — | **exists in `mia-decay-measurement` §item-4/§3(a) — wiring only** |
| **W-2** | **Wire the packing numbers** 61/64, per-offered 0.9531, ×1.05 → 64/64, vs refuse-and-relocate 43/64. | SF-2. | — | **exists in `placement-landing` §3 / CM-23(v) — wiring only** |
| **W-3** | **Wire the Jacobian-vs-Hessian instrument note** from V2 App K.4. | MF-1. | — | **exists in V2 draft — wiring only** |
| **W-4** | **Wire the k-regime scope clause + the executed-scout sentence** from `v2-cite-check` §Ship-guidance. | MF-11, MF-12. | — | **exists — wiring only** |

---

## I. The three sentences a hostile reviewer would quote

1. *"The paper's headline is 'one law across eleven orders of magnitude in μ²', but the low endpoint (1.7×10⁻¹²) is the numerical resolution floor of the authors' own ring-profile probe — two sentences earlier the same designed coset is quoted at μ²≈10⁻¹⁵ — and every point on the curve is read off a one-step Jacobian whose only reported rollout cross-check disagrees by 19–43%; so the eleven decades are one decade of measured emergent variation attached to ten decades of numerical zero."*
2. *"Figure 1 is offered as the headline and it does not contain the headline: panel (a) plots three emergent seeds spanning less than half a decade of μ², the designed arm appears only as a legend entry with no line, half the figure is given to a designed-only negative, and the panel title cites 'Cor-13' — an object that appears nowhere in this paper."*
3. *"Contribution 2 is a bit-exactness verification of Blelloch & Golovin's 2007 table — which the authors correctly concede is not theirs — on a three-dimensional store of eight slots with no learning anywhere; the abstract nonetheless states 'store-level deletion is exact' without the three conditions, the capacity scope or the recency exclusion that the body carefully attaches to it; and the tighter of the two laundering controls the authors ran — a boolean TTL flag, indistinguishable from their physical decay to within 0.017 AUC against an exact adversary — is in their supplementary reports and not in the paper."*

---

## Open questions / follow-ups / risks

1. **The 4-pp budget is genuinely tight against the MUST-FIX set.** Recommend SF-8 (§3.4 → App F) be taken *first*; it is the cheapest 0.19 pp and it also fixes SF-9 (three contributions in a short).
2. **PALM topic-scope remains unscouted** (Add.7/Add.8 Q7, Head-deferred). Every fit judgement in this report about the venue's literature (SF-7) is my inference from the venue's name and the Add.4 banked description, not from a verified call.
3. **The F5 note's arXiv status is the paper's only hard external dependency** (MF-14) and it has been the standing #1 blocker since w13.
4. **I did not rebuild the PDF** (no TeX run this session); the page counts quoted are from the committed `draft.log`/`maincount.log` of the v0.2 build, `Output written on … (4 pages)` / `(17 pages)`.
5. **Not re-derived:** every number was checked against `.claude/outputs/*` source reports, not against raw JSON. Two numbers I could not locate a primary for and would re-verify before freeze: §3.1's designed floor "27.03 steps at γ=0.05" (consistent with V2's seven sites and with 2ln2/(−ln0.95) = 27.03 by hand ✅) and E.6's "measured log-log slope −0.5264" (matches N129 ✅). Both check out.

**Git footprint:** none — read-only pass; the only file created is this report.

---

## Proposed handover updates (for the Hub)

1. **V5 v0.2 refereed: BORDERLINE → weak-accept after 14 MUST-FIX / 12 SHOULD-FIX / 6 NICE.** Report at `.claude/outputs/v5-referee-v02.md`. No MUST-FIX requires a new experiment; four are pure wiring from existing outputs (W-1…W-4).
2. **Add.24 routing item 4 is now load-bearing on a draft, not just a source doc:** `venue-follow-up.md` §3 claim-(a) basis is uncorrected **and V5 App H(a)/K.2 inherited the error**. Owner needed at this review (MF-10).
3. **Carry-item 1 (Add.21) discharged with a lockstep table: 16 rows identical, 2 divergent** — the "two vs three values of μ" wording (MF-2) and the missing Jacobian-vs-Hessian instrument note (MF-1). Both fixes are V5-side; the Hub should confirm V2's K.4 note survives `v2-revision-7`.
4. **Carry-item 2 (Add.23) discharged: V5 FAILS both required items** — the 2503.21536-as-CD mis-citation (×2) and the missing k-regime scope clause — plus a stale "pending scout" marker and no bibliography.
5. **New candidate registry entries** (Hub to rule): (a) the Jacobian-vs-rollout n₁/₂ discrepancy (19–43%) as an *instrument* caveat travelling with every V-curve quote — the N51 class; (b) the designed-arm "1.7e-12" as an instrument floor, never a spectral mass — a never-quote candidate, since the "eleven decades" phrase now appears in two drafts.
6. **Missing-experiment candidates for tasking:** ME-1 (rollout-validated V-curve) and ME-3 (vault on an emergent checkpoint) are the two that would move V5 from weak-accept toward accept at laptop cost; ME-7 (PALM scope + agent-memory lit scout) is blocking for §4 regardless of the experimental programme.
7. **Anonymization action before any supplement ships:** SF-12 — `chlu/`, `../CHLU-*` worktree names and `.claude/scratch/*` paths in App A are project-identifying under PALM's code-inclusive rule.

## Flags

- ⚠ **MF-8 is an arithmetic error in a shipped appendix** (`γ_eff/(2−γ_eff) = 13.88×` → the expression equals 0.356). It is a transcription defect only — the source `v5-gate` §2.1 is correct and every downstream number (7.942, 8.11 ± 0.37, 110.25, 107.77) is consistent. But it sits inside the discriminator that makes the vault credible.
- ⚠ **The compiled `draft.pdf` currently has ~39 wrong section cross-references** because `\section{Results}` numbers as §2 while the text hard-codes §3.x. Anyone quoting "V5 §3.3" from the PDF is quoting *Related work*.
- ⚠ **M3 exposure is at its program maximum**: V5 and V2 carry the same `t-lever`/`v5-gate` measurements *and* the same erosion study, with C-8 forbidding either from disclosing the other. Head decision, flagged not resolved.
- ✅ **The never-quote sweep is clean** — 25 forbidden patterns, 0 hits, instrument positively controlled. The seven non-zero patterns were each inspected and are compliant. The v5-rebuild spoke's own sweep claim is confirmed independently.
