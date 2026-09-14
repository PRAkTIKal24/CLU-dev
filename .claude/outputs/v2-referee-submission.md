# v2-referee-submission — paper-referee report
**Task + acceptance criterion:** final adversarial pass on `condensed_paper.pdf` (NeurReps 2026 Extended Abstract track) before submission; answer the main-text-standalone question, measure the page extent, judge the new derivation appendix, sweep defects, answer the desk-reject question, and give one unhedged submit/don't-submit call.
**Status:** done.
**DIAL DECLARATION (echoed):** none — read-and-report. **Zero writes outside this file** (verified: no `git status` change, no edits to `.claude/scratch/v2-final-build/**` or anywhere else; my only artifacts were `/tmp/v2_layout.txt`, `/tmp/v2.stext.xml`, `/tmp/pg{3,4,10}.png`, outside the repo).

---

## 0. Independence bar — stated at the head, as required

I read **none** of: `v2-referee-final`, `v2-referee-v07`, `pj-referee-v2*`, `pj-fidelity-v2*`, `v2-condensation-equivalence`, `v2-derivation-appendix`, `v2-cite-pass`, `v2-bib-doi-list`, or any `BUILD-NOTE*`. Every finding below was derived from the PDF, the `.tex`, `refs.bib`, the figure PNGs, `AGENT_PROTOCOL.md`, the Positioning Charter (C-1…C-10), `critique_register.md`, and **primary experiment reports only** (`v2-prefreeze-baselines.md`, `anchor-robustness.md`, `sleep-erosion-study.md`, `shorts-evidence-inventory.md`, `handover_context.md`, `advisor-head-shorts-charter.md` — the last grepped narrowly for the banked venue page-limit fact).

⚠ **One disclosure.** A `grep -rn "116\b" .claude/outputs/*.md` sweep for the erosion-onset epoch returned **one line of `pj-fidelity-v2-r2.md`** in its output. I did not open the file and no finding below derives from it (my Appendix-K and negatives findings were already written from the PDF before that grep ran). Flagging it so the Hub can weigh the independence claim honestly.

**Order of work, as mandated:** main text (pp. 1–5, abstract → §5) read **alone and in full, twice, before any appendix was opened**. Only then pp. 10–25.

---

## 1. VERDICTS

### (a) Content — **BORDERLINE, leaning WEAK-ACCEPT *for this track specifically*.**
The Extended Abstract track's stated purpose is verbatim *"Early-stage results, negative findings, opinion pieces, or novel datasets."* This paper is an early-stage result **plus** an unusually forthright set of negative findings (the register is designed-not-learned; emergent potentials give 1–1.6 bits on a washboard; the retention advantage inverts under compute normalization). Against that purpose it is a good fit and a rare instance of a submission whose limitations section is load-bearing rather than decorative. Appendix B is a **correct** derivation (I checked the matrix, det/tr, both Δ roots, the overdamped expansion, the floor and the AR(1) coset-diffusion coefficient — all six reproduce). What holds it at borderline is that the *headline* quantitative result is a Hellmann–Feynman identity evaluated on a potential whose flat direction is architecturally imposed, the one comparative performance number is a 3-seed median at T=0 against models that are 14–15× cheaper per step, and the word **"Trained"** in the title is carrying more weight than the experiments give it. Against a general workshop bar (not this track's) I would score it weak-reject.

### (b) The artifact as shipped — **CLEAN EXCEPT FOR ONE THING, AND THAT THING IS LIVE.**
Anonymity, class option, metadata, figure metadata, references and bibliography are all correct. **The page count is not.**

### (c) ⛔ THE DESK-REJECT QUESTION, ANSWERED EXPLICITLY
**Yes, there is one live desk-reject vector, and it is the page limit. Nothing else in the artifact would stop it before review.**

- **Main-text extent, measured (not estimated):** the body text block is `y ∈ [93.6, 705.6] pt` = **612.0 pt per page**. The `References` heading sits at `y = 306.7 pt` on **page 5**. Main text therefore occupies **4 + (306.7 − 93.6)/612.0 = 4.348 → 4.3 pages** (to the nearest tenth; 4.35 to two).
- **Bibliography begins:** page 5, 34.8 % down the page. It runs to page 9. **Appendix A begins on page 10.** Total 25 pp.
- **Against the banked venue fact** (Head-verified from the venue site 2026-08-18: *"4 pages main text, with references and supplementary material excluded from the limit and of any length"*): **the paper is over by 0.35 pp ≈ 213 pt ≈ 15 body lines.** The Head's working target of ~4.5 pp is **not the venue's number**; on the banked fact the paper is over, and there is no 4.5-pp reading of "up to 4 pages."
- **What a chair does, at both readings.** (i) *Strict / mechanical* — the check a PMLR-style chair actually runs is "does the main text end on or before page N": here it does not, and a submission whose §5 and its Discussion are printed on page 5 reads as a 5-page main text. Desk-rejected without review. (ii) *Lenient / eyeball* — a chair who sees that 65 % of page 5 is bibliography may wave it through as "essentially 4 pages." **Both readings exist in the wild; the strict one is the one that requires no judgement and therefore the one that gets applied at scale.** I would put this at a materially non-trivial probability of a desk reject, and it is the only such probability in the artifact.
- ⛔ **I was instructed not to recommend content cuts, and I am not.** I am reporting one *mechanical* measurement the Head should have: **Figure 2's float on page 4 occupies `y ∈ [268.8, 491.1]` = 222 pt including float skips. The overrun is 213 pt.** Figure 2 (`fig_lifetime_headtohead.png` at `0.33\linewidth`) is **byte-identical to Figure 4 in Appendix F**, which prints the same file at full `\linewidth`. Removing the main-text instance therefore recovers ≈222 pt against a 213 pt overrun **with zero loss of information from the document**, and lands the main text at the foot of page 4 with ~9 pt (under one line) of margin. That is tighter than I would like — LaTeX float placement can move it by a line — but it is the only lever in the paper that costs nothing. It is the Head's call, not mine.

### (d) ⭐ THE SINGLE JUDGEMENT
# **SUBMIT AFTER THE MUSTS.**
The science is honest and the track fit is genuinely good; the defects are, with two exceptions, sub-ten-minute edits whose absence is disproportionately expensive. Of the nine MUSTs, **seven are typo-class or one-`\ref`-class**, one is an arithmetic honesty fix in the abstract, and one is the page number, which is a Head decision rather than an edit. Do not submit this build unedited: it currently contains an **empty appendix section**, a **derivation appendix that nothing in the paper points to**, a **mathematically false sentence** in the Discussion ("non-abelian tori") at a venue whose reviewers are symmetry specialists, a **misspelling on the last line of main text**, and an **abstract multiplier whose denominator does not exist in the paper**. Any one of those alone is survivable; together they read as unproofread, and reviewer tone is set by that impression in the first two pages. Fix the MUSTs, resolve the page number, submit.

---

## 2. ⭐⭐ THE CENTRAL QUESTION: DOES THE MAIN TEXT STAND ALONE?

**Read alone, before any appendix was opened. Answer: NO — decisively, and in more ways than the condensation would have predicted.**

A NeurReps EA reviewer is not required to open an appendix. A reviewer who does not, receives a paper that (i) never states what task was trained, (ii) never states the symmetry group, (iii) never states the integrator step size, (iv) states an absolute claim ("infinite half-life") whose invalidating condition is appendix-only, (v) carries **no** experimental configuration at all (C-7: zero flag provenance in main text), and (vi) is told by the contributions list that two of the paper's three contributions are located in appendices.

**Claims stated in main text and qualified ONLY in an appendix — the exhaustive list.**

| # | Main-text sentence (quoted) | Where the qualification actually lives | Severity |
|---|---|---|---|
| **1** | §3: *"**The Latch:** (µ = 0, symmetry-protected). Exhibits frozen displacement … and **an infinite half-life**."* | App. E: *"The infinite half-life of the latch is a deterministic, **T = 0** property. At T > 0, the coset diffuses with coefficient D_θ = εT(2−γ)/(2F²γ), yielding a finite computable lifetime."* | ⛔ **The worst one.** A main-text absolute that is false in the paper's own model at any T > 0, with the correction in an appendix. This is charter **C-6** ("the fine print must never be invertible into the review") in textbook form. |
| **2** | Abstract: *"holding at **≈ 20×** the temporal horizon where the objective naturally degrades it."* | App. F.2 / App. K — and it is not there either (see MUST-3). | ⛔ Abstract-level unbacked multiplier. |
| **3** | §4: *"the generically-trained CLU holds for **≈ 263 map-steps**"* | Fig. 6 (App. I) caption: *"emergent measurements represent the median across … **3** seeds"*. Source (`v2-prefreeze-baselines.md` L58): **263 (range 184–436, n = 3)**. | ⛔ **C-5.** The paper's only comparative performance number, quoted as a point estimate with no n and no spread. |
| **4** | §4: *"Crucially, the designed unit **never drifts across all 5 seeds** as expected."* | Fig. 6 caption explicitly disclaims it: *"the designed curve is a single representative checkpoint—**the 5/5-seed latch statement is Sec. F.1's, not this figure's**"* — and F.1 contains no such statement (see SHOULD-1). Also T = 0 deterministic, per the same caption. | ⛔ A main-text claim whose own figure denies owning it, pointing at a section that does not contain it. |
| **5** | §4: *"the compute efficiency of the CLU is **much worse** (see App. J)"* | App. J: 14–15× FLOPs/step, 6.2×/3.1× wall, and the paper's own normalization table at **54.8× / 70.7×**. | Qualification of *kind* present, of *degree* absent. Charter-compliant in letter, not in spirit. |
| **6** | Contribution 1: *"demonstrate its **survival under training-time corrections**"*; Contribution 3: *"**Operational boundaries and negative results** (Sec. F.1)"* | App. F.2 and App. F.1 respectively; §4 carries **one sentence** pointing there. | Two of three contributions have **no main-text evidence**. |
| **7** | Everything: dim 4, hidden 64, dt = 0.05, unit circle × 256, 150 epochs, designed 5 seeds / emergent 3 seeds, T = 0. | App. E, in one paragraph. Main text carries "dim 4, 5 seeds, γ = 0.05" **only inside Figure 1's caption**, and "a local Apple M1 chip" in prose. | ⛔ **C-7 violated in main text.** ε is never given a value in main text at all, so a main-text reader cannot evaluate `h = εµ ≈ γ/2` — the paper's own crossover criterion. |
| **8** | §3–§4 assert three closed forms as fact. | App. B derives them. **Nothing in the paper references App. B** (see MUST-5). | ⛔ The derivation is unreachable. |
| **9** | The training task itself, and **G = SO(2)**. | App. E ("a unit circle at 256 points"); §5 mentions SO(2) only in a future-work clause. | A main-text-only reader never learns what was trained on what, or which group. |

**What the condensation gained and lost, in a reviewer's eyes.** Gained: a tight, readable, well-signposted 4.3 pages with a genuine "however/therefore" rhythm and no physics jargon in the main text (charter **C-3 is satisfied** — a physics reviewer cannot dismiss the main text as a damped-oscillator problem set, because the QFT vocabulary is entirely quarantined in App. C/M). Lost: **every scale qualifier, every configuration number, and both T = 0 conditions**. The condensation cut the fine print and kept the claims. That is exactly the failure mode the task named, and it is real here.

---

## 3. ⭐ THE NEW DERIVATION APPENDIX (App. B) — judged as a reviewer would

**Verdict: the best-executed part of the paper — and orphaned.**

- **Assumptions stated?** Yes, and precisely, in the first sentence: *"All closed forms quoted in Sec. 3 follow from the one-step map **restricted to a single mass-whitened normal mode of transverse curvature µ² at a critical point of V_θ**."* That single clause carries the harmonic/linearisation assumption, the mode-decoupling assumption and the critical-point restriction. Honest.
- **Followable?** Yes. Nine steps, each one line, no gaps. It typesets cleanly (I rendered p. 10 to check eq. (3), which `pdftotext` garbles — the rendered version is fine).
- **Does it support what the main text asserts?** **Yes, and I verified it rather than taking it on faith.** `det A = 1−γ` ✓; `tr A = (2−γ)(1−h²/2)` ✓; both roots of Δ = 0 via `(1∓√(1−γ))² = (2−γ)∓2√(1−γ)` ✓, giving `h* = γ/2 + O(γ²)` and `h_f = 2 − O(γ²)` ✓; the overdamped expansion `−γδ + (2−γ)h²/2 = 0 ⇒ n₁/₂ ≈ 2γln2/[(2−γ)(εµ)²]` ✓; the floor `|λ±|² = det A = 1−γ ⇒ n₁/₂ = 2ln2/(−ln(1−γ)) = 27.03` at γ = 0.05 ✓ (matches §4.1 exactly); the AR(1) coset-diffusion sum `1 + 2(1−γ)/γ = (2−γ)/γ ⇒ D_θ = εT(2−γ)/(2F²γ)` ✓. **All six reproduce.** Nothing in App. B is asserted-rather-than-shown.
- **⛔ The defect:** `\label{app:derivation}` is defined and **`\ref{app:derivation}` appears zero times in the source.** §3 instead sends the reader to *"App. C"* for *"further map definition"* — which is the coset/curvature-instantiation appendix, not the derivation. **The one thing added since the last review is invisible from the main text.** So is **App. H** (`app:loan`, the physics-prior Pareto — the paper's entire answer to "which component buys what") and **App. I** (`app:retention`, which holds the figure backing the 263-vs-69 claim). Three orphaned appendices; three missing `\ref`s.

**Where the appendix is asserted rather than shown (App. H, not App. B):** *"This penalty is **proven mathematically** to be contraction-forbidden by the **volume conservation axiom**."* No proof appears anywhere in the paper, and no "volume conservation axiom" is ever stated. That sentence is doing the work App. B does honestly, dishonestly.

---

## 4. MUST-FIX — each anchored to a quoted sentence or named object, with the cost of NOT fixing

> **Seven of these nine are ≤ 10-minute edits.** I have ordered them by (damage × cheapness).

**MUST-1 — The page number. `4.3 pp` main text against a Head-verified 4-pp limit.**
*Object:* `References` heading at `y = 306.7` on page 5.
*Cost of not fixing:* the single mechanical check a workshop chair performs, run before any reviewer opens the file. A desk reject costs the entire submission cycle and yields **zero reviewer feedback** — the worst possible return on the work already banked. Every other defect in this report costs at most a score point.
*Note:* the mechanical fact in §1(c) — Figure 2's float is 222 pt, the overrun is 213 pt, and Figure 2 is byte-identical to Figure 4 in App. F — is offered as a measurement, not a recommendation.

**MUST-2 — §3: *"Exhibits frozen displacement q∞ = q₀ + εp₀/(Mγ) and an infinite half-life."***
*Attack:* the paper's only unconditional claim is false in the paper's own model at T > 0, and the correction is buried in App. E. A reviewer who finds App. E writes "the main text overclaims and the appendix quietly retracts it" — the most damaging sentence a review can contain, because it licenses distrust of every other number.
*Evidence:* App. E, *"The infinite half-life of the latch is a deterministic, T = 0 property."*
*Fix cost:* three words — *"…and an infinite half-life **at T = 0**."*
*Charter:* **C-6**, direct hit.

**MUST-3 — Abstract: *"holding at ≈ 20× the temporal horizon where the objective naturally degrades it."***
*Attack:* the multiplier has **no denominator anywhere in the paper**, and the only denominator a reader can extract gives a different answer. App. F.2 states the anchored run is **3000 epochs** and that *"At 1000 epochs, the designed vacuum inverts in 8/8 runs"* → **3×, not 20×**. App. K gives *"inversion at epoch 116/442/959 by sleep frequency"* → 25.9× / 6.8× / 3.1×. The **20× is 3000/150**, i.e. 20× the *training baseline* (App. E: "150 epochs"), which is not "the horizon where the objective degrades it."
*Evidence:* primary source `sleep-erosion-study.md` L43/48 (116/442/959 at f = 1/5/20); `anchor-robustness.md` (3000 epochs, λ = 100, 5/5 seeds) — **neither states any "20×" ratio.** The multiplier is a drafting-time reformulation.
*Cost of not fixing:* a reviewer who does the division from §F.2 gets 3 and concludes the **abstract inflates by ~7×**. Abstract-level arithmetic that a reviewer can falsify from the paper's own appendix is a reject-grade finding regardless of track.
*Fix cost:* one clause — state the denominator, e.g. *"3000 anchored epochs = 20× the 150-epoch training budget and ≈26× the earliest measured inversion epoch (116)"* — or drop the multiplier and say "3000 epochs."

**MUST-4 — §5: *"generalizing the CRR beyond abelian SO(2) to **non-abelian tori** constructions."***
*Attack:* **a torus is abelian by construction.** "Non-abelian tori" is a contradiction in terms, in the Discussion, at a venue called *Symmetry and Geometry in Neural Representations*. This is the one sentence in the paper whose readership is guaranteed to catch it instantly.
*Evidence:* the program's own register (G7a) correctly lists **T² (abelian coset)** and **SO(3)→SO(2)** as two *separate* targets; the draft has merged them into a false object.
*Cost of not fixing:* it costs the reviewer's assumption of mathematical care, in the last paragraph they read, right before they score. Cheapest possible credibility loss.
*Fix cost:* two words — *"…beyond SO(2) to higher-rank tori Tⁿ and to non-abelian groups such as SO(3)."*

**MUST-5 — Appendices B, H and I are never cross-referenced. `\ref{app:derivation}`, `\ref{app:loan}`, `\ref{app:retention}` each appear 0 times.**
*Attack:* the new derivation appendix — the pass's headline addition — cannot be found from the main text; §3 points at App. C instead. App. I holds Figure 6, the figure that backs §4's 263-vs-5.6/56/69 comparison, and §4 points only at App. J. App. H is the paper's whole G2 answer and is reachable from nowhere.
*Cost of not fixing:* a reviewer who wants the derivation of the three bands does not find it and writes "the closed forms are asserted"; a reviewer who wants the baseline figure does not find it and writes "no figure for the head-to-head." **The paper is judged as not containing work it does contain.**
*Fix cost:* three `\ref`s.

**MUST-6 — Appendix I has no body. `\section{The autonomous-retention head-to-head against learned baselines}` is followed immediately by a `figure` environment and then `\section{Per-step compute requirements}`.**
*Attack:* an appendix section consisting of a heading and a float, with zero prose, reads as a document that was cut and not re-read. It sits directly above the compute appendix that a reviewer *will* read, because it is where the honest tail lives.
*Cost of not fixing:* it converts "condensed" into "truncated" in the reviewer's mental model, and it is adjacent to the material the paper most needs believed.
*Fix cost:* one or two sentences, or fold Fig. 6 into App. J. (App. G, whose entire body is *"See Fig. 5."*, is the same species one grade less severe — see NICE.)

**MUST-7 — Figure 2 is unreadable, and both main-text captions advertise the page problem.**
*Object:* `\includegraphics[width=0.33\linewidth]{figs/fig_lifetime_headtohead.png}` — a 2044×1118 PNG rendered ≈143 pt wide. I rendered page 4 at 110 dpi: the axis labels, tick labels and the three-entry legend are at or below legibility at print size.
*Attack:* this is the **only figure for Contribution 2** ("Evidence on a learned system"). A reviewer cannot read the paper's evidence figure. Worse, both Figure 1 and Figure 2 carry the clause **"(Figure downsized for space constraints, see App. F)"** — the paper is *telling the reviewer it is over the page limit*, on pages 3 and 4, in a submission that is over the page limit. And Figures 3 and 4 in App. F carry the **same clause verbatim while being in App. F**, pointing at themselves.
*Cost of not fixing:* an illegible headline-adjacent figure is scored as a missing figure; the "downsized for space" clause is a written confession that primes the chair for the page check.
*Fix cost:* delete the clause from all four captions (trivial); the sizing is entangled with MUST-1 and resolves with it.

**MUST-8 — Figure 1's in-image title reads "GMOR spectral-mass law: exact on the ***learned*** vacuum", while the caption, §4.1 and the entire honesty architecture say ***designed***.**
*Object:* rendered panel title, `figs/fig1_gmor.png`, right panel; caption says *"Trained **designed** checkpoints"*; App. F.1 says *"the continuous coset register itself is strictly a **designed** feature, **not learned**."*
*Attack:* designed-vs-learned is the one distinction this paper's credibility is built on (charter **C-2**), and the main-text figure contradicts it in pixels. A reviewer who reads panels before prose — most do — starts from the overclaim. The same panel also uses **"GMOR"**, an acronym never expanded in main text.
*Cost of not fixing:* it hands a hostile reviewer a screenshot. "The figure says the law is exact on the *learned* vacuum; §F.1 says the register is not learned at all."
*Fix cost:* regenerating the PNG is the right fix; hours before submission, a caption clause naming the discrepancy ("panel titles abbreviate 'trained checkpoint'; the vacuum is architecturally designed — see App. F.1") is a defensible stopgap.

**MUST-9 — Proofreading defects on main-text pages 2 and 5.**
- p. 5, last line of main text: *"when the algorithm reaches desired **maturiity**."* (also a comma splice: *"however, being an early-stage idea we defer…"*).
- p. 2, second sentence of the paper's positioning: *"…provides a closed-form analysis of latent retention on a trained recurrent model **Jawahar and Pierini (2026)**."* — `~\citet` used where `\citep` is required; the sentence is ungrammatical as printed.
- p. 2, §3: *"As in prior works like **the CHLU**, the Hamiltonian dynamics only define how the latent state evolves"* — **uncited**, and "CHLU" is never expanded or introduced anywhere in the main text. (App. A cites it properly.)
*Cost of not fixing:* the task's own weighting is right — *"a typo on page 2 costs more than three in an appendix."* These are on the two pages that set reviewer tone, and one of them is the paper's positioning sentence. Free to fix.

---

## 5. SHOULD-FIX

**SF-1 — Fig. 6 caption: *"the 5/5-seed latch statement is Sec. F.1's, not this figure's."*** Sec. F.1 is *"Where the CRR does not extend"* and contains no 5/5-seed latch statement (it is about the emergent MLP arm, 3 seeds). The cross-reference is wrong, and it is the pointer that main-text claim #4 depends on. Also: appendix subsections are cited as "Sec. F.1" / "Sec. F.2" throughout while sibling appendices are cited as "App." — pick one.

**SF-2 — Seed-count collision inside one paragraph.** App. F.2: *"**Over 5 seeds**, the anchor maintains r\* = 0.911 ± 0.016"* … *"the retention laws remain entirely intact (Appendix G)."* Figure 5 (App. G): *"anchored λ = 100, **3 seeds**."* Both are true of different measurements (the anchor sweep is 5 seeds — `anchor-robustness.md` seeds {42…46}; the anchored-CRR re-measurement is 3, per App. M *"3 anchored seeds at 3000 epochs"*), but the reader meets "5" and then "3" two lines apart with no explanation.

**SF-3 — The anchor's price is euphemised.** App. F.2: *"trading off **weaker noise rejection and a higher wake MSE**."* Source (`anchor-robustness.md` §1): *"λ = 100 is bulletproof across 5/5 seeds … at the cost of weaker noise rejection (gap ≈ 0.60) and **~35× higher wake MSE**."* **35× is not "a higher wake MSE."** A reviewer who learns the number elsewhere reads the paper as having hidden it; a reviewer who reads "35×" in the paper reads it as candour. Charter **C-6**. The same paragraph also drops the measured envelope (λ ≈ 10 gives the best rejection but 1/5 seeds still collapses) — a **C-9** negative.

**SF-4 — App. E's finite-temperature "verification" quotes the wrong number.** *"…verified on designed checkpoints where increasing friction from γ = 0.05 to 0.2 lengthens memory by **3.77 ± 0.23×**."* The paper's own closed form `D_θ = εT(2−γ)/(2F²γ)` predicts lifetime ∝ γ/(2−γ), i.e. **(0.2/1.8)/(0.05/1.95) = 4.33×**. The quoted measurement is **13 % low, ≈2.4σ**, and the paper states no prediction against which to read it — so a physics-literate reviewer does the division in the margin and finds an unflagged discrepancy in a sentence containing the word "verified." The primary source (`t-lever-forgetting`, via `handover_context.md`) actually carries the *right* number: **"D_θ law verified to 1.0068 ± 0.0219 over 25 cells."** Quote that as the verification and the 3.77× as its consequence.

**SF-5 — App. H asserts a proof it does not give.** *"This penalty is **proven mathematically** to be contraction-forbidden by the **volume conservation axiom**, as the broken-volume baseline precisely **recovers ≈ 2.4× of the performance gap** purely by leaking volume."* No proof; no "axiom" defined; and "recovers 2.4× of a gap" is not a well-formed quantity — the broken-volume model's MSE is not given in that paragraph, so the claim is uncheckable. Contrast with App. B, which shows its work.

**SF-6 — App. J's "Normalization Standard" is undefined, and two defensible numbers exist.** The table reports **FLOP-normalized 54.8× / 70.7×**. Per *retained step* the figure is `14.39 / 3.81 = 3.8×`; the 54.8× is `14.39 × 3.81`, i.e. **total FLOPs burned over the retention window**, which charges the CLU for lasting longer. Both are computable from the table; only one is labelled. (The error is self-penalising, so this is a clarity problem, not an overclaim — but "your normalization multiplies where it should divide" is a bad question to be asked live.) Per-step ratios all check out against the table (6.17×, 3.11×, 14.4×, 15.1× ✓), as does the raw 263/69 = 3.8× and 263/56 = 4.7×.

**SF-7 — §4.1's headline number is unfalsifiable as written.** *"a curvature ratio µ²_meas/µ²_pred = 1.000000 ± 5 × 10⁻¹²."* **µ²_pred is never defined in the main text.** Figure 1's right-panel y-axis reveals it: `µ²_meas / [δΣ/(M F²)]` — i.e. the GMOR/Hellmann–Feynman identity with **Σ and F² measured on the same checkpoint**. The paper's own caption calls it *"the curvature **identity**."* An identity holding to 1e-12 is a statement about float64 and autodiff, not about memory. §4.1 partly defuses this (*"exact verification of the theory rather than emergent discoveries"* — **C-2 compliant, and credit where due**) and App. M is explicit (*"a supporting result, not one of this paper's claims"*), but the main text still **leads** with it. See hostile quote #1.

**SF-8 — §4.2 concedes more than it needs to, and claims more than it shows.** Contribution 2 is *"Evidence on a learned system."* What is shown is that a **first-order** estimator misses **second-order** structure above an exceptional point — which is true by construction and which the estimator's author never claimed. App. D already concedes *"We do not claim novelty over these qualitative lifetime predictions or existence proofs."* State in §4.2 that Mo's estimator makes no claim above h\*, so "fails predictably" reads as containment rather than as a defeated rival. Also **charter C-3** asks the Mo head-to-head to *lead* the results; it is §4.2, behind the verification result. Ordering is a real reviewer lever here (see hostile quote #1) but re-ordering hours before submission is not worth the risk — flagging for the record, not for action.

**SF-9 — §4.2 states one number twice as if it were two findings.** *"mispredicting by ≈ **3.2×** at the deepest trained-model breaking magnitude (δ = 4) … and declining to **0.309** ± 0.012 deep in the underdamped regime."* `1/0.309 = 3.24`. If these are the same measurement, say so; if not, distinguish them.

**SF-10 — Undefined symbols and terms.** Main text: **M** in the latch formula (`εp₀/(Mγ)`) vs **m** two lines earlier (`µ² = k/m`); **k** and **m** themselves; **ε** never given a value; *"**Kinetic isotropy** acts as the price for an equivariant write current"* in §5 — the concept appears nowhere else in the main text. Appendices: **r\*** (order parameter, used throughout F.2/H/K, never defined), **Γ/2α = 4.0** (App. K), **ℓ_θ/Δ** (App. K, with a reading rule attached to it), **M_ch**, **µ²_rad**, **f** (App. M.5).

**SF-11 — App. M.5 is a truncated stub.** *"**The expansion variable, and where it stops being small.** With µ²_LO ≡ δΣ(0)/F²(0), the theory predicts a relative leading-order error x ≡ δ/(M_ch µ²_rad f); this is the x of Figure 7(d)."* — and the paper ends. **The section's title promises the answer its body never gives**, on the last page. Either answer it in one sentence or retitle. (Borderline MUST; demoted only because the task weights appendices below main text.)

**SF-12 — App. K contains a paragraph that belongs to App. M.** *"**This appendix** is tree-level and classical: no loops, chiral logarithms, running low-energy constants or anomalies … so **GMOR** there should be tested as δ → δ + δ_eff."* This sits inside *"Documented negative results and formal limitations."* GMOR is App. M's subject. Reads as a mis-paste.

**SF-13 — A cross-table µ² collision that the disclaimer does not cover.** App. H's non-comparability note is explicitly scoped to MSE (*"The absolute MSE scale and its associated twin (0.0047) are not mathematically cross-comparable…"*) — good, and clearly the fix for a previously-found defect. But the same appendix reports the CLU's flat direction as **µ² = 0.008** in prose and **5.2 × 10⁻³** in the table, and a **Latch Drift of 0.778** for a "CLU" row while §4/App. F say the designed latch freezes to `≤ 1.2 × 10⁻¹⁵` rad. Different arms, but adjacent on one page and unlabelled. Extend the disclaimer's scope.

**SF-14 — Two citation styles for one work.** Main text and Figs. 2/4: **"Mo (2026)"**. App. D and Fig. 6: **"a recent preprint (arXiv:2605.03338)"** and *"the S¹ protocol (arXiv:2605.03338)"*. `refs.bib` confirms `mo_symmetry-protected_2026` **is** arXiv:2605.03338. Bare arXiv IDs in running text where a `\citep` exists reads as a late patch.

**SF-15 — §5 generalizes without its scale qualifier (C-5).** *"**A trained memory network** is best interpreted as a composite of exact latches … and tunable registers."* Scope-free, on the evidence of one architecture at latent dimension 4 on S¹ with ≤5 seeds. Contrast with §F.1's *"On **this architecture class**, flat directions must be designed…"*, which is correctly scoped. The abstract is likewise scope-free throughout.

**SF-16 — WIRING NOTE (exists in outputs, not cited): the LSTM fragility result.** `v2-prefreeze-baselines.md` L11: *"their memory is **fragile**: a 0.1 hidden-state perturbation **collapses LSTM retention 69 → 2 steps**."* This is **absent from the paper entirely**, and it is the single most valuable uncited asset in the ledger. It converts §4's comparison from "263 vs 69, a 3.8× claim that App. J then normalizes away" into a **qualitative** claim that compute normalization cannot touch: the baselines' stored value is not merely shorter-lived, it is *destroyed by a small kick*. One sentence, already measured, at 5 seeds. If any single addition would move a reviewer's score, it is this one.

---

## 6. NICE

- `\ref` without `~` throughout: **"Fig.1"**, **"App.F"** (5×), **"Sec.4.1"** print with no space; elsewhere "App. F" (2×) and "Sec. 4.1" do. Visible on page 3.
- Figure 1 caption: *"γ = 0.05.(Figure downsized…"* — missing space after the period.
- App. K: *"The limitations of this early-stage **works** are broadly:"*
- App. M header: *"GMOR-proper results below, are a supporting result, not one of this paper's claims."* — comma splice; also the paper's most important honesty sentence, so it deserves to be well-formed.
- Page-1 running head prints **"Under Review - Extended Abstract Track 1–25, 2026"** — the string "1–25" advertises a 25-page extended abstract at the top of page 1. Template-generated and normal for `jmlr.cls`, but against a 4-page limit it is an unhelpful first impression.
- `\citet` inside the §4.2 heading propagates a hyperlinked author name into the section title, TOC and PDF bookmarks.
- **App. G's entire body is "See Fig. 5."** Same species as MUST-6, one grade less severe.
- **Figures 3 and 4 are byte-identical re-includes of Figures 1 and 2** (`fig1_gmor.png`, `fig_lifetime_headtohead.png`). Legitimate as full-size versions — but see MUST-7 on their self-referential captions.
- **No code or data availability statement anywhere.** Non-archival EA tracks rarely require one; reviewers ask anyway, and "single-core CPU, dim 4, 150 epochs" is a reproducibility story worth claiming out loud.

---

## 7. ⭐ THE STANDING QUESTION: *"Our results hold generally for the class of damped symplectic recurrences"*

**First, a correction the Head needs: that sentence is not in this build.** `grep` on the source returns no *"hold generally"*. The shipped sentence (p. 2, line 43) is:

> *"The laws are derived for the class of damped symplectic recurrences; we verify them on one trained instance."*

That is a materially different and much better sentence — it has already been split into a derivation claim and a verification claim. The question is therefore whether **the shipped sentence** is defensible.

**Verdict: not as written — but it fails on precision, not on honesty.** Two problems.

1. **"derived for the class" is broader than App. B's derivation.** App. B derives everything from *"the one-step map restricted to a single mass-whitened normal mode … at a critical point of V_θ"* — i.e. under **four** conditions: (i) a critical point, (ii) the harmonic/quadratic neighbourhood, (iii) decoupled normal modes after mass-whitening, (iv) this specific kick–drift–kick + `p ↦ (1−γ)p` discretization. Those conditions **are** the content of "for the class"; without them, "the class of damped symplectic recurrences" includes non-separable Hamiltonians, non-diagonal mass matrices, and anharmonic excursions — and the paper's own evidence shows condition (ii) biting: the softest emergent mode is *"genuinely anharmonic"* with retention lengthening 1.07 → 1.55 with kick amplitude (`v2-prefreeze-baselines.md`), and App. K records that explicit-breaking coefficients *"do not scale linearly with system retention outside of purely designed SO(2) geometric environments."* The sentence as shipped claims class-generality while the paper elsewhere documents where the class-generality stops.
2. **"one trained instance" is both an undersell and a scale-qualifier evasion (C-5).** It is 5 designed seeds + 3 emergent seeds at latent dimension 4 with G = SO(2) on a single CPU. "One instance" tells the reviewer neither the dimension nor the group nor the seed count — and this sentence is the paper's **only** attempt at a scope statement in main text.

**Minimal wording that would be defensible** (one sentence, same length class, no new claims):

> *"The laws are derived for conformally-symplectic kick–drift–kick recurrences in the harmonic neighbourhood of a critical point of V_θ, one mass-whitened normal mode at a time; we verify them at latent dimension 4 with G = SO(2), on 5 designed and 3 emergent trained seeds, on a single CPU."*

That is defensible because every word of it is proved (the first clause by App. B, which I checked line by line) or measured (the second by App. E). It also single-handedly repairs main-text gaps #7 and #9 in §2's table — which makes it, per unit of text, the highest-value edit available after the MUSTs.

---

## 8. Reviewer-hat attack pass (register composites, applied to THIS draft) + fresh attacks

**G1 — "a unit test on a testbed built to satisfy the theory."** Sharpest form here: §4.1's three headline numbers (`3.2e-10` constitutive identity, `1.000000 ± 5e-12` curvature ratio, `−0.985` slope) are produced by applying an **analytic** tilt `δ·cos nθ` to an **architecturally SO(2)-invariant** potential and confirming the Hessian eigenvalue equals `δΣ/(MF²)` with Σ and F² read off the same checkpoint. **Mitigated** — the paper labels it *"exact verification of the theory rather than emergent discoveries"* (C-2 satisfied) and App. M demotes GMOR to *"a supporting result, not one of this paper's claims."* **Not neutralised** — it still leads §4, the panel is titled "the curvature identity," and the identity's precision is what the abstract's tone rests on.

**G2 — "which component buys what."** Answered, well, in App. H (broken-volume, unconstrained twin, +γ, +γ_φ; BIBO 1.00 vs 0.33, µ² 0.008 vs 0.122, r\* 0.72 vs 0, 92 % / −24 % recovery — I verified the % Gap Recovered column against its own MSEs: 91.6 % and −23.9 % ✓). **Zero of it is in main text, and App. H is orphaned (MUST-5).** For an EA track that is acceptable; the orphaning is not.

**G3 — toy scale.** dim 4, S¹, ≤5 seeds, one architecture, laptop CPU. **Correctly and fully stated — in App. K.** In main text it survives only as "dim 4, 5 seeds" inside a figure caption and "a local Apple M1 chip" in prose. See §2 rows 3, 7, 9 and SF-15.

**G5 — certificate fine print.** No formal certificates here, but the structural analogue is MUST-2: an unconditional "infinite half-life" whose T = 0 scope is appendix-only. The FDT caveat itself **is** correctly stated (App. E: *"All finite-temperature results strictly require fluctuation–dissipation-consistent noise (σ\*ᵢ = √(MᵢTγ(2−γ))) and a Newtonian kinetic mode"*) — that is exactly the mandatory flag from CM-16 and it is present. Credit.

**G6 — foundational-paper falsifications.** **Clean.** No audit-confession paragraph anywhere (C-1 as reversed), no legacy mechanism-number is load-bearing, and J&P 2026 is cited in third person for the primitive only. Compliant.

**M2/M3 — de-anonymization / salami optics.** Effective de-anonymization is essentially complete — "CHLU" + "Causal Learning Unit (CLU)" + a third-person citation to `arXiv:2603.01768` identifies the group in one search. This is **charter-sanctioned** and is not a desk-reject vector under double-blind norms (citing one's own prior work in third person is permitted). The avoidable part is MUST-9's *uncited* "the CHLU" on page 2, which is strictly worse than a cited one. No salami exposure: the paper cites no sibling short (**C-8 satisfied**).

### Fresh attacks the register did not anticipate

**⭐ NEW-1 — "Trained" in the title is unearned.** The designed family's potential is **architecturally** SO(2)-invariant: the flat direction exists *before* training and training cannot destroy what the parametrization forbids. Every machine-precision result in the paper is on that family. So *"Transverse Curvature sets Retention in a **Trained** Recurrent Memory"* rests on checkpoints whose relevant property is structural, while the arm where training genuinely determines the geometry (the emergent MLP) is the arm that **fails** (App. F.1: µ² = 5.1–5.9e-2, 13–14 orders off; 1–1.6 bits; *"strictly a designed feature, not learned"*). The paper is honest about this in App. F.1 — but the title, abstract and §4.1 all lean on "trained." **This is the attack I would lead with as a reviewer, and it is not in the register.** It cannot be fixed hours before submission; it should be logged for the longs.

**NEW-2 — The register is never used.** Every measurement writes a phase and reads it back autonomously with the input removed at T = 0. No experiment shows anything downstream depending on retention. §1 asserts *"its operational utility as an AI memory mechanism"*; App. K correctly disclaims task benchmarks. Under this track's purpose ("early-stage results") this is survivable, and the paper says so — but it is the reviewer's second sentence.

**NEW-3 — µ⁻² is six lines of App. B.** A physics-literate reviewer will observe that `n₁/₂ ∝ µ⁻²` is the overdamped limit of a damped harmonic oscillator, that App. B derives it in six lines, and that §1's framing (*"the quantitative relationship … remains unmeasured"*) invites the reply *"unmeasured because elementary."* The paper's real defence is the **crossover, the floor and the exceptional-point structure** — genuinely more than the textbook result — plus the mapping onto a trained Hessian. The main text should say that in one sentence; currently it does not.

**NEW-4 — The paper's best result is in an appendix, and it is the one the track asked for.** App. F.1 (designed-not-learned; washboard; 1–1.6 bits) is the paper's most interesting *and* most damaging finding, and the EA track's stated purpose explicitly welcomes **"negative findings."** It occupies eight lines of App. F.1 and one clause of Contribution 3. Strategically, this is the material that most differentiates the submission from a generic workshop abstract, and it is the material most buried. (Charter **C-9** is satisfied in letter — the negatives are documented, and App. K's hypothesis-failure table is genuinely strong. This is a positioning observation, not a compliance one.)

---

## 9. Claim–evidence audit — what checks out, and what does not

**Verified against primary sources / internal arithmetic — CORRECT:**
- `5.6 / 56 / 69` map-steps for coRNN/LEM/LSTM ✓ (`v2-prefreeze-baselines.md` L152, 5 seeds).
- `263` map-steps, CLU-emergent ✓ **but see MUST-9**: source is `263 (184–436, n = 3)`.
- `27.03` steps floor at γ = 0.05 ✓ = `2ln2/(−ln0.95)`; matches App. B, §4.1, Figs. 1/3/5.
- `h* ≈ γ/2`, `h_f = 2 − O(γ²)`, `det A = 1−γ`, `tr A = (2−γ)(1−h²/2)`, `D_θ = εT(2−γ)/(2F²γ)` ✓ all re-derived.
- `det J = (1−γ)^d`, `J^⊤ΩJ = (1−γ)Ω` ✓ consistent with d = dim(q), phase space 2d.
- App. H `% Gap Recovered` 92 % / −24 % ✓ reproduce from the table's own MSEs.
- App. J per-step ratios 6.2× / 3.1× / 14–15× ✓; 263/69 = 3.8×, 263/56 = 4.7× ✓.
- `3.77 ± 0.23×` friction lengthening ✓ backed (`t-lever-forgetting`, 5 seeds, fdt + retied) — **but see SF-4** on what it is being asked to verify.
- `r* = 0.911 ± 0.016`, λ = 100, 5/5 seeds ✓ (`anchor-robustness.md`).
- Anchored slope `−0.956` (per-point) vs Fig. 5's `−0.961` (seed-mean OLS) — **difference is disclosed in-text; not a defect.** Same for App. H's twin 0.0047 vs 0.0128, which carries an explicit non-comparability note. Both read as previously-found defects correctly closed. Credit.
- Bibliography: 0 unresolved references (`??` count = 0). Metadata: `/Author()`, `/Title()`, `/Keywords()` all empty; no filesystem paths in the PDF or in any figure PNG. Class option `mlabstract` ✓.

**Does NOT check out:**
- **"≈ 20×"** in the abstract and in App. F.2 / Fig. 5 — **no denominator in the paper; no such ratio in any primary source.** MUST-3.
- **"the 5/5-seed latch statement is Sec. F.1's"** — it is not in F.1. SF-1.
- **"a higher wake MSE"** — the source says **~35×**. SF-3.
- **"FLOP-normalized 54.8× more"** — undefined normalization; 3.8× per retained step. SF-6.
- **"proven mathematically … volume conservation axiom"** — no proof, no axiom. SF-5.
- **"non-abelian tori"** — false. MUST-4.

---

## 10. Missing-experiment list for the Hub

**Genuinely missing (task candidates, not wiring):**
1. **Real data.** The standing #1 gap; the paper names it (*"Extensive results on real data are the next steps"*). The register's G7b route (robot joint space T ⁿ = U(1)ⁿ on `voraus-AD`) kills beyond-SO(2), beyond-toy-task and no-real-data in one push.
2. **A task in which retention buys accuracy.** Input-driven, with the equivariant-control wrapper §5 names. Closes NEW-2, and is the only thing that converts "the value survives" into "the memory works."
3. **dim ≳ 64 with mode mixing.** Named in §5; the whole framework assumes decoupled normal modes, and that is the assumption most likely to fail first.
4. **Width-matched compute comparison.** App. J flags the confound honestly (*"the comparison is not width-matched, CLU at hidden 64 … against baselines at hidden 16"*) but the experiment is not run. This is the ablation a reviewer demands the moment they read the compute appendix.
5. **Anchored-CRR at 5 seeds** (currently 3), to remove SF-2.
6. **The D_θ γ-scaling pre-registered against its own closed form** — predict 4.33×, measure, report the residual (SF-4).
7. **The symmetric arm of the Mo comparison** — run the CRR's prediction on Mo's own models, not only Mo's estimator on ours. Currently the "head-to-head" runs in one direction only (SF-8).
8. **Beyond SO(2)** — T² and SO(3)→SO(2), per register G7a, with the register's already-derived predictions.

**Exists in outputs, not cited — wiring notes (cheap, high value):**
- ⭐ **LSTM 69 → 2 under a 0.1 hidden-state perturbation** (`v2-prefreeze-baselines.md`). See SF-16. Top of the list.
- The 263 seed range **184–436, n = 3**.
- The anchor's **~35× wake-MSE cost** and the **λ-envelope** (λ ≈ 10 best rejection, 1/5 collapse).
- **D_θ verified to 1.0068 ± 0.0219 over 25 cells** — a better verification sentence than the 3.77×.
- **Tilt immunity / "erosion ∝ flatness"** (`anchor-robustness.md`): lifting the flat direction with δ ≈ 0.05 immunizes the vacuum with **no anchor at all**. This is a genuine theoretical result — erosion attacks flatness specifically — and it is absent from the paper.
- The erosion-onset epochs **116 / 442 / 959 by sleep frequency** (`sleep-erosion-study.md`) — present in App. K, but not where MUST-3 needs them.

---

## 11. ⭐ The three sentences a hostile reviewer would quote

> **1.** *"The paper's headline precision — a curvature ratio of 1.000000 ± 5×10⁻¹² and a constitutive identity to 3.2×10⁻¹⁰ — is a Hellmann–Feynman relation µ²F² = δΣ evaluated with Σ and F² measured on the same checkpoint, under an analytic tilt applied to a potential whose flat direction is imposed by the architecture rather than found by training; the authors' own figure calls it 'the curvature identity,' and an identity that holds to 10⁻¹² is a unit test of their autodiff, not a measurement of memory in a trained network."*

> **2.** *"The abstract claims the retention profile survives '≈20× the temporal horizon where the objective naturally degrades it,' but §F.2 reports the vacuum inverting at 1000 epochs and the anchored run lasting 3000 — a factor of 3 — and no denominator for the multiplier appears anywhere in the paper."*

> **3.** *"The single comparative performance number is 263 map-steps against 69 for an LSTM: a 3-seed median whose 184–436 spread is never disclosed, measured at T = 0 with the input removed, against models the authors' own Appendix J shows are 14–15× cheaper per step — and the paper's main text states that its latch has 'an infinite half-life' while Appendix E states that this is true only at T = 0."*

---

## 12. Craft summary

**Page budget:** 4.3 pp main text (measured), refs from p. 5.35 to p. 9, appendices pp. 10–25, 25 pp total. Main text ≈1766 words; appendices ≈4640. **Headline figure:** Figure 1 is the right choice and reads well at 0.7\linewidth — but its in-image title contradicts the paper (MUST-8). **Figure quality:** Fig. 1 ✓, Fig. 5 ✓, Fig. 7 ✓; **Fig. 2 illegible** (MUST-7); Figs. 3/4 are duplicates carrying self-referential captions. **Contribution clarity on p. 1:** strong — three numbered contributions, the negative result promoted to contribution status, and the "This work does not propose a new architecture" disclaimer is exactly the right register for this track. Undercut only by two of the three contributions pointing into appendices. **Related work vs the scout ledgers:** thorough and current — Keller flow-equivariance, Lillemark world models, Ságodi fine-tuning, Burak–Fiete Fisher bound, Dinc ghost mechanism, Haputhanthri, Vafidis, Xu, van der Ouderaa, Titans, HiPPO, NTM/DNC. The *"Delineation from current literature"* paragraph in App. D is the best-written paragraph in the paper. **Appendix maximalism (C-10):** satisfied in volume; undercut by three orphaned appendices, one empty section, one one-sentence section and one truncated stub. **C-7 (flag provenance):** satisfied in appendices (App. E and App. M both carry proper configuration blocks); **violated in main text**, which carries none.

---

## Proposed handover updates (for the Hub)

- **Reconciliation list — has an owner requirement.** MUST-3 ("≈20×") is a **number-provenance defect that reaches the abstract**; it is the same species as the retracted "13.9× memory vault" precedent in `AGENT_PROTOCOL.md` §5 — a drafting-time reformulation with no measured backing that propagated into lead position. Whoever fixes it must also fix App. F.2 and Fig. 5's caption (three sites), and the erosion-onset epochs (116/442/959) should travel with it.
- **Register addition candidate — G8 ("the 'trained' word").** NEW-1 is not covered by G1 (testbed satisfies theory), G3 (scale) or G7 (stickman). It is: *the property being measured is architectural, so the checkpoints' trained-ness is not load-bearing, while the arm where training does determine the geometry is the arm that fails.* This will be the first thing an ICLR/NMI reviewer says about the longs, and it needs a claim-shape answer before then.
- **Charter compliance summary for this draft:** C-1 ✓ · C-2 ✓ (label present, ordering weak) · C-3 ✓ (main text is ML-first; physics quarantined) · C-4 ✓ · **C-5 ✗** (abstract and §5 scope-free; SF-15, MUST-9) · **C-6 ✗** (MUST-2, SF-3) · **C-7 ✗ in main text** (✓ in appendices) · C-8 ✓ · C-9 ✓ · C-10 ✓ in volume, ✗ in wiring (MUST-5/6).
- **Venue fact to re-verify at submission:** the banked 4-pp EA limit is Head-verified 2026-08-18; the measured build is 4.3 pp. The standing per-venue re-verification rule applies and this is the one field where re-reading one CFP line has an asymmetric payoff.
