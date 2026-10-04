# v2-referee-final — paper-referee report
Task + acceptance criterion: submission-state adversarial review of `~/Desktop/V2_NeurReps_Submission/paper.pdf` (NeurReps 2026 EA track); zero writes outside this file; every MUST anchored to a quotation; self-containment question answered explicitly.
Status: done

**DIAL DECLARATION (echo): none — read-and-report. Zero edits made to the submission folder or any other file; sole write = this report.**

**Independence bar (stated per task):** I did NOT read `pj-referee-v2*`, `pj-fidelity-v2*`, `v2-referee-v07`, `v2-cite-pass`, `v2-bib-doi-list`, `v2-figure-text-pass`, or `BUILD-NOTE.md`. Inputs: AGENT_PROTOCOL, Positioning Charter (C-1…C-10), claims matrix (§1 canonical constants, CM-1…CM-23, CM-15/16a/4/6 in detail), critique register (G1–G7/V*/M*), the two audience calibration ledgers, `paper.tex`, `refs.bib`, `paper.bbl/.blg/.log`, the built PDF (rendered pages + text extraction), and all five PNGs. Every defect below was found against the artifact itself.

What I did: full read of `paper.tex` (328 lines) + built PDF; charter C-1…C-10 item pass; claims-matrix cross-check of every headline number (spot-check mode per task — spine previously verified); bibliography audit (52 bib entries, 47 cited keys, bbl 48 items); anonymity/metadata sweep (`pdfinfo`, log, bib fields, figure filenames); figure legibility pass at render; page-split accounting from pdftotext page map.

How I verified: `pdfinfo` (Author/Title empty, 23 pp); `grep` of log (only benign warnings: "No \author given", obsolete `\rm`, hyperref Unicode bookmarks); key-by-key bib↔tex diff; rendered p.1, p.4 excerpts, and all five figures; floor 2ln2/(−ln 0.95)=27.03 re-derived; 0.190/0.0128=14.8×≈CM-1's "~15×"; 4557/4549=+0.18% vs the quoted "±0.05%".

---

## Simulated verdict (EA track: "early-stage results, negative findings, opinion pieces, or novel datasets")

**(a) Content: WEAK-ACCEPT, arguing toward accept.** Meta-review: the paper does exactly one thing and does it honestly — it measures the constitutive exchange rate between transverse curvature and retention on a trained symmetry-broken potential, locates the exceptional-point crossover and the curvature-independent floor, and shows a published Lyapunov lifetime estimator (arXiv:2605.03338) is the law's overdamped branch, failing predictably (0.31×) past the EP. The designed-verification vs learned-evidence discipline is stated on p.2 and mostly executed; the negatives appendix is unusually honest (a genuine fit for this track's charter); the compute inversion is disclosed rather than hidden. What holds it at weak-accept: everything is dim-4/S¹/CPU with the flat direction installed by construction (the paper's own §4.3 concedes it cannot be trained for), the closed-form laws are asserted nowhere-derived in a now-self-contained submission, and the main text is roughly double the track's budget.

**(b) Artifact as mechanically shipped: NOT SUBMITTABLE TODAY.** Three mechanical defects a chair or reviewer hits in the first four pages — a broken sentence inside Contribution 1, a main-text figure reference that points at the wrong figure (rendering as "Fig.'5"), and an ~8-page main text against a 4-page limit — plus one internal number contradiction in App D. After the MUST list, (a) governs.

## The ONE contribution, in my own words (⭐ deliverable)

*On a trained damped-symplectic recurrent unit whose learned potential spontaneously breaks SO(2), the lifetime of a value written on the orbit is set by the transverse curvature through a measured two-branch law — n₁/₂ ∝ μ⁻² below the exceptional point εμ ≈ γ/2, a curvature-independent floor 2ln2/(−ln(1−γ)) above it — and a published Lyapunov-based lifetime estimator is exactly this law's overdamped branch.*

That is one sentence, so the Head's de-scope **lands**. The three-item contributions list (CRR / head-to-head / boundaries) reads as one claim plus its evidence and its perimeter, which is the right shape. The dilution risk is §4.4 (anchor survival) being framed as co-equal rather than as a robustness property of the one claim — minor.

## True page split and the chair's mechanical response

- **Main text (title → end of §5 Discussion): pp. 1–8 (~7.6 pp).** References: pp. 8–12. Appendices A–I: pp. 12–23. Total 23 pp.
- **Venue limit: 4 pp main text** (refs+appendices unlimited). The header itself advertises "1–23".
- **What a chair does:** the EA track checks the main-text budget mechanically; at ~2× over, this is a **desk reject without review** at most PMLR-style workshops. ⚠ Per the standing Head ruling, page limits are DEFERRED and I recommend no cuts here — but the compression pass is not optional before submission; it is the difference between being reviewed and not. Natural moves exist without losing claims (the §4.x subsections carry appendix-grade detail inline), and I flag only that the pass must exist, not its content.

---

## MUST-FIX (blocks submission)

**MF-1 — Broken sentence inside Contribution 1, p.2.** Quotation (as rendered): *"1. The curvature-retention relationship (CRR) (Sec. 4.1, Sec. 4.4): We establish the **isrelationship**, identify the crossover boundary…"* (`paper.tex` line 47–48: "We establish the\nisrelationship"). The paper's single most important sentence — the statement of its contribution — is ungrammatical garbage on page 2. A reviewer's confidence in the numeric care collapses here first.

**MF-2 — §4.1 cites the WRONG figure, with a rendered stray quote; Figure 1 is never referenced at all.** Quotation (rendered, p.4): *"The fitted overdamped slope is −0.985 (n = 35) as shown in **Fig.'5**, closely matching the theoretical −1."* Source: `Fig.`\ref{fig:gmor}`` (line 77) — a backtick typo AND the wrong label: `fig:gmor` is the appendix-I GMOR-condensate figure (Figure 5, p.22); the sentence describes `fig:pricelist` (Figure 1, directly below it on p.5). `\ref{fig:pricelist}` appears nowhere in the document — **the headline figure is uncited by the text**. Fix: `Fig.~\ref{fig:pricelist}`.

**MF-3 — Self-containment FAILS (the pass's ⭐⭐ question, answered explicitly): the closed-form laws are asserted with no derivation anywhere in the submission.** With the theory note removed, the following rest on nothing the reader can see:
  - §3: *"The Overdamped Register… Resolves to a half-life of n₁/₂ ≈ 2γln2/[(2−γ)(εμ)²]"*, the floor *"2ln2/(−ln(1−γ))"*, and *"the exceptional point h\*≈γ/2, where the block matrix becomes defective"* — the paper's central formulas. The only citation is \citep{hairer_geometric_2006}, a general geometric-integration text that does not contain them. The 2×2-block reduction is stated but never solved.
  - App B: *"the coset diffuses with coefficient D_θ = εT(2−γ)/(2F²γ)"* — asserted.
  - App F: *"The closed-form charge-oscillation amplitude bounds strictly dictate on-vacuum-orbit behaviors"* — the closed form is nowhere in the paper.
  - App D: *"This penalty is **proven mathematically** to be contraction-forbidden by the volume conservation axiom"* — no proof exists in the submission (and the supporting clause is a measurement, not a proof; see SF-4).
  The fix is cheap and additive under C-10: a ~1-page appendix deriving the three bands from the damped-leapfrog 2×2 block (eigenvalues, defectiveness at h\*, half-life of the slow eigenvalue, the γ-only envelope). Without it, a reviewer reproducing the overdamped coefficient 2γ/(2−γ) has to re-derive the discrete map themselves and will treat every "exact" claim as unverifiable.

**MF-4 — Undefined load-bearing symbols; one contradicts the claims matrix.** App I.5 quotation: *"the theory predicts a relative leading-order error x ≡ δ/(M_ch μ_rad² **f**)"* — **f is never defined**, and the same letter is used two paragraphs earlier for tilt harmonics (*"Where the quality factor permits (f = 2,4)"*). CM-15's canonical form is *"LO-GMOR rel. error = δ/(M_ch μ_rad² **r\***)"* — either f ≡ r\* (then say so; Figure 5(c)'s legend "angular tilt: Σ ≡ f" hints at it) or this is a transcription drift from the matrix. Additionally **M_ch, μ_rad, n and r\* are used but never defined in the text**, and three of them sit on **Figure 1's y-axis** (main text): "μ²_meas/[δn²/(M_ch r\*²)]". A reader cannot parse the headline figure's axis from the paper.

**MF-5 — Internal parameter-match contradiction inside App D (the exact MF-3/w7 class this program's register exists to prevent).** First paragraph: *"parameter-matched controls (dimension 4, 150 epochs, **±0.05% parameters**, 3 seeds…)"*; the data-config note five lines later: *"parameter-matched to **±0.2%** (CLU 4549 / broken-volume 4557 / twin 4551)"* — and 4557 vs 4549 is +0.18%, arithmetically incompatible with ±0.05%. If these are two different experiments (the note says the tables are not cross-comparable), each tolerance must be pinned to its table; as written it is a contradiction a reviewer can construct in 30 seconds. (CM-1 records ±0.05% for Part A — so the ±0.2% table needs its own labeled provenance.)

## SHOULD-FIX

**SF-1 — "orders of magnitutde" is misspelled at least 7 times** (lines 79, 83, 147, 263, 309, 313, 319 — it renders on pp. 4, 5, 15, 19, 21, 22), plus *"desired maturiity"* (§4.3), *"this early-stage works"* (App F), *"GMOR-proper results below, are a supporting result"* (comma splice, App I). A repeated identical typo reads as "never proofread."

**SF-2 — C-5 scale-qualifier violation, §2 (the register's G3, verbatim pattern).** Quotation: *"**Our results hold generally for the class of damped symplectic recurrences**, guided by an exactly-solvable underlying theory."* Everything is measured at dim 4, S¹, 5+3 seeds, CPU. Rephrase: the *quadratic-core theory* applies to the class; the *measurements* are dim-4/S¹. (Abstract and App F carry their qualifiers correctly; this is the one bare generalization, and it sits in the framing section.)

**SF-3 — "cannot manifest in first-order dynamical systems" is overbroad and attackable** (§2: *"regime transitions (floors and crossovers) that cannot manifest in first-order dynamical systems"*; §4.2: *"which standard first-order dynamical systems inherently miss"*). A first-order *vector* system (any 2-d linear ODE) has complex eigenvalues and EPs; what is meant is gradient-flow/overdamped/single-exponential models. Say that.

**SF-4 — App D over-claims its mechanism evidence and its arithmetic is nonsensical as phrased.** Quotation: *"proven mathematically to be contraction-forbidden… as the broken-volume baseline precisely **recovers ≈2.4× of the performance gap** purely by leaking volume."* (i) A measurement is not a mathematical proof — CM-1's approved wording is "the **measured** cost is contraction-forbidden"; (ii) "2.4× of a gap" is not a well-formed fraction — state what was measured (e.g., broken-volume closes the gap by a factor ≈2.4 in MSE terms).

**SF-5 — The paper never says the emergent 263-step lifetime is LAW-PREDICTED — its strongest cross-link, already measured, is left out.** §4.3 reports *"the generically-trained CLU holds for ≈263 map-steps"* as a bare number. CM-4 records the CLU-emergent lifetime as **law-predicted within a +12–15% envelope (per-seed +5 to +29%, kick-probe)** and the perturbation-fragility foil (**LSTM 69→2 under a 0.1 hidden kick**). Both exist in `v2-prefreeze-baselines` outputs and neither is wired in. Adding one sentence converts §4.3 from "our thing drifts less" into "the §4.1 law *predicts* the trained system's lifetime" — the paper's own G1 defense. This is a wiring note, not a missing experiment.

**SF-6 — Figure/text vocabulary split: the embedded figure titles say "GMOR", the paper says "CRR".** Figure 1: "GMOR retention law on trained CLUs"; Figure 3: "GMOR retention at 3000 ep". The captions were relabelled to CRR but the PNGs were not regenerated. A reviewer sees the paper's named object change identity between text and its own figures — and "GMOR" in the headline figure is the exact "damped-oscillator problem set" surface C-3 warns about. Also Figure 1 right panel carries matplotlib's raw "1e−9+1" offset axis.

**SF-7 — Small internal number frictions.** (i) §4.4 text: curvature ratio *"holds to 1.5×10⁻¹²"* vs Fig. 3 caption *"1.0000 ± 10⁻¹²"* — pick one form. (ii) §4.2: *"corr(log pred, log meas) = 0.9987"* vs CM-4's recorded "corr 0.9995" for Mo's λ̂(128) tracking — if these are different correlations (estimator-vs-CRR vs λ̂-vs-budget), fine, but a program-consistency reader (M2) will diff them; confirm provenance. (iii) Abstract's *"≈20× the temporal horizon"*: App F gives inversion epochs 116/442/959 by sleep frequency, so 3000 ep is 3.1×–26× depending on config; pin which horizon the 20× is against (3000/150?).

**SF-8 — Citation-practice inconsistencies (item-1 pass).** The Mo preprint is cited three ways: `\citet{mo_symmetry-protected_2026}` (§4.2, captions), raw *"arXiv:2605.03338"* (§4.2 body, App A ×2, Fig. 4 caption), and *"The equivariant-Lyapunov preprint"* — use the key everywhere. App A: *"concurrent workshop work has explored soft symmetry regularization for continuous attractors"* has **no citation** (the scout ledger's N2 is marked "do not cite until read" — if that's the reason, the sentence should either name it as unciteable-at-review or be cut; as printed it's an unsupported claim about specific prior work). Four bib entries are never cited: `rusch_unicornn_2021`, `gardner_toroidal_2022`, `khona_attractor_2022`, `kim_ring_2017` — harmless to the build (bbl is citation-driven) but two are real gaps: **UnICORNN** is the architecture CM-21/22 says "occupies the stability ground with a theorem" and belongs in App H's positioning list next to HiPPO; **kim_ring_2017/khona_attractor_2022** are the ring-attractor canon this audience expects in App A ¶3. Otherwise the ~47 wired citations check out: keys match claims sensibly (Burak–Fiete for Fisher-bounded diffusion, Seung for integrators, Renart for homeostasis, McLachlan–Perlmutter for conformal symplecticity, GMOR 1968, HiPPO/Kong/NTM/DNC exactly where CM-21 demands them), `\citet`/`\citep`/`\citealp` are used correctly throughout, and no citation is decorative.

**SF-9 — "the Experiment-D configuration of the reference unit" (App B) is insider jargon.** The numbers that follow (dim 4, h64, dt 0.05, 256 pts, 150 ep) make it reproducible, so this is optics: "Experiment-D" is meaningful only to someone holding the cited paper's internal artifacts, which sharpens the M2 de-anonymization surface (see below). Drop the label, keep the numbers.

**SF-10 — Charter C-3 deviation, for the Head to ratify or waive:** C-3 says the Mo head-to-head *leads* the results; here §4.1 (designed-vacuum verification) leads and the head-to-head is §4.2. Defensible for this audience (the law must exist before its containment claim), but it is a recorded deviation from a binding writing rule, so it needs an explicit waiver, not silence.

## NICE

- N-1: Abstract is ~280 words with *"we identify the bounds for these results by identifying where the laws do not extend"* — repetitious; also title capitalization ("Transverse Curvature **sets** Retention").
- N-2: *"generated by a local Apple M1 chip"* (§4.3) — honest but oddly phrased; "single laptop CPU (Apple M1)" reads better and matches App E.
- N-3: Fig. 2 legend says "CLU (5 seeds)" but ~3 traces are visually distinguishable; if seeds overlap, say so in the caption (the censoring annotation "10/70 runs" is excellent).
- N-4: Audience ledger notes the room's sharpest term is now Mo's "symmetry-protected neutral mode" and "canonicalization" for the coset coordinate; the title's "flat direction" is fine, but App A could adopt "canonicalization" once (currently "canonical quotient coordinates").
- N-5: `\rm` obsolete-command warnings (log lines 243/260/325) and hyperref math-in-bookmark warnings — cosmetic; fix if rebuilt anyway.
- N-6: §3 *"whether or not it is even possible to define one for a given set of input modalities such as images, text etc."* — informal; tighten.

## Novelty boundary vs arXiv:2605.03338 (item 3) — CLEAN, with one heading to watch

The line is respected in both directions. Credit given: abstract and §1 attribute the zero-exponent protection and fragility results to prior work; App A ¶2 states *"We do not claim novelty over these qualitative lifetime predictions or existence proofs"* and preserves "at least dim(G/H)" (their sufficiency-lower-bound form). Ours claimed correctly: two-branch closed form, EP, floor, trained-potential measurement, estimator-as-overdamped-face; the units firewall (1/time vs 1/time²) is handled exactly as the task prescribes — *"cross-field comparisons are performed strictly by running published rate-based estimators unchanged on our trained models."* The §4.2 heading *"…is the CRR's overdamped face"* and *"theoretical containment rather than conflict"* sit at the aggressive edge of the defensible line but inside it, because the containment is measured (1.012–1.029 vs their 1.013 below h\*; 0.31 above), not asserted. CM-4 compliance: 3.2× at trained δ=4, "≈5×" exact-map-only — both stated with the right scopes.

## Anonymity / desk-reject surface (item 5)

- **Class option: `mlabstract` confirmed** (line 2; header renders "Under Review - Extended Abstract Track"). Not `mlmain`. ✓
- **PDF metadata clean:** Author/Title/Keywords empty; no acknowledgements; author block absent (log: benign "No \author given"); footer "© 2026 ." with empty holder — standard for this class. Figure filenames generic. No URLs in the rendered text beyond bib entries. ✓
- **The M2 surface that remains, for the Head's judgment, not a fix I can make:** the paper cites `jawahar_chlu_2026` in third person (charter-compliant) but then (a) *renames the cited architecture* — *"similar to \citet{jawahar_chlu_2026}, presented here as the Causal Learning Unit (CLU)"* — and (b) evaluates it via internal artifact labels ("Experiment-D configuration of the reference unit"). No third party renames someone else's architecture and holds its internal experiment taxonomy; a motivated reviewer infers authorship in one step. Non-archival + third-person self-citation makes this survivable, but it is the single largest de-anon tell, and SF-9 removes half of it for free. Note also the F5-era Head constraint was "no CLU coinage" — if that ruling was meant to extend past the F5 note, this paper violates it; if not, record the waiver.

## Charter / matrix compliance summary

C-1 ✓ (no audit confession; legacy paper cited for the primitive only). C-2 ✓ and explicitly encoded in the text (p.2 and §4.1 closing sentence). C-3 ⚠ deviation (SF-10) + GMOR-titled figures (SF-6). C-4 ✓ (future work confined to §5). C-5 ⚠ one violation (SF-2); otherwise qualifiers are consistently in-sentence, including the abstract. C-6 ✓ (BIBO scoped to coercive-potential + autonomous in App D/F). C-7 ✓ adequate (App B + App D config paragraphs + per-figure setups; no commit hashes, acceptable for a paper). C-8 ✓ hermetic (no other short cited or assumed). C-9 ✓ exemplary — App F's two negative-results tables are the strongest compliance artifact in the submission. C-10 ✓. Forbidden-claims sweep: CM-3 (energy-signal superiority) absent ✓; CM-21(a–d) each explicitly disavowed in App A/H with the mandated citations ✓; CM-22(b) extrapolation claims absent ✓; "19×" absent ✓; no external-benchmark claims (App F: "We do not present or claim external task benchmark superiority") ✓; CM-16a designed-only split respected (§4.3: continuum register "strictly a designed feature") ✓; CM-4 compute retirement respected (App E presents the 3.8×/4.7× only inside the normalization table with the 54.8×/70.7×/23.5×/14.6× inversions adjacent) ✓. Canonical constants: floor, EP, latch transport, σ\*, D_θ, (1−γ)^d volume — all match §1 of the matrix ✓.

## Missing-experiment list (for the Hub)

**A reviewer will demand (genuinely missing runs):**
1. **Width/param-matched baseline retention.** The 263-vs-69/56/5.6 triad comes from CLU(h64, 4549 params) vs baselines(h16, 1186 params). App E concedes non-width-matching for *compute* but the *retention* comparison inherits the same confound. One run: LSTM/LEM/coRNN at matched params (or CLU at h16) through the same S¹ write-hold protocol.
2. **Direct T>0 retention curve.** All results are T=0; the D_θ = εT(2−γ)/(2F²γ) law is supported only by one friction ratio (3.77±0.23×). A memory paper whose noise law is asserted-plus-one-point will be asked for n₁/₂(T) at fixed γ.
3. **One latent dimension above 4** (8 or 16): does the 2×2 normal-mode reduction survive mode mixing? The Discussion itself names dim ≳64 as open; a single dim-8 point would defuse "dim-4 island" cheaply.

**Nice to have (or Head-assigned elsewhere):**
4. Torus T² / non-abelian SO(3)→SO(2) — G7's mandate, ruled to the longs; the Discussion correctly parks it.
5. Emergent-arm GMOR with self-breaking δ_eff — App F names it untested; running it converts a stated limitation into a boundary measurement.
6. Input-driven (non-autonomous) retention under task load — the paper scopes it out (App F "Off-Distribution Stability"); the scope note suffices for this track.

**Exists in outputs but not cited (wiring, not experiments):** the law-predicted +12–15% envelope for the emergent 263 and the kick-probe fragility foil (SF-5, from `v2-prefreeze-baselines`); the anchor λ∈{1,10,100} strength↔robustness envelope (`anchor-robustness`) behind §4.4's one-line "trading off weaker noise rejection" — one sentence each.

## The three sentences a hostile reviewer would quote

1. *"Our results hold generally for the class of damped symplectic recurrences, guided by an exactly-solvable underlying theory."* — quoted against dim 4, S¹, 5 seeds, one architecture, laptop CPU.
2. *"On this architecture class, flat directions must be designed rather than trained for; the continuous register poorly transfers to an emergent potential."* — quoted as: the phenomenon being priced does not arise in trained networks unless the authors install it (G1 in the paper's own words).
3. *"We establish the isrelationship, identify the crossover boundary where this law halts…"* — quoted verbatim as evidence the submission was not proofread.

Open questions / follow-ups / risks: (i) SF-7(ii) 0.9987-vs-0.9995 needs a provenance check I could not close without the forbidden prior reports; (ii) the "no CLU coinage" ruling's scope (F5-only or program-wide) is a Head question the coinage in §1 forces; (iii) whether the compression pass or a derivation appendix lands first matters — MF-3's appendix ADDS pages, so sequence it before the compression pass measures the final split.

## Proposed handover updates (for the Hub)
- V2 submission state: content weak-accept-grade at the EA track; artifact blocked on 5 MUSTs — MF-1 broken Contribution-1 sentence; MF-2 wrong+corrupted figure ref (Fig.'5→Fig.1, fig:pricelist uncited); MF-3 self-containment failure (three-band closed forms, D_θ, charge-oscillation bound, "proven" contraction-forbidden claim — all underived in-submission; needs a ~1-page 2×2-block derivation appendix); MF-4 undefined f/M_ch/μ_rad/n/r\* incl. Figure 1's axis, with f-vs-r\* drift against CM-15; MF-5 ±0.05% vs ±0.2% contradiction in App D.
- True split 7.6/4.4/11 pp (main/refs/appendix) vs a 4-pp limit — compression pass required before submission; MF-3's appendix must precede it.
- Head decisions needed: C-3 ordering waiver (SF-10); CLU-coinage scope + the rename/Experiment-D de-anon tell (SF-9/anonymity section); whether to add UnICORNN to App H (CM-21/22 says yes).
- Wiring wins available at one sentence each: law-predicted emergent lifetime + kick-probe foil (CM-4), anchor λ-envelope (SF-5, SF-7iii).
- Missing-experiment candidates, reviewer-demand tier: width/param-matched retention; direct n₁/₂(T) curve; one dim-8 point.
