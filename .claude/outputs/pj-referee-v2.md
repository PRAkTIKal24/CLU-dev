# pj-referee-v2 — paper-referee report

**Task + acceptance criterion:** review `NIPSsubmission/v2-neurreps/pj_sub` as a composite NeurReps 2026 Extended-Abstract-track reviewer receives it; itemized MUST-FIX / SHOULD-FIX / NICE with locations and concrete failure scenarios; plain verdict; the three hostile quotes; what the compression bought and cost.
**Status: done.**
**DIAL DECLARATION (echoed): none — adversarial review; no performance claim; no laundering control applies.**

⛔ **Zero edits. `pj_sub.tex` was NOT touched** (sha256 `4aafc2c2…14bb3`, mtime Aug 22 00:37:54, 10,101 bytes — unchanged before and after this pass). Nothing in `.claude/NIPSsubmission/**` was written to. All my scratch is under `.claude/scratch/pj-referee-v2/`.

⚠ **Independence honoured:** I did not open `outputs/pj-fidelity-v2.md` (verified: never read, never grepped). Every finding below is from the artifact plus the primary sources I went to on my own (`submission.tex` in the same folder, `claims_matrix.md` CM-4, `philosophy-synthesis.md` Positioning Charter, `critique_register.md`, `audience-refresh-2025-2026.md`, `n1-fulltext-and-track-check.md`).

---

## ⛔ PRECONDITION DEVIATION — read this first

**`pj_sub.pdf` does not exist on disk.** The folder contains `pj_sub.tex` (Aug 22 00:37) and a *different* artifact, `submission.pdf` (Aug 21 19:16, 18 pp, the clean base). My mechanical precondition therefore failed as written.

I proceeded rather than block, by building the PDF myself: copied `pj_sub.tex` + `neurips_2025_ml4ps.sty` + `figs/` into `.claude/scratch/pj-referee-v2/build/` and compiled with **tectonic 
(xdvipdfmx)**, not the Head's `pdflatex`. Result: **3 pages**, 1 underfull `\hbox` (badness 2229, abstract, line 31), 0 errors, 0 undefined refs, 0 undefined citations (there are none to define). ⚠ **A pdflatex build could paginate differently by a few lines; the "3 pages" figure is mine, not the Head's.** If the Head's own build differs materially, re-run me. Everything else in this report is source-level and build-independent.

---

## VERDICT (simulated, NeurReps 2026 Extended Abstract track)

# **REJECT as the artifact stands.** *(Would flip to weak-accept — plausibly accept — after roughly two hours of restoration work, none of which costs a single line of the 4-page budget.)*

**Meta-review.** The track's stated purpose is "early-stage results, negative findings, opinion pieces, or novel datasets," and on *content* this submission is a good fit for it: a small, honest, single-mechanism study with a real negative result, in a room that demonstrably accepts toy scale (2025 accepted set includes dihedral-group multiplication and modular addition), accepts physics-native machinery without apology (Noether 2024, symplectic integrators 2024, Poisson algebra 2025), and explicitly names negative findings in its own track text. I want to like it. But the artifact I was handed **contains zero citations, zero figures, zero tables and zero appendix, and uses 2.6 of its 4 permitted pages** — and the EA track's page limit is *"4 pages, excl. refs + appendices,"* so every one of those omissions was free. The consequences are not cosmetic. §2 states another group's theorem ("exactly equivariant fields possess zero Lyapunov exponents tangent to the orbit") and reproduces that preprint's own title phrase ("symmetry-protected Lyapunov neutral modes") with no citation; §4.2's entire empirical contribution is "a published single-exponential lifetime estimator" that is **never identified**, so no reviewer can check the paper's second contribution at all. Meanwhile the headline number promoted into the abstract — a 12-significant-figure identity — is the one result the paper's own longer version labels *"verification of the theory's exactness … not a discovery,"* and that label has been deleted; and §4.3 reinstates a baseline comparison that the program's claims matrix **explicitly retired** ("the '≈4× longer' RETIRED as a compute claim") without the compute normalization that killed it. I would not need to be hostile to reject this; I would only need to be careful. Under this venue's norms the missing bibliography alone is close to a desk-level defect, and the specific uncited work is a three-month-old single-author preprint whose author is a plausible reviewer in this exact room.

---

## MUST-FIX (blocks submission)

**MF-1 — There is not one citation in the paper. §2 (Related Work), lines 46–59; whole document.**
`grep -c "\\cite"` = **0**; no `\bibliography`, no `thebibliography`. §2 asserts, unattributed: "Continuous attractors represent states that are marginally stable tangent to the manifold and stable normal to it" (that is Ságodi et al. 2024, quoted almost verbatim in the base at line 55); "Standard established dynamics indicate that noise diffuses coordinates along these manifolds"; "Restorative homeostatic mechanisms are standard in the literature"; "observations of approximate continuous attractors in task-trained recurrent networks"; "flow equivariance"; "recent proofs regarding symmetry-protected Lyapunov neutral modes." **Failure scenario:** reviewer opens the PDF, scrolls to the end, finds no reference list, and stops evaluating. In a room where "flow equivariance" *is* a named person's 2025 NeurIPS spotlight and "symmetry-protected Lyapunov neutral modes" *is* the verbatim title of arXiv:2605.03338, a paper that recites both phrases with no citation reads as either careless or worse. The EA limit excludes references — this cost nothing and gained nothing.

**MF-2 — The paper's second contribution cannot be evaluated because its object is anonymous. Abstract l.17–19; Contributions l.39–41; §4.2 l.83–88.**
"a published single-exponential lifetime estimator," "Running the exact protocol across our trained models." *Which* estimator? *Which* protocol? The base version gives all of it — arXiv:2605.03338, and the code-level constants "phase 0.35 rad, threshold 0.2 rad, its censoring, cap 15000 steps … across 14 breaking magnitudes on 5 trained models." pj_sub gives **none of it, not even n**. **Failure scenario:** reviewer writes "the head-to-head is the most interesting claim here and I cannot check a single thing about it: I do not know whose estimator it is, what its protocol is, how many regimes or seeds were run, or what its own published number was." Note the specific loss: the base reproduces N1's *own reported median of 1.013* — the single strongest evidential sentence in the whole paper, because it shows the instrument was run correctly — and pj_sub replaces it with "ratios of ≈ 1."

**MF-3 — The abstract states another paper's theorem as unattributed fact, in the paper's second sentence. Abstract l.2–3.**
*"Exact equivariance protects these neutral directions, yielding zero Lyapunov exponents tangent to the group orbit."* No hedge, no citation, no attribution. The fence survives only in §2 ("We do not claim this existence result") — 55 lines later, and itself uncited. The base version deliberately puts the attribution **in the abstract, before any of our results**, naming arXiv:2605.03338 explicitly. That protection has been removed by the condensation. **Failure scenario:** the reviewer is (or has read) Mo 2026; they read sentence 2 of the abstract, see their own theorem asserted flatly, see their own title phrase in §2, find no citation anywhere, and the review is over regardless of what §4 contains. This is the single highest-variance defect in the document.

**MF-4 — §4.3 reinstates a comparison the program retired, stripped of the normalization that retired it. §4.3 l.92–95.**
*"baseline networks (coRNN, LEM, LSTM) fail to maintain the stored analog phase beyond ≈5.6 to 69 map-steps … whereas our generic unit maintains retention for ≈263 map-steps."* `claims_matrix.md` **CM-4** is explicit: *"w8 amendment (SF-2): the '≈4× longer' RETIRED as a compute claim — per-step CLU Verlet (h64) costs 6.2× LSTM / 3.1× LEM wall (14–15× FLOPs; not width-matched), so retention-per-compute INVERTS (23.5×/14.6× more wall). **Lead with the qualitative triad (compute-independent)**."* The base honours this in a dedicated "Honest gap" paragraph. pj_sub deletes the paragraph and keeps the numbers, in the "whereas" construction that *is* the retired claim. **Failure scenario:** any reviewer asks the reflex question of this room — "what does the geometry buy that an architecture does not?" — computes 263/69 ≈ 3.8×, then asks about per-step cost, and the paper has no answer on the page. Worse for the Head: the answer exists, is measured, and inverts the sign.

**MF-5 — §4.3's negative is stated in exactly the universal form the base forbids, against published counterevidence. Abstract l.19–21; Contributions l.42–44; §4.3 l.90–92.**
pj_sub: *"we establish that symmetric training data does not inherently induce a flat direction in emergent architectures, confining the continuous register to designed geometries."* Base, same finding: *"This is a measurement on our architecture class and training recipe, **not a general statement that learning cannot produce a tuned flat direction — a local learning rule that does produce one is published (Vafidis et al. 2022)**."* **Failure scenario:** a NeurReps reviewer knows Vafidis et al. 2022 (a ring-attractor learning-rule paper squarely in this room's canon) and rejects the paper's third contribution with one sentence and one citation, and the paper has no defense on the page. Charter **C-5** violation compounded: the evidence is **one** MLP potential on **3 seeds** (base line 132: "$\mu^2=5.1/5.9/5.4\times10^{-2}$ on 3 seeds"), generalized to "emergent architectures," plural, scope-free. pj_sub also quietly reports only the single softest of the three seed values.

**MF-6 — "bit-identical" is factually wrong as widened. §4.4 l.99–101.**
pj_sub: *"the core retention laws and the exceptional-point onset remain bit-identical after 3000 anchored epochs."* Base line 140, verbatim: *"3000 anchored epochs, 3 seeds, curvature ratio exact to $1.5\times10^{-12}$ over 4.6 decades, **slope $-0.956$**, the same floor, and the **exceptional-point onset bit-identical at 0.5165**."* Only the *onset* is bit-identical. The retention slope moves **−0.985 → −0.956** (≈3%), the curvature ratio moves 5e-12 → 1.5e-12, the decade span moves 4.5 → 4.6, and the seed count drops 5 → 3. **Failure scenario:** the reviewer asks "bit-identical to what precision? §4.1 reports −0.985" and the honest answer is −0.956. That answer retroactively devalues every other exactness superlative in the abstract. Fix by restoring the base's own sentence, which is both stronger and true.

**MF-7 — The paper claims a "closed-form" two-branch law and prints only one of the three closed forms. §3 l.66–69; §4.1 l.76–81.**
§3 gives the overdamped branch in closed form. The floor is described only as "a curvature-independent envelope," with the *value* 27.03 appearing 12 lines later; the base gives the closed form $2\ln2/(-\ln(1-\gamma))$. The crossover is given only as "$\varepsilon\mu \approx \gamma/2$"; the base carries it as a function $h^*(\gamma)$. **Failure scenario (checkable, and a careful reviewer will do it):** plug the paper's own numbers into the paper's own formula. At $\gamma=0.05$, the printed overdamped branch equals the printed floor of 27.03 at $\varepsilon\mu = 0.0363$, **not** at $\gamma/2 = 0.025$ — a 1.45× gap between the stated crossover and where the two stated branches actually meet. In the base this is invisible because Figure 1 shows the real join; here there is no figure, so the approximation becomes an apparent internal inconsistency the reviewer discovers with a calculator. Either print $h^*(\gamma)$ exactly, or print the floor's closed form and say the "$\approx\gamma/2$" is an order-of-magnitude locator.

**MF-8 — The abstract's headline divergence number never reappears in the results. Abstract l.18–19 vs §4.2 l.86–88.**
Abstract: "diverges by $\approx 3.2\times$ above the crossover." §4.2: "a $2.2\times$ exceptional-point delay spike and declining to $0.31\times$ deep within the underdamped regime." The reconciliation ($1/0.309 \approx 3.2$, at $\delta=4$, the deepest breaking tested) is in the base at line 122 and is **absent here**. **Failure scenario:** reviewer lists three numbers — 3.2×, 2.2×, 0.31× — notes the abstract's number appears nowhere in §4, and flags the paper as internally inconsistent. It is not inconsistent; it is unreconciled. One clause fixes it. ⚠ Also carry CM-4's never-quote fine print: the ≈3.2× is the **trained-model** number; the "≈5×" is exact-map only.

**MF-9 — No figure. Whole document; 5 rendered PNGs sit unused in the same folder.**
`grep -c "\\includegraphics"` = **0**, while `figs/` contains `fig1_gmor.png` (the price list), `fig_lifetime_headtohead.png` (the head-to-head), and three more. The paper claims a two-branch law with a crossover, a floor and a second-order signature, and shows the reader **nothing**. At NeurReps — a geometry-and-representations room whose accepted sets are visual — a figure-free 2.6-page submission reads as unfinished. **Failure scenario:** reviewer scores "presentation: 2/5, no figures" and, on the substance, writes "I am asked to accept a claimed crossover on the basis of two numbers in running text." Figure 1 alone would also dissolve MF-7.

**MF-10 — Terms are used before (or without) definition; the paper does not stand alone. Throughout.**
Concrete inventory, each a place a fresh reader stops:
- **The task is never described.** What are these models trained on? "symmetric training data" (§4.3), "an $S^1$ testbed" (abstract), "150 epochs" (§3) — there is no dataset, no objective, no task anywhere in the paper. The base names it ("the preprint's $S^1$ path-integration family").
- **$n_{1/2}$ is never defined** — half-life *of what*, measured *how*, in what units. "map-steps" first appears at l.93, in §4.3.
- **Which retention metric?** The base states, at the point of measurement, that envelope half-life and first-crossing time split by 3.2× underdamped, *"which is why every lifetime in this paper names its metric."* pj_sub names no metric anywhere, while quoting lifetimes to four significant figures (27.03).
- **The fitted slope's abscissa is never stated.** "$n_{1/2}\propto\mu^{-2}$ … a fitted overdamped slope of $-0.985$ against a predicted $-1$" — slope in $\log n_{1/2}$ vs $\log\mu$ would be −2, vs $\log\mu^2$ would be −1. The reader cannot check −0.985 against −1. (The base's `n=35` is also dropped, so the fit has no sample size.)
- **"4.5 decades" of what?** The independent variable is the analytic tilt $\delta$; **$\delta$ is never introduced in pj_sub**. §4.1 says only "we tilt the trained potential by a known amount."
- **$\varepsilon$ is never given a value** (is $\varepsilon = \mathrm{dt} = 0.05$?), so $\varepsilon\mu \approx \gamma/2$ is uncheckable. **$k$ and $m$ in "$\mu^2 = k/m$" are undefined**; "spectral mass," "learned-Newtonian kinetic term," "order parameter," "$V(\text{data})$ energy anchor" and "Wake-Sleep contrastive divergence" all arrive with no setup (the reader has not been told the model is trained by wake–sleep at all).
- **"exceptional point"** is used in §4.2 and §4.4 but never introduced; §3 calls the same object "the crossover."
- **$G$ is never instantiated.** "For a $G$-invariant potential" — in a symmetry-and-equivariance room, name the group ($SO(2)$/$U(1)$) once.
- **Three names for two objects:** "designed flat mode," "emergent architectures," "our generic unit," "designed geometries." The designed-vs-emergent-vs-generic distinction is load-bearing for §4.3 and §4.4 and is never defined.

**MF-11 — Prior-art credit for the anchor was deleted, so the anchor now reads as ours. §4.4 l.97–99.**
Base: *"That a corrective term can keep a flat direction alive is not new (Renart, Song & Wang 2003). What we add is the third part of the price list."* pj_sub: *"However, introducing a $V(\text{data})$ energy anchor successfully restores and maintains the symmetry breaking."* The disclaimer and the citation are both gone; the "what we add" boundary is gone. **Failure scenario:** a computational-neuroscience reviewer — and this room is full of them; the Findings advisory board is entirely systems neuro — recognizes homeostatic restoration of a fine-tuned attractor as a 20-year-old idea and marks the paper for claiming it. §2 even *mentions* that "restorative homeostatic mechanisms are standard in the literature," which makes §4.4's unattributed presentation worse, not better.

---

## SHOULD-FIX

**SF-1 — Charter C-2 is violated in the abstract, and it hands the room its own reflex objection.** The base labels §4.1 verbatim: *"These are trained checkpoints carrying analytic tilts: **verification of the theory's exactness on a learned potential, not a discovery**,"* and the Figure 1 caption repeats it. pj_sub deletes the label and promotes "$1.000000 \pm 5\times10^{-12}$ over 4.5 decades" to the abstract as headline evidence. That is the G1 attack ("a unit test on a testbed built to satisfy the theory") delivered by the authors themselves. Restore the six-word label; it converts the strongest attack surface into a credibility signal.

**SF-2 — The learned/anharmonic arm — the *evidence*-grade result — is gone entirely.** The base's price-list block carries both the designed-vacuum verification and the learned arm. pj_sub keeps only the designed one and then titles itself *"…in a Trained Recurrent Memory."* The title promises the arm the paper cut. Either restore one learned-vacuum number or soften the title.

**SF-3 — Zero 2025–26 neighbours are named; the positioning built for this room was deleted.** The base names Keller 2025 / Lillemark et al. 2026 ("they generalize, we price"), the Symmetry-Regularized Continuous Attractors NeurReps-2025 poster as the emergent-arm competitor, Huang/Singh/Martinelli/Rajan 2025 (solution degeneracy) as the instrument a referee will demand, and **Haputhanthri et al. 2025 — which includes a NeurReps organizer, and whose arrow §4.4 runs backwards.** pj_sub names nobody. "The closest parallel to our framework is flow equivariance" without naming Keller is the worst of both worlds: it advertises that the authors know the neighbour and did not cite it. ⚠ **In a room where you can name your reviewers' own live line and it is directly relevant, not naming it is a self-inflicted wound.**

**SF-4 — §4.2's actual punchline was cut.** The base closes the head-to-head with the reason it matters: *"the regime structure — floor, ringing, exceptional point — is what a first-order dynamical system cannot exhibit and what a **constitutive** curvature adds. This is containment, not conflict."* pj_sub ends at "However, it fails above the crossover." As written, §4.2 reads as "we found a bug in someone else's estimator," which is a worse paper than "their estimator is one face of a two-face law." One sentence, high value.

**SF-5 — Every cost of the anchor was deleted.** Base: $\lambda=100$, 5/5 seeds, $r^*=0.911\pm0.016$, *"at the cost of weaker noise rejection and $\approx35\times$ higher wake MSE,"* plus the travelling scope clause on `sleep_steps`. pj_sub reports only "successfully restores and maintains." A track that advertises negative findings rewards the cost sentence; hiding it invites "the authors report only the success case."

**SF-6 — The "$\approx 20\times$" cannot be reconstructed by any reader. Abstract l.15–16.** *"operating at approximately $20\times$ the horizon over which the objective would otherwise degrade it."* The paper never says what the erosion horizon is. If it is the 150 training epochs, $3000/150 = 20$ ✓ — but the base says contrastive divergence inverts the vacuum *"in 8/8 runs at the default 1000 epochs,"* and $3000/1000 = 3$. **I cannot tell from either text which horizon the 20× is against.** State the erosion horizon in map-steps or epochs next to the number.

**SF-7 — No flag-provenance table (Charter C-7); the scale qualifiers are scattered and partly wrong-place.** §3's "Operational Assumptions" gives dim 4 / width 64 / dt 0.05 / 150 epochs but no $\gamma$, no $\varepsilon$, no seeds; $\gamma=0.05$ first appears at l.81 in Results; "$\le 5$ seeds" appears only in §5, *after* every number it governs. And "$\le 5$ seeds" is a ceiling that conceals that the two most contestable arms — the emergent negative and the anchor survival — are **n = 3**. A four-row table costs ~6 lines of a page you are not using.

**SF-8 — Anti-strawman evidence for the baselines was cut. §4.3.** Base: *"the baselines are trained with a learning-rate sweep, best RMSE kept, and are not strawmen: LSTM and LEM reach train-horizon RMSE 0.18/0.23 rad."* Without that clause, "coRNN/LEM/LSTM fail beyond ≈5.6 to 69 map-steps" is the textbook shape of an undertrained-baseline result, and this room says so reflexively. Restoring one clause converts a suspicion into a strength. (⚠ Also restore CM-4's scope: S¹ protocol, map-steps ≠ wall-time, no input-driven task-RMSE axis.)

**SF-9 — "contracting phase-space volume by $(1-\gamma)^d$ per step while preserving geometry" is self-contradictory as phrased. §3 l.63–65.** A physics-literate reviewer in this room (Noether and symplectic integrators are attested 2024 NeurReps subject matter) will read "preserves geometry" next to "contracts volume" and stop. The correct term is *conformally symplectic* — preserves the symplectic form up to a constant factor. One word, and it is vocabulary this audience already has.

**SF-10 — No appendix, in a track whose page limit excludes appendices, when an 18-page version exists in the same folder.** Charter **C-10** (appendix maximalism) and the venue rule point the same way. At minimum: the flag-provenance table, the head-to-head protocol constants, the anchor costs, and the compute normalization for MF-4 — all of it already written in `submission.tex`. This is pure wiring, not new work.

**SF-11 — 2.6 of 4 pages used, and the abstract carries the whole paper.** The 21-line abstract states nearly every number, which §4 then restates verbatim. The compression optimized against a budget that was not binding.

---

## NICE

- **N-a — Lead the results with the head-to-head** (Charter C-3: *"The Mo head-to-head — 'a published ML result is our overdamped face' — leads the results"*). As ordered, §4.1 is a 12-digit self-consistency check and §4.2 is the external validation; the room will value them in the reverse order.
- **N-b — The negative is the most track-native content and it is third.** The track text names "negative findings" and 2025 accepted a paper literally titled "…An Observation and a Negative Result." Promoting §4.3 would fit the venue better than a curvature identity.
- **N-c — Consider adopting "pseudo-gap"** (the refresh's §3.1 recommends *adopt the word, and cite*). The current draft's total avoidance is defensible only if the citation is present; with no citation it just removes the room's own handle on the object. Head's call — this collides with the strictest reading of the N1 fence.
- **N-d — Title check:** "in a Trained Recurrent Memory" over-promises given SF-2.
- **N-e — Abstract sentence 4** ("how effectively a written value survives infinitesimal perturbations") conflates perturbation-robustness with curvature-induced decay; the paper measures the latter.
- **N-f — Cosmetic:** one underfull `\hbox` in the abstract (badness 2229, l.31).
- **N-g — Typography drifted from the base's own discipline.** `\emph` = **0** and `\textbf` = **9** in main text, where the base swept specifically for *"`\textbf` before `\appendix` → 0 hits"* and used `\emph` for ABT signposting throughout. Acceptable in an extended abstract, but it is why the draft reads as a bulleted summary rather than as prose: §3 and §5 are entirely bold-labelled `itemize` blocks, and §5's three bullets are the paper's limitations — the one place this track rewards full sentences.

---

## Missing-experiment list for the Hub

Distinguishing **wiring** (exists in `outputs/`, absent from this draft) from **genuinely missing**:

**WIRING ONLY — nothing new to run, all of it already in `submission.tex`/`outputs/` (MF-2, MF-4, MF-6, MF-8, SF-1…SF-8, SF-10).** The head-to-head protocol constants and N1's own 1.013 median; the compute normalization (6.2×/3.1× wall, 14–15× FLOPs, 23.5×/14.6× per-retention); the anchored-run true numbers (−0.956, 1.5e-12, 4.6 decades, 3 seeds); the 3.2×/0.31× reconciliation at δ=4; the Vafidis 2022 scope clause; the baseline LR-sweep/RMSE anti-strawman clause; the anchor's λ=100 / r*=0.911 / 35× cost; the verification-vs-evidence labels; the four 2025–26 neighbours. **Recommend the Hub issue one curator/writer task, not a research task.**

**GENUINELY MISSING (Hub task candidates, none blocking for an EA):**
1. **N4 solution-degeneracy applied to the designed-vs-emergent gap** (Huang, Singh, Martinelli & Rajan 2025). The base itself concedes *"we do not run it here."* This is the instrument this room will ask for, and it is the only clean way to separate "recipe" from "architecture" in MF-5. Feeds the longs.
2. **A second emergent architecture for the §4.3 negative.** One MLP potential on 3 seeds cannot support a claim about "emergent architectures." Even two more potential parametrizations would turn MF-5's overclaim into a defensible bounded one.
3. **A learned/anharmonic price-list arm at EA scale** so the title's "Trained" is earned by a main-text number (SF-2).
4. **Beyond $S^1$** (register G7a: $T^2$ / $SO(3)\to SO(2)$). Not an EA blocker; the standing longs mandate.
5. **The erosion horizon measured and reported as a number**, so the ≈20× has a denominator (SF-6).

---

## The three sentences a hostile reviewer would quote back

1. > *"The retention law yields a curvature ratio of $1.000000 \pm 5 \times 10^{-12}$ across $4.5$ decades."*
   — "Twelve significant figures of agreement between a theory and a testbed the authors constructed to satisfy it, presented as the paper's headline result. This is a unit test, not evidence."

2. > *"We validate these dynamics against a published single-exponential lifetime estimator."*
   — "The paper contains no references. I am asked to accept its central empirical claim against an instrument the authors decline to name."

3. > *"Finally, we bound this framework by demonstrating that symmetric training data alone does not yield a near-flat direction on emergent architectures without explicit design."*
   — "One MLP, three seeds, one testbed, and a universal claim about learned architectures — contradicted by published local learning rules that do produce tuned ring attractors."

*(Runner-up, and the one that would hurt most in discussion: "our generic unit maintains retention for ≈263 map-steps" against baselines' 69 — the authors' own longer version retires this ratio because it inverts under compute normalization.)*

---

## Does it stand alone? — direct answer to review item 1

**No.** A reader who has never seen the long version cannot: (i) say what task or data any model was trained on; (ii) say what $n_{1/2}$ measures or in what units; (iii) check the fitted slope against its prediction, because the abscissa is unstated; (iv) identify the "4.5 decades" variable, because $\delta$ is never introduced; (v) evaluate §4.2 at all, because its object is unnamed; (vi) reconcile the abstract's 3.2× with §4.2's numbers; (vii) reproduce the crossover, because $\varepsilon$ has no value. The claim and the scope *do* come through — the paper is honest about dimension 4, width 64, ≤5 seeds and laptop CPU, and §5 is a real limitations section, not a ritual one. It is the **evidence** that does not survive the compression, not the modesty.

## Fit for the track — direct answer to review item 7

**It reads as a compressed full paper, not as an extended abstract**, and that is the compression's signature failure. Tell: five numbered sections in the shape of a full paper (Intro / Related Work / Setup / Results with four subsections / Discussion), a 21-line abstract that pre-states every result, results reported as bare four-significant-figure values with no figure and no setup, and a Related Work written as a survey paragraph rather than as a positioning move. A native extended abstract would do the opposite — one claim, one figure, the negative up front, the scope loud, the apparatus deferred to an appendix that does not count against the limit. **The good news is that the content is genuinely track-appropriate** (small scale is accepted here; negatives are explicitly invited; physics machinery needs no apology); it is the *form* that is wrong, and the form is cheaper to fix than the content would be.

---

## What this version gained and lost against a longer treatment

**Gained — and these are real gains, not consolation.**
1. **A clean, single-sentence thesis.** "Transverse curvature sets retention, with a crossover and a floor" survives the cut intact and is far sharper here than in 18 pages. The title is the best asset in the document.
2. **Three contributions on page 1, enumerated.** A reviewer knows the shape of the paper by line 44.
3. **The band taxonomy** (latch / overdamped register / underdamped working memory) is the clearest statement of the idea anywhere in the program — it is the paper's best paragraph and it exists only in this version.
4. **The limitations section is short, prominent and un-hedged** — dimension 4, width 64, ≤5 seeds, laptop CPU, local to the critical point. Longer versions bury this; here it is a section.
5. **Hermeticity and anonymization are clean.** Zero hits for internal apparatus (`.claude`, CM-n, SF-n, PREREG, Hub, Advisor, CSF3, 13.9), zero for CLU/CHLU/Jawahar/Pierini, no companion-paper leakage, `\author{}` empty, no venue string, no PDF metadata. A reviewer cannot de-anonymize this from the text (M2/M3 clean).

**Lost — and the losses cluster, which is the important pattern.**
The cut fell almost entirely on **attributions, scope clauses and cost sentences** — i.e. on precisely the material that protects the paper — while the **superlatives were kept**. Concretely: every citation (11); every figure (5); the verification-vs-evidence labels; the "not a general statement about learning" clause; the "not strawmen" clause; the compute normalization; the anchor's price; the prior-art credit for corrective terms; the metric-naming discipline; the units disambiguation between $\lambda$ (1/time) and $\mu^2$ (1/time²); and the four 2025–26 neighbours. Meanwhile "1.000000 ± 5×10⁻¹²", "bit-identical", "perfectly matches", "≈263" and "we establish" all survived — several of them *widened* in the process (MF-5, MF-6). **A compression that removes the hedges and keeps the claims does not produce a shorter paper; it produces a more aggressive one.** That is the whole diagnosis.

The cost was also unnecessary. The EA limit is 4 pages **excluding references and appendices**; this build uses **2.6 pages and has neither**. Roughly a page of main text and an unlimited appendix were available for free. My estimate of the restoration: a bibliography (~15 entries), Figure 1 and the head-to-head figure, the four scope clauses at MF-4/MF-5/MF-6/MF-11, the estimator's identity and protocol at MF-2, the abstract attribution at MF-3, and a provenance table — **all of it already written and measured**, none of it requiring a new run. That is the difference between the reject above and a weak-accept.

---

## Coverage check against the standing register (this draft, not the program)

| ID | attack | status in `pj_sub` |
|---|---|---|
| **G1** | unit test on a testbed built to satisfy the theory | ⛔ **worse than the base** — the base's own disarming label was deleted (SF-1); the 12-digit identity is now the abstract's headline |
| **G2** | which component buys what | ⚠ partially answered by §4.3's baselines, but with the retired ratio and no compute normalization (MF-4); no minus-the-physics arm cited |
| **G3** | toy scale | ✅ **handled well** — dim 4 / width 64 / ≤5 seeds / laptop CPU stated in abstract and §5; venue tolerates |
| **G5** | certificate fine print | ⚠ "bit-identical" (MF-6) and "perfectly matches" (§4.1) are altitude violations of the C-6 spirit |
| **G6** | foundational-paper falsifications | ✅ no audit confession, no legacy-number load-bearing (C-1 clean) |
| **M2/M3** | de-anon / salami optics | ✅ clean: no cross-short references, no program vocabulary, no author tokens |
| **M4/C-7** | flag provenance | ⛔ absent (SF-7) |
| **C-5** | scale qualifiers on generalizing claims | ⛔ violated at MF-5 ("emergent architectures"); partially at §4.3's baseline sentence |
| **C-9** | negatives documented, prominent | ⚠ present but third, and over-generalized (MF-5) |
| **C-10** | appendix maximalism | ⛔ no appendix (SF-10) |

---

## Proposed handover updates (for the Hub)

1. **`pj_sub.pdf` does not exist.** Any downstream gate keyed to it (including the fidelity pass) should be re-checked; the Head may not have built. My review is against a tectonic build of `pj_sub.tex` (sha256 `4aafc2c2…14bb3`), 3 pages.
2. **One curator/writer task, not a research task, closes most of this.** MF-1/2/3/4/5/6/8/9/11 and SF-1…SF-8/SF-10 are all restorations from `submission.tex` (same folder) plus `figs/`. Recommend a single `pj-sub-restore` task with the itemized list above as its checklist. **None of it costs main-text budget** (refs and appendices are excluded from the EA 4-page limit; the build uses 2.6 pp).
3. ⛔ **MF-3 is the escalation item.** The abstract asserts arXiv:2605.03338's theorem with no attribution and no citation, and §2 reuses that preprint's verbatim title phrase. The BUILD-NOTE §3 records that the base deliberately puts this attribution *in the abstract, before any of our results*; the condensation removed it. Given the deadline (Aug 24 AoE) and that Mo is a plausible reviewer in this room, this is Head-level, not writer-level.
4. **MF-4 is a claims-matrix breach, not a style note.** CM-4's w8/SF-2 amendment retires the 263-vs-69 ratio as a compute claim. Either restore the "Honest gap" normalization or drop the "whereas" comparison and keep only the qualitative triad, which CM-4 designates as the load-bearing (compute-independent) claim.
5. **MF-6 is a number contradiction of the MF-3/M4 class** the register exists to prevent (cf. v2-referee's MF-3 precedent): §4.4's "bit-identical" is inconsistent with §4.1's −0.985 once the anchored slope (−0.956) is known. Flag for the claims-consistency sweep.
6. **Convergence note for the Head:** where this report and `pj-fidelity-v2.md` agree, treat the finding as confirmed by two independent passes; where only one fires, the other pass's silence is not evidence of absence (I reviewed the artifact as received, not the diff).

## Flags

- **F1** — Precondition failure (`pj_sub.pdf` missing); reviewed a self-built tectonic PDF. Pagination is mine, not the Head's; the 3-page/2.6-page figures should be re-confirmed on a pdflatex build before anyone acts on the "unused budget" argument.
- **F2** — I did not verify Vafidis et al. 2022, Renart/Song/Wang 2003 or Ságodi et al. 2024 at source; I take them from `submission.tex`'s own citations. The *argument* (that these attributions were deleted) does not depend on their correctness.
- **F3** — The MF-7 arithmetic (branches meeting at $\varepsilon\mu = 0.0363$, not $\gamma/2 = 0.025$) is my own calculation from the two formulas printed in `pj_sub` §3 and the floor value in §4.1. It reproduces the floor exactly ($2\ln2/(-\ln 0.95) = 27.03$ ✓), which is evidence the formulas were read correctly, but the true $h^*(\gamma)$ of the map may legitimately differ from the naive branch intersection. Framed as "the paper as printed does not let a reader resolve this," not as "the paper is wrong."
- **F4** — Two `grep -o -E` sweeps over `pj_sub.tex` hit catastrophic backtracking, were moved to background, and **failed (exit 2)**. I did not let any claim rest on them: every count below was **re-verified with a Python literal-substring pass, positive-controlled**, and is the number quoted in this report.
  `\cite` **0** · `\includegraphics` **0** · `\begin{table}` **0** · `\bibliography` / `thebibliography` **0** · `\appendix` **0** · `\ref{` **0** · `pseudo-gap` **0** · `lifetime law` **0** · `CLU`/`CHLU`/`Jawahar`/`arXiv` **0** · `.claude`/`Advisor`/`PREREG`/`13.9` **0**.
  **Positive controls fire:** `curvature` **11**, `\textbf` **9**, `Goldstone` **1**, `seeds` **1**. (Per the standing lesson on gitignored-dir sweeps, no negative here is unaccompanied by a positive control.)
  ⚠ Two of these controls are themselves findings: **`seeds` occurs exactly once in the whole paper**, in §5, after every number it governs (SF-7); and **`\emph` occurs zero times** while `\textbf` occurs nine — see N-g.
- **F5** — The venue facts I weigh (EA = 4 pp excl. refs + appendices, non-archival, double-blind; track purpose text; 2025 accepted-set character) are taken from `outputs/n1-fulltext-and-track-check.md` §Q2 and `outputs/audience-refresh-2025-2026.md` §1.8 (retrieval 2026-08-21), not re-verified by me.
