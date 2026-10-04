# pj-fidelity-v2 — doc-curator report

Task + acceptance criterion: fidelity-audit the Head's `NIPSsubmission/v2-neurreps/pj_sub.tex` against the Advisor-accepted clean base `submission.tex` (Part A: numbers + claim equivalence + attribution + author-name rule; Part B: mandatory-content losses; Part C: render `pj_sub.pdf`), with `pj_sub.tex` byte-unchanged.
Status: **partial** — Parts A and B complete; ⛔ **Part C BLOCKED: this spoke has no shell/execution tool** (tools available: Read/Write/Edit/Grep/Glob only), so `pdflatex` could not be run and no `md5` could be computed. Everything Part C needs is pre-analysed in §C below so the next agent (or the Head) can build it in one command.
**DIAL DECLARATION (echoed): none — verification/audit pass; no performance claim; no laundering control applies.**

⭐ **THIS REPORT CONTAINS A DOWNSTREAM RECONCILIATION LIST AND IT NEEDS AN OWNER** (protocol §5 corollary): §A-VERDICTS rows A1–A6 and §B rows B1–B8 are **edits the Head owes their own file**. They are stated as findings, never applied. ⛔ Nothing in `pj_sub.tex` was touched.

**Headline for the Hub, in one line:** *no number in `pj_sub.tex` is fabricated and no number is mis-transcribed in any digit — the file's fidelity problem is not arithmetic, it is (i) four claim-strength upgrades (three of them ⛔ WIDER, one CHANGED IN KIND), (ii) the total loss of the citation apparatus, which leaves two results that are explicitly **not ours** reading as unattributed background, and (iii) the removal of every mandatory rider, including the compute-normalization retirement (CM-4) and the no-external-benchmark score sentence.*

---

## 0. What I compared, and the discipline used

| item | value |
|---|---|
| audited file | `.claude/NIPSsubmission/v2-neurreps/pj_sub.tex` (84 lines, pure ASCII, single file) |
| source of truth | `.claude/NIPSsubmission/v2-neurreps/submission.tex` (422 lines; Advisor-accepted at Add.52) |
| registries consulted | `claims_matrix.md` §0.1–§0.14 + §2 rows CM-1/CM-4/CM-6/CM-15/**CM-16a/b**/CM-21/CM-22; `negative_results.md` **N46**; `advisor-head-shorts-charter.md` **Add.49 / 50 / 51 / 52 / 53** |
| survival ratio (per Add.53, not re-measured) | 1,263 words vs 10,369 (~12 %) |
| structural inventory of `pj_sub.tex` | **0 `\cite`, 0 `\label`, 0 `\ref`, 0 `\includegraphics`, 0 References section, 0 appendices** (grep, printed in §B1/§B7) |
| ⛔ edit check | see §C4 — no write was attempted; the file is unchanged at end of pass, verified by re-read, **not** by md5 (no shell) |

Numbers are quoted **exactly** as they appear in each file, with line numbers of both.

---

## A. PART A — is what survived FAITHFUL?

### A.1 Numbers, exactly — the full token ledger

Every numeric token in `pj_sub.tex`, enumerated by hand (line-by-line read, not a script), matched against `submission.tex`.

| # | `pj_sub.tex` token (line) | source token (line) | verdict |
|---|---|---|---|
| 1 | latent dimension `$4$` (30, 56, 76) | "latent dimension $4$" (30); "dim $4$" (100) | **exact** |
| 2 | hidden width `$64$` (56, 76) | "hidden $64$" (100) | **exact** |
| 3 | `$\mathrm{dt}=0.05$` (56) | "$\mathrm{dt}=0.05$" (100) | **exact** |
| 4 | `$150$` epochs (56) | "$150$ epochs" (100) | **exact** |
| 5 | fitted overdamped slope `$-0.985$` (30, 62) | "$-0.985$" (30, 112) | **exact** |
| 6 | predicted `$-1$` (30) | "predicted $-1$" (30, 112) | **exact** |
| 7 | `$1.000000 \pm 5 \times 10^{-12}$` (30, 62) | "$1.000000\pm5\!\times\!10^{-12}$" (30, 112) | **exact** (incl. the ±) |
| 8 | `4.5` decades (30, 62) | "$4.5$ decades" (30, 112) | **exact** |
| 9 | crossover `$\varepsilon\mu \approx \gamma/2$` (30) | "$\varepsilon\mu\approx\gamma/2$" (30) | **exact** |
| 10 | `$20\times$` the horizon (30) | "$\approx20\times$ the horizon" (30) | **exact** |
| 11 | `$\le 3\%$` deviation (30) | "matching to $\le3\%$ below the crossover" (30) | **exact** |
| 12 | `$\approx 3.2\times$` above the crossover (30) | "mispredicting by $\approx3.2\times$ above it" (30) | **exact** (⚠ scope, see A2-A7) |
| 13 | `$(1-\gamma)^d$` (54) | "$\det J=(1-\gamma)^d$" (96) | **exact** |
| 14 | `$\mu^2 = k/m$, $h = \varepsilon\mu$` (55) | "($\mu^2=k/m$, $h=\varepsilon\mu$)" (96) | **exact** |
| 15 | overdamped band `$0 < \varepsilon\mu \lesssim \gamma/2$`, `$n_{1/2} \approx 2\gamma\ln 2/[(2-\gamma)(\varepsilon\mu)^2]$` (55) | identical formula (96) | **exact** |
| 16 | underdamped band `$\gamma/2 \lesssim \varepsilon\mu < 2$` (55) | identical (96) | **exact** |
| 17 | `$\mu^2 \le 2.4 \times 10^{-15}$` (62, 68) | "$\mu^2\le2.4\!\times\!10^{-15}$" (112, 132) | **exact** |
| 18 | `$27.03$`-step floor at `$\gamma=0.05$` (62) | "floor $27.03$ steps at $\gamma=0.05$" (112) | **exact** |
| 19 | `$2.2\times$` EP delay spike (65) | "the $2.2\times$ exceptional-point delay spike" (122); full value `2.202\pm0.155` (122) | **exact** (± dropped — see B4) |
| 20 | `$0.31\times$` deep underdamped (65) | "declining to $0.31\times$" (126); full value `0.309\pm0.012` (122) | **exact** (± dropped — see B4) |
| 21 | softest emergent `$\mu^2 \approx 5.1 \times 10^{-2}$` (68) | "$5.1/5.9/5.4\!\times\!10^{-2}$ on $3$ seeds" (132) | **first of three seeds only; `n=3` dropped** — see A2-A4 |
| 22 | `$\approx 5.6$ to $69$ map-steps` (68) | "$\approx5.6/56/69$ map-steps" (134) | **endpoints exact; the middle value `56` (LEM) is absent** — see A2-A5 |
| 23 | `$\approx 263$` map-steps (68) | "$\approx263$ map-steps" (134) | **exact** |
| 24 | `$3000$` anchored epochs (71) | "$3000$ anchored epochs" (140) | **exact** |
| 25 | `$\le 5$ seeds` (76) | "$\le5$ seeds" (30, 145) | **exact** |
| 26 | `$T>0$` (56) | "$T>0$" (103) | **exact** |

⭐ **The most serious finding available here is EMPTY: there is no number in `pj_sub.tex` without an ancestor in `submission.tex`.** No value, no precision, no unit, no exponent and no ± is altered in any digit anywhere. Two tokens are *reductions* of a multi-seed set (#21, #22) and two lose their ± (#19, #20); those are adjudicated as claim-scope items below, not as transcription errors.

### A.2 Claim equivalence — both texts quoted, verdict vocabulary exact

**A2-A1 — the constitutive identity. ⛔ WIDER (a misrepresentation).**
- source (112): *"the constitutive identity holds: the Hessian-derived spectral mass **equals the measured one-step-Jacobian gap to $3.2\times10^{-10}$ over all $70$ tilt rows**."*
- `pj_sub` (62): *"the Hessian-derived spectral mass **perfectly matches** the measured one-step-Jacobian gap."*
- **Ruling: WIDER.** A quantified finite-precision agreement (`3.2e-10`, n=70 rows) becomes an unbounded qualitative absolute ("perfectly"). This is exactly the *"a bound becoming a point estimate"* drift mode in its worst form — the bound becomes *no* bound. The paper's own Appendix F.1 precision fine print (413) exists to stop precisely this reading (*"Do not quote '2.2e-16 relative' for this experiment"*; the `ε/δ` roundoff floor governs **every** `μ²` probe in the paper). Sample size `70` also disappears. **Head owes their file one edit: restore "to $3.2\times10^{-10}$ over all $70$ tilt rows" or delete "perfectly".**

**A2-A2 — the anchored survival. ⛔ WIDER (a misrepresentation), and the most quantitatively wrong sentence in the file.**
- source (140): *"every headline law of it still holds under the correction, at $\approx20\times$ the erosion horizon — $3000$ anchored epochs, $3$ seeds, curvature ratio exact to $1.5\times10^{-12}$ over $4.6$ decades, **slope $-0.956$**, the same floor, **and the exceptional-point onset bit-identical at $0.5165$**."*
- `pj_sub` (71): *"the **core retention laws and** the exceptional-point onset remain **bit-identical** after $3000$ anchored epochs."*
- **Ruling: WIDER.** In the source **exactly one** quantity is bit-identical — the EP onset (`0.5165`). The retention law is *not*: its slope **moves from `-0.985` to `-0.956`** (and Appendix A's figure caption, 215, reports `-0.961` for the same run), the ratio from `5e-12` to `1.5e-12`, the span from `4.5` to `4.6` decades. `pj_sub` extends "bit-identical" across a set of quantities that measurably changed. `3` seeds is also dropped.
- ⚠ **Flagged, not resolved (source-internal):** `submission.tex` itself reports the anchored slope as **`-0.956`** in §sec:cure (140) and **`-0.961`** in the Appendix-A caption (215). Two different digits for one quantity in the accepted base. This is a pre-existing discrepancy in the Head's *source*, not something `pj_sub` introduced; the Hub owns which is right.

**A2-A3 — the anchor cure. ⛔ WIDER.**
- source (140): *"A $V(\text{data})$ energy anchor holds the condensate up --- $\lambda=100$: $5/5$ seeds, $r^*=0.911\pm0.016$, **at the cost of weaker noise rejection and $\approx35\times$ higher wake MSE**."*
- `pj_sub` (71): *"introducing a $V(\text{data})$ energy anchor **successfully restores and maintains** the symmetry breaking."*
- **Ruling: WIDER.** A result carrying two named costs becomes an unqualified success. This is the *"no harm becoming an improvement"* drift mode. `λ=100`, `5/5`, `r*=0.911±0.016` all drop with it (CM-6's canonical envelope).

**A2-A4 — the emergent negative's scope. ⛔ WIDER, twice, and it collides with N46.**
- source (132): *"**On this architecture class** it does not. A generic MLP potential trained on that data develops no near-flat direction: softest emergent $\mu^2=5.1/5.9/5.4\times10^{-2}$ **on $3$ seeds** … a gap of $13$--$14$ orders … **This is a measurement on our architecture class and training recipe, not a general statement that learning cannot produce a tuned flat direction --- a local learning rule that does produce one is published (Vafidis et al. 2022).**"*
- `pj_sub` abstract (30): *"we bound this framework by demonstrating that symmetric training data alone does not yield a near-flat direction **on emergent architectures** without explicit design."* / contributions (42): *"**We establish** that symmetric training data does not inherently induce a flat direction **in emergent architectures**."*
- **Ruling: WIDER.** "on this architecture class / our recipe" → "on emergent architectures" (unbounded plural) + "We establish". A 3-seed measurement on one MLP recipe now reads as a general statement about emergent architectures — the exact reading N46's disposition forbids (*"it must travel with every CM-16(a) citation"*), and the exact reading the source's Vafidis counterexample exists to block. **The rider is gone (B2) and the claim is simultaneously widened; these two findings compound.**
- ⚠ **Flagged, not resolved (registry-vs-source):** `submission.tex` (132) gives softest emergent `μ² = 5.1/5.9/5.4e-2`; **`negative_results.md` N46 gives `5.449e-2 / 2.029e-2 / 5.132e-2`** (`v5-gate` §3.2). `pj_sub`'s `≈5.1e-2` has a clean ancestor in the *paper*, so Part A is satisfied — but the paper and the registry do not agree on the middle seed by a factor ≈2.9. **I do not resolve this; the Hub owns it.** (If the registry is right, the source's "13–14 orders" span and its seed ordering both want re-checking.)

**A2-A5 — the baseline comparison. ⛔ CHANGED IN KIND (this is the top-ranked defect in the file; see B1).**
- source (134, 136): *"**the structural triad of latch, $\mu^{-2}$ law and bounded motion … is qualitatively absent in them**: coRNN/LEM/LSTM forget the stored analog phase in $\approx5.6/56/69$ map-steps … while the generically-trained unit holds $\approx263$ map-steps … **Honest gap.** The load-bearing claim is the qualitative triad, architecture- and compute-independent. **The raw $263$-versus-$69$ map-step ratio does not survive compute normalization, and we retire it as a compute claim**: per step the Verlet update costs $\approx6.2\times$ an LSTM cell and $\approx3.1\times$ a LEM cell in wall time, $\approx14$--$15\times$ FLOPs … $\approx23.5\times$/$\approx14.6\times$ more wall time than the baselines it out-holds."*
- `pj_sub` (68): *"baseline networks (coRNN, LEM, LSTM) fail to maintain the stored analog phase beyond $\approx 5.6$ to $69$ map-steps and are inherently perturbation-fragile, whereas our generic unit maintains retention for $\approx 263$ map-steps with bounded coordinate motion."*
- **Ruling: CHANGED IN KIND.** The source's claim is a **qualitative structural absence**, explicitly compute-independent, with the raw ratio **retired**. `pj_sub` keeps only the raw ratio (`263` vs `5.6–69`) and drops the retirement, the per-step factors and the whole Honest-gap paragraph — so a **retired compute claim is reinstated as the paragraph's headline**, and it stands with no normalization anywhere in the file. This is CM-4's amendment inverted (*"the '≈4× longer' RETIRED as a compute claim … Lead with the qualitative triad; state the per-step factor honestly if a compute line is kept"*) and it is the class of sentence CM-22 exists to forbid.
- Also lost with it: the baselines' non-strawman evidence (*"trained with a learning-rate sweep, best RMSE kept, and are not strawmen: LSTM and LEM reach train-horizon RMSE $0.18$/$0.23$ rad"*), the fragility numbers (`0.1` kick: LSTM `69→2`, LEM `56→5`), the `1.2` rad randomization, the unit's `≈0.35` rad bound and `+12`–`15 %` law agreement, and the designed unit's `5/5` seeds.

**A2-A6 — verification vs evidence. ⛔ CHANGED IN KIND.**
- source (50–51): *"Designed-testbed results --- invariant potentials, analytic tilts and spurions --- are **verification** of the theory's exactness; learned-system, training-dynamics and head-to-head results are **evidence**."* and at the site itself (112): *"**These are trained checkpoints carrying analytic tilts: verification of the theory's exactness on a learned potential, not a discovery.**"*
- `pj_sub`: the two-layer scheme appears **nowhere**; §4.1 (62) presents the tilt battery as a plain result ("The retention law yields …").
- **Ruling: CHANGED IN KIND.** A designed-testbed **verification** now reads as a measurement/discovery. This is the ledger's own status-tag discipline erased from the artifact, and it is the drift mode *"a designed-arm result reading as general"* in its structural form. Every figure label that carried `\emph{Verification.}` / `\emph{Evidence.}` went with the figures (B7).

**A2-A7 — claims that are FAITHFUL, recorded so the Head gets credit for them.**

| claim | source | `pj_sub` | verdict |
|---|---|---|---|
| the three bands (latch / overdamped register / underdamped working memory), with the `n₁/₂` formula and the `h<2` limit | 96 | 55 | **IDENTICAL** |
| conformal symplecticity as per-step volume contraction `(1−γ)^d`, "geometry preserved" | 96 | 54 | **IDENTICAL** |
| `μ⁻²` law, fitted `−0.985` vs predicted `−1`, curvature identity `1.000000±5e-12` over `4.5` decades | 30, 112 | 30, 62 | **IDENTICAL** |
| the crossover at `εμ≈γ/2` and the curvature-independent floor | 30, 96, 112 | 30, 55, 62 | **IDENTICAL** |
| SSB stated in the ordinary sense; the neutral coordinate is the Nambu–Goldstone mode of the trained potential | 35 | 34 | **IDENTICAL** (Add.52's audience scoping survives) |
| the units conversion is load-bearing: their λ is inverse time, our μ² inverse time squared, and we compare by *running their instrument*, not converting units | 37, 84, 122 | 48, 65 | **IDENTICAL** — ⭐ the single most important thing in the file, and it survived intact |
| "they generalize, we price" + **no generalization claim of any kind** | 64–66 | 48 | **IDENTICAL** |
| "flow" disambiguated as a one-parameter Lie subgroup acting through time | 61–62 | 48 | **IDENTICAL** (Add.52 change #3 survives) |
| we do not claim the zero-Lyapunov existence result | 79 | 48 | **IDENTICAL in wording** (⚠ but now uncited — A3-1) |
| the estimator is the overdamped face and fails above the crossover | 44–45, 122 | 41, 65 | **NARROWER (safe)** — "in the direction the list predicts" / the ballistic mechanism dropped |
| measured/predicted `≈1` overdamped | 122 (`1.012±0.000→1.029±0.001`; their median `1.013`; corr `0.9987`) | 65 | **NARROWER (safe)** — vaguer, not wrong; the `≤3 %` bound is retained in the abstract |
| scale qualifiers dim 4 / hidden 64 / ≤5 seeds / laptop-CPU | 30, 145 | 76 | **NARROWER (safe)** in placement — present once in Discussion, absent from the abstract and from every result (B3) |
| local-at-the-critical-point scope | 145 | 77 | **NARROWER (safe)** — the `+5` to `+29 %` anharmonic numbers dropped |
| finite-`T` ⇒ FDT-consistent noise **and** Newtonian kinetic mode | 103, 105, 145 | 56 + 77 (split across two sections) | **NARROWER (safe)**, both halves present; the `legacy`-default warning and the relativistic no-go are gone (B5) |
| the CD objective destroys a designed flat mode | 140 | 71 | **NARROWER (safe)** — "rapidly degrades … effectively destroying" vs `r*→0`, `8/8` runs, `1000` epochs; ⚠ abstract (30) softens "destroy" to "degrade" |
| closing positioning: "latches and registers within physics-structured associative memories" | 92 (CM-21 approved replacement wording), 148 | 81 | **NARROWER (safe)** — compatible with CM-21's approved form; the controllability-not-capability sentence is gone (B6) |

### A.3 Attribution and priority (Add.49 N1 scoping · Add.51 author rule)

**A3-1 ⛔⛔ THE CITATION APPARATUS IS ENTIRELY ABSENT — 0 `\cite`, no References section — and two of the three things Add.49 rules are *not ours* now appear as unattributed statements.**

| Add.49 item | status in `pj_sub.tex` | verdict |
|---|---|---|
| the **zero-Lyapunov-exponent theorem** ("still theirs, cite and never claim") | abstract (30): *"Exact equivariance protects these neutral directions, **yielding zero Lyapunov exponents tangent to the group orbit**."* — declarative, **no citation**, in our own abstract, immediately followed by *"Therefore, we measure…"*. Related Work (48) does keep the fence: *"recent proofs … establish that exactly equivariant fields possess zero Lyapunov exponents tangent to the orbit. **We do not claim this existence result**"* — but "recent proofs" names nobody. | ⛔ **the fence survives in §2 and is missing in the abstract.** Add.49 requires *cite*, and there is no citation anywhere in the file to give. **Highest-consequence attribution finding.** |
| the qualitative ***breaking ⇒ finite lifetime*** prediction | source attributes it three times (30, 78–79, 122: *"that prediction is its result, not ours"*). In `pj_sub` **all three attributions are gone**; §4.2 (65) instead calls it *"a published single-exponential lifetime estimator"* with no owner. | ⛔ **not claimed, but no longer attributed.** |
| the word ***pseudo-gap*** | **absent (0 occurrences)** — sweep §C3 | ✅ compliant |
| what we *do* claim: two-branch law · crossover · floor · trained-potential measurement · regime structure | all five present (30, 55, 62, 65) | ✅ compliant — and Add.49's upward re-ruling is honoured: the abstract's *"which strictly matches our overdamped regime … but diverges … above the crossover"* is precisely Add.49's *"they predict a lifetime in one regime; we say which regime"* |

**A3-2 ⛔ A verbatim quotation from a cited source now appears unquoted and unattributed.**
- source (55): *"marginally stable tangent to the manifold and stable normal to the manifold" **(S\'agodi et al. 2024)*** — in quotation marks, with citation.
- `pj_sub` (46): *"Continuous attractors represent states that are marginally stable tangent to the manifold and stable normal to it."* — no quotation marks, no citation.
- **Ruling: CHANGED IN KIND, and the most reputationally hazardous single line in the file.** Near-verbatim borrowed phrasing stripped of both its marks and its owner.

**A3-3 ⛔ The anchor's non-novelty disclaimer is gone at the point of claim.**
- source (140): *"**That a corrective term can keep a flat direction alive is not new** (Renart, Song \& Wang 2003). What we add is the third part of the price list…"*
- `pj_sub` (71): *"**However, introducing a $V(\text{data})$ energy anchor successfully restores** and maintains the symmetry breaking."*
- Partial mitigation: Related Work (46) keeps *"Restorative homeostatic mechanisms are standard in the literature"* (uncited). **Ruling: the disclaimer survives generically in §2 and is absent where the claim is made.** A reader of §4.4 alone reads the anchor as ours.

**A3-4 ⚠ Status mislabel: a preprint described as "published".**
`pj_sub` says *"a **published** single-exponential lifetime estimator"* (30, 41, 65). Add.50: arXiv:2605.03338 is *"a preprint, never peer-reviewed."* The word has an ancestor — `submission.tex` (122) does write *"A published estimator gives the direct test"* — but the source pairs it with "recent preprint" (30, 75), "recently posted" (44, 122) and a bibliography entry marked `(preprint; single author)`. With the bibliography deleted, "published" is the **only** status signal left in the artifact, and it is the wrong one. **Ruling: CHANGED IN KIND, inherited-then-amplified.** Low cost to fix ("a recently posted", one word ×3).

**A3-5 ✅ Naming/priority items that are clean.** Nothing in `pj_sub` claims the theorem; nothing claims the corrective-term pattern as a first report; nothing claims a generalization result; nothing claims a benchmark win; the "flow" disambiguation survives; `pseudo-gap` = 0.

### A.4 The author-name rule (Add.51) — ✅ COMPLIANT, with one consequence to note

| sweep (per-file, positive-controlled) | `pj_sub.tex` | positive control |
|---|---|---|
| `\bMo\b\|\bMorse\b\|\bMoser\b\|\bhis\b\|\bHis\b\|\bhe\b\|\bhim\b` | **0 hits** | same regex on `submission.tex` → **3 hits**: `Morse` ×1 (157), `Moser` ×2 (168), `Mo` ×1 (190, the bibliography entry) ⇒ **the instrument fires and is not vacuously clean** |
| filenames | the file itself is `pj_sub.tex`; **0 `\includegraphics`** ⇒ no figure filenames to check | — |
| body text / captions / labels | no captions and no labels exist in the file (0 `\label`) | — |

✅ **The token is absent from body text, captions, labels and filenames.** ⚠ Two consequences the Head should know: (i) the *permitted* occurrence — the bibliography entry, which Add.51 explicitly says keeps its authors — is also gone, because the bibliography is gone (B1); (ii) the `Morse`/`Moser` survival trap is moot here only because both cited works were deleted, so the trap is untested on this file rather than passed.

---

## B. PART B — what was LOST (ranked by consequence; ⛔ flagged, never fixed)

**B1 ⛔⛔ #1 — the compute-normalization retirement, and with it the entire "Honest gap" paragraph.**
*The rule:* CM-4 amendment (SF-2) — *"the '≈4× longer' RETIRED as a compute claim … Lead with the qualitative triad (compute-independent); state the per-step factor honestly if a compute line is kept"*; source §sec:boundary "Honest gap" (136) + Appendix D `app:compute` (271–299). *Where it was:* main text ¶ after the baselines, plus a dedicated appendix with two tables. *Which claim now stands unqualified:* `pj_sub` (68) — the `263` vs `5.6–69` map-step comparison, i.e. **a retention-superiority number with no compute normalization anywhere in the artifact.** Also lost: `6.2×`/`3.1×` per-step wall, `14–15×` FLOPs, `23.5×`/`14.6×` per-retention, the not-width-matched confound, and the sentence that no task-RMSE was fabricated. **This is a claims violation, not a style matter.**

**B2 ⛔⛔ #2 — the N46 designed-only rider and its published counterexample.**
*The rule:* N46 disposition — *"it must travel with every CM-16(a) citation"*; source (132) carries it verbatim (*"a measurement on our architecture class and training recipe, not a general statement that learning cannot produce a tuned flat direction — a local learning rule that does produce one is published (Vafidis et al. 2022)"*), plus the Related-work fact (57). *Where it was:* immediately after the emergent negative, both sites. *Which claim now stands unqualified:* `pj_sub` (30, 42, 68) — the emergent negative, **widened to "emergent architectures" in the same stroke** (A2-A4). Also lost: `13–14` orders, the `15`-orders attribution-instrument separation, `1–1.6` bits on `2–3` washboard minima, "a written δ relaxing completely", `n=3`.

**B3 ⛔⛔ #3 — the score sentence, and its replacement by a design-intent sentence.**
*The rule:* Add.53's V2 do-not-cut list ("the score sentence"); Head's binding framing (matrix §2 note, 2026-07-23): *the program's score = external benchmarks won on their own headline metric — currently ZERO.* Source scope box (145) item (ii): *"**No external benchmark is won on its own headline metric anywhere in this paper**; the comparisons are diagnostic retention protocols on a synthetic $S^1$ family and matched-parameter ablations, not leaderboard results --- and §4.3's honest gap is part of the claim, not a caveat to it."* *What replaced it:* `pj_sub` (78) *"This architecture is explicitly designed for bounded retention and symmetry protection rather than raw state-space tracking capabilities."* **Ruling: CHANGED IN KIND — a measured absence of any benchmark win is recast as a design choice.** Also lost from the same box: *"the unit does not enter the input-driven task-RMSE axis"* (source 145) and the "honest gap is part of the claim" clause.

**B4 ⛔ #4 — the metric-naming rule, and every metric-bifurcation number that motivates it.**
*The rule:* source states it three times (112, 116, 382): *"the retention metric bifurcates there, envelope half-life and first-crossing time agreeing to $0.01\%$ overdamped and splitting by $3.2\times$ at the deepest tilt, **which is why every lifetime in this paper names its metric**"*; §1 (37) *"each is reported with the metric defining it, because those metrics bifurcate past the crossover."* *Where it was:* §4.1, Figure 1's caption, §1's two-masses paragraph. *Which claims now stand unqualified:* every lifetime in `pj_sub` — the `27.03`-step floor (62), `263`/`5.6`/`69` map-steps (68), the `n₁/₂` band formulas (55) — **none names its metric, in a paper that keeps the crossover where the two metrics split by 3.2×.** Also lost: the `φ=0` on `15/15` rows and `0.5165` vs predicted `1/2` frequency-onset signature, i.e. both second-order confirmations that the crossover is a genuine exceptional point rather than a fitting artifact; and the ± on `2.202±0.155` / `0.309±0.012`.

**B5 ⛔ #5 — the `fdt` + Newtonian mandatory flag, reduced to half a sentence.**
*The rule:* CM-16 MANDATORY FLAG (*"all of this holds only under `langevin_noise="fdt"` AND a Newtonian kinetic mode … The repo default is `legacy`, where T is not in energy units and NONE of these laws hold"*), source "Mandatory flag, travels with every finite-temperature number in this paper" (105) + fine print (a) (103). *What survives:* `pj_sub` (56) *"All finite-temperature outcomes ($T>0$) strictly assume fluctuation-dissipation-consistent noise"* + (77) *"The finite-temperature diffusion law is strictly constrained to Newtonian kinetic modes."* — **both halves present, split across two sections, neither naming the flag.** *What is gone:* the `legacy`-default warning; the relativistic no-noise-scale-targets-Gibbs clause (CM-17); `σ*_i=√(M_iTγ(2−γ))`; `D_θ=εT(2−γ)/(2F²γ)`; the `3.77±0.23×` friction-lengthens-memory result on `5/5` seeds; the `T*≈3e-3` crossover; and the reassurance that all main-text results are `T=0`. *Consequence, mitigating:* `pj_sub` carries **no finite-temperature number at all**, so the flag currently has no referent — the loss is latent, and becomes live the moment any finite-`T` sentence returns.

**B6 ⛔ #6 — the CM-21 retirements and the controllability-not-capability sentence.**
*The rule:* CM-21 (*"ALL drafts, binding"*) — four retired positioning claims (HiPPO-LegS has a retention-guarantee analogue · "retrieval is a rollout" is Kong et al.'s · continuity is no NTM/DNC escape hatch · not a transformer competitor) + the approved replacement wording (*"controllability, which we evidence, not capability, which we do not claim against transformers or state-space models"*). *Where it was:* source §2's dedicated retirement paragraph (92), elaborated in Appendix E `app:pos` (364–374). *Status in `pj_sub`:* the paragraph and appendix are **absent**. ⭐ **Mitigating, and it matters: no sentence in `pj_sub` re-asserts any of the four retired claims**, and its closing line (81) is compatible with CM-21's approved wording. So this is a **lost pre-emptive defence, not a violation** — the referee's first four objections are now unanswered in writing, which is exactly what that paragraph was written to prevent.

**B7 ⛔ #7 — all five figures, and with them the mandatory multi-seed scope label.**
*The rule:* Add.52 / source `app:retention` caption (267): *"**Multi-seed status, stated plainly:** each baseline curve is the median over $5$ seeds and the emergent curve the median over $3$ seeds, while the designed curve is a **single representative checkpoint** --- the $5/5$-seed latch statement is §4.3's, not this figure's."* *Status:* `pj_sub` has **0 `\includegraphics`**; all five figures and all five captions are gone, including the `\emph{Verification.}` / `\emph{Evidence.}` labels (A2-A6) and Figure 1's `5` seeds / dim `4` / `γ=0.05` / laptop-CPU provenance line. *Which claim now stands unqualified:* §4.3's baseline paragraph (68) — **the `263`-vs-baselines comparison now carries no seed count of any kind, having previously carried a caption that specifically warned the designed curve was single-seed.** (No caption/figure *mismatch* is possible; the finding is total absence.)

**B8 ⛔ #8 — the negatives appendix, the substrate-scope sentence, and the remaining scope-box lines.**
- **the prominent-negatives appendix** (`app:neg`, source 301–361, two tables, eleven rows + three reading rules): **absent.** Head policy 2026-07-07 puts the most prominent negatives in the named papers' appendices; `pj_sub` has no appendices at all. Notably gone: *"Can this unit enter the input-driven path-integration task-RMSE axis? **No** — no native velocity ingestion … **No task-RMSE was fabricated**"*; the tilt-is-not-a-lifetime-dial-on-a-learned-store fence with `τ_max=Γ/2α=4.0` (N150/N149 — the reading rule that *"no sentence in this paper may be read as claiming that a tilt magnitude is a lifetime dial on a learned store"*); the friction-cannot-stabilize-a-saddle limit; the sampler-not-thermodynamics scope. ⚠ **Live risk from this one:** `pj_sub` (62) says *"we tilt the trained potential by a known amount to extract the resulting transverse curvature"* on a **designed** vacuum, which is in scope — but with the fence deleted, nothing in the artifact stops a reader from generalizing it to a learned store, which is precisely the sentence N149/N150 forbid.
- ⛔ **the substrate-scope sentence** (Add.53's list; source 145 item (iii)): *"these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, with its parameter and state-byte budget ledgered"* — **absent.**
- ⛔ **the CM-16a/b split** (Add.53's list): `pj_sub` states only the CM-16a designed-only half and **never states that the unification and the `T>0` face DO generalize to the emergent arm** (source 132: *"though the damping-optimum retention law itself still generalizes to it"*; CM-16b). N46's own instruction is *"⚠ The split IS the result — record both halves."* Half the result is missing, and the missing half is the *favourable* one.
- also absent: the BIBO / coercive-potential scope sentence and its "nothing here claims symplectic structure buys off-distribution stability in a trained, input-driven setting" clause; *"friction never stabilizes a saddle"*; *"The symmetry framing is tree-level only"* and the no-loops/no-thermodynamic-limit paragraph (98); the *"we make no claim about biological systems and do not model neural data"* disclaimer (55); the CD **scope clause** on `sleep_steps∈{50,500}` and mixing time (140); the theory-note and CLU/CHLU naming-continuity citations (35); the fine print (b) Sylvester/kinetically-broken-battery block (103); the directions list (148).
- ✅ **present and correct:** the laptop-CPU / dim `4` / hidden `64` / `≤5` seeds scale qualifiers (76) and the local-at-a-critical-point scope (77) — the two C-5 items that did survive.

**B9 — never-quote sweep (matrix §0.1–§0.14): ✅ CLEAN, positive-controlled, zero-list printed.** Pattern (37 alternates, run per-file): `SF-[0-9]|CM-[0-9]|Cor-[0-9]|\.claude|/Users|scratch/|handover|Advisor|Hub|spoke|never-quote|PREREG|N[0-9]{2,3}|CSF3|CAMELS|CMAPSS|organizer swap|13\.9|bprime|CLU-former|claims matrix|wave-[0-9]|charter|certified|unlearning|deletion-compliant|2\.6\b|24\.5|0\.99985|19×|pseudo-gap` → **`pj_sub.tex` = 0 hits.** Semantic-hermeticity pattern `companion (paper|short)|our other short|\bV1\b|\bV3\b|\bV5\b|sister paper|our unpublished|under review elsewhere|CLU|CHLU|Jawahar|theory note|GMOR|Gell-Mann|ChPT|spurion|condensate` → **`pj_sub.tex` = 0 hits.** **Positive controls on `submission.tex`: 1 line and 29 lines respectively** ⇒ both instruments fire. ⚠ The single control hit on pattern 1 is a **false positive** — `2.6` matching `$h-h^*=2.6\times10^{-6}$` / `$2.6\times10^{-2}$` (submission.tex 382), unrelated to the retracted capacity "2.6". **The condensation introduced no never-quote violation; it removed the vocabulary that could produce one.**

**B10 — citations, tabulated as the task asks.** Every claim's supporting citation was dropped, because **all ~50 references were dropped**: 0 `\cite`, no `\section*{References}`, no bibliography. Conversely, **0 citations with no bibliography entry** (vacuously true). Claims whose citation loss is load-bearing, in order: the zero-Lyapunov theorem (A3-1) · the borrowed continuous-attractor definition (A3-2) · the anchor's prior art (A3-3) · Vafidis' published counterexample (B2) · flow equivariance / the emergent-arm competitor / the solution-degeneracy instrument (all three of Add.52's positioning anchors, now nameless in §2) · the three baselines LSTM/LEM/coRNN · the theory note and the CHLU naming-continuity line.

---

## C. PART C — render (⛔ BLOCKED, with everything pre-computed for whoever runs it)

**C1 Why blocked, plainly.** This spoke's toolset is Read / Write / Edit / Grep / Glob — **there is no shell, no Bash, no execution tool.** I cannot run `pdflatex`, cannot run `md5`, cannot run `pdftotext -bbox`. I did not simulate any of them. `pj_sub.pdf` therefore **does not exist** at the end of this pass, and the page split (main / references / appendices) is **not measured**.

**C2 One command, expected to succeed on the first pass — static compile analysis.**
```
cd .claude/NIPSsubmission/v2-neurreps && \
/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex && \
/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex && \
/Library/TeX/texbin/pdflatex -interaction=nonstopmode pj_sub.tex
```
Risk assessment, from the source text alone:
- `neurips_2025_ml4ps.sty` **is present in the folder** (Glob) — the one hard external dependency is satisfied.
- the preamble is **byte-identical to `submission.tex`'s** (lines 1–23, incl. the `\makeatletter\renewcommand{\@notice}{}` venue-neutral suppression and empty `\author{}`), and that file built with **0 errors / 0 undefined references** (BUILD-NOTE §1).
- **0 `\cite` / 0 `\ref` / 0 `\label`** ⇒ no `.bbl`, no bibliography pass, no undefined-reference or undefined-citation warnings possible; the ×3 runs are needed only for `hyperref`/`\maketitle` bookkeeping.
- **0 `\includegraphics`** ⇒ `figs/` is not required at all (note: `figs/` does **not** exist in this folder, and it does not need to).
- the file is **pure ASCII** (grep for `[^\x00-\x7F]` → 0) ⇒ no `inputenc` risk.
- all math is `amsmath`-standard; no `booktabs` tables, no floats, no `\appendix`.
- ⇒ **prediction: compiles clean; ~2 pp of output; page split will be `main = all of it`, `references = 0`, `appendices = 0`.** If it does compile, the task's "build from a copy and list every repair" branch never triggers and `pj_sub_buildcopy.tex` must **not** be created. If it does *not* compile, that branch is unexecuted here and still owed.

**C3 What I substituted for the mechanical checks I could not run.** Structural inventory by grep (§0, B7, B10); numeric ledger by hand-enumeration line by line (§A.1) rather than by token-multiset diff — so my ledger is **complete but manually derived**, and a scripted two-way token check (the instrument BUILD-NOTE §4 used) is still worth running as an independent confirmation. It should return: *in source, not in `pj_sub`* = large; ***in `pj_sub`, not in source* = empty** — that second set being the claim I am making in §A.1, and the cheapest way to falsify me.

**C4 ⛔ The byte-identity check, honestly labelled.** `md5 pj_sub.tex` before/after **could not be computed (no shell)**. What I can state: **I made zero Write and zero Edit calls against any file in `NIPSsubmission/`** — and under this harness a Write to a file not previously read, or any Edit whose `old_string` did not match, would have errored rather than silently succeeded; `Read` and `Grep` are non-mutating. I re-read `pj_sub.tex` at the end of the pass (lines 59–83 plus the full-file read at the start): **84 lines, content identical to the opening read, `\end{document}` at line 83.** ⇒ **the file is unchanged, verified by re-read, NOT by md5.** The Hub should treat the md5 acceptance criterion as **owed** and satisfiable in one command: `md5 .claude/NIPSsubmission/v2-neurreps/pj_sub.tex`.

---

## Findings summary — the ranked list the Head asked for

**Part A (fidelity of what survived).** ✅ Zero fabricated numbers; zero mis-transcribed digits; author-name rule clean; the units conversion, the band structure, the "they generalize, we price" fence and the Add.52 audience scoping all survive intact. ⛔ Four claim-strength defects, in order: **A2-A5** the retired compute claim reinstated as a bare retention ratio (CHANGED IN KIND) · **A2-A2** "bit-identical" extended to laws whose slope moved `−0.985→−0.956` (WIDER) · **A2-A4** a 3-seed one-recipe negative widened to "emergent architectures" + "We establish" (WIDER) · **A2-A1** `3.2e-10` over `70` rows → "perfectly matches" (WIDER); plus **A2-A3** a two-cost cure reading as an unqualified success (WIDER) and **A2-A6** verification reading as discovery (CHANGED IN KIND). ⛔ Attribution: **A3-1** the zero-Lyapunov theorem stated in our abstract with no citation available anywhere in the file, and **A3-2** a cited author's phrasing reproduced unquoted and unattributed.

**Part B (mandatory content lost).** Ranked: **B1** compute-normalization retirement (CM-4) → a claims violation · **B2** N46 designed-only rider + Vafidis counterexample · **B3** the score sentence, replaced by design intent · **B4** the metric-naming rule and both exceptional-point signatures · **B5** the `fdt`+Newtonian flag reduced to halves (currently no referent) · **B6** CM-21's four retirements (defence lost, no violation) · **B7** all five figures incl. the mandatory single-seed label · **B8** the negatives appendix, the substrate-scope sentence, the CM-16a/b split's favourable half, and the remaining scope-box lines. **B9** never-quote sweep clean, positive-controlled. **B10** the whole citation apparatus.

**Two source-internal discrepancies I flag and deliberately do NOT resolve** (protocol: I organize, I never reinterpret a verdict): (i) `submission.tex` gives the anchored slope as `−0.956` (140) and `−0.961` (215); (ii) `submission.tex` gives softest emergent `μ² = 5.1/5.9/5.4e-2` (132) where `negative_results.md` **N46** gives `5.449e-2 / 2.029e-2 / 5.132e-2`. Both predate `pj_sub.tex`. **Hub owns both.**

---

## Curator-scope note (what I did NOT edit, deliberately)

My standing transfer docs — `HEP_primers.md`, `philosophy-synthesis.md`, `negative_results.md`, `future_work.md` — are **untouched**. This task scopes exactly one report and its acceptance criteria name no registry fold. Two candidate registry actions are surfaced here for the Hub to task explicitly if it wants them: the N46-vs-`submission.tex` per-seed discrepancy (a `negative_results.md` N46 numbers line), and a `future_work.md`-style note that the anchored-slope digit (`−0.956` / `−0.961`) is unpinned in the accepted base. **I did not write either.**

## Proposed handover updates (for the Hub)

1. **`pj-fidelity-v2` = PARTIAL.** Parts A + B delivered in full. ⛔ **Part C not delivered — `pj_sub.pdf` does not exist**; this spoke had no execution tool. Two mechanical items are owed and each is one command: the ×3 `pdflatex` build (§C2, predicted clean, ~2 pp, no figures needed) and the before/after `md5` (§C4). ⚠ **`tasks/pj-referee-v2.md` is gated on that PDF** and cannot run until someone builds it — either re-spawn this spoke with shell access or hand §C2 to the Head.
2. **Head-facing answer to the question actually asked ("has the paraphrase misrepresented any fact?"):** *no number is wrong and none is invented; six claims changed strength.* Four are worth an edit before this goes anywhere: (a) restore `3.2\times10^{-10}` / `70` rows or drop "perfectly" (62); (b) confine "bit-identical" to the exceptional-point onset (71); (c) restore "at the cost of weaker noise rejection and ≈35× higher wake MSE" (71); (d) re-scope "emergent architectures" → "this architecture class and training recipe" (30, 42).
3. **Two rider restorations I would rank above all stylistic work**, because each is a claims violation as it stands: the **compute-normalization retirement** beside the `263`-vs-baselines sentence (68) — CM-4 binding — and the **score sentence** in the Discussion (78) — the Head's own 2026-07-23 framing. Both are one sentence each.
4. **The citation apparatus is the file's largest single risk and it is not a fidelity question:** with 0 `\cite` and no bibliography, the zero-Lyapunov theorem (30) and a cited author's own definition of a continuous attractor (46) both read as ours. Add.49 says *cite and never claim*; the file currently cannot cite. Recommend the Hub treat "restore a minimal bibliography (N1 + Ságodi + Renart + Vafidis + Keller + the three baselines)" as the first content decision on this draft, not a polish item.
5. **Registry actions surfaced, not taken** (see Curator-scope note): the N46 per-seed discrepancy vs `submission.tex` §4.3, and the unpinned anchored slope `−0.956`/`−0.961` in the accepted base. Both want an owner; neither is mine without a task.
6. **For the independent referee's calibration (do not forward this report):** the PDF they will read has no figures, no references and no appendices. If the referee reports "unsupported claims / no citations", that is a **true positive** and agrees with A3-1/B10 — the two blind passes converging, exactly as Add.53 designed.
