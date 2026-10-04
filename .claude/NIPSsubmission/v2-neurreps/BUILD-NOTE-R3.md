# BUILD-NOTE-R3 — bounded restoration pass on `pj_sub.tex` (task `pj-restore-v2`, 2026-08-22)

**Scope executed:** exactly one content file edited — `pj_sub.tex` (+ its rebuild artifacts `pj_sub.pdf/.aux/.log`). `submission.tex`, `submission.pdf`, `submission.log`, `BUILD-NOTE.md`, `neurips_2025_ml4ps.sty`, `figs/*` and `pj_sub.out` are **byte-identical to their pre-pass md5s** (§7).
**Discipline:** additive by default; the Head's prose is not rewritten, reordered or compressed. The only non-additive edits are Part B (a)–(j), printed before→after in §2. Zero new measurements. Every number added carries a named ancestor (§4).

**md5 of the final `pj_sub.tex` = `396a00d0c41285a471fddc8567ac256d`** (pre-pass: `d15de78712d90eb94d2495d4bd9ad948`, which matched the Advisor's expected hash on entry).

---

## 1. Build result

| item | before | after |
|---|---|---|
| build | `pdflatex -interaction=nonstopmode pj_sub.tex` ×2 | same, ×2 |
| errors | 0 | **0** |
| undefined references / citations | 0 | **0** (`\@abspage@last{17}`, all `\newlabel` resolve) |
| overfull hboxes | **1** (11.28 pt, the `app:loan` 6-column table) | **1** — the *same* pre-existing box, verified by rebuilding the pre-pass file in a scratch dir (`scratch/pj-restore-v2/beforebuild/`): 14 pp, identical 11.27979 pt overfull. **No new bad boxes.** |
| pages | 14 | **17** |

**True page split** (instrument: `outputs/figure-render-pass/pagesplit.py`, PDF word bounding boxes, text block 72–720 pt):

| block | before (fidelity r2 / referee r2) | **after** |
|---|---|---|
| main text (§1–§5) | ≈6.6 pp | **7.50 pp** (§1 p1 · §2 p2 · §3 p4 · §4 p5 · §5 Discussion p8) |
| References | ≈2.2 pp | **1.93 pp** (one entry removed — Part B(i)) |
| appendices A–G | ≈5.2 pp | **7.57 pp** (A p10 · B p11 · C p11 · D p11 · E p13 · F p14 · G p15) |
| total | 14 pp | **17 pp** |

Page limits deferred by Head ruling; recorded, not scored.

## 2. Part B — the enumerated corrections, before → after

| # | site (final line) | before | after |
|---|---|---|---|
| **(a)** R2-1 | 28 | "…and exact equivariance **is required to protect** a neutral direction…" | "…and exact equivariance **protects** a neutral direction…" |
| **(b)** S-5 | 60 | "…**guarantees** $\dim(G/\mathcal H)$ zero Lyapunov exponents…" | "…**guarantees at least** $\dim(G/\mathcal H)$ zero Lyapunov exponents…" |
| **(c)** MF-2 | 44 | "**This chapter** provides a summary…" | "**This paper** provides a summary…" |
| **(d)** S-4 | 28 | "…**infinitesimal perturbations** to the dynamics destroy it…" | "…**most infinitesimal perturbations** to the dynamics destroy it…" |
| **(e)** MF-5 | 28 | "…the quantitative exchange rate---how the retention of a written value scales with the transverse curvature of a trained potential---**remains unmeasured**." | "…**no closed-form constitutive law linking the transverse curvature of a trained potential to lifetime has been measured.**" |
| **(e)** MF-5 | 60 | "We do not claim novelty over these **qualitative** lifetime predictions **or** existence proofs." | "We do not claim novelty over these **estimator-based (kinematic)** lifetime predictions **and** existence proofs." |
| **(f)** R2-5 | 265 | "This penalty is **proven mathematically** to be contraction-forbidden by the volume conservation axiom, **as** the broken-volume baseline precisely recovers $\approx2.4\times$ of the performance gap purely by leaking volume." | "This penalty **is** contraction-forbidden by volume conservation---the broken-volume baseline recovers $\approx2.4\times$ of the performance gap precisely by leaking volume---**and is not the causal velocity cap, inactive here.**" |
| **(g)** S-1/R2-6 | 417 | "Above the established crossover $h-h^*=2.6{\times}10^{-6}$, trajectory and Jacobian frequencies align tightly to $0.06$--$0.3\%$." | "**The $\sqrt{h-h^*}$ onset holds down to** $h-h^*=2.6{\times}10^{-6}$ on a fine multiplicative grid ($h-h^*\in[-4.2{\times}10^{-3},2.6{\times}10^{-2}]$). **Where the quality factor permits ($f=2,4$)**, the trajectory frequency matches the Jacobian frequency to $0.06$--$0.3\%$; **near onset the mode decays before one period completes, so the exceptional point is spectroscopically real but dynamically silent there.**" |
| **(h)** P-8 | 47, 104 | "a recently **published** … lifetime estimator" (×2) | "a recently **posted** … lifetime estimator" (×2) |
| **(i)** MF-3 | 58; ref list | "Similarly, recent efforts utilize symmetry regularization to induce flat directions **(NeurReps 2025 workshop poster)**." + the authorless References entry | "Similarly, **concurrent workshop work has explored soft symmetry regularization for continuous attractors.**" + **entry deleted** (References 51 → 50) |
| **(j)** CM-4 | 124 | "We explicitly retire it as a **wall-time efficiency** claim." | "We explicitly retire it as a **compute** claim." |
| **(j)** CM-21 | 64 | "…retention is governed as a designed, per-item property controlled by $\mu^2$." | "…controlled by $\mu^2$**---\emph{controllability}, which we evidence, not \emph{capability}, which we do not claim against transformers or state-space models.**" (base 92, verbatim) |

## 3. Part A — the additive restorations, mapped to worklist item and ancestor

Ancestor column: `S:n` = `submission.tex` line *n*; otherwise the named output file + line.

| worklist | final line | what was inserted | ancestor |
|---|---|---|---|
| **A1 / P-1** | 139 | new Discussion bullet **Substrate Scope** — "these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, with its parameter and state-byte budget ledgered" (verbatim) | `S:145(iii)` |
| **A1 / P-2, P-5(a), P-7** | 137 | "\emph{No external benchmark is won on its own headline metric anywhere in this paper}; the comparisons are diagnostic retention protocols on a synthetic $S^1$ family and matched-parameter ablations, not leaderboard results---and §4.3's honest gap is part of the claim, not a caveat to it. The head-to-head is on \emph{autonomous retention} in map-application units, and the unit does not enter the input-driven task-RMSE axis." (verbatim; the Head's two sentences **kept**, the measured form added) | `S:145(ii)`, `S:136` |
| **A2 / sufficiency fence** | 60 | "The theorem states sufficiency; equivariance is not necessary, the latch being a modulus of $V_\theta$." | `S:86` |
| **A2 / 103(b)** | 81 | "…map equivariance is sufficient but not necessary for a neutral memory direction---our kinetically-broken battery is a \emph{measured} non-equivariant map that latches exactly (App. E)." | `S:103(b)` |
| **A2 / R2-2 (CM-17)** | 398 | the sampler-scope paragraph restored **in full**: MJ mechanism + "**The failure is in the \emph{sampler}, not in the \emph{thermodynamics}**" + thermostat repair + "We therefore never assert that a relativistic unit ``has no equilibrium''." + "the trained units here are `newtonian_learned` throughout" | `S:361` |
| **A2 / R2-3** | 307 | "…, while the designed curve is a single representative checkpoint---the $5/5$-seed latch statement is §4.3's, not this figure's." | `S:267` |
| **A2 / P-6** | 83 | `langevin_noise="fdt"` + `newtonian_identity`/`newtonian_learned` + "The reference implementation's default is `legacy`, under which $T$ is not in energy units and none of these laws hold." + "The checkpoints here are `newtonian_learned`, so every finite-temperature number here is in scope." | `S:105` |
| **A2 / P-3 (N46)** | 120 | "…the preprint's own attribution instrument separates the architectures by $15$ orders. This is a measurement on our architecture class and training recipe, not a general statement that learning cannot produce a tuned flat direction---a local learning rule that does produce one is published (Vafidis et al. 2022)." | `S:132` |
| **A2 / P-4** | 130 | "\emph{That a corrective term can keep a flat direction alive is not new} (Renart, Song \& Wang 2003)." | `S:140` |
| **A2 / R2-12** | 108 | "It also does not rest on substituting the exact Jacobian gap for their predictor: the result reproduces on the preprint's own finite-horizon estimator (App. F)." | `S:122` |
| **A2 / R2-12 (App F)** | 411 | "**The head-to-head reproduced on the preprint's own instrument.** … $\mathrm{corr}=0.9995$ overdamped, meas/pred $0.86$--$1.03$, and $0.30$ at $\delta=4$ against the exact-gap $0.31$." | `S:376` |
| **A3 / P-5(a)** | 365 | negatives row: "Can this unit enter the input-driven path-integration task-RMSE axis? — No… **No task-RMSE was fabricated.**" | `S:325–327` |
| **A3 / P-5(b) N51** | 377 | negatives row: raw $n_{1/2}$ exponents = instrument, not physics + "**No $n_{1/2}$ may be quoted without its $\Delta$ and $\ell_\theta/\Delta$**" + the six exponents + $\ell_\theta/\Delta\le2.0$ | `S:319–321` |
| **A3 / P-5(d)** | 380, 383, 386, 389 | four negatives rows restored: friction-field × governor composition; friction-cannot-stabilize-a-saddle / energy-gate; mean-spectrum chaos regularizer; `sleep_temperature` silent knob | `S:338–352` |
| **A3 / P-5(e)** | 352, 355, 358, 360 | within-row numbers: "tied controls exactly $1.0000$"; "inversion at epoch $116$/$442$/$959$ by sleep frequency"; "$\sim\!12$ orders; written $\delta$ retained $\le2.1{\times}10^{-3}$; capacity $\approx1$--$1.6$ bits"; second $\lambda_{\min}$ range "$+0.3291\to-1.1980$"; tilt-vacuum residual "$0.140$--$0.343$ against $1/\mathrm{dim}=0.167$"; "A designed degeneracy does not survive superposition." | `S:312, 315, 318, 323, 324` |
| **A3 / P-5(g)** | 396 | "Off the orbit only boundedness holds… no proportionality constant should be quoted as a law… never a ``drift rate''." | `S:357` |
| **A3 / P-5(f)** | 396 | "the breaking coefficient of that battery is an **explicit-symmetry-breaking magnitude**; it is **not** this paper's integrator step $\varepsilon$, and the two must never be conflated." | `S:359(a)` |
| **A3 / P-5(c)** | 138 | "; and friction never stabilizes a saddle (App. E)" appended to the Off-Distribution bullet (pairs with the restored saddle row) | `S:145` |
| **A3 / R2-7** | 415 | demotion label: "**GMOR proper---G.3 onward---is demoted from the main text and retained here in full; it is a supporting result, not one of this paper's claims.**" | `S:380` |
| **A3 / R2-7** | 431 | the **Precision fine print** paragraph in full, incl. "**Do not quote ``$2.2{\times}10^{-16}$ relative'' for this experiment.**" and "The same $\epsilon/\delta$ floor applies to every $\mu^2$ probe in this paper…" | `S:413` |
| **A3 / R2-7** | 433 | new **G.5** — defines $x\equiv\delta/(M_{\rm ch}\mu_{\rm rad}^2f)$ (**resolves the orphan $x$ of caption (d)**) and carries "**$\delta=0.3$ must not be quoted for the next-to-leading-order claim**: cap at $x<0.25$ or state $x$ explicitly" | `S:417` |
| **A3 / R2-7** | 435 | new **G.6 Honest scope** — tree-level/classical, no loops or chiral logs, probe-only, one architecture family, emergent MLP arm untested = an open falsifiable, not a result | `S:419` |
| **A3 / R2-8** | 313 | "Wall time is the median of $7$ scan-amortized repetitions over $2{\times}10^5$ steps; the naïve single-call timing is dispatch-bound at a $\approx5\,\mu$s floor and is \emph{not} reported." | `S:273` |
| **A3 / R2-8, SF-4** | 339 | "**Confound, flagged:** the comparison is not width-matched, CLU at hidden $64$ … the sign is robust… GPU or batched throughput could shift the wall ratios, though not the FLOPs." | `S:299` |
| **A3 / SF-5** | 253 | Appendix A prose (3 sentences) pointing at §4.4 + §4.1 and the figure. ⚠ **connective prose written for this pass — no base ancestor exists** (base's App A is figure-only); asserts no number (§6, deviation 5) | — |
| **A4 / S-2** | 80 | "…lengthens memory by $3.77\pm0.23\times$ **on $5/5$ seeds**" | `S:103` |
| **A4 / S-3** | 419 | "a V-shaped curve **of depth $\approx8\times$**, whose argmin sits **at, or just below**, … **on all $5$ designed seeds**" | `S:384` |
| **A4 / S-6** | 421 | "**($10$ values, $8$ decades)**" and "**over all $80$ pairs**" | `S:386` |
| **A4 / S-7, R2-4** | 269, 301 | "wake-only objective **except the $\gamma_\phi$ rung**"; and the footnote "*The $+\gamma_\phi$ rung retrained with the sleep phase on, so the ``wrong tool'' verdict is robust but the $-24\%$ is not a clean single-knob delta.*" | `S:225`, `S:259` |
| **A5 / MF-1** | 130, 257 | fit-spec at **both** slope sites — §4.4: "$-0.956$---the per-point fit over all overdamped rows, where Figure 2 prints $-0.961$, the seed-mean OLS over the $7$ overdamped $\delta$, for the same data"; Fig-2 caption: the mirror clause. **Neither number is picked over the other.** | `figure-render-pass.md:101, 207` |
| **A5 / MF-4** | 108 | "$=0.9987$ **overdamped-only; $0.973$ pooled---the drop is the regime structure**" | `v2-full-runs.md:81` |
| **A5 / SF-1** | 130 | "over this grid's $4.6$ decades of $\delta$ (the $150$-epoch grid of §4.1 spans $4.5$)" | `S:140` / `S:112` |
| **A5 / SF-2** | 130 | "Across $3000$ anchored epochs **on $3$ seeds---a separate re-measurement from the $5$-seed anchor sweep above---**…" | `S:140`; `anchor-robustness.md:48`, `v2-referee-experiments.md:142` |
| **A5 / SF-3** | 130 | "At $1000$ epochs **(sleep frequency $5$, $500$ sleep steps, CD sampler)**…" | `anchor-robustness.md:146` (`f5/s500/CD`) |
| **A5 / SF-8** | 130 | "a higher wake MSE" → "**$\approx35\times$ higher wake MSE**" | `S:140`; `anchor-robustness.md:64` |
| **A5 / N-2** | 98 | "; the saturated rows scatter within the kick-phase ripple ($\pm8$ steps at $\delta=4$)" | `v2-full-runs.md:66` |
| **A5 / N-6** | 92, 417 | both sites now read "**over all $70$ tilt rows**" | `S:112`, `S:382` |
| **A5 / N-3** | 122 | footnote: "At the fixed default oscillator hyperparameters coRNN is the honest-weak entrant (train-horizon RMSE $0.82$ rad); the triad-absence claim rests on the well-trained LSTM and LEM, which reach train-horizon RMSE $0.18$/$0.23$ rad." | `philosophy-synthesis.md:628` (0.82) + `S:134` (0.18/0.23) |
| **A6** | 62 | "the standard leapfrog $h<2$ stability limit (Hairer, Lubich \& Wanner 2006)" | `S:90` |
| **A6** | 142 | "…solution degeneracy between the designed and emergent arms **with the instrument of Huang et al. (2025)**…" | `S:73`, `S:148` |
| **A6** | 405 | App F(1) gains: EDEN tunable-permanence occupancy; UnICORNN "**has** the gradient-bound theorem we lack"; Ramsauer Hopfield≡attention coincidence at $\beta=1/\sqrt D$ | `S:368` |
| **A6** | 408 | App F(4) gains: "**no win over a trivial baseline on any external benchmark**, and formal limits on fixed-size latent state for copying and retrieval (Jelassi et al. 2024) … it prevents \emph{decay}, not \emph{capacity}" | `S:374` |
| **A6** | 216 | Mo bibliography entry regains "**(preprint; single author)**" | `S:190` |
| **A7** | — | **figures: verification only.** 5 `\includegraphics`, 5 PNGs in `figs/`, **5/5 used, 0 unused, 0 added** (`fig1_gmor`, `fig_lifetime_headtohead`, `fig2_anchor_cure_laws`, `fig3_retention_overlay`, `fig3_gmor_condensate`) | — |

**Orphan-reference status after the pass:** all six previously-uncited retained entries now carry an in-text use (Hairer, Huang, EDEN/Karuvally, UnICORNN/Rusch & Mishra 2021b, Ramsauer, Jelassi). The one authorless entry is deleted (B(i)). **References: 51 → 50; 0 new records; 0 dangling cites.**

## 4. Two-way numeric check

Method: numeric-token bag of the pre-pass file vs the post-pass file (`\d+(\.\d+)?`), both directions; every token whose count **increased** is checked against `submission.tex` and, failing that, against the named output file.

- **Tokens whose count increased: 79 distinct values.** **77 of 79 have a `submission.tex` ancestor.** The **two** without one are exactly the two the task authorizes from named output files:
  - **`0.973`** — pooled corr — ancestor `v2-full-runs.md:81` (task A5 / MF-4).
  - **`0.82`** — coRNN train-horizon RMSE — ancestor `philosophy-synthesis.md:628` (task A5 / N-3).
- **Tokens whose count decreased: exactly one — `2025` (−1)**, the deleted authorless poster entry (Part B(i)). Nothing else was removed.
- **No value, precision, unit, exponent or ± anywhere in the file was altered in any digit.** Every added token is on the edit map in §3.
- Reverse direction inherited: `pj-fidelity-v2-r2` §A.1 established that the pre-pass file had **no number without an ancestor**; this pass adds only ancestored numbers, so the property holds for the whole file.

## 5. Sweeps (per-file, positive-controlled — the standing gitignored-dir Grep hazard)

| sweep | `pj_sub.tex` | positive control | verdict |
|---|---|---|---|
| never-quote, 33-alternate pattern (`SF-\d`, `CM-\d`, `.claude`, `/Users`, `handover`, `Advisor`, `Hub`, `spoke`, `PREREG`, `N\d{2,3}`, `wave-\d`, `charter`, `certified`, `unlearning`, `2\.6\b`, `pseudo-gap`, …) | **1 hit — line 417**, the *same inherited false positive* both fidelity rounds recorded: `$h-h^*=2.6{\times}10^{-6}$` matching `2\.6\b`, ancestor `submission.tex:382` | `advisor-head-shorts-charter.md` = **372 hits** | ✅ **CLEAN** |
| author-token rule (`\bMo\b|\bMorse\b|\bMoser\b|\bhis\b|\bHis\b|\bhim\b`) | **2 hits** — line **216** the *permitted* bibliography entry; line **150** "…via the lens of **Morse** theory" (the survival trap, correctly survived). **0 in body text, captions, labels or filenames**; all 5 `\includegraphics` paths are `figs/fig*_*.png` | `submission.tex` = **3 hits** | ✅ **COMPLIANT** |
| `pseudo-gap` | **0** | `advisor-head-shorts-charter.md` = 5, `mo-deep-read.md` = 7 | ✅ **0** |
| semantic hermeticity (`companion (paper|short|note)`, `our (other|sibling|companion)`, `the program`, `sibling (paper|short)`, `in a companion`, `our V\d`, `the three shorts`) | **0** | `philosophy-synthesis.md` = **132 hits** | ✅ **0** |

## 6. Deviations, and items not executed — stated honestly

1. **B(f) is a form-restoration, not a verbatim base copy.** The base sentence uses its own nouns ("That cost", "the broken-volume arm"). To honour rule 1 (the Head's prose is not rewritten) I kept `pj_sub`'s nouns and restored only the *assertion structure* — "proven mathematically … as" → "is … --- … --- and is not the causal velocity cap, inactive here". The load-bearing corrections (measured-not-proven; the causal-cap clause) are both in.
2. **R2-7 demotion label drops two base words.** Base 380 reads "demoted from the main text **of this abstract**". `pj_sub` calls itself a paper (see B(c)), so the restored label reads "demoted from the main text and retained here in full; it is a supporting result, not one of this paper's claims." Register adaptation only.
3. **G.5 restores the fence, not the claim.** Base F.5 contains both the definition of $x$ (needed to resolve the orphan symbol) **and** the resonance-saturation claim. The worklist asks for the *fences*, not for a new claim; I restored the definition of $x$ and the δ=0.3 breakdown fence and **did not** import the resonance-saturation claim. Consequently base F.6's clause *"'resonance saturation' is an exact algebraic statement … not a phenomenological fit"* is also omitted — it fences a claim that is not in this paper. Everything else in F.6 is restored.
4. **MF-1 labels both slopes** (task instruction: do not pick one). Reconciling `−0.956` vs `−0.961` into a single canonical statistic remains the Hub's, not this pass's.
5. **Appendix A prose has no ancestor.** Base's App A is figure-only; the referee's SF-5 asks for three pointing sentences. Per rule 2 this is connective prose written for this pass in `pj_sub`'s register; it asserts **no number and no claim**, only cross-references (§4.4, §4.1, the figure) and the already-stated 3-seed/3000-epoch/λ=100 configuration. Flagged rather than silently naturalised.
6. **Punctuation-only change at the retention caption** (`respectively.` → `respectively,`) so the verbatim single-checkpoint rider attaches as its clause.
7. **Em-dash convention.** Restored riders use `pj_sub`'s unspaced `---`; `submission.tex` uses spaced ` --- `. Text is otherwise diff-identical to its base form.
8. **Not restored, per the task's own instruction:** DOIs and the other stripped bibliography annotations (`(preprint)` ×6, `(spotlight)`, `(short paper, …)`, `(preprint; never formally published)`). Only the claim-bearing `(preprint; single author)` was restored. ⭐ **One-word Head option:** say the word and the remaining annotations and 28 DOIs go back in one pass.
9. **Not on the worklist, therefore untouched — three items the Head may want ruled:**
   (i) **SF-7 / claims-table #24** — §1 line 35 still reads "Our results hold **generally** for the class of damped symplectic recurrences"; the base has no "generally". Fidelity ranked it *WIDER (trivial)*; the referee ranked it SHOULD-FIX as a C-5 scope-free generalization. **Left as the Head wrote it.**
   (ii) **App C (`app:retention`) still has no lead-in prose**; base 263's one-liner ("every number in it is stated there and none is added here") was not on the worklist.
   (iii) **Per-figure `\emph{Verification.}` / `\emph{Evidence.}` caption tags** were not restored (not on the worklist); the two-layer scheme survives in its §1 form at line 50.
10. **Two source-internal discrepancies remain unowned and were not touched** (both predate `pj_sub.tex`): the N46 per-seed values in `negative_results.md` vs `submission.tex:132`; and the un-pinned anchored slope. This pass *labels* the second; it does not resolve it.
11. **Nothing on the worklist was blocked.** Every Part-A item and every Part-B correction was executed.
