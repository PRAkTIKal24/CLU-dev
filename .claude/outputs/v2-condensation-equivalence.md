# v2-condensation-equivalence — doc-curator report

Task + acceptance criterion: Determine whether `.claude/NIPSsubmission/v2-neurreps/condensed_paper.tex` is the same paper as `~/Desktop/V2_NeurReps_Submission/paper.tex` with flow-only changes; deliver relocation map, loss list, addition list, rider-adjacency table, five-defect status. Zero edits.
Status: **done**

**DIAL DECLARATION (echoed):** none — read-and-report. Laundering control: n/a. Falsifies: n/a. Does NOT falsify: n/a.

⚠ **THIS REPORT CONTAINS A DOWNSTREAM RECONCILIATION LIST** (protocol §5 corollary). See **§7 — Reconciliation list for the Head/Hub**: 6 items the condensed file needs before submission. It needs an owner assigned at the review that accepts this report. I made **zero edits** (task-mandated), including zero edits to my four standing transfer docs — see §8.

**HEADLINE VERDICT (one line):** Content-wise the condensation is **materially faithful — zero numeric values lost, zero numeric values changed, zero citation keys lost, zero figures/tables/negative-rows lost** — but it is **not purely flow-only**: 3 sentences and 4 clauses were deleted or paraphrased rather than relocated, and **relocation has stripped the main text of both of its C-2 designed-verification riders and of the lifetime-metric standardisation rule**.

---

## 0. Method + honesty declarations

- Tools available: Read/Grep/Glob/Write only. **No shell.** Both files read in full, line by line, then per-file greps with positive controls.
- **Positive controls run** (memory hazard: directory-level grep over `.claude/` silently returns nothing — every negative below is per-file and controlled): `conformally symplectic` → 1 hit in condensed; `magnitutde` → 7 hits in condensed; `handover|Hub|wave` → 741 hits in `handover_context.md`. All negative sweeps below are therefore genuine absences, not grep failures.
- **NOT verified, declared rather than faked (needs a shell/build):**
  - The 6,185 vs 5,992 word counts. Not independently counted. (§3 offers an arithmetic reconciliation of the +193 that is consistent, but unverified.)
  - The predicted LaTeX warnings (`Label 'fig:pricelist' multiply defined`, the rendering of `model~\citet{...}`). These are derived from LaTeX/natbib semantics, **not observed from a log**. No build was attempted.
- **⛔ CORRECTION TO THE TASK'S STATED PREMISE.** The task states the condensed file "carries **more distinct citation keys (50 vs 48)**" and asks me to "list the 2 new citation keys." **There are none.** Both files contain **exactly 48 distinct citation keys, and the sets are identical** — no key added, no key dropped. I enumerated every `\cite*{}` command in both files and diffed the key sets.
  What actually grew is the number of **citation *commands***: **44 → 48 (+4)**, all re-uses of existing keys:
  1. `C:44` `\citet{jawahar_chlu_2026}` — new instance (the key's original instance relocated to `C:103`).
  2. `C:79` `\cite{mo_symmetry-protected_2026}` — new instance, replacing the baseline's uncited phrase "The equivariant-Lyapunov preprint (arXiv:2605.03338)".
  3. `C:149` `\citet{mo_symmetry-protected_2026}` — inside the **duplicated** Fig-2 caption in App E.
  4. Net +1 from §2: baseline `B:56` held 3 commands; condensed merges them into 1 clump at `C:48` **and** re-states all 3 at `C:119` (App C).
  ⇒ Whatever tool produced "50 vs 48" was not counting distinct keys. **Deliverable 3's "list the 2 new citation keys" resolves to: there are no new citation keys.**

Notation: `B:n` = baseline `paper.tex` line n. `C:n` = `condensed_paper.tex` line n.

---

## 1. Deliverable 1 — the relocation map

Baseline main text ran §1 Intro · §2 Related · §3 Setup · §4 Results {4.1 pricelist, 4.2 headtohead, 4.3 boundary, 4.4 cure} · §5 Discussion, then App A–I.
Condensed main text runs §1 Intro · §2 Related · §3 Setup · §4 Results {4.1 pricelist, 4.2 headtohead} · §5 Discussion, then App A–L. Main-text subsections drop 4 → 2.

### → App A `Extended Setup and Definitions` (`app:defs`, NEW section)
| Block | Was | Now | Note |
|---|---|---|---|
| Architecture description ("We analyze an architecture that advances a phase-space state $(q,p)$ … supply controllable forgetting.") | `B:36` §1 ¶2 | `C:103` | ⚠ **¶ split**: its last two sentences ("This work does not propose a new architecture… exactly-solvable underlying theory.") stayed in main text at `C:44`. See loss **L5**. |
| ¶ "Definitions and conversions" incl. the inertial-mass / spectral-mass bullets and the $\mu$/lifetime standardisation rule | `B:38–43` §1 | `C:105–110` | Verbatim except loss **L8**. ⛔ Rider consequence: see **D4 row 7**. |

### → App B `Curvature instantiation and coset storage` (`app:curcos`, NEW section)
| Block | Was | Now | Note |
|---|---|---|---|
| "If a learned potential $V_\theta$ is $G$-invariant but its minimiser is not… functions as the continuous register." (carries `watanabe_counting_2020,minami_spontaneous_2018`) | `B:34` §1 ¶1 | `C:115` | Verbatim. |
| "In this map, relaxation onto the orbit dictates the fast normal flow… effectively forming an exact latch." | `B:67` §3 | `C:113` | "In this map" → "In our CLU map". |
| ¶ "Curvature instantiation and coset storage." (the three $\mu^2$ states, tree-level Goldstone, coset coordinate, $(\varepsilon,\gamma,T)$ mapping) | `B:69` §3 | `C:115` | Verbatim (`Nambu--Goldstone` → `Nambu-Goldstone`, typography only). |

### → App C `Extended related work` (`app:relpos`)
| Block | Was | Now | Note |
|---|---|---|---|
| "Neutrality along group-orbit directions is a foundational principle… theoretical substrate for neural integrators." (3 separate `\citep`) | `B:56` §2 ¶1 | `C:119` | Restored verbatim, prefixed by a Head-authorised duplicate of the §2 opening sentence. Main-text §2 keeps a 5-key merged clump (`C:48`) — see loss **L9**. |
| All other `app:relpos` ¶¶ (Delineation / Protection mechanisms / Positioning boundaries) | `B:125–131` | `C:121–127` | Byte-identical. |

### → App D `Operational assumptions` (`app:opass`)
| Block | Was | Now | Note |
|---|---|---|---|
| **C-2 labelling sentence:** "Results derived from designed testbeds (e.g., invariant potentials, analytic tilts, spurions) serve as verification of the exact theory. Conversely, outcomes from learned-system… are presented as empirical evidence." | `B:52` — **§1 main text, immediately after the Contributions list** | `C:131` — **Appendix D, first sentence** | ⛔ Prefixed "In this early-stage work," and **the word "exact" deleted**. See **L7** and **D4 row 11**. |
| Experiment-D config sentence + the 4 scope bullets (Finite Temperature Latches / Kinetic Breaking / Tilt Limitations / Noise Scales) | `B:135–141` | `C:131–137` | Byte-identical. |

### → App E `Extended Results` (`app:resEx`, NEW section)
| Block | Was | Now | Note |
|---|---|---|---|
| Crossover second-order-signature ¶ (0.01% agreement; 3.2× divergence; $\varphi=0$; slope 0.5165; transport law ≤1%; ≤1.2e-15 rad) **and its trailing C-2 rider** | `B:79` — **§4.1 main text** | `C:153` | Numbers verbatim. Opening reworded to "We verify the crossover in Sec.\ref{sec:pricelist}…". ⛔ The trailing rider went with it — **D4 row 11(ii)**. |
| §4.3 "Where the CRR does not extend" (5.1/5.9/5.4e-2; 13–14 orders; washboard; 1–1.6 bits) | `B:101–105` — **main text §4.3** | `C:155–159` — **App E.1 subsection** | Verbatim. `\label{sec:boundary}` now resolves into the appendix. |
| §4.4 "The CRR survives the training-time correction" ($r^*\to0$, 8/8, 1000 ep, $\lambda=100$, $0.911\pm0.016$, 3000 ep, ≈20×, 1.5e-12, −0.956, 0.5165) | `B:109–113` — **main text §4.4** | `C:161–165` — **App E.2 subsection** | Verbatim. `\label{sec:cure}` now resolves into the appendix. |
| Fig. 1 (`fig1_gmor.png`) + Fig. 2 (`fig_lifetime_headtohead.png`) **full-width copies** | — | `C:142–151` | Head-authorised duplication. ⛔ But see additions **A-x1** (false caption text) and **A-x2** (duplicate `\label`). |

### Moved *within* the main text (not to an appendix)
| Block | Was | Now | Note |
|---|---|---|---|
| ¶ "Autonomous retention against learned baselines:" (5.6/56/69 map-steps; 1.2 rad; 263; 0.35 rad; 5 seeds; Apple M1; compute caveat) | `B:107` — inside §4.3 `sec:boundary` | `C:91` — inside **§4.2 `sec:headtohead`** | Verbatim. Moves **up** into main text. ⚠ Structural mis-nesting: a baselines paragraph now sits under the subsection heading "Evidence: The lifetime estimator in Mo et al. is the CRR's overdamped face". Its riders travelled with it — **D4 row 4 ✓**. |

### Unchanged appendices (verified line-by-line; every number, table cell, caption and bullet identical)
`app:anchor` (+ new "See Fig.~\ref{fig:sf3}."), `app:loan` (both tables), `app:retention`, `app:compute` (both tables + Confound ¶), `app:neg` (all 5 limitation bullets, the tree-level ¶, **both negative-results tables, all 9 rows**, the closing clarification ¶), `app:pos` (all 4 items), `app:gmor` (all 4 subsections, the figure, the roundoff-floor ¶, the expansion-variable subsection).

---

## 2. Deliverable 2 — ⛔ LOSS

### 2.1 The systematic sweep: what is NOT lost
Stated first so the losses below are read against a clean baseline.

- **Numeric values: ZERO lost, ZERO changed.** I walked every numeric value in the baseline — abstract, §1–§5, all 9 appendices, all 5 data tables, all 5 figure captions, both negative-results tables — and located each in the condensed file with **identical value, identical precision, identical ±, identical seed count, identical units**. This includes: `2.4e-15`, `3.2e-10`, `70`, `1.000000±5e-12`, `4.5 orders`, `−0.985 (n=35)`, `27.03`, `γ=0.05`, `0.01%`, `3.2×`, `0.5165`, `≤1%`, `1.2e-15 rad`, `0.35 rad`, `0.2 rad`, `15000`, `5 seeds`, `14 magnitudes`, `1.012±0.000`, `1.029±0.001`, `1.013`, `0.9987`, `δ=4`, `2.202±0.155`, `0.309±0.012`, `≈5×`, `5.1/5.9/5.4e-2`, `13–14 orders`, `1–1.6 bits`, `5.6/56/69`, `1.2 rad`, `263`, `8/8`, `1000`, `λ=100`, `0.911±0.016`, `3000`, `≈20×`, `1.5e-12`, `−0.956`, `−0.961`, `0.516`, `1.0000±1e-12`, `4.6 orders`, `7 overdamped δ`, `3.77±0.23×`, `σ*=√(M_iTγ(2−γ))`, `D_θ=εT(2−γ)/(2F²γ)`, `±0.05%`, `1.00/0.33`, `0.008/0.122`, `0.72/0`, `0.0128`, `0.190`, `2.4×`, `700`, `196`, `5000`, `0.20–0.23`, `±0.2%`, `4549/4557/4551`, the entire 6-row horizon/MSE table, `0.0047`, `0.2066`, `0.0216`, `92%`, `0.2548`, `−24%`, `0.778`, `0.706`, `5.2e-3`, `2.1e-2`, `6.2×`, `3.1×`, `14–15×`, `36148`, `1554`, `1186`, `2512`, `2400`, `252`, `500`, `3.8×`, `4.7×`, `54.8×`, `70.7×`, `23.5×`, `14.6×`, `7 repetitions`, `2e5`, `≈5 µs`, `0.98/1.18/5.4`, `5.4/1.6/0.082e-2`, `1.0000`, `116/442/959`, `1.1e-15`, `~12 orders`, `2.1e-3`, `+0.0994→−8.2846`, `+0.3291→−1.1980`, `τ_max=Γ/2α=4.0`, `0.140–0.343`, `1/dim=0.167`, `−0.53,−0.60,−1.04`, `+0.78,+0.63,+0.55`, `ℓ_θ/Δ up to 2.0`, `2/6`, `2/3`, `0.861`, `1.4e-5`, `2.6e-6`, `[−4.2e-3, 2.6e-2]`, `f=2,4`, `0.06–0.3%`, `γ*≈2εμ`, `8 checkpoints`, `[1e-8, 0.3]`, `1.33e-15`, `+16.1%→+210.1%`, `2.7e-14`, `1.1–2.2e-16`, `2.28e-8→4.22e-15`, `δ≥1e-4`, `~1e-12`, `O(tL/√N)`.
- **Citation keys: ZERO lost.** 48 = 48, identical sets (see §0).
- **Figures: ZERO lost.** All 5 image files (`fig1_gmor`, `fig_lifetime_headtohead`, `fig2_anchor_cure_laws`, `fig3_retention_overlay`, `fig3_gmor_condensate`) present; 2 of them now appear twice.
- **Captions: ZERO lost.** All 5 baseline captions present verbatim; 2 are duplicated with an added clause (see **A-x1**).
- **Tables: ZERO lost.** All 5 tables (horizon/MSE; variant/gap; unit/FLOPs; normalization; ×2 negatives) byte-identical.
- **Negative-results rows: ZERO lost.** All **9** rows across the two `app:neg` tables present verbatim, including the CM-17 sampler row and the Δ / ℓ_θ/Δ row.
- **Scope bullets: ZERO lost.** All 4 `app:opass` bullets, all 5 `app:neg` limitation bullets, all 4 `app:pos` items, the tree-level/probe-only ¶ and the δ→δ+δ_eff clause: all verbatim.

### 2.2 ⛔ The actual losses (7)

**L1 ⛔⛔ SUBSTRATE-SCOPE CLAUSE TRUNCATED — highest-severity loss in this pass.**
Baseline `B:61` (§3 Setup):
> "As in prior works like the CHLU, the Hamiltonian dynamics only define how the latent state evolves parametrized by some encoder, **and has no direct relation to the Hamiltonian of the inputs whether or not it is even possible to define one for a given set of input modalities such as images, text etc.**"

Condensed `C:53` (§3 Setup):
> "As in prior works like the CHLU, the Hamiltonian dynamics only define how the latent state evolves, parametrized by some encoder **and is unrelated to the input space without it.**"

Absence confirmed: grep `modalities` → 0 hits in condensed; grep `Hamiltonian of the inputs` → 0 hits in condensed (positive control `conformally symplectic` → 1 hit). **The clause exists nowhere in the condensed file.**
*Claim it used to support:* that the CLU's learned Hamiltonian is a parameterisation of **latent** dynamics and is **not** a model of the input data's physics — the program's standing "CLU is a latent information carrier, not a model of the data's dynamics" position. The replacement clause ("is unrelated to the input space without it") is a compressed paraphrase that (a) drops the explicit "no Hamiltonian of the inputs" statement, (b) drops the "whether or not one can even be defined" concession, and (c) drops the modality examples (images, text) that made the statement legible to an ML reviewer. This is **duplication-hazard class 3 (approved wording paraphrased)** applied to a single instance: the wording is not preserved verbatim anywhere.

**L2 ⛔ ABSTRACT LOSES THE C-5 SCALE QUALIFIER (dim 4, $S^1$).**
Baseline `B:29`: "As an initial empirical study, we measure this retention profile **at latent dimension $4$ on an $S^1$ testbed** and we find that the retention half-life follows $n_{1/2}\propto\mu^{-2}$."
Condensed `C:29`: "As an initial empirical study, we measure this retention profile and find that the retention half-life follows $n_{1/2}\propto\mu^{-2}$."
Confirmed: grep `S\^1 testbed` → condensed hits **only** `C:260` (App J limitation bullet); grep `latent dimension \$4\$` → condensed hits only `C:260`. The scale qualifier survives in App D and App J and in the Fig-1 caption, but **not in the abstract sentence that states the law.** See **D4 row 10**.

**L3 ⛔ ABSTRACT LOSES ITS BOUNDARY SENTENCE.**
Baseline `B:29`, final sentence: "**Finally, we identify the bounds for these results by identifying where the laws do not extend.**"
Condensed: deleted (grep `identify the bounds` → 0 hits). The abstract now ends on the survives-training-correction positive and never signals that the paper documents where the laws fail. Contribution 3 (`C:41`) still carries it, so the substance survives in main text — but the abstract's own scope-limiting sentence is gone.

**L4 ⚠ ABSTRACT'S MODEL DESCRIPTION DEMOTED TO A NON-RENDERING LaTeX COMMENT.**
Baseline `B:29`: "We evaluate a recurrent unit that advances a latent state $(q,p)$ via a damped symplectic velocity-Verlet step of a learned Hamiltonian that governs latent-space dynamics."
Condensed: this exact sentence is at `C:25` **preceded by `%`** — inside the author-block comment region, above `\begin{document}`. It will not appear in the built PDF. A different formulation of the same content is in App A (`C:103`). ⇒ the abstract no longer tells the reader what the model is. Flagged as loss-from-the-rendered-document, not loss-from-the-file.

**L5 ⚠ "symplectic recurrent model" → "recurrent model" (scope broadening).**
Baseline `B:36`: "…a closed-form analysis of latent retention on a trained **symplectic** recurrent model."
Condensed `C:44`: "…a closed-form analysis of latent retention on a trained recurrent model~\citet{jawahar_chlu_2026}."
grep `symplectic recurrent` → **0 hits in condensed** (positive control passes). Partially mitigated by the immediately following sentence, retained verbatim in both: "Our results hold generally for the class of damped symplectic recurrences." Still, the sentence that scopes the *contribution* now claims analysis of "a trained recurrent model" simpliciter.

**L6 ⚠ RELATED-WORK POSITIONING CLAUSE DELETED.**
Baseline `B:58`: "\emph{We address this gap} **by quantifying this relationship in closed form, outlining regime transitions (floors and crossovers) that cannot manifest in first-order dynamical systems.**"
Condensed `C:50`: "\emph{We address this gap}."
grep `regime transitions` → 0 hits in condensed. grep `first-order dynamical` → condensed hits only `C:81` ("…which standard first-order dynamical systems inherently miss"), so the *substance* survives one section later in §4.2, and "closed-form analysis" survives at `C:44`. Partial loss of the positioning statement, not of the claim.

**L7 ⚠ "verification of the exact theory" → "verification of the theory".**
grep `exact theory` → 0 hits in condensed (baseline `B:52`). The word "exact" was the pointer to the exactly-solvable core. Mild precision loss; not a claims escalation. (App L's "to verify the theory's exactness" survives in both.)

**L8 ⚠ Def-2 house-definition marker dropped.**
Baseline `B:41`: "**Spectral mass ($\mu_k$): Defined here as** the \emph{transverse curvature}, where $\mu_k^2=\lambda_k(\ldots)$".
Condensed `C:108`: "**Spectral mass ($\mu_k$): The** \emph{transverse curvature}, where …".
grep `Defined here as` → 0 hits in condensed. "Defined here as" marked this as a house definition (nomenclature ledger Def-2, inertial $M$ / spectral $\mu$) rather than a field-standard one. The new phrasing asserts it as the definition. Low severity, but it is the definitional sentence of the paper's core nomenclature.

**L9 ⚠ Attribution precision in main-text §2.**
Baseline `B:56` attributed three distinct claims to three distinct sources (`sagodi_back_2025` → marginally-stable-tangent / strictly-stable-normal manifold structure; `seung_how_1996` → neural integrators; the Golubitsky/Krupa/Rumberger triple → group-orbit neutrality). Condensed `C:48` collapses all of this into one sentence with a single 5-key clump. The per-claim attributions are restored verbatim in App C (`C:119`), so nothing is lost from the file — but main-text attribution granularity is.

---

## 3. Deliverable 3 — ⛔ ADDITION

**⛔ Top-finding check: there is NO new numeric value anywhere in the condensed file.** Every number in the condensed file has an exact ancestor in the baseline (§2.1). **And no new citation key** (§0). The condensation therefore does not introduce an unsupported quantitative claim.

### (a) Connective / structural prose — expected and fine
| # | Location | Text | Class |
|---|---|---|---|
| A-c1 | `C:44` | "See App.~\ref{app:defs} for definitions of terminology used in our scope." | connective |
| A-c2 | `C:59` | "For further map definition, the curvature instantiation and coset storage details see App.~\ref{app:curcos}." | connective |
| A-c3 | `C:67` | "See App.\ref{app:resEx} for further results on this." | connective |
| A-c4 | `C:89` | "For results on the CRR surviving training-time corrections and for results on where the CRR does not extend, see App.~\ref{app:resEx}." | connective |
| A-c5 | `C:168` | "See Fig.~\ref{fig:sf3}." | connective — also *fixes* a baseline dangling figure (App C's figure was never referenced in the baseline) |
| A-c6 | `C:113` | "In this map" → "**In our CLU map**" | framing |
| A-c7 | `C:131` | "**In this early-stage work,** results derived from designed testbeds…" | framing |
| A-c8 | `C:48` | "…converges on the continuous attractor **on a manifold**~\citep{…}" | framing |
| A-c9 | `C:29` | "The trained **latent** potential $V_\theta$ is $G$-invariant…" (baseline: "The trained potential") | clarifying |
| A-c10 | `C:21` | `\titlebreak` removed from the title | typographic |

### (b) Head-authorised duplication / restored items — not findings per the pre-authorisation
| # | Location | Content |
|---|---|---|
| A-d1 | `C:142–146` | Full-width duplicate of Fig. 1 (`fig1_gmor.png`) + its caption, duplicating `C:69–73`. Captions are character-identical between the two copies — **checked for hazard 1: every number agrees exactly** (−0.985, −1, 27.03, γ=0.05, 1.000000±5×10⁻¹², 4.5 orders, 5 seeds, dim 4). |
| A-d2 | `C:147–151` | Full-width duplicate of Fig. 2 (`fig_lifetime_headtohead.png`) + its caption, duplicating `C:83–87`. Character-identical; numbers agree (≈1, 2.2×, 0.31×). |
| A-d3 | `C:119` | App C re-opens with a duplicate of the §2 opening sentence ("The intersection of temporal deep learning dynamics… converges on the continuous attractor on a manifold.") so the appendix reads free-standing. Exactly the flow-duplication the Head authorised. |

### (c) ⛔ New text that is a defect (no new *claim*, but not clean)
| # | Location | Text | Why it is a finding |
|---|---|---|---|
| **A-x1** | `C:144`, `C:149` | "**(Figure downsized for space constraints, see App.\ref{app:resEx})**" — appearing inside the App E figure captions | The clause is **factually false in these two copies**: the App E figures are `width=\linewidth`, i.e. the *full-size* ones. And it is **self-referential**: App E's own captions direct the reader to App E. The clause is correct only on the main-text copies (`C:71` at `0.7\linewidth`, `C:85` at `0.33\linewidth`), where it is a legitimate addition. |
| **A-x2** | `C:145`, `C:150` | `\label{fig:pricelist}` and `\label{fig:lifetime_headtohead}` **redefined** (first defined at `C:72`, `C:86`) | Duplicate `\label`s. LaTeX will emit "Label … multiply defined" and any `\ref` to them resolves to the **App E** copy, not the main-text figure. *(Predicted from LaTeX semantics — no build run; declared.)* |
| **A-x3** | `C:44` | "…on a trained recurrent model**~\citet{jawahar_chlu_2026}**." | `\citet` after a tie renders as "…on a trained recurrent model Jawahar et al. (2026)." — malformed; should be `\citep`. New defect, not one of the five in Deliverable 5. |
| **A-x4** | `C:79` | "The **single-exponential estimator** presented in~\cite{mo_symmetry-protected_2026} isolates a finite memory lifetime using a **single-exponential estimator** driven by a Lyapunov exponent…" | Replaces baseline `B:91` "The equivariant-Lyapunov preprint (arXiv:2605.03338) isolates a finite memory lifetime using a single-exponential estimator…". Introduces a stutter, and uses bare `\cite` where every other citation in both files is `\citep`/`\citet`. Content-neutral; the arXiv identifier still appears at `C:123` and `C:219`, so nothing is lost. |

### Reconciling the +193 words (unverified arithmetic, no shell)
Additions ≈ +200 (two duplicated figure captions) +18 (A-d3) +~65 (A-c1–c5) +~10 (A-c6–c9, A-x1 on main-text copies) ≈ **+293**; deletions ≈ −40 (abstract L2/L3/L4) −25 (L1 truncation) −22 (L6) −~10 (L5, L7, L8, "We lay the foundations for the discrete map.", "Defined here as") ≈ **−97**. Net ≈ **+196**, consistent with the reported +193. ⇒ **the growth is fully accounted for by authorised duplication + connective prose; no hidden new content.**

---

## 4. Deliverable 4 — ⭐⭐ THE RIDER-ADJACENCY CHECK

Verdict key: **✓ intact** = claim and rider still co-located, no regression vs baseline · **⛔ SEPARATED** = the condensation moved one and not the other · **⚠ inherited** = a gap that exists identically in the baseline (not a condensation finding, but flagged).

| # | Claim / rider pair | Baseline location | Condensed location | Same section after the move? | Verdict |
|---|---|---|---|---|---|
| 1 | **CM-16a/b** — friction preserves (γ 0.05→0.2 lengthens memory 3.77±0.23×) / temperature erases (T>0 coset diffuses, $D_\theta=\varepsilon T(2-\gamma)/(2F^2\gamma)$, finite lifetime) | App B, bullet 1 (`B:137`) — both halves in **one bullet** | App D, bullet 1 (`C:133`) — both halves in **one bullet**, verbatim | **Yes** | **✓ intact.** CM-16 is never cited whole in either file; the 3.77× number appears exactly once, inside the bullet that also states the T=0 restriction and "All Sec.~\ref{sec:results} measurements are conducted at $T=0$." No duplication ⇒ no hazard-1/2/3 exposure. |
| 2 | **CM-17** — relativistic no-go is a failure **of the sampler**, not the thermodynamics; never "has no equilibrium" | App G, negatives table 1, row 5 (`B:264–266`) | App J, negatives table 1, row 5 (`C:288–290`) | **Yes**, single instance, verbatim | **✓ intact & compliant.** Wording is sampler-scoped: "it proves FDT-exact strictly in Newtonian execution. Under relativistic operations, **no noise scale converges the chain to a Gibbs invariant**." No "has no equilibrium" phrasing anywhere in either file. |
| 3 | **FDT-consistent noise + Newtonian kinetic mode required for every finite-temperature result** | App B, bullet 4 (`B:140`) | App D, bullet 4 (`C:136`), verbatim | **Yes** — same appendix as the only finite-$T$ result (bullet 1), two bullets away | **✓ intact.** ⚠ inherited: neither file states this in main text — but neither file reports a finite-$T$ number in main text either, so adjacency holds in both. The $\sigma^\star_i=\sqrt{M_iT\gamma(2-\gamma)}$ scale is duplicated at App D bullet 4 and App J table row 5 — **checked: identical, both instances FDT-labelled.** |
| 4 | **CM-4** — retention advantage does not survive compute normalisation; retired as a compute claim | Contribution 3 (`B:50`, main) + in-¶ caveat (`B:107`, main §4.3) + App F | Contribution 3 (`C:41`, main) + in-¶ caveat (`C:91`, main §4.2) + App I | **Yes** — the caveat is inside the same paragraph as the 263/69/56 numbers and travelled with it | **✓ intact — in fact improved.** The rider ("the compute efficiency of the CLU is much worse (see App.\ref{app:compute})… defer compute optimization to future work") sits in the same sentence-block as the claim, now on an earlier main-text page. App I's "necessitates our retirement of the naive 'map-step' claim as a formal metric of compute efficiency" is verbatim. |
| 5 | **CM-1** — the loan called at ≈700 steps; boundedness, not plateau | App D ¶2 (`B:157`) | App G ¶2 (`C:181`), verbatim | **Yes**; the qualifier "The constrained model's primary asset is **bounded execution by construction, not achieving the lowest steady-state MSE**" is the immediately preceding sentence in both | **✓ intact.** ⚠ inherited (NOT a condensation finding, flagged for the Hub): the sentence itself reads "the physically-informed unit maintains a **bounded plateau** of ≈0.20–0.23" — the word "plateau" is the one CM-1 cautions against. Identical in both files; the condensation neither introduced nor fixed it. |
| 6 | **N46** — designed-only scope on the emergent negative, **with** the counterexample clause | Main text §4.3 (`B:105`) + App G δ→δ+δ_eff clause (`B:243`) | **App E.1** (`C:159`) + App J δ→δ+δ_eff clause (`C:267`) | Counterexample clause **travelled with the claim** ✓ ("**While the damping-optimum retention law mathematically holds here**, the continuous coset register itself is strictly a designed feature, not learned."). **But the whole pair left the main text.** | **⛔ SEPARATED at main-text level.** Contribution 3 (`C:41`, main text) still asserts "flat directions must be designed rather than trained for; the continuous register poorly transfers to an emergent potential" — and now the counterexample clause, the 5.1/5.9/5.4e-2 evidence and the 13–14-orders framing are **appendix-only**. In the baseline a main-text reader met the qualified version in §4.3. **A rider in Appendix E does not qualify a claim on page 2.** |
| 7 | **Every lifetime naming its metric** — "what the broader literature refers to as *memory lifetime* is standardized here as the half-life $n_{1/2}$, measured in map applications. Metrics such as time constants ($\tau$) or diffusion coefficients ($D$) are evaluated and reported explicitly alongside the condition that defines them, as these metrics bifurcate past critical crossover thresholds." | **§1 main text** (`B:43`) | **App A** (`C:110`) | **No** | **⛔ SEPARATED.** Main text §4.1 now quotes $n_{1/2}$ values (floor 27.03, slope −0.985) and §4.2 quotes the Mo estimator's "measured-to-predicted lifetime ratio" (1.012±0.000 … 0.309±0.012) with the standardisation rule sitting in Appendix A. The very sentence warning that these metrics **bifurcate past the crossover** is now in an appendix, while the main text asserts a bifurcating ratio. |
| 7b | **The Δ and $\ell_\theta/\Delta$ reporting rule** — "No $n_{1/2}$ may be quoted without its $\Delta$ and $\ell_\theta/\Delta$." | App G negatives table 2, row 1 (`B:277`) | App J negatives table 2, row 1 (`C:301`), verbatim | **Yes** (single instance) | **✓ unchanged / ⚠ inherited.** Both files quote main-text $n_{1/2}$ values without $\Delta$; the rule is scoped in-row to the designed-vs-emergent finite-$T$ discrimination context. No regression introduced by the condensation. |
| 8a | **Substrate-scope sentence** (registers vs end-to-end) — "These laws govern a latent memory register; end-to-end performance additionally depends on the encoder, measured separately, with its parameter and state-byte budget ledgered." | App G bullet 5 (`B:240`) | App J bullet 5 (`C:264`), verbatim | **Yes**, single instance | **✓ intact.** |
| 8b | **Substrate-scope sentence** (latent dynamics ≠ input dynamics) | §3 Setup (`B:61`) | §3 Setup (`C:53`) — **same section** | Same section, but **the text changed** | **⛔ PARAPHRASED — hazard class 3.** See **L1**. Section adjacency is preserved; the *wording* is not, and the modality clause is gone from the whole file. |
| 9 | **The score sentence in its measured form** | — | — | — | **N/A for this document.** grep `score` → **0 hits in BOTH files** (positive controls: `conformally symplectic` 1 hit in condensed; `magnitutde` 7 hits in condensed). This rider belongs to a different short; it is neither present nor lost here. Flagged to the Hub in case the rider list was meant for a different object. |
| 10 | **C-5 scale qualifiers in-sentence** (dim 4, $S^1$, ≤5 seeds, laptop CPU) | abstract `B:29` (dim 4 + $S^1$) · Fig-1 caption `B:83` (5 seeds, dim 4, γ=0.05) · §4.2 `B:91` (5 seeds, 14 magnitudes) · §4.3 `B:107` (Apple M1, all 5 seeds) · App B `B:135` · App G `B:236` | abstract `C:29` **← qualifier deleted** · Fig-1 caption `C:71` **and** `C:144` ✓ · §4.2 `C:79` ✓ · main-text ¶ `C:91` ✓ (Apple M1, all 5 seeds) · App D `C:131` ✓ · App J `C:260` ✓ | Mixed | **⛔ ONE REGRESSION: the abstract.** Every other C-5 carrier survives, in-sentence, and the two duplicated Fig-1 captions both carry the full setup line (hazard-2 checked: **neither copy is unqualified**). But the abstract's law statement — the single most-read sentence — no longer names dim 4 or $S^1$. See **L2**. |
| 11 | **C-2 designed-verification vs learned-evidence labelling** | **two main-text carriers:** (i) §1 `B:52`, immediately after the Contributions list; (ii) §4.1 `B:79` final sentence, "These results utilize analytic tilts on trained checkpoints, serving as **exact verification of the theory rather than emergent discoveries**" | (i) → **App D** `C:131` (and "exact" dropped, **L7**); (ii) → **App E** `C:153` | **No — both left the main text** | **⛔⛔ DOUBLE SEPARATION — the most consequential rider finding of this pass.** Condensed §4.1 main text (`C:67`) now presents $\mu^2\le2.4\times10^{-15}$, the $3.2\times10^{-10}$ constitutive identity over **all 70 tilt configurations**, $1.000000\pm5\times10^{-12}$ "at every tilt and seed", the $-0.985$ slope and the 27.03 floor **with no main-text statement that these are designed-testbed verification of an exact theory rather than emergent findings on a learned system.** The only surviving main-text designed-scope marker is four words inside the Fig-1 caption ("Trained designed checkpoints, analytic tilts"). Combined with **L2** (abstract loses dim 4/$S^1$) and **row 6** (the emergent negative is now appendix-only), a main-text-only reader of the condensed version gets a materially less-qualified paper than a main-text-only reader of the baseline — **without a single number having been edited.** This is exactly the failure mode this pass exists to catch. |

### Duplication hazards 1–3: explicit findings
Per the Head pre-authorisation I report duplication **only** under the three heads. Results:

- **Hazard 1 (a number that appears twice and disagrees): NONE FOUND.** Every duplicated numeric value was checked for value, precision, ±, seed count and units. The duplicated Fig-1 and Fig-2 captions are character-identical to their main-text twins. Cross-location duplicates checked and agreeing: 27.03 (`C:67`, `C:71`, `C:144`, `C:171`); 5.6/56/69 (`C:91`, `C:219`); 263 (`C:91`, `C:225`, `C:246`); 69/56 (`C:91`, `C:225`, `C:246`); 0.35 rad (`C:91`, `C:219`); 1.2 rad (`C:91`, `C:219`); 0.2 rad threshold (`C:79`, `C:219`); $0.911\pm0.016$ (`C:165`, `C:281`); 8/8 at 1000 ep (`C:165`, `C:281`); λ=100 (`C:165`, `C:171`, `C:281`); 3000 ep / ≈20× (`C:165`, `C:171`); 1–1.6 bits (`C:159`, `C:284`, `C:326`); 2.4e-15 (`C:67`, `C:159`); 1.000000±5e-12 (`C:67`, `C:71`, `C:144`, `C:333`); 3.2e-10 over 70 rows (`C:67`, `C:333`); 4549 (`C:183`, `C:234`); $\sigma^\star_i$ (`C:136`, `C:290`); 1.33e-15 (`C:337`, `C:343`); 3.2× (`C:81` misprediction / `C:153` bifurcation — **distinct quantities**, same in baseline).
  - ⚠ Two *inherited* precision inconsistencies, present identically in the baseline, neither introduced by the condensation, both flagged for the Hub:
    (i) **EP onset: `0.5165` (`C:153`, `C:165`) vs `0.516` (`C:171`).** Same quantity, one rounded. Condensation side-effect: they are now ~18 lines apart in adjacent appendices (E and F) rather than split across main text and appendix, so the mismatch is *more* visible than before.
    (ii) **`±0.05%` (`C:179`) vs `±0.2%` (`C:183`)** — the known parameter-match contradiction; see Deliverable 5.
  - ⚠ Three *slope* values coexist and are **not** a hazard-1 pair: `−0.985` (baseline 150-ep, $n=35$, `C:67`), `−0.956` (anchored 3000-ep, "the per-point fit over all overdamped rows", `C:165`), `−0.961` (anchored 3000-ep, "the seed-mean OLS over the $7$ overdamped $\delta$", `C:171`). **Each carries its fit method and dataset in-sentence, in both files.** Identical to baseline. This is the pattern the task warned about (one slope, two fits) and it is correctly disambiguated in both files. Condensation improved main-text exposure: only `−0.985` now appears in main text.
- **Hazard 2 (claim duplicated, only one copy carries its rider): NONE FOUND among duplicated passages.** The two duplicated figure captions are character-identical, so both carry "Setup: Trained designed checkpoints, analytic tilts, 5 seeds, dim 4, γ=0.05." The App C duplicated opening sentence (A-d3) carries no claim requiring a rider. *(The rider failures found in rows 6, 7 and 11 are* separation *failures, not duplication failures — the rider exists once and moved.)*
- **Hazard 3 (approved wording / mandatory rider duplicated but paraphrased): ONE FOUND, and it is a single-instance paraphrase rather than a duplicate — reported anyway because the wording is binding.** **L1**, the substrate-scope Setup sentence. Secondary, lower-severity: **L7** ("exact theory" → "theory") and **L8** ("Defined here as" dropped).

---

## 5. Deliverable 5 — status of five known defects

| # | Defect | Baseline | Condensed | Status |
|---|---|---|---|---|
| 1 | `isrelationship` — broken sentence inside Contribution 1 | `B:47–48`: "We establish the\n isrelationship, identify the crossover boundary…" | `C:38–39`: **byte-identical**, including the mid-sentence line break | ⛔ **PRESENT — not fixed.** |
| 2 | `magnitutde` misspelling | 6 lines (`B:79, 83, 147, 263, 309, 319`) | **7 lines** (`C:71, 144, 153, 171, 287, 333, 343`) | ⛔ **PRESENT — and one occurrence MULTIPLIED** by the Fig-1 caption duplication (`C:71`→`C:144`). *(Also present in both: `maturiity`, `C:91` / `B:107`.)* |
| 3 | Backtick figure reference ``Fig.\` `` | `B:77`: ``Fig.\`\ref{fig:gmor}`` | `C:67`: `Fig.\ref{fig:gmor}` — backtick removed. Sweep of all backticks in the condensed file returns only legitimate LaTeX open-quotes (`C:105` ``mass''`, `C:121` ``flow''`, `C:227` ``map-step''`) | ⭐ **FIXED.** ⚠ **But the underlying mis-reference survives**: `C:67` says the $-0.985$ overdamped slope is "as shown in **Fig.\ref{fig:gmor}**", and `fig:gmor` is the App L GMOR-condensate figure, **not** the headline Fig. 1 that actually shows that slope. Removing the backtick converts a visibly-broken reference into a clean-rendering **wrong** reference — arguably a net regression in detectability. |
| 4 | `±0.05%` vs `±0.2%` parameter-match contradiction | `B:155` ("$\pm0.05\%$ parameters") vs `B:159` ("parameter-matched to $\pm0.2\%$ (CLU $4549$ / broken-volume $4557$ / twin $4551$)") | `C:179` vs `C:183` — **byte-identical, same 4-line separation, same appendix** | ⛔ **PRESENT — not fixed.** *Observation (arithmetic, offered as an observation not a verdict): 4557/4549 = +0.176% and 4551/4549 = +0.044%, so the spread across the three models is consistent with ±0.2% and inconsistent with ±0.05%. Which figure is correct is the owning vertical's call, not mine.* |
| 5 | Headline figure's label referenced by the text? (`\ref{fig:pricelist}`) | grep `ref{fig:pricelist}` → **0 hits** | grep `ref{fig:pricelist}` → **0 hits** | ⛔ **STILL NOT REFERENCED — and WORSE.** `fig:pricelist` is now `\label`ed **twice** (`C:72`, `C:145`), as is `fig:lifetime_headtohead` (`C:86`, `C:150`). LaTeX will warn "Label multiply defined", and if anyone later adds `\ref{fig:pricelist}` it will resolve to the **App E** copy, not the main-text figure. The headline figure of the paper is still never pointed at by any sentence. *(Multiply-defined warning predicted from LaTeX semantics; no build run — declared.)* |

---

## 6. Answer to the Head's question

*"Make sure content-wise everything is the same and just the flow is changed."*

**Substantively yes, with seven exceptions and one structural class of exception.**

- **Quantitatively: identical.** Zero numbers lost, zero numbers changed, zero new numbers, zero citation keys added or dropped (the "50 vs 48" premise does not survive checking), zero figures/tables/captions/negative-rows/scope-bullets lost. The whole appendix corpus from `app:loan` onward is byte-identical.
- **Relocation: clean and well-formed** — five destination appendices, every relocated block accounted for (Deliverable 1).
- **Not flow-only, in three ways:**
  1. **Seven deletions/paraphrases** (L1–L9 in §2.2) that are not relocations. The one that matters is **L1**, the truncated substrate-scope clause — the wording that separates latent dynamics from input-space dynamics no longer exists anywhere in the file.
  2. **The abstract lost two things** it used to carry: the dim-4/$S^1$ scale qualifier (**L2**) and the boundary sentence (**L3**); a third, the model description, now renders nowhere because it was demoted to a `%` comment (**L4**).
  3. **Rider adjacency degraded in three places** (D4 rows 6, 7, 11). Both C-2 designed-verification riders and the lifetime-metric standardisation rule left the main text; the emergent-register negative and its counterexample clause left the main text. The main text's quantitative content is unchanged but is now **less qualified** than the baseline's, with no word having been edited — which is precisely the hazard this pass was commissioned to find.
- **The growth is fully explained and benign**: authorised figure/prose duplication plus five cross-reference sentences. No new claim rode in with it.

---

## 7. Reconciliation list for the Head/Hub (needs an owner — protocol §5 corollary)

Ordered by severity. **I made no edits; these are for whoever owns the file.**

1. ⛔⛔ **Restore a C-2 designed-verification marker to the main text.** At minimum, return "These results utilize analytic tilts on trained checkpoints, serving as exact verification of the theory rather than emergent discoveries" to the end of §4.1 (`C:67`), *and/or* return the `B:52` labelling sentence to the Contributions block. Currently both carriers are in appendices.
2. ⛔⛔ **Restore the substrate-scope wording (L1)** verbatim at `C:53`, or wherever the Head prefers — the "no direct relation to the Hamiltonian of the inputs… images, text etc." clause is absent from the whole file.
3. ⛔ **Restore "at latent dimension $4$ on an $S^1$ testbed" to the abstract** (`C:29`) — the C-5 in-sentence scale qualifier.
4. ⛔ **Fix the duplicate `\label`s** (`C:145`, `C:150`) — rename the App E copies — and **fix the false "(Figure downsized for space constraints, see App.\ref{app:resEx})" clause in the App E captions** (`C:144`, `C:149`), which is both untrue there and self-referential.
5. ⚠ **Decide whether the abstract should keep a boundary sentence (L3) and a rendered model description (L4** — currently commented out at `C:25`**).**
6. ⚠ **Carried-over defects, unchanged and still live:** `isrelationship` (`C:39`), `magnitutde` ×7, `maturiity` (`C:91`), `±0.05%` vs `±0.2%` (`C:179`/`C:183`), `fig:pricelist` never referenced, and `C:67`'s `Fig.\ref{fig:gmor}` pointing at the wrong figure. Plus one **new** one: `C:44`'s `model~\citet{jawahar_chlu_2026}` should be `\citep`.

---

## 8. Curator-doc note (why my four standing docs are untouched)

My standing remit is `HEP_primers.md`, `philosophy-synthesis.md`, `negative_results.md`, `future_work.md`. **This task declares ⛔ ZERO edits to any file, my report being the only write — so all four are untouched, deliberately.** Two items would otherwise have been folded in, and I flag them for the Hub to task separately if it wants them landed:

- **→ `negative_results.md` / paper-appendix mining:** nothing new. All 9 negative-results rows in `app:neg` are unchanged between the two files; the condensation neither added nor dropped a negative. The registry needs no delta from this pass. *(Confirming, for the paper-writer: the V2 appendix negatives table is stable across the condensation and can be mined from either file.)*
- **→ `philosophy-synthesis.md` (⟲ append-only):** the condensation itself is a ledger-relevant event only insofar as **CM-1's "bounded plateau" phrasing** (D4 row 5) and the **0.5165/0.516** and **±0.05%/±0.2%** inconsistencies are *pre-existing* and survive into the submitted artefact. These are not new wave findings; they are standing exposures I re-confirmed. No addendum written.
- **Handover check:** grep of `.claude/handover_context.md` for `condensed_paper|condensation|Addendum 75|v2-neurreps` → **0 hits** (positive control `handover|Hub|wave` → 741 hits, so this is a genuine absence). **This task and its object are not yet recorded in the handover doc.** The Hub will need to log the commission and this report.

Git footprint: none — no tracked file touched, no code changes, no branch.

Open questions / risks:
1. **The task's "50 vs 48 citation keys" and "6,185 vs 5,992 words" premises.** I disproved the first (48 = 48, identical sets) and could not verify the second (no shell). If the same tool produced both figures, the word counts may also warrant re-checking; my §3 arithmetic is consistent with +193 either way.
2. **The "score sentence in its measured form" rider (D4 row 9)** has no referent in either file. Either the rider list was assembled for a different short, or a score-based claim was expected in V2 and never landed. Flagged, not resolved.
3. **No build was run**, so multiply-defined-label warnings, the `\citet`-after-tie rendering, and the actual page count of the condensed main text (the "~4.5 pp" target) are unverified. Someone with a shell should compile before submission — the duplicate labels are the item most likely to surface as a visible artefact.
4. **I did not resolve any verdict conflicts** and found none between the two files' numbers. Where a value differs (0.5165/0.516; ±0.05%/±0.2%; the three slopes), I have named both instances and their qualifiers and left the resolution to the owning vertical.

## Proposed handover updates (for the Hub)
- **New §10 review entry needed** — this task is absent from `handover_context.md` (verified by controlled grep). Suggested substance: *"w-current: `v2-condensation-equivalence` (doc-curator, read-only). Verdict — the V2 condensation is quantitatively identical to the built baseline (0 numbers lost/changed/added; 48 = 48 distinct citation keys, task's '50 vs 48' premise disproved; all figures/tables/captions/9 negative-rows/scope-bullets preserved; `app:loan`→`app:gmor` byte-identical). Not flow-only: 7 deletions/paraphrases (top: the Setup substrate-scope clause 'no direct relation to the Hamiltonian of the inputs… images, text etc.' is absent from the whole file) and 3 rider-adjacency regressions (both C-2 designed-verification carriers and the lifetime-metric standardisation rule left the main text; the emergent-register negative is now appendix-only; the abstract lost its dim-4/S¹ C-5 qualifier). Backtick `Fig.\`` FIXED; `isrelationship`, `magnitutde`×7, `±0.05%`/`±0.2%`, unreferenced `fig:pricelist` all still live; two NEW build defects (duplicate `\label`s at C:145/C:150; false 'downsized' clause in the App E captions). 6-item reconciliation list in `.claude/outputs/v2-condensation-equivalence.md` §7 needs an owner."*
- **Assign an owner for §7.** Items 1–4 are claims-exposure or build-breaking and should land before submission; items 5–6 are Head-discretionary/cosmetic.
- **Consider a small follow-up with a shell**: compile `condensed_paper.tex`, confirm the multiply-defined-label warnings and the main-text page count against the ~4.5 pp target, and re-derive the word counts. I could not.
- **Nomenclature watch (Def-2):** the condensed file drops "Defined here as" from the spectral-mass definition (`C:108`). If the primer's nomenclature ledger quotes that sentence as the canonical house definition, the two will now differ in wording. Low priority; noted rather than acted on, per the zero-edit directive.
