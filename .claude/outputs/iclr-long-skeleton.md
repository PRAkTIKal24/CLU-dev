# iclr-long-skeleton — paper-writer report (PLANNING; structure + claims→evidence map + gap list)
Task + acceptance criterion: a recommended ICLR-long thesis (alternatives argued); an 8–9pp section structure (main vs appendix); a complete claims→evidence map (honest status per claim); a prioritized gap list mapping every gap to an experiment (unassigned ones flagged); the dual-submission note. This is the ICLR-long's spine + the w16+ experiment-priority driver.
Status: **done** (planning only — no prose drafted, no results fabricated; every claim carries CM row + source report + PENDING flags where evidence does not yet exist).

**First-line owner flag (protocol §5 corollary — reconciliation list for the Hub):**
1. **The load-bearing accept-maker (real-data CLU-vs-floor on voraus) does NOT exist yet.** `g7b-torus-voraus` is scoped+ready but unlaunched; the only real-data numbers we have are a `--quick` smoke where **CLU sits BELOW baselines** (`clu-anomaly-scorer`, AUROC 0.38–0.51, explicitly NOT a claim) and a 24-channel laptop pre-smoke of the *baseline floors* (`voraus-baseline-floors`, knn 0.77). **The recommended thesis therefore forks on the voraus outcome (§1 below). This fork is the single most important planning fact.** Owner: Hub, at the g7b-torus-voraus review.
2. **CM-10 wording is now N≤16** (was "2-unit trained lattices") per `v3-pricing-n-scaling` (Hub-recovered) — the long inherits the extended wording, NOT the short's stale N=2 scoping. `v3-revision-5` is folding this into the V3 short; the long must draft from the extended form.

---

## What I did
- Read (in order): AGENT_PROTOCOL, task file, `claims_matrix.md` (all CM rows), the Positioning Charter (philosophy-synthesis §"Positioning Charter", C-1…C-10 verbatim) + the ledger's ICLR gap analysis + scorecard + positioning-consequences, `research_roadmap.md` (the two-flagship read: ICLR = the systems story = V1+V3+real data), `handover_context.md` w15/w16 runway.
- Read the three assembling sources: `papers/v3-short/draft.md` (structure, contributions, figures, appendix map), `papers/v1-short/draft.md` (certificate stack, three pillars, appendices), and the real-data surface: `clu-anomaly-scorer.md`, `voraus-baseline-floors.md`, `xy-1d-control.md`, `v3-pricing-n-scaling.md`, and `g7b-torus-voraus.md` (the pending flagship task).
- Cross-checked every load-bearing claim against its CM row + source report; classified evidence grade per Charter C-2 (verification vs evidence vs structural) and per protocol (verified/trained-model/synthetic-only/PENDING).
- Wrote this report + `papers/iclr-long/outline.md` (the section stub).

---

## 1. THESIS — recommendation + alternatives argued

**Venue frame (binding, from the roadmap two-flagship read):** ICLR long = **the systems story** ("*the how / does-it-work*"); Nature-MI owns the *why/what* (physics). They share only the CLU definition + F5 formalism (cited, not the contribution). ⇒ the ICLR thesis must read **ML-first (Charter C-3)** and compose **V1 (certified test-time compute) + V3 (firewall / pricing / reversibility) + real industrial data**, not the physics per se.

### Recommended thesis (primary)
> **Conservation-by-construction is a usable deep-network substrate: a single symplectic memory primitive whose stability (BIBO), interference-firewall, priced inter-unit communication, certified test-time compute, and O(1)-memory reversible training all *survive composition* into deep lattices — and hold on real industrial robot time-series — provided the governing symmetry is *designed in*, not awaited.**

Two load-bearing halves:
- **Positive systems claim** (the ML contribution, venue-appropriate): guarantees-by-construction that compose (firewall degree-bounded not width-linear; pricing law flat in N≤16; reversible O(1)-in-T; certificates with distribution-free refusal) → *demonstrated on a real benchmark* (voraus-AD, robot joints = literal torus the theory predicts).
- **The designed-precondition clause** (the honesty spine, and the paper's own answer to its killer critique): the cross-thread insight the program owns from **five independent directions** — *emergent systems have no coset register* (12-order softness gap, `v5-gate` R1 / N46 / CM-16a); *the XY/KT reduction needs a designed channel-restricted coupling*, a generic learned `W` breaks U(1) (`xy-1d-control`, `xy-lattice-theory`); *conformal init recovers it for free* (`lattice-xy-prereqs`); *the mass hierarchy is designed-in, never emergent* (3× "no hierarchy," N7 / CM-5); *the pricing law is only readable on U(1)-preserving `channel_spring` lattices* (App C artifact, `v3-pricing-n-scaling`). This is **Hyp-3** ("designed-in or induced, not awaited"). Folding it into the thesis as a *scoping clause* converts the reviewers' biggest attack ("designed structure, doesn't emerge — a gauge artifact") into a **stated, evidenced design principle**.

**Why this over the pure alternatives:** it is the only framing that (a) reads ML-first (C-3), (b) composes all of V1+V3+real-data under one sentence, and (c) pre-answers the designed-vs-emergent critique instead of hiding it in §Limitations.

### Alternative framings (honestly argued, per task)

**Alt-A — "Designed symmetry is the precondition, not a convenience" as the *headline*** (the ledger's nominated strongest thesis). *For:* it is the deepest, most defensible thing the program owns; five-direction support; nothing else is as robust. *Against for ICLR:* it is a **meta / partly-negative** finding ("your structure won't emerge") — an NMI/position-paper register, not a systems-contribution headline. An ICLR systems reviewer wants "a primitive that does X on real data," and a headline that leads with a precondition reads as a caveat. **Recommendation: keep it as the recommended thesis's *scoping clause*, not the headline.** (It becomes headline-weight only in the voraus-loses fork, below.)

**Alt-B — pure systems-benchmark framing** ("scales to deep nets with certified stability, validated on real data — and it wins"). *For:* cleanest ICLR shape. *Against:* **it is not true yet and may never be** — the real-data win is PENDING and the smoke currently shows CLU *below* strong statistical baselines (knn 0.77 on voraus; `clu-anomaly-scorer` CLU 0.38–0.51 on the tiny smoke). Leading with an unearned "it wins" is exactly the R5/CM-3 trap. **Reject as written; adopt only its measured half (guarantees compose) unconditionally.**

**Alt-C — G7c "Pareto-not-podium: memory with a predictable lifetime"** (the honest-Pareto framing). *For:* survivable even if CLU never tops a leaderboard — the claim becomes "physically-motivated (torus matches the data's own topology), certified, and with a *measured, predictable* memory lifetime and interference budget, Pareto-competitive with baselines rather than dominating." *Against:* weaker sell; needs the voraus result to at least *tie* / Pareto-match somewhere. **This is the designated FALLBACK thesis.**

### The fork (the plan the Hub must hold)
All three framings **share the same section skeleton (§2)**; only the abstract + §1 framing sentence + headline figure differ. The fork resolves at the `g7b-torus-voraus` result:
- **voraus CLU Pareto-dominates or wins some category** → **Recommended thesis**, headline figure = the voraus per-category Pareto/ROC panel.
- **voraus CLU ties / Pareto-matches** (likely, given the smoke) → **Alt-C (G7c) fallback**; headline stays the **interference-scaling figure** (the strongest *measured* systems result we own), voraus is "physically-motivated, certified, Pareto-competitive," and the **designed-precondition clause (Alt-A) gains headline weight**.
- **voraus CLU clearly loses everywhere** → the long is **not submittable as a systems paper**; pivot to Alt-A as a position/theory-of-composition paper, or hold for the next cycle. **Flag for the Head now.**

---

## 2. SECTION STRUCTURE (8–9pp ICLR main track; main vs appendix)

Naming discipline (binding): the continuity sentence in §2 — *"the CLU, introduced as CHLU in Jawahar & Pierini (2026)"*; nomenclature **inertial M vs spectral μ**, never bare "mass" (HEP_primers ledger, CM §1). Citations hermetic (C-8/M1): J&P 2026 + the F5 note (third person) only; no other program shorts exist as citations.

### Main text (target ~8pp)
- **§1 Introduction** — thesis (§1 above); **contributions enumerated on page 1** (C-3); the "what the physics buys and what it does not" composition owned up front (lift the V3 short's §1 device: firewall = parameter separation, physics = the priced channel you keep *while* modular); one headline figure named. **Charter bite: C-1 (no audit ¶), C-3 (ML-first).**
- **§2 The CLU primitive and the CLU-Net** — the unit (H=T(p)+V_θ(q), leapfrog, γ), the lattice composition, conserved charges; continuity sentence + M/μ nomenclature; F5 formalism cited as apparatus, not contribution.
- **§3 Guarantees that compose** — BIBO boundedness + the latch (CM-1; CM-7 payoff), conserved Noether charge (1−γ)ⁿ, **reversible O(1)-memory training** (CM-13, labeled **structural**). This is "what a single joint symplectic Hamiltonian keeps under composition."
- **§4 The interference firewall at scale (HEADLINE, V3)** — degree-bounded vs width-linear (CM-9), N≤16, 12 seeds; the parameter-separation-not-physics concession in-text (C-2 honesty). **Headline figure candidate #1.**
- **§5 The priced channel** — pricing law sync∝κ^−1/2, n₁/₂∝κ^−1 (CM-10), now **N∈{2,4,8,16} both topologies, 5 seeds, pre-registered** (`v3-pricing-n-scaling`); designed-lattice verification at N≤8 (C-2).
- **§6 Certified test-time compute (V1)** — paid-access reach/escape certificate (CM-12, **verification**, oracle-placement scope inline per C-6); calibrated escalatable compute-rationing gate (CM-2, **evidence**); one-hop edge not energy-gating (CM-7). **CM-3 forbidden (no energy-superiority); C-6 certificate fine print inline.**
- **§7 Real industrial data — the CLU on voraus-AD (the accept-maker)** — torus-coset CLU vs the statistical-baseline floors, **identical episode-AUROC protocol** (CM-3: physically-grounded, not superior-by-assertion). **PENDING (`g7b-torus-voraus`).** Headline figure candidate #2 (fork-dependent).
- **§8 Related work** — positioning lifted from scout reports (see §5 of this report for provenance).
- **§9 Limitations, scope, and the designed-symmetry precondition** — the Alt-A clause as owned scope; the negatives that bound main-text claims (noise wall, oracle placement, structural-not-trainer reversibility, N≤16 ceiling).

### Appendices (C-10 maximalism — fully written, nothing pruned now)
- **App A** Flag-provenance tables (C-7) — inherited verbatim from V1/V3 short appendices + the real-data provenance (`clu-anomaly-scorer`, `voraus-baseline-floors`, `g7b-torus-voraus` when it lands). **Every §-number's table.**
- **App B** Banding as a method not oracle gift (CM-11) + mass-lr doctrine full grid (CM-5, N7). *(supporting; not main-text for the systems long.)*
- **App C** The block-monolith + coordination-number controls (CM-9 tail); metric discipline (why basin-displacement R, never NTK cosine).
- **App D** The full regime map / cost story (CM-8) + Hopfield parity + the noise wall (N37).
- **App E** Certificate derivations — squeeze-MH kernel (CM-14), BIBO battery, analytic verifications (C-6 material).
- **App F** Reversible-O(1) full memory-vs-T tables + the γ>0 horizon (CM-13).
- **App G** Erosion + anchor cure (CM-6) — training-dynamics caveat for depth (C-9 future-work anchor).
- **App H** Prominent negatives registry (C-9): N7 (mass not emergent), N37 (noise wall), N46 (emergent has no coset — CM-16a designed-only), erosion horizon, retries-bought-nothing (CM-14/F.6), energy≈margin (CM-3), pricing n₁/₂ pointwise 47–56%, voraus smoke CLU-below-baseline.
- **App I** [PENDING] full voraus per-category floors (130ch CSF3) + the topology-match control; TEP-Rieth secondary.
- **App J** [optional, NMI-adjacent] the XY/KT designed-coupling reduction as designed-precondition evidence (`xy-1d-control`) — cross-referenced from §9, kept minimal (hermetic: it is same-program but appears only as this long's own appendix, not a citation).

**Main-vs-appendix rule:** everything mass-spectrum / forgetting-law / GMOR / relativistic-Gibbs / erosion is **appendix or NMI-side**; the main text carries only the five composing guarantees + certificates + the real-data result.

**Headline figure (name one, per craft rule):** **Figure 1 = interference scaling, degree-bounded vs width-linear (N∈{2,4,6,8,12,16}, 12 seeds)** — inherited from V3 short `figures/fig1_scaling_curve.png`. This is the strongest *measured* systems figure we own today and is fork-independent. **Reassign to the voraus per-category panel iff the g7b result is a win.**

---

## 3. CLAIMS → EVIDENCE MAP (core deliverable)

Grade key: **VER** = verification on designed/architecturally-invariant testbed (C-2, "not a discovery"); **EV-T** = evidence on trained/learned model; **EV-S** = synthetic/toy evidence; **STRUCT** = property of integrator/autodiff graph (untrained); **PENDING** = experiment not yet run. Every row's scope qualifier is **mandatory in-sentence (C-5)**.

| # | claim (load-bearing in the long) | CM row | grade | scale (dim/N/seeds/HW) | honest scope / negative that travels |
|---|---|---|---|---|---|
| L1 | BIBO boundedness + the latch by construction; the physics prior costs raw fit (~15× MSE); loan called ≈700 steps | CM-1, CM-7(payoff) | VER + EV-T | dim 2–4, synthetic, laptop, 3–5 seeds | mechanism = contraction-forbidden (volume), NOT the causal cap; receipt buys the latch not the jump; free ledger ≠ BIBO (coercive-membership clause); do NOT claim lowest long-horizon plateau |
| L2 | Reversible O(1)-in-T training at γ=0; grads match BPTT ≤2.1e-6 (f32); 946× peak-mem at T=1024,N=2 | CM-13 | **STRUCT** | untrained models, CPU, D≤16, 3 seeds | **NOT in `train_chlu`; not GPU-validated; exact ONLY at γ=0** (γ>0 horizon finite); mem metric = XLA scratch proxy; training-indistinguishability untested |
| L3 | Conserved Noether charge decays exactly (1−γ)ⁿ; latch machine-flat | CM §1 constants | VER | designed SO(2), f64 | verification, not discovery (C-2) |
| L4 | **Interference is degree-bounded, not width-linear** (HEADLINE): S=deg·R̄_edge exactly; modular b=+0.46±0.31 vs monolith N^{1.18±0.17}; slopes diverge Welch p=3.3e-4 | CM-9 | **EV-T** | 2-dim units, MLP potential, κ=0.05, N≤16, 12 seeds, **at init**, laptop | say "degree-bounded," NEVER "flat in N"; firewall = **parameter separation** not physics (block-untied S≡0 beats the lattice); through-training only shown N=2→300ep (N∈{4,8,16} = follow-up); report basin-displacement R, never NTK cosine |
| L5 | The priced channel: sync∝κ^−1/2, n₁/₂∝κ^−1; predictive not descriptive; **N∈{2,4,8,16}, both topologies, 5 seeds, pre-registered** | CM-10 | **EV-T** (+VER at N≤8) | trained lattices N≤16; U(1)-preserving `channel_spring` | sync ≤7.5% rel-to-prediction; **n₁/₂ ranking-only** (ρ=1.0, top-rank censored) — do NOT quote pointwise n₁/₂ % (47–56%); App C "inconclusive" = random-W U(1)-breaking artifact, resolved |
| L6 | Paid-access reach/escape certificate: reach = causal box C_T (energy-blind); squeeze cures escape, provably can't beat C_T; wormhole cures reach (det J=1, ledger=0, latch transported) | CM-12 | **VER** | analytic + toy dim 2&4, 5 seeds, **oracle placement** | ORACLE channel placement; **learned entrance-steering out of scope** (the crux at scale); γ-reabsorption unmeasured; squeeze bound = matched-quadratic-H certificate (C-6 inline) |
| L7 | Calibrated escalatable compute-rationing gate; memory-agnostic stack; extra compute buys accuracy (4.8×@kv16) | CM-2 | **EV-T** | MQAR vocab-256, kv≤32, laptop, 5 seeds | one-shot memories get no allocation payoff; **CM-3 FORBIDDEN** (energy ≠ better confidence than readout margin); LTT exchangeability + ECE≈0.10 inline (C-6) |
| L8 | The one-hop non-local edge is flat in N where the N-hop chain scales; claim the **edge**, not energy-gating | CM-7 | **EV-T** | MQAR-style N≤8, 5 seeds | energy-gated wormhole **LOSES** to a 449-param learned router (FLOPs + acc); **CM-3 forbidden**; det-J=0 router destroys the Goldstone spread (measured consequence) |
| L9 | Regime map / cost story: escalatable accuracy under a rationing gate on **clean** retrieval; Hopfield keeps cost + noise-robustness | CM-8 | **EV-T** | laptop, MQAR, f32 | **THE NOISE WALL (N37, dominant negative):** no cell closes under cue noise σ≥0.3; Hopfield cheaper (~1 matvec); "6–10× savings" = intra-CLU rationing |
| L10 | Banding is a method not an oracle gift; FFT selector recovers the oracle band exactly when timescales spectrally separable | CM-11 | **EV-T** | 2-unit, 16× ratio, 5 seeds | wrong equal-spread prior is worse than none; mass-lr induce-then-snap is fragile foil (4/5) |
| L11 | Mass-hierarchy ordering is inducible (mult≈10 safe default); designed **magnitude** is not emergent | CM-5 | **EV-T** | 2–4-unit, 4–16× ratios | ≥30 degrades, 100× inverts, 4× runaway; read align/spread never MSE; **designed magnitude fits 7–14× better — hierarchy does NOT emerge (N7)** |
| **L12** | **The accept-maker: torus-coset CLU vs statistical-baseline floors on voraus-AD, identical episode-AUROC protocol** | CM-3 (bound) | **PENDING** | voraus-AD, 130ch, episode-labelled, CSF3, seeds≥3 | **NOT RUN.** Floors (partial): knn 0.77/lof 0.75/iforest 0.63/pca 0.53 on a 24ch laptop subset (`voraus-baseline-floors`); smoke: CLU 0.38–0.51 **below** baselines (`clu-anomaly-scorer`, NOT a claim). **CM-3: no superiority-by-assertion — the honest comparison IS the result.** Topology-match control (angle-permutation) pre-registered |
| L13 | Designed symmetry is the precondition (Hyp-3): emergent has no coset register; XY needs designed coupling; hierarchy designed-in | CM-16a, CM-5, (xy-1d-control) | EV-T + VER | designed SO(2) exact vs emergent MLP; N≤16 | **CM-16a is DESIGNED-ONLY** (emergent coset 1.7–4.9× softer, ~12 orders vs designed; capacity 1–1.6 bits, N46); never quote n₁/₂ without Δ + ℓ_θ/Δ |

**Ruthlessness note (task ask):** L2 is **STRUCT not EV** (no trainer, no GPU) — the "O(1)-memory training" contribution is currently a property of the autodiff graph, not a trained-system result. L6 is **oracle-placement VER** — the paid-access "win" has no learned-placement or trained-task evidence. L12 (the entire real-data spine) is **PENDING and trending negative on the smoke.** These three are the difference between a submittable systems long and a toy composition.

---

## 4. GAP LIST (prioritized; every gap → an experiment, or flagged UNASSIGNED)

| rank | gap | fills which claim | experiment / owner | status |
|---|---|---|---|---|
| **G1 (accept-maker)** | real-data CLU-vs-floor on voraus (episode-AUROC, identical protocol) | L12 | **`g7b-torus-voraus`** (engineer): literal joint-angle→so2 torus map + 2 CSF blockers (`--extra eval`; episode-AUROC not VUS-PR) | **scoped, ready, NOT launched.** Bridge exists (`clu-anomaly-scorer`). **RISK: smoke trends negative → may force Alt-C fork.** |
| **G1b** | full baseline floors, 130ch (the reference line CLU must beat/match) | L12 reference | **`voraus-baseline-floors`** — full CSF3 run (drivers + jobscripts ready; laptop de-risk done) | **PARTIAL** (24ch laptop pre-smoke only); full 130ch **pending CSF3** |
| G2 | pricing at N>2 | L5 | `v3-pricing-n-scaling` | **CLOSED** (N≤16, both topologies, 5 seeds, pre-registered). Long drafts from extended CM-10 wording; `v3-revision-5` folds it |
| G3 | firewall/composition honesty (parameter-separation-not-physics) | L4 | `v3-revision-4` device | **RESOLVED for the short** — long **inherits the fix** (own the composition in §1); not a new experiment, a **writing requirement** |
| **G4** | reversible O(1) wired into `train_chlu` + GPU/HBM memory-vs-depth | L2 (STRUCT→EV) | **UNASSIGNED** — needs "wire recompute-backwards into the trainer + accelerator memory/wall curve" | **⚠ NO EXPERIMENT ASSIGNED.** Bounds the systems claim to "structural." Long-standing ledger backlog item |
| **G5** | scale beyond N≤16 + depth (the "deep networks" in the thesis) | L4, L5 | **UNASSIGNED at scale** (CSF3); through-training interference at N∈{4,8,16} is **defined but unrun** | **⚠ largely UNASSIGNED.** "Scales to deep networks" is currently N≤16-at-init |
| **G6** | learned baselines: coRNN/LEM (stable-RNN), RevNet/Momentum-ResNet (O(1)-mem), EWC/replay (continual/firewall), a strong deep-TSAD (Transformer/TCN) on voraus | L2, L4, L12 | **PARTIAL/UNASSIGNED** — statistical floors partial (`voraus-baseline-floors`); RevNet+MomentumNet **cite-refs pinned** (`venue-follow-up`); **coRNN/LEM/EWC/deep-TSAD experiments UNASSIGNED** | **⚠ big unassigned cluster.** A systems reviewer will demand at least the O(1)-mem head-to-head (RevNet) and one deep-TSAD baseline |
| G7 | learned entrance-steering for paid access (N35, the pillar-4 crux at scale) | L6 (VER→EV) | **UNASSIGNED** | ⚠ bounds L6 to oracle-placement; explicit future work in V1, still unbuilt |
| G8 | noise-wall diagnosis (noise-aware τ / FDT sampler / denoising init) | L9 | **UNASSIGNED** (N37, the V1 narrative threat) | ⚠ unbuilt; the reversal dies under cue noise |
| G9 | designed-vs-emergent (mass hierarchy does not emerge) | L11, L13 | **not fixable — it is a finding (N7/N46)** | the honesty travels as the designed-precondition clause (Alt-A) — this is a feature of the thesis, not a gap to close |

**The gaps that decide submittability (unassigned, flagged for the Head/Hub):** **G4** (reversible-O(1) still not in the trainer → the systems headline is structural-only), **G5** (no genuine scale beyond N≤16 → "deep networks" is aspirational), **G6** (missing learned + deep-TSAD baselines → a systems reviewer's first demand). **G1 is scoped but unlaunched and trending negative.** A long that ships with G4+G5+G6 all open is a *composition-of-guarantees* paper, not a *scales-and-wins* paper — which is exactly why the recommended thesis leads with "guarantees that compose" (measured) and treats the real-data win as the fork.

---

## 5. RELATED-WORK POSITIONING (lifted from scout reports — provenance per craft rule)

Draft prose to be lifted from, cited by the report I took it from:
- **Geometry-first structured recurrence** (Di Bernardo; Keller): apply the guard-rail — do NOT headline "we choose G/H"; headline the *pricing/budget* consequences once geometry is fixed. Source: `di-bernardo-skim.md` (already in V3 short §4).
- **Goldstone / channel-allocation memory** (Mo 2026; Di Bernardo): they own the *construction* — lead with EFT-of-memory + pricing/floor, never "we invented Goldstone allocation." Source: `scout-goldstone-positioning.md`.
- **Adaptive/test-time compute** (CALM/early-exit, entropy cascades): position the calibrated gate + LTT certificates against these; energy≈margin, so claim mechanism+certificates not energy superiority. Source: `scout-adaptive-compute-prior-art.md`.
- **Reversible architectures** (RevNet, NeurIPS 2017, arXiv:1707.04585; Momentum-ResNet, ICML 2021, arXiv:2102.07870): the O(1)-memory anchors; position CLU's leapfrog reversibility as symplectic-integrator-exact at γ=0 vs their residual reversibility. Symplectic-integrator cite (Leimkuhler & Reich 2004) **verify before use**. Source: `venue-follow-up.md` (bib pinned).
- **Industrial TSAD** (voraus-AD / Brockmann et al., IEEE T-RO 2024, arXiv:2311.04765, MVT-Flow ~0.9 AUROC; SKAB; TEP): statistical floors + point-adjust-forbidden + unit-level splits (binding F2 rules). Source: `scout-industrial-datasets.md`, `voraus-baseline-floors.md`.
- **Interference / modularity** (crowded neighbourhood, all cited; κ²-firewall claim clear at specific-claim level): Source: `scout-modular-interference.md` (lift-ready ¶).

---

## 6. DUAL-SUBMISSION / POLICY NOTE

- **Facts (confirmed, `venue-follow-up`/`scout-venues-deadlines`):** NeurIPS workshop papers are **non-archival venue-wide**; **ICLR dual-submission exempts workshop-presented work**. Constraint: one short = one workshop (no multi-workshop hedging).
- **Consequence for the long:** the three shorts' material (V1 certificates, V3 firewall/pricing/reversibility, and V5-adjacent forgetting if used) **may be freely reused** in the ICLR long — **no self-plagiarism issue** because the shorts are non-archival. The long may cite its own workshop versions if they are presented, OR simply subsume them.
- **Drafting discipline (R5 anti-Frankenstein):** the long is **NOT a staple of three shorts.** It is assembled from the **shared F5 formalism** + the shorts rewritten as ICLR sections against this skeleton, from **day one**, with one composing thesis (§1). Per hermetic-citation (C-8/M1) the long's *citations* remain J&P 2026 + the F5 note only; the program's other unpublished shorts **do not exist as citations**.
- **Timing:** the shorts freeze ≤ Aug 17 (workshop deadlines); ICLR assembly Sept 1, freeze ~Aug 28; ICLR 2027 CFP still 404 ("West Coast NA"), working estimate abstract ~Sep 19–27. Do not propagate the "Brazil/Sep-19" numbers (ICLR-2026 conflation).

---

## 7. CHARTER BITE-POINTS FOR THE EVENTUAL DRAFT (where the rules will hit)
- **C-1 (no audit confession):** the legacy CHLU falsifications + the two production bugs (Lyapunov regularizer inert; shipped Langevin never sampled Gibbs) must **never** appear as a defensive paragraph. Corrected-mechanism theory lives in the F5 note as neutral theorems.
- **C-2 (verify vs evidence):** label every figure. VER = L3, L6, pricing-at-N≤8, GMOR, XY reduction. EV = L4, L5, L7–L12. STRUCT = L2. Foreground the constitutive-vs-kinematic contrast in main text.
- **C-3 (ML-first):** a physics-literate reviewer must NOT be able to summarize the long as "a damped-oscillator problem set." Lead §4/§7 with the ML result.
- **C-5:** scale qualifier in-sentence on every generalizing claim — grep for scope-free plurals ("CLUs provide…").
- **C-6:** certificate fine print next to the claim (LTT exchangeability + ECE≈0.10; Prop-12 compact-set; paid-access oracle-placement; squeeze matched-quadratic-H).
- **C-9 / C-10:** all negatives + corollaries fully written in appendices (App B–J); nothing self-pruned now.
- **CM-3 (forbidden, never even hedged):** no energy-as-better-confidence-signal claim in §6 (gate), §6 (routing), or §7 (voraus energy arm).

---

## Proposed handover updates (for the Hub)
- **The ICLR-long thesis forks on `g7b-torus-voraus`.** Recommended = "conservation-by-construction composes + holds on real data, given designed symmetry"; fallback (voraus ties/loses) = G7c "Pareto-not-podium, predictable-lifetime, designed-precondition." Section skeleton is fork-invariant; only abstract + §1 sentence + headline figure change. **This is the top planning decision; owner = Hub at the g7b review.**
- **Submittability is bounded by three UNASSIGNED gaps: G4** (reversible-O(1) not in trainer / not GPU), **G5** (no scale beyond N≤16 / no depth), **G6** (missing learned + deep-TSAD baselines: coRNN/LEM/EWC/Transformer-TSAD). Recommend the Hub open experiment tasks for at least G4 (wire recompute-backwards into `train_chlu` + one accelerator memory-vs-depth curve) and G6 (RevNet O(1)-mem head-to-head + one deep-TSAD baseline on voraus) before the long can claim more than "guarantees compose."
- **CM-10 is now N≤16** (`v3-pricing-n-scaling`, Hub-recovered) — the long drafts from the extended wording, not the V3 short's N=2 scoping. Confirm `v3-revision-5` has landed the fold before long-drafting.
- **G1b (full 130ch voraus floors)** is the reference line for every CLU-vs-floor claim; it needs the same CSF3 push as G1. Sequence them together.
- **Headline figure** = interference-scaling (fork-invariant, strongest measured systems figure) unless voraus lands a win.
- Stub written: `.claude/papers/iclr-long/outline.md`.
