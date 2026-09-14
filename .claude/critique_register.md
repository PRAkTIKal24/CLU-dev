# Critique Register — Reviewer-Hat Audit (2026-07-07)

> **Provenance:** authored by the previous Hub (reviewer-hat pass, composite ML4PS/NeurReps/ICLR reviewer), adopted by the Head as binding program input 2026-07-07. **Nothing here is parked.** Every item gets an execution/validation/writing plan; skip decisions (if time-pressed) are made later, explicitly, by the Head.
>
> **Status legend:** OPEN · TASKED(<slug>) · ANSWERED(evidence) · WRITING-RULE(where encoded). Update statuses at every wave review.

## Response-type key
- **EXP** — requires a new experiment/ablation (we have the apparatus).
- **DEC** — requires a Head/Hub decision before drafting.
- **WRI** — writing/positioning discipline; encode in the positioning charter + ledger, enforce at draft review.

---

## Priority order (execution sequence, nothing parked)

### Tier 0 — decisions & architecture BEFORE any drafting (this week; zero compute)

| P | ID | Type | Item | Plan |
|---|----|------|------|------|
| P1 | **M1** | DEC | Cross-citation trap: shorts can't cite each other anonymously | Settle citation architecture NOW: each short hermetically self-contained; may cite only (a) the published CHLU paper, (b) a citable F5-style preprint IF arXiv'd in time. **Head decision needed: arXiv F5 (or a distilled "budget-table" note) by early Aug, yes/no.** Check every task file & skeleton against this before drafting. |
| P2 | **M3** | DEC | Salami-slicing + August bandwidth: 3–4 shorts may underperform 2 polished ones | Keep portfolio count OPEN until Tier-1 results land (they decide whether V1 stands alone or merges into V2/V3). Explicit Head decision at wave-5 review, not a default. |
| P3 | **M4** | WRI | Flag-provenance table or bust | Mandate: every result in every short carries a training-config table (flags: lyapunov mode, langevin_noise, anchor, epochs, sleep_frequency, seeds). Template added to AGENT_PROTOCOL reporting rules; papers inherit from the ledger. |
| P4 | **M2** | WRI/DEC | Effective de-anonymization; a reviewer will review the *program* | Build a cross-short claims-consistency matrix (one page, Hub-maintained) — no contradictions between shorts; consistent scope qualifiers. Interacts with P1 (F5 preprint further de-anonymizes — same decision). |

### Tier 1 — cheapest load-bearing experiments (wave-5, launch immediately; all laptop-scale)

| P | ID | Type | Item | Plan |
|---|----|------|------|------|
| P5 | **G4** | EXP | Mass-specific lr never tried before "hierarchy must be designed-in" doctrine shipped | **Cheapest, highest-leverage.** Engineer+analyst: mass-lr (two-timescale, like γ_φ fix) × longer horizon × timescale-curriculum on the banding testbed. Outcome either retires Hyp-3 doctrine (unlocks "mass-narrowness pivot" three ways) or fortifies it with the missing control. Couples to V3.2. |
| P6 | **G2** | EXP | No "CLU minus the physics" control anywhere | Non-symplectic twin: identical dims/params, unconstrained update (and a leapfrog-with-broken-volume variant as second arm), run through the SAME measurement harness (retention, latch, budget-table observables, erosion). Decides what symplecticity *functionally* buys. Cross-cutting: feeds all three shorts. |
| P7 | **V1.1** | EXP | Gate stack is memory-agnostic; energy ≈ readout margin | Run the identical calibration/allocation/LTT stack on Hopfield (and one autoencoder-style scalar). If it allocates equally well → V1 reframes as "gate mechanism, physics provides certificates"; if not → CLU-specificity measured. Same family as P6 — one task file, shared protocol. |
| P8 | **V2.3** | EXP | No learned-architecture baseline in V2 | coRNN + LEM + LSTM on retention-vs-perturbation + Mo-S¹ task (already pre-freeze list — confirmed BLOCKING, not optional). Folds into the pre-freeze V2 analyst task with S1 adaptive-K re-run. |

### Tier 2 — vertical fortification (wave-5/6, before freeze)

| P | ID | Type | Item | Plan |
|---|----|------|------|------|
| P9 | **V1.2** | EXP | Wormhole lacks the boring baseline | Add learned-router-MLP arm (2-layer on query embedding, same sparsity, no physics); ≥5 seeds; non-50/50 workloads; cost in FLOPs not unit-steps. |
| P10 | **V3.2** | EXP | Banding confound: bands were matched to data | Band-selection story: mis-banded control (graceful degradation?) + a selection heuristic (spectral/curriculum). CONTINGENT on P5: if mass-lr learns hierarchy, story becomes "learnable with the right optimizer"; if not, "prior with payoff + selection recipe". |
| P11 | **V2.4** | EXP | Anchor cure validated at one setting, 2 seeds | Anchor robustness sweep (λ × seeds × epochs × datasets incl. Exp-B immune control); present demarcation law + cure as THEORY (wake-invisible flat directions), never as patch note. |
| P12 | **V3.1** | EXP | Interference NTK unmeasured — V3's own named firewall | Measure Θ(q,q′) across units at N=4/8, banded vs uniform, during training; the biggest V3 credibility gap (ledger agrees). |
| P13 | **V3.3** | EXP | Pricing measured on designed potentials only | One trained multi-unit task where the κ-law *predicts task behavior* (e.g., predict which lattice fails a recall-horizon from measured κ_eff before running it). |
| P14 | **V1.3** | EXP | Regime map reads as self-refutation; "fixable" asserted not shown | Memory-fidelity workstream: at least one training-quality intervention (anchor cure from V2.4? longer/persistent training? capacity) that measurably moves a regime-map cell. Write-up as boundary-mapping regardless of outcome. |

### Tier 3 — writing/positioning discipline (drafting-time; encode NOW in the positioning charter)

| P | ID | Type | Rule |
|---|----|------|------|
| P15 | **G6** | WRI | ~~Own the falsifications first~~ **REVERSED (Head, 2026-07-07):** no audit confession in any paper (legacy paper = territory-claiming poster; nobody cross-validates it). G6's risk is instead closed by: F5 note carries the corrected mechanisms as neutral class theorems; shorts describe current fixed mechanisms with exact flag-provenance; no legacy-number is ever load-bearing. See charter C-1 (rewritten). |
| P16 | **G1** | WRI | Foreground the constitutive-vs-kinematic defense + the learned-anharmonic 2–15%-with-predicted-deviations results in the MAIN text; designed-testbed results labeled as verification, learned-system results as evidence. |
| P17 | **V2.2** | WRI | V2 reads ML-first: "what memory in a trained network obeys," physics as derivation apparatus. The Mo head-to-head ("a published ML result is our overdamped face") leads. |
| P18 | **V2.1** | WRI | Lead with measured (constitutive law, Mo separation, latch); EFT-of-memory demoted to horizon/future-work until results exist. |
| P19 | **G3** | WRI | Scale qualifiers attached to EVERY generalizing claim (dim, N, vocab, seeds); no bare "CLUs provide…". Draft-review checklist item. |
| P20 | **G5** | WRI | Certificate language altitude: LTT exchangeability caveat + ECE≈0.10 + Prop-12 compact-set scope in main text next to the claim, not appendix; "cannot destabilize" → "certified within [scope]". |

---

## Status block — 2026-07-07 (wave-5 launch)
- **P1 (M1):** DECIDED — Head approved the F5 arXiv note, WITH anonymization-minimizing constraints (no CLU coinage; neutral title/vocab; third-person self-citation; author list = Head's call). → TASKED(`f5-arxiv-note`).
- **P2 (M3):** CONFIRMED on the wave-5 review agenda — explicit portfolio-count decision there.
- **P3 (M4):** ENCODED — flag-provenance rule now mandatory in `AGENT_PROTOCOL.md` §5.
- **P4 (M2):** Hub builds the claims-consistency matrix at wave-5 review (when Tier-1 results fix what the claims are).
- **P5 (G4):** TASKED(`mass-lr-doctrine-test`, engineer).
- **P6 (G2) + P7 (V1.1):** TASKED(`minus-the-physics`, engineer — shared protocol as planned).
- **P8 (V2.3):** TASKED(`v2-prefreeze-baselines`, analyst; includes Mo-S¹ + S1 extras).
- **Erosion-novelty confirm (supports P11/V2.4 claims):** folded into `venue-follow-up` sub-task 5 (fires ≥Jul 11).
- **P9–P14 (Tier 2):** OPEN — wave-6 tasking after wave-5 review (P10 contingent on P5's outcome; P14 candidate = P11's anchor cure).
- **P15–P20 (Tier 3):** ENCODED — Positioning Charter section appended to the ledger (`philosophy-synthesis.md`), binding for all drafting.

## Status block — 2026-07-07 (wave-5 REVIEW: Tier 1 fully answered)
- **P5 (G4): ANSWERED — doctrine SHARPENED.** Mass ordering IS inducible (mass-lr 10× + 1500 ep: Spearman +0.89 5/5 seeds, spread 6.5× ceiling; MSE 1.85→0.64); designed magnitude is NOT reached (designed fits 7–14× better); 100× lr inverts ordering (runaway); curriculum didn't help. The ~0.08 ceiling = 300-ep budget artifact. Control exists; Hyp-3 holds in sharpened form (see claims matrix CM-5). Evidence: `mass-lr-doctrine-test.md`.
- **P6 (G2): ANSWERED.** Volume conservation buys BIBO (1.0 vs 0.33) + protected flat direction (μ² 0.008 vs 0.122) + CD-robust vacuum (survives vs collapses); leapfrog buys the latch (0.19 vs 1.15 coset drift); physics costs raw fit (0.19 vs 0.013 MSE). CM-1. Evidence: `minus-the-physics.md` Part A.
- **P7 (V1.1): ANSWERED — gate stack IS memory-agnostic** (Hopfield 0.18→0.88 ≈ CLU 0.43→0.87; LTT valid on both). Non-transferable asset = escalatable memory (allocation payoff needs full>base). V1 claims fixed per CM-2/CM-3. Evidence: `minus-the-physics.md` Part B.
- **P8 (V2.3): ANSWERED — CLEARED, V2 strengthened.** Triad qualitatively absent in coRNN/LEM/LSTM (5.6/56/69 steps, fragile 69→2); CLU-emergent 263 bounded + law-predicted; designed ∞ latch. CM-4. Evidence: `v2-prefreeze-baselines.md` (+ item-4 resolves the +12–28% bias decomposition; item-3 validates adaptive-K + compact gates).
- **P4 (M2): DONE — claims-consistency matrix v1.0 at `.claude/claims_matrix.md`** (canonical constants + CM-1…CM-8 approved wordings).
- **P1 (M1):** F5-note draft DELIVERED (all 14 checks re-run green; anonymization audit clean). Awaiting Head: title pick (3 proposed), Cor-3 footnote keep/cut, anharmonic-% policy, author list. Then paper-writer LaTeX conversion.
- **P2 (M3):** decision NOW DUE — Hub recommendation in the wave-5 review brief.
- **Tier 2 (P9–P14): wave-6 candidates** — P9 (router baseline; CM-7 blocked on it), P10 (band-selection, REFRAMED per CM-5: "optimizer reaches ordering, not magnitude"), P11+P14 (anchor robustness = the memory-fidelity intervention candidate), P12 (interference NTK — V3 gate), P13 (pricing→task).

## Status block — 2026-07-07 (Head decisions + wave-6 launch)
- **P2 (M3): DECIDED** — V2 GO (drafting now), V1 GO reframed, V3 conditional on wave-6 (P12+P10), erosion → V2 appendix pending Jul-11 confirm; fallback to 2 shorts pre-agreed.
- **New Head policies encoded:** charter **C-9** (negatives documented; prominent ones → paper appendices; registry = `.claude/negative_results.md`) and **C-10** (appendix maximalism until pruning passes; placeholder titles/authors; reviewer agents critique drafts → iterate editorially or with missing experiments).
- **F5-note editorial:** placeholder title (workshop at end); ALL cors kept, corollary-grade → appendix; nothing omitted pre-pruning; author placeholders. Supersedes the earlier "cut Cor-3 footnote" recommendation.
- **Wave-6 TASKED:** P9 → `v1-router-baseline` · P10 → `v3-band-selection` · P11+P14 → `anchor-robustness` · P12+P13 → `v3-interference-ntk` · NEW (Head, Thread 8) → `fit-gap-anatomy` (CM-1 scope-filler: contraction-vs-reach decomposition, horizon crossing, paid-mechanism recovery ladder) · drafting → `v2-short-draft` (paper-writer) · docs → `transfer-docs-sync` (doc-curator). Three new agent defs created (paper-writer, paper-referee, doc-curator).

## Status block — 2026-07-07 (wave-6 REVIEW: Tier 2 fully answered — REGISTER COMPLETE except writing-rule enforcement)
- **P9 (V1.2): ANSWERED — the router WINS** (1.000/0.948 @8.81e7 FLOP vs gated 0.887/0.715 @1.18e8; 5 seeds, all mixes). CM-7 REWRITTEN; salvage = the 1-hop-edge-vs-chain mechanism claim; impostor-composition cause isolated (probe-mix mismatch, archive-only cuts local FP 53%→7%). Third instance of the energy≈learned-signal pattern.
- **P10 (V3.2): ANSWERED end-to-end** — confound killed (matched 1.18 < uniform 2.42 < orthogonal 6.92 < anti 12.79, 5/5); FFT selector = oracle (gap 0.000, 5/5, separable-timescales qualifier); mult≈10 generalizes, ≥30 fails ratio-dependently. CM-11 added.
- **P11 (V2.4): ANSWERED** — anchor holds 3000 ep, λ∈{1,10,100}; envelope = strength↔robustness tradeoff (λ=10 best rejection, 1/5 collapse; λ=100 bulletproof 5/5 @35× wake cost); demarcation = THEORY (tilt δ≥0.05 immunizes with NO anchor; μ² witness). Cross-link: anchor does NOT rescue broken-volume — CD-robustness is a volume-conservation payoff, anchor orthogonal (CM-1/CM-6 sharpened).
- **P12 (V3.1): ANSWERED — POSITIVE.** Firewall measured: modular 2e-5 vs monolith 0.20 (1:9,000), ∝κ², R_far≡0, mass-independent, O(1)-in-N vs monolith O(N) exceeding self-signal at N=8. CM-9 added. V3-gate evidence in favor.
- **P13 (V3.3): ANSWERED — POSITIVE.** κ_eff blind from learned curvature → recall-horizon ranking Spearman 1.0, sync ≤8%, registered-before-measured. CM-10 added.
- **P14 (V1.3): ANSWERED — MAJOR REVERSAL PENDING CONFIRM.** The 26/26 Hopfield map was under-trained: 2000 ep closes 3/3 sampled losing cells (fid→1.00, 9–10× savings, kv64 included). CM-8 frozen PROVISIONAL; full-grid ≥5-seed re-map = w7 `regime-remap-2000ep` before any claim ships. Anchor ≠ memory-fidelity tool (init-pin, wrong target).
- **Thread-8 (Head): fit-gap-anatomy + paid-access-theory both delivered.** Loan called ≈700 steps (CM-1 filled); +γ recovers 92% at zero structural cost; γ_φ wrong tool (−24%); reach real-but-secondary at single-unit scale; reach/escape dichotomy proven (squeeze=escape-only, wormhole=reach, latch transport pᵀXΔ exact); l0-gate-null-doesn't-test-access argued; w7 testbed specs sharp; **mass-banding = prerequisite for the directional squeeze claim**. CM-12 added.
- **Tier 3 (P15–P20): encoded + first enforcement PASSED** — v2-short draft is charter/matrix-compliant by construction (evidence map audited); paper-referee pass = w7.

## Status block — 2026-07-07 (wave-7 REVIEW: register fully discharged; V1 identity now decided-by-evidence)
- **Pillar-4 gate (Thread-8/CM-12): PASSED.** paid-access-experiments met all four criteria (reach crossover, certificates, latch transport, beats router + dense-V). The intra-unit wormhole is **V1's one physics-specific, certificate-backed win** — the answer to the V1.1 "calibration paper in a physics costume" attack. Scope caveats logged (oracle placement; γ-reabsorption unmeasured).
- **CM-8 (P14): cost story FINAL, accuracy story PARTIAL.** regime-remap Item 3 (complete) corrected the anchor-robustness over-claim: Hopfield stays cheaper (1 matvec); the 9–10× is intra-CLU. Accuracy-reversal needs the full grid (w8 `regime-remap-complete`).
- **v2-referee: 5 MUST-FIX / 6 SHOULD-FIX; borderline→weak-accept after fixes.** MF-3 = a genuine internal number contradiction (§3.4 vs App D) — the exact M4/cross-section-drift class this register exists to prevent, caught pre-submission. MF-1 (audit-¶ removal) = the C-1 reversal enforced. **MF-5 Cor-3 footnote = escalated DEC** (C-10 keep-all-cors vs C-1/M2 anonymity collide — Head-only).
- **CM-13 added (reversible O(1) measured).** Backlog #1 fully discharged.
- **Register state:** ALL P1–P20 answered or encoded; all CM rows have measured backing or a stated pending-experiment. The critique program is complete; remaining work is drafting-discipline enforcement (referee passes) + the two pending grids/decisions above.

## G7 — the "stickman" critique (Head, 2026-07-09) — a LONGS mandate, not a shorts fix

> *"We only show things on SO(2) for a simplistic ring case with a simplistic task of closed-circle tracing — the prime critique would be that we use very stickman-like experiments."* Sharper and more dangerous than G1 (which is about the *testbed satisfying the theory*); G7 is about the **narrowness of the symmetry AND the triviality of the task**, and it will be the first thing an ICLR/NMI reviewer says.

| # | item | plan | status |
|---|---|---|---|
| G7a | **Only SO(2)** (abelian, one Goldstone) | Two ready targets, both with predictions already derived: **torus T²** (tests "abelian coset ⇒ exactly independent registers"; nearly free) and **SO(3)→SO(2)** (curved S²: O(α⁵) order-insensitivity, α² holonomy, **custodial n₁/n₂ = m₁/m₂**). → V4 / longs | `OPEN` |
| G7b | **Only closed-circle tracing** | **THE UNLOCK: robot joint angles live on a torus.** `voraus-AD` (loader-ready, F2 harness) has joint space T^n = U(1)^n — exactly the abelian coset the theory predicts gives n independent dissipation-proof registers. Matching the CLU's coset to the *data's own topology* makes the architecture physically motivated, and kills **beyond-SO(2) + beyond-toy-task + real-data** with one CSF3 push | `OPEN` — highest-leverage item for both flagships |
| G7c | **The claim shape the longs need** | *"Primitives P are bad at characteristic **X**; at matched params the CLU exhibits **X** with guarantees **Y**, though worse at raw next-step fit."* Instantiate: **X = a memory whose lifetime is predictable and controllable from the model's own parameters** (read retention off the Hessian: n₁/₂∝μ⁻², floor, γ_crit=2εμ, D_θ∝T/(γF²)) — impossible for LSTM/LEM/Transformer. Foil measured (CM-4): baselines forget in 57–69 steps, collapse to 2 under perturbation; designed CLU latches forever. Honest tail (N40): 6–23× more compute per unit retention ⇒ **Pareto-not-podium with a real X.** Secondary X's: certified test-time compute (V1); degree-bounded interference firewall (V3); **thermal stability / memory phase diagram** (Thread 10, if it lands) | `OPEN` |

**Binding decision (Head, 2026-07-09): do NOT expand the shorts.** They are correctly scoped, carry C-2/C-5 discipline, and present designed-vs-emergent *as a finding*. Widening now risks the freeze. **G7 is discharged in the ICLR + Nature-MI longs**, where it is the organizing mandate.

## Interaction map (read before tasking)
- **P5(G4) ↔ P10(V3.2) ↔ ledger's "mass-narrowness pivot":** one experiment, three consequences.
- **P6(G2) + P7(V1.1):** same "minus-the-physics" family → one shared protocol/task pair.
- **P1(M1) + P4(M2):** the F5-preprint decision controls both.
- **P2(M3) waits on P5–P8** outcomes (V1 identity, V2 baseline strength).
- **P11(V2.4) ↔ P14(V1.3):** the anchor cure is the candidate memory-fidelity intervention.
