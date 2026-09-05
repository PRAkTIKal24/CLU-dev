# v5-short-draft — paper-writer report
Task + acceptance criterion: first draft of the V5 "Forgetting" short (draft.md + draft.tex + PDF), thesis led by CM-16b, CM-16a scoped designed-only with N46 owned, vault at 107×, every caption carrying the fdt+Newtonian flag and Δ/ℓ_θ, erosion novelty per ship rules, matrix-consistent with V2, F5 cited as Anonymous.
Status: **done** (draft builds as md; tex delivered as source — no TeX toolchain on machine, see below). One downstream reconciliation item + editorial questions for the Hub/Head at the end.

## Downstream reconciliation list (owner needed — flagged in first 10 lines per protocol)
- **None that change existing docs.** This is a fresh draft; it inherits provenance tables A.7/A.8 from the V2 draft verbatim (renumbered A.2/A.3 here) and introduces no new numbers. The only cross-paper item the Hub should track: **V2 App-J and V5 present the same t-lever/v5-gate material** (V2 as appendix, V5 as lead) — kept numerically identical on purpose; if either is edited, the other must move in lockstep (see "V2↔V5 consistency" below).

## What I did
- Read (in order): AGENT_PROTOCOL, task file, claims_matrix (CM-16a/b, CM-17, CM-3, canonical constants §1), Positioning Charter C-1…C-10 (C-1 REVERSED for this paper), v5-gate.md, t-lever-forgetting.md, venue-follow-up §3, sleep-erosion-study §4, and the V2 draft (sibling) for vocabulary/consistency.
- Wrote `.claude/papers/v5-short/{draft.md, draft.tex, CHANGELOG.md}` + copied 7 figures into `figs/`.
- draft.md is canonical; draft.tex is a faithful LaTeX port on a generic `article` template (swap for the venue style at submission).

## Structure (appendix-maximalism, C-10)
Main text ≤4pp: Abstract · §1 Intro + 4 enumerated contributions (page 1) + CLU continuity sentence + M/μ nomenclature + the Δ/ℓ_θ measurement discipline · §2 Setup (map, budget, coset-diffusion law, trained models) · §3.1 **the (μ,γ,T) budget cube / V-curve (headline, CM-16b, generalizes emergent 3/3)** · §3.2 friction-preserves-temperature-erases (CM-16a designed latch + coset-diffusion law + sign flip) · §3.3 the 107× refrigerator vault · §3.4 the designed-symmetry precondition (N46, owned) + the instrument warning · §4 Related work · §5 Discussion/limits/horizon.
Appendices (fully written, not pruned): A flag-provenance (4 groups) · B coset-diffusion law + sign flip + latch + V-curve · C emergent arm in full (N46, T*, instrument warning, two-observables) · D vault in full (absorb-only mechanism, discriminator, write attenuation) · E **erosion-as-symmetry-restoration** (per venue-follow-up §3 ship rules) · F Coleman/MW + mandatory fdt+Newtonian scope · G T_φ shredder (future work) · H prominent negatives.

## Evidence backing each main-text section (all traced)
- **§3.1 budget cube / V-curve** — `t-lever-forgetting` §4.3 (designed massive-mode V-curve, min at γ_crit=2εμ, slope −1.006/+1.23–1.27, 5 seeds) + `v5-gate` §3.4 (emergent V-curve, argmin 0.902±0.003×γ_crit, slope −1.0020/+1.116, 3/3). Designed=verification, emergent=evidence. Headline fig = fig1_vcurve (from v5-gate fig2_emergent_vcurve).
- **§3.2 sign flip / diffusion law** — `t-lever` §2 (latch ||λ|−1|≤1.7e-14, drift ≤4.9e-12/200k), §3 (D_θ=εT(2−γ)/(2F²γ) to 1.0068±0.0219, 25 cells), §4 (∂n₁/₂/∂γ>0 10/10, +0.955±0.042, 3.77±0.23×). Designed=verification.
- **§3.3 vault** — `v5-gate` R3 (pre-registered): refrigerator Var(p)/(MT)=0.12600±0.00031, D̂-vault 107.77±4.78×, scalar control 13.28±0.12×, field/scalar 8.11±0.37, T=0 erasure ≤2.0e-15. Designed+analytic field=verification. Fig2_vault (from v5-gate fig1).
- **§3.4 precondition / N46** — `v5-gate` R1: emergent 1−|λ_coset|≈1e-3 (12 orders vs designed), δ relaxes ≤2.1e-3, capacity ≈1–1.6 bits; T*≈3e-3; the raw-exponent instrument warning (matched designed control gives same shallow slopes). Emergent=evidence.
- **§4 related work** — positioning lifted from: `venue-follow-up` §4b (Minami–Hidaka dissipative-Goldstone propagating→diffusive), `mo-deep-read`/v2 related-work (Mo 2026 kinematics-vs-constitutive), `di-bernardo-skim`/venue-follow-up §4c (Di Bernardo 2025), venue-follow-up §3 (Fischer-Igel, Nijkamp, Du-Mordatch, RBM 2503.21536 for erosion).

## Binding-rule compliance (self-checked)
- **CM-16b leads** (§3.1 headline); **CM-16a scoped designed-only** everywhere it appears (§3.2, §3.4); **N46 owned** in main text §3.4 + Appendix C.2/H. **CM-16 split honored** — "CM-16" never cited as one claim.
- **Vault = 107.77±4.78×** (D̂, the number to quote); raw FPT 86.97±2.94× given as the honest figure-axis number in §3.3 + App D with the ℓ_θ/Δ reason. **13.9×/13.88× correctly recast** as the retracted coupled-bath prediction = the measured value of the scalar-γ control (13.28±0.12×); abstract says "not the naive ≈14×".
- **Mandatory flag** `langevin_noise="fdt"` AND Newtonian kinetic mode in every quantitative caption (Fig 1, Fig 2), §2, §3.2 fine print, App A (every group), App F. **CM-17** honored: relativistic Gibbs no-go stated as scope caveat only (App F, Prop-9′), no novelty claimed (matches CM-17 NOVELTY SCOPE).
- **Every n₁/₂ carries Δ and ℓ_θ/Δ** where it is a T>0 quantity; T=0 Jacobian half-lives (operator gaps) explicitly noted as not needing Δ. The instrument warning (raw exponents non-discriminating) is main-text §3.4 + App C.5.
- **Erosion novelty (venue-follow-up §3)** — App E ships as instance + demarcation + frequency-law + anchor cure; (b)/(c) marked NOVEL single-sourced, (a)/(d) cite-the-substrate (Fischer-Igel, Nijkamp, RBM 2503.21536; Du-Mordatch). **"We do not claim to have discovered CD is biased"** stated twice; `sleep-erosion-study §4` honest framing preserved verbatim ("sharp, quantified instance + demarcation + cheap cure, not the discovery that CD is biased").
- **CM-3** — no energy-as-signal claim anywhere (grep clean).
- **C-8 hermetic** — F5 note cited once as "(Anonymous, 2026)" then "the theory note", third person; **two initial cross-short references removed** (I had written "companion memory results" pointing at V2 in §3.1 and §3.4 — both rewritten to self-contained statements). Only external published work + J&P 2026 + Anonymous 2026 cited.
- **C-1 REVERSED** — no audit-confession paragraph anywhere.
- **C-2** — designed testbeds labeled verification, emergent/learned labeled evidence, in-caption and in-section.
- **C-5** — scale qualifiers in-sentence (dim 4, hidden 64, laptop-CPU, designed 5 seeds / emergent 3 seeds) on every generalizing claim.
- **Naming** — CLU continuity sentence present verbatim ("the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"); inertial M vs spectral μ nomenclature block per HEP_primers ledger.
- **Placeholders** — `[WORKING TITLE: …]` and `[AUTHORS PLACEHOLDER]`.

## V2↔V5 consistency (task asked to report any contradiction risk)
Checked V5 against the V2 draft. **No numeric or scope contradiction found.** Shared objects are identical: n₁/₂∝μ⁻², floor 27.03, γ_crit=2εμ, EP h*≈γ/2, D_θ=εT(2−γ)/(2F²γ) with F²=M_ch r*², the 3.77±0.23× friction-lengthens result, N46, T*≈3e-3, the fdt+Newtonian mandatory flag, and the verification/evidence labels. V2 carries this material as **Appendix J** ("the T=0 face of the budget cube"); V5 makes it the lead. **Two things to watch:** (1) the **107× vault is new to V5** (not in V2 App-J), so a reviewer reading both finds no conflict, only V5 going further — fine, but if V2 is ever widened to mention the vault it must use 107.77±4.78× (D̂), never 13.9×. (2) V2 App-J's title phrase "the T=0 face of the budget cube" and V5's framing of the *same* cube as three-parameter (μ,γ,T) are compatible (V2 means the T=0 latch is one corner), but the Hub may want the two titles harmonized at the pruning pass so the shared appendix reads identically.

## Build status
No `pdflatex`/`xelatex`/`latexmk` on the machine — **draft.tex delivered as source, NOT built** (not pseudo-verified, per craft rule). draft.md is the canonical, fully-readable artifact. Figures copied and referenced by relative path in both. Swap `\documentclass` for the venue template (ICLR/ML4PS/EBM-workshop) at submission.

## Open editorial questions for the Hub / Head
1. **Venue not yet fixed** (venue-follow-up: EBM/generative-if-accepted else ML4PS-appendix-grade → own venue; list not posted as of Jul 19). The tex template is generic pending that call. Which style file?
2. **Headline figure**: I made the **V-curve (fig1_vcurve)** the headline per the task's "the V-curve across μ² decades is the natural headline," with the vault as Fig 2. v5-gate §4.1 instead nominated the vault panel as "the one figure the short is built around." Confirm the V-curve-as-headline choice, or swap.
3. **Title workshop** — current working title leads with "budget cube." Alternatives to weigh at the title pass: "Friction Preserves, Temperature Erases" (the quotable headline) vs the cube framing (the portable result). Deferred per C-10.
4. **Erosion placement** — shipped as Appendix E here (the material is a training-dynamics result, sibling-consistent with V2 §3.5). venue-follow-up flags the (b)/(c) novelty as single-sourced; a pre-camera-ready scout confirm is still open. Keep in V5, or is it double-counted against V2 §3.5? (Both are hermetic, so a reviewer of only one sees no issue; a reviewer of both sees the same honest instance told from two angles — acceptable, but Head may want it in exactly one.)
5. **T_φ shredder (App G)** is future-work-only (no promissory in main text, per C-4). If v5-gate's R4 (`T_φ` build) lands before freeze, §3 gains the 2×2 {γ_φ,T_φ}×{T=0,T>0} figure and the "shredder" becomes a result — flag whether to hold a slot.

## Git footprint
None (all artifacts under `.claude/`, gitignored). No tracked files touched.
