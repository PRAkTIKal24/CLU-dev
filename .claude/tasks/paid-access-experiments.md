# Task: paid-access-experiments — build + test intra-unit wormholes & squeeze-as-access (w7; V1 gate for the conditional 4th pillar)

> **FINALIZED at w6 review (2026-07-07)** from `paid-access-theory.md` §7.1–7.3 + `fit-gap-anatomy.md` items 1/4. The theory is [proven; verified ×6]; this task is the discriminating end-to-end test.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/paid-access-experiments.md`
- **Read first:** protocol (§4 worktree-venv rule!) · `paid-access-theory.md` (BINDING: Def-A1…A10, Props A2/A6/A7/A9, §3.3, §4.1–4.2) · `fit-gap-anatomy.md` item 4 (IntraWormholePotential build spec) + item 1 (reach increment context) · claims matrix CM-12 + CM-7 (the router lesson: always a no-physics control) · brainstorm Thread 8.
- **Git:** branch `agent/experiment-engineer/paid-access-experiments`; worktree if concurrent (`fix-pack-4` runs this wave — coordinate: it touches config/train; you touch core/potentials + a new experiment).

## Build (from fit-gap-anatomy item 4 — the theorist-endorsed construction)
1. **`IntraWormholePotential(eqx.Module)`** in `chlu/core/potentials.py` — V-wrapper (TiltedPotential precedent), sparse block-pair channels, **hard gate frozen at capture** (paid-access-theory Prop-A6 design rule: a state-dependent gate on the jump breaks volume by exactly ∇g·Δ — forbidden). Constant-translation channel variant `q→q+Δ_k, p→p` with per-jump energy ledger ΔH=V(q+Δ)−V(q). NO edits to integrators/chlu_unit.step/lattice.
2. Tests (≥4): det J = 1 to ~1e-12 through a jump; bit-exact closed-gate reduction to plain CHLU; bounded-energy/ledger correctness; latch-transit shift = pᵀXΔ (theory Prop-A7).
3. **Mass banding is a PREREQUISITE** (theory §3.3 reason 2 + open-risk 4): all squeeze arms run with a **designed band** (e.g. [4, 0.25] via `banded_mass_scales` or the FFT selector) — uniform-M squeeze arms would repeat the l0-gate ambiguity. State the band in every provenance table.

## The battery (theory §7.1–7.3, verbatim specs)
4. **§7.1 multi-basin REACH task:** relativistic CLU, d 2–4, K-basin potential with basin distances d_k spanning below AND above the causal box L=T·ε·c/√M; governor fixed so plain relaxation-in-T provably fails (report T_min per basin). **Arms:** plain relax · S^(M) squeeze (line-searched ζ) · intra-unit wormhole (matched channel per target) · **Newtonian-squeeze control** (energy DOES buy reach — validates the cap is the constraint) · **no-physics router/controller baseline** (CM-7 lesson, mandatory). **Sharp predictions to test:** squeeze landing ≈1 for d<L above ζ* (threshold inside the bracket [ζ_exact, ζ_kinetic]) and **drops to ≈0 for d>L**; wormhole flat ≈1 at all d; the d=L crossover is the falsifiable heart. ≥5 seeds.
5. **§7.2 latch-transit test:** SO(2) sector with latched charge Q; wormhole Δ tangent vs across the coset → Q′−Q = pᵀXΔ to ≤1e-6 of charge scale (≈0 tangent); squeeze preserves Q; random-shift baseline erases it unpredictably.
6. **§7.3 certificate verification on every arm:** per-jump det J (1±1e-12 / (1−γ)^d damped), ledger vs V(b)−V(a), injection ≤ e^{2ζ}H, governor re-absorption vs 2ζ/γ_c.
7. **The dense-MLP discriminator** (fit-gap-anatomy item-4 risk): the gated-sparse intra-unit channel must beat a **dense-MLP-potential** arm on the reach task (else it is a dominated reparameterization — report it as such per C-9).

## Gate criteria (Head-approved framework, Thread 8)
V1 adopts as pillar 4 iff: (a) reach crossover at d=L matches theory (squeeze step + drop; wormhole flat); (b) certificates hold as measured; (c) latch transported per prediction; (d) beats the no-physics controller AND the dense-MLP arm. Any miss → C-9 negatives fully written, mechanisms stay V3/future-work, V1 ships 3 pillars. **Known open risk to report on either way:** learned entrance-steering (theory open-risk 1) — for this battery, channel entrances may be placed by construction (oracle placement); LEARNED placement is explicitly out of scope (state it).

Flag-provenance per §5 (band, c, T, γ, ζ grid, basin geometry per run).
