# Task: mass-visible-objective — make the loss able to see the mass spectrum (w20)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/mass-visible-objective.md` · **Branch:** `agent/experiment-engineer/mass-visible-objective`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/clu-latent-io-audit.md` (your own w19 audit — this fixes what it found) · `.claude/outputs/clu-memory-architecture.md` (theorist Prop 6)
- **⚠ DEPENDENCY: run AFTER `dt-units-split`.** The `E_reg` dominance below is measured at the conflated `dt=0.05` and may change once units are fixed. Re-measure the baseline before changing the objective. If `dt-units-split` has not landed, say so and measure both.

## Why
The vision makes **masses the access keys** — inference selects a mass to retrieve an item. Theorist **Prop 6** (verified to 2.2e-14) supports this: with per-item masses only the overall *scale* is gauge, and **all ratios become wake-visible and exactly learnable**. But w19 measured that the current objective **cannot see the spectrum**:

1. ⭐ **The EBM contrastive term has EXACTLY ZERO mass gradient.** Negatives perturb `q` only, so kinetic terms cancel identically in `H(data) − H(neg)`. **The objective's representational half is structurally blind to the mass.**
2. **99.8% of the mass gradient comes from `E_reg`**, whose only lever is inflating `M` *uniformly* — pure common-mode pressure.
3. **Result:** `log_mass` moves (+3.56) but as common mode (differential:common = **1:39**), and `M_max/M_min` ends at **1.153, BELOW its random init of 1.265** — **training makes the spectrum more uniform.**
4. ⚠ **Softplus trap:** `M = softplus(log_mass)` is linear for `x≫0`, so once `E_reg` drives `log_mass`→3.5, a *log*-scale spread stops buying *exponential* range (Std(log M)=0.56 → ratio only 1.28).

**If masses are the access keys and the loss cannot see the access structure, the architecture is untrainable by construction.** This is a missing term, not a tuning problem.

## Item 1 — reproduce the diagnosis
Confirm (1)–(4) post-`dt` fix. Report the differential:common ratio, final `M_max/M_min` vs init, the `E_reg` share of the mass gradient, and — analytically *and* numerically — that the contrastive term's mass gradient is identically zero. **If the `dt` fix already changes any of these, that is the finding.**

## Item 2 — give the objective a mass-visible term
Implement and compare candidates that produce a **non-common-mode** mass gradient. At minimum:
- **(a) mass-perturbed negatives** — extend the EBM contrastive term so negatives perturb `m` as well as `q`, breaking the kinetic cancellation. *(Hub's preferred candidate: it fixes the defect at its source rather than adding a new regularizer.)*
- **(b) zero-mean constraint** — normalize `log_mass` to zero mean so `E_reg` can no longer express itself as common mode, forcing its pressure into the differential direction.
- **(c) reparameterization** out of the softplus trap so log-scale spread buys exponential range.

Report for each: does the differential:common ratio move off 1:39? Does `M_max/M_min` exceed its init? **Ablate — a term that only raises the ratio by co-moving `E_reg` has not fixed anything.**

## Item 3 — is the resulting spectrum *functional*?
A spread spectrum is not automatically a useful one. Measure whether the learned spectrum produces an actual **timescale hierarchy**: do different-mass particles resolve information at different rates? Report a timescale-vs-mass curve. ⚠ **This is the real acceptance criterion** — `M_max/M_min` going up is necessary, not sufficient.

## Item 4 — per-address masses (the architectural change Prop 6 implies)
w19's conclusion: **"masses must be PER-ADDRESS codebook entries, not a global model parameter. As shipped, `m` is not an address component at all."** Scope what changing this requires — config, model, scorer — and **implement it if bounded**; otherwise deliver a precise change-list for a follow-up task. ⚠ The corrected D1 partition is **K-dependent: `η_m/(η_m + K·η_k)`** (K=2 → 1/21, matched to 13 digits) — the old `1/11` is wrong for multi-item settings. Use the corrected form.

## Acceptance
The reproduced diagnosis, the candidate comparison with ablations, the timescale-vs-mass curve, and either per-address masses implemented or a precise change-list. Tests green.

⚠ **`mass_lr_mult` is not the fix and is not the target.** w19: it buys +0.055 on FD001 but is non-additive with the relax lever (both plateau ≈0.714, still under raw-stats 0.7486), **and both work by suppressing ballistic free-streaming, not by "more physics"** (force share *falls* 0.170→0.101). If a candidate here improves a number by suppressing dynamics, report it as suppression.
