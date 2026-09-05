# Task: generative-studies — FDT-imbalance conjecture + PCD-vs-CD study (Exp-C science)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/generative-studies.md` (+ figures in `.claude/outputs/generative-studies/`)
- **Read first:** protocol · handover §7.9 + §7.4 · F5 Prop-9 (the conjecture) · `.claude/outputs/fix-pack-2.md` (the `langevin_noise` flag + the "dream rollouts not wired" gap + the "negative-chain scales hardcoded" note) · mass-spectrum-peek (checkpoint provenance).

**Goal:** two pending studies on the generative path, now unblocked by fix-pack-2.

## Study A — the FDT-imbalance conjecture (F5 Prop-9 / §7.9)
**Conjecture on file:** legacy Langevin noise gives each mode its own effective temperature `T_eff,i ∝ 1/M_eff,i` ⇒ no Gibbs invariant ⇒ candidate mechanism for the paper's MNIST 3/5/8/9 mode imbalance.
1. **Direct per-mode test on existing checkpoints** (mnist/mnistFF/mnistFFF — skip runaway mnistF for magnitudes): run legacy-noise Langevin chains on the trained model, measure per-mode `Var(p_i)·M_eff,i` — flat (Gibbs-consistent) or spread ∝ predicted `T_eff,i` profile? (You can call `model.stochastic_rollout(..., noise_mode=...)` directly in your own scripts — **do not edit `chlu/`**; the exp_c wiring gap is an engineer item, not yours.)
2. **Generation comparison:** dream batches legacy vs `"fdt"` noise from the same checkpoint + same keys; score digit-mode distribution (simple classifier: sklearn logistic/MLP on MNIST is fine — record it) + sample diversity. Does FDT noise flatten the 3/5/8/9 imbalance? ≥3 noise seeds. NOTE: checkpoint was *trained* with legacy sleep noise — a train/sample mismatch is itself informative; if results are ambiguous, a small retrain-with-fdt run (quick epochs, laptop) is in scope.
3. Verdict on the conjecture: supported / refuted / mixed, with the numbers.

## Study B — PCD vs CD (the §7.4 switch, built in first-fixes)
`training.persistent_sleep_buffer ∈ {False (historical CD), True (paper Algorithm 1 PCD)}` on the **dynamics** path (Exp A or B; pick B — it has the energy-floor/governor readouts). Matched seeds (≥3), quick-to-moderate epochs: compare wake-loss convergence, learned energy landscape (energy gap data-vs-random, target_energy), and downstream noise-rejection MSE at 2–3 σ levels. Question: does persistence change anything the paper should care about? (Either answer settles a §7.4 decision that's been open since day one.)

## Rules
Record every config/flag/seed (defaults changed post-wave-2 — state `lyapunov_penalty`/`langevin_noise` per run). Laptop-scale; note runtimes. Flag (don't fix) any code gaps for the engineer — known ones: exp_c dream-rollout wiring; `train_generative` negative-chain scales hardcoded for [−1,1].
