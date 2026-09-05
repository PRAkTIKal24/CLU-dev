# Task: mass-lr-doctrine-test — try the mass-specific lr BEFORE the "designed-in" doctrine ships (critique P5/G4)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/mass-lr-doctrine-test.md`
- **Read first:** protocol · `.claude/critique_register.md` (P5/G4 — the reason this task exists) · `.claude/outputs/gamma-field-build.md` (the two-timescale-lr lessons: q-space params can't move at base Adam lr) · `.claude/outputs/seed-sweeps.md` item 1 (banded-vs-uniform testbed + "optimizer never finds the hierarchy") · `.claude/outputs/v2-full-runs/` isotropization verdict · `.claude/outputs/mass-spectrum-peek.md`.
- **Why:** three corroborations say learned M never differentiates → Hyp-3 doctrine "hierarchy must be designed-in or induced." But nobody ever gave `log_mass` its own learning rate — while the γ_φ work *proved* exactly this class of parameter needs one. Cheapest experiment that can save or properly fortify a load-bearing claim. Either outcome is a result: doctrine retired (unlocks the ledger's "mass-narrowness pivot" three ways) or doctrine fortified with the missing control.
- **Git:** branch `agent/experiment-engineer/mass-lr-doctrine-test` — **worktree MANDATORY** (§3.2; `minus-the-physics` runs concurrently and also touches `chlu/`).

## Items
1. **Param-group lr:** `optax.multi_transform` (or equivalent) splitting `log_mass` leaves from the rest. New flag `training.mass_lr_mult` (default **1.0 = bit-compatible** with current behavior; add the round-trip config test). Keep the diff minimal — train.py + config only.
2. **Doctrine test on the banded-lattice testbed** (same data/protocol as seed-sweeps item 1, **uniform-init M**, banding NOT designed in): sweep `mass_lr_mult ∈ {1, 10, 100}` × epochs {300, 1500} × **5 seeds**. Metrics per run: (a) final `log_mass` spread (log-std) vs the known ~0.08 ceiling; (b) alignment of learned M with the data's timescale structure (rank-correlation against the designed-band assignment); (c) eval rollout MSE vs the designed-banded and uniform reference arms.
3. **Curriculum arm (cheap, 1 config):** slow-components-first data ordering at the best mass_lr_mult — does curriculum + mass-lr beat mass-lr alone?
4. **(optional, cheap)** single-unit spot-check: does mass-lr change the V2 isotropization verdict or exp-a M spread? 1 seed each.

## Acceptance criterion
A quotable verdict with error bars: **"mass hierarchy IS learnable given a mass-specific lr (evidence: …)"** or **"doctrine fortified — even at 100× lr / 5× epochs / curriculum, M stays uniform (control now exists)."** Every reported number carries a flag-provenance table (commit, seed, all non-default flags) per protocol §5.
