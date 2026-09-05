# Task: v3-interference-ntk — measure the interference firewall + pricing→task prediction (critiques P12/V3.1 + P13/V3.3)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/v3-interference-ntk.md` (+ figures/npz in `.claude/outputs/v3-interference-ntk/`)
- **Read first:** protocol · `.claude/critique_register.md` (V3.1: "the interference NTK — your own theory's named firewall — is unmeasured"; V3.3) · F5 §6 (potential NTK Θ(q,q′), modularity = only hard firewall) · `.claude/outputs/v3-lattice-build.md` + `seed-sweeps.md` (lattice apparatus, κ_eff extractor) · `.claude/claims_matrix.md` CM-5.
- Repo **read-only**; scratch in `.claude/scratch/v3-interference-ntk/`; laptop-CPU (flag anything >1h for CSF3 instead).

## Items
1. **Measure Θ(q,q′) across units during training** (N=4 and 8; banded vs uniform; 3+ seeds): for pairs of units' data loci, track the cross-unit potential-NTK through training. Deliverables: (a) does one unit's wake update measurably move another unit's basin (interference events: quantify basin displacement of unit B per update batch on unit A)? (b) does banding change the interference structure (heavy/slow units more or less protected)? (c) where does the shared-V_θ interference actually bite vs the F5 catalog's prediction that only modularity is a hard firewall?
2. **P13 — pricing predicts task behavior:** on trained lattices spanning κ (reuse/extend seed-sweeps' trained-coupling models), use the **measured κ_eff to PREDICT a task-level observable before measuring it** (register the prediction in the report first): e.g. recall-horizon or sync-time ranking across lattices. Then measure. Deliverable: predicted-vs-actual table — the V3 short's "the price list is predictive, not descriptive" result.
3. **(cheap)** interference-vs-N scaling sniff: does per-unit interference grow with N at fixed density (the scale-risk V3.1 names)? Even a 2-point (N=4 vs 8) slope with error bars is citable.

**Report:** per-item numbers + error bars, one-line verdicts, flag-provenance tables (§5). This is the V3-short gate evidence (M3 conditional): state plainly whether V3's "guarantees survive scaling" claim survives contact with measured interference.
