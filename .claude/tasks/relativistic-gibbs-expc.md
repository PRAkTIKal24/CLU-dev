# Task: relativistic-gibbs-expc — F-9: does the missing Maxwell–Jüttner tail drive the MNIST imbalance? (w14, analyst, CHEAP + DECISIVE)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/relativistic-gibbs-expc.md`
- **Read first:** protocol (**§5 flag-provenance is mandatory**) · **`.claude/outputs/v2-symmetry-deepdive.md` §7bis R8 + falsifiable F-9 + open question O8** · `.claude/outputs/generative-studies.md` **Study A** (the result this task re-opens) · `.claude/negative_results.md` **N10** and **N18** · `chlu/experiments/exp_c_dreaming.py` · `chlu/config.py` (`experiment_c`, `training.langevin_noise`).
- **Repo:** read-only. Analysis + report only; flag any code change for the engineer.

## Why this exists — a closed negative is not as closed as we thought
**N10 (tier A)** says: *"the FDT per-mode-temperature violation does NOT drive the MNIST 3/5/8/9 imbalance — the imbalance is the learned landscape, not the sampler"* (χ²=0.08, p=1.00; scale-matched `fdt` vs `legacy` on Exp-C checkpoints).

**But `experiment_c.kinetic_energy_mode = "relativistic"` (`config.py:220`), and R8 proves that in relativistic mode *no* σ gives the coded Langevin a Gibbs invariant** — the O-step is a linear OU recursion (Gaussian stationary law) where Gibbs demands Maxwell–Jüttner. So **both arms of the N10 comparison were non-Gibbs samplers.** N10 tested one sampler defect (per-mode `T_eff` — a *Newtonian-mode* statement) while a second, larger defect was present in both arms and invisible to its instrument (variance-level metrics cannot see a tail-shape error).

Worse, Exp-C's defaults sit exactly where the defect is largest: `m₀=1, c=1, sleep_temperature=0.5`, anneal `1.0→0.01` ⇒ it **starts at `T/(m₀c²) = 1`**, where `Var_MJ/(M_eff T) = 2.70` and `KL(MJ‖Gauss) = 0.384` nats. The paper-run project `finalA` used `c=5` ⇒ `T/(m₀c²) = 0.04`, benign — **which may be why it behaved better.**

**N10 is not refuted. Its evidential basis is weakened, and the sampler-bias hypothesis has never been tested with a nearly-correct sampler.** This task settles it, cheaply, on checkpoints that already exist.

## Items
1. **The direct instrument (do this first — it needs no retraining).** On a trained Exp-C chain, measure the **momentum marginal's shape**, not its variance: histogram `p` at equilibrium and test Gaussian vs Maxwell–Jüttner (excess kurtosis; KL to each; a QQ plot). The coded sampler must give Gaussian; Gibbs demands MJ. **Predicted at `T/(m₀c²)=1`: excess kurtosis ≈1.86, `Var_MJ/(M_eff·T) ≈ 2.70`.** This is a yes/no on whether the defect is live on the real path, independent of any imbalance question.
2. **F-9 proper.** Re-run the Exp-C generative sweep at `c ∈ {1, 5}` (equivalently vary `m₀`), i.e. `T/(m₀c²) ∈ {1, 0.04}`, **everything else fixed**, `langevin_noise="fdt"`. Measure the digit-mode histogram. **Prediction: if the missing MJ tails drive the imbalance, the 3/5/8/9 over-representation measurably shrinks at `c=5`.** Use the same mode-classification and χ² instrument as `generative-studies` Study A so the numbers are directly comparable.
3. **A 2×2, not a 1×2.** Cross `c ∈ {1,5}` with `langevin_noise ∈ {legacy, fdt}` so the two defects (per-mode `T_eff`; missing MJ tail) are separated rather than confounded — that confound is precisely what N10 fell into. State which cell, if any, is the closest thing to a Gibbs sampler this codebase can currently run.
4. **Verdict on N10.** One of: *(a) upheld* (imbalance survives at `c=5` ⇒ landscape, and now on stronger ground); *(b) overturned* (imbalance shrinks ⇒ sampler, and the fix is one free config line); *(c) partial*. Write the replacement/addendum text for the negatives registry either way — **C-9: negatives are never dropped, and a negative that gets re-opened and re-closed is stronger than one that was never challenged.**
5. **Watch the confound the deep-dive names:** varying `c` changes the dynamics, not only the sampler. Include a control that isolates sampling from dynamics (e.g. hold the trained potential fixed and vary `c` **only in the dream rollout**, not in training) — and if the checkpoints were trained at `c=1`, say plainly that `c=5` dreaming on a `c=1`-trained potential is off-distribution, and report both.
6. **`fix-pack-5` interaction:** it lands the `effective_mass()→effective_inertia()` fix, which is correct and necessary but **orthogonal** to this defect. Run against post-fix `main`; state the commit.

**Acceptance:** momentum-marginal shape measured on the real path; the 2×2 run; an explicit verdict on N10 with registry-ready text; the dynamics-vs-sampler confound handled or honestly flagged. **If (b), this is a free fix to the published paper's Exp III mechanism and a genuine result** — say so plainly.
