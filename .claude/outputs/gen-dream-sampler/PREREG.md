# PREREG — `gen-dream-sampler`

**Written 2026-09-11 23:15 CEST, BEFORE `chlu/core/gen_dream.py` or
`chlu/experiments/exp_gen_dream.py` existed** (protocol §5 pre-registration rule: the acceptance
criterion is a *measured separation*, so the separation is committed to first).
File mtime is the evidence. Nothing below is edited after a number exists; amendments are appended
with their own timestamp.

---

## 0. DIAL DECLARATION (echoed from the task file)

- **Dial:** *isolation* — whether clamping the wells a request addresses actually steers what comes out.
- **Laundering control:** ⭐ the **zero-step rollout** — decode the clamp directly, no Langevin. Run on every number.
- **Falsifies:** zero-step matches the full rollout ⇒ the sampler is inert and we are decoding the clamp.
  Also falsifying: flat energy traces, or a chain that never leaves its initialisation.
- **Does NOT falsify:** blurry / low-fidelity samples in absolute terms. Read as a ratio to the ceilings (R3.2), never absolutely.

---

## 1. ⭐ THE DESIGN DECISION I AM COMMITTING TO IN ADVANCE, AND WHY

**A per-well store energy is SEPARABLE, and a separable energy makes conditional dreaming
provably inert.** If

  `E(x) = Σ_k E_k(x_k)`

then the conditional law of the *relaxed* wells given the *clamped* wells does not depend on the
clamp at all — it is the product of the relaxed wells' own marginals. No sampler, however good, can
steer an unaddressed well from a clamped one. Langevin would then only re-find whatever basin its
initialisation sits in, and the zero-step control would match the full rollout **by construction**.

⇒ The landscape I build therefore has **two terms**:

1. **within-well** mixture free energy over the store's own modes (the store's content), and
2. ⭐ an **inter-well compatibility field** `J_{kl}[m, m']`, fitted as the **pointwise mutual
   information of the empirical code co-occurrence on the TRAIN split** — i.e. a Potts/MRF over the
   `K` code variables, learned from data, label-free, and never a function of the features at read time.

This is a **declared addition to the store**, not a silent one:
- it costs `K(K−1)/2 · M² · 4 B = 10 · 4096 · 4 = 163,840 B` on top of the store's `524,288 B`.
  ⚠ **That breaks `gen-baselines`'s byte-match target as currently filed (524,288 B) and is
  reconciliation item #1 in my report.**
- the read channel is **unchanged at 30 bits** (`K·log₂M = 5·6`, R20's declared design constant);
  the decode still goes `x → address_codes → read_from_codes → decoder`, so the no-bypass
  property is preserved exactly.
- **λ = 0 is run as an explicit arm** (`no_coupling`) so the separability argument is *measured*,
  not merely asserted.

## 2. The sampler, declared in advance (the Head's sampler ruling)

- **Family:** overdamped Langevin on `x ∈ R^D` with the clamped coordinates **hard-projected** back
  to their clamp after every step (conditional sampling on the sub-manifold).
- **Primary: MALA — a Metropolis–Hastings correction IS applied.** Proposal
  `x' = x − ε∇E(x) + √(2εT)·ξ`, accepted with the exact MH ratio for that proposal. I take the
  corrected sampler as primary *because Du et al. (ICML 2023, arXiv:2302.11552) is a critique of
  uncorrected samplers on composed energies, and the cheapest way to be immune to it is to not be
  uncorrected.*
- **Companion arms, same proposal, same seeds:** `ula` (identical proposal, **always accept** — this
  isolates the Metropolis correction exactly, one bit of difference) and `uld` (the repo's own
  underdamped `chlu.core.integrators.langevin_step` at `noise_mode="fdt"`, `m_eff = 1`, so T is in
  energy units — §7.22 discipline; ⛔ `legacy` noise is never used for a reported number).
- **Acceptance is logged every step in every arm**, including `ula`, where the acceptance probability
  is computed **and then ignored** — so the size of the correction the unadjusted arm is skipping is
  on disk and auditable by someone who does not trust us.
- **Annealing:** the soft-assignment temperature `τ` and the sampling temperature `T` are both
  annealed geometrically (start → end). ⚠ Declared in advance because a sharp `τ` gives deep
  separated basins that an unannealed chain cannot leave — the mixing failure Du et al. are about.

## 3. Flag provenance intended for the run (any deviation is reported)

| | value |
|---|---|
| base | `main @ 9a4457d` |
| dataset / pretext / condition | `shapes3d` / `recon` / `recomb2element` (Montero) — `--condition` parameterised for wide |
| seeds | {0, 1, 2}; spread = population std over seeds (ddof=0) |
| decoder | ⭐ **`v1_capacity_schedule`** winner from `gen-decoder-lift`, param hash asserted per seed against `winner/winner_hashes.json` (`01ef91f7…` / `001011bc…` / `2d7eea75…`). ⛔ NOT the `dec_recipe="baseline"` default (0.7191, failed the gate) |
| store | refit from `store_plan.json` **verbatim** (`K=5`, labels, `n_modes=64`, `D=2048`) on the same rows as R0 |
| `langevin_noise` | `fdt` (⛔ never `legacy`) for the `uld` arm |
| metrics | balanced accuracy (R23.1 gate statistic) **and** top-1 accuracy; `shape` reported separately from the aggregate (R16) |
| ceilings | **two, always** (R23.5): the **decoder ceiling** (real features → decoder → probe) and the **STORE ceiling** (real features → store read → decoder → probe) |
| cells | ⛔ **A only.** No B, no C, no kill-rule verdict. |

## 4. ⭐ THE PROTOCOL AMBIGUITY I AM RESOLVING IN ADVANCE (so it cannot be a post-hoc excuse)

An A/B/C request is a **full assignment of all six factors**. If the addressing head's posterior is
sharp on every well, then **every well is clamped, the relaxed set is empty, and `full == zero_step`
identically — for a reason that has nothing to do with the sampler.** Naive-Bayes posteriors are
famously overconfident, so this is a live possibility.

⛔ I therefore commit, before measuring anything:

- I report **two request protocols**:
  - **`full`** — all six factors (cell-native, what `gen-cell-run` will use);
  - **`partial`** — `request_drop_factors = 2` factors dropped per request (seeded, fixed per
    request id), scored on the **requested** factors; this is what "conditional" generation means
    and it guarantees a non-empty relaxed set.
- ⭐ **The headline contrast is read off the protocol whose mean relaxed-well count is ≥ 1.**
  If both qualify, **`full` is the headline.** If `full` has a mean relaxed count of 0, its Δ = 0 is
  reported as a **protocol degeneracy, explicitly NOT as the falsifier firing**, and the headline
  moves to `partial`. `mean_n_clamped` / `mean_n_relaxed` are reported on every row either way.

## 5. ⭐ THE PREDICTIONS (point estimate, then the threshold that decides)

All on the **`clu`** arm unless stated, 3 seeds, Shapes3D/`recon`/Montero, `n_dream = 512` A-split
requests, aggregate = unweighted mean over the six factors' top-1 accuracy.

| # | prediction | point | threshold that decides |
|---|---|---|---|
| **P1** | ⭐ **full rollout − zero-step, aggregate**, on the headline protocol | **+0.08** | **≥ +0.03** and positive on ≥2/3 seeds ⇒ the sampler is NOT inert. **< +0.01 ⇒ FALSIFIED, the sampler is inert** |
| **P2** | full − zero-step on the **`shape`** term (the only compositional factor, R16) | **+0.06** | ≥ +0.02 |
| **P3** | ⭐ **`no_coupling` (λ=0) full − zero-step** — the separability control | **+0.00** | **|Δ| < 0.02**, AND strictly smaller than the `clu` arm's Δ. If λ=0 separates as well as λ>0, my separability argument is wrong and I say so |
| **P4** | **`store_removed`** arm aggregate (identical path, store content deleted) | **0.125** | within ±0.03 of the eval set's mean majority-class rate (R0 measured 0.1245); and its full − zero-step Δ within ±0.01 of 0 |
| **P5** | ⭐ **MALA acceptance rate** at the shipped step size, mean over steps/requests | **0.70** | in [0.4, 0.95]. **< 0.2 ⇒ the step size is wrong and I retune it and say so in the report** (retuning the *step size* is legitimate and pre-registered here; retuning the *claim* is not) |
| **P6** | ⭐ **|MALA − ULA| on the headline aggregate** — the measured cost of skipping Metropolis | **0.01** | **< 0.02 ⇒ "the correction is unnecessary at this scale" is EARNED, with the acceptance trace as its evidence.** ≥ 0.05 ⇒ Du et al.'s critique bites here and I report that it does |
| **P7** | ⭐ **energy trace is non-flat**: mean relative drop `(E₀ − E_T)/|E₀|` on the `clu` arm | **0.05** | ≥ 0.005, AND ≥ 50 % of rollouts change at least one code between step 0 and step T (the chain leaves its initialisation) |
| **P8** | ⭐ **A-cell aggregate as a fraction of `ceiling_A`** (decoder ceiling), `clu` full rollout | **0.45** | in [0.20, 0.70]. ⚠ Bounded above by the **STORE ceiling**: R22 measured the in-corner store read at 0.6866 vs a 0.9414 real-feature ceiling, and R0's A-split store read at 0.5997/0.9507 ≈ 0.63 of the decoder ceiling. ⛔ **I predict the dream CANNOT beat the store ceiling** and that is the number the fraction must be read against |
| **P9** | clamp drift: the clamped coordinates after the rollout | **exactly 0.0** | hard projection ⇒ 0.0 by construction. The *informative* quantity is the **clamp pull** (the pre-projection displacement the landscape applies to the clamp), predicted **> 0** — if it is 0 the coupling is not acting on the clamped wells at all |

### Counter-hypotheses registered now, so that surviving them counts

- **CH-1 (the laundering hypothesis):** the dream is decoding its clamp. ⇒ P1 < +0.01. If this is
  what I measure, I report the sampler as **inert** and do not dress it up.
- **CH-2 (the mixing hypothesis, Du et al.):** the chain cannot leave its initialisation basin at
  the sharp `τ` the store was trained at. ⇒ P7 fails (flat energy, no code changes) even with λ > 0.
  The registered response is **not** to delete the finding: it is to report the annealing schedule
  actually needed and the acceptance trace that shows the failure.
- **CH-3 (the degenerate-clamp hypothesis):** every well is clamped (see §4). ⇒ Δ ≡ 0 on `full`,
  and the headline moves to `partial` under the rule in §4 — **pre-registered, not improvised.**

## 6. What I will NOT do

⛔ No B cell, no C queries, no certificate, no kill-rule verdict, no cell number.
⛔ No decoder retraining or substitution — hash-asserted against `gen-decoder-lift`'s winner.
⛔ No edit to `gen_store.py`, `gen_bridge.py`, anything under `chlu/eval/` or `scripts/csf3/`.
⛔ No CSF3 job (2FA, unreachable): local CPU, and the submit line is emitted and reported **NOT RUN**.

---

# ⟲ AMENDMENT A1 — 2026-09-12 00:34 CEST. Written BEFORE any post-fix number exists.

**What triggered it: SAMPLER DIAGNOSTICS ONLY, from the seed-0 run preserved at
`runs_absolute_tau/` and `run_absolute_tau.log`.** No accuracy number is used to justify
anything below; the three defects are visible in the acceptance / code-change columns alone.

### The three defects seed 0 exposed
1. ⛔ **The soft-assignment sharpness `tau` was specified in ABSOLUTE units** (1.0 → 0.2), inherited
   from the store's *training* schedule. On a 2048-dim feature cache the squared distance from a
   state to a mode is of order hundreds, so `exp(−d²/2τ²)` is **saturated one-hot at every τ in the
   schedule**. A saturated softmax has (a) vanishing gradient in the coupling term and (b) isolated
   quadratic basins with no communication ⇒ the chain cannot move. **Measured: `moved = 0.05`** (the
   fraction of rollouts that changed even one code) on the `clu`/`mala` arm.
2. ⛔ **The step size was tuned once, on the `full` protocol, and reused for `partial`.** The two
   protocols differ ~6× in the dimension of the free block (0.58 vs 3.54 relaxed wells), and MALA's
   acceptance is dimension-sensitive. **Measured: acceptance 0.592 on `full`, 0.052 on `partial`** at
   the same `eps = 1.6`. The `partial` chain was rejecting 95 % of its moves.
3. ⛔ **`energy_relative_drop` divides by a per-sample `|E₀|` that can be ~0** (a clamped well sits
   exactly on its mode, contributing ~0 at small τ). **Measured: `ΔE = −149011.5`** on one arm — a
   division artefact, not a physical quantity.

### The fixes, all declared as SAMPLER TUNING, none touching the claim
- **`tau_calibration = "mode_distance"`** (new default): `tau_start`/`tau_end` become **fractions of
  the store's own median inter-mode distance**, a property of the train-fitted modes alone. ⛔ No
  eval row, no probe reading and no accuracy enters the calibration. The realised absolute values are
  written to the manifest.
- **`temp_calibration = "energy_scale"`** (new default): the sampling temperature becomes a fraction
  of the median `|E(x₀)|` at the reference sharpness — i.e. T is expressed in the landscape's own
  energy units, which is the only way `exp(−ΔE/T)` is meaningful after the τ change.
- **The step size is tuned PER PROTOCOL**, on that protocol's own train requests, still to an
  **acceptance** target (never an accuracy).
- **`energy_relative_drop` becomes a ratio of means**, `(mean E₀ − mean E_T)/mean|E₀|`.
- **New audit column `mean_max_pi`** — the mean top-1 soft-occupancy mass — so softmax saturation is
  a *logged number* rather than an inference from behaviour. ⭐ This is the column that convicts
  defect 1 and it is now on disk for every rollout.

### ⭐ What does NOT change
- Every prediction P1–P9 and every threshold in §5 stands **exactly as written**, and is scored
  against the post-fix run.
- The pre-fix seed-0 numbers are **kept and reported**, not deleted, as the evidence for this
  amendment.
- ⭐ **New pre-registered prediction, filed now:** **A1-P10 — the fix will raise `moved` above 0.5 on
  the `clu`/`partial` arm.** If the chain is *still* frozen after calibrating τ to the store's own
  geometry, then the mixing failure is **structural, not a tuning artefact**, CH-2 is confirmed, and
  I report Du et al.'s critique as **biting here**, in those words.
