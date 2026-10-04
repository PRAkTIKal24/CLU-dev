# clu-latent-io-audit — experiment-engineer report

Task + acceptance criterion: establish ground truth on what CLU actually writes into and reads out of the latent — does `log_mass` move (+FD001 delta with mass active), what is `(q₀,p₀)` at train vs inference, read-out inventory tagged, `dt`-units verdict, persistence gate in latent space.
Status: **done** for items 1–5. **Item 6 (context sweep) NOT done** — see Open questions.

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **N7 ("masses stay ≈0.7 = init") is WRONG on this path.** `log_mass` moves by **+3.56** (M: 0.67 → 3.56) at the default `mass_lr_mult=1.0`, and moves **4.28× more per parameter than `V_θ`**. The mass is not frozen and not in a null direction.
> 2. **The theorist's D1 ("rich-gradient potential absorbs the signal; log_mass sits in a near-null direction") is FALSIFIED on this code path.** Measured the opposite: the `log_mass` gradient is **8 200× larger per-parameter** than `V_θ`'s. The Hub's *conclusion* (no timescale hierarchy) survives; its *mechanism* does not.
> 3. **The Hub's "`mass_lr_mult` is not referenced in `clu_scorer.py` ⇒ the mass is inert" inference is half right.** The knob was genuinely absent (now ported, commit `98be102`), but its absence is not why there is no hierarchy.
> 4. **The raw-space persistence gate ("CLU loses to persistence at n=1") does not reproduce under my protocol.** I measure CLU **winning** at n=1 (MSE 0.4545 vs 0.5673) and losing from n=5. Whoever owns that earlier number should reconcile protocols before it is quoted again.
> 5. **The "single-basin collapse" finding is REINSTATED**, in a form that survives my own overflow retraction — see item 4.

---

## Answer first

1. **`log_mass` moves — but only as a COMMON MODE.** Drift is +3.56 on every one of the 14 channels (differential std 0.091, ratio **39:1**). After training, the mass *ratio* `M_max/M_min` is **1.153** — *lower than its random init's* **1.265**. Training makes the mass spectrum **more uniform, not less**. So the Head is right that there is no timescale hierarchy, but the reason is not that the mass is frozen.
2. **`mass_lr_mult=10` gives a real FD001 delta: h-AUROC 0.6540 → 0.7092 (+0.055).** But it does *not* create a hierarchy either (ratio 1.283), and it is **non-additive with the relaxation-budget lever** — both plateau at ≈0.714. **Two knobs, one mechanism.**
3. **There is NO random initial latent anywhere.** The Head's specific concern does not apply. But there *is* a train/inference mismatch of a different kind: training only ever rolls from window frame 0 with **γ=0**, while the dominant inference feature (`q*`) rolls from frame **L−2** with **γ>0**. **The damped dynamics that produces the embedding is never trained.**
4. **`dt=0.05` on cycle-indexed data is a genuine units bug with large consequences.** It inflates K by 400×, making the *energy-magnitude regularizer* **99.2% of the loss** and **99.8% of the mass gradient**. The rollout is **98.3% ballistic free-streaming**. `H` is `K` (corr 0.999996).
5. **The two persistence gates disagree — and the latent one is WORSE for CLU**, the opposite of what the reframing hoped.

---

## Item 1 — is the mass actually being learned?

### The gradient-path partition (theorist OQ1, on the real code path)
Measured at init on real FD001, batch 64, per-parameter RMS gradient norms:

| loss term | value | `|g|` rms on `log_mass` (n=14) | `|g|` rms on `V_θ` (n=5185) |
|---|---|---|---|
| `predict_mse` | 112 | 1.336e+01 | 5.697e-04 |
| `E_contrast` | −0.164 | **0.000e+00** | 7.620e-04 |
| `E_reg` (energy-magnitude) | **6.522e+04** | **7.739e+03** | 9.435e-01 |
| **TOTAL** | | **7.752e+03** | 9.440e-01 |

**Three findings:**
- `log_mass` receives a gradient **8 200× larger per-parameter** than `V_θ`. **D1's premise is inverted here.**
- **The EBM contrastive term has EXACTLY ZERO mass gradient.** Negatives are made by perturbing `q` only, so the kinetic terms cancel identically in `H(data) − H(neg)`. The objective's *representational* half is structurally blind to the mass.
- **99.8% of the mass gradient comes from `E_reg`** — a numerical-hygiene term whose only lever on `H` is to inflate `M` uniformly. **That is a pure common-mode pressure, and it is what actually sets the mass spectrum.**

### What training does to the spectrum (FD001, 150 epochs, seed 42)

| arm | `mass_lr_mult` | relax budget | Std(log M) init→final | common drift | differential | c/d | **M_max/M_min** | movement rms_ratio | **h-AUROC** |
|---|---|---|---|---|---|---|---|---|---|
| default | 1.0 | 0.16 | 0.0877 → 0.1334 | **+3.5634** | 0.0906 | **39.3** | **1.153** | 4.28 | **0.6540** |
| mass10 | 10.0 | 0.16 | 0.0877 → **0.5609** | +8.2454 | 0.5475 | 15.1 | **1.283** | 14.84 | **0.7092** |
| tuned | 1.0 | 1.60 | 0.0877 → 0.1334 | +3.5634 | 0.0906 | 39.3 | 1.153 | 4.28 | **0.7158** |
| tuned+mass10 | 10.0 | 1.60 | 0.0877 → 0.5609 | +8.2454 | 0.5475 | 15.1 | 1.283 | 14.84 | **0.7137** |

*(`M_max/M_min` at random init = **1.265**.)*

**Unambiguous answer.** `log_mass` is **not inert** — it moves 4.28× more per parameter than `V_θ` and its mean travels +3.56. But **the mass SPECTRUM is inert**: the surviving ratio (1.153) is *below* the init ratio (1.265). **Every CLU run in this program has been in a degenerate, single-timescale configuration — the Head's conclusion is confirmed — but by common-mode runaway, not by frozen parameters.**

⚠ **A parameterization trap worth its own line.** `M = softplus(log_mass)`, and `softplus(x) ≈ x` for `x ≫ 0`. Once `E_reg` drives `log_mass` to ≈3.5 (and ≈8.2 at mult=10), softplus is in its **linear** regime, so a *log*-scale spread stops buying *exponential* dynamic range. At mult=10, Std(log M)=0.56 buys a mass ratio of only **1.28**. **The common-mode runaway actively destroys the spectrum's expressiveness.** Any future mass-hierarchy work must either fix the common mode (normalize `log_mass` to zero mean) or reparameterize.

### The FD001 delta, and why it is not what it looks like
`mass_lr_mult=10` buys **+0.055** at the default budget — comparable to the relax-budget lever (+0.061). But **the two are non-additive** (0.7092 / 0.7158 / 0.7137; seed spread ≈0.002 from my prior 3-seed run). Mechanism, measured on trained models:

| arm | mean M | force/free-streaming ratio (γ=0.1, 32) | `q*` cross-sample spread | h-AUROC |
|---|---|---|---|---|
| mult=1 | 1.179 | 0.170 | 5.47 | 0.654 |
| mult=10 | 3.158 | 0.101 | 2.19 | 0.709 |
| mult=1, γ=0.5/64 | 1.179 | 0.969 | 1.41 | 0.716 |

**Both levers work by suppressing the ballistic free-streaming of `q*`** (heavier particle vs more friction) — and the h-AUROC orders exactly with the `q*` spread. Note the force share *falls* under mass10 (0.170→0.101): **the gain is not "more physics", it is "less blow-up".** Both roads dead-end at ≈0.714, still **below the 56-d raw-statistics baseline at 0.7486**.

### The R-1 mass-spread term: implemented, and unusable at `dt=0.05`
Added (`mass_spread_lambda`, on from epoch 0 per T3). **It produced bit-identical results at λ=0, 1 and 50.** Not a bug — measured cause: its gradient is 4.69e-02 against a base of 2.90e+04, a relative perturbation of **1.6e-6**, invisible to Adam. **At `dt=1.0` the same term is at 7% of the base gradient and becomes usable.** ⇒ **R-1 is gated on fixing the `dt` units first.**

---

## Item 2 — the actual `(q₀,p₀)` construction, train vs inference

**Traced from code, not config intent.**

| path | `q₀` | `p₀` | γ | steps |
|---|---|---|---|---|
| **TRAIN** wake MSE (`clu_scorer.py:392-402`) | `batch[:, 0, :]` — window frame **0** | `(frame1 − frame0)/dt` | **0.0** | `predict_horizon`=16 |
| **TRAIN** contrastive (`:404-409`) | all frames `[:-1]` | finite diff | n/a (no rollout) | — |
| **INFER** `predict_mse` (`cafe_model.py:120`) | `w[0]` — frame **0** | `(w[1]−w[0])/dt` | **0.0** | 16 |
| **INFER** `relax`/`basin_coords` → **`q*`** (`cafe_model.py:110`) | `q[-1]` — frame **L−2** | `(w[L−1]−w[L−2])/dt` | **0.1** (0.5 tuned) | 32 (64) |

**Verdict on the Head's concern: there is NO random initial latent, at training or inference.** `φ` is deterministic and identity-like: `q₀` is a literal (z-scored) sensor frame, `p₀` a literal scaled first difference. The Head's "mismatched random init" hypothesis **does not apply** — this is a real negative and should stop that line.

**But there IS a mismatch, and it is on the load-bearing feature.** `q*` — which carries essentially all of CLU's h-AUROC — is produced by a rollout that differs from anything trained in **two** ways:
1. **launch point**: frame L−2, never frame 0;
2. **friction**: **γ=0.1 or 0.5, whereas training uses γ=0.0 exclusively.**

**The damped dynamics that generates the embedding is never trained.** (2) is the serious one: the *only* tuned lever we have (`relax_budget`, +0.061) is a knob on an **untrained** dynamical regime. Combined with item 4, `q*` is largely a damped straight-line extrapolation whose damping was hand-tuned post hoc.

---

## Item 3 — read-out inventory (D=30: 16 scalars + 14 `q*`)

FD001 test split, default config. `auroc` = mean-over-horizons univariate, best orientation. `auroc_bin` = same through a 12-bin monotone-free lookup (gap ⇒ signal a linear probe cannot see). `r2_raw` = out-of-sample R² predicting the feature from 56 raw window stats.

| feature | intended latent property | auroc | gap | r2_raw | tag |
|---|---|---|---|---|---|
| `energy_mean` | basin height / off-manifold | 0.5177 | 0.012 | 0.51 | **CHANCE** |
| `energy_last` | " at window end | 0.5011 | 0.009 | 0.02 | **CHANCE** |
| `energy_std` | energy volatility | 0.5025 | 0.033 | 0.15 | **CHANCE**, non-monotone |
| `energy_trend` | **basin exit** (the headline scalar) | 0.5088 | 0.012 | −0.02 | **CHANCE** |
| `V_mean` | potential depth | 0.5922 | 0.009 | 0.27 | ok |
| `V_last` | " at end | 0.5437 | 0.011 | 0.09 | **CHANCE** |
| `V_trend` | climbing out of basin | 0.5960 | 0.007 | 0.49 | ok |
| `K_mean` | kinetic activity | 0.5169 | 0.009 | 0.51 | **CHANCE** |
| `K_last` | " at end | 0.5011 | 0.009 | 0.02 | **CHANCE** |
| `gradV_mean` | force magnitude / slope | **0.6042** | 0.018 | −0.04 | **best scalar** |
| `gradV_last` | " at end | 0.5454 | 0.010 | −0.01 | **CHANCE** |
| `gradV_trend` | steepening | 0.5807 | 0.005 | −0.02 | ok |
| `relax_residual` | fails to settle | 0.5177 | 0.019 | 0.04 | **CHANCE** |
| `relax_V_star` | settled height | 0.5116 | 0.010 | 0.12 | **CHANCE** |
| `relax_drift` | distance to basin | 0.5265 | 0.007 | 0.17 | **CHANCE** |
| `predict_mse` | dynamics fit | 0.5100 | 0.001 | 0.08 | **CHANCE** |
| `q_star_0..13` | basin identity / settled coords | 0.51–0.61 | ≈0 | **0.67–0.84** | **largely raw redescription** |

**Reads:**
- **11 of 16 physics scalars are at chance (≤0.55).** Every energy and kinetic feature; every relaxation feature; `predict_mse`. The only above-chance scalars are **potential-based** (`gradV_mean` 0.604, `V_trend` 0.596, `V_mean` 0.592, `gradV_trend` 0.581) — never energy-based.
- **`energy_*` ≡ `K_*` to 4 d.p.** (`energy_last` 0.5011 = `K_last` 0.5011; r²_raw 0.0189 vs 0.0188). Direct measurement: **kinetic share of H = 14.7, corr(H,K) = 0.999996, corr(H,V) = 0.055**. The "energy" read-out is a **kinetic** read-out — i.e. a squared scaled first-difference of the raw sensors. **This is item 4 showing up in the read-out.**
- **(c) is essentially a NON-issue.** `auroc_bin − auroc` is <0.02 nearly everywhere. **The information is not hiding behind nonlinearity — it is absent.** Only `q_star_9` (0.050) and `energy_std` (0.033) show any non-monotone signal.
- **(b): `q*` coords carry `r2_raw` 0.67–0.84** — 67–84% of their variance is linearly recoverable from 56 raw window stats, and they still score *below* raw stats' own 0.7486. Consistent with `q*` being a nonlinear smoother of the sensors, not new information.

---

## Item 4 — the `dt` / time-units verdict

**VERDICT: `dt=0.05` on cycle-indexed data is a real mismatch. The data implies `dt = 1.0` (one cycle per sample).**

`p₀ = Δq/dt = 20·Δq` ⇒ K inflated by 1/dt² = **400×**. The *velocity* scale stays self-consistent (a pure time rescale), **but the force and friction do not rescale with it**: the force enters `q` at O(dt²) while free-streaming enters at O(dt), so **the potential's share of the rollout is O(dt)**.

Measured at init on real FD001, identical batch:

| | `dt=0.05` | `dt=1.0` | ratio |
|---|---|---|---|
| mean `H` | 2351 | 6.588 | 357× (≈1/dt²=400) |
| `predict_mse` | 112 | 5.924 | 19× |
| **`E_reg` (share of total loss)** | **6.52e4 (99.2%)** | 0.514 (7.6%) | |
| `E_reg` share of the mass gradient | **99.8%** | 30.6% | |
| R-1 spread term vs base gradient | 1.6e-6 (**invisible**) | 0.071 (**usable**) | |

**Symptom, measured on trained models** — force contribution vs closed-form free-streaming (`‖Δq_force‖/‖Δq_free‖`):

| dt | γ, steps (what it is) | force/free | cos | `q*` spread |
|---|---|---|---|---|
| 0.05 | 0.0, 16 — **the training wake rollout & `predict_mse`** | **0.0166** | −0.85 | 13.73 |
| 0.05 | 0.1, 32 — **the DEFAULT encode relax → `q*`** | **0.0674** | 0.84 | 7.92 |
| 0.05 | 0.5, 64 — the tuned arm | 0.9490 | 0.999 | 1.79 |
| 1.0 | 0.0, 16 | 0.9712 | −0.99 | 1.41 |
| 1.0 | 0.1, 32 | 0.9679 | −0.99 | 0.22 |
| 1.0 | 0.5, 64 | 1.4122 | −0.44 | **0.0000** |

**The training objective's rollout is 98.3% ballistic free-streaming — `V_θ` contributes 1.7%.** The default `q*` is **93% free-streaming**. **This is a candidate root cause of the whole negative result set**: the wake MSE barely trains the potential, `H` is kinetic, and `q*` is a straight-line extrapolation. It also explains why the tuned arm helps — γ=0.5/64 is the only configuration in which the potential actually acts.

**⭐ Single-basin collapse REINSTATED (correcting my own retraction).** At `dt=1.0, γ=0.5, 64` the `q*` spread is 0.0000 with **100% finite values**, per-dim std ~1e-6, and a genuine finite fixed point (|q*| ≈ 0.17–0.32). My earlier retraction was correct *for that measurement* (γ>2 overflow). **At correct time units the collapse is real physics, not overflow.** The learned potential does have essentially one basin.

---

## Item 5 — the persistence gate, in both spaces

**First, the structural point the Hub asked for: with `φ = identity`, the latent `q`-space IS the raw sensor space.** The "latent-space persistence gate" as literally specified is **definitionally the same measurement** as the raw-space one. It cannot be made distinct until a learned `φ` exists. The only non-trivial version available today is **feature space** (the read-out `ψ`), which is the HEPA-shaped question: predict the future *representation* `ψ(w_{t+n})`. Both reported.

FD001 train split, 3000 windows, same-engine guarded, launched from the window's final state:

| n | RAW: CLU | RAW: persist | winner | FEAT (median z²): CLU | FEAT: persist | winner | features CLU wins | ‖traj‖/‖q₀‖ |
|---|---|---|---|---|---|---|---|---|
| 1 | **0.4545** | 0.5673 | **CLU** | 9.657 | **0.171** | PER | 0/30 | 6.85 |
| 5 | 0.5844 | **0.5723** | PER | 10.504 | **0.227** | PER | 0/30 | 7.91 |
| 10 | 1.3794 | **0.5875** | PER | 12.141 | **0.271** | PER | 1/30 | 9.32 |
| 20 | 4.5197 | **0.6272** | PER | 18.021 | **0.347** | PER | 2/30 | 12.24 |

*(FEAT uses **medians**: the CLU rollout diverges on a minority of windows, and a mean reports the divergence rather than the prediction. Mean-based FEAT numbers reach 3.6e8 — reported for honesty, not interpretable.)*

**They disagree, and the disagreement is the finding — but it runs against the reframing.**
- **In raw space CLU genuinely beats persistence at n=1** (0.4545 vs 0.5673, −20%), then loses from n=5. **This does not reproduce the earlier "CLU loses at n=1" claim** — reconciliation #4.
- **In feature space CLU loses at every horizon, by 20–56×, on 30/30 features at n=1.** Because `ψ` is built from energies and `‖∇V‖` — quadratic in a state that free-streams away at 6.9–12.2× over the horizon — **the read-out amplifies exactly the failure mode the rollout has.**

**Honest read: moving to representation space makes CLU look WORSE, not better.** The hoped-for "CLU is being unfairly judged as a raw forecaster" does not hold up with today's handcrafted `ψ`. It may hold with a *learned* `ψ` — but that is an argument for the theorist's design task, not evidence for it, and it should not be asserted until measured.

---

## Git footprint
- Branch **`agent/experiment-engineer/clu-latent-io-audit`**, off local `main` @ `1e7ace5`. Rebased onto `main` (no-op, base unmoved). **Not pushed.** Tree clean.
- Commits: **`98be102`** (port `mass_lr_mult` + mass-spread term + diagnostics to the eval path; tests), **`eae1919`** (expose on the CAFE runner).
- Files: `chlu/eval/clu_scorer.py`, `chlu/eval/config.py` (**additive only** — 2 new fields, both no-op defaults; no existing default changed), `scripts/cafe/run_clu_cafe.py`, `tests/test_eval_mass_lr.py` (new, 9 tests).
- **Verification:** `ruff check` clean on all touched files; **full suite 379 passed** (370 baseline + 9). Default-arm h-AUROC reproduced my prior run **exactly** (0.654 / 0.7158), and univariate AUROCs match to ≤0.001 — the port is behavior-preserving at default.
- One of my own tests failed first (`std_final` did not widen at `mass_lr_mult=10` in the tiny regime). **That was the real common-mode phenomenon, not a flake**; I replaced the assertion with `test_log_mass_drift_is_dominated_by_the_common_mode`, which pins the actual defect.

### Flag provenance
| item | value |
|---|---|
| commit | `eae1919` (CAFE arms produced at `98be102`; identical config) |
| seeds | **42 (single seed for the 4 new CAFE arms)**; prior 3-seed spread on this path ≈0.0017 |
| dataset | `cmapss_fd001`, CAFE loader, window 30, C=14, horizons 1…125, `~/cafe-data` |
| model | `clu`, `encode()`-only, default CoxPH probe (`penalizer=0.1`) |
| CLU config | `kinetic_mode=newtonian_learned`, `potential_type=mlp`, `hidden=64`, **`dt=0.05`**, `gamma=0.1`, `epochs=150`, `lr=1e-3`, `batch=64`, `max_fit_windows=4000`, `predict_horizon=16`, `relax_steps=32`, `neg_noise_scale=0.5`, `energy_reg=0.005`, `momentum_init=finite_diff`, no lattice, `mass_spread_lambda=0.0` |
| varied | `mass_lr_mult ∈ {1.0, 10.0}`; encode `relax_gamma/steps ∈ {0.1/32 (budget 0.16), 0.5/64 (budget 1.60)}` |
| diagnostics | `max_fit_windows=2000` for the ballistic/mechanism tables (stated inline); gradient partition at **init**, batch 64, seed 42 |
| env | JAX **0.9.0**, equinox 0.13.4, main `.venv` (no worktree), CPU; CAFE `~/cafe-bench` @ `dc3dbd0` |

No PREREG: the acceptance criterion is an audit/first-measurement of existing quantities, not a predicted ratio/exponent/law. The lever comparisons are exploratory and reported as such.

---

## Open questions / follow-ups / risks
1. **Item 6 (context sweep) NOT done.** CAFE's C-MAPSS loader hardcodes `window=30`; varying context needs a custom loader (HEPA-style full-engine, cycle-as-patch), which is a task-sized job, not a fold-in. Given items 1–5 landed substantive results I did not start it. **It remains a live confound on the headline number** (handover 2026-07-21) and should be its own task — ideally merged with the HEPA-protocol loader replication, since both need the same loader.
2. **Single seed on the 4 new CAFE arms.** +0.055 is ~30× the known seed spread so it is safe; the **−0.0021** for tuned+mass10 is *within* noise and should not be read as a regression without more seeds.
3. **The `dt` fix is untested end-to-end.** I measured its *mechanism* thoroughly but did **not** run a full FD001 arm at `dt=1.0`. That is the single highest-value cheap follow-up: it plausibly moves `E_reg` out of dominance, makes R-1 usable, un-freezes the mass spectrum, and makes the wake MSE actually train `V_θ`. **Caution:** at `dt=1.0` the potential is strong enough to collapse `q*` to a single point under damping, so the relax budget will need retuning simultaneously — do not vary `dt` alone.
4. **The `q*` rollout is trained at γ=0 and used at γ>0.** Cheapest possible fix, no architecture: train the wake rollout at the γ used at encode time. Worth trying before any `φ`/`ψ` redesign.
5. **Softplus reparameterization.** Zero-mean the `log_mass` (or bound the common mode) so the spectrum keeps exponential dynamic range. Without this, any mass-hierarchy mechanism is fighting the `E_reg` runaway.
6. I did **not** re-run voraus, which shares `_SharedCLUFit`. The port is default-no-op and the default-arm C-MAPSS numbers reproduce bit-for-bit, so I believe voraus is unaffected, but it is unverified.

## Proposed handover updates (for the Hub)
1. **Correct N7 and D1 for the eval path** (reconciliation 1–3): `log_mass` moves +3.56 and carries an 8 200×-larger per-parameter gradient than `V_θ`. **The conclusion "no timescale hierarchy" stands; the mechanism is common-mode runaway driven by `energy_reg`, not gradient starvation.** D1 should be marked "not applicable to `clu_scorer`'s objective; re-derive."
2. **New §7 issue — `dt=0.05` on cycle-indexed data.** `E_reg` = 99.2% of the loss, `H` ≡ `K` (corr 0.999996), the wake rollout is 98.3% ballistic. Candidate root cause of the physics-scalars-at-chance result. **Blocks R-1.**
3. **New §7 issue — train/inference dynamics mismatch:** wake trains only γ=0 from frame 0; `q*` (the load-bearing feature) is generated at γ>0 from frame L−2. The tuned lever operates on untrained dynamics.
4. **New §7 issue — the EBM contrastive term has identically zero mass gradient** (negatives perturb `q` only).
5. **The Head's "random initial latent at inference" hypothesis is DISCONFIRMED** — `φ` is fully deterministic on both paths. Close that line.
6. **`mass_lr_mult` now works on the eval/CAFE path** (`98be102`), plus `mass_spread_lambda`. Both default no-op; §3 config table should gain `CLUScorerConfig.mass_lr_mult` / `.mass_spread_lambda`.
7. **Record the delta:** FD001 `mass_lr_mult=10` → **0.7092** (+0.055 over 0.6540). **Non-additive with the relax-budget lever; both plateau at ≈0.714, still below raw-stats 0.7486.** The raw-stats reference must keep travelling with the number.
8. **Reinstate the single-basin finding** at correct time units (100% finite, genuine fixed point) — it was retracted only for the overflow measurement.
9. **The latent-space persistence gate cannot be distinct from the raw one while `φ` is the identity.** The feature-space version is worse for CLU (20–56×, 30/30 features). **The "we were unfairly measured as a raw forecaster" framing is not yet supported by evidence** and should be held as a hypothesis about a *future learned* `ψ`, not a defence of the current one.
