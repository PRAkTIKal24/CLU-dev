# mass-spectrum-peek — results-analyst report

Task + acceptance criterion: test Thread-5 falsifiable (i) — do already-trained (CH)LU checkpoints carry non-trivial, interpretable learned mass spectra M = softplus(log_mass)? — using only existing checkpoints (no retraining), with numbers + figures.
Status: **done** (all 6 checkpoints loaded and analyzed; bonus mass-vs-curvature done for all 4 exp_c variants + exp_b).

## Verdict (one paragraph)
**Supported with qualifications.** Every checkpoint whose Hamiltonian actually reads M (exp_b relativistic; 4× exp_c relativistic) moved **every single mass component** away from init (784/784 for each exp_c; KS vs init distribution p≈0), while the identity-mode control (exp_a) stayed **bit-for-bit at init** — so the movement is genuine gradient signal, not optimizer noise. The learned structure is **interpretable and hyperparameter-stable**: all 4 MNIST variants lightened masses globally and imprinted the *same* spatial pattern (pairwise Δ-pattern Pearson 0.40–0.93 despite different lr/dt/friction/temperature *and* different code states), namely **digit-region pixels made lighter, border pixels kept relatively heavy**, with a saturating dependence on per-pixel data variance (Spearman −0.44…−0.72, p<1e-34). Direction matches Thread-5's "semantic inertia" reading (light = responsive/informative dims, heavy = frozen background: q̇ᵢ ∝ 1/Mᵢ). **Qualification:** in 3 of 4 healthy runs the per-dimension differentiation beyond the global shift is *small* (σ_struct = 2–23% of the init σ; max/min ratio ≤1.6, similar to init's own spread) — vanilla 500-epoch EBM training produces a *reliable but weak* mass organization, not yet a mass **hierarchy**. Thread 5 is not refuted; it is under-expressed by current training and likely needs explicit encouragement (longer training, mass-aware objectives, or dynamics data with true timescales — MNIST is static, so "feature timescale" could only be proxied by spatial data variance here).

---

## 1. Setup

**Environment** (recorded from the run): Python 3.11.13, macOS-26.5.2-arm64; jax 0.9.0, jaxlib 0.9.0, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0, scikit-learn 1.8.0; CPU; `jax_threefry_partitionable=True` (default). JAX import was **warm (~2 s)**, not the 20-min cold start — cache was already built this session.

**Commands** (cwd = repo root):
- `uv run --no-sync python .claude/scratch/mass-spectrum-peek/analyze.py` → all stats, fig1–fig6, `results.json`, `arrays.npz`
- `.venv/bin/python .claude/scratch/mass-spectrum-peek/fig3b.py` → fig3b (mean-removed Δ maps)
- Seeds irrelevant (pure analysis); the only RNG used is the *reconstruction* of training-time inits (see below) and a fixed `np.random.default_rng(0)` for the 10⁶-sample init reference.

**Method — exact init reconstruction (stronger than distributional tests).** All projects use `project.seed=42`. I verified via `git show` at the repo-state commit matching each checkpoint's mtime that the PRNG key chains never changed across the relevant code era (Feb 2–10):
- exp_a: `PRNGKey(42)→k1,k2=split→k2,k3=split(k2)→k3,k4,k5,k6=split(k3,4)→CHLU(key=k4)`, dim=2
- exp_b: `PRNGKey(42)→k1,k2=split→k2,k3,k4,k5=split(k2,4)→CHLU(key=k3)`, dim=1
- exp_c: `PRNGKey(42)→k1,k2=split→CHLU(key=k2)`, dim=784 — **identical chain for all 4 mnist\* projects ⇒ all four share bit-identical init log_mass** (verified from the npz: True)
- CHLU internal (all commits): `kk1,kk2=split(key,2); log_mass=normal(kk2,(dim,))·0.1`

**Built-in validation:** exp_a is `newtonian_identity`, whose H never reads `log_mass` ⇒ gradients are identically zero ⇒ plain `optax.adam` update is exactly zero ⇒ trained log_mass must equal init bitwise. **Observed: bit-for-bit equal** (`[-0.03957045, -0.07538826]`, 0/2 components moved). This single check simultaneously validates (a) the key-chain reconstruction, (b) PRNG bitstream stability from the Feb code era to jax 0.9.0 under `partitionable=True`, and (c) absence of optimizer artifacts (no weight decay / numerical drift on zero-grad params). All Δ-from-init numbers below therefore rest on a validated pipeline.

**Init reference distribution** softplus(0.1·N(0,1)): mean 0.6944, σ 0.0501, 0.1–99.9% band [0.550, 0.860].

**Checkpoint provenance** (mode/hyperparams read from the pickles themselves, not the YAML):

| checkpoint | mtime | code state at creation | dim | kinetic | potential | m₀ | c | metadata |
|---|---|---|---|---|---|---|---|---|
| finalA/exp_a | 02-02 10:37 | `4168d51` | 2 | newtonian_identity | PotentialMLP | 1.0 | *(field absent — predates c param `3b18f77`)* | epoch 1000, loss 0.1234 |
| finalA/exp_b | 02-03 08:28 | `76a7685` | 1 | **relativistic** | PotentialMLP | 1.0 | **5.0** | epoch 1000, target_energy 24.886 |
| mnist/exp_c | 02-04 00:48 | `449ba58` | 784 | relativistic | ConvPotential | 1.0 | 1.0 | epoch 500, loss −8 708.7 |
| mnistF/exp_c | 02-04 14:27 | `f16b95a` | 784 | relativistic | ConvPotential | 1.0 | 1.0 | epoch 500, loss **−470 389** |
| mnistFF/exp_c | 02-06 13:27 | `fea7484` | 784 | relativistic | ConvPotential | 1.0 | 1.0 | epoch 500, loss −193.2 |
| mnistFFF/exp_c | 02-10 07:31 | `3497727` | 784 | relativistic | ConvPotential | 1.0 | 1.0 | epoch 500, loss −195.5 |

Confirms handover §3/§5: relativistic is what the paper-relevant checkpoints actually use; finalA exp_b has c=5.0; exp_b `target_energy=24.886 ≲ m₀c²=25` (rest energy dominates H at c=5; settled states sit at slightly negative V+T_excess).

---

## 2. Results

### Q1 — Is M non-trivial? (per-checkpoint stats; `results.json` has full precision)

| checkpoint | M range | mean | CV | max/min | meanΔlog_mass | σ_struct(Δ)† | moved >1e-7 | KS vs init (stat, p) | Pearson(trained, init) |
|---|---|---|---|---|---|---|---|---|---|
| exp_a (control) | 0.656–0.674 | 0.665 | 0.013 | 1.027 | **0 (bitwise init)** | 0 | **0/2** | — | 1 (exact) |
| exp_b | 0.8854 (scalar) | — | — | — | **+0.1677** | — | 1/1 | — | — |
| mnist | 0.531–0.818 | 0.657 | 0.075 | 1.541 | −0.0763 | 0.0206 | 784/784 | 0.308, ≈0 | 0.979 |
| mnistF | 0.238–0.647 | **0.320** | 0.147 | 2.715 | **−0.9858** | **0.1283** | 784/784 | 0.997, ≈0 | 0.597 |
| mnistFF | 0.501–0.794 | 0.645 | 0.074 | 1.585 | −0.1007 | 0.0019 | 784/784 | 0.393, ≈0 | 0.9998 |
| mnistFFF | 0.491–0.777 | 0.620 | 0.076 | 1.584 | −0.1541 | 0.0229 | 784/784 | 0.564, ≈0 | 0.975 |

† σ_struct = std of (Δlog_mass − meanΔ) = amplitude of the *non-uniform* (structured) drift; compare to init σ = 0.1.

- **Movement is universal and signed:** every relativistic model **lightened** its masses on average (meanΔ < 0 in all four exp_c; exp_b is the one exception, +12% heavier — dim=1, different task/objective). Mean shifts are 21–41 standard errors from zero for exp_c (SE = 0.1/√784 ≈ 0.0036) — unambiguous.
- **But spread barely grows** in the healthy runs: trained CV 0.074–0.076 vs init CV 0.072; max/min 1.54–1.59 vs ≈1.6 expected for 784 init draws. The spectrum is *reorganized and shifted*, not (yet) *stretched*. Only mnistF stretched (CV 0.147, max/min 2.72, σ_struct 1.28×init σ, trained-init correlation degraded to 0.60).
- **exp_b:** M: 0.7903 → 0.8854 (Δlog_mass +0.1677, +12.0% in M). Single scalar; direction only, n=1.
- **mnistF caveat:** it is from the pre-regularizer code era and shows an energy-scale runaway (final loss −470 389; V(centroid) ≈ −3.9×10⁵ vs ≈ −80…−94 for the others). Its dramatic global mass collapse (mean M 0.694→0.320) co-occurs with that pathology — treat its *magnitudes* as pathology-adjacent, though its *spatial pattern* agrees with the healthy runs (below).

Figures: `fig1_mass_spectra_all.png` (sorted spectra vs init band — mnistF sits entirely below the band; mnist/FF/FFF straddle it with shifted medians), `fig2_expc_mass_hists.png`.

### Q2 — Is M interpretable? (784-dim spatial structure)

- **28×28 maps** (`fig3_expc_mass_maps.png` common scale; **`fig3b_expc_dlogmass_structured.png`** mean-removed per-panel scale — the informative one): mnist, mnistF, mnistFFF show a clear **center-light / border-heavy frame**: interior (digit region) received extra lightening, the outer 1–2 pixel ring stayed relatively heavy. mnistFFF additionally has a strong lightening blob left-of-center. mnistFF's structure is much weaker (σ_struct 0.0019) with faint interior lightening + a checkerboard-ish texture (possible conv stride-2 artifact — worth remembering when reading conv-potential mass maps).
- **Cross-variant stability** — pairwise Pearson of Δlog_mass patterns (init shared, so this is pure training signal):

  | | mnist | mnistF | mnistFF | mnistFFF |
  |---|---|---|---|---|
  | mnist | 1 | **0.929** | 0.521 | 0.734 |
  | mnistF | | 1 | 0.399 | 0.656 |
  | mnistFF | | | 1 | 0.613 |
  | mnistFFF | | | | 1 |

  Same sign everywhere, mean off-diagonal ≈ 0.64, despite the variants differing in lr (1e-4…1e-3), dt (0.01…0.05), k_steps (20…200), sampler friction/temperature schedule, init mode, *and* code drift (input-noise & energy regularizer added mid-sequence). The pattern is a robust attractor of training, not a fluke.
- **Data alignment** (`fig4_mass_vs_pixelstd.png`): Δlog_mass vs per-pixel MNIST std (70k images): Spearman **−0.720 / −0.652 / −0.440 / −0.601** (mnist/F/FF/FFF), all p<1e-34; Pearson −0.42…−0.65. The relation is a **saturating L-shape**: any pixel with non-negligible data variance is pushed to the maximal-lightening plateau; zero-variance border pixels retain a spread of milder shifts (hence Spearman > |Pearson|). Ink-vs-border means (M): 0.6522/0.6661 (mnist), 0.3036/0.3492 (mnistF), 0.6452/0.6456 (mnistFF), 0.6167/0.6273 (mnistFFF) — ink lighter in 4/4.
- **Physical reading:** with relativistic T, q̇ᵢ = ∂T/∂pᵢ ∝ pᵢ/Mᵢ — the sampler learned to make informative dimensions *fast/responsive* and constant background *stiff/sluggish*. That is Thread-5's "mass prices responsiveness/search", with the empirical sign convention: **information → light, background → inert**.

### Q3 — Mass vs curvature (bonus; `fig5_mass_vs_curvature.png`)

diag Hess V_θ per pixel, at the MNIST centroid and at a "relaxed" state (300-step deterministic rollout from centroid, dt=0.05, γ=0.5):

| variant | Spearman(M, diagH) @centroid | @relaxed | neg. eigenmodes of M^{-1/2}·Hess·M^{-1/2} @relaxed |
|---|---|---|---|
| mnist | −0.159 | −0.156 | 24/784 |
| mnistF | **−0.569** | −0.313 | 3/784 |
| mnistFF | −0.062 | −0.069 | 30/784 |
| mnistFFF | −0.144 | −0.086 | 113/784 |

Weak-to-moderate **anti**-correlation: lighter dims sit on higher-curvature directions (consistent with Q2, since curvature concentrates in the ink region). This is a *hint*, not a law — the mode-frequency object M⁻¹·Hess V is not strongly organized by vanilla training. Caveats: relaxed states are **not minima** (negative eigen-directions present) and drifted outside the trained cube (|q|max 1.43–2.55 > 1; raw `__call__` doesn't apply the training-time clamp), so centroid-state numbers are the safer ones; mnistF's Hessian magnitudes (up to 1753) reflect its energy runaway.

### Q4 — Rest mass & c context

m₀=1.0 everywhere; c=1.0 for all exp_c, c=5.0 for exp_b, exp_a predates the c field (absent from its pickle; its stored config says model default was then 1.0). exp_b extra context (`fig6_expb_potential.png`): learned V(q) has its minimum at q\*=0.010, V(q\*)=−10.96, V''(q\*)=5.902 ⇒ small-oscillation ω = √(V''/(m₀M)) = **2.58 rad/s** vs training band ω∈[0.5, 2.0] — the learned well is stiffer than the fastest training wave; with relativistic KE the effective frequency drops with amplitude, so larger orbits can still match slower waves (speculative, single scalar, not pursued).

---

## 3. Interpretation vs Thread-5 falsifiable (i)

Claim: *"trained CHLUs develop non-trivial mass spectra correlated with feature timescales."*
- **Non-trivial:** yes — gradient flow into M is real, universal (784/784 per model), consistently signed, and the identity-mode control proves it is not an artifact. But the effect is **globally dominated** (uniform lightening) with a small structured residual in healthy runs (σ_struct 2–23% of init σ). No mass *hierarchy* emerges from 500 epochs of vanilla EBM training.
- **Interpretable / feature-correlated:** yes for spatial *information content* (variance), which is the only available proxy on static MNIST — light↔informative, heavy↔background, stable across 4 hyperparameter/code variants. The genuine "timescale" claim is **untested here** (needs dynamics data with per-dimension timescales).
- **Net:** falsifiable (i) survives its cheapest test and gains a sign convention, but the "budget allocator" is currently a *latent tendency*, not an expressed structure. If Thread 5/F5 wants M as the central object, training will likely need to *give M a reason* to differentiate (multi-timescale tasks, longer horizons, or mass-aware regularization) — consistent with the roadmap's plan rather than contradicting it.

## 4. Limitations / confounds
1. **n=1 seed per condition** (all seed 42). Cross-variant agreement partially substitutes for seed replication, but all four exp_c runs share one init — pattern stability across *inits* is unverified.
2. **mnist\* variants differ by code era, not just config** (input-noise σ and the energy-magnitude regularizer were added between mnistF and mnistFF) — cross-variant differences (esp. mnistF's collapse) conflate config and code. It makes the *pattern agreement* more impressive, but forbids clean attribution of *magnitude* differences.
3. **exp_c training data was an unseeded subsample** — `load_mnist_pca` uses `np.random.choice` with no seed (chlu/data/mnist.py:37), so each variant trained on a different random 10k subset; also means exp_c is not bit-reproducible even at fixed seed. (My pixel stats use all 70k images; negligible effect.)
4. Curvature probes evaluated at non-minima and (for the relaxed state) outside the trained cube; diag-Hessian ≠ full mode analysis.
5. exp_b conclusions are a single scalar (n=1 in every sense).
6. Reconstruction rests on the code state matching the checkpoint mtimes and runs from committed code; the exp_a bit-match strongly supports but cannot strictly prove this for exp_c (same-era chains were verified identical in git for every candidate commit, so risk is minimal).

## 5. Recommended next experiments (cheapest first)
1. **Seed sweep** (3–5 seeds) of exp_c-quick: variance of the Δ map; is center-light/border-heavy init-independent? (Fix limitation 1; ~laptop-scale.)
2. **Mass-trajectory logging**: record log_mass every N epochs during one exp_c run — does σ_struct grow monotonically (under-trained) or plateau (structurally limited)? Decides whether "just train longer" closes the gap.
3. **True timescale test for falsifiable (i)**: synthetic dim≥2 dataset with per-dimension frequencies (e.g., dim-i sine at ωᵢ) in relativistic mode — prediction: M anti-tracks ω (heavy=slow). This is the honest version of (i) that MNIST cannot test, and previews V2's M⁻¹·Hess V object.
4. **Theory pass (physics-theorist):** derive sign(∂L_CD/∂log_massᵢ) in terms of real-vs-hallucination momentum second moments — should explain both the global lightening (exp_c) and heavier exp_b, and say when CD training can produce a hierarchy at all.
5. **Engineer fixes:** seed the `load_mnist_pca` subsample; consider clamping in `__call__`-based rollouts used for analysis.

## 6. Environment/ops notes for other spokes
- First run failed with `ModuleNotFoundError: No module named 'chlu'` **while a concurrent agent's `uv sync` was rewriting the editable install** — exactly the mid-session rebuild the task file predicted. `uv run --no-sync` + an explicit `sys.path.insert(0, repo_root)` makes analysis scripts immune. Direct venv import worked seconds later.
- JAX import was ~2 s (warm XLA cache) — the 20-min figure is a *cold-cache* cost, not per-session.
- Old checkpoints (pre-Feb-3 class schema: no `c`/`potential_type` fields) **unpickle and `eqx.combine` fine** under current equinox 0.13.4; just guard attribute access with `getattr`. No template needed — `load_checkpoint`'s `model_template` arg is dead.

Git footprint: none (no tracked files touched; all artifacts under `.claude/`).

Open questions / follow-ups / risks: why does CD training *globally lighten* masses (mechanism)? Is mnistFF's weak structure due to its k_steps=200/dt=0.025 config or code era? Is the checkerboard in mnistFF's Δ map a conv-stride artifact (if so, ConvPotential imprints its architecture on M — relevant to F5's interference section)?

## Proposed handover updates (for the Hub)

**§1.6/§8 (Thread-5 falsifiable (i) status)** — add: *Tested on existing checkpoints (2026-07-04, `.claude/outputs/mass-spectrum-peek.md`). Supported with qualifications: relativistic checkpoints move all mass components (exp_c: 784/784, KS p≈0; global lightening meanΔlog_mass −0.076…−0.99), with a hyperparameter-stable, data-aligned spatial pattern (cross-variant Δ-pattern Pearson 0.40–0.93; Spearman vs pixel-variance −0.44…−0.72: ink light, border heavy) — but structured differentiation is weak in healthy runs (σ_struct 2–23% of init σ; max/min ≤ 1.6 ≈ init) — a latent tendency, not yet a hierarchy. Identity-mode control (exp_a) bit-identical to init (validates pipeline + confirms M is dead weight in newtonian_identity). Mass–curvature (M vs diag Hess V) only weakly anti-correlated (Spearman −0.06…−0.57). exp_b scalar mass +12% (heavier), ω_learned=2.58 vs data ω∈[0.5,2].*

**§5 provenance** — add per-checkpoint ground truth from the pickles: exp_a stored *without* a `c` field (predates `3b18f77`); exp_b confirmed relativistic c=5.0, target_energy 24.886 (≈ m₀c²=25); all mnist\* relativistic/conv c=1.0 epoch 500 with final losses −8 708 / **−470 389** / −193 / −195 — mnistF is an energy-runaway run (pre-regularizer era) and should be flagged as such in the provenance table. All four mnist\* share bit-identical log_mass init (same seed+chain).

**§7 new issues** — (9) `load_mnist_pca` subsamples with **unseeded** `np.random.choice` (chlu/data/mnist.py:37): exp_c training sets are irreproducible even at fixed seed; one-line fix for experiment-engineer. (10) Doc note: checkpoint pickles survive class-schema drift so far (old field sets restore cleanly; `load_checkpoint`'s template arg is unused); attribute access on old checkpoints needs `getattr` guards. (11) Ops: editable install can transiently vanish during a concurrent `uv sync` — spokes should prefer `uv run --no-sync` for analysis and expect to retry.

**§6 ops caveat refinement** — JAX 20-min cost is cold-cache only; warm-session imports are ~2 s (measured today).
