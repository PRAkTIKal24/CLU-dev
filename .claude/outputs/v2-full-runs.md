# v2-full-runs — results-analyst report

Task + acceptance criterion: paper-grade measured results battery for the V2 short — GMOR δ-sweep (both C2 retention metrics), Mo head-to-head table, γ-sweep through γ\* (C1), emergent-symmetry variant, isotropization falsifiable (F5 §4.1), ≥5 seeds on headline items, EP signatures (stretch).
Status: **done** (all 7 items incl. stretch; one deliberate, diagnosed deviation from defaults — `train_epochs=150` — forced by a new blocking finding, itself fully characterized: Finding 0).

Everything below is laptop-CPU; **no CSF3 needed** (largest single item ≈ 3 min wall).

---

## 1. Setup (configs / seeds / commands)

- **Base:** `main` @ `dbeb2c2` (wave-2 merged), clean tree. Repo read-only; all artifacts under `.claude/scratch/v2-full-runs/` (scripts, checkpoints, logs) and `.claude/outputs/v2-full-runs/` (npz/json tables + figures). Env bug §7.12 hit again (editable `.pth` re-hidden); fixed via `make fix-env`; all commands run as `PYTHONPATH=/Users/user/Desktop/CHLU uv run --no-sync python <script>` from repo root.
- **Two checkpoint batteries** (each: `designed` = so2_invariant+tied ×5 seeds {42..46}; `emergent` = mlp+tied ×3 seeds {42,43,44}; `brokeniso` = so2_invariant+untied ×3 seeds {42,43,44}; all `ExperimentDConfig` defaults otherwise: dim 4, hidden 64, newtonian_learned, dt 0.05, circle R=1.0 ×256 pts, window 64; trained f32 via `run_experiment_d`):
  - **Battery-1 (negative control, kept):** `train_epochs=1000` (the default) → vacuum COLLAPSED in 8/8 so2_invariant runs (Finding 0). Tags `runs/{designed,emergent,brokeniso}_s*`.
  - **Battery-2 (physics battery, used for all items):** `train_epochs=150` → all vacua intact. Tags `runs/*150_s*`. ~16 s/model.
  - **RECORDED training flags (both batteries):** `lyapunov_penalty="max"` (live default; λ=0.01), `langevin_noise="legacy"`, `persistent_sleep_buffer=False`, `sleep_temperature=0.5`, `sleep_frequency=5`, `sleep_friction=0.0`, lr 1e-3, clamp 1000→1 ramp 0.5. (Penalty choice proven irrelevant here — see Finding 0.)
- **Probes:** separate x64 processes on f32 checkpoints (weights cast to f64); vacuum = damped settle (γ=0.1×4000) + BFGS polish to |∇V| ≈ 1e-9..1e-14; modes identified by **eigenvector overlap** with the angular/radial channel directions (robust to mode reordering at large tilt); tilts applied by wrapping the trained checkpoint's potential with `TiltedPotential` (n=1) — no retraining. Probe γ=0.05 (h\*=0.025643, floor 2ln2/(−ln 0.95)=27.03 steps), canonical kick 0.1, dt=0.05.
- **Scripts** (all with full config in headers): `train_battery{,2}.py`, `diagnose_penalty.py`, `diagnose_collapse2.py`, `probe_common.py`, `sweep_gmor.py`, `sweep_gamma.py`, `analyze_emergent.py`, `analyze_isotropy.py`, `sweep_ep.py`, `make_figures.py`.

---

## 2. Finding 0 (new, blocking, fully characterized): sleep-phase erosion destroys the designed vacuum at default epochs

**Figure: `fig7_collapse.png`.** At `train_epochs=1000` (the `ExperimentDConfig` default), every so2_invariant run (8/8 across both variants and 5 seeds) settles to **r\* = 0.0**: the learned radial well is gone, the data ring at R=1.0 sits near a local *maximum*, channel μ² pair degenerate (~1.0) — the SO(2)-degenerate vacuum the whole Exp-D apparatus depends on no longer exists. Downstream, exp-d's own Noether metric divides by r\*→0 and returns NaN (`exp_d_goldstone.py:261`).

**Attribution (2×2 + wake-only + bracketing, seed 42, `diagnosis{,2}.json`):**

| cell | r\*(settle) | ring depth V(0)−V(ring) |
|---|---|---|
| max, 150 ep | 0.9670 | **+0.0596** (intact) |
| legacy_degenerate, 150 ep | 0.9670 | +0.0596 (identical to 4 decimals) |
| max, 300 ep | 0.8726 | +0.0223 (eroding) |
| max, 600 ep | ~0 (6e-16) | **−0.0163 (inverted)** |
| max, 1000 ep | 0.0 | −0.0475 |
| legacy_degenerate, 1000 ep | 0.0 | −0.0455 (identical) |
| **wake-only** (sleep_freq=1e9), 1000 ep | **1.0000** | **+0.0743 (deepest of all; pinned exactly at data R)** |

- **The fix-pack-2 `lyapunov_penalty="max"` default is EXONERATED**: max vs legacy differ at the 4th decimal in every observable, both at 150 and 1000 epochs (λ=0.01·max_i log σ_i is gradient-negligible at this scale).
- **The sleep phase is the destroyer**: monotone erosion of ring depth with number of sleep updates (+0.060 → +0.022 → −0.016 → −0.047 at 30/60/120/200 sleeps), inversion between 300–600 epochs; disabling sleep preserves (indeed deepens and re-centers) the well through 1000 epochs. Mechanism (consistent with the code): PCD negatives thermalize into the low-V ring, CD keeps raising V there, and nothing anchors V's *value* at data (wake is MSE-on-trajectories only) — a measured instance of long-run CD/PCD landscape distortion.
- Consequence for this task: battery-2 trains at 150 epochs (the engineer-validated regime; recorded deviation). Battery-1 kept as documented negative control.

---

## 3. Item 1 — GMOR δ-sweep (headline law) ✅  [`fig1_gmor.png`, `gmor_sweep.npz`, `gmor_per_seed.json`]

Designed battery, 5 seeds × 14 δ ∈ {1e-4…6e-2 (Mo's grid), 0.1, 0.17, 0.3, 0.6, 1, 2, 4}; tilted vacuum is exact (tilt has no radial dependence: q\*(δ) = q\*(0) rotated to θ=π), verified |∇V| ≈ 1e-9.

**Baselines (x64, per seed):** r\* = 0.9591–0.9907, flat-mode μ² = **1.2e-16…2.4e-15** (machine-flat; the previously-reported 8.7e-7 was the f32 probe floor), M_ch = 0.660–0.688, massive spectrum {0.018–0.25 (spectators), 0.67–1.35 (radial)}.

**(a) GMOR spectral-mass law is exact on the learned vacuum:** μ²_meas / [δn²/(M_ch r\*²)] = **1.000000 ± 5e-12** at every δ, every seed, across 4.5 decades. (Not perturbative-only, unlike Mo's ε-linearity: exact for the cos tilt at the exact vacuum.)

**(b) Retention law (both C2 metrics, mean±std over 5 seeds):**

| δ | regime | n₁/₂ envelope | n₁/₂ raw-\|d\| | F5 exact-map pred |
|---|---|---|---|---|
| 1e-4 | overdamped | 90336 ± 3028 | 90326 ± 3028 | 90945 |
| 1e-3 | overdamped | 9048 ± 303 | 9039 ± 303 | 9082 |
| 1e-2 | overdamped | 919.6 ± 30.1 | 910.2 ± 30.4 | 895.9 |
| 0.1 | overdamped | 110.6 ± 2.9 | 100.6 ± 2.9 | 74.4 |
| 0.17 | ≈EP | 75.8 ± 1.6 | 65.4 ± 1.6 | 28.4 |
| 0.6 | underdamped | 38.2 ± 0.4 | 28.0 ± 0.6 | 27.0 |
| 2 | underdamped | 24.8 ± 0.4 | 14.0 ± 0.0 | 27.0 |
| 4 | underdamped | 31.0 ± 0.0 | 9.8 ± 0.4 | 27.0 |

- **Overdamped power law: slope −0.9851** (log n₁/₂ vs log δ, δ≤0.06, n=35; F5 predicts −1); absolute agreement 0.7–3% deep overdamped.
- **Saturation at the mass-independent floor**: δ≥0.6 rows straddle 27.03 (38→31→25→31), within the kick-phase ripple ±(γ/2h)/|ln√(1−γ)| (±8 steps at δ=4) documented by v2-so2-build; near-EP rows (δ=0.17) sit 2.7× above the floor — the C3 algebraic prefactor (see item 7).
- **C2 metric bifurcation measured**: envelope and raw-|d| agree to 0.01% throughout the overdamped band, then split past h\* up to **3.2×** at δ=4 (31 vs 9.8) — first-crossing of the readout is ballistic (∝1/εμ), envelope is retention.
- Constitutive gap (Hessian μ² → exact 2×2 map) ≡ measured one-step-Jacobian gap to **3.2e-10** (max rel dev over all 70 rows): the 2×2 theory *is* the trained map's angular block.

## 4. Item 2 — Mo head-to-head (the generosity+separator figure) ✅  [`fig2_mo.png`]

Mo's exact code-level protocol (φ₀=0.35 rad on the ring, threshold 0.2 rad, cap 15000 steps ≡ his 1500 t.u./dt 0.1, censoring; predicted = 0.847298/gap, gap from the measured Jacobian at the settled tilted vacuum), run unchanged on the 5 trained models × 14 δ:

| regime | measured/predicted |
|---|---|
| overdamped δ=1e-3…1e-2 | **1.012 ± 0.000 → 1.029 ± 0.001** (Mo's published median: 1.013) |
| approaching EP (δ=0.06–0.1) | 1.155 ± 0.005 → 1.313 ± 0.018 |
| at the EP (δ=0.17, h≈h\*) | **2.202 ± 0.155** (theorist's hand-built Check-6: 2.298) |
| underdamped δ=0.6→4 | 0.938 ± 0.019 → **0.309 ± 0.012** (5× failure, calculable direction) |

- Mo-format reporting: uncensored 60/70 (fraction 0.857 — numerically equal to Mo's learned-cell 6/7, a grid-design coincidence worth a wink in a footnote); censored rows are exactly the δ≤3e-4 rows, the same pattern as Mo's own censored ε=1e-4 row. corr(log pred, log meas): **0.9987 overdamped-only** (his regime; consistency reproduced), 0.973 pooled (the drop *is* the regime structure).
- This is mo-deep-read's proposed headline figure, now measured on **trained** CLUs with seed error bars: his law = the overdamped face of our budget table; past h\* his single-exponential predictor fails in the calculable ballistic direction, with the EP delay spike as the third signature.
- Methods caveat measured: Mo's finite-horizon estimator λ̂(T=128) deviates from the asymptotic gap by up to 44% when gap·T ≲ 0.1 (transient-dominated at a fixed point) — we used the exact Jacobian gap for the prediction; report both if adopting his protocol verbatim.

## 5. Item 3 — γ-sweep through γ\* (C1) + latch immunity ✅  [`fig3_gamma.png`, `gamma_sweep.npz`, `flat_immunity_raw.json`]

Radial mode of each trained model (μ²_rad = 0.67–1.35 across seeds → exact γ\* = 0.079–0.110), γ ∈ {0.005…0.64} (12 common + per-seed exact γ\* and 2εμ):

- **C1 retention minimum confirmed, 5/5 seeds:** n₁/₂ = 276–318 (γ=0.005) → **min 31–47** at γ≈0.06–0.08 → 187–370 (γ=0.64). Non-monotone V-shape, ~8× depth. Measured argmin sits at/just below exact γ\* (grid resolution 0.02–0.04 + the near-EP first-crossing bias inflates measured n₁/₂ *at* γ\* by ~2.7×, same C3 prefactor as items 1/2). Branch agreement with exact-map predictions: underdamped 2–6%, overdamped (γ≥0.24) ≤6%; near-EP measured/pred up to 2.7 (documented artifact, not a law failure — the Jacobian eigenvalues match theory to 3e-10 there).
- **Flat-mode immunity at every γ** (re-measured on raw |d| after finding a harness artifact, below): coset angle written-then-held drifts ≤ **1.2e-15 rad over 4000 steps** at all γ ∈ {0.005, 0.05, 0.64} × 5 seeds; raw |d| retention ≥ 0.985 (the flat 0.8% offset is chord-vs-arc chart geometry of the position kick — γ-independent, provably not decay since the angle is frozen).
- **Latch transport law across two decades of γ:** Δθ_meas/Δθ_pred with Δθ_pred = εp₀/(√M r\* γ): **0.9899 ± 0.0026 (γ=0.005, a 1.25-rad transport!) rising monotonically to 1.0000 ± 0.0000 (γ=0.64)** — F5's global-flatness latch verified to ≤1% including trans-linear transports; freeze drift ≤5.7e-5 rad (f32-training-limited) over the last 2000 steps at every γ.
- **Harness caveat found (engineer flag):** `mode_amplitude`'s `mu_floor=1e-8` is *below* the f32-trained flat-mode residual (μ ≈ 4e-8 when probed pre-polish), so the flat mode's "envelope" divides numerical pc noise by μ≈4e-8 and spuriously inflates (observed flat "retention" 8.8 at γ=0.005). Raw |d| / coset angle are the correct latch readings; recommend making `mu_floor` an argument with guidance ≈ 10×√(baseline residual μ²).

## 6. Item 4 — Emergent-symmetry variant ✅  [`fig4_emergent.png`, `emergent_summary.json`, `emergent_ring_profiles.npz`]

Same data, `potential_type="mlp"`, 3 seeds (+5 designed as reference), 16-angle vacuum survey with polish:

- **No near-flat direction emerges.** Softest emergent μ² = **5.1e-2 / 5.9e-2 / 5.4e-2** vs designed flat |μ²| ≤ 2.4e-15: a **13–14 order-of-magnitude gap** between designed and emergent protection. The emergent softest mode is (mostly) angular (overlap 0.70–0.89) but sits squarely in the *register* band, not near latch.
- **Self-induced washboard measured:** V ripple along the ring = 1.2e-2–5.7e-2 — i.e., training a generic MLP on perfectly symmetric data self-breaks at the equivalent of a GMOR tilt δ ≈ 0.01–0.06, *coincidentally inside Mo's breaking grid*. Designed ripple: ≤1.1e-16 (exact invariance). Emergent minima are discrete pinned angles (deepest at r = 1.19/0.55/1.07 — note s43's best basin is far off-ring); designed models settle wherever started (16/16 angles retained, operational ring degeneracy).
- **F5 prices the emergent defect correctly:** measured n₁/₂ (envelope) 277/257/303 vs exact-map prediction from measured μ²: 247/227/263 — **+12–15%** (consistent positive bias: mlp anharmonicity + settle residual; honest deviation, reported as such).
- **Write test contrast:** emergent written angle relaxes back to the pinning angle (finite lifetime, pseudo-Goldstone); designed transports and freezes at the *new* angle (∞).
- **E_eq^V (Mo Eq. 4 refined, 192 pairs):** designed max ≈ **3e-16** (architectural exactness at f64) vs emergent max 0.077–0.113, median ≈ 0.035 — the attribution instrument separates by 15 orders.

## 7. Item 5 — Isotropization falsifiable (F5 §4.1) ✅  [`fig5_isotropy.png`, `isotropy_summary.json`]

brokeniso battery (V exactly invariant by architecture ⇒ E_eq^V ≈ 2.4–3.2e-16 measured; the ONLY breaking channel is kinetic: untied random-init masses), init log-masses reconstructed **bitwise** (f32-dtype draw — see Limitations for the dtype trap I hit and fixed):

- **Falsifiable answered: NO isotropization.** |log M₀ − log M₁| init → 150 ep: 0.157→0.155 (ratio 0.98), 0.037→0.044 (1.18), 0.0004→0.0023 (5.4, absolute change +0.0019 = the generic per-entry drift scale ~0.006–0.013). The split random-walks; symmetric data exerts no measurable isotropizing pressure. **Tied controls: ratio exactly 1.0000** (equal gradients through the tie ⇒ identical Adam updates — doubles as a pipeline validation). Third independent corroboration (after mass-spectrum-peek, v1-l0-gate) that **M does not self-organize; kinetic isotropy must be designed in** (`tie_channel_mass`).
- **The price of the split is exactly where F5 says it is — the charge law, not the statics:**
  - Hessian μ²_ang stays ~1e-15 (kinetic breaking invisible to the fixed-point spectrum; the ring of fixed points survives);
  - the **latch survives too**: angular n₁/₂ = ∞, write-freeze = 0.0 exactly, on all broken models;
  - **Noether violations scale linearly with the split** (E_eq^T = ‖[M,X]‖_F = √2|M₀−M₁|): γ=0 charge drift over 2e4 steps = 5.4e-2 / 1.6e-2 / 8.2e-4 for M-splits 0.0766/0.0219/0.0011 (≈ 0.7×‖[M,X]‖); (1−γ)ⁿ-law error at γ=0.05 = 3.0e-5 / 1.0e-5 / 8.1e-7. Tied controls: 1–3e-14 and 4–7e-16 (machine).
  - **E_eq attribution split works as designed:** E^V ≈ 3e-16, E^T = 0.0016–0.108 — "the breaking lives in the kinetic sector" is now a measured, attributable statement (our refinement of Mo's single-object E_eq).

## 8. Item 7 (stretch) — EP signatures (C3) ✅  [`fig6_ep.png`, `ep_sweep.npz`]

Fine multiplicative grid around δ\* per seed (h−h\* from −4.2e-3 to +2.6e-2):

- **Below the EP: φ = 0 exactly** (15/15 rows — real eigenvalue pair).
- **Above: onset φ ∝ √(h−h\*)** measured from the deployed map's Jacobian down to h−h\* = 2.6e-6: fitted slope **0.5165** (C3 predicts 1/2; slight excess from far-field bending at the top decade), prefactor 0.268.
- Trajectory corroboration where the quality factor permits (f = 2, 4): freq_traj matches freq_jac to **0.06–0.3%**. Near onset the mode decays before one period completes (period ≫ 1/gap), so only the Jacobian sees the complex pair — itself a physically meaningful statement for V2 (the EP is spectroscopically real but dynamically silent at onset).

---

## 9. How I verified (beyond the numbers above)

- Apparatus cross-check: engineer's legacy 150-ep checkpoint re-probed in x64 → r\* = 0.966992645629197, μ² = [1.66e-15, 0.1359, 0.2064, 0.6703] (their reported values, flat mode sharpened from the f32 floor 8.7e-7 to machine-flat).
- Constitutive-vs-operational identity: gap(Hessian μ² → 2×2) vs gap(full 8×8 step Jacobian) agree to 3.2e-10 over all 70 tilt rows.
- Tied-update symmetry: gradient on tied entries measured exactly equal (−0.62856596 both), and raw-split preservation exact over 150 epochs — internal consistency of trainer + reconstruction.
- All runs seeded; every npz row carries seed, δ/γ, measured & predicted values; scripts headers carry full configs and commands.

## 10. Limitations / confounds

1. **train_epochs=150 deviation from defaults** — forced by Finding 0, diagnosed and recorded; all headline results are in the wake-dominated regime. Whether longer *healthy* training (e.g., wake-only or sleep-fixed) changes M_ch/r\*/spectra is untested.
2. Small seed-to-seed variance on GMOR/Mo (M_ch and r\* nearly seed-invariant ⇒ tiny error bars); the real seed variation shows in the massive spectrum (radial μ² 0.67–1.35) and is exercised by the γ-sweep (per-seed γ\*).
3. Single kick amplitude (0.1 canonical); emergent +12–15% retention bias unresolved between anharmonicity and settle residual (a kick-size sweep would decompose it — cheap follow-up).
4. Near-EP first-crossing measurements carry the algebraic-prefactor bias (~2.2–2.7×) by construction; envelope-rate fits are the clean observable there. Metric discipline (C2) is load-bearing everywhere past h\*.
5. Mo comparability: lifetimes in integrator steps; his exact hand-built exp19 systems aren't public — the protocol (from his code) is faithfully reproduced, constants are ours.
6. **Methodology trap (hit & fixed, worth recording):** bitwise init reconstruction inside an x64 probe process silently consumes a different PRNG bitstream (f64 draws) — must request `dtype=float32` explicitly. My first isotropy pass was wrong because of this; corrected and re-verified (tied ratio exactly 1.0000).
7. Battery-1 (1000 ep) emergent runs did *not* collapse (r\* 1.31–1.58) — the sleep-erosion finding is characterized for the so2_invariant architecture; mlp landscapes deform differently (their minima wander off-ring instead). Not further pursued here.

## 11. Recommended next experiments

1. **(engineer, small)** Exp-D fixes: default `train_epochs` → 150 (or expose a sleep-off/anneal switch); guard Noether metric at r\*≈0; make `mu_floor` a parameter of `mode_amplitude`.
2. **(engineer/analyst)** Sleep-erosion study as its own result: erosion rate vs `sleep_frequency`/`sleep_temperature`/`persistent_sleep_buffer`; does PCD persistence (§7.4 switch) change the erosion horizon? Connects to the generative-studies task.
3. **(analyst, cheap)** Kick-size sweep on emergent models to decompose the +13% bias; designed mass-split *sweep* (not just random inits) to nail the Qdrift ∝ ‖[M,X]‖ constant.
4. **(V2 short)** Figure set is ready: fig1 (GMOR law + bifurcation), fig2 (Mo head-to-head — recommended headline), fig3 (C1 + latch), fig4/5 (designed-vs-emergent + isotropy attribution), fig6 (EP), fig7 (sleep erosion, for the methods/honesty section). Next experimental increment for the paper = Mo's S¹ path-integration *task* head-to-head with his published baseline numbers + coRNN/LEM (per mo-deep-read §3.4/§3.6).

## Git footprint

None — no tracked files touched (repo read-only task). All artifacts under gitignored `.claude/`.
Artifacts: `.claude/outputs/v2-full-runs/` = `gmor_sweep.npz`, `gmor_per_seed.json`, `gamma_sweep.npz`, `gamma_per_seed.json`, `flat_immunity_raw.json`, `emergent_summary.json`, `emergent_ring_profiles.npz`, `isotropy_summary.json`, `ep_sweep.npz`, `fig1..fig7*.png`. Scratch: `.claude/scratch/v2-full-runs/` (scripts, 22 checkpoints across two batteries + 7 diagnosis cells, manifests, logs).

## Open questions / follow-ups / risks

1. Where exactly is the erosion horizon as a function of sleep hyperparameters (only bracketed 300–600 at defaults, seed 42)?
2. Emergent +13% retention bias attribution (kick-size sweep).
3. Should the V2 short's GMOR figure use the exact-γ\* form 2ln2/(−ln(1−γ)) everywhere (it does here) — mo-deep-read already mandated this; confirmed numerically (27.03 vs small-γ 27.7).
4. Mo speed risk unchanged: his stated next step is our trained-model experiment; this battery closes the measurement gap on our side.

## Proposed handover updates (for the Hub)

- **§7 new known issue (candidate 7.14):** *Exp-D sleep-phase vacuum erosion.* At `ExperimentDConfig` defaults (1000 ep), wake–sleep training destroys the designed SO(2) vacuum (8/8 seeds: r\*→0, data ring → local max; inversion between 300–600 ep; ring depth +0.060@150 → −0.047@1000). Wake-only (sleep_freq→∞) is intact and *exactly* data-pinned (r\*=1.0000) at 1000 ep. `lyapunov_penalty` default change exonerated (max ≡ legacy to 4 decimals). Also: exp-d Noether metric NaNs at r\*≈0; harness `mu_floor=1e-8` too tight for f32-trained flat modes. Engineer fixes: default epochs 150 (or sleep switch), r\* guard, `mu_floor` param.
- **§1.6/§5 (V2 results, fold in):** V2 battery complete on trained CLUs, 5 seeds, laptop-scale, all under `.claude/outputs/v2-full-runs/`: (i) GMOR μ² law exact to 1e-12 over 4.5 decades; retention slope −0.985 (pred −1); saturation at 27.03 within documented ripple; C2 bifurcation up to 3.2×; (ii) **Mo head-to-head on trained models: 1.012–1.029 overdamped (his median 1.013) → 2.20±0.16 at the EP → 0.309±0.012 deep underdamped; censoring pattern mirrors his** — headline-figure ready; (iii) C1 minimum confirmed 5/5 seeds (~8× depth, at/just below exact γ\*), latch transport ∝1/γ to ≤1% over two decades, coset freeze ≤1e-15 rad at every γ; (iv) designed-vs-emergent protection gap = 13–14 orders in μ²; emergent self-breaking ≈ δ 0.01–0.06 with F5-priced lifetimes (+12–15%); E_eq^V separates architectures by 15 orders; (v) **isotropization falsifiable: NO** (third corroboration of mass inertness) — kinetic breaking invisible to Hessian & latch, fully visible in the charge law, linear in ‖[M,X]‖; (vi) EP onset ∝ √(h−h\*), slope 0.5165, φ=0 exactly below — measured on the deployed map to h−h\*=2.6e-6.
- **F5 v1.1 inputs:** the x64 baseline flat μ² = 1.2e-16–2.4e-15 (machine-flat statement now measured on 5 learned models); EP prefactor on learned models 0.268 (vs theorist's 2×2-normalized 0.3247 — units/normalization to reconcile in one line); λ̂(T) transient bias (≤44% at gap·T≲0.1) as a protocol note.
