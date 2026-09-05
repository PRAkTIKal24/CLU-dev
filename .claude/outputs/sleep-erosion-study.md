# sleep-erosion-study — results-analyst report

Task + acceptance criterion: characterize CD/PCD wake–sleep *erosion* of the designed degenerate SO(2) vacuum (Head decision 2c, ICLR-grade) — erosion phase diagram vs sleep hyperparameters (Q1), mechanism evidence (Q2), cure verdict (Q3), generality probe (Q4), literature/novelty hook (Q5).
Status: **done** — all 5 questions answered with numbers; 33 checkpointed exp-d runs + Q4 sine battery, 2 seeds on headline items. One methodological upgrade: a **bit-faithful checkpointing replica** of `train_chlu` (validated to max|Δparam|=0.0 vs `run_experiment_d`) so one 1000-epoch run yields the whole erosion trajectory.

Everything below is laptop-CPU. Repo **read-only** (no tracked files touched). Artifacts: figures/tables in `.claude/outputs/sleep-erosion-study/`, scripts+checkpoints in `.claude/scratch/sleep-erosion-study/`.

---

## 1. Setup (configs / seeds / commands)

- **Base:** branch `agent/experiment-engineer/fix-pack-3` — **which has 0 commits ahead of `main` @ `dbeb2c2`** (fix-pack-3 item 1's `sleep_mode` switch is NOT implemented; the branch is empty). Per task instructions I therefore **replicated wake-only via `sleep_frequency=1e9`** (exactly as v2-full-runs did). Confirm this before merging fix-pack-3 — item 1 is still open.
- **Model/data:** `ExperimentDConfig` defaults — designed SO(2): `potential_type="so2_invariant"`, `tie_channel_mass=True`, `newtonian_learned`, dim 4, hidden 64, dt 0.05, circle R=1.0 ×256 pts, window 64. Training flags at repo defaults except the swept ones: `lyapunov_penalty="max"`, `langevin_noise="legacy"`, lr 1e-3, clamp 1000→1 (ramp 0.5), buffer 10000, batch 64.
- **Faithful driver (`driver.py`):** copies `train_chlu`'s loop verbatim for the no-friction-field case (exp-d models carry `friction_field=None`, so every field branch in `train.py` is dead), adds checkpointing at epochs {0,100,…,1000}, per-sleep-event logging of the negatives' radius/energy (Q2), and two Q3 cure hooks. **Validated bit-faithful:** `validate_driver.py` → `max|param diff| replica vs run_experiment_d @150ep = 0.000e+00`, prod r*=0.96699399 (matches v2-full-runs). Probing is a separate x64 pass (`probe.py`, `probe_grid.py`).
- **Metrics (x64, on f32 checkpoints cast to f64):**
  - **ring depth** = `V(0) − mean_θ V(R·(cosθ,sinθ,0,0))` (>0 ⇒ ring is a well; the erosion order parameter).
  - **r\*** = radius of the damped-settled vacuum (γ=0.1, 2000 steps) from a data point (catastrophic-collapse indicator).
  - **noise_gap** = `mean H(N(0,1) noise) − mean H(data ring)`, p=0 (>0 ⇒ model assigns garbage higher energy = the sleep-phase benefit; the "still rejects noise à la Exp-B" check).
- **Grid (33 exp-d runs, seeds {42,43}):**
  - **MAIN phase diagram** (sleep_friction=0 = true exp-d): `sleep_frequency ∈ {1,5,20}` × `sleep_steps ∈ {50,500}` × `persistent_sleep_buffer ∈ {F,T}` = 12 cells × 2 seeds = 24 runs.
  - **TEMP sub-study** (seed 42): `(sleep_friction, sleep_temperature) ∈ {(0,2.0), (0.2,0.5), (0.2,2.0)}` @ freq5/steps500/CD.
  - **CURES** (seeds {42,43}) @ the erosive default freq5/steps500/CD: wake-only, energy-anchor, energy-gated (baseline = the MAIN f5/s500/P0 cell).
  - **Q4** (`q4_sine.py`, seeds {42,43}): Exp-B sine, wake+sleep vs wake-only, via production `train_chlu`.
- **Commands** (cwd = repo root; `chflags nohidden .venv/.../*.pth` preamble per §7.12):
  `PYTHONPATH=$PWD uv run --no-sync python .claude/scratch/sleep-erosion-study/{validate_driver,run_grid,probe_grid,make_figures,q4_sine}.py`

---

## 2. ⚠ First, a knob that does nothing: temperature is inert at the exp-d default

At the exp-d default `sleep_friction=0`, **`sleep_temperature` has zero effect** — legacy Langevin noise is σ=√(2·γ·T·dt) ∝ √γ, and the integrator docstring confirms "gamma must be > 0 for temperature to have effect." Measured proof: `temp_f0.0_T2.0` is **bit-identical** to the `T=0.5` baseline (ring depth −0.1258 vs −0.1258 at 1000 ep). So the task's `sleep_temperature ∈ {0,0.5,2}` axis is **degenerate at γ_sleep=0**; T=0 (deterministic branch) and any T>0 (stochastic branch, zero noise) produce the same conservative isoenergetic negatives. I therefore replaced that axis with a **friction-activated** sub-study (§3.4). This is itself a finding for the methods section: *the sleep temperature reported for Exp-A/B/D is a no-op unless `sleep_friction>0`.*

---

## 3. Results

### 3.1 Q1 — Erosion phase diagram  [`fig1_erosion_curves.png`, `fig2_collapse.png`]

The ring **forms** in the first 100 epochs (wake pins it: ring depth −0.11 → +0.079), then **erodes** at a rate set almost entirely by **sleep-update frequency**. Inversion epoch (ring depth crosses 0), mean over 2 seeds × 2 step-counts × 2 persistence:

| sleep_frequency | inversion epoch (range) | sleep updates @ inversion | state @1000 ep |
|---|---|---|---|
| **1** (every epoch) | **116** (114–119) | 116 | collapsed, r\*=0, rd ≈ −0.07…−0.17 |
| **5** (default) | **442** (391–495) | 88 | collapsed, r\*=0, rd ≈ −0.10…−0.13 |
| **20** | **959** (919–998) | 48 | ~intact, r\*≈0.90–0.93, rd ≈ −0.0002…−0.010 |

**Findings:**
1. **The erosion horizon is set by sleep frequency, not by sleep "work."** More frequent sleep ⇒ earlier inversion (f1 ~116 ep, f5 ~442 ep, f20 ~959 ep). At f20 the ring survives essentially the whole run.
2. **`sleep_steps` (50 vs 500) is IRRELEVANT.** Within every frequency the inversion epoch is identical to ±1 and the final ring depth matches to 3 decimals (e.g. f5: s50 −0.1258 vs s500 −0.1258 @seed42). So the "erosion horizon as a function of total sleep updates × step count" is **not** a function of step count — it is set by the **number of CD gradient applications**. Mechanistically: even 50 conservative steps spread the negatives across their energy shell enough to overlap the ring; longer evolution does not raise the per-update CD gradient at the ring. **Curves do NOT collapse on total sleep-work (updates×steps); they collapse on frequency-vs-clamp-schedule** (fig2).
3. **The wake clamp schedule gates the horizon.** `clamp_strength` anneals 1000→1 over `clamp_ramp·epochs`=500 epochs; while it is strong (0–500 ep) it pins the ring hard. f20 (few sleep events) cannot win until the clamp relaxes ⇒ inverts right at ~960; f1 (1000 sleep events) overwhelms even the strong early clamp ⇒ inverts at ~116, *during* the strong-clamp phase. This is why fewer sleep updates are needed to invert at lower frequency (116→88→48): more wake-epochs elapse per update.
4. **Persistence (PCD vs CD) barely moves the horizon.** Inversion epochs differ ≤9 ep between P0/P1 everywhere; the only visible effect is a slightly deeper *post-inversion floor* under maximal sleep pressure (f1/s50: P1 −0.1714 vs P0 −0.1215 @seed42). **The generative-studies §7.4 null approximately carries over**: negatives *can* reach the ring here (that is why erosion happens at all, unlike the dynamics-path null), but PCD-vs-CD specifically is second-order because the negatives' quasi-stationary shell distribution is similar whether fresh-random-then-evolved or persistently-evolved. **Verdict: persistence does NOT change the erosion horizon; frequency does.**

### 3.2 Q2 — Mechanism: fixed-energy negatives overlap the ring; CD deposits energy there  [`fig3_mechanism.png`]

Tracking the sleep negatives over training (f5/s500/CD, seed 42; `neg_log`):

| phase | frac negatives in ring band \|r−1\|<0.25 | neg mean channel radius | neg mean energy H | ring depth |
|---|---|---|---|---|
| forming (ep 0–140) | **0.22–0.31** | 1.4–1.6 | 3.1–3.8 | +0.079 |
| eroding (ep 300–500) | 0.09–0.22 | 1.6–1.9 | 4.0–5.8 | +0.046 → −0.004 |
| collapsed (ep 700–980) | 0.03–0.11 | 2.3–**3.9** | 7 → **12.3** | −0.069 → −0.126 |

- **Early on, 22–31% of the (conservative, N(0,1)-seeded) negatives sit inside the ring band** — the initial-negative energy shell *geometrically overlaps* the low-V data ring. They do not need noise to get there (γ_sleep=0 ⇒ zero Langevin noise); the random buffer already samples the ring region.
- **CD raises V wherever the negatives are, including the ring.** The signature: neg mean energy climbs **monotonically 3.4→12.3** (CD is doing its job) while **ring depth falls in lock-step** — the energy CD adds is being deposited into (among others) the ring, inverting it.
- **After inversion the ring expels the negatives:** as V(ring) rises, the fixed-energy shell requires larger radius, so neg mean radius grows 1.45→3.9 and the ring-band fraction drops to ~5%. This is the self-terminating "thermalize-then-raise" loop the hypothesis predicted, made quantitative: **overlap → raise → expel**.

### 3.3 Q3 — Cures: the cheap energy-anchor wins on BOTH axes  [`fig4_cures.png`]

All @1000 ep, erosive default (f5/s500/CD); anchor pins `mean_θ V(ring)` to its epoch-0 value (λ=10); energy-gated weights the sleep-energy term by `sigmoid((H_neg−max H_data)/std)`:

| cure | ring depth @1000 | r\* | noise_gap @1000 | vacuum preserved? | noise rejection? |
|---|---|---|---|---|---|
| baseline (disease) | −0.126 | 0.00 | **−0.028** | ✗ inverted | ✗ *prefers noise* |
| energy-gated (b) | −0.126 | 0.00 | −0.028 | ✗ (≡ baseline) | ✗ |
| wake-only (c) | +0.074 | 1.00 | +0.199 | ✓ | ✓ |
| **energy-anchor (a)** | **+0.069** | **0.96** | **+0.244** | ✓ | ✓✓ **best** |

- **Energy-anchor is the verdict.** It holds ring depth **perfectly flat at +0.068…+0.072 for all 1000 epochs (no erosion at all)**, keeps r\*≈0.96, and — because it retains the sleep phase — its **noise_gap grows monotonically +0.172→+0.244, beating even wake-only (+0.199)**. Cheapest cure that preserves the vacuum AND *improves* Exp-B-style noise rejection. Mechanism: the anchor pins V's *value* at data (the thing the diagnosis said nothing anchored), so sleep can no longer drag the ring up; sleep still raises V at off-ring hallucinations ⇒ best of both.
- **Energy-gating FAILS** (identical to baseline). Gating on total H doesn't exclude the ring-eroding negatives because they carry kinetic energy (evolved from N(0,1) momentum) that keeps H above the p=0 data band even when their *position* is on the ring. Gating on V(q) rather than H might work, but the anchor is simpler and already sufficient.
- **Wake-only preserves the vacuum but is the crude fix** (loses the sleep phase; noise_gap +0.199 < anchor's +0.244, and r\* is data-pinned to exactly 1.0 — no operational ring degeneracy left to relax).

### 3.4 Q3-addendum / TEMP — live Langevin noise mildly accelerates erosion

Activating friction+noise (γ_sleep=0.2, live legacy Langevin) @ f5/s500/CD, seed 42, @1000 ep: ring depth −0.1290 (T=0.5) / −0.1304 (T=2.0) vs baseline −0.1258. So **thermalization deepens erosion slightly** (consistent with the mechanism — noise helps negatives reach the ring) but the dominant erosion is present even at zero noise, because the conservative shell already overlaps the ring.

### 3.5 Q4 — Generality: the non-degenerate Exp-B vacuum is IMMUNE  [`q4_sine.json`]

Exp-B sine (dim 1, newtonian_learned, sleep_friction=0.2), wake+sleep vs wake-only, 1000 ep, 2 seeds:

| seed | arm | final wake loss | well_gap H(noise)−H(data) | governed MSE σ=0/0.5/1.0 |
|---|---|---|---|---|
| 42 | wake+sleep | 19.99 | **+5.05** | 1.21 / 1.36 / 1.26 |
| 42 | wake-only | 20.07 | +4.44 | 1.31 / 1.47 / 1.43 |
| 43 | wake+sleep | 13.89 | **+3.08** | 1.23 / 1.15 / 1.74 |
| 43 | wake-only | 13.85 | +2.72 | 1.22 / 1.14 / 1.73 |

- **Sleep does NOT erode the non-degenerate sine vacuum** — wake+sleep and wake-only give near-identical loss and noise-rejection MSE, and the well_gap is actually **slightly higher with sleep** (+5.05 vs +4.44; +3.08 vs +2.72): here the sleep phase behaves as intended (raises hallucination energy) with no structural damage.
- **⇒ Erosion is SPECIFIC to degenerate/flat vacua.** The mechanism explains why: the sine trajectory is non-degenerate, so the clamped MSE pins V's value at *every* data point — there is no flat direction for the CD energy to accumulate along. The SO(2) ring has an exact flat direction the trajectory-MSE cannot see, so V's value along it floats free and sleep fills it in. **This is the crisp scientific statement: wake–sleep CD inverts a designed vacuum iff that vacuum has a flat (degenerate) direction unconstrained by the wake objective.**

---

## 4. Q5 — Literature hook & novelty assessment

*(Citations from training knowledge — years/venues to be confirmed by web-scout; no deep dive per task.)*

- **Wake–sleep:** Hinton, Dayan, Frey, Neal, *"The wake-sleep algorithm for unsupervised neural networks"*, Science 1995.
- **PCD:** Tieleman, *"Training RBMs using approximations to the likelihood gradient"*, ICML 2008 (persistent chains).
- **CD is a biased gradient / "does not follow the gradient of any function":** Sutskever & Tieleman, *"On the convergence properties of contrastive divergence"*, AISTATS 2010; Carreira-Perpiñán & Hinton, *"On contrastive divergence learning"*, AISTATS 2005.
- **CD/PCD energy divergence & landscape distortion:** Fischer & Igel, *"Empirical analysis of the divergence of Gibbs sampling based learning algorithms for RBMs"* (ICANN 2010) / *"Bounding the bias of contrastive divergence learning"* (Neural Comput. 2011); Nijkamp, Hill, Zhu, Wu, *"On the anatomy of MCMC-based maximum likelihood learning of EBMs"*, AAAI 2020 (short-run/non-convergent MCMC ⇒ malformed but usable energy landscapes).

**Novelty verdict.** The *underlying* phenomenon — CD/PCD is a biased update that can drive the energy landscape to diverge/distort — is **classical** (Fischer–Igel, Nijkamp). What appears **novel and worth claiming**, pending a scout confirmation:
1. **A measured *inversion of a designed, symmetry-degenerate vacuum*** — a deliberately engineered flat SO(2) basin is turned into a local *maximum* by prolonged wake–sleep (ring depth +0.079 → −0.126; r\* 1.0 → 0). Landscape *distortion* is known; *this specific "designed structural prior is destroyed by its own contrastive trainer"* framing on a symplectic/EBM primitive is, to my knowledge, undocumented.
2. **The degeneracy-specificity result** (§3.5): erosion strikes flat directions and *spares* non-degenerate vacua because the wake (trajectory-MSE) objective cannot see the flat coordinate — a clean, testable demarcation.
3. **The horizon law** (§3.1): erosion is governed by CD-update *frequency* racing the wake-clamp schedule, **independent of chain length** — a practitioner-relevant scaling not obviously in the RBM-era literature (which focuses on chain-mixing/k).
4. **The V(data)-anchor cure** (§3.3) that preserves both the prior and the contrastive benefit — a targeted fix for "CD has no value anchor," cheaper than full likelihood-based EBM stabilizers.

Recommend positioning: **V2-short methods/appendix + a candidate standalone short "Contrastive divergence erodes designed vacua: when wake–sleep destroys the prior it was given."** Honest framing: we contribute a sharp, quantified *instance + demarcation + cheap cure*, not the discovery that CD is biased.

---

## 5. How I verified

- **Driver faithfulness:** max|param diff| replica vs `run_experiment_d` @150ep = **0.000e+00** (bit-identical); prod r\*=0.96699399 reproduces v2-full-runs.
- **Temperature inertness:** `temp_f0.0_T2.0` ≡ baseline ring depth to 4 decimals (−0.1258), confirming σ∝√γ_sleep.
- **Seed robustness:** all headline items 2 seeds; inversion epochs tight within frequency (f1 114–119; f20 919–998); cures reproduce on both seeds (anchor rd +0.069/+0.077, wake-only +0.074/+0.099).
- **No NaNs/divergence** in any of the 33+4 runs. Every checkpoint carries seed/epoch; every run has `meta.json` (config, wall, final loss) + `neg_log.json`.

## 6. Limitations / confounds

1. **Grid = 2 seeds** (headline) — inversion epochs are tight but the seed-43 f5 inversion (~395) is notably earlier than seed-42 (~490); the *ordering* by frequency is unambiguous across seeds, absolute horizons carry ~20% seed spread.
2. **`sleep_temperature` axis** could only be exercised via `sleep_friction>0` (§2); the requested {0,0.5,2} sweep at γ_sleep=0 is provably degenerate, so the temperature *phase diagram* is a 2-point friction study, not the full 3×… grid. If the Head wants temperature characterized properly, it must be run at γ_sleep>0 (and confounds with damping-driven thermalization).
3. **Anchor cure target = epoch-0 mean V(ring)** (an arbitrary but principled reference); λ=10 not tuned. The cure clearly works (flat to ±0.004 over 1000 ep) but the *optimal* anchor form (absolute V vs V(data)−V(reference-region); learned target) is unstudied.
4. **Q4 uses the governed_rollout / sine setting** (sleep_friction=0.2), so it is not a perfectly matched control to exp-d's γ_sleep=0; the immunity is robust (near-identical arms) but the two experiments differ in more than the vacuum's degeneracy. A matched control (a *non-degenerate* variant of the circle task) would be the cleaner generality test.
5. Citations in §4 are from memory — **web-scout should verify** venues/years and check whether the "designed-vacuum inversion" framing has prior art (esp. in the EBM-stability / continual-EBM literature).

## 7. Recommended next experiments

1. **(engineer)** Implement fix-pack-3 item 1 properly (the branch is empty): default `experiment_d.train_epochs`→150 **and/or** add the energy-anchor as a first-class `experiment_d`/`training` option (`anchor_data_energy_lambda`) — it is the cure, not just a diagnostic. Ship the anchor, not just wake-only.
2. **(analyst, cheap)** Anchor-form ablation: absolute-V anchor vs V(data)−V(far-field) vs a *learned* target; λ sweep; confirm the anchor generalizes to the mlp (emergent) exp-d and to a banded-mass V3 lattice (where a real flat direction is designed in).
3. **(analyst)** Matched non-degenerate control for Q4: a circle task with a weak radial+*angular* pin (tilt δ>0) — verify erosion vanishes continuously as the flat direction is lifted (predicts: erosion rate ∝ flatness).
4. **(analyst)** V(q)-gated (not H-gated) sleep negatives — the energy-gate failed because it used total H; a position/potential gate may recover a sleep-preserving cure without an explicit anchor.
5. **(scout)** Novelty check on "contrastive trainer destroys a designed degenerate prior" + verify §4 citations.

## Proposed handover updates (for the Hub)

- **§7.14 (promote candidate → confirmed, with the full characterization):** *Exp-D wake–sleep vacuum erosion — CHARACTERIZED.* The designed SO(2) vacuum forms by ep 100 (ring depth −0.11→+0.079) then inverts; **horizon set by sleep FREQUENCY racing the wake clamp schedule, independent of sleep_steps and ~independent of persistence:** inversion at ep **116 (f1) / 442 (f5) / 959 (f20)**, sleep_steps 50≡500 to 3 decimals, PCD vs CD ≤9-ep shift. `sleep_temperature` is a **no-op at the exp-d default** (σ∝√sleep_friction=0; `temp_f0.0_T2.0`≡baseline bit-for-bit) — flag for methods. Mechanism measured: initial N(0,1) negatives' energy shell overlaps the ring (22–31% in-band early), CD deposits energy there (neg H 3.4→12.3 monotone as ring depth falls), then expels them (neg radius 1.4→3.9). **Cure = energy-anchor on mean V(data): ring depth flat +0.068…+0.072 over 1000 ep, r\*≈0.96, and noise_gap GROWS to +0.244 > wake-only +0.199** — preserves the prior AND improves noise rejection; energy-gating on H fails. **Generality: non-degenerate Exp-B sine vacuum is immune** (wake+sleep ≈ wake-only; well_gap even slightly higher with sleep) ⇒ erosion is specific to flat/degenerate directions unconstrained by the wake objective.
- **§1.6 / §5 (V2 + training-methodology inputs):** fold the erosion phase diagram (fig1/fig2), mechanism (fig3), and cure (fig4) — ready for the V2-short methods/appendix and the ICLR training-methodology section. The **anchor cure is a shippable contribution**, not just a caveat.
- **Fix-pack-3 status flag for engineer:** branch `agent/experiment-engineer/fix-pack-3` is **empty (0 commits ahead of main)** — item 1's `sleep_mode` switch is unimplemented; I used the `sleep_frequency=1e9` workaround. Also re-flag: `train_chlu` discards sleep-loss history (cheap observability win); and consider exposing `mode_amplitude(mu_floor=...)` (still open from v2-full-runs).
- **Reusable assets:** `.claude/scratch/sleep-erosion-study/driver.py` = bit-faithful checkpointing `train_chlu` replica (validated 0.0 param diff) — reuse for any "metric-vs-epoch" study on the dynamics path; `probe.py` = ring_depth/r\*/noise_gap probes.

## Git footprint
None — no tracked files touched (repo read-only task). Branch `agent/experiment-engineer/fix-pack-3` checked out but **not modified** (and is identical to main). All artifacts under gitignored `.claude/`.
Figures: `.claude/outputs/sleep-erosion-study/{fig1_erosion_curves,fig2_collapse,fig3_mechanism,fig4_cures}.png`; tables: `results.json`, `q4_sine.json`. Scratch: `.claude/scratch/sleep-erosion-study/` (driver, probes, runners, 33 runs × 11 checkpoints, logs).
