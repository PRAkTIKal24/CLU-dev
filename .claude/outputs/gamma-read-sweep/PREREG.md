# PREREG — `gamma-read-sweep` (w21)

**Written before any sweep cell was run.** Repo state at time of writing: branch
`agent/experiment-engineer/gamma-read-sweep` off `main` @ `31c3e15`; code changes
(read mode + sweep driver + tests) implemented and unit-tested, but **zero training
cells executed**. Everything below is a prediction, not a report.

---

## 0. The two competing hypotheses (both stated, one nominated)

The task asks whether CLU's 0-of-3 result in `primitive-harness` is an artifact of
γ = 0.05/token, inherited from a w19 measurement that `learned-landscape-write-read`
§5 and `address-space-dimension-scaling` §4 subsequently retracted.

- **H1 — dissipation artifact.** The information from early markers is damped below
  usability before the readout. Falling γ should produce a **sharp transition** in
  the adding-problem MSE at roughly **γ\* = 2 ln2 / T**.
- **H2 — the primitive genuinely cannot integrate.** γ is not the binding constraint;
  the failure is structural. Then the curve is **flat, or monotone-but-shallow**, and
  the negative result stands (and hardens).

### ⭐ I nominate **H2** as the primary prediction. Three reasons, stated in advance.

1. **The near marker is already survivable at the shipped γ, and was not used.**
   `generate_adding` draws marker 1 from `[0, T/2)` and marker 2 from `[T/2, T)`, with
   readout at `T−1`. So the *second* marker sits a mean of **32 tokens** from the
   readout. The code damps `p ← (1−γ)p` once per Verlet sub-step; for a lightly-damped
   mode (half kinetic / half potential over a cycle) the amplitude half-life is
   `ln2 / (−½·clu_steps·ln(1−γ)) ≈ 2 ln2/γ` = **27.0 tokens at γ=0.05**. Surviving
   amplitude at d=32 is therefore **0.44** — barely more than one half-life.
   A block that learned only "output 0.5 + v(nearest marker)" would score
   **MSE = Var(v₁) = 1/12 = 0.0833**, less than half the control floor. The shipped
   CLU scored **0.1825 = the control floor**, i.e. **zero partial credit**. Dissipation
   cannot explain the absence of a signal that dissipation had barely touched.
2. **Amplitude attenuation is not information loss here.** The block output re-enters as
   `LayerNorm(h + block(h))`, which renormalises scale every layer, and the read is a
   trained linear map. A uniform ×0.085 factor is ~10⁻¹, nowhere near float32 noise.
3. **The write current is unconditionally linear in the input.** `CLUBlock` writes
   `p += W_in x_t` with no input-dependent gate anywhere: no multiplicative interaction
   between the marker channel and the value channel *inside* the block. The adding
   target `v_{i1} + v_{i2}` is **bilinear** in the input (value × marker); GRU
   (gates), selective SSM (input-dependent Δ) and attention (softmax QK) all have
   explicit multiplicative gating, and all three solve it to ≤0.001. This is a
   structural handicap that **no value of γ can remove**. (It is not a strict
   impossibility at `n_layers=2` — layer 1's nonlinearity can feed layer 2 — which is
   why H2 is a prediction and not a theorem.)

**Corollary — a second, already-available check that H1 is in trouble.** On parity the
target is supervised at *every* position, so a block with memory `h` tokens should get
roughly the first `h` positions right and chance thereafter:
`acc ≈ (h + (T−h)/2)/T`. At γ=0.05, h=27.0, T=64 ⇒ **predicted 0.71 if memory were the
only limiter.** Shipped CLU parity = **0.538**. The shipped number is already *below*
what the dissipation hypothesis predicts at the shipped γ.

---

## 1. Item 1 — the γ sweep. Numbers I commit to.

Grid γ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}; 3 seeds (42/1042/2042); everything
else byte-identical to the shipped harness (d_model 64, 2 layers, 40 k block params,
LR grid {3e-4,1e-3,3e-3}, tune 400 / train 1200 steps, batch 32, eval 256).

**Derived transition point (registered):** γ\* = 2 ln2/T =
**0.01083 for adding T=128** and **0.02166 for parity T=64** and
**0.01083 for MQAR T=128**. On the grid, γ\*(adding) falls **between 0.01 and 0.02**.

Amplitude surviving to the readout (computed from the code's damping law):

| γ | half-life (tok) | surv @ d=32 | @ d=96 | @ d=128 |
|---|---|---|---|---|
| 0 | ∞ | 1.000 | 1.000 | 1.000 |
| 0.001 | 1386 | 0.984 | 0.953 | 0.938 |
| 0.005 | 277 | 0.923 | 0.786 | 0.726 |
| 0.01 | 138 | 0.851 | 0.617 | 0.526 |
| 0.02 | 68.6 | 0.724 | 0.379 | 0.275 |
| **0.05 (shipped)** | **27.0** | **0.440** | **0.085** | **0.038** |
| 0.1 | 13.2 | 0.185 | 0.006 | 0.001 |

### Predictions (adding, T=128; MSE, lower better; control floor 0.1825)

| # | prediction | falsified if |
|---|---|---|
| **A1** ⭐ | **No sharp transition.** `min` over the γ grid **≥ 0.15**. | any cell ≤ 0.15 |
| **A2** ⭐ | **No cell reaches the near-marker level 0.0833.** | any cell ≤ 0.0833 |
| **A3** | Spread `max − min` over the whole grid **≤ 0.035** | spread > 0.035 |
| **A4** | MSE(γ=0) ∈ **[0.15, 0.20]** | outside |
| **A5** | If H1 is right instead, the transition sits at γ ∈ [0.005, 0.02] and MSE(γ≤0.005) ≤ 0.05 while MSE(γ≥0.05) ≥ 0.15 | — (this is the H1 branch, scored if A1 fails) |
| **A6** | ≤ 1 of 3 seeds diverges (NaN) at γ ≤ 0.001; γ=0 removes the only energy sink for the driven Hamiltonian, so divergence is a live risk and will be reported, not silently dropped | >1 seed diverges and I fail to report it |

### Predictions (parity, T=64; accuracy, chance 0.5; shipped 0.538)

| # | prediction | falsified if |
|---|---|---|
| **P1** ⭐ | accuracy **≤ 0.60 at every γ** | any cell > 0.60 |
| **P2** | best-γ − shipped(0.538) **≤ +0.06** | > +0.06 |
| **P3** | the finite-memory model (acc ≈ (h+(T−h)/2)/T) **over-predicts** at every γ, i.e. measured < predicted | measured ≥ predicted at ≥2 γ |

### Predictions (MQAR T=128 kv=4; accuracy, chance 0.008; shipped 0.3464)

MQAR is the family where H1 has its best case: keys are early, queries are late, and
CLU demonstrably *is* doing the task (0.35 vs a 0.012 no-mixing floor), so any real
memory-length effect should show here.

| # | prediction | falsified if |
|---|---|---|
| **M1** | best-γ accuracy ∈ **[0.34, 0.46]** | outside |
| **M2** ⭐ | best-γ accuracy **< 0.486 (the GRU)** — γ does not produce a ranking change | ≥ 0.486 |
| **M3** | improvement over shipped **≤ +0.11**; and — unlike adding/parity — I do expect a *monotone* improvement as γ falls here, of **at least +0.02** | no improvement at all, or > +0.11 |

---

## 2. Item 2 — trajectory read vs endpoint read

Implemented as the Prop-11 fiber read: `y_t = W_out[q_1;p_1;…;q_K;p_K]` over the
`K = clu_steps` intra-token sub-steps, versus the shipped `y_t = W_out[q_K;p_K]`.

| # | prediction | falsified if |
|---|---|---|
| **R0** ⭐ | **At `clu_steps=1` the two read modes are the SAME MAP, bit-exactly** (the fiber has one element; `w_out` has identical shape and consumes the identical key). Measured spread across read mode at clu_steps=1 = **exactly 0.0000**. Consequence: **the shipped harness's read-mode axis is degenerate and the 2-D table is only non-trivial for `clu_steps>1`.** | any nonzero difference (that would be a bug) |
| **R1** | At `clu_steps>1`, trajectory − endpoint on adding: **|Δ MSE| ≤ 0.02** (no rescue) | > 0.02 in CLU's favour |
| **R2** ⭐ | **No γ × read-mode interaction**: (traj−end at γ=0) − (traj−end at γ=0.05) is **< 0.02 MSE**. I.e. the task's registered prediction — "the trajectory read is the one that benefits from γ→0" — is predicted **not to hold** at this budget. | interaction ≥ 0.02 in the predicted direction |
| **R3** | The fiber read is paid for out of state width: at matched 40 k block params, `d_clu` **shrinks** for trajectory mode, roughly as `w_out` grows K×. Registered: `d_clu(traj, K=2) < d_clu(end) = 83`. | d_clu does not shrink |

---

## 3. Item 3 — `clu_steps`

Note a confound I register in advance: at fixed γ, `clu_steps=K` damps **K times per
token**, so `γ_eff(token) = 1−(1−γ)^K` and the half-life shortens by exactly K. More
integration therefore *buys more dissipation* unless γ is rescaled. Both the raw
sweep and the half-life are reported.

| # | prediction | falsified if |
|---|---|---|
| **S1** | wall-clock multiple vs `clu_steps=1`: **≈1.0 / 1.7 / 3.1** for K=1/2/4 (sub-linear: `w_in`, `w_out`, scan overhead and the token loop are shared; only ∇V scales) — each within ±0.5× | outside ±0.5 |
| **S2** ⭐ | MSE improvement from K=1→4 on adding at the best γ: **≤ 0.02**, and I expect it to be **non-monotone or worse** because of the γ_eff confound | improvement > 0.02 |

---

## 4. Item 4 — the corrected three-family table

| # | prediction |
|---|---|
| **F1** ⭐ | **No better configuration will be found**, so the corrected table will not be produced and **"CLU wins 0 of 3 families" stands.** The one hedge: MQAR may improve by a few points (M3) without changing any ranking. |

**If F1 is wrong** I will run the full symmetric LR-rescue pass at the winning config
and state the fairness category of every changed knob (all three — γ, `clu_steps`,
read mode — are category (a): knobs no other primitive has).

---

## 5. What would make me change my mind (the H1 branch)

If **A1 or A2 fails** — any adding cell at ≤0.15, or ≤0.0833 — H2 is dead and I will
report the transition, locate γ\* on the grid, compare it to the derived 0.01083, and
run Item 4 in full. I am not going to defend H2 against the data.

**No tuning past this grid.** Any point outside {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}
× {endpoint, trajectory} × {1, 2, 4} is a separate, explicitly-labelled exploratory arm.
