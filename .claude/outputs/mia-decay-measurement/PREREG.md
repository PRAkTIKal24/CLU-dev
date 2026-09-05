# PREREG — mia-decay-measurement (w25)

**Written BEFORE the measurement harness was run** (protocol §5 pre-registration rule).
Author: `results-analyst`. Base: local `main` @ `63c668d`. Harness: `.claude/scratch/mia-decay-measurement/mia_harness.py` (to be written after this file).

> Timestamps of record: this file is written at the point where only (a) the shipped code has been *read*, (b) a 3-line timing benchmark of a vmapped read has been run (`bench.py`, no scientific output), and (c) `controller-mvp` §3(b) has been read. No distinguishability number has been computed.

---

## ⭐ DIAL DECLARATION (echoed from the task, protocol §7)
- **Dial:** lifetimes.
- **Laundering control:** (i) a **TTL vector-store** (nearest-neighbour dict, row present until expiry then deleted) and (ii) **CLU-with-a-TTL-flag** — the identical `AtomStorePotential` + controller, but the item is held at **amp ≡ 1.0** until the same expiry tick and then `evict`ed (no amplitude decay). (ii) is the tighter control: same store, same adversary, only "physical amplitude" vs "boolean flag" differs.
- **Falsifies:** distinguishability does not decay with amplitude.
- **Does NOT falsify:** distinguishability persisting *longer* than retention.

---

## 0. Threat models (registered in advance)

| id | access | observable | statistic(s) |
|---|---|---|---|
| **TM-1** | query/read (black box) | for each query: settled address `addr_q` after phase 1, and the shipped tail-mean payload read `v` (tail 25 %, 8 subsamples, coord 2) | `s1 = −mean_q |v − a_i|` (value-return) · `s2 = −mean_q ‖addr_q − c_i‖` (address-capture) |
| **TM-2a** | potential access, **address channel only** (adversary nulls the payload term by minimising over `q₂`) | `U(q) = α|q|² − Σ m_j A_j exp(−d²/2s²)` | `s4 = mean_{8 dirs,|δ|=s} U(c_i+δ) − U(c_i)` (local depth at scale `s`) |
| **TM-2b** | potential access, **full V at q₂ = 0** | `V(q)` as shipped | `s5 = mean_{8 dirs,|δ|=s} V(c_i+δ) − V(c_i)` |
| **TM-3** | TM-2a **with finite measurement resolution** `σ_obs` (each probe evaluation corrupted by `N(0,σ_obs²)`) | as TM-2a | `s4 + noise`; `σ_obs ∈ {0.01, 0.03, 0.1, 0.3}` |
| **TM-4** | raw data-structure dump (`centers`, `payloads`, `amps`, `active` arrays) | — | exact-recovery indicator |

**Per-example U-LiRA discipline (Hayes et al. 2024).** For each target example `i` (a fixed `(c_i, a_i)`), `R = 128` paired *world* draws: background item set `S_r` is re-drawn per world; **IN** = `Store(S_r ∪ {i})` after the decay, **OUT** = `Store(S_r)` never written. Two OUT variants:
- **paired-placement**: OUT uses the *same final background sites* as IN ⇒ the two stores differ **only** by the target atom. Isolates the amplitude channel.
- **history**: OUT re-runs the same background offer sequence with `i` never offered ⇒ background placement differs (refuse-and-relocate / budget history). Isolates the allocator trace (recon item 3 / N99).

Per example: AUC = Mann–Whitney over the `128×128` IN/OUT score pairs; LiRA-Gaussian log-LR ⇒ ROC ⇒ TPR@FPR = 5 % and 1 % (1 % is at the resolution limit of 128 OUT samples — will be reported with that caveat). **Scores are never pooled across examples**; the reported number is the mean ± std of the per-example AUC over `M = 8` targets × 3 seeds = 24 per-example values.

**Store/geometry (fixed):** `AtomStorePotential(dim=3, capacity=8, α=0.02, s=0.35, κ=1.0)`, `d_safe = 4.4·s = 1.54`, proposal disk `R = radius_for_capacity(8, d_safe) = 2.287`, 7 background items + 1 target, payload codebook `designed_payloads(8, seed=0)` (spacing 1/7 = 0.1429 ⇒ `payload_tol = min(0.1, 0.35×0.1429) = 0.0500`). Retrieval exactly as shipped (`dt 0.05`, `γ_addr 0.05 × 400`, `γ_read 0.0 × 800`, tail 0.25, 8 subsamples, `q₂(0)=p₂(0)=0`). Amplitude axis `τ = leak·t = −ln A`, exact for the shipped law `amps *= exp(−leak)`; floor `A_floor = 0.05` ⇒ `τ_evict = ln 20 = 2.996`.

---

## 1. Registered predictions

### P1 — retention is a STEP, not a graded curve (at the native query radius)
The shipped value-recovery retention of a decaying item stays ≈ 1.0 for **every** `A ∈ [0.05, 1]` and falls to 0 only at self-eviction.
*Derivation (two independent reasons, both read off the shipped code):* (a) the payload channel `S(q) = Σ m_j a_j exp(−d²/2s_pay²)` **does not contain `amps`** (`memory_potentials.py` L1047–1051) — decay shallows the address well and leaves the value bump untouched; (b) at the floor the well's restoring force `A/s² = 0.05/0.1225 = 0.41` per unit displacement still exceeds the coercive pull `2α|c| ≈ 0.04×1.5 = 0.06` by ~7×. Consistent with `controller-mvp` §3(b) (retention 1.00 through 8 ticks, then 0).
**Registered numbers:** retention ≥ 0.95 for all `A ≥ 0.051`; retention = 0.00 post-evict.

### P2 — TM-1 (query) distinguishability is also a STEP
**AUC ≥ 0.99 for every `A ∈ [0.05, 1]`; = 0.50 post-evict in the paired condition.** Same mechanism as P1(a): `s1` is amplitude-independent while the OUT world returns `≈0` against `|a_i| ≥ 0.143`.

### P3 — TM-2a (white-box, noiseless) distinguishability does NOT decay either
**AUC = 1.000 for all `A ≥ 0.05`.** The background contributes `exp(−4.4²/2) = 6.3e−5` at `c_i`, while the target contributes `s4 = (1 − e^{−1/2})·A = 0.3935·A ≥ 0.0197` at the floor — a 300× margin. Deterministic system ⇒ perfect separation.
**⇒ I am pre-registering that the DIAL DECLARATION's falsifier FIRES on the AUC metric.** The registered positive claim is the refinement: **the AUC does not decay, but the *effect size* (separation in units of the background scatter) decays exactly linearly in `A`**, slope `0.3935` (units of `A`) on the `s4` statistic. Predicted `s4(A)/A = 0.3935 ± 0.01` across the whole range.

### P4 — the amplitude-independent payload channel (strict)
`|s1(A = 1.0) − s1(A = 0.051)| < 0.01` and `|s5(A = 1.0) − s5(A = 0.051)| < 0.01` in the IN world at the native radius. (If this fails, P1–P3's mechanism is wrong.)

### P5 — the grading lives in the READ RADIUS, not in the amplitude ⭐
The genuinely graded, measurable quantity is `R₅₀(A)`: the adversary/user launch radius at which retention (resp. MIA AUC → 0.75) falls to half. Predicted from the saddle of `A·(d/s²)·exp(−d²/2s²) = 2α|c|` with `|c| ≈ 1.2`:

| A | 1.0 | 0.5 | 0.2 | 0.1 | 0.06 |
|---|---|---|---|---|---|
| **predicted `R₅₀`** | 1.15 | 1.05 | 0.90 | 0.80 | 0.72 |

tolerance ±0.20 each; the registered *shape* claim is monotone decreasing with ratio `R₅₀(1.0)/R₅₀(0.06) = 1.60 ± 0.25`. The TTL controls have `R₅₀` **constant** in age (`R_lookup` for the TTL dict; the `A=1` value for CLU-with-a-TTL-flag).

### P6 — crossing direction (the ⭐ deliverable's headline; my genuine prior)
**MIA outlives retention** — "the store stops answering before it stops leaking."
- On the `τ` axis at the native radius: the two curves **do not cross**; both are steps at `τ_evict = 2.996` (P1 + P2). Post-evict they separate: retention = 0 while the *history*-condition MIA stays > 0.5 (P7b).
- On the radius axis at fixed `A`: `R₅₀^MIA − R₅₀^retention ≥ +0.10` for every `A` (MIA needs only a distributional difference; retention needs `|v − a_i| < 0.05` **and** the correct basin).

### P7 — post-evict residuals
- **(a) paired-placement:** AUC = 0.500 ± 0.05 for every query and potential statistic (the two stores are `V`-identical). A deviation would be a harness bug.
- **(b) history condition:** AUC > 0.5. The strongest statistic is the **hole**: `z = dist(c_i, nearest live site)` — in IN a `d_safe = 1.54` exclusion disk around `c_i` is guaranteed, in OUT it is not. **Registered: AUC(z_hole) = 0.85 ± 0.10**, and `AUC(n_live)` (IN admits fewer background items because `i` caused refusals) `= 0.60 ± 0.10`.
- **(c) TM-4 data-structure dump:** **exact recovery of `(c_i, a_i)` with probability 1.** `AtomStorePotential.evict` (L1017–1027) zeroes only `active` and `amps`; `centers[slot]` and `payloads[slot]` keep the written values verbatim. Registered as a certainty from code-read, to be verified by assertion.

### P8 — TM-3: a resolution-limited adversary DOES see a graded curve ⭐
With probe noise `σ_obs` the `s4` statistic has effective noise `σ√(1 + 1/8) = 1.06σ`, so
`AUC(A) = Φ( 0.3935·A / (√2 · 1.06 · σ_obs) ) = Φ(0.3935 A / (1.5 σ_obs))`.
**Registered point predictions at `σ_obs = 0.1`:**

| A | 1.0 | 0.5 | 0.2 | 0.1 | 0.06 |
|---|---|---|---|---|---|
| **AUC** | 0.996 | 0.905 | 0.700 | 0.603 | 0.562 |

and the registered scaling law **`A₇₅ = 2.57 · σ_obs`** (the amplitude at which AUC = 0.75), tested across `σ_obs ∈ {0.01, 0.03, 0.1, 0.3}` — predicted `A₇₅ = 0.026, 0.077, 0.257, 0.771`. **The matched laundering control (CLU-with-a-TTL-flag) at the same `σ_obs` has AUC ≈ Φ(0.3935/(1.5σ)) ≈ constant until expiry** (its "amplitude" never leaves 1.0) — i.e. a step. *This is the measurement that decides whether "physical amplitude ≠ bookkeeping flag" has anything behind it.*

### P9 — the decay law itself (N94 discipline)
Running `Controller.tick()` at `leak = 0.35` for 8 ticks reproduces `amps[t] = exp(−0.35 t)` to float precision (`0.7047` at `t = 1`), and self-eviction fires at the first `t` with `exp(−0.35 t) < 0.05`, i.e. `t = 9` (`exp(−2.8) = 0.0608 ≥ 0.05`, `exp(−3.15) = 0.0429 < 0.05`).

---

## 2. What each outcome means (registered in advance)

| outcome | reading |
|---|---|
| P3 holds (AUC flat = 1.0 in `A`) | the declaration's falsifier fires **on the AUC metric**: against an exact adversary, amplitude decay buys **nothing** until eviction. Report as the negative it is; the surviving claim is P8 (resolution-relative grading) + P5 (basin-radius grading). |
| P3 fails (AUC decays with `A` even noiselessly) | I mis-modelled the background; the graded claim is stronger than predicted — re-derive the noise floor before claiming it. |
| P8 holds *and* the TTL-flag control is a step | "graded, physical" has a real measurement: distinguishability is graded **relative to a fixed adversary resolution**, and the boolean substitute is not. This is the paper sentence. |
| P8 holds *but* the TTL-flag control is also graded | the differentiator is dead; report as a laundering-control FIRE. |
| P6 holds | "the store stops answering before it stops leaking" — publishable asymmetry, explicitly non-falsifying per the declaration. |
| P7(c) holds | ⚠ a shipped-code finding: eviction is not erasure at the data-structure level. Flag for `experiment-engineer`; it is also the baseline for `order-independent-placement`. |

## 3. Standing scope (registered)
Store-level only: `φ` and the payload channel are separate leak surfaces; no system-level erasure is claimed. Forbidden words: *certified*, *unlearning*, *privacy guarantee* (CM-22 m/n/o). The paper sentence that applies: *"we make no (ε,δ) claim; our guarantee is structural and algorithmic, not statistical."*
