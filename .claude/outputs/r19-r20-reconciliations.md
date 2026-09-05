# r19-r20-reconciliations — results-analyst report
Task + acceptance criterion: adjudicate the three *measurement* conflicts (R19-1 persistence gates, R19-2 single-basin collapse, R20-1 ballistic fraction) — per-conflict verdict + reproducible evidence + proposed registry wording; pre-register which side survives.
Status: **done** — all three adjudicated, each with a decisive run. No code touched (analysis only).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines). For the next curator/Hub pass:**
> 1. **R19-1 RESOLVED:** the n=1 sign is carried **entirely by the launch frame** (L-2 audit vs L-1 fetch), *not* by γ/dt/anchors. The defensible "does the rollout beat persistence" measurement launches from the **last observed frame (L-1)**, and under it **CLU LOSES at every horizon incl. n=1** (0.7705 vs 0.5669, legacy). N86's direction is upheld; the audit's L-2 "win" is a stale-launch artifact. Proposed wording below → curator applies to N86 / R19-1.
> 2. **R19-2 RESOLVED — the reinstated single-basin collapse does NOT survive re-verification.** It is a ~10-epoch under-trained transient (spread 9.15e-08 at ep10 → 0.31–0.67 at ep150, 3 seeds, 100% finite, f32≡f64). Neither overflow *nor* a trained-model property. **Do NOT promote as an N-candidate "the learned potential has one basin."** Proposed wording below.
> 3. **R20-1 RESOLVED:** 98.3% was a 40-epoch/2000-window under-trained fit; the fully-trained legacy number is **~77–80%**, shipped-`dt` is **~50%**. One canonical statement below.
> 4. **Out of scope (untouched):** R19-1 issue **(b)** — the Hub's withdrawal of the raw-space gate as "testing a claim we don't make". I resolved only **(a)**, the measurement conflict. R19-3, R19-4 untouched per task.

---

## Setup — configs, seeds, commands (all reproducible)
- **Repo:** `main` @ `7ff0651` (post-`dt-units-split`; shipped defaults `dt=0.125, data_dt=1.0, substeps=8, epochs=150, γ=0.1, relax_steps=32`). Legacy = conflated `dt=data_dt=0.05`. Reinstating config = conflated `dt=data_dt=1.0`.
- **Env:** main `.venv` (no worktree), JAX **0.9.0**, equinox 0.13.4, CPU. Data: FD001 train windows `(17731,30,14)` from `.claude/scratch/dt-units-split/fd001.npz` (raw CAFE loader output). Fits are production-faithful via `CLUCafeMixin.fit` (standardize → `_SharedCLUFit.ensure_fit`, subsample to `max_fit_windows`, train).
- **Harness:** `.claude/scratch/r19-r20-reconciliations/adjudicate.py` (+`r19_2_40ep.py`). Outputs `r19_1.json`, `r20_1.json`, `r19_2.json`, `*.log`.
- **Commands:** `adjudicate.py r19_1` · `r20_1` · `r19_2`; then `r19_2_40ep.py`. Seed 42 (+43,44 for R19-2).
- **PREREG:** `.claude/outputs/r19-r20-reconciliations/PREREG.md` written before the harness ran. **All three pre-registrations HELD** (details per section).

---

## R19-1 — the two persistence gates disagree  ✅ RESOLVED: **launch frame carries the sign**

### The protocol deltas (enumerated)
| delta | audit `clu-latent-io-audit` (CLU WINS n=1) | fetch `cmapss-fd002-004-fetch` (CLU LOSES n=1) |
|---|---|---|
| **launch frame** | **L-2** (`q0=w[-2]`) — = the shipped `q*` encode launch (`_window_features:120`, `q[-1]=w[L-2]`) | **L-1** (`q0=w[-1]`, literal last frame) |
| target/alignment | traj-step-k vs cycle E+1+k ⇒ **1-cycle lag** (a consequence of the L-2 launch) | 1 step ↔ 1 cycle ahead ⇒ **aligned** |
| γ | 0.0 | 0.5 (also reports 0.0/0.1) |
| persistence baseline | hold frame E (=L-1) | hold frame E (=L-1) — **identical** |
| anchors | 3000, same-engine guarded | 1500, unit-id eligible |
| dt era | legacy 0.05 | legacy 0.05 |

Persistence agrees across both (≈0.567 vs 0.600 = anchor-sampling noise). The disagreement is entirely in the **CLU** number.

### The crossing (single-knob flip, legacy dt=0.05, 150ep, seed 42; `r19_1.json`)
n=1 CLU MSE vs persistence 0.5669:

| launch | γ=0.0 | γ=0.5 | verdict |
|---|---|---|---|
| **L-1** (fetch) | **0.7705 → PER** | **0.7705 → PER** | CLU **loses** — reproduces fetch (0.825 vs 0.600) |
| **L-2** (audit) | **0.4568 → CLU** | **0.4568 → CLU** | CLU **wins** — reproduces audit (0.4545 vs 0.5673) |

**Three facts, each confirming a pre-registration:**
1. **The launch frame carries the entire sign.** Flipping L-2→L-1 (holding *everything else fixed*) flips 0.4568 (win) → 0.7705 (lose). I reproduce **both** original numbers to ≤0.02 by flipping this one knob.
2. **γ is not the carrier at n=1** — γ=0.0 and γ=0.5 give *identical* CLU error within each launch (0.7705/0.7705, 0.4568/0.4568). One Verlet step scales `p` by (1−γ) once; the sign is set before γ can act.
3. **Reproduces at shipped dt=0.125 too** (L-2/γ=0.5 wins n=1 at 0.3984; L-1 loses at 0.6076/1.2342). Not a legacy artifact.

### Mechanism (why L-2 "wins")
With the L-2 launch the CLU's n-step rollout only reaches cycle **E+n−1** while it is scored against cycle **E+n** — the launch is one frame stale, so at n=1 the CLU "prediction" is a *reconstruction of the last observed frame E* (it was handed E in the momentum `p0=(x_E−x_{E-1})/dt`) plus a correctly-signed velocity nudge toward E+1. That trivially beats a flat hold-at-E on smoothly-drifting degradation sensors. It is a **first-order extrapolator scored one cycle early**, not a demonstration that the learned Hamiltonian forecasts. Under the honest L-1 launch (rollout genuinely reaches E+n), the same first-order extrapolation **overshoots** the small true increments and loses.

### Which protocol is the defensible "does the rollout beat persistence"
**The fetch (L-1) protocol.** The persistence gate asks: *from information available at window end (through cycle E), does the CLU rollout predict the future (E+n) better than holding the last value?* The forecaster **must launch from the last observed state E** and its k-step output must be scored against E+k. That is exactly the fetch construction (launch L-1, aligned target, persistence = hold E). The audit's L-2 launch discards the newest frame from the state (keeping it only in the momentum), producing a one-cycle-stale "forecast" — defensible as a diagnostic of the *encode `q*` launch*, but **not** as a test of forecasting skill. **Verdict: CLU does NOT beat persistence** (0.7705 vs 0.5669 at n=1, and worse at every longer horizon); N86's qualitative conclusion stands.

*(Caveat carried forward, not in dispute here: at L-2/γ=0.5 CLU "wins" all horizons because γ=0.5 overdamps the state so `q*`≈launch — the "wins by not doing physics" pattern the fetch report already flagged. It is not a forecasting win.)*

### PROPOSED REGISTRY WORDING (R19-1 / N86; curator applies)
> **R19-1 RESOLVED (w23 `r19-r20-reconciliations`).** The two persistence gates differ by **one variable — the rollout launch frame.** Launched from the **last observed frame (L-1)** — the only construction that tests forecasting skill — the CLU rollout **does NOT beat persistence at any horizon, including n=1** (MSE 0.7705 vs 0.5669, legacy dt=0.05, 150ep, seed 42; reproduces at shipped dt=0.125). The audit's n=1 "win" (0.4568 vs 0.5669, reproduced) is an artifact of launching from **L-2**: the rollout is scored one cycle early, so at n=1 it reconstructs the last observed frame (handed to it via the momentum) rather than forecasting. **γ is not the carrier** (n=1 is γ-independent). **N86's conclusion — CLU's rollout does not beat persistence — is UPHELD; the quotable number is the L-1 forecasting number.** Issue (b) (Hub's raw-space-gate withdrawal) is separate and unaffected.

---

## R20-1 — the ballistic fraction (98.3% vs 79.7%)  ✅ RESOLVED

All rows: legacy dt=0.05, γ=0, 16 cycles, launch L-2, seed 42 (`r20_1.json`). "ballistic frac" = 1/(1+force/free).

| arm | epochs | windows | input | **ballistic (global)** | ballistic (per-sample) | mean M |
|---|---|---|---|---|---|---|
| **A** audit-repro | 40 | 2000 | z-scored | **94.75%** | 96.33% | 1.19 |
| **D** | 40 | 4000 | z-scored | 87.20% | 92.40% | 1.62 |
| **B** canonical | 150 | 4000 | z-scored | **76.97%** | 86.11% | 3.55 |
| **C** item3-style | 150 | 4000 | **raw** | 75.05% | 83.11% | 3.47 |
| **E** shipped `dt=0.125` | 150 | 4000 | z-scored | **50.49%** | 50.50% | 0.89 |

**The definitional difference is training maturity, not dt.** Both originals were at legacy dt=0.05, so dt is *not* the carrier. The carriers, in order:
- **Training epochs (40→150): −10pp** (87.2%→77.0% at matched 4000w). The dominant knob.
- **Window count (2000→4000 at 40ep): −7.5pp** (94.75%→87.2%).
- **Aggregation (global vs per-sample-norm): +9pp** per-sample always higher (dt-units used global-ish; audit used global).
- **Preprocessing (z-scored vs raw): ~−2pp** (B 76.97% vs C 75.05%) — minor, contra the dt-units guess that it was the likely cause.

**Reconciliation:** the audit's **98.3%** is the **under-trained regime** — my 40ep/2000w repro lands at 94.75% (global)/96.33% (per-sample); I reproduce the *regime* (under-trained → ~95–98% ballistic) but not the exact digit, because the audit's specific manual-z-score + seed subsample differ. The dt-units **79.7%** is the **fully-trained legacy** number — my canonical B reproduces it at **76.97%**. At shipped defaults it is **50.49%** (dt-units 50.6%, matched).

### PROPOSED REGISTRY WORDING (R20-1; curator applies)
> **R20-1 RESOLVED (w23).** The 98.3% vs 79.7% gap is **training maturity, not dt** (both legacy). Under one definition (fully-trained 150-epoch model, γ=0/16, launch L-2, global-norm): **~77–80% ballistic at legacy dt=0.05, ~50% at the shipped dt=0.125 default.** The 98.3% is **retired** as a 40-epoch/2000-window under-trained fit (independently reproduced as the ~95–98% under-trained regime); the fully-trained legacy figure ≈77% (dt-units' 79.7% confirmed). Carriers: epochs (−10pp), window count (−7pp), aggregation convention (±9pp), preprocessing (~2pp). **Quotable: "free-streaming-dominated — ≈50% at the shipped `dt`, ≈80% at legacy `dt`; the 98.3% was an under-trained-model artifact."**

---

## R19-2 — single-basin collapse at correct units  ✅ RESOLVED: **not reproduced — under-trained transient**

Reinstating config = conflated dt=1.0, launch L-2 (encode launch), float64-resolved, 512 anchors.

### 3-seed × γ re-verification at 150 epochs (`r19_2.json`)
100% finite everywhere; **f32 ≡ f64 to 6 digits** (fully resolved — not an overflow/rounding artifact):

| seed | γ=0.2 | γ=0.5 | γ=1.0 | mean\|q*\| | participation ratio |
|---|---|---|---|---|---|
| 42 | 0.308 | 0.370 | 0.375 | 1.17–1.29 | **1.0** (≈1-D ray) |
| 43 | 0.456 | 0.518 | 0.507 | 1.43–1.59 | 1.06 |
| 44 | 0.653 | 0.652 | 0.672 | 1.97–2.01 | 1.88 |

**No collapse.** Spread is 0.31–0.67 (seed-dependent), never near the audit's 0.0000; \|q*\|≈1.3–2.0 (not the audit's 0.17–0.32).

### Why the audit saw 0.0000 — the epoch sweep (dt=1.0, seed 42, γ=0.5, 64 steps; `r19_2_40ep.log`)
| epochs | q* spread (f64) | finite | mean\|q*\| |
|---|---|---|---|
| **10** | **9.15e-08** | 100% | 0.74 |
| 20 | 0.0872 | 100% | 1.14 |
| 40 | 0.1305 | 100% | 1.31 |
| 80 | 0.6202 | 100% | 1.90 |
| 150 | 0.31–0.67 | 100% | 1.3–2.0 |

**The collapse is a ~10-epoch transient of the barely-trained potential** (one trivial well → everything settles to ≈one point, spread 9e-8, small \|q*\|≈0.74 — exactly the audit's "spread 0.0000, \|q*\|≈0.2" signature). It **anneals monotonically away by 20 epochs** and is fully gone by the time the model is trained. It is **100% finite and f32≡f64**, so the audit was right that it is *not* overflow — but it is *also not* a property of the trained potential.

### Shipped encode path — directly answers dt-units follow-up 8
At shipped defaults (dt=0.125, γ=0.1, relax_steps=32 ⇒ budget 0.40): **spread 0.456** (γ=0.5: 0.837; γ=1.0: 0.998), 100% finite. **`q*` does NOT collapse on the shipped path.**

### Milder real structure (record, don't overclaim)
Even trained, the settled cloud is **low-rank** (participation ratio ≈1.0–1.9, often ≈1 ⇒ a near-1-D settled ray, not a filled cloud). So the honest statement is "`q*` settles onto a low-dimensional manifold," **not** "single basin / single point."

### Verdict + PROPOSED REGISTRY WORDING (R19-2; curator applies)
> **R19-2 RESOLVED (w23) — the reinstated single-basin collapse does NOT survive independent re-verification.** 3 seeds × γ∈{0.2,0.5,1.0} × horizons {64,256} at dt=1.0, 150 epochs: spread **0.31–0.67**, 100% finite, **f32≡f64** (resolved). An epoch sweep localizes the collapse to the **under-trained regime only**: spread **9.15e-08 at 10 epochs** (the audit's "0.0000", small \|q*\|) → 0.087 (20ep) → 0.13 (40ep) → 0.62 (80ep) → 0.3–0.7 (150ep). It is **neither overflow** (finite, resolved — the audit's retraction of the overflow explanation was correct) **nor a trained-model property** (it anneals away). The **shipped encode budget does not collapse** (spread 0.456). **DO NOT promote "the learned potential has one basin" as an N-candidate.** Milder real observation worth keeping: the *trained* `q*` cloud is low-rank (participation ratio ≈1 ⇒ near-1-D settled ray), consistent with `q*` ≈ a smoother of the last observation along the degradation direction.

---

## Limitations / confounds
- **Single-seed for R19-1 and R20-1** (seed 42; R19-2 used 3 seeds). Known FD001 seed spread on this path ≈0.002 (h-AUROC); the R19-1 sign margins (0.77 vs 0.57; 0.46 vs 0.57) and the R20-1 gaps (95%→77%→50%) are ≫ that, so the *verdicts* are seed-robust, but exact digits are single-seed.
- **R20-1 exact 98.3% not bit-reproduced** — I reproduce the under-trained *regime* (94.75%/96.33% at 40ep/2000w), not the audit's precise digit (its manual global z-score + seed subsample differ). The conclusion (under-trained artifact) is unaffected.
- **R19-2 residual gap:** the audit reported force/free=1.41 (strong potential) alongside its 0.0000; my 40-ep models have weaker force (spread 0.11–0.13, not 0.0000), and only ep≈10 hits 9e-8. I could not reproduce a *trained* model with both a strong potential *and* a point collapse. The epoch sweep makes the "under-trained transient" reading decisive regardless, but if the Hub wants certainty the decider is: re-fit the audit's *exact* `_SharedCLUFit` construction from `clu-latent-io-audit/ballistic.py` and log its per-epoch spread.
- All fits use FD001 train windows only; the anchor distribution is the CAFE window distribution (I did not test a synthetic init distribution — the FD001 windows are the production one).

## Recommended next experiments
1. **(cheap) R19-1 confirm at 3 seeds** — lock the sign margin; then the curator can quote the L-1 forecasting number without a single-seed caveat.
2. **(cheap) R19-2 exact-audit-model decider** — if the Hub wants the residual gap closed, log per-epoch spread of the audit's literal `ballistic.py` dt=1.0 model.
3. **(medium) learned-`ψ`/`φ` forecasting** — the only way any "CLU beats persistence" claim could become true is a learned read-in/read-out; today (identity `φ`, handcrafted `ψ`) it loses at L-1 in raw *and* feature space (audit already showed feature-space is 20–56× worse). Do not attempt to quote a persistence win until a learned map exists.

## Git footprint
No tracked code changed (analysis only). All artifacts under `.claude/scratch/r19-r20-reconciliations/` and `.claude/outputs/r19-r20-reconciliations/`. No branch, no commits. **No code defect found** to flag for the engineer — the harnesses behaved as written; the conflicts were protocol/training-maturity differences, not bugs.

---

## Proposed handover updates (for the Hub)
Fold into §1.6 (experiments) / §5 (provenance) / §8, and hand the three **PROPOSED REGISTRY WORDING** blocks to the next doc-curator pass (they own `negative_results.md`; I do not edit it):

1. **R19-1 → RESOLVED.** Sign carried by launch frame (L-2 vs L-1), γ-independent at n=1. Defensible = **L-1 launch ⇒ CLU LOSES persistence at all horizons** (0.7705 vs 0.5669 legacy; reproduces at shipped dt). **N86 conclusion UPHELD**; unblock N86 by quoting the L-1 number. Retire the audit's 0.4545 "win" as a stale-launch artifact. Issue (b) untouched.
2. **R20-1 → RESOLVED.** Ballistic fraction is **~50% at shipped dt, ~80% at legacy dt**; the **98.3% was a 40-epoch under-trained fit** (retire it). Carrier = training maturity, not dt. Update N63's "98.3% ballistic" (already quote-blocked) with the trained numbers.
3. **R19-2 → RESOLVED as NOT-reproduced.** Single-basin collapse is a ~10-epoch under-trained transient (spread 9e-8→0.3–0.7), 100% finite/f64-resolved (**not overflow**, per the audit) but **not a trained-model property** (**do not promote to an N-entry**). Shipped encode path does not collapse (spread 0.456) — closes dt-units follow-up 8. Keep only the milder observation: trained `q*` is low-rank (PR≈1, near-1-D ray).
4. **Cross-cut for §5:** all three conflicts dissolved into **training-maturity / launch-convention differences, not physics disagreements** — reinforces the standing rules (overflow-suspect for FP-exact order params; test the derivative/launch of everything). Two of three "reinstated/striking" numbers (98.3% ballistic, 0.0000 single-basin) were **under-trained-model artifacts**; recommend a standing caveat: *diagnostics on `<40`-epoch CAFE fits are not properties of the shipped model.*
