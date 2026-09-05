# address-space-dimension-scaling — experiment-engineer report

**Task + acceptance criterion:** generalize the w19 2-D ring retrieval demo to d-dimensional address spaces and measure `K_max` vs `d` (with blank controls on every cell, the fitted growth law vs the packing bound, per-`d` selectivity/regime, and the γ dependence) — to decide whether the 8-item ceiling is a ring artifact or CLU's real capacity.

**Status: done.** All of `d ∈ {1,2,3,4,6,8,12,16}` measured; `d ≥ 12` is **censored at the compute cap** and reported as a lower bound, never as `K_max` (§5). Tests green (**428 passed** on the full suite, 37 of them new; the one failure was found, root-caused to a repo-wide test-isolation landmine, and fixed — §6).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary).**
> 1. **The "8-item ceiling" must not be quoted as CLU's capacity.** Capacity is **exponential in `d`** — fitted base **A=2.126** (codebook, R²=0.986) / **2.964** (addressing, R²=0.988), exactly `4·2^d` over d≥2: **1024 items at d=8**, 128× the ring ceiling. H0 (flat at ~8) is **decisively rejected**.
> 2. **The theorist's packing bound `(1+2R/w)^d` is WRONG as a law** — not merely loose. Its *base* is 14.25 vs a measured 2.0, and its **`w`-exponent is falsified**: measured `K_max ∝ w^(−β)` with **β = 0.96/1.28/1.65 at d = 2/3/4**, where the bound requires **β = d**. Task Item 2 says explicitly this is "more important than the scaling curve". The `clu-memory-architecture` §4 packing-bound line needs correcting.
> 3. **w19's "retrieval requires dissipation" (0.813 at γ=0) does NOT replicate here, and the replacement claim is sharper:** at γ=0 *identity* retrieval is fine (0.969–1.000, and it **improves** with d), but the *payload value* is wrong (abs err **0.570** vs 0.000 at γ=0.02). Dissipation is required for **value** retrieval, not for addressing.
> 4. **My own pre-registered w19-consistency check FAILED in float32 and PASSES in float64** (§7) — a precision caveat that must travel with the d=2 prefactor.

---

## Flag-provenance table

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/address-space-dimension-scaling` @ `20f5f2c`, base local `main` @ `089cc6e` |
| seed | `42` (single seed — see risks) |
| **float precision** | **float32** (JAX default). ⚠ **Load-bearing** — see §6/§7; float64 moves the d=2 cell |
| kinetic mode | `newtonian_learned`, inertia = identity (geometry varied, not mass) |
| potential | **hand-designed** `BallRegisterPotential`; **no training anywhere, no gradient descent at all** |
| γ / dt / steps | **0.02** / 0.05 / **1200** (steps deliberately not reduced: at 600 the d=2 K=16 cell flips 0.906→0.867) |
| landscape | R=1.0, **wall_margin=0.5**, w=0.15, b=1.0, **payload_kappa=0.1**, c_conf=10.0, site_seed=0, payload_seed=0 |
| queries | ≤32/item (budget `clip(8192//K, 4, 32)`), **`query_noise_mode="fixed_norm"`** (σ_norm=0.15 at every d), σ_p=0.05, `y(0)=p_y(0)=0` always |
| read | tail 25%, 8 subsample pts; **linear codebook read** (w19 verbatim) + nearest-centroid + decoder-free selectivity |
| criteria | acc ≥ 0.90 **AND** blank ≤ chance + 0.15 (blank vetoes) |
| JAX | 0.9.0, main venv reused (protocol §4; no worktree sync) |
| langevin_noise | **N/A** — deterministic Verlet, no Langevin, no temperature |

**Designed vs learned:** *everything* — landscape, site packing, addresses, payloads. **Nothing here is evidence of emergence** (N46 precedent).

**⚠ Provenance caveat (stated because the tree moved under me).** The sweeps launched ~10:35; commit `bfc0906` (authored by the Head live, not by me) landed 10:42 and changed **ladder-walk bookkeeping**, not per-cell physics. I therefore re-ran boundary cells against the **current** committed code: `d=2 K=16 → 0.906`, `d=3 K=32 → 0.996`, `d=4 K=64 → 1.000`, `d=4 K=128 → 0.769`, `d=6 K=256 → 1.000` — **exact agreement** with the sweep on all five. The one non-exact cell is `d=6 K=512` (**0.642** now vs 0.599 in the sweep, from the per-cell query-budget change); both are far below the 0.90 threshold, so **every `K_max` verdict is identical**. `K_max` is recomputed by me from per-cell pass/fail, so the reported numbers are valid for the code on the branch.

---

## 1 ⭐ THE DELIVERABLE — `K_max` vs `d`. Capacity is EXPONENTIAL.

Every cell below has a **passing blank control** (blank ≤ chance + 0.15); no cell is reported without one.

Numbers below are from the **complete shipped run** (`results/exp_dim_scaling_metrics.json`, seed 42, `k_cap=2048`, all of `d ∈ {1,2,3,4,6,8,12,16}`). It **reproduces my own sweep's codebook ladder exactly**.

| d | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| **`K_max` (codebook read, w19 verbatim)** | 4 | **16** | 32 | 64 | 256 | **1024** | ≥2048 *(cens.)* | ≥2048 *(cens.)* |
| **`K_max` (selectivity, decoder-free)** | 4 | 16 | 64 | 128 | 1024 | ≥2048 *(cens.)* | ≥2048 *(cens.)* | ≥2048 *(cens.)* |
| selectivity at `K_max` | 0.977 | 0.943 | 0.997 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| blank at `K_max` | 0.234 | 0.082 | 0.031 | 0.012 | 0.005 | 0.001 | 0.001 | 0.001 |

**Fitted growth — censored points excluded (they are lower bounds, not measurements):**

| criterion | base `A` | exp R² | poly α | poly R² | verdict | n |
|---|---|---|---|---|---|---|
| **codebook** | **2.126** | **0.9861** | 2.565 | 0.9600 | **exponential wins** | 6 |
| **selectivity** | **2.964** | **0.9882** | 3.009 | 0.9628 | **exponential wins** | 5 |

Both arms are exponential and **both bases sit inside my registered band `A ∈ [2.0, 3.5]`.** Over `d ≥ 2` the codebook arm is exactly `K_max = 4·2^d` (`log₂ K_max = 4,5,6,8,10` at `d = 2,3,4,6,8` — precisely `d+2`, R²=1.0000): **capacity doubles per address dimension.**

> ⚠ **RETRACTION of my own draft caveat.** An earlier version of this report stated that a polynomial `d^2.97` fits the *selectivity* arm marginally better than an exponential. **That was wrong, and it was wrong for exactly the reason the run now guards against: I included the censored d=8 point.** A cell pinned at the compute cap is a lower bound, and including it flattens the curve precisely where growth is fastest. With censored points excluded the exponential wins on that arm too (0.9882 vs 0.9628). The same artifact at larger scale drops the codebook base 2.13→1.51 and R² 0.986→0.844, **inverting the headline** — this is now pinned by `_fit_growth` and a regression test.
> I also mis-stated `K_max(selectivity, d=6)` as 512; the correct value is **1024** (my figure was parsed from the *codebook* ladder, which had already terminated at 512).

### Why there are two curves (and why quoting only the first understates capacity)
All K items are read back through **one scalar payload channel** whose codebook values crowd as `1/K`. That channel saturates *before the address space does*:

| | d=6, K=512 | d=6, K=1024 | d=8, K=2048 |
|---|---|---|---|
| codebook read | 0.642 | 0.066 | 0.286 |
| **selectivity (addressing)** | **1.000** | **0.993** | **1.000** |

The two curves separate by a factor 2–4 from d=3 upward (e.g. d=6: 256 vs 1024).

At d=8, K=2048, **every query still settles in its own well** (selectivity 1.000) while the scalar read is at 0.286. This is a **read-out resolution limit, not a capacity limit** — precisely the estimator artifact w19 flagged (one-hot at 0.484 with payload error 7e-4). I added the decoder-free criterion for this reason (commit `f152377`).

### `R` and `w` as MEASURED (task requirement)
Measured off the landscape, not assumed: `w_measured` (radius of maximal restoring force, probed at the payload equilibrium `y=s(x)`) recovers the shape parameter to <1% — **0.081/0.151/0.221/0.301/0.452** for w = 0.08/0.15/0.22/0.30/0.45. `R_wall` recovers `R + wall_margin`; sites occupy `R_sites ≈ 1.0`.

### The packing bound vs the measurement
With measured `R=1.0`, `w=0.151` ⇒ bound base `1 + 2R/w = 14.25`:

| d | measured | `(1+2R/w)^d` | **overestimate factor** |
|---|---|---|---|
| 2 | 16 | 2.03e2 | **12.7×** |
| 3 | 32 | 2.89e3 | 90× |
| 4 | 64 | 4.12e4 | 643× |
| 6 | 256 | 8.36e6 | 3.3e4× |
| 8 | 1024 | 1.70e9 | **1.7e6×** |

Both are exponential; **the bases differ (2.00 vs 14.25)**, so the gap compounds. My registered H2 (bound overestimates, gap grows with d, ≥5× at d=4 and ≥100× at d=8) is **confirmed, and by far more than registered**.

---

## 2 ⭐ ITEM 2 — geometry vs dimension: **the packing bound's `w`-scaling is FALSIFIED**

At fixed `d`, sweeping the basin width and fitting `K_max ∝ w^(−β)`:

| d | w=0.08 | 0.15 | 0.22 | 0.30 | 0.45 | **measured β** | **bound requires β** |
|---|---|---|---|---|---|---|---|
| 2 | 16 | 16 | 8 | 4 | 4 | **0.96** | 2 |
| 3 | 32 | 32 | 16 | 8 | 4 | **1.28** | 3 |
| 4 | 128 | 64 | 32 | 16 | 8 | **1.65** | 4 |

`K_max` **does** fall with `w` — so basin width is a real capacity knob and geometry is not irrelevant — but it falls **far more slowly than `(1+2R/w)^d` requires**. Measured β grows roughly as `≈0.4d`, not as `d`. Registered discriminator `K_max(0.15)/K_max(0.30)`: bound predicts 3.6 at d=2 and **47 at d=4**; measured **4.0 at both**.

⇒ **The task's stated packing bound is not the law.** It is a valid *upper* bound (never violated, §1) but its functional form is wrong in `w`, and per the task's own instruction this outranks the scaling curve in importance.

⚠ **My own registered alternative also failed.** I pre-registered a *plateau* for `w < c·σ_q/2 = 0.375` (query-noise-limited), i.e. ratio ≈1.0. Measured ratio is 4.0 — no plateau at d=4 (128→64 across w=0.08→0.15). **Neither the packing bound nor my registered plateau is correct.**

### The replacement law (measured, and it closes quantitatively)

Two measurements explain the intermediate `w`-dependence, and together they predict the headline base:

**(a) The resolution floor is a fixed multiple of the basin width, INDEPENDENT of `d`.** Taking `Δ_req` = achieved site separation at the last passing cell:

| regime | measurement |
|---|---|
| width-limited (`w ≥ 0.15`, 12 cells over d=2,3,4) | `Δ_req / w` = **3.12 ± 0.41** (range 2.53–3.86) |
| query-noise-limited (`w = 0.08`) | `Δ_req / σ` = **2.84, 3.58, 3.00** at d = 2, 3, 4 |

> **`Δ_req ≈ 3.1 · max(w, σ_query)`** — both constants ≈3, no `d` anywhere.

**(b) The achieved packing is NOT ideal.** Fitting `log Δ = −(1/d_eff)·log K` on the measured separation curve (ideal farthest-point packing in a `d`-ball would give `d_eff = d`):

| d | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| `d_eff` | 0.67 | 1.64 | 2.48 | 2.92 | 4.33 | 5.72 | 7.80 | 9.29 |
| `d_eff / d` | 0.67 | 0.82 | 0.83 | 0.73 | 0.72 | 0.72 | 0.65 | 0.58 |

**`d_eff ≈ 0.72–0.83 · d`** — points concentrate near the shell of a high-`d` ball, so packing is systematically less efficient than the volume argument assumes.

**Putting them together:** `K_max = (2R / Δ_req)^{d_eff}`. For the addressing arm (`Δ_req ≈ 0.43`, `d_eff/d ≈ 0.72–0.78`):

> predicted base `A = (2/0.43)^{0.72…0.78} = **3.0–3.3**  vs  **measured 2.964** — agreement within ~10%, and within 2% at the lower end.

So the `w`-exponent is `d_eff`(≈0.72d) attenuated further by the σ-floor still partially binding across the sweep — which is why `β ≈ 0.4d` rather than `d`. **This is the law that should replace `(1+2R/w)^d`.** The residual (β/d_eff ≈ 0.55) is the one factor not yet independently derived — flagged for the theorist.

---

## 3 ITEM 3 — capacity regimes: **NO death zone at any `d`** (designed structure reaches regime 1/3)

Selectivity is **monotone non-increasing in K at every `d`** (checked programmatically, not eyeballed); the death-zone detector (dip below 0.7 followed by a >0.15 recovery) fires **nowhere**.

| d | selectivity trajectory across the K ladder |
|---|---|
| 2 | 1.000 → 1.000 → 1.000 → 0.943 → **0.282** (monotone collapse, no recovery) |
| 3 | 1.000 ×4 → 0.997 → 0.922 → 0.253 |
| 4 | 1.000 ×6 → 0.993 → 0.779 |
| 6 | 1.000 ×8 → 1.000 → 0.992 → 0.799 |
| 8 | **1.000 at every K up to 2048** |

- **Regime 1 (barrier-protected, selectivity ≈1.00)** holds all the way to `K_max` at every `d`, and the higher the `d` the longer it holds — at d=8 selectivity is *exactly* 1.000 across the entire ladder.
- **Regime 2 (washboard death zone, selectivity ≈0.49) is ABSENT.** As pre-registered: it is a property of the **1-D ring washboard** (one periodic coordinate with residual azimuthal barriers), and an isotropic d-ball of Gaussian wells has no coherent residual force to produce it.
- **The three-regime picture is NOT falsified** — the standing claim was that *designed* structure escapes the death zone, and it does. Beyond `K_max` capacity degrades **smoothly and monotonically**, with no non-monotone recovery (no regime-3 continuum recovery either, because a d-ball has no compact periodic register direction to merge into).

---

## 4 ITEM 4 — dissipation: **w19's claim does NOT replicate; the corrected claim is sharper**

γ sweep at K=8 (blank passes on all 18 cells):

| γ | 0.000 | 0.002 | 0.005 | 0.010 | 0.020 | 0.050 |
|---|---|---|---|---|---|---|
| acc, d=2 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| acc, d=4 | 0.977 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| acc, d=8 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| selectivity (all d) | 0.992–1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **payload abs err** (d=8) | **5.70e-1** | 1.47e-1 | 2.17e-2 | 2.79e-4 | **1.76e-6** | 9.56e-4 |

**Three findings:**
1. **Identity retrieval does NOT require dissipation** here (0.969–1.000 at γ=0), contradicting w19's ring result of 0.813. The linear codebook read can classify an *unsettled, oscillating* payload because the oscillation is item-specific.
2. **Value retrieval DOES require dissipation**: payload error is **5.70e-1 at γ=0** vs **1.76e-6** at γ=0.02 — five orders of magnitude. The corrected statement is *"retrieval of the stored **value** requires dissipation; addressing does not."*
3. **Required γ is `d`-INDEPENDENT** (registered prediction (iii) ✅): on a payload-fidelity criterion (`err < 0.1`), **γ_min ≈ 0.005 at d = 2, 4 and 8** alike. It is set by settling time against rollout length, which contains no `d`.

⚠ **Registered prediction (ii) FALSIFIED, in the opposite direction.** I predicted the γ=0 degradation would get **worse** with d (`<0.6` at d=8). Measured it gets **better**: 0.969 → 0.977 → **1.000** at d = 2 → 4 → 8. Mechanism: site separation *grows* with d at fixed K (0.649 → 1.139 → 1.339 at K=8), so an unsettled orbit still remains nearest its own site. **Higher-dimensional address spaces are more robust to zero dissipation, not less.**

---

## 5 Censoring and what I did NOT measure (per prereg §7)

- **d=12 and d=16 are both CENSORED at the compute cap** (`k_cap=2048`): every cell passed, at acc 1.000 / sel 1.000, and the ladder simply ran out. `K_max ≥ 2048` at both — a **lower bound, not a measurement**. The fitted law predicts 1.7e4 (d=12) and 3.5e5 (d=16), far above any affordable cap, so **these dimensions are structurally uncensorable on this hardware** rather than merely under-run.
- **`K_max(selectivity)` is censored from d=8 upward** for the same reason.
- **Censored points are excluded from every fit** and drawn hollow with up-arrows in fig 1, so the plot cannot be misread as "capacity saturates at high d" — that flattening is the cap, not the physics. (Including them inverts the headline; see the retraction in §1.)
- **Compute truncation is reported as censoring, never as `K_max`** — the prereg §7 commitment, honoured.
- Cost scales **linearly in K at fixed query budget** (per-step work is O(K·d) per query), so each ladder rung costs about as much as all previous rungs combined — this is what makes the cap bind so hard.
- The `per_axis` query-noise arm was **built but not swept** (only `fixed_norm` is reported).

---

## 6 What I fixed / found while running (would otherwise have produced wrong conclusions)

1. **⭐ A repo-wide test-isolation landmine (pre-existing, not mine).** Six test modules — `test_goldstone`, `test_friction_field`, `test_lattice`, `test_kt`, `test_wormhole`, `test_twins` — call `jax.config.update("jax_enable_x64", True)` at **module level**. pytest imports every module before running any test, so **x64 is globally ON in a full-suite run and OFF when a file runs alone.** My marginal test passed standalone and failed in the full suite. Measured at d=2, K=16: `float32 → 0.906` vs `float64 → 0.594` (kappa=0.1). **Any test asserting on a numerically marginal cell is order-dependent today.** Fixed on my side with a save/restore `float32_dynamics` fixture; verified by deliberately leaking x64 (`pytest tests/test_twins.py tests/test_dim_scaling.py` → **44 passed**).
2. **Blank-arm `payload_r2` was a degenerate 0/0 reporting as `1.000`.** The blank codebook is all-zeros ⇒ constant regression target ⇒ `ss_tot=0` ⇒ `1 − 0/1e-12 = 1.000`. Not a leak, but a "blank R² = 1.000" in a results table is exactly the number a reviewer would read as one. Now reported as `None` + `payload_r2_undefined_constant_target`, with the meaningful channel-magnitude check alongside.
3. **Added the decoder-free selectivity criterion** (§1) after observing the scalar-channel saturation — without it I would have reported d≥6 capacity as ~2–4× lower than the addressing actually supports.
4. Inherited from the interrupted first thread and verified: `wall_margin` (boundary sites slingshotting queries), `payload_kappa` 1.0→0.1 (payload spring driving the address plane), `query_noise_mode="fixed_norm"` (per-axis σ silently degrading query precision as `√d` — this one would have **manufactured a fake capacity ceiling**).

---

## 7 PREREG scorecard (honest; `PREREG.md` was written before any harness ran)

| # | registered prediction | measured | verdict |
|---|---|---|---|
| H1 | exponential; `A ∈ [2.0, 3.5]`; log K linear in d, R² ≥ 0.95 | **A = 2.126** (codebook, R²=0.986) and **2.964** (addressing, R²=0.988); exactly `4·2^d` for d≥2 | ✅ **both arms inside the band**, exponential beats polynomial on both |
| H1 | point predictions 7.1 / 51 / 2.6e3 at d = 2 / 4 / 8 | 16 / 64 / 1024 | ◐ **form right, constants off by ≤2.5×** |
| H1 | **consistency: d=2 must land in 6–12 or "the harness is wrong"** | **16 (float32)** / **8 (float64)** | ❌ **FAILED as registered in float32.** See below. |
| H0 | `K_max` flat at ≈8 regardless of d | 1024 at d=8 | ✅ **REJECTED**, decisively (128×) |
| H2 | bound overestimates; gap grows with d; ≥5× at d=4, ≥100× at d=8 | 643× at d=4, **1.7e6×** at d=8 | ✅ confirmed, far exceeding the registered margins |
| Item2 | `K_max` does NOT follow `(1+2R/w)^d` | β = 0.96/1.28/1.65 vs required 2/3/4 | ✅ bound falsified |
| Item2 | **my** law: plateau for w<0.375, ratio ≈1.0 | ratio **4.0** at d=2 and d=4 | ❌ **my resolution-floor law FAILED too** |
| Item3 | no death zone at any d; selectivity monotone | monotone at every d; detector never fires | ✅ |
| Item4 (i) | γ=0 degradation persists at every d | **does not replicate at all** (0.969–1.000) | ❌ **falsified** — refined to value-vs-identity (§4) |
| Item4 (ii) | degradation gets **worse** with d (<0.6 at d=8) | gets **better**: 1.000 at d=8 | ❌ **falsified, opposite direction** |
| Item4 (iii) | required γ ≈ d-independent (0.005–0.01) | **γ_min ≈ 0.005 at d=2,4,8** | ✅ |

**On the failed consistency check — the one I committed to treat as "the harness is wrong".** In float32 d=2 gives 16, outside the registered 6–12. Two things are true and both must travel with the number:
- **In float64 the same cell gives exactly 8** — w19's ring ceiling, inside the registered window. Spot-checked under x64: d=2 K=16 → 0.594 (fails ⇒ `K_max`=8), while **d=3 K=32 → 0.996 and d=4 K=64 → 1.000 are unchanged**. So the precision sensitivity is confined to the d=2 boundary cell; **the growth law (base 2.0) is precision-robust.**
- I also registered the wrong comparison point: w19's ring is a **1-D address manifold**, whose analogue here is **d=1 (`K_max`=4)**, not d=2. A ring of circumference `2πR` vs an interval of length `2R` differs by π, giving ≈12.6 against w19's 8.

⇒ **Honest verdict: the check fails as literally registered.** The prefactor `4` in `4·2^d` carries a ±1-ladder-rung precision uncertainty at d=2; the **exponent does not**. I am not rescuing this by redefining the criterion after the fact — I am reporting the failure and the diagnosis.

---

## How I verified

- Runner: `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …` (main venv reused, protocol §4 w6 lesson; JAX **0.9.0**, no worktree sync).
- `python -m chlu.experiments.exp_dim_scaling --quick` → exit 0, full battery incl. figures.
- **Full suite at final HEAD (`5c9e164`): `428 passed`, 0 failed, in 447s** (37 of them new). An earlier pass was `424 passed, 1 failed` → root-caused to the x64 import leak → fixed → verified under a *deliberate* leak (`pytest tests/test_twins.py tests/test_dim_scaling.py` → **44 passed**). `ruff check chlu/ tests/` → **All checks passed**.
- Raw logs, harvested JSON and the summary figure: `.claude/outputs/address-space-dimension-scaling/` and `.claude/scratch/address-space-dimension-scaling/` (`full_run.log`, `item5.log`, `items24.json`, `harvested_results.json`, `dimscaling_summary.png`).
- **The full battery completed and wrote its own artifact.** `python -m chlu.experiments.exp_dim_scaling --seed 42` ran to completion (~3.5 h, all four items, all 8 dimensions) and produced `results/exp_dim_scaling_metrics.json`, copied verbatim to `.claude/outputs/address-space-dimension-scaling/` together with the three figures and `fullrun.log`. **Every number in this report is re-derived from that JSON**, not from log parsing. (An earlier draft of this section said the JSON was never written because a *previous, aborted* sweep had been killed during d=12; that no longer applies.)

## Git footprint

- **Branch** `agent/experiment-engineer/address-space-dimension-scaling`, base local `main` @ `089cc6e`. Rebase onto `main` = **no-op** (already up to date). Did **not** touch `origin/main` (§7.21). **Not pushed**, per protocol.
- **Worked in the pre-existing worktree** `../CHLU-dimscale` — mandatory here: the **main checkout has another agent's branch checked out** (`agent/experiment-engineer/dt-units-split`, with that agent's script running live, PID 99498). **No collision.**
- Commits (**9**, all verified present on the shared ref from the **main** repo per the w4 lesson; `main` itself untouched at `089cc6e`): `14a800e` (BallRegisterPotential), `1d2eba6` (experiment+config+CLI), `bba7dec` (tests), `f152377` (criterion split), `bfc0906` (landscape fixes), `20f5f2c` (float32 fixture), `e60509e` (exclude censored points from the fit + shipped full run), `7adac72`, `053c75e` (tidy-ups), `5c9e164` (record every load-bearing flag in the results config dump — the first run's JSON omitted `query_noise_mode`, `wall_margin` and the query budget, which had already propagated one wrong figure into a draft of this report).
  ⚠ **The branch was edited concurrently by the Head (Pratik Jawahar) while I worked** — `bfc0906`, `e60509e`, `7adac72`, `053c75e` are his, not mine; `e60509e` also carries the complete run's artifacts. I verified their content against my own measurements rather than assuming it (§Flag-provenance; the §1 retraction is a direct consequence of checking `e60509e` instead of trusting my earlier fit). **The branch is still moving — re-read `git log` before review.**
- Files: **+** `chlu/experiments/exp_dim_scaling.py`, `tests/test_dim_scaling.py`; **M** `chlu/core/memory_potentials.py`, `chlu/config.py`, `chlu/cli/experiment_cmd.py`. Did not touch `utils/plotting.py` (shared) — figures local, per the `exp_retrieval`/`exp_paid_access` precedent. `results/` deliberately **not** committed.
- ⚠ **Thread-recovery note.** This task's first thread **died mid-flight**. I found its worktree with uncommitted work, a completed `PREREG.md`, and an **orphaned measurement job (PPID=1) still burning CPU** whose stdout went to a dead parent (results unrecoverable). Per my liveness-check rule I confirmed the *agent* was dead (orphan, not a live spoke) before adopting the work, then killed the orphan. All of its implementation is reviewed, tested and committed above.

## Open questions / follow-ups / risks

- **Single seed (42), single geometry (ball + farthest-point packing), single landscape family.** The exponent 2.0 is one draw. Farthest-point sampling is a **best-case** packing, so `K_max` is an **upper envelope** of this design, not a typical draw.
- **d=12 AND d=16 both ran the full ladder and are CENSORED at `k_cap=2048`** (every cell passed at acc 1.000 / sel 1.000). Their capacity is a **lower bound (≥2048), never a measurement**, and both are excluded from every fit. The exponential is established on d=1–8 (a 256× range in `K_max`); the fitted law predicts 1.7e4 (d=12) and 3.5e5 (d=16), which is out of reach on this hardware since per-cell cost grows ∝ K.
- **β ≈ 0.4d is now largely explained** (§2 replacement law): a `d`-independent resolution floor `Δ_req ≈ 3.1·max(w, σ)` combined with a non-ideal achieved packing `d_eff ≈ 0.72–0.83·d`. That chain predicts the addressing base to within ~10% (3.0–3.3 vs measured 2.964). **One residual factor (β/d_eff ≈ 0.55) is not yet independently derived** — that is the remaining open theory question, and it is much narrower than "unexplained".
- **The scalar payload channel is the binding end-to-end constraint at d≥6.** A vector-valued payload (or several channels) is the obvious next lever and would likely move the codebook curve up toward the selectivity curve — **untested**.
- The capacity is of a **hand-designed** landscape. Whether *trained* structure reaches any of it is untouched here and remains the D3/N46 risk.
- Precision: reported numbers are float32; the d=2 prefactor cell moves under float64 (§7).

## Proposed handover updates (for the Hub)

1. **§6 ground truth — new headline:** capacity of the designed d-dimensional register is **`K_max = 4·2^d`** (R²=1.0000, d=2–8; 1024 items at d=8, blank controls passing). **The 8-item w19 ceiling is a ring artifact and is now retired as a capacity claim.** The associative-memory framing survives on capacity grounds.
2. **§7 — NEW known issue (test isolation, repo-wide):** six test modules enable `jax_enable_x64` at **module import**, so full-suite runs execute *all* later tests in float64 while standalone runs use float32. Numerically marginal tests are therefore **order-dependent** (measured flip: 0.906 → 0.594). Suggest a `conftest.py` fixture or moving those updates inside the tests. **This is a latent landmine for every future numeric test in this repo.**
3. **Correction to `clu-memory-architecture` §4:** the packing bound `(1+2R/w)^d` is an upper bound but **not the law** — its `w`-exponent is falsified (measured β=0.96/1.28/1.65 at d=2/3/4 vs required 2/3/4) and it overestimates absolute capacity by 12.7× (d=2) to 1.7e6× (d=8). **Replacement, measured (§2):** a `d`-independent resolution floor **`Δ_req ≈ 3.1·max(w, σ_query)`** (3.12±0.41 width-limited; ≈3.1σ noise-limited) combined with a non-ideal achieved packing **`d_eff ≈ 0.72–0.83·d`** (shell concentration), giving `K_max = (2R/Δ_req)^{d_eff}` — which predicts the measured addressing base 2.964 to within ~10%. Empirically `K_max = 4·2^d` over d=2–8. **Needs a curator/theorist owner.**
4. **Correction to the w19 dissipation claim:** *"retrieval requires dissipation"* → *"**value** retrieval requires dissipation; **identity** retrieval does not, and is more γ=0-robust at higher d"* (payload err 0.570 at γ=0 vs 0.000 at γ=0.02; γ_min ≈ 0.005 independent of d).
5. **The three-regime picture is CONFIRMED for designed structure, not falsified** — no death zone at any d; regime 1 holds to `K_max` and is *more* robust at higher d. Regime 3 (continuum recovery) does **not** appear in a d-ball, which has no periodic register direction — worth noting as a geometry-dependence of the regime taxonomy.
6. **New CLI/config surface:** `chlu exp-dim-scaling`, `ExperimentDimScalingConfig`. Defaults `wall_margin=0.5`, `payload_kappa=0.1`, `query_noise_mode="fixed_norm"`, `steps=1200` are **measurement-derived and load-bearing**, not arbitrary.
7. **Cheap high-value follow-up:** a **vector-valued payload** — the scalar read, not the address space, is what caps end-to-end capacity at d≥6 (d=8, K=2048: read 0.276, selectivity 1.000).
