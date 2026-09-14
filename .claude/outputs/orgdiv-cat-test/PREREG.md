# PREREG — `orgdiv-cat-test` (C2W5, experiment-engineer)

**Filed 2026-08-01T10:23Z, BEFORE any harness that measures a number in this task was written or
run.** Base `main @ 29fc22b`, branch `agent/experiment-engineer/orgdiv-cat-test`, worktree
`../CHLU-orgdiv-cat-test`, main venv (`jax 0.9.0 / equinox 0.13.4 / numpy 2.4.1`, verified).

Governing document: `.claude/outputs/orgdiv-prereg/PREREG-TierII.md` (BINDING). This file registers
(i) the **design decisions the prereg left to the C2W5 lead** (its §8 open questions), (ii) my
**numeric predictions** for every K-verdict and falsifier, and (iii) **one structural prediction that
I believe kills the registered family as literally written** — registered here in advance precisely
so that, if it lands, it is a finding and not a post-hoc excuse.

⛔ Nothing below is a result.

---

## 0. Dial declaration (echoed from the task file)

- **Dial / pillar:** TIER ii — the organization dividend.
- **Control:** the ORGANIZER SWAP. The settle-deleted / matched-bytes launder is TIER i's control,
  run and reported here only as an *inherited diagnostic*.
- **Falsifies:** prereg §3's F1–F5 at their registered signs/thresholds/seed counts; K1–K5 as
  pre-conditions that can kill the family before any arm comparison.
- **Does NOT falsify:** a TIE (`|OD_min| ≤ 0.05`); losing to a table on SEEN queries; dividend ≈ 0 on
  the inherited tier-i launder.

---

## 1. ⭐ THE STRUCTURAL PREDICTION I REGISTER FIRST (because it is the one that can kill the family)

### SP-1 — "the additive-payload family is query-decodable at unbounded reader capacity"

**Claim.** With `y(x) = Σ_{j∈A(x)} v_j`, `K = 128` seen items and `N_a = 32` wells, the map
`1_{A(x)} ↦ y` is **linear with exactly `N_a = 32` unknowns and `K = 128` equations**. Therefore *any*
reader that (a) can identify `A(x)` from `φ(x)` and (b) has `≥ N_a` effective degrees of freedom,
**solves the whole family from the SEEN split with no store at all**, and generalises perfectly to
rule-4-valid unseen combinations. Prereg §2.5 control 2 (query-only reader ≤ chance + 0.05) would then
fail and the family would be VOID by K4 — *not because of a leak, but because the family's ground
truth is a 32-parameter linear code.*

**Registered prediction (numbers).** An **out-of-class**, deliberately over-capacity query-only probe
— ordinary least squares of `y` on the *true* indicator `1_{A(x)}`, 32 dof, fitted on the 128 seen
items — will reach **exact-set accuracy ≥ 0.95** on the 512 rule-4-valid unseen queries (I predict
**1.00 ± 0.00**, i.e. machine-precision recovery, because the system is exactly determined and
noiseless). Its residual `‖v̂ − v‖∞` will be **< 1e-6**.

**Registered consequence, decided in advance (this is the deviation I will argue in the report).**
The family is only well-posed if the **reader class is capacity-bounded below `N_a`**. I therefore
freeze, *before any arm runs*, a reader class in which **every member has fewer than `N_a = 32`
fitted parameters** (§3 below), and I register the 32-dof probe above as a **declared out-of-class
diagnostic** whose high score is the family's structural ceiling — **reported, never scored as a K4
failure, and never scored as an arm.** If SP-1's probe comes out *low* (< 0.95), my analysis is wrong
and I will say so.

**Corollary registered with it (SP-1b).** Because a capacity-bounded reader cannot memorise 32
payloads, the *store's* job is reduced to exactly one thing: **supplying `v_j` through the landscape's
payload channel**, so that the reader needs O(1) parameters. That is the only mechanism by which this
family can show a store effect at all, and I register it as the mechanism under test.

### SP-2 — ⭐⭐ "the address dimension `d` is squeezed from both sides, and the window may be EMPTY"

**Registered 2026-08-01T10:41Z, analytically derived, before any harness existed. This is the
sharpest thing I expect this wave to produce, so it is registered as a number before it is measured.**

Let `φ(x) = Σ_{j∈A(x)} e_j` with `e_j` iid unit vectors in `R^d` (the lossy set-code of §2.1).

**Upper squeeze (K4).** The map `1_A ↦ y` is linear with `N_a` dof; the map `1_A ↦ φ(x)` is linear
with rank `min(d, N_a)`. A *linear* query-only reader can therefore explain exactly the fraction
`R² = d/N_a` of `var(y)` (in expectation, for `v` drawn independently of `E`). With
`tol = 0.25·sd(y)` its exact-set accuracy is `2Φ(0.25/√(1−d/N_a)) − 1`. Registered values at
`N_a = 32`, chance `0.197`, K4 bar `0.247`:

| `d` | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| `R² = d/N_a` | 0.125 | 0.250 | 0.500 | 0.750 | 1.000 |
| **predicted query-only accuracy** | **0.211** | **0.227** | **0.276** | **0.383** | **1.000** |
| K4 verdict (bar 0.247) | PASS | PASS | ⛔ **FAIL** | ⛔ FAIL | ⛔ FAIL |

⇒ **registered: K4 fails for `d ≥ 16`, and `d ≤ 8` is forced.**

**Lower squeeze (the store).** For the fan-out to prefer members of `A`, the matched-filter margin is
`⟨e_j, φ(x)⟩ = 1 ± √(F−1)/√d` for `j ∈ A` versus `0 ± √F/√d` for `j ∉ A`. The per-well
discriminability is `Δ = 1 / (√F/√d) = √(d/F) = √(d)/2`. With `N_a − F = 28` distractors, top-`F`
recovery needs roughly `Δ ≳ √(2 ln(N_a−F)) = 2.58`, i.e. **`d ≳ 4Δ² = 26.7`**.
More carefully, exact 4-sparse recovery from `d` linear measurements needs
`d ≳ 2F·ln(N_a/F) = 2·4·ln 8 = 16.6`.

⇒ **registered: the store needs `d ≳ 17` (information-theoretic floor) to `d ≳ 27` (matched filter).**

### ⛔ SP-2's registered verdict: **the two constraints do not intersect.** `d ≤ 8` (K4) vs `d ≥ 17`
(recoverability). **I predict the registered cat-test family at `N_a = 32, F = 4, K = 128` is
INFEASIBLE at every `d`, for every organizer, and that this is a property of the FAMILY and not of
physics.** Specifically I register:

- **SP-2a:** across the registered sweep `d ∈ {4, 8, 16, 32}` there is **no `d`** at which
  (i) the in-class query-only reader is ≤ chance + 0.05 **and** (ii) the physics arm exceeds
  chance + 0.05 on unseen. Confidence **0.70**.
- **SP-2b:** at `d = 8` (the largest K4-passing `d`) the physics arm's unseen accuracy is
  **0.20–0.28**, i.e. within 0.08 of chance. Point prediction **0.24**.
- **SP-2c:** at `d = 32` the physics arm's unseen accuracy is **≥ 0.50** *and* the query-only reader
  is **≥ 0.90** — the family works and is simultaneously void.
- **SP-2d:** the *occupancy precision* (fraction of particles landing in a well of `A(x)`) rises
  monotonically with `d` and crosses 0.50 between `d = 16` and `d = 32`.

**If SP-2 lands, the honest wave verdict is a K4/F-grade KILL of the vehicle, not a tier-ii null**,
and the report must say the organizer swap was never reached rather than reporting `OD_min ≈ 0` as a
tier-ii finding. **If SP-2 is wrong** — if some `d` in the sweep passes K4 with a non-chance physics
arm — then the arms are scored there and `OD_min` is reported as the prereg intends.

⚠ **This supersedes nothing in `PREREG-TierII.md`**; it is a derivation *about* the registered family
that its §2.4 feasibility check (which tested only rule-4 combinatorics, never reader-identifiability)
did not perform. Both documents' predictions are scored.

---

## 2. Design decisions (prereg §8's open questions), registered before measurement

| # | prereg OQ | my decision | why |
|---|---|---|---|
| 1 | effective `s` of a learned multi-atom well | **fit `A e^{−r²/2s²}` to each written well's radial profile**, `r` sampled on random rays from the well's own settled minimum; the operating point is set on **measured** `d/s`, with the well spacing solved as `sep = (d/s)_target · s_measured` | prereg §7's blocking first-day instrument |
| 2 | does the write converge at `a ≈ 12` atoms/well | **K1 sweep over `a ∈ {4, 12, 32}`**, masked per-well write, localized atom init | prereg §3.4 / §8.2 |
| 3 | which reader class exactly | **frozen in §3 below, all < `N_a` params, all permutation-invariant over particles** | SP-1 forces the capacity bound |
| 4 | `bprime-c6`'s re-located `B` | **LANDED and used: `B ≥ 0.542` unrefuted, measured `s = 0.40` on the shipped store** (`bprime-c6` §1.1/§2). Soft certificate ON with `B = 0.542`. Declared. | task §1.4 |
| 5 | sum or mean | **sum** (registered primary) | prereg §2.2 |
| 6 | is `allocate` used | **NO.** Placement is admission + canonical placement (a pure function of the frozen query codes and shared policy params), so C4/C5's allocation-shuffle test is satisfied vacuously; the shuffle test is still run and reported | minimal surface |
| 7 | seeds/compute | **5 seeds**; my share of the 300-cell budget is the physics arm: 5 seeds × 1 arm × 4 readers × 3 γ = **60 cells** + the K-sweeps. Budget/stagger in §6 | prereg §8.7 |

### 2.1 The vehicle, fixed (⛔ frozen before the first arm)

- **Store:** `AtomDictionaryPotential` with `n_groups = N_a = 32` **shared wells**, `a` atoms per
  well, `dim = d + m`, confinement `α = 0.05`. Well `j`'s target is `z_j = (u_j, v_j)`.
  Masked write per well-group (`atom_write_mask_fn`) ⇒ C3-local by construction.
- **φ (FROZEN, identical on every arm):** frozen query codes `e_j ∈ R^d`, unit-norm, one per well,
  drawn from a fixed key that does **not** depend on the arm or the seed's store draw.
  `φ(x) = normalise(Σ_{j∈A(x)} e_j) · ρ` — ⭐ **a single, deliberately LOSSY set-code**: `d ≪ N_a`, so
  `A(x)` is *not* recoverable from `φ(x)` by a capacity-bounded reader. This is what discharges
  prereg rule 2.
- **Launch protocol (designed parameters, ledgered, identical on every arm):** `P = 4` designed
  offsets `o_p ∈ R^d`, `q0_p = (φ(x) + o_p + σ_q ξ_p, 0)`, `p0 = 0`. The occupancy vector is a
  per-read transient (F4), never state.
- **Read:** two-phase damped Verlet (`γ_address → γ_read`), **multi-particle**, `z ∈ R^{P×(d+m)}` =
  the P settled states. ⛔ Never a single-particle settled point (Theorem O1).
- **Payloads `v_j`:** drawn at **write** time from `N(0,1)`, existing only in the store.
- **Operating point:** `d/s ∈ [2.5, 2.9]` on **measured** `s`; depth heterogeneity ≥ 3×;
  `γ ∈ {0.02, 0.05, 0.2}` axis with claim cells at `γ ∈ [0.05, 0.1]`; `N_a=32, F=4, K=128, a=12`.
- **Metric:** exact-set accuracy on `Q_unseen`, `tol = 0.25 · sd(y_unseen)`.
  **Registered chance** = accuracy of the constant predictor `mean(y_seen)`; predicted **≈ 0.197**
  (`2Φ(0.25) − 1`), reported per split.

---

## 3. ⛔ THE FROZEN READER CLASS (fixed here, before any arm runs; sizes + fitting protocol)

All readers are **permutation-invariant over the P particles**, fitted by least squares / Adam on the
**SEEN split only**, with an internal held-out-from-seen validation split for any hyperparameter.
⭐ **Capacity bound (SP-1): every member has < `N_a = 32` fitted parameters.**

| id | reader | form | fitted params (at `d=4, m=1, P=4`) |
|---|---|---|---|
| **R1** | `sum_linear` | `ŷ = Σ_p wᵀz_p + b` | `d+m+1 = 6` |
| **R2** | `well_table` | assign `p → argmin_j ‖z_p[:d] − c_j‖`; `ŷ = a·Σ_p v̂_{j_p} + b` with `v̂_j` **read from the store's landscape** (not fitted) | **2** |
| **R3** | `knn` | k-NN over SEEN in `z`-space (sorted-canonicalised), IDW, `k ∈ {1,2,3,5,10}` selected on the seen-validation split | 0 fitted + 128 stored (nonparametric; declared) |
| **R4** | `mlp_small` | `ŷ = Σ_p g(z_p)`, `g: R^{d+m} → R`, one hidden layer of width **4**, tanh | `5·4+4 + 4+1 = 29` |

`OD_min ≡ min_{R∈{R1..R4}} OD(R)`.

---

## 4. ⭐ REGISTERED PREDICTIONS (the scorecard I will be graded against)

### 4.1 Instruments / operating point
| # | prediction |
|---|---|
| **I1** | the effective-`s` estimator, run on the **shipped** `bprime-c6` geometry as a cross-check, reproduces `s = 0.40 ± 0.06` (their two independent measurements: 0.3979 fit, 0.4006 `well_fits`) |
| **I2** | on the factored store with `a = 12` atoms/well, `atom_width` init 0.30 and localized init, fitted `s ∈ [0.28, 0.55]`; **point prediction 0.40** |
| **I3** | to land `d/s = 2.7` the required well spacing is `sep = 2.7·s ∈ [0.76, 1.49]`; **point prediction 1.08**. 32 wells at that spacing need an address ball of radius `R ∈ [1.4, 2.6]` at `d = 4`; **point prediction R = 1.9** |

### 4.2 K-verdicts
| # | prediction | bar |
|---|---|---|
| **K1** | **PASSES at `a = 12` and `a = 32`; FAILS at `a = 4`** (prereg's registered prediction, adopted). Point predictions at `a=12`: endpoint write loss **0.01–0.05**, `λ_min > 0` at **≥ 95 %** of wells, SC-6 capture radius ≥ σ_q at **≥ 90 %** | loss ≤ 0.05 · λ_min>0 ≥90 % · capture ≥90 % |
| **K2** | my independent rule-4 construction reproduces the prereg's count to **within 10 %** (predicted **23 193 ± 2 300** valid held-out at `N_a=32,F=4,K=128`), and **100 %** of the 512 sampled unseen queries pass both assertions (`\|A∩B\| ≤ 2 ∀ stored B`; `min_B ‖y(A)−y(B)‖ ≥ tol`) | 100 % |
| **K3** | nearest-item table scores **0.15 ± 0.10** — far under the 0.60 bar ⇒ **PASSES**. Reason: rule 4 forces ≥ 2 of 4 wells wrong, so its error is ≥ `√2·sd(v)` ≈ 1.41 vs `tol ≈ 0.5` | ≤ 0.60 |
| **K4** | blank store, permuted payloads and the address-leak probe all land **within ±0.03 of chance (0.197)** ⇒ PASS. **In-class query-only reader: 0.20–0.30**, i.e. I predict it **PASSES but only just** (bar = 0.247), and I register **P(K4 fails on the query-only leg) ≈ 0.35** as the single most likely death (N68) | ≤ chance + 0.05 |
| **K5** | the physics read beats the `K`-row nearest-stored-item table by **> 0.10 on ≥ 1 reader** ⇒ **PASSES**, *conditional on the physics arm scoring > 0.25 at all*. Point prediction of the margin: **+0.15**. ⚠ If the physics arm is at chance, K5 fails vacuously and that is the real verdict | > 0.10 |

### 4.3 Falsifiers
| # | prediction |
|---|---|
| **F1** | **TIE.** `\|OD_min\| ≤ 0.05` against every null I can run in-house (N3 static-geometric, N4 kNN). Point prediction `OD_min = 0.00 ± 0.04`. (Prereg prior: P(TIE)=0.55; I concur and register 0.60.) |
| **F2** | does **not** fire: I predict the per-reader `OD(R)` curve has the same sign at all four readers |
| **F3** | does **not** fire (K3 and K5 both pass — see above) |
| **F4** | does **not** fire: K1 passes at `a = 12`, ratio `[A(D+2)+d]/(d+m) = [3·6+4]/5 = 4.40` (corrected law, `A = N_at/K = 384/128 = 3`) ⚠ **note: the corrected byte law gives 4.40×, not the prereg §5.2 table's 5.00×** — I predict this discrepancy is real and is the `harness-debt` correction (`1.4A + 0.8 = 5.00` uses `D+2=7`; `[A(D+2)+d]/(d+m)` with `D=4` gives 4.40). **Registered in advance as a documentation conflict I expect to find.** |
| **F5** | **does NOT fire** at `d/s ∈ [2.5,2.9]`, `γ = 0.05`, depth ratio ≥ 3: the fitted static-geometric rule reproduces the physics assignment on **85–98 %** of held-out queries (point prediction **0.93**), i.e. below the 99 % bar. ⚠ I predict my measured irreducible disagreement (**0.07**) is **smaller** than the prereg toy's 0.193–0.203, because a learned multi-atom well is broader and less momentum-selective than a single-atom toy well. **FIRES at `γ = 0.2`** (agreement ≥ 0.99), the internal VQ-collapse control |

### 4.4 The physics arm's absolute score (registered so a null cannot be dressed up)
| # | prediction |
|---|---|
| **A1** | physics arm exact-set accuracy on unseen, `γ=0.05`, best reader: **0.35 ± 0.15**; worst reader: **0.25 ± 0.15**. Chance 0.197 |
| **A2** | the fraction of particles that settle into a well **belonging to `A(x)`** ("occupancy precision") on unseen queries: **0.45 ± 0.15** (chance `F/N_a = 0.125`) |
| **A3** | on **seen** queries the physics arm scores **0.15 higher** than on unseen (the generalisation gap) |
| **A4** | ⭐ **the risk I name in advance:** P(physics arm is statistically indistinguishable from chance on unseen, i.e. A1 ≤ 0.25) ≈ **0.40**. If that lands, the wave's verdict is *"the family is unbuildable at this weight class"*, reported as such and **not** as an organizer-swap null |

### 4.5 Riders
| # | prediction |
|---|---|
| **M11** | monitor #11's zero-depth guard is **caller-only**: `saddle_reach_threshold` guards `s <= 0` but computes `beta = D/(2·al·s·s)`, and `s·s` **underflows to 0.0 for `s < ~1.5e-154`** while `s > 0` — so the shipped function still raises `ZeroDivisionError`. I predict I can reproduce the crash in one line and that the fix is a denominator guard |
| **DEL** | deletion curve: exactness on the private fraction **= 1.0 by construction (byte equality)**; measured degradation on the shared fraction is **monotone increasing in `p`** with read error rising by **≥ 2×** from `p = 0.0045` to `p = 0.094` (the `A_tot = 32` anchors) |

---

## 5. What would make me say I was wrong

- SP-1's 32-dof probe scoring **< 0.95** ⇒ my structural analysis of the family is wrong.
- K1 passing at `a = 4` ⇒ the write is cheaper than both the prereg and I registered.
- F5 **firing** at `d/s ∈ [2.5,2.9]`, `γ = 0.05` ⇒ the physics organizer is a VQ even in the band the
  prereg derived as non-VQ, and tier ii has no vehicle at any operating point.
- `OD_min` clearing **+0.05** against N3/N4 ⇒ my TIE prediction is refuted and tier ii has a
  dividend in-house before the null arms even land.

## 6. Compute declaration (prereg §8.7), BEFORE the first run

| stage | cells | est. wall |
|---|---|---|
| effective-`s` + operating-point calibration | 3 `a`-values × 3 seeds | ~10 min |
| K1 write-admissibility sweep | 3 `a` × 3 seeds × 32 wells | ~25 min |
| K2 family construction (numpy only) | 5 seeds | < 1 min |
| K3/K4/K5 on the physics arm alone | 5 seeds × 4 readers × 6 controls | ~30 min |
| physics arm training + scoring | 5 seeds × 3 γ × 4 readers = 60 cells | ~90 min |
| deletion curve | 4 `p` anchors × 3 seeds | ~15 min |
**Stagger:** strictly sequential inside one worktree (worktree 1 of ≤ 3). No parallel JAX processes.
**Seeds:** 0–4, threaded explicitly through `jax.random.PRNGKey`; insertion order re-shuffled per seed
(prereg rule 1).

---

*Filed before the first line of `chlu/core/factored_store.py` was written.*
