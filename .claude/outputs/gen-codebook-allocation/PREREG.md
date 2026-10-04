# PREREG — gen-codebook-allocation (D6): spend codewords where the distortion is

**Filed before the harness that measures any of it was written or run.**
Author: experiment-engineer. Base: `main @ 107fea9`. Branch:
`agent/experiment-engineer/gen-codebook-allocation`. Worktree `../CHLU-gen-alloc`.

## 0. DIAL DECLARATION (echoed verbatim from the task)
- **Dial:** codeword allocation across wells, **at CONSTANT TOTAL BYTES**.
- **Laundering control:** uniform `M = 64` at the same total bytes — the shipped arm,
  **re-fitted through the identical code path** (never transcribed).
- **Falsifies:** the scaling law's actionability. This is a **CONTROL ARM for the
  dynamical read, NOT a gate on it.**
- **Does NOT falsify:** losing to `knn_proto` (metric-native ceiling theorem);
  failing to rescue the high-`PR` wells (§0.5 consequence 2 says it cannot).

## 1. The rule, stated before it is applied
Per-well effective dimension `PR_k = (Σλ)²/Σλ²` is measured **label-free** from the
covariance of the well's own store-training features. High-rate quantisation theory
gives a per-well relative distortion

        ε_k(M_k) = M_k^(−2/PR_k)

and the store's byte cost is `Σ_k M_k·d_k·4`. Three allocations at one budget
`B = 64·D·4 = 524,288 B`:

| arm | rule | role |
|---|---|---|
| `uniform` | `M_k = 64` ∀k | ⭐ **the laundering control / shipped arm** |
| `equal_distortion` | `M_k = ε^(−PR_k/2)`, `ε` solved so `Σ M_k d_k ≤ 64·D` | ⭐ **the task's named rule (PRIMARY)** |
| `mse_optimal` | `argmin_M Σ_k v_k ε_k(M_k)` s.t. `Σ M_k d_k ≤ 64·D` (Lagrangian: `M_k ∝ (2v_k/(λ d_k PR_k))^{PR_k/(2+PR_k)}`) | **SECONDARY** — the allocation that actually minimises the read error |

`v_k` = the well's total variance `Σλ`. Both derived allocations use **only**
store-training features — ⛔ no factor labels anywhere.

### 1.1 ⛔ The byte constraint, and the one place I depart from the task's wording
The task's falsifier #1 is *"total bytes are not exactly equal ⇒ it is a bigger
store."* With integer `M_k` and unequal `d_k` (283…744) `Σ M_k d_k = 131,072`
**exactly** is a Diophantine problem with no guaranteed solution. **I therefore
pre-register the conservative form and assert it in code:**

> `bytes(allocated) ≤ bytes(uniform)`, **never `>`**, with the exact shortfall
> reported per seed. Predicted shortfall **≤ 0.2 %** (≤ 1,100 B of 524,288 B).

A *smaller* store cannot be "a bigger store", so the falsifier's intent is preserved
and the allocated arm is if anything handicapped. If any arm exceeds the uniform
arm's bytes by one byte the harness **raises**.

## 2. ⭐⭐ THE CENTRAL PREDICTION, and it is that the named rule LOSES
Computed from `gen-variance-equalisation`'s published per-seed `PR`, `d_k` and `Σλ`
(`.claude/scratch/gen-codebook-allocation/predict.py`, run **before** this file was
filed; output `predict.json`). ⚠ These are *predictions from a prior spoke's
numbers*, not measurements of mine.

**Equal-distortion allocation levels the easy wells UP.** Equalising `ε_k` at a fixed
byte budget does not reduce total error — it moves error **into** the cheap wells.
Predicted variance-weighted read error `Σ v_k ε_k / Σ v_k`:

| seed | `uniform` | `equal_distortion` | `mse_optimal` |
|---|---|---|---|
| 0 | 0.2622 | **0.4427** | 0.2611 |
| 1 | 0.2233 | **0.4278** | 0.2217 |
| 2 | 0.2743 | **0.4632** | 0.2707 |

**And the MSE-optimal allocation is ≈ uniform.** Predicted improvement over uniform:
**0.42 % / 0.72 % / 1.31 %** — i.e. ⭐ **`M = 64` everywhere is already within ~1 % of
the byte-constrained optimum.** That is the pre-registered headline: *the law is
real and re-allocation is not a lever.*

### 2.1 The predicted allocations (per seed, wells in store order)
| seed | `PR_k` | `d_k` | `equal_distortion` `M_k` | `mse_optimal` `M_k` |
|---|---|---|---|---|
| 0 | 3.12 · 3.46 · 12.55 · 10.50 · 4.07 | 303·305·573·501·366 | **4 · 4 · 160 · 70 · 5** | 51 · 64 · 72 · 53 · 77 |
| 1 | 5.62 · 4.01 · 3.37 · 3.86 · 12.00 | 397·300·307·300·744 | **11 · 6 · 4 · 5 · 164** | 83 · 62 · 59 · 78 · 51 |
| 2 | 3.46 · 3.85 · 8.45 · 13.64 · 5.64 | 283·283·390·616·476 | **4 · 4 · 25 · 186 · 9** | 58 · 54 · 68 · 46 · 93 |

Predicted bytes: `equal_distortion` 524,048 / 523,644 / 523,496 B;
`mse_optimal` 523,856 / 524,032 / 523,280 B. **All ≤ 524,288 B.**

### 2.2 ⛔ The channel width is NOT preserved by the named rule
`Σ_k log₂M_k`: uniform **30.00 bits** in every seed; `equal_distortion`
**19.77 / 17.72 / 19.35 bits**. ⭐ On **seed 1 that is 17.72 < log₂(480,000) = 18.87**
— the aggregate code becomes **INSUFFICIENT**, which it never was before. I predict
this is visible as a distinct-of-72 collapse on seed 1 specifically.

### 2.3 The one way the named rule could still be interesting
`scale`'s headline well is the **highest-`PR` well in all three seeds** (10.50 /
12.00 / 13.64), and `equal_distortion` gives that well **70 / 164 / 186** modes
against 64. So the rule is a **trade**: hues starved (4–11 modes), `scale`/`shape`/
`object_hue` fed. ⭐ **If TOTAL-RETAINED falls but `scale` rises, that is the
pre-registered TRADE reading and is a finding**, not a pass.

## 3. Numbered predictions (80 % intervals; 3-seed means unless stated)

**Allocation / bytes**
| id | quantity | predicted | 80 % interval |
|---|---|---|---|
| A1 | harness-measured `PR_k` vs veq's published `PR_k`, max rel. deviation | 0.08 | [0.00, 0.35] |
| A2 | `equal_distortion` byte shortfall vs uniform, worst seed | 792 B | [0, 1,100] |
| A3 | `mse_optimal` byte shortfall vs uniform, worst seed | 1,008 B | [0, 1,500] |
| A4 | any arm exceeds 524,288 B | **NO** | — |
| A5 | `equal_distortion` min `M_k` over all seeds | 4 | [2, 12] |
| A6 | `equal_distortion` max `M_k` over all seeds | 186 | [80, 400] |

**The law (acceptance 5 — "report the fit, not just the win")**
| id | quantity | predicted | 80 % interval |
|---|---|---|---|
| L1 | `uniform` measured variance-weighted relative read error, 3-seed mean | 0.278 | [0.24, 0.32] |
| L2 | `equal_distortion` measured variance-weighted relative read error | 0.455 | [0.36, 0.56] |
| L3 | `mse_optimal` measured variance-weighted relative read error | 0.276 | [0.24, 0.32] |
| L4 | ⭐ sign of `L3 − L1` (does the optimal allocation beat uniform?) | **negative but < 0.01 in magnitude** | [−0.02, +0.01] |
| L5 | OLS slope of `log ε_measured` on `log ε_predicted`, pooled over all wells × arms × seeds (45 points) | **0.85** | [0.55, 1.20] |
| L6 | Pearson `r` of the same pooled log-log fit | **0.93** | [0.75, 0.995] |
| L7 | median ratio `ε_measured / ε_predicted` per well (uniform arm) | **1.10** | [0.7, 1.8] |
| L8 | ⭐ does the law hold *per well* (not just variance-weighted)? my prior is that per-well it is **noisier than the aggregate** — fraction of the 45 points with ratio inside [0.5, 2.0] | 0.80 | [0.45, 1.00] |

**The ladder (`SPREAD`, retention, 3-seed mean)**
| id | factor | `uniform` | `equal_distortion` | `mse_optimal` |
|---|---|---|---|---|
| R1 | `floor_hue` | **0.997** [0.96,1.00] | **0.42** [0.15,0.75] | 0.99 [0.94,1.00] |
| R2 | `wall_hue` | **0.995** [0.96,1.00] | **0.45** [0.15,0.78] | 0.99 [0.94,1.00] |
| R3 | `object_hue` | **0.665** [0.58,0.75] | **0.52** [0.25,0.80] | 0.67 [0.58,0.76] |
| R4 | ⭐ `scale` | **0.224** [0.19,0.26] | ⭐ **0.32** [0.18,0.50] | 0.23 [0.18,0.28] |
| R5 | `shape` | **0.555** [0.48,0.63] | **0.52** [0.30,0.75] | 0.56 [0.47,0.64] |
| R6 | ⭐ `orientation` | **0.243** [0.18,0.31] | **0.28** [0.15,0.45] | 0.25 [0.18,0.32] |
| R7 | ⭐⭐ **TOTAL-RETAINED** | **3.68** [3.4,3.9] | ⛔ **2.51** [1.6,3.3] | **3.69** [3.4,3.9] |
| R8 | ⭐ `Δ TOTAL` (`mse_optimal` − `uniform`) | **+0.01** | [−0.20, +0.25] |

**Distinct configurations on the 72 held-out `CORNER_B` requests (acceptance 4)**
| id | quantity | predicted | 80 % interval |
|---|---|---|---|
| D1 | `uniform` distinct-of-72 | **9.0** | [6, 14] |
| D2 | `equal_distortion` distinct-of-72 | **7.0** | [2, 16] |
| D3 | `mse_optimal` distinct-of-72 | **9.0** | [5, 15] |
| D4 | feature-side distinct codes / 4,096 SPREAD, `uniform` | **3,638** | [3,300, 3,900] |
| D5 | same, `equal_distortion` | **2,600** | [1,200, 3,700] |

**Controls**
| id | quantity | predicted | 80 % interval |
|---|---|---|---|
| C1 | `shipped` reference row reproduces `uniform` ladder to within seed spread | **YES** | — |
| C2 | random-code control at **each arm's own** `M_k` widths — TOTAL-RETAINED | **< 0.35** every arm | [0, 0.6] |
| C3 | ⭐ one-hot probe width `Σ M_k`: uniform 320; `equal_distortion` 243/190/228 | the allocated arm's probe is **narrower**, so a gain cannot be probe capacity | — |
| C4 | `knn_proto` embedding bit-identical across arms | **YES**, max \|Δ\| = 0.0 | — |

## 4. ⭐ The pre-registered PASS, and the two readings that are not it
- ⭐ **PASS** = an allocated arm's **TOTAL-RETAINED rises above `uniform`'s beyond
  seed spread** *and* its **distinct-of-72 does not fall**, at `bytes ≤ uniform`.
- **TRADE reading** (not a pass): TOTAL falls but `scale`+`orientation` rise beyond
  seed spread. ⭐ Reported as a trade, ⛔ never promoted.
- **NULL reading** (not a pass): `mse_optimal` is within seed spread of `uniform` on
  every row ⇒ ⭐ **uniform `M` is already optimal and re-allocation is not a lever**
  — which is what I predict, and which is a first-class negative result for D6.

## 5. What would prove this spoke wrong (task §"What would prove this spoke wrong")
1. **Total bytes not ≤ the uniform arm's** ⇒ it is a bigger store, not an allocation.
   Asserted in code; the run raises.
2. **The measured distortions do not follow `M^(−2/PR)`** (L5/L6/L8 outside their
   intervals) ⇒ the law does not hold per well and §0.5 needs revisiting. ⭐ I am
   pre-registering that the **aggregate** law is solid (veq measured it) and that the
   **per-well** law is the untested part — L8 is the honest place for it to fail.

## 6. Fixed before measuring
seeds 0,1,2 · `shapes3d`/`recon` · grouping loaded from `gen-commitment-collapse`
(⛔ never re-derived) · store recipe = `gen-read-compose`'s verbatim
(`train_steps=1200`, `train_batch=128`, `train_lr=3e-3`, `commitment_beta=0.25`,
`tau 1.0 → 0.1`, `n_store_train=32768`, `PRNGKey(seed·10007 + 20260911)`) — ⛔
identical across arms · probe `WellProbe`, `probe_steps=3000`, `probe_batch=512`,
`probe_lr=3e-3`, `PRNGKey(20260914 + seed)`, encoder point fitted once and shared ·
rows = `exp_gen_attribution`'s draw verbatim · `dec_recipe="v1_capacity_schedule"`
with the per-seed `param_hash` asserted (⛔ never `"baseline"`) · `±` = population sd
(ddof = 0).

⛔ Nothing in this file is edited after filing. Corrections go in a numbered
`PREREG_ADDENDUM_n.md`, filed before the run it governs.
