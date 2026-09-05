# PREREG — `c2w8p3-capture-strong-phi` (C2W8 pass 3, THE SPINE)

**Filed 2026-08-09 by the spoke, BEFORE any real (non-synthetic) census cell runs.**
Only a synthetic-data smoke (`.claude/scratch/c2w8p3-capture-strong-phi/smoke.py`, toy
CIFAR-shaped arrays, `d = 4`, 1 seed) has been executed at filing time; **no CIFAR-10 cell, no
`d = 12` cell, and no encoder fit has been run.**

Base: worktree `../CHLU-c2w8p3c`, branch `c2w8p3-capture-strong-phi`, base **local `main @ 18b4205`**.
Binding: `PREREG-C2W8-PASS3.md` §4/§6/§8/§9 · `ERRATA-C2W8-PASS3.md` §1 (R1–R4) ·
charter ADDENDUM 10 §A29.6 / §A30.2 · both precondition JSONs.

---

## 0. What is fixed before measurement (no degrees of freedom left)

| item | value | why it is not a choice |
|---|---|---|
| substrate | **Split-CIFAR-10** (reduced protocol) | Head ruling **R1** |
| `d` + atom budget (**ONE joint dial**) | **`d = 12` → 32 768 atoms** | **R3 resolved by measurement**: wt2 measured the geometry-favoured `d = 16` **INERT** (median fitted depth 5.44e-7 at a fully honoured 131 072-atom budget) |
| store arm | arm A **CO-SCALED WIDTH**, `atom_width_frac_spacing = 1.5` | the **banked** census value; the shipped default 0.5 does **not** clear the pass-2 gate (`c2w8p3-gate-addr` recon. item 2). Kernel form (`wendland`) is a **DECLARED SECONDARY axis** (§A29.4(i)) |
| φ arms | `simclr` (primary, `enc_steps = 8000`), `randconv` (control), **`pca` at `d = 12`** (the INTERNAL weak-φ reference, **in this run**) | R1 forbids any pass-1/2 comparison |
| map | wt2's `PhiProjection(form="pca")`, fitted `task1_only`; `identity` on the `pca` arm (provably neutral) | R2 |
| launder | same-keys kNN-in-φ **in the SAME projected φ**, asserted bit-identical in code | R2(b) |
| regime | `task1_only` only | binding |
| seeds | 0, 1, 2 (paired: the stream is bit-identical across arms at a seed) | ≥ 3 seeds |
| label | `geometry_go = true` (wt2, 3/3 seeds, all three `d`) ⇒ **no NO-GO re-label applies** | R4 |

## 1. ⛔ A STRUCTURAL FACT ABOUT THE G-ADDR CUE, REGISTERED BEFORE THE NUMBERS

`gate_addr` draws `q = c_i + κ_q · spacing · ε` with **`κ_q = 1.0` dimensionless**, so
`cue_sigma / spacing_ref ≡ 1.0` **on every arm and every `d` by construction**. Two consequences I
register now so they cannot be re-discovered as findings later:

1. **A stronger φ cannot make the cue easier.** The cue is scale-free and spacing-relative, so
   wt2's measured σ_q/spacing improvement (0.334 → 0.210 at `d = 12`) **does not transport to A1**.
   Whatever A1 does at strong φ is the *store's* doing, not the encoder's — which is exactly the
   spine's question, but it also means **A1 is not the place strong φ was ever going to show up.**
2. **The cue gets HARDER with `d`.** The expected displacement is `κ_q · spacing · √d`, i.e.
   `2.83 × spacing` at `d = 8` and **`3.46 × spacing` at `d = 12`**. Any A1 at `d = 12` compared to
   wt1's `d = 8` MNIST rescore is therefore comparing across a **22 % harder cue** as well as across
   dataset and encoder. ⛔ I will not make that comparison; I record the ratio
   (`cue_displacement_over_codebook_spacing`) on every cell so nobody else does either.

## 2. Predictions (point + band). Filed before the first CIFAR cell.

### 2.1 The store — measured FIRST (Head ruling R1's attached risk)

| # | quantity | point | band | rationale |
|---|---|---|---|---|
| **S1** | `depth_raw_median` at `d = 12`, strong arm, **INERT?** | **NOT inert** | `> 1e-6` on ≥ 2/3 seeds | wt2 measured 0.0186 at `d = 12` **without** arm A's lever; the co-scaled width + site-local init is what made `d = 8` MNIST dig 0.61 |
| **S1b** | `depth_raw_median`, strong arm | **0.25** | 0.02 – 0.70 | between wt2's un-levered 0.019 and pass-2 MNIST arm A's 0.61 |
| **S2** | ≥ 1 of the 3 arms is INERT (`< 1e-6`) at `d = 12` | **no** | prior 0.25 | the `pca` arm has the worst geometry and is the candidate |

### 2.2 The completed gate

| # | quantity | point | band | rationale |
|---|---|---|---|---|
| **G1** | **A1 (correct-basin), best arm** — the Hub's **Q4** (band 0.15–0.45) | **0.32** | **0.10 – 0.60** | wt1 measured 0.42–0.52 at `d = 8` MNIST on the identical instrument; §1.2's 22 %-harder cue pulls down, arm A's lever is the same |
| **G2** | A1 ordering `simclr ≥ randconv > pca` | **holds** | prior **0.55** | wt2: the address-geometry gain is the **architecture's**, not the objective's (`randconv` matched or beat `simclr`), so I do **not** predict `simclr > randconv` |
| **G3** | A3a **launder rate** (1-NN over stored keys, same cue) | **0.93** | 0.80 – 1.00 | wt1 measured 0.898 at `d = 8`; in higher `d` an isotropic jitter about the true key makes 1-NN *more* reliable, not less |
| **G4** | G-CAP passes (majority `capture_radius > 0`), best arm | **passes** | prior 0.60 | arm A's lever fixed capture at `d = 8` (46/48); `d = 12` is untested |
| **G5** | G-DEC `decode` vs chance 0.0625 | **above chance beyond 2 SE** | prior 0.50 | pass-2 arm A cleared it on MNIST; CIFAR φ is untested with a learned `V_θ` |
| **G6** | G-DRIFT `ratio` (median site drift ÷ spacing), best arm | **0.02** | 0.001 – 0.30 | pass-2 arm A: 0.0071–0.0474. ⛔ **low is NOT good here — see §2.4** |
| **G7** | the COMPLETED gate (all four legs, 3/3 seeds) passes on **any** arm | **no** | prior **0.15** | wt1 measured arm A **failing** G-ADDR 3/3 on A3; nothing in this build was tuned to fix A3 |

### 2.3 ⭐ THE BRANCH — both pre-registered as reportable

Registered rule (computed by `daylight_verdict`, never argued):
> **(a) DAYLIGHT** iff a launder margin is **POSITIVE beyond 2 SE on 3 seeds** — cue (`A3a`,
> McNemar SE) or stream (`A3b`, pooled binomial SE). **(b) NO DAYLIGHT** otherwise.

| # | quantity | prediction | prior |
|---|---|---|---|
| **B1** | the branch | **(b) NO DAYLIGHT** | **0.88** (Hub Q6 = 0.70, Q5 = 0.15) |
| **B2** | `A3a` cue margin, best arm | **−0.30** | band −0.75 … +0.05 |
| **B3** | `A3b` stream margin, best arm | **−0.30** | band −0.70 … 0.00 |

⛔ **Branch (b) is a FINDING, not a shortfall.** Registering it here is what stops it being tuned
away. ⛔ Neither branch is a tier-ii verdict, a full-CLU verdict, or an arm-race adjudication;
**no paper number is produced.**

### 2.4 ⚠⚠ THE D2a DIAGNOSTIC — registered TWO-SIDED, never a target

`G-DRIFT → 0` ⇒ the settled point approaches a deterministic function of the stored key = **D2a =
table-expressible** (intervention §8.2 prohibits it as a target). The CIFAR arm already measured
settle = same-keys kNN to **±0.0007** at strong φ.

| # | quantity | point | band |
|---|---|---|---|
| **D1** | `d2a.agreement_rate` (settle resolves to the SAME item the launder does), best arm | **0.72** | 0.30 – 0.99 |
| **D2** | `median_settle_to_launder_key_over_spacing`, best arm | **0.35** | 0.01 – 1.5 |
| **D3** | the best-scoring cell is **also** the lowest-drift cell (`best_is_also_lowest_drift`) | **true** | prior 0.60 — ⛔ **if it fires it is the D2a signature, NOT a success**, and it is said prominently |

### 2.5 The byte ledger

| # | quantity | prediction |
|---|---|---|
| **L1** | `phi_param_floats` on the `simclr` and `randconv` arms | **225 536** each (bit-identical to the banked `encoder_price.json`) |
| **L2** | `map_param_floats` at `d = 12` | **3 328** (= 256 mean + 12·256 components) |
| **L3** | the φ term is the **SAME number** on the store row and the launder row | **true** — they read the same object |
| **L4** | `ratio_clu_over_knn_launder` (address-only, matched-**items**) | **> 100×**; the 1 253× stream-launder caveat travels regardless |

### 2.6 The price (registered so a cut is a declared cut, not a silent one)

| # | quantity | prediction |
|---|---|---|
| **P1** | wall per census cell at `d = 12`, 32 768 atoms | **2 550 s** (874 s measured at `d = 8`/8 192 atoms × 2.92, the `d=12 ÷ d=8` ratio of wt2's depth probe) |
| **P2** | `simclr` encoder fit, per seed | 1 300 – 2 500 s (wt2 measured 2489/1648/1316) |
| **P3** | total for 3 arms × 3 seeds | **≈ 8 h** |

⛔ **Declared abort/cut rule, registered before the first cell:** if the measured cell price exceeds
**3 × P1 (7 650 s)**, I **cut seeds before cutting a cell** and **declare the cut** in the report with
the arithmetic. ⛔ A cell that is cut is a **declared NOT-RUN**, never a null.

---

*Filed before the first CIFAR-10 cell. Corrections and post-hoc additions go in a dated `ERRATA.md`
in this directory; this file is not edited.*
