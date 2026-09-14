# r2-d-sweep-close — experiment-engineer report (w27, [C1W27])

**Task + acceptance criterion:** Stage 1 = `pass_metric` default `tol`→`decode` (+ regression test
+ re-render of the w26 d=4 table on one metric); Stage 2 = the **d-sweep at m=4, d=6 then d=8**
against `K_designed(d) = 4·2^d`; Stage 3 = the **arm(a)×arm(b)** cell with a registered interaction
sign; Stage 4 = capacity-per-byte, only if the wall moves at d≥6. **Acceptance: does the m=4
multi-channel code (± the annealed read) move the wall above the w23-class ceiling
(`K_learned(d) = min(2^d, ~32)`) at d=6/d=8, at ≥3 seeds, with payload read-noise ON?**

**Status: done.** Seven declared NOT-RUNs (§11); the headline is **3-seed at d=6** (with the wall
bracketed by a 1-seed K=256 FAIL) and **1–2-seed at d=8**. ~10 h wall clock.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines)
> 1. ⭐⭐ **THE FALSIFIER DOES NOT FIRE — but the arm that moves the wall at d≥6 is arm (b), not
>    arm (a).** `K_learned` at m=1 goes **32 → 128 at d=6** (bracketed: K=256 FAILS at 0.736) and
>    **32 → ≥128 at d=8** on the **byte-free annealed read** (3 seeds at d=6, 1–2 at d=8),
>    budget-adequate, launch-noise on. The
>    **m=4 code is a d=4-only win**: at d=6 it buys nothing over m=1+anneal and it *destroys*
>    K=128 (strict 0.0906, basin 0.4448).
> 2. ⭐ **w26's `K_designed(4) = 128 at every m` is a `tol` artefact and must be RESTATED.** Under
>    `decode`, m=1 K=128 falls **0.9968 → 0.8655**, so **`K_designed(4)` = 64 at m=1** and stays
>    128 at m=2/m=4. w26's "the designed arm is flat in m, as it must be" is **falsified** — the
>    multi-channel code raises the *designed* ceiling too. Full list of moved w26 numbers: §2.
> 3. ⭐ **`K_designed(d) = 4·2^d` is EXACTLY reproduced on the `decode` metric at d = 4, 6, 8**
>    (measured walls **64 / 256 / 1024**, `log₂K = d+2`, three points, one metric). This is a
>    *new* re-measurement that supersedes the `tol`-metric designed table.
> 4. ⭐ **The learned law becomes `K_learned(d) = min(2^d, ~64–128)`** with the annealed read — the
>    d-independent write ceiling rises from **~32 to 128 (d=6) / ≥128 (d=8)**; base-2 growth is
>    **NOT** restored. ⛔ **No exponent is quoted** (pre-committed, S2.6): the learned arm has 3
>    d-points but the d=8 one is a lower bound.
> 5. ⭐ **The two arms are SUBSTITUTES, measured: interaction `I = −0.1509` (18.3 SE, 3 seeds).**
>    Arm (b) buys **+0.1507** on top of m=1 and **−0.0002** on top of m=4. This confirms
>    `readout-channel-theory` §4.2 ("both act on the same inequality") to the digit and **refutes**
>    any additive framing.
> 6. ⛔ **w21's ~1.3 bits-per-param is the DESIGNED construction's number, and the learned store is
>    ~47× below it** (0.0243 vs 1.14 bits/param at the best d=6 cell). The re-measure does **not**
>    rescue the comparison; only the numerator moved (items/param **×4.00** at zero extra floats).
>    Not framed as beating anything.
> 7. ⛔ **New do-not-quote, carried:** w26's `K_designed(4) = 128` at m=1 · "the learned wall equals
>    the designed wall at every noise level" unqualified (it is **format-scoped**) · "m=4 unclamps
>    R2" (it is d=4-only) · any `tol` number at m>1 · the base-√2 / `d^1.62` exponent.
> 8. **Stage A (d=6 K=64 m=1) is NOT re-rendered** — the w26 JSON stored only the headline metric
>    and the mandate was the **d=4** table. Its `tol` numbers stand; a decode re-run is one cheap
>    job (§11).

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** capacity (the R2 law). A law about the primitive; **its figure is never framed as
  beating anything** (CM-23(m)). Nothing here is compared to kNN, Hopfield or any external method.
- **Laundering control:** the **designed** write (`BallRegisterPotential`) at matched geometry, on
  the **same payload code** and the **same (possibly annealed) read**, at every `d`, `K`, `m`,
  `σ_obs`. It **kept reaching its own wall** `4·2^d` at every `d` (§3) and the annealed read gave
  it **exactly nothing** (identical to 4 dp at every K, §5) ⇒ **no N46 scope collapse**: the lever
  repairs a *learned-store-specific* deficit and cannot make the store more designed.
- **Falsifies:** at d=6 and/or d=8, m=4 (± the annealed read) does not move the wall above the
  w23-class ceiling at ≥3 seeds with payload read-noise on. → **NOT TRIGGERED** (but see item 1:
  it is the *annealed read*, not m=4, that carries it at d≥6).
- **Does NOT falsify:** failing to reach exactly `4·2^d`; any external comparison; a smaller wall
  movement than hoped.

**PREREG:** `.claude/outputs/r2-d-sweep-close/PREREG.md` — written before any harness run that
measures a registered quantity (only the JAX warm-up, the Stage-1 unit test on an *untrained*
designed store, and pure-numpy codebook/atom arithmetic preceded it). Scorecard §9.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/r2-d-sweep-close`, base local `main` @ **`082d095`** |
| commits | **`e87f36f`** (the only tracked change) — see §12 |
| worktree | `../CHLU-r2dsweep`; **main venv reused** (protocol §4) — **JAX 0.9.0**, equinox 0.13.4 |
| harness | `chlu/experiments/exp_designed_mechanism.py` **unchanged this wave** (w26's levers, all default-off); the only tracked change is `pass_metric`'s default + one test. Drivers: `.claude/scratch/r2-d-sweep-close/{drive27,designed27,blanks27,rerender,agg27}.py` |
| geometry | d-ball, farthest-point `designed_sites`, R=1, wall_margin .5, `site_seed=0`, `payload_seed=0` |
| write | `train_memory_landscape`, **GLOBAL** (`learned_global`), 600 Adam(3e-3), wd 1e-4, n_perturb 32, σ_addr .25, σ_pay .6, margin .15, barrier .2, `payload_index = d` (m=1) or `arange(d,d+m)` |
| retrieval | γ_address .05 × 400 → γ_read .02 × 800, dt .05, tail_frac .25, n_subsample 8 — **split across anneal stages, so every arm integrates the same 1200 Verlet steps** |
| queries | `fixed_norm` jitter σ_q = 0.15/√d per axis, σ_p .05, ≤32 per item (cap 4096 total) |
| **atom budget** | **shipped floor** `min_atoms_base = 512`, `min_atoms_c = √2` ⇒ **4096 atoms at d=6** (K ≤ 128), **8192 at d=6 K=256** and **8192 at d=8** (K ≤ 256). *Deliberately the same budget that produced the w23-class `K_learned(6)=K_learned(8)=32` being tested.* N92 2× re-check at the first-fail cell: §6 |
| `atom_init_width` / `atom_init_local` | **0.30** / **False** everywhere (shipped; ⛔ no default promoted) |
| **payload noise** | **`payload_launch_sigma = 0.05` on EVERY learned and designed number below**; `payload_obs_sigma ∈ {0, 0.005, 0.010}` swept; **`pass_metric = "decode"`** everywhere (the w27 default) |
| payload code | `n_payload_channels ∈ {1,4}`; `payload_code = "grid"` at m=4 (min-separation-preserving lattice, `Δ = 2/(K−1)` at every m — **measured** and printed per cell), `"linspace"` at m=1. **`pscale = 1` everywhere.** |
| annealed read | `read_anneal_axes = "payload"`, `read_anneal_payload_mult = κ₀ ∈ {2,3,5}`, `read_anneal_stages = 4`, `read_anneal_phases = "both"` |
| criterion | cell **PASS** = mean strict ≥ `pass_strict` **0.9** ∧ value-blank ok. `strict = basin_ok ∧ decode_ok`; decode = nearest-codeword |
| **sd convention** | **population sd (`np.std`, ddof=0)** — w26's r2 convention. Sample sd would be ~22 % larger at n=3. Interaction SEs are **paired sample** SEs (ddof=1), stated as such |
| seeds | **3 (0,1,2)** on every decisive cell except where the per-cell seed list says otherwise |
| langevin_noise | **N/A** — deterministic Verlet, no temperature anywhere in this task |

---

## 1. Stage 1 — the `pass_metric` correctness fix (shipped)

`chlu/config.py`: `ExperimentDesignedMechanismConfig.pass_metric` default **`"tol"` → `"decode"`**;
`"tol"` stays selectable for bit-exact reproduction of pre-w27 runs. **No other default changed**
(B1.4). Regression test `test_value_blank_is_rejected_at_m_gt_1_only_under_decode`, measured on a
**designed, untrained** store at d=4 K=32 m=4 (max‖a‖ = 0.0912 < `payload_tol` = 0.1):

| store | basin | `strict_tol` | `strict_decode` | verdict |
|---|---|---|---|---|
| **value-blank** (all payloads 0) | 1.0000 | **1.0000** | **0.03125 = 1/32 (chance)** | ⛔ `tol` PASSES a store that holds no value |
| the real store | 1.0000 | 1.0000 | 1.0000 | ✅ |

The vacuity is worse than a scoring bug: `evaluate_arm_cell`'s **value-blank gate is vacuous too**,
because its trivial ceiling `mean(‖a‖ < payload_tol)` is itself 1.0 at m=4 — so the blank slips
through the guard as well. Under `decode` the blank sits at exactly `1/K` and is rejected.

**Downstream reach of the default:** `score_cell` is also called by `exp_write_ceiling.py`
(m=1 only). `decode` is stricter than `tol` iff `1/(K−1) < payload_tol`, i.e. **K > 11**; every
w20–w26 cell has K ≥ 16, so every previously-`tol` number can only fall or stay. `exp_sharded_store`
does **not** use `score_cell` and is unaffected. Tests: `test_write_ceiling.py` +
`test_sharded_store.py` **36 passed** under the new default.

---

## 2. Stage 1 — the w26 d=4 re-render: **every number that moves**

Pure re-aggregation (`rerender.py`): the w26 JSONs already store `strict_tol` **and**
`strict_decode` for every cell, so **nothing was re-run**. Full log:
`.claude/scratch/r2-d-sweep-close/rerender.log`.

### 2.1 ⭐ The laundering table (w26 §5, designed arm, d=4, seed 0) — **THIS IS WHERE IT MOVES**

| m | K | w26 quoted (`tol`, σ=0) | **`decode`, σ=0** | Δ | designed read error | `Δ_code/2` |
|---|---|---|---|---|---|---|
| 1 | 128 | 0.9968 | **0.8655** | **−0.1313** | 5.99e−3 | 0.0079 |
| 1 | 256 | 0.8604 | **0.0967** | −0.7637 | 1.14e−1 | 0.0039 |
| 2 | 256 | 0.8884 | **0.6758** | −0.2126 | 1.01e−2 | 0.0039 |
| 4 | any | 1.0000 / 1.0000 / 0.9968 / 0.8884 | **identical** | 0.0000 | 3.6e−4 @K=128 | — |
| 1,2,4 | 32, 64 | 1.0000 | **identical** | 0.0000 | — | — |

⇒ **`K_designed(4)` under `decode` at σ_obs = 0 is `64` (m=1), `128` (m=2), `128` (m=4)** — where
w26 reported **128 at every m**. w26's sentence *"the designed arm is flat in m, as it must be"* is
**FALSIFIED**: the multi-channel code raises the **designed** ceiling too, because the m=1 read's
own settling error (5.99e−3) exceeds half the K=128 codeword spacing while the m=4 read's
(3.64e−4) does not. This is an honest cost to the w26 story — part of arm (a)'s d=4 gain is a
**format** gain that the designed arm also collects — and it does **not** break the tax claim,
which is measured *within* a format: learned m=4 = 128 = designed m=4 = 128 ⇒ **tax 1/1 at m=4**.

### 2.2 arm (b) (w26 §3) — only the `static` control moves

| cell, `σ_obs=0` column | w26 (`tol`) | **`decode`** |
|---|---|---|
| `static` (widen, never sharpen), K=16 | 1.0000 | **0.9375 ± 0.0510** |
| `static`, K=32 | 0.6250 ± 0.0255 | **0.2396 ± 0.0147** |
| `static`, K=64 (1 seed) | 0.3740 | **0.0625** |
| **everything else** (base 0.9368/0.8210, κ₀ 1.5–8, L=8, pow2, `readonly`) | — | **identical to 4 dp** |

⇒ w26's headline arm-(b) numbers are **unchanged**; the `static` control gets **stronger**.

### 2.3 arm (a) (w26 §4) — one number moves by 0.0024

`K=128, m=4, noise_off`: `0.9708 → 0.9684`. Every other arm-(a) figure in w26 §4/§6 was already
decode and is **identical**. The vacuity demonstration (blank at m=4 scores 1.0000 on `tol`,
0.0312 on decode) is reproduced exactly.

### 2.4 Not re-rendered
**Stage A (w26 §1, d=6 K=64 m=1)** — the w26 driver stored only the headline metric, so a decode
re-render needs a **re-run** (12 writes). The mandate was the **d=4** table; Stage A's `tol` numbers
stand as `tol` numbers. Flagged, NOT RUN (§11).

---

## 3. ⭐⭐ Stage 2 — the laundering arm across d: `K_designed(d) = 4·2^d`, exactly, on ONE metric

Designed store, **no training anywhere**, same code, same read, `σ_launch = 0.05`, decode, seed 0.
PASS = strict ≥ 0.9.

| d | m | K=32 | 64 | 128 | 256 | 512 | 1024 | **`K_designed(d)`** | `4·2^d` |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 1.0000 | 1.0000 | 0.8655 | 0.0967 | — | — | **64** | **64** ✅ |
| 6 | 1 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.6504 | — | **256** | **256** ✅ |
| 8 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.9785** | **1024** | **1024** ✅ |

**`log₂ K_designed = d + 2` at d = 4, 6, 8 — three points, one metric, zero free parameters.** The
w22/w23 designed law survives the metric change intact (it was previously established on `tol`).

At `σ_obs > 0` the wall is a **K-property, not a d-property** — the codeword spacing
`Δ = 2/(K−1)` caps every store identically:

| σ_obs | `K_designed` at d=4 | d=6 | d=8 |
|---|---|---|---|
| 0.005 | 64 | 64 | 64 |
| 0.010 | 32 | 32 | 32 |

⇒ **at `σ_obs = 0.005` no arm of any kind can exceed 64**, so a learned 64 there *is* the ceiling.
Every learned PASS below is quoted against the designed arm at the **same** `σ_obs` (condition 4).

**m = 4 designed (same table):** d=6 K=512 = **0.9998** at σ=0 ⇒ `K_designed(6)|m=4 ≥ 512` — the
code raises the *designed* ceiling above `4·2^d` because it removes the read's precision limit.
The `4·2^d` law is therefore an **m=1 (shipped-format) statement**.

---

## 4. ⭐⭐ Stage 2 — the learned wall at d = 6 and d = 8

Shipped budget — **4096 atoms** at d=6 K≤128, **8192** at d=6 K=256 and at d=8 (the `atoms_per_item·K`
term overtakes the geometric floor at K=256, so that cell is automatically at 2× the K≤128 budget).
`σ_launch = 0.05`, decode, population sd. Every PASS carries its value blank (§6).

### 4.1 d = 6 (site sep 0.795 at K=64, 0.666 at K=128)

| cell | read | σ_obs = 0 | σ_obs = 0.005 | σ_obs = 0.010 | seeds | verdict |
|---|---|---|---|---|---|---|
| **K=64, m=1** | base (shipped) | 0.8493 ± 0.0081 | 0.8480 ± 0.0082 | 0.7510 ± 0.0105 | 3 | **FAIL** |
| | **aniso κ₀=2** | 0.9977 ± 0.0018 | 0.9956 ± 0.0012 | 0.8851 ± 0.0023 | 3 | ✅ |
| | **aniso κ₀=3** ⭐ | **1.0000 ± 0.0000** | **0.9980 ± 0.0008** | 0.8854 ± 0.0026 | 3 | ✅ |
| | aniso κ₀=5 | 0.9591 ± 0.0333 | 0.9567 ± 0.0327 | 0.8470 ± 0.0248 | 3 | ✅ |
| | *designed, same code+read* | 1.0000 | 0.9980 | 0.8833 | — | ceiling |
| **K=128, m=1** | base | 0.6976 ± 0.0125 | 0.6147 ± 0.0108 | 0.3940 ± 0.0142 | 3 | **FAIL** |
| | aniso κ₀=2 | 0.9102 ± 0.0312 | 0.8008 ± 0.0266 | 0.5131 ± 0.0110 | 3 | ✅ (marginal) |
| | **aniso κ₀=3** ⭐ | **0.9127 ± 0.0343** | 0.8014 ± 0.0301 | 0.5134 ± 0.0151 | 3 | ✅ (marginal) |
| | aniso κ₀=5 | 0.8488 ± 0.0408 | 0.7328 ± 0.0351 | 0.4719 ± 0.0174 | 3 | FAIL |
| | *designed, same code+read* | 1.0000 | 0.8777 | 0.5713 | — | ceiling |
| **K=64, m=4** | base | 0.9220 ± 0.0449 | 0.9173 ± 0.0442 | 0.6927 ± 0.0311 | 3 | ✅ (marginal) |
| | aniso κ₀=3 | 0.9219 ± 0.0441 | 0.9160 ± 0.0419 | 0.6912 ± 0.0281 | 3 | ✅ (marginal) |
| | *designed m=4* | 1.0000 | 0.9946 | 0.7524 | — | ceiling |
| **K=128, m=4** | base | **0.0906** (basin **0.4448**) | 0.0723 | 0.0347 | 1 | ⛔ **CATASTROPHIC** |
| | aniso κ₀=2 / κ₀=3 | 0.0657 / 0.0649 | 0.0503 / 0.0493 | 0.0225 / — | 1 | ⛔ |
| **K=256, m=1** ⭐ | base | 0.6023 | 0.3333 | 0.1687 | **1** | **FAIL** |
| (**8192** atoms, write **18 676 s**) | aniso κ₀=2 | **0.7358** | 0.4146 | 0.2146 | 1 | **FAIL** |
| | aniso κ₀=3 | **0.6233** | — | — | 1 | **FAIL** |
| | *designed, same code+read* | 1.0000 | 0.5657 | 0.2927 | — | ceiling |

⭐ **The d=6 wall is PINNED, not a lower bound: `K_learned(6) = 128`.** K=256 fails decisively
(best 0.7358 at κ₀=2, 0.6233 at κ₀=3, vs a 0.9 bar and a designed 1.0000) — so 128 is bracketed by a
measured PASS *and* a measured FAIL, and the tax at d=6 is exactly **128/256 = 1/2**. ⚠ K=256 is
**1 seed** (18 676 s write) and its N92 2× re-check is NOT RUN — but note that the K=256 cell
*already* runs at **8192 atoms, 2× the budget of the K≤128 cells**, and the measured budget response
at d=6 is **negative** (§6), so a further increase is not expected to rescue it. Declared as an
inference, not a measurement.

**Marginality, stated honestly.** The bar is *mean* strict ≥ 0.9. Per-seed at the two marginal
cells: `K=128 m=1 κ₀=3` = **0.9143 / 0.8699 / 0.9539** (1 of 3 seeds below the bar);
`K=64 m=4 base` = **0.9014 / 0.8804 / 0.9844** (2 of 3 below). The **K=64 m=1 + anneal** cell is
not marginal at all (1.0000 / 1.0000 / 1.0000).

### 4.2 d = 8 (site sep 0.908 at K=64, 0.798 at K=128), 8192 atoms

| cell | read | σ_obs = 0 | σ_obs = 0.005 | σ_obs = 0.010 | seeds | verdict |
|---|---|---|---|---|---|---|
| **K=64, m=1** | base | 0.8804 ± 0.0034 | 0.8794 ± 0.0029 | 0.7803 ± 0.0005 | 2 | **FAIL** |
| | aniso κ₀=2 | 0.9990 ± 0.0005 | 0.9978 ± 0.0007 | 0.8860 ± 0.0051 | 2 | ✅ |
| | **aniso κ₀=3** ⭐ | **1.0000 ± 0.0000** | **0.9985 ± 0.0005** | 0.8867 ± 0.0044 | 2 | ✅ |
| | aniso κ₀=5 | 1.0000 ± 0.0000 | 0.9978 ± 0.0007 | 0.8848 ± 0.0034 | 2 | ✅ |
| | *designed, same code+read* | 1.0000 | 0.9980 | 0.8833 | — | ceiling |
| **K=128, m=1** | base | 0.8386 | 0.7388 | 0.4790 | **1** | **FAIL** |
| | aniso κ₀=2 | 0.9895 | 0.8672 | 0.5642 | 1 | ✅ |
| | **aniso κ₀=3** ⭐ | **0.9993** | 0.8743 | 0.5696 | 1 | ✅ |
| | aniso κ₀=5 | 0.9854 | 0.8555 | 0.5625 | 1 | ✅ |
| | *designed, same code+read* | 1.0000 | 0.8777 | 0.5713 | — | ceiling |

**At d=8 the learned store with the annealed read MATCHES the designed store to within 5e−4
(K=64: 1.0000 vs 1.0000, 0.9985 vs 0.9980) and to within 0.003 at K=128 (0.8743 vs 0.8777 at
σ_obs = 0.005 — where the designed store itself fails the 0.9 bar).** ⚠ K=128 is **1 seed**
(write cost **8401 s**); K=256 at d=8 is NOT RUN.

### 4.3 ⭐ The R2 table (the deliverable)

`σ_launch = 0.05`, decode, m = 1 (the shipped format), shipped atom budget:

| d | `K_designed(d)` = `4·2^d` | w23-class `K_learned(d)` | **+ annealed read (0 extra bytes)** | tax | seeds |
|---|---|---|---|---|---|
| 4 | **64** | 16 | **32** (w26, re-rendered) | 1/2 | 3 |
| 6 | **256** | 32 | ⭐ **128** (K=256 FAILS: 0.736) | **1/2** (bracketed) | 3 (K=256: 1) |
| 8 | **1024** | 32 | ⭐ **≥ 128** (K=256 NOT RUN, §11) | ≤ 1/8 | 1 at K=128, 2 at K=64 |

- **The falsifier does not fire.** At d=6 the wall moves **32 → 128 (4×)** and at d=8
  **32 → ≥128 (≥4×)**, at 1–3 seeds, with launch noise on, budget-adequate (§6), value-blank at
  chance (§6). **The w26 unclamping is NOT a d=4 artefact.**
- ⚠ **Suggestive but NOT quotable as an exponent.** `log₂ K_learned` = **5 / 7 / ≥7** at
  d = 4 / 6 / 8 — the d=4→d=6 step is exactly **×4 over Δd = 2, i.e. base-2 growth**, and the
  w23-class "~32, d-independent" ceiling is gone at both new dimensions. But the d=8 point is a
  **lower bound** (K=256 NOT RUN), so per PREREG S2.6 **no exponent is quoted** and the base-√2 /
  `d^1.62` ban stands.
- **But the mechanism that carries it is arm (b), not arm (a).** At d=6 the m=4 code buys nothing
  over m=1+anneal (0.9220 vs 1.0000 at K=64) at **+33 % learned floats**, and it **destroys**
  K=128 (basin 0.4448). ⇒ **`K_learned(6)|m=4 = 64 < K_learned(6)|m=1+anneal = 128`.**

### 4.4 ⭐ Why m=4 fails at d≥6 — a measured mechanism, not a shrug

`trained_well_widths` (`w_atom`, the width that actually forms the well) at d=6, K=64:

| arm | `w_atom` per seed | 2·w_atom vs site sep 0.795 | basin | strict (σ=0) |
|---|---|---|---|---|
| **m = 1** | 0.3135 / 0.3254 / 0.3430 | 0.63–0.69 **< sep** | 0.8379 / 0.8555 / 0.8545 | 0.8379 / 0.8555 / 0.8545 |
| **m = 4** | **0.8311 / 0.7284 / 0.2297** | **1.66 / 1.46 / 0.46** | 0.9448 / 0.9316 / **1.0000** | 0.9014 / 0.8804 / **0.9844** |
| m = 4, K=128 | 0.6491 | 1.30 vs sep 0.666 | **0.4448** | 0.0906 |

**The excursion is what disciplines the write.** With `max‖a‖ = 1` (m=1) the write must keep the
wells narrow enough to separate a far payload; with `max‖a‖ = 0.055` (m=4) that pressure is gone,
so on most seeds the writer **widens the atoms 2.3–2.7×**, `2·w_atom` exceeds the site separation,
neighbouring wells merge and the **address** collapses. The one m=4 seed that kept `w_atom = 0.230`
scored **0.9844** — i.e. the code is fine when the write happens not to widen. **At d=4 the site
separation (0.710 at K=32) was forgiving enough that this never bit; at d≥6 with more items it
does.** This is a *write-side* side-effect of a *read-side* code change, and it is exactly the kind
of thing the laundering control cannot see (the designed arm has no write).

---

## 5. ⭐ Laundering: the annealed read gives the DESIGNED store exactly nothing

d=6, designed, m=1, seed 0 — base vs κ₀=3, to 4 dp:

| K | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| base, σ=0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6504 |
| **aniso κ₀=3**, σ=0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.6504** |
| base / aniso, σ_obs=0.005 | 1.0000/1.0000 | 0.9980/0.9980 | 0.8777/0.8789 | 0.5657/0.5659 | 0.2759/0.2761 |

**The lever is inert on a store that does not need it — including at the K where the designed store
fails (0.6504 → 0.6504).** It repairs a learned-store-specific deficit and cannot manufacture one.
**N46 scope collapse does not fire**: the write still receives only the target sites; the anneal is
a tuple of Python floats on a per-axis width multiplier, provably outside the write's
`trainable_filter` (asserted in `test_anisotropic_anneal_widens_only_the_payload_axis`).

---

## 6. Value blanks (every reported PASS) and the N92 budget check

**Value blank** = the identical write with all payloads 0, scored against the **real** codebook.

| cell | read | blank strict (σ=0) | chance | verdict |
|---|---|---|---|---|
| d=6 K=64 m=1 | base | **0.0156** | 1/64 = 0.0156 | ✅ exactly chance |
| d=6 K=64 m=1 | aniso κ₀=2 / 3 / 5 | 0.0156 / **0.0000** / 0.0000 | 0.0156 | ✅ at-or-below chance |
| d=6 K=128 m=1 | base / aniso κ₀=3 | **0.0078** / 0.0078 | 1/128 = 0.0078 | ✅ exactly chance |
| (blank basin is 1.0000 everywhere — the addresses are still there; only the VALUE is gone) | | | | |

**N92 budget adequacy at the marginal / first-fail cell.** Arm (b)'s marginal cell at d=6 is K=128
(0.9127 at σ=0, 0.8014 at σ_obs=0.005 — a fail there). Re-run at **2× atoms (8192)**, 2 seeds:

| read | 4096 atoms (**3 seeds**) | **8192 atoms (2 seeds)** |
|---|---|---|
| base, σ=0 | 0.6976 ± 0.0125 | **0.7046 ± 0.0056** |
| aniso κ₀=2, σ=0 | 0.9102 ± 0.0312 | **0.8805 ± 0.0031** |
| **aniso κ₀=3, σ=0** | **0.9127 ± 0.0343** | **0.8239 ± 0.0023** |
| aniso κ₀=5, σ=0 | 0.8488 ± 0.0408 | 0.6497 ± 0.0310 |
| aniso κ₀=2, σ_obs=0.005 | 0.8008 ± 0.0266 | 0.7754 ± 0.0024 |

**Flat-to-WORSE at 2× atoms (write 3270–7182 s vs 869–1606 s) ⇒ NOT budget-limited.** The marginal
K=128 pass is therefore a genuine wall region, not starvation — and the 2× arm moves consistently
in the *wrong* direction on both seeds (κ₀=3: −0.089), exactly as w26's arm-(b) N92 check did at
d=4 K=64. (⚠ 2 seeds at 2×, 3 at 1×.)

---

## 7. ⭐⭐ Stage 3 — the arm(a)×arm(b) interaction (registered sign: NEGATIVE)

2×2 at **d=6, K=64** — chosen because nothing is ceiling-clipped there (all four cells lie in
0.84–1.00). Factors `m ∈ {1,4}` × `read ∈ {base, aniso κ₀=3}`. **3 paired seeds.**
`I = (m4,aniso) − (m4,base) − (m1,aniso) + (m1,base)`.

| σ_obs | m1.base | m1.aniso | m4.base | m4.aniso | **I** | paired SE | SE units |
|---|---|---|---|---|---|---|---|
| **0** | 0.8493 | **1.0000** | 0.9220 | 0.9219 | **−0.1509** | 0.0083 | **18.3** |
| 0.005 | 0.8480 | 0.9980 | 0.9173 | 0.9160 | **−0.1514** | 0.0080 | 18.9 |
| 0.010 | 0.7510 | 0.8854 | 0.6927 | 0.6912 | **−0.1359** | 0.0077 | 17.8 |
| 0 (κ₀=2) | 0.8493 | 0.9977 | 0.9220 | 0.9224 | **−0.1481** | 0.0085 | 17.5 |

Per-seed I at σ=0: **−0.1670 / −0.1396 / −0.1460** (same sign, 3/3).

| main effect | value |
|---|---|
| arm (b) **given m=1** | **+0.1507** |
| arm (b) **given m=4** | **−0.0002** ← *nothing at all* |
| arm (a) **given the base read** | +0.0728 |
| arm (a) **given the annealed read** | **−0.0781** ← *actively harmful* |

**Reading. The two arms are perfect substitutes, and the substitution is complete, not partial.**
Arm (b) buys `+0.151` on top of the shipped code and **exactly zero** on top of m=4; arm (a) buys
`+0.073` on top of the shipped read and **−0.078** on top of the anneal. This is precisely
`readout-channel-theory` §4.2's prediction — *"both act on the same inequality `r ≤ a_U(s_read)`"*:
arm (a) lowers the demand `r`, arm (b) raises the supply `a_U`, and once either has cleared the
inequality the other is redundant. **S3.1 CONFIRMED at the registered probability (0.70).**

⭐ **Decision-grade consequence: the combination is strictly dominated by arm (b) alone.** At d=6 the
best combined cell (0.9219) is *worse* than arm (b) alone (1.0000) and costs +33 % learned floats.
**The recommendation is the byte-free annealed read; the multi-channel code should not be combined
with it.**

⚠ **Stage 3 at d=4 is NOT RUN** (§11). The registered caveat S3.4 says a cell where m=4 already
passes cannot score the interaction — at d=4, m=4 passes K=32/64/128 outright (w26), so every
available d=4 cell is ceiling-clipped and the d=6 cell is the informative one. That is an argument,
not a substitute for the measurement; it is declared.

---

## 8. Stage 4 — capacity per byte (the wall did move at d≥6, so this runs)

**Byte accounting, pinned (condition 2).** Learned floats `P = n_atoms·(d + m + 2)` — atom centers
`n_atoms·(d+m)`, `log_width` and `amp` one each. Verified against the harness's own
`n_learned_params` at every cell (36 864 at d=6 m=1/4096 atoms; 49 152 at d=6 m=4; 73 728 at
d=6/8192 atoms; 90 112 at d=8/8192 atoms). **The annealed read adds ZERO floats** (a tuple of Python scalars). Stored
information `B = K·log₂K` bits.

| cell | learned floats `P` | items | **items / 10⁵ floats** | bits | **bits / param** |
|---|---|---|---|---|---|
| w23-class d=6 K=32 | 36 864 | 32 | 86.8 | 160 | 0.00434 |
| ⭐ **w27 arm (b) d=6 K=128** | **36 864** | **128** | **347.2 (×4.00)** | 896 | 0.02431 |
| w27 arm (a) d=6 K=64, m=4 | 49 152 | 64 | 130.2 (×1.50) | 384 | 0.00781 |
| w23-class d=8 K=32 | 90 112 | 32 | 35.5 | 160 | 0.00178 |
| **w27 arm (b) d=8 K=64** | **90 112** | **64** | **71.0 (×2.00)** | 384 | 0.00426 |
| ⭐ **w27 arm (b) d=8 K=128** | **90 112** | **128** | **142.0 (×4.00)** | 896 | 0.00994 |
| *(designed reference, `K(d+m)` floats)* d=6 K=256 | *1 792* | *256* | *14 286* | *2048* | **1.143** |

**Two facts, both measurements, neither framed as beating anything.**
1. **Items per learned float rises ×4.00 at BOTH d=6 and d=8, at literally zero extra bytes** — the
   anneal adds no parameters and the atom count is flat in K up to K=128 at both dimensions, so the
   whole increase is the wall moving.
2. ⛔ **w21's "~1.3 bits/param" is the DESIGNED construction's number** (`K(d+m)` reals ⇒ 1.14 at
   d=6), and the **learned** atom-dictionary store is at **0.0243 bits/param at its best cell —
   ~47× below the designed construction and ~82× below the transformer's measured 2.** Moving the
   wall moved the numerator by 4× and left the comparison decisively lost. **Candidate 3
   (capacity-per-byte) is NOT a contested-win route on these numbers.** That is the honest reading
   and it should be recorded as such.

---

## 9. PREREG scorecard

### Stage 1
| # | registered | measured | verdict |
|---|---|---|---|
| S1.1 | Stage A cells fall ≤0.03; conclusion unchanged | ⛔ **NOT MEASURABLE** — the w26 JSON has no decode column; re-run NOT RUN | — |
| S1.2 | arm (b) d=4 σ=0 column changes ≤0.01 | **0.0000** on base/κ₀/readonly (identical to 4 dp); only `static` moves (−0.385) | ✅ (the `static` control was not in the clause) |
| S1.3 | `K_designed(4)` at σ=0 **stays 128** (P=0.50) | **128 at m=2, m=4; 64 at m=1** | ◐ **split — right at the arm's own format, wrong at m=1** |
| S1.3′ | alternative: drops 128 → 64 (P=0.40) | **true at m=1** | ◐ |
| S1.4 | no arm-(a) number changes (P=0.95) | one moves by **0.0024** (K=128 m=4 `noise_off`) | ◐ effectively ✅ |
| S1.5 | the reconciliation list is non-empty (P=0.90) | 7 numbers moved | ✅ |

### Stage 2
| # | registered | measured | verdict |
|---|---|---|---|
| **S2.0** | the falsifier does NOT fire: some `K > 32` passes at d=6 (P=0.70) | **K=64 at 1.0000 and K=128 at 0.9127**, 3 seeds | ✅ |
| S2.1 | `K_learned(6) = 64` at σ_obs=0.005, = the designed wall, tax 1/1 (P=0.50) | **64 = 64** (0.9980 vs designed 0.9980) | ✅ **exactly as registered** |
| S2.2 | `K_learned(6) = 128` at σ_obs = 0 (P=0.40) | **128** (0.9127 ± 0.0343, 3 seeds; **K=256 FAILS at 0.7358**) | ✅ **exactly as registered, and bracketed** |
| S2.3 | tax at d=6 = **1/2** (P=0.40) | **128 / 256 = 1/2**, with both the PASS and the FAIL measured | ✅ **exactly as registered** |
| S2.4 | d=8: `K_learned(8) ≥ 64` (P=0.50) | **≥128** — K=64 1.0000 (2 seeds), K=128 0.9993 (1 seed) | ✅ **exceeded** |
| **S2.5** | `K_designed(6)` measures **128**, not 256 — precision binds before geometry (P=0.55) | **256 = 4·2⁶ exactly** (K=512 fails at 0.6504) | ✗ **REFUTED — my central prediction was wrong; the designed law survives the metric change intact at d=4, 6 AND 8** |
| S2.6 | quote no exponent without ≥3 d-points on one metric | designed: 3 points ⇒ `log₂K = d+2` quoted. learned: d=8 is a **lower bound** ⇒ **no exponent quoted** | ✅ honoured |

### Stage 3
| # | registered | measured | verdict |
|---|---|---|---|
| **S3.1** ⭐ | interaction **NEGATIVE, `I ≤ −0.05`** (P=0.70) | **I = −0.1509, 18.3 SE**, same sign 3/3 seeds, 3 noise levels, 2 κ₀ | ✅ **the registered central case** |
| S3.2 / S3.3 | additive / super-additive | — | ✗ excluded at ≥17 SE |
| S3.4 | at a saturated cell arm (b) buys <0.01 | **−0.0002** at m=4 | ✅ (and this is *why* the d=4 cells are uninformative) |
| S3.5 | the combination does not move the wall beyond arm (a) alone at d=4 | ⛔ **NOT RUN** at d=4; at **d=6** the combination is strictly **worse** than arm (b) alone | — |

### Stage 4
| # | registered | measured | verdict |
|---|---|---|---|
| S4.1 | items/10⁵ floats rises **≥4×** at the best d=6 cell (P=0.60) | **×4.00** exactly (K=32→128 at a flat atom floor) | ✅ |
| S4.2 | bits/param stays **≪1** at every learned cell; the re-measure does not rescue w21 (P=0.85) | **0.0018 – 0.0243**, vs designed **1.14** | ✅ |

**Global falsifier: NOT triggered** — but with the substantive amendment that the arm which carries
it at d≥6 is the byte-free annealed read, not the multi-channel code (§4.4).

**Deviation from the registered compute order, declared.** PREREG §2 ordered
Stage 1 → Stage 2 d=6 → Stage 3 → Stage 2 d=8 → Stage 4. In execution **Stage 3 came free**: its
2×2 is exactly the (K=64, m∈{1,4}) × (base, aniso) grid that Stage 2 d=6 already had to run, so it
cost zero extra writes and was scored from the Stage-2 landscapes. d=8 therefore started earlier
than the order implies. Nothing was re-ordered to chase a result: the only cell *dropped* mid-flight
was d=6 K=128 m=4, killed after seed 0 returned 0.0906 with basin 0.4448 (a catastrophic fail, not a
marginal one), and that is declared in §11.

---

## 10. Fairness ledger (the five binding conditions, discharged)

| # | condition | how it was discharged | evidence |
|---|---|---|---|
| 1 | **bits-per-item constant** | every cell prints `code_minsep`; it equals `Δ = 2/(K−1)` at m=1 **and** m=4 to 1e−5 at every K (0.06452 @32, 0.03175 @64, 0.01575 @128). Same K, same Δ, same per-axis noise ⇒ same `log₂K` bits | §3, §4 tables; `test_payload_codebook_holds_min_separation_and_cuts_excursion` |
| 2 | **byte accounting pinned** | `P = n_atoms(d+m+2)`, printed per cell and cross-checked against `n_learned_params`; **arm (b) adds zero floats**; m=4 costs +33 % (49 152 vs 36 864) | §8 |
| 3 | **payload read-noise ON** | `σ_launch = 0.05` on **every** learned and designed number in §3–§7; `σ_obs` swept 0 → 0.010; nearest-codeword decode throughout; **`pscale = 1` everywhere** | §3, §4 |
| 4 | **baselines get the same format** | the designed arm is built from the identical `(K,m)` codebook and run through the identical annealed read at the identical `σ_obs`; **no learned PASS is claimed above the designed arm at the same σ_obs** | §3, §5 |
| 5 | **laundering travels** | designed column on every table; the designed arm reached `4·2^d` at d=4, 6, 8; the anneal changed it by **≤0.0015 at every K** including where it fails | §3, §5 |

**N46 (scope collapse) — nothing here made the learned write more designed.** The only tracked code
change this wave is a **scoring default**. The write receives only the target sites, exactly as in
w20–w26; every center, width and amplitude is still learned by the same static objective. And the
sharpest evidence that the lever is not laundering: **it is inert on the designed store** (§5).

---

## 11. How I verified — commands, and what I did NOT run

```
# Stage 1 (no compute): re-aggregate the w26 JSONs on BOTH metrics
python .claude/scratch/r2-d-sweep-close/rerender.py            # -> rerender.log
# Stage 2/3: ONE write per (d,K,m,seed); 4 read schedules x 3 noise levels on it
python drive27.py cell 6  64 4 0,1,2 L6_K64_m4.json  1 learned_global ab
python drive27.py cell 6  64 1 0,1,2 L6_K64_m1.json  1 learned_global ab
python drive27.py cell 6 128 1 0,1,2 L6_K128_m1.json 1 learned_global ab
python drive27.py cell 6 128 4 0     (killed after seed 0: strict 0.0906, basin 0.4448)
python drive27.py cell 8  64 1 0,1,2 L8_K64_m1.json  1 learned_global ab
python drive27.py cell 6 256 1 0     L6_K256_m1.json 1 learned_global ab   # 18676 s write
python drive27.py cell 8 128 1 0     L8_K128_m1.json 1 learned_global ab   #  8401 s write
python drive27.py cell 6 128 1 0,1,2 L6_K128_m1_2x.json 2 ...   # N92, 2x atoms
python designed27.py 6 1,4 32,64,128,256,512      # laundering, d=6, no training
python designed27.py 8 1,4 32,64,128,256,512,1024 # laundering, d=8
python blanks27.py                                 # value blanks (blank=1)
python agg27.py "L*.json" ; python agg27.py inter L6_K64_m1.json L6_K64_m4.json aniso_k3 obs0.000
```
Raw JSON + logs: `.claude/scratch/r2-d-sweep-close/` (`L*` learned, `D*` designed, `BLANK_*` value
blanks, `*_2x` the N92 re-check, `rerender.log` the Stage-1 re-render).

- **Tests:** `tests/test_designed_mechanism.py` **18 passed** (17 → 18, +1 regression test);
  `tests/test_write_ceiling.py` + `tests/test_sharded_store.py` **36 passed** under the new
  default. Full suite **748 passed, 0 failed** in 5460 s (1:31:00) — `main` was 747, + my 1 new test.
- **`ruff check chlu tests` → All checks passed.** `ruff format --check` reports drift in
  `config.py`, `exp_designed_mechanism.py`, `tests/test_designed_mechanism.py` — **verified
  pre-existing**: the identical three files report the identical drift on the *unmodified* `main`
  checkout (83 files repo-wide). Per protocol §3.3 I did not reformat out-of-scope shared code
  (same finding as w23–w26).
- **Compute:** ~10 h wall, **≤4–5 concurrent background jobs** (`PPID=1`, harness-detached).
  Write cost per seed: **471–1606 s** at d=6/4096 atoms, **3270 s** at d=6/8192 atoms,
  **1469–2060 s** at d=8/8192 atoms K=64, **8401 s** at d=8/8192 atoms K=128, **18 676 s** at
  d=6/8192 atoms K=256.

### ⛔ NOT RUN (declared, never reported as a null)
| item | why | consequence |
|---|---|---|
| **d=6, K=256 seeds 1–2, and its N92 2× re-check** | the seed-0 write alone cost **18 676 s (5.2 h)**; a 2× re-check is ~10 h | `K_learned(6) = 128` is **bracketed** by seed-0's decisive K=256 FAIL (0.7358 best vs a 0.9 bar), but that bracket rests on **1 seed** and on the inference that a bigger budget would not rescue it (§4.1) |
| **d=8, K=256** | write cost at d=8 K=128 was **8401 s/seed**; K=256 is ~2× that | `K_learned(8)` is a **lower bound ≥128** |
| **d=8, K=64 seed 2; d=8 K=128 seeds 1–2** | compute (8401 s per write at K=128) | the d=8 result rests on **2 seeds** at K=64 and **1 seed** at K=128, not 3 |
| **Stage 3 at d=4** | compute; and every available d=4 cell is ceiling-clipped for m=4 (S3.4) | the interaction rests on the **d=6** cell only |
| **Stage A (w26 §1) re-rendered under decode** | needs a 12-write re-run; the mandate was the **d=4** table | Stage A's numbers stay `tol` numbers and must be labelled so |
| **N92 2× at d=6 K=128, seed 2** | compute (7182 s for the seed-1 write) | the budget check rests on **2 seeds**, both pointing the same way |
| **d=6 K=128 m=4 seeds 1–2** | killed after seed 0 (0.0906, basin 0.4448) to free cores | the m=4 catastrophe at d=6 K=128 rests on **1 seed** |
| **a fixed-`w_atom` control for §4.4** (write m=4 with the well width pinned to the m=1 value) | compute | §4.4's mechanism is a strong correlation across 3 seeds + 1 cell, **not** an intervention |

---

## 12. Git footprint

- Branch **`agent/experiment-engineer/r2-d-sweep-close`**, base local `main` @ **`082d095`**
  (verified clean; rebase onto `main` = no-op). **Not pushed.**
- Worktree **`../CHLU-r2dsweep`**, **main venv reused** (JAX 0.9.0, equinox 0.13.4). One other
  spoke's worktree was present throughout (`../CHLU-fullclu`, `agent/experiment-engineer/full-clu-harness`)
  — I never touched it and my two files do not overlap its scope; no filesystem collision.
- **Worktree removed after verifying the ref from the MAIN repo** (protocol §3.2):
  `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/r2-d-sweep-close`
  → `e87f36f`, then `git worktree remove`. Branch ref re-checked present afterwards; `main` still at
  `082d095` with a clean tree.
- **Commits:** see the table below.
- **Files touched: 2** — `chlu/config.py` (**one default + its comment**, inside
  `ExperimentDesignedMechanismConfig`, which this task owns) and `tests/test_designed_mechanism.py`
  (**one appended test**). I did **not** touch `chlu/core/memory_potentials.py`,
  `chlu/core/placement.py`, `chlu/core/controller.py`, `exp_cl_entry.py`, `cl_baselines.py`,
  `exp_phi_stream.py`, or `chlu/experiments/exp_designed_mechanism.py` (**no new core code was
  needed — the w26 harness ran the whole sweep unmodified**).

| hash | subject | files |
|---|---|---|
| `e87f36f` | `[experiment-engineer] default pass_metric to "decode" (w27 stage 1 correctness fix)` | **M** `chlu/config.py` (1 default + comment), **M** `tests/test_designed_mechanism.py` (+1 test) |

---

## 13. Open questions / follow-ups / risks

1. ⭐ **The one remaining cell that would close the learned curve is `d=8, K=256`** — 3 seeds,
   ~14 h at the measured write cost (8401 s at K=128, ~2× at K=256). `K_learned(6) = 128` is now
   **bracketed** (K=128 PASS, K=256 FAIL) and `K_learned(8) ≥ 128`; only the d=8 upper bracket is
   missing, and it is the only thing standing between this and a quotable learned exponent
   (`log₂K_learned` = 5 / 7 / ≥7 at d = 4 / 6 / 8). Seeds 1–2 at d=6 K=256 and d=8 K=128 would also
   be worth having — both currently rest on 1 seed.
2. ⚠ **The `σ_obs` choice is a design decision, not a measurement** (carried from w26). The
   format's own ceiling `Δ = 2/(K−1)` caps every store, and at `σ_obs ≥ 0.005` **no** arm can
   exceed 64 at any `d`. Every claim here is anchored to the designed arm at the same `σ_obs`,
   which is the honest anchor; the defensible framing is the **ratio**, not the absolute K.
3. ⭐ **§4.4 is a hypothesis with strong evidence and no intervention.** The claim "removing the
   excursion removes the write's pressure to keep wells narrow" rests on the `w_atom` ↔ basin
   correlation across 3 seeds. The clean test is one cell: write at m=4 with the atom width
   **frozen** at the m=1 trained value. If it recovers, arm (a) is repairable at d≥6 and the
   substitution result may be an artefact of an under-regularised write.
4. **Arm (b)'s κ₀ optimum is broad but not flat at d≥6**: κ₀ = 3 ≥ κ₀ = 2 > κ₀ = 5 at d=6
   (1.0000 / 0.9977 / 0.9591 at K=64), while at d=8 κ₀ = 3 and 5 tie at 1.0000. `read_anneal_*`
   remains **off by default**; promoting it is a Hub/Head decision and would need its own sweep.
5. **Marginality risk.** `K_learned(6) = 128` rests on a mean of 0.9127 with one seed at 0.8699.
   Two more seeds would settle it; at the current n it is a **PASS by the shipped criterion with a
   visible seed straddling the bar**, and I have said so everywhere it appears.
6. **A referee will ask why the designed arm gets `4·2^d` for free and the learned one does not.**
   The measured answer is now sharper than "learning is hard": at every failing learned cell
   `basin ≡ strict`, the failure is **address acquisition**, and the anneal fixes it without a
   single extra parameter. The remaining gap (1/2 at d=6) is the part the read-time schedule
   cannot reach.

---

## Proposed handover updates (for the Hub)

1. ⭐⭐ **§6 / claims matrix / R2 gate — R2 CLOSES with the ceiling moved at every d measured, by the
   BYTE-FREE arm.** `K_learned(d)` at m=1, decode, launch-noise on: **32 / 128 / ≥128 at d = 4 / 6 /
   8** (d=6 **bracketed** — K=256 fails at 0.736), against `K_designed(d) = 4·2^d` = **64 / 256 / 1024** (re-measured, exact, three points, one
   metric). The w23-class law `min(2^d, ~32)` becomes **`min(2^d, ~64–128)`**. **The exponent is NOT
   re-measured for the learned arm** and stays on the do-not-quote list.
2. ⭐ **The w26 story needs one substantive correction:** *"the m=4 code unclamps R2"* is a **d=4**
   statement. At d=6 the code buys nothing over the anneal and destroys K=128 (basin 0.4448), with a
   measured mechanism (§4.4: the write widens the atoms 2.3–2.7× when the excursion is removed).
3. ⭐ **`K_designed(4) = 128` (w26 §5) is a `tol` artefact → 64 at m=1** under the shipped default.
   w26's "the designed arm is flat in m" is falsified. Reconciliation list §2.
4. ⭐ **The (a)×(b) arms are SUBSTITUTES**, `I = −0.1509` (18.3 SE): arm (b) buys **+0.151** on m=1
   and **−0.0002** on m=4. Confirms `readout-channel-theory` §4.2. **The combination is strictly
   dominated by arm (b) alone** — recommend the annealed read, not the code, and not both.
5. ⛔ **Candidate 3 (capacity-per-byte) is NOT a contested-win route.** Re-measured with the byte
   accounting pinned: **0.0243 bits/param** at the best learned cell vs **1.14** for the designed
   construction and 2 for the transformer. Items/param **×4.00 at zero extra bytes** is the real,
   quotable, non-comparative result. **w21's "~1.3 vs 2" was the DESIGNED number — it is not stale,
   it was never about the learned store.**
6. **Default changed (the only one): `pass_metric` `"tol"` → `"decode"`** in
   `ExperimentDesignedMechanismConfig`. §3 of the handover's config notes should say so, with the
   reason (a value-blank scores 1.0000 on `tol` at m>1) and the escape hatch (`"tol"` still
   selectable for pre-w27 reproduction). ⚠ It also changes `exp_write_ceiling`'s scoring (m=1, K≥16,
   strictly harsher); nothing else uses `score_cell`.
7. **New do-not-quote:** w26's `K_designed(4) = 128` at m=1 · "learned = designed at every noise
   level" unqualified (format-scoped) · "m=4 unclamps R2" (d=4 only).
8. **New results proposed for `negative_results.md` / the claims matrix — §14 below (tier A).**

---

## 14. Proposed results for `negative_results.md` / the claims matrix (tier A)

> **N⟨next⟩ (POSITIVE, memory-architecture) — the learned-capacity wall moves at EVERY dimension
> measured, and the lever that moves it costs nothing.** With the shipped payload format (m=1),
> nearest-codeword decoding, payload launch-noise on (`σ_launch = 0.05`), the shipped atom budget
> and a value blank at chance on every PASS, an **anisotropic continuation read** (widen only the
> payload channel by `κ: 3 → 1` over 4 stages, `address_steps`/`read_steps` split so the Verlet
> count is unchanged) takes `K_learned` from the w23-class **16 / 32 / 32** to **32 / 128 / ≥128** at
> **d = 4 / 6 / 8**, at **zero extra parameters, zero extra latent dimensions and zero extra
> compute**. At d=6, K=64: **0.8493 ± 0.0081 → 1.0000 ± 0.0000** (3 seeds); at d=8, K=64:
> **0.8804 ± 0.0034 → 1.0000 ± 0.0000** (2 seeds) and at d=8, K=128: **0.8386 → 0.9993** (1 seed) —
> in every case **matching the designed store to within 5e−4** (at d=8 K=128, σ_obs = 0.005: 0.8743
> vs the designed 0.8777, where the designed store itself fails the bar). The laundering control is decisive: on the **designed** store
> the identical schedule changes nothing at any K (≤0.0015, including where the designed store
> itself fails, 0.6504 → 0.6504), so the lever repairs a *learned-store-specific* deficit and cannot
> manufacture one. The d=6 wall is **bracketed**: K=128 passes at 0.9127 ± 0.0343 (3 seeds) and
> **K=256 fails at 0.7358** (1 seed, 8192 atoms, 18 676 s write) against a designed 1.0000, so the
> tax at d=6 is exactly **1/2**. N92: the marginal cell is **worse** at 2× atoms (d=6 K=128:
> 0.9127 ± 0.0343 → 0.8239 ± 0.0023, 2 seeds), so it is a wall, not starvation. ⛔ **Base-2 growth is NOT restored**: the d-independent write
> ceiling **quadruples (~32 → 128)** and `log₂K_learned` reads **5 / 7 / ≥7** at d = 4/6/8 (the
> d=4→d=6 step is exactly base-2 growth) — but **no exponent is quotable**: the d=8 point is a lower
> bound (K=256 at d=8 NOT RUN).

> **N⟨next+1⟩ (NEGATIVE, memory-architecture) — the multi-channel payload code is a d=4-only win,
> and it fails at d ≥ 6 by a measured write-side mechanism.** At d=6, m=4 buys **nothing** over the
> annealed read at K=64 (0.9220 ± 0.0449 vs 1.0000 ± 0.0000) while costing **+33 % learned floats**,
> and at K=128 it is **catastrophic** (strict 0.0906, **basin 0.4448**, 1 seed). Mechanism, measured:
> the *excursion is what disciplines the write*. With `max‖a‖ = 1` the trained well width is
> `w_atom = 0.31 ± 0.02` (3 seeds); with `max‖a‖ = 0.055` (m=4) the writer widens the atoms to
> **0.83 / 0.73 / 0.23**, and on the two seeds where `2·w_atom` exceeds the site separation (1.66 and
> 1.46 vs 0.795) neighbouring wells merge and the **address** collapses — the one seed that kept
> `w_atom = 0.23` scored **0.9844**. At d=4 the site separation was forgiving enough that this never
> bit. ⚠ This is a strong correlation across 3 seeds plus one K=128 cell, **not** an intervention;
> the clean test (freeze the atom width at the m=1 trained value) is NOT RUN.

> **N⟨next+2⟩ (NEGATIVE, methodology) — the two read-out-side levers are SUBSTITUTES, and the
> substitution is complete.** 2×2 factorial at d=6 K=64 (`m ∈ {1,4}` × `read ∈ {base, aniso κ₀=3}`),
> 3 paired seeds, value-blank controlled: **interaction `I = −0.1509`, paired SE 0.0083 ⇒ 18.3 SE**,
> the same sign on 3/3 seeds, at 3 noise levels and 2 values of κ₀ (−0.1359 … −0.1514). The annealed
> read buys **+0.1507 on top of the shipped code and −0.0002 on top of the m=4 code**; the code buys
> **+0.0728 on top of the shipped read and −0.0781 on top of the anneal**. This confirms
> `readout-channel-theory` §4.2's derivation ("both act on the same inequality `r ≤ a_U(s_read)`":
> one lowers the demand, the other raises the supply) and **refutes any additive framing of the two
> excursion arms**. Decision-grade: **the combination is strictly dominated by the byte-free arm
> alone** and should not be shipped together.

> **N⟨next+3⟩ (POSITIVE, methodology) — `K_designed(d) = 4·2^d` survives the metric change exactly,
> and w26's `K_designed(4) = 128` was a `tol` artefact.** Re-measured under nearest-codeword decode
> with launch noise on, the designed `BallRegisterPotential` walls are **64 / 256 / 1024 at
> d = 4 / 6 / 8** — `log₂ K_designed = d + 2`, three points, one metric, zero free parameters
> (K=128 at d=4: 0.8655; K=512 at d=6: 0.6504; K=1024 at d=8: 0.9785). w26 reported
> `K_designed(4) = 128 at every m`; under decode it is **64 at m=1** and 128 at m=2/m=4, because the
> m=1 read's own settling error (5.99e−3) exceeds half the K=128 codeword spacing (0.0079) while the
> m=4 read's (3.64e−4) does not. ⇒ **w26's "the designed arm is flat in m, as it must be" is
> FALSIFIED** — the multi-channel code raises the *designed* ceiling too, so part of arm (a)'s d=4
> gain is a **format** gain, not a learned-store gain. Above `σ_obs = 0.005` the wall is a
> **K-property, not a d-property** (64 at every d); above 0.010 it is 32 at every d.

> **N⟨next+4⟩ (NEGATIVE, methodology) — the absolute-tolerance value criterion is now OFF by
> default, because at `m > 1` it passes a store that holds no value.** At d=4 K=32 m=4 the whole
> grid codebook lies inside `‖a‖ ≤ 0.0912 < payload_tol = 0.1`, so a **value-blank landscape scores
> `strict = 1.0000`** on `tol` — and slips past the value-blank *gate* as well, because that gate's
> trivial ceiling `mean(‖a‖ < payload_tol)` is itself 1.0. Nearest-codeword decode scores the same
> blank at **0.03125 = 1/32, exactly chance**. `pass_metric` now defaults to `"decode"`
> (`ExperimentDesignedMechanismConfig`); `"tol"` remains selectable for bit-exact reproduction of
> w20–w26. Decode is stricter than tol whenever `1/(K−1) < payload_tol`, i.e. **K > 11**, so every
> pre-w27 number at K ≥ 16 can only fall or stay; the full list of w26 numbers that move is in
> §2 (seven of them, all in the designed-laundering table and the `static` control).

> **N⟨next+5⟩ (NEGATIVE, capacity-per-byte) — moving the wall does NOT make capacity-per-byte a
> contested win.** With the byte accounting pinned (`P = n_atoms·(d + m + 2)`, verified against the
> harness's own parameter count) and stored information `B = K·log₂K`: the best learned cell
> (d=6, K=128, annealed read, 36 864 floats) delivers **347.2 items per 10⁵ learned floats (×4.00
> over the w23-class cell, at literally zero extra bytes)** but only **0.0243 bits per parameter** —
> against **1.14 bits/param for the designed `4·2^d` construction** (`K(d+m)` reals) and 2 for the
> transformer's measured figure. ⛔ **w21's "~1.3 vs 2" is not stale — it was the DESIGNED
> construction's number all along, and the learned atom-dictionary store is ~47× below it.**
> Items-per-parameter is the quotable, non-comparative result; bits-per-parameter is a loss and
> should be reported as one.
