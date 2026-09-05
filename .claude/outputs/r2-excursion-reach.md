# r2-excursion-reach — experiment-engineer report (w26)

**Task + acceptance criterion:** Stage A = the init×width 2×2 factorial with the interaction
term (3 seeds, d=6 K=64, monolithic) — *measure, do not promote*; Stage B = both excursion arms
((a) multi-channel payload, (b) annealed/continuation read) under the **five binding fairness
conditions**, the crux being **payload read-noise ON**. Acceptance: does either arm move the
`K_learned` wall at ≥3 seeds, budget-adequate, with noise ON?

**Status: done. ⭐⭐ BOTH ARMS MOVE THE WALL, and they survive payload read-noise while the w25
manipulation they replace does not.** `K_learned(4)` goes **16 → 32** on arm (b) at **zero extra
bytes, zero extra dimensions and zero extra compute** (3 seeds, ~110 SE), and **16 → 128** on
arm (a) — `K_designed(4)` itself. At **every** read-noise level measured the learned wall now
**equals** the designed wall (128 = 128, 64 = 64, 32 = 32), i.e. **the learned/designed prefactor
tax at d = 4 goes from 1/8 to 1/1.** ⚠ **d = 4 only** — d = 6 and d = 8 are NOT RUN, so no
exponent is re-measured. Stage A read out cleanly and **refutes my own registered prediction**:
the two levers are **not additive** (interaction +0.459, 14.3 SE) — but `atom_init_local`'s main
effect at the shipped width is **+0.007 ± 0.016, a null**, so the factorial does **not** license
promoting it (Head B1.4 respected; no default changed).

> **⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **R2 is UNCLAMPED at d=4, with a named mechanism and payload noise ON.** `K_learned(4)`
>    16 → 32 (arm b, 3 seeds) / 16 → 128 (arm a, 2 seeds at K=128, 3 at K=64).
>    **d = 6 / d = 8 NOT RUN.** The "⛔ never quote `K_learned` at pscale ≠ 1" ban stands and
>    is *reinforced* by §4 (I measured the pscale free lunch dying under noise); the new numbers
>    are at **pscale = 1** and carry the noise condition, so they are quotable **with the
>    σ_obs and the designed-arm reference stated in the same sentence**.
> 2. **w21's bits-per-param (~1.3 vs 2) is STALE** — it was measured under the reach ceiling.
>    Items-per-parameter at d=4 rises **2.0×** (arm b, zero extra params) and **2.8–5.6×**
>    (arm a) at the same cell. Someone must re-measure bits-per-param properly; I did not.
> 3. **N98's "+0.051 monolithic at d=6 K=64" does NOT reproduce on the designed-mechanism
>    harness** (+0.007 ± 0.016, 3 seeds, value-blank controlled). Either the effect is
>    harness-specific (`exp_sharded_store`'s allocator/router) or it was seed noise. This needs
>    a wording fix wherever N98 is quoted as a general localized-init gain.
> 4. **The reach account (N⟨r2geom⟩) is CONFIRMED and sharpened, then partly superseded**: it is
>    a *conflict* between two conditions (`s ≳ |a|max/κ` for reach vs `s ≲ sep/2.4` for
>    packing), and the read-time schedule dissolves the conflict rather than trading it off.
>    §3's `readonly` and `static` controls localize the effect to the **address phase**, not the
>    value phase — the r2geom §4 wording ("the ball is dropped and cannot see the well") is
>    right, but the failure it causes is scored as a **basin** miss.
> 5. **New negative/positive results proposed in §9** (tier A).
> 6. Do-not-quote unchanged otherwise: base √2 / `d^1.62`, "the write operator is the ceiling",
>    width-lock-as-cause, "~32, d-independent" as settled.

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** capacity (the R2 law). A law about the primitive; **its figure is never framed as
  beating anything** (CM-23(m)). Nothing here is compared to kNN, Hopfield or any external
  method.
- **Laundering control:** the designed write at matched geometry, re-measured **at every format
  change** — `K_designed(4) = 128` noise-free at m = 1, 2 and 4 (§5). It never degraded and no
  lever made the learned write more designed (§10, N46).
- **Falsifies:** neither arm moves the wall at ≥3 seeds, budget-adequate, with noise ON.
  → **not triggered**.
- **Does NOT falsify:** failing to reach `4·2^d`; any comparison to external methods.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/r2-excursion-reach`, base local `main` @ `ff85573` |
| commits | `54315c3`, `334c5f2`, `8e51f0d` — see §12 |
| worktree | `../CHLU-r2reach`; **main venv reused** (protocol §4), **JAX 0.9.0**, equinox 0.13.4 |
| harness | `chlu/experiments/exp_designed_mechanism.py` (levers added this wave, all default-off); drivers in `.claude/scratch/r2-excursion-reach/drive.py` |
| geometry | d-ball, farthest-point `designed_sites`, R=1, wall_margin .5, `site_seed=0`, `payload_seed=0` |
| write | `train_memory_landscape`, **GLOBAL** (`learned_global`), 600 Adam(3e-3), wd 1e-4, n_perturb 32, σ_addr .25, σ_pay .6, margin .15, barrier .2, `payload_index = d` (m=1) or `arange(d, d+m)` |
| retrieval | γ_address .05 × 400 → γ_read .02 × 800, dt .05, tail_frac .25, n_subsample 8 — **split across anneal stages, so every arm integrates the same 1200 Verlet steps** |
| queries | `fixed_norm` jitter σ_q = 0.15/√d per axis, σ_p .05, ≤32 per item (cap 4096) |
| **atom budget** | `min_atoms_base = 2048` ⇒ **8192 atoms at d=4** (w25 Stage-1 "coverage-raised" budget, so every number is directly comparable to `r2-geometry-revival`); **`min_atoms_base = 512` ⇒ 4096 atoms at d=6** (the shipped w23 budget) for Stage A |
| `atom_init_width` | **0.30** everywhere except the Stage-A 0.15 arms |
| `atom_init_local` | **False** everywhere except the Stage-A local arms (`atom_init_local_mult = 2.0`) |
| **payload noise** | `payload_launch_sigma = 0.05`, `payload_obs_sigma ∈ {0, 0.005, 0.010, 0.020}`, `pass_metric = "decode"` whenever noise is on; `noise_off` = the shipped `tol` metric at σ=0 |
| payload code | `n_payload_channels ∈ {1,2,4}`, `payload_code = "grid"` (min-separation-preserving lattice, Δ = 2/(K−1) at every m) |
| annealed read | `read_anneal_axes = "payload"`, `read_anneal_payload_mult = κ₀ ∈ {1.5,2,3,5,8}`, `read_anneal_stages = 4` (L=8 and power=2 checked), `read_anneal_phases = "both"` |
| criterion | cell PASS = mean strict ≥ **0.9** ∧ value-blank ok. `strict` = basin_ok ∧ value-ok; value-ok = `‖read−a_i‖ < payload_tol=0.1` (tol) or nearest-codeword decode (decode) |
| seeds | Stage A **3 (0,1,2)**; arm-(b)/(a) headline cells **3** where stated, otherwise the per-cell seed list is printed in every table |
| langevin_noise | **N/A** — deterministic Verlet, no temperature anywhere in this task |

**PREREG:** `.claude/outputs/r2-excursion-reach/PREREG.md`, written before any harness that
measures a registered quantity (only a JAX warm-up, a bit-identity regression cell and a pure
numpy codebook check preceded it).

---

## 0. Harness-integrity check (what licenses every comparison below)

The levers are additive and default-off; the refactored read path is **bit-identical** to the
shipped one when they are off. Three independent checks:

| check | w23/w25 value | this session | verdict |
|---|---|---|---|
| `learned_global` d=4 K=16 seed 0, **shipped default budget** (2048 atoms) | — | `strict = 0.865234375` before **and after** the whole refactor, to every digit | ✅ bit-identical |
| `learned_global` d=4 K=16, **8192 atoms, 3 seeds** | `0.9368 ± 0.0133` (r2geom Stage 1) | **`0.9368 ± 0.0109`** | ✅ reproduced |
| `learned_global` d=4 K=32, 8192 atoms, seed 0 | `0.8242` (r2geom; w23 wall 0.825–0.840 over a 16× atom sweep) | **`0.8242`** | ✅ reproduced |
| designed d=4: K=32 / K=128 / K=256 | `1.0000 / 0.9971 / 0.8577` (w23, r2geom §6) | **`1.0000 / 0.9971 / 0.8577`** | ✅ identical |

---

## 1. ⭐ STAGE A — the init × width factorial (d=6, K=64, monolithic, 3 seeds, 4096 atoms, value-blank ✅ on all 12 cells)

| `atom_init_local` | `atom_init_width` | **strict (3 seeds)** | per-seed | blank ok |
|---|---|---|---|---|
| False | **0.30** (shipped) | **0.8551 ± 0.0106** | 0.8418 / 0.8677 / 0.8560 | ✅✅✅ |
| **True** | **0.30** | **0.8620 ± 0.0114** | 0.8652 / 0.8740 / 0.8467 | ✅✅✅ |
| False | 0.15 | **0.1429 ± 0.0424** | 0.1377 / 0.0938 / 0.1973 | ✅✅✅ |
| **True** | 0.15 | **0.6087 ± 0.0046** | 0.6147 / 0.6079 / 0.6035 | ✅✅✅ |

| contrast | value |
|---|---|
| main effect `atom_init_local` (avg over width) | **+0.2363** |
| main effect width 0.30→0.15 (avg over init) | **−0.4827** |
| **simple effect of `local` at the shipped width 0.30** | **+0.0068** (± ~0.016) — **a null** |
| **simple effect of `local` at width 0.15** | **+0.4658** |
| ⭐ **INTERACTION (local × width)** | **+0.4590, SE 0.0321 ⇒ 14.3 SE** |

**Reading.** The two levers are **not one effect and not two additive effects** — they are
**substitutes**. The localized init repairs **65 %** of the damage a bad (narrow) width does
(0.143 → 0.609), and buys **nothing** once the width is right (+0.007 at 0.30). Both act on
*"atoms in the wrong place"*, and `atom_init_width = 0.30` **already puts them there**, so
there is nothing left for the localized init to buy. The residual 0.855 − 0.609 = **0.246** that
localization does *not* recover is the part of the narrow-width collapse the address-side lever
cannot touch — consistent with r2geom's payload-axis reach account (the localized init is
address-only by construction, N46).

⛔ **Decision-grade output (Head B1.4): the factorial does NOT license promoting
`atom_init_local`.** Its effect at the shipped configuration is **+0.007 ± 0.016**, a null.
**No default was changed in this task.**

⚠ **Two honest caveats.**
1. **A radius confound I did not resolve.** The localization radius is
   `atom_init_local_mult × atom_init_width`, so the width-0.15 arms localize into a ball of
   radius 0.30 while the width-0.30 arms use 0.60. The interaction is therefore
   "localization *scaled with the width*", not "localization at fixed radius". The clean
   control (`mult = 4.0` at width 0.15 ⇒ radius 0.60) is **NOT RUN** (compute).
2. **N98's `+0.051` (monolithic, d=6 K=64) did not reproduce here** (+0.007 ± 0.016). The two
   harnesses differ (`exp_sharded_store` allocator/partition/router vs this value-blank-gated
   one), so this is a **cross-harness non-replication**, not a refutation of N98's own number —
   but it means the localized init cannot be quoted as a general gain.

---

## 2. Arm (b), first attempt — the ISOTROPIC blur is a null, then a catastrophe (and it says why)

`s_eff(t) = √(s² + s_extra(t)²)`, `s_extra` → 0 over L = 4 stages, amplitude-preserving,
d=4 K=32, 8192 atoms, width 0.30 (site separation **0.710**, trained `w_atom` **0.313**):

| `s_extra(0)` | 0 (base) | 0.15 | 0.30 | 0.45 | 0.60 | 0.90 |
|---|---|---|---|---|---|---|
| **learned** strict (seed 0) | **0.8242** | 0.8203 | 0.2383 | 0.0312 | 0.0312 | 0.0312 |
| **designed** strict (K=32) | 1.0000 | 1.0000 | 0.3438 | 0.0312 | — | — |
| mass-preserving mode @0.30 | — | — | 0.3086 | — | — | — |

**Verdict: refuted, and the refutation is the mechanism.** The isotropic blur cannot buy reach
without paying for it in *packing*: to reach a payload at `|a|max = 1` from the payload-zero
manifold you need `s ≳ |a|max/κ ≈ 0.33`; to keep neighbouring wells apart you need
`s ≲ sep/2.4 = 0.30`. **Those two windows do not overlap at K=32, d=4** — and the designed arm,
which has no learning in it at all, collapses on exactly the same schedule (1.000 → 0.344),
proving the damage is geometric, not optimisational. `read_anneal_mode="mass"` (the exact
Gaussian convolution, depth falling as `(s/s_eff)^dim`) is no better (0.309) — **PREREG B3 ✅
(predicted null-or-loss)**, but B1's registered 0.93 for the isotropic schedule is **✗ REFUTED**
(measured 0.238).

---

## 3. ⭐⭐ Arm (b), the fix — the ANISOTROPIC (payload-only) continuation read

The conflict above is between two *different axes*: reach is needed along the **payload**
channel, packing along the **address** axes. So widen only the payload channel, on a schedule
that returns to the stored landscape:
`width_payload(l) = κ(l)·s_j`, `κ: κ₀ → 1` over L stages, address widths untouched.
It is **read-only** (no change to the stored format, the write objective, or the atom budget),
**byte-free** (a scalar schedule; `axis_width_scale` is a tuple of Python floats and is provably
invisible to the write's `trainable_filter` — asserted in a test), and **compute-free**
(`address_steps`/`read_steps` are split across the stages, so the annealed read integrates
exactly the same 1200 Verlet steps as the baseline).

### d=4, K=16, 8192 atoms, **3 seeds**, value-blank ✅
| read | σ_obs=0 (tol) | σ_obs=0.005 (decode) | σ_obs=0.010 (decode) |
|---|---|---|---|
| base (shipped single-stage) | 0.9368 ± 0.0109 | 0.9271 ± 0.0106 | 0.9271 ± 0.0106 |
| **aniso κ₀ = 1.5 … 8, L=4** | **1.0000 ± 0.0000** | **1.0000** | **1.0000** |
| aniso κ₀=3, L=8 | 1.0000 | 1.0000 | 1.0000 |
| aniso κ₀=3, power=2 | 1.0000 | 1.0000 | 1.0000 |
| **control: `readonly`** (anneal the value phase only) | **0.9368 ± 0.0109** | 0.9271 | 0.9271 |
| **control: `static`** (widen and never sharpen) | 1.0000 | **0.9349 ± 0.0335** | **0.9160 ± 0.0209** |

### ⭐ d=4, **K = 32** — the w23 firm wall (flat at 0.825–0.840 over a 16× atom sweep), **3 seeds**, value-blank ✅
| read | σ_obs=0 (tol) | σ_obs=0.005 (decode) | σ_obs=0.010 (decode) | basin |
|---|---|---|---|---|
| base (shipped) | 0.8210 ± 0.0028 | 0.8203 ± 0.0029 | 0.8190 ± 0.0033 | 0.8203 |
| aniso κ₀=1.5 | 0.9561 ± 0.0014 | 0.9538 ± 0.0017 | 0.9525 ± 0.0020 | 0.9538 |
| aniso κ₀=2.0 | 0.9935 ± 0.0020 | 0.9919 ± 0.0024 | 0.9906 ± 0.0020 | 0.9919 |
| **aniso κ₀=3.0, L=4** ⭐ | **0.9997 ± 0.0005** | **0.9993 ± 0.0005** | **0.9980 ± 0.0000** | 0.9993 |
| aniso κ₀=5.0 | 0.9990 ± 0.0014 | 0.9984 ± 0.0017 | 0.9967 ± 0.0020 | 0.9984 |
| aniso κ₀=8.0 | 0.9889 ± 0.0096 | 0.9883 ± 0.0096 | 0.9860 ± 0.0096 | 0.9883 |
| κ₀=3, **L=8** | 0.9997 ± 0.0005 | 0.9993 | 0.9980 | 0.9993 |
| κ₀=3, **power=2** | 0.9997 ± 0.0005 | 0.9993 | 0.9980 | 0.9993 |
| **control: `readonly`** (value phase only) | **0.8210 ± 0.0028** | 0.8203 | 0.8190 | 0.8203 |
| **control: `static`** (never sharpen) | **0.6250 ± 0.0255** | **0.2389 ± 0.0150** | **0.2305 ± 0.0180** | **0.9993** |

`Δ(κ₀=3 − base) = +0.1787` with per-seed sds 0.0028 / 0.0005 ⇒ **SE of the difference 0.0016,
i.e. ~110 SE**; the two 3-seed ranges (0.8174–0.8242 vs 0.9990–1.0000) do not come close to
touching. **The w23 firm wall at d=4 K=32 is cleared at 0.9997, and it is cleared under payload
read-noise (0.9980 at σ_obs = 0.010) where the designed reference itself still passes (0.9990).**

**Three things that make this a mechanism result and not a knob-turn.**
1. **`readonly` reproduces the baseline to four decimals** (0.8242 vs 0.8242 at K=32; 0.9368 vs
   0.9368 at K=16). The *entire* gain comes from annealing during the **address-relaxation**
   phase. The widened payload well is what lets the ball *find and hold the right basin*; by the
   time the value phase starts the decision is already made. This is r2geom's reach account,
   localized: a well whose payload sits far from the launch manifold is invisible **while the
   address is being chosen**, which is why its failure was always scored as a *basin* miss
   (`basin ≡ strict` at every failing cell, in w23, w25 and here).
2. **`static` (widen and never sharpen) fails hard — with `basin = 0.9990`.** A permanently
   widened payload well gets the address *perfectly right* and then reads the **wrong value**
   (0.625 at K=32, 0.205 under decode), because the value is read off a blurred landscape.
   ⇒ "just use wider wells" is not the same result, and the **continuation is load-bearing**:
   reach needs a wide well, fidelity needs the true one, and the schedule delivers both.
3. **A broad, smooth optimum** (κ₀ ∈ [2, 5], with L ∈ {4, 8} and two schedule shapes agreeing to
   ≤0.01) — not a knife-edge tuning artifact. κ₀ = 8 begins to cost (0.990).

---

## 4. ⭐⭐ Arm (a) — the multi-channel payload code (d=4, 8192 atoms, width 0.30)

The code (`payload_code="grid"`) puts the K codewords on the **Δ-spaced lattice in `R^m`** with
**Δ = 2/(K−1)**, the spacing of the shipped 1-channel `linspace(−1,1,K)`, keeping the K
smallest-norm lattice points. **Minimum codeword separation is identical at every m** (measured,
below) — so at a given per-axis read noise the K codewords are exactly as distinguishable and
the item carries the same `log₂K` bits. Only the **excursion** falls.

| K=32, d=4, **3 seeds** unless noted | max‖a‖ | code Δ | learned params | σ=0 (decode) | σ=0.005 | **σ=0.010** | σ=0.020 |
|---|---|---|---|---|---|---|---|
| **m = 1** (the shipped format) | 1.0000 | 0.0645 | 57 344 | 0.8047 ± 0.0060 | 0.8047 ± 0.0060 | **0.8034 ± 0.0064** | 0.7174 |
| **m = 4, grid** ⭐ | **0.0912** | **0.0645** | 81 920 | **1.0000** | **1.0000** | **0.9980 ± 0.0008** | 0.8099 |
| **control: m = 4 `spectator`** (4 channels allocated, code in channel 0, excursion 1.0) — 2 seeds | 1.0000 | 0.0645 | 81 920 | 0.8389 ± 0.0137 | 0.8389 | **0.8374 ± 0.0142** | 0.7471 |
| **control: w25 `pscale = 0.5`** (codebook *and* tolerance halved — the w25 manipulation) — 2 seeds | 0.5000 | **0.0323** | 57 344 | 0.9995 | 0.9985 | **0.8936 ± 0.0049** | 0.5894 |
| **K = 64, m = 4** (3 seeds) | **0.0550** | 0.0317 | 81 920 | 0.9998 ± 0.0002 | **0.9950 ± 0.0008** | 0.7492 | 0.2939 |
| **K = 128, m = 4** (2 seeds) | **0.0352** | 0.0157 | 81 920 | **0.9634 ± 0.0159** | 0.7079 | 0.2445 | 0.0717 |
| K = 64, m = 1 (`armb` baseline, seed 0) | 1.0000 | 0.0317 | 57 344 | 0.6401 | 0.6401 | 0.5630 | — |

### The two controls are the whole fairness argument, and both land
- **`spectator` (C3 ✅).** Give the store the *same* `m = 4` latent geometry and the *same*
  81 920 parameters, but leave the code at excursion 1.0: **0.8389**, i.e. m=1's 0.8047 to
  within +0.034. **The extra dimensions and the extra 43 % of parameters buy essentially
  nothing — the gain is the CODE, not the bytes.**
- **`pscale = 0.5` (C4 ✅, and this is condition 3's whole point).** w25's manipulation
  reproduces its free lunch at zero noise (0.9995) — and then **dies as the read noise rises**
  (0.9985 → 0.8936 → 0.5894), because it halves Δ. The multi-channel code, which holds Δ fixed,
  **does not** (1.0000 → 0.9980 → 0.8099). At the headline σ_obs = 0.010 the difference is
  decisive: **m=4 PASSES at 0.998, pscale FAILS at 0.894.** *The free lunch dies; the code
  survives.* This is why the w25 ban on quoting `K_learned` at pscale ≠ 1 was right, and why
  these numbers (all at pscale = 1) are not the same object.

### ⚠ A harness defect this exposed, and it matters for anyone reusing the criterion
**The shipped absolute-tolerance value test is VACUOUS for any code whose excursion is below
`payload_tol`.** At m=4, K=32 the whole codebook lives inside ‖a‖ ≤ 0.091 < `payload_tol` = 0.1,
so a **blank landscape scores `strict = 1.0000`** on the `tol` metric (measured). The
nearest-codeword **decode** metric is unaffected: the same blank scores **0.0312 = 1/32,
exactly chance** (measured). **Every arm-(a) number quoted above is the decode metric**, and no
`tol`-metric number at m > 1 may ever be quoted. (`evaluate_arm_cell`'s `trivial_ceiling` guard
already detects this and would have refused the cell — it is the reason the guard exists.)

---

## 5. ⭐ The laundering control — the designed write at matched geometry, at EVERY format

`BallRegisterPotential` re-measured on this harness with the **same** payload code and the
**same** read (seed 0, no learning anywhere in this arm):

| m | K=32 | K=64 | K=128 | K=256 | **`K_designed(4)`** |
|---|---|---|---|---|---|
| **1** (σ=0) | 1.0000 | 1.0000 | **0.9968** | 0.8604 | **128** |
| **2** (σ=0) | 1.0000 | 1.0000 | **0.9968** | 0.8884 | **128** |
| **4** (σ=0) | 1.0000 | 1.0000 | **0.9968** | 0.8884 | **128** |
| 1 (σ_obs=0.005) | 1.0000 | **0.9980** | 0.7227 | 0.0815 | **64** |
| 2 (σ_obs=0.005) | 1.0000 | 0.9966 | 0.7930 | 0.2729 | **64** |
| 4 (σ_obs=0.005) | 1.0000 | 0.9946 | 0.7329 | 0.1704 | **64** |
| 1 (σ_obs=0.010) | 0.9990 | 0.8838 | 0.5046 | 0.0720 | **32** |

1. **`K_designed(4) = 128` at every m.** The laundering line is intact: no format change made
   the designed reference weaker (or stronger — it is flat in m, as it must be, see §6).
2. **The designed arm also measures the FORMAT's own noise ceiling** — and this is the piece of
   protocol the w26 conditions actually needed. At σ_obs the codebook spacing Δ = 2/(K−1) caps
   *any* store: at σ_obs = 0.005 the designed register itself fails K=128 (0.72) and passes
   K=64 (0.998); at σ_obs = 0.010 it fails K=64 (0.884). **A learned wall is only claimable
   where the designed arm still passes**, and every headline below obeys that rule.
3. The multi-channel code is very slightly **worse** for the designed arm under noise
   (0.9946 vs 0.9980 at K=64) — the lattice kissing-number penalty I pre-registered as arm
   (a)'s honest cost. **The code is never flattered by the criterion.**

---

## 6. ⭐⭐ The result: where the wall is now

d = 4, 8192 atoms, `atom_init_width` 0.30, global write, pscale = 1, **payload read-noise ON**
(σ_launch = 0.05), nearest-codeword decode, value-blank ✅ on every PASS.

| σ_obs (σ_launch = 0.05 throughout) | designed (the format ceiling) | learned, shipped read + m=1 | learned + **arm (b)** (0 extra bytes) | learned + **arm (a)** m=4 |
|---|---|---|---|---|
| 0 (decode) | **128** | **16** | **32** | **128** — 0.9634 ± 0.0159, 2 seeds |
| 0.005 | **64** | **16** | **32** | **64** — 0.9950 ± 0.0008, 3 seeds |
| 0.010 | **32** | **16** | **32** — 0.9980 ± 0.0000, 3 seeds | **32** — 0.9980 ± 0.0008, 3 seeds |

- **arm (b)** moves `K_learned(4)` **16 → 32** — *one ladder rung, at zero extra parameters, zero
  extra latent dimensions, zero extra Verlet steps and no change to the stored format at all.*
  It **fails at K=64** (best 0.8506 at κ₀=3, seed 0; N92 re-check below), so its wall is 32.
- **arm (a)** moves it **16 → 128** — `K_designed(4)` itself. At every σ_obs measured the learned
  wall **matches the designed wall**: **≥128 vs 128** (σ=0), **64 = 64** (σ=0.005) and
  **32 = 32** (σ=0.010) — the last two bracketed by a measured PASS *and* a measured FAIL on
  **both** arms.
  **The learned/designed prefactor tax at d = 4 goes from 1/8 (w23: 16 vs 128) to 1/1.**
  ⚠ K=128 is **2 seeds** (0.9595 / 0.9822 at σ=0 tol; 0.9475 / 0.9792 decode), value-blank
  0.0063 ≈ 1/128; K=256 was not run, so the learned wall is `≥128` and the designed one is
  exactly 128 (0.8884 at K=256).

**Capacity per parameter (reconciliation item 2).** At the same cell and the same write:
| arm | items | learned floats | items / 10⁵ floats |
|---|---|---|---|
| shipped | 16 | 57 344 | 27.9 |
| **arm (b)** | 32 | **57 344** | **55.8 (×2.00)** |
| **arm (a)** m=4, σ_obs = 0.005 | 64 | 81 920 | **78.1 (×2.80)** |
| **arm (a)** m=4, σ_obs = 0 | 128 | 81 920 | **156.3 (×5.60)** |
w21's ~1.3-bits-per-param figure was measured **under** this ceiling and is therefore **stale**;
I did not re-derive it, I only show that the numerator moved.

---

## 7. Value-blank controls (every reported PASS carries one) and the N92 budget check

**Value blank** = the identical write with all payloads set to 0, scored against the *real*
codebook. A working store must fail there.

| cell | metric | blank strict | chance / trivial ceiling | verdict |
|---|---|---|---|---|
| arm (b), K=32, m=1, **every** annealed schedule (κ₀ 1.5…8, L=4/8, pow2, readonly, static), 3 seeds | decode | **0.0335** | 1/32 = 0.0313 | ✅ at chance |
| " | tol | 0.1250 | 4/32 = 0.125 (trivial ceiling) | ✅ at the ceiling |
| arm (a), K=32, m=4, 2 seeds | decode | **0.0312** | 1/32 = 0.0313 | ✅ exactly chance |
| arm (a), K=64, m=4, 1 seed | decode | **0.0156** | 1/64 = 0.0156 | ✅ exactly chance |
| arm (a), K=128, m=4, 1 seed | decode | **0.0063** | 1/128 = 0.0078 | ✅ at chance |
| arm (a), K=32/64/128, m=4 | tol | 1.0000 / 0.9990 / 0.9341 | **vacuous** (whole code inside `payload_tol`) | ⛔ metric unusable at m>1 |

**N92 budget adequacy at the first-fail cell.** Arm (b) first fails at **d=4, K=64**. Re-run at
**2× atoms (16 384)**, seed 0:

| read | 8192 atoms | **16 384 atoms** |
|---|---|---|
| base | 0.6387 | 0.6270 |
| aniso κ₀=1.5 | 0.7637 | 0.7524 |
| aniso κ₀=2.0 | 0.8359 | 0.8257 |
| aniso κ₀=3.0 | 0.8506 | 0.8027 |
| aniso κ₀=5.0 | 0.6992 | 0.6064 |

**Flat-to-slightly-worse ⇒ not budget-limited ⇒ arm (b)'s wall at 32 is real, not starvation.**

---

## 8. PREREG scorecard

### Stage A
| # | registered | measured | verdict |
|---|---|---|---|
| A0 | baseline (local=F, w=0.30) **0.85 ± 0.10** | **0.8551 ± 0.0106** | ✅ |
| A1 | main effect of `atom_init_local` at w=0.30 = **+0.051** (N98's own value) | **+0.0068 ± ~0.016** | ✗ **REFUTED — N98's number does not reproduce on this harness** |
| A2 | width 0.30→0.15 at local=F = **−0.25 ± 0.15**, sign certain | **−0.712** | ◐ sign right, magnitude ~3× larger |
| **A3 (mine)** | interaction **0.00 ± 0.05** — "two additive effects", P=0.70 | **+0.4590, 14.3 SE** | ✗ **REFUTED — my central prediction and my confidence were both wrong** |
| **A3′ (Advisor-2)** | "substantially ONE effect": interaction ≥ **+0.15** *and* localization rescues w=0.15 to **≥0.85** | interaction **+0.459 ✓**; rescue to **0.6087 ✗** | ◐ **Advisor-2 wins the interaction call; the rescue clause fails.** The levers are **substitutes**, not one effect and not two additive ones |
| A4 | best cell = local=T, w=0.30 | 0.8620 (tied with F/0.30 at +0.007) | ◐ nominally right, statistically a tie |

### Arm (b)
| # | registered | measured | verdict |
|---|---|---|---|
| B0 | baseline reproduction K=16 **0.937**, K=32 **0.824** | **0.9368 ± 0.0109**, **0.8210 ± 0.0028** | ✅ |
| B1 | **isotropic** κ, `s0=0.30`, K=32 → **0.93** [0.85, 0.99], P(≥0.9)=0.50 | **0.2383** | ✗ **REFUTED** — and the refutation named the mechanism (reach/packing conflict, §2) |
| B2 | K=16 → ≥0.97 | 1.0000 (anisotropic); 0.9368 (isotropic, no change) | ◐ true only for the arm I had not yet designed |
| B3 | mass mode ≤ baseline + 0.02 | 0.3086 vs 0.8242 | ✅ (predicted null-or-loss) |
| B4 | optimum `s0 ∈ [0.2, 0.4]`; degrades at `s0 ≥ sep/2` | isotropic degrades from **s0 = 0.30 onward** (sep/2 = 0.355) | ✅ the *threshold* was right; I registered a gain on the wrong side of it |
| B5 | annealed gain persists to within **0.03** under noise | gain **+0.1787** at σ=0 vs **+0.1790** at σ_obs=0.010 (Δ = 0.0003) | ✅✅ |
| B6 | wall 16 → **32**, P=0.45 | **32** | ✅ **the registered central case, at the registered probability** |
| — | N92 at the first-fail | 2× atoms flat (§7) | ✅ |

### Arm (a)
| # | registered | measured | verdict |
|---|---|---|---|
| C1 | K=32, m=4 ≥ **0.95** vs m=1 ≈ 0.82, P=0.55 | **1.0000 ± 0.0000** vs **0.8047 ± 0.0060** | ✅ |
| C2 | K=32, m=2 intermediate 0.88–0.95 | ⛔ **NOT RUN** (compute) | — |
| C3 | **spectator** control within ±0.03 of m=1 | m=1 0.8047, spectator **0.8389** (+0.034) | ◐ **just outside my ±0.03 band**, but the conclusion holds: the extra dims/params buy +0.034 while the code buys +0.195 |
| C4 | pscale control loses ≥0.10 by σ=0.02 while m=4 loses <0.03 | pscale **0.9995 → 0.5894** (−0.41 ✓); m=4 **1.0000 → 0.8099** (−0.19, **not** <0.03) | ◐ **the decisive clause ✅** (at σ=0.010 m=4 passes 0.998, pscale fails 0.894); my "<0.03" was too optimistic — the lattice kissing-number cost I registered in §2 of the PREREG is what shows up |
| C5 | `K_designed(4) = 128` at m = 1, 2, 4 | **128 / 128 / 128** | ✅ |
| C6 | wall 16 → **≥32** with noise ON, P=0.55 | **64** at σ_obs=0.005 (3 seeds); **128** at σ_obs=0 (2 seeds) | ✅ **exceeded** |

**Global falsifier: NOT triggered.** Both arms move the wall at ≥3 seeds (arm b) / 3 seeds at
K=32 and 2–3 at K=64 (arm a), budget-adequate, with payload read-noise on.

---

## 9. What this means for R2 (stated with the precision rules intact)

**The reach account of w25 is confirmed, localized, and then dissolved.** r2geom established that
the binding constraint on the learned atom dictionary is a **read-out reach** condition: the read
launches on the payload-zero manifold and a well whose payload sits at `|a| = 1` exerts almost no
force there (`exp(−r²/2s²) = 5.5e−3` at `s = 0.31`). This wave adds three things:

1. **The constraint is a CONFLICT, not a threshold.** Reach wants `s ≳ |a|max/κ ≈ 0.33`; packing
   wants `s ≲ sep/2.4 = 0.30` at d=4 K=32. The two windows do not overlap — which is why the
   isotropic blur, and any single global width, must fail (§2, and the *designed* register fails
   on the identical schedule, so this is geometry, not optimisation).
2. **The conflict is between two DIFFERENT AXES, so it can be dissolved rather than traded.**
   Reach is needed along the payload channel; packing along the address axes. Widen only the
   payload channel, on a schedule that returns to the stored landscape, and both conditions are
   met at once. This costs **no parameters, no latent dimensions and no Verlet steps**.
3. **The failure is an ADDRESS-ACQUISITION failure, not a value-read failure.** The `readonly`
   control (anneal the value phase only) reproduces the baseline to four decimals across three
   seeds; the `static` control (widen and never sharpen) gets `basin = 0.9993` and then reads the
   *wrong value* (0.239 under decode). Reach is needed *while the basin is being chosen*; fidelity
   is needed *after*. That is why every failing cell in w23/w25/here has `basin ≡ strict`.

**The consequence for the R2 law, at d = 4, with payload read-noise on and the designed arm as
the reference at every noise level:**

| σ_obs (σ_launch = 0.05, decode metric) | `K_designed(4)` | `K_learned(4)` shipped | + arm (b) | + arm (a), m=4 |
|---|---|---|---|---|
| 0 | **128** | 16 | 32 | **128** (0.9634 ± 0.0159, 2 seeds) |
| 0.005 | **64** | 16 | 32 | **64** (0.9954 ± 0.0007, 2 seeds) |
| 0.010 | **32** | 16 | **32** | **32** (0.9980 ± 0.0008, 3 seeds) |

**The learned/designed prefactor gap at d = 4 — the 4× / "1/16 tax" that has stood since w23 —
closes to 1/1 at every noise level measured**, once the read-out code stops demanding an
excursion the Gaussian basis cannot reach. The precision rules stay intact: this is a statement
about **d = 4 only**; nothing here re-measures the exponent, and `K_designed = 4·2^d` is
unchanged (128 = 4·2⁴ ✓, measured, not assumed).

### Proposed results for `negative_results.md` / the claims matrix (tier A)

> **N⟨next⟩ (POSITIVE, memory-architecture) — the learned-capacity wall at d=4 is a property of
> the READ-OUT CODE, and it moves when the code moves.** The w23 wall `K_learned(4) = 16` (with a
> *firm* fail at K=32, flat over a 16× atom sweep) is not a property of the write operator, the
> atom budget or the address packing. Two independent read-out-side interventions move it, both
> at `pscale = 1`, both with payload read-noise on (`σ_launch = 0.05`, nearest-codeword decode),
> both with a value-blank at chance, and both with the designed register re-measured at
> `K_designed(4) = 128` on the identical harness:
> **(a)** a **multi-channel payload code** that holds the codeword minimum separation at the
> 1-channel value `Δ = 2/(K−1)` (hence `log₂K` bits at the same per-axis noise) while cutting the
> per-item excursion `max‖a‖` from 1.000 to 0.091 (m=4, K=32) takes `K_learned(4)` to **64** at
> `σ_obs = 0.005` and **128** at `σ_obs = 0` — i.e. to `K_designed(4)` at the same noise;
> **(b)** an **anisotropic continuation read** (widen only the payload channel by `κ: 3 → 1` over
> 4 stages, `address_steps`/`read_steps` split so the step count is unchanged) takes the K=32
> cell from **0.8210 ± 0.0028 to 0.9997 ± 0.0005** (3 seeds, ~110 SE) at **zero extra parameters,
> zero extra dimensions and zero extra compute**, and fails at K=64 (0.851, flat at 2× atoms).
> Three controls make this a mechanism rather than a knob: the **spectator** control (m=4 latent
> dims and +43 % parameters, code left at excursion 1.0) gains only +0.034 while the code gains
> +0.195; the **`readonly`** control (anneal the value phase only) reproduces the baseline to four
> decimals; and the **`static`** control (widen and never sharpen) attains `basin = 0.9993` and
> then reads the wrong value (0.239), so the continuation, not the width, is what buys fidelity.
> **The isotropic blur is REFUTED** (K=32: 0.824 → 0.238 at `s_extra = 0.30`) and the refutation
> is the mechanism: reach needs `s ≳ |a|max/κ ≈ 0.33` while packing needs `s ≲ sep/2.4 = 0.30`,
> windows that do not overlap — the *designed* register collapses on the identical schedule
> (1.000 → 0.344), so the damage is geometric.

> **N⟨next+1⟩ (NEGATIVE, methodology) — the absolute-tolerance value criterion is VACUOUS for any
> read-out code whose excursion is below `payload_tol`.** At m=4, K=32 the entire codebook lies
> inside `‖a‖ ≤ 0.091 < payload_tol = 0.1`, and a **blank landscape scores `strict = 1.0000`**.
> The nearest-codeword decode metric is unaffected (same blank: **0.0312 = 1/32, exactly
> chance**; at K=64: **0.0156 = 1/64**). Any future excursion or codebook manipulation must be
> scored by decoding, not by an absolute error tolerance — an absolute tolerance is structurally
> blind to the codebook spacing and therefore neither rewards nor punishes a change of excursion,
> which is precisely the hole the w25 `pscale` probe fell through. Measured here: the w25
> `pscale = 0.5` manipulation reproduces its free lunch at zero noise (0.9995) and **dies as read
> noise rises** (0.8936 at σ_obs = 0.010, 0.5894 at 0.020) because it halves Δ, while the
> multi-channel code — which holds Δ fixed — passes at 0.9980.

> **N⟨next+2⟩ (NEGATIVE, memory-architecture) — the localized atom init and `atom_init_width` are
> SUBSTITUTES, and at the shipped width the localized init is a null.** 2×2 factorial, d=6 K=64
> monolithic, 3 seeds, value-blank on all 12 cells: `atom_init_local` buys **+0.0068 ± 0.016 at
> `atom_init_width = 0.30`** (a null) and **+0.4658 at width 0.15**; interaction **+0.4590,
> 14.3 SE**. Both levers attack "atoms in the wrong place", and the shipped width already puts
> them there. **N98's +0.051 for the monolithic d=6 K=64 cell does not reproduce on the
> value-blank-gated designed-mechanism harness** — a cross-harness non-replication that must be
> attached wherever N98 is quoted as a general localized-init gain. ⚠ Confound not resolved: the
> localization radius is `mult × atom_init_width`, so the two width arms localize at different
> radii; the fixed-radius control is NOT RUN.

---

## 10. Fairness ledger (the five conditions, discharged)

| # | condition | how it was discharged | evidence |
|---|---|---|---|
| 1 | **bits-per-item constant** | every arm stores one of **K** codewords with **minimum separation Δ = 2/(K−1)** — measured to 1e−5 at every (m, K) and asserted in a test | §4 table (`code minsep` column identical across m); `test_payload_codebook_holds_min_separation_and_cuts_excursion` |
| 2 | **byte accounting pinned** | learned floats printed per cell: m=1 → **57 344**, m=4 → **81 920** (+42.9 %, all in the atom centers). Arm (b) adds **zero**. The **spectator** control gives the extra dims+params *without* the code and gains only +0.034 | §4, §6 |
| 3 | **payload read-noise ON** | `σ_launch = 0.05` on every Stage-B number; `σ_obs` swept 0 → 0.020; scoring by nearest-codeword **decode**. The w25 `pscale` manipulation was run *inside the same sweep* and **dies** while the code survives | §4 |
| 4 | **baselines get the same format** | `BallRegisterPotential` generalized to (K,m); designed arm re-measured at m = 1, 2, 4 **and** through the annealed schedule | §5 |
| 5 | **laundering travels** | `K_designed(4) = 128` at every format and every table; the designed arm also *defines* the format's noise ceiling, and no learned PASS is claimed above it | §5, §6 |

**N46 (scope collapse) — nothing here made the learned write more designed.** The write still
receives only the target sites, exactly as in w20–w25; every center, width and amplitude is still
learned by the same static objective. Arm (b) touches **only the read** (a scalar schedule on a
per-axis width multiplier held as Python floats, provably outside the write's `trainable_filter`
— asserted in a test). Arm (a) changes **only the output code**, which the Head ruled a legitimate
interface parameter, and the designed arm reads the identical code. The **designed arm kept
reaching its own wall at every format change** (§5), so the laundering control never fired.

---

## 11. How I verified — commands, and what I did NOT run

```
# Stage A (2x2 x 3 seeds, d=6 K=64, shipped 4096-atom budget)
PYTHONPATH=../CHLU-r2reach .venv/bin/python drive.py stageA {0,1} {0.30,0.15} 0,1,2 A_*.json
# arm (b): ONE write per (K,seed), then 11 read schedules x 3 noise settings on it
        drive.py armb {16,32,64} 0,1,2 B_aniso_K*.json 4 4 learned_global aniso
# arm (b) N92 budget re-check at 2x atoms
        drive.py armb 64 0 B_aniso_K64_2x.json 4 8 learned_global aniso
# arm (a): multi-channel code + spectator + pscale controls
        drive.py arma {32,64,128} {1,2,4} 0,1,2 C_*.json 4 4 {grid,spectator,pscale0.5}
# laundering: the designed arm at every format (no training)
        designed2.py                       # m in {1,2,4} x K in {32,64,128,256}
# value-blank controls
        drive.py armb_blank {32,64,128} 0,1,2 BLANK_*.json 4 4 {1,4} [aniso]
```
All raw JSON + logs: `.claude/scratch/r2-excursion-reach/` (`A_*` Stage A, `B_*` arm (b),
`C_*` arm (a), `D_*` designed/laundering, `BLANK_*` value blanks); aggregator `agg.py`.

- **Harness integrity:** §0 — the refactor is bit-identical when the levers are off
  (`0.865234375` to every digit), and w23/w25's decisive numbers reproduce (0.9368, 0.8242,
  1.0000/0.9971/0.8577).
- `ruff check` **clean** on `chlu` and `tests`. `ruff format --check` reports drift in
  `config.py`, `memory_potentials.py`, `exp_designed_mechanism.py`,
  `tests/test_designed_mechanism.py` — **verified pre-existing**: the identical four files report
  the identical drift on the *unmodified* `main` checkout. Per protocol §3.3 I did not reformat
  out-of-scope shared code (same finding as w23/w24/w25).

### ⛔ NOT RUN (declared, never reported as a null)
| item | why |
|---|---|
| **d = 6 and d = 8 for either arm** | compute. Priority order was declared in the PREREG (Stage A → arm b d=4 → arm a d=4 → d=6 → d=8) and the machine ran 8–10 concurrent worktree jobs on 8 shared cores all session (Stage A alone cost ~1160 s **per cell per seed**, ~5× its solo cost). **The wall movement is a d=4 result only.** |
| **arm (a) at m = 2** (C2) | compute; m=1 and m=4 bracket it and the designed arm was run at m=2 |
| **arm (b) at K=64, seeds 1–2** | killed to free cores once seed 0 + the 2× atom budget re-check had both settled the FAIL. The K=64 arm-(b) failure rests on **1 seed + N92** |
| **arm (a) at K=256** | would establish whether `K_learned(4)` at m=4 is exactly 128 or higher; the designed arm already fails at 256 (0.888), so it would likely be a FAIL either way |
| **the Stage-A fixed-radius control** (`mult=4` at width 0.15 ⇒ radius 0.60, matched to the width-0.30 arms) | compute; leaves the radius confound in §1 open |
| **a per-item `|a_i|`-split reach diagnostic** on the annealed read (r2geom §4's probe) | compute; the `readonly`/`static` controls carry the mechanism instead |
| **re-measuring bits-per-param / capacity-per-byte properly** | out of scope; §6 reports only items-per-parameter at the same cell |

---

## 12. Git footprint

- Branch **`agent/experiment-engineer/r2-excursion-reach`**, base local `main` @ **`ff85573`**
  (verified: `main` still at `ff85573`, clean tree). **Not pushed.** Rebase onto `main` = no-op.
- Worktree **`../CHLU-r2reach`**, main venv reused (JAX 0.9.0). Two other spokes' worktrees
  (`CHLU-cl-encoder`, `CHLU-mbf`) were present throughout — no filesystem collision; the only
  shared file I touched is `chlu/config.py` and my edit is a **contiguous additive block inside
  `ExperimentDesignedMechanismConfig`** (which I own), so the merge conflict, if any, is
  syntax-only.
- **Commits (3):**

| hash | subject | files |
|---|---|---|
| `54315c3` | add the read-out excursion levers to the designed-mechanism harness (w26) | **M** `chlu/config.py` (+13 fields in `ExperimentDesignedMechanismConfig` only), **M** `chlu/core/memory_potentials.py` (`BallRegisterPotential` → (K,m) payloads), **M** `chlu/experiments/exp_designed_mechanism.py`, **M** `tests/test_designed_mechanism.py` (+6 tests) |
| `334c5f2` | add the ANISOTROPIC (payload-only) annealed read (w26 arm b) | **M** `chlu/config.py` (+2 fields), **M** `chlu/core/memory_potentials.py` (`AtomDictionaryPotential.axis_width_scale`), **M** `chlu/experiments/exp_designed_mechanism.py`, **M** `tests/test_designed_mechanism.py` (+1 test) |
| `8e51f0d` | keep the single-channel `_two_phase` return shape (shipped contract) | **M** `chlu/experiments/exp_designed_mechanism.py`, **M** `chlu/core/memory_potentials.py` (`payload_profile` scalar at m=1) |

- **Files touched: 4.** I did **not** touch `chlu/core/controller.py`, `AtomStorePotential.evict`
  (`placement-landing`), `exp_cl_entry.py`, `cl_baselines.py` or `exp_phi_stream.py`.
- `chlu/core/memory_potentials.py` is shared with `placement-landing`; my two edits there are
  confined to **`BallRegisterPotential.__init__`/`payload_profile`/`__call__`** and
  **`AtomDictionaryPotential.__init__`/`__call__`** (one new optional field), i.e. lines nowhere
  near `AtomStorePotential`.
- **Tests: 17 → 24 in `tests/test_designed_mechanism.py`** (+7). **Full suite: `697 passed,
  0 failed` in 667 s** (`690` on `main` after the `ff85573` duplicate cleanup, + my 7). The one
  failure the first full run surfaced —
  `test_sharded_store.py::test_monolithic_arm_reproduces_the_w23_read_path`, a *shape* regression
  from the multi-channel return value — is fixed by `8e51f0d` and was **not** a numerical change
  (the bit-identity regression cell still returns `0.865234375`).
- **Every default is unchanged.** All 15 new config fields default to the shipped behaviour and
  the m=1/no-anneal path is **bit-identical** (`0.865234375`).

---

## 13. Open questions / follow-ups / risks

1. **⭐ The d-sweep is the obvious next task and it is now worth the compute.** Everything here
   is **d = 4**. If the multi-channel code closes the learned/designed gap at d = 6 and d = 8,
   the R2 figure comes back as `K_learned(d) = K_designed(d) = 4·2^d` **with a mechanism story
   attached**, which is the R2 deliverable the task file describes. If it closes only at d = 4,
   that is a different (and still publishable) claim. **Cheapest decisive cell: d = 6, K = 64
   and K = 128, m = 4, 3 seeds**, plus the designed arm at the same cells.
2. **Which arm should become the recommendation?** They are not equivalent:
   - **arm (b)** is free (0 bytes, 0 dims, 0 steps) and needs **no change to the stored format**,
     but only buys one rung (16 → 32) and needs `κ₀` chosen (broad optimum 2–5).
   - **arm (a)** buys four rungs but costs `m−1` latent dimensions (+43 % learned floats at m=4)
     and changes the output code — which the Head has ruled legitimate, but it is a change.
   - **They are untested in combination.** Arm (b) is read-side and arm (a) is code-side, so they
     should compose; `drive.py arma ... anneal=κ` supports it and it was **NOT RUN**.
3. **Risk — the σ_obs choice is a design decision, not a measurement.** The format's own noise
   ceiling (`Δ = 2/(K−1)` vs σ_obs) caps *every* store, and I chose the sweep. I have tied every
   claim to the **designed arm at the same σ_obs**, which is the honest anchor, but a referee can
   still ask why σ_obs = 0.005 rather than 0.02. The defensible framing is the **ratio**: at
   every σ_obs measured, learned = designed, and that ratio is what the R2 tax was about.
4. **Risk — `payload_tol` is now doing nothing at m > 1.** §4 shows the tol metric is vacuous
   there. If anyone re-uses this harness with a shrunken code and the tol metric they will get a
   fake PASS. The guard exists (`trivial_ceiling`) but the *criterion* should probably move to
   decode by default. **I did not change the default** (B1.4 discipline); this is a Hub call.
5. **The Stage-A radius confound** (§1) is unresolved and cheap to close (one cell × 3 seeds).
6. **Arm (b)'s K=64 failure rests on 1 seed + the N92 2× check.** If the Hub wants the arm-(b)
   wall pinned at exactly 32, two more seeds at K=64 are ~40 min of solo compute.
7. **`atom_init_local` should stay off.** Its main effect at the shipped configuration is a null
   and its only measured value is as a repair for a width that is itself wrong.

---

## Proposed handover updates (for the Hub)

1. **⭐⭐ §6 / claims matrix / R2 gate — R2 is UNCLAMPED at d = 4.** `K_learned(4)` moves
   **16 → 32** (arm b, zero bytes) and **16 → 128** (arm a, m=4) at σ_obs = 0, **= K_designed(4)
   at every noise level measured**. The standing R2 wording ("learned pays a widening tax vs
   designed; 4× prefactor gap; `K_learned = min(2^d, ceiling)`") is **d-scoped to the SHIPPED
   read-out code** and must say so. ⚠ **d = 6 and d = 8 are NOT RUN** — the exponent is not
   re-measured and `K_designed = 4·2^d` is untouched.
2. **⛔ Carry the do-not-quote on `K_learned` at `pscale ≠ 1`, and add the reason:** I measured
   the w25 pscale manipulation dying under read noise (0.9995 → 0.5894) while the constant-Δ code
   survives. The ban was right. **New numbers are at pscale = 1** and are quotable *with σ_obs and
   the designed-arm reference in the same sentence*.
3. **New do-not-quote: any `tol`-metric number at m > 1** (a blank landscape scores 1.0000 there).
4. **N92/N96 wording:** the reach account survives and is sharpened into a *two-window conflict*;
   the isotropic-blur refutation and the `readonly`/`static` controls should be attached to it.
5. **N98 needs a cross-harness caveat** (§1 / reconciliation item 3).
6. **w21's bits-per-param (~1.3 vs 2) → STALE**, flagged not re-measured (§6).
7. **Three new results proposed in §9** (one positive, two negative), tier A.
8. **⛔ No default changed.** `atom_init_local` stays `False` (its main effect is +0.007 ± 0.016);
   `atom_init_width` stays 0.30; `n_payload_channels` stays 1; `read_anneal_stages` stays 1;
   `pass_metric` stays `"tol"`. Promoting any of them is a Hub/Head decision, and item 4 above
   (`pass_metric`) is the one I would raise first.
9. **Next task, if the Hub wants R2 closed:** the **d-sweep at m=4** (§13.1) plus the
   **arm (a) × arm (b) combination** (§13.2). Both run on the harness as committed — no new code.
