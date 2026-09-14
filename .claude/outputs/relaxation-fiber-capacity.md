# relaxation-fiber-capacity — physics-theorist report

Task + acceptance criterion: close OQ-4 — the V↔M degeneracy verdict with its separating observable, a channel inventory with per-channel bits, the composition rule `K_total = K_spatial × K_fiber` with validity conditions, and the read-length cost. Every result labelled proven / verified-numerically / conjectured.
Status: **done** (PREREG + addendum written before each harness; all 12 predictions scored in §6, **2 failed**).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). 4 sites.**
> 1. **`clu-autoencoder` (task A) must NOT write `K_total = K_spatial × K_fiber` as an item count.** The correct composition is **slots × payload-bits-per-slot**: `B_total = K_spatial · B_fiber`. Two items *cannot* occupy one address with two different jets — a location has exactly one landscape. Proven, §4.
> 2. **`relaxation-addressing-theory` Prop 11 (my own, w20) needs an amendment**: it says the fiber "carries the Hessian spectrum, read as trajectory frequencies `ω_i = √(k_i/M_i)`". In Newtonian mode **`k` and `M` are not separately readable at all** — the fiber's entire deterministic content is the *effective* force field `G = M⁻¹∇V`, never `V` and `M` apart. Prop 11's toy is correct only because it fixed `M=1`. §2, Prop F1.
> 3. **`clu-memory-architecture` Prop 1 escape route (b)** — "anisotropic per-channel masses breaks the dilation" — is **true but needs its condition**: anisotropic mass is *exactly* gauge-equivalent to reshaping `V` whenever `V` is **separable** at the address. Mass ratios are identifiable **iff the landscape couples the coordinates** (`∂_i∂_j V ≠ 0`), and then only through the *asymmetry* of the effective force Jacobian. Measured: 8.6 bits (coupled) vs **0.20 bits** (separable, = the prior floor).
> 4. **A hard cap nobody has quoted:** for a *learned* `V_θ`, `B_total ≤ P·b_θ` (parameters × bits per parameter). `PotentialMLP(d=8, hidden=32)` has **1377 params**; the measured `K_max = 4·2^d = 1024` wells at d=8, each carrying its own d=8 Hessian (36 reals), needs **≥36 864 independent reals**. **The measured spatial capacity and any per-well fiber are jointly unreachable by the shipped MLP** — the `4·2^d` law was measured on a hand-designed analytic landscape (effectively unlimited parameters). §4.3.

**Headline (one line each).**
- **Item 1 — the Hub's proposed resolution is REFUTED.** Anharmonicity does **not** separate mass from landscape. In Newtonian CLU the degeneracy is an **exact gauge of the whole map** — same trajectory, not merely the same period — for *any* `V`, harmonic or not, dissipative or not. What separates them is **landscape coupling** (`d − k` dimensions, `k` = number of separable blocks), a **momentum-carrying launch**, or the **relativistic kinetic term** (breaking `∝ (v/c)²`, accumulating `∝ N`).
- **Item 2 — the fiber has exactly ONE *storing* channel: the local jet of `V_θ`.** `M` and `p₀` are address-side (reader-supplied ⇒ they **select**, they do not **store**); `γ` is a spatial field (identical for all items at a location ⇒ 0 per-item bits); temperature is the noise floor, not a code.
- **Item 3 — capacity does not multiply as item counts; it multiplies as slots × bits.** The fiber multiplier is **logarithmic in read precision** and roughly **linear in the number of identifiable jet directions**.
- **Item 4 — the fiber is a log channel:** `B(N) = B₀ + (Σᵢαᵢ)·log₂N` with measured `Σα = 11.15` bits/doubling at d=2. **Read length is exponential in bits** — buy capacity with more modes/wells/launches, never with longer reads.
- **The fiber is nonetheless large:** **129 bits** (1 launch) → **174 bits** (2 launches) per well at d=2, N=1200, σ_read/A = 1e-3 — i.e. **payload ≫ address** is real, and the honest number is a bit-count, not an item-count.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| repo state | **untouched, read-only; no branch, no commits.** No repo code imported. |
| scripts | `.claude/scratch/relaxation-fiber-capacity/{common,item1_gauge,item2_capacity,item3_costs,item4_sigma_d2}.py` + `item{1,2,3,4}_results.json` alongside |
| prereg | `.claude/outputs/relaxation-fiber-capacity/PREREG.md` (P1–P7 before any run; addendum P8–P10 committed before the Item-2/3 harness, after Item 1) |
| env | main `.venv` Python 3.11.13, **numpy 2.4.1, float64, no JAX** (protocol §4: main venv reused, no worktree sync) |
| seed | `default_rng(11)` (only the P5 Monte-Carlo validation is stochastic; everything else is deterministic Fisher/CRB algebra) |
| integrator | **line-for-line the shipped `velocity_verlet_step`**: `p½ = p − ½ε∇V(q)`, `q⁺ = q + ε∇T(p½)`, `p⁺ = p½ − ½ε∇V(q⁺)`, `p⁺ ← (1−γ)p⁺` |
| ε / γ | ε = 0.05 throughout; γ = 0.02 for the gauge tests, **γ = 0 for all capacity/read tests** (the fiber read is the γ=0 rollout of the two-phase architecture) |
| kinetic modes | `newtonian_learned` (`T = ½pᵀM⁻¹p`) **and** `relativistic` (`T = c√(pᵀM⁻¹p + (m₀c)²)`, the shipped spelling, `c = m₀ = 1`) |
| mass | **per-launch** `M` (the shipped `mass_override` path, `chlu_unit.py:312`) |
| ⚠ code deviation | toys invert `M` exactly; **the shipped code inverts `M + 1e-6`**, which is itself an explicit (tiny) gauge-breaking term — a repo-side gauge test would saturate at ~1e-6, not 1e-16 |
| noise model | i.i.d. read noise σ per sampled coordinate ("linear-decoder resolution"); default σ/A = 1e-3. Thermal version derived in §3.4, **not measured** |
| landscapes | **hand-designed** polynomial jets. **Nothing here is evidence of emergence** (N46 precedent). No training, no gradient descent anywhere |
| relativity caveat | the structural results (Prop F1, the inventory, the composition rule) are exact for both kinetic modes as stated; all *rates* are from these toys at d ≤ 2 |

---

## 1. The object: what "the fiber" actually is

**Reduction (proven, one line).** Newtonian CLU with diagonal per-launch mass `M`, landscape `V`, per-step damping `(1−γ)` (⇒ continuous friction rate `Γ = γ/ε` on `p`). In the **velocity** variable `u = M⁻¹p`:

```
q̇ = u ,    u̇ = −M⁻¹∇V(q) − Γu ,    u(0) = M⁻¹p₀
```

⇒ **every read — any functional of the sampled `q(·)`, any γ, any read length — depends on `(M, V, p₀)` only through `( G(q) := M⁻¹∇V(q), u₀ := M⁻¹p₀, Γ )`.**

This is the correct definition of the fiber: not "the jet of `V`", but **the jet of the effective force field `G = M⁻¹∇V`**. Everything in Items 1–4 follows from it.

---

## 2. Item 1 — the V↔M degeneracy. **Verdict: exact, and NOT broken by anharmonicity. Proven + verified to 1e-15.**

### 2.1 Proposition F1 (mass–landscape gauge). *Proven.*

Let `Λ = diag(λ₁…λ_d) ≻ 0`. The map

```
(M, V, p₀)  ⟶  (ΛM, Ṽ, Λp₀)      with  ∇Ṽ = Λ∇V
```

leaves **the entire trajectory `q_n` pointwise invariant** — not the period, not the path image, the trajectory — for any `V`, any γ, any `p₀`, in continuous time *and* under the shipped discrete Verlet map. `Λ∇V` is a gradient field iff `(λ_i − λ_j)∂_i∂_jV = 0`, i.e. **`Λ` is constant on each separable block of `V` near the address.**

> **Corollary F1.1 (the counting law).** `dim(gauge) = k` = number of separable blocks of `V` at the address. Mass has `d` parameters ⇒ **mass contributes exactly `d − k` observable dimensions.**
> **Corollary F1.2 (`d = 1`, or any fully separable `V`): mass contributes ZERO** — harmonic *or* anharmonic, at rest *or* under momentum. It is not a channel at all.
> **Corollary F1.3.** In the shipped `PotentialMLP`, the hard-coded `0.05‖q‖²` confinement is **not** rescaled with the learned part of `V`, so a writer that scales only the MLP output breaks the gauge weakly, at the confinement's relative strength. This is a *bug-shaped* gauge breaking, not a channel.

**Why the Hub's proposed resolution fails.** "Mass is a pure time-reparameterization; landscape shape is not" — the first clause is true (Prop 1, `clu-memory-architecture`), but **an overall rescale of `V` is *also* a pure time-reparameterization at a rest launch** (`t → t/√λ`). The two reparameterizations cancel exactly. Anharmonicity changes the *waveform*, but it changes it identically on both sides of the gauge, because the anharmonic coefficients rescale too. Anharmonicity buys nothing.

**Verified (`item1_gauge.py`, float64, ε=0.05, γ=0.02, N=2000, λ=3.7):**

| test | result |
|---|---|
| P1 — anharmonic `V`, `p₀ ≠ 0`, scale `(M,V,p₀)` | max abs Δq = **6.2e-16** (orbit amplitude 0.699) |
| P2a — anharmonic `V`, rest launch, scale `(M,V)` | **1.0e-15** ⇒ **anharmonicity does not break it** |
| harmonic control | **3.9e-16** |
| P2b — scale `(M, k)` only, leave `β,δ` fixed | **rel. difference 0.499** ⇒ the degeneracy is with **the whole `V`**, not with "curvature" |

**The information-theoretic version** (`item3_costs.py`, 1-D, params `(log k, log M)` rotated into gauge/observable coordinates, N=1200, σ_rel=4e-4):

| direction | Newtonian | relativistic (v/c = 0.10) |
|---|---|---|
| gauge direction `log k + log M` | **1.3e-9 bits** | **3.84 bits** |
| observable direction `log k − log M` | 19.51 bits | 12.90 bits |

The Newtonian gauge direction carries **zero** information — the strongest possible form of "one channel, not two".

### 2.2 The three separating observables (and what each costs)

| # | separator | what it measures | condition | cost | status |
|---|---|---|---|---|---|
| **S1** | **asymmetry of the effective force Jacobian** `∂G = M⁻¹∇²V` | `M_i/M_j = (∂G)_{ji}/(∂G)_{ij}` | **requires a coupled landscape** `∂_i∂_jV ≠ 0`; `d ≥ 2` | **1 rest launch**, N ≳ a few periods (used N=4000 = 32 periods for the exactness check) | **proven + verified**: mass ratio recovered to **1.2e-15**; separable control gives off-diagonals **1.3e-16** ⇒ 0/0, unidentifiable |
| **S2** | **momentum launch** `p₀ ≠ 0` with `p₀` known to the reader | `u₀ = M⁻¹p₀` ⇒ the mass *scale* directly | any `V`, any `d`, incl. `d=1` | 1 launch, `N = O(1)` steps (the initial velocity is visible immediately) | **proven** (immediate from F1: fixing `p₀` while scaling `(M,V)` is off the gauge orbit) |
| **S3** | **relativistic kinetic term** | `T = c√(pᵀM⁻¹p+(m₀c)²)` is not homogeneous ⇒ the gauge survives only to Newtonian order, broken at `ξ/(m₀c)² ≈ (v/c)²` | `kinetic_mode="relativistic"` and `v/c` not ≪ 1 | breaking **accumulates ∝ N** (it is a frequency shift) | **proven + verified**: P4 log–log slope **1.989** (registered 2.0±0.25); Newtonian control **exactly 0.0**; P10 secular slope in N = **1.029 / 1.033** |

**S3 is a theoretical payoff worth a paper sentence:** *the relativistic governor is not only a speed limit — it is the term that gauge-fixes `M` against `V`, making inertia an observable. In the Newtonian modes, inertia is unobservable in principle.* Measured worth of that channel (N=1200, σ_rel=4e-4): **0.20 / 1.89 / 3.84 / 5.82 / 7.74 / 9.47 bits** at `v/c = 0.02 / 0.05 / 0.10 / 0.20 / 0.38 / 0.65` — i.e. exactly **+2 bits per doubling of `v/c`**, confirming SNR ∝ `(v/c)²` to 3 significant figures over 4 octaves.

### 2.3 The sharp design constraint the task asked for

> **If your landscape is separable at the address (in particular: any 1-D fiber, and any well whose Hessian is diagonal in the mass basis), the mass channel is worth exactly nothing — harmonic or anharmonic, at any read length, in either Newtonian mode.** Measured: **0.20 bits** (separable) vs **8.6–12.5 bits** (coupled), a ~50× information ratio at identical read cost.

⚠ This is a **stronger** constraint than the harmonic-only one the task anticipated: the killer is **separability**, not harmonicity.

---

## 3. Item 2 — the channel inventory

**The decisive distinction (proven, and it does most of the work): address-side channels *select*; only landscape-side channels *store*.** A quantity the reader must supply at launch time cannot deliver stored bits — the bits are then in whatever table supplied it. If the selector emits `M` per item, the memory is the codebook, not the physics.

| channel | side | per-item at a fixed location? | learnable — which Prop-7-compliant route | bits from a length-N read | status |
|---|---|---|---|---|---|
| **curvature `∇²V(a*)`** | **landscape** | **yes** (iff `V` is a local dictionary; **no** for a global MLP — CM-6 write leakage) | (ii) θ-grad through relaxation (IFT); or designed-in | `d(d+1)/2` coefficients; **the dominant term of the payload** — measured 3 modes carrying ~17–20 bits each at σ_rel=1e-3 | verified-numerically |
| **anharmonic coefficients, order r** | **landscape** | yes (same condition) | (ii) / designed-in | `C(d+r−1, r)` coefficients; readable only at orbit amplitude `A`, relative signal `(A/L)^{r−2}`; measured **5 of 8** 1-D coefficients at `A/L = 0.4`, **7 of 8** at `A/L = 0.55` (just sub-barrier) | verified-numerically |
| **`μ²` (soft/flat Hessian eigenvalue)** | **landscape** | yes | (ii) for `μ² > 0`; **exact zero is designed-only** (R-4 no-go, 12-order gap) | *same coefficient as curvature*, but its read costs `N ≳ 2π√(M/μ²)` ⇒ **permanence costs read latency**; `μ²→0` is unreadable as a frequency and instead becomes an **endpoint** coordinate (the coset register) ⇒ it belongs to `K_spatial`, **not** to the fiber | proven (structural) |
| **`M` (scalar / global part)** | **address** (per-launch) | per-launch ⇒ **cannot store** | (i) selector regression; Prop 6: ratios learnable to 2.2e-14 | **0 bits, Newtonian** (exact gauge, measured 1.3e-9); **0.2–9.5 bits relativistic**, `≈2·log₂(v/c) + const`; **O(1) bits Newtonian iff `p₀ ≠ 0`** | proven + verified |
| **`M` (anisotropy, `d−1` ratios)** | **address** (per-launch) | per-launch ⇒ **cannot store** | (i) | `d − k` dimensions (`k` = separable blocks). Measured d=2: **8.6 bits** coupled (1 launch) → 12.5 (8 launches); **0.20 bits** separable | proven + verified |
| **`p₀`** | **address** (per-launch) | per-launch ⇒ **cannot store** | (i) regression; (iii) derivative-free boost ladder | selects the energy shell (Prop 2 / `M* = p₀²/2h`); as a *read key* it demultiplexes the stored jet (§4.2) | proven (imported) |
| **`γ`** | landscape (**global scalar or a spatial field `γ_φ(q)`**) | **NO** — identical for every item at a location | (ii) | **0 per-item fiber bits.** It carries *spatial-map* bits (a second scalar field over space: retention/vault/deletion) — a different resource | proven (structural, per task constraint) |
| **temperature `T`** | neither — **stochastic, and unbuilt as a field** | n/a | — | **not a code: it is the noise floor.** Derived: `σ_read²  →  k_BT/k` per mode ⇒ `B_fiber ≈ ½Σ log₂(1 + 2E_orbit/k_BT) + read-length gain` | derived, **not verified here** |

**⇒ Inventory verdict (proven): the fiber contains exactly one storing channel — the local jet of `V_θ`, i.e. of `G = M⁻¹∇V`. Everything else on the Head's list is a key, a field, or noise.** The channel is *rich* (many coefficients), which is why the payload>address budget survives; it is not *plural*.

---

## 4. Item 3 — independence / multiplexing. **They do NOT multiplex as item counts.**

### 4.1 The composition rule

> **`B_total = K_spatial · B_fiber`** — **slots × payload-bits-per-slot.** *Bits* multiply. **Item counts do not.**

**Why (proven).** Two items at one address would require the same location to have two different landscape jets simultaneously. A location has one landscape. The only reader-side freedom (`M`, `p₀`) is address-side: different keys give different *measurements of the same stored jet*, not different stored items. So the fiber converts a slot into a **multi-bit slot**, never into multiple slots.

**Validity conditions (all four must hold; each is a real failure mode):**
1. **Locality of the jet.** `B_fiber` is per-item only if `V_θ` is a localized dictionary (RBF/well atoms). For a global MLP the jets at neighbouring wells are correlated and writes leak (CM-6/D2). **Untested for learned landscapes — the standing D3/N46 risk.**
2. **Parameter ceiling.** `B_total ≤ P·b_θ` (§4.3). This binds hard and is currently unquoted anywhere in the program.
3. **Sub-barrier reading.** `B_fiber` rises with orbit amplitude and is maximal *at* the escape amplitude (§4.4) — the last bits are bought at the isolation boundary (Prop 2).
4. **Read budget.** `B_fiber` is per-launch-set; the numbers below are for 1–8 launches × N=1200 steps *per item read*.

### 4.2 What the fiber is worth (verified-numerically, `item2_capacity.py`)

d=2, jet through order 4 (**12 coefficients**) + `log(M₂/M₁)` = 13 payload parameters, N=1200 at ε=0.05, γ=0, σ_read/A = 1e-3, prior width ρ=0.5 per coefficient. Gaussian-channel bits `B = ½Σᵢlog₂(1+sᵢ²)`.

| launches | coupled `V`: bits (jet / mass) | modes at `s>1` | modes at `s>100` | separable `V`: bits (jet / mass) | modes at `s>1` |
|---|---|---|---|---|---|
| 1 | **129.1** (114.7 / 8.60) | 13 | 9 | 63.6 (49.8 / **0.20**) | **5** |
| 2 | **174.0** (156.9 / 10.56) | 13 | 13 | 91.3 (74.1 / **0.23**) | **6** |
| 4 | 189.3 (171.7 / 12.04) | 13 | 13 | 96.9 (79.3 / 0.23) | 6 |
| 8 | 195.8 (177.8 / 12.47) | 13 | 13 | 99.8 (81.6 / 0.23) | 6 |

**Precision dependence of the same coupled d=2 jet** (N=1200; the sweep that scores P7 honestly — see §6):

| σ_read/A | 1 launch: bits (mass bits, modes at s>1) | 2 launches: bits (mass bits, modes) |
|---|---|---|
| 1e-1 | **51.6** (2.13, 9/13) | 87.7 (3.92, 13/13) |
| 1e-2 | **87.2** (5.30, 12/13) | 130.8 (7.23, 13/13) |
| 1e-3 | **129.1** (8.60, 13/13) | 174.0 (10.56, 13/13) |

≈ **+38 bits per decade of read precision**, and the *number of resolvable directions* is itself precision-dependent (9 → 13). **The registered 20–60-bit band is reached only at a 10 % read-noise operating point.**

**The rank test is exact and is the cleanest confirmation of Prop F1.** Separable arm: 13 parameters − 6 identically-null (mixed monomials are absent from the force) − **1 residual gauge** = **6** identifiable directions. Measured spectrum: 6 modes at `s ≥ 1.9e4`, then a **7-decade gap** to `s ≈ 1e-3` and exact zeros. Coupled arm: 13 parameters − 0 residual gauge (`M₁≡1` fixes the only `λI` gauge) = **13**, measured 13. The counting law `mass = d − k` is confirmed *as a matrix rank*, not just as a trend.

**Launch multiplicity (P8, partially failed as registered — see §6).** The marginal value of launches, against the trivial "more data" floor of `½·13·log₂2 = 6.5` bits per doubling:

| doubling | Δbits | excess over data-doubling |
|---|---|---|
| 1 → 2 | **+44.9** | **+38.4** (real new directions: the weakest mode's `s` rises 6.1 → 343, ×56) |
| 2 → 4 | +15.3 | +8.8 |
| 4 → 8 | **+6.5** | **+0.0** (exactly the data floor — nothing new left to see) |

⇒ **~2 launches suffice at d=2; beyond `O(d)` launches you are only buying `√N`.** (The `O(d)` scaling is **conjectured** — only d=2 measured.)

**Nuisance launch conditions (P9 ✅).** Marginalising `(q₀,p₀)` per launch (query-noise-limited regime, 8 nuisance parameters, query σ=0.05) costs **4.6 %** of the bits (174.0 → 165.9). **Uncertainty about where you launched from is nearly free** — the nuisance directions are not aligned with the jet directions.

### 4.3 The parameter ceiling (proven, and it binds)

`B_total ≤ P · b_θ`, `P` = number of `V_θ` parameters, `b_θ` = bits actually resolvable per parameter (float32 ⇒ ≤32; SGD/precision-realistic ⇒ 10–20).

`PotentialMLP(dim=d, hidden=32)` ⇒ `P = 32d + 1121` ⇒ **1377 params at d=8** ⇒ `B_total ≲ 1.4e4 … 4.4e4` bits.
The measured spatial law `K_max = 4·2^d` gives **1024 wells at d=8**; a *per-well* Hessian at d=8 is 36 reals ⇒ **≥36 864 independent reals ≫ P**.

> **⇒ The `4·2^d` spatial capacity and a per-well fiber are jointly unreachable by the shipped MLP landscape.** `address-space-dimension-scaling` measured `4·2^d` on a **hand-designed analytic** `BallRegisterPotential` (effectively unlimited parameters), so its own result is untouched — but any *learned* CLU inherits this ceiling, and it is the binding constraint on the flagship, not the physics. **This is the number `clu-autoencoder` most needs.**

### 4.4 The one real coupling between the two resources

`B_fiber` depends on the well's **shape** (`A/L`, `σ/A`), not its **size** — so at fixed relative read precision, shrinking wells to pack more of them costs **zero** fiber bits. The coupling enters only through *absolute* read noise (`σ` fixed, `A ∝ w`):

| σ_read/A | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-6 |
|---|---|---|---|---|---|
| bits (1-D, 8-coefficient jet) | 24.3 | 37.9 | 54.7 | 75.8 | 126.6 |
| coefficients at <50 % rel. err | 3 | 4 | 5 | 6 | 8 |

≈ **+4 to +6.4 bits per octave of `σ/A`** (rising, as new coefficients enter). Since `K_spatial ∝ w^{−d_eff}` (power law) while `B_fiber ∝ log w` (logarithm), **shrinking wells is always net-positive for `B_total` until the packing itself fails.** The spatial channel dominates; the fiber is a *logarithmic multiplier*.

**Amplitude sweep — the fiber's bits are bounded by the barrier that guarantees isolation:**

| orbit amplitude `A` | 0.05 | 0.1 | 0.2 | 0.4 | 0.45 | 0.5 | 0.55 | 0.58 |
|---|---|---|---|---|---|---|---|---|
| bits | 24.3 | 29.4 | 39.0 | 54.7 | 59.5 | 65.4 | **75.7** | **escapes** |
| coefficients resolved | 2 | 3 | 4 | 5 | 5 | 6 | **7** | — |

> **The fiber's last bits sit exactly at the escape boundary.** Reading high-order jet coefficients requires large amplitude; large amplitude is what Prop 2's sub-barrier isolation forbids. This is the same trade as boost-retry (Item 4 of w20), spent in the opposite direction: **isolation, retry and fiber depth all draw on one resource — the barrier height `h`.**

---

## 5. Item 4 — the cost side: how long a read buys how many bits

**Derived + verified.** With i.i.d. read noise, each identifiable mode's SNR grows as `sᵢ ∝ N^{αᵢ}` with `αᵢ ∈ [½, 3/2]` — `½` for amplitude-like coefficients, `3/2` for phase-coherent (frequency-like) ones. Hence

```
B(N) = B(N₀) + (Σᵢ αᵢ) · log₂(N/N₀)          ⇒     N(b) = N₀ · 2^{(b−b₀)/Σα}
```

**Verified.** 1-D CRB: `σ_k ∝ N^{−1.518}` (registered −1.5±0.15) and `σ_{q₀} ∝ N^{−0.494}` (registered −0.5±0.10); CRB validated against a 300-trial Monte Carlo (**4.60e-6 measured vs 4.25e-6 predicted**, 8 %). d=2, 13 modes:

| N | 300 | 600 | 1200 | 2400 | 4800 |
|---|---|---|---|---|---|
| bits | 150.3 | 163.0 | 174.0 | 184.5 | 194.9 |

⇒ **`Σα = 11.15` bits per doubling of read length**, mean `α = 0.86` per mode.

> **The honest price.** Read length is **exponential in bits**: doubling the payload from 174 → 348 bits needs `2^{15.6} ≈ 5·10⁴×` the read. The exponent is divided by the number of modes, so the mitigation is **more modes, more launches, more wells — never a longer rollout.** This is the standard analog-channel cost (`½log₂(1+SNR)` per use), not a CLU pathology; the spatial channel pays the same price for site resolution. **It does not make the fiber a curiosity** — at practical read lengths the fiber already delivers 10²-bit payloads — **but it does mean the fiber cannot be scaled by rolling out longer.**

**Read cost per bit, in one line for the engineer:** at d=2, σ_rel=1e-3, the efficient operating point is **2 launches × N≈1200** (174 bits); the 3rd–8th launches and any `N > 2400` are in strongly diminishing returns.

---

## 6. PREREG scorecard (`outputs/relaxation-fiber-capacity/PREREG.md`; P1–P7 before any run, P8–P10 before the Item-2/3 harness)

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | scale gauge exact, ≤1e-12 | **6.2e-16** | ✅ |
| P2a | anharmonicity does **not** break it, ≤1e-12 | **1.0e-15** | ✅ (Hub's proposal refuted) |
| P2b | partial scaling observable, ≥0.10 | **0.499** | ✅ |
| P3a | coupled d=2: `A` and `M₁/M₂` to ≤1e-6 | **4.4e-16 / 1.2e-15** | ✅ |
| P3b | separable: off-diagonals ≤1e-10 ⇒ unidentifiable | **1.3e-16** | ✅ |
| P4 | relativistic breaking `(v/c)²`, slope 2.0±0.25 | **1.989** (Newtonian control exactly 0.0) | ✅ |
| P5a | `σ_ω ∝ N^{−1.5±0.15}` | **−1.518** | ✅ |
| P5b | amplitude channel `N^{−0.5±0.10}` | **−0.494** | ✅ |
| P6 | 1-D resolvable coefficients ∈ [5,9], point est. 7 | **5** | ✅ (at the low edge; my point estimate was 40 % high) |
| **P7** | **d=2 fiber payload 20–60 bits** | **129.1 (1 launch) / 174.0 (2)** at σ_rel=1e-3 | ❌ **FAILED — under-predicted by 2–3×.** Diagnosis: I costed 3 Hessian coefficients at ~8 bits each; the jet has **12** coefficients and each eigen-mode carries `log₂ sᵢ ≈ 15–20` bits at σ_rel=1e-3. Corrected law: `B = ½Σlog₂(1+sᵢ²)`, `sᵢ ≈ (ρ/σ)·√(N·gainᵢ)`. **The failure is in the optimistic direction and it is NOT rescued by assuming realistic precision** (see the sweep below): only at a 10 %-read-noise operating point does the band contain the answer. |
| **P8** | one launch resolves **≤8 of 13** at `s>1`; bits ~linear in launches | **13/13 at `s>1`** (my threshold was far too weak: the weakest mode had `s=6.1 ≫ 1`); at `s>100`: **9/13** ✓-in-spirit. Bits **strongly sublinear**: +44.9 / +15.3 / **+6.5** vs a data-doubling floor of 6.5 | ❌ **FAILED as registered** (both clauses). Mechanism vindicated by the 4→8 doubling landing **exactly** on the data floor, and by the `s>100` count. |
| P9 | nuisance launch marginalisation costs <30 % | **4.6 %** | ✅ |
| P10 | secular accumulation, slope in N = 1.0±0.2 | **1.029 / 1.033** | ✅ |
| composition rule | committed before measuring: one storing channel; `B_total = K·B_fiber`; item counts do not multiply | nothing contradicted it; the rank test (6 = 7−1, 13 = 13−0) is a direct confirmation | ✅ |

---

## 7. Verdict labels

**Proven.** Prop F1 and Corollaries F1.1–F1.3 (the gauge, the `d−k` counting law, the confinement caveat); the reduction to `G = M⁻¹∇V`; "address-side selects, landscape-side stores"; the composition rule `B_total = K·B_fiber` and the impossibility of two jets at one location; the parameter ceiling `B_total ≤ P·b_θ`; `γ` carries 0 per-item fiber bits; the `B(N) = B₀+Σα·log₂N` form.
**Verified-numerically.** Every number in §2–§5 at the quoted precision, at **d ≤ 2, hand-designed polynomial jets, single geometry**: the 1e-15/1e-16 gauge exactness, the rank test (6 vs 13), 129/174/189/196 bits, `Σα = 11.15`, `+2 bits per octave of v/c`, the σ- and amplitude-sweeps, escape between `A/L = 0.55` and `0.58`.
**Conjectured.** That `O(d)` launches suffice at general `d` (only d=2 measured); that the jet stays per-item in a *learned* MLP landscape (contradicted in direction by CM-6 leakage — treat as a **risk**, not a conjecture, until measured); the thermal noise-floor formula `B ≈ ½Σlog₂(1+2E/k_BT)` (derived, unverified); that adding fiber structure to wells leaves the measured `4·2^d` packing unchanged.

---

## 8. Open questions / follow-ups / risks

- **OQ-F1 (the one that gates the flagship).** Everything here is a *capacity* statement about a landscape that already has the jets. **Whether training can put 10²-bit jets into `V_θ` at 10³ wells is untouched, and §4.3 says the shipped MLP cannot hold them.** The decisive experiment is not a bigger fiber demo; it is a **localized landscape parameterization (RBF/well dictionary) with a measured parameter budget**. Recommend as the engineer spec below.
- **OQ-F2.** The mass channel is worth `d−k` dimensions *only in a coupled landscape*, and it is **address-side ⇒ it stores nothing**. If the program wants mass to be a memory channel rather than a key, it must say **where the per-item `M` is stored** — and the answer is "in the codebook", which is an explicit table. Head-level design question, not a physics question.
- **OQ-F3.** `d ≤ 2` only. The two structural results (rank counting, gauge) are dimension-agnostic and exact; the bit-counts are not. A d=4/d=8 replication is cheap (the harness is 200 lines of numpy) and would settle the `O(d)`-launches conjecture.
- **OQ-F4.** All Newtonian except S3. The relativistic fiber capacity (does the `(v/c)²` non-linearity add *jet* directions, or only the one gauge direction?) is underived.
- **Risk.** My P7 miss was in the *optimistic* direction. Any fiber bit-count quoted without `(σ_rel, N, launches, prior width)` is meaningless — the **same d=2 jet reads 52 / 87 / 129 bits** at σ_rel = 1e-1 / 1e-2 / 1e-3, and the 1-D jet spans 24 → 127 bits over the same kind of sweep. **Never quote a fiber bit-count naked.** The precision-independent numbers — and therefore the ones safe for a paper — are the **rank counts** (6 vs 13; `d − k`) and the **scaling exponents** (`Σα = 11.15` bits/doubling; `+2` bits per octave of `v/c`; `+38` bits per decade of σ).

**Engineer spec (precise, for whoever owns the follow-up — I touched no code):**
1. A **localized potential** (`RBFDictionaryPotential`: `V(q) = Σ_k [ −h_k exp(−‖q−c_k‖²/2w²) + ½(q−c_k)ᵀA_k(q−c_k)·mask_k ]`) so per-well jets are literally per-well parameters and the parameter budget is explicit and countable.
2. `mass_override` already exists (`chlu_unit.py:312`) — **no core change needed** for the mass channel.
3. A fiber-read primitive: γ=0 rollout of length N from `L` rest launches around `a*`, returning stacked samples; the decoder is ridge regression (w20 Item-5 verbatim).
4. **Do not** add a "fiber capacity" flag or metric until (1) exists — with an MLP landscape the measurement is capped by §4.3 and would measure the MLP, not the fiber.

---

## Proposed handover updates (for the Hub)

- **§1 (memory formalism) — new proposition.** Adopt **Prop F1 (mass–landscape gauge)**: the deterministic read depends on `(M,V,p₀)` only through `(M⁻¹∇V, M⁻¹p₀)`; the gauge group is block-scaling of dimension `k` = number of separable blocks; **mass contributes exactly `d−k` observable dimensions and ZERO for `d=1` or any separable `V`, harmonic or anharmonic** (verified to 1e-15). Corollary: *inertia is unobservable in principle in the Newtonian kinetic modes*; the **relativistic governor is what gauge-fixes it**, with strength `∝(v/c)²` accumulating `∝N` (slope 1.989 / 1.03 measured). This is a new, quotable role for CHLU's headline contribution.
- **§1 — the fiber, corrected.** The fiber's content is **the local jet of the effective force field `M⁻¹∇V`**, and it is **one storing channel**. `M`, `p₀` = address-side keys (select, don't store); `γ` = spatial field (0 per-item bits); `T` = noise floor. **Amend Prop 11** (`relaxation-addressing-theory` Item 5) accordingly — its `ω_i = √(k_i/M_i)` reading is valid only at fixed `M`.
- **§1/§7 — amend `clu-memory-architecture` Prop 1 escape (b):** anisotropic mass breaks the *scalar* dilation but is **exactly gauge for separable `V`**; mass discriminates **iff the landscape couples coordinates**, via the asymmetry of `M⁻¹∇²V` (`M_i/M_j = (∂G)_{ji}/(∂G)_{ij}`). Measured 8.6 bits coupled vs **0.20 bits** separable.
- **§6/§8 — the composition rule, for `clu-autoencoder` and any paper:** **`B_total = K_spatial · B_fiber` (slots × bits/slot). Item counts do not multiply.** `K_spatial = 4·2^d` stands; `B_fiber` = 129–196 bits/well at d=2, N=1200, σ_rel=1e-3, 1–8 launches — **always quoted with `(σ_rel, N, launches)`**; the same jet is 87 bits at σ_rel=1e-2 and 52 bits at σ_rel=1e-1 (≈ **+38 bits per decade of read precision**).
- **§7 — NEW known issue (the parameter ceiling, currently unquoted anywhere):** `B_total ≤ P·b_θ`. `PotentialMLP(d=8,hidden=32)` has **1377 params**, while `4·2^d` wells with per-well d=8 Hessians need **≥36 864 reals**. **The measured spatial capacity and a per-well fiber are jointly unreachable by the shipped learned MLP.** `address-space-dimension-scaling`'s result is unaffected (hand-designed analytic landscape) but every *learned* extrapolation from it must carry this ceiling.
- **§7 — minor code note:** the shipped `M_inv = 1/(M + 1e-6)` is an explicit gauge-breaking term at relative order 1e-6; a repo-side gauge test will saturate there, not at 1e-16. Also, `PotentialMLP`'s hard-coded `0.05‖q‖²` is not co-scaled with the learned part, so a writer that rescales only the MLP output breaks the gauge weakly (Cor. F1.3).
- **§8 — closes OQ-4** (`relaxation-addressing-theory`): the fiber's quantitative capacity is now `B(N) = B₀ + Σα·log₂N` with `Σα = 11.15` bits/doubling at d=2 (13 modes, mean α=0.86), bounded above by the escape amplitude and by the parameter ceiling. **Opens OQ-F1** (can training put the jets there? the shipped MLP cannot) as the successor question — and it is a *capacity-of-the-writer* question, not a physics question.
- **Owner needed:** the 4-item reconciliation list at the top of this report.
