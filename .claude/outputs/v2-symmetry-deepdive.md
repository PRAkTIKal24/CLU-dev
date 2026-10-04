# v2-symmetry-deepdive — physics-theorist report (running note, **v1.1**)

> **v1.1 (this pass) closes open question O5: the relativistic running decay constant.** New: §7bis (R1–R8), claims **S10–S13**, falsifiables **F-9…F-12**, checks `checks5–8.py`. Headlines: the relativistic latch **stores rapidity, not momentum** (transport logarithmic in the write impulse); a **causal retention–bandwidth bound** `n₁/₂·θ̇_max² = 2γln2·m₀c²/((2−γ)ε²δΣ)`, independent of the decay constant; a **causal memory-lifetime floor** against arbitrary noise (the memory-side payoff of the governor, which the paper never claims); and a **new proven defect — F5 Prop-9's FDT fix is Newtonian-only**, so the coded relativistic Langevin (Exp-C's default) has *no* Gibbs invariant for any σ. Two of my v1.1 statements were also self-corrected (§10, R4–R5).

**Task + acceptance criterion:** HEP-legible running note on V2's symmetry-realization structure — (1) tighten the GMOR/ChPT correspondence and say what full ChPT adds; (2) verdict on the symmetry-realization taxonomy as V2's organizing frame, with the two-axis subtlety untangled; (3) assess *custodial ↔ kinetic-isotropy/Schur*. Deliver a correspondence table, an organizing-frame recommendation, new falsifiables, V4 seeds, honest flags.
**Status:** done (v1.0 of a running note — this is a brainstorm thread; expect the Head to extend it).
**No tracked code touched.** No program doc edited (F5 note, claims matrix, ledger, V2 short all untouched, per task).

**What I did**
- Read: protocol, handover (§1/§2/§7/§8/§10), F5 `formalism-note.md` (Prop-1…16, Cor-13/14/15), `HEP_primers.md` §3–4, `papers/v2-short/draft.md` (full), `v2-full-runs.md`, `future_work.md`, and the ground truth in `chlu/core/potentials.py` (`SO2InvariantPotential`, `TiltedPotential`).
- Derived: the exact GMOR relation with a *measured* condensate; the current-theoretic definition of the CLU decay constant; a **kinetic-spurion blindness theorem** (new); the custodial/ρ_CLU structure and the multiplet-universal GMOR invariant; the exact discrete coset-diffusion constant at T>0 (0+1D Coleman); the Arrhenius channel for pseudo-Goldstones; the compact-register aliasing bound; the non-abelian bank's order/holonomy structure.
- Verified every load-bearing claim numerically (pure-numpy float64 toys of the coded map). **Two of my own conjectures were refuted in the process and are reported as such** (§10).

**How I verified**
`uv run --no-project --with numpy python .claude/scratch/v2-symmetry-deepdive/{checks,checks2,debug,checks3,checks4}.py` — numpy 2.0.2, python 3.9.16, float64 throughout. Full flag-provenance in **Appendix P**. Headline numbers: GMOR exactness `2.2e-16`; flat μ² under random anisotropic M `0.0` (exact); latch drift under anisotropy `0.0` (exact); ρ_CLU deviation `0.0`; GMOR multiplet-universality deviation `0.0`; Poisson algebra deviation `0.0` (200 random states); measured-vs-exact-map half-life `1.5e-10`/`7.1e-9`; diffusion exponents `d log D/d log T = 0.998`, `d log D/d log γ_c = −0.980`; Arrhenius slope/barrier `1.074`; write-ordering exponents `3.000` (equal-arclength) and `5.000` (CLU charge-impulse); holonomy/α² → `1.0001`.

---

## 0. Executive summary (thirteen claims, status-tagged; **S10–S13 are v1.1**)

| # | Claim | Status |
|---|---|---|
| **S1** | With a **linear (ambient) spurion**, GMOR is **exact**, not asymptotic: `μ² F² = δ·Σ`, where `F² = M_ch r*²` (coset inertia) and `Σ = r*(δ)` (the vacuum radius = the condensate). The **shipped `TiltedPotential` (angular tilt) normalizes the condensate away** — it can never measure Σ. | proven; verified `2.2e-16` |
| **S2** | `F` is the CLU's **pion decay constant** in the *defining* sense: `Q = F² θ̇` on the vacuum manifold, so `F² = lim_{p→0} Q/θ̇` — the current–Goldstone coupling at zero momentum. F5's "decay constant f = orbit radius" is off by `√M_ch`. | proven; verified (O(p₀²) tail, `dev/p₀² → 0.475`) |
| **S3** | The **leading low-energy constant is saturated by the radial (σ/Higgs) resonance**: LO-GMOR relative error `= δ/(M_ch μ_rad² r*)`, measurable on any trained checkpoint. | proven; verified (ratio `0.9959`) |
| **S4** | **Kinetic-spurion blindness theorem (new).** For any invertible `M`, `rank(M^{-1/2} K M^{-1/2}) = rank(K)`. Hence a purely *kinetic* breaking of G leaves the vacuum manifold, **every zero mode, and the latch exactly intact**; it perturbs only the *current*. The charge violation is a **bounded oscillation**, not secular drift, with amplitude `√(2E)·r·(√M_max − √M_min)`. | proven; verified (flat μ² `= 0.0`, latch drift `= 0.0`, secular/amplitude `≤ 4.2e-4` over 3×10⁵ steps) |
| **S5** | Therefore: **the CLU latch is a *modulus* phenomenon, not a Goldstone phenomenon.** V-flatness + γ>0 suffices; no symmetry of the full H is required. **Equivariance is sufficient but not necessary for neutral memory** — and V2's own broken-iso battery is already an unremarked counterexample to the necessity direction of Mo's hypothesis. | proven; verified (+ already *measured* in `v2-full-runs` item 5) |
| **S6** | **F5's kinetic-isotropy clause is wrong, and it is unhedged in the arXiv-bound note** (`papers/f5-note/f5-note.tex:124`): *"the channel becomes a pseudo-Goldstone one with μ² ∝ the mass splitting."* Kinetic anisotropy gives `μ² ≡ 0` **exactly**. Blocking correction. | proven; verified; **contradicted by the program's own data** (`μ²_ang ~ 1e-15` in the broken-iso battery) |
| **S7** | **Custodial ↔ kinetic isotropy is tight at tree level, and load-bearing only non-abelian.** The protected quantity is the pNG multiplet degeneracy `ρ_CLU := μ_1²/μ_2² = m_2/m_1`; the surviving invariant is `μ_a² F_a² = δΣ`, **multiplet-universal even when both μ_a and F_a split** (the CLU's `f_K ≠ f_π`). Retention ratio `n_1/n_2 = m_1/m_2`, *independent of δ*. | proven; verified (`ρ` dev `0.0`; universality dev `0.0`; `n₁/₂` vs exact map `1e-10`) |
| **S8** | **The latch's infinite half-life is a T = 0 statement (0+1D Coleman).** At T>0 the coset coordinate diffuses with `D = εT(2−γ)/(2 M_ch r*² γ)` — derived exactly for the flat mode. Consequences: **only temperature erases a flat direction; more friction makes erasure *slower*** (Einstein). A pseudo-Goldstone additionally hops between wells at an Arrhenius rate `∝ e^{−2δr*/T}`. | proven; verified (`D` ratio `0.98–1.03`; exponents `+0.998`, `−0.980`; Arrhenius slope/barrier `1.074`) |
| **S10** | **The relativistic decay constant runs with the write current:** `F_Q²(p) = F_Q²(0)·cosh ζ` (the mode's Lorentz factor), while the coset angular velocity is **capped**: `θ̇_max = c/(√M_ch r*) = c√m₀/F_Q(0)`. GMOR stays exact **at rest** — as in ChPT, where `f_π` is defined at zero momentum. | proven; verified (`F_Q` ratio `0.99999–0.9982`, deviation = centrifugal) |
| **S11** | **The relativistic latch stores *rapidity*, not momentum:** `Δθ = ε θ̇_max Σₙ tanh ζₙ`, `sinh ζₙ = (1−γ)ⁿ sinh ζ₀` → `Δθ → θ̇_max·ζ₀/γ_c`. Transport is **logarithmic in the impulse** (a write impulse of 500 stores `7.68` where the Newtonian latch stores `625.0`). The register performs automatic dynamic-range compression. | proven; verified (exact sum to `2.2e-15`; rapidity slope ratio `1.000000`) |
| **S12** | **Causal retention–bandwidth bound (new).** `n₁/₂ · θ̇_max² = 2γ ln2 · m₀c² / ((2−γ) ε² δΣ)` — **independent of `M` and `r*`, hence of the decay constant `F`.** `F` sets the *trade* (retention `∝F²`, bandwidth `∝1/F`); only `c`, `m₀` or `δΣ` move the product. **The trade-off exists only in relativistic mode.** Thermal analogue with `T` in place of `δΣ`. Plus an assumption-free **causal memory-lifetime floor**: erasure to tolerance `Δ` takes `≥ Δ/(ε θ̇_max)` steps under *any* noise or adversary. | proven; verified (product `681.7/696.0/702.9/706.4` vs `709.7`; residual `= (εμ/γ)²` to 3 digits; `D` saturates with `T` where Newtonian `D` diverges) |
| **S13** | **[NEW DEFECT] F5 Prop-9's FDT fix is Newtonian-only.** The O-step `p←(1−γ)p+σξ` is a *linear* OU recursion, so its stationary momentum law is **exactly Gaussian**. The relativistic Gibbs momentum marginal is **Maxwell–Jüttner**. Hence **no σ gives the coded relativistic Langevin a Gibbs invariant.** Defect controlled by `T/(m₀c²)`; **Exp-C's default starts at `T/(m₀c²) = 1`** (`Var_MJ = 2.70·M_eff T`, `KL = 0.38` nats). | proven (free-particle: the p-recursion is autonomous & linear); verified (q-marginal error depends on `T/(m₀c²)` **alone** — two configs at ratio 8 give bit-identical `−0.7290074`) |
| **S9** | **Non-abelian bank (V4).** Write currents do not commute (`{Q_a,Q_b} = pᵀ[X_a,X_b]q`, exact). But `S² = SO(3)/SO(2)` is a **symmetric space** (`[m,m] ⊂ h`), so write-ordering **rotates the register basis (holonomy = α², Gauss–Bonnet) without corrupting the stored value** (ordering error `O(α³)` for equal-arclength writes; **`O(α⁵)` for the CLU's own charge-impulse write**). Cross-talk is **multiplicative, in the addressing**: `F_φ²(θ) = M f² sin²θ`. | proven; verified (algebra `0.0`; exponents `3.000`/`5.000`; holonomy/α² `1.0001`; gain law to `1e-5`) |

**One-sentence thesis for the Head/co-author.** *V2's memory is a modulus with a Goldstone's paperwork*: the flat direction is protected by the **potential's** flatness (so it survives kinetic symmetry breaking), while the **symmetry** protects the *current* that writes it — and once you say it that way, GMOR becomes exact with a measurable condensate, custodial symmetry becomes a retention-anisotropy prediction, and the latch acquires the finite-temperature lifetime it was always missing.

---

## 1. The two-axis untangling (task item 2, the "3 modes" subtlety)

The Head's three modes (∞ / overdamped / underdamped) **mix two independent axes**. They must be separated or the taxonomy will mislead a physics reviewer.

- **Axis 1 — symmetry realization.** Sets `μ²` (the spectral mass), a property of `V_θ` and the vacuum. Purely a statement about the *potential and its symmetry*; no dynamics, no `γ`, no `ε`.
- **Axis 2 — the map's regime.** Given `μ²`, the pair `(εμ, γ)` decides what the damped-Verlet map *does* with it (F5 §3.3/§3.4). Purely a statement about the *integrator + dissipation*; no symmetry.

The exact phase table (all entries proven in F5 or here):

| Axis-1 realization | `μ²` | Axis-2 cell | behaviour | half-life |
|---|---|---|---|---|
| **Wigner–Weyl** (unbroken; vacuum is G-invariant) | `μ² > 0`, **degenerate within each irrep** (Schur) | any | massive multiplet, no register | budget table (§3.4) |
| **Nambu–Goldstone** (SSB; vacuum on an orbit) | `μ² = 0` exactly | `γ = 0` | **marginal drifting integrator** | ∞ but *unfrozen* (drifts) |
| ″ | `μ² = 0` | `γ > 0`, `T = 0` | **latch** (frozen displacement) | **∞** |
| ″ | `μ² = 0` | `γ > 0`, `T > 0` | **diffusive register** | `≈ Δ²/(2Dε)`, `D = εT(2−γ)/(2M_ch r*²γ)` ← **new (S8)** |
| **pseudo-Goldstone** (explicit breaking δ) | `μ² = δΣ/F² > 0`, small | `εμ ≲ γ/2` | overdamped register | `≈ 2γln2/((2−γ)(εμ)²)` |
| ″ | ″ | `εμ ≳ γ/2` | underdamped working memory | `2ln2/(−ln(1−γ))` (mass-independent floor) |
| ″ | ″ | `T > 0` | + Arrhenius inter-well hopping | `∝ e^{+2δr*/T}` ← **new (S8)** |
| — (saddle) | `μ² < 0` | any `γ` | expanding; never stabilized | F5 Prop-3.3(d) |

**Reading.** "∞" is a *joint* (axis-1 ∧ axis-2) statement: `μ = 0` **and** `γ > 0` **and** `T = 0`. "Overdamped/underdamped" is *pure axis-2* at fixed `μ > 0`. The Head's three modes are therefore one axis-1 cell and two axis-2 cells — not three of a kind. The clean statement:

> **Symmetry realization sets `μ²`. The map's `(ε, γ, T)` sets what `μ²` buys.**
> The mode-mass budget table is the *axis-2* law. The realization taxonomy is the *axis-1* law. V2 currently ships axis-2 with axis-1 implicit.

---

## 2. GMOR / ChPT, done properly (task item 1)

### 2.1 The setup is literally a linear sigma model

`SO2InvariantPotential` computes `V = f_θ(r²) + α r²` on the channel — a learned radial profile of the invariant `r²`. That **is** the O(2) linear sigma model with a learned potential. The identifications below are therefore *structural*, not analogical.

### 2.2 The three objects, and how each is measured

Work on the channel, vacuum radius `r*`, channel inertial mass `M_ch`, coset angle `θ`.

| ChPT object | definition in ChPT | CLU object | how to measure it in CLU |
|---|---|---|---|
| pion `π` | Goldstone of χSB | coset coordinate `θ` | the latched register |
| decay constant `f_π` | `⟨0|A_μ|π⟩ = i f_π p_μ` (current–Goldstone coupling) | **`F² = M_ch r*²`** | (a) geometry: `M_ch r*²`; (b) **dynamics: `F² = lim_{p→0} Q/θ̇`** |
| quark mass `m_q` | explicit-breaking spurion, linear in the field | tilt `δ` | probe parameter |
| condensate `Σ = |⟨q̄q⟩|` | `−∂E_vac/∂m_q` | **`Σ = r*(δ)`**, the VEV / order parameter | `−∂E_vac/∂δ`, or just the measured vacuum radius |
| GMOR | `m_π² f_π² = 2 m_q Σ` | **`μ² F² = δ Σ`** | three independent measurements, one identity |

`Q = F² θ̇` holds exactly on the vacuum manifold, so **`F` is the CLU's decay constant in the *defining* sense**, not by analogy. Verified: `F²_dyn → M_ch r*²` with an `O(p₀²)` centrifugal tail (`dev/p₀² = 0.4745 → 0.4762`, converged).

> **Nomenclature correction for F5.** F5 §3.3(c)/§8 call `f` (the orbit radius) "the decay constant." The object that appears in GMOR and in the current relation is `F = √(M_ch)·r*`. F5's own formula `μ² = δn²/(M_eff f²)` is consistent with `F² = M_eff f²`; only the *name* is misplaced. Proposed: reserve **decay constant `F`** for `√(M_eff)·r*`, and call `r*` the **vacuum radius / condensate `Σ`**.

### 2.3 GMOR is *exact*, and the shipped tilt hides the condensate

With the **linear (ambient) spurion** `V → V − δ·q_1` (the ChPT quark-mass term), stationarity gives `λ(r*²−f²)r* = δ`, hence the angular curvature at the vacuum is **exactly** `K_θθ = δ/r*`, so

```
μ² = δ /(M_ch r*)      ⇒      μ² F² = δ · r*  =  δ · Σ      [EXACT, all δ]
```

Verified to `2.2e-16` over `δ ∈ [1e-8, 0.3]` (checks2 §A′; the earlier `3.9e-10` was catastrophic cancellation in `r*²−f²`, fixed by solving Newton in `u = r*−f`).

With the **shipped angular spurion** `TiltedPotential`: `V → V + δ(1 − cos nθ)`, which is *radius-independent*, so `μ² F² = δ n²`. The "condensate" is the pure number `n²`. **The shipped probe cannot see Σ**; it measures the product `m_q Σ` as a single knob. This is not an error — it is the correct way to *verify a power law* — but it is why V2's GMOR reads as "a tilt produces `μ² ∝ δ`" rather than as GMOR proper.

### 2.4 What the full ChPT machinery adds

1. **NLO / the running condensate.** `Σ(δ) = r*(δ)` runs with the spurion; LO-GMOR (which uses `Σ(0) = f`) therefore has a **predicted** relative error
   ```
   (μ²_LO − μ²)/μ² = u/f = δ/(2λf³) = δ / (M_ch μ_rad² r*)          [all quantities measurable]
   ```
   Verified: measured/predicted `= 0.9959` at `δ = 1e-2` (checks2 §A′). **The leading LEC is `1/(M_ch μ_rad² r*)` — saturated by the radial (Higgs/σ) resonance.** This is exactly ChPT's *resonance saturation of low-energy constants*, and in the CLU it is exact rather than phenomenological. It is also precisely F5 §4.3 route-(iii)'s statement that HEFT is the `μ_rad → ∞` decoupling limit with `1/μ_rad²` corrections — now with a measured coefficient.
2. **The chiral expansion parameter** is `μ²/μ_rad²` (pNG mass² over radial mass²) — the CLU's `m_π²/m_σ²`. Retention corrections are organised in powers of it. This *predicts* where the measured anharmonic bias should appear: on the softest-`μ²` seed, which is exactly what `v2-prefreeze-baselines` item 4 found (seed 43: softest mode, ratio runs `1.07 → 1.55` with kick amplitude).
3. **Sum rules (Gell-Mann–Okubo).** With `n_G = dim(G/H) ≥ 2` and a spurion in a definite irrep, the pNG masses obey linear relations. Since `n₁/₂ ∝ 1/μ²` (overdamped), these become **sum rules among *inverse retention half-lives*** — e.g. an SU(3)-flavour-like spurion gives `4/n_K = 1/n_π + 3/n_η`. A designed register bank could test this. (V4.)
4. **`f_K ≠ f_π`.** Kinetic anisotropy splits the decay constants `F_a² = M_a r*²` while GMOR holds **mode-by-mode with a common Σ** (§4). Verified: `μ_a² F_a²` equal to `0.0` across a split multiplet.
5. **Finite-T ChPT / chiral restoration.** The condensate melts; `F → 0` at restoration and `μ² = δΣ/F²` diverges ⇒ **memory dies at the transition**. V2 has already *measured* this transition (§3).
6. **Lattice/Symanzik.** F5's shadow Hamiltonian **is** the Symanzik effective Hamiltonian; `O(ε²)` is a lattice artifact renormalising `ω`; F5 Prop-8 (a symmetry of `T` and `V` protects the flat direction under discretisation) is the statement that an **exactly realised lattice symmetry forbids symmetry-breaking counterterms**. This gives the `ε²` leg of the power counting for free, with a citable literature anchor.

### 2.5 What ChPT does **not** give us (honest limits)

- **No loops, no chiral logs.** The CLU is 0+1D classical mechanics: there is no `ħ`. The two genuine expansion parameters are `ε²` (Symanzik/lattice) and `T` (thermal), plus `1/F²` for coset curvature. Chiral logarithms come from massless propagators in `d ≥ 2`; **they have no CLU analogue**. Do not put "one-loop ChPT" in a paper.
- **No anomalies / no WZW term.** No fermions. Topology enters instead through `π₁(G/H)` (register compactness, §6) and coset curvature (holonomy, §7). *Do not reach for anomalies.*
- **The emergent arm's power counting is marginal.** Its self-breaking `δ_eff ≈ 0.01–0.06` against a ring depth `≈ 0.08` gives an expansion parameter that is *not* parametrically small; "pseudo-Goldstone" is the right word, but ChPT power counting there is `O(1)`-ish — consistent with the measured `+5…+29%` anharmonic spread.

---

## 3. The symmetry-realization taxonomy: **verdict = adopt, as an overlay** (task item 2)

### 3.1 The decisive argument: V2 has already measured all three realizations

This is not a rebranding. **The three realizations are the three empirical regimes the V2 short already reports**, and one of them is currently unnamed:

| realization | V2's arm | evidence already in hand |
|---|---|---|
| **Nambu–Goldstone** | designed `so2_invariant` | flat `μ² ≈ 1e-15`; latch `∞`, 5/5 seeds |
| **pseudo-Goldstone** | emergent MLP (self-broken washboard) + analytic tilts | `δ_eff ≈ 0.01–0.06`; `n₁/₂ ∝ 1/μ²`, slope `−0.985` |
| **Wigner–Weyl (unbroken)** | **the eroded vacuum** (`v2-full-runs` Finding 0) | `r* = 0`, data ring becomes a local max, **"channel μ² pair degenerate (~1.0)"** |

That last row is the tell. **A degenerate doublet at a symmetric vacuum is Schur's lemma**, i.e. the Wigner–Weyl realization — the paper reports it as a training pathology and never names the physics. And then:

> **Sleep-CD erosion is a symmetry-restoration (condensate-melting) transition.** `r*` is *literally* the order parameter; `r* → 0` is *literally* the restoration of the G-symmetric vacuum. The `V(data)` anchor is what holds the condensate up. This is structurally exact, **not** a metaphor.

So the taxonomy (i) classifies three regimes the paper already has, (ii) names the erosion result in physics terms, (iii) sharpens the Mo contrast (his neutrality theorem lives entirely inside the NG row), and (iv) costs **zero new experiments** for the framing itself.

### 3.2 Recommendation (Hub decides; I made no edits)

**Adopt as a framing overlay in §1/§2 + one short subsection; do NOT restructure §3.**
- Keep the **mode-mass budget as the quantitative spine** — it is axis-2 and it is what is *measured*.
- Add the axis-1 taxonomy as the organizing frame: *symmetry realization sets `μ²`; the map decides what `μ²` buys.*
- Re-label §3.5 (erosion) as a **symmetry-restoration transition** with `r*` as the order parameter. This is a large clarity win for a physics reviewer and a free narrative arc (SSB → the training destroys the SSB → the anchor restores it).

### 3.3 Obligations this incurs (four; all meetable, one needs a cheap measurement)

1. **Define SSB for a finite-dimensional deterministic system.** Required, one paragraph. We use only the *classical/tree-level* Goldstone theorem: the minimiser is not G-invariant ⇒ the Hessian has a zero eigenvalue along the orbit. That is exact and is all we ever use. A physicist *will* say "there is no SSB in a 0-dimensional system." Pre-empt it.
2. **State Coleman/Mermin–Wagner honestly.** At `T > 0` there is **no** long-range order: the coset coordinate diffuses (S8). The latch is exact at `T = 0`; at `T > 0` it is a diffusive register with a computed lifetime. We now *have* the law, so this obligation is an asset, not a liability. (It also explains Thread-9's measured coset random walk from first principles.)
3. **Do not import loop-level ChPT.** §2.5.
4. **One cheap new measurement (F-3 below):** quantify the doublet splitting `|μ₁²−μ₂²|/μ̄²` at the collapsed vacuum. *Caveat:* for `so2_invariant` the degeneracy is **architecturally guaranteed** (any smooth invariant function of `r²` has Hessian `∝ I₂` at the origin), so on the designed arm this is a *pipeline check*, not a discovery. The interesting version is on the **emergent** arm, where the splitting is a genuine, novel attribution instrument for explicit breaking *in the unbroken phase* — complementary to Mo's `E^V_eq`.

**Cost/risk to weigh.** A reviewer may read "SSB" as claiming a thermodynamic phase transition. The predictive content that defuses this: Schur degeneracy in the unbroken phase, exact GMOR with a *measured* condensate, `ρ_CLU`, and the diffusion law. If the Hub does not want to add obligation (1)–(2), do **not** adopt the frame — half-adopting it is worse than the current mass-first framing.

---

## 4. Chiral vs custodial (task item 3)

### 4.1 The kinetic-spurion blindness theorem (new, and it changes a design rule)

**Theorem.** Let `V_θ` be G-invariant with a vacuum manifold `G/H`, `K = ∇²V(q*)`, and let `M ≻ 0` be *any* inertia (not necessarily commuting with the G-action). Then
```
ker(W) = M^{1/2} ker(K),      dim ker(M^{-1/2} K M^{-1/2}) = dim ker(K).
```
*Proof.* `M^{±1/2}` is invertible; congruence by an invertible matrix preserves rank. ∎

**Corollaries.**
- (i) The vacuum manifold is untouched (`M` does not enter `V`).
- (ii) **Every Goldstone direction stays exactly flat**, `μ² = 0`, for any anisotropy.
- (iii) The `γ>0` **latch survives exactly** (the per-mode map is `[[1, ε/m],[0,1−γ]]` in the canonical basis).
- (iv) Only the **current** notices: `M` deforms the induced metric on the orbit. On the ring, `F²(θ) = r²(M_0 sin²θ + M_1 cos²θ)`, and since `E = ½F²(θ)θ̇²` is conserved while `Q = F²(θ)θ̇ = √(2E)·F(θ)`, the Noether charge **oscillates between `√(2E) r √M_min` and `√(2E) r √M_max` — bounded, not secular.**

Verified: flat `μ² = 0.0` exactly for 6 random anisotropic `M`; latch drift `= 0.0` exactly over 6000 steps; charge-oscillation amplitude vs the closed form `A_pred = √(2E)·r·(√M_max−√M_min)` gives ratio `1.019–1.038` (residual = centrifugal orbit shift `r > f`), while the **secular shift of the oscillation centre over 3×10⁵ steps is `≤ 4.2e-4` of the amplitude** (checks2 §C′).

**Three consequences, in increasing order of importance:**

- **(a) F5 contains a proven error, and it is *unhedged in the arXiv-bound note*.** Three sites, escalating:

  | site | text | severity |
  |---|---|---|
  | `outputs/formalism-note.md:233` (§4.1) | "*Falsifiable corollary for V2: … If not, the channel is pseudo-Goldstone with `μ² ∝` mass splitting.*" | hedged as a falsifiable — but tagged **[proven; verified (g)]** |
  | `outputs/formalism-note.md:435` (§9 summary) | repeats it as a design constraint + falsifiable | hedged |
  | **`outputs/f5-arxiv-note.md:139` → `papers/f5-note/f5-note.tex:124`** | "*…and the channel becomes a pseudo-Goldstone one with `μ² ∝` the mass splitting.*" | **flat assertion, arXiv-bound, cited by both shorts** |

  This is false: `μ² ≡ 0` for **any** invertible `M` (rank/congruence, above; verified `0.0` exactly). It is **already contradicted by the program's own measurement** (`v2-full-runs` item 5: `μ²_ang ~ 1e-15` on the broken-iso battery, angular `n₁/₂ = ∞`, write-freeze `= 0` exactly).

  **Two aggravating details.** (i) The **`[proven; verified (g)]` tag over-covers**: check (g) verifies only the *charge* half ("with `M=(1,2)` the charge drifts by 2.6"), never the `μ² ∝ splitting` half — which is the false half. (ii) Check (g)'s own "drifts by 2.6" is very likely the **bounded-oscillation envelope**, not secular drift (F-2): for `M=(1,2)` the predicted amplitude is `√(2E)·r·(√2−1) = 0.414·√(2E)·r`. **F5 is the Hub's #1 critical-path item; this is a blocking correction, and fixing it *strengthens* the note** — the blindness theorem is a better result than the false corollary it replaces.
- **(b) `A/‖[M,X]‖` is a leading-order proxy, not a law.** V2 Appendix C says *"the Noether-charge drift scales linearly with the split."* The measured constant `≈ 0.7 ≈ 1/√2` is exactly what `‖[M,X]‖_F = √2|ΔM|` predicts for a quantity linear in `|ΔM|` — but the *exact* statement is a **bounded oscillation** with amplitude `√(2E)r(√M_max−√M_min)`, and my toy shows `A/‖[M,X]‖` is **not** constant (`0.096 → 0.076 → 0.062` as `ΔM` grows). Wording should say "bounded charge oscillation, amplitude linear in the split at small split."
- **(c) The latch is a *modulus*, not a *Goldstone*.** Define:
  - **Goldstone** = flat direction protected by a symmetry of the **full** `H = T + V` (needs kinetic isotropy).
  - **Modulus** = flat direction of `V` alone (protected by `V`'s symmetry, or accidental).

  The CLU latch requires only V-flatness + `γ>0`. **Equivariance is sufficient but not necessary for neutral memory.** F5 Prop-16 (Mo's Thm-1 adapted) assumes an equivariant map; the broken-iso battery is *not* equivariant (`E^T_eq = 0.0016–0.108`) and latches anyway. **V2 therefore already holds an unremarked counterexample to the necessity direction** — a free, sharp, zero-cost differentiator against Mo. The symmetry buys the *charge* (the write current); the potential's flatness buys the *register*.

### 4.2 Is "custodial ↔ kinetic isotropy" tight?

**Verdict: tight at tree level, in the pNG sector, and load-bearing only for non-abelian G.** Rated *structurally tight*, with two named disanalogies.

**Where it is tight.**
- Both are a symmetry of *one sector* of the Lagrangian, spoiled by *another sector*: SM — a symmetry of the Higgs potential (`SO(4)`), spoiled by the gauge (`g′`) and Yukawa (`y_t−y_b`) sectors; CLU — a symmetry of `V_θ`, spoiled by the kinetic sector (`M`).
- In both the protected object is a **degeneracy within a multiplet of the unbroken subgroup**, forced by Schur.
- The breaking parameter is a **coupling asymmetry**: `g′`, `y_t−y_b` ↔ `‖[M,X]‖`, equivalently `m_a/m_b − 1`.

**The CLU ρ-parameter.** Take `G = SO(3) → H = SO(2)` (coset `S²`, two Goldstones forming a 2D irrep of `H`), with an `H`-preserving linear spurion `δ` along the vacuum axis. Then `K_coset = (δ/r*)·I₂` exactly, so
```
μ_a² = δ/(M_a r*)            F_a² = M_a r*²            μ_a² F_a² = δ r* = δΣ   (a = 1,2)
ρ_CLU := μ_1²/μ_2² = m_2/m_1                  [ = 1 iff kinetic isotropy, by Schur ]
```
Verified: `ρ_CLU` deviation `0.0`; multiplet-universality of `μ_a²F_a²` deviation `0.0`.

**The measurable payoff — a custodial-breaking retention-anisotropy prediction.** Since `n₁/₂ ∝ 1/μ²` (overdamped),
```
n_{1/2,1} / n_{1/2,2} = μ_2²/μ_1² = m_1/m_2       — independent of δ, λ, r*
```
i.e. **within one multiplet, the retention ratio equals the inertial-mass ratio.** Verified on the actual damped map: measured `n₁/₂ = 413.99 / 1056.23` steps against the exact 2×2-map values `413.99 / 1056.23` (`|meas/exact − 1| = 1.5e-10`, `7.1e-9`); the *asymptotic* ratio `0.3920` vs `m_1/m_2 = 0.4000` differs only by the `O(h²)` finite-step correction. **Heavier inertial mass ⇒ longer retention, within a multiplet, at fixed spurion.** That is exactly the falsifiable the task asked for (§8 F-4).

**Where it is loose (name these, don't hide them).**
1. **Origin.** SM custodial `SU(2)_V` is *accidental* — nobody imposed it on the Higgs potential. CLU's G-invariance is *designed* (`SO2InvariantPotential`). Same protection mechanism, different provenance.
2. **Order in perturbation theory.** Custodial breaking shows up in observables at *loop* level (the `T`-parameter, `ρ − 1 ∝ g′²/16π²`). The CLU's kinetic spurion is **tree-level and exact**. Do not import the oblique-parameter machinery — there are no loops (§2.5).
3. **For abelian `SO(2)` it is degenerate, as the task suspected — but not vacuously.** With `dim(G/H) = 1` there is no mass ratio to protect, so `ρ_CLU` does not exist. What kinetic isotropy still protects at `SO(2)` is **exact conservation of `Q`** — and by S4 that is *only* about the current, never about the register. Confirmed: load-bearing (as a *retention* statement) only at V4.

**The protected quantity, stated precisely.** Two different observables, two different regimes:

| regime | protected by kinetic isotropy | breaking parameter | observable |
|---|---|---|---|
| `δ = 0`, any `G` | exact conservation `Q_n = (1−γ)^n Q_0` | `‖[M,X]‖` | bounded charge oscillation (**not** the register) |
| `δ ≠ 0`, `dim(G/H) ≥ 2`, `H` irreducible on the coset | **`ρ_CLU = 1`** (pNG mass degeneracy) | `m_a/m_b − 1` | **retention anisotropy across the register bank** |

---

## 5. Finite temperature: the latch's missing lifetime (0+1D Coleman)

**Derivation (exact for the flat mode of the coded map).** Along a flat direction the kicks vanish, so the coded Langevin step reduces to
`q_{n+1} = q_n + (ε/m)p_n`, `p_{n+1} = (1−γ)p_n + σ*ξ_n` with the FDT noise `σ*² = mTγ(2−γ)` (F5 Prop-9). `p_n` is a stationary AR(1) with `Var = mT`, so `Var(Σ_{n<N} p_n) = N·mT·(2−γ)/γ + O(1)` and

```
⟨Δq²⟩ = 2 D N ε ,        D_disc = ε T (2−γ) / (2 m γ)          [flat mode, exact]
```
(continuum Einstein `D = T/(γ_c m)` agrees to `O(γ²)`). On the ring, `Δθ = Δs/r*` and `m → M_ch`, so
```
D_θ = ε T (2−γ) / (2 M_ch r*² γ) = ε T (2−γ) / (2 F² γ)
```
**Verified** (checks3, 5 seeds, 4000 walkers): free-particle control ratio `0.984 ± 0.016`; ring (`λ=120`) ratio `1.020 ± 0.020`; exponents `d log D/d log T = 0.998` (predicted `+1`), `d log D/d log γ_c = −0.980` (predicted `−1`).

**Three consequences, all program-relevant.**

1. **The latch has a finite half-life at `T > 0`:** `n₁/₂ ≈ Δ²/(2 D_θ ε)` for a tolerance `Δ`. F5's `∞` is a `T = 0` statement. This is the *first-principles* explanation of Thread-9's measured coset random walk (`thread9-mh-kernel`, CM-14), and it supplies the missing constant.
2. **Only temperature erases a flat direction.** F5 Cor-13 already proved friction cannot delete coset content. Now: `D ∝ T` and `D ∝ 1/γ_c`, so **increasing friction makes thermal erasure *slower*** (Einstein). A friction field `γ_φ(q)` is therefore *provably* the wrong lever for deleting latched memory — which is precisely why the `γ_φ` rung recovered `−24%` of the fit gap (`fit-gap-anatomy`, N12/N13). **The trash-region program needs a learned *temperature* field `T_φ(q)`, not (only) a friction field** — pleasingly, the original Thread-1 Hawking framing. Erasure rate `∝ T_φ/(γ_c F²)`.
3. **The pseudo-Goldstone gets a second, exponential forgetting channel.** In a tilted well the angle is confined (`⟨θ²⟩ = T/(δ r*)`, verified ratios `1.048`, `1.094`) and hops between wells at an Arrhenius rate `∝ exp(−2δr*/T)`. Verified by censoring-robust survival fits: `d ln k/d(1/T) = −0.326` vs barrier `2δr* = 0.304` (**ratio `1.074`**). So at `T>0` the budget table gains a knob whose lever is *exponential*, not power-law:
```
1/n_total  ≈  1/n_relax(μ, γ, ε)  +  1/n_hop(δ, r*, T)
```
**The V2 budget table is the `T = 0` face of a `(μ, γ, T)` budget cube.**

---

## 6. Register capacity is topological (`π₁(G/H)`)

F5's latch transport law, in coset-angle form: `Δθ = ε p₀/(M_ch r* γ)`. Verified to `1.3e-4` across `γ ∈ {0.008, 0.02, 0.05}` (ratio `0.99987` at every `γ`).

Because the coset is `S¹` (`π₁ = ℤ`), the register has a **bounded dynamic range**: writes wrap at
```
p_crit = 2π M_ch r* γ / ε
```
Above `p_crit` the stored value **aliases**. Two consequences worth a sentence in any capacity discussion: (i) a register's dynamic range is set by `γ/ε`, so the same knob that sets *forgetting* sets *capacity*; (ii) a non-abelian coset with `π₁ = 0` (e.g. `S² = SO(3)/SO(2)`) has **no aliasing**, but is capacity-limited by curvature instead (§7). This is a genuinely different capacity mechanism for the two coset types, and it is free.

---

## 7. The non-abelian bank (V4 seeds) — and a refuted conjecture

**H1 (proven; verified `0.0` over 200 random states).** `{Q_a, Q_b} = pᵀ[X_a,X_b]q`, so the **write currents obey the non-abelian algebra**: a register bank is a group, not a vector space.

**H2 — the cross-talk is multiplicative and lives in the metric (proven; verified to `1e-5`).** On `S²` with coset coordinates `(θ, φ)`, the induced metric is `f²(dθ² + sin²θ dφ²)`, so the **φ-register's decay constant depends on the θ-register's stored value**:
```
F_φ²(θ) = M f² sin²θ
```
Hence, for a *fixed Cartesian momentum kick*, the write gain is `Δφ ∝ 1/(f sinθ)` (verified: `1.15465` vs `1/sin(π/3) = 1.15470`; `1.41403` vs `1.41421`); for a *fixed charge injection* `p_φ`, `Δφ ∝ 1/(M f² sin²θ)`. **Idle (latched, `θ̇ = 0`) registers do not interact** — this is ChPT derivative coupling: Goldstone interactions vanish at zero momentum. So:

> **Non-abelian cross-talk is multiplicative (gain/addressing), not additive (content), and it is switched off while the registers are idle.**

**H3 — my conjecture was wrong; the truth is better (refuted → replaced, both verified).**
I conjectured the endpoint separation between write orders scales as `α²` (the naive `[X₁,X₂] ≠ 0`). **Refuted.** `S² = SO(3)/SO(2)` is a **symmetric space**: `[m,m] ⊂ h`, so the leading commutator lies in the **stabilizer of the base point** and rotates the *frame*, not the *point*. Measured (exact geometry, and the damped map reproduces it to `2.5e-15`):

| write protocol | endpoint ordering error | verified exponent |
|---|---|---|
| equal arclength `(α, α)` | `O(α³)` | `2.972 → 3.000` |
| **CLU's own charge impulse** `p = a·Xq` ⇒ legs `(α, α·cos α)` | **`O(α⁵)`** | `4.942 → 5.000` |
| closed loop, frame rotation (**holonomy**) | `α² × K·Area` (Gauss–Bonnet) | `rot/α² = 1.0006 → 1.0001` |

The `α⁵` is not an accident: the CLU's impulse write has a leg length `∝ |Xq|`, which shortens by exactly `α³/2` on the second leg and **cancels the `α³` ordering term**. So:

> **V4 design claim.** On a symmetric coset, the *stored value* of a non-abelian register bank is nearly order-independent (`O(α⁵)` under the CLU's native write); what write-ordering changes is the **basis in which subsequent writes are addressed** (holonomy `≈ α²`). Registers are *far more independent* than `[X_a,X_b] ≠ 0` suggests.
>
> **Corollary design rule.** For an **independent** register bank, choose an **abelian (flat) coset** — a torus `T^n = U(1)^n` — where the holonomy vanishes identically and the registers are exactly independent (at the price of `π₁ = ℤⁿ` aliasing, §6). Choose a curved non-abelian coset only if the addressing curvature is *wanted*.

This is the sharpest V4 seed in this note, it is proven+verified, and it *contradicts the intuition* that non-abelian ⇒ messy interference.

---

## 7bis. The relativistic running decay constant (v1.1 — open question **O5, now closed**)

All of §1–§7 assumed `newtonian_learned` (V2's actual channel). Here `T(p) = c√(pᵀM⁻¹p + m₀²c²)`, code ground truth, `M_eff = m₀M` at rest.

### R0. Rapidity is the natural coordinate

For one coordinate of inertial mass `M`, set `p = √M · m₀c · sinh ζ`. Then
```
S := √(p²/M + m₀²c²) = m₀c·cosh ζ ,     v = ∂T/∂p = (c/√M)·tanh ζ
```
so `v_max = c/√M` (F5 Prop-1 recovered) and **ζ is the mode's rapidity** — the same variable V1's squeeze/boost ladder uses.

### R1–R2. The decay constant runs; the write speed is capped

On the vacuum ring with tangential momentum, `Q = r*·p` and `θ̇ = |q̇|/r*`, so exactly
```
F_Q²(p) := Q/θ̇ = M_eff r*² · cosh ζ = F_Q²(0) · γ_L(p)
θ̇_max   = c/(√M_ch r*) = c√m₀ / F_Q(0)          [= c/F_Q(0) at the code default m₀ = 1]
```
**The decay constant is Lorentz-enhanced by the write current.** Verified on a stiff ring: `F_Q²_meas / (M_eff r̄² cosh ζ) = 0.99999 → 0.9982` for `p₀ = 0.2 → 10` (residual = the O(v²) centrifugal excursion `r > r*`), and `θ̇ → 0.992·θ̇_max` at `ζ = 3.0`.

**The write *current* `Q = r*p` is unbounded; the write *speed* `θ̇` is capped.** They decouple, and `F_Q` is the ratio.

> **Honest flag.** This is **not** a ChPT form factor. Lorentz covariance fixes `⟨0|A_μ|π(p)⟩ = i f_π p_μ` to be exactly linear in `p` — there is no running. Our `cosh ζ` is the relativistic **rotor**: the moment of inertia of the vacuum angle is `γ_L`-enhanced. Use "running decay constant" as a CLU term, not as a QCD claim.

### R3. The relativistic latch stores **rapidity**, not momentum

Along an exactly flat direction the kicks vanish, so `pₙ = (1−γ)ⁿp₀` exactly and
```
Δθ = ε·θ̇_max · Σ_{n≥0} tanh ζₙ ,      sinh ζₙ = (1−γ)ⁿ sinh ζ₀        [EXACT]
```
Verified against the simulated map on a gutter potential (an exactly flat direction): `max |sim/closed-form − 1| = 2.2e-15`.

Limits:
- **Newtonian** (`ζ₀ ≪ 1`): `Σ tanh ζₙ → ζ₀/γ` ⇒ `Δθ = εp₀/(M_eff r* γ)` — F5's latch law, recovered.
- **Ultra-relativistic** (`ζ₀ ≫ 1`): `Σ tanh ζₙ → ζ₀/(−ln(1−γ))` ⇒ **`Δθ → θ̇_max · ζ₀ / γ_c`**, i.e. *linear in rapidity, logarithmic in the impulse*. Verified: `d(Σtanh)/dζ₀ × (−ln(1−γ)) = 1.000000`.

Concretely (`M=0.8, ε=γ=0.05`): a write impulse `p₀ = 500` stores `Δθ = 7.678`, where the Newtonian latch would store `625.0`. **The causal cap turns the register into a logarithmic (companding) encoder of the write.**

*Scope:* exact for a flat direction with stiff transverse confinement. On a **curved** coset a hard write leaves the vacuum manifold (centrifugal excursion `∝ (θ̇r*)²`), so the law acquires a geometric correction growing with `ζ` — visible as the `0.2%` drift in R1.

### R4. Everything in §2–§4 survives, at rest, with `M_eff = m₀M`

- **GMOR is a rest-frame relation and stays exact:** `μ² F_Q²(0) = δΣ` to `0.0` / `3.5e-18` across `(m₀, M) ∈ {(1,1),(0.5,1),(1,2.5),(0.3,0.7)}`. This is *precisely* why `f_π` is defined at zero momentum in ChPT — the coupling runs, so the relation is quoted at `p=0`.
- **`ρ_CLU = μ₁²/μ₂² = M₂/M₁`** — `m₀` cancels, so the custodial structure (§4.2) is untouched.
- **The kinetic-spurion blindness theorem (§4.1) holds verbatim:** the rank/congruence argument only needs `M_eff ≻ 0`. Verified: flat `μ² = 0.0` exactly for random anisotropic `M` under relativistic `M_eff`; relativistic latch drift `= 0.0` exactly over 8000 steps under `M = diag(1, 1.7)`.

*Caveat (F5 Prop-2).* The one-mode `2×2` reduction is exact only near `p ≈ 0`: at finite momentum the relativistic `∇²_pT` couples modes through the shared square root. All the rest-frame statements above are therefore exact; but the **write** transports through finite `p` (R3), and the **spectra of a hot state** are not the rest spectra.

### R5. A causal retention–bandwidth bound (new)

Overdamped retention is `n₁/₂ ≈ 2γln2/((2−γ)(εμ)²)` and GMOR gives `μ² = δΣ/F²`, so `n₁/₂ ∝ F²`. But `θ̇_max = c√m₀/F`, so bandwidth `∝ 1/F`. **The decay constant cancels from the product:**

```
n₁/₂ · θ̇_max²  =  2γ ln2 · m₀c²  /  ( (2−γ) ε² · δΣ )        — independent of M and r*
```

Verified on the actual damped relativistic map (`δ=0.02, γ=ε=0.05`), `M ∈ {0.5, 1, 2, 4}`:

| `M` | `εμ/γ` | `n₁/₂` measured | exact-map | `θ̇_max` | product | / predicted `709.7` |
|---|---|---|---|---|---|---|
| 0.5 | 0.200 | 341.98 | 341.98 | 1.4119 | 681.7 | 0.9605 |
| 1.0 | 0.141 | 698.32 | 698.32 | 0.9983 | 696.0 | 0.9806 |
| 2.0 | 0.100 | 1410.56 | 1410.56 | 0.7059 | 702.9 | 0.9904 |
| 4.0 | 0.071 | 2834.83 | 2834.83 | 0.4992 | 706.4 | 0.9952 |

The measured half-lives match the exact 2×2 map to the printed digits; the residual is **exactly the deep-overdamped correction `(εμ/γ)²`** (deficits `3.95 / 1.94 / 0.96 / 0.48 %` against `(εμ/γ)² = 4.00 / 2.00 / 1.00 / 0.50 %`). The bound is therefore an `εμ ≪ γ` statement, with a known leading correction.

**Thermal analogue** (from §5's `D`): `n_T·θ̇_max² = Δ²γ·m₀c²/(ε²T(2−γ))` — same structure, with `T` replacing `δΣ` as the forgetting drive. [proven from the verified `D`-law + algebra; the relativistic Langevin diffusion at `T>0` was not re-measured.]

> **Reading.** `F` buys robustness (`n₁/₂ ∝ F²`) and pays for it in write bandwidth (`θ̇_max ∝ 1/F`). The *product* is set by the causal budget `m₀c²` over the forgetting drive (`δΣ`, or `T`). **Only raising `c` (or `m₀`), or lowering the breaking, buys both.** In Newtonian mode `θ̇` is unbounded, so there is no bandwidth constraint and no bound: **this trade-off is created by the causal governor.**

### R6. Write overflow: aliasing → saturation, but **conditionally** *(corrects my first phrasing)*

I first said the relativistic aliasing threshold (§6) is "exponentially harder to reach." That is only half true. Exactly:
```
p_crit^rel / p_crit^Newt = sinh(ζ*)/ζ* ,      ζ* = 2π γ_c / θ̇_max
```
Verified: `48.86 / 2.479 / 1.288` at `γ = 0.05 / 0.02 / 0.01`. The protection is exponential in `γ_c/θ̇_max` and **vanishes as `γ→0`** — a long coast means the write never has to be relativistic. So: *the causal cap converts write-overflow aliasing into graceful saturation in the fast-forgetting / narrow-causal-bandwidth regime, and does nothing in the conservative regime.* A conditional benefit, stated as such.

### R7. A causal memory-lifetime floor — the governor's *memory-side* payoff

Because `|q̇| ≤ v_max` at every step, trivially and without any assumption:
```
|Δθ_n| ≤ n · ε · θ̇_max      ⇒      erasing a register to tolerance Δ takes  n ≥ Δ/(ε θ̇_max) steps
```
**for any noise, any temperature, any adversary.** Newtonian mode admits no such bound. Diffusive corollary, verified: with the coded Langevin on a flat direction, `D_rel` **saturates** (`0.0093 → 0.671` over `T ∈ [0.01, 1000]`) while `D_newt` grows linearly (`D_rel/D_newt = 0.971 → 7.0e-4`).

Combining with §5: **temperature is the only eraser of a flat direction, and the causal cap bounds how fast temperature can erase it.** The paper currently sells the relativistic governor as *velocity safety*; it is also a **memory-robustness guarantee** — bounded forgetting under unbounded noise injection. That is an ML-measurable consequence and it earns the physics (P1).

### R8. [NEW DEFECT] F5 Prop-9's FDT fix is Newtonian-only

**Proof (free particle, `V=0`).** The coded step reduces to `p_{n+1} = (1−γ)p_n + σξ_n` — *autonomous in `p` and linear*, so its stationary law is **exactly Gaussian**, `N(0, σ²/(γ(2−γ)))`, `= N(0, M_eff T)` under F5's `σ*`. But `H = T(p)` and the Gibbs momentum marginal is **Maxwell–Jüttner**, `π(p) ∝ exp(−(c/T)√(pᵀM⁻¹p + m₀²c²))`, which is non-Gaussian (exponential tails). **A linear OU O-step cannot have a non-Gaussian stationary law, so no choice of `σ` gives the coded relativistic Langevin a Gibbs invariant.** ∎

Quantified (`M = m₀ = 1`), as a function of `T/(m₀c²)`:

| `T/(m₀c²)` | `Var_MJ/(M_eff T)` | excess kurtosis | `KL(MJ‖Gauss)` |
|---|---|---|---|
| 0.01 | 1.0150 | 0.030 | 7.4e-5 |
| 0.10 | 1.1534 | 0.295 | 6.8e-3 |
| **1.00** | **2.6995** | **1.857** | **0.384 nats** |
| 8.00 | 16.282 | 2.907 | 6.31 nats |

Corroborated dynamically (harmonic well, valid coded-OU harness): `Var(q)/(T/k) − 1` = `−0.0036` for the **Newtonian control at every `T`** (the `O(ε²)` shadow floor), versus `−0.073 / −0.115 / −0.313 / −0.538 / −0.729` for relativistic at `T/(m₀c²) = 0.056 / 0.1 / 0.5 / 2 / 8` — and the two configurations with `T/(m₀c²) = 8`, namely `(c=1,T=8)` and `(c=0.5,T=2)`, give **bit-identical** `−0.7290074`. The defect is a function of `T/(m₀c²)` alone.

**Why this matters.** `Exp-C` (the MNIST generative experiment, the paper's Exp III) runs `kinetic_mode='relativistic'` with `m₀ = 1`, `c = 1`, `sleep_temperature = 0.5`, anneal `1.0 → 0.01`: it **starts at `T/(m₀c²) = 1`**, where the true momentum law has `2.7×` the variance the sampler enforces. The paper-run project `finalA` used `c = 5` ⇒ `T/(m₀c²) = 0.04`, benign — which may be why it behaved better.

**This is a *second, independent* candidate mechanism for the MNIST 3/5/8/9 mode imbalance**, distinct from §7.9's per-mode `T_eff,i` (which is a Newtonian-mode statement). Fixes, in increasing cost: **(iii) raise `c` (or `m₀`) so `T ≪ m₀c²`** — free, one config line, and already validated by `finalA`; (ii) Metropolis-adjust; (i) replace the O-step by an exact Maxwell–Jüttner momentum refresh.

*Honesty note:* my first attempt to demonstrate the fix (a Maxwell–Jüttner refresh arm, `checks7.py`) was **under-converged** — full momentum refresh with a single leapfrog step mixes position on a `~1/(ε²k/M) ≈ 10⁴`-step timescale, longer than the run. It is uninformative and is **not** used as evidence. The defect rests on the free-particle proof plus the `T/(m₀c²)` scaling of the coded chain.

---

## 8. The correspondence table (deliverable 1)

Status: **[P]** proven here or in F5 · **[V]** additionally verified numerically (script in App. P) · **[E]** evidenced · **[C]** conjectured · **[R]** refuted · **[V4]** deferred.

| HEP concept | CLU object | where in our program | status |
|---|---|---|---|
| linear sigma model | `SO2InvariantPotential` = `f_θ(r²) + αr²` | `chlu/core/potentials.py:176` | [P] structural, not analogy |
| order parameter / VEV `⟨σ⟩` | vacuum radius `r*` | V2 §3.1, §3.5 | [P][V] |
| chiral condensate `Σ = |⟨q̄q⟩|` | `Σ = r*(δ) = −∂E_vac/∂δ` | **not currently measured** (shipped tilt hides it) | [P][V] → **F-1** |
| pion `π` | coset coordinate `θ` (the latched register) | F5 §3.3a, §4.1 | [P][V] |
| pion decay constant `f_π` | **`F = √(M_ch)·r*`**, via `Q = F²θ̇` at zero momentum | F5 calls `r*` "f" — **name is misplaced** | [P][V] → **corrigendum** |
| quark mass `m_q` (spurion) | tilt `δ` (linear/ambient spurion) | `TiltedPotential` is *angular*, not linear | [P] → **F-1** |
| GMOR `m_π²f_π² = 2m_qΣ` | **`μ² F² = δ Σ`, exact** | V2 §3.1 verifies `μ² ∝ δ` only | [P][V] `2.2e-16` |
| resonance saturation of LECs | leading LEC `= 1/(M_ch μ_rad² r*)` (radial/σ mode) | F5 §4.3 route-(iii) `1/μ_rad²` | [P][V] `0.9959` |
| chiral expansion parameter `m_π²/m_σ²` | `μ²/μ_rad²` | explains softest-seed anharmonicity | [E] |
| Goldstone theorem (classical) | zero Hessian eigenvalue along the orbit | F5 §4.2 | [P] |
| **Goldstone vs modulus** | symmetry protects the **charge**; V-flatness protects the **register** | **new; resolves V2 App. C** | [P][V] |
| Wigner–Weyl (unbroken) | eroded vacuum `r*=0`, degenerate doublet (Schur) | `v2-full-runs` Finding 0 — **unnamed** | [P][E] → **F-3** |
| chiral restoration | sleep-CD vacuum erosion (`r*→0`); anchor `λ` holds `Σ` up | V2 §3.5 | [P] structurally exact |
| pseudo-Goldstone | tilted / self-broken register, `n₁/₂ ∝ 1/δ` | V2 §3.1, §3.4 | [P][V] |
| custodial `SU(2)_V`, `ρ = 1` | kinetic isotropy `[M,X]=0` ⇒ **`ρ_CLU = μ_1²/μ_2² = 1`** | F5 §4.1 (Schur) | [P][V] · **V4-load-bearing** |
| custodial breaking (`g′`, `y_t−y_b`) | mass anisotropy `‖[M,X]‖` | V2 App. C | [P][V] tree-level only |
| `f_K ≠ f_π` from `SU(3)` breaking | `F_a² = M_a r*²` split; `μ_a²F_a² = δΣ` universal | — | [P][V] → **F-4** |
| Gell-Mann–Okubo mass relations | **sum rules among inverse retention half-lives** | — | [C] → V4 |
| Coleman / Mermin–Wagner (no SSB, low `d`) | coset diffusion at `T>0`; `D = εT(2−γ)/(2F²γ)` | Thread-9's measured latch erosion | [P][V] |
| finite-`T` condensate melting | `F → 0`, `μ² = δΣ/F² → ∞` ⇒ memory dies at the transition | — | [E] → **F-5** |
| Symanzik effective action (lattice) | shadow Hamiltonian; `O(ε²)` artifacts; exact lattice symmetry ⇒ no counterterms | F5 Prop-7/8 | [P][V] (naming only) |
| HEFT/SMEFT (nonlinear/linear realization) | coset vs ambient parameterization | F5 §4.3 | [P] |
| Goldstone derivative coupling | **cross-talk only during writes**; `F_φ²(θ)` metric gain | — | [P][V] → V4 |
| non-abelian Goldstone self-interaction | symmetric-coset holonomy `α²`; value order-error `α⁵` | — | [P][V], **[R]** for my `α²` guess |
| `π₁(G/H) = ℤ` (compact coset) | register aliasing at `p_crit = 2πM_ch r*γ/ε` | — | [P][V] |
| `f_π` at zero momentum | GMOR is a **rest-frame** relation; `F_Q²(p) = F_Q²(0)cosh ζ` runs | §7bis R1 | [P][V] |
| relativistic rotor (**not** a form factor) | Lorentz-enhanced coset inertia | §7bis R1 | [P][V] · **[R] "form factor"** |
| rapidity `ζ` | what the relativistic latch actually stores; also V1's squeeze variable | §7bis R3 | [P][V] |
| light cone | **causal memory-lifetime floor** `n ≥ Δ/(εθ̇_max)` vs any noise | §7bis R7 | [P][V] |
| Maxwell–Jüttner distribution | the Gibbs momentum law in relativistic mode — **not** what the code samples | §7bis R8 | [P][V] · **new defect** |
| chiral logs, loops, LEC running | **no analogue** (0+1D, no `ħ`) | — | **[R] do not use** |
| anomalies / WZW term | **no analogue** (no fermions) | — | **[R] do not use** |

---

## 9. New falsifiables (deliverable 3)

> **✅ F-1 CLOSED — independent cross-validation on the shipped code (recorded 2026-07-09).**
> While v1.1 was being written, `experiment-engineer` landed `LinearSpurionPotential` + config wiring + tests on `main` (`2aca35c`, `64af0e7`, `9bc2cf7`). I re-ran them: **`5 passed`** (`uv run --no-sync pytest tests/test_goldstone.py -k "gmor or spurion or condensate"`, 20.05 s, `main @ 9bc2cf7`).
> The test measures the three GMOR objects independently on the real JAX/Equinox path — `μ²` from the autodiff-Hessian spectrum probe, `F² = M_ch r*²`, `Σ = r*(δ)` — and asserts `μ²F² = δΣ` with relative deviation `< 1e-12` for `δ ≥ 1e-2`, plus `r*` demonstrably *running* with `δ` (`r_hi − r_lo > 1e-3`) where the shipped **angular** tilt leaves `r*` exactly at `f`.
> Two independent confirmations of my §2.3 result, on different normalizations of the hat (theirs `λ(r²−f²)²`, mine `(λ/4)(r²−f²)²` — GMOR is normalization-independent, as it must be) and different differentiation routes (their autodiff Hessian, my analytic one).
> **They independently rediscovered the catastrophic-cancellation floor** I hit in `checks2.py §A′`: the autodiff Hessian reconstructs `K_ang = δ/r*` as a difference of `O(‖K‖)` terms, so the *absolute* deviation floors at `ε‖K‖F²` and the *relative* one at `~ε/δ`. Their test asserts against the absolute floor and checks relative exactness only for `δ ≥ 1e-2` — the correct discipline. My cancellation-free Newton-in-`u` solve gives `2.2e-16` **relative at every `δ`**, which is the sharper statement and is available to them if wanted.

Ordered by (value ÷ cost). All are probe-only unless noted.

| id | falsifiable | prediction | cost / owner |
|---|---|---|---|
| **F-1** | ~~**Condensate-resolving GMOR.**~~ **→ IMPLEMENTED & CONFIRMED** on the real code path while this note was being written (see box below). | `μ²F² = δΣ` **exactly**. ✅ | **DONE** (engineer). Remaining: the *NLO* half — LO-GMOR rel. error `= δ/(M_ch μ_rad² r*)`, the resonance-saturated LEC — is **not yet tested**. |
| **F-2** | **Oscillation vs drift.** Re-examine `v2-full-runs` item-5 raw `Q(t)` traces at `γ=0` — **and F5's own check (g)** (`M=(1,2)`, "charge drifts by 2.6"). | The charge non-conservation is a **bounded oscillation** (period = half a revolution), amplitude `√(2E)r(√M_max−√M_min)` — for F5's `M=(1,2)`: `0.414·√(2E)·r` — with **no secular drift**. | ~1 h, **analyst**, data already on disk. **Claim-affecting:** softens V2 App. C *and* F5 check (g), and weakens the `tie_channel_mass` design rule (the register survives anisotropy; only the write current is θ-modulated). |
| **F-3** | **Schur degeneracy in the unbroken phase.** On the collapsed (`r*=0`) checkpoints measure `|μ₁²−μ₂²|/μ̄²`. | Designed: `~1e-15` (architectural ⇒ pipeline check). **Emergent: splitting `= O(E^V_eq) ≈ 0.03–0.11`** — a *new* attribution instrument for explicit breaking that works in the **unbroken** phase, where `E^V_eq` on a vacuum orbit does not apply. | ~1 h, **analyst**. Cashes the taxonomy's first free prediction. |
| **F-4** | **Custodial retention anisotropy** (the task's requested prediction). Designed 2-register bank (`SO(3)→SO(2)`, or two channels in a 2D irrep of `H`), untie the multiplet inertias, apply an `H`-preserving spurion. | `n_{1/2,1}/n_{1/2,2} = m_1/m_2` **exactly and independently of `δ`**; and `μ_a²F_a² = δΣ` **common to the multiplet** even though `μ_a` and `F_a` both split (`f_K ≠ f_π`). | Medium (needs a 2-Goldstone potential ⇒ V4, or a hand-built V2 appendix). **engineer.** Verified in toy to `1e-10`. |
| **F-5** | **The `(μ, γ, T)` budget cube.** Sweep `T` on a latched designed checkpoint. | `D = εT(2−γ)/(2M_ch r*²γ)`; latch `n₁/₂ ∝ F²γ_c/T`; **`∂n₁/₂/∂γ > 0`** (more friction ⇒ *longer* memory at fixed `T`) — the opposite sign to F5 Cor-13's massive-mode result. Plus Arrhenius `n_hop ∝ e^{2δr*/T}` for the pNG. | Small, **analyst** (Langevin rollout exists). Gives V2/V4 a third axis and explains Thread-9's coset diffusion from first principles. |
| **F-6** | **Erasing a Goldstone register needs a temperature field, not a friction field.** | `γ_φ(q)` cannot delete latched coset content (F5 Cor-13) and **raising `γ` slows thermal erasure** (`D ∝ 1/γ_c`). A learned `T_φ(q)` erases at rate `∝ T_φ/(γ_c F²)`. | **Design-changing.** Explains the existing negatives N12/N13/`fit-gap-anatomy` `−24%`. Spec for `experiment-engineer`: extend the `γ_φ` machinery with a learned `T_φ(q)` (Thread-1's original Hawking framing). |
| **F-7** | **Register-bank geometry.** Compare an abelian (torus `T²`) bank against a symmetric-coset (`S²`) bank at matched capacity. | Torus: exactly independent registers, zero holonomy, `π₁`-aliasing at `p_crit`. `S²`: value order-error `O(α⁵)`, **basis holonomy `α²`**, multiplicative gain `1/sinθ`. | V4. **The design rule falls out: pick abelian cosets for independent registers.** |
| **F-9** | **The relativistic Gibbs defect on Exp-C** (§7bis R8). Re-run the Exp-C generative sweep at `c ∈ {1, 5}` (equivalently `m₀`), i.e. `T/(m₀c²) ∈ {1, 0.04}`, everything else fixed. | If the sampler's missing Maxwell–Jüttner tails drive the mode imbalance, the 3/5/8/9 over-representation should **measurably shrink at `c=5`**. Also directly measurable: the momentum marginal of a trained chain is Gaussian (coded) where Gibbs demands MJ (`Var_MJ/M_effT = 2.70` at `T/(m₀c²)=1`). | **Tiny** (one config flag, checkpoints exist). **analyst.** Tests §7.9's long-standing conjecture with a *second, independent* mechanism, and the fix is free. |
| **F-10** | **Causal memory-lifetime floor** (§7bis R7) — the governor's memory-side payoff. Latch a designed register; inject escalating noise (or an adversarial impulse train); measure steps-to-erasure for `newtonian_learned` vs `relativistic`. | Relativistic: `n ≥ Δ/(ε θ̇_max)` **regardless of noise amplitude**; `D` saturates. Newtonian: erasure time `∝ 1/T`, unbounded. | Small, **analyst/engineer**. **This is the ML-measurable benefit that earns the relativistic governor a place in the memory story** (P1), and V2 currently runs `newtonian_learned`. |
| **F-11** | **The rapidity register / write companding** (§7bis R3). Sweep write impulse `p₀` over 3–4 decades on an SO(2) channel, relativistic vs Newtonian. | `Δθ = ε θ̇_max Σ tanh ζₙ`, exact; transport **logarithmic** in `p₀` (`p₀=500 → Δθ=7.68` vs Newtonian `625`). Aliasing protection `= sinh(ζ*)/ζ*`, `ζ* = 2πγ_c/θ̇_max` — present at `γ=0.05`, absent at `γ=0.01`. | Small, **analyst**. Gives the compact register a *graceful-saturation* story instead of wraparound corruption. |
| **F-12** | **The retention–bandwidth bound** (§7bis R5) on trained checkpoints, using V3's mass banding to vary `M` (hence `F`) at fixed `δΣ`. | `n₁/₂ · θ̇_max²` invariant across bands to `(εμ/γ)²`. | Medium, **analyst**. Turns "M is the budget allocator" (F5 §5) into a *conservation law*: `M` moves retention and bandwidth in opposite directions, product fixed. |
| **F-8** | **Learned (data-driven) breaking.** Train on data with a controlled angular asymmetry `α`; measure `μ²`, `F`, and the induced `δ_eff`. | GMOR predicts `μ²F² = δ_eff Σ` with `δ_eff` linear in the data asymmetry ⇒ **"task symmetry sets memory lifetime," from data rather than by hand.** | Already in `future_work.md` ("Learned vs analytic explicit breaking") — this note gives it a *quantitative* target. |

---

## 10. Honest flags — what is evocative, what is refuted, what I could not do

> *Label note: refutations are **X**-numbered to avoid collision with §7bis's **R**-numbered relativistic claims.*

**Refuted (mine, in this thread).**
- **X1.** I conjectured that non-abelian write-ordering displaces the stored value at `O(α²)`. **False.** `S²` is a symmetric space; the leading commutator sits in the stabilizer. Truth: `O(α³)` for equal-arclength writes, `O(α⁵)` for the CLU's own charge-impulse write, with the non-commutativity showing up as an `α²` **frame holonomy** (§7). Both exponents verified to 3 decimal places and cross-checked against the damped map (`2.5e-15`).
- **X2.** My first `f_π` cross-talk formula (`gain ∝ 1/sin²θ` for a Cartesian kick) was off by one power of `sinθ` — the Cartesian impulse also *projects* onto `p_φ` with a `sinθ`. Correct: `1/sinθ` for a fixed Cartesian kick, `1/sin²θ` for a fixed charge. Verified to `1e-5`.
- **X4 (v1.1).** I first wrote that the relativistic aliasing threshold is "exponentially harder to reach." **Only conditionally.** The exact factor is `sinh(ζ*)/ζ*` with `ζ* = 2πγ_c/θ̇_max`; it is `48.9×` at `γ=0.05` but `1.29×` at `γ=0.01`, and `→1` as `γ→0`. Corrected in §7bis R6.
- **X5 (v1.1, harness).** My Maxwell–Jüttner "fix demonstration" (`checks7.py`, MJ-refresh arm) was **under-converged** and is discarded as evidence; the §7bis R8 defect rests on the free-particle proof + the `T/(m₀c²)` scaling. Reported rather than dropped. (`checks6.py`'s first T2 harness was outright invalid — an unclosed Verlet step; superseded.)

**Refuted (the program's).**
- **X3.** F5 §4.1: "if `M` does not isotropize, the channel is pseudo-Goldstone with `μ² ∝ mass splitting`." **Proven false and already contradicted by our own data.** See §4.1(a). **Blocking for the F5 arXiv push.**
- **X6 (v1.1).** F5 Prop-9's `σ*_i = √(M_eff,i T γ(2−γ))` is presented as *the exact discrete-FDT noise*. It is exact **only for the Newtonian kinetic modes**. In `relativistic` mode no `σ` works (§7bis R8) — the O-step is linear, the Gibbs momentum law is not Gaussian. V2's Appendix F currently states the `σ*` fix as a class-level "neutral theorem"; that wording needs a kinetic-mode qualifier.

**Evocative but *not* structural — keep out of the papers (P1 discipline).**
- "Chiral logarithms", "one-loop ChPT", "LEC running": no `ħ`, no `d ≥ 2`, no loops. The only expansions are `ε²` (Symanzik) and `T` (thermal) and `1/F²` (curvature).
- "Anomaly", "WZW", "topological term": no fermions. The topology we *do* have is `π₁(G/H)` (aliasing) and coset curvature (holonomy) — use those instead.
- "`ρ`-parameter" for `SO(2)`: there is no mass ratio to protect in a 1-dimensional coset. Only say "custodial" when `dim(G/H) ≥ 2`.
- "Spontaneous symmetry breaking" without the `T=0`/finite-dimensional caveat. At `T>0` there is provably no long-range order in `0+1D` (§5).
- "Pseudo-Goldstone" for the emergent arm is *correct terminology* but its ChPT power counting is marginal (`δ_eff` is not parametrically small) — say so.

**What I could not prove / did not do.**
- **O1.** The `(μ, γ, T)` retention law is verified as **two additive channels** (`1/n_relax + 1/n_hop`) only heuristically; I did not derive the exact crossover, nor the joint law in the underdamped band. Open.
- **O2.** The diffusion derivation is exact for a *strictly* flat direction of the coded map. On the curved ring the measured ratio is `1.020 ± 0.020` (free-particle control `0.984 ± 0.016`); both are consistent with `1` at `1σ`, but I did not derive the curvature/radial-coupling correction. Open (small).
- **O3.** Everything here is on **toy Mexican-hat potentials**, not on trained CLU checkpoints. Every claim is stated at the level of the *class* (separable `H`, damped Verlet, `V`-invariant channel) and is therefore inherited by the trained models — but F-1…F-5 are exactly the confirmations on trained models, and they are **not done**. Nothing in §0's table is a claim about a trained CLU except where it re-derives an *already-measured* V2 result.
- **O4.** Non-abelian GMOR/latch on a **curved coset with a learned potential** (not a hand-built hat) is untouched. That is `future_work.md`'s "Non-abelian latch/GMOR on a curved coset" and stays open.
- **O5. → CLOSED in v1.1 (§7bis).** The running decay constant is `F_Q²(p) = F_Q²(0)cosh ζ`; GMOR, `ρ_CLU` and blindness survive at rest with `M_eff = m₀M`; and the cap yields the RB bound, the causal retention floor, and the rapidity-register law. It also exposed the relativistic FDT defect (R8).
- **O6 (new).** The RB bound is derived in the deep-overdamped band (`εμ ≪ γ`) with a verified `(εμ/γ)²` correction. Its **underdamped** counterpart (where `n₁/₂` is mass-independent) is *not* derived — there the product presumably collapses to a `γ`-only statement. Open, easy.
- **O7 (new).** The rapidity-register law is exact on a *flat* direction. On a **curved** coset a hard write leaves the vacuum manifold; I quantified the effect only at `O(0.2%)` for `ζ ≤ 3`. The large-`ζ` geometric correction (and whether a hard write can *dislodge* a latch off the orbit) is open — and is the natural failure mode of a companding register.
- **O8 (new).** `Exp-C` runs relativistic Langevin at `T/(m₀c²) = 1`. Whether the missing Maxwell–Jüttner tails actually cause the 3/5/8/9 imbalance is **conjectured**, not shown (F-9 decides it, cheaply).

---

## Appendix P — flag provenance for the numerical checks

All checks are **repo-read-only pure-numpy toys** of the coded map (`chlu/core/integrators.py::velocity_verlet_step` semantics: KDK, then `p ← (1−γ)p`; Langevin: full Verlet step, then damp, then noise). They do **not** load CLU checkpoints and carry no training config.

| item | script | key parameters | seeds | observed |
|---|---|---|---|---|
| GMOR exact | `checks2.py` §A′ | hat `λ=3, f=1`, `M=0.7·I`, linear spurion `δ∈[1e-8,0.3]`; Newton in `u=r*−f` | — | max rel dev `2.220e-16`; LEC ratio `0.99586` |
| `F` two ways | `checks3.py` §B* | `λ=3,f=1,m=0.7`, `ε=0.005`, `γ=0`, `6e4` steps, revolution-averaged `θ̇` | — | `dev/p₀² = 0.4745, 0.4762` (converged) |
| kinetic blindness | `checks2.py` §C′ | random `M = exp(N(0,0.8))`, `λ=3,f=1`; latch `γ=0.05, ε=0.05`, 12 000 steps; charge run `ε=0.02, γ=0`, `3e5` steps | rng 0 | flat `μ² = 0.0`; latch drift `0.0`; `A_meas/A_pred = 1.038/1.028/1.019`; secular/amp `4.2e-4 / 4.8e-6 / 7.1e-6` |
| custodial / `ρ_CLU` | `checks2.py` §D′ | `SO(3)`, `λ=3,f=1`, `δ=0.02` along `z`, `M=diag(0.6,1.5,1.0)`, `ε=0.05, γ=0.05`, `6e4` steps | — | `ρ` dev `0.0`; universality dev `0.0`; `n₁/₂` meas/exact `1.5e-10`, `7.1e-9` |
| coset diffusion | `checks3.py` §E* | `λ=120` (ring) / `λ=0` (free), `ε=0.02`, `m=f=1`, `γ∈{.05,.1,.2,.4}`, `T∈{.01,.02,.04,.08}`, `nw=4000`, `3e4` steps, online MSD | 1–5 | free `0.984±0.016`; ring `1.020±0.020`; `d log D/d log T = 0.9976`; `d log D/d log γ_c = −0.9797` |
| Arrhenius | `checks2.py` §F′ | `λ=6,f=1,m=1`, `ε=0.05`, `γ=0.10` (`γ_c/ω₀ = 5.5`, overdamped), `δ=0.15`, `ΔE=0.3037`, `T = ΔE/{2,3,4}`, `nw=400`, `1.5e5` steps, **survival-function** rate fit | 101 | slope `−0.3261` vs `−ΔE = −0.3037`; ratio `1.0739`; 400/400/398 escaped |
| equipartition | `checks2.py` §F′ | same, `T∈{0.02,0.05}` | 11 | `⟨θ²⟩/(T/δr*) = 1.048, 1.094` |
| aliasing / transport | `checks.py` §G | `λ=6,f=1,m=1,ε=0.05`, `γ∈{.05,.02,.008}`, `p₀=0.05`, `4e4` steps | — | `Δθ_meas/Δθ_pred = 0.99988` (all `γ`) |
| Poisson algebra | `checks2.py` §H1 | random `(q,p)∈ℝ³`, 200 draws, `so(3)` | rng 5 | max dev `0.0` |
| metric cross-talk | `checks2.py` §H2 | `S²`, `λ=30,f=1`, `ε=0.02, γ=0.02`, `4e4` steps, kick `0.02` | — | `1.15465` vs `1.15470`; `1.41403` vs `1.41421` |
| ordering / holonomy | `checks4.py` | exact `SO(3)` geodesics + damped map cross-check (`debug.py`) | — | exponents `3.000` / `5.000`; map↔geometry `2.45e-15`; `rot/α² → 1.0001` |
| **v1.1 — rapidity latch (R3)** | `checks5.py` | gutter `V=½kq₁²` (exactly flat `q₀`), `M=0.8, m₀=c=1, ε=γ=0.05, k=4`, 4000 steps, `p₀∈[0.05,500]` | — | `max|sim/closed-form−1| = 2.2e-15`; rapidity-slope ratio `1.000000` |
| **v1.1 — running `F_Q` (R1/R2)** | `checks5.py` | stiff ring `λ=2000, r*=1, M=m₀=c=1, ε=0.002, γ=0`, 3000 steps | — | `F_Q²/(M_eff r̄²cosh ζ) = 0.99999 → 0.9982` (`p₀ = 0.2→10`) |
| **v1.1 — GMOR/ρ/blindness, relativistic (R4)** | `checks5.py` | `λ=3, f=1, δ=0.02`; `(m₀,M)` grid; anisotropic latch `λ=50, ε=0.02, γ=0.05`, 16 000 steps | rng 0 | GMOR dev `0.0`/`3.5e-18`; flat `μ² = 0.0`; latch drift `0.0` |
| **v1.1 — RB bound (R5)** | `checks5.py` | ring + linear spurion `λ=6, δ=0.02, ε=γ=0.05, m₀=c=1`, `M∈{.5,1,2,4}`, 80 000 steps | — | products `681.7/696.0/702.9/706.4` vs `709.7`; residual `= (εμ/γ)²` (`3.95/1.94/0.96/0.48%` vs `4.00/2.00/1.00/0.50%`) |
| **v1.1 — aliasing factor (R6)** | `checks5.py` | bisection on `ζ*`, `M=m₀=c=1, ε=0.05`, `γ∈{.05,.02,.01}` | — | `p_crit^rel/p_crit^N = 48.86 / 2.479 / 1.288` `= sinh(ζ*)/ζ*` |
| **v1.1 — causal floor / D saturation (R7)** | `checks6.py` | flat direction, coded Langevin, `M=m₀=c=1, ε=γ=0.05`, `T∈[0.01,1000]`, 3000 walkers × 6000 steps | rng 4 | `D_rel = 0.0093 → 0.671` (saturates); `D_rel/D_newt = 0.971 → 7.0e-4` |
| **v1.1 — relativistic Gibbs defect (R8)** | `checks7.py`, `checks8.py` | harmonic well `k=1`, coded-OU, `ε=0.01, γ=0.1`, 4000 walkers × 20 000 steps; MJ tables on 2×10⁶-pt grids | 11 | Newtonian control `−0.0036` at **every** `T`; relativistic `−0.073…−0.729`; `(c=1,T=8)` ≡ `(c=0.5,T=2)` → `−0.7290074` bit-identical. `Var_MJ/(M_eff T) = 2.70`, `KL = 0.384` nats at `T/(m₀c²)=1` |

**Git footprint:** none. No tracked file created, modified, or staged by me. Repo was at `27f232f` when v1.0 was written and had advanced to **`9bc2cf7`** by the end of v1.1 (three `experiment-engineer` commits landing F-1 — see the box in §9); working tree clean throughout. The only command I ran against tracked code was a read-only `pytest` invocation. Scratch lives in `.claude/scratch/v2-symmetry-deepdive/` (`checks.py`, `checks2.py`, `debug.py`, `checks3.py`, `checks4.py`, `checks5.py`, `checks6.py`, `checks7.py`, `checks8.py`).
*(Audit trail, kept deliberately: `checks.py` contains two superseded predictions — the `‖[M,X]‖` amplitude proxy and the `α²` ordering guess — superseded by `checks2/3/4.py`. `checks6.py`'s T2 block used an invalid harness (unclosed Verlet step) and `checks7.py`'s MJ-refresh arm is under-converged; both are superseded by `checks8.py` + `checks7.py`'s coded-OU columns. Nothing was silently deleted.)*

---

## Open questions for the Head / co-author

1. ~~F-1~~ **is now DONE and green on `main`** (§9 box) — so the question becomes: **is V2 frozen, or can the condensate-resolving GMOR land as a probe-only appendix?** The remaining half is the **NLO test** (LO error `= δ/(M_ch μ_rad² r*)`, the radial-resonance-saturated LEC): one extra assertion in the existing test, and it is the part that makes the claim *ChPT* rather than *a power law*.
2. **The Goldstone-vs-modulus distinction (S5)** yields a free, sharp differentiator against Mo (equivariance is *sufficient, not necessary*), already supported by V2's own broken-iso battery. Should it go into V2's related work, or be held for V4?
3. **F5 §4.1's error (S6/R3)** is blocking the arXiv push. Correcting it also *strengthens* the note (the blindness theorem is a better result than the false corollary). Who lands it?
4. **F-6** implies the trash-region program's central lever is mis-chosen for flat-direction content. Does the Hub want a theory-side spec for a learned `T_φ(q)` field?
5. Is the co-author interested in **O5** (a momentum-dependent / "running" decay constant from the relativistic kinetic term)? It is the most natural genuinely-new theory question this thread surfaced.

---

## Proposed handover updates (for the Hub)

**§1 (the physics) — add:**
- The **two-axis** statement: *symmetry realization sets `μ²` (axis 1); `(ε, γ, T)` set what `μ²` buys (axis 2)*. The mode-mass budget is axis-2; the realization taxonomy is axis-1.
- **Goldstone vs modulus (S5):** the latch needs only `V`-flatness + `γ>0`; the symmetry protects the *charge*, not the *register*. Equivariance is sufficient, **not necessary**, for neutral memory.
- **Finite-`T` coset diffusion (S8):** `D = εT(2−γ)/(2 M_ch r*² γ)`; the latch's `∞` half-life is a `T=0` statement; **temperature erases flat directions, friction does not** (and more friction erases *slower*).

**§7 (discrepancies) — add two, one blocking:**
- **7.15 [PROVEN, BLOCKING — arXiv]** F5's kinetic-isotropy clause ("no isotropization ⇒ pseudo-Goldstone with `μ² ∝ mass splitting`") is **false**: `μ² ≡ 0` for any invertible `M` (rank/congruence; verified `0.0` exactly). Already contradicted by `v2-full-runs` item 5 (`μ²_ang ~ 1e-15`, latch `∞`, write-freeze `0`). Sites: `formalism-note.md:233` + `:435` (hedged, but carrying a **`[proven; verified (g)]` tag that over-covers** — check (g) tests only the charge half), and **`f5-arxiv-note.md:139` → `papers/f5-note/f5-note.tex:124` (unhedged, arXiv-bound, cited by both shorts).** Replace with the *kinetic-spurion blindness theorem* + the bounded-charge-oscillation corollary — a strictly stronger result. **Fix before push.**
- **7.16 [PROVEN]** F5 §3.3(c)/§8 misname the decay constant: the object in GMOR and in `Q = F²θ̇` is `F = √(M_eff)·r*`, not the orbit radius `r*`. Formulae are correct; the *name* is misplaced. (Consequence: V2's "`f` buys robustness" statement should read "`F² = M_ch r*²` buys robustness" — the inertial mass is half of it.)
- **7.18 [PROVEN, NEW — affects Exp-C defaults]** **F5 Prop-9's FDT fix is Newtonian-only.** The O-step `p←(1−γ)p+σξ` is a linear OU recursion ⇒ stationary momentum is *exactly Gaussian*; the relativistic Gibbs momentum marginal is **Maxwell–Jüttner**. **No `σ` gives the coded relativistic Langevin a Gibbs invariant.** Defect scales with `T/(m₀c²)` (verified: the coded chain's `q`-marginal error depends on that ratio *alone* — `(c=1,T=8)` and `(c=0.5,T=2)` give bit-identical `−0.7290074`; Newtonian control flat at `−0.0036`). **Exp-C default (`relativistic`, `m₀=1`, `c=1`, `T`: `1.0→0.01`) starts at `T/(m₀c²)=1`**, where `Var_MJ = 2.70·M_eff T` and `KL = 0.384` nats; `finalA` used `c=5` ⇒ `0.04`, benign. **Second, independent candidate mechanism for the MNIST 3/5/8/9 imbalance** (§7.9's per-mode `T_eff` is a Newtonian statement). Cheapest fix: raise `c` (one config line). Also: **V2 App. F's class-level "neutral theorem" wording for `σ*` needs a kinetic-mode qualifier**, and F5 Prop-9 needs the same.
- **7.17 [wording]** V2 Appendix C's "Noether-charge **drift** scales linearly with the split" → the exact statement is a **bounded oscillation** (amplitude `√(2E)r(√M_max−√M_min)`, linear in the split at small split; `A/‖[M,X]‖` is *not* constant). Pending **F-2** on the raw traces.

**§8 (open directions) — add:**
- **F-1** condensate-resolving GMOR (probe-only; upgrades V2's headline to GMOR proper with a measured `Σ` and a predicted, resonance-saturated NLO coefficient).
- **The relativistic governor has a memory-side payoff, not just velocity safety** (§7bis): a **causal retention floor** `n ≥ Δ/(εθ̇_max)` against *any* noise (Newtonian has none); a **retention–bandwidth conservation law** `n₁/₂·θ̇_max² = 2γln2·m₀c²/((2−γ)ε²δΣ)`, independent of the decay constant; and a **rapidity register** whose write transport is logarithmic in the impulse (graceful saturation instead of wraparound). **V2 currently runs `newtonian_learned` and therefore shows none of this.** F-10 is the ML-measurable benefit that would earn the relativistic term a place in the memory story (P1).
- **F-9** the Exp-C relativistic-Gibbs defect (one config flag; tests the long-open §7.9 mode-imbalance conjecture via a *second* mechanism).
- **F-6** the trash-region lever is mis-chosen: deleting latched (flat-direction) memory requires a learned **temperature** field `T_φ(q)`, not a friction field `γ_φ(q)` — which retro-explains negatives N12/N13 and `fit-gap-anatomy`'s `−24%`.
- **V4 design rule:** for an *independent* register bank choose an **abelian (torus) coset**; on a symmetric coset the stored value is order-insensitive to `O(α⁵)` but the *addressing basis* rotates by the holonomy `α²`. Non-abelian cross-talk is **multiplicative and idle-free**.
- **F-5** the `(μ, γ, T)` budget cube (the current table is its `T=0` face).

**`future_work.md` — sharpen two existing entries:**
- *"Non-abelian / multi-Goldstone registers", item (iii) "Goldstone self-interaction — registers may couple"* → now **[P][V]**: they couple **multiplicatively through the coset metric, only while writing**; idle registers are decoupled. Retarget the open question to *addressing/holonomy*, not content interference.
- *"Learned (vs analytic) explicit breaking"* → now has a quantitative target: `μ²F² = δ_eff Σ` with `δ_eff` linear in the data asymmetry (**F-8**).
- Add: *"Relativistic running decay constant"* (**O5**) under Foundational theory.
