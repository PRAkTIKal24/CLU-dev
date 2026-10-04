# The three shorts: experiment & derivation inventory

**STATUS: COMPLETE.** Sections 0–4 were published early (2026-09-20T06:37Z) to unblock
`hub/potentials` and `hub/retention`; Section 5 (the per-experiment rows) and Section 6 were appended
at 2026-09-20T06:52Z. **Nothing published in the partial was altered when Section 5 landed** — the
only edit to earlier text is this status banner.

Spoke: `w6-shorts-inventory` (documentation only — no scored run, no claim).
Sources: `/c/chlu/shorts/`. Text recovered with `pdftotext -layout -enc UTF-8`.

> **Extraction note that matters for every formula below.** The default `pdftotext` (Latin-1)
> **silently drops every Greek glyph** in these PDFs. Under that extraction the overdamped retention
> law comes out as `n1/2 ~ 2 ln 2/[(2 - )( )2]` — it loses the **gamma in the numerator** and both
> symbols inside `(eps*mu)^2`. Every formula in this document was re-extracted with `-enc UTF-8` and
> is quoted from that recovery. **Do not pre-register against a Latin-1 extraction of these PDFs.**

---

## 0. Which PDFs are genuinely distinct papers

Five files, **three papers**. Both suspected pairs are not "venue variants" — they are
**byte-identical files under two names**. There is no substance to compare.

| File | sha256 | Verdict |
|---|---|---|
| `F_v1_LCFM_finalSub_2.pdf` | `1a01cfb873bad918796c90fdf1111adf85fc6af7f8fc3e3aeff4105ddc561dd1` | **V1** |
| `F_v1_TTCL_finalSub_2.pdf` | `1a01cfb873bad918796c90fdf1111adf85fc6af7f8fc3e3aeff4105ddc561dd1` | duplicate of V1 (same bytes) |
| `F_v2MemShort_bento_final2.pdf` | `5a0fc126c07975a41ef4a8a139594b010ae95f2c9aa2e1e39d2ffbd2e48b4e11` | **V2** (unique) |
| `F_v5_PaLM_Subm3.pdf` | `ff12667570ce5b4e24f5d0333d0515130dc4f23aa7e3c05cc4df6ebb5eea4514` | **V5** |
| `F_v5_stody_Subm3.pdf` | `ff12667570ce5b4e24f5d0333d0515130dc4f23aa7e3c05cc4df6ebb5eea4514` | duplicate of V5 (same bytes) |

Verified by both md5 and sha256 on 2026-09-20. The pairs are not merely the same size; they are the
same file.

The three papers:

- **V1** — *"Certified Access: Test-Time Compute on a Conservative Physics-Structured AI Memory"*
  (NeurIPS 2026 submission). Reach/escape certificates on a damped-symplectic latent memory.
- **V2** — *"The Price of a Flat Direction: Transverse Curvature sets Retention in a Trained
  Recurrent Memory"* (NeurIPS 2026 submission). The curvature–retention relationship (CRR).
- **V5** — *"Retention You Can Predict, Scope and Delete: a (mu, gamma, T) Law and a Structural
  Deletion Guarantee for a Physics-Structured Memory Store"*. The (mu, gamma, T) budget, the
  friction-hole vault, and exact store-level deletion.

All three use the same reference unit, the **CLU / CHLU** (Jawahar & Pierini 2026, arXiv:2603.01768):
a latent state `(q,p)` advanced by one dissipative velocity-Verlet (kick–drift–kick) step of a learned
separable Hamiltonian `H = T(p) + V_theta(q)`, followed by `p -> (1 - gamma) p`.

---

## 1. LAWS THE WAVE-6 PLAN ASSUMES THAT ARE **NOT IN THE SHORTS**

**Read this before pre-registering anything.**

### 1.1 "Mexican-hat width -> interpolation floor" — **ABSENT. Do not pre-register against it.**

- The string **"interpolation" (any inflection) does not occur once in any of the three papers.**
  There is no interpolation floor, no interpolation experiment, and no interpolation claim anywhere
  in this corpus.
- The phrase **"Mexican hat" occurs exactly once in the whole corpus**, in V2 App. M.4, and it is
  **not a law**. It is a *numerical control potential used to diagnose autodiff round-off*:

  > *"This was confirmed directly: on the analytic Mexican hat a cancellation-free Hessian returns
  > relative deviation 1.1–2.2x10^-16 at every delta in [10^-8, 0.3], while the autodiff Hessian on
  > the same potential returns 2.28x10^-8 -> 4.22x10^-15 across that range, exactly proportional to
  > 1/delta."*
  > — V2, App. M.4 (p. 17)

  The Mexican hat appears here only to show that a **cancellation-free Hessian** is accurate where
  the **autodiff Hessian** has an `eps/delta` round-off floor. No width parameter is introduced, no
  separation parameter is introduced, and nothing is derived from the hat's shape.
- **No paper in this corpus states a Mexican-hat width parameter, a well-separation parameter, or
  any closed form relating either to a retention, accuracy, or interpolation floor.**
- "Sombrero" does not occur. The only potential shapes given a closed form anywhere are:
  - V1's BIBO test potential `V = (1/2) k q_0^2 - epsilon q_0^4` (non-coercive quartic — a *designed
    testbed*, not a hat);
  - V2's angular tilt `V -> V + delta (1 - cos n theta)` and linear ambient spurion
    `V -> V - delta (u . q)` (App. M.3) — *symmetry-breaking probes applied to an already-trained
    potential*, not a Mexican-hat parameterisation.

  **Neither is a Mexican-hat width/separation law and neither should be substituted for one.**

**If a wave-6 hub's pre-registration depends on a Mexican-hat width or separation setting an
interpolation floor, that dependency has no source in these shorts.** The nearest genuinely derived
object is the **floor of the V2/V5 retention curve** (D7 below), which is
**curvature-INdependent** — a floor that explicitly *does not* depend on the potential's shape.
Substituting one for the other would invert the papers' actual finding.

### 1.2 "The exact-decay law" — **AMBIGUOUS; there is no single law by that name.**

No paper in the corpus uses the phrase "exact decay law". There are **three distinct exact
statements about decay/deletion** in the corpus and they are not interchangeable. A hub
pre-registering against "the exact-decay law" must say which:

1. **The exact overdamped decay rate** (V2 App. C Eq. (6) / V5 App. G.3) — the closed form for the
   half-life, quoted verbatim at D5. This is *exact only to leading order in* `h^2/gamma^2`: the
   papers write `≈` and `≃`, **not** `=`.
2. **The exact latch** (V2 App. C / V5 App. G.4) — `|lambda_flat| = 1` exactly, an *infinite*
   half-life at `mu = 0`, `T = 0`. This one **is** an exactness claim (verified to the float64 floor).
3. **"Decay commutes with deletion"** (V5 App. G.8) — an exact *algebraic* statement that amplitude
   decay and deletion act on disjoint components of the store state and therefore commute. This is
   the only thing in the corpus that is simultaneously "exact" and about "decay" in the deletion
   sense.

**Say which one.** (1) is an asymptotic law, (2) is an exactness claim, (3) is a commutation
argument. They license completely different pre-registrations.

### 1.3 Present and safe to pre-register against

Friction, mass, settle time, decay, lifetime and diffusion **are** derived in closed form, and are
catalogued verbatim in §2. Nothing in §2 is a paraphrase.

---

## 2. THE DERIVED LAW CATALOGUE — verbatim, with paper and section

**DERIVED** = a formula the paper writes down. **MEASURED** = a number the paper reports. The two
are separated throughout: this section is DERIVED only; measured numbers live in §5 and in the
explicitly-marked MEASURED lines.

Symbols as the papers define them: `eps` step size, `gamma` per-step damping, `mu` spectral mass,
`M` inertial mass, `T` temperature, `h = eps*mu`, `n_half` half-life in map applications,
`F^2` coset scale, `r*` ring radius, `Delta` read tolerance.

### 2.1 The shared map (all three papers)

| # | Law (DERIVED, verbatim) | Paper · section |
|---|---|---|
| **D1** | `J^T Omega J = (1 - gamma) Omega` and `det J = (1 - gamma)^d` — conformal symplecticity | V1 §2; V2 §3; V5 §2.1 (derived in V5 App. G.2) |
| **D2** | Position-gated form: the gate contributes exactly `(1 - gamma*phi(q'))^d` | V5 App. G.2 |
| **D3** | Per-mode step matrix, "entirely fixed by `(eps*mu, gamma)`": `tr J = (2 - gamma)(1 - h^2/2)`, `det J = 1 - gamma` | V5 App. G.1; V2 App. C Eq. (3) |
| **D4** | Eigenvalues `lambda_pm = (1/2)(tr J ± sqrt(tr^2 J - 4 det J))` | V5 App. G.3; V2 App. C Eq. (4) |

### 2.2 Retention: the band structure (V2 §3, derived V2 App. C; V5 App. G.3)

> **The three bands, quoted verbatim from V2 §3:**
>
> - *"**The Latch:** (mu = 0, symmetry-protected). Exhibits frozen displacement
>   `q_inf = q_0 + eps*p_0/(M*gamma)` and an infinite half-life."*
> - *"**The Overdamped Register:** (0 < eps*mu <~ gamma/2). Resolves to a half-life of
>   `n_half ≈ 2*gamma*ln2 / [(2 - gamma)(eps*mu)^2]`."*
> - *"**The Underdamped Working Memory:** (gamma/2 <~ eps*mu < 2). Bound by a curvature-independent
>   envelope half-life of `2*ln2 / (-ln(1 - gamma))`."*

| # | Law (DERIVED, verbatim) | Paper · section |
|---|---|---|
| **D5** | **Overdamped half-life:** `n_half = ln2/(-ln lambda_+) ≈ 2*gamma*ln2 / [(2 - gamma)(eps*mu)^2]`, with slow root `lambda_+ = 1 - (2-gamma)h^2/(2*gamma) + O(h^4/gamma^3)` | V2 App. C **Eq. (6)**; restated in V5 App. G.3 as `n_half = ln2/(-ln(1-delta)) ≃ 2*ln2*gamma/[(2-gamma) eps^2 mu^2]` |
| **D6** | **The mu^-2 scaling:** `n_half ∝ mu^-2` — "slope -1 in log n_half against log mu^2" | V2 App. C; V2 Abstract |
| **D7** | **Curvature-independent floor:** `n_half = 2*ln2 / (-ln(1 - gamma))`, because `\|lambda_pm\|^2 = lambda_+ lambda_- = det A = 1 - gamma` **regardless of mu** | V2 App. C; V5 App. G.3 (small-gamma limit `-> 2*ln2/gamma`) |
| **D8** | **Crossover / exceptional point:** `h* = sqrt(2/(2-gamma)) * (1 - sqrt(1-gamma)) = gamma/2 + O(gamma^2)`, i.e. `h* ≈ gamma/2` | V2 App. C **Eq. (5)** |
| **D9** | **Upper band edge / stability limit:** `h_f = sqrt(2/(2-gamma))(1 + sqrt(1-gamma)) = 2 - O(gamma^2)`; "the map loses stability at `h = 2` exactly, for every gamma" | V2 App. C Eq. (5) |
| **D10** | **The exceptional point is defective:** at `h = h*`, `lambda_+ = lambda_- = (1/2) tr A = sqrt(1-gamma)`; `A_12 = eps ≠ 0` so one Jordan block, secular transients `∝ n*lambda^n` | V2 App. C |
| **D11** | **Square-root onset:** `Delta ≃ -4(2-gamma) sqrt(1-gamma) h*(h - h*)`, so the phase opens as `phi = arg lambda_+ ∝ sqrt(h - h*)` (exponent 1/2) | V2 App. C |
| **D12** | **The latch, derived:** at `mu = 0`, `A` is upper-triangular with eigenvalues `{1, 1-gamma}`; `p_n = (1-gamma)^n p_0` and `q_inf = q_0 + (eps*p_0/M) * sum_{n>=0}(1-gamma)^n = q_0 + eps*p_0/(M*gamma)` | V2 App. C; V5 App. G.4 |
| **D13** | **Why `gamma = 0` is not a latch:** at `gamma = 0`, `J` is a Jordan block for `lambda = 1`; residual momentum is never removed and `theta_n = theta_0 + n*eps*p_theta/F^2` (ballistic drift) | V5 App. G.4 |
| **D14** | **Confinement caps the lifetime:** with a confinement term `alpha*\|\|q\|\|^2`, every Hessian eigenvalue is lifted by `2*alpha`, so a flat direction becomes `mu^2 = 2*alpha` and the overdamped rate caps the lifetime at `tau_max = Gamma/(2*alpha)`, damping rate `Gamma = gamma/eps` | V5 App. G.4 |
| **D15** | **Spectral mass definition:** `mu_k^2 = lambda_k(M^-1/2 grad^2 V_theta M^-1/2)` at a critical point | V2 App. A; V5 §1 (Nomenclature) |

### 2.3 The friction dial and its critical point (V5 §2.1, derived V5 App. G.3)

| # | Law (DERIVED, verbatim) | Paper · section |
|---|---|---|
| **D16** | **The V-curve critical point:** branches meet where `tr^2 J = 4 det J`, "solved exactly by `h*(gamma) = (1 - sqrt(1-gamma)) * sqrt(2/(2-gamma))`. Inverting at small h, `gamma* = 2*eps*mu*(1 + O(eps*mu))`: the retention dial's computable minimum **`gamma_crit = 2*eps*mu`**" | V5 App. G.3; V2 App. M.2 states the same as `gamma* ≈ 2*eps*mu` |
| **D17** | **Overdamped log-slope in gamma:** `d ln n_half / d ln gamma = 1 + gamma/(2 - gamma)` — "it approaches the +1 asymptote only as gamma -> 0 and **exceeds it at every finite gamma**" | V5 App. G.3 |
| **D18** | **Underdamped log-slope in gamma:** `-gamma/[(1-gamma)(-ln(1-gamma))] = -1 - gamma/2 + O(gamma^2)` — "hugs -1" | V5 App. G.3 |
| **D19** | **A flat coset is the `mu -> 0` limit of the same curve**, with `gamma_crit -> 0` (the unification claim of V5 §2.1) | V5 App. G.3 |

> **D17 is why the measured branches are asymmetric** (V5: "slopes near -1 below, +1.1 to +1.3
> above"). A hub pre-registering "slope +1" on the underdamped branch would be pre-registering
> against a value the paper explicitly derives as *not* +1 at finite gamma.

### 2.4 Finite temperature: FDT noise, diffusion, and the sign flip

| # | Law (DERIVED, verbatim) | Paper · section |
|---|---|---|
| **D20** | **Discrete FDT noise scale:** stationary variance `Var(p) = sigma^2/[gamma(2-gamma)]`; matching Gibbs equipartition `Var(p_i) = M_i T` forces **`sigma*_i = sqrt(M_i * T * gamma * (2 - gamma))`** | V5 App. G.5 (**derived**); V2 App. F (quoted); V1 App. G (quoted, with `M_eff,i`) |
| **D21** | **Coset diffusion coefficient:** `Var(theta_n) = eps^2 * T * n * (2-gamma)/(F^2 * gamma)`, so with `t = n*eps` and `Var = 2*D_theta*t`: **`D_theta = eps*T*(2 - gamma) / (2*F^2*gamma)`** | V5 App. G.6 (**derived**); V2 App. C and App. F (same form); V5 §2.2 (quoted) |
| **D22** | **Coset scale:** `F^2` = the mass-weighted squared displacement of the latent state per unit stored coordinate; for a stored angle on a ring of radius `r*` with tied channel mass `M`, **`F^2 = M * r*^2`** | V5 App. G (Assumptions); V5 §2.2 |
| **D23** | **Diffusive half-life and the sign flip:** `n_half ∝ Delta^2/(eps*D_theta) ∝ gamma/[(2 - gamma) T]` — "increasing gamma lengthens coset memory (`n_half ∝ gamma^+1`) and only T erases it (`n_half ∝ T^-1`)" | V5 App. G.6 |
| **D24** | **Thermal persistence length:** **`l_theta = eps * sqrt(T/M) / (gamma * r*)`** — "the coset register's physical resolution limit at temperature T" | V5 §1 (Evaluation Metrics) |
| **D25** | **Per-step coset increment:** along a flat direction the force vanishes, so `Delta_theta = eps * p_theta/(M * r*)`, with `p_theta` an AR(1) sequence of autocorrelation `rho = 1 - gamma` and stationary variance `M*T` | V5 App. G.6 |

> **Scope rider the papers attach to every T > 0 statement:** `T > 0` requires **both**
> `langevin_noise="fdt"` **and a Newtonian kinetic mode**. V5 App. G.5 derives why: *"For a
> relativistic kinetic term the stationary momentum law of this linear recursion remains Gaussian
> while Gibbs demands a non-Gaussian marginal, which is why T > 0 claims are restricted to the
> Newtonian mode."* A hub pre-registering a finite-T law under a relativistic kinetic mode is
> outside the derivation's stated scope.

### 2.5 The friction hole / memory vault (V5 §2.2, derived V5 App. G.7)

| # | Law (DERIVED, verbatim) | Paper · section |
|---|---|---|
| **D26** | **Effective damping in the hole:** `p' = (1 - gamma_phi)(1 - gamma) p + sigma* xi`, so **`gamma_eff = 1 - (1 - gamma)(1 - gamma_phi)`** | V5 App. G.7 |
| **D27** | **The refrigerator law (absorb-only):** `Var(p) = M*T*gamma*(2-gamma) / [gamma_eff (2 - gamma_eff)] ≡ M*T_local`, i.e. **`T_local = T * gamma*(2-gamma) / [gamma_eff (2 - gamma_eff)]`** | V5 App. G.7 |
| **D28** | **Diffusion inside the hole:** **`D_theta = eps*T*gamma*(2-gamma) / (2*F^2*gamma_eff^2)`** | V5 App. G.7 |
| **D29** | **The vault factor:** `vault = D_theta(no hole)/D_theta(hole) = **(gamma_eff/gamma)^2**` — "(every (2 - gamma) factor cancels)" | V5 App. G.7 |
| **D30** | **The rejected coupled-bath alternative (derived, then experimentally falsified):** noise rescaled to `sigma^2 = M*T*gamma_eff(2-gamma_eff)` gives `T_local = T` and `D_theta = eps*T*(2-gamma_eff)/(2*F^2*gamma_eff) ∝ gamma_eff^-1`, a vault of `(2-gamma)*gamma_eff / [gamma*(2-gamma_eff)]` | V5 App. G.7 |
| **D31** | **The two hypotheses' separation equals the refrigerator factor, identically:** `gamma_eff(2-gamma_eff) / [gamma(2-gamma)] = T/T_local` | V5 App. G.7 |

### 2.6 Deletion (V5 §2.3, derived V5 App. G.8)

| # | Law (DERIVED) | Paper · section |
|---|---|---|
| **D32** | **Decay commutes with deletion.** Store state `(K, {a_i})` with canonical placement `L(K)` a function of the live key set `K` and the fixed geometry alone (and in particular independent of the amplitudes); decay acts item-wise `a_i -> f(a_i)`, touching neither `K` nor `L(K)`; deletion removes one key and recanonicalises, touching neither the surviving amplitudes nor their decay clocks. The two operations act on disjoint components of `(K, {a_i})`, so they commute, and the post-delete store is byte-identical to the store built from the never-inserted history with the same elapsed decay. | V5 App. G.8 |
| **D33** | **Why recency eviction is excluded (derived, not asserted):** recency eviction makes `L` a function of the insertion history rather than of `K`. | V5 App. G.8 |

> **D32 holds only under V5's stated conditions**: a budget sufficient for the cell count, zero
> baseline leakage, and **priority/attribute-based eviction**. V5 states it is a *store-level*
> guarantee only — *"the frozen encoder and any residue of past writes in a learned landscape are
> separate channels, and we claim no certified (epsilon, delta) unlearning."*

### 2.7 Reach, escape and access certificates (V1, derived V1 App. H)

| # | Law (DERIVED, verbatim) | Paper · section |
|---|---|---|
| **D34** | **Relativistic kinetic term:** `T(p) = sqrt(c^2 * p^T M^-1 p + m_0^2 c^4)`, `M = diag(M_1..M_d) > 0` the learned inertial mass, `m_0 > 0` the rest mass | V1 App. H |
| **D35** | **The causal box:** one drift sub-step advances each coordinate by `\|q_dot_i\| < c/sqrt(M_i)`; hence `Q_T(z_0) ⊆ C_T(q_0) := {q : \|q_i - q_0,i\| <= L_i, **L_i = T * eps * c / sqrt(M_i)**}` | V1 §2 + App. H.1 (full derivation) |
| **D36** | **Energy-blindness of the box:** "The bound involves only `(eps, c, M_i)`: neither `V_theta` nor the magnitude of `p` enters." Relative deficit `1 - \|q_dot_i\| sqrt(M_i)/c ≈ m_0^2 c^2 M_i/(2 p_i^2)` as `\|p_i\| -> inf` | V1 App. H.1 |
| **D37** | **The squeeze map:** in mass-weighted coordinates `u = sqrt(M_i) q_i`, `v = p_i/sqrt(M_i)`, the hyperbolic rotation `u' = u cosh(z) + v sinh(z)`, `v' = v cosh(z) + u sinh(z)`; determinant `cosh^2 z - sinh^2 z = 1` on the active pair and 1 elsewhere, hence **`det S_z(M) = 1` exactly** | V1 App. H.2 |
| **D38** | **Bounded-injection certificate:** `u'^2 + v'^2 = (u^2+v^2)cosh2z + 2uv sinh2z <= (u^2+v^2)(cosh2z + \|sinh2z\|) = (u^2+v^2) e^{2\|z\|}` (using `2\|uv\| <= u^2+v^2`), hence **`H(S_z z) <= e^{2\|z\|} H(z)`** and injected energy **`Delta_H <= (e^{2\|z\|} - 1) H`**. Equality exactly on the diagonal `u = ±v`. | V1 App. H.2 |
| **D39** | **Reach bracket:** instantaneous displacement `delta_q' = (p_0/M_0) sinh(z)`; squeeze-extended reach ⊆ `[L, L + p_0 sinh(z)/M_0]`; landing at distance `d` within tolerance 0.4 requires **`z >= arcsinh((d - 0.4 - L) M_0/p_0)`** | V1 App. H.2 |
| **D40** | **Exponential in rapidity, quadratic in excess distance:** writing `delta = (p_0/M_0) sinh z`, the launch-state ratio is `cosh 2z = 1 + 2 sinh^2 z = 1 + 2(M_0 delta/p_0)^2`, "exactly quadratic in delta", while the certificate `e^{2\|z\|} = (cosh z + \|sinh z\|)^2` is exponential in the rapidity. **"the energy is not exponential in the distance itself."** | V1 App. H.2 |
| **D41** | **Gated channel (canonical translation):** `q -> q + Delta`, `p -> p`, with `Delta = b - a`; mirrors the flow of the linear Hamiltonian `W = Delta . p / tau`, so the Jacobian is `I_2d`, yielding **`det J = 1` exactly, independent of Delta and of the gate** | V1 §3 |
| **D42** | **Channel energy:** **`Delta_H_wh = V_theta(q + Delta) - V_theta(q)`**, a discrete jump that must be tracked explicitly; matched loci give transport at zero energy | V1 §3 |
| **D43** | **Hard-gate Jacobian defect:** if the gate varies during the jump, `q -> q + g(q)Delta`, the Jacobian's position block is the rank-one update `I_d + Delta grad_g(q)^T`, so by the matrix determinant lemma **`det J = det(I_d + Delta grad_g(q)^T) = 1 + grad_g(q)^T Delta`**; freezing `g` at capture restores `det J = 1` exactly. "The defect is first order in the gate's variation along the jump, so no smallness of Delta removes it." | V1 App. H.3 |
| **D44** | **Latch-transport law:** for a protected direction with invariant `Q = p^T X q` (the Goldstone charge), `X` the broken symmetry generator, the channel shift exactly evaluates to **`Q' - Q = p^T X Delta`** | V1 §3 |
| **D45** | **Governor and re-absorption:** state-dependent governor `gamma_n = s * tanh(max(0, H - E*))`; re-absorption in **`≈ 2z/gamma_c`** steps (squeeze) and **`≈ (1/gamma_c) ln(1 + Delta_V/E*)`** (gated channel) | V1 §2, §3, App. C Table 1 |

### 2.8 GMOR / spectral-mass identity (V2 App. M — flagged by V2 itself as *supporting*, not a claim)

| # | Law (DERIVED, verbatim) | Paper · section |
|---|---|---|
| **D46** | **GMOR relation:** `mu^2 F^2 = delta * Sigma` — "the retention scaling law `n_half ∝ mu^-2` functions analytically as a Gell-Mann–Oakes–Renner (GMOR) relationship within the spectral mass context" | V2 App. M.1 |
| **D47** | **Angular curvature and its round-off floor:** `K_theta_theta = delta / r*`, reconstructed as a difference of `O(\|\|K\|\|)` terms, so its absolute error is `~ eps*\|\|K\|\|` and its relative error `~ eps/delta` | V2 App. M.4 |
| **D48** | **Expansion variable:** with `mu^2_LO ≡ delta*Sigma(0)/F^2(0)`, the theory predicts relative leading-order error `x ≡ delta/(M_ch * mu^2_rad * f)` | V2 App. M.5 |
| **D49** | **Tree-level Goldstone theorem:** differentiating the stationarity condition along the vacuum orbit gives `grad^2 V * J q = 0` (V2 Eq. 2) — `Jq` is a Hessian eigenvector with eigenvalue zero | V2 App. B |
| **D50** | **SO(2) invariance condition:** `V(R(theta) q) = V(q)` for all theta (V2 Eq. 1), equivalent to a purely radial potential `V ≡ V(r)` | V2 App. B |

> V2 states explicitly: *"GMOR-proper results below, are a supporting result, not one of this paper's
> claims."* (App. M preamble). D47's `eps/delta` floor **applies to every `mu^2` probe in the paper**;
> V2 keeps all other sweeps at `delta >= 10^-4` where the floor is `~10^-12`.

---

## 3. ASSERTED WITHOUT DERIVATION — do **not** pre-register against these

Each of these is stated in the shorts but **has no derivation in the shorts**.

| # | Assertion | Where | Why it is not a derivation |
|---|---|---|---|
| **A1** | **The Lifetime Trilemma** — "exact value fidelity, amplitude-independent address hold, and amplitude-independent read latency. A system may only optimize two." | V5 §2.3; V1 §5 (as: "with a quadratic payload channel and a fixed read budget, a store cannot simultaneously possess…") | Stated as fact in both papers. **No proof, no derivation, no impossibility argument appears in either.** V1 adds the qualifier "with a quadratic payload channel and a fixed read budget" but does not derive the trilemma from it. V1 does draw one consequence — "a faded memory mathematically requires more integration steps to read" — again without derivation. |
| **A2** | **Throat latch impact `mu^2 ∝ A/l^2`** | V1 App. C Table 1 (a certificate-table cell) | A bare proportionality in a table cell. `A` and `l` are **never defined anywhere in the paper**, and no derivation appears in App. H (which derives only the causal box, the injection bound, and the gate Jacobian). **This is the closest thing in the corpus to a "width" law and it must not be treated as one.** |
| **A3** | **Emergent register capacity "≈ 1–1.6 bits"** | V2 §G.1 and App. L; V5 Fig. 8 caption | A reported figure-level characterisation of discrete washboard minima. No capacity derivation. |
| **A4** | **"Kinetic isotropy acts as the price for an equivariant write current"** | V2 §5 (Discussion) | Discussion-level statement; no derivation. |
| **A5** | **Minimum-separation certificate from lattice packing** | V5 App. F: "slots that are a lattice packing, so the canonical representation carries a minimum-separation certificate" | The *measured* structural spacing is **1.540000** (V5 App. F). **No closed form relating lattice geometry to that separation is stated anywhere in the corpus.** This is the only "separation" quantity in the three papers and it is a measured constant, not a derivation. |
| **A6** | **`G -> H` generalisation: the `dim(G/H)` orbit tangents are "exactly flat"** | V2 §5 | V2 itself says: *"This is a direction on which this paper offers no evidence, but is part of our future work."* |
| **A7** | **BIBO stability requires coercive-component membership** | V1 §3, App. C.2 | The *necessity* is demonstrated empirically (the `b = 5.0` zero-energy escape), not derived. V1 states it as "the mandatory operational clause" on the strength of the battery, not as a theorem. |
| **A8** | **Kinetic breaking: "A flat direction maintains `mu^2 = 0` under inertial anisotropy by Sylvester's law of inertia."** | V2 App. F | Cites Sylvester's law but gives no derivation in the paper. Plausible and probably right — but it is a one-line assertion, not a stated derivation. |

---

## 4. WHICH SHORT SUPPLIES WHICH WAVE-6 HUB'S PRE-REGISTRATION

| Wave-6 hub | Primary source | Laws it can pre-register against | What it must NOT assume |
|---|---|---|---|
| **`hub/retention`** | **V2** (primary: the band structure and the CRR) + **V5** (the gamma and T axes, the vault) | D5–D19 (band structure, `mu^-2`, the floor, crossover `h* ≈ gamma/2`, the latch `q_inf = q_0 + eps*p_0/(M*gamma)`, `gamma_crit = 2*eps*mu`, **both** log-slopes), D20–D25 (FDT scale, `D_theta`, the sign flip, `l_theta`), D26–D31 (the vault), D14 (`tau_max = Gamma/(2*alpha)`) | **No "exact-decay law"** exists under that name — pick one of §1.2's three and name it. The overdamped law is `≈`/`≃`, **not exact**. The underdamped slope is **not** +1 at finite gamma (D17). Every `T > 0` law requires fdt noise **and** a Newtonian kinetic mode. A1 (trilemma) is an assertion. |
| **`hub/potentials`** | **V2 App. B/D/M** (potential geometry, spectral mass, GMOR) + **V1 App. H** (reach and energy on the potential) | D15 (`mu^2` from the mass-whitened Hessian), D46–D50 (GMOR `mu^2 F^2 = delta*Sigma`, `K_theta_theta = delta/r*`, tree-level Goldstone `grad^2 V Jq = 0`, the SO(2) invariance condition), D34–D45 (causal box, squeeze bound, gated channel, gate defect, latch transport), D14 (confinement lifts every eigenvalue by `2*alpha`) | **There is NO Mexican-hat width or separation law and NO interpolation floor anywhere in the corpus (§1.1).** The only potential *shapes* with closed forms are V1's `V = (1/2)k q_0^2 - eps q_0^4` testbed and V2's probes `V -> V + delta(1 - cos n theta)` and `V -> V - delta(u . q)` — all three are **designed probes applied to a trained potential**, not a hat parameterisation. A2 (`mu^2 ∝ A/l^2`) is an undefined table cell, not a width law. A5 (minimum separation) is a measured constant, not a law. |

**Shared rider both hubs inherit from the papers.** Every quantitative result in all three shorts is
at **dimension 4** (V1 also uses dim 2), **hidden 64**, **eps = 0.05**, **single-core CPU**, on
**5 designed seeds and/or 3 emergent seeds**. V2 and V5 both classify every result as
**verification** (a designed SO(2) testbed — exact verification of the theory) or **evidence**
(a learned MLP checkpoint). A hub should pre-register against the arm it will actually run on, and
should carry the classification with the number.

---

## 5. Per-experiment rows

*In progress — appended below as completed. Every experiment and derivation in the three shorts gets
one row: what was run, on what object, the claim it supported, the number (MEASURED), and the derived
law it tests (DERIVED, cross-referenced to §2).*

*Completed 2026-09-20. Columns: **what was run · on what object · the claim it supported ·
the number (MEASURED) · the derived law it tests (DERIVED, cross-referenced to §2)**. Where a row's
number has no derived law behind it, the DERIVED column says so explicitly — those rows are
measurements, not verifications, and a hub must not read a law out of them.*

*Extraction caveat carried into this section: the tables below were re-read in PDF reading order
(`pdftotext -enc UTF-8` without `-layout`) wherever the `-layout` rendering interleaved columns.
Two cells remained genuinely unrecoverable from the text layer and are marked **[alignment
unrecoverable]** rather than guessed.*

---

### 5.1 V1 — *Certified Access* (analytic testbed)

**Shared configuration (V1 App. B.1, B.5):** JAX 0.9.0; relativistic kinetic mode (Newtonian for the
control arm); mass band `[4.0, 0.25]` giving `M_eff,0 = 4.0` and 16x inertial contrast; `c = 1`;
`eps = 0.05`; `T = 100`, so `L = T*eps*c/sqrt(M_0) = 2.5` and `v_max,0 = 0.5`; rapidity grid
`[0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]`; double well along coord 0 with barrier
`V_b = 1.0`; `d ∈ {0.8, 1.6, 2.4, 3.2, 4.0, 5.0}`; launch `q_0 = 0`, `p_0 = (1.2, 0, …) + 0.02 N`
(kinetic energy below barrier); landing criterion `min_traj |q_0 - d| < 0.4`; seeds `{0,1,2,3,4}`;
dimensions 2 and 4; reach rollout at `gamma = 0` (sharp box). **All experiments on standard laptop
CPUs.**

| # | What was run | On what object | Claim it supported | MEASURED | DERIVED law tested |
|---|---|---|---|---|---|
| **E1.1** | Reach battery — plain relaxation | K-basin double well, `L = 2.5` | Plain relaxation is mathematically escape-blocked | Landing rate **0** at every `d` (0.8/1.6/2.4/3.2/4.0/5.0); kinetic energy **0.72 < V_b = 1**; `det J = (1-gamma)^d` | D1 (conformal symplecticity) |
| **E1.2** | Reach battery — Lorentz squeeze `S_z(M)` | same | Squeeze cures escape and extends reach, but at energy exponential in rapidity | Landing **1/1/1** at `d<L`; **1/0/0** at `d = 3.2/4.0/5.0`; `det S_z(M) = 1.000 ± 4e-6` | **D37** (`det S = 1`), **D38** (`H(S_z z) <= e^{2|z|}H`), **D39** (reach bracket) |
| **E1.3** | Squeeze energy-bound sweep across rapidities | matched quadratic potential | The injected energy tracks the analytic bound | `(z, ratio, bound)` = **(0.25, 1.13, 1.65)**, **(0.5, 1.55, 2.72)**, **(1.0, 3.79, 7.39)**, **(2.0, 27.5, 54.6)**; `det S = 1.000 (±4e-6)` | **D38**, **D40**. V1 notes the bound is a *matched-quadratic* certificate — on a quartic well the raw ratio can legitimately exceed `e^{2|z|}` |
| **E1.4** | Reach-bracket rapidity prediction | analytic | The bracket predicts exactly where the squeeze stops | Reaching `d = 4.0` requires **`z >= 2.0105`**; `d = 5.0` requires **`z >= 2.64`**; both exceed the swept budget `z <= 2.0` | **D39** (`z >= arcsinh((d-0.4-L)M_0/p_0)`) |
| **E1.5** | Reach battery — gated channel | same | Flat access scaling; global reach at exact volume preservation | Landing **1** at every `d`; `det J = 1.0` **exact**; energy change **0.0 exact**; translation to `q = +1.000` | **D41** (`det J = 1`), **D42** (`ΔH_wh = V(q+Δ) - V(q)`) |
| **E1.6** | Reach battery — Newtonian control | same, Newtonian kinetic mode | Confirms the relativistic cap, not energy, is what bounds the flow | Landing **1** at every `d`; `det J = 1.0` | **D35/D36** — Newtonian `grad_p T = M^-1 p` is unbounded, so no box exists |
| **E1.7** | Reach battery — untrained state-replacing map `(q,p) -> (b,p)` | same | Identical landing rate to the channel, but volume-annihilating | Landing **1** at every `d`; `det J = 0.0` **exact** (forward-mode autodiff) | Contrast to **D41**. Non-invertible, erases the state |
| **E1.8** | Reach battery — dense / V-throat | same | Modulates the barrier; fails global reach | `d = 0.8`: **1**; `d = 1.6`: **0.8**; `d >= 2.4`: **0**; `det J = (1-gamma)^d` | D1. *(Note: the §3.1 summary column reports the **minimum** over each band, hence "0 0" for this row.)* |
| **E1.9** | Latch-transport cloud — gated channel, coset-tangent | 16 states drawn uniformly from the capture ball (radius 0.3 < `Δ` = 0.35), SO(2) sector, `default_rng(0)` | The certified channel transports the Goldstone charge and preserves its spread | `det J` **1.0000**; `ΔQ` mean **-0.0000**, std **0.0000**; predicted `p^T X Δ` = **0.0000**; max error **1.2e-7**; `std(Q_out)` = **0.0803** = `std(Q_in)`; pre-jump state recoverable, `q_in = q_out - Δ` to max err **2.2e-8**; predicted zero-shift precision **7.5e-10** | **D44** (`Q' - Q = p^T X Δ`) |
| **E1.10** | Latch-transport cloud — gated channel, across-coset | same | The shift is exactly the predicted constant | `det J` **1.0000**; `ΔQ` = **0.2500**, std **0.0000**; predicted `p^T X Δ` = **0.2500**; max error **0.0**; `std(Q_out)` = **0.0803** | **D44** |
| **E1.11** | Latch-transport cloud — random shift baseline | same | `det J = 1` alone is **not** the latch certificate | `det J` **1.0000** but `ΔQ` = **0.0972 ± 0.2465**; `std(Q_out)` = **0.2793** (spread scrambled) | Negative control for **D44** — volume preservation is necessary but insufficient |
| **E1.12** | Latch-transport cloud — state-replacing map | same | `det J = 0` irreversibly erases the coset content | `det J` **0.0000**; all 16 states map to a singular bit-identical charge, `std(Q_out)` = **0** | Contrast to **D44** |
| **E1.13** | BIBO battery | controlled non-coercive potential `V = (1/2)k q_0^2 - eps q_0^4`, `k = 1.0`, `eps = 0.02`, coercive edge `x_b = 3.5355`, `V_b = 3.125`; exits `b ∈ {1.0, 2.0, 3.0, 3.6, 4.0, 5.0}`; horizon `T = 2000`; damping 0.02; seeds `{0..4}` | `det J = 1` plus zero energy change is **insufficient** for BIBO stability; coercive-component membership is the operative clause | Max excursion `r` at 2T — **certified channel**: 1.01 / 2.01 / 3.01 / **0.09 / 0.09 / 0.09**; **channel ablation** and **state-replacing map** (identical): 1.01 / 2.01 / 3.01 / **102.13 / 103.43 / 104.83**. Escape rate certified **0** everywhere, blind **0/0/0/1.0/1.0/1.0**. Growth `r(2T)/r(T) ≈ 1.9`. **At `b = 5.0`: `ΔH = 0.0` exactly, the energy-only sub-level test admits, and the trajectory escapes.** At `b = 3.0`: `ΔH = 2.88`, bounded. | Tests the **limits** of D41/D42. ⚠ **A7** — the necessity of coercive screening is *demonstrated*, not derived |
| **E1.14** | Hard-gate unit test (gate varying during the jump) | analytic | A dynamically-varying gate breaks volume preservation by an uncompensated contraction | `det J` measured at **2.05** (vs 1 for a frozen gate) | **D43** (`det J = 1 + grad_g(q)^T Δ`) |
| **E1.15** | Numerical verification protocols | analytic, CPU | The exactness claims hold to machine precision | Causal cap to relative error **2.0e-12**; Jacobian preservation **1.000** with state-dependent variations tracking to **2.2e-16**; Goldstone latch transit error **< 1.8e-15** | **D35**, **D41**, **D44** |
| **E1.16** | Calibrated compute-rationing (Learn-then-Test gate) | trained CLU-EBM associative memory, MQAR, vocab 256, `kv ∈ {16,24,32}`, 5 seeds (base 42), 90 models, 400 epochs, lr 1e-3, batch 16, k_steps 50, buffer 128, friction 0.3, temperature 0.3, input noise 0.05; exact binomial p-values, targets `alpha ∈ {0.05, 0.1}` | The gate rations compute on clean retrieval; it is not a superior physics signal | LTT wrapper **held on all 30 evaluated cells**, coverage **0.647 ± 0.063** at `alpha = 0.05`. Step-reduction vs always-full budget: **4.81 ± 0.44x** (kv16), **1.57 ± 0.07x** (kv24), **1.14 ± 0.06x** (kv32) at 400 epochs. At 2000 epochs: **strict parity**, not dominance. Residual adds **at most +0.024 AUROC** over a standard readout margin. | ⚠ **No derived law.** Empirical calibration result. The exchangeability assumption between write-time probes and deployment queries is stated by V1 as only approximated by its jittered protocol |
| **E1.17** | Capacity x epoch-budget regime map | 198-job grid; epochs `{500, 1000, 2000, 4000}`, `kv ∈ {32, 64, 96, 128}`, embed_dim 16 (CLU dim 32), hidden 128, vocab 256, relax_steps 300, governor_sensitivity 0.95, seeds `{42..46}` | Required epochs scale with capacity; degradation past kv64 is an epoch-budget artifact, not a hard capacity limit | Fidelity / gate acc / intra-CLU step-reduction — **N128/kv32**: 500 ep → 0.76±0.09 / 0.31±0.04 / 1.2x; 2000 ep → 1.00±0.00 / 1.00±0.00 / **9.9x**. **N256/kv64**: 500 → 0.43±0.05 / 0.06±0.02 / 1.0x; 2000 → 1.00±0.00 / 0.99±0.00 / **9.5x**. **N384/kv96**: 500 → 0.40±0.03 / 0.02±0.01 / 1.0x; 2000 → 0.97±0.01 / 0.91±0.02 / **6.2x**. kv96 needs 4000 epochs for accuracy 0.975. **kv32 over-trains between 2000 and 4000 epochs, degrading 1.00 → 0.89.** | ⚠ **No derived law.** Empirical scaling map |
| **E1.18** | Evaluation cue-noise sweep | same gate, `sigma ∈ {0.3, 0.6}` | The compute dial confers **no** robustness to corrupted cues (declared negative result) | At `sigma = 0.6` on kv32: gate accuracy **0.36** while underlying CLU storage fidelity remains **1.0**. V1 records this "noise wall" as an open problem | ⚠ **No derived law.** Declared negative result |
| **E1.19** | Non-local routing grid | lattice distance `N ∈ {4, 8}`, embed_dim 12, vocab 128; router MLP hidden 32 (449 params), 300 epochs, lr 3e-3; workload mixes `[[0.5,0.5],[0.8,0.2],[0.95,0.05]]`; 5 seeds | A one-hop non-local edge has flat FLOPs under distance scaling — **but energy-gating it is beaten by a physics-free router on both FLOPs and accuracy** (declared negative result) | **N=4** (acc (L/D), FLOP): Local only 0.500 (1.00/0.00), 5.88e7 · Gated energy **0.887±0.139** (0.90/0.88), 1.18e8 · Dense 0.500 (0.00/1.00), 1.18e8 · Diffusion chain 0.652±0.159 (0.90/0.41), 1.76e8 · **Physics-free router 1.000±0.000 (1.00/1.00), 8.81e7**. **N=8**: Local only 0.500 (1.00/0.00), 5.88e7 · Gated energy **0.715±0.172** (0.82/0.61), 1.18e8 · Dense 0.448±0.070 (0.00/0.90), 1.18e8 · Diffusion chain 0.548±0.138 (0.82/0.28), 2.94e8 · **Physics-free router 0.948±0.070 (1.00/0.90), 8.81e7** | ⚠ **No derived law.** Declared negative result: the physics-free parameter-matched router wins on both axes at both distances |
| **E1.20** | FDT calibration of the Langevin noise scale | certified Markov kernel (App. G/I) | Correct FDT calibration removes finite-budget shadow bias | Shadow-bias finite-budget error **0.0995 → 0.0065** | **D20** (`sigma*_i = sqrt(M_eff,i T gamma (2-gamma))`) |
| **E1.21** | Squeeze-retry ablation at single-unit scale | already-reachable attractors | The squeeze addresses kinematic reach, **not** local selection (declared negative result) | **No performance recovered** | ⚠ **No derived law.** Declared negative result |

**V1 rows carrying no derivation (do not pre-register against these): E1.16, E1.17, E1.18, E1.19,
E1.21.** Also note the certificate table (App. C.1) cell **`adds mu^2 ∝ A/l^2`** for the throat's
latch impact — see **A2**: `A` and `l` are never defined in the paper.

---

### 5.2 V2 — *The Price of a Flat Direction* (trained checkpoints)

**Shared configuration (V2 App. F):** Experiment-D configuration of the reference unit — **dimension
4, hidden 64, learned-Newtonian kinetic term, `dt = 0.05`, a unit circle at 256 points, 150 epochs**.
Checkpoints are either the **designed** family (architecturally SO(2)-invariant `V`, 5 seeds) or the
**emergent** family (generic MLP `V`, 3 seeds). All Sec. 4 measurements at `T = 0`. `gamma = 0.05`
throughout unless swept. Single-core laptop CPU. V2 classifies designed-testbed outcomes as
**verification of the theory** and learned-system outcomes as **empirical evidence**.

| # | What was run | On what object | Claim it supported | MEASURED | DERIVED law tested |
|---|---|---|---|---|---|
| **E2.1** | Analytic tilt sweep, curvature measured via the Hessian | trained designed SO(2) checkpoints, **70 tilt configurations**, 5 seeds, dim 4 | The constitutive identity (Hessian spectral mass = one-step-Jacobian gap) holds | Flat mode measures `mu^2 ≈ 2.4e-15` (machine precision). Hessian-derived spectral mass aligns with the measured one-step-Jacobian gap to **3.2e-10 across all 70 tilt configurations** | **D15** (`mu_k^2 = lambda_k(M^-1/2 grad^2 V M^-1/2)`), **D3/D4** |
| **E2.2** | Curvature-ratio check across 4.5 orders of magnitude of tilt | same | The retention law's curvature input is exact | `mu^2_meas / mu^2_pred` = **1.000000 ± 5e-12** at every tilt and seed, across **4.5 orders of magnitude** | **D15**, **D46** (`mu^2 F^2 = delta*Sigma`) |
| **E2.3** | Overdamped-branch slope fit | same, `n = 35` overdamped rows | `n_half ∝ mu^-2` | Fitted slope **-0.985** (predicted **-1**) | **D5**, **D6** |
| **E2.4** | Floor saturation past the crossover | same | Past the crossover the half-life decouples from curvature entirely | Saturates at **27.03 steps** at `gamma = 0.05` | **D7** (`n_half = 2 ln2/(-ln(1-gamma))`; at `gamma = 0.05` this evaluates to 27.03) |
| **E2.5** | Exceptional-point signature 1 — metric bifurcation | same | The crossover is a genuine exceptional point, not a smooth knee | Overdamped: envelope half-life and first-crossing time agree to **0.01%**. Underdamped: they **diverge by up to 3.2x** at the deepest tilt | **D8**, **D10** (defective matrix, one Jordan block) |
| **E2.6** | Exceptional-point signature 2 — square-root frequency onset | same, fine multiplicative grid | The phase opens with the predicted 1/2 exponent | Below `h*`: `omega = 0` **exactly**. Above: `omega ∝ sqrt(h - h*)` with fitted slope **0.5165** (predicted **0.5**). Onset holds down to `h - h* = 2.6e-6` on a grid `h - h* ∈ [-4.2e-3, 2.6e-2]`. Where the quality factor permits (`f = 2, 4`) trajectory frequency matches Jacobian frequency to **0.06–0.3%** | **D11** (`phi = arg lambda_+ ∝ sqrt(h - h*)`) |
| **E2.7** | Latch transport-law check | designed flat mode, across two orders of magnitude of `gamma` | The latch obeys its closed form and freezes the written angle | `q_inf = q_0 + eps*p_0/(M*gamma)` obeyed to **≤ 1%**; written angle frozen to **≤ 1.2e-15 rad** | **D12** (the latch), **D5**'s `mu = 0` corner |
| **E2.8** | Mo [2026] single-exponential lifetime estimator, run **unchanged** | V2's own trained checkpoints; their exact code-level protocol (phase 0.35 rad, threshold 0.2 rad, censoring cap 15000 steps), 5 seeds, **14 breaking magnitudes** | The published rate-based estimator is the overdamped *face* of the CRR — contained, not contradicted | Overdamped measured/predicted ratio **1.012 ± 0.000 to 1.029 ± 0.001** (their reported median: 1.013); `corr(log pred, log meas) = 0.9987` overdamped-only. **Above the exceptional point it fails predictably**: mispredicts by **≈ 3.2x** at the deepest trained-model breaking magnitude (`delta = 4`), a **2.202 ± 0.155 delay spike** at the crossover, declining to **0.309 ± 0.012** deep underdamped. On exact analytic maps the divergence extends to **≈ 5x** | **D5** (overdamped), **D7/D8** (why a first-order estimator must fail above `h*`) |
| **E2.9** | Autonomous retention, S1 protocol | designed CLU (5 seeds), generically-trained CLU, and LSTM / LEM / coRNN baselines trained with learning-rate sweeps; dim 4, hidden 64, drift threshold 0.2 rad, deterministic `T = 0`, single-core CPU | The structural triad (exact latch, `mu^-2` scaling, bounded coordinate motion) is absent in learned baselines | Phase lost within **≈ 5.6 / 56 / 69 map-steps** (coRNN / LEM / LSTM), coordinates randomising **beyond 1.2 rad**. Generically-trained CLU holds **≈ 263 map-steps** with motion bounded **≤ 0.35 rad**. **The designed unit never drifts across all 5 seeds.** Baselines are the median across 5 seeds, emergent across 3; the designed curve is a **single representative checkpoint** | **D12** (the latch). ⚠ V2 states this is "empirical evidence only, generated by a local computing setup" |
| **E2.10** | Emergent-arm test: does symmetric training data alone produce a flat direction? | generic MLP potential trained on symmetric data, 3 seeds | **No** — flat directions must be designed, not trained for (declared negative result) | Softest emergent curvatures **`mu^2` = 5.1 / 5.9 / 5.4 e-2** across 3 seeds, versus designed flat **≈ 2.4e-15** — a gap of **13–14 orders of magnitude**. The emergent arm operates on **discrete washboard minima (≈ 1–1.6 bits capacity)**, where any continuous written coordinate fully relaxes | ⚠ Declared negative result. The damping-optimum law still holds here mathematically, but the continuous coset register is a **designed** feature |
| **E2.11** | Wake-sleep contrastive-divergence erosion | designed vacuum, 1000 epochs | The primary training objective is itself the perturbation that destroys the register | CD drives the order parameter `r*` to zero; **the designed vacuum inverts in 8/8 runs**, eliminating the orbit and generating invariant minimisers with strictly positive `mu^2` | Manifestation of the fine-tuning problem. ⚠ No derived law — an observed training pathology |
| **E2.12** | The anchored cure, run to 20x the erosion horizon | `V(data)` energy anchor, `lambda = 100`, 5 seeds; then **3000 anchored epochs** (≈ 20x the baseline erosion horizon), 3 seeds | The CRR **survives** the corrective intervention intact | Anchor maintains `r* = 0.911 ± 0.016` (5 seeds), trading weaker noise rejection and higher wake MSE. At 3000 epochs: curvature ratio holds to **1.5e-12**; slope **-0.956** (per-point fit over all overdamped rows) / **-0.961** (seed-mean OLS over the 7 overdamped `delta`); **floor unchanged at 27.03**; exceptional-point onset **bit-identical at 0.5165** (Fig. 4 fitted 0.516) | **D5, D6, D7, D11** — all four survive |
| **E2.13** | Physics-prior Pareto trade-off, parameter-matched | dim 4, 150 epochs, 3 seeds, identical MLP potential; parameter-matched to ±0.2% (CLU 4549 / broken-volume 4557 / twin 4551) | Volume conservation buys boundedness and flatness and **costs** raw fit | BIBO **1.00 vs 0.33** (broken-volume baseline); near-flat direction `mu^2` **0.008 vs 0.122**; vacuum survival under CD `r* = 0.72 vs 0`. **Eval MSE: unconstrained twin 0.0128 vs constrained 0.190.** The broken-volume baseline recovers **≈ 2.4x** of the gap purely by leaking volume | **D1** — the penalty is attributed to the volume-conservation axiom (contraction-forbidden) |
| **E2.14** | Cumulative fit MSE against horizon | stationary circle-vacuum orbit, wake-only | The constrained unit's asset is a bounded plateau, not the lowest MSE | Horizon (Twin / CLU / Broken-Volume / LSTM): **10** 2.9e-6 / 2.2e-5 / 1.1e-4 / 7.5e-2 · **50** 6.2e-5 / 1.2e-3 / 1.0e-3 / 4.9e-2 · **100** 2.4e-4 / 1.2e-2 / 4.1e-3 / 4.9e-2 · **500** 1.1e-2 / 1.8e-1 / 9.7e-2 / 7.7e-2 · **1000** 3.2e-1 / 2.0e-1 / 1.2e-1 / 9.9e-2 · **5000** **1.96e2** / 2.2e-1 / 1.4e-1 / 1.3e-1. Twin diverges after ≈ 700 steps; CLU holds a plateau of ≈ 0.20–0.23 | **D1** |
| **E2.15** | Independent wake-only variant ladder | same family, separate seeds/objective | Only global damping recovers the lost fit; a learned friction field does not substitute | Model / Eval MSE / % gap recovered / BIBO / latch drift / flat `mu^2` — **Twin (no physics)** 0.0047 / 100% (ref) / 1.00 / — / — · **CLU conservative (`gamma = 0`)** 0.2066 / 0% (ref) / 1.00 / 0.778 / 5.2e-3 · **+ global `gamma = 0.05`** 0.0216 / **92%** / 1.00 / 0.778 / 5.2e-3 · **+ learned field `gamma_phi` (K = 2)** 0.2548 / **-24%** / 1.00 / 0.706 / 2.1e-2 | **D1**, **D26** (the gated friction field). ⚠ V2: absolute MSE scales are **not** cross-comparable with E2.13's table; only intra-table ratios |
| **E2.16** | Per-step compute measurement | single-core CPU, f32 XLA, median of 7 scan-amortised repetitions over 2e5 steps; CLU step = two `grad_q V` backprops through a `dim 4 → 64 → 64 → 1` tanh MLP | The raw map-step advantage is **not** a normalised compute win (declared negative result) | Params / FLOPs per step / wall ns per step — **CLU Verlet (h64, dim 4, KDK)** 4549 / 36148 / 1554 · **LSTM cell (h16)** 1186 / 2512 / 252 · **LEM cell (h16)** 1186 / 2400 / 500. Normalisations (CLU vs LSTM / vs LEM): raw map-steps **3.8x / 4.7x longer**; **FLOP-normalised 54.8x / 70.7x more**; **wall-normalised 23.5x / 14.6x more** | ⚠ **No derived law.** Declared negative result. V2 states the confound: not width-matched (CLU h64 vs baselines h16) |
| **E2.17** | GMOR verification with a **linear ambient spurion** `V -> V - delta(u . q)` | 8 trained checkpoints, `delta ∈ [1e-8, 0.3]`, f64 probes, zero retraining | The spurion separates `mu^2`, `F^2` and `Sigma` so the identity can be checked independently | **`max |mu^2 F^2 - delta*Sigma| = 1.33e-15`** across the 8 checkpoints. Relative exactness **≤ 2.7e-14** for `delta >= 1e-2`. The running condensate `Sigma` scales by **+16.1% to +210.1%** across the `delta` grid | **D46** (`mu^2 F^2 = delta*Sigma`), **D48** |
| **E2.18** | Autodiff-Hessian round-off control **on the analytic Mexican hat** | analytic Mexican-hat potential, `delta ∈ [1e-8, 0.3]` | The growing relative deviation at small `delta` is a **round-off floor**, not a failure of the law | Cancellation-free Hessian: relative deviation **1.1–2.2e-16** at every `delta`. Autodiff Hessian on the **same** potential: **2.28e-8 → 4.22e-15**, exactly `∝ 1/delta` | **D47** (`K_theta_theta = delta/r*`, relative error `~ eps/delta`). ⛔ **This is the corpus's only occurrence of "Mexican hat" — see §1.1. It is a numerical control, not a width law.** |
| **E2.19** | Damping-corollary sweep at fixed spectral mass | designed checkpoints | `n_half(gamma)` is **non-monotone**, with a minimum at the critical damping | The curve is **V-shaped**, minimum directly beneath **`gamma* ≈ 2*eps*mu`**. V2's consequence: an isolated retention measurement at a single `gamma` is **uninformative about the sign of `∂n_half/∂gamma`** unless its position relative to the critical threshold is specified | **D16** (`gamma_crit = 2*eps*mu`) |

**V2 rows carrying no derivation (do not pre-register against these): E2.10 (negative), E2.11,
E2.16 (negative).** V2 also declares App. M as *supporting*, not a claim of the paper (E2.17, E2.18).

---

### 5.3 V5 — *Retention You Can Predict, Scope and Delete*

**Shared configuration (V5 §2.1, §3, Table 1):** dim 4, hidden 64, `eps = 0.05`, 150 epochs,
single-CPU. **Designed** units = architecturally SO(2)-invariant `V`, 5 seeds (**verification**);
**emergent** units = MLP `V` trained on identical symmetric data, 3 seeds (**evidence**).
**Every `T > 0` cell uses `langevin_noise="fdt"` with a Newtonian kinetic mode** — V5 states the
legacy default "fundamentally violates these thermodynamic laws". Every half-life is reported with
its read tolerance `Delta` and the ratio `Delta/l_theta`.

| # | What was run | On what object | Claim it supported | MEASURED | DERIVED law tested |
|---|---|---|---|---|---|
| **E5.1** | `n_half(gamma)` V-curve on the massive radial mode | 5 trained designed SO(2) seeds, `T = 0`, one-step Jacobian; per-seed `mu^2_rad` = 0.670, 0.771, 1.190, 1.093, 1.348 | The retention dial has a computable critical point fixed by the spectral mass | argmin lies between **0.076 and 0.096** against computationally predicted bounds **0.082–0.116**. Asymptotic log-slopes **-1.006** and **+1.23 to +1.27**. Mass-independent floor of **exactly 27.03 steps** at `gamma = 0.05` | **D16** (`gamma_crit = 2*eps*mu`), **D5/D6** (overdamped `mu^-2`), **D7** (floor), **D17/D18** (the asymmetric slopes) |
| **E5.2** | The same V-curve on the emergent coset mode, read on **four instruments** | 3 emergent MLP seeds, `T = 0`, **48 `gamma` values**, `Delta = 0.05 rad`. I-J = exact one-step Jacobian (`n_half = ln2/(-ln|lambda_ret|)`); I-R1 = rollout first crossing of `|theta| <= Delta/2`; I-R2 = rollout last crossing; **I-R3 (primary)** = rollout envelope-rate fit over the late window | The law transfers from designed to learned units; the rate instrument reproduces the Jacobian, both threshold instruments fail below `gamma_crit` | `gamma_crit` per seed (42/43/44): **0.02334 / 0.01424 / 0.02265**. **Argmin/`gamma_crit`** — I-J 0.8994/0.9046/0.9055 → **0.9032 ± 0.0027**; I-R1 → 0.1048 ± 0.0252; I-R2 → 0.2168 ± 0.0668; I-R3 → **0.9001 ± 0.0052**. **Slope below** — I-J → **-1.0020 ± 0.0003**; I-R1 → +0.0623 ± 0.0120; I-R2 → -0.6240 ± 0.4948; I-R3 → **-0.9827 ± 0.0134**. **Slope above** — I-J → **+1.1182 ± 0.0107**; I-R1/I-R2 → +1.0964 ± 0.0057; I-R3 → **+1.1172 ± 0.0101**. I-R3 fit quality `R^2 >= 0.9955` (median 0.9996) in every cell. V5 headline (Table 1): argmin **0.902 ± 0.003 × gamma_crit**, overdamped log-slope **-1.0020 ± 0.0003**, underdamped **+1.116 ± 0.011** | **D16**, **D17**, **D18**. ⚠ **No rollout `n_half` is ever quoted from a threshold instrument below `gamma_crit`** — V5's own rule |
| **E5.3** | Cross-instrument check | same cells | The gap between instruments is a **level**, not a rate | The two argmins agree to **0.35%** in units of `gamma_crit` (0.9032 vs 0.9001). `Gamma_jac/Gamma_R3` asymptotic-window mean **0.995 ± 0.003** (per-seed values 0.9935 / 1.0000 / 0.9948 appear in Table 4; **[alignment unrecoverable]** from the text layer). Threshold ratio `n_jac/n_R1` = **1.3307 / 1.7981 / 1.8204** (seeds 42/43/44), CV **2.2–3.7%** over 21–25 `gamma`-points, `|d log(ratio)/d log gamma| <= 0.0034`. Seed 44's full-window value **0.9216 ± 0.0027** sits **0.0004 below its pre-registered band** because that window still contains a second, faster mode (early-window ratio 0.539). **Measured cause of the threshold offset**: the ring write deposits only a fraction of its amplitude into the slow branch (envelope-fit intercept over `Delta` = **0.638 / 0.644 / 0.327**) | **D5** — the decay *rate* is instrument-independent |
| **E5.4** | Finite write-amplitude sweep | `Delta = 0.05 / 0.2 / 0.5 rad`, I-R3 | The argmin is stable across a 10x range of write amplitude | I-R3 argmin: **0.8928/0.9007/0.8968** (seed 42), **0.9047/0.9045/0.9174** (seed 43), **0.9028/0.9037/0.9070** (seed 44). Max shift **+1.4%** across a full order of magnitude; slope variances **< 1.9%**. At `Delta = 0.5` every iteration relaxed to baseline (`|theta_final| <= 6.5e-10`) — no seed's write cleared its washboard barrier | **D16** |
| **E5.5** | Left-branch exclusion test | `gamma = 0.002`, emergent seeds 42 (`mu^2 = 5.449e-2`) and 43 (`mu^2 = 2.029e-2`) | Below `gamma_crit` the coset eigenpair is complex and retention is **independent of `mu^2` and of the checkpoint** | Both seeds return **identical** `lambda_ret = 0.998999499` and `n_half = 692.5`, against the calculated `2 ln2/0.002 = 693.1` | **D7** (the `|lambda|^2 = 1 - gamma` underdamped branch), **D4** |
| **E5.6** | Coset diffusion law, 25-cell grid | 5 `gamma` × 5 `T`, one trained designed checkpoint (seed 44), `Delta = 0.5 rad` | The derived diffusion coefficient is right on both axes | **`D_hat/D_pred` = 1.0068 ± 0.0219**, bounded between **0.9644** and **1.0484**. `d log D/d log T` = **+0.998, +0.997, +0.992, +1.010, +1.006**. `d log D/d log gamma` = **-1.014, -0.980, -0.999, -1.001, -0.982** (after isolating the discrete `(2-gamma)` factor) | **D21** (`D_theta = eps*T*(2-gamma)/(2*F^2*gamma)`), **D22** |
| **E5.7** | The sign flip: friction preserves, temperature erases | designed testbed, 5 seeds, `T = 1e-3` baseline, `Delta = 0.5 rad`, `Delta/l_theta < 0.05` | `n_half ∝ gamma^+1` on the flat coset — the opposite sign to the massive mode | `d log n_half/d log gamma` = **+0.9552 ± 0.0422**, positive in **10/10** examined seed x temperature conditions. Effect size for `gamma: 0.05 → 0.2` is **3.77 ± 0.23x** (per-seed 3.55, 3.64, 3.61, 3.89, 4.15). Inverse thermal driver `d log n_half/d log T` = **-0.956 to -0.979**. Massive underdamped mode responds with slope **-1.006**, and `T` there sets only the fluctuation floor (max/min ≈ 1.013) | **D23** (`n_half ∝ gamma/[(2-gamma)T]`) |
| **E5.8** | `T = 0` latch, exhaustive friction sweep | designed SO(2) coset, **22 `gamma` values × 5 seeds**, `gamma ∈ [0.002, 0.5]` | Friction cannot erase the latch | **`||lambda_flat| - 1| <= 1.7e-14`**; drift restricted to **`<= 4.9e-12 rad per 200k integration steps` across 30/30 evaluated cells** | **D12**, and the `|lambda_flat| = 1` statement of **D13**'s companion (V5 App. G.4) |
| **E5.9** | `gamma = 0` control | same | `gamma = 0` is **not** a latch — it is a Jordan block and the coordinate drifts ballistically | **142.7 rad drift over 20k steps** — i.e. over a 10x *shorter* horizon than E5.8's 200k | **D13** (`theta_n = theta_0 + n*eps*p_theta/F^2`) |
| **E5.10** | Friction-hole vault, designed arm | designed SO(2) with an analytic position-gated friction field, 3 seeds, `T = 1e-3`, `Delta = 0.5 rad` | A localised friction hole is simultaneously a brake and a refrigerator, giving a `(gamma_eff/gamma)^2` vault | `D_hat`-estimator vault: **8.42x, 22.39x, 44.31x, 107.77 ± 4.78x** at `gamma_phi = 0.1/0.2/0.3/0.5`, against the absorb-baseline prediction **110.25**. Uniform-scalar-friction control at matched `gamma_eff`: **13.28 ± 0.12x**. Raw first-passage ratio **86.97 ± 2.94x** — **explicitly not quoted as the vault** because it is boundary-layer biased on the outside arm (`Delta/l_theta = 0.079`). The diffusion constant differs by **8x** between a friction field and a scalar friction at the same `gamma_eff` | **D26** (`gamma_eff`), **D28**, **D29** (`vault = (gamma_eff/gamma)^2`) |
| **E5.11** | The refrigerator law, designed arm | same | The field is **absorb-only**; the coupled-bath alternative is refuted | `Var(p_i)/(M_i T)` = **0.9986** at `gamma_phi = 0`, then **0.36171 ± 0.00129** (0.1), **0.23044** (0.2), **0.17513** (0.3), **0.12600 ± 0.00031** (0.5) — against the pure-absorb prediction **0.12591**. `T_local` drops to **1.26e-4** vs `1e-3` outside: a **7.94x** refrigerator. **The coupled-bath hypothesis predicts a vault of 13.88x and is comprehensively refuted**; the two hypotheses separate by **7.942x** | **D27** (`T_local`), **D30** (the rejected alternative), **D31** (separation = refrigerator factor, identically) |
| **E5.12** | `T = 0` vault control | same, `T = 0` | The hole erases nothing at absolute zero | Drift **<= 1.75e-12 rad per 200k steps, inside and outside** | **D12** |
| **E5.13** | Confinement by the hole | same | The localised field confines the register | The state fraction outside the boundary (`|theta| > 1 rad`) drops **definitively to 0** within the hole | **D14** (confinement lifts every Hessian eigenvalue by `2*alpha`) |
| **E5.14** | Refrigerator law, **emergent** arm | 2048 walkers × 40 samples, 3 emergent seeds, **24 field cells**, `T ∈ {4, 8}e-3` | The refrigerator law transfers exactly to a learned register | Over all 24 cells **observed/absorb = 0.9998 ± 0.0019**. At `gamma_phi = 0.5`: `T_local/T` = **0.12570**, a **7.955x** refrigerator against **7.942** predicted. **The coupled bath predicts 1.0 everywhere and is measured at 0.2235** | **D27**, **D30**, **D31** |
| **E5.15** | Vault, **emergent** arm | emergent coset, `mu^2_adiab = 2.77e-2 to 8.93e-2`; window-free Ornstein-Uhlenbeck fit `MSD(n) = A(1 - e^{-n/tau})`, `D_hat = A/(2*tau)` (the designed block estimator is biased low by ~55% on bounded cells) | The absolute vault transfers; **the contrast metric does not** | Law-referenced absolute vault **106.1 ± 5.0x** against the prediction **110.25x**. ⛔ **But the measured field-to-scalar control variance is 23.39 ± 10.06 against a pre-registered bound of [6.5, 9.5] — V5 reports this as TRIGGERING ITS OWN DECLARED FALSIFIER**, because the control arm intrinsically delocalises while the field adheres to the absorb limit | **D29** holds in absolute terms; **the contrast statement is reported by V5 as native to the designed architecture only** |
| **E5.16** | Laundering/confinement control, emergent arm | `T = 4e-3`, 1024 walkers, burn-in 20 000–80 000 steps, spread by IQR, "hop" = `|theta| > 1 rad` | Friction alone does not confine — but the control itself failed | ⛔ **V5 declares this comparison VOID**: *"The pre-registered stationary-spread comparison is void here and no in/out spread ratio may be quoted from these checkpoints: the control that must return exactly 1.000 returns 0.4586 ± 0.1181."* Per-seed hop fractions are tabulated (e.g. seed 42: no hole 5.50%, scalar control 0.73%, hole 0.0000) | ⚠ **Void. Do not pre-register against any in/out spread ratio from this table.** |
| **E5.17** | Crossover temperature `T*` | 2 emergent seeds + 1 matched designed control (an `n < 3` cell), `Delta = 0.5 rad`, `Delta/l_theta < 0.06` | Bias-corrected, the emergent coset joins the exact-Goldstone law above a crossover temperature | The raw excess over the exact-Goldstone law is **the same on the matched designed control as on the emergent seeds**, so the excess is **estimator bias, not physics**. Bias-corrected, the emergent coset joins the law above **`T* ≈ 3e-3`**; below it the cells are confinement-protected | **D21/D23**. ⚠ V5 flags this explicitly as an `n < 3` cell |
| **E5.18** | Does the emergent unit have a continuous coset register? | written coset value `delta` held **20 000 steps** at `T = 0`, `gamma = 0.05`; 3 emergent MLP seeds + matched designed control | **No** (declared negative result) | On **every** emergent seed the value relaxes to the washboard minimum and **nothing is retained**; the designed checkpoint returns `delta` **exactly**. This is the evidence behind the **≈ 1–1.6 bits** capacity statement | ⚠ Declared negative result. See **A3** — the bit figure itself is not derived |
| **E5.19** | Exact store-level deletion | **24 exhaustive structural orders at `n = 4`**; **200 randomised sequence orders at `n ∈ {8, 16, 40, 64}`**; **100 write/delete interleavings** plus mid-decay continuous deletes; Blelloch–Golovin stable-matching table rules with the waitlist engaged | Post-deletion the store is byte-identical to a state that never held the item | Deletion **exact across all evaluated loads, 0.29x to 1.71x of standard lattice capacity**. Structural spacing enforces an exact boundary of **1.540000**. Structural centres, payloads, physical amplitudes and the active logic mask are **byte-equal**; the energy profile `V(q)` is **byte-equal across 64 randomised direct queries** | **D32** (decay commutes with deletion), **D33**. ⚠ Store-level only: V5 claims **no certified `(epsilon, delta)` unlearning**; the frozen encoder and residue in a learned landscape are separate channels. ⚠ The **1.540000** spacing is **MEASURED** — see **A5**, there is no closed form for it |
| **E5.20** | The negative packing price | ceiling `K = 64`, identical geometric seeds | Canonical placement costs admissions relative to a stochastic allocator — but far less than the stochastic allocator's own ceiling | Canonical stable-matching rule admits **61/64** deterministically at `delta = 0` (per-offered 0.9531); climbs to **64/64** once lattice sizing is inflated by **x1.05**. The stochastic refuse-and-relocate allocator admits only **43/64** (per-offered ceiling 0.6719) | ⚠ No derived law; a measured allocation price |
| **E5.21** | Overflow **without** the waitlist | 7 baseline cells receiving 8 capacity offers | Structural exactness collapses predictably at peak overflow | **`AUC(n_live)` returns 1.00000 ± 0.0000 and the byte-equal architectural fraction drops to 0.0000`** | Boundary of **D32** — the "sufficient budget for the cell count" precondition |
| **E5.22** | The lifetime trilemma and retrieval geometry | shipped store with the two-phase read geometry | Physical decay yields retrieval geometry, not cryptographic privacy | The shipped store abandons amplitude-independent address hold: lifetime correlation **`r = -0.85`** scaling with `a_i^2`. Value-index criterion **`A = 1 - tol/|a_i|`**, hard-death amplitudes **0.90/0.80/0.70/0.30**. **`R50` contracts 1.146 / 1.083 / 0.979 / 0.874 / 0.752** as `A: 1 → 0.5 → 0.2 → 0.1 → 0.06` — a **1.52x contraction**, matching the pre-registered saddle prediction at all five points (±0.20). A TTL vector store's lookup radius is a **constant hard step at 0.75–0.77, independent of age**. The gated channel contracts `R50` from **1.135 → 0.771** | ⛔ **The trilemma itself is A1 — asserted, never derived, in either V5 or V1.** The `R50` contraction is a measured retrieval geometry. **V5 defines no deletion-cost statement** because canonical synchronisation mandates an **O(n) rebuild per operation** |
| **E5.23** | Laundering control (adversary models) | designed store, two-phase read geometry; **8 active targets × 3 seeds, 128 paired environments per target, 18 continuous amplitude limits** | Physical decay provides retrieval geometry, not privacy | Physical decay / TTL flag (same store) / limit resolution separation — **white-box architecture, fully exact**: 1.000 / 1.000 / 0.000 · **query-level membership depth, fully exact**: 0.983 / 1.000 / 0.017 · **query-level membership depth, `sigma_obs = 0.1`**: 0.559 / 0.996 / 0.437 | ⚠ No derived law. Measured adversary-model separation |

**V5 rows a hub must treat with care:** **E5.15** (V5's own declared falsifier was triggered on the
emergent contrast metric), **E5.16** (V5 declares the comparison **void**), **E5.17** (`n < 3` cell),
**E5.18/E5.20/E5.22/E5.23** (no derived law behind the number).

---

## 6. Summary counts

- **3 distinct papers** from 5 files (two byte-identical duplicate pairs).
- **50 DERIVED laws** catalogued verbatim (D1–D50), each with paper and section.
- **8 assertions without derivation** flagged (A1–A8).
- **~62 experiments / measurement batteries** inventoried (E1.1–E1.21, E2.1–E2.19, E5.1–E5.23).
- **2 laws the wave-6 plan assumes that are NOT in the shorts** — the Mexican-hat
  width → interpolation floor (**absent entirely**) and "the exact-decay law" (**no such named law;
  three non-interchangeable candidates**). Both are raised in §1.
- **3 results the papers themselves disqualify**: V5 E5.15 (declared falsifier triggered), V5 E5.16
  (comparison declared void), V5 E5.17 (`n < 3`).

**Dial: the record.** This spoke's falsifier was *"a derivation the physics hubs need is not actually
in the shorts — in which case say so explicitly."* That condition **fired**, on the Mexican-hat
width → interpolation floor. It is reported in §1.1 rather than papered over, and no formula has been
invented to fill the gap.
