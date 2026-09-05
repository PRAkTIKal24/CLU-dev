# thread9-mh-kernel — physics-theorist report

**Task + acceptance criterion:** determine whether *squeeze-proposal + Metropolis–Hastings acceptance* is a **certified MCMC kernel** on the trained CLU memory EBM; characterise the governor-in-the-loop, the latch/coset erosion, and the temperature semantics; deliver verdicts, a V1-grade physics paragraph, and a w9 experiment spec. Every claim tagged **[proven]/[verified]/[design hypothesis]**.
**Status:** done. Theory + 4 numerical checks (pure numpy float64, seed-fixed); all load-bearing claims verified.
**Output consumed by:** Hub → V1 short (horizon/mechanism section, foldable at revision) + w9 tasking.

---

## Flag-provenance

| item | value |
|---|---|
| repo commit (context) | `63fea62` region (w8 integration HEAD per paid-access-experiments); **repo read-only, no code touched** |
| numerics | `.claude/scratch/thread9-mh-kernel/checks.py`, `/Users/user/Desktop/CHLU/.venv/bin/python` (numpy **2.4.1**, float64), seeds `default_rng(1..30)` |
| toy EBMs | (C1) asym. quartic well `V=¼(q²−1)²+0.3q`, m=1; (C2) harmonic `V=½q²`; (C3) SO(2) radial well `V=½k(‖q‖−r₀)²`, k=4, r₀=1.5; (C4) symmetric quartic `V=¼(q²−1)²` |
| MH config | T=1, squeeze rapidity ζ₀∈{0.3,0.5,0.6}, sign-symmetrized, HMC-style Gibbs momentum refresh; N=0.6–0.8M steps, 10% burn-in dropped |
| relevant F5 objects | Prop-12 (S^(M) symplectic, det=1, injection ≤e^{2\|ζ\|}H), Prop-9 (FDT σ\*), Prop-3 (conformal symplecticity det J=(1−γ)^d), §4 latch/Goldstone charge Q=pᵀXq |
| ties to code flags | `kinetic_mode`, `use_governor`, `sleep_temperature`, `langevin_noise∈{legacy,fdt}`, `dt`, `γ` |

All numbers quoted inline are from one `checks.py` run; consolidated in **Appendix N**.

---

## 0. Headline (one paragraph)

Squeeze + MH **is** a proper MCMC kernel on the memory EBM — but only as an *HMC-family* kernel, and the certificate is narrower than the retry-cascade pitch assumes. The squeeze `S^{(M)}_ζ` is symplectic with `det J = 1` exactly (F5 Prop-12), so the Metropolis ratio needs **no Jacobian correction** and the acceptance is exactly `min(1, e^{−ΔH/T})` — this is precisely why HMC uses symplectic proposals. Detailed balance w.r.t. the Gibbs measure `π ∝ e^{−H/T}` holds **[proven; verified, Check 1, L1=0.0095]** *provided* two things the bare pitch omits: (i) the proposal is **reversible**, supplied by sign-symmetrizing ζ (or, as we found, by the momentum-refresh's own p-sign symmetry), and (ii) the chain is **ergodic**, which the squeeze family alone is **not** — `{S_ζ}` is a one-parameter subgroup whose orbit of any point is a single hyperbola, so squeeze-only is reducible (Check 1, L1=1.81, stuck). Ergodicity requires HMC-style momentum refreshment between squeezes. Against HMC proper, the squeeze **replaces the leapfrog trajectory** with one linear, gradient-free, mass-preconditioned shear: it buys certificate/implementation simplicity (no ∇V_θ at test time, no L/step-size tuning, det=1 exact) at the cost of HMC's gradient-informed mixing (squeeze injects up to `(e^{2|ζ|}−1)H`, so it is a high-rejection, random-walk-class move). Two things then break or bound the pitch: **the governor destroys the invariant** — state-dependent γ(H)>0 is non-π-preserving dissipation, so the actual squeeze-then-relax cascade is **Metropolis-within-annealing**, converging not to π but to a colder, MAP-seeking measure (Check 2, T_eff: 1.0→0.61 as γ:0→0.2); and **certified retry erodes the latch** — the Gibbs measure is exactly flat along the Goldstone coset, so any proposal with a coset-tangent component is accepted with probability 1 and random-walks the stored register at rate `D=½s²` per step, `N_erode≈(Δ_read/s)²` (Check 3, D=1.29e-3 vs predicted 1.25e-3; even the "charge-preserving" isotropic squeeze erodes the coset *position*). Recommended architecture: **γ=0 within certified segments (governor as outer safety layer), a mixture kernel `½·MALA(σ\*) + ½·sign-symmetrized squeeze-MH`, and proposals projected off the coset tangent** — this is the only combination that is simultaneously certified (detailed balance), well-mixing (gradient-informed MALA with our FDT-correct σ\*, Check 4 L1=0.0065 vs unadjusted 0.0995), and latch-safe.

---

## PART 1 — The kernel, stated precisely (Item 1)

### Prop-MH1 (squeeze-MH is π-reversible). [proven; verified (Check 1)]
Target `π(z) ∝ e^{−H(z)/T}`, `z=(q,p)`, `H = T(p) + V_θ(q)`. Define one sweep:
1. **Momentum Gibbs-refresh** `p ~ 𝒩(0, M_eff·T)` — leaves the momentum marginal (hence π) exactly invariant; always accepted.
2. **Sign-symmetrized squeeze proposal**: draw `s=±1` uniformly (or `ζ` from any density symmetric under `ζ↦−ζ`), propose `z' = S^{(M)}_{sζ}(z)`; accept with
   `a(z,z') = min(1, exp(−[H(z')−H(z)]/T))`.

Because `S^{(M)}_ζ = 𝒩⁻¹S_ζ𝒩` is symplectic with **`det J = 1` exactly** (F5 Prop-12) and `S^{(M)}_{−ζ} = (S^{(M)}_ζ)^{-1}`, the proposal is a volume-preserving reversible map: the change-of-variables Jacobian in the Metropolis–Green ratio is 1, so **no Hastings correction is needed** and the ratio collapses to the energy ratio `e^{−ΔH/T}`. Each sub-step satisfies detailed balance w.r.t. π; their composition is π-invariant. ∎ *(Tierney 1998 deterministic-proposal MH; Green 1995 for the general |det J| factor, here ≡1.)*

*Verified (Check 1):* sign-symmetric kernel reproduces the Gibbs q-marginal to `L1=0.0095` (var 0.9907 vs Gibbs 0.9895; mean −0.3075 vs −0.3072), acceptance 0.66.

### Two conditions the bare pitch omits.
- **Reversibility needs `S_{−ζ}` offered.** `S_ζ` is *not* an involution (`S_ζ∘S_ζ=S_{2ζ}`), so a fixed-ζ, fixed-sign squeeze with `min(1,e^{−ΔH/T})` is **not** valid on its own — the reverse move requires `S_{−ζ}`. *Nuance found numerically:* with full Gibbs momentum refresh, even a fixed-sign squeeze reproduces the q-marginal (Check 1, `L1=0.0122`), because `S_{+ζ}(q,−p)` and `S_{−ζ}(q,+p)` share the same q-component and the refresh makes `±p` equiprobable — the refresh's momentum-sign symmetry *substitutes* for explicit sign-symmetrization. **Sign-symmetrization is load-bearing only in partial/no-refresh regimes.** [proven; verified]
- **Ergodicity needs momentum refresh.** `{S^{(M)}_ζ : ζ∈ℝ}` is a one-parameter subgroup; the reachable set of any `z₀` under squeezes alone is the single hyperbola `{S_ζ z₀}` — the chain is **reducible** and does **not** sample π. *Verified (Check 1):* squeeze-only-no-refresh gives `L1=1.81` (stuck near init, var 0.058). Refresh-only-no-squeeze cannot move q (`var=0`). **Only refresh + squeeze together sample the EBM.** This is exactly HMC's structure (deterministic symplectic map made ergodic by momentum resampling). [proven; verified]

### Relation to HMC (Duane et al. 1987; Neal 2011). [proven]
HMC's proposal is `leapfrog(L)∘momentum-flip` — a volume-preserving involution accepted at `min(1,e^{−ΔH})`, momentum resampled each iteration. **Squeeze-MH keeps the momentum-resampling backbone and replaces the L-step leapfrog trajectory with one linear symplectic shear `S^{(M)}_ζ`.** Trade:

| axis | HMC leapfrog proposal | squeeze `S^{(M)}_ζ` proposal |
|---|---|---|
| gradient use | yes (follows ∇H) | **none** (linear, landscape-blind) |
| ΔH per proposal | `O(ε²)` (near-conserving) ⇒ high accept | up to `(e^{2\|ζ\|}−1)H` (Prop-12) ⇒ accept falls with ζ |
| mixing class | gradient-informed; cost `~N^{1/4}` | preconditioned **random-walk**; cost `~N` |
| tuning | step ε (stability `h<2`), path length L | rapidity ζ only; no stability limit |
| test-time cost | `L` gradients of learned `V_θ` | **zero gradients** of `V_θ` |
| built-in structure | identity mass metric (unless tuned) | **mass metric M baked in** (`S^{(M)}=𝒩⁻¹S𝒩`): light-mode-preferential global moves = the reach asset |

**Verdict:** squeeze-MH is a **certificate-first, gradient-free, mass-aware member of the preconditioned-symmetric-proposal Metropolis family** — *not* a competitor to HMC's mixing efficiency. It costs mixing (high rejection / random-walk autocorrelation) and gains implementation and certificate simplicity plus the learned mass preconditioner. On a rough trained EBM its acceptance degrades with ζ, so as a *sampler* it is weak; its value is as a **global reach move** (paid-access §3) inside a mixture, not as the primary sampler.

### The composite (squeeze-then-relax) is NOT a valid sampler kernel. [proven; verified (Check 2)]
The retry cascade interleaves **governed relaxation** `Φ_{ε,γ(H)}` between proposals. `γ(H)=s·tanh(max(0,H−E*))>0` is **state-dependent dissipation**: `det J = (1−γ)^d < 1` (F5 Prop-3), it strictly drains energy, and it does **not** preserve `π=e^{−H/T}`. So `K = R_relax ∘ K_MH` has no Gibbs stationary law. **What it converges to:** an alternation of a π-reversible kernel with a deterministic contraction toward the low-energy set = a **Metropolis-within-annealing / stochastic-approximation** recursion. Its invariant concentrates on a *colder* measure `π_{T_eff}`, `T_eff<T`, drifting toward `argmin H` (MAP-seeking) as relaxation strength grows. *Verified (Check 2):* harmonic EBM (Gibbs var(q)=1.0 exactly); γ=0 composite gives var=1.010 ✓ (kernel valid when relaxation off), γ=0.05→0.795, γ=0.20→0.609, i.e. `T_eff = k·var(q)`: **1.0 → 0.61**. The certificate that survives the composite is therefore **not stationarity w.r.t. π** but an **annealing/optimization statement**: with γ annealed to 0 across the cascade you recover π-sampling in the limit; with fixed γ>0 you get a MAP/low-energy-mode seeker. This is the honest decomposition the task asked for.

---

## PART 2 — The governor in the loop (Item 2, the crux)

The governor is state-dependent dissipation ⇒ it destroys the Gibbs invariant (Part 1, Check 2). Analysing the three options:

- **(a) Freeze γ=0 during MH segments; governor as outer safety layer.** Within a segment the kernel is exactly π_T-reversible (Prop-MH1). The governor acts *between* segments as a declared projection/cooling step. Certificate: **per-segment detailed balance**; the whole scheme is honestly a Metropolis-within-annealing with a *stationarity certificate on each frozen-γ segment*. **Cleanest, strongest certificate. RECOMMENDED backbone.**
- **(b) Governor-on Metropolis-within-annealing.** Only an asymptotic-annealing guarantee survives, and `γ(H)` is not a standard (monotone, slow) cooling schedule — convergence to π (or to global min) is not certifiable in general. Weaker. Not recommended as primary.
- **(c) FDT-corrected Langevin (MALA with σ\*) as the proposal.** Our unique asset. The discrete Langevin O-step with the **FDT-correct noise** `σ_i^\* = √(M_{eff,i}·T·γ(2−γ))` (F5 Prop-9), **Metropolis-adjusted** (MALA), samples π *exactly*. It is gradient-informed (fixes squeeze's poor mixing) and σ\* is the calibrated scale we alone possess (nominal T is ~11× off, κ≈0.09; generative-studies). *Verified (Check 4):* MALA(σ\*) reproduces Gibbs to `L1=0.0065` (var 1.0414 vs 1.0418, accept 0.955); **unadjusted** Langevin (ULA) is biased, `L1=0.0995` (15× worse), var low by shadow-`O(ε²)` bias. The Metropolis adjustment is what upgrades our FDT sampler from "shadow-Gibbs, O(ε²)-biased" to "exact-Gibbs, certified."

**Recommended architecture (with certificate):**
> **`K = ½·MALA(σ\*) + ½·sign-symmetrized squeeze-MH`, run at γ=0 inside a certified segment; governor applied only between segments (BIBO safety); all proposals projected off the coset tangent (Part 3).**

A mixture of two π-reversible kernels is π-reversible ⇒ the segment has an exact detailed-balance certificate. MALA(σ\*) supplies mixing + the FDT-calibrated scale; squeeze supplies mass-aware global/reach moves that local diffusion misses; the governor supplies bounded-input-bounded-output safety *outside* the certified window (so it never corrupts the invariant). **When is FDT-Langevin strictly better than squeeze as the proposal?** Whenever (i) `∇V_θ` is available at test time (it always is — differentiable), (ii) the goal is to *sample* π with good mixing, and (iii) a global light-mode reach jump is not required. Squeeze is strictly preferred only for gradient-free, mass-metric *reach* moves (paid-access §3). Hence MALA(σ\*) = default certified sampler; squeeze = reach add-on.

---

## PART 3 — Latch / coset behaviour: does certified retry destroy memory? (Item 3)

**Yes, unless proposals are coset-projected. [proven; verified (Check 3)]**

The stored register lives in the Goldstone **coset position** `φ` (F5 §3.3 latch theorem: a write = momentum impulse transporting `q` along the flat direction to `q_∞`). The Gibbs measure is **exactly flat along the coset** (`μ=0`, `V` constant along it), so `π|_coset` is uniform.

### Prop-MH2 (latch diffusion under MH). [proven; verified]
Any proposal with a coset-tangent component `δφ` has `ΔH = 0` along that component, hence is accepted with probability **1**. The latch coordinate therefore performs an **unbiased random walk**: with symmetric coset-step variance `s²`,
```
D_φ = ½ s²  (per accepted step);  Var(φ_N) = 2 D_φ N = s² N;  N_erode ≈ (Δ_read / s)²,
```
where `Δ_read` is the read-out angular margin (latch corrupted when `√Var(φ_N) ≳ Δ_read`).
*Verified (Check 3):* direct coset walk, `s=0.05` → **acceptance = 1.000**, measured `D = 1.29e-3` vs predicted `½s² = 1.25e-3` (3%); an off-coset (radial) proposal leaves `Var(φ)=0` exactly — **the latch is quenched**. Concretely, for `Δ_read=0.5 rad, s=0.05`: `N_erode ≈ 100` accepted coset-touching retries erase the stored value.

### The critical caution for V1: charge-preservation ≠ position-preservation.
w7 measured that a **channel-isotropic squeeze preserves the Goldstone charge `Q=pᵀXq` (ΔQ=1.2e-7)**. But `Q` is the *conserved conjugate momentum*, **not** the stored coset position `φ`. *Verified (Check 3):* the isotropic squeeze **+ momentum refresh** diffuses the coset *position* strongly (`D=1.7e-2`, `Var(θ)` grows without bound), because the refreshed isotropic momentum enters the shear `q'=q coshζ + p sinhζ` and kicks `q` around the flat orbit. **So the w7 latch-preservation certificate does not protect stored memory under repeated MH — it protects the charge, while the register (position) still random-walks.** This must be stated before any MH paragraph ships.

### Answer & design rule.
- **Is diffusion acceptable if the latch is quenched (only touched by explicit writes)?** Yes — and that is exactly the fix. Restrict the certified MH/MALA kernel to the **coset-orthogonal complement** (project proposal noise off the coset tangent; equivalently sample only the massive/orthogonal sector). Then `φ` is frozen (Check 3 off-coset: `D=0`) and only explicit writes move it, consistent with the F5 latch theorem. [design rule; proven mechanism]
- **Without the projection**, certified retry has a hard **erosion budget `N_erode=(Δ_read/s)²`** — a fixed number of retries before stored coset content is lost. Any V1 MH paragraph must either (i) adopt the coset-orthogonal projection, or (ii) quote `N_erode` as an explicit lifetime and cap the retry count below it.

---

## PART 4 — Temperature semantics (Item 4)

### The acceptance rule is intra-model; cross-model incomparability does NOT break validity. [proven]
`ΔH = H(z')−H(z)` is always computed within **one chain on one model**, so the raw-R lesson (pooled cross-model energy AUROC 0.43<0.5, anti-ranked; v1-pivot) does **not** invalidate `min(1,e^{−ΔH/T})`. Detailed balance holds for **any fixed T>0** on any single trained H. This part of the certificate is genuinely global-T-agnostic and parameter-light.

### But the *operating point* is model-specific and FDT-miscalibrated. [proven + evidenced]
T sets the acceptance scale: `T→0` freezes (reject all), `T→∞` random-walks (accept all, no selection). The useful T scales with **that model's barrier/basin energy**, which is model-specific. Worse, by F5 Prop-9 the code temperature is not in energy units: nominal `T` is `~11×` off effective (κ≈0.09 at γ=0.3, dt=0.05), `T_eff ∝ dt`, and M is absorbed (generative-studies verified on real Exp-C checkpoints, slope(logVar(p),logM)=0 legacy). **A hardcoded global T in the acceptance rule is therefore meaningless across dt/M/architecture.** Principled T: **per-model, calibrated from write-time self-test probes** — repurpose the v1-pivot calibration apparatus to estimate the typical `ΔH` scale between correct-retrieval and impostor/perturbed states, then set `T` to match a target acceptance rate (~0.5–0.7) or that energy scale.

### Honest conclusion (CM-3 discipline). [design hypothesis; states the deflation]
**The stationarity certificate is free and holds for any T; the *useful* per-model T re-imports the learned calibration head.** Setting T from write-time probes is exactly the v1-pivot per-model Platt/affine head (raw 0.43 → calibrated 0.87 requires it) repurposed onto the acceptance temperature. So the "parameter-light MH kernel" pitch is **partially deflated**: MH's *certificate* is parameter-light, but its *operating point* carries the same per-instance learned parameter as the τ-gate. **What MH adds over learned-τ is a certificate layer** (a detailed-balance / stationarity statement about the accepted ensemble), **not** parameter parsimony and **not** — on current evidence — performance (energy≈learned-signal three times over: v1-l0-gate, minus-the-physics, v1-router). Say this plainly in V1.

---

## Numerical checks — summary (Appendix N)

| id | claim | observed | predicted | verdict |
|---|---|---|---|---|
| C1 | sign-symm. squeeze-MH samples Gibbs (Prop-MH1) | q-marginal L1=**0.0095**, var 0.9907, accept 0.66 | Gibbs var 0.9895 | ✓ |
| C1 | fixed-sign OK *with* refresh (nuance) | L1=**0.0122** | ≈ symmetric | ✓ |
| C1 | squeeze-only non-ergodic (subgroup orbit) | L1=**1.81**, stuck | reducible | ✓ |
| C1 | refresh alone cannot move q | var=**0.0** | — | ✓ |
| C2 | governor destroys Gibbs invariant → colder | T_eff **1.0→0.795→0.609** (γ=0,.05,.2) | γ=0 ⇒ 1.0 | ✓ |
| C3 | coset walk: accept=1, D=½s² (Prop-MH2) | accept **1.000**, D=**1.29e-3** | ½·0.05²=1.25e-3 | ✓ |
| C3 | off-coset proposal quenches latch | D=**0**, Var(φ)=**0** | frozen | ✓ |
| C3 | isotropic squeeze erodes coset *position* | D=**1.7e-2** (Q preserved ≠ φ preserved) | diffuses | ✓ |
| C4 | MALA(σ\*) samples Gibbs; ULA biased | L1 **0.0065** vs **0.0995**; accept 0.955 | exact vs O(ε²) | ✓ |

Repro: `cd .claude/scratch/thread9-mh-kernel && /Users/user/Desktop/CHLU/.venv/bin/python checks.py` (numpy 2.4.1, ~few min pure-python).

---

## Verdicts (tagged)

| claim | verdict |
|---|---|
| Squeeze-MH (refresh + sign-symm.) is π-reversible on the memory EBM, det J=1 ⇒ no Jacobian term (Prop-MH1) | **[proven; verified C1]** |
| Squeeze family alone is non-ergodic (1-param subgroup); needs momentum refresh | **[proven; verified C1]** |
| Squeeze-MH ∈ preconditioned-random-walk Metropolis, not an HMC-mixing competitor (gradient-free, mass-aware, high-rejection) | **[proven]** |
| Squeeze-then-relax composite breaks the Gibbs invariant → Metropolis-within-annealing / MAP-seeking (T_eff<T) | **[proven; verified C2]** |
| Recommended kernel: ½MALA(σ\*)+½squeeze, γ=0 in-segment, governor outer, coset-projected | **[design hypothesis; components verified C2,C4]** |
| Certified retry erodes the latch: coset moves accepted w.p.1, D=½s², N_erode=(Δ_read/s)² | **[proven; verified C3]** |
| w7 charge-preservation (ΔQ≈0) does NOT imply coset-position preservation; isotropic squeeze still erodes memory | **[proven; verified C3]** |
| Coset-orthogonal projection quenches the latch (only explicit writes move it) — the fix | **[proven mechanism; verified C3]** |
| Acceptance is intra-model ⇒ cross-model incomparability doesn't break validity; but useful T is per-model | **[proven / evidenced]** |
| Per-model calibrated T re-imports the learned gate → "parameter-light" partially deflated; MH's asset is the *certificate*, not parsimony/performance | **[design hypothesis; CM-3-honest]** |

---

## V1-grade position paragraph (physics-heavy, ML4PS register — candidate for the horizon/mechanism section)

> **Test-time retries as a certified Markov kernel.** Because the CLU memory is a *conservative* Hamiltonian system, its Lorentz squeeze `S^{(M)}_ζ` is an exactly volume-preserving symplectic map (`det J = 1`), which is the defining property Hamiltonian Monte Carlo exploits to avoid a Jacobian correction. A retry that proposes a sign-symmetrized squeeze and accepts it with the Metropolis rule `min(1, e^{−ΔH/T})` is therefore a **detailed-balance kernel for the Gibbs measure `e^{−H/T}` of the trained energy** — the retry cascade becomes *test-time compute as MCMC with a stationarity certificate*, rather than a heuristic. Two physics facts sharpen this into an honest claim. First, the squeeze family is a one-parameter subgroup, so it samples only a hyperbola of phase space; ergodicity requires HMC-style momentum refreshment, and the squeeze is best understood not as a sampler in its own right but as a **mass-metric-preconditioned global move** layered on a Metropolis-adjusted Langevin (MALA) step whose noise scale we fix by the discrete fluctuation–dissipation relation `σ_i^\*=√(M_{eff,i}Tγ(2−γ))` — a calibration our conservative-integrator analysis uniquely supplies. Second, and decisively for a *memory*, the Gibbs measure is exactly flat along the system's Goldstone coset, so unconstrained certified retries **random-walk the stored register at diffusion rate `D=½s²`, erasing it after `N_erode≈(Δ_read/s)²` accepted moves**; the fix is to run the kernel in the coset-orthogonal complement, leaving the latch quenched and touched only by explicit writes. The resulting primitive — *governor-composed, bounded-injection, coset-projected, FDT-calibrated Metropolis moves on a trained conservative memory* — is what distinguishes it from generic HMC/annealing: it is an MCMC certificate attached to a learned associative memory, with an explicit accounting of when certified compute preserves versus destroys the thing it is retrieving. We are candid that the certificate is the contribution: the acceptance temperature that makes it *useful* is a per-model calibrated quantity, i.e. the same learned gate our calibration head already fits, so the value of the MH framing is the stationarity guarantee, not parameter parsimony.

*(Foldable at revision; the last sentence is the CM-3-mandatory honesty and should not be cut.)*

---

## w9 experiment spec (vs learned-τ AND router baselines — CM-3/CM-7 discipline)

**Goal:** test whether the certified kernel *(½MALA(σ\*)+½squeeze, γ=0, coset-projected)* buys anything measurable over the learned-τ gate (v1-pivot) and the no-physics router (v1-router-baseline), and quantify latch erosion.

**Arms (all on the same trained CLU-EBM MQAR checkpoints, kv∈{16,24,32}, ≥5 seeds):**
1. **learned-τ gate** (v1-pivot baseline; the boring baseline any acceptance rule must beat).
2. **no-physics router** (v1-router baseline; CM-7).
3. **squeeze-MH only** (sign-symm., refresh, per-model calibrated T from write-time probes).
4. **MALA(σ\*) only** (Metropolis-adjusted, FDT noise, per-model T).
5. **mixture ½MALA+½squeeze** (the recommended kernel), coset-projected.
6. **mixture, NOT coset-projected** (to measure latch erosion in situ).

**Certificate checks (per arm):** empirical detailed balance (histogram of accepted ensemble vs `e^{−H/T}` on a held-out probe manifold; L1/χ²); acceptance rate vs calibrated T; `det J` of proposals (expect 1±1e-12 squeeze).

**Metrics:** retrieval accuracy & risk–coverage (AURC, cov@risk) vs compute (steps); **latch lifetime** — measured `D_φ` and `N_erode` vs (ζ, T, coset-projection on/off), against the `½s²` / `(Δ_read/s)²` predictions; effective temperature of the composite when γ is turned on (T_eff vs γ, replicating Check 2 on a trained model).

**Predicted signatures (falsifiable):** (i) arm 5 shows exact detailed balance (L1→0) where arms 1–2 have *no* stationarity certificate; (ii) arm 6 shows latch AUROC/accuracy decaying as `~e^{−N/N_erode}` while arm 5 is flat — the "does certified retry destroy memory" money plot; (iii) MALA(σ\*) mixes faster than squeeze-only (lower autocorrelation / higher accept at matched T); (iv) **honest null risk (CM-3):** arm 5 accuracy ≈ arm 1 (learned-τ) — if so, the contribution is *only* the certificate + latch accounting, which the paragraph already concedes. **Kill criterion:** if arm 6 does *not* erode (latch survives unconstrained retries), Prop-MH2 is refuted and the coset-projection story is unnecessary — report as C-9 negative.

**Provenance to record:** commit, seeds, `langevin_noise=fdt`, per-model T source, γ schedule, ζ grid, coset-projection basis, Δ_read definition.

---

## Prior-art positioning (cite, don't claim)

- **HMC** (Duane, Kennedy, Pendleton, Roweth 1987; Neal 2011 *MCMC using Hamiltonian dynamics*): symplectic, det=1 proposals + Metropolis, momentum resampling. **Our squeeze = a linear, gradient-free, mass-preconditioned replacement for the leapfrog trajectory** — a special case in structure, weaker in mixing; differentiator is the *learned* H and the *conservative-memory* application, not the MCMC machinery.
- **MALA / Metropolis-adjusted Langevin** (Roberts–Tweedie 1996): gradient-drift proposal + MH. **Our contribution is the FDT-correct `σ\*` (Prop-9) that makes the CLU Langevin exact rather than shadow-biased** — a calibration, not a new algorithm.
- **Simulated annealing / basin-hopping** (Kirkpatrick 1983; Wales–Doye 1997): the squeeze-then-relax composite *is* an annealing scheme with a state-dependent (governor) schedule. Cite; our differentiator is the *symplectic, bounded-injection, certificate-bearing* move and the explicit T_eff(γ) accounting.
- **Parallel tempering / replica exchange, delayed-rejection** (Swendsen–Wang 1986; Tierney–Mira 1999): adjacent retry/temperature machinery; our T semantics (per-model FDT-calibrated) and latch accounting are the non-overlapping content.
- **Goldstone-mode / coset diffusion:** the latch-erosion result is a Mermin–Wagner-flavoured statement (no restoring force along a broken continuous symmetry ⇒ unbounded diffusion of the order-parameter phase). Cite the physics; the ML novelty is *quantifying stored-memory erosion under a certified sampler* and the coset-projection fix.

**Contamination flags for web-scout:** "MCMC as test-time compute" is increasingly crowded (diffusion samplers, EBM-Langevin inference); our defensible novelty is narrow and must be stated as such — *governor-composed, coset-projected, FDT-calibrated, bounded-injection acceptance on a trained conservative associative memory, with an explicit certified-retry-erosion budget*. Not "MCMC is good."

---

## Open questions / risks

1. **Relativistic Gibbs marginal is non-Gaussian** (F5 Open-3): momentum refresh `p~𝒩(0,M_effT)` is exact only for `T(p)=½pᵀM⁻¹p`. In relativistic mode the correct refresh draws from `∝e^{−T(p)/T}` (non-Gaussian); Gaussian refresh introduces an `O(?)` bias. Un-quantified — flag for w9 (use Newtonian-learned mode for the clean certificate, or a rejection-sampled relativistic refresh).
2. **σ\* under a learned mass hierarchy:** per-mode σ\* requires knowing `M_eff,i`; on near-uniform learned M (log-std≈0.08) the preconditioner is inert (paid-access §3.3 reason 2) — the squeeze-reach and per-mode-T advantages are untestable without banded M. w9 should band M or the result repeats the l0-gate ambiguity.
3. **Coset-projection basis at inference:** projecting off the coset tangent requires knowing the broken generator X (or estimating the flat direction) at test time — un-built; the engineering crux mirrors paid-access open-risk 1 (learned placement).
4. **Composite convergence rate:** I characterised *what* the governor-on composite converges to (colder MAP-seeking) but not the *rate* / basin-selection bias — a first-passage analysis is a scoped open problem.
5. **N_erode vs real readout margin:** `Δ_read` is the decode tolerance; its value on trained MQAR checkpoints is unmeasured — needed to turn `N_erode=(Δ_read/s)²` into a concrete retry cap.

---

## Proposed handover updates (for the Hub)

**§1 (physics):** add the **certified-kernel result** — squeeze+MH is a π-reversible HMC-family kernel (`det J=1` ⇒ no Jacobian; Prop-MH1), non-ergodic without momentum refresh; the **squeeze-then-relax cascade is Metropolis-within-annealing** (governor destroys the Gibbs invariant → colder MAP-seeking, T_eff<T, verified); recommended kernel `½MALA(σ\*)+½squeeze, γ=0 in-segment, coset-projected`.

**§7 (discrepancies / cautions):** record that **w7's charge-preservation (ΔQ=1.2e-7) does NOT imply latch-content preservation under MH** — the isotropic squeeze erodes the coset *position* (stored register) even while conserving the charge Q; certified retry has an erosion budget `N_erode≈(Δ_read/s)²` unless proposals are coset-projected. Prevents a reviewer contradiction ("latch is preserved" vs "retries erase memory").

**§8 (open questions):** MH-kernel theory is complete with a discriminating w9 spec (vs learned-τ + router). Remaining open: relativistic non-Gaussian momentum refresh (Open-3 tie-in), coset-projection basis at inference, composite convergence rate, and Δ_read measurement to make `N_erode` concrete. **CM-3 honesty to inherit:** MH's asset is the *stationarity certificate*, not parameter parsimony (per-model T re-imports the learned gate) and not — on current evidence — performance.

**New F5 candidates:** Prop-MH1 (squeeze-MH detailed balance + non-ergodicity of the subgroup), Prop-MH2 (latch diffusion `D=½s²`, `N_erode`), the composite Metropolis-within-annealing decomposition, and the MALA(σ\*) exactness (Prop-9 + Metropolis adjustment) — all with the Appendix-N numerics.
