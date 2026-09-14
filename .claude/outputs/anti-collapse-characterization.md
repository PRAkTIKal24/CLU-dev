# anti-collapse-characterization — physics-theorist report

Task + acceptance criterion: collapse taxonomy (lever · order parameter · driver · fatal-for-which-task) + VICReg-for-CLU formalization (named structural-diversity regularizers, each derived) + lever→task map + ≥1 numerical collapse-and-cure demo.
Status: **done.** No downstream reconciliation list (nothing here retracts a shipped number); this report *proposes* new claims, it does not amend existing ones. No tracked code touched (theory task).

What I did:
- Unified the four measured collapse phenomena (N7/CM-5 mass narrowing, CM-6/CM-16 vacuum erosion, N46 coset non-formation, CM-16b band structure) plus two latent ones (coercivity loss, forgetting-structure failure) under **one demarcation principle** with three distinct drivers; derived it; classified which driver produces which collapse.
- Formalized the VICReg analogy precisely (what maps, what does *not*, and what is genuinely new) and specified a **named family of six structural regularizers**, each with a derivation of what it penalizes and a task-compatibility (null-direction) analysis.
- Built the lever→CAFE-task fatality map with confidence labels.
- Ran a **pre-registered** collapse-and-cure demo (5 predictions committed before running; 1 protocol-compliant amendment, recorded): all 5 landed, two of them to 4 decimal places against closed-form derivations.

How I verified: `.claude/scratch/anti-collapse-characterization/demo.py` (pure numpy, no repo code, no JAX); results `.claude/outputs/anti-collapse-characterization/demo_results.json`; prereg + amendment `.claude/outputs/anti-collapse-characterization/PREREG.md`. Observed numbers quoted throughout §5.

**Flag-provenance (demo).** Self-contained toy, zero repo/config dependence: numpy 2.4.1, Python 3.11.13, repo at `a5978f6` (untouched, clean tree). Constants: dt=0.05, 128 steps, multi-scale windows {16,32,64,128} (PREREG Amendment 1), q0=1, p0=0, k*=(1,1), M*=(4,0.25), init m=0.7·e^{N(0,1e-3)}, η_k=0.05, η_m=0.005, GD 4000 steps, central FD δ=1e-5, λ_s=1.0, s\*=½·log16. Seeds: arms A/B/D `default_rng(0..4)`, arm C `default_rng(100..109)` (jitter only). A voided first run (non-converged, per prereg deviation protocol) is archived as `demo_results_VOID_prereg_amendment1.json`.

---

## 1. The unifying principle: collapse = drift along wake-invisible directions (with three distinct drivers)

**Setup.** Training objective `L_task(θ)` (wake MSE, or wake+CD). A **structural lever** is summarized by an **order parameter** `O(θ)` — a scalar/low-dim functional of the *parameters or the induced dynamics operator* (not of activations; see §3 for why this is the substantive departure from VICReg). Examples: `Std(log M)` (mass spread), ring depth / `μ²_coset` (vacuum flatness), `noise_gap` (coercivity contrast), damping-band occupancy.

**Definition (wake-visibility).** `O` is *wake-invisible at θ* iff there exists a parameter direction `v` with `∇L_task·v = 0` (task-null) and `∇O·v ≠ 0` — i.e. `O` can be moved at zero task cost. Operationally testable (this is my P3 test, and it is the exact generalization of the erosion demarcation "a flat direction unconstrained by the wake objective," CM-6/anchor-robustness §2).

**Claim T1 (demarcation, generalizing CM-6).** *A structural lever collapses under training iff its order parameter is wake-invisible, and the collapsed value is then selected not by the task but by one of three drivers:*

- **D1 — laziness / lr-partition (implicit bias).** When a fast lever (rich-gradient V_θ) and a slow lever (log_mass) *jointly* parameterize the wake-visible quantity, the fast one absorbs the signal; afterwards the slow lever sits in an (emergent) null direction and freezes near init. Exact law in the gauge case (derived §5.1, verified to 4 dp): the slow lever receives only the fraction `(η_m/η_k)/(1+η_m/η_k)` of the required log-displacement. Collapse *to init* — this is N7's "masses stay ≈0.7" and mass-spectrum-peek's σ_struct = 2–23%. The CM-5 cure (mass-lr 10×) is literally re-tuning this partition; its failure at 100× (N8 inversion) is the partition overshooting into the task-transient's wrong phase.
- **D2 — adversarial bias (CD sleep).** The sleep gradient is *nonzero and signed* along wake-null directions (raise V wherever negatives sit), so it actively drives `O` to an inverted value at a rate set by CD-update frequency racing the wake clamp (the measured horizon law, sleep-erosion §3.1: inversion at ep 116/442/959 for f=1/5/20, independent of sleep_steps). Collapse *to inversion* — CM-6/CM-16.
- **D3 — non-formation (measure-zero structure).** Structure requiring an exact parameter coincidence (a continuous symmetry of an MLP potential = measure-zero in function space) is never *reached* by generic training, and any accidental near-flatness is lifted by SGD noise + anharmonicity. Collapse *by default* — N46: the emergent "flat" direction is a mid-spectrum massive mode, capacity 1–1.6 bits.

**Claim T2 (anti-collapse = gauge fixing; zero expressivity price).** Add `λ·G(O(θ))` with minimum at a healthy `O*`. *If the level set `argmin L_task ∩ {O = O*}` is nonempty (true whenever `O` is wake-invisible along the optimum manifold), then `min(L_task + λG) = min L_task` — the regularizer selects among task-optimal solutions and costs zero fit at optimum.* Proof: evaluate at any point of the level set; both terms are simultaneously minimal. ∎
Two honest corollaries: (i) the price at *finite training time* is an optimization-stiffness price, not an expressivity price — this is exactly the measured anchor envelope (λ=100 holds the vacuum 5/5 seeds at 35× the wake-MSE *floor*, which is a rate effect, not a representability wall; anchor-robustness §1); (ii) if `O` is wake-*visible* (null lifted), collapse does not happen anyway (my arm D) and the regularizer should carry a deadband/hinge so it cannot fight a task that already determines `O`.

**Claim T3 (diversity ≠ assignment — the family's intrinsic limitation).** Any regularizer built from a *permutation-symmetric* statistic of the spectrum (variance, entropy, Wasserstein-to-target) restores **diversity** but cannot choose the **assignment** (which channel gets which timescale). Near the collapsed point the Var-form gradient is proportional to the existing asymmetry (`∂G/∂Δ ∝ (Var−s*²)·Δ`, vanishing at Δ=0), so it *amplifies whatever seed exists*: the task's early transient if the regularizer is on from epoch 0 (correct assignment), or pure init noise if switched on after the fast lever has absorbed the signal (coin-flip assignment). Verified exactly: §5, P2 (5/5 correct from-start) vs P4 (6/10 warm-start, and the final ordering equals the sign of the 1e-3 init jitter **10/10**). Design rules that follow: **(a) structural regularizers on from epoch 0, not as a patch; (b) prefer the Var-form (gradient ∝ Δ, an amplifier — the collapsed point becomes an unstable fixed point) over a Std-hinge (O(1) subgradient at Δ=0 that kicks in an arbitrary direction).** This retro-explains CM-5's schedule sensitivity: ordering is inducible (mild 10× amplification, task transient intact) while naive 100× inverts (amplifier overwhelms the seed).

---

## 2. Collapse taxonomy (lever · order parameter · collapsed value · driver · evidence · fatal-for)

| # | lever | order parameter `O` | collapsed value | driver | evidence | fatal for (see §4) |
|---|---|---|---|---|---|---|
| C1 | **mass-spectrum diversity** (inertial `M` — CM nomenclature: inertial M, never bare "mass") | `Std(log M)`; alignment `Spearman(M, M*)` | `Std → lr-partition floor` (≈init); σ_struct 2–23% | **D1** (V_θ↔M timescale gauge + gradient-stiffness asymmetry) | N7, mass-spectrum-peek, seed-sweeps item 1 (uniform never recovers 16×), CM-5; toy: §5 P1 exact | multi-timescale memory (V3 banding: matched banding halves eval-MSE); Event prediction |
| C2 | **designed symmetry / vacuum** | ring depth; `μ²_coset` (flatness); `r*` | ring inverts (+0.079→−0.126), flat mode stiffens `μ²: 0→+0.04–0.21` | **D2** (CD deposits energy along wake-invisible flat direction; overlap→raise→expel measured) | CM-6/CM-16, sleep-erosion §3.2, anchor-robustness §2 | coset-register memory (V2/V5); energy-scored anomaly (see C5 coupling: post-erosion `noise_gap = −0.028` — the model *prefers noise*) |
| C3 | **coset register existence** (emergent) | `1−\|λ_coset\|`; flatness ratio `μ²_min/μ²_max`; register capacity (bits) | no register ever forms: coset is mid-spectrum massive, 1–1.6 bits washboard | **D3** (exact continuous symmetry is measure-zero for MLP V_θ) | N46, v5-gate, CM-16a designed-only split | continuous-register uses (path integration, drift tracking); benign for first-pass CAFE |
| C4 | **memory-mode band coverage** | occupancy of damping bands `x_i = 2εμ_i/γ` ∈ {latch ≈0, budget (0,1), floor ≥1} | all modes in one band (narrow μ² × single global γ) | composite: **D1** (via C1) + **D3** (no exact zeros) + single scalar γ | CM-16b V-curve unification (t-lever §4.3): floor is *mass-independent* past crossover — retention diversity exists **only** in the budget band | mixed-horizon tasks; Event prediction (needs slow persistent + near-critical responsive modes) |
| C5 | **potential coercivity / valley contrast** | `noise_gap = H(noise)−H(data)`; far-field margin; valley depth | V flattens/inverts off-manifold; `noise_gap < 0` | **D1-like** (wake only constrains V *on* the data manifold; off-manifold V free) + architecture (§7.7: Deep/Conv omit the α‖q‖² confinement) + **D2** post-erosion | sleep-erosion Q3 baseline; §7.7; Exp-C deterministic mode-collapse; broken-volume divergence (orthogonal failure, anchor-robustness §3) | **anomaly detection** (the score *is* the valley structure); Event (basin-exit needs basins); generation |
| C6 | **forgetting structure** (γ_φ / T_φ localization) | field contrast `γ_φ(noise)/γ_φ(data)`; locus hit-rate | field never finds the locus (seed-dep. `γ_on_noise ≈ 1e-4`); governor+field Pareto-dominated | sparse/exploration-limited gradient to localized field params; and γ_φ is *provably the wrong operator* on flat content (t-lever §7: a friction hole is a memory vault) | N12, N13, seed-sweeps §3; t-lever | streaming/continual deployment; benign for offline CAFE scoring |

The taxonomy's punchline: **C1–C6 are one phenomenon (T1) with three drivers**, and the driver dictates the cure class — D1 needs *preconditioning or a diversity term*, D2 needs a *value pin* (anchor family), D3 needs *design-in or an inducing term* (soft cures provably cannot reach machine-flat; §3, R-4).

---

## 3. VICReg-for-CLU, made precise

VICReg: `L = λ·invariance + μ·Σ_j hinge(1 − Std_batch(z_j)) + ν·Σ_{i≠j} Cov_ij²`. Its collapse mode (constant/low-rank embedding) is precisely a **null direction of the invariance loss**; the variance term is a gauge-fixing potential on it; the covariance term removes redundancy among the surviving directions. So the analogy is *structural*, not loose — same theorem T2 applies. The substantive difference, which is the novelty claim: **VICReg's order parameters are batch statistics of activations; CLU's are spectra of the dynamics operator and geometry of the energy landscape — parameter/operator-space objects that exist without any batch.** The proposed family ("**SDR — structural-diversity regularizers**"):

| id | name | form | VICReg analog | prevents | derivation summary |
|---|---|---|---|---|---|
| R-1 | **mass-spread** | `λ_s(Var(log M) − s*²)²` (Var-form; optional deadband) | variance term | C1 | acts along the exact `(δlog k = δlog M)` gauge direction (task-null when only positions are wake-observed ⇒ zero fit price, T2; proven §5 P3: ΔL_task = 0.0 at δ=0.2). Var-form for stability (T3). Ordering-blind (T3) ⇒ on from epoch 0. |
| R-2 | **energy anchor** (exists: `anchor_data_energy_lambda`) | `λ(mean V(data) − V₀)²` | *(no analog — needs a landscape)* | C2 | the 0th-moment value pin along the wake-null flat orbit: dynamics see only ∇V, so the orbit's V-*level* is wake-invisible and CD (D2) owns it unless pinned. Measured envelope: λ=10 max rejection, λ=100 bulletproof 5/5 (CM-6). |
| R-3 | **Fourier-anchor / flatness keeper** *(new)* | `λ Σ_{k=1..K} \|FFT_θ(V(orbit))_k\|²` | *(no analog)* | C2's stiffening face + C3's washboard | R-2 pins the mean (k=0); erosion/washboard live in the k≥1 modes — v5-gate's 2–3 washboard minima = dominant low-k Fourier content, and register capacity is set by washboard depth. Penalizing k≥1 keeps `μ²_coset ≈ 0`, i.e. keeps the *register*, not just the well. Designed orbits only (needs the parameterization). |
| R-4 | **orbit-variance / symmetry inducer** *(new)* | `λ E_{q∼data} Var_{g∼G}[V(g·q)]` | *(no analog)* | C3 | zero iff V is G-invariant ⇒ exact Goldstone flatness at any SSB vacuum. At finite λ it yields *approximate* invariance ⇒ pseudo-Goldstone with residual `μ²` (via GMOR, `μ²F² = δΣ`, δ ∼ residual asymmetry): **a soft penalty buys `n₁/₂ ∝ 1/μ²(λ)`, not the ∞ latch** — CM-16a's 12-orders flatness gap is unreachable by finite λ. Scope it to designated register units (block-untied, CM-9 parameter-separation) so it never fights the task units. |
| R-5 | **band-coverage** *(new)* | `λ·KL(soft-occupancy of x_i = 2εμ_i/γ over {latch, budget, floor} ‖ ρ_target)`, or level-repulsion on `log μ_i` within the budget band | covariance/decorrelation term (no two memory channels redundant in timescale) | C4 | from CM-16b: retention is mass-independent in the floor band (all modes decay at ≈γ/2) and `∝ μ⁻²` only in the budget band — so *retention diversity requires spectral (μ) diversity below the crossover `εμ ≲ γ/2`*, which neither R-1 (inertial M ≠ spectral μ) nor any activation statistic sees. Requires a μ-spectrum probe (Hessian eigs at data; K Hutchinson/Lanczos HVPs — the cost item). Note γ co-determines the bands: a global γ shift moves all boundaries together; this term and the γ knob must be co-scheduled. |
| R-6 | **coercivity floor** *(new)* | `λ E_{q∼far shell}[softplus(m + mean V(data) − V(q))]` | variance-term cousin (contrast diversity) | C5 | wake-null by construction (wake never visits the far field) and *sleep-aligned* (CD also wants garbage high) ⇒ no conflict with either phase. Weaker than the architectural α‖q‖² (which holds off-distribution at inference) — prefer design-in per program doctrine; R-6 is the retrofit for Deep/Conv (§7.7). **Cannot rescue a non-symplectic substrate** (measured: anchor-robustness §3) — coercivity is necessary, volume conservation is the other half. |

**Distributional generalization (R-1 ⊕ R-5 in one term):** `R_specW = λ·W₁(sorted empirical {log μ_i} , target spectrum)` — 1-D sorted Wasserstein, O(d log d), differentiable; default target = log-uniform ladder over the decades the task needs ("scale-free memory ladder"). This is the analog of SigReg/LeJEPA's pin-the-embedding-distribution-to-a-fixed-target move, transplanted from activation space to the dynamical spectrum. *(SigReg details are from my training knowledge — scout must verify before any paper cites it.)*

**Task-nullity classification (which terms are free, which are priced):**
- *Free (wake-null, T2 exact or near-exact):* R-1 when only positions are observed (proven in toy; for real CLU the V_θ↔M gauge is approximate — see open question OQ1); R-2 (orbit V-level is exactly dynamics-null); R-6 (off-manifold).
- *Priced, knowably:* R-3/R-4 price = the data's actual asymmetry along the orbit (measure it first: orbit-variance of the *data* energy is the price estimate); R-5 is **free for encode()/representation tasks** (latent timescales under-determined by the task — the CAFE case) but **priced for trajectory-fit tasks** (frequencies are wake-visible). This free/priced split *is* the task-appropriateness principle of §4.

---

## 4. Lever→task map (CAFE): which collapses are fatal, benign, or irrelevant

Organizing thesis (Head, 2026-07-20): Anomaly ← valleys/EBM locality; Event ← basin-exit/stability; Classification ← global reach. Confidence labels: **[M]** = mechanism measured in-program, **[C]** = conjectured mapping (needs the CAFE integration runs to test).

| collapse | Anomaly (VUS-PR) | Event / fault onset (h-AUROC) | Classification (Macro-F1) | continual (non-CAFE) |
|---|---|---|---|---|
| C1 mass narrowing | moderate [C] | **fatal** [C] — degradation is a slow trend riding fast dynamics; needs a slow accumulator band + responsive band | moderate [C] | moderate |
| C2 vacuum erosion | **fatal if energy-scored** [M] — post-erosion `noise_gap<0`: the score *inverts* | fatal if designed basins are the exit-detector [M-adjacent] | benign [C] | fatal for register memory [M] |
| C3 coset non-formation | benign [M: N46 scope] | possibly useful lever lost (drift register) [C] | benign [C] | fatal for continuous registers [M] |
| C4 band collapse | moderate [C] | **fatal** [C] — basin-exit sensitivity is maximal near critical damping (EP, `h*≈γ/2` onset) while trend persistence needs the budget band; one band can't do both | benign [C] | fatal |
| C5 coercivity loss | **fatal** [M-adjacent] — the anomaly score is the valley structure; a flat/inverted V scores noise as normal | **fatal** [C] — no basin ⇒ no basin-exit | moderate [C] | moderate |
| C6 forgetting structure | irrelevant offline [M] | irrelevant offline | irrelevant | **fatal** [M: N12/N13 + t-lever] |
| → regularizer set | R-2 (+R-6 mandatory; R-3 if designed vacua used) | R-6 + R-5 + R-1 (the full dynamical set) | R-1 mild; structural terms mostly off (reach is wormhole/boost territory, CM-7/CM-12, not SDR territory) | R-4 + T_φ line (t-lever §8), not γ_φ |

The map's practical content: **SDR is task-relative — ship it as a per-task recipe, not a blanket loss.** For the FD001 headline run, the minimal defensible set is R-6 + R-1 (cheap, wake-null) with R-5 as the follow-up once a μ-probe exists.

---

## 5. Numerical verification — pre-registered collapse-and-cure (all 5 predictions landed)

Toy (see PREREG for full spec): two decoupled harmonic CLU channels, learnable `(log k_i, log m_i)`, ground truth `M* = (4, 0.25)` (the seed-sweeps 16× setting), uniform mass init 0.7, fast-lever lr 10× slow-lever lr (emulating MLP-V_θ vs log_mass gradient richness), q-only wake loss unless stated. **PREREG Amendment 1** (recorded before the registered run): first run hit the pre-flagged frequency side-lobe trap (task MSE 0.84, void, archived); task loss changed uniformly to multi-scale windows; the P1 partition derivation is landscape-independent so no registered quantity changed.

### 5.1 The exact collapse law (P1 — derived, then measured to 4 dp)
For any q-only loss, `∂L/∂log k_i = −∂L/∂log m_i` pointwise (trajectory depends on `(k,m)` only through `ω² = k/m` when p0=0), so under GD the slow lever receives exactly `−(η_m/η_k)/(1+η_m/η_k) = −1/11` of each channel's required `Δlog ω²` — independent of the loss landscape.
**Predicted:** `Δlog m = (+0.158452, −0.093602)`, `Std(log m) = 0.12603` (9.1% of s\*=1.38629).
**Measured (arm A, 5 seeds):** `Std(log m) = 0.1264 ± 0.0009`; seed-0 masses `(0.8203, 0.6374)` ⇒ `Δlog m = (+0.1585, −0.0936)`; task MSE `4.4e-23`; learned k absorbed the timescales (`k = (0.205, 2.55)`). **Collapse to the lr-partition floor, exactly as derived.** This is D1 in vitro — the mechanism I attribute (with OQ1's caveat) to N7's real-code masses-stay-at-init.

### 5.2 The cure and its zero fit price (P2, P3)
Arm B (R-1 on from epoch 0, λ_s=1): `Std(log m) = 1.38629 ± 0.0000 = s*` exactly; task MSE `4.6e-23` (= arm A's; ratio ≈ 1.05, within the registered 2×); **ordering correct 5/5** (task transient seeds the sign, T3).
P3 nullity at arm B's optimum: co-shift `δlog k_1 = δlog m_1 = 0.2` ⇒ `ΔL_task = 0.0` (exactly; closed-form q depends only on ω with p0=0) while `ΔStd(log m) = 0.100`. **The regularizer moves along an exactly task-null direction — T2 verified, not just proved.**

### 5.3 The honesty arm: diversity ≠ assignment (P4)
Arm C (warm-start at the collapsed optimum — k pre-absorbed, masses uniform + 1e-3 jitter — then task+R-1): `Std` recovers to `s*` (10/10), task MSE stays ≈0 (gauge direction), but **ordering correct only 6/10** (registered interval 2–8 ✓), and the final assignment equals the *sign of the init jitter* **10/10**. My own candidate fails as predicted when applied as a patch: **R-1 restores diversity, never assignment.** Consistent with CM-5's schedule sensitivity and N8's inversion; hardens the "on from epoch 0" design rule.

### 5.4 The demarcation (P5)
Arm D (momenta also observed ⇒ mass is wake-visible; **no regularizer**): `Std(log m) = 1.3680` (1.3% off s\*, within the registered 20%), `m = (4.000, 0.259)` ≈ `M*`, ordering 5/5. **When the order parameter is wake-visible, there is no collapse and no regularizer is needed** — T1's "iff", mass-lever face, matching the erosion side (tilt δ≥0.05 immunizes with no anchor, anchor-robustness §2).

---

## 6. Verdicts

**Proven** (derivation + exact numerical agreement): T2 (zero expressivity price for wake-null order parameters — one-line proof, general); the D1 lr-partition collapse law (P1, 4 dp); exact task-nullity of the R-1 direction in the toy (P3, ΔL = 0.0); T1's mass-lever demarcation in the toy (P1 vs P5); T3's amplifier mechanism (P2 5/5 vs P4 6/10 with 10/10 jitter-sign match).
**Strongly evidenced** (others' in-program measurements, unified here): T1 on the vacuum lever (CM-6 demarcation + tilt immunity + anchor cure); C1 collapse in real training (N7, seed-sweeps); C2 mechanism (sleep-erosion); the anchor's finite-λ optimization price being a rate-not-representability cost (anchor-robustness envelope).
**Conjectured** (derivation sketch, no CLU-scale numerics yet): R-3/R-4/R-5 efficacy on real checkpoints; the §4 CAFE fatality map's [C] cells; that real-CLU mass collapse is *dominantly* D1 (the real V_θ↔M gauge is approximate, not exact — OQ1 below); R-4's `n₁/₂ ∝ 1/μ²(λ)` dial estimate via GMOR.

---

## 7. Implications for CHLU / engineer follow-ups (specify, don't build — per task)

Priority-ordered spec for `experiment-engineer` (new config group `training.structural_reg`, all default-off, epoch-0-on when enabled per T3):
1. **R-1 mass-spread** (`mass_spread_lambda`, `mass_spread_target` in log-units, `mass_spread_deadband`): trivial — `log_mass` is a leaf; **Var-form, not Std-hinge** (T3 stability derivation); acceptance test = re-run seed-sweeps item 1 uniform-init arm with R-1: predict `Std(log M)` → target and eval-MSE ≤ uniform baseline (banded-matched is the ceiling).
2. **R-6 coercivity floor** (`coercivity_margin_lambda`, far-shell radius multiplier): CAFE-anomaly-relevant; acceptance = `noise_gap` stays >0 through 1000 erosive epochs on a Deep/Conv (confinement-free) potential where the baseline goes negative.
3. **R-3 Fourier-anchor** on the designed exp-d ring (extend the existing anchor hook to k=1..K FFT modes): acceptance = washboard k≥1 modes suppressed and register capacity above the v5-gate 1–1.6-bit baseline.
4. **R-5 band-coverage**: blocked on a cheap μ-spectrum probe (K-probe Lanczos/HVP at data batch); prototype on the lattice where per-unit channel curvature is cheap (goldstone harness exists).
5. **R-4 orbit-variance**: research-grade; register-units-only scoping (CM-9 block-untied).
6. **OQ1 verification run (analyst/engineer, cheap):** instrument `exp-lattice` training to log the gradient-path partition between potential params and `log_mass` (the real-code test of D1: predict the integrated-gradient ratio ≈ the spread deficit). This decides whether real mass collapse is D1 (partition) vs a genuine identifiability gap — the one place my toy could mislead.

Open questions / risks: **OQ1** as above (D1-dominance on real MLP potentials is my biggest extrapolation). **OQ2**: R-5's μ-probe cost at d=784. **OQ3**: interactions among SDR terms are unstudied (the governor+field N12 negative warns that structurally-motivated terms can compose destructively — every pairwise composition needs its own check before shipping a "full SDR" loss). **OQ4**: SigReg/LeJEPA citation details are training-knowledge only — scout must verify before paper use; novelty of "structural (operator-space) VICReg" also needs a scout pass.

## Proposed handover updates (for the Hub)

- **§1 (new subsection candidate, "structural collapse & the SDR family"):** record T1–T3: *collapse = drift of wake-invisible order parameters, three drivers (lr-partition / CD bias / measure-zero non-formation); anti-collapse = gauge-fixing terms with provably zero expressivity price on wake-null levers; diversity ≠ assignment (regularizers must be on from epoch 0).* Demo: pre-registered, 5/5 predictions, P1 to 4 dp (`Std_A = 0.1264±0.0009` vs derived 0.12603; nullity ΔL = 0.0 exact; warm-start assignment = jitter sign 10/10).
- **§8:** add the SDR family (R-1…R-6 + R_specW) as the general Hyp-3 answer and candidate ICLR methods contribution; note the task-relative recipe (§4 map) aligned with the CAFE-mechanism thesis; minimal set for FD001 = R-6 + R-1.
- **§7 (discipline):** any future SDR experiment must state epoch-0-on vs warm-start (T3 makes the two regimes categorically different); and OQ3's composition warning (N12 precedent) before combining terms.
- **Claims-matrix note:** CM-5's "ordering inducible / magnitude designed-in" now has a mechanism-level reading (D1 partition + T3 amplifier) — no wording change required, but the V3 appendix could cite the partition law as the *why* behind the 10×-safe/100×-inverts window.
