# relaxation-addressing-theory — physics-theorist report

Task + acceptance criterion: adjudicate the Hub's two-phase proposal (addressing = γ>0 relaxation, retrieval = γ=0 rollout, selector = regression amortization), Items 1–6, each labelled proven / verified-numerically / conjectured, with a ranked load-bearing list.
Status: **done** (PREREG written before any harness and before reading any engineer output; 2 post-hoc follow-up arms, thresholds committed in-file before running; all scored in §8).

> ⚠ **Reconciliation list (first-10-lines rule) — 3 sites need an owner:**
> 1. **w19 `clu-memory-architecture` §8 "retry-as-capture" and stage-8 of the feasibility table must be amended:** naive score-gated boost-retry **failed its safety prereg** (P-I4d: 2/267 correct reads destroyed; mechanism proven — a spurious attractor can outscore the true item). The surviving, measured-safe form is **codebook-gated retry** (accept only settlements at known write addresses): 0 degradations, retrieval 0.6675→**0.8475** end-to-end. Any plan or draft quoting "boost-retry" must carry the codebook-gating condition.
> 2. **The write rule as circulated ("write item c at location c") is off by a systematic bias:** the relaxation endpoint of the content is **not** the content — measured a\* = 0.988·c (1.2% inward shift from the α‖q‖² confinement alone). The writer must record **a\* = relax(c)**, not c. Any codebook built from raw contents inherits this bias.
> 3. **The shipped α‖q‖² confinement regularizer manufactures a spurious "no-item" attractor** (a local minimum at the origin for any landscape whose wells don't cover it). In my *designed* K=4 landscape it captured **80–100% of all failed reads** (dominant failure mode at every noise level). Every retrieval pipeline needs a non-codebook-settlement = reject-and-retry policy; this also pre-registers my prediction E3 for the engineer's learned-landscape run (in progress; see §3.4).

**Headline verdict: the Hub's phase-separation proposal is ACCEPTED — it survives all six items — subject to three mandatory amendments (codebook-gated retry; record a\*=relax(c) and re-derive addresses after V-updates; spurious-attractor rejection policy).** No item refutes it; Item 2's no-go is promoted to a proposition whose precise scope is exactly the two-phase boundary.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| repo state | untouched, read-only; no repo code imported; no branch |
| scripts | `.claude/scratch/relaxation-addressing-theory/{common,item1_backdoor,item3_coincidence,item4_retry,item4_followup,item5_fiber,item6_legs,item6_followup}.py` + `item*_results.json` alongside |
| env | main `.venv` Python 3.11.13, numpy 2.4.1 (no JAX, no worktree sync) |
| seeds | item1: deterministic (no rng) · item3: `default_rng(0)` · item4 & follow-up: `default_rng(7)` · item5: `default_rng(3)` · item6: deterministic |
| shared landscape (items 3/4) | 2-D, `V=0.05‖q‖²−Σ₄exp(−‖q−c_k‖²/2s²)`, s=0.35, ring R=2; dissipative Verlet **line-for-line the shipped update** (p-half, drift, p-half, ×(1−γ)); ε=0.05, γ=0.02, M=1 Newtonian, relaxations N=3000–4000, rest starts unless kicked |
| item1 | 1-D `V=(q²−1)²+θq`, exact forward-mode tangents through the same update, N=4000 |
| item5 | 1-D harmonic k∈{1,4}, same integrator; reads: 64 subsampled γ=0 rollout points vs damped settled endpoint; ridge λ=1e-6, 120 train/80 test |
| item6 | HiPPO-LegS N=64, exact A/B from Gu et al.; Euler dt=0.005 from t₀=1 (main run); follow-up = exact quadrature projection, validated vs ODE (C1 rel err **0.028**) |
| prereg | `.claude/outputs/relaxation-addressing-theory/PREREG.md`, written before any run and before reading any engineer artifact; engineer's `learned-landscape-write-read` had **only** its own PREREG (10:27) + empty log (10:48) on disk at that time — no results existed |
| relativity caveat | all toys **Newtonian**. Structural results (IFT, robustness exclusion, idempotency, LegS linearity) are kinetic-agnostic; quantitative *rates* (38% friction tax, capture rates) could shift under the relativistic T (same caveat as w19) |

Measured landscape geometry (vs my hand-derived prereg bands, all hit): well bottom −0.8000 · escape saddle **0.0340** [0.02,0.05] · **h_well = 0.834** [0.75,0.90] · **h_origin = 0.0340** [0.02,0.06] (hierarchy factor 24.5) · **margin = 1.052** [0.85,1.15].

---

## Item 1 — the back-door gradient through the address definition: EXISTS but is BENIGN. **Proven + verified (machine precision).**

**Claim.** The gradient path loss → readout → rollout → address a\*(θ) = relax_θ(c) → V_θ does **not** re-introduce Prop 5. Prop 5's zero is specific to differentiation w.r.t. **phase-space launch coordinates** (the contraction eats the tangent along the flow); differentiation w.r.t. **parameters that move the attractor itself** is governed by the implicit function theorem and is O(1) and well-conditioned.

**Derivation.** a\*(θ) satisfies ∇V_θ(a\*) = 0 with H = ∇²V(a\*) ≻ 0 (relaxation ends at a minimum; saddles are measure-zero-unstable). IFT: **∂a\*/∂θ = −H⁻¹ ∂_θ∇V(a\*)** — finite, with conditioning set by H, not by γ or the horizon. Unrolled differentiation of a contracting fixed-point iteration converges to exactly this value; the query-tangent contracts as ((1−γ)|λ|)^N → 0.

**Verified (`item1_backdoor.py`).** N=4000, γ=0.02: unrolled dq_N/dθ = −0.141589992772883**62** vs IFT −1/V″(a\*) = …**67** — rel err **3.9e-16**; independent FD cross-check agrees to 3.0e-10. Same run, same map: dq_N/dq₀ = **1.2e-18** (Prop 5, observed in the same object). Convergence unrolled→IFT: rel err 0.72 (N=50) → 0.014 (800) → 8.7e-6 (1600) — oscillatory transient, then geometric. In-basin a\*(θ) is smooth with slope −1/V″ (max rel err 1.9e-6 vs FD), with a **jump of 1.999** (= inter-minimum distance) at the θ where a near-separatrix query is re-assigned.

**Consequences.**
- The proposal's "never differentiated through" is **sufficient but not necessary**. Three legal treatments of the address node: (i) **stop-gradient** (recommended — matches the proposal, keeps phases decoupled); (ii) **implicit differentiation** (one PD Hessian solve; valid a.e. — off the measure-zero re-assignment set); (iii) unrolled backprop w.r.t. **θ only** (converges to IFT, wasteful but correct). The one illegal move is backprop w.r.t. the query/selector output through the relaxation — that factor is the 1e-18.
- Per-query loss-in-θ is piecewise smooth with re-assignment jumps; over a query *distribution* the expected loss is continuous (jump set is measure-zero per θ), so even end-to-end θ-training through the address definition is a.e. well-posed. It is **not recommended**, because it re-entangles read-training with landscape shaping and makes selector targets non-stationary — but it is not forbidden by any gradient pathology.
- **Address non-stationarity law (feeds Item 3):** when training moves V by δ∇V, every address moves by δa\* ≈ −H⁻¹δ∇V(a\*). The amortized selector's regression targets drift ⇒ addresses must be **re-derived after landscape updates** (cheap: re-relax stored contents, or first-order continuation with the same Hessian solve).

---

## Item 2 — the no-go, promoted with its exact scope. **Proven** (elementary, but the scope is the result). Strong form **refuted**.

**Proposition 7 (robustness–gradient exclusion).** Let R: 𝒜 → ℝ^m be the full read (relaxation/rollout + readout) and call retrieval **ρ-robust at a** if R is constant on B(a,ρ). Then ∇_a R ≡ 0 on int B(a,ρ), and for any differentiable loss L, any learning signal containing the factor ∇_a(L∘R) is identically zero there. Quantitatively: if osc_{B(a,ρ)}(R) ≤ ε, the mean directional derivative along any chord is ≤ ε/ρ — **the product (robustness)×(address gradient) is bounded; they are the same quantity with opposite sign requirements.** *Proof:* constancy ⇒ vanishing derivative; the quantitative form is the mean value theorem. ∎

**Scope (the load-bearing part).** The excluded set is *everything trained by backprop through the read*: one-shot GD on q₀; GD on (q₀,p₀,m); **and end-to-end training of a selector f_φ through the read**, since ∂L/∂φ = (∂f_φ/∂φ)ᵀ∇_a(L∘R)|_{a=f_φ(x)} carries the dead factor. **The Head's weakening (descend over epochs) does not rescue any of these:** the epoch/data average of a field that is pointwise ≡ 0 on basin interiors is 0 plus boundary terms, and the boundary set is the separatrix cliff layer already measured dead (Toy D staircase; engineer's cliff ratio 1.7e7, 4 protocols ≤ chance).

**Not excluded (three escape routes, all now measured):** (i) **regression on write-time (content, address) pairs** — the signal ∇_φ‖f_φ(c) − a\*‖² contains no ∇_a R factor and is O(1) (trivially; and the labels a\* exist because the writer derives them, Item 3); (ii) **θ-gradients through the relaxation** (Item 1: IFT, O(1), verified 3.9e-16); (iii) **derivative-free score selection** (Item 4: codebook-gated retry, measured 0.8475).

**Verdict on the candidate no-go:** *promoted* as: “a retrieval map that is exactly robust on a neighborhood yields zero first-order learning signal **through itself** there; hence a dynamically-robust addressable memory must source its address learning **outside the read path**.” The strong form — no useful gradient for *any* learned addressing including regression-amortized — is **refuted** (escape routes (i)–(iii)). **The two-phase proposal is precisely the architecture that respects Prop 7's boundary**, which is why it survives.

**Quotable contrast:** softmax attention is nowhere robust (every key gets nonzero weight) and therefore everywhere trainable; exact robustness ⟺ zero read-path gradient. Attention buys trainability with leakage; CLU's two-phase buys exact isolation by moving the learning signal to write time. This is a *trade*, not a defect on either side — and it is the principled answer to “why not just backprop like attention does.”

Numerical verification: no new harness needed — the engineer's γ-scan (∇ falls 7 orders, 3.3e-1→1.2e-7, while settling error saturates at 3.7e-3) and Item 1's paired 3.9e-16 / 1.2e-18 are the measured instances of the two sides of Prop 7.

---

## Item 3 — does relaxation land where the writer wrote? **Proven by construction (with the construction's exact conditions) + verified; the failure modes are ranked and none is measure-zero-fragile.**

**Proposition 9 (coincidence conditions).** With the write rule a\*_i = relax_{γ>0}(c_i), the read relaxation of query q̃ lands on a\*_i iff:
1. **Same operator, frozen V** between write and read ⇒ for q̃ = c_i coincidence is **exact** (determinism + idempotency: relax∘relax = relax). Measured: idempotency drift 0.0; σ=0 coincidence 1.0.
2. **Noise inside the basin:** coincidence(σ) = Gaussian mass of Basin(a\*_i) — pure geometry, smooth in σ, no cliff. Measured: 1.000 at σ=margin/6, **0.9965 at margin/3** (P-I3a ✓), 0.8515 at margin/1.5, **0.661 at margin** (P-I3b ✓, band [0.45,0.75]), 0.4895 at 1.5·margin.
3. **Drift budget:** if V trains between write and read, a\* moves by ≈ ‖H⁻¹δ∇V‖ (Item 1 law) and the basin deforms; coincidence requires accumulated drift + noise < margin.

**Genericity (the D3 question):** the condition is **generic, not measure-zero** — for Morse V (full measure) minima are hyperbolic and basins are open; coincidence degrades *smoothly* under perturbation of V and σ. This is the structural opposite of the D3 exact-symmetry conditions: the write rule is robust **by default** and designable further (deepen wells / widen margins).

**The measured threat ranking (surprise strength):** spurious attractors ≫ noise margin ≫ wrong-well confusion. At every σ, **80–100% of failures were captures by the origin trap** — a non-item minimum created by the α‖q‖² confinement itself — while wrong-well confusion stayed rare (0 of 2000 at margin/3; 138 vs 540 at margin). In a *designed* landscape. Corollary for the shipped code: the paper-era regularizer that guarantees BIBO also guarantees a garbage attractor unless the codebook policy treats non-codebook settlements as "miss → retry" (Item 4 provides exactly that) or the confinement is re-centered on the vacuum manifold.

### 3.4 Predictions for the engineer (registered before their results existed)
E1–E4 are in my PREREG verbatim: E1 frozen-V, converged, σ→0 ⇒ ≥99% (a miss falsifies the write rule as stated); E2 smooth geometric degradation; E3 **spurious-critical-point capture is the dominant failure mode in a learned MLP landscape** (>50% of failures); E4 drift law ‖H⁻¹δ∇V‖. At time of writing the engineer's directory contains only their own PREREG (their design — a 5-rung design-freedom ladder with a static write objective — is architecturally consistent with everything here) and an empty `full_run.log`; **the Hub should score E1–E4 against their table when it lands.** What would falsify me: systematic mismatch at σ→0 with frozen, converged V (impossible under the stated assumptions — so it would localize a harness violation: non-convergence, ε-too-large limit cycling, or V not actually frozen); or item-confusion dominating spurious-capture in their failure split (kills E3 and weakens my threat ranking).

---

## Item 4 — capture by annealing: **the acceptance structure is right IFF acceptance is codebook-gated. Naive score-gating failed its safety prereg (mechanism proven). Verified at 2-D/K=4 scale.**

**Escape pricing (P-I4a ✓, and a new quantitative law).** Boost-escape threshold measured at **KE_c = 1.380·h** (γ=0.02) vs **1.000·h** (γ=0) — Prop 2's `M* = p₀²/2h` crossed deliberately, plus a **38% friction tax** on the climb at this γ and geometry. Design rule: ladder rungs must be priced in (1+δ_fric(γ))·h units, not bare h; w19's stage-8 costing under-prices retries by ~40%.

**The prereg failures, both with proven mechanisms (this is the section's real content):**
- **P-I4d ✗ (safety):** 2/267 initially-correct reads were destroyed by the query-directed ladder. Diagnostic D1: exactly **2** queries in that class had ‖q̃−0‖ < ‖q̃−a\*_target‖ — the spurious origin attractor *legitimately outscores the true item* under the naive score s = −‖q̃ − q_settled‖. The count matches exactly; mechanism proven, not conjectured. **Naive score-gated retry is unsafe whenever any non-item attractor exists** — and Item 3 shows the shipped confinement guarantees one.
- **P-I4e ✗ (efficiency):** query-directed rung-1 recovery from the origin trap was **0.013** vs oracle **0.63**. Mechanism: aim. A kick from the origin must hit a well core (radius ≈0.5) at distance 2 — angular tolerance ≈14°; σ=1 query noise gives ≈27° aim error; misses relax back into the trap (deterministic aim ⇒ correlated failures across rungs).
- **P-I4b/c vacuous:** at σ=1, rest-start relaxation is Voronoi-like among {wells, origin plateau}: **cross-well dynamical misrouting essentially never occurs** (wrong-well starts: 4%, all info-lost). The real recoverable error class in this geometry is spurious-trap capture (29.25% of all reads).

**The follow-up arm (post-hoc, thresholds committed in-file before running — all three pass):**
- **F1 — codebook-gated acceptance** (accept only settlements within 0.15 of a known write address; score against the matched address) **+ 15° angular jitter**, rungs [0.3, 0.5, 0.8]·h: origin-trap recovery **0.861** (≥0.5 ✓), **0 degradations** of correct reads (✓, exact — for a correct-and-nearest start no codebook point can outscore the target), end-to-end retrieval **0.6675 → 0.8475** at σ=1, K=4. Info-lost success 0.074 (≈0, as it must be).
- **F2 — synthetic wrong-well starts** (the w19 Toy-D configuration), rungs [1.5, 1.6, 1.8]·h (priced above the *measured* 1.38h threshold): recovery **0.665** (≥0.4 ✓) — cross-basin capture, measured dead for gradient search in w19 (best 3/18 ≈ chance across 4 protocols), runs at **2/3 success** under derivative-free boost-retry.
- **F3 — isolation:** accepted final states 100% sub-barrier (E < V_saddle); post-settlement tail window (last 25%) **100% pure**; supra-barrier transit does visit foreign basins (foreign fraction 0.011 > 0 among moved cases).

**The M\* tension, resolved.** Yes — every useful retry crosses M\*; isolation and retry are the same threshold spent in opposite directions. But the violation is (i) **transient** (settling restores sub-barrier energy in ~ln(E/h)/2γ steps; measured 100% restoration), (ii) **read-window excludable** (tail reads never see the transit; the engineer's tail-25% read already implements this), (iii) **write-free** (retries never touch V_θ — no stored content is at risk; the only asset at risk is the current answer, and codebook gating makes the ladder monotone: never worse, sometimes better). **There is a usable window and it is the entire supra-barrier range; the price is retry count, not correctness.** The measured barrier hierarchy (h_origin = 0.034 vs h_well = 0.834, ×24.5) makes the ladder naturally hierarchical — cheap rungs fix trap errors, expensive rungs fix well errors — the w19 §8 "energy-shell filtration" is now a measured phenomenon, not a formal role.

**Verdict:** verified-numerically at toy scale, with the acceptance-structure claims **proven conditional on codebook gating**; scale generalization (d≫2, K≫4, learned V) conjectured — the known scaling risk is aim tolerance shrinking with d (solid-angle of a well core ∝ (r/R)^{d−1}), partially offset by jittered multi-retry being rejection sampling.

---

## Item 5 — the Ramsauer objection: **the distinction is real and formalizable as payload in the fiber of the endpoint map. Proven + verified in the minimal toy. But it must be argued correctly or we lose.**

**Formal statement (Prop 11, endpoint-fiber payload).** Let Π map a read to its settled endpoint. Π is many-to-one: the fiber over q\* carries the local jet of V_θ at q\* — Hessian spectrum (read as trajectory frequencies ω_i = √(k_i/M_i)), anharmonic coefficients, coset/register coordinates — none of which are functions of q\*. Modern Hopfield has a **trivial fiber**: the fixed point *is* the stored pattern; content dim = endpoint dim by construction. CLU stores content in the local geometry of V around a low-dimensional address, so content per item can exceed address dimension — this is exactly the payload>address budget (w19 item 7) and it is deliverable **only** by the rollout.

**Verified (`item5_fiber.py`).** Two items, same location, payload written in curvature (k=1 vs 4): settled endpoints identical to **2.2e-16**; settled-endpoint read at **chance (0.425)**; γ=0 trajectory frequency ratio **2.0079** vs exact √(k₂/k₁)=2 (+0.4%); linear read on 64 trajectory samples: **100%** (200 noisy trials). One number each for "the endpoint cannot carry it" and "the trajectory does."

**What the trajectory carries that its endpoint does not (the referee answer, in order of strength):** (1) **fiber payload** — content dimension decoupled from address dimension (measured above); (2) **exact sub-barrier isolation** (Prop 2) — a softmax read mixes every stored key with nonzero weight (exp-small at large β; *exactly zero* for a sub-barrier particle). Honest magnitude: "0 vs exp-small" is a modest advantage *per read*; its real force is compositional (no leakage accumulation over chained/repeated reads) — flag: that accumulation claim is **conjectured**, not measured; (3) **per-item retention control** of the thing the trajectory reads (μ² write modes; in-program measured). **What we must NOT say:** "we return trajectories, Hopfield returns points" as a novelty — Kong et al. (Nat. Commun. 2024) already retrieve dynamical trajectories by address (scout §4); and nothing here touches capacity (we lose that comparison; scout §3).
**Open (conjectured):** how many jet coefficients are linearly readable from a length-N rollout at energy E in d dims — the quantitative fiber capacity is not derived here.

---

## Item 6 — the HiPPO question: **the Hub's "addressability" answer survives in a narrowed, provable form: per-item write-time retention control + exact read isolation. My quantitative fade-law preregs partially failed; the structural claim is proven trivially and is the one to use.**

**Proven (structural, one line).** LegS's history→state map is fixed and linear (A, B constant; no input- or content-dependence). Therefore the retention/resolution profile of *every* stored feature is determined by (x, τ, T, N) alone — **per-item differential retention is impossible inside LegS**: no write-time act can exempt one item from the fade schedule. CLU's write modes assign retention per item at write time (μ²=0 latch: age-independent, 12-order flatness gap, measured in-program). This is the theorem-grade form of "addressability"; it does not overclaim that HiPPO "has no notion of items" (a linear functional can address a time window — it just fades on a fixed schedule).

**Measured (with honest prereg misses).** LegS N=64, unit Gaussian bump items (τ=2):
- **Every fixed item fades polynomially with total history length T.** Early item (x=5): amplitude 1.003 → 0.401 over T=60→38400. **Recent item at fixed age 10: 1.005 → 0.27** — the recent past fades too, as the budget spreads (CD-kernel law ℓ(x,T) = (π/N)√(x(T−x)): at fixed distance-from-present d, ℓ ∝ √(dT)). This was my registered P-I6c **miss** — a finding whose direction *strengthens* the contrast, reported as the miss it is.
- **Single-power-law preregs failed:** P-I6a primary (−0.5±0.2) ✗ and competing (−1) ✗ in the registered window (measured −0.114, pre-asymptotic); post-hoc C2 (−0.5±0.15 at large T) ✗ (measured −0.24 for early; the bump enters a **left-edge kernel-enhancement regime** — plateau at ≈0.61 for T=4800–19200 — before the deep-edge asymptote); C4 collapse ✗ (gap 0.267; the bulk formula misses the edge transition). The recent bump follows ≈ **−0.36…−0.39** over 600–19200, marginally shallower than the CD-kernel −1/2. **Verdict on the law: polynomial fading verified; the exponent is regime-dependent (bulk ≈ −1/2, edge-enhanced shallower); my single-exponent claims are refuted in detail.** C3 ✓: 0.7-crossing at T\*=3333 ∈ [2400,7000]. Bonus validation: LegS ODE state = exact L2 projection to **2.8%** (C1 — Gu et al.'s theorem, numerically confirmed).
- **Budget-sharing crosstalk is real despite exact linearity** (P-I6d ✓): a late distractor changes the reconstruction in a disjoint early window by RMS 0.036 (finite-N projection leakage). CLU's sub-barrier read has crosstalk exactly 0 (Prop 2) — conditional, as always, on write-side locality (MLP V_θ erosion is the program's known counter-channel; CM-6).

**The honest cost column (a fair comparison must say it):** LegS has *theorems* CLU lacks — O(tL/√N) whole-history approximation, Θ(1/t) gradient bound, **no dt at all**, exact timescale equivariance, and parallel-scan trainability; CLU has no whole-history guarantee, a dt/stiffness constraint, and its write/selector controller is the unsolved component (this wave's subject). Mamba's selectivity is the linear line's own input-dependent retention control — the surviving CLU differentiators against the whole SSM line are exactly three: **exact per-item latch (age-independent), exact sub-barrier read isolation, and fiber payload (Item 5)** — and capacity is explicitly not one of them (Jelassi; scout §1.3). **Recommended referee sentence:** "HiPPO-LegS provably remembers *everything, fading on a fixed schedule*; CLU remembers *chosen things, at chosen permanence, with isolated reads* — and pays for it with a write controller and the loss of the whole-history guarantee."

---

## 7. Ranked load-bearing list

1. **Item 1 (back-door benign; IFT law)** — proven + verified to 3.9e-16. Without it the two-phase training story collapses; with it, stop-grad vs implicit-diff is an implementation choice, not a correctness issue.
2. **Item 2 (Prop 7 + scope)** — proven. The architecture's license: it says *why* regression amortization is the only read-path-compliant selector training, kills read-side search permanently (incl. end-to-end selector backprop — a stronger scope than w19 stated), and survives the Head's over-epochs weakening.
3. **Item 3 (coincidence generic; spurious-attractor threat #1; a\*≠c)** — proven by construction + verified; predictions E1–E4 registered for the engineer. The write rule is sound; its enemies are spurious minima and drift, both quantified, neither measure-zero-fragile.
4. **Item 4 (codebook-gated retry)** — verified at toy scale; safety proven conditional on gating; naive gating measured unsafe. This is the sole cross-basin mechanism and it now has numbers: 0.67→0.85, 0 degradations, isolation restored, 38% friction tax on rung pricing.
5. **Item 5 (fiber payload)** — proven + verified minimally; load-bearing only in the referee fight, where it is the correct answer to Ramsauer (and "we return trajectories" is the wrong one).
6. **Item 6 (LegS structural no-per-item-control + measured fading)** — commentary/positioning; the structural half is theorem-grade, the exponent details are honest-but-messy and should be quoted qualitatively only ("polynomial, regime-dependent, no exemptions").

## 8. PREREG scorecard (registered file: `outputs/relaxation-addressing-theory/PREREG.md`)

| pred | registered | outcome |
|---|---|---|
| geometry (4 bands) | h∈[0.75,0.90], saddle∈[0.02,0.05], h_origin∈[0.02,0.06], margin∈[0.85,1.15] | ✅ 0.834 / 0.0340 / 0.0340 / 1.052 |
| P-I1a unrolled=IFT ≤1e-8 | | ✅ 3.9e-16 |
| P-I1b dq/dq₀ ≤1e-10 | | ✅ 1.2e-18 |
| P-I1c jump ≈2 + smooth branch ≤1% | | ✅ 1.9988 / 1.9e-6 (after fixing a bug in my *check*, not the physics — first version differenced across the jump) |
| P-I3a ≥0.90 @ margin/3 | | ✅ 0.9965 |
| P-I3b [0.45,0.75] @ margin | | ✅ 0.661 |
| P-I3c origin ≥25% of failures | | ✅ 80% (and 93–100% at lower σ — under-predicted the dominance) |
| P-I4a KE_c/h ∈[1.0,1.4] damped, [1.0,1.1] γ=0 | | ✅ 1.38 / 1.000 |
| P-I4b undirected wrong-well [0.30,0.65] | | ⚠ **vacuous** — class empty at σ=1 (n=0); rest-start relaxation is Voronoi-like; replaced by F2 |
| P-I4c directed ≥0.75 / oracle ≥0.85 | | ⚠ vacuous (same); F2 committed 0.40, measured 0.665 |
| P-I4d 0 degradations | | ❌ **FAILED: 2/267** (query arm). Mechanism proven (D1: exactly the 2 queries where the origin outscores the target address). **The wave's safety finding.** Fixed by codebook gating (F1: 0). |
| P-I4e origin recovery ≥0.90 rung-1 | | ❌ **FAILED: 0.013** (aim error 27° vs 14° tolerance; oracle 0.63). F1 (jitter+gating, committed ≥0.5): **0.861**. |
| P-I4f isolation | | ✅ (as F3: tail purity 1.0, final E sub-barrier 1.0, transit foreign 0.011>0) |
| P-I4g info-lost ≤0.15 | | ✅ 0.11 / 0.00 (oracle 0.30 — oracle is not a realistic arm) |
| P-I5a/b/c | | ✅ 2.2e-16 / 2.0079 / 1.00 vs 0.425 |
| P-I6a slope −0.5±0.2 (primary) or −1 (competing) | | ❌ **both failed** (−0.114 in registered window; pre-asymptotic + edge enhancement) |
| P-I6b T\* ∈[170,660] | | ❌ no crossing in registered window (constant in amp~τ/ℓ was wrong); post-hoc C3 (committed [2400,7000]): ✅ 3333 |
| P-I6c recent ≥0.8 always | | ❌ **FAILED — recent past fades too** (0.63@2400, 0.27@19200); favorable-direction miss, corrected law ℓ∝√(dT) |
| P-I6d crosstalk >0.02 | | ✅ 0.036 |
| *post-hoc committed:* F1 ≥0.50 & 0 degr. / F2 ≥0.40 / F3 / C1 <5e-2 / C2 −0.5±0.15 / C4 <0.15 | | ✅ 0.861,0 / ✅ 0.665 / ✅ / ✅ 0.028 / ❌ −0.24 / ❌ 0.267 |

**Verdict labels.** *Proven:* Prop 7 + scope; Item-1 IFT benignity; Item-3 coincidence conditions + genericity; Item-6 structural no-per-item-control; Prop 11 fiber statement. *Verified-numerically:* everything in the flag table at the quoted precision; codebook-gated retry at 2-D/K=4. *Conjectured:* retry scaling in d and K; leakage-accumulation advantage over softmax under chained reads; fiber capacity count; that the threat ranking (spurious ≫ confusion) transfers to learned MLP landscapes (my E3 — engineer will settle it).

## Open questions / follow-ups / risks
- **OQ-1 (scale):** all Items-3/4 numbers are one 2-D geometry, K=4, single seed per script. Mechanisms (D1 exact count, aim geometry, idempotency, IFT) are seed-independent; the *rates* (0.861, 0.665, 0.8475) need replication before print. Aim tolerance shrinks as (r/R)^{d−1} — the d-scaling of jittered retry is the decisive unknown.
- **OQ-2:** friction tax δ_fric(γ, path length) — measured 38% at one point; a two-line theory (∫2γ·KE dt along the climb) is derivable and would let ladders be priced analytically. Cheap follow-up.
- **OQ-3:** relativistic arm — all structural results kinetic-agnostic, all rates Newtonian-only (standing w19 caveat).
- **OQ-4:** Item-5's fiber capacity (how many jet coefficients per read) is the quantitative version of the flagship's payload>address principle; worth a dedicated derivation if the memory paper leads with it.
- **Risk:** my E1–E4 remain unscored until the engineer's run lands; if their failure split contradicts E3, the Item-3 threat ranking must be revised (the write rule itself is not at risk — E1 is construction-proven).

## Proposed handover updates (for the Hub)
- **§1 (memory formalism):** adopt the two-phase architecture as *accepted with three amendments*: (1) retry acceptance must be **codebook-gated** (naive score-gating measured unsafe, mechanism proven); (2) writer records **a\*=relax(c) ≠ c** (1.2% shift measured from confinement alone) and re-derives addresses after V-updates (δa\* = −H⁻¹δ∇V); (3) non-codebook settlements = reject-and-retry (the α‖q‖² confinement manufactures a spurious origin attractor that caused 80–100% of designed-landscape read failures). Record Prop 7 with its scope: **end-to-end selector backprop through the read is as dead as raw address GD**; regression amortization / IFT-θ / derivative-free retry are the three compliant signal routes.
- **§1/§7 (Prop 5 refinement):** Prop 5 kills tangents w.r.t. *launch coordinates*, not w.r.t. *parameters* — ∂a\*/∂θ is IFT-valued, O(1), verified to 3.9e-16 against unrolled backprop. "Forward-only" is sufficient, not necessary; stop-grad recommended for phase hygiene, not correctness.
- **§7 (corrections to w19/stage-8):** boost-retry costing must use (1+δ_fric)·h (measured 1.38h at γ=0.02, exactly 1.000h at γ=0); the "energy-shell filtration" is now measured (h_origin:h_well = 1:24.5, hierarchical ladder works); w19's "retry-as-capture" survives **only** in codebook-gated form.
- **§8:** cross-basin capture, dead under 4 gradient protocols in w19, runs at **0.665** under codebook-gated boost-retry (F2) and end-to-end retrieval goes 0.67→**0.85** (F1) — the derived-address retrieval gate's ≥80% prereg now has a measured mechanism behind it. Referee kit: Item-5 fiber answer to Ramsauer (never "we return trajectories" — Kong 2024); Item-6 sentence for HiPPO ("chosen things, chosen permanence, isolated reads" vs "everything, fading on schedule") + the measured recent-past-fades-too law (ℓ ∝ √(dT)), quoted qualitatively.
- **Owner needed:** score my E1–E4 against `learned-landscape-write-read` when it lands; propagate the three reconciliation items in my header block.
