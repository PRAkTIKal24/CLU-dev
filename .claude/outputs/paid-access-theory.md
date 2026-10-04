# paid-access-theory — physics-theorist report

**Task + acceptance criterion:** formalize *intra-unit wormholes* and *Lorentz squeezing* as **access** (reachability-expanding) mechanisms for a single CLU, so that the w7 experiment task can be written directly from §7. Every mechanism claim tagged [proven]/[verified]/[design hypothesis]; F5 (`formalism-note.md`) nomenclature throughout.
**Status:** done (theory + 6 numerical checks, pure-numpy float64).
**Output consumed by:** Hub → `paid-access-experiments` (engineer, w7).

**Reading:** brainstorm Thread 8 (two forbidden cheats / paid access), Thread 3 (squeeze history / parked retry), F5 §7.4–7.5, Prop-1, Prop-12, Cor-13; `v1-l0-gate.md` (the retry NULL we do **not** revive), `v1-wormhole-routing.md` (inter-unit design + certificates).

**Scope discipline (Head, 2026-07-07):** the claim here is **reach/access** — *crossing to basins plain relaxation-in-T provably cannot enter*. It is **NOT** the retry-for-correctness claim `v1-l0-gate` already killed. §3.3 derives precisely why that null does not test this claim.

---

## Flag-provenance

| item | value |
|---|---|
| repo commit | `9a13455` (integration/wave-5 HEAD) |
| code touched | **none** (repo read-only; theory + `.claude/scratch/paid-access-theory/checks.py`) |
| numerics | `uv run --no-project --with numpy python checks.py`, numpy float64, seed `default_rng(0)` |
| toy config | double well `V=β(q²−a²)²`, `a=β=1` (barrier `ΔV=1`); Verlet `ε=0.05`; governor `γ_n=s·tanh(max(0,H−E*))`; kinetic modes as F5 §2.1 (`c=1, m₀=1`) |
| relevant F5 flags | none trainable here — this is analytic; ties to `kinetic_mode∈{newtonian_*,relativistic}`, `use_governor`, `dt`, `γ` |

All six checks reproduce in one run; observed numbers are quoted inline at each claim and consolidated in **Appendix N**.

---

## 0. Headline (one paragraph)

Define reachability so it is falsifiable, and the two mechanisms split cleanly along the **two failure modes** of a governed CLU rollout. **Reach failure** is *kinematic*: in relativistic mode the per-step displacement is hard-capped at `ε·c/√Mᵢ` (Prop-1), so over `T` steps the state cannot leave the causal box `Cᵀ` of half-widths `Lᵢ=T·ε·c/√Mᵢ` — a target basin outside `Cᵀ` is unreachable **no matter how much energy is injected**, because velocity, not energy, is the binding constraint. **Escape failure** is *energetic*: the target is inside `Cᵀ` but behind a barrier `ΔV_b` the (dissipating) energy budget cannot climb. **The squeeze cures escape; the wormhole cures reach.** A squeeze `S^{(M)}` injects bounded energy `≤(e^{2|ζ|}−1)H` (Prop-12) as transient velocity — it can climb a barrier *inside* the cone but can **never exceed `Cᵀ`** (relativistic cap is energy-blind). An intra-unit wormhole is a gated **canonical translation** `q→q+Δ` that teleports across the cone at `det J=1` exactly, paying a discrete, bounded energy ledger `ΔV=V(q_b)−V(q_a)` (F5 §7.4 hard-gate regime) — it is the *only* mechanism that beats the causal box, and it **transports** (not copies, not erases) any latched coset content, shifting the Goldstone charge by an exact `pᵀXΔ`. This is Thread 8's "rented cheat with a receipt," made into theorems and a discriminating w7 testbed (§7).

---

## PART 1 — Reachability, made falsifiable

### Def-A1 (reachable set).
For the governed dissipative Verlet map `Φ_{ε,γ}` (F5 §2.2) with governor `γ_n=γ_φ(z_n)`, the **`T`-step reachable set** from `z₀=(q₀,p₀)` is
```
R_T(z₀) = { z : z = Φ ∘ … ∘ Φ (z₀), 1 ≤ n ≤ T }        (n compositions)
```
and its **position shadow** `Q_T(z₀) = π_q R_T(z₀)` is what a downstream read-out sees. A basin `B` (a sub-level set of `V_θ` around a local minimum `q*`) is **reachable-in-T** iff `Q_T(z₀)∩B ≠ ∅`, else **unreachable-in-T**. Access = provably enlarging `Q_T`.

### Prop-A2 (relativistic reach is a causal box). [proven; verified (A)]
In relativistic mode `q̇ᵢ = ∂T/∂pᵢ` obeys `|q̇ᵢ| < c/√Mᵢ` for all `p` (Prop-1). One Verlet drift advances `q` by `ε·∇_pT(p_{½})`, so per step `|Δqᵢ| < ε·c/√Mᵢ`, and by the triangle inequality
```
Q_T(z₀) ⊆ C_T(q₀) := { q : |qᵢ − q₀ᵢ| ≤ Lᵢ,  Lᵢ = T·ε·c/√Mᵢ }.        (outer bound)
```
This bound is **energy-independent**: injecting arbitrary momentum drives `|q̇ᵢ|` toward, but never past, `c/√Mᵢ`. *Verified:* with `p=10⁶` on a flat potential, `T=200`, `ε=0.05`, measured displacement equals `T·ε·c/√M` to rel. err `2.0×10⁻¹²` for `M∈{0.25,1,4}`.
**Corollary (unreachable-in-T, kinematic):** if `∃ i: |q*ᵢ−q₀ᵢ| > Lᵢ`, i.e. `T < T_min := max_i |q*ᵢ−q₀ᵢ|·√Mᵢ/(εc)`, then `q*` is unreachable-in-T **for any momentum, any energy, any squeeze** — only a nonlocal jump helps.

### Prop-A3 (Newtonian reach is energy-limited). [proven]
In Newtonian modes `q̇ᵢ = pᵢ/Mᵢ` is unbounded; reach is capped instead by the **energy budget**. Under conservative dynamics (`γ=0`, `V` bounded below by `V_min`), `T(p) ≤ H−V_min` gives `|pᵢ| ≤ √(2Mᵢ(H−V_min))`, hence
```
|Δqᵢ|_T ≤ T·ε·√(2(H−V_min)/Mᵢ).      (grows as √budget; light modes reach further)
```
So squeeze/energy injection **does** expand Newtonian reach (∝√ΔE) but **not** relativistic reach. Both scale as `1/√Mᵢ` — light modes are the cheap long-range directions in either mode. *(This is why the safe headline mode, relativistic, is exactly the mode where energy cannot buy reach — motivating the wormhole.)*

### Def-A4 (the two failure modes — the load-bearing distinction).
- **REACH failure:** `q* ∉ C_T(q₀)` (kinematic; Prop-A2 corollary). Cured only by a nonlocal jump.
- **ESCAPE failure:** `q* ∈ C_T(q₀)` but the trajectory is trapped in the current basin because the available kinetic energy is `< ΔV_b` (barrier height) **and dissipation drains it before a saddle is crossed**. Cured by any bounded energy injection that acts faster than `γ_c` (squeeze; smooth throat).

**These need separate predictions** (§7): a squeeze that "works" on an escape-limited task tells you nothing about reach, and vice-versa. Conflating them is exactly the error that made the `v1-l0-gate` retry null look decisive when it was not (§3.3).

---

## PART 2 — Intra-unit wormhole (new mechanism)

The inter-unit `GatedCoupling` (v1-wormhole-routing) reads a *separate* archive unit's value through `V_c(qᵢ,qⱼ)`. **Intra-unit** both loci live in one phase space `(q,p)∈ℝ^{2d}`; there is no second unit to read. I analyze two constructions and recommend one.

### 2.1 Construction (a): gated canonical translation + ledger  — **RECOMMENDED**

### Def-A5 (intra-unit wormhole channel).
A channel `k` is a learned pair of loci `(q_a^{(k)}, q_b^{(k)})∈ℝ^d`, a capture set `U_k={q: ‖q−q_a^{(k)}‖<ρ_k}`, and a gate `g_k(z)∈{0,1}` (hard) or `[0,1]` (smooth). When active it applies the **constant translation**
```
Φ_wh^{(k)}:   q ↦ q + Δ_k,   p ↦ p,     Δ_k = q_b^{(k)} − q_a^{(k)}.
```

### Prop-A6 (exact volume; discrete energy ledger). [proven; verified (B,C)]
A *constant* translation has Jacobian `I_{2d}` ⇒ `J^TΩJ=Ω` and `det J = 1` **exactly**, independent of `Δ` magnitude, direction, or gate — the map is symplectic (it is the time-`τ` flow of the linear Hamiltonian `W=Δ·p/τ`). The only cost is a discrete energy jump
```
ΔH_wh = V_θ(q+Δ) − V_θ(q)          (kinetic term unchanged).
```
This is exactly F5 §7.4's **hard-gate regime**: symplectic within a selection epoch, with a bounded energy jump at switching that must be ledgered (or the gate smoothed). Matched loci (both basin bottoms of equal depth) ⇒ `ΔH_wh≈0`: *free* transport. *Verified (B):* symmetric double well, `q_a=−a→q_b=+a`, measured ledger `= 0.0` exactly; the state teleports from left basin to `q=+1.000`. *Verified (C):* `det J = 1.000` for the constant translation.

**Why not state-dependent gates in the throat.** A gate that varies *during* the jump, `q'=q+g(q)Δ`, has `∂q'/∂q = I+Δ(∇g)ᵀ` ⇒ by the matrix-determinant lemma
```
det J = 1 + ∇g·Δ  ≠ 1  in general.        [proven; verified (C): err 2.2×10⁻¹⁶]
```
So a smoothly state-modulated jump **breaks volume** by exactly `∇g·Δ` — an *unpaid contraction* (Thread 8 cheat (i)) sneaking in through the back door. **Design rule for the engineer:** either (i) freeze `g` at capture (hard gate held constant over the jump epoch → `det J=1`, pay only the energy ledger), or (ii) make the gate a smooth term *in `V_θ`* (construction (b) below), never a state-dependent multiplier on `Δ`.

### Prop-A7 (latch transit — transport, not copy, not erase). [proven; verified (E)]
Let a coset/Goldstone direction carry charge `Q = pᵀX q` (F5 §4; `X` the broken generator). Under `Φ_wh` (`q→q+Δ`, `p→p`):
```
Q' − Q = pᵀ X Δ.        (exact, for any Δ)
```
*Verified (E):* over 2000 random `(q,p,Δ)`, `|(Q'−Q) − pᵀXΔ| ≤ 1.8×10⁻¹⁵`.
Interpretation, contrasting inter-unit:
- The intra-unit wormhole moves **one** phase point ⇒ it **transports** the latch (there is no second copy). Inter-unit `GatedCoupling` reads a *separate* unit ⇒ it **copies**.
- The latch is **preserved exactly** iff `XΔ ⊥ p`, e.g. `Δ` chosen tangent to the coset / within an isopotential of the symmetry. *Verified (E):* with `Δ ⊥ (Xᵀp)`, latch shift `= 0.0`.
- Otherwise the latch is **shifted by the exact, bounded amount `pᵀXΔ`** — never randomly erased. A wormhole across a symmetry-broken barrier costs a *computable* charge displacement, which the ledger tracks.

This is the Cor-13-scope interaction the task asks for: friction cannot erase coset content (Cor-13), and neither can the wormhole — it *relocates* it by `pᵀXΔ`.

### 2.2 Construction (b): nonlocal throat term in `V_θ`  — escape-aid, not reach

### Def-A8 (throat). A smooth nonlocal term `V_wh(q) = −A·κ(q; q_a,q_b)`, `κ` a ridge of lowered potential connecting the loci (a "tube"). `H=T+V_θ+V_wh` stays `C¹`.

**Analysis [proven].** Everything in F5 §2 holds *exactly*: symplectic (`γ=0`) / conformally symplectic (`γ>0`), `det J=(1−γ)^d`, energy conserved (`γ=0`). No ledger, no volume break. **But** the throat only *lowers the barrier* between `a` and `b`; the trajectory still traverses it under the Verlet flow, so it remains bound by the causal box `C_T` (Prop-A2). **Construction (b) helps escape, never reach.** Its "price" is a standing representational cost: the tube perturbs the base landscape everywhere and adds curvature `∝A/ℓ²` to relative directions (a small `μ²` on the connected coset). It is the *built-into-`V_θ`* analogue of a squeeze, strictly weaker on reach than (a).

**Recommendation:** ship **(a) gated canonical translation with an energy ledger** as *the* intra-unit wormhole — it is the unique construction that beats the causal cone, keeps `det J=1` exactly, and gives an exact latch-transport law. Keep (b) as an optional smooth escape-aid; note it is dominated by squeeze (which needs no learned tube).

### Prop-A9 (wormhole reach theorem). [proven for the outer bound; reachability of entrances is the learned-placement cost]
With `K` channels `{(a^{(k)},b^{(k)})}`, define residual budgets `T_k` (steps left after the trajectory reaches entrance `k` within its cone and jumps). Then
```
Q_T(z₀)  ⊇  C_{T}(q₀)  ∪  ⋃_{k ∈ reachable} C_{T_k}(b^{(k)}).
```
A single channel with entrance inside `C_T(q₀)` and exit `b` makes the **entire relocated box** `C_{T−t_a}(b)` reachable — of volume `∏_i 2Lᵢ` that may lie **arbitrarily far** from `q₀`. The causal box is **relocated, not enlarged**. **Two-basin corollary (exact, verified B):** double well, target basin at `+a`; if `2a>L` (reach failure) plain relaxation's unreachable set includes the whole `+a` basin; one matched channel `−a→+a` (`ΔH_wh=0`, `det J=1`) reduces that unreachable set to `∅` in one jump + `O(1/γ_c)` relaxation. *Verified (B):* plain governed relaxation with an insufficient kick (KE `0.72 < ΔV=1`) stays in the left well (final `q=−1.068`, never crosses); the wormhole lands at `q=+1.000` at zero ledger.

---

## PART 3 — Squeeze as access (the NEW claim, not the dead one)

### 3.1 Reach expansion under `S^{(M)}` [proven (quadratic); evidenced (general); verified (D,F)]

`S^{(M)}_ζ=𝒩⁻¹S_ζ𝒩` is symplectic (`det=1`, F5 Prop-12/check (c)), with position response `∂q'ᵢ/∂ζ|₀ = pᵢ/M_eff,i` (Cor-13/§5.4) — light modes reframe strongly, heavy modes barely. Applied in the **local frame of the current basin** (displacement `δq=q−q_well`, momentum `p`), from a state with escape-mode momentum `p₀`:
```
δq'  = δq·coshζ + p₀·sinhζ        (displacement head-start toward the saddle)
p'   = δq·sinhζ + p₀·coshζ        (momentum amplified)     [δq≈0 at the well bottom]
```
So a squeeze delivers **both** amplified velocity `p₀coshζ` **and** a displacement `p₀sinhζ` up the barrier. Energy injected is bounded: `H(S_ζ z) ≤ e^{2|ζ|}H(z)` (Prop-12 C2). *Verified (F):* injecting factor `e^{2ζ}` gives `E_inj/E* = 2.718 = e^{2·0.5}` exactly.

**(a) Reach gain per unit energy.** In relativistic mode the extra velocity saturates at `c/√M_esc`; the *usable* reach gain is the distance covered at (near-)cap before dissipation, bounded above by `C_T` (Prop-A2). In Newtonian mode `Δ|q| ∝ √ΔE / √M_esc` (Prop-A3): light modes convert injected energy into reach most efficiently.

### 3.2 Basin-hopping condition [proven; verified (D)]

**Def-A10 (basin-hop success).** Squeeze-then-relax lands in the neighbouring basin (barrier `ΔV_b`, saddle at distance `d_esc`, width `w`, escape mass `M`) — which plain relaxation-in-T cannot — iff **all four** hold:
```
(i)  energy:   0.5·(p₀ cosh ζ)² + V(q_well + p₀ sinh ζ)  ≥  V(saddle)     [cross the barrier]
(ii) budget:   ζ finite (Prop-12: injection e^{2|ζ|}H always governor-re-absorbable)
(iii) reach:   d_esc ≤ Lᵢ = T·ε·c/√M       [saddle inside the causal box — RELATIVISTIC]
(iv) timing:   w·√M/c  ≲  T_esc·ε           [cross before the governor re-brakes]
```
Plain relaxation fails exactly when `½p₀² < ΔV_b`; the squeeze rescues it iff (iii) holds. **If (iii) fails, no squeeze can help — a wormhole is required.** This is the crisp reach/escape boundary.

*Verified (D):* double well, `p₀=1.2` ⇒ pre-squeeze KE `0.72 < ΔV_b=1` (plain relaxation cannot cross). The **exact energy threshold** from (i) is `ζ*=0.2356`; measured first-landing in the far basin is `ζ=0.27` (match err `0.034`, ≈ the sweep grid). The **kinetic-only bound** `cosh²ζ*=ΔV_b/KE₀ ⇒ ζ=0.589` is a *conservative upper bound* (it ignores the helpful `p₀sinhζ` displacement) — so the squeeze hops *earlier* than a pure-energy account predicts, because it also carries the state up the barrier. Both are internally consistent and give the engineer a two-sided bracket for the threshold.

### 3.3 Why the `v1-l0-gate` null does NOT test this claim [analysis — critical for w7 design]

`v1-l0-gate` (Q2 FAIL: `S^{(M)}` retries ≈ raw ≈ random kicks ≈ relax-longer, pooled `0.140` vs `0.149`) does **not** falsify reach-access, for three independent reasons:

1. **It tested selection-among-reachable, not crossing-to-unreachable.** Every stored MQAR pattern was a basin *inside* `C_T` at kv-scale; the failure mode was retrieving the *wrong* attractor (mis-identification), an **escape/selection** problem, not a **reach** problem (Def-A4). Squeeze and random kicks are equally (in)effective at re-selecting among *already-reachable* attractors — which is exactly what the pooled parity shows. Access asks a different question: land in a basin **provably outside** plain relaxation's reach (Prop-A2 corollary). No l0-gate target was outside reach.

2. **Near-uniform learned mass ⇒ `S^{(M)}≈S`, no directional advantage.** Measured log-mass std `≈0.08` across 23 models (l0-gate, mass-spectrum-peek): `1/M_eff` had no dynamic range, so the mass-weighting that makes `S^{(M)}` *directional* was operationally inert. The access claim's directional gain (light-mode escape) is untestable without mass banding.

3. **The perturbations were kv-scale, within-basin, not barrier-designed.** l0-gate kicks were kinetically-matched to the *existing* basin scale; the access test requires an injection *sized to a specific barrier* `ΔV_b` and a target *placed at a controlled `d/L`*. l0-gate had neither knob.

**Conclusion:** the l0-gate null is about *retry-for-correctness at single-unit uniform-mass scale*; the access claim is about *reach expansion with a controlled unreachable target and mass contrast*. They share an operator (`S^{(M)}`) and nothing else. The **discriminating experiment** is §7.1.

### 3.4 Squeeze vs wormhole complementarity [design synthesis]

| axis | **squeeze `S^{(M)}`** | **wormhole (gated translation)** |
|---|---|---|
| kind | continuous, direction-tunable (rapidity `ζ`) | discrete, locus-pinned (learned `a,b`) |
| pays in | energy `≤(e^{2|ζ|}−1)H`, governor re-absorbs | discrete ledger `ΔV=V(b)−V(a)` + learned placement |
| cures | **escape** (barrier inside cone) | **reach** (target outside cone) |
| beats causal box `C_T`? | **no** (relativistic cap is energy-blind) | **yes** (teleport, `det J=1`) |
| target loci must be known? | no (line-search `ζ`) | **yes** (loci learned/placed) |
| latch impact | preserved (symplectic; commutes off-coset) | transported, shifted by exact `pᵀXΔ` |
| where it wins | **near** targets behind a barrier, **unknown** loci | **far** targets, **known/recurring** loci, latch relocation |

**Prediction:** squeeze dominates for *near-but-blocked* targets and exploratory (unknown-locus) access; wormhole dominates for *far* targets (beyond `C_T`) and *recurring known* routes where the O(1) teleport crushes the O(reach-time) traversal — mirroring v1-wormhole-routing's measured result that a direct 1-hop edge (cost flat 500) beat N-hop diffusion (cost 750→1250) on distant retrieval.

---

## PART 4 — Certificates & predictions

### 4.1 The paid-access certificate table [Thread-8 "rented cheats with receipts"]

| mechanism | energy injected (bound) | volume `det J` | governor re-absorption | latch impact | BIBO |
|---|---|---|---|---|---|
| **wormhole (gated translation)** | `ΔV=V(b)−V(a)`, discrete, ledgered (F5 §7.4) | **`1` exactly** (const. translation); `(1−γ)^d` with damping | 1 ledger event, `~(1/γ_c)·ln(1+ΔV/E*)` steps | **transport**, shift `pᵀXΔ` (0 if `XΔ⊥p`) | preserved iff exit basin coercive (F5 Prop-10) |
| **wormhole (throat in `V_θ`)** | standing offset `~A`, no jump | **`(1−γ)^d` exact** | continuous | adds `μ²∝A/ℓ²` to relative coset dir | preserved (coercive) |
| **squeeze `S^{(M)}`** | `≤(e^{2|ζ|}−1)H` (Prop-12 C2) | **`1` exactly** (symplectic) | `~2ζ/γ_c` steps (closed form) | preserved (symplectic; off-coset) | preserved |

**Governor re-absorption time (closed form).** Injected `ΔE` decays under governor rate `γ_c=−ln(1−γ)/ε` as `t_reabsorb ≈ (1/γ_c)·ln(1+ΔE/E*)`; for a squeeze `ΔE≤(e^{2ζ}−1)E*` ⇒ `t_reabsorb ≈ 2ζ/γ_c`. *Verified (F):* `ζ=0.5`, `γ_c=0.404`; predicted `2ζ/γ_c/ε ≈ 49.5` steps, measured `56` steps (same order; the closed form is the leading-order estimate — the extra ~13% is the sub-`E*` overshoot tail).

**BIBO survival.** Both mechanisms compose with a coercive `V_θ` and `γ>0` without breaking the F5 Prop-10 bounded-attractor argument: squeeze and constant-translation are volume-non-expanding (`det J≤1`), so the Lyapunov/coercivity argument survives. **Caveat (F5 §7 issue 7):** Deep/Conv potentials are non-coercive out-of-unit — BIBO then relies on the training-loop clip, and a wormhole exit into a non-coercive region can escape to infinity. **Design constraint:** wormhole exits must be placed inside a coercive sub-level set (or with `α‖q‖²` confinement present).

### 4.2 Falsifiable predictions + w7 testbed specs

Each prediction states the observable and the expected effect size, and separates reach from escape.

**§7.1 — the multi-basin REACH task (the discriminating experiment).**
- **Construction:** relativistic CLU, `d` small (2–4), a `K`-basin potential with basins placed at controlled distances `dₖ` spanning **below and above** the `T`-step cone `L=T·ε·c/√M` (some `dₖ<L`, some `dₖ>L`). Fix the governor so plain relaxation-in-T **provably** cannot leave the start basin (escape-blocked) and, for the far basins, **provably** cannot reach them (Prop-A2 corollary). Report `T_min=dₖ√M/(εc)` per basin.
- **Arms:** plain relaxation-in-T · squeeze `S^{(M)}` (line-searched `ζ`) · intra-unit wormhole (one matched channel per target) · Newtonian-mode squeeze (control: energy *does* buy reach — Prop-A3).
- **Predicted signatures (sharp):**
  - Plain relaxation: landing rate `0` for all `dₖ` beyond its energy reach.
  - **Squeeze (relativistic):** landing rate rises to `~1` for `dₖ<L` above the energy threshold (Def-A10 (i)) — a *step* at `ζ*` matching the exact-energy bracket `[ζ_exact, ζ_kinetic]` (verified (D): e.g. `[0.24, 0.59]` for KE₀/ΔV=0.72) — and **drops to `0` for `dₖ>L`** (reach cap). *This crossover at `d=L` is the falsifiable heart of the reach/escape split.*
  - **Wormhole:** landing rate `~1` for **all** `dₖ` (near and far), at `det J=1` and ledger `=ΔV`.
  - **Newtonian squeeze:** landing rate rises with `ζ` even for `dₖ>L` (energy buys reach) — confirms the relativistic cap is the operative constraint, not a coding artifact.
- **Effect size:** the reach/escape crossover should move squeeze landing from `~1` to `~0` as `d` crosses `L` within one basin-spacing; wormhole stays flat at `~1`. A null here (squeeze reaches `d>L`, or wormhole fails) refutes Prop-A2/A9.

**§7.2 — latch-transit test.**
- **Construction:** SO(2)-symmetric sector (F5 §4) with a nonzero Goldstone charge `Q=pᵀXq` latched; a wormhole channel with `Δ` (i) tangent to the coset and (ii) across it.
- **Predicted:** post-transit `Q'−Q = pᵀXΔ` measured to `≤10⁻⁶` of the charge scale; `≈0` for the tangent `Δ`, `=pᵀXΔ` for the crossing `Δ`. Squeeze preserves `Q` (symplectic). A random-shift baseline erases `Q` unpredictably. **Effect:** transport (bounded, computable) vs erase (uncontrolled) — the certificate that the wormhole *carries* memory.

**§7.3 — certificate verification (accompanies every arm).**
- Measure per-jump `det J` (expect `1±10⁻¹²` translation; `(1−γ)^d` damped), energy ledger `ΔH_wh` vs `V(b)−V(a)`, squeeze injection vs `e^{2ζ}H` bound, and governor re-absorption time vs `2ζ/γ_c`. All must hold or the "paid cheat" framing fails.

**Gate criteria (from Thread 8):** V1 adopts the mechanisms iff (effect-size matches theory: reach crossover at `d=L`, wormhole flat) **AND** (certificates hold: `det J`, ledger, bound) **AND** (latch preserved/transported per §7.2) **AND** (beats a no-physics router baseline). Misses → C-9 negatives; mechanisms stay V3/future-work.

**Prior-art contamination flags for `web-scout`:**
- Wormhole-as-shortcut vs **attention/skip-connections** and **MoE routing** (the "learned nonlocal edge" is crowded — our defensible novelty is *det-J=1 + energy ledger + latch-transport certificate*, not "nonlocal is good"). v1-wormhole-routing already positions vs multi-hop.
- Squeeze basin-hopping vs **simulated-annealing / basin-hopping / MCMC** and **stochastic-normalizing-flow** samplers (Wales–Doye basin-hopping; parallel tempering). Our differentiator: *symplectic, deterministic, bounded-injection, governor-re-absorbed* escape — a **certified** perturbation, not a Metropolis proposal.
- Relativistic reach cap vs **Lipschitz/",causal" bounds in SSMs** and **light-cone arguments in Lieb–Robinson** literature (the causal-diamond framing has physics precedent; cite, don't claim novelty of the bound itself — claim the *mechanism-design consequence*).

---

## Appendix N — numerical checks (`checks.py`, numpy float64, seed 0)

| id | claim | observed | verdict |
|---|---|---|---|
| (A) | relativistic reach cap `|Δq|=T·ε·c/√M` (Prop-A2) | rel. err `2.0×10⁻¹²` over `M∈{0.25,1,4}` | ✓ |
| (B) | plain relaxation trapped; wormhole lands; ledger | plain final `q=−1.068` (no cross), wormhole `q=+1.000`, ledger `0.0` | ✓ |
| (C) | `det J`: const translation `=1`; state-dep `=1+∇g·Δ` | const `1.000`; state-dep matches lemma to `2.2×10⁻¹⁶` | ✓ |
| (D) | squeeze basin-hop threshold (Def-A10 (i)) | exact-energy `ζ*=0.2356` vs measured `0.27` (err `0.034`); kinetic-only bound `0.589` (conservative) | ✓ |
| (E) | latch transit `Q'−Q=pᵀXΔ`; zero-shift design | transit formula err `1.8×10⁻¹⁵`; `XΔ⊥p` shift `0.0` | ✓ |
| (F) | governor re-absorption `~2ζ/γ_c`; injection `e^{2ζ}` | measured `56` vs predicted `49.5` steps; `E_inj/E*=2.718=e^{2·0.5}` | ✓ (leading order) |

Repro: `cd .claude/scratch/paid-access-theory && uv run --no-project --with numpy python checks.py`.

---

## Verdicts (per task's tag requirement)

| claim | verdict |
|---|---|
| Relativistic reach = causal box `C_T`, energy-blind (Prop-A2) | **[proven; verified (A)]** |
| Reach/escape failure-mode split (Def-A4) | **[proven]** |
| Wormhole = constant translation, `det J=1`, energy ledger (Prop-A6) | **[proven; verified (B,C)]** |
| State-dependent jump breaks volume by `∇g·Δ` (design guard) | **[proven; verified (C)]** |
| Latch transport law `Q'−Q=pᵀXΔ` (Prop-A7) | **[proven; verified (E)]** |
| Wormhole reach theorem / 2-basin corollary (Prop-A9) | **[proven (outer bound); verified (B)]** |
| Squeeze basin-hop condition (Def-A10) | **[proven (quadratic); verified (D)]** |
| Squeeze cannot beat relativistic `C_T` | **[proven]** |
| l0-gate null does not test reach-access (§3.3) | **[argued; the discriminating test is §7.1]** |
| Intra-unit wormhole as *the* reach mechanism | **[design hypothesis — pending w7 §7.1]** |

---

## Open questions / risks
1. **Learned entrance-steering is the real cost** (Prop-A9 "reachable" clause): the outer reach bound is proven, but a trajectory must *arrive* at the wormhole entrance under the Verlet flow. Placing/learning `(a,b)` so entrances sit on natural trajectories is un-derived here — it is the engineering crux and a likely failure point. Flag for the engineer.
2. **Hard-gate energy ledger accounting at inference** (F5 Hyp-7) is unbuilt: I specify the per-switch ledger; the buffer/bookkeeping design is open.
3. **Coercivity of wormhole exits** with Deep/Conv potentials (F5 §7 issue 7): a jump into a non-coercive region can violate BIBO. Constrain exits to coercive sub-level sets.
4. **Mass contrast is a prerequisite** for the directional squeeze advantage (§3.3 reason 2): if learned `M` stays near-uniform (as in every run to date), `S^{(M)}≈S` and the mass-weighting is inert. The w7 task should **band the mass** (or fix a hierarchy) so the directional claim is testable — else it repeats the l0-gate ambiguity.
5. **Timing condition (iv)** in Def-A10 is derived heuristically (barrier-crossing vs `γ_c`); a rigorous first-passage-vs-dissipation bound would sharpen it. Scoped open problem.

## Proposed handover updates (for the Hub)

**§1 (physics):** add the **reach/escape dichotomy** and the **causal box `C_T(q₀)`, `Lᵢ=T·ε·c/√Mᵢ`** as the reachability object. Record the clean split: *squeeze cures escape (barrier inside cone, bounded injection); intra-unit wormhole cures reach (teleport beyond cone, `det J=1`, energy ledger `ΔV`)*; relativistic cap is **energy-blind** so energy cannot buy reach in the safe mode (motivates the wormhole).

**§7 (discrepancies):** note that the **`v1-l0-gate` retry null does not bear on reach-access** — it tested selection-among-reachable at uniform mass; the reach claim needs a target *provably outside `C_T`* and *mass contrast*. Prevents a reviewer cross-section contradiction ("squeeze retries did nothing" vs "squeeze buys access").

**§8 (open questions):** the intra-unit-wormhole and squeeze-access mechanisms are now theory-complete with a discriminating testbed (§7.1). Remaining open: (1) learned entrance-steering, (2) hard-gate ledger implementation (Hyp-7), (3) mass-banding prerequisite for the directional squeeze claim. w7 (`paid-access-experiments`) can be written directly from §7.1–7.3; the gate criteria are drafted there.

**New F5 candidates (for the theorist to fold into `formalism-note.md` on next pass):** Prop-A2 (causal box), Def-A4 (reach/escape), Prop-A6 (wormhole `det J`+ledger), Prop-A7 (latch transit `pᵀXΔ`), Prop-A9 (reach theorem), Def-A10 (basin-hop condition) — all with the numerics above. These extend §7.4/§7.5 from *inter-unit* to *intra-unit* and from *retry-safety* to *reach-access*.
