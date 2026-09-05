# f5-corrigendum-2 — physics-theorist report

**Task + acceptance criterion:** re-scope F5 Prop-9 (the σ* discrete-FDT noise) as Newtonian-only at all three sites; state the relativistic no-go as a theorem with its control parameter; report the blast radius; clean PDF; downstream reconciliation list. **Acceptance met.**
**Status:** done. F5 is **arXiv-ready on this axis.** The corrected statement is, as in corrigendum-1, **strictly stronger** than the false one — and it turned up **two results the deep-dive did not have**: the control parameter is `d·Θ`, not `Θ` (so the deep-dive's "raise c to 5, benign" fix is **wrong at Exp-C's `d=784`**), and there is an **exact, cheap repair** (latent inverse-Gaussian mass) that keeps the coded Gaussian O-step.

---

## What I did

1. **Re-derived the defect independently** (did not take the deep-dive or `xy-lattice-theory` on trust) and found a **sharper theorem** than the one handed to me. The deep-dive's proof is a *free-particle* argument ("the p-recursion is autonomous and linear ⇒ Gaussian"). That proof is correct but its scope is `V=0`, and it leaves open exactly the question my task asked: *can an interacting `V` restore non-Gaussianity?* The right tool is a **characteristic-function argument** that needs neither `V=0` nor linearity of the damping (§1).
2. **Found and proved the `d`-amplification** (§2) — the single most consequential new fact here. It **corrects CM-17 and the deep-dive's fix ranking**, and **it invalidates the design of falsifiable F-9 as written** (which would have compared two ultrarelativistic arms and likely returned an uninformative null).
3. **Derived an exact fix that nobody had** (§3): Maxwell–Jüttner is a Gaussian scale mixture, so the correct relativistic thermostat is the *coded Gaussian O-step with a randomized inertia equal to the relativistic mass* `m₀γ_Lorentz`. This also **discharges the deep-dive's discarded X5** (its MJ-refresh demo was under-converged; mine converges and is a better fix than the one it tried).
4. **Verified everything numerically** — 5 self-contained scripts, §"How I verified". The deep-dive's `d=1` table is reproduced to ≤2e-6 from an independent derivation.
5. **Measured the blast radius on the real consumer** (§5): built Thread-9's Prop-MH1 sweep and showed it is biased by **−0.389** in a *fully Metropolis-adjusted, discretization-free* kernel. The certificate is Newtonian-scoped and one-line repairable.
6. **Edited all three F5 sites**, rebuilt the PDF clean, bumped the CHANGELOG. **Item 4 (S2 naming) checked: already correct at all three sites — no fix needed.**

---

## 1. The theorem (proven; stronger than specified)

### Lemma-9a (Gaussian smoothing) — the load-bearing new statement
Let the **last** sub-step of a Markov chain on `(q,p)` be
`p_{n+1} = D_n + σ⊙ξ_n`, `D_n` = *any* function of the pre-noise state, `σ_i>0` constant, `ξ_n ~ N(0,I)` independent of that state.
Then every invariant probability measure `μ` has momentum marginal `μ_p = ν * N(0,Σ)`, `Σ = diag(σ_i²)`; equivalently
```
|φ_{μ_p}(t)| ≤ exp(−½ tᵀΣt)   ∀ t ∈ R^d.
```
*Proof.* `φ_{μ_p}(t) = E[e^{i tᵀD}]·e^{−½tᵀΣt}` by independence; `|E[e^{i tᵀD}]| ≤ 1`. ∎

This is the whole game. It uses **nothing** about `D` — not linearity, not the value of `γ`, not `V`.

### Prop-9′ (relativistic Gibbs no-go)
`H = T(p) + V_θ(q)` is **separable**, so the Gibbs momentum marginal is **potential-free**:
`π_p(p) ∝ e^{−T(p)/T}` for *every* `V_θ`. In relativistic mode that is **Maxwell–Jüttner**, whose characteristic function decays **exponentially**, not Gaussianly (1D, `M=1`, `β=c/T`, `μ=m₀c`, `s=√(β²+t²)`):
```
φ_MJ(t) = (β/s)·K₁(μs)/K₁(βμ)  ~  C|t|^{−3/2} e^{−m₀c|t|}.
```
By Lemma-9a an invariant `π_p` would need `|φ_MJ(t)| ≤ e^{−σ²t²/2}`, which fails for all `|t| ≳ 2m₀c/σ²`. Hence **no σ — no per-mode `σ_i>0`, no full `Σ≻0`** — gives the coded relativistic Langevin a Gibbs invariant. ∎

**This answers the task's open question directly.** An interacting `V` **cannot** restore non-Gaussianity, and the reason is not "the O-step is a distinct substep" (which is a necessary but insufficient framing) — it is that **separability makes the Gibbs momentum marginal independent of `V` altogether**, so the constraint Lemma-9a imposes on `π_p` cannot be relieved by anything in `V`. A *non-separable* `H` would escape, because then `π_p = ∫e^{−H/T}dq` is a `V`-dependent mixture (and Gaussian scale mixtures can have exponential tails). CHLU's `H` is separable by construction.

**Scope, sharply.** (i) `V`-independent. (ii) `γ`-independent, and independent of the *form* of the damping (Lemma-9a never touches `D`) — so this kills nonlinear dampings too, not just `p←(1−γ)p`. (iii) The Newtonian case is exactly the **boundary**: `π_p` is Gaussian and `σ*` **saturates** the bound (`σ*²/(γ(2−γ)) = M_eff T`). The corrected statement therefore *contains* Prop-9 as its equality case. (iv) Exactly three escapes: **state-dependent `Σ`**, a **Metropolis** accept/reject *after* the kick, or non-separability.
(v) **Lemma-9a has no continuous-time analogue** — additive-noise SDEs routinely have non-Gaussian invariant laws (overdamped Langevin!). It is a statement about *discrete splittings whose final operation is an exact Gaussian convolution*. The coded `langevin_step` returns `p_next` immediately after the noise (`integrators.py:185–187`), so it bites exactly.

> **Where the deep-dive and `xy-lattice-theory` were right:** the free-particle proof (correct, narrower) and the root cause (`xy-lattice` §5(v): Gibbs-preserving underdamped Langevin damps the **velocity** `∇_pT`; the code damps `p`; these coincide iff `∇_pT ∝ p`, i.e. Newtonian). Both statements survive verbatim. And `xy-lattice`'s subtlety is the one the note now carries: **the failure is in the sampler, not the thermodynamics** — `π_q ∝ e^{−V_θ/T}` is relativity-insensitive, so a relativistic unit has a perfectly good equilibrium; the chain just does not sample it.

## 2. NEW, and it changes a recommendation: the control parameter is `dΘ`, not `Θ`

The coded `T(p) = c√(Σᵢpᵢ²/Mᵢ + m₀²c²)` has **one square root over all `d` coordinates** (`chlu_unit.py:255–263`). So the equilibrium's "relativistic-ness" is set by the **total** kinetic energy: equipartition gives `⟨T_kin⟩/(m₀c²) ≈ dΘ/2` with `Θ := T/(m₀c²)`. Non-relativistic ⟺ **`dΘ ≪ 1`**. Derived and verified:

```
Var_MJ(p_i)/(M_eff,i·T) = 1 + (d+2)Θ/2 + O((dΘ)²)   ──dΘ≫1──▶  (d+1)Θ
KL(π_p ‖ N(0,M_eff T)) = d(d+2)(d+3)/16 · Θ² + O(Θ³)   [joint, all d coords]
excess kurtosis (d=1)   = 3Θ + O(Θ²)
Var ratio (d=1)         = K₂(1/Θ)/K₁(1/Θ)   exactly
```
The `d=1` specializations `1+1.5Θ`, `3Θ`, `0.75Θ²` reproduce the deep-dive's table (`1.0150/1.1534/2.6995/16.2819`; `0.030/0.295/1.857/2.907`; `7.4e-5/6.8e-3/0.384/6.31`) **exactly** — independent confirmation, and the closed forms are new.

**Consequence (this is the part that matters).** The deep-dive's harness was `d=1`. Its conclusion *"the defect is a function of `T/(m₀c²)` alone"* is **true only at fixed `d`** (I verified the scaling lemma bit-identically, §"How I verified"). Applied to Exp-C, where `d=784`:

| config | `c` | `T` | `Θ` | **`dΘ`** | `Var_Gibbs/Var_coded` | regime |
|---|---|---|---|---|---|---|
| `experiment_c` default | 1 | 1.0 | 1.00 | **784** | **785×** | ultrarelativistic |
| `experiment_c` default | 1 | 0.01 (anneal end) | 0.01 | **7.8** | 8.0× | ultrarelativistic |
| **`finalA` (paper run)** | **5** | **1.0** | **0.04** | **31.4** | **31.4×** | **ultrarelativistic** |
| `finalA` | 5 | 0.01 | 4e-4 | 0.31 | 1.17× | marginal |
| needed for `dΘ<1` at `T=1` | **≳28** | 1.0 | 1.3e-3 | 1.0 | 1.6× | marginal |

⚠ **CM-17 and deep-dive R8 both say `finalA` used `c=5 ⇒ Θ=0.04`, "benign". At `d=784` that is `dΘ=31.4` — still ultrarelativistic, and the true momentum variance is `31×` the sampler's.** The free fix requires `c ≳ √(dT/m₀) ≈ 28` at `T=1`, not `c=5`. **Verified on a real chain** (not just the equilibrium measure): at fixed `Θ=0.1`, `Var(q)/(T/k)−1 = −0.111 / −0.196 / −0.390 / −0.633` for `d = 1/4/16/64`, against a `d`-independent Newtonian control `≤1e-3`.

## 3. NEW: an exact, cheap repair — the latent-mass thermostat

`A ↦ √A` is a Bernstein function, so `e^{−β√A}` is completely monotone in `A` and (subordinator identity)
`e^{−β√A} = (β/2√π)∫₀^∞ s^{−3/2}e^{−β²/4s}e^{−As}ds`.
With `A = pᵀM⁻¹p + m₀²c²` this exhibits Maxwell–Jüttner as a **Gaussian scale mixture with a single shared latent scale**:
```
p | s ~ N(0, M/(2s))                                            ← the coded Gaussian O-step!
s | p ~ InverseGaussian(mean = c²/(2·T·T(p)),  shape = c²/(2T²))
```
**Fix F2.** Replace the O-step by: *draw `s|p`, then* `p ← (1−γ)p + √((1−(1−γ)²)·M/(2s))·ξ`. Each stage preserves the joint `π(p,s)`, so the composite preserves `π_p = MJ` **exactly**.

Since `1/(2·E[s|p]) = T·T(p)/c²`, the physics is clean and quotable:
> **The exact relativistic FDT noise is the coded Gaussian noise with a *randomized inertia* equal to the relativistic mass `m₀γ_Lorentz`.**

This is precisely the state-dependent-`Σ` escape hatch Lemma-9a leaves open. It costs one inverse-Gaussian draw per step (closed-form, jax-able), keeps momentum persistence (so it **dominates** an exact MJ refresh, which decorrelates `p` every step and slows `q`-mixing), and **is the demonstration the deep-dive tried and discarded as under-converged (X5)** — mine converges: the `O(1)` bias `−0.311/−0.536/−0.727` collapses to `+0.0006/+0.0011/+0.0011`, i.e. the Newtonian `O(ε²)` shadow floor (`+0.00014`).

**Fix ranking (replaces R8's).**

| | fix | exact? | cost | note |
|---|---|---|---|---|
| **F1** | raise `c` or `m₀` until **`dΘ ≪ 1`** | no — `O(dΘ)` bias | one config line | ⚠ **not** `Θ≪1`; needs `c ≳ √(dT/m₀)` (≈28 at Exp-C `T=1`) |
| **F2** | **latent-mass thermostat** (§3) | **yes** (momentum marginal) | 1 InvGauss draw/step | keeps the Gaussian O-step; **dominates F3** |
| **F3** | exact MJ momentum refresh (Andersen) | yes | MJ sampler | kills momentum persistence ⇒ slower `q`-mixing |
| **F4** | Metropolis-adjust the composite | yes, **and** removes the `O(ε²)` shadow bias | `H`-evals + reversibility care | only route to exact Gibbs in `H` |
| **F5** | `kinetic_mode="newtonian_learned"` | n/a — Prop-9 holds | free | what every `T>0` law in the program assumes |

## 4. What is *not* broken (guard against over-correction)

- **The thermodynamics.** `π_q ∝ e^{−V_θ/T}` exactly, for every kinetic mode — the momentum integral factorizes out. "Relativistic CLUs have no equilibrium" is **false**. The correct claim is "the coded relativistic Langevin does not sample its equilibrium."
- **The coded `σ*` is not arbitrary** — it is the exact `Θ→0` limit of `Var_MJ` (`K₂/K₁ → 1`). It is the correct *leading-order* noise; the error is `O(dΘ)`.
- **Pathwise / `γ=0` / deterministic results** (latch, retention, GMOR, blindness, the R7 causal floor) need no invariant measure and are **untouched**.
- **Prop-MH2 (latch erosion `D=½s²`, `N_erode`)** is a statement about the Gibbs measure's flatness along the coset — configurational, hence relativity-insensitive. **SAFE.**
- **CM-16** — `t-lever` ran `experiment_d`'s default `newtonian_learned`. **SAFE** (already flagged in the matrix).

---

## 5. Blast radius (item 3) — the σ* consumer really is hit

**Thread-9 / CM-14, Prop-MH1 step 1:** *"Momentum Gibbs-refresh `p ~ N(0, M_eff·T)` — leaves the momentum marginal (hence π) exactly invariant; always accepted."*
**This is false in relativistic mode** (`π_p` is MJ). I built the exact sweep — momentum refresh + sign-symmetrized squeeze-MH with `min(1,e^{−ΔH/T})`, **no Verlet, no `ε`** — so nothing can be blamed on discretization:

| arm (target `Var(q)=T/k`) | `Var(q)/(T/k)−1` @ `Θ=1` | @ `Θ=4` | accept |
|---|---|---|---|
| A. Gaussian refresh + squeeze-MH, **Newtonian** (control) | `+0.000005` | `+0.000005` | 0.78 |
| **B. Gaussian refresh + squeeze-MH, relativistic** | **`−0.388875`** | **`−0.618490`** | 0.85/0.90 |
| C. **MJ refresh** + squeeze-MH, relativistic (fix F3) | `−0.000710` | `−0.000690` | 0.79 |
| D. **latent-mass** + squeeze-MH, relativistic (fix F2) | `−0.000150` | `−0.000062` | 0.79 |

**Verdict.** Prop-MH1's *statement* ("a momentum Gibbs-refresh leaves π invariant") is true; its *formula* is Newtonian. Step 2's acceptance `min(1,e^{−ΔH/T})` is correct for any `T(p)` (it uses the true `H`, `det J=1`). So **the certificate is repairable by a one-line change to the refresh** (arms C/D restore it). Blast radius is real but contained.

**"MALA(σ*) samples π exactly" (thread9 §(c), CM-14 "the FDT-correct σ* is load-bearing").** Two readings, both needing the qualifier:
- *(b) the standard reading* (unadjusted `π_p`-preserving O-step + MH-adjusted Verlet, as in OBABO/GLA): exactness requires the O-step to preserve `π_p` ⇒ **Newtonian-only**.
- *(a) fully-MH'd proposal*: exact for **any** `σ`, but then `σ*` is a *proposal-tuning scale*, not an FDT condition — and the Hastings ratio needs the proposal density (the O-step is a non-reversible contraction), not just `e^{−ΔH/T}`.
Either way, **"σ* is load-bearing" is a mixing/efficiency claim, not a correctness claim**, and Check 4 (`L1 = 0.0065` vs `0.0995`) was run on Newtonian toy EBMs. The `0.0995` unadjusted-vs-adjusted gap stands.

⚠ **F-9's experimental design is compromised as written** (deep-dive §11): *"re-run Exp-C at `c ∈ {1,5}`, i.e. `T/(m₀c²) ∈ {1, 0.04}`"*. At `d=784` those are `dΘ = 784` and `31.4` — **both ultrarelativistic**. The `c=5` arm is **not** a non-relativistic control, so a null result would be uninformative and could be misread as refuting the mechanism. **Re-spec before it is run** (see §7).

---

## 6. Every edit (change-log)

All three files are gitignored under `.claude/**` (verified: `git check-ignore -v` → `.gitignore:18: .claude/**` for each).

### `.claude/outputs/formalism-note.md` (internal F5 source)
| site | edit |
|---|---|
| §2.4 intro | noted that the additive Gaussian kick is the **last** sub-step (the structural fact Prop-9′ turns on) |
| §2.4 (new) | dated **Corrigendum blockquote**: σ* was stated class-level; it is Newtonian-only; Open-3's "`O(ε²)`" guess **retracted** (bias is `O(1)`, `ε`-independent) |
| §2.4 Prop-9 | retitled **"exact discrete FDT — Newtonian kinetic modes only"**; hypothesis `T=½pᵀM_eff⁻¹p` made explicit; uniqueness (`γ∈(0,2)`) stated; MNIST/Exp-C attribution moved out of the class statement |
| §2.4 (new) | **Lemma-9a (Gaussian-smoothing bound)** + 1-line proof + the discrete-splitting caveat |
| §2.4 (new) | **Prop-9′ (relativistic Gibbs no-go)** + proof + 4-part scope (V-, γ-, damping-independence; the 3 escapes) |
| §2.4 (new) | **control parameter `dΘ`**, the Var/KL/kurtosis closed forms, the scaling lemma, and the Exp-C numbers (`dΘ=784→7.8`; `c=5 ⇒ 31.4`; `c≳28` needed) |
| §2.4 (new) | **Fix table F1–F5** (engineer spec) + **F2 latent-mass derivation** (Bernstein/subordinator, InvGauss conditional, `m₀γ_Lorentz` reading) |
| §2.4 (new) | root-cause one-liner crediting `xy-lattice-theory` §5(v) + **"the failure is in the sampler, not the thermodynamics"** |
| §8 glossary | `σ_i*` row scoped "**Newtonian modes only**"; **new rows** for `Θ`/`dΘ` and Maxwell–Jüttner |
| §9 Open-3 | **struck through** + `[CLOSED 2026-07-10 by Prop-9′ — the question was mis-posed]`; the `O(ε²)` heuristic explicitly retracted |
| §9 item 2 | added the kinetic-mode scope + "raising `c` to 5 is **not** sufficient at `d=784`" |
| App-N | row (e) re-labelled **Newtonian**; **new row (e′)** with all numbers |

### `.claude/outputs/f5-arxiv-note.md` (arXiv-bound markdown)
| site | edit |
|---|---|
| §0 scope list | + "the relativistic Gibbs no-go + latent-mass repair (corrigendum-2, §6.2)" |
| **§6.2 (new subsection)** | **Lemma (Gaussian smoothing)** + proof; **Proposition 12′ (relativistic Gibbs no-go)** + proof; scope paragraph; `dΘ` control parameter; latent-mass repair; `[verified — check (e′)]` |
| **§10 limitation bullet** | the **false sentence** *"the corrected chain is `O(ε²)`-biased"* **deleted**; replaced by the kinetic-mode split (Newtonian: `σ*` + `O(ε²)` shadow; relativistic: no `σ` exists, bias `O(1)`, `ε`-independent, `∝ dT/(m₀c²)`) |
| results table | row (e) re-labelled **Newtonian**; **new row (e′)** |

> **Numbering note (same convention as corrigendum-1).** The `.md` uses a hand-maintained counter, so the no-go is **Prop 12′** (right after Prop 12, the velocity cap). The `.tex` auto-numbers, where it renders as **Proposition 11** (and the smoothing lemma as **Proposition 10**), pushing position-gated 11→12 and equivariant-neutrality 12→13. Every `.tex` cross-reference is a `\ref`, so nothing broke; `md`↔`tex` numbering already diverged in corrigendum-1 and remains documented rather than reconciled.

### `.claude/papers/f5-note/f5-note.tex` (**the copy that gates the push**)
Same four changes: new **§6.2 "Thermal budget: a no-go for additive-noise thermostats"** (`\label{sec:thermal}`) carrying `\begin{proposition}[Gaussian smoothing]\label{prop:gsmooth}` + proof, `\begin{proposition}[relativistic Gibbs no-go]\label{prop:relnogo}`, the `dΘ` paragraph, the latent-mass paragraph, and `\paragraph{Verified (check e$'$)}`; §11 limitation bullet re-scoped; table row (e′) added, (e) re-labelled; provenance appendix given a **second dated Corrigendum note** naming both retracted claims (`σ*` as class-level; the "`O(ε²)`-biased" sentence) plus script paths, seeds and check-count reconciliation (15 rows).
**Abstract untouched** — unlike corrigendum-1, the abstract never carried this error (it does not mention Langevin/FDT). Flagged in §7 as a Hub option.

**Build:** `tectonic -X compile f5-note.tex` → **exit 0, 0 unresolved refs, 0 overfull, 13 pages** (was 11). The single remaining warning is an underfull hbox inside the *strip-on-arXiv* provenance block. `f5-note.pdf` refreshed in place. `CHANGELOG.md`: new **v0.5 corrigendum-2** entry at the top.

---

## How I verified

`cd /Users/user/Desktop/CHLU && uv run --no-project --with numpy --with scipy python .claude/scratch/f5-corrigendum-2/<script>.py`

| check | claim | observed |
|---|---|---|
| `mj_theory` (A) | `d=1` Var ratio `= K₂(1/Θ)/K₁(1/Θ)` **exactly** | rel.err `≤ 2.2e-6` over `Θ∈[0.01,8]` |
| `mj_theory` (B) | `Var ratio = 1+(d+2)Θ/2 + O((dΘ)²)` | rel.err `≤ 2e-3` at `dΘ ≤ 0.05`, `d ∈ {1,2,3,10}` |
| `mj_theory` (C) | excess kurtosis `= 3Θ` (`d=1`) | ratio `0.9998` (`Θ=0.01`), `0.985` (`Θ=0.1`) |
| `mj_theory` (D) | `KL = Θ²d(d+2)(d+3)/16` | ratio `0.999–1.004` at `Θ=1e-3`, `d ∈ {1,3,10}` |
| `mj_theory` | **deep-dive `d=1` table reproduced independently** | `1.0150/1.1534/2.6995/16.2819`; `0.030/0.295/1.857/2.907`; `7.4e-5/6.8e-3/0.384/6.31` nats — all match |
| `mj_theory` (F) | char.fn. closed form `= (β/s)K₁(μs)/K₁(βμ)` vs numerical FT | `≤1.4e-14`; decay rate `→ m₀c` (`rate − m₀c − 3/(2t) = −1.7e-4` at `t=100`) ⇒ `|φ|~C t^{−3/2}e^{−m₀c t}` |
| `mj_theory` (G) | **the no-go**: `|φ_MJ| ≤ e^{−σ*²t²/2}` fails | first violated at `\|t\|=22.79` (`T=.5,γ=.1`), `3.62` (`T=1,γ=.5`); heuristic `2m₀c/σ*²` = `21.05`, `2.67` |
| `mj_dim` | `dΘ` control parameter; UR asymptote `(d+1)Θ` | `d=784,Θ=1`: `785.0013` vs `785` (0.02%); Exp-C table in §2 |
| `nogo_chain` (a) | free particle **on a torus** (`V=0`, normalizable Gibbs): coded stationary `p` is Gaussian, MJ rejected | `Var(p)=0.997297` (`=M_eff T`), exc.kurt `−0.014`; KS vs `N`: `D=0.00109, p=0.971`; **KS vs MJ: `D=0.0845, p=0`** |
| `nogo_chain` (b) | Newtonian control flat in `T`; relativistic grows with `Θ` | Newtonian `−0.0002784` **identical at all 5 `T`** (exact `T`-scale invariance); relativistic `−0.070/−0.112/−0.312/−0.537/−0.716` at `Θ=.056/.1/.5/2/8` (deep-dive: `−0.073/−0.115/−0.313/−0.538/−0.729`, ~2% apart on config) |
| `nogo_chain` (b) | deep-dive's `Θ`-only bit-identity, at fixed `d` | `(c=1,T=8)` vs `(c=0.5,T=2)` → **`−0.7163343` bit-identical**, `\|diff\|=0.00e+00` |
| `fixes` (I) | **Lemma-9a's identity** on the chain: `φ_post = φ_pre·e^{−σ²t²/2}` | rel.err `9.7e-5` (`t=.5`), `3.4e-4` (`t=1`); MC noise floor `7.9e-5` on `1.6e8` samples. **Exact (no MC):** `\|φ_MJ\|/bound = 2.7e6 → 5.0e119` (`t=6→20`, `Θ=8`) |
| `nogo_chain` (d) | **scaling lemma** (new): at fixed `d`, reduced params `(ε√(k/m₀), γ, Θ)` | two independent reparameterizations **bit-identical** `0.6932785821` (`Δ=0.000e+00`); negative control differs (`0.4669329032`) |
| `fixes` (II) | **`d`-amplification on a real chain**, `Θ=0.1` fixed | `−0.111/−0.196/−0.390/−0.633` for `d=1/4/16/64`; Newtonian control `≤1e-3`, `d`-independent |
| `fixes` (III) | **F2 latent-mass O-step alone reproduces MJ** (`Θ=8`) | `Var=130.98` vs exact `130.2555 = (K₂/K₁)m₀T`; exc.kurt `+2.847` vs `+2.907`; **KS `D=0.00133, p=0.474`** |
| `fixes` (IV) | **F2 in the full chain kills the `O(1)` bias** (fixes X5) | `−0.311/−0.536/−0.727` → `+0.00065/+0.00108/+0.00105`; Newtonian floor `+0.000136` |
| `fixes` (V) | F1 must target `dΘ`, not `Θ` | `d=16`: `c:1→5` (`Θ:1→0.04`) still leaves `−0.231` (`dΘ=0.64`); `c=12` → `−0.056` |
| `blast_radius` | **Thread-9 Prop-MH1 step 1 broken; C/D repair it** | see §5 table: `A +5e-6` · **`B −0.389 / −0.618`** · `C −7.1e-4` · `D −1.5e-4` |

**Flag provenance.** Repo `d6f8bac` (**untouched**; `git status --short` → 0 lines). Python 3.9.16, numpy 2.0.2, scipy 1.13.1, float64, ephemeral `uv --no-project` env (no repo venv, no JAX, **no checkpoints, no trained model, no training flags** — every claim is a property of the map/measure, so there is no training config to report). Seeds `default_rng(3/5/7/11/41/99/2026/12345/20260710)`. Map semantics mirror `chlu/core/integrators.py::langevin_step` (`BAB`, then `p←(1−γ)p`, then `+σξ` **last**) and `chlu_unit.py::T` (single shared `√`) and `::effective_inertia` (`M_eff = m₀M` relativistic). Map params: free particle `γ=0.2, ε=0.05, 2e5 walkers × 4000 steps, torus L=1`; harmonic `k=1, γ=0.1, ε∈{0.01,0.05}, 2e3–2e4 walkers × 2e4–3e4 steps, burn 4e3–8e3`; blast-radius `ζ₀=0.35, γ=0.1, 2e5 walkers × 3000 sweeps, burn 800`. Quadrature: Simpson on `≤4e6`-pt grids; `d=784` moments via log-space `logsumexp` radial quadrature (cross-checked against the exact `K₂/K₁` at `d=1` and the `(d+1)Θ` UR asymptote at `d=784`, agreeing to `0.02%`).

**Honesty notes.** (1) In `nogo_chain` (c) the *empirical* `|φ_chain(t)|` bottoms out at the MC noise floor `~N^{−1/2} ≈ 1.2e-4`, so apparent "bound violations" at `t≥6` there are **estimator noise, not chain behaviour**; I re-verified Lemma-9a via its exact identity instead (`fixes` (I)) and used the **closed-form** `φ_MJ` for the violation itself (no MC). (2) My Newtonian shadow floor (`−2.8e-4`) differs from the deep-dive's (`−0.0036`) — a config difference (their `m`, burn-in), not a disagreement: **both are flat in `T`**, which is the load-bearing claim, and I show that flatness is *exact* (the Newtonian chain is invariant under `(q,p)→(q,p)/√T`). (3) F2's exactness is for the **momentum marginal / O-step**; the composite `BAB∘O` retains the usual `O(ε²)` shadow bias — visible as the residual `+0.001` vs the Newtonian `+0.00014` (the relativistic shadow is slightly larger). I did not attempt a shadow-Hamiltonian expansion for relativistic `T`.

---

## Downstream reconciliation list (FLAGGED, not edited — per task scope)

| where | text | verdict | action |
|---|---|---|---|
| **`claims_matrix.md` CM-17** | "Defect controlled by `T/(m₀c²)` **alone**"; "`finalA` used `c=5` ⇒ `0.04`, **benign**"; fix ranking led by "raise `c` (free)" | ⚠ **`Θ`-only is true only at fixed `d`; "benign" is WRONG** (`d=784 ⇒ dΘ=31.4`) | control parameter → **`dΘ = dT/(m₀c²)`**; `experiment_c` default `dΘ = 784→7.8`; `finalA` `dΘ = 31.4→0.31` (**ultrarelativistic for most of the anneal**); `c ≳ √(dT/m₀) ≈ 28` for `dΘ<1` at `T=1`. Add **F2 latent-mass** as the exact cheap fix (dominates MJ-refresh). Add: the no-go holds for **any** damping (Lemma-9a), not just linear OU |
| **`claims_matrix.md` CM-14** | Prop-MH1 step 1 "`p ~ N(0,M_eff·T)` … leaves π exactly invariant" | ❌ **FALSE in relativistic** — measured `−0.389` in a discretization-free MH kernel | scope to Newtonian **or** replace the formula with "draw `p ~ π_p`" (MJ / latent-mass). Certificate then survives verbatim (arms C/D: `−7e-4`, `−1.5e-4`) |
| **`claims_matrix.md` CM-14** | "mixture ½MALA(σ*)+½squeeze … the FDT-correct σ* is **load-bearing**" | ⚠ **scope** | σ* is a *proposal/mixing* scale, not a correctness condition; exactness under MH holds for any σ. Add "Newtonian" to the *FDT-correct* framing. `L1 0.0065 vs 0.0995` (adjusted vs unadjusted) **stands** |
| **`claims_matrix.md` CM-14** | Prop-MH2 latch erosion `D=½s²`, `N_erode`; "governor destroys the invariant" | ✅ **SAFE** | configurational / no-invariant statements; relativity-insensitive |
| **`claims_matrix.md` CM-16** | mandatory flag already tightened to `fdt` **and** Newtonian (v1.8) | ✅ **already correct** | none |
| **`negative_results.md` N18** | "Shipped Langevin sampler never sampled Gibbs (per-mode `T_eff`, ~11× off)" | ⚠ **needs addendum** | *"The `fdt` **fix** restores Gibbs only in the Newtonian modes. In `relativistic` mode **no σ** does (F5 Prop-9′) — the shipped Exp-C default is relativistic at `dΘ=784`. Exact fixes: latent-mass thermostat, Metropolis, or a Newtonian mode."* |
| **`negative_results.md` N10** | `legacy`-vs-`fdt` on Exp-C | ⚠ **already flagged by CM-17** | reinforce: **both arms were non-Gibbs**; the sampler-bias hypothesis is untested, not refuted |
| **`papers/v2-short/draft.md:271`** (App-F) | "momentum-coupled Langevin sampling needs a per-mode FDT scale σ* **to target a Gibbs measure**" — stated as a class-level **neutral theorem** | ⚠ **needs kinetic-mode qualifier** | → "…in the **Newtonian** kinetic modes. For the relativistic kinetic term no noise scale targets Gibbs (theory note §6.2), the correct thermostat carrying a randomized, momentum-dependent inertia." |
| **`papers/v2-short/draft.md:388`** (App-J mandatory flag) | "All results require `langevin_noise='fdt'`" | ⚠ **incomplete** | → "…**and a Newtonian kinetic mode**". *Numbers are safe*: `t-lever` ran `newtonian_learned` |
| **`papers/v2-short/draft.md:58`** | "this holds only under the FDT-consistent noise scale σ*" | ⚠ **incomplete** | same qualifier; the `D_θ`/`n₁/₂` laws are Newtonian statements |
| **deep-dive F-9** (falsifiable) | "re-run Exp-C at `c∈{1,5}`, i.e. `Θ∈{1,0.04}`" | ❌ **design compromised** | at `d=784` both arms are ultrarelativistic (`dΘ = 784, 31.4`). **Re-spec**: `c ∈ {1, 28}` (or `m₀ ∈ {1, 784}`), or — much better — a **`relativistic` vs `newtonian_learned`** arm, or a **latent-mass arm** (F2) as the "correct sampler" reference. As written, a null is uninformative |
| **deep-dive X5** | MJ-refresh demo discarded as under-converged | ✅ **superseded** | F2 (latent-mass) converges and is a *better* fix; `fixes.py` (III)+(IV) is the demonstration |
| **`formalism-note.md` Open-3** | "exact stationary distribution of the corrected-FDT relativistic chain" | ✅ **CLOSED here** | question was mis-posed; no Gibbs stationary law exists |
| **`xy-lattice-theory` §5(v), `future_work.md` P4** | "must run `newtonian_learned`" | ✅ **already correct** | none; §5(v)'s root-cause line is now quoted in the note |
| **`chlu/core/integrators.py:109–116`** (`langevin_step` docstring) | `"fdt"`: *"exact discrete-FDT noise … matching Maxwell-Boltzmann `Var(p_i)=M_eff_i*T`"* | ⚠ **code doc, class-level & wrong for relativistic** | **engineer spec:** add *"Exact only for `kinetic_mode ∈ {newtonian_identity, newtonian_learned}`. In `relativistic` mode the Gibbs momentum marginal is Maxwell–Jüttner and **no σ yields a Gibbs invariant** (F5 Prop-9′); `σ*` is then the `T ≪ m₀c²/d` limit, with `O(dT/(m₀c²))` error."* Consider a runtime warning when `noise_mode="fdt"` **and** `kinetic_mode="relativistic"` |
| **`chlu/core/chlu_unit.py:296–320`** (`effective_mass` docstring) | *"Delegating here restores Maxwell-Boltzmann `Var(p_i)=effective_inertia()_i*T`"* | ⚠ **code doc** | **engineer spec:** same qualifier — true in Newtonian; in relativistic the target is MJ with `Var = M_eff T·K₂/K₁`(`d=1`) |
| **`chlu/core/integrators.py`** (new capability) | — | 🆕 **engineer spec (F2)** | optional `noise_mode="fdt_relativistic"`: `u=Σp²/M`; `s ~ InvGauss(mean=β/(2√(u+(m₀c)²)), shape=β²/2)` with `β=c/T`; `p ← (1−γ)p + √((1−(1−γ)²)·M/(2s))·ξ`. One shared scalar `s` per sample (not per coordinate). Exactly preserves the MJ momentum marginal. Guard `T>0`; `sqrt(0)` NaN-gradient care (cf. `xy-lattice` P1 / N47) |

---

## Open questions / risks

1. **Shadow theory for relativistic `T` is untouched.** F2 removes the `O(1)` bias but leaves `+0.001` vs the Newtonian floor `+0.00014` — a *relativistic* `O(ε²)` shadow correction I did not derive. (Related to deep-dive **O5** and formalism **Open-5**.) Cleanly scoped, cheap, and would complete the story.
2. **Is `dΘ` the whole story in `d≫1`?** I verified the `Var`/KL formulas and the chain's `d`-scaling up to `d=64` (chain) and `d=784` (measure). I did **not** verify the *chain* at `d=784` — cost, and Exp-C's `V_θ` is a trained ConvNet, not a harmonic well. The measure-level statement is exact; the chain-level extrapolation to `d=784` is **strongly evidenced, not proven**.
3. **The trained-model consequence remains conjectured** (deep-dive **O8**): whether the missing MJ tails actually drive the MNIST 3/5/8/9 imbalance. My work sharpens *how to test it* (F-9 re-spec above) but does not test it. Unchanged status: **conjectured**.
4. **`.md` vs `.tex` proposition numbering diverges further** (md `12′` ↔ tex `11`; tex Props 11/12 renumbered to 12/13). Documented in-line, as in corrigendum-1. Full `.md` renumbering remains a Hub call.
5. **The note grew 11 → 13 pages.** §6.2 is the new material. If there is a page target, the compressible pieces are the `Verified (check e′)` paragraph (duplicated in the table) and the latent-mass paragraph; the two propositions + proofs are not.
6. **Abstract not edited.** Unlike corrigendum-1 the abstract never carried this error. But the note now proves a class-level impossibility theorem that is arguably abstract-worthy ("…and show that no additive-noise thermostat can equilibrate the relativistic branch, whose Gibbs momentum law is Maxwell–Jüttner"). **Hub/Head call.**
7. **Concurrency:** `git status` clean, `HEAD = d6f8bac`, no other agent's uncommitted work present. No tracked file touched.

## Git footprint

**None.** No tracked file created, modified, or staged; `HEAD` unchanged at `d6f8bac`; no branch created. All four edited files (`formalism-note.md`, `f5-arxiv-note.md`, `f5-note.tex`, `CHANGELOG.md`) are gitignored under `.claude/**` (verified with `git check-ignore -v`); `f5-note.pdf` rebuilt in place. Scratch: `.claude/scratch/f5-corrigendum-2/{mj_theory,mj_dim,nogo_chain,fixes,blast_radius}.py`.

---

## Proposed handover updates (for the Hub)

**§7 — resolve one, correct one, add one:**
- **7.18 [PROVEN, NEW — was `f5-corrigendum-2`'s target] → ✅ LANDED (2026-07-10).** All three F5 sites corrected. `f5-note.tex` now carries **Prop `prop:gsmooth` (Gaussian smoothing)** and **Prop `prop:relnogo` (relativistic Gibbs no-go)** in a new §6.2. The corrected result is **strictly stronger** than the deep-dive's: the free-particle/linear-OU argument is replaced by a characteristic-function argument that needs neither `V=0` nor linear damping. **No σ, no Σ≻0, no V, no γ, no damping law** gives the coded relativistic Langevin a Gibbs invariant. PDF clean (13 pp, 0 unresolved refs). **F5's arXiv push is unblocked on this axis.**
- **7.18 ⚠ CORRECTION TO THE PROGRAM'S OWN NUMBER — the control parameter is `dΘ`, not `Θ`.** `T(p)` shares one square root over all `d` coordinates ⇒ `⟨T_kin⟩/(m₀c²) ≈ dΘ/2`. Deep-dive R8 and **CM-17 are wrong to call `finalA`'s `c=5` "benign"**: at `d=784` that is `dΘ = 31.4`, ultrarelativistic (`Var_Gibbs/Var_coded = 31×`). `experiment_c`'s default is `dΘ = 784 → 7.8` across the anneal (`785× → 8×`). The free fix needs **`c ≳ √(dT/m₀) ≈ 28`** at `T=1`. Closed forms: `Var_MJ/(M_eff T) = 1+(d+2)Θ/2 → (d+1)Θ`; `KL = d(d+2)(d+3)Θ²/16`; `= K₂(1/Θ)/K₁(1/Θ)` exactly at `d=1`. **`Θ`-only scaling is true only at fixed `d`** (verified bit-identically). **Never quote "c=5 is benign".**
- **7.18 🆕 An exact, cheap fix exists (F2, latent-mass thermostat).** Maxwell–Jüttner is a Gaussian scale mixture (Bernstein/subordinator): `p|s ~ N(0,M/2s)`, `s|p ~ InvGauss(mean c²/(2T·T(p)), shape c²/(2T²))`. Drawing `s|p` then running the *same* linear Gaussian O-step with variance `M/(2s)` preserves MJ **exactly**. Physics: **the exact relativistic FDT noise is the coded Gaussian noise with a randomized inertia equal to the relativistic mass `m₀γ_Lorentz`.** Costs one inverse-Gaussian draw/step; **dominates the MJ-refresh** (keeps momentum persistence). This **discharges deep-dive X5** (their refresh demo was under-converged; F2 converges: bias `−0.727 → +0.0011`, the Newtonian shadow floor). Engineer spec in `.claude/outputs/f5-corrigendum-2.md` §"Downstream".
- **7.23 [NEW — blast radius] CM-14 / Thread-9 Prop-MH1 step 1 is FALSE in relativistic mode.** "Momentum Gibbs-refresh `p ~ N(0,M_eff·T)`" is not a Gibbs refresh when `π_p` is Maxwell–Jüttner. Measured in a **discretization-free, fully Metropolis-adjusted** kernel: `Var(q)/(T/k)−1 = −0.389` (`Θ=1`), `−0.618` (`Θ=4`), against a Newtonian control of `+5e-6`. **One-line repairable** (draw `p ~ π_p`, or use F2): repaired arms give `−7e-4` / `−1.5e-4`. Prop-MH2 (latch erosion) and the governor/annealing decomposition are **SAFE** (configurational / no-invariant statements). "MALA(σ*) samples π exactly" needs the Newtonian qualifier: σ* is a *proposal-tuning* scale, not a correctness condition, once MH is applied.

**§1 (the physics) — add two lines:**
- **The relativistic Gibbs no-go.** `H` separable ⇒ the Gibbs momentum marginal `π_p ∝ e^{−T(p)/T}` is **potential-free**, and in relativistic mode it is **Maxwell–Jüttner**. Any discrete chain whose last substep is an additive Gaussian kick with fixed covariance has a **Gaussian-smoothed** invariant momentum marginal. MJ is not one (its characteristic function decays like `e^{−m₀c|t|}`). Hence **the coded relativistic Langevin has no Gibbs invariant, for any noise scale.** *The failure is in the sampler, not the thermodynamics* — `π_q ∝ e^{−V_θ/T}` is relativity-insensitive, so the unit's equilibrium is fine; the chain just doesn't sample it. Control parameter **`dT/(m₀c²)`**. (F5 Props `prop:gsmooth`/`prop:relnogo`.)
- **Relativistic inertia is random, not fixed.** The exact relativistic thermostat draws a latent inverse-Gaussian scale and injects Gaussian noise with inertia `m₀γ_Lorentz` — the "randomized mass" reading of Maxwell–Jüttner as a Gaussian scale mixture. Prop-9's `σ*` (rest inertia `M_eff = m₀M`) is exactly its `dΘ→0` limit; the error is `O(dΘ)`.

**§8 — one closure, one re-spec, two carries:**
- Deep-dive **R8 / X6 closed** (landed at all three sites; strengthened). Deep-dive **X5 discharged** (F2 supersedes the discarded demo).
- ⚠ **Re-spec falsifiable F-9 before funding it.** As written (`c ∈ {1,5}`) both Exp-C arms are ultrarelativistic (`dΘ = 784, 31.4`); a null would be uninformative. Use `c ∈ {1, 28}`, or an explicit `relativistic` vs `newtonian_learned` contrast, or add a latent-mass (F2) "correct-sampler" arm. **O8 (does the MJ deficit drive the 3/5/8/9 imbalance?) remains conjectured** and is still the cheapest open empirical question.
- Carry **O5** (relativistic running decay constant) and add **the relativistic shadow Hamiltonian** as a newly-sharpened open question: F2 leaves a residual `+0.001` vs the Newtonian floor `+0.00014`, i.e. an unquantified relativistic `O(ε²)` shadow bias. This is the last unproven piece of the sampling story.
- **S2 naming (carry-over item 4): VERIFIED CLEAN.** `F := √(M_eff)·r*` is the decay constant and `r*` the vacuum radius/condensate at **all three sites** (`formalism-note` §3.3(c)/§4.3/§8 naming-rule box; `f5-arxiv-note:151`; `f5-note.tex:137`, which states explicitly "*and not the orbit radius `r*`*"). The GMOR theorem statements carry `M_eff` and curvature, no bare `f`. **No site says "f = orbit radius." No fix required.**
