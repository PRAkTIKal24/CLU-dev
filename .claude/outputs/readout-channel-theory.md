# readout-channel-theory — physics-theorist report (w26)

Task + acceptance criterion: one rigorous pass over the read-out channel — derive the reach
constant κ (Q1), rule on the payload-dependent lifetime (Q2), pre-register both excursion arms (Q3).
Status: **done.** No tracked code touched. Q3 predictions were posted early at
`.claude/outputs/readout-channel-theory/Q3_PREDICTIONS.md`.

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). FIVE items.**
> 1. ⛔ **w25's reach bound `s ≳ |a|max/κ`, κ = O(3), is superseded.** The correct criterion is a
>    **saddle condition on `L = |z_i| = √(|c_i|² + a_i²)`**, not on `|a_i|/s`, and it is verified to
>    ±2.5 % on the shipped integrator and **31/32 items** on the trained shipped `V` with **zero free
>    parameters** (§1). `r2-geometry-revival` §4/§8's `κ≈3` wording and the proposed negative-result
>    text must be restated. **Owner: `doc-curator` (N⟨next⟩ text) + Hub (§6/§7).**
> 2. ⛔ **`r2-geometry-revival` §5's conjecture** ("sep/w correlates because both numerator and
>    denominator drift smoothly with d") is **replaced, not confirmed** (§2): the denominator is the
>    write's *chosen* `s`, the signal is in `sep`, and **`sep` alone classifies 11/11 cells** where
>    `sep/w` classifies only d ≥ 4. **Owner: `doc-curator`.**
> 3. ⛔ **`mia-decay-measurement` D3's two proposed fixes are BOTH refuted** (§3): (a) `payloads*amps`
>    kills the value criterion at `A = 1 − tol/|a_i|` (verified to the digit); (b) launching at
>    `q₂ = S(q_addr)` hands 67 % of queries the answer with no dynamics. **The engineer must not
>    implement either.** Recommended instead: **(d) gated-stiffness payload channel** — measured
>    retention **1.000, payload-independent, at every amplitude down to the floor**. **Owner:
>    `experiment-engineer`.**
> 4. ⚠ **`learned_confine = 0.05` and the ball radius `R = 1.0` are reach parameters, not free
>    knobs** — reach depends on `2α|c_i|`, so both enter the capacity ceiling (§1.6, §4). **Owner:
>    Hub §3 config note.**
> 5. ⚠ **`mia-decay` §5's retention magnitudes are placement-dependent**: on a max-radius ring the
>    same store gives 0.500 at `A = 0.06` where their controller-placed disk gives 0.886 (§3.1). Any
>    quoted retention number needs its `|c|` distribution stated. **Owner: `experiment-engineer`.**

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** capacity (the reach condition) + lifetimes (the payload-dependence ruling).
- **Laundering control:** N/A for derivations. Every numerical check states **shipped vs idealised**
  in its own row; the two headline checks (§1.5, §3) use the **shipped `V` and the shipped read**.
- **Falsifies:** a derivation contradicting `r2-geometry-revival` §4 (force collapse 5.1×,
  corr −0.887 with `|a_i|`, half-excursion 0.824 → 1.000). **It does not** — §1.5 reproduces that
  cell bit-for-bit (`w_atom` 0.3013 vs their 0.30129, strict 0.9473 vs 0.947266) and explains all
  three numbers.
- **Does NOT falsify:** predicting a smaller wall movement than hoped; an impossibility result on
  either excursion arm.

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| base commit | local `main` @ **`ff85573`**, working tree clean, **no tracked file touched** |
| JAX / venv | **0.9.0**, equinox 0.13.4, main venv (protocol §4, no worktree) |
| PREREG | `.claude/outputs/readout-channel-theory/PREREG.md`, written before any measurement script existed; **Amendment A1** declared before `surrogate.py` was written |
| scripts | `.claude/scratch/readout-channel-theory/`: `kappa_static.py`, `geom.py`, `dyn_toy.py`, `criterion_U.py`, `fit_reach.py`, `score_trained.py`, `cells.py`, `surrogate.py`, `q2_designed.py`, `q2_round2.py` |
| learned cell | `d=4, K=16`, `min_atoms_base` ×4 ⇒ **8192 atoms**, GLOBAL write, 600 Adam(3e-3) wd 1e-4, n_perturb 32, σ_addr .25, σ_pay .6, margin .15, barrier .2, payload_index 4, write key `PRNGKey(seed+7919)`, query seed 0 — **identical to `r2-geometry-revival`'s `mech_reach.py`** (reproduced: `w_atom` 0.3013 / 0.1841, strict 0.9473 / 0.5859) |
| read (learned) | shipped `_two_phase`: γ_address .05 × 400 → γ_read .02 × 800, dt .05, tail .25, 8 subsamples, `payload_tol` 0.1, `query_sigma` 0.15 fixed_norm, σ_p .05 |
| designed store | shipped `AtomStorePotential(dim=3, capacity=8, α=0.02, s=s_pay=0.35, κ=1.0)`, `designed_payloads(8, seed=0)`, **ring placement** ρ = 2.2869 (sep 1.750) and ρ = 2.0120 (sep 1.540 = `d_safe`); read `experiment_controller_mvp`: γ_address .05 × 400 → **γ_read 0.0** × 800, `n_query` 16, tol 0.1 |
| seeds | learned cells **1 seed (0)** (declared limitation, §6); designed-store curves **3 seeds (0,1,2)**; toy sweeps 1 seed × 64 queries/point |
| langevin_noise | **N/A** — deterministic Verlet everywhere, no temperature |
| N94 | learned cells: 600 write steps (shipped default). Designed store: **no training anywhere** |

---

## 1. Q1 — the reach condition, derived and verified

### 1.0 The correct object (this is the result)

For a well of depth `D`, width `s` at `z_i = (c_i, a_i)` in a confinement `α|q|²`, **every**
stationary point of `V` lies on the ray from the origin through `z_i`, at distance `R` from `z_i`
solving

```
(D·R/s²)·exp(−R²/2s²)  =  2α·(L − R),        L ≡ |z_i| = √(|c_i|² + a_i²)          (★)
```

(proved by solving `∇V = 0` componentwise: the well force is parallel to `q − z_i`, the confinement
to `q`, so both must be parallel to `z_i`.) Substituting `v = R/s`, `A = L/s`,
`β = D/(2αs²)` turns (★) into `h(v) ≡ v(1 + β e^{−v²/2}) = A`, whose root structure is:

| `β` / `A` | roots | meaning |
|---|---|---|
| `β ≤ e^{3/2}/2 = 2.2408` | one, ∀A | **no spurious minimum can exist, ever** |
| `A < κ_stat(β) ≡ h(v_b)` | one (`R₁`) | the well is the only attractor: reach unconditional |
| `A > κ_stat(β)` | three `R₁<R₂<R₃` | well minimum, **saddle at `R₂`**, spurious minimum near the origin |

The launch point sits a distance `|a_i|` from the target (payload-zero guard), so:

> ### **(U) REACH CRITERION.** The query is captured by item `i` iff
> ### `L/s < κ_stat(β)` **or** `|a_i| < R₂(L, s, D, α)`.

**What sets κ.** Every closed form for the ceiling is a **square root of a logarithm** —
`κ_stat ≈ √(2 ln β)·(1+…)`, and the conservative address-hold form derived in the PREREG is
`κ_R = √(2 ln(D/(2α|c|s)) − 1)`. Measured (`kappa_static.py`): `κ_stat` = 3.33 / 4.08 / 4.67 / 6.06
at `β` = 10 / 10² / 10³ / 10⁶. **This is why κ is stubbornly O(3) and why no amount of depth buys
reach:** raising `κ_stat` 4 → 5 needs `β` ×55 (at `s=0.3`: `D` 0.69 → 37.9); 4 → 6 needs `β` ×9200
(`D` → 6.3·10³). **Reach is logarithmically un-buyable in the well depth.** The only levers with
leverage are `s`, `|a|max`, `α` and `|c|` — and all four enter (★) algebraically.

### 1.1 Regime of validity (stated before the evidence)
Single dominant well; isotropic Gaussian; quadratic confinement centred at the origin; launch at
rest on the payload-zero manifold; deterministic damped dynamics with a time budget long compared
to the payload descent `t_pay ≈ (η s⁴/(D a²))e^{a²/2s²}` (verified slack, §1.4); neighbouring wells
neglected (**this is the one assumption that fails**, §2/§5).

### 1.2 The form the engineer can evaluate (no idealisation needed)
For a *proposed configuration*, do not use the closed form. Evaluate on the real `V`:

```
FLOW TEST:  x ← x − lr·∇V(x)  from  (c_i + ξ, 0)  in the FULL latent, 4000 normalised steps.
            Item i is reachable  ⟺  the flow converges within  sep/2  of  z_i.
```
Cost: `K × 4000` gradient evaluations (seconds). It scored **31/32** on the trained shipped `V`
(§1.5). **⚠ Do not pin the payload channel at 0 while descending** — the pinned variant is the
conservative bound and scores only **23/32** (§1.5, and it is the same 1.36× conservatism that made
the PREREG's closed form `Ψ` too tight).

### 1.3 Verification 1 — the exact criterion on the **shipped integrator** (idealised `V`)
`dyn_toy.py`: one Gaussian well + confinement in dim 5, **shipped** `_two_phase` schedule and query
construction, 64 queries/point, capture threshold `a*` bisected on a 0.1 grid.

| variant | `s` | `D` | `α` | `|c|` | **`a*` measured** | `a_U` from (U) | `a*/a_U` | `Ψ` (PREREG form) | `κ_stat·s` |
|---|---|---|---|---|---|---|---|---|---|
| base | .30 | 1.0 | .05 | 0.9 | **1.050** | 1.023 | 1.026 | 0.748 | 1.232 |
| α halved | .30 | 1.0 | .025 | 0.9 | **1.146** | 1.092 | 1.050 | 0.828 | 1.289 |
| D ×4 | .30 | 4.0 | .05 | 0.9 | **1.157** | 1.155 | 1.002 | 0.900 | 1.342 |
| site near centre | .30 | 1.0 | .05 | 0.3 | **1.150** | 1.195 | 0.962 | 0.871 | 1.232 |
| wide | .45 | 1.0 | .05 | 0.9 | **1.550** | 1.522 | 1.018 | 1.047 | 1.742 |
| narrow | .184 | 1.0 | .05 | 0.9 | **0.648** | 0.638 | 1.017 | 0.494 | 0.804 |
| D = 3.2 | .30 | 3.2 | .05 | 0.9 | **1.150** | 1.135 | 1.013 | 0.877 | 1.325 |

**(U): ratio 1.013 ± 0.025 (rel. sd 2.4 %) over 7 variants.** `Ψ` is systematically 1.36× too tight
(1.357 ± 0.064) and `κ_stat·s` 15 % too loose (0.871 ± 0.036) — both scale correctly, neither is the
threshold. The transition is a **saddle-node bifurcation**: `both` goes 1.000 → 0.000 in one 0.1
grid step at every variant. Failures collapse to the origin (`|x_end|` 0.014 → 0.001 vs `|c|` = 0.9):
**P1.8 confirmed, 100 %.**

### 1.4 Which of the three candidate mechanisms binds
| mechanism | fires on the trained `V`? |
|---|---|
| **(S) spurious minimum on the payload ray** (`y ↦ V(c_i,y)` has an interior minimum) | **0 / 32 items, both widths.** Never. |
| **(T) payload-descent time budget** | slack by ≥ 10× at both widths (`t_pay` 0.26 vs `T_read` 40 at `s=0.30`) |
| **(R/U) address loses the site because the well's address stiffness is payload-gated** (`k_eff = (D/s²)e^{−a²/2s²}` vs the confinement's constant pull `2α|c_i|`) | **the binding one** — 31/32 |

⇒ **The reach failure is an ADDRESS failure, not a payload failure.** This explains r2's unexplained
observation that `basin ≡ strict` at every failing cell: they are the same event. The stored value's
only role is to *gate* the address stiffness by `e^{−a_i²/2s²}`.

### 1.5 Verification 2 — the **trained shipped `V`**, zero free parameters (the headline)
`fit_reach.py` reproduces r2's `mech_reach` cell exactly, then fits `V0 + D(1−e^{−r²/2s²})` to each
site's direction-averaged radial profile (64 directions × 40 radii; **R² ≥ 0.9938 at all 32 sites**)
and feeds the measured `(D_i, s_i, |c_i|)` into (★).

| width | `D_fit` median (range) | `s_fit` median | (U) classifies `strict>0.5` | flow test | **pinned-slice test** | cell strict: predicted vs measured |
|---|---|---|---|---|---|---|
| 0.30 | **0.910** (0.722–1.045) | 0.320 | **15/16** | 16/16 | 10/16 | 1.000 vs **0.9473** |
| 0.15 | **0.459** (0.371–0.504) | 0.200 | **16/16** | 15/16 | 13/16 | **0.625** vs **0.5859** |

The single (U) miss is item 15 (`a=+1.0`, predicted margin +0.333, measured strict 0.406). At width
0.15 the criterion even calls the *marginal* item: item 9 (`a = 0.600`) has `a_U = 0.596`, margin
**−0.004**, and measures **0.469**.

**Where failures go — the mechanism, caught in the act.** Every failed flow at width 0.15 lands at
`|x_end| = 0.064`, `y_end = 0.067`: that is **item 13**, the single site the farthest-point sampler
placed at the ball centre (`|c₁₃| = 0.064`, `a₁₃ = 0.067`). 6/7 failures at width 0.15 and the one
failure at width 0.30 are captured by it. The confinement drags the query to the origin and the
nearest-the-origin item eats it.

**A new, unreported behaviour of the write (measured here):** `corr(s_fit, |a_i|) = +0.821` at width
0.30 and **+0.630** at width 0.15 — the write gives **1.31× wider wells to the far-payload items**
(0.353 vs 0.270). The write is already solving the reach problem the only way it can. It also digs
**shallower** wells when the atoms are narrow: `D` 0.910 → 0.459 as `s` 0.320 → 0.200, i.e.
`D ∝ s^{1.46}` — so `atom_init_width` moves the reach ceiling through **both** `s` and `D`.

### 1.6 The bound, in the form the engineer asked for
For a proposed `(D, s, α, |c|max, |a|max)`: reachable **iff** `|a|max < R₂` from (★), i.e. compute
`a_U = a_ceiling(|c|, s, D, α)` (`criterion_U.py`, 30 lines, no JAX). Two-sided design window:

```
   |a|max ≤ a_U(s, D, α, |c|)                      [REACH — this task]
   2s + c_j σ_q ≤ sep                              [MERGE — confirmed §4.2]
```
Both must hold at the **read** width. Shipped d=4 operating point: `a_U(0.302, 0.91, 0.05, 0.98)`
= **1.06** against `|a|max = 1.0` — a **6 % margin**. The shipped store is sitting *on* the reach
boundary, which is why every one-flag intervention moves it.

---

## 2. Q1, second half — why `sep/w` is a razor-sharp classifier and still not causal

**w25's conjecture is replaced.** Not "both numerator and denominator drift smoothly with d", but:

1. **The denominator is a consequence, not a variable.** `w` is the width the *write chooses*, and it
   is essentially a function of `d` alone in the r2 table (0.245 / 0.271 / 0.306 / 0.324 / 0.354 /
   0.364 at d = 2/3/4/5/6/8; only ±3 % spread within a `d`). It varies because the write's atom
   budget and near-site atom density vary with `d`, not because the cell is passing or failing.
2. **The numerator carries all the signal.** In r2's own Stage-0 table, **`sep` alone separates
   11/11 cells** (min PASS **0.849**, max FAIL **0.795**) — including d=2 and d=3, where the
   `sep/w` classifier explicitly fails. `sep/w` therefore classifies at d ≥ 4 only because
   `w(d)` happens to be flat there (0.30–0.36); it fails at d ≤ 3 because `w` drops to 0.215–0.29
   while the `sep` boundary does not move. *(POST-HOC-ON-EXISTING-DATA: computed from r2 §1.)*
3. **Why the ratio's critical value is so stable (1.36× over d=2…8).** The feasibility condition is
   `sep ≥ 2|a|max/κ_R(s)` and `κ_R = √(2 ln(D/(2α|c|s)) − 1)` — a **square root of a logarithm** of
   the width. Over the whole 11-cell width range (0.215–0.364) `κ_R` moves only **2.38 → 2.57**
   (7 %). A criterion that is nearly a pure threshold on `sep` will look like a sharp threshold in
   *any* monotone reparameterisation of `sep`, and `sep/w(d)` is one.
4. **Why forcing `sep/w = 4.90` destroys retrieval.** That intervention reached the ratio by
   *shrinking `s`* (0.30 → 0.184), which drops `a_U` 1.06 → 0.63 and drags `D` down with it
   (0.91 → 0.46, §1.5). The measured strict then equals the codebook fraction inside `a_U`:
   **predicted 0.625, measured 0.5859.** Sep/width was never the causal variable; `s` is, through
   (★) — and it enters with the *opposite* sign from the ratio's implication.
5. **Honest limit.** A pure reach account (`cells.py`, one fitted `D`) classifies **8/11**, missing
   exactly the three crowded high-K FAIL cells (d4K32, d5K64, d6K64), whose reach margins are
   comfortable. **Those three are a multi-well interference failure that criterion (U) does not
   model.** The complete cell-level law needs a two-factor form
   `strict ≈ min(1, a_U/|a|max) × P_addr(sep, s, σ_q)`; I derived and verified the first factor and
   only bounded the second (§4.2). **This is the main open question I am handing back.**

---

## 3. Q2 — RULING on the payload-dependent lifetime

**Mechanism (exact, shipped `V`).** The payload term is `0.5κ(q₂ − S(q_addr))²` with
`S(c_i) = a_i`. Any potential that returns the right value at the site necessarily puts a **hill of
height `0.5κ_eff·a_i²`** at the launch point, of address stiffness `κ_eff a_i²/s²`, against a well of
stiffness `A_i/s²`. In the shipped store `κ_eff = κ` is **constant while `A_i` decays**, so the ratio
`κa_i²/A_i` diverges: the site turns into a net address maximum. That is `mia-decay` §5's `r = −0.846`.

> ### ⭐ THE TRILEMMA (this is the ruling's backbone)
> With a quadratic payload channel, the `q₂(0) = 0` guard and a **fixed** read budget, you cannot
> have all three of
> **(i) exact value fidelity** · **(ii) amplitude-independent address hold** · **(iii) amplitude-independent read latency.**
> *Proof sketch.* (i) ⇒ the launch point is a hill of height `0.5κ_eff a_i²`; (ii) then requires
> `κ_eff ∝ A_i` (so that hill/well is A-independent); but the payload settling time is
> `τ_y = η/κ_eff ∝ 1/A_i`, which violates (iii). Dropping (i) is option (a); dropping (ii) is the
> shipped store; dropping (iii) is option (d). ∎
> **The payload-dependent lifetime is therefore not a bug in the potential — it is the corner of the
> trilemma the shipped store chose.**

### 3.1 Baseline reproduction and the `|c|` control (shipped store, 3 seeds)
| `A` | 1.0 → 0.15 | 0.10 | 0.08 | 0.06 | 0.051 |
|---|---|---|---|---|---|
| mean strict, ring ρ=2.287 | 1.000 | 0.961 | 0.896 | **0.500** | 0.427 |
| mean strict, ring ρ=2.012 | 1.000 | 0.969 | 0.901 | **0.734** | 0.500 |
| per-item at `A=0.06`, ρ=2.287 | — | — | — | `±0.143`:1.00 `±0.429`:1.00 `±0.714`:**0.00** `±1`:**0.00** | |
| per-item at `A=0.06`, ρ=2.012 | — | — | — | `±0.143`:1.00 `±0.429`:1.00 `±0.714`:**0.92/0.96** `±1`:0.00 | |

Ordering is strictly monotone in `a_i²` (**P2.2's ordering confirmed**), but my ring is harsher than
`mia-decay`'s controller-placed disk (they measure 0.886 at `A ≈ 0.06`). **The cause is `|c|`, and it
is a prediction of the reach criterion:** moving every site from ρ=2.287 to ρ=2.012 (−12 % in `|c|`)
lifts the `|a| = 0.714` items from 0.00 to 0.92–0.96 and the mean from 0.500 to 0.734.
**⇒ reconciliation item 5: retention numbers on this store are placement-dependent.**

### 3.2 The four candidates, ruled

| option | verdict | evidence |
|---|---|---|
| **(a) `payloads * amps`** | ⛔ **REFUTED — strictly worse than the defect it fixes** | The read returns `A·a_i`, so the value criterion dies at `A = 1 − tol/\|a_i\|`. Measured death amplitudes **0.90 / 0.80 / 0.70 / 0.30** for `\|a\| = 1 / 0.714 / 0.429 / 0.143` vs the formula's **0.900 / 0.860 / 0.767 / 0.300** — exact. Everything is dead by `A = 0.3`, and the payload dependence **inverts** (small-`\|a\|` items outlive large ones by **11×** in `τ`). |
| **(b) launch at `q₂ = S(q_addr)`** | ⛔ **REFUTED — it breaks the anti-decoration guard** | The **trivial substitute** (return `S` at the launch point, run *no dynamics at all*) already passes the shipped value criterion on **0.672** of queries (per-item 0.44–1.00, median err 0.057). It is also **amplitude-independent** (0.672 at `A` = 1.0, 0.5, 0.06): under (b) the value channel would stop responding to decay altogether, deleting the lifetimes dial's value-side signal. |
| **(c) numerically solve `leak` per item for a target half-life** | ◐ **honest-but-thin: makes the dial *calibrated*, not *physical*** | Retention halves before the floor for only **4 of 8** codewords (`\|a\| ≥ 0.714`); for the other four `R` never reaches 0.5 (0.688–1.000 at the floor), so a half-life can be "hit" only by making **self-eviction** the half-life — which is exactly the boolean TTL that `mia-decay` §3(a) showed is indistinguishable from CLU against an exact adversary. And for the solvable four, `A*` ∈ [0.070, 0.077] against a floor of 0.05: **the half-life lives in the last 8 % of the amplitude range**, so the retention *shape* is a near-step for every item. A referee probing the curve shape (not the half-life) sees it. |
| **(d) gated-stiffness payload channel** ⭐ | ✅ **RECOMMENDED (with the trilemma's price stated)** | see below |

### 3.3 The recommendation — option (d), and its exact price
```
V_pay = 0.5·κ·G(x)·(y − ā(x))²,     G(x) = g₀ + Σ_i m_i A_i e_i      (stiffness: floored)
                                     ā(x) = Σ_i m_i A_i a_i e_i / (ε + Σ_i m_i A_i e_i)
```
The stiffness gate makes hill/well `= κa_i²`, **amplitude-independent**; the *un*floored normaliser
keeps `ā(c_i) = a_i` exactly at every amplitude. ⚠ **My first implementation floored the normaliser
too** (`ā = A a/(g₀+A)`), which destroys the value at small `A` — that version measured *worse* than
baseline and **refuted my own P2.7**; the corrected version is below. (Non-shipped theorist's toy,
`GatedStore2` in `q2_round2.py`; shipped read, 3 seeds.)

| variant | `A` = 1.0 → 0.15 | 0.10 | 0.08 | 0.06 | 0.051 | payload dependence |
|---|---|---|---|---|---|---|
| shipped baseline | 1.000 | 0.961 | 0.896 | 0.500 | 0.427 | **strong** (`r ≈ −0.85` with `a²`) |
| **(d), `g₀ = 0.05` (= `amp_floor`)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **none** (value err 0.029 at the floor) |
| (d), `g₀ = 0.005` | 1.000 | 1.000 | 0.750 | 0.643 | 0.508 | reappears |
| (d), `g₀ = 0.005`, **read × 4** | 1.000 | 1.000 | **1.000** | **0.995** | 0.763 | **essentially none to `A=0.06`** |

**Read this as the trilemma, not as a free win.** `g₀` is the dial between the two surviving corners:
- `g₀ = amp_floor`: retention becomes **flat 1.000 until self-eviction** — payload-independent, but
  the on-site retention curve is now a **step**, i.e. exactly the TTL shape. *(The `R₅₀`-radius
  differentiator `mia-decay` §3(b) leads with should survive — the address well still shallows —
  but that must be re-measured before the claim is re-used.)*
- `g₀ ≪ amp_floor`: retention stays **graded** and payload-independent, but the read must get
  **longer as the item decays** (`τ_y = η/(κ(g₀+A))`); 4× read steps buys payload-independence down
  to `A = 0.06`. ⭐ **That is not a defect for this program — it is the compute-adaptive-read dial.**
  "A faded memory costs more integration steps to read" is an on-brand, *measurable*, physical
  statement that a timestamped row cannot make.

**Recommended to the engineer:** implement (d) with `g₀` **exposed as a config knob**, default
`g₀ = amp_floor`; re-measure `mia-decay` §1/§3(b)/§5 on it; and report the `g₀ ≪ amp_floor` +
long-read variant as the *graded* configuration. Do **not** implement (a) or (b).

---

## 4. Q3 — the two excursion arms (predictions posted early; three already scored here)

Full paste-able scorecard: `.claude/outputs/readout-channel-theory/Q3_PREDICTIONS.md`. Summary of the
derivation and of what I could already measure.

### 4.1 Arm (a), multi-channel payload — **real, not a cancelling free lunch** (the ⭐ question)
Reach depends on the **Euclidean** payload radius `r`, so splitting a value across `m` channels at
constant total precision gives `r(m) = zσ_a(K^{1/m} − 1)√m`, minimised at **`m* = ⌈log₂K⌉`** with
`r(m*) = zσ_a√(log₂K)`. At K=16 with the shipped per-axis spacing: `r` = **1.000 / 0.283 / 0.133** at
m = 1 / 2 / 4.

**Does read noise cancel the gain?** No — **provided the tolerance is an absolute latent-space
quantity.** `σ_a` is a property of the read-out, not of the code, so per-axis resolution is fixed and
splitting the value across axes costs no value information at all. At `m = 1`, halving `r` costs
**one** dimension of value resolution and buys `2^{1/μ}` of address packing (`μ = 0.350` at d=4),
i.e. a net **`2^{d_eff−1}` = 3.6×** per halving. **The excursion trades 1 dimension of value against
`d_eff` dimensions of address — that is the whole reason the arm works, and it is why it fails at
`d_eff ≤ 1`.**
**⚠ Fairness consequence, stated in advance:** w25's `pscale` probe scaled `payload_tol` *with* the
codebook. Under that convention the trade is invisible and the gain is genuinely free-and-fake. **Any
excursion experiment must hold `payload_tol` (or `σ_a`) absolutely fixed.**
*Already-measured corroboration:* on the designed-Gaussian surrogate with `tol` scaled the w25 way,
d=4 K=32 goes 0.763 (r=1) → **1.000** (r=0.5) → **0.844** (r=0.283) → 0.842 (r=0.25) — **non-monotone**,
because below `r ≈ 0.3` the read's own settling-error floor eats the shrinking tolerance. That is the
value-resolution cost appearing even in a noise-free harness.

### 4.2 Arm (b), the annealed read — **bounded, one K-rung, and now MEASURED**
Implemented with shipped pieces: atom widths `s_j → √(s_j² + s_a²)` during the **address phase only**,
native widths for the read phase (`fit_reach.py`, trained `V`, d=4 K=16, sep 0.9028).

| `s_eff` | `2s_eff/sep` | strict, native `s`=0.184 | strict, native `s`=0.301 | value err, anneal | value err, **widened through the read** |
|---|---|---|---|---|---|
| native | — | 0.5859 | 0.9473 | 1e−4 | — |
| 0.238 / 0.337 | 0.53 / 0.75 | **0.7480** | **0.9512** | 8.7e−5 | 1.1e−3 / 3.0e−3 |
| 0.310 / 0.392 | 0.69 / 0.87 | **0.8223** ⭐ | 0.9023 | 4.5e−5 | 3.2e−3 / 7.9e−3 |
| 0.395 / 0.462 | 0.88 / **1.02** | 0.7637 | **0.2520** | 5.2e−5 | 9.7e−3 / **1.7e−1** |
| 0.533 / 0.584 | 1.18 / 1.29 | 0.0625 | 0.0625 | 5.3e−1 | 3.1e−1 |

Three things are confirmed at once:
1. **The anneal works only where reach binds.** At the reach-limited width it buys **+0.236 strict**
   (0.586 → 0.822); at the already-reaching width it buys **+0.004** and then hurts.
2. **The ceiling is the merge condition `2s_eff = sep`, confirmed to one grid step** — the collapse
   happens between `2s_eff/sep` = 0.88 (fine) and **1.02** (0.252).
3. **The anneal must terminate before the read phase** (P3.6, registered, **confirmed**): keeping the
   width wide through the read costs **100× value error at `s_eff = 0.39` and 2000× at 0.46**
   (1.7e−1 > `tol` = 0.1). Predicted cross-talk bound `s_read ≤ 0.38`; measured between 0.395 (err
   9.7e−3, fine) and 0.462 (err 0.17, fails). Adiabaticity is a non-constraint (`|ds/dt| ≤ ω²s/η ≈ 10`
   vs a 20-time-unit phase): **only the endpoints matter, the schedule shape does not.**

⇒ Arm (b) is worth **one K-rung** and is *not* additive with arm (a) — both act on the same
inequality `r ≤ a_U(s_read)`.

### 4.3 The cheap arm nobody listed — lower the confinement
`a_U` depends on `α` and `|c|` only through the product `2α|c|`, and both are free design
parameters. Measured in the toy: halving `α` moves `a*` **1.050 → 1.146 (+9.1 %)**; moving the site
from `|c| = 0.9` to 0.3 moves it **1.050 → 1.150 (+9.5 %)**. One flag (`learned_confine`), no new
code, and it is an **independent causal test of (★)** on a parameter that is not the excursion.
⚠ Check coercivity (F5 Prop-10) and the value-blank control first.

---

## 5. PREREG scorecard (`PREREG.md` + Amendment A1)

| # | registered | measured | verdict |
|---|---|---|---|
| P1.1 | `D_fit ∈ [1.5, 6.0]` | **0.910** (w0.30), 0.459 (w0.15); R² ≥ 0.9938 | ✗ **band wrong (too high)**. The stated consequence ("the idealisation cannot be calibrated") is **contradicted**: the fit is excellent and (U) with the measured `D` scores 31/32 |
| P1.2 | (R) binds, (S)/(T) slack | (S) 0/32, (T) slack ≥10× | ✅ confirmed |
| P1.3 ⭐ | strict 0.88 ± 0.10 (w0.30), 0.56 ± 0.10 (w0.15), Δ 0.31 ± 0.10 | **0.947 / 0.586 / 0.361** | ✅ **all three inside the bands** |
| P1.4 | **pinned-slice** test ≥ 14/16 each | **10/16 and 13/16** | ✗ **falsified as registered.** The *unpinned* flow test — the physically correct one — scores **16/16 and 15/16**; the pinned version is the same 1.36× conservatism as `Ψ` |
| P1.5 | (S) fires ≤ 4/16 and ≤ 1/16 | **0/16 and 0/16** | ✅ confirmed, more strongly than registered |
| P1.6 | toy `a*/s ∈ [2.2, 3.2]` | **3.50** | ✗ outside band, **falsifier (>3.8) not triggered**; the truth sits between `κ_R` and `κ_stat`, and (U) gets it to 2.6 % |
| P1.7 | ratios 1.13 / 1.24 / 1.20 / 1.66 (±) | **1.091 / 1.102 / 1.095 / 1.476** | ◐ **all four right-signed, 3/4 below band**; (U) predicts 1.067 / 1.129 / 1.168 / 1.488 |
| P1.8 | ≥80 % of failures end nearer the centre | **100 %** (toy); trained `V`: 7/8 land on the ball-centre item | ✅ confirmed |
| P2.1 | reproduce `mia-decay` §5 within ±0.15 | ordering ✅; magnitudes harsher (0.500 vs 0.886 at `A`=0.06) | ✗ **magnitude miss, cause identified** (`|c|`; ρ=2.012 recovers 0.734) |
| P2.2 ⭐ | `A_crit` = 0.55 ± 0.20 / 0.10 ± 0.10 / <0.05 | **0.077 / 0.070 / <0.051** | ✗ the closed-form race criterion is **7× wrong at `a`=1**; ordering monotone in `a²` ✅ (falsifier not triggered) |
| P2.3 ⭐ | (a) dies at `A = 1 − tol/\|a\|` | **0.90 / 0.80 / 0.70 / 0.30** vs 0.900 / 0.860 / 0.767 / 0.300 | ✅ **confirmed to the digit** |
| P2.4 ⭐ | trivial substitute ≥ 0.75 (pass) / < 0.45 (falsify) | **0.672** | ◐ **inconclusive band**; the guard is two-thirds broken, which is enough to reject (b) |
| P2.5 | (c) solvable for 2/8 (≤3/8) | **4/8** | ◐ outside band, **falsifier (≥5/8) not triggered**; the ruling stands |
| P2.6 | shape differs ≥ 5× across the codebook | **4/8 never halve at all** (∞) | ✅ confirmed |
| P2.7 ⭐ | gated spring collapses the curves | **✗ as first implemented** (floored normaliser), **✅ after the declared correction**: 1.000 flat, payload-independent, value err 0.029 | ◐→✅ **my implementation was refuted, the mechanism confirmed** |
| A1.2a | surrogate reproduces ≥ 9/11 | **5/11** (every cell FAILs) | ✗ **falsified.** Cause found: the write adapts per-item widths (`corr(s_fit,\|a\|) = +0.82`), which a constant-`s` surrogate cannot |
| A1.2b | the 3 high-K cells < 0.9 in the surrogate | 0.763 / 0.686 / 0.562 | ✅ (weakly — everything failed) |
| P3.5 | anneal moves the wall one rung; ceiling at `2s_eff = sep` | **+0.236 strict** at the reach-limited width; collapse at `2s_eff/sep = 1.02` | ✅ confirmed (at d=4 K=16; the K=32 wall test is unrun) |
| P3.6 | annealing through the read costs err ≥ 0.15 | **0.17 / 0.31** at `s_eff` = 0.46 / 0.58 | ✅ confirmed |

**Score: 8 confirmed, 5 partial, 6 falsified.** The two most useful falsifications (P1.4-pinned,
A1.2a) both taught the same lesson: **conservative reductions of the read — pinning the payload, or
freezing the per-item width — systematically under-predict what the real system does.**

---

## 6. Limitations, stated before a referee does

1. **One seed per learned cell.** Both trained cells are seed 0. They reproduce r2's 3-seed values to
   4 dp, but the per-item classification (31/32) has no error bar. A 3-seed repeat would cost ~10 min.
2. **Criterion (U) is single-well.** It has no neighbour term, and the three crowded high-K FAIL cells
   are exactly where it fails (§2.5). Nothing here explains the K-dependence at fixed width.
3. **(U) was derived after seeing the toy's 1.36× offset.** Its agreement with the toy is a fit
   (2.4 %); its agreement with the trained `V` (31/32, zero free parameters) is out-of-sample and is
   the evidence that matters.
4. **The Q2 store is a ring**, not `mia-decay`'s controller-placed disk, and every site sits at the
   same `|c|`. That is a *worst case* for reach, and it is why my baseline is harsher than theirs.
5. **Option (d) is a theorist's toy** (`GatedStore2`), not shipped code, tested on one 8-item store
   with no controller, no eviction, no `φ`. Its `R₅₀`-radius behaviour — the differentiator
   `mia-decay` leads with — was **not** measured.
6. **No read noise anywhere.** All Q3 statements about `σ_a` are derivations; the harness is
   noise-free. The one measurement that bears on them (§4.1, the surrogate's non-monotone excursion
   sweep) uses the w25 relative-tolerance convention, which is the convention I am arguing against.
7. **`a_U` uses the fitted `D`, `s` of the *core*.** The trained `V`'s far field is fatter than a
   single Gaussian (measured launch gradient 0.233 vs 0.122 predicted at `|a|=1`, width 0.30), which
   is why the surrogate under-performs the trained `V`. The criterion works anyway because the
   *bifurcation* is controlled by the core, but the closed form is not a bound on the far field.

## Git footprint
**None.** No tracked file created, modified or deleted; `git status --short` empty before and after.
No branch, no commits. All artefacts under `.claude/outputs/readout-channel-theory/` and
`.claude/scratch/readout-channel-theory/`.

---

## Proposed handover updates (for the Hub)

1. **§1 / §6 — replace the reach bound.** w25's `s ≳ |a|max/κ`, `κ = O(3)` becomes: *"the read is
   captured iff `|a_i| < R₂`, the middle root of `(D R/s²)e^{−R²/2s²} = 2α(L−R)` with
   `L = √(|c_i|² + a_i²)` — verified to ±2.5 % on the shipped integrator and 31/32 items on the
   trained shipped `V` with zero free parameters. The failure is an **address** failure: the well's
   address stiffness is gated by `e^{−a_i²/2s²}`, so a far-payload item presents an exponentially
   soft well to the confinement's constant inward pull `2α|c_i|`, and the query is dragged to the
   ball centre. `basin ≡ strict` at every failing cell because they are the same event."*
2. **§1 — the quotable no-go.** *"Reach is logarithmically un-buyable: every closed form for the
   ceiling is `√(2 ln(·))`, so raising `κ` from 4 to 5 needs a 55× deeper well and 4 to 6 needs
   9200×. The only levers are the excursion, the width, the confinement `α` and the site radius
   `|c|`."*
3. **§7 / theory queue — the Q2 ruling (this closes `mia-decay` D3).** Fixes (a) and (b) are
   **refuted**; (c) is calibration, not physics, and is infeasible for 4/8 codewords; the recommended
   fix is the **gated-stiffness payload channel (d)**, measured **1.000 payload-independent retention
   at every amplitude to the floor**. Register the **trilemma** as a standing result: *value fidelity,
   amplitude-independent hold, amplitude-independent read latency — pick two.* The third corner is
   the **compute-adaptive-read dial**, which makes "a faded memory costs more integration steps"
   a physical claim the TTL substitute cannot make.
4. **§8 / negative registry — candidate new N (recommend registering).** *"The learned-capacity
   ceiling's reach half is a saddle-node bifurcation of the address dynamics, not a payload-channel
   effect: the spurious-minimum-on-the-payload-ray mechanism fires on 0/32 trained sites, while the
   address-hold criterion classifies 31/32. Consequently `sep/width` is a monotone reparameterisation
   of `sep`, and `sep` alone separates 11/11 Stage-0 cells (boundary in (0.795, 0.849)) where
   `sep/width` separates only d ≥ 4."*
5. **§3 config notes — two knobs are now load-bearing.** `learned_confine = 0.05` and the ball radius
   `R = 1.0` enter the capacity ceiling through `2α|c_i|`. Also: **`atom_init_width` moves the ceiling
   through two channels** — the trained `D ∝ s^{1.46}` (measured 0.910 at `s`=0.320, 0.459 at 0.200),
   so a width change is never "just an initialisation".
6. **§1 / new measured fact about the write.** The write **adapts the per-item well width to the
   payload excursion**: `corr(s_fit, |a_i|) = +0.82` (width 0.30), far-payload wells **1.31× wider**.
   A constant-width designed surrogate on the same geometry reproduces only 5/11 cells — **the
   write's per-item width adaptation is load-bearing**, which is a point *for* the learned arm.
7. **§5 provenance — new artefact set.** `.claude/outputs/readout-channel-theory/`: `PREREG.md`
   (+ Amendment A1), `Q3_PREDICTIONS.md`, this report; scratch in
   `.claude/scratch/readout-channel-theory/` (10 scripts + `dyn_toy.json`, `tr_d4K16_w0{30,15}.json`,
   `surrogate.json`, `q2_designed.json`, `q2_round2.json`). Base `ff85573`, JAX 0.9.0, no tracked code.
8. **Commission next, ranked.** (i) `learned_confine` 0.05 → 0.022 at d=4 K=16 width 0.15, 3 seeds —
   an independent causal test of (★) for ~1 h; (ii) arm (a) with **absolutely fixed** `payload_tol`,
   m = 1/2/4 at d=4, K = 16…256; (iii) the annealed read at d=4 **K=32** (the P3.5 wall test, unrun);
   (iv) option (d) in `AtomStorePotential` behind a `payload_gate` flag + a re-run of the
   `mia-decay` harness, including `R₅₀`.
