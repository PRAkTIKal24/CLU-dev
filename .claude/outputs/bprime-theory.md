# bprime-theory — physics-theorist report

> ## ⛔ DATED ERRATUM BANNER (Hub, 2026-08-01, C2W4 second review — appended under the ratified C-3 precedent; the body below is NOT edited and stands as its own dated record)
> **T5.5's placement of our own rig on the `d/s` axis is WRONG BY 45–52×, measured by `bprime-c6` (C2W4).** T5.5 reads `d_safe_override = 0.58` — an admission gate, i.e. a *refusal radius* — as the achieved spacing. The shipped cell's **achieved** separation is **`sep = 1.3459`** ⇒ **`d/s = 4.34`** (atom ruler) / **3.72** (fitted ruler), measured coupling **`1.53e-2 ± 0.006`** — *not* `0.69–0.80`, and **already inside the designed-gate regime**. ⛔ **The "1089× span" is RETIRED**: the measured span across the admissibly-writable sweep is **10^2.72 = 525×**, and run-point→gate is **≈10×**. ⭐ The exponential law itself is **CONFIRMED on the learned store** (`R² = 0.9953`) and hands back the fitted well width **`s = 0.40`** (three estimates within 0.7 % — §9 item 2 discharged for this store). ⚠ `atom_init_width = 0.30` is a **1.33×-low proxy** for `s`; every `d/s` statement must name its ruler. **Direction favourable: §A9.5's kill and §A14.1's scoped kill both STRENGTHEN.** Affected sites in this file: the T5's-one-line-answer block below · §T5.5's table + caption · §6 C6's rationale · §8's fourth falsifier · the proposed-handover block. Registry: **N204**; charter: §A12 Add.4 (Advisor, in place).

Task + acceptance criterion: state B′'s theorem set (T1 byte floor · T2 Prop D2a · T3 the one statement) with **assumptions + domains of validity**, the T4 protocol caveats, and review the `allocate` spec **before** stage 2 builds it. Status: **done.**

> ## ⭐ T5's ONE-LINE ANSWER (`route3-stage2` reads this before it writes a design)
> **The only inter-slot coupling a per-slot matched-bytes table provably cannot express is THIRD-PARTY STORE ATTRIBUTION: a change in slot content caused by a stored item the query did *not* select.** A per-slot table's output changes by **exactly 0** when a non-selected row is deleted (it is never read); the CLU's slot content changes by `Δq(t) ≈ (t²/2M)‖∇V_j(q_t)‖ ≠ 0`, because `∇V` is a sum over **all** wells. Two riders, both decision-grade: **(a)** the coupling is suppressed as `exp(−½(d/s)²)`, so **the gate setting decides whether it is measurable at all** — at the *designed* gate `d_safe = 4.4 s` it is **7.0e-4** of the item's own gradient (invisible), but ⭐ **at the rig C2W3 actually runs (`d_safe_override = 0.58`, `atom_init_width = 0.30` ⇒ `d/s ≈ 1.9–2.0`) it is `0.69–0.80` — O(1), a `1089×` span from the designed gate, and already measurable today**, which is why stage 1 should probe it *before* stage 2 spends a worktree (request **C6**); the soft certificate (`d/s ≈ 2.9`) is the principled middle at **0.111**, and the span from the designed gate to there is **157× (2.20 decades)** at C2W2's measured price of `λ_min ÷ 2.2–6.0`. **(b)** ⛔ **slot count buys no per-item capacity** — for a fixed store the slot vector is the image of a `(3d+1)`-dimensional launch map (measured **rank = 13 = 3d+1** at `d=4` for every `S ≥ 2`; **rank = d = 4** under the shipped read), so a per-slot table can hold strictly *more* per-item slot content than the CLU can generate. **The claim can only be cross-item, never per-item.**

> ## ⚠ RECONCILIATION LIST — first-10-lines rule (protocol §5 corollary). Items 1–3 are the curator's Pass-C 19/21; item 4 is NEW and heaviest.
> | # | site | corrected wording / change | owner |
> |---|---|---|---|
> | **R-9** (handover #9) | doctrine **I-7** wherever it says *"the gauge is Newtonian-only"* (`controller-doctrine.md` §I-7 + §5 row 7; any registry line) | ⭐ verbatim replacement: **"Monitor #7's mass gauge `(M,V,p₀) → (λM,λV,λp₀)` is exact only under `kinetic_mode = newtonian_learned` (trajectory residual **2.52e-7**). Under `newtonian_identity` it is **not a gauge orbit at all** (residual **0.2505**) because `T = ½p²` ignores `M`, so `V` and `p₀` are rescaled with nothing to compensate them; N76 ('mass stores nothing') still holds there, trivially. Under `relativistic` it is broken at `O(1/c²)` (residual **0.0274** at `c = 1`). The test must compare the **whole trajectory**, never the endpoint (endpoint comparison passes vacuously once both runs settle: 9.1e-2 → 3.6e-3 by doubling `N` alone)."** | theorist (this file) + curator |
> | **E-1** (my erratum, C2W2 S6) | `controller-doctrine.md` §2 row 1, the γ band | **Retire both quoted intervals.** `γ∈[0.05,0.5]@N=400` was scored at `tol=1e-3`, `γ∈[0.02,0.9]@N=1500` at `tol≤1e-4`; the row claimed `1e-6`. Replacement (portable, harness-independent): **band ⇔ `C ≡ Σ_p N_p ln(1/ρ_p) ≥ ln(1/tol)`**, edges `γ_min = 1−exp(−2ln(1/tol)/N)`, `γ_max = 2κ/(2ln(1/tol)+κ)`, `κ = N dt² λ̄`; `ρ` = spectral radius of the exact 2×2 propagator (the published closed form exceeds 1 in the weakly-damped regime and then predicts divergence). Predicts 4/4 measured band edges. | curator |
> | **E-2** (my erratum, C2W2 S7) | `controller-doctrine.md` §2b row 8 (*"`sep/2` over-claims by up to 4.8 %"*) | **"4.8 % is that store's number, inside the domain `s/sep ∈ [0.15, 0.30]` and `λ_min > 0`. Outside it the proxy fails ≥ 29 % (below `s/sep = 0.15` capture is INERTIAL and no static proxy applies) or by 100 % (above 0.30, where the shallow basin is empty). The corrected proxy `r_i ≈ min_j[d_ij/2 + ln(A_i/A_j)/(d_ij/s² − 4/d_ij)]` is 14.55× more accurate **inside that domain only**."** ⛔ `sep/2` may never be quoted as a *certified* inradius. | curator |
> | ⭐ **R-BYTE** (**NEW, this task**) | ⛔ **`PREREG-Bprime.md` §7 · `memory-gym-v0` §2/§3.1/PREREG-B1 · `track2-admissibility` §2 · charter §A2.3 — every site saying the byte law is *"verified to 1e-9 in all 28 cells"*** | **It holds in 24 of 28.** The 4 `n_spectator = 1` (`manifold`) cells disagree by **+8.667 (20 %)**: measured **52.00×**, published closed form **43.33×**. Cause: `memory_gym.byte_ratio_law` divides by the **store** dim `D` where the launder row is `(d+m)` floats. ⭐ **The corrected law `ratio = [A(D+2) + d]/(d+m)` is exact in ALL 28 cells in integer arithmetic**, the floor **rises** (2.20× → 2.40× at `n_spec=1`; the shipped `floor_note` prints **2.00×** on those cells), and **B′'s reuse licence therefore SURVIVES — the theorem is not falsified, its published verification statement and the shipped formula are wrong in the conservative direction.** Code fix = request **C1** below. | curator + the `memory_gym.py` owner (⛔ **no C2W3 task owns that file — the Hub must assign it**) |

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form — echoed before the first result)
- **Dial / pillar:** **none — theory.** No dial, no leaderboard, no dividend, no performance number.
- **Laundering control:** n/a for a derivation. **Every proposition below carries a DOMAIN OF VALIDITY**; anything I could not bound is in §10 (NOT DERIVED) and is not filed as a theorem.
- **Falsifies:** task §4. **Verdicts:** T3 **does** hold as one statement (falsifier did NOT fire) · the byte-floor law **is** a theorem, with a corrected statement (falsifier **partially fired on the published wording, not on the theorem**) · `allocate` **can** be made byte-ledger-conserving by construction, under three named conditions · an inter-slot coupling a per-slot table cannot express **does** exist and is named (falsifier did NOT fire) — its magnitude is `exp(−½(d/s)²)`, invisible at the *designed* admission gate and **O(1) at the gate the gym actually runs**.
- **Does NOT falsify:** a theorem narrower than hoped · a caveat that costs a claim · the pseudo-Goldstone ruling surviving only as geometry.

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| repo | **read-only**, `main @ 6ff4c1d`, clean tree, **no worktree, zero tracked-code edits, no branch** |
| code | pure **numpy 2.4.1** / **scipy 1.17.0**, main venv `/Users/user/Desktop/CHLU/.venv` (py 3.11.13). **No repo module imported anywhere**; the integrator is the shipped damped velocity-Verlet reproduced line-for-line (3 substeps then `p ← (1−γ)p`), **float64** throughout. No JAX ⇒ the w6 worktree-JAX hazard does not apply. |
| artifacts read (read-only) | `.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json` (28 cells) · `chlu/experiments/memory_gym.py`, `chlu/eval/dividend.py`, `chlu/core/{clu_system,memory_potentials}.py`, `tests/test_memory_gym.py` (**read only, for the sizing convention**) |
| scripts (all re-runnable, each writes its own JSON) | `.claude/scratch/bprime-theory/`: `t1_bytes.py · t2_d2a.py · t2b_bands.py · t2c_dlaw.py · t3_transient.py · t3_refit.py · t3_pergamma.py · t3_twophase.py · t3_w19_retro.py · t4_blockers.py · t5_slots.py · t5b_exchange.py` + `common.py` (copied verbatim from `doctrine-repairs`) + `{t1_bytes,t2,t2b,t2c,t3,t3_refit,t3_pergamma,t3_twophase,t3_w19,t4,t5,t5b_exchange}_results.json` |
| PREREG | `.claude/outputs/bprime-theory/PREREG.md`, written **before any script existed**, with one item explicitly declared **POST-HOC-DISCOVERY** |
| kinetic mode | **Newtonian**, `M = I` unless a mass sweep is named; `T = 0`, `p₀ = 0`, no Langevin, no wake–sleep, no training anywhere |
| constants | `dt = 0.05`; two-phase read `γ₁ = 0.05, N₁ = 400` then `γ₂ = 0.02, N₂ = 800` (the harness's shipped read); `V = α‖q‖² − Σ A_i exp(−‖q−c_i‖²/2s²)`, `α = 0.05`, `s ∈ {0.20…0.90}`, `A ∈ {0.7…6}` |
| seeds | `default_rng(0…3)` as named per script; **every number here is a THEORY check, none is a paper number** |
| wall | ≈ 40 s total (t2 4.4 s · t2b 7.3 s · t3 ≈ 20 s · t4 7.2 s · t5 0.6 s · t1 instant) |

---

# T1 — ⭐ THE BYTE-FLOOR THEOREM

## T1.1 Statement (corrected)

> **Theorem T1 (per-item atom-group byte floor).** Let the store be an atom dictionary
> `V_θ(q) = α‖q‖² − Σ_{j=1..N_at} A_j exp(−‖q−c_j‖²/2s_j²)` with learnable `(c_j ∈ R^D, log s_j, amp_j)`, partitioned into **one atom group per item slot** (the masked / C3-local write), and let the matched-bytes launder be the table of `K` live rows `(key ∈ R^d, payload ∈ R^m)`. Then, with `A ≡ N_at/K` atoms per live item and `D = d + m + n_spectator` the store dimension,
> ### `ratio ≡ full_bytes / launder_bytes = [A·(D+2) + d] / (d+m)`
> exactly, **independent of `K`**, and — since one atom group per item forces `A ≥ 1` —
> ### `ratio ≥ [(D+2) + d]/(d+m)` = **2.20×** at `(d,m,n_spec) = (4,1,0)`, **2.40×** at `(4,1,1)`.

**Assumptions (each load-bearing).** (A1) every learnable leaf of the store is one of `centers / log_width / amp` — `4(D+2)` bytes per atom in float32 (verified in code: `AtomDictionaryPotential` has exactly these three inexact-array leaves; `confine`, `n_groups`, `axis_width_scale` are static/inert). (A2) the codebook of live addresses (`4Kd`) is counted in the CLU's ledger and the launder's key column is the same `4Kd` — so the address block **cancels to a constant `d/(d+m)`**, not to zero. (A3) the launder row is `(d+m)` floats, i.e. the table stores the *address* and the *payload*, not the spectator coordinates. (A4) atoms are private: no atom's parameters are read by two items' reads-with-writes.

**Domain of validity.** Exactly the shipped store family (atom dictionary, masked write, one group per slot). ⛔ It does **not** apply to an MLP/Hopfield store, to a shared/factored substrate, or to any store whose write is not group-masked. It **does** apply to the shell atom with `(D+2) → (D+3)`.

## T1.2 Verification (`t1_bytes.py`, exact integer/rational arithmetic on the 28 C2W1 cells)

| check | result |
|---|---|
| byte decomposition `V/4 = N_at(D+2)`, `code/4 = Kd`, `launder/4 = K(d+m)` | ✅ **28/28 exact (integers)** |
| corrected law reproduces `byte_ledger.ratio` | ✅ **28/28 exact (rationals, 0 ulp)** |
| **shipped** `byte_ratio_law` reproduces it | ⛔ **24/28**; the 4 `manifold` (`n_spec=1`) cells miss by **+8.6667** (52.00 measured vs 43.33 published) |
| shell-atom surcharge on `manifold/base` | ✅ `52.00 → 58.40` **exactly**; **`+1/(D+2) = +12.5 %` on the atom term**, `×1.123077` on the ratio — reproduces C2W2's measured pair digit-for-digit |
| floors | ✅ `2.20 / 2.40` (Gaussian, `n_spec = 0/1`), `2.40 / 2.60` (shell) |

⇒ **The falsifier "the 1e-9 agreement is a coincidence of the sizing convention" fires on the *published statement* and NOT on the theorem.** The agreement is structural — it is an accounting identity over the parameter leaves — but it was verified with a formula carrying a denominator bug that is invisible whenever `n_spectator = 0` (the only geometry the unit test covers: `tests/test_memory_gym.py:118-141` passes `n_spectator = 0` literally, and the end-to-end test is parametrised over `aggregate`/`recency` only). **`PREREG-Bprime.md` §7's reuse licence STANDS** — with the corrected sentence in R-BYTE above. `bprime-rivals` does **not** need to re-measure.

## T1.3 ⭐ What a shared substrate must buy, and at what rate

Let `A_tot` be an item's atom budget and `S` the number of items sharing each non-private atom, so `N_at = K·A_tot/S`:
### `ratio(S) = A_tot(D+2)/[S(d+m)] + d/(d+m)`, and `ratio = 1 ⟺ S* = A_tot(D+2)/m`.

| quantity | value at `d=4, m=1, n_spec=0` |
|---|---|
| items per atom needed for **matched bytes** at `A_tot = 1` | **`S* = (D+2)/m = 7`** |
| … at the shipped anchor `A_tot = 341` | **`S* = 2387`** |
| private-parameter fraction attainable at ratio `r` | **`p ≤ [(d+m)r − d]/[(D+2)A_tot]`** |
| … at `r = 1`, `A_tot = 341` | **`p ≤ 4.19e-4`** (0.042 % of an item's mass) |
| exchange rate | **`dp/dr = (d+m)/[(D+2)A_tot] = 2.10e-3` per unit of byte ratio** |

**The sharpest statement of what sharing must buy:** each shared atom's `(D+2)` learnable floats must carry the retrievable content of `S` items, i.e. **the per-atom information density must rise by exactly the sharing factor**; matched bytes needs `S ≥ (D+2)/m = 7` *even if every item is reduced to a single atom*. This is an identity, not an optimisation difficulty — no training budget, initialisation, or basis change moves it. ⛔ **A basis change is not a route to matched bytes: the shell atom raises the floor** (`(D+3)` per atom, `2.20 → 2.40`).

## T1.4 ⭐⭐ The deletion-vs-sharing frontier (§A9.9 formalised — they are the SAME trade)

> **Prop T1.4.** Byte-exact deletion (`AUC 0.5000 ± 0.0000`, byte-equal `3072/3072`) is available in `O(1)` **exactly on an item's private-atom fraction `p`**, because exactness *is* the statement that item `i`'s parameters form a block disjoint from every other item's — which is the same property (A4) that forces `A ≥ 1` and hence `ratio ≥ 2.20×`. On the shared fraction `1−p` there are only two options: **(i)** leave the shared atoms ⇒ the deletion is not byte-exact and carries a residual; **(ii)** re-fit them ⇒ every co-tenant item's bytes change (its own deletion certificate and its read both move), and the cost is a *write*, not a delete — which is precisely the retraining baseline exact unlearning is defined against. And `p` is pinned to the byte ratio by **`p ≤ [(d+m)r − d]/[(D+2)A_tot]`**.
>
> **Corollary A (the exchange rate).** `dp/dr = (d+m)/[(D+2)A_tot]`. At `r = 1` and `A_tot = 341`, **at most 0.042 % of an item's parameter mass may remain privately deletable.**
> **Corollary B (what 2.20× actually is).** Since a private atom is indivisible, `p ≥ 1/A_tot` forces `r ≥ [(D+2)+d]/(d+m)`. ⭐ **The 2.20× architectural floor is exactly the byte price of ONE privately-deletable atom per item.** It is not a property of the shipped budget; it is the cost of the smallest unit of exact deletion.

**Domain.** `A_tot` counted in atoms; `p` a parameter-mass fraction; float32. The proposition bounds *byte* exactness only — it says nothing about *behavioural* unlearning metrics, which is where the MUNKEY narrowing (arXiv:2603.15033, gap to retraining `0.56 ± 0.21`, ViT classifier not a sequence memory) applies.

**What §A9.9's deletion curve should plot** (this is the instruction the standing ruling needs): x-axis `p` (equivalently `r`, via the linear relation above); y-axis **two** series — *exactness on the private fraction* (byte-equality, `1.0` by construction) and *measured degradation on the shared fraction* (MIA-AUROC and read error after (i), plus wall-clock after (ii)). A single scalar "deletion still works" number is not admissible on a shared substrate.

---

# T2 — PROP D2a, STATED PROPERLY

> **Prop D2a.** Under **(H1)** separable wells (enforced spacing, every stored site a strict local minimum with `λ_min > 0`), **(H2)** an endpoint-trained landscape whose wells are *equal-depth* at the stored centres, **(H3)** a **settled-point-only** read `ŷ = ψ(q*)`, and **(H4)** a query law supported inside the certified basins, the settle map is **exactly** arg-min over the stored centres. Hence the dividend against a matched-key arg-min launder is **exactly 0** (`D = 0`, not "small"), and the shipped anchor's dividend of **0.0000 with `D = 0`** is structural.
>
> ⚠ **`D` (monitor #2) is the dividend's VARIANCE, not its magnitude.** `D` is the query mass lying between the arg-min (Voronoi) boundary and the settle's true basin boundary. It bounds *where* a dividend could live and says **nothing about its sign**: the gym's largest disagreement (`D = 0.931`) has the **worst** dividend (`−0.875`). ⛔ **`D` is never a progress signal.**

## T2.1 Which hypothesis, dropped, breaks the conclusion (`t2_d2a.py`, `t2b_bands.py`)

| dropped | mechanism (derived) | measured | consequence |
|---|---|---|---|
| **(H2) equal depth** | the axial separatrix shifts off the midpoint by `δ = ln(A_i/A_j)/(d_ij/s² − 4/d_ij)` (my `doctrine-repairs` R1 form) | boundary at `+0.0163 / +0.0295 / +0.0502 / +0.0555 / +0.0791` for `A₁/A₂ = 1.25/1.5/2/2.155/3` vs predicted `+0.0156/+0.0283/+0.0484/+0.0536/+0.0767` — **rel. err 3.1–4.8 %** | `D > 0` with a **derived** value: `D` = query mass in the shift shell; measured/predicted = **1.141** over 7 cells with `D_pred > 0.005` |
| **(H1) separability** | wells merge; sites stop being minima | `D = 0.0000` down to `d/s = 2.86`; **`0.0550` at `d/s = 2.29`; `1.0000` at `d/s = 1.71`** | the conclusion fails **and the store fails with it** — this is the gym's sub-shipped regime (`λ_min < 0`), where `D` is large *and* the dividend is worst |
| **(H4) query law** | ⛔ **REFUTED as a hypothesis in the equal-depth case** | `D = 0.00000` at **every** `sep/σ_q ∈ [2,10]` (n=4000/cell) | with equal depths the settle boundary **is** the Voronoi boundary by symmetry, so `D = 0` **independently of the query law**. (H4) is needed for *accuracy*, not for settle ≡ arg-min. My registered `Φ(−R/2)` was the bisector-**crossing** rate, which is not a disagreement — both labels flip together. |
| ⭐ **(H3) settled-point-only read** | the **only** drop that opens a channel without degrading the store | within-basin sd of `q*` = **6.6e-10** (piecewise constant); within-basin sd of `q_t`/`σ_q` = **0.99 (t=1) · 0.43 (t=10) · 0.042 (t=100) · <1e-3 (t=240) · 4.7e-9 (t=1200)**; `p_t` peaks at **1.88 σ_q at t≈10** | ⭐ **This is the map of where a dividend could structurally live.** A table returns **one** value per basin; the trajectory returns a continuum — but only for `t ≲ 240` steps, and the momentum channel carries **more** of it than position (charter §A8.1, measured). |

## T2.2 Why this is a proposition and not a theorem
The equal-depth case is a **symmetry argument** (exact: the boundary is the fixed-point set of the reflection exchanging the two wells; measured offset **2.9e-8** against a `2e-3` bar). The unequal-depth case is a **first-order linearisation**, valid for `|δ| ≤ 0.25 d_ij` and `s/sep ∈ [0.15, 0.30]` (T4.3). Outside that domain the boundary is not static at all (T4.1). ⇒ **filed as PROVEN in the symmetric case, EVIDENCED (3–5 % on a 2-D toy) in the asymmetric case, and INVALID below `s/sep = 0.15`.**

---

# T3 — ⭐⭐ THE ONE STATEMENT

> ## Theorem T3 (the transient/fixed-point dichotomy). **A settled-point read is untrainable end-to-end, in both directions, and for one reason.**
> For the shipped dissipative velocity-Verlet map `T_θ` with separable `H = T(p) + V_θ(q)` (Newtonian **or** relativistic kinetic), for every `γ ∈ (0,2)`, `dt > 0`, `M ≻ 0`:
> **`Fix(T_θ) = {(q,0) : ∇V_θ(q) = 0}`** (Prop Q1.1). The defining equation contains `θ` **only**. Therefore the parameters of the read split into two classes:
> - **fixed-point parameters** `θ` (the store): `∂q*/∂θ = −(Hess V_θ(q*))⁻¹ ∂_θ∇V_θ(q*)`, exact, with no `(γ, dt, M)` correction;
> - **transient parameters** `ζ ∈ {q₀, p₀, M, γ, dt, integrator}`: they appear in the *approach* and in nothing else, so
> ### `∂z*/∂ζ ≡ 0` — **exactly, for every transient parameter, in the implicit/fixed-point limit**,
> and at finite budget the surviving sensitivity is the un-decayed remnant of the contraction:
> ### `‖∂z_N/∂ζ‖ ≍ K_ζ · exp(−C)`, `C ≡ Σ_p N_p ln(1/ρ_p)`, `ρ_p` = spectral radius of the exact 2×2 damped-Verlet propagator,
> with `K_ζ = O(1)` for `ζ = q₀` (a pure Jacobian product) and `K_ζ = O(N)` for `ζ ∈ {M, γ}` (injected at every step).
>
> **Both C2 zeros are the same statement.** `q₀` is the read-in's output (`φ`); `M, γ` are the particle's attributes. The read-in direction (`‖∂L/∂φ‖ = 0.0` implicit / `2.654e-9` unroll / `6.421e-3` trajectory, ratio **2.42e6**) and the particle direction (mass gradient **exactly 0.0 bitwise 3/3 seeds**; trajectory/point ratios **1.7–2.9e5** mass, **2.6–4.9e5** friction) are two instances of *"a parameter that is not in `∇V_θ(q)=0`"*. ⛔ **The falsifier did NOT fire: the two zeros have the same mechanism and unify.**

**Assumptions.** `Hess V_θ(q*)` nonsingular (else the implicit function does not exist — the merger/decay bifurcations, T4.4); the read factors through `z*` alone; `γ ≠ 2`. **Domain.** Any `V_θ`; both kinetics; any `M ≻ 0`; `γ ∈ (0, ~1.5]` in practice (at `γ=1.99` the settle does not converge in 20 000 steps).

**Sharp form of "a.e."** `q*(q₀)` is *piecewise constant* — constant on each basin, with jump discontinuities on the codimension-1 separatrices. The gradient is 0 a.e.; on the null set it is a distribution, not a usable descent direction. This is C1's measured "staircase with a 1.7e7 cliff ratio", derived.

## T3.1 Verification (`t3_transient.py`, `t3_refit.py`)

| check | result |
|---|---|
| `Fix(T)` is `(γ, M)`-independent | settled `q*` spread over `γ∈{0.02,0.05,0.1,0.3} × M∈{0.5,1,2}` = **1.67e-15**; `max‖p*‖ = 1.5e-15`; `max‖∇V(q*)‖ = 1.3e-14` |
| `‖∂q_N/∂q₀‖ ∝ e^{−C}` | slope of `log₁₀|dq|` vs `C/ln10` = **−0.9941 over 143.9 decades** (25 cells, `γ×N` grid); **per-γ slopes −0.981 / −1.007**; prefactor `|dq|/e^{−C} ∈ [0.33, 8.3]` |
| `‖∂q_N/∂{log M, γ}‖ ∝ N e^{−C}` | raw per-γ slopes −0.90/−0.91; **after dividing by the derived `N` prefactor: −0.989 / −1.002 / −0.996 / −0.996** ⇒ the corrected law is exact to **±1 %** |
| the instrument's own floor | ⚠ 12 of 25 `{M,γ}` cells sit on a **float64 roundoff floor at ~1e-16** (accumulated injections) and were excluded before fitting — **the same failure mode as C1's `2.2e-12` FD floor** (`trainability-spike-theory` R-1). Reporting them as physics would have refuted a correct law. |
| the shipped read's own budget | `ρ(γ=0.05) = 0.974679`, `ρ(γ=0.02) = 0.989949` ⇒ **`C = 18.34`, `e^{−C} = 1.084e-8`** |
| ⭐ 1-D toy run at the **shipped two-phase schedule** vs the harness | toy `‖∂q_N/∂q₀‖ = 4.19e-9 … 1.79e-8` (harness `‖∂L/∂φ‖ = 2.654e-9`); toy ratios `logM/q₀ = 10.6–35.7` (harness **3.29**), `γ/q₀ = 8.0–128.6` (harness **24.6**) — ⭐ **all three harness point-arm gradients are reproduced within one order of magnitude by a two-well toy, and the friction > mass ordering with them** |

## T3.2 ⭐ The owed retro-explanation (§A2.2, outstanding since Addendum 1) — **DISCHARGED, with zero fitted parameters**

w19/N61 killed learned addressing: **4.2 % (1/24)** engineer, **2/18 strict · Adam 1/18 · energy-annealed 0/18** theorist, and the damped arm's loss **frozen at 7.9035443 to 7 significant digits across 4000 GD steps**. The C1 γ-scan (`clu-retrieval-demo` §6; `dt = 0.05`, **`steps = 1200`**, one launch point, one loss) is the law, measured before the law existed. Since every mode is underdamped at those `γ` (`λ_crit = γ²m/(2(2−γ)dt²) ≤ 0.04 ≪ λ`), `ρ = √(1−γ)` **exactly and landscape-independently**, so the prediction has **no free parameter but the `γ=0` anchor**:

| γ | 0.0 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|---|
| measured `‖∇_address‖` | 3.3e-1 | 9.1e-2 | 1.9e-2 | 1.7e-3 | 1.6e-5 | 5.0e-8 | 1.2e-7 |
| **predicted `3.3e-1·(1−γ)^{600}`** | 3.3e-1 | 9.9e-2 | 1.6e-2 | 7.9e-4 | 1.8e-6 | 1.4e-14 | 1.2e-28 |
| ratio meas/pred | 1.00 | 0.92 | 1.17 | 2.14 | 8.9 | — | — |

⭐ **Five decades reproduced within 0.92–8.9×**, and the last two points are **non-monotone in γ** (5.0e-8 → 1.2e-7) — i.e. they are that instrument's numerical floor, the same artefact T3.1 found in my own tangent recursion. ⇒ **w19's death is Theorem T3 evaluated at `C ≈ 12–31`.** At C1's 3000-step probe, `C = 76.9` ⇒ `|∂q_N/∂q₀| ~ 10^{−33.4}`, **26 orders below float32 `eps`** — gradient descent was optimising round-off. And `φ`'s whole history follows: every read of record has been a settled-point read, so **`φ` has never received a gradient larger than `e^{−C}`; the "weak φ" failures were not encoder-capacity failures, they were `C ≈ 18` failures.**

> ⛔ **The retrieval/learnability trade is definitional, not tunable** — C1 said it, T3 proves it: retrieval robustness *is* `∂(final)/∂q₀ → 0`, and `C` is the single knob that sets both. The only escape is to stop reading a fixed point: a trajectory read has `ρ = 1` on its `γ = 0` leg and `C` is not spent.

## T3.3 ⚠ Scope of the mass-gauge half (Prop F1)
**`M`'s dissolution is a statement about the READ, not about `V`.** The endpoint is `M`-independent (Prop Q1.1); the trajectory is not (`∇T` sets the time parametrisation). ⭐ **"Mass as selector" is live only under a trajectory read.** Independently, doctrine I-7's gauge orbit `(M,V,p₀) → (λM,λV,λp₀)` is exact **only under `newtonian_learned`** (R-9). Two different statements about `M`; both must be quoted with their scope.

---

# T4 — PROTOCOL CAVEATS, as numbered propositions with domains

**T4.1 ⛔ Below `s/sep ≈ 0.15` the basin boundary is INERTIAL; no static proxy is valid there.**
*Domain:* the shipped underdamped read (`γ ≲ 0.1`). *Measurement (`t4_blockers.py`, independent re-run):* at `s/sep = 0.10` (depth ratio 2) the deeper well's measured capture radius is **1.306** against a midpoint of **1.000** (+30.6 %), while the static correction predicts **+0.0144** — a 21× miss. ⭐ **New this task — the mechanism is confirmed by its `γ`-dependence:** the same store gives `r_deep = 1.308 / 1.306 / 1.002 / 1.000 / 1.000` at `γ = 0.02 / 0.05 / 0.20 / 0.50 / 0.90`. **The asymmetry is destroyed by damping ⇒ it is momentum-carried, not a property of `V`.** A static watershed is a `γ → ∞` object.

**T4.2 ⛔ `λ_min > 0` does NOT certify a nonempty basin.**
*Domain:* every store; the certified radius is **dynamical** (`γ`, barrier height), never spectral. *Measurement:* at `s/sep = 0.375` the shallow well is a genuine minimum with **`λ_min = +0.910`** and a measured capture radius of **0.000** — every trajectory escapes over the low barrier (reproduced here independently; also `0.000` at `λ_min = +0.388`). ⇒ **N3 needs a capture-radius leg** (SC-6: 32-direction bisection at ≥1 site per consolidation; refuse to certify any site whose measured basin is below `σ_q`).

**T4.3 ⛔ `sep/2` is NOT a certified inradius, and "Prop D1 is violated (1.5–7.44×)" is RETIRED.**
`D ≤ U` is a theorem **under a certified ball**; the gym computed `U` from `sep/2` on stores whose sites **were not minima** — all 7 cells with `D/U > 4` have `λ_min ∈ [−1.199, −0.372]`, so their certified radius is **0** and `U` was measured against a certificate that did not exist. Corrected proxy `r_i ≈ min_j[d_ij/2 + ln(A_i/A_j)/(d_ij/s² − 4/d_ij)]`, **14.55× more accurate inside its domain**; *domain (all four required):* `λ_min,i > 0` · **`s/sep ∈ [0.15, 0.30]`** · `|δ_ij| ≤ 0.25 d_ij` · a nonempty basin under the shipped `γ` (T4.2). Outside: ≥ 29 % error below 0.15 (T4.1), 100 % above 0.30. **State the domain every time.**

**T4.4 ⛔ `k*` governs `∂q_N/∂θ` ONLY where fixed-point sensitivity dominates the transient.**
`k*(ε) = ln(1/ε)/ln(1/ρ)`, `ρ = max(√(1−γ), 1−(2−γ)dt²λ_min/(2γm))`. *Domain:* parameters whose transient/fixed-point sensitivity ratio is `O(10)` or less — the settled well's own depth (ratio **64×** ⇒ truncation error **2.5e-5** at `k=270`, holds) but **not** a far well (ratio **27 396×** ⇒ error **0.448, flat in `k`**, wrong by 456×). ⚠ **In a `K`-item store the far-well parameters are exactly the interference gradients** ⇒ truncated BPTT preserves the on-well gradient and destroys the crowding gradient. Never quote `k*` without *"of `∂q_N/∂θ`, and only where the fixed-point sensitivity dominates the transient"*. Companion: `‖∂L/∂φ‖` flows only through the **head** (`h* = ln(1/ε)/ln(1/ρ)`); a tail-retention seam zeroes it **exactly**.

**T4.5 ⚠ `ε` is not the manifold-payload lifetime dial; `α` is its ceiling — and the pseudo-Goldstone ruling survives only as GEOMETRY.**
The shipped confinement floors the soft mode at `2α`, so the payload lifetime is capped at **`τ_max = Γ/2α = 4.0`** rather than growing as `1/ε`: the registered damped-mode floor `τ = 2/Γ = 5.0` time units is confirmed (measured **4.23–6.58**) and the above-knee slope is **0**, but the `1/ε` branch is **unreachable** because `2α` floors `λ_soft` 2.5× above the knee — and **lowering `α` breaks the write**. Every manifold-lifetime claim carries the **`2α` coercivity coupling**.
⭐ **The line, drawn precisely (this is the charter amendment):**
> *`λ = ε` is a theorem of the **single-atom shell geometry** in which it was specified (unit-tested to 5 %; a shell inside the shipped confinement has `λ_tan = 2α‖c‖/ρ ≈ 2α = 0.100`, measured **0.0994** — 0.6 %). It is **FALSE for a written site of a learned store**: a designed degeneracy does not survive superposition. Measured cause: a group's atom-centre spread is **1.19–1.95× the designed shell radius**, so the written site is not a common vacuum of its own atoms — its tilt vacuum residual is **0.140–0.343** against a random-orientation baseline of `1/dim = 0.167`, i.e. **at or worse than random**. Consequently the tilt **monotonically reduces** `λ_min` (`+0.0994 → −8.28`), sign-refuted on two independent implementations and every family.*
> ⇒ **Keep:** the geometric ruling (*be exactly flat architecturally, or comfortably massive; the intermediate band buys `1/b` conditioning and no extra storage*), which is a statement about **one** designed atom and remains verified. ⛔ **Drop:** every claim that a tilt `ε` instantiates it on a learned superposed store, and every use of `ε` as a lifetime dial.

**T4.6 ⚠ Doctrine I-7's gauge is `newtonian_learned`-ONLY** — see R-9 for the verbatim replacement (residual **2.52e-7** learned vs **0.2505** identity vs **0.0274** relativistic; endpoint comparison is vacuous, compare the whole trajectory).

**T4.7 ⚠ My own two C2W2 errata are CLOSED here** — the γ-band row's mixed tolerances (E-1: retire both intervals, quote the convergence budget `C`) and the `sep/2` 4.8 % figure's missing validity domain (E-2). Both corrected wordings are in the reconciliation table above and are ready for the curator to file verbatim.

---

# T5 — ⭐ REVIEW OF THE `allocate` SPEC (§A9.6), BEFORE STAGE 2 BUILDS IT

## T5.1 Is the action space byte-ledger-conserving **by construction**, or only by convention?

**It can be made structurally conserving, and it is not so automatically.** The store ledger is the identity `full_bytes = 4[N_at(D+2) + Kd]` (T1, exact 28/28) — **it contains no allocation variable**, so *routing existing content across slots cannot change it*. That is the structural half. The three ways allocation can still smuggle bytes:

| leak | why it is a leak | structural fix |
|---|---|---|
| **L1 — per-item stored allocation state** | "item `i` uses slots `{t₁,t₂}`" is `S·log₂T` bits **per item** ⇒ F2 state under B′'s own ledger rule, and it grows the launder-relative ratio | ⛔ **`allocate` must be a pure function of `(query, shared policy parameters)` — never a per-item table.** Then the ledger is invariant by construction, not by audit |
| **L2 — the policy's own parameters** | a learned allocation head is F1 parameters; unledgered, it is a free capacity increase | ledger the policy on **every arm including the launder** (the `phi_bytes` / `PhiMismatchError` precedent already exists and should be reused verbatim) |
| **L3 — the slot buffer** | `S × 2d` floats per query are F4 per-read transients, not state — but only if the **launder gets the same `S`** | declare `S` in the ledger as a *transient* line and give the table the same `S` |

⭐ **Runtime invariant for the engineer (cheap, blocking):** the **allocation-shuffle test** — apply any permutation of the allocation action and assert `full_bytes` is **bit-identical**, and that the tuple `(N_at, D, K, d, m)` is unchanged. `ledger drift` then cannot occur silently; it can only occur through L1/L2, both of which the shuffle test also catches (a per-item allocation table changes `full_bytes` under a permutation of *items*).

## T5.2 ⭐ Is *"the launder receives the same allocation budget"* a fair comparison, formally?

**It is necessary but it is the WEAK form of the null, and stage 2 should not rest on it.** Formally, a comparison is fair iff the two arms are compared at matched **(bytes, allocation freedom, policy class)**. Handing the table one sampled allocation matches the *budget* but not the *search*: the CLU's allocation is optimised by a learned policy while the table's is fixed. The honest null is the **supremum over the launder's allocation class**:

> **`null* = max_{a ∈ A} score(table with allocation a)`**, with `A` the same designed action space. Because `A` is small and discrete-or-simplex, this maximum is **computable**, not estimated: enumerate (or optimise on the family's own train split) and report `null*`, not `null(a_CLU)`.

Three riders: **(i)** the strongest cheap version is the **oracle-allocation table** (allocation chosen on the *test* split) — an upper bound on any table; a CLU that beats it is safe against any allocation policy anyone could have written. **(ii)** the **+0 B substitute audit must also be re-run per allocation** — C2W1's audit went **0-for-4** precisely because the substitutes were allowed to use free information the frozen launder was not. **(iii)** the §A6 precedent is binding: *"beat your own launder"* and *"be the best reader of your own bytes"* are different bars, and only the second is a result. ⇒ **Recommendation: `null*` (enumerated or oracle) is a MANDATORY column; `null(a_CLU)` alone is not sufficient for a headline.**

## T5.3 Does `allocate` stay inside §3.2 (designed action space, learned policy)?

**The instantiation is conformant if and only if three checks pass, and two of them are new:**
1. **The action space is closed and enumerable** (simplex over a fixed slot/dim set) and the **mechanism** — how a slot is written and read — is *not* learned. ✅ as specified in §A9.6.
2. ⚠ **The policy must be able to see the loss.** C2W2's new failure mode is not w20's ("free learning erases design") but its weaker, worse sibling: **design the objective cannot see is neither erased nor used** — the learned shell radius moved **0.500 → 0.501 in 300 steps**. ⇒ **report `‖∂L/∂(allocation logits)‖` at init alongside the liveness anchor**; if it is at the float32 noise floor, the verb is decorative and the cell is an under-powered grid, not a null.
3. ⭐ **A T3 corollary that constrains the build:** *an allocation policy trained through a **settled-point** read has **exactly zero** gradient* — the allocation is a transient parameter (it changes which part of the approach is read, and `Fix(T_θ)` contains no allocation variable). ⛔ **`allocate` is untrainable unless at least one allocated slot is not the endpoint**, and its gradient magnitude is bounded by `e^{−C}` for any slot at `t` with `Σ_{p>t} N_p ln(1/ρ_p)` remaining. **Allocation collapse to all-endpoint is therefore not just a monitored outcome — it is a gradient-flow *attractor*: the moment the policy reaches all-endpoint, its own gradient dies and it cannot leave.** Monitor it as declared (§A9.6, *"D2a from a new angle"*), but initialise **away** from the corner and report the trajectory of the allocation, not only its endpoint.

## T5.4 ⭐⭐ The weakest inter-slot coupling a per-slot table provably cannot express

> **Prop T5.4 (the shared-index bottleneck).** A per-slot matched-bytes table launder computes `ŷ = ψ(x, r_{i(x),1..S})` where `i(x)` is one query-dependent row selection. Its dependence on the **store** therefore factors through finitely many selected rows: for fixed `x`, `∂ŷ/∂(any non-selected row) = 0` **exactly**, and the store-attributable part of its output is **piecewise constant** on the selection cells. A CLU slot obeys `q̈ = −M⁻¹∇V` with `∇V` a sum over **all** wells ⇒ its store-attributable part is a **continuous, non-local** function of the query.

**⇒ The named candidates, in decreasing order of what they are worth to stage 2:**

1. ⭐⭐ **Third-party (non-selected-item) attribution — the kill-condition's only true escape.** *Measurement (cheapest possible, and it composes with stage 1's existing rig): delete a stored item the query did **not** select and measure the change in each slot's content. A per-slot table gives exactly 0. Report the curve vs `t`.* Measured on a toy: `|Δq(t)|` matches the ballistic prediction `(t²/2M)‖∇V_j(q₀)‖` to **0.61–0.73×** at `t = 10` steps (the residual is free-fall toward the item's own well; the registered 20 % bar is ⛔ **missed** — the *scaling* holds, the prefactor needs the free-fall correction), and its amplitude falls **2.60 decades** across `r/s = 2 → 4` (registered 2–4 decades ✅).
2. ⭐ **Within-basin continuous store-attribution.** A table returns one store-attributable value per basin (measured within-basin sd of `q*` = **6.6e-10**); the trajectory returns a continuum (**0.99 → 0.043 σ_q over `t = 1 → 100`**). Any target whose store-attributable component varies *inside* a basin is inexpressible by a per-slot table, and the table's error is lower-bounded by the within-cell variance of that target. ⚠ **Caveat: the launder may read `x` directly** (the +0 B substitutes do) — so the separation must be stated on the **store-attributable** component, which is exactly what stage 1's attribution curve already measures.
3. **Itinerary / deflection order.** Which wells the path passes, and in what order, depends on launch momentum and well *depth*, not on distance ranking; a distance-ranked table cannot reproduce it without `K²` rows. ⚠ **Honest counter: a `k`-NN table with `k = 2` expresses most 2-item versions of this** (the gym's own `knn2` substitute), so it is only a real separation for `≥3` interacting items.
4. ⛔ **NOT a candidate — richer per-item slot content.** Slots are *dynamically redundant*: for a fixed store the slot vector is the image of a `(3d+1)`-dimensional launch map. Measured rank (`t5_slots.py`, `d=4`, central differences, float64): **13 = 3d+1 at every `S ∈ {2,4,10,20}`** (slot dims 16 → 160), **8 at `S=1`** (slot-dim-limited), and **4 = d under the shipped read** (`p₀=0`, `M`, `γ` fixed). ⇒ **From `S = 2` onward, extra slots add zero per-item degrees of freedom.** *(The `≤ 3d+1` half is a chain-rule theorem; what the run adds is that the rank is **full** — there is no hidden gauge among `(q₀,p₀,M,γ)` at fixed `V`, so the manifold really is `3d+1`-dimensional, not less.)*

## T5.5 ⭐ The exchange rate that prices Route 3 (`t5b_exchange.json`)

Third-party coupling relative to the item's own gradient at `σ_q = 0.15`, `s = 0.35`:

| `d/s` | **1.9–2.0** | 2.5 | **2.9** | 3.0 | 3.5 | 4.0 | **4.4** | 5.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|
| `‖∇V_j‖ / own` | **0.80 / 0.69** | 0.281 | **0.111** | 0.085 | 0.020 | 3.4e-3 | **7.0e-4** | 4.8e-5 | 2.3e-7 |
| | ⭐ **where the gym actually runs** (`d_safe_override = 0.58`, `s ≈ 0.30`) — also the merger band, `λ_min` collapsing | | **soft-certificate witnesses** (deficit ≈33 % of `sep`) | | | | **designed gate `d_safe = 4.4 s`** (C1 §C5-A1) | | |

> ⭐⭐ **The design-deciding sentences for `route3-stage2`.**
> **(1)** Under the **designed** admission gate the only coupling a per-slot table cannot express sits at the `7e-4` level — below every noise floor the harness has. **Under the rig C2W3 actually runs it is O(1)** (`d/s ≈ 1.9–2.0`, coupling `0.28–0.69`), because the gym declared `d_safe_override = 0.58` (deliberately out of band, `memory-gym-v0` §2). ⇒ ⭐ **the §A9.5-satisfying measurement is available on the EXISTING rig, today, at zero build cost** — run request **C6** in stage 1 rather than discovering it in stage 2.
> **(2)** The span from the designed gate to the soft-certificate region is **157× (2.20 decades)** of coupling at C2W2's measured price of `λ_min ÷ 2.2–6.0` (hence `≈1/λ_min` worse implicit conditioning and truncation depth). ⇒ any stage 2 that *restores* the designed gate to buy conditioning is buying away its own headline claim, and must say so.
> ⚠ Two caveats that bound both sentences: `B = 0.33`'s outer edge was located with the **broken `sep/2` ruler** (T4.3) — re-locate it with the corrected radius before freezing (`doctrine-repairs` OQ-A, ≈10 min, still owed); and **`s` for a learned multi-atom well is an open modelling question** (§9 item 2), so `d/s ≈ 1.9–2.0` uses `atom_init_width` as a proxy for the fitted well width and is a bracket, not a measurement.

---

# 6. ⭐ CODE CHANGES MY THEOREMS IMPLY (numbered, for the Hub to route — I edit nothing)

| # | change | file / site | priority |
|---|---|---|---|
| **C1** | ⛔ **Fix `byte_ratio_law`'s denominator: `(addr_dim + payload_dim)`, not `dim`.** `return A*(dim+2)/(d+m) + d/(d+m)`. It is wrong **only** when `n_spectator > 0`, where it *understates* both the ratio (43.33 vs 52.00) and the floor (prints **2.00×**, true **2.40×**). | `chlu/experiments/memory_gym.py:321-339` and the `floor_note` at `:553` | **P0** — B′ prints this law |
| **C2** | **Add the missing regression test**: the current `test_byte_ratio_law_matches_the_measured_ledger` passes `n_spectator = 0` literally and the end-to-end test is parametrised over `aggregate`/`recency` only, so **no test exercises a spectator dim**. Add `manifold` (or `n_spectator=1`) to both. | `tests/test_memory_gym.py:118-141, 294-311` | **P0** |
| **C3** | **Assert the ledger identity structurally** rather than checking a float ratio: `full == 4[N_at(D+2) + Kd]` and `launder == 4K(d+m)` as integers. This is also T5.1's ledger-drift guard, so `allocate` inherits it free. | `chlu/eval/dividend.py::byte_account` (append-only surface) | P1 |
| **C4** | `allocate` must be a **pure function of `(query, shared policy params)`**; **no per-item allocation state**. Ship the **allocation-shuffle test** (permute the action ⇒ `full_bytes` bit-identical) as a blocking test. | `chlu/core/allocate.py` (new, `route3-stage2`) | **P0 for stage 2** |
| **C5** | Report **`‖∂L/∂(allocation logits)‖` at init** next to the liveness anchor; and initialise the allocation **away** from the all-endpoint corner (T5.3: it is a gradient-flow attractor, not merely a monitored outcome). | `chlu/core/allocate.py` + PREREG | **P0 for stage 2** |
| **C6** | ⭐ Add the **third-party attribution probe**: delete a **non-selected** item, measure per-slot Δ, report the curve vs `t`. It is the §A9.5-satisfying measurement, it is ~free on the rig that already exists, and **the gym's own `d_safe_override = 0.58` puts the coupling at O(1) (`0.69–0.80` of the item's own gradient) rather than the designed gate's `7e-4`** — so it is measurable **now**, in stage 1, and does not need stage 2 to exist. | `chlu/eval/attribution.py` (`route3-stage1-plus-2x2`) | ⭐ **P0 — do it in stage 1** |
| **C7** | **Slot-placement directive:** put slots in `t ∈ [1, 240]` — beyond `t ≈ 240` steps the within-basin (query) variance is `<1e-3 σ_q` and by `t = 1200` it is `4.7e-9 σ_q`. **Record `p_t`, not only `q_t`**: momentum peaks at **1.88 σ_q at `t ≈ 10`**, ~4× the position channel. | stage-1/2 slot grids | P1 |
| **C8** | Monitor #2: `r_i := 0` and **INAPPLICABLE** wherever `λ_min,i ≤ 0`; use the corrected inradius **inside its domain only**; add the **capture-radius** leg (T4.2, SC-6). *(Re-affirmation of `doctrine-repairs` S5 — still open.)* | `chlu/core/monitors.py`, `chlu/core/clu_system.py:906` (the silent `max(λ,1e-9)` clamp) | P1 |

---

# 7. PREREG scorecard (`.claude/outputs/bprime-theory/PREREG.md`)

| # | registered | measured | verdict |
|---|---|---|---|
| P-T1a | corrected law exact 28/28; shipped 24/28 failing the 4 `n_spec=1` cells by 8.6667; decomposition exact 28/28 | **28/28 · 24/28 · +8.6667 on exactly those 4 · 28/28** | ✅ (⚠ discovery declared POST-HOC) |
| P-T1b | floors 2.20/2.40 (gauss), 2.40/2.60 (shell); shell surcharge `×9/8` on the atom term, `52.00 → 58.40` | **exactly those**, surcharge `+1/(D+2) = 12.5 %` | ✅ **exact** |
| P-T1c | `S* = 7` at `A_tot=1`, **2387** at 341; `p ≤ 4.19e-4` at `r=1` | **7 · 2387 · 4.19e-4**; `dp/dr = 2.10e-3` | ✅ **exact** |
| P-T2.1 | equal depths ⇒ boundary within `1e-3·sep` of the midpoint | **2.92e-8** (bar 2e-3) | ✅✅ 5 orders inside |
| P-T2.2 | `δ` within 10 % for `s/sep ∈ [0.15,0.30]` | **3.1–4.8 %** over 5 depth ratios | ✅ |
| P-T2.3 | `D ≈ Φ((|δ|−d/2)/σ) − …` within 20 % | ⛔ **REFUTED as registered** — my formula was the *bisector-crossing* rate (`D = 0.00000` at every `sep/σ_q ∈ [2,10]`, equal depths). ⭐ **Corrected law verified instead:** `D` = query mass **between the two boundaries**, measured/predicted **1.141** over 7 cells | ⛔→✅ **refutation is the finding** |
| P-T2.4 | dropping (H3) opens a channel with `D = 0` | within-basin sd: `q*` **6.6e-10** vs `q_t` **0.99 → 0.043 σ_q** | ✅ (restated as a variance, not a per-query rate) |
| P-T3.1 | slope `−1.000 ± 0.02` for `q₀` over ≥6 decades; same ±0.05 for `logM, γ` | `q₀` **−0.981 / −1.007** per γ (**143.9 decades** pooled) ✅; `logM, γ` **−0.90/−0.91 raw** ⛔ → **−0.989…−1.002 after the derived `N` prefactor** | ◐ **half confirmed, half corrected** |
| P-T3.2 | cross-parameter ratios in `[1e-2, 1e2]` | **[1.2, 479]** above the floor | ⛔ **MISS ×4.8** — the excess is exactly the `O(N)` injection prefactor |
| P-T3.3 | `∂q*/∂ζ = 0` exactly | `q*` spread **1.67e-15** over the `γ×M` grid; `‖p*‖ ≤ 1.5e-15` | ✅ |
| P-T3.4 | `C = 18.34 ⇒ e^{−C} = 1.08e-8` brackets 2.654e-9 and 8.73e-9 within one order | **1.084e-8**; toy at the shipped schedule gives **4.19e-9…1.79e-8** and reproduces the harness's `γ/q₀` ratio (**24.6** measured, **8–129** toy) | ✅✅ |
| P-T5a | rank `= 13 = 3d+1` for `S ≥ 2`; `8` at `S=1`; `4` under the shipped read | **13 / 13 / 13 / 13 · 8 · 4** | ✅ (the `≤` half is a theorem; the **full-rank** half is what was measured) |
| P-T5b | ballistic within 20 % for `t ≤ 20` steps; 2–4 decades over `r/s = 2→4`; table exactly 0 | ⛔ **0.61–0.73× at `t=10`, 0.17–0.34× at `t=20`** (free-fall correction missing); ✅ **2.60 decades**; ✅ table 0 by construction | ◐ **scaling confirmed, prefactor refuted** |

**Score: 8 ✅ (4 exact) · 2 ◐ · 3 ⛔.** All three refutations produced a corrected law that was then verified, and one of them (P-T2.3) corrects a *mechanism* that would otherwise have mis-specified monitor #2.

---

# 8. Falsifiers, adjudicated (task §4)

- ⛔ **"T3 does not hold as one statement"** — **did NOT fire.** Both zeros are `Fix(T_θ) = {(q,0): ∇V_θ(q)=0}` containing no transient parameter. One theorem, two corollaries, one shared finite-`N` law `e^{−C}`, and it retro-predicts w19's γ-scan over 5 decades with no free parameter.
- ⛔ **"The byte-floor theorem is not a theorem"** — **fired on the published wording, not on the theorem.** The law is an accounting identity (28/28 exact); the *shipped formula* and the sentence *"1e-9 in all 28 cells"* are wrong for `n_spectator > 0`, in the direction that **understates** the floor. **`PREREG-Bprime.md` §7's reuse licence stands; `bprime-rivals` must NOT re-measure**, but every quoting site takes R-BYTE's replacement sentence. *(Reported in the first 10 lines, same day, per the task's instruction.)*
- ⛔ **"`allocate` cannot be made byte-ledger-conserving"** — **did NOT fire**, conditional on C4: no per-item allocation state, policy parameters ledgered on every arm, and the allocation-shuffle test as a blocking check.
- ⛔ **"No inter-slot coupling exists that a per-slot table cannot express"** — **did NOT fire**: third-party store attribution is provably inexpressible by a row-selecting table and measurably nonzero for the CLU. ⚠ **But it is exponentially suppressed by the shipped admission gate (7.0e-4 at `d_safe = 4.4 s`)**, so §A9.5's kill-condition is satisfiable **only inside the soft certificate**. That is a narrowing, not a kill — and it is far cheaper to learn here than from a build.

---

# 9. ⛔ DECLARED **NOT DERIVED** — never to be quoted as settled

1. **The prefactor of the harness's own particle gradients.** My toy law says `‖∂q_N/∂{M,γ}‖ ≍ N e^{−C}`; the harness's point-arm mass gradient (**8.73e-9**) sits at `≈ e^{−C}`, i.e. `N ≈ 1200×` **below** the toy's law, while `‖∂L/∂φ‖` matches the `O(1)` law. Either the shipped per-query mass enters far fewer steps than assumed, or the batch/ψ scaling absorbs it. **The structural claim (exactly 0 implicit, exponentially small unrolled) is unaffected; the prefactor is open.** A 10-minute engineer check would settle it (see OQ-1).
2. **Everything geometric here is 1-D/2-D/4-D designed wells with a single width `s`.** ⚠ **Naming `s` for a learned multi-atom well is an unsolved modelling question and it gates the transfer of every domain statement** (`s/sep`, `d/s`, `s_max/σ_q` are the three ratios that control T4.1–T4.3, and **none of them is measured on the shipped learned `V_θ`**). Carried unchanged from `doctrine-repairs` OQ-C.
3. **T5.4's coupling list is not proven exhaustive.** I proved that *third-party attribution* is inexpressible and that *per-item slot richness* is not a route; I did **not** prove that no other coupling class exists.
4. **The 2.60-decade third-party scaling is a two-well toy at `p₀ = 0`**; with a live particle head (`p₀ ≠ 0`) the path can be steered toward third parties and the suppression could be materially weaker. **Untested, and it is the most promising thing stage 2 could try.**
5. **`B = 0.33`'s outer edge is still located by the broken `sep/2` ruler** (T4.3). Not re-derived here.
6. **No claim about a *learned* store developing approximately-symmetric flat directions.** Untested; would not rely on it.

# 10. Open questions / follow-ups / risks

- **OQ-1 (cheap, decisive).** Measure `‖∂L/∂log M‖` for the shipped point read at `N ∈ {200, 400, 800, 1200}` with everything else fixed. Theorem T3 predicts `∝ N e^{−C(N)}`; item 1 above predicts it may be `∝ e^{−C(N)}`. Either outcome pins the prefactor and is a one-flag experiment on `exp_phi_particle --part grad`.
- **OQ-2.** The third-party probe (C6) with a **live particle head**: does a learned `p₀` steer the path toward non-selected wells and beat the `exp(−½(d/s)²)` suppression? This is the only mechanism I can see that would make §A9.5 satisfiable *without* the soft certificate.
- **OQ-3.** `memory_gym.py` has **no owner in C2W3** and now carries a P0 fix (C1) plus a missing test (C2). The Hub must assign it — B′'s headline byte law is printed from that file.
- **Risk.** All checks Newtonian (`M = I` unless swept), `p₀ = 0`, `T = 0`, no training, single-geometry. Constants are kinetic-mode-specific even where the structure is not.

## Git footprint
**None.** No tracked code touched; repo read-only at `main @ 6ff4c1d`, clean tree, no branch, no worktree. All artifacts under `.claude/scratch/bprime-theory/` and `.claude/outputs/bprime-theory/`.

---

## Proposed handover updates (for the Hub)

**§1 (physics addendum) — three additions.**
- ⭐ **Theorem T3 (the transient/fixed-point dichotomy).** `Fix(T_θ) = {(q,0) : ∇V_θ(q)=0}` contains **no** `q₀, p₀, M, γ, dt`. ⇒ a settled-point read has **exactly zero** gradient to its read-in **and** to the particle's attributes; at finite budget `‖∂z_N/∂ζ‖ ≍ K_ζ e^{−C}`, `C = Σ_p N_p ln(1/ρ_p)`, with `K_ζ = O(1)` for `q₀` and `O(N)` for `{M, γ}`. Verified: slope **−1.00 ± 0.01** per γ over 143.9 decades; `C = 18.34` for the shipped read ⇒ `e^{−C} = 1.08e-8`, bracketing the measured `2.654e-9` (φ) and `8.73e-9` (mass). ⭐ **It retro-explains w19/N61 with zero fitted parameters** — the C1 γ-scan is `3.3e-1·(1−γ)^{600}` to within `0.92–8.9×` over five decades, and its last two points are that instrument's floor (non-monotone in γ). At C1's 3000-step probe the address gradient is `10^{−33.4}`, **26 orders below float32 `eps`**.
- ⭐ **The byte law is `ratio = [A(D+2) + d]/(d+m)`** — exact in **all 28** C2W1 cells in integer arithmetic. `2.20×` is **the byte price of one privately-deletable atom per item**; matched bytes needs `S* = (D+2)A_tot/m` items per atom (**7** at `A_tot=1`, **2387** at the shipped anchor) and caps the byte-exactly-deletable fraction at `p ≤ [(d+m)r − d]/[(D+2)A_tot]` (**0.042 %** at `r=1`, `A_tot=341`). **Compression and byte-exact deletion are the same trade, with that exchange rate.**
- **The basin boundary is dynamical.** New evidence that T4.1's inertial boundary is momentum-carried: the same store's capture radius goes `1.308 → 1.306 → 1.002 → 1.000` as `γ = 0.02 → 0.05 → 0.20 → 0.50`. A static watershed is the `γ → ∞` limit.

**§7 (known issues / live) — one new entry, one replacement.**
- ⛔ **NEW: `byte_ratio_law` is wrong for `n_spectator > 0`** (`memory_gym.py:321-339`): it divides by the store dim where the launder row is `(d+m)` floats. Measured `52.00×` vs published `43.33×` on the 4 `manifold` cells; the `floor_note` prints `2.00×` where the true floor is `2.40×`. **Untested** (both byte tests use `n_spectator = 0`). Every site saying *"verified to 1e-9 in all 28 cells"* becomes **"24 of 28; the corrected law is exact in 28 of 28"**. **The floor and B′'s reuse licence are unaffected — the error is conservative.**
- ⚠ **Replace the I-7 line** with R-9's verbatim wording (`newtonian_learned`-only; identity residual `0.2505`; relativistic `0.0274`; compare the trajectory, never the endpoint).

**§8/§10 (record).**
- ⭐ **B′'s theorem set is filed**: T1 (byte floor + the sharing/deletion frontier with an exchange rate), T2 (D2a with its four hypotheses and the drop map — **(H3) is the only drop that opens a channel without degrading the store**), T3 (**one statement, both directions**, with the `e^{−C}` law and the w19 retro-explanation **discharged**), T4 (seven numbered caveats, each with a domain), T5 (the `allocate` review).
- ⭐⭐ **For Route 3 and the §A9.5 kill-condition:** the only coupling a per-slot table cannot express is **third-party store attribution**, suppressed as `exp(−½(d/s)²)`. **The gate setting decides everything:** `7.0e-4` of the item's own gradient at the *designed* `d_safe = 4.4 s`, `0.111` in the soft-certificate region, and **`0.69–0.80` at the gym's own declared `d_safe_override = 0.58`** — a `1089×` span. ⇒ **the kill-condition is measurable on the EXISTING rig in stage 1** (code request C6, P0), and any stage 2 that restores the designed gate to buy `λ_min` is buying away its own claim. Independently, **slot count buys no per-item capacity** (measured rank `3d+1` for every `S ≥ 2`), so the claim can only ever be cross-item.
- ⚠ **Ownership gap:** `chlu/experiments/memory_gym.py` has no C2W3 owner and now carries a **P0** fix.
