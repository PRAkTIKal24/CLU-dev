# PREREG — `bprime-theory` (C2W3, physics-theorist)

**Filed BEFORE any script in `.claude/scratch/bprime-theory/` was written or run.**
Protocol §5 pre-registration rule + task §3 (*"where you run a numerical check to confirm or refute a
derived quantity, write the predicted value and its derivation BEFORE you run it"*).

⚠ **Provenance honesty (declared up front).** One item below (P-T1a) was **discovered by inspection**
of an existing artifact (`.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json`, the per-cell
`byte_ledger.breakdown`) **before** this file was written; it is labelled **POST-HOC-DISCOVERY** and its
pre-registered part is only the *form of the corrected law* and its exactness in integer arithmetic.
Everything else here is a genuine forward prediction.

---

## P-T1a — the byte-floor law's true form ⚠ POST-HOC-DISCOVERY, pre-registered confirmation only

**Derivation.** `AtomDictionaryPotential` has exactly three inexact-array leaves — `centers (N_at × D)`,
`log_width (N_at)`, `amp (N_at)` — so `store.n_bytes() = 4·N_at·(D+2)` with `D = addr+payload+spectator`
the **store** dimension. `CluSystem.n_bytes() = store + 4·K·d`. The launder table stores keys
(`K × d`) and payloads (`K × m`), i.e. `4·K·(d+m)` — **`d+m`, NOT `D`**. Hence

> **`ratio = [A·(D+2) + d] / (d+m)`,  `A ≡ N_at/K` (atoms per live item).**

The shipped `chlu/experiments/memory_gym.py::byte_ratio_law` uses `D` in the denominator
(`A(D+2)/D + d/D`), which coincides with the correct law **iff `n_spectator = 0`**.

**Predictions.**
1. The corrected law reproduces `byte_ledger.ratio` in **28 of 28** C2W1 gym cells with **zero** error in
   exact integer arithmetic (`full_bytes·(d+m)·K == launder_bytes·[A(D+2)+d]` as integers).
2. The **shipped** closed form reproduces **24 of 28** and fails on exactly the **4 `n_spectator = 1`
   (manifold) cells**, by `[A(D+2)+d]·(1/(d+m) − 1/D) = 8.6667` (52.00 measured vs 43.33 published).
3. The bytes decompose exactly as `V_theta/4 = N_at(D+2)`, `codebook/4 = K·d`, `launder/4 = K(d+m)` in
   **all 28** cells.

## P-T1b — the architectural floor, and what the shell atom does to it (GENUINE)

**Derivation.** One atom group per item (what makes the write masked / C3-local) forces `A ≥ 1`, and
`A = 1` is the smallest store in which every item owns at least one **private** parameter. Then
`floor = [(D+2) + d]/(d+m)`. A shell atom adds one learnable radius per atom ⇒ `(D+3)` per atom.

**Predictions.** `floor(d=4, m=1, n_spec=0) = 11/5 = 2.20` exactly · `floor(n_spec=1) = 12/5 = 2.40` ·
shell `floor = 12/5 = 2.40` (`n_spec=0`) and `13/5 = 2.60` (`n_spec=1`) · the shell surcharge on the
measured manifold cell is **exactly ×9/8**: `52.00 → 58.40`, i.e. `+1/(D+2) = +12.5 %` on the atom term.

## P-T1c — the sharing exchange rate (GENUINE)

**Derivation.** Let `S` = items sharing each atom, so `N_at = K·A_tot/S` with `A_tot` atoms of
expressivity per item. `ratio = A_tot(D+2)/(S(d+m)) + d/(d+m)`. Setting `ratio = 1`:

> **`S* = (D+2)·A_tot/m`**, and with `A_tot = 1`, **`S* = (D+2)/m`**.

**Predictions.** At `d=4, m=1, n_spec=0`: matched bytes requires **≥ 7 items per atom** at `A_tot = 1`,
and **≥ 2387 items per atom** at the shipped anchor's `A_tot = 341`. Equivalently, the private-atom
fraction at ratio `r` obeys **`p ≤ [(d+m)r − d]/[(D+2)A_tot]`**; at `r = 1`, `A_tot = 341` ⇒
`p ≤ 4.19e-4` (**0.042 %** of an item's parameter mass may remain private at matched bytes).

## P-T2 — Prop D2a: the basin boundary and where `D` comes from (GENUINE)

**Derivation** (two equal-width Gaussian wells, separation `d₁₂`, amplitudes `A₁,A₂`, my own
`doctrine-repairs` §3-R1 corrected inradius): the static separatrix on the axis is displaced from the
midpoint by `δ = ln(A₁/A₂)/(d₁₂/s² − 4/d₁₂) → (s²/d₁₂)ln(A₁/A₂)` for `d₁₂ ≫ 2s`.

**Predictions.**
1. Equal depths ⇒ measured boundary at the midpoint to `< 1e-3·d₁₂` ⇒ **settle ≡ arg-min exactly**
   (`D = 0`), for every launch on the axis outside a `1e-3` window.
2. Unequal depths ⇒ measured boundary displacement matches `δ` within **10 %** for `s/sep ∈ [0.15,0.30]`
   and `|δ| ≤ 0.25 d₁₂`.
3. The disagreement rate under an isotropic query law `N(c_i, σ_q²I)` is the query mass between the
   bisector and the shifted boundary; predicted `D ≈ Φ((|δ| − d₁₂/2)/σ_q) − Φ(−(d₁₂/2 + |δ|)/σ_q)`,
   verified by Monte-Carlo within **20 % relative** (or `<0.01` absolute) at `σ_q = 0.15`.
4. ⭐ Dropping hypothesis (H3) (settled-point-only read) is the ONLY one of the four that leaves
   `D = 0` while creating a channel: the trajectory read's output differs between the two arms even
   when `q*` is identical. Predicted: `‖ψ_traj(full) − ψ_traj(launder)‖ > 0` on ≥ 90 % of queries at a
   separable equal-depth store where `D = 0` exactly.

## P-T3 — the ONE statement, made quantitative (GENUINE — the load-bearing check)

**Derivation.** `Fix(T_θ) = {(q,0) : ∇V_θ(q) = 0}` (Prop Q1.1) contains **no** `q₀`, `p₀`, `M`, `γ`, `dt`.
So every such parameter is a *transient* parameter: it changes only the approach, never the fixed point.
At finite `N` the surviving sensitivity is the un-decayed remnant of the contraction, i.e.

> **`‖∂z_N/∂ζ‖ ≍ ‖∂z_N/∂z_0‖ · O(1) = exp(−C)`, `C ≡ Σ_p N_p ln(1/ρ_p)`** (my `doctrine-repairs`
> convergence budget), `ρ_p` = spectral radius of the exact 2×2 damped-Verlet propagator, for every
> `ζ ∈ {q₀, p₀, log M, γ}`.

**Predictions.**
1. On a 1-D two-well toy at the shipped constants, `log₁₀‖∂q_N/∂q₀‖` versus `C/ln 10` has slope
   **−1.000 ± 0.02** over **≥ 6 decades** (sweeping `N` and `γ`), with the same slope (±0.05) for
   `∂q_N/∂log M` and `∂q_N/∂γ`.
2. The three transient sensitivities differ from each other only by an `O(1)` prefactor: predicted
   `ratio ∈ [1e-2, 1e2]` at every grid point.
3. `∂q*/∂ζ = 0` **exactly** (bitwise 0.0, not small) for the implicit/fixed-point path, for all four
   `ζ`, because the defining equation does not contain them.
4. **Consistency with the harness (declared post-hoc):** the shipped two-phase read
   (`γ₁=0.05, N₁=400; γ₂=0.02, N₂=800`) gives `C = 400·ln(1/0.97468) + 800·ln(1/0.98995) = 18.34`
   ⇒ `e^{−C} = 1.08e-8`, which must bracket the measured `2.654e-9` (`‖∂L/∂φ‖`, C2W1) and
   `8.73e-9` (`‖∂L/∂log M‖`, C2W2) **within one order of magnitude**.

## P-T5a — the slot manifold's dimension (GENUINE)

**Derivation.** For a fixed store, the whole read trajectory is a deterministic function of the launch
parameters. With the C2W2 particle head those are `(q₀, p₀, log M, γ) ∈ R^{3d+1}` (diagonal `M`, one
scalar friction). The slot vector lives in `R^{2dS}` but is the image of a `(3d+1)`-dimensional map.

**Predictions.** At `d = 4`: `rank(∂slots/∂launch) = 13 = 3d+1` **exactly** for every `S ≥ 2`
(`2dS = 8S ≥ 16 > 13`), with `σ_14/σ_1 < 1e-8`; at `S = 1` rank `= min(13, 8) = 8`. Under the **shipped**
read (`p₀ = 0`, `M`, `γ` fixed) rank `= d = 4` for every `S`. ⇒ **slot count buys no per-item capacity.**

## P-T5b — third-party (non-selected-item) attribution (GENUINE)

**Derivation.** A per-slot table launder selects rows by the query; deleting a row it did **not** select
changes its output by **exactly 0**. The CLU's slot content obeys `q̈ = −M⁻¹∇V` with `∇V` the sum over
**all** wells, so deleting a non-selected well `j` at distance `r` changes the early trajectory by
`Δq(t) ≈ (t²/2M)‖∇V_j(q₀)‖` with `‖∇V_j(q₀)‖ = (A_j/s²)·r·exp(−r²/2s²)`.

**Predictions.** `Δq(t)` matches `(t²/2)‖∇V_j(q₀)‖` within **20 %** for `t ≤ 20·dt` and `r/s ∈ [2,4]`;
the amplitude falls by **≥ 3 decades** across `r/s = 2 → 4` (predicted factor
`exp(−(16−4)/2)·(4/2) = 4.96e-3`, i.e. **2.3 decades** — registered range **2–4 decades**);
and it is **exactly 0** for the per-slot table launder on every cell.

---

**Scoring rule.** Every prediction above is scored ✅ / ◐ / ⛔ in `.claude/outputs/bprime-theory.md` §9.
A refuted prediction is a finding and is reported as such, never quietly dropped.
