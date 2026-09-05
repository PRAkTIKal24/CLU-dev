# ssb-shell-atoms — experiment-engineer report

Task + acceptance criterion: **Route 2 (designed degeneracy)** — build the charter §A4.1 shell atom with
its blocking `r=0` regression gate and the §A4.2 pseudo-Goldstone tilt dial, then race it against Route 1
on the shared C2W2 card (3 families × 3 seeds, 4 launders + the +0 B substitute + byte ledger + monitors
+ `gate_admissible`).
Status: **done.** r=0 gate **GREEN** (bit-identical `V`, `∇V`, `Hess V` and written store). Race card
emitted: **90 cells, 0 clearing 0 by 2 SE, 16 `le_zero_vote`, 12 `abstain`, 2 `under_powered_grid`** ⇒
Route 2's half of the C2W2 gate is **≤ 0 on every family**.
⭐ **Headline (task §6's second falsifier FIRED — reported as the headline, not buried): the tilt does NOT
produce `λ_min ≈ ε > 0` at any registered `ε`. It monotonically REDUCES `λ_min`, on two independent
implementations, on every family.** The mechanism is measured, not guessed (§4.2).

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5 corollary, first-10-lines rule)
> **R1 — the FREEZE commit's `store_potential_factory` hook is INSUFFICIENT, and I had to work around it
> in my own module (task §7 asked me to report exactly this).** The hook swaps the *potential*; it does
> not let a new store family supply its own **update mask**. `clu_system._write_item` hardcodes
> `atom_write_mask_fn(...)`, which masks exactly 3 leaves; a shell store has 5, so an unmasked
> `radius_raw` lets a write for item *i* move **every** item's radius — C3 locality gone. Asserted in
> `tests/test_shell_atoms.py::test_shipped_mask_would_leak_the_radius`. **Proposed one-line fix (owner:
> `traj-write-objective` / whoever owns `clu_system.py` next): `CluSystemConfig.store_write_mask_factory`,
> resolved the same way, defaulting to `atom_write_mask_fn`.** Until then Route 2 runs via a declared
> context-manager patch (§2.1).
> **R2 — `overload` at the gym's base atom budget is 0/18 admissible INCLUDING the shipped Gaussian
> control** (`λ_min = −0.43 … −1.20` on `gauss` itself). That is the gym's own §3.1 write-failure regime,
> *predating Route 2*, and it means **no Route-1/Route-2 comparison on `overload/base` is interpretable**.
> Both routes should quote `overload` at the `load1x_shipped` 478× anchor (where my coverage is **3/3 =
> 100 %**). *(Owner: Hub, at race-card integration.)*
> **R3 — theory §Q2.4's `cond(H) = 7.69` for the shipped ring reappears on the learned atom store**:
> measured hierarchy `λ_max/λ_min` = **6.83** (envelope arm) / **7.73** (depth arm) at `ε = 0`. Possibly
> the same number twice by coincidence, possibly not; the theorist should say. *(Owner: theorist.)*
> **R4 — PREREG §P5's admissibility rule was relaxed BEFORE the science run** (λ_min only, dropping the
> `‖∇V‖ < 0.1` leg), because the smoke run measured `grad_norm_max = 0.1417` on the **control**, i.e. the
> registered rule would have excluded 100 % of every arm. Both coverages are emitted
> (`gate_admissible` and `flags.prereg_strict_admissible`). *(Owner: Hub — accept or reject the relaxation.)*

---

## ⭐ DIAL DECLARATION (echoed, protocol §7 / C2 form)
- **Dial / pillar:** **pillar 1 (manifold-valued latents) via pillar-(d) storage.** KPI = the **dividend**
  on the shared race card. No accuracy number below appears without its dividend; no dividend without its
  bytes.
- **Laundering control:** the four harness-native ones (settle-deleted launder · same-keys null · blank
  store · trajectory launder) **plus the family's +0 B substitute**, all on every cell, all inherited
  bit-for-bit from `exp_memory_gym.run_cell` (§2.1). ⚠ The `manifold` family's `echo` substitute is
  **1.0000 by construction**, so its substitute margin is **≈ −1.00 on every arm including the control** —
  stated here rather than left to read as a clean result.
- **Falsifies:** §6 of the task. **Does NOT falsify:** a ≤0 manifold dividend with the r=0 gate passed and
  the dial demonstrably live; `matched=False` bytes; needing a nonzero tilt.
- **Trajectory launder:** ⚠ **NOT RUN** — no ψ in these cells can see the address block (they use the
  shipped `settled_point_psi`), so the launder has nothing to fire on. Declared, never reported as a pass.

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/ssb-shell-atoms` @ **`647c54e`**, 4 commits off **`1ce02e4`** (`traj-write-objective`'s RACE+SEAM FREEZE) |
| ⚠ base caveat | the measured cells were **run on `main @ 233fd9e`** and the branch was rebased onto `1ce02e4` afterwards; the race card was then re-emitted from the saved records via `emit_race_from_json` (no write re-run). The freeze's two seams are default-off and asserted bit-identical, and my `gauss` control reproduces the gym exactly (§3.1), so this is a no-op — but it is disclosed. |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0**, equinox 0.13.4, **no worktree `uv sync`**; worktree `../CHLU-ssb` |
| commands | `python -u -m chlu.experiments.exp_ssb_shell` (72 cells, 31.7 min) · `… --families overload --gym-arm load1x_shipped --family-label overload_shipped` (18 cells, 9.0 min) · `emit_race_from_json` · `plot_from_json` |
| artifacts | `.claude/outputs/ssb-shell-atoms/{PREREG.md, exp_ssb_shell_metrics.json, escalation/exp_ssb_shell_metrics.json, race_card_route2_combined.json, race_cells_route2.json, compute_bench.json, exp_ssb_shell.png, run_full.log, pytest_full.log}` |
| **seeds** | **{0,1,2} on every race cell** (90 cells). Sample sd (`ddof=1`), `SE = sd/√3`. The ε-dial sweep and the low-confinement extra are **seed 0 only** and are labelled as such |
| geometry | gym defaults, **unmodified**: `manifold` `d=4,m=1,n_spectator=1 ⇒ dim=6`, K=6, 192 atoms; `overload` `dim=5`, 18 offers/198 atoms (base) or 2046 atoms (`load1x_shipped`); `aggregate` `dim=5`, K=6, 192 atoms |
| store | `DesignFreedomPotential(rung="free_mlp", family="atoms")` with `.learned` swapped for `ShellAtomDictionaryPotential`; `confine α = 0.05`, `atom_init_width 0.3`, `atom_depth_init 1e-4`, `atom_init_scale 1.0` |
| **Route-2 flags** | `radius_scale ∈ {None(gauss), 0.0(shell_r0), 1.0}` · **`r_init = R_DESIGNED = 0.5`** · `freeze_radius` on the `shell_fixed`/tilt arms · **`ε ∈ {0, 1e−3, 1e−2, 1e−1, 1, 10}`** (pre-registered; 5 non-zero values over **4 decades**, `ε = 10` the destructive anchor) · `tilt_weight ∈ {envelope, depth}` · tilt rank 1, per group, `tilt_dir` learned, init on the last coordinate axis |
| write | shipped: **300 steps**, Adam(W) `lr 3e-3`, `wd 1e-4`, `n_perturb 32`, `σ_addr 0.25`, `σ_pay 0.6`, `margin 0.15`, `barrier 0.2`, `barrier_pairs "nn"`, masked/C3-local (via `shell_write_mask_fn`) |
| read | shipped: `dt 0.05`, `γ_address 0.05`, `γ_read 0.02`, 400 + 800 steps, `traj_stride 8`, `kinetic_mode newtonian_learned`, `psi = settled_point_psi` |
| langevin / lyapunov / wake–sleep | **N/A** — `T=0`, `p₀=0`, no Langevin; the write is `train_memory_landscape`'s static objective |
| τ probe | displacement `δ = 0.1` along the softest eigenvector, **4000** phase-2 steps at `γ_read = 0.02`, `dt = 0.05` (⇒ `Γ = γ/dt = 0.4`), decay rate fitted on the monotone upper envelope over `[0.92 c₀, 0.03 c₀]`, median over ≤4 sites |
| ⚠ **declared deviations** | (a) admissibility = λ_min only (R4); (b) the **low-confinement `α = 0.005` dial cell** was added **after** the smoke run showed the shipped store sits above the registered knee — the *prediction* is unchanged, only the operating point (§4.3); (c) the naive charter-literal tilt was replaced by a bounded **angular** form **before** any science cell, because it was measured to destabilise (§2.2) |

---

## 1. ⛔ Admissible-cell coverage, FIRST (gate ruling (i), the Head's named failure mode)

| family | cells | admissible | **coverage** | escalated? | why the exclusions |
|---|---|---|---|---|---|
| `manifold` | 36 | 12 | **33.3 %** | no (family has >0) | `λ_min < 0` at ≥1 recorded site; concentrated on the `ε ≥ 1` tilt arms (0/3 each) |
| `aggregate` | 18 | 5 | **27.8 %** | no | ″; both `ε` arms 0/3 |
| `overload` (base budget) | 18 | **0** | **0.0 %** | ⭐ **YES** | ⛔ **including the `gauss` control** (`λ_min` −0.43…−1.20) — the gym's own sub-shipped write-failure regime (R2) |
| ⭐ `overload_shipped` (the escalation) | 18 | 18 | **100.0 %** | (is the escalation) | none; `λ_min` +1.70…+3.24, `final_write_loss` 1.6e−4…9.8e−4 |

The one bounded budget escalation permitted by ruling (i) was spent on `overload`: re-run at the gym's
`load1x_shipped` arm (2046 atoms, the **478×** anchor, the only gym cell where the store actually works).
It took coverage from **0 % to 100 %**, so `overload` does **not** abstain — it votes, at the shipped
budget. Every excluded cell is listed with its reason in `race_card_route2_combined.json → coverage[*].reasons`.

## 2. What I did

### 2.1 The rig — identical to Route 1's, by construction
`chlu/experiments/exp_ssb_shell.py` runs **`exp_memory_gym.run_cell` unmodified**; a context manager
patches exactly two symbols for the duration of a cell: `exp_memory_gym.build_system` (swap the store's
`.learned` subtree for shell atoms — the blank/empty-store control included, or it is not a control) and
`chlu.core.clu_system.atom_write_mask_fn` (extend C3 locality over the shell's extra leaves; it degrades
to shipped behaviour on a Gaussian store, so the `gauss` arm is untouched). **No file owned by another
agent was edited** (§7 git footprint). Consequence: every launder, control, monitor, byte-ledger and
scorer path is bit-for-bit the one Route 1 uses — *"same rig or it is not a race."* Verified in §3.1.

### 2.2 The shell atom, and the tilt I had to redesign
`chlu/core/shell_atoms.py`: `V = α‖q‖² − Σ_j A_j exp(−(‖q−c_j‖−r_j)²/(2s_j²+1e−9))`, learnable
`r_j = radius_scale·softplus(radius_raw_j)`, plus a rank-1 per-group learned tilt.

⛔ **The charter-literal tilt `(ε/2)Σ_j w_j (û·(q−c_j))²` was MEASURED to fail and was replaced before any
science cell ran.** It grows quadratically away from the atom, so at a real written site (where a group's
atoms are scattered) its Hessian is dominated by `(û·δ)²·Hess g ≈ −ε(û·δ)²/2s²`. Measured on a real
store: `λ_min` went **+0.080 → −0.42 → −5.2 → −53** across `ε = 0 → 0.1 → 1 → 10` — the ridge write's
saddle failure in a new costume. The shipped form is the **bounded angular** one,

    V_tilt = (ε/2) Σ_j w_j · r_j² · (û_{o(j)} · n̂_j)²,   n̂_j = (q−c_j)/‖q−c_j‖,

which still gives **`Hess V_tilt = ε ûûᵀ` exactly** at an on-shell vacuum (unit-tested at
`ε ∈ {1e−3,1e−2,1e−1,1}`, `λ = ε` to 5 %) but cannot blow up away from it. A **second implementation**
(`tilt_weight="depth"`, q-independent weights, no envelope curvature at all) ships alongside so §4.2's
finding is not an artefact of one weighting. Both are raced.

## 3. Results

### 3.1 ⛔ THE r=0 REGRESSION GATE — GREEN, and it needed two layers

**Unit level** (`tests/test_shell_atoms.py`, 22 tests): with `r_j ≡ 0`, `V`, `∇V` and `Hess V` are
**bit-identical** to `AtomDictionaryPotential` — `max|Δ| == 0.0` exactly, in **float32 and float64**,
including at `q == c`, and both on the static short-circuit path and on the **arithmetic** path forced
with `softplus(−1000) == 0.0` exactly. A 60-step masked write leaves `centers`/`log_width`/`amp`
bit-identical and the loss history identical.

⚠ **The arithmetic alone is NOT sufficient, and this is a finding worth carrying.** Writing the shell
displacement as `d2 − 2ρr + r²` (not `(ρ−r)²`) makes the `r=0` reduction exact **in eager mode**; but
under `eqx.filter_jit` **XLA reassociates the enlarged expression and the PARAMETER gradient shifts by
~1 ULP**, which Adam amplifies to **9.5e−7 in `centers` after 60 steps** while the forward pass stays
bit-identical. Since `radius_scale` is a *static* field, the `r=0` arm short-circuits to the shipped code
path, the emitted HLO is identical, and the written-store leg of the gate is exact.

**System level** — `shell_r0` vs `gauss`, 3 seeds, full config, all four families:

| family / metric | `gauss` | `shell_r0` | Δ | `λ_min_min` (both) | final write loss (both) |
|---|---|---|---|---|---|
| `manifold` `r2` | −0.1802 ± 0.1708 | **−0.1802 ± 0.1708** | **0.0000** | −0.0131 | 0.1380 |
| `overload` `decode` | 0.2593 | **0.2593** | **0.0000** | −0.8529 | 0.2325 |
| `aggregate` `neg_mae` | −0.5261 | **−0.5261** | **0.0000** | −0.2034 | 0.2407 |
| `overload_shipped` `decode` | 0.9722 ± 0.0139 | **0.9722 ± 0.0139** | **0.0000** | **+3.2394** | **0.00016** |

⭐ **And the `gauss` control reproduces `memory-gym-v0` exactly** — `manifold r2 −0.1802 ± 0.1708`
(gym: −0.1802 ± 0.1708), `overload decode 0.2593` (gym: 0.2593 ± 0.0668), `aggregate neg_mae −0.5261`
(gym: −0.5261 ± 0.0863), and at the shipped anchor `0.9722 ± 0.0139` with `λ_min +3.24` and loss `0.0002`
(gym: identical). **The rig is bit-faithful to the gym, so every Route-2 difference below is the shell.**

⚠ **The gate is not free:** `shell_r0` is behaviourally identical and costs **+14.26 % bytes**
(478.20× → 546.40× at the shipped anchor) because `radius_raw` is stored regardless.

### 3.2 The race card (the gate's Route-2 half)

`dividend = full − settle-deleted launder`; 3 seeds; `SE = sd/√3`; grade by `chlu.eval.race.score_card`.

| family | arm | adm | **dividend ± SE** | clears 2 SE | **+0 B substitute margin** | byte ratio | grade |
|---|---|---|---|---|---|---|---|
| `manifold` | `gauss` | 2/3 | −0.2608 ± 0.2609 | no | **−1.2608** | 52.00× | `le_zero_vote` |
| `manifold` | `shell_r0` | 2/3 | −0.2608 ± 0.2609 | no | −1.2608 | 58.40× | `le_zero_vote` |
| `manifold` | `shell` | 1/3 | −0.0167 | no | −1.0167 | 58.40× | `le_zero_vote` |
| `manifold` | **`shell_fixed`** | 1/3 | **−0.0072** | no | −1.0072 | 58.40× | `le_zero_vote` |
| `manifold` | `shell_tilt_0.001` | 1/3 | −0.0083 | no | −1.0083 | 59.60× | `le_zero_vote` |
| `manifold` | `shell_tilt_0.01` | 2/3 | −0.0192 ± 0.0159 | no | −1.0192 | 59.60× | `le_zero_vote` |
| `manifold` | `shell_tilt_0.1` | 1/3 | **+0.0005** | no | −0.9995 | 59.60× | `le_zero_vote` |
| `manifold` | `shell_tilt_1` / `_10` | **0/3** | — | — | — | 59.60× | `abstain` |
| `manifold` | `shell_tiltd_0.01` | 2/3 | −0.0245 ± 0.0288 | no | −1.0245 | 59.60× | `le_zero_vote` |
| `manifold` | `shell_tiltd_1` / `_10` | **0/3** | — | — | — | 59.60× | `abstain` |
| `aggregate` | `gauss` / `shell_r0` | 1/3 | −0.1863 | no | −0.4676 | 54.56 / 62.24× | `under_powered_grid` |
| `aggregate` | `shell` | 2/3 | −0.0837 ± 0.0042 | no | −0.3320 | 62.24× | `le_zero_vote` |
| `aggregate` | `shell_fixed` | 1/3 | −0.0705 | no | −0.2858 | 62.24× | `le_zero_vote` |
| `aggregate` | `shell_tilt_0.01` / `_1` | **0/3** | — | — | — | 63.44× | `abstain` |
| `overload` (base) | **all 6 arms** | **0/3** | — | — | — | 17.11–20.49× | `abstain` (R2) |
| ⭐ `overload_shipped` | `gauss` / `shell_r0` | 3/3 | **−0.0278 ± 0.0139** | no | **+0.2639** | 478.20 / 546.40× | `le_zero_vote` |
| ⭐ `overload_shipped` | `shell_fixed` | 3/3 | **−0.0556 ± 0.0367** | no | **+0.2361** | 546.40× | `le_zero_vote` |
| ⭐ `overload_shipped` | `shell` | 3/3 | −0.0833 ± 0.0417 | no | +0.2083 | 546.40× | `le_zero_vote` |
| ⭐ `overload_shipped` | `shell_tilt_0.01` / `_1` | 3/3 | −0.0833 ± 0.0636 / ± 0.0481 | no | +0.2083 | 547.40× | `le_zero_vote` |

**Gate arithmetic (Route 2): `any_family_clears = False`. 0 cleared · 0 weak-proceed · 16 `le_zero_vote`
· 12 `abstain` · 2 `under_powered_grid`.** ⛔ **No cell is quotable as a byte-matched dividend** — the
minimum ratio anywhere is **17.11×**, and the shell raises the architectural floor by
**+1/(dim+2)** (§3.5).

⭐ **The one real signal, stated with its caveat.** The shell **halves-to-eliminates the manifold
family's negative dividend**: `gauss −0.2608 → shell_fixed −0.0072` (all-cell means: **−0.1802 →
−0.0108**, i.e. `Δ = +0.169`), and the per-seed variance collapses (`gauss` r2 = `[0.0001, −0.0192,
−0.5216]` vs `shell_fixed` `[−0.0072, −0.0234, −0.0017]`, **SE 0.171 → 0.0065**). ⚠ But `r2 ≈ 0` is what
a store that returns **the launch point's zero** scores, and the +0 B `echo` substitute scores
**1.0000**, so the honest reading is *"the shell stops the settle from actively destroying the spectator
coordinate"* — **not** *"the shell expresses a manifold"*. The spectral evidence (§3.4) says the same
thing: the softest mode's spectator participation goes 0.022 → 0.032, i.e. it does **not** become the
designed coordinate.

### 3.3 ⭐⭐ THE HEADLINE — §A4.2's tilt mechanism is REFUTED on a real store

Task §6: *"the tilt does not produce `λ_min ≈ ε > 0` at any registered `ε` ⇒ §A4.2's mechanism is refuted
on a real store; that is a **major finding** and must be reported as the headline."* **It fired.**

The ε sweep on **one written store** (so nothing but the dial changes; seed 0, `manifold`):

| ε | `λ_min` median — **envelope** weight | `λ_min` median — **depth** weight | §A4.2 predicts | `λ_max` (envelope) | soft-mode participation on `û` |
|---|---|---|---|---|---|
| 0 | **+0.0994** | **+0.3291** | (baseline) | 3.899 | — |
| 1e−3 | +0.0987 | +0.3291 | 0.1004 / 0.3301 | 3.899 | 0.087 |
| 1e−2 | +0.0884 | +0.3289 | 0.1094 / 0.3391 | 3.910 | 0.076 |
| 1e−1 | +0.0556 | +0.3178 | 0.1994 / 0.4291 | 4.029 | 0.068 |
| 1 | **−0.4741** | +0.1939 | 1.0994 / 1.3291 | 5.183 | 0.098 |
| 10 | **−8.2846** | **−1.1980** | 10.0994 / 10.3291 | 12.301 | 0.187 |

**`λ_min` decreases monotonically in `ε` on both implementations and on every family.** The registered
`κ` (P2a: `λ_soft = λ_soft(0) + κ·ε`, `κ = 1.00`) is **not merely wrong in magnitude — it is wrong in
sign**: the fitted slope is negative everywhere. Same on the low-confinement extra
(`−0.1173 → −7.6717`) and in the race arms (`manifold shell_tilt_10` `λ_min_min = −13.56 ± …`, `r2 =
−2.62 ± 0.41`).

⭐ **And the mechanism is measured, not inferred.** §A4.2's `λ_soft = ε` is exact *at a vacuum*, i.e.
where `û · n̂_j = 0` for the atoms that carry the site. I instrumented exactly that condition:

| quantity | measured | ideal (§A4.2 holds) | random-orientation baseline |
|---|---|---|---|
| `tilt_vacuum_residual` = `Σ_j w_j (û·n̂_j)²` at written sites | **0.140** (envelope) / **0.343** (depth); 0.197–0.339 across race arms | **0** | `1/dim = 0.167` |
| group atom-centre spread (depth-weighted) ÷ shell radius | **1.95** (`manifold`) / 1.92 (`aggregate`) / 1.19 (`overload`) | ≪ 1 | — |
| effective atoms carrying a site (depth participation) | 0.093–0.159 of 32 (`≈ 3–5` atoms) | 1 shell | — |

⇒ **A written site is not a common vacuum of its own atoms — it is at or *worse than* a random
orientation** — because the learned store spreads a group's 32 shells over ~2× the designed radius. The
tilt therefore couples through the `h·Hess h` channel (negative) instead of the `∇h∇hᵀ` channel
(positive, `= ε`). **Stated generally: a designed degeneracy does not survive superposition.** One shell
has an exactly degenerate minimum set (unit-tested: two zero eigenvalues and one massive, `λ = ε` on the
tilted tangent to 5 %); a *sum* of 32 shells at spread centres has none, and the pseudo-Goldstone
construction has no vacuum to sit at. This is the same lesson as the gym's ridge saddle, one level up:
**the write objective decides the geometry, and it is not asked to preserve the basis's symmetry.**

### 3.4 The curvature hierarchy and the soft mode

| quantity | `gauss` | `shell_fixed` | registered (PREREG) | verdict |
|---|---|---|---|---|
| `λ_soft` at written sites (`ε = 0`) | 0.617 (median over sites) | **+0.0994** | **P2d: 0.100** [0.02, 0.40] | ⭐⭐ **CONFIRMED to 0.6 %** |
| `λ_massive` | — | **3.899** | P3a: 3.3 [0.5, 12] | ✅ CONFIRMED |
| hierarchy `λ_max/λ_min` | — | **6.83** (envelope) / **7.73** (depth) | P3b: ≥ 10 (≥3) | ◐ **REFUTED at 10, inside the ≥3 fallback**. ⚠ 7.7 ≈ theory §Q2.4's shipped `cond(H) = 7.69` (R3) |
| soft-mode participation on the **designed** coordinate `û` | — | **0.068–0.187** | P3c: ≥ 0.7 (≥0.5) | ⛔ **REFUTED decisively** |
| soft-mode participation on the **spectator** axis | 0.022 | 0.032 | — | the shell does **not** make the payload axis the soft mode |
| `λ_min` at the site: valley or saddle? | — | median **+0.0994**, min over sites **−0.0749** | P3d: +0.02 [−0.02, +0.15] | ◐ median in range; min slightly outside. **Not the ridge write's −0.5946 saddle** |

⭐ **P2d is the prediction I am most pleased to have registered and least pleased to have confirmed.**
I derived, before running, that a shell inside the shipped confinement is *not* degenerate: stationarity
on the shell gives a residual tangential curvature `λ_tan = 2α‖c‖/ρ ≈ 2α = 0.10`. Measured: **0.0994**.
⇒ **The confinement already tilts the shell by exactly the `2α` the Gaussian store was floored at
(gym: `λ_min = 0.0846–0.1000 ≈ 2α`).** Route 2 does not remove the floor `memory-gym-v0` §3.5 measured;
it reproduces it, from a different direction, to three digits.

### 3.5 ⭐ The lifetime dial `τ(ε)` — the registered floor confirmed, the `1/ε` branch unreachable

PREREG P4 derived, from the shipped `γ_read = 0.02`, `dt = 0.05` (`Γ = 0.4`), that the soft mode is a
damped oscillator with **two** branches, against the charter's pure `1/ε`:

| registered | prediction | measured | verdict |
|---|---|---|---|
| **P4d** the floor `τ = 2/Γ` | **5.0 time units = 100 steps** [60, 180] | **4.23–6.58 time = 85–132 steps** | ⭐⭐ **CONFIRMED** |
| **P4c** slope above the knee | **0.00** [−0.25, +0.25] | **+0.068 / −0.003 / +0.015** (three configs) | ⭐ **CONFIRMED** |
| **P4b** a knee at `ε* = Γ²/4` | **0.04** [0.01, 0.15] | consistent — every measured point lies on the flat branch | ◐ consistent, not isolated |
| **P4a** slope `−1` below the knee | **−1.00** [−1.15, −0.85] | **NOT TESTED** — see below | ⚠ **not refuted; unreachable** |

⛔ **Why P4a could not be tested, which is itself the result:** `τ ∝ 1/λ_soft` only while
`λ_soft < Γ²/4 = 0.04`, and **the shipped confinement floors `λ_soft` at `2α = 0.10`, 2.5× above the
knee.** So on the shipped store the ε dial can only ever move `τ` along the flat branch — measured slope
**+0.005 below the registered knee**, i.e. dead. The declared extra cell at `α = 0.005` (floor 0.01 <
`ε*`) did **not** rescue it: that store does not write at all (`λ_min = −0.117`, `r2 = −0.034`).

⭐ **The actionable statement: `ε` is not the manifold-payload lifetime dial on this store — `α` is its
ceiling.** `τ_max = Γ/2α = 4.0` time units unless the confinement is lowered, and lowering the
confinement breaks the write. That is a **new, specific, and falsifiable coupling** between charter
§A4.2's lifetime claim and the shipped coercivity, and it did not exist before this task.

### 3.6 Does the basis change destroy addressing? (task §6's third falsifier)

Measured at the **shipped atom budget**, where the store actually works (3 seeds):

| arm | `decode` | `acq` | `strict` | `λ_min_min` | final write loss | byte ratio |
|---|---|---|---|---|---|---|
| `gauss` = `shell_r0` | **0.9722** | **1.000** | 1.0000 | **+3.2394** | 0.00016 | 478.20 / 546.40× |
| `shell_fixed` | 0.9444 | 0.681 | 0.6806 | +2.0625 | 0.00052 | 546.40× |
| `shell` (learned r) | 0.9167 | **0.389** | 0.3889 | +1.9081 | 0.00044 | 546.40× |
| `shell_tilt_0.01` | 0.9167 | 0.708 | 0.7083 | +2.1697 | 0.00085 | 547.40× |
| `shell_tilt_1` | 0.9167 | 0.750 | 0.7500 | +1.6979 | 0.00098 | 547.40× |

**Both halves, as the task requires.** ✅ Decoding is **not** destroyed (0.9722 → 0.9167–0.9444; my
registered operational falsifier `decode < 0.10 AND acq < 0.5` does **not** fire). ⛔ But **acquisition
is: 1.000 → 0.389–0.750**, and monitor **#5 (addressing) newly trips on every shell arm** at the shipped
anchor where the Gaussian control trips only #3 and #6. And `λ_min` falls 3.24 → 1.70–2.16 — the shell
does exactly what it is designed to do (softens the well) and the address pays for it. At the base
budget the cost is larger: `overload decode 0.2593 → 0.1250` (`shell_fixed`), inside my registered range
[0.00, 0.28].

### 3.7 The byte ledger (two-sided, on every arm, launder included)

| quantity | predicted (PREREG §7) | measured | verdict |
|---|---|---|---|
| shell surcharge on the atom bytes | `1/(dim+2)`: **+12.5 %** at `dim=6`, **+14.3 %** at `dim=5` | **+12.50 %** / **+14.29 %**, all cells | ⭐⭐ **CONFIRMED exactly** |
| `manifold` ratio `gauss → shell` | **52.0× → 58.4×** | **52.00× → 58.40×** | ⭐⭐ **CONFIRMED exactly** |
| `manifold` ratio with the tilt | **59.6×** | **59.60×** | ⭐⭐ **CONFIRMED exactly** |
| `matched` | `False` everywhere | `False` on 90/90 cells; min ratio **17.11×** | ✅ |
| architectural floor | shell raises it to ≥ 2.40× at `dim=5` | consistent (`gym` law + 1 float/atom) | ✅ |

Also measured: `overload` 17.11 → 19.44 → 20.49×; `aggregate` 54.56 → 62.24 → 63.44×;
`overload_shipped` **478.20 → 546.40 → 547.40×**.

### 3.8 Compute (task §3's 2× bar), `n_atoms = 2046`, median of 5×60 reps, machine idle

| module | `V` | ×gauss | `∇V` | ×gauss | `Hess V` | ×gauss |
|---|---|---|---|---|---|---|
| `gauss` | 2.782 ms | 1.00 | 6.020 ms | 1.00 | 1.652 ms | 1.00 |
| `shell_r0` | 2.653 | **0.95** | 5.762 | **0.96** | 1.681 | **1.02** |
| `shell` | 3.429 | **1.23** | 7.779 | **1.29** | 1.603 | **0.97** |
| ⛔ `shell + tilt` (envelope) | 6.595 | **2.37** | 19.780 | **3.29** | 5.091 | **3.08** |
| `shell + tilt` (depth) | 5.535 | 1.99 | 10.823 | 1.80 | 2.074 | 1.26 |

⛔ **The tilt exceeds the 2× bar and this travels with every tilt claim, like the ψ's 17.1×.** The shell
alone does not (1.23–1.29×, P8a confirmed). Wall clock: **72 cells in 31.7 min + 18 in 9.0 min**;
**total measured compute ≈ 45 min**, far inside the 6 h budget.

---

## 4. PREREG scorecard (`.claude/outputs/ssb-shell-atoms/PREREG.md`, written before any measured run)

| id | prediction | outcome |
|---|---|---|
| **P1a/b/c** | `V`, `∇V`, `Hess V` bit-identical at `r=0`, **exact 0** | ⭐⭐ **CONFIRMED**, f32 **and** f64, including at `q == c`, on the short-circuit **and** the forced-arithmetic path |
| **P1d** | written store bit-identical | ⭐ **CONFIRMED** — but only via the static short-circuit; **the registered XLA risk MATERIALISED** (9.5e−7 in `centers` after 60 Adam steps on the arithmetic path). Registered in advance, reported, fixed |
| P1e | `radius_scale=1`, `r → 0` recovers the Gaussian | ✅ CONFIRMED (rel. err < 1e−5 at `r = 1e−4`) |
| **P2a** | `λ_soft = λ_soft(0) + κ·ε`, **κ = 1.00** [0.5, 2.0] | ⛔⛔ **REFUTED IN SIGN.** `λ_min` *decreases* monotonically in `ε`, both implementations, every family |
| **P2b** | log–log slope of `Δλ` vs `ε` = **1.00** [0.85, 1.15] | ⛔ **REFUTED** — `Δλ < 0` at every `ε`, so the registered fit is undefined |
| **P2c** | `λ_min > 0` at every `ε ≥ 1e−3` | ⛔ **REFUTED** (`ε ≥ 1` gives saddles; 0/3 admissible) |
| ⭐ **P2d** | `λ_soft(0) = 2α‖c‖/ρ ≈ **0.100**` [0.02, 0.40] | ⭐⭐ **CONFIRMED to 0.6 %: 0.0994.** The confinement, not the design, sets the soft scale |
| **P3a** | `λ_massive ≈ **3.3**` [0.5, 12] | ✅ CONFIRMED — **3.899** |
| **P3b** | hierarchy ≥ **10** (fallback ≥3) | ◐ **REFUTED at 10**: **6.83 / 7.73** (inside the fallback) |
| **P3c** | soft-mode participation on `û` ≥ **0.85** (≥0.5) | ⛔ **REFUTED** — **0.068–0.187** |
| **P3d** | `λ_min(shell_fixed) = **+0.02**` [−0.02, +0.15], a valley not a saddle | ◐ **PARTIAL**: median **+0.0994** (in range), min over sites **−0.0749** (outside). Not the ridge's −0.5946 |
| **P3e** | `ε = 10` collapses the hierarchy and degrades the metric | ✅ CONFIRMED — the declared destructive anchor, `r2 = −2.62 ± 0.41`, `λ_min = −13.56` |
| **P4a** | `τ` slope **−1.00** below the knee | ⚠ **NOT TESTED** — `2α` floors `λ_soft` 2.5× above the knee (§3.5). Not refuted |
| **P4b** | a knee at `ε* = **0.04**` | ◐ consistent; every point is on the flat branch |
| **P4c** | slope **0.00** above the knee [−0.25,+0.25] | ⭐ **CONFIRMED** — +0.068 / −0.003 / +0.015 |
| **P4d** | floor `τ = **5.0** time = 100 steps` [60,180] | ⭐⭐ **CONFIRMED** — 4.23–6.58 time = **85–132 steps** |
| **P4e** | payload survives the read at `ε ≤ 1e−2`, erased at `ε ≥ 0.1` | ⛔ **REFUTED** — `τ ≈ 5` ≪ the 800-step read at **every** `ε`; the payload is erased regardless |
| **P5a** | `Δ final_write_loss(shell_fixed − gauss) = **+0.05**` [0.005, 0.15] | ◐ **MIXED, and interesting**: `manifold` **+0.108** ✅ in range; `overload` **+0.204** (above); `aggregate` **−0.151** (**wrong sign**) |
| P5b | `final_write_loss(shell_fixed) = 0.10` [0.02, 0.30] | ✅ CONFIRMED: 0.246 / 0.089 (2 of 3 families in range; `overload` 0.436 outside) |
| **P5c** | learned `r` **collapses to < 0.15** (w20: free learning erases design) | ⛔ **REFUTED** — mean `r` = **0.501** ≈ its init. The write objective is **blind to the radius**, neither erasing nor exploiting it |
| P5d | `r2(shell) ≈ r2(gauss)` [−0.25, +0.25] | ✅ CONFIRMED numerically (Δ = +0.172, just outside) but for the **opposite reason** — see P5c |
| **P6 manifold** | `shell_fixed r2 = +0.10` [−0.60,+0.60]; **directional `> gauss`, Δ = +0.28** | ✅ **CONFIRMED**: `r2 = −0.0108` (in range), **Δ = +0.169** (in range), sign correct |
| **P6 overload** | `shell_fixed decode = 0.10` [0.00, 0.28] | ✅ CONFIRMED — **0.1250** |
| P6 aggregate | `shell_fixed neg_mae = −0.65` [−0.95,−0.45] | ✅ CONFIRMED — **−0.5841** |
| **P7a/b/c** | +12.5 % / +14.3 %; 52.0→58.4→59.6× | ⭐⭐ **CONFIRMED EXACTLY** (+12.50 %/+14.29 %; 52.00→58.40→59.60×) |
| P7d | `matched=False` everywhere | ✅ 90/90 |
| **P8a** | shell `V` cost **1.15×** [1.0,1.6] | ✅ CONFIRMED — **1.23×** |
| **P8b** | shell+tilt **1.5×** [1.0,2.5] | ◐ `V` **2.37×** (just outside); `∇V` **3.29×** ⛔ outside; reported in the ledger |
| P8c | 81 cells in **45 min** [20 min, 3 h] | ✅ CONFIRMED — 90 cells in **41 min** |

**Score: 14 confirmed (5 of them exact or near-exact) · 6 partial · 8 refuted · 1 not tested.**
The refutations cluster hard on **one object — the tilt (P2a/b/c, P3c, P4e)** — and on the assumption
that the write would engage the design (**P5c**). Every confirmation is on a quantity I derived from the
shipped constants (`2α`, `2/Γ`, `1/(dim+2)`, `A/s²`), and every refutation is of something the charter or
I hoped the *learning* would do. That is the same direction the gym's scorecard failed in.

## 5. Falsifiers, adjudicated

| falsifier (task §6) | outcome |
|---|---|
| ⛔ `r=0` does not reduce to the Gaussian ⇒ everything invalid | ✅ **did NOT fire** — gate green at unit and system level (§3.1) |
| ⛔ the tilt does not produce `λ_min ≈ ε > 0` at any registered `ε` ⇒ §A4.2 refuted, **report as headline** | ⭐⭐ **FIRED.** Headlined (§3.3), on two implementations, with the mechanism measured |
| ⛔ shell atoms destroy addressing (`overload`/`aggregate` outside my registered range) | ◐ **did not fire on my registered range** (decode 0.1250, range [0.00,0.28]); ⚠ **but acquisition falls 1.000 → 0.389–0.750 at the shipped budget and monitor #5 newly trips on every shell arm.** Both halves reported (§3.6) |
| does NOT falsify: ≤0 manifold dividend with `echo` at 1.0000 | invoked — the substitute margin is ≈ **−1.00 on every manifold arm including the control** |

## 6. NOT RUN (declared; never reported as a null)

- ⛔ **D4, the Route-1 × Route-2 2×2 cell.** `traj-write-objective`'s write-objective commit (`fed58c5`)
  is on **its own branch, not on `main`**, at the time of my final race run; the task's condition
  ("has landed on `main`") was not met. **NOT-RUN, for the Hub to run at integration** — everything it
  needs is in place (the seam takes `write_objective=...`; my `shell_rig` is orthogonal to it).
- The **trajectory launder** (no ψ here sees the address block — §Dial declaration).
- `recency` (Route 1 owns its D4 diagnostic; out of the gate until then).
- Rank > 1 tilts; per-atom tilt directions; a learned `ε`; a tilt direction *chosen* to minimise the
  vacuum residual (the obvious next design move — see §8.2).
- `manifold`/`aggregate` at the shipped atom budget (only `overload` got the one permitted escalation).

## 7. Git footprint

- **Branch** `agent/experiment-engineer/ssb-shell-atoms`, worktree `../CHLU-ssb`, base
  **`1ce02e4`** (`traj-write-objective`'s RACE+SEAM FREEZE), rebased cleanly (no conflicts).
- **Commits (4):** `f92adfa` shell atom + tilt dial, r=0 gate · `a213815` Route-2 race runner + dial probe
  + τ(ε) · `ff33eb4` liveness table + budget escalation + figure · `647c54e` figure arm ordering.
- **Files touched (3, all new, all mine):** `chlu/core/shell_atoms.py` (515) ·
  `chlu/experiments/exp_ssb_shell.py` (722) · `tests/test_shell_atoms.py` (365). **+1602 lines, 0
  deletions, 0 files modified.**
- **Read-only compliance: ZERO violations.** `git diff --stat 1ce02e4..HEAD` shows only the three new
  files; `clu_system.py`, `train_memory.py`, `eval/{race,dividend}.py`, `experiments/{memory_gym,
  exp_memory_gym}.py`, `cli/experiment_cmd.py`, `core/{psi_readout,monitors,implicit_grad,
  memory_potentials}.py` and `config.py` are **untouched**. ⚠ The two runtime monkey-patches (§2.1) are
  confined to a context manager in **my** module and are restored in a `finally`.
- **Tests:** `tests/test_shell_atoms.py` **22 passed** (20.8 s; 17 test functions, 5 parametrised).
  ✅ **Full suite on the branch: `925 passed, 0 failed` in 19 m 48 s** (`pytest -q`, worktree, main venv,
  JAX 0.9.0) — log `.claude/outputs/ssb-shell-atoms/pytest_full.log`. The branch adds only new files, so
  this is base(`1ce02e4`) + 22, no regressions. The module carries the standard autouse
  `float32_dynamics` fixture (handover §7.2, the x64-at-import ordering trap).
- `ruff check chlu/ tests/` clean. **Not pushed, not merged.**

## 8. Open questions / follow-ups / risks

1. ⛔ **R1 (the mask hook) blocks any future store family, not just mine.** One config field fixes it.
2. ⭐ **The deepest result here is not the tilt — it is that a designed degeneracy does not survive
   superposition.** The write spreads a group's 32 shells over **~2× the designed radius**, so no
   symmetry-based construction on the *atom* has a vacuum at the *site*. Two concrete C2W3 moves follow,
   both cheap: (a) constrain a group's atoms to a common centre (1 shell per item, or a shared centre
   with learned radii) — this is the **factored store**'s question in miniature; (b) choose `û` per group
   to minimise the measured `tilt_vacuum_residual` (designed direction, learned placement) instead of
   learning it against an objective that cannot see it.
3. ⚠ **The write objective is blind to the shell radius** (P5c: `r` moves from 0.500 to 0.501 in 300
   steps). Route 1's `λ_path` term is exactly the missing signal — **the 2×2 is the wave's most
   informative unrun cell**, not a nice-to-have.
4. ⚠ **`ε` is not a lifetime dial on this store; `α` is its ceiling** (`τ_max = Γ/2α = 4.0` time units).
   Any charter text that treats `ε` as "the manifold-payload lifetime dial ∝ 1/ε" needs the coupling to
   the coercivity attached (§3.5). Lowering `α` to reach the `1/ε` branch **breaks the write** (measured).
5. ⚠ **Admissible coverage is 28–33 % on `manifold`/`aggregate`.** I did **not** spend their escalation
   (only `overload` had zero admissible cells, the ruling's trigger). If the Hub wants those families at
   full power, the same `load1x_shipped`-style budget escalation is the move, at ≈ 30 s/cell.
6. **Risk to flag:** the manifold family's `r2 ≈ 0` is scored identically by *"the store returns the
   launch spectator coordinate"* and *"the store returns zero and the target happens to be
   zero-mean."* The `echo` substitute at 1.0000 makes the family unable to distinguish them. **F4 needs a
   target the launch point does not already contain** before it can measure pillar (d) at all — this is
   the same redesign the gym asked for in its own §4(i), now with a second independent reason.

---

## Proposed handover updates (for the Hub)

**§2/§3 (architecture + config).**
- New module `chlu/core/shell_atoms.py`: `ShellAtomDictionaryPotential` (drop-in for
  `AtomDictionaryPotential`; extra leaves `radius_raw`, `tilt_dir`; statics `radius_scale`, `tilt_eps`,
  `tilt_weight`), `shell_write_mask_fn`, `shell_potential_from` (the `store_potential_factory` entry),
  `shell_hessian_spectrum`. New runner `chlu/experiments/exp_ssb_shell.py` (**no CLI hook** — module
  invocation, `--families/--arms/--seeds/--gym-arm/--family-label/--quick`), plus `emit_race_from_json`,
  `liveness_table`, `plot_from_json`. New `tests/test_shell_atoms.py` (22 tests).
- ⛔ **`store_potential_factory` is insufficient without a companion `store_write_mask_factory`** — R1.

**§7 (known issues / live).**
- ⚠ **A bit-identical-arithmetic regression gate is NOT sufficient under `eqx.filter_jit`.** XLA
  reassociates an enlarged expression and the *parameter* gradient moves ~1 ULP (measured **9.5e−7** in
  `centers` after 60 Adam steps) while the forward pass stays exact. Regression gates that must survive a
  *write* need a **static code-path short-circuit**, not just careful arithmetic. (New, generalisable.)
- ⛔ **`overload` at the gym's base atom budget is unusable as a gate family**: 0/18 admissible including
  the Gaussian control (`λ_min −0.43…−1.20`). Quote `overload` only at `load1x_shipped` (478×), where
  coverage is 100 %.
- ⚠ **`2α` is a hard floor on the soft curvature of *any* store element**, not a property of Gaussian
  atoms: a shell atom inside the shipped confinement has `λ_tan = 2α‖c‖/ρ`, **measured 0.0994 vs the
  gym's `0.0846–0.1000 ≈ 2α`**. Route 2 reproduces `memory-gym-v0` §3.5's floor from the basis side.
- ⚠ **The shell costs +1/(dim+2) bytes for bit-identical `r=0` behaviour** (+12.5 % at `dim=6`, +14.3 % at
  `dim=5`); the envelope-weighted tilt costs **2.37× `V` / 3.29× `∇V`** and must travel with every tilt
  claim.

**§8/§10 (record).**
- ⭐⭐ **Charter §A4.2's pseudo-Goldstone mechanism is REFUTED on a learned store.** The tilt does not give
  `λ_min = ε > 0`; it **monotonically reduces** `λ_min` (`+0.0994 → −8.28` over `ε = 0 → 10`, envelope
  weight; `+0.3291 → −1.198`, depth weight), and drives sites to saddles at `ε ≥ 1`. It **is** exact in
  the idealised single-atom geometry it was specified in (unit-tested: `λ = ε` to 5 % at four `ε`).
- ⭐⭐ **The mechanism of the failure, measured: a designed degeneracy does not survive superposition.**
  `tilt_vacuum_residual = 0.140–0.343` vs the ideal 0 and the random-orientation baseline `1/dim = 0.167`;
  a group's depth-weighted atom-centre spread is **1.19–1.95× the designed shell radius**. A written site
  is not a common vacuum of its own atoms. Generalises the gym's ridge-saddle finding from the objective
  to the basis.
- ⭐ **`ε` is not the manifold-payload lifetime dial on the shipped store.** The registered damped-mode
  floor `τ = 2/Γ = 5.0` time units is **CONFIRMED (measured 4.23–6.58)**, the above-knee slope **0**
  is **CONFIRMED**, and the `1/ε` branch is **unreachable** because `2α = 0.10` floors `λ_soft` 2.5×
  above the knee `ε* = Γ²/4 = 0.04`. `τ_max = Γ/2α = 4.0`; the ceiling is the coercivity.
- ⭐ **The r=0 gate is GREEN** and `gauss` reproduces `memory-gym-v0` exactly on all four families
  (`manifold −0.1802 ± 0.1708`, `overload 0.2593`, `aggregate −0.5261`, shipped anchor
  `0.9722 ± 0.0139 / λ_min +3.24 / loss 1.6e−4`) — so the C2W2 rig is bit-faithful across two branches.
- ⭐ **Route 2's gate verdict: `any_family_clears = False`** — 90 cells, 0 clearing 2 SE, **16
  `le_zero_vote`**, 12 `abstain`, 2 `under_powered_grid`; the best cell is `manifold/shell_tilt_0.1` at
  **+0.0005 (n=1)** and the best multi-seed arm is `manifold/shell_fixed` at **−0.0072**, against an
  `echo` substitute of **1.0000**.
- ⚠ **The shell removes the manifold family's negative dividend but does not express a manifold**
  (`−0.1802 → −0.0108`, SE `0.171 → 0.0065`; soft-mode spectator participation `0.022 → 0.032`). It stops
  the settle destroying the spectator coordinate; it does not make that coordinate the soft mode.
- ⚠ **The shipped write objective is blind to a designed radius**: learned `r` moves 0.500 → **0.501** in
  300 steps. w20's "free learning erases design" did **not** happen — something weaker and worse did:
  the design was **ignored**.
