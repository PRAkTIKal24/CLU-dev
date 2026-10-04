# r2-geometry-revival — experiment-engineer report (w25)

**Task + acceptance criterion:** Stage 0 = the free `lattice-capacity-theory` §5.0 trained-
`log_width` dump (adjudicate the geometric vs write-operator account of the `K_learned`
ceiling); if it survives → Stage 1, the one-flag revival `atom_init_width 0.30 → 0.15`
(predicted `~2^d` wall movement); Stage 2 = the d-sweep figure, conditional on Stage 1
moving the wall.

**Status: done — Stage 0 SURVIVES its registered test (and classifies all seven d ≥ 4 cells
perfectly, crack cell included); Stage 1 REFUTES the account Stage 0 supports (the flag
*destroys* the wall: 0.9368 → 0.5944, 3 seeds); Stage 2 NOT RUN, correctly gated off. The
mechanism probes then found what actually binds — a well-width ↔ read-out-excursion REACH
condition — and confirmed it by intervention.**

> **⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **The N92/N96 mechanism question is SETTLED, and neither contestant won.**
>    `sep/width` is an **excellent correlate** — on trained widths it classifies all seven
>    d ≥ 4 cells perfectly with a threshold in (2.30, 2.49), *including* the w23 d=8-K=64
>    crack — but it is **not the cause**: forcing the ratio to 4.90 destroys retrieval (§2).
>    And a *d-independent write-operator* ceiling cannot produce a correlate that sharp.
>    The measured binding constraint is a third thing: a **well-width ↔ payload-excursion
>    REACH condition** (§4), confirmed by intervention (§4/§4.1). The CONTESTED flag on
>    N92/N96 resolves as *"contested — and the answer was option three."*
> 2. **`lattice-capacity-theory` §5.2 must be marked REFUTED BY EXPERIMENT.** `atom_init_width
>    0.30 → 0.15` does not raise `K_learned(4)` from 16 to 64–128; it **destroys** the wall
>    (K=16 falls 0.937 → 0.594, 3 seeds, budget-adequate). No `2^d`.
> 3. **`atom_init_width = 0.30` must stop being described as "just an initialisation."**
>    It is a near-optimum of a measured monotone trade and the trained width tracks it only
>    loosely (init 0.15 → trained 0.185; init 0.30 → trained 0.31).
> 4. **⭐ The highest-value item, and it needs a Hub decision, not a wording fix:**
>    **the w23 walls are on the far side of the read-out excursion.** Halving the payload
>    excursion (tolerance halved with it) takes **d=4 K=32** from **0.824** — w23's *firm*
>    wall, flat across a 16× atom sweep — to **1.000**, and **d=4 K=64**, four rungs above
>    `K_learned(4)=16`, to **0.9922**, at *both* widths (§4.1). ⚠ **This is a mechanism
>    result at a different task point, NOT a capacity number** (this harness's payload channel
>    is noise-free, so shrinking the excursion is free here). The Hub must decide whether the
>    read-out excursion is a legitimate design parameter of the primitive and, if so, commission
>    the fair version (payload read noise, or a multi-channel payload) — §10.1.
> 5. **New negative result proposed (tier A, memory-architecture)** — text in §8.
> 6. Do-not-quote unchanged: base √2 / `d^1.62` (CM-22(j)).

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** capacity (the R2 law) — a law about the primitive, exempt from the masked-recall
  demotion; its figure is never framed as beating anything (CM-23(m)).
- **Laundering control:** the designed write at matched geometry (§6).
- **Falsifies (per stage):** §5.0 — trained widths < ~0.18 or minsep/width varying >2×
  across d. §5.2 — no movement of the wall at ≥3 seeds under an adequate budget.
- **Does NOT falsify:** the ceiling failing to reach the designed `4·2^d` (the 4× prefactor
  gap is expected); any comparison to kNN or external methods.
- **Fairness category of every knob moved (N46):** `atom_init_width` and `min_atoms_base`
  are **initialisation/budget** knobs — they set where atoms start and how many there are.
  **No arm supplies placement, formula-set centers or hand-set per-item widths**; every
  center/width/amplitude is still learned by the same static write. The payload-rescale
  probe (§5) rescales the *task's* codebook and its tolerance by the same factor, so it
  changes the task, not the mechanism, and is reported as a mechanism probe, never as a
  capacity number.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/r2-geometry-revival`, base local `main` @ `63c668d` |
| commits | `f770656` (width-dump helper + 2 tests) — see §9 |
| worktree | `../CHLU-r2geom`; **main venv reused** (protocol §4), **JAX 0.9.0**, equinox 0.13.4 |
| harness | `chlu/experiments/exp_designed_mechanism.py`, unmodified write/read/score path; drivers in `.claude/scratch/r2-geometry-revival/` |
| geometry | d-ball, farthest-point `designed_sites`, R=1, wall_margin .5, `site_seed=0`, `payload_seed` default; payload channel `q[d]`, codebook `linspace(-1,1,K)` |
| write objective | `train_memory_landscape`, **GLOBAL** (`learned_arm=learned_global`), 600 Adam(3e-3), wd 1e-4, n_perturb 32, σ_addr .25, **σ_pay .6**, margin .15, barrier .2, payload_index=d |
| retrieval | γ_address .05 × 400 → γ_read .02 × 800, dt .05, tail_frac .25, n_subsample 8 |
| queries | `fixed_norm` jitter **σ_q = 0.15**/√d per axis, σ_p .05, n_query_per_item 32 (cap 4096) |
| criterion | strict = basin_ok ∧ \|read−a_i\|<`payload_tol`=0.1; cell PASS = mean strict ≥ **0.9** ∧ value-blank ok |
| learned mechanism | `AtomDictionaryPotential` in `DesignFreedomPotential(free_mlp, atoms)`; init_scale 1.0, depth_init 1e-4, confine .05, `n_groups=K` |
| **atom budget** | `n_atoms = max(32·K, min_atoms_base·√2^d)`. **Stage 0: w23 values** (`min_atoms_base` 512, ×2 where the w23 cell used 2× atoms). **Stage 1: coverage-raised per theorist spec `min_atoms_base × 2^{d/2}`** ⇒ `min_atoms_base=2048` at d=4 ⇒ **8192 atoms**; N92 re-check at **16384** |
| **the one flag** | `atom_init_width` **0.30 (w23) → 0.15 (Stage 1)**; every other knob at its w23 value |
| seeds | Stage 0 **1 seed (0)** per cell (declared limitation, §7); Stage 1 headline **3 seeds (0,1,2)**; explores/probes 1 seed |
| langevin_noise | **N/A** — deterministic Verlet, no temperature anywhere in this task |

**PREREG:** `.claude/outputs/r2-geometry-revival/PREREG.md`, written before any harness ran
(only a JAX warm-up + `git worktree add` preceded it). **Amendment A1** (declared in the
file) fixed the S0.2 reading rule after the d=2 K=4 smoke cell and before the other 11.

---

## 1. ⭐ STAGE 0 — the trained width dump (§5.0). The account SURVIVES its registered test.

`log_width` is trainable, so the width that enters any `minsep/width` argument is a
**measured** quantity. Written landscape only, global write, w23 flags, seed 0.
`w_atom` = contribution-weighted median trained width of the atoms forming the well at a
stored site (median over sites); `s_fit` = width from a 3-parameter fit of the REAL learned
`V`'s direction-averaged radial profile `V0 + D(1−exp(−r²/2s²))` around each site.

| d | K | w23 verdict | site sep | **w_atom** | s_fit | **sep/w_atom** | sep/s_fit | fit R² |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | PASS | 1.159 | 0.274 | 0.271 | **4.23** | 4.27 | 0.999 |
| 2 | 8 | FAIL | 0.649 | 0.215 | 0.206 | **3.02** | 3.15 | 1.000 |
| 3 | 8 | PASS | 0.923 | 0.290 | 0.258 | **3.19** | 3.58 | 1.000 |
| 3 | 16 | FAIL | 0.744 | 0.251 | 0.221 | **2.96** | 3.37 | 0.999 |
| 4 | 16 | PASS (0.928) | 0.903 | 0.302 | 0.279 | **2.99** | 3.24 | 1.000 |
| 4 | 32 | FAIL | 0.710 | 0.310 | 0.251 | **2.30** | 2.83 | 1.000 |
| 5 | 32 | PASS (0.898, marginal) | 0.849 | 0.337 | 0.306 | **2.52** | 2.78 | 1.000 |
| 5 | 64 | FAIL | 0.690 | 0.311 | 0.258 | **2.22** | 2.67 | 1.000 |
| 6 | 32 | PASS | 0.924 | 0.360 | 0.347 | **2.57** | 2.66 | 1.000 |
| 6 | 64 | **FAIL (0.818)** | 0.795 | 0.347 | 0.313 | **2.29** | 2.54 | 1.000 |
| **8** | **64** | **PASS (0.907)** ⭐ the crack cell | 0.908 | **0.364** | 0.356 | **2.49** | 2.55 | 1.000 |

**⭐ The clean regularity: for d ≥ 4 the pass/fail split is separated PERFECTLY by a single
threshold `t* ∈ (2.30, 2.49)`.**

| d ≥ 4 cells | sep/w_atom |
|---|---|
| **PASS**: d4 K16 · d5 K32 · d6 K32 · **d8 K64** | 2.99 · 2.52 · 2.57 · **2.49** |
| **FAIL**: d4 K32 · d5 K64 · d6 K64 | 2.30 · 2.22 · **2.29** |

Seven cells, four dimensions, a 4× K range, **zero misclassifications**, and the boundary is
tight (2.30 | 2.49). This *includes the w23 crack* the task was built around: **d=8 K=64
passes at 2.49 while d=6 K=64 fails at 2.29**, i.e. the width-lock account predicted the
crack's sign correctly out of sample, on trained rather than assumed widths.
It is **d = 2 and d = 3 that break the universality** (fails at 3.02 and 2.96, above three of
the d ≥ 4 passes) — exactly the two dimensions where the write *narrows* the atoms
(0.215–0.274) and where farthest-point packing in a fixed ball is boundary-dominated.

**PREREG scorecard, Stage 0**

| # | registered | measured | verdict |
|---|---|---|---|
| **S0.1** | median well width **≥ 0.28** ("the write does not narrow the atoms") | over-cell median **0.310**; **8/11 cells ≥ 0.28**, and **every cell at d ≥ 4 is ≥ 0.30**; the three sub-0.28 cells are d=2/d=3 (0.215–0.274) | ◐ **holds where it matters, fails at d ≤ 3** |
| **S0.1 falsifier** | width < 0.18 ⇒ account dead | **min over all cells 0.215**; no cell below 0.18 | ✅ **not triggered** |
| **S0.2** | `minsep/width ∈ [2.4, 3.1]` at every d | first-FAIL column **3.02 / 2.96 / 2.30 / 2.22 / 2.29** (d=2/3/4/5/6): two inside the band, three just below | ◐ band slightly too high/tight |
| **S0.2 falsifier** | ratio varies **>2×** across d | first-fail range **1.36×** (2.22→3.02); last-pass range 1.68× | ✅ **not triggered** |
| **S0.2 sharper (A1)** | a single threshold t\* ∈[2.4,3.1] separates all PASS from all FAIL | **no universal threshold exists** (min PASS 2.52 at d=5 < max FAIL 3.02 at d=2) — **but t\* ≈ 2.4 separates every cell at d ≥ 4 perfectly** | ◐ **refuted as universal; sharp for d ≥ 4** |
| **S0.3** ⭐ | d=8 K=64 (PASS) and d=6 K=64 (FAIL) trained sep/width **straddle** the transition (registered: d=8 ≥ ~2.9, d=6 ≤ ~2.7) | **d=8 K=64 = 2.49 (pass side), d=6 K=64 = 2.29 (fail side)** — they straddle the measured d≥4 boundary (2.30, 2.49) exactly, but **both sit below the registered absolute band** because the trained widths (0.36, 0.35) are wider than the 0.30 the band assumed | ◐ **straddle CONFIRMED in sign and order; the registered absolute values were wrong** |
| **S0.4** (mine) | `w_atom ∈ [0.24, 0.31]` (slight downward drift) | only **5/11 cells** land in my band; measured span **0.215–0.364**, and **the drift is UPWARD at d ≥ 5** (0.311, 0.337, 0.347, 0.360, 0.364) — I registered the wrong sign | ✗ **my band is wrong; the write WIDENS the atoms at high d** |
| **S0.5** (mine) | `s_fit ≥ w_atom`, ratio ∈[1.0, 2.0] | `s_fit/w_atom` = **0.90–1.00** (median 0.91) — the fitted well is marginally NARROWER than the constituent atoms | ✗ **rejected** (the offset-atom superposition sharpens rather than broadens the core; see note) |
| **S0.6** (mine) | no payload/address width split (>1.5× is a finding) | width ratio payload-dominant / address-dominant atoms = **0.88–1.13** | ✅ **null confirmed** |

*Note on S0.5.* The radial fit is an r²-weighted compromise over a slightly super-Gaussian
core (many offset atoms), so `s_fit` sits ~9 % below the atom width; it is not a
contradiction of "a sum of positive Gaussians of width s cannot be narrower than s" in the
tail sense, but my registered ≥1.0 band was wrong. All ratios below use `w_atom`.

**Stage-0 verdict (registered rule): the geometric account SURVIVES, and rather well.** No
falsifier fired: widths are 0.215–0.364 (never below 0.18) and the critical ratio moves only
**1.36×** across a 4× dimension range and a 16× K range; at d ≥ 4 the ratio is a **perfect
classifier**. Two refinements the account did not predict: (a) the trained width is **not**
locked at init — it drifts **down** with crowding at low d and **up** with dimension (0.215
→ 0.364), so "`atom_init_width` is only an initialisation" is half-true at best, and the
account's own arithmetic (which assumed 0.30 everywhere) mis-states every ratio it quoted;
(b) at d ≤ 3 the criterion is not even a window — two FAIL cells sit above three d ≥ 4 PASS
cells.

**Per §5.0's own rule this authorised Stage 1.** It also, by itself, deprecates the
N92/N96 wording: a d-independent operator ceiling has no business producing a critical
`sep/width` this stable.

---

## 2. ⭐⭐ STAGE 1 — the one-flag revival is REFUTED. The wall moves, downward.

3 seeds, shipped `evaluate_arm_cell` (written **and** value-blank landscapes), the
theorist's coverage-raised budget (`min_atoms_base × 2^{d/2}` ⇒ 8192 atoms at d=4), the
ONLY difference between the two rows being `atom_init_width`:

| d | K | `atom_init_width` | atoms | **strict (3 seeds)** | basin | blank ok | **cell** |
|---|---|---|---|---|---|---|---|
| 4 | 16 | **0.30** (w23) | 8192 | **0.9368 ± 0.0133** | 0.9368 | ✅ | **PASS** |
| 4 | 16 | **0.15** (the flag) | 8192 | **0.5944 ± 0.0164** | 0.5944 | ✅ | **FAIL** |

**Δstrict = −0.3424** (0.9368 − 0.5944); per-seed sds 0.0133 / 0.0164 ⇒ SE of the difference
**0.0122**, i.e. the gap is **28 SE** (or 16× the pooled per-seed sd) and the two 3-seed
ranges do not come close to touching (**0.9473 / 0.9414 / 0.9219** vs
**0.5859 / 0.6133 / 0.5840**).
The flag does not raise `K_learned(4)` from 16
to 64–128; it takes `K_learned(4)` **below 16** — the w23 wall itself no longer holds.

**N92 budget adequacy (mandatory at every first-fail):** the failing width-0.15 cell at
**2× atoms** (16384): strict **0.5859 → 0.6094** (seed 0, +0.023, still 0.29 below the bar).
**Flat ⇒ not budget-limited ⇒ a real collapse, not starvation.** (The coverage raise the
theorist prescribed is therefore *not* the missing ingredient; it was already applied in the
headline row, and doubling it again does nothing.)

**Ladder context and the trade curve** (1 seed, written-only, d=4, 8192 atoms; the
exploratory cells carry no blank control — declared, and every decisive cell above is the
full shipped harness):

| d | K | width | trained `w_atom` | sep/w_atom | strict | basin | payload err |
|---|---|---|---|---|---|---|---|
| 4 | 16 | 0.15 | 0.184 | **4.90** | 0.586 | 0.586 | 0.347 |
| 4 | 16 | 0.15, **2× atoms** | 0.182 | 4.96 | 0.609 | 0.609 | 0.334 |
| 4 | 32 | 0.15 | 0.185 | **3.84** | 0.600 | 0.631 | 0.330 |
| 4 | 32 | 0.20 | 0.235 | 3.02 | 0.776 | 0.776 | 0.200 |
| 4 | 32 | 0.30 | 0.313 | 2.27 | 0.824 | 0.824 | 0.139 |
| 4 | 32 | 0.40 | 0.403 | **1.76** | **0.832** | 0.832 | 0.129 |

The trade curve at the decisive cell is **monotone increasing in width across 0.15 → 0.40
and flat at the top** (0.600 → 0.776 → 0.824 → 0.832), i.e. the shipped 0.30 sits on the
shoulder of a plateau, and `sep/width` runs from 3.84 down to **1.76** across that curve
while strict goes *up*. Under the geometric reading this is upside down.

Two things in that table are fatal to the geometric account **as a causal claim**:

1. **d=4 K=16 at width 0.15 sits at `sep/width = 4.90`** — *above every ratio at which the
   w23 ladder ever passed* (max observed PASS 4.23), and 2× the d ≥ 4 boundary (2.30, 2.49)
   that classified all seven Stage-0 cells perfectly — **and it fails at 0.594.** The account
   predicts a comfortable PASS; the same variable that classifies the *unmanipulated* cells
   without error is useless the moment it is manipulated. That is the signature of a
   **correlate**, not a cause.
2. **Capacity is monotone increasing in width over the measured range** (0.15 → 0.20 → 0.30
   gives 0.600 → 0.776 → 0.824 at the same cell), i.e. exactly the opposite sign to the
   `K ∝ (sep/s)^d` reading, which wants *narrower* wells to buy capacity.

**PREREG scorecard, Stage 1**

| # | registered | measured | verdict |
|---|---|---|---|
| S1.1 (theorist) | `K_learned(4)`: 16 → **64–128** | **< 16** (K=16 fails at 0.594) | ✗ **REFUTED, and in the opposite direction** |
| S1.2 (theorist) | `K_learned(6)`: 32 → ≥128 | not run at 3 seeds (compute; §7) — but d=4's collapse is at the *same* flag and the mechanism (§4) is d-independent | ⛔ NOT RUN (declared) |
| S1.3 (theorist) | the wall moves by ~`2^d` | it moves by **< 1/1** — downward | ✗ REFUTED |
| S1.4 (mine) | central estimate `K_learned(4) = 32`; P(≥1 rung up)=0.6; P(full `2^d`)=0.2 | **0 rungs up, ≥1 rung DOWN** | ✗ **my own central estimate is refuted too — I was wrong in the same direction as the theorist, just less** |
| **S1.5 (mine, the trade)** | at cells passing under both widths, width 0.15 loses **0.01–0.06** strict, concentrated in `basin_ok` | loses **0.342** — an order of magnitude more than I registered — and it **is** concentrated in basin (basin ≡ strict at every cell) | ◐ **direction and locus exactly right, magnitude wrong by ~6–30×** |
| S1.6 (mine, laundering) | designed arm unaffected by `atom_init_width` | §6: designed arm re-measured, **identical to w23**, and provably does not read the flag | ✅ |
| — | budget adequacy (N92) at every first-fail | 2× atoms: 0.586 → 0.609, flat | ✅ **the collapse is not starvation** |

**⇒ STAGE-1 VERDICT: NO REVIVAL. `atom_init_width 0.30 → 0.15` lowers `K_learned(4)` below
16.** Per the task's own falsifier this kills the width route to R2. **Stage 2 (the d-sweep
figure at the surviving width) is therefore correctly NOT RUN** — there is no surviving
width other than the shipped 0.30, and re-drawing the w23 figure at 0.30 would be a re-run,
not a result.

---

## 3. The width the flag actually buys (why "one flag" is a misnomer)

`atom_init_width` is an **initialisation**; the write moves it. Measured init → trained:

| init | trained `w_atom` (d=4 K=16/32) | all-atom population median |
|---|---|---|
| 0.15 | **0.184 / 0.185** (+23 %) | 0.150 |
| 0.20 | 0.235 (+18 %) | — |
| 0.30 | 0.302 / 0.313 (+2 %) | 0.299 |

The **population** median stays at init (the write only touches the few hundred atoms it
uses — 191–761 of 1024–8192 atoms carry amplitude > 1 % of max), but the atoms that
**form the wells** are pushed **outward** from a narrow init and left alone from a wide one.
The write is therefore already exerting pressure toward ≈0.19–0.35 — consistent with §4's
mechanism, and it means the flag delivers only a **1.7× narrowing, not 2×**.

---

## 4. ⭐⭐⭐ The actual binding constraint: a well-width ↔ payload-excursion REACH condition

The read launches every query on the **payload = 0 manifold** (the anti-decoration guard),
a distance `|a_i|` along the payload channel from the stored target `z_i = (c_i, a_i)`, with
`a_i ∈ linspace(−1, 1, K)`. A Gaussian well of width `s` and depth `D` exerts force
`D·r/s²·exp(−r²/2s²)` at radius `r`, which **collapses super-exponentially in `1/s`**. A
narrow well is deep and sharp but *invisible from the launch manifold* for large `|a_i|`.

Per-item probe (`mech_reach.py`, d=4 K=16, seed 0, same write, per-item strict split at the
median `|a_i|`):

| width | trained `w_atom` | overall strict | strict, **small \|a_i\|** | strict, **large \|a_i\|** | corr(strict, \|a_i\|) | **‖∇V‖ at the launch manifold** (median over items) |
|---|---|---|---|---|---|---|
| **0.15** | 0.184 | 0.586 | **1.000** | **0.172** | **−0.887** | **0.171** |
| **0.30** | 0.301 | 0.947 | 1.000 | **0.895** | −0.508 | **0.881** |

The force the store can exert on an arriving query **collapses 5.1×** (0.881 → 0.171) when
the well is narrowed 1.63×, and the damage is entirely in the outer half of the codebook
(0.895 → 0.172); the inner half is untouched (1.000 → 1.000) at both widths.

**At width 0.15 the store is not broken — it is broken *only for items stored far from the
launch manifold*.** Items whose payload sits near 0 retrieve **perfectly** (1.000); items in
the outer half of the codebook retrieve at 0.172. That is not a capacity failure, an
addressing-interference failure, or an atom-budget failure. It is a **reach** failure, and
it is why `basin ≡ strict` at every failing cell (a query that is never pulled along the
payload channel also drifts off its address, so it is scored as a wrong basin).

**Corollary test (pre-specified by the mechanism, run to confirm it).** If reach is the
cause, then halving the payload excursion must restore the narrow width. Halving the
codebook **and the tolerance together** (`payload_tol 0.1 → 0.05`, so the required precision
stays the same *fraction* of the range — the task is not made easier):

| d=4 K=16 | width | payload scale | payload_tol | trained `w_atom` | sep/w | **strict** | basin | payload err |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.15 | 1.0 | 0.10 | 0.184 | 4.90 | **0.586** | 0.586 | 0.347 |
| **reach restored** | **0.15** | **0.5** | 0.05 | 0.182 | 4.95 | **1.000** | 1.000 | **6.4e-5** |
| control | 0.30 | 0.5 | 0.05 | 0.282 | 3.20 | 1.000 | 1.000 | 6.7e-5 |

**Halving the payload excursion takes the narrow-well store from 0.586 to 1.000 — a perfect
retrieval — with the well width, the atom budget, the site geometry, the write objective and
the read schedule all unchanged, and the required precision held at the same fraction of the
codebook range.** The payload error falls by **5400×** (0.347 → 6.4e-5). The narrow well was
never the problem; the distance it had to reach was. This is the mechanism, confirmed by
intervention rather than inferred from a correlation.

### 4.1 ⭐⭐ The w23 "capacity wall" at d=4 is a reach wall — it moves when the excursion moves

Same probe, walking K at the halved excursion (1 seed, written-only, 8192 atoms, d=4):

| d=4, payload scale **0.5**, tol 0.05 | K=16 | K=32 | **K=64** |
|---|---|---|---|
| width **0.15** | **1.000** (sep/w 4.95) | **1.000** (3.95) | **0.9922** (3.25) |
| width **0.30** | **1.000** (sep/w 3.20) | **1.000** (2.25) | **0.9897** (1.94) |
| *the same cells at FULL excursion (width 0.15 / 0.30)* | 0.586 / **0.937** | 0.600 / **0.824** | — |

**K = 32 at d = 4 is the cell w23 established as a firm wall** — flat at 0.825–0.840 across a
**16× atom sweep** (2048 → 32768) and reproduced here at 0.824. With the read-out excursion
halved it goes to **1.000 with a payload error of 8e-5**. **K = 64 — four ladder rungs above
the w23 wall of 16 — clears the 0.9 bar at 0.9922 / 0.9897.** At *both* widths, so this is
not a width effect at all: **it is the excursion.**

Two sanity checks that this is not an artefact:
- **Halving the payloads makes the packing marginally HARDER, not easier**
  (`|z_i−z_j|² = |c_i−c_j|² + (a_i−a_j)²`, and the second term shrinks), and the write
  objective's payload perturbation `σ_pay = 0.6` is unchanged. Nothing about the address
  problem was made easier.
- The gain is in **`basin_ok`**, which is address discrimination (basin ≡ strict at 0.9922),
  not in a loosened value test. *(Footnote: at K=64 the payload tolerance exceeds the codebook
  spacing — 0.05 vs 0.016 — so the value test is weak at large K. That is a property of the
  **shipped** criterion at every pscale, including w23's, not something introduced here; the
  discriminating quantity at these K is the basin.)*

⚠ **Fairness caveat, stated up front and not negotiable.** This is a **different task point**,
not a capacity win. The tolerance is scaled with the codebook so the *relative* precision is
unchanged, but **this harness's payload channel carries no read-out noise** (queries launch at
exactly `payload = 0`), so shrinking the payload excursion is free *here* in a way it would not
be under a payload-noise model. **No `K_learned` number may be quoted at pscale ≠ 1**, and the
R2 law is unchanged by this section. What it establishes is *mechanism*: the binding variable
is the excursion, and the w23 walls are on its far side.

**The design rule this yields** (and it is a rule about the primitive, stated with no
comparison to anything): **the well width is set by the read-out channel's excursion, not by
the address-space packing.** `s` must satisfy *both*
`s ≳ σ_q` (the basin must contain the jittered query — the floor the theorist identified)
**and** `s ≳ |a|_max / κ` with κ = O(3) (the basin must be visible from the launch
manifold — the ceiling-side condition nobody had written down). With `|a|_max = 1` the second
condition binds at `s ≈ 0.3`, which is exactly where the shipped default sits and exactly
where the write pushes any narrower init. **There is no room underneath it, so there is no
width route to R2.**

---

## 5. What this does to the two competing accounts

| account | status after this task |
|---|---|
| **N92/N96 "a d-independent write-operator ceiling at `K ≈ 32`"** | **weakened, and its *stated mechanism* is wrong.** A pure optimizer/representation ceiling cannot produce a critical `sep/width` stable to 1.36× across d — a perfect PASS/FAIL classifier at d ≥ 4 — and cannot produce a failure that is perfectly correlated (−0.89) with the *stored value* rather than with the address. The measured object is not d-independent and not "the write operator." |
| **`lattice-capacity-theory` §4.2/§5.0 "width-locked geometry"** | **its correlational prediction SURVIVED (§1) and its causal prediction was REFUTED (§2).** Widths are not locked (they drift 0.215–0.364, so every ratio the account quoted at a fixed 0.30 is mis-stated), the criterion is sharp only at d ≥ 4, and forcing the ratio to 4.90 — deep in the "safe" zone — destroys retrieval. Sep/width is a **proxy** for the real variable, not the real variable. |
| **⭐ the reach condition (this task)** | the failure is **width ↔ payload-excursion**, measured directly (§4). It explains *both* other accounts' partial successes: sep/width correlates with the wall because both the numerator (packing) and the denominator (the write's preferred width) drift smoothly with d; and the ceiling looks d-independent because `|a|_max = 1` is d-independent. |

---

## 6. Laundering control (dial declaration) — the designed write at matched geometry

Re-measured on this harness at d=4, seed 0, value-blank controlled, at **both** values of the
Stage-1 flag:

| arm | K=32 | K=128 | K=256 | `K_designed(4)` |
|---|---|---|---|---|
| designed, `atom_init_width` **0.15** | 1.0000 | 0.9971 | 0.8577 | **128** |
| designed, `atom_init_width` **0.30** | 1.0000 | 0.9971 | 0.8577 | **128** |

Two things this control establishes:
1. **The designed write keeps reaching its own wall**: `K_designed(4) = 128`, exactly the w23
   value, on this harness, in this session. The task's lever did not quietly degrade the
   reference arm.
2. **The Stage-1 flag is provably inert on the designed arm** — the two rows are **identical
   to every printed digit** (1.0000 / 0.9971 / 0.8577), as they must be:
   `build_designed_model` reads `well_width`, `well_depth`, `payload_kappa`, never
   `atom_init_width`. **PREREG S1.6 ✅.**

**No knob in this task made the learned write more designed** (N46): the write still
receives only the target sites, exactly as in w20–w24, and every center/width/amplitude is
still learned. The *negative* result cannot be a scope collapse — nothing was won.

---

## 7. How I verified — commands, and what I did NOT run

```
# Stage 0 (12 cells launched, 11 completed + 1 killed, 1 seed each, written landscape only)
PYTHONPATH=../CHLU-r2geom .venv/bin/python .../stage0_width_dump.py <d> <K> <mult> 0 out.json
# Stage 1 explore (1 seed, written-only) / confirm (3 seeds, shipped evaluate_arm_cell)
.../stage1_explore.py <d> <K> <width> <base_mult> <seed> out.json
.../stage1_cell.py learned_global 4 16 {0.15,0.30} 4 0,1,2 out.json
# laundering control (designed arm, no training)
.../stage1_cell.py designed 4 {32,128,256} {0.15,0.30} 4 0 out.json
# mechanism + corollary
.../mech_reach.py 4 16 {0.15,0.30} 4 0 out.json
.../payload_scale.py 4 {16,32,64} {0.15,0.30} 4 {0.5,1.0} 0 out.json
# tests
pytest tests/test_designed_mechanism.py tests/test_config.py
    -> 17 passed (431 s, contended machine); ruff check clean on both touched files
```
All raw JSON + logs: `.claude/scratch/r2-geometry-revival/` (`s0_*` Stage 0, `e1_*` Stage-1
explore, `c1_*` Stage-1 confirm + designed control, `m_*` mechanism, `p_*` payload-scale);
aggregator `summarize.py`.

- **Harness-integrity checks that passed** (these are what license the comparison):
  `d=4 K=32, width 0.30, 8192 atoms → 0.8242` vs **w23's saturation row 0.825–0.840 over a
  16× atom sweep**; `d=4 K=16, width 0.30, 3 seeds → 0.9368` vs **w23/multi-seed-w23's
  0.928** at 2× atoms. The width-0.30 path reproduces w23 to within seed noise, so the
  width-0.15 collapse is the flag and nothing else. The Stage-1 explore and confirm paths
  agree bit-for-bit at seed 0 (0.5859 both).
- **`ruff check` clean** on both touched files. `ruff format --check` reports drift in
  `exp_designed_mechanism.py` and `tests/test_designed_mechanism.py` — **pre-existing**, in
  hunks I never touched (same finding as w23/w24); my added lines produce no format diff, and
  per protocol §3.3 I did not reformat out-of-scope shared code.
- **NOT RUN, declared:** (a) **Stage 2** — correctly gated off by the Stage-1 falsifier;
  (b) **S1.2 at d=6 and any d=8 Stage-1 cell** — the theorist's coverage spec is
  `512·2^d` atoms (32768 at d=6, 131072 at d=8) and the measured write cost is ~1340 s per
  write per seed already at d=8/16384 atoms; on a machine shared with other spokes this was
  out of budget, and the d=4 refutation is at the same flag with a d-independent mechanism.
  This deviation was **registered in PREREG before Stage 1 ran**; (c) Stage-0 cells are
  **1 seed each** — the widths are a smooth function of (d, K) and no falsifier was close,
  but the Stage-0 table carries no error bars; (d) `d=4 K=128 @ width 0.15`, `d=4 K=64` and
  the Stage-0 cell `d=8 K=32` were started and **killed** to free cores (the first two once
  K=16 had already failed; the third once d=8 K=64 had answered S0.3) — the machine was
  shared with another spoke throughout and every kill is logged here rather than silently
  dropped; (e) **the designed arm at halved excursion** (§10.3).

---

## 8. Proposed new negative result (tier A, memory-architecture)

> **N⟨next⟩ — the learned-capacity ceiling is a READ-OUT REACH condition, not a width-locked
> packing wall and not a d-independent write-operator ceiling.** Narrowing the atom
> dictionary's wells (`atom_init_width` 0.30 → 0.15, trained width 0.31 → 0.185) at the
> theorist's coverage-raised budget does **not** raise `K_learned(4)` from 16 toward
> `2^d`-many; it **drops it below 16** (strict 0.9368 ± 0.0133 → 0.5944 ± 0.0164, 3 seeds,
> value-blank controlled, flat under a 2× atom re-check). The failure is **not** in the
> address geometry: the failing cell sits at `minsep/width = 4.90`, well *inside* the
> geometrically safe zone. It is per-item and perfectly ordered by the **stored value**:
> items with small `|a_i|` retrieve at **1.000**, items with large `|a_i|` at **0.172**
> (corr −0.887). A Gaussian well of width `s` exerts force `∝ D r/s² exp(−r²/2s²)` at the
> payload = 0 launch manifold, so the width is bounded **below** by the query noise
> (`s ≳ σ_q`) and **above the useful range** by the read-out excursion (`s ≳ |a|_max/κ`,
> κ≈3). The shipped default 0.30 sits at that second bound, and the write pushes any
> narrower init back toward it (+23 % from 0.15). **There is no width route to R2**, and the
> w23 walls themselves are on the excursion's far side: **halving the read-out excursion
> (with the tolerance halved to hold the relative precision) takes d=4 K=32 from 0.824
> (w23's firm wall, flat over a 16× atom sweep) to 1.000, and d=4 K=64 — four rungs above
> `K_learned(4)=16` — to 0.9922, at BOTH widths.** ⚠ That last clause is a **mechanism
> result at a different task point, not a capacity number**: this harness's payload channel
> has no read-out noise, so shrinking the excursion is free here in a way it would not be
> under a payload-noise model. **No `K_learned` may be quoted at pscale ≠ 1.** The
> `lattice-capacity-theory` §4.2 `sep/width` ratio is a real *correlate* (critical value
> stable to 1.36× across d=2…8 and a 16× K range, and a perfect PASS/FAIL classifier at
> d ≥ 4 with a boundary in (2.30, 2.49); trained widths 0.215–0.364, never below 0.18) but **not the causal variable**; and N92/N96's "d-independent write-operator
> ceiling" is not the mechanism either.

---

## 9. Git footprint

- Branch **`agent/experiment-engineer/r2-geometry-revival`**, base local `main` @ `63c668d`
  (verified: `main` still at `63c668d`, working tree clean). **Not pushed.** Rebase onto
  `main` = no-op (base == current `main`).
- Worktree `../CHLU-r2geom` — **removed at the end, but only after verifying from the main
  repo that the shared ref had advanced** (`git -C CHLU log main..agent/…` → `f770656`), per
  the protocol §3.2 wave-4 lesson. Two other spokes' worktrees (`CHLU-cl-entry`,
  `CHLU-shard`) were present throughout and a concurrent spoke was observed running
  `run_cells.py --stage 0` in the same venv — no file-level collision; I throttled and
  suspended my own jobs repeatedly to share the 8 cores, which is why several wall-clock
  numbers in the logs are 3–6× the solo cost.
- Commit (1): **`f770656`** — *"add trained_well_widths dump for the §5.0 geometry check"*.
  Files: **M** `chlu/experiments/exp_designed_mechanism.py` (+1 function, `trained_well_widths`),
  **M** `tests/test_designed_mechanism.py` (+2 tests). **No config knob added** — Stage 1
  needed none (`atom_init_width`, `min_atoms_base` already exist and are already registered
  at all four config sites), so no `save_config` enumeration change was required.
- Tests: `tests/test_designed_mechanism.py` **10 passed** (was 8) and `tests/test_config.py`
  **7 passed** — **17 passed** together, re-run after every edit. No other test touched.
  Verified from the main repo after finishing: `main` is still at `63c668d` with a clean
  working tree, and `git log main..agent/experiment-engineer/r2-geometry-revival` shows
  `f770656` (protocol §3.2 — the shared-ref check that lost 8 commits in wave-4).
- All drivers/results/figures are untracked, under `.claude/scratch/r2-geometry-revival/`
  and `.claude/outputs/r2-geometry-revival/`.

## 10. Open questions / follow-ups / risks

1. **⭐⭐ The real R2 lever is the read-out excursion, and the fair version of the experiment
   is now well-posed.** Measured here (§4.1): at half excursion, d=4 K=32 → 1.000 and
   K=64 → 0.9922. But this harness's payload channel is **noise-free**, so shrinking the
   excursion costs nothing *here*. The fair successor experiment has to restore that cost,
   and there are exactly two clean ways: **(a)** add read-out noise `σ_a` to the payload
   channel and hold `|a|_max/σ_a` fixed while sweeping the excursion — then any surviving
   capacity gain is real; **(b)** keep `|a|_max` and add **payload channels** (`m` channels of
   excursion `|a|_max/√m` carrying the same total value precision), which lowers the per-axis
   reach demand without lowering the information content. **(b) is the one I would run**: it
   is a change to the *primitive's* read-out geometry, tests the same mechanism, and cannot be
   dismissed as an easier task. Neither was run here.
2. **Risk to that idea, stated up front:** rescaling payloads while holding `payload_tol`
   fixed would be **laundering** — it makes the read task easier. My probe rescales the
   tolerance by the same factor, which holds the *relative* precision but does **not**
   simulate read-out noise, so it is reported as mechanism only. **No `K_learned` may be
   quoted at pscale ≠ 1.** A second caveat for whoever runs the successor: at large K the
   shipped `payload_tol` already exceeds the codebook spacing, so the value test is weak and
   the basin term carries the discrimination — the successor should tighten `payload_tol` with
   K, or report basin and value separately.
3. **The designed arm was NOT re-measured at half excursion**, so the designed/learned tax at
   that task point is unknown; the "learned reaches 64 at d=4" number therefore cannot be
   turned into a closed prefactor-gap claim (`4·2^4 = 64` is a coincidence of arithmetic
   until the designed arm is run at matched excursion).
3. **Stage-0 error bars.** Every Stage-0 cell is 1 seed. The verdict has large margins, but a
   3-seed repeat of the two decisive cells (d=6 K=64, d=8 K=64) would tighten S0.3.
4. **d=6/d=8 Stage 1 unrun.** If the Hub wants the negative closed at a second dimension, the
   cheapest decisive cell is **d=6 K=32, width 0.15 vs 0.30, 8192 atoms, 3 seeds** (~2 h at
   current machine load), not the full ladder.
5. **The 0.30 default is now load-bearing** and should be documented as such (§3 of the
   handover): it is near a measured optimum, not an arbitrary init. I did **not** change the
   default (preserve current behaviour unless told).

## Proposed handover updates (for the Hub)

1. **§6 / claims — the mechanism question raised in N92/N96 is SETTLED, and neither
   candidate won.** Replace "the ceiling is the write operator (d-independent, ~32)"
   *and* any width-lock wording with the **reach condition** (§4/§8). The CONTESTED flag on
   N92/N96 can be cleared with that third answer.
2. **`lattice-capacity-theory` §5.2 → REFUTED BY EXPERIMENT** (§2), and §4.2 →
   "surviving correlate, refuted as cause" (§1 + §2). §5.0's own falsifiers did **not** fire;
   it was the §5.2 causal test that killed the account, exactly as the staged design intended.
   Both stages behaved as pre-registered — this is the pre-registration rule paying off twice
   in one task (the theorist's S1.1–S1.3 **and** my own S1.4 were refuted).
3. **§3 config note:** `experiment_designed_mechanism.atom_init_width = 0.30` is **not**
   an arbitrary initialisation — capacity is monotone in it over 0.15→0.30 and it sits at the
   measured reach bound. Default **unchanged** by this task.
4. **New helper in the codebase:** `trained_well_widths(V, targets)` in
   `chlu/experiments/exp_designed_mechanism.py` — the §5.0 measurement, reproducible from the
   repo rather than from scratch scripts.
5. **⭐ Commission the fair excursion experiment (§10.1(b), multi-channel payload).** This is
   the one thing in this report that could move R2, and it needs a Hub decision on whether the
   read-out excursion counts as a design parameter of the primitive before anyone spends
   compute on it.
6. **Gate on R2:** unchanged from w24 — it is **not** unclamped; and the width route is now
   closed. R2's figure remains `min(2^d, ceiling)` with the ceiling **re-attributed** (§8).
   Precision rule intact: designed `4·2^d`, learned `2^d`, a **4× prefactor gap**; never
   "exactly the designed rate".
