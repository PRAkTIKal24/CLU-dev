# PREREG — `pilot-placement-probe`

**Filed BEFORE any probe harness was built or run** (protocol §5 pre-registration rule; the
acceptance criterion is a measured set of ratios/levels, so predictions are committed here first).
The two B′ riders (§0 of the task) are pure re-aggregation / a labelled frontier row and carry no
prediction; they are excluded from this document by design and their provenance lives in the report.

Author: `experiment-engineer`. Branch `agent/experiment-engineer/pilot-placement-probe`,
worktree `../CHLU-placement`, base local `main @ 29fc22b`.

---

## 0. DIAL DECLARATION (echoed from the task, binding)

- **Dial / pillar:** none — **instrument/diagnostic** (tier-iii mechanism isolation). ⛔ Nothing here
  is a paper number. It is config evidence for the CSF3 run and design evidence for tier ii.
- **Control:** the pilot's own arms, paired seeds — **live vs blank vs memory-deleted**, plus the
  in-block acquisition self-probe against its own chance level.
  ⭐ **The laundering control that must run beside every acquisition number:** the **blank store with
  the SAME localized init**. Localizing 128 atoms per group near K sites makes the read relax toward
  the nearest group centre *whether or not anything was written*, so a self-probe hit can be bought
  by the initialisation alone. `acq(live, localized) − acq(blank, localized)`, paired per seed, is
  the only acquisition quantity I will treat as evidence of a **write**.
- **Falsifies (this probe's own hypotheses):** H1 fails if `atom_local_radius` at its designed band
  leaves acquisition at chance AND live = blank at float32. H2 same bar. Both failing ⇒ the placement
  hypothesis is refuted at toy scale — a FINDING that re-prices, never closes, the route (A18.4).
- **Does NOT falsify:** anything about tier ii or the CSF3 scale run; toy-scale GRU superiority
  (already measured, pre-registered); monitor #13 / N94 non-promotability is inherited
  (**4 inner write steps vs floor 40**) and is quoted beside every reading.

---

## 1. The rig (fixed before the run; every number below is at this configuration)

`exp_cluformer_pilot.TOY`, unchanged: `d_model 64`, `n_layers 2`, `seq_len 512`, `batch 4`,
`chunk 32`, store `dim 3` (`addr_dim 2` + `payload_dim 1`), `capacity 8`, `budget 6`,
`n_atoms 1024` (`atoms_per_item 128`), `atom_width 0.3`, `atom_init_scale 1.0`,
`atom_depth_init 1e-4`, `confine 0.05`, `ball_radius 1.0`, `query_sigma 0.15`, soft certificate ON
(`ζ = 0.6` ⇒ `d_safe = 0.3895`, `sep_expected = 0.6492`), read `24 + 24 (+24 gated retry)` Verlet
steps at `dt 0.05`, `γ_address 0.05`, `γ_read 0.02`, write **4 inner sign-SGD steps at lr 0.05**,
`write_n_perturb 8`, φ-gain calibrated per seed by the shipped rule (RMS address norm = ball radius).
Seeds **0, 1, 2** on every cell (paired: identical φ, identical data order, identical plan pass).

**The designed band for `atom_local_radius` (N98):** N98's own fix is *"a ball of radius ≈ 2s around
the site"*, measured at `radius = 2·atom_width = 0.6`. At this rig `atom_width = 0.3`, so the
**designed band is `r = 0.6`**, and I register `r ∈ {0.3, 0.6}` (1s and 2s) as the primary pair with
`r = 0.15` (0.5s) as an extra. ⚠ `d_safe = 0.3895 < 0.6`, so at the designed radius a group's ball
overlaps its neighbour's — declared in advance, not discovered.

**The designed band for `λ_traj` (C2W2 `traj-write-objective`):** `{0.03, 0.3, 3, 30}`. I register
`{0.3, 3}` as the primary pair (the middle of the measured grid).

---

## 2. The mechanism I am predicting from (stated so the prediction is falsifiable, not fitted)

Sign-SGD moves **every unmasked parameter by exactly `lr` per step**, so 4 steps move an atom's `amp`
by at most `4 × 0.05 = 0.20`. With `amp₀ = √1e-4 = 0.01`, one atom's depth ceiling after a write is
`A = amp² ≤ (0.21)² = 0.0441`.

⭐ **This reproduces the pilot's saturation to 2 significant figures: the measured
`D = 0.0331 / 0.0448 / 0.0442` at 4/16/64 inner steps is "ONE atom's worth of amplitude at the
site".** With 128 atoms scattered as `N(0, 1)` in a `dim = 3` ball, the Gaussian weight
`exp(−d²/2s²)` at `s = 0.3` is `~1e-7` for a typical atom (`d ≈ 1.7`) and `O(0.5)` only for the
nearest one or two. So depth is not step-limited — it is **atom-count-limited at the site**, which is
exactly the placement hypothesis.

**Therefore the quantitative prediction for H1 follows from arithmetic, not from hope.** Localizing a
group's 128 atoms in a ball of radius `r` around a site puts a mean distance `E‖x‖ = (3/4)r` on the
address axes; at `r = 0.6` that is `0.45`, giving weight `exp(−0.45²/(2·0.3²)) = 0.325`, and the
depth at the site becomes `~128 × 0.0441 × 0.325 ≈ 1.8` — with a large downward correction for the
un-localized **payload** axis (N46: localization is address-axes-only by construction, so the payload
coordinate keeps its `N(0,1)` scatter and contributes its own `exp(−a²/2s²)` factor, `~0.1` in
expectation).

---

## 3. REGISTERED PREDICTIONS (point value, band, and probability)

Success signal, verbatim from the task (all three reported per arm; partial wakings are the
informative outcome and will not be binarized):
**(a)** in-block acquisition self-probe off chance (`> chance + 2·SE`, 3 paired seeds);
**(b)** live ≠ blank at float32 resolution; **(c)** well depth leaves the 0.045 saturation toward the
shipped `D = 0.46–0.80` band.

| # | quantity | registered prediction | band | P |
|---|---|---|---|---|
| **P1** | **(c)** median fitted well depth `D`, H1 at `r = 0.6` | **1.0** | `[0.3, 4.0]` — i.e. **leaves the 0.045 saturation and reaches or overshoots the shipped 0.46–0.80 band** | **0.85** |
| **P2** | **(c)** `D` at `r = 0.3` | **2.5** | `[0.5, 8.0]` (tighter ball ⇒ deeper) | 0.75 |
| **P3** | **(b)** live-vs-blank held-out NLL gap, H1 at `r = 0.6` | **> 1e-3 bpc** (i.e. NOT float32 round-off) | `> 1e-5` | **0.85** |
| **P4** | **(a)** live acquisition at H1 `r = 0.6`, vs its own chance (`0.167`) | **at chance** ⇒ **H1 FAILS criterion (a)** | I predict `acq − chance < 2 SE` | **0.65** |
| **P5** | ⭐ the **laundering control**: `acq(blank, localized) > acq(blank, scatter)` | **YES — the localized init buys self-probe hits with NO write** | — | **0.70** |
| **P6** | ⭐ the **paired write effect** `acq(live) − acq(blank)` at the same init, H1 `r = 0.6` | **≈ 0** (`|Δ| < 0.1`) | | **0.60** |
| **P7** | H2 (`λ_traj ∈ {0.3, 3}`) alone: acquisition | **at chance** (C2W2 measured a monotone COST for this term; nothing in it addresses atom count at the site) | | **0.80** |
| **P8** | H2 alone: depth `D` | **unchanged vs baseline within ±50 %** (`[0.02, 0.10]`) | | 0.70 |
| **P9** | H1+H2 interaction on acquisition | **no rescue** — if H1 and H2 each fail (a), the combination fails (a) | | 0.75 |
| **P10** | baseline (scatter, `λ_traj = 0`) reproduces the pilot | `D ∈ [0.02, 0.07]`, `acq = chance`, live−blank ≤ 1e-6 | | **0.90** |
| **P11** | monitor #5 (addressing) trips on **every** arm | trips 3/3 seeds everywhere | | 0.70 |
| **P12** | ⚠ the certificate monitor #8 gets **worse** under H1 | `sep/σ_q` unchanged (it is a φ property) but `λ_min` degrades / basins merge at `r = 0.6 > d_safe = 0.3895` | | 0.55 |

**⭐ The pre-registered headline verdict, committed now:** I predict the probe returns a
**PARTIAL WAKING — depth moves by 1–2 orders, acquisition does not** — and therefore that the
placement hypothesis is *the correct diagnosis of the depth failure* but **not sufficient for
retrieval**, because the remaining binding constraint is the **payload channel** (localization is
address-only by N46, and 4 sign-SGD steps move a centre by ≤ 0.2 in a payload coordinate drawn from
`N(0,1)`). If that is what the numbers say, the CSF3 recommendation block must NOT simply switch
`atom_local_radius` on and declare victory.

**The competing hypothesis, registered as the alternative so the test is two-sided:** *localization
is sufficient* — depth AND acquisition both move, `acq − chance > 2 SE` on 3/3 seeds, and the store
wakes. I assign this **P = 0.25**. If it happens, it is the wave's headline and the CSF3 config
changes materially.

**The third outcome, also registered:** *nothing moves at all* (depth stays ≈ 0.045 under
localization). **P = 0.10.** This would refute my own §2 arithmetic and would mean the write is not
the binding path either — I would report it as a refutation of the mechanism story, not as noise.

---

## 4. Declared NOT-RUNs (never reported as nulls)

- ⛔ **The ψ payload residual** (the pilot's third §5.3 candidate) — task-declared third priority,
  built only if H1 AND H2 both fail and budget remains.
- ⛔ **Any 26–47 M / CSF3 run.** Not reachable from this machine (the pilot pre-registered the DNS/VPN
  failure); this probe is TOY scale only and no number of it is a pilot-scale number.
- ⛔ **The full 5-arm swap table at each probe cell.** The probe's controls are live/blank/
  memory-deleted; the matched GRU/TTT swap is the *pilot's* protocol and is not re-run per cell.
- ⛔ **`atom_init_width` (the N111 substitute lever).** N111 shows width and localization are
  substitutes; moving both would confound the probe. Width is pinned at the shipped `0.3`.
- ⛔ **WikiText-103**, multi-layer/deeper stacks, larger `capacity`.

## 5. Budget declared in advance

≤ 6 h local wall clock, 1 worktree, 0 CSF3 hours. Cut order if exceeded: the `r = 0.15` extra → the
H1+H2 interaction row → the trained tier (§ screen tier alone still answers (a) and (c)).

---

# ADDENDUM 1 — **H1b, localized placement AT WRITE** (filed before the H1b cells ran)

**Why an addendum rather than a silent addition.** Two smoke measurements of the registered H1
(`baseline` and `h1_r0.6`, seed 0) exposed a structural mismatch that §3 did not anticipate and that
changes what "H1" can possibly mean in a *streaming* block:

> `atom_local_radius` as N98 ships it is a **static init** localization around targets that must be
> fixed **before the stream starts**. In the shipped store harness those targets are the designed
> item sites and are known. In the streaming block, **the site an item will occupy is chosen by the
> controller when the chunk arrives** — so localizing group *j* around the φ-image of calibration
> chunk *j* localizes it around a point the item slot *j* later holds has no reason to be near.
> Smoke reading (seed 0, `r = 0.6`): median fitted depth **8.1e-08**, i.e. *worse than the scattered
> baseline's 0.0131* — the localization moved the group's atoms AWAY from the item.

That is itself a finding and is reported. But it means the registered H1 cannot test the pilot's
actual hypothesis ("atoms seeded near the φ-image of **the chunk**"), so I register its streaming
form now, with predictions, before running it:

**H1b — `atom_place_radius`:** at write time the written slot's atoms have their **address**
coordinates re-drawn into a ball of radius `r` around the **incoming chunk's own address** (payload
coordinates untouched — N46). Fixed key-free offset jig ⇒ no parameter, no state byte, C3-local,
bit-identical when off, and a refused offer still leaves `V_θ` bit-identical.

| # | quantity | registered prediction | band | P |
|---|---|---|---|---|
| **A1** | **(c)** median depth `D`, H1b at `r = 0.3` | **1.26** (derived: `128 × 0.0441 × 0.287 (payload factor) × 0.78 (address factor)`) | `[0.3, 4.0]` — **in or above the shipped 0.46–0.80 band** | **0.85** |
| **A2** | **(c)** median depth `D`, H1b at `r = 0.6` | **0.60** (same arithmetic, address factor 0.37) | `[0.15, 2.0]` | 0.80 |
| **A3** | **(b)** live-vs-blank read-output delta, H1b | **≥ 10× the baseline's** | | 0.85 |
| **A4** | **(a)** acquisition at H1b `r = 0.3` | **0.35 vs chance ≈ 0.30 — NOT off chance by 2 SE** ⇒ H1b still fails (a) | | **0.60** |
| **A5** | the paired write effect `acq(live) − acq(blank)` under H1b | `+0.05`, not > 2 SE | | 0.60 |
| **A6** | ⭐ **the mechanism claim being tested by A4:** the residual binding constraint after placement is the **PAYLOAD channel** — 4 sign-SGD steps move a centre by ≤ 0.2 in a payload coordinate whose scatter is `N(0,1)`, so the well's payload location is ≈ the group's mean atom payload (≈ 0 for every item) and every item decodes to the same stored payload. **Falsifier:** if H1b's acquisition DOES come off chance, this mechanism claim is wrong and the report says so. | | 0.60 |
| **A7** | H1b × H2 (`r = 0.3`, `λ_traj = 0.3`) | **no rescue of (a)** beyond H1b alone | | 0.70 |

**Registered headline for the addendum, committed:** H1b produces the **partial waking** —
`(c)` YES by 1–2 orders, `(b)` YES, `(a)` NO — and the CSF3 recommendation must therefore turn
placement ON **and** name the payload channel as the remaining open mechanism, rather than declare
the store woken.

---

# ADDENDUM 2 — the **write-budget × placement interaction** (filed before those two cells ran)

**Why.** The pilot refuted the write budget as the cause of inertness — but it did so **at the
scattered init** (4 → 16 → 64 inner steps buys 1.33× depth and zero acquisition). §2's arithmetic
says why: sign-SGD's amp ceiling after `n` steps is `(0.01 + 0.05n)²`, so 64 steps affords a per-atom
depth of **10.4** and the measured depth is **0.044** — i.e. the budget was never the binding
constraint *because the gradient at a far atom is ≈ 0 and `sign(0) = 0`*. The CSF3 config question is
therefore not "budget or placement" but **whether they are complements**: once the atoms are at the
site the gradient is non-zero, and steps should compound.

Two cells, registered now: `baseline_w40` and `h1b_r0.3_w40`, at `write_inner_steps = 40` =
**monitor #13 / N94's maturity floor** (so these are the only readings in the probe that #13 does
not demote).

| # | quantity | registered prediction | P |
|---|---|---|---|
| **B1** | `baseline_w40` depth | **unchanged at ≈ 0.045** (reproducing the pilot's 64-step 0.0442 — the budget is inert at the scattered init) | **0.85** |
| **B2** | `h1b_r0.3_w40` depth | **≥ 0.5** — i.e. **reaches or exceeds the shipped 0.46–0.80 band**; point estimate **2.0** | **0.70** |
| **B3** | ⭐ the interaction itself | `depth(h1b, w40) / depth(h1b, w4)` **≥ 5×**, while `depth(base, w40) / depth(base, w4)` **≤ 2×** ⇒ **placement and budget are COMPLEMENTS, and the pilot's "the budget is not why" is scoped to the scattered init** | **0.70** |
| **B4** | acquisition at `h1b_r0.3_w40` | ⭐ **the one cell where I give (a) a real chance: `acq − chance > 2 SE` with P = 0.45.** 40 steps move a centre by up to 2.0 in the payload coordinate, which is the first budget that can actually place a payload well. **This is the falsifier of ADDENDUM 1's A6 mechanism claim** — if acquisition still sits at chance here, the payload channel is not merely under-budgeted, it is structurally unreachable by this write. | 0.45 |
| **B5** | `acq(live) − acq(blank)` at `h1b_r0.3_w40` | `> 0` and larger than at any other cell | 0.55 |

---

# ADDENDUM 3 — **the hinge margin is the depth ceiling** (filed before the margin cells ran)

**The observation that forced it.** With `h1b_r0.3_w40` (atoms placed at the site AND N94's 40-step
budget) the measured well depth is **0.0426 / 0.0832** — statistically the same as `h1b_r0.3_w4`
(0.0533 / 0.0629) and the same as `baseline_w40` (0.0519 / 0.0432). ⛔ **ADDENDUM 2's B2 and B3 are
refuted:** placement and budget do **not** compound, and 10× the budget buys nothing at either init.
Depth saturates at **~0.05–0.14 in every configuration measured, across a 10× budget range and a
3-order-of-magnitude range of atom placement.**

**The mechanism that explains ALL of it, stated before the test.** `write_loss`'s minimum term is a
**relu hinge**: `l_min = relu(V(z) − V(z + δ) + margin)`. It asks the well to be `margin` deeper than
its own neighbourhood and **nothing more** — once `V(z) + margin ≤ V(z + δ)` for the sampled
perturbations, the loss is exactly 0 and the gradient dies. At this rig `write_margin = 0.15`.
⭐ **The measured saturation (0.05–0.14) IS the margin (0.15).** Neither the budget nor the placement
was ever the binding constraint on DEPTH; the *objective's own hinge* is. The shipped store escapes
only because it optimises all items jointly for 300 Adam steps, where the barrier/crowding terms keep
pushing after the per-item hinge is satisfied.

**Registered predictions** (cells: `h1b_r0.3` and `baseline` re-run at `write_margin ∈ {0.6, 1.0}`,
3 seeds):

| # | quantity | registered prediction | P |
|---|---|---|---|
| **C1** | depth scales with the margin, roughly linearly | `depth(margin) / depth(0.15) ≈ margin / 0.15` ⇒ at `margin = 1.0`, **depth ≈ 0.35–0.9**, i.e. **INTO the shipped 0.46–0.80 band** | **0.75** |
| **C2** | the same holds at the scattered init (`baseline` + margin) | depth rises there too, but **less** than under placement, because at the scattered init few atoms are close enough for the hinge to act on | 0.60 |
| **C3** | ⭐ **acquisition STILL at chance at `margin = 1.0`** | the hinge sets how DEEP the well is, not WHERE in the payload coordinate its minimum sits; the payload location is still ≈ the group's mean atom payload (≈ 0 for every item) | **0.70** |
| **C4** | ⛔ **the falsifier of the whole probe's conclusion:** if C3 is wrong and acquisition DOES come off chance once the well is deep enough, then depth was the binding constraint all along and the recommendation flips to "raise the margin" as the primary CSF3 change. | | 0.30 |
| **C5** | monitor #8 (certificates) degrades as the margin rises — deeper wells at `d_safe = 0.3895` merge sooner | | 0.60 |

⛔ **Declared CUTS (PREREG §5 cut order, applied):** `h1h2` and `h1bh2` (the two interaction rows) are
**CUT, not null** — H2 was measured bit-identical at `λ_traj = 0.3` on 3/3 seeds at 3.9× the wall
clock, and the direct gradient measurement (`~1e-13`, coefficient-linear) settles the coefficient
scaling without the cells. The budget freed pays for the margin cells above, which are decisive where
the interaction rows are not.
