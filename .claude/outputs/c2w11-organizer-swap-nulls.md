# c2w11-organizer-swap-nulls — experiment-engineer report

**Task + acceptance criterion:** land N1–N5 in their strongest registered forms on spoke A's frozen
substrate, compute `null*` over the entire registered grid, and emit V1 / V2a / V2b / V3 + L1–L4 +
the recomputed decodability ceiling — ⛔ **producing no `OD`, no swap verdict, no paper number.**
**Status: done.** All five arms landed, **594 configs × 5 score seeds** computed, **L1–L4 all pass**,
V1/V2a/V2b/V3 emitted, ceiling recomputed, **22 new tests**, **full suite 1 703 passed / 0 failed**
(§10.1 — ⚠ run 1 had 19 failures from a real defect of mine; both runs reported).

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. ⭐⭐ **THE VALUE LAUNCH KEY IS NOT FROZEN (rider R1) — and spoke B is running concurrently.**
   `FROZEN-INTERFACES-C2W11.json` freezes `launch_keys` for k0/k6_k7cap/k3_k4_k5/m6/coverage and
   registers **none** for the V-legs. I adopted `PRNGKey(7000 + seed)` (the `k3_k4_k5` key), SEEN on
   the key, `Q_unseen` on `fold_in(key, 1)`. **If spoke B used anything else, V1 and V3 are scored on
   different launches and the swap is void.** Byte-checkable: my unseen launch hash at seed 0 is
   **`3577d5233f35a473e2685bd37c7551d0`** (md5 of the float32 `(512, 4, 12)` block).
2. **The V2 novel/known split is not frozen (rider R2):** `n_novel = 4` wells by
   `default_rng(20260811 + seed)`, never written. Spoke B must use the same rule or V2 is not a swap.
3. **My oracle-imitation target is the WRITTEN, UN-ORGANIZED physics store (rider R3)** — the trained
   organizer is spoke B's. The 0.9497 below is *not* an imitation of a trained organizer.
4. ⚠ **`PREREG-C2W11.md` §5's V2 designed negative (permuted payloads ⇒ AUROC ≈ 0.5) does not bind an
   address-geometry confidence channel** and is measured **structurally invariant** on 4 of 5 arms
   (§5.2). The negative that can fail is the **label permutation**; someone must decide which one the
   physics arm is held to.
5. **`orgdiv-null-arms` §4's "0.2719 as-launched" must not be carried into C2W11 text** — the
   recomputed value on the feature-factored launches is **0.3449 ± 0.0175** (§6), and it is
   **structural, not noise** (0.3424 at `σ_q = 0`) — a **launch-head** property that belongs beside
   the C2W9 coverage trigger, not in an organizer's ledger.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** **TIER ii — the ORGANIZER SWAP's null side.** ⛔ **I am the control.** No `OD`, no
  swap verdict, no paper number, no tier-ii verdict, no full-CLU verdict is computed here.
- **Laundering control:** ⛔ no launder margin is a pass condition anywhere. My guards are (i)
  byte-compared φ + a bit-identical launch protocol (both pytest-asserted), (ii) the two-sided byte
  ledger per arm, (iii) a shuffle-φ launder beside every score, (iv) the L1–L4 anchors.
- **Falsifies:** nothing of mine — I produce `null*`. The selection guard is documented mechanically
  in §9.
- **Does NOT falsify:** an arm beating the physics arm would be a legitimate outcome, reported
  unsoftened.
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94 discipline. ⛔ **Wells are never named
  semantically:** *"Wells `{j}` are co-activated by queries whose ground-truth factor set contains
  factor `f`, with co-activation correlation `ρ = …` (95 % CI …), measured against a permutation null.
  No well is identified with any factor; the claim is a correlation between
  co-activation/wormhole/shell-position statistics and task structure."*

---

# 1. ⛔ THE MECHANICAL PRECONDITIONS — VERIFIED ON DISK, BY ME, BEFORE ANY CODE

Both files exist and were read *for content*, not existence:

| # | file | verified |
|---|---|---|
| 1 | `.claude/outputs/c2w11/GATE-SEMANTICS-C2W11.json` | **`kills_all_discriminating_passed = true`** ✅ |
| 2 | `.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json` | present; every family/launch/φ/reader/ledger/`v3_budget_grid` value below is **read from it at run time**, never re-derived (`load_frozen`, and the run records the resolved path) |

**(1)'s per-leg table, quoted:**

| leg | passed | discriminated | note |
|---|---|---|---|
| K0 | ✅ | ✅ | same harness reports 0.0378 on C2W5's offsets vs 0.9967 here |
| K1 | ✅ | ✅ | fails 3/3 at the inherited `w_frac = 1.5` |
| K2 | ✅ | ✅ | payload half 0.5 % at `m=1`, 100 % at `m=8` |
| K3 | ✅ | ⛔ **abstains** | vacuous 3/3 |
| K4 | ✅ | ⛔ **abstains** | vacuous 3/3 — the store-only leak control is uninformative |
| K5 | ❌ | ⛔ **abstains** | ⛔ **NO K5 VERDICT EXISTS IN THIS WAVE** (Head condition (d)); nothing about table-expressibility is quoted from spoke A anywhere below |
| K6 | ✅ | reported, not gating | 0.0007 |
| K7-CAP | ✅ | ✅ | SP-1 escape 1.0000 — the escape is demonstrably live |
| K8 | ✅ | ✅ | 1.0000 → 0.0625 across the structural cell |

**The gate = AND over the discriminating legs {K0, K1, K2, K7-CAP, K8} = TRUE.** ⇒ I proceeded.

## 1.1 ⚠ The payload-repair trap — closed mechanically, and it fired twice on me
`tol`, `chance` and every y-scale are homogeneous of degree 1 in `payload_radius` (1.0 → **0.60**).
Nothing is hard-coded: `assert_frozen_match` reads `family.tol` / `payload_radius` from the frozen
artifact **on every cell** and raises `FrozenCellMismatch`. It caught two real defects during
bring-up:

1. **`tol` is a per-seed sample statistic.** Frozen `0.286960063782279` is **seed 0's**; seeds 1–4
   measure 0.2828 / 0.2796 / … The guard is therefore **exact at the frozen seed** and **banded
   (±10 %) elsewhere** — the pre-repair value is **1.667×**, so the band separates "another seed" from
   "a pre-repair constant" by a factor of six. Measured `tol_ratio_to_frozen` = 1.000 / 0.9857 / 0.9744
   / … ✅
2. **φ "mismatched" because I hashed it differently.** My md5 of identical bytes disagreed with spoke
   A's sha256. Fixed by importing **spoke A's own `_phi_hash`** — a byte-comparison against a frozen
   digest is only a comparison if both sides use the same digest.

---

# 2. Flag provenance (mandatory — governs every number in this report)

**Commits** `717fd67` · `3644818` · `741af9b` · `1289c6a` on branch **`c2w11-organizer-swap-nulls`**,
rebased onto local **`main @ a47ed20`** (named base `168a892` is its ancestor). Worktree
`../CHLU-c2w11c`, **main venv reused** (protocol §4 w6 lesson): `jax 0.9.0`, float32.
**Score seeds 0–4; tune seeds 0–2.** Artifacts: `.claude/outputs/c2w11-organizer-swap-nulls/results/`.

| flag | value | source |
|---|---|---|
| `N_a / F / K / a / d / m` | 32 / 4 / 128 / 12 / 4 / 8 | frozen `family`, **asserted** per cell |
| `n_unseen` | 512 (grain **0.001953**) | frozen |
| ⭐ `payload_radius` = `atom_payload_init_radius` | **0.60** (the reach repair) | frozen; asserted |
| ⭐ `tol` | **0.286960063782279** (seed 0) | frozen; asserted |
| `chance_per_seed` | [0.0, 0.00195, 0.0, …]; mean **0.000390625** | measured, matches frozen at seeds 0–2 |
| `write_mode` | **placing** | frozen substrate |
| `atom_width_frac_spacing` = `atom_width_selected_frac` | **0.37** (re-selected; guard armed) | frozen `selected_atom_width` |
| `launch_mode` / `k` / `σ_q` / `R` | **feature_factored** / 4 / 0.15 / 2.0 | frozen `launch_protocol` |
| φ | `build_phi(cfg, phi_seed=20260801)`, **576 B**, sha256[:32] **`a2713a0f…dbb1`** | **matches frozen exactly, 3/3 seeds** |
| launch keys | ⚠ **`PRNGKey(7000 + seed)`, unseen `fold_in(·,1)`** | ⛔ **rider R1 — NOT frozen** |
| `γ_address / γ_read / dt` | 0.05 / 0.02 / 0.05 | frozen |
| V3 budget grid | **[50, 100, 200, 400, 800, 1200]** total Verlet steps, 4 particles, `addr = round(b/3)` | frozen `v3_budget_grid`, read at run time |
| grid | **5 lr × 3 capacity × 3 tune seeds**, **594 configs**, `steps = 400` | `C2W11_GRID` |
| validation | 32 of 128 SEEN, **rule-4-valid vs every training row: 32/32** | §9 |
| γ as an axis | ⛔ **NOT RUN** — no static arm has a rollout; in V3 γ is the frozen grid's | declared |

---

# 3. ⭐⭐ V1 — THE HEADLINE NULL TABLE (5 seeds, unseen exact-set accuracy, `tol = 0.28696`)

`chance = 0.000390625` · bar (context only — ⛔ **I compute no verdict**) `= chance + 0.05 = 0.05039`.

| arm | selected config | store B | state B | total B | **unseen (best reader)** | ±2 SE | **SEEN** | shuffle-φ launder | read mult-adds |
|---|---|---|---|---|---|---|---|---|---|
| ⭐ **N1** gradient-placed | `lr 0.1, a=64, τ=0.2, shell, soft` | 114 688 | 0 | 115 680 | **0.00078** | 0.00096 | ⭐ **1.0000** | 0.00000 | 40 960 |
| **N2** VQ | `product_vq, 32 codes, fitted` | 2 560 | 0 | 3 552 | **0.00039** | 0.00078 | 0.0000 | 0.00000 | 1 280 |
| **N3** static-geometric | `lr 0.03, csb, written, τ=0.2` | 1 792 | 0 | 2 784 | **0.00039** | 0.00078 | 0.0047 | 0.00000 | 640 |
| **N4** kNN | `set_code ⚠, k=10, uniform` | 0 | 6 144 | 6 720 | **0.00000** | 0.00000 | 0.0000 | 0.00000 | 1 536 |
| **N5** Titans | `lr 0.01, h=64, η=0, α=0.01, surprise, chunk 32` | 6 432 | 6 432 | 13 440 | **0.00000** | 0.00000 | 0.4766 | 0.00000 | 768 |

⚠ N4's selected key space is `set_code` — **the noiseless-key variant**, flagged exactly as C2W5
flagged it. Reader parameters measured **from the code**: `sum_linear 104 · well_table 72 · knn 0 ·
mlp 92 · well_identity 0 · sum_identity 0`, all **< `N_a·m` = 256** — an independent reproduction of
the frozen `reader_class.measured_params`.

## 3.1 ⭐⭐ `null*` — COMPUTED over the ENTIRE registered grid (never estimated)

Every one of **594 configurations** refit on all of SEEN and scored on `Q_unseen`, **5 seeds**
(⛔ an **oracle-selected upper bound**, so the statement reads *"no configuration clears"* rather than
*"the one we picked didn't"* — ⛔ it is never any arm's own score):

| arm | configs | **grid-max (mean over 5 seeds)** | best single seed | argmax |
|---|---|---|---|---|
| N1 | 180 | 0.00039 | 0.00195 | `lr 1e-3, a=64, τ=0.2, written, soft` |
| N2 | 84 | **0.00195** | 0.00195 | `kmeans, 128 codes, written` |
| N3 | 60 | **0.00195** | 0.00391 | `lr 0.03, b, written, τ=0.05` |
| N4 | 30 | 0.00078 | 0.00195 | `set_code ⚠, k=2, idw` |
| N5 | 240 | 0.00039 | 0.00195 | `lr 1e-3, h=64, η=0, α=0.01, gate=none, chunk 1` |
| ⭐ | **594 × 5 seeds** | ### **`null*` = 0.001953125** | — | **N2 (N3 ties)**; bar 0.05039 ⇒ **25.8× short** |

**N5 divergences: 70 / 240** configs (the momentum × lr corners; banked 81/240). A diverged fit is
recorded, given a finite enormous MSE so it **must lose selection**, and never disappears.

> ⭐ ⛔ **THE ONE-LINE RESULT, AND IT IS NEVER QUOTED WITHOUT ITS SECOND CLAUSE:**
> **no matched-capacity non-physics organizer clears — `null*` = 0.00195 against a bar of 0.05039,
> computed over 594 configurations × 5 seeds — while the very launch points every arm consumes remain
> exactly decodable on 34.5 % ± 1.8 % of unseen queries** (§6). The information is in the launch; the
> quantise-then-sum read expresses **0.6 %** of it.

## 3.2 The `tol`-multiple curve — POOR, not INERT (seed 0, native read)
`x1 / x2 / x4` of `tol`: N1 0.0000 / 0.0117 / **0.3340** · N2 0.0000 / 0.0137 / **0.4805** ·
N3 0.0000 / 0.0059 / 0.2227 · N4 0.0000 / 0.0273 / **0.6309** · N5 0.0000 / 0.0020 / 0.2871.
⇒ every arm is producing an answer in the right *region* and missing the set — the same
"POOR, not INERT" signature C2W5 measured on the physics arm.

---

# 4. ⭐ THE FOUR INTERNAL-VALIDITY ANCHORS — **ALL FOUR PASS**

⛔ **No "nothing works" statement ships without L1**, and L1 is the sharpest fact in this report.

| # | anchor | bar | **measured** | banked C2W5 | verdict |
|---|---|---|---|---|---|
| **L1** | N1 fits its own training items (best of 180 configs) | ≥ 0.50 | ⭐ **1.0000** | 1.0000 | ✅✅ |
| **L2** | N4 `k=1` memorises SEEN (6 configs) | ≥ 0.95 | **1.0000** (all 3 key spaces) | 1.0000 | ✅ |
| **L3** | shuffle-φ launder, worst over 25 cells | ≤ chance + 0.005 | **0.00000** | ≤ 0.00039 | ✅ |
| **L4** | N1 capacity flatness on `Q_unseen`, `a ∈ {12,32,64}` | ≤ 0.02 | **0.00039** (0.0 / 0.00039 / 0.00039) | 0.0000 | ✅ |

> ⭐ **N1 at 1.0000 train / 0.00078 held-out, with 28 672 free floats and the identical store
> parameterisation, is what makes "no arm clears" a statement about the PROBLEM and not about my
> optimiser. It memorises perfectly and composes not at all — the banked anchor reproduces exactly on
> the repaired substrate.** Capacity is not the binding constraint (L4: identical at 21 504 B,
> 57 344 B and 114 688 B of store).

**Train-fit liveness (reported, unregistered):** N2 **1.0000** · N5 **1.0000** · ⚠ N3 **0.0104**.

⚠⚠ **N3 IS STILL THE WEAK ARM-SIDE NUMBER, AND MY OWN PREDICTION IS REFUTED.** I registered (P12)
that a real fitting budget would take N3's in-sample fit from the banked 0.0799 to ≈ 0.35. Measured
**0.0104** over 60 configs × 3 seeds — *worse* than banked. But ⛔ this is **not** an optimisation
failure: the same arm reaches **0.9497** agreement when its objective is the assignment rather than
the read (§7). **N3's fitting problem is the read objective, not its capacity** — see the sentence I
am required to carry, in §7.

---

# 5. V2 — THE MATCHED CONFIDENCE CHANNEL (VALUE leg ii's null side)

**Construction (rider R2, mine):** 4 of 32 wells reserved and **never written**; SEEN = 128
rule-4-valid `F`-subsets of the 28 written wells; eval = 384 queries with **0 / 1 / 2 novel channels**
(128 each), rule-4-valid against every SEEN row. φ untouched. Per-particle label = *this channel's
launch code is an unwritten well* (**212 / 1 536** particles novel).

**Every arm has a principled channel; ⛔ none is a NOT-RUN and none is scored as "uncalibrated".**
N1 = the winning atom's read-objective weight · N2 = distance-to-codebook · N3 = the fitted rule's
evidence (see below) · N4 = per-channel neighbour distance · N5 = the read-time surprise
`−‖M(k) − M₀(k)‖²`. Sign convention fixed **in advance**: higher = more familiar.

| arm | **V2a AUROC** | ±2 SE | label-permutation negative | permuted-payload negative | **V2b ECE** | its accuracy |
|---|---|---|---|---|---|---|
| N1 | **0.5106** | 0.0754 | 0.5086 ± 0.1104 | 0.6119 | 0.2419 | 0.00000 |
| N2 | **0.5367** | 0.0957 | 0.5692 ± 0.1527 | 0.5367 | 0.4979 | 0.00000 |
| N3 | **0.4431** | 0.1233 | 0.4050 ± 0.1070 | 0.4597 | 0.5539 | 0.00000 |
| N4 | **0.5690** | 0.0506 | 0.4937 ± 0.0390 | 0.5690 | ⚠ **0.0005** | 0.00052 |
| N5 | **0.5970** | 0.0528 | 0.4582 ± 0.0458 | 0.5881 | 0.4811 | 0.00000 |
| ⭐ | **`null*_V2a` = 0.5970 (N5)** | | | | **best ECE 0.0005 (N4)** | |

> ⛔⛔ **THE HONEST READING, AND IT IS NOT "the nulls are weakly calibrated":** **no null arm's novelty
> channel is distinguishable from its own permutation null.** Every AUROC's 2 SE interval contains the
> label-permutation negative, and the mean confidence is flat across novel counts (N1:
> −0.7656 / −0.7816 / −0.7705 at 0/1/2 novel channels). ⛔ **At this cell V2a's null side is a
> measured NON-CHANNEL, not a weak channel.**

**⭐ The mechanism, because it is a property of the wave's own launch and it will bite spoke B too.**
A "novel" well is **unwritten**, not **unvisited**: the launch head deflates against **all 32** φ
codes, so SEEN queries send particles to novel wells' regions routinely (occupancy precision is
0.2303 — 77 % of channels land outside `A(x)`). Every arm's codebook therefore has mass exactly where
"novel" lives, and **a distance-based confidence cannot see the distinction by construction.**

## 5.1 ⚠ My prediction was refuted, and by more than a factor
I registered N2 = 0.88, N4 = 0.85, N1 = 0.82, N3 = 0.75, N5 = 0.65 (P13–P17) on the reasoning that
"novel = far from everything I fitted". **That premise is false on this launch protocol** (above).
Measured 0.44–0.60. ⛔ Recorded as a refutation, not softened.

## 5.2 ⚠ The designed negative that does not bind — measured, declared, NOT a pass
`PREREG-C2W11.md` §5 registers *permuted payloads ⇒ AUROC ≈ 0.5*. Four of five channels are functions
of the **address** geometry alone, so a payload permutation leaves them **structurally invariant** —
and it does: N2 0.5367 → 0.5367 and N4 0.5690 → 0.5690, **bit-identical**. ⛔ **That is a measurement,
not a pass.** The negative that *can* fail here is the **label permutation**, and it is run per arm
per seed (column 4). Only N1/N3/N5 move under payload permutation at all (their fits consume `y`).

## 5.3 ⚠ V2b's registered degeneracy FIRED (P19, `P = 0.80`)
Every arm's set accuracy is ≈ 0, so ECE collapses to *mean confidence* and **the most under-confident
arm wins by construction**. N4 "wins" at **0.0005** because its calibrated confidence is ≈ 0
everywhere — which, at an accuracy of 0.00052, is **correct**. ⛔ The harness marks `degenerate = true`
on 25/25 cells and ECE is never reported without its accuracy beside it.

## 5.4 ⚠ A channel I changed, and why (decided by a TEST, not by a score)
The task registers N3's channel as *the fitted rule's margin*. The margin is a **broken novelty
statistic**: as `z` leaves the codebook the two leading `−d²/2σ²` terms separate without bound, so it
reports **maximal confidence where the arm knows least** (fixture: 6.98 at the fitted points vs
**280.99** at `z + 50`). N3 therefore gets the **evidence** form `max_j score_j`, the exact analogue of
N1's and N2's channels; **the registered margin form is still computed and reported** (`0.5948 ±`) —
and note it scores *higher* than the sound one, which is the signature of a statistic tracking
distance rather than novelty.

---

# 6. ⛔ THE DECODABILITY CEILING, RECOMPUTED ON THE C2W11 LAUNCHES (out-of-class, never an arm)

⚠ `phi_decodability_ceiling` **cannot be reused** — it de-offsets by `phi.offsets`, which the
feature-factored head does not have. The new matched filter compares the observed `(k, d)` launch
block against each of the **35 960** combinations' own noiseless launch blocks, in the deflation's
deterministic channel order.

| condition | banked C2W5 | **measured (5 seeds)** | my prereg |
|---|---|---|---|
| **noiseless** (exact set-code) | 1.0000 | **1.0000 ± 0.0000** | P21: 1.00 ✅ |
| ⭐ **as-launched** (the points every arm sees) | 0.2719 ± 0.0126 | **0.3449 ± 0.0175** | P22: point 0.90, band [0.30, 1.00] — ⚠ **inside the band at its lower edge; the point prediction is REFUTED by 2.6×** |
| per-channel asserted-set exact (K6's statistic) | — | **0.0012 ± 0.0010** | — |
| per-channel asserted-set precision | — | **0.2356** | — |

> ⭐⭐ **The registered counter-hypothesis is the one that fired, and it is the finding.** I predicted
> ≈ 0.90 on the reasoning that `σ_q = 0.15` against a code separation of ≈ 2.8 makes the channel picks
> recoverable, so the decoder reduces to *"which `A` deflates to this pick-tuple"* — injective unless
> it collides. **It collides.**
>
> ⭐ **And the two candidate mechanisms are separated by measurement, not by assertion** (⚠ run
> **after** the fact, declared out-of-class, **not** pre-registered — `results/
> ceiling_collision_vs_noise.json`, 3 seeds). Re-running the identical decoder on a **noiseless**
> launch (`σ_q = 0`) gives **0.3424** against the as-launched **0.3372** — **the query noise costs
> 0.005 and the remaining 0.66 is STRUCTURAL**: two thirds of unseen combinations deflate to a
> pick-tuple another combination also produces. ⛔ **So the ceiling is not a noise ceiling and cannot
> be bought back with a smaller `σ_q`; it is a property of the greedy matched-filter deflation.**
>
> ⛔ **The feature-factored launch buys addressability (`≥ F`
> distinct wells 0.050 → 0.9967) and pays for it in set information (0.27 → 0.34 is a gain, but the
> ceiling is nowhere near the 1.0000 the noiseless code carries).**
> ⭐ And the gap that matters: **ceiling 0.3449 vs `null*` 0.00195 — a factor of 177.**

---

# 7. THE ORACLE-IMITATION NULL (T5.2 rider (i)) — 3 seeds

⛔ **Rider R3:** the target is the **written, un-organized** physics store's own assignments.

| quantity | banked C2W5 | **measured** |
|---|---|---|
| N3 fitted **on the store's own assignments**, agreement on unseen | 0.8888 | **0.9497 ± 0.0059** |
| N3 fitted **on the read objective**, agreement (F5's registered null) | 0.2576 | **0.8239 ± 0.0116** |
| F5 fires (≥ 0.99)? | NO | **NO** |
| that store's own zero-parameter unseen score | — | 0.00130 |

> ⭐ **Carry this sentence wherever F5 is discussed (verbatim, as instructed):** *the 0.89-vs-0.26 gap
> is an optimisation gap in fitting the diagram, NOT evidence of a structurally non-VQ channel.*
> ⭐ **And this wave supplies the confirming measurement:** with a fitting budget that reaches an
> optimum (fitted payloads, 800 steps) the read-objective agreement moves **0.2576 → 0.8239** while
> the assignment-fitted agreement barely moves (0.8888 → 0.9497). **The gap was the optimiser, and it
> has now largely closed.** ⇒ this substrate's placed-store assignment is ~95 % reproducible by a
> static power diagram.

---

# 8. ⭐⭐ V3 — THE NULL SIDE HAS A CURVE (leg iii), AT THE FROZEN BUDGET GRID

**Budget axis read from `FROZEN-INTERFACES-C2W11.json::v3_budget_grid`** at run time:
`[50, 100, 200, 400, 800, 1200]` total Verlet steps, 4 particles, `address = round(b/3)`,
`γ_address → γ_read`, `dt = 0.05`. Each arm's codebook is **instantiated as a landscape** by the
**physics arm's own placing write** and read by the **physics arm's own `multi_particle_read`**.

| arm | 50 | 100 | 200 | 400 | 800 | 1200 | flat? | monotone? | span |
|---|---|---|---|---|---|---|---|---|---|
| **N1** | 0.00078 | 0.00039 | 0.00000 | 0.00039 | 0.00039 | 0.00039 | **yes** | no | 0.00078 |
| **N2** | 0.00000 | 0.00078 | 0.00078 | 0.00039 | 0.00039 | 0.00039 | **yes** | no | 0.00078 |
| **N3** | 0.00000 | 0.00078 | 0.00039 | 0.00078 | 0.00078 | 0.00078 | **yes** | no | 0.00078 |
| ⭐ **`null*_V3`** (grid-max over {N1,N2,N3} × the width axis, 5 seeds) | 0.00078 | 0.00117 | 0.00078 | 0.00117 | 0.00117 | **0.00117** | | | |

⛔ **N4 and N5 are DECLARED NOT-RUN for V3** — no landscape exists. Their **flat reference lines** are
their static V1 scores: **N4 = 0.00000, N5 = 0.00000**. ⛔ They are never scored as "un-navigable".

**⭐ The read-compute ledger, and it is the cleanest matched comparison in this report.** The
instantiated null at `b = 1200` costs **23 961 600 mult-adds/query** on 384 atoms — **exactly the
physics arm's own read cost** (`4 × 1200 × 384 × 13`), because it *is* the same read on the same atom
budget. Store bytes **21 504** = the frozen `store_bytes_at_this_cell`. So the V3 row is
byte-matched **and** compute-matched by construction.
⚠ Reported separately, and **labelled a READ-COMPUTE RATIO** (not wall-clock, not training cost):
physics settled read **23 961 600** ÷ N1's matched-capacity **static** read **40 960** = **585×**
(banked 3 360× at C2W5's `a = 32`; my prereg P29 said ≈ 1 000×, band [300, 5 000] ✅).

## 8.1 ⚠⚠ THE MIRROR-IMAGE ATTACK — closed by measurement, with one honest caveat
The instantiation's three hobble-able knobs were swept F3-style: **5 widths × 3 depths × 2 atom
budgets = 30 configs × 3 tune seeds per arm**, selected on the held-out-from-seen split.
⛔ **The honest caveat, stated because the alternative is a false claim of tuning:** **all 30 configs
scored `val_acc = 0.0000` on every arm** — the selection statistic **did not resolve**, and the
"selected" instantiation (`width_frac 0.20, depth 0.15, 384 atoms`) is the first of thirty ties.
**What actually closes the attack is the grid-max row**, which re-scores the *whole width axis* on
`Q_unseen` at **every** budget point and still tops out at **0.00117**. ⇒ the statement is *"no
instantiation on the registered axis clears at any budget"*, not *"the one we picked didn't"*.

## 8.2 V3-MECHANICS on the null side
All three curves are **flat** (span 0.00078 = 0.4 grains) and **non-monotone**. ⛔ **This is NOT the
claim "the null store is empty"** — the store demonstrably carries the payloads (it is written by the
same placing write, and the same store reads 0.22–0.48 at 4×`tol`). It is the claim that **at this
read protocol the budget dial has nothing to resolve on the null side**, which is the reference line
spoke B's physics curve is differenced against.

---

# 9. ⛔ THE SELECTION GUARD, MECHANICALLY (the wave-invalidating condition)

- **Where `Q_unseen` is constructed:** `seed_setup()` — **one function, one place** (`ind_u`, `q0_u`,
  `family.y_unseen`).
- **Every place it is read:** `stage_score`, `stage_gridmax`, and the declared out-of-class
  diagnostics `stage_ceiling` / `stage_oracle`. `stage_v3`'s **phase 1 (the instantiation sweep) does
  not read it**; phase 2 does, and is the reported grid-max.
- **The assertion that no fit ever sees it:** every `_fit_arm` call passes `S["tr"]` (grid) or all of
  SEEN (score/gridmax); the selection statistic is the arm's own read on `S["va"]`.
  **`tests/test_c2w11_nulls.py::test_q_unseen_is_never_an_input_to_a_fit`** poisons `q0_u`/`ind_u` to
  zeros, refits all five arms, and asserts the SEEN-side predictions are **bit-identical**.
- **The validation split inherits the family's own rule 4:** 32 rows held out with
  `|A_val ∩ A_train| ≤ F−2` against **every** retained training row — **achieved 32/32
  (`frac_rule4_valid = 1.0`)**, asserted in a test.
- ⚠ **Honest power, as C2W5 warned:** at 32 validation rows the grain is `1/32 = 0.031`, and **every
  config again scored `val_acc = 0.0000`** except N3/N5 (0.0104 = 1 row of 96 on one seed). Selection
  therefore ran essentially on the declared MSE tie-break — **which is exactly why `null*` is computed
  over the whole grid on `Q_unseen` (§3.1) and the verdict does not depend on the selection statistic
  resolving anything.**

---

# 10. How I verified (commands + observed output)

```
git worktree add ../CHLU-c2w11c -b c2w11-organizer-swap-nulls 168a892
ruff check chlu/ tests/                                     # All checks passed!
chlu exp-c2w11-nulls --quick --out-dir .../quick            # end-to-end, ~4 min
chlu exp-c2w11-nulls --stages guards ceiling --seeds 0..4    # phi/launch guards + the ceiling
chlu exp-c2w11-nulls --stages grid score gridmax v2 v3 oracle --seeds 0..4   # ~55 min
chlu exp-c2w11-nulls --stages v2 anchors --seeds 0..4        # v2 re-run on the corrected channels
pytest tests/test_c2w11_nulls.py -q                          # 22 passed
pytest tests/ -q                                             # §10.1
```
Wall: grid **548 s** · gridmax **1 208 s** · v3 ≈ 1 500 s · v2 + ceiling + oracle ≈ 300 s.
Artifacts: `results/stage_{guards,grid,score,gridmax,v2,v3,ceiling,oracle,anchors}.json` +
`c2w11_nulls_summary.json`; logs `run_guards.log`, `run_full.log`, `run_v2b.log`, `quick.log`.

## 10.1 Full suite — count arithmetic, checkout named
**Checkout: `../CHLU-c2w11c`, branch `c2w11-organizer-swap-nulls` rebased onto `main @ a47ed20`.**
Same-checkout collect-only: **1 681 selected without my test file → 1 703 with it = +22**, matching
my new test count exactly. ⚠ `main` verified at `a47ed20` **before and after** each run
(HEAD-stability rule).

**Run 1: ⛔ 19 failed / 1 684 passed (2 405 s) — reported because it happened.**
**Run 2 after the fix `0d6aa17`: ✅ 1 703 passed / 0 failed / 29 warnings (2 686 s).**
**Arithmetic exact, same checkout: 1 681 (base) + 22 (mine) = 1 703.**
`main` verified `a47ed20` **before and after** run 2 — HEAD-stable, so the green is against the base
it was run on.

> ⭐⭐ **The failure, and it is the banked repo hazard in its mirror image.** My new x64 regression test
> enabled `jax_enable_x64` and restored a **hard-coded `False`**. But several repo modules enable x64
> **at import**, so the ambient state during a suite run is **ON** — I therefore turned it **OFF for
> everything alphabetically downstream**: **18 failures in `test_goldstone.py`,
> `test_friction_field.py` and `test_lattice*.py`, every one of which passes alone**, plus one of my
> own (a module-scoped fixture built under float64 and re-read under float32).
> Fixed in `0d6aa17` by saving and restoring `prev`, which is the convention already used by
> `test_blocks.py` and `test_cl_baselines_x64.py`. ⛔ **No measured number in this report moves** —
> every experiment stage runs in its own process under the default flag, and the three-file check
> (`test_c2w11_nulls` + `test_goldstone` + `test_friction_field`) goes **10 failed → 59 passed**.

---

# 11. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)

1. **`OD` / `OD_min` / any swap verdict / any tier-ii verdict / any paper number** — ⛔ not mine.
2. **N4 and N5 for V3** — no landscape exists (Hub-registered). Flat reference lines given in §8.
3. **The γ axis** — no static arm has a rollout; in V3, γ is the frozen grid's, not an axis.
4. **ψ, the novelty head, the organization loss, K4-at-full-ψ, K5-as-blocking** — spoke B's.
5. **Any K-verdict re-adjudication** — spoke A's; quoted, never re-scored. ⛔ **No K5 verdict exists in
   this wave and nothing about table-expressibility is claimed anywhere above.**
6. **Readers on the full 594-config grid** — `stage_gridmax` scores the arm's *own* read (fitting 6
   readers 2 970 times is unaffordable). On the selected configs, where all six readers do run,
   native and reader scores agree to the last digit; on the grid it is an assumption, stated.
7. **N5 with a learned key projection; N2 with a learned encoder** — both break the identical-launch
   match.
8. **`P > 4`** — the launch is the frozen protocol; changing `k` is a different cell.

---

# 12. ⭐ PREREG SCORECARD (`PREREG.md`, filed before `exp_c2w11_nulls.py` existed)

| # | registered | measured | verdict |
|---|---|---|---|
| P1 N1 | 0.0020 [0, 0.010] | 0.00078 | ✅ |
| P2 N2 | 0.0020 [0, 0.010] | 0.00039 | ✅ |
| P3 N3 | 0.0020 [0, 0.010] | 0.00039 | ✅ |
| P4 N4 | 0.0020 [0, 0.012] | 0.00000 | ✅ |
| P5 N5 | 0.0020 [0, 0.012] | 0.00000 | ✅ |
| ⭐ P6 `null*` | 0.0039 [0.001, 0.020] | **0.00195** | ✅ (over-predicted 2×) |
| P7 no arm clears (`P = 0.92`) | NO arm clears | **none clears** | ✅ |
| P8 / L1 | 1.0000 | **1.0000** | ✅✅ |
| P9 / L2 | 1.0000 | **1.0000** | ✅ |
| P10 / L3 | ≤ 0.0020 | **0.0000** | ✅ |
| P11 / L4 | 0.0000 | **0.00039** | ✅ |
| ⚠ P12 N3 in-sample | **0.35** [0.10, 0.80] | **0.0104** | ⛔ **REFUTED** — and §7 says why it is not an optimiser failure |
| P13 V2a N2 | 0.88 [0.60, 0.99] | **0.5367** | ⛔ **REFUTED** |
| P14 V2a N1 | 0.82 [0.55, 0.98] | **0.5106** | ⛔ **REFUTED** |
| P15 V2a N3 | 0.75 [0.50, 0.95] | **0.4431** | ⛔ **REFUTED** |
| P16 V2a N4 | 0.85 [0.55, 0.98] | **0.5690** | ⛔ **REFUTED** |
| P17 V2a N5 | 0.65 [0.45, 0.90] | **0.5970** | ✅ in band |
| P18 `null*_V2a` | 0.92 | **0.5970** | ⛔ **REFUTED** |
| ⚠ P19 V2b degeneracy fires (`P = 0.80`) | fires | **fired, 25/25 cells** | ✅ |
| P20 designed negative ≈ 0.5 | 0.50 ± 0.05 | ⚠ **structurally invariant on 4/5 arms** | ◐ the negative does not bind (§5.2) |
| P21 ceiling noiseless | 1.0000 [0.98, 1.00] | **1.0000** | ✅ |
| ⭐ P22 ceiling as-launched | **0.90** [0.30, 1.00] | **0.3449 ± 0.0175** | ◐ **in band, point REFUTED 2.6×** — the registered counter-hypothesis (heavy pick-tuple collision) is what fired |
| ⭐ P23 the gap ⇒ a READ-PROTOCOL refutation, not a family refutation | `null* ≈ 0.004` vs ceiling ≈ 0.90 | **0.00195 vs 0.3449 — a factor of 177** | ◐ **the registered branch fired; both numbers moved** |
| P24 N1 V3 @1200 | 0.0020 [0, 0.010] | 0.00039 | ✅ |
| P25 N2/N3 V3 @1200 | 0.0020 each | 0.00039 / 0.00078 | ✅ |
| P26 `null*_V3` | 0.0039 | **0.00117** | ✅ |
| P27 null curve flat (`P = 0.75`) | flat | **flat, 3/3 arms** (span 0.00078) | ✅ |
| P29 read-compute ratio | ≈ 1 000× [300, 5 000] | **585×** | ✅ |

**Score: 19 ✅ · 3 ◐ · 6 ⛔ (28 registered rows).** ⭐ **The six refutations are all in one place — V2's novelty channel and
N3's in-sample fit — and both have a measured mechanism** (§5, §7). My own registered falsifier list
(§6 of the prereg): L1 held, L3 held, no arm cleared; **P22 landed at its band's lower edge, which is
falsifier 3's "the launch destroyed set information" branch in its weaker form.**

---

# 13. Git footprint

- **Branch:** `c2w11-organizer-swap-nulls`, worktree `../CHLU-c2w11c`, base `168a892`,
  **rebased onto local `main @ a47ed20`** (⚠ **not** `origin/main`; `168a892` is an ancestor of
  `a47ed20`, so the rebase was clean and touched nothing outside my files). ⛔ Not pushed, no PR, no
  merge; `clu-dev` untouched.
- **Commits (verified from the MAIN repo, protocol §3.2):**
  - `717fd67` the C2W11 null-arm core (feature launches, confidence channels, landscape instantiation)
  - `3644818` the organizer-swap null harness (`exp-c2w11-nulls`)
  - `741af9b` tests: matching obligations + selection guard
  - `1289c6a` L1–L4 as a computed stage, and N4's confidence in its own key space
  - `0d6aa17` restore the x64 flag to its **previous** value, not to `False` (§10.1's 18 failures)
- **`git diff --stat main..c2w11-organizer-swap-nulls` (run FROM THE MAIN REPO, protocol §3.2) =
  3 files changed, 2 356 insertions(+), 0 deletions(-):** `chlu/core/null_arms.py` (+443, additive
  block) · `chlu/experiments/exp_c2w11_nulls.py` (new, 1 489) · `tests/test_c2w11_nulls.py`
  (new, 424). All five commits verified visible on the shared ref from the main repo.
- ⛔ **NOT touched, as declared:** `chlu/cli/experiment_cmd.py` (spoke A landed my subcommand; my
  handler lives in my module) · `chlu/config.py` · `factored_store.py` / `feature_launch.py` /
  `exp_c2w11_substrate.py` / `test_c2w11_substrate.py` (spoke A's — imported read-only) ·
  `psi_readout.py` / `novelty_read.py` / `exp_c2w11_organizer.py` / `training/losses.py` /
  `test_c2w11_organizer.py` (spoke B's) · `scripts/csf3/` / `train_cluformer.py` / `blocks.py` /
  `exp_cluformer_pilot.py` (the pilot's) · `well_lifecycle.py` / `clu_system.py` /
  `soft_certificate.py` / `test_gate_addr.py` / `test_well_lifecycle.py` / `test_cifar_strong_phi.py`
  (C2W8-close's).
- **Concurrency:** the shared checkout sat on the live pilot spoke's branch throughout and was never
  touched. `../CHLU-c2w11a` (spoke A's) exists; **zero file overlap** with mine — verified by the diff
  above. ⚠ **Worktree left in place for Hub review**; branch ref verified from the main repo.

---

# 14. Open questions / follow-ups / risks

1. ⭐⭐ **The read protocol is again the measured cap, and now with the launch repair CONTROLLED FOR.**
   C2W5's diagnosis was "the launch cannot reach `F` distinct wells" (2.20/4). That is **fixed**
   (3.998/4). The arms did **not** move (0.00117 → 0.00195). ⇒ **addressability was not the binding
   constraint; the quantise-each-particle-then-sum read is.** Two requirements fall straight out and
   both bind the physics arm exactly as hard: the read must (i) get the *right* `F` wells (precision
   0.2356, not distinctness) and (ii) not discard the continuous launch coordinate the 0.3449 ceiling
   decodes.
2. ⚠ **Spoke B's V2 is at risk from §5's mechanism, not from its own design.** If the physics novelty
   head is also a function of address geometry, the same non-channel applies to it. Worth flagging
   before its V2a floor (`AUROC_phys > 0.60`) is read as a physics failure — **on this launch protocol
   a 0.60 floor may be unreachable by any address-geometry method.**
3. ⚠ **The instantiation selection did not resolve** (§8.1). If the Hub wants a *selected* (rather
   than grid-maxed) V3 null, the instantiation sweep needs a validation statistic with grain finer
   than 1/32 — the same fix §9 needs.
4. **`null*_V2a` and `null*_V3` are emitted but are not comparable to anything yet** — spoke B's
   physics numbers are required, on **the same launch key** (rider R1) and **the same novel/known
   split** (rider R2). Until both are confirmed, ⛔ no V2/V3 swap difference should be computed.
5. **N2 at 128 codes is the `null*` holder.** It is 4× the vocabulary at 1/8 the atoms — a reviewer
   will ask whether the byte match is honest. It is: 128 codes × (4 + 8) floats = 6 144 B vs the
   physics store's 21 504 B, i.e. the winning null is **3.5× smaller**, not larger.

---

## Proposed handover updates (for the Hub)

- **§3 config / CLI — NEW:** `chlu exp-c2w11-nulls [--stages guards grid score gridmax v2 v3 ceiling
  oracle anchors] [--arms N1..N5] [--seeds …] [--quick]` is now **live** (spoke A's stub is filled by
  `chlu/experiments/exp_c2w11_nulls.py`). New env var **`CHLU_C2W11_FROZEN`** points the harness at
  `FROZEN-INTERFACES-C2W11.json`; ⚠ **required when running from a worktree**, because `.claude/` is
  gitignored and exists only in the main checkout. A non-`--quick` run without it now **raises**
  rather than silently running unverified.
- **§7 Known Issues — NEW (open, and it is a MEASUREMENT, not a bug):** *feature-factored launches fix
  distinctness and not precision.* `≥ F` distinct wells 0.050 → **0.9967**; occupancy precision
  0.4061 → **0.2303**; asserted-set-exact **0.0012**. Any experiment whose target is a sum over the
  *correct* `F` wells must check precision, not distinctness.
- **§7 Known Issues — NEW (open):** *a payload-permutation designed negative does not bind an
  address-geometry confidence channel* — measured bit-identical on N2/N4. Any leg using it needs a
  label-permutation negative beside it.
- **§7 Known Issues — NEW (closed by this spoke):** the C2W5 `phi_decodability_ceiling` is **invalid
  under feature-factored launches** (it de-offsets by `phi.offsets`); use
  `feature_decodability_ceiling`. The banked **0.2719** must not be carried into C2W11 text — the
  recomputed value is **0.3449 ± 0.0175**.
- **§7 Known Issues — NEW (open, and it is a C2W9 input):** *the greedy matched-filter deflation is
  ~66 % non-injective at this cell.* Measured by re-running the decoder at `σ_q = 0`: **0.3424
  noiseless-launch vs 0.3372 as-launched ⇒ noise costs 0.005 and the rest is structural.** ⛔ A
  smaller `σ_q` cannot buy the ceiling back; the launch head's own decomposition is the cap, which is
  a **launch-head** finding and belongs beside the coverage trigger rather than in any organizer's
  ledger.
- **Registry / doctrine candidates:** (i) ⭐ *an organizer audit needs an in-class fit anchor* —
  reconfirmed at a second substrate (N1: 1.0000 train / 0.00078 held-out); (ii) ⭐ *quote the
  decodability ceiling beside every null-arm null* — it cost 0.7 s/seed and turned "no arm clears"
  into "the read expresses 0.6 % of what the launch carries"; (iii) ⭐ **new:** *a confidence channel
  must be tested for sign, not just for existence* — N3's registered margin channel is inverted in the
  tail and would have silently produced a spurious V2a number; (iv) *a guard that compares against a
  frozen digest must use the frozen artifact's own hash function.*
- **§7 Known Issues — NEW (open, and it is the x64 hazard's MIRROR IMAGE):** the banked entry says
  *"a module mixing float32 data with flag-following initialisers is silently x64-dependent"*. This
  spoke hit the **opposite** failure: **a test that RESTORES `jax_enable_x64` to a hard-coded `False`
  disables x64 for every alphabetically-later module**, and since several modules enable it *at
  import*, that is a suite-wide state change (18 failures here, all passing alone). ⛔ **The rule is
  `prev = jax.config.jax_enable_x64` … `update(..., prev)`, never `update(..., False)`.** Worth a
  one-line note wherever the x64 hazard is documented, because the two failure modes look identical
  from a per-file run (both are "passes alone, fails in suite").
- **`PREREG-TierII.md` erratum:** §3.5's F5 discussion should now carry **both** halves — the
  0.89-vs-0.26 gap is an optimisation gap, **and this wave measured it closing to 0.9497-vs-0.8239**
  once the fitting budget reaches an optimum.
