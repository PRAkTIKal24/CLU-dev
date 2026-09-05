# PREREG — `reader-fitting-audit`

**Filed 2026-08-05, BEFORE any re-scoring harness ran and before a single line of
`well_identity` / `soft_well_identity` / `gated_well_identity` / `sum_identity` existed.**
Base: local `main @ d1149a4`. Branch `reader-fitting-audit`, worktree `../CHLU-readeraudit`.

⛔ Nothing in this file may be edited after the first re-score. Deviations go in the report.

---

## 0. The question, in one line

`c2w7-read-cardinality` §4 measured, on one latent, that a **least-squares-fitted** 72-parameter
reader scores **0.0000** where a **zero-parameter identity** reader scores **0.0539 ± 0.0207**,
because lstsq shrinks `diag(W) → 0.40` and pushes the *correct* queries' residual (0.006) past
`tol = 0.234` to 0.537. **Does that pathology also void the published zeros of `orgdiv-null-arms`
and `tierii-read-fix`?**

## 1. ⭐ THE READING, COMMITTED IN ADVANCE (verbatim from the task file, with my numeric threshold)

Let `bar_a = chance_a + 0.05` be each cell's own registered bar, and let `I_a` be the best
**zero-parameter identity** reader's unseen exact-set accuracy on arm `a` (5 seeds), `F_a` the best
**fitted** reader's on the same arm/seed set (the published statistic).

| verdict | mechanical condition | consequence |
|---|---|---|
| **SURVIVES** | `I_a − 2 SE ≤ bar_a` for **every** re-scored arm **AND** `max_a (I_a − F_a) < 0.01` | the C2W7 finding is **scoped to the multiplicity read**; C2W5's conclusions stand unchanged; the wave's caveat is lifted |
| **MOVES** | `∃ a : I_a − 2 SE > bar_a` | the affected published zeros are **superseded**; erratum owed; the C2W5 attribution re-opens at the Advisor's review |
| **PARTIAL** | not MOVES, but `∃ a : (I_a − F_a) ≥ 0.01` **and** that gain is `≥ 2 SE` from zero | the zeros stand as verdicts but the *instrument* is impeached; the doctrine question goes to the Advisor |

⭐ **The `0.01` threshold is registered here numerically** (task §PREREG obligation 1, "state your
threshold numerically"): 0.01 = 20 % of the 0.05 bar and **12.8×** the `orgdiv-null-arms`
measurement grain (1/2560 = 3.9e-4) / **12.8×** `tierii-read-fix`'s (1/1280 = 7.8e-4). A gain below
0.01 cannot change any published verdict and is at most a handful of queries.

## 2. ⭐⭐ THE PREDICTION AND HOW IT WAS DERIVED (not a guess — an arithmetic bound)

**The derivation.** All three cells score `exact_set_accuracy(ŷ, y, tol)` with
`y(x) = Σ_{j∈A(x)} v_j`, `v_j` drawn **uniformly on a sphere of radius `R = payload_radius`** in
`R^m` (`factored_store.build_family`, lines 347–349), and `tol = 0.25 · RMS‖y − ȳ‖`. A
zero-parameter identity reader that asserts a set `S` outputs `Σ_{j∈S} v_j`, so its residual is
`‖Σ_{S} v − Σ_{A} v‖`. For a **single**-well substitution (`|S Δ A| = 2`, one well swapped) that
residual is `‖v_a − v_b‖`, and `E‖v_a − v_b‖² = 2R²` ⇒ residual `≈ √2 R`:

| cell | `R` | `tol` (published) | `√2 R` | **residual / tol** |
|---|---|---|---|---|
| `orgdiv-null-arms` / `orgdiv-cat-test` | 1.0 | 0.478 | 1.414 | **2.96×** |
| `tierii-read-fix` | 0.5 | 0.2338 | 0.707 | **3.02×** |

⭐ **Therefore the identity reader's score is, to within a rounding of one query, EXACTLY the
asserted set's exact-set accuracy.** It cannot absorb even one wrong well. This is the whole
prediction, and it makes every number below **already measured and published**:

| # | quantity (unseen, 5 seeds) | **the published statistic that bounds it** | **predicted value** | registered band |
|---|---|---|---|---|
| **P1** | `well_identity`, `orgdiv-null-arms` N1 @ selected config | exact-set occupancy `0.0000 / 2560` (null-arms §3) | **0.0000** | [0, 0.002] |
| **P2** | `well_identity`, N1–N5 @ each arm's **grid-max argmax** config | same, and N2/N3's fitted assignments also measured 0.0000 | **0.0000** | [0, 0.002] |
| **P3** | `sum_identity` (= the published `native` read), N1 @ selected | published `native_unseen` = 0.0000 | **0.0000** | exactly 0.0000 |
| **P4** | `sum_identity` SEEN, N1 @ selected | published `native_seen` = 1.0000 | **1.0000** | exactly 1.0000 |
| **P5** | `well_identity` **SEEN**, N1 @ selected | published `well_table` SEEN = 0.0000 | **0.0000** | [0, 0.05] |
| **P6** | ⭐ `I − F` (identity minus fitted), every `orgdiv-null-arms` arm | — | **0.0000** | [−0.002, +0.002] |
| **P7** | `gated_well_identity`, `tierii-read-fix` physics arm | `exact_set_occupancy_gated` = **0.0023 ± 0.0047** | **0.0023** | [0.000, 0.010] |
| **P8** | `soft_well_identity`, `tierii-read-fix` physics arm | π spreads over 11.3 wells; a 4-term sum is unreachable | **0.0000** | [0, 0.002] |
| **P9** | `well_identity` (hard, dedup), `tierii-read-fix` physics | `exact_set_occupancy_raw` = 0.0000 (`k=12 > F=4` ⇒ 0 by construction) | **0.0000** | exactly 0.0000 |
| **P10** | `sum_identity`, `tierii-read-fix` physics | sums 12 particle payload blocks against a 4-term target | **0.0000** | [0, 0.002] |
| **P11** | `gated_well_identity` on the **live launder** `L_a` | same bound; launder gated exact-set ≈ physics's | **0.0023** | [0.000, 0.015] |
| **P12** | ⭐ `OD_min` and `G1_min` recomputed **including** the identity members | the identity twins tie at ≈ 0 | `OD_min` **−0.0008**, `G1_min` **−0.0016** (unchanged in sign) | ±0.01 of the published values |

**P13 — the MECHANISM, measured not assumed (the task's named suspects).** I predict the C2W7
mechanism is *present but has nothing to destroy* at these cells:
- **P13a** `diag(W)` of the fitted `well_table` / `soft_well_table` on the null-arms and tierii cells
  is **shrunk**, mean `|diag(W)| ∈ [0.05, 0.70]` (C2W7 measured 0.40) — the shrinkage reproduces;
- **P13b** the number of unseen queries whose **asserted set is exactly right** (the population the
  shrinkage destroys) is **0 of 2560** at `orgdiv-null-arms` and **≈ 3 of 1280** at
  `tierii-read-fix`; ⇒ the pathology has **0** (resp. ~3) queries of headroom, against C2W7's ~18 %.
- **P13c** on those queries (where they exist) the identity residual is `< tol` and the fitted
  residual is `> tol` — i.e. the C2W7 crossing is real wherever the population is non-empty.

**P14 — the doctrine positive control (the identity reader's OWN failure mode).** On a synthetic
latent whose asserted set is **exactly right on 100 % of queries but whose payload table is scaled by
`α = 2`**, I predict the **fitted** reader scores **1.0000** and the **identity** reader **0.0000**.
This is the registered demonstration that an identity reader assumes the latent is already in the
target's units, and it is why it must be **added** to the class, never substituted for it.

**P15 — the verdict.** I predict **SURVIVES**, with `P(SURVIVES) = 0.88`, `P(PARTIAL) = 0.11`,
`P(MOVES) = 0.01`. Predicted `max_a (I_a − F_a) = 0.0023` (tierii physics), a factor **4.3 below**
the registered 0.01 PARTIAL threshold.

⛔ **If P13b is wrong — if a non-trivial population of exactly-right queries exists at either cell —
then the mechanism transfers and I expect PARTIAL or MOVES.** That is the falsifier of my own
prediction and it is a single, cheap, measurable count.

## 3. The exact arm / config / seed list I will re-score (bit-for-bit with the published cells)

**Step 1 — the reproduction gate, run FIRST with the EXISTING fitted readers, no new code.**
`orgdiv-null-arms` **N1 @ its selected config** `{lr 0.1, atoms_per_well 64, tau 1.0,
init "written", read "soft"}`, seeds **0,1,2,3,4**, `CatTestConfig(atoms_per_well=32)` (the harness
default), selected configs read from the **banked** `stage_grid.json`. It must reproduce, to the last
digit:

| statistic | published (`stage_score.json`) |
|---|---|
| `native_unseen` | 0.0000 (all 5 seeds) |
| `native_seen` | **1.0000** (all 5 seeds) |
| `readers_unseen` | `sum_linear` 0.0, `well_table` 0.0, `knn` 0.0, `mlp` 0.0 (all 5 seeds) |
| `readers_seen` | `sum_linear` 1.0, `well_table` 0.0, `knn` 1.0, `mlp` 0.0703125 / 0.0703125 / 0.0859375 / 0.0546875 / 0.0546875 |
| `chance` / `bar` / `tol` | 3.90625e-4 / 0.050390625 / 0.478 |
| reader params | 104 / 72 / 0 / 92 |

⛔ **Any mismatch ⇒ STOP and report.** Everything downstream is void until explained.

**Step 2 — `orgdiv-null-arms`, identity added to the class.** Configs (6 cells × 5 seeds):
1. **N1 @ selected** `{lr 0.1, a=64, tau 1.0, written, soft}` — the "memorises perfectly, composes
   not at all" arm (1.0000 SEEN / 0.0000 unseen);
2. N1 @ grid-max argmax `{lr 1e-3, a=12, tau 1.0, written, soft}`;
3. N2 @ grid-max argmax `{product_vq, n_codes 32, fitted, commitment 0.0, lr 0.0}`;
4. N3 @ grid-max argmax `{lr 1e-2, level csb, written, tau 0.2}`;
5. ⭐ **N5 @ the `null*` argmax** `{lr 3e-3, h=64, momentum 0.9, decay 0.01, gate none, chunk 1}` —
   the computed grid-max, `null* = 0.00117`;
6. N4 @ grid-max argmax `{set_code, k=2, idw}` (⚠ the noiseless-key variant, flagged as such).
   N4/N5 have **no latent and no codebook** — their read is already zero-parameter (`native`), which
   is reported as the identity column rather than recomputed.
Seeds **0–4**, `PRNGKey(2000+seed)` / `fold_in(·,1)` launches, φ `build_phi(cfg)` — the harness's own
frozen path, unchanged. ⛔ **The 584-config grid is NOT re-scored** (task §Scope 2).

**Step 3 — `tierii-read-fix` iteration 1**, `stage arms`, seeds **0–4**, `registered_cfg()` +
`registered_mw(k_particles=12)`, `organize_steps=60`: its **physics arm**, its **null `N1′`** and its
**live launder `L_a`**, each scored through fitted **and** identity members side by side.

**Step 4 — `orgdiv-cat-test`'s physics arm** — run **iff** steps 2–3 are cheap and clean; otherwise
declared NOT-RUN with its reason.

## 4. Constraints I am bound by (restated so a reviewer can check them)
1. The identity members are **0 parameters**, fit on **nothing** (`fit` ignores `y`), inside the
   `< N_a·m = 256` capacity bound, and applied **identically** to every arm, null and launder.
2. They are **ADDED** to the class, never substituted: every table carries fitted **and** identity
   columns, and the default `which=` of every shipped `fit_readers*` is **unchanged**, so all prior
   code paths stay bit-identical.
3. ⛔ **No selection on `Q_unseen` anywhere.** The identity readers have nothing to select.
4. ⛔ **No re-tuning, re-training or re-initialisation of any arm.** Same seeds, same weights, same
   latents, same φ, same launch keys; the ONLY change is which decoder reads them.
5. 5 seeds on every number; declared NOT-RUNs never reported as nulls.
6. ⛔ `PREREG-TierII.md` is **not** edited. An erratum block goes in `ERRATA-TierII.md` **iff** the
   verdict is MOVES or PARTIAL.
