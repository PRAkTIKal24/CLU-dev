# reader-fitting-audit — experiment-engineer report

**Task + acceptance criterion (one line):** decide whether the C2W7 reader-fitting pathology (a
least-squares reader scoring an informative latent at exactly 0 under a thresholded metric) voids
the published zeros of `orgdiv-null-arms`, `tierii-read-fix` and `orgdiv-cat-test` — reproduction
gate first, PREREG before any re-score, fitted vs identity columns side by side, mechanism measured
not assumed, verdict stated mechanically.
**Status: done.** All four scope steps ran (step 4 was **not** declared NOT-RUN — steps 2–3 were
cheap and clean, so `orgdiv-cat-test` ran too). ⛔ **No erratum is owed.**

> ## ⛔ THE ONE-LINE VERDICT
> ## **SURVIVES.** The C2W7 finding is **scoped to the multiplicity read**. C2W5's conclusions stand
> unchanged and the wave's caveat is lifted.
> ⭐ **And the mechanism is CONFIRMED, not refuted** — the lstsq shrinkage reproduces at every cell
> (`diag(W) = 0.13–0.45`, C2W7 measured 0.40) and the `tol` crossing **fires** on two of them. It
> simply has **nothing to destroy**: the population it destroys — queries whose asserted set is
> *exactly right* — is **2 of 2560** at `orgdiv-null-arms`, **3 of 1280** at `tierii-read-fix` and
> **0 of 2560** at `orgdiv-cat-test`, against C2W7's **~18 %**. Largest identity-minus-fitted gain
> anywhere: **+0.0023** (tierii physics), **21× below** the 0.05 bar and **4.3× below** the
> pre-registered 0.01 PARTIAL threshold.

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. ⭐ **`c2w7-read-cardinality` reconciliation 1 must be RE-SCOPED, not withdrawn.** Its sentence
   *"every arm this programme has scored at ≈ 0 may have been scored through a reader that destroys
   its own signal"* is **measured false for C2W5**: three cells, 15 seeds, re-scored, verdict
   SURVIVES. The correct standing statement is the narrower one, which this task confirms:
   *a fitted reader destroys signal **only in proportion to the fraction of queries whose asserted
   set is already exactly right** — which is why it cost C2W7 (18 %) and cost C2W5 nothing (0.1 %).*
   ⭐ **That fraction is a one-line pre-condition, computable before any reader is fitted**, and it
   is the thing that should be registered (proposed **K6**, §8).
2. **The published `orgdiv-cat-test` §6.1 SEEN column (`sum_linear` 0.0109 / `well_table` 0.0031) is
   NOT in `stage_arm.json`** — that artifact carries the unseen `physics` column only. I gated on
   what the artifact contains (unseen, bit-for-bit, all 5 seeds, including the non-zero seeds 1–2);
   my measured SEEN is 0.0094 / 0.0047. Someone should locate the provenance of the report's SEEN
   column or footnote it.
3. **`c2w7-read-cardinality`'s proposed §7 Known-Issue entry needs the qualifier added** before it
   reaches the handover as written (§9).

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial:** ⛔ **none: instrument audit / re-measurement.** No tier-ii claim is made or revived, no
  arm is resurrected, no new read protocol is proposed.
- **Laundering control:** the re-score is itself the control on a published instrument. Its own
  controls are (i) ⭐ **three reproduction gates** (every published fitted column re-derived
  bit-for-bit before a single identity number was looked at), (ii) the identity reader applied
  **identically** to every arm including the nulls and the live launder, (iii) a **designed
  positive control** for the identity reader's own failure mode (`PREREG` P14), pytest-asserted.
- **Falsifies the claim (SURVIVES):** any re-scored arm's identity score clearing `chance + 0.05`
  by 2 SE ⇒ MOVES; or an identity-minus-fitted gain ≥ 0.01 at > 2 SE ⇒ PARTIAL. Both registered
  numerically **before** the harness ran.
- **Does NOT falsify:** the arms still reading ≈ 0 is not news and is not mine to defend; the
  identity reader losing to a fitted reader on an in-sample or mis-scaled cell is the reader's known
  assumption failing, **stated in advance** (P14), not a defect.

---

# 1. ⭐⭐ THE THREE REPRODUCTION GATES — ALL PASSED, BIT-FOR-BIT

⛔ Step 1 of the scope, run **first**, with the **existing** fitted readers and **no new code**
(the identity readers did not exist yet — commit `d1149a4`, the untouched base).

| gate | cell | statistic | published | **re-measured** | verdict |
|---|---|---|---|---|---|
| **G-A** ⭐ | `orgdiv-null-arms` **N1 @ selected** `{lr 0.1, a=64, τ=1.0, written, soft}`, seeds 0–4 | `native_unseen` | 0.0000 ×5 | **0.0000 ×5** | ✅ |
| | | `native_seen` | **1.0000 ×5** | **1.0000 ×5** | ✅ |
| | | `readers_unseen` | 0/0/0/0 ×5 | **0/0/0/0 ×5** | ✅ |
| | | `readers_seen` (`sum_linear`/`well_table`/`knn`/`mlp`) | 1.0 / 0.0 / 1.0 / **0.0703125, 0.0703125, 0.0859375, 0.0546875, 0.0546875** | **identical, every digit, every seed** | ✅ |
| | | reader params · chance · bar · tol | 104/72/0/92 · 3.90625e-4 · 0.050390625 · 0.478 | **identical** | ✅ |
| **G-B** | `tierii-read-fix` `stage arms`, seeds 0–4 | all 5 fitted readers × {physics, null `N1′`, launder `L_a`} | §3's table | **0 diffs at 1e-12** | ✅ |
| | | `OD_min` · `G1_min` | −0.00078125 · −0.0015625 | **−0.00078125 · −0.0015625** | ✅ |
| **G-C** | `orgdiv-cat-test` `stage_arm`, **claim cell γ=0.05**, seeds 0–4 | `physics` per reader per seed | incl. the **non-zero** seeds 1–2 (`sum_linear`/`well_table` 0.001953125; `mlp` 0.001953125 on seed 1 **only**) | **0 diffs at 1e-12** | ✅ |

⭐ **G-C and G-B are non-trivial reproductions** (they match non-zero, seed-asymmetric values, not
just a field of zeros). ⭐ **A fourth, unplanned reproduction fell out of the audit:** re-running N5
at the published `null*` argmax reproduces `null* = 0.001171875` **exactly**, per-seed
`[0, 0.00391, 0.00195, 0, 0]` — matching the published grid-max mean *and* its "best single seed"
0.00391.

⛔ **Nothing was stopped.** Everything downstream is therefore in force.

---

# 2. Flag provenance (mandatory — every quantitative result in this report)

Commits `4138c9b` · `7763e95` · `a93819e` on branch **`reader-fitting-audit`**, base local
`main @ d1149a4`, worktree `../CHLU-readeraudit`, **main venv reused** (protocol §4, w6 lesson):
**jax 0.9.0, equinox 0.13.4, numpy 2.4.1, float32, CPU**. **Seeds 0–4 on every number.**

⛔ **The only change to any scored cell is which decoder reads it.** No arm was re-tuned, re-trained
or re-initialised; same configs, same seeds, same frozen φ, same launch keys, same fits, same stores.

| cell | flags in effect (all inherited unchanged from the published run) |
|---|---|
| `orgdiv-null-arms` | `CatTestConfig(atoms_per_well=32)`; `n_wells/f_subset/n_items/n_unseen = 32/4/128/512`; `d = 4`, `m = 8`; `payload_radius = atom_payload_init_radius = 1.0`; `ball/launch/σ_q = 2.0/0.6/0.15`; `P = 4`; `s = 0.318`, `target_ds = 2.7`, `depth_ratio = 3.0`; φ `build_phi(cfg)`; launches `PRNGKey(2000+seed)` / `fold_in(·,1)`; **`tol = 0.478`, chance `3.906e-4`, bar `0.05039`**; readers fitted on SEEN only; grain **1/2560 = 3.9e-4** |
| `tierii-read-fix` | `registered_cfg()`: `a=32`, **`d = 8`**, `s_measured = 0.2879`, `payload_radius = atom_payload_init_radius = 0.5`, `n_unseen = 256`; `registered_mw(k_particles=12)`, `payload_ref 0.5`, `conf_w 0.05`, `head_settle 60+120`, read budget **400+800** Verlet steps, `dt = 0.05`, `γ_address/γ_read = 0.05/0.02`, `occ_tau 0.25`, `dedupe = noisy_or`, `descent_gate = True`, `kinetic = newtonian_learned`; organizer **60 Adam @ 3e-3 through the settle**, null `N1′` **400 Adam @ 1e-3**, `τ=1.0`, `init="written"`; **`tol = 0.2338`, chance `0.0000`, bar `0.05`**; grain **1/1280 = 7.8e-4** |
| `orgdiv-cat-test` | `CatTestConfig(atoms_per_well=32)`, `build_physics_arm` (place → write → `organize_physics`), read at the **claim cell `γ_address = 0.05`** (⚠ `γ = 0.2` is the internal VQ-collapse control and was **not** re-scored); launches `PRNGKey(2000+seed)`; **`tol = 0.478`, chance `3.906e-4`**; grain 1/2560 |
| the identity readers | **0 fitted parameters** each · fitted on **nothing** (they ignore `y`; pytest-asserted with a shuffled-`y` invariance test) · inside the storeless bound `N_a·m = 256` · **0 bytes** of state · read cost `N_a·m = 256` mult-adds/query, i.e. **3.7e-6** of the null-arms physics read (6.88e7) and **1.0e-6** of the tierii read (2.507e8) |

---

# 3. ⭐⭐ `orgdiv-null-arms` — THE DECISION-CRITICAL CELL (6 cells × 5 seeds, `Q_unseen`)

`chance = 3.906e-4` · **bar = 0.05039** · clears iff `mean − 2 SE > bar`. **Fitted and identity
columns side by side; nothing is re-based.**

| cell | `sum_linear` (104) | `well_table` (72) | `knn` (0) | `mlp` (92) | ⭐ **`well_identity` (0)** | ⭐ **`sum_identity` (0)** | **best fitted** | **best identity** | **I − F** | clears? |
|---|---|---|---|---|---|---|---|---|---|---|
| ⭐ **N1 @ selected** (the "memorises perfectly, composes not at all" arm) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | **0.0000** | 0.0000 | **0.0000** | **+0.0000 ± 0.0000** | ⛔ no |
| N1 @ grid-max | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00039 | 0.00039 | 0.0000 | **0.00039** | +0.00078 ± 0.00096 | ⛔ no |
| N2 @ grid-max | 0.0000 | 0.00078 | 0.0000 | 0.0000 | 0.0000 | 0.00039 | 0.00078 | 0.00039 | **−0.00039 ± 0.00146** | ⛔ no |
| N3 @ grid-max | 0.0000 | 0.0000 | 0.00039 | 0.00039 | **0.00117** | 0.00078 | 0.00039 | **0.00117** | +0.00039 ± 0.00146 | ⛔ no |
| N4 @ grid-max (⚠ noiseless-key `set_code`) | — no latent — | | | | *its read is already 0-param:* | **0.00078** | — | 0.00078 | **0 by definition** | ⛔ no |
| ⭐ **N5 @ the `null*` argmax** | — no latent — | | | | *its read is already 0-param:* | **0.00117** | — | 0.00117 | **0 by definition** | ⛔ no |

- ⭐ **`null*` is unchanged: 0.00117.** The zero-parameter re-score's maximum over all six cells is
  **0.00117**, identical to the published grid-max, and **43× short of the bar** — the published
  headline number does not move by one query.
- ⭐ **N1 @ its selected config — the sharpest fact of the published report — is COMPLETELY
  UNMOVED.** `1.0000` on SEEN and **`0.0000` on unseen through the zero-parameter reader too**, on
  5/5 seeds, with the identity-minus-fitted gain **exactly 0.0000 with zero variance**. "It
  memorises perfectly and composes not at all" was never a statement about the reader.
- ⛔ **N4/N5 are reported as `0 by definition`, not as a gain.** Their published read *is* already
  zero-parameter (no latent, no codebook), so there is no fitted twin to beat; scoring them against
  an absent fitted column would have manufactured a spurious +0.00117 (an earlier revision of my own
  aggregate did exactly that — caught and fixed in `7763e95`, and the fix is recorded here rather
  than hidden).
- **Max identity-minus-fitted across all six cells: +0.00078 (2 queries of 2560).**

---

# 4. `tierii-read-fix` — physics · null `N1′` · live launder, 9 readers × 3 arms × 5 seeds

`chance = 0.0000` · bar `0.05` · `tol = 0.2338` · grain 1/1280. ⭐ **The five fitted columns
reproduce the published §3 table with zero diffs** (gate G-B).

| reader | params | **physics** | **null `N1′`** | **live launder `L_a`** |
|---|---|---|---|---|
| `sum_linear` | 136 | 0.0000 | 0.0000 | 0.0000 |
| `well_table` (hard) | 72 | 0.00078 ± 0.00156 | 0.0000 | 0.00078 ± 0.00156 |
| `knn` | 0 | 0.0000 | 0.0000 | 0.00078 ± 0.00156 |
| `mlp` | 108 | 0.0000 | 0.0000 | 0.0000 |
| `soft_well_table` (D8 twin) | 72 | 0.0000 | 0.00078 ± 0.00156 | 0.00078 ± 0.00156 |
| ⭐ **`gated_well_identity`** (NEW, **0**) | **0** | ⭐ **0.00234 ± 0.00469** | 0.0000 | 0.0000 |
| ⭐ `soft_well_identity` (NEW, **0**) | **0** | 0.00078 ± 0.00156 | 0.0000 | 0.0000 |
| ⭐ `well_identity` (NEW, **0**) | **0** | 0.0000 | 0.0000 | 0.0000 |
| ⭐ `sum_identity` (NEW, **0**) | **0** | 0.0000 | 0.0000 | 0.00078 ± 0.00156 |
| — | | **best fitted 0.00078 · best identity 0.00234** | 0.00078 · 0.0000 | 0.00156 · 0.00078 |

- ⭐⭐ **`gated_well_identity` recovers R1's statistic EXACTLY.** Its physics score is
  **0.00234 ± 0.00469**, per seed `[0, 0, 0, 0.01172, 0]` — bit-identical to the published
  `exact_set_occupancy_gated = 0.0023 ± 0.0047`, per seed `[0, 0, 0, 0.01172, 0]`. **P7 confirmed to
  the last digit.** This is the derivation of the PREREG working: a zero-parameter reader of the
  asserted set scores *exactly* the asserted set's exact-set accuracy, because one wrong well costs
  ~3× `tol`.
- ⛔ **The "vacuous tie" is a vacuous tie under BOTH reader classes.** `OD_min` with the identity
  members included is **−0.00078 ± 0.00156** — **identical** to the published fitted-only value. The
  §3 verdict does not move.
- ⛔ **G1 still fires, and adding the identity readers makes it slightly WORSE, not better:**
  `G1_min` goes **−0.0016 → −0.0023** (per seed `[0, 0, −0.0039, 0, −0.0039]` →
  `[−0.0039, 0, −0.0039, 0, −0.0039]`), because on seed 0 the *launder's* `sum_identity` scores
  0.0039 while the physics arm's scores 0. ⭐ **The zero-parameter reader does not rescue the
  store's dividend; it deepens the deficit.** Anyone hoping the fitting artifact was hiding a
  physics dividend has that hope measured and refuted.
- **`S_eff = 16.00` and `exact_set_occupancy_raw = 0.0000` both reproduce.**

---

# 5. `orgdiv-cat-test`'s physics arm — ⭐ RUN (not declared NOT-RUN), claim cell `γ = 0.05`

| reader | params | **unseen** | **SEEN** |
|---|---|---|---|
| `sum_linear` | 104 | **0.00078 ± 0.00096** | 0.0094 |
| `well_table` | 72 | **0.00078 ± 0.00096** | 0.0047 |
| `knn` | 0 | 0.0000 | 0.9969 |
| `mlp` | 92 | 0.00039 ± 0.00078 | 0.0016 |
| ⭐ `well_identity` (NEW) | **0** | **0.0000 ± 0.0000** | 0.0000 |
| ⭐ `sum_identity` (NEW) | **0** | **0.0000 ± 0.0000** | 0.0031 |
| — | | best fitted **0.00078** · best identity **0.0000** · **I − F = −0.00078 ± 0.00096** | |

⛔ **Here the identity reader is strictly WORSE than the fitted one** (0.0000 vs 0.00078, on 2 of 5
seeds), which is reported as a negative result and is part of the doctrine evidence (§7). The
published `0.0008 ± 0.0008` stands, and K5's kill is untouched.

---

# 6. ⭐⭐ THE MECHANISM — MEASURED, NOT ASSUMED (both named suspects)

The task named two suspects: **the `diag(W)` shrinkage** and **the `tol` crossing**. Both were
measured on every cell with `null_arms.shrinkage_report`, never assumed.

| cell | fitted / identity twin | **`diag(W)` mean** | ⭐ **queries whose asserted set is EXACTLY right** | `tol` | **crossing fires?** |
|---|---|---|---|---|---|
| null-arms **N1 @ selected** | `well_table` / `well_identity` | **+0.276** | **0 / 2560** | 0.478 | no — *nothing to destroy* |
| null-arms N1 @ grid-max | " | **+0.446** | **0 / 2560** | 0.478 | no |
| null-arms N2 @ grid-max | " | **+0.376** | **0 / 2560** | 0.478 | no |
| ⭐ null-arms N3 @ grid-max | " | **+0.357** | **2 / 2560** | 0.478 | ⭐ **YES (seed 2)** |
| ⭐ **tierii physics** | `soft_well_table` / `gated_well_identity` | **+0.430** | **3 / 1280** | 0.234 | ⭐ **YES (seed 3)** |
| tierii physics | `well_table` / `well_identity` | **+0.128** | **0 / 1280** | 0.234 | no |
| cat-test physics | `well_table` / `well_identity` | **+0.218** | **0 / 2560** | 0.478 | no |
| cat-test physics | `sum_linear` / `sum_identity` | ⚠ **+46.53** | **0 / 2560** | 0.478 | no |
| *(C2W7's own cell, for reference)* | `count_table` / `count_identity` | *0.40* | ***~18 %*** | *0.234* | *YES* |

## 6.1 ⭐ THE CROSSING, ON THE TWO CELLS WHERE IT FIRES — C2W7's mechanism, exactly

| | **`orgdiv-null-arms` N3 @ grid-max, seed 2** | **`tierii-read-fix` physics, seed 3** |
|---|---|---|
| `diag(W)` of the fitted reader | **0.3624** | **0.4455** |
| queries with the asserted set exactly right | **2 of 512** | **3 of 256** |
| **identity residual on those queries** | ⭐ **0.0000** (exact) | ⭐ **0.0000** (exact) |
| **fitted residual on those queries** | **1.3539** | **0.5737** |
| `tol` | 0.4660 | 0.2320 |
| fitted residual / `tol` | ⭐ **2.91×** | ⭐ **2.47×** |
| kept by identity / by fitted | **2 / 0** | **3 / 0** |
| cell accuracy, identity vs fitted | **0.00391 vs 0.0000** | **0.01172 vs 0.0000** |

⭐⭐ **This is C2W7's pathology, reproduced end-to-end on C2W5 cells, with its own numbers.** The
shrinkage is real, the crossing is real, the identity reader keeps *every* correct query and the
fitted reader keeps *none*. ⛔ **And it is worth 2 and 3 queries respectively**, because the
population it acts on is 0.08 % and 1.2 % rather than 18 %.

⚠ **A second, opposite fitting pathology was found and is reported:** at the cat-test cell
`sum_linear`'s lstsq **inflates** rather than shrinks (`diag(W) = +46.5`), driving its mean residual
to **13.48** against the identity's 1.84. Least squares against a thresholded metric is unstable in
*both* directions when the latent's payload block is far off scale. It changes no verdict (both are
≈ 0 at `tol`) but it is the reason the fitted reader's *residual* must never be quoted as evidence
that a store is "close".

## 6.2 The derivation that made all of this predictable in advance (PREREG §2)
Payloads are drawn on a sphere of radius `R` (`factored_store.build_family` L347–349) and
`tol = 0.25·RMS‖y − ȳ‖`. A single substituted well costs `‖v_a − v_b‖ ≈ √2 R`:

| cell | `R` | `tol` | `√2 R` | ratio |
|---|---|---|---|---|
| null-arms / cat-test | 1.0 | 0.478 | 1.414 | **2.96×** |
| tierii | 0.5 | 0.2338 | 0.707 | **3.02×** |

⇒ **a zero-parameter identity reader's score IS the asserted set's exact-set accuracy**, to within
one query. Measured: `gated_well_identity` = `exact_set_occupancy_gated` to 5 decimals (§4). The
prediction was arithmetic, not a guess, and it held.

---

# 7. ⭐ THE DOCTRINE QUESTION — answered with the measurement

> **Should the shipped reader class carry a zero-parameter member everywhere, permanently?**

## 7.1 Does the identity reader ever score *worse*? — **YES. Measured, on real banked cells.**

| where | fitted | identity | **fitted − identity** |
|---|---|---|---|
| ⭐ `orgdiv-cat-test` physics, **unseen** (`sum_linear`/`well_table` vs twins) | **0.00078** | 0.0000 | **+0.00078** (2 of 5 seeds) |
| `orgdiv-null-arms` N2 @ grid-max, unseen (`well_table` vs `well_identity`) | 0.00078 | 0.0000 | +0.00078 |
| `tierii-read-fix` physics, unseen (`well_table` vs `well_identity`) | 0.00078 | 0.0000 | +0.00078 |
| ⭐ `tierii-read-fix` **null `N1′`, SEEN** (`sum_linear` vs `sum_identity`) | **0.8078** | **0.7969** | ⭐ **+0.0109** — the largest cost measured on any real cell |
| ⭐ the **designed** positive control (PREREG P14, pytest-asserted): payload table scaled by `α ∈ {2, 0.5}`, asserted set right on **100 %** of queries | **1.0000** | **≤ 0.0104** | ⭐ **> 0.99** |

## 7.2 The failure mode, named where it can be checked
**An identity reader assumes the latent is already in the target's units and scale.** It has no gain
and no bias. The assumption is **false** whenever:
1. the arm's codebook is *fitted* rather than *written*, or the arm is under-trained, so its payload
   block sits at a different radius than the family's (`N1 @ lr 1e-3`, `N2` with
   `payload_source="fitted"`);
2. the code is not normalised to the target's cardinality — `soft_well_identity` inherits π's mass,
   which sums to the number of occupied wells (~11.3 here), not to `F = 4`. Measured: it scores
   0.00078 where `gated_well_identity`, which *does* commit to a set, scores 0.00234;
3. the payload table is rescaled anywhere in the pipeline (the α control above: the fitted reader
   absorbs α exactly, the identity reader dies).
⛔ **Cost of adding it:** 0 parameters, 0 bytes, `N_a·m = 256` mult-adds/query = **3.7e-6** of the
physics read. **There is no capacity, byte or compute argument against it.**

## 7.3 ⭐ MY RECOMMENDATION (the Advisor rules; this is evidence + a recommendation)
**YES — the shipped reader class should carry a zero-parameter member permanently, on this exact
condition: ADDED, never substituted, and never reported alone.** Three reasons, each measured here:
1. it is **free** (§7.2) and it is the only member whose score has a **closed-form ceiling** (the
   asserted set's exact-set accuracy), which turns "the reader failed" and "the store failed" into
   separable statements;
2. the pathology it guards against is **real and reproduces** (§6.1) — it cost C2W7 a factor 28 and
   would have cost this audit 2–3 queries had it gone the other way;
3. it can be **strictly worse** than its fitted twin (§7.1, up to +0.0109 on a real cell and > 0.99
   on a designed one), so substituting it for the class would itself be a laundering step.
⭐ **And the cheaper half of the recommendation:** register the **exactly-right-set fraction** as a
standing pre-condition (proposed **K6**, §8). It is one line, needs no reader, and it is the
quantity that decides in advance whether a fitting artifact *can* move a number at all — 18 % at
C2W7, 0.08–1.2 % at C2W5.

---

# 8. PREREG SCORECARD (`.claude/outputs/reader-fitting-audit/PREREG.md`, filed before any re-score)

| # | registered | measured | verdict |
|---|---|---|---|
| **gate** | N1 @ selected reproduces to the last digit | **bit-identical, incl. `mlp`'s 5 distinct per-seed values** | ✅✅ |
| **P1** | `well_identity` null-arms N1 @ selected = 0.0000, [0, 0.002] | **0.0000 ± 0.0000** | ✅✅ |
| **P2** | `well_identity` at every grid-max argmax = 0.0000, [0, 0.002] | **0.0000 / 0.00039 / 0.00117** | ✅ in band |
| **P3** | `sum_identity` unseen = published `native` = 0.0000 | **0.0000** | ✅✅ |
| **P4** | `sum_identity` SEEN, N1 @ selected = 1.0000 exactly | **1.0000** | ✅✅ |
| **P5** | `well_identity` SEEN, N1 @ selected = 0.0000, [0, 0.05] | **0.0000** | ✅✅ |
| **P6** | `I − F` on every null-arms arm = 0.0000, [−0.002, +0.002] | **+0.00078 … −0.00039** | ✅ in band |
| **P7** | ⭐ `gated_well_identity` tierii physics = **0.0023**, [0, 0.010] | ⭐ **0.00234**, per-seed identical to `exact_set_occupancy_gated` | ✅✅ **exact** |
| **P8** | `soft_well_identity` tierii = 0.0000, [0, 0.002] | **0.00078** | ✅ in band |
| **P9** | `well_identity` tierii = 0.0000 exactly | **0.0000** | ✅✅ |
| **P10** | `sum_identity` tierii physics = 0.0000, [0, 0.002] | **0.0000** | ✅✅ |
| **P11** | `gated_well_identity` on the launder = 0.0023, [0, 0.015] | **0.0000** | ✅ in band |
| **P12** | `OD_min`/`G1_min` with identity within ±0.01 of published | `OD_min` **−0.00078 (identical)**, `G1_min` **−0.0023** vs −0.0016 | ✅ in band |
| **P13a** | `diag(W)` shrunk, mean ∈ [0.05, 0.70] | **0.128 / 0.218 / 0.276 / 0.357 / 0.376 / 0.430 / 0.446** | ✅ 7/8 in band; ⛔ **1 outlier: `sum_linear` at cat-test INFLATES to +46.5** — a finding (§6) |
| **P13b** | exactly-right population **0 / 2560** (null-arms), **≈ 3 / 1280** (tierii) | **2 / 2560** and **3 / 1280** | ✅✅ (tierii **exact**) |
| **P13c** | where the population is non-empty, identity keeps them and fitted loses them | **2/0 and 3/0**, residuals 0.0000 vs 2.5–2.9× `tol` | ✅✅ |
| **P14** | the α-scaling control: fitted 1.0000, identity 0.0000 | **1.0000 vs ≤ 0.0104** | ✅ (identity is not *exactly* 0 at α=0.5 — 1 query of 96 lands under `tol`) |
| **P15** | ⭐ **SURVIVES** (`P = 0.88`); `max_a(I − F) = 0.0023` | ⭐ **SURVIVES**; **max = 0.00234** (tierii physics) | ✅✅ |

**Score: 17 ✅ · 1 partial-⛔ (P13a's outlier, which is itself a finding).** ⭐ The pre-registered
*derivation* — that the identity reader's score is the asserted set's exact-set accuracy — is the
thing that carried every prediction, and it was exact on P7 and P13b.

---

# 9. ⛔ THE VERDICT, STATED MECHANICALLY AGAINST THE REGISTERED READING

| condition (PREREG §1) | measured | fires? |
|---|---|---|
| **MOVES**: `∃ a : I_a − 2 SE > chance + 0.05` | max `I_a` = **0.00234**, `I_a − 2 SE = −0.0023`; bar 0.05 | ⛔ **NO** |
| **PARTIAL**: `∃ a : (I_a − F_a) ≥ 0.01` and `> 2 SE` from 0 | max gain **+0.00234** (0.0016 at tierii, 0.00078 at null-arms), **and 2 SE straddles 0 everywhere** | ⛔ **NO** |
| ⭐ **SURVIVES** | every arm below its bar; `max(I − F) = 0.0023 < 0.01` | ✅ **FIRES** |

## ⇒ **SURVIVES.** Consequences, per the registered reading:
1. **The C2W7 finding is scoped to the multiplicity read.** `c2w7-read-cardinality` reconciliation 1
   is re-scoped (not withdrawn) per reconciliation item 1 above.
2. **`orgdiv-null-arms`' `null* = 0.00117`, its N1 "1.0000 train / 0.0000 held-out", the C2W5
   read-protocol refutation's attribution, `tierii-read-fix`'s `OD_min = −0.0008 ± 0.0016` vacuous
   tie and `orgdiv-cat-test`'s `0.0008` all STAND, unchanged.**
3. ⛔ **NO `ERRATA-TierII.md` block is owed** (it is owed only on MOVES or PARTIAL). `ERRATA-TierII.md`
   was **not** created and `PREREG-TierII.md` was **not** touched.
4. **The wave's caveat is lifted.**

⚠ **One honest qualifier the Hub should carry with the SURVIVES:** the tie is a tie *at ≈ 0*. Both
reader classes read essentially nothing at these cells, so "the fitted reader did not destroy the
signal" here also means "there was almost no signal to destroy" — measured as 2, 3 and 0
exactly-right queries. The audit rules out a *reader artifact*; it does not make any arm alive.

---

# 10. How I verified (commands + observed output)

```
git worktree add ../CHLU-readeraudit -b reader-fitting-audit main     # base d1149a4
# main venv reused (protocol §4, w6 lesson); jax 0.9.0 / eqx 0.13.4 / numpy 2.4.1
# ⛔ STEP 1, BEFORE any new code existed:
PYTHONPATH=$PWD .venv/bin/python .claude/scratch/reader-fitting-audit/gate_null_arms.py
#   ⭐ REPRODUCTION GATE PASSED: every scored number is bit-identical to the published cell.
PYTHONPATH=$PWD .venv/bin/python -m ruff check chlu/                   # All checks passed!
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-null-arms --quick --stages reader_audit --out-dir …/quick
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-null-arms --stages reader_audit --out-dir …/null_arms
#   verdict SURVIVES  max I-F 0.00078125  bar 0.050390625              (~35 s)
PYTHONPATH=$PWD .venv/bin/python .claude/scratch/reader-fitting-audit/tierii_reader_audit.py
#   verdict: SURVIVES | max identity-fitted: 0.0015625
#   reproduces published fitted columns: True []                       (802 s, 5 seeds)
PYTHONPATH=$PWD .venv/bin/python .claude/scratch/reader-fitting-audit/cat_test_reader_audit.py
#   verdict: SURVIVES | identity-fitted: -0.00078125 | reproduces published: True []   (453 s)
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_reader_identity.py -q   # 12 passed in 5.05 s
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/ -q
#   ⭐ 1409 passed, 0 failed (31 warnings, 2 210.23 s = 36:50) — my 12 included
```
⚠ **On the suite baseline, precisely:** `c2w7-read-cardinality` measured **1379** at `fdab86d`;
`main` has since merged `c2w6-anti-erosion` (`d1149a4`). I did **not** re-run a clean-`main` baseline
— the shared checkout was taken over by a concurrent agent mid-task (§12) and a baseline run costs
another 37 min. So `1409 − 12 = 1397` at `d1149a4` is **arithmetic, not a measurement**; what IS
measured is **1409 passed / 0 failed on my branch**.
Artifacts under `.claude/outputs/reader-fitting-audit/`: `PREREG.md` · `gate_null_arms.json` ·
`null_arms/stage_reader_audit.json` (every cell × seed × reader + both mechanism reports) ·
`tierii_reader_audit.json` · `cat_test_reader_audit.json` · `quick/` · `run_*.log` ·
`pytest_full.log`. Scratch drivers: `.claude/scratch/reader-fitting-audit/{gate_null_arms,
tierii_reader_audit,cat_test_reader_audit}.py`.

**⚠ FAILURES AND MISSTEPS, REPORTED.**
1. ⛔ **I edited the SHARED MAIN CHECKOUT by mistake** on my first code edit (absolute path
   `/Users/user/Desktop/CHLU/chlu/core/null_arms.py` instead of the worktree). Caught immediately by
   a line-count mismatch; `git status` in main showed **only my own 12 added lines** (main had been
   verified clean at spawn and no other worktree was registered), so `git checkout --` reverted
   exactly my own edit and main is clean. **No foreign work was touched.** ⚠ Protocol §3.2's hazard
   is real even when you *have* created the worktree — the shell cwd and the editor path are
   separate things.
2. ⛔ **Two of my own tests failed on first write and both assertions were too strong**, not the code:
   (i) `fitted_within_tol_on_exact == 0` — on the designed toy the shrunk reader still keeps 3 of
   its correct queries, so the assertion is now `fitted_kept < identity_kept`; (ii) the α = 0.5 leg
   of the P14 control gives identity **0.0104**, not 0.0000 (1 of 96 queries lands under `tol` at
   half scale), so the assertion is now `≤ 0.05` with a `> 0.9` gap. Both are recorded in P14's
   scorecard row rather than quietly relaxed.
3. ⛔ **My first aggregate manufactured a spurious +0.00117 "gain" for N4/N5** by comparing their
   zero-parameter native read against an *absent* fitted column (`default=0.0`). Caught before the
   verdict was written, fixed in `7763e95` (`has_fitted_column` is now on every row), and the
   verdict is unchanged either way.

---

# 11. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)
1. **The full 584-config `orgdiv-null-arms` grid.** Per the task's scope §2, only the 5 argmax
   configs + N1 @ selected were re-scored. ⛔ If a future wave wants the grid, it is a **separate
   funded task**, and this audit gives it no reason to run: the mechanism's headroom
   (`n_set_exactly_right`) is a *property of the read protocol*, not of the organizer's
   hyperparameters, and it measured 0–2 of 2560 at four different configs spanning three arms.
2. **`orgdiv-cat-test` at `γ ∈ {0.02, 0.2}`.** Only the registered claim cell `γ = 0.05` was
   re-scored (`γ = 0.2` is the published internal VQ-collapse control and is never a claim cell).
3. **`c2w7-read-cardinality`'s own cells.** Not re-run — they are the *source* of the finding, not
   a subject of the audit.
4. **Any arm revival, any re-tune, any new read protocol.** Explicitly out of scope.
5. **An identity twin for `knn` and `mlp`.** `knn` is already 0-parameter; `mlp` has no natural
   unfitted form. Reported, not invented.
6. **The `γ`, `d`, `k` and `P` axes** — no axis was swept; every cell is the published one.

---

# 12. Git footprint
- **Branch:** `reader-fitting-audit` (off local `main @ d1149a4`), worktree `../CHLU-readeraudit`.
  ⛔ Not pushed, no PR, no merge. `origin` untouched, `clu-dev` untouched. Left for review.
- **Commits** (verified **from the MAIN repo**, protocol §3.2 — the wave-4 lesson):
  - `4138c9b` `[experiment-engineer] the zero-parameter identity readers (C2W7 reconciliation 1)`
  - `7763e95` `[experiment-engineer] exp-null-arms: the reader_audit stage + its CLI hook`
  - `a93819e` `[experiment-engineer] tests: the identity readers' admissibility, the C2W7 crossing, and its own failure mode`
- **Files touched (the declared ownership list, exactly):** `chlu/core/null_arms.py` (**+191/−0**:
  6 new readers/helpers + `shrinkage_report`, `__all__` and one import extended; ⭐ **zero deletions —
  no existing line altered**) · `chlu/core/multiwell_read.py` (**+96/−4**: 4 new members + dispatch in
  `fit_readers_mw`/`apply_reader_mw`; the 4 changed lines are those two dispatch functions, whose
  **default `which=READERS_MW` is unchanged and pytest-asserted bit-identical**) ·
  `chlu/experiments/exp_null_arms.py` (**+191/−1**: one new opt-in stage + its aggregate, wired
  behind `"reader_audit" in stages` which is **not** in the default tuple) ·
  `chlu/cli/experiment_cmd.py` (**+5/−2**: one added `--stages` choice + its help text) ·
  `tests/test_reader_identity.py` (**new**, 257 L, 12 tests).
  `git diff --numstat main..reader-fitting-audit` = **5 files, +740 / −7**.
  ⛔ **Not touched:** `chlu/training/train_cluformer.py`, `chlu/core/blocks.py`, `scripts/csf3/`
  (C2W6/CSF3 territory) · `chlu/core/psi_readout.py` · `chlu/core/factored_store.py` (read-only —
  no additive hunk was needed) · `chlu/core/multiplicity_read.py` (read-only) ·
  `chlu/experiments/exp_tierii_read.py` and `exp_cat_test.py` (**read-only**: their re-scores are
  driven from scratch drivers through their existing public/module API, so no file outside my
  ownership list changed) · `chlu/config.py` · `PREREG-TierII.md`.
- **Rebase:** onto local `main` (⚠ **not** `origin/main`, §7.21) — `Current branch
  reader-fitting-audit is up to date`, base unmoved, no-op.
- **Worktree:** ⭐ verified from the MAIN repo **before** removal (`git -C /Users/user/Desktop/CHLU
  log --oneline main..reader-fitting-audit` = the 3 commits — the wave-4 lesson), then
  `git worktree remove ../CHLU-readeraudit`, then **re-verified after removal** (same 3 commits).
  **wt2 is free.** The branch remains for review.
- **Concurrent work:** ⚠ **none at spawn; one appeared mid-task.** At spawn `git worktree list` was
  main-only and the shared checkout was on `main`, clean (which is why §10.1's accidental edit could
  be reverted safely — it touched only my own 12 lines). By the time I finished, the **shared main
  checkout had been switched to `agent/experiment-engineer/c2w6-phi-leak-docstring` @ `2d3a843`** by
  another agent. ⭐ **Zero file overlap** (verified: their branch touches `chlu/core/blocks.py`,
  `chlu/training/train_cluformer.py`, `tests/test_anti_erosion.py` — exactly the three my task file
  declares NOT mine). Local `main` is still `d1149a4`; my base has not moved; I never touched the
  shared checkout again after §10.1.

---

# 13. Open questions / follow-ups / risks
1. ⭐⭐ **The cheap standing instrument this audit produced is the exactly-right-set fraction.**
   `n_set_exactly_right / n_queries` costs one line, no reader and no fit, and it is *exactly* the
   headroom any reader-fitting artifact has. 18 % ⇒ the artifact can be worth 28×; 0.1 % ⇒ it is
   worth 2 queries. **Proposed K6** (§8, §9 of the handover updates below).
2. **The `sum_linear` inflation (`diag(W) = +46.5`) is unexamined and is not mine.** It says the
   cat-test's settled payload block is ~1/46 of the family's scale — a *store* fact, not a reader
   fact, and possibly the same basin-reach issue `tierii-read-fix` D10 measured at
   `payload_radius/s = 2.8`. Someone owning the physics arm should look.
3. **Risk to how this gets quoted.** "SURVIVES" must never be quoted as *"the readers are fine"* —
   it means *"the fitting artifact cannot have produced these particular zeros"*. The artifact is
   **real** (§6.1, two live firings) and the class now carries the guard. ⛔ And SURVIVES must never
   be quoted without §9's qualifier: the tie is a tie at ≈ 0.
4. **`soft_well_identity` vs `gated_well_identity` is a live design question for any future read.**
   The one that commits to a *set* scores 3× the one that inherits π's mass (0.00234 vs 0.00078) on
   the same latent — the same "commit to a cardinality" lesson C2W7 learned, now visible without a
   cardinality head.
5. **I did not re-score the 584-config grid** and the audit gives a positive reason not to (§11.1),
   but that is an inference, not a measurement. If the Advisor wants it closed measurably, the cheap
   version is to compute `n_set_exactly_right` over the whole grid (no readers) — minutes, not hours.

---

## Proposed handover updates (for the Hub)

- **§7 Known Issues — AMEND the entry `c2w7-read-cardinality` proposed** (*"a least-squares-fitted
  reader can score an informative latent at exactly 0 under a thresholded metric"*). It should ship
  **with its scope and its pre-condition attached**: *the artifact's size is bounded by the fraction
  of queries whose asserted set is already exactly right — 18 % at C2W7 (worth 28×), 0.08 % at
  `orgdiv-null-arms`, 1.2 % at `tierii-read-fix`, 0 % at `orgdiv-cat-test` (worth 2–3 queries). The
  C2W5 zeros were re-scored at 3 cells × 5 seeds through zero-parameter readers and **do not move**
  (`reader-fitting-audit`, verdict SURVIVES).* ⛔ Without the qualifier the entry reads as a
  program-wide invalidation, which is measured false.
- **§7 Known Issues — NEW (open, and the useful half of the above):** *least squares against a
  thresholded metric is unstable in BOTH directions.* Measured `diag(W)`: **0.128–0.446** (shrinkage,
  6 cells) and **+46.5** (inflation, `sum_linear` at the cat-test cell, mean residual 13.48 vs the
  identity's 1.84). ⇒ **a fitted reader's residual is never evidence about the store.**
- **§3 CLI/config — NEW:** `chlu exp-null-arms --stages reader_audit` (opt-in; not in the default
  stage tuple, so no existing invocation changes). New public API:
  `null_arms.{READERS_IDENTITY, READERS_PLUS_IDENTITY, well_identity_*, sum_identity_*,
  fit_readers_plus_identity, apply_reader_plus_identity, score_readers_plus_identity,
  shrinkage_report}` and `multiwell_read.{READERS_MW_IDENTITY, READERS_MW_PLUS_IDENTITY,
  soft_well_identity_*, gated_well_identity_*}`. ⛔ **`READERS`, `READERS_MW` and every default
  `which=` are unchanged and pytest-asserted bit-identical** — no published number moves.
- **§2 architecture — NEW:** `tests/test_reader_identity.py` (12 tests) carries the **designed
  positive AND negative** for the fitting pathology: the C2W7 crossing reproduced as a unit test,
  its empty-population negative, and the identity reader's own α-scaling failure.
- **Registry/doctrine candidates:**
  (i) ⭐ **proposed `K6`** — *before any reader is fitted, report the fraction of queries whose
  asserted set is exactly right.* It is the closed-form ceiling of every zero-parameter reader and
  the exact headroom of every fitting artifact; it costs one line and no store. It would have
  answered this entire task in seconds.
  (ii) ⭐ *a reader class should carry a zero-parameter member, **added and never substituted**, and
  it must be reported beside its fitted twin* — the twin can be **strictly worse** (measured
  +0.0109 on a real cell, > 0.99 on a designed one), so substitution would itself launder.
  (iii) ⭐ *a re-measurement task must open with a reproduction gate on a NON-ZERO, seed-asymmetric
  published value.* All three gates here passed, and two of them (`tierii`, `cat-test`) matched
  seed-asymmetric non-zeros — which is what makes "it reproduces" mean something.
- **Reconciliation owner needed:** item 2 of the list above (`orgdiv-cat-test` §6.1's SEEN column has
  no artifact provenance).
