# gen-metric-invariance — experiment-engineer report

**Task + acceptance criterion (one line):** run the pre-registered configuration-invariance check ("the axis must give
the same value for the same features under different partitions") on **DCI-C**, with raw probe-transfer leakage as a
negative control that *must fail*, and issue a PASS/FAIL verdict against tolerances fixed in `PREREG.md` before any
measurement. ⛔ R1 was not run.

**Status: done.** (Resumed the amended task: the sixth cell `inv_montero_seed_2` completed at 11:52 before this
thread started; this thread verified all six cells, ran the aggregation/verdict, added one test, rebased, and reports.)

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** *none — instrument qualification.* No arm, no rung result, no CLU claim.
- **Laundering control:** the failed axis — raw probe-transfer leakage is scored through the **identical** gate
  functions as the negative control.
- **Falsifies:** DCI-C fails the check ⇒ the ladder has no qualified x-axis and returns to Head + Advisor.
- **Does NOT falsify:** DCI-C being a *weaker* separator than leakage on any single cell; the criterion is invariance,
  not effect size.

## ⛔ Downstream reconciliation list (first-10-lines notice, protocol §5)
Three published statements need an owner (details in §7): (a) the R22/R25 leakage margins are correct in magnitude
but **the sign convention in my tables is `corr − rand`, the opposite of R22's**; (b) **class-count matching does NOT
repair the leakage axis** — the sign inversion survives it (3/6), so `gen-r0-wide-refit` §4.2's "6-vs-15-class
numerator/denominator" is *a* cause but demonstrably **not the whole cause**; (c) the Advisor's chance-corrected
companion is **worse**, not better (P7 falsified in direction).

---

## 1. THE VERDICT

| axis | role | verdict |
|---|---|---|
| `leak_raw` — raw balanced-accuracy transfer leakage | ⭐ **NEGATIVE CONTROL, scored first** | **FAIL** ✅ (the check has teeth) |
| `leak_cc` — chance-corrected `(BA − 1/c)/(1 − 1/c)` transfer leakage | the Advisor's companion | **FAIL** |
| `dci_c` — DCI completeness | ⭐ **the candidate (R26.2)** | **PASS** |

⭐ **DCI-C PASSES the pre-registered configuration-invariance check and is qualified to serve as the ladder's x-axis.**
The check was shown capable of failing first: the axis we already know is broken was rejected by the same code path,
on the same data, before the candidate was scored.

## 2. What was held fixed and what moved
One **cell** = one (configuration, seed) frozen encoder. Inside a cell, `X_store` (32,768 × 2,048) is gathered **once**
and never re-bound; the channel-correlation matrix, dead-channel mask and reference grouping are computed once; `K = 5`
is loaded verbatim from `gen-store-grouping`'s `store_plan.json`. **Six partitions** of the same 2,048 coordinates are
then built from that one correlation matrix and each is fitted, addressed and scored:
`correlation` · `random` · `random_alt` · `well_permuted` · `contiguous` · `balanced_random`.
6 cells × 6 partitions = **36 fits**, ~4.0 h of CPU in total (per-cell seconds in §8).

## 3. Leg-by-leg result

| axis | leg 1a relabelling (tol 1e-12) | leg 2 class-count S, montero | leg 2 wide | leg 3 configuration signs | leg 3 magnitude ratio (band [1/3,3]) | verdict |
|---|---|---|---|---|---|---|
| `dci_c` | **1.28e-16** ✅ | **0.367** ✅ | **0.019** ✅ | **6/6** ✅ | **1.34** ✅ | **PASS** |
| `leak_cc` | 4.44e-16 ✅ | 0.657 ❌ | 1.219 ❌ | **3/6** ❌ | **11.37** ❌ | **FAIL** |
| `leak_raw` | 6.66e-16 ✅ | 0.244 (sign clause ❌) | 0.469 (sign clause ❌) | **3/6** ❌ | 2.02 ✅ | **FAIL** |

`leak_raw`'s leg-2 *magnitude* statistic is inside tolerance; it fails leg 2 on the pre-registered **≥5-of-6 cell
sign-agreement clause**, and leg 3 on the 6/6 sign clause. ⭐ It fails on **sign**, which is exactly the disease.

## 4. ⭐ The margin `m = axis(correlation) − axis(random)`, per configuration, per seed
⛔ Nothing is averaged across a sign inversion; every seed is shown.

| axis | form | montero s0 / s1 / s2 | montero mean | wide s0 / s1 / s2 | wide mean | signs |
|---|---|---|---|---|---|---|
| `dci_c` | full | +0.1662 / +0.1648 / +0.1926 | **+0.1745 ± 0.0128** | +0.1314 / +0.1250 / +0.1334 | **+0.1299 ± 0.0036** | `++++++` ✅ |
| `dci_c` | class-count matched | +0.0994 / +0.0995 / +0.1327 | +0.1105 ± 0.0157 | +0.1299 / +0.1197 / +0.1329 | +0.1275 ± 0.0057 | `++++++` ✅ |
| `leak_cc` | full (mismatched) | −0.7238 / −1.0271 / −1.0551 | −0.9353 ± 0.1500 | +0.0448 / +0.0968 / +0.1052 | +0.0823 ± 0.0267 | `−−−+++` ❌ |
| `leak_cc` | matched | −1.0321 / −1.8956 / −1.7207 | −1.5495 ± 0.3727 | +0.1558 / +0.1748 / +0.2171 | +0.1826 ± 0.0256 | `−−−+++` ❌ |
| `leak_raw` | full (mismatched) | −0.0512 / −0.1624 / −0.1873 | −0.1337 ± 0.0592 | +0.0458 / +0.0746 / +0.0778 | +0.0660 ± 0.0144 | `−−−+++` ❌ |
| `leak_raw` | matched | −0.0299 / −0.1054 / −0.1678 | −0.1010 ± 0.0564 | +0.0883 / +0.0910 / +0.1118 | +0.0970 ± 0.0105 | `−−−+++` ❌ |
| `mig` (secondary) | full | +0.1629 / +0.1491 / +0.2349 | +0.1823 ± 0.0376 | +0.1468 / +0.1287 / +0.1665 | +0.1473 ± 0.0154 | `++++++` |
| `dci_d` (secondary) | full | −0.0180 / −0.0266 / +0.0104 | −0.0114 ± 0.0158 | −0.0037 / −0.0203 / −0.0415 | −0.0218 ± 0.0155 | `−−+−−−` ❌ |

⭐ **DCI-C is the only gated axis with 6/6 sign agreement in both forms.** ⚠ Note `dci_d` (DCI *disentanglement*) has a
**single inverted seed** (montero s2, `+0.0104`) and would FAIL leg 3 — it is reported, not averaged away, and is a
standing reason not to substitute D for C.

## 5. ⭐ Class-count stress — the specific failure mode the old axis died of
The wide corners fix `scale`/`orientation`, so the shipped leakage ratio scores a small-class numerator against a
15-class denominator. Realised counts, measured:

| run | numerator rows | numerator orientation/scale classes | denominator rows | denominator classes | rows surviving class-matching |
|---|---|---|---|---|---|
| montero s0/s1/s2 | 270 / 271 / 271 | **3 / 3** | 1024 | 15 / 8 | 69 / 63 / 66 |
| wide s0/s1/s2 | 3103 / 3112 / 3120 | **6 / 6** | 1024 | 15 / 8 | 314 / 310 / 296 |

**Is DCI-C immune, and by how much?** Two independent answers, both measured:

1. **Structural (P8):** DCI-C handed the **full grid's** `cardinalities` vs the **realised** ones, on identical
   restricted rows — max |Δ| over **36 cells = 3.82e-17**, exactly zero in 24/36. ⭐ **DCI-C does not read a declared
   class count at all**; its normalisers are `log K` (a ladder constant) and the realised `H_f`. The ratio axes have no
   such property: their denominator *is* a second evaluation set.
2. **Empirical (leg 2):** restricting the evaluation rows to the numerator's realised classes moves the DCI-C margin by
   `S = 0.367` (montero, a **3×3-class** restriction keeping 2.4k of 32.8k rows) and `S = 0.019` (wide, 6×6). Under the
   **PREREG-literal** 6×6 budget (`class_budget = {orientation: 6, scale: 6}`, ~9.8k rows) the same statistic is
   **0.057 (montero) / 0.088 (wide)** — see §7 P5. Sign never moves, in any of the 6 cells, under either operator.

**The comparison that matters:** under the identical operator the *ratio* axes change their margin by `S' = 0.32`
(`leak_raw`, both configurations) and `0.40 / 0.55` (`leak_cc`) — **and, crucially, they are still inverted afterwards**
(next section). DCI-C's immunity is exact at the level of the declared class count and ≤0.09 empirically at the
pre-registered budget.

## 6. ⭐ The negative control fails — and fails *more deeply* than R25 diagnosed
`leak_raw` reproduces the published inversion exactly: montero **−0.1337 ± 0.0592**, wide **+0.0660 ± 0.0144**
(R22 `+0.1337 ± 0.0592` / R25 `−0.0660 ± 0.0144`; ⚠ identical magnitudes to 4 d.p. **including the seed sd**, opposite
sign convention — my margin is `corr − rand` of the *leakage*, and **less leakage is better**, so my `−0.1337` and
R22's `+0.1337` are the same statement: on Montero the clustered grouping leaks less, on wide it leaks more. 3/3 seeds
each side, magnitude ratio 2.02×.)

⭐ **New finding: matching the class counts does not repair it.** With numerator and denominator restricted *together*
to the numerator's own realised classes, `leak_raw` is still montero **−0.1010 ± 0.0564** vs wide **+0.0970 ± 0.0105**
— 3/6 sign agreement, a *tighter* magnitude ratio (1.04) around an inverted sign. So the 6-vs-15-class mismatch
diagnosed in `gen-r0-wide-refit` §4.2 is **a** contributor, not **the** cause: the leakage axis inverts between
configurations even when the class-count nuisance is removed. ⛔ This closes the "fix the mismatch and the transfer axis
comes back" route.

## 7. Pre-registered predictions, scored (⛔ nothing below was re-tuned)

| id | prediction | measured | verdict |
|---|---|---|---|
| P1 | `leak_raw` FAILS, leg 3 fails on sign | FAIL, 3/6 signs | ✅ (⚠ not blind, §1 of PREREG) |
| P2 | `leak_raw` leg 2b `S' ≥ 1.0` **and** sign flips under restriction | `S' = 0.323 / 0.319`; sign does **not** flip within a configuration | ❌ **falsified** — the artefact is cross-configuration, not within-cell |
| P3 | `leak_raw` PASSES leg 2 in the matched form | matched form still inverted across configurations (3/6) ⇒ leg 2's sign clause fails | ❌ **falsified**, and this is the §6 finding |
| P4 | `dci_c` leg 1a exactly `Δ = 0.0` | `1.28e-16`, exact in **0/36** cells | ❌ on the strict clause, ✅ on the gate (4 orders inside 1e-12). Cause: summation order in `_mi_bits`, i.e. float ulp, **not** a class-count dependence |
| P5 | `dci_c` leg 2 `S ≤ 0.25` | **0.057 / 0.088** under the PREREG-literal 6×6 operator ✅; **0.367 / 0.019** under the harsher numerator-realised operator (montero = 3×3 classes, 2.4k rows) | ✅ as pre-registered; ⚠ the harsher variant exceeds 0.25 while clearing the gate 0.50 |
| P6 | `dci_c` leg 3: 6/6 positive, ratio ≈0.74 | 6/6 positive, ratio **1.34** — the gate reports `max/min`, and `1/1.343 = 0.745`, i.e. the predicted number to 3 s.f. | ✅ |
| P7 | `leak_cc` FAILS leg 2b with `S'` **reduced 1.5–4×** vs `leak_raw` | FAILS ✅, but `S'` is **increased** ×1.23 (montero) / ×1.72 (wide), and leg 3's magnitude ratio blows out to **11.37** | ⚠ **half-falsified** — ⭐ the chance correction makes the class-count dependence **worse** |
| P8 | `dci_c` bit-identical under declared vs realised cardinalities | max Δ **3.82e-17** over 36 cells | ✅ |
| P9 | reproduce shipped R0 DCI-C to ≤1e-6 | reproduced to **≤2e-5** (0.1903/0.0241, 0.1859/0.0211, 0.2142/0.0216); wide margin +0.1299 ± 0.0036 vs published ±0.0036 | ⚠ ✅ in substance, the 1e-6 clause missed (published values are 4-dp rounded) |
| P10 | `well_permuted` ≈ `correlation` within σ_pipe | montero `+0.0001` mean, wide `−0.0070` mean; max cell 0.0082 = σ_pipe itself | ✅ |
| P11 | `contiguous`, `balanced_random` ≤ `random ± 3σ` | ⚠ montero `contiguous = 0.0400` and `balanced_random = 0.0357` vs `random + 3σ = 0.0262` — **above**; wide both inside | ❌ **falsified on montero** — see below |
| P12 | `σ_pipe ≤ 0.02` DCI-C units | **0.0082** (max cell), 0.0050 mean | ✅ |

**P11 is a finding, as pre-registered.** DCI-C reads a small amount of *well-size / index-contiguity* structure beyond
pure chance (montero: 0.040 contiguous vs 0.022 random). ⭐ It does not touch the verdict — the correlation arm is
**0.197**, i.e. `corr − contiguous = +0.157 ± 0.017` (3/3 seeds) and `corr − balanced_random = +0.161` (3/3) on montero,
`+0.110` and `+0.142` on wide — but the honest statement is that `random` is not the *only* floor, and a future arm that
merely reorders channels contiguously would score ~0.04, not 0.00.

### Every partition, DCI-C (the features never moved)
| partition | montero mean ± sd | wide mean ± sd |
|---|---|---|
| `correlation` | **0.1968 ± 0.0124** | **0.1630 ± 0.0061** |
| `well_permuted` (same set partition, relabelled ⇒ re-fitted) | 0.1967 ± 0.0112 | 0.1700 ± 0.0048 |
| `contiguous` | 0.0400 ± 0.0044 | 0.0532 ± 0.0107 |
| `balanced_random` | 0.0357 ± 0.0145 | 0.0211 ± 0.0086 |
| `random_alt` | 0.0239 ± 0.0025 | 0.0256 ± 0.0046 |
| `random` | 0.0223 ± 0.0013 | 0.0330 ± 0.0079 |

⭐ `well_permuted` vs `correlation` is the single sharpest line in the table: **the same set partition, a different well
order, an independently re-fitted store** reproduces DCI-C to `0.0001` (montero) / `0.0070` (wide). That *is* the
criterion, measured end-to-end through the pipeline rather than at the arithmetic.

## 8. Flag provenance
Base `main @ 1bcfcb9` (⚠ the task file names `3faef27`; local `main` had fast-forwarded to `1bcfcb9`, the
`gen-baselines` merge, before this spoke branched — declared in `PREREG.md` §0, not chosen). Branch
`agent/experiment-engineer/gen-metric-invariance`, cells produced at `7d11d77`.

| cell | condition | encoder param hash | feature-cache sha256 | cache scope | numerator rows | seconds |
|---|---|---|---|---|---|---|
| montero s0 | `recomb2element` | `98d326028316ae2c` | `d6728f01d102` | full | 270 = eval_b(72) ∪ matched_seen(198) | 163 (resumed) |
| montero s1 | `recomb2element` | `fc05e2c5f39694b2` | `484b446c4dbb` | used | 271 | 2967 |
| montero s2 | `recomb2element` | `154d498610648fac` | `8c0a69e88beb` | used | 271 | 4936 |
| wide s0 | `recomb2element_wide` | `e159c6b5257ed125` | `cbc713adc137` | full | 3103 = 13 cells, pos ∪ neg | 1210 |
| wide s1 | `recomb2element_wide` | `5b7a04d18f8e59d4` | `910be2d19445` | full | 3112 | 1896 |
| wide s2 | `recomb2element_wide` | `35b72d3891e2fb46` | `9c53bcb8cdb0` | full | 3120 | 2971 |

Non-default / pinned flags, identical in all 36 fits: `K = 5` (⛔ loaded verbatim from `store_plan.json`, never
re-derived) · `D = 2048` · `n_modes = 64` · `n_store_train = 32768` · `n_corr = 16384` · `train_steps = 1200` ·
`n_probe_train = 8192` · `probe_steps = 400` · `grouping_method = ward` · seeds `{0,1,2}` ·
`gen_metric_invariance.relabel_tol = 1e-12` · `class_count_tol_ratio = 0.5` · `class_count_min_sign_agreement = 5` ·
`configuration_band = 3.0` · `class_budget = {orientation: 6, scale: 6}` · `random_alt_offset = 1000` ·
`negative_control_axis = leak_raw` · `candidate_axis = dci_c` · `jax_enable_x64` default (off) ·
JAX from the **main venv** (⛔ no `uv sync` in the worktree). Store key per partition:
`PRNGKey(seed·10007 + 20260911 + partition_offset)`. ⛔ No decoder is used anywhere in this spoke — no axis here decodes.
Hold-out hygiene asserted per cell (`assert_no_holdout_leak`, `n_holdout_leaked == 0`; numerator ∩ store_train = ∅).

## 9. How I verified
- `agg.py` over all six cell JSONs (main venv, worktree on `PYTHONPATH`):
  `{"dci_c": "PASS", "leak_cc": "FAIL", "leak_raw": "FAIL"}` · `negative control = leak_raw -> FAIL`.
- Artefacts: `.claude/outputs/gen-metric-invariance/{summary.json, tables.md, invariance_summary.png, PREREG.md,
  METHOD.md, inv_<config>_seed_<s>.json ×6, parts/*.json ×36}`.
- **Features-fixed**, two independent legs: (i) structural — `X_store` is bound once per cell and the 1.97 GB mmap is
  released *before* the partition loop, so no partition can re-gather; (ii) `feature_cache_sha256` recorded per cell;
  (iii) NEW test `test_fitting_a_store_under_a_new_partition_does_not_move_the_features` asserts the store fit does not
  write into the array it is handed, across two different partitions. ⚠ The planned `features_fixed.json` end-to-end
  digest was **not** produced (its script was a casualty of the OOM); the three legs above replace it.
- Suite: see §11.

## 10. What I did (this thread)
1. Verified all six cells present and complete (`inv_montero_seed_2` landed 11:52 from lane C before this thread).
2. Ran the aggregation/verdict, which had never been run over the complete set of six.
3. Scored every pre-registered prediction, including the three that failed (P2, P3, P11) and the two that half-failed
   (P4, P7).
4. Computed the **PREREG-literal** class-count operator alongside the harsher shipped one, because commit `7d11d77`
   changed the operator after `PREREG.md` was written (§12 risk).
5. Added one test + its `_Cfg` fixture; rebased; reported.

## 11. Suite
⭐ **GREEN.** Full `pytest -q` on the **rebased** branch head `cd7c8b4`, main venv (⛔ no `uv sync` in the worktree):

```
2321 passed, 15 skipped, 45 warnings in 3310.70s (0:55:10)
```

- **0 failed, 0 errors.** ⭐ A green *after*-suite over a diff that is **purely additive** (`+785 / −1`, and the single
  deleted line is `"dci_scores"` moving one row up inside `__all__` to stay alphabetical) is a sufficient no-regression
  result; a separate *before* run on `main` could only have confirmed it, so I did not spend a second 55-minute run on
  it. ⚠ Stated as a judgement, not as a measurement: **the before-suite was NOT re-run in this thread.**
  (An earlier partial before-log from 06:50 at the *old* base `1bcfcb9` shows one `F` at ~31 % before it was killed; it
  was never identified, and it is now moot — the same tree at the current base passes 2321/2321.)
- Targeted re-run, named: `pytest tests/test_gen_metric_invariance.py -v` → **27 passed in 13.94 s**, including
  `test_NEGATIVE_CONTROL_a_class_count_ratio_axis_is_rejected`,
  `test_configuration_invariance_fires_on_a_single_inverted_seed`,
  `test_dci_completeness_does_not_read_the_declared_class_count`, and the new
  `test_fitting_a_store_under_a_new_partition_does_not_move_the_features`.
- `ruff check chlu/config.py chlu/core/gen_store.py tests/test_gen_metric_invariance.py` → **All checks passed!**
  ⚠ `ruff format --check` would reformat all three — **so would it on `main`'s own** `chlu/config.py` and
  `chlu/core/gen_store.py` at `423cbfa`, i.e. pre-existing repo state, not introduced here.
- ⚠ **Stray file found and quarantined.** `tests/test_gen_metric_invariance 2.py` (untracked, mode 600, dated
  **00:43** — an editor/sync artefact of this spoke's *first* thread) was sitting in `tests/` during the suite run.
  It contributed **zero** tests: post-removal `--collect-only` gives **2336**, exactly `2321 passed + 15 skipped`.
  Moved to `.claude/scratch/gen-metric-invariance/stray/`, ⛔ not deleted. `git status` is now clean.

### ⚠ Suite-stability note (memory: "suite runs need a stable HEAD")
`main` moved **during** the 55-minute run: `423cbfa → 5fee3f4`. The mover was the Hub, and the commit is
`5fee3f4 [hub] merge gen-metric-invariance (gen-phase1): DCI-C qualified as the ladder x-axis` — ⭐ **it merged this
very branch**, and `git diff cd7c8b4 5fee3f4` is **empty**. So the tree under test never moved, the result stands for
both refs, and no second rebase is needed (my branch is now an ancestor of `main`).
⚠ For the Hub's awareness: **the merge landed before this report existed.** Nothing in the report contradicts the
merge — the verdict is PASS — but three items in §7/§12 (P2, P3, P11 falsified) were not visible at merge time.

## 12. Open questions / follow-ups / risks
1. ⚠ **Declared deviation from `PREREG.md` §4 leg 2.** The PREREG restriction operator was "orientation realises 6 of
   15, scale 6 of 8". Commit `7d11d77` (written *during* the run, before the montero cells were scored) changed the
   operator to *the numerator's own realised classes*, because the Montero corner fixes the **highest**-indexed values
   and the literal 6-of-15 budget (which keeps the lowest-indexed) empties it — `class_count_restriction` returns 0 rows
   (asserted in `test_class_count_restriction_accepts_an_explicit_value_set`). The harness computes **both**; the
   verdict above uses the **harsher** one, and §5/§7-P5 report the PREREG-literal one (`S = 0.057 / 0.088`). ⭐ The
   verdict is unchanged under either. ⛔ No tolerance was retuned.
2. **Montero's leg-2 restriction is small-n**: 2.4k of 32.8k rows, 3 classes of orientation. `S = 0.367` there is
   partly finite-sample. The PREREG-budget number (9.8k rows, 6×6 classes, `S = 0.057`) is the better-powered one.
3. ⭐ **The chance-corrected companion is dead.** P7's direction is falsified: `(BA − 1/c)/(1 − 1/c)` *amplifies* the
   class-count dependence (leg-3 magnitude ratio 11.37 vs `leak_raw`'s 2.02) because dividing by `1 − 1/c` inflates the
   small-`c` arm. The Advisor should be told it is not a route back to a transfer axis.
4. **DCI-C reads a little partition geometry** (P11): `contiguous` scores 0.040 vs `random` 0.022 on montero. Any future
   arm whose grouping is index-contiguous should be compared against `contiguous`, not `random`.
5. ⛔ **Not done, by instruction:** R1. DCI-C is now qualified; the ladder may run.

## 13. Git footprint
- **Branch:** `agent/experiment-engineer/gen-metric-invariance` · **worktree** `../CHLU-gen-inv` (kept — ⛔ not removed;
  the branch ref is verified from the main repo below).
- **Base:** branched off `main @ 1bcfcb9`; **rebased onto local `main @ 423cbfa`** (⛔ never `origin/main`, which is
  frozen at `40c2f31`). Clean rebase, no conflicts.
- **Commits (post-rebase hashes):**
  | hash | subject |
  |---|---|
  | `65f1851` | `[experiment-engineer] the pre-registered metric-invariance check R26 makes standing` |
  | `dbb9521` | `[experiment-engineer] let the class-count operator take the numerator's own realised classes` |
  | `cd7c8b4` | `[experiment-engineer] assert the features do not move when the partition does` (this thread) |
- **Files touched (3):** `chlu/config.py` (+83, −0 — new `GenMetricInvarianceConfig` + its wiring in
  `get_default_config`/`load_config`, appended at the shared anchor, **zero deletions**) ·
  `chlu/core/gen_store.py` (+314, −1 — new section 10 gate API; the deletion is an `__all__` re-sort) ·
  `tests/test_gen_metric_invariance.py` (+388, new file, 27 tests).
- **Verified from the MAIN repo** (protocol §3.2): `git -C /Users/user/Desktop/CHLU log --oneline main..agent/…` is
  **empty because the Hub already merged the branch into `main` at `5fee3f4`**; `git branch --contains cd7c8b4`
  reports `main`, and `git diff cd7c8b4 5fee3f4` is empty. ⭐ The commits are safe on a shared ref.
- **Not pushed, no PR** (per protocol; and moot — already merged locally by the Hub).
- **Unresolved conflicts:** none.
- **Collisions with other agents:** none in tracked files. One untracked stray (`tests/test_gen_metric_invariance 2.py`,
  this spoke's own earlier thread) quarantined to `.claude/scratch/…/stray/`, §11.

## Proposed handover updates (for the Hub)
- **§3 config:** new `GenMetricInvarianceConfig` block (`chlu/config.py`), defaults are **pre-registered tolerances** —
  ⛔ they must not be tuned: `relabel_tol = 1e-12`, `class_count_tol_ratio = 0.5`,
  `class_count_min_sign_agreement = 5`, `configuration_band = 3.0`, `class_budget = {orientation: 6, scale: 6}`,
  `negative_control_axis = "leak_raw"`, `candidate_axis = "dci_c"`.
- **§2 architecture:** `chlu/core/gen_store.py` gains a section-10 gate API —
  `relabelling_invariance` · `class_count_invariance` · `configuration_invariance` · `invariance_verdict` ·
  `class_count_restriction` · `chance_corrected_balanced_accuracy` · `permute_well_codes` · `permute_mode_codes` ·
  `contiguous_grouping` · `balanced_random_grouping`. **Any metric proposed as an axis is run through these.**
- **§7 Known issues — ADD:** *"Probe-transfer leakage is not repaired by class-count matching."* The
  `gen-r0-wide-refit` §4.2 diagnosis (6-class numerator vs 15-class denominator) is a contributor, not the cause: with
  numerator and denominator matched, the margin is still montero −0.1010 vs wide +0.0970 (3/6 signs). ⛔ Do not spend
  another spoke trying to fix the transfer axis by matching class counts.
- **§7 Known issues — ADD:** *"A full-scope feature cache is 1.97 GB on disk / 3.93 GB upcast; two concurrent cells
  OOM a 16 GB machine."* Use `mmap_mode='r'`, gather once, and **release the mmap before the fit loop** (this spoke's
  `inv.py` does; it is what made the resumed run survive).
- **§7 Known issues — RESOLVED/SUPERSEDED:** the ladder's x-axis question. DCI-C is qualified by a pre-registered check
  whose negative control fails; ⛔ metric-shopping is closed.
