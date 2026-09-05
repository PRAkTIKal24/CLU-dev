# c2w8-close-gate-hardening — experiment-engineer report

**Task + acceptance criterion (one line):** repair the census gate's six instrument debts and ship
`GATE-HARDENING-DONE.json` with a mechanical per-item boolean — **`gate_hardening_done = true`,
12/12 items, both mandatory designed negatives pytest-asserted, full suite 1579/0.**
**Status: DONE.**

## ⚠ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — needs an owner at the review that accepts this)

1. ⛔⛔ **REMOVING A3 FROM THE PASS CONDITION FLIPS ARM B's BANKED G-ADDR VERDICT TO PASS.** Arm B's
   **measured** settle-side legs are `A1 = 0.9297 / 0.9844 / 0.8750` (pass 3/3) and
   `A2 = 0.0625 / 0.0 / 0.125` (pass 3/3); it failed **only** on A3. §A33.1 makes A3 a DIAGNOSTIC ⇒
   arm B's banked configuration now passes both MECHANICS legs. **Stated as a mechanical consequence
   of a ratified rule — ⛔ NOT a re-scoring into a claim, ⛔ NOT an arm-race adjudication** (§A30.1:
   the race is VOID and stays unadjudicated; arm B is claim-barred by `NO_TIER_II_CLAIM` regardless).
   **The Hub must decide whether any banked `gate_addr_pass` value is re-quoted or retired.**
2. ⛔⛔ **THE ORIGINAL N1 DESIGNED NEGATIVE WAS BUILT ON THE RETIRED LAUNCH-POINT STATISTIC.**
   `tests/test_gate_addr.py` fed `a2 = 1.0` from arm B's banked `usage.frac_never_read` — exactly the
   `covered`-derived number erratum 1 retired. Rewritten, not deleted, with the truth in the
   docstring. G-ADDR's falsifiability now rests on the **live planted** negatives N1′/N2 (which run
   the shipped code), not on banked arithmetic.
3. ⚠ **BANKED ARM-A / SPINE CELLS ARE NO LONGER BIT-REPRODUCIBLE AT THE SHIPPED DEFAULT.** Every arm
   factory recovers the spacing as `d_safe / d_safe_frac`, so repairing the `d_safe` population also
   moves an arm's **co-scaled atom width** onto the store population. Set
   `d_safe_population = "sizing"` to reproduce banked cells exactly. Declared in code.
4. ⚠ **`n_never_read` / `frac_never_read` banked before this branch are launch-gated numbers and are
   not comparable to post-branch ones.** Captions corrected in code; **the curator owns the banked
   prose** (charter §A32.2's two never-quotes).
5. ⚠ **The task file's base test count (1555 selected at `9e0bb25`) does not reproduce here:** a
   fresh detached worktree at `9e0bb25` collects **1564** with the main venv. Reported as measured,
   not reconciled by assumption.
6. ⚠ **The drift FLOOR's declared value (0.01 × codebook spacing) fires on real banked cells** — the
   pass-3 arm-A seed-0 cell sits at ratio **0.0071** ⇒ it would **FAIL** the repaired leg on the D2a
   side. That is the intended behaviour (§A31.5), but it means the C2W11 census will fail cells the
   pass-3 census passed. **The Hub should confirm the floor before C2W11 scores anything.**

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** **none — instrument repair.** ⛔ No claim cell, no performance number, no verdict, no
  re-scoring of banked results into new claims.
- **Laundering control:** **N/A, and that is now the point** — A3 becomes a diagnostic column, never
  a pass condition (§A33.1).
- **Falsifies:** a repaired leg that cannot fail its own designed negative does not ship.
- **Does NOT falsify:** nothing here is a performance comparison; losing to a launder is not scored.
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94 discipline on every reading.

**Pre-registration:** `.claude/outputs/c2w8-close/PREREG.md`, filed **before** any code or harness
(the drift floor's value + derivation, the guard's repaired rule, the (v) before/after direction).
**P1–P5 all held; none was tuned.** Details in §3.

---

## 1. THE DELIVERABLE — `.claude/outputs/c2w8-close/GATE-HARDENING-DONE.json`

**`gate_hardening_done = true`**, computed **mechanically** as the AND over 12 items
(`make_gate_hardening_done.py`, in the same directory, is the generator).

| item | done | what changed | asserted by |
|---|---|---|---|
| **(i)** two-sided drift leg / floor | ✅ | `well_lifecycle.drift_leg` + `census`: `floor ≤ median(site_drift) < ceiling`, **both fractions of the MEASURED codebook spacing**; `fails_high` vs `fails_low_D2a_table_expressible` named apart; `one_sided_pass2_pass` emitted beside them | ⛔ **designed negative** `test_designed_negative_table_like_store_fails_the_two_sided_drift_leg` + 2 |
| **(ii)** A1 margin-in-SE | ✅ | `margin_in_se_vs_threshold`, `n_correct`, `n_correct_needed`, `reads_to_flip` on every A1 emission (leg, arm verdict, per-seed print) | `test_a1_reports_its_margin_and_reads_to_flip_beside_the_boolean` + 1 |
| **(iii)** scale guard | ✅ | `well_lifecycle.scale_guard` — pass condition = **verdict stability**; metric bound demoted to a diagnostic; `LEGAL_RESCALE` + `full_state_coscaled_config` = address **and** payload | `test_scale_guard_fails_on_the_banked_metric_bounded_verdict_flip` + 3 |
| **(iv)** `covered` / `n_never_read` | ✅ | `launch_covered` + **`settle_covered`** emitted; `covered` kept as a documented launch-side alias (monitor `settle_argmin` needs it for Prop D1); `attach_reads` gates on the settle side; captions corrected in 3 places | ⛔ **designed negative** `test_designed_negative_launch_coverage_is_store_invariant_settle_is_not` + 1 |
| **(v)** `d_safe` population | ✅ | `soft_certificate.population_median_nn`; `d_safe_population = "store"` default | `test_d_safe_population_spacing_is_larger_than_the_sizing_set_spacing` + measured before/after |
| **(vi.1)** `own_foreign` kernel | ✅ | **verified on disk** — landed at pass 3, cross-kernel test present and asserts compact-support foreign = **exactly 0** | 4 existing tests |
| **(vi.2)** `theta_att` / `P` | ✅ | `census` emits `P_comparability` (n_non_capturing, degeneracy, comparability) **beside every `P`** — enforced in the emitter | `test_P_is_never_emitted_without_the_theta_att_degeneracy_qualifier` |
| **(vi.3)** ERRATA numbering | ✅ | **verified on disk, recorded CLOSED, NOT re-resolved** — my check agrees with the Hub's account | (verification, no test) |
| **(vi.4)** stale x64 comment | ✅ | **verified on disk** — already corrected at pass 3 rider 4c; commit `42b781c` and `tests/test_cl_baselines_x64.py` both present | (verification, no test) |
| **(vi.5)** unselected width | ✅ | `UnselectedAtomWidth` — the census **refuses, loudly**, unless the effective width fraction is explicitly declared | `test_census_refuses_to_run_at_an_unselected_atom_width` + 1 |
| **(vi.6)** cue difficulty | ✅ | **FULL fix declared**: κ_q normalised on the **codebook** spacing; **both** ratios emitted on every cell regardless | e2e cell assertions |
| **acceptance 3** A3 → DIAGNOSTIC | ✅ | `gate_addr_pass = A1 AND A2`; A3 `label: DIAGNOSTIC`, `in_pass_condition: False`, still measured two-sided; **every leg carries a MECHANICS/DIAGNOSTIC label** | `test_the_launder_margin_is_a_diagnostic_and_cannot_decide_the_verdict` + 1 |

## 2. THE TWO MANDATORY DESIGNED NEGATIVES (both pytest-asserted, both measured)

**(i) — a planted near-zero-drift, table-like store FAILS the drift leg.** Wells planted with their
atoms exactly at their own sites (depth 4.0, width 0.12, radius 0.9, 4 items on a ring):

| quantity | measured |
|---|---|
| median site drift / codebook spacing | **0.0042** |
| registered floor | **0.0100** (`0.01 × codebook spacing`) |
| repaired leg | **FAIL**, `fails_low_D2a_table_expressible = True`, `fails_high = False` |
| ⭐ the **pass-2 one-sided rule** on the same store | **PASS** — the defect, stated as an assertion |

**(iv) — the launch-side statistic is store-invariant; the settle-side one is not.** Same queries,
same codebook; the store is mutated (learned atom amplitudes zeroed) so the reads land elsewhere:

| statistic | before | after |
|---|---|---|
| `launch_covered` | 8/8 | **8/8 — bit-identical** (`np.array_equal` asserted) |
| `settle_covered` | 8/8 | **0/8 — moved** |

That single pair is the proof the two are different quantities, and it is the mechanical explanation
of the "58 / 62 / 62 unassigned, digit-identical" reading that erratum 1 retired.

## 3. PRE-REGISTERED PREDICTIONS vs MEASUREMENT (PREREG.md, filed first)

| # | registered prediction | measured | |
|---|---|---|---|
| P1 | a planted table-like store fails the new leg on the FLOOR while passing the old one-sided rule | ratio 0.0042 < 0.01 ⇒ FAIL(low); `one_sided_pass2_pass = True` | ✅ |
| P2 | the repaired leg changes at least one banked cell's leg verdict | banked arm-A s0 ratio **0.0071 < 0.01** ⇒ would fail (recorded, ⛔ not re-scored) | ✅ |
| P3 | fed the banked `A1 0.24219 → 0.28906` pair, the guard returns `metric_bounded = True` **and** `verdict_stable = False` ⇒ FAIL | exactly that (Δ = 0.04688 ≤ 0.05; A1 threshold 0.25 straddled) | ✅ |
| P4 | store-population spacing **>** sizing-set spacing ⇒ `d_safe` rises | ratio **1.567 / 1.435 / 1.843** | ✅ |
| P5 | refusal rate **≥** the legacy one (direction only; magnitude not predicted) | **0.000/0.250/0.000 → 0.250/0.250/0.143** — rose 2/3, equal 1/3, fell 0/3 | ✅ |

⚠ **The refusal rate is REPORTED, never tuned to a target.** The change removes the *arithmetic
cause* (wrong population), it does not aim at a rate.

## 4. Item (v)+(vi.6) — ONE defect, three sites, one fix

The task's ⭐ observation is confirmed in code: `d_safe`, the G-ADDR cue normalisation and the
retracted §A29.5 mechanism are the **same substitution** — the ~200-key *sizing* spacing standing in
for the ~16-item *store* spacing. Fixing the population choice fixes all three, and it also (item 3
of the reconciliation list) moves the arms' **co-scaled atom width** onto the right population.

**Measured on the toy regression rig (⛔ regression cell, not a science cell; 3 seeds):**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| median-NN, 200-key sizing set | 0.0314 | 0.0462 | 0.0300 |
| median-NN, store population | **0.0492** | **0.0663** | **0.0552** |
| `d_safe` sizing → store | 0.0276 → **0.0433** | 0.0407 → **0.0584** | 0.0264 → **0.0486** |
| refusal rate sizing → store | 0.000 → **0.250** | 0.250 → 0.250 | 0.000 → **0.143** |
| `cue_sigma / codebook_spacing` (repaired) | 1.000 | 1.000 | 1.000 |
| `cue_sigma / sizing_spacing` (legacy) | 1.566 | **6.014** | 2.116 |

The legacy cue normalisation varies **3.8×** across seeds on this rig — the same defect class as the
0.927 / 0.875 / **0.710** spread across the pass-3 arms.

## 5. Flag provenance (protocol §5)

| | |
|---|---|
| commits | `f80c17d`, `dfa7f43`, `b01d474`, `4cdc7e1`, `70b11ae` on `c2w8-close-gate-hardening`, base `main @ 9e0bb25` |
| seeds | 0, 1, 2 (regression cells); 0 (planted rigs, all deterministic, `jitter = 0`) |
| interpreter | `/Users/user/Desktop/CHLU/.venv/bin/python` (**the MAIN venv**, protocol §4 — no worktree `uv sync`, so no JAX version drift) |
| new non-default flags introduced (all additive) | `gdrift_floor_frac_spacing = 0.01`, `gdrift_ceil_frac_spacing = 1.0`, `d_safe_population = "store"` ⭐ *changed default*, `d_safe_sizing_n = 200`, `d_safe_population_draws = 64`, `gaddr_spacing_population = "codebook"` ⭐ *changed default*, `atom_width_selection = None`, `refuse_unselected_atom_width = True`, `atom_width_selection_rtol = 1e-6` |
| regression-cell flags | `addr_dim = 3`, `capacity = 6`, `well_budget = 3`, `n_offer_per_task = 6`, `write_steps = 20`, `read_steps = 60`, `address_steps = 40`, `read_batch = 4`, `capture_dirs = 4`, `bisect_steps = 3`, `n_tasks = 3`, `n_train_per_task = 20`, quick mode, synthetic 10-class Gaussian data |
| planted-rig flags | `addr_dim = 2`, `atoms_per_item = 8`, `min_atoms = 32`, `d_safe_override = 0.001`, `read_steps = 60/800`, `address_steps = 40/400` |
| ⛔ **two shipped defaults changed** | `d_safe_population` and `gaddr_spacing_population` — both legacy behaviours reachable by config; see reconciliation item 3 |

## 6. How I verified — commands and observed output

```
$ /Users/user/Desktop/CHLU/.venv/bin/python -m pytest tests/test_well_lifecycle.py tests/test_gate_addr.py -q --no-cov
50 passed in 38.95s                       # (2 honest failures first — see below)

$ /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q --no-cov      # FULL SUITE @ 4cdc7e1
1579 passed, 36 warnings in 2078.25s (0:34:38)

$ /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q --no-cov      # FULL SUITE @ 70b11ae (HEAD)
1579 passed, 29 warnings in 2068.15s (0:34:28)

$ (fresh detached worktree at 9e0bb25) pytest -q --collect-only --no-cov
1564 tests collected in 7.88s

$ (this branch) pytest -q --collect-only --no-cov
1579 tests collected in 4.43s

$ ruff check <8 touched files>
All checks passed!
```

**Count arithmetic, checkout named:** base `9e0bb25` in a **fresh detached worktree**
(`../CHLU-c2w8base`) collects **1564**; branch `c2w8-close-gate-hardening` in `../CHLU-c2w8close`
collects **1579** ⇒ **+15**, all new tests in the two files I own (11 in `test_well_lifecycle.py`,
6 in `test_gate_addr.py`, minus 2 rewritten in place). ⚠ **The task file's 1555 does not reproduce**
— reported as measured (reconciliation item 5). Counts are comparable only within one checkout; both
numbers were taken with the same interpreter on the same machine minutes apart.

**Two honest first-pass failures, both of my test rigs (not of the legs), both fixed by measurement
rather than by weakening an assertion:**
1. the first planted "table-like" store measured drift/spacing = **0.0512**, above the floor — a
   *shallow* planted well still relaxes toward the confinement bowl, so the residual was the census
   relaxation's own floor, not the store's. Fixed by planting deep/narrow (4.0 / 0.12) at radius 0.9
   ⇒ 0.0042. Scan recorded: n=6 r=0.6 gives 0.0254; **n=4 r=0.9 gives 0.0042**; depth ≥ 8 at width
   0.08 *destabilises* the relaxation (ratio 0.85, then 7.14) — worth knowing, the census's
   relaxation is not unconditionally stable at very sharp wells.
2. the first coverage negative left `settle_covered` unchanged at the default 60/40 step budget: the
   flattened landscape had **not yet pulled the read off its launch point**. It would have passed for
   the wrong reason at 300/200 too; at **800/400** it moves 8/8 → 0/8. The step budget is now
   explicit in the test with the reason.

## 7. What I did NOT do (declared NOT-RUNs, never nulls)

- ⛔ **No banked arm was re-run** (arm A, arm B, the spine's 9 cells). Every consequence of a repair
  on a banked number is stated as a *mechanical consequence*, never re-scored.
- ⛔ **No file in the forbidden set was touched**: `scripts/csf3/`, `train_cluformer.py`, `blocks.py`,
  `exp_cluformer_pilot.py`, `emission_head.py`, `exp_capture_strong_phi.py`. I also did **not** edit
  `exp_capture_armA.py` / `exp_capture_armB.py` (banked arms, read-only).
- ⛔ No merge / prune / restoration verb. ⛔ No pass 4, no daylight chase, no arm-race adjudication,
  no tier-ii / full-CLU verdict, no paper number.
- ⚠ **Two files outside the declared ownership list were edited, minimally and by necessity**:
  `chlu/experiments/usage_telemetry.py` (**3 hunks** — `attach_reads`'s coverage key + 2 caption
  fields; item (iv) is unimplementable without it, and the file is not in the ⛔ list) and one helper
  signature in `tests/test_well_lifecycle.py` (`_tiny_system` now `setdefault`s its two step budgets
  so an override is possible; behaviour for every existing caller is unchanged).

## 8. Git footprint

**Branch** `c2w8-close-gate-hardening` (worktree `../CHLU-c2w8close`), base `main @ 9e0bb25`.
**⛔ Not pushed, not merged — left for Hub review.** Rebase onto local `main`: no-op (base unmoved).

| commit | subject |
|---|---|
| `f80c17d` | harden the census gate: two-sided drift, A1 margins, verdict-stable scale guard, A3 → DIAGNOSTIC |
| `dfa7f43` | split launch-point coverage from settle-side coverage |
| `b01d474` | size d_safe and the G-ADDR cue on the STORE population, and refuse an unselected width |
| `4cdc7e1` | designed negatives for the hardened gate, and the rewritten N1 |
| `70b11ae` | declare the arm-facing consequence of the d_safe population repair (comment only) |

**Files touched (8):** `chlu/config.py` (additive) · `chlu/core/well_lifecycle.py` ·
`chlu/core/clu_system.py` (`_read_diagnostics` only) · `chlu/core/soft_certificate.py` (one new
function) · `chlu/experiments/exp_well_lifecycle.py` · `chlu/experiments/usage_telemetry.py` ·
`tests/test_well_lifecycle.py` · `tests/test_gate_addr.py`.
**Not touched:** `tests/test_cifar_strong_phi.py` (item vi.4 was already correct — verified, not
edited) and `ERRATA-C2W8-PASS2.md` (item vi.3 verify-only).
**No unresolved conflicts.** ⚠ The **shared checkout was on another spoke's branch**
(`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`) throughout — I worked only in my worktree
and never wrote to it. A second **detached** worktree `../CHLU-c2w8base` at `9e0bb25` was created for
the base count and is removed at the end of this report.

## 9. Open questions / follow-ups / risks

1. ⚠ **The drift floor's value is a declared choice, not a measured law.** 0.01 × codebook spacing is
   derived from *one* banked D2a datum (±0.0007 vs spacing ~0.14). It is config-overridable
   (`gdrift_floor_frac_spacing`). **Risk:** on rigs whose census relaxation has a numerical drift
   floor *above* 0.01 × spacing (the toy rig here runs 2–15 × spacing), the low side can never fire
   and the leg is effectively one-sided again. The instrument's own relaxation floor should be
   measured per rig before the floor is quoted as a bound.
2. ⚠ **The scale guard is a pure verdict-comparison function; it does not itself run the rescaled
   cell.** Running the legal rescale end-to-end (via `full_state_coscaled_config`) is a two-cell cost
   I did not spend — it is a **declared NOT-RUN**, not a null. The guard's repair is proven against
   the exact banked pair that defeated the old guard.
3. **Should `covered` (the deprecated launch-side alias) be removed?** I kept it because monitor
   `settle_argmin` needs the launch-side `U` for Prop D1 (`D ≤ U`). If the Hub wants the name gone,
   the monitor must be switched to `launch_covered` in the same commit.
4. **The width refusal accepts any fraction declared *anywhere* in the arm configs.** That catches
   "nobody declared this width" but not "the spine declared 1.5 while running arm A's 0.5 default".
   Pinning `experiment_well_lifecycle.atom_width_selection` per experiment closes it completely; I
   left the default `None` so no existing runner breaks.

---

## Proposed handover updates (for the Hub)

**§3 CLI/Config — two shipped defaults CHANGED (both legacy paths reachable):**
- `experiment_well_lifecycle.d_safe_population = "store"` (was, implicitly, `"sizing"`) — `d_safe`
  and every arm's co-scaled atom width are now sized on the **store population's** NN spacing.
  ⚠ **Banked arm-A / spine cells need `"sizing"` to reproduce bit-for-bit.**
- `experiment_well_lifecycle.gaddr_spacing_population = "codebook"` (was, implicitly, `"sizing"`) —
  the G-ADDR cue jitter is normalised on the codebook spacing.
- New additive knobs: `gdrift_floor_frac_spacing = 0.01`, `gdrift_ceil_frac_spacing = 1.0`,
  `d_safe_sizing_n = 200`, `d_safe_population_draws = 64`, `atom_width_selection = None`,
  `refuse_unselected_atom_width = True`, `atom_width_selection_rtol = 1e-6`.

**§7 Known Issues — these can be CLOSED (with the caveat named):**
- *`covered` / `n_never_read` are launch-point statistics* → **closed**; `settle_covered` shipped,
  telemetry re-gated, captions corrected. ⚠ Add: **pre-`dfa7f43` `n_never_read` values are
  launch-gated and are not comparable to post-`dfa7f43` ones.**
- *`d_safe` sized on the wrong population / monitor #3's 0.000 refusal is arithmetic* → **closed at
  the cause**; refusal rate measured 0.083 → 0.214 (mean, toy rig), reported not tuned.
- *`own_foreign_site_depth` hard-codes the Gaussian kernel* → **closed** (landed pass 3, verified
  here with its cross-kernel test).
- *`theta_att` / `P` not comparable across arms* → **closed**; enforced in the emitter.
- *`ERRATA-C2W8-PASS2` §2 numbering collision* → **closed**, verified on disk, not re-resolved.
- *stale x64 comment in `tests/test_cifar_strong_phi.py`* → **closed**, verified on disk.
- *the pass-2 gate rewards drift → 0* → **closed**; G-DRIFT is two-sided with a designed negative.
- *the §4 scale guard bounds the metric, not the verdict* → **closed**; `scale_guard` asserts the
  verdict, and the legal rescale is full-state co-scaling.

**NEW §7 entries the Hub should add:**
- ⚠ **A3 is out of `gate_addr_pass` (§A33.1)** ⇒ **every banked `gate_addr_pass` value predates the
  rule and must not be re-quoted without restating which legs decided it.** Arm B's banked
  configuration passes the two MECHANICS legs under the repaired instrument (reconciliation item 1).
- ⚠ **The banked pass-3 arm-A seed-0 cell (G-DRIFT ratio 0.0071) fails the repaired drift leg on the
  D2a side.** Expected and intended; it means the C2W11 census will fail cells pass 3 passed.
- ⚠ **The census relaxation is not unconditionally stable at very sharp planted wells** (depth 8 /
  width 0.08 measured drift/spacing 0.85, depth 20 / width 0.06 measured 7.14). Relevant to anyone
  planting deep narrow wells for a control.

**§10 running log:** `gate_hardening_done = true` (12/12 items, both mandatory designed negatives
pytest-asserted, suite 1579/0 at `70b11ae`) ⇒ **the §A32.3 mechanical gate C2W11 waits on is
SATISFIED**, subject to the six reconciliation items above having an owner.
