# c2w5-close-fixes — experiment-engineer report

**Task + acceptance criterion:** the five-item C2W5 wave-close fix pack — re-render Fig 1/2/5 to the
r4 App-K target, bank the SSD arm's two declared missing aggregations, promote the CLU columns into
`audit_table` (+ regression test), re-measure `bprime-c6`'s `s` with the confinement-subtracted
estimator, and add the `exp-psi-residual` CLI hook. **Status: done (5 of 5).**

**Dial declaration (echoed).** The task file carried no DIAL DECLARATION block (it predates the
§7 template in this wave's pack); the honest reading is **none: instrument / reconciliation** for all
five items. Laundering control: not applicable — no performance number is produced here; item 4's
control is the *no-subtraction* arm on the identical store, item 2's is the re-aggregation of the five
incumbent arms whose published values must come back digit-for-digit. Falsifies: item 4's ruler claim
dies if the corrected `s` lands outside the pre-registered band (PREREG §2); item 3's dies if the
harness's emitted columns differ from the published `n = 9` numbers.

⭐ **Downstream reconciliation list (owner needed — see §7):** three items. (a) the r4 App-K sentence
"the delta arms carry no learned-initial-state component" is **false against the artifacts**
(`S0_init = 5184 B` on `deltanet`/`gdn`/`gdn2`/`mamba2`); (b) App K's figure-provenance pointer must
move to the new 64-entry table; (c) §7.28/N224's flag on `bprime-c6` is now **discharged** and the
`s = 0.40` datum is CONFIRMED (0.4188 ± 0.0060 on the corrected estimator).

---

## 1. What I did

| # | item | outcome |
|---|---|---|
| 1 | ⭐ Fig 1/2/5 re-render to the r4 App-K target | done — 7-bar Fig 1 with the three hatch classes **derived from artifacts**, Fig 2 with the SSD bar + modal caps, Fig 5 caption **0 of 20** (counted, not transcribed); PNG + PDF at the existing paths; new 64-entry provenance table |
| 2 | the two declared missing SSD aggregations | done — `full − null` **+0.3702 ± 0.0946** paired (n = 9), per-reader `+0 B` means, and **all five** frontier `+0 B` margins; JSONs banked beside `run_agg_n9` |
| 3 | CLU columns into `audit_table` (+ test) | done — `_clu_columns`, 3 new tests, published `n = 9` column reproduced exactly |
| 4 | ⭐ the `s`-ruler re-measurement of `bprime-c6` | done — **`s = 0.40` CONFIRMED**; corrected `s = 0.4188 ± 0.0060`, `R² = 0.9947`; the §7.28 mechanism is real on this rig (1.36× inflation) but c6's own estimator already subtracted |
| 5 | the `exp-psi-residual` CLI hook | done — `+1` subcommand on the `exp-tierii-read` pattern, no existing line altered |

---

## 2. Item 1 — the Fig 1/2/5 re-render (r4 reconciliation 1)

**Renderer:** `.claude/scratch/c2w5-close-fixes/render_figures.py` — a **fork** of
`.claude/scratch/bprime-referee-closures/render_figures.py`; that spoke's file is left byte-untouched.
**Figures:** `.claude/papers/bprime/figures/fig{1..5}_*.{png,pdf}` (same paths; Fig 3/4 re-rendered
unchanged from the same inputs so the provenance table stays a complete 1:1 record).
**Provenance:** `.claude/outputs/c2w5-close-fixes/figure_provenance.json` — **64 entries** (was 48),
with a `supersedes` pointer; ⛔ the closures spoke's own table is left in place, unedited.

**Fig 1 = seven bars, uniform `n = 9`.** Rival values now come from the SSD run
(`.claude/outputs/bprime-mamba2-arm/run_agg_n9/exp_bprime_rivals_metrics.json`,
`audit_table_by_selection.f3`), which is a single uniform source for all six arms; the CLU bar from
`n9_clu_column.json`. ⭐ **App A.1f is asserted in the renderer, not quoted**: the five incumbents'
`raw_table_margin(_se)` are **float-identical** between the SSD run and
`pilot-placement-probe/n9_full_columns.json` (5/5 True, logged in the provenance table).

⭐ **The hatch classes are derived, not hardcoded** (`_verdict`, rule in the docstring): rescued under
all three registered selections ⇒ RESCUED; the two fit-split rules only ⇒ SELECTION-DEPENDENT; no
selection here but rescued under the first pass's init scheme ⇒ INIT-UNSTABLE; else NOT RESCUED. The
derivation lands **exactly** on the r4 spec's membership:

| arm | f3 | f3_lite | f3_val | other init scheme | class | bar value |
|---|---|---|---|---|---|---|
| ttt_linear | ✗ | ✗ | ✗ | ✅ | **INIT-UNSTABLE** | −0.4602 ± 0.1038 |
| ttt_mlp | ✗ | ✗ | ✗ | ✗ | **NOT RESCUED** | −0.4425 ± 0.0869 |
| deltanet | ✅ | ✅ | ✗ | ✅ | **SELECTION-DEPENDENT** | −0.2732 ± 0.0395 |
| gdn | ✅ | ✅ | ✅ | ✅ | RESCUED (unhatched) | −0.2600 ± 0.0278 |
| gdn2 | ✅ | ✅ | ✗ | ✅ | **SELECTION-DEPENDENT** | −0.2592 ± 0.0292 |
| **mamba2** | ✅ | ✅ | ✅ | — (not run there) | RESCUED (unhatched) | **−0.2563 ± 0.0416** |
| **CLU** | ✗ | — | — | ✗ | **NOT RESCUED** | **−0.2897 ± 0.0328** |

Every bar carries `n = 9`; the title states family, byte budget, `d_in`, item count, stream length,
CPU, the selection rule and the uniform seed count. No mixed-`n` language survives anywhere.

**Fig 2.** Ledger bars now come from the SSD run's own per-seed cells (all six arms + the CLU), with
the incumbents' state bytes cross-checked against the F3 artifacts per seed (True, provenanced). The
SSD bar renders at **F1 8380 B / F2 5184 B / own table 5184 B**, ratio 1.00 — exactly the spec.
Modal-over-nine-seeds with the alternates drawn as caps is unchanged (TTT arms + CLU are the rows
that move; delta/SSD are seed-constant at 5184 B).

⚠ **Spec/artifact discrepancy, rendered to spec and flagged (reconciliation item (a)).** App K says
*"the delta arms carry no learned-initial-state component in their parameter block and are drawn as
all-parameter"*. The artifacts say otherwise: `deltanet`, `gdn`, `gdn2` **and** `mamba2` each declare
`S0_init = 1296 floats = 5184 B` inside `byte_ledger.rival.param_breakdown` (and
`tests/test_bprime_rivals.py::test_learned_initial_state_rule_puts_the_init_in_PARAMETERS` asserts
every arm has one). The render keeps the existing `W0_init`-only hatch rule so the figure matches the
r4 caption, records each arm's `S0_init` bytes in the provenance table, and emits a `declared_notes`
entry. **Either the sentence or the hatch rule has to change — that is a writer's ruling, not a
renderer's.**

**Fig 5.** Caption count is now **computed**: the incumbent frontier rows
(`pilot-placement-probe/n9_deltanet_frontier.json`, 15 cells) plus the SSD arm's 5 (item 2's new
artifact) give **20 cells, 0 clearing their own blank store** ⇒ the caption reads *"at n = 9, 0 of 20
(arm × head-width) cells on this column clear their own blank store"*. The per-cell booleans are in
the provenance table. Data (banked CLU curve, `n = 3`, two write loads plotted as two series) is
untouched.

## 3. Item 2 — the two declared missing aggregations (SSD arm)

Script `.claude/scratch/c2w5-close-fixes/ssd_missing_aggregations.py`; artifacts written **beside**
the run: `.claude/outputs/bprime-mamba2-arm/run_agg_n9/ssd_paired_full_minus_null.json` and
`…/ssd_zero_byte_readers.json`. ⛔ Nothing measured — re-aggregation of banked per-seed cells, with
the same conventions as the artifacts they sit beside (sample sd ddof=1, SE = sd/√n, **paired per
seed**). ⛔ No draft edited.

**(a) The paired `full − null`, `aggregate@base`, n = 9.** ⭐ The control first: all five incumbent
arms come back **digit-for-digit** on the published column (−0.2063 ± 0.1016 · −0.1995 ± 0.0665 ·
+0.2174 ± 0.0749 · +0.5642 ± 0.1032 · +0.7438 ± 0.1242 — 5/5 exact), which is what licenses the new
row to sit in the same table. Then:

| selection | SSD `full − null` (paired) | \|t\| | sign |
|---|---|---|---|
| **`f3` (primary)** | **+0.3702 ± 0.0946** | **3.91** | positive on **8 of 9** seeds, median +0.3741 |
| `f3_lite_control` | +0.3905 ± 0.0960 | 4.07 | — |
| `f3_val` (held-out) | +0.1808 ± 0.0771 | 2.34 | — |

⇒ the draft's hedged form (*"on unpaired means it sits on the delta-rule side… we quote no SE or
significance for a difference we did not pair"*) can be replaced by a paired number: **the SSD arm
reads better than a store handed its own keys with the wrong payloads, by 3.9 SE**, joining the three
delta-rule arms and splitting from both TTT arms. The unpaired means the draft quotes (−0.4036 vs
−0.7739) reproduce exactly.

**(b) Per-reader `+0 B` means (the dashed row of App I.1c(b)), `f3`, n = 9:**
`knn2_mean_+0B` **−0.6793 ± 0.1167** · `knn2_idw_+0B` **−0.6758 ± 0.1173** ·
`table_mean_+0B` **−0.4221 ± 0.0481`. Re-deriving the margin from these readers reproduces the
shipped `+0 B` margin **per seed, exactly** (`margins_agree = True` on all three selections), so the
dashes can be filled without disturbing the margin column (+0.0047 ± 0.0519 primary; −0.0045 held-out).

**Per-head-width frontier `+0 B` margins (App H.1b's declared NOT-RUN), n = 9, `best_of_3` rule
identical to the incumbent frontier rows:**

| cell | state B | full | **+0 B margin** | lift over own blank | RESCUED | beats +0 B |
|---|---|---|---|---|---|---|
| mamba2@d2 | 16 | 0.1528 ± 0.0354 | **−0.0139 ± 0.0354** | +0.0417 ± 0.0333 | ⛔ | ⛔ |
| mamba2@d4 | 64 | 0.1620 ± 0.0383 | **−0.0231 ± 0.0348** | −0.0185 ± 0.0440 | ⛔ | ⛔ |
| mamba2@d8 | 256 | 0.1296 ± 0.0306 | **−0.0556 ± 0.0269** | +0.0046 ± 0.0245 | ⛔ | ⛔ |
| mamba2@d16 | 1024 | 0.1759 ± 0.0359 | **−0.0417 ± 0.0354** | +0.0417 ± 0.0367 | ⛔ | ⛔ |
| mamba2@d36 | 5184 | 0.2037 ± 0.0306 | **−0.0324 ± 0.0404** | +0.0370 ± 0.0364 | ⛔ | ⛔ |

⇒ **0 of 5 beat their own `+0 B` reader set, 0 of 5 clear their own blank store** — the arm's five
frontier cells are a labelled null in both columns, which is what makes the 0-of-20 count in Fig 5's
caption an *aggregated* fact rather than an inference.

## 4. Item 3 — the CLU columns, first-class in the harness

`chlu/experiments/exp_bprime_rivals.py`: new `_clu_columns(cells)` feeding
`audit_table(...)["<family>"]["clu_reproduced"]`. Rules are byte-identical to the scratch script the
paper discloses in App A.1e (`n9_clu_column.py`): sample sd (ddof=1), SE = sd/√n, **every
margin/lift paired per seed**, the `+0 B` reader arg-maxed **per seed** over the exclusive set, the
raw-table reader over that set ∪ {`settle_deleted`}. Legacy keys (`full`, `full_se`, `launder`,
`launder_se`, `dividend`) are preserved unchanged; everything else is additive.

Emitted on the same banked inputs (`bprime-rivals-f3/{run400,seeds3to8}`, 9 aggregate cells):

```
full -0.437047 ± 0.041739   launder -0.380978 ± 0.034476   dividend -0.056070 ± 0.031549
blank -0.390587 ± 0.012374  same_keys_null -0.651176 ± 0.038282
full - null +0.214129 ± 0.044291
+0 B margin -0.289729 ± 0.032793   raw-table margin -0.289729 ± 0.032793 (identical, 9/9 seeds)
lift -0.046460 ± 0.040631   RESCUED_above_own_blank_2se = False
```

— i.e. the published `−0.3906 / −0.6512 / −0.2897 / −0.0465` column, reproduced by the harness.

**Tests** (`tests/test_bprime_rivals.py`, +3): the published `n = 9` values (mean and SE, 4 dp) with
the NOT-RESCUED verdict; the `+0 B` ≡ raw-margin identity with a by-hand recomputation of the paired
arg-max rule; and a degradation test — a record predating `all_launder_scores` contributes `nan` to
the reader columns instead of crashing or dropping the cell.

## 5. Item 4 — ⭐ the `s`-ruler re-measurement (curator G-5 / N224 / §7.28)

**Pre-registered before running** — `.claude/outputs/c2w5-close-fixes/PREREG.md` (H-A confirm band
[0.33, 0.45] derived from the shipped code; H-B the §7.28 alternative `≈ 0.252` from the cat-test's
1.44× inflation; H-C the mechanism check `≥ 1.10`; decision rule fixed in §2).
Harness `.claude/scratch/c2w5-close-fixes/s_ruler_c6.py` (+ `s_ruler_hint.py`), artifact
`.claude/outputs/c2w5-close-fixes/s_ruler_c6.json`.

**Rig:** c6's shipped cell rebuilt bit-for-bit via `_write_and_query("overload","load1x_shipped",seed,
clu_extra=_sweep_overrides(1.0))` — write path only, no reads, no deletions. 6 live items per seed,
`λ_min = 3.01…3.41`, endpoint write loss 0.0019–0.0032, admissible on 3/3 seeds.

### ⭐ VERDICT: **`s = 0.40` CONFIRMED** (and §7.28's flag on `bprime-c6` is discharged)

| estimator (same store, same items) | `s` (mean ± SE over 3 seeds) | `R²` | implied `d/s` (`d = 1.3006`) |
|---|---|---|---|
| **corrected `effective_s`, `confine = α = 0.05`** | **0.4188 ± 0.0060** | **0.9947** | **3.11 ± 0.04** |
| control: identical call, `confine = 0` | 0.5677 ± 0.0014 | — | 2.29 |
| `CluSystem.well_fits()` (the route c6 published) | 0.3625 ± 0.0032 | — | 3.59 ± 0.03 |
| c6's published law fit (`κ` vs `d²`) | 0.3979 (`R²` 0.9953) | — | 3.27 |

- **Every finite fit converged:** 6/6 items on 3/3 seeds, per-item `s` spread 0.350–0.460.
- `well_fits()` returns **0.3625**, i.e. c6's own published shipped-cell row (`s_fit = 0.3625`)
  **digit-for-digit** — the rig was rebuilt correctly.
- **H-C holds and the §7.28 mechanism is real on this rig too:** dropping the subtraction inflates the
  fitted width by **1.356 ± 0.020×** (cat-test measured 1.44× on its own store). Robustness over the
  estimator's radius ladder (`s_hint` = 0.3 / 0.4 / 0.5 ⇒ `r_max` = 1.2 / 1.6 / 2.0):

  | `s_hint` | corrected `s` | `R²` | uncorrected `s` | inflation |
  |---|---|---|---|---|
  | 0.3 | 0.3873 ± 0.0015 | 0.9991 | 0.4231 | 1.09× |
  | 0.4 | 0.4188 ± 0.0060 | 0.9947 | 0.5677 | 1.36× |
  | 0.5 | 0.4402 ± 0.0150 | 0.9808 | 0.7578 | **1.72×** |

  ⭐ **The corrected width is stable to ±7 % across the ladder while the uncorrected one runs away
  (0.42 → 0.76).** That *is* the §7.28 argument, measured: the bowl's share of the profile grows with
  the sampled radius, so an unsubtracted fit has no fixed point. Any future `s` quoted without the
  subtraction must also state its `r_max`.

**Why CONFIRMED — the mechanism, from the shipped code (pre-registered as H-A).** Both of c6's routes
are confinement-clean by construction: (i) `CluSystem._well_fit` subtracts `α(‖z+ru‖² − ‖z‖²)`
analytically (`clu_system.py:1200`, present in that function's first commit `4cd1a9a` — verified with
`git log -L`), and (ii) the law route fits `κ = ‖∇(V_full − V_{−k})‖ / ‖∇(V_full − V_{−sel})‖`, in
which the identical `α‖q‖²` term cancels **exactly** inside each gradient difference before the ratio.
`DesignFreedomPotential` is `v_learned(q) + confine·‖q‖²` exactly
(`chlu/core/memory_potentials.py:639`), so the subtraction is analytic. **c6's `s = 0.40` was never
contaminated; the §7.28 flag on that rig can be closed.**

⚠ **One honest deviation from the direction on record.** The task states *"subtraction makes `s`
smaller and `d/s` LARGER"*. That is true of an estimator that had **not** subtracted (control 0.5677 →
corrected 0.4188 ⇒ `d/s` 2.29 → 3.11). It is **not** what happens to c6's published number, because
c6 already subtracted: the corrected log-linear estimator returns a width **+15.5 % larger** than
`well_fits`'s non-linear fit (0.4188 vs 0.3625), so `d/s` moves **3.59 → 3.11**, i.e. slightly *out*
of the designed-gate regime, not further in. The gap between the two is an **estimator difference**
(log-linear fit of the profile relative to the outermost radius, vs `D(1 − e^{−r²/2s²})` least squares
on a fixed 0.15–1.5 ladder) — pre-declared as such in PREREG §3, and bounded by the `s_hint` sweep
above. Both numbers sit inside the pre-registered CONFIRM band, and the law route (0.3979) sits
between them. ⇒ **the honest quotable form is `s = 0.40 ± 0.03` (three estimators, 0.36–0.42) and
`d/s = 3.1–3.6` at the audited cell**, not a single 4-digit width.

**Flag provenance (item 4).**

| field | value |
|---|---|
| commit | branch `agent/experiment-engineer/c2w5-close-fixes` @ `fafa1fc` (base local `main` @ `483c4ba`) |
| seeds | 0, 1, 2 (c6's own count at this radius); write RNG `PRNGKey(seed)`, stream key `PRNGKey(seed+1)` |
| rig | `overload@load1x_shipped`, **`ball_radius = 1.0`**, **`d_safe_override = 0.58`** (non-default), write path only |
| store | `dim = 5` (`addr 4` + `payload 1`), `capacity = 6`, `n_atoms = 2046`, `atoms_per_item = 341`, `atom_width = 0.3`, `atom_depth_init = 1e-4`, `atom_init_scale = 1.0`, **`confine = α = 0.05`** |
| dynamics | `write_steps = 300`, `address_steps = 400`, `read_steps = 800`, `dt = 0.05`, `γ_addr = 0.05`, `γ_read = 0.02`, `kinetic_mode = newtonian_learned` |
| estimator | `effective_s`: 16 rays × 24 radii, `r_max = 4·s_hint`, `s_hint ∈ {0.3, **0.4**, 0.5}`, profile relative to the outermost radius, `keep > 2 % of depth` |
| `d` | **1.3006** — c6's own `rows[R=1.0].d` (mean 2nd-nearest key distance), **quoted, not recomputed** |
| JAX | 0.9.0 (main venv, `uv run --no-sync`) |
| ⛔ | no registry / N224 / charter / draft edited |

## 6. Item 5 — the `exp-psi-residual` CLI hook

`chlu/cli/experiment_cmd.py` gains the subcommand on the `exp-tierii-read` pattern
(`--tier {ledger,trained}`, `--cells`, `--seeds`, `--steps`, `--eval-batches`, `--out-dir`, `--tag`),
forwarding the module's own argv contract to `exp_psi_residual.main` so validation stays in one place.
No existing line altered. Test asserts both parses and that a bad cell is rejected by `main`'s own
validator (`SystemExit: unknown cells ['not_a_cell']`) — i.e. the argv really reaches the module.

---

## 7. How I verified (commands + observed output)

| check | command | observed |
|---|---|---|
| CLI hook parses + forwards | `python -c "…setup_experiment_parsers…; a.func(a)"` | `Namespace(tier='trained', cells=['run1'], seeds=[0,1], steps=5, …)`; `SystemExit -> unknown cells ['not_a_cell']; known: [...]` |
| CLU columns vs published | `python -c "audit_table(f3 cells)['aggregate']['clu_reproduced']"` | `blank −0.390587 · null −0.651176 · +0B −0.289729 · lift −0.046460 · RESCUED False` (= published 4-dp values) |
| SSD aggregations + control | `python .claude/scratch/c2w5-close-fixes/ssd_missing_aggregations.py` | 5/5 incumbents reproduce the published `full − null`; SSD `+0.3702 ± 0.0946`; frontier 0/5 |
| figures | `python .claude/scratch/c2w5-close-fixes/render_figures.py` | 10 files written; provenance 64 entries; 4 `declared_notes` (the `S0_init` discrepancy) |
| `s`-ruler | `python .claude/scratch/c2w5-close-fixes/s_ruler_c6.py` (+`_hint.py`) | `corrected 0.4188 ± 0.0060 (R² 0.9947)`, `no-subtraction 0.5677 (×1.356)`, `well_fits 0.3625` ⇒ **CONFIRMED** |
| targeted tests | `pytest tests/test_bprime_rivals.py tests/test_rivals_ledger.py tests/test_fb4_gate.py -q` | **107 passed** |
| psi tests | `pytest tests/test_psi_residual.py -q` | **16 passed** |
| full suite | `pytest -q -p no:randomly` (log: `.claude/scratch/c2w5-close-fixes/pytest_full.log`) | **1348 passed, 0 failed, 24 warnings, 30:17** at `fafa1fc` |
| lint | `ruff check chlu/ tests/` (touched files) | All checks passed |

## 8. Git footprint

- **Branch** `agent/experiment-engineer/c2w5-close-fixes`, off local `main` @ `483c4ba`. Not pushed,
  not merged. No worktree (main checkout was clean at start and stayed mine). Base unmoved ⇒ no
  rebase needed; `git status` clean at hand-back (the `coverage.xml` pytest byproduct removed).
- **Suite green at the tip:** **1348 passed, 0 failed** (30 min, `-p no:randomly`). ⚠ App M records
  "1143 passed" at its own commit — the count has grown with the wave's merges; nothing red.
- `1e89dc2` `[experiment-engineer] the exp-psi-residual CLI hook (psi-payload-residual §8)` —
  `chlu/cli/experiment_cmd.py`, `tests/test_psi_residual.py`
- `fafa1fc` `[experiment-engineer] audit_table emits the CLU's own columns first-class (r4 editorial 4)`
  — `chlu/experiments/exp_bprime_rivals.py`, `tests/test_bprime_rivals.py`
- Everything else is under gitignored `.claude/` (scripts in `.claude/scratch/c2w5-close-fixes/`,
  artifacts in `.claude/outputs/c2w5-close-fixes/` and beside `run_agg_n9`, figures in
  `.claude/papers/bprime/figures/`). No conflicts; no rebase needed (base unmoved).

## 9. Open questions / follow-ups / risks

1. ⚠ **`S0_init` vs App K (needs a writer ruling).** Rendered to spec, flagged in the provenance
   table. If the rule changes, the delta/SSD bars gain a 5184 B hatched init block and Fig 2's
   "all-parameter" sentence goes.
2. **Figure provenance pointer.** App K currently points at the closures spoke's 48-entry table; the
   shipped figures now trace to `.claude/outputs/c2w5-close-fixes/figure_provenance.json` (64
   entries). I did **not** edit any draft (paper-writer is live in `.claude/papers/`).
3. **The SSD paired `full − null` unlocks paper text** that r4 deliberately hedged (App J NOT-RUN,
   §4.1.1 "not paired"): the r5 writer can now state the six-arm split — both TTT arms below their own
   null, all four delta/SSD arms above it.
4. **`d/s` bookkeeping.** Item 4 says the *ruler* is sound but that the program's `d/s` numbers are
   estimator-dependent at the ±15 % level. Anyone quoting a 3-significant-figure `d/s` should quote the
   estimator with it (`well_fits` 3.59 · law 3.27 · corrected `effective_s` 3.11 at the audited cell).
5. **Not done, declared:** no draft/registry/N224 edits; no new *measurement* of any rival or CLU cell;
   the `s` re-measurement covers the **shipped** radius only (`R = 1.0`, 3 seeds), not the other five
   radii of the c6 sweep — re-running those would let the *law* be re-fitted on the corrected ruler
   (~10 min of compute, worth one follow-up task if the Hub wants the exponent re-derived).

## Proposed handover updates (for the Hub)

- **§7.28 → RESOLVED for `bprime-c6` (keep the standing rule).** New text: *"the effective-`s`
  estimator must subtract `α‖q‖²`; the inflation is real and grows with the radius ladder (measured on
  the c6 rig: 1.09× / 1.36× / 1.72× at `r_max` = 1.2 / 1.6 / 2.0; 1.44× on the cat-test store).
  **`bprime-c6` is now re-measured and CLEARED** (`c2w5-close-fixes` §5): both of its routes subtract
  by construction (`CluSystem._well_fit:1200`; the ∇V-ratio cancels `α` exactly), corrected
  `s = 0.4188 ± 0.0060`, `R² = 0.9947`, `well_fits` reproduces 0.3625 digit-for-digit. Quote
  `s = 0.40 ± 0.03` and `d/s = 3.1–3.6` **with the estimator named** — the three rulers differ by up
  to 15 %."* ⇒ N224 / charter §A20.5 / curator G-5 can be closed by the curator's next pass.
- **§7 new (tests/harness):** `audit_table` now emits the CLU's own audit columns; the App A.1e
  scratch-aggregation disclosure is obsolete for anything re-run after `fafa1fc`.
- **§3 (CLI):** new subcommand `chlu exp-psi-residual [--tier {ledger,trained}] [--cells …]
  [--seeds …] [--steps I] [--eval-batches I] [--out-dir P] [--tag S]`.
- **Never-quote candidate (new):** *"the SSD arm's `full − null` is unpaired / unquotable"* — it is
  **+0.3702 ± 0.0946** (3.9 SE, 8/9 seeds) as of this pass. And *"the byte-frontier column's SSD
  margins were not aggregated"* — all five are banked and negative.
- **Reconciliation owner needed** for the `S0_init` / App K sentence (item 1 above) — that is a
  paper-writer or curator task, not an engineering one.
