# w12r spoke/artefact-mustfix — FINDINGS (report to the Hub)

**Agent:** `experiment-engineer`. **Branch:** `w12r/artefact-mustfix`.
**Scored runs spent: ⛔ ZERO. Compute spent: ⛔ ZERO** (no GPU, no scored leg, no background job; the
only execution was `python build_tables.py` and `python verify_tables.py`, both of which re-read
artefacts and re-plot).
**Window:** `START_UTC.txt` 2026-09-21T12:39:00Z → `END_UTC.txt`.
⛔ **`$CHLU_CLAUDE_ROOT` was never opened.** ⛔ **`/c/chlu/evidence/` was never written.**
⛔ **No `w12-*` session, worktree, output dir or process was touched, and no box-wide kill was run.**
⭐ **Every number below names the file it came from.**

---

## 0. WHAT SHIPPED, AND THE ACCEPTANCE CRITERION

Everything is in `/c/chlu/outputs/w12r-artefact-mustfix/`.

| file | what it is |
|---|---|
| `tables-r1/` | the w10 seeds-and-tables artefact, **revised**. A copy — the w10 dir was not written |
| `tables-r1/build_tables.py` | the **generator**, edited. 79,116 B → 170,662 B |
| `tables-r1/TABLES.json` | regenerated. 263,860 B → 383,709 B |
| `tables-r1/TABLES.md` | regenerated. 40,881 B → 89,016 B |
| `tables-r1/verify_tables.py` | ⛔ **UNCHANGED** from w10 (8,561 B, same bytes) — it is the check, so it was not touched |
| `tables-r1/figures/` | 5 figures, regenerated; `fig1` renamed `fig1_cmaze2_main_scoreboard.png` |
| `patch/p1..p8*.py` | the eight exact-string patch scripts, each asserting its anchor occurs exactly once |
| `LOG.md`, `FINDINGS.md`, `START_UTC.txt`, `END_UTC.txt` | the record |

⭐ **I edited the GENERATOR, never the outputs by hand.** That is what makes `verify_tables.py` a
real check rather than a formality, and it keeps the manifest regenerable from the source paths.

### ⭐ THE ACCEPTANCE CRITERION — `verify_tables.py` still PASSES

| | BEFORE (`/c/chlu/outputs/w10-seeds-and-tables`, pristine) | AFTER (`tables-r1`) |
|---|---|---|
| **verdict** | **PASS** | ⭐ **PASS** |
| sources registered | 27 | **32** |
| source hashes re-verified | 27 | **32** |
| cells found in `TABLES.json` | 240 | **286** |
| per-seed values re-resolved | 720 (3 via a declared reduction) | **854** (3 via a declared reduction) |
| figures (hashes re-verified) | 5 (5) | 5 (5) |
| declared NOT-RUN rows | 5 | 5 |
| corrections | 4 | **14** |

**Determinism, re-checked after every edit round** by rebuilding into a scratch directory:
`TABLES.md` byte-identical except the `**Built**` timestamp line, `TABLES.json` equal field-for-field
ex-timestamp, **all five figure sha256 identical**. The artefact regenerates from its own sources.

⛔ **Two things I deliberately did NOT do to make the verifier pass:** I did not weaken
`verify_tables.py` (it is byte-identical to w10's), and I did not shape derived or markdown-only
numbers like cells. A `cell` is `{values, provenance}` and the verifier asserts it against the field
on disk; a **derived** number carries `{derived, from, formula}` and a **markdown locator** carries
`{printed, path, lines}`, both with `machine_checked_by_verify_tables: false` on the face of them.
The 854 re-resolved values are all genuine field reads.

---

## 1. THE FIVE, IN THE HEAD'S ORDER — all five CLOSED

### ⭐ 1. MF3 — THE BYTE ROW. **CLOSED.** Four-part ledger on every table that carries the store.

⛔ **Never netted into one number.** Every table that carries a store now prints its four parts
separately, and a table that carries no store says so rather than printing nothing.

**T4, w8/downdate leg — the lead result, the 524,288 B pin at tolerance 0**, per seed 0/1/2, all from
`/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json` unless marked:

| part | seed 0 | seed 1 | seed 2 | pointer |
|---|---:|---:|---:|---|
| 1. atoms — principal subspaces (**the store**) | 524,288 | 524,288 | 524,288 | `/per_seed/{s}/store_bytes` |
| 2a. running covariance, float64 `Σ_k d_k²` | 12,310,672 | 11,941,360 | 11,850,912 | `/cov_bytes_per_seed/{s}` |
| 2b. running means + per-well counts | 16,424 | 16,424 | 16,424 | **derived** — see below |
| 3. request count table | 11,488 | 11,488 | 11,488 | `/c/chlu/outputs/w8-downdate/LOG.md:420,1041` |
| 4. coupling + holes | 5,288 + 1,322 | 5,288 + 1,322 | 5,288 + 1,322 | `/c/chlu/outputs/w8-downdate/LOG.md:421,1042` |
| **honest total** | **12,869,482** | **12,500,170** | **12,409,722** | `/per_seed/{s}/honest_total_bytes` |
| **× the 524,288 B pin** | **24.5466×** | **23.8422×** | **23.6697×** | `/per_seed/{s}/times_over_budget` |

⭐⭐ **A NEW FINDING, and it is the reason the ledger is now printable at all.** The source spoke's own
four-part table (`/c/chlu/outputs/w8-downdate/LOG.md:414-422` and `:1036-1044`) **does not close**:
store + covariance + 11,488 + 5,288 + 1,322 leaves **exactly 16,424 B** unaccounted on **all three**
TEST seeds. I identified it as the **running MEANS** — `Σ_k d_k × 8 = 16,384 B`, and `Σ_k d_k = 2,048
= D` on every seed, from `/per_seed/{s}/content_cost/per_well` `d_k` = [777,939,216,58,58] /
[813,882,208,73,72] / [839,847,225,65,72] — plus the **per-well COUNTS**, `5 × 8 = 40 B`.
**16,384 + 40 = 16,424**, and with it the ledger closes **to the byte** against `honest_total_bytes`
on every seed. ⚠ The *identification* is my inference from `d_k`; the *residual* is arithmetic. The
manifest prints a `ledger closes to the byte?` row (True / True / True) so this is checkable.

⚠ **A second residual, DECLARED and not netted away.** `/per_seed/{s}/items_kept_detail/object_bytes_full`
= 12,885,866 / 12,516,554 / 12,426,106 exceeds `honest_total_bytes` by **exactly 16,384 B on every
seed**. I could not attribute it to a named field. It is printed as `DECLARED RESIDUAL, unexplained`.
It does not move `times_over_budget`, which the artefact computes from `honest_total_bytes`.

**T1 / T4's w9/maze leg (CMAZE-2)**, all three seeds identical, from
`/c/chlu/outputs/w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` `/ledger`:

| part | bytes | pointer |
|---|---:|---|
| 1. atoms (the store) | **31,744** | `/ledger/store_atom_floats` = 7,936 × 4 (float32) |
| 2. sufficient statistic, float64 | **35,904** | `/ledger/statistic_bytes` (4,488 float64) |
| 3. address book (the head) | **1,611,008** | `/ledger/head_floats` = 402,752 × 4 |
| 4. coupling / hole table | ⛔ **DECLARED NOT PRESENT** | see §4 below |
| deployed (atoms + book) | **1,642,752** | `/ledger/deployed_bytes` |
| honest total (deployed + statistic) | **1,678,656** | derived |
| statistic as a share of deployed | **2.19 %** | derived |

⛔ **And a correction to the referee on this leg.** MF3's suggested w9 row reads *"statistic declared
beside the read · 35,904 B = 2.2 % of deployed"* — correct — but T1 is **not under the 524,288 B pin**
at all. Its budget is the matched-PARAMETER budget (mlp 410,920 / lstm 409,846 / transformer 405,776
from `/baseline_params`, against the CLU's 410,688 read floats). The ledger block says so explicitly:
*do not print a `× the pin` figure for this table.*

**T2 (Shapes3D, w8/encoder-loop)** — store **524,288 B** exact per arm per seed
(`/c/chlu/outputs/w8-encoder-loop/run/encoder_loop_report.json`
`/rows/seed{s}:loop_swap/store_bytes/store_bytes`); address book **195,064 B** declared beside,
never netted (`/c/chlu/outputs/w8-encoder-loop/LOG.md:702,871` — markdown locator); no sufficient
statistic and no coupling table, both **DECLARED NOT PRESENT** with the reason.

**T3 (CMAZE-1 at the harness tier)** — ⛔ **NOT APPLICABLE, declared, not omitted.** Every arm is a
learned parametric baseline (mlp 60,047 / lstm 59,965 / transformer 59,819,
`/c/chlu/outputs/w9-harness/harness_seed0.json` `/matched_widths`) or a reference column. No atoms,
no statistic, no address book, no coupling table. **No row in T3 is under the 524,288 B pin.**

**T6 (w5 compose-field)** — new table, see §2 below.

---

### ⭐ 2. MF4 — DEFINE "EXACT". **CLOSED.** A citable field, and every cell flagged.

`TABLES.json` `/exactness_definition` (id `EXACT-STORE-SCOPED`), rendered at the head of `TABLES.md`
and carried in T4's caption:

> **STORE-SCOPED.** EXACT means: after deleting a set of rows, the STORE CONTENT equals, to float
> roundoff, the content fitted from the retained rows alone, **HOLDING THE FROZEN ENCODER AND DECODER
> FIXED**. Neither encoder nor decoder is refit at any point. It is NOT a claim that the renderer
> forgets, and it is NOT retrain-equivalence of the generative model.

**Why only this scope is supported**, with the evidence:

| evidence | path | pointer / locator |
|---|---|---|
| frozen AE trained on eval-bank items: `eval_bank_train` **543 / 529 / 552**, `pool` 4,860, `pool_sites` 270 | `/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_*_test.json` | `/ae_train_items` |
| *"rendered by the same **frozen decoder** through the same latent standardisation"* | `/c/chlu/outputs/w9-maze/LOG.md` | line 376 |
| `refit` = **False** — the deletion refits nothing | `/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` | `/display/delete_green_exactness/refit` |

⛔ **Flagged.** `TABLES.md` names every place the word appears and separates the **two senses**, which
the draft conflates: an **exact JUDGE** is integer arithmetic with no tolerance
(`/c/chlu/outputs/w9-harness/harness_seed0.json` `/judge_audit/exact` = True) and appears in T1's
RESULT and ceiling row headers; **exact DELETION** is the store-scoped definition and appears in T1
§1d's post-deletion rows, all of T4 and all of T5.

⭐ **Also carried, because MF4 and MF5 are the same sentence from two sides:** T4 now prints
`exact_is_not_free` — surviving-target composition on TEST goes **0.1024 (0.1569 / 0.0702 / 0.0802) →
0.0000 (0.0000 / 0.0000 / 0.0000)**, from `/display/delete_green/other_{before,after}/validity` on
the three maze runs.

---

### ⭐ 3. MF7 — THE MISLABELLED ROW. **CLOSED.** Relabelled to what the artefact says.

T4's w8/downdate leg was `class RESULT`. It is now:

> **`ASSERTION BATTERY (object property) - 3 seeds - TEST condition - 0 SCORED RUNS`**

⛔ **The artefact was NOT relabelled to match the table.** `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json`
`/note`, verbatim: *"⛔ **No scored run and no claim**: these are properties of the object (deviations,
byte counts, row counts), not outcomes. **hub/single-design scores the object.**"* The hub
commissioned to score it scored **0 of 3** and is this file's own **NR3** (*"NOT RUN — 0 of 3 scored,
full budget intact"*). Nobody scored that leg.

⭐ The w9/maze leg **keeps** `RESULT`, and now says why: it sits inside the pre-registered scored runs
S1/S2/S3, 3 of 3 spent. Both legs carry an explicit `scored_runs` field (3 and 0).

---

### ⭐ 4. MF10 — THE ENVIRONMENT RELABEL. **CLOSED.** And the two never share a table.

`TABLES.json` `/environments` defines both in one line each and carries the rule.

| | definition | tables |
|---|---|---|
| **CMAZE-1** | the env-scout adopted generator `w8_envscout.maze_colour`, called with its own defaults | **T3** |
| **CMAZE-2** | **a REQUEST LAYER over CMAZE-1**: the engine, the solver and the exact integer judge are `w8_envscout.maze_colour`, **imported unmodified**; the layer adds named agents and named targets inside one maze, across two door-free components, and seals one target behind walls per maze | **T1**, T4's w9 leg, T5 impl. 1 |

*Source of the definition:* `/c/chlu/outputs/w9-maze/LOG.md:366-372` (the FINAL REPORT's own §1).

⭐ **The three numbers, now separated, each with its path:**

| BLIND-SEQUENTIAL | value | where |
|---|---|---|
| published CMAZE-1 | **0.4980** | `/c/chlu/outputs/w8-env-scout/ENV.md` lines 42, 49, 358 |
| **CMAZE-1 at the w9/harness tier** | **0.5317** (0.5550 / 0.5300 / 0.5100) | `/c/chlu/outputs/w9-harness/harness_seed{0,1,2}.json` `/columns/blind-sequential (key order from A)/validity` — **T3** |
| **CMAZE-2** | **0.9037** (0.9077 / 0.8905 / 0.9128) | `/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_*_test.json` `/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity` — **T1** |

Relabelled: T1's title, pins (`environment` = CMAZE-2, plus an explicit `environment_is_NOT` field),
caption, its BAR-3 note, figure 1 (renamed `fig1_cmaze2_main_scoreboard.png`), figure 5's title and
annotation, T4's w9 row label, and the `rules_applied` list. T3's title and pins say **CMAZE-1
PROPER**.

⛔ **T3's caption said `the two are different compute budgets on the same environment`.** That
sentence is gone. It now separates the two grounds explicitly and states that they are a **different
environment, not merely a different budget**, and must never share a table. T3's closing note no
longer says *"the difference is the REQUEST LAYER, not the environment"* — the request layer **is**
the difference between the two environments.

---

### ⭐ 5. MF11 — FRAGMENT-LOOKUP. **CLOSED.** Both quantities printed and labelled.

New block `T1 §1c-bis`. Every value a machine-checked cell:

| column | r=1 | r=2 |
|---|---|---|
| **CROSS-MAZE** (the published tripwire; the split does NOT leak) | **0.0006** (0.0017 / 0.0000 / 0.0000) | **0.0000** (0.0000 / 0.0000 / 0.0000) |
| **SAME-MAZE** (a different quantity, and not a like-for-like arm) | **0.0741** (0.0599 / 0.0960 / 0.0665) | **0.5114** (0.5342 / 0.5364 / 0.4636) |

*Pointers:* `/env_columns/fragment_cross_maze/FRAG-cross-r{1,2}/validity` and
`/env_columns/fragment/FRAG-r{1,2}/validity` on the three scored maze runs.
Carried with them: ⛔ *FRAG reads the true maze at every step, which charter B.3 forbids the CLU*, and
⭐ the exhibit the draft left on the floor — **same-maze r=2 at 81 % of row 2's pipeline ceiling of
0.6319, where the CLU reaches 20 % and the best learned baseline 29 %.**

T3's rows are relabelled `FRAGMENT-LOOKUP r=1 / r=2 (CMAZE-1 tier column)` — their **0.0000 on every
seed** is correct **for that environment** and is not the same quantity as T1's same-maze column.

⚠⚠ **A DISCREPANCY I FOUND AND DID NOT PAPER OVER — correction C11.** The referee (MF11) and
`/c/chlu/outputs/w9-maze/LOG.md:405` quote same-maze r=2 as **0.5105**. The three per-seed
validities are 0.5342 / 0.5364 / 0.4636 at n = **584 / 604 / 632**
(`/env_columns/fragment/FRAG-r2/{validity,n}`). The **unweighted seed mean is 0.51143**; the
**n-weighted pooled mean is 0.51044**, which is the quoted 0.5105. 0.0010 absolute — it changes no
reading, but this file's convention everywhere else is the unweighted seed mean, so **the two must
not be used interchangeably**. `TABLES.md` prints 0.5114 and names 0.5105 as the pooled figure.

---

## 2. THE PHRASING THREE SPOKES FLAGGED — **CLOSED**, as a new TABLE 5

⛔ **"9/9 seeds" is on the never-print list** and `TABLES.json` carries
`/seed_and_configuration_counts` with **seed count and configuration count as SEPARATE fields**:

| implementation | seeds | configurations | legs | scored runs | configuration |
|---|---:|---:|---:|---:|---|
| w9/maze `CluRead.delete_rows_exact` | 3 | **1** | 3 | **3** | K=8, n_atoms=31 |
| w8 `spoke/downdate-build` | 3 | **1** | 3 | 0 | factor K=5, A=64 |
| w8 `hub/single-design` | 3 | **3** | **9** | 0 | ward_balanced K=40 / 44 / 48 — **this is where "9/9" comes from** |

⭐ **TABLE 5 prints the three implementations as three rows**, bands kept apart:

| impl. | band | source |
|---|---|---|
| 1. w9/maze (CMAZE-2, TEST, **3 scored runs**) | statistic **7.522e-16 … 8.826e-16**; subspace projector **exactly 0.000e+00** | `/display/delete_green_exactness/{statistic_max_rel_residual,subspace_max_projector_residual}` on the three maze runs |
| 2. w8 downdate-build (TEST, 0 scored) | **2.842e-14 … 1.126e-09** — five orders coarser | `DOWNDATE-READY-test.json` `/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev` = 4.547e-13 / 1.126e-09 / 2.842e-14 |
| 3. w8 single-design (TEST, 0 scored) | **5.551e-16 … 1.490e-13** on seed 0; worst over all nine legs **0.094 atom ulp / 0.0251 subspace ulp** | `/c/chlu/outputs/w8-single-design/LOG.md:472` (absolutes), `:476-481` (atom), `:495-503` (subspace) |

⚠ Row 3's numbers exist only in markdown and are carried as **locators**, flagged
`machine_checked_by_verify_tables: false`. `verify_tables.py` re-hashes that file but cannot
re-resolve those values. That is a limitation of the evidence base, declared on the face of the row.

⭐⭐ **THE FAILURE AGREEMENT IS NOW A CITABLE ROW** (`TABLES.json` `/tables/T5/failure_agreement`).
Same object (`factor K=5, A=64`, the incumbent's coarse imbalanced partition), same TEST seed 2, same
well, different code, different authors, different sign conventions:

| leg | float32 ulp at unit scale | source |
|---|---:|---|
| downdate-build, atom level | **32.90** (= 3.921e-06); also **35.46** (= 4.227e-06) after the cut | `/c/chlu/outputs/w8-downdate/LOG.md:625-640`, `:728,794,905` |
| downdate-build, subspace level | **60.36282** | `DOWNDATE-READY-test.json` `/a1_subspace/worst_in_float32_ulp` ⭐ a stored field |
| downdate-build, its own subspace floor | **66.11072** | `/a1_subspace/worst_floor_subspace_in_float32_ulp` ⭐ a stored field |
| single-design, atom level | **14.377** | `/c/chlu/outputs/w8-single-design/LOG.md:479,481` |
| single-design, subspace level | **14.4518** | `:501` |
| single-design, its own two-order floor | **36.9305** | `:501,506` |

⭐⭐ **The citable sentence, and it is stronger than either implementation's successes:** the second
implementation reproduced the same seed-2 failure with different code and its own sign convention —
**14.377 ulp against 32.90** — and **both implementations' own reproducibility floors EXCEED their
observed deviations there** (66.11 > 60.36; 36.93 > 14.45), so both independently conclude that seed
2's content is not a function of its data **by any method**. Closed by a variance-share admission cut
with **no free parameter**, `τ` = **4.144858e-09** (= 4.941056e-16 / 1.1920929e-07,
`/per_seed/{s}/content_cost/var_share_tol`), declared before the re-run and not moved.

---

## 3. THE OTHER ARTEFACT-LEVEL MUST-FIXES I ALSO CLOSED

These were wrong **in `TABLES.*`**, so they were mine and not the Head's.

- ⭐ **MF6 — the two caveats inside the cited gate file. CLOSED.** Both now print under T4's w8 row
  and in its caption. (a) `/per_seed/{s}/cut_is_marginal` = **True on all three seeds**. (b)
  `/a1_subspace/passed_every_seed` = **False**, `seeds_failing` = **["2"]**, at **60.36282 ulp**
  against its own **66.11072 ulp** floor — so the failure is a property of the **content**, not the
  downdate, which is what `/reproducibility_floor/what` says. And the trap is named: the top-level
  `all_three_assertions_passed = true` **EXCLUDES** `a1_subspace`.
- ⭐ **MF8 — T4 printed the narrower field. CLOSED.** The spoke's own row is now first:
  `/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev` = **4.547e-13 / 1.126e-09 / 2.842e-14**
  ("worst over the pinned 11,024-row deletion AND the random-64 deletion"), **19× and 4×** larger on
  seeds 0 and 2 than the `max_abs_dev_per_well_by_seed` figure the draft printed. The old field is
  kept, labelled, so the difference is visible. **Figure 4 now plots the corrected field.**
- ⭐ **MF9 — only one unresolvable leg was declared. CLOSED.** Each leg's own floor is a printed cell
  (`/per_seed/{s}/reproducibility_floor/floor_max_abs_dev` = 2.127e-13 / **4.282e-09** /
  **5.684e-14**) with a `resolvable at all?` row reading **True / False / False**: **two of three**
  TEST seeds, not one. Figure 4 marks the floors.
- ⭐ **MF12 — the parameter match was quoted from a corrected-away figure. CLOSED, as TABLE 6.**
  From `/c/chlu/outputs/w5-compose-field/scored/test_seed{0,1,2}/READ-TABLE.json`: `pca_tensor`
  `param_count` 131,072 + `address_bytes`/4 (189,344 / 195,064 / 195,064) = **178,408 / 179,838 /
  179,838** against `rival_mlp_180k` `param_count` **178,952** ⇒ **−0.304 % / +0.495 % / +0.495 %**,
  not the val leg's −0.01 %. **On two of three seeds the arm carries MORE parameters than the bar**,
  stated that way round.
- ⭐ **MF13 — "at 524,288 B" invited a byte comparison the arm does not win as stated. CLOSED in
  TABLE 6, and it comes out STRONGER.** Total deployed = `store_bytes` + `address_bytes` =
  **713,632 / 719,352 / 719,352** against the rival's 178,952 × 4 = **715,808** ⇒ **99.70 % /
  100.50 % / 100.50 %** — a genuinely matched comparison to within ±0.5 %, which is a better claim
  than "27 % cheaper" and is the one the artefacts prove.
- ⭐ **MF14 — row 4 has no BLIND column in existence. CLOSED as a DECLARED GAP** (`T1 §1c-ter`).
  `/env_columns` carries exactly four keys — `unseen_pair`, `nway`, `fragment`,
  `fragment_cross_maze` — on **every** scored seed. There is no blind block for the
  budget-interpolation row on any seed. Declared NOT MEASURED, **not a blank and not a zero**, and
  the charter rule in `rules_applied` now carries the exception.
- ⭐ **MF15 — the tier-A negative the draft dropped. CLOSED at artefact level.** T1 §1d's refusal
  block now carries the reading, not just the rows: *refusal is **novelty detection**, not wall
  detection*; the store refuses the reachable-but-unwritten combination as often as the walled one;
  false-refusal rate on grounded-possible requests **0.8182 against a pre-registered 0.20** (a 4×
  miss); the supervised refuser separates better; **no refusal win is claimed** (registry `N-I8-19`).
- ⭐ **MF16 — "at K ≥ 36 composition collapses" is stated unconditionally. CLOSED as a qualifier on
  NR3.** The NOT-RUN register now carries: the claim is true **of the one configuration at K ≥ 36
  whose composition was measured** (K=40, A=7, span 280, 519,794 B = 0.991×, the 23:22Z entry). The
  18:32Z `ward_balanced` sweep lists **five** configurations that fit the pin at float64 — K=36 (span
  144), K=38 (266), K=40 (360), K=42 (504), K=48 (864) — and **four were never measured**; three of
  them carry **more** deployed span than the object that composes at 0.8918. The threshold is also
  **dtype-conditional**: a float32 matched-bytes point at K=21, A=94, 517,952 B = 0.988×, span 336
  (`/c/chlu/outputs/w8-single-design/LOG.md:99`) was excluded **by argument** and never measured.
  Supported form: *no object **with a float64 sufficient statistic** composes and forgets inside
  524,288 B, on the one such configuration measured.*
- ⭐ **SF2 — the ceiling ladder's rungs differ 9× in address-book size. CLOSED in TABLE 6.**
  `oracle_span` and `span_head` carry `address_bytes` = **1,766,400** — **9.3×** the arm's own book
  and **3.4×** the store pin — while `oracle_content_{fine,items,pool}` and `oracle_span_field` carry
  **0**. All machine-checked cells.
- ⭐ **A bug I found and fixed while rendering:** booleans were being averaged, so `refit performed`
  printed as `0.0 (False / False / False)`. `cell()` now excludes `bool` from the numeric mean.

⭐ **All fourteen changes are recorded as CORRECTIONS C5–C14 inside the manifest itself**, each with
its claim-as-written, what the artefact says, the effect, the provenance and the action taken, so the
artefact documents its own revision and the Head can cite each one by id.

---

## 4. ⛔ DECLARED OPEN ITEMS — nothing was silently dropped

### 4.1 A GAP IN THE EVIDENCE BASE (the falsifier the task named)

⛔ **The w9/maze leg has NO coupling table and NO hole table in any artefact.** I grepped all three
scored run files (`/c/chlu/outputs/w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json`): zero
occurrences of `hole` and zero of `coupl`. Part 4 of that leg's four-part ledger is therefore printed
as **DECLARED NOT PRESENT** with the reason — that leg's deletion downdates `(S, t, n)` and the head
statistics `(A, Mx)` and has no interface hole table of the kind the w8 leg carries. ⚠ **This is a
gap in the evidence base, reported as one**, exactly as the task's falsifier requires. It is not a
zero. Closing it would need the w9 instrument re-instrumented to emit a coupling/hole ledger — **a
new measurement**, so I did not invent the number.

Same status, same reason, for **T2**: no sufficient statistic and no coupling table exist on the
encoder-loop instrument (it makes no deletion claim), and its address-book figure **195,064 B** exists
only in markdown (`/c/chlu/outputs/w8-encoder-loop/LOG.md:702,871`), not in any JSON field.

### 4.2 THE UNEXPLAINED 16,384 B

⚠ `object_bytes_full − honest_total_bytes` = **exactly 16,384 B on all three TEST seeds**. I could
not attribute it to a named field and refused to net it away. Printed as a DECLARED RESIDUAL. It does
not move `times_over_budget`. **Closing it needs the serialiser's own inventory, not a run** — the
26-array inventory the spoke describes at `/c/chlu/outputs/w8-downdate/LOG.md:1030` would settle it
if someone dumps it.

### 4.3 MUST-FIXES THAT ARE NOT ARTEFACT-LEVEL — the Head's, and untouched by me

⛔ I fix files. These are sentences, and the Head writes sentences. Each is listed so it is not lost.

| item | why it is not mine | what I did leave for it |
|---|---|---|
| **MF1** — §1.2 points the Head at a sentence containing novelty **STOP #1** (`request-indexed tensor head`) | the pointer is in `DRAFT-v0.md`, and the phrase is in `w8-novelty/NOVELTY.md` §3 — neither is in my artefact | nothing needed from the tables |
| **MF2** — *"whose whole content is a sufficient statistic"* is false of both legs | a contribution noun phrase | ⭐ the numbers that decide it are now printed: statistic / deployed = **2.19 %** (w9) and statistic / store = **2,348 % / 2,278 % / 2,261 %** (w8, computable from the printed ledger) |
| **MF5** — "exact deletion" printed with no cost | a section-3 sentence | ⭐ T4 now prints `exact_is_not_free`: **0.1024 → 0.0000** on TEST. ⚠ **OPEN:** the w5 item-store eviction figures (**−2.09 / −0.91 / −6.74** against ±1.0) come from `/c/chlu/evidence/outputs/w5-advisor-report.md` §2, which is **not a registered source** — see 4.4 |
| **SF1** — §4 uses `oracle_span`'s numbers where `oracle_span_field` is meant | a validity comparison, not a byte fact | ⚠ **OPEN in my artefact.** T6 registers the w5 READ-TABLEs, so the fix is now one cell away: `oracle_span` reads 44.1837 / 50.8980 / 53.0000 and `oracle_span_field` 40.9388 / 50.4286 / 54.2041. I did not add the validity rows because that is a new T6 section beyond the byte ledger this pass was scoped to |
| the remaining **SF3–SF20** and the eight nice-to-haves | prose | not attempted; the referee's file is the register |

### 4.4 WHAT I DID NOT REGISTER, AND WHY

⚠ The advisor reports (`/c/chlu/evidence/outputs/w5-advisor-report.md`,
`w9-advisor-report.md`, `w10-advisor-report.md`) and `/c/chlu/evidence/outputs/w10-registry-fold/`
are **not** in the source register. `/c/chlu/evidence/` is read-only by standing rule and I read them
for context, but registering a file makes `verify_tables.py` re-hash it on every run, which would tie
the tables' acceptance to files another wave may edit. ⛔ **Consequence, declared:** any number whose
only home is an advisor report or the registry fold — including MF5's **−2.09 / −0.91 / −6.74** and
`N-I8-19`'s tier — cannot be a cell in this manifest. `N-I8-19`'s *numbers* (0.9801 / 0.9781 /
0.8182) **are** cells, because they are in the maze runs; only its registry id and tier are not.
**Hub decision, not mine:** whether the evidence base's own reports should join the register.

### 4.5 ⛔ NOT DONE, AND IT NEEDS A RUN

Nothing in this pass needed one, and **no run was taken**. For completeness, the two items in the
referee's list that cannot be closed without new measurement are, by name:

1. **NR5** — w9/maze post-deletion composition with the **repaired** deletion, on TEST at 3 seeds.
   Measured on VAL seed 0 only (−0.0120, inside BAR 6's ±0.05). The maze budget is **0, exhausted in
   w9**. The launch line is already in the NOT-RUN register.
2. **MF16's four unmeasured pin-fitting configurations** (K=36 / 38 / 42 / 48) and the float32
   K=21 / A=94 point. Closing them needs `exp_w8_single_design --stage score` runs against
   `single-design`'s 3/3 live budget. ⛔ **I did not invent those numbers**; I qualified the claim
   instead, which is the zero-run fix.

---

## 5. ⛔ WHAT WOULD PROVE THIS SPOKE WRONG — checked against my own result

| the task's falsifier | checked |
|---|---|
| **a must-fix that cannot be closed without a new measurement ⇒ say so and name the run** | ⭐ none of the five needed one, and none was taken. The two items that DO need a run are named in §4.5 with the run they need. ⛔ No number was invented. |
| **the byte ledger for some arm does not exist in any artefact ⇒ report it as a gap** | ⛔ **This fired, twice.** The w9/maze leg has no coupling/hole table in any artefact, and T2's address book exists only in markdown. Both are printed as DECLARED NOT PRESENT / markdown-locator with the reason, in §4.1. Neither is a zero. |
| **a fix that breaks the verifier is not a fix** | ⭐ `verify_tables.py` is byte-identical to w10's and **PASSES**: 32 sources, 286 cells, 854 values re-resolved, 5 figure hashes. Before: PASS at 27 / 240 / 720 / 5. |
| **ZERO compute, ZERO scored runs** | ⭐ held. Only `build_tables.py` (re-reads artefacts, re-plots 5 PNGs) and `verify_tables.py` ran. |
| **write only to my own output dir and worktree** | ⭐ held. `/c/chlu/evidence/` not written; `/c/chlu/outputs/w10-*` not written; no `w12-*` anything touched; no box-wide kill. |

---

## 6. WHAT THE HUB AND THE HEAD SHOULD TAKE FROM THIS

1. ⭐ **The byte row exists now, on every table that has a store, and it closes to the byte.** The
   sentence MF3 asked for is available verbatim: *the object that forgets exactly is itself 24.55× /
   23.84× / 23.67× the 524,288 B pin, which is why §5 exists.*
2. ⭐ **"Exact" is a citable field**, and the two senses of the word are separated in the artefact so
   the prose does not have to keep them apart by hand.
3. ⛔ **T1 and T3 can no longer be merged by accident** — different `environment` field, different
   title, different caption, and the rule is in `rules_applied`.
4. ⭐⭐ **The strongest sentence in the deletion story is now a row**: the failure agreement, with both
   implementations' floors exceeding their own deviations.
5. ⚠ **One number the Head must choose knowingly:** same-maze FRAGMENT r=2 is **0.5114** unweighted
   (this file's convention) or **0.5105** pooled (what the LOG and the referee quote). Correction C11
   names both.
