# gen-c-set-a7 — experiment-engineer report

**Task + acceptance criterion:** ship **A7** (the term-count control, built from valid requests alone) and
make the **no-model structural detector a suite test that runs on the registered query-set artifact
discovered from disk**, asserting AUROC ≤ 0.5 + tolerance against the declared control; plus make
`load_c_records` **raise** on a wrong `data_root`.

**Status: done.** 1 new test file (**24 tests**: 21 + 3 disk-discovered registries), A7 + registry emitted and
shipped for **Shapes3D, dSprites and the wide condition**, all seven pre-existing artifacts reproduce
**bit-identically**, and the **full suite is green at the final HEAD: 2275 passed, 0 failed** (§7).

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 3 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐ **The re-score is now unblocked, and `A7` is on disk** at
>    `.claude/data/gen/{shapes3d,dsprites,shapes3d/recomb2element_wide}/a7_redundant.json`.
>    ⛔ **`exp_gen_baselines` and `exp_gen_certificate` do not read it yet** — they still build (or
>    do not build) their own A7 in memory. Wiring the *artifact* into those two experiments is
>    **outside my file scope** and needs an owner before the re-score, or the two paths can drift.
>    (§6)
> 2. ⚠ **`load_c_records` now RAISES.** Any pipeline or cluster script that relied on the old
>    silent-skip will now fail loudly — which is the point, but it is a behaviour change on a
>    shared path used by `exp_gen_store`. `scripts/csf3/` is not mine to edit; whoever owns the
>    launcher should confirm the `--data-root` it passes is the **parent** spelling. (§5)
> 3. ⚠ **The `id` field and the record key-set are set labels, not features** — excluded from the
>    assertion by declaration, *measured and reported anyway* (id suffix 0.9912, record key count
>    1.0000). If the Hub disagrees with that boundary, the fix is in the C set's record schema, not
>    in the detector. (§4.3)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — instrument repair.* ⛔ No store, no arm, no CLU number is produced here.
- **Laundering control:** the no-model detector **is** the control, and it is the deliverable.
- **Falsifies:** the detector still separates C from its valid control above chance after A7 ⇒
  another structural giveaway remains. **It did not fire:** every feature, every kind, both
  datasets **and the wide condition**, **exactly 0.5000**.
- **Does NOT falsify:** `out_of_range` remaining separable by a **grid lookup on the values** — that
  is intrinsic to the kind (certificate §3a), is a *value* property, and the detector is
  value-blind by construction.

## FLAG-PROVENANCE TABLE (protocol §5, mandatory)
| | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-c-set-a7` — `c6a9bfe`, `74070df`, `465ac89`, `23b753d`, `3737757`, `dd0c842` (HEAD) |
| base | ⚠ **`main @ 1bcfcb9`**, not the `3faef27` named in the task: `main` had already moved (`gen-baselines` merged) and `3faef27` is its ancestor. No conflict; no rebase needed. |
| datasets / conditions | `shapes3d` (`recomb2element`), `shapes3d` (`recomb2element_wide`), `dsprites` (`recomb2element`) — **all three** registries audited |
| C-set config (unchanged defaults) | `c_invalid_n_per_kind None` → derived `ceil(n_test/2)` = 36 (s3d) / 3146 (dsprites) / 468 (wide) · `c_invalid_seed 0` |
| **new** `gen_data` config | `emit_a7 True` · `a7_n_requests None` → `2 * n_per_kind` = `\|C\|` · `a7_seed 0` |
| A7 row pool | ⛔ **TRAIN rows only**, deterministic `np.random.default_rng(a7_seed).choice`, sorted; asserted disjoint from `test_rows` |
| detector tolerance | **0.05**, declared in `tests/test_gen_c_set_a7.py::DETECTOR_TOLERANCE` with its derivation |
| detector features | **11** structural (the only asserted view) + **3** diagnostic + **2** definitional, reported; factor values masked to `0.0` before any feature is computed |
| `base` control sample | 2,048 rows drawn from `train ∪ test` at `base_seed=0` (the detector's A∪B pool) |
| JAX / env | **main venv** `/Users/user/Desktop/CHLU/.venv` via `PYTHONPATH=<worktree>` (protocol §4). ⛔ **no `uv sync` in the worktree.** JAX 0.9.0 |
| worktree | `../CHLU-gen-a7`, fresh, `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen` |

---

## 1. ⭐ THE RESULT — detector AUROC before and after A7, per kind and pooled

Orientation-free AUROC `max(a, 1−a)`, **max over all 11 structural features**. "Before" = the kind
scored against the uncontrolled 6-term valid pool (A∪B), which is the world as it shipped; "after" =
the kind scored against the control the registry now declares for it.

| dataset / condition | kind | n (inv/ctrl) | **BEFORE** (vs `base`) | control | **AFTER** |
|---|---|---|---|---|---|
| shapes3d | `duplicate_factor` | 36 / 72 | **1.0000** | `a7_redundant` | **0.5000** |
| shapes3d | `out_of_range` | 36 / 2048 | 0.5000 | `base` | **0.5000** |
| shapes3d | ⭐ **pooled** | 72 | **0.7500** | — | **0.5000** |
| dsprites | `duplicate_factor` | 3146 / 6292 | **1.0000** | `a7_redundant` | **0.5000** |
| dsprites | `out_of_range` | 3146 / 2048 | 0.5000 | `base` | **0.5000** |
| dsprites | ⭐ **pooled** | 6292 | **0.7500** | — | **0.5000** |
| shapes3d/wide | `duplicate_factor` | 468 / 936 | **1.0000** | `a7_redundant` | **0.5000** |
| shapes3d/wide | `out_of_range` | 468 / 2048 | 0.5000 | `base` | **0.5000** |
| shapes3d/wide | ⭐ **pooled** | 936 | **0.7500** | — | **0.5000** |

⭐ **The "before" column reproduces `gen-certificate` §3 exactly** (`count` arm: 1.0000 / 0.5000 /
0.7500) — from a completely independent implementation, on the same shipped `c_invalid.json` files.
That is the cross-check that the leak I closed is the leak that was found.

**AFTER is not "≤ 0.55", it is 0.5000 on every one of the 11 features**, per kind, both datasets:

| feature | dup vs `base` (BEFORE) | dup vs `a7_redundant` (AFTER) |
|---|---|---|
| `n_terms` | **1.0000** | 0.5000 |
| `n_redundant_terms` | **1.0000** | 0.5000 |
| `max_factor_multiplicity` | **1.0000** | 0.5000 |
| `first_duplicate_position` | **1.0000** | 0.5000 |
| `masked_request_json_len` | **1.0000** | 0.5000 |
| `is_factor_order_sorted` | 0.9167 | 0.5000 |
| `n_order_inversions` | 0.9167 | 0.5000 |
| `last_factor_index` | 0.9167 | 0.5000 |
| `sum_factor_indices` | 0.9167 | 0.5000 |
| `n_distinct_factors` | 0.5000 | 0.5000 |
| `first_factor_index` | 0.5000 | 0.5000 |

⚠ **Four features besides term count were already above 0.9** (the *ordering* ones: appending the
duplicate at the end makes the factor-index sequence unsorted, which `is_factor_order_sorted`,
`n_order_inversions`, `last_factor_index` and `sum_factor_indices` all see). ⛔ **A fix that only
equalised term count would have left 0.9167 on the table.** A7 closes them because it appends its
redundant term under the *same* rule (end of list, factor index cycling `i % n_factors`) — which is
why the two sets tie exactly rather than approximately.

Full per-feature tables for all three registries: `.claude/outputs/gen-c-set-a7/detector_report.json`.

## 2. Pre-registration scorecard (`.claude/outputs/gen-c-set-a7/PREREG.md`, written before the harness ran)

| # | prediction | realised | verdict |
|---|---|---|---|
| P1 | before: dup 1.0000, oor 0.5000, pooled 0.7500 | 1.0000 / 0.5000 / 0.7500, ×3 registries | ⭐ **exact** |
| P2 | after: every feature exactly 0.5000, per kind and pooled | 0.5000 on 11/11 features, 6/6 kind-arms | ⭐ **exact** |
| P3 | the 0.05 tolerance is 100 % unused | unused: realised max 0.5000 | ⭐ **held** |
| P4 | falsifier: any feature > 0.55 after A7 ⇒ spoke has not closed the leak | did not fire | — |
| P5 | `c_invalid.json` reproduces byte-for-byte | 6/6 artifacts bit-identical (§3) | ⭐ **held** |

⚠ P1 was *derived from the builder's source*, not guessed, and it names the ordering features that a
term-count-only reading of the bug would have missed — which is the part of the prereg that earned
its keep.

## 3. ⛔ The C records are untouched — asserted, not promised

The emitter was re-run into a scratch root (`.claude/scratch/gen-c-set-a7/{s3d,dsp}`, dataset files
symlinked) so nothing shipped was overwritten, and every pre-existing artifact was re-derived and
compared:

```
OK  shapes3d/c_invalid.json                      b886da239e97b19e9001a88ce04fded329dda4884d301f78c3097ea6cc66ba49
OK  shapes3d/factor_index.npz                    5f1fce25c40cceae6de2d00fd2145c4cdc73dcc38cee7f6545cbcaea3ce78f49
OK  shapes3d/recomb2element_wide/c_invalid.json  34cfcf2ee46c7131d0835b8dad227f802a148207f67a21ccbb08acd197ae89cd
OK  shapes3d/recomb2element_wide/factor_index.npz e6303358dc020f847365aa4736b4ecb062b157b8cc2f97a17e81a77f790deedb
OK  shapes3d/recomb2element_wide/wide_cells.json 4ed655e957f273b799e544477a7a7a1759860b9d0569aa706b97dd2afada9b4d
OK  dsprites/c_invalid.json                      10a39e713864c9f0c28f740caad49394cbda6f34f3d1d29bbc61dab9fc288666
OK  dsprites/factor_index.npz                    c6419fedd437e1ff6ec630839a1a4b16b02dc04b573208d8a5b734931e15fe45
```

Only the two **new** files were then copied into the real data root (`cp -n`, never overwriting).
A unit test asserts the same property independently of the data:
`test_emitting_a7_leaves_the_c_set_bit_identical` builds with `emit_a7=True` and `emit_a7=False` and
compares bytes.

### The shipped A7 artifacts
| path | sha256 | records | terms |
|---|---|---|---|
| `shapes3d/a7_redundant.json` | `536773ed9c1bb8cdede0dbd956a5921f0f0bc7746f14c1cb6a261ff9c4e01191` | 72 | 7 |
| `dsprites/a7_redundant.json` | `482cba0182275762048ab4e35293805b46d4f2482c50e72e9f4e38bf2f7b3afc` | 6292 | 6 |
| `shapes3d/recomb2element_wide/a7_redundant.json` | `4869a9f0d3018d970821024b61aed20133f4d82c35ee5fd8a7dfc8ff12916f53` | 936 | 7 |
| `shapes3d/query_sets.json` | `42031e4b65c399d4a81848e279856147bf760501105a98c4c36b60ff0198b1a3` (registry) | — | — |

⚠ **dSprites A7 is SIX terms, not seven** — dSprites has 5 factors, so `n_factors + 1 = 6`. The
"7 terms" framing is Shapes3D-specific; the invariant is `n_factors + 1`.

**`|A7| = |A|` — the convention, stated.** The construction is a **1:1 map over valid requests**
(`build_a7_redundant(grid, rows)` emits exactly one record per row, asserted in
`test_a7_is_deterministic_and_one_per_valid_request`), so any consumer passing its own A gets
`|A7| = |A|` — which is what `gen_certificate.redundant_requests` does in memory (1,024 = |A|). The
**shipped file** instantiates A as a deterministic **train-row** sample of size
`a7_n_requests` (default `|C|`), so the artifact is class-balanced against the set it controls for.
⛔ Held-out rows are never spent on a control (`test_a7_never_spends_a_held_out_row`).

## 4. ⭐ The detector — why it is a test and not a convention

### 4.1 It runs on the REGISTERED artifact, discovered from disk
`discover_query_set_registries()` globs `**/query_sets.json` under the gen data root; the suite test
is `@pytest.mark.parametrize`d over whatever it finds. Realised collection in this run:

```
tests/test_gen_c_set_a7.py::test_registered_query_sets_on_disk_are_at_chance[dsprites]
tests/test_gen_c_set_a7.py::test_registered_query_sets_on_disk_are_at_chance[shapes3d]
tests/test_gen_c_set_a7.py::test_registered_query_sets_on_disk_are_at_chance[recomb2element_wide]
```

⛔ **There is no list of today's sets anywhere in the test.** A new dataset directory, a new
condition, or a new query set is audited the moment it is emitted — and
`test_every_query_set_on_disk_is_registered` fails if a `c_invalid.json` ever appears *without* a
`query_sets.json` beside it, which is the only way to escape the audit.
⚠ An empty parametrisation would make the test vanish silently, so a missing data root yields one
explicitly **SKIPPED** case (verified by running the file in the worktree, which has no `.claude/`).

### 4.2 Three loud failure modes, each with a test that makes it fire
| guard | test | what it prevents |
|---|---|---|
| ⭐ **positive control** — point the dup kind's control back at `base` | `test_the_detector_catches_the_leak_it_was_built_for` (measures **1.0000** and pooled **0.75**, then raises) | the detector quietly becoming decoration |
| an invalid **kind** with no declared control | `test_an_invalid_kind_with_no_declared_control_fails_loudly` | ⭐ **a future C kind shipping uncontrolled** — the inheritance mechanism |
| a record **field** outside the allowlist | `test_a_new_record_field_fails_loudly` | the next giveaway arriving as a new column |
| a **term-key order** other than the single-sourced one | `test_term_key_order_is_single_sourced` | key order becoming a giveaway |
| a registered set **missing from disk** | `test_a_registered_set_missing_from_disk_is_not_a_silent_pass` | a silent pass on an absent artifact |

### 4.3 ⚠ What the assertion deliberately does NOT score, and why
The scored view is **the request term list with every factor value masked** — the object an arm
consumes. Excluded **by declaration** (`EXCLUDED_RECORD_KEYS`, written into `query_sets.json`):
`kind`, `violated_factor`, `reason` (the answer key), `row`, `redundant_factor` (provenance).
⛔ **Also excluded: the record `id`.** A set's ids name their set by construction (`shapes3d-dup-003`
vs `shapes3d-a7-003`); no id scheme can be both unique-per-set and uninformative. Rather than hide
that, the detector **measures** the id features and reports them:

| diagnostic feature (NOT asserted) | dup vs base | oor vs base |
|---|---|---|
| `id_numeric_suffix` | 0.9912 (s3d) / 0.6745 (dsprites) | same |
| `record_key_count` | **1.0000** | **1.0000** |
| `id_token_count` | 0.5000 | 0.5000 |

⚠ **`record_key_count` is 1.0 because C records carry `violated_factor` and `reason` and valid ones
do not** — i.e. the C file ships its own answer key, as it must for attribution scoring. An arm that
reads those is not being fooled by a structural leak; it is reading the label. **If the Hub wants
that boundary drawn elsewhere, the change is to the C record schema (⛔ which I was told not to
touch), not to the detector.**

### 4.4 ⭐ THE RESIDUAL A7 CANNOT CLOSE — found by being adversarial, and it is not structural
The task warned that a detector which only tried term count would prove this spoke wrong. Trying
harder turns up one more separator, and it is worth the Hub's attention:

⭐ **`duplicate_factor` means "one factor, two different values", which is decidable from the request
by VALUE EQUALITY alone** — no values read, no grid, no model, one line of code. It is the exact
counterpart of the grid lookup that decides `out_of_range` (`reach`, AUROC 1.0000, certificate §3a).

| definitional feature (NOT asserted) | dup vs base | oor vs base |
|---|---|---|
| `has_conflicting_repeat` | **1.0000** | 0.5000 |
| `n_distinct_values` | 0.7545 (s3d) / 0.9309 (dsprites) / 0.8048 (wide) | ~0.52–0.55 |

⛔ **No control can remove this and A7 does not claim to**: it is *what makes the record invalid*.
A7 closes the **structural** giveaway — request length and shape, which had nothing to do with the
contradiction — and the structural view stays at 0.5000 with the definitional one at 1.0.
⚠ **What this means for the cell's claim, stated plainly:** *both* C kinds are decidable in factor
space by a syntactic rule. The refusal claim is therefore **not** "no rule could decide these"; it
is that **the barrier is computed from the store's own geometry with zero invalid supervision**, and
that it beats the byte-matched competitors *on the controlled metric*. ⛔ Any paper sentence of the
form "impossible requests cannot be detected without a model" is false and must not be written.
It is reported per registry in `detector_report.json` under `definitional_vs_base`, and pinned by
`test_the_definitional_giveaway_is_reported_not_hidden`.

## 5. `data_root` fails loudly (R26 ruling 7)
`load_c_records` raised a warning and returned `[]`. It now raises `FileNotFoundError`, with **no
fallback** (a fallback would also hide a `--data-root` pointing at the *wrong dataset*), and the
message resolves the parent/dataset-dir ambiguity when that is the actual mistake. Three tests:
the raise, the disambiguating message, and ⛔ the negative control that a **correct** root still
loads 72 records — so the raise is not reachable from a good configuration.

## 6. How I verified (commands + observed output)

```bash
# 1. the emitter, end to end, into a SCRATCH root (dataset files symlinked) so
#    nothing shipped could be overwritten -- all three conditions
PYTHONPATH=../CHLU-gen-a7 .venv/bin/python -c "...cmd_data_shapes3d / cmd_data_dsprites..."
#   -> c_invalid / factor_index / wide_cells sha256 IDENTICAL to the shipped files (7/7)
#   -> a7_redundant.json + query_sets.json emitted, 72 / 6292 / 936 records
# 2. the audit on the REAL artifacts, discovered from disk
#   -> 3 registries, 0 violations, every kind 0.5000 (table §1)
# 3. the new test file, from the worktree, on the main venv
PYTHONPATH=../CHLU-gen-a7 CHLU_GEN_DATA_ROOT=.../.claude/data/gen \
  .venv/bin/python -m pytest tests/test_gen_c_set_a7.py -q --no-cov -p no:cacheprovider
#   -> 24 passed   (21 + 3 disk-discovered registries)
# 4. the same file with NO data root (a clone / a worktree)
#   -> 19 passed, 2 skipped  (⛔ the parametrised test SKIPS visibly, it does not vanish)
# 5. ruff
.venv/bin/python -m ruff check <5 files>   -> All checks passed!
.venv/bin/python -m ruff format --check chlu/data/gen_splits.py tests/test_gen_c_set_a7.py -> clean
#   ⛔ chlu/cli/data_cmd.py was ALREADY non-conforming at HEAD (verified by formatting
#      `git show HEAD:chlu/cli/data_cmd.py`), so I did not reformat it -- that would have
#      been a 349-line diff outside my scope.
```

⚠ **Machine load caveat (honest):** for most of the session **four other spokes were running full
suites concurrently** (`ps` showed 5 pytest processes). The same 24-test file took **8.5 s** early
and **195 s** later. ⛔ No timing here is a performance number — except the reported full suite
(§7), which I deliberately re-launched **after the machine was idle** so its 1:49:37 is at least
comparable to a normal run.

## 7. Suite

**AFTER — full suite at HEAD `dd0c842`, main venv, from the worktree:**

```
PYTHONPATH=../CHLU-gen-a7 CHLU_GEN_DATA_ROOT=.../.claude/data/gen \
  .venv/bin/python -m pytest -q --no-cov -p no:cacheprovider
2275 passed, 45 warnings in 6577.39s (1:49:37)
```

⭐ **2275 passed, 0 failed, 0 errors, 0 skipped.**

**BEFORE — the delta is exactly my new file, shown by collection at the base:**

```
git checkout 1bcfcb9 && pytest -q --collect-only   ->  2251 tests collected in 37.38s
```

**2275 − 2251 = 24 = exactly `tests/test_gen_c_set_a7.py`.** ⛔ No pre-existing test was added,
removed, renamed or re-parametrised, and every one of the 2,251 that existed before passes at my
HEAD. ⚠ **I did not run the full suite a second time at the base**, and I am saying so rather than
implying it: with **zero** failures after, there is no failure whose provenance a baseline run could
disambiguate. The collection count is the part that *can* hide a change, and it is shown.

⚠ **Two earlier full-suite attempts were discarded, honestly:** one was killed at 18 % when I chose
to land a further change rather than report a green suite for a tree the run had not tested
(§8 item 4), and one died with the session at 18 %. The run above is the only one reported, it is at the
**final** HEAD, and the tree was `git status`-clean throughout it.

## 8. Open questions / follow-ups / risks
1. ⭐ **Nothing consumes the A7 artifact yet** (reconciliation item 1). `exp_gen_certificate` builds
   its own A7 in memory; `exp_gen_baselines` builds none. The re-score needs one of them to read
   `a7_redundant.json`, and `test_the_on_disk_a7_matches_the_in_memory_rule_used_by_the_certificate`
   already guards the two implementations against drift in the meantime.
2. ⚠ **`out_of_range` remains separable by a grid lookup on the VALUES** (AUROC ~1.0 for `reach`,
   certificate §3a). That is not a structural leak and A7 does not address it; it is a property of
   the kind. The detector is value-blind, so it reports 0.5000 there — ⛔ **do not read that 0.5000
   as "oor is hard"**.
3. ⚠ The `base` control is sampled at **2,048** rows from A∪B for speed. With `|C[oor]| = 3146` on
   dSprites the invalid side is larger than the control; AUROC is unaffected by class ratio, but a
   future *statistical* (non-degenerate) feature would have ~0.013 of sampling noise there, which is
   inside the declared 0.05.
4. ✅ **Resolved, and worth recording as process:** `audit_query_sets` used to append one schema
   violation per offending **record**, so a future 6,292-record C set with a bad term-key order
   would have raised one message with 6,292 identical lines in it. I wrote the fix, **reverted it**
   because a full suite was already running and I would not report a green suite for a HEAD it had
   not tested, and then **landed it as `dd0c842` and re-ran the whole suite from scratch** once the
   machine was free. ⛔ The reported 2275/0 is at that final HEAD.
5. ⚠ **`a7_n_requests` defaults to `|C|`, not to `|A|`** in the shipped file. A consumer that wants
   the literal `|A7| = |A|` for its own A must call `build_a7_redundant` with its own rows (the same
   function, same rule) rather than reading the file. Stated in the config docstring and §3.

## 9. Git footprint
- **Branch:** `agent/experiment-engineer/gen-c-set-a7`, in a fresh worktree `../CHLU-gen-a7`.
- **Base:** `main @ 1bcfcb9` (⚠ task said `3faef27`; `main` had moved, `3faef27` is its ancestor).
  `main` had **not** moved again at the end (`merge-base --is-ancestor main <branch>` → true), so
  **no rebase was needed** and none was performed. ⛔ `origin/main` untouched, nothing pushed.
- **Commits (6, oldest first):**
  | hash | subject |
  |---|---|
  | `c6a9bfe` | A7 + the query-set registry + the no-model detector |
  | `74070df` | emit A7 and the query-set registry beside the C set |
  | `465ac89` | a missing C set must RAISE, not log a warning and continue |
  | `23b753d` | make the no-model detector a suite test, not a convention |
  | `3737757` | name the residual A7 cannot close, and measure it |
  | `dd0c842` | report a schema fault once per set, not once per record |
- **Files touched (exactly the five my task names):**
  | file | ± | what |
  |---|---|---|
  | `chlu/data/gen_splits.py` | **+674 / −0** | A7 builder + validator, `build_base_requests`, `select_a7_rows`, the registry, the detector. ⛔ **purely additive** — `build_c_invalid` and `validate_c_invalid` are byte-for-byte unchanged |
  | `chlu/config.py` | +14 / −0 | `emit_a7`, `a7_n_requests`, `a7_seed` **inside the existing `GenDataConfig`** |
  | `chlu/cli/data_cmd.py` | +34 / −2 | `_build_gen_artifacts` only |
  | `chlu/experiments/exp_gen_store.py` | +27 / −3 | ⛔ `load_c_records` **only**; nothing else in that file |
  | `tests/test_gen_c_set_a7.py` | new, +446, 24 tests | — |
- ⛔ **Nothing under `chlu/core/`, `chlu/eval/`, `scripts/csf3/`.**
- **Worktree/ref safety (protocol §3.2):** the shared checkout was on **another agent's branch**
  (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`) with its files present, so I never
  touched it; all work is in `../CHLU-gen-a7`. ⭐ The branch ref was verified **from the main repo**
  (`git log --oneline main..agent/experiment-engineer/gen-c-set-a7` → the 5 hashes above) and the
  worktree is **left in place** for review, not removed.
- **Untracked artifacts written (all under `.claude/`, gitignored):**
  `.claude/data/gen/{shapes3d,dsprites,shapes3d/recomb2element_wide}/{a7_redundant.json,query_sets.json}`
  (added with `cp -n`, ⛔ nothing overwritten), `.claude/outputs/gen-c-set-a7/{PREREG.md,detector_report.json}`,
  `.claude/scratch/gen-c-set-a7/`.

## Proposed handover updates (for the Hub)

**§3 (CLI/config) — new `gen_data` knobs**
- `gen_data.emit_a7: bool = True` — emit `a7_redundant.json` beside `c_invalid.json`.
- `gen_data.a7_n_requests: Optional[int] = None` → `|C|`; `gen_data.a7_seed: int = 0`.
- `chlu data shapes3d|dsprites` now also writes **`a7_redundant.json`** and **`query_sets.json`**
  (the query-set registry) into the dataset directory. ⛔ Pre-existing artifacts are bit-identical.

**§7 Known Issues — one entry to CLOSE, one to ADD**
- ⭐ **CLOSE (partially): "the C-set term-count leak".** A7 ships, the detector is a suite test, and
  the realised no-model AUROC against the declared control is **0.5000** (was **1.0000** dup /
  **0.7500** pooled) on all three registries. ⚠ **Do not close it entirely until `exp_gen_baselines`
  and `exp_gen_certificate` read the artifact** — reconciliation item 1.
- ⭐ **CLOSE: "`gen_data.data_root` silently skips the C set".** `load_c_records` now raises; three
  tests, including the negative control that a correct root still loads.
- ⚠ **ADD:** *"Both C kinds are decidable in factor space by a syntactic rule — `out_of_range` by a
  grid lookup, `duplicate_factor` by value equality. A7 removes the **structural** giveaway (length
  and shape), not the definitional one, and nothing can remove the definitional one. The cell's
  claim is 'from the store's geometry, zero invalid supervision, beats byte-matched competitors on
  the controlled metric' — ⛔ never 'undetectable without a model'."*
- ⚠ **ADD:** *"Every query-set directory now needs a `query_sets.json` beside its `c_invalid.json`;
  `tests/test_gen_c_set_a7.py` fails if one is missing, and a new invalid **kind** fails unless it
  declares a structural control. Rebuild with `chlu data <dataset>` after adding a query set."*

**§3 git discipline — a datum for the ledger**
- ⚠ Five full suites were running concurrently on this machine. `pkill -f "pytest -q --no-cov"`
  matched **only** my own process (the others' command lines interleave `-p no:cacheprovider`
  differently), but that was luck, not design. ⛔ **A spoke killing its own background suite should
  match on its PID, not on a command-line pattern.**
