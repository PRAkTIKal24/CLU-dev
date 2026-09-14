# gen-wide-split — experiment-engineer report

**Task + acceptance criterion:** Define a labelled SECONDARY hold-out (`recomb2element_wide`) for Shapes3D — extra held-out cells spanning the photometric contrast range, Montero's 72 untouched — such that the absence assertion **fires on the positive control**.

**Status: done.** The positive control fires, pooled and **per cell** (13/13). Montero's condition re-emits **bit-identically**.

**Downstream reconciliation list: NO.** One item needs an owner and it is a *scoping* item, not a correction: **the launcher's `CONDITION` axis** (§6 below) — I did not edit `scripts/csf3/job_gpu_gen.sh`, as instructed.

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — instrument.* ⛔ No store, no arm, no encoder, no CLU number, no performance claim appears below.
- **Laundering control:** N/A (nothing is measured about a model).
- **Falsifies:** N/A for a dial. My own falsifier, stated in the task: *the absence assertion passes on the wide split
  but does not fire on a corrupted one.* It fired — evidence in §3.

---

## 0. Pre-registration
`.claude/outputs/gen-wide-split/PREREG.md`, written **before the builder was first executed** (P1–P6: per-cell count, pooled counts, the contrast ladder, the assertion, Montero bit-identity, |C|). **All six predictions held exactly** — including the contrast ladder to float precision. The ladder was derived by hand from the grid arithmetic, not fitted.

**Timestamp evidence:** `PREREG.md` mtime `Sep 10 09:52`; the first execution of the wide builder was `09:53` (the `--condition` CLI runs are logged at 09:57–10:01 in `.claude/scratch/gen-wide-split/`). Verified by mtime, not by claim.

⚠ Honest note on what that buys: P1/P2/P6 are counting arithmetic and P3 is a closed-form mean over a 2×2×2 enumeration — a pre-registration that survives here is a **correctness check on my implementation**, not a scientific finding. P4 (the assertion firing) is the one that could genuinely have gone either way.

---

## 1. What I built

| File | Change | What |
|---|---|---|
| `chlu/data/gen_splits.py` | **+167, 0 deleted** | `FactorCell`, `cells_test_predicate`, `assert_cells_disjoint`, `build_cell_split` |
| `chlu/data/shapes3d.py` | **+290, 2 replaced** | the whole `recomb2element_wide` builder + `condition` dispatch in `load_shapes3d` |
| `chlu/config.py` | **+13, 0 deleted** | 3 fields **inside the existing `GenDataConfig`** |
| `chlu/cli/data_cmd.py` | **+32, 2 replaced** | one flag `--condition`, + its own output subdirectory |
| `tests/test_gen_wide_split.py` | new, 18 tests | incl. the positive control, per cell |

**The design (the rule, in one paragraph).** A held-out *cell* is a conjunction over all six factors. Montero's cell is exactly one such conjunction. The wide condition is a **union of cells of the same kind**: a 2-value band on each of `floor_hue`/`wall_hue`/`object_hue`, a 3-value corner of `scale`, exactly one `shape`, a 3-value corner of `orientation` ⇒ **72 rows per cell, the same size and shape as Montero's**. ⭐ The widths 2/3/1/3 are **derived from Montero's own cell** (`|{hue≥0.8}|`, `|{scale≥1.1}|`, `|{orientation>20}|`) rather than chosen, so "structurally the same kind" is enforced by construction, not by my judgement. Only the *placement* moves.

**Why placement, not volume** (R15's finding used as the design input, not rediscovered): Montero's `all three hues ≥ 0.8` puts floor, wall and object in one band, and the measured `scale` deficit on B (0.708) is indistinguishable from the same deficit on photometrically matched *seen* data (0.703). So the wide cells walk the **object-vs-background hue offset** from 0 to 5 grid steps, and each cell carries a **measured** contrast label. Downstream analysis can then condition on rendering contrast instead of confounding it with composition.

**The rule verbatim** is stored in the emitted `wide_cells.json` as `rule` (a 1,100-char string, asserted equal to `s3d.WIDE_RULE` by test), including the RNG (`numpy.random.default_rng(seed)`, one draw per candidate, in cell order), the rejection-and-redraw policy, and the circular-hue convention.

### `chlu/config.py` — exact hunk (shared file, `gen-store-grouping` concurrent)
**Lines 3294–3306 inclusive, 13 lines, pure insertion**, at the **end of the existing `GenDataConfig` body** (which previously ended at line 3293 with `c_invalid_seed`). `GenBridgeConfig` and everything below it are byte-identical. Fields added: `wide_n_cells: int = 12`, `wide_seed: int = 0`, `wide_include_montero: bool = True`. ⛔ I touched **nothing** below `GenDataConfig`, and did not add a new dataclass.
No new `load_config`/`save_config` sites were needed — `GenDataConfig` is already wired at all three house sites, so the new fields round-trip automatically (verified: `save_config`→`load_config` returns `recomb2element_wide / 24 / 7 / False`).

---

## 2. Realised counts and contrast labels — MEASURED

⛔ Nothing was adjusted to hit a target. `n_cells=12`, `seed=0`, `include_montero=True` (the shipped defaults).

| cell | shape | floor/wall hue band | object hue band | scale corner | orient corner | n (realised) | **contrast** |
|---|---|---|---|---|---|---|---|
| `montero` | 1 (cylinder) | {0.8,0.9} | {0.8,0.9} | high (≥1.1) | high (>20) | 72 | **0.050** |
| `wide-000` | 0 | {0.8,0.9} | {0.8,0.9} | high | high | 72 | **0.050** |
| `wide-001` | 1 | {0.6,0.7} | {0.7,0.8} | high | high | 72 | **0.100** |
| `wide-002` | 2 | {0.5,0.6} | {0.7,0.8} | low | high | 72 | **0.200** |
| `wide-003` | 3 | {0.2,0.3} | {0.5,0.6} | low | low | 72 | **0.300** |
| `wide-004` | 0 | {0.3,0.4} | {0.7,0.8} | high | low | 72 | **0.400** |
| `wide-005` | 1 | {0.0,0.1} | {0.5,0.6} | high | low | 72 | **0.450** |
| `wide-006` | 2 | {0.0,0.1} | {0.0,0.1} | low | high | 72 | **0.050** |
| `wide-007` | 3 | {0.0,0.1} | {0.1,0.2} | low | high | 72 | **0.100** |
| `wide-008` | 0 | {0.1,0.2} | {0.3,0.4} | high | high | 72 | **0.200** |
| `wide-009` | 1 | {0.8,0.9} | {0.1,0.2} | high | low | 72 | **0.300** |
| `wide-010` | 2 | {0.6,0.7} | {0.0,0.1} | low | low | 72 | **0.400** |
| `wide-011` | 3 | {0.0,0.9}† | {0.4,0.5} | low | low | 72 | **0.450** |

† not a bug: hue is circular, so the band `{index 9, index 0}` = `{0.9, 0.0}` is two *adjacent* hues. Sorted for display.

- **Pooled realised: `n_test = 936`, `n_train = 479,064`, `n_total = 480,000`.** Cells are pairwise disjoint (`assert_cells_disjoint` passes; `Σ per-cell = pooled`), and **0 candidates were rejected** at seed 0 — every first draw was admissible.
- **Contrast label** = mean over the cell's realised rows of `½[d(object,floor) + d(object,wall)]`, `d` = circular hue distance (period **1.0, derived from the grid**, not hard-coded). A measured quantity from the factor values; `min`/`max`/`nearest-surface` variants are also emitted per cell.
- **Span: 0.050 → 0.450, against a theoretical maximum of 0.500 — 80% of the range, 90% of the maximum.** ⛔ Acceptance item 2's falsifier (all cells clustered in Montero's corner) does **not** fire: 8 of 13 cells sit >0.05 away from Montero's label, and 6 sit at ≥0.30.
- ⭐ **`wide-000` is deliberately Montero's photometric corner with a different shape** (cube, same hues, same scale/orientation corners) — i.e. R15's `A_matched` control, but now **held out**. R16 made the matched-seen control a permanent companion row; this gives the wide condition a **matched-contrast companion that is itself held out**, so the "is it rendering or composition?" question can be asked *within* the held-out set.
- **Shape balance:** the 12 additional cells are exactly 3 per shape ⇒ 216 rows per shape. Pooled with Montero's cylinder cell: 216/288/216/216.

---

## 3. ⭐ The positive control — the criterion this spoke lives on

Assertion route unchanged from `gen-data-splits`: each train row's factors are reconstructed **through the grid bijection** and the condition's own predicate (here the OR over the 13 cells) is re-evaluated on the reconstruction — the cached mask is never re-read.

```
[wide] assertion PASSED over 479064 train combinations   (== n_train, so not silently empty)
[wide] POSITIVE CONTROL FIRED:
  shapes3d/recomb2element_wide: 1 train row(s) satisfy the held-out predicate; first is row 479969
  (factor indices [9, 9, 9, 7, 1, 14], values [0.9, 0.9, 0.9, 1.25, 1.0, 30.0])
```

⭐ **And once per cell.** `test_wide_positive_control_fires_for_every_cell` injects one row from **each** of the 13 cells in turn and requires `HoldoutViolation` to name that row. A single-cell control would leave 12 cells unverified — an OR-predicate whose 13th disjunct is wrong passes a one-row check. **13/13 fire.**

---

## 4. ⛔ Montero's condition is provably unchanged (acceptance item 4 — checked first)

| artifact | sha256 at base `fcdca43` | after my change |
|---|---|---|
| `shapes3d/factor_index.npz` | `5f1fce25c40cceae6de2d00fd2145c4cdc73dcc38cee7f6545cbcaea3ce78f49` | **identical** |
| `shapes3d/c_invalid.json` | `b886da239e97b19e9001a88ce04fded329dda4884d301f78c3097ea6cc66ba49` | **identical** |
| `n_test` / `n_train` | 72 / 479,928 | **72 / 479,928** |

Three independent confirmations: (a) `uv`-venv CLI re-run of `chlu data shapes3d` (no flag) into a clean root reproduced both checksums exactly; (b) running the **wide** condition afterwards left both files byte-identical, because the wide condition writes into its own subdirectory `shapes3d/recomb2element_wide/`; (c) `test_montero_split_is_unchanged_by_the_wide_condition` **pins both checksums in the test suite**, so any future edit that perturbs the headline split goes red.
Also asserted: `montero_cell(grid).rows()` reproduces `montero_test_mask()` **exactly** (the cell representation of Montero's split is checked, not assumed).

⭐ **Filesystem evidence, independent of any checksum I computed:** the shipped `shapes3d/factor_index.npz` and `shapes3d/c_invalid.json` still carry their **original mtimes of `Sep 9 14:31`** (written by `gen-data-splits`/R14.2) after everything this spoke ran. They were never even *opened for writing*.

---

## 5. ⛔ THE COST, NOT HIDDEN: the frozen models have SEEN these rows

Measured, not asserted rhetorically:

| quantity | value |
|---|---|
| wide `|B|` | **936** |
| of those, rows **inside Montero's TRAIN split** — i.e. **the shipped frozen encoder/decoder trained on them** | **864** |
| of those, rows already held out by Montero | 72 |
| wide train ⊂ Montero train | **True** (the wide condition only ever *removes* 864 more rows) |

⛔ **The shipped encoder/decoder (and any store built on their cache) cannot be scored on this split.** The wide condition is a **second full configuration**: encoder → decoder → store → cell must be re-run against it. Nothing in this spoke re-ran anything, and nothing here should be read as implying the existing artifacts can be evaluated on these rows. The warning is carried **inside the emitted `wide_cells.json`** (`frozen_artifact_warning`), not only in this report, so a downstream consumer meets it at the data.

**Why `include_montero=True` is the default:** the wide hold-out is then a strict **superset** of Montero's, so a wide-trained stack has never seen Montero's 72 either. ⛔ That does **not** license pooling — Montero's headline numbers come from the Montero configuration, and the wide condition is reported separately, per the task. It only means the wide run is not *disqualified* from ever touching Montero's cell.
The 12 additional cells are **bit-identical whether or not the flag is set** (Montero's rows are reserved from the draws either way), so the flag moves no cell.

**Statistical-power arithmetic (derived, not measured — this is why R17 made the spoke mandatory).** Binomial SE at p≈0.5: `0.5/√72 = 0.0589` per seed for Montero's cell vs `0.5/√936 = 0.0163` for the wide pooled set — **3.61× tighter (=√13)**; over 3 seeds, 0.0340 → 0.0094. Per-shape (n=216) it is 0.0340 per seed. ⚠ This is idealised arithmetic on independent Bernoulli draws; correlations across the 72 rows of one cell will inflate it, so treat it as an upper bound on the gain.

---

## 6. ⭐ The launcher's CONDITION axis — specified, NOT edited

⛔ `scripts/csf3/job_gpu_gen.sh` is untouched (`git diff` proves it). Read-only inspection says the axis needs **five edits, and one of them is load-bearing**:

1. ⭐ **THE LOAD-BEARING ONE — the workspace path (§2, line ~139).** Today `WORKSPACE="$GEN_ROOT/workspaces/${DATASET}_${PRETEXT}"`. Two conditions of the same dataset/pretext would write their **frozen encoder/decoder/store into the same run dirs** and silently overwrite each other — which is precisely the leak §5 exists to prevent. Back-compatible form (keeps every existing Montero path valid):
   ```sh
   WORKSPACE="$GEN_ROOT/workspaces/${DATASET}_${PRETEXT}"
   [ "$CONDITION" = "recomb2element" ] || WORKSPACE="${WORKSPACE}_${CONDITION}"
   ```
   ⭐ Free bonus: §7's upstream-manifest gate resolves through `$WORKSPACE`, so this **mechanically prevents** a wide `store` stage from consuming a Montero-trained encoder. No extra check needed.
2. **Axis declaration + validation (§1, after the `STAGE` case at line ~98).**
   ```sh
   DEF_CONDITION="recomb2element"
   CONDITION="${CONDITION:-$DEF_CONDITION}"
   case "$CONDITION" in recomb2element|recomb2element_wide) ;;
       *) die "CONDITION='$CONDITION' is not one of: recomb2element recomb2element_wide" ;; esac
   [ "$CONDITION" != "recomb2element_wide" ] || [ "$DATASET" = "shapes3d" ] \
       || die "CONDITION=recomb2element_wide is Shapes3D-only"
   ```
   ⚠ **A defaulting decision the Hub must take, not me.** The script's own policy (lines 74–79) is *"a silently-defaulted axis is a silently-decided result"*, which argues for `[ -n "${CONDITION:-}" ] || die` like DATASET/PRETEXT/STAGE. **I recommend defaulting to `recomb2element`** instead, because (a) it keeps every submit line already in the ledger valid, (b) edit 1 puts the condition in the path so it can never be *silent*, and (c) edit 4 records any non-default in the manifest. If the Hub prefers strictness, `die`-on-unset is a one-line swap and costs only the existing docs.
3. **PLAN output (§3, lines ~148–153):** add `CONDITION=${CONDITION}` to the echoed `sbatch --export=...` string and to the `=== gen launch plan ===` header. (The `--export` value must stay one shell word — it already is.)
4. **Provenance (§6):** `note_flag CONDITION "$CONDITION" "$DEF_CONDITION"`, plus `GEN_MF_CONDITION="$CONDITION"` in `write_manifest`'s env block and `"condition": e("GEN_MF_CONDITION")` inside the manifest's **`axes`** dict. ⭐ Without the `axes` entry, a wide-run artifact is indistinguishable from a Montero one at analysis time — this is the edit that must not be skipped if edit 2 lands as "defaulted".
5. **Payload argv (§9, line ~434):** add `--condition "$CONDITION"` to `"$PY_BIN" -u -m "$MODULE" …`. ⇒ every `exp_gen_*` stage module must accept `--condition` and forward it to `chlu.data.shapes3d.load_shapes3d(condition=…, wide_n_cells=…, wide_seed=…, wide_include_montero=…)`, whose config side is **already shipped** here (`gen_data.condition`, `gen_data.wide_*`). Header comment at line 12 also needs the path updated.

⛔ Not needed: `ARMS`, `SEEDS`, the `-a` arithmetic and the concurrency cap are all orthogonal to the condition — `N_TASKS` stays `ARMS × SEEDS` and the condition is a *whole-sweep* axis, one submission per condition, exactly like `STAGE`.

---

## 7. Artifacts emitted (gitignored, under `.claude/data/gen/shapes3d/recomb2element_wide/`)

| File | Bytes | sha256 |
|---|---|---|
| `factor_index.npz` | 3,844,214 | `e6303358dc020f847365aa4736b4ecb062b157b8cc2f97a17e81a77f790deedb` |
| `c_invalid.json` (936 records) | 917,443 | `34cfcf2ee46c7131d0835b8dad227f802a148207f67a21ccbb08acd197ae89cd` |
| `wide_cells.json` (rule + 13 cells) | 15,950 | `4ed655e957f273b799e544477a7a7a1759860b9d0569aa706b97dd2afada9b4d` |

**C set (acceptance item 5):** built by the *existing* derivation — `n_per_kind = ceil(n_test/2) = 468` per kind ⇒ **|C| = 936 = |B|**, satisfying R14.2 for this condition. Validated by `validate_c_invalid` (every record re-derived as unsatisfiable from the grid). ⛔ It is a **different** C set from Montero's condition's (different `n_per_kind`), which is correct — R14.2 sizes C to *its own* B — and it lives in its own directory, so Montero's `b886da23…` C set is untouched.

⚠ Reproducibility note: these three files are **rebuildable** with `chlu data shapes3d --condition recomb2element_wide` at defaults; the checksums above are the anchor.

---

## 8. FLAG PROVENANCE (protocol §5)

| item | value |
|---|---|
| commit | `92aaa2b` (branch tip; base `main @ fcdca43`) |
| dataset | Shapes3D `3dshapes.h5`, sha256 `0a0f6ed9…`, 267,573,662 B |
| condition | `recomb2element_wide` / variant `leave1out_multicell` |
| `gen_data.wide_n_cells` | **12** (default) |
| `gen_data.wide_seed` | **0** (default) |
| `gen_data.wide_include_montero` | **True** (default) |
| `gen_data.c_invalid_n_per_kind` | `None` ⇒ derived 468 |
| `gen_data.c_invalid_seed` | 0 |
| `gen_data.holdout_assert` | True |
| split seed | ⛔ **none** — the split's cell *placement* is seeded (`wide_seed`), the row→factor mapping and the predicate are deterministic. Montero's split remains seedless. |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, Python 3.11.13, numpy 2.4.1, h5py 3.16.0; `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`, `PYTHONPATH=<worktree>` |
| ⛔ JAX | **not used anywhere in this spoke** — no model, no PRNGKey, no `eqx`. All randomness is `numpy.random.default_rng`. |

---

## 9. How I verified (commands + observed output)

```
# targeted
pytest tests/test_gen_wide_split.py  -q -p no:randomly --no-cov  ->  18 passed in 11.70s
pytest tests/test_gen_data_splits.py -q -p no:randomly --no-cov  ->  24 passed in 16.93s   (no regression)

# CLI, Montero condition, clean root (symlinked h5)      rc=0
  n_total=480000  n_train=479928  n_test=72
  ✓ hold-out assertion passed over 479928 train combinations
  index:     …/factor_index.npz  sha256=5f1fce25…      <- IDENTICAL to base
  c_invalid: …/c_invalid.json    sha256=b886da23…      <- IDENTICAL to base  (72 records)

# CLI, wide condition                                    rc=0
  n_total=480000  n_train=479064  n_test=936
  ✓ hold-out assertion passed over 479064 train combinations
  index:     …/recomb2element_wide/factor_index.npz  sha256=e6303358…
  c_invalid: …/recomb2element_wide/c_invalid.json    sha256=34cfcf2e…  (936 records)
  wide_cells:…/recomb2element_wide/wide_cells.json   sha256=4ed655e9…
  cells: 13 (contrast labels 0.050, 0.050, 0.100, 0.200, 0.300, 0.400, 0.450,
                              0.050, 0.100, 0.200, 0.300, 0.400, 0.450)

# determinism / seed sensitivity (real labels)
  build_wide_split(seed=0) twice -> test_rows identical: True
  build_wide_split(seed=1)       -> test_rows differ:    True

# config round-trip (three-site rule)
  defaults:  recomb2element 12 0 True
  roundtrip: recomb2element_wide 24 7 False

# lint / format
  ruff check  (5 touched files)                 -> All checks passed!
  ruff format (new test file + my shapes3d hunks) -> 2 files reformatted, then clean
  git diff -U0 | grep '^-[^-]' -> exactly 4 replaced lines, all intended
```

**Full suite (main venv, `-p no:randomly --no-cov`, both runs):**
- **Before** — detached worktree `../CHLU-gen-wide-base @ fcdca43`: **2010 passed, 41 warnings in 15341.01s (4:15:41)**, 0 failed, 0 skipped
- **After** — `agent/experiment-engineer/gen-wide-split @ 92aaa2b`: **2028 passed, 41 warnings in 15899.54s (4:24:59)**, 0 failed, 0 skipped
- **Delta:** **+18 = exactly my 18 new tests.** No pre-existing test changed state. ⭐ Cross-check: the *before* count reproduces R15's reported 2010 at `fcdca43` exactly (independent run, main venv, so no h5py-pruning skew: all 3 of my real-data tests and all of `gen-data-splits`' 8 executed rather than skipping).

⚠ The first "after" run was **discarded and relaunched**: I edited two files (§5's reservation change) while it was in flight, so its pass count would not have described the tree it was labelled with. The number above is from a run started **after** the final commit.

---

## 10. Risks / open questions

1. **`n_cells = 12` is my choice, and it is a power/effort trade, not a derivation.** 12 gives 936 rows, 3 cells per shape, and each contrast rung twice. If the Hub wants more, it is one config value — but ⛔ **every increase is paid in a full re-run of the whole stack**, since the split defines the training set. Deciding this *before* the wide encoder is trained is much cheaper than after.
2. **The contrast ladder is a hue-offset ladder, not a luminance one.** Shapes3D's floor/wall/object share a saturation/value setting, so hue offset is the available contrast axis. A cell with 0.45 hue offset is not guaranteed to be *perceptually* high-contrast for the encoder; the label is the honest measurable, not a rendering measurement. If someone wants a pixel-space contrast, it needs the images (and belongs to whoever runs the wide encoder).
3. **`orientation` corner is not balanced against contrast** at `n_cells=12` (contrast steps 0 and 3 both land on the same orientation corner twice). `scale` corner and `shape` **are** balanced. Documented; not worth a rule change, but a reader must not read an orientation effect off this split.
4. **dSprites has no wide condition.** Out of scope (R6: the A/B/C cell is Shapes3D only), and the launcher spec in §6 `die`s on the combination rather than allowing it silently.
5. **`.claude/data/gen/shapes3d/recomb2element_wide/` (4.8 MB) is gitignored.** §7's table is the backup; the files rebuild from one command.
6. ⚠ **`h5py` is still undeclared** (R12 / §7 Known-issue candidate). Unchanged by me — I used the main venv where it is installed. The 3 real-data tests in my new file `importorskip` it, so a `uv sync` that prunes h5py turns them into **skips, not failures**. Same silent-skip hazard R12 already owns; I add 3 tests to its blast radius.

---

## Git footprint

- **Branch:** `agent/experiment-engineer/gen-wide-split`, base **`main @ fcdca43`** (as named in the task).
- **Worktree:** `../CHLU-gen-wide` (fresh, created by me). A second **detached** worktree `../CHLU-gen-wide-base @ fcdca43` was used only for the baseline suite count. ⛔ I did not touch `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge`, nor the shared main checkout (which is on another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring @ 7fcef50`, left untouched and clean).

| Hash | Subject |
|---|---|
| `1d97edb` | `[experiment-engineer] add FactorCell: multi-cell hold-out conditions over a factor grid` |
| `201f1f2` | `[experiment-engineer] Shapes3D recomb2element_wide: the widened SECONDARY hold-out (R17)` |
| `f00590b` | `[experiment-engineer] wire \`chlu data shapes3d --condition\`` |
| `6e9e656` | `[experiment-engineer] test the wide split by making its assertion fire, cell by cell` |
| `92aaa2b` | `[experiment-engineer] reserve Montero's rows from the wide draws regardless of the flag` |

- **Files touched:** `chlu/data/gen_splits.py`, `chlu/data/shapes3d.py`, `chlu/config.py`, `chlu/cli/data_cmd.py`, `tests/test_gen_wide_split.py` (new). **No other tracked file** — in particular ⛔ **nothing under `chlu/core/`, `chlu/eval/` or `scripts/csf3/`**, and nothing below `GenDataConfig` in `chlu/config.py`.
- **Rebase:** `git rebase main` (local `main` = `fcdca43`, my named base) → *"Current branch … is up to date"*, a no-op. ⛔ Did **not** rebase onto `origin/main` (stale at `40c2f31`, §7.21). No conflicts, so no cross-agent hunk to resolve — in particular `chlu/config.py` did not conflict with `gen-store-grouping`, whose insertion point is below mine.
- **Ref verified from the MAIN repo before any worktree removal:** yes, twice. `git log --oneline main..agent/experiment-engineer/gen-wide-split` from `/Users/user/Desktop/CHLU` listed all 5 commits and `git diff --stat main..` showed the 5 files / 842 insertions / 4 deletions **before** removal; re-verified **after** removing both worktrees at `92aaa2b56972cca50d5cafb73b9dad8edc92d964`, 5 commits still reachable. The shared main checkout was never edited and is still clean on `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring @ 7fcef50`. ⛔ `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` untouched; concurrent worktrees `CHLU-gen-store` (`gen-store-grouping`) and `CHLU-gen-declift` (`gen-decoder-lift`) appeared during my run and were left alone.
- **Not pushed, no PR** — branch left for review.

## Proposed handover updates (for the Hub)

- **New condition `recomb2element_wide` (Shapes3D only).** 13 cells / **936** held out of 480,000 (`n_train = 479,064`) at defaults. Montero's `recomb2element` (72) is **unchanged and remains the headline**; the two are **separate configurations** and ⛔ **must never be pooled**.
- **⛔ New standing trap for §7 Known Issues:** *the frozen encoder/decoder/store from the Montero configuration have SEEN 864 of the wide condition's 936 held-out rows. They must not be scored on `recomb2element_wide`; it requires a full re-run of encoder→decoder→store→cell.* This is the single most likely mistake a later spoke will make with this split.
- **New config fields** (`gen_data.wide_n_cells=12`, `wide_seed=0`, `wide_include_montero=True`) and **new CLI flag** `chlu data shapes3d --condition {recomb2element,recomb2element_wide}`. Wide artifacts land in `<root>/shapes3d/recomb2element_wide/`, so Montero's are structurally protected from overwrite.
- **§3 CLI/config:** `chlu data shapes3d` now takes `--condition`; default behaviour is unchanged.
- **NEEDS AN OWNER (scoping, not correction):** the `CONDITION` axis in `scripts/csf3/job_gpu_gen.sh` — §6 above specifies all five edits, incl. one Hub decision (default vs `die`-on-unset). ⛔ **Edit 1 (the workspace path) is load-bearing**: without it the two conditions overwrite each other's frozen artifacts.
- **R15/R16 companion-control note:** the wide condition contains `wide-000`, a **held-out** cell in Montero's exact photometric corner with a different shape — i.e. a matched-contrast control that is itself unseen. Worth naming in the phase's table design.
