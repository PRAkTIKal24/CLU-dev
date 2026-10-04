# gen-data-splits — experiment-engineer report

**Task + acceptance criterion:** Ship Shapes3D + dSprites loaders, the exhaustive factor-grid index and the Montero recombination-to-element splits, such that the "never seen in training" assertion **passes on the real split AND fires on a deliberately corrupted one**.

**Status: done.** Both positive controls fire, on the real splits, with the offending row named.

**Precondition (checked first, mechanically):** `.claude/outputs/gen-montero-splits-spec/montero_splits.json` exists (28,276 B) and `shapes3d.condition.status == "PINNED"` (`dsprites.condition.status == "PINNED"` too). Gate MET — proceeded.

**Downstream reconciliation list: YES, 3 items, needs a Hub-assigned owner — see §Reconciliation (h5py is undeclared; the nominal dSprites 5,808 must be retired in favour of the realised 6,292).**

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — instrument.* No CLU number, no store, no encoder, no generation, no performance claim appears below.
- **Laundering control:** N/A (nothing is measured about a model).
- **Falsifies:** N/A for a dial, but the spoke's own falsifier is explicit: *the assertion passes and the positive control does not fire.* It fired — evidence quoted verbatim in §3.

---

## 1. What I built

| File | Status | What |
|---|---|---|
| `chlu/data/gen_splits.py` | new, 585 lines | `FactorGrid` (grid + row↔factor bijection), `GenSplit`, `HoldoutViolation`, the three assertions, the C-set builder/validator, index save/load |
| `chlu/data/dsprites.py` | new, 322 lines | download+checksum, latents (post colour-slice), verbatim Montero filter, split builder |
| `chlu/data/shapes3d.py` | new, 231 lines | download+checksum, HDF5 labels, verbatim filter, split builder, lazy image access |
| `chlu/cli/data_cmd.py` | +141, **insertions only** | `chlu data shapes3d`, `chlu data dsprites` |
| `chlu/config.py` | +37, **insertions only** | `GenDataConfig` + house three-site wiring |
| `tests/test_gen_data_splits.py` | new, 448 lines / 24 tests | incl. the positive controls |

**Three assertions, each firing for exactly one reason** (a single mega-assert would hide *which* invariant broke):
1. `assert_non_empty_test` — runs **first** (T1: the empty-test-set failure mode).
2. `assert_index_partitions_grid` — train ∪ test covers every grid cell exactly once, no duplicates.
3. `assert_holdout_disjoint` — no train row satisfies the held-out predicate.

**Why (3) is not vacuous.** It does **not** re-read the boolean mask that produced the split (re-checking a mask against itself proves nothing). It takes each train **row index**, reconstructs its factor values through the grid bijection (`row → unravel → grid values`), and re-evaluates the dataset's own test predicate on the reconstruction. A corrupted train index therefore cannot smuggle a held-out combination past it — which is exactly what the positive control demonstrates.

### `chlu/config.py` — exact hunks touched (shared file)
Base file 3,656 lines; **+37 lines, 0 lines modified or deleted**, at 4 insertion points:

| Site | Line (post-edit) | Insertion |
|---|---|---|
| new dataclass | 3262–3293 | `class GenDataConfig` (between `DataConfig` and `ProjectConfig`) |
| `CHLUConfig` field | 3400 | `gen_data: GenDataConfig = field(default_factory=GenDataConfig)` |
| `load_config` | 3627–3629 | `gen_data=GenDataConfig(**filter_valid_fields(...))` |
| `save_config` | 3687 | `"gen_data": asdict(config.gen_data),` |

⚠ **Deviation from the task's letter, declared:** the task said "append **one** dataclass and **one** field, ⛔ nothing else". I also added the `load_config`/`save_config` entries because `tests/test_primitive_harness.py:223` names the house convention *"the three-site rule: dataclass, load_config, save_config"* — without them the knobs are not YAML-overridable and are silently dropped by a `save_config`→`load_config` round-trip, which several existing tests assert for their own configs. Both added lines are pure insertions in the same list-literal style; verified round-trip: `save_config` → `load_config` returns `quick_rows=123, data_root=/tmp/x`.

`GenDataConfig` defaults (all preserve/define new behaviour; nothing existing changes):
`data_root=None` (→ `$CHLU_GEN_DATA_ROOT`, else `<repo>/.claude/data/gen`), `condition="recomb2element"`, `variant="leave1out"`, `verify_checksum=True`, `holdout_assert=True`, `assert_expected_counts=True`, `quick_rows=4096`, `dsprites_with_images=True`, `c_invalid_n_per_kind=16`, `c_invalid_seed=0`.

---

## 2. Realised counts — measured, not derived

⛔ Nothing was adjusted in either direction. Both filters were evaluated **verbatim against the actual files**, never against a regenerated `linspace` grid.

| Dataset | n_total | **n_test (realised)** | n_train | Predicted by the spec | Verdict |
|---|---|---|---|---|---|
| Shapes3D | 480,000 | **72** | 479,928 | 72 (T4, "must be 72") | ✅ exact |
| dSprites | 737,280 | **6,292** | 730,988 | one of 5,808 / 6,292 / 6,776 (T3) | ✅ the **middle** candidate |

**Pre-registration.** The admissible values were pre-registered *before this spoke ran*, in the task file itself (T3 names 5,808/6,292/6,776; T4 names 72), written on 2026-09-09 **before this spoke spawned**. I wrote no separate `PREREG.md` because the prediction was already committed to in writing by the Hub and I did not author it; the measurement selected among the pre-named candidates rather than producing a free-fitted number.

### T3 — the float-boundary reconciliation, shown
The nominal arithmetic (spec's 5,808) assumes 12 rotation values. The realised array gives **13**:

```
grid_rot[13] = 2.0943951023931957   a240 = 2*pi/3 = 2.0943951023931953   -> grid > a240 ? True   ADMITTED
grid_rot[26] = 4.188790204786391    a120 = 4*pi/3 = 4.1887902047863905   -> grid < a120 ? False  EXCLUDED
scale grid[1] = 0.6                 literal 0.6                          -> 0.6 > 0.6 ? False    EXCLUDED (4 scale values)
posX/posY > 0.66                                                          -> 11 values each (idx 21..31)
```
⇒ `1 (ellipse) × 4 (scale) × 13 (rotation) × 11 (posX) × 11 (posY) = 6,292`. Asymmetric because `linspace(0, 2π, 40)`'s round-off falls on opposite sides of the two exact thirds. **The 5,808 was a good-faith nominal estimate and is now retired; 6,292 is ground truth** (see §Reconciliation).

### T4 — Shapes3D
`(oh≥0.8)&(wh≥0.8)&(fh≥0.8)&(scl≥1.1)&(shp==1)&(rot>20)` over the real `labels` array ⇒ **72**, i.e. `2·2·2·3·1·3`. No boundary ties. Held-out rows verified to contain only `shape == 1.0` (cylinder), all `rot > 20`, all `scl ≥ 1.1`.

### Trap compliance
- **T0** — implemented from `montero_splits.json` (i.e. the code), never from the paper's prose. Not "checked against the paper".
- **T1** — `load_latents` slices `[:, 1:]` at load; `montero_test_mask` **raises `ValueError`** on a 6-column array rather than silently selecting nothing. Non-empty assertion runs before every other check.
- **T2** — `a90, a120, a180, a240 = np.pi/2, 4*np.pi/3, np.pi, 2*np.pi/3` reproduced **verbatim** with a ⛔-comment, plus a test (`test_dsprites_angle_constants_are_the_upstream_swapped_ones`) that fails if anyone "repairs" them.
- **T5** — dSprites uses `latent_classes[:,shp]==1` (ellipse); Shapes3D uses label **values** `==1` (cylinder). Verified empirically: for dSprites, `latent_classes[:,shp]==1 ⟺ latent_values[:,shp]==2.0`, and the shape-1 sprite has fill ratio 0.810 (ellipse), vs 1.000 (square) and 0.627 (heart).

---

## 3. ⭐ The positive control — the criterion this spoke lives or dies on

Injected one known held-out row into each **real** train index and re-ran the assertion. Verbatim output (`.claude/scratch/gen-data-splits/positive_control.txt`):

```
[shapes3d] REAL split: n_total=480000 n_train=479928 n_test=72
[shapes3d] assertion PASSED over 479928 train combinations
[shapes3d] POSITIVE CONTROL FIRED: shapes3d/recomb2element: 1 train row(s) satisfy the held-out
           predicate; first is row 426567 (factor indices [8, 8, 8, 5, 1, 12],
           values [0.8, 0.8, 0.8, 1.1071428571428572, 1.0, 21.42857142857143])

[dsprites] REAL split: n_total=737280 n_train=730988 n_test=6292
[dsprites] assertion PASSED over 730988 train combinations
[dsprites] POSITIVE CONTROL FIRED: dsprites/recomb2element: 1 train row(s) satisfy the held-out
           predicate; first is row 341685 (factor indices [1, 2, 13, 21, 21],
           values [2.0, 0.7, 2.0943951023931957, 0.6774193548387096, 0.6774193548387096])
```
The dSprites offender's rotation value is **exactly `grid_rot[13]`**, the float-boundary row — the assertion is discriminating at the very boundary that decides the count.

**Combinations checked: Shapes3D 479,928; dSprites 730,988** (equal to `n_train` in both cases, so the check is not silently empty).

Made permanent as tests, not one-offs:
`test_real_shapes3d_holdout_assertion_and_POSITIVE_CONTROL`, `test_real_dsprites_holdout_assertion_and_POSITIVE_CONTROL` (both assert the injected row id appears in the raised message), plus toy-grid equivalents that run with no data present, plus small-chunk variants (so chunked iteration cannot skip the offender), plus firing tests for the empty-test-set, lost-row, duplicated-row and permuted-axis failure modes.

---

## 4. Axis order — verified empirically, not from the docs

⚠ Documented order was **not trusted**. Two independent checks per dataset.

**(a) Bijection, exhaustive.** For every row of both files, `grid.values_of(row) == file[row]` exactly, in C order: **737,280/737,280** (dSprites) and **480,000/480,000** (Shapes3D). A permuted axis order fails this for almost every row (`test_verify_bijection_catches_permuted_axes` proves the check fires).

**(b) Image decode of known rows** (`.claude/scratch/gen-data-splits/axis_order_evidence.txt`):

dSprites (raw 6-axis order `color, shape, scale, orientation, posX, posY`):

| Axis | Prediction | Measured |
|---|---|---|
| 4 = posX | centre-of-mass x rises, y fixed | cx 17.5 → 26.0 → 32.0 → 40.0 → 46.5, cy ≡ 32.0 |
| 5 = posY | y rises, x fixed | cy 17.5 → 26.0 → 32.0 → 40.0 → 46.5, cx ≡ 32.0 |
| 2 = scale | area strictly rises | 81, 121, 169, 225, 289, 361 |
| 3 = orientation | index 39 = 2π wraps to index 0 | `imgs[...,39,...] == imgs[...,0,...]` exactly; index 10 differs |
| 1 = shape | distinct fill ratios | square 1.000, ellipse 0.810, heart 0.627 |

Shapes3D — per-factor image change map (std over that factor's values, 8×8 coarse); each factor's change lands exactly where its name says:

| Factor | Where the change is |
|---|---|
| 0 floor_hue | bottom 3 of 8 row-bands only; top half ≡ 0 |
| 1 wall_hue | bands 2–5; bottom two bands ≡ 0 |
| 2 object_hue | the central object block only (>90% of total change inside rows 28–56, cols 16–48) |
| 3 scale | object block, size-changing annulus |
| 4 shape | object silhouette |
| 5 orientation | broad (camera azimuth: wall + object) |

**No off-by-one.** The dSprites post-slice indices `shp,scl,rot,tx,ty = 0,1,2,3,4` and Shapes3D's `fh,wh,oh,scl,shp,rot = 0..5` are the ones the data actually carries.

Also cross-checked: the dSprites **hdf5 mirror is row-identical to the npz** (`latents/values` and `latents/classes` equal on a sparse 94-row sample), so image access via hdf5 indexes the same grid as the split built from the npz.

---

## 5. The C set (structurally invalid factor requests)

Emitted at `.claude/data/gen/<dataset>/c_invalid.json`, **32 records each** (16 per kind, `seed=0`), built from the factor grid alone.

- `duplicate_factor` — a complete factor assignment **plus** a second, different value for one factor. e.g. `dsprites-dup-000`: *"factor 'shape' is requested at both 3.0 and 1.0; an image has exactly one value per factor."*
- `out_of_range` — one term replaced by a value strictly outside that factor's realised range. e.g. `shapes3d-oor-015`: *"factor 'scale' requested at 0.6071428571428574, outside its realised range [0.75, 1.25]."*

Schema is embedded in the file under `schema` (documents `record`, both `kinds`, and the arm-agnostic rule). ⛔ **Arm-agnostic by test:** `test_c_invalid_is_structurally_invalid_and_arm_agnostic` scans the emitted `records` + `factors` blocks and fails if the strings `well`, `mode`, `cluster`, `arm`, `encoder`, `clu` or `latent` appear anywhere. Deterministic given `(grid, n_per_kind, seed)` — asserted. `validate_c_invalid` re-derives invalidity from the grid and is itself shown to reject a satisfiable request (`test_validate_c_invalid_rejects_a_satisfiable_request`).

---

## 6. Provenance — this table IS the backup

All under `.claude/data/gen/` (gitignored; **nothing entered git**). Checksums computed on first download, 2026-09-09.

| File | Source URL | Bytes | sha256 | Licence |
|---|---|---|---|---|
| `dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` | `https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` | 19,318,135 | `857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2` | zlib/libpng |
| `dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5` | same repo, `…64x64.hdf5` | 28,024,106 | `729fa22e884c9c500e1f17415d580746b4c5894fbbb780a1319bbcfa39a745bb` | zlib/libpng |
| `shapes3d/3dshapes.h5` | `https://storage.googleapis.com/3d-shapes/3dshapes.h5` (`gs://3d-shapes/3dshapes.h5`) | 267,573,662 | `0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785` | Apache-2.0 |

Derived artifacts (rebuildable; checksums recorded so a stale/hand-edited index is detectable):

| File | Bytes | sha256 |
|---|---|---|
| `dsprites/factor_index.npz` | 5,902,508 | `c6419fedd437e1ff6ec630839a1a4b16b02dc04b573208d8a5b734931e15fe45` |
| `dsprites/c_invalid.json` | 29,278 | `2638eedea85156bb8ef644ec0c9acc68282f9bc1c471636de3790ca100f44bfa` |
| `shapes3d/factor_index.npz` | 3,844,150 | `5f1fce25c40cceae6de2d00fd2145c4cdc73dcc38cee7f6545cbcaea3ce78f49` |
| `shapes3d/c_invalid.json` | 33,008 | `aafb804e917d609824b1f1133b90b834076348f842fd60629313747ffd65c735` |

Index checksums are **reproducible**: `np.savez` stamps every zip member with the fixed `ZipInfo` epoch `(1980,1,1,0,0,0)`, verified by writing the same index twice 5 s apart and comparing digests (asserted in `test_save_index_roundtrips_and_is_checksum_stable`).

**Notes worth carrying:** the spec's `download_url_status: RECONSTRUCTED` for the GCS URL is now **CONFIRMED** (HTTP 200, `x-goog-stored-content-length: 267573662`). The spec's `exact_bytes: NOT OBTAINED` for `3dshapes.h5` is now **267,573,662** (TFDS's "255.18 MiB" is the same file). The dSprites npz byte size matches the spec exactly (19,318,135). No upstream checksum exists for any of the three files — ours are now the anchor.

### Cluster staging recipe (not executed)
```bash
# CSF3, from ~/scratch/CHLU
export CHLU_GEN_DATA_ROOT=$HOME/scratch/clu_gen_data     # NOT $CHLU_DATA_ROOT (that is the industrial cache)
mkdir -p "$CHLU_GEN_DATA_ROOT"

# h5py is NOT in scripts/csf3/requirements-csf3.txt — install it once into the job venv:
uv pip install h5py

# A) fetch on a COMPUTE node (login nodes have no off-campus access):
source scripts/csf3/env.sh
uv run chlu data dsprites --download     # 47 MB (npz + hdf5)
uv run chlu data shapes3d --download     # 268 MB

# B) or push the already-verified copies from the laptop (315 MB total):
rsync -av --progress ~/Desktop/CHLU/.claude/data/gen/ csf3:~/scratch/clu_gen_data/
```
Either way the checksums pinned in `chlu/data/{dsprites,shapes3d}.py` are re-verified on load, so a truncated transfer or a scratch-cleaner casualty is caught rather than silently mis-splitting.

---

## 7. How I verified (commands + observed output)

Main venv per protocol §4 (`/Users/user/Desktop/CHLU/.venv`, Python 3.11.13, numpy 2.4.1, **h5py 3.16.0**), `PYTHONPATH=<worktree>`, `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`.

**End-to-end CLI (both commands, real downloads):**
```
Preparing dSprites (Montero recomb-to-element)
  …npz: 19318135 bytes sha256=857dc001…   …hdf5: 28024106 bytes sha256=729fa22e…
  Factors: ['shape','scale','orientation','posX','posY']   Cardinalities: [3,6,40,32,32]
  n_total=737280  n_train=730988  n_test=6292
✓ hold-out assertion passed over 730988 train combinations
Preparing Shapes3D (Montero recomb-to-element)
  3dshapes.h5: 267573662 bytes sha256=0a0f6ed9…
  Factors: ['floor_hue','wall_hue','object_hue','scale','shape','orientation']  Cardinalities: [10,10,10,8,4,15]
  n_total=480000  n_train=479928  n_test=72
✓ hold-out assertion passed over 479928 train combinations
rc 0 / rc 0
```
**`--quick`:** `quick_train_rows (1000,)`, `n_checked 479928` — i.e. `--quick` shrinks the returned train index but the hold-out assertion still runs over **all** train rows. Subsample is deterministic at fixed seed (`np.array_equal` across two calls: `True`).

**Parser + config:** `chlu data shapes3d --download --quick` and `chlu data dsprites` parse to the right functions; `chlu data mnist` still resolves to `cmd_data_mnist` (no existing line touched). `save_config`→`load_config` round-trips `gen_data`.

**Targeted tests:** `pytest tests/test_gen_data_splits.py -v` → **24 passed in 19.32 s**, including all 8 real-data tests (they skip when the datasets are absent).

**Lint:** `ruff check` clean on all six touched files. `ruff format` applied to the four **new** files only — the repo baseline is *not* format-clean (207 of 282 files would reformat), so formatting shared files would have produced a huge out-of-scope diff.

**Full suite (main venv, both runs `-p no:randomly --no-cov`):**
- **Before** — `b0257bc` in a detached worktree (`../CHLU-gen-data-base`): **1919 passed, 36 warnings in 4834.27 s (1:20:34)**, 0 failed, 0 skipped.
- **After** — `agent/experiment-engineer/gen-data-splits` @ `5fe4dad`: **1943 passed, 36 warnings in 4901.67 s (1:21:41)**, 0 failed, 0 skipped.
- **Delta = +24 = exactly my new tests.** No pre-existing test changed state. ⚠ Note for the record: the suite is now **1919 tests at base**, not the ~354 of the older extras-dependent note; and because I reused the **main** venv (protocol §4) the optional-extra tests all ran — no N−7 drop, and h5py was present so all 8 real-data tests executed rather than skipping.

---

## 8. Risks / open questions

1. **h5py is an undeclared dependency.** Shapes3D is a single HDF5 file, so it is unavoidable. I installed it into the main venv with `uv pip install --python .venv/bin/python h5py` (→ **h5py 3.16.0**) and did **not** touch `pyproject.toml`/`uv.lock`: adding an extra without re-locking makes every concurrent agent's `uv sync --frozen` fail, and re-locking mid-wave is exactly the kind of shared-state change that breaks other spokes. Consequences: (a) another agent's `uv sync` will **prune h5py** and the 8 real-data tests will start skipping (they `importorskip`, so nothing goes red — but nothing goes green either); (b) CSF3 needs the manual install above. **Hub decision needed: add a `gen = ["h5py>=3.0"]` extra + re-lock + regenerate `requirements-csf3.txt`, at a quiet moment.**
2. **The 5,808 → 6,292 correction must propagate** — see §Reconciliation.
3. **`.claude/data/gen/` is 315 MB and gitignored.** The provenance table in §6 is the only backup. If a machine is reimaged, re-download and the pinned checksums confirm identity.
4. **Only the `recomb2element` / `leave1out` condition is implemented.** `recomb2range` and `extrp` are specified in `montero_splits.json` but not built (not requested). Adding them is ~15 lines each given the scaffolding.
5. **Images are not loaded anywhere yet.** `open_images()` returns a lazy h5py handle for both datasets. Whoever wires the encoder must not do `f['images'][:]` (1.7 GB) — the docstrings say so, but nothing enforces it.
6. **I did not read `.claude/handover_context.md`** — the task file says ⛔ *"Read no campaign ledger, charter or handover"*, which contradicts the standing spoke prompt. I honoured the task file (information hygiene for an instrument spoke) and took the CLI/config conventions from the code instead. If any §7 Known Issue applies to `chlu/data/` or `chlu/config.py`, I may have re-introduced it unknowingly; the Hub should spot-check.

## Reconciliation list (⭐ needs a Hub-assigned owner at this review)
1. **dSprites `n_test` is 6,292, not 5,808.** Every site that quotes 5,808 (`montero_splits.json` `expected_counts.n_test` + `n_test_derivation`; `.claude/outputs/gen-montero-splits-spec.md` §Findings; the `gen-data-splits` task file T3) should be updated to *realised 6,292, nominal estimate 5,808 superseded*. The spec was right to flag it and right to refuse to pick — but downstream readers will otherwise cite the wrong number. `n_train` is **730,988**, not 731,472.
2. **h5py extra** (item 1 above) — needs an owner and a quiet window to re-lock.
3. **The spec's two `NOT OBTAINED` provenance fields are now obtained** and should be back-filled into `montero_splits.json`: `3dshapes.h5` exact bytes = 267,573,662; GCS URL status RECONSTRUCTED → CONFIRMED. Also all three dataset sha256s now exist (they did not upstream).

---

## Git footprint

- **Branch:** `agent/experiment-engineer/gen-data-splits`, based on **`main @ b0257bc`** (as named in the task).
- **Worktree:** `../CHLU-gen-data` (fresh, per the task). A second **detached** worktree `../CHLU-gen-data-base @ b0257bc` was created solely for the baseline suite count. ⛔ I did not touch `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge`, or the shared main checkout (which had another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` checked out at `7fcef50` — noted and left alone).
- **Commits** (4, atomic):

| Hash | Subject |
|---|---|
| `ad8df5c` | `[experiment-engineer] add the factor-grid core and a hold-out assertion that can fire` |
| `76aefe6` | `[experiment-engineer] Shapes3D + dSprites loaders with the verbatim Montero filters` |
| `e09eaae` | `[experiment-engineer] wire GenDataConfig and \`chlu data shapes3d\` / \`chlu data dsprites\`` |
| `5fe4dad` | `[experiment-engineer] test the hold-out assertion by making it fire` |

- **Files touched:** `chlu/data/gen_splits.py` (new), `chlu/data/dsprites.py` (new), `chlu/data/shapes3d.py` (new), `chlu/cli/data_cmd.py` (+141, insertions only), `chlu/config.py` (+37, insertions only), `tests/test_gen_data_splits.py` (new). No other tracked file.
- **Rebase:** `git rebase main` (local `main` = `b0257bc`, i.e. my named base) → *"Current branch … is up to date"*, a no-op. ⛔ Did **not** rebase onto `origin/main` (stale at `40c2f31`). No conflicts.
- **Ref verified from the main repo before worktree removal:** yes — `git log --oneline main..agent/experiment-engineer/gen-data-splits` from `/Users/user/Desktop/CHLU` listed all 4 commits, and `git diff --stat` showed the 6 files / 1,764 insertions. Both worktrees then removed; the ref re-verified **after** removal at `5fe4dad5f2745df39b3ea4d94f82cb8dec30585a` (4 commits still reachable). The main checkout was left untouched and clean throughout (`git status --short` empty, still on `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` @ `7fcef50`).
- **Not pushed, no PR** — branch left for review, per the standing rule.

## Proposed handover updates (for the Hub)

- **New config block `gen_data` (`GenDataConfig`)** in `chlu/config.py`, wired at all three house sites. Defaults listed in §1. Overridable via `projects/<name>/config/config.yaml` under `gen_data:`.
- **New CLI:** `chlu data shapes3d [--download] [--quick] [--data-root] [--no-verify]` and `chlu data dsprites` (same flags). Both are fetch → verify → index → assert → emit.
- **New env var `CHLU_GEN_DATA_ROOT`** (distinct from the industrial `CHLU_DATA_ROOT`); default `<repo>/.claude/data/gen`. Worktrees have no `.claude/`, so anything running from a worktree must set it.
- **Known-issue candidate:** *h5py is required by `chlu/data/shapes3d.py` but is not in `pyproject.toml`; a `uv sync` will prune it and silently turn 8 tests into skips.*
- **Pinned facts for §2 (architecture/data):** Montero recomb-to-element realised held-out sets are **Shapes3D 72/480,000** and **dSprites 6,292/737,280** (⛔ not 5,808 — float-boundary realised). Splits are deterministic and carry **no seed**.
- **Retire the 5,808** wherever it appears (reconciliation item 1).
