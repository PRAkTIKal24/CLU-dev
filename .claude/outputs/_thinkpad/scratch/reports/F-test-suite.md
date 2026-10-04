# SPOKE F — Is the test suite green on native Windows?

**Run:** 2026-09-16, `chlu-hub-bundle`, `.venv` Python 3.12.10, jax 0.9.0, numpy 2.4.1, `-n 6`,
`--ignore=tests/test_hub_workspace.py`. Two full runs, plus targeted serial re-runs.

---

## ⭐ THE ANSWER: can a hub trust this machine's Python for the generative line?

**YES — with two named exceptions and one hard prohibition.**

- ✅ Every generative module imports and runs: `exp_gen_cell`, `exp_gen_baselines`, `exp_gen_atom_mean`,
  `gen_store`, `gen_bridge`, `gen_splits`, `gen_certificate`, `chlu.eval`, `chlu.eval.text_slices`.
  All three `exp_gen_*` entry points answer `python -m … --help` correctly.
- ✅ Of 37 `test_gen_*` / `test_eval*` files, **33 are fully green.** The 4 that are not are analysed below.
- ✅ The Shapes3D split is **scientifically bit-identical** to the pinned macOS artefact — I diffed it
  member-by-member against `.claude/data/gen/shapes3d/factor_index.npz` (read-only). All 11 members byte-equal.

⛔ **The hard prohibition: `import chlu.cli` and `python -m chlu` are DEAD on this machine.**
`CONTEXT.md` says `train_cluformer`'s `import resource` "is not on the `import chlu` path" — true, but it
**is** on the `chlu.cli` path. `chlu/cli/__init__.py:4` → `experiment_cmd.py` → `train_cluformer` → `resource`.
**A hub must invoke `python -m chlu.experiments.exp_gen_*`, never the `chlu` CLI.** This is the single most
load-bearing operational fact in this report.

⚠ **The two exceptions** — one real, one cosmetic:
1. `store_hash` in `test_gen_encoder_unfreeze.py` **genuinely diverges** on this machine (stable, thread-count
   independent). `well_sizes`, `codes_sum`, `bits_per_read`, `store_bytes` all match frozen. Structure is
   right; float parameter bytes are not. **Unresolved — see open question 1.**
2. `test_gen_wide_split` fails on a **ZIP container byte**, not on science (see bucket A9).

---

## Headline counts

| | run 1 (`-rf`) | run 2 (`-rfsE`) |
|---|---|---|
| failed | **117** | **116** |
| passed | 2602 | 2603 |
| skipped | 25 | 25 |
| errors | **42** | **42** |
| wall clock | 563.29 s (9 m 23 s) | 573.70 s (9 m 33 s) |

Two runs differ by exactly one test (`test_download_concurrency`, a genuine Windows file-lock flake — A8).
**Everything else is deterministic.** 19 modules never collected at all; their tests are in neither count.

### Startup cost — the ~20-minute JAX fear does **not** reproduce
- Cold `import jax` + first `jnp` op: **1.4 s**. `import chlu`: **7.0 s**.
- `-n 6` startup was not pathological; no fallback to a lower `-n` was needed. Budget ~10 min for a full run.

### ⚠ `pytest-timeout` is NOT installed
I dropped `--timeout=600`; the flag would have errored. **A hung test would have stalled silently, not
failed.** Neither run hung (both finished in ~9.5 min), so nothing was missed — but the guard was absent.

### Coverage run — not done
`pyproject.toml` forces `--cov`. The no-cov run is 9.5 min and other spokes share the cores; a coverage run
was a declared nice-to-have and I spent the budget on triage instead. Not done, deliberately.

---

## Triage table

Every one of the 117 failures + 42 errors is in exactly one bucket. Counts are run 1.

| # | Test / pattern | n | Error class | Windows-specific? | Bucket | Severity |
|---|---|---|---|---|---|---|
| A1 | 13 modules + `test_c3_run3_budget_exemption` setup | 71 F + 36 E | `ModuleNotFoundError: No module named 'resource'` | **Yes** — Unix-only stdlib | A | High (kills the CLI) |
| A2 | `test_job_cpu_gen.py` | 6 E | `AttributeError: module 'signal' has no attribute 'SIGKILL'` | **Yes** | A | Low (CSF3 only) |
| A3 | `test_gen_launcher.py` | 17 | `FATAL[job_gpu_gen]: refusing to run: RUN_DIR=…` | **Yes** — `case` glob vs `\` | A | Low (CSF3 only) |
| A4 | `test_gen_launcher.py` plan tests | 2 | `AssertionError` — expected built with `\`, script emits `/` | **Yes** | A | Low |
| A5 | `test_stage_gcs.py` | 12 | rc 49 "Python was not found… Microsoft Store" | **Yes** — `PATH` joined with `":"` | A | Low (cloud only) |
| A6 | `test_stage_artefacts.py` | 5 | sha256 is 65 chars, leading `\` | **Yes** to trigger, **latent everywhere** | A/B | Medium |
| A7 | `test_cloud_launch.py` | 3 | `[WinError 193] %1 is not a valid Win32 application` | **Yes** — `gcloud` not directly execable | A | Low (cloud only) |
| A8 | `test_download_concurrency.py` | 1 | `PermissionError: [WinError 5] Access is denied` | **Yes** — file locking | A | Low (**flaky**: run 1 only) |
| A9 | `test_gen_wide_split.py` | 1 | sha256 differs; **11 bytes, all ZIP `create_system`** | **Yes** — CPython `zipfile` | A | Medium (see below) |
| B1 | `test_gen_encoder_unfreeze.py` ×2 | 2 | frozen fingerprint moved | **No** — `jax_enable_x64` leak | **B** | **High** |
| B2 | `test_gen_wells.py` | 1 | `assert -1.14e-08 >= 0.0` | **No** — float32 epsilon | **B** | Medium |
| B3 | `test_kt.py` | 1 | `0.82504… == 0.82448… ± 1e-15` | **No** — machine-pinned golden | **B** | Medium |
| B4 | `test_monitors.py` | 1 | `assert False is True` | **No** — sub-ULP construction | **B** | Medium |
| **C** | 3 live `os.access(…, X_OK)` sites | 4 passed | **vacuous pass** | **Yes** | **C** | **High (misleading)** |
| **D** | dataset/artefact absence | 25 skipped | silent skips | No | **D** | Medium |

**Totals:** A = 112 failures + 42 errors. B = 5 failures. 112 + 5 = 117 ✓

---

## Bucket B in full — the valuable ones

### B1. `test_gen_encoder_unfreeze.py` — a frozen scientific fingerprint corrupted by a leaked global. ⭐⭐

**This is the most important finding in the report.** Proven, not inferred:

```
x64=True  enc_hash  = 094ae9c3941b8fb69234199213d22a061dd47d6bfc2bfa70d64f6a974fb95646
full-suite observed = 094ae9c3941b8fb69234199213d22a061dd47d6bfc2bfa70d64f6a974fb95646   <- exact match
frozen/expected     = 7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693
```

Another test module enables `jax_enable_x64` **process-wide**; `test_gen_encoder_unfreeze.py` does not
defend itself, so its "frozen path is bit-identical" check silently measures the wrong precision.

- Run the file **alone** → only `store_hash` fails, and `test_the_baseline_decoder_recipe_…` **passes**.
- Run it in the **full suite** → `enc_hash` fails and the baseline-decoder test fails too.
- Under `-n 6`, which modules share a worker process is **scheduling-dependent**, so this test's verdict
  is nondeterministic **on any platform, including the Mac.**

⛔ **The x64 leak changes the science, not just a hash:**

| key | x64 OFF (correct) | x64 ON (leaked) |
|---|---|---|
| `well_sizes` | `[8, 23, 225]` ✅ matches frozen | `[11, 220, 25]` ❌ |
| `codes_sum` | `84` ✅ matches frozen | `114` ❌ |

The codebase **already knows about this hazard** — ≥6 modules (`test_blocks`, `test_c2w11_nulls`,
`test_c2w11_substrate`, `test_clu_system`, `test_cluformer_pilot`, `test_anti_erosion`) carry explicit
save/restore guards and comments saying "another test enables `jax_enable_x64` process-wide".
`test_gen_encoder_unfreeze.py` was never given one. **Not a Windows problem. Fix upstream.**

### B2. `test_gen_wells.py::test_the_objective_rewards_multimodality_and_penalises_dependence`
`assert float(Array(-1.1415797e-08, float32)) >= 0.0`. The objective is analytically ≥ 0 but computed in
float32, where it lands at −1.1e-08. The test demands exact non-negativity of a float32 quantity with no
tolerance. Latent on any platform; x86-64 rounding merely lands on the wrong side.

### B3. `test_kt.py::test_reduced_rho_s_round_trip_exact`
`0.8250428728431519` vs expected `0.8244828012941149 ± 1e-15`. Off by **5.6e-4 — 10¹¹× the tolerance**, so
this is *not* a rounding artefact: the Monte-Carlo aggregate genuinely differs. The docstring says it
"reproduces the laptop aggregate exactly" — a golden constant pinned to one machine's libm, asserted at
`abs=1e-15`. Its sibling `test_winding2d_round_trip_exact` (integer-valued) **passes**, which isolates the
divergence to the float reduction. ⚠ This test can only ever pass on the machine that produced the constant.

### B4. `test_monitors.py::test_monitor6_dead_band_does_not_trip_on_a_numerically_zero_slope`
The dead-band assertions **pass** — the feature works. What fails is the audit field
`assert r.detail["tripped_pre_repair"] is True`. The test manufactures its "epsilon artefact" with
`loss * (1.0 - 1e-16 * i)`, a sub-ULP perturbation of a float64. On x86-64 that rounds to exactly zero
slope, so the pre-repair predicate never trips and there is nothing to record. The *test scaffolding* is
machine-dependent, not the monitor.

---

## Bucket C — vacuous passes ⛔ REPORT BY NAME

`os.access(p, os.X_OK)` returns `True` for **any readable file** on Windows. Proven:

```
scripts/csf3/job_gpu_c3_seeds.sh: mode=0o100666 X_OK=True     <- no exec bit at all
scripts/csf3/job_gpu_gen.sh:      mode=0o100666 X_OK=True
CONTROL pyproject.toml:           mode=0o100666 X_OK=True     <- not executable by any definition
CONTROL README.md:                mode=0o100666 X_OK=True
```

**These 4 test outcomes are green and prove nothing. Do not quote them as evidence of executability:**

| Test | Site |
|---|---|
| `test_c3_csf3_harness.py::test_the_ladder_script_exists_is_executable_and_parses` | `:488` |
| `test_gen_launcher.py::test_script_exists_and_parses` | `:98` |
| `test_stage_artefacts.py::test_script_exists_and_parses[STAGE_SH]` | `:217` |
| `test_stage_artefacts.py::test_script_exists_and_parses[VERIFY_SH]` | `:217` |

(A 5th site, `test_hub_workspace.py:233`, was deselected as instructed.)

⚠ A **second vacuity of the same shape**: `scripts/cloud/stage_gcs.sh:150` guards with
`command -v "$PY_BIN" || die "no python at…"`. On Windows the Microsoft Store **alias stub** satisfies
`command -v`, so the guard passes and the script then dies at exit 49. The guard is vacuous here too.

---

## Bucket D — skips that should have been runs (25 total)

**Prediction 6 is half right and half wrong.** `h5py 3.16.0` **is installed**, so `importorskip("h5py")`
never fired. **Every skip is dataset absence, not a missing library.**

| Reason | n |
|---|---|
| Shapes3D `3dshapes.h5` absent (255 MiB, never committed) | 11 |
| dSprites `.npz` / `.hdf5` mirror absent | 10 |
| C2W6 artifact tree absent (gitignored) | 3 |
| industrial loaders need the `eval` extra | 1 |

Affected files: `test_gen_data_splits` (8), `test_gen_bridge_dsprites` (4), `test_gen_bridge_wide` (3),
`test_gen_wide_split` (2), `test_gen_codebook_exponent` (2), `test_gen_rowsplit` (1), `test_gen_classifier` (1),
`test_store_lifecycle` (3), `test_industrial_loaders` (1).

⭐ **21 of 25 skips are real-data generative tests.** `pyproject.toml`'s warning is justified: this gap is
invisible unless you read the skip count. Any claim resting on real-data Shapes3D/dSprites behaviour is
**untested on this machine.**

---

## Predicted failures — confirmed / refuted

| # | Prediction | Verdict |
|---|---|---|
| 1 | `test_job_cpu_gen.py` errors on `SIGKILL`/`SIGUSR1` | ✅ **Confirmed.** 6 collection errors, `AttributeError … 'SIGKILL'`. Fails at collection, exactly as predicted. |
| 2 | ~11 `subprocess.run(["bash", …])` files; `/c/…` vs `C:\…` risk | ⚠ **Partly.** `bash` itself works fine. The breakage is never bash — it is `case` globs vs `\` (A3), `PATH` joined with `:` (A5), and `sha256sum` escaping (A6). |
| 3 | 4 `os.access(X_OK)` files pass vacuously | ✅ **Confirmed and proven.** 4 live vacuous passes named above. |
| 4 | Anything hard-coding `/tmp` breaks | ❌ **Refuted.** All 5 sites pass `/tmp/x` as an opaque argparse string; nothing touches the filesystem. `test_gen_bridge`, `test_v1_hopfield_gate` all green. |
| 5 | Bare `Path.read_text()` survives | ✅ **Refuted as a risk / masked as predicted.** **Zero** `UnicodeDecodeError`/`UnicodeEncodeError` across both runs. `PYTHONUTF8=1` fully covers it. |
| 6 | `test_gen_data_splits` silently skips on `h5py` + absent data | ⚠ **Half.** h5py *is* installed; all skips are dataset absence. 25 skips, reasons listed above. |
| 7 | `import resource` kills `exp_anti_erosion` + `text_slices` | ✅ **Confirmed, and worse.** It also kills **all of `chlu.cli`** and `python -m chlu`. `chlu.eval.text_slices` imports fine (its `train_cluformer` import is lazy and not triggered). |

---

## What is broken — root causes, with the evidence

### A1 — `import resource` (71 F + 36 E, by far the largest class)
```
File "…\chlu\training\train_cluformer.py", line 42, in <module>
    import resource
ModuleNotFoundError: No module named 'resource'
```
13 modules die at collection; `test_c3_run3_budget_exemption` contributes 23 setup errors + 13 failures.
**Operationally the only part that matters is that it takes `chlu.cli` with it** (headline, above).

### A6 — `sha256sum` prefixes a `\` when the filename contains a backslash ⭐
5 `test_stage_artefacts` failures share one cause. The manifest sha is parsed as **65 characters**:
```
assert ('\e74f70a32bd68fdc38a125c830530d4c015ef6e778de3e4b58108b162679f081' and 65 == 64)
```
Reproduced directly:
```
$ sha256sum "C:\Users\PratikJawahar\chlu\chlu-hub-bundle\pyproject.toml"
\bb278675c3c46f99338f7cfdded36a0bcac35cd3fc2ec205f25e923bc4a4533b *C:\\Users\\…\\pyproject.toml
  ^ GNU coreutils escape prefix
```
GNU coreutils escapes filenames containing a backslash or newline by prefixing the whole line with `\`.
The script takes field 1 verbatim. On Windows **every** path triggers it. ⚠ Classed A/B: Windows triggers
it universally, but the parser would mis-handle any backslash-bearing filename on Linux too.

### A9 — the npz checksum is platform-locked by CPython, and the science is fine ⭐
`test_gen_wide_split` asserts `sha == 5f1fce25…`; we produce `d27f7af5…`. I diffed my `.npz` against the
pinned `.claude/data/gen/shapes3d/factor_index.npz`:

```
file len equal: True 3844150
num differing bytes: 11
byte values mine: [0,0,0,0,0,0,0,0,0,0,0]
byte values gold: [3,3,3,3,3,3,3,3,3,3,3]
```
All **11 ZIP members are byte-identical**; the 11 differing bytes are one `create_system` field per member
in the central directory — `0` (MS-DOS/FAT) vs `3` (Unix). This is CPython's `zipfile.ZipInfo.__init__`
branching on `sys.platform == 'win32'`. npz timestamps are already normalised to 1980-01-01, so the
checksum is otherwise perfectly deterministic (verified stable across a 2.5 s gap).

⭐ **Consequence: every `.npz` sha256 pin in this codebase is OS-locked.** An `.npz` regenerated on Windows
can never match a macOS-pinned checksum even when the scientific content is bit-perfect. The check still
detects content corruption — it just *also* detects the operating system. The split itself
(`n_total=480000, n_test=72, n_train=479928`, `condition=recomb2element`) asserts **green** before the sha line.

### A5 — `test_stage_gcs.py:80` joins `PATH` with a literal `":"`
```python
env["PATH"] = "%s:%s" % (bindir, env["PATH"])     # os.pathsep on this platform is ';'
```
On Windows this fuses `bindir` and the venv `Scripts` entry into one nonexistent directory, so the test's
own fake `gcloud` stub is unreachable **and** `python3` falls through to the Microsoft Store alias:
`stage_gcs.sh:95` does `PY_BIN="${STAGE_PYTHON:-python3}"` → rc 49, empty stdout, all 12 tests fail.

### A3 — the launcher's own guard rejects a Windows path
```bash
case "$RUN_DIR" in
    */CHLU-gen/workspaces/*) ;;
    *) die "refusing to run: RUN_DIR='$RUN_DIR' is not under a CHLU-gen workspace" ;;
esac
```
`tmp_path` arrives backslash-separated, so the forward-slash glob cannot match. The guard is working as
designed; the path form defeats it.

---

## What I could not resolve honestly

1. **`store_hash` genuinely diverges and I could not explain it.** In a clean process with x64 off,
   `enc_hash` and `dec_hash_v1` reproduce the frozen values **exactly**, but
   `store_hash = 6ac774f838be…` ≠ frozen `a2a916f4a054…`. It is **not** the x64 leak (x64 ON gives a third
   value, `11b3c5652fdd…`) and **not** thread count (identical at `OMP_NUM_THREADS` = 1, 4, 8).
   Every structural sibling matches frozen. I could not determine whether this is an x86-64-vs-ARM float
   difference or a real drift, and I did not have a macOS box to compare against.
2. **Whether the Mac is actually green on the 5 bucket-B tests.** I reason B1 is scheduling-dependent
   everywhere, but I could not run macOS to confirm. "Green over there" is a report, not something I verified.
3. **The real-data generative path is untested here** — 21 skips, no Shapes3D/dSprites on disk. I did not
   download 255 MiB; that was not in scope and would have been a cloud/network spend decision.
4. **Coverage run not performed** (see above) — a declared nice-to-have, traded for triage depth.
5. **`tests/test_hub_workspace.py` was deselected as instructed** and is not represented anywhere in these
   counts. I did not run it, not even once.

### Concurrency with other spokes — what I checked
Spoke E mutates `.git/`. **No failure in this report is a concurrent-git artefact.** I re-ran the 6
suspicious tests **serially, single-process** and all 6 reproduced byte-identically; I then re-ran
`test_gen_encoder_unfreeze.py` alone and as a single test. Tests re-run: the 6 in B1–B4 plus
`test_gen_wide_split`, `test_gen_encoder_unfreeze` (×3 configurations). The only non-deterministic result
across two full runs was `test_download_concurrency` (A8), which is a Windows file lock, not git.

⚠ **Observed at 19:45, after my runs finished:** another spoke is running
`tests/test_gen_encoder_unfreeze.py tests/test_gen_wide_split.py` under **Python 3.11** from
`C:\clu-venv\py311-probe` and a `uv` 3.11, plus an `_hub_scratch/I-probe/load_check.py`. Those are the two
files carrying my most important findings. **My numbers are Python 3.12.10 from `chlu-hub-bundle/.venv`** —
the hub should not merge them with a 3.11 probe's results without checking the interpreter.

### ⭐ Sub-agent check
**I spawned no sub-agents.** Both of my background runs completed (exit 0). I verified no `pytest`/`xdist`/
`execnet` process of mine survives:
```
--- any pytest/xdist in cmdline? ---
NONE
```
The 4 live `python.exe` processes at exit all belong to other spokes (paths: `clu-venv\py311-probe`,
`uv\cpython-3.11`, `_hub_scratch/I-probe/`), started *after* my last run ended. **Nothing of mine is running.**

---

## Numbered open questions

1. ⭐ **Is `store_hash = 6ac774f838be…` a Windows float difference or a real drift?** This is the one
   generative-line result I could not clear. Cheapest resolution: run
   `pytest tests/test_gen_encoder_unfreeze.py` on the Mac that produced `FROZEN_FINGERPRINT` and report
   `param_hash(store)`. If the Mac also gives `6ac774f8…`, the frozen constant is simply stale and the
   Windows box is exonerated entirely.
2. ⭐ **Who owns fixing the `jax_enable_x64` leak (B1)?** ≥6 modules already carry defensive save/restore
   guards. Should `test_gen_encoder_unfreeze.py` get the same guard, or should a `conftest.py` (there is
   **none** in this repo) reset x64 per module? Until then this test is a coin flip under xdist on every
   platform, and its verdict on the "frozen path" cannot be trusted anywhere.
3. **Do any pinned `.npz` checksums in `BUNDLE_MANIFEST.json` get re-verified after regeneration on a
   non-Mac machine?** If so they will fail on the `create_system` byte alone (A9). Does the programme want
   a content-hash (hash the members, not the container) instead?
4. **Was `FROZEN_FINGERPRINT` recorded with x64 on or off?** `well_sizes` and `codes_sum` match the x64-OFF
   path, which suggests off — worth confirming, because if any frozen constant was captured with the leak
   active it is wrong at the source.
5. **Is the `chlu` CLI needed by any hub target on this box?** If yes, `train_cluformer.py:42` needs a
   guarded import (`if sys.platform != "win32"`) — but the file is pinned and ⛔ I did not touch it.
6. **Should `test_kt.py`'s `abs=1e-15` golden (B3) be re-pinned with a physical tolerance?** As written it
   asserts cross-machine bit-reproducibility of a Monte-Carlo float reduction, which no platform can promise.
7. **Is `pytest-timeout` meant to be in the venv?** It is not. Without it a hung test stalls silently rather
   than failing — worth installing before any long autonomous run.
8. **Do the humans want the A5/A6 test-harness bugs (`PATH` `:` join, `sha256sum` `\` prefix) fixed?** Both
   are one-line fixes (`os.pathsep`; strip a leading `\`) in pinned `tests/` content. ⛔ I changed nothing.

---

## Appendix — exact commands and artefacts

```bash
export CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-hub-bundle
cd "$CHLU_ROOT" && source ./env_hub.sh
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export PYTHONUTF8=1 VERIFY_PYTHON="$CHLU_ROOT/.venv/Scripts/python.exe"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -m pytest -q --no-cov -p no:cacheprovider -n 6 \
  --ignore=tests/test_hub_workspace.py -rf --maxfail=0
```
`--timeout=600` omitted — `pytest-timeout` is not installed.

Artefacts written (all beside the bundle, none inside it, nothing in `chlu/`, `tests/` or `.claude/`):
- `_hub_scratch/reports/F-test-suite.md` — this report
- `_hub_scratch/reports/_F-suite.log` — run 1, full output, 7851 lines
- `_hub_scratch/reports/_F-suite-run2.log` — run 2, with skip/error reasons
- `_hub_scratch/reports/_F-triage.json`, `_F-bodies.txt` — parsed per-failure classification

⛔ **No file in `BUNDLE_MANIFEST.json` was modified, moved or deleted. No test was fixed. No `git` write of
any kind. No GPU, no cloud spend. `tests/test_hub_workspace.py` was never executed.**
