# SPOKE F — Is the test suite green on native Windows?
**Core budget: `-n 6` workers, no more.** Two other spokes share these 16 cores.

Read `_hub_scratch/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`/c/Users/PratikJawahar/chlu/_hub_scratch/reports/F-test-suite.md`**

## Why you exist
**2,711 tests across 145 files, and the suite has never been run on Windows.** There is **not a single
`sys.platform` skipif in the entire suite** — nothing is guarded for this platform. The one datapoint we
have is bad: of the 7 targeted tests in `tests/test_hub_workspace.py`, **2 failed** on a Windows-specific
cp1252 encoding bug, on byte-perfect files. Finding the rest of that class now costs an hour. Finding it
at hour six of an autonomous ten-hour run costs the run.

The suite is reportedly **green at 354 on macOS** (that count is of a subset the humans quote; treat it as
"green over there", not as a target number here).

## ⛔ Your write surface
Test artefacts only: `.coverage`, `coverage.xml`, `.pytest_cache/` (all gitignored) and your report.
⛔ **Do not fix a single test.** Every failure is a finding to report, not a defect to patch —
`tests/` is pinned content and the whole point of this exercise is an honest failure inventory.

## ⛔⛔ THE ONE THING THAT CAN GO BADLY WRONG — read before running anything
`tests/test_hub_workspace.py` contains **19 tests that build bundles and run
`verify_hub_workspace.sh`, which itself runs pytest.** The script carries an explicit recursion guard
(`CHLU_WORKSPACE_VERIFY_DEPTH`) because this **already happened once: the run forked unboundedly and 28
processes had to be killed by hand.** At depth 0 — which is where you will be — those 19 tests are
*unguarded and will execute.*

⛔ **Deselect that file: `--ignore=tests/test_hub_workspace.py`.** Spoke E owns the bundle/worktree/return
ground and it is already verified green at depth 1. Do not run it, and do not "just try one".

## Division of labour with the other spokes
Spoke E is mutating `.git/` (deleting AppleDouble files, `git worktree add`, `git bundle create`).
**~11 test files shell out via `subprocess.run(["bash", …])` and several touch git.** If one of those
fails in a way that smells like a concurrent-git-index problem, **say so and re-run just that file
afterwards** rather than recording a flaky result as a Windows finding. Name any test you re-ran.

## How to run it
Activate the shell exactly as `CONTEXT.md` §"How to open a working shell" specifies — **`PYTHONUTF8=1` is
load-bearing**; without it you will get a wave of `UnicodeDecodeError`s that are environment artefacts,
not real failures.

`pyproject.toml` forces `addopts = "--cov=chlu --cov-report=xml --cov-report=term"` on every run, which is
expensive under xdist. Suggested:
```bash
python -m pytest -q --no-cov -p no:cacheprovider -n 6 \
  --ignore=tests/test_hub_workspace.py \
  -rf --maxfail=0 --timeout=600 2>&1 | tee /c/Users/PratikJawahar/chlu/_hub_scratch/reports/_F-suite.log
```
- `--maxfail=0` — **do not stop at the first failure.** A complete inventory is the deliverable.
- `--timeout=600` only if `pytest-timeout` is installed; check first, and say if it is not. A hung test is
  a finding too.
- ⚠ **A cold JAX import is ~20 minutes on the Mac** and is paid **once per xdist worker**. Budget for it,
  and report the wall-clock cost of startup separately from the tests. If `-n 6` startup proves
  pathological, report that and fall back to a lower `-n`.
- Do a **coverage run only if the no-cov run finishes cleanly and quickly enough** — it is a nice-to-have.

## Failures that are already predicted — confirm or refute each, do not rediscover
| # | File / pattern | Prediction |
|---|---|---|
| 1 | `tests/test_job_cpu_gen.py` | ⛔ **Will error.** Parametrises over `signal.SIGKILL` and sends `signal.SIGUSR1` — **neither exists on Windows.** May fail at collection. |
| 2 | ~11 files shelling `subprocess.run(["bash", …])` — `test_c3_csf3_harness`, `test_c3_gb_geometry`, `test_c3_run3_budget_exemption`, `test_cloud_launch`, `test_cloud_runner`, `test_gen_launcher`, `test_stage_artefacts`, `test_stage_gcs`, `test_ttt_stability_and_d5_wiring` | May work via Git Bash's `bash.exe`, but `/c/…` vs `C:\…` translation is the risk |
| 3 | 4 files asserting `os.access(p, os.X_OK)` | ⚠ **Will PASS VACUOUSLY** — Windows returns `True` for any readable file. **Report these by name: a vacuous pass is worse than a failure**, because someone will quote it. |
| 4 | Anything hard-coding `/tmp` | Git Bash maps it, native Python may not |
| 5 | Bare `Path.read_text()` on UTF-8 files | Should be masked by `PYTHONUTF8=1` — confirm none survive |
| 6 | `test_gen_data_splits.py` (8 real-data tests) | `importorskip` on `h5py` + the absent 255 MiB `3dshapes.h5` ⇒ **silent SKIPs**. `pyproject.toml` itself warns this gap is "invisible unless you read the skip count". **Read the skip count.** |
| 7 | Anything importing `chlu/training/train_cluformer.py` | ⛔ module-scope `import resource`, Unix-only ⇒ `ModuleNotFoundError`. Reached by `exp_anti_erosion.py` (module scope) and lazily by `chlu/eval/text_slices.py`. |

## The deliverable — a triaged inventory, not a number
`N passed, M failed` is not useful on its own. Produce a table: `test | error class | Windows-specific? |
severity`, with every failure sorted into exactly one bucket:

- **A — Windows-only environment defect** (path form, encoding, signals, `resource`, exec bit). Cosmetic
  to the science; blocks nobody if the affected work is not run here.
- **B — genuine bug the Mac also has**, merely surfaced here first. ⭐ **These are the valuable ones.**
- **C — vacuous pass** (see prediction 3). Report by name.
- **D — skip that should have been a run** (see prediction 6). Report the skip count and the reason string.

Then answer the question that actually matters: ⭐ **does any failure touch the generative line —
`exp_gen_cell`, `exp_gen_baselines`, `exp_gen_atom_mean`, `gen_store`, `gen_bridge`, `gen_splits`,
`gen_certificate`, `gen_dream`, the `chlu/eval/` scoring path?** That is the code a hub will actually run
here. A hundred failures in the CSF3/cloud/cluformer harness matter far less than one in `gen_bridge`.

## Report shape
Lead with: **can a hub trust this machine's Python for the generative line, yes or no** — then the
headline counts, then the triage table, then the bucket-B list in full. End with numbered open questions.

⭐ Before you finish, **confirm no sub-agent of yours is still running.**
