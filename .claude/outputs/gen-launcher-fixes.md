# gen-launcher-fixes — experiment-engineer report

**Task + acceptance criterion:** add a first-class `certificate` stage to `scripts/csf3/job_gpu_gen.sh` with its
own workspace path (never `cell/`), keep every pre-existing guard firing (proven by tests), and hand the Head a
ready-to-run submit block — ⛔ with every cluster item reported as NOT RUN.

**Status:** **done** (code + docs + tests; full suite green before *and* after, 2191 → 2205 passed). ⛔ **Every cluster item: NOT RUN.** No job was submitted and no job id exists — for the `certificate` stage or for any other gen stage, the plumbing smoke included.

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** *none — instrument.* ⛔ Nothing scientific ran; no measured ratio/exponent/slope ⇒ **no PREREG.md** (the
  pre-registration rule is not triggered).
- **Laundering control:** N/A. **Falsifies:** N/A — the failure mode is a script that looks right and has never run.

## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — read this in the first 10 lines)
1. **`.claude/outputs/gen-certificate.md` §10 is now stale.** Its submit block uses the banned workaround
   (`STAGE=cell` + `EXTRA_ARGS=--stage certificate`) and its "Realised output paths" all say `.../cell/clu_s*/`.
   The correct paths are `.../certificate/clu_s*/`. **Owner needed: curator**, or the Hub simply strikes §10's
   block and points at `scripts/csf3/README.md` §7.6. ⛔ Do not submit from that report.
2. **R26 ruling 7's second half is NOT done and has no owner.** "⛔ The `gen_data.data_root` ambiguity that
   silently skips the C set … must fail loudly, no default." That lives in `chlu/config.py` /
   `chlu/data/gen_data.py`, and my task file says **⛔ Nothing under `chlu/`**. I did not touch it. **It needs its
   own engineer task**, and it is a *silent-skip* defect, i.e. exactly the "declared not-run reported as null"
   class. ⚠ It is not fixed by this spoke and must not be assumed fixed because the launcher landed.

---

## 1. What I did

### 1.1 `scripts/csf3/job_gpu_gen.sh` — `certificate` is now a first-class stage
| change | before | after |
|---|---|---|
| `STAGE` whitelist | `encoder\|decoder\|store\|cell` | `encoder\|decoder\|store\|cell\|certificate` |
| unset-`STAGE` message | `…set STAGE=encoder\|decoder\|store\|cell` | `…\|cell\|certificate` |
| invalid-`STAGE` message | `not one of: encoder decoder store cell` | `… cell certificate` |
| default payload module | (none for certificate) | `chlu.experiments.exp_gen_certificate` |
| upstream | (none) | **`store`** |
| workspace path | (rode on `cell/` via `EXTRA_ARGS`) | **`…/<dataset>_<pretext>/certificate/<arm>_s<seed>/`** |

⭐ **The path needed no new code.** `run_dir_for()` already derives the directory from `$STAGE`
(`$WORKSPACE/$STAGE/${arm}_s${seed}`), so admitting `certificate` to the whitelist *automatically* gives it its own
subdirectory. That is also why the old workaround was so dangerous: it did not "share" `cell/` by accident, it was
*forced* to, because `STAGE` was the only thing the path was built from and the payload's stage was smuggled past it
in `EXTRA_ARGS`. The fix removes the smuggling channel rather than papering over it.

### 1.2 ⭐ Declared upstream: `certificate` → `store`, **not** `cell`
The DAG is a chain with **two leaves**:
```
encoder -> decoder -> store -+-> cell
                             `-> certificate
```
**Evidence for the declaration** (read out of `chlu/experiments/exp_gen_certificate.py`, not assumed):
- it loads `store_plan.json` (`load_store_plan(store_runs_dir(...) / "store_plan.json")`, line 297) and
  `seed_<n>/store_manifest.json` (line 299) — both **store** artefacts;
- it imports `gather_features`, `select_store_rows`, `_assert_cache_reencodes` from `exp_gen_store`, and the bridge
  encoder via `bridge_root`/`$CHLU_GEN_BRIDGE_ROOT` — i.e. it reaches the encoder/decoder *through* the store's own
  dependency, which the chain already enforces;
- it imports **nothing** from any cell module, and `chlu/experiments/exp_gen_cell.py` **does not exist yet**.

⇒ Gating `certificate` on `cell` would serialise two independent leaves and let a cell failure block a result that
does not depend on the cell. `store` is the tightest correct upstream. The existing per-(arm, seed)
`status == "ok"` manifest check applies to it unchanged.

### 1.3 `tests/test_gen_launcher.py` — 25 → 39 tests (+14)
New `§5` block, plus `certificate` added to the existing upstream parametrize:
| test | what it pins |
|---|---|
| `test_unset_stage_message_names_the_certificate_stage` | the "no default" message enumerates every stage that exists |
| `test_invalid_stage_message_names_the_certificate_stage` | ditto for the whitelist rejection |
| `test_certificate_plans_its_own_paths_never_the_cells` | PLAN prints `-a 0-2%2`, `certificate/clu_s{0,1,2}`, and **`/cell/` appears nowhere in the output** |
| `test_certificate_upstream_is_the_store_not_the_cell` | runs enc→dec→store, asserts `cell/` does **not** exist, certificate still succeeds and prints `upstream ok: …/store/clu_s0/run_manifest.json` |
| `test_certificate_writes_its_own_workspace_and_leaves_the_cell_alone` | cell and certificate both run; two distinct dirs, two manifests, each `axes.stage` correct, `"/cell/" not in cert["paths"]["run_dir"]` |
| `test_certificate_default_payload_module` | manifest `payload.module == chlu.experiments.exp_gen_certificate`, `extra_args == ""`, no `EXTRA_ARGS=` in `non_default_flags` — ⛔ the workaround's fingerprint is absent |
| `test_certificate_payload_module_exists_on_disk` | the typo class that would otherwise surface only after a GPU is allocated (path check, not an import — importing pulls in JAX) |
| `test_stage_refuses_to_run_on_a_missing_upstream[certificate-store]` | the upstream guard, parametrized |
| `test_certificate_stage_keeps_the_commit_guard` | `GEN_GIT_COMMIT` still mandatory **under the new stage**, and no manifest is written |
| `test_certificate_stage_keeps_the_workspace_guard` | still refuses a `RUN_DIR` outside a `CHLU-gen` workspace; `projects/` not created |
| `test_certificate_stage_keeps_the_short_array_guard` | `-a 0-1` against 1 arm × 3 seeds is refused; `PARTIAL=1` accepts it |
| `test_certificate_stage_keeps_the_grid_bounds_guard` | task id 3 in a 3-task grid is refused |
| `test_certificate_stage_keeps_the_unset_axis_guards[DATASET]` / `[PRETEXT]` | the other two axes still fail loudly |

⭐ Guards are proven **by execution under `STAGE=certificate`**, not by reading the diff — that was the explicit
acceptance criterion.

### 1.4 `scripts/csf3/README.md` — new §7.6
Appended §7.6 (`certificate` stage): the two-leaf DAG, why the upstream is `store`, the default module and workspace
path, the **ban** on the `STAGE=cell` workaround, the one-literal-line submit block, and an explicit
`[UNTESTED-ON-CLUSTER, 2026-09-13]` marker.

⚠ **Two lines outside the appended subsection were edited, both inside §7 and both truth-preserving** (declaring
them because the task said "do not reformat or edit any other line"):
- §7.1 `STAGE` table row — it listed a whitelist that no longer matches the script. Leaving it would have shipped a
  doc that contradicts the code it documents.
- §7.5 verification bullet — said "25 tests"; there are now 39. Leaving it would have been a false verification claim.
No other line in the file was touched (`git diff` on README: **53 insertions, 2 deletions** — the 2 are exactly these).

---

## 2. ⛔ CLUSTER: NOT RUN — every item

**Standing constraint:** CSF3 needs GlobalProtect **and** 2FA; it is unreachable from an agent session. Nothing
below was submitted.

| cluster acceptance item | status |
|---|---|
| `-p serial` plumbing smoke of `job_gpu_gen.sh` | ⛔ **NOT RUN** — still never submitted, as of 2026-09-13 |
| `certificate` stage on CSF3 (any seed) | ⛔ **NOT RUN** |
| any gen stage at all (encoder/decoder/store/cell) | ⛔ **NOT RUN** — zero jobs, zero job ids |
| GPU preflight / `env.sh` on a compute node / exit-trap slurm-stream copy / `$SCRATCH` resolution / wallclock sizing | ⛔ **NEVER OBSERVED** |

⛔ **I am not reporting any cluster item as a pass, and there is no job id anywhere in this report.**

---

## 3. ⭐ THE SUBMIT BLOCK FOR THE HEAD

⚠ **Read §3.3 before the real submission** — there is a genuine input-path gap that the smoke does not expose and
that I could not fix (it is under `chlu/`, which my task forbids).

### 3.0 Commit to pin
- Branch tip **`78882cd`** on `agent/experiment-engineer/gen-launcher-fixes` (base `main @ 3faef27`).
- ⭐ **Pin the commit you actually push.** After the Hub merges, use the merge commit:
  `export GEN_GIT_COMMIT=$(git rev-parse HEAD)` **on the laptop, on the branch you pushed** — `push_repo.sh`
  excludes `.git/`, so the cluster cannot resolve it and the launcher **refuses to start without it** (verified,
  §4.3).

### 3.1 Laptop, then login node
```bash
export GEN_GIT_COMMIT=$(git rev-parse HEAD)
export CSF3_USER=<username>; scripts/csf3/push_repo.sh
```
```bash
ssh $CSF3_USER@csf3.itservices.manchester.ac.uk
cd ~/scratch/CHLU && mkdir -p logs
```

### 3.2 The plumbing smoke — ⭐ do this FIRST; it has never been run
⭐ **One literal command line per submission.** The whole `--export` value is ONE quoted shell word (it contains
spaces); `sbatch --export` splits on **commas only**. ⛔ Do not build it in a variable and expand it — zsh does not
word-split an expanded variable and this has submitted garbage on this programme before.

```bash
DATASET=shapes3d PRETEXT=recon STAGE=encoder ARMS=clu SEEDS=0 PLAN=1 bash scripts/csf3/job_gpu_gen.sh
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=encoder,ARMS=clu,SEEDS=0,SMOKE=1,SKIP_GPU_PREFLIGHT=1,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -p serial -a 0-0 -t 0:20:00 --job-name=clu-gen-smoke scripts/csf3/job_gpu_gen.sh
```
And the same smoke **for the new stage** (`ALLOW_MISSING_UPSTREAM=1` because no `store` stage has ever run on the
cluster; it is recorded in the manifest as a non-default flag, so the exemption is not silent):
```bash
DATASET=shapes3d PRETEXT=recon STAGE=certificate ARMS=clu SEEDS="0 1 2" PLAN=1 bash scripts/csf3/job_gpu_gen.sh
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=certificate,ARMS=clu,SEEDS=0 1 2,SMOKE=1,SKIP_GPU_PREFLIGHT=1,ALLOW_MISSING_UPSTREAM=1,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -p serial -a 0-2%2 -t 0:20:00 --job-name=clu-gen-cert-smoke scripts/csf3/job_gpu_gen.sh
```
Then, before walking away:
```bash
squeue --me
ls -l logs/clu-gen-smoke-*.out logs/clu-gen-cert-smoke-*.out | wc -l    # expect 1 and 3
```
**Realised paths the smoke will produce** (`$SCRATCH` = `~/scratch`):
```
~/scratch/CHLU-gen/workspaces/shapes3d_recon/encoder/clu_s0/{logs/stage.log,artifacts/smoke_artifact.json,run_manifest.json}
~/scratch/CHLU-gen/workspaces/shapes3d_recon/certificate/clu_s0/{logs/stage.log,artifacts/smoke_artifact.json,run_manifest.json}
~/scratch/CHLU-gen/workspaces/shapes3d_recon/certificate/clu_s1/…
~/scratch/CHLU-gen/workspaces/shapes3d_recon/certificate/clu_s2/…
```
⭐ Note `certificate/`, **not** `cell/` — that is the whole point of this spoke.

### 3.3 ⚠ The real certificate run — the input-path gap, stated honestly
The launcher's plumbing is fixed, but **`exp_gen_certificate` does not read its inputs from the launcher's
workspace.** Read out of the module (lines 144–172):
- `bridge_root` defaults to `<repo>/.claude/outputs/gen-feature-bridge/runs`
- `store_runs` defaults to `<repo>/.claude/outputs/gen-store-grouping/runs`

and **`push_repo.sh` has `--exclude '.claude/'`** (line 42). ⇒ On CSF3 those directories **do not exist**, and the
module will raise `FileNotFoundError` ("no frozen bridge at …" / "no store plan at …"). That failure is **loud**,
not silent, so no wrong number can come out of it — but the run will not produce a result either.

⛔ **Therefore a real certificate submission is NOT ready to fire today.** One of these must happen first:
- **(a)** stage the frozen bridge + store artefacts onto scratch (rsync them explicitly — `push_repo.sh` will not),
  and point the payload at them with `EXTRA_ARGS`; or
- **(b)** run `STAGE=store` (module `chlu.experiments.exp_gen_store`, which **does** exist) through the launcher
  first and teach the certificate to read the launcher's workspace. That is a `chlu/` change — ⛔ out of my scope.

Once the artefacts are staged, the submission is (paths to be filled in by whoever stages them):
```bash
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=certificate,ARMS=clu,SEEDS=0 1 2,ALLOW_MISSING_UPSTREAM=1,EXTRA_ARGS=--store-runs <SCRATCH_STORE_RUNS> --bridge-root <SCRATCH_BRIDGE_RUNS> --data-root <SCRATCH_GEN_DATA>,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-2%2 -t 04:00:00 --job-name=clu-gen-cert scripts/csf3/job_gpu_gen.sh
```
**Realised paths:**
```
~/scratch/CHLU-gen/workspaces/shapes3d_recon/certificate/clu_s{0,1,2}/
    artifacts/seed_{0,1,2}/{cert_manifest.json,scores_correlation.json,scores_random.json,store_correlation.npz,store_random.npz}
    logs/stage.log   run_manifest.json
```
⚠ Compare with `gen-certificate.md` §10, which lists the same artefacts under **`…/cell/clu_s*/`**. That report's
submit block is superseded (reconciliation item 1).

**Verified locally** that this `EXTRA_ARGS` mechanism reaches the payload intact (§4.4): the words split correctly,
`--stage certificate` is passed **once** natively (no duplicate-wins hack), and the full string is recorded verbatim
in `run_manifest.json` → `payload.extra_args` and in `non_default_flags`.

⛔ Also note `ALLOW_MISSING_UPSTREAM=1` is a **real weakening of the upstream guard** — it is only correct here
because the store artefacts are being supplied out-of-band. If the store stage is ever run through the launcher,
drop the flag.

---

## 4. How I verified (commands + observed output)

### 4.0 Flag-provenance
⚠ **No quantitative scientific result is reported** (dial = instrument), so the usual training-config table does not
apply. The provenance that *does* apply:

| item | value |
|---|---|
| base | `main @ 3faef27` |
| branch tip | `78882cd` (`8940f1c` launcher+tests, `78882cd` README) |
| python / venv | **main venv reused** (`/Users/user/Desktop/CHLU/.venv/bin/python`, `PYTHONPATH=<worktree>`) — ⛔ no `uv sync` in the worktree, so no JAX drift |
| jax | 0.9.0 (unchanged; no worktree sync performed) |
| shellcheck | `uvx --from shellcheck-py shellcheck` |
| seeds | N/A — the launcher tests are deterministic shell/JSON; `SEEDS` appears only as a launcher axis |
| non-default flags exercised | `GEN_LOCAL=1`, `SMOKE=1`, `PLAN=1`, `PARTIAL=1`, `ALLOW_MISSING_UPSTREAM=1`, `PAYLOAD_MODULE`, `EXTRA_ARGS`, `GEN_ROOT`, `GEN_TASK_ID`, `CONCURRENCY`, `ARMS`, `SEEDS` |

### 4.1 `bash -n` + shellcheck
```
$ bash -n scripts/csf3/job_gpu_gen.sh        -> bash -n OK
$ uvx --from shellcheck-py shellcheck scripts/csf3/job_gpu_gen.sh ; echo rc=$?
rc=0
```
⭐ **Positive control** (the negative-result trap: a checker that isn't running "passes" everything). The same
invocation on a deliberately broken file **does** fire:
```
$ printf '#!/bin/bash\nfoo=1\necho $bar\n' > /tmp/bad.sh
$ uvx --from shellcheck-py shellcheck /tmp/bad.sh ; echo control_rc=$?
SC2034 foo appears unused · SC2154 bar referenced but not assigned · SC2086 double quote
control_rc=1
```

### 4.2 The tests

**`tests/test_gen_launcher.py`** (main venv both times):
```
BEFORE (worktree at 3faef27):  25 passed in 77.10s
AFTER  (branch 78882cd):       39 passed in 71.52s
```

**Full suite, before AND after** (`pytest -q -p no:cacheprovider --no-cov`, main venv,
`PYTHONPATH=<worktree>`, both on a quiet machine):

| | HEAD | result | wall |
|---|---|---|---|
| **BEFORE** | `3faef27` (worktree `../CHLU-launchfix-base2`, detached) | **2191 passed, 13 skipped**, `rc=0` | 52:35 |
| **AFTER** | `78882cd` (branch worktree) | **2205 passed, 13 skipped**, `rc=0` | 58:05 |

⭐ **Δ = +14 passed, +0 failed, +0 skipped — exactly the 14 tests I added, and nothing else moved.** After:
2205 + 13 = **2218 = the full collected count**, so nothing was deselected or errored out of collection.

⚠ **Honesty note on how these numbers were obtained — three runs were discarded before these two.**
1. A first AFTER attempt was **killed at 25 % by session teardown** — zero failures to that point, but **no
   verdict, so it is not a result** and is not reported as one.
2. It was competing with **five other agents' pytest suites**; the rate collapsed to ~6 tests/min (finding 5).
3. A first BEFORE attempt **ran against the wrong tree entirely** after a third party deleted the base worktree
   mid-flight (finding 5). ⛔ Discarded. The re-run was launched with a `cd <worktree> || exit 1` guard and a
   first log line recording `cwd` and `HEAD` (`cwd=/Users/user/Desktop/CHLU-launchfix-base2 head=3faef27`) so the
   tree under test is provable from the log rather than assumed.
LOG: `.claude/outputs/gen-launcher-fixes/suite_after.log`  ·  BEFORE:
`.claude/outputs/gen-launcher-fixes/suite_before.log`

**Blast radius of the diff, checked rather than assumed.** Three files changed; I grepped for every test that could
see them:
```
$ grep -rln "job_gpu_gen" tests/     -> tests/test_gen_launcher.py, tests/test_gen_bridge.py
$ grep -rln "csf3.*README" tests/    -> (none)
```
⚠ I had assumed only `test_gen_launcher.py` could reach my files; that was **wrong** —
`tests/test_gen_bridge.py` names `job_gpu_gen.sh` too. Inspected: it is a **docstring only**
(`test_decoder_stage_finds_the_encoder_in_the_launcher_layout`), asserting the
`<ws>/<stage>/<arm>_s<seed>/artifacts` layout convention. It never executes the script, and my change **preserves
that convention exactly** — `certificate` is simply a new `<stage>` slot in it. It passes in the after suite.
(The other 7 `csf3`-mentioning tests name `job_gpu_cluformer.sh` / `job_gpu_c3_seeds.sh`, which I did not touch.)


### 4.3 ⭐ Every pre-existing guard, fired live under `STAGE=certificate`
Not inspection — actual runs, `rc` and stderr copied verbatim:
```
--- unset DATASET, STAGE=certificate
rc=2  FATAL[job_gpu_gen]: DATASET is unset. No default: set DATASET=shapes3d|dsprites
--- no GEN_GIT_COMMIT
rc=2  FATAL[job_gpu_gen]: cannot resolve the git commit (push_repo.sh excludes .git). …
--- short array without PARTIAL   (1 arm x 3 seeds, SLURM_ARRAY_TASK_COUNT=2)
rc=2  FATAL[job_gpu_gen]: array has 2 tasks but ARMS x SEEDS = 3 — cells would be silently missing. …
--- workspace guard (GEN_ROOT into projects/)
rc=2  FATAL[job_gpu_gen]: refusing to run: RUN_DIR='…/projects/csf3_b1/workspaces/shapes3d_recon/certificate/clu_s0' is not under a CHLU-gen workspace
--- missing store upstream
rc=2  FATAL[job_gpu_gen]: upstream stage 'store' has no manifest at …/dsprites_cls/store/clu_s0/run_manifest.json — run it first (or set ALLOW_MISSING_UPSTREAM=1)
--- bad STAGE
rc=2  FATAL[job_gpu_gen]: STAGE='readout' is not one of: encoder decoder store cell certificate
```
Plus, from the local chain run (`GEN_LOCAL=1 SMOKE=1`, enc→dec→store→certificate):
```
[gen] === shapes3d_recon / certificate / clu_s0  (task 0/3) ===
[gen] run_dir=…/workspaces/shapes3d_recon/certificate/clu_s0
[gen] commit=8940f1c  module=chlu.experiments.exp_gen_certificate  smoke=1
[gen] upstream ok: …/workspaces/shapes3d_recon/store/clu_s0/run_manifest.json
```
and the realised tree — ⭐ **no `cell/` directory exists at all**:
```
…/shapes3d_recon/{encoder,decoder,store,certificate}/clu_s0
```

### 4.4 The `EXTRA_ARGS` mechanism the real submission depends on
Payload replaced by a trivial argv-echo module, everything else real:
```
ARGV: ['…/echoargs.py', '--dataset', 'shapes3d', '--pretext', 'recon', '--stage', 'certificate',
       '--arm', 'clu', '--seed', '0', '--out', '…/certificate/clu_s0/artifacts',
       '--store-runs', '/scratch/gen/store', '--bridge-root', '/scratch/gen/bridge',
       '--data-root', '/scratch/gen/data']
run_manifest.json payload.extra_args = "--store-runs /scratch/gen/store --bridge-root … --data-root …"
non_default_flags                    = [ARMS=clu, GEN_ROOT=…, GEN_LOCAL=1, EXTRA_ARGS=…, PAYLOAD_MODULE=echoargs]
```
⭐ `--stage certificate` appears **once**, natively. The old workaround relied on argparse's "later `--stage` wins"
with the flag passed **twice**; that dependency is gone.

### 4.5 The PLAN line is one literal command line
```
$ DATASET=shapes3d PRETEXT=recon STAGE=certificate ARMS=clu SEEDS="0 1 2" PLAN=1 bash scripts/csf3/job_gpu_gen.sh
=== gen launch plan: shapes3d_recon stage=certificate ===
arms  (1): clu
seeds (3): 0 1 2
tasks: 3   throttle: %2
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=certificate,ARMS=clu,SEEDS=0 1 2,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-2%2 -t <WALLCLOCK> scripts/csf3/job_gpu_gen.sh
task 0: arm=clu seed=0 -> …/workspaces/shapes3d_recon/certificate/clu_s0
task 1: arm=clu seed=1 -> …/workspaces/shapes3d_recon/certificate/clu_s1
task 2: arm=clu seed=2 -> …/workspaces/shapes3d_recon/certificate/clu_s2
```
No backslash continuation inside `--export`; the whole value is one quoted word.

---

## 5. Findings (beyond "it works")

1. ⭐ **The `cell/` corruption was structural, not a slip.** `run_dir_for()` builds the path from `$STAGE` alone,
   and `$STAGE` was also the only thing handed to the payload. With no `certificate` in the whitelist, the *only*
   way to run the certificate was to lie to the path in order to tell the truth to the payload. Adding the stage
   removes the trade-off; there is no remaining way to express the mislabelled combination.
2. ⭐ **`certificate`'s upstream is `store`, and that is a real design decision, not bookkeeping.** Making it a
   child of `cell` would have made a cell failure block a result that has no dependency on the cell — and
   `chlu/experiments/exp_gen_cell.py` **does not exist yet**, so it would have made the certificate unrunnable.
3. ⚠ **Three of the five launcher stages have no payload module.** `exp_gen_encoder`, `exp_gen_decoder` and
   `exp_gen_cell` are all **MISSING**; only `exp_gen_store`, `exp_gen_certificate` (and `exp_gen_bridge`, which is
   not a launcher stage) exist. The launcher's `find_spec` preflight aborts loudly on the missing ones before any
   wallclock is spent — so this is safe, but it means **the `encoder`/`decoder` stages cannot run a real payload
   today**, and the §3.2 smoke must use `SMOKE=1`.
4. ⚠ **The payload's input paths do not come from the launcher workspace** (§3.3): `bridge_root` and `store_runs`
   default into `<repo>/.claude/outputs/…`, which `push_repo.sh` explicitly excludes from the rsync. This is the
   one thing standing between the fixed launcher and a real cluster certificate run, and it is under `chlu/`
   (out of my scope).
5. ⭐ **The full-suite cost, and two environment hazards worth logging.** 2218 tests, ~58 min on a quiet machine.
   - **Concurrency collapse.** With **six** agent pytest suites running at once the rate fell to ~6 tests/min
     (~6 h projected). Serialising is strictly better than overlapping; I killed my own concurrent baseline run
     rather than make it worse for everyone.
   - ⚠ **A third party deleted my base worktree mid-run.** `../CHLU-genfix-base` vanished between creation and
     use (another agent pruning worktrees). My `cd` into it failed, and because the `cd` was **unguarded** the
     suite silently ran in the **shared main checkout** (on *another* agent's branch, `7fcef50`) with a
     nonexistent `PYTHONPATH` — i.e. it measured the wrong tree entirely. ⛔ **That run was discarded, not
     reported.** Verified afterwards that the shared checkout was left clean (`git status` empty — `-p
     no:cacheprovider --no-cov` meant nothing was written). ⭐ **Lesson for every spoke: `cd <worktree> || exit 1`.
     An unguarded `cd` in a background job does not fail, it relocates — and a suite that "passed" in the wrong
     tree looks exactly like a suite that passed.**

---

## 6. Git footprint

- **Worktree:** `../CHLU-gen-launchfix` (fresh, as the task specified). ⭐ The shared main checkout was on **another
  agent's** branch (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring @ 7fcef50`) when I started — I never
  edited it (protocol §3.2), and I re-verified `git status` clean at the end. Throwaway base worktrees
  (detached @ `3faef27`) were used for the before-state runs: `../CHLU-genfix-base` (deleted out from under me by
  another agent — see finding 5) and its replacement `../CHLU-launchfix-base2`.
- **Branch:** `agent/experiment-engineer/gen-launcher-fixes`, base `main @ 3faef27` (as specified).
- **Commits:**
  | hash | subject |
  |---|---|
  | `8940f1c` | `[experiment-engineer] add a certificate stage to the gen launcher, with its own workspace` |
  | `78882cd` | `[experiment-engineer] document the certificate stage in the csf3 README` |
- **Files touched (tracked): 3 — none under `chlu/`.**
  ```
  scripts/csf3/README.md        +53 -2
  scripts/csf3/job_gpu_gen.sh   +40 -13
  tests/test_gen_launcher.py   +204 -6
  ```
- **Untracked artefacts:** `.claude/outputs/gen-launcher-fixes.md` (this file) and
  `.claude/outputs/gen-launcher-fixes/suite_{before,after}.log`.
- **Non-repo doc edited (as the task assigned):** `.claude/csf3-operations.md` §8.3, §11, §12 **only**.
- **Rebase:** onto my named base `main @ 3faef27` — a **no-op**, the base has not moved.
  ⚠ **Local `main` advanced to `1bcfcb9`** (`[hub] merge gen-baselines`) *during* this spoke.
  `git diff --name-only 3faef27..main -- scripts/csf3 tests/test_gen_launcher.py` → **0 files**, and
  `git merge-tree --write-tree main agent/experiment-engineer/gen-launcher-fixes` → **rc=0, no conflicts**
  (tree `ce9acef`). ⭐ I deliberately did **not** rebase onto `1bcfcb9`: it would have moved HEAD out from under the
  running full-suite pass (the "a suite result needs a stable HEAD" lesson). The Hub's merge is clean either way.
- ⛔ **Not pushed. No PR. Branch left for review.**

---

## 7. Open questions / follow-ups / risks

1. ⚠ **(risk, blocking a real run)** `exp_gen_certificate`'s `bridge_root`/`store_runs` default into
   `<repo>/.claude/outputs/…`, which `push_repo.sh` excludes. Needs either an explicit artefact-staging step plus
   `EXTRA_ARGS`, or a `chlu/` change to read the launcher workspace. ⛔ Out of my scope; **no owner**.
2. ⚠ **(R26 ruling 7, second half, no owner)** `gen_data.data_root` must fail loudly with no default. Untouched —
   it is under `chlu/`. See reconciliation item 2 at the top.
3. **(follow-up)** `exp_gen_encoder` / `exp_gen_decoder` / `exp_gen_cell` do not exist. The launcher aborts loudly
   on them, so nothing can silently no-op, but the stage chain is currently only runnable at `store` and
   `certificate`.
4. **(question for the Hub)** Should `certificate` default to `ARMS=clu` when `STAGE=certificate`? The payload
   scores every arm inside one job, so the 4-arm default would submit 12 tasks that each redo the same work. I did
   **not** add a per-stage `ARMS` default — a stage-dependent default is exactly the kind of silent axis this
   launcher exists to prevent — so the submit lines pass `ARMS=clu` explicitly and it is recorded as non-default.
   Flagging it in case the Hub prefers a loud *refusal* (e.g. "certificate with >1 arm is almost certainly wrong").
5. **(risk, low)** The exit-trap slurm-stream copy, the GPU preflight and `env.sh` sourcing are still
   never-executed-on-a-node code paths. The `-p serial` smoke exercises the first; only a `gpuA` job exercises the
   rest.

---

## Proposed handover updates (for the Hub)

**§7 Known Issues — proposed new entries:**
- **7.NEW-a [RESOLVED 2026-09-13 by `gen-launcher-fixes`] the gen launcher had no `certificate` stage.** The
  workaround (`STAGE=cell` + `EXTRA_ARGS=--stage certificate`) wrote certificate artefacts into the **cell's**
  workspace and corrupted another stage's provenance. `certificate` is now a first-class stage with its own
  workspace path and its own default module, upstream `store`; 14 tests (39 total) keep the workaround from
  returning. ⛔ **`.claude/outputs/gen-certificate.md` §10's submit block and its `…/cell/clu_s*/` paths are
  superseded** — the realised paths are `…/certificate/clu_s*/`.
- **7.NEW-b [OPEN, no owner] the certificate's cluster inputs are unreachable on CSF3.**
  `exp_gen_certificate.bridge_root`/`store_runs` default to `<repo>/.claude/outputs/{gen-feature-bridge,
  gen-store-grouping}/runs`, and `scripts/csf3/push_repo.sh` has `--exclude '.claude/'`. A real certificate job on
  the cluster will `FileNotFoundError` (loudly — no wrong number can escape) until the artefacts are staged and
  passed via `EXTRA_ARGS`, or the module learns the launcher workspace. **Blocks the first real gen cluster run.**
- **7.NEW-c [OPEN] three launcher stages have no payload module:** `exp_gen_encoder`, `exp_gen_decoder`,
  `exp_gen_cell` are MISSING. `find_spec` preflight aborts before wallclock is spent, so this is fail-loud, not
  silent — but `encoder`/`decoder` cluster smokes must use `SMOKE=1`.
- **7.NEW-d [standing, honesty]** ⛔ **Nothing of the generative pivot has ever run on CSF3** — zero jobs, zero job
  ids, the `-p serial` plumbing smoke included. Every gen number to date (the certificate barrier 0.9146 vs naive
  0.5630 among them) was produced **locally on CPU**. R26 ruling 7: **the cluster smoke still gates the cell.**

**`.claude/csf3-operations.md`** — I already edited **§8.3** (certificate stage, two-leaf DAG, submit block, banned
workaround), **§11** (split into what is [LOCAL]-verified vs. ⛔ never-observed-on-cluster; added the 2FA fact) and
**§12** ("has STILL not been launched", plus the "produced locally on CPU" clarification). ⛔ I marked **nothing**
verified that I did not see. §5's script table still says "the generative pivot (added 2026-09-09)"; the Hub may
want to append "+ certificate stage 2026-09-13" — I left it alone as it is outside my assigned sections.

**No `chlu/config.py` default changed** (no code under `chlu/` was touched), so §3 needs no edit from me.
