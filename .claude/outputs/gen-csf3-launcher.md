# gen-csf3-launcher — experiment-engineer report

**Task + acceptance criterion:** deliver ONE CSF3 launch script (`scripts/csf3/job_gpu_gen.sh`) covering all four
generative-pivot stages plus the fresh output tree it writes into, with loud failures on defaultless axes, a populated
`run_manifest.json`, CPU tests, and **a real CSF3 smoke run through it**.

**Status: partial** — everything is delivered and locally verified **except acceptance item 2 (the CSF3 smoke run),
which is `NOT RUN`: the cluster is unreachable from this laptop.** See §"Acceptance item 2" below for exactly what was
tried and the one command the Head needs to run.

**⚠ DOWNSTREAM ITEMS NEEDING AN OWNER (protocol §5 corollary — flagged in the first 10 lines):**
1. **The unrun CSF3 smoke.** Needs a Head action (VPN + one 20-min `-p serial` sbatch), then a Hub re-check.
2. **The payload CLI contract.** Every gen stage module must accept
   `--dataset --pretext --stage --arm --seed --out`. The spokes writing `exp_gen_{encoder,decoder,store,cell}`
   must be told this in their task files, or the launcher will abort on them.
3. **`GEN_GIT_COMMIT=$(git rev-parse HEAD)` must appear in every gen submit line the Hub writes** — the launcher
   refuses to run without it.
Suite numbers below were measured against the **named base `b0257bc`**; `main` advanced to `9e87f72` mid-flight
(`gen-data-splits` merged) and my branch is rebased onto it cleanly — details in the git footprint.

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — instrument.* No CLU number, no store, no arm, no performance claim. Nothing scientific ran.
- **Laundering control:** N/A.
- **Falsifies:** N/A. The failure mode is a script that looks right and has never run — which is precisely the part
  I could not close (item 2).
- No flag-provenance table and no `PREREG.md`: this report contains **no measured quantity** (the only numbers are
  test counts and exit codes).

## Reading-order note
The task file says ⛔ *"Read no campaign ledger, charter or handover — different programme, it will mislead you."*
That overrides the spoke prompt's standing "read `.claude/handover_context.md`" instruction, so **I did not read the
handover.** I read `AGENT_PROTOCOL.md`, the task file, and the six named `scripts/csf3/` files (plus
`job_gpu_cluformer.sh` for the current in-repo idiom). Consequence: I own no file under `chlu/`, and I have made no
assumption about any CLU internals.

## What I did
- **`scripts/csf3/job_gpu_gen.sh` (new, 441 lines)** — the one launcher, all four stages, Slurm array over
  **arms × seeds**, one A100 per task, same `#SBATCH` shape and `%N` throttle idiom as `job_gpu_array_seeds.sh`,
  same `module purge` → `source env.sh` → GPU preflight → payload → artifacts structure as `job_gpu_single.sh`.
  No new way of talking to the cluster: `env.sh`, `push_repo.sh` and rsync are reused as-is.
- **`scripts/csf3/README.md`** — appended `## 7. Generative pivot` (**99 insertions, 0 deletions**; verified below
  that not one pre-existing line changed).
- **`tests/test_gen_launcher.py` (new, 396 lines, 25 tests, 4.35 s, CPU, no JAX import).**

### Axes and their defaults
| Var | Values | Default | Behaviour |
|---|---|---|---|
| `DATASET` | `shapes3d` \| `dsprites` | **none** | unset ⇒ `FATAL … DATASET is unset`, exit 2 |
| `PRETEXT` | `recon` \| `cls` | **none** | unset ⇒ exit 2. Carried, not privileged: `recon` and `cls` take identical paths |
| `STAGE` | `encoder`\|`decoder`\|`store`\|`cell` | **none** | unset ⇒ exit 2 |
| `ARMS` | subset of `clu store_removed lace knn` | all four | see assumption A1 |
| `SEEDS` | ints | `0 1 2` | non-integer ⇒ exit 2 |
| `CONCURRENCY` | 1–4 | `2` | >4 rejected (CSF3 free-tier 4-GPU cap) |

Out-of-whitelist values are rejected too (`DATASET='mnist'`, `ARM 'oracle'`, `SEED 'one'`, `CONCURRENCY=9`).
Array index = `arm_idx * n_seeds + seed_idx`; `PLAN=1` prints the exact `sbatch … -a 0-N%C` line **and every
task→path mapping**, so the grid is auditable before a single GPU is claimed.

**A1 (stated assumption).** The task names `ARM` as an axis but only requires `DATASET`/`PRETEXT`/`STAGE` to fail
loudly. `ARMS` therefore defaults to the **complete** four-arm set: an exhaustive default cannot decide a result,
whereas a *narrowed* arm list is necessarily explicit and is recorded verbatim in `non_default_flags`. If the Hub
prefers `ARMS` to be defaultless too, it is a two-line change.

### The fresh tree (exactly the Hub convention)
```
$SCRATCH/CHLU-gen/workspaces/<dataset>_<pretext>/<stage>/<arm>_s<seed>/
    logs/              stage.log  +  a copy of this task's slurm .out/.err
    artifacts/
    run_manifest.json
```
`$SCRATCH` falls back to `$HOME/scratch` (CSF3 does not export `$SCRATCH`; `env.sh` uses `$HOME/scratch`), and
`GEN_ROOT` overrides the root for tests. Two hard guards: the script **refuses to run** unless `RUN_DIR` matches
`*/CHLU-gen/workspaces/*`, and refuses outright if it matches `*/projects/*`. It writes nothing anywhere else.

### The manifest (the actual deliverable)
`run_manifest.json` is written **twice** — `status:"running"` before any work, rewritten from an `EXIT` trap with
`ok`/`failed` + `exit_code`. Fields: `schema`, `status`, `exit_code`, `started_at`/`finished_at`, `axes`
(dataset/pretext/stage/arm/seed), `grid` (arms, seeds, concurrency, array_task_id, n_tasks), `git_commit`,
`versions` (python/jax/jaxlib/equinox/optax/chlu), `non_default_flags`, `payload` (module, extra_args, smoke),
`slurm` (job_id, array_job_id, array_task_id, array_task_count, partition, nodelist, gpus, CUDA_VISIBLE_DEVICES),
`paths`, `host`. Versions come from `importlib.metadata`, **not** `import jax` — so writing the manifest costs
milliseconds and never pays the JAX import.

**A run with no resolvable git commit is refused.** `push_repo.sh` excludes `.git/`, so the cluster cannot work the
commit out itself; the launcher resolves `GEN_GIT_COMMIT` → local `git rev-parse` → a `$CLU_REPO/GIT_COMMIT` stamp,
and dies naming the fix if all three fail. The documented submit line carries
`GEN_GIT_COMMIT=$(git rev-parse HEAD)`.

### Silent-miss guards (the failure modes this repo has been bitten by)
- `-a` range not covering arms × seeds ⇒ **refused** (`cells would be silently missing`) unless `PARTIAL=1`.
- array task id ≥ grid size ⇒ refused; no array context at all ⇒ refused.
- each stage requires its upstream (`encoder→decoder→store→cell`) manifest to be `status=="ok"` **for the same arm
  and seed**; missing or failed upstream ⇒ refused (`ALLOW_MISSING_UPSTREAM=1` to override).
- a stage whose payload module does not exist yet ⇒ **aborts naming the module**, never a quiet no-op. (The four
  stage modules `chlu.experiments.exp_gen_{encoder,decoder,store,cell}` are referenced **by name only** — I created
  no file under `chlu/`. `PAYLOAD_MODULE` overrides them; `SMOKE=1` bypasses them.)
- a failed run still writes its manifest with `status:"failed"`, so a dead cell is traceable rather than absent.

## How I verified (real commands, real output)
```
$ bash -n scripts/csf3/job_gpu_gen.sh                                  -> clean
$ uvx --from shellcheck-py shellcheck -s bash scripts/csf3/job_gpu_gen.sh
  ShellCheck 0.11.0  ->  EXIT=0, zero warnings
$ .venv/bin/python -m pytest tests/test_gen_launcher.py -q --no-cov
  25 passed in 4.35s
$ .venv/bin/python -m ruff check / ruff format tests/test_gen_launcher.py  -> clean
```
Shellcheck was not installed on this machine; I obtained 0.11.0 via `uvx --from shellcheck-py` (network was up).
The script is bash-3.2 safe (no associative arrays, no `${var,,}`) — checked against the laptop's GNU bash 3.2.57.

**Loud failures, observed verbatim** (run with a *clean* env — note `env -u X X=1` does **not** unset, a trap the
tests avoid by building the environment dict explicitly):
```
FATAL[job_gpu_gen]: DATASET is unset. No default: set DATASET=shapes3d|dsprites            exit=2
FATAL[job_gpu_gen]: PRETEXT is unset. No default: set PRETEXT=recon|cls                    exit=2
FATAL[job_gpu_gen]: STAGE is unset. No default: set STAGE=encoder|decoder|store|cell       exit=2
FATAL[job_gpu_gen]: DATASET='mnist' is not one of: shapes3d dsprites                       exit=2
FATAL[job_gpu_gen]: ARM 'oracle' is not one of: clu store_removed lace knn                 exit=2
FATAL[job_gpu_gen]: array task 12 is outside the grid (ARMS x SEEDS = 12). …               exit=2
FATAL[job_gpu_gen]: array has 3 tasks but ARMS x SEEDS = 12 — cells would be silently missing. …
FATAL[job_gpu_gen]: upstream stage 'encoder' has no manifest at …/encoder/clu_s0/run_manifest.json …
FATAL[job_gpu_gen]: cannot resolve the git commit (push_repo.sh excludes .git). …
FATAL[job_gpu_gen]: refusing to run: RUN_DIR='…/projects/csf3_b1/workspaces/…' is not under a CHLU-gen workspace
```

**A full local dry run through the real path plumbing** (`GEN_LOCAL=1 SMOKE=1`, task 4 of a 12-task grid), realised
tree:
```
/tmp/genrun.E3Gn/CHLU-gen/workspaces/shapes3d_recon/encoder/store_removed_s1/artifacts/smoke_artifact.json
/tmp/genrun.E3Gn/CHLU-gen/workspaces/shapes3d_recon/encoder/store_removed_s1/logs/stage.log
/tmp/genrun.E3Gn/CHLU-gen/workspaces/shapes3d_recon/encoder/store_removed_s1/run_manifest.json
```
and its manifest (abridged; **this is the shape the cluster run will produce**):
```json
{"schema":"chlu-gen-run-manifest/1","status":"ok","exit_code":0,
 "axes":{"arm":"store_removed","dataset":"shapes3d","pretext":"recon","seed":1,"stage":"encoder"},
 "grid":{"arms":["clu","store_removed","lace","knn"],"seeds":[0,1,2],"concurrency":2,
         "array_task_id":4,"n_tasks":12},
 "git_commit":"b0257bc8536ba00ec370c703ff072157949a2cba",
 "non_default_flags":["GEN_ROOT=/tmp/genrun.E3Gn/CHLU-gen","SMOKE=1","GEN_LOCAL=1"],
 "slurm":{"array_job_id":"999000","array_task_count":"12","array_task_id":"4","job_id":"999001",
          "partition":"gpuA"},
 "started_at":"2026-09-09T08:18:29.518236+00:00","finished_at":"2026-09-09T08:18:29.800169+00:00"}
```
With `GEN_PYTHON=<main venv python>` the `versions` block resolves for real:
`{"chlu":"0.2.4","equinox":"0.13.4","jax":"0.9.0","jaxlib":"0.9.0","optax":"0.2.6","python":"3.11.13"}`
— i.e. the manifest reports the **resolved** JAX version, in <1 s, with no JAX import.

**Stage chain, run for real locally:** encoder→decoder→store→cell all `ok` in order; `decoder` on a *missing*
encoder ⇒ exit 2 with a `failed` manifest; `decoder` on an encoder manifest flipped to `status:"failed"` ⇒
`upstream stage 'encoder' did not finish ok`.

**Slurm-stream capture** (the only cluster-only line) simulated with a fake `$CLU_REPO/logs/` + the array env vars:
`clu-gen-777000_2.out` / `.err` land in `…/encoder/clu_s2/logs/` alongside `stage.log`. Covered by a test.

**README append is pure append:** `diff` of the pre-edit file against the first N lines of the post-edit file is
empty (`EXISTING README LINES UNCHANGED`); `git diff --stat` = `99 ++…, 1 file changed, 99 insertions(+)`.

### Suite (acceptance item 4) — main venv, per protocol §4
- **Before (base `main` @ `b0257bc`, worktree, main venv):** **`1919 passed, 36 warnings in 4755.92s (1:19:15)`** — 0 failed.
- **After (my branch, same base content + my 3 files):** **`1944 passed, 29 warnings in 3778.86s (1:02:58)`** — 0 failed, exit 0.
- **Delta: exactly +25**, i.e. only my new `tests/test_gen_launcher.py`. No pre-existing test moved.
Command (both): `cd <worktree> && PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q -p no:randomly`.
The **main venv** was reused (no worktree `uv sync`), so the ~7-test extras drop does not apply here; resolved
`jax 0.9.0` (confirmed via the manifest's own `importlib.metadata` read of that venv).

## ⛔ Acceptance item 2 — the CSF3 smoke run: **NOT RUN**
This is a genuine gap, not a pass and not a failure.

What I tried, verbatim:
```
$ nslookup csf3.itservices.manchester.ac.uk
  ** server can't find csf3.itservices.manchester.ac.uk: NXDOMAIN
$ ssh -o ConnectTimeout=8 -o BatchMode=yes csf3 'echo CONNECTED'
  ssh: Could not resolve hostname csf3.itservices.manchester.ac.uk: nodename nor servname provided
$ ssh -o ConnectTimeout=8 -o BatchMode=yes csf4 'echo CONNECTED4'
  ssh: connect to host csf4.itservices.manchester.ac.uk port 22: Operation timed out
```
General internet is up (`ping 8.8.8.8` = 53.9 ms). `~/.ssh/config` has `Host csf3 … User x10719pj`, and
`GlobalProtect.app` **is running** (`PanGPS`, pid 3828) but **not connected** — `utun4` is `POINTOPOINT,RUNNING` with
no `UP` flag, and `csf3` resolves only on campus DNS. Establishing the tunnel needs interactive UoM SSO/Duo, which an
agent cannot do. Retried twice, ~40 min apart, same result.

**The exact commands the Head should run first** (in order; the first is cheap and is the item-2 test):
```bash
# LAPTOP, VPN up:
export CSF3_USER=x10719pj GEN_GIT_COMMIT=$(git rev-parse HEAD)
scripts/csf3/push_repo.sh
ssh csf3
# CSF3:
cd ~/scratch/CHLU && mkdir -p logs
DATASET=shapes3d PRETEXT=recon STAGE=encoder PLAN=1 bash scripts/csf3/job_gpu_gen.sh   # audit the grid
sbatch --export=ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=encoder,SEEDS=0,ARMS=clu,SMOKE=1,\
SKIP_GPU_PREFLIGHT=1,GEN_GIT_COMMIT=$GEN_GIT_COMMIT -p serial -a 0-0 -t 0:20:00 \
       scripts/csf3/job_gpu_gen.sh
# then check:
cat ~/scratch/CHLU-gen/workspaces/shapes3d_recon/encoder/clu_s0/run_manifest.json
ls  ~/scratch/CHLU-gen/workspaces/shapes3d_recon/encoder/clu_s0/{logs,artifacts}
```
Expected: `status:"ok"`, a real `slurm.job_id`, `versions.jax == "0.9.0"`, `smoke_artifact.json` in `artifacts/`,
and `clu-gen-<jobid>_0.out/.err` + `stage.log` in `logs/`. Drop `SMOKE=1 SKIP_GPU_PREFLIGHT=1 -p serial` and use
`-p gpuA -G 1` for the A100 version once a stage module exists.
The plumbing that the cluster adds over the local dry run is exactly three things — `env.sh`/venv activation, the
`nvidia-smi`+JAX-GPU preflight, and the real `$SLURM_*` values — all of which are already exercised by
`job_gpu_single.sh`/`job_gpu_array_seeds.sh` on this cluster.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-csf3-launcher`, off **`main` @ `b0257bc`** (verified `b0257bc` is `main`'s
  tip and an ancestor; the shared checkout was on **another agent's branch**, `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`
  @ `7fcef50`, so per §3.2 I worked entirely in a fresh worktree `../CHLU-gen-launcher` and touched nothing in the
  main checkout).
- **Commits (6, post-rebase, `main..agent/experiment-engineer/gen-csf3-launcher`):**
  - `ffeb802` one CSF3 launch script + fresh tree for the generative pivot (`scripts/csf3/job_gpu_gen.sh`,
    `tests/test_gen_launcher.py`)
  - `4a42d07` runbook: document the generative-pivot launcher (append-only) (`scripts/csf3/README.md` +99, −0)
  - `731e524` test: the launcher copies its slurm streams into the run's `logs/`
  - `117b0aa` runbook: 25 launcher tests, not 24
  - `1d401be` header: submit example without a misleading comment continuation
  - `6c4f238` PLAN prints one quoted `--export` word (sbatch splits on commas only; a space-containing
    `ARMS=` value must survive as one shell word — pinned by a test)
- **Files touched (3, all owned by this task; `3 files changed, 943 insertions(+), 0 deletions`):**
  `scripts/csf3/job_gpu_gen.sh` (new, 444), `tests/test_gen_launcher.py` (new, 400),
  `scripts/csf3/README.md` (+99, append-only).
  **Zero files under `chlu/`. Zero files under `projects/`.** No push, no PR, no merge.
- **Rebase onto `main`:** ⚠ **`main` moved under me mid-flight**, `b0257bc` → `9e87f72` (the Hub merged
  `gen-data-splits`: `chlu/data/{shapes3d,dsprites,gen_splits}.py` + `tests/test_gen_data_splits.py`, +1764 lines).
  I rebased onto the **named base, local `main`** (never `origin/main`, which is stale at `40c2f31`): all 6 commits
  replayed cleanly, **zero conflicts** — `git diff --name-only b0257bc..main` has **no overlap** with my three files.
  Commit hashes therefore changed; the ones above are post-rebase. Post-rebase re-verification:
  `bash -n` + `shellcheck` clean, `tests/test_gen_launcher.py` 25 passed in 3.69 s, and
  `pytest --collect-only` at the rebased HEAD collects **1968** tests (= my 1944 + gen-data-splits' 24), consistent.
- **Suite-timing caveat (honest):** the 1919/1944 pair was measured against the **named base `b0257bc`**, which is
  what the acceptance criterion asks for; it is a like-for-like pair. It is **not** a full-suite run of the rebased
  `9e87f72` HEAD — I did not re-run 1:03 h of suite against a base that had already moved once and could move again.
  The rebased HEAD is verified by collection count + my own test file only.
- **Worktree removal:** verified from the MAIN repo first (`git -C /Users/user/Desktop/CHLU log --oneline
  main..agent/experiment-engineer/gen-csf3-launcher` → all 6 commits, diffstat `3 files changed, 943 insertions(+)`,
  worktree clean), **then** `git worktree remove ../CHLU-gen-launcher`. The parked worktrees
  `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` were **not touched**, and the shared main checkout
  (on `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`) was **never edited**.

## Open questions / follow-ups / risks
1. **The one real risk is item 2.** Every line of the script has run on the laptop, but no line has run under Slurm.
   The cluster-specific surface is small and reused from working scripts, but "looks right and never ran" is exactly
   the failure the task named. **Recommend the Hub gate the first real gen stage behind the 20-minute `-p serial`
   `SMOKE=1` submission above**, not behind a GPU run.
2. **Payload contract.** Non-smoke runs invoke
   `python -u -m <module> --dataset --pretext --stage --arm --seed --out <run_dir>/artifacts $EXTRA_ARGS`.
   The spokes writing `exp_gen_{encoder,decoder,store,cell}` must accept exactly those six flags, or must be launched
   with `PAYLOAD_MODULE`/`EXTRA_ARGS`. **This is an interface other spokes have to match — worth stating in the Hub's
   next task files.** They should also write their real artifacts into `--out` and nothing outside it.
3. **`sync_project.sh` is not extended** (I do not own it) — it only moves `projects/<name>/`, which the gen pivot
   never uses. The launcher prints the correct per-run `rsync` line into the *consuming spoke's own* output dir
   instead. If the Hub wants a first-class `sync_gen.sh`, that is a separate, small task.
4. **`GEN_GIT_COMMIT` is mandatory** and easy to forget. It fails fast and loudly rather than recording `"unknown"`,
   which is the right trade for a provenance record — but the Hub should put it in every gen submit line it writes.
5. `ARMS` defaulting to all four arms (assumption A1) — flag if the Hub wants it defaultless.

## Proposed handover updates (for the Hub)
- **New instrument.** `scripts/csf3/job_gpu_gen.sh` is the **single** launcher for the generative pivot
  (stages `encoder|decoder|store|cell`; axes `DATASET`, `PRETEXT`, `ARMS`, `SEEDS`, `CONCURRENCY`), documented in
  `scripts/csf3/README.md` §7. `DATASET`/`PRETEXT`/`STAGE` have **no defaults** and abort when unset.
- **New path convention (other spokes must use it):**
  `$SCRATCH/CHLU-gen/workspaces/<dataset>_<pretext>/<stage>/<arm>_s<seed>/{logs,artifacts,run_manifest.json}`.
  The gen pivot **never** writes to `projects/` or to any C2/C3 workspace; the launcher enforces both.
- **New provenance artifact:** `run_manifest.json`, schema `chlu-gen-run-manifest/1`. Every later gen number should
  cite its manifest; the flag-provenance tables for the gen pivot can be generated from these files.
- **New submit-line requirement:** `GEN_GIT_COMMIT=$(git rev-parse HEAD)` in every gen `sbatch --export`
  (`push_repo.sh` excludes `.git/`, and the launcher refuses to run without a resolvable commit).
- **New known issue:** the CSF3 smoke of this launcher is **UNRUN** — the laptop cannot reach CSF3 without
  GlobalProtect, which agents cannot establish. Track it until the Head runs the `-p serial SMOKE=1` job.
- **Test-count delta:** `tests/test_gen_launcher.py` adds **25** CPU tests (~4 s, no JAX).
