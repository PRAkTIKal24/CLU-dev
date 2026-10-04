# Task: gen-csf3-launcher — ONE launch script and a fresh folder tree for the generative pivot

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-csf3-launcher.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all of it) · `scripts/csf3/README.md`, `env.sh`, `job_gpu_array_seeds.sh`, `job_gpu_single.sh`, `job_cpu_smoke.sh`, `sync_project.sh`, `push_repo.sh`. ⛔ Read no campaign ledger, charter or handover — different programme, it will mislead you.
- **Base:** `main @ b0257bc` · **Branch:** `agent/experiment-engineer/gen-csf3-launcher` · **Worktree:** yes, fresh (`../CHLU-gen-launcher`). ⛔ Do not reuse or touch the existing cold worktrees `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` — they are parked, not free.
- **Precondition:** none. Spawnable now, in parallel with `gen-data-splits`.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — instrument.* ⛔ No CLU number, no store, no arm, no performance claim. Nothing scientific runs here.
- **Laundering control:** N/A.
- **Falsifies:** N/A. The failure mode is a script that looks right and has never actually run.

## The one job
Deliver **one** new CSF3 launch script covering every stage of the generative pivot, plus the fresh output-folder tree
it writes into, and **prove it by running a smoke job through it end to end.**

## Head ruling that defines this task (2026-09-09)
> Smoke locally; launch on CSF3 in line with the existing scripts — but with **a single new launch script** and
> **fresh folders** for the gen pivot.

⛔ **Single** means one script, not a family. ⛔ **Fresh** means the gen pivot never writes into a C2/C3 workspace path.
⭐ **In line with** means reuse `env.sh`, the module/venv setup, the array/throttle idiom and the sync scripts —
do not invent a second way of talking to the cluster.

## Files you own (nobody else touches these)
- `scripts/csf3/job_gpu_gen.sh` — **new, the one script.**
- `scripts/csf3/README.md` — **append one `## Generative pivot` section only.** ⛔ Do not reformat or edit any existing line.
- `tests/test_gen_launcher.py` — new.
⛔ You own **no** file under `chlu/`. If you think you need one, stop and report instead.

## The parameterisation the script must carry
Stages, run independently and in this dependency order: `encoder` · `decoder` · `store` · `cell`.
Axes: `DATASET` (`shapes3d` | `dsprites`) · `PRETEXT` (`recon` | `cls`) · `ARM` (`clu` | `store_removed` | `lace` | `knn`)
· `SEEDS` (space-separated, default `0 1 2`) · `CONCURRENCY` (array `%N` throttle, default 2).
- Sized as a SLURM array over arms × seeds, exactly as `job_gpu_array_seeds.sh` does it.
- ⛔ **No axis has a silent default that could decide a result.** `DATASET`, `PRETEXT` and `STAGE` must **fail loudly if unset.**
  Seeds and concurrency may default. This repo has been bitten by an inherited default before.
- ⚠ `PRETEXT=recon` is the primary; `cls` exists only as a diagnostic. The script must not privilege or hide either —
  it just carries the axis.

## The fresh folder tree (Hub convention — other spokes will write into it, so get it exactly right)
Cluster: `$SCRATCH/CHLU-gen/workspaces/<dataset>_<pretext>/<stage>/<arm>_s<seed>/`
with `logs/`, `artifacts/`, and a per-run `run_manifest.json` recording git commit, resolved JAX version, every
non-default flag in effect, and the SLURM job id.
Local (pulled back by `sync_project.sh`): under the **consuming spoke's own** output dir, never a shared one.
⛔ Never write anything to `projects/` or to any existing C2/C3 workspace.

## Acceptance
1. `bash -n` and `shellcheck` clean; every axis with no safe default is proven to **fail loudly when unset** by a test.
2. ⭐ **A smoke job actually ran on CSF3 through this script** — submitted, completed, and wrote logs, artifacts and a
   populated `run_manifest.json` into the fresh tree. Paste the job id, the realised paths and the manifest contents.
   The payload can be trivial; the *path plumbing and the manifest are what is under test.*
3. `tests/test_gen_launcher.py` covers the unset-axis failures and the path construction. Runs on CPU in seconds.
4. Suite green against the named base. Record the pass count **before** your change and after, using the **main venv**
   per protocol §4 (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m pytest -q`).
   ⚠ A fresh worktree venv resolves different extras and drops ~7 tests — that is not a regression, but if you see it,
   say so rather than reporting a smaller number.
5. Re-verify `git log --oneline main..agent/experiment-engineer/gen-csf3-launcher` **from the main repo** before any
   worktree removal.

## If you cannot reach the cluster
Say so. Deliver the script, the tests and a **dry run**, and report acceptance item 2 as **NOT RUN**, with what you tried.
⛔ Never report an unrun smoke as a pass, and never soften it into a zero or a failure.

## What would prove this spoke wrong
- The script runs but writes into an inherited C3 path, or the manifest is empty. **The manifest is the deliverable that
  makes every later number traceable — a run that produces one is the point, not a formality.**
