# Task: gen-csf3-cpu-jobs — ⭐ our CPU sweeps are pinning a laptop while 1,500 entitled cores sit idle.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-csf3-cpu-jobs.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ `scripts/csf3/job_gpu_wells.sh` (**the pattern to follow** — checkpoint/resume, preflight, signal handling) · `scripts/csf3/env.sh` · `scripts/csf3/README.md` §1
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's tip · **Branch:** `agent/experiment-engineer/gen-csf3-cpu-jobs` · **Worktree:** `../CHLU-cpujobs`
- ⚠ **Budget ≈ 0.5 day.** ⛔ **No sub-agents.** ⛔ No science, no arm, no number that enters any table.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — infrastructure.** **Laundering control:** n/a. **Falsifies:** nothing.
- ⛔ **No PREREG** — the acceptance is a mechanism, not a measurement.

## Why (measured, 2026-09-15)
⭐ **A four-hour `exp_gen_atom_mean` sweep held a 16 GB laptop at 570 % CPU all evening and BLOCKED the scored cell
arm** — the headline run — because both want the same cores. ⛔ It is **pure CPU work**: that leg and the
`gen-val-split-fix` leg before it both ran `JAX_PLATFORMS=cpu`.
⭐ **And we are entitled to far more, on an account we are not using:**
| association | GrpTRES | MaxTRESPU |
|---|---|---|
| ⭐ **`fatpou01`** | **cpu=1500** | ⭐ **cpu=84 per user** |
| `gpu-free` (what the GPU jobs use) | — | `a100_80g=4, l40s=4, gpu=6` |
⛔ **`fatpou01` carries NO GPU** — ⭐ **that is the point: it is uncontended.** `multicore` just gained **1,680
cores** in the CSF3/CSF4 merger, while **both GPU partitions are at their group caps** (`gpua` 76/76, `gpul` 84/84)
and we sit on the **free/scavenger** QOS at priority 5759.
⇒ ⭐ **84 cores, queued in minutes, instead of 8 laptop cores blocking the headline.**

## The one job
**A CPU jobscript that runs the gen CPU experiments on `multicore` under the `fatpou01` account, checkpointed.**

## Scope
1. **`scripts/csf3/job_cpu_gen.sh` (new)** — modelled on `job_gpu_wells.sh`:
   `#SBATCH -p multicore`, ⭐ **`-A fatpou01`** (⛔ **never `gpu-free`** — that is the GPU association and has no CPU
   grant), `-c` **overridable, ⛔ never above 84**, `-t` overridable (max `7-0` on `multicore`).
   ⭐ Carry over **`--requeue`**, **`--signal=B:USR1@300`**, `--open-mode=append`.
2. ⭐ **`JAX_PLATFORMS=cpu` set explicitly**, and **asserted** — ⛔ a job that silently finds no GPU and runs anyway
   is fine here, but one that *reports* GPU numbers from a CPU run is not. ⚠ Expect the benign
   `cuInit(0) failed: Unknown CUDA error 303` on a GPU-less node; ⛔ **suppress it by setting the platform, do not
   let it look like a failure.**
3. **Parameterised by experiment**, not hardcoded to one: at minimum `exp_gen_atom_mean` and `exp_gen_baselines`,
   selected by an `--export` variable, with the args passed through. ⛔ **Do not build a second launcher.**
4. ⛔⛔ **CHECKPOINT AND RESUME — the standing Head rule.** ⭐ **Reuse `pilot-checkpoint-resume`'s pattern**
   (bitwise-verified on both paths); ⛔ do not invent a second one. Time-based, **atomic temp+rename**, keep the
   last N, **resume verifies step + config hash and refuses a mismatch**. Checkpoint to `~/scratch`,
   ⛔ never `$TMPDIR` (wiped at job end), ⛔ never `$HOME` (quota).
   ⚠ A sweep is **many units**: ⭐ **per-unit `skip_if_exists` is the resume**, exactly as the local sweep does it
   (its `SKIP … (manifest exists)` lines are why a kill costs one unit, not the run).
5. **Artefact roots** — `env.sh` already exports the five `CHLU_*` from `$CLU_REPO/artefacts`; ⭐ **assert them at
   job start** and die naming the missing one. ⛔ Never reach the work to find it cannot open an artefact.

## ⛔ Cluster discipline
⛔⛔ **CSF3 needs 2FA. You cannot reach it.** Emit submit lines; ⭐ **report every cluster leg `NOT RUN`, naming it.**
⛔ `git pull` on the cluster serves a stale fork — `push_repo.sh` is the only code path. ⛔ Never `-p serial` for
multi-core work (it is a **1-core** partition). ⚠ Login nodes have **no off-campus internet**.

## Files you own — and nothing else
- `scripts/csf3/job_cpu_gen.sh` **(new)** · `tests/test_job_cpu_gen.py` **(new)** · `.claude/outputs/gen-csf3-cpu-jobs/**`
⛔ Not `env.sh`, ⛔ not `job_gpu_wells.sh`, ⛔ not `setup_env_job.sh`, ⛔ not `stage_artefacts.sh`, ⛔ not `chlu/**`,
⛔ not `chlu/config.py`. ⭐ **Read them, follow them, leave them alone** — they are live paths for a cluster user.

## Acceptance
1. `job_cpu_gen.sh` exists, `bash -n` clean, `-A fatpou01`, `-p multicore`, `-c` overridable and **refusing > 84**
   with a message naming the limit.
2. ⭐ **A local dry-run** proving the experiment is invoked with the right args and roots. ⛔ It cannot run on CSF3.
3. ⭐⭐ **Resume demonstrated LOCALLY by killing it** — `kill -9` mid-sweep, restart, show it **skips completed units
   and resumes**, ⛔ not restarts. ⛔ Not waivable for time; an untested resume path has never worked here.
4. The `-c > 84` refusal and the missing-artefact-root abort **both demonstrated firing**.
5. ⭐ **One copy-able submit block for the Head**, with expected wall time and the resume command.
6. Targeted tests green; **passed / skipped / failed**. ⛔ No full suite. A **cost line** (CSF3 CPU = free).

## What would prove this spoke wrong
- ⭐ **You submitted under `gpu-free`** — it has no CPU grant, so the job is rejected or scavenges, and the whole
  point of using an uncontended entitlement is lost.
- ⭐ **You asked for more than 84 cores** and the job pends forever on `AssocGrpCpuLimit` exactly as the GPU jobs
  are pending on `AssocGrpGRES` right now.
- ⭐ **The resume path was written but never killed.**
- You hardcoded one experiment, so the next sweep needs a second jobscript.
