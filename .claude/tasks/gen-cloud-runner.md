# Task: gen-cloud-runner — ⭐ the image proves the plumbing and runs NO SCIENCE. Make it run our experiments, unattended.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-cloud-runner.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`.claude/outputs/gen-cloud-enablement.md` §7 items 1 and 4 — this task IS those two items** · `scripts/cloud/README.md` · `scripts/cloud/startup.sh` · `scripts/cloud/ckpt.py`
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 9bf6422` or the Hub's tip · **Branch:** `agent/experiment-engineer/gen-cloud-runner` · **Worktree:** `../CHLU-runner`
- ⚠ **Budget ≈ 1 day.** ⛔ **No sub-agents.** ⛔ No science, no arm, ⛔ no number that enters any table.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — infrastructure.** **Laundering control:** n/a. **Falsifies:** nothing.
- ⛔ **No PREREG** — the acceptance is a mechanism, not a measurement.

## Why (measured by the peer spoke, 2026-09-15)
⭐ `gen-cloud-enablement` proved the whole path — pinned image, quota refusal, GCS checkpointing, mid-run sync-back,
cost line — for **$1.37**, and survived **being pre-empted twice unprompted**, with five runs converging on one
64-character hash chain over 600 steps. ⛔ **But it runs a SYNTHETIC SMOKE PAYLOAD.** Its own §7:
- **item 4 — "the image carries no dataset and no real entry point."**
- **item 1 — "a spot job needs a relaunch supervisor and I did not ship one."** `det-ctl` needed **three launches**
  to finish 600 steps and a human drove all three.
⇒ ⛔⛔ **Today GCP cannot run a single CHLU experiment, and a spot job needs a babysitter.** ⭐ That is the whole gap
between "the cloud works" and "the cloud is a fallback."

## The one job
**Run a real CHLU experiment on GCP, unattended, to completion, across pre-emption.**

## Scope — two things, no more
### 1. ⭐ A REAL ENTRY POINT
Point `startup.sh`'s `docker run` at an actual experiment module, selected per job (`EXP=`), args passed through —
the `job_cpu_gen.sh` pattern: ⭐ **one launcher, parameterised**, ⛔ never a second launcher per experiment.
- ⭐ **Export the five `CHLU_*` roots** from the GCS artefact prefix, each `${VAR:-default}`, and ⭐ **assert them at
  job start, naming the missing one.** ⛔ A job must never reach the GPU to discover it cannot open an artefact.
- ⭐ **Reuse `ckpt.py`.** ⛔⛔ **Do not write a second checkpoint mechanism** — that was explicit in the peer's task,
  it held, and it still holds.
- ⚠ **`dec_recipe` defaults to `"baseline"`, the recipe that FAILED the gate.** The winner is `v1_capacity_schedule`
  and its hash must be asserted. ⛔ A baseline run **looks entirely normal in the logs and is void.**

### 2. ⭐⭐ THE RELAUNCH SUPERVISOR — `scripts/cloud/run_until_done.sh`
Relaunch while there is no `result.json`; **back off on stockout**; ⭐ **a hard cap on attempts and on total spend**,
⛔ never an unbounded loop. Report attempts used.
⚠ **Spot pre-emption is not an edge case here** — it happened **twice unprompted in one short acceptance window.**
⛔ **A fallback that needs a human to notice and relaunch is not a fallback.**

## ⛔ MECHANICAL PRECONDITION for the END-TO-END leg only
⛔ `.claude/outputs/gen-cloud-artefact-bridge/STAGED-GCS.json` with `"status": "complete"`, `"checksum_match": true`
and **0 failed to load**. ⭐ **A peer spoke owns that and runs in parallel.**
⭐ **Build and test everything else against the smoke payload first** — ⛔ do not idle waiting, and ⛔ **do not
improvise your own artefact staging.** If the file never appears, report the end-to-end leg **`NOT RUN — precondition
absent`** and ship the rest.

## Files you own — and nothing else
- `scripts/cloud/startup.sh` · `scripts/cloud/launch.sh` · `scripts/cloud/Dockerfile` · `scripts/cloud/README.md`
- `scripts/cloud/run_until_done.sh` **(new)** · `tests/test_cloud_runner.py` **(new)** · `.claude/outputs/gen-cloud-runner/**`
⛔ **Not `scripts/cloud/stage_gcs.sh` / `verify_gcs.sh` — `gen-cloud-artefact-bridge` holds those**, ⛔ not
`scripts/cloud/ckpt.py` (⭐ **use it, do not change it**), ⛔ not `scripts/csf3/**`, ⛔ not `chlu/**`, ⛔ not `chlu/config.py`.

## ⚠ Cards — measured, and ⛔ NOT what an earlier brief said
⭐ **L4 in `us-central1` is the proven path** (6 free; the peer's acceptance ran there).
⚠ ⛔ **"A100-80GB does not exist for this project" was WRONG — it is REGIONAL.** `us-east4` holds quota **2, one free
on-demand**. ⛔ **But it is quota-verified and NEVER RUN** — stock was absent all window. ⭐ If you use it, **say it is
the first job on that path.** ⛔ **FOUR** other-project instances are running in `us-central1`: ⛔ evict none.

## Acceptance
1. ⭐ **A real CHLU experiment ran on a GPU VM to completion**, artefacts synced back under `.claude/outputs/…`.
   ⛔ Not the synthetic payload. ⛔ Not a dry run. **Report the experiment, the card, the wall time and the cost.**
2. ⭐⭐ **UNATTENDED ACROSS PRE-EMPTION:** `run_until_done.sh` drives a **spot** job that is **killed mid-run**,
   relaunches **itself**, and finishes — ⛔ **with no human in the loop**, which is the entire point.
   ⛔ Not waivable for time.
3. The five `CHLU_*` roots asserted at job start; **demonstrated firing** with one removed.
4. `dec_recipe` asserted by name **and** hash; **demonstrated firing** on a deliberate mismatch.
5. Attempt cap and spend cap both **demonstrated refusing**. ⛔ No unbounded relaunch loop.
6. Targeted tests green; **passed / skipped / failed**. A **cost line**.

## What would prove this spoke wrong
- ⭐⭐ **You "tested" the supervisor by killing the process instead of the VM.** Pre-emption destroys the machine;
  a process kill does not test relaunch, stockout back-off, or a new VM picking up the checkpoint.
- ⭐ **Unbounded relaunch.** A stockout loop that retries forever spends real money for nothing and is worse than failing.
- ⭐ **You wrote a second checkpoint mechanism** rather than reusing `ckpt.py`.
- ⭐ **The run looked fine and used the `baseline` decoder**, so every number it produces is void.
- A second launcher per experiment, so the next arm needs a third.
