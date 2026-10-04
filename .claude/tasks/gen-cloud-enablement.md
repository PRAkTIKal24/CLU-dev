# Task: gen-cloud-enablement — ⭐ make GCP runnable. We have quota and credentials and NOTHING to run on them.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-cloud-enablement.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `scripts/csf3/README.md` §3 (the env recipe to mirror) · `scripts/csf3/env.sh`
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 115d4b5` · **Branch:** `agent/experiment-engineer/gen-cloud-enablement` · **Worktree:** `../CHLU-cloud`
- ⚠ **Budget ≈ 1 day.** ⛔ **No sub-agents.**

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — infrastructure.** No arm, no mechanism, ⛔ no number that enters any table.
- **Laundering control:** n/a. **Falsifies:** nothing. ⛔ No PREREG — the acceptance is a mechanism, not a measurement.

## The measured gap
`gcloud` is authenticated (`pratik.jawahar@forgis.com`, project `forgisprova`) and quota exists. ⛔ **The repo has
ZERO cloud tooling** — no Dockerfile, no launch script, no sync-back, no image. **Verified: nothing under
`scripts/` matches `*cloud*`/`*gcp*`/`Dockerfile*`.** ⇒ the ladder and the scored read have nowhere to run.

## ⭐ THE MEASURED QUOTA — ⛔ build against THIS, not against the assumed cards
| card | us-central1 (**where every prior job ran**) | europe-west4 |
|---|---|---|
| **L4** | ⭐ **8, 2 IN USE ⇒ 6 free** | 8 |
| A100-40GB on-demand | ⛔ **1, CONSUMED by a running instance** | 1 |
| ⭐ **A100-40GB SPOT** | ⭐ **16 free — the training path** | 16 |
| ⛔ **A100-80GB** | ⛔ **0 — quota request not yet filed** | ⛔ **0** |
| ⛔ **H100 / H200** | ⛔ **not offered** | ⛔ **not offered** |
⛔⛔ **Another project holds live capacity in `us-central1` — the instance `tsfm-pair-a` MUST NOT be evicted or
touched.** ⭐ **Default region: `us-central1`.** ⭐ **L4 for iteration, A100-40GB SPOT for training.**
⭐ **80 GB A100s exist on CSF3, free at point of use** ⇒ ⛔ **do not design around an 80 GB cloud card. It does not exist.**

## The one job
**Deliver a working, checkpoint-first path from this repo to a running GPU job on GCP and back.**

## Scope — five things, no more
1. **A pinned image.** Mirror the CSF3 recipe: `uv sync --frozen`, **pip CUDA wheels, ⛔ never a host CUDA module**
   (mixing them shadows the bundled libs and breaks JAX). ⭐ **Pin by digest, not by tag** — a floating tag makes a
   re-run irreproducible and we would never know.
2. **A launch script** — `scripts/cloud/launch.sh`: card, spot-vs-on-demand, region, job name, `--dry-run`.
   ⭐ **Refuses to start if the requested card has no quota**, naming the card and the measured limit.
3. ⛔⛔ **CHECKPOINT-FIRST — this is the point of the task, not a feature of it.**
   ⭐ **Spot instances are pre-empted with ~30 s notice. A run that cannot resume WILL lose work — it is a matter of
   when.** Requirements are the standing ones: **time-based** checkpoints (pre-emption does not wait for an epoch);
   **atomic** temp-file + rename; keep the last N; **resume verifies step + config hash and refuses a mismatch**;
   checkpoint to a **persisted** location, ⛔ never instance-local scratch that dies with the VM.
   ⭐ **Reuse the proven pattern** — `.claude/outputs/pilot-checkpoint-resume.md`, verified BITWISE on both the
   straight and the resume path. ⛔ Do not invent a second mechanism.
   ⭐ **Handle the pre-emption signal**: checkpoint on notice, then exit cleanly.
4. **Sync-back** — results land under `.claude/outputs/<slug>/`. ⛔ **A result that is not synced back does not
   exist.** ⭐ Sync-back runs on a **timer during** the job, ⛔ not only at the end — a pre-empted VM never reaches
   its last write. *(This programme has already lost a final write to a detached job.)*
5. **A cost line** — the script prints card, hours and dollars at exit, and the report carries the same.

## ⛔ What you do NOT do
- ⛔ **Do not file the A100-80GB quota request.** It posts under the Head's account. **Emit the exact console URL and
  metric names; the Head files it.**
- ⛔ **Do not touch, stop, resize or evict any existing instance**, above all `tsfm-pair-a`.
- ⛔ **Do not move any CSF3 path.** Cloud is additive. `push_repo.sh` / `stage_artefacts.sh` are untouched.
- ⛔ No science. No arm. No dataset staged into an image.

## Files you own — and nothing else
- `scripts/cloud/**` **(new)** · `tests/test_cloud_launch.py` **(new)** · `.claude/outputs/gen-cloud-enablement/**`
⛔ Not `chlu/config.py` (a peer holds it), ⛔ not `scripts/csf3/**`, ⛔ not any `chlu/**` module.

## Acceptance
1. ⭐ **One real GPU job ran end to end on an L4** — trivial payload, but a real VM, real image, real sync-back.
   ⛔ Not a dry run. **Report its cost.**
2. ⭐⭐ **PRE-EMPTION SURVIVED, DEMONSTRATED.** Start a job on **spot**, **kill the instance mid-run**, relaunch,
   and show it **resumes rather than restarts** — with a bit-identity check against the uninterrupted run.
   ⛔ **A resume path that has never actually lost its VM is not a resume path.** ⛔ Not waivable for time.
3. The launch script refuses an un-quota'd card with a message naming card and limit.
4. Image pinned **by digest**, recorded in the report.
5. Sync-back demonstrated **mid-run**, not only at exit.
6. The quota-request URL + metric names, for the Head. ⭐ Every cloud item the Head must run is listed as such.
7. Targeted tests green; passed/skipped/failed. A **cost line**.

## What would prove this spoke wrong
- ⭐⭐ **You built it on spot and never had an instance taken away.** Then the checkpointing is a hypothesis and the
  first real run pays for it.
- ⭐ **Sync-back only at exit** — so a pre-empted job produces nothing, which is the common case, not the rare one.
- ⭐ **You designed around A100-80GB or H200.** Neither exists for this project. A plan that needs one is not a plan.
- You evicted or disturbed another project's running instance.
- The image is pinned by tag, so a re-run silently gets a different environment.
