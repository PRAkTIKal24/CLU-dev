# Task: gen-launcher-fixes — a `certificate` stage on the cluster, and the smoke that was never run

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-launcher-fixes.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/csf3-operations.md` §8.3, §10, §11, §12 — the authoritative reference** · **`.claude/tasks/gen-clu-phase1-rulings.md` R26 ruling 7** · `.claude/outputs/gen-csf3-launcher.md` · `.claude/outputs/gen-certificate.md` **§10** (the blocked submit line).
- ⛔ No campaign ledger beyond the above — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3faef27` · **Branch:** `agent/experiment-engineer/gen-launcher-fixes` · **Worktree:** fresh (`../CHLU-gen-launchfix`).
- **Env:** main venv. ⛔ **CSF3 requires 2FA and is unreachable from an agent session — you will NOT submit anything.**

## ⭐ DIAL DECLARATION
- **Dial:** *none — instrument.* ⛔ Nothing scientific runs.
- **Laundering control:** N/A. **Falsifies:** N/A — the failure mode is a script that looks right and has never run.

## The one job
Add the **`certificate` stage** to `scripts/csf3/job_gpu_gen.sh`, and leave the Head a submit block that works.

## What is wrong now
`job_gpu_gen.sh` hard-rejects any `STAGE` outside `encoder|decoder|store|cell`, so there is no certificate stage. The
workaround found by `gen-certificate` (`STAGE=cell` + `EXTRA_ARGS="--stage certificate"`) **writes into a `cell/`
workspace path**, which corrupts the provenance of a different stage's directory. ⛔ Not acceptable.

## What to do
1. Add `certificate` to the permitted `STAGE` set, with its own default payload module
   (`chlu.experiments.exp_gen_certificate`) and its **own workspace path** — ⛔ never sharing `cell/`.
2. ⭐ **Keep every existing guard intact:** `DATASET`/`PRETEXT`/`STAGE` still fail loudly if unset; `GEN_GIT_COMMIT`
   still mandatory; the array-size check still refuses a partial range without `PARTIAL=1`; the upstream-manifest
   `status == "ok"` check still applies for the same arm and seed. **Declare what the certificate stage's upstream is.**
3. ⭐ **ONE literal command line per submission** (`csf3-operations.md` §10) — no backslash continuation inside
   `--export`; zsh does not word-split an expanded variable and it has bitten this programme live.
4. Extend `tests/test_gen_launcher.py` (you own it) to cover the new stage, its path, and its unset-axis failures.
5. Append a `certificate` subsection to the `## Generative pivot` section of `scripts/csf3/README.md`.
   ⛔ Do not reformat or edit any other line.
6. ⭐ **Update `.claude/csf3-operations.md` §8.3 and §12** — §12 currently reads *"the generative pivot has not been
   launched"*, and §11 marks the gen launcher `[UNTESTED-ON-CLUSTER]`. Those stay accurate until a smoke actually
   returns; say precisely what is and is not verified. ⛔ Do not mark anything verified that you did not see.

## Files you own
`scripts/csf3/job_gpu_gen.sh` · `scripts/csf3/README.md` (**the gen section only**) · `tests/test_gen_launcher.py` ·
`.claude/csf3-operations.md` (**§8.3, §11, §12 only**).
⛔ Nothing under `chlu/`.

## Acceptance
1. `certificate` stage accepted, with its own workspace path; `bash -n` + `shellcheck` clean.
2. Every pre-existing guard still fires — proven by tests, not by inspection.
3. Launcher tests extended and green; full suite green (before/after, main venv).
4. ⭐ **A ready-to-run submit block for the Head**, one literal line per submission, covering the smoke **and** the
   certificate stage, with the realised paths each will produce and the exact commit to pin.
5. ⛔ Cluster items reported as **NOT RUN**. ⛔ Never report an unrun job as a pass.

## What would prove this spoke wrong
- You marked the launcher cluster-verified without a job id. ⭐ **§11's honesty labels exist precisely for this**, and
  the smoke has still never run.
- The certificate stage writes into `cell/`, or a guard you did not intend to touch stopped firing.
