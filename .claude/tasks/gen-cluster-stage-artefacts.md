# Task: gen-cluster-stage-artefacts — ⛔ THE GATE. No real cluster science can run until this lands.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-cluster-stage-artefacts.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `scripts/csf3/push_repo.sh` · `scripts/csf3/sync_project.sh` · `scripts/csf3/README.md`
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3eb9eb0` · **Branch:** `agent/experiment-engineer/gen-cluster-stage-artefacts` · **Worktree:** `../CHLU-stage`
- **Env:** main venv. pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**, ⛔ never passed alone.
- ⚠ **Budget ≈ 0.5 day.** This is plumbing. ⛔ Do not turn it into a science task.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — plumbing.** No arm, no mechanism, no verdict, no number that enters any table.
- **Laundering control:** n/a — nothing is scored here.
- **Falsifies:** nothing. ⚠ It may reveal that the artefact set we believe is complete is not.

## The defect (mechanical, measured)
`scripts/csf3/push_repo.sh:42` carries `--exclude '.claude/'`. Every gen artefact — the feature bridge, the shipped
store, the winner decoder, the partition, the external classifier — lives under `.claude/outputs/**`.
`sync_project.sh` only moves `projects/<name>/`, which these are **not**.
⇒ ⛔ **There is currently NO path that puts a gen artefact on CSF3 scratch.** The green cluster smoke ran `SMOKE=1`
with **no stage module and no science**; it proves the plumbing starts, ⛔ not that a real run could load anything.

## The one job
**Build and prove a verified staging path that puts the gen artefact set on CSF3 scratch, checksummed end to end.**

## ⛔ THE HARD CONSTRAINT — read this before planning anything
⛔⛔ **CSF3 requires 2FA. You cannot reach it. No agent ever has.**
⇒ **Your deliverable is the MECHANISM plus the PROOF THE FILE SET IS COMPLETE — ⛔ not the transfer.**
- You build the script, you compute the manifest and checksums **locally**, and you **emit the exact command lines**
  for the Head to run.
- ⭐ **Every cluster item is reported `NOT RUN`, naming the leg.** ⛔ Never a zero, never a failure, never a pass.
- ⭐ `GEN_GIT_COMMIT` is exported **on the login node**, not the laptop. ⛔ Never `-p serial`. ⛔ `git pull` on the
  cluster serves a stale fork — **`push_repo.sh` is the only path for code.**

## Scope
1. **Enumerate the artefact set a gen cluster run actually opens.** ⛔ Do not guess it from the directory listing —
   **derive it from the code paths** (`exp_gen_*` config loads, bridge/store/decoder/classifier resolvers) and say
   which module opens each file. ⚠ An artefact nobody can name a loader for does not go.
2. **`scripts/csf3/stage_artefacts.sh` (new).** Push that set to `~/scratch/CHLU/artefacts/`, preserving the relative
   layout the loaders expect. ⭐ Idempotent, resumable, `--dry-run` first.
3. **A manifest** — `stage_manifest.json`: every file, its size, its **sha256**, and the loader that opens it.
4. **A verify job** — a tiny GPU-less cluster job that re-checksums on the far side and **opens every artefact through
   its real loader**, writing `STAGED.json` with `status`, `checksum_match`, and a per-file result.
   ⭐ **Opening it is the test. A file that transferred and will not load is a failure, not a success.**
5. ⚠ **Report the total bytes.** CSF3 scratch has **no backup and a 3-month auto-cleanup**; anything staged is
   re-stageable or it is lost. ⛔ Never stage a dataset (`.claude/data/` is ~10 GB and re-downloadable).

## Acceptance
1. `stage_artefacts.sh` exists, is idempotent, has a working `--dry-run`, and refuses to run with an empty file set.
2. `stage_manifest.json` lists every artefact with sha256 **and the named loader**. ⛔ A file with no loader is not in it.
3. The verify job script exists and is **demonstrated failing** — corrupt one byte locally and show it reports a
   mismatch. ⛔ A checker that cannot fail proves nothing.
4. ⭐ **One copy-able block of command lines for the Head**, in order, with what each should print.
5. Targeted tests green with the data root set, passed/skipped/failed reported.
6. ⭐ **Every cluster-side item reported `NOT RUN`.**

## Output that GATES the next spoke
⭐ `.claude/outputs/gen-cluster-stage-artefacts/stage_manifest.json` — written by you.
⛔ `.claude/outputs/gen-cluster-stage-artefacts/STAGED.json` with `"status": "complete"` and
`"checksum_match": true` — ⛔ **written only after the HEAD runs the transfer and the verify job.** It is the
mechanical precondition for `gen-wells-route-b`.

## ⛔ FILES YOU OWN — and nothing else
- `scripts/csf3/stage_artefacts.sh` **(new)** · `scripts/csf3/verify_artefacts.sh` **(new)**
- `tests/test_stage_artefacts.py` **(new)**
- `.claude/outputs/gen-cluster-stage-artefacts/**`
⛔ **You do NOT own `chlu/config.py`** (a peer spoke holds it this wave), ⛔ nor any `chlu/**` module, ⛔ nor
`push_repo.sh` / `sync_project.sh` — ⭐ **report the `--exclude '.claude/'` line as a finding; do not edit it.**
Changing the shared push path under a live cluster user is how a working sync silently stops working.

## What would prove this spoke wrong
- ⭐ **You staged a file set derived from the directory listing rather than from the loaders** — so it is missing the
  one file the job opens on line 400, and we find out four days into a cluster run.
- ⭐ **The verify job checksums but never opens.** A byte-perfect artefact that the loader rejects is still a dead run.
- You quietly attempted the transfer and reported a cluster result. ⛔ You cannot reach CSF3. Saying you did is the
  one unrecoverable failure here.
