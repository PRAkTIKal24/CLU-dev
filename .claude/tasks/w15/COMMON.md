# WAVE 15 — COMMON RULES (every spoke reads this first)

**Base:** `base/iter8 = 148f3b7` in `/c/chlu/workspace-iter8`. Your worktree is `/c/chlu/worktrees/CHLU-w15-<slug>`, your branch `w15/<slug>`, your output dir `$CHLU_HUB_OUTPUTS/w15-<slug>/`. ⛔ Never write outside your own output dir and worktree.

## Environment — do this before any number
```bash
cd /c/chlu/worktrees/CHLU-w15-<slug>
source /c/chlu/env_w3.sh
python -c "import os,pathlib,chlu; cf=pathlib.Path(chlu.__file__).resolve(); assert str(cf).lower().startswith(str(pathlib.Path(os.environ['CHLU_ROOT']).resolve()).lower()), cf; print('OK',cf)"
```
⛔ **If that assert fails, STOP.** You are running the Head's checkout and every number is void.
⛔ Do NOT build a venv, do NOT `uv sync`. The system python + wave-2 site-packages is the pinned env (jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4).

## Non-negotiables
- ⛔ **Never run a box-wide kill.** Capture your own PIDs at launch into `PIDS.txt` and kill only those. Another agent's training run is on this box.
- ⛔ **Poll your background jobs to completion INSIDE a turn** (loop on their output files with sleep). ⛔ Do not end a turn while detached work is running — nothing will re-invoke you and the work finishes into silence.
- ⛔ **No sub-agents.** If the job is too big, say so in your LOG and do the part you can.
- ⭐ **Write `LOG.md` continuously** in your output dir — it is the liveness signal and the record.
- ⭐ Write `START_UTC.txt` at the start and `END_UTC.txt` only when genuinely finished.
- ⭐ **Commit and bundle the branch WHOLE before you finish:** `git add -A && git commit` then `git bundle create $CHLU_HUB_OUTPUTS/w15-<slug>/w15-<slug>.bundle --all` is WRONG — use `git bundle create <out>/w15-<slug>.bundle w15/<slug>` (your branch only) and `git bundle verify` it.
- ⛔ `git status` cannot see gitignored outputs. Verify outputs by directory listing.

## Scored-run discipline (only spokes that spend a scored run)
- ⭐ **Write the ledger row in `/c/chlu/SCORED-LEDGER.md` BEFORE the run**, naming the PREREG path. A run with no prior ledger row does not exist.
- ⭐ **PREREG before any number.** Bars pre-declared. 3 seeds, all or none. ⛔ Never "9/9 seeds".
- ⭐ **Val-first:** every new config is measured on val (free) before any scored run. A config failing its val bar is DECLARED and the next config starts.
- ⭐ Report all scored runs; ⛔ the best is never reported alone.

## Claims discipline
- **Charter-C controls** (`landscape_removed`, `landscape_shuffled`, `landscape_shuffled_across_wells`, `random_atoms`) must COLLAPSE for any store claim. If they do not, it is an **interface** claim or **nothing** — say which.
- **Naive vs naive:** rivals matched on read-path parameters (~130k floats on Shapes). The memory ledger is printed separately, never matched. ⛔ The inflated-rival row does not exist.
- The **lookup table is a printed floor**, never a rival.
- ⛔ **Every byte figure travels with its convention named.** Seed count is a column on every table.
- ⛔ CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T never share a table.
- **Deletion language:** store-scoped exactness (store after deletion == store fitted without the rows); frozen encoder/decoder declared OUTSIDE the deletable object; post-deletion adaptation is light and data-free — ⛔ never "no changes". Rival deletes only by retrain-without, cost printed.
- **Notation:** n = factors jointly composed; K = wells.
- ⭐ **Say what did NOT run.** Skipped work is reported as skipped, never as a zero or a failure. "DECLARED NOT-RUN" with the reason is a complete answer.

## Where a choice is open
⭐ Pick the option closest to an **ICLR main-track result**, record the choice and the reasoning in `LOG.md`, and proceed. ⛔ Do not wait for the Hub, the Head or the Advisor. Nothing in wave 15 requires anyone's approval.

## Deliverable
`FINDINGS.md` in your output dir: what you did, every number with the artefact path it came from, what did not run and why, and what would have proved the idea wrong. ⛔ No number that lives only in prose.
