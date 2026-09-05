# dt-units-split — COLLISION NOTE (second experiment-engineer instance)

**Status: stood down. No work performed. No files written outside this note.**

I was spawned on `.claude/tasks/dt-units-split.md` at 2026-07-21 ~10:24 BST and found **another
experiment-engineer instance already live on this exact task**. I did not proceed.

## Liveness evidence (per protocol §3 / verify-before-recovery)

| check | observation |
|---|---|
| branch | `agent/experiment-engineer/dt-units-split` already exists, checked out, **2 commits ahead of `main`** |
| commits | `3760087` split data_dt from the Verlet step on the eval path · `6dd43bd` test the dt/data_dt split; expose `--data-dt` on the CAFE runner (10:04:13) |
| running process | PID **99498**, `.venv/bin/python3 .claude/scratch/dt-units-split/dt_selfconsistent.py`, ELAPSED 00:36, 156% CPU — started ~10:24, i.e. **live at the moment I checked** |
| scratch dir | `.claude/scratch/dt-units-split/` — continuous writes 09:52 → 10:24 (`dt_selfconsistent.py` written 24 s before my check) |
| prereg | `.claude/outputs/dt-units-split/PREREG.md` written 10:00 (pre-registration rule already satisfied by the live instance) |
| working tree | clean — no orphaned uncommitted work to rescue |

## Progress already made by the live instance (inferred from artifacts, not verified by me)

- **Item 1 (the split)** — committed: `chlu/eval/config.py`, `chlu/eval/clu_scorer.py`, `chlu/eval/cafe_model.py`, `scripts/cafe/run_clu_cafe.py`, new `tests/test_eval_dt_units.py` (194 lines). Diff 332+/20−.
- **Item 2 (integrator retune)** — `omega_scan.py`, `trained_omega.py/.log`, and the in-flight `dt_selfconsistent.py` are clearly the stability/ω-margin scan.
- **Item 3 (re-measure)** — six FD001 arms already run to completion: `arm_legacy`, `arm_split_b0.16`, `arm_split_b1.6`, `arm_split_b3.2`, `arm_split_mass10`, `arm_split_r1`, with `res_*/` result dirs and `item3.json` / `item3.log`. `full_suite.txt` (10:06) suggests the test suite was run.

So the task is **substantially complete and still advancing**; this is not a dead agent needing recovery.

## Why I did not proceed

Proceeding would have violated protocol §3 in three ways at once: two agents committing to one
branch, my scratch writes landing in a directory the live instance is actively reading/writing,
and a race to overwrite `.claude/outputs/dt-units-split.md`. The w4 precedent (a `git add`
sweeping ~90 foreign lines into another agent's commit) is exactly this failure mode.

I deliberately did **not** write `.claude/outputs/dt-units-split.md` — that file belongs to the
live instance.

## Recommendation for the Head / Hub

1. **Let PID 99498 finish.** Its report will land at `.claude/outputs/dt-units-split.md`.
2. **Do not re-spawn this task** unless that instance dies before writing its output. Check
   liveness first (`ps -p <pid>` + scratch mtimes), then look for the branch commits — the
   Item 1 fix is already durable on `agent/experiment-engineer/dt-units-split`.
3. If it *does* die mid-flight, a recovery spawn should **resume, not restart**: commits
   `3760087`/`6dd43bd` land Item 1, and `item3.json` + the six `arm_*.log` files likely hold
   enough to write up Item 3 without re-running (the FD001 arms are the expensive part).
4. This note can be deleted once the real report lands.

## Proposed handover updates (for the Hub)
None from me — I measured nothing. The live instance owns all handover deltas for this slug
(expected: §7 `dt=0.05` units issue, §3 config table gaining `data_dt`).
