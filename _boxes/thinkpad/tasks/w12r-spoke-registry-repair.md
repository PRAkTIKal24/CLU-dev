> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · registry + matrix at `/c/chlu/evidence/`.
>   ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — frozen, older, every number stale.
>   ⛔ `/c/chlu/evidence/` is **READ-ONLY**; write copies into your own output dir.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w12r-registry-repair/`** and your own worktree.
> - ⭐ **START_UTC.txt first, LOG.md as you go, END_UTC.txt last.**
> - ⭐⭐ **ZERO compute, ZERO scored runs.** ⛔ If you think you need a run, you have misread the job —
>   name the run you would need and leave the item open.
> - ⛔⛔ **Never a box-wide kill.** ⚠ **Wave 12 is RUNNING on this box right now** (3 agents,
>   `w12-*` tmux sessions). ⛔ Do not touch their sessions, worktrees, output dirs or processes.
> - ⭐ **Commit and bundle your branch WHOLE:** `git bundle create <f> <your branch>` — ⛔ never
>   `--all`. Verify, and check `list-heads` shows exactly one ref.
> - ⭐ **Anything not closed is a DECLARED OPEN ITEM with the reason** — ⛔ never silently dropped.
> - ⭐ **Every number names the file it came from.**

# w12r spoke/registry-repair — repair the negatives registry's structural defects

- **Agent:** `doc-curator`. ⭐ **ZERO compute. ZERO scored runs.**
- **Output:** `/c/chlu/outputs/w12r-registry-repair/` — updated `negative_results.md` +
  `claims_matrix.md` **as copies**, plus LOG.md and FINDINGS.md.
- **Owns:** its output dir + branch `w12r/registry-repair`. ⛔ `/c/chlu/evidence/` is **READ-ONLY**;
  the Hub folds your copies back.

## THE ONE JOB
Repair the three **structural** defects two wave-10 spokes found in the registry, and fold in the
wave 8/9 negatives that are missing.

## THE DEFECTS — each found by a spoke, each with its evidence
1. ⛔⛔ **N89/N90 are ALLOCATED TWICE** — a 2026-09-11 gen-pivot addendum re-used them.
   ⭐ `w10-registry-fold` **allocated no number in either series rather than compounding it**, which
   was right. ⛔ **You must now renumber**, and print a **redirect line** at each old number so no
   existing citation silently breaks.
2. ⛔ **The gen-pivot negatives have NO MATRIX ROW AT ALL.** Give them rows.
3. ⛔ **N1–N313 contain ZERO w5/w6 entries** — `grep -in "gym-1|gym-2|illegal"` returns **0 hits**.
   ⚠ The physics spoke had to draw those negatives from the wave records instead. ⭐ Add them.

## THE NEGATIVES TO FOLD IN — none may be dropped silently
- ⛔ **encoder-loop is a WASH, and the DECOMPOSITION is the finding:** unfreezing HURTS, the swap loss
  REPAIRS it, the net is two real effects cancelling. ⭐ Corroborated on the GPU leg at 10× steps and
  4× data: **swap loss positive on all 3 seeds** (+0.0100 / +0.0129 / +0.0075).
- ⛔ **K ≥ 36 collapse** — the covariance forces it and composition dies above K ≈ 5 **at the old pin**.
- ⛔ **The TensorFaces / MLC-VAE STOP** — never claim request-indexed factored generation.
- ⛔ **MiniGrid is composition-blind** — DoorKey scores 1.0000 blind; rejected from the generator code.
- ⛔ **The Advisor's error:** the Mexican-hat / interpolation law **does not exist in the corpus.**
- ⛔ **w9: composition does NOT transfer to a route task** (CMAZE-2, 3 seeds) — and ⭐ the **mechanism**:
  plan-space overlay **1.0000**, latent-space averaging **0.13**.
- ⚠ **w9 instrument defect:** CMAZE-2's rule B is untested on most routes — **1242 of 1449 routes use
  exactly one key**, so blind-sequential reads 0.9037 there against 0.5317 on CMAZE-1 proper.
- ⚠ **w9 refuser does not work, and WHY:** separation **+0.047**; a ridge probe on the same
  observation at t=0 scores **AUC 0.48–0.55 — chance** — across all six seed×tier runs.
  ⭐ **Deciding solvability-within-budget is close to solving the maze.** Record it as a property of
  the task, ⛔ not as a wiring bug.
- ⚠ **BAR C.2 fails on all 3 GPU seeds** — the equal-seconds control got **6–12 % LESS** compute than
  the treatment (it sized its steps from `alone`'s seconds/step and then ran faster), so it is biased
  **in the treatment's favour**. ⭐ Cause and one-line fix both recorded.
- ⚠ **`base/iter8` (aa91785) ships 3 PRE-EXISTING failing tests** in `tests/test_gen_value_holdout.py`
  (stale `row2_unseen` vs shipped `row1_seen`; missing `reg['row5_vs_row2']`). ⭐ **Unowned** — record
  it so it cannot be lost.

## ACCEPTANCE CRITERION
⭐ Every entry carries **a number and a path**. ⭐ Every renumbered entry carries a **redirect line**.
⛔ A verdict recorded without the measurement that decided it is not a registry entry.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
The numbering cannot be repaired without breaking existing citations ⇒ ⭐ **say so and propose the
scheme**; ⛔ do not renumber silently and leave dangling references.
