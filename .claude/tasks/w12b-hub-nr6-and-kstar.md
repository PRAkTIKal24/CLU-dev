> ## ⛔ STANDING RULES — evidence base `/c/chlu/evidence/outputs/` (read w9/w10/w12 reports, ⛔ NOT $CHLU_CLAUDE_ROOT, which is a stale frozen snapshot). Write ONLY to your own output dir + worktree. START_UTC.txt first, LOG.md throughout, END_UTC.txt last. Assert `chlu.__file__` is inside YOUR worktree before any number. Poll background jobs to completion INSIDE a turn — never end a turn with detached work running. ⛔ NEVER a box-wide kill; capture your own PIDs. Bundle YOUR branch only, never --all; verify list-heads shows exactly one ref. A published file is not a finished file — check completeness, not presence. Anything not run is a DECLARED NOT-RUN with a reason. Every number names its file.

# w12b hub/nr6-and-kstar — finish wave 12's two open legs

- **Agent:** autonomous hub. **Targets:** `single-design` (**2/3 left**) + replications (free).
- **Output:** `/c/chlu/outputs/w12b-nr6-and-kstar/`. **Branch:** `w12b/nr6-and-kstar`.
- **Base:** `base/iter8` at **`e58ab05`** (w12 + w12r + two w13 spokes folded).
- ⚠ **`hub/encoder-adapt` is TRAINING on this box.** ⛔ Never touch its session, worktree, output dir
  or PIDs. Use **3 threads**.

## THE TWO JOBS, in this order

### 1. ⭐ NR6 — the headline row is ONE SEED (scored; `single-design` 2 runs available)
T7 (rows 3 and 4 at **K\* = 32**, TEST) rests on **seed 0 only**. Run the remaining seeds.
⛔⛔ **Run 2/3 must FIRST buy the missing TEST number at K = 5.** The crossover hub declared it a
NOT-RUN and said so itself: *"the third pre-declared falsification is untestable while only K\* has a
TEST row."* ⭐ Without it the val/test gap cannot be checked at all.
- ⭐ **Write your ledger row in `/c/chlu/SCORED-LEDGER.md` BEFORE each run** — target `single-design`,
  runs **2/3** then **3/3**, your OWN rows.
- ⭐ Reuse the crossover hub's PREREG (`PREREG-w12-single-design-crossover-1.md` + AMENDMENT 1); this
  is a **same-PREREG replication at further seeds**, ⛔ not a new design. Append an amendment only if
  something is genuinely new, and ⛔ **move no bar**.
- ⭐ **Seeds 3 and 4 caches now exist** (`CACHE-READY.json`, 59 blocks each, gate passed). Use a seed
  **only if its entry has a non-null `array_shape` AND 59 blocks.** ⚠ Seed 4 has **no external
  classifier**, so it cannot be judged on the instrument every other number used — ⛔ if that still
  holds, **say so and use seeds 0–3**, declaring the count.
- ⭐ **Run the controls on every scored row:** `landscape_shuffled`, `random_atoms`,
  `landscape_removed`. ⛔ If they do not collapse, **no row is claimed** — that is the result.

### 2. ⭐ THE K\* REPLICATION — free, and it may flip the ladder's verdict
The n-way ladder ran its store arm at **K = 5**, so its opponent `learned_1x` carried
**1,029,508 params**. ⭐ **Head ruling: the main text carries ONE K only — K\* — and the K=5 ladder
goes to the appendix as the sweep's endpoint. Never both in main text.**
- **Re-run the store arm and `learned_1x` at K\* = 32**, all five rungs, 3 seeds, same estimand and
  bar. At K\* the 1× budget is **535,784 params** — ⭐ **the opponent is roughly HALVED.**
- ⚠ **The current verdict (store loses at 4 of 5 rungs by −0.054…−0.077) was measured against the
  LARGER opponent.** ⛔ Do not assume it carries; measure it.
- ⭐ These are **replications — free**. ⛔ No ledger row, no scored budget.

## ⛔ TWO THINGS MEASURED THIS WAVE THAT BIND YOU
1. ⛔ **The ladder is NOT a composition-depth axis for n ≥ 3** — rungs 3,4,5,6 probe the **identical
   120 corners**; what varies is held-out fraction. ⭐ **Report the rungs, but label the axis honestly.**
2. ⛔ **`rival_transformer` and `rival_lstm` could not train at all** through `w4_learners.train`
   before a dtype fix folded at `e58ab05` (a float32 cast broke `Embedding` indexing).
   ⭐⭐ **Re-fit both on every row you produce and state whether your numbers pre- or post-date the
   fix.** ⚠ Their flat pre-fix scores (0.489–0.531, 0.579–0.604) may be artefacts.

## ACCEPTANCE CRITERION
⭐ T7 carries **≥ 3 seeds** with the count printed, the **K = 5 TEST** number exists, and the ladder's
store arm has a **K\* = 32** reading at all five rungs with `learned_1x` refit at 535,784 params.

## ⛔ WHAT WOULD PROVE THIS HUB WRONG
- The controls fail to collapse at K\* on a scored row ⇒ the store is decorative there; **claim nothing**.
- The K\* replication **flips the sign** ⇒ ⭐ that is the wave's headline and it must be reported as a
  correction to the K=5 verdict, not buried.
- Seed 4 remains unjudgeable ⇒ declare 4 seeds and say why; ⛔ never pad to 5.
