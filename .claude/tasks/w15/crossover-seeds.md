# w15 spoke/crossover-seeds — THE CROSSOVER AT SEEDS 1 AND 2
**Read `COMMON.md` first.** Slug `crossover-seeds`. Agent type: results-analyst.
**Scored: ZERO. All val. Free.**

## The one job
Run **K = 8, 10, 12, 16, 20, 24, 28 at seeds 1 and 2** (14 legs), same frozen rule and same instrument as the w12 crossover, and **regenerate the crossover figure with n = 3 at every K**.

## What to print
- The **honest four-part total** (atoms + cov/count + address + coupling) beside composition, at every K, ⛔ **never netted**, with the **convention named** on the figure and in the JSON.
- ⭐ The total is **U-shaped in K with its minimum near K = 24** — confirm or refute that on 3 seeds and say which.
- Composition score vs K, 3 seeds, seed band shown.

## Acceptance criterion
`CROSSOVER-3SEED.json` + `CROSSOVER-3SEED.md` + a regenerated figure (vector PDF **and** PNG), every K carrying `n = 3` and the seed list, and the four-part ledger per K per seed. Byte-identical on rebuild.

## What would prove this wrong
If the U-shape's minimum moves materially across seeds, the "minimum near K = 24" sentence cannot be claimed — report the per-seed minima and say the claim does not hold. ⭐ That is the result, not a failure.

## Files you own
`scripts/w15_crossover_seeds.py`, your worktree, your output dir. ⛔ Read the w12 crossover artefacts; never edit them.
