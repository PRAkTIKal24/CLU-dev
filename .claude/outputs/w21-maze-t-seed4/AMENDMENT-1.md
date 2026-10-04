# AMENDMENT 1 — the Head, 2026-09-24. ⛔ THIS REPLACES THE BRIEF'S "STEP 1 — SEED SCREENING".

## ⛔ WHAT IS CANCELLED
The accept/reject screening is **GONE**. ⛔ There is no ±0.07 window, no acceptance test, and **no stopping at the first seed that passes**. ⛔ Do not discard a seed for any reason.

## ⭐⭐ WHAT REPLACES IT
**Run THREE new seeds — 3, 4 and 5 — and keep all three.** On every one of them, run:
1. ⭐ **The winner row: words-plus-observation on TEST**, CLU **A1 unchanged**, **with ALL RIVALS** (8k and 80k, the 10×-read-floats diagnostic, all four controls with the derangement asserted, the learned blind arms, the floors, deletion rows, per-step, rollout). **Persist objects with hashes.**
   ⭐ **These are the 3 scored runs of target `maze-t-seed4` — one per new seed.** Ledger row before each, append only, seed + condition, no expected outcome.
2. ⭐ **The comparable row: observation (OBS) on VAL**, CLU A1 + **ALL RIVALS**, same three seeds. ⭐ **VAL is FREE — this consumes no scored run.** (w19's OBS was VAL-only and its COMPARABLE label is a VAL label, so this is the like-for-like extension.)

⛔ **`words-full` does NOT get new seeds** — it is a LOSS row and the Head did not include it. ⚠ Its row will therefore show **3 seeds** while the winner and comparable rows show **6**. **Print the seed count per row; do not pad and do not average across different seed sets.**

## ⭐⭐ HOW THE RESULT IS PRESENTED — this is the point of the amendment
- **Easy seeds → MAIN TEXT. Hard seeds → APPENDIX.** Both are printed in full; ⛔ nothing is dropped.
- ⭐⭐ **The finding to support, with numbers:** on a hard seed **every model struggles, ours included** — which shows **that seed's held-out data is inherently hard**, rather than showing a CLU failure.
- ⭐ **The PER-SEED CEILING ROW is now the load-bearing evidence for that claim, not a nicety.** For every seed print **the best score ANY arm reaches** and **name the arm**. A seed where the ceiling itself is low is a hard seed, and the ceiling is what proves it. Seed 1 is the known case: every strong arm sits at ~0.567.
- ⭐ **Classify each seed as easy or hard by its measured CEILING, and say in the caption what the cut was.** ⛔ Do not classify by the CLU's own score — that would be choosing seeds by our own result.
- ⭐ Extend the **zero-run diagnosis** (brief §"TWO ADDITIONS", item 2) to **every** seed you classify as hard, not only seed 1: which held-out combinations are unreachable from the words, and why.

## SEEDS COLUMN AND CAPTION
Tables now carry **up to 6 seeds** (3 original + 3 new). The seeds column states the count **for that row**. The caption says: three original seeds plus three added seeds, all run and all printed, easy seeds in the main text and hard seeds in the appendix, classified by measured ceiling.

## ⛔ UNCHANGED FROM THE BRIEF
No PREREG, no bars, no labels, no predictions before results. Produce and print everything — the Head decides what is quotable. Provenance (config, seed, hashes) as always. Run 3/3's improved-CLU leg from `hub/maze-t-clu-improve` is **superseded**: that peer spends its own runs on `maze-t-symmetric` 7–9. ⭐ If its kept config exists when you are otherwise done, run it on the new seeds as a **FREE VAL** leg and print it; ⛔ do not spend a scored run on it.
