# w21 `spoke/words-full-test` — BRIEF
**Target `maze-t-symmetric`, runs 4-6/9. Base = `w19/maze-t-symmetric` tip 0455961. Local CPU.**

## THE ONE JOB
Score the **WORDS-FULL** condition on the **TEST** bank, **exactly as w19 ran it on VAL**. Nothing is re-tuned, re-designed or re-selected. This is a measurement, not an experiment.

## WHAT "EXACTLY AS RUN ON VAL" MEANS - all of it is FROZEN
- CLU **Config A1 UNCHANGED** (latent 128, K 2, 6 atoms, ridge 1.0).
- The condition input: the **318-wide `[bias|main|pair]` design to EVERY arm**, CLU and rivals alike.
- Every rival kind already in the w19 WORDS-FULL table: `mlp`, `lstm`, `transformer`, `cont_mlp`, each matched on read-path floats by the standing rule, trained at **8,000 steps AND 80,000 steps**.
- The **10x-read-floats diagnostic column** (8,000 steps). It is printed and is **never a bar**.
- All four charter-C controls, the learned blind arms, and the WORDS lookup floor.
- Banks, hold-outs, judge, codec, state encoder: untouched, imported.
- **3 seeds (0, 1, 2), all or none.**

## ROWS TO PRODUCE
`nway` (**primary**), `seen`, `pair`, `value`, and **per-step exactness**. Plus **deletion rows on the scored leg** (both modes, as w19 did on BOTH): store-scoped exactness, downdate-vs-fitted-without agreement on surviving rows, CLU adaptation time, and the rival's retrain-without cost. Per the Advisor's standing artefact ruling, deletion measurement **(d) prints the PRE-deletion composition on the SAME surviving-colour rows beside the post-deletion number**.

## THE LABEL RULE - the Head's, and it REPLACES the w19 PREREG margin bars
Every condition prints in full with per-seed numbers. The label is the **measured relation to the best matched rival over both budgets**:
- **WIN** - ahead on the mean AND on a majority of seeds, AND the store gate holds.
- **COMPARABLE** - within one seed's spread.
- **LOSS** - behind.
⛔ **No numeric margin bar.** Name the mechanism alongside the label. w19's VAL reading for this condition was CLU 0.7247 vs transformer-8k 0.7420 - so **COMPARABLE or LOSS is the expected outcome and is a perfectly good result.** ⛔ Do NOT re-tune to move it.

## ⭐⭐ THE INSTRUMENT CHECK THAT CAN KILL THIS RUN - do it BEFORE any TEST number
Re-run the CLU on **WORDS-FULL VAL** and confirm it reproduces w19 to the printed digits: **0.8019 / 0.5689 / 0.8034** (n-way, seeds 0/1/2). Also confirm `landscape_shuffled_across_wells` is a **DERANGEMENT** and print the permutation.
⛔ **If the VAL triple does not reproduce, the instrument has drifted. STOP, spend nothing, write it up and finish.** A scored run on a drifted instrument is worse than no run.

## ACCEPTANCE
- `test/BARS-WORDS-FULL-test.json` + per-seed `test/WORDS-FULL_test_seed{0,1,2}.json`, all rows, deletion rows.
- `TABLES.md` (every printed cell traceable to a JSON), `FINDINGS.md`, `WORDS-FULL-TEST.json`.
- Each rival row prints its **hidden width beside its matched float count**.
- The rollout line prints **per-step exactness AND mean first-error step** for CLU and best rival, and states that no arm validates a rollout.
- A label (WIN / COMPARABLE / LOSS) with the mechanism named.
- Branch committed WHOLE and bundled; `git bundle verify` passes.

## BUDGET AND THE LEDGER
Runs **4, 5, 6 of 9** on target `maze-t-symmetric` - one per seed. ⛔ **Write the ledger row to `/c/chlu/SCORED-LEDGER.md` BEFORE each run**, and **APPEND ONLY (`>>`)** - ⛔ never rewrite that file; a peer agent is appending its own rows to it at the same time.
⛔ Runs 7-9 belong to `hub/maze-t-clu-improve`. **Do not touch them.**

## OUTPUT
`/c/chlu/outputs/w21-words-full-test/` - the only place you write outside your worktree.
