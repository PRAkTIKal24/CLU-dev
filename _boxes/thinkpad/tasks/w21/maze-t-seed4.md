# w21 `spoke/maze-t-seed4` — BRIEF
**Wave-plan §5c. NEW TARGET `maze-t-seed4`, 3 scored runs. Base = `w19/maze-t-symmetric` tip 0455961. Local CPU.**

## ⛔⛔ THE STANDING RULE THAT CHANGED (Head, 2026-09-24, ALL WAVES)
**PRE-DECLARATIONS AND PRE-REGISTRATIONS ARE BANNED.** ⛔ Do **not** write a PREREG. ⛔ Do **not** declare a bar, a label, a threshold or a prediction before you have the number.
⭐ **PRODUCE AND PRINT EVERYTHING.** Every arm, every seed, every row, every candidate — including the ones that look bad. **The Head decides what is quotable, after the fact.**
⭐ **Provenance logging continues exactly as before:** config, seed, and hashes for every object, in `LOG.md`.
⚠ **The ledger row is NOT a pre-declaration — keep writing it.** It is the SPEND record that stops a target being overspent. Write it before each scored run, **APPEND ONLY (`>>`)**, and state the seed and condition — ⛔ but no expected outcome, no bar.
⭐⭐ **Because nothing is pre-registered, completeness is now the ONLY protection against post-hoc selection. A number you measured and did not print is a defect.**

## STEP 1 — SEED SCREENING (not a scored run; CLU ALONE, no rivals)
The three original seeds are 0, 1, 2. The **seeds-0/2 CLU average on words-plus-observation TEST unseen n-way is 0.804** (0.800, 0.808). ⚠ Seed 1 (**0.566**) is deliberately **not** in that average.
1. Run the **CLU ALONE** (no rivals, CLU **A1 unchanged**) on **candidate seed 3**, words-plus-observation.
2. **Accept it if its TEST unseen n-way is within ±0.07 of 0.804 — i.e. in [0.734, 0.874].**
3. Otherwise **record it** and try **seed 4**, then **seed 5**, in that order, same test.
4. ⭐⭐ **Every tried seed's CLU numbers — VAL and TEST, ALL rows — go in `LOG.md` AND are printed in the appendix table, accepted or rejected.** ⛔ A rejected seed is a printed row, never a deleted one.
⚠ **State plainly in your report that the window is centred on seeds 0 and 2 and EXCLUDES seed 1**, so a seed resembling seed 1 would be rejected. That is the Head's rule and you execute it — ⛔ but the reader must be able to see it. Report **N seeds tried**.

## THE THREE SCORED RUNS — all on the ONE accepted seed
- **Run 1/3 — words-plus-observation, in full.** Every rival at **8k and 80k**, the **10×-read-floats diagnostic**, all four controls with the **derangement asserted and printed**, the learned blind arms, the floors, **deletion rows**, **per-step exactness**, **rollout**. ⭐ **Persist every object with its param hash and file hash** (as w19 did).
- **Run 2/3 — words-full.** CLU **A1** + every rival, **same seed**, ⛔ **no screening**.
- **Run 3/3 — the improved CLU** from `hub/maze-t-clu-improve` (its kept config), on **words-plus-observation**, **same seed**, ⛔ **no screening**.
  ⛔ **Mechanical precondition:** `/c/chlu/outputs/w21-maze-t-clu-improve/` contains the kept config with its hash (look for the config + hash recorded in its `LOG.md` / its `test/` outputs). **If that config does not exist yet, do runs 1 and 2, say run 3 is WAITING, and finish the rest** — ⛔ do not invent a config and do not guess one.

## TWO ADDITIONS TO EVERY MAZE TABLE (Head, 2026-09-24)
1. ⭐⭐ **A PER-SEED CEILING ROW: the best score ANY arm reaches on that seed**, and **name the arm that reached it.** This is what makes a hard seed visible instead of looking like a CLU failure. Seed 1 is the case in point: every strong arm sits at ~0.567 there.
2. ⭐⭐ **A ZERO-RUN DIAGNOSIS OF SEED 1** (analysis only, no fitting): **which of seed 1's held-out combinations are unreachable from the words, and why.** For each held-out n-way tuple, determine from the **bank and the design** whether the information needed to compose it is present in train at all — e.g. whether the required pair/interaction columns have any train support. ⭐ w19 measured that the interaction family `(c,c,c)` scores 0.000 for **both** the CLU and A1 because it has almost no train support; that is the shape of the answer. Print the count and the list.

## SEEDS COLUMN AND CAPTION
Every CMAZE-T table prints **all four seeds**, with a **seeds column = 4** and the caption line: *three original seeds plus one added seed, chosen by the screening rule, N seeds tried.*

## ACCEPTANCE
- The screening table (every tried seed, accepted and rejected, VAL + TEST, all rows).
- Three scored runs on the accepted seed — or two plus a declared WAITING run 3.
- Every maze table carrying the **ceiling row** and the **4-seed column**; the seed-1 diagnosis with counts and the list.
- `TABLES.md` (every cell traceable to a JSON), `FINDINGS.md`, machine JSON, persisted objects with hashes.
- Branch committed WHOLE and bundled; `git bundle verify` passes.
- ⛔ **No PREREG file anywhere in your output.**

## OUTPUT
`/c/chlu/outputs/w21-maze-t-seed4/` — the only place you write outside your worktree.
