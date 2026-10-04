# w15 hub/single-design-seeds — T7 AT THREE SEEDS ON TEST
**Read `COMMON.md` first.** Slug `single-design-seeds`. Agent type: autonomous-hub.
**Scored: `single-design` runs 2/3 and 3/3. ⚠ This EXHAUSTS the target.**

## The one job
Run **seeds 1 and 2 at K\* = 32 on TEST**, under the **same PREREG as run 1** (`outputs/w12-single-design-crossover/PREREG-w12-single-design-crossover-1.md` + AMENDMENT 1), so the main table prints **n = 3**.

⛔ **This is a PURE REPLICATION. No bar moves. No K is selected. No design changes.** Byte-for-byte run 1/3's configuration: same `CovContentStore`, same `ward_balanced` K\* = 32 cut (wells 64 wide, A = 63, rank/width 0.9844), same rows 3 + 4, same `distinct_valid_novel_tuples` per corner, same `fw.paired_estimand`, same 1.0 margin.

⚠ **Check first whether w12b already produced these seeds.** `outputs/w12b-nr6-and-kstar/NR6-T7.md` reports K32 at seeds [0,1,2] already. **If seeds 1 and 2 at K\*=32 on TEST already exist as scored artefacts, DO NOT RE-SPEND the runs** — report that in `FINDINGS.md`, mark the two runs **DECLARED NOT-NEEDED**, and spend your effort on the deletion rows below instead. ⭐ Establish this by reading the artefacts, not by assuming.

## The deletion rows (the part that is genuinely new)
At those seeds, produce the deletion rows: **red value**, **red ∧ sphere conjunction**, **1 % items**.
- Store-scoped exactness: store after deletion == store fitted without the rows. Frozen encoder/decoder **declared outside the deletable object**.
- Post-deletion adaptation is **light and data-free** — ⛔ never "no changes".
- **Rival deletes only by retrain-without, cost printed.** Also print rival as-trained.
- ⛔ "red ∧ cube" deletes ZERO rows (cube is the condition's own held-out shape). The conjunction pin is `object_hue = 0.0 ∧ shape = 2.0`, named **"red sphere"** everywhere.

## Control gate
⛔ **NO ROW IS CLAIMED unless all four charter-C controls COLLAPSE on EVERY seed.** A control that collapsed at seed 0 is NOT assumed to collapse at seeds 1 and 2 — measure it.

## Ledger
⭐ Write the row in `/c/chlu/SCORED-LEDGER.md` BEFORE each run, naming the PREREG. If you conclude the runs are not needed, write a ledger note saying so rather than a run row.

## Acceptance criterion
`FINDINGS.md` + `T7-3SEEDS.json` showing rows 3 and 4 at K\*=32 on TEST with **n = 3 and the seed list printed**, the four-part byte ledger per seed (⛔ never netted, convention named), all four controls per seed, and the three deletion rows with adaptation cost.

## What would prove this wrong
A control failing to collapse on seed 1 or 2 ⇒ **no store row is claimed** and you say so as the result. That is a complete, publishable answer.

## Files you own
Your worktree, `$CHLU_HUB_OUTPUTS/w15-single-design-seeds/`, and your ledger rows. ⛔ You do NOT own the w12/w12b output trees — read them.
