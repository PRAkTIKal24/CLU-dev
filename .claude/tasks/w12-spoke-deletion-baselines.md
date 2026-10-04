# w12 spoke/deletion-baselines — the comparators every deletion row needs

- **Agent:** `experiment-engineer`.
- **Output:** `/c/chlu/outputs/w12-deletion-baselines/` (LOG.md, FINDINGS.md, **BASELINES-READY.json**).
- **Owns:** its output dir + branch `w12/deletion-baselines`. ⛔ No other spoke's tree.
- **Scored runs:** ⛔ **NONE.** You build and wire comparators; the crossover hub spends the runs.

## THE ONE JOB
Build the three deletion comparators and **wire them into the rows-3/4 harness so the crossover hub
calls them BY NAME**, not by reimplementing them.

## THE THREE COMPARATORS
1. ⭐ **Retrain-from-scratch-without** — the exact reference. Retrain the learned arm on the data
   **minus the deleted rows**. ⛔ **Print its cost:** seconds, data rows read, bytes rewritten.
2. **Approximate unlearning, one per learned arm, at MATCHED COMPUTE** — negative-gradient or
   fine-tune unlearning on the deleted rows. ⛔ Matched compute is the point; a weakened comparator
   makes the whole table worthless.
3. **The refuse-token transformer** — trained on exactly-certified impossible examples. It is the
   **only permitted comparator for refusal DISPLAY rows.**

## ACCEPTANCE CRITERION
⭐ **`BASELINES-READY.json` names each comparator, its entry point, its matched-parameter count, and
its cost fields** — and a smoke call of each returns a number. ⛔ The crossover hub gates on this file.

## ⛔ THE COST LINE IS PART OF THE DELIVERABLE, NOT A NICETY
The Head's ruling: ⛔ **never say "no changes after deletion".** Say **post-deletion adaptation is
LIGHT and DATA-FREE** — re-eigendecompose the ledger, carry the address book, **no row of data read** —
against baselines that can only delete by **retraining from scratch without the deleted rows**.
⭐ **Every deletion row prints, for EVERY arm: seconds · bytes rewritten · data rows read.**
⛔ A comparator with no cost line cannot appear in a table.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- An approximate unlearning baseline that cannot be matched on compute ⇒ say so; ⛔ do not ship a
  weakened one and call it matched.
- The refuse-token transformer fails to separate possible from impossible ⇒ ⭐ **that is a finding,
  not a bug** — wave 9's refuser separated at only +0.047 and a probe on the same observation scored
  at chance. Report the separation number; ⛔ do not tune until it looks better.
