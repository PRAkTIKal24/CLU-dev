# w20 hub/shapes-data-matched — THE ONE-TO-ONE TABLE THAT CLOSES M1 AND M2
**Agent type: autonomous-hub. Target `shapes-matched`. 3 scored runs; replications free. Local CPU.**
⭐⭐ **PRE-AUTHORISED IN FULL. No decision from the Head or the Advisor is needed. Never wait for either.**

## Why this wave exists
⛔ **No data-matched table exists today, and the paper's headline depends on one.**
- The headline table's rivals (from `w12-deletion-baselines`) were fit on **16,384 rows** — that spoke's **declared BUILD/SMOKE scale**, built for the retrain-without COST line and then **reused as the headline comparators**.
- The ladder's rivals (spoke-A, wave-4 protocol) were fit on **32,256 rows**.
- **The store's pool is 131,072 rows.**
⇒ Referee **M1** (fairness is parameter-only) and **M2** (two rival sets an order of magnitude apart on identical corners) both trace here. ⭐ **This ONE rival set replaces the two sets in the draft and closes both.**

## Build TWO one-to-one tables. The Head picks which leads; the other goes to the appendix.
### Table M-131k
Every rival — **MLP / LSTM / transformer / continuous MLP at the 131,072-float read-path match, spoke-A protocol** — **fit on the store's EXACT 131,072-row pool**, same seeds, **trained to convergence with the loss curve printed**, plus the **10×-steps diagnostic column**. The store at **K\* = 32** and at **K = 5** as already fitted. **Rows 3–4 and every ladder rung. 3 seeds.**
⚠ **T7 already has 3 seeds** — `w15-single-design-seeds` landed and spent **0** scored runs because `single-design` was exhausted. ⛔ **Never duplicate a landed run.** Read `/c/chlu/outputs/w15-single-design-seeds/T7-3SEEDS.json` first and reuse it.

### Table M-16k
The store (**K\* = 32** and **K = 5**) **re-fit on the rivals' 16,384-row pool**, and as a third column **on spoke-A's 32,256-row pool**, rows 3–4 and the ladder, **against the existing 16k and 32k rivals. 3 seeds.**

## Every table must carry
⭐ **Both fit-pool sizes in the caption.** ⭐ **A seeds column.** ⭐ **Controls, the table floor, and the ceiling per corner on every row.**
⭐ The **LACE/PoE rows from wave 15** added where they exist (`/c/chlu/outputs/w15-lace-poe-rows/GENTABLE-KSTAR.json`) — ⛔ carry their scoping verbatim: LACE/PoE **as instantiated in this feature space at their own declared budgets**, ⛔ **not** a refutation of LACE in general.
⛔ **The inflated-rival row does not exist.**

## ⚠ Two defects you must not re-introduce
- ⛔ **A rival that has not reached its declared budget cannot carry a margin.** Wave 16 withdrew a recorded win purely over under-trained rivals. **Print the loss curve and the 10× diagnostic.**
- ⛔ **`landscape_shuffled_across_wells` must be a DERANGEMENT, printed.** `rng.permutation(K)` at K=2 is the identity half the time (registry `N350`).

## Spending
**Val-first; VAL IS FREE.** ⭐ Ledger row in `/c/chlu/SCORED-LEDGER.md` **BEFORE** each scored run, naming its PREREG. Replications of the same PREREG are free. Budget **3**.

## Acceptance criterion
`FINDINGS.md` + `SHAPES-MATCHED.json`: both tables complete with fit-pool sizes, seeds, controls, floor and ceiling; the convergence evidence per rival; the budget accounted; and an explicit statement of **what the data-matched margin is** against the draft's current +39…+49. ⛔ If the margin collapses, **that is the result** — report it plainly; it is exactly what M1 predicted.

## Operating rules
⛔ Never a box-wide kill; PIDs into `PIDS.txt`. ⛔ No sub-agents. ⛔ Poll background jobs INSIDE a turn. ⛔ Avoid `rm <glob>` (the dangerous-rm guard stalls you) — use a fresh timestamped subdir. ⚠ This box's python is Windows-native: convert msys paths with `cygpath -m` before opening them. ⭐ `LOG.md` continuously; `START_UTC.txt` now; `END_UTC.txt` only when finished; bundle the branch and verify. ⭐ **Say what did NOT run.**
