# w21 `spoke/ladder-rivals-32k` — BRIEF
**Wave-plan §6a. FREE REPLICATION — ⛔ NO LEDGER ROW, NO SCORED RUN** (the ladder is free by the w12b precedent, and `w20`'s own PREREG says so). Local CPU.

## WHY THIS EXISTS — the defect it removes
Referee-4 **MF-2**, verified in `kstar_ladder/*/LEG.json` (`n_fit_rows 32768`): the headline table's store is fitted on **131,072** rows, but the **LADDER** store's content is fitted on **32,768** rows of each rung's pool. §6's Table M-131k is right for **rows 3–4**; for the **ladder** it hands the rivals **4× the store's data**. ⛔ That reverses the very asymmetry the wave exists to remove.
⭐ The Head's ruling: **keep and print the 131k ladder rows as briefed**, and **add a third set fitted on the store's own pool.**

## THE ONE JOB
Fit **MLP / LSTM / transformer / continuous MLP** at the **read-path float match**, **spoke-A protocol**, **fitted on the SAME 32,768 rows as the ladder store's content**, on **every rung (n = 2…6)**, **3 seeds**.
- **Trained to convergence, with the loss curve printed**, plus the **10×-steps column**.
- Printed as a **THIRD rival column set** in the ladder table, beside the **32,256-row** set (combination-split, ⚠ which carries the held-out values) and the **131,072-row** set.
- ⭐⭐ **Pool sizes go in the caption for all three sets.** A reader must never have to guess which pool a column was fitted on.

## ⭐⭐ THE CROSS-CHECK THAT CAN FALSIFY THIS SPOKE
The existing **data-matched MLP row** (`learned_ref_ownpool`, **130,514 params**, the same 32,768 rows, 20 epochs, lr 1e-3 = 2,560 steps) **STAYS, and it is the control on your own work.**
⭐ **Your new MLP column must reproduce it within seed spread.**
⛔⛔ **If it does not, you REPORT THE DISCREPANCY. You do NOT resolve it, tune it away, or pick whichever looks better.** Two numbers for the same quantity disagreeing is the finding. Print both, name the difference, and say what you checked.

## PRECONDITIONS — mechanical, check them, do not assume
- `/c/chlu/outputs/w20-shapes-data-matched/END_UTC.txt` **exists** (⛔ `hub/shapes-data-matched` must have FINISHED — never interrupt it, never run while it is live).
- Its 131k ladder legs are on disk and readable. ⭐ You are ADDING a column set beside them; ⛔ you never modify, refit or re-score anything w20 produced.
- The rung pools and `n_store_train = 32768` are read from the artefacts, ⛔ not from any brief — including this one.

## ACCEPTANCE
- A third rival column set: **4 families × 5 rungs × 3 seeds**, each with its loss curve and its 10×-steps column.
- The MLP cross-check either **reproduces `learned_ref_ownpool` within seed spread**, or the discrepancy is **reported with both numbers**.
- `TABLES.md` with every cell traceable to a JSON, `FINDINGS.md`, machine JSON.
- Caption text naming **all three pool sizes**, ready for `figures-redraw` to drop in as a data file.
- Branch committed WHOLE and bundled; `git bundle verify` passes.

## ⛔ WHAT WOULD MAKE THIS WORK WORTHLESS
Fitting on any pool other than the ladder store's own 32,768 rows, or silently reconciling the cross-check. ⭐ The whole point is that the rival and the store see **the same data**.

## OUTPUT
`/c/chlu/outputs/w21-ladder-rivals-32k/` — the only place you write outside your worktree.

---

# ⭐⭐ AMENDMENT (Hub, 2026-09-24, from `hub/shapes-data-matched`'s reconciliation at 06:57Z)
⛔⛔ **THE OPERATIVE ASYMMETRY IS TRAINING STEPS, NOT DATA.** w20 measured it directly, mlp on rows 3–4, all at **20,480 steps**:
| pool | steps | score |
|---|---|---|
| 16,384 rows | 20,480 | **38.606** |
| 32,768 rows | 20,480 | **39.067** |
| 131,072 rows | 20,480 | **39.150** |
| `learned_ref_ownpool`, 32,768 rows | **2,560** | **11.810** |
| T7 comparator, 16,384 rows | **1,280** | **5.535** |
⇒ **At matched steps, 16k ≈ 32k ≈ 131k.** The rivals' deficit in the draft was a **training-schedule** deficit. ⭐ `learned_ref_ownpool` is **NOT contradicted** — it is the same rows at **1/8 the steps**.

## WHAT THIS CHANGES FOR YOU
- ⛔ **Do NOT treat the pool as the variable of interest on its own.** Report **pool AND step budget together in every column header**; a column labelled only by rows is misleading and is exactly how this defect got in.
- ⭐⭐ **The cross-check is now a STEP-MATCHED comparison, and you should EXPECT it to differ.** Your new MLP at convergence will be far above `learned_ref_ownpool`'s 11.810 **because of steps, not data**. ⛔ That is NOT the discrepancy you were told to report — it is the known, measured effect. ⭐ **Report it as reconciled**, showing both step budgets side by side, exactly as w20 did.
- ⭐ **The discrepancy that WOULD matter:** your MLP at the **same rows AND the same steps** as an existing row disagreeing with it. That one you report and do not resolve.
- ⭐ Keep the **10×-steps column** — with steps as the operative variable, it is now the most informative column in the table, not a diagnostic afterthought.
