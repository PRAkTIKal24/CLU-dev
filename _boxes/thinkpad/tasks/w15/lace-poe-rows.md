# w15 spoke/lace-poe-rows — WIRE K* INTO THE GENERATION TABLE, ADD LACE AND PoE
**Read `COMMON.md` first.** Slug `lace-poe-rows`. Agent type: experiment-engineer.
**Scored: `generation-table` run 3 (the last one). Replications free.**

## The one job
Wire the **K\* single design** into the generation-table harness — the `exp_gen_cell` registry (`clu` / `store_removed` / `knn_proto` / `lace` / `ebm_compose` / `dynread`) — and **spend `generation-table` run 3 on it**.

Then add **LACE** and **PoE** as rows:
- on Shapes **rows 3–4**, and
- on **every ladder rung**,
- **3 seeds**, **at their own budgets, declared**.

⛔⛔ **LACE and PoE NEVER replace the learned-rival columns. They are ADDED.** MLP / LSTM / transformer / continuous MLP all stay.

## Acceptance criterion
`GENTABLE-K STAR.json` (name it `GENTABLE-KSTAR.json`) + `FINDINGS.md`: the K\* single design present in the registry and scoring, LACE and PoE rows on Shapes 3–4 and every rung at 3 seeds with **their own budgets printed in the row**, the learned-rival columns intact beside them, and the ledger row written before the scored run.

## What would prove this wrong
⭐ If LACE or PoE **beats** the CLU on rows 3–4 at a declared budget, that is a headline negative and it goes in the main text. ⛔ Do not bury it, do not re-tune them down. Print it.

## Files you own
`chlu/exp_gen_cell*` registry entries for lace/poe wiring, `scripts/w15_lace_poe_*.py`, your worktree, your output dir. ⚠ The registry is shared code — ⛔ **add entries only; do not refactor it**, and do not touch entries you did not add.
