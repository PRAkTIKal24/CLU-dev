# AMENDMENT 3 to `PREREG.md` §5 — the best learned baseline gets HARDER, not easier

**Written 2026-09-21T11:49:46Z** (`AMENDMENT-3.timestamp.txt`). At that instant the only
`rows_*.json` on disk in this hub's entire `runs/` tree was
`runs/3_seed0/rows_learned_ref_ownpool.json` — **one learned arm, and not one arm of this
hub's own**. No `single_design_K5` score, no `pca_tensor` score, no control score, at any rung
or seed. The change below therefore cannot have been shaped by an arm's number, and it moves
the bar **up**.

## What happened

`AMENDMENT-1` added `learned_ref_ownpool`: the same 524 288 B parameter pin as spoke A's rival
set (**130 514 parameters**, `match_width` −0.43 % of the 131 072 target, width 61), but fitted
on **the rung's own fit pool** instead of the `montero_combination` pool. Its first measurement,
at k = 3 seed 0 over 120 corners
(`runs/3_seed0/rows_learned_ref_ownpool.json`):

> **`distinct_valid_novel_tuples` = 8.2167 ± 9.3864**, judge accuracy 0.1141,
> fitted on 32 768 rows, final train MSE 0.0325.

The published rival set on the shipped rows sits four orders below that —
`/c/chlu/evidence/outputs/w10-seeds-and-tables/TABLES.md` prints
`rival_mlp_180k @ alone` at **0.006** on row 3 and **0.000** on row 4.

⇒ **Fitting the learner on the pool the arm is fitted on is worth more than any architecture in
spoke A's set.** Keeping `PREREG.md` §5's rule as written would mean this hub competes against
an opponent it has already measured to be far weaker than one it can build, and quotes the
result as *"the best learned baseline"*. That would be a soft bar wearing a hard bar's name.

## The change

> **`PREREG.md` §5 is amended so that the candidate set for the best learned baseline is
> `{rival_mlp, rival_lstm, rival_transformer, rival_cont_mlp, rival_table, learned_ref_ownpool,
> learned_1x}`** — every learned arm on this hub's table, including the two this hub fits
> itself. The selection rule is unchanged: per (rung, seed), the candidate with the **largest**
> mean `distinct_valid_novel_tuples` over that rung's corners. The winner's identity is printed
> beside every margin.

Everything else in §5 stands: per (rung, seed), largest mean, same corners in the same order,
`exp_gen_factor_wells.paired_estimand`.

**Why this is not a free hand.** The new candidates can only *raise* the opponent's score, and
the margin is `arm − opponent`, so every margin this hub reports can only get **smaller**. The
amendment cannot flatter any arm of this hub at any rung.

## What is reported either way

⛔ The pre-declared narrow reading is not discarded. Both are printed on every row:

- **`best_learned_5`** — the `PREREG.md` §5 rule exactly as written 11:38:41Z, over the five
  pre-existing rival arms only. This is what the figure's panel B used to be pinned to and it
  stays on the record.
- **`best_learned_all`** — this amendment's rule, over all seven learned arms. ⭐ **This is the
  one the figure's panel B plots and the one any verdict in FINDINGS is taken against.**

If the two disagree at a rung, the disagreement is itself reported.

## A consequence stated in advance

If `learned_ref_ownpool` (or `learned_1x`) is the best learned baseline at every rung and the
store arms do not clear the pre-declared **+1.0** margin against it, then under
`AMENDMENT-2` §3 the store arms **never cleared the bar** and have **no breaking point** —
and that, not a curve, is the answer this hub reports. It is written down here before the
first store-arm number exists so that it cannot later be softened.
