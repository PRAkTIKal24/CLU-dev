# PREREG-w9-maze-1 — SCORED RUN 1 of 3 (budget: `maze`, 3 runs)

**Written 2026-09-21, BEFORE any held-out (test-leg) number exists for this hub.**
Base `base/iter8 = fa60b15`. Branch `w9/maze`. Harness gate
`/c/chlu/outputs/w9-harness/HARNESS-READY.json` present (state recorded in the ledger row).

## 0. What is being scored, and with what

**Scored leg = the TEST split** (`--which test`), which no DEV leg has touched. The VAL split is
where every choice in §2 was made. The two splits hold out DIFFERENT pairs from the SAME fitted
object: per maze, one direction block supplies the val holdouts and the other the test holdouts.

**Scored run 1 = seed 0. Runs 2 and 3 are seeds 1 and 2**, same configuration, pre-registered by
`PREREG-w9-maze-2.md` and `-3.md` before each executes. **All three are reported; the best is
never reported alone.** Scored-run count travels with every number quoted ("scored 1/3" …).

## 1. The object and the comparators

| arm | what it is | parameters |
|---|---|---|
| `clu` | `CovContentStore` atoms (per-well leading principal subspace of the shared encoder's TRAIN latents) + `gen_request_head` factorial address book, closed-form ridge | the ledger's `read_floats` |
| `mlp` / `lstm` / `transformer` | learned request→latent maps, main-effect one-hots only (no pair block) | matched to `read_floats` within 2 % |
| `lookup_matched` / `lookup_full` | charter C.1 floor: stored latents, nearest by request agreement | matched bytes / all items |
| `frag_same_maze` / `frag_cross_maze` | `w8_envscout.screen`'s retrieval floor, on this split and cross-maze | — |
| `landscape_removed` / `landscape_shuffled` / `random_atoms` | charter C.2 controls | same shape |
| `clu_nopair` | the pair-interaction ablation | smaller |

Every arm emits a PLAN MAP, is rendered by the SAME frozen decoder through the SAME latent
standardisation, executed by the SAME parameterless action head, and graded by the SAME exact
integer judge (`w8_envscout.maze_colour.judge`, unmodified).

## 2. Configuration, frozen here (chosen on VAL only)

Filled in from the DEV legs and **not changed after the first test number**:

- mazes per bank 96, per_side 3 (6 agent words, 6 target words), budget words {30, 45, 60},
  interpolation words {38, 53};
- shared encoder: conv encoder + projection decoder, latent D = 256, trained by reconstruction on
  the bank's TRAIN items plus the ENCODER POOL (mazes from a disjoint seed range); **no held-out
  item is ever seen by the encoder**;
- store, PINNED from the VAL sweep (DEV leg 3, LOG 08:05Z): **K = 8 wells by label-free
  `ward_balanced`, `n_atoms` = 31, ridge = 0.01**, giving rank/d_k = 0.938 and a deployed read of
  410,688 floats. ⚠ That saturation ratio is DECLARED on every row: on this content
  **span fraction == saturation ratio**, and 0.938 sits close to the wave-8 saturation trap;
- a SECOND, pre-registered CLU arm `clu_lowrank`: **K = 8, n_atoms = 8, ridge = 0.01**
  (rank/d_k = 0.219, 105,984 floats) — the charter-comfortable, clearly non-saturated store,
  which is the regime the Shapes3D incumbent lives in. It is reported BESIDE the selected arm and
  never instead of it;
- the learned baselines are matched to the SELECTED arm's 410,688 floats to within 2 %;
- 3 seeds, one scored run each.

## 3. The estimand, the corners and the margin

**Estimand:** the paired difference in exact-judge VALIDITY between the CLU and each comparator on
the held-out TEST rows at matched parameters, paired by (row, seed, rule-B-bend class).

**Corners:** rows {seen, unseen_pair, nway, budget_interp} × seeds {0,1,2} × bend class
{bends, does not bend} × comparators {mlp, lstm, transformer, lookup_matched, frag_same_maze}
= **120 paired corners** (≥ 49 required).

**Margin, declared now: 0.05 absolute validity.** Direction: CLU > comparator.

## 4. The bars — fixed before the first held-out number

- **BAR 1 — THE DIAL.** Mean over 3 seeds, on the test leg, the CLU's validity on `unseen_pair`,
  `nway` and `budget_interp` each exceeds the BEST of {mlp, lstm, transformer} by ≥ +0.05.
  ⛔ **If a matched-parameter LSTM or transformer beats the CLU on any of these rows, the Shapes3D
  result does not transfer on that row, and THAT IS THE WAVE ANSWER — it is reported.**
- **BAR 2 — the floor (charter C.1).** The CLU exceeds `lookup_matched` and `frag_same_maze` by
  ≥ +0.05 on the same rows. Otherwise the store is reported as decorative on that row. A tie counts
  only under charter C.3 (refusal / exact deletion / novel valid tuples on the same artefacts).
- **BAR 3 — the controls (charter C.2).** `landscape_removed` and `landscape_shuffled` validity
  ≤ 0.05 on the `seen` row. If they do not collapse, no row is claimed.
- **BAR 4 — the easy half.** The CLU-minus-best-baseline margin is > 0 in BOTH bend classes on
  `unseen_pair`. ⛔ If the margin exists only where rule B does not bend the route, the hub reports
  that it measured the easy half.
- **BAR 5 — ceiling honesty.** Every row is quoted beside (a) the representation ceiling, (b) the
  pipeline ceiling on that row, and (c) the store-span ceiling. **A row whose pipeline ceiling is
  below 0.50 is labelled CEILING-BOUND and its comparisons are reported as bounded, not as a
  transfer result.**
- **BAR 6 — DISPLAY rows, never quoted as RESULTs.** Refusal ≥ 0.80 on `budget_infeasible` with
  false-refusal ≤ 0.20 on `grounded_possible`; the `walled` vs `mt_unseen` contrast reported either
  way (if they are equal, the walled refusal is novelty detection and the hub says so); deletion
  exactness ≤ 1e-10 (statistic, relative) and ≤ 1e-6 (subspace projector); post-deletion composition
  on the surviving targets within ±0.05 validity of pre-deletion, **with no retraining**.

## 5. What travels with every RESULT row

- **BLIND-SEQUENTIAL: 0.4980 published on CMAZE-1, and THIS hub's re-measured value on ITS split**
  (both printed; the re-measured value is far higher and the reason is measured and declared —
  see the LOG bullet of 06:10Z).
- **FRAGMENT-LOOKUP: 0.0000/0.0000 published at r=1/r=2 on CMAZE-1's cross-maze split, and THIS
  hub's same-maze and cross-maze values.**
- **Rule B bends 81.6 % of routes on CMAZE-1, and this hub's re-measured bend fraction.**
- The scored-run count, the ledger row, and the ceilings.

## 6. Expected direction (stated before the run)

The CLU is expected to beat the learned baselines on `unseen_pair` and `nway` because its address
book is factorial: a route from a new (agent, target) pair is the sum of a `maze × agent` term and a
`maze × target` term, and the door bottleneck makes that decomposition real. It is expected to tie
or lose on `seen`, where a lookup that stores the items outright is unbeatable at these byte counts.
`budget_interp` is expected to be ~equal to the trained budget words for the CLU, because the budget
enters through `one_hot_interpolated` rather than through anything learned.
