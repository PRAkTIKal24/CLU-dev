# PREREG — w19 hub/maze-t-symmetric: does the CMAZE-T win survive SYMMETRIC inputs?

Written 2026-09-23 BEFORE any w19 number exists. Target `maze-t-symmetric`, budget 9 scored runs.
Base `w16/maze-wm` tip 9f80f59. Runner `scripts/w19_symmetric.py`, bars `scripts/w19_bars.py`.

## What is fixed (UNCHANGED from w16 PREREG-T1 + AMENDMENTS 1–4)
Engine, solver, judge, banks (train base..base+200k 90 mazes; VAL base+200k..+300k; TEST
base+300k..+400k, 45 mazes each; base = 2,000,000 + 400,000·seed), hold-outs
`make_holdout(4, seed)`, row caps, frozen codec recipe, the store config **A1 = latent 128,
K 2, 6 atoms, ridge 1.0** (C-512 is NOT re-run: it is A1's replication and the brief
changes only the input), the four charter-C controls with **`landscape_shuffled_across_wells`
a DERANGEMENT (printed per seed)**, the learned blind arms, the lookup floors, the planner
columns, the rival recipe (8,000 steps, alternating single/trajectory batches, same
trajectories rng) and read-path-float matching. Seeds 0, 1, 2 — all or none.

## The only change: the INPUT, identical for every arm within a condition
| condition | CLU request (design) | every rival's input | WORDS lookup key |
|---|---|---|---|
| WORDS-MAIN | true words, [bias|main] (= learned BLIND-SEQUENTIAL) | 27 main-effect one-hots (+ cont-MLP: words as numbers) | true words |
| WORDS-FULL | true words, [bias|main|pair] 318 | the 318-wide design (+ cont-MLP) | true words |
| OBS | words from a DECLARED extractor (obs -> 5 words, TRAIN rows only), [bias|main|pair] | (a) raw obs 634 = conv-AE patch+global ⊕ 29 dyn ⊕ action; (b) obs ⊕ extractor one-hots | extracted words |
| BOTH | true words, [bias|main|pair] | obs ⊕ 318 design | true words |
Extractor accuracy (per word + whole tuple) printed on every OBS row incl. held-out combos.
Rivals matched on read-path floats to the CLU (in OBS: CLU read floats + extractor floats).

## Rows (per condition)
`nway` = unseen n-way (**PRIMARY**) · `seen` · `pair` = unseen 2-way · `value` = value
hold-out · per-step exactness (from the rollout block). Rollout validity printed ONCE as a
limitation. Charter-C controls per condition. Rivals: 8,000-step matched row + 80,000-step
(10×) matched column + 10×-read-floats capacity column (8,000 steps, never a bar).

## VAL bars (per condition; means over seeds 0/1/2 of VAL `nway`, `exact`)
* **VB1 composition**: CLU ≥ R* + 0.10 AND CLU > R* on every seed, where R* = the best
  matched rival over every kind/input variant of the condition AND over BOTH budgets
  (8,000 and 80,000 steps). (A margin is never claimed against under-trained rivals.)
* **VB2 store gate**: every charter-C control ≤ CLU − 0.15.
* **VB3 falsifier**: max(learned BLIND-COLOUR, learned BLIND-SEQUENTIAL) ≤ CLU − 0.05.
  WORDS-MAIN: BLIND-SEQUENTIAL IS the CLU, so VB3 is evaluated on BLIND-COLOUR only (declared).
* **VB4 floor**: WORDS lookup ≤ CLU − 0.10.
A condition passing VB1–VB4 earns 3 scored TEST runs; failing, it is DECLARED (printed row) and
does not consume a run. ⛔ No re-tuning to rescue a condition.

## SCORED bars (TEST, 3 seeds): SB1–SB4 = VB1–VB4 on TEST `nway`.
All hold → **STORE row** for that condition. SB1, SB3, SB4 hold and SB2 fails → INTERFACE row.
Otherwise → LOSS row. Every condition is printed in the condition table whatever its outcome.

## Spending order
WORDS-MAIN, WORDS-FULL, OBS (3 runs each if VAL passes); BOTH gets 3 runs only if runs remain
after those three and BOTH passes VAL. Unspent runs are DECLARED NOT-RUN with the reason.
Ledger row in /c/chlu/SCORED-LEDGER.md BEFORE each scored run.

## Other rows' labels
`seen`, `pair`, `value` reported with the same control gate + falsifier as labels (store /
interface / nothing), not bars. Deletion rows on every scored condition's TEST runs.

## What would prove the w16 win an input artefact
WORDS-FULL: a matched rival given the CLU's exact 318-wide design reaching the CLU on `nway`
(the pair block, not the store, carries the margin). OBS: the CLU losing once its request
must itself be read from the observation the rivals get.

---
## AMENDMENT 1 — 2026-09-23 ~17:40Z, during the OBS VAL runs, BEFORE any OBS row number exists
Observed (fit diagnostic only, no eval row seen): the declared extractor (634→32→27, 6,000
steps) reaches TRAIN whole-tuple accuracy 0.404 / 0.683 / 0.767 (seeds 0/1/2) — under-fit
on its own training rows. The rivals carry a 10×-steps column (the w16 lesson: no margin
against an under-trained arm); the SAME treatment is given to the CLU's extractor:
* **OBS-X10 diagnostic** (VAL only, `scripts/w19_extractor_diag.py`): extractor trained
  **60,000 steps** (10×), otherwise identical; the CLU, controls, blind arms and WORDS floor
  refitted on its words; rivals `obs ⊕ extractor-X10 one-hots` (matched, 8,000 + 80,000 steps).
* ⛔ **The OBS verdict (VB1–VB4) is computed on the DECLARED extractor only.** The X10 column
  is printed beside it and cannot pass the condition or earn a scored run. Nothing else moves.
