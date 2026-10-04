# CORRECTION to `AMENDMENT-3.md` — the rule stands, its stated reason was wrong

**Written 2026-09-21T13:05:53Z** (`AMENDMENT-3-CORRECTION.timestamp.txt`). This corrects a
claim this hub made about its own evidence, in the direction that removes a flattering
comparison.

## What `AMENDMENT-3.md` said at 11:49:46Z

It widened the best-learned-baseline candidate set from the five pre-existing rivals to all
seven learned arms, and justified it like this:

> *"Fitting the learner on the pool the arm is fitted on is worth more than any architecture in
> spoke A's set."*

on the evidence that `learned_ref_ownpool` scored **8.2167** at k = 3 seed 0 while
`/c/chlu/evidence/outputs/w10-seeds-and-tables/TABLES.md` prints `rival_mlp_180k @ alone` at
**0.006** on row 3.

## ⛔ That comparison was CROSS-INSTRUMENT and the conclusion from it is false

The 0.006 was produced by the shortened **`w8_v1_short`** decoder. That table says so in its own
words — *"NOT comparable with the incumbent's row-3/row-4 values (a shortened decoder recipe is
a different instrument, not a different setting)"* — and this hub had already quoted that
sentence in `LOG.md` before leaning on the number anyway. Every leg here runs the **incumbent**
frozen decoder `01ef91f75eea (v1_capacity_schedule)`.

Measured on the same instrument, at k = 2, all three seeds (`LADDER.json:rungs.2.arms`):

| arm @ the 524,288 B reference pin | pool | across-seed mean | per seed |
|---|---|---|---|
| `rival_lstm` | montero_combination | **20.8333** | 19.3333 / 23.7500 / 19.4167 |
| `rival_mlp` | montero_combination | **20.7833** | 17.0167 / 22.0000 / 23.3333 |
| `rival_transformer` | montero_combination | **17.5944** | 12.7500 / 15.7833 / 24.2500 |
| `learned_ref_ownpool` | **the rung's own pool** | **7.4833** | 5.5667 / 7.5833 / 9.3000 |

⇒ **Own-pool fitting at the reference pin is WEAKER, not stronger — by roughly 13 points.**
The plausible reason is coverage: the montero_combination pool contains every factor value,
while a rung's own pool has a whole hue and a whole shape carved out, so the own-pool learner
has strictly less to learn the corner's factors from. ⛔ The sentence quoted above is withdrawn.

## ⭐ The RULE is unchanged, and it was right for a reason I had not measured yet

`AMENDMENT-3`'s rule — *per (rung, seed), the learned arm with the largest mean, over all seven
learned arms* — is a **maximum over a superset**. It can only ever raise the opponent, so it
cannot flatter this hub whatever the members score. That argument never depended on the
withdrawn sentence and stands.

And the widened set turns out to matter enormously — but through **`learned_1x`**, not through
`learned_ref_ownpool`. At k = 2 the 1×-byte-matched learner is the best learned baseline on
**every** seed and it **beats both store arms**:

| seed | `learned_1x` | params | store arms | paired Δ (store − 1×) | CI95 |
|---|---|---|---|---|---|
| 0 | **33.2000** | 3,241,076 | 28.6333 | **−4.5667** | [−6.383, −2.883] |
| 1 | **33.8667** | 3,161,048 | 31.6500 | **−2.2167** | [−3.367, −1.200] |
| 2 | **32.9000** | 3,131,564 | 31.3333 | **−1.5667** | [−2.717, −0.500] |

Every CI95 upper bound is below zero. ⛔ Had this hub kept the narrow five-rival rule it would
have reported +9.2167 / +9.6500 / +7.0833 and called it a win.

## What this changes in the record

- `AMENDMENT-3.md`'s **rule**: unchanged, still in force, still the thing the figure's panel B
  and every verdict use.
- `AMENDMENT-3.md`'s **stated motivation**: withdrawn and replaced by the table above.
- ⛔ No number moves. Both readings continue to be printed on every row — `best_learned_all`
  (the rule) and `best_learned_5` (the narrow reading as written at 11:38:41Z) — so a reader
  can see exactly how much of the store's apparent margin was the baseline's byte pin.
