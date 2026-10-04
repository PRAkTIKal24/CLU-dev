# AMENDMENT 2 to `PREREG.md` — "deteriorates" and "breaking point", defined before the numbers

**Written 2026-09-21T11:46:30Z** (`AMENDMENT-2.timestamp.txt`). At that instant
`C:/chlu/outputs/w12-composition-ladder/runs/*/rows_*.json` matched **nothing**: no arm of
this hub had produced a score, on any rung, at any seed. The task asks for a *breaking point
per arm*, and a threshold chosen after the curve is visible is not a threshold. So both words
are fixed here, in advance, out of quantities that were already pre-declared elsewhere.

## 1. The scale problem that forces a normalisation — measured, not assumed

The metric is a **count** of distinct valid novel tuples per corner, and a corner cannot yield
more distinct tuples than it has rows. `instrument/INSTRUMENT.json` measures the rungs' corner
sizes: **36 rows at k = 2** and **72 rows at k = 3, 4, 5, 6**. The published tables already say
this in their own words — `/c/chlu/evidence/outputs/w10-seeds-and-tables/TABLES.md`:
*"row 3's ceiling is 36 tuples/corner and row 4's is 72, so the two rows are not on one
absolute scale either."*

⇒ **A raw k = 2 count and a raw k ≥ 3 count are not comparable.** Every table and the figure
print the raw count (it is the estimand's own currency and the margin +1.0 lives there) **and**
the ceiling, and the definitions below use the **ceiling-normalised** score
`s̃(arm, k, seed) = mean distinct_valid_novel_tuples / ceiling(k)` wherever a comparison
crosses k = 2 ↔ k ≥ 3. ⛔ The normalisation is never used to replace the raw number, only to
make a cross-k statement legal.

## 2. DETERIORATES — the ONE QUESTION's word

> An arm **deteriorates at rung k** (k ≥ 3) iff `s̃(arm, k, seed) < s̃(arm, 2, seed)` **on every
> seed**.

Three seeds, all of them, or the statement is not made. If the sign is not the same on all
three, the rung is reported as **"no consistent direction across seeds"** — ⛔ not as a
deterioration and ⛔ not as a non-deterioration.

## 3. BREAKING POINT — built only from the harness's own pre-declared bar

The bar is the harness's `beats_beyond_margin`
(`exp_gen_factor_wells.paired_estimand`): mean paired per-corner delta **> +1.0** *and* the
paired-bootstrap CI95 **lower bound also > +1.0**. The opponent is the best learned baseline
under `PREREG.md` section 5, which is fixed per (rung, seed).

> An arm **clears the bar at rung k** iff `beats_beyond_margin` is true against that rung's
> best learned baseline on **every** seed.
>
> The arm's **breaking point** is the **smallest k ≥ 3 such that the arm clears the bar at
> every rung below k and does not clear it at k**.

Three ways there is no breaking point, and each is reported in its own words rather than as a
number:

- **"no breaking point in 2 ≤ k ≤ 6"** — the arm clears the bar at every rung that ran;
- **"never cleared the bar"** — the arm does not clear it at k = 2, so there is nothing to
  break; the arm's whole curve is reported and no breaking point is claimed;
- **"non-monotone"** — the arm fails the bar at some k and clears it again at a larger k. The
  first failure is reported as the **first failure**, explicitly not as a breaking point, and
  the full pattern of clears/fails is printed.

## 4. ⛔ The control gate comes FIRST, and it can remove a rung from the question

For the two store arms, a rung at which **either** charter-C.2 control fails to collapse is
**DECORATIVE at that rung** (`PREREG.md` section 4). Such a rung:

- has **no row claimed** for that store arm;
- is **excluded from the breaking-point determination** for that arm, and the exclusion is
  printed beside the breaking point rather than folded into it;
- still has its raw number reported, labelled DECORATIVE.

⛔ If a store arm is decorative at k = 2 it has no breaking point at all, because it never had
a row to lose.

## 5. What this amendment does NOT change

The metric, the margin, the estimand, the bootstrap, the sign test, the seed count, K = 5, the
arm list, and the best-learned-baseline rule are all untouched. This amendment adds only the
two words the task asks to be reported in, and it adds them before there is a number that
could shape them.
