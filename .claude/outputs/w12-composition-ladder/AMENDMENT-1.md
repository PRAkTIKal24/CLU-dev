# AMENDMENT 1 to `PREREG.md` — the 1× byte matching becomes a RUN, not a NOT-RUN

**Written 2026-09-21T11:42:09Z** (`AMENDMENT-1.timestamp.txt`). At that instant
`C:/chlu/outputs/w12-composition-ladder/runs/*/rows_*.json` matched **nothing** — no arm of
this hub had produced a score, on any rung, at any seed. This amendment therefore moves a
declared NOT-RUN **into** the run, before any number exists that could motivate it.

## What changed, and why

`PREREG.md` section 9 declared the **1× byte-matched learned column** a NOT-RUN, on the
ground that every fitted rival on this box is matched to **131 072 float32 parameters** and
no 1×-matched checkpoint exists.

⭐ **That ground was wrong, and the measurement that shows it is another spoke's.**
`/c/chlu/outputs/w12-deletion-baselines/comparators/seed_0/*.rival.json` reports its
learner fits costing **8.5 – 11.3 s** each (`cost_line.seconds`: `cont_mlp__as_trained`
11.2617 s, `cont_mlp__retrain_without__value-object_hue-0` 8.8698 s, at 16 384 rows and 20
epochs). A learner at this byte pin is **seconds**, not hours. So a 1×-matched learner is not
unavailable — it simply had not been fitted.

## The two columns this amendment adds

Both are fitted by `scripts/w12_ladder_1x.py` from `chlu.eval.rivals.w4_learners`'s own
`builders` / `match_width` / `train`, on **each rung's own fit pool**:

| arm | parameter target | fit rows | what it fixes |
|---|---|---|---|
| `learned_1x` | the **measured** total bytes of `single_design_K5` **on that leg** ÷ 4 (atoms + sufficient statistics + address book, never netted) | that leg's own pool, first `PINS['n_store_train'] = 32 768` rows — **the same rows the store's content is fitted from** | the missing **1× byte matching** |
| `learned_ref_ownpool` | **131 072** parameters = the 524 288 B REFERENCE pin | the same 32 768 rows | the **population** caveat: a reference-pin learner fitted on the rung's OWN pool instead of spoke A's `montero_combination` pool |

Fixed now, before any number:

1. **Family = `mlp`** (one-hot request in, 2048-d feature out), the family
   `w4_learners.builders` exposes and the one spoke A's reference column already uses. One
   family, not a sweep: a sweep would be a selection on the scored set.
2. **Width** is chosen by `w4_learners.match_width`, which measures the built model's
   parameter count rather than a formula, within its own shipped `PARAM_TOLERANCE = 0.10`.
   ⛔ If no width lands inside that band the arm is a **DECLARED NOT-RUN with the closest
   achievable count printed on each side** — `match_width` raises rather than shipping a
   mismatched rival, and that refusal is reported as the result.
3. **Optimisation** = `w4_learners.train` at the shipped `BATCH = 256`, 20 epochs,
   `lr = 1e-3`, seeded by the leg seed. No tuning grid and no selection set.
4. **These two arms do NOT change the best-learned-baseline rule of `PREREG.md` section 5.**
   That rule is over the five pre-existing learned arms and it stays exactly as written, so
   panel B of the figure is unchanged. `learned_1x` and `learned_ref_ownpool` are reported as
   their own columns, with their own margins, and are named as **this hub's own fits**.
5. **Declared cost, against this hub's interest:** a 1×-matched learner carries roughly
   **7–8× the parameters** of the reference-pin rivals and is therefore **stronger**, not
   weaker. That is the point of running it.

## What is still NOT-RUN after this amendment

- A 1×-matched **LSTM / transformer / cont_mlp / table**. One family is fitted, declared;
  the other three are a NOT-RUN with wall clock as the reason.
- dSprites, unchanged from `PREREG.md` section 9.
