# HUB CHARTER — `hub/composition`

- **Agent:** `autonomous-hub` · **Branch:** `hub/composition` · **Worktree:** `../CHLU-hub-composition`
- **Base:** ⭐ **the PIN `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`**
- **Boot:** `.claude/autonomous-hub-charter.md` → `.claude/AGENT_PROTOCOL.md` → `.claude/handover_context.md` §7 → this file. ⛔ Nothing else.
- **Report:** `.claude/outputs/hub-composition.md` · ⭐ **running log: `.claude/outputs/hub-composition/LOG.md`, updated as you go**
- **Limits:** ≤ 10 h · ⛔ **≤ 3 scored runs, all reported** · ⭐ CPU default · ⛔ **GPU only for a headline**

---

## ⭐⭐ YOUR TARGET
**A compositional-generation win: held-out combinations rendered CORRECTLY, read by the external classifier.**
Three things together, ⛔ not two of three:
1. **Beat every rival — including `knn_proto` — on the compositional factor.**
2. ⭐ **≥ 60 / 72 DISTINCT VALID *NOVEL* TUPLES** — valid per the external classifier, and **not the tuple of any
   stored atom**. ⛔ Both conditions, or it is a lookup demonstration.
3. ⭐ **Dreaming is INSIDE this target**, not beside it: at β = 0 with the pre-registered finite-temperature grid,
   produce valid novel tuples — **or report dreaming DEAD, in those words.**

## ⛔⛔ THE TWO NUMBERS THAT DEFINE THE DIFFICULTY
| | distinct tuples / 72 |
|---|---|
| ⛔ `clu` codebook read | **9.00 ± 0.82** |
| ⛔ the dynamical read | 13.3 (A) / 16.0 (B) |
| ⚠ `knn_proto` (the ceiling) | **56.00 ± 3.74** — ⛔ **not 72/72**, an earlier docstring said so and was never measured |
| `lace` | 71.67 | 
| ⭐ **your bar** | **≥ 60, VALID and NOVEL** |
⇒ ⭐ **You are asked for a ~4× improvement on our best, on a criterion `knn_proto` itself does not meet.**
⛔ **Know that before you plan.** ⭐ A well-controlled failure that names the obstacle is a result; a soft
re-definition of "novel" is not.

## ⛔⛔ DREAMING IS MEASURED DEAD AT ZERO TEMPERATURE — this is your starting fact
At β = 0 every chain **collapses to ONE distinct tuple of 72** and scores **0.0000** at rung B on 3/3 seeds
(mean pairwise separation ≈ 0, identical final energy from different launches).
⭐ **The cause is not a mystery: the settle runs at `temp = 0.0` — pure zero-temperature descent has no reason to
produce more than one mode.** ⭐ The finite-T grid **`temp ∈ {0.0, 0.005, 0.05, 0.5}` is already pre-registered for
this cell** — ⛔ it may not be introduced as a post-hoc excuse, and ⛔ it may not be widened after you see a result.

## ⭐ §E.4 — THE MECHANISM FACTS YOU INHERIT (measured, ⛔ do not re-derive)
- ⭐⭐ **The launch beats the settled state 0.6852 → 0.2114.** ⛔ **The dynamics are the defect** — descending the
  energy loses accuracy, so the minima are not at task-correct states. ⛔ **More steps makes it worse.**
  ⚠ **0.6852 is at 8× bytes; at matched bytes the launch scores 0.5008 and LOSES to the codebook read.**
- ⭐ **Settling SEPARATES requests the discrete clamp COLLIDED** — 9–11 % of launch pairs sit at distance exactly 0
  (two requests addressing one atom tuple); after settling **0.0–0.2 %** remain collided. ⭐ **That is the one thing
  the dynamics demonstrably do well, and it is a compositional property.**
- ⭐ **Placement is not the lever** (learned placement nulled at 0.0019 vs a 0.10 bar; the inherited ward grouping
  already beats the random floor by 0.0975).
- ⭐⭐ **The codebook costs 90.5 % of `scale` and 100 % of `orientation`** — the largest identified loss.
- ⚠ **Roughly the bottom third of composition-B aggregate is FREE**: a 2.67-tuple constant output scores 0.3272 and
  ties `lace`. ⭐ **Always report distinct tuples beside any aggregate.**

## ⛔ Falsifiers
1. **Held-out combinations rendered correctly, beating every rival incl. `knn_proto` on the compositional factor.**
2. ⭐ **≥ 60/72 distinct VALID NOVEL tuples**, novelty checked against the stored atoms' own tuples.
3. ⭐ **Dreaming: valid novel tuples at some declared temperature, or DEAD.** ⛔ Never a null.
4. ⭐ **The store is load-bearing** — `store_removed` collapses the arm.

## ⭐ What would make this hub's answer worthless
- ⛔⛔ **Redefining "novel" or "valid" to make the bar reachable.** Novel = not a stored atom's tuple. Valid = the
  external classifier says so. ⛔ Neither is negotiable.
- ⛔ **Counting distinct SETTLED STATES instead of decoded factor tuples.** Continuous states are trivially
  distinct and that number is worthless.
- ⛔ **Reporting a dreaming collapse as a null** rather than as DEAD with its temperature named.
- ⛔ A per-factor number without `n_values_realised` — Montero's corner pins `shape` to 1 of 4 values and yields a
  confident-looking zero. **That has produced a wrong headline here once already.**
