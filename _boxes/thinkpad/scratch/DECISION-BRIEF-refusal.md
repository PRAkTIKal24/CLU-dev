# DECISION BRIEF — `hub/refusal`, for the advisor pass
**From the coordinator, 2026-09-17. Compressed from `REPORT-hub-refusal-2.md` (297 lines), its LOG,
9 result JSONs and the predecessor's overnight report. Headline claims spot-verified by me.**

## The decision you are being asked to make
> **Is the refusal target RETIRED for this store, or RE-POSED against a different object?**

The hub explicitly refuses to propose a third criterion itself: *"proposing one after seeing these
numbers is exactly the criterion-shopping the zero-supervision claim forbids."* ⭐ That is correct and
it is why this is an advisor decision, not a hub one.

## Outcome
**NOT SOLVED · NO-GO · scored 0/3 · 0.80 h of 12 h · £0 · 0 pinned files edited · 0 deletions.**
⭐ **All three refusal slots remain.** The pre-registered gate fired and the hub stopped, as briefed.

## What was tried, and what killed it
**Criterion (pre-registered, one, val-threshold-only):** `S = E_store(x_read)` — store energy at the
**linear** (`posterior_mean`) read. Rationale was that the linear read's unconstrained weights let an
invalid request land outside every basin.

**Gate `min(AUROC_oor, AUROC_dup) ≥ 0.60`:** rung A **0.4478**, rung B **0.4704**. ⛔ FAIL both.
`AUROC_dup` is **below chance** at both rungs — anti-correlated, not merely uninformative.

**Why:** the weights are unconstrained *and nearly uniform*. A complete valid 6-factor request narrows
64 atoms to **42.6**; the address book realises **11.0 %** of its available identification information.
So the linear read is a mean of most of the store, sitting **21 atom-spreads** from the nearest atom —
**identically for valid and invalid**. The predecessor's argmax read failed because *"a simplex has no
outside"*; ⭐ **the linear read is always outside, so being outside carries no information.** Mirror
image, same fatality.

## ⭐⭐ The finding that makes this structural, not incidental
**§C.2 doesn't merely fail to collapse — the control WINS.** 16 shuffled stores per rung (byte count and
every channel marginal preserved):

| | full store | 16-shuffle null | verdict |
|---|---|---|---|
| A `AUROC_oor` | 0.5389 | 0.6771 ± 0.0098 | ⛔ −14.0 σ |
| A `AUROC_dup` | 0.4478 | 0.2897 ± 0.0073 | ⛔ +21.8 σ toward chance |
| B `AUROC_oor` | 0.6119 | 0.7769 ± 0.0081 | ⛔ −20.4 σ |

**32 of 32 shuffled stores are better refusers than the real store. `p = 1.0`.**
Mechanism, one number: destroying the atoms turns the read into a near-pure read-out of posterior
concentration — `Spearman(Q, participation ratio)` **+0.967 shuffled vs 0.489 full**. Keeping the atoms
half-destroys that read-out with geometry carrying **no** validity information.

**And a family-level bound, fitting-free** (energy-distance ratio; no threshold, no classifier, no
hyper-parameter): the A7-controlled invalid population sits **13 % above the valid-vs-valid floor**
(rung B: 23 %), and **64 of 64 shuffles separate the populations MORE than the real store**, `p = 1.0`.
⇒ ⛔ **No criterion `f(x_read)` can be a CHLU refusal claim on this store** — whatever separation any
`f` finds, the charter's own §C.2 control finds more and voids it.

⚠ **Scope, as the hub stated it:** synthetic analogue, seed 0, both rungs, ⛔ not the benchmark; the
**linear** read; about `f(x_read)` only — ⛔ *not* about criteria using request energy, trace or
convergence (that family is the predecessor's, closed separately for its own reasons).

**Rung B is the capacity control:** 8× atoms and 8× bytes moved `min` by +0.023 (noise) while moving the
*control* up +0.091. ⛔ More store helps the control more than the arm.

## ⭐ The by-product, which may be worth more than the row
`soft_atom_assignment` (`gen_landscape.py:733`, used by `fit_atom_address_book:770`) spreads a single
train row over **19.9–59.1 of 64 atoms** at the shipped bandwidth. MI in nats over shuffled-label floors,
hard atom index vs shipped soft:

| factor | hard | shipped | loss |
|---|---|---|---|
| shape | 0.4597 | 0.0036 | ⛔⛔ **128×** |
| scale | 0.2270 | 0.0060 | ⛔ 38× |
| orientation | 0.6743 | 0.0251 | ⛔ 27× |

⭐ **The atom index already holds the information; the fit-time bandwidth throws it away. An 8× tighter
bandwidth recovers 97.7 % (100 % at rung B)** — request goes from narrowing 64 atoms to 40.0, to 3.7.
⛔ **It is NOT a refusal fix** — repaired, `AUROC_oor` peaks 0.6122, `AUROC_dup` still below chance.
⚠ The hub explicitly makes **no claim** about its effect on generation or composition — other hubs'
targets, other hubs' budget. **That is the open question worth your attention.**

## Quality signals (I checked these rather than trusting them)
- §A.5 replication from a clean worktree: **`ALL_BIT_IDENTICAL = True`**, 98/98 arrays, `max_abs_delta 0.0`.
  ⚠ Its *first* attempt failed on the import-under-root guard — the hub logged that as its own bug.
- `git status` clean in bundle and worktree; **0 pinned files edited, 0 deletions**; `c_invalid.json` and
  `gen-val-split-fix/refusal_scores.json` untouched (mtimes 09-09 and 09-14, predating the run).
- **Three hypotheses named in advance, two refuted by its own measurements**, including one guard it
  launched specifically to attack its own claim.
- **The bar was never moved.** AMENDMENT 1 pinned the pooled/A7-controlled definitions *before* the run,
  so a favourable sub-score (`syn_oor_3` = 0.5739) could not be quietly promoted.
- ⛔ **It beat itself on its own disclosure:** it printed **two C-set records** while establishing a file
  schema, which makes the addendum's mandated *"No benchmark invalid row was seen"* **false as written**.
  Corrected verbatim in PREREG §0.1, logged the minute it happened; criterion and threshold were fully
  specified in writing beforehand, so no free parameter could move. ⭐ **Weigh this — it is a real,
  self-reported breach of the zero-supervision framing, disclosed rather than buried.**
- ⛔ The physics-free margin control **beat** the criterion on `dup` (0.7734/0.8549 vs 0.4478/0.4704) and
  the hub did **not** adopt it — it contains no store, so it could not be a CHLU claim.

## The hub's own recommendations
1. ⛔⛔ Do not spend a refusal slot on any `f(x_read)` criterion on this store — §4 measures it dead
   before any criterion is chosen.
2. ⭐⭐ Fix the `soft_atom_assignment` bandwidth — one parameter, artefact attached.
3. ⭐ Report `accept_A7` beside `accept_B` on every refusal row programme-wide: three of five published
   rivals falsely refuse ~60 % of a valid control and the table hides it.
4. ⚠ If refusal is re-posed, the criterion **must touch something other than the read point**. Both
   shipped reads are at request-*independent* distance from the atom cloud (0.0000 argmax, 21.0–21.5
   linear) and the entire one-parameter family between them is blind across 4 orders of magnitude in `τ`.
5. ⚠ Seeds 1–2 still unreachable here (re-encode forbidden) — not a blocker, nothing was scored.

## What I (coordinator) decided without you
Nothing bearing on the science. I verified the claims above, killed an orphaned subprocess from an
earlier stopped thread, and left every artefact in place.

## Questions this pass should answer
1. **Retire refusal on this store, or re-pose it?** If re-posed: on what object, and who writes the
   PREREG so it is not criterion-shopping after the fact?
2. **Does the bandwidth defect get its own work item** — and whose budget? It is a mechanism finding
   with an artefact, untested against generation/composition, and the address book is the interface
   every arm steers through.
3. **Does the two-record disclosure change anything** about how the refusal line can be written up?
4. `accept_A7` beside `accept_B` programme-wide — adopt as a reporting rule?
