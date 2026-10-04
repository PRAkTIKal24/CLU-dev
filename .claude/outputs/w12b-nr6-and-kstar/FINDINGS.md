# w12b hub/nr6-and-kstar — FINDINGS

**Report to the Hub.** Branch `w12b/nr6-and-kstar`, base `base/iter8 = e58ab05`.
Box FORGIS_14, local CPU, **3 lanes × 1 thread** throughout.
`chlu.__file__ = C:\chlu\worktrees\CHLU-w12b-nr6-and-kstar\chlu\__init__.py`, asserted at the
top of every process and written into every artefact.
⛔ `hub/encoder-adapt` was TRAINING on this box for most of this run. Its session, worktree,
output directory and PIDs were never touched, and **there is no box-wide kill anywhere in
this hub's record.**

**Scored spend: 2 runs** (`single-design` **2/3** and **3/3**). Both ledger rows were written
to `/c/chlu/SCORED-LEDGER.md` at **2026-09-21T21:33:51Z**, before either process started.
The K\* ladder replication is **free** and carries no row.

Every number below names the file it came from. ⛔ Anything not run is a **DECLARED NOT-RUN**
with a reason — never a zero, never a blank.

---

## 0. THE SCOREBOARD

| job | verdict | scored |
|---|---|---|
| ⭐ **NR6 — T7 at more than one seed** | **T7 carries n = 3 seeds; the seed-0 headline sits inside the spread** | 3/3 |
| ⭐ **NR6 — the missing K = 5 TEST number** | **bought; the val/test gap does NOT fire** | 2/3 |
| ⭐⭐ **The K\* replication of the ladder** | ⭐⭐ **THE VERDICT REVERSES — the store clears at ALL FIVE rungs at K\***, where at K = 5 it cleared at none | 0 (replication) |
| ⭐ The BRIDGE arm (this hub's addition) | **the flip is the K step, not the construction** — the bridge at K = 5 still loses at 4 of 5 | 0 |

**Acceptance criterion — all three clauses MET:**
1. T7 carries **≥ 3 seeds with the count printed** — 3 seeds (0, 1, 2), printed on the row.
2. The **K = 5 TEST number exists** — `scored/run2_seed0_K5/scored.json`.
3. The ladder's store arm has a **K\* = 32 reading at all five rungs** with `learned_1x`
   refit at **536,270–536,972 params** (the task file's 535,784 pin).

---

## 1. ⭐⭐ THE HEADLINE: THE K\* REPLICATION REVERSES THE LADDER'S VERDICT

**The ladder's own pre-declared bar, evaluated at both K.** `AMENDMENT-2 §3` of
`w12 hub/composition-ladder`: the arm clears only if it beats the opponent by
`DECORATION_MARGIN_NOVELTY = 1.0` novel tuple per corner **on every declared seed**, not on
the seed mean. ⛔ **The bar is not moved here.** It is evaluated at the other K as well.

| rung | clears at K = 5? | worst seed K=5 | clears at **K\* = 32**? | worst seed K\* |
|---|---|---|---|---|
| n=2 | no | −4.567 | **YES** | **+3.117** |
| n=3 | no | −0.033 | **YES** | **+11.533** |
| n=4 | no | −6.350 | **YES** | **+8.625** |
| n=5 | no | −4.808 | **YES** | **+6.517** |
| n=6 | no | −5.775 | **YES** | **+6.108** |

`KSTAR-LADDER.json:seed_matched_K5_vs_Kstar`, `KSTAR-LADDER.md`.

⇒ The ladder's published finding was *"There is NO BREAKING POINT, because the store **NEVER
CLEARED THE BAR**."* ⭐⭐ **At K\* it clears at all five rungs on every seed.**
**All 15 per-seed K\* margins are positive; the smallest is +3.117.**

### The seed-matched table, 3 seeds
⛔ Seed-matched throughout: a 3-seed mean is never compared against a 1-seed mean, because
that would charge a seed difference to K.

| rung | store K=5 | store K\* | opp K=5 params | opp K\* params | shrank | margin K=5 | margin K\* | flipped |
|---|---|---|---|---|---|---|---|---|
| n=2 | 30.539 | **32.644** | 3,177,896 | 536,972 | 5.92× | −2.783 | **+4.356** | **YES** |
| n=3 ⚠ | 59.750 | **61.614** | 3,174,386 | 536,270 | 5.92× | +4.242 | **+14.072** | no (already +) |
| n=4 ⚠ | 62.708 | **66.086** | 3,214,400 | 536,270 | 5.99× | −5.283 | **+10.022** | **YES** |
| n=5 ⚠ | 64.719 | **66.581** | 3,177,194 | 536,972 | 5.92× | −3.900 | **+8.356** | **YES** |
| n=6 ⚠ | 64.556 | **66.603** | 3,180,704 | 536,972 | 5.92× | −4.061 | **+7.978** | **YES** |

⭐ **Both halves move, and it matters which.** The store's own score **rises at every rung**
(+1.86 to +3.38 novel tuples per corner) — the finer partition genuinely reads better, and
that is not a baseline artefact. The opponent **falls 5.92–5.99×** because it is byte-matched
to an object whose four-part total is that much smaller.

### ⛔ THE CONTROL GATE, CHECKED BEFORE ANY MARGIN — all 15 K\* legs
All 15 collapse by the pre-declared bar. **13 of 15 read exactly 0.0000.**
⚠ **Two do not: n=3 seed 0 and n=3 seed 1, `landscape_shuffled` = 0.0083** — one novel tuple
surviving across 120 corners, `judge_accuracy 0.0001`. ⛔ **Printed, not rounded away.**
Against arms of 60.150 and 62.783 they collapse by a factor of ~7,200.
⛔ Two different things are kept apart and neither is moved: *"exactly 0.0000"* was the
crossover hub's **observation** at K\* on val and was never declared as a bar; the
**pre-declared bar** is `LADDER-READY.json`'s `DECORATION_MARGIN_NOVELTY = 1.0`.
All 5 bridge legs read exactly 0.0000 on all three controls.

---

## 2. ⭐⭐ THE BRIDGE: THE FLIP IS **K**, NOT THE CONSTRUCTION

Going from the ladder's K = 5 arm to the K\* arm changes **two** things at once — the well
count **and** the construction (the ladder's arm is a `DowndatableStore` on the **label-MI
`factor`** partition; K\* is defined by the crossover hub's frozen rule on the **label-free
`ward_balanced`** cut of a `CovContentStore`, with a different covariance convention).

⭐ So this hub built a **BRIDGE** arm nobody asked for: the crossover construction held fixed
with **K set back to 5**, on the same rungs and the same pools.

| rung | seed 0 margin: ladder K=5 → **BRIDGE K=5** → K\* | construction step | **K step** | K's share |
|---|---|---|---|---|
| n=2 | −4.567 → **−0.250** → +6.667 | +4.317 | **+6.917** | 62 % |
| n=3 | +4.817 → **+8.425** → +17.875 | +3.608 | **+9.450** | 72 % |
| n=4 | −6.242 → **−1.017** → +12.700 | +5.225 | **+13.717** | 72 % |
| n=5 | −4.808 → **−0.517** → +10.350 | +4.292 | **+10.867** | 72 % |
| n=6 | −5.775 → **−2.550** → +9.375 | +3.225 | **+11.925** | 79 % |

⭐⭐ **THE DECISIVE FACT: the bridge STILL LOSES at 4 of 5 rungs** — it wins only at n=3, the
**identical pattern to the ladder's own K = 5 verdict**. ⇒ **Changing the construction alone
does NOT flip anything. The flip is the K step**, worth 62–79 % of every swing.

⛔ Without the bridge the entire swing would have been charged to K, and the claim would have
been unattributable. With it, *"K\* flips the ladder's verdict"* is **measured**, not inferred.

---

## 3. ⛔⛔ A CORRECTION THE ADVISOR REPORT NEEDS

`w12-advisor-report.md:146` states: *"the ladder's verdict stands against a **1,029,508-param**
opponent where the main text will use 535,784."*

⛔ **Measured, the ladder's `learned_1x` carried 3,131,564 – 3,260,030 parameters**, read
from `w12-composition-ladder/runs/<rung>_seed<s>/rows_learned_1x.json:summary.param_count` on
all 15 legs (per-rung 3-seed means 3,174,386 / 3,174,386 / 3,214,400 / 3,177,194 /
3,180,704). The ladder's own FINDINGS says so in words — *"a learner matched to its own
**24.4×** total"*.

**1,029,508 is a different object's number.** It is the **crossover construction's** K = 5
four-part total in params, from `LEDGER-ACROSS-K.json` — ⚠ the same file whose first key is
`SMOKE_NOT_A_CLAIM: true`, and which the advisor report's own §6 already had to correct once.
The two objects price the same statistic differently:

| | K = 5 statistics | convention |
|---|---|---|
| ladder `DowndatableStore` | **12,284,920 B** | FULL-SQUARE float64, `8·Σd_k²` |
| crossover `CovContentStore` | **3,391,552 B** | symmetric-TRIANGLE float64, `8·Σd_k(d_k+1)/2` |

and they sit on different partitions. ⭐ **This hub's bridge leg proves it directly**: built
on the ladder's own rung-2 pool, the crossover construction at K = 5 measures a four-part
total of **4,118,034 B → 1× target 1,029,508 params** — the advisor's figure, exactly.

⇒ ⭐ *"The opponent is roughly HALVED"* is **true within the crossover convention**
(1,029,776 → 536,270–536,972 = **1.92×**, and that is precisely the bridge → K\* step) **and
is not a statement about the ladder's opponent.** Against the ladder's actual opponent the
K\* figure is **5.92–5.99× smaller**.

---

## 4. ⭐ NR6 — T7 NO LONGER PRINTS `n = 1`

`NR6-T7.json` / `NR6-T7.md`, built by `scripts/w12b_nr6_table.py` from the scored artefacts
only. Seed 0 is the w12 hub's own `run1_seed0_K32/scored.json`; seeds 1 and 2 are this hub's
run 3/3.

| K | n seeds | row 3 (pair, 60 corners) | row 4 (n-way, 49 corners) |
|---|---|---|---|
| **32 = K\*** | **3** (0, 1, 2) | **32.350 ± 0.815** (31.600 / 32.233 / 33.217) | **67.177 ± 1.320** (66.959 / 65.980 / 68.592) |
| **5** | **1** (0) | **28.817** | **62.122** |

⭐ **The seed-0 headline sits INSIDE the three-seed spread on both rows. T7 is not a one-seed
artefact.**

### ⛔ The control gate, all four legs
| leg | arm | `landscape_removed` | `landscape_shuffled` | `landscape_shuffled_across_wells` | `random_atoms` | pass |
|---|---|---|---|---|---|---|
| K32 seed 0 | 47.4954 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |
| K32 seed 1 | 47.4037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |
| K32 seed 2 | 49.1193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |
| K5 seed 0 | 43.7890 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |

⚠ **A control that collapsed at seed 0 was not assumed to collapse at seeds 1 and 2 — it was
measured, and it did.** The verdict function is `fw.paired_estimand`, the corner is the unit,
and on every leg every comparator is beaten beyond the 1.0 margin on **109 of 109 corners**;
the smallest paired delta against any comparator is **+35.560** (the `table` FLOOR, K5 seed 0).

### ⭐ The K\* = 32 four-part ledger across seeds
| seed | atoms | cov+count | address book | coupling | honest total | 1× params |
|---|---|---|---|---|---|---|
| 0 | 516,096 | 560,608 | 1,060,864 | 6,610 | 2,144,178 | 536,044 |
| 1 | 516,096 | 560,608 | 1,054,104 | 6,610 | **2,137,418** | **534,354** |
| 2 | 516,096 | 560,608 | 1,058,264 | 6,610 | **2,141,578** | **535,394** |
| (val pin) | 516,096 | 560,608 | 1,059,824 | 6,610 | 2,143,138 | 535,784 |
⭐ **The task file's 535,784 pin holds across all three seeds to within 0.32 %.** Only part 3,
the address book, varies with the seed; parts 1, 2 and 4 are arithmetic on the partition shape.
⛔ Summed and never netted. ⚠ The ledger is U-SHAPED in K with its minimum at K = 24, so
**K\* is not the cheapest point** — that stands from wave 12 and is not improved here.

### ⚠ Seeds 1 and 2 carry rank-deficient wells the seed-0 row never showed
`rank/width` at K\*: seed 0 **0.9688–0.9844 uniformly**, seed 1 **0.5938–0.9844**, seed 2
**0.4375–0.9844**. The saturation check still passes at the maximum, and the controls still
collapse — but the one-seed row understated the spread of the object's geometry.

---

## 5. ⭐ THE VAL/TEST GAP — the third pre-declared falsification, now testable, DOES NOT FIRE

The crossover hub declared it a NOT-RUN in its own words: *"the third pre-declared
falsification is untestable while only K\* has a TEST row."* Run 2/3 bought the second point.

The frozen rule is `K* = max{K : v(5) − mean_seeds v(K) ≤ 0.02}`, where `v` is the mean over
corners of the per-corner valid fraction. ⚠ **The corner SETS differ** (24 val corners vs
60 + 49 test corners), so levels are never subtracted across conditions; the **drop within a
condition** is the rule's own quantity.

| condition | v(K=5) | v(K=32) | drop v(5) − v(32) | inside the 0.02 band? |
|---|---|---|---|---|
| val (3 seeds) | 0.89525 | 0.92901 | **−0.03376** | yes |
| TEST (paired, seed 0) | 0.82849 | 0.90125 | **−0.07276** | yes |

⇒ ⭐ **The drop is NEGATIVE on both conditions — K\* is BETTER than K = 5 on the held-out
condition, and by a WIDER margin there than on the selection set.** The rule's band is a
one-sided bound on a drop, and there is no drop to bound. ⛔ No adverse val/test gap.

---

## 6. ⭐ THE TWO TOKEN FAMILIES — pre- vs post-fix, MEASURED not asserted

The wave's second binding instruction. The `w4_learners.train` dtype repair
(`_as_model_input`) is at **`1635e0a`**, an **ancestor of this base `e58ab05`** — verified
with `git merge-base --is-ancestor`. The second, functionally equivalent repair **`8e5f202`**
lives on `w13/encoder-adapt` and is **NOT** in this base.

⭐ Not relied on: `transformer` and `lstm` were **re-fitted from scratch inside each scoring
process**, at the same 131,072-param pin (width 45 / 130,433 and width 51 / 130,721 —
identical to the loaded comparators' pins), on the same 16,384 fit rows, 327,680 gradient
row-visits. So their numbers **post-date the fix by construction**, not by assertion.

| leg | `transformer` loaded → refit (row3/row4) | `lstm` loaded → refit (row3/row4) |
|---|---|---|
| K5 s0 | 0.667/1.918 → **0.367/1.102** | 2.500/7.531 → **2.567/7.714** |
| K32 s1 | 0.667/0.633 → **0.300/1.143** | 2.700/7.000 → **3.083/5.020** |
| K32 s2 | 1.717/4.735 → **1.733/3.163** | 4.917/14.898 → **4.367/8.061** |

⇒ ⭐ **The re-fits land in the same band as their loaded twins on every leg.** The
`w12-deletion-baselines` comparators therefore genuinely post-date the repair and are **not**
the flat pre-fix artefacts the wave warned about.
⛔ **The artefact warning belongs to SPOKE A's wave-4 set** (`rival_transformer` 0.489–0.531,
`rival_lstm` 0.579–0.604), which the **ladder** scores and which **no row in this hub uses**.

---

## 7. ⛔ THE AXIS IS LABELLED HONESTLY

`w12 spoke/seeds-and-tables` measured that rungs **n = 3, 4, 5, 6 probe the IDENTICAL 120
corners**; what varies across them is how many rows the cell removed from the fit pool. ⛔ For
n ≥ 3 this is a **training-data axis, not a composition-depth axis**, and only **n = 2 vs
n = 6** is a genuine depth contrast. The rungs are reported and the axis is named for what it
is, on every table this hub writes (`KSTAR-LADDER.json:axis_label`, and a ⚠ marker on each
affected row).

⭐ **The flip is present at BOTH ends of the only genuine contrast** — n=2 (−2.783 → +4.356)
and n=6 (−4.061 → +7.978) — so it does not rest on the confounded middle rungs.

---

## 8. ⛔ A DEFECT OF THIS HUB'S OWN, FOUND BEFORE IT COST A NUMBER

`w12b_scored_leg.py` passed `assert_generation2(seed)`'s dict to `save_comparator`. That dict
carries only `{generation, scope, n_rows, root}` — **no `features_sha256`**, which
`w4_api.save_learner` requires. ⚠ The refit block's per-family `try/except` would have
**swallowed the `KeyError`**, the scored rows would have completed looking perfectly healthy,
and the two token families — the one thing §6 exists to settle — would have come back as a
NOT-RUN with a misleading reason. The smoke test missed it because it passed
`fw.feature_cache_manifest` directly.

Fixed in **both** runners with the reason written at the call site. **Restart by PID and only
by PID**: my own 34684 / 8736 / 35684 were stopped with `Stop-Process -Id`, and
`hub/encoder-adapt`'s **33948 / 21076 / 29968 / 27236 were verified still running immediately
afterwards**. ⛔ No box-wide kill. Cost: ~6 minutes, none of it a number.

---

## 9. ⛔ DECLARED NOT-RUNS

1. **K = 5 on TEST at seeds 1 and 2.** The run 2/3 ledger row declared seeds 0, 1, 2; **seed 0
   was run**. Reason: 3 usable threads shared with `hub/encoder-adapt`, ~100 min per K = 32
   leg (address book alone 2,000 s at one thread), and the acceptance criterion's ladder
   clause needed every thread. ⇒ **K = 5 on TEST is reported as n = 1, printed, never padded**,
   and the §5 val/test gap is a **seed-0 paired** comparison, stated as such.
2. **The BRIDGE arm at seeds 1 and 2.** Run at **seed 0 on all five rungs**; the §2
   decomposition is therefore a **single-seed** attribution and is labelled as one. The K\*
   column itself carries all 3 seeds.
3. **The ladder's STRONGER `random_atoms`** (random atoms with the address book **refit**
   against them). The crossover hub's form was used instead — perturbation after the book is
   fitted, read through a basis of its own atoms at the book's widths. Reason: at K\* the book
   costs ~2,000 s of single-thread CPU per leg and refitting it per control per leg was not
   affordable. ⚠ The weaker form is the conservative direction to report only because it
   collapsed anyway — all 15 legs.
4. **`rival_transformer` / `rival_lstm` re-fitted on the ladder's rungs.** The machinery was
   built (`--arms rivals_refit` in `w12b_ladder_kstar.py`) and not run: the rows this hub
   produces are store and `learned_1x` rows, and those two families do not appear on them.
   ⛔ The ladder's published `rival_*` columns are still spoke A's pre-fix numbers and this
   hub does not repair them.
5. **Seeds 3 and 4, everywhere.** See below — this is a measurement, not an omission.

---

## 10. ⛔ SEEDS 3 AND 4: THE TASK EXPECTED ONE PROBLEM AND THERE ARE TWO

The task file anticipated that **seed 4** has no external classifier. Measured today, **it
holds for seed 3 as well**, at the same two paths:

| seed | cache admission rule | `decoder_v1_s<n>.pkl` | `gen-external-classifier/seed_<n>` |
|---|---|---|---|
| 0, 1, 2 | shipped gen-2 | PRESENT | PRESENT |
| **3** | **PASSES** — `array_shape [480000,128,4,4]`, `n_blocks 59`, 59 block hashes, `status READY` | ⛔ **ABSENT** | ⛔ **ABSENT** |
| **4** | **PASSES** — same | ⛔ **ABSENT** | ⛔ **ABSENT** |

`winner_hashes.json`'s keys are exactly `['seed_0','seed_1','seed_2']`, and
`gen-external-classifier/` contains exactly `seed_0|1|2`.

⇒ ⭐ **Both seeds now clear the cache rule and NEITHER can be judged.** A number produced on a
different decoder and a different classifier is a number on a different instrument, and this
programme's rows are comparable only because one instrument produced all of them.
⇒ ⛔ **The count is 3 (seeds 0, 1, 2), printed on every row, and it is never padded.**

---

## 11. WHAT WOULD PROVE THIS HUB WRONG — checked, and the outcome

| the task file's falsifier | outcome |
|---|---|
| The controls fail to collapse at K\* on a scored row ⇒ claim nothing | ⭐ **They collapse.** All 4 scored legs exactly 0.0000 on all four controls; all 15 K\* ladder legs collapse by the declared bar, 13 of 15 exactly zero, the two exceptions printed. |
| The K\* replication flips the sign ⇒ that is the wave's headline, reported as a correction | ⭐⭐ **IT FLIPS**, at 4 of 5 rungs, and under the ladder's own every-seed bar the store goes from clearing at **no** rung to clearing at **all five**. Reported as §1, first, not buried. |
| Seed 4 remains unjudgeable ⇒ declare 4 seeds and say why; never pad to 5 | ⛔ **Seed 3 is unjudgeable too.** 3 seeds declared, with the two absent paths named. |
| (this hub's own AMENDMENT 2) The K = 5 TEST number contradicts the val number ⇒ K\*'s provenance is damaged | ⭐ **It does not.** The drop is negative on both conditions. |
| (this hub's own AMENDMENT 2) A re-fitted token family scores far above its loaded twin ⇒ the comparators are pre-fix artefacts | ⭐ **They do not.** Same band on every leg. |

---

## 12. LIMITS

Two scored runs · **K = 5 on TEST is one seed** · the **bridge decomposition is one seed** ·
for n ≥ 3 the ladder remains a **training-data axis** · the ladder's `random_atoms` is the
weaker post-fit form at K\* · the ladder's own `rival_transformer`/`rival_lstm` columns are
still spoke A's pre-fix numbers and are **not** repaired here · the four-part ledger is
**U-shaped in K and K\* is not its minimum** (K = 24 is) · seeds 3 and 4 cannot be judged at
all, so **no row in this programme can exceed 3 seeds** until a decoder and an external
classifier exist for them.

---

## 13. WHAT THE HUB IS ASKED TO RULE ON

1. ⭐⭐ **The K = 5 ladder verdict must be withdrawn from the main text** and replaced by the
   K\* reading, per the Head's own one-K ruling. The K = 5 table goes to the appendix **with
   its sign stated as a property of K = 5**, not of the store.
2. ⛔ **`w12-advisor-report.md:146` needs the §3 correction** — the ladder's opponent was
   ~3.18 M params, not 1,029,508, and "roughly halved" describes the bridge → K\* step.
3. ⚠ **Does the bridge column belong in the paper?** It is what makes "K\* flips it"
   attributable. This hub recommends one appendix row: *ladder K=5 → bridge K=5 → K\**.
4. ⚠ **K = 5 on TEST at seeds 1 and 2, and the bridge at seeds 1 and 2** — worth a further
   replication, or is the single-seed attribution enough?
