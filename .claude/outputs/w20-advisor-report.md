# WAVE 20 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-24. One hub, machine `forgis-14` (local CPU). Target `shapes-matched`: 3 of 3 scored runs SPENT, target exhausted.**
Record: `/c/chlu/outputs/w20-shapes-data-matched/` (1.1 GB, 1,789 files); 1,482 mirrored to `.claude/outputs/w20-shapes-data-matched/`. Code `w20/shapes-data-matched = 7627afd`, bundle verified, **88 files / 41,683 insertions / 0 deletions**.
⭐ Every number below is quoted from the spoke's own artefacts; the Hub verified the load-bearing ones against the run JSONs.
⛔ **No pre-registration governs this report's framing** — the Head banned pre-declarations on 2026-09-24. **Everything measured is printed**, including what was previously withheld.

---

## 0. ⭐⭐ THE ANSWER — the headline number is wrong by about 6×, and the reason is not the one the referee gave

**The data-matched margin is +7.03 novel tuples per corner** (seeds **+7.49 / +8.60 / +5.02**), and **+6.54** against the rival at its best over *every* fit that ran, 10× and 30× included.
**The draft's current margin, recomputed from the same score files, is +39.7 … +49.1.**

⭐ **It shrinks ~6×. It does not collapse.** On every seed the bootstrap CI clears +1.0 (lowest lower bound **+3.73**), and the store beats the strongest data-matched rival on **67–84 of 109 corners** per seed.

⛔⛔ **But M1's stated cause is WRONG, and this is the finding the Advisor most needs.** The referee said the rivals had an 8× *data* disadvantage. They did — and it barely mattered. **The rivals scored 5.5 because they were trained for 1,280 steps, untuned.** Give an MLP the rows-3–4 selected schedule (20,480 steps):

| MLP fit | rows | steps | score |
|---|---|---|---|
| the draft's comparator | 16,384 | **1,280** | **5.5** |
| step-matched | **16,384** | 20,480 | **38.6** |
| step-matched | 32,768 | 20,480 | 39.1 |
| step-matched | 131,072 | 20,480 | 39.2 |

**At matched steps, 8× more data buys the MLP ≈ 0.5 novel tuples per corner.** The store is just as data-insensitive (K\*=32: 47.28 at 16,384 rows → 48.01 at 131,072).
⇒ ⭐⭐ **The draft's +39…+49 was carried by an UNDER-TRAINED comparator. Data matching is not what closes it; the training budget is.** ⛔ Fixing the data match alone would not have fixed the paper.

---

## 1. Convergence — the defect that produced the old number, now instrumented

**P1 per seed** (DECLARED = selected refit; TENX = 10× optimiser steps):

| family | DECLARED | TENX | TENX − DECLARED | P1 | 30× diagnostic |
|---|---|---|---|---|---|
| mlp | 36.71 / 37.57 / 43.17 | 40.01 / 38.81 / 44.10 | +3.30 / +1.24 / +0.93 | **FAIL / FAIL / PASS** → TENX carries s0, s1 | 41.48 / 36.54 / 43.50 — ⚠ **s0 still +1.47 over TENX: NOT CONVERGED at 10×** |
| cont_mlp | 17.92 / 16.54 / 26.84 | 28.53 / 32.94 / 42.60 | **+10.6 / +16.4 / +15.8** | **FAIL ×3** → TENX carries | 27.28 / 34.01 / 38.71 (s1 flagged) |
| lstm | 33.97 / 32.36 / 40.70 | 33.69 / 32.54 / 39.85 | −0.28 / +0.18 / −0.84 | PASS ×3 | — |
| transformer | 25.56 / 27.79 / 38.65 | 25.30 / 27.24 / 37.84 | −0.26 / −0.55 / −0.81 | PASS ×3 | — |

⭐ **No margin in this report is carried by a rival that failed its budget** — where P1 failed, the TENX fit carries the row, and the headline is also printed against the best fit of all.
⚠ **Every family selected lr = 3e-3 and 40 epochs — the EDGE of both search grids.** The grids were **not** widened after the fact; the TENX column is the instrument for that. ⛔ The honest reading is that the rival search space may still be truncated.
⚠ **The residual:** at seed 0 the MLP still gains +1.47 from 10× to 30× (36.71 → 40.01 → 41.48). Increments shrink but convergence is **not proven**. **+6.02 (best-over-all-fits) is the honest seed-0 number.**

## 2. ⭐⭐ DELETION UNDER THE HEAD'S FOUR-MEASUREMENT DEFINITION — the mechanism is exact, the outcome is not

**Definition (AMENDMENT-2):** forget *c* = downdate the store **and** downdate the address book's statistics and **refit the book from them**, then punch the hole. ⛔ The "*c* still composes after forgetting *c*" row is **RETIRED** — not computed, not printed.

| scope (rows) | (a) store == refit-without | (b1) book statistic | (b2) rendered features | (c) DISPLAY | (d) survivors before → after | (d) downdate − refit-without |
|---|---|---|---|---|---|---|
| value red (11,044) | 5.96e-08 / 4.82e-08 / 2.01e-08 | **0** | 1.2–1.9e-09 | ⭐ red requests **refused 1.000**, false refusal **0.000** | 50.92→50.87 / 50.53→50.85 / 52.61→52.00 | **+0.000** ×3 |
| **red ∧ sphere (5,511)** | 5.38e-07 / 1.07e-08 / 6.25e-08 | **0** | 2.0e-09–7.8e-08 | ⚠ refusal **UNDEFINED** (no rows-3–4 corner request is red∧sphere); false refusal 0.000 | ⛔ **47.50→41.37 / 47.40→40.97 / 49.12→42.82** | **+0.000** ×3 |
| 1 % items (1,311) | 2.34e-07 / 1.49e-08 / 3.16e-08 | **0** | 5.8–9.7e-10 | N/A — no hole exists | 47.50→47.48 / 47.40→47.29 / 49.12→49.11 | **+0.000** ×3 |

⭐ **(a) and (b) are EXACT.** The book refit from the downdated statistic equals the refit-without book to 1e-8 in rendered features, and composition is **identical (+0.000 on every corner, 9/9 legs)**. ⭐ **The mechanism works.**

⛔⛔ **(d) FAILS for `red ∧ sphere` on 3 of 3 seeds** (Δ **−6.13 / −6.43 / −6.30** against a |Δ| ≤ 1.0 bar). Seed-0 breakdown: the **20 corners requesting red with OTHER shapes drop −32.1 each**; the 36 non-red sphere corners −1.47; the other 53 **+0.51**.
⭐⭐ **And the control kills the obvious rebuttal:** deleting a **random 1 %** moves survivors by −0.02 / −0.11 / −0.01. **So this is the deleted CONTENT, not refit instability.**
⇒ **Under the Head's definition, deleting a conjunction removes the store's ability to compose one of its values with other values.** ⭐ **The w15 "composition unchanged" row held only because the book was CARRIED.** This is referee **M3's objection, measured and replicated.** (Value scope passes: −0.06 / +0.33 / −0.61.)

### ⚠⚠ A cost that changes what "data-free deletion" may claim
**The ALS address book has NO compact exact statistic.** It sees data only through per-tuple aggregates, and every Shapes3D tuple is one row — so its exact statistic is the per-row first moment of its 117,964 fit rows: **0.97 GB, i.e. the fit data itself.**
⭐ *"Adaptation reads 0 pool rows"* is **true**. ⛔ **It is NOT a size claim**, and the paper must not let it read as one. The book refit costs **2,367–5,913 s** — about the same as refitting the book from scratch.

### Against the rivals' retrain-without (same rows, same pool)
Store-after − rival-retrain-without on survivor corners: **value** +7.47…+32.61, **item** +6.95…+28.84 — all clear of zero.
⛔ **`red ∧ sphere` is NOT significant on 3 of 12 legs:** vs mlp s1 **+2.49 [−0.94, 6.02]**, s2 **+2.60 [−0.50, 5.67]**; vs lstm s2 **+2.48 [−0.87, 5.82]**.
⛔⛔ **The asymmetry that hurts:** under `red ∧ sphere` **the rivals lose < 1 novel tuple per corner; the store loses 6.** Their crude retrain-from-scratch is *cleaner* on collateral damage than our exact deletion.
⭐ **Cost, which is the store's real advantage:** rival retrain reads **120,028–129,761 rows** and takes **548–2,065 s**; the store downdate reads **0 pool rows** in seconds.
⚠ Rival retrains ran at the DECLARED schedule, so the **mlp (s0, s1) and cont_mlp P1 flags travel with these rows.**

## 3. The ladder (free replication) — the shrink is real but much smaller

Every rung's store **and** its data-matched rivals fitted on that rung's own 131,072-row pool; mlp and cont_mlp carried at TENX on every leg.

| rung | store K\*=32 | strongest data-matched rival | **K\* − strongest [mean]** | draft-style (vs spoke-A `@ref`) |
|---|---|---|---|---|
| n=2 | 32.64 | mlp tenx 27.51 | **+5.13** | +9.4 … +12.4 |
| n=3 | 61.61 | mlp tenx 50.43 | **+11.18** | +9.9 … +21.2 |
| n=4 | 66.09 | mlp tenx 53.54 | **+12.55** | +15.2 … +24.3 |
| n=5 | 66.58 | mlp tenx 55.89 | **+10.69** | +15.3 … +25.3 |
| n=6 | 66.60 | mlp tenx 56.30 | **+10.30** | +15.7 … +24.7 |

⭐ **CI95 above 0 on every seed and rung** (lowest lower bound +2.46); smallest seed-level margin +3.85. ⛔ **No rung collapses.**
⭐ The store refit on 16,384 or 32,768 rung rows matches the 131,072-row store within ~0.6 — **the ladder repeats the rows-3–4 picture: the store barely uses extra data.**
⭐ Rebuilt K\*=32 ladder stores reproduce w12b's per-corner rows on **15/15** legs.
⛔⛔ **A claimability defect the Advisor must rule on:** the **EXISTING ladder K = 5 arm** (w12's `DowndatableStore`, reused "as already fitted") carries only **2 of the 4 controls**, and its strong-form `random_atoms` reads **1.24 at n=4 seed 2** and **2.03 at n=5 seed 2**, over the 1.0 bar. ⇒ **Its n=4 and n=5 rows are NOT claimable under P2.** Every *w20* store collapses on all four controls with the derangement printed.
⚠ `lstm` also fails a ladder spot check (n=2 s0: 20.55 → 21.85 at 10×, +1.30) — printed, not netted.

## 4. Defects found and fixed DURING the run — all self-caught

1. A dev smoke ran on **generation-1** features; caught by an assertion, output tree unused, every script now asserts generation 2.
2. The val `seen_mask` marked every val tuple seen (val corners live in `train_rows`) ⇒ **first val pass VOID**, moved aside, fixed.
3. ⭐⭐ **Rung dead-channel masks are PER SEED**; the loader read seed 0's for every seed. **Seed-1/2 rung stores and their scores were VOIDED and rebuilt.** Rows 3–4 unaffected. ⛔ Reported as a void, not patched over.
4. A ladder rebuild threshold-width check; rebuilt rows reproduce w12b per corner.
5. One orphan slow-path scorer ran beside the fast path (verified identical per corner at seed 0; stopped).

## 5. What did NOT run — DECLARED with reasons (⭐ printed per the Head's omit-nothing order)

1. **LACE/PoE @131,072 on ladder rungs n=2…6 × 3 seeds (15 legs).** Measured cost: a 36-trial selection of 5,955–11,034 s **plus a 50-epoch classifier that took ~7.5 h on the val leg**, ~3.5 GB each, dispatch-bound at ~0.3 core. Two rung legs started and were stopped before producing a number. ⭐ Rows 3–4 at 131k **did** run, 3 seeds: **LACE 0.0183 / 0.0459 / 0.0734**, **PoE ≈ 0**, against the store's 64.92 on the same corners — they render essentially nothing valid at scale.
2. **VAL `lstm`** — fit stopped for memory; lstm is measured on test with its own TENX.
3. **TENX on the ladder** — only as the pre-declared spot check (rungs 2 and 6, seed 0, all four families).
4. **30× diagnostics** — only for mlp and cont_mlp, the two families that failed P1.
5. **cont_mlp / mlp retrain-without at TENX or 30×** (9 fits, ~1–5 h each) — deletion rivals run at DECLARED; the P1 flags travel.
6. ⚠⚠ **`learned_1x` (536k and ~3.2M params) is NOT PRINTED** — matched to the store's four-part MEMORY ledger rather than its read path, and the standing rule is that rivals match on **read-path** parameters. ⭐ **Kept in artefacts.** ⛔ It is the highest-capacity rival anyone fitted; under the omit-nothing order the Advisor should decide whether it belongs in an appendix rather than being absent.
7. **`landscape_removed` and `landscape_shuffled_across_wells` on the existing ladder K=5 arm** — never run on that arm (§3).
8. **Spoke A's literal 32,256-row Montero pool as a store fit pool** — it contains both held-out values (3,157 rows at object_hue 0.5; 7,972 at shape 0.0), so the store used the 32,768-row nested prefix of its own pool instead.

## 6. What would have proved this wrong, and what happened

| the test | outcome |
|---|---|
| the margin **collapsing** (≤ +1.0) | ⛔ did not happen — **+7.03** (+6.54 vs best of all fits) |
| the margin **holding** (≥ +39) | ⛔ did not happen — ⭐ **the draft's +39…+49 is not a data-matched number** |
| rivals still improving at TENX | ⚠ **partly** — cont_mlp (all seeds) and mlp (s0, s1) needed TENX; mlp s0 still creeping at 30× |
| the T7 store not reproducing | ⭐ it reproduced per corner on 3/3 seeds |
| the across-wells derangement not collapsing | ⭐ it collapsed (0.000) everywhere |

## 7. What the Advisor is asked to rule on

1. ⭐⭐ **The abstract's headline must change from +39…+49 to ≈ +7** (rows 3–4, K\*=32), with the convergence table beside it. This is the single largest claim change in the programme. Confirm the replacement number and where the P1 table sits.
2. ⭐⭐ **M1's mechanism was misdiagnosed.** The unfairness was **training schedule**, not data. Does the paper say so plainly — including that the published comparator was trained for 1,280 steps — or only report the corrected number?
3. ⭐⭐ **M3 is now measured, replicated, and controlled.** Deleting `red ∧ sphere` costs the store **−6.1 to −6.4** on survivors while rivals lose **< 1**, and a random-1 % control moves nothing. How is "exact, scoped deletion" restated given that the exactness (a, b) holds but the outcome (d) does not?
4. ⚠ **"Data-free deletion" needs rewording.** Adaptation reads **0 pool rows** — true — but the book's exact statistic is **0.97 GB**, the size of the fit data, and the refit costs as much as refitting from scratch. What may be claimed?
5. ⚠ **The existing ladder K = 5 arm's n=4 and n=5 rows are not claimable** (strong-form `random_atoms` at 1.24 and 2.03, over the 1.0 bar, and two controls never run). Drop them, re-run them, or print them as not-claimable?
6. ⚠ **`learned_1x` is absent from the tables on a matching-rule ground.** Under the Head's omit-nothing order, should the highest-capacity rival appear in an appendix with its rule stated?
