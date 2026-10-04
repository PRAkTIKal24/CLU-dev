# PREREG — `hub/deletion-2` · the deletion cell · seed 0
**Written 2026-09-17T11:20Z, BEFORE any deletion build existed.** Relaunch START_UTC `11:13:33Z`.
Machine: FORGIS_14, Windows 11 AMD64, 4 threads. Branch `hub/deletion-2` @ `945a3e9` (base/iter2).
⛔ Nothing in this file may be edited after the first deletion build is scored. Corrections go in
`LOG.md` as dated amendments, and any amendment made *after* seeing a number is marked POST-HOC and the
affected claim is demoted.

---

## 0. What is being claimed, in one sentence
That **exact structural deletion** of a factor value from the CLU's generative store — `delete(v)` :=
drop `v`'s rows from the live set and **recanonicalise** — costs **less collateral damage to the rest of
the space** than the same deletion costs `knn_proto` at matched bytes, and that the deleted value is
genuinely unrecoverable from requests that name it.

⛔ **What is explicitly NOT claimed as a headline:** that the rebuild is byte-exact. It is, and it is
proven (§4), but it is **true by construction once determinism holds** and this programme has ruled that
a method-section fact, never a result. ⛔ "The rival has no delete operator" is likewise worth nothing
and is not claimed — `knn_proto` *does* get an exact delete here, by refit.

---

## 1. The mechanism (ported, ⛔ not invented)
*Retention You Can Predict, Scope and Delete*, §2.3 / App. F / App. G.8. Deletion is exact when the store
layout is a **canonical function of the live item set and the fixed geometry**. Here the whole deployed
object is rebuilt from the sorted live row vector:

```
build(live) :  X      = frozen_cache[live]                      (⛔ never re-encoded)
               corr   = channel_correlation(X[:16384])
               labels = grouping_labels(corr, K=5, "auto"->ward) ; wc = wells
               store  = train_store(init_store(X, wc, M=64), X)  1200 steps, batch 128
               codes  = address_codes(store, X)
               book   = fit_address_book(codes, factor_tuples(live), cardinalities)
delete(S,P) := build( live(S) \ rows(P) )
```
**Cost: O(n) rebuild per delete**, accepted openly; measured wall-clock is in the cost line.
⚠ Caveat carried over **verbatim**: *store-level guarantee only; the frozen encoder is a separate
channel; ⛔ no certified (ε, δ) unlearning is claimed.*

## 2. The read, declared
**Request-addressed**, because the charter's question is about *requests*, and because it is the only
frame in which `knn_proto` exists:
```
request (a factor tuple) -> address_requests(book) -> argmax mode per well (K·log2 M = 30 bits)
                         -> read_from_codes(store) -> FROZEN decoder -> FROZEN external probe
```
⛔ Nothing on that path touches the feature cache. A feature-addressed read is reported **beside** it as a
companion (it separates "the store lost `v`" from "the addressing lost `v`") and ⛔ never instead of it.
Laundering floor on every set: `store_removed` = `delete_store(mode="mean")` at identical bytes and
identical codes. ⛔ That is the shipped laundering control, not this hub's delete operator.

## 3. The six deletions — fixed now, ⛔ not chosen after seeing anything
One value per factor, **the median grid index**, so that "could `v` be re-composed from its neighbours"
is not handicapped by `v` sitting at an edge of the grid with a one-sided neighbourhood:

| # | factor | card | deleted value index | rows removed from the 32,768-row pool (expected) |
|---|---|---|---|---|
| 1 | `floor_hue`   | 10 | 5 | ~3,277 |
| 2 | `wall_hue`    | 10 | 5 | ~3,277 |
| 3 | `object_hue`  | 10 | 5 | ~3,277 |
| 4 | `scale`       |  8 | 4 | ~4,096 |
| 5 | `shape`       |  4 | 2 | ~8,192 |
| 6 | `orientation` | 15 | 7 | ~2,185 |

⚠ The six differ in size by 3.7x, which would confound collateral with *how much data was removed*. So
each is paired with a **PLACEBO**: a rebuild dropping the **same number** of rows chosen at random
(fixed seed), which is not aligned to any factor value. The placebo is the **churn floor** — how much a
rebuild moves the numbers for no semantic reason at all.

## 4. Condition 3 — provenance (the hash equality). Bar: exact.
`store_digest` = sha256( well_channels ‖ param_hash(modes) ‖ sha256(log_prior ‖ log_cond) ).
⚠ Strictly stronger than the `param_hash` that already existed, which covers **array leaves only** and
would have missed both `well_channels` (an `eqx` static field) and the **address book** — and the book is
fitted on the live rows' labels, i.e. it is exactly where a deleted value could survive.

- **4a. DETERMINISM (blocking gate, nothing downstream is meaningful without it).** Two independent
  processes, same live set, same batching: `digest(det_B) == digest(det_C)`, **exact**.
  ⛔ If they differ I stop, report both digests and the batching, and do **not** loosen
  "byte-identical" to "close" (falsifier §5.5).
- **4b. PROVENANCE.** `digest(delete(S, v)) == digest(build(pool \ rows(v)))`, **exact**.
- **4c. PATH-INDEPENDENCE** (the part of provenance that is *not* free): deleting `floor_hue=5` then
  `wall_hue=5` must equal deleting them jointly, **exact**.
- **4d.** ⚠ `delete_atom(j)` — zeroing a mode's column — is **NOT canonical**; the other modes were
  fitted with `j` present. Reported as a **scoped** operation with its measured residue, ⛔ never as exact.

## 5. Condition 2 — the PRE-DELETION control (falsifier 2). Bar stated, and it may fail.
For each deletion, a **targeted request set**: 1,024 requests naming `v` on the deleted factor, drawn
from train rows that the store pool never held, fixed seed, **identical for every build**.
**Bar: pre-deletion per-factor accuracy on the targeted set ≥ 0.50 for the deleted factor.**
⛔ If a factor fails this bar, "unrecoverable after deletion" is **vacuous** for that factor and it is
reported **NOT MEASURABLE**, ⛔ never counted as a deletion success.
⚠ **I expect `orientation` and possibly `scale` to FAIL this bar, and I am saying so before running.**
The inherited §E.4 fact is that the codebook already costs **90.5 % of `scale` and 100 % of
`orientation`**. A value that the store cannot render *before* the deletion proves nothing by being
unrenderable after.

## 6. Condition 1 — unrecoverable. Bars.
On the targeted set, after deletion, for the deleted factor `f` with cardinality `V`:
- **6a.** per-factor accuracy ≤ `1/V + 0.05` (chance + 5 points), **and**
- **6b.** fraction of *rendered* tuples whose `f` reads back as `v` ≤ `1/V + 0.05`.
⭐ 6b is the charter's *"no rendered tuple contains it"* and is the stricter of the two: 6a can be passed
by a store that renders `v` for the *wrong* requests.
⭐⭐ **This is the one place the result cannot be true by construction**, and I expect it to be the
hardest: a compositional store may re-compose `v` from the parts it still holds. ⛔ **If it re-composes,
that is the finding and I report it** — *"a compositional memory cannot forget a value whose parts it
still holds"* is a real result and an honest negative.

## 7. ⭐⭐ THE HEADLINE — COLLATERAL. The bar, as a number, before the run.
For each deletion `v`, on the requests that do **not** name `v` (the `dev` set, 4,096 requests, minus any
naming `v`), per-factor accuracy aggregated over the 6 factors:
```
C_arm(v) = acc_arm_after(v) - acc_arm_before          (<= 0 is damage)
Delta(v) = C_clu(v) - C_knn_proto(v)                  (> 0 = the CLU is damaged LESS)
```
paired **per request** (the same 4,096 requests, both arms, both before and after).
**Primary object: `Delta` pooled over the 6 deletions**, with a paired 95 % CI (BCa bootstrap, 10,000
resamples, resampling *requests*; the 6 deletions enter as a fixed effect, not as the resampling unit).

**Verdict rule — R1, taken verbatim, with m = 0.020, fixed before the run:**
- **win** := `Delta > 0` and the 95 % CI excludes 0
- **tie** := the CI includes 0, **or** `|Delta| <= 0.020`
- **loss** := `Delta < -0.020` and the CI excludes 0

⭐ **A tie is banked and reported**, then I keep pushing with the remaining time.
**Bytes (R3, same-device, both arms rebuilt here from Mac-encoded features):**
CLU store = **524,288 B** (64 modes x 2048 x 4). `knn_proto` table = **475,136 B** ((57+1) x 2048 x 4).
⚠ `knn_proto` is the **smaller** object, so it is not being flattered by the byte match; both numbers are
reported and neither arm's addressing head is hidden.

### 7b. Condition "preserved", bar
Per-factor accuracy on non-`v` requests must stay **within the placebo band** — the same statistic under
a size-matched random deletion. ⚠ The charter says *"within pre-deletion seed spread"*; I have **one
seed** (§1c forbids spending seeds 1/2 here — their caches are `scope="used"`). ⭐ The placebo band is
declared as the substitute **and is the stricter, more honest control**, because it isolates semantic
deletion from rebuild churn, which a seed spread does not.

## 8. Rivals
- `knn_proto` — **run**, identical live set (read off the CLU build's own artefact and sha256-checked
  against it), identical predicate, same frozen decoder + probe.
- atom-mean — **run** as `store_removed` (`delete_store(mode="mean")`), the laundering floor.
- `lace` / `ebm_compose` — ⛔ **NOT RUN**. They have no exact delete operator; retrain-without-`v` is 6
  retrains x 2 arms and does not fit beside 16 rebuilds in 12 h. Reported as **NOT RUN** with this
  reason, as a method-section line, ⛔ never as zeros.

## 9. Scored-run budget
**At most ONE scored run, seed 0 only** (§1c). Everything in §§4–7 is first measured on **DEV**
(`dev` = 4,096 non-held-out train rows the pool never held; `a` = the 1,024-row eval_A set). The 72
**held-out** `B` rows are touched **once**, only if the DEV `Delta` clears the §7 bar with §4 and §5
passing, and the result is reported as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*.
⛔ If DEV does not clear, **B is not spent** and the hub reports NOT SOLVED with the DEV numbers.

## 10. What would make this hub's answer worthless (restated as a checklist I will answer)
- [ ] a win that is true by construction → say so in those words
- [ ] no pre-deletion control → §5 is mandatory and may fail
- [ ] rivals not given the same deletion → §8, sha256-checked
- [ ] "absent from a direct read" reported as "unrecoverable from every composition" → §6b, and the
      request-addressed read is the declared one

---

## AMENDMENT A1 — 2026-09-17T11:24Z
⭐ **Made before any deletion build had finished; `runs/` at this moment contains only `det_A` and
`det_B`, both undeleted full-pool controls.** No deletion number exists, so this cannot be a choice made
to suit one.

**The problem it closes.** Pre-deletion on `dev`, `clu` = 0.5999 and `knn_proto` = 0.7257, with the
laundering floor at 0.1230. The CLU therefore has **less room to fall** than the rival. A §7 `Delta > 0`
("the CLU was damaged less") could then be a **floor effect** rather than a property of the deletion.
The sibling thread made the same point about held-out B independently.

**Three views of the same paired data, all reported, ⛔ none of them selected after the fact:**
1. **PRIMARY (§7, unchanged): `Delta` on raw accuracy points**, R1 with m = 0.020. This alone decides the
   win/tie/loss verdict.
2. **RELATIVE: `C_arm / acc_arm_before`** — each arm's damage as a fraction of what it had to lose.
3. **CHURN-CORRECTED: `[C_clu(del) − C_clu(plc)] − [C_knn(del) − C_knn(plc)]`** at matched deletion
   size — the semantics of the deletion with pure rebuild re-randomisation subtracted.

**The rule, fixed now:**
- ⭐ The **primary governs the R1 verdict.** I do not get to switch estimand after seeing the numbers.
- ⛔ **If the primary and the relative view disagree in SIGN, the headline is reported as the primary
  requires AND immediately beside it, in the same sentence, the statement that the two disagree and that
  the result is therefore NOT robust to the floor effect.** A `win` that survives only in raw points is
  reported as *"a win in raw points that does not survive normalisation by each arm's own headroom"* —
  ⛔ never as an unqualified win.
- ⭐ **If a deletion's collateral sits inside the churn band** (§7b, the three n=8234 placebo
  replicates), it has **not been shown to cost anything beyond rebuilding**, whatever the sign of Delta,
  and is reported that way.

---

## AMENDMENT A2 — 2026-09-17T11:25Z
⭐ **Window check, performed and recorded before writing this:** `runs/` contains `det_A` and `det_B`
only — both **undeleted** full-pool controls — and `ls runs/del_*/eval.json` returns **nothing**. No
deletion has been scored. Per §0 this is therefore a pre-hoc amendment, ⛔ not POST-HOC.
Prompted by `REVIEW-01-prereg-and-analysis-clu-dev-1e.md` (the sibling thread's independent review).
I adopt the parts that are arguments; ⛔ I do not adopt any number I have not sourced myself.

### A2.1 — §7's pooled statistic was unequally weighted. PRIMARY replaced.
⚠ `nanmean` over deletions per request averages each request over however many deletions it *survives*.
Because a request is excluded from a deletion that names it, and the six deletions remove different
fractions, **the six do not carry equal weight**: realised weights 0.135–0.182 against 0.1667, i.e.
`shape=2` — **the largest deletion and the one most likely to show collateral — enters at 0.81x** while
`orientation` enters at 1.09x (max/min = 1.35).
⭐ **PRIMARY is now the unweighted mean of the six per-deletion means** (`collateral_pooled_equal_weight`):
average **within** each deletion first, then **across** the six, so each carries exactly 1/6. The BCa
bootstrap resamples **requests**, shared across deletions so the pairing survives. The old
request-averaged number is still computed and reported; ⛔ if the two disagree materially the
request-averaged one is the one to drop, and I say so here rather than after seeing them.

### A2.2 — the churn-corrected estimand is promoted to CO-PRIMARY.
⚠ The §7 primary has **two biases of opposite sign** and both are measurable rather than arguable:
- **the level gap favours the CLU** — it starts at 0.5999 vs `knn_proto`'s 0.7257 on `dev`, so if damage
  is roughly proportional to level it loses fewer *raw points* from the gap alone;
- **rebuild churn favours `knn_proto`** — the CLU's "after" is a fresh 1200-step optimisation on a
  re-derived channel partition, while `knn_proto` has **no optimiser at all** (its table is a closed-form
  mean). The uncorrected primary therefore charges the CLU for churn as if it were deletion damage.
⭐ `delta_churn_corrected` = `[C_clu(del) − C_clu(plc)] − [C_knn(del) − C_knn(plc)]` at matched size is
**co-primary**, because it is the one that answers the actual scientific question — *does deleting a
MEANING cost more than deleting the same number of random rows?* The uncorrected primary answers the
different, also legitimate question *what does it cost in practice to delete from a CLU*. Both reported,
both labelled with which question they answer.

### A2.3 — RETENTION added (the estimand in which the level gap cancels).
Of the `(request, factor)` cells an arm got **right before** the deletion, what fraction is still right
after? Each arm is scored only on **its own competence set**, so the 13-point level gap cancels *by
construction* and the statistic is bounded in [0, 1]. This is the unlearning literature's standard
"retain accuracy" and it is immune to the *"the CLU had less to lose"* objection in a way that
`delta_relative` is not. Reported per deletion and pooled.

### A2.4 — ⭐ §5's pre-deletion bar REPLACED. The intent stands; the number goes.
⛔ **The 0.50 was arbitrary and, worse, backwards.** Chance is `1/V`, so a flat 0.50 is 2.0x chance for
`shape`, 4.0x for `scale`, 5.0x for the hues and **7.5x for `orientation`** — hardest for exactly the
factors the codebook already damages most, which is the wrong direction for a *control*, and it treats
`shape` far more leniently than `orientation` for no stated reason.
**Replacement — a factor passes the pre-deletion control iff BOTH:**
1. **above chance** — `acc_pre` on the targeted set beats `1/V` by a one-sided binomial test at
   α = 0.05 (sharp at n = 1024); **and**
2. ⭐⭐ **above the arm's own null** — `acc_pre` exceeds the **laundering floor**
   (`delete_store(mode="mean")`) on the **same rows and the same codes**.
Clause 2 carries the argument and is better motivated than any absolute threshold, because "the store
supplied this factor" *means* "better than the same read with the store's contents removed". It is not
hypothetical: on held-out B the live CLU scores **0.0417** on `scale` and **0.0000** on `orientation`
while the fully ablated store scores **0.3333** on both. ⭐ When the ablated store **beats** the live one,
the store demonstrably was not supplying that factor, and *"recoverable before"* is false at **any**
absolute accuracy. ⛔ The intent of §5 is unchanged and is the part I care about: the control is
mandatory, it was declared in advance that it may fail, and a failure is reported **NOT MEASURABLE**,
never as a deletion success.
⚠ **Citation correction.** §5 justified its expected failure with *"the codebook costs 90.5 % of `scale`
and 100 % of `orientation`"*. That figure's arithmetic is **not in the bundle**; two independent
re-derivations by the sibling thread give ~75 %/~82 %. ⛔ I have not re-derived it myself, so I quote
**neither** number as mine: the expectation now rests on the qualitative inherited fact plus the
directly-measured B row above, which is my own instrument.

### A2.5 — §6b is measured over ALL requests, not only the targeted ones.
A store could avoid emitting `v` when *asked* for `v` while still emitting it elsewhere, and would pass a
targeted-set-only §6b while failing the charter's *"no rendered tuple contains it"*. §6b is
`(preds[:, f] == v).mean()` over the **whole `dev` set**. (This was already how `analyse.py` computed it;
the PREREG text now matches the code.)

### A2.6 — when a factor goes NOT MEASURABLE, report its CONFUSION ROW.
⛔ A bare "NOT MEASURABLE" is a hole in the table. What the store renders *instead of* `v`, before and
after, is free from `preds.npz` and is an observation worth having.

### A2.7 — two document-vs-run corrections (no scientific content)
- §4a says the gate is `det_B` vs `det_C`. **The gate actually run is `det_A` vs `det_B`** — two
  full-pool builds from an identical live set made by two *independent processes* — **arbitrated by
  `det_C`** if they disagree. The executed version is the stronger one; the document now matches it.
- §3's row counts were nominal. **Realised:** 3269 / 3224 / 3207 / 4095 / 8234 / 2142, conjunction 852.
  The placebos are sized to the realised counts.

---

## AMENDMENT A3 — 2026-09-17T11:30Z
⭐ **Window check, recorded before writing:** `ls runs/del_*/eval.json` returns **nothing**; `runs/` holds
`det_A`, `det_B`, `det_C` — all **undeleted** full-pool controls. ⛔ Pre-hoc, not POST-HOC.
Prompted by `clu-dev-1e`'s third review. ⚠ One of its claims is **corrected** below rather than adopted.

### A3.1 ⛔⛔ §6a AS WRITTEN CAN BE SATISFIED BY A ONE-STEP SHIFT. This is the serious one.
Measured on `det_B`, **pre-deletion**, the CLU's render distance from the requested value on each
targeted set (share of 1,024 requests):
| factor | \|d\|=0 | \|d\|=1 | \|d\|=2 | \|d\|≥3 | within 1 |
|---|---|---|---|---|---|
| floor_hue | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| wall_hue | 0.995 | 0.005 | 0.000 | 0.000 | 1.000 |
| **object_hue** | **0.511** | **0.489** | 0.000 | 0.000 | **1.000** |
| scale | 0.400 | 0.483 | 0.109 | 0.007 | 0.884 |
| shape | 0.826 | 0.167 | 0.007 | 0.000 | 0.993 |
| orientation | 0.500 | 0.368 | 0.004 | 0.128 | 0.868 |

⛔ **`object_hue` is the sharp case.** Pre-deletion the store renders hue 5 on 51 % of requests naming
hue 5 and hue 4-or-6 on the other 49 %, with **literally zero mass beyond one step**. So after deleting
hue 5 the overwhelmingly likely outcome is that those requests render **hue 4 or 6** — accuracy near 0,
which **PASSES §6a's bar of `1/V + 0.05 = 0.15`** while the store is still emitting a colour **one grid
step away**. ⭐ **That is not forgetting in any sense the charter means, and an exact-match probe cannot
tell the two apart.**

**Replacement, and it costs nothing because the pre-deletion baseline above is already measured.**
§6a's threshold is **retained but demoted to a companion**; the reported statistic is the
**post-deletion distance profile against the pre-deletion one**, with **three outcomes pre-registered
now**:
1. **STRUCTURAL DELETION** — mass at `|d|=0` collapses toward 0 and moves to `|d|=1`, the rest of the
   distribution roughly unchanged.
2. **RE-COMPOSED FROM ITS PARTS** — mass **stays** at `|d|=0`. ⭐ This is the gs charter's central
   tension and an honest negative if it happens.
3. **DAMAGE, NOT DELETION** — mass spreads to `|d|≥2`.
⭐ These three are distinguishable; *"accuracy ≤ 0.15"* is not.

### A3.2 Factor geometry — VERIFIED, and one reviewer claim corrected
Checked against `factor_index.npz`, ⛔ not assumed:
- `floor_hue` / `wall_hue` / `object_hue`: 10 values over 0.0–0.9 → **CYCLIC** (hue wraps), distance is
  `min(|d|, 10−|d|)`.
- `scale`: 8 values over 0.75–1.25 → linear ordinal.
- `shape`: 4 values `{0,1,2,3}` → **CATEGORICAL**. ⛔ Distance is meaningless; read exact-match only.
- `orientation`: 15 values over **−30°…+30°** → linear ordinal. ⚠ **The reviewer suggested its 12.8 %
  mass at `|d|≥3` was wrap-around on a cyclic factor. It is not** — the grid spans **60°, not 360°**, so
  orientation does **not** wrap. That mass is **genuine error** and ⛔ must not be explained away.

### A3.3 §5's margin is reported against the MARGINAL null, not against `1/V`
⚠ The right null for a single-value targeted set is the store's **own marginal rate of rendering `v`** on
a diverse set. If the store simply over-renders `v`, `acc_pre` is **bias, not steering**:
| factor | `acc_pre` | ×chance | marginal | base rate | **render bias** | **×marginal** |
|---|---|---|---|---|---|---|
| floor_hue | 1.0000 | 10.00 | 0.103 | 0.103 | 1.00 | 9.68 |
| wall_hue | 0.9951 | 9.95 | 0.099 | 0.099 | 0.99 | 10.09 |
| object_hue | 0.5107 | 5.11 | 0.074 | 0.112 | 0.66 | 6.93 |
| scale | 0.4004 | 3.20 | 0.260 | 0.122 | **2.14** | **1.54** |
| shape | 0.8262 | 3.30 | 0.363 | 0.246 | **1.47** | **2.28** |
| orientation | 0.5000 | 7.50 | 0.203 | 0.075 | **2.69** | **2.46** |
⭐ **All six still clear this harder null, so §5 passes under a stricter test than was registered.** ⛔ But
the margins are much smaller than clause 1 suggests: `orientation`'s "7.5× chance" is really **2.46× its
own render bias**, and `scale`'s "3.2×" is really **1.54×**. Margins are reported against the marginal.
⚠ **Disclosed confound:** the **median-index** deletion rule (§3) chose the middle value to avoid
one-sided neighbourhoods at grid edges — and the middle value is exactly the one a shrinking decoder
**over-renders** (`scale` 2.14×, `orientation` 2.69×). The rule is kept, and the over-rendering is
disclosed here rather than discovered later. `object_hue` is the exception at 0.66 — it *under*-renders
hue 5 — so the confound is ⛔ not uniform.

### A3.4 The stale baseline needs no rebuild
`det_A`'s eval used `n_dev=1024`/`n_target=256`; the cell uses 4096/1024, so its targeted row sets moved
too. ⛔ No re-run is needed: **`det_A` and `det_B` are byte-identical stores**, and `BASE` is already
`det_B`, which has the correct set sizes. ⭐ The bit-identity is exactly what licenses the substitution.
⭐ Third independent confirmation of that bit-identity, through the **whole eval path** (store reload →
address book → request posterior → decoder → probe) rather than through the digest: on the fixed 1,024-row
`eval_a` set, `det_A`'s and `det_B`'s CLU predictions agree at **1.000000**.

### A3.5 — the render-bias confound has a MECHANISM, and it is narrower than A3.3 said
⭐ A3.3 recorded the over-rendering as *"the confound isn't uniform"*. That is an observation, not an
explanation, and a reviewer would rightly ask **why not**. Verified against `factor_index.npz`:

| factor | card | median index | **the value it selects** | render bias |
|---|---|---|---|---|
| floor_hue | 10 | 5 | `0.5` — a point on a circle, no privileged status | **1.00** |
| wall_hue | 10 | 5 | `0.5` — likewise | **0.99** |
| object_hue | 10 | 5 | `0.5` — likewise | **0.66** |
| scale | 8 | 4 | **`1.0357` — essentially the DEFAULT OBJECT SIZE** | **2.14** |
| shape | 4 | 2 | `2.0` — categorical, no null | 1.47 |
| orientation | 15 | 7 | **`0.0°` — EXACTLY the NULL POSE, zero rotation** | **2.69** |

⭐⭐ **The median index `card//2` lands on the physical null for exactly the two factors that HAVE one.**
An averaging or shrinking decoder reverts toward the default size and the zero-rotation pose, so for
`scale` and `orientation` the rule selects precisely the value such a decoder over-produces anyway —
hence 2.14× and 2.69×. For a **hue** there is no "default hue" to revert to, so there is no inflation,
and the three hues come out at 1.00 / 0.99 / 0.66.

⛔ **So the caveat is NOT "median-index selection is biased".** It is: **median-index selection is biased
for factors with a physical null**, and is unbiased for circular factors. That is narrower, more
defensible, and it tells a reader when the rule is safe to reuse.
**Testable prediction attached** (⛔ not run here, no budget): running the rule at a different index
should make the inflation **shrink for `scale` and `orientation` as the index moves away from the null
pose, while the hues stay flat**.

### A3.6 — `orientation` needs care in the three-way test
⚠ `orientation` is the one factor whose pre-deletion profile is already **bimodal**: 0.500 exact, 0.368
at one step, then a **0.128 tail at `|d| ≥ 3`** — and (A3.2) that tail is genuine error, ⛔ not wrap.
So for `orientation` the three-way classification must be read against a distribution that already has a
populated far tail: **mass moving INTO the far tail rather than to `d=1` reads as DAMAGE, not deletion.**
Registered now so the reading is not invented afterwards.

### A3.7 — the DISTANCE PROFILE needs the churn control too
⚠ A3.1 compares the post-deletion profile to the **pre-deletion** one. That carries the **same defect
already fixed for collateral**: a rebuild moves the profile on its own. So the decisive quantity is
**`d0(deletion) − d0(size-matched PLACEBO)`**, ⛔ not `d0(deletion) − d0(base)`.
`plc_f2v5` removes 3,207 random rows, rebuilds, and scores the **identical** targeted set — so it is the
churn floor and it is **free**. ⭐ My own sentence applies to my own statistic: *without the band, this
would be reporting the rebuild, not the deletion.* The three-way verdict is reported **churn-corrected**,
with the uncorrected version beside it.

### A3.8 — ⭐⭐ THE MECHANISM CHECK: no probe, no decoder, no accuracy
After `delete(object_hue = 5)`, `fit_address_book` fits that conditional row on **zero live rows**, so
Laplace smoothing gives `(0+1)/(0+1·M) = 1/64` for **every** mode. The row must therefore be **exactly**
`log(1/64) = −4.1588830833596715` in all 5 × 64 entries, with within-well range **exactly 0.0**.
**Calibrated on `det_B`** (measured, not assumed): every value's row is far from uniform —
within-well log-range `v=0: 5.386`, `v=4: 5.551`, `v=5: 5.434`, `v=6: 5.467`, `v=9: 5.364`.
⭐ **This is decisive where an accuracy is only suggestive:**
- if the row is **not** exactly uniform, the delete did **not** do what the mechanism says — whatever the
  accuracies show;
- if it **is** exactly uniform **and the value still comes back**, the addressing term is **provably
  gone** and the value re-composed anyway. That is the **strongest possible form** of the re-composition
  finding, and it is the gs charter's central question answered at the level of the mechanism rather
  than of a score.
⭐ `object_hue` is the unit to decide this on: its pre-deletion far tail is **exactly 0.000**, so **any**
appreciable `|d| ≥ 2` mass after the deletion is **new**, and is damage rather than deletion.

### A3.9 — ⭐⭐ THE CONCEPTUAL LIMIT, stated plainly because it decides how a null reads
⛔ **I overstated the mechanism and the correction matters.** I wrote that after deletion *"there is no
surviving hue-5 mode for the store to route to"*. That is **stronger than the code supports.** A mode is
**not hue-labelled**: `init_store` runs k-means++ over `X_live` restricted to a well's **channel** subset
(well sizes 303/305/573/501/366 of 2048) and `train_store` then moves those centres. Hue is one of six
factors varying in that subspace, and with **M = 64 modes against 10×8×4×15 = 4,800 factor combinations**
the modes cannot be hue-pure. Removing the hue-5 rows shifts every centre — but **a centre fitted on
hue-4 and hue-6 material can land at a hue-5-like position by interpolation, because it is a mean of
vectors either side of it.**
⭐ The correct statement: after deletion **no mode was FITTED on a hue-5 feature**, but ⛔ **nothing
guarantees no mode SITS at a hue-5 position in feature space.**

⭐⭐ **Therefore, and this is the heart of this hub:**
> **The ported mechanism guarantees the store is byte-identical to one that never held `v`.
> It does NOT guarantee that no request can elicit `v`. Those are different claims, and the charter's
> condition 1 asks for the second. A store that never held `v` can still emit `v` by interpolation.**

⛔ If that is what is measured, it is **not a failure of the deletion** — it is the **honest limit of what
store-level deletion buys**, and it is exactly the caveat App. G.8 already carries verbatim
(*"store-level guarantee only"*). It will be reported in those words, ⛔ not as a defect and ⛔ not as a win.

### A3.10 — the full rendered histogram, because the distance bins collapse two different outcomes
⚠ `d=1` mass and "flat over all hues" are **different findings** and the `|d|` binning merges them. The
**full rendered-hue histogram** on `tgt_f2v5` post-deletion discriminates:
- **mass concentrates on {4, 6}** → the request still carries hue information; the addressing lost the
  *specific value* while the **ordinal neighbourhood survived**. ⭐ Arguably the **best** outcome for the
  programme — surgical removal of one value from an intact structure — ⛔ not a partial result.
- **mass ≈ uniform over the 10 hues (~0.10 each)** → hue addressing **collapsed entirely** for that
  request: the value is gone, and so is the store's ability to place the request anywhere on the circle.
- **mass stays at 5** → re-composition — and since A3.8 will have proven the conditional is *exactly*
  uniform, it is **interpolation through the decoder**, ⛔ not a retained route.
Reported as the histogram plus a concentration statistic, ⛔ never as a single distance number.

### A3.11 — ⭐ THE FEATURE-LEVEL TEST: geometry, not the probe's verdict
The probe is a 10-way classifier with its own biases, so *"the probe said hue 5"* is weaker than
*"the emitted feature IS hue-5-like"*. ⛔ **Needs no encode** — every comparison row is already in the
frozen cache.
**Construction, exploiting the complete grid** (verified: row `= ((((fh·10 + wh)·10 + oh)·8 + sc)·4 + sh)·15 + ori`
reproduces `factor_labels` on every probe row): for each targeted request — a full tuple with
`object_hue = 5` — the **same scene with only the hue changed** to 4 and to 6 exists as real rows. So:
```
x_i   = the feature the store EMITS for request i          (request-addressed, post-deletion)
f5_i  = the CACHED feature of the real row, oh = 5          (same scene)
f4_i, f6_i = the CACHED features of the same scene, oh = 4 and 6
```
**Statistic:** the fraction of requests where `‖x − f5‖ < min(‖x − f4‖, ‖x − f6‖)` — the emitted feature
is closer to genuine hue-5 content than to either neighbour. ⭐ Reported against the same statistic on
`det_B` (pre-deletion) and on `plc_f2v5` (the churn floor), so it carries its own controls.
⭐ This converts *"did it re-compose"* from a probe verdict into a **geometric measurement**, and it is
the version that survives the objection that the probe is doing the rounding.

### A3.12 — ⭐⭐ THE PARTITION MOVES. Registered after `del_f2v5`'s BUILD, before its EVAL.
⚠ **Timing, stated honestly:** this is written after `del_f2v5`'s **build.json** exists but before its
`eval.json` — so I have seen the **layout**, ⛔ not any accuracy, verdict or collateral number. The
statistic added below is computed from `modes.npz` alone. ⛔ Marked as post-build so a reader can
discount it accordingly; ⭐ it is **predicted** by `AUDIT-01` (written at 11:07Z, long before any
deletion) and is therefore not a discovery made to suit an outcome.

**Measured on `del_f2v5` (deleting `object_hue = 5`, 3,207 of 32,768 rows):**
| | undeleted (`det_A/B/C`) | `del_f2v5` |
|---|---|---|
| well sizes | `[303, 305, 573, 501, 366]` | **`[672, 234, 234, 546, 362]`** |
| ARI vs undeleted | 1.0000 (all three pairs) | **0.5656** |
| best agreement under relabelling | — | **0.8027** (1,644 / 2,048 channels) |

⭐ `AUDIT-01` predicted exactly this: `n_corr = 16384`, so the correlation matrix is fitted on the **first
16,384 rows of the SORTED live vector**; removing 3,207 rows shifts that window and `ward` returns a
different partition. **~20 % of channels changed wells and the largest well more than doubled
(303 → 672).**

⛔⛔ **Consequence, and it is stronger than what A2.2 claimed.** A post-delete store is **not a
perturbation** of the pre-delete store — it is a **different object with a different layout**. So every
per-factor difference between `del_f2v5` and `det_C` confounds *"`object_hue=5` was deleted"* with
*"a fifth of the representation was re-partitioned"*. ⭐ That is **the mechanism working as specified** —
a canonical function of the live set is entitled to return a different layout for a different live set —
but it means **the uncorrected collateral number is close to uninterpretable on its own**, which is a
stronger statement than "the churn-corrected version is co-primary". It is now the **only** meaningful
collateral number, and the uncorrected one is reported as a diagnostic.

**`ARI vs undeleted` is added as a reported statistic for every unit** (free, from `modes.npz`, no eval).
**The decisive comparison is `ARI(deletion)` vs `ARI(size-matched placebo)`:**
- **comparable** ⇒ partition churn is **generic to rebuilding at that size**, not caused by the
  semantics, and the churn-corrected collateral is the only number that means anything;
- **placebo much higher** ⇒ the **structured** deletion destabilises the layout in a way random removal
  does not — ⭐ a finding in its own right that **no accuracy number would have revealed**.
⚠ Note for the table: `loss_final` is **31.64753** for `del_f2v5` against **32.04901** undeleted. Lower
loss on a **smaller** live set is expected and is ⛔ **not** evidence that deleting improved the store.

---

## AMENDMENT A4 — 2026-09-17T12:02Z · ⭐⭐ THE THREE-RUNG LADDER
⭐ **Window: pre-hoc for every unit it concerns.** `plc_f2v5` has **no `build.json` yet**, and
`plc_f0v5`, `plc_f1v5`, `plc_churn_r1/r2/r3` are **unbuilt**. ⛔ No number from any of them exists.

**The insight (from `clu-dev-1e`): the decisive test of the MECHANISM does not need an undeleted store
at all.** `placebo vs undeleted` still differs in `n_live` (32,768 vs ~24,534–29,561). But **two placebos
of the same size differ ONLY in which arbitrary rows went** — matched on `n_live`, on deletion size, on
code, on everything except the draw. So:

| rung | comparison | what it isolates | status |
|---|---|---|---|
| **1** | `det_A` / `det_B` / `det_C` — **identical** live set | reproducibility | ⭐ **PROVEN**: 22/22 bitwise identical, **0.00 %** answer churn, ARI **1.0000** |
| **2** | ⭐⭐ **placebo vs placebo at the same size** — different random draws | **pure recanonicalisation instability**, ⛔ independent of *what* was deleted and even of the *fact* that something semantic was | **pending** |
| **3** | `del_f2v5` vs `det_C` — structured deletion | deletion + rebuild, confounded | measured: **19.77 %** churn, κ **0.5726**, ARI **0.5656** |

⭐ **Rung 2 is strictly stronger than placebo-vs-undeleted** and it is **already in the plan twice**:
- **n ≈ 3,233** — `plc_f0v5` (3,269) · `plc_f1v5` (3,224) · `plc_f2v5` (3,207). Three pairwise
  comparisons, `n_live` within **0.2 %**. ⭐ **These arrive ~2 h earlier than the n = 8,234 set**, which
  neither of us had noticed. Size mismatch stated wherever used.
- **n = 8,234** — `plc_f4v2` · `plc_churn_r1/r2/r3`. Six pairwise comparisons, `n_live` **identical**.
**Reported per pair: answer churn (lost/gained), κ, and ARI** — giving a *spread*, ⛔ not a point, before
any *"X % per rebuild"* sentence is written.

### ⭐⭐ The pre-registered reading, and why this is a CONTRIBUTION rather than a caveat
**If rung 2 returns ≈ 20 % churn / κ ≈ 0.57 / ARI ≈ 0.57**, then recanonicalisation is **intrinsically
unstable**, and the instability has nothing to do with the semantics of what was deleted — or with
anything having been *meaningfully* deleted at all.
⭐ **That is a finding about the PORTED MECHANISM, not about the CLU.** *Retention You Can Predict, Scope
and Delete* guarantees the post-delete store is **byte-identical to one that never held `v`**, and prices
the mechanism honestly in **time** (O(n) rebuild per delete, stated openly). ⛔ **It says nothing about
behavioural stability** — because *byte-identity-with-a-counterfactual* and *closeness-to-the-pre-deletion
store* are **different properties**, and the paper claims only the first.
⚠ But a practitioner deleting one record from a deployed system would reasonably expect the system's
**other answers** to be stable. If a fifth of them change, that is a **real caveat on the ported method
that the source paper does not carry**.
⭐⭐ **And unlike condition 1, a lookup table does NOT get this for free** — it has no layout to
recanonicalise, which is exactly why `knn_proto` churns 1.11 % against the store's 19.77 %. **This is the
one place in this hub where the CLU's structure being load-bearing produces a FINDING rather than an
excuse.**
**Candidate headline sentence, fixed now:** *"Exact structural deletion is achievable and cheap to
verify, but recanonicalisation is not answer-preserving: it costs ~X % of available persistence per
rebuild, independent of what is deleted."* ⛔ `X` is written only after rung 2 supplies a spread.
