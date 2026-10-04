# PREREG — gen-loss-attribution
**Filed BEFORE the first measurement.** ⛔ Never edited after filing; any addendum is a NEW file
(`PREREG-addendum-N.md`, R34 standing rule, seventh strike).

Agent: experiment-engineer · Branch `agent/experiment-engineer/gen-loss-attribution` off `main @ 994ca9b`
Worktree `../CHLU-gen-attrib`, main venv (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`.

---

## 0. ⭐ THE PRE-REGISTERED READING — copied VERBATIM from the task file

| outcome | reading |
|---|---|
| encoder **≥ ~0.9** AND code **≈ chance** | ⭐ **World A — the STORE drops it.** The fix is in the write-side quantiser. |
| encoder **lacks it** | ⭐ **World B — the store is INNOCENT.** The fix is the encoder, and the encoder track owns it. |

⛔ Anything in between is reported as in-between, ⛔ never rounded to whichever world is more convenient.

## 0a. DIAL DECLARATION (echoed)
- **Dial:** *attribution* — ⛔ no fix, no new mechanism, no arm. Locating a known loss, not repairing it.
- **Laundering control:** the **same probe class, same hyper-parameters, same train/test rows, same seeds** at all
  three points.
- **Falsifies:** nothing of ours. Both outcomes are informative by design.
- **Proves this spoke wrong:** the **random-code control** scoring materially above the majority baseline.

---

## 1. Operational definitions, fixed here

- **Point 1 `encoder`** — linear probe on the full frozen 2048-d feature vector.
- **Point 2 `well`** — the SAME probe on one well's sub-vector `X[:, well_channels[k]]`, one fit per well
  (`K = 5`, `d_k` ∈ {303, 305, 573, 501, 366} at seed 0). Headline = **`max_k`** (the best single well).
  ⚠ The wells PARTITION all 2048 coordinates, so the *concatenation* of all five sub-vectors is the full feature
  vector up to a column permutation; the only meaningful point-2 statistic is therefore **per-well**.
- **Point 3 `code`** — the SAME probe on the **one-hot** of the integer K-tuple `address_codes(store, X)`,
  width `K·M = 5·64 = 320`.
- **Control `code_random`** — the SAME probe on a one-hot of `K` uniform draws from `[0, M)`, width 320.
- **Probe:** `chlu.core.gen_store.WellProbe` (one `eqx.nn.Linear` head per factor, multinomial logistic, Adam),
  identical `probe_steps` / `probe_batch` / `probe_lr` / PRNG key at every point. Per-point **train** accuracy is
  reported alongside test accuracy so an under-trained probe cannot be mistaken for a loss.
- **Rows (SPREAD / "the grid"):** a seeded sample of `split.train_rows \ store_train_rows`, i.e. rows spread over
  the whole factor space that the **store never fitted**; disjoint seeded train/test halves.
- **Rows (CORNER, reported SEPARATELY):** `eval_b` (the 72 Montero held-out rows) and, as R16's permanent
  companion, the matched-seen rows `MS` in the same corner. The probe is **trained on SPREAD** and evaluated on
  these; ⛔ no probe is ever trained on 72 rows.
- **`chance`** = `1 / cardinality` (uniform). **`majority`** = the train-majority class evaluated on the test rows.
  Both reported; the "≈ chance" clause is adjudicated against `max(majority, chance)`.
- **`retention`** at a point = `(acc_point − chance) / (acc_encoder − chance)` — the fraction of the encoder's
  above-chance signal surviving to that point. ⛔ Undefined (reported as such) if `acc_encoder ≤ chance`.
- **`n_values_realised`** is reported beside every per-factor number on every row set (R31, fourth strike).

## 2. ⭐ The probe-capacity asymmetry, declared IN ADVANCE

A linear probe on 2048 float dims and a linear probe on a 320-d one-hot are **not** equally expressive, and that
asymmetry could by itself manufacture a "loss". Two things bound it, both pre-registered:

1. **`code_random`** (above) — what the 320-d probe scores with **zero** information.
2. **A non-parametric code-tuple LOOKUP upper bound.** Train: the majority factor value of each observed K-tuple;
   test: that value, falling back to the global majority for unseen tuples. This is ⭐ **strictly more expressive
   than any linear head on the one-hot** (it can represent arbitrary inter-well interactions, which an additive
   linear probe cannot), so if the lookup bound is ALSO near chance the loss cannot be blamed on probe capacity.
   Its own `code_random` counterpart is reported to expose lookup overfitting.

## 3. ⭐ POINT PREDICTIONS (committed here, with derivations)

Numbers are the 3-seed mean, SPREAD test rows, factor `scale` (8 classes, chance 0.125) and `orientation`
(15 classes, chance 0.067).

| id | quantity | prediction | 80 % interval | derivation |
|---|---|---|---|---|
| **P1** | `encoder` scale | **0.75** | [0.55, 0.92] | `gen-feature-bridge` reads `scale` at **0.8555 ± 0.0543** on A through decoder→CNN-probe; the information is demonstrably in the features. A *linear* head on raw 2048-d features should lose some of it. |
| **P2** | `encoder` orientation | **0.60** | [0.35, 0.85] | same chain reads `ceiling_A` orientation at **0.9074**; 15 smooth classes penalise a linear head harder than 8 scale classes. |
| **P3** | best-well `scale` | **0.60** | [0.35, 0.85] | a well holds 15–28 % of the coordinates; `gen-store-grouping`'s relevance matrix shows `scale` is **not** concentrated on any well (max 0.42 of 3.00 bits), so the best well should lose more than a proportional share. |
| **P4** | best-well `orientation` | **0.45** | [0.20, 0.75] | same matrix: orientation's best well carries 1.23 of 3.91 bits — more concentrated than scale, less than the hues. |
| **P5** | `code` scale | **0.20** | [0.13, 0.35] | R34's external classifier reads `scale` **0.229** (chance 0.125) through read→decode→classify on the spread grid; the MI row caps it (0.42/3.00 bits ⇒ ~14 % of the factor's entropy). |
| **P6** | `code` orientation | **0.17** | [0.07, 0.35] | R34 external classifier **0.236** (chance 0.067); MI 1.23/3.91 bits ⇒ ~31 % of entropy, but spread over wells an additive linear head cannot combine. |
| **P7** | `code_random` scale | **0.128** | [0.11, 0.16] | zero information ⇒ the majority baseline of a near-balanced 8-class grid. ⛔ **If this exceeds 0.20 the spoke is wrong** and every code-level conclusion is withdrawn. |
| **P8** | `code_random` orientation | **0.068** | [0.05, 0.10] | same, 15 classes. |
| **P9** | lookup bound ≥ linear `code` probe | **holds**, gap ≤ 0.15 | — | a lookup is strictly more expressive on the same variable. ⛔ A *negative* gap means the lookup is under-trained/over-fit and must be reported as such. |
| **P10** | `code` **hue** factors | **≥ 0.70** | [0.55, 0.95] | ⭐ the mandatory positive control: R31/R34 say the 30-bit read **carries** hue. If hues also collapse, the harness is broken, ⛔ not the store. |
| **P11** | CORNER `code` scale vs SPREAD `code` scale | CORNER **lower** | — | R34: the loss is global but *worse* in the corner (external classifier 0.021 in-corner vs 0.229 on the grid). |
| **P12** | MI row for `scale`, max over wells | **≤ 0.6 bits** of `H = 3.00` | [0.2, 1.0] | `gen-store-grouping` seed 0 measured 0.42 bits max. ⭐ Independent of the probe. |

## 4. ⭐ THE VERDICT RULE — fixed before measuring, applied mechanically

Per factor `f`, with `c = chance_f`:

- **World A** iff `acc_encoder ≥ 0.90` **AND** `acc_code ≤ max(majority_f, acc_code_random_f) + 0.05`.
- **World B** iff `acc_encoder < (c + 0.90) / 2`  (⇒ `scale` < **0.5125**, `orientation` < **0.4833**) —
  the operational reading of "the encoder lacks it".
- **Otherwise IN-BETWEEN**, reported in those words, with the retention fractions naming the direction.

## 5. ⭐ MY PRE-REGISTERED VERDICT

Given P1 (0.75 < 0.90) and P5 (0.20, ≈ code-random + 0.07), the mechanical rule returns
⭐ **IN-BETWEEN, strongly World-A-leaning**, for `scale`: *the encoder holds `scale` well above the World-B
threshold but not at the 0.9 bar, and the code is near chance.*
For `orientation` (P2 = 0.60 > 0.4833) the same: ⭐ **IN-BETWEEN, World-A-leaning**.

⛔ I am explicitly **not** predicting a clean World A, and I will not round to one. The interesting quantity is
then the **partition vs k-means split** (point 2): if `best-well ≈ encoder` the partition is innocent and
**k-means** threw the factor away; if `best-well ≈ code` the **partition** threw it away before k-means saw it.
**My prediction: the partition costs ~0.15 and k-means costs the remaining ~0.40** on `scale` — i.e. ⭐ **k-means
is the larger culprit**, which is R40.B's mundane hypothesis (k-means keeps high-variance directions).

## 6. What is measured but NOT predicted
The per-well MI table read out of the store (the free instrument), `n_values_realised` on every row set, and the
CORNER/MS numbers beyond P11's direction. ⛔ Any disagreement between the MI table and the probe is **reported as a
disagreement**, never resolved in favour of the convenient one.
