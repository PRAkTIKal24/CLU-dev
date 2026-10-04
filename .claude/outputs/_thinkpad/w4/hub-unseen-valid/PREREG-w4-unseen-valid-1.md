# PREREG — w4 hub/unseen-valid — SCORED RUN 1 of 3

**Written 2026-09-19T17:46Z, BEFORE any arm number on the held-out estimand.**
Hub `w4/unseen-valid` · base `base/iter4 = 9dfed5e` · worktree `/c/chlu/worktrees/CHLU-w4-unseen-valid`
Target: **scoreboard row 2 — unseen-valid** (a hue requested strictly between two stored values).
⛔ Charter §A.3: **I may not change a bar in this file once I have seen a result against it.**
Amendments are appended, dated, never edited.

---

## 1. The instrument — the SECOND JUDGE, and why judge 1 cannot score this row

`w4-hub-common.md` §4.5: top-1 judging cannot separate a perfect arm from a perfect lookup on row 2 —
usable span exactly **0.0000**. The pinned judge (`Judge.read_back` → `predict_factors`) is `argmax`
over **10 hue classes at spacing 0.1**; a request at 0.05 has **no correct class**.

**The second judge, as I built it** (it is not in the pin; §3 names it, it does not ship it):

1. The classifier RAW HUE LOGITS on the decoded render (⛔ not the softmax — see §1.1).
2. The readout is the **logit difference of the two bracket classes**: `r = L[hi] - L[lo]`.
3. It is **ORACLE-REFERENCED**: for each request the two **real dataset rows** that bracket it (same
   five other factors, hue at the two stored values) are decoded through the same decoder and the same
   classifier, giving anchors `a_lo`, `a_hi`. Both anchor rows are **guaranteed to be in the store fit
   pool** by the pinned builder own bracket condition, and my code **asserts it**.
4. **Position** `p = (r - a_lo) / (a_hi - a_lo)` — `p=0` the low stored value, `p=1` the high one,
   **`p=0.5` the requested point**.
5. **The scored quantity** `e = |p - 0.5|`. **0 = the requested point exactly; 0.5 = a perfect lookup.**

⭐ This is the anchor my task file requires the snap to be measured against — ⛔ not the raw readout.

### 1.1 Why the logit readout and not the softmax — decided by measurement (n=256, seed 0)

The classifier is near one-hot on real renders (max prob **0.99988** / **0.99992**), so the softmax
posterior mean is effectively a QUANTISER and a genuine interpolation leaves the interior:

| readout | measured span | interior (.2,.8) | interior (.05,.95) |
|---|---|---|---|
| softmax posterior mean | 0.09999 +/- 0.00009 | 0.180 | 0.387 |
| two-class softmax | 0.99987 +/- 0.00081 | 0.176 | 0.383 |
| ⭐ **RAW LOGIT DIFFERENCE** | **23.64 +/- 2.53** | **0.824** | **0.996** |

---

## 2. ⭐⭐ THE MEASURED SPAN — stated before the first arm number, as my task file requires

Measured on **cache generation 2**, seeds **0, 1, 2**, n = 4096 requests, **70 corners**, on two
reference objects, ⛔ **NEITHER of which is an arm**:

* **LOOKUP FLOOR** — a real bracket feature fed straight back (a perfect lookup): **e = 0.5000**, exact.
* **REFERENCE INTERPOLATOR** — the feature-space midpoint of the two real bracketing features; a
  genuine interpolation available to no store: **e = 0.1728 +/- 0.0074**.

| seed | anchor span (logits) | lookup e | interpolator e | SPAN | corners positive | calibration monotone |
|---|---|---|---|---|---|---|
| 0 | 23.79 | 0.5000 | 0.1629 | **0.3307 +/- 0.0068** | 70 / 70 | yes |
| 1 | 24.54 | 0.5000 | 0.1806 | **0.3154 +/- 0.0072** | 70 / 70 | yes |
| 2 | 23.63 | 0.5000 | 0.1748 | **0.3194 +/- 0.0072** | 70 / 70 | yes |

### ⭐⭐ **THE SPAN = 0.3218 +/- 0.0065 (sd across 3 seeds).** Anchor span 23.98 +/- 0.40 logits.

**The span has a usable interior, proved not assumed.** The calibration sweep along the segment at
`t = 0, .25, .5, .75, 1` gives `p(t) = 0.0000 / 0.1378 / 0.5307 / 0.8874 / 1.0000` — **monotone on
every seed** — and `frac_span_positive = 1.000`, i.e. monotone on every single request. ⭐ The judge
resolves **position**, not merely "not an endpoint".

---

## 3. ⭐⭐ THE MARGIN — declared INSIDE that span, before the first arm number

### **MARGIN d = +0.0300 on `e`, per-corner paired mean. That is 9.3 % of the measured span 0.3218.**

**Decision rule** (the pin own `paired_estimand`): an arm BEATS a comparator iff
`mean_paired_delta > d` **AND** `bootstrap_CI95_lower > d`, where `delta = e(comparator) - e(arm)`
and lower `e` is better. Otherwise it is a **TIE** (or a loss). Required on **all three seeds**.

**Why 0.0300 and not something else:**

* the per-corner SEM of the span is about 0.0070, so **d is about 4.3 SEM** — not a rounding artefact;
* it is **discriminating**, which is the point: on `val` it **separates** my two named rivals —
  table interpolation (dev delta **+0.0677**, CI lo +0.0462) **clears** it, while the continuous-input
  MLP (dev delta **+0.0051**, CI lo -0.0099) **does not**. A margin that both rivals cleared, or that
  neither cleared, would decide nothing.

---

## 4. The estimand

* **Unit: the CORNER**, paired. ⛔ Never a single cell (§3).
* **Corner tiling (declared):** (bracket pair) x (floor_hue band, width 2) x (scale half, width 4)
  = 7 x 5 x 2 = **70 possible, disjoint and exhaustive** — every request lands in exactly one.
  **70 realised, which is >= 49**, asserted in code before any number.
  min 38 / median 56 / max 99 requests per corner.
* **Requests:** `build_row2_unseen`, the **pinned** builder, ⛔ UNEDITED, at `n=4096`, `row2_seed=0`;
  the scored block is records **0 ... 4095**.
* **⭐ SUPERSET ASSERTION:** records 0...511 are **byte-identical** (`json.dumps`, sorted keys) to the
  frozen `row2_unseen.json`. Asserted in code, every seed, before any number.
* **Selection set (disjoint, already spent):** records **4096 ... 8191** under `condition=val`.
  ⛔ No scored number comes from it.
* **Statistics:** mean paired delta, 10,000-sample paired bootstrap CI95, two-sided sign test,
  corners won/tied/lost. Three seeds (0, 1, 2), reported separately and together.

## 5. What runs, with its ledger

Cache **generation 2 only**; the per-seed encoder hash is asserted equal to the hash the pinned decoder
was fit against. `condition=test`. Every read below runs on the **same store, same head, same bytes**
where it can, so the only thing that differs is the read verb.

| object | role | bytes / params |
|---|---|---|
| `linear_span_ARM` (`factor_aligned_balanced` span read) | **THE ARM** | 524,288 B store + 1,766,400 B head |
| `cont_mlp_ARMMATCHED_RIVAL` | ⭐ **RIVAL**, named falsifier | 573,440 params = 100.1 % of the arm |
| `cont_mlp_TABLEMATCHED_RIVAL` | ⭐ **RIVAL** at the spoke-A basis | 118,784 params |
| `table_interp_knn_proto_RIVAL` | ⭐ **RIVAL**, named falsifier | 475,136 B |
| `lookup_argmax_FLOOR` | charter §C.1/C.4 matched-bytes lookup | the same atoms |
| `correlation_split_linear_head_ABLATION` (`basis_head`) | §1 like-for-like ablation | 524,288 B |
| `valley_ARM` (rho=1.0, lam=0.0) | ⭐ **the DIAL, reported as a declared NULL** | identical to the arm |
| `landscape_removed_CONTROL` | §C.2 — must collapse | byte-identical |
| `landscape_shuffled_CONTROL` | §C.2 — must collapse | byte-identical |

## 6. Expected direction, declared in advance

1. `table_interp` — the arm **BEATS** it beyond d. (dev +0.0677)
2. `cont_mlp_ARMMATCHED` — **TIE**, does not clear d. ⛔ **This is my dial falsifier and I expect it to
   fire.** (dev +0.0051, CI straddles zero, corners 36/34)
3. `lookup_argmax` — the arm **BEATS** it beyond d. (dev +0.2178)
4. `valley_ARM` — **TIE with the arm** (dev |delta| <= 0.0014, all 12 CIs straddle zero). A declared
   NULL.
5. `landscape_removed` / `landscape_shuffled` — **must COLLAPSE** (§C.2).
6. `basis_head` — an ablation: reported, ⛔ NEVER claimed as a result (§1).

## 7. ⛔ What this run may NOT be used to claim

* ⛔ A win over `basis_head`, `linear_span` or `valley` is an **ABLATION**, not a result (§1).
* ⛔ **Zero-supervision structure is not claimed.** Labels are used ONCE at fit time to align wells and
  at the address book — declared in every caption (§5).
* ⛔ If `cont_mlp_ARMMATCHED` ties or beats the arm, **the DIAL DECLARATION is FALSIFIED** and I report
  it as such. ⛔ A win over the table-matched MLP alone is a win over a rival **4.8x smaller than the
  arm it is compared against**, and will be labelled so on the row.
* ⛔ Rows other than 2 are **REPORTED, NEVER CLAIMED**.

## 8. Stop rule

⛔ The run does not start until `RIVALS-READY.json` exists and the rivals I need are `true` in it. If
`cont_mlp` is still `false` at run time I record that fact and state exactly which rival was reproduced
in-harness at which declared parameter basis. ⛔ Scored runs spent before this one: **0 / 3**.
