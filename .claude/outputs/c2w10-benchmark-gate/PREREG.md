# PREREG — c2w10-benchmark-gate (protocol §5 pre-registration rule)

**Filed 2026-08-10 by results-analyst, BEFORE the B1-ARF and B2 exemplar-arm harness was run.**
Append-only. Amendments dated and labelled.

## 0. What is registered here vs. inherited

The **decision rules are NOT mine to derive** — they are registered in
`.claude/outputs/c2w10-persistent-store/PREREG-C2W10.md` §3 (legs B1–B4) and I apply them mechanically:

- **B1 PASS** iff our No-Change AND our ARF are each within **±2.0 points** of Souza Table 5
  (**No-Change 40.46**, **ARF 77.13**) on `incremental-reoccurring (balanced)`.
- **B2**: `criterion4_cleared = false` iff `max(exemplar arms) ≥ ARF − 2.0` points; `true` iff the best
  exemplar arm is **clearly below** ARF, i.e. `max(exemplar arms) < ARF − 2.0`.
  ⇒ `b2_margin_pts = max(exemplar) − ARF`; `criterion4_cleared = (b2_margin_pts < −2.0)`.
- **B3**: ARF's `κ_per > 0` required.

What this file registers is **my point predictions**, **the two protocol choices I had to make that are
not fixed by the prereg**, and **why**.

## 1. HONEST DISCLOSURE — one number was computed before this file was written

The **No-Change (persistence) baseline is a deterministic function of the label column** (`ŷ_t = y_{t−1}`,
no model, no seed, no free parameter), and I computed it while establishing the stream facts, i.e.
**before** writing this prereg. Reported value, disclosed here rather than presented as pre-registered:

- `incremental_reoccurring_balanced`: **40.4526 %** (published 40.46)
- `incremental_abrupt_balanced`: **42.3779 %** (published 42.39)

Every number produced by a *fitted* arm (ARF, SAM-kNN, kNN_S) post-dates this file.

## 2. Two protocol choices that the inherited prereg does not fix (declared BEFORE running)

### 2.1 ARF ensemble size — MOA default, not river default
Souza's Table 5 used **MOA** implementations. MOA's `AdaptiveRandomForest` default ensemble size is
**100** trees; **river's `ARFClassifier` default is `n_models=10`**. These are not the same estimator.
⇒ **The registered B1/B2 ARF reference arm is `n_models=100`** (MOA default, matching Souza's
toolchain), with `n_models=10` reported as a labelled secondary row. If runtime forces a smaller
ensemble I will declare it and report both, per the task's anti-hobbling clause.
Other river ARF defaults already match MOA: `lambda_value=6`, `max_features='sqrt'`,
`drift_detector=ADWIN(0.001)`, `warning_detector=ADWIN(0.01)`.

### 2.2 Feature scaling for the exemplar arms — the F3 anti-hobbling decision
The frozen river CSV is **not normalised**: 7 of 33 features are O(10²–10³) (measured range up to
**8014.5**) while 26 are O(1). ARF is a tree ensemble and is **scale-invariant**; Euclidean kNN is not.
Running the exemplar arms on raw features would let 7 columns dominate the metric and would be a
**hobbled baseline** — the same referee attack in mirror image (task §4, F3).
⇒ **Every exemplar arm is run TWICE: `raw` and `std`**, where `std` is a **prequential, causal**
per-feature standardisation (running mean/var over instances seen so far, updated *after* the
prediction; **no lookahead, no global statistics**). **The B2 arithmetic consumes `max` over BOTH
variants** — the exemplar store gets its strongest honest shot. Both are reported.

### 2.3 SAM-kNN implementation provenance
`river 0.25.0` does **not** ship `SAMKNNClassifier` (the scout's §5 claim that it does is **wrong** and
is on my reconciliation list). ⇒ I port **the authors' own reference implementation**
(github.com/vlosing/SAMkNN, `SAMKNN/SAMKNN.py`) from Python 2 to Python 3, replacing only the
`libNearestNeighbor` C extension with a numpy equivalent whose semantics I read off
`nearestNeighbor/nearestNeighbor.cpp` (squared-Euclidean distances; `nArgMin` = n smallest with
lowest-index tie-break; `linearWeightedLabels` = weight `1/max(d,1e-9)` on the *Euclidean* distance,
lowest-label tie-break). Published defaults: **k=5, L_min=50, L_max=5000, `knnWeights='distance'`,
`LTMSizeProportion=0.4`, `recalculateSTMError=False`** (the authors' own test-script setting).
**kNN_S(L)** is the same file at `useLTM=False, recalculateSTMError=None, maxSize=L` — which the
authors' own commented-out test config identifies as the plain sliding-window kNN arm.
**Equivalence of the port to the C extension is asserted by a unit test against a brute-force
reference** before any arm is run; that test result is reported.

## 3. Point predictions (scored at the review either way)

`incremental_reoccurring_balanced`, prequential accuracy, sliding window 1000, m = 1 (undecimated).

| # | quantity | my point prediction | my 80 % interval | prior |
|---|---|---|---|---|
| P1 | ARF (100 trees) accuracy | 77.0 | 73–80 | — |
| P2 | **B1 passes** (both within ±2.0) | — | — | **0.55** |
| P3 | kNN_S(5000, std) accuracy | **76.0** | 70–82 | — |
| P4 | SAM-kNN(5000, std) accuracy | **78.0** | 72–84 | — |
| P5 | kNN_S(1000, std) accuracy | 73.0 | 66–80 | — |
| P6 | kNN_S(5000, raw) accuracy | 62.0 | 50–74 | — |
| P7 | `b2_margin_pts` = max(exemplar) − ARF | **+1.0** | −4 to +6 | — |
| P8 | **`criterion4_cleared = true`** (exemplar clearly below ARF) | — | — | **0.25** |
| P9 | ARF `κ_per` > 0 | — | — | 0.97 |

**Reasoning for P8 = 0.25, deliberately BELOW the Hub's Q2 = 0.55.** The mechanism the Hub is counting
on is "the regime variable is hidden, so the label is not a function of the query point alone". That
argument constrains a *pooled* metric learner. It does **not** constrain a **recency-windowed** one.
A 5 000-example window on a 79 986-instance stream spans ≈6.3 % of the stream ≈ **1.3 °C** of a 20 °C
sweep — so kNN_S(5000) is, to a good approximation, *the per-temperature classifier Souza measured*,
and Souza measured per-temperature classifiers at **90 %** vs **84 %** pooled (86 % vs 66 % at 24 °C).
Recency **is** the hidden regime variable on this stream, and a sliding window reads it for free.
⇒ my prior is that the exemplar arms are strong here, that B2 **fires**, and that the honest product
of this task is the registered admissibility finding, not a cleared venue.

**What would surprise me (and is the outcome the Hub is priced for):** exemplar arms in the 55–68 %
band while ARF is ~77 %. That would mean the 33-D input metric is *not* informative enough locally even
within a temperature band, i.e. the class-conditional overlap (36 % pooled / 23 % per-regime) bites the
1-NN neighbourhood harder than it bites an ensemble's axis-aligned splits.

## 4. Registered NOT-RUNs for this task (never to be reported as nulls)

`out-of-control` (905 145 × 24 classes) baselines — fetched and frozen if the USP link resolves, but
**no baseline arm is run on it in this task**; it is C2W10's V3 null, not a gate leg ·
imbalanced-reoccurring (452 044) · Metro Interstate drift map · any CLU cell of any kind ·
any decimated (`m > 1`) baseline compared against Souza's published numbers (task §5b: the published
comparison is **m = 1 only**; decimated numbers are an internal comparison).
