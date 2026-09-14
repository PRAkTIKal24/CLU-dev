# PREREG — `c2w8-close-gate-hardening`

Filed **before** any harness in this spoke ran. Base `main @ 9e0bb25`, branch
`c2w8-close-gate-hardening`, worktree `../CHLU-c2w8close`.

⛔ This spoke is **instrument repair**. It registers **declared thresholds** and **one
before/after measurement**, not a claim. No dial, no performance number, no verdict.

---

## 1. Item (i) — the drift FLOOR, declared before it is coded

**Rule (registered):** the repaired drift leg is TWO-SIDED and MECHANICS-labelled:

    PASS  iff  floor <= median(site_drift) < ceiling
    ceiling = ceil_frac  x codebook_spacing        (ceil_frac = 1.0, the pass-2 rule, unchanged)
    floor   = floor_frac x codebook_spacing        (floor_frac = 0.01, NEW)

Both bounds are fractions of a **measured** quantity — the live store's own median
nearest-neighbour **codebook** spacing (⛔ not the ~200-key sizing spacing; that
substitution is item (v)/(vi.6)'s defect). Neither bound is an absolute constant.

**Derivation of `floor_frac = 0.01` (declared, not tuned).** The banked D2a signature is
the strong-φ CIFAR arm: settle equals the same-keys kNN to **±0.0007** against a measured
codebook spacing of order **0.14** ⇒ a configuration already adjudicated table-expressible
sits at **≈0.005 × spacing**. The floor is set at **0.01 × spacing** = **2× above the
measured table-expressible point**, i.e. the smallest round decade that (a) is strictly
above the measured D2a operating point and (b) leaves a factor-2 margin so the rule is not
knife-edge on it. ⛔ It is not tuned against any arm's pass/fail.

**Registered predictions (falsifiable):**
- P1: a planted store whose atoms sit exactly at their own sites relaxes with
  `median(site_drift) / codebook_spacing < 0.01` ⇒ **FAILS the repaired leg on the FLOOR**
  while **passing** the old one-sided pass-2 rule. (This is the mandatory designed negative.)
- P2: the banked arm-A pass-3 G-DRIFT ratios (0.0071 … 2.014 across the 9 cells) are
  *not* all inside the two-sided band ⇒ the repaired leg changes at least one banked
  cell's leg verdict. ⛔ Recorded as a mechanical consequence, **never re-scored into a
  claim** (§A32.1: no pass 4).

## 2. Item (iii) — the repaired scale guard

**Rule (registered):** the legal rescale is **FULL-STATE co-scaling (address AND payload)**
(§A31.6, Head-ratified). The guard's **pass condition is VERDICT STABILITY** — every leg
boolean identical across the declared scales. Metric movement is reported as a
**DIAGNOSTIC** and is **not** a pass condition.

- P3: fed the banked pair `A1 = 0.24219 → 0.28906` (Δ = **+0.04688**, inside the old 0.05
  metric bound) with the registered threshold 0.25, the repaired guard returns
  `metric_bounded = True` **and** `verdict_stable = False` ⇒ **guard FAILS**. The old
  guard passed this exact pair. (Regression test.)

## 3. Items (v) + (vi.6) — one defect, one fix, before/after stated

Both are the ~200-key **sizing** spacing standing in for the ~16-item **store** spacing.
- P4: at the census's operating point the store-population NN spacing is **strictly
  larger** than the 200-key sizing spacing (NN spacing grows as n falls), so
  `d_safe_store > d_safe_sizing` and the admission gate's refusal rate can only **rise or
  stay equal**, never fall. Registered direction; magnitude NOT predicted.
- P5: on the `--quick` census (the only cell this spoke runs, a regression cell and NOT a
  result), the refusal rate under the repaired `d_safe` is **≥** the legacy one.
  ⚠ **The refusal rate is REPORTED, never tuned to a target** (task §v).

## 4. Items NOT pre-registered because they carry no measured number

(ii) margin-in-SE emission · (iv) launch/settle split · (vi.1)–(vi.5) housekeeping.
Each ships with a pytest, and (i)/(iv) ship with the two mandatory designed negatives.

## 5. Declared NOT-RUNs (never nulls)

- No re-run of any banked arm (arm A / arm B / the spine's 9 cells). Consequences of the
  repairs on banked numbers are stated as *mechanical consequences*, never re-scored.
- No merge/prune/restoration verb (deferred).
- No pass 4, no daylight chase, no arm-race adjudication, no tier-ii/full-CLU verdict.
