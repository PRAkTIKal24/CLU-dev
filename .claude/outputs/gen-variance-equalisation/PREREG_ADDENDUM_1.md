# PREREG ADDENDUM 1 — `gen-variance-equalisation`
⛔ **A NEW FILE.** `PREREG.md` is **not edited** (R34's seventh strike). Filed **after** the `--quick`
plumbing smoke and **BEFORE** the production run. ⛔ No `--quick` number is reported anywhere.

## What changed, and why it is a CONTROL and not a treatment
The plumbing smoke exposed a **confound in the trainer, not in the metric**:
`chlu.core.gen_store.train_store` anneals a softmax over `−d²/(2τ²)` with **`τ` in absolute distance units**
(`tau_start=1.0 → tau_end=0.1`), and Adam's step size is absolute too. Per-well PCA whitening changes a well's
overall scale by ≈15× at the shipped cell (smoke: the store-only loss reads ≈46 unwhitened vs ≈695 whitened).
⇒ **an unmatched whitened arm would see a different anneal and a different effective learning rate, and would
lose (or win) for a reason that has nothing to do with the metric.** ⛔ That is a confound, not a result.

**The remedy, added to `chlu/core/gen_whiten.py::WellTransform.fit` as `scale_match=True` (config
`GenVarianceEqConfig.scale_match_to_raw`, default `True`):** after the transform, multiply each well's block by a
**single scalar** `α_k = sqrt( E‖x_raw‖² / E‖z‖² )`.

⭐ **Why this is metric-neutral, and therefore a matched-compute control rather than a second treatment:**
Euclidean k-means is **equivariant under a uniform scaling** — cluster assignments of `αZ` with centres `αC` are
**identical** to those of `Z` with `C`. The scalar therefore changes the *units the trainer's `τ` and learning rate
are measured in*, and ⛔ **nothing about which points the metric says are close.**
It is folded into `A_k` and `A_k⁻¹`, so **exact invertibility and the byte ledger are unchanged**
(`test_byte_ledger_reproduces_the_hub_arithmetic` asserts the second, `test_scale_matching_is_METRIC_NEUTRAL…`
the first, by checking the co-clustering matrices are equal).
For the `identity` control `α_k ≡ 1.0` **exactly**, so the control stays bit-identical to the shipped path.

## ⛔ Nothing in `PREREG.md` §3–§9 is revised
The premise test, the stop rule, both falsifiers, every predicted value and both byte prices stand **unchanged**.

## ⭐ The sensitivity leg this buys, pre-registered here
`scale_match_to_raw=False` remains reachable (`--no-scale-match`). If it is run, it is reported as an explicitly
labelled **confound measurement** — ⛔ never as the arm.
**Prediction, committed now:** the unmatched `pca_whiten` arm scores **WORSE** than the matched one on
TOTAL-RETAINED (80 % interval: unmatched total is 0.2–1.5 lower), because it is under-trained at the shipped
`τ` schedule, not because its metric is worse.
