# PREREG — phi-stream-discipline (w24)

**Written BEFORE the MNIST harness was run** (protocol §5 pre-registration rule). Code was
written and smoke-tested on synthetic labelled blobs only (`tests/test_phi_stream.py`, 10
passed); **no MNIST number had been produced when this file was committed.**

Harness: `chlu/experiments/exp_phi_stream.py` at branch
`agent/experiment-engineer/phi-stream-discipline`.

## The measured quantities

Split-MNIST-shaped stream, 5 tasks × 2 classes, `items_per_task=32` ⇒ **M=160** wells at the
end of the stream. Queries = 50 %-dropout-masked stored items (repo-verbatim), **identical
across regimes/arms/positions**. Designed Gaussian store, `s = 0.3·median-NN(φ keys)`,
`phi_dim=32`, seeds {0,1,2}.

- `acc(regime, arm, line, τ)` — accuracy on task index τ **at the end of the stream**
  (`identity_acc` = payload index correct; `class_acc` = payload's class label correct).
- **gap(τ) = acc(generic_frozen, τ) − acc(task1_only, τ)** = the **cost of strictness**.
- `slope_per_task_index` = OLS slope of gap(τ) vs τ.
- laundering margin = `acc(CLU-in-φ) − acc(kNN-in-φ)`, per regime (N89 control).

## Predictions (point estimate + interval; identity_acc, PCA arm, CLU-in-φ unless stated)

| # | prediction | value | derivation |
|---|---|---|---|
| **P1** | gap at the LAST task index > 0 (strictness costs accuracy) | **+0.10**, in [0.03, 0.20] | A PCA basis fit on {0,1} still spans most low-frequency MNIST ink-mass variance, so later digits are not destroyed — but their *discriminative* directions are under-represented, and the masked query lowers SNR further. |
| **P2** | gap at task index 1 ≈ 0 | **0.00 ± 0.02** | Task 1 is in-distribution for BOTH regimes; the two φ's differ only in what else they saw. |
| **P3** | gap GROWS with task index (slope > 0) | **+0.025/task**, in [0.005, 0.06] | Later tasks are progressively further from the {0,1} subspace, and the store is simultaneously getting fuller (M grows), so the two effects compound. |
| **P4** | ⭐ **decision rule for w25 (registered in advance):** `gap_end_of_stream` (mean over seen tasks, end of stream) | **< 0.10 ⇒ task-1-only φ is VIABLE as primary; 0.10–0.20 ⇒ viable with a declared caveat; > 0.20 ⇒ the entry NEEDS online φ** | Set before seeing data so the verdict cannot be back-fitted. |
| **P5** | absolute level, generic_frozen (the declared upper bound), CLU-in-φ | **≈0.85**, in [0.75, 0.95] | w23 `phi-read-in` measured 0.891 at M=128 MNIST/PCA; M=160 is slightly harder. |
| **P6** | absolute level, generic_frozen, kNN-in-φ | **≈1.00**, in [0.95, 1.00] | w23 measured 1.000 at M=128 and 0.996 at M=256. |
| **P7** | ⭐ **laundering (N89) FIRES in BOTH regimes** — CLU-in-φ never beats kNN-in-φ outside the 0.03 tie band | **fires in 2/2 regimes**, max CLU margin ≤ 0.03 | N89 fired on all 4 cells in w23 with max CLU margin 0.000. Strictness changes φ, not the store-vs-kNN relationship. |
| **P8** | **WATCH-ITEM (Item 4), both hypotheses registered.** delta = margin(task1_only) − margin(generic_frozen) | **H0 (registered as the expected outcome): delta ≈ 0, \|delta\| < 0.05, NO store advantage.** H1 (the competing hypothesis, registered so a hit is evidence and not a hunt): delta > +0.05 **and** laundering does NOT fire in the strict regime ⇒ first evidence of a store advantage — a finite-width well tolerates address noise that kNN's hard argmin does not. | H0 rationale: kNN and CLU read the SAME φ geometry; degrading φ degrades both. I do not go looking for H1; it is reported if and only if it appears. |
| **P9** | ⭐ **AE arm pays a LARGER cost of strictness than the PCA arm** | gap(AE) − gap(PCA) > 0 at the last task index | A nonlinear encoder fit on 3000 images of {0,1} only extrapolates badly off its training manifold; a linear projection degrades gracefully. Directly relevant to which φ family w25 should use. |
| **P10** | `class_acc` gap < `identity_acc` gap | strictly smaller at the last task index | Getting the right *class* is a coarser demand than getting the right *item*; a degraded φ loses item identity before it loses class neighbourhood. |

## What would falsify the entry's premise

If **P4** lands in the `> 0.20` band, the honest report is: *task-1-only φ cannot represent
later classes; the w25 Class-IL entry cannot use a frozen task-1 φ as its primary arm and
must either (a) build online φ first, or (b) declare a generic pretrained φ and defend it as
a legitimate "pretrained feature extractor" setting (the L2P/DualPrompt precedent) rather
than as a from-scratch entry.*

## Scope caveat carried into every number (Head, binding)

Masked/static retrieval is a task where **equalling a simple baseline is our best case**,
because CLU approximates the nearest-neighbour method that wins it. Every number here is
**diagnostic of φ's stream discipline**, not a competitive claim. No tie with kNN is a win;
masked recall is permanently appendix-only.
