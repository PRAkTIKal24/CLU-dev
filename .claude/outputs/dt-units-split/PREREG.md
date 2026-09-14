# PREREG — dt-units-split (w20)

Written **before** running the Item-3 harness. Commit at time of writing: working tree on
`agent/experiment-engineer/dt-units-split` off `089cc6e`, implementation complete, no Item-3
numbers measured yet. (Item 2's init-only curvature/drift scan WAS run first — it was needed to
choose the integrator default that Item 3 is measured at, and it is reported as-is.)

## What is being predicted
The w19 audit (`clu-latent-io-audit`) measured its `dt=1.0` column **at init, on a single batch,
with `dt` conflated** (i.e. it changed the momentum scale and the integrator step together). Item 3
asks for the same quantities **after the split and after 150 epochs of training**. Those are two
different experiments, so w19's numbers are predictions, not observations, for my setting.

## Derivations

`p = Δq/data_dt` ⇒ `K ∝ data_dt⁻²`. Going 0.05 → 1.0 divides K by 400. `V` is unchanged
(it is a function of `q` only). So every prediction below follows from "K falls 400×, V holds".

| # | quantity | w19 @ conflated dt=0.05 | **my prediction @ split (data_dt=1.0, dt=1.0)** | derivation |
|---|---|---|---|---|
| 1 | `E_reg` share of loss (init) | 99.2% | **< 15%** (point est. 7.6%) | `E_reg = ereg·E[H²]`, `H≈K` ⇒ falls ~400²… but `predict_mse` also falls, so I take w19's directly-measured 7.6% and allow slack for the trained regime |
| 2 | `E_reg` share of mass gradient (init) | 99.8% | **< 50%** (point est. 30.6%) | same; w19 measured 30.6% at init |
| 3 | `corr(H,K)` after training | 0.999996 | **< 0.90**, and `corr(H,V)` **> 0.30** | K falls 400×, V unchanged ⇒ V stops being a rounding error in H |
| 4 | ballistic fraction of the wake rollout | 98.3% free-streaming | **force/free ratio > 0.5**, i.e. ballistic fraction **< 67%** | w19 measured force/free 0.9712 at dt=1.0, γ=0, 16 |
| 5 | R-1 relative gradient (`mass_spread_lambda`) | 1.6e-6 (invisible) | **> 1e-3**, and λ∈{0,1,50} give **non-identical** h-AUROC | w19 measured 0.071 at dt=1.0 |
| 6a | mass differential:common ratio | 39:1 | **< 10:1** | the 39:1 common mode was *driven by* `E_reg` (99.8% of the mass gradient); removing that dominance should remove the common-mode runaway |
| 6b | final `M_max/M_min` | 1.153 (**below** init 1.265) | **> 1.265** (i.e. at or above init) | with softplus out of its linear regime and no common-mode runaway, training should no longer *compress* the spectrum |

## The gate
**Item 3.4/3.6 is the SDR gate.** SDR comes off hold iff prediction 5 holds (R-1 is a live lever)
**and** 6b holds (training stops compressing the mass spectrum). I am pre-committing: if
`M_max/M_min` still ends **below** its 1.265 init, the split did **not** fix the degeneracy and SDR
stays on hold regardless of what FD001 does.

## Explicitly NOT predicted
**FD001 h-AUROC.** Per the task's final line this is a correctness fix and the FD001 delta is a
measurement, not a target. I pre-commit to reporting it whatever it is, including worse than
w19's 0.6540 / 0.7092. I note in advance one reason it *could* get worse without the fix being
wrong: the default relax budget `γ·steps·dt` rescales from 0.16 to **3.2** at `dt_eff=1.0`,
overshooting the ~1.6 that w19 measured as best. Any h-AUROC change is therefore confounded with
a budget change unless the budget is held fixed — I will report both.

## Falsification
If predictions 1–4 fail, the "dt units are the root cause" story is wrong and the w19 item-4
verdict needs revisiting. If 1–4 hold but 5–6 fail, the units bug was real but was **not** what was
suppressing the mass spectrum — a different mechanism owns that, and the SDR hold stands.
