# PREREG ADDENDUM — headroom-retry-benchmark (w24), gate iteration 2

**Written BEFORE running gate iteration 2** (i.e. before `block_rescale=False`, before `crowd_rho<1`,
and before the norm-corrected `packing_slack` was ever evaluated on data). Invoked under PREREG §5
("iterate levels **once**, on the gate only") — with two protocol amendments declared here, not later.

## 0. What gate iteration 1 measured (seed 0, full `clu_steps=150`, `.claude/outputs/.../gate_seed0.json`)

| regime | M | level | first-pass | NN floor | verdict |
|---|---|---|---|---|---|
| iid:block (rescaled) | 128 | f=0.4 | 0.086 | 0.383 | fail (first ≪ band) |
| iid:block (rescaled) | 128 | f=0.5/0.6/0.7 | 0.023/0.016/0.008 | 0.172/0.086/0.062 | fail |
| iid:block (rescaled) | 256 | f=0.4…0.7 | 0.055…0.004 | 0.383…0.039 | fail |
| crowded:mask | 128/256 | p=0.2/0.3 | 0.88–0.99 | **1.000** | fail (NN at ceiling) |
| crowded:mask | 128/256 | p=0.5 | 0.31/0.35 | **0.992/1.000** | fail (NN at ceiling) |

**0 / 14 cells passed.** Two diagnosed causes, each with an amendment:

- **C1 — the inherited `1/(1-f)` survivor rescaling turns *correlated* erasure into DESTRUCTION.**
  Under iid dropout the rescaling is unbiased *on average over the whole image*; under a contiguous
  block it multiplies a surviving **crop** by 2–3.3× and throws the query far outside every basin —
  the exact σ≥0.4-cliff failure the task forbids as a headroom source. **Amendment A:** add
  `block_rescale` (default **True** = inherited convention, nothing shipped changes) and gate the
  `False` arm too. `False` = plain occlusion, i.e. the task's Item-1.3 *partial-key* reading:
  the query carries a strict subset of the address dimensions, unamplified.
- **C2 — the crowding lever is NEUTRALISED by the store-adaptive well width, and the slack metric
  was degenerate.** `s = 0.3·median-NN` is recomputed per store, so a tighter cluster gets
  proportionally tighter wells: measured slack was **1.075 in all 14 cells** — and
  `1/(3.1·0.3) = 1.0753` exactly, i.e. the number was a tautology, not a measurement, whenever
  `s ≥ σ_q`. It was pinned at `s` because `σ_q` was coded as a **per-element** RMS while `median_NN`
  is a **vector** distance (a factor √D = 28 unit mismatch). **Amendment B(i):** `packing_slack`
  uses the displacement **norm** `σ_q = RMS_i‖q_i − ξ_i‖`, same units as `median_NN`
  (this matches `exp_dim_scaling`'s `delta_req_sqrtd` treatment). **Amendment B(ii):** add
  `crowd_rho` — contract the store about its centroid, `ξ' = c + ρ(ξ − c)` (default **1.0** =
  current behaviour). This shrinks `median_NN` by ρ while leaving the erasure displacement
  (∝ ‖ξ'‖ ≈ ‖c‖) untouched, so `σ_q/median_NN` grows by 1/ρ — geometry-sourced ambiguity with a
  single monotone knob, which the NN-cluster construction failed to deliver.

## 1. Pre-registered numbers for gate iteration 2

| # | quantity | registered value |
|---|---|---|
| **A1** | corrected slack, w23 iid store, mask p=0.5, M=128 | **0.25 ± 0.15** (i.e. w23 already ran **past** the bound; the "1.08" is retracted as an artifact) |
| **A2** | `iid:block`, rescale **True**, best cell over f∈{0.1,0.2,0.3} | first-pass reaches the band at **f≈0.2** with NN floor **0.90 ± 0.08** ⇒ **gate PASSES with ~40 % probability** (NN is the binding half) |
| **A3** | `iid:block`, rescale **False**, f=0.5 | first-pass **0.75 ± 0.15**, NN floor **0.97 ± 0.03** (still near ceiling ⇒ likely fail) |
| **A4** | `iid:block`, rescale **False**, f=0.7 | first-pass **0.45 ± 0.20**, NN floor **0.85 ± 0.10** ⇒ **gate PASSES with ~50 % probability** |
| **A5** | `crowded:mask` at **ρ=0.25**, p=0.3, M=128 | slack **< 0.3**; NN floor **0.70 ± 0.20**; first-pass **0.40 ± 0.25** ⇒ PASS with ~45 % probability |
| **A6** | ≥1 cell passes BOTH halves somewhere in iteration 2 | **YES (~75 %)** — but I register that it will most likely be a `block_rescale=False` or `crowd_rho≤0.25` cell, i.e. one of the two amendments, not the as-registered R-BLOCK/R-CROWD |
| **A7** | monotonicity guard | NN floor is **monotone decreasing** in f (rescale=False) and in 1/ρ; if it is not, the lever is not doing what it claims and I report the regime as failed |

## 2. Item-4 verdict — the prediction is UNCHANGED from `PREREG.md` §3

H-A (NO — the NN floor still dominates at every ambiguity level) remains my primary hypothesis at
**~70 %**, for the unchanged reason: NN's `argmin_i‖q−ξ_i‖` and the CLU settle read the *same*
Euclidean evidence, and the boost's aim point *is* the ambiguous query, so ambiguity removes
information from both rules together. Nothing in Amendment A or B changes that argument.

**New fairness rider (registered now so it cannot be added post-hoc as an excuse).** Under
`block_rescale=False` the zeroed dimensions are *missing*, not *observed-as-zero*, so the harness's
full-vector NN is **not** the ML-optimal rule there. If, and only if, `clu_gated` beats
`feedforward_nn` in a `block_rescale=False` cell, that win is **not reportable as a benchmark win**
until it also survives an **observed-dimensions-only NN** (`feedforward_nn_masked`, distances over
the surviving coordinates = the true ML rule under erasure). I commit to adding that 7th line in
that event, and I predict it **restores NN dominance (~80 %)**.
