# Task: class-count-scaling — the image-domain instance of CLU's one favourable result (w22)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/class-count-scaling.md` · **Branch:** `agent/experiment-engineer/class-count-scaling`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/primitive-harness.md` §2 Family 1b (**the kv capacity sweep this transplants**) and §4 (the fairness protocol — **reuse verbatim**) · `.claude/outputs/address-space-dimension-scaling.md` §1 · `.claude/outputs/clu-controller-spec.md` §C2 (the gauge theorem)
- ⚠ **Sequencing:** run after `gamma-read-sweep` and `potential-function-class` — both decide settings this task would otherwise guess.

## Why — and the two arguments that make this worth doing
**1. ⭐ In supervised classification the LABELS SUPPLY THE ASSIGNMENT.** Assignment — which item goes in which basin — is the thing **T3 proves no regularizer can choose**, that gradient search cannot do (4 protocols, ≈chance), and that MVC-0 currently hand-codes. **Labels hand it to us for free.** So classification tests the memory machinery (write, address, retrieve, isolation, capacity) **without first solving the program's hardest unsolved problem.** That decoupling is the main reason to run this.

**2. Head's three-way positioning, which engages the strongest competitor.** Unconstrained latents (vanilla MLP/CNN/transformer) have no reliable structure. **Contrastively-trained encoders are the middle ground: they impose a *metric* structure (similar → near) but not a *discrete addressable* one** — no basins, no barriers, no isolation. Global semantics, no local semantics. CLU-designed claims the local structure. **That is the gap the paper would claim, and the contrastive arm is the baseline that makes the claim non-trivial.**

## ⛔ SCOPE — what this task is NOT
**It is NOT "beat SOTA on CIFAR-10".** We would lose, and single-dataset top-line accuracy tests discrimination, not memory. **The claim is about the SLOPE, not the intercept.**

**The measurement is: accuracy vs NUMBER OF CLASSES at fixed representational capacity.** This is the image-domain transplant of the one axis where CLU beat a standard primitive at matched parameters — `primitive-harness` Family 1b: **GRU 0.997 → 0.008 (chance) for kv 2→16 while CLU held 0.525 → 0.154**, crossing above at kv≈8. Many-class / long-tail regimes are exactly where softmax heads and contrastive embeddings degrade.

## Item 1 — the sweep (the deliverable)
Frozen-encoder + **linear probe** protocol (the standard SSL evaluation, and the same contract as HEPA/CAFE). Sweep **class count ∈ {10, 100, 1000}** (add a larger cell if compute allows; state where you stopped and why).
Arms at **matched parameters and matched tuning budget**:
1. **CLU** (designed/gated landscape, one basin per class — labels give the assignment)
2. **contrastively-trained encoder** (the real competitor — the metric-structure middle ground)
3. **standard supervised encoder** (unconstrained latent)
4. **control** — untrained/random encoder + linear probe (the floor; `primitive-harness` showed a no-mixing control is what makes "at the floor" legible instead of "got 0.18")

**Deliverable: accuracy vs class count, four curves, never averaged.** Report the **slope**, and the crossover class-count if one exists. ⚠ **Report absolute accuracy honestly alongside** — if CLU's intercept is far below the baselines, a favourable slope does not rescue it, and the report must say so.

## Item 2 — is the mechanism the one we claim?
A favourable slope is only interesting if it comes from the physics. Measure:
- **Prop 2 in classification form:** are confusions **sub-barrier** (a query cannot reach a foreign class's basin)? Report the confusion matrix against the barrier structure. **This is Prop 2's honest test — note it was FALSIFIED as a recall-vs-attention advantage in `primitive-harness`, so it is on probation and must earn its place here.**
- **basin occupancy:** does each class actually get its own basin, or do classes merge? (N1 violation ⇒ the merged pair decodes at chance — theorist measured 0.508.)
- **capacity headroom:** with `d` the address dimension, `4·2^d` is the designed spatial bound. Is the class count anywhere near it, or is the binding constraint elsewhere?

## Item 3 — the honest baseline obligation
Tune every arm at least as hard as CLU using the **symmetric, monotone LR-rescue pass** from `primitive-harness` §4, and report the tuning budget per arm. ⚠ **The contrastive baseline must be a real one** — a weak SSL baseline is worse than none, because it produces a number we later retract. If a credible contrastive baseline is out of compute reach, **say so and report the arm as absent rather than shipping a weak one.**

## Acceptance
The four-curve accuracy-vs-class-count figure with slopes and absolute values, the mechanism checks (Item 2), and the per-arm tuning budgets. Tests green.

⚠ **Pre-register the predicted slopes and any crossover before running.** ⚠ **This would be CLU's fourth benchmark family; three of the first three were losses.** A fourth loss is an acceptable and informative outcome — **report it plainly** — but do not soften it, and do not add CLU-only capacity to avoid it. **If the slope claim fails, the "graceful degradation under item load" story loses its only cross-domain support, and the program needs to know that immediately.**
