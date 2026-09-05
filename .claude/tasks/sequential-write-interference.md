# Task B: sequential-write-interference — turn the wave's worst negative into a measured claim (w21)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/sequential-write-interference.md` · **Branch:** `agent/experiment-engineer/sequential-write-interference`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/learned-landscape-write-read.md` §3 (the interference measurement) · `.claude/outputs/clu-controller-spec.md` §C3/§C5 + §4 (MVC-0, the spacing gate) · `.claude/outputs/primitive-harness.md` (the drop-in slot + fairness protocol — **reuse it, do not rebuild**)

## Why — the Hub's adjudication, now a test
w20's worst result (write A, write B, A is destroyed: strict **1.000 → 0.000**) was measured with **UNGATED** writes — "continue training", no spacing rule. The theorist independently measured the same contrast **with an admission gate**:

| | ungated | **gated** (`d ≥ d_safe = 4.4s`) |
|---|---|---|
| max fixed-point drift per write | **8.39** | **8.0e-5** |
| selectivity | collapses to **0.35** | **1.000 at every K** |

⇒ **The claim under test: "unconstrained writes destroy stored items; an admission gate suppresses it by ~5 orders" — i.e. locality is a certificate the controller CHECKS, not a property learning provides.** If it holds, our worst negative becomes a statement about *everyone else's* primitive, since MLPs/transformers/RNNs all write to latent state without any such gate.

## Item 1 — the gate, on the engineer's own failing setup
Take the **exact** `learned-landscape-write-read` §3 protocol (write A on a K-site ring, then write B, re-read A) and add the MVC-0 admission rule:
- **spacing gate:** admit a new site iff `min_j d(q_new, q_j) ≥ d_safe`; **refuse-and-relocate** if it fails (theorist A2: refusal is a *correct* controller output — 7/20 refused at zero accuracy cost).
- **the C3 admissibility check** on stored items: `‖H_i⁻¹∇δV(q*_i)‖ ≤ δ_budget` — the controller can *compute* this as a dot product even when it cannot guarantee it.

Report corruption and strict-A-after-B, **gated vs ungated**, ≥5 seeds. **Reference points: designed+ungated 0.000; learned+ungated 2.9e-2…5.0e-1; theorist's designed+gated 8.0e-5.**
⚠ Blank control over the strongest read on every cell (w20 method finding). ⚠ Value-recovery metric, not classification.

## Item 2 — the sequential-write curve (the deliverable)
Write **K = 1…16 items one at a time**, and after each write re-read **all** previously stored items. Deliverable: **retention of item 1 vs number of subsequent writes**, plus mean retention over all stored items.
Arms: **CLU designed+gated · CLU learned+gated · CLU learned+ungated**. This is catastrophic forgetting in memory framing, and it is the axis on which the gate should be visible at a glance.

## Item 3 — ⭐ the cross-primitive comparison (what makes it a claim rather than an ablation)
Run the same sequential-write protocol against **transformer, GRU, MLP** in the `primitive-harness` drop-in slot at **matched parameters**, using its **equal LR grid + symmetric rescue** protocol verbatim. Task: store K key→value pairs by sequential gradient updates (one item at a time), then probe retention of all of them.

**Report retention-vs-K per primitive, never averaged.** Two secondary measurements, both cheap and both directly on the Head's hypothesis:
- **compute-to-criterion** — steps for item K to reach threshold *without* dropping items 1..K−1 below threshold. (The Head's "wasted compute reorganizing to conserve key info" — ⚠ currently a **hypothesis**, and this is the measurement that would evidence it. If the compute cost is flat across primitives, say so; that would refute it.)
- **retrieval cost scaling in K** — CLU's rollout is O(steps) and, for a *parametric* landscape, independent of K; attention is O(K) per query. Measure it, don't assert it.

## Item 4 — the honest scope statement (required, not optional)
⚠ **CLU's writes here are PARAMETRIC** (into `V_θ`, during training) while **attention's memory is CONTEXTUAL** (a KV cache, written at inference). These are different capabilities and a referee will say so. **State the distinction explicitly in the report**, make clear which capability each arm exercises, and **do not claim a contextual-memory win from a parametric-memory experiment.** The defensible comparison is against other *parametric* stores (MLP/FFN) and against fixed-state recurrences (GRU); the transformer arm is context, reported for completeness with the caveat attached.

## Acceptance
Gated-vs-ungated on the exact w20 failing setup, the sequential-write retention curve, the cross-primitive comparison with compute-to-criterion and retrieval-cost scaling, and the parametric/contextual scope statement. ≥5 seeds throughout. Tests green.

⚠ **Pre-register the predicted gated corruption and the predicted crossover K before running.** ⚠ **If the gate does NOT rescue the engineer's setup, that is the most important possible outcome of this wave** — it would mean the theorist's 8.0e-5 is a property of designed landscapes only, and MVC-0's central mechanism does not transfer. Report it loudly.
