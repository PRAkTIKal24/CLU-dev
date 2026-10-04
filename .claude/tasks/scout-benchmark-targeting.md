# Task: scout-benchmark-targeting — which popular benchmarks actually match what CLU is? (w21)

- **Agent:** `web-scout` · **Output:** `.claude/outputs/scout-benchmark-targeting.md` · read-only, no repo edits
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/primitive-harness.md` (the 0-of-3 result and its protocol) · `.claude/outputs/address-space-dimension-scaling.md` §1 (the capacity law) · `.claude/handover_context.md` §10, the `2026-07-21 (night)` entry

## Why — we may be losing on benchmarks that test a capability we did not build
**Head's standing concern:** *"we need to show how this transfers to performance on existing popular benchmarks. We are really failing in this performance eval direction and we need to fix this for an impactful ICLR/NMI submission."*

**The Hub's diagnosis, which this task must check rather than assume.** The program's vision writes to memory **during training** — capacity in `V_θ`, addressed by a small pointer. That is **parametric memory** (like an FFN). But `primitive-harness` benchmarked CLU on **MQAR**, which is **in-context recall** — items presented at inference and retrieved from the same sequence, i.e. exactly what a **KV cache** is for. Attention was flat (0.996, drop 0.001) because distractors cost an O(T) cache nothing. **We may have set up a comparison we could not win, and then recorded the loss as a property of the primitive.**

## Item 1 — is the parametric/contextual framing standard, and where is the line?
Establish how the literature actually carves this (parametric vs contextual/working memory; the "knowledge in weights vs knowledge in context" line). **Is this a recognized distinction a reviewer will accept, or is the Hub reifying a folk taxonomy?** ⚠ **Say so plainly if the latter.** Give the canonical citations.

## Item 2 — ⭐ the capacity-per-parameter comparison (the highest-value item)
There is a known line of work measuring **how many bits of knowledge a transformer stores per parameter** (the Hub recalls a figure around 2 bits/param from the *Physics of Language Models* line, **but does not trust the recall and will not quote it unverified**).
- **Find the actual result, its exact constant, its measurement protocol, and its scope conditions.**
- Determine whether our measured law — **`K_max = 4·2^d` items in a designed landscape** — can be put on the same axis, and **exactly what we would have to measure** for the comparison to be legitimate (bits per item? parameters counted how? is a designed landscape even admissible as a comparison point?).
- ⚠ **If the comparison is not legitimate, say so and explain why** — that is more valuable than a flattering number we would later retract.

## Item 3 — the benchmark shortlist
For each capability where CLU has **measured** evidence, name the standard benchmarks, the current SOTA, the weight class, and the honest difficulty of a submission-grade result:
1. **Capacity under item load** — our one favourable measurement (GRU → chance at kv=16, CLU retains 19×, matched params).
2. **Sequential/continual writing without catastrophic forgetting** — the admission gate's natural axis (gated drift 8.0e-5 vs ungated 8.39). What are the standard continual-learning benchmarks and what would a credible entry look like?
3. **Long-horizon retention** — CHLU's founding claim, never tested on a public benchmark. Where does LRA / needle-in-a-haystack / long-context retrieval sit, and are those contextual (i.e. wrong for us)?
4. **Reconstruction at matched latent dimension** — the `clu-autoencoder` axis. What is the standard AE/VAE comparison protocol and what baselines are expected?

**Rank the shortlist by (winnability × reviewer legibility), and state which are ruled out.**

## Item 4 — the retrieval-cost claim
The Hub asserts: **CLU's retrieval is O(steps) and independent of the number of stored items** (they live in `V_θ`), whereas **attention is O(K) per query**. ⚠ Note the engineer measured cost scaling *linearly in K* — but with a potential that explicitly **sums over K wells**, which a *parametric* landscape would not. **Is the asymptotic claim already made (and answered) in the literature?** Check it against the sub-quadratic-attention and associative-memory lines before we build on it.

## Item 5 — prior-art debts still open (blocking, from the w19/w20 registry)
Close as many as the budget allows, flagging any left open: **Ramsauer's exact capacity constant** (never obtained; exponent form is single-sourced secondary — **must not enter a paper**) · **the SRNN initial-state-optimization passage** (flagged as the most likely preemption of any learned-address novelty claim) · **UnICORNN's Hamiltonian claim** (unverified).

## Acceptance
A cited, verified brief: the parametric/contextual verdict, the capacity-per-parameter comparison with its legitimacy assessment, the ranked benchmark shortlist with SOTA and weight classes, the retrieval-cost check, and the prior-art debts closed-or-flagged.

⚠ **Separate what you verified from a primary source, from a secondary source, from what you could not confirm** — the w19 scout's single-sourced exponent is the standing lesson. ⚠ **A finding that "there is no benchmark where CLU's measured capability is competitive" is a legitimate and important outcome.** Do not manufacture a target.
