# Task: headroom-retry-benchmark — a regime where the retry curve can actually win (w24)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/headroom-retry-benchmark.md` · **Branch:** `agent/experiment-engineer/headroom-retry-benchmark`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (**§3.2 worktrees mandatory**) · `.claude/outputs/retry-compute-study.md` (**the harness you extend — reuse it; §6 is the RUD-C spec**) · `.claude/negative_results.md` **N90** (tier A) · `.claude/claims_matrix.md` v2.2 **CM-23(b) as split**
- **This is the gate on R3** ("the anytime read") in the Head's result set.

## Why
w23 proved the **mechanism** (random-kick and ensemble-of-k are dead flat in all 8 cells; the directed boost is real) but could not win the **benchmark**: on masked-pixel MNIST the trivial NN floor sits at **0.99–1.00** and beats gated retry in every cell (−3.5…−42.2 pp). That is a **headroom** problem, not a mechanism problem — there is nothing for extra compute to buy when the baseline is already at ceiling. **[HEAD RULING] Build the headroom benchmark:** a regime where all methods land ~**0.6–0.7** and the boosted retry buys a visible margin at stated extra compute.

## ⭐ The design constraint (advisor, derived from the w23 physics — this is the crux)
The failure mode must be **AMBIGUITY**, not **DESTRUCTION**.
- The boost aims from a wrong well toward the (partially observed) query. Under **structured erasure** the surviving evidence pulls toward the true basin ⇒ retry recovers strongly (measured +36…+76 pp on mask).
- Under **full-field Gaussian noise** the query leaves every basin (the σ≥0.4 cliff) and **no retry can recover it** — every arm sits at chance (measured: +0 lift at σ=0.2 ceiling cells, and total collapse past the cliff).
⇒ **Do not build the headroom out of noise.** Build it out of ambiguity.

## Item 1 — build ≥2 ambiguity regimes (pick from, or justify alternatives)
1. **Structured / correlated occlusion** — contiguous block or quadrant masks (NOT iid `torch.dropout`), so the surviving evidence is *consistent with several stored items* rather than uniquely identifying one. This is the direct fix for "surviving pixels uniquely identify the pattern," which is *why* the NN floor wins today.
2. **Crowded-store retrieval** — `M` pushed to/past the packing bound so basins genuinely overlap (`Δ_req ≈ 3.1·max(w,σ_q)`; w23 φ ran at slack ≡ 1.08, i.e. **no slack**). Ambiguity from geometry rather than from the query. ⭐ **This one is free inside the w25 CL entry** — prefer it if you must choose.
3. **Partial-key retrieval** — query carries a strict subset of the address dimensions.

## Item 2 — verify the headroom before spending a full grid
Gate: **first-pass accuracy in ~[0.5, 0.75] AND the feedforward-NN floor NOT at ceiling.** If the NN floor is still ≥0.95, the regime has failed its purpose — report that and iterate the regime, do not run the full ladder into a saturated cell. This check is cheap; do it first.

## Item 3 — the full RUD-C protocol
Run the `retry-compute-study` harness unchanged on the new regimes: ladder k ∈ {0,1,2,4,8}, gated retry + **all five controls** (ungated-all, ensemble-of-k, random-kick, feedforward-NN matched, Hopfield-k-steps), ≥2 loads × ≥2 ambiguity levels. Keep the compute axis and the "generous to the baselines" placement convention, and keep saying so.

## Item 4 — ⭐ the verdict that matters
**Does a regime exist where CLU-gated retry beats the feedforward-NN floor at matched compute?**
- **YES** ⇒ R3 has its benchmark; report the curve, the margin, the compute multiplier at saturation, and the mechanism controls together.
- **NO, in every ambiguity regime tried** ⇒ that is **decision-grade for R3** and must be reported as plainly as w23 reported the mask result. The mechanism claim survives either way (it is control-backed); the *leaderboard* claim does not.

## Acceptance
PREREG **before running** — predict, per regime, whether the NN floor stays dominant and at what first-pass accuracy the margin appears. The headroom gate (Item 2) reported per regime. Full curves with the five controls. ≥3 seeds on any cell that produces a headline (w23's single-seed is exactly what point 7 of the direction queue is fixing). Tests green; `ruff` clean; config at **all three sites plus `save_config`**.

## ⚠ Standing traps
- **CM-23(b) is SPLIT:** the *shape* claim survives ("a rising curve a saturated feedforward memory cannot draw"); the *absolute-dominance* claim is **retracted**. Do not re-assert dominance anywhere except from new measurement in this task.
- Cosine-to-nearest-well is a good **ranking** signal and a **useless acceptance** signal post-settle (w23 design note — recorded so you do not re-hit the zero-lift trap). Gate + lock-on-retry.
- Do not tune the regime until CLU wins. Pre-register the regimes, then report what they give.
