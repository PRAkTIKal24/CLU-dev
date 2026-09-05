# learned-confine-causal — an independent causal test of the saddle criterion (★)

**Agent:** physics-theorist. **No worktree, no tracked code** (§3.2 exempt — scratch + report only).
Base local `main @ 082d095` (post-w26).
**Campaign tag: [C1W27].** This is **your own commission item (i)** from
`.claude/outputs/readout-channel-theory.md` §8, ruled in scope by the Head 2026-07-29. Budget ~1 h.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** none — **theory / instrument.** This is a causal test of a criterion, not a performance
  claim, and it must not be reported against any performance axis.
- **Laundering control:** n/a (no performance number). Instead: the **prediction must be registered
  before the run**, with its numeric band, and scored.
- **Falsifies the claim:** retrieval does **not** move in the predicted direction, or moves outside
  your registered band ⇒ `learned_confine` is **not** entering the ceiling through `2α|c_i|` and the
  "(★) reach parameters, not free knobs" claim from w26 is downgraded to a correlation.
- **Does NOT falsify:** a smaller-than-predicted effect that is still in the right direction and
  inside the band; any absolute retrieval level (this is a *difference* test).

## The test (your own spec, unchanged)
`learned_confine` **0.05 → 0.022** at **d=4, K=16, `atom_init_width` 0.15, 3 seeds**.

Why this cell: w26 established that `learned_confine = 0.05` and the ball radius `R = 1.0` enter the
capacity ceiling through `2α|c_i|` — but only *observationally*. Halving α at fixed geometry is the
independent intervention. `atom_init_width` 0.15 is chosen because that is the width at which w26
measured retrieval *destroyed* (`D ∝ s^{1.46}`: 0.910 at s=0.320, 0.459 at s=0.200) — i.e. the cell
where the criterion has the most room to be wrong.

## Requirements
1. **Register the prediction first** — direction, magnitude and band, derived from the saddle
   condition on `L = √(|c|² + a²)` (not from the superseded `|a|max/κ` bound, and not from κ≈3).
   Write it to `.claude/outputs/learned-confine-causal/PREREG.md` **before** running anything.
2. **3 seeds**, value-blank control, and state your sd convention (w26's r2 tables quote
   **population** sd; say which you use).
3. **Say what it means for the two load-bearing knobs.** If (★) survives, `learned_confine` and the
   ball radius `R` are *predictions of the ceiling*, and that is a statement the `r2-d-sweep-close`
   engineer needs — but **you do not edit shipped code and you do not message another spoke**; put
   the implication in your report and the Hub will relay it.
4. **Do not re-open** the trilemma, option (d), or the excursion arms — those are owned elsewhere
   this wave. If your result bears on them, say so in one paragraph and stop.

## File ownership
**You own:** `.claude/scratch/learned-confine-causal/` and `.claude/outputs/learned-confine-causal*`.
⛔ **Do NOT edit** `chlu/core/memory_potentials.py` (split between two engineers this wave),
`chlu/core/placement.py`, `chlu/core/controller.py`, `chlu/experiments/exp_designed_mechanism.py`, or
any transfer doc (`negative_results.md`, `claims_matrix.md`, `research_roadmap.md`,
`philosophy-synthesis.md` — the curator owns those).

## Compute
Two engineer worktrees are running heavy sweeps on the same 8 cores this wave. Keep this to **≤2
concurrent jobs** and to your ~1 h budget; if the cell will not fit, report **NOT RUN** with the
measured cost rather than shrinking the seed count below 3.

## Deliverable
`.claude/outputs/learned-confine-causal.md` — PREREG scorecard first, the measured effect with its
band, the verdict on (★) as a *causal* statement, and the one-paragraph implication for the ceiling
knobs. No git branch needed (no tracked code); if you touch anything tracked by accident, revert it.

⛔ **Do-not-quote, carried:** κ ≈ 3 / the `|a|max/κ` reach bound (superseded by the saddle criterion,
31/32 on the shipped `V`, zero free parameters) · "2.6" (retracted, never reproducible) · the base-√2
/ `d^1.62` exponent · **quote the curve, not the endpoint.**
