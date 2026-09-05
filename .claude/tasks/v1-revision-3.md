# Task: v1-revision-3 — close the three MUST-FIXes that dropped V1 to borderline (w15, writer)

- **Agent:** `paper-writer` · **Output:** `.claude/outputs/v1-revision-3.md`
- **Read first:** **`.claude/outputs/v1-referee-2.md`** (the whole report — MF-A/B/C and the three hostile quotes) · Charter (**C-1 REVERSED** — no audit confessions) · `.claude/claims_matrix.md` **v1.9** (CM-3, CM-7, CM-8, CM-12, **CM-14 AMENDMENT — new**) · `.claude/outputs/v1-router-baseline.md` (esp. L17) · `.claude/outputs/v1-certificate-payoff.md` · `.claude/outputs/paid-access-experiments.md` §7.1 + App A.1 ζ-grid.
- **Draft:** `.claude/papers/v1-short/draft.md`. Rebuild the PDF.

## Where we are
The referee moved V1 from weak-accept to **borderline** — not because the revision failed, but because it *succeeded* and then over-reached in the new page-1 prose. **Keep what worked:** F3 is earned, Payoff B's fine print survived intact in all six places, CM-3 discipline is clean, the noise wall is plotted (Fig 2c) and travels with every reversal claim, F6's kv16 is named. **Fixed, this is a clean weak-accept — and with MF-B done as a pricing law, plausibly an accept.**

## MF-A — the flagship sentence is contradicted by the paper's own §4.2
The draft fuses two different objects that every source keeps apart:
- **§3's `no_physics_router`** = an *untrained analytic constant map* `(q,p) ↦ (b,p)`, det J = 0.
- **§4.2's router** = a *449-param **learned** decision head* that **routes through the paper's own det J = 1 wormhole edge** (`v1-router-baseline` L17: *"Routes via the **same** direct wormhole edge"*).

So the abstract's "physics-free **learned** router" (det J = 0), §3.2.1(ii)'s "**the same** physics-free router", and §5's "reaches the same targets **more cheaply**" (a §4.2 FLOPs number, for a different arm, on a different task) are wrong.

**The rewrite makes the paper better, and this is the real content:** *decision ≠ transport.* §4.2's learned router **decides** *whether* to take the certified edge; §3's analytic map **is** an uncertified transport that destroys volume. A learned gate on top of a certified channel is not a counterexample to the certificate — it is a consumer of it. Disentangle the two objects everywhere (abstract, §3.2.1, §5), and state the distinction once, sharply. Do not let §4.2's FLOPs number migrate into a §3 sentence.

## MF-B — the reach panel's headline falsification is a line-search artifact
"The squeeze collapses past the causal box" is, **by the paper's own bracket formula and its own App A.1 ζ-grid**, an artifact of stopping the line search at `ζ = 2.0`. The ζ required to land `d = 4.0` is **`2.0105`** (the referee computed it). Two options — choose one, and say which:
- **(preferred) Restate the theorem as a pricing law.** The squeeze does not *fail* past the box; it costs rapidity, and the cost diverges as the target approaches the causal boundary. That is a **stronger, more physical claim** than a collapse, it is consistent with `paid-access-theory`'s bracket `[L, L + p₀ sinh ζ / M₀]`, and it matches the program's "paid access" framing exactly. **This is what I recommend.**
- (fallback) Extend the ζ line search past 2.0105 and report the collapse where it genuinely occurs — a minutes-long analytic run, no training.

Either way the sentence *"collapses past the causal box"* must not survive unqualified.

## MF-C — a C-7 cross-section contradiction, constructible from the paper alone
The F6 fix imported kv24/kv32 numbers, and now **§4.1's memory has fidelity 0.717 / accuracy 0.286 at kv32, while §4.3's kv32 has 1.00 / 1.00** and calls the low regime "an under-training artifact." Both are true of *different training budgets*; the paper never says so at the point of collision. Label the epoch budget inline at both sites. This is the exact M4 class the register exists to prevent — a reviewer needs no external source to construct it.

## Also fold
- **CM-14 was amended at matrix v1.9:** Thread-9's Prop-MH1 step 1 (momentum Gibbs-refresh `p ~ N(0, M_eff·T)`) is **FALSE in relativistic mode** (measured bias `−0.389` at `Θ=1` in a discretization-free, fully Metropolis-adjusted kernel; Newtonian control `+5e-6`); it is **one-line repairable**; **Prop-MH2 (latch erosion) and the governor/annealing decomposition are SAFE.** If §5's design-rule paragraph asserts "MALA(σ*) samples π exactly," it now needs the **Newtonian qualifier** — σ* is a proposal-tuning scale, not a correctness condition, once MH is applied. V1's units are Newtonian, so **this is a scope clause, not a retraction** — write it as one.
- **MF-1 stays open and is not yours:** the theory note is still `(Anonymous, 2026)`. Head critical path. Do not itemize it again.

**Acceptance:** MF-A disentangled (decision ≠ transport, stated once and sharply); MF-B restated as a pricing law (or the run done); MF-C's budgets labelled inline; CM-14's Newtonian qualifier landed; nothing that currently works is disturbed; PDF rebuilds clean. **Report which MF-B option you took and why.**
