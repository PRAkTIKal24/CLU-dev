# w6 hub/retention — TARGET: lifetimes as deletion-by-decay, on the field store

**Boot:** charter → `w6-hub-common.md` → this file. **Base `base/iter6 = df025685`.**
**Worktree:** `/c/chlu/worktrees/CHLU-w6-retention` · **LOG:** `/c/chlu/outputs/w6-retention/LOG.md`
**Scored budget: the `deletion` target, 1 spent in w5, ⭐ 2 remaining.**

## DIAL DECLARATION
**Dial: time** — a per-item lifetime as the deletion operator. Controls: landscape-removed,
landscape-shuffled, the table with rows removed, ⭐ **retrain-from-scratch with its cost line**, and
⭐ **the 8-byte gate printed beside every refusal number.** **Falsified if** decay is behaviourally
indistinguishable from eviction — because wave 5 measured what eviction costs.

## The target
**Reproduce the shorts' exact-decay law on the field store, and measure refuse-after-decay against time.**
Pre-register the law **from the paper's own formula** before measuring. The inventory spoke supplies it;
⛔ if it reports the law absent, **say so and do not invent one.**

## ⭐⭐ The question wave 5 handed you, and it is the interesting one
Wave 5 established, on this same object:
- ⭐ **the coupling hole is exact and FREE** — refuses at 1.0000 for **1 byte**, 0 outside, no refit,
  **0.0000 composition cost**, 101 other never-held conjunctions untouched, zero seed spread;
- ⛔ **eviction is NOT free** — it refuses perfectly but **fails the composition bar**
  (−2.09 / −0.91 / −6.74), because *an item store loses composition when it loses atoms, whichever
  atoms they are.*

⭐⭐ **Decay is eviction spread over time. So the sharp question is: does a lifetime buy anything
eviction does not — or does it simply pay the same composition cost more slowly?**
**Measure the composition cost as a function of decay, not just the refusal.** ⛔ A refusal curve
without the composition curve beside it answers nothing this programme still needs.

## The comparison
Refuse-after-decay vs time, **with the composition cost on the same axis**, against: instant eviction
(w5's measured −2.09/−0.91/−6.74), the coupling hole (0.0000), the table with rows removed,
retrain-from-scratch **with its cost line**, and the 8-byte gate. MLP/transformer/LSTM get a dash.
⚠ **Churn is a cost line only, one definition, ⛔ never a claim.**

## ⛔ What would prove this hub wrong
- Decay's composition cost equals instant eviction's at every horizon ⇒ **lifetimes buy nothing but
  latency. Report it plainly** — that closes a mechanism the programme has carried for a long time.
- The decay law fits only with a free parameter the shorts do not derive ⇒ it is a fitted curve, not
  a law; label it so.
- Refuse-after-decay needs a refit anywhere ⇒ ⛔ the central property fails.
- ⛔ Do not choose `scale`, `orientation` or `shape = cube` as a decay target — all three are
  degradation attractors on this decoder.
