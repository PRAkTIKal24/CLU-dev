# w4 hub/placement — TARGET: beat the label-MI partition while holding row 4 against the rivals

**Boot:** `autonomous-hub-charter.md` → `w4-hub-common.md` → this file. **Base: `base/iter4 = 9dfed5e`.**
**Worktree:** `/c/chlu/worktrees/CHLU-w4-placement` · **LOG:** `/c/chlu/outputs/hub-placement/LOG.md`
**Scored budget: target `placement`, 3 runs.** ⛔ Gated on `RIVALS-READY.json`. DEV starts now.

## DIAL DECLARATION
**Dial: WHERE the wells are** — how channels are assigned to wells. ⛔ Not the read, ⛔ not the content.
Controls on every row: landscape-removed, landscape-shuffled, correlation-split + linear-head.
**Falsified if** nothing beats the label-MI partition, or if beating it costs row 4 against the rivals.

## The target
**Beat the label-MI partition — which is an ABLATION baseline, w4-common §1 — while HOLDING row 4
against the full rival set.** ⭐ Both halves are required: a placement that wins the ablation and loses
the rivals has achieved nothing.

## ⭐ Try in this order, and report ALL of them, including the ones that fail
- **(a) two-stage** — MI-partition init, then refine.
- **(b) swap loss** — swap one well between two images, decode, and let the **judge** score validity.
- **(c) independence** — a between-well independence penalty plus a within-well variance floor.
- **(d) swap loss with the encoder in the loop** — ⛔ **last**, and only if (a)–(c) are exhausted.

⛔ **KNOWN, do not rediscover:** reconstruction-only placement was a **null**, and **unfreezing the
encoder alone made it worse** — which is why (d) is last rather than first.
⚠ **The encoder bounds what is reachable:** only **12 of 1984** live channels are top-ranked by `shape`
and **4** by `scale`; three of five wells go to hue. ⭐ A placement method cannot invent structure the
encoder does not carry, so **report what fraction of your gain is placement and what is the ceiling.**

## ⛔ What would prove this hub wrong
- Nothing beats the MI partition ⇒ label-alignment is already at the reachable optimum. Report it;
  that closes the placement question for this encoder.
- A method beats it on the ablation but loses row 4 to the rivals ⇒ ⛔ not a result.
- The gain is the encoder's, not the placement's ⇒ ⭐ say which, with the channel-level number.
- (d) helps only with the encoder unfrozen ⇒ it is no longer a placement result; it is a representation
  result and must be labelled as one.
