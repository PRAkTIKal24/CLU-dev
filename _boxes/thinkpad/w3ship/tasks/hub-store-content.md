# hub/store-content — TARGET: store EARLIER-LAYER content and recover scale, orientation and shape

**Boot:** `.claude/autonomous-hub-charter.md` → `.claude/tasks/w3-hub-common.md` → this file. Nothing else.
**Worktree/branch:** `hub/store-content`, off the pin **`4125fce`** + the `w3-value-holdout` additive modules.
**Log:** `.claude/outputs/hub-store-content/LOG.md` · **Scored budget: the `generation-table` target, 1 spent, 2 left.**

## DIAL DECLARATION
**Dial: WHAT IS STORED.** ⭐ You are the only hub of the four that changes the content rather than the
organisation or the read. Laundering control beside every number: landscape-removed must collapse, and
the byte ledger is asserted per row. **Falsified if** earlier-layer content at matched bytes fails to
move `scale` or `orientation` — then the loss is the quantiser's, not the layer's.

## The target
**Win scoreboard rows 1–4 by writing earlier-layer features into the store alongside the penultimate
map, at matched bytes.**

## The measured gap you are attacking
⭐ **The encoder carries `scale` at 0.94 and `orientation` at 0.99. The quantised store loses them** —
the codebook loses **90.5 % of `scale` and 100 % of `orientation`**. The linear read recovers part.
⇒ **The information exists in the representation and does not survive into the store.** That is the one
place in this programme where more content, rather than more bytes, is the plausible lever.

## The constraint that makes it a real experiment
⛔⛔ **Matched bytes.** You are not adding capacity, you are **reallocating** it: earlier-layer content
comes out of the penultimate map's budget, not on top of it. ⭐ More bytes is measured not to be the
lever — above ~2× budget the addressing head binds, and doubling from 2× to 4× buys +0.08 tuples
(common §4.6). ⛔ An arm that wins on extra bytes has demonstrated nothing.

⚠ **The definitional point, so you do not chase a phantom:** in this codebase **the encoder's output IS
the penultimate feature map** — `GenEncoder` is `[conv → GroupNorm → ReLU] × len(channels)` and the
penultimate map is the last block's post-ReLU output, `(128, 4, 4) = 2048` dims, ⛔ not pooled. There is
no skipped head. "Earlier layer" therefore means **an earlier conv block's output**, and you will have
to decide how to bring its spatial extent into the store's budget. ⭐ Say what you chose and why.

⚠ **`D = 2048` is a design constraint, not a free choice:** the LACE baseline is only valid in whitened
coordinates and needs a `D × D` eigendecomposition, feasible to about `D ≲ 2×10⁴`. ⛔ If your layer
choice pushes the whitened dimension past that, LACE is no longer a fair row and you must say so rather
than quietly degrade it.

## The bar
**Beat fitted basis + linear head at matched bytes on ≥ 49 corners, paired, margin declared in advance.**
`knn_proto` second row. ⛔ Never a single cell. ⭐ Pin the store fit completely.
⭐ **Report the per-factor breakdown on every row.** The aggregate can move for reasons that have nothing
to do with your hypothesis; `scale` and `orientation` moving is the hypothesis. In wave 2 the entire
generation-table deficit was **four `shape` images** while three hues were at 1.0000 — an aggregate
hides that completely.

## ⛔ What would prove this hub wrong
- Earlier-layer content at matched bytes leaves `scale` and `orientation` where they are ⇒ the loss is
  the **quantiser's**, not the layer's, and the content dial is the wrong dial. **Report it plainly** —
  that is a clean negative and it retires a whole line of attack.
- The aggregate improves but `scale` and `orientation` do not ⇒ you won on something else. Say what.
- The gain vanishes at matched bytes and needs extra capacity ⇒ no claim (see the constraint above).
- A matched-bytes lookup over your atoms matches the arm ⇒ §C.1: the store is a lookup and is reported
  as one.
- ⚠ Earlier-layer features change the feature cache ⇒ ⛔⛔ **the re-encode gate binds.** Check your cache
  coverage **first** and record `shortfall`; if a run would extend the cache, it is enforced at full
  strength. **Establish this before you design the arm, not after.**
