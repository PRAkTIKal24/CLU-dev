# AMENDMENT 1 to PREREG-w13-encoder-adapt-1.md

**Appended 2026-09-21, before any w13 number on any condition — val or test — existed,
and before any row was written to `/c/chlu/SCORED-LEDGER.md`.**
Exact time in `PREREG-w13-encoder-adapt-1-AMENDMENT-1.timestamp.txt`.
⛔ **No bar is moved by this amendment.** §4 (the MARGIN bar), §5 (the probe as a margin),
§7 (3 seeds, all or none) and §8 (the falsifiers) are unchanged word for word.

---

## What changed, and why it is a MEASUREMENT and not a choice

§3 of the PREREG named **one** store at K\* = 32 and let a reader assume the encoder
trained through it. It cannot. Measured before any spend, on the `--quick` warm smoke at
K = 32 (`dev/logs/smoke_warm_s0.log`), the frozen harness refuses:

> `AssertionError: well(s) [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
> 21, 22, 23, 24, 26, 27, 28, 29, 30, 31] own no factor -- the swap loss has no target
> for them and is ill-posed; reported, not patched`
> — `chlu/core/gen_swap_loss.py:148`

**26 of 32 wells own no factor.** The swap loss assigns each well exactly ONE owned
factor by a Hungarian match on a measured transfer matrix, and Shapes3D has **6** factors,
so at most 6 wells can own one at any K. `derive_fixed_ownership`'s own docstring states
the design: *"With K=5 wells and 6 Shapes3D factors exactly one factor is UNOWNED by
construction; that is a property of the pinned ledger."*

⛔ **This hub does not patch it.** Restricting the swap draw to the six owning wells would
be overriding a refusal the frozen harness makes deliberately, in its own words "reported,
not patched". ⭐ **That the swap loss is ill-posed at K\* = 32 is a FINDING of this wave
and is reported as one.**

## The amended §3 — TWO objects, different K, said plainly

| object | K | A | atom ledger | role |
|---|---|---|---|---|
| **the SCORED single design** | **32 (K\*, wave 12, unchanged)** | 63 | 63 × 2048 × 4 = **516,096 B** (= w12's own `p1_atom_bytes`) | ⭐ what the MARGIN is measured on — `CovContentStore` rebuilt from **each arm's own** generation-2 cache by wave 12's unmodified `scripts/w12_crossover_lib.py` |
| the **in-loop VQ store** the encoder trains through | 5 | 64 | 64 × 2048 × 4 = **524,288 B** (w8's pin exactly) | the object the swap loss and the store-read-error term are defined on |

⭐ **The task's pin is honoured exactly where the task puts it** — "Store design FIXED at
K\* = 32" is a statement about the store DESIGN that is scored, and that store is at
K\* = 32 on every arm, unchanged. The training-time VQ store is a different object and
always was; the PREREG now says so instead of leaving it to be assumed.

⚠ **The consequence for the reader, stated rather than buried:** the adaptation gradient
reaches the encoder through a **K = 5** store, and the margin is then measured on a
**K = 32** store built from the resulting features. If adaptation helps at K = 5 and the
gain does not survive the rebuild at K\* = 32, that is a real limit of this design and
will be reported as one, not as a failure of adaptation.
