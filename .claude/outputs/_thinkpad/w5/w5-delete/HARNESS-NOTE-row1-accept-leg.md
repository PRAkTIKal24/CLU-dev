# HARNESS NOTE — the frozen accept leg is insensitive to a CONJUNCTION deletion by construction

**Filed by** w5 hub/delete, 2026-09-19T23:xx / 2026-09-20T00:18Z, **on behalf of w5 hub/delete and
w5 hub/refuse jointly** — both hubs measured it independently, on different deletions and different
splits, and agreed to file it once rather than twice.
**Severity** medium (a silent 1.0000 that looks like evidence and is not).
⛔ **Not fixed here.** `chlu/data/gen_value_holdout.py` and the frozen `score_refusal_row` are
harness assets (charter §A.4). **This is a report, not a patch.** The fix is a docstring, at a re-pin.

## What is wrong

`score_refusal_row` pairs **"reject on the deleted set"** against **"accept on row 1"**
(`row1_seen`). For a **conjunction** deletion, that accept leg cannot move, because `row1_seen`
never realises a conjunction at all: the emitter holds **one axis value constant per kind** — done
deliberately, so that row 1 structurally matches row 5 — with the consequence that both *marginals*
of any pair appear in row 1 while the *pair* never does.

So a hub deleting a conjunction gets **1.0000 on the accept leg no matter what its deletion did**,
and that 1.0000 reads like evidence of surgical precision when it is a property of the query set.

## Measured, twice, on two different targets

| hub | deleted conjunction | row-1 records naming marginal A | naming marginal B | **naming the CONJUNCTION** |
|---|---|---|---|---|
| w5 hub/delete | `object_hue=0.0 ∧ shape=1.0` | 28 of 512 (`object_hue=0.0`) | 84 of 512 (`shape=cylinder`) | **0** |
| w5 hub/refuse | `object_hue=0.1 ∧ shape=1.0` | — | — | **0** |

Two different conjunctions, two different splits, marginals present in both cases, the pair present
in neither. ⭐ **That makes it structural, not a coincidence of either target choice.**

## A second, sharper consequence — and a cross-check worth keeping

The same census explains the **value** deletion's accept-leg drop exactly. w5 hub/refuse's scored
post-eviction row-1 acceptance, all three seeds, seed spread 0.0000, is **0.9453125**. This hub's
request-set census finds **28 of 512** row-1 records naming the evicted value `object_hue = 0.0`:

    1 − 28/512 = 0.9453125      (exact, verified)

The 28 requests the accept leg loses are **precisely** the 28 that name the evicted value, and it is
**correct** that they are refused — so that drop is a feature of the row, not a defect in it. One
number came from a request-set census, the other from a rendered-and-judged scored pipeline, and
they agree to the digit. ⭐ Worth keeping as a standing self-check on the row.

## The fix (for the harness owner, at a re-pin)

A docstring, in `chlu/data/gen_value_holdout.py` where `row1_seen` is built and/or on
`score_refusal_row`, saying in substance:

> `row1_seen` holds one axis value constant per kind so that row 1 structurally matches row 5. A
> consequence is that both marginals of any factor pair occur in row 1 while the pair itself never
> does. **Therefore the row-1 accept leg is insensitive to a CONJUNCTION deletion by construction**
> and its 1.0000 is not evidence about one. A hub scoring a conjunction deletion should pair its
> reject leg against a set that could actually have been damaged — e.g. the never-held-conjunction
> requests, or corners that survive the deletion.

⛔ No number changes. This prevents a future hub from quoting a 1.0000 that cannot mean what it looks
like it means.

## What the two hubs did instead, for the record

- **w5 hub/refuse** pre-registered `accept_unseen_conjunction` as a separate quantity and rests its
  claim on the 2,160 never-held-conjunction requests, not on the frozen pair.
- **w5 hub/delete** uses the **89 survivor corners (4,968 requests)** as its accept leg. Those
  contain both never-held conjunctions (`pair-001` green ∧ sphere, `pair-002` blue ∧ oblong) and
  held ones, and a whole-value eviction genuinely could damage them — so its accept number is
  sensitive by construction rather than insensitive by construction. ⚠ That was chosen for a
  different reason (the paired per-corner estimand) and only checked against this trap afterwards.
