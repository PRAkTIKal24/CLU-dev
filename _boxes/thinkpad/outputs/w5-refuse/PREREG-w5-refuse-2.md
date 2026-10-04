# PREREG-w5-refuse-2 — refuse-after-deletion, on the field store. ⭐ THE run.

**Hub** `w5 hub/refuse` · **target** `refusal` · **scored run 2 of 3**
**Written** 2026-09-20, BEFORE the run and before any number from run 1 was read.
⛔ Not edited afterwards; amendments are appended and dated.
**Runner** `python -m chlu.experiments.exp_gen_refuse --stage deletion --seed S --out /c/chlu/outputs/w5-refuse/runs/r1`

---

## 0. ⭐⭐ Where the claim lives, and why it is EXACT rather than measured

**A per-factor gate cannot express a conjunction deletion at all.** Delete
`object_hue = h* AND shape = s*` from a store that still holds `h*` and still holds `s*`, and
every bit of a per-factor value-presence bitmap is unchanged — there is no bit in the object
whose meaning is "this pair". Its `delta_reject` is **0.0000 by construction**.
`exp_gen_refuse.bitmap8_bits_changed_by_a_conjunction_deletion()` returns the integer **0** so
that this is checkable, not rhetorical.

**The joint table that CAN express it pays on the other leg.** A pair-PRESENCE table refuses
every conjunction the pool never held, and ⛔ **every row-3 and every row-4 request is an unseen
conjunction**. Measured on the split index (hub LOG, probe 4): the 109 scored corners name 15
`(object_hue, shape)` conjunctions, of which exactly **3 are never held** — the three scored
pair cells `(0,1)`, `(3,2)`, `(7,3)` — and **12 are held**. So the scored set contains both the
conjunctions that must stay ANSWERED and the held conjunctions one of which can be DELETED, and
they are disjoint by construction.

**⚠ THE RESIDUAL, DECLARED NOW rather than left for a referee.** A third object *does* match the
field store on this row: a pair-DELETION table kept **separate** from presence, so "never held"
and "deleted" are different arrays. That is not a hole in the argument — it is exactly what
`gen_field`'s `coupling_hole` segment *is*. It is built as an explicit comparator
(`pair_deletion`), priced, and reported. What the row then shows is that the cheap refuser which
follows a conjunction deletion is **not** the 8-byte rule, it is the 8-byte rule **plus** a
1,322-cell table — a 166-byte store with no generator attached.

## 1. The two deletions, declared in code before the run

| | operation | target | permitted bytes |
|---|---|---|---|
| **A. HOLE** ⭐ | `gen_field.punch_hole` | `object_hue` index **1** ∧ `shape` index **1** | **1 byte** of `coupling_hole` |
| **B. EVICT** | `gen_field.evict_term` | `object_hue` index **0** | the evicted items' bytes |

**Why this hole.** `(object_hue 1, shape 1)` is **held** by the test-condition train index
(11,676 rows — so the deletion is a real removal, not a no-op), it is **named by the scored
row-4 corners**, neither of its values is a held-out value, and it is ⛔ **disjoint from all
three scored pair cells**, which are the conjunctions that must stay answered.

**Why this eviction target.** `object_hue = 0` is inherited **verbatim** from the frozen
harness's own declaration (`exp_gen_factor_wells.DELETED_FACTOR / DELETED_VALUE_INDEX`). It is
pair-000's hue, so the survivor set is the other two pair cells' sub-corners plus every n-way
corner.

⛔ **No refit.** `assert_no_refit` must report every fitted segment byte-identical across both
operations, and `assert_changed_bytes_within` must report **0 bytes changed outside** the
permitted ranges. A failure of either voids the run. The negative control (a 1e-3 nudge to one
address-book row) is run to show the instrument is not blind.

## 2. What is measured, on what

**Request sets** (test condition, seeds 0/1/2, cache generation 2):
- the **109 scored corners** — 60 row-3 pair sub-corners + 49 row-4 n-way corners (≥ 49, w5-common §2);
- the **512 `row1_seen`** requests = the accept leg;
- **four matched probe groups** constructed on row 1's own backgrounds, differing only in the two
  hole factors: `both` (the deleted conjunction), `A_only`, `B_only`, `neither`. ⛔ Constructed,
  not filtered — the conjunction is held out of nothing and a filter over pool rows would test it
  on zero requests.

**Per refuser, before and after each deletion:**

| quantity | on |
|---|---|
| `reject_deleted` | the corner requests naming the deleted conjunction (hole) / value (evict) |
| ⭐ `delta_reject` | `reject_deleted(post) − reject_deleted(pre)` |
| `accept_unseen_valid` | `row1_seen`, post-deletion, same setting (the frozen pair's accept leg) |
| ⭐ `accept_unseen_conjunction` | the corner requests naming a **never-held** `(object_hue, shape)` |
| `distinct_valid_novel_tuples` | the survivor corners, judged by the frozen external classifier |
| the 4-way group table | `both` / `A_only` / `B_only` / `neither`, pre and post |

**Judge economy, and it is asserted not assumed.** The hole changes **no rendered feature**, so
one judge pass is reused pre and post. The runner **asserts** `np.array_equal` on every corner
and accept-leg feature and **raises**, voiding the run, if a single feature moved. The eviction
*does* change features, so it is re-rendered and re-judged.

## 3. THE BARS, declared in advance

**Bar 1 (the claim).** On the HOLE, on all three seeds:
`field_geom` reaches `reject_deleted ≥ 0.99` **and** `accept_unseen_conjunction ≥ 0.99`,
while `bitmap8` has `delta_reject = 0.0000` exactly and `pair_presence` has
`accept_unseen_conjunction = 0.0000`.

**Bar 2 (no refit).** 1 byte changed, 0 outside, `assert_no_refit` passes on every fitted
segment, features bit-identical — all three seeds.

**Bar 3 (both halves, or the row is nothing).** The survivor corners still produce
`distinct_valid_novel_tuples > 0` after the hole. ⛔ A refuser that damages the generator has
shown damage, not deletion.

**Bar 4 (the eviction, and it is NOT the claim).** `field_geom` reaches `reject_deleted ≥ 0.99`
on the evicted value with survivors still answered; `bitmap8` **also** reaches it, by clearing
one bit. ⭐ I expect a TIE here and I am declaring that in advance so the row cannot later be
sold as a win.

**Bar 5 (the residual, declared).** I expect `pair_deletion` to **match** `field_geom` on the
hole. That is the honest comparator and it is on the table by my own choice.

## 4. ⛔ What would prove this run wrong

- **Refuse-after-deletion works only if I refit** ⇒ the whole point is eviction without refit;
  `assert_no_refit` is the check and a failure voids the run.
- The hole refuses **unseen** conjunctions too ⇒ I have built the joint presence table by
  accident and the row is worth nothing. `accept_unseen_conjunction` is the direct test.
- The hole changes the read on requests that do not name it ⇒ the byte assertion and the
  feature-equality assertion both catch it.
- `field_geom` and `pair_deletion` tie **and** `pair_deletion` needs no generator ⇒ the claim
  narrows to "the flag is 1 byte inside a deployed image that also answers", and I report that
  narrowing rather than the wider claim.

## 5. Population footnote and labels

As PREREG-w5-refuse-1 §6 and §7, unchanged and binding.

## 6. Cost

Local CPU, 3 threads. No GPU. Two judge passes over ≈ 8,400 requests per seed; estimated
≤ 35 min/seed. Resumable per (stage, seed); wall-clock capped.

---

# ⚠ AMENDMENT 1 — 2026-09-20, appended after the §A.5 replication reported. ⛔ Nothing above is edited.

The independent replication spoke reproduced every number and every byte proof at all three
seeds, and raised three framing objections. **Two are upheld and one is upheld in part.** No
bar is moved and no result changes; what changes is how §0 is allowed to describe itself.

**1. UPHELD, and it was an overclaim.** §0 said `bitmap8_bits_changed_by_a_conjunction_deletion()`
returns 0 "so that this is checkable, not rhetorical". ⛔ The function was literally
`return 0` with a docstring — a hard-coded literal cannot be its own evidence, and that
sentence was false as written. **Fixed in code** (`eff702e`→ next commit): the function now
rebuilds the presence bitmap from the pool, rebuilds it again from the pool with every row
holding the conjunction removed, packs both and counts differing bits. Result: **0 bits, with
`before_hex == after_hex == ffffffbffdffff80`** — byte-for-byte the hex the spoke computed
independently. ⭐ It also now returns the **empirical side condition** the claim rests on:
23,424 rows still hold `object_hue = 1` and 83,100 still hold `shape = 1` after the deletion.
Had the conjunction been the sole carrier of either value, a bit *would* have flipped.

**2. UPHELD.** `delta_reject = 0.0000` for the per-factor gate is **definitional**, not a
measurement in the same sense as `field_geom`'s 1.0000 → 0.0000, and putting the two in one
table invites reading them as comparable results. The scoreboard row now carries the
qualifier **"0 BY CONSTRUCTION"** explicitly rather than leaving it to the prose.

**3. UPHELD IN PART.** The spoke instrumented `gap_read` and confirmed that post-hole, on the
324 deleted-conjunction requests, `hole` fires at 1.0000 while `empty_basin` and
`out_of_range` fire at 0.0000 — i.e. ⛔ **on this row the field store does not refuse
geometrically at all; it is a flag-byte lookup in `coupling_hole`.** §5 of the report already
conceded exactly this, and my own mechanism decomposition measured the same thing
independently. ⚠ But the spoke is right that the *headline* sentence let the word "geometry"
do work it had not done, and the report's §1 and §3A are reworded accordingly.

**⭐ ADOPTED FROM THE SPOKE — a better argument than mine, and it is theirs.** The per-factor
gate's *only* available move is to clear value `v`'s bit, and that **over-deletes**: measured
on the scored corner set, it refuses the 324 requests naming the deleted conjunction **and
576 more** naming `object_hue = 1` with a different partner — **900 refusals to delete 324,
64 % of them collateral**. That is a **measured** cost rather than a definitional zero and it
makes the point about per-factor gates better than `delta_reject = 0` does. ⛔ It is computed
from the **split index alone** — no store, no read, no judge — so it is a structural count and
**costs no scored run**; this hub has spent all three and does not smuggle a fourth.
