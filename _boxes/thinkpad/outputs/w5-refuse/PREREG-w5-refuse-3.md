# PREREG-w5-refuse-3 — the rival refusers the target file names

**Hub** `w5 hub/refuse` · **target** `refusal` · **scored run 3 of 3**
**Written** 2026-09-20, BEFORE the run. ⛔ Not edited afterwards.
**Runner** `python -m chlu.experiments.exp_gen_refuse --stage rivals --seed S --out /c/chlu/outputs/w5-refuse/runs/r1`

---

## 0. Why this run exists, and why it cannot win

The target file names four rivals on the refusal rows: **the table (refuses by absence),
LACE energy, kNN density, and the 8-byte rule.** Runs 1 and 2 covered the 8-byte rule. This
run covers the other three, and ⛔ **none of them can produce a win for this hub** — runs 1
and 2 already established that rows 5 and 6 tie 8 bytes and cannot be beaten (the 8-byte pair
is `(1.0000, 1.0000)`; nothing beats perfect). This run exists to **complete the rival set
honestly**, and because two of the three answer a question the programme keeps asking.

## 1. The three rivals

### `table_absence` — the prototype table, refusing by absence
Refit **like-for-like on my own value-holdout fit pool**, so it is not the
`montero_combination` artefact that saw both held-out values. Accept iff no duplicate, no
out-of-range, and every asked on-grid value **has a prototype** (the pool held it); off-grid
values take the same bracket rule every arm uses.
**Ledger: `(57 + 1) × 2048 × 4 = 475,136 B`.**

⭐ **Predicted in advance from the mechanism alone:** it decides on *exactly the same fact* as
the 8-byte bitmap — which `(factor, value)` the pool holds — so it must return **identical
answers**, at a byte multiple of **59,392×**. If it does anything else, my understanding of one
of the two objects is wrong and I will say so. w5-common §3.9 requires the byte multiple beside
the row; it will be printed.

### `knn_density` — ⛔ the length scale the target file forbids ME, as a rival
Refuse iff the request's address is further than `τ_knn` from its k-th nearest **live atom**,
in units of **the atoms' own median nearest-neighbour spacing**.

**`τ_knn = 4.5`, `k = 1`. ⛔ SET ON VALID DATA ONLY AND WRITTEN DOWN HERE BEFORE ANY INVALID
ROW IS SCORED WITH IT.** Rule declared before the number was read: the 99th percentile of that
score over the **val** field store's own fit-pool requests, max over the three seeds, rounded
up to 2 dp. Measured p99 = 4.4972 / 3.8885 / 3.6813; p50 = 1.2667 / 1.2007 / 1.1372.
Artifact: `knn_calibration_val.json`.

⭐⭐ **This is the point of including it.** My own `in_gap` measures to **basin coordinates** in
units of a **within-value spread**; `knn_density` measures to **atoms** in units of **inter-atom
spacing** — the length scale the admission gate died on. The pair isolates exactly that
difference. ⛔ **No number this hub claims is computed with `knn_density`.**

### `lace_energy` — ⛔ DECLARED NOT RUN
Spoke A does not expose LACE as a callable: its output is a **stochastic draw, not a function
of the request**, so an energy cannot be evaluated on my requests. ⛔ A declared NOT-RUN is
never a null. Established **mechanically** (`load_rival` raises `RivalRefusal`), not asserted.
LACE is **95.1×** the store budget and that multiple is printed beside its dash.

## 2. The full rival set on a refusal row — established mechanically, not asserted

For every rival in `RIVALS-READY.json` the runner records two separate facts:
1. **Is there a refusal mechanism at all?** Every learner in the set is a generator —
   `r(values) → (features, out_of_range)`. It answers every in-range request. On a refusal row
   it *is* the `accept_everything` degenerate arm and scores **0** by construction.
2. **Is the fit like-for-like?** Checked with `load_rival(...).like_for_like({object_hue:[0.5],
   shape:[0.0]})`, ⛔ not by trusting the note.

⭐ A rival that cannot enter a row takes **a dash and a footnote**, and ⛔ a dash is never a win
for us (w5-common §1).

## 3. What is measured

Rows 1, 2, 5 and 6 (512 requests each) and the **conjunction deletion**, seeds 0/1/2, test
condition, generation 2 — the same requests runs 1 and 2 used, so the columns are comparable.

## 4. Expected direction, declared in advance

1. `table_absence` = `bitmap8` **exactly**, on every row, at 59,392× the bytes.
2. `knn_density` **fails to separate**: it will refuse a non-trivial fraction of row 1 (its
   threshold has a 1 % false-refusal budget *by construction*) and will **not** reach 1.0000
   on row 5, because a held-out value's address is a legitimate point of the address manifold
   and sits near atoms. ⇒ a pair strictly worse than 8 bytes.
3. **Both rivals have `delta_reject = 0.0000` on the conjunction deletion**, for the same exact
   reason the 8-byte rule does: `table_absence` has no pair cell to change, and `knn_density`
   reads only atoms, which a hole does not move. ⭐ This **widens** run 2's claim from "a
   per-factor gate cannot follow" to "**no per-factor and no atom-density rule can follow**",
   and it is the one genuinely new thing this run buys.

## 5. ⛔ What would prove this run wrong

- `table_absence` differs from `bitmap8` anywhere ⇒ I have mis-stated one of the two objects.
- `knn_density` reaches `(1.0, 1.0)` ⇒ the forbidden length scale works on this encoder after
  all, and my §4.2 prediction and the admission gate's post-mortem are both wrong.
- `knn_density` shows a non-zero `delta_reject` on the hole ⇒ the hole moved an atom, which
  would contradict run 2's byte assertion.

## 6. Cost

Local CPU, 3 threads. No GPU. Estimated ≤ 5 min/seed.
