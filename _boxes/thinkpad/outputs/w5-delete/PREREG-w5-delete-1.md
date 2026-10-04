# PREREG — w5 hub/delete — scored run **1/3**, target `deletion`

**Written 2026-09-20T00:12Z, BEFORE any held-out number.** Charter §A.1/A.2/A.3.
Base `base/iter5 = 6675904`. Worktree `/c/chlu/worktrees/CHLU-w5-delete`.
⛔ **No bar in this file is moved once a result has been seen against it (§A.3). Amendments are
appended and dated, never edited.**

---

## 0. The claim this run tests

> **Row 7 on the field store, by eviction and by the coupling hole, NEITHER by refit.**
> Two deletions, and for each one BOTH halves — the refusal *and* the surviving composition.

**Deletion A — evict `object_hue = 0.0` (red) by lifetime/eviction.** Red is refused everywhere;
green and blue still compose on every shape; no refit anywhere in the path.

**Deletion B — punch the coupling hole on `red ∧ cylinder` ONLY.** Red∧cylinder is refused; the
never-seen conjunctions `green ∧ sphere` and `blue ∧ oblong` stay **ALLOWED**; red's other shapes
stay allowed; cylinder's other hues stay allowed.

## 1. Estimand, instrument, seeds

| item | value |
|---|---|
| condition | `value_holdout` (the **scored** condition), fit pool = `fit_pool(train_rows, 131072, 20260919)` |
| feature cache | **GENERATION 2**, `CHLU_GEN_FEATURES_ROOT=/c/chlu/outputs/spoke-a-rivals/cache_generation2`, scope `full`, 480,000 rows. ⛔ The runner refuses to start on generation 1 or on a non-`full` scope. |
| seeds | **bridge seed = store-fit seed = 0, 1, 2**, three genuinely different fit pools, decoders and classifiers |
| judge | frozen `decoder_v1_s{seed}` + hash-asserted external classifier; ONE instrument for every arm |
| corners | **109** disjoint corners, every row in `test_rows`, **0** in any fit pool: 49 n-way (72 requests each) + 60 pair sub-corners (36 each) = **5,688 requests** |
| **deletion set** | the **20** `pair-000` sub-corners = **720 requests** = red ∧ cylinder |
| **survivor set** | the **89** remaining corners = **4,968 requests** = 49 n-way + 20 `pair-001` (green∧sphere) + 20 `pair-002` (blue∧oblong). **89 ≥ 49** (w5-common §2) |
| composition metric | `distinct_valid_novel_tuples` **per corner**, paired by corner |
| refusal metric | each arm's **own** refusal verb, reported beside the shared free 8-byte `accept_rule` |
| byte budget | **524,288 B** store (PRIMARY). Parameter budget named separately wherever quoted (w5-common §1). |
| labels | used **once at fit time** to align wells and at the address book, nowhere else (w5-common §4) |

**Why this corner set and not another:** the three held-out pair cells *are* the target sentence.
`pair-000` = red∧cylinder (the deletion), `pair-001` = green∧sphere, `pair-002` = blue∧oblong (the
survivors). All three are held out of `train_rows` entirely, so all three carry `coupling_w = 0` —
**identical evidence state**. The matched control is a property of the instrument, not a choice I made.

⛔ **Targets NOT used, per the task file:** `scale`, `orientation`, `shape = cube` — the three
degradation attractors. The deleted factor is `object_hue`; the holed pair's shape is `cylinder`.

## 2. The bars — declared now, in advance

### BAR 1 — Deletion A, the REFUSAL half
After `evict_term(object_hue, 0.0)`: the field store's own refusal rate on the **720** deletion-set
requests is **≥ 0.99**, and on a constructed red-on-other-shapes probe (red∧sphere, red∧oblong,
256 backgrounds each) **≥ 0.99**, on **every seed**. Pre-deletion the same rate is reported and is
expected near 0. The firing mechanism must be **`empty_basin`** — that is what an atom eviction leaves.

### BAR 2 — Deletion A, the SURVIVING-COMPOSITION half
Paired per-corner Δ in `distinct_valid_novel_tuples`, `field/evict_red` − `field/base`, over the
**89 survivor corners**: **|Δ| ≤ 1.0** novel tuple per corner (non-inferiority at **−1.0**), on every
seed. Broken out and also reported separately for **green** (`pair-001`) and **blue** (`pair-002`)
and for each of the three shapes among the n-way corners — that break-out is the literal target
sentence "green and blue still compose on **every shape**".
⭐ +1.0/corner is the programme's own standing margin (`DECORATION_MARGIN_NOVELTY`), not a new number.

### BAR 3 — Deletion B, the REFUSAL half
After `punch_hole(red, cylinder)`: `hole` flag **= 1.0000** and refusal **≥ 0.99** on the 720
deletion-set requests and on a constructed red∧cylinder probe, on every seed.

### BAR 4 — Deletion B is SURGICAL (the trap test)
On every seed, after the same hole:
- **(a)** `green ∧ sphere` and `blue ∧ oblong` — never seen, not holed — `hole` flag **= 0.0000** and
  refusal changed by **≤ 0.01** absolute.
- **(b)** **every other term pair the pool never held** (`coupling_w == 0`): `hole` flag **= 0.0000**,
  refusal changed by **≤ 0.01**. This is w5-common §3.7's "96 unseen conjunctions still answered",
  now scored. The exact count is reported, not assumed.
- **(c)** `red ∧ sphere` and `red ∧ oblong` (red's other shapes, **seen**) stay ALLOWED: refusal
  changed by **≤ 0.01**. This is what separates the hole from the eviction.
- **(d)** paired Δ on the 89 survivor corners **= 0.0000 exactly**, with the rendered features
  asserted **bit-identical** pre/post. `punch_hole` touches no segment the read consults.

### BAR 5 — The byte property, on the SCORED store (not the val demo)
For both deletions, on every seed: bytes changed **outside** the permitted ranges = **0** — exact,
no tolerance — and `assert_no_refit` passes. The hole is **1 byte**. The negative control (a 1e-3
nudge to one address-book row, the smallest possible refit) moves ≫ 0 bytes, proving the instrument
is not blind. ⛔ If any byte moves outside, the central property fails and the row dies.

### BAR 6 — The 8-byte gate, printed beside me
- **(a)** On the **value** deletion the gate **succeeds**: clearing red's bit refuses red everywhere
  at 1.0000 for **1 bit**. ⭐ **I expect to TIE the gate here and I declare that now.** Charter §C.3
  then governs: a tie counts only if the same object also delivers *deletion that changes the output
  exactly* — which is BAR 5, and which the gate does not have (the gate changes no store byte).
- **(b)** On the **conjunction** deletion the per-factor gate changes **ZERO bits** — asserted by
  byte-diffing the 8-byte bitmap — and refuses red∧cylinder at the **base rate**. It cannot express
  the deletion at all.
- **(c)** The **joint** `object_hue × shape` gate that *can* express it refuses **every** row-3
  request and every row-4 request whose conjunction the pool never held. Reported as a measured rate.
- ⭐⭐ **The claim is the pair (b)+(c): a gate can have the conjunction deletion or compositional
  generation, not both.** Neither half alone is the claim.

### BAR 7 — Table row-removal
The prototype table **refitted on my own pool** (so it is like-for-like; the shipped rival is on the
Montero combination population and `row_eligibility.json` says so).
- **(a)** Value deletion by row-removal works; **I expect to tie it** on the refusal half.
- **(b)** Conjunction deletion: the marginal table has **no conjunction slot**; **zero bytes** of its
  prototypes change — asserted by byte-diff. It **cannot enter** BAR 3/BAR 4.
- **(c)** Its composition Δ on the 89 survivor corners is reported.

### BAR 8 — Retrain-from-scratch, the GOLD STANDARD, with its cost line
Refit from scratch on the pool minus **every** row carrying red: the prototype table, and a
size-matched MLP. Behavioural comparison on both halves, plus **the cost line**.
⛔ **FALSIFICATION (from the task file):** if retrain-from-scratch is indistinguishable from eviction
on **every** measure, the mechanism buys only time, and I report exactly that.

### BAR 9 — Churn: a COST LINE only, never a claim
**One definition, fixed here:** *churn = the fraction of the 89 survivor corners' 4,968 requests whose
judged 6-tuple changes between the pre-deletion and the post-deletion read.* Reported for every
method. ⛔ **Never quoted as evidence for or against any arm**, in either direction. Wave 3 measured
99.5 % and noted that near-total churn is also instability.

## 3. Rivals, and who gets a dash

| rival | enters the refusal half? | enters the composition half? | note |
|---|---|---|---|
| **field store** (this arm) | yes — `empty_basin` / `hole` | yes | the object under test |
| **table, row-removed** | yes (value only) | yes | refitted on MY pool; **cannot** express a conjunction |
| **table, retrain-from-scratch** | yes | yes | ⭐ **GOLD STANDARD**, with its cost line |
| **8-byte per-factor gate** | yes (value); **zero bits** (conjunction) | n/a — it is a gate, not a generator | printed beside me |
| **joint hue×shape gate** | yes (conjunction) | ⛔ measured to refuse the row-3/4 requests | the trap, measured |
| **MLP / transformer / LSTM** | **—** | — | **DASH + FOOTNOTE.** No refusal verb: they always emit. Deletion is available to them *only* by retrain-from-scratch, whose cost is on the cost line. ⛔ A dash is never a win for us. |
| **LACE / PoE-EBM** | **—** | — | **DASH + FOOTNOTE.** Sampler draws, not functions of the request (spoke A's own API refuses them); and 95.1× / 27.1× the store budget. |

## 4. Charter §C floor — one line each, on every scored row
1. **Matched-bytes lookup** (atom-mean / kNN on the same atoms) on both halves.
2. **Landscape-removed** (atoms zeroed) and **landscape-shuffled** (atom coordinates scrambled) on
   every field-store row; both must collapse.
3. **§C.3 tie clause:** where I tie the gate or the table, the tie counts only via *deletion that
   changes the output exactly* — BAR 5.
4. **§C.4:** nearest-atom distance of the settled state is reported; if it → 0 the arm is a lookup
   table and is reported as one.

## 5. ⛔ What would prove this hub wrong — each mapped to a measurement

| the task file's falsifier | the measurement that decides it |
|---|---|
| retrain-from-scratch indistinguishable from eviction on every measure | BAR 8 |
| deleting red breaks green/blue composition | BAR 2 (fails if Δ < −1.0) |
| the hole refuses a never-seen conjunction that must stay allowed | BAR 4(a)/(b) (fails if `hole` flag > 0 or refusal rises > 0.01) |
| eviction requires a refit anywhere in the path | BAR 5 (`assert_no_refit`) |
| eviction is behaviourally indistinguishable from a gate bolted onto the table (the **DIAL** falsifier) | BAR 6 + BAR 7 together, on **both** halves |

## 6. Expected directions, declared in advance — including against myself
- BAR 1, 3, 5: **pass**. The field-store hub demonstrated the mechanism unscored; this run moves it
  to the scored corners, which is not free but is not a coin flip either.
- BAR 2, 4: **pass**, with BAR 4(d) exactly 0.
- BAR 6(a), BAR 7(a): ⭐ **I expect to TIE — the 8-byte gate and the table both delete a whole VALUE
  as well as I do.** I say so before the run. The value deletion is **not** where this row is won.
- BAR 6(b)+(c): **the conjunction is where it is won or lost**, and it is the only place where I
  expect a separation no cheaper object can reproduce.
- BAR 8: I expect retrain-from-scratch to match eviction **behaviourally** and to cost orders more.
  ⚠ If it also matches on cost, my dial is falsified.

## 7. Budget
Scored run **1 of 3** for target `deletion`. Runs 2 and 3 remain banked. Replications of THIS PREREG
are free (§A.2) and one independent replication on a clean checkout is required before any number is
spoken (§A.5). All three runs are reported; the best is never reported alone.
