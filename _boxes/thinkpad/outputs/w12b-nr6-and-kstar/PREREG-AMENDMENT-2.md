# AMENDMENT 2 to `PREREG-w12-single-design-crossover-1.md` — w12b hub/nr6-and-kstar

**Appended, never an edit (charter §A.3).** Timestamp in `PREREG-AMENDMENT-2.timestamp.txt`,
written BEFORE either ledger row and BEFORE any process that produces a number.

⛔ **THIS IS A SAME-PREREG REPLICATION AT FURTHER SEEDS AND ONE FURTHER K. IT IS NOT A NEW
DESIGN AND IT MOVES NO BAR.** The governing document is
`/c/chlu/outputs/w12-single-design-crossover/PREREG-w12-single-design-crossover-1.md`
(including its own AMENDMENT 1, §A1.1–A1.4), read and NOT modified — that hub's output
directory is read-only to this hub. Everything below is carried over unchanged:

| carried over unchanged | value | source |
|---|---|---|
| estimand | `distinct_valid_novel_tuples` **per corner**, corner = the unit | PREREG §4 |
| verdict function | `exp_gen_factor_wells.paired_estimand`, never a single cell | PREREG §4 |
| margin | `DECORATION_MARGIN_NOVELTY = 1.0` novel tuple / corner | PREREG §4 |
| control gate | `landscape_removed`, `landscape_shuffled`, `landscape_shuffled_across_wells`, `random_atoms` — **all four re-run on TEST on every row, and no row is claimed unless all pass** | PREREG §5 |
| rows scored | row 3 (60 pair sub-corners) + row 4 (49 n-way corners), TEST | PREREG §4 |
| comparators | the **524,288 B REFERENCE column** (`param_target = 131,072`); the 1× column stays a **DECLARED NOT-RUN** | AMENDMENT 1 §A1.2 |
| K\* | **32**, read from `CROSSOVER-val.json`, chosen by a rule frozen 2026-09-21T11:10:24Z | PREREG §3 |
| object | `CovContentStore`, `ward_balanced` label-free cut, `A = min(64, min_k d_k − 1)` | PREREG §2 |
| four-part ledger | atoms · covariance+count · address book · coupling/hole, **summed, never netted** | PREREG §6 |

---

## A2.1 ⭐ What is genuinely new — three things, and nothing else

**(i) Run 2/3 buys the K = 5 number on TEST.** The crossover hub declared it a NOT-RUN in its
own words: *"the third pre-declared falsification is untestable while only K\* has a TEST
row."* Without a second K on TEST the val/test gap cannot be checked at all, because a gap
needs two points on the same condition. K = 5 is the **sweep's own endpoint** and the K the
whole programme's published rows were built at; it is not a newly chosen K and no selection
is made here. ⛔ The range is not extended and no third K is added to manufacture a trend.

**(ii) Run 3/3 adds seeds 1 and 2 at K\* = 32 on TEST**, so T7 carries **3 seeds** instead of
one. Same K, same object, same corners, same instrument. ⛔ A pure replication.

**(iii) `transformer` and `lstm` are RE-FITTED in-process and scored as two EXTRA arms.**
Wave 12 measured that `w4_learners.train` cast token ids to `float32`, after which
`eqx.nn.Embedding` under jax 0.9.0 raises *"Indexer must have integer or boolean type"* — so
the two `tokens` families **could not be fitted at all** before the repair. The repair
(`_as_model_input`) is in this base at `1635e0a`, an ancestor of `e58ab05`; the second,
functionally equivalent repair `8e5f202` lives on `w13/encoder-adapt` and is **not** in this
base. The two families are therefore re-fitted here from scratch, at the **same** 131,072-
parameter pin and on the **same** fit pool the loaded comparators used, so that every number
this hub prints for them **post-dates the fix by construction** rather than by assertion.
⛔ The loaded comparators are kept and printed beside them; nothing is replaced.

## A2.2 ⛔ THE SEED COUNT IS 3, RE-VERIFIED TODAY, AND IT IS NOT PADDED

AMENDMENT 1 §A1.1 ruled seeds 3 and 4 out. The cache situation has **changed** since — both
seeds now pass the cache admission rule — and the ruling is therefore re-measured rather than
inherited:

| seed | `CACHE-READY.json` entry | judge instrument |
|---|---|---|
| 0, 1, 2 | (not in that file — the shipped gen-2 cache) | decoder **and** external classifier present |
| **3** | `array_shape [480000,128,4,4]`, `n_blocks 59`, 59 block hashes, `status READY` — **PASSES** | ⛔ `decoder_v1_s3.pkl` **absent**; `gen-external-classifier/seed_3` **absent** |
| **4** | `array_shape [480000,128,4,4]`, `n_blocks 59`, 59 block hashes, `status READY` — **PASSES** | ⛔ `decoder_v1_s4.pkl` **absent**; `gen-external-classifier/seed_4` **absent** |

⇒ ⭐ **Both seed 3 and seed 4 now clear the cache rule and NEITHER can be judged.** The task
file anticipated this for seed 4 only; measured here, it holds for **seed 3 as well**, for the
same reason and at the same two paths. ⛔ **The count is 3 (seeds 0, 1, 2), printed on every
row, and it is never padded to 4 or 5.** A number produced on a different decoder and a
different classifier is a number on a different instrument, and this programme's rows are
comparable only because one instrument produced all of them.

## A2.3 What would falsify the rows this amendment authorises

1. ⛔ **The controls fail to collapse at K\* on a scored row** ⇒ the store is decorative there
   and **no row is claimed**. This is the gate, unchanged, and it is checked before any margin.
2. ⛔ **The K = 5 TEST number contradicts the K = 5 val number** beyond the 0.02 band the
   selection rule used ⇒ the val sweep selected on a set that does not transfer, and K\*'s
   provenance is reported as damaged.
3. ⛔ **The extra seeds move the K\* row outside the seed-0 result's own spread** ⇒ T7's
   headline is a one-seed artefact and is reported as such.
4. ⛔ **A re-fitted token family scores far above its loaded twin** ⇒ the loaded comparators
   are pre-fix artefacts after all and every row built on them is weakened; reported as a
   correction, not buried.

## A2.4 Budget

`single-design` stands at **1/3 spent** (`/c/chlu/SCORED-LEDGER.md:39`). This amendment
authorises **runs 2/3 and 3/3 and no more**. Both ledger rows are written before either
process starts. ⛔ The K\* ladder replication in this hub's second job is a **replication**:
free, no row, no budget.
