# PREREG — w4 hub/unseen-valid — SCORED RUN 2 of 3

**Written 2026-09-19T20:48Z, BEFORE any number from these objects touches the held-out set.**
Hub `w4/unseen-valid` · base `base/iter4 = 9dfed5e` · target **row 2 (unseen-valid)**.
⛔ Charter §A.3: bars in this file are fixed once a result is seen against them. Amendments append only.
⭐ Runs spent before this one: **1 / 3**. This makes **2 / 3**.

---

## 1. Why a second run at all

Run 1/3 falsified this hub's DIAL: a continuous-input MLP matched to my arm's own deployed size beats
the arm on all three seeds (mean paired delta **−0.0192 ± 0.0028**, every CI95 below zero).
⛔ **That falsifier is MY OWN CODE.** A referee can fairly ask whether my MLP is simply a good MLP and
my arm a badly-wired one. ⭐ This run answers that with an **independently built** MLP.

`w4-spoke-a` fitted, **on my exact value-holdout pool** (131,072 row ids I exported to them), at **my
protocol derived from my formula rather than imported from my code**, on the **generation-2** cache:

| object | params | role here |
|---|---:|---|
| `cont_mlp_notune_572k` | 573,440 | ⭐ **PRIMARY** — independent replicate of my falsifier |
| `cont_mlp_notune` | 118,153 | secondary — implementation agreement at matched budget |
| `table` | 118,784 | tertiary — a CONFIRMATION whose answer I have already derived |

Training agreement already established (⛔ **not** a corner result): 573k MSE mine 0.01085 / 0.01188 /
0.01306 vs theirs 0.01094 / 0.01209 / 0.012682 — within 0.8 % / 1.8 % / 2.9 %.

## 2. The instrument, estimand and margin — ALL UNCHANGED from PREREG-1

Second judge (oracle-referenced raw hue-logit difference), `e = |p − 0.5|`, 4,096 requests
(records 0…4095, superset of the frozen file asserted), the same **70-corner** disjoint tiling, the
same paired statistics, seeds **0, 1, 2**, **cache generation 2 only**.

### **MARGIN d = +0.0300, UNCHANGED.** Same rule: BEATS iff `mean_delta > d` AND `CI95_lo > d`, all seeds.

⭐ The measured SPAN it sits inside is unchanged and already published in PREREG-1: **0.3218 ± 0.0065**.

## 3. Pre-scoring ASSERTIONS (run in code, before any number)

1. Each checkpoint's `population.n_per_value` has **`object_hue[5] == 0`** AND **`shape[0] == 0`**.
2. `n_rows_fit == 131_072`.
3. The exported request values equal mine element-wise (`np.allclose`).
4. Corners realised **≥ 49** (expect 70); records 0…511 byte-identical to the frozen `row2_unseen.json`.
⛔ Any failure aborts the leg and it is reported as NOT RUN, not as a number.

## 4. Expected direction, declared in advance

1. ⭐ **PRIMARY — `spokeA_cont_mlp_notune_572k` BEATS my arm**, i.e. paired delta **negative**,
   reproducing run 1's **−0.0192** in sign and roughly in size. ⛔ **I expect my own falsification to
   replicate under someone else's implementation.**
2. `spokeA_cont_mlp_notune` (118k) vs **my** `cont_mlp_TABLEMATCHED` (118,153): **a TIE**, |delta|
   inside the margin. ⚠ Declared in advance: **both implementations show a seed-2 jump** on the small
   model (mine 0.03413 / 0.03497 / **0.04516**; theirs 0.03532 / 0.03515 / **0.04333**), so a seed-2
   discrepancy is **the small model being seed-sensitive**, ⛔ NOT an implementation difference.
3. `spokeA_table` (vhpool) vs **my** `knn_proto`: **AGREEMENT, to numerical noise.** ⭐ I already proved
   this by direct array comparison — the two fitted objects match to **max |Δ| = 2.42e-05** — so this
   leg is a **CONFIRMATION of a derived identity**, and ⛔ **a null here is the expected result, not a
   failed refit.** It is reported, never claimed.

## 5. ⛔ What this run may NOT be used to claim

* ⛔ It cannot rescue the dial. The dial is **already falsified** by run 1 and this run can only
  **corroborate or complicate** that, never reverse it — a tie here would mean the independent MLP
  fails to reproduce my result, which I would report as **weakening my own falsification**.
* ⛔ It is **not** a win for my arm if the independent MLP happens to score worse than mine. An arm that
  beats one implementation and loses to another has not beaten the rival class.
* ⛔ **Zero-supervision structure is not claimed.** Labels are used ONCE at fit time to align wells and
  at the address book (§5), declared in the caption of every table.
* ⛔ The `landscape_removed` control is a **literal constant** (1 distinct render / 4,096 requests) that
  nonetheless scores at chance and wins 41/70 corners against the arm. ⭐ The distinct-output column
  goes beside **every** accuracy in this run, per §1 — the arm and the matched MLP were measured
  **equally diverse** (4073 vs 4061 distinct renders), which is why run 1's margin is read as clean.

## 6. Stop rule

⛔ If the assertions in §3 fail, or the exported features do not match my requests, the leg is **NOT
RUN** and is reported as such. ⭐ A declared NOT-RUN is never a null. Budget after this run: **2 / 3**.
