# PREREG — hub/factor-wells — scored run 1

**Written 2026-09-19, BEFORE the first number on any held-out corner.**
**Target:** compositional generation, scoreboard **rows 3 and 4**, with FACTOR-ALIGNED wells.
**Budget:** the `composition` target — 1 spent before this hub, **2 left**. This is **scored 2/3**.
**Charter §A.3:** I may not change a bar I have already seen a result against. Amendments are appended
below, dated, never edited.

---

## 1. The claim, in one sentence

Partitioning the store's channels **by the factor they carry** — wells carry a factor, modes are its
values — produces more distinct valid novel factor tuples per held-out corner, on rows 3 and 4, than the
same object partitioned by channel correlation, at an identical byte ledger.

## 2. The primary arm and THE like-for-like

| | arm | control |
|---|---|---|
| name | `factor_aligned_balanced` | **`basis_head`** |
| partition | argmax of `I(bin(channel); factor)` on the fit pool, rebalanced to equal wells | the shipped **ward** correlation cut |
| K | 5 | 5 |
| modes M | 64 | 64 |
| D | 2048 | 2048 |
| store bytes | **524,288** | **524,288** |
| bits per read | **30.0** | **30.0** |
| reachable read rank `Σ_k min(d_k, M)` | **320** | **320** |
| read verb | signed linear span | signed linear span |
| head | request-conditioned ridge, pair-interaction block | identical |
| ridge λ | 1.0 | 1.0 |
| head bytes | 1,766,400 | 1,766,400 |
| fit pool | identical | identical |
| decoder / judge | identical, hash-asserted | identical, hash-asserted |

**The label vector is the only thing that differs.** ARI(`factor_aligned_balanced`, `ward`) = 0.337.

`knn_proto` is the **second** row (w3-hub-common §1). Its prototype table is 475,136 B against the
524,288 B budget and it carries a `budget_note` declaring that, plus the standing declaration that the
shipped `knn_proto` configuration was selected at the retired 688,128 B budget.

## 3. ⭐ THE MARGIN, DECLARED IN ADVANCE

**Estimand:** the **mean paired per-corner difference** in `distinct_valid_novel_tuples`, over
**49 n-way corners (row 4)** and **60 tiled pair sub-corners (row 3)**, `factor_aligned_balanced` minus
`basis_head`, corner by corner in a fixed order.

**Margin: `+1.0` distinct valid novel tuple per corner.** One whole extra distinct valid combination per
corner, on corners of 72 requests (row 4) and 36 requests (row 3).

**The arm beats the control only if BOTH hold:**
1. `mean_paired_delta > 1.0`, **and**
2. the lower end of the 10,000-sample paired-bootstrap 95 % interval is also `> 1.0`.

A two-sided sign test over the corners is reported beside it. Rows 1–2 carry a declared accuracy margin
of `+0.02` absolute and are **reported, never claimed**.

**⛔ If the estimand does not clear the margin, the verdict is written in these words: *a CLU mechanism
that ties basis+head is decoration and is reported as such.*** `gen_scoreboard.verdict_vs_control`
writes that sentence itself; I do not get to phrase it.

**⛔ No single cell decides anything.** A cell may be reported beside the estimand; the estimand is the
verdict.

## 4. Expected direction

I expect `factor_aligned_balanced` **to beat** `basis_head` on rows 3 and 4, and to beat its own
size-matched random floor by more than it beats `basis_head`.

⚠ **The measured reason I might be wrong, recorded before the run:** the frozen encoder barely carries
`shape` and `scale` in separable channels — only **12** of 1984 live channels are top-ranked by `shape`
and **4** by `scale`, and the best single-channel MI is 0.250 b for `shape` and 0.317 b for `scale`
against 2.066 b for `floor_hue`. A factor-aligned partition of *this* encoder therefore spends three of
five wells on hue. If the arm ties, that is the most likely mechanism, and I will say so.

## 5. THE STORE FIT, PINNED COMPLETELY (w3-hub-common §3)

| knob | value |
|---|---|
| partition | the label vector above; every arm's is written to `run/partitions.npz` |
| fit pool | 131,072 rows, `np.random.default_rng(20260919).permutation` of the value-holdout **train** index, first 131,072 |
| store fit rows | the pool's **first 32,768** (the shipped `gen_store.n_store_train`) |
| mode init | `gen_store.init_store` — k-means++ seeding, `default_rng(seed * 2654435761 mod 2^32)`, seed 0 |
| **Lloyd budget** | **0 sweeps** — the shipped path is k-means++ *seeding only*, then adam |
| optimiser | `optax.adam(lr = 0.003)`, 1200 steps, batch 128 |
| commitment β | 0.25 |
| τ anneal | geometric 1.0 → 0.1 over the 1200 steps |
| minibatch RNG | `jax.random.PRNGKey(0)`, `split` then `choice(n, (128,), replace=False)` per step |
| **head ridge λ** | **1.0**, end-to-end closed form (Sylvester, simultaneous diagonalisation) |
| head design | bias + 57 main-effect one-hots + 1322 pair-interaction one-hots = **1380** columns |
| relevance binning | 8 quantile bins, edges from a 65,536-row subsample, `default_rng(seed * 1000003 + 20260918)` |
| merge rule | `min_relevance_mass` (6 factors → 5 wells) |
| balance rule | channels by relevance **margin** descending into their best group with capacity `⌊/⌉ D/K` |
| dead channels | → lowest-occupancy well, the rule `gen_store.correlation_grouping` already uses |
| decoder | `decoder_v1_s0.pkl`, param hash asserted against `winner_hashes.json` |
| judge | `gen-external-classifier/seed_0`, param hash asserted against its own manifest |
| seed | **0 only** (w3-hub-common §5: seeds 1–2 are not produced this wave) |

⚠ **One declared deviation from the shipped code.** `gen_store.train_store` passes `τ` to an
`eqx.filter_jit` step as a Python float, which `filter_jit` treats as static; τ moves every step, so the
step recompiles 1200 times (measured: 1.33 s/step, 27 min per arm, 14 arms). I use
`train_store_traced_tau`, which is the same loss, optimiser, anneal and minibatch RNG stream with τ
traced. Measured side by side at 40 steps: max |loss difference| 0.597 on a loss of ~860 (0.07 %),
reconstruction MSE 1.68627071 vs 1.68607640. **It is not bit-identical and I do not claim it is.** It is
the same fitter for every arm, so it cannot favour a partition.

## 6. Selection happens on `val`, and NOWHERE else (charter §A.2)

Every mechanism choice — which partition, whether to balance, the pair-interaction ablation — is made on
a **selection condition that costs no scored budget**: 24 Montero-shaped corners taken from n-way slots
**49–72**, which the condition's builder had capacity for (120 slots) and never emitted, so their rows
sit in the value-holdout *train* index. Under `--condition val` those rows are removed from the fit
pool, so the store has never held one. Verified: 24 corners × 72 rows = 1,728 rows, all in train,
**disjoint from the 49 scored corners**.

## 7. ⛔ The ablation that runs BEFORE any scored run

The target file: *"The gain disappears when the head's pair-interaction block is removed ⇒ you measured
the head, not the partition. Run that ablation before any scored run."*

`factor_aligned_balanced` vs `basis_head`, **with and without** the pair block, **on `val`**. The
ablation's result is recorded in `LOG.md` before the scored run is launched. If removing the block
removes the margin, I say in the report that I measured the head and not the partition.

## 8. Every floor, on every row (charter §C, w3-hub-common §7)

| floor | arm | must |
|---|---|---|
| §C.1 matched-bytes lookup | `lookup_argmax`, `lookup_convex` — the SAME atoms, read as one atom per well / as a convex mixture | arm must beat them beyond seed spread, or the store is decorative and is reported as such |
| §C.2 landscape-removed | `landscape_removed` — atoms zeroed, ledger unchanged | must collapse |
| laundering (partition) | `random_dmatched_balanced` / `_ward` / `_factor` — the arm's own label vector permuted, size histogram held exactly | every partition number is a margin over this, never an absolute |
| §C.4 lookup-table check | `lookup_argmax` ties the arm ⇒ the arm is a lookup table and is reported as one | — |
| no-model rule | `grid_syntax_rule` on the refusal rows | no refusal number ships without it |
| metric falsifiers | `refuse_everything` / `accept_everything` | both must score 0 on the refusal pair |

## 9. All seven rows, and which are claimed

| row | claimed? |
|---|---|
| 1 seen · 2 unseen-valid | reported |
| **3 comp 2-way · 4 comp n-way** | ⭐ **CLAIMED — my target** |
| 5 comp refusal · 6 invalid refusal | reported. ⛔ w3-hub-common §7: refusal died on measurement; an 8-byte value-presence bitmap wins these at 1.0000. I use exactly that rule, identically for every arm, and quote **no** row-5/6 number as evidence for the store. |
| 7 deletion | reported. ⚠ Declared: under a shared-basis read, deletion cannot be an operation on **atoms** (charter §B.4) — no atom belongs to one value. My row 7 deletes `object_hue = 0.0` from the **address book** (head columns zeroed, presence bit cleared, store modes bit-identical, no retraining) and is reported as exactly that. This is the leading indicator `hub/refuse-delete` asked for. |

## 10. What this run spends, and what run 3 is for

- **Scored run 1 (this one, 2/3):** the whole table at once — every arm, all seven rows, 49 + 60 corners,
  on the test condition.
- **Scored run 3 (reserved):** the independent replication required by charter §A.5 — a separate spoke
  re-running this PREREG's winning configuration on a clean checkout of `hub/factor-wells`. If run 1
  exposes a defect in the instrument rather than a result, I may instead spend it on the repaired run,
  and that decision is written into `LOG.md` as a dated bullet and appended here as an amendment.
- ⛔ All scored runs are reported, including the ones that lose. The best is never reported alone.

## 11. Label-path audit (the thing that voids the claim)

Labels enter **once**: `I(bin(channel_d); factor_f)` on the store's own fit pool — the same quantity and
the same rows the address book is fitted from. `assert_relevance_is_train_only` fires if the fit pool
intersects the hold-out; it reported **0 shared rows**. No held-out row, no cell membership, no arm
score, no judged outcome and no learned rotation entered the partition.
⛔ If the arm beats correlation grouping only because the partition saw a label beyond the address book,
the claim is void — the audit is emitted as `run/partitions.json → label_path_audit`.
