# INHERITED FACTS — what the three PREDECESSOR hubs measured, and what it does to my design
**2026-09-17, T+0:09.** Compiled by a sub-agent on Opus from the predecessor hubs' reports and the
frozen Mac manifests, every number carrying its source path and JSON key. ⛔ No concurrent peer's
outputs were read. ⭐ These are inherited facts, cited not re-run.

⚠ **Read this before the report is written. Four of these facts change what my headline can mean, and
two of them retire numbers the charter itself quotes.**

---
## 1. ⭐⭐ THE FREE FLOOR ON B IS 0.5278, NOT 0.3272 — and this is the single most important inherited fact
`hub-generation-table/LOG.md:122-167`. `eval_b` is a complete Cartesian product with modal factor
fractions `(½, ½, ½, ⅓, 1, ⅓)` — **`shape` takes exactly one value in B (cylinder)** — so **one constant
output** scores `mean(½,½,½,⅓,1,⅓) = 0.5278` with **1.00 distinct tuple of 72**. Realised empirically
(`LOG.md:255-260`): emitting one real corner feature 72 times scored **0.5278 to the last digit**.
⭐ The charter's *"a 2.67-tuple constant output scores 0.3272"* is what **one particular** constant
scores; **0.5278 is what the best constant scores.** ⛔ I quote 0.5278 as the floor.
⚠ **Consequence for me: the honest dynamic range on B is 0.5278 → 0.8650 (`knn_proto`), width 0.337,
and the shipped CLU arm's 0.5949 sits +0.0671 into it.** Six of the ten published rows lose to emitting
the same picture 72 times.

## 2. ⭐⭐ THE PRE-DELETION CLU ARM ON B, PER FACTOR — my headline's instrument, and it is a poor one
`gen-read-cell-arm/runs/cell/seed_0/seed_0/cell_manifest.json`, `arms.clu.external.composition_B`:

| factor | CLU seed 0 | `knn_proto` seed 0 | modal (free) fraction | `store_removed` seed 0 |
|---|---|---|---|---|
| floor_hue | **1.0000** | 1.0000 | 0.5 | — |
| wall_hue | **1.0000** | 1.0000 | 0.5 | — |
| object_hue | **1.0000** | 1.0000 | 0.5 | — |
| scale | **0.0417** | 0.1667 | 0.333 | **0.3333** |
| shape | **0.5000** | 0.9861 | **1.000** | — |
| orientation | **0.0000** | 1.0000 | 0.333 | **0.3333** |
| aggregate | 0.5903 | 0.8588 | 0.5278 | — |
| exact_match | **0.0000** | 0.1667 | — | 0.0000 |

⛔⛔ **Three of six factors are pinned at 1.0000 (ceiling — they can only fall), and two are at or below
the free floor.** `scale = 0.0417` and `orientation = 0.0000` are **worse than the 0.3333 a constant
gets, and worse than the fully-ablated `store_removed` store**, which scores 0.3333 on both.
⭐ **So on B the CLU arm has real signal in exactly one place — the three hues — and it is saturated
there.** A collateral measurement on B therefore has almost no dynamic range in either direction.
⚠ **This is a measured limitation of the pre-registered scored object, found before the scored run was
fired.** It is recorded here, and it is why `dev` (n = 4096) carries the DEV headline while B (n = 72)
remains the pre-registered scored cell.

## 3. ⛔ THE CHARTER'S "codebook costs 90.5 % of `scale` and 100 % of `orientation`" IS UNSOURCED — and two re-derivations disagree
The figure appears only as an assertion (`autonomous-hub-charter.md:103`, repeated in two task files).
**No file under `.claude/outputs/**` contains the arithmetic.** Two independent re-derivations:
- `hub-composition/LOG.md:788-796` on `val`: `scale` **75.5 %** of ceiling lost, `orientation` **82.6 %**.
- From the seed-0 store manifest: `scale` `(0.8594 − 0.2119)/0.8594` = **75.34 %**;
  `orientation` `(0.9238 − 0.1689)/0.9238` = **81.71 %**.
- The companion claim *"partition 9.5 % / 0 %"* also fails: the seed-0 codebook/partition split on
  `orientation` is ≈ **89 % / 11 %**, ⛔ not 100 % / 0 %.
⭐ **I will not quote 90.5 % / 100 %.** I quote the manifest-derived 75.3 % / 81.7 % with the key they
came from, and I say the charter figure is unsourced. ⚠ The *direction* of the charter's point survives
intact and is what matters to me: **`scale` and `orientation` are largely destroyed before any
deletion**, so "unrecoverable after deletion" is **VACUOUS** for them (PREREG §5.1, falsifier 2).
⭐ On B it is worse than vacuous — the live store is *below* the ablated store on both.

## 4. ⭐ COMPOSITION LOST, AND BADLY — which cuts both ways for me
`hub-composition.md`, `RESULTS-LEDGER.md` (0 of 3 scored spent; target NOT MET):
CLU **exact_match 0.00 / 72, 9.00 ± 0.82 distinct tuples**; `knn_proto` **14.33 ± 4.78 exact,
56.00 ± 3.74 distinct**; decoder ceiling **67.67 ± 2.49**. R1 verdict vs `knn_proto`: **LOSS**.
Measured capacity law `rel-err = e^1.346 · bytes^−0.1651`, **R² = 0.9960** over a 128× span ⇒ clearing the
bar needs **≈ 77 GB ≈ 147,000×** the deployed 524,288 B; merely **matching `knn_proto` needs ≈ 162×**.
Diagnosed obstacle: **self-alignment of the quantiser's residual** — at one fixed rel-err, `x − e`
scores 162/512 while `x + e` (what the store actually emits) scores 10/512, replicated exactly.
⚠ **Bearing on my conjunction test:** the gs charter warns that *the better composition works, the
harder unrecoverability becomes*. ⭐ **Composition does not work on this store at all** — zero correct
held-out tuples, ever. So a "the conjunction re-composed" finding cannot be reported as a strength of
the composer; it has to be measured directly and reported for what it is.
⚠ A replication note (`hub-composition/replication/REPLICATION.md`) **refutes** the identity
`distinct_valid_novel = 72 × exact_match` as a derivation — only `≥` follows — and flags `knn_proto`'s
14.33 as a possible **under**-count. ⛔ So I do not lean on that identity anywhere.

## 5. ⭐⭐ REPORT I, CORRECTLY SCOPED — thread count is ELIMINATED, and the batching result is about the ENCODER
`_hub_scratch/reports/I-numerical-divergence.md`. The quote my LAUNCH gives me, §4 lines 209-212:
> **Changing only the batching of an encoder forward pass — same machine, same binary, same weights,
> same inputs, same second — produces a different `store_hash`.** … it pins **one particular XLA CPU
> lowering of a float32 convolution.**

⚠ **Two scoping corrections I must carry:**
1. **That batching is the ENCODER forward pass, ⛔ not `train_store`'s training batch**, and the store
   involved was tiny (`well_sizes [8, 23, 225]`). ⛔ I must not quote it as a statement about
   `train_store` batch shape. Magnitude was `max|Δ| = 3.576e-07 ≈ 3 float32 ULP` on 96/8192 floats, and
   ⭐ *"everything scientific downstream is invariant: the channel grouping, the well sizes, the address
   codes and the byte ledger are identical across all four variants."*
2. ⭐⭐ **Thread count was ELIMINATED upstream** (`I-numerical-divergence.md:342-350`): three runs at
   `OMP/MKL/OPENBLAS = 4, 1, 4` gave **all twelve reported hashes byte-identical**, store_hash
   included. Python 3.11.16 vs 3.12.10 likewise byte-identical.
   ⇒ AUDIT-01's top-ranked risk (BLAS thread count perturbing `channel_correlation` and flipping a
   greedy linkage merge) **has already been probed upstream and did not fire**. ⛔ That is prior
   evidence about a different store at a different scale; it does **not** discharge my gate, which is
   the whole point of running it. But it lowers the prior, and it means a thread-sensitivity probe
   (`det_C`) is **low value** and I deprioritise it.
⭐ Confirmed independently: **`store_manifest.json` carries no `store_hash` field at all** (I grepped
all three seeds). There was nothing to compare against, which is why a digest had to be defined.
⚠ Report I's open question 2 — *"is `store_hash` an over-specified fingerprint?"* — is **still
unanswered**; ⛔ not mine to answer (LAUNCH §-1: log it, the coordinator escalates).

## 6. ⚠ THE RIVAL'S TWO DECLARED ADVANTAGES — both must appear beside my headline
From `cell_manifest.json`, `arms.knn_proto.byte_ledger` and `arms.knn_proto.mechanism`:
- **Bytes:** `knn_proto` **475,136 B** vs CLU store **524,288 B** ⇒ the rival is **90.6 %** of my bytes,
  i.e. the *smaller* model — conservative against me. ⚠ But its shipped configuration *"was selected at
  the retired 688128 B budget — a ~31 % byte advantage it should not have (R26.3)"*, measured
  **byte-invariant** (Δ = 0.000000, 3/3 seeds).
- ⛔⛔ **Channel asymmetry, stated in the manifest itself:** `knn_proto`'s **whole D-float vector reaches
  the decoder — "an UNBOUNDED channel, ⛔ never compared in the same column as 30 bits"**. The CLU's
  `channel_bits = 30.0`; `knn_proto`'s is `null`. ⚠ My headline is a **difference of differences**
  (collateral), so the level asymmetry largely cancels — ⛔ but it does not vanish, and it is reported.
- ⚠ The CLU's **addressing head (74,240 B)** and **inter-well coupling (163,840 B)** are *excluded* from
  `store_bytes` by the shipped convention. ⭐ Stated, not silently accepted.

## 7. The laundering-control deltas already in hand (seed 0, `gen-store-grouping/.../store_manifest.json`, `contrast`)
`delete_mean_delta_aggregate` **0.4736**, `delete_zero_delta_aggregate` **0.4714**,
`delete_scramble_delta_aggregate` **0.4344** (correlation grouping; random grouping ≈ 0.32/0.30).
Post-ablation absolutes on `eval_a` (n=1024, `exact_match = 0.0` for all three): `mean` **0.1237**,
`zero` **0.1260**, `scramble` **0.1629**. ⭐ This is the pre-deletion laundering control, in hand.
⛔ It is **not** my delete operator and is never reported as one.

## 8. Traps re-confirmed
- ⛔ `gen-arms-a7-wiring/refusal_scores.json` is the SUPERSEDED broken-split file. ⚠ **Its `what`,
  `legs`, `controls`, `seeds_present`, `not_run: []` and `byte_target_checks` blocks are byte-for-byte
  identical to the good `gen-val-split-fix` file** — only the `schema` prefix and the size differ, and
  `status=complete` is true of both. ⭐ **Discriminate on the DIRECTORY, never the basename.** I quote
  neither; I have no use for refusal scores.
- ⛔ Instrument collisions: `lace` is 0.3117 on the CELL harness and 0.3241 on BASELINES at 688,128 B —
  both real, different instruments. `knn_proto` refusal AUROC differs by byte leg (0.8457 vs 0.8358).
  ⛔ No leg-shopping.
- ⚠ Live retractions on record from `hub-generation-table`: the PCA control was unfair and the fair
  affine control **reverses** it (Δ = −0.0219) — *"the store is not just any 832-dimensional subspace"
  is WITHDRAWN*; a +0.134 became +0.0147 on six corners; a "TIE (+0.0072)" at ten corners became a
  **LOSS (−0.0172)** at 49 corners. ⭐ The pattern in every case is **too few corners**. My `dev` set is
  n = 4096 and my targeted sets are 1024 each, specifically so I am not the next entry on that list.
- ⚠ `K` must be reported with any §C.2-style result (a soft read over a fine partition gets `K` free
  scalars). **I am at K = 5**, so I am clear — and I report it.
- ⛔ **Machine-mixing:** `_hub_scratch/CONTEXT.md` names `hub/deletion` explicitly as one of the two
  hubs most likely to want to rebuild. ⭐ **My `knn_proto` is re-fitted in my own process on my own live
  row vector; the published 0.8650 is a regression check, ⛔ never my bar.**

## 9. Free provenance, already matched
`canonical_build`'s first training step prints `loss = 47.49073`; the Mac manifest records
`arms.correlation.training.loss_first = 47.4907341003418`. Identical to all five printed digits, on
Windows, from the Mac-written cache. ⭐ R3 provenance that my rebuild path is the shipped path.
⛔ **Not a parity claim** — §4 of LAUNCH-COMMON already fixes parity at ~3 ULP, not bit-identity.

## 10. Three things the sub-agent reports as NOT FOUND (⛔ stated, not inferred)
1. The arithmetic behind **90.5 % / 100 %** (and its "partition 9.5 % / 0 %" companion).
2. The raw artefacts behind the **0.3272 / 0.3927 / 0.5008 / 0.6852** rows — task-file text only.
3. Any statement of **what was actually broken** in the superseded split. Only the labels
   "broken-split" and "repaired split" exist.
