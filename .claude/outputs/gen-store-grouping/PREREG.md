# PREREG — gen-store-grouping (R0 of the factorisation ladder)

Written **before** `chlu/core/gen_store.py` existed and before any grouping, store, probe or
delete-the-store measurement was run. Author: experiment-engineer. Base `main @ fcdca43`.
Feature cache: FROZEN, `gen-feature-bridge` `shapes3d/recon` seeds {0,1,2} (R18 — the store is
upstream of the decoder, so `gen-decoder-lift` cannot invalidate anything here).

## 0. DIAL DECLARATION (echoed)
- **Dial:** *isolation* — can a factored store be addressed and read without the read path routing around it.
- **Laundering control:** ⭐ random grouping at the same `K`, **same well-size histogram**, full arm on every number.
- **Falsifies:** random grouping matches correlation clustering within seed spread; **or** deleting the store
  leaves the read output unchanged.
- **Does NOT falsify:** low absolute read accuracy that is low for *both* groupings.

## 1. Design decisions FIXED here, before measurement

**D1. "Channel" = one of the `D = 2048` flattened penultimate coordinates** (`128 x 4 x 4`), not one of the
128 conv channels. The store reads the feature *vector*, so the partition is over the vector's coordinates.

**D2. `K` is chosen by the largest normalised eigengap of the channel-correlation affinity spectrum,
computed on TRAIN-split cached activations only, seed 0, over the a-priori candidate range `K in {4..16}`.**
The range is declared now and for this reason: at `M = 64` modes the read channel is `K*log2(M) = 6K` bits, so
`K >= 4` gives >= 24 bits (the Shapes3D factor grid needs `log2(10*10*10*8*4*15) = 19.4` bits) and `K <= 16`
keeps the well set enumerable. ⛔ No held-out quantity enters this choice. The realised `K` is whatever the rule
returns — I am NOT predicting its value and will not change the rule after seeing it.

**D3. `M = 64` modes per well, float32** => `store_bytes = M * D * 4 = 524,288 B` **exactly and independently of
`K`**, because the wells partition the 2048 coordinates. Declared now so `gen-baselines` can byte-match.

**D4. Read metric (primary):** probe top-1 accuracy on images obtained by decoding the **store read** of cached
A-split features through the ONE frozen decoder. Reported as `aggregate` **and** `shape` separately (bridge
finding: 4 of 6 factors sit at ~1.0 and carry no compositional information). **`shape` is the compositional term
(R16).** Secondary: feature-space reconstruction MSE.
Measured on the bridge's `eval_a` rows (1024, held back from every fitting stage). ⛔ No B accuracy, no C
verdict, no kill rule.

**D5. Addressing is HARD (one-hot per well) at read time.** The read output is a function of
`(mode indices, store)` only; the read function that produces the output **does not take the features as an
argument at all**. That is the structural no-bypass guarantee, and it is additionally measured (D6/P2).

## 2. PREDICTIONS (point + threshold). 3 seeds {0,1,2}, spread = population std over seeds.

**P1 — clustered beats random at the same `K` on the read metric.**
Mechanism: a hard product quantiser at fixed bit budget wastes bits on *cross-block* redundancy; putting
correlated coordinates in the same well minimises the residual cross-block correlation (the Optimized-PQ
argument). Random grouping spreads correlated coordinates across wells, so each well's sub-vector has higher
intrinsic dimension and 64 modes cover it worse.
- **Predicted:** `feature MSE` **10% lower** for clustered (threshold: **>= 5% relative**, and the margin
  exceeds 1 seed-spread).
- **Predicted:** probe `aggregate` on A **+0.04** for clustered (threshold: **>= +0.02**, margin > seed spread).
- **Predicted:** probe `shape` on A **+0.05** for clustered (threshold: **>= +0.02**). ⚠ `shape` on 1024 A rows
  is the noisiest column; if only `shape` fails while MSE and aggregate hold, that is a **partial** hold and I
  will say so, not round it up.
- ⛔ If random matches clustered within seed spread on **all three**, the grouping carries no structure and this
  spoke has failed its dial claim. That outcome is reported as the finding.

**P2 — deleting the store changes the read output a lot.**
`delete` = every mode vector set to zero (the store's content removed; the addressing head untouched).
- **Predicted:** with the store deleted the decoded image is a single constant, so per-factor accuracy collapses
  to the eval set's majority-class rate: `mean_f (1/card_f)` over `[10,10,10,8,4,15]` = **0.117**.
  With an intact store I expect `aggregate ~ 0.70-0.80`, so **predicted `delta_aggregate ~ 0.60`**.
- **Threshold: `delta_aggregate >= 0.30` and `delta_shape >= 0.10`.**
- ⛔ **`delta_aggregate < 0.01` is the programme's known failure mode (the block routing around the memory)
  arriving one stage early. If it fires I report it loudly as a FAILED TASK and do not repair-and-rerun.**

**P3 — leakage (the ladder x-axis).** `leakage(k) = 1 - transfer(k)`,
`transfer(k,f) = acc_heldout(k,f) / acc_matched_seen(k,f)` (R16 normaliser: same hue corner, seen shape),
probes trained per well on that well's sub-vector, on seen rows only, label-free store notwithstanding
(R1 clarification: probes are measurement).
- **Predicted direction: clustered has LOWER pooled leakage than random**, because a coherent well's sub-vector
  carries a factor's code intact and is less entangled with the factors that are novel in the held-out cell.
- **Predicted:** pooled leakage margin **0.08** (threshold **>= 0.03**); larger on `shape` than pooled.
- ⚠ **Registered counter-hypothesis:** a random 1/K subset of 2048 coordinates is a random projection and may
  be *more* informative per well in absolute terms, which could invert the normalised ratio. **If clustered
  leakage is HIGHER than random beyond seed spread, that is a finding, not a bug**, and R0's x-coordinate is
  simply on the other side. I commit to reporting whichever way it lands.

**P4 — companions (descriptive; ⛔ none replaces leakage as the x-axis).**
- **Block-diagonality** clustered `>= 3x` random. ⚠ This is **near-tautological** (the clustering optimises a
  monotone function of it) and is pre-declared as a *sanity check that the clustering ran*, never as evidence.
- **MIG / DCI-D**: predicted clustered > random, margin small (**+0.01 to +0.05**); ⚠ plug-in MI over
  `M = 64` bins is upward-biased, identically for both arms (same `K`, `M`, `N`), so only the contrast is read.

**P5 — well-space C mapping (R2 addendum, diagnostic only, ⛔ never a query set).**
`intra_well_conflict_fraction` = fraction of the 72 C records whose `violated_factor`'s well-relevance mass has
top-1 well share `>= 0.5` (the structural conflict is localised inside ONE well).
- **Predicted:** clustered **0.6**, random **0.3**. Threshold for "grouping localises conflicts": clustered
  exceeds random beyond seed spread. No verdict rides on this.

## 3. What I will NOT do
⛔ No arm comparison on B, no certificate, no C-as-queries, no kill-rule verdict, no re-derivation of `K` after
seeing any result, no semantic well names anywhere. If a prediction fails I report the falsification.

---
## ADDENDUM D6 (appended 2026-09-10 09:5x, still before any harness existed or ran)
**Clustering algorithm declared a priori:** average-linkage agglomerative clustering (`scipy.cluster.hierarchy`)
on the condensed correlation distance `d_ij = 1 - |corr_ij|`, cut at `K` (`fcluster(criterion='maxclust')`).
Textbook correlation clustering; deterministic given the correlation matrix, so the *grouping* carries no seed of
its own (only the feature cache does). `ward` and `spectral` are implemented as config alternatives and are
**not run**. ⛔ If average linkage returns degenerate well sizes I report the sizes; I do not switch method to
get prettier ones. The random control matches the realised size histogram **exactly**, so a degenerate
clustering is compared against an equally degenerate random partition and the contrast stays fair.

---
## ⛔ AMENDMENT D6-b (2026-09-10, ~10:2x) — the pre-registered clustering method FAILED. Recorded, not hidden.

**Trigger.** Before any read metric, leakage number, deletion delta or store existed, the grouping step was run on
the frozen seed-0 cache. **Plain average linkage chains catastrophically on this correlation matrix.** Evidence,
all of it from TRAIN activations only:

| observation | value |
|---|---|
| `D` | 2048 |
| **DEAD coordinates** (constant post-ReLU, `std <= 1e-8`) | **65** |
| mean off-diagonal `|corr|` over live coordinates | 0.143 |
| `fcluster(average, maxclust)` on all 2048, `k in {4,6,8,12,16}` | **1 cluster every time** (top-of-dendrogram tie plateau at height exactly 1.0 — the dead coordinates sit at distance 1 from everything, so no cut exists) |
| `fcluster(complete, maxclust)` on all 2048 | **1 cluster every time**, same cause |
| average linkage on the **1983 live** coordinates, `k=7` | `[1972, 4, 3, 1, 1, 1, 1]` — chaining |
| ward on the live coordinates, `k=6 / k=8` | `[553,550,438,251,158,98]` / `[553,308,251,243,242,195,158,98]` |

**Two changes, both mechanical, both decided on train activations with no read/leakage/deletion number in
existence:**
1. **Dead coordinates are clustered around, not with.** They correlate with nothing, which is exactly what creates
   the tie plateau. They are placed afterwards, deterministically, into the currently-smallest well. A constant
   coordinate is reconstructed exactly by any mode, so where it lands carries no information.
2. **`grouping_method="auto"`:** walk `GROUPING_ORDER = ("average", "ward")` — the pre-registered default FIRST —
   and take the first method whose **smallest well holds >= 10% of an equal share**. The 10% floor is deliberately
   permissive: it rejects only near-empty wells.

⭐ **Why this is not method-shopping.** The acceptance test is a property of the *partition*, evaluated on train
activations, with **no held-out quantity, no read metric and no arm outcome** in it — and `ward` was already named
in D6 as an implemented alternative before any of this was measured. The method actually used is recorded in
`store_manifest.json.arms.*.grouping_info` and in `store_plan.json`. ⛔ The predicted margins in §2 are UNCHANGED.
⭐ **The chaining itself is a reportable finding: on this feature cache the textbook correlation-clustering default
does not produce a usable partition at all.**

---
## ADDENDUM P2-b (2026-09-10, ~10:3x — after the QUICK plumbing smoke, before any reported run)
The first `shuffle` deletion variant (permute a well's codebook **rows**) was measured at **Δ = 0.0000 exactly**
and is a **no-op by construction**: nearest-neighbour addressing re-finds the same vector. It is replaced by two
deletions, both pre-registered here:
- **`mean`** — every mode replaced by its well's mean mode. ⭐ The *fairest* deletion: the read output stays an
  on-manifold, correctly-scaled feature vector, so the decoder is not merely handed garbage, yet the store's
  `K·log2(M)` bits carry zero information. **Predicted `delta_aggregate` >= 0.15**; I expect it to be *smaller*
  than the `zero` delta because part of the `zero` delta is the decoder being fed an off-manifold input.
  ⛔ If `mean` gives a near-zero delta while `zero` does not, the honest reading is that the `zero` delta was
  mostly a decoder artefact — and I will report it that way.
- **`scramble`** — mode-matrix entries permuted (content destroyed, per-well marginal preserved). Predicted to
  behave like `zero`.
`zero` remains the headline and its thresholds in §2/P2 are unchanged.
