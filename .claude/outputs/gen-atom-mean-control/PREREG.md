# PREREG — atom-mean lookup, the like-for-like control (`gen-atom-mean-control`)
**experiment-engineer, 2026-09-14. Filed BEFORE any implementation and BEFORE the harness produced a number.**
⛔ Not edited after the run. Amendments are appended as a dated `PREREG-addendum-N.md`.

## 0. Why this file exists at all
My acceptance criterion is *"build a control"*, not a law — but acceptance item 2 makes me measure **read error at
two budgets**, which is a **two-point slope**, and `gen-dynamical-read/PREREG.md` §2 states the competing law. So the
slope is pre-registered here. ⛔ Committing to a number first is the whole point.

## 1. What the arm is (declared before it is coded)
`atom_mean` = **`knn_proto`'s query, restricted to the stored atoms.**
- **Query:** the R8 composed per-factor prototype embedding, `FactorPrototypes.embed_many(request)` — *bit-identical
  to `knn_proto`'s*. ⛔ No feature tensor on the deployed path.
- **Store:** **atoms = actual TRAIN DATA POINTS** from the store-train pool (32,768 rows), selected **per well** on
  that well's own channels. ⛔ Never k-means centres — that is the shipped codebook, a different arm.
- **Read:** per well, retrieve the `k` nearest atoms **in that well's channels** and **average them**; concatenate the
  K well slices into the D-vector; decode through THE one frozen `v1_capacity_schedule` decoder.
  ⇒ piecewise-constant over atom index-sets. **It does not settle.** That is the entire content of the control.
- **Channel:** `K · k · log2(N)` bits (budget A, k=1: 30.0 bits — ⭐ *exactly the store's 30*).

**Selection method (⭐ chosen to be the STRONGEST honest version, per the never-weaken rule):** per well, Lloyd
k-means with `N` clusters on that well's channels, then **snap each centroid to its nearest actual train row**
(k-medoids), de-duplicating so exactly `N` **distinct** atoms are kept. ⛔ A random subsample is the weak member and
is run only as a declared companion.

**Rows are inherited, never re-chosen:** `n_fit = 32,256`, `n_val = 512` (`GenBaselinesConfig`'s), pool =
`select_store_rows(...)["store_train"]`, atoms drawn from that pool — **the same stored set the landscape will use**,
persisted to `atoms_s{seed}_{A,B}.npz` with a sha256 so `gen-dynamical-read` loads *these* atoms.

## 2. ⛔ Byte ledger, declared before the run
`D = 2048`, `K = 5` wells, `Σ_k d_k = D`. An atom costs `Σ_k d_k · 4 = D · 4 = 8,192 B`.

| budget | atoms/well `N` | bytes | equals |
|---|---|---|---|
| **A** | **64** | `64 · 2048 · 4` = **524,288 B** | ⭐ the deployed CLU store **exactly** (which is 64 codewords/well) |
| **B** | **512** | `512 · 2048 · 4` = **4,194,304 B** | 8× A, `gen-dynamical-read` §3's declared larger budget |

⛔ Asserted in-run; the run aborts on any other realised value.

## 3. ⭐ PRE-REGISTERED POINT PREDICTIONS
Derivation, stated so it can be wrong:
1. `gen-dynamical-read/PREREG.md` §2 anchors the shipped **centroid** quantiser at relative read error (squared-error
   ratio) **0.2783** at `M = 64`, implying aggregate `PR = 6.503`; the law is `ε(N) = 0.2783 · (N/64)^(−2/PR)`.
2. **Medoid penalty.** Replacing a cell's centroid by the nearest *data point* adds `E‖m − c‖²` to that cell's
   distortion. With `n_c = 32,768/N` points per cell in effective dimension `PR`, the nearest-point distance scales as
   `r_typ · n_c^(−1/PR)`, so `distortion_medoid / distortion_centroid ≈ 1 + n_c^(−2/PR)`.
   ⇒ **×1.147** at `N = 64` (512 pts/cell), **×1.278** at `N = 512` (64 pts/cell).

| # | quantity (feature-addressed "oracle" query, `k = 1`) | **predicted** | band |
|---|---|---|---|
| **A1** | budget A relative read error | **0.319** | [0.25, 0.42] |
| **A2** | budget B relative read error | **0.188** | [0.14, 0.26] |
| **A3** | ⭐ **fitted exponent** `d ln ε / d ln N` over the two budgets | **−0.256** (effective `PR` = **7.82**) | [−0.35, −0.18] |

⭐ **A3 is the number that matters to the Hub: the atom lookup's exponent is predicted to be SHALLOWER than the
centroid quantiser's (−0.308, `PR` 6.50), because the medoid penalty grows as `N` grows.** If the dynamical read
cannot beat **−0.256**, it has not beaten a plain lookup over its own atoms.

| # | quantity | **predicted** |
|---|---|---|
| **A4** | validation-selected `k` (nearest-atom count) | **`k ≥ 3` at budget B**; averaging `k` atoms is already a zeroth-order smoother and should recover most of the medoid penalty ⇒ the selected-`k` oracle error should land **at or below** the centroid law (0.2783 / 0.1467) |
| **A5** | ⭐ **request-driven (deployed) read error is budget-INSENSITIVE** — the R8 embedding's own error dominates the quantisation term, so A-vs-B differ by **< 20 % relative**, far less than the 1.70× the oracle path predicts | |
| **A6** | six-factor retention ladder on B, aggregate (probe), budget A | **0.72** [0.60, 0.83], i.e. **below** `knn_proto`'s 0.796 (a bounded channel cannot beat the unbounded one it is a restriction of) |
| **A7** | same, budget B | **0.77** [0.65, 0.85], strictly between A and `knn_proto` |
| **A8** | ⭐ **distinct configurations on the 72 held-out requests**, budget A | **34** [15, 65] — ⛔ far above the CLU read's **9**, because 64 atoms/well × 5 wells is a much finer partition than a 64-word code the address book collapses onto 9 points |
| **A9** | same, budget B | **≥ A8**, predicted **55** [30, 72] |
| **A10** | ⛔ `knn_proto` (full, unrestricted) probe in-corner balacc reproduced **Δ = 0.0 EXACTLY**, 3/3 seeds, against `0.8762626262626263 / 0.8919597989949748 / 0.8969849246231156` | **bit-identical** |

## 4. ⛔ What would prove this spoke WRONG (the falsifier is mine, not the read's)
- ⭐ **A control that is weaker than it should be.** Concretely: if `atom_mean` at budget B does **not** beat
  `atom_mean` at budget A on the retention ladder, or if its oracle read error is **worse than 1.5×** the centroid
  law, the atom selection is broken and ⛔ **no number here may be used as a rival.**
- ⛔ **`knn_proto` moved** (A10 fails) ⇒ I changed something out of scope; the comparison is void.
- A per-factor row without `n_values_realised` ⇒ the row is meaningless and must be struck.

## 5. ⛔ Not in scope, declared NOT RUN in advance
No landscape, no settling, no CLU number, no dial verdict. The `clu` / `store_removed` / `lace` / `ebm_compose` arms
of `gen-cell-harness` are **not re-run** here; only `knn_proto` is, as the regression check.
