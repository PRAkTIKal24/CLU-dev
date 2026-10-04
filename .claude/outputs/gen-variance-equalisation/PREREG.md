# PREREG — `gen-variance-equalisation`
**Filed before any measurement of this spoke.** Agent: `experiment-engineer`.
Base `main @ 81ba81b`, branch `agent/experiment-engineer/gen-variance-equalisation`,
worktree `../CHLU-gen-varianceeq`, main venv.
⛔ **Never edited in place.** Any addendum is a NEW file (R34's seventh strike).

---

## §0 — DIAL DECLARATION (echoed verbatim from the task)
- **Dial:** *feature scaling before the store.* ⛔ The encoder, the decoder, the partition and `K` are all UNCHANGED.
- **Laundering control:** ⭐ **`knn_proto` must be BIT-IDENTICAL.** It never sees the transform, so if its number moves
  **you have changed something you did not intend** — stop and report.
- **Falsifiers, pre-registered:** ⭐ **(a)** the `scale` + `orientation` probe **on the store code** rises above the
  unwhitened baseline **beyond seed spread**; **(b)** the **distinct configuration count rises**. ⛔ If neither moves,
  the variance hypothesis is **refuted** and that is the result.

---

## §1 — The hypothesis under test, stated so it can die
k-means minimises squared error, so it spends codewords on **high-variance directions**. If `scale` and
`orientation` live in **low-variance directions** of the per-well features, they are quantised away — not because the
information is absent (`gen-loss-attribution`: the encoder holds them at 0.9352 / 0.9915) but because the objective
does not value them.

## §2 — Leg 1 statistics, defined before they are computed
Per seed, per well `k` (channels `w_k`, `d_k = |w_k|`), on the **store-training rows** (the rows the codebook is
actually fitted on), on the **unwhitened** cached features:

1. centre `Xk = X[:, w_k] − mean`; covariance `Ck`; eigendecomposition `Ck = V Λ Vᵀ`, `λ_1 ≥ … ≥ λ_{d_k}`;
   principal-axis scores `T = Xk V`.
2. for factor `f` realising values `v` with counts `n_v`, the **between-class variance of PC `j`**:
   `b_{j,f} = Σ_v (n_v/N) · (mean_v T_j − mean T_j)²`.
3. **`VE(k,f)` = variance-explained-per-factor** `= (Σ_j b_{j,f}) / (Σ_j λ_j)` — the share of the well's total
   variance that the factor's class means account for. (Basis-independent; the PCs are only how it is decomposed.)
4. **`rank_frac_median(k,f)`** — with `w_{j,f} = b_{j,f} / Σ_j b_{j,f}`, the smallest `r/d_k` at which
   `Σ_{j≤r} w_{j,f} ≥ 0.5`. ⭐ **This is the literal "which part of the variance spectrum carries it" number.**
5. **`frac_signal_in_top10pct`** `= Σ_{j ≤ ceil(0.1 d_k)} w_{j,f}`.
6. **`var_frac_at_median_rank`** `= (Σ_{j≤r} λ_j)/(Σ_j λ_j)` at the `r` of (4) — *how much of the well's variance you
   must keep to keep half of this factor's signal.*
7. **Headline well for a factor** `= argmax_k VE(k,f)`. Every row carries `n_values_realised` (R31).

## §3 — ⛔ THE PREMISE TEST, and the STOP rule, both mechanical
**The premise is the DIRECTION claim**, so `rank_frac_median` adjudicates it and `VE` is reported beside it.

> **PREMISE HOLDS** iff, on **≥ 2 of 3 seeds**, for **BOTH** `scale` and `orientation`, in that factor's own headline
> well: `rank_frac_median(f) > 2 × max( rank_frac_median(floor_hue), rank_frac_median(wall_hue) )`,
> each hue evaluated in **its own** headline well.
>
> **PREMISE DEAD** otherwise ⇒ ⛔ **STOP after Leg 1. Do not whiten.**
>
> **PARTIAL** is declared (and Leg 2 still runs, with the partiality leading the report) **only** in the one case
> where the direction test passes but the supporting share test fails, i.e. the premise holds but
> `VE(scale) ≥ 0.5·min(VE(floor_hue), VE(wall_hue))`. ⛔ A failure of the direction test is DEAD with no escape.

## §4 — Leg 1 predictions (3-seed mean, each factor in its own headline well)
| id | quantity | prediction | 80 % interval |
|---|---|---|---|
| **L1** | `VE(floor_hue)` | 0.28 | [0.10, 0.55] |
| **L2** | `VE(wall_hue)` | 0.26 | [0.10, 0.55] |
| **L3** | `VE(object_hue)` | 0.10 | [0.02, 0.30] |
| **L4** | `VE(shape)` | 0.05 | [0.01, 0.15] |
| **L5** | ⭐ `VE(scale)` | 0.02 | [0.003, 0.08] |
| **L6** | ⭐ `VE(orientation)` | 0.02 | [0.003, 0.08] |
| **L7** | `rank_frac_median(floor_hue)` | 0.006 | [0.001, 0.03] |
| **L8** | `rank_frac_median(wall_hue)` | 0.006 | [0.001, 0.03] |
| **L9** | ⭐ `rank_frac_median(scale)` | 0.05 | [0.01, 0.30] |
| **L10** | ⭐ `rank_frac_median(orientation)` | 0.08 | [0.01, 0.40] |
| **L11** | premise verdict by §3 | **HOLDS** | — |

*Derivation.* A reconstruction (`recon`) encoder is trained on pixel MSE. On Shapes3D a floor/wall hue change repaints
the majority of the image ⇒ enormous pixel energy ⇒ it must dominate the leading principal axes. `scale` moves a
small central object a few pixels, `orientation` rotates the camera slightly: small pixel energy, hence (under an MSE
pretext) small feature energy. ⛔ This is an argument from the *pretext loss*, not from the probe numbers, and it can
be wrong: a nonlinear encoder is free to allocate a high-variance direction to a low-pixel-energy factor.

## §5 — Leg 2 design, ruled and fixed before the run
- ⭐ **Grouping fitted on UNWHITENED features FIRST, then whiten.** The partition is `gen-commitment-collapse`'s
  cached `labels_vec` (unwhitened), asserted **channel-for-channel** identical to the shipped store's
  `well_channels`. ⛔ The transform is fitted **after** the partition exists and can never move it.
- **PRIMARY ARM `pca_whiten`:** full PCA whitening **within each well's own sub-space**,
  `z ↦ diag((λ+ε)^(-1/2)) Vᵀ (x − μ)`, per well. ⛔ Not across `D`.
- **SECONDARY ARM `diag_std`:** per-channel `(x − μ)/σ`.
- **CONTROL ARM `unwhitened`:** the identity transform, **re-fitted through the identical code path**, same rows,
  same key, same steps. ⛔ The baseline is re-measured here, never transcribed.
- **REFERENCE ROW `shipped`:** `gen-read-compose`'s frozen store, loaded and hash-asserted, so this spoke's ladder
  can be checked against `gen-loss-attribution`'s.
- Encoder, decoder, partition, `K`, `M`, `n_store_train`, `train_steps`, PRNG key: **identical across all arms.**
  The only difference is the metric k-means sees.

## §6 — Leg 2/3 predictions (SPREAD grid, 3-seed mean, `pca_whiten` vs `unwhitened` refit)
"Retention" = `(acc − chance)/(acc_encoder − chance)` with the **same** encoder point for every arm.
Baseline ladder to be **re-measured**, not transcribed (`gen-loss-attribution` reported
floor 0.997 · wall 0.995 · obj 0.665 · shape 0.555 · orient 0.243 · scale 0.225, TOTAL **3.680**).

| id | quantity | prediction | 80 % interval |
|---|---|---|---|
| **P1** | `unwhitened` refit TOTAL-RETAINED (6 factors) | 3.68 | [3.2, 4.1] |
| **P2** | ⭐ `pca_whiten` retention `scale` | 0.38 | [0.20, 0.62] |
| **P3** | ⭐ `pca_whiten` retention `orientation` | 0.40 | [0.20, 0.65] |
| **P4** | `pca_whiten` retention `floor_hue` | 0.90 | [0.60, 0.999] |
| **P5** | `pca_whiten` retention `wall_hue` | 0.90 | [0.60, 0.999] |
| **P6** | `pca_whiten` retention `object_hue` | 0.55 | [0.30, 0.80] |
| **P7** | `pca_whiten` retention `shape` | 0.50 | [0.25, 0.75] |
| **P8** | ⭐⭐ `pca_whiten` **TOTAL-RETAINED** | 3.70 | [3.1, 4.4] |
| **P9** | ⭐⭐ TOTAL-RETAINED **change** (whitened − unwhitened) | **+0.02 — i.e. FLAT** | [−0.5, +0.7] |
| **P10** | `diag_std` TOTAL-RETAINED change | +0.01 | [−0.3, +0.4] |
| **P11** | feature-space reconstruction MSE of the hard read, `pca_whiten` vs `unwhitened` | **WORSE (higher)** | ratio [1.0, 3.0] |
| **P12** | round-trip worst `|inverse(forward(x)) − x|`, `pca_whiten` | ≤ 1e-2 | [0, 1e-1] |

## §7 — ⭐ THE PRE-REGISTERED FALSIFIERS, in the words they will be answered in
- **Falsifier (a) — "does the probe rise?"** Answered **YES** iff, on the SPREAD grid,
  `retention_code(pca_whiten) − retention_code(unwhitened)` exceeds **the larger of the two arms' 3-seed population
  sds** for **BOTH** `scale` and `orientation`. Anything else is **NO**.
  ⭐ **My prediction: YES** (P2/P3 vs a re-measured ~0.23).
- **Falsifier (b) — "does the distinct configuration count rise?"** ⭐ **PRIMARY: the request-side count on the 72
  held-out `B` requests** (`compose_codes(request_posterior(book, req_B))`, the programme's "9 answers to 72
  questions"). Answered **YES** iff the 3-seed mean rises beyond the larger sd.
  **SECONDARY (reported, not adjudicating): the feature-side distinct `(N,K)` code count on the 4,096 SPREAD rows**
  (shipped store: 3,598 / 3,594 / 3,723 — already near-injective, so little headroom exists there by construction).
  | id | quantity | prediction | 80 % interval |
  |---|---|---|---|
  | **P13** | `unwhitened` refit distinct-of-72 on B | 9 | [4, 20] |
  | **P14** | ⭐ `pca_whiten` distinct-of-72 on B | 16 | [6, 40] |
  | **P15** | `pca_whiten` distinct codes on 4,096 SPREAD rows | 3,600 | [3,000, 4,000] |
- ⭐ **The pass the task demands is BOTH.** The probe rising without the distinct count (or vice versa) is
  ⛔ **not** the pre-registered pass and will be reported as the split it is.

## §8 — ⭐⭐ Leg 4 — THE PER-WELL CAPACITY BOUND, PRE-REGISTERED BEFORE MEASURING
`M = 64 = 2^6` modes per well; `K = 5` wells; `K·log₂M = 30 bits` aggregate vs `log₂(480,000) = 18.87 bits`
⇒ ⭐ **the code is SUFFICIENT in aggregate** and ⛔ "30 bits" may **not** be written as the cause.
**Per well it is not.** A well that must resolve a set `F` of factors needs `Π_{f∈F} C_f ≤ 64`:

| what a single well must resolve | combinations | vs `M = 64` |
|---|---|---|
| one hue alone (`C=10`) | 10 | fits (6.4× spare) |
| `shape` × `scale` (4×8) | 32 | fits (2× spare) |
| ⛔ **two hues (10×10)** | **100** | **1.56× over** |
| ⛔ **one hue × `orientation` (10×15)** | **150** | **2.3× over** |
| ⛔ `orientation` × `scale` (15×8) | 120 | 1.9× over |
| ⛔ **all six** | **480,000** | **7,500× over** |

**Operational per-well bound, measured the same way in every arm:** a well `k` *carries* factor `f` iff
`MI(code_k; f)/H(f) ≥ 0.25`; the well's **load** is `Π_{f carried} C_f`, and its **surplus** is `load / 64`.
| id | quantity | prediction | 80 % interval |
|---|---|---|---|
| **P16** | number of wells with `load > 64` (`unwhitened`, 3-seed mean of 5) | 2.0 | [0, 4] |
| **P17** | ⭐ same for `pca_whiten` | 3.0 | [1, 5] |

> ⭐⭐ **PRE-REGISTERED READING (verbatim from the task).** *"If the TOTAL-RETAINED line is flat and only the
> DISTRIBUTION moves, CAPACITY is the binding constraint, not the metric."* ⛔ That is a first-class result, not a
> null — and it sends the repair to `M` (or to a different representation), **never to the metric.**
> ⭐ **P9 predicts exactly this outcome**: `scale`/`orientation` up, hues down, total flat.
> ⛔ Whitening **cannot** beat the bound; it can only choose which factors spend the modes.

## §9 — ⭐ THE BYTE LEDGER, and the fit-time/deployed declaration, decided in advance
Well sizes are **per seed** (s0 `[303,305,573,501,366]`; s1 `[397,300,307,300,744]`; s2 `[283,283,390,616,476]`),
so `Σ d_k²` is measured per seed and the table below is the Hub's s0 arithmetic, to be **re-derived in code**.

| transform | formula | s0 bytes | vs the 524,288 B store budget |
|---|---|---|---|
| `diag_std` (secondary) | `2·D·4` (μ, σ) | **16,384** | 3.1 % |
| ⭐ `pca_whiten` (primary) | `Σ_k d_k²·4 + D·4` (V with `λ^{-1/2}` folded in, + μ) | **3,600,672** | ⛔ **6.87×** |
| `pca_whiten` + explicit inverse | `+ D·4` (keep `λ`) | 3,608,864 | 6.88× |
| *(full-`D` whitening — ⛔ NOT the ruled arm)* | `D²·4` | *16,777,216* | *32×* |

⭐ **Declared IN ADVANCE, not after seeing the result** — the arm is priced **two ways and both are reported**:
1. ⭐ **FIT-TIME-ONLY (0 deployed bytes).** The codebook is *fitted* in the whitened metric and the `M` centres per
   well are then mapped **back** through the inverse and **stored un-whitened**. The deployed read is
   `request → address book → code → read_from_codes`, a plain gather of un-whitened centres — ⛔ **the transform is
   never evaluated at read time and costs 0 deployed bytes.** This is the arm's honest deployment price *on the
   request-side (cell) path*, and it will be **proved** by asserting
   `inverse(read_from_codes(store_whitened, c)) == read_from_codes(store_unwhitened_equivalent, c)`.
2. ⛔ **DEPLOYED (the table above).** Any path that must map a **feature vector** to a code at read time
   (`address_codes`) must evaluate the forward transform, and pays **3,600,672 B = 6.87× the whole store budget.**
⛔ A deployed 6.87× transform reported as though it were free is the easiest way for this arm to look like a win when
it is a much larger model. Both prices ship, in bytes, in the report.

## §10 — ⭐ The laundering control, and what would prove this spoke wrong
- **`knn_proto` BIT-IDENTICAL.** Two independent checks:
  (i) **source** — `git diff main..HEAD` touches **0 lines** of any file on the `knn_proto` path
  (`chlu/core/gen_request.py`, `chlu/core/gen_compose.py`, `chlu/experiments/exp_gen_cell.py`,
  `chlu/experiments/exp_gen_baselines.py`, `chlu/eval/**`);
  (ii) ⭐ **numerical** — the `PrototypeTable` embedding of the eval rows and a sha256 of the feature arrays are taken
  **before** the whitening pipeline runs and **again after**, and `max|Δ|` must be **exactly 0.0**. ⭐ This is the
  check that actually bites: it catches an in-place mutation of the shared feature array, which is the realistic way
  a transform escapes its scope.
  ⚠ **What it does NOT cover, declared:** the *shipped* `knn_proto` balanced-accuracy number is a decoder+classifier
  cell result that this spoke does not re-run; the claim proved here is that the feature inputs and the prototype
  embedding it is computed from are byte-identical, and that no file on its path changed.
- **Would prove this spoke wrong:** `knn_proto` moved · the probe rises but the distinct count does not (or vice
  versa) — both reported, neither alone is the pass · whitening done without the Leg-1 premise test.

## §11 — Provenance intentions
3 seeds `{0,1,2}`; `±` is **population sd, ddof = 0**; every per-factor row carries `n_values_realised`; the
headline grid is **SPREAD**, the corner is reported **separately**; the decoder, if used, is
`gen-decoder-lift`'s **`v1_capacity_schedule`** winner with its `param_hash` asserted — ⛔ never `dec_recipe="baseline"`.
