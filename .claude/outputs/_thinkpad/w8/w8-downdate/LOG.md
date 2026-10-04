# w8 spoke/downdate-build — LOG

Task: `/c/chlu/tasks-w8/w8-spoke-downdate-build.md` · Worktree: `/c/chlu/worktrees/CHLU-w8-downdate`
Base: `base/iter7 = ec2ef7c` · Output: `/c/chlu/outputs/w8-downdate/`
⛔ Build only. No scored run. `hub/single-design` scores this object.

## 2026-09-20

- **16:36Z** START_UTC.txt written (no live one existed). `chlu.__file__` resolves to
  `C:\chlu\worktrees\CHLU-w8-downdate\chlu\__init__.py` — inside my worktree. `git log -1` = `ec2ef7c`,
  the declared base. Worktree clean.
- **16:37Z** Read the task file. The deliverable is three assertions on a four-part store:
  (1) downdate == without-rows refit to float precision, max abs dev printed per well;
  (2) no pass over the data, asserted mechanically; (3) items never kept.
- **16:38Z** Located the incumbent this object must reproduce: `exp_w5_compose_field.PCASpanStore`
  (`chlu/experiments/exp_w5_compose_field.py:218`) built by `build_pca_content` (:468).
  Per well `k`: take pool features on well `k`'s channels, mean `mu`, center, `np.linalg.svd`,
  keep top `A-1` right singular vectors, atom block = `[mu; Vt[:A-1]]`, zero-padded to `A` rows.
  `A = 524288 // (4*D)`. This is the object whose bytes are pinned at 524,288 B.
  ⭐ Consequence for the design: the SVD route holds no sufficient statistic, so it cannot be
  downdated. The downdatable route is the **covariance** route — `S_k = sum x x^T`, `t_k = sum x`,
  `n` — eigendecomposed. That is why the ledger must carry `d_k^2` per well BESIDE the store.
- **16:45Z** Environment probe. `CHLU_GEN_FEATURES_ROOT` is NOT in `env_w8.sh`, so the default
  resolves to the GENERATION-1 bridge. ⛔ Set explicitly to
  `C:/chlu/outputs/spoke-a-rivals/cache_generation2` (from `RIVALS-READY.json.cache_gen2_root`'s
  parent); manifest confirms `generation: 2`, `features_sha256 ef6359a4…`, 480,000 rows.
  Reused w6's `dev/val_seed0` caches (`partitions.npz`, `pca_content.pkl`, the incumbent address
  book) by copying them into `build/val_seed0` — the partition is not a dial here.
- **16:46Z** ⭐ MEASURED, and it corrects an assumption in flight: the incumbent's partition is
  `partitions.npz['factor']`, well sizes **[771, 940, 217, 60, 60]** (confirmed twice: recomputed
  from the label array, and `partitions.json.well_sizes.factor`). `sum_k d_k^2 = 1,532,330`.
  So the honest covariance cost at the pinned partition is **12,258,640 B float64 full square**
  (6,137,512 B if only the symmetric triangle is held) against a 524,288 B store — **23.4x**
  (11.7x triangle). Printed, never netted, per task §3.
- **16:50Z** ⛔⭐ THE FIRST RUN OF THE HEADLINE ASSERTION, and it FAILED on two wells — then the
  cause turned out to be a defect in the store's *definition*, not in the downdate. Raw result,
  deleting the 11,024 pool rows carrying `object_hue = 0.0` (the programme's own pinned row-7
  value, `exp_gen_factor_wells.DELETED_FACTOR, DELETED_VALUE_INDEX = 2, 0`):

  | well | d_k | max abs dev, downdate vs without-rows refit |
  |---|---|---|
  | 0 | 771 | 1.164e-10 |
  | 1 | 940 | **0** (bit-identical) |
  | 2 | 217 | **0** (bit-identical) |
  | 3 | 60 | **1.000e+00** |
  | 4 | 60 | **1.000e+00** |

  ⭐ Diagnosis, measured not guessed: wells 3 and 4 hold **48 and 19 dead channels** out of 60
  (`partitions.npz['dead']`), so their covariance is **rank-deficient** — effective rank 16 and 41.
  `r = min(A-1, d_k) = 60` asks for 60 directions, so the store writes 44 and 19 eigenvectors of a
  **numerically null eigenspace** (`lambda ~ -2e-11`, `min_gap = 0`). In a null eigenspace the
  eigenvector is *not a function of the data*: LAPACK returns an arbitrary rotation. Both the
  downdate and the refit are "right"; they simply disagree, and so would two refits of the same
  rows on two builds of LAPACK. ⛔ **The incumbent `PCASpanStore` has the same non-uniqueness** —
  it takes `Vt[:r]` from an SVD without a rank cut — it just never had to be a function of its
  fit set twice.
- **16:52Z** ⭐ FIX, and it is a well-posedness fix rather than a tolerance that buys a pass: a
  direction whose eigenvalue is numerically zero carries no variance and is **not a principal
  direction**, so the store writes it as a **zero row** — which is exactly the convention
  `PCASpanStore` already states for a well narrower than `A` ("deploy them anyway as zero rows:
  they cost the same bytes and buy no span"). Declared pin `eig_rel_tol = 1e-12`:
  `keep = min(A-1, d_k, #{j : lambda_j > 1e-12 * lambda_max})`. With it:

  | well | d_k | rank kept | max abs dev | cut margin below / above |
  |---|---|---|---|---|
  | 0 | 771 | 63 | 1.164e-10 | 1.24e+09 / — (rank cut binds, not the tolerance) |
  | 1 | 940 | 63 | **0** | 2.84e+09 / — |
  | 2 | 217 | 63 | **0** | 2.99e+09 / — |
  | 3 | 60 | 16 | 5.057e-12 | 2.31e+03 / 2.78e+03 |
  | 4 | 60 | 41 | 1.141e-13 | 4.58e+08 / 5.36e+04 |

  ⭐ The cut is **not marginal**: the smallest kept eigenvalue is >= 2.3e3 x the threshold and the
  largest dropped one <= 3.6e-4 x it, in every well. The ranks agree between downdate and refit in
  every well — asserted, not observed. ⭐⭐ **The headline assertion holds**: max abs dev
  **5.06e-12** over all five wells, two wells bit-identical, against float32 atoms whose ulp at the
  atom scale is ~3.7e-9. Detail follows once it is in the instrument rather than a probe.
- **16:55Z** Also adopting a sign convention as part of the store's definition, for the same
  well-posedness reason: an eigenvector's sign is free, so each direction is signed to make its
  largest-magnitude component positive (first index breaks ties).
- **16:58Z** Message from `hub/single-design` (it scores this object): it had costed the covariance
  from a **test fixture** (`tests/test_gen_dynamical_read.py:152`), not the deployed partition, and
  proposed re-pinning to `K=22` so the statistic fits inside 524,288 B. Replied with the measured
  partition and the corrected arithmetic, and declined the re-pin: task section 2 pins 64 directions
  and 524,288 B at **tolerance 0** and section 3 forbids netting the covariance into the store's
  budget, so re-pinning `K` to bring the statistic under the pin **pays for it by deleting content**
  — which is netting by another route. The hub withdrew the request, confirmed the val/test split of
  the well sizes ([771,940,217,60,60] on val, [777,939,216,58,58] on the scored leg) and is building
  the matched-bytes variant as its own arm. Also contributed the framing I have adopted: the
  non-uniqueness the rank cut fixes is **the incumbent's property**, not something the covariance
  route introduces — the deletion property is simply what finally asked the question.
- **17:05Z** Instrument written: `chlu/core/downdate_store.py` (the object) and
  `chlu/experiments/exp_w8_downdate.py` (fit / assertions / smoke / sweep). The store presents the
  programme's existing span interface (`modes`, `well_channels`, `dim`, `K`, `byte_ledger`) so
  `gen_read.span_basis` / `render_span` / `TensorAddress` read it unchanged and no other arm's code
  path moves. `assert_layout_matches_field_store` asserts the hole's index space **is**
  `gen_field`'s — 1,322 pairs, 60 indices cross-checked — so a hole punched here is the hole w5
  measured, not a lookalike.
- **17:08Z** STAGE `fit`. The store fits at the pin, `DOWNDATE-READY.json` published **PARTIAL**
  (hub unblocked). THE LEDGER, four parts, printed separately and never netted:

  | part | bytes | note |
  |---|---:|---|
  | 1. principal subspaces (the store) | **524,288** | pin 524,288, matches, tolerance 0 |
  | 2. running covariance, float64 `d_k^2` | **12,258,640** | per well [594441, 883600, 47089, 3600, 3600] |
  | 3. request count table | 11,488 | `term_n` + `request_n` + `pair_n` |
  | 4. coupling + holes | 5,288 + 1,322 | 1 byte per pair, value in a different array |
  | **honest total** | **12,817,450** | **24.45x the pin** |

  **THE HONEST LEDGER IS 24.45x OVER THE PIN** (11.7x if the statistic is held as a symmetric
  triangle, 6.4x as a float32 triangle — but float64 is not optional, see 17:12Z). Task section 6
  names this as an expected outcome to print, and it is printed, not netted against the store and
  not netted against the address book. The *content* is at 524,288 B exactly; what is over budget
  is the price of exact deletion.
  Ranks kept per well **[63, 63, 63, 16, 41]**, `rank/d` **[0.082, 0.067, 0.290, 0.267, 0.683]**.
  Wells 3 and 4 have `d_k = 60 < A = 64`: **saturated**, carried on the artifact. That is the
  incumbent's property too, not this object's.
- **17:12Z** Why the statistic is float64 and that is not a preference: MEASURED, the downdate
  residual `|(S_all - S_del) - S_keep|` runs to **4.95e-10** absolute on entries of order 1e+9.
  A float32 accumulator (eps ~1e-7 relative) buries the exactness this object exists to have.
  Note the *products* are exact either way — float32 x float32 has at most 48 significand bits and
  lands in a float64 unrounded — so the only error in the whole downdate is summation order.
- **17:18Z** STAGE `assertions`. **ALL THREE PASS.**

  **ASSERTION 1 — the downdated store EQUALS the store fitted WITHOUT those rows.**
  Two deletions, because exactness must not be a property of the deletion's size: the pinned
  `object_hue = 0.0` (**11,024 rows**, the programme's own row-7 value) and a random **64 rows**.

  | well | d_k | max abs dev (11,024 rows) | in float32 ulp @1.0 | subspace proj dev | bit-identical |
  |---|---|---|---|---|---|
  | 0 | 771 | 1.164e-10 | 9.77e-04 | 1.15e-10 | no |
  | 1 | 940 | **0** | 0 | 0 | **yes** |
  | 2 | 217 | **0** | 0 | 0 | **yes** |
  | 3 | 60 | 5.057e-12 | 4.2e-05 | 1.11e-11 | no |
  | 4 | 60 | 1.141e-13 | 1.0e-06 | 1.65e-13 | no |

  Random-64 deletion: [1.137e-13, 2.842e-14, **0**, 1.798e-11, 6.284e-14].
  Worst deviation over both deletions and all five wells: **1.164e-10 = 9.77e-4 of ONE float32 ulp
  at unit scale** — a thousand times finer than a float32 store can represent a direction.
  Ranks agree in every well; row counts, `term_n` and `pair_n` all exactly equal; the coupling
  table equal to float32.
  On the unit: I measured this three ways and report all three, because two of them are wrong.
  At the *block's* maximum the test is nearly free (the mean atom has norm ~18). *Entrywise* at
  each entry's own magnitude it is meaningless in the other direction — a near-zero atom component
  has a denormal ulp of 1e-45, so a 1e-10 difference there scores 3e7 ulp and the statistic is
  decided entirely by components carrying no direction. The unit that fits the object is absolute
  precision **at unit scale**, because every direction row is a unit vector.
  **NEGATIVE CONTROL, and it is why the passes mean anything**: one atom of the without-rows refit
  nudged by 1e-3 gives max abs dev 1.000e-03, **the check FIRED**. A check that has never failed is
  not evidence. (Suggested by `hub/single-design`; adopted.)

  **ASSERTION 2 — no pass over the data, mechanically.** Two independent proofs, because each
  alone has a hole. (i) **Tripwire**: every named route to the corpus — `fw.features_mmap`,
  `fw.gather`, `np.load`, `np.memmap`, `np.fromfile` — is replaced for the duration of the call by
  an object that *raises on any use*. Not a mock returning zeros: a mock lets a delete that
  secretly reads the corpus finish and merely get the wrong answer. (ii) **Row counter**: the one
  function in the module that touches a feature row increments `ROWS_CONSUMED`, read before and
  after. Result, at two corpus sizes:

  | corpus | rows | delete(64) consumed | per well |
  |---|---:|---:|---:|
  | full pool | 131,072 | 320 row-reads | 64 |
  | half pool | 65,536 | **320 row-reads** | 64 |

  The number is *the same at both corpus sizes* and equals exactly the rows handed in, once per
  well. That is what turns "it read few rows" into "it read only the rows it was asked to forget".
  DECLARED, not hidden: `delete()` is **handed** the rows; it does not look them up. The store
  never held them and does not after. What is asserted is that the **retained** corpus is never
  opened.

  **ASSERTION 3 — items are never kept, mechanically.** (i) The inventory is **complete**: every
  ndarray reachable from `vars(store)` is cross-checked against the declared list, so an array
  added later and forgotten fails the audit instead of slipping past it — **26 arrays**. (ii) Every
  shape is explained by `(A, d_k, K, T, NC)`; not one axis is the corpus size. (iii) The decisive
  one — **the object's byte count is IDENTICAL at two corpus sizes**: 12,833,834 B at 131,072 rows
  and 12,833,834 B at 65,536 rows, **grew by 0 B**. An object that keeps items grows with the
  corpus. The audit also asserts `n > d_k` per well and reports the margin (min 65,476), because
  the one way the argument could fail is a well with fewer rows than channels, whose covariance is
  rank-deficient in a way that can expose the rows themselves.
- **17:22Z** STAGE `sweep` — arithmetic only, no build, handed to `hub/single-design` because it
  asked. On the true partition (1,981 live channels of 2,048), wells as equal as the grid allows,
  float64 symmetric triangle: the covariance falls quadratically in `K` while the atom block falls
  only linearly, so the honest total crosses the pin at `K = 32`. **K=37, A=9 gives 522,088 B =
  0.996x, span 333, unsaturated** — independently reproducing the hub's own corrected point.
  This spoke does not take that trade and did not build it; sections 2 and 3 forbid it. Recorded as
  `k_sweep.json` so the hub has the numbers without rebuilding my instrument.
- **17:32Z** Unit tests written, `tests/test_downdate_store.py` — **17 tests, 4.0 s, no feature
  cache needed**, so the exactness claim can be re-checked by anyone without rebuilding the
  instrument. Every negative test asserts its check *fires*: the equality check on a nudged atom,
  the tripwire on a `np.load`, the retention audit on an array smuggled onto the object, and the
  audit on a well with fewer rows than channels.
  ⚠ Two of them failed first and the fixes are findings, not bookkeeping.
  (a) `test_rank_cut_is_what_makes_the_store_well_posed` initially showed `loose_dev = 0.0` — the
  fixture made its dead channels **exactly zero**, which puts an exactly-zero row and column in the
  covariance, and `eigh` resolves that block to the canonical axes *deterministically on both
  paths*. The two stores then agree even with no rank cut and the test proves nothing. Real rank
  deficiency is **collinearity**, whose null eigenspace is rotated and genuinely arbitrary — the
  fixture now makes two of well 2's channels linear combinations of a third, giving a
  2-dimensional null eigenspace (a 1-dimensional one is pinned by the sign convention anyway).
  (b) The `n <= d_k` test used `n = 4`, which collides with a declared axis length and tripped the
  n_rows-axis check first — it would have passed for the wrong reason. Now `n = 2`, with an assert
  that the fixture really has wells wider than the corpus.
- **17:40Z** STAGE `smoke`, SVD basis. ⛔ `val` only, no claim.
  First attempt died **after** the 5.5-minute address-book fit with `KeyError: 'spec'`: I called
  `fw.arm_features`, which is wave 3's own dispatcher and knows nothing of the `tensor` verb, so it
  fell through to the ridge head. The span read lives in `exp_gen_read.arm_features`. Noted in
  code at the call site.

  | arm | valid mean | sd | min | max | corners |
  |---|---|---|---|---|---|
  | `downdate_store` | **0.8941** | 0.0903 | 0.7361 | 1.0000 | 24 |
  | `pca_tensor` | **0.8941** | 0.0903 | 0.7361 | 1.0000 | 24 |

  **Paired delta +0.0000 on every one of the 24 val corners.** The object fits, reads, renders
  through the shared frozen decoder `01ef91f75eea` and is read back by the shared judge
  `c8741ac417d8`, and it reproduces the incumbent's read at the pin. Charter B.3 discharged on the
  deployed object: the read's signature carries no feature argument and is bit-stable under a
  feature perturbation, 64 requests checked.
- **17:44Z** ⭐⭐ That delta of exactly zero looked too clean, so I checked whether the two arms are
  actually different objects. They are, and the result is the most informative thing measured
  today. Against `PCASpanStore` on the same pool, same partition, same `A`:

  | quantity | value |
  |---|---|
  | max abs difference in the **atom bytes**, per well | [5.85e-01, 5.59e-01, 6.14e-01, **2.00e+00**, 1.36e+00] |
  | max abs difference in the **rendered vector** | 8.05e-06 (signal scale 3.72) |
  | median row L2 difference | 4.96e-05 (median row norm 31.3 → **1.6e-6 relative**) |

  ⭐ **Two stores whose content differs by O(1) render the same vector to 1.6e-6**, because each
  was read through its own address book fitted against its own basis. The content of this object is
  **gauge-arbitrary on its own**; only the *(basis, book) pair* carries meaning. That is not a
  curiosity — it is the mechanism behind the next entry.
- **17:48Z** ⛔⭐ THE POST-DELETE READ COLLAPSED, and the cause is the gauge, not the deletion.
  Deleting the 11,024 pinned rows and reading back with the pre-delete address book:
  **valid mean 0.8941 → 0.0000** over all 24 corners. Total destruction.
  `hub/single-design` had warned of exactly this an hour earlier (measured 2.1–5.4 max render
  deviation at three matched-bytes partitions between stores whose atoms agree to 1e-15) and its
  own fix — orthogonal Procrustes on the atoms — had failed and got *worse* at one partition.
  ⭐ DIAGNOSIS, and it is a defect in the *read*, not in the store: `gen_read.span_basis`
  (`chlu/core/gen_read.py:91`) re-derives `U_k` by taking the **SVD of the atom block**. On this
  content the block is `[mu ; v_1 .. v_r]` with the `v_j` **orthonormal**, so its singular spectrum
  is `1` repeated `r` times plus one larger value from the mean — **maximally degenerate**. Inside
  a degenerate singular subspace the SVD's rotation is arbitrary, so a 1e-10 perturbation of the
  atoms can rotate `U_k` by O(1) and every coefficient the book emits is re-interpreted.
  ⚠ `pca_tensor` has the identical gauge freedom and never pays for it: its deletion verb
  (`exp_gen_read.delete_value_from_address`) is declared in its own docstring as "an operation on
  the INTERFACE, not on atoms" and never moves an atom. ⛔ **The incumbent evades this by not
  having the property.** An object whose deletion really is a charter B.4 operation on atoms cannot
  evade it.
- **17:52Z** ⭐ THE FIX, and it is not Procrustes: stop asking a degenerate SVD where the basis
  should point. `downdate_store.pinned_span_basis` uses the store's **own direction rows** as the
  basis — they are already orthonormal and already ordered by eigenvalue — and appends the mean's
  orthogonal component. Zero bytes, exactly as `span_basis` costs zero; no SVD; nothing degenerate;
  and it moves *continuously* with the atoms. STAGE `gauge`, handing IDENTICAL coefficients to a
  downdated store and a refit-without store that agree to 1.164e-10, so every difference is gauge:

  | basis | max abs render dev | relative to render scale | span ranks |
  |---|---|---|---|
  | `gen_read.span_basis` (SVD) | **2.994e-03** | 7.41e-04 | [64,64,64,16,41] |
  | `pinned_span_basis` | **3.725e-09** | 8.53e-10 | [64,64,64,16,41] |

  **800,000x**, with the ranks agreeing.
  ⚠ A trap inside the fix, found by measuring and not by reasoning: the first version admitted the
  mean's orthogonal component whenever its norm cleared 1e-8. In the rank-deficient wells 3 and 4
  that component is **2.3e-8 and 4.9e-8 relative** — the mean lies *inside* the directions' span
  and the leftover is numerical noise, so I was putting an arbitrary null direction back into the
  basis at the interface after carefully excluding it from the store. Margin was only 2.3x. Pinning
  the admission threshold at `MU_PERP_REL_TOL = 1e-4` — four orders clear on both sides, since
  wells 0–2 sit at 0.69–0.80 — moved the gauge deviation from 3.45e-07 to 3.725e-09. ⭐ The lesson
  is general: a **basis** decision needs a declared margin exactly as an eigenvalue cut does.
- **17:08Z** STAGE `smoke --basis pinned`. Pre-delete the pinned basis costs nothing:
  `downdate_store` 0.8941 / sd 0.0903, `pca_tensor` 0.8941, paired delta **+0.0000** on all 24
  corners, identical to the SVD-basis run. ⛔ But the post-delete read is **0.0584** (was 0.0000
  on the SVD basis) — better, and still a collapse. So the pinned basis fixes the gauge *between
  two equal stores* (3.7e-9) and does **not** rescue a read across a real deletion.
- **17:11Z** STAGE `postdelete`, three ways. `hub/single-design` had measured that a deletion makes
  nearly-degenerate directions **swap rank positions**, so I reproduced it on my leg and tested
  the fix. Reordering, deleting the 11,024 rows: **49 of 246 slots moved**, per well
  [9, 18, 20, 2, 0] of [63, 63, 63, 16, 41], max shift [2, 4, 2, 1, 0], **min matched cosine
  0.3597**. Re-indexing the basis into the pre-delete correspondence — zero bytes of the book
  changed — gave **0.0451**, i.e. *no recovery at all* and slightly worse.
  ⭐ So the collapse is neither the SVD gauge nor the slot swapping. The min matched cosine of
  0.36 says the directions genuinely moved.
- **17:26Z** ⭐⭐ THE CONTROL THAT SETTLED IT. Refit the address book **on the downdated store**
  and read again: **0.9005** (sd 0.0908), against 0.8941 before the deletion. ⭐ So the deletion
  did **not** damage the content — the post-deletion store is exactly as good as the pre-deletion
  one. The entire 0.8941 → 0.058 collapse was the **stale interface**, not lost composition.
  ⛔ But a refit is an ALS pass over the fit pool, so this leg buys the answer at the price of
  "no pass over the data". It is a diagnostic, not the deletion verb.
- **17:27Z** ⭐ `hub/single-design` then separated the two effects in feature space on my own
  partition (4,096 probe rows outside the fit pool, oracle span coordinates — a **diagnostic**,
  disqualified as an arm under charter B.3, and its numbers bound what the judge can see rather
  than predicting a valid fraction): of the naive read's error, **90.03% of the signal norm is
  coordinate correspondence and 5.76% is the deletion** — a ratio of **15.63x**. It also withdrew
  its own earlier advice to me: a permutation removes only **63.25%** of the error, because the
  residual is rotation *within* near-degenerate subspaces, which a slot swap cannot express. That
  matches my 0.0451 exactly. The full linear correspondence `M_k = U0_k^T U1_k` is what expresses
  it. ⭐ Credit where due: the mechanism below is the hub's, adopted after its measurement beat my
  permutation.
- **17:30Z** ⭐⭐⭐ STAGE `carry`. **THE OBJECT DOES WHAT IT WAS BUILT TO DO.**
  `W'_k = W_k M_k` with `M_k = U0_k^T U1_k`, formed from two bases already in hand at delete time.

  | leg | valid mean | sd | data rows touched | refit |
  |---|---|---|---|---|
  | before delete | 0.8941 | 0.0903 | — | — |
  | after delete, book **unchanged** | **0.0584** | 0.0701 | 0 | no |
  | after delete, book **carried** | **0.8964** | 0.0924 | **0** | **no** |
  | after delete, book **refit** (diagnostic) | 0.9005 | 0.0908 | 131,072 | yes |

  ⭐⭐ **Delete 11,024 rows — 8.4% of the pool, an entire factor value — touch ZERO data rows,
  refit NOTHING, and composition is intact: 0.8964 against 0.8941 before.** The carry recovers
  **100.3%** of the collapse and lands within 0.004 of the full ALS refit.
  `M`'s minimum singular value per well [0.672, 0.732, 0.796, 1.000, 1.000] and its maximum
  departure from the identity [1.96, 2.00, 1.96, 1.09, 2.00] — so the correspondence is a large,
  genuinely non-trivial map, not a near-identity tidy-up.
  ⛔ THE COST, declared and not netted: the carry **rewrites the book's `W` values** —
  **127,488 B of the address book's 158,664 B change value** (not count). No data row is read and
  no ALS runs, so charter B.4's "no retrain" holds; but this is not the coupling hole's *one byte*
  and it must not be quoted as if it were. ⭐ The pinned basis, by contrast, changes **zero** bytes
  — it just is not sufficient on its own.
- **17:33Z** So the honest shape of the result, and it is a different shape from what I expected
  at 16:50Z: the **store** forgets exactly (1.164e-10, no data pass, no items kept) and the
  **interface** must be brought along by a data-free linear correspondence that rewrites 127,488 B.
  The design's premise survives. ⛔ What it costs beyond the incumbent is now fully visible: a
  12,258,640 B statistic and a 127,488 B interface carry per deletion.
- **17:30Z** ⛔ INCIDENT — **my readiness file was overwritten, and the bug was mine.**
  `hub/single-design` ran my module with `--out` pointing into its own tree to fit a
  **test**-condition store. `OUT_ROOT` and `READY` are **module-level in
  `exp_w8_downdate.py` and ignore `--out` entirely**, so its publish landed in my gate and
  replaced the val build's descriptors (`condition`, `well_sizes`, `dead_per_well`, `cov_bytes`,
  `honest_total_bytes`, `times_over_budget`, `content_sha256`, `rank_*`) with the test build's.
  ⭐ My `bd9b1cf` merge fix held — every assertion payload survived byte-for-byte — but for ~90 s
  the gate read `condition: "test"` beside `all_three_assertions_passed: true`, for assertions
  only ever run on val. ⛔ That is precisely the failure `bd9b1cf` was written to prevent,
  arriving through a different door: I guarded against *my own* re-run blanking the file and not
  against *another build* describing it. The hub restored the descriptors from my `store.json`,
  kept the pre-restore file at `DOWNDATE-READY.json.clobbered-by-single-design.bak`, and recorded
  an `INCIDENT_2026-09-20T19:29Z` field. ⭐ It told me rather than quietly fixing it, which is why
  this is a paragraph and not a silent wrong number in someone's table.
- **17:32Z** FIXED properly, on my side, and both guards TESTED to fire:
  1. a publish whose `out_dir` is outside `OUT_ROOT` is **refused, loudly** — verified by running
     `--stage fit --out <scratch>`: the store fitted, the publish refused, and the gate's `utc`
     did not move;
  2. a publish whose `condition` differs from the one already recorded is **refused** —
     descriptors and assertions in one file must be about one object.
  Then re-ran `--stage fit` on val so my own code, not a reconstruction, wrote every descriptor:
  `condition val`, `content_sha256 43dc1147…` (unchanged from the original build), `utc 17:31:40Z`.
  ⚠ Verified rather than trusted, as the hub asked.

---

# FINAL REPORT — w8 spoke/downdate-build

**Agent:** experiment-engineer spoke. **Base:** `base/iter7 = ec2ef7c`. **Branch:** `w8/downdate`,
head `2fab429`, bundled to `/c/chlu/outputs/w8-downdate/w8-downdate.bundle`.
**Condition:** `val`, seed 0, feature cache **generation 2** (`ef6359a4cd54b082`).
**Start** 2026-09-20T16:36:33Z. ⛔ **Build only. No scored run. No claim. `hub/single-design`
scores this object.**

## 1. What was built

`chlu/core/downdate_store.py` :: `DowndatableStore` — the single-design store, four parts, one
object, presenting the programme's existing span interface so `gen_read.span_basis` /
`render_span` / `TensorAddress` read it unchanged and no other arm's code path moves.

`chlu/experiments/exp_w8_downdate.py` :: stages `fit`, `assertions`, `smoke`, `gauge`,
`postdelete`, `postdelete_refit`, `carry`, `sweep`.
`tests/test_downdate_store.py` :: **17 tests, ~4 s, no feature cache required.**

The mechanism in one line: a principal subspace is a function of a **second moment**, a second
moment is a **sum over rows**, and a sum can be un-summed. So the store keeps no row and still
forgets a row exactly.

## 2. ⭐⭐ THE THREE ASSERTIONS — THE DELIVERABLE. ALL THREE PASS.

**1. The downdated store EQUALS the store fitted WITHOUT those rows.**
Two deletions, so exactness is not a property of the deletion's size.

| well | d_k | max abs dev, deleting 11,024 rows | deleting 64 random rows | subspace proj dev | bit-identical |
|---|---|---|---|---|---|
| 0 | 771 | 1.164e-10 | 1.137e-13 | 1.15e-10 | no |
| 1 | 940 | **0** | 2.842e-14 | 0 | **yes** |
| 2 | 217 | **0** | **0** | 0 | **yes** |
| 3 | 60 | 5.057e-12 | 1.798e-11 | 1.11e-11 | no |
| 4 | 60 | 1.141e-13 | 6.284e-14 | 1.65e-13 | no |

Worst over both deletions and all five wells: **1.164e-10 = 9.77e-4 of ONE float32 ulp at unit
scale**. Ranks agree in every well; row counts, `term_n` and `pair_n` exactly equal; coupling
table equal to float32.
⭐ **Negative control FIRED**: one atom of the refit nudged by 1e-3 → dev 1.000e-03, the check
fails as it must. A check that has never failed is not evidence.

**2. No pass over the data — mechanically, two independent proofs.**
*Tripwire*: `features_mmap`, `gather`, `np.load`, `np.memmap`, `np.fromfile` all replaced during
the call by an object that **raises**. Not a zero-returning mock — a mock lets a delete that
secretly reads the corpus finish and merely get the wrong answer.
*Counter*: the one function in the module that touches a feature row is instrumented.

| corpus | rows | `delete(64)` consumed | per well |
|---|---:|---:|---:|
| full pool | 131,072 | 320 row-reads | 64 |
| half pool | 65,536 | **320 row-reads** | 64 |

Identical at both corpus sizes. That is what turns "it read few rows" into "it read only the rows
it was asked to forget".

**3. Items are never kept — mechanically.**
Inventory of **26 arrays** cross-checked against `vars(store)`, so an array added later and
forgotten fails the audit rather than slipping past; every shape explained by `(A, d_k, K, T, NC)`;
no axis is the corpus size; and the decisive form — **object bytes 12,833,834 at 131,072 rows and
12,833,834 at 65,536 rows, grew by 0 B.** An object that keeps items grows with the corpus.
Also asserts `n > d_k` per well (min margin 65,476), the one case where a covariance can expose
its own rows.

## 3. ⛔ THE LEDGER — four parts, printed separately, never netted

| part | bytes | vs the 524,288 B pin |
|---|---:|---|
| 1. principal subspaces (**the store**) | **524,288** | **matches exactly, tolerance 0** |
| 2. running covariance, float64 `d_k^2` | **12,258,640** | 23.4x |
| 3. request count table | 11,488 | |
| 4. coupling + holes | 5,288 + 1,322 | 1 byte per hole |
| **honest total** | **12,817,450** | **24.45x OVER** |

11.72x if the statistic is held as a float64 symmetric triangle; 6.36x as float32 — but float64 is
**not optional**: the downdate residual on `S` runs to 4.95e-10 absolute on entries of order 1e+9.
⛔ Task §6 names this as an expected outcome to print. The **content** is at 524,288 B exactly;
what is over budget is the price of exact deletion. It is not netted against the store and not
netted against the address book.

## 4. ⭐⭐ DOES IT COMPOSE *AND* FORGET? — yes, and the interface must be carried

Smoke, val, 24 corners, shared frozen decoder `01ef91f75eea` + judge `c8741ac417d8`:

| leg | valid mean | sd | data rows touched | refit |
|---|---|---|---|---|
| `pca_tensor` (the incumbent) | 0.8941 | 0.0903 | — | — |
| `downdate_store`, before delete | **0.8941** | 0.0903 | — | — |
| after delete, book **unchanged** | **0.0584** | 0.0701 | 0 | no |
| after delete, book **carried** | **0.8964** | 0.0924 | **0** | **no** |
| after delete, book **refit** (diagnostic) | 0.9005 | 0.0908 | 131,072 | yes |

⭐ Paired delta against `pca_tensor` **+0.0000 on every one of the 24 corners** — the covariance
route reproduces the incumbent's read at the pin.
⭐⭐ **Delete 11,024 rows — 8.4% of the pool, an entire factor value — touch ZERO data rows, refit
NOTHING, and composition is intact.** The carry recovers **100.3%** of the collapse and lands
within 0.004 of a full ALS refit.
⛔ **The carry's cost, declared:** it rewrites **127,488 B of the address book's 158,664 B**
(values, not count). No data row is read and no ALS runs, so charter B.4's "no retrain" holds —
but this is **not** the coupling hole's ONE byte and must not be quoted beside it as if it were.

## 5. Three findings that were not in the plan

**(a) The store's definition was not well-posed, and this is the INCUMBENT's defect.**
Wells 3 and 4 hold 48 and 19 **dead channels** of 60, so their covariance is rank-deficient
(effective rank 16 and 41) and `min(A-1, d_k) = 60` makes the store write eigenvectors of a
**numerically null** eigenspace, where the eigenvector is not a function of the data at all — two
refits of the *same* rows would disagree. Raw deviation in those wells was **1.0**.
Fix: a declared rank cut, `keep = min(A-1, d_k, #{j : lambda_j > 1e-12 lambda_max})`, zero rows
below it — which is `PCASpanStore`'s own stated convention for an atom a well cannot use. Margins
are not marginal: smallest kept ≥ **1.98e3 ×** the threshold, largest dropped ≤ **2.02e-4 ×** it,
and `cut_is_marginal` is scored only over the two wells where the tolerance actually binds.
⛔ `PCASpanStore` has the identical non-uniqueness and has never had to be a function of its fit
set twice. **The deletion property is what finally asked the question.**

**(b) An exactly-forgetting store can still read completely wrong.**
The post-delete read collapsed to **0.0000**, then 0.0584 with the gauge pinned, then 0.0451 with
the slots re-indexed. The content is **gauge-arbitrary on its own**: my atom bytes differ from
`PCASpanStore`'s by up to **2.00** per well, yet each read through its own book renders the same
vector to **1.6e-6 relative**. Only the *(basis, book) pair* carries meaning.
Two contributions, both now measured: `gen_read.span_basis` takes the SVD of an atom block whose
spectrum is maximally degenerate (r unit rows), so a 1e-10 atom change rotates it by O(1) —
`pinned_span_basis` removes that at zero bytes (2.994e-03 → **3.725e-09**, 800,000x). And the
deletion genuinely rotates the span, which only the full linear correspondence expresses.

**(c) A basis decision needs a declared margin exactly as an eigenvalue cut does.**
My first `pinned_span_basis` admitted the mean's orthogonal component whenever it cleared 1e-8. In
the rank-deficient wells that component is **2.3e-8 relative** — the mean lies *inside* the
directions' span and the leftover is noise — so I was putting an arbitrary null direction back
into the basis after carefully excluding it from the store. Margin was 2.3x. Pinning it at 1e-4
(four orders clear both sides) took the gauge deviation from 3.45e-07 to 3.725e-09.
`hub/single-design` reached the same generalisation from its own failed orthogonality audit.

## 6. What this spoke did NOT do — declared, never a null

- ⛔ **No scored run, no claim.** Everything is `val`, charter A.2's free selection condition.
- ⛔ **Seed 0 only, condition `val` only.** The §5 template shows `seeds:[0,1,2]`; I ran seed 0.
  My gate is `val` **by construction** — I was scoped to build, not to score — so nobody should
  wait on me for a test-condition ready. The instrument is condition-parameterised: the same
  stages run with `--condition test` against a test-leg `partitions.npz`, and the three assertions
  are condition- and seed-independent in form. `hub/single-design` has independently reproduced
  assertion 1 on the **test** leg (3.725e-09, ranks agree) and fitted a test-condition store
  (honest total 12,869,482 B = 24.55x).
- ⛔ **Row 3 is not measured.** On `val` the harness scores only the 24 `val_nway` corners; there
  is no row-3 tiling on that condition. "Rows 3–4 within reach of `pca_tensor`'s spread" is
  discharged as the row-4-shaped val corners, where the paired delta is +0.0000.
- ⛔ **The matched-bytes variant (K≈37, A=9) was NOT built.** §2 pins 64 directions and 524,288 B
  at tolerance 0 and §3 forbids netting the statistic into the store's budget; re-pinning K to fit
  the covariance under the pin pays for it by deleting content. The corrected arithmetic is handed
  over as `k_sweep.json` (crossing point K=32; K=37/A=9 → 522,088 B = 0.996x, span 333,
  unsaturated — independently matching the hub's own figure). `hub/single-design` is building it.
- ⚠ **`delete()` is handed the rows; it does not look them up.** The store never held them and
  does not after. What is asserted is that the **retained** corpus is never opened.
- ⚠ **Wells 3 and 4 are SATURATED** (`d_k = 60 < A = 64`): the atom block spans the whole well and
  the content there carries no information. That is the incumbent's property too, confirmed by the
  hub on the test leg, and `rank_over_d` is carried per well so it is visible rather than inferred.

## 7. ⛔ Incident

My `DOWNDATE-READY.json` was overwritten for ~90 s, and **the bug was mine**: `OUT_ROOT`/`READY`
were module-level and ignored `--out`, so a peer fitting a **test** store into its own tree
published into my gate — leaving `condition:"test"` beside `all_three_assertions_passed:true` for
assertions only ever run on `val`. My earlier merge fix stopped a *re-run* blanking the file; it
did not stop *another build* describing it. Both guards are now in and **tested to fire**: a
publish from outside `OUT_ROOT` is refused loudly, and a publish under a different `condition` is
refused. Descriptors were then rewritten by my own `--stage fit` on val
(`content_sha256 43dc1147067b6e55…`, unchanged from the original build).

## 8. Verdict

⭐⭐ **The design's premise survives.** The object composes — paired delta +0.0000 against
`pca_tensor` at the pin — and it forgets exactly: the downdated store equals the without-rows
refit to 1.164e-10, with no pass over the data and no row retained anywhere. Deleting 8.4% of the
pool and carrying the interface leaves composition intact at 0.8964 against 0.8941, with zero data
rows touched.
⛔ **And the price is fully visible, not netted:** a **12,258,640 B** sufficient statistic beside a
524,288 B store — an honest total of **24.45x the pin** — plus **127,488 B** of interface carry per
deletion. Whether that price is worth paying is `hub/single-design`'s call, not mine.

## 9. Artifacts

`/c/chlu/outputs/w8-downdate/`
`DOWNDATE-READY.json` (status **ready**) · `LOG.md` · `START_UTC.txt` · `w8-downdate.bundle`
`smoke.log` · `smoke_pinned.log` · `postdelete_refit.log` · `carry.log`
`build/val_seed0/` : `store.json` · `store.pkl` · `assertions.json` · `gauge.json` ·
`post_delete.json` · `post_delete_refit.json` · `carry.json` · `smoke.json` ·
`smoke_pinned.json` · `smoke_after_delete*.json` · `k_sweep.json` · address books.

---

**Addendum, 17:36Z — a stale log file, renamed so the outputs directory does not mislead.**
`smoke.log` held the **failed first attempt** (`KeyError: 'spec'`, 17:40Z entry), because the
successful re-run was executed in the foreground and its console output was not redirected to
that file. Renamed to `smoke_attempt1_FAILED_KeyError.log` so the name matches the contents. ⭐ The
successful SVD-basis smoke's results are in `build/val_seed0/smoke.json` and
`smoke_after_delete.json` (written 18:57 / 18:58 local), and are the numbers quoted in §4 of the
final report. `smoke_pinned.log`, `postdelete_refit.log` and `carry.log` are complete runs ending
in `done`.

---

**Addendum, 17:44Z — the three assertions INDEPENDENTLY REPRODUCED on the test leg.**
⭐ `hub/single-design` ran this spoke's instrument against the **test**-condition partition
([777, 939, 216, 58, 58], ranks [63, 63, 63, 13, 40]) in its own output tree, and all three
assertions pass there too:

| assertion | test leg (hub's build) | val leg (this spoke) |
|---|---|---|
| A1, pinned `object_hue=0.0` | [4.547e-13, 1.421e-14, **0**, 5.249e-12, 2.873e-14], worst **5.249e-12** = 4.40e-05 ulp @1.0, 1/5 bit-identical | worst **1.164e-10** = 9.77e-04 ulp @1.0, 2/5 bit-identical |
| A1, random 64 rows | worst 1.507e-11, 2/5 bit-identical | worst 1.798e-11, 1/5 bit-identical |
| A1, negative control | **FIRED** at 1.000e-03 | **FIRED** at 1.000e-03 |
| A2, no pass over data | 320 row-reads at 131,072 **and** at 65,536, 5 tripwires | identical |
| A3, items never kept | 12,885,866 B at both corpus sizes, **grew by 0** | 12,833,834 B, grew by 0 |

Ranks agree and every count is equal on both legs. ⭐ So the exactness claim is now measured on
**two partition legs, by two implementations, in two output trees** — the leg is the only
difference. ⛔ This does not make anything here scored: both legs are DEV, the hub's scored budget
is still 0/3, and the val gate this spoke publishes remains `condition: val` by construction.
⚠ The hub also confirms it will carry the **127,488 B carry cost as a binding PREREG line** rather
than a footnote, with the two interface lines side by side — this spoke's pinned basis (zero bytes,
insufficient alone, 0.0584) and the hub's carry (sufficient, 127,488 B, 0.8964) — and that neither
is allowed to stand in for the other, nor to be quoted beside the coupling hole's ONE byte.
⚠ And it records, before seeing any scored number, that the object that exists is this one at
K=5 with an honest sum of 24.55x on its leg, so ⛔ **rows 3–4 against `pca_tensor` cannot be a
matched-bytes claim under any outcome** — the overrun is printed, not netted.

⭐ One correction to my own §4 framing, on the hub's prompting and I agree with it: I under-sold
the **0.9005 refit control**. It is not a side diagnostic. Read together with
`downdate == refit-without at 1.164e-10`, it says the post-deletion store is as good as the
pre-deletion one (0.9005 vs 0.8941) — so **the deletion damages the content not at all**, the
entire 0.8941 → 0.058 collapse was interface staleness, and 0.9005 is also the score a store
fitted from scratch without those rows would get. That is the load-bearing sentence, and it
belongs beside the carry result rather than beneath it.

---

**Addendum, 17:50Z — a boundary on my own 17:44Z wording, before it can drift.**
⛔ The 17:44Z addendum says the exactness claim is "measured on two partition legs, by two
implementations, in two output trees". That is true and it is **not replication in the charter's
sense.** `hub/single-design` drew the line and it is right: charter A.5's "unreplicated = not a
result" is about a hub's **winning configuration re-run from its PREREG on a clean checkout**, and
that cannot happen until a scored run exists. What exists here is a cross-leg, cross-implementation
replication of the **store-exactness assertions** — a property of an object, measured twice. ⛔ It
is **not a replicated result**, it licenses **no row**, and nothing in this spoke's output may be
read as one. Recorded in my own log rather than only in the hub's, because the sentence a reader
could over-read is mine.

**Closing state, 17:52Z.** ⛔ `hub/single-design` declined the offer to re-run or extend the val
stages — nothing outstanding from this spoke. It is holding at **scored 0/3** and has surfaced the
reason rather than taken the call alone: the only object that exists is 24.55x the pin on its leg,
and its Amendment 2 pre-commits that rows 3–4 from it cannot be a matched-bytes claim under any
outcome. ⭐ The gate is no longer blocking anything: `DOWNDATE-READY.json` went **PARTIAL at
16:44Z** (8 minutes after START) and **ready at 17:31Z**. The hub vendored this spoke's three
files verbatim at its `bd9b1cf` with provenance in the commit message.

⭐ How the gauge was actually found, since neither side found it alone and the record should say
so: I measured that the read broke (0.0000, then 0.0584, then 0.0451); the hub measured **why** the
span moved (90.03% correspondence vs 5.76% deletion, 15.63x); I found the **pinned basis** (zero
bytes, 800,000x on the gauge, insufficient alone); the hub found the **carry** (sufficient,
127,488 B). Each of us withdrew at least one over-claim on the other's measurement — its
permutation advice and its fixture well sizes, my under-sold 0.9005 control and my `cut_is_marginal`
reporting bug. ⛔ None of it is scored. All of it is `val` or DEV.

**FINAL:** branch `w8/downdate` at `2fab429`, bundle verified, tree clean, no processes running,
`DOWNDATE-READY.json` status **ready**, condition **val**, seed 0. This spoke is done.

---

# ⛔⭐ REOPENED 17:51Z — THE TEST LEG, AND ASSERTION 1 **FAILS ON SEED 2**

The user directed me to run the assertions on the `test` condition as well, `single-design`
being blocked on a test-condition store for its RESULT rows. ⛔ Task §6 requires that a failure
of the downdate be reported **immediately**, so this section is written before any fix.

- **17:51Z** Instrument made condition- and seed-aware: `--seed` threads the bridge/feature seed
  through `fw.PINS`, and the readiness file is now **one file per object** —
  `DOWNDATE-READY.json` stays the pinned `val` gate and the test object gets
  `DOWNDATE-READY-test.json`. ⭐ A new `gate` stage aggregates the per-seed artifacts, because
  three seeds publishing into one file is the 19:29Z incident again with a different index: the
  last writer's descriptors would sit beside all three seeds' assertions. Top-level flags are the
  **AND** over seeds and the top-level deviation is the **WORST** over seeds — a mean would hide
  a seed that failed. Verified the val gate was untouched throughout.
- **17:53–18:07Z** Ran `fit` + `assertions` on test seeds 0, 1, 2 (partitions from w6's own
  scored dirs; feature cache generation 2, per-seed).

  | test seed | wells | well-3 rank | store | honest total | A1 worst dev | in ulp @1.0 | **A1** |
  |---|---|---|---|---|---|---|---|
  | 0 | [777, 939, 216, 58, 58] | 13/58 | 524,288 B | 12,869,482 B = 24.55x | 8.624e-12 | 0.0001 | **pass** |
  | 1 | [813, 882, 208, 73, 72] | 29/73 | 524,288 B | 12,500,170 B = 23.84x | 1.619e-09 | 0.0136 | **pass** |
  | 2 | [839, 847, 225, 65, 72] | 25/65 | 524,288 B | 12,409,722 B = 23.67x | **3.921e-06** | **32.90** | ⛔ **FAIL** |

  Assertions **2 and 3 pass on all three seeds** (320 row-reads at both corpus sizes; object bytes
  grew by 0 at 131,072 and 65,536 rows). The negative control **fired on all three**. Ranks and all
  counts agree on all three. ⛔ **Assertion 1 fails on seed 2** against my declared criterion of
  ≤ 1 float32 ulp at unit scale: **3.921e-06 = 32.90 ulp**, a factor of 33 over.
  `DOWNDATE-READY-test.json` is therefore **`status: partial`, `all_three_assertions_passed:
  false`** — published as it came out.
- **18:10Z** ⭐⭐ CAUSE, measured and quantitative — and it is **not the downdate arithmetic.**
  On seed 2 wells 0, 1, 2 the deviation is **exactly 0**; the failure is confined to **well 3**,
  the rank-deficient, tolerance-limited well, in every seed. Classical eigenvector perturbation
  theory says the rotation of an eigenvector goes as `||dC|| / gap`, and with the downdate's
  residual at machine epsilon the predictor is `eps / (smallest relative gap among KEPT
  directions)`. Measured:

  | test seed | smallest rel gap among kept | predicted `eps/gap` | observed max abs dev |
  |---|---|---|---|
  | 0 | 1.359e-03 | 1.62e-13 | 8.62e-12 |
  | 1 | 2.014e-09 | 1.09e-07 | 1.62e-09 |
  | 2 | **4.950e-11** | **4.45e-06** | **3.92e-06** |

  ⭐ Seed 2's prediction 4.445e-06 against an observed 3.921e-06 — the mechanism is identified,
  not guessed. Seed 2's well 3 keeps directions down to relative eigenvalue **7.54e-12** and
  among them sits a pair separated by only **4.95e-11** relative.
- **18:12Z** ⛔⭐ **WHAT I GOT WRONG, and it is a defect in the store's DEFINITION, one level
  deeper than the 16:52Z fix.** At 16:52Z I pinned the eigenvalue **magnitude** (`> 1e-12
  lambda_max`) and argued the cut was well-posed because its margins were wide. That was
  incomplete: what determines whether an eigenvector is a function of the data is not its
  eigenvalue's **magnitude** but its **gap** to its neighbours. A direction comfortably above the
  magnitude threshold can still be nearly degenerate with the one beside it, and then its
  direction is no more determined than a null direction's. ⛔ **I pinned the wrong quantity, and
  the val leg happened not to expose it** — val well 3's smallest kept gap is ~1e-3, seven orders
  safer than seed 2's. A single leg passing is exactly how this hid.
- **18:13Z** ⛔ Stated plainly, before any remedy: **the downdate does not equal the without-rows
  fit on test seed 2 to the precision I declared.** The sufficient statistics downdate correctly
  — three of five wells are bit-identical and the residual on `S` is at machine epsilon — but the
  **eigendecomposition of those statistics does not have a unique answer** in a near-degenerate
  cluster, so the store's *bytes* are not determined there by either path. ⛔ I will not retune
  the tolerance to make seed 2 pass; choosing a threshold by which seed survives is precisely the
  move I said at 16:52Z I was not making, and it would be that move for real.
- **18:16Z** Built the fit-time predictor (`well_posedness_report`) — `eps / gap` from the
  statistics alone, no deletion, no second fit — and then **validated it, and it does not work as
  a gate.** Across the four legs it over-predicts the observed deviation by **7.7x to 217,466x**
  and flags `store_bytes_well_posed = False` on **three legs that passed**:

  | leg | predicted worst | in ulp | observed worst | predictor's verdict vs reality |
  |---|---|---|---|---|
  | val 0 | 5.078e-07 | 4.26 | 1.164e-10 | **false alarm** |
  | test 0 | 1.875e-06 | 15.73 | 8.624e-12 | **false alarm** |
  | test 1 | 1.286e-07 | 1.08 | 2.592e-09 | **false alarm** |
  | test 2 | 3.023e-05 | 253.60 | 4.227e-06 | agrees |

  ⛔ Why it is loose: it bounds `||dC||` by `eps * lambda_max` **globally**, while the real
  residual inside a tiny-eigenvalue subspace is far smaller. It is a rigorous upper bound that
  certifies nothing. ⭐ It is kept — it localises the fragile well correctly, well 3 in all four
  legs — but it is labelled in its own docstring as a **useless gate**, because shipping it as
  "the object declares its precision" when it is five orders loose would be its own dishonesty.
- **18:19Z** ⭐⭐ WHAT DOES WORK, and it is measured rather than bounded: `--stage selfcheck`
  downdates **the same rows twice, presented in two different orders**. Both are valid downdates
  of the same set and both are entitled to the same answer, so any disagreement is pure numerical
  non-determinism in the accumulate-then-eigendecompose path. ⭐ It is a **FLOOR** on any
  exactness claim about the store on that data, it needs **no refit**, and it costs one extra
  downdate.

  | leg | floor (worst) | in float32 ulp | observed downdate-vs-refit | floor / observed |
  |---|---|---|---|---|
  | val 0 | 5.690e-12 | 0.00 | 1.164e-10 | 0.05 |
  | test 0 | 8.931e-12 | 0.00 | 8.624e-12 | 1.04 |
  | test 1 | 4.282e-09 | 0.04 | 2.592e-09 | 1.65 |
  | test 2 | **4.746e-06** | **39.82** | 4.227e-06 | **1.12** |

  ⭐ The floor tracks the observed deviation within a factor of ~2 on three legs and **1.12 on the
  failing one**, it isolates the same well (3) every time, and it flags **exactly** the seed that
  fails and no other. Where the `eps/gap` bound false-alarmed 3 times in 4, this discriminates
  cleanly at a 1-ulp threshold.
- **18:20Z** ⭐⭐ THE DECISIVE FACT, and it exonerates the downdate completely: on test seed 2 the
  **floor (4.746e-06) is LARGER than the observed downdate-vs-refit deviation (4.227e-06)**. The
  same rows downdated twice, in two orders, disagree by *more* than the downdate disagrees with
  the from-scratch refit. ⛔ So seed 2's store **cannot be reproduced to float32 precision by any
  method at all** — not by the downdate, not by a refit, not by re-running the identical
  computation. Its atom bytes are **not a function of its data**. The downdate is not merely
  blameless; it is performing slightly better than the object's own reproducibility.
- **18:21Z** THE TEST GATE, published as it came out —
  `/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json`, **`status: partial`**:

  | | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | store bytes | 524,288 ✓ | 524,288 ✓ | 524,288 ✓ |
  | honest total | 24.55x | 23.84x | 23.67x |
  | **A1 downdate == refit** | **pass** (8.624e-12) | **pass** (2.592e-09) | ⛔ **FAIL** (4.227e-06 = 35.46 ulp) |
  | A1 negative control | fired | fired | fired |
  | **A2 no pass over data** | **pass** | **pass** | **pass** |
  | **A3 items never kept** | **pass** | **pass** | **pass** |
  | reproducibility floor | 0.00 ulp | 0.04 ulp | **39.82 ulp** |

  ⛔ `all_three_assertions_passed: false`. Assertions **2 and 3 hold on every seed**; assertion 1
  holds on two of three. Top-level flags are the AND over seeds and the top-level deviation is the
  WORST over seeds, so the gate cannot read as a pass because two seeds did.
- **18:22Z** ⛔ WHAT I AM NOT DOING, stated so the record is unambiguous: I am **not** retuning
  `eig_rel_tol` so seed 2 passes, and I am **not** switching the comparison to the subspace
  projector so the failure disappears into a different metric. Both would work and both are the
  move I said at 16:52Z I was not making. The remedies that are *available* — a gap-aware rank
  cut, a different content, or declaring the subspace rather than the eigenvectors to be the
  object — each change what the store IS, and that is a design call above a build spoke. ⭐ What I
  owe is the measurement and the diagnosis, and both are here.
- **18:23Z** ⭐ Scope note on what the val result now means. The val gate is unchanged and still
  correct: `DOWNDATE-READY.json` remains `status: ready`, A1 worst 1.164e-10, floor 5.690e-12,
  all three assertions passing. ⛔ But "this object forgets exactly" is **no longer supportable as
  a general claim** — it is supported on val seed 0 and test seeds 0 and 1, and **falsified on
  test seed 2**, by a mechanism that is a property of the content rather than of the deletion.
  ⚠ Had the user not directed me to run the test condition, this spoke would have shipped a
  one-leg pass as if it were the property.

---

# ⛔⭐ 18:28Z — A1 RE-RUN UNDER THE HUB DESIGN CALL: **THE SUBSPACE DOES NOT RESCUE SEED 2**

**Design call (Hub, Head-cleared):** the stored object is the **SUBSPACE**, not the eigenvector
basis. Re-run A1 under that definition on all seeds including test seed 2. ⛔ Do **not** retune
`eig_rel_tol`. Implemented exactly as instructed: the store is **bit-for-bit the one that failed**
— `eig_rel_tol` is untouched at 1e-12 and `content_sha256` is unchanged — and only the question
asked of it has changed.

- **18:30Z** Implemented `compare_subspaces`: per well, the largest principal angle between the
  **deployed spans** (what `render_span` actually reads through), the span ranks, the same for the
  directions with the mean excluded, and the mean compared **entrywise** — the mean is `t/n`, a
  determined quantity, and is not a subspace question.
  ⚠ Guarded against the obvious hazard, that a softer ruler makes the failure vanish: the same two
  deletions are run, the **byte-level number is printed beside every subspace number**, and the
  **two-order floor is recomputed under the subspace metric**. If the span were genuinely
  determined, the same rows downdated twice must agree on it to ~0.
- **18:32Z** ⛔ TWO BUGS IN MY OWN MEASUREMENT, both caught by sanity checks rather than by
  reasoning, and both worth recording because either would have produced a confident wrong answer.
  1. **`deployed_span` used QR.** Un-pivoted Householder QR is **not rank-revealing**, so on a
     block whose mean lies nearly inside the directions' span (measured earlier: relative
     perpendicular norm **2.3e-08** in exactly these rank-deficient wells) the small entry of
     `diag(R)` is not in a known position and truncating to the first `r` columns of `Q` discards
     an **arbitrary** direction. ⛔ MEASURED consequence: two **bit-identical** stores (test seed 0,
     byte deviation exactly 0) reported a principal angle of **0.97 rad — 56 degrees**. The
     byte-level contrast printed beside it is what exposed it: an absurd number next to a
     known-zero one. Fixed by using the **SVD**, which is rank-revealing and ordered.
  2. **`arccos` of the singular values of `A^T B`** is the textbook principal-angle formula and is
     useless here: `arccos` is ill-conditioned at 1 with an error floor of `sqrt(eps) ~ 1.5e-08`
     rad. ⛔ MEASURED: comparing a store **with itself**, where the answer is exactly 0, returned
     **4.47e-08 rad** — the same order as the real signal on the passing legs, which would have
     drowned them in the metric's own noise. Fixed by the projector distance
     `||P_A - P_B||_2 = sin(theta_max)`, which is stable and exactly 0 for identical spans.
     ⭐ Verified: self-comparison now returns **exactly 0.000e+00 on all four legs**.
- **18:34Z** ⭐⭐ THE RESULT, on the fixed metric, all four legs:

  | leg | pinned deletion | random 64 | **floor (two orders)** | byte-level A1 | **verdict** |
  |---|---|---|---|---|---|
  | val 0 | 1.151e-10 rad (0.001 ulp) | 3.146e-11 (0.000) | 1.462e-11 (0.000) | 0.001 ulp | **pass** |
  | test 0 | 1.251e-11 rad (0.000 ulp) | 6.820e-12 (0.000) | 1.387e-11 (0.000) | 0.000 ulp | **pass** |
  | test 1 | 3.420e-09 rad (0.029 ulp) | 4.534e-09 (0.038) | 6.190e-09 (0.052) | 0.022 ulp | **pass** |
  | test 2 | **5.524e-06 rad (46.34 ulp)** | **7.196e-06 (60.36)** | **7.881e-06 (66.11)** | 35.46 ulp | ⛔ **FAIL** |

  ⭐ The mean is **exactly equal on every well of every leg** (`mean_max_abs_dev = 0.0`), and the
  span ranks agree everywhere. ⛔ **Test seed 2 still fails, and by more than it did at byte
  level** — 60.4 ulp against 35.5. The subspace definition did not soften the verdict; on this
  seed it sharpens it.
- **18:36Z** ⭐⭐ WHY THE SUBSPACE CANNOT HELP HERE, and it is the substantive finding. The guard
  fired: the **two-order floor under the subspace metric is 66.11 ulp**, *larger* than either
  observed deviation (46.3 and 60.4). So the same rows downdated twice disagree about the **span**
  by more than the downdate disagrees with the from-scratch refit. Seed 2's *subspace* is not a
  function of its data either.
  ⭐ The reason is visible in the spectrum measured at 18:10Z. Well 3's tightest degeneracy is not
  *inside* the kept set — it **straddles the rank boundary**:

  | | relative eigenvalue | gap to next |
  |---|---|---|
  | j=23 | 5.70e-11 | — |
  | j=24 **last kept** | 7.54e-12 | 4.95e-11 |
  | j=25 **first dropped** | 1.12e-16 | **7.54e-12** |

  ⛔ A rotation *within* the kept set preserves the span, and the subspace definition would have
  absorbed it perfectly. A rotation between a **kept** direction and a **dropped** one **moves the
  span**, and no definition of the object as a subspace can absorb that. The same mechanism is
  present on every leg — the smallest gap always straddles the boundary — and seed 2 is simply
  where it gets small enough to bite: test seed 0's boundary gap is 1.386e-10, **18x** wider.
- **18:38Z** ⛔ So the design call is discharged and the answer is negative: **defining the object
  as the subspace does not make the seed-2 failure go away**, because the instability is in *which
  directions are in the store at all*, not in *how they are oriented inside it*. ⭐ The direction
  responsible carries relative eigenvalue **7.54e-12** — it contributes 7.5e-12 of its well's
  variance. Whether a direction at that level belongs in the store is the live design question,
  and the number is on the record for whoever answers it. ⛔ I have not answered it, and
  `eig_rel_tol` is unchanged.
- **18:39Z** Gate updated: `DOWNDATE-READY-test.json` carries `a1_subspace` per seed and in
  aggregate — `passed_every_seed: false`, `seeds_failing: ["2"]`, worst 60.36 ulp, floor 66.11
  ulp — beside the byte-level A1 and the byte-level floor. ⛔ `status` remains **`partial`**.
  Assertions **2 and 3 still pass on every seed of both legs** and are untouched by any of this.

---

# ⭐ 18:45Z — THE VARIANCE-SHARE CUT, **DECLARED BEFORE THE RE-RUN**

**Hub decision:** the subspace call is withdrawn (it cannot absorb membership churn — 18:36Z).
Next: a variance-share rank cut **tied to the measured reproducibility floor**, declared before
re-running and applied uniformly to all seeds. ⛔ Not tuned so seed 2 passes; if it still fails at
a principled cut, report the falsification and the content cost.

## The input, measured first, with no deletion and no refit

`delta_rel` = the relative perturbation of the centred scatter `C` under a change that must not
matter — accumulating the **same** rows in a different order. It is the numerical noise floor of
the accumulate step, and it is what the reproducibility floor is made of. Measured per well on all
four legs:

| leg | delta_rel per well | leg worst |
|---|---|---|
| val 0 | 7.33e-17, 7.91e-17, 1.52e-16, 2.27e-16, 3.89e-16 | 3.8930e-16 |
| test 0 | 7.56e-17, 6.16e-17, 1.23e-16, 3.05e-16, 3.52e-16 | 3.5154e-16 |
| test 1 | 9.02e-17, 1.56e-16, 1.57e-16, 3.86e-16, 3.11e-16 | 3.8649e-16 |
| test 2 | 6.87e-17, 5.84e-17, 2.55e-16, 4.94e-16, 4.09e-16 | **4.9411e-16** |

⭐ **`delta_rel_worst = 4.941056e-16`**, over all four legs and all wells.

## ⛔ THE DECLARATION — fixed now, before anything is re-run

```
tau  =  delta_rel_worst / ulp_float32
     =  4.941056e-16 / 1.1920929e-07
     =  4.144858e-09
```

**Admission rule.** Direction `j` is admitted iff its share of its well's variance satisfies

```
lambda_j / sum_i max(lambda_i, 0)  >=  tau        (tau = 4.144858e-09)
```

in addition to the pre-existing limits `j < A-1` and `j < d_k`. ⛔ `eig_rel_tol` stays at **1e-12,
untouched** — it was never retuned and is not retuned now; whichever of the two binds, binds.

**Why this number and not another.** `delta_rel` is the precision to which the accumulate step
determines `C` at all. `ulp_float32` is the finest angular distinction a float32 store can
represent, and is therefore the target. Their ratio is the variance share below which a
direction's eigenvalue sits at or under the level the accumulation can resolve to that target — so
its orientation is not a function of the data, and it is not content. ⭐ **There is no free
parameter**: one measured quantity, one target fixed by the storage format, and their quotient.

⛔ **Not rounded.** I considered rounding 4.1449e-09 up to a clean 1e-8 and rejected it, and the
reason is worth recording because it runs opposite to intuition: rounding **up** makes the cut
**stricter**, which drops *more* marginal directions and is therefore *more* likely to make seed 2
pass. Rounding up is conservative about content and **anti**-conservative about the verdict. Using
the measured quotient exactly leaves me no room to have chosen.

**Uniform.** One `tau` for all four legs and both conditions, as instructed.

**What would falsify the remedy.** If test seed 2 still fails A1 at this cut, that is a real
falsification of the design at a principled threshold, and it is to be reported with the content
cost rather than adjusted away. ⛔ I will not move `tau` after seeing the result.

## 19:20Z — THE RESULT AT THE DECLARED CUT: **A1 PASSES ON ALL FOUR LEGS**

⛔ `tau = 4.144858e-09` exactly as declared at 18:45Z, applied uniformly, **not moved**.
`eig_rel_tol` unchanged at 1e-12 on every leg (verified in all four `store.json`). Both cuts
applied; whichever binds, binds.

### Assertion 1, byte level, before and after the declared cut

| leg | A1 worst BEFORE | in ulp | A1 worst AFTER | in ulp | floor AFTER | verdict |
|---|---|---|---|---|---|---|
| val 0 | 1.164e-10 | 0.0010 | 1.164e-10 | 0.0010 | 3.887e-12 | pass |
| test 0 | 8.624e-12 | 0.0001 | **4.547e-13** | 0.0000 | 2.127e-13 | pass |
| test 1 | 2.592e-09 | 0.0217 | **1.126e-09** | 0.0094 | 4.282e-09 | pass |
| test 2 | **4.227e-06** | **35.46** | **2.842e-14** | **0.0000** | 5.684e-14 | ⭐ **pass** |

⭐⭐ **Test seed 2 goes from 4.227e-06 (35.46 ulp, a 35x failure) to 2.842e-14 (2.4e-07 ulp) — a
factor of 1.5e+08.** Its reproducibility floor falls from 4.746e-06 (39.8 ulp) to 5.684e-14. No
seed is now flagged as irreproducible at float32. The negative control fires on every seed.
Assertions 2 and 3 were already passing on every seed and are unaffected.
⭐ `DOWNDATE-READY-test.json` is now **`status: ready`, `all_three_assertions_passed: true`,
`seeds: [0, 1, 2]`**, worst A1 over seeds **1.126e-09 = 0.0094 ulp**, worst floor **0.0359 ulp**.

### ⛔ THE CONTENT COST — the invoice, which the hub asked for either way

| leg | ranks before cut | ranks after cut | directions dropped | worst variance share lost in a well |
|---|---|---|---|---|
| val 0 | [63, 63, 63, **16**, 41] | [63, 63, 63, **13**, 41] | 3 | 4.24e-09 |
| test 0 | [63, 63, 63, **13**, 40] | [63, 63, 63, **12**, 40] | 1 | 5.76e-11 |
| test 1 | [63, 63, 63, **29**, 63] | [63, 63, 63, **25**, 63] | 4 | 8.94e-09 |
| test 2 | [63, 63, 63, **25**, 63] | [63, 63, 63, **20**, 63] | 5 | 2.99e-09 |

⭐ Every dropped direction is in **well 3**, the rank-deficient one; wells 0, 1, 2 and 4 are
untouched on every leg. The cost is **1 to 5 directions** and at most **8.94e-09 of a well's
variance** — nine parts per billion. ⛔ The store's **bytes do not change**: dropped directions are
written as zero rows, so the 524,288 B pin holds exactly and the cost is content, not budget.
⚠ It is still a cost and it is not nothing: `rank/d` in well 3 moves from 0.31 to 0.21 on test
seed 2, so the deployed span in that well shrinks by a fifth. Whether that is visible downstream
is a read question, measured below and not asserted here.

### Why this is a principled cut and not a tuned one — the record

- `tau` was **written into `LOG.md` and into the code at 18:45Z, before the re-run**, and has not
  been touched since. The same number appears in all four artifacts.
- It has **no free parameter**: `delta_rel_worst` (measured, 4.941056e-16) over `ulp_float32`
  (fixed by the storage format, 1.1920929e-07).
- ⛔ I explicitly declined to round it to a clean 1e-8, **because rounding up would have been
  stricter and therefore more likely to make seed 2 pass** — the opposite of conservative for the
  question at hand.
- It is **uniform**: one `tau` for four legs and two conditions.
- The cut binds in **exactly one well** on every leg, and that well was independently identified
  as the fragile one at 18:10Z by a different measurement (the eigenvalue gap), before this cut
  existed.

⚠ ⛔ WHAT I AM **NOT** CLAIMING. That seed 2 passes does not mean the earlier failure was
illusory or that the design was never in trouble. It means a direction carrying **7.54e-12** of
its well's variance was in the store, its orientation was not a function of the data, and
removing it — at a threshold fixed by measurement rather than by outcome — removes the failure.
⭐ Had the cut been declared and seed 2 still failed, that would have been the falsification, and
it would have been reported in this section instead.

## 19:40Z — THE READ, RE-ESTABLISHED ON THE CUT STORE

⛔ The declared cut **changed the store** (`content_sha256 43dc1147… -> 671b4e2f…`, well 3 rank
16 -> 13 on val), so every read number measured before it described an object that no longer
exists. I marked them stale in the gate immediately, deleted the val address books fitted against
the old atoms, and re-fitted from scratch. ⭐ Nothing here is carried over.

| leg | before the cut | after the cut |
|---|---|---|
| `downdate_store` val valid mean | 0.8941 (sd 0.0903) | **0.8941 (sd 0.0903)** |
| `pca_tensor` on the same corners | 0.8941 | **0.8941** |
| paired delta over 24 corners | +0.0000 | **+0.0000** |
| after delete, book unchanged | 0.0584 | **0.0584** |
| after delete, book **carried** | 0.8964 | **0.8964** |
| carry recovers | 100.3% | **100.3%** |
| gauge: SVD basis / pinned basis | 2.994e-03 / 3.725e-09 | **2.994e-03 / 3.725e-09** |

⭐⭐ **The content cost is invisible downstream.** Dropping 3 directions carrying 4.24e-09 of well
3's variance changes the val read by **nothing at four decimal places** — the paired delta against
`pca_tensor` is still exactly +0.0000 on all 24 corners, and the carry still recovers the full
collapse with **0 data rows touched**. The carry's byte cost falls slightly, 127,488 B -> **125,952
B**, because the deployed span is one direction narrower.
⚠ So the 19:20Z caveat resolves: `rank/d` in well 3 does shrink, and it does not cost anything the
judge can see on this condition. ⛔ That is a `val` measurement of a `val`-sized effect; it is not
a claim that the cut is free on the scored corners.
- **19:42Z** Three gate fields are **NOT re-run and are marked SUPERSEDED rather than refreshed**,
  each pointing at the fresh field that replaces it: `smoke` (SVD basis — its `pca_tensor`
  comparison is reproduced exactly by `smoke_pinned`), `post_delete` (its re-indexed leg was a
  negative result whose conclusion is unchanged), and `post_delete_refit` (0.9005 with the book
  refitted — superseded by the carry's 0.8964 with **no** refit, which is the stronger form of the
  same finding). ⛔ Marked, not silently left looking current, and not quietly deleted either.

---

# ⭐⭐ FINAL REPORT — w8 spoke/downdate-build

⚠ **This supersedes the earlier "FINAL REPORT" section written at 17:30Z**, which was accurate at
the time and was written before the user directed me to run the test condition. That section is
left in place unedited, per the never-rewritten rule; where the two differ, this one governs. The
substantive difference: at 17:30Z assertion 1 had been measured on **one leg**, and a one-leg pass
was on the verge of being shipped as the property.

**Agent:** experiment-engineer spoke, build only. **Base:** `base/iter7 = ec2ef7c`.
**Branch:** `w8/downdate` · **Start** 2026-09-20T16:36:33Z.
**Feature cache:** generation 2 (`ef6359a4cd54b082`), per seed.
⛔ **No scored run. No claim. `hub/single-design` scores this object** and is cleared to do so.

## 1. The object

`chlu/core/downdate_store.py` :: `DowndatableStore` — four parts, one object, at the programme's
existing span interface so `gen_read` reads it unchanged and no other arm's code path moves.
`chlu/experiments/exp_w8_downdate.py` :: ten stages. `tests/test_downdate_store.py` :: **17 tests,
~15 s, no feature cache**, every negative control asserted to fire.

The mechanism in one line: a principal subspace is a function of a **second moment**, a second
moment is a **sum over rows**, and a sum can be un-summed — so the store keeps no row and still
forgets a row exactly.

## 2. ⭐⭐ THE THREE ASSERTIONS — the deliverable. All three pass, on **four legs**.

**val seed 0** and **test seeds 0, 1, 2**, at the declared variance-share cut.

**1 — the downdated store EQUALS the store fitted WITHOUT those rows.**

| leg | A1 worst (pinned 11,024 rows + random 64) | in float32 ulp @1.0 | reproducibility floor |
|---|---|---|---|
| val 0 | 1.164e-10 | 0.0010 | 3.887e-12 |
| test 0 | 4.547e-13 | 0.0000 | 2.127e-13 |
| test 1 | 1.126e-09 | 0.0094 | 4.282e-09 |
| test 2 | 2.842e-14 | 0.0000 | 5.684e-14 |

Ranks agree in every well; row counts, `term_n` and `pair_n` exactly equal; the negative control
(one atom nudged by 1e-3) **fires on every seed**.

**2 — no pass over the data**, mechanically, two independent proofs: every named route to the
corpus (`features_mmap`, `gather`, `np.load`, `np.memmap`, `np.fromfile`) **raises** during the
call, and the row counter reads **320 row-reads = 64 rows × 5 wells at BOTH 131,072 and 65,536
corpus rows**, on every seed. Invariant to the corpus, not merely small.

**3 — items are never kept**, mechanically: a 26-array inventory cross-checked against
`vars(store)`, every shape explained by `(A, d_k, K, T, NC)`, no axis the corpus size, and the
decisive form — **object bytes identical at two corpus sizes, grew by 0 B**, on every seed.
Also asserts `n > d_k` per well, the one case where a covariance can expose its own rows.

## 3. ⛔ The ledger — four parts, separate, never netted

| part | val seed 0 | worst over test seeds |
|---|---:|---:|
| 1. principal subspaces (**the store**) | **524,288 B** | **524,288 B** (matches the pin on every seed) |
| 2. running covariance, float64 `d_k^2` | 12,258,640 B | 12,310,672 B |
| 3. request count table | 11,488 B | 11,488 B |
| 4. coupling + holes | 5,288 + 1,322 B | 5,288 + 1,322 B |
| **honest total** | **12,817,450 B = 24.45x** | **12,869,482 B = 24.55x** |

⛔ **The honest ledger is ~24.5x over the pin.** The *content* is at 524,288 B exactly; the
overrun is the price of exact deletion. Printed, not netted against the store and not netted
against the address book. 11.7x if the statistic is held as a float64 symmetric triangle, 6.4x as
float32 — but float64 is not optional: the downdate residual on `S` reaches 4.95e-10 on entries of
order 1e+9.

## 4. ⭐⭐ It composes AND it forgets

val, 24 corners, shared frozen decoder `01ef91f75eea` + judge `c8741ac417d8`, re-measured from
scratch on the final store:

| leg | valid mean | data rows touched | refit |
|---|---|---|---|
| `pca_tensor` (incumbent) | 0.8941 | — | — |
| `downdate_store`, before delete | **0.8941** | — | — |
| after delete, book unchanged | 0.0584 | 0 | no |
| after delete, **book carried** | **0.8964** | **0** | **no** |

**Paired delta against `pca_tensor`: +0.0000 on every one of the 24 corners.**
⭐ Delete 11,024 rows — 8.4% of the pool, an entire factor value — touch **zero data rows**, refit
**nothing**, and composition is intact. The carry recovers **100.3%** of the collapse.
⛔ Its cost, declared: it rewrites **125,952 B of the address book's 158,664 B** (values, not
count). Not the coupling hole's ONE byte and never to be quoted beside it.

## 5. What was found that was not in the plan

**(a) The store's definition was not well-posed — and the defect is the INCUMBENT's.** Wells 3/4
are 80%/32% dead channels, so `min(A-1, d_k)` wrote eigenvectors of a numerically null eigenspace,
where the eigenvector is not a function of the data. Raw deviation there: **1.0**. `PCASpanStore`
has the identical non-uniqueness and had simply never been asked to be a function of its fit set
twice. **The deletion property is what finally asked the question.**

**(b) An exactly-forgetting store can still read completely wrong.** The naive post-delete read
collapsed to 0.0000. My atom bytes differ from `PCASpanStore`'s by up to **2.00** per well yet
render the same vector to **1.6e-6 relative** — the content is gauge-arbitrary alone; only the
*(basis, book) pair* means anything. `gen_read.span_basis` takes the SVD of an atom block whose
spectrum is maximally degenerate, so a 1e-10 atom change rotates it by O(1); `pinned_span_basis`
removes that at zero bytes (2.994e-03 → 3.725e-09), and the hub's carry handles the rest.

**(c) ⛔ ASSERTION 1 FAILED ON TEST SEED 2, and the failure is the most valuable thing here.**
At 35.46 ulp — a 35x failure — invisible on val. Cause: I had pinned the eigenvalue **magnitude**
and what determines whether an eigenvector is a function of the data is its **gap** to its
neighbours. Seed 2's well 3 kept a direction at relative eigenvalue 7.54e-12 with a neighbour
4.95e-11 away. ⭐ **The decisive measurement**: the two-order reproducibility floor (4.746e-06)
**exceeded** the observed deviation (4.227e-06) — the same rows downdated twice disagreed by more
than the downdate disagreed with the refit, so the failure was a property of the **content**, not
of the deletion.

**(d) The subspace definition could not rescue it** (hub call, since withdrawn). Measured
faithfully and reported negative: seed 2 failed *worse* under it (60.4 ulp), and the subspace
floor (66.1 ulp) again exceeded the deviation, because the degeneracy **straddles the rank
boundary** — a rotation between a *kept* and a *dropped* direction moves the span, and no
subspace definition can absorb that.

**(e) The declared variance-share cut resolved it.** `tau = delta_rel_worst / ulp_float32 =
4.144858e-09`, declared in LOG and code **before** the re-run, uniform across seeds, no free
parameter. Seed 2: **4.227e-06 → 2.842e-14**, a factor of 1.5e+08. **Content cost:** 1–5
directions per leg, all in well 3, at most **8.94e-09 of a well's variance**; store bytes
unchanged (zero rows); and **invisible downstream** — paired delta still +0.0000, carry still
0.8964.

## 6. ⛔ Declared NOT-RUN and not-claimed — never nulls

- **No scored run, no claim.** Everything is `val` or DEV.
- **The read smoke, gauge and carry are `val` only.** Not run on `test`. A declared NOT-RUN.
- **Row 3 is not measured** — the val harness scores only the 24 `val_nway` corners.
- **The matched-bytes variant (K≈37, A=9) was NOT built** — §2/§3 forbid netting the statistic
  into the store's budget. Arithmetic handed over as `k_sweep.json`; the hub built it.
- **`delete()` is handed the rows; it does not look them up.** The store never held them and does
  not after. What is asserted is that the *retained* corpus is never opened.
- **Wells 3 and 4 are saturated** (`d_k < A`) — the incumbent's property too; `rank_over_d` is
  carried per well.
- **Three gate fields are SUPERSEDED, not refreshed**: `smoke`, `post_delete`,
  `post_delete_refit`, each pointing at the fresh field that replaces it.
- **A fit-time `eps/gap` predictor was built and does NOT work as a gate** — it over-predicts by
  7.7x to 217,466x and false-alarms on three legs that passed. Kept and labelled useless as a
  gate, because shipping it as "the object declares its precision" would have been its own
  dishonesty.

## 7. ⛔ Mistakes, in one place

1. **Pinned the wrong quantity** (eigenvalue magnitude, not gap) at 16:52Z, and argued
   well-posedness from margins that did not govern. Caught only by the test leg.
2. **`cut_is_marginal` reported True** on a cut whose real margins are 2e3 / 2e-4, by scoring
   wells where the *atom budget* binds. Fixed.
3. **`publish_ready` had no guard**, so a peer's test-condition build overwrote my gate for ~90 s,
   leaving `condition:"test"` beside `all_three_assertions_passed:true`. Both guards now tested to
   fire.
4. **`deployed_span` used QR**, which is not rank-revealing: two **bit-identical** stores reported
   a 0.97 rad subspace angle. Caught by printing the byte-level number beside it.
5. **`arccos` principal angles** have a sqrt(eps) ~1.5e-08 floor: a store compared with *itself*
   returned 4.47e-08 rad. Replaced by the projector distance, verified exactly 0.
6. **`fw.arm_features` instead of `R.arm_features`** cost a 5.5-minute address-book fit.
7. **A unit-test fixture whose dead channels were exactly zero did not reproduce the degeneracy**
   (`eigh` resolves an exactly-zero block deterministically), and an `n <= d_k` test passed for
   the wrong reason. Both fixed.

⭐ Items 4 and 5 were caught by sanity checks rather than reasoning, and item 1 only because the
user directed a second leg. That is the pattern worth keeping.

## 8. Verdict

⭐⭐ **The design's premise holds, on four legs.** The object composes — paired delta +0.0000
against `pca_tensor` at the pin — and it forgets exactly: the downdated store equals the
without-rows refit to within 0.01 of one float32 ulp on every seed, with no pass over the data and
no row retained anywhere. Deleting 8.4% of the pool and carrying the interface leaves composition
intact with zero data rows touched.
⛔ **And the price is fully visible:** a **12.3 MB** sufficient statistic beside a 524,288 B store
— an honest total of **~24.5x the pin** — plus **125,952 B** of interface carry per deletion, plus
a content cut that removes up to 5 directions per well. Whether that price is worth paying is
`hub/single-design`'s call and the Head's, not mine.
⚠ **The result is only this strong because it was allowed to fail first.** Assertion 1 failed on
test seed 2 at 35 ulp; the threshold was not tuned to hide it, and the hub's own subspace remedy
was implemented and reported negative. The cut that fixed it was declared before it was tested.

## 9. Artifacts

`/c/chlu/outputs/w8-downdate/`
**`DOWNDATE-READY.json`** — val, `status: ready` · **`DOWNDATE-READY-test.json`** — test,
`status: ready`, seeds [0,1,2], `all_three_assertions_passed: true`
`LOG.md` · `START_UTC.txt` · `w8-downdate.bundle` · nine run logs ·
`DOWNDATE-READY.json.clobbered-by-single-design.bak` (kept as incident evidence)
`build/{val_seed0,test_seed0,test_seed1,test_seed2}/` : `store.json` · `store.pkl` ·
`assertions.json` · `selfcheck.json` · `a1_subspace.json` · `gauge.json` · `carry.json` ·
`smoke*.json` · `post_delete*.json` · `k_sweep.json` · address books.
