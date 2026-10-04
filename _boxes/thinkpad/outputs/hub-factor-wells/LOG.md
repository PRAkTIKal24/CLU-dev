# hub/factor-wells — running log

TARGET: compositional generation, scoreboard rows 3-4, with FACTOR-ALIGNED wells.
BRANCH: hub/factor-wells @ afd37cf (pin 4125fce + w3-value-holdout additive modules).
SCORED BUDGET: `composition` target, 1 spent, 2 left.

---

## 2026-09-18

- 22:20Z **START.** No live `START_UTC.txt` in `/c/chlu/outputs/hub-factor-wells` -> wrote one
  (`2026-09-18T22:20:11Z`, pid 2779, host FORGIS_14, branch `hub/factor-wells`, head `afd37cf`).
  Booted charter -> w3-hub-common -> hub-factor-wells.md, nothing else. Did not open the campaign
  ledgers or any wave-2 report.
- 22:20Z **Environment verified.** `chlu.__file__` = `C:\chlu\worktrees\CHLU-hub-factor-wells\chlu\__init__.py`
  — inside my worktree, so numbers from this process are mine. jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4,
  python 3.11.9. System python, no venv created, nothing installed.
- 22:35Z **Instrument survey — what exists and what I must build.**
  - `chlu/experiments/gen_scoreboard.py` (962 lines, additive w3 module) **scores**, it does not run a
    model: an arm hands it per-row `make_outcome(accept, valid, tuples, threshold, setting, byte_ledger)`.
    `PRIMARY_CONTROL = "basis_head"`, `CONTROL_ARMS = (basis_head, knn_proto)`, `LAUNDERING_ARM =
    "grid_syntax_rule"`, `byte_budget = 524_288`, `byte_match_tolerance = 0`, `decoration_margin` is a
    declared-in-advance field. `verdict_vs_control` writes the "decoration" sentence itself.
  - `chlu/data/gen_value_holdout.py` emitted the splits already present at
    `CHLU_GEN_DATA_ROOT/shapes3d/value_holdout/`: **49 `nway-*` cells of exactly 72 rows each** (3528
    rows — the multi-corner estimand, ≥49 corners, already built), **3 `pair-*` cells of 12,000 rows**,
    2 `value-*` cells, and query sets `row1_seen` / `row2_unseen` / `row5_held_out` / `c_invalid` /
    `a7_redundant` at 512 records each. Held-out values: `object_hue=0.5`, `shape=0.0`.
    Split: train 284,472 / test 195,528 of 480,000.
  - ⛔ **There is no runner.** `exp_gen_cell` runs ONE 72-item cell on the ordinary split. `basis_head`
    exists only as a *name* in the scoreboard — no implementation anywhere in the tree (grepped).
    ⇒ I must build (a) the multi-corner runner over the value-holdout split, (b) `basis_head`.
- 22:35Z **⭐ The feature cache is FULL — 480,000 rows, D=2048, float16, 1.97 GB, `scope: "full"`.**
  Every row of Shapes3D is already encoded. ⇒ **no encoding, no cache extension, so the re-encode gate
  never binds and there is no GPU hazard on this hub.** Compute left is: store fit, head fit, decode
  ~3.5k images per arm, classifier. **This is a CPU hub. I do not expect to request a GPU.**
- 22:52Z **The pinned store plan, read from the frozen instrument.**
  `gen-store-grouping/runs/shapes3d_recon/store_plan.json`: `K=5`, `n_modes=64`, `D=2048`,
  `feat_shape=[128,4,4]`, `store_bytes=524288`, `bits_per_read=30.0`, ward grouping,
  `well_sizes=[303,305,573,501,366]`, `n_dead_channels=71`. Its own note: *"only the grouping (the
  `labels` field) is allowed to change between rungs."* ⭐ That is exactly my dial, and it means my arm
  and its control can be made to differ in **the label vector and nothing else**.
  ⭐ `GenStore` docstring confirms `store_bytes = M·D·4` **independently of K** — but `bits_per_read =
  K·log2(M)` is **not** independent of K. ⇒ holding K=5 holds BOTH the byte ledger and the read-channel
  width fixed. I hold K=5.
- 22:52Z **DESIGN — the arms, and why the comparison is a true like-for-like.**
  Every arm below has the identical store fit recipe, identical read, identical head, identical
  524,288 B ledger and identical 30-bit read channel. **The label vector is the only thing that moves.**
  | arm | partition | role |
  |---|---|---|
  | `factor_aligned` | channels grouped by address-book relevance to factors | ⭐ my arm |
  | `basis_head` | the shipped ward correlation grouping, fitted atoms + linear head | ⭐ THE like-for-like (w3 §1) |
  | `basis_head_k1` | ⛔ NO partition at all — one well, 2048 channels, 64 atoms | the strict reading of "unfactored basis"; ⚠ NOT K-matched, declared |
  | `random_dmatched_factor` | `factor_aligned`'s labels permuted (size histogram held) | laundering floor for my arm |
  | `random_dmatched_ward` | ward labels permuted | laundering floor for the control |
  | `landscape_removed` | atoms zeroed / shuffled | charter §C.2, must collapse |
  | `knn_proto` | — | w3 §1 second row |
  | `matched_bytes_lookup` | atom-mean / kNN on the same atoms | charter §C.1 floor |
  ⚠ `basis_head_k1` is reported, never used as the bar: K=1 changes the reachable subspace from
  K·M = 320 dimensions to M = 64, so a K=5-vs-K=1 margin is **not** a partition effect. The bar is
  `factor_aligned` vs `basis_head`, both at K=5.
- 22:52Z **DESIGN — the read, held fixed across every arm.** Common §4.4/4.5 are binding: the read verb
  is the largest measured effect in the programme and the store is used as a **signed basis**, not a
  mixture (~47 % negative coefficients, 0.00 % of wells form a probability vector). So the read is the
  signed linear span: `x̂_k = c_k(request)ᵀ G_k`, coefficients from a request-conditioned head,
  `G_k` = well k's 64 modes. ⛔ No simplex is imposed. ⛔ No feature enters the read — the head's only
  input is the request, so the no-bypass property is structural and I assert it.
  ⚠ Stated plainly so no reader has to infer it: **under this read the CLU arm and `basis_head` are the
  same architecture.** That is not a defect of the design, it is what makes the partition the only
  moving part — and it is precisely the falsification the target file names.
- 22:52Z **DESIGN — the multi-corner estimand.** Row 4: the **49 `nway-*` corners** already built, 72
  rows each, paired per corner. Row 3: the 3 `pair-*` cells are only 3 corners, so I tile each into
  Montero-shaped sub-corners with the **same widths the n-way builder uses** (floor/wall band 2, scale
  corner 3, orientation corner 3; object_hue and shape pinned by the pair) → 5 floor bands x 2 scale
  corners x 2 pose corners = **20 sub-corners of 36 rows per pair cell, 60 corners total**. ≥ 49, paired,
  structurally disjoint by the same argument the n-way builder uses.
- 23:18Z **Built the harness.** Three new modules in the worktree (code, not outputs):
  `chlu/core/gen_factor_partition.py` (the dial: channel-factor relevance -> labels, plus the
  label-path audit and the rank-headroom check), `chlu/core/gen_request_head.py` (the read: signed
  linear span with a request-conditioned ridge head, pair-interaction block as one switch), and
  `chlu/experiments/exp_gen_factor_wells.py` (the multi-corner runner, stages `corners` / `partition` /
  `fit` / `run`, every stage resumable and wall-clock capped).
- 23:18Z **CORNERS PROVEN (stage `corners`, 15 s).** Row 4: **49 n-way corners, 72 rows each, 3528
  rows**, pairwise disjoint, every row inside the hold-out. Row 3: my tiling of the 3 `pair-*` cells at
  the n-way builder's own widths gives **60 sub-corners, 36 rows each, 2160 rows**, also pairwise
  disjoint and entirely held out. ⇒ both target rows have ≥49 paired corners. Artifact:
  `run/corners.json`.
- 23:18Z **⭐ THE PARTITION, MEASURED (stage `partition`, 59 s).** Fit pool = 131,072 rows drawn by a
  pinned RNG from the value-holdout train index; `assert_relevance_is_train_only` reports **0 rows
  shared with the hold-out**. Channel-factor relevance `I(bin(channel); factor)` over 2048 channels x 6
  factors in 41 s, 8 quantile bins.
  | grouping | well sizes | in-block |corr| mass | lift over its OWN size-matched random floor |
  |---|---|---|---|
  | `ward` (shipped) | 332 · 364 · 421 · 411 · 520 | 0.4715 | **2.306** |
  | `factor_aligned` | 777 · 939 · 216 · 58 · 58 | 0.7527 | **2.053** |
  | `random_ward` | same as ward | 0.2050 | 1.002 |
  | `random_factor` | same as factor | 0.3649 | 0.995 |
  ARI(`factor_aligned`, `ward`) = **0.2776** — the two partitions are genuinely different objects, not
  relabellings of each other. Both floors sit at lift ≈ 1.00, which is what a floor is supposed to do.
- 23:18Z **⚠⚠ THE MEASUREMENT THAT SHAPES THE REST OF THIS HUB — the frozen encoder barely carries
  `shape` and `scale` in separable channels.** Channels whose top factor is:
  `wall_hue` 935 · `floor_hue` 777 · `object_hue` 216 · `orientation` 40 · **`shape` 12** · **`scale` 4**
  (of 1984 live). Best single-channel MI: floor_hue 2.066 b · wall_hue 1.880 b · orientation 1.197 b ·
  object_hue 1.179 b · **scale 0.317 b** · **shape 0.250 b**.
  ⇒ a pure argmax factor-aligned partition puts 4 channels in a `scale` well and 12 in a `shape` well.
  The declared merge rule (`min_relevance_mass`) folded `scale` into `wall_hue`, and the inherited dead
  -channel rule (dead -> smallest well, identical to `correlation_grouping`) then filled the two small
  wells to 58 each — **46 of well 3's 58 channels are DEAD**. Both small wells are also **starved**
  (58 < M = 64): the arm's reachable subspace is `sum_k min(d_k, M)` = **308**, against ward's **320**.
  ⭐ That is a handicap in the CONSERVATIVE direction for my own arm and it is declared, not repaired.
  ⛔ This is not a bug in the partition — it is a property of the frozen encoder, measured, and it is
  the mechanism-level reason a factor-aligned partition could fail on this instrument.
- 23:18Z **NEXT (mechanism iteration, §A.1, no scored run involved).** Add `factor_aligned_balanced`:
  the same argmax relevance, then rebalanced to equal wells by moving the lowest-**margin** channels
  (max relevance minus second-best) out of oversized groups. It keeps the factor alignment, removes the
  starvation, and is the balanced variant the target file explicitly sanctions ("a balanced variant
  cutting the dendrogram leaf order gives equal wells at any K"). Train-only, deterministic, no label
  beyond the address book. ⛔ Declared BEFORE any arm is fitted and before any scored run.
- 23:52Z **`factor_aligned_balanced` measured, and it is the arm I will pre-register.** Equal wells
  **[410, 410, 410, 409, 409]**, **0 starved wells**, reachable read rank **320 — exactly ward's 320**,
  block-diagonality lift **2.237** against ward's **2.306** and its own size-matched random floor's
  **1.004**. ARI vs ward **0.337**, vs the unbalanced factor arm **0.273** — a genuinely different
  partition, not a relabelling. Group provenance: well 0 `floor_hue` · well 1 `wall_hue`+`scale` ·
  well 2 `object_hue` · well 3 `shape` · well 4 `orientation`.
  ⇒ against `basis_head` this arm matches **K, M, D, store bytes, bits-per-read, reachable rank, the
  read verb, the head design, the ridge, the fit pool, the decoder and the judge**. The label vector is
  the only thing that moves. The unbalanced `factor_aligned` arm is kept and reported beside it.
- 23:52Z **⚠ A MEASURED PERFORMANCE DEFECT IN THE SHIPPED `gen_store.train_store`, and what I did.**
  First `basis_head` fit ran **1.33 s per adam step** and was killed at step 240/1200 (~27 min per arm,
  14 arms = 6 h of pure recompilation). Cause, diagnosed not guessed: `train_store` passes `tau` to an
  `eqx.filter_jit`-ed step as a **Python float**, which `filter_jit` treats as STATIC; `tau` moves every
  step, so the step function is recompiled 1200 times.
  ⭐ I did **not** edit the pin. I wrote `train_store_traced_tau` in my own module with the identical
  loss (soft occupancy in GRAM form + commitment pull), identical `optax.adam(0.003)`, identical
  geometric anneal 1.0 -> 0.1, identical `jax.random.split` / `choice(replace=False)` minibatch stream
  from the same key — `tau` passed as a traced `jnp.float32` instead.
  **Verified side by side, 40 steps, same init, same key:** shipped **51.89 s** vs traced **2.30 s**
  (22.6x); max |loss difference| over the 40 steps **0.597** on a loss of ~860 (0.07 %); reconstruction
  MSE 1.68627071 vs 1.68607640 (1.2e-4 relative); max |mode difference| 0.046.
  ⚠ **It is NOT bit-identical** — constant-folded vs traced `tau` changes float32 accumulation order —
  and I say so rather than claim equivalence. What matters for this hub is that it is **the same fitter
  for every arm**, so it is a constant of the comparison and cannot favour one partition.
- 00:20Z **PREREG written and the margin declared, BEFORE any held-out number.**
  `PREREG-hub-factor-wells-1.md`. Primary arm `factor_aligned_balanced`; like-for-like `basis_head`
  (shipped ward cut, fitted atoms, same head); `knn_proto` second row. **Estimand: the mean paired
  per-corner difference in `distinct_valid_novel_tuples` over 49 n-way + 60 pair-tiled corners.
  MARGIN = +1.0 novel tuple per corner, and the paired-bootstrap 95 % lower bound must also exceed
  1.0.** Store fit pinned completely (partition, pool, RNG stream, k-means++ seeding, **Lloyd budget 0
  sweeps**, adam 1200/128/0.003, β 0.25, τ 1.0->0.1, ridge λ 1.0, 1380-column head design). Seed 0 only.
- 00:20Z **⭐ A SELECTION CONDITION THAT COSTS NO SCORED BUDGET (charter §A.2).** Running an arm on the
  49 scored corners IS a scored run, so mechanism choices cannot be made there. The condition's own
  n-way builder had capacity for **120** structurally-disjoint Montero slots and emitted only **49** —
  the rest were never emitted, so their rows sit in the value-holdout **train** index. I take slots
  49–72 as `val`: **24 corners x 72 rows = 1,728 rows**, verified all in train and **disjoint from the
  49 scored corners**, and under `--condition val` those rows are removed from the fit pool so the store
  has never held one. Every mechanism choice and the mandatory pair-interaction ablation happen here.
- 00:20Z ⚠ **Process note against myself: I killed a live fit for no reason.** The first batch fit was
  piped through `| tail -4`, which buffers until exit, so it printed nothing for 15 minutes and I read
  silence as a hang and stopped it. Profiling afterwards showed the gram pass is ~42 s and a whole arm
  is ~3 min — it was progressing normally. Relaunched unbuffered to a log file. Cost: ~15 min of CPU.
- 01:35Z **VAL FITS DONE — 6 arms, ~140–205 s each.** Store **524,288 B** on every arm, head
  **1,766,400 B** with the pair block and **74,240 B** without it. ⭐ Note: 74,240 B is *exactly* the
  shipped `gen_cell.addressing_head_bytes_reference` — my no-pair head (58 columns x 64 modes x 5 wells
  x 4 B) is byte-identical to the shipped CLU address book, which is a useful independent check that
  the head I built is the object the programme already calls an addressing head.
  ⚠ The full head is **3.37x the store budget** and that is printed on every row, never netted in.
- 01:35Z **⭐ VAL RESULTS (24 selection corners, 72 requests each, never scored).** Mean per corner:
  | arm | judge accuracy | distinct valid NOVEL tuples / corner |
  |---|---|---|
  | `factor_aligned_balanced` | **0.3536** | **25.46** |
  | `factor_aligned` (unbalanced) | 0.3519 | 25.33 |
  | `basis_head` (ward, the bar) | 0.2801 | 20.17 |
  | `random_dmatched_balanced` (floor) | 0.1626 | 11.71 |
  | `balanced_nopair` | 0.0833 | 6.00 |
  | `basis_head_nopair` | 0.0584 | 4.21 |
  Paired over the 24 corners, margin 1.0, 10,000-sample bootstrap:
  - **`factor_aligned_balanced` − `basis_head` = +5.292**, CI95 **[+2.167, +8.751]**, 17 W / 1 T / 6 L,
    sign-test p = 0.035 ⇒ **clears the declared margin on val.**
  - vs its own random floor: **+13.750**, CI95 [+7.875, +19.293], 20/0/4, p = 0.0015.
  - `basis_head` vs the same floor: +8.458, CI95 [+2.833, +14.042] ⇒ **both partitions beat the floor,
    and factor-alignment beats it by 63 % more.** The floors behave like floors.
  - **`factor_aligned_balanced` − `factor_aligned` = +0.125**, CI95 [−4.083, +4.167], 10/1/13,
    p = 0.68 ⇒ **balancing is a TIE.** Reported as a tie. I keep `factor_aligned_balanced` as the
    pre-registered primary because its reachable rank is 320 — exactly ward's — which is what makes it
    the clean like-for-like, and that reason was declared before this number existed.
- 01:35Z **⛔⭐ THE PAIR-INTERACTION ABLATION — RUN BEFORE ANY SCORED RUN, as the target file requires.**
  Removing the head's pair-interaction block:
  - **`balanced_nopair` − `basis_head_nopair` = +1.792**, CI95 **[+0.625, +3.125]**, 14 W / 7 T / 3 L,
    sign-test **p = 0.013**.
  - The block itself is worth **+19.46** to the factor-aligned arm and **+15.96** to ward.
  **VERDICT, stated plainly:** the gain **does not disappear** — the partition effect survives the
  ablation with the same sign and p = 0.013 — but it **shrinks from +5.292 to +1.792**, i.e. about
  **two thirds of the measured margin is the partition interacting with the head's pair block** and
  about one third is the partition alone. ⚠ **The surviving third does NOT clear my pre-registered
  margin** (its CI lower bound is 0.625 < 1.0). ⇒ I did not merely measure the head, but I cannot claim
  the partition alone clears the bar, and I will not.
  This also reproduces w3-hub-common §4's finding from a different direction: the block is worth ~16–19
  tuples per corner to **every** arm, so the head is the dominant term for everybody.
- 01:35Z **NEXT:** add `landscape_shuffled` (atoms kept, mode rows permuted per well after the head is
  solved — same bytes, same atoms, broken addressing) as the informative §C.2 control beside the zeroed
  one, then fit the full test-condition arm set and spend scored run 2/3.
- 23:12Z ⚠ **CORRECTION — my LOG timestamps drifted ahead of the real clock.** `START_UTC.txt` is
  `2026-09-18T22:20:11Z` (correct). The bullets I stamped `23:18Z`, `23:52Z`, `00:20Z` and `01:35Z` were
  my own estimates, not clock reads; the true wall time at the last of them was **23:10Z**, so they run
  roughly 2.5 h fast. ⛔ I am not rewriting them (charter §E.2: corrections are new bullets). Every
  bullet from here is a real `date -u` read. The ORDER of the bullets is correct and no measurement is
  affected — only the labels are.
- 23:12Z **SCORED-LEDGER ROW WRITTEN, BEFORE THE RUN STARTS.** `composition 2/3`, hub/factor-wells,
  pointing at `PREREG-hub-factor-wells-1.md`. ⚠ I first wrote it with the drifted timestamp and then
  appended a correction row to the append-only ledger rather than editing it. **One scored run, not
  two** — the correction row says so in those words. **Every number from the test condition below
  carries "scored 2/3".**
- 00:33Z **⭐⭐ SCORED RUN 2/3 — THE RESULT, on the held-out corners. It SPLITS by row, and I report both.**
  Primary arm `factor_aligned_balanced`; like-for-like `basis_head`; margin **+1.0 novel tuple per
  corner**, declared in `PREREG-hub-factor-wells-1.md` before any of these numbers existed.
  **Mean distinct valid NOVEL tuples per corner** (row 4 over 49 corners of 72 requests; row 3 over 60
  corners of 36):
  | arm | row 4 | row 3 | row-4 judge acc | row 1 | row 2 |
  |---|---|---|---|---|---|
  | `factor_aligned_balanced` | **41.551** | **15.083** | 0.5771 | 0.6699 | 0.6602 |
  | `basis_head` (the bar) | 29.939 | 13.883 | 0.4158 | 0.5625 | 0.5762 |
  | `factor_aligned` (unbalanced) | 27.000 | 11.667 | 0.3750 | 0.5566 | 0.5527 |
  | `basis_head_k1` (no partition) | 14.020 | 4.200 | 0.1947 | 0.2617 | 0.2383 |
  | `random_dmatched_ward` | 12.755 | 4.217 | 0.1772 | 0.2832 | 0.2715 |
  | `random_dmatched_balanced` | 12.510 | 4.900 | 0.1738 | 0.3086 | 0.2832 |
  | `balanced_nopair` | 9.816 | 4.083 | 0.1363 | 0.1641 | 0.1699 |
  | `basis_head_nopair` | 5.898 | 2.600 | 0.0819 | 0.1387 | 0.1348 |
  | `lookup_argmax` (§C.1/§C.4) | 0.714 | 0.017 | 0.0099 | 0.0098 | 0.0117 |
  | `lookup_convex` (§C.1) | 0.184 | 0.133 | 0.0026 | 0.0020 | 0.0020 |
  | `landscape_shuffled` (§C.2) | **0.000** | **0.000** | 0.0000 | 0.0000 | 0.0000 |
  | `landscape_removed` (§C.2) | **0.000** | **0.000** | 0.0000 | 0.0000 | 0.0000 |
- 00:33Z **⭐ ROW 4 (compositional n-way) — A WIN, beyond the pre-registered margin.**
  `factor_aligned_balanced` − `basis_head` = **+11.612** novel tuples per corner, paired over
  **49 corners**, bootstrap CI95 **[+7.612, +15.408]**, **40 W / 0 T / 9 L**, sign-test
  **p = 9.26e-06**. Both pre-registered criteria hold: the mean delta and the CI lower bound are above
  +1.0. ⇒ **beats basis+head beyond the declared margin.** **(scored 2/3)**
- 00:33Z **⛔ ROW 3 (compositional 2-way) — A TIE. The partition is decoration on this row and I say so.**
  `factor_aligned_balanced` − `basis_head` = **+1.200**, paired over **60 corners**, CI95
  **[−0.850, +3.150]**, **29 W / 10 T / 21 L**, sign-test **p = 0.322**. The CI includes zero; neither
  criterion holds. ⇒ **⛔ A CLU mechanism that ties basis+head is decoration and is reported as such.**
  On row 3 that is the verdict, in those words. **(scored 2/3)**
- 00:33Z **THE FLOORS — all of them, on both target rows. Every one passes, emphatically.**
  - §C.2 landscape-removed (atoms zeroed): **0.000 / 0.000** — total collapse.
  - §C.2 landscape-shuffled (atoms KEPT, ledger byte-identical, only the mode-row order permuted after
    the head was solved — so the reachable span is unchanged and only the ADDRESSING is broken):
    **0.000 / 0.000**. ⭐ This is the informative half and it collapses just as hard: the arm's output is
    not a property of the atoms alone, it is a property of the atoms being addressed correctly.
  - §C.1 matched-bytes lookup on the **same atoms**: `lookup_argmax` **0.714 / 0.017**,
    `lookup_convex` **0.184 / 0.133**, against the arm's 41.551 / 15.083. Arm − lookup_argmax on row 4
    = **+40.837**, **49 W / 0 T / 0 L**. ⇒ the store is **not** decorative and §C.4 does not fire: this
    arm is not a lookup table, by a factor of 58.
  - Partition laundering: arm − `random_dmatched_balanced` (its OWN label vector permuted, size
    histogram held exactly) = **+29.041** (row 4) and **+10.183** (row 3). The two random floors sit at
    12.5 and 12.8 — indistinguishable from each other, which is what a floor should do.
  - ⭐ This also reproduces w3-hub-common §4.4's read-verb ladder on my own partition, in the same
    direction: nearest mode (`lookup_argmax`) < convex combination (`lookup_convex`) ≪ signed linear
    span. Here the gap is far larger than wave 2's 0.4884 / 0.5509 / 0.8264 because the metric is
    distinct novel tuples rather than accuracy.
- 00:33Z **THE ABLATION, re-measured on the scored corners (it was pre-run on `val`, as required).**
  With the pair block removed: row 4 `balanced_nopair` − `basis_head_nopair` = **+3.918**,
  CI95 [+2.673, +5.306], 31/17/1, p = 1.54e-08 ⇒ clears the margin. Row 3 = **+1.483**,
  CI95 [+0.817, +2.183], 30/22/8, p = 4.72e-04 ⇒ does **not** clear it (CI lower bound 0.817 < 1.0).
  ⇒ On row 4 the partition effect survives the referee's ablation and still clears the bar. The block is
  worth ~32 tuples/corner to the arm and ~24 to the control, so w3-hub-common §4's "the head does the
  work" stands — but it is **not all** the head: with the head crippled identically on both sides, the
  factor-aligned partition still wins row 4 beyond the margin.
- 00:33Z ⚠ **VAL and TEST disagree about the balanced variant, and that is itself a finding.** On the 24 val
  corners `factor_aligned_balanced` and `factor_aligned` **tied** (+0.125, p = 0.68). On the 49
  scored corners the balanced variant beats the unbalanced one by **+14.551** (CI95 [+10.27, +18.86],
  42/0/7, p = 3.6e-07). ⛔ I pre-registered the balanced arm on a **structural** ground — its reachable
  rank is 320, exactly ward's — before either number existed, so this is not selection. It is direct
  evidence for w3-hub-common §3: corner sets differ enough that a 24-corner read can miss an effect a
  49-corner read finds at p = 1e-07, which is precisely why the estimand is multi-corner.
- 00:33Z ⚠ **`knn_proto` errored on a bookkeeping line and is being re-run.** Its row-7 block asked for
  `obj["store"]`, which an arm deploying no store does not have (`KeyError: 'store'`). The other 12
  arms completed. Patch: one guarded helper that fingerprints the prototype table instead of the modes;
  diff is 6 lines, entirely inside a branch the span arms never take. ⛔ I am not asserting that from
  the diff alone — I am re-running `basis_head` post-patch into a scratch directory and will compare
  its arrays to the pre-patch ones bit-for-bit before any number here is called final.
- 00:41Z **The patch is proven inert, not argued to be.** Re-ran `basis_head` post-patch into a scratch
  directory and compared **465 arrays** (every corner's accept/valid/tuples, every query row, both
  deletion halves, all 40 survivor corners) against the pre-patch run: **ALL BIT-IDENTICAL**, fit
  fingerprints equal. `knn_proto` then completed. **All 13 arms are on the table.**
- 00:41Z **`knn_proto` — the second row (w3-hub-common §1).** 15.408 novel tuples/corner on row 4, 3.817 on
  row 3. Paired: row 4 **+26.143**, CI95 [+20.408, +32.102], 42/1/6, p = 1.01e-07; row 3 **+11.267**,
  CI95 [+8.517, +14.000], 49/4/7, p = 7.46e-09. ⭐ The arm beats the prototype table on **both** target
  rows beyond the margin. ⚠ Declared: `knn_proto` deploys 475,136 B against the 524,288 B budget and
  carries its `budget_note`; its shipped configuration was selected at the retired 688,128 B budget.
- 00:41Z **⚠⚠ THE SCOREBOARD'S OWN ROW-3 CELL AND MY PRE-REGISTERED ESTIMAND DISAGREE, AND THE ESTIMAND
  WINS.** `gen_scoreboard` scores rows 3/4 as **pooled** distinct novel tuples over all corners at once
  — row 3: 905 (arm) vs 833 (`basis_head`), row 4: 2036 vs 1467 — and, comparing those pooled counts
  against a margin of 1.0, writes *"beats basis+head beyond the declared margin"* on **both** rows.
  ⛔ **I do not quote that for row 3.** The pooled count is unpaired and carries no uncertainty; 905−833
  = 72 over 60 corners is **+1.2 per corner**, the same central estimate my paired statistic gives, and
  my paired CI95 is **[−0.850, +3.150]**, which contains zero. w3-hub-common §3 makes the multi-corner
  paired estimand the verdict and a single pooled cell not a verdict at all. ⇒ **row 3 is a tie and is
  reported as decoration, in those words**, and the scoreboard artifact is emitted unedited with its own
  cell visible so a referee can see exactly this disagreement and check my arithmetic.
- 00:41Z **Rows 1 and 2 (reported, not claimed).** Judge accuracy: arm **0.6699 / 0.6602**, `basis_head`
  0.5625 / 0.5762, `knn_proto` 0.1875 / 0.2031, both random floors ~0.28, `basis_head_k1` 0.26.
  The +0.02 accuracy margin is cleared on both, but these rows are **reported, never claimed** — they
  are `hub/unseen-valid`'s target, not mine.
- 00:41Z **Rows 5 and 6 (reported, and reported as FREE — w3-hub-common §7 reproduced from a new angle).**
  Every arm that deploys any acceptance rule scores the refusal pair at **(1.0000, 1.0000)** — the arm,
  `basis_head`, `knn_proto`, `lookup_argmax`, **and `landscape_shuffled`**. ⭐ That last one is
  the cleanest demonstration I can offer that these rows carry no information about the store:
  `landscape_shuffled` scores **0.000 novel tuples on rows 3 and 4** — it generates nothing at all —
  and still scores a **perfect 1.0000 refusal pair**. My acceptance rule is exactly §7's 8-byte
  value-presence bitmap plus a bracket check, used identically by every arm. ⛔ I quote no row-5 or
  row-6 number as evidence for the store.
  ⭐ The no-model `grid_syntax_rule` scores **0.0000** on both refusal rows, which is AMENDMENT 1's
  predicted collapse (row 1 is all-on-grid, so the syntactic rule refuses the accept leg too) —
  acceptance check §4.1 of the amendment, measured here rather than asserted.
- 00:41Z **Row 7 (reported, with a declared limitation).** Deleting `object_hue = 0.0` from the address
  book: acceptance on its own requests **1.0000 -> 0.0000**, `delta_reject = 1.0`, store modes
  **bit-identical before and after** (`no_retraining_verified`), survivors still compose — 619 distinct
  valid novel tuples for the arm, 618 for `basis_head`, 178 for `knn_proto`, **0 for
  `landscape_shuffled`**, whose `both_hold` is therefore `false`. ⛔ **The limitation, stated
  plainly and not buried:** under a shared-basis read this deletion is an operation on the **interface**,
  not on atoms, so it does **not** satisfy charter §B.4. No atom belongs to one value — every well's
  modes are fitted over the whole pool. ⭐ **That is the leading indicator `hub/refuse-delete` and
  `hub/erasure` asked for: this architecture can gate a value perfectly and cannot erase one, and the
  reason is structural rather than a tuning failure.**
- 00:41Z **⭐ A DISSOCIATION WORTH MORE THAN THE HEADLINE: the best-reconstructing store is NOT the
  best-generating store.** Store reconstruction MSE on the fit pool, beside row-4 novel tuples/corner:
  | partition | recon MSE | relative MSE | row-4 novel/corner |
  |---|---|---|---|
  | `factor_aligned` (unbalanced) | **0.05321** (best) | 0.23986 | 27.000 |
  | `factor_aligned_balanced` | 0.05390 | 0.24297 | **41.551** (best) |
  | `basis_head` (ward) | 0.05677 | 0.25590 | 29.939 |
  | `random_dmatched_balanced` | 0.08842 | 0.39859 | 12.510 |
  | `random_dmatched_ward` | 0.08831 | 0.39809 | 12.755 |
  | `basis_head_k1` (no partition) | 0.09880 | 0.44536 | 14.020 |
  The unbalanced factor partition **reconstructs best of all and generates 35 % worse than the balanced
  one.** And between balanced and ward, a **5.1 %** reconstruction improvement goes with a **38.8 %**
  generation improvement. ⇒ the partition's advantage is **not** that it compresses the features better;
  it is that the head can address it. ⭐ This is the same lesson w3-hub-common §4.5/§4.6 states from the
  byte side — the store is a basis and the addressing is the binding constraint — arriving here from the
  partition side, and it is why "pick the partition with the lowest distortion" would have chosen the
  losing arm.
- 00:41Z **Structural guarantees, asserted on every arm, not inspected afterwards.**
  `head_read`'s signature is `(heads, store, Phi)` — **no feature argument exists on the read at
  all**, and the read is verified bitwise stable while a decoy feature array is perturbed
  (`assert_head_request_only`, 64 requests per arm). Charter §B.3 holds by construction.
  Byte ledger: **524,288 B on every store arm**, `bits_per_read = 30.0`, uniform 64 modes per well,
  per-well bytes [104960, 104960, 104960, 104704, 104704]; the scoreboard's own `assert_byte_ledger`
  passed, including its check that each arm's ledger is **constant across all seven rows**.
- 00:41Z **Replication spoke launched (charter §A.5).** Committed the hub's three modules at
  `de1d692` (nothing at the pin edited), added a clean detached worktree at that commit, and handed an
  independent spoke the winning configuration from the PREREG with instructions to report whatever it
  gets and to tune toward nothing. Awaiting its result; ⛔ unreplicated = not a result, so no number
  above is final until it lands.
- 00:42Z **Reproducibility self-checks, before anything is called final.**
  - The test-condition `partitions.npz` was written **before** I added the `--condition` switch, so I
    re-derived the fit pool with the current code under `--condition test`: **identical, all 131,072
    row indices.** The val pool likewise reproduces its own artifact exactly, is **disjoint from the val
    corners**, and shares only 90,728 of 131,072 rows with the test pool — the two conditions are
    genuinely different fits, not the same store twice.
  - ⚠ One documentation gap, recorded rather than patched over: the **test** `partitions.json` has no
    `condition` field (that field post-dates it), while `run-val/partitions.json` has it. The pool it
    records is the `condition: test` pool, proven above by re-derivation.
  - Metric falsifiers, required by the PREREG: `refuse_everything` scores **0.0** and
    `accept_everything` scores **0.0** on the refusal pair. The metric cannot be won by degenerating.
  - The scoreboard's own `assert_byte_ledger` passed across **14 arms x 7 rows**, including its check
    that each arm's ledger is constant across the seven rows.
- 00:42Z **Return artifacts.** `hub-factor-wells.bundle` (33,085 B, `afd37cf..de1d692`, verified) plus the
  outputs tree below — ⛔ `git status` shows none of the outputs, so here is the directory listing
  w3-hub-common §6.5 asks for:
  ```
  LOG.md · PREREG-hub-factor-wells-1.md · START_UTC.txt · hub-factor-wells.bundle
  run/{corners,partitions,scoreboard,table,ESTIMAND,test_per_corner,test_estimands_partial}.json
  run/{partitions,gram_pair,gram_nopair}.npz · run/arms/<13 arms>/{arm,rows}.{json,npz,pkl}
  run-val/{partitions,val_per_corner,val_estimands}.json · run-val/arms/<6 arms>/
  run-patchcheck/arms/basis_head/rows.npz   (the bit-identity proof)
  run-test.log · run-val.log · replication/
  ```
- 00:46Z **⚠⚠ THE CONFOUND A REFEREE WILL PULL FIRST, and I went and looked: is the win "factor-aligned" or
  just "equal wells"?** My arm has equal wells [410,410,410,409,409]; the shipped ward cut does not
  [332,364,421,411,520]. The target file itself names the variant that separates them — *"a balanced
  variant cutting the dendrogram leaf order gives equal wells at any K. Both keep the correlation
  structure and neither sees a label."* So I built exactly that: `ward_balanced` — the ward linkage's
  **dendrogram leaf order** cut into 5 equal contiguous blocks, same K, same 524,288 B, same rank 320,
  same equal-well histogram as my arm, **and no label anywhere.**
  ⛔ **I ran it on `val` only — it costs no scored budget.** Regenerating the val partitions left all
  seven previously-emitted label vectors **bit-identical** (checked), so nothing already measured moved.
  | arm (val, 24 corners) | novel/corner | judge acc | recon MSE |
  |---|---|---|---|
  | `factor_aligned_balanced` | **25.458** | 0.3536 | 0.05390 |
  | `ward_balanced` (the confound) | 21.500 | 0.2986 | 0.05396 |
  | `basis_head` (ward, unbalanced) | 20.167 | 0.2801 | 0.05677 |
  | `random_dmatched_balanced` (floor) | 11.708 | 0.1626 | 0.08842 |
  - `ward_balanced` − `basis_head` = **+1.333**, CI95 **[−0.958, +3.583]**, p = 0.286 ⇒ balancing
    ward alone is a **tie**.
  - `factor_aligned_balanced` − `ward_balanced` = **+3.958**, CI95 **[+0.250, +7.667]**,
    **14 W / 0 T / 10 L**, sign-test **p = 0.541**.
  **VERDICT, stated against my own interest: the confound is NOT resolved.** The point estimate says
  factor-alignment carries about three times what equal wells carry (+3.96 vs +1.33), and the bootstrap
  CI on that difference excludes zero — but the sign test over 24 corners is 14/10 and nowhere near
  significant. **24 val corners do not have the power to separate the two**, which is the same lesson
  §3 already taught me when val and test disagreed about the balanced variant.
  ⇒ As it stands my row-4 headline is a win over the **shipped ward partition**, and how much of it is
  factor-alignment rather than equal well sizes is **open**. I will not quote it as more than that.
- 00:46Z **DECISION: spend scored run 3/3 on the confound, not on anything else.** Reasoning, recorded
  before the run: (a) it is the single objection that can kill the headline, and it is decidable;
  (b) w3-hub-common §5 says a scored run this wave buys a **verdict**, and this is exactly a verdict;
  (c) I have no other question worth the budget. A second PREREG follows before any number.
  ⚠ **My reading of the budget, stated so the Head can overrule it:** I count the §A.5 replication as
  part of **scored run 2/3** — it re-runs the *same* pre-registered configuration from the same PREREG
  with no new selection — which is what leaves 3/3 free. If the Head counts a replication as its own
  scored run, then run 3/3 is already spent and the `ward_balanced` test leg must be struck from this
  report; everything else stands unchanged.
- 00:55Z **⭐⭐ SCORED RUN 3/3 — THE CONFOUND IS RESOLVED, AND IT INVERTS.** `ward_balanced` on the same
  49 + 60 corners, per `PREREG-hub-factor-wells-2.md`. Partition regeneration first re-emitted all ten
  previously-stored arrays **bit-identically**, so nothing already scored moved.
  | arm (SCORED corners) | row 4 | row 3 | row-4 acc | recon MSE |
  |---|---|---|---|---|
  | `factor_aligned_balanced` | **41.551** | **15.083** | 0.5771 | 0.05390 |
  | `basis_head` (ward, UNequal wells) | 29.939 | 13.883 | 0.4158 | 0.05677 |
  | `ward_balanced` (ward, EQUAL wells) | **15.980** | 8.617 | 0.2219 | 0.05905 |
  | `random_dmatched_balanced` (floor) | 12.510 | 4.900 | 0.1738 | 0.08842 |
  - `factor_aligned_balanced` − `ward_balanced`, **row 4 = +25.571**, CI95 **[+21.041, +30.143]**,
    **47 W / 0 T / 2 L**, **p = 4.36e-12** ⇒ clears the margin. Row 3 = **+6.467**, CI95
    [+4.783, +8.167], 45/9/6, p = 1.83e-08 ⇒ clears it too.
  - `ward_balanced` − `basis_head`, row 4 = **−13.959**, CI95 [−17.429, −10.571], 5 W / 0 T / 44 L,
    p = 7.60e-09. Row 3 = −5.267.
  **VERDICT: equal well sizes are not the cause of the win — for the correlation partition they are a
  LIABILITY.** Cutting the ward dendrogram's leaf order into equal contiguous blocks costs ward **14
  novel tuples per corner** and leaves it barely above its own random floor (+3.469 on row 4, sign test
  p = 0.56). So the row-4 margin is **factor-alignment**, not balance: my arm is equal-welled and wins;
  the correlation arm is equal-welled and loses badly. ⛔ The outcome I wrote down in PREREG 2 as "ties
  ⇒ withdraw the factor-alignment claim" did not occur, and the sentence I pre-wrote for it is not the
  one I get to use.
  ⚠ **It also means run 1's +11.612 over ward does NOT decompose into "alignment + balance".** Balance
  is worth roughly **−14** to a correlation partition and my arm wins **despite** carrying it. Against
  `ward_balanced`, the honest statement of the factor-alignment effect at a **fully matched well-size
  histogram** is **+25.571**.
- 00:55Z ⚠⚠ **AGAINST MYSELF: my `val` condition was a weak proxy, and this is the second time it misled.**
  On val, `ward_balanced` scored **21.500** — *above* `basis_head`'s 20.167. On the scored corners it
  scores **15.980**, far *below* `basis_head`'s 29.939. That is a rank inversion, not a shift. Earlier,
  val called `factor_aligned_balanced` vs `factor_aligned` a tie (p = 0.68) where the scored corners
  separate them at p = 3.6e-07. ⇒ **24 val corners on a fit pool 31 % different from the scored one did
  not predict the scored ranking.** This does **not** touch the pre-registration — I chose the primary
  arm on a structural ground (reachable rank 320, matching ward) before any number existed, and no val
  number changed a single choice — but a hub that had *selected* on that val set would have selected
  wrong. **If I ran this again I would carve the val corners from the same fit pool the scored arms use,
  and I would use far more than 24.** Recorded as an error against my own method.
- 00:56Z **⭐ REPLICATION (charter §A.5) — FULL MATCH on a clean checkout.** An independent spoke made a
  detached worktree at `de1d692`, verified `chlu.__file__` resolved inside **its own** checkout
  (`C:\chlu\worktrees\CHLU-fw-replication\chlu\__init__.py`), created no venv, installed nothing,
  edited no code (`git status` clean) and wrote only into `outputs/hub-factor-wells/replication/`.
  It re-ran the winning configuration from `PREREG-hub-factor-wells-1.md` end to end.
  | | replication | hub | Δ |
  |---|---|---|---|
  | row 4 `factor_aligned_balanced` | 41.551 | 41.551 | **0** |
  | row 4 `basis_head` | 29.939 | 29.939 | **0** |
  | row 3 `factor_aligned_balanced` | 15.083 | 15.083 | **0** |
  | row 3 `basis_head` | 13.883 | 13.883 | **0** |
  | row 4 paired delta | +11.612, CI95 [+7.612, +15.408], 40/0/9, p = 9.264e-06, **beats margin** | identical | **0** |
  | row 3 paired delta | +1.200, CI95 [−0.850, +3.150], 29/10/21, p = 0.322, does not beat | identical | **0** |
  | well sizes balanced / ward | [410,410,410,409,409] / [332,364,421,411,520] | identical | — |
  | ARI balanced-vs-ward | 0.3365615399233504 | identical | — |
  Nothing differed, nothing errored, no run hit its wall-clock cap.
  ⚠ **The spoke's own caveat, which I adopt rather than bury:** every knob of this pipeline is a
  hardcoded constant in `PINS` including the seeds, and the frozen assets are read-only, so exact
  agreement is *what that setup predicts*. It establishes that the pipeline is reproducible and that my
  reported numbers are faithful to what the code produces. ⛔ It does **not** probe seed or data
  sensitivity — and w3-hub-common §5 already forbids a paper number below 3 seeds, of which this wave
  produces only seed 0. **This run buys a verdict, not a paper number.**

---

# FINAL REPORT — hub/factor-wells

**Target:** compositional generation, scoreboard **rows 3 and 4**, with FACTOR-ALIGNED wells.
**Branch:** `hub/factor-wells` @ `de1d692`, off the pin `4125fce` + the `w3-value-holdout` modules.
**Scored:** the `composition` target, **runs 2/3 and 3/3**. Both are reported. Seed 0 only.
**Scope statement, which travels with every number below:** the hold-out is from the **store's fit
pool**, not from the encoder, the decoder or the judge — those three are frozen and were fitted on a
split containing every hue and every shape. "Never trained on" means **the store never held a row
carrying that combination**.

## 1. The answer, in four sentences

**Row 4 (compositional n-way) is a WIN.** Partitioning the store's channels by the factor they carry
beats the fitted basis + linear head on the shipped correlation partition by **+11.612 distinct valid
novel tuples per corner**, paired over **49 held-out corners**, CI95 **[+7.612, +15.408]**, 40 W / 0 T /
9 L, sign-test **p = 9.26e-06** — clearing a **+1.0** margin declared in advance.
**Row 3 (compositional 2-way) is a TIE**, +1.200 with CI95 **[−0.850, +3.150]**, p = 0.322: on that row
**a CLU mechanism that ties basis+head is decoration and is reported as such, in those words.**
The win is **factor-alignment and not well-size balance** — the separating arm, a correlation partition
forced to the same equal well sizes, comes in **13.959 tuples per corner BELOW** the unbalanced ward
control, so balance is a liability for correlation grouping and my arm wins while carrying it.
Everything replicated exactly on a clean checkout, and every floor in charter §C passes.

## 2. The scoreboard — all seven rows, every arm, both controls

Byte ledger **524,288 B on every store arm**, `bits_per_read = 30.0`, constant across all seven rows
(the scoreboard's own `assert_byte_ledger` passed over 14 arms x 7 rows). `knn_proto` deploys 475,136 B
and carries its `budget_note`; `grid_syntax_rule` deploys 0 B and carries its own.

Rows 3 and 4 as **mean distinct valid NOVEL tuples per corner** (49 corners of 72 requests; 60 corners
of 36). Rows 1, 2 as judge accuracy. Rows 5, 6 as the refusal pair's geometric mean. Row 7 as the
deletion score.

| arm | r1 | r2 | **r3** | **r4** | r5 | r6 | r7 |
|---|---|---|---|---|---|---|---|
| **`factor_aligned_balanced`** (mine) | 0.6699 | 0.6602 | **15.083** | **41.551** | 1.0000 | 1.0000 | 1.0000 |
| `basis_head` — **the like-for-like** | 0.5625 | 0.5762 | 13.883 | 29.939 | 1.0000 | 1.0000 | 1.0000 |
| `factor_aligned` (unbalanced) | 0.5566 | 0.5527 | 11.667 | 27.000 | 1.0000 | 1.0000 | 1.0000 |
| `ward_balanced` (the confound arm) | 0.3096 | 0.3066 | 8.617 | 15.980 | 1.0000 | 1.0000 | 1.0000 |
| `knn_proto` — **the second row** | 0.1875 | 0.2031 | 3.817 | 15.408 | 1.0000 | 1.0000 | 1.0000 |
| `basis_head_k1` (no partition at all) | 0.2617 | 0.2383 | 4.200 | 14.020 | 1.0000 | 1.0000 | 1.0000 |
| `random_dmatched_ward` (floor) | 0.2832 | 0.2715 | 4.217 | 12.755 | 1.0000 | 1.0000 | 1.0000 |
| `random_dmatched_balanced` (floor) | 0.3086 | 0.2832 | 4.900 | 12.510 | 1.0000 | 1.0000 | 1.0000 |
| `balanced_nopair` (ablation) | 0.1641 | 0.1699 | 4.083 | 9.816 | 1.0000 | 1.0000 | 1.0000 |
| `basis_head_nopair` (ablation) | 0.1387 | 0.1348 | 2.600 | 5.898 | 1.0000 | 1.0000 | 1.0000 |
| `lookup_argmax` (§C.1/§C.4) | 0.0098 | 0.0117 | 0.017 | 0.714 | 1.0000 | 1.0000 | 1.0000 |
| `lookup_convex` (§C.1) | 0.0020 | 0.0020 | 0.133 | 0.184 | 1.0000 | 1.0000 | 1.0000 |
| `landscape_shuffled` (§C.2) | 0.0000 | 0.0000 | **0.000** | **0.000** | 1.0000 | 1.0000 | 0.0000 |
| `landscape_removed` (§C.2) | 0.0000 | 0.0000 | **0.000** | **0.000** | 1.0000 | 1.0000 | 0.0000 |
| `grid_syntax_rule` (no-model rule) | 0.0000 | 0.0000 | 0.000 | 0.000 | **0.0000** | **0.0000** | — |

⛔ **Rows 1, 2, 5, 6 and 7 are reported and claimed as nothing.** Rows 1–2 are `hub/unseen-valid`'s
target. Rows 5–6 are free: see §5.

## 3. The pre-registered runs — ALL of them

### Scored run 2/3 — `PREREG-hub-factor-wells-1.md`

`factor_aligned_balanced` vs `basis_head`, margin **+1.0 novel tuple per corner** with the bootstrap CI
lower bound also required above it, declared before any held-out number existed.

| row | corners | mean paired Δ | bootstrap CI95 | W/T/L | sign-test p | verdict |
|---|---|---|---|---|---|---|
| **4 comp n-way** | 49 | **+11.612** | **[+7.612, +15.408]** | 40/0/9 | 9.26e-06 | ⭐ **beats basis+head beyond the declared margin** |
| **3 comp 2-way** | 60 | +1.200 | [−0.850, +3.150] | 29/10/21 | 0.322 | ⛔ **a CLU mechanism that ties basis+head is decoration and is reported as such** |

Against the second row, `knn_proto`: row 4 **+26.143** (CI95 [+20.408, +32.102], 42/1/6, p = 1.01e-07);
row 3 **+11.267** (CI95 [+8.517, +14.000], 49/4/7, p = 7.46e-09). Both clear the margin.

### Scored run 3/3 — `PREREG-hub-factor-wells-2.md` (the confound leg)

Is the win factor-alignment, or merely equal well sizes? `ward_balanced` holds the equal-well histogram
**exactly** and sees no label.

| comparison | row | mean paired Δ | CI95 | W/T/L | p |
|---|---|---|---|---|---|
| mine − `ward_balanced` | 4 | **+25.571** | [+21.041, +30.143] | 47/0/2 | 4.36e-12 |
| mine − `ward_balanced` | 3 | **+6.467** | [+4.783, +8.167] | 45/9/6 | 1.83e-08 |
| `ward_balanced` − `basis_head` | 4 | **−13.959** | [−17.429, −10.571] | 5/0/44 | 7.60e-09 |

⇒ **Equal wells do not explain the win; for a correlation partition they cost 14 tuples per corner.**
At a fully matched well-size histogram the factor-alignment effect on row 4 is **+25.571**. The outcome
PREREG 2 pre-wrote as "ties ⇒ withdraw the factor-alignment claim" did not occur.

### ⛔ The ablation the target file demanded BEFORE any scored run — run on `val`, then re-measured

Removing the head's pair-interaction block:

| where | row | `balanced_nopair` − `basis_head_nopair` | CI95 | p | clears +1.0? |
|---|---|---|---|---|---|
| **val (pre-registration)** | n-way | **+1.792** | [+0.625, +3.125] | 0.013 | **no** |
| scored corners | 4 | +3.918 | [+2.673, +5.306] | 1.54e-08 | yes |
| scored corners | 3 | +1.483 | [+0.817, +2.183] | 4.72e-04 | no |

**The gain does not disappear** — with the head crippled identically on both sides the factor-aligned
partition still wins, at p = 0.013 on val and p = 1.5e-08 on the scored n-way corners. **But it
shrinks**: on val from +5.292 to +1.792, so roughly **two thirds of the measured margin is the
partition interacting with the head's pair block** and one third is the partition alone. The block is
worth ~16–19 tuples per corner to *every* arm. ⇒ I did not merely measure the head; I also cannot claim
the partition alone clears the bar on val, and I do not.

## 4. The floors — charter §C, one line each

- **§C.1 matched-bytes lookup.** `lookup_argmax` (one atom per well) and `lookup_convex` (the simplex
  mixture) over the **same atoms at the same 524,288 B**: **0.714 / 0.184** on row 4 against the arm's
  **41.551**. Paired: **+40.837**, **49 W / 0 T / 0 L**. ⭐ **PASS — the store is not decorative.**
- **§C.2 landscape-removed.** Atoms zeroed: **0.000 / 0.000**. ⭐ **PASS — total collapse.**
- **§C.2, the informative half.** `landscape_shuffled` keeps the atoms and the byte-identical ledger and
  permutes only the mode-row order *after* the head was solved, so the reachable span is unchanged and
  only the addressing is broken: **0.000 / 0.000**. ⭐ **PASS, and it is the stronger statement** — the
  arm's output is not a property of the atoms, it is a property of the atoms being addressed correctly.
- **Partition laundering.** Against its own label vector permuted with the size histogram held exactly:
  **+29.041** (row 4) and **+10.183** (row 3). Both random floors sit at 12.5 / 12.8, indistinguishable
  from each other. ⭐ **PASS — every partition number here is a margin over that floor.**
- **§C.4 lookup-table check.** The arm beats `lookup_argmax` by a factor of **58** on row 4. ⭐ **Does
  not fire — this arm is not a lookup table.**
- **§B.3 request-only / no bypass.** `head_read`'s signature is `(heads, store, Phi)` — **no feature
  argument exists on the read at all** — and the read is verified bitwise stable under an arbitrary
  feature perturbation, on every arm. ⭐ **Structural, not inspected.**
- **§B.5 bytes.** 524,288 B, `bits_per_read = 30.0`, constant across all seven rows for every arm; the
  two unmatched arms carry `budget_note`s. ⭐ **PASS.**
- **§B.6 label path.** Labels enter **once**: `I(bin(channel_d); factor_f)` on the store's own fit pool,
  the same quantity and rows the address book is fitted from. `assert_relevance_is_train_only` reports
  **0 rows shared** with the hold-out. No held-out row, no cell, no arm score, no judged outcome, no
  learned rotation. Emitted as `run/partitions.json -> label_path_audit`. ⭐ **PASS.**
- **Metric falsifiers.** `refuse_everything` scores **0.0**; `accept_everything` scores **0.0**.

## 5. Rows 5–7 — reported as free, and one thing `hub/refuse-delete` needs

**Rows 5 and 6 are free and I quote no number from them as evidence for the store.** Every arm with an
acceptance rule scores the pair at **(1.0000, 1.0000)** — including **`landscape_shuffled`, which
produces 0.000 novel tuples on rows 3 and 4 and still scores a perfect refusal pair.** My acceptance
rule *is* w3-hub-common §7's 8-byte value-presence bitmap plus a bracket check, used identically by
every arm so the rows compare arms at one threshold. The no-model `grid_syntax_rule` scores **0.0000**,
which is AMENDMENT 1's predicted collapse, measured.

**Row 7, with its limitation in the open.** Deleting `object_hue = 0.0` from the address book:
acceptance on its own requests **1.0000 -> 0.0000**, `delta_reject = 1.0`, **store modes bit-identical
before and after** (no retraining), survivors still compose (619 novel tuples for my arm, 618 for
`basis_head`, 178 for `knn_proto`, **0** for `landscape_shuffled`, whose `both_hold` is `false`).
⛔ **This is an operation on the interface, not on atoms, so it does NOT satisfy charter §B.4.** Under a
shared-basis read no atom belongs to one value — every well's modes are fitted over the whole pool.
⭐ **The finding `hub/refuse-delete` and `hub/erasure` asked for: this architecture gates a value
perfectly and cannot erase one, and the reason is structural rather than a tuning failure.**

## 6. What the mechanism actually is, and the result I did not expect

**The partition.** `R[d,f] = I(bin(channel_d); factor_f)` on the fit pool -> each live channel to its
argmax factor -> the declared `min_relevance_mass` merge from 6 factors to the pinned K = 5 -> rebalanced
to equal wells by relevance **margin**. Wells: floor_hue · wall_hue+scale · object_hue · shape ·
orientation, sizes **[410, 410, 410, 409, 409]**, block-diagonality lift **2.237** over its own floor's
1.004, ARI vs ward **0.337**.

**The read, held fixed across every arm:** the signed linear span over the modes with a
request-conditioned ridge head (bias + 57 main-effect one-hots + 1322 pair-interaction one-hots = 1380
columns, λ = 1.0, closed-form Sylvester solve). No simplex imposed (§4.5). ⚠ The head is **1,766,400 B —
3.37x the store budget** — and that is printed on every row and never netted in. Its no-pair variant is
**74,240 B, byte-identical to the shipped `addressing_head_bytes_reference`.**

**⭐ The result I did not expect, and it may be worth more than the headline: the best-reconstructing
store is not the best-generating store.**

| partition | recon MSE | row-4 novel/corner |
|---|---|---|
| `factor_aligned` (unbalanced) | **0.05321 (best)** | 27.000 |
| `factor_aligned_balanced` | 0.05390 | **41.551 (best)** |
| `basis_head` (ward) | 0.05677 | 29.939 |
| `ward_balanced` | 0.05905 | 15.980 |
| `random_dmatched_balanced` | 0.08842 | 12.510 |

A **5.1 %** reconstruction improvement (balanced over ward) goes with a **38.8 %** generation
improvement, and the partition with the **lowest distortion of all** generates 35 % worse than the one I
pre-registered. ⇒ **selecting a partition by reconstruction would have chosen a losing arm.** This is
w3-hub-common §4.5/§4.6's "the store is a basis, the addressing is the binding constraint" arriving from
the partition side.

## 7. ⚠ Limitations — the things that could still kill this

1. **Seed 0 only.** w3-hub-common §5: no number reaches the paper below 3 seeds. **This buys a verdict,
   not a paper number.** The replication re-ran fully-pinned deterministic code against the same frozen
   assets, so — as the spoke itself said — exact agreement is what that setup predicts; it proves the
   pipeline reproducible and the numbers faithful, and it **does not probe seed or data sensitivity.**
2. **Row 3 is a tie.** Half my target row set is decoration. I do not know why the effect is confined to
   the n-way corners, and I did not spend budget guessing.
3. **My `val` condition was a weak proxy and misled twice** — it called the balanced-vs-unbalanced
   contrast a tie (p = 0.68) where the scored corners give p = 3.6e-07, and it ranked `ward_balanced`
   *above* `basis_head` where the scored corners put it far below. No val number changed any choice
   (the primary arm was chosen on the structural ground of matching ward's reachable rank 320, before
   any number existed), but a hub that had genuinely selected on it would have selected wrong. **If I
   ran this again I would carve the val corners from the same fit pool the scored arms use, and use far
   more than 24.** Recorded as an error against my own method.
4. **The head is 3.37x the store.** Every arm pays it identically so it cannot favour a partition, but
   the absolute regime is one where §4.6's "the addressing head is the binding constraint" is plainly
   true, and the ablation says two thirds of my val margin lives in the partition x head interaction.
5. **The encoder, not the idea, limits factor alignment here.** Only **12** of 1984 live channels are
   top-ranked by `shape` and **4** by `scale`; best single-channel MI is 0.250 b and 0.317 b against
   2.066 b for `floor_hue`. Three of five wells go to hue. A factor-aligned partition of a *better
   disentangled* encoder is untested and is the obvious next step.
6. **One declared deviation from the shipped fitter.** `train_store_traced_tau` traces `τ` instead of
   letting `filter_jit` treat it as static (1.33 s/step -> 0.035 s/step, measured; the shipped path
   recompiles on all 1200 steps). Max |loss difference| 0.597 on ~860 over 40 steps, recon MSE
   1.68627071 vs 1.68607640. **Not bit-identical, and not claimed to be.** Identical for every arm.
7. **`knn_proto` was selected at the retired 688,128 B budget** (~31 % advantage) and deploys 475,136 B
   here — declared, and in the conservative direction.

## 8. Cost line

| | |
|---|---|
| wall clock | **2 h 36 min** (`START_UTC` 2026-09-18T22:20:11Z), against a ≤10 h budget |
| compute | **local CPU only, 4 threads.** ⛔ **No GPU requested and none needed** — the frozen feature cache is `scope: "full"` (all 480,000 rows encoded), so nothing was encoded, the cache was never extended, and the re-encode gate never bound. The three GPU justification lines were never written because there was no GPU case. |
| store + head fits | 15 arms x 2 conditions, ~907 s total on the scored condition |
| judged renders | ~14 arms x ~9,900 requests through the frozen decoder + hash-asserted classifier |
| scored runs | **composition 2/3 and 3/3.** Both reported. Neither reported alone. |
| sub-agents | one replication spoke (§A.5), 14 min, **confirmed finished** |
| peak memory | well inside cap; features read via `mmap`, gathers chunked at 16,384 rows |

## 9. Return artifacts

`hub-factor-wells.bundle` (`afd37cf..de1d692`, verified) — ⛔ carries **code only**. `git status` shows
none of the outputs below, so this is the directory listing w3-hub-common §6.5 requires:

```
LOG.md · PREREG-hub-factor-wells-1.md · PREREG-hub-factor-wells-2.md · START_UTC.txt
hub-factor-wells.bundle
run/ESTIMAND.json          <- the pre-registered estimand, every arm, both rows
run/scoreboard.json        <- gen_scoreboard's own artifact, emitted UNEDITED
run/table.json · run/confound_estimand.json · run/test_per_corner.json
run/corners.json · run/partitions.json  (+ label_path_audit) · run/partitions.npz
run/arms/<14 arms>/{arm.json, arm.pkl, rows.npz}
run-val/{partitions,val_per_corner,val_estimands}.json · run-val/arms/<7 arms>/
run-patchcheck/arms/basis_head/rows.npz   <- the 465-array bit-identity proof
replication/              <- the independent spoke's full re-run
run-test.log · run-val.log
```

⚠ **One disagreement left visible on purpose.** `gen_scoreboard` scores rows 3–4 as *pooled* counts over
all corners and therefore writes "beats basis+head beyond the declared margin" on **row 3** as well
(905 vs 833). ⛔ I do not quote that. A pooled count is unpaired and carries no uncertainty; the same
data paired gives +1.200 with CI95 [−0.850, +3.150]. w3-hub-common §3 makes the multi-corner paired
estimand the verdict, so **row 3 is a tie**. The scoreboard artifact ships unedited so a referee can see
exactly this and check my arithmetic.

## 10. One line for the campaign

⭐ **Factor-aligned wells win compositional n-way generation outright (+11.612 novel tuples/corner over
basis+head, 49 corners, p = 9.3e-06, replicated) and tie on 2-way, where the partition is decoration;
the win is alignment and not balance (+25.571 at a matched well-size histogram); and the partition with
the lowest reconstruction error is not the one that generates best.**

---

## Post-report addenda (⛔ the report above is not rewritten; these are new bullets)

- 00:57Z **Every headline number in the report was re-verified against the artifacts on disk**, not
  transcribed from memory: 24 checks against `run/ESTIMAND.json` and `run-val/val_estimands.json` —
  all means, both target rows' deltas, both CI endpoints, the W/T/L counts, both margin booleans, the
  confound leg, both ablation legs, the §C floors and the two val figures. **All matched.**
- 00:57Z **Final commit `3ea14ac`** (the confound arm, `landscape_shuffled`, the `knn_proto` row-7
  guard, the `--condition` switch). Bundle rebuilt as `afd37cf..3ea14ac` and **verified okay**.
  Worktree clean.
- 00:57Z **Sub-agents: confirmed none running** (w3-hub-common §6.3). The single replication spoke completed
  and reported; its temporary worktree `CHLU-fw-replication` has been removed. No `exp_gen_factor_wells`
  process is alive.
- 00:57Z **⛔ Standing caveat repeated at the bottom of the file so it cannot be read past:** this hub's
  numbers are **seed 0 only** and therefore **buy a verdict, not a paper number** (w3-hub-common §5).
- 00:57Z **HUB COMPLETE.** Wall clock 2 h 36 min of a ≤10 h budget. Scored budget: `composition` 2/3 and
  3/3 spent, both reported, **the target's budget is now exhausted**. No GPU requested. Nothing at the
  pin edited; three additive modules on the branch.
