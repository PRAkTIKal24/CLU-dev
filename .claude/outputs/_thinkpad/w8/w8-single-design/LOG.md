# w8 hub/single-design — LOG

TARGET: one object that composes AND forgets.
Base `base/iter7 = ec2ef7c`. Worktree `/c/chlu/worktrees/CHLU-w8-single-design`.
Budget: single-design 3 scored runs + generation-table run 3 (the last one).

---

- 2026-09-20T16:36Z — START_UTC.txt written (no live one existed). Boot read in order: autonomous-hub-charter.md, w8-hub-common.md, w8-hub-single-design.md. Nothing else read.
- 2026-09-20T16:36Z — Env verified: `chlu.__file__` = `C:\chlu\worktrees\CHLU-w8-single-design\chlu\__init__.py` (inside worktree, good). jax 0.9.0, numpy 2.4.1, equinox 0.13.4. HEAD = ec2ef7c.
- 2026-09-20T16:36Z — GATE CHECK: `/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json` ABSENT. Scoring is gated. DEV work starts now per task file.
- 2026-09-20T16:40Z — Coordination read of peer spoke `/c/chlu/outputs/w8-downdate/LOG.md` (I am gated on its DOWNDATE-READY.json, so its interface is operationally necessary; this is not a wave report). It states the object it is building:
  - Incumbent to reproduce: `exp_w5_compose_field.PCASpanStore` (`chlu/experiments/exp_w5_compose_field.py:218`), built by `build_pca_content` (:468). Per well k: pool features on well k's channels -> mean `mu`, center, `np.linalg.svd`, keep top `A-1` right singular vectors; atom block = `[mu; Vt[:A-1]]` zero-padded to A rows, `A = 524288 // (4*D)`. Bytes pinned 524,288 B.
  - ⭐ THE DESIGN POINT: the SVD route holds no sufficient statistic, so it CANNOT be downdated. The downdatable route is the **covariance** route: `S_k = sum x x^T`, `t_k = sum x`, `n_k`, eigendecomposed. Same leading subspace, but now per-item deletion is exact subtraction of `x x^T` and `x` with NO refit and NO pass over data.
  - ⚠ Direct consequence for my rows: the honest byte ledger must carry `d_k^2` per well BESIDE the 524,288 B store. My task file pre-commits me to PRINT that rather than net it. This is the single most likely way the comparison comes out unmatched, and I must handle it head-on.
- 2026-09-20T16:40Z — Spawned 3 read-only Explore sub-agents (charter §A permits sub-agents) to map, in parallel: (a) `pca_tensor` incumbent internals + byte ledger + any existing delete path; (b) `exp_gen_cell.py` generation-table harness, rows 3/4, learned baselines at matched parameters, judge, corners, seeds; (c) exact-deletion machinery: conjunction/coupling hole, field store, eviction, `assert_no_bypass`, refusal rule. No scored run touched. Ledger untouched (no scored run pre-registered yet).
- 2026-09-20T16:45Z — Surveyed the shared assets my rows depend on. Env is sourced correctly (`CHLU_ROOT` = my worktree; `PYTHONPATH` puts CHLU_ROOT first; 3 threads).
  - ⚠ `env_w8.sh` exports `CHLU_GEN_BRIDGE_ROOT` etc. pointing at **generation-1** caches. w8-common §2 says **generation 2 only**. The gen-2 cache root is `C:\chlu\outputs\spoke-a-rivals\cache_generation2\shapes3d_recon` (3 seeds, 480,000 rows, float16, CPU, HIGHEST). ⛔ Every scored run of mine uses gen-2; DEV may use gen-1 but I will not mix them in one table.
  - Rival callables: `chlu.eval.rivals.w4_api.load_rival(seed, name)` -> `(features (N,D) float32, out_of_range (N,) bool)`. Present for all 3 seeds: mlp, mlp_180k, lstm, transformer, cont_mlp, table, poe_ebm, lace_repaired, plus `*_dropout` variants.
  - Param match: target **118,784** (the prototype table's own count, 475,136 B at float32), tolerance 0.1. Actuals: mlp 117,878 · lstm 119,125 · transformer 119,858 · cont_mlp 119,525 · table 118,784. So "matched parameters" is a declared, mechanically checkable thing.
  - ⚠ `rescore_generation2.verdict`: "AT LEAST ONE ARM MOVES BY MORE THAN ITS SEED SPREAD" — `lace/ld_corrected` at 0.0177 vs 0.0151 spread, on 1 of 2 seeds compared; seed 1 NOT RUN. The three deterministic arms (clu, knn_proto, store_removed) move by exactly 0.0000. I must carry this caveat into any table quoting lace.
  - ⚠ Population caveat that binds my rows 3-4: rivals are fitted on the `montero_combination` pool (32,256 rows) which contains every factor VALUE — only COMBINATIONS are held out. `row_eligibility.json` says rows 3-4 (pair and n-way corners, FULL six-factor requests) `can_enter: true` for every learner — that is the row these rivals are built for. Good: my RESULT rows 3-4 are like-for-like.
- 2026-09-20T16:47Z — ⭐ FOUND: the "retrained-without" comparator I am required to print already has machinery. `/c/chlu/outputs/spoke-a-rivals/rivals/seed_*/deletion_comparator.json` carries retrain-from-scratch GOLD STANDARDS per learner with their cost lines. For the existing instance (deleting `object_hue=0.5`, grid index 5):
  - rows_dropped 3,157 · rows_refitted_by_gold_standard 29,099 · rows_refitted_by_edit 0.
  - wall clock: gold-standard refit 3.08s reported at the comparator level; per-learner retrain-from-scratch e.g. cont_mlp 75.57s (20 epochs, 2,260 steps, batch 256, lr 1e-3), lstm similar. So a retrain-without costs ~75s/learner/seed — CHEAP. There is no excuse not to run both comparators.
  - ⛔ BUT: those scores are `forget`/`retain` judge aggregates, NOT rows 3-4 composition corners, and the deleted value is `object_hue=0.5`, not red. So the machinery transfers; the numbers do not. I must re-run retrain-without for the RED value and score it on the rows 3-4 corners.
  - ⚠ Its declared SCOPE, which I inherit verbatim: the value is held out of the ARM'S FIT POOL only — not from the encoder, decoder or judge, which are frozen and saw every hue. "Never trained on" means the arm never fitted a row carrying that value. That is what makes the comparison scorable at all, and it is a caption I must print.
- 2026-09-20T16:52Z — Pinned the factor grid from `C:\chlu\data\gen\shapes3d\factor_index.npz` (dataset shapes3d, condition `recomb2element`, variant `leave1out`, n_total 480,000; n_train 479,928; n_test 72):
  - factors/cardinalities: floor_hue 10 · wall_hue 10 · object_hue 10 · scale 8 · shape 4 · orientation 15.
  - `object_hue` values = 0.0,0.1,...,0.9 (10 steps). ⛔ There is NO colour-name mapping anywhere in the repo — "red"/"green"/"blue" in my task file are shorthand for object_hue values and I must DECLARE the mapping in the PREREG rather than assume one.
  - Declaration I will pre-register: on the HSV hue ring, red = hue 0.0, green = hue 1/3, blue = hue 2/3. Nearest grid points: **red = object_hue 0.0**, **green = object_hue 0.3**, **blue = object_hue 0.7** (2/3 = 0.667 is nearest 0.7). This is a pre-registered convention, not a measurement, and the row is unaffected by which two retained hues are called green and blue — what matters is that the deleted value is one hue and the scored corners carry other hues.
  - ⚠ Note the existing declared value-holdout is `object_hue = 0.5`, a DIFFERENT value, chosen for its own rationale. I am deleting 0.0, per my task file. Rows 3-4 live on the Montero COMBINATION hold-out where every value is present in the fit pool, so deleting red is a deletion I impose, not one the split already made.
- 2026-09-20T16:53Z — The retrain-without machinery is parameterised, not hard-coded: `chlu/eval/rivals/w4_deletion.py:150 run_deletion_comparator(ctx, factor=DELETED_FACTOR, value=DELETED_VALUE, families=...)`. Module constants `DELETED_FACTOR="object_hue"`, `DELETED_VALUE=0.5` are defaults I can override with red. It already implements `retrain` (gold standard, refit from scratch on the reduced pool) vs `edit` (O(D) mean-substitution) and a `cost_line`. ⭐ This is the honest comparator my task demands, and it exists at the pin — I do not have to build it, only point it at red and score its output on the rows 3-4 corners instead of forget/retain judge aggregates.
- 2026-09-20T16:58Z — Read the incumbent's own source. ⚠ CORRECTION to a reading I nearly made: `PCASpanStore`'s class docstring calls itself "a DECLARED NON-ARM twice over", which looks at first like `pca_tensor` is disqualified. It is not. `PCASpanStore` is the CONTENT CONTAINER; arm-ness is a property of the READ. The non-arm language applies to the `oracle_content_*` / `oracle_span_field` legs, which are handed the true feature and (for `source="corners"`) fit content on the scored rows. `exp_w5_compose_field.py:78` labels `pca_tensor` "**THE ARM**: principal content + the product address", and `FIELD_ARM_CLASS["pca_tensor"] = "ARM"` (:112).
- 2026-09-20T16:58Z — PINS of the incumbent object, taken from `build_pca_content` (`exp_w5_compose_field.py:468`) and confirmed arithmetically:
  - `D = 2048`, `A = n_atoms = store_bytes // (4*D) = 524288 // 8192 = **64**`, `K = 5` wells, partition = inherited `factor` label-MI wells (`factor_aligned_balanced`), placement held fixed because placement is declared exhausted.
  - Content = per well k, the A=64 leading principal directions of the FIT POOL's features on well k's channels, first atom = the pool mean. Exactly `A x D` float32 = **524,288 B**, ledger-asserted (`matches_budget`, raises if the pin moved).
  - Three admissibility properties it declares: (1) ⭐ no label enters the content — the partition uses labels once at fit time, the decomposition itself is unsupervised; (2) ⭐ no held-out row enters it — fitted from `partitions.npz['pool']`, the same pool every other store arm uses; (3) ⚠ **it gives up exact per-item deletion** — "a principal direction is not a written item, so charter B.4 does not hold of this content the way it holds of the field store's."
  - ⭐⭐ Property (3) IS this hub's open problem, written by the incumbent's own author as a declared cost. The single design's claim is precisely that the covariance route restores B.4 without moving (1) or (2): `S_k, t_k, n_k` are sufficient statistics, so deleting an item is an exact subtract of `x x^T`, `x`, `1` with no refit and no pass over data, and the leading eigenvectors of the downdated `S_k` are the same content.
  - Its charter-C family is already built and lives beside it on the same 524,288 B block: `pca_landscape_removed` (atoms zeroed, C.2), `pca_landscape_shuffled` (values permuted, C.2), `pca_lookup_argmax` / `pca_lookup_convex` (C.1/C.4 matched-bytes floors), `pca_random_refit`, `pca_span_rotated` (predicted no-op), `pca_tensor_sum` (ablation: address summed not multiplied). ⭐ My single design inherits this whole control family for free — it is the same content block at the same bytes, so every control transfers.
- 2026-09-20T17:00Z — ⚠ THE LEDGER RISK, quantified early because my task file pre-commits me to print it rather than net it. Exact downdate needs `S_k` (d_k x d_k) per well. `sum_k d_k = D = 2048` over K=5 wells; if wells were balanced d_k ~ 410, so `sum_k d_k^2 ~ 5 * 410^2 = 840,500` floats = **~3.36 MB at float32, ~6.4x the 524,288 B store**. Honest ledger (store + covariance) would then be ~3.89 MB. I must measure the ACTUAL well sizes rather than assume balance, and print the honest total beside the pinned store. ⛔ I will not net it, and if it exceeds budget the comparison is declared not-matched on that row and said so plainly.
- 2026-09-20T17:06Z — ⭐⭐ DECISIVE: "delete red / score green and blue" is NOT my coinage — it is the repo's own pinned deletion, and the harness for my headline row already exists.
  - `exp_gen_factor_wells.py:1348`: `DELETED_FACTOR, DELETED_VALUE_INDEX = 2, 0` -> **object_hue = 0.0**, with the comment "it is `pair-000`'s hue, so the survivor set is exactly the sub-corners of the other two pair cells." So RED = object_hue 0.0 = pair-000's hue; GREEN and BLUE = the hues of pair-001 and pair-002. ⭐ My earlier HSV-convention guess (red 0.0) is confirmed by the code; I withdraw the green=0.3/blue=0.7 guess and will take green/blue from `build_pair_cells` (`chlu/data/gen_value_holdout.py:536`, hues at even stride over the retained hue ring, shapes cycling, no two cells sharing a hue or a shape).
  - `exp_gen_delete_row7.py` IS the post-deletion-composition instrument: `_per_corner` (:1853) reduces a corner to `distinct_valid_novel_tuples`; `_paired` (:1873) is the paired per-corner difference after-minus-before with a 2,000-draw bootstrap CI; **BAR 2** (:2193) is `abs(mean_delta) <= MARGIN_NOVELTY = 1.0` on the **89 survivor corners**; and `S["BAR2_breakout_green_and_blue_on_every_shape"]` (:2121) is literally the green-and-blue breakout my task names. ⛔ I do not build this; I add an arm to it.
  - **BAR 4** (:2206): the hole's paired delta must be exactly 0.0 with features bit-identical. That is the DISPLAY conjunction-hole row, already mechanised.
- 2026-09-20T17:06Z — The failure mode my task tells me to watch for is already measured and named. `evict_size_control` (`exp_gen_delete_row7.py:1585`, declared POST-HOC, its own stage) evicts the SAME NUMBER of items (~6 of 64) drawn at random from items NOT holding red, 8 draws, and scores the identical paired delta: **−2.7584**. Against the real eviction deltas **−2.0899 / −0.9101 / −6.7416** (`exp_w6_retention.py:150`), the real delta falls INSIDE the random range, and the code's own verdict string (:1708) is: *"THE COST IS THE MECHANISM, NOT THE TARGET ... An item store loses composition when it loses atoms, whichever atoms they are."* ⭐ This is exactly the w5 collapse my task file says would prove the hub wrong, and it tells me what the single design has to avoid: **it must not lose span when it forgets.** The covariance route does not evict atoms — it downdates a sufficient statistic and re-derives the same rank-64 subspace — so there is a real mechanism by which it can avoid this. That is the hypothesis, not yet a result.
- 2026-09-20T17:06Z — Exactness instruments I will reuse verbatim rather than reinvent (charter A.4 keeps me off the frozen harness; these are not in it):
  - `gen_field.assert_changed_bytes_within` (`chlu/core/gen_field.py:354`, "THE ACCEPTANCE ASSERTION") on `byte_diff` (:325) — returns `bytes_changed_outside_permitted`, which must be 0; refuses to compare images of different sizes.
  - `gen_field.assert_no_refit` (:388) — byte-identity of the fitted segments.
  - Their negative control (`exp_gen_delete_row7.py:993`): nudge one row of `term_mu` by 1e-3 and `assert_no_refit` must FIRE. ⭐ I will carry a negative control of the same shape for the covariance downdate, or the exactness claim is not instrumented.
- 2026-09-20T17:06Z — Two ledger idioms exist and they differ; I must pick one and say which. `gen_field.field_byte_ledger` (:839) reports `deployed_bytes = atoms only = 524,288 B` with `interface_bytes` printed BESIDE it, never netted. `gen_landscape.landscape_byte_ledger` (:414) SUMS store + coupling and bounds the total at 2x budget (`d3_rule`). ⭐ For the covariance the honest choice is the SECOND idiom — a sum with a declared multiple — because the covariance is what buys the deletion property and hiding it beside the ledger would be exactly the netting my task file forbids.
  - Measured well sizes at seed 0 are `[303, 305, 573, 501, 366]` (sum 2048; recorded `tests/test_gen_dynamical_read.py:152`), NOT balanced. So `sum_k d_k^2 = 898,120` and the symmetric-triangle count is `sum_k d_k(d_k+1)/2 = 450,084` floats. Covariance at float32 = **1,800,336 B** (triangle) or 3,592,480 B (full). Honest total (store + triangle covariance) = **2,324,624 B = 4.43x the 524,288 B budget**. ⛔ That is over budget and I will print it as such, not net it.
- 2026-09-20T17:06Z — ⚠ Noted but NOT fixed: `exp_gen_field_store.sweep` (:930) references `store_seed` while its parameter is `seed` (:884) — `--stage sweep` would raise NameError. I do not need that stage; touching it is out of my dial and I leave it alone. Recording it so it is not rediscovered.
- 2026-09-20T17:14Z — ⭐⭐ DEV RESULT (arithmetic, not a measurement): **a matched-bytes single design exists, and the partition granularity is the dial that buys it.**
  Setup: exact downdate needs the per-well second moment `S_k` (symmetric, so `d_k(d_k+1)/2` floats), plus `t_k = sum x` (D floats total) and `n_k` (K floats). Deployed atoms cost `A x D` floats. Everything at float32; budget 524,288 B = 131,072 floats.
  - At the incumbent's own partition (K=5, wells `[303,305,573,501,366]`): covariance triangle = 450,084 floats = **1,800,336 B**; honest total with the 524,288 B atom block = **2,324,624 B = 4.434x budget**. ⛔ Exact downdate at K=5 is NOT affordable — not marginally, but by 4.4x. If the single design keeps the incumbent's partition, the rows-3/4 comparison is NOT matched-bytes and I must say so.
  - But `sum_k d_k^2` falls QUADRATICALLY as the wells get finer while the atom block only falls linearly. Sweeping K with wells as equal as the grid allows, and requiring `A*D + sum_k d_k(d_k+1)/2 + D + K <= 131,072`:
    | K | d | A max | rank/d | total span | cov B | atoms B | honest total |
    |---|---|---|---|---|---|---|---|
    | 5 (incumbent) | 410 | 64 | 0.156 | 320 | 1,800,336 | 524,288 | 2,324,624 (4.434x) |
    | 20 | 102 | 11 | 0.107 | 220 | 423,536 | 90,112 | 521,920 (0.995x) |
    | **22** | **93** | **15** | **0.161** | **330** | **385,400** | **122,880** | **516,560 (0.985x)** |
    | 24 | 85 | 19 | 0.223 | 456 | 353,632 | 155,648 | 517,568 (0.987x) |
    | 32 | 64 | 30 | 0.469 | 960 | 266,240 | 245,760 | 520,320 (0.993x) |
    | 52 | 39 | 42 | 1.066 | 2048 | 165,440 | 344,064 | 517,904 (0.988x) |
  - ⭐ **K=22, A=15 is the like-for-like point**: restriction ratio `A/d = 0.161` against the incumbent's `64/410 = 0.156`, total deployed span 330 against the incumbent's 320, and the honest ledger (atoms + full covariance + means + counts) is **516,560 B = 0.985x of the 524,288 B pin — UNDER budget.** ⭐ So "composes AND forgets at matched bytes" is arithmetically available. Whether it also composes is a measurement I have not made.
  - ⚠ **THE TRAP, recorded so I do not walk into it:** from K~52 upward `A >= d_k`, so the per-well rank saturates at `d_k` and the atom block spans the whole well. The read is then an identity projection and the CONTENT carries no information — the address book does all the work. That store is DECORATIVE by charter C.1/C.4, and its C.2 controls would stop behaving: `pca_landscape_removed` (zeroed) still collapses, but `pca_random_refit` (random atoms, book refit) would NOT, because any full-rank block spans the same space. ⛔ A fine partition is not free: past the saturation point it buys the byte budget by giving up the store. The design must sit where `A < d_k` strictly, and I will report `rank/d` on every row so a saturated arm is visible rather than hidden.
  - ⭐ This is the dial w5 left open in code rather than a new idea: `CONTENT_PARTITION`/`oracle_content_fine` (`exp_w5_compose_field.py:139-148`) exists precisely to ask "whether the byte pin binds the content ceiling at all, or whether the WELL GRANULARITY does." My answer is that granularity is what makes the DELETION property affordable, which is a question w5 did not ask of it.
  - ⛔ Charter compliance: charter §B's closing sentence lists **partition** among the things a hub MAY change. "Placement exhausted" (w8-common §3.5) is about LEARNED well PLACEMENT returning a null against inherited grouping; re-cutting granularity with the same inherited label-MI criterion is a different dial and is not the exhausted one. I will keep the placement RULE inherited and change only K.
- 2026-09-20T17:22Z — ⚠ INSTRUMENT DISTINCTION my task file's wording hides, and I must not conflate: **"row 3 / row 4" is NOT `exp_gen_cell` vocabulary.** Two different instruments:
  - `exp_gen_cell.py` (the "generation-table cell") scores `in_corner` (balanced accuracy, positives = B corner, negatives = matched-seen) and `composition_B` (per-factor + aggregate probe accuracy) on a **single 72-request corner**, against `ARMS = (clu, store_removed, knn_proto, lace, ebm_compose, dynread)` plus `x0`. Headline = `DYNREAD_HEADLINE = "external composition-B aggregate"`.
  - Rows 3 and 4 and the >=49 paired-corner estimand belong to `gen_scoreboard` / `exp_gen_factor_wells`: row 3 = `r3_comp_2way` (60 tiled `pair-*` sub-corners), row 4 = `r4_comp_nway` (49 Montero `nway-*` corners), both scored as **distinct valid NOVEL tuples** (`score_novelty_row`, `gen_scoreboard.py:410`), estimand = mean paired per-corner difference, margin `DECORATION_MARGIN_NOVELTY = 1.0`, bootstrap 10,000 with `rng = default_rng(20_260_919)` plus an exact two-sided sign test.
  - `exp_gen_placement_table.py:48` already makes this criticism in code ("scored on a single 72-request cell against an estimand that requires >= 49 paired corners"). ⭐ So my three RESULT rows are on TWO instruments and I will label every number with which: rows 3/4 -> `exp_gen_factor_wells` estimand; generation-table run 3 -> `exp_gen_cell`. ⛔ I will not quote a cell number against a row-3/4 verdict.
- 2026-09-20T17:24Z — ⭐⭐ THE LIKE-FOR-LIKE TRIANGLE CLOSES. Address book (`TensorAddress`, R=128) costs `4*(sum_f card_f*R + sum_k R*r_k + 2*sum_k r_k)`; `sum_f card_f = 57`.
  | object | store (honest) | address book | TOTAL params | vs 179,968 |
  |---|---|---|---|---|
  | `pca_tensor` (incumbent, K=5, A=64) | 524,288 B = 131,072 p | 195,584 B = 48,896 p | **179,968** | 0.00% |
  | `mlp_180k` (the learned bar) | — | — | **179,968** | 0.00% |
  | **single design K=22, A=15** | 516,560 B = 129,140 p | 200,784 B = 50,196 p | **179,336** | **−0.35%** |
  - ⭐ The matched-bytes single design lands within **0.35%** of BOTH the incumbent arm and `mlp_180k`, against a declared `PARAM_TOLERANCE = 0.10`. So rows 3-4 can be scored three-ways at matched parameters with no new rival to build and no budget argument to make. That is the cleanest comparison this hub could have asked for, and it was not designed in — it falls out of holding the restriction ratio at the incumbent's 0.156.
  - ⛔ For contrast, the same object at the incumbent's K=5 partition totals **632,105 params = 3.51x** `mlp_180k`. That is the version I would have to disclaim. K=22 is not a nicety; it is the difference between a result and a caveat.
- 2026-09-20T17:25Z — Sent the full arithmetic (K sweep, saturation trap, exactness-assertion pointers, and the `DELETED_FACTOR/VALUE = 2,0` pin) to peer session `chlu-w8-downdate-bc`, which is building the object I score and whose LOG showed it about to reproduce the incumbent's K=5 partition. ⭐ Coordination, not instruction: I said explicitly that if it has a reason to hold K=5 I will score it as built and report the 4.43x ledger honestly. Delivery confirmed queued; no reply yet, and I will not treat silence as agreement.
- 2026-09-20T17:32Z — Wrote `PREREG-w8-single-design-1.md` (scored run 1 of 3), BEFORE any held-out number, per charter A.1. It declares: the object (`cov_tensor` = sufficient-statistic content, K=22/A=15, same read, same label use), the summed-not-netted ledger, selection on `val` only, the three RESULT rows with their bars and EXPECTED DIRECTIONS, the two DISPLAY rows with the transformer-with-refuse-token comparator, the exactness assertions WITH a negative control, and the charter §C floor checks. It also pre-commits the two falsifications my task file names: (i) losing rows 3-4 to `pca_tensor` means the properties stay incompatible and that is the answer; (ii) if the retrained-without baseline matches us, our advantage is COST not capability and I say so in those words with both cost lines printed.
- 2026-09-20T17:32Z — ⛔ No ledger row written to `/c/chlu/SCORED-LEDGER.md` yet: the ledger row goes in immediately BEFORE the run, and the run is gated. Nothing scored so far.
- 2026-09-20T17:48Z — ⚠⚠ **CORRECTION, and it goes against my earlier arithmetic.** Peer `chlu-w8-downdate-bc` challenged my well sizes; it was right that mine were wrong, and I have now measured the realised partition myself from `partitions.npz['factor']` rather than trusting either of us. BOTH of us quoted a different leg:
  - My `[303, 305, 573, 501, 366]` came from `tests/test_gen_dynamical_read.py:152` — a **test fixture**, not a deployed partition. ⛔ Withdrawn. That is the error, and it invalidates the K-sweep constants in my 17:14Z entry (the quadratic-vs-linear SHAPE of that argument survives; the numbers do not).
  - The peer's `[771, 940, 217, 60, 60]` is the **val/selection** leg (`hub-factor-wells/run-val`).
  - The **test/scored** leg — the one my RESULT rows are scored on — is **`[777, 939, 216, 58, 58]`** (`hub-factor-wells/run/partitions.npz`, pool 131,072 rows; identical in `hub-field-store/runs/base`).
  - Measured honestly on the scored leg, `A=64`: triangle 770,441 floats; honest total with a **float64** statistic = **6,704,240 B = 12.79x the 524,288 B pin** (f32 statistic: 3,614,264 B = 6.89x). My 4.43x was too kind by ~3x. The peer's 12.71x is the val leg; 12.79x is the scored leg. ⭐ The conclusion is unchanged and stronger: exact downdate at K=5 is not affordable by an order of magnitude.
- 2026-09-20T17:48Z — ⭐⭐ INDEPENDENT FINDING that falls out of measuring the real partition, and it is about the INCUMBENT, not my object: **`pca_tensor` is already saturated in 2 of its 5 wells.** On the scored leg the wells are `[777, 939, 216, 58, 58]` and `A = 64`, so wells 4 and 5 have `d_k = 58 < 64`: their atom block spans the whole well, the projection there is the identity, and the content in those wells carries no information. Worse on live rank — `partitions.npz['dead']` marks 64 dead channels, `[0, 0, 0, 46, 18]` per well, so the EFFECTIVE well sizes are `[777, 939, 216, 12, 40]`. Those two wells contribute an effective rank of 12 and 40 against 64 deployed atom rows apiece.
  - Consequence for the charter-C controls, which nobody has had to state: in those two wells `pca_random_refit` (random atoms, book refit) CANNOT collapse, because any full-rank block spans the same space. The C.2 control is partially blind there. It is 5.7% of the atom bytes (116 of 2048 channels), so it does not overturn the incumbent — but `rank/d_k` belongs on every row and I will print it.
  - ⭐ This also sharpens the peer's well-posedness fix: a rank-deficient well's trailing eigenvectors are not a function of the data at all (LAPACK returns an arbitrary rotation of a numerically null eigenspace), so "two refits of the same rows disagree" is a property of the INCUMBENT's content too, not a defect the covariance route introduces. It has simply never had to be a function of its fit set twice.
- 2026-09-20T17:50Z — Re-ran the matched-bytes sweep on the TRUE arithmetic, over the 1,984 live channels, with the statistic at float64 (the peer measured downdate residuals of 1e-10..5e-10 on `S` and reports float32 would not hold them; I accept that measurement and treat f64 as required):
  | statistic | K | d | A | rank/d | cov B | atoms B | honest total | x budget | span |
  |---|---|---|---|---|---|---|---|---|---|
  | float64 | **37** | 53 | **9** | **0.170** | 449,680 | 73,728 | **523,408** | **0.998** | 333 |
  | float64 | 42 | 47 | 15 | 0.319 | 399,056 | 122,880 | 521,936 | 0.996 | 630 |
  | float64 | 64 | 31 | 31 | 1.000 | 270,336 | 253,952 | 524,288 | 1.000 | SATURATED |
  | float32 | 21 | 94 | 16 | 0.170 | 386,880 | 131,072 | 517,952 | 0.988 | 336 |
  ⭐ **A matched-bytes single design still exists: K=37, A=9, float64 statistic, 523,408 B = 0.998x the pin, unsaturated, total deployed span 333** against the incumbent's realised span of 308 (`sum_k min(64, d_k)` on `[777,939,216,58,58]`) — or 244 on live rank. So the design survives the correction; ⛔ my earlier K=22/A=15 point is withdrawn and superseded by K=37/A=9.
  - ⚠ The param triangle must be recomputed too, and the −0.35% coincidence does NOT survive: address book at K=37, r=9, R=128 is a different size. I will recompute it before quoting it and will not repeat the 179,336 figure.
- 2026-09-20T17:50Z — DIVISION OF OBJECTS, agreed in substance with the peer. Its task file pins K=5 with tolerance 0 and explicitly forbids netting the covariance into the store budget; it is right that re-pinning K to bring the covariance under 524,288 B is paying for the statistic by deleting content, and it is a spoke and builds what is pinned. ⭐ So there are TWO objects and they answer two different questions, and I will score both rather than argue about one:
  - **Object A (the peer's, K=5, honest overrun 12.79x)** — answers "can ONE object carry both properties at all". My task file explicitly anticipates this outcome: "the honest ledger (store + covariance) exceeds the budget => the comparison is not matched. ⛔ Print it rather than netting it."
  - **Object B (mine, K=37/A=9, 0.998x)** — answers "can it carry both AT MATCHED BYTES", which is the only version that can win a RESULT row against `pca_tensor` without a caveat. Charter §B permits a hub to change the partition; I build this one in my own worktree as my arm, not as a request to the spoke.
- 2026-09-20T18:05Z — Built the single design in my worktree: `chlu/experiments/exp_w8_single_design.py` (`WellStats`, `CovContentStore`, `plan_partition`, `assert_not_saturated`, `honest_byte_ledger`) + `scripts/w8_single_design_dev.py`. ⛔ DEV only, NOT a scored run, no held-out row touched. Deliberately NOT a subclass of `PCASpanStore`: it exposes the same three attributes the span interface reads (`modes`, `well_channels`, `dim`) so every charter-C control, floor and read transfers unchanged, but the derivation and the ledger differ and inheriting would hide both.
- 2026-09-20T18:05Z — ⭐⭐ **DEV MEASUREMENT 1 — the exactness claim HOLDS.** Scored-leg partition K=5 `[777,939,216,58,58]`, A=64, 20,000-row pool subsample, seed 0, 1,669 red rows (8.34%), `object_hue = 0.0`.
  | well | d_k | effective | rank downdated | rank refit | max abs dev | subspace dist |
  |---|---|---|---|---|---|---|
  | 0 | 777 | 777 | 63 | 63 | 9.313e-10 | 3.106e-09 |
  | 1 | 939 | 939 | 63 | 63 | 2.274e-13 | 0.000e+00 |
  | 2 | 216 | 216 | 63 | 63 | 0.000e+00 | 0.000e+00 |
  | 3 | 58 | 12 | 12 | 12 | 4.274e-15 | 6.131e-15 |
  | 4 | 58 | 40 | 40 | 40 | 8.016e-14 | 1.334e-13 |
  **max abs dev 9.313e-10; ranks agree in every well.** ⭐ The downdate reproduces a refit-without-those-rows to float precision, with no pass over the data and the items never kept.
  - ⭐ INDEPENDENT CONFIRMATION that the rank cut is finding real structure and not a threshold artefact: the cut lands at exactly 12 and 40 in wells 3 and 4, which is precisely `58 − 46` and `58 − 18`, the LIVE channel counts from `partitions.npz['dead']`. Two unrelated instruments (an eigenvalue threshold at 1e-12 relative, and a dead-channel mask computed at partition time) agree exactly. I did not tune the threshold to make that happen.
- 2026-09-20T18:05Z — ⚠ **MY FIRST NEGATIVE CONTROL DID NOT BIND, and I am keeping it in the report rather than replacing it quietly.** Nudging a single entry `S[0,0]` by 1e-3 relative moved the atoms by only 8.680e-06, under my 1e-4 threshold — because one entry of a 777x777 matrix barely moves a leading eigenspace. That is evidence about the CONTROL, not about the check. Deleting it would have hidden a real limitation of the obvious control design, so it stays in the output JSON labelled "did not bind".
  - The control that BINDS asks the question the exactness claim actually makes: **would this comparison notice if the store had been downdated by the WRONG ROWS?** Downdating a size-matched set of 1,669 NON-red rows and comparing against the red refit gives **max abs dev 1.952e+00** against the true downdate's 9.313e-10 — a **separation of 2.1e9x**. ⭐ The check can fail, and it fails hugely when it should.
- 2026-09-20T18:05Z — DEV cost line (20k subsample; the full-pool line follows and will be far more favourable since only the refit side scales):
  - downdate: **0.276 s, 0 rows refitted, 0 passes over the data**
  - refit-without: **1.429 s, 18,331 rows refitted, 1 pass** — ratio **5.2x**
- 2026-09-20T18:05Z — ⚠ LEDGER CORRECTED UPWARD, against my own interest. My module first counted the statistic at 4 B/float. It is retained at **float64** (the downdate residual is 1e-10..5e-10 and float32 does not hold it), so counting it at 4 would be netting by dtype. `WellStats.stat_bytes()` now counts 8 B/float and the measured honest ledger at K=5/A=64 is **6,704,240 B = 12.787x budget** (atoms 524,288 + statistic 6,179,952) — matching my hand calculation exactly.
- 2026-09-20T18:12Z — ⭐⭐ **DEV MEASUREMENT 2 — the exactness claim holds on the FULL pool.** 131,072 pool rows, 11,044 red (8.43%), seed 0, scored-leg partition, A=64.
  | well | d_k | effective | rank dd | rank refit | max abs dev | subspace dist |
  |---|---|---|---|---|---|---|
  | 0 | 777 | 777 | 63 | 63 | 7.276e-12 | 9.975e-14 |
  | 1 | 939 | 939 | 63 | 63 | 3.725e-09 | 9.100e-09 |
  | 2 | 216 | 216 | 63 | 63 | 0.000e+00 | 0.000e+00 |
  | 3 | 58 | 12 | 13 | 13 | 5.199e-12 | 1.201e-11 |
  | 4 | 58 | 40 | 40 | 40 | 1.324e-14 | 2.851e-14 |
  **max abs dev 3.725e-09; ranks agree in every well.** Negative control (b) fired at 1.958e+00 — **separation 5.25e8x**. Control (a) again did not bind (9.704e-06), reported as such.
  - ⚠ Small honest discrepancy worth recording: on the full pool well 3's rank cut keeps **13** directions while the dead-channel mask says 12 live. The downdate and the refit AGREE at 13, so exactness is unaffected — but the two instruments that agreed exactly on the 20k subsample disagree by one here. The dead mask is a partition-time heuristic; the eigenvalue cut is the operative criterion. ⛔ I will not present them as interchangeable.
  - **Cost line, full pool:** downdate **0.904 s, 0 rows refitted, 0 passes over the data**; refit-without **7.476 s, 120,028 rows refitted, 1 pass**. Ratio **8.3x** (5.2x at 20k — the ratio grows with pool size because only the refit side scales).
  - ⛔ This is the CLU-side cost only. The comparator my headline row needs is the LEARNED baselines retrained without red, which is a different and much larger number (the existing instance of that comparator refits 29,099 rows at ~75 s per learner per seed). Both cost lines get printed on the row; neither stands in for the other.
- 2026-09-20T18:12Z — STATUS: the "forgets" half of the target is now **measured**, not argued, at the incumbent's own partition — but at **12.787x the byte budget**. The "composes" half and the matched-bytes question are both open. Next DEV step: re-cut the partition to the matched-bytes point (K=37, A=9, float64 statistic, 0.998x budget) using the SAME inherited label-MI criterion, and measure composition on `val` only.
- 2026-09-20T18:22Z — ⛔ **STRUCTURAL BLOCK on the matched-bytes route via the inherited criterion, found by running it.** `factor_aligned_balanced_labels` refuses every K > 6: *"k=N exceeds the 6 factors -- a factor-aligned partition cannot make more wells than there are factors to align them to"* (`chlu/core/gen_factor_partition.py:270`). So the label-MI criterion caps K at 6, and my K=37 plan cannot be reached with it. The re-cut sweep returned NO matched-bytes point and exited 1.
  - ⭐ But the repo already ships the variant that does work, and its own comment says so: the `ward_balanced` partition cuts the correlation **dendrogram leaf order** into equal wells, and `exp_gen_factor_wells.py:592-594` records the target file's sanction verbatim — *"a balanced variant cutting the dendrogram leaf order gives equal wells at any K. Both keep the correlation structure and neither sees a label."* ⭐⭐ It is **label-free**, so a single design built on it is **strictly MORE label-free than the incumbent** (whose partition uses label-MI), not less. That strengthens charter B.6 rather than bending it.
  - ⚠ It does change the comparison: the incumbent is on `factor`, so a ward_balanced arm differs from it in TWO ways at once (criterion and granularity). ⛔ I will not confound those. The control that separates them is **ward_balanced at K=5, A=64** — same criterion as my arm, same granularity as the incumbent — so criterion and granularity each get their own row.
- 2026-09-20T18:22Z — Also measured on the way past, and it is the honest headline for the K=5 object: the incumbent's realised deployed span is **308**, not 320, because two of its wells are 58 channels wide and cap at 58 rather than 64. Honest ledger for the single design at the incumbent's partition: **6,704,240 B = 12.787x**.
- 2026-09-20T18:32Z — ⭐ A matched-bytes partition EXISTS via the label-free `ward_balanced` cut. Sweep over K (equal wells, A = largest that fits the honest float64 ledger and saturates no well):
  | K | wells | A | rank/d | span | atoms B | statistic B | total B | x budget |
  |---|---|---|---|---|---|---|---|---|
  | <=34 | — | — | — | — | — | — | — | no atoms fit |
  | 36 | 56..57 | 4 | 0.071 | 144 | 32,768 | 490,912 | 523,680 | 0.999 |
  | **38** | 53..54 | **7** | 0.132 | 266 | 57,344 | 466,400 | **523,744** | **0.999** |
  | 40 | 51..52 | 9 | 0.176 | 360 | 73,728 | 444,352 | 518,080 | 0.988 |
  | 42 | 48..49 | 12 | 0.250 | 504 | 98,304 | 424,400 | 522,704 | 0.997 |
  | 48 | 42..43 | 18 | 0.429 | 864 | 147,456 | 374,528 | 521,984 | 0.996 |
  ⛔ Below K=36 **no atoms fit at all** — the float64 statistic alone exceeds 524,288 B. So the matched-bytes window is narrow and it is entirely in fine-partition territory.
- 2026-09-20T18:35Z — ⚠⚠ **THE FINDING THAT MATTERS MOST TODAY, and it is a threat to the headline row, not a confirmation.** At the matched-bytes partition the downdate reproduces the refit's SUBSPACE but not its BASIS, and the read uses the basis.
  | K | A | max abs dev (atoms) | max SUBSPACE dist | min rel spectral gap at the cut | ranks agree |
  |---|---|---|---|---|---|
  | 38 | 7 | **1.087e+00** | 1.195e-15 | 1.647e-03 | yes |
  | 40 | 9 | **1.826e+00** | 9.670e-16 | 1.107e-03 | yes |
  | 42 | 12 | 1.421e-14 | 2.479e-15 | 3.077e-04 | yes |
  | 44 | 14 | **6.625e-01** | 7.362e-14 | 4.313e-05 | yes |
  | 48 | 18 | 1.490e-13 | 1.623e-13 | 6.925e-05 | yes |
  - The span is identical to **1e-15** in every case. The individual eigenvectors rotate inside near-degenerate eigenvalue pairs — eigenvector non-uniqueness, not a downdate error. ⭐ So "downdate == refit" is TRUE at the level of what the store can represent.
  - ⛔ **But I then checked end-to-end through the repo's own read** (`span_basis` + `render_span`, identical coefficient vectors through both stores): **max rendered deviation 2.14 / 5.45 / 3.85** at K=38/42/48, while the **projector** agrees to **1.5e-07..2.1e-07**. So the subspace is the same and the READ IS NOT. The basis `U_k` is re-derived from the atoms on load (`gen_read.py:91-106`, it costs zero bytes), so a rotation inside the span silently re-interprets every address-book coefficient.
  - ⚠ **Consequence, stated plainly:** an address book fitted before the deletion is **stale** against the post-deletion basis. If I score post-deletion composition naively, it will collapse — and it would collapse for a reason that has NOTHING to do with the w5 eviction mechanism, while looking exactly like it. ⛔ That is a trap I would have walked into and reported as a null.
  - ⭐ The gauge is fixable and cheaply: after downdating, re-express the new atoms in the gauge closest to the old basis (an orthogonal Procrustes inside the span, O(A^2 d), no pass over data, span provably unchanged). That keeps the address book meaningful while leaving the deletion exact. ⛔ It is a DESIGN DECISION and I will declare it in the PREREG as an amendment, not adopt it silently — and it needs its own control: the Procrustes must be shown not to be doing any of the representational work (it is a rotation; the span is bit-identical before and after it).
  - ⭐ Note the two partitions where the atoms already agree without any gauge fix — K=42/A=12 (1.4e-14) and K=48/A=18 (1.5e-13) — are the ones whose spectral gap at the truncation rank happens to be benign in every well. The gap is measurable per well and printable, so "is this object's deletion exact at the atom level" is a checkable property of a candidate, not a hope. ⛔ Gap alone does not predict it (K=44 has a larger min gap than K=48 and fails), because what matters is the gap in the WORST well, so I will print the per-well gap and not a summary.
- 2026-09-20T18:44Z — ⛔ **MY GAUGE FIX FAILED. Reporting it as a negative result rather than iterating quietly.** Orthogonal-Procrustes alignment of the downdated atom rows to the pre-deletion basis did NOT restore read identity: after the gauge, rendered deviation was **1.739 / 2.778 / 4.462** at K=38/42/48 (before: 2.137 / 5.449 / 3.854) while the projector still agreed to ~1e-7. At K=48 it got WORSE.
  - **Why, and it is instructive:** `span_basis` re-derives the basis by SVD of the ATOM BLOCK. My gauge made the direction rows exactly orthonormal, so the block's singular values became degenerate (all 1 apart from the mean row) and the SVD's rotation inside that degenerate spectrum is once again arbitrary. I destroyed the very eigenvalue scaling that would have made the basis determinate. ⭐ Deploying unit eigenvectors is the INCUMBENT's convention too (`PCASpanStore` stores `Vt` rows), so `pca_tensor`'s span basis is equally gauge-arbitrary — it is simply never re-derived from changed atoms, because row 7 for `pca_tensor` deletes from the ADDRESS BOOK and leaves the atoms untouched (`exp_gen_read.py:966 delete_value_from_address`, declared "an operation on the INTERFACE, not on atoms").
  - ⭐⭐ **So the real structure of the problem is now clear, and it is not a bug in the downdate.** For ANY store whose atoms change, an address book fitted against the old atoms is stale. The incumbent evades this by never changing its atoms. The single design changes its atoms — that is the whole point, it is what makes the deletion a B.4 operation on atoms — and the price is that the INTERFACE must be brought back into correspondence.
  - ⚠ **Consequence for the target, stated plainly and early:** the single design's deletion is exact in the STORE (subspace to 1e-15, the peer's independent measurement 1.164e-10 absolute) but the address book must be re-fitted, and an ALS refit is a pass over the pool. ⛔ That would break the "no pass over the data" property at the interface even though it holds at the store. **This is the live risk to the headline row and I am not going to hide it behind a passing store-level assertion.**
  - ⭐ The candidate resolution, not yet built or measured: the address book is a ridge regression, so its normal equations are themselves sufficient statistics and can be downdated by the same rank-1 subtract. That would make the whole object — content AND interface — downdatable with no pass over the data. It costs more bytes and it is a design change, so it goes in as a PREREG amendment with its own ledger, not as a silent fix.
- 2026-09-20T18:52Z — ⭐⭐⭐ **THE GAUGE PROBLEM IS SOLVED, and the fix costs no bytes and no data pass.** The error in my failed attempt was fixing the wrong object: I tried to rotate the ATOMS to preserve the basis. The right move is to leave the atoms alone — they are correct — and **carry the ADDRESS BOOK across the basis change**.
  - The read is `x_hat = c_k U_k^T` with `c_k = a(request) W_k`. If deletion takes the basis `U_k -> U'_k`, the coefficients that mean the same thing in the new basis are `c'_k = c_k R_k` with **`R_k = U_k^T U'_k`** (r x r). So the updated book is simply `W'_k = W_k R_k`.
  - ⭐ `R_k` is computable **at delete time** from the two bases the delete already has in hand. It is `r x r` per well, it touches no pool row, and it is transient — applied to `W_k` and discarded — so it adds **nothing to the deployed ledger**.
  - **MEASURED**, downdated store with its book carried, against refit-without store with its book carried the same way, identical requests:
    | K | A | max abs render dev | signal scale | relative |
    |---|---|---|---|---|
    | 38 | 7 | 3.576e-07 | 3.214 | 1.1e-07 |
    | 42 | 12 | 9.537e-07 | 3.766 | 2.5e-07 |
    | 48 | 18 | 9.537e-07 | 4.291 | 2.2e-07 |
    ⭐ Down from **O(1)** (2.137 / 5.449 / 3.854) to **float32 precision**. The single design now downdates **content AND interface** exactly, with **no pass over the data** at either.
  - ⛔ Honest statement of what this is and is not: it is not a repair of an inexact deletion — the deletion was always exact in the store. It is the observation that the store's basis is a GAUGE, that the address book is expressed in that gauge, and that a change of gauge is an exact, data-free change of coordinates. The residual 1e-7 is float32 arithmetic, not approximation.
  - ⚠ It needs its own control before it can be quoted: `R_k` must be shown to be a ROTATION doing no representational work (`R_k^T R_k = I` to precision, and the span bit-identical before and after applying it). A coordinate change that quietly improved the answer would be a mechanism, not a gauge fix. That control is not yet run.
- 2026-09-20T19:02Z — ⚠ **MY OWN CONTROL CAUGHT ME OVER-CLAIMING, which is what it was for.** The orthogonality audit on `R_k = U_k^T U'_k` FAILED: max deviation from `R^T R = I` was **0.912** (K=42) and **0.621** (K=48). So `R_k` is **not a rotation** and "gauge carry" was the wrong name for it. ⛔ I am not quietly renaming a passing claim — the first framing was wrong and the control is why I know.
  - `R_k` is not orthogonal because the deletion genuinely MOVES the span: `U_k^T U'_k` is the matrix of principal-angle cosines between the pre- and post-deletion subspaces, an isometry only if the span is unchanged. Under a real deletion it must not be.
- 2026-09-20T19:02Z — ⭐⭐ **And characterising it produced the best result of the session.** Principal-angle cosines for the red deletion at K=42/A=12, across all 42 wells:
  | deletion | min cos | median cos | frac cos > 0.99 | frac cos < 0.5 |
  |---|---|---|---|---|
  | **red** (`object_hue = 0.0`, 11,044 rows) | **0.1999** | 0.9999 | 0.938 | **0.002** |
  | **size-matched RANDOM** (11,044 non-red rows) | **0.9976** | 1.0000 | — | **0.000** |
  - ⭐ Deleting red leaves **93.8%** of directions essentially untouched (cos > 0.99) and strongly moves a **tiny, specific** set (min cos 0.20). A size-matched random deletion of the SAME number of rows moves **nothing** (min cos 0.9976).
  - ⭐⭐ **This is the exact inverse of the w5 eviction finding.** There, `evict_size_control` showed the real deletion's composition cost (−2.09/−0.91/−6.74) sat INSIDE the random-eviction range (−2.7584), and the verdict was "THE COST IS THE MECHANISM, NOT THE TARGET -- an item store loses composition when it loses atoms, whichever atoms." Here the target and the size-matched control are **cleanly separated**: the covariance route's change is concentrated on the deleted content and a random deletion of equal size is almost a no-op. ⛔ That is a mechanism/target separation measured on the object itself — it is NOT yet a composition number, and it does not become one until the corners are scored.
  - So the honest description of `R_k`: **near-identity on the 94% of directions the deletion does not touch, and a genuine contraction on the few it does.** Carrying the address book through it transports the book unchanged where the content is unchanged, and registers real change where the content really changed. That is the correct behaviour, and it is why the downdate-vs-refit rendered agreement is 1e-7 while `R` is nowhere near orthogonal.
  - ⛔ RENAMED in code and in every future caption: not a "gauge rotation" but a **basis-change carry**, with its principal-angle spectrum printed. The orthogonality audit stays, inverted in meaning: it now measures HOW FAR the span moved, and a value near zero would mean the deletion did nothing.
- 2026-09-20T19:12Z — Gate re-checked: `/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json` exists, `all_three_assertions_passed`, but **`status: partial`** and **`condition: val`**. ⛔ My RESULT rows score on the `test` condition. Spending a scored run against a val-condition partial object would burn irreplaceable budget on a number that could not enter a table. **I do not spend a scored run.** Budget remains 3/3 single-design + generation-table run 3.
- 2026-09-20T19:12Z — Committed `704a0ec`, tree clean, branch bundled to `/c/chlu/outputs/w8-single-design/w8-single-design.bundle` — `git bundle verify` reports "The bundle records a complete history." Outputs verified by directory listing (they are gitignored): `LOG.md`, `PREREG-w8-single-design-1.md` (+ Amendment 1), `START_UTC.txt`, `dev/downdate_check_K5_A64_seed0.json`, `dev/matched_bytes_recut.json`, the bundle.
- 2026-09-20T19:14Z — Wrote PREREG Amendment 1: declares the basis-change carry (with its measured effect and its zero deployed byte cost), records the failed Procrustes attempt, records the control that failed and the claim it corrected, withdraws K=22/A=15 and the 179,336-param triangle with the test fixture they rested on, and pre-registers the `ward_balanced @ K=5, A=64` control that separates criterion from granularity.

---

# FINAL REPORT — w8 hub/single-design

**Target:** one object that composes AND forgets.
**Window:** 2026-09-20 16:36Z to 19:14Z (2 h 38 m). Local CPU, 3 threads. Base `base/iter7 = ec2ef7c`.
**Frozen harness untouched.** Branch `w8/single-design`, HEAD `704a0ec`, bundled and verified.

## 1. Scored runs — ALL of them

| run | status | reason |
|---|---|---|
| single-design 1/3 | ⛔ **NOT RUN** | gate `DOWNDATE-READY.json` is `status: partial`, `condition: val`; RESULT rows require `test` |
| single-design 2/3 | ⛔ **NOT RUN** | not reached |
| single-design 3/3 | ⛔ **NOT RUN** | not reached |
| generation-table 3 (the last one) | ⛔ **NOT RUN** | not reached |

⛔ **A declared NOT-RUN is never a null.** Nothing below is a held-out number. No row of
`/c/chlu/SCORED-LEDGER.md` was written, because a ledger row is written immediately before a run and no
run was started. **Scored 0/3.** The full budget survives intact for whoever picks this up.

## 2. What was established — DEV, on the fit pool and the val-side artefacts only

**2.1 ⭐⭐ The "forgets" half is measured, not argued.** The covariance route — retain `S_k = sum x x^T`,
`t_k = sum x`, `n_k` per well; derive atoms as `[mu_k ; leading eigenvectors]`; delete by rank-1 subtract —
reproduces a refit-without-those-rows **to float precision**, on the full 131,072-row pool at the incumbent's
own partition: **max abs dev 3.725e-09**, ranks agree in every well. The independent build (spoke
`w8-downdate`, val leg) measures **1.164e-10**, and on two different deletion sizes, so exactness is not a
property of how much is deleted. This restores charter B.4 to the content that wins rows 3 and 4 — the
property `pca_tensor` explicitly declares it gives up (`exp_w5_compose_field.py:485`).

**2.2 ⭐ The cost line the CLU does not pay.** Downdate **0.904 s, 0 rows refitted, 0 passes over the data**
against refit-without **7.476 s, 120,028 rows, 1 pass** — **8.3x**, and the ratio grows with pool size because
only the refit side scales. ⛔ This is the CLU-side number only; the comparator the headline row needs is the
learned baselines retrained without red, which is a different and larger number (~75 s per learner per seed
for 29,099 rows in the existing instance). Both belong on the row; neither substitutes for the other.

**2.3 ⚠ The honest ledger is over budget, and it is printed, not netted.** The statistic must be float64
(residuals ~5e-10 on entries of order 1e+9). At the incumbent's K=5 partition that is
**6,704,240 B = 12.787x** the 524,288 B pin; the spoke's four-part total including coupling and counts is
**24.45x**. ⛔ So at the incumbent's partition the object carries both properties **but the rows-3/4
comparison against `pca_tensor` is NOT matched-bytes**, exactly the outcome the target file told me to print
rather than net.

**2.4 ⭐ A matched-bytes design exists, and granularity is what buys it.** `sum_k d_k^2` falls quadratically
as wells get finer while the atom block falls only linearly. The inherited label-MI criterion **cannot** get
there — `factor_aligned_balanced_labels` refuses any `k > 6`. The shipped **label-free** `ward_balanced` cut
can, and the matched-bytes window is narrow: **K = 36..48**, e.g. **K=38/A=7 at 523,744 B = 0.999x** or
**K=42/A=12 at 522,704 B = 0.997x**, unsaturated. ⛔ Below K=36 the statistic alone exceeds the pin and no
atoms fit at all. ⭐ Because `ward_balanced` sees no label, such an object is **strictly more label-free than
the incumbent**.

**2.5 ⚠⚠ The trap that would have produced a false null, found and closed.** The deletion changes the atoms,
and `span_basis` re-derives the basis from the atoms on load — so an address book fitted before the deletion
is expressed in a basis that no longer exists. Rendered reads differed by **O(1)** between a downdated and a
refit-without store whose projectors agreed to **2e-7**. Scoring post-deletion composition naively would have
collapsed, **looking exactly like the w5 eviction failure while having nothing to do with it.**
The fix is to carry the address book, not the atoms: `W_k <- W_k R_k`, `R_k = U_k^T U'_k`, formed at delete
time from bases already in hand, touching no pool row, **zero deployed bytes**. Rendered downdate-vs-refit
agreement goes from O(1) to **~1e-7 relative**. ⛔ An earlier fix that rotated the ATOMS failed and is
recorded (it degenerates the atom block's spectrum, so the basis is arbitrary again).

**2.6 ⭐⭐ The most suggestive measurement, and it is NOT a result.** `R_k`'s singular values are the
principal-angle cosines between pre- and post-deletion spans:

| deletion | cos min | cos median | frac > 0.99 | frac < 0.5 |
|---|---|---|---|---|
| red (`object_hue = 0.0`, 11,044 rows) | **0.1999** | 0.9999 | 0.938 | 0.002 |
| size-matched RANDOM (11,044 non-red rows) | **0.9976** | 1.0000 | — | 0.000 |

Deleting red moves a small specific set of directions and leaves 93.8 % untouched; deleting the same number
of random rows moves nothing. ⭐ That is the **inverse** of w5's eviction verdict, where the real deletion's
cost sat inside the random-eviction range and the conclusion was "the cost is the mechanism, not the target."
⛔ **This is a property of the store, not a composition number, and it is not quoted as one.** It says the
mechanism *can* separate target from size. It does **not** say composition survives. That requires the
corners, and the corners were not scored.

**2.7 ⭐ An independent finding about the incumbent.** `pca_tensor` is **already saturated in 2 of its 5
wells** (`d_k = 58 < A = 64`; effective ranks 12 and 40 once dead channels are counted). Its realised deployed
span is **308**, not 320, and `pca_random_refit` — the charter C.2 random-atom control — **cannot collapse**
in those two wells, because any full-rank block spans the same space. It is 5.7 % of the atom bytes and does
not overturn the incumbent, but `rank/d_k` belongs on every row from here on.

## 3. Corrections I made against my own earlier work

- ⛔ Well sizes `[303, 305, 573, 501, 366]` were a **test fixture**, not a deployed partition. Withdrawn;
  every byte figure derived from them was wrong by ~3x. The scored leg is `[777, 939, 216, 58, 58]`.
- ⛔ `K=22 / A=15` and the "−0.35 % vs `mlp_180k`" param triangle are **withdrawn** with that fixture.
- ⛔ My ledger first counted the statistic at float32. Corrected **upward**, against my own interest, to
  float64 — 12.787x rather than 6.89x.
- ⛔ "Gauge rotation" was the wrong name for `R_k`, and my own orthogonality control is what proved it
  (0.912 deviation). Renamed to **basis-change carry**; the audit now reports the spectrum instead of
  asserting it away.
- ⛔ My first negative control did not bind (8.7e-06 against a 1e-04 threshold). Kept in the output labelled
  "did not bind" rather than deleted, and replaced by one that does: downdating by size-matched WRONG rows,
  which fires at **1.958** — a separation of **5.25e8x**.

## 4. Charter §C floor checks — one line each

- **C.1 matched-bytes lookup** — ⛔ NOT RUN. `pca_lookup_argmax` / `pca_lookup_convex` transfer unchanged to
  this content (same atom block, same interface) but were not scored.
- **C.2 landscape-removed / landscape-shuffled** — ⛔ NOT RUN. ⚠ Pre-registered caveat: in the incumbent's two
  saturated wells `pca_random_refit` **cannot** collapse; `rank/d_k` is printed so this is visible.
- **C.3 tie clause** — not reached. The object does deliver exact deletion on the same artefacts, which is one
  of the three qualifying properties, but no tie has been measured to apply it to.
- **C.4 distance-to-nearest-atom** — ⛔ NOT RUN.
- **§C.2 analogue that WAS run:** the deletion's own size-matched random control (§2.6), at the store level.

## 5. Cost line for the hub

2 h 38 m wall clock, local CPU, 3 threads, one machine. No GPU. No cluster leg (CSF3 unreachable — ⛔ NOT RUN,
per charter §D). 3 read-only sub-agents for code mapping; none running at close. Compute spent: partition
re-cut sweeps, two full-pool covariance builds + downdates + refits, three matched-bytes verification passes.
⛔ **Zero scored budget consumed.**

## 6. What the next hub should do first

1. **Wait for `DOWNDATE-READY.json` to reach `status: ready` on the `test` condition**, then score. The object
   and both comparators are specified in PREREG §3 and Amendment 1; nothing needs re-deriving.
2. **Apply the basis-change carry before scoring post-deletion composition**, or the row will read as a
   collapse that is pure coordinate staleness (§2.5). This is the single highest-value thing in this log.
3. **Run the `ward_balanced @ K=5, A=64` control** (Amendment 1 §A1.4) or rows 3-4 from a re-cut object are
   confounded between criterion and granularity.
4. **Decide the ledger question explicitly**, because it decides what kind of claim is available: at the
   incumbent's partition the object is 12.787x over budget and rows 3-4 are not matched-bytes; at K=38..48 it
   is matched-bytes but on a different (label-free) partition. ⛔ Both are honest; they are not the same claim.
5. ⚠ Not fixed, recorded so it is not rediscovered: `exp_gen_field_store.sweep` (:930) references `store_seed`
   while its parameter is `seed` (:884) — `--stage sweep` raises `NameError`. Out of my dial; untouched.

## 7. The one-sentence answer, as far as it is earned

⭐ **The two properties are not incompatible in principle** — one object now demonstrably composes-in-content
and forgets exactly, because the content is derivable from an additive sufficient statistic — **but at the
incumbent's partition it costs 12.8x the byte budget, a matched-bytes version exists only on a finer
label-free partition, and whether either still composes is UNMEASURED.** ⛔ The wave's question is not
answered; it is made scorable, with the trap that would have produced a false negative found and closed first.

---

# POST-REPORT ADDENDUM — 2026-09-20T19:40Z
⛔ Appended, not rewritten. The report above stands as written at 19:14Z; this section records what
arrived and what I measured afterwards, including a conclusion in §7 of that report that is now superseded.

- 2026-09-20T19:35Z — Peer `chlu-w8-downdate-bc` delivered its val smoke, and it closes the "composes" half at the pin. 24 val corners, shared frozen decoder (`01ef91f75eea`) + judge (`c8741ac417d8`):
  - `downdate_store` valid mean **0.8941** sd 0.0903 min 0.7361 max 1.0000
  - `pca_tensor`     valid mean **0.8941** sd 0.0903 min 0.7361 max 1.0000
  - paired delta **+0.0000 on every one of the 24 corners**.
  ⭐ So the covariance route reproduces the incumbent's read **exactly**, not approximately, at the byte pin. ⛔ It is `val`, it is 24 corners not the >=49 paired-corner estimand, and it is theirs to hand over rather than mine to quote as a result — but it means the object is not paying for its deletion property with composition at the pre-deletion read. ⚠ **This supersedes my report §7's "whether either still composes is UNMEASURED"** for the pre-deletion, val, K=5 case. It remains unmeasured post-deletion and on `test`.
- 2026-09-20T19:36Z — ⭐ Their diagnostic explains my failed Procrustes better than my own account did. Comparing their atom block against `PCASpanStore`'s on the same pool/partition/A: **max |atom difference| per well = [5.85e-01, 5.59e-01, 6.14e-01, 2.00e+00, 1.36e+00]** — the content differs by O(1) — yet each read **through its own address book** agrees to **8.05e-06** against a signal scale of 3.72 (median row L2 4.96e-05 against median row norm 31.3, 1.6e-6 relative). ⭐⭐ **The content is gauge-arbitrary on its own; only the (basis, book) PAIR carries meaning.** My Procrustes rotated one member of the pair to a target gauge while the other member — a basis re-derived by an SVD whose spectrum I had just made *more* degenerate — moved out from under it.
- 2026-09-20T19:36Z — ⭐ Their fix is better than mine for the gauge, and I say so plainly. `pinned_span_basis` stops asking a degenerate SVD where the basis should point: the store's direction rows are already orthonormal and already eigenvalue-ordered, so they ARE the basis, plus the mean's orthogonal component. Zero bytes, no SVD. Measured downdate-vs-refit through each: `gen_read.span_basis` **2.994e-03**, `pinned_span_basis` **3.725e-09** — **800,000x**, ranks identical. ⚠ Their own trap is the same shape as mine: admitting the mean's orthogonal component at a 1e-8 threshold let pure noise back in (the mean lies inside the directions' span in the rank-deficient wells; margin only 2.3x); pinning admission at 1e-4 took the deviation from 3.45e-07 to 3.725e-09. ⭐ The generalisation both of us arrived at from opposite directions: **a basis decision needs a declared margin exactly as an eigenvalue cut does.**
- 2026-09-20T19:40Z — ⭐⭐ **MEASURED, and it is a concrete gap in their fix that mine covers — and it bites the smoke they are running right now.** A pinned eigenvalue-ORDERED basis is only well-defined if the deletion does not change the ORDER. It does. Deleting red (11,044 rows), K=5, A=64, comparing pre- and post-deletion leading directions by overlap matching:
  | well | r | min rel eigen-gap (pre) | directions whose RANK-ORDER changed | max shift |
  |---|---|---|---|---|
  | 0 | 63 | 7.288e-06 | **7 / 63 (11.1%)** | 2 |
  | 1 | 63 | 3.930e-05 | **14 / 63 (22.2%)** | 3 |
  | 2 | 63 | 5.042e-05 | **22 / 63 (34.9%)** | 2 |
  | 3 | 13 | 1.184e-10 | **2 / 13 (15.4%)** | 1 |
  | 4 | 40 | 2.368e-05 | 0 / 40 (0.0%) | 0 |
  - ⭐ **Why this does NOT affect their 3.725e-09 exactness number:** that comparison is downdate vs refit-without, and BOTH sides have the same post-deletion spectrum, so both reorder identically and the permutation cancels.
  - ⛔ **But it does affect the read-back-with-no-refit smoke**, which is the pre-deletion BOOK against the post-deletion BASIS. There, slot *j* of the book multiplies a direction that is no longer direction *j* in 11-35% of slots. That is a **permutation mismatch, gauge in character, not a deletion effect** — so a drop in that smoke cannot be read as "the deletion cost composition" without separating it out first.
  - ⭐ My `W_k <- W_k R_k` carry absorbs this automatically: `R_k = U_k^T U'_k` is a general linear map and tracks the correspondence including permutations. So the two fixes are **complementary, not competing** — theirs removes the degenerate-SVD gauge, and a permutation match (or my carry) is still needed on top of it for the no-refit read.
  - ⚠ Honest bound on how much it matters: the reordered pairs sit at relative eigen-gaps of 7e-6 to 5e-5, i.e. they are nearly degenerate, so the SPAN barely moves when they swap. The damage is confined to the difference between the two swapped slots' coefficients, not the whole answer. ⛔ I have measured that the reordering HAPPENS; I have not measured what it costs the read, and I will not assert the second from the first.
- 2026-09-20T19:40Z — ⚠ CORRECTION TO MY OWN FRAMING, prompted by their point that my carry "absorbs BOTH the gauge and the real rotation". I checked the algebra rather than conceding it: with `W' = W R`, the read becomes `x_hat = a W R U'^T = (a W U^T) U' U'^T` — the OLD ambient answer **projected onto the NEW span**. So the carry does **not** mask the deletion; the deleted content is projected out, which is the correct semantics. ⭐ But their design is still cleaner on one axis that matters for the row-7 claim: with `pinned_span_basis` the address book's **bytes do not change at all**, so "no refit of anything" is literally true of the interface, whereas my carry rewrites `W`. ⛔ For a charter B.4 / row-7 argument that is a real advantage and I would take their version plus a permutation match over mine.
- 2026-09-20T19:52Z — Gate re-checked: `DOWNDATE-READY.json` is now **`status: ready`** — but **`condition: val`**, seeds [0], as the spoke said it always would be (it was scoped to build, not to score). ⛔ My RESULT rows need `test`, so this still does not open a scored run. **Scored budget remains 0/3 spent.** Recorded from the gate: `well_sizes [771,940,217,60,60]`, `rank_kept_per_well [63,63,63,16,41]`, `honest_total_bytes 12,817,450 = 24.4473x`, `worst_max_abs_dev 1.164e-10` (= 9.77e-4 of one float32 ulp at unit scale), negative control fired, all three assertions passed.
- 2026-09-20T19:52Z — The spoke's pinned-basis smoke, read from its log: pre-deletion `downdate_store` and `pca_tensor` both **0.8941** (paired +0.0000 over 24 val corners); **after deleting 11,024 red rows with no refit of anything: 0.0584.** ⭐ So the pinned basis alone does NOT save the no-refit read — it collapses. The spoke is now refitting the address book on the downdated store (`postdelete_refit`, ALS 40 sweeps, running).
- 2026-09-20T19:55Z — ⭐⭐⭐ **THE SEPARATION, RUN. The 0.8941 -> 0.0584 collapse is overwhelmingly GAUGE, not forgetting.** `scripts/w8_gauge_separation.py`, val-leg partition K=5 A=64, seed 0, 4,096 probe rows drawn from the 348,928 rows **outside** the fit pool. Four reads through the SAME post-deletion content, differing only in how pre-deletion coefficients are put into correspondence with post-deletion directions:
  | well | r | reordered | REAL deletion | NAIVE (the smoke) | after perm match | scale |
  |---|---|---|---|---|---|---|
  | 0 | 63 | 8/63 | 0.9261 | 18.2054 | 4.3569 | 15.133 |
  | 1 | 63 | 17/63 | 0.8647 | 11.6566 | 6.0056 | 18.024 |
  | 2 | 63 | 20/63 | 0.7835 | 8.3283 | 4.2144 | 9.476 |
  | 3 | 16 | 2/16 | 0.0000 | 0.4142 | 0.0798 | 1.369 |
  | 4 | 41 | 0/41 | 0.0000 | 2.2596 | 0.6195 | 4.812 |
  **Whole-vector totals** (signal scale 25.8591):
  - **REAL deletion effect** `||before − carry||` = **1.4898 = 5.76% of scale**
  - **NAIVE read error** `||naive − carry||` = **23.2798 = 90.03% of scale**
  - after permutation match = **8.5558 = 33.09% of scale**
  - ⭐ **naive error / real deletion effect = 15.63x.** The permutation removes **63.25%** of the naive error.
  - `carry` is provably the correct post-deletion answer: `c M A1^T = before · A1 A1^T`, the pre-deletion ambient answer projected onto the post-deletion span, so the deleted content is projected out and nothing else moves. Hence `||before − carry||` IS the real deletion effect — the component of the old answer lying outside the new span. **5.76%**, against red being 8.4% of pool rows. That is a sane magnitude.
  - ⚠ **REFINEMENT OF MY OWN EARLIER ADVICE, which was incomplete.** I told the spoke a permutation match would separate gauge from deletion. It only removes **63%** — a **33% residual** remains, which is rotation *within* near-degenerate subspaces, not mere slot swapping. ⛔ So a permutation is NOT sufficient; the full linear correspondence `M = U0^T U1` (my `W <- W M` carry) is what is needed. I corrected this to the spoke immediately rather than letting the weaker advice stand.
  - ⚠ Caveats that travel with every number above, non-negotiable: this is **FEATURE space, not the judge** — it bounds what the judge can see and does **not** predict a valid-fraction; and the coefficients are **ORACLE span coordinates**, which makes this a **DIAGNOSTIC disqualified as an arm by charter B.3 forever**. ⛔ No number here is a composition score and none may be quoted as one.
  - ⭐ What it licenses, and only this: the spoke's post-deletion ALS refit is repairing a **coordinate correspondence**, not recovering from lost content. A data-free `O(r^2 d)` correspondence should recover most of it, which would restore "delete, refit nothing" as a literal claim.
- 2026-09-20T20:00Z — Separation committed `5f024ef`, tree clean, branch rebundled and `git bundle verify` clean. Sent the full decomposition, the caveats and the explicit correction of my own permutation advice to the spoke while its ALS refit is still running, with a cheap concrete test it can run before the refit finishes (`W'_k = W_k M_k` from the two bases it already holds at delete time, then re-run the same 24-corner read). ⛔ No detached work; every run this session was foreground and polled to completion inside its turn. ⛔ Scored budget still 0/3 — the gate is `ready` but `condition: val`, and my RESULT rows need `test`.

---

- 2026-09-20T20:10Z — Re-ran the separation with the spoke's greedy matcher after it pointed out my per-row `argmax` can double-assign (so what I called a permutation was not guaranteed to be one). ⭐ The criticism was correct; the effect was small. Double-assignments per well: 1, 1, 5, 0, 0. With the greedy rule: after-permutation residual **32.66%** of scale (was 33.09%), permutation removes **63.72%** (was 63.25%). ⛔ Conclusion unchanged and I am not dressing a 0.4-point shift up as a correction of substance: real deletion effect **5.76%** of scale, naive read error **90.03%**, ratio **15.63x**, and a permutation still leaves a third of the error on the table. The residual is genuine rotation inside near-degenerate subspaces, which is why the full correspondence `M = U0^T U1` is needed rather than a re-indexing.
- 2026-09-20T20:10Z — Vendored the spoke's three additive files verbatim at its commit `bd9b1cf` (`chlu/core/downdate_store.py`, `chlu/experiments/exp_w8_downdate.py`, `tests/test_downdate_store.py`), committed `b32d3d1` with explicit provenance. ⛔ Vendored, NOT merged, so my branch's history is not entangled with a peer's (charter: a hub branches from the pin, not from a peer's branch) and attribution stays unambiguous.
- 2026-09-20T20:12Z — ⭐ **TEST-CONDITION STORE FITS.** `--stage fit --condition test`, my own out dir, using the SAME `partitions.npz` the scored arms use (`hub-factor-wells/run`, copied in rather than rebuilt — rebuilding could differ and would break comparability):
  - wells `[777, 939, 216, 58, 58]`, dead `[0, 0, 0, 46, 18]`, ranks `[63, 63, 63, 13, 40]`, rank/d `[0.0811, 0.0671, 0.2917, 0.2241, 0.6897]` — matching my own independent measurement of the test leg exactly.
  - store **524,288 B exact** (pin matches, tolerance 0) · covariance **12,310,672 B** (float64 `d_k^2` per well `[603729, 881721, 46656, 3364, 3364]`) · counts 11,488 B · coupling 5,288 + 1,322 B of holes.
  - **HONEST SUM 12,869,482 B = 24.5466x the pin** (spoke's val: 24.4473x). ⛔ Over, printed not netted.
  - coupling layout equals `gen_field`'s: 1,322 pairs, 60 indices checked. Hole punched at pair-000. Fit 29 s.
  - ⛔ Assertions NOT yet run on the test object. The fit alone is not the gate.
- 2026-09-20T20:14Z — ⛔⛔ **INCIDENT, MINE, AND I CAUSED IT: I overwrote the peer spoke's `DOWNDATE-READY.json`.** `exp_w8_downdate.py:73-74` defines `OUT_ROOT = $CHLU_HUB_OUTPUTS/w8-downdate` and `READY = OUT_ROOT/"DOWNDATE-READY.json"` at MODULE level — they ignore `--out` entirely. So my test-condition fit wrote its publish into the SPOKE's directory even though `--out` correctly sent the store and store.json to mine. ⛔ My error: I read the stage signature and the `--condition` plumbing before running a peer's code, but not the publish path.
  - **Damage, precisely.** The spoke's own `bd9b1cf` fix held — `fit` MERGED rather than blanked, so every assertion payload survived byte-for-byte (`downdate_equals_refit`, `no_pass_over_data_detail`, `items_kept_detail`, `smoke`, `smoke_pinned`, `post_delete`, `post_delete_refit`, `gauge`, `rank_report`). What WAS overwritten: `condition`, `well_sizes`, `dead_per_well`, `effective_well_sizes`, `rank_kept_per_well`, `rank_over_d_per_well`, `cov_bytes`, `honest_total_bytes`, `times_over_budget`, `content_sha256`, `utc`.
  - ⛔ **The dangerous state, live for ~90 seconds:** `condition: "test"` beside `all_three_assertions_passed: true`, when those assertions were run on **val**. A reader would have concluded a test-condition object passed assertions it never ran. That is exactly the failure the spoke's `bd9b1cf` commit was written to prevent, arriving through a different door.
  - **Restored** every overwritten descriptor from the spoke's own `build/val_seed0/store.json`. ⚠ First restore pass was WRONG on two fields: `rank_kept_per_well` / `rank_over_d_per_well` are not in `store.json`'s `rank_report`, so they fell back to my clobbered test values; caught and set to the val values `[63,63,63,16,41]` / `[0.082,0.067,0.29,0.267,0.683]` from the spoke's own message. Added an `INCIDENT_2026-09-20T19:29Z` field to the file describing all of it, and kept the pre-restore file at `DOWNDATE-READY.json.clobbered-by-single-design.bak` so the spoke can diff rather than take my word.
  - ⚠ **The restore is MY reconstruction, not the spoke's write.** I told it so explicitly and asked it to re-publish from its own code on val rather than trust my repair.
  - **Recurrence prevented** by redirecting `CHLU_HUB_OUTPUTS` to my own subtree for any run of that module, so `OUT_ROOT` resolves inside `/c/chlu/outputs/w8-single-design/` and my publishes cannot reach a peer. ⭐ Suggested a permanent fix to the spoke: derive `READY` from `--out`, or make `_publish` refuse to write a file whose existing `condition` differs from the one being published.
- 2026-09-20T20:20Z — Re-ran the test fit with `CHLU_HUB_OUTPUTS` redirected to my own sandbox. Publish landed at `/c/chlu/outputs/w8-single-design/sandbox/w8-downdate/DOWNDATE-READY.json`; the peer's gate was not touched. ⭐ Also checked a risk I introduced by writing their file as UTF-8: their reader specifies `encoding="utf-8"` explicitly (`publish_ready`, :118), so it is safe — the decode error was in my own check script, not their code.
- 2026-09-20T20:22Z — ⭐⭐ **ALL THREE ASSERTIONS PASS ON THE TEST-CONDITION OBJECT** (`--stage assertions --condition test`, my build dir, test-leg partitions):
  - **A1, pinned value `object_hue = 0.0` (11,044 rows):** max abs dev per well `[4.547e-13, 1.421e-14, 0.000e+00, 5.249e-12, 2.873e-14]`; **worst 5.249e-12 = 4.403e-05 of one float32 ulp at unit scale**; 1/5 wells bit-identical; ranks agree; counts, term counts and pair counts all equal.
  - **A1, random 64 rows:** worst **1.507e-11**; 2/5 wells bit-identical. ⭐ Two deletion sizes, so exactness is not a property of how much is removed.
  - **A1 negative control: FIRED** — one atom nudged by 1e-3 gives max abs dev 1.000e-03, check fires. ⭐ The passes are worth something because the check can fail.
  - **A2, no pass over the data:** corpus 131,072 -> `delete(64)` consumed **320 row-reads** (64 x 5 wells) with 5 tripwires armed; corpus 65,536 -> **320** again. Same number at both corpus sizes, which is what turns "it read few rows" into "it read only the rows it was asked to forget".
  - **A3, items never kept:** 26 arrays, every shape explained by `(A, d_k, K, T, NC)`; **object bytes 12,885,866 at 131,072 rows and 12,885,866 at 65,536 rows — grew by 0 B.**
  - ⭐ So the test-condition object is built and asserted. ⛔ The gate that blocked my RESULT rows is cleared **by my own build**, not by the spoke's val gate.
- 2026-09-20T20:24Z — ⭐⭐⭐ **THE CARRY WORKS IN JUDGE SPACE. The peer ran it on val, 24 corners, shared frozen decoder `01ef91f75eea` + judge `c8741ac417d8`:**
  | read | valid mean | sd | cost |
  |---|---|---|---|
  | before delete | 0.8941 | 0.0903 | — |
  | after delete, book UNCHANGED (naive) | **0.0584** | 0.0701 | — |
  | after delete, book RE-INDEXED (permutation) | **0.0451** | — | slots only |
  | **after delete, book CARRIED (`W <- W M`)** | **0.8964** | 0.0924 | **0 data rows, no refit** |
  | after delete, book REFIT (diagnostic) | 0.9005 | 0.0908 | 131,072 rows, ALS |
  - ⭐ The carry recovers **100.3%** of the collapse and lands within **0.004** of a full ALS refit, reading **zero data rows**. ⭐ My feature-space prediction held in judge space, which it did not have to — I was explicit that feature-space L2 bounds what the judge can see and does not predict a valid-fraction, and it happened to transfer.
  - ⭐ **The re-indexing leg independently confirms my 63% ceiling from the other side:** the permutation gives **0.0451**, i.e. no recovery at all and slightly worse than naive. On the peer's leg 49 of 246 slots move with **min matched cosine 0.3597** — the directions genuinely moved, so a permutation cannot express the change. ⛔ My withdrawal of the permutation advice was right, and this is the same finding arriving in judge space before the peer had read my separation.
  - ⭐⭐ **The control that settles what the collapse WAS:** refitting the book on the downdated store gives **0.9005 against 0.8941 before**. The deletion does not damage the content at all — the post-deletion store is as good as the pre-deletion one. The entire collapse was the stale interface. And since the downdated store equals the without-those-rows refit to 1.164e-10, **0.9005 is also the score a store fitted from scratch without those rows would get.**
- 2026-09-20T20:24Z — ⛔ **THE COST OF THE CARRY, declared by the peer and printed here rather than inferred: it rewrites 127,488 B of the address book's 158,664 B** — values, not count. No data row is read and no ALS runs, so charter B.4's "no retrain" holds, but ⛔ **this is NOT the coupling hole's ONE byte and must never be quoted beside it as if it were.** `M` is genuinely far from the identity: min singular value per well `[0.672, 0.732, 0.796, 1.000, 1.000]`, max departure from `I` `[1.96, 2.00, 1.96, 1.09, 2.00]`.
  - ⭐ The two honest lines, neither subsuming the other: the peer's **pinned basis changes ZERO bytes but is insufficient alone** (0.0584); **my carry is sufficient and costs 127,488 B per deletion** (0.8964).
  - **The object's honest shape for my table:** the STORE forgets exactly (1.164e-10 val / 5.249e-12 test, no data pass, no items kept) and the INTERFACE must be carried by a data-free linear map rewriting 127,488 B. Standing cost over the incumbent: a **12,258,640 B statistic (val) / 12,310,672 B (test)** plus a **127,488 B carry per deletion**.
- 2026-09-20T20:32Z — Wrote PREREG **Amendment 2** BEFORE any test-condition number, declaring: (a) the object that would actually be scored is the spoke's `DowndatableStore` at K=5 on the test leg, ⛔ **NOT** Amendment 1's matched-bytes `K=38/A=7`, which was never built; (b) its honest sum **12,869,482 B = 24.5466x**, so ⛔ rows 3-4 cannot be a matched-bytes claim under any outcome; (c) the fixed deletion procedure (rank-1 subtract -> `pinned_span_basis` -> carry `W <- W M`) with the carry's **127,488 B** cost printed on the row and never quoted beside the hole's 1 byte; (d) the spoke's val numbers recorded explicitly as ITS numbers and not-scored, so a later scored run cannot be presented as having discovered them; (e) expected directions and the "cost not capability" clause verbatim.
- 2026-09-20T20:33Z — Committed `fc322a4`, bundled, `git bundle verify` clean, tree clean. ⛔ **Scored 0/3.** The gate is cleared by my own test-condition build; spending a run is now a live decision rather than a blocked one, and it is a budget decision I am surfacing rather than taking unilaterally.
- 2026-09-20T20:40Z — Peer spoke `w8/downdate` closed out: committed `2fab429`, bundled to `/c/chlu/outputs/w8-downdate/w8-downdate.bundle`, bundle verify clean, tree clean, no processes. Its `DOWNDATE-READY.json` stays `status: ready, condition: val, seed 0` by design — it was scoped to build, never to score. Nothing outstanding between us; I declined its offer to re-run or extend the val-leg stages, having no pending need for them.
- 2026-09-20T20:40Z — ⭐⭐ **CROSS-LEG, CROSS-IMPLEMENTATION REPLICATION OF THE EXACTNESS CLAIM, and it cost no scored budget.** The same claim now stands on two partition legs, from two independently written implementations, in two output trees, with the leg as the only difference:
  | | spoke, val leg | me, test leg |
  |---|---|---|
  | worst max abs dev | 1.164e-10 | **5.249e-12** |
  | in float32 ulp @1.0 | 9.77e-04 | **4.40e-05** |
  | negative control | fired at 1.000e-03 | fired at 1.000e-03 |
  | row-reads at delete(64), two corpus sizes | 320 / 320 | 320 / 320 |
  | object bytes growth with corpus | 0 B | 0 B |
  ⚠ This is replication of the **store-exactness assertions**, not of a scored row. ⛔ Charter §A.5's "unreplicated = not a result" concerns a hub's winning configuration re-run from its PREREG on a clean checkout; that has not happened and cannot until a scored run exists. ⛔ I will not let a replicated assertion be read as a replicated result.
- 2026-09-20T20:40Z — ⭐ The spoke conceded a framing point I raised and fixed it by addendum rather than by editing what it had written: its **0.9005** book-refit leg is not a diagnostic but the control that settles what the collapse was. Read with `downdate == refit-without` at 1.164e-10, it says the post-deletion store is **as good as the pre-deletion one** (0.9005 vs 0.8941), so the deletion damages the content not at all, the whole 0.8941 -> 0.058 collapse was interface staleness, and 0.9005 is simultaneously the score a from-scratch refit without those rows would get — because the two stores ARE the same store.
- 2026-09-20T20:41Z — ⛔ **HOLDING AT SCORED 0/3, and this is the reason, stated where a reader will see it:** the only object that exists is the K=5 one at **24.5466x** the byte pin. Amendment 2 pre-commits, before any test number, that rows 3-4 from it **cannot be a matched-bytes claim under any outcome**. Spending one of three irreplaceable runs on an object that has already conceded the byte comparison is a budget decision with consequences beyond this hub, so I surfaced it rather than took it. ⛔ The alternative — building the matched-bytes object first so the run buys a stronger row — is unstarted. Neither path is blocked; the choice is open and the gate is no longer what is stopping it.
- 2026-09-20T20:48Z — Spoke `w8/downdate` final state, recorded for my report: branch `2fab429`, bundled to `/c/chlu/outputs/w8-downdate/w8-downdate.bundle`, bundle verify clean, tree clean, no processes, `DOWNDATE-READY.json` = `status: ready, condition: val, seed 0`, LOG 614 lines ending in a final report plus three addenda. ⭐ It carried the A.5 boundary into its own artifact against its own wording ("measured on two partition legs, by two implementations, in two output trees" — true, and exactly the sentence a later reader could over-read as a replicated row), and recorded that the gate stopped being my blocker after 16:44Z, so anyone reconstructing why my budget went unspent finds the byte concession rather than a missing dependency. ⛔ Nothing outstanding between the two hubs.
- 2026-09-20T20:48Z — ⭐ One process observation worth keeping, because it is the only part of today that generalises past this object: **every withdrawal on either side came from a measurement, not an argument.** Mine: the permutation advice (withdrawn by the 63.72% ceiling), the fixture well sizes (withdrawn by reading `partitions.npz`), the "gauge rotation" name (withdrawn by my own orthogonality audit failing at 0.912), the float32 statistic (corrected upward by the residuals). The spoke's: an under-sold 0.9005 control, and a `cut_is_marginal` reporting bug. ⛔ Not one of those was settled by who argued better.
- 2026-09-20T20:48Z — ⛔ **FINAL STATE OF THIS HUB, pending a budget decision that is not mine to take.** Scored **0/3**, generation-table run 3 unspent. Branch `w8/single-design` at `fc322a4`, bundled, `git bundle verify` clean, tree clean, no processes, no detached work at any point today. Outputs verified by directory listing: `LOG.md`, `PREREG-w8-single-design-1.md` (+ Amendments 1 and 2), `START_UTC.txt`, `build/test_seed0/{store.pkl, store.json, assertions.json, partitions.npz}`, `dev/{downdate_check, gauge_separation, matched_bytes_recut}.json`, `sandbox/` (my contained publish root), the bundle. Everything needed to spend a scored run is staged; what is NOT decided is whether to spend one on a 24.5466x object that Amendment 2 already concedes cannot carry a matched-bytes row, or to build the matched-bytes object first.

---

- 2026-09-20T20:55Z — ⭐ **DECISION TAKEN (Head): build the matched-bytes object first.** With a clarification that sharpens Amendment 2 rather than merely agreeing with it: **the byte pin is PRIMARY at tolerance 0, so a 24.5466x object cannot produce a RESULT row AT ALL** — not a weak one, not a caveated one. It can only produce DISPLAY rows. ⛔ That retires the option of spending a scored run on the K=5 object entirely, and it means my Amendment 2 was too generous to that object in framing it as "a row with the overrun printed".
- 2026-09-20T20:57Z — ⚠ **A CONVENTION PROBLEM I HAVE TO FIX BEFORE BUILDING, and it moves my window by 2x.** My matched-bytes sweep counted the covariance as a **symmetric triangle** (`d(d+1)/2`). The spoke's honest ledger counts the **FULL square** (`covariance_bytes 12,258,640 = 8 * sum_k d_k^2`, per well `[594441, 883600, 47089, 3600, 3600]` = `771^2, 940^2, ...`). ⛔ The spoke is right **for its object**: its `S` is a full `(d, d)` float64 array, and counting a triangle it does not actually store would be netting by convention — exactly what my target file forbids.
  - ⛔ My `WellStats.S` is also a full array today, so my triangle count has the same defect. I am not going to count a packing I have not implemented.
  - ⭐ **Fix: actually store packed.** The matrix is symmetric, so the triangle is sufficient; I will make the packed triangle the canonical retained state and materialise the full matrix only transiently for `eigh`. Then the triangle count is honest because it is what is deployed.
- 2026-09-20T20:58Z — Recomputed the matched-bytes window under the **four-part** ledger (store + covariance + counts + coupling), so it is comparable with the spoke's. Fixed overhead: counts **11,488 B** + coupling **5,288 + 1,322 B** = **18,098 B**, all independent of K (they depend on the factor cardinalities, not the partition). Available for atoms + covariance: **506,190 B**. Packed triangle at float64 = `4 * (sum_k d_k^2 + D)`.
  | K | d | A max | rank/d | covariance B | atoms B | four-part total |
  |---|---|---|---|---|---|---|
  | 38 | 54 | 6 | 0.111 | 449,698 | 49,152 | 516,948 |
  | **40** | **51** | **9** | **0.176** | **427,622** | **73,728** | **519,448** |
  | 44 | 47 | 14 | 0.300 | 389,492 | 114,688 | 522,278 |
  | 48 | 43 | 18 | 0.419 | 357,717 | 147,456 | 523,271 |
  | 64 | 32 | 28 | 0.875 | 270,336 | 229,376 | 517,810 |
  ⭐ Window is **K ~ 38..56**; ⛔ K >= 64 approaches saturation (`A/d -> 1`) and is excluded a priori, not by selection. Candidates to select **on `val` only**: **K in {40, 44, 48}**. K=40/A=9 sits at rank/d **0.176**, inside the incumbent's own per-well range `[0.0811, 0.0671, 0.2917, 0.2241, 0.6897]`.
- 2026-09-20T21:08Z — Implemented PACKED triangle storage in `WellStats` (`_Sp` + `_iu`; the full matrix is materialised only transiently inside `covariance()` for `eigh`). Verified against a full-matrix reference: packed `S` matches to 1e-9, `covariance()` matches `np.cov` to 1e-10, downdate matches a from-scratch refit to 1e-9. `stat_floats()` now asserts the array and the formula agree, so ⭐ the ledger is answerable to what is stored rather than to a convention.
  - ⚠ Bug found and fixed on the way: my earlier blanket rename `last_gauge_rotation -> last_basis_change` missed the string literal inside `getattr(self, "last_gauge_rotation", None)`, so `basis_change_audit` raised. ⛔ No published number used that path after the rename (the audit was last called BEFORE it, under its old name), so nothing already reported is affected.
- 2026-09-20T21:10Z — ⭐⭐⭐ **THE MATCHED-BYTES OBJECT IS BUILT.** `scripts/w8_build_matched_bytes.py`, condition `val`, label-free `ward_balanced` partition, four-part ledger (atoms + packed float64 covariance + counts 11,488 B + coupling 6,610 B), summed and never netted:
  | build | wells | ranks | atoms B | covariance B | **four-part total** | **x pin** | max abs dev | subspace dist | ranks agree |
  |---|---|---|---|---|---|---|---|---|---|
  | **K=40 A=7** | 51..52 | 6 | 57,344 | 444,352 | **519,794** | **0.9914** | **5.829e-16** | 8.597e-16 | yes |
  | **K=44 A=12** | 46..47 | 11 | 98,304 | 406,272 | **522,674** | **0.9969** | **9.825e-15** | 1.013e-14 | yes |
  | **K=48 A=16** | 42..43 | 15 | 131,072 | 374,528 | **523,698** | **0.9989** | **8.447e-14** | 9.471e-14 | yes |
  | CONTROL ward_bal K=5 A=64 | 409..410 | 63 | 524,288 | 3,380,064 | 3,922,450 | 7.4815 | 4.628e-15 | 1.063e-14 | yes |
  - ⭐ **3 of 4 builds are WITHIN BUDGET and unsaturated, with the downdate exact to 1e-16..1e-14 and no saturated wells.** ⛔ The byte pin is PRIMARY at tolerance 0 and these objects meet it, so unlike the K=5 object they CAN carry a RESULT row.
  - Downdate cost per build: **2.1–2.9 s, 0 rows refitted, 0 data passes.**
  - Basis-change spectrum (`cos_min` / median / frac>0.99): K=40 **0.3408**/0.9999/0.943 · K=44 **0.5602**/0.9999/0.938 · K=48 **0.7119**/1.0000/0.952 · K=5 ctl **0.1837**/0.9999/0.909. ⭐ Finer wells move the span LESS under the same deletion — the deleted content is more localised when the partition is finer.
- 2026-09-20T21:10Z — ⭐ **AN UNEXPECTED FINDING FROM THE K=5 CONTROL, and it is about the incumbent's partition, not mine.** `ward_balanced` at K=5 gives EQUAL wells `[409, 410, ...]` and a four-part ledger of **7.4815x**; the incumbent's `factor` partition at the same K=5 and the same A=64 gives `[771, 940, 217, 60, 60]` and ~**11.8x packed** (24.5x at the spoke's full-square count). The covariance cost is `sum_k d_k^2`, which for fixed `K` and fixed `sum_k d_k` is MINIMISED by equal wells — so the label-MI partition's imbalance alone costs about **1.6x** in covariance bytes, before any question of granularity. ⛔ Not a result; an arithmetic property of the partition, measured.
  - ⚠ It also means the K=5 control is a genuinely useful row: it separates criterion from granularity as pre-registered, AND it shows that even at the incumbent's granularity the label-free equal-well cut is 1.6x cheaper to make downdatable.
- 2026-09-20T21:10Z — ⛔ **K IS NOT YET SELECTED.** Charter A.2: selection on `val` only. The three candidates are built and their ledgers and exactness are settled; choosing among them requires fitting each one's address book and scoring the `val` corners, which is the next block of work and has not been done. ⛔ No scored run has been spent: still **0/3**.

---

- 2026-09-20T21:30Z — ⛔⛔ **DIRECTIVE (Head): HOLD ALL SCORED RUNS.** `downdate`'s exactness is falsified on test seed 2 (A1 fails; A2/A3 pass on every seed). The object definition may change. val/DEV work continues; ⛔ no budget is spent until cleared. **Scored 0/3, generation-table run 3 unspent.** Nothing was in flight — I had not started a scored run and had already flagged the spend as a decision rather than taking it.
- 2026-09-20T21:22Z — ⚠⚠ **A FAILURE THAT WAS MINE, NOT THE OBJECT'S, and it makes two earlier numbers luck rather than evidence.** Running my matched-bytes candidates on the TEST leg, K=40/A=7 and K=44/A=12 came out at **1.826** and **0.6625** max abs atom deviation — 15.3 million and 5.6 million float32 ulp against a <=1 ulp criterion — where my val builds had passed at 1e-16.
  - I proposed **two wrong causes before finding the right one**, and both are recorded because the elimination is the evidence: (i) smallest gap AMONG kept directions — refuted, the predictor said 1.3e-13 while observation was 1.8; (ii) the TRUNCATION BOUNDARY gap `lambda_r` vs `lambda_{r+1}` — refuted, the boundary gap at the worst well was 8.7e-2, large.
  - ⭐ **The actual cause: a missing SIGN CONVENTION in my own `atoms()`.** The subspace deviation was **9.258e-16** while the atoms differed by 1.826 — the signature of `eigh` returning `v` for one path and `-v` for the other. Measured directly: **8 of 240 direction rows sign-flipped**, and allowing a flip took the same comparison to **5.551e-16**.
  - ⛔ The spoke had `_canonical_sign` (`downdate_store.py:162`) from the start and I did not. I have adopted its rule verbatim — sign each row so its largest-magnitude entry is positive, ties by lowest index — so the two objects' bytes stay comparable.
  - ⚠ **Consequence I must state plainly: my val-leg builds passed at 1e-16 only because the signs happened to agree there. They were luck, not correctness.** After the fix, test leg seed 0: K=40 **5.551e-16**, K=44 **1.421e-14**, K=48 **1.490e-13** — all 0.0000 ulp, all PASS.
- 2026-09-20T21:26Z — ⭐⭐ **THE MATCHED-BYTES OBJECT DOES NOT HAVE THE SEED-2 FAILURE.** All three candidates, all three seeds, on the **generation-2** cache (`CHLU_GEN_FEATURES_ROOT` redirect; gen-1 seed 1 holds only 9,288 rows and is unusable, gen-2 has 480,000 for every seed), criterion <= 1 float32 ulp:
  | seed | config | min kept rel gap | observed ulp | verdict |
  |---|---|---|---|---|
  | 0 | ward_bal K=40 A=7 / K=44 A=12 / K=48 A=16 | 1.68e-03 / 4.01e-04 / 8.54e-05 | 0.000 / 0.000 / 0.000 | PASS |
  | 1 | same three | 9.52e-06 / 3.13e-07 / 5.84e-09 | 0.000 / 0.016 / 0.021 | PASS |
  | 2 | same three | 1.35e-03 / 1.94e-10 / 1.01e-08 | 0.000 / 0.006 / **0.094** | PASS |
  | 0,1 | **factor K=5 A=64** | 7.29e-06 / 2.35e-08 | 0.000 / 0.003 | PASS |
  | **2** | **factor K=5 A=64** | 9.12e-08 | **14.377** | ⛔ **FAIL** |
  - ⭐ **9/9 PASS for the matched-bytes candidates; worst observed 0.0938 ulp.** ⛔ The one FAIL is the incumbent's own coarse, imbalanced partition at A=64 on seed 2 — **I reproduce the spoke's seed-2 failure independently**, with my own implementation and my own sign convention (my 14.377 ulp vs its 32.9/35.46 ulp; same seed, same well, different code).
  - ⭐ Plausible mechanism, stated as a hypothesis and not a result: the K=5 object at A=64 cuts **into the degenerate tail** of a rank-deficient well, where eigenvectors are not determined; the matched-bytes builds keep only the top **6–15** directions of 42–52-channel wells, where the spectrum is well separated. ⛔ Not established — it is consistent with the numbers, and one seed is not a mechanism.
- 2026-09-20T21:30Z — ⛔ **MY `spectral_gap_report` IS A FAILED GATE and I am reporting it rather than shipping it.** The spoke validated the same `eps/gap` idea across four legs and found it over-predicts by 7.7x to 217,466x with false alarms on three PASSING legs. My own data reproduces the defect: test seed 2 at K=44/A=12 **predicts 9.603 ulp and observes 0.006 ulp** — a ~1,600x over-prediction and a false alarm. It bounds `||dC||` by `eps * lambda_max` globally while the real residual inside a tiny-eigenvalue subspace is far smaller. ⭐ It does localise the fragile well correctly, so it stays as a diagnostic, labelled useless as a gate.
  - ⭐ The instrument that DOES discriminate is the spoke's **reproducibility floor**: downdate the same rows twice in two different orders, no refit needed. On test seed 2 it measures **4.746e-06 against an observed downdate-vs-refit deviation of 4.227e-06** — the same rows downdated twice disagree by MORE than the downdate disagrees with a from-scratch refit. ⭐⭐ That **exonerates the downdate**: seed 2's atom bytes are not a function of its data by ANY method. ⛔ I had begun implementing this floor for my own object when the hold arrived; the edit was not applied and my module does not yet carry it.
- 2026-09-20T21:40Z — ⭐⭐ **DESIGN CALL (Hub, Head-cleared): the object is the SUBSPACE, with the carry `M = U0^T U1` mapping bases. HOLD LIFTED conditionally** — scored runs resume once `downdate`'s TEST gate passes A1 under the new definition. **Matched-bytes remains primary.** Recording what this decides, what it costs, and one caution, before I build to it.
  - ⭐ **The justification is not that it rescues seed 2.** It is that the read never uses an eigenvector: `x_hat = c_k U_k^T` with `U_k` re-derived from the atoms on load (`gen_read.py:91-106`, zero bytes). Two atom blocks spanning the same subspace give the same read once the book is expressed in the right basis — which is exactly what the carry does, and it is measured, not argued: rendered agreement went from O(1) to ~1e-7 relative, and in the spoke's judge space from 0.0584 to **0.8964 against 0.8941 before deletion**.
  - ⭐ The evidence that the eigenvector was always the wrong unit: the spoke measured its atom block differing from `PCASpanStore`'s by up to **2.0 per well** while each read through its own book rendered the same vector to **1.6e-06** relative. The content is gauge-arbitrary alone; only the `(basis, book)` PAIR carries meaning. My own sign-flip bug is the same lesson from a third direction — 8 of 240 rows flipped, atoms differing by 1.826, subspace identical to **9.258e-16**.
  - ⚠ **CAUTION I OWE THE RECORD, and it is not an objection.** The spoke explicitly declined to make this switch itself: *"I did NOT switch the comparison to the subspace projector so the failure vanishes into a friendlier metric. Both would work. Both are exactly the move I told you at 16:52Z I was not making."* It was right to refuse — a build spoke changing the metric that judges it is the move — and it is right for this to be decided one level up with the reasoning stated. ⛔ But the burden that refusal placed on the decision does not disappear because the decision was taken properly: **the subspace metric must be justified by what the read uses, not by which seed survives.** It is, above. I am recording both halves so a later reader sees the switch was made with the objection in view rather than around it.
  - ⛔ **WHAT THE NEW DEFINITION COSTS, stated plainly:**
    1. **Byte-level exactness claims are no longer available for the content.** If two byte-different atom blocks spanning one subspace are the same object, then `assert_changed_bytes_within` cannot adjudicate a content deletion, and charter B.4's "deletion is an exact operation on **atoms**" is satisfied only in a weaker, subspace sense. ⛔ The coupling hole's "1 byte, 0 outside" is a genuinely byte-level claim about the INTERFACE and is untouched — but it must never be printed beside a subspace-level content claim as though they were the same kind of statement.
    2. **The carry becomes load-bearing, not optional.** Under the old definition the carry was a fix for a gauge problem; under the new one the basis map IS part of the object's definition, so its **127,488 B** per deletion is a standing cost of the design and not an implementation detail.
    3. The reproducibility floor and A1 both have to be restated in the subspace metric, or they are measuring a quantity the object no longer claims.
- 2026-09-20T22:15Z — ⚠ **AN ARTIFACT OF UNCERTAIN PROVENANCE, quarantined rather than used or deleted.** `dev/reproducibility_floor.json` (mtime 21:39) appears to have been written by the script invocation the user INTERRUPTED at ~21:30, i.e. by a code version that was never persisted, and it predates the QR->SVD correction. ⛔ No number in it has been quoted anywhere and none will be. Renamed `UNTRUSTED_reproducibility_floor.json.quarantined` with a README stating why. Deleting an artifact whose provenance is in question is worse than keeping it labelled.
- 2026-09-20T22:16Z — ⛔ **PROCESS NOTE ON MYSELF:** the cross-seed subspace run exceeded the 600 s foreground limit and the harness moved it to the background — so I had detached work running against a standing instruction not to. I polled it to completion inside the turn rather than ending, which is what the rule requires, but the right fix is upstream: I should have sized the job (48 store builds across 3 seeds on the 1.96 GB gen-2 cache, ~75 min) before launching it in the foreground, and used `-u` so its progress was visible rather than fully buffered to a zero-byte file for an hour.
- 2026-09-20T22:20Z — ⭐⭐⭐ **A1 UNDER THE SUBSPACE DEFINITION, ALL THREE SEEDS, TEST LEG, GEN-2** (criterion <= 1 float32 ulp; `ward_bal` = my matched-bytes candidates, `factor K=5 A=64` = the incumbent-grain object the spoke builds):
  | seed | config | atom ulp | **subspace ulp** | two-order floor ulp | verdict |
  |---|---|---|---|---|---|
  | 0 | ward_bal K=40 / K=44 / K=48 | 0.000 | 0.0000 / 0.0000 / 0.0000 | 0.0000 | **PASS** |
  | 1 | ward_bal K=40 / K=44 / K=48 | 0.000 / 0.016 / 0.021 | 0.0000 / 0.0007 / **0.0251** | 0.0000 | **PASS** |
  | 2 | ward_bal K=40 / K=44 / K=48 | 0.000 / 0.006 / 0.094 | 0.0000 / 0.0059 / 0.0047 | 0.0000 | **PASS** |
  | 0, 1 | factor K=5 A=64 | 0.000 / 0.003 | 0.0001 / 0.0050 | 0.0001 / 0.0000 | PASS |
  | **2** | **factor K=5 A=64** | 14.377 | **14.4518** | **36.9305** | ⛔ **FAIL** |
  - ⭐ **matched-bytes: 9/9 PASS, worst subspace 0.0251 ulp** — and they also passed at ATOM level (9/9, worst 0.094 ulp), so ⭐⭐ **the matched-bytes object does not need the definition change to survive. It passes under BOTH definitions.**
  - ⛔ **I independently reproduce the spoke's negative answer:** the subspace definition does NOT rescue `factor K=5 A=64` on seed 2. It fails at **14.4518 ulp**, and the **two-order floor (36.9305 ulp) EXCEEDS the observed deviation** — the same rows downdated twice disagree about the SPAN by more than the downdate disagrees with a from-scratch refit. Seed 2's subspace is not a function of its data. Different implementation, different magnitude from the spoke's (14.45 vs its 60.36 ulp), same structure and same verdict.
  - ⚠ **Basis-correctness check, because the spoke found two bugs in exactly this measurement.** The run above used a **QR** span basis; the spoke showed un-pivoted Householder QR is not rank-revealing and made two BIT-IDENTICAL stores report 0.97 rad. I corrected mine to **SVD** and re-ran the two decisive cells on seed 2: `ward_bal K=48` **0.0047 ulp** and `factor K=5` **14.4518 ulp** — **identical to the QR run to four figures**, with a self-comparison sanity check returning **exactly 0.000e+00**. So my wells were not pathological in the way that triggers the QR defect, and both verdicts stand under either basis. ⭐ I also use the projector distance rather than `arccos` of singular values, which is the spoke's second fix, and had done so from the start.
  - ⭐ **The mechanism was already in my own data before I knew it was the mechanism.** The spoke's diagnosis is that the tightest degeneracy straddles the RANK BOUNDARY (last kept `j=24` at relative eigenvalue 7.54e-12, first dropped `j=25` at 1.12e-16), so the instability is in WHICH directions are in the store, not their orientation — which no subspace definition can absorb. My earlier boundary-gap measurement (21:22Z, taken while chasing the wrong hypothesis for my own sign bug) recorded min boundary gaps of **1.079e-03 / 4.467e-04 / 2.661e-04** for K=40/44/48 — **seven to eight orders of magnitude wider** than the K=5 well's 7.54e-12. ⛔ Consistent, not proven: three seeds and one partition family is not a law.
- 2026-09-20T22:45Z — ⭐⭐ **HOLD LIFTED (Head).** `downdate`'s TEST gate passes A1 under a variance-share rank cut tied to the measured reproducibility floor (39.8 ulp -> 5.684e-14). **Authorised to spend: single-design 3 runs + generation-table run 3. Matched-bytes object; byte pin PRIMARY.**
  - I verified the gate myself rather than on report: `DOWNDATE-READY-test.json` = `status: ready`, `condition: test`, `seeds [0,1,2]`, `all_three_assertions_passed: true`; A1 byte-level `passed_every_seed: true`, worst `1.1255e-09` = **0.0094 ulp**, negative control fired on every seed. ⭐ That is BYTE-level passing, strictly stronger than the subspace definition my release condition named.
  - ⚠ One field I flag rather than pass over: `cut_is_marginal_any_seed: true`, carried as a bare boolean with no per-seed detail in the gate. A marginal cut means a direction sits near the threshold, so a small perturbation could change WHICH directions are kept — the same instability class as the original failure. ⛔ It does not block the gate and I am not treating it as a defect, but it belongs in the caption of anything I score on that object.
  - ⛔ It does not bind on MY object anyway: matched-bytes passed 9/9 across three seeds at BOTH atom and subspace level with no cut at all, at boundary gaps of 1.08e-03..2.66e-04 against the cut's 4.14e-09.
- 2026-09-20T23:22Z — Address book fitted on the matched-bytes object, val, K=40 A=7 seed 0: **945.7 s**, book **174,784 B**. ⭐ **Charter B.3 PASSES**: `assert_read_request_only` reports `feature_argument_present: False`, `bitwise_stable_under_feature_perturbation: True`, 64 checked — "no feature tensor enters launch, settle or read". Span ranks 7 in every one of the 40 wells, sum 280 (incumbent's realised span 308).
  - Ledger: four-part store **519,794 B (0.991x the pin)** + address book **174,784 B** reported beside it, as the programme does for every arm. Params ~129,948 + 43,696 = **~173,644 vs `mlp_180k`'s 179,968 = −3.5%**, inside `PARAM_TOLERANCE = 0.10`.
  - ⚠ `val_mse=nan` throughout the ALS: benign — I did not pass `Y_val`/`targets_val`, so there was no val split to score. ⛔ But the INCUMBENT's fit does use a 90/10 split (`exp_w5_compose_field.py:558-564`), so my fit protocol differs from the object I am comparing against. That is a confound I would have to close before any scored comparison.
- 2026-09-20T23:45Z — ⛔⛔ **THE MATCHED-BYTES OBJECT DOES NOT COMPOSE. val corners, shared frozen decoder and judge, same code path as the spoke's smoke:**
  | arm | valid mean | sd | min | max | corners |
  |---|---|---|---|---|---|
  | **cov_tensor matched-bytes K=40 A=7** | **0.0608** | 0.0604 | 0.0000 | 0.2222 | 24 |
  | `pca_tensor` / `downdate_store` (spoke, same corners, same judge, same seed) | **0.8941** | 0.0903 | 0.7361 | 1.0000 | 24 |
  - ⛔ This is a **collapse, not a shortfall** — 0.06 against 0.89. ⭐ It cost **no scored budget**: this is exactly what charter A.2's val condition is for, and it did its job. ⛔ **I will not spend a scored run on this object.**
  - ⭐ **Leading hypothesis, and it is the confound I pre-registered in Amendment 1 §A1.4:** the matched-bytes route had to abandon the factor-aligned criterion (`factor_aligned_balanced_labels` refuses any `k > 6`) for the **label-free `ward_balanced`** correlation cut. But the read is a PRODUCT address over factors, `a(request) = prod_f w_f(v_f)^T E_f`, and the incumbent's wells are FACTOR-ALIGNED. Correlation-grouped wells scramble factor structure across 40 wells, so the product address has nothing factor-shaped to address. If that is right, the label-freeness I called "strictly more label-free than the incumbent" is not a bonus — it is the thing that breaks it.
  - ⛔ Not established. Criterion and granularity moved together, which is precisely why the control was pre-registered. Running `ward_balanced` at **K=5, A=64** now — my criterion at the incumbent's granularity. If it also collapses, the CRITERION is the cause; if it reads near 0.89, GRANULARITY is.
- 2026-09-21T00:20Z — ⭐⭐⭐ **THE CONTROL SETTLES IT, AND IT REFUTES MY OWN HYPOTHESIS.** Same corners, same frozen decoder and judge, same code path, same seed:
  | arm | criterion | K | A | span | four-part ledger | **valid mean** |
  |---|---|---|---|---|---|---|
  | `pca_tensor` / spoke's store | factor (label-MI) | 5 | 64 | 308 | 24.5x | **0.8941** |
  | **ward_balanced control** | **label-free** | **5** | **64** | 315 | 7.48x | **0.8935** |
  | **matched-bytes** | **label-free** | **40** | **7** | 280 | **0.991x** | **0.0608** |
  - ⛔ **My leading hypothesis was WRONG.** I guessed the label-free `ward_balanced` criterion scrambled factor structure and broke the product address. It does not: at the incumbent's granularity the label-free cut reads **0.8935 against 0.8941** — statistically indistinguishable, and ⭐ it does so **without using labels in the partition at all**, so the criterion is not merely harmless, it is a free win on charter B.6.
  - ⭐⭐ **GRANULARITY is what breaks composition, not the criterion.** K=40 wells of ~51 channels at rank 7 collapse to 0.0608 while K=5 wells of ~410 channels at rank 64 hold 0.8935. The pre-registered control (Amendment 1 §A1.4) did exactly the job it was written for, and it separated a confound I would otherwise have attributed to the wrong cause.
- 2026-09-21T00:22Z — ⭐⭐⭐ **THE WAVE'S ANSWER, and it is the negative one my task file named in advance.** Chaining the measurements:
  1. Exact per-item deletion requires retaining the per-well covariance — that is the whole single design, and it WORKS: downdate == refit to 1e-16..1e-14, no data pass, items never kept, 9/9 across three seeds.
  2. The covariance costs `sum_k d_k^2`, so fitting it inside the 524,288 B pin **forces** K >= 36 (below that the statistic alone exceeds the budget with zero atoms).
  3. At K >= 36 composition **collapses** (0.0608 vs 0.8941).
  4. At K = 5, composition holds (0.8935) but the honest ledger is **7.48x** (label-free) or **24.5x** (the spoke's, factor partition, full-square covariance) — and the byte pin is PRIMARY at tolerance 0, so ⛔ that object cannot carry a RESULT row at all.
  ⭐ **So the two properties are incompatible AT MATCHED BYTES — not because the criterion is wrong, not because the deletion mechanism fails (it is exact), but because the byte budget forces a granularity at which the read stops composing.** ⛔ That is my task file's declared falsification, arriving as the wave's answer: *"If the single design loses rows 3-4 to `pca_tensor` the properties are still incompatible and that is the wave's answer even though it is not the hoped one."*
  - ⭐ It cost **ZERO scored budget**. Charter A.2's val condition is exactly the instrument that produced it, and it produced it before any irreplaceable run was spent.
  - ⚠ Scope, stated: this is `val`, seed 0, 24 corners, one arm per configuration. It is NOT the >=49 paired-corner estimand and it is NOT a RESULT row. It is strong enough to decide where budget should go and not strong enough to be quoted as a scored finding.
- 2026-09-21T00:30Z — Background diagnostic launched before the hold-lift completed; recording it rather than discarding it, though it is now a footnote. **Does the spoke's declared cut `tau = 4.144858e-09` bind on the matched-bytes object? NO — zero directions dropped, every candidate, every seed it reached.** Smallest kept relative eigenvalue: seed 0 `1.93e-02 / 4.51e-03 / 3.08e-03` (K=40/44/48), seed 1 `1.42e-04 / 1.10e-06 / 1.46e-08`, seed 2 `5.31e-03` (K=40; the run reached that far). ⭐ Every one is above `tau`, most by 3–6 orders. So the two cuts are compatible: adopting the spoke's declared threshold alongside my `EIG_REL_TOL = 1e-12` would change nothing about my object, and the matched-bytes candidates' exactness never depended on a cut at all.
  - ⛔ This is now a footnote because the matched-bytes object is falsified on COMPOSITION (0.0608), not on exactness. Its exactness was never the problem.
  - ⚠ Process, recorded against myself: this job and one before it exceeded the 600 s foreground limit and were moved to the background, so I twice had detached work running under a standing instruction not to. Both were polled to completion inside a turn and neither was stranded, but the fault is sizing the work before launching it, and I repeated it after logging it once.
- 2026-09-21T01:15Z — ⭐⭐ **THE ALS CONFOUND IS CLOSED, and the falsification survives it.** Refitted BOTH arms under the incumbent's protocol exactly — `Y[:ntr]`/`T[:ntr]` with `ntr = int(0.9*N)`, `Y_val`/`targets_val` on the last 10%, and `R.READ_PINS` for every hyperparameter (`rank 128, ridge 0.01, sweeps 40, address_seed 0`), matching `exp_w5_compose_field.py:558-564` and the spoke's `_fit_address` line for line: **117,964 train / 13,108 val.**
  | arm | 100% fit (first pass) | **90/10, incumbent's protocol** | move |
  |---|---|---|---|
  | matched-bytes K=40 A=7 | 0.0608 | **0.0648** | +0.0040 |
  | ward_balanced control K=5 A=64 | 0.8935 | **0.8918** | −0.0017 |
  | `pca_tensor` / spoke's store (reference) | — | 0.8941 | — |
  - ⭐ The protocol moved both arms by **< 0.005**, in opposite directions, and the gap is unchanged: **0.0648 vs 0.8918, a factor of ~14.** ⛔ The falsification does not rest on a fit-protocol difference.
  - ⚠ Worth recording WHY I did this rather than argue it away: my hyperparameters already matched, so the only difference was the split, and the split gave my arms ~11% MORE training data — i.e. the confound ran in my favour and could not have manufactured a collapse. That is a sound argument and it is not a measurement. The number this hub's answer turns on should not rest on an argument, so it was measured. ⭐ The Head was right to require this before the report.
  - Address books under the matched protocol: K=40 **174,784 B** (fit 587.5 s), K=5 **195,584 B** (fit 771.6 s) — the latter identical to the incumbent's book size, as it must be at the same K and rank.

---

# ⭐ FINAL REPORT (REVISED) — w8 hub/single-design

⛔ **This supersedes the interim report written at 19:14Z**, which was written under a gate
that had not opened and before the object was built, read or falsified. That section is left
standing above, unedited, per the programme's log discipline; where the two disagree, this one
is correct.

**Target:** one object that composes AND forgets.
**Window:** 2026-09-20 16:36Z → 2026-09-21 01:20Z (8 h 44 m). Local CPU, 3 threads. No GPU.
**Base** `base/iter7 = ec2ef7c`. Frozen harness untouched. Branch `w8/single-design`, bundled and verified.

## 1. Scored runs — ALL of them

| run | status | reason |
|---|---|---|
| single-design 1/3 | ⛔ **NOT RUN** | the object was falsified on `val` before any scored run was justified |
| single-design 2/3 | ⛔ **NOT RUN** | — |
| single-design 3/3 | ⛔ **NOT RUN** | — |
| generation-table run 3 | ⛔ **NOT RUN** | — |

⛔ **Scored 0/3. No row was written to `/c/chlu/SCORED-LEDGER.md`.** Nothing below is a held-out
number; every measurement is on the fit pool or the `val` selection condition. ⛔ A declared
NOT-RUN is never a null. The full budget survives intact.

⭐ **This is the outcome charter A.2's val condition exists to produce**: the design question was
settled before an irreplaceable run was spent on it.

## 2. ⭐⭐ THE ANSWER

**The two properties are incompatible at matched bytes.** Not because the deletion mechanism
fails — it works exactly — and not because of the partition criterion, which turns out to be free.
Because the byte budget forces a granularity at which the read stops composing.

The chain, each link measured:

1. **Exact per-item deletion works.** Retain `S_k = Σ x xᵀ`, `t_k = Σ x`, `n_k`; derive atoms as the
   leading eigenvectors; delete by rank-1 subtract. Downdate == refit-without-those-rows to
   **1e-16..1e-14**, ranks agreeing in every well, **0 rows refitted, 0 passes over the data,
   items never kept**, **9/9 across three seeds** on generation-2. Independently corroborated by
   the spoke on its own leg (1.164e-10) with different code.
2. **The covariance costs `Σ_k d_k²`.** Fitting it inside the 524,288 B pin **forces K ≥ 36** —
   below that the statistic alone exceeds the budget with zero atoms left for content.
3. **At K ≥ 36 composition collapses.** K=40/A=7, honest four-part ledger **519,794 B = 0.991×**
   the pin: **valid mean 0.0648** on the val corners.
4. **At K = 5 composition holds** — **0.8918**, against the incumbent's 0.8941 — **but the honest
   ledger is 7.48×** (label-free cut) or **24.5×** (the spoke's factor partition, full-square
   covariance). The byte pin is PRIMARY at tolerance 0, so ⛔ that object cannot carry a RESULT row.

⭐ This is the falsification this hub's task file named in advance: *"If the single design loses
rows 3-4 to `pca_tensor` the properties are still incompatible and that is the wave's answer even
though it is not the hoped one."*

## 3. ⭐ UNSCORED VAL OBSERVATION — the ward_balanced K=5 object

⛔ **Explicitly NOT a RESULT row.** `val` selection condition, seed 0, 24 corners, one arm —
not the ≥49 paired-corner estimand, no learned-baseline comparison, no scored budget spent.

| property | measured |
|---|---|
| composition (val corners, shared frozen decoder + judge) | **0.8918** sd 0.1007, vs incumbent **0.8941** |
| exact deletion | downdate == refit, **0 data passes, 0 rows refitted, items never kept** |
| labels in the partition | ⭐ **NONE** — `ward_balanced` is a label-free correlation-dendrogram cut |
| honest four-part ledger | **3,922,450 B = 7.48× the pin** ⛔ over budget, printed not netted |

⭐ **It answers the wave's question in the only form available: both properties DO coexist in one
object — at 7.5× the byte budget.** ⛔ It is an observation, not a claim; it is quoted with its
overrun attached or not at all.

## 4. Measurements that stand on their own

**4.1 The gauge, and the carry.** The deletion changes the atoms, and `gen_read.span_basis`
re-derives the basis from them, so an address book fitted before a deletion is stale after it —
rendered reads differ by **O(1)** between a downdated store and a refit-without store whose
projectors agree to **2e-07**. ⛔ Scoring post-deletion composition naively would have produced a
collapse that looks exactly like the w5 eviction failure and has nothing to do with it.
The fix is to carry the **book**, not the atoms: `W_k ← W_k M_k`, `M_k = U_kᵀ U'_k`, formed at
delete time, **no data row read, zero deployed bytes**. The spoke measured it in judge space:
**0.0584 → 0.8964 against 0.8941 before deletion — 100.3% recovery with no refit of anything.**
⛔ Its cost, printed rather than inferred: it rewrites **127,488 B of the address book's 158,664 B**,
and is **never** quoted beside the coupling hole's 1 byte.

**4.2 Target/size separation at the store level.** `M_k`'s principal-angle spectrum under the red
deletion: **cos_min 0.1999, median 0.9999, 93.8 % of directions above 0.99**. Under a
**size-matched random** deletion of the same 11,044 rows: **cos_min 0.9976, nothing moves.**
⭐ The inverse of w5's eviction verdict ("the cost is the mechanism, not the target").
⛔ A property of the store, **not** a composition number, and not quoted as one.

**4.3 Granularity, not criterion.** The pre-registered control (Amendment 1 §A1.4) separated a
confound I would otherwise have mis-attributed: at the incumbent's granularity the **label-free**
cut reads **0.8918 vs 0.8941** — so the criterion is not merely harmless, it is a free win on
charter B.6. ⛔ **My own leading hypothesis — that the label-free criterion scrambled the factor
structure the product address needs — was wrong.**

**4.4 About the incumbent.** `pca_tensor` is **already saturated in 2 of its 5 wells**
(`d_k = 58 < A = 64`; effective ranks 12 and 40 after dead channels), so its realised deployed span
is **308, not 320**, and `pca_random_refit` cannot collapse there. 5.7 % of the atom bytes — it does
not overturn the incumbent, but `rank/d_k` belongs on every row.

## 5. ⚠ Errors I made, and how each was caught

⛔ Every one was caught by a measurement or a control, not by an argument.

1. **Test-fixture well sizes.** I took `[303,305,573,501,366]` from a unit test and built a byte
   argument on it. Real scored-leg sizes are `[777,939,216,58,58]`. Caught by the spoke; every
   derived figure was wrong by ~3×. **K=22/A=15 and a param-match claim were withdrawn.**
2. **A ledger that netted by dtype.** I counted the covariance as a packed triangle while holding a
   full `(d,d)` array. Fixed by actually implementing packed storage — and corrected **upward**,
   against my own interest, from float32 to float64.
3. **A missing sign convention.** Atoms differed by **1.826** between downdate and refit while the
   subspace agreed to **9.258e-16** — 8 of 240 rows sign-flipped. ⚠ My val builds had passed at
   1e-16 **by luck**. Adopted the spoke's `_canonical_sign`.
4. **Two wrong causes proposed before that one** (gap among kept directions; truncation-boundary
   gap), each refuted by measurement. Recorded, because the elimination is the evidence.
5. **"Gauge rotation" was the wrong name** for `M_k`, and **my own orthogonality control proved it**
   (0.912 deviation). Renamed; the audit now reports the spectrum instead of asserting it away.
6. **A negative control that did not bind** (one-entry nudge, 8.7e-06 vs a 1e-04 threshold). Kept
   in the artifact labelled as such, and replaced by one that fires at **1.958** — separation 5.25e8×.
7. **A predictor that did not work.** `spectral_gap_report` over-predicts (9.603 ulp predicted,
   0.006 observed) and false-alarms on passing legs. Kept as a diagnostic, **labelled useless as a gate**.
8. **⛔ I overwrote the peer spoke's `DOWNDATE-READY.json`** by running its module without reading
   its publish path. For ~90 s the file read `condition: test` beside `all_three_assertions_passed:
   true` when those assertions ran on val. Restored from its own artifact, incident recorded in the
   file, pre-restore copy kept as `.bak`, and told the spoke rather than repairing quietly.
9. **QR where SVD was required** in the subspace metric. Corrected before quoting; re-verified
   identical to four figures with a self-comparison of exactly 0.
10. **Twice let a job exceed the foreground limit** and be backgrounded under a standing instruction
    not to. Both polled to completion inside a turn; neither stranded. I repeated it after logging it once.

## 6. Charter §C floor checks

- **C.1 matched-bytes lookup** — ⛔ NOT RUN (no scored run).
- **C.2 landscape-removed / shuffled** — ⛔ NOT RUN. ⚠ `pca_random_refit` cannot collapse in the
  incumbent's two saturated wells; `rank/d_k` printed.
- **C.3 tie clause** — not reached.
- **C.4 distance-to-nearest-atom** — ⛔ NOT RUN.
- **Run instead, at store level:** the deletion's own size-matched random control (§4.2), and
  charter **B.3** — `assert_read_request_only` passes on both arms: no feature argument, bitwise
  stable under feature perturbation, 64 checked.

## 7. Cost line

8 h 44 m wall clock, one machine, local CPU, 3 threads. **No GPU. No cluster leg** (CSF3 unreachable
— ⛔ NOT RUN, charter §D). 3 read-only sub-agents for code mapping; none running at close.
Compute: ~60 covariance store builds across 3 seeds and 4 partitions, 4 address-book ALS fits
(587–946 s each), 4 corner-scoring passes through the shared frozen decoder and judge.
⛔ **Zero scored budget consumed.**

## 8. What the next hub should do

1. ⛔ **Do not re-run the matched-bytes route as specified.** 0.0648 vs 0.8918 is a collapse, and it
   survives the incumbent's exact fit protocol. The granularity that makes the covariance affordable
   is the granularity that kills the read.
2. ⭐ **The label-free `ward_balanced` partition is a free win and should be inherited** — 0.8918 vs
   0.8941 at the incumbent's granularity, with **no labels in the partition at all**. It strengthens
   charter B.6 and costs nothing. It is also **1.6× cheaper** to make downdatable than the label-MI
   cut, because `Σ d_k²` is minimised by equal wells.
3. ⭐ **The carry (§4.1) is load-bearing** and must be applied before any post-deletion read, or the
   row reads as a collapse that is pure coordinate staleness.
4. **The open question this hub could not close:** whether a content whose exact deletion does NOT
   require a `d×d` statistic exists. Every route here paid `Σ d_k²`, and that single term is what
   forces the fatal granularity. A sketch with a deletion guarantee, or a different unit of content,
   is where the remaining headroom is.
5. ⚠ Not fixed, recorded so it is not rediscovered: `exp_gen_field_store.sweep` (:930) references
   `store_seed` while its parameter is `seed` (:884) — `--stage sweep` raises `NameError`.

## 9. The one-sentence answer

⭐ **One object can both compose and forget exactly — the covariance route does it, measured, with no
pass over the data and items never kept — but not within 524,288 B: the statistic that buys exact
deletion costs `Σ_k d_k²`, affording it forces wells fine enough that composition collapses from
0.89 to 0.06, and so at matched bytes the two properties remain incompatible.**
