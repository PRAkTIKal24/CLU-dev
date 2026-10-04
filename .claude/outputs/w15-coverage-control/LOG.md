# w15 coverage-control LOG
- START (see START_UTC.txt). Env: env_w3.sh PYTHONPATH is mixed-form (C:/..:/c/..) and git-bash did not convert it -> equinox not importable. Fix: export PYTHONPATH="C:/chlu/worktrees/CHLU-w15-coverage-control;C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.venv/Lib/site-packages". Assert OK: chlu from worktree, jax 0.9.0 numpy 2.4.1 eqx 0.13.4. No venv built.

## Ledger verification (before any number) — `/c/chlu/SCORED-LEDGER.md`
- Binding rule (ledger header + w3-hub-common §5): "Three scored runs per TARGET ... counted from 2026-09-16".
- `composition` rows since 2026-09-16: L8 wave-2 (1/3, spent) · L11/L12 hub/factor-wells run 2/3 (one run, clock-corrected) · L15 hub/factor-wells 3/3 · L27 w5 compose-field (labelled "1/3") · L30/L32 w6 compose-physics (labelled "2/3 SPENT; 3/3 BANKED").
- => 5 composition spends against a cap of 3. The "2/3, 3/3 banked" label comes from a count restarted at w5 that no written rule authorises. Under the header rule the target is EXHAUSTED (and already over-drawn).
- DECISION: composition is DECLARED EXHAUSTED. No scored run is spent. Val leg only (task file's own instruction). A NOTE row (0 runs spent) is appended to the ledger so the verification is visible.

## Instrument reading
- Ladder = chlu/data/gen_kway_ladder.py via scripts/w12b_ladder_leg.py (condition swap) + scripts/w12b_ladder_kstar.py (the main-text K*=32 CovContentStore on ward_balanced). Estimand distinct_valid_novel_tuples per corner (fw.per_corner_scores), FastJudge asserted == Judge.
- Test rungs n=3 and n=6 probe the identical 120 corners (8,640 rows); train_3 = 180,000 rows ⊂ train_6 = 279,360 rows (checked: train3 subset of train6 = True).
- fit_pool = rng(pool_rng_seed=20260919).permutation(train)[:131072]; pool_rng_seed is the SAME for every seed. ⇒ ⭐ "n=6 with pool restricted to n=3's train index" draws EXACTLY n=3's pool ⇒ its store+address book are bit-identical to n=3's. The only thing that then separates it from n=3 is the ESTIMAND's train-derived masks (seen_tuple_mask, value_presence_bitmap from n=6's train index). So the control decomposes the n=3→n=6 rise into STORE-COVERAGE (B−C) and SCORING-MASK (C−A) parts.
- No val geometry exists for the ladder: every coordinate option (object_hue 2 bands, shape 3, scale 2 corners, orientation 2 corners, floorwall 5) is used by the full sweep, and the existing fw.val_corners (Montero slots 49..72) fall INSIDE the k=6 test cells.

## CHOICE: the val ladder (recorded, ICLR-closest)
- Build a VAL copy of rungs 3 and 6 with the scale coordinate moved to the one band no test cell touches: the MIDDLE scale band (0.964, 1.036; width 2 — the only width available). Same hue/shape/floorwall/orientation options, same nesting, same builder.
- val k=3 cells: 2 hue x 3 shape x 1 scale = 6 cells x 6,000 rows = 36,000 rows. val k=6 cells: 60 (1:1 corners). Corners: 60 x 48 rows (vs test 120 x 72) — ≥49 bar met; geometry differs only in scale-band width (2 vs 3), DECLARED.
- val rung-n condition: train = test-rung-n train \ val-rung-n cells; held-out = test rows ∪ val cells. Val corner rows are asserted to be in the TRAIN index of both test rungs (never a test row).
- train_3val = 144,000 ≥ 131,072 (pool size pin holds); train_6val = 276,480.
- Arms per seed (seeds 0,1,2): A = n=3 (own pool, n=3 estimand) · B = n=6 (own pool, n=6 estimand) · C = n=6 estimand, pool from n=3's train (store = A's, asserted pool-identical) · D (diagnostic, completes the 2x2) = n=3 estimand read by B's store.
- Store: K*=32 CovContentStore / ward_balanced, the main-text object; controls landscape_removed / landscape_shuffled / random_atoms (post-fit form, as w12b); learned_1x at 1x the K* four-part total. pca_tensor / K=5 DECLARED NOT-RUN (not the main-text object).
- 07:13:31Z launched val legs seeds 0,1,2 (THREADS=1 each); PIDs in PIDS.txt
- 07:24:27Z all 3 legs: pool recompute == w12 test pools {3:True,6:True}; FastJudge identical True; pool3 stores built (rank/width 0.95-0.98); address books fitting (~37 min each). Wrote scripts/w15_coverage_report.py (aggregator, PREREG bars verbatim).
- 07:54:21Z still fitting pool3 address books (CPU-contended box: ~0.7 core/leg)
- 08:24:16Z FAILED launch 1: all cells 0.0000/judge_acc 0 — val width-2 middle-scale cells + test k=3 cells exhaust the ladder hues from train_3val → acceptance refuses all. Killed my PIDs 30220/33864/12732 only. AMENDMENT-1: width-1 (scale idx 3) + no-absent-value assertion; dry run accept 1.0. Relaunched seeds 0,1,2.
- 08:44:09Z amend-1 legs: pool3 stores built, address books fitting
- 09:24:27Z still fitting pool3 books (box heavily shared: ~14 python procs)
- 09:54:14Z CPU contention: each leg ~0.2 core (880 CPU-s in ~70 min); book needs ~3200 CPU-s => expect hours per book. Continuing; no design change.
- 10:18:36Z pool3 books fitted (~6500s each). A (val n=3): s0 21.7667, s1 23.0500, s2 23.4833 (ceiling 24/corner); controls 0.0000. ⚠ val is near ceiling — noted before B/C seen.
- 10:27:40Z ⭐ C == A to 4 dp on all 3 seeds (store AND learned_1x). IDENTITY-CHECK.txt: (1) pool(train_3) == w12 rung-3 pool; (2) TEST acceptance 1.000000 under both bitmaps for both rungs' corners; (3) distinct_valid == distinct_valid_novel in 1800/1800 val rows and 3600/3600 published TEST K* rung-3/6 rows (judge_valid demands all 6 factors == request, so a valid tuple IS a held-out corner tuple, novel under either mask); (4) TEST rung-3/6 corner rows identical (8640). ⇒ on the TEST ladder "n=6 at n=3's coverage" == rung n=3 BY CONSTRUCTION; C_test is DERIVED = A_test, not measured. The mask channel is identically zero; the whole n=3→6 rise can only be store pool coverage.
- 10:57:51Z pool6 stores built; books fitting (expect ~1.8h)
- 11:27:37Z polling pool6 books
- 11:57:17Z polling pool6 books
- 12:13:29Z all 3 val legs done, failed=[] each; pool recompute True/True and FastJudge True on every seed. Report: COVERAGE-CONTROL.json. Val: A 22.767, C 22.767 (=A every seed), B 23.478; B-A +1.200/+0.567/+0.367 → PREREG verdict NO RISE TO DECOMPOSE ON VAL; mask C-A = 0.000 every seed; controls 0.0000 every cell. Test (published) A 61.614, B 66.603; C_test DERIVED = A_test by identity (IDENTITY-CHECK.txt). FINDINGS.md written.
- END 2026-09-23T12:13:32Z
