# PREREG — w15 spoke/bar-d-score: BAR D on the w8 GPU leg (full decoder)

**Written:** 2026-09-23T07:11:36Z, BEFORE any held-out number from the GPU leg exists (none has ever been computed:
`rivals`, `score` and the cross-arm report on the GPU leg are a DECLARED NOT-RUN in
`/c/chlu/evidence/outputs/w8-BAR-D-fold.md` §2/§4).
**Target:** `encoder-loop`. **Spend:** 1 scored run (encoder-loop 2/3; 1/3 was the w8 CPU leg).
**Branch/base:** `w15/bar-d-score` off `base/iter8 = 148f3b7`.
**Charter:** A.1 · A.2 (selection on val only) · A.3 (bars never moved; amendments appended, dated).

## 0. ⛔⛔ BUDGET PINS — this table is its own instrument
| pin | **THIS PREREG (GPU leg)** | w8 CPU leg (PREREG-w8-encoder-loop-1) |
|---|---|---|
| `joint_steps` | **4000** | 400 |
| `n_train` | **32768** | 8192 |
| `dec_refit_steps` (`w8_v1_short` recipe) | **1800** (the gated step count) | 400 |
| `warm_steps` | **4000** | 1500 |
| images_seen per arm | 256,000 | 25,600 |
Source of the GPU pins: the 46 GPU manifests at `/c/chlu/evidence/outputs/w8-encoder-loop/cloud/manifests/`
(`seed_*/<arm>/arm.json` → `compute.gradient_steps`, `n_train_rows`, `decoder_refit.recipe.steps`).
⛔ **Every number scored under this PREREG NEVER SHARES A TABLE with the w8 CPU leg.** The module's own
rule (`exp_w8_encoder_loop.apply_compute_overrides`): *"a run at one budget and a run at another do not
share a table."* CPU numbers may be printed in a SEPARATE block, with their pins, for the direction
comparison in §4.2 only. No row, mean or pooled statistic combines them.

## 1. The question
**BAR D — do the baselines close the gap?** Within each encoder arm X, M(X) = `pca_tensor@X − rival_mlp_180k@X`
(the baseline REFIT on X's own features, `w4_learners` verbatim, 178,952 params, `dropout=False`).
Does M GROW from `alone` to `loop_swap`? If not, the features are better (or worse) **for everyone** —
a finding about the ENCODER, ⛔ not about the store.
Why the GPU leg: on the CPU leg the 400-step decoder zeroed `rival_mlp_180k` on every seed and row
(`/c/chlu/outputs/w8-encoder-loop/LOG.md` Addendum A: 0.000 ×6), so the CPU "margin" was `pca_tensor`'s
absolute and **CPU BAR D is VOID, not a result**. The 1800-step decoder is what makes BAR D answerable.

## 2. Pipeline (unmodified code; only the run-base and pins move)
Per seed S ∈ {0,1,2}, encoder arms {alone, loop_nosw, loop_swap} from the GPU leg's own trained weights:
1. `exp_w8_encoder_loop --stage cache` ON THIS BOX (one-CPU-device guard; 480,000 rows, gen-2).
   ⭐ Cross-machine check: regenerated block_sha256 compared against the Mac's `cache.json` in the manifests;
   agreement or disagreement is REPORTED, and a mismatch does not void the run (the cache is built from the
   hash-asserted encoder on this box's CPU, which is the instrument that scores).
2. `--stage rivals --families mlp_180k` on `alone` and `loop_swap` (the two BAR D arms), at the GPU pins.
3. `scripts/w8_encoder_loop_score.sh` (the unmodified w5 leg) via `scripts/w15_bar_d_score.sh`, which adds
   THIS spoke's own PREREG + ledger guard. Score arms:
   - `alone`, `loop_swap`: `pca_tensor rival_mlp_180k pca_landscape_removed pca_landscape_shuffled pca_random_refit pca_lookup_argmax`
   - `loop_nosw`: `pca_tensor` (the decomposition column only; as in the CPU leg's declared scope)
4. `scripts/w8_encoder_loop_report.py` → CROSS-ARM.json; `w15_bar_d_report.py` → BAR-D.json.
The judge (external classifier, per-seed, hash-asserted) DOES NOT MOVE across arms.

## 3. Estimand
Rows 4 (`r4_comp_nway`, 49 corners, ceiling 72) and 3 (`r3_comp_2way`, 60 sub-corners, ceiling 36).
Metric `distinct_valid_novel_tuples` per corner; `fw.paired_estimand` (corner = unit, sign test, 10,000-draw
paired bootstrap CI95); margin `DECORATION_MARGIN_NOVELTY = 1.0`. Seeds 0/1/2, each reported, never pooled.
**Growth statistic (new, declared now):** per corner c, g_c = [P_swap(c) − R_swap(c)] − [P_alone(c) − R_alone(c)];
growth = mean_c g_c (identical to M(loop_swap) − M(alone)); CI95 by a 10,000-draw corner bootstrap, rng seed 20260923.

## 4. ⭐ THE BARS — declared now, never moved
### 4.0 VAL GATE (free; decides whether the scored run is spent)
On `val`, seed 0, arms `alone` and `loop_swap`:
- V1 **C.2 collapses**: `pca_landscape_removed` mean ≤ 1.0 novel tuples/corner on row 4, both arms.
- V2 **the baseline is not degenerate**: `rival_mlp_180k` mean > 0 on row 4 in at least one of the two arms.
  (If V2 fails, BAR D would be VOID exactly as on CPU; the test run is **not spent** and that is DECLARED.)
- V3 the arm renders: `pca_tensor` row-4 mean > `pca_landscape_removed` + 1.0 in both arms.
Any failure → the test run is DECLARED NOT-RUN with the val numbers printed.

### 4.1 BAR D-GPU (PRIMARY), row 4; row 3 reported beside it
- **GROWS** (store-shaped representation that the store exploits): growth > 0 on **all three** seeds AND
  growth CI95 low > 0 on **≥ 2 of 3** seeds.
- **DOES NOT GROW** (a finding about the ENCODER, not the store): growth ≤ 0 on **≥ 2 of 3** seeds.
- **MIXED** otherwise: reported as such; ⛔ no claim either way.
- **VOID on a seed** if `rival_mlp_180k` reads exactly 0.000 on row 4 in both arms of that seed (the CPU failure
  mode); a VOID seed counts toward neither GROWS nor DOES-NOT-GROW, and BAR D-GPU needs 3 non-void seeds to GROW.
- Charter C: any store claim additionally requires `pca_landscape_removed`, `pca_landscape_shuffled`,
  `pca_random_refit` to COLLAPSE (row-4 mean ≤ 1.0) on the scored `alone` and `loop_swap` legs. If they do
  not, the result is an **interface** claim or **nothing**, and it is said which.
  `pca_lookup_argmax` is a printed FLOOR, never a rival.

### 4.2 ORDERING (SECONDARY; the decomposition paragraph "unfreezing hurts / swap loss repairs")
A′ = `pca_tensor@loop_nosw − @alone`, A″ = `pca_tensor@loop_swap − @loop_nosw`, 3 seeds × 2 rows = 6 cells.
CPU leg (400/8192, `/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json`): A′ < 0 in 5/6 cells, A″ > 0 in 4/6.
- **REPRODUCED** on GPU if A′ < 0 in ≥ 4/6 AND A″ > 0 in ≥ 4/6.
- **NOT REPRODUCED** otherwise — and ⭐ that is the finding, printed with both sets of pins, never merged.
Also printed (secondary, never the headline): BAR A (`loop_swap − alone`) per seed per row under the w8 rule.

## 5. What would prove this wrong / what is out of scope
- BAR D DOES NOT GROW ⇒ "the store in the loop shapes a representation the store itself exploits" is FALSE at
  these pins. A MIXED or VOID result is not evidence for it.
- ⛔ NOT in scope, DECLARED NOT-RUN: `frozen` and `alone_wc` scoring (w8's declared scope; and BAR C.2 already
  FAILS on the GPU leg, N322, so no equal-seconds claim is possible); BAR F cross-judge; the w8 single design;
  `landscape_shuffled_across_wells` (not implemented for `pca_tensor` in `exp_w5_compose_field.PCA_ARMS`);
  rivals other than `mlp_180k`.
- Deviation inherited from w8 PREREG §3: decoder recipe is `w8_v1_short` by name, here at 1800 steps.
  Absolutes do not share a table with the incumbent's either.

---
## AMENDMENT 1 — 2026-09-23T11:32:33Z (appended, never an edit; charter A.3). Written after the FREE val gate, BEFORE any test number.

**What happened.** Val gate seed 0 (`VAL-GATE` inputs: `score/{alone,loop_swap}/val_seed0/READ-TABLE.json`):
`pca_tensor` 40.958 / 71.750, `pca_landscape_removed` 0.000 / 0.000, `rival_mlp_180k` **0.000 / 0.000**
(24 val n-way corners). V1 PASS, V3 PASS, **V2 FAIL**.
**Diagnosis (measured, not guessed).** `exp_gen_read.py` loaded the rival MODEL with `load_rival(seed, name)` — no
`root` — so it came from `w4_api.DEFAULT_ROOT = C:/chlu/outputs/spoke-a-rivals/rivals` (fit on the FROZEN encoder's
features), while `$CHLU_RIVALS_ROOT` was honoured only for the METADATA json. Seed 0: default-root rival
`features_sha256 = ef6359a4…`; this arm's refit rival `95206c71…`. ⇒ the scored `rival_mlp_180k` was NOT the
PREREG §1 object ("the baseline REFIT on X's own features"). ⛔ The same wiring was used by the w8 CPU leg, so its
0.000 rival rows (and its VOID BAR D) are equally NOT a measurement of the refit baseline.
**Repair.** `exp_gen_read.py` now passes `root=$CHLU_RIVALS_ROOT` (unchanged behaviour when unset) and REFUSES a
rival whose `features_sha256` differs from the scored cache's. The `_dropout` variant also follows the root (none is
fitted per arm; rows 3/4 are full requests). The stale rival rows are moved aside, not deleted
(`score/*/val_seed0/STALE_rival_mlp_180k_spokeA/`).
**What does NOT change.** No bar, margin, estimand, corner set, seed or scope in §§2–5 moves. V2 is re-evaluated on
val with the repaired read; if it still fails, the test run is NOT spent and that is declared.
