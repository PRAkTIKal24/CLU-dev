# w15 spoke/bar-d-score — FINDINGS

**BAR D on the w8 GPU leg (full decoder), scored once under `PREREG-w15-bar-d.md` (+ AMENDMENT 1).**
Target `encoder-loop`, spend **2/3**. Seeds **0/1/2**, seed count **3** on every table. Condition `test`.
All paths below are relative to `/c/chlu/outputs/w15-bar-d-score/` unless absolute.

> ⛔⛔ **THIS TABLE NEVER SHARES A TABLE WITH THE w8 CPU LEG.** GPU pins **joint_steps 4000 / n_train 32768 /
> dec_refit_steps 1800 / warm_steps 4000**; CPU pins 400 / 8192 / 400 / 1500. The module's own rule
> (`exp_w8_encoder_loop.apply_compute_overrides`): *"a run at one budget and a run at another do not share a table."*
> CPU numbers appear below ONLY in separate blocks labelled with their pins, for a direction comparison. Nothing is pooled.

---

## 0. The answer in five lines

1. **BAR D-GPU = MIXED on row 4 and on row 3** by the pre-declared rule: no claim either way (`BAR-D.json` → `BAR_D_GPU.*.VERDICT`).
   The CLU-minus-refit-baseline margin is **inside ±1 novel tuple/corner in all six arm-seed cells on row 4**, and every
   growth CI95 spans 0.
2. **Descriptively, and this is what the decomposition paragraph should cite:** `loop_swap` lifts the refit baseline
   **as much as** it lifts the store (row 4: store +33.47 / +7.18 / +32.59, baseline +34.39 / +6.84 / +31.67), and both
   end at **0.99–1.00 of the 72-tuple ceiling**. Per PREREG §1 that is the signature of a finding **about the ENCODER, not
   about the store**. ⛔ It is not a BAR D verdict (the rule returned MIXED), and the ceiling limits what BAR D could show at all.
3. **ORDERING NOT REPRODUCED.** *"Unfreezing hurts / swap loss repairs"* holds on the CPU leg (A′ < 0 in 5/6 cells, A″ > 0
   in 4/6). On the GPU leg A′ < 0 in only **2/6** cells, while A″ > 0 in **6/6**. At 4000/32768 the swap loss still helps
   everywhere; unfreezing *helps* by ~+32 tuples/corner on seeds 0 and 2 and hurts only on seed 1.
   ⭐ **A negative, and it is the result.**
4. ⭐ **A wiring bug, found and repaired before any test number:** the w8 scoring path never loaded the refit baseline.
   It loaded spoke A's rival, which was fit on the **frozen** encoder's features. **The w8 CPU leg's `rival_mlp_180k` = 0.000 rows
   are therefore not a measurement of the refit baseline** (§4).
5. Charter C: all three controls **collapse** on every scored `alone` and `loop_swap` leg (max 0.490/corner). The margin is
   ~0, so **no store claim is made** from BAR D in either direction.

---

## 1. What ran

| step | what | artefact |
|---|---|---|
| env | `env_w3.sh` joins PYTHONPATH with `:` → Windows python mis-parses it → `equinox` not found. Re-exported with `;` (same two entries). Assert passed: `chlu.__file__` = `C:\chlu\worktrees\CHLU-w15-bar-d-score\chlu\__init__.py` | `env.sh`, `LOG.md` |
| inputs | GPU weights not on this box → `NEEDS-MAC.json` (45 files). Hub staged them; **integrity 9/9 arms PASS** (arm.json byte-identical to the evidence manifests; encoder/decoder `param_hash` and declift sha256 match; pins `[4000, 32768, 1800]` on all 9) | `NEEDS-MAC.json`, `STAGED-INTEGRITY.json` |
| caches | 9 × 480,000-row gen-2 caches regenerated **on this box** (CPU device guard), ~31 min each | `gpu-run/seed_*/*/cache.json` |
| cross-machine | **0/59 blocks** sha256-identical to the Mac's caches on **9/9** arms. Expected across CPU architectures (ULP-level); per PREREG §2 this is reported and does not void the run | `CACHE-XMACHINE.json` |
| rivals | `mlp_180k` refit on `alone` and `loop_swap` × 3 seeds, **178,952 params** (target 179,968, −0.56 %), pool 131,072, `w4_learners` verbatim | `gpu-run/seed_*/{alone,loop_swap}/rivals.json` |
| val gate | seed 0, free, **PASS** after the repair (§4) | `VAL-GATE.json` |
| ledger | row written **2026-09-23T11:44:05Z**; first test leg launched **11:44:16Z** | `/c/chlu/SCORED-LEDGER.md`, `PIDS.txt` |
| test | 9 legs, 39 arm-results, 109 corners each (49 row-4 + 60 row-3); all three guards (w15, w8, w5) authorised by w15's own row | `score/*/test_seed*/READ-TABLE.json`, `logs/test_*.log` |
| report | w8 cross-arm table verbatim + w15 BAR D / ordering | `CROSS-ARM-GPU-test.json`, `BAR-D.json` |

**Manifest count, measured:** **46 files** at `/c/chlu/evidence/outputs/w8-encoder-loop/cloud/manifests/`
(`find -type f | wc -l`; 254,660 B apparent size). The handover's "46" **is confirmed by count**. They are not at
`outputs/w8-encoder-loop/cloud/manifests/`, where the handover put them: that directory does not exist on this box
(`BAR-D.json` → `manifest_count_measured`).

---

## 2. BAR D-GPU — the scored row (pins 4000 / 32768 / dec 1800 · seeds 3 · test)

M(X) = mean paired Δ `pca_tensor@X − rival_mlp_180k@X` (novel tuples/corner). Growth = corner-level
[M(loop_swap) − M(alone)] with a 10,000-draw bootstrap. Source: `BAR-D.json` → `BAR_D_GPU`.

| row | seed | M(alone) [CI95] | M(loop_swap) [CI95] | **growth** [CI95] | corners +/− |
|---|---|---|---|---|---|
| r4 (49) | 0 | +0.918 [−1.633, +3.347] | +0.000 [0, 0] | **−0.918** [−3.245, +1.694] | 21/26 |
| r4 | 1 | −0.061 [−1.735, +1.612] | +0.286 [0, +0.714] | **+0.347** [−1.163, +1.898] | 17/21 |
| r4 | 2 | −0.837 [−3.857, +2.184] | +0.082 [−0.143, +0.347] | **+0.918** [−2.163, +4.000] | 27/22 |
| r3 (60) | 0 | −0.517 [−2.117, +1.083] | +0.217 [−0.150, +0.583] | **+0.733** [−0.950, +2.450] | 27/27 |
| r3 | 1 | +0.867 [−0.383, +2.050] | −0.017 [−0.200, +0.183] | **−0.883** [−2.133, +0.400] | 20/29 |
| r3 | 2 | −2.783 [−4.533, −1.050] | +1.183 [+0.383, +2.167] | **+3.967** [+1.983, +6.117] | 38/18 |

**Verdicts (pre-declared rule):** row 4 **MIXED** (growth > 0 on 2/3, not all 3; ≤ 0 on 1/3, not ≥ 2).
Row 3 **MIXED** (same pattern). VOID seeds: **0** (the refit rival is non-degenerate everywhere).

**Absolute means (row 4, ceiling 72)**, from `score/<arm>/test_seed<S>/READ-TABLE.json`:

| seed | `pca_tensor@alone` | `rival@alone` | `pca_tensor@loop_nosw` | `pca_tensor@loop_swap` | `rival@loop_swap` |
|---|---|---|---|---|---|
| 0 | 38.531 | 37.612 | 70.776 | 72.000 | 72.000 |
| 1 | 64.449 | 64.510 | 45.082 | 71.633 | 71.347 |
| 2 | 39.286 | 40.122 | 71.510 | 71.878 | 71.796 |

Row 3 (ceiling 36): `pca_tensor@alone` 14.650 / 30.783 / 16.883; `rival@alone` 15.167 / 29.917 / 19.667;
`pca_tensor@loop_swap` 35.817 / 35.767 / 35.650; `rival@loop_swap` 35.600 / 35.783 / 34.467 (same files).

**Charter C** (row 4 means on `alone` / `loop_swap`, same files). Collapse bar ≤ 1.0:
`landscape_removed` 0.000 everywhere · `landscape_shuffled` ≤ 0.020 · `random_refit` ≤ **0.490** (s1 `alone`) ·
`lookup_argmax` (printed FLOOR, never a rival) 0.000 everywhere. **All collapse on all 6 legs, both rows**
(`BAR-D.json` → `charter_C_collapse_all_seeds: true`). The store reads what is in its atoms. But BAR D's margin is ~0,
so this licenses **no store-over-baseline claim**.

**Why BAR D could not come out GROWS here (descriptive, not a bar).** `loop_swap` saturates: `pca_tensor` at
1.000 / 0.995 / 0.998 of the row-4 ceiling and the rival at 1.000 / 0.991 / 0.997 (`BAR-D.json` →
`DESCRIPTIVE_not_a_bar`). With both at the ceiling, M(loop_swap) ≈ 0 by construction. The alone→loop_swap gains are
near-identical for store and baseline (row 4: +33.47 vs +34.39, +7.18 vs +6.84, +32.59 vs +31.67). **Better features
help everyone.** That is the PREREG §1 description of a result about the encoder, not the store.

---

## 3. ORDERING — GPU vs CPU, two separate blocks, never merged

Rule (PREREG §4.2): REPRODUCED iff on GPU A′ < 0 in ≥ 4/6 cells AND A″ > 0 in ≥ 4/6.
A′ = `pca_tensor@loop_nosw − @alone`; A″ = `@loop_swap − @loop_nosw`. Source `BAR-D.json` → `ORDERING`; paired CIs from
`CROSS-ARM-GPU-test.json` → `per_seed.*.encoder_comparison`.

**GPU block — pins 4000 / 32768 / dec 1800, seeds 3, test:**

| cell | A′ (nosw − alone) [CI95] | A″ (swap − nosw) [CI95], sign p |
|---|---|---|
| s0 r4 | **+32.245** [+28.367, +36.347] | +1.224 [+0.388, +2.245], p=0.002 |
| s1 r4 | **−19.367** [−22.898, −15.775] | +26.551 [+23.204, +29.918], p<1e-4 |
| s2 r4 | **+32.224** [+28.082, +36.469] | +0.367 [+0.061, +0.735], p=0.039 |
| s0 r3 | **+18.367** [+15.233, +21.500] | +2.800 [+1.200, +4.617], p=0.0015 |
| s1 r3 | **−10.867** [−13.267, −8.400] | +15.850 [+13.600, +18.083], p<1e-4 |
| s2 r3 | **+18.600** [+16.300, +20.850] | +0.167 [−0.067, +0.417], p=0.30 |

A′ < 0: **2/6**. A″ > 0: **6/6** (CI excludes 0 in 5/6). ⇒ **NOT REPRODUCED.**

**CPU block — pins 400 / 8192 / dec 400, seeds 3, test** (`/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json`; recomputed
identically in `CPU-LEG-CROSS-ARM-recomputed.json`): A′ = +0.653 / −4.837 / −3.449 (r4), −5.967 / −2.617 / −2.283 (r3) →
**5/6 negative**. A″ = −1.367 / +5.694 / +7.388 (r4), −0.800 / +2.367 / +3.983 (r3) → **4/6 positive**.

**Reading.** The swap-loss half of the decomposition **survives, and gets stronger**, at 10× steps and 4× data (6/6).
The "unfreezing hurts" half **does not survive**. Its sign tracks the `alone` control, which is unstable. Seed 1 is the
only seed where `alone` trained well:
- `alone`'s probe margin over random init is −0.0505 / **+0.0242** / −0.0862
  (`/c/chlu/evidence/outputs/w8-encoder-loop/cloud/manifests/seed_{0,1,2}/alone/probe.json` → `MARGIN_OVER_RANDOM_INIT.aggregate`).
- `alone`'s row-4 score is 38.5 / **64.4** / 39.3.
- Seed 1 is exactly the seed where A′ < 0.

⛔ **The decomposition paragraph must not say "unfreezing hurts" as a general finding.** At GPU pins it is a property of one
seed's well-trained control. What the evidence supports at both pins is **"the swap loss helps"** (A″ > 0: CPU 4/6, GPU 6/6).

**Secondary, never the headline** (w8 rule, `CROSS-ARM-GPU-test.json` → `VERDICT`): BAR A (`loop_swap − alone`) row 4 =
+33.469 / +7.184 / +32.592, CI95 lows +29.469 / +5.143 / +28.429 → `BAR_A_PASS: true`; row 3 +21.167 / +4.983 / +18.767 →
PASS. ⛔ This is **not** a store result: BAR D shows the baseline gains the same, and BAR C.2 (equal seconds) **fails on all
three GPU seeds** (registry N322: `alone_wc` 297.6 / 312.8 / 318.2 s vs `loop_swap` 338.8 / 340.6 / 336.8 s). Only equal
**steps** holds: 4000 steps and 256,000 images, identical across arms (`CROSS-ARM-GPU-test.json` → `compute_match.0.BAR_C_equal_steps`).

---

## 4. ⭐ The rival wiring bug (PREREG AMENDMENT 1, written after the free val gate, before any test number)

- **Symptom:** val seed 0, first pass: `rival_mlp_180k` = **0.000** in both arms, with `pca_tensor` 40.958 / 71.750
  (`score/*/val_seed0/STALE_rival_mlp_180k_spokeA/READ-TABLE.pre-repair.json`). V2 FAIL.
- **Cause, measured:** `chlu/experiments/exp_gen_read.py` loaded the rival **model** with `load_rival(seed, name)` and no
  `root`, so it read from `w4_api.DEFAULT_ROOT = C:/chlu/outputs/spoke-a-rivals/rivals`. `$CHLU_RIVALS_ROOT` was
  honoured only for the metadata JSON. Seed 0: default-root rival `features_sha256 = ef6359a4…` (frozen features); this
  arm's refit rival `95206c71…`. The scored "baseline" was a model from another feature space, rendered through this
  arm's decoder.
- **Repair (this branch):** `load_rival(..., root=$CHLU_RIVALS_ROOT)`, with no behaviour change when the variable is unset,
  plus a hard refusal if the loaded rival's `features_sha256` ≠ the scored cache's.
- **After the repair**, val s0: rival **40.083** (`alone`) / **71.875** (`loop_swap`) vs `pca_tensor` 40.958 / 71.750;
  `landscape_removed` 0 / 0 → gate **PASS** (`VAL-GATE.json`).
- ⛔ **Consequence for wave 8.** The w8 CPU leg used the same scorer (`scripts/w8_encoder_loop_score.sh` → w5 leg →
  `exp_gen_read`). **Its `rival_mlp_180k` 0.000 rows (w8 `LOG.md` Addendum A) are spoke A's frozen-feature rival, not the
  refit one.** The w8 attribution of the zero to *"the 400-step decoder zeroes them all"* is **not established**. CPU BAR D
  is **unmeasured**, not "VOID because the baseline scores 0". `CPU-LEG-BAR-D-under-w15-rules.json` (VOID 3/3 on row 4)
  records what the pre-repair wiring produces, for the record only.
- Not affected: the w13 encoder-adapt scorer passes `root=` explicitly (`scripts/w13_score_leg.py:309`). The w5 legs
  never set `CHLU_RIVALS_ROOT`, so they correctly score spoke A's rivals on frozen features.

---

## 5. What did NOT run — DECLARED NOT-RUN, never nulls

| item | why | launch line |
|---|---|---|
| val on seeds 1, 2 | the PREREG gate reads seed 0 only; the legs were killed (own PIDs) to free contended CPU | `bash scripts/w15_bar_d_score.sh val <s> <arm> gpu-run score pca_tensor rival_mlp_180k pca_landscape_removed` |
| `frozen`, `alone_wc` scoring | w8's declared scope; BAR C.2 already fails on GPU (N322) so equal-seconds cannot be claimed | — |
| controls on `loop_nosw` | declared scope: `loop_nosw` is the decomposition column only (`pca_tensor`) | `... test <s> loop_nosw ... pca_landscape_removed pca_landscape_shuffled pca_random_refit` |
| `landscape_shuffled_across_wells` | not implemented for `pca_tensor` (`exp_w5_compose_field.PCA_ARMS`) | — |
| rivals other than `mlp_180k` | declared scope (the BAR D baseline is `mlp_180k`) | `--stage rivals --families cont_mlp transformer lstm` |
| BAR F cross-judge | out of scope | w8 `LOG.md` §6 item 2 |
| **CPU leg re-scored with the repaired rival** | it would be another scored run on `encoder-loop` (the **last**, 3/3) under its own PREREG; the CPU refit rivals already exist in `/c/chlu/outputs/w8-encoder-loop/run/seed_*/{alone,loop_swap}/rivals/` | `bash scripts/w8_encoder_loop_score.sh test <s> <arm> /c/chlu/outputs/w8-encoder-loop/run <new-out> rival_mlp_180k` (+ new PREREG + ledger row) |

---

## 6. What would have proved this wrong, and what did

- **BAR D would have shown the store shapes a representation the store itself exploits** if growth > 0 on all 3 seeds with
  CI low > 0 on ≥ 2. **It did not** (row 4 growth −0.918 / +0.347 / +0.918, every CI spans 0).
- **The CPU decomposition would have been confirmed** if A′ < 0 and A″ > 0 in ≥ 4/6 GPU cells. **A′ failed (2/6).** The
  "unfreezing hurts" half is refuted at these pins. The "swap loss helps" half survives (6/6).
- **Threats to this result.**
  - (i) Ceiling saturation at `loop_swap` (≥ 0.957 of ceiling for every cell) bounds BAR D.
  - (ii) The `alone` control is unstable across seeds (probe margins and row-4 scores above), so every A′ is dominated by
    control variance, as registry N321 warned.
  - (iii) Caches are not bit-identical to the Mac's (0/59 blocks). Every number here is on this box's caches, built from
    the hash-asserted encoders.
  - (iv) The rival repair touches a shared harness file. It is conditional on `CHLU_RIVALS_ROOT`, and it refuses a
    feature-space mismatch rather than scoring through one.

## 7. Byte figures, convention named

Store **524,288 B** (64 atoms × 2048 × float32, the `pca_tensor` atom block; `score/*/test_seed*/READ-TABLE.json` →
`arms.pca_tensor.store_bytes`). Address book printed separately, **never netted**: e.g. 194,544 B at `alone` s0 val
(`score/alone/val_seed0/READ-TABLE.json`). Rival **178,952 parameters** (parameter count, not bytes;
`gpu-run/seed_*/{alone,loop_swap}/rivals.json`), matched on read-path parameters to the 179,968 target; the memory ledger
is not matched.
