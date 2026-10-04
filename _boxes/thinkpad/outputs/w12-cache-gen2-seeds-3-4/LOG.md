# w12 spoke/cache-gen2-seeds-3-4 — LOG

START_UTC: 2026-09-21T11:09:27Z (see START_UTC.txt)
Worktree: C:/chlu/worktrees/CHLU-w12-cache-gen2-seeds-3-4  branch w12/cache-gen2-seeds-3-4
Output dir: /c/chlu/outputs/w12-cache-gen2-seeds-3-4/

## T+0 — read the task + standing rules
- Task file: /c/chlu/tasks/w12-spoke-cache-gen2-seeds-3-4.md. ONE JOB: gen-2 feature caches for
  seeds 3 and 4, re-encode gate applied, layout matching seeds 0-2. Scored runs: NONE.
- Read /c/chlu/evidence/outputs/w10-advisor-report.md (current programme state).

## T+1 — located the machinery
- Builder: chlu/eval/rivals/w4_cache2.py (441 lines) — device_guard, determinism_check,
  build_gen2_cache, compare_to_generation1. Schema "chlu-gen-feature-cache/2".
- Driver: chlu/experiments/exp_w4_rivals.py — stages determinism / cache2 / compare.
  gen2_root() = $CHLU_HUB_OUTPUTS/spoke-a-rivals/cache_generation2/shapes3d_recon
- Existing gen-2 caches (seeds 0,1,2) on disk at that root:
  each features.npy = 1,966,080,128 B, feature_rows.npy = 3,840,128 B.
- Re-encode gate audit already on disk: /c/chlu/outputs/spoke-a-rivals/reencode_gate_audit.json
  gen2 seed0 pass=true (1 ULP), seed1 pass=FALSE (3 ULP), seed2 pass=true (1 ULP);
  bounds = {differing_fraction 0.01, per_element_ulp 1.0}.

## T+2 — THE BLOCKER UNDER INVESTIGATION
- $CHLU_GEN_BRIDGE_ROOT = C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-feature-bridge/runs
  `ls .../shapes3d_recon/` → seed_0, seed_1, seed_2 ONLY. No seed_3, no seed_4.
- build_gen2_cache requires the FROZEN bridge encoder for the seed (encoder.pkl + the gen-1
  manifest's encoder_param_hash to assert against). For seeds 3/4 neither exists yet.
- Now searching for any seed-3/4 encoder elsewhere (w8 encoder-loop, wide bridge root).

## T+3 — BLOCKER CONFIRMED, and the route around it
- CONFIRMED: `ls $CHLU_GEN_BRIDGE_ROOT/shapes3d_recon/` → seed_0, seed_1, seed_2 only.
  There is **no generation-1 bridge, and therefore no frozen encoder, for seed 3 or seed 4
  anywhere on this box.** `exp_w4_rivals.load_encoder` asserts encoder.pkl's param_hash against
  the generation-1 manifest, so `--stage cache2 --seeds 3 4` cannot run as shipped.
- ROUTE: `chlu/experiments/exp_gen_bridge.py --stage all --seed N` is the shipped module that
  BUILT seeds 0-2's bridges (encoder → feature cache → shared decoder → probe → ceiling gate).
  Running it for seeds 3 and 4 produces the frozen encoder the gen-2 build needs.
- COST EVIDENCE (why this is affordable): /c/chlu/outputs/spoke-a-rivals/run_cache2b.log
  seed 0 full-scope gen-2 build = 59 blocks of 8192 = 480,000 rows in **4.5 min** on this CPU.
  Data is local: /c/chlu/data/gen/shapes3d/3dshapes.h5 (267,573,662 B). Disk free 662 G.
- chlu.__file__ ASSERTED = C:\chlu\worktrees\CHLU-w12-cache-gen2-seeds-3-4\chlu\__init__.py
  (inside THIS worktree). jax 0.9.0, numpy 2.4.1, python 3.11.
- Threads: capped at 4 by process affinity mask 0xF + OMP/MKL_NUM_THREADS=4.

## T+4 — plumbing smoke, then the real bridge builds launched
- SMOKE (scratchpad, NOT an artefact): `exp_gen_bridge --stage all --seed 3 --cache-scope used
  --n-train 512 --enc-steps 5 --dec-steps 5 --probe-steps 5` ran end-to-end, rc=0, in ~3 min,
  producing encoder.pkl / encoder_manifest.json / features.npy / feature_rows.npy /
  feature_cache.json / decoder.pkl / probe.pkl / bridge_manifest.json / shared_decoder_hash.json.
  Fix needed and applied: PYTHONIOENCODING=utf-8 (exp_gen_bridge prints U+26D4 and the Windows
  cp1252 console kills the run at the banner; exp_w4_rivals reconfigures stdout itself,
  exp_gen_bridge does not).
- REAL RUN LAUNCHED: scripts/w12_run_bridge.sh 3 4 →
  `exp_gen_bridge --dataset shapes3d --pretext recon --stage all --seed N --cache-scope used
   --no-plots --out <OUT>/bridge/shapes3d_recon/seed_N`
  (`--cache-scope used` mirrors generation-1 seeds 1 and 2, whose gen-1 caches are scope="used",
   n_rows=9288. The FULL-scope 480,000-row cache is generation 2's job, built next.)
  PID 21316, ProcessorAffinity forced to 0xF = 4 cores (PIDS.txt). OMP/MKL/OPENBLAS=4.
  Logs: bridge_seed_3.log / bridge_seed_4.log in the output dir.
- Build script written: scripts/w12_cache_gen2_seeds_3_4.py (stages det / cache2 / gate / ready).
  It calls w4_cache2.build_gen2_cache and exp_gen_store._assert_cache_reencodes VERBATIM and
  asserts chlu.__file__ is inside this worktree before it writes a byte.

## T+5 — the blocker is EXHAUSTIVELY established, and the route has PRECEDENT
- `find /c/chlu/outputs "$CHLU_CLAUDE_ROOT/outputs" -name encoder.pkl` → **29 hits, every one of
  them seed 0, 1 or 2.** There is no seed-3 or seed-4 encoder anywhere on this box.
  (3 in $CHLU_GEN_BRIDGE_ROOT/shapes3d_recon, 3 in spoke-a-rivals/cache_generation2, 23 in
  /c/chlu/outputs/w8-encoder-loop/.)
- PRECEDENT for building a generation-2 cache from an encoder fit ON THIS BOX:
  /c/chlu/outputs/w8-encoder-loop/run/seed_0/alone/cache/shapes3d_recon/seed_0/feature_cache.json
  = schema chlu-gen-feature-cache/2, generation 2, scope full, n_rows 480000, float16,
  feat_shape [128,4,4], D 2048, block 8192, 59 blocks, features_bytes 1,966,080,128,
  device ['cpu:0'] host FORGIS_14, wall_clock_s 600.82,
  source_generation1 = {'w8_arm':'alone','w8_hub':'encoder-loop'} — i.e. w8 also had no gen-1
  manifest to name and said so in the field. w12 does the same.
- THE SHAPE/DTYPE PIN, read from the three existing generation-2 manifests
  (C:/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/seed_{0,1,2}/feature_cache.json):
  schema=chlu-gen-feature-cache/2 · generation=2 · scope=full · n_rows=480000 · dtype=float16 ·
  feat_shape=[128,4,4] · D=2048 · block=8192 · n_blocks=59 · features_bytes=1,966,080,128 ·
  rows_sha256=dbddda1cf6d5a73f0384365068bf7fd7d2db66d70053d4601b5c89a8eff24e6d (IDENTICAL on all
  three: the row vector is arange(480000), so seeds 3/4 must reproduce this hash exactly) ·
  device ['cpu:0'] backend cpu XLA_FLAGS null precision HIGHEST.
  Per-seed features_sha256: s0 ef6359a4... s1 86b44d0f... s2 3ab14e58...
  Per-seed encoder_param_hash: s0 98d32602... s1 fc05e2c5... s2 154d4986...
  Build wall clock on this box: s0 269.23 s · s1 290.73 s · s2 111.67 s.

## T+6 — gate call path validated against an EXISTING cache before the new ones exist
- Dry-run of `exp_gen_store._assert_cache_reencodes(will_extend=False)` against
  C:/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/seed_2 (read-only):
  64 rows, **2/131072 elements differ (0.002%), max |delta| = 9.53674e-07 = 1.000 float16 ULP**,
  bounds 1 ULP / 1.0% → would have PASSED.
  ⭐ Those are the SAME numbers as /c/chlu/outputs/spoke-a-rivals/reencode_gate_audit.json
  generation_2["2"] (n_differing 2, frac 1.52587890625e-05, max_ulp_delta 1.0, max_abs_delta
  9.5367431640625e-07) — so w12's gate harness reproduces the w4 spoke's audit bit for bit.
- Control script written: scripts/w12_run_control.sh — refits seed 2's bridge ENCODER here with
  the identical shipped command, to MEASURE (not assume) whether an encoder trained on this box
  reproduces the frozen generation-1 encoder trained on the production box. Queued after seeds 3/4.

## T+7 — measured rate, and the box is shared
- seed 3 encoder: step 1 at ~11:15:30Z, **step 300/1500 at 11:25:52Z ⇒ 2.06 s/step** at 4 cores.
  Projected encoder stage ≈ 51 min/seed. loss 0.17087 → 0.00720 by step 300.
- The 4-thread cap is NOT optional here: `Get-CimInstance Win32_Process` shows four other w12
  spokes running on this box concurrently (w12_deletion_baselines PID 3216; three w12_val_sweep
  PIDs 26460 / 30616 / 30436). My only process is PID 21316, affinity 0xF. No box-wide kill will
  be issued; PIDS.txt holds mine.
- Projected w12 cost: ~1 h 50 m per seed (encoder 1500 + used-scope cache + decoder 1200 +
  probe 1500 + ceiling), then ~5 min per seed for the 480,000-row generation-2 build,
  then ~1 min per seed for the re-encode gate.

## T+8 — REVISED PROJECTION (the box got busier; this is the honest cost)
- seed 3 encoder step timestamps, from bridge_seed_3.log + `date -u`:
  step 1 ~11:15:30Z · step 300 11:25:52Z (2.06 s/step) · step 600 11:53:18Z (5.49 s/step).
  The box went from 5 to 10 concurrent python processes between those two samples.
- At 5.5 s/step the remaining 900 encoder steps ≈ 82 min ⇒ seed 3 encoder ends ≈ 13:15Z.
  Whole seed-3 bridge (encoder + used-scope cache + decoder 1200 + probe 1500 + ceiling)
  projected ≈ 15:00Z; seed 4 sequentially after it.
- ⛔ I will NOT raise the 4-core cap: four other w12 spokes are on this box.
- ⭐ MITIGATION: CACHE-READY.json is written INCREMENTALLY. The moment seed 3's generation-2
  cache and gate exist it is published with seed 3 READY and seed 4 as a DECLARED NOT-RUN with
  its reason, so a downstream hub can gate on a 4-seed wave rather than wait for a 5-seed one.

## T+9 — seed 3 encoder FIT
- bridge_seed_3.log: encoder 1500/1500 steps done, loss 0.17087 (step 1) → 0.00720 (300) →
  0.00462 (600) → 0.00325 (900) → 0.00290 (1200) → **0.00243 (1500)**.
- Now caching penultimate features: scope=used, 9288 rows (matches generation-1 seeds 1 and 2,
  whose gen-1 caches are scope="used", n_rows=9288).

## T+10 — seed 3 encoder FROZEN; generation-2 build launched alongside the decoder stage
From bridge/shapes3d_recon/seed_3/encoder_manifest.json:
- encoder_param_hash = **41a0f08ca1cf0643ba4e29dbcb445b32d440614325b94cdae6addf69578fd0d9**
- feat_shape [128,4,4] · D 2048 · param_count 428576 · steps 1500 · batch 128 · lr 0.001
- loss_first 0.17086586356163025 → loss_final **0.002434287453070283**
- rows: train 8192 · eval_a 1024 · eval_b 72 — IDENTICAL to seeds 0-2's bridge manifests
- R3.3 no-leak: 8192 checked, 72/72 held-out excluded, **0 leaked**
- its generation-1-schema "used" cache: schema chlu-gen-feature-cache/1, scope used,
  n_rows 9288, float16, feat_shape [128,4,4], D 2048, **features_bytes 38,043,776** —
  ⭐ byte-for-byte the same size as generation-1 seeds 1 and 2 (38,043,776 B).
  features_sha256 d98c70d3... rows_sha256 017d78fa...
- whitening: full_whitening_feasible = true (D=2048 <= 20000), same as seeds 0-2.

LAUNCHED `scripts/w12_run_gen2.sh 3` (det falsifier → cache2 → gate) as PID 34976, pinned to the
SAME 0xF mask as PID 21316 so the two w12 processes SHARE 4 cores and the box load does not rise.
The generation-2 build needs the ENCODER only, so it does not wait on the decoder/probe stage.

## T+11 — seed 3: THE FALSIFIER PASSES, device guard PASSES, 480k build under way
From /c/chlu/outputs/w12-cache-gen2-seeds-3-4/determinism_check.json (seed 3):
- re-encode one shard twice, 8192 rows (rows 0..8191), dtype float16
  pass 1 sha256 = **f76f1f809cb607b23963b4f4930b09e07173c6c5cccaa845af1f451bb0b90e5a**
  pass 2 sha256 = **f76f1f809cb607b23963b4f4930b09e07173c6c5cccaa845af1f451bb0b90e5a**
  passed = **true**, wall_clock_s 48.789,
  VERDICT "reproducible: the same rows encode to the same bytes".
- gen2_seed_3.log: "device guard PASSES: one device, cpu:0 (['cpu']), precision=HIGHEST, jax 0.9.0"
- 480,000-row build running at ~7 s/block, 59 blocks ⇒ ~7 min.

## T+12 — ⭐ SEED 3'S GENERATION-2 CACHE IS BUILT AND GATED
From cache_generation2/shapes3d_recon/seed_3/feature_cache.json:
- schema **chlu-gen-feature-cache/2** · generation **2** · scope **full** · n_rows **480000**
- dtype **float16** · feat_shape **[128,4,4]** · array shape **[480000,128,4,4]** · D **2048**
- block **8192** · n_blocks **59** · **59 per-block sha256 recorded**
- features_bytes **1,966,080,128** · features_sha256
  **79ae328ca1432233a93d91f7922c79e060c8bd5eb12c199badfc50a07409f058**
- rows_sha256 **dbddda1cf6d5a73f0384365068bf7fd7d2db66d70053d4601b5c89a8eff24e6d**
  ⭐ IDENTICAL to seeds 0, 1 and 2 — the row vector is bit-for-bit the same object.
- encoder_param_hash 41a0f08ca1cf0643... · built_on_one_device **true**
- device ['cpu:0'] backend cpu host FORGIS_14 XLA_FLAGS null precision HIGHEST — same as 0-2
- wall_clock_s **396.583** (6.6 min)
⭐ EVERY pin key matches seeds 0-2 exactly (CACHE-READY.json
  seeds["3"].shape_dtype_matches_seeds_0_2 = true, shape_dtype_mismatches = null).

## T+13 — ⭐⭐ THE RE-ENCODE GATE ON SEED 3: BIT-IDENTICAL, 0.0 ULP
From reencode_gate_seeds_3_4.json, generations.generation_2_w12["3"]:
- n_checked **64 rows / 131,072 elements** · n_differing **0** · frac_differing **0.0**
- max_abs_delta **0.0** · max_ulp_delta **0.0** · n_nonfinite 0
- bounds applied: 1.0 ULP, 1e-2 fraction · would_have_failed **false** · **passed = true**
⭐ COMPARE /c/chlu/outputs/spoke-a-rivals/reencode_gate_audit.json, same gate, same bounds:
  generation 2 seeds 0/1/2 sat at **1.0 / 3.0 / 1.0 ULP** (seed 1 FAILED the 1-ULP bound);
  generation 1 seeds 0/1/2 sat at **51 / 81 / 59 ULP** and failed on every seed.
  Seed 3 is at **0.0 ULP** — strictly the tightest cache in the programme, because its encoder
  was fit on the SAME box and the SAME single CPU device that then wrote the cache.
⛔ The gate is therefore NOT a reason to stop: it does not reject seed 3, it passes it exactly.

## T+14 — interim CACHE-READY.json published (do not wait for seed 4 to gate a 4-seed wave)
- seed 3 status **CACHE_READY_BRIDGE_PENDING** (cache complete + gated; the bridge's
  decoder/probe/ceiling stage was still running when the file was written).
- seed 4 status **NOT_RUN**, reason "generation-2 cache for this seed was not completed".
- `shape_dtype_pin_disagreements_within_seeds_0_2` = null: the three existing generation-2
  manifests agree with each other on every pin key, so the pin IS a pin.
- Added a `bridgify` stage (verbatim copy of exp_w4_rivals.BRIDGE_SIDECARS semantics) so the
  cache dir becomes a drop-in bridge dir once the decoder/probe land.

## T+15 — seed 3 decoder rate, and a deliberate decision NOT to parallelise seed 4
- seed 3 decoder: started ~12:26Z, **step 240/1200 at 12:33:47Z ⇒ 1.9 s/step**, loss 0.16593 → 0.00663.
  Projected: decoder ends ~13:04Z, probe (1500 steps) ~13:55Z, seed-3 bridge ~14:00Z,
  seed-4 bridge ~14:00→16:00Z, seed-4 generation-2 cache + gate ~16:15Z.
- ⛔ DECISION: seed 4's bridge is NOT launched in parallel. `scripts/w12_run_bridge.sh` is a
  sequential loop whose skip-guard is `[ -f bridge_manifest.json ]`; launching seed 4 myself now
  would leave that loop free to start a SECOND seed-4 process writing into the same directory the
  moment seed 3 finishes. Two concurrent writers into one cache dir is precisely the silent
  corruption these rules exist to prevent, and no wall-clock saving is worth it — the two jobs
  would in any case share the same 4 cores, so throughput would barely move.
  The cost of this decision is wall clock only, and it is stated here rather than hidden.

## T+16 — seed 3 decoder fit; probe now fitting
- seed 3 THE shared decoder: 1200/1200 steps, loss 0.16593 (step 1) → 0.00663 (240) →
  0.00485 (480) → 0.00373 (720) → 0.00332 (960) → **0.00334 (1200)**.
- Now fitting the probe on REAL images, 1500 steps. Then measure_ceiling + the R3.1 gate
  (PROCEED iff ceiling_B >= 0.8 * ceiling_A).

## T+17 — ⭐⭐ SEED 3 IS **READY**. Its bridge PROCEEDS.
From bridge/shapes3d_recon/seed_3/bridge_manifest.json, `result.gate`:
- **ceiling_A = 0.9405924479166666 · ceiling_B = 0.9675925925925926 · ratio = 1.028705466151391**
  · floor 0.8 · **verdict PROCEED** (R3.1: PROCEED iff ceiling_B >= 0.8 x ceiling_A).
  ⭐ Seed 3 clears the gate more comfortably than seed 0 does
  (seed 0, $CHLU_GEN_BRIDGE_ROOT/shapes3d_recon/seed_0/bridge_manifest.json:
   ceiling_A 0.9475911458333334, ceiling_B 0.8680555555555557, ratio 0.9160654986831559, PROCEED).
- per split A (n=1024): ceiling 0.9405924479166666, exact_match 0.6826171875,
  decode_mse 0.0031485091894865036, probe_on_real **1.0**
- per split B (n=72):   ceiling 0.9675925925925926, exact_match 0.8055555555555556,
  decode_mse 0.0031639207154512405, probe_on_real **1.0**
- decoder_param_hash 022839727248226b… · probe_param_hash 41a3cfa1dc52259e…
- dec_loss 0.16593053936958313 → **0.003341174451634288** (1200 steps, recipe "baseline",
  channels [128,64,32], groups 8, lr 1e-3 constant, batch 128 — the SHIPPED recipe, unmodified)
- probe_loss 2.22971 → **0.0001294322137255221** (1500 steps)

BRIDGIFY: bridgify.json records the four BRIDGE_SIDECARS copied beside seed 3's generation-2
cache with sha256 byte-identity asserted against the frozen originals. The cache dir is now a
drop-in bridge dir, exactly as exp_w4_rivals.stage_bridgify makes seeds 0-2.
CACHE-READY.json: **seeds["3"].status = "READY"**; seeds["4"].status = "NOT_RUN".

## T+18 — seed 4's bridge started 13:41Z (PID 21672, affinity 0xF, recorded in PIDS.txt)

## T+19 — seed 4 encoder FROZEN; its generation-2 build launched alongside its decoder stage
- seed 4 encoder: 1500/1500 steps, loss 0.16948 (step 1) → 0.00948 (300) → 0.00470 (600) →
  0.00359 (900) → 0.00276 (1200) → **0.00239 (1500)**. D = 2048, feat_shape (128,4,4).
- `scripts/w12_run_gen2.sh 4` launched as PID 27740, pinned to the same 0xF mask (PIDS.txt).

## T+20 — ⭐⭐ BOTH CACHES BUILT. BOTH PASS THE RE-ENCODE GATE BIT-IDENTICALLY.
seed 4 falsifier (determinism_check.json): rows 0..8191, pass1 = pass2 =
**dc3e62c15f05d3ae6d5af03a7d206e5dad6f9e67a503981533ad8592d6b23cc4**, passed true, 25.2 s.
device guard PASSES: one device, cpu:0, precision HIGHEST, jax 0.9.0.

seed 4 cache (cache_generation2/shapes3d_recon/seed_4/feature_cache.json):
- schema chlu-gen-feature-cache/2 · generation 2 · scope full · n_rows **480000** · float16
- feat_shape **[128,4,4]** · D 2048 · block 8192 · n_blocks **59** · 59 per-block sha256
- features_bytes **1,966,080,128**
- features_sha256 **fc4ab4bcbc14de770d9ac64ca0f5041af8ab70c23a55d3b94ed15beec9d77c70**
- rows_sha256 **dbddda1cf6d5a73f0384365068bf7fd7d2db66d70053d4601b5c89a8eff24e6d** ⭐ = seeds 0-2
- encoder_param_hash 435eb9e39f7f648b6293be33694d9f516092fc485de1a52e85dc318b42bc2fc4
- built_on_one_device true · device ['cpu:0'] cpu FORGIS_14 XLA_FLAGS null precision HIGHEST
- wall_clock_s **340.853**
⭐ block_sha256["0"] = dc3e62c1… = the falsifier's own hash, exactly as on seed 3. The bytes
proved reproducible are the bytes that landed in features.npy.

RE-ENCODE GATE (reencode_gate_seeds_3_4.json → verdict):
  seeds_gated ["3","4"] · passes ["3","4"] · **failures [] ** · max_ulp_range **[0.0, 0.0]**
  Both seeds: n_differing **0** / 131,072 elements, frac_differing 0.0, max_abs_delta 0.0,
  max_ulp_delta **0.0**, n_nonfinite 0, would_have_failed false.
⛔ The task's declared falsifier — "the re-encode gate rejects seeds 3/4 ⇒ say so and stop" —
   DID NOT FIRE on either seed. No 3-seed declaration is required on gate grounds.

Remaining: seed 4's decoder (1200) + probe (1500) + ceiling gate, then bridgify + final ready.

## T+21 — ⭐⭐ SEED 4 IS READY TOO. BOTH SEEDS READY.
seed 4 bridge (bridge/shapes3d_recon/seed_4/bridge_manifest.json → result.gate):
- **ceiling_A 0.9591471354166666 · ceiling_B 0.9004629629629629 · ratio 0.938816298056074 ·
  floor 0.8 → PROCEED** (wider margin than seed 0's 0.9160654986831559).
- split A n=1024: ceiling 0.9591471354166666, exact 0.7724609375, mse 0.002690932946279645,
  probe_on_real 1.0. split B n=72: ceiling 0.9004629629629629, exact 0.4861111111111111,
  mse 0.002643734682351351, probe_on_real 1.0.
- dec_loss 0.15768934786319733 → 0.002600476611405611 (1200 steps);
  probe_loss 2.24418306350708 → 0.00011871553579112515 (1500 steps).
- decoder_param_hash 65b5409142bb2b03… probe_param_hash 1b1ad821b6d18ca7…
- bridgify.json: both seeds' four sidecars copied, sha256 byte-identity asserted.
- CACHE-READY.json: **seeds["3"].status = "READY"**, **seeds["4"].status = "READY"**.

## T+22 — ⚠ A DEFECT IN MY OWN CODE, FOUND, FIXED, AND THE NUMBER RE-MEASURED
- `stage_det` wrote determinism_check.json by OVERWRITING. Running the falsifier per seed meant
  seed 4's launch erased seed 3's record: `json.load(...).keys()` returned `['4']`.
- CAUGHT BY `stage_ready`, which surfaced seed 3's `determinism_check` as **null** rather than
  silently dropping the key — the failure was visible because the file is assembled, not narrated.
- FIXED: `stage_det` now MERGES into any existing file (commented in the source with the date).
- ⭐ seed 3's falsifier was **RE-RUN, not transcribed from this LOG**: fresh process, ~1 h later,
  pass1 = pass2 = **f76f1f809cb607b23963b4f4930b09e07173c6c5cccaa845af1f451bb0b90e5a** — the
  SAME hash on a THIRD independent encode (11.1 s). determinism_check.json now holds both seeds.
- ⛔ No cache byte was affected: the defect was in a record file; the 59 per-block sha256 inside
  each cache manifest were never at risk, and §8 of FINDINGS re-verifies all 118 of them.

## T+23 — INDEPENDENT VERIFICATION (VERIFY.json) — all_pass = true
A script that does NOT import `chlu` recomputed every hash in CACHE-READY.json from the bytes:
- seed 3 / seed 4: features_sha256 match **true** · rows_sha256 match **true** ·
  shape on disk [480000,128,4,4] matches **true** · dtype float16 matches **true** ·
  features_bytes 1,966,080,128 matches **true** ·
  **59/59 per-block sha256 rehashed and matched** on each seed (118 in total) ·
  rows_are_arange_480000 **true** · nonfinite in first+last block **0** ·
  verify 3.5 s / 4.8 s.

## T+24 — SEEDS 0-2 NOT TOUCHED (the task's explicit prohibition)
`ls -la --time-style=full-iso` on the three existing generation-2 features.npy:
  seed_0 1,966,080,128 B 2026-09-19 19:12:36 · seed_1 … 19:02:12 · seed_2 … 19:07:03.
All three retain their 2026-09-19 mtimes and byte counts. This spoke wrote only inside
/c/chlu/outputs/w12-cache-gen2-seeds-3-4/ and its own worktree.

## T+25 — no detached work remains
`Get-CimInstance Win32_Process` filtered to exp_gen_bridge / w12_cache_gen2 → **NONE running**.
PIDs this spoke owned and the only ones it would ever have killed (PIDS.txt):
21316 (bridge seed 3) · 34976 (gen2 seed 3) · 21672 (bridge seed 4) · 27740 (gen2 seed 4).
⛔ No box-wide kill was issued at any point; four other w12 spokes shared this box throughout.

## T+26 — CLOSE
- Commits on w12/cache-gen2-seeds-3-4: 5dd5f19, c39ecab, **25c578c** (HEAD).
- Bundle: /c/chlu/outputs/w12-cache-gen2-seeds-3-4/w12-cache-gen2-seeds-3-4.bundle (6,005,239 B),
  built from THAT BRANCH ONLY (never --all).
  `git bundle verify` → "is okay … The bundle records a complete history."
  `git bundle list-heads` → **exactly one ref**:
  25c578cfd947defdd4fc4a1dabca2e860e5c6aca refs/heads/w12/cache-gen2-seeds-3-4
- END_UTC.txt written.
