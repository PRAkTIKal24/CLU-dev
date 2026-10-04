# w12 spoke/cache-gen2-seeds-3-4 — FINDINGS

**Agent:** experiment-engineer. **Scored runs: ZERO** — this spoke builds an instrument and
measures nothing about the primitive, as its task requires. **Box:** FORGIS_14.
**Branch:** `w12/cache-gen2-seeds-3-4`. **Worktree:** `C:/chlu/worktrees/CHLU-w12-cache-gen2-seeds-3-4`.
**`chlu.__file__` asserted** = `C:\chlu\worktrees\CHLU-w12-cache-gen2-seeds-3-4\chlu\__init__.py`
— inside this worktree, asserted **in code before a byte is written**
(`scripts/w12_cache_gen2_seeds_3_4.py`, the `WORKTREE not in _chlu.parents` guard) and echoed at
the head of every stage log and into `CACHE-READY.json.chlu_file`.

**THE DELIVERABLE:** `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json`.
Every number below names the file it came from.

---

## 0. THE ANSWER

⭐⭐ **BOTH SEEDS ARE READY. THE WAVE CAN REPORT 5 SEEDS.**

`CACHE-READY.json` → `seeds["3"].status = "READY"`, `seeds["4"].status = "READY"`.

Both generation-2 caches are built at full scope, match seeds 0–2 on **every** shape/dtype key,
and **pass the shipped re-encode gate bit-identically — 0 of 131,072 elements differing, 0.000
ULP, on both seeds.**

⛔ **The spoke's declared falsifier did not fire.** The task said: *"the re-encode gate rejects
seeds 3/4 ⇒ say so and stop."* It rejected neither. No 3-seed declaration is required on gate
grounds.

⚠ **One caveat the Hub must carry into any 5-seed caption — §3.** It is about *provenance*, not
about the gate.

---

## 1. ⛔ THE BLOCKER THE TASK DID NOT ANTICIPATE

**There is no generation-1 bridge — and therefore no frozen encoder — for seed 3 or seed 4
anywhere on this box.** Generation 2 is defined as *a re-encode through the frozen bridge
encoder*; with nothing to re-encode, the shipped path
(`exp_w4_rivals --stage cache2 --seeds 3 4`) cannot run at all.

| evidence | result |
|---|---|
| `ls $CHLU_GEN_BRIDGE_ROOT/shapes3d_recon/` | `seed_0`, `seed_1`, `seed_2` — **nothing else** |
| `find /c/chlu/outputs "$CHLU_CLAUDE_ROOT/outputs" -name encoder.pkl` | **29 hits, every one seed 0, 1 or 2** |
| `ls /c/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/` | `seed_0`, `seed_1`, `seed_2` |

`exp_w4_rivals.load_encoder` asserts `encoder.pkl`'s `param_hash` against the **generation-1
manifest** before re-encoding anything. For seeds 3 and 4 there is no encoder and no manifest,
so that assertion has nothing to stand on.

**THE ROUTE TAKEN.** The encoders for seeds 3 and 4 were **fit here by the shipped
`chlu.experiments.exp_gen_bridge --stage all`** — the same module that built seeds 0–2 — and the
generation-2 caches were then built from them by `w4_cache2.build_gen2_cache`, called verbatim.
⛔ No shipped module was edited.

⭐ **This route has direct precedent in this programme.** `w8/encoder-loop` built generation-2
full-scope caches on this same box from encoders **it trained itself**, and named the absence of
a generation-1 source in the manifest field rather than faking one —
`/c/chlu/outputs/w8-encoder-loop/run/seed_0/alone/cache/shapes3d_recon/seed_0/feature_cache.json`:
`schema chlu-gen-feature-cache/2`, `scope full`, `n_rows 480000`,
`features_bytes 1,966,080,128`, `device ['cpu:0']`, `host FORGIS_14`,
`source_generation1 = {"w8_arm":"alone","w8_hub":"encoder-loop"}`.
w12 does the same: `source_generation1.NOTE` reads *"there is NO generation-1 cache for this
seed; the source is the w12 bridge built by exp_gen_bridge --stage all on this box."*

---

## 2. ⭐ THE CACHES

`CACHE-READY.json` → `shape_dtype_pin` is read from the three existing generation-2 manifests,
and `shape_dtype_pin_disagreements_within_seeds_0_2` is **null** — the three agree with each
other on every key, so the pin is a pin and not one seed's opinion.

| key | **seed 3** | **seed 4** | pin (seeds 0,1,2) | match |
|---|---|---|---|---|
| `schema` | `chlu-gen-feature-cache/2` | `chlu-gen-feature-cache/2` | same | ⭐ |
| `generation` | 2 | 2 | 2 | ⭐ |
| `scope` | `full` | `full` | `full` | ⭐ |
| `n_rows` | **480000** | **480000** | 480000 | ⭐ |
| `dtype` | **float16** | **float16** | float16 | ⭐ |
| `feat_shape` | **[128, 4, 4]** | **[128, 4, 4]** | [128, 4, 4] | ⭐ |
| array shape | **[480000, 128, 4, 4]** | **[480000, 128, 4, 4]** | — | ⭐ |
| `D` | 2048 | 2048 | 2048 | ⭐ |
| `block` / `n_blocks` | 8192 / **59** | 8192 / **59** | 8192 / 59 | ⭐ |
| `features_bytes` | **1,966,080,128** | **1,966,080,128** | 1,966,080,128 | ⭐ |

`CACHE-READY.json` → `seeds["3"].shape_dtype_matches_seeds_0_2 = true`,
`shape_dtype_mismatches = null`; identically for seed 4.

**Per-seed hashes and paths** (from each `cache_generation2/shapes3d_recon/seed_<n>/feature_cache.json`):

| | seed 3 | seed 4 |
|---|---|---|
| cache dir | `C:\chlu\outputs\w12-cache-gen2-seeds-3-4\cache_generation2\shapes3d_recon\seed_3` | `…\seed_4` |
| `features_sha256` | `79ae328ca1432233a93d91f7922c79e060c8bd5eb12c199badfc50a07409f058` | `fc4ab4bcbc14de770d9ac64ca0f5041af8ab70c23a55d3b94ed15beec9d77c70` |
| `rows_sha256` | `dbddda1cf6d5a73f0384365068bf7fd7d2db66d70053d4601b5c89a8eff24e6d` | **the same** |
| `encoder_param_hash` | `41a0f08ca1cf0643ba4e29dbcb445b32d440614325b94cdae6addf69578fd0d9` | `435eb9e39f7f648b6293be33694d9f516092fc485de1a52e85dc318b42bc2fc4` |
| `block_sha256["0"]` | `f76f1f809cb607b23963b4f4930b09e07173c6c5cccaa845af1f451bb0b90e5a` | `dc3e62c15f05d3ae6d5af03a7d206e5dad6f9e67a503981533ad8592d6b23cc4` |
| per-block hashes | **59 recorded** | **59 recorded** |
| `wall_clock_s` | **396.583** | **340.853** |

⭐⭐ **`rows_sha256` is IDENTICAL on seeds 0, 1, 2, 3 and 4** —
`dbddda1cf6d5a73f0384365068bf7fd7d2db66d70053d4601b5c89a8eff24e6d`. The row vector is bit-for-bit
the same object in all five caches, so they index the same rows in the same order and are
poolable **row-for-row**, not merely "the same shape".

**ONE CPU DEVICE.** Both manifests carry `built_on_one_device: true` and
`device = {backend: "cpu", n_devices: 1, devices: ["cpu:0"], device_kinds: ["cpu"],
JAX_PLATFORMS: "cpu", XLA_FLAGS: null, host: "FORGIS_14",
jax_version "0.9.0", numpy_version "2.4.1", platform "Windows-10-10.0.26200-SP0",
precision.resolved "HIGHEST"}` — **the same device record as seeds 0–2**, whose manifests read
`['cpu:0'] / cpu / XLA_FLAGS None / HIGHEST`.
`w4_cache2.device_guard` logged `device guard PASSES: one device, cpu:0 (['cpu']),
precision=HIGHEST, jax 0.9.0` before either build wrote a byte (`gen2_seed_3.log`, `gen2_seed_4.log`).

---

## 3. ⚠⚠ THE ONE CAVEAT — read before writing a 5-seed caption

**Seeds 0–2's encoders were trained on a different machine.** Seed 0's generation-1 manifest
records `encoder_dir: "/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0"`
— a macOS path. Seeds 3–4's encoders were trained **here**, on FORGIS_14, on one CPU device.

⭐ **What is NOT a problem.** The encoder is a deterministic function of the seed alone
(`key = jax.random.PRNGKey(seed * 10_007 + 20_260_909)`, `exp_gen_bridge.run_encoder_stage`), so
*every* seed has a different encoder by construction; seeds 3–4 differing from 0–2 is the design,
not a defect. Architecture, budget and protocol are **identical and unmodified** — verified
against seeds 0–2's own manifests: `enc_channels [32,64,128,128]`, `kernel 4`, `stride 2`,
`groups 8`, **1500 steps × batch 128 × lr 1e-3**, `param_count 428576`, `D 2048`,
`feat_shape [128,4,4]`, rows `train 8192 / eval_a 1024 / eval_b 72`,
`whitening.full_whitening_feasible true`.

⚠ **What IS a declared caveat.** Seeds 3–4's encoders were produced in a **different
floating-point environment** from seeds 0–2's.
⛔ **Do not write "five seeds from one production run."**
⭐ **The honest form:** *five seeds, all from the identical shipped recipe and budget; seeds 0–2
fit on the original production box, seeds 3–4 fit on FORGIS_14 under the one-CPU-device guard.*

⭐ **How large is that environment difference? There is already a number for it on disk, and it
is not small.** `/c/chlu/outputs/spoke-a-rivals/reencode_gate_audit.json` shows that the
**generation-1 caches, written by the production box's encoders, do not reproduce on FORGIS_14 at
all** — 51/81/59 ULP on 0.49–0.98 % of elements, failing the 1-ULP bound on every seed — and that
audit's own verdict states: *"Generation 1 … cannot be reproduced, extended or audited here."*
That is a measurement of the **forward pass**. A 1500-step *training trajectory* in the same two
environments can only diverge at least that much, never less. ⛔ So the caveat is real and should
not be written away; ⭐ but it bears on **provenance**, not on whether seeds 3–4 are internally
sound, which §4 and §5 answer directly.

---

## 4. ⭐⭐ THE RE-ENCODE GATE — BOTH SEEDS PASS BIT-IDENTICALLY

`exp_gen_store._assert_cache_reencodes` — *the* gate `exp_gen_cell` fires at the top of every
`run_seed` — called **verbatim**, bounds unchanged (`ULP_BOUND 1.0`, `FRAC_BOUND 1e-2`; the
function has no parameter that can widen either). File: `reencode_gate_seeds_3_4.json`.

`verdict`: `seeds_gated ["3","4"]` · `passes ["3","4"]` · **`failures []`** ·
`max_ulp_range [0.0, 0.0]`.

| generation | seed | n_differing / n_elements | frac | max_abs_delta | **max_ulp** | n_nonfinite | passed |
|---|---|---|---|---|---|---|---|
| **2 (w12)** | **3** | **0 / 131,072** | **0.0** | **0.0** | **0.000** | 0 | ⭐ **true** |
| **2 (w12)** | **4** | **0 / 131,072** | **0.0** | **0.0** | **0.000** | 0 | ⭐ **true** |
| 2 (w4) | 0 | 1 / 131,072 | 7.63e-06 | 9.5367431640625e-07 | 1.0 | — | true |
| 2 (w4) | 1 | 3 / 131,072 | 2.29e-05 | 0.000244140625 | **3.0** | — | ⛔ **false** |
| 2 (w4) | 2 | 2 / 131,072 | 1.53e-05 | 9.5367431640625e-07 | 1.0 | — | true |
| 1 (w4) | 0 | 1280 / 131,072 | 0.00977 | 0.001953125 | **51.0** | — | ⛔ false |
| 1 (w4) | 1 | 640 / 131,072 | 0.00488 | 0.001953125 | **81.0** | — | ⛔ false |
| 1 (w4) | 2 | 1231 / 131,072 | 0.00939 | 0.001953125 | **59.0** | — | ⛔ false |

Rows for seeds 0–2 are quoted from `/c/chlu/outputs/spoke-a-rivals/reencode_gate_audit.json`.

The gate was invoked with `will_extend=False` and a mandatory `reason`, because the call encodes
nothing new and therefore cannot write an inconsistent row; ⭐ the **bounds are unchanged** and
`would_have_failed` is recorded — it reads **false** on both seeds, i.e. an enforced call would
have passed too.

⭐ **Why seeds 3–4 are tighter than seeds 0–2 — stated so nobody misreads it as a better cache.**
Their encoders were fit on the *same box and the same single CPU device* that then wrote the
cache, so the re-encode is the identical computation on the identical bytes. Seeds 0–2's 1–3 ULP
is the floor that batch-shape reordering allows once the encoder has crossed a machine boundary.
⛔ **This is a statement about provenance, not about feature quality.**

⭐ **The gate harness is not a different instrument.** Before either new cache existed, the same
call path was dry-run read-only against `spoke-a-rivals/.../seed_2` and returned
**2 / 131,072, frac 1.52587890625e-05, 1.000 ULP** — reproducing the w4 spoke's audit for that
seed exactly.

---

## 5. ⭐ THE FALSIFIER, AND AN INTERNAL CONSISTENCY PROOF

`w4_cache2.determinism_check` — re-encode one shard twice, compare the sha256 of the **quantised
cache bytes** — run **before** the expensive pass on each seed, as the module requires.
File: `determinism_check.json`.

| seed | rows | pass 1 sha256 | pass 2 | `passed` | wall clock |
|---|---|---|---|---|---|
| 3 | 0..8191 (8192) | `f76f1f809cb607b23963b4f4930b09e07173c6c5cccaa845af1f451bb0b90e5a` | **identical** | true | 48.789 s (first run) |
| 4 | 0..8191 (8192) | `dc3e62c15f05d3ae6d5af03a7d206e5dad6f9e67a503981533ad8592d6b23cc4` | **identical** | true | 25.2 s |

VERDICT on both: *"reproducible: the same rows encode to the same bytes."*

⭐⭐ **The falsifier's hash and the cache's block-0 hash are the same string, on both seeds.**
The falsifier hashed rows 0..8191; block 0 of each finished cache covers rows 0:8192 and its
recorded `block_sha256["0"]` is exactly `f76f1f80…` (seed 3) and `dc3e62c1…` (seed 4). **The bytes
proved reproducible are demonstrably the bytes that landed in `features.npy`.**

⭐ **A third, independent encode.** Seed 3's falsifier was re-run hours later in a fresh process
(after the defect in §8) and produced `f76f1f80…` again — the same hash on a third encode,
separated by ~1 h and a process boundary.

---

## 6. ⭐ THE BRIDGES — BOTH PROCEED

`bridge/shapes3d_recon/seed_<n>/bridge_manifest.json` → `result.gate`
(R3.1: PROCEED iff `ceiling_B >= 0.8 × ceiling_A`):

| seed | ceiling_A | ceiling_B | ratio | verdict |
|---|---|---|---|---|
| **3** | 0.9405924479166666 | 0.9675925925925926 | **1.028705466151391** | ⭐ **PROCEED** |
| **4** | 0.9591471354166666 | 0.9004629629629629 | **0.938816298056074** | ⭐ **PROCEED** |
| 0 (existing) | 0.9475911458333334 | 0.8680555555555557 | 0.9160654986831559 | PROCEED |

⭐ **Both new seeds clear the gate by a wider margin than seed 0 does.**

| | seed 3 | seed 4 |
|---|---|---|
| split A (n=1024): ceiling / exact_match / decode_mse / probe_on_real | 0.9405924479166666 / 0.6826171875 / 0.0031485091894865036 / **1.0** | 0.9591471354166666 / 0.7724609375 / 0.002690932946279645 / **1.0** |
| split B (n=72): ceiling / exact_match / decode_mse / probe_on_real | 0.9675925925925926 / 0.8055555555555556 / 0.0031639207154512405 / **1.0** | 0.9004629629629629 / 0.4861111111111111 / 0.002643734682351351 / **1.0** |
| encoder loss first → final (1500 steps) | 0.17086586356163025 → **0.002434287453070283** | 0.16947725415229797 → **0.00239490601234138** |
| decoder loss first → final (1200 steps) | 0.16593053936958313 → **0.003341174451634288** | 0.15768934786319733 → **0.002600476611405611** |
| probe loss first → final (1500 steps) | 2.22971 → **0.0001294322137255221** | 2.24418306350708 → **0.00011871553579112515** |
| decoder_param_hash | `022839727248226b…` | `65b5409142bb2b03…` |
| probe_param_hash | `41a3cfa1dc52259e…` | `1b1ad821b6d18ca7…` |

**R3.3 no-leak, asserted twice per seed** (once in the encoder stage, once in the decoder stage):
8192 training rows checked, **72/72 held-out images excluded, 0 leaked**, on both seeds.
The decoder recipe is the shipped `baseline`, unmodified: channels [128,64,32], groups 8,
constant lr 1e-3, batch 128, 1200 steps.

---

## 7. ⭐ THE LAYOUT DOWNSTREAM READS

`bridgify.json` records the four `BRIDGE_SIDECARS` copied beside each generation-2 cache with
**sha256 byte-identity asserted against the frozen originals** — the identical operation
`exp_w4_rivals.stage_bridgify` performs for seeds 0–2:

| sidecar | seed 3 | seed 4 |
|---|---|---|
| `encoder.pkl` | `51e70b3acdf053ac798ecebd8336537fbff7b4cebcb058cb3292b06946e64180` | (in `bridgify.json`) |
| `probe.pkl` | `98354141ea4b16881a553e24c437b9ec5a8d3473b28f88ee004740417466ca04` | (in `bridgify.json`) |
| `decoder.pkl` | `d3d0a495576edea01931bc8a67c3ecc52d25cbbd510718f102b3e372f51ab857` | (in `bridgify.json`) |
| `bridge_manifest.json` | `8bac94cdef74963d790f64e4b3acebe6a0a67f66220de464dd4190ad8ff1c038` | (in `bridgify.json`) |

**HOW A DOWNSTREAM STAGE ADOPTS THESE CACHES** (also recorded in
`CACHE-READY.json.downstream_layout`): set

```
gen_store.bridge_root = C:/chlu/outputs/w12-cache-gen2-seeds-3-4/cache_generation2
```

`exp_gen_store.bridge_dir` then resolves `shapes3d_recon/seed_3` and `…/seed_4`, whose contents
are `features.npy` · `feature_rows.npy` · `feature_cache.json` · `encoder.pkl` · `probe.pkl` ·
`decoder.pkl` · `bridge_manifest.json` — **the same shape as
`C:/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/seed_{0,1,2}/`.**

⚠ **The caches live under this spoke's own output directory, not beside seeds 0–2.** That is the
standing write rule, not an accident. ⭐ **Seeds 0–2 were not touched:** their `features.npy`
files still carry their 2026-09-19 19:02/19:07/19:12 mtimes and their 1,966,080,128-byte sizes.

---

## 8. ⭐ INDEPENDENT VERIFICATION, AND ONE DEFECT THIS SPOKE FOUND IN ITSELF

**`VERIFY.json`** — every hash in `CACHE-READY.json` recomputed **from the bytes on disk by a
script that does not import `chlu`**. `all_pass = true`. Per seed:

- `features_sha256_matches` **true** · `rows_sha256_matches` **true**
- `shape_matches_manifest` **true** (`[480000, 128, 4, 4]` on disk) · `dtype_matches_manifest`
  **true** (`float16`) · `bytes_match_manifest` **true** (1,966,080,128)
- **all 59 per-block sha256 rehashed and matched** (`n_blocks_rehashed 59`,
  `all_block_sha256_match true`)
- `rows_are_arange_480000` **true**
- `nonfinite_in_first_and_last_block` **0**

⚠ **DEFECT FOUND AND FIXED, disclosed against this spoke's own work.** `stage_det` wrote
`determinism_check.json` by **overwriting**, so launching the falsifier per seed erased the
previous seed's record — seed 4's run left a file containing only key `"4"`. It was caught by
`stage_ready`, which surfaced seed 3's determinism record as `null` instead of silently omitting
it. Fixed to **merge**, and seed 3's falsifier was **re-run rather than transcribed**, which is
why §5 can report a third independent encode. ⛔ No cache byte was affected: the defect was in a
record file, and the block hashes inside each cache manifest were never at risk.

---

## 9. ⛔ DECLARED NOT-RUN

1. ⛔ **Cross-box encoder-reproducibility control.** Refit seed 2's bridge *encoder* here with
   the identical shipped command and compare its `param_hash` against the frozen generation-1
   value `154d498610648fac12e9a72be73a69303c1cc3a114ec0401b4cc308a3166b7e1`. The script is
   written and committed (`scripts/w12_run_control.sh`). **NOT RUN.** Reason: it costs a further
   ~1 h encoder stage at the 4-core cap, the wave is gated on the caches rather than on this
   control, **and its direction is already bounded by evidence on disk** — the generation-1
   caches written by the production box's encoders fail to re-encode here by 51–81 ULP
   (`reencode_gate_audit.json`), so a *retrained* encoder cannot be expected to match bit-for-bit
   either. ⛔ What the control would add is the **magnitude** of the training-trajectory
   divergence, which remains unmeasured. Reported as an open question, never as a zero.
2. ⛔ **No scored run of any kind**, by the task's own instruction (*"Scored runs: NONE"*).
3. ⛔ **No statistic was re-derived.** Every seed-0–2 figure quoted here is copied from the named
   file on disk.
4. ⛔ **No comparison of a generation-1 number with a generation-2 number**, in either direction.
   `compare_to_generation1` was not run for seeds 3/4 because there is no generation 1 for them.

---

## 10. WHAT WOULD PROVE THIS SPOKE WRONG — checked

- ⛔ *The re-encode gate rejects seeds 3/4.* **It did not** — 0 elements differing, 0.000 ULP,
  on both. Had it fired, the instruction was to stop and declare a 3-seed wave.
- ⛔ *The caches do not match seeds 0–2 in shape/dtype, so they are not poolable.* **They match on
  every pin key**, and `rows_sha256` is bit-identical across all five seeds — a stronger statement
  than shape agreement.
- ⛔ *The build used more than one device, or a non-CPU one.* `device_guard` passed before either
  build wrote a byte; both manifests carry `n_devices: 1`, `devices: ["cpu:0"]`,
  `built_on_one_device: true`.
- ⛔ *`chlu` resolved outside this worktree.* Asserted in code before every stage;
  `CACHE-READY.json.chlu_file` records the resolved path.
- ⚠ *A 5-seed plot without a printed `n`.* This spoke cannot prevent that downstream; it can only
  make the honest caption available — §3 gives its exact wording.

---

## 11. FILE MANIFEST — `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/`

| file | what it is |
|---|---|
| **`CACHE-READY.json`** | ⭐ **THE DELIVERABLE.** Per seed: cache path, row count, dtype/shape, all 59 per-block sha256, features/rows sha256, device record, gate result, falsifier result, bridge summary, and the shape/dtype pin the match is asserted against. |
| `VERIFY.json` | every hash recomputed from the bytes by a script that does not import `chlu`; `all_pass true` |
| `reencode_gate_seeds_3_4.json` | the shipped gate's output for both seeds |
| `determinism_check.json` | the falsifier, both seeds |
| `bridgify.json` | the four sidecars, sha256 byte-identity asserted |
| `cache_generation2/shapes3d_recon/seed_{3,4}/` | the caches themselves, drop-in bridge dirs |
| `bridge/shapes3d_recon/seed_{3,4}/` | the bridges (encoder, decoder, probe, manifests) |
| `LOG.md` · `START_UTC.txt` · `END_UTC.txt` · `PIDS.txt` | the running record, timestamps, and the PIDs this spoke owned |
| `bridge_seed_{3,4}.log` · `gen2_seed_{3,4}.log` · `run.log` | raw stage logs |
| `w12-cache-gen2-seeds-3-4.bundle` | the branch, alone |
