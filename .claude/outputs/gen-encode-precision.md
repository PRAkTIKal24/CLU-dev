# gen-encode-precision — experiment-engineer report

Task + acceptance criterion: make the frozen encoder produce the same features on GPU as on CPU by forcing full
float32 on the encode path — **CPU numbers must be bit-identical before and after**, and `_assert_cache_reencodes`
must be untouched.
Status: **done (laptop half). ⛔ The fix is UNCONFIRMED — every cluster leg is `NOT RUN`, and the cluster run is
the test.**

**⚠ Reconciliation list, in the first 10 lines (protocol §5 corollary — this needs an owner):** one **out-of-scope
one-line hookup** is owed before criterion 3 is fully met — the precision record reaches the **log** and the
**feature-cache manifest** (both mine), but not the **store run manifest**'s `cache_reencode_check` block, which is
assembled in `chlu/experiments/exp_gen_store.py` (⛔ forbidden to me). Exact diff in §7. No other site changes.

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** ⛔ **none — a correctness repair.** **Falsifies:** nothing.
- **Laundering control:** ⭐ CPU results bit-identical before vs after. ✅ **HELD** — see §3.
- **Does NOT falsify:** a GPU/CPU disagreement that survives at ≤ 1 ULP on a handful of elements is the gate's
  designed pass band, not a residual defect.

---

## 1. The diagnosis the fix is built on

`grep` was right: **nothing in `chlu/` ever set a matmul precision.** `eqx.nn.Conv2d.__call__` calls
`lax.conv_general_dilated` with **no `precision=` argument** (equinox `nn/_conv.py:239`), so every convolution on
the encode path bound `precision=None`, which `lax.canonicalize_precision` resolves against
`jax_default_matmul_precision` — **unset**, therefore `Precision.DEFAULT`, therefore **TF32 permitted for float32
convolutions on Ampere/Ada**. The cache was written on CPU, where `Precision` is a documented no-op and the convs
ran in true float32. That is the whole defect, and it matches the measured signature exactly: absolute error
pinned at one float16 ULP of the tensor's **max** magnitude (my fixed batch maxes at **4.157**, so 2^-8 is the ULP
of [4, 8) — the reported `0.00390625`), relative error unbounded (4181 ULP on a ~1e-3 element).

**Negative control, measured, not assumed.** The jaxpr of the function `encode_images` hands to `filter_jit`:

| tree | conv equations | `precision` param |
|---|---|---|
| `main` @ `dcd667e` | 2 | `[None, None]` ⇒ XLA default ⇒ **TF32 on Ada** |
| this branch @ `70f3bf6` | 2 | `[(Precision.HIGHEST, Precision.HIGHEST), (…)]` |

## 2. What I changed (`chlu/core/gen_bridge.py`, encode path only)

1. **`ENCODE_PRECISION = "highest"`** — a module constant with the L40S measurement and the TF32 reasoning written
   into its docstring, so the next reader does not have to rediscover it.
2. **`encode_images` enters `jax.default_matmul_precision(ENCODE_PRECISION)` *inside* the function handed to
   `eqx.filter_jit`**, not around the call site. That matters: a `precision=None` op resolves the default at
   **trace** time and freezes it into the jaxpr, so putting the context at the call site would make the arithmetic
   depend on which caller happened to trigger the trace. ⛔ **No global flag is mutated, and nothing at all happens
   at import time** (§7.23 — the six-module ordering hazard is not made worse).
3. **`encode_precision_record()`** — requested precision, resolved precision, backend, device kind, device count,
   JAX version. Logged **once per process per distinct (precision, backend, device)** so a 300-block cache build
   prints one line, and written into `feature_cache.json` as `encode_precision`.
   ⭐ That manifest field is the point, not decoration: **a cache written under one precision and extended under
   another is a different computation with a matching `encoder_param_hash`** — the exact failure the task says
   nothing downstream would notice. It is additive and optional; pre-existing manifests do not carry it, so every
   reader must use `.get("encode_precision")`. `schema` stays `chlu-gen-feature-cache/1` (nothing asserts on the
   key set; the only other mention is a fixture in `tests/test_stage_artefacts.py:119`).
4. **Signature:** `encode_images(encoder, images, batch=512, log=None)`. `log` is new, keyword, defaults to
   `print`; all four existing call sites pass positionally and are unaffected.

⛔ **`chlu/experiments/exp_gen_store.py` is byte-untouched** — `git diff main..HEAD -- chlu/experiments/exp_gen_store.py`
is empty. `ULP_BOUND` and `FRAC_BOUND` are as the peer left them. **I widened nothing.**

## 3. ⭐⭐ Acceptance 1 — CPU bit-identity, before vs after

Same script, same seeds, two trees: `main @ dcd667e` exported with `git archive` to a scratch dir, and this branch.
Shipped default bridge encoder `(3,64,64) → [32,64,128,128], k=4, s=2, groups=8 → (128,4,4)`; 64 images from
`np.random.default_rng(1234)`; `batch=32`; **`D = 2048`, `64 × 2048 = 131072` elements — the same tensor size the
cluster convicted.**

| | `before` (main `dcd667e`) | `after` (branch `70f3bf6`) |
|---|---|---|
| sha256 of raw float32 bytes | `971983389ce7eb6c6366d42b604083ad6daa50014e82eaec77564f5de8ea87f9` | **identical** |
| sha256 of the `.npy` file | `5f1b6658047a57da8cafd7dc6f5ca4e78c079ef2817ea96b92d66ed5e06aafb5` | **identical** (`cmp` clean) |
| `sum` in float64 | `52256.795945495745` | `52256.795945495745` |
| `max` | `4.157252311706543` | `4.157252311706543` |
| `encoder_param_hash` | `19ce3a862ff5c70f97206531a9d80d4c8b6173807968ca42dee495fc0d8a177a` | identical |

✅ **Every byte equal.** The laundering control holds; the laptop's numbers did not move, so no cached feature and
no asserted hash in the programme is put in doubt. Pinned as a test as well
(`test_encode_precision_does_not_move_the_numbers_on_cpu`), and independently: compiled once and compared directly,
`HIGHEST` vs `DEFAULT` on CPU gives `max_abs_delta = 0.0`, `bitwise_equal = True`.

## 4. ⚠ Acceptance 6 — what it costs

`encode_images` rebuilds its jit wrapper on every call (true on `main` too — unchanged here), so its wall time is
dominated by trace+compile and hides the arithmetic. Both are reported.

| measurement (CPU, 64 imgs, default encoder) | DEFAULT | HIGHEST | ratio |
|---|---|---|---|
| compiled once, 40 reps — **min** | 12.195 ms | 12.196 ms | **1.00005×** |
| compiled once, 40 reps — median | 12.785 ms | 12.951 ms | 1.013× |
| whole `encode_images` call, 25 reps — min | 131.6 ms | 131.8 ms | 1.002× |
| whole `encode_images` call, 25 reps — median | 139.6 ms | 143.3 ms | 1.026× |

⭐ **CPU cost line: free (1.00005× on the cleanest estimator; the median spread is machine noise), exactly as
documented — `Precision` has no effect on CPU backends.** ⛔ Note this also means **the CPU can never demonstrate
the fix**, only that it is harmless.

**Expected GPU cost — ESTIMATED, `NOT MEASURED`.** Forcing `HIGHEST` moves the convolutions off the TF32
tensor-core path onto a true-float32 path, so the *conv kernels* can slow by a factor in the low single digits up
to ~an order of magnitude depending on the card's FP32-vs-dense-TF32 ratio (much wider on A100-class than on Ada).
⭐ **What bounds the impact is the absolute size of the work, which I did measure.** The encoder is
**45.09 MFLOP/image** forward-only (22,544,384 MACs: 1.57 + 8.39 + 8.39 + 4.19 M over the four blocks), so:
- the **preflight re-encode** is 64 images = **2.89 GFLOP** — milliseconds on any GPU, at any precision;
- a **full `scope=full` cache build** is 480,000 images (Shapes3D) = **21.6 TFLOP**, or 737,280 images (dSprites) =
  **33.2 TFLOP** — seconds of GPU compute, against a path that streams 1.7 GB of gzip-chunked HDF5 and is
  **I/O-bound, not arithmetic-bound**.
For scale: this laptop's CPU sustains **237 GFLOP/s** on this encoder, i.e. ~91 s for the whole Shapes3D cache, and
`HIGHEST` adds ~0–1.2 s of that.
⇒ **I do not expect the slowdown to be visible in a stage's wallclock.** If it is, say so — it would mean the conv
shapes are hitting a much worse FP32 path than this FLOP budget predicts, and that is worth knowing.
⛔ **Training is not touched.** `train_encoder` / `train_decoder` / `train_joint` do not route through
`encode_images`; only the **frozen** encode path is pinned.

## 5. ⭐ The falsifiable prediction — stated BEFORE the run

With `ENCODE_PRECISION = "highest"` in force on an L40S, re-running the `store` preflight on the same cache:

- **PREDICTED:** `_assert_cache_reencodes` **passes**. `max_ulp_delta ≤ 1.0` (bound), `frac_differing` collapses
  from **20.85 %** to **≲ 1e-4** (order the gate's own derivation predicts for a pure float32 accumulation
  reorder; the historical single-element observation was 7.6e-6), `max_abs_delta` falls from **0.00390625** to at
  most one float16 ULP *at each element's own magnitude*, `n_nonfinite = 0`.
- ⛔ **IF ~20 % OF ELEMENTS STILL DIFFER, TF32 WAS NOT THE CAUSE.** Then this fix is wrong or incomplete and
  ⛔ **nobody widens the tolerance.** It comes back for a different diagnosis. Named next suspects, in order:
  (a) cuDNN picking a reduced-precision algorithm regardless of `precision_config` — test with
  `NVIDIA_TF32_OVERRIDE=0` and `XLA_FLAGS=--xla_allow_excess_precision=false`;
  (b) the **GroupNorm reductions**, which `Precision` does not govern at all;
  (c) a genuinely different encoder or a cache/checkpoint mismatch the parameter hash cannot see.
  ⛔ I deliberately did **not** put (a)'s env vars on the submit line: they would confound the test of whether the
  local, explicit precision is sufficient on its own. Keep them for leg 2 if leg 1 fails.
- **Intermediate outcome that is still informative:** delta drops by orders of magnitude but lands just over 1 ULP
  → TF32 *was* the dominant term and something smaller remains. Report the number; still do not widen the bound.

**The log line to look for in the job output, immediately above the gate's own line:**
```
  ⭐ encode precision: requested=highest resolved=HIGHEST backend=gpu device=NVIDIA L40S n_devices=1 (jax 0.9.0)
```
⛔ **If `resolved=` is not `HIGHEST`, or `backend=` is not `gpu`, the leg tested nothing** — read that line before
reading the verdict.

## 6. ⭐ The submit line (one line, `-p gpuL`, carrying `GEN_GIT_COMMIT`)

`STAGE=store` is the sharp leg: its preflight is the **earliest and cheapest** place the gate fires (13 other
modules call `_assert_cache_reencodes` too, but all of them are downstream of `store`). One arm, one seed is
enough to fire it, and the gate runs before any arm does, so a refusal costs minutes rather than hours. Push the branch with `scripts/csf3/push_repo.sh` first (it rsyncs without `.git`, which is
why the commit is inlined below).

```
cd ~/scratch/CHLU && mkdir -p logs && sbatch -p gpuL --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=store,ARMS=clu,SEEDS=0,GEN_GIT_COMMIT=70f3bf613538a9f8e9761a92d751d8e6f45d8e49" -a 0-0 -t 02:00:00 scripts/csf3/job_gpu_gen.sh
```

⚠ `job_gpu_gen.sh` has `#SBATCH -p gpuA` and `-c 8` baked in; `-p gpuL` overrides the partition but **not** the
core request, so if `gpuL` enforces a different cores-per-GPU limit the submit will need `-c` adjusting too. The
in-job GPU preflight only asserts `jax.default_backend() == "gpu"`, so it is partition-agnostic.

### Cluster legs
| leg | status |
|---|---|
| L40S `store` preflight re-encode under forced precision | ⛔ **NOT RUN** |
| A100 (`gpuA`) cross-check | ⛔ **NOT RUN** |
| any GPU cache build / cache extension under forced precision | ⛔ **NOT RUN** |

⛔⛔ **CSF3 needs 2FA and this laptop is Apple Silicon (JAX CPU only). I could not confirm the fix and I do not
claim to have.** What is confirmed here is (i) the requested precision reaches the real convolution equations,
(ii) it was absent before, and (iii) it costs the CPU nothing and moves not one byte.

## 7. Reconciliation list (⚠ needs an owner — Hub to assign)

**One item.** Criterion 3 asks for the precision in the **preflight record**. It reaches the log (§5) and
`feature_cache.json`, both mine. The store run manifest's preflight block is assembled in
`chlu/experiments/exp_gen_store.py`, which I was forbidden to touch. Exact one-line diff, for whoever owns that file:

```python
# chlu/experiments/exp_gen_store.py, in the `manifest = {...}` literal (~line 721)
         "cache_reencode_check": reenc,
+        # which ARITHMETIC ran the re-encode, not just which encoder
+        "encode_precision": encode_precision_record(),   # add to the gen_bridge import block
```

The same one-liner would have to be repeated in the **other 13 modules that call the gate** (`exp_gen_atom_mean` · `attribution` · `baselines` · `cell` · `certificate` · `classifier` · `codebook_alloc` · `codebook_exponent` · `compose` · `dream` · `dynamical_read` · `variance_eq` · `wells_route_b`) if the Hub wants it everywhere. The alternative — fold it into the `stats` dict `_assert_cache_reencodes` already returns — is **one line in one place and reaches all 14 callers at once**. ⭐ That is the better fix, and it is a one-line addition to the one function I was explicitly told to leave alone, so the Hub decides, not me.

## 8. How I verified

All commands from the `../CHLU-prec` worktree, **main venv** (`/Users/user/Desktop/CHLU/.venv`, JAX **0.9.0** —
no `uv sync`, so the w6 worktree-venv drift does not apply), `PYTHONDONTWRITEBYTECODE=1`,
`CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`.

| suite | result |
|---|---|
| `tests/test_gen_bridge.py` (+5 new tests) · `test_gen_store.py` · `test_stage_artefacts.py` · `test_gen_bridge_wide.py` · `test_gen_bridge_dsprites.py` | **161 passed · 0 skipped · 0 failed** (175.83 s; `-q -p no:randomly --no-cov -rs`) |
| `ruff check chlu/core/gen_bridge.py tests/test_gen_bridge.py` | All checks passed |
| `ruff format --diff` on my added lines | clean (⛔ the two files are *not* format-clean overall on `main`; I did **not** reformat anyone else's lines) |
| `git diff main..HEAD -- chlu/experiments/exp_gen_store.py` | **empty** ✅ |

Tests added (`tests/test_gen_bridge.py`, section 6b):
- `test_encode_path_convolutions_carry_the_requested_precision` — ⭐⭐ the load-bearing one. Intercepts
  `eqx.filter_jit` to capture the **shipped** callable (not a reconstruction of it), takes `jax.make_jaxpr` of it,
  and asserts every `conv_general_dilated` equation carries `(HIGHEST, HIGHEST)`. **Verified to fail if the fix is
  dropped**: the identical probe run against `main`'s tree returns `[None, None]`.
- `test_encode_precision_does_not_move_the_numbers_on_cpu` — the laundering control as a test.
- `test_encode_precision_record_names_the_precision_backend_and_device`
- `test_encode_images_announces_the_precision_once_per_process` — including that it is *once*, not per block.
- `test_feature_cache_manifest_records_which_arithmetic_wrote_it`

### Artifacts (`.claude/outputs/gen-encode-precision/`)
| file | what |
|---|---|
| `encode_probe.py` | the CPU bit-identity + wall-time probe; run under `PYTHONPATH=<tree>` against either tree |
| `before_features.npy` / `after_features.npy` | the two 64x(128,4,4) float32 tensors — `cmp` clean |
| `before_probe.json` / `after_probe.json` | their digests, sums, and timings |
| `precision_cost.py` | HIGHEST vs DEFAULT, compiled once — the isolated arithmetic cost |
| `jaxpr_precision_probe.py` | ⭐ the negative control: prints the real conv equations' `precision`; `[None, None]` on `main`, `[(HIGHEST, HIGHEST), …]` here |

### Flag provenance
| | |
|---|---|
| base | `main @ dcd667e` ("retire three stale 'bit-identical' claims") |
| branch commit | `70f3bf613538a9f8e9761a92d751d8e6f45d8e49` |
| seeds | encoder `jax.random.PRNGKey(0)`; images `np.random.default_rng(1234)`; cost-probe images `default_rng(1234)` |
| non-default flags | **`ENCODE_PRECISION = "highest"` — the only one.** Encoder config is the shipped `GenBridgeConfig` default (`enc_channels=[32,64,128,128]`, `enc_kernel=4`, `enc_stride=2`, `enc_groups=8`), `batch=32`, `cache_dtype` untouched |
| env | JAX 0.9.0, backend `cpu`, `device_kind` `cpu`, 1 device, macOS/Apple Silicon, main venv |
| ⛔ device the defect lives on | NVIDIA L40S (Ada sm_89) — **not available here** |

## 9. Git footprint
- **Branch** `agent/experiment-engineer/gen-encode-precision`, worktree `../CHLU-prec`, base **local `main`**
  (⛔ not `origin/main` — still frozen at `40c2f31`). `main` is an ancestor of the branch, so the rebase is a no-op;
  no rebase performed, no conflicts.
- **Commit** `70f3bf6` — *"pin the frozen encoder's arithmetic, not the gate's tolerance"*, 2 files, **+209 / −4**.
- **Files touched:** `chlu/core/gen_bridge.py` (**+99 / −4** — `encode_images`, the new `ENCODE_PRECISION` /
  `encode_precision_record` / `_announce_encode_precision` helpers, and two lines in `build_feature_cache`:
  `log=log` on its `encode_images` call and the `encode_precision` manifest key),
  `tests/test_gen_bridge.py` (+110, additive: two imports and one new section).
- **Nothing else.** ⛔ Not `exp_gen_store.py`, not `exp_gen_pretrain.py` / `exp_gen_bridge.py` /
  `exp_gen_wells_route_b.py`, not `chlu/config.py`, not `scripts/**`.
- No pushes, no PR. Worktree `../CHLU-prec` **left in place** for Hub review; branch ref verified visible from the
  main checkout (`git log main..agent/experiment-engineer/gen-encode-precision` → `70f3bf6`).
- No collision: `git status` in the shared checkout was clean on arrival and I never edited it.

## 10. Open questions / risks
1. ⛔⛔ **The fix is untested where it matters.** Treat it as a hypothesis with a stated prediction until the L40S
   leg runs. §5 says in advance what refutes it.
2. ⚠ **`Precision` governs matmuls and convolutions, not reductions.** GroupNorm's mean/variance are unpinned.
   That is the right residual scale (~1e-7 relative, the gate's own tolerated regime) — but it is a *reason the
   prediction could land just over 1 ULP rather than far under it*, and it is suspect (b) in §5.
3. ⚠ **`build_feature_cache` records the precision; it does not enforce it.** A cache written by an older commit
   carries no `encode_precision` key, and nothing yet refuses to *extend* such a cache on a different backend. The
   record makes it visible; making it fatal is a separate decision (and would live in the gate, which is not mine).
4. ⚠ **`encode_images` recompiles on every call** (a fresh `filter_jit` wrapper each time — ~130 ms of the ~143 ms
   per call here). Pre-existing on `main`; I preserved it deliberately so the cost comparison is not confounded. It
   is free money for someone with a mandate to take it.

## Proposed handover updates (for the Hub)

**§7 Known Issues — add:**
> **§7.2x — the frozen encoder's arithmetic was never pinned (found 2026-09-16, fix on
> `agent/experiment-engineer/gen-encode-precision` @ `70f3bf6`, ⛔ UNCONFIRMED — GPU leg NOT RUN).** Nothing in
> `chlu/` set a matmul precision, and `eqx.nn.Conv2d` passes no `precision` to `lax.conv_general_dilated`, so every
> convolution resolved to XLA's default — **TF32 on Ampere/Ada**. The feature cache was written on CPU in true
> float32. On an L40S the frozen encoder therefore failed to re-encode its own cached rows: `max |delta| = 2^-8 =
> 4181 float16 ULP` on 20.85 % of elements with the encoder parameter hash **matching**. ⛔ The stake was never the
> failing preflight — a GPU run that *extends* a cache writes features inconsistent with the ones already in it and
> nothing downstream would notice. `encode_images` now runs under a **local, trace-time**
> `jax.default_matmul_precision("highest")` (⛔ no import-time global — §7.23 unaffected) and records
> precision/backend/device in the log and in `feature_cache.json`. **CPU output is byte-identical to `main`**
> (sha256 `971983389c…`), so nothing already cached or hashed is in doubt. **The gate `_assert_cache_reencodes` was
> deliberately left untouched and no tolerance was widened** — it is the instrument that will say whether this
> worked. ⭐ The prediction is registered in `.claude/outputs/gen-encode-precision.md` §5: **if ~20 % of elements
> still differ, TF32 was not the cause and it comes back for a different diagnosis — nobody widens the bound.**

**Two decisions for the Hub:**
1. **Who lands the `cache_reencode_check` precision field** (§7 above) — the better version is one line inside
   `_assert_cache_reencodes`'s returned `stats`, which is a file I was told to leave alone.
2. **Whether a cache lacking `encode_precision`, or carrying a different one, may be extended at all** (§10.3).
   Today it is recorded, not enforced.
