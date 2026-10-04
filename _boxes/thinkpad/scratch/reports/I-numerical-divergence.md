# SPOKE I — Do numbers computed on THIS box match the Mac's?

**Report path:** `_hub_scratch/reports/I-numerical-divergence.md`
**Workspace:** `/c/Users/PratikJawahar/chlu/chlu-hub-bundle` — **read-only, nothing patched.**
**Core budget honoured:** `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` (plus a deliberate `=1` control).
**No scored runs. No hub. No GPU. No cloud spend. No test, fingerprint or source file edited.**

---

# ⭐ THE ANSWER: **YES — with one named, bounded exception.**

**This machine can produce numbers comparable with the Mac's.** The alarm in `CONTEXT.md` —
*"NUMBERS COMPUTED HERE DO NOT MATCH THE MAC'S — UNRESOLVED"* — does not survive contact with
the evidence. **I could not confirm a single genuine cross-platform numerical divergence in the
generative line.** All three "recorded fingerprint no longer reproduces" failures have been run
down to a specific, non-numerical cause:

| # | Failing test | Real cause | Numerical? |
|---|---|---|---|
| 1 | `test_gen_encoder_unfreeze::test_the_baseline_decoder_recipe_still_builds_the_same_decoder` | **Test-suite contamination.** `jax_enable_x64` leaks process-globally from another test module → `equinox` builds **float64** weights. Reproduced exactly. | **No** |
| 2 | `test_gen_encoder_unfreeze::test_frozen_path_is_bit_identical…` (`enc_hash` leg) | **Same x64 leak.** Reproduced exactly. | **No** |
| 2b | …same test, **`store_hash` leg** (the real residual) | ⚠ **Genuine bit-level float32 divergence** — but the fingerprint is over-specified and is **not stable even on this one machine**. See §4. | **Yes, bounded** |
| 3 | `test_gen_wide_split::test_montero_split_is_unchanged_by_the_wide_condition` | **11 bytes of ZIP container metadata.** `zipfile.ZipInfo.create_system` = `0` on Windows vs `3` on Unix. **Every data byte and every CRC is identical.** | **No** |
| 4 | `test_blocks.py` ×5 "shares a **bit-identical** shell" | **`ModuleNotFoundError: No module named 'resource'`.** They fail at `import`; they never reach arithmetic. | **No** |
| 5 | `test_gen_wells::…penalises_dependence` (`-1.14e-08 >= 0.0`) | Float32 dust (**0.096 ULP of 1.0**) against a **zero-tolerance** assertion. **Passes in isolation on this box.** | **No** |
| 6 | `test_kt::test_reduced_rho_s_round_trip_exact` (`0.82504` vs `0.82448`) | **Metropolis Monte-Carlo chain decorrelation.** Delta = **1.36σ of the test's own quoted standard error.** Statistical, not numerical. Not on the generative path. | **No** |

### The one-line practical verdict
> **A hub that LOADS frozen artefacts and renders is exactly reproducible here — 37/37 recorded
> hashes match, including the 1.9 GB `features.npy` sha256.** A pipeline that **rebuilds a
> k-means++ store from a recipe** will get a bit-different store; that path is not
> bit-reproducible *anywhere*, including Mac-to-Mac.

`CONTEXT.md`'s suspected distinction — *"the failing tests rebuild from a recipe; a hub loads
frozen pickles and renders"* — **is correct, and it is the whole answer.**

---

# 1. LOAD vs COMPUTE — ⭐ 37 MATCH, 0 DIFFER

Script: `_hub_scratch/I-probe/load_check.py` · Full log: `_hub_scratch/I-probe/load_check.log`
Read-only: opens files, writes nothing.

Every frozen artefact was opened through its real loader (`chlu.utils.checkpoints.load_model` →
`pickle.load` + `eqx.combine`) and its `param_hash` / `param_count` compared to **the hashes the
Mac recorded in the shipped manifests**.

**`winner_hashes.json` and `classifier_gate.json` — the two the brief called the single most
valuable check — both reproduce exactly.**

```
A. WINNER DECODERS  (.claude/outputs/gen-decoder-lift/winner/)
   seed 0  raw .pkl sha256      MATCH  bcf39c42bc7885d81b21847a4ae13324ee5d49991f96ca5fc9cc79a6e5295153
   seed 0  param_hash(loaded)   MATCH  01ef91f75eeaae7be8d83a220c9cc6629120345357da9033db22375c278ab196
   seed 0  param_count          MATCH  1516227
   seed 0  vs classifier_gate   MATCH  (same hash, independent record)
   seed 1  all four             MATCH  001011bc603a7cb914cfaa592f43087685727431902ffc133755de30dbb21e4f
   seed 2  all four             MATCH  2d7eea75add770441fcd0068ffc7a3dcc0b02cd16d6a31b3e095f47b64aa2455

B. FROZEN ENCODER / DECODER / PROBE  (gen-feature-bridge, bridge_manifest.json)
   seeds 0,1,2 x {encoder_param_hash, feature_cache.encoder_param_hash, feat_shape,
                  decoder_param_hash, decoder_param_count, probe_param_hash,
                  probe_param_count, winner_hashes.encoder_param_hash_it_was_fit_against}
   -> 24/24 MATCH   e.g. encoder s0 = 98d326028316ae2cd6b960b26a922cd47fe8a2c9e73199fb1ba2927864a29c0d

E. THE FEATURE CACHE + A REAL FORWARD PASS
   features.npy      1,966,080,128 B  (manifest: 1,966,080,128 B)
   raw sha256        MATCH  d6728f01d102b0d0301198dc617f40bfb8ddb6c5b4a12b21a1ec08e1d974f2e2
   dtype float16, shape (480000, 128, 4, 4)
   decoder(8 cached rows) -> (8, 3, 64, 64) float32
     sha256(render) = a90903030ccbb879efccb1688a22f12d4d036f356dd598ef6c3145a99c04ff59
     mean = 0.4535047709941864   std = 0.44089126586914062

SUMMARY: 37 MATCH, 0 DIFFER
```

**What this establishes.** Pickle round-trip, `eqx.combine`, array dtypes, endianness, tree-leaf
ordering and `np.asarray(...).tobytes()` are all bit-identical Windows-x86-64 ↔ Darwin-arm64.
The 1.9 GB float16 cache the Mac computed is byte-perfect here. **The inference path is stable.**

⚠ **Honest limit.** The render sha `a90903…` is a number *I computed here*; there is no Mac-side
record of it to compare against. The **inputs** to that render (decoder params, cached features)
are proven bit-identical, so only XLA's conv lowering is unverified — see §4 and §11-Q1.

⚠ `classifier_manifest.json` (§C) carries **no** `param_hash`/`param_count` keys, so the external
classifiers could not be hash-checked. Their keys are
`{classifier, condition, dataset, elapsed_seconds, frozen, gate, leg3, no_leak, pretext, real_reads, rows, seed}`.
The `.pkl`s load cleanly; `classifier_gate.json` cross-checks the **decoder** hashes instead, and
those match (§A).

---

# 2. THE SMALLEST REPRODUCER — ⭐ and it already answers H1 vs H3

`_hub_scratch/I-probe/repro.py` — self-contained, prints a ladder from raw PRNG up to the three
fingerprints. **Run it on the Mac unmodified and diff the logs.** Here it yields:

```
python 3.12.10 | Windows-11-10.0.26200-SP0 AMD64 | jax 0.9.0 jaxlib 0.9.0 numpy 2.4.1 equinox 0.13.4

L0b split(PRNGKey(20260914),3)  aec05c6004b952043267fd53836e825fc5d0285d253e8d19cc61e346f81b306e
L0c uniform(ek,(4,4))           bd5c0433bba2632bf833d8e132cbbeb1635fbfd437a72152dd240b5d4daaf08c
L0d normal(ek,(4,4))            1e570f706130b25b64e502a611c8cd91f6f34bfd707193de59776b87a2d31139
L1a eqx.nn.Conv2d(3,8,4,s2,p1)  f2d03b45f8fc28b5efee217266bad39c208d9e69776fcbf6f72c92a3e88b12a3

L2a enc_hash            7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693   ⭐ MATCHES Mac
L2b dec_hash_baseline   457977e44d7b2a6ac30742fa5632aad8d31cfab67c71c61e24265fb50b38fea3   ⭐ MATCHES Mac
L2c dec_hash_v1         40236b47ba4f53c913b5ac247af50abad16fd742d04c54fdacc3f588a34251df   ⭐ MATCHES Mac

L3a imgs sha (numpy PCG64)      a1061d631fa040557be553e2fc044868e37ed3959ad046f77b7fd5fae7f27657
L3b Z sha (FORWARD PASS)        1df1e9677c07bcba44ef2a2ac3279419017c15197f42468f1ca48e81c4d25e87   <- UNKNOWN on Mac
L3c labels                      identical to the shipped grouping
L3d store_hash          6ac774f838beb5ea5b33507c91b39019da3b3223470a428390e8b58cf562ac6d   ⛔ != a2a916f4…
        well_sizes [8,23,225] = expected · codes_sum 84 = expected
        bits_per_read 6.0 = expected · store_bytes 4096 = expected
```

⭐ **Three of the four fingerprints reproduce bit-exactly on this box.** The `enc_hash`
`094ae9c3…` in `_F-suite.log` is **not what this machine computes** — it is what the machine
computes *after another test module has switched JAX into float64* (§3).

**The only unmatched quantity is `store_hash`, and `store_hash` is a hash of the forward-pass
output `Z`** — `_kmeanspp_init` returns `Z[idx].astype(np.float32)`, i.e. literal selected rows
of `Z` (`chlu/core/gen_store.py:701`), which `GenStore.__init__` stores verbatim
(`chlu/core/gen_store.py:418`). So `store_hash` **is** a bit-exactness test of
`jax.vmap(enc)(imgs)` on the CPU backend, wearing a different name.

### ⭐ What the Mac must run to settle the last question — copy-pasteable
```bash
export CHLU_ROOT=/path/to/bundle && cd "$CHLU_ROOT" && source ./env_hub.sh
python /path/to/repro.py > mac_repro.log
```
Then report **`L3b Z sha`** and **`L3d store_hash`**. Expected readings:
- **`L3b` matches `1df1e967…`** ⇒ even the conv forward pass is bit-identical across ISAs, and the
  `a2a916f4…` fingerprint is stale w.r.t. some other change. Full parity.
- **`L3b` differs** ⇒ confirms §4: float32 conv output is not bit-portable — which is already
  demonstrably true *within* this single machine, so it indicts the fingerprint, not the box.

The file is at `_hub_scratch/I-probe/repro.py` (copy it; it only needs `$CHLU_ROOT` set).

---

# 3. ⭐ FAILURES 1 & 2: `jax_enable_x64` LEAKS PROCESS-GLOBALLY — reproduced exactly

`chlu/core/gen_bridge.py:280` warns about this in the source itself:
> *"§7.23: a module-scoped `jax_enable_x64` fixture elsewhere in the suite makes eqx build float64
> weights"*

**24 test modules touch `jax_enable_x64`; several call `jax.config.update("jax_enable_x64", True)`
at MODULE IMPORT**, e.g. `tests/test_friction_field.py:21`, `tests/test_kt.py:27`. JAX's config is
**process-global**, so under `pytest-xdist` a worker that imports one of those modules before
`test_gen_encoder_unfreeze` builds **float64** encoder/decoder weights — different bytes *and* a
different `dtype` string inside `param_hash`.

Forced locally (`_hub_scratch/I-probe/x64_leak.log`):

```
jax_enable_x64=False  weight dtype=float32
   enc_hash          = 7dcf753fe48ce80fd2416150b316017afbd4a681950fa2d1b821212eefe7e693  == FROZEN ✔
   dec_hash_baseline = 457977e44d7b2a6ac30742fa5632aad8d31cfab67c71c61e24265fb50b38fea3  == FROZEN ✔

jax_enable_x64=True   weight dtype=float64
   enc_hash          = 094ae9c3941b8fb69234199213d22a061dd47d6bfc2bfa70d64f6a974fb95646  ⭐ == the SUITE FAILURE
   dec_hash_baseline = a73a12cd18a83e672c19fed05aada8d807cf11bcd927493d63ed5b2335abe20c  ⭐ == the SUITE FAILURE
```

⭐ **Both suite-failure hashes are reproduced exactly, digit for digit, by flipping one config
flag.** This is a closed question. It is a **test-isolation defect, not a platform divergence** —
and it is **latent on the Mac too**; the Mac simply did not hit the ordering that triggers it.

**Isolated re-run on this box** (`-p no:randomly --no-cov`):
```
tests/test_gen_encoder_unfreeze.py tests/test_gen_wide_split.py
  ->  2 failed, 43 passed, 2 skipped in 23.24s
```
Down from 3 failures. `test_the_baseline_decoder_recipe_still_builds_the_same_decoder` **passes**,
and `test_frozen_path…` no longer fails on `enc_hash` — it advances to the `store_hash` leg.

⛔ **Not patched.** `tests/` is pinned content.

⚠ **The Mac/Windows comparison in `CONTEXT.md` is not like-for-like.** The Mac's *"29 passed, 1
skipped"* is those files run **alone**; the Windows *"117 failed"* is the **full suite** under
`xdist`. Two of the three headline divergences are artefacts of that difference, not of the
platform. (Note the counts still differ — 30 Mac tests vs 47 here for the two files; see §11-Q3.)

---

# 4. ⛔ THE ONE GENUINE DIVERGENCE: `store_hash` — and it is **not stable on this machine either**

`store_hash` hashes selected rows of the float32 conv output `Z`. I reproduced the cross-machine
situation **locally**, with no Mac, by computing the *same* `Z` four numerically-equivalent ways
(`_hub_scratch/I-probe/z_variants.log`):

```
-- vmap (what the test does)     Z sha 229afaf0…  ULP diff 0     store_hash 6ac774f8…
-- jit(vmap)                     Z sha 229afaf0…  ULP diff 0     store_hash 6ac774f8…
-- python loop, one image        Z sha 7cc7f0ea…  96/8192 floats differ, max|Δ|=3.576e-07
                                                                 store_hash a8e190b8…   <- MOVED
-- vmap on 2 half-batches        Z sha 7cc7f0ea…  96/8192 floats differ, max|Δ|=3.576e-07
                                                                 store_hash a8e190b8…   <- MOVED
```
In every variant: `well_sizes [8,23,225]` ✔, `codes_sum 84` ✔, correlation `labels` identical ✔.

### ⭐ The finding
**Changing only the batching of an encoder forward pass — same machine, same binary, same
weights, same inputs, same second — produces a different `store_hash`.** The `a2a916f4…`
fingerprint does not pin a scientific quantity; it pins **one particular XLA CPU lowering of a
float32 convolution**. XLA does not guarantee that across batch shape, let alone across ISA
(AVX-512 / Intel Core Ultra 9 vs NEON / Apple Silicon).

**Magnitude: last-bit, benign.** `max|Δ| = 3.576e-07` on activations of order 0.6 ≈ **3 float32
ULPs** — textbook reduction-order noise. (The raw ULP *count* peaks at 1487, but only on
near-zero post-ReLU outputs where ULP counting is meaningless; the absolute delta is the honest
measure.) Crucially, **everything scientific downstream is invariant**: the channel grouping, the
well sizes, the address codes and the byte ledger are identical across all four variants.

⛔ **This is an over-specified fingerprint, not a broken machine.** Reported, not patched.

---

# 5. FAILURE 3: `test_gen_wide_split` is **11 bytes of ZIP metadata** — zero numerical content

The expected sha `5f1fce25…` **is the shipped `.claude/data/gen/shapes3d/factor_index.npz`**
(verified: `sha256 = 5f1fce25c40cceae6de2d00fd2145c4cdc73dcc38cee7f6545cbcaea3ce78f49`). So the
Mac's own artefact is on disk here and I diffed against it directly — **no Mac needed**.

First, `np.savez` is byte-deterministic: numpy pins every ZIP entry to `date_time (1980,1,1,0,0,0)`,
so no timestamp noise (`_hub_scratch/I-probe/npz_probe.py`).

**Member-by-member (`npz_diff.log`) — all 11 members identical, same order, same file size:**
```
train_rows          int64 (479928,)   identical=True      cardinalities   int64   (6,)  identical=True
test_rows           int64 (72,)       identical=True      factor_names    <U11    (6,)  identical=True
meta                <U130 ()          identical=True      values__*  (6 arrays) identical=True
raw compressed member bytes: all 11 equal=True   (train_rows.npy 3,839,552 B both)
```

**Byte-level (`zip_hdr.log`) — exactly 11 differing bytes out of 3,844,150:**
```
n differing bytes: 11 of 3844150
offsets: [3843439, 3843499, 3843558, 3843621, 3843683, 3843737, 3843804, 3843870, 3843938, 3844001, 3844064]
  off 3843439 (0x3aa56f): gen=0x00 ship=0x03   ctx=b'@PK\x01\x02-\x00-\x00\x00\x00\x00'
  ... 11x, one per central-directory entry (PK\x01\x02), all 0x00 vs 0x03

ZipInfo field diff, every member:
   create_system      gen=0    ship=3    DIFFERS
   create_version=45  extract_version=45  external_attr=25165824  flag_bits=0
   date_time=(1980,1,1,0,0,0)  compress_type=0  CRC  -> ALL EQUAL

sys.platform = win32 -> zipfile.ZipInfo("x").create_system = 0
(CPython zipfile.ZipInfo.__init__: create_system = 0 if sys.platform == 'win32' else 3)
```

⭐ **Every CRC matches. Every data byte matches. The split is bit-identical.** The failure is
CPython's `zipfile` stamping the "host OS that made this archive" field. It is **100% deterministic
for any Windows machine** and would reproduce on Windows + Python 3.11, 3.13, any CPU.

⛔ Not patched. Flagged for the humans: `save_index`'s comment says *"deterministic bytes"*
(`chlu/data/gen_splits.py:1376`) — it is deterministic **per OS family**, not across them.

---

# 6. THE TWO NUMERIC ASSERTIONS — **do not average them into "small"**

### 6a. `test_gen_wells.py:559` — `-1.1415797196434596e-08 >= 0.0` — **float32 dust, and it passes in isolation**
```
mean_pairwise_mi_nats = float(Array(-1.1415797e-08, dtype=float32))
absolute value  1.14e-08
float32 eps     1.1921e-07
-> the value is 0.096 ULP of 1.0 -- an order of magnitude BELOW one float32 ULP
```
Mutual information is analytically ≥ 0, but this estimator is a **difference of entropies in
float32**; at this magnitude its sign is arbitrary. The assertion carries **zero tolerance**.

⭐ **Re-run in isolation on this box: `1 passed in 3.39s`.** It is a suite-interaction artefact,
not a platform divergence. A standalone probe of the same quantity (at smaller `D,K,M,N,F` than
the test's `48,4,6,400,3`, so not the test's exact configuration) showed it moving from
`-8.4e-09` to `+6.5e-08` purely by flipping `jax_enable_x64` — i.e. it straddles zero and its sign
is decided by float32 noise. ⚠ I did **not** establish which specific suite interaction flipped it
in `_F-suite.log`; I established only that it is float32 dust and that it passes here in isolation.

### 6b. `test_kt.py:61` — `0.8250428728431519` vs `0.8244828012941149` — **Monte-Carlo, not arithmetic**
```
absolute delta   5.600715e-04
relative delta   6.793005e-04
in float64 ULPs  5.0e+12          <- correctly noted as far too large for summation order
tolerance asked  abs=1e-15, rel=0
```
⭐ **But the right denominator is not the ULP — it is the test's own standard error.** Two lines
below, the same test asserts `rf.std()/sqrt(3) == 0.00041260235470954687`. So:
```
delta / (test's own quoted standard error) = 5.600715e-04 / 4.1260e-04 = 1.357 sigma
```
`run_cell` (`chlu/experiments/kt/reduced_xy.py:226`) is a **Metropolis Monte-Carlo** XY-model
simulation — 3 seeds × 4 walkers × (1500 therm + 4000 meas) sweeps over L²=64 sites, ≈1.4 M
accept/reject decisions of the form (line 71):
```python
acc = rng.random(m.sum()) < np.exp(-beta * dE)      # dE built from np.cos(...)
```
`np.exp` and `np.cos` are SIMD-vectorised and differ by ~1 ULP between AVX-512 and NEON. **One
flipped accept/reject decorrelates the entire chain**, and over 1.4 M decisions at least one flip
is essentially certain. The two runs are then *different but statistically equivalent* MCMC
realisations, and a 1.36σ separation is exactly what that predicts.

⭐ **Consistent with this:** only **one** of the four `test_kt` round-trip tests failed
(`FAILED tests/test_kt.py::test_reduced_rho_s_round_trip_exact` is the sole `test_kt` entry in
`_F-suite.log`). `test_winding2d_round_trip_exact`, same MCMC machinery but reporting robust
near-integer statistics (`tau_med` 16.0/27.0/29.0), **passes** — precisely what chain
decorrelation with stable coarse statistics looks like.

⚠ **Honest limit:** this is a mechanism argument backed by the failure pattern and the σ
arithmetic, not a direct demonstration of a flipped accept/reject. I did not re-run the MCMC to
bisect it — that would cost minutes of CPU for a test that is off the generative path.

⛔ **Not on the generative path.** `test_kt` is the KT/XY memory-phase line, unrelated to
`hub/generation-table` or `hub/composition`. The docstring's claim that the numerics *"ROUND-TRIP
bit-exactly"* is only achievable on the exact machine that produced the reference. Reported,
not patched.

---

# 7. DETERMINISM AND THE ELIMINATED HYPOTHESES

### 7a. ⭐ H1b (interpreter minor version) — **ELIMINATED**
Built `/c/clu-venv/py311-probe` **outside the bundle** via `UV_PROJECT_ENVIRONMENT`, exactly as
the brief specified. The working `.venv/` was not touched.
```bash
uv venv --python 3.11 /c/clu-venv/py311-probe                                  # -> cpython-3.11.16
UV_PROJECT_ENVIRONMENT=/c/clu-venv/py311-probe uv sync --extra gen --group test --frozen
# both exit 0;  uv at C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\Scripts\uv.exe
```
```
python 3.11.16 | Windows AMD64 | jax 0.9.0 jaxlib 0.9.0 numpy 2.4.1 equinox 0.13.4
  -> EVERY line of repro.py byte-identical to Python 3.12.10, L0a through L3d,
     including store_hash 6ac774f838beb5ea…
  -> pytest isolated: 2 failed, 43 passed, 2 skipped in 21.88s   (identical failures, identical hashes)
```
⭐ **Pinning the hub to Python 3.11 would change nothing.** Do not spend effort there.
⚠ Minor: uv supplied **3.11.16**, not the Mac's 3.11.13 — same minor, different patch. Irrelevant
to the conclusion, since 3.11.16 and 3.12.10 already agree bit-for-bit.

### 7b. Determinism and thread count — **both ELIMINATED**
Three runs, `OMP/MKL/OPENBLAS = 4, 1, 4` (`determinism.log`):
```
run 1 (t=4) | run 2 (t=1) | run 3 (t=4)   ->  ALL TWELVE reported hashes byte-identical,
                                              including L3b Z sha and L3d store_hash
```
⭐ **The hash does not move with thread count.** So the more serious diagnosis the brief warned
about — unseeded RNG, dict ordering, thread-dependent reduction order — **is ruled out.** Results
on this box are perfectly repeatable, which is what makes the §4 batching result meaningful.

### 7c. H2 extended — equinox also matches
The brief eliminated jax/jaxlib/numpy. `equinox` is the library that *actually constructs* the
hashed parameters, so I checked it too: **0.13.4 here**, and `uv.lock:551-552` pins
`equinox==0.13.4` with no platform marker. Same on both machines. H2 fully closed.

---

# 8. THE NUMERICAL ENVIRONMENT (brief §3)

| Item | Value |
|---|---|
| python | 3.12.10 `[MSC v.1943 64 bit (AMD64)]` (+ 3.11.16 probe, identical results) |
| platform | `Windows-11-10.0.26200-SP0` · AMD64 · `Intel64 Family 6 Model 197 Stepping 2, GenuineIntel` (Core Ultra 9) |
| jax / jaxlib / numpy | **0.9.0 / 0.9.0 / 2.4.1** — identical to the Mac |
| equinox | **0.13.4** — pinned in `uv.lock`, identical to the Mac |
| `jax.devices()` | `[CpuDevice(id=0)]` |
| `jax.default_backend()` | `cpu` |
| `jax.config.jax_enable_x64` | `False` at import ⚠ **but 24 test modules mutate it process-globally — see §3** |
| `jax.config.jax_default_matmul_precision` | `None` |
| `XLA_FLAGS` / `JAX_*` env vars | **none set** (only `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `TF_CPP_MIN_LOG_LEVEL=1`) |
| `chlu.__file__` | `C:\Users\PratikJawahar\chlu\chlu-hub-bundle\chlu\__init__.py` ✔ bundle, not the OneDrive checkout |

**Matmul precision** — the brief flagged it as the classic cross-platform divergence source. It is
**not** the cause here: `jax_default_matmul_precision` is `None` (default) on both machines and
**the code sets it nowhere**. It also cannot explain §3 or §5, and §4 is a convolution, not a
matmul. Ruled out.

**numpy BLAS backend** — the standard PyPI wheel **OpenBLAS** build. ⚠ The Mac's arm64 wheel links
a different OpenBLAS kernel set; this, alongside XLA's own conv lowering, is the most likely
residual source of the §4 last-bit deltas. Neither is configurable away.

---

# 9. THE CLI BLOCKER — confirmed, with a working route

**Confirmed exactly as `CONTEXT.md` states. Not re-litigated, just re-verified.**
```
$ python -c "import chlu.cli"
  File "...\chlu\training\train_cluformer.py", line 42, in <module>
    import resource
ModuleNotFoundError: No module named 'resource'

$ chlu --version           -> identical ModuleNotFoundError
```
**The library itself is fine:**
```
OK   chlu                               OK   chlu.core.gen_bridge
OK   chlu.experiments.exp_gen_cell      OK   chlu.core.gen_store
OK   chlu.experiments.exp_gen_atom_mean OK   chlu.data.gen_splits
OK   chlu.utils.checkpoints             OK   chlu.data.shapes3d
```

⭐ **This also resolves the `test_blocks.py` ×5 failures the brief listed as numerical.** They are
the *same* `import resource` error: `tests/test_blocks.py:287` does
`from chlu.training.train_cluformer import assert_shared_shell_identical`, which raises
`ModuleNotFoundError` before any arithmetic runs. The phrase "bit-identical shell" is only in the
test *name*. **Nothing numerical is being asserted there at all.**

### Can the generative work be driven without `chlu.cli`? **Yes.**
The `chlu.experiments.exp_gen_*` modules do **not** import `chlu.cli`; they import
`chlu.config` / `chlu.core.gen_*` / `chlu.data.*` directly, all of which are clean. Their
`__main__` paths are self-sufficient. `python -m chlu.experiments.exp_gen_cell --help` prints the
full surface (`--dataset --pretext --stage {cell,aggregate,all,encoder,decoder,store,dream,
certificate,baselines,compose} --seed --arm --out --quick --config`).
⭐ This report's §1 is itself a demonstration: every frozen artefact was loaded and a decoder
forward pass run **without `chlu.cli` ever being imported.**

### ⚠ The dataset-download route IS dead, and I found no in-tree alternative
`chlu data shapes3d --download` is a `chlu.cli` command, so it cannot run here. **However, this is
not blocking for the generative line as bundled:** the `.claude/` tree already ships the
**precomputed `features.npy` cache** (480,000 rows, sha256-verified in §1) and
`factor_index.npz`, which is what `hub/composition` consumes. The 268 MB raw Shapes3D file is
needed only to *rebuild* the cache. Options, none of which I took: (a) apply the declared
lazy-import overlay from supplement 2 and use the CLI, (b) call the download helper inside
`chlu.data.shapes3d` directly via `python -c`, bypassing `chlu.cli`, (c) fetch the file manually.
See §11-Q4.

---

# 10. ⭐ WHAT I COULD NOT RESOLVE HONESTLY

1. **I never ran anything on the Mac, and I cannot.** Everything above about the Mac is inferred
   from artefacts and hashes it left in the bundle. The §3 and §5 conclusions are *proven here*
   (a config flag and 11 named bytes reproduce the exact failing values). The §4 conclusion —
   that `Z` differs at the last bit between ISAs — is **strongly indicated but not directly
   verified**, because no Mac-side record of `Z` exists. `repro.py` closes it in one run.
2. **`features.npy` for seeds 1 and 2 was not sha-verified.** Only seed 0's 1.9 GB cache was
   hashed (§1-E), to stay inside a sane I/O budget with other spokes on the same disk. Seed 0
   matched exactly; seeds 1 and 2 are pinned in `BUNDLE_MANIFEST.json` and were reported green by
   the workspace verifier.
3. **I did not re-run the full suite**, per the brief. My triage is confined to the 8 failures the
   brief named. The other ~109 are untouched and may contain further real issues.
4. **I did not verify that the external classifiers' *weights* match the Mac's** — their manifests
   carry no hash keys (§1-C). Only that they load.
5. **§6a: I did not identify the specific suite interaction** that flipped the MI sign in
   `_F-suite.log`. I showed only that the quantity is float32 dust and passes in isolation here.

---

# 11. ⭐ NUMBERED OPEN QUESTIONS

1. **Can the Mac run `_hub_scratch/I-probe/repro.py` and report `L3b Z sha` and `L3d store_hash`?**
   This is the single outstanding measurement. If
   `L3b == 1df1e9677c07bcba44ef2a2ac3279419017c15197f42468f1ca48e81c4d25e87`, even the float32 conv
   path is bit-portable and parity is total. It takes ~20 s and no GPU.
2. **Ruling wanted: is `store_hash` an over-specified fingerprint?** §4 shows it moves under a pure
   re-batching **on one machine**. If the humans agree, the fix is upstream (assert `well_sizes` /
   `codes_sum` / `store_bytes` — all of which pass here — rather than a float32 byte hash). ⛔ I
   have not touched it.
3. **Why 30 tests on the Mac vs 47 here for the same two files?** Mac reported *"29 passed, 1
   skipped"*; this box reports *"2 failed, 43 passed, 2 skipped"*. Collection differs by 17 tests.
   Was the Mac run a narrower selection (`-k`), or is collection differing here? **This could hide
   failures and should be checked before any parity claim is finalised.**
4. **How should Shapes3D be downloaded on this box?** (§9). Recommend the supplement-2 lazy-import
   overlay so `chlu data shapes3d --download` works, rather than a bespoke bypass. **Needs a
   ruling — I did not act.**
5. **Should `tests/test_kt.py`'s `abs=1e-15` round-trip assertions be re-scoped?** §6b shows they
   pin a Monte-Carlo estimate to a tolerance far tighter than its own quoted standard error. They
   can only ever pass on the machine that generated the reference. ⛔ Not patched.
6. **The `jax_enable_x64` process-global leak (§3) is latent on the Mac too.** It is an ordering
   accident the Mac has not yet hit. Worth fixing upstream (a session-scoped autouse fixture)
   before it silently corrupts a Mac-side fingerprint.
7. **Does the `CONTEXT.md` warning get lifted?** On this evidence I recommend narrowing it to:
   *"numbers LOADED and RENDERED here are comparable; numbers from a REBUILT k-means++ store are
   not bit-comparable with the corpus — and were never bit-comparable machine-to-machine."*
   That is a hub-scoped decision, not mine.

---

# 12. SUB-AGENT CONFIRMATION

⭐ **I spawned no sub-agents. None is running.** All work was done in-process via foreground and
background shell commands, and **every background command has completed** (`uv` build exit 0,
py311 probe exit 0, determinism sweep exit 0). `ListAgents` shows only unrelated peer sessions
(other spokes), none of them mine.

**Bundle integrity: I modified nothing.** All probe scripts and logs live outside the bundle at
`_hub_scratch/I-probe/`. The py311 venv is at `/c/clu-venv/py311-probe`, outside the bundle. The
working `.venv/` was not touched. No test, fingerprint, manifest or source file was edited.

### Artefacts produced (all outside the bundle)
```
_hub_scratch/I-probe/repro.py           ⭐ the reproducer for the Mac (self-contained)
_hub_scratch/I-probe/repro_py312.log       py312 output
_hub_scratch/I-probe/py311.log             py311 output + isolated pytest
_hub_scratch/I-probe/determinism.log       3 runs x OMP 4/1/4
_hub_scratch/I-probe/load_check.py/.log ⭐ 37 MATCH, 0 DIFFER
_hub_scratch/I-probe/z_variants.py/.log ⭐ store_hash moves under re-batching
_hub_scratch/I-probe/npz_diff.py/.log      11 members all identical
_hub_scratch/I-probe/zip_hdr.py/.log    ⭐ the 11 create_system bytes
_hub_scratch/I-probe/x64_leak.py/.log   ⭐ suite-failure hashes reproduced exactly
_hub_scratch/I-probe/npz_probe.py          np.savez byte-determinism
_hub_scratch/reports/_I-py311-build.log    uv venv + uv sync
```
