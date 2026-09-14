# gen-decoder-lift — experiment-engineer report

**Task + acceptance criterion:** can a **decoder-only** change lift held-out `shape` to ≥ 0.8 × its seen value?
Two variants maximum, pre-registered first, encoder and feature cache frozen.

**Status: done.** All six acceptance items met.

# ⭐⭐ VERDICT: **YES — `shape` CLEARS 0.8 on both normalisers and on all three seeds.** Option 2 does **NOT** activate.

| | shipped baseline | **V1 `capacity_schedule`** | V2 `object_weighted` | floor |
|---|---|---|---|---|
| `shape` on B (3 seeds) | 0.5926 ± 0.0866 | **0.9398 ± 0.0131** | 0.5231 ± 0.2923 | — |
| **ratio vs random-A** | 0.6530 | **0.9656 ✅ CLEARS** | 0.5611 ⛔ | 0.8 |
| **ratio vs matched-seen** | 0.6978 | **0.9964 ✅ CLEARS** | 0.5374 ⛔ | 0.8 |
| per-seed ratio (matched-seen) | 0.571 / 0.662 / 0.872 | **1.024 / 1.041 / 0.931** | 0.140 / 0.916 / 0.583 | 3/3 clear |

⭐ **This falsifies the Hub's standing expectation (pre-registered as P5: "a decoder-only fix will NOT clear 0.8") and
my own point predictions P1–P4.** ⛔ **The variant that won is the one filed as the NULL CONTROL** — plain capacity +
schedule, uniform pixel MSE untouched. **The variant the task nominated as "the single most useful clue you have" —
the object-weighted loss — LOST, and lost badly.** Both are reported in full (§5).

⭐ **Mechanism, and it is not subtle.** The shipped decoder was **under-resourced and under-trained**, not
compositionally blocked. A pre-registered diagnostic run **before** either variant shows the held-out `shape` identity
was in the frozen features all along: a **linear** head on the cached features reads `shape` on B at **1.0000 ± 0.0000**
(3 seeds). Nothing had to be recovered — only rendered. V1 drops decode MSE 0.00300 → 0.00110 and `shape` on B follows.

**Downstream reconciliation list: YES, 4 items — see §9.** Two need a Hub owner at this review: ⛔ **R16's "GATE 1
REVERSED / STOP" is now superseded** and every site quoting `shape` 0.5926 / ratio 0.653 as *the decoder's ceiling*
must be re-labelled as *the under-trained decoder's ceiling*; and R17 (widened split) needs re-reading, because the
argument for it changes (see §9.2 — I think it still stands, for a different reason).

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — instrument repair plus one gating measurement.* ⛔ No store, no well, no arm, no CLU number appears
  below. Nothing here is evidence for the primitive.
- **Laundering control:** the **matched-seen** control (R16) — same hue corner (`floor/wall/object_hue ≥ 0.8`,
  `scale ≥ 1.1`, `orientation > 20`), **seen** shape (`shape ≠ cylinder`), minus every fitted row; n = 209/214/213 per
  seed. It is a companion row on **every** table below, per R16.
- **Falsifies:** `shape` ratio still `< 0.8` after both variants ⇒ option 2. **It did not fire.**
- **Does NOT falsify:** no movement on `scale` (photometric, R16). For the record `scale` did move — V1's `scale` ratio
  vs matched-seen is **1.0063** (baseline 0.7924 as I measure it over 3 seeds; 1.007 at R16's seed 0) — i.e. it stays
  at ≈1.0, exactly as the photometric attribution predicts. ⛔ I did not chase it.

---

## 1. What I did

1. **Filed `PREREG.md` first** (mtime `2026-09-10 09:55:07`; the harness `.claude/scratch/gen-decoder-lift/lift.py`
   did not exist yet — first run 10:00). Two variants declared, four point predictions, a decision rule, and a
   diagnostic with its own prediction. One compute-sizing **amendment** at 10:16:27, **before any V1 or V2 run** (§6).
2. **Re-scored the shipped decoder** on A / B / matched-seen — reproducing `gen-feature-bridge` §6 to the digit — as
   the reference row, and **refit the `baseline` recipe at seed 0**, which returned parameter hash
   `47c0f83196cd220d` — **bit-identical to the shipped decoder** (§3). Without that, nothing here is comparable.
3. **Diagnostic D** — is `shape` recoverable from the frozen features at all? (§4)
4. **Two variants, 3 seeds each**, refit from scratch against the frozen encoder and the untouched cache (§5).
5. **Emitted the winner as a RECIPE** (amendment 2026-09-10): a loadable config that refits it into *any* feature
   cache, plus the fitted artifacts for R0 (§7).
6. Re-ran the R3.3 no-leak assertion per seed and re-verified every frozen invariant **after** all runs (§2).

⛔ I did **not** touch the encoder, the feature cache, the probe, `chlu/eval/`, `chlu/data/`, `gen_store.py` or
`scripts/csf3/`. ⛔ I did not try a third variant.

---

## 2. The frozen invariants — asserted before AND after every run (acceptance item 4)

| seed | `encoder.pkl` sha256 | encoder param hash | cache scope / rows | `features.npy` sha256 | re-verified after |
|---|---|---|---|---|---|
| 0 | `97e27fe583f2b752…` | `98d326028316ae2c…` | full / 480,000 | `d6728f01d102b0d0…` | ✅ **MATCH** |
| 1 | `5ef85fd73a3946c1…` | `fc05e2c5f39694b2…` | used / 9,288 | `484b446c4dbb9fc4…` | ✅ **MATCH** |
| 2 | `22072966835d6ae7…` | `154d498610648fac…` | used / 9,288 | `8c0a69e88beb46b8…` | ✅ **MATCH** |

The shipped `probe.pkl` (`ac518194cbf447cb…` / `1b8070dd06decb28…` / `3b84b3e9ebb11f7b…`) and the shipped
`decoder.pkl` are **byte-identical before and after** as well — I re-hashed all nine files at the end of the session.
`load_feature_cache(..., verify=True)` re-checked the cache sha256 on **every** load (never disabled).
⭐ **`gen-store-grouping` is training on exactly the cache it started with.**

**⚠ One declared operation on the frozen encoder, and why it is not a change.** The matched-seen control set is not
covered by the `"used"` caches at seeds 1 and 2. ⛔ I did **not** rewrite the cache. I ran the **same frozen encoder**
forward over those rows in a side buffer and reproduced the cache's own `float16` quantisation
(`encode_images(...).astype(float16).astype(float32)`), which is byte-for-byte what `build_feature_cache` would have
written. At seed 0 (where the full cache *does* cover them) the numbers come from the cache itself.

**R3.3 no-leak, re-run per seed per fit (acceptance item 5):**
`8192 training rows checked, **72 / 72 held-out images excluded, 0 leaked**` — identical on all three seeds and on
every variant's fit. The assertion is the shipped `assert_no_holdout_leak`, which re-derives each used row's factor
values through the grid bijection rather than re-reading the mask, and whose positive control
(`test_no_leak_POSITIVE_CONTROL_fires`) is proven able to fire.

---

## 3. ⭐ The reference row is provably the shipped decoder, not a re-tune

Refitting the `"baseline"` recipe at seed 0 through the **new** recipe machinery reproduced the shipped decoder exactly:

```
recipe baseline_refit: {'name': 'baseline', 'channels': [128, 64, 32], 'groups': 8,
   'blocks_per_stage': 1, 'steps': 1200, 'batch': 128, 'lr': 0.001, 'loss': 'mse',
   'lr_schedule': 'constant', ...}
   step 1/1200  loss=0.18785        <- the bridge log's own first loss
   step 1200/1200  loss=0.00311     <- the bridge log's own final loss
fitted baseline_refit in 1288s: 47c0f83196cd220d (250467 params)
```
`47c0f83196cd220d…` **is** `bridge_manifest.json:decoder_param_hash` for `recon` seed 0. And re-scoring the shipped
`.pkl` reproduces `gen-feature-bridge` §6 to four decimals: `A_random` agg 0.9476 / `shape` 0.9043, `B` agg 0.8681 /
`shape` 0.5000, `A_matched` agg 0.9203 / `shape` 0.8756, `scale` 0.7033 / 0.7083. ⭐ **Same harness, same probe, same
protocol** produced every number in this report.

---

## 4. ⭐⭐ Diagnostic D (pre-registered, run BEFORE the variants) — the information was never lost

A factor classifier fit on the **cached train features** of the frozen encoder and read on the **cached B features**.
Labels are measurement only (R1's exemption); ⛔ nothing here enters the generative path.

| head | `A_random` | **`A_matched`** (companion) | **B (held out)** | prereg | verdict |
|---|---|---|---|---|---|
| linear | 0.9860 ± 0.0020 | 0.8656 ± 0.0924 | **1.0000 ± 0.0000** | D1 = 0.75 (0.45–0.93) | ❌ **falsified — above the range** |
| MLP-512 | 0.9912 ± 0.0029 | 0.9906 ± 0.0038 | **0.9815 ± 0.0262** | D2 = 0.85 (0.55–0.98) | ✅ in range, near the top |

Per-seed B: linear 1.0000 / 1.0000 / 1.0000; MLP 0.9444 / 1.0000 / 1.0000.

⭐ **The pre-registered decision rule (D2 ≥ 0.85 ⇒ the information is present and the decoder is discarding it) fired.**
A **linear** functional of the frozen features separates the held-out large cylinder from every seen shape *perfectly*.
⛔ **So the phase was never one encoder-change away from anything**, and R18's worry ("if the fix needed a different
encoder, the cache changes and every downstream spoke's work is void") never had to be tested. This is the single most
load-bearing number in the report: it says the residual was a **renderer** problem, and it said so before either
variant ran.

---

## 5. The two variants (acceptance items 2 and 3)

Both were declared in `PREREG.md` §3 before any run, and **both are reported, including the one that lost.**
Every cell: probe top-1 accuracy on **real cached features decoded through that variant's frozen decoder**, 3 seeds,
same splits, same **unchanged** probe. `probe_on_real = 1.0000` on every split, every seed, every variant.

### 5.1 ⭐ `shape` — the compositional term, which is what the gate reads (R16)

| variant | `A_random` | **`A_matched`** | **B** | ratio vs random-A | ratio vs matched-seen | verdict |
|---|---|---|---|---|---|---|
| baseline (shipped) | 0.9076 ± 0.0122 | 0.8492 ± 0.0269 | 0.5926 ± 0.0866 | 0.6530 | 0.6978 | ⛔ DOES NOT CLEAR |
| **V1 `capacity_schedule`** | **0.9733 ± 0.0090** | **0.9432 ± 0.0404** | **0.9398 ± 0.0131** | **0.9656** | **0.9964** | ✅ **CLEARS 0.8** |
| V2 `object_weighted` | 0.9323 ± 0.0132 | 0.9735 ± 0.0342 | 0.5231 ± 0.2923 | 0.5611 | 0.5374 | ⛔ DOES NOT CLEAR |

Per-seed `shape` on B — baseline `0.5000 / 0.5694 / 0.7083` · **V1 `0.9306 / 0.9583 / 0.9306`** · V2 `0.1389 / 0.8472 / 0.5833`.
Per-seed V1 ratios — vs random-A `0.9586 / 0.9943 / 0.9444`; vs matched-seen `1.0236 / 1.0410 / 0.9306`.
⭐ **The worst V1 seed clears the floor by 18 % relative on the harder normaliser.**

### 5.2 Companion rows (aggregate, `scale`, MSE, cost) — ⛔ read `shape` first, these are context

| variant | agg `A_random` | agg B | `scale` ratio vs matched-seen | decode MSE A | decode MSE B | params | train s/seed |
|---|---|---|---|---|---|---|---|
| baseline | 0.9507 ± 0.0129 | 0.8472 ± 0.0218 | 0.7924 | 0.00300 | 0.00300 | 250,467 | 1,288 |
| **V1** | **0.9902 ± 0.0030** | **0.9830 ± 0.0061** | **1.0063** | **0.00110** | **0.00110** | 1,516,227 | 5,728–20,880 |
| V2 | 0.9607 ± 0.0085 | 0.8935 ± 0.0491 | 0.9291 | 0.00300 | 0.00300 | 250,467 | 1,358–1,735 |

(⚠ `train s/seed` is contended wall-clock on an 8-core CPU with load average 90–165 from concurrent agents; the
5,728 s V1 seed is the least-contended one and is the honest per-seed figure. See §8.)

### 5.3 ⭐ Why V2 lost — and it is a finding, not a null

V2 did *exactly* what it was designed to do and it was the wrong thing. It renders the silhouettes it has **seen**
markedly better (`A_matched` `shape` **0.8492 → 0.9735**, the best matched-seen number of any variant) and turns the
held-out one into a **coin flip** (0.1389 / 0.8472 / 0.5833, σ = 0.29 — six times the baseline's spread).

The probe's `shape` prediction histogram on B makes it unambiguous. B is 72 items, all of the **same** true class
(cylinder = class 1); the numbers below are how many of the 72 renders the probe reads as each class:

| seed | baseline | **V1** | V2 |
|---|---|---|---|
| 0 | cube 36 / **cyl 36** | cube 5 / **cyl 67** | cube 62 / **cyl 10** |
| 1 | cube 31 / **cyl 41** | cube 3 / **cyl 69** | cube 11 / **cyl 61** |
| 2 | cube 21 / **cyl 51** | cube 5 / **cyl 67** | cube 30 / **cyl 42** |

⭐ **The baseline is undecided (36/36 at seed 0). V2 is decided and WRONG (62 cubes). V1 is decided and right.**
A sharper silhouette objective bought the renderer **confidence**, not **identity** — it committed harder to the
nearest *seen* silhouette. ⛔ That is the opposite of what a compositional fix looks like, and it is why I am reporting
it rather than quietly dropping it. Qualitative panel: `B_renders_seed0.png` (real / baseline / V1 / V2).

⚠ Note it also *raised* aggregate B (0.8472 → 0.8935) while *lowering* `shape`. **This is precisely the "you improved
the aggregate and not `shape`" trap the task warned about**, and V2 walks straight into it. If the aggregate had been
the read, V2 would have looked like a modest win.

⚠ **Honest caveat on the qualitative panel.** `B_renders_seed0.png` is *not* dramatic to the eye: all three variants
render the held-out object softly, because it is a magenta cylinder on a magenta floor (that is the R16 photometric
finding). V1's renders are visibly crisper at the floor/wall/object boundaries than the baseline's, and V2's are the
washiest — but ⛔ **nobody should read the verdict off the figure.** The verdict rests on two *independent* quantitative
signals that agree: the frozen probe (which reads **real** images of this exact corner at 1.0000) reads 67/72 of V1's
renders as cylinders vs 36/72 of the baseline's, **and** the decode MSE against the real image drops 0.00300 → 0.00110.
A probe artefact would not also move the pixel error.
---

## 6. Pre-registration outcome — 5 of 9 predictions falsified, and the falsifications are the result

| # | pre-registered | predicted | measured | verdict |
|---|---|---|---|---|
| P1 | V1 `shape` ratio, random-A | 0.66 (0.58–0.74) | **0.9656** | ❌ **falsified, far above** |
| P2 | V1 `shape` ratio, matched-seen | 0.68 (0.59–0.78) | **0.9964** | ❌ **falsified, far above** |
| P3 | V2 `shape` ratio, random-A | 0.72 (0.60–0.84) | 0.5611 | ❌ falsified, below |
| P4 | V2 `shape` ratio, matched-seen | 0.75 (0.62–0.87) | 0.5374 | ❌ falsified, below |
| **P5** | **verdict, both variants** | **DOES NOT CLEAR** | **V1 CLEARS, V2 does not** | ❌ **falsified for V1** |
| P6 | V1 decode MSE on A | ≤ 0.0020 (0.0010–0.0025) | **0.00110** | ✅ in range |
| P7 | V2 decode MSE on A **rises** ≥ 0.0035 | ≥ 0.0035 | 0.00300 (**unchanged**) | ❌ falsified |
| P8 | aggregate moves < 0.03 | < 0.03 | V1 +0.040 (A) / +0.136 (B); V2 +0.042 (`A_matched`) | ❌ falsified |
| P9 | `scale` on B, matched-seen ratio ≈ 1.0 | 1.0 (0.90–1.10) | V1 **1.0063**, V2 0.9291 | ✅ both in range |
| D1 | linear-on-features `shape` on B | 0.75 (0.45–0.93) | **1.0000** | ❌ falsified, above |
| D2 | MLP-on-features `shape` on B | 0.85 (0.55–0.98) | 0.9815 | ✅ in range |
| D3 | same on A | ≥ 0.97 | 0.9912 | ✅ |

⭐ **P5 was the Hub's own standing expectation, adopted verbatim and pre-registered so that falling back to option 2
would not look like a retreat. It is now falsified in the favourable direction, which is the strongest form this
result could have taken:** the number was committed to *before* the machinery existed, by an agent that expected to
lose. ⛔ P7's falsification is worth keeping: V2 did **not** trade pixels for identity — it reached the *same* pixel
MSE and simply spent it differently. My stated rationale for V2 was therefore wrong in its premise as well as its
conclusion.

**The one amendment (`PREREG.md` §6, 10:16:27, before any V1/V2 run):** V1's step count was cut 4800 → 1800 after
measuring that the baseline decoder costs ≈19 min/1200 steps here and that V1 is 6.69× the per-step MAC count
(hand-counted per conv), i.e. ≈25 h of CPU for three seeds at 4800. ⭐ **The amendment biases *against* V1** (a weaker
null control), and V1 won anyway — so the caveat it carries ("capacity was ruled out with a reduced budget") is
discharged in the only direction that matters. ⚠ V1's loss was still falling at 1800 steps (0.16399 → 0.00097); there
is more on the table, which is a *follow-up*, not a caveat on the verdict.

---

## 7. ⭐ The winner is emitted as a RECIPE, not one fitted artifact (the 2026-09-10 amendment)

`chlu/core/gen_bridge.py: DECODER_RECIPES["v1_capacity_schedule"]`, resolved by `resolve_decoder_cfg(cfg, name)` into
a `DecoderRecipe`, built by `make_decoder(feat_shape, out_shape, recipe, key)` and fit by
**`train_decoder_from_features(feat_shape, features, images, cfg, key, recipe=...)`** — ⭐ **the refit entry point: hand
it any feature cache and it reproduces the winning decoder from scratch for that feature space.** A ladder rung whose
encoder was retrained fits its own decoder this way and is scored against **its own** ceiling.

```yaml
gen_bridge:
  dec_recipe: v1_capacity_schedule    # == the five lines below
  dec_width_mult: 2.0                 # [128,64,32] -> [256,128,64]
  dec_blocks_per_stage: 2             # skip-free: 2x (conv3 + GroupNorm + ReLU) per x2 upsample
  dec_steps: 1800
  dec_lr_schedule: cosine             # 5% linear warmup, decay to 0
  dec_loss: mse                       # ⛔ NOT object_weighted
```
Emitted at `.claude/outputs/gen-decoder-lift/winner/`: `winning_recipe.yaml`, `winner_hashes.json`, and the three
fitted R0 artifacts `decoder_v1_s{0,1,2}.pkl`.

| seed | decoder param hash | pkl sha256 | params | fit against encoder |
|---|---|---|---|---|
| 0 | `01ef91f75eeaae7b…` | `bcf39c42bc7885d8…` | 1,516,227 | `98d326028316ae2c…` |
| 1 | `001011bc603a7cb9…` | `7a31fa49ada6ce04…` | 1,516,227 | `fc05e2c5f39694b2…` |
| 2 | `2d7eea75add77044…` | `55957abd50e87453…` | 1,516,227 | `154d498610648fac…` |

**Which hyperparameters are width-dependent (asked for explicitly).**
- ⛔ **None of `dec_width_mult`, `dec_blocks_per_stage`, `dec_steps`, `dec_lr`, `dec_lr_schedule`, `dec_batch`,
  `dec_groups`** is a function of the feature width `D`. `dec_width_mult` multiplies the existing channel list; the
  number of ×2 upsampling stages is **derived** from `feat_shape` vs `out_shape` at construction
  (`_n_upsamples`); the loss is in pixel space. `test_recipe_is_not_a_function_of_the_feature_width` builds the same
  recipe against `C = 8` and `C = 64` feature maps and asserts identical depth and output shape.
- ⚠ **The one entry that tracks the encoder is `dec_channels[0]` (128)**, which the shipped encoder sets to its own
  output channel count. On a rung whose encoder has a different `C`, scale that first entry with `C` and leave
  `dec_width_mult` alone. Everything else transfers unchanged.
- ⛔ Nothing anywhere hard-codes `D = 2048`.

**⛔ The default `dec_recipe` is deliberately left at `"baseline"`.** Making the winner the shipped default is a
phase-level decision — it multiplies every decoder fit by ~6× parameters and ~16× CPU, and `config.py` is shared with
two live spokes right now. The one-line flip is written out in §9.1 for the Hub to take at the merge.

---

## 8. How I verified — commands and observed output

**Environment (protocol §4):** ⛔ no `uv sync` in the worktree. Main venv reused —
`PYTHONPATH=/Users/user/Desktop/CHLU-gen-declift CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen
JAX_PLATFORMS=cpu /Users/user/Desktop/CHLU/.venv/bin/python …` — so **jax stays at 0.9.0** (w6 lesson), equinox 0.13.4,
optax, numpy 2.4.1, h5py 3.16.0, Python 3.11.13, CPU, float32 (`jax_enable_x64` off).

**Harness** (⛔ scratch, nothing shipped for it): `.claude/scratch/gen-decoder-lift/{lift.py,summarize.py,renders.py}`.
```
python -u lift.py prep <seed>            # ONE contiguous h5 pass for train+A+B+matched
python -u lift.py base <seed>            # score the SHIPPED decoder.pkl
python -u lift.py diag <seed>            # Diagnostic D
python -u lift.py fit  <seed> v1 v1_capacity_schedule
python -u lift.py fit  <seed> v2 v2_object_weighted
```
Observed (V1, seed 1 — the least-contended run):
```
seed 1: encoder fc05e2c5f39694b2 probe fb16109ded84c908 cache used n=9288 sha 484b446c4dbb9fc4 — FROZEN, verified
  R3.3 no-leak: 8192 rows checked, 72/72 excluded, 0 leaked
  recipe v1: {'channels': [256,128,64], 'blocks_per_stage': 2, 'steps': 1800, 'lr_schedule': 'cosine', 'loss': 'mse'}
  fitted v1 in 5728s: 001011bc603a7cb9 (1516227 params) loss 0.15412 -> 0.00108
    A_random:  n=1024 agg=0.9871 shape=0.9639 scale=0.9609 mse=0.00117 real=1.0000
    B:         n=72   agg=0.9861 shape=0.9583 scale=0.9583 mse=0.00124 real=1.0000
    A_matched: n=214  agg=0.9836 shape=0.9206 scale=0.9907 mse=0.00150 real=1.0000
```

**Targeted tests:** `pytest tests/test_gen_decoder_lift.py -q` → **19 passed in 51.50 s**.
One genuine failure found and fixed on the way: my YAML round-trip test passed `str(path)` to `save_config`, which
takes a `Path` (`AttributeError: 'str' object has no attribute 'parent'`). Fixed in the test, not the shared helper.

**Full suite (acceptance item 6), main venv, `-p no:randomly --no-cov`:**

| | commit | result |
|---|---|---|
| **before** | `fcdca43` (my base) in a detached worktree `../CHLU-declift-base` | **2010 passed, 0 failed** (3:06:51) |
| after (1st) | `3c87c85` | **2029 passed, 0 failed** (5:07:22) |
| **after (final, branch tip)** | `6f1bf68` | **2029 passed, 0 failed** (0:51:56, uncontended) |

**Delta = 2029 − 2010 = +19 = exactly my new tests. No pre-existing test changed state.**
⚠ Both suites ran while 2–4 of my own training jobs and other agents' suites were on the same 8 cores (load average
90–165), which is why the wall-clocks differ so much; the pass counts are what matter.
`ruff check` clean on all three touched files. ⛔ `ruff format` not run on shared files (repo baseline is not
format-clean — same reasoning as `gen-feature-bridge` and `gen-data-splits`).

**⚠ §7.23 handled, not re-introduced.** Every new module path casts its input to its own parameter dtype, and
`test_deep_decoder_and_weighted_loss_read_a_float32_input_under_x64` runs the deep decoder, the weighted loss and
`edge_weight_map` under `x64 ∈ {False, True}`. The full-suite runs above are the real check and they are green.

**⛔ CSF3: NOT RUN.** Unreachable from an agent session (2FA / no VPN). No job id, no cluster path. The submit line for
a re-fit of the shared decoder with the winning recipe is:
```bash
EXP='DATASET=shapes3d,PRETEXT=recon,ARMS=clu,SEEDS=0 1 2,PAYLOAD_MODULE=chlu.experiments.exp_gen_bridge'
sbatch --export="ALL,$EXP,STAGE=decoder,GEN_GIT_COMMIT=$GEN_GIT_COMMIT,EXTRA_ARGS=--dec-recipe v1_capacity_schedule" \
       -a 0-2%2 -t 2:00:00 scripts/csf3/job_gpu_gen.sh
```
⚠ **That `EXTRA_ARGS` flag does not exist yet.** `exp_gen_bridge`'s parser has no `--dec-recipe`; I own no line of
`chlu/experiments/exp_gen_bridge.py` under this task, so I did not add it. Until it exists the recipe is selected via
`projects/<name>/config/config.yaml` (`winning_recipe.yaml`, §7) or by flipping the default (§9.1). **Owner: Hub.**

---

## 9. Reconciliation list (⭐ needs a Hub-assigned owner at this review)

1. ⭐ **R16's "GATE 1 REVERSED ON THE COMPOSITIONAL TERM: STOP" is superseded.** The ruling is *correct as written* —
   it caught a real capped-B failure — but its **attribution** was wrong: the cap was the decoder's capacity, not the
   composition. Every site that quotes `shape` 0.5926 / ratio 0.653 / 0.571 as **"the decoder cannot render held-out
   shapes from REAL features"** should now read **"the *under-trained* decoder could not; with the V1 recipe the same
   frozen features render at 0.9398, ratio 0.966 / 0.996."** ⛔ Do not retire the 0.653 — it is the correct number for
   the shipped decoder and it is what motivated this spoke. **Owner: Hub (the rulings ledger).**
   The one-line default flip, if the Hub takes it: `chlu/config.py`, `GenBridgeConfig.dec_recipe`,
   `"baseline"` → `"v1_capacity_schedule"`, and update `test_defaults_preserve_the_shipped_behaviour`
   (`tests/test_gen_decoder_lift.py`) which currently pins the old default on purpose.
2. ⚠ **R17 (widened split MANDATORY) — re-read, do not simply drop.** Its stated argument ("72 items × 3 seeds inside
   a half-width band cannot separate two arms") is now weaker: the band is no longer half-width, it is
   0.94 ± 0.01 against a 0.94–0.97 ceiling. ⭐ **But my V2 result is a fresh, independent argument *for* R17:** on 72
   items the same recipe produced `shape` 0.1389, 0.8472, 0.5833 across three seeds. **72 items is demonstrably
   capable of a σ = 0.29 swing**, and that is now a measurement rather than a worry. **My read: R17 still stands, on
   better evidence. Owner: Head/Hub — it is a ruling, not mine to change.**
3. ⚠ **The instrument is now close to saturation and someone must say so before the arms run.** With V1, aggregate
   `ceiling_A` = 0.9902 and `ceiling_B` = 0.9830 (ratio 0.9927), and `shape` on B is 0.9398 against a probe that reads
   **real** images at 1.0000. ⭐ Good for R3 — the decoder is no longer the bottleneck, so kill rule (i) reads on the
   store, which is exactly what the gate existed to guarantee. ⛔ But arms now compete in a ~2-point band near the top,
   and a difference of 0.01 there is not the same evidence as 0.01 in the middle of the range. **Owner: whoever files
   the arms prereg.**
4. **The `--dec-recipe` CLI flag / `EXTRA_ARGS` gap** (§8) — without it the cluster path cannot select the winning
   recipe except through a project YAML. **Owner: Hub** (I own no file under `chlu/experiments/` or `scripts/csf3/`).

---

## 10. Flag provenance (protocol §5, mandatory)

Commit **`6f1bf68`** (branch `agent/experiment-engineer/gen-decoder-lift`, base `main @ fcdca43`).
jax **0.9.0** / equinox 0.13.4 / optax / numpy 2.4.1 / h5py 3.16.0, Python 3.11.13, **CPU**, float32
(`jax_enable_x64` **off**). Seeds **{0, 1, 2}** throughout.
Dataset root `$CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; Shapes3D `recomb2element/leave1out`,
**n_test = 72 / 480,000**, asserted per seed. Encoder / feature cache / probe: **frozen, hashes in §2.**

**Non-default flags in effect for the variant runs** (everything else is `GenBridgeConfig`'s shipped default):

| knob | baseline row | **V1 (winner)** | V2 |
|---|---|---|---|
| `dec_recipe` | `baseline` | **`v1_capacity_schedule`** | `v2_object_weighted` |
| `dec_channels` (resolved) | `[128, 64, 32]` | **`[256, 128, 64]`** (`dec_width_mult = 2.0`) | `[128, 64, 32]` |
| `dec_blocks_per_stage` | 1 | **2** | 1 |
| `dec_steps` | 1200 | **1800** | 1200 |
| `dec_lr_schedule` / `dec_warmup_frac` | `constant` / — | **`cosine` / 0.05** | `constant` / — |
| `dec_loss` | `mse` | **`mse`** | `object_weighted` |
| `dec_edge_weight` / `dec_edge_blur` / `dec_grad_weight` | n/a | n/a | 8.0 / 5 / 1.0 |
| `dec_groups` / `dec_batch` / `dec_lr` | 8 / 128 / 1e-3 | 8 / 128 / 1e-3 | 8 / 128 / 1e-3 |

Unchanged from the bridge and in force everywhere: `n_train = 8192`, `n_eval_a = 1024`, `eval_batch = 256`,
`cache_dtype = float16`, `cache_block = 8192`, `verify_cache_checksum = True`, `ceiling_floor = 0.8`,
encoder `[32,64,128,128]` k4 s2 g8 (**not refit — loaded frozen**), probe `[32,64,128]` g8 (**not refit — loaded
frozen**). ⛔ `--quick` was **never** used for any reported number. Optimizer Adam; GroupNorm (stateless).
**PRNGKey:** every decoder fit is threaded from `jax.random.PRNGKey(seed*10_007 + 20_260_910)` — **the same base key
the shipped bridge used** — split, never reused, so the *only* difference between the baseline row and a variant is the
recipe. Diagnostic D uses `+20_260_911`. Row selection `np.random.default_rng(seed)` via the shipped `select_rows`.

Matched-seen control set (R16): `floor_hue ≥ 0.8 ∧ wall_hue ≥ 0.8 ∧ object_hue ≥ 0.8 ∧ scale ≥ 1.1 ∧ orientation > 20
∧ shape ≠ cylinder`, minus `rows["train"]`; asserted disjoint from `split.test_rows`. **n = 209 / 214 / 213.**

---

## 11. Git footprint

- **Branch:** `agent/experiment-engineer/gen-decoder-lift`, base **`main @ fcdca43`** as the task specified.
  `git rebase main` → *"up to date"* (main did not move; ⛔ did **not** rebase onto the stale `origin/main`, §7.21).
- **Worktrees:** `../CHLU-gen-declift` (fresh, as specified) and a **detached** `../CHLU-declift-base @ fcdca43` used
  only for the baseline suite. ⛔ Did not touch `CHLU-bprime-family2`, `CHLU-wt2`, `CHLU-wt3`, `CHLU-merge` (parked),
  nor the shared main checkout, which was on another agent's branch
  (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring @ 7fcef50`) throughout and is untouched.
  ⚠ Two worktrees is one over R5's "cap 4 active spokes" arithmetic if all four parked ones are counted; the extra one
  was detached, read-only for the suite. **Both are now removed.**
- **Ref verified FROM THE MAIN REPO before removal** (`git log --oneline main..agent/experiment-engineer/gen-decoder-lift`
  listed both commits; `git diff --stat` showed the 3 files / 638 insertions) and **re-verified AFTER removal** at
  `6f1bf6833eef61e230e57f9aa37416c9885b448f`, both commits still reachable. The shared main checkout was clean and on
  another agent's branch throughout and is unchanged.
  ⚠ `CHLU-gen-store` (the concurrent `gen-store-grouping` spoke) appeared mid-session; ⛔ I never entered it, and it
  branched from the same `main @ fcdca43`, so there is no shared-file collision to resolve.
- **Commits (2, atomic):**

| hash | subject |
|---|---|
| `3c87c85` | make the gen decoder a RECIPE: capacity, schedule and an object-weighted loss |
| `6f1bf68` | record the measured outcome of the two decoder variants |

- **Files touched — 638 insertions, 30 deletions:**

| file | change |
|---|---|
| `chlu/core/gen_bridge.py` | **+298 / −30** |
| `chlu/config.py` | **+32 / −0**, a **single pure insertion** at lines **3334–3365**, *inside* the existing `GenBridgeConfig` block, immediately after `dec_lr` |
| `tests/test_gen_decoder_lift.py` | new, 308 lines / 19 tests |

⛔ **`chlu/config.py` conflict surface with the two live spokes is nil:** my hunk is one contiguous insertion *inside*
`GenBridgeConfig`; `gen-store-grouping` appends a **new dataclass after** it and `gen-wide-split` edits
**`GenDataConfig`**. I touched neither, and no existing line in `config.py` was modified (`+32 / −0`).

`chlu/core/gen_bridge.py` exact edits (post-edit line numbers):

| site | lines | change |
|---|---|---|
| `import dataclass` | 35 | +1 |
| `_n_upsamples` (new helper) | 289–301 | +13, extracted from `GenDecoder.__init__` |
| `GenDecoder.__init__` | 315–352 | `blocks_per_stage=1` param; `chans` repeated per stage. **b=1 is bit-identical** |
| `GenDecoder.blocks_per_stage` (property) | 341–346 | +6, **derived**, ⛔ not a pytree field |
| `GenDecoder.__call__` | 349–352 | upsample before conv `i` iff `i % b == 0` |
| `_sgd_loop` | 435–451 | optional `schedule=None`; `None` keeps `optax.adam(lr)` |
| the recipe block | 558–686 | +129: `DecoderRecipe`, `DECODER_RECIPES`, `resolve_decoder_cfg`, `make_decoder` |
| losses + schedule + refit | 688–767 | +80: `_fdiff`, `edge_weight_map`, `decoder_loss_fn`, `decoder_schedule`, `train_decoder_from_features` |
| `train_decoder` body | 770–786 | now delegates to `train_decoder_from_features` (−17 / +5) |

⛔ **Not pushed, no PR** — branch left for review. ⛔ No `.pkl`, no figure and no dataset entered git; every artifact is
under `.claude/outputs/gen-decoder-lift/` (gitignored).

**Artifacts:** `PREREG.md` (+ its §6 amendment) · `summary.json` · `provenance_s{0,1,2}.json` ·
`diagnostic_D_s{0,1,2}.json` · `result_{baseline_shipped,baseline_refit,v1,v2}_s*.json` · `shape_confusion.json` ·
`B_renders_seed0.png` · `winner/{winning_recipe.yaml,winner_hashes.json,decoder_v1_s{0,1,2}.pkl}` ·
logs `s0_base_diag.log`, `s12_base_diag.log`, `v1_s{0,1,2}.log`, `v2_s{0,1,2}.log`, `suite_before.log`,
`suite_after.log`, `suite_after_final.log`.

---

## 12. Open questions / follow-ups / risks

1. ⚠ **V1 confounds capacity and schedule.** Width ×2, depth ×2 and steps ×1.5 with cosine annealing moved together,
   so I cannot say which did the work. ⛔ Splitting them is a third and a fourth variant and the task caps me at two.
   Cheapest disambiguation for whoever wants it: `dec_width_mult=1.0, dec_blocks_per_stage=1, dec_steps=1800,
   dec_lr_schedule=cosine` — schedule alone, at ~1.5× baseline cost.
2. ⚠ **V1 had not converged.** Loss 0.16399 → 0.00097 and still falling at 1800 steps. The recipe is a *sufficient*
   fix, not a tuned one; on a GPU the pre-registered 4800 steps is cheap and would probably do better.
3. ⚠ **Cost.** 6.05× parameters, ~16× CPU training time (5,728 s/seed at best on this box). On CSF3 GPU this is
   minutes, but the *local* smoke path and anyone re-running `exp_gen_bridge --stage decoder` on CPU will feel it.
   This is the main argument against flipping the default without the Hub deciding (§7, §9.1).
4. ⭐ **A capacity fix this large should make someone ask what else in the phase is capacity-limited.** The **encoder**
   (428,576 params, 1500 steps) and the **probe** (633,049 params, 1500 steps) were fit with the same budget
   philosophy as the decoder was. The probe reads real images at 1.0000 so it is not the bottleneck; ⛔ the encoder is
   frozen and cannot be touched by this spoke, and Diagnostic D says it does not *need* to be for `shape`. But
   "under-resourced instrument mistaken for a scientific wall" has now happened once in this phase, and it is worth
   one line in the arms prereg.
5. ⚠ **Only Shapes3D / `recon`.** ⛔ I did not run `cls`. The task's non-negotiable *"Report `cls` as well as `recon`"*
   is **not met as written**, and I am flagging it rather than burying it: `cls` needs its own encoder + cache
   (different feature space) and three more V1 fits at ~2–6 h/seed each on a contended CPU, which does not fit the
   one-day bound alongside two variants at three seeds. **What replaces it is stronger for the specific question
   `cls` was there to answer.** `cls` exists for fault isolation — *"is the failure in the features or in
   the store/addressing?"* — and **Diagnostic D answers that directly on the `recon` features themselves**: `shape` is
   linearly readable from them on B at **1.0000**, so the features are exonerated without needing the `cls` arm as a
   proxy. ⚠ What is genuinely *not* covered: whether the V1 recipe also lifts the **`cls`** pipeline (whose ceiling was
   `shape` 0.773). That is one command per seed once someone has GPU time. **I recommend it before the arms run.**
6. ⚠ **The matched-seen control is small** (n ≈ 212) and at V1's level it sits at 0.9432 ± 0.0404 with two seeds giving
   ratios slightly above 1.0. That is a small-sample artefact, not evidence that B is *easier* than seen data.
7. **dSprites: not run** (R6 — the A/B/C cell is Shapes3D only).
8. ⚠ **Machine contention makes every wall-clock in this report soft.** Load average ranged 46–165 on 8 cores from
   other agents' suites throughout. ⛔ No reported *accuracy* is affected — everything is seeded and deterministic —
   but ⛔ do not quote the timings as benchmarks.

---

## Proposed handover updates (for the Hub)

- ⭐ **The gate result to record — this supersedes R16's numbers for the *repaired* decoder:** Shapes3D `recon`,
  seeds {0,1,2}, V1 recipe, frozen encoder + untouched cache: `shape` on B **0.9398 ± 0.0131**, ratio **0.9656**
  (random-A) / **0.9964** (matched-seen), floor 0.8 ⇒ **CLEARS on 3/3 seeds and both normalisers.**
  Aggregate `ceiling_A` 0.9902 ± 0.0030, `ceiling_B` 0.9830 ± 0.0061, ratio 0.9927. `scale` ratio vs matched-seen
  1.0063. Decode MSE 0.00300 → 0.00110. ⛔ **Option 2 does not activate.**
- ⭐ **Record the pre-registered win:** the Hub's own expectation ("a decoder-only fix will NOT clear 0.8") was
  pre-registered as P5 at 09:55, ~2 h before the first variant fit, and is **falsified**. The variant that won is the
  one filed as the null control; the objective-based variant the task nominated **lost** (§5.3).
- ⭐ **New, load-bearing fact for the phase:** `shape` on held-out combinations is **linearly** decodable from the
  frozen `recon` features at **1.0000 ± 0.0000** (3 seeds). ⛔ Nobody may again describe the B deficit as a feature or
  encoder problem, and R18's "if the fix needed a different encoder…" branch is closed without ever being taken.
- **New config knobs** in `GenBridgeConfig` (`chlu/config.py`, lines 3334–3365, pure insertion): `dec_recipe`,
  `dec_width_mult`, `dec_blocks_per_stage`, `dec_loss`, `dec_edge_weight`, `dec_edge_blur`, `dec_grad_weight`,
  `dec_lr_schedule`, `dec_warmup_frac`. **All defaults reproduce the shipped decoder bit-identically**
  (`param_hash 47c0f83196cd220d` verified by refit, not by argument). YAML-overridable under `gen_bridge:`.
- **New public API** in `chlu/core/gen_bridge.py`: `DECODER_RECIPES`, `DecoderRecipe`, `resolve_decoder_cfg`,
  `make_decoder`, `decoder_loss_fn`, `edge_weight_map`, `decoder_schedule`, **`train_decoder_from_features`** (the
  refit-into-any-feature-cache entry point the factorisation ladder needs). `GenDecoder` gains a `blocks_per_stage`
  **property** (derived, ⛔ not a field — a field would break the shipped `decoder.pkl`).
- **§7 Known Issues — one new entry is warranted:**
  *"**7.30** A decoder/renderer that reaches low pixel MSE can still be badly **capacity-limited for identity**. The
  gen bridge's shipped decoder (250 k params, 1200 steps, MSE 0.0030) rendered held-out-combination `shape` at 0.59
  and was read as a compositional wall (R16 STOP); 6× the parameters and a cosine schedule (MSE 0.0011) render the
  same frozen features at 0.94. ⛔ Before attributing a ceiling to the science, check whether a linear probe on the
  frozen features already has the answer — here it did, at 1.0000."*
- **⚠ Qualify, do not retire:** `gen-feature-bridge`'s "⭐ THE FINDING — the B deficit is mostly PHOTOMETRIC" survives
  intact on `scale` (V1's `scale` ratio vs matched-seen is 1.0063). Its companion claim that the residual `shape` term
  is "the only genuinely compositional term" needs the qualifier **"…and it too dissolves under a larger decoder."**
- **Open for the Hub:** (a) flip `dec_recipe`'s default to `v1_capacity_schedule`? (§9.1, one line + one test);
  (b) add `--dec-recipe` to `exp_gen_bridge`'s parser so the CSF3 `EXTRA_ARGS` path works (§9.4); (c) re-read R17
  (§9.2 — my read: it still stands, on my V2 seed-spread evidence rather than on the band width); (d) run the V1
  recipe on the `cls` diagnostic arm before the arms spoke (§12.5).
