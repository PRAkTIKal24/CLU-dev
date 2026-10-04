# gen-encoder-pretrain — experiment-engineer report

**Task + acceptance criterion:** train self-supervised **reconstruction** encoders at reduced scale and find one whose
features **retain `scale` and `orientation`**; acceptance = a plain **PASS/FAIL against the ≥0.90 bar on SEEN data**, on
the **spread grid**, 3 seeds, with the current `recon` encoder as a full control row and the corner reported separately.
**Status: partial — the headline question is answered at 3 seeds; the reduced-scale arms grid is ragged.** Seeds
completed per arm: `random_init_hires` **3** · `recon_control` / `random_init` / `ae_plain` **2** ·
`recon_control_2x` / `ae_deep` / `mae` / `ae_hires` / `vae` **1**. ⛔ Every table row prints its own seed count, and
the run is resumable (`skip_if_exists`, per-(variant, seed) writes) — the missing seeds cost only their own compute.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 3 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⛔⛔ **The number that RETIRED this task is cleared by ZERO TRAINING.** The task file was retired mid-flight on
>    `gen-loss-attribution`'s `scale` **0.9352 ± 0.0058** / `orientation` **0.9915 ± 0.0018** for the shipped frozen
>    `recon` encoder. Under a **protocol-matched re-measurement** (8,192 fit rows · 1,500 steps · one linear head per
>    factor on standardised features · 3,000 Adam steps · batch 512 · lr 3e-3 · 12,288 probe-fit / 4,096 probe-eval
>    spread rows · 3 seeds), an **UNTRAINED random-init encoder of the identical architecture** scores
>    `scale` **0.9361 ± 0.0030** and `orientation` **0.9865 ± 0.0023** against the trained control's
>    **0.9408 ± 0.0072** / **0.9912 ± 0.0009**. The gap is **+0.0047** on both — **smaller than the control's own
>    seed spread on `scale` (0.0072)** — for **1,246× the compute** (2,492.7 s vs 2.0 s mean, and the 2.0 s is JIT,
>    not training). `random_init` **beats** the trained control on `scale` on **1 of 3 seeds**.
>    ⇒ ⛔ **"≥0.9 on seen data" does not measure the pretext. It measures that a 2,048-d random non-linear projection
>    of a 64×64 Shapes3D image is linearly decodable.** The retirement's *arithmetic* is right; the inference
>    *"recon already shipped a good representation"* is **unsupported** — the laundering control was missing at that
>    point and it fires.
>    ⭐ **This does NOT overturn `gen-loss-attribution`'s verdict.** A random encoder also carrying the factors makes
>    *"the encoder is innocent, the k-means codebook is the culprit"* **stronger**, not weaker.
>    **Owner: Hub — before `gen-encoder-unfreeze`'s results are read and before any full-scale cluster encoder run.**
> 2. ⚠ **R40.E's encoder falsifier is SATURATED and cannot discriminate between encoders.** Trained and untrained both
>    PASS. Any future encoder bar must be stated **as a margin over the D-matched random-init floor**, never as an
>    absolute accuracy. **Owner: Hub, when R40.E's remaining clause (*swap within the family* — ⛔ still NOT measured by
>    anyone, including this spoke) is scoped.**
> 3. ⚠ **ENV, will cost the next spoke a day.** With `jax_threefry_partitionable=True` (the JAX 0.9 default here)
>    `jax.random.split(key, n)` is **prefix-stable**: `split(k, 2)[1]` **IS** `split(k, 3)[1]`. Two arms that "differ
>    only by an extra subkey split" are **bit-identical, not decorrelated**. Measured, not assumed — this spoke's
>    `ae_plain` arm came out with the same parameter hash (`8bdef74f5a79fcfa`) *and* the same final loss
>    (`0.007310017943382263`) as the shipped control, on **both** completed seeds. **Owner: Hub, §7 Known Issues.**

---

## ⭐ DIAL DECLARATION (echoed before the first result, protocol §7)
- **Dial:** *the encoder's pretext.* ⛔ **The store was NOT touched. The encoder stays FROZEN.** Nothing was promoted;
  `chlu/core/gen_bridge.py` is byte-identical to `main` (assertion in §6).
- **Laundering controls (three, all full rows):** ① the current `recon` encoder — the shipped
  `gen_bridge.train_encoder(pretext="recon")` **called verbatim**, hash-asserted in the tests; ② `random_init`, an
  **untrained** conv stack of the identical architecture — the trivial substitute; ③ `random_init_hires`, the same at
  the **matched feature width** of the best trained arm. Plus ④ `recon_control_2x`, the **compute** control.
- **Falsifies:** `scale` **and** `orientation` ≥ 0.90 on the spread grid, on SEEN data, for some variant, before the
  store is touched.
- **Does NOT falsify:** losing on the hue factors (≥0.99 for every arm including the untrained ones); a corner-only
  pass; the VAE comparison arm trailing the AE headline.

## ⚠ THE TASK WAS RETIRED MID-FLIGHT — what I did about it
`.claude/tasks/gen-encoder-pretrain.md` acquired a *"⛔ RETIRED … Do not spawn"* banner **while this spoke was already
executing** (seed 0 of the reduced-scale grid was in its fourth arm). I was spawned before the banner and did not
spawn after it. Rather than discard the in-flight compute I **re-pointed the remaining budget at the retirement's own
premise** and ran the laundering control that the retiring number lacked. That control is §2 and it is this report's
headline. The originally-scoped arms grid is reported in §3, at the seed counts it actually reached.

---

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | **`b27a615`** (branch tip, **after** rebase). Every number below was produced at tree `a072d01`, whose content for `gen_encoders.py` / `exp_gen_pretrain.py` / `config.py`'s new block is identical to `b27a615` (the rebase touched only `config.py`'s merge anchors — see §7) |
| branch / base | `agent/experiment-engineer/gen-encoder-pretrain`, cut off **`main @ 994ca9b`**; ⚠ `main` **moved during the run** to **`81ba81b`** (`gen-loss-attribution` + `gen-encoder-unfreeze` merged) and the branch was **rebased onto local `main`**, 4 conflicts, all resolved keep-both (§7) |
| worktree / venv | `../CHLU-gen-pretrain`, **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0** |
| dataset · condition | `shapes3d` · `recomb2element` (Montero). Grid 480,000 rows = 10·10·10·8·4·15; `n_train` 479,928, `n_test` 72 |
| seeds | 0, 1, 2 — ⚠ **per-row seed counts differ and are printed in every table**. `±` is **population sd (ddof = 0)**, via the shipped `chlu.eval.gen_probe.summarize_per_factor` |
| encoder key chain | `PRNGKey(seed·10_007 + 20_260_909)` — the bridge's own chain, so the control arm is the shipped draw |
| probe key chain | `PRNGKey(seed·104_729 + 20_260_914)`, ⛔ never the encoder's |
| probe (⭐ identical for every arm, pool and seed) | `FeatureProbe` = `chlu.core.gen_bridge.FactorHeads` (one `eqx.nn.Linear` per factor) on **flattened, z-scored** features; `mlp` variant adds one SiLU hidden layer (256). Fitted with the **shipped** `gen_bridge._sgd_loop` + `_xent`. Scored by the **shipped** `chlu.eval.gen_probe.probe_accuracy` |
| standardisation | per-dimension mean/std of the **probe-fit rows only**; ⛔ never re-fit on eval rows (tested) |
| REDUCED-SCALE run (§3) | `n_enc_train` **4,096** · `n_probe_train` **4,096** · `n_probe_eval` **2,048** · `enc_steps` **400** (`recon_control_2x`: 800) · `enc_batch` 128 · `enc_lr` 1e-3 · `probe_steps` 1500 · `probe_batch` 256 · `probe_lr` 1e-3. ⛔ **Below the shipped bridge's 8,192 / 1,500 — a declared axis, not a caveat** |
| MATCHED run (§2) | `n_enc_train` **8,192** · `enc_steps` **1,500** (both = the shipped `GenBridgeConfig`) · `n_probe_train` **12,288** · `n_probe_eval` **4,096** · `probe_steps` **3000** · `probe_batch` **512** · `probe_lr` **3e-3** · `probe_kinds` `[linear]` — matched to `gen-loss-attribution`'s `WellProbe` settings |
| rows | `enc_train` drawn from `split.train_rows` exactly as the shipped `select_rows` (⛔ corner not excluded, because the shipped encoder sees corner rows and the control must too); probe pools drawn from the remainder **with the corner removed**; **disjointness asserted every seed** (`assert_rows_disjoint`, positive control in the tests) |
| CORNER pool | `exp_gen_store.matched_seen_rows` = the **seen** rows of Montero's photometric corner, 216 realised, 209–216 after removing any that landed in `enc_train` (the exact count is in each `result.json`) |
| ⛔ NOT used | any decoder, the store, `read_from_codes`, the address book, `gen_compose`, `gen_dream`, the external classifier, the held-out 72 rows. ⛔ **No feature cache was written, moved, re-fit or invalidated** |
| decoder trap | `dec_recipe` never read — this spoke renders **nothing**, so §7's `"baseline"` default is structurally unreachable here |
| wall clock | reduced-scale run **2 h 41 m** (stopped at the budget, mid-seed-1) · matched run **2 h 55 m** · hires-floor run **4 m**. CPU only, 8 cores / 16 GB. ⛔ No cluster |
| PREREG | `.claude/outputs/gen-encoder-pretrain/PREREG.md`, mtime **08:00:12**, ⛔ never edited since. First production artefact `run/recon_control/seed_0/result.json` **08:21:24**; first matched artefact `matched/random_init/seed_0/result.json` **08:38:51**; last **10:49:28**. ⇒ ⭐ **every number postdates the prereg on disk** |

---

## ⭐⭐ 2. THE HEADLINE — the laundering control at the ATTRIBUTION spoke's own scale and probe

**3 seeds, `n = 4,096` spread-grid rows, linear probe. ⛔ This is a FULL-SCALE measurement, not a reduced-scale one.**

### probe=linear · pool=grid · n=4096 · seeds=3
| variant | seeds | steps | floor_hue | wall_hue | object_hue | scale | shape | orientation | aggregate | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| **n_values_realised** | — | — | 10 | 10 | 10 | 8 | 4 | 15 | — | — |
| **chance** | — | — | 0.1000 | 0.1000 | 0.1000 | 0.1250 | 0.2500 | 0.0667 | — | — |
| **majority rate** | — | — | 0.1072 | 0.1086 | 0.1106 | 0.1287 | 0.2534 | 0.0747 | — | — |
| `random_init` | 3 | 0 | 0.9988 ± 0.0003 | 0.9989 ± 0.0005 | 0.9991 ± 0.0003 | 0.9361 ± 0.0030 | 0.9812 ± 0.0022 | 0.9865 ± 0.0023 | 0.9834 | **PASS** |
| `recon_control` | 3 | 1500 | 0.9995 ± 0.0002 | 0.9994 ± 0.0003 | 0.9993 ± 0.0004 | 0.9408 ± 0.0072 | 0.9960 ± 0.0010 | 0.9912 ± 0.0009 | 0.9877 | **PASS** |

**Reading it.**
1. ⭐ **`recon_control` reproduces `gen-loss-attribution` on an independent harness**: `scale` **0.9408 ± 0.0072** vs
   their **0.9352 ± 0.0058**; `orientation` **0.9912 ± 0.0009** vs their **0.9915 ± 0.0018**. Different rows, different
   probe implementation, different key chain, same answer. ⭐ The retiring number is **replicated**, not disputed.
2. ⛔⛔ **The untrained encoder is level with it.** Δ = **+0.0047** on `scale` and **+0.0047** on `orientation`, and the
   `scale` Δ is **smaller than the control's own across-seed sd (0.0072)**. Per seed, `scale`:
   `recon_control` [0.9380, 0.9338, 0.9507] vs `random_init` [0.9329, 0.9402, 0.9353] — the untrained encoder **wins
   seed 1**. On `orientation` the control wins 3/3 but by 0.0027 / 0.0034 / 0.0081.
3. ⭐ **The compute ledger makes the point unanswerable.** 2,492.7 s mean (1,870 / 4,246 / 1,362) against **2.0 s**
   (5.84 / 0.064 / 0.064 — and that is JIT + module construction; the arm runs **zero** optimisation steps).
   **≈1,246×.** ⛔ 1,500 Adam steps of pixel reconstruction buy **0.5 accuracy points** on the two factors the
   programme cares about.
4. ⚠ **And in the corner the untrained encoder is AHEAD on `scale`** — 0.8801 ± 0.0372 vs 0.8470 ± 0.0475, 3 seeds,
   and **both FAIL the 0.90 bar there**:

### probe=linear · pool=corner · n=209 · seeds=3
| variant | seeds | steps | floor_hue | wall_hue | object_hue | scale | shape | orientation | aggregate | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| **n_values_realised** | — | — | 2 | 2 | 2 | 3 | 3 | 3 | — | — |
| **chance** | — | — | 0.5000 | 0.5000 | 0.5000 | 0.3333 | 0.3333 | 0.3333 | — | — |
| **majority rate** | — | — | 0.5024 | 0.5024 | 0.5024 | 0.3445 | 0.3397 | 0.3349 | — | — |
| `random_init` | 3 | 0 | 0.9890 ± 0.0155 | 0.9969 ± 0.0044 | 0.9937 ± 0.0089 | 0.8801 ± 0.0372 | 0.9114 ± 0.0558 | 0.9859 ± 0.0167 | 0.9595 | **FAIL** |
| `recon_control` | 3 | 1500 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8470 ± 0.0475 | 1.0000 ± 0.0000 | 0.9968 ± 0.0045 | 0.9740 | **FAIL** |

   ⚠ ⭐ **This is the task's named trap running backwards:** a **grid PASS with a corner FAIL**. `n_values_realised`
   is 3 (of 8) for `scale` and 3 (of 15) for `orientation` in the corner, against 8 and 15 on the grid — ⛔ the two
   columns are not comparable and the corner number must never be quoted as the headline.

---

## 3. The originally-scoped arms grid, at REDUCED SCALE

⛔ **Reduced scale is a declared axis.** `n_enc_train` 4,096 and `enc_steps` 400 against the shipped bridge's 8,192 and
1,500. ⛔ **No row here may be compared with a full-scale baseline without that sentence attached.** Seed counts differ
per row and are printed; the run was stopped at the budget and is resumable.

### probe=linear · pool=grid · n=2048 · seeds=2
| variant | seeds | steps | floor_hue | wall_hue | object_hue | scale | shape | orientation | aggregate | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| **n_values_realised** | — | — | 10 | 10 | 10 | 8 | 4 | 15 | — | — |
| **chance** | — | — | 0.1000 | 0.1000 | 0.1000 | 0.1250 | 0.2500 | 0.0667 | — | — |
| **majority rate** | — | — | 0.1089 | 0.1162 | 0.1040 | 0.1343 | 0.2622 | 0.0728 | — | — |
| `recon_control` | 2 | 400 | 0.9990 ± 0.0005 | 0.9988 ± 0.0007 | 0.9988 ± 0.0002 | 0.6653 ± 0.0007 | 0.9531 ± 0.0020 | 0.8079 ± 0.0105 | 0.9038 | **FAIL** |
| `random_init` | 2 | 0 | 0.9961 ± 0.0015 | 0.9954 ± 0.0017 | 0.9944 ± 0.0007 | 0.8171 ± 0.0149 | 0.9458 ± 0.0024 | 0.9348 ± 0.0066 | 0.9473 | **FAIL** |
| `ae_plain` | 2 | 400 | 0.9990 ± 0.0005 | 0.9988 ± 0.0007 | 0.9988 ± 0.0002 | 0.6653 ± 0.0007 | 0.9531 ± 0.0020 | 0.8079 ± 0.0105 | 0.9038 | **FAIL** |
| `recon_control_2x` | 1 | 800 | 0.9976 ± 0.0000 | 0.9971 ± 0.0000 | 0.9985 ± 0.0000 | 0.7266 ± 0.0000 | 0.9683 ± 0.0000 | 0.8872 ± 0.0000 | 0.9292 | **FAIL** |
| `ae_deep` | 1 | 400 | 0.9980 ± 0.0000 | 1.0000 ± 0.0000 | 0.9971 ± 0.0000 | 0.6157 ± 0.0000 | 0.9219 ± 0.0000 | 0.6885 ± 0.0000 | 0.8702 | **FAIL** |
| `mae` | 1 | 400 | 0.9966 ± 0.0000 | 0.9966 ± 0.0000 | 0.9912 ± 0.0000 | 0.5776 ± 0.0000 | 0.8760 ± 0.0000 | 0.7451 ± 0.0000 | 0.8639 | **FAIL** |
| `ae_hires` | 1 | 400 | 0.9956 ± 0.0000 | 0.9956 ± 0.0000 | 0.9956 ± 0.0000 | 0.8467 ± 0.0000 | 0.9785 ± 0.0000 | 0.9609 ± 0.0000 | 0.9622 | **FAIL** |
| `random_init_hires` | 3 | 0 | 0.9891 ± 0.0005 | 0.9899 ± 0.0005 | 0.9881 ± 0.0020 | 0.8971 ± 0.0028 | 0.9714 ± 0.0043 | 0.9718 ± 0.0020 | 0.9679 | **FAIL** |
| `vae` | 1 | 400 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.6650 ± 0.0000 | 0.9341 ± 0.0000 | 0.9380 ± 0.0000 | 0.9229 | **FAIL** |

### probe=linear · pool=corner · n=213 · seeds=2
| variant | seeds | steps | floor_hue | wall_hue | object_hue | scale | shape | orientation | aggregate | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| **n_values_realised** | — | — | 2 | 2 | 2 | 3 | 3 | 3 | — | — |
| **chance** | — | — | 0.5000 | 0.5000 | 0.5000 | 0.3333 | 0.3333 | 0.3333 | — | — |
| **majority rate** | — | — | 0.5023 | 0.5023 | 0.5023 | 0.3380 | 0.3333 | 0.3380 | — | — |
| `recon_control` | 2 | 400 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5436 ± 0.0667 | 0.8326 ± 0.0688 | 0.8721 ± 0.0481 | 0.8747 | **FAIL** |
| `random_init` | 2 | 0 | 0.9931 ± 0.0069 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7295 ± 0.0112 | 0.8712 ± 0.0825 | 0.9906 ± 0.0094 | 0.9307 | **FAIL** |
| `ae_plain` | 2 | 400 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5436 ± 0.0667 | 0.8326 ± 0.0688 | 0.8721 ± 0.0481 | 0.8747 | **FAIL** |
| `recon_control_2x` | 1 | 800 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.6948 ± 0.0000 | 0.9061 ± 0.0000 | 0.9765 ± 0.0000 | 0.9296 | **FAIL** |
| `ae_deep` | 1 | 400 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4883 ± 0.0000 | 0.9296 ± 0.0000 | 0.8404 ± 0.0000 | 0.8764 | **FAIL** |
| `mae` | 1 | 400 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5493 ± 0.0000 | 0.9390 ± 0.0000 | 0.7840 ± 0.0000 | 0.8787 | **FAIL** |
| `ae_hires` | 1 | 400 | 0.9718 ± 0.0000 | 0.9671 ± 0.0000 | 0.9953 ± 0.0000 | 0.7934 ± 0.0000 | 0.9108 ± 0.0000 | 0.9624 ± 0.0000 | 0.9335 | **FAIL** |
| `random_init_hires` | 3 | 0 | 0.9753 ± 0.0257 | 0.9567 ± 0.0309 | 0.9474 ± 0.0308 | 0.7410 ± 0.0616 | 0.8710 ± 0.0274 | 0.9411 ± 0.0241 | 0.9054 | **FAIL** |
| `vae` | 1 | 400 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.6244 ± 0.0000 | 0.8592 ± 0.0000 | 0.9718 ± 0.0000 | 0.9092 | **FAIL** |

### the capacity check (MLP probe, spread grid) — ⛔ reported beside the headline, never instead of it
### probe=mlp · pool=grid · n=2048 · seeds=2
| variant | seeds | steps | floor_hue | wall_hue | object_hue | scale | shape | orientation | aggregate | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| **n_values_realised** | — | — | 10 | 10 | 10 | 8 | 4 | 15 | — | — |
| **chance** | — | — | 0.1000 | 0.1000 | 0.1000 | 0.1250 | 0.2500 | 0.0667 | — | — |
| **majority rate** | — | — | 0.1089 | 0.1162 | 0.1040 | 0.1343 | 0.2622 | 0.0728 | — | — |
| `recon_control` | 2 | 400 | 0.9990 ± 0.0000 | 0.9985 ± 0.0005 | 0.9961 ± 0.0015 | 0.7256 ± 0.0010 | 0.9529 ± 0.0002 | 0.8142 ± 0.0120 | 0.9144 | **FAIL** |
| `random_init` | 2 | 0 | 0.9958 ± 0.0017 | 0.9949 ± 0.0012 | 0.9927 ± 0.0015 | 0.7988 ± 0.0078 | 0.9436 ± 0.0012 | 0.9031 ± 0.0012 | 0.9382 | **FAIL** |
| `ae_plain` | 2 | 400 | 0.9990 ± 0.0000 | 0.9985 ± 0.0005 | 0.9961 ± 0.0015 | 0.7256 ± 0.0010 | 0.9529 ± 0.0002 | 0.8142 ± 0.0120 | 0.9144 | **FAIL** |
| `recon_control_2x` | 1 | 800 | 0.9966 ± 0.0000 | 0.9976 ± 0.0000 | 0.9966 ± 0.0000 | 0.7881 ± 0.0000 | 0.9629 ± 0.0000 | 0.8848 ± 0.0000 | 0.9377 | **FAIL** |
| `ae_deep` | 1 | 400 | 0.9980 ± 0.0000 | 0.9980 ± 0.0000 | 0.9961 ± 0.0000 | 0.6694 ± 0.0000 | 0.9282 ± 0.0000 | 0.7241 ± 0.0000 | 0.8857 | **FAIL** |
| `mae` | 1 | 400 | 0.9961 ± 0.0000 | 0.9976 ± 0.0000 | 0.9941 ± 0.0000 | 0.6152 ± 0.0000 | 0.8809 ± 0.0000 | 0.7197 ± 0.0000 | 0.8673 | **FAIL** |
| `ae_hires` | 1 | 400 | 0.9966 ± 0.0000 | 0.9951 ± 0.0000 | 0.9951 ± 0.0000 | 0.8105 ± 0.0000 | 0.9785 ± 0.0000 | 0.9165 ± 0.0000 | 0.9487 | **FAIL** |
| `random_init_hires` | 3 | 0 | 0.9899 ± 0.0008 | 0.9886 ± 0.0006 | 0.9881 ± 0.0010 | 0.8687 ± 0.0098 | 0.9689 ± 0.0040 | 0.9505 ± 0.0037 | 0.9591 | **FAIL** |
| `vae` | 1 | 400 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9985 ± 0.0000 | 0.7617 ± 0.0000 | 0.9556 ± 0.0000 | 0.9360 ± 0.0000 | 0.9420 | **FAIL** |

### compute
| variant | family | steps | enc params | pretext-head params | D | feat_shape | train s (mean) | final loss |
|---|---|---|---|---|---|---|---|---|
| `recon_control` | recon | 400 | 428,576 | 250,467 | 2048 | [128, 4, 4] | 508.6 | 0.006977 |
| `random_init` | random | 0 | 428,576 | 250,467 | 2048 | [128, 4, 4] | 0.0 | nan |
| `ae_plain` | ae | 400 | 428,576 | 250,467 | 2048 | [128, 4, 4] | 1430.0 | 0.006977 |
| `recon_control_2x` | recon | 800 | 428,576 | 250,467 | 2048 | [128, 4, 4] | 588.8 | 0.004453 |
| `ae_deep` | ae | 400 | 428,576 | 453,987 | 2048 | [128, 4, 4] | 706.9 | 0.007877 |
| `mae` | mae | 400 | 428,576 | 250,467 | 2048 | [128, 4, 4] | 478.5 | 0.013132 |
| `ae_hires` | ae | 400 | 166,048 | 241,155 | 8192 | [128, 8, 8] | 795.4 | 0.003769 |
| `random_init_hires` | random | 0 | 166,048 | 241,155 | 8192 | [128, 8, 8] | 1.2 | nan |
| `vae` | vae | 400 | 461,600 | 250,467 | 2048 | [128, 4, 4] | 615.8 | 0.016243 |

### per-seed bar verdicts (linear · grid)
- `recon_control`: ['FAIL', 'FAIL']  seeds=[0, 1]
- `random_init`: ['FAIL', 'FAIL']  seeds=[0, 1]
- `ae_plain`: ['FAIL', 'FAIL']  seeds=[0, 1]
- `recon_control_2x`: ['FAIL']  seeds=[0]
- `ae_deep`: ['FAIL']  seeds=[0]
- `mae`: ['FAIL']  seeds=[0]
- `ae_hires`: ['FAIL']  seeds=[0]
- `random_init_hires`: ['FAIL', 'FAIL', 'PASS']  seeds=[0, 1, 2]
- `vae`: ['FAIL']  seeds=[0]


**Reading it.**
1. ⛔ **PASS/FAIL, plainly: FAIL.** No variant reaches 0.90 on **both** `scale` and `orientation` on the spread grid
   with the headline linear probe — **0 of 9 rows**, at any seed. Per the pre-registered stop rule I **stopped**: the
   store was not touched and no cache was promoted.
2. ⛔⛔ **Every trained arm is at or below its own untrained floor.** `recon_control` (400 steps) `scale` **0.6653**
   vs `random_init` (0 steps) **0.8171**; `ae_hires` (400 steps, D = 8192) **0.8467** vs `random_init_hires`
   (0 steps, D = 8192) **0.8971**. ⚠ ⭐ **The D-matched floor is what makes the second comparison readable, and it was
   added only after `ae_hires` looked like a winner against the D = 2048 floor** — its apparent +0.03 over
   `random_init` was **feature width**, and against a width-matched floor it is **−0.05**.
3. ⭐ **The compute control answers "did you just train longer?" — and the answer is that longer training helps but
   never catches the floor.** `recon_control` 400 → 800 steps moves `scale` 0.6653 → 0.7266 and `orientation`
   0.8079 → 0.8872; extrapolating to 1,500 steps + 8,192 rows lands at the §2 measurement of 0.9408 / 0.9912 — which
   is still only level with a zero-step encoder. ⇒ ⛔ **the reduced-scale FAIL is compute-limited, and that is exactly
   why §2 exists**: the question is not answered at reduced scale and I do not claim it is.
4. **The declared comparison arm.** `vae` (β = 0.05) `scale` **0.6650** / `orientation` **0.9380** at 1 seed: level
   with the AE control on `scale`, clearly better on `orientation`, ⛔ still below the untrained floor on both.
   ⚠ 1 seed — not a result, a pointer.
5. `mae` is the **worst** arm on `scale` (0.5776): masking half the patches and scoring only the masked pixels makes
   the object's extent *less* recoverable, not more. `ae_deep` (a deeper throwaway decoder) is second worst — ⛔ more
   reconstruction pressure did **not** buy factor retention. ⚠ Both 1 seed.
6. **`ae_plain` is bit-identical to `recon_control`**, to the last digit, on both seeds — see reconciliation item 3.
   ⇒ ⭐ this spoke's loop is the shipped loop, verified rather than asserted.
7. ⚠ **The MLP capacity check does not rescue anything**: no row clears the bar there either, and on the *untrained*
   arms the MLP is **worse** than the linear probe (`random_init` `scale` 0.7988 vs 0.8171), i.e. the features are
   already close to linearly separable and the extra layer costs sample efficiency at 4,096 fit rows.

---

## 4. ⭐ The pre-registered scorecard (⛔ failures reported as findings)

| # | prediction (filed before any probe) | measured | |
|---|---|---|---|
| **P1** | `recon_control` hues ≥ 0.90 each, central 0.97 | 0.9990 / 0.9988 / 0.9988 | ⭐ **HELD** |
| **P2** | `recon_control` `scale` central **0.55**, interval 0.35–0.75 | **0.6653 ± 0.0007** | ⭐ **HELD** (inside interval) |
| **P3** | `recon_control` `orientation` central **0.60**, interval 0.40–0.85 | **0.8079 ± 0.0105** | ⭐ **HELD** (inside interval) |
| **P4** | best AE/MAE beats `recon_control` on `scale` by **+0.12** (0.00–0.30) | `ae_hires` **+0.181** (⚠ 1 seed vs 2) | ⭐ **HELD** — ⛔ but see §3.2: the margin is feature **width**, not representation. Against the D-matched floor it is **−0.050** |
| **P5** | `vae` scores **below** the best AE on `scale` | 0.6650 vs `ae_hires` 0.8467 | ⭐ **HELD** |
| **P6** | `recon_control_2x` gains **+0.05** on `scale` (0.00–0.15) | **+0.0613** (0.6653 → 0.7266) | ⭐ **HELD** |
| **P7** | corner `scale` > grid `scale` for **every** arm | ⛔ **FALSE for 11 of 11 rows** (9 reduced-scale + 2 matched). `recon_control` 0.5436 < 0.6653 · `random_init` 0.7295 < 0.8171 · `ae_hires` 0.7934 < 0.8467 · `random_init_hires` 0.7410 < 0.8971 · matched `recon_control` 0.8470 < 0.9408. ⭐ `orientation` goes the **other** way (corner > grid for most arms) — the effect is specific to `scale` | ⛔ **FALSIFIED, unanimously** |
| **P8** | **HEADLINE: no variant clears 0.90 on both on the spread grid at reduced scale ⇒ FAIL** | FAIL, 0 of 9 rows, every seed | ⭐ **HELD** |

⛔ **P7's failure is a finding, not a miss.** I predicted the corner would be *easier* because it realises 3 classes
instead of 8. It is **harder** — for every arm. The corner's three `scale` values are the **three largest** (indices
5, 6, 7 of 8) and its three `orientation` values the three most extreme (12, 13, 14 of 15): adjacent extreme values of
an ordered factor, in a saturated-hue corner, at n = 213 (209 in the matched run). ⇒ ⚠ **"fewer classes" is not
"easier"**, and any future
reader normalising a corner number by `1/n_values_realised` will over-credit it. This is the same corner where
`gen-navigation` measured `scale` fidelity **below** chance.

⭐ **The two worlds (PREREG §6) are adjudicated, and it is World A's encoder clause:** `recon_control` **does** clear
~0.90 at shipped scale ⇒ the encoder is **not** the defect ⇒ this spoke's answer to *"is a new encoder needed to fix
`scale`/`orientation`?"* is **NO** — in agreement with `gen-loss-attribution`. ⛔ **But the clause is cleared by an
untrained encoder too**, so it is evidence about the *input*, not about the *pretext*.

---

## 5. ⛔ WHAT A PROMOTED CACHE WOULD VOID — the re-measurement list

⛔ **Nothing here was promoted, and the question is now moot** (§2: no candidate beat the shipped encoder, and the
shipped encoder is already at the random-projection ceiling). The list is filed because the task requires it and
because `gen-encoder-unfreeze` will face exactly the same bill.

A changed encoder changes the **feature cache**, and every phase-1 number is computed on that cache. ⭐ **Refusal is
RE-MEASURED, never assumed**, and ⭐ **`knn_proto` gets the same features** or the comparison is rigged.

| # | must be re-measured | why |
|---|---|---|
| 1 | the **feature cache** itself: `encoder.pkl` hash, `D`, `feat_shape`, checksum, `whitening_feasibility` | `ae_hires` alone moves `D` 2048 → **8192**, which changes the LACE whitening budget and the store byte ledger |
| 2 | the **ONE shared decoder** — refit from scratch | a `DecoderRecipe` is refit per feature space by construction (R19/`gen-decoder-lift`); the shipped `decoder.pkl` is **not** transferable |
| 3 | `ceiling_A`, `ceiling_B`, the **R3.1 gate verdict** | measured by pushing *cached features* through *that* decoder |
| 4 | `gen-decoder-lift`'s **0.5926 → 0.9398 `shape` lift** and its 0.9964 / 0.9656 ratios | same reason |
| 5 | the **store**: the eigengap `K`, `store_plan.json`, the correlation partition, every per-well k-means codebook, all per-well×factor MI | the store trains on the cache. ⛔ `K` is a **pinned ladder constant** — a new cache's eigengap may not agree with it, and that conflict needs a ruling, not a silent re-pin |
| 6 | `gen-store-grouping`'s `delete_zero_delta_aggregate = +0.4752` and the +0.1504 / +0.2015 correlation-vs-random margins | store-derived |
| 7 | **`knn_proto` and LACE**, on the **same** new features | ⭐ R40.E is explicit; a rival scored on the old cache is not a rival |
| 8 | the `524,288 B` **byte-match target** (`store_bytes.json`) | a function of `K` × codebook × `D` |
| 9 | ⭐ **refusal**: `out_of_range`, invalid-combination detection, the whole C set, the `out_of_range`-vs-kNN **tie** | ⛔ re-measured, never assumed |
| 10 | the **composition / generation headline table** (`gen-cell-harness`), incl. the A7 redundant arm and `gen-compose`'s read | every cell is a cache number |
| 11 | `gen-commitment-collapse` §7 per-factor fidelity and `gen-navigation`'s `NAV`, 0.229 / 0.236, endpoint-code identity, and the `scramble` control | all read the store, which reads the cache |
| 12 | `gen-metric-invariance`'s **DCI-C** (the R1 unblock gate) | computed on the cache |
| 13 | `gen-loss-attribution`'s three-point profile (9.5 % / 90.5 % split, retention 1.0037, the random-code control) | the encoder point *is* the cache |
| 14 | the **dSprites** second-dataset numbers, incl. `0.9616` | if the recipe changes, it changes there too |
| 15 | ⛔ **NOT** the external classifier | it reads **raw images** and never a cached feature — ⭐ it is the one instrument a cache change leaves standing |

⚠ **And the cache non-uniformity survives any promotion:** seed 0 is `scope="full"` (480,000 rows), seeds 1–2 are
`scope="used"` (9,288) and contain **none** of the matched-seen control rows.

---

## 6. ⭐ The bit-identity assertion — the existing encoder path did not change

| check | result |
|---|---|
| `git diff main -- chlu/core/gen_bridge.py` | **empty** |
| blob sha of `chlu/core/gen_bridge.py` on `main` vs on my branch | **`7de6316…` == `7de6316…`** |
| `git diff --stat main..HEAD -- gen_store.py gen_compose.py gen_dream.py chlu/eval/ scripts/csf3/` | **empty** — ⛔ none of the forbidden paths touched |
| files changed vs `main` | `chlu/config.py` (+129/−1, the −1 is a missing-trailing-newline marker), `chlu/core/gen_encoders.py` (new), `chlu/experiments/exp_gen_pretrain.py` (new), `tests/test_gen_pretrain.py` (new) |
| runtime assertion (test) | `test_the_control_arm_is_the_SHIPPED_encoder_path_bit_for_bit`: `param_hash(train_variant(..., "recon_control"))` **==** `param_hash(gen_bridge.train_encoder(..., "recon", ...))` for the same key and config |
| guard (test, positive control) | `test_a_control_arm_whose_config_DRIFTED_is_refused`: if `gen_pretrain` and `gen_bridge` disagree on any like-for-like field the control arm **raises and names the field**, rather than silently becoming a different encoder |
| borrowed-config hygiene (test) | `test_the_control_arm_restores_the_bridge_config_it_borrowed` — `gen_bridge.enc_steps` is restored in a `finally` |
| artifacts | every byte written under `.claude/outputs/gen-encoder-pretrain/`. ⛔ **No feature cache written**; ⛔ nothing under `runs/` read, moved or overwritten |

---

## 7. Git footprint

- **Branch:** `agent/experiment-engineer/gen-encoder-pretrain` · ⛔ not pushed, no PR, left for review.
- **Worktree `../CHLU-gen-pretrain` was REMOVED after verifying from the MAIN checkout that the branch ref had
  advanced** (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-encoder-pretrain`
  → all 5 commits, `b27a615` at the tip), per protocol §3.2's wave-4 precedent. ⭐ Freed a slot against R5's
  4-active-worktree cap (8 were live when I finished).
- **Base at spawn `main @ 994ca9b`; rebased onto local `main @ 81ba81b`** after `main` moved mid-run
  (`gen-loss-attribution` + `gen-encoder-unfreeze` merged). ⛔ Not `origin/main` (§7.21).
- **Commits (post-rebase):**
  | hash | subject |
  |---|---|
  | `a19634a` | candidate reconstruction encoders that must KEEP scale and orientation |
  | `35d4b43` | the pretrain harness: spread grid vs corner, one bar, two controls |
  | `cd16265` | tests: the control IS the shipped path, and a corner-only pass FAILS |
  | `17147cd` | test: `ae_plain` IS the control, and why splitting one more subkey does not decorrelate |
  | `b27a615` | add the D-MATCHED random floor for the hires arm |
- **Conflicts: 4, all in `chlu/config.py`, all at the shared EOF/`CHLUConfig` append anchor shared with
  `gen-encoder-unfreeze`.** Resolved **keep-both, peer block first**, then verified mechanically:
  `git diff main -- chlu/config.py | grep -c '^-[^-]'` → **1**, and that one line is
  `-# --- END gen-encoder-unfreeze ---` reappearing immediately with a trailing newline added (`\ No newline at end of
  file` on `main`'s side). ⇒ ⛔ **zero content deleted from the peer's block.**
- **Files touched:** `chlu/config.py` (one additive dataclass at EOF + the three house wiring sites:
  the `CHLUConfig` field, `load_config`, `save_config`), `chlu/core/gen_encoders.py` (new),
  `chlu/experiments/exp_gen_pretrain.py` (new), `tests/test_gen_pretrain.py` (new).
- ⛔ **No CLI hook added** — the gen experiments are all standalone `python -m chlu.experiments.exp_gen_*` with their
  own argparse, and the task's file list does not include `chlu/cli/experiment_cmd.py`.

## 8. Tests — passed / skipped / failed, with the data root set

```
CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen \
  uv-venv python -m pytest tests/test_gen_pretrain.py -q
→ 33 passed, 0 skipped, 0 failed, 2 warnings (21.72 s)     [on the REBASED tip b27a615]

… tests/test_gen_pretrain.py tests/test_gen_attribution.py \
  tests/test_gen_encoder_unfreeze.py tests/test_gen_bridge.py -q -rs
→ 118 passed, 0 skipped, 0 failed, 12 warnings (55.44 s)   [the neighbours my config edit could break]

ruff check (config.py, gen_encoders.py, exp_gen_pretrain.py, test_gen_pretrain.py) → All checks passed!
```
⛔ **No full suite** (task instruction; also the 16 GB / 8-core constraint).
⚠ `ruff format --check` reports my three files as "would reformat" — so do **16 of the 16** existing `gen_*` modules
on `main`, so the house standard here is `ruff check`, not `ruff format`. ⛔ I did not reformat anything.

**The load-bearing tests (the ones that can fail on purpose):**
`test_the_control_arm_is_the_SHIPPED_encoder_path_bit_for_bit` · `test_a_control_arm_whose_config_DRIFTED_is_refused`
(positive control) · `test_a_CORNER_only_pass_does_NOT_clear_the_bar` (⭐ the exact way the task said this spoke could
be wrong) · `test_an_MLP_only_pass_is_reported_SEPARATELY_and_never_as_PASS` ·
`test_the_row_disjointness_assertion_FIRES` (positive control) · `test_no_augmentation_contrastive_arm_exists` (a
**design** assertion: such an arm would optimise away the two factors being measured) ·
`test_the_hires_arm_has_a_D_MATCHED_random_floor`.

## 9. Artifacts
```
.claude/outputs/gen-encoder-pretrain/
  PREREG.md                     ⭐ filed before the first probe, never edited
  matched/summary.json  tables.md   random_init|recon_control / seed_(0, 1, 2) / result.json   ← §2, 3 seeds
  run/summary.json      tables.md   <variant>/seed_<s>/result.json                           ← §3, ragged seeds
  run/bar_scale_orientation.png    matched/bar_scale_orientation.png
  run.log   matched.log   hires_floor.log
.claude/scratch/gen-encoder-pretrain/   matched.yaml (the §2 config), mktables.py, reagg.py
```
Every `result.json` carries its own full flag block, `n_values_realised` per pool, row counts, parameter hashes,
parameter counts and wall clock. ⭐ **Resumable:** re-running the driver with the same `--out` skips every
`(variant, seed)` already on disk, so §3's missing seeds cost only their own compute.

## 10. Open questions / follow-ups / risks
1. ⭐ **The obvious next control, ~10 min, not run:** `random_init` at **shipped scale with the store fitted on its
   features**. If a store built on *random* features composes as well as the shipped one, the phase's
   "learned-from-data" sentence needs re-scoping. ⛔ I did not run it — the store is out of my scope by the task file.
2. ⚠ **`swap within the family`** — R40.E's *other* falsifier clause — is **still not measured by anyone**
   (`gen-loss-attribution` says so too). ⛔ Nothing in this report bears on it.
3. ⚠ **§3 is 1-seed for `ae_deep` / `mae` / `ae_hires` / `vae` / `recon_control_2x`.** Those rows are pointers, not
   results. Finishing them costs ≈ 110 min of the same harness with no code change.
4. ⚠ **Risk I could not close:** my probe is *matched to* `gen-loss-attribution`'s described settings but is not
   literally its `WellProbe` (that module landed on `main` only after my base was cut). The agreement of my
   `recon_control` with their number (0.9408 vs 0.9352; 0.9912 vs 0.9915) is the evidence that the two instruments
   read the same thing; ⛔ it is not a proof of identity.
5. ⚠ **`random_init_hires` PASSES on 1 of 3 seeds** (seed 2: `scale` 0.9004, `orientation` 0.9697). Reported as FAIL
   on the 3-seed mean (0.8971). ⛔ An untrained encoder sitting *on* the bar is one more reason the bar cannot stand
   as an absolute.
6. **Cost note for the cluster step:** at shipped scale one `recon` encoder is ~25–70 min of CPU here. The full-scale
   encoder work R40.F.6 schedules should budget from the 4,246 s worst case, not the 1,362 s best.

---

## Proposed handover updates (for the Hub)

- **§7 NEW (result, binding — ⛔ highest priority):** ⭐⭐ **An UNTRAINED random-init encoder matches the shipped
  `recon` encoder on the `scale`/`orientation` probe.** Matched protocol, 3 seeds, spread grid, `n = 4,096`:
  `random_init` `scale` **0.9361 ± 0.0030** / `orientation` **0.9865 ± 0.0023** vs `recon_control`
  **0.9408 ± 0.0072** / **0.9912 ± 0.0009** — Δ **+0.0047** on both, *inside* the control's own seed spread on
  `scale`, for **≈1,246×** the compute; the untrained arm wins `scale` on 1/3 seeds. ⇒ ⛔ **"the encoder retains
  `scale` and `orientation`" is a statement about a random non-linear projection of the image, not about the
  reconstruction pretext.** ⭐ It **strengthens** `gen-loss-attribution`'s verdict (the encoder is innocent) and
  **removes** any support for *"recon already shipped a good representation."*
- **§7 NEW (instrument, binding):** ⛔ **R40.E's "≥0.9 on seen data" bar is SATURATED** — trained and untrained both
  PASS. ⭐ **Every future encoder bar must be a margin over the D-matched random-init floor.** Absolute accuracy on a
  factor probe over a wide conv feature map does not discriminate between encoders.
- **§7 NEW (trap, measured twice):** ⚠ **Feature WIDTH masquerades as representation.** `ae_hires` (D = 8192) beat the
  D = 2048 untrained floor by +0.03 and lost to the **D-matched** floor by −0.05. ⛔ A probe comparison across arms of
  different `D` is not a comparison. This cost one extra control to catch and it caught it in 4 minutes.
- **§7 NEW (ENV, standing):** ⚠ with `jax_threefry_partitionable=True` (JAX 0.9 default here) **`jax.random.split` is
  prefix-stable**: `split(k, 2)[1] == split(k, 3)[1]`. Arms intended to differ only by an extra subkey split come out
  **bit-identical**. Measured: this spoke's `ae_plain` matched the shipped control's parameter hash
  (`8bdef74f5a79fcfa`) and final loss (`0.007310017943382263`) exactly, on both seeds.
- **§7 NEW (metric, corrects a natural assumption):** ⚠ **In Montero's corner, FEWER realised classes is HARDER, not
  easier.** `scale` corner accuracy < grid accuracy for **11 of 11** arm-rows (9 reduced-scale + 2 matched) despite
  3 realised values vs 8 — and `orientation` goes the **other** way, so the effect is `scale`-specific. The corner's
  three `scale` values are the three largest and its three `orientation` values the three most extreme — adjacent
  extremes of an ordered factor. ⛔ Do not normalise a corner number by `1/n_values_realised` and read it as
  "comfortably above chance".
- **§7 CONFIRMED (discipline):** `chlu/config.py`'s EOF append anchor is now shared by **three** gen spokes
  (`gen-encoder-unfreeze`, `gen-encoder-pretrain`, and whatever is next). The keep-both resolution works, but the
  mechanical check that made it safe was `git diff main -- chlu/config.py | grep -c '^-[^-]'` → expect **0**
  (1 here, and it is the trailing-newline marker). ⭐ Worth making that grep the standard post-rebase check.
- **⚠ Process note, not a §7 item:** this task's file was **retired mid-execution**. The retirement was correct on its
  own terms and I did not re-litigate it — but the spoke was already running, and the budget it had left produced the
  control that re-scopes the retiring number. ⭐ If a task is retired after spawn, it is worth telling the spoke what
  the *new* question is rather than only that the old one is closed.
