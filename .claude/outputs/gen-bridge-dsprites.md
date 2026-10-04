# gen-bridge-dsprites — experiment-engineer report

**Task + acceptance criterion:** fit the encoder + decoder on **dSprites** and measure the **R3 decoder-ceiling gate on
the compositional term**. One acceptance criterion: **the gate verdict.**

**Status: done.** All 8 acceptance items met, **3 of 3 seeds complete** (none NOT RUN).

# ⭐⭐ VERDICT: **PASS — 3/3 seeds. The instrument stack exists on dSprites.**
**In-corner `shape` balanced accuracy, held-out `0.9616 ± 0.0099`, seen normaliser `0.9553 ± 0.0089`, ratio
`1.0066 ± 0.0014`, AUROC `0.9991 ± 0.0008`** against the pre-registered rule (≥0.75 absolute AND ≥0.80 of the seen
normaliser). Per seed: **PASS / PASS / PASS**, worst seed 0.9518.
⭐ **The dSprites number sits ABOVE Shapes3D's Montero-configuration `0.9414 ± 0.0179` on the same metric** — the
two datasets agree in kind, and the second one is not the harder one.

⛔ **Downstream reconciliation list: YES, 5 items — §9. Two need a Hub owner at this review:** (1) ⚠ **on dSprites
the probe is NOT saturated** — it reads `orientation` on **real** images at **0.15**, so the bridge's *aggregate*
ceiling is structurally uninterpretable on this dataset and R15's "the probe reads real images at 1.0000" must never
be carried across; (2) **the dSprites bridge path had never executed** — the first call raised
`ValueError: too many values to unpack`, and a second, silent defect (`/255` on a `{0,1}` uint8 mask) would have
produced an entirely black dataset that trains and evaluates without a single error.

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *the dataset.* ⛔ No arm, no store, no well, no CLU number, no kill-rule verdict appears below. Nothing
  here is evidence for the primitive.
- **Laundering control:** the **matched-seen** corner (R16's permanent companion) — the held-out cell's whole
  conjunction with the `shape` clause relaxed to a **seen** shape (square / heart), minus every fitted and every
  held-out row. It is the negative set in **every** gate number below, not a side table.
- **Falsifies:** the gate FAILs ⇒ everything downstream on dSprites stops. **It did not fire.**
- **Does NOT falsify:** a dSprites number below Shapes3D's 0.9414 (different dataset, cell 57× larger in fraction);
  a low **aggregate** ceiling (pre-registered as P8 — the `orientation` head is structurally capped by the shapes'
  own 90°/180° symmetry).

---

## 0. Pre-registration (acceptance item 1)

`.claude/outputs/gen-bridge-dsprites/PREREG.md`, mtime **2026-09-13 22:04:02**. The production harness
`.claude/scratch/gen-bridge-dsprites/run_prod.py` did not exist yet and the first production fit started **22:07:21**
(log `prod_s0.log`). Verified by mtime and by process start, not by claim. ⛔ No prediction was edited afterwards; §4
of the prereg ("Amendments") is still empty.

⚠ **Declared in the prereg itself, not smoothed over:** two cheap audit measurements were taken **before** it was
written — the realised split counts and the C-set sha — because the task requires them before anything is fitted and
because a wrong split voids everything downstream. They are recorded in prereg §3 as anchors, ⛔ not as predictions.
No model had been fitted and no discrimination number computed at that point.

### Outcome table — **7 of 11 scoreable predictions held, 4 falsified; 3 of the 4 in the same direction (better)**

| # | pre-registered | predicted (80 % interval) | measured | verdict |
|---|---|---|---|---|
| **P1** | **balacc held-out (THE GATE)** | 0.85 [0.62, 0.96] | **0.9616 ± 0.0099** | ❌ **falsified, above** |
| **P2** | balacc seen normaliser | 0.94 [0.84, 0.99] | **0.9553 ± 0.0089** | ✅ in range |
| **P3** | ratio held/seen | 0.90 [0.70, 1.03] | **1.0066 ± 0.0014** | ✅ in range, at the top |
| **P4** | ⭐ **THE VERDICT** (stated probability 0.60) | **PASS** | **PASS, 3/3** | ✅ |
| **P5** | AUROC held-out | 0.93 [0.78, 0.995] | **0.9991 ± 0.0008** | ❌ falsified, above |
| P6 | aggregate ceiling ratio B/A | 0.95 [0.82, 1.02] | **1.0447 ± 0.0107** | ❌ falsified, above |
| **P7** | balacc on **REAL** images (instrument) | 0.99 [0.95, 1.000] | **0.9866 ± 0.0008** | ✅ in range |
| **P8** | ⚠ probe `orientation` on REAL images | **< 0.35** [0.05, 0.60] | **A 0.1475 ± 0.0079 · B 0.0625 ± 0.0026** | ✅ **confirmed — and it is the finding in §5** |
| **P9** | decode MSE held-out ÷ matched-seen | 1.00 [0.85, 1.25] | **0.333 ± 0.001** | ❌ **falsified, below — see §5.2** |
| P10 | realised split 737,280 / 6,292 / 730,988 | (anchor, measured pre-filing) | **exact** | ✅ |
| P11 | Shapes3D bit-identity | unchanged | **all_pass true**, max_abs_diff 0.0 | ✅ |
| P12 | matched-seen pool = 12,584 | (anchor, measured pre-filing) | **exact** | ✅ |

⭐ **P1/P5/P6 all came in above their intervals for the same reason, and I got the reason wrong in advance.** I
predicted a penalty from "dSprites carries no texture or colour cue, so shape identity is entirely silhouette". It is
entirely silhouette — and that turns out to be *easier*, not harder, for a uniform-pixel-MSE renderer at this
capacity: the held-out cell's four admitted scales are the **largest four of six**, so every held-out sprite is big.
⛔ P9's falsification is the direct evidence and it goes in the negatives registry (§9.5).

---

## 1. ⛔ FIRST — the traps, because the whole report is void without them (acceptance item 2)

**Measured against the real npz before anything was fitted.** ⛔ Assumed nothing.

| | R11 / reference | measured | |
|---|---|---|---|
| n_total / **n_test** / n_train | 737,280 / **6,292** / 730,988 | **737,280 / 6,292 / 730,988** | ✅ exact |
| realised values in the cell | — | `shape` **1** · `scale` **4** (0.7/0.8/0.9/1.0) · `orientation` **13** · `posX` **11** · `posY` **11** | ✅ 1·4·13·11·11 = 6,292 |
| realised rotation band | [120°, 240°] | **120.000° – 230.769°** (T2 holds: `a120`=240°, `a240`=120°) | ✅ |
| `c_invalid.json` sha256 | `10a39e71…8666` | **`10a39e713864c9f0c28f740caad49394cbda6f34f3d1d29bbc61dab9fc288666`** | ✅ exact |
| matched-seen corner | — | **18,876** total = **6,292 per shape** × 3 | measured |
| matched-seen negative pool | — | **12,584** (6,292 squares + 6,292 hearts) | measured |
| seen-normaliser pool | — | **22,748** | measured |

⛔ **I did not touch the angle constants and I did not touch `montero_test_mask`.** The new corner mask is a separate
function, and the test suite asserts **on the real 737,280-row file** that
`montero_test_mask(v, c) == (c[:, shp] == 1) & montero_corner_mask(v)` **exactly** — so the verbatim upstream filter
stays verbatim and the new mask provably cannot drift from it.

### 1.1 R3.3 no-leak, per seed, **and shown able to fire** (acceptance item 3)
`.claude/outputs/gen-bridge-dsprites/no_leak_positive_control.json`

| seed | fitted rows checked | held-out excluded | **leaked** | positive control |
|---|---|---|---|---|
| 0 / 1 / 2 | 8,192 each | **6,292 / 6,292** each | **0** each | ⭐ **fires**, naming the injected row (341685 / 341686 / 341687) |

The assertion is the shipped `assert_no_holdout_leak`, which re-derives every used row's factor values through the
grid bijection rather than re-reading the mask. It runs **twice per seed inside the production module** (encoder stage
and decoder stage) and a **third time** in the discrimination stage; all three logged `6292/6292 excluded, 0 leaked`.
The matched-seen negatives and the seen-normaliser positives are additionally asserted disjoint from both the held-out
split and the fitted rows, and their pools shrink by exactly the fitted overlap (12,584 → 12,444 / 12,442 / 12,452).

---

## 2. ⭐⭐ THE GATE (acceptance item 5) — `shape`, in-corner, balanced accuracy

**Definition, stated before the numbers** (R20 unchanged, R23.1's statistic):
**positives** = the 6,292 held-out ellipses; **negatives** = matched-seen, the cell's whole conjunction with `shape`
relaxed to a seen shape, minus every fitted and every held-out row; **decision** = argmax of the frozen probe's
`shape` head on the **decoded** image equals ellipse; **balanced accuracy** = ½(TPR + TNR); **AUROC** over
`p(ellipse)`. **Seen normaliser** = the cell's *position* corner with its own shape but **outside** the cell (scale
and orientation relaxed), scored against the **same** negatives — `gen-bridge-wide`'s definition transposed, declared
in the prereg before measurement.

| quantity | seed 0 | seed 1 | seed 2 | **mean ± std** |
|---|---|---|---|---|
| ⭐ **balanced accuracy, held-out (THE GATE)** | 0.9518 | 0.9751 | 0.9579 | **0.9616 ± 0.0099** |
| **balanced accuracy, matched-seen normaliser** | 0.9474 | 0.9678 | 0.9508 | **0.9553 ± 0.0089** |
| ⭐ **ratio (held-out / seen)** | 1.0046 | 1.0076 | 1.0075 | **1.0066 ± 0.0014** |
| AUROC held-out (⛔ companion, never the gate) | 0.9997 | 0.9980 | 0.9995 | 0.9991 ± 0.0008 |
| AUROC seen (companion) | 0.9966 | 0.9956 | 0.9957 | 0.9960 ± 0.0005 |
| TPR ( = ⛔ the **detection** rate) | 1.0000 | 0.9822 | 0.9995 | 0.9939 ± 0.0083 |
| TNR | 0.9036 | 0.9680 | 0.9163 | 0.9293 ± 0.0279 |
| **REAL images, same metric (the instrument)** | 0.9870 | 0.9854 | 0.9873 | **0.9866 ± 0.0008** |
| **VERDICT** (≥0.75 abs AND ≥0.80 ratio) | **PASS** | **PASS** | **PASS** | ⭐ **PASS** |

n per seed: positives **6,292**, negatives **4,096**, seen **4,096** (caps; realised pools 6,292 / ~12,45k / ~22,5k).

### 2.1 ⭐ Why R20's correction is load-bearing *again*, on this dataset, at this level
On seed 0 the **detection rate is exactly 1.0000** — the probe calls every one of the 6,292 held-out renders an
ellipse. ⛔ **On a one-class cell that number is worth nothing**, and the plain per-factor `ceiling_B['shape']` in the
bridge's own manifest agrees with it (**1.0000 / 0.9822 / 0.9995**, §4). The entire information content of the gate
sits in **TNR = 0.9293 ± 0.0279** — whether the renderer's *square* and *heart* in the same corner are still read as
square and heart. ⭐ **Had the gate been read as detection, dSprites would have scored 0.994 and told us nothing.**

### 2.2 Comparison to Shapes3D — the two datasets agree in kind

| configuration | cell | n_neg | **balacc held-out** | AUROC | TPR | TNR |
|---|---|---|---|---|---|---|
| Shapes3D, Montero, V1 decoder (R22/R23.5) | 1 (cylinder) | ≈198 | **0.9414 ± 0.0179** | 0.9886 | 0.9398 | 0.9429 |
| Shapes3D, wide, pooled over 13 (R23.3) | 13, 4 shapes | 141–216 | 0.9798 ± 0.0088 | 0.9994 | 0.9694 | 0.9902 |
| ⭐ **dSprites, Montero, V1 recipe (this spoke)** | 1 (ellipse) | **4,096** | **0.9616 ± 0.0099** | **0.9991** | 0.9939 | 0.9293 |

⚠ **Three caveats on this comparison, stated rather than left for a reader to find.** (1) dSprites' negative set is
**21× larger** (4,096 vs ≈198), so its TNR is a far better-resolved number and its seed spread is correspondingly
tight. (2) The three quantities are *not* the same experiment — different encoder, different decoder, different
probe, different factor grid; they are the same **metric**. (3) ⛔ The dSprites row is **not** evidence that dSprites
is "easier": its held-out cell is 0.85 % of the grid against Shapes3D's 0.015 %, which cuts the other way.

---

## 3. ⛔ The frozen provenance, per seed (flag-provenance rule)

**Commit `bead62c`** (branch tip after rebase onto `main @ 22f9dc5`). ⛔ Every number in §2/§4/§5 comes from these
artifacts and no other.

| seed | encoder hash | **decoder hash** | probe hash | decoder params | recipe | cache | features sha256 | wall |
|---|---|---|---|---|---|---|---|---|
| 0 | `e15cfe0266f8f330…` | **`ccc6635b460cb56d…`** | `7499bd52c5f6e956…` | 1,515,073 | **`v1_capacity_schedule`**, 1800 steps | `used` / 15,508 | `55abddfe3becc1fa…` | 5,616 s |
| 1 | `3eefabfd20e33cc2…` | **`194d37be319b1512…`** | `858af570db673b5a…` | 1,515,073 | `v1_capacity_schedule`, 1800 | `used` / 15,508 | `7e80b8cabd1c2c3b…` | 5,017 s |
| 2 | `29d7e2a983b35ed6…` | **`70fb2f430f8bcb18…`** | `552aacf2536a5917…` | 1,515,073 | `v1_capacity_schedule`, 1800 | `used` / 15,508 | `b65f4290bb4a6d7e…` | 4,953 s |

**Full flag set in effect (non-default marked ⭐):**
`dataset=dsprites` ⭐ · `pretext=recon` · `condition=recomb2element` · **`dec_recipe=v1_capacity_schedule`** ⭐
(`dec_width_mult=2.0`, `dec_blocks_per_stage=2`, `dec_steps=1800`, `dec_lr_schedule=cosine`, `dec_warmup_frac=0.05`,
`dec_loss=mse` — ⛔ **not** `object_weighted`, which LOST at 0.537) · **`cache_scope="used"`** ⭐ (declared, §3.1) ·
`cache_block=23040` ⭐ (read geometry only) · `n_train=8192` · `n_eval_a=1024` · `enc_steps=1500` · `probe_steps=1500` ·
`enc_channels=[32,64,128,128]`, `D=2048` · `cache_dtype=float16` · `verify_cache_checksum=True` ·
`max_positives=6292 / max_negatives=4096 / max_seen=4096 / min_negatives=20` · `disc_floor=0.8`,
`disc_abs_floor=0.75` · `seeds=[0,1,2]` · JAX **0.9.0** (main venv, ⛔ no `uv sync` in the worktree) ·
`CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`.
⛔ No CHLU dynamics flags (`langevin_noise`, `lyapunov`, `kinetic_mode`, γ, dt) are in play — this spoke touches no
CHLU unit at all.

### 3.1 ⭐ The decoder recipe is **asserted**, not assumed — and the assertion is shown able to fire
`bridge_manifest.json → training.dec_recipe.name` is compared to `expected_decoder_recipe` on **every** seed before
a single number is computed; a mismatch raises `⛔ decoder recipe 'baseline' != expected 'v1_capacity_schedule':
this run is VOID`. **That assertion actually fired during development**, on the `--quick` smoke (whose profile
deliberately uses `baseline` so a smoke run does not inherit an 1,800-step schedule) — the raise is in
`quick2.log`, and under `--quick` it is now a loud warning instead. ⛔ In a non-quick run it still raises.

**Declared deviation: `cache_scope="used"`, not `"full"`.** A full-scope dSprites cache is 737,280 × 2048 × float16 =
**3.0 GB per seed**. `"used"` covers train + eval_A + all 6,292 held-out rows (15,508). The matched-seen negatives and
the seen-normaliser positives are in **no** `used` cache — exactly the constraint `exp_gen_store` documents for
Shapes3D seeds 1–2 — so their features come from the bridge's **own frozen encoder** in a side buffer, quantised to
the cache's own dtype. ⭐ **The stage proves the side buffer byte-equal to the cache on 32 rows the cache *does* hold
before trusting it on rows it does not: `side-buffer == cache: True` on all three seeds.**

---

## 4. The bridge's own aggregate ceiling (context — ⛔ **NOT** the gate)

| quantity | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| ceiling_A aggregate (n=1,024) | 0.5154 | 0.4922 | 0.4932 | 0.5003 ± 0.0107 |
| ceiling_B aggregate (n=6,292) | 0.5307 | 0.5182 | 0.5187 | 0.5225 ± 0.0058 |
| ratio B/A, aggregate → `PROCEED` | 1.0296 | 1.0528 | 1.0518 | **1.0447 ± 0.0107** |
| `shape`  A / B | 0.846 / 1.000 | 0.798 / 0.982 | 0.817 / 1.000 | 0.820 / **0.994** ⛔ *degenerate, §2.1* |
| `scale`  A / B | 0.601 / 0.573 | 0.568 / 0.544 | 0.565 / 0.505 | 0.578 / 0.541 |
| `posX`   A / B | 0.478 / 0.410 | 0.475 / 0.429 | 0.465 / 0.447 | 0.472 / 0.429 |
| `posY`   A / B | 0.488 / 0.609 | 0.473 / 0.577 | 0.477 / 0.582 | 0.479 / 0.589 |
| ⚠ `orientation` A / B | 0.165 / 0.062 | 0.148 / 0.060 | 0.143 / 0.061 | **0.152 / 0.061** |
| decode MSE A / B | 0.00097 / 0.00052 | 0.00098 / 0.00051 | 0.00097 / 0.00051 | 0.00097 / 0.00051 |

⛔ **Do not quote the aggregate as a dSprites ceiling.** It averages five heads of which three are position/orientation
heads over 32/32/40 classes, and one of those is structurally unanswerable (§5.1). The gate reads `shape` (§2).

### 4.1 The 3-way `shape` companion (⛔ a **different** quantity, in its own row)
| set | classes present | macro recall, seed 0 / 1 / 2 |
|---|---|---|
| held-out positives (n=6,292) | [1] | 1.0000 / 0.9822 / 0.9995 |
| matched-seen negatives (n=4,096) | [0, 2] | **0.6725 / 0.6648 / 0.7076** |
| seen normaliser (n=4,096) | [1] | 0.9912 / 0.9675 / 0.9854 |
⭐ The residual is **square ↔ heart**, not anything ↔ ellipse: the negatives' 4-way recall is ~0.68 while their TNR
against "ellipse" is 0.93. ⛔ It is never divided into the binary number.

---

## 5. ⚠⚠ THE FINDING THE HUB NEEDS: on dSprites the probe is **NOT saturated**

R15 established, on Shapes3D, that *"the probe reads REAL images at 1.0000 on both A and B ⇒ the instrument is sound
and saturated; nobody may later read probe accuracy as a difficulty axis."* ⛔ **That sentence does not carry to
dSprites and must not be quoted as if it did.**

| probe on **REAL** images | A (seen) | B (held-out) |
|---|---|---|
| aggregate | **0.5467 ± 0.0070** | **0.5234 ± 0.0067** |
| `shape` | 0.9421 ± 0.0112 | 0.9912 ± 0.0113 |
| ⚠ `orientation` | **0.1475 ± 0.0079** | **0.0625 ± 0.0026** |

⭐ **And it is not a defect — it is a property of the dataset, pre-registered as P8 before it was measured.** dSprites'
`orientation` factor has **40 classes**, but the square has **90° symmetry** and the ellipse **180°**, so orientation
classes are *provably* not identifiable from the image: a square at 10° and at 100° are the same 64×64 bitmap. The
ceiling on a 40-way orientation head is therefore ≈ 1/4 for squares and ≈ 1/2 for ellipses **on real images**, and
0.0625 on an all-ellipse held-out set is close to the reachable bound, not a rendering failure.

⇒ **Consequences (both belong in the ledger):**
1. ⛔ **On dSprites, the aggregate ceiling is uninterpretable** and the compositional-term read is not a refinement,
   it is the only sound read. R16's option-1 ruling turns out to be *more* necessary on dataset 2 than on dataset 1.
2. ⭐ **The gate's own instrument row settles it independently:** the same in-corner metric on **real** images is
   **0.9866 ± 0.0008** (AUROC 0.9997) — i.e. the probe reads *`shape`* essentially perfectly, and the 0.9616 decoded
   number is **97.5 % of its own instrument**, not of a hypothetical perfect one.

### 5.2 ⭐ P9 falsified, and it inverts a Shapes3D intuition
`gen-decoder-lift` measured that the Shapes3D held-out corner is **not harder to reconstruct**, only harder to render
*identifiably* (MSE 0.0027 held-out vs 0.0031 seen). On dSprites the held-out corner is **3× easier to reconstruct**:

| decode MSE (discrimination stage) | positives (held-out ellipses) | matched-seen negatives | seen normaliser |
|---|---|---|---|
| seed 0 / 1 / 2 | 0.000518 / 0.000512 / 0.000512 | **0.001554 / 0.001560 / 0.001552** | 0.000452 / 0.000437 / 0.000417 |

⭐ **The ellipse is the cheapest silhouette in the dataset for a uniform-pixel-MSE renderer** — smooth, convex, no
corners — and the held-out cell admits only the four **largest** scales. The negatives (squares and hearts, same
corner) cost 3× the MSE. ⛔ This is why P1 came in high and why I predicted it low: I reasoned "silhouette-only ⇒
harder" and the actual mechanism is "the held-out silhouette is the easy one". The honest statement is that **this
cell's PASS is partly a property of which shape Montero held out**, and a dSprites cell holding out *hearts* is a
different and probably harder measurement. ⛔ Not run; filed as a follow-up (§10).

---

## 6. ⛔ The Shapes3D path is bit-identical (acceptance item 6)

`.claude/outputs/gen-bridge-dsprites/shapes3d_bitidentity.json` — **`all_pass: true`**. Proved against the **shipped
frozen artifacts**, end to end, not by inspection:

| check | result |
|---|---|
| `features.npy` sha256 re-verified on load | ✅ |
| **128 rows re-streamed through the modified `stream_images`, re-encoded by the shipped frozen encoder, quantised to float16** | ⭐ **byte-identical to the cache, `max_abs_diff = 0.0`** |
| streamed image range | [0.0, 1.0] (Shapes3D divisor still **255.0**) |
| `encoder.pkl` param hash | `98d326028316ae2cd6b960b26a922cd47fe8a2c9e73199fb1ba2927864a29c0d` = manifest ✅ |
| `decoder.pkl` param hash | `47c0f83196cd220d69c56a948a4733dc2cb295027889e2789f42790748ccfc43` = manifest ✅ (= `gen-decoder-lift` §3) |
| `probe.pkl` param hash | = manifest ✅ |
| default `resolve_decoder_cfg(GenBridgeConfig())` | still **`baseline`**, 1200 steps, constant LR, mse ✅ |
| `montero_test_mask` (Shapes3D **and** dSprites) | **not touched** ✅ |

Plus the mechanical side: `git diff main..HEAD` on `chlu/config.py` is **88 insertions, 0 deletions**; every dSprites
entry point is gated on `dataset == "dsprites"`; and `tests/test_gen_bridge_dsprites.py::
test_profile_leaves_the_shapes3d_defaults_untouched` asserts the Shapes3D recipe resolution is unchanged.

---

## 7. Tests (acceptance item 8) — **passed / skipped / failed, with every skip named**

⚠ Run with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen` — without it every real-data test **skips
silently**. ⛔ No full suite was started.

| command (post-rebase, commit `bead62c`) | result |
|---|---|
| `pytest tests/test_gen_bridge_dsprites.py` (new, 20 tests) | **20 passed, 0 skipped, 0 failed** (20.9 s) |
| `pytest test_gen_bridge_dsprites + test_gen_bridge + test_gen_bridge_wide + test_gen_data_splits + test_gen_decoder_lift + test_gen_wide_split + test_config` | **156 passed, 0 skipped, 0 failed** (45.3 s) |
| `ruff check` on all 5 touched files | **All checks passed** |

**Skips: none.** (⛔ Reported explicitly because the default is a silent skip: `h5py` is present in the main venv and
`CHLU_GEN_DATA_ROOT` was set, so the `needs_dsprites` / `needs_shapes3d` marks did not fire. R12's h5py-extra issue is
still open but does not bite here because the worktree reuses the main venv.)

**The load-bearing new tests** (positive controls, not just assertions that pass):
`test_T2_angle_constants_are_swapped_and_must_stay_that_way` · `test_T1_corner_mask_rejects_the_raw_six_column_arrays`
· `test_corner_mask_is_the_test_filter_with_exactly_one_clause_removed` (and its real-file twin) ·
`test_real_realised_counts_match_R11` · `test_real_images_are_binary_and_are_not_divided_by_255` ·
`test_populations_POSITIVE_CONTROL_a_fitted_negative_raises` ·
`test_the_gate_statistic_is_balanced_accuracy_not_detection` (constant "ellipse" answer ⇒ **exactly 0.5** balacc,
1.000 detection) · `test_profile_selects_the_recipe_that_PASSED_not_the_one_that_failed`.

**Test-count accounting for the merge:** `tests/test_gen_bridge_dsprites.py` adds exactly **20**; no existing test was
modified, renamed or deleted.

---

## 8. Two real defects found, both of which had never surfaced because the path had never run

1. ⭐ **`ValueError: too many values to unpack (expected 2)`** on the very first dSprites call.
   `exp_gen_bridge.load_split_and_cells` did `split, _ = mod.build_split(...)`; `shapes3d.build_split` returns
   `(split, labels)` but `dsprites.build_split` returns `(split, latent_values, latent_classes)`. ⭐ **This is the
   proof that the dSprites bridge path had never been executed end to end**, exactly as the task said. Now indexed.
2. ⭐⭐ **`/255` on a binary mask.** `stream_images` took the divisor from the dtype alone (`uint8 → 255.0`). dSprites'
   `imgs` is a **binary mask stored as uint8 with values {0, 1}** (measured: `np.unique(imgs[:100]) == [0, 1]`), so
   every dSprites image arrived with maximum pixel **0.0039**. ⛔ **Nothing would have raised.** The encoder trains,
   the cache writes, the decoder fits, the probe fits, the gate emits a verdict — on an essentially black dataset.
   Replaced by an explicit per-dataset `UINT8_SCALES` table; Shapes3D keeps 255.0 and is asserted byte-identical (§6).

⚠ **One cosmetic defect found and deliberately NOT fixed (it is on the Shapes3D path).** `run_decoder_stage` logs
`f"fitting THE shared decoder ({cfg.dec_steps} steps…)"` and writes `manifest.training.dec_steps = cfg.dec_steps` —
both read the **raw knob**, not the resolved recipe. Under `v1_capacity_schedule` the log says **"1200 steps"** while
the loop runs **1800** (visible in every production log above). `manifest.training.dec_recipe.steps` is correct, and
that is what my assertion reads. ⛔ I did not change it because it would alter what a Shapes3D run writes to its
manifest. **Filed for the Hub as §9.4.**

---

## 9. ⭐ Downstream reconciliation list (5 items — the Hub needs to own 1 and 2 at this review)

1. ⚠⚠ **R15's "the probe reads REAL images at 1.0000 ⇒ the instrument is saturated" is Shapes3D-only.** On dSprites
   the probe reads real images at **0.5467 aggregate / 0.1475 on `orientation`**. Any ledger sentence, paper sentence
   or future task that carries "the probe is saturated" across datasets is wrong. **Owner: Hub / curator.** ⭐ The
   *compositional-term* instrument is fine (0.9866 in-corner on real images) — it is the **aggregate** that dies.
2. ⭐ **"dSprites is fully wired" should be retired as a phrase.** The loader, splits, C set and A7 set were wired
   and checksummed; the **bridge** path raised on its first call and carried a silent pixel-scale defect. ⛔ Any
   future "X is already wired, just run it" should be read as "X has never been run". **Owner: Hub.**
3. **The gate now has a second dataset**: `0.9616 ± 0.0099` (dSprites/Montero/V1) belongs beside R23.5's
   `0.9414 ± 0.0179` (Shapes3D/Montero/V1) and R23.3's `0.9894 ± 0.0021` (Shapes3D/wide, 11 cells) in any table that
   quotes "the decoder ceiling". ⛔ The three are the same **metric**, not the same experiment.
4. **`dec_steps` log/manifest reads the raw knob, not the recipe** (§8). A Shapes3D-path one-liner; needs an owner
   because "a baseline run looks entirely normal in the logs" is precisely the failure mode R20 warned about.
5. **For the negatives registry — one falsified prediction that is a finding:** *"dSprites' texture-free silhouettes
   make held-out shape identity harder than Shapes3D's"* — **FALSIFIED**. The held-out silhouette (ellipse) is the
   **cheapest** one in the dataset for a uniform-MSE renderer (decode MSE **0.000512** vs **0.001554** for the
   square/heart negatives in the same corner), and the cell admits only the four largest scales. ⭐ The PASS is
   therefore partly a property of *which* shape Montero held out.

---

## 10. Open questions / follow-ups / risks

- ⚠ **The PASS is cell-specific.** Montero's dSprites cell holds out the **ellipse**, the easiest silhouette. A cell
  holding out **hearts** (corners, concavity) is a materially different measurement and is the obvious dSprites
  analogue of R17's widened split. ⛔ **Not run**, and I did not run it — it is not this spoke's gate.
- **`cache_scope="used"`, not `"full"`** (§3.1) — a full dSprites cache is 3.0 GB/seed. Any downstream dSprites spoke
  that needs arbitrary rows must use the same frozen-encoder side buffer (it is now a reusable function,
  `features_cache_or_encoder`), or budget for the full cache.
- **CSF3: NOT RUN.** CSF3 needs 2FA and is unreachable. ⛔ Also, the launcher would reject this stage as written:
  `scripts/csf3/job_gpu_gen.sh` whitelists `STAGE=encoder|decoder|store|cell|certificate` and passes `--stage
  "$STAGE"` straight through, so `discrimination` dies at the launcher's own guard. Submit line, **for the Hub to
  land after the launcher gains the axis** (⛔ `scripts/csf3/` is outside my scope and I did not touch it):
  ```
  # 1) bridge (encoder + decoder + probe + the in-corner gate, all in one)
  sbatch --export="ALL,DATASET=dsprites,PRETEXT=recon,STAGE=decoder,ARMS=clu,SEEDS=0 1 2,\
  PAYLOAD_MODULE=chlu.experiments.exp_gen_bridge,EXTRA_ARGS=--cache-scope used,\
  GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-2%3 -t 06:00:00 scripts/csf3/job_gpu_gen.sh
  # 2) the gate alone, once the launcher's STAGE whitelist admits 'discrimination'
  #    (until then it must be a PAYLOAD_MODULE run with --stage discrimination)
  ```
- ⚠ **Wall-clock is contended.** Load average ranged **9.7–16.3** on 8 cores from concurrent agents; per-seed times
  (5,616 / 5,017 / 4,953 s) are soft. Total ≈ **4.3 h** of fitting for 3 seeds.
- **No seed is NOT RUN.** All three completed and are reported in full.

---

## Git footprint

- **Branch:** `agent/experiment-engineer/gen-bridge-dsprites` (left for review; ⛔ not pushed, no PR).
- **Worktree:** `../CHLU-gen-dsprites` (fresh, as specified). Main venv reused — ⛔ **no `uv sync` in the worktree**;
  resolved **JAX 0.9.0**. ⭐ Branch ref verified from the MAIN repo (`git -C /Users/user/Desktop/CHLU log main..<branch>`
  shows all 5 commits) **before** any worktree teardown; the worktree is still in place.
- **Base:** task named `main @ be27713`; `main` advanced to **`22f9dc5`** during the run and I **rebased onto local
  `main`** (⛔ never `origin/main`, §7.21). Post-rebase commits:

| commit | subject | files |
|---|---|---|
| `15e3d2a` | stop dSprites' binary mask being divided by 255 | `chlu/core/gen_bridge.py` (+13 −1) |
| `dc2a4d3` | add the dSprites matched-seen corner masks (R16/R20) | `chlu/data/dsprites.py` (+51 −0) |
| `78a3380` | add `GenBridgeDspritesConfig` (one block, appended at EOF) | `chlu/config.py` (+88 −0) |
| `5bfa64b` | run the bridge on dSprites and gate it on the compositional term | `chlu/experiments/exp_gen_bridge.py` (+472 −3) |
| `bead62c` | tests: the dSprites traps, populations and the pinned recipe | `tests/test_gen_bridge_dsprites.py` (+345) |

**Total: 5 files, +969 / −4.**

⚠ **ONE CONFLICT, and it was resolved by the task's own rule.** `chlu/config.py` conflicted on rebase (the peer
spokes' blocks landed in `main` while I was running). ⛔ **I did not merge hunk-by-hunk.** I took **the peer's whole
file** (`git checkout main -- chlu/config.py`), re-applied my four insertions onto it by anchor, and **asserted the
result: `git diff --numstat main -- chlu/config.py` → `88  0`, i.e. 0 deletions.** No other conflict.
⛔ **Nothing under `chlu/core/gen_store.py`, `gen_compose.py`, `gen_navigate.py`, `gen_certificate.py`,
`gen_request.py`, `gen_dream.py`, `chlu/eval/`, `scripts/csf3/` or `chlu/cli/` was touched** — the in-corner metric
is the **shipped** `chlu/eval/gen_probe.py` functions, called, never edited. (⚠ Consequence: `chlu exp-gen-bridge`
gains the dSprites gate *without* a `experiment_cmd.py` edit, because the driver defaults `discrimination=True` for
`dataset=="dsprites"` and `False` for everything else.)

**Artifacts** (all under `.claude/`, 217 MB):
`.claude/outputs/gen-bridge-dsprites/PREREG.md` · `tables.md` · `shapes3d_bitidentity.json` ·
`no_leak_positive_control.json` · `runs/dsprites_recon/{discrimination.json, discrimination_gate.json, ceiling.json,
gate.json}` · `runs/dsprites_recon/seed_{0,1,2}/{discrimination.json, bridge_manifest.json, encoder_manifest.json,
encoder.pkl, decoder.pkl, probe.pkl, features.npy, feature_cache.json, ceiling_grid.png, ceiling_bars.png}` ·
scratch harness + full logs under `.claude/scratch/gen-bridge-dsprites/`.

---

## Proposed handover updates (for the Hub)

**§7 Known Issues — two new entries:**
- **7.30 [FIXED on this branch, `15e3d2a`] `stream_images` divided dSprites' binary mask by 255.** The divisor was
  inferred from the dtype alone; dSprites' `imgs` is uint8 with values `{0,1}`, so every image arrived at max pixel
  0.0039 and **nothing raised** — the encoder, cache, decoder, probe and gate all ran on an essentially black
  dataset. Now an explicit per-dataset `UINT8_SCALES`. ⛔ Shapes3D unchanged (asserted byte-identical, 128 rows,
  `max_abs_diff = 0.0`). **Standing lesson: a per-dataset pixel convention is not inferable from a dtype.**
- **7.31 [OPEN, cosmetic but dangerous — no owner] `run_decoder_stage` logs and manifests the RAW `dec_steps` knob,
  not the resolved recipe's.** Under `v1_capacity_schedule` the log reads "1200 steps" while the loop runs 1800.
  `manifest.training.dec_recipe.steps` is correct. ⛔ Left untouched because the fix changes what a **Shapes3D** run
  writes; it needs a Hub decision. This is exactly the "a baseline run looks entirely normal in the logs" hazard.

**§3 config defaults — one new block:** `CHLUConfig.gen_bridge_dsprites` (`GenBridgeDspritesConfig`, appended at EOF,
`+88 / −0`). ⭐ Its `dec_recipe = "v1_capacity_schedule"` is a **dSprites-only** override applied by
`apply_dsprites_profile`; **`GenBridgeConfig.dec_recipe` still defaults to `"baseline"` globally and was NOT
changed.** ⚠ Whether the global default should move to the winner is a Head/Hub call this spoke deliberately did not
make.

**For the rulings ledger (a proposed R-entry, Hub to word):** *"GATE 2 — the second dataset. dSprites,
`recomb2element`, `recon`, V1 recipe, 3 seeds: in-corner `shape` balanced accuracy **0.9616 ± 0.0099** held-out vs
**0.9553 ± 0.0089** matched-seen, ratio **1.0066**, AUROC **0.9991**, **PASS 3/3**. Realised split verified
737,280 / 6,292 / 730,988; R3.3 no-leak 6,292/6,292, 0 leaked, positive control fires. ⛔ Instrument caveat: on
dSprites the probe is NOT saturated (real-image aggregate 0.547, `orientation` 0.148) — the aggregate ceiling is
uninterpretable on this dataset and only the compositional term is."*

**For the negatives registry:** see §9.5 — the "texture-free silhouettes make it harder" prediction is falsified, and
the held-out ellipse is measurably the **cheapest** silhouette in the dataset (decode MSE 0.000512 vs 0.001554).
