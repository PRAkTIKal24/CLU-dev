# gen-dsprites-exponent — experiment-engineer report

**Task + acceptance criterion:** run the **same codebook `M` sweep and the same exponent fit** on **dSprites** and on
a **self-re-run Shapes3D**, and answer the pre-registered directional claim (§11b) — *is the dSprites exponent
STEEPER than Shapes3D's, non-overlapping across 3 seeds?* — **yes/no in those words**.
**Status: done.** 6/6 (dataset × seed) cells complete, 3 seeds, 7 `M` points each, nothing NOT-RUN.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 3 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⭐⭐ **§11b's PRIMARY DIRECTIONAL CLAIM IS ANSWERED AND THE ANSWER IS `NO` — by 0.0011.** dSprites is steeper
>    **on the mean** (−0.2803 vs −0.2500) and on **8 of 9** cross-seed pairs, but the pre-registered rule is
>    *non-overlapping*, and one pair inverts: dSprites seed 1 **−0.2605** against Shapes3D seed 0 **−0.2616**.
>    ⛔ By §11b's own words **the factor-manifold account is REFUTED** and −0.26's universality needs a different
>    explanation. ⭐ **Owner: Hub — §11b says "enters the paper only if it holds"; it did not hold, so the decision
>    is whether the *mean-level* difference (well-level `t = −4.42`, `p = 1.5e−4`, `PR`-controlled) is re-scoped as
>    a NEW, pre-registered question or dropped.** ⛔ It may not be quietly re-labelled a pass.
> 2. ⭐⭐ **`gen-codebook-allocation` §5d's "the exponent is essentially CONSTANT across `PR`" is WRONG and this
>    spoke corrects it.** With 7 `M` points instead of 3, the exponent depends on `log PR` **within each dataset**
>    (slope `+0.0702 ± 0.0136` Shapes3D, `+0.0466 ± 0.0092` dSprites, `r = 0.82`, `p = 2.1e−4`). ⛔ §5d's *headline*
>    survives — the law's `−2/PR` still overpredicts the **rate** by ~3× (measured `−0.287 → −0.200` over
>    `PR = 3 → 13`, the law `−0.667 → −0.154`) — but "constant" is not the right word and is now on the record twice.
>    **Owner: Hub — a correction note wherever §5d's constancy is quoted.**
> 3. ⚠ **`−0.2631 ± 0.0555` (§5d) and `−0.2500 ± 0.0094` (here) are the SAME quantity measured two ways**, and the
>    **seed spread falls 6×** with a proper sweep. ⛔ Any downstream text that carries `± 0.0555` as *the* precision
>    of the Shapes3D exponent is quoting a sweep-geometry artefact. **Owner: Hub.**

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *the dataset*, as a test of **why** the exponent is what it is. ⛔ No arm, no CLU claim, no verdict on
  admission / lifetimes / isolation / compute-adaptive reads. Nothing below is evidence for the primitive.
- **Laundering control:** ⭐ **the identical sweep and the identical fit on Shapes3D, re-run by me** — same module,
  same `M` grid, same `K`, same rows, same store recipe, same fit function. ⛔ `−0.2631` was **never** used as the
  comparator; it is carried in the config as a labelled historical reference and appears in the summary only under
  the key `reference_exponent_shapes3d_HISTORICAL` with `"role": "⛔ NEVER the comparator"`.
  ⭐ **The control landed bit-tight:** my Shapes3D store at `M = 64` reproduces `gen-codebook-allocation`'s published
  uniform arm at **`ε = 0.2783 ± 0.0064` vs its `0.2783 ± 0.0064`** — four decimals *and* the same seed spread —
  and my per-seed partition reproduces `gen-commitment-collapse`'s `labels_vec` **bit-identically on 3/3 seeds**.
- **Falsifies:** the factor-manifold account. ⛔ **It FIRED.**
- **Does NOT falsify:** anything about CLU, the store's value, or any rival comparison. ⛔ A dSprites number below or
  above Shapes3D's is a statement about **two datasets' encoders and partitions**, not about the primitive.

## ⭐⭐ THE RESULT — in the pre-registered words

⭐ **Is the dSprites exponent STEEPER than Shapes3D's, non-overlapping across 3 seeds? → ⛔ NO.**

| dataset | factors | ⭐ exponent (seed-level) | per seed | 95 % CI (t, 2 df) | well-level (15 cells) |
|---|---|---|---|---|---|
| ⭐ **shapes3d** (control, re-run) | 6 | **−0.2500 ± 0.0094** | −0.2616 · −0.2498 · −0.2386 | [−0.2785, −0.2214] | −0.2500 ± 0.0450 |
| ⭐ **dsprites** | 5 | **−0.2803 ± 0.0158** | −0.2991 · −0.2605 · −0.2812 | [−0.3283, −0.2323] | −0.2803 ± 0.0280 |

- **Δ mean = −0.0303** (dSprites steeper) · **means order as predicted = TRUE** · **seed sets non-overlapping =
  FALSE**. The rule is `max(dSprites) < min(Shapes3D)`: **−0.26052 vs −0.26158 — it misses by 0.00106.**
- **8 of 9** cross-seed pairs have dSprites steeper; the 9th is the one above.
- ⛔ **Per PREREG §4 the answer is NO and a NO may not be rescued by a well-level or pooled statistic.** The
  post-hoc statistics in §4 below are reported because they are the substance the Hub needs, ⛔ **not** as a rescue.

**Which companion is closer** (⛔ a distance table; ⛔ neither is "the prediction" — §11b):

| companion | value | \|measured − companion\| |
|---|---|---|
| ⭐ Shapes3D's offset carried, `−2/(5 × 1.267)` | **−0.316** | **0.0357** ← closer |
| pure factor count, `−2/5` | −0.400 | 0.1197 |

⚠ **And the companion that is closer is closer for a reason that undercuts both:** my Shapes3D control is
**−0.2500**, not −0.2631, so the "×1.267 offset" is itself re-measured at **×1.333** (`d = 8.00`), which would put
the carried-offset companion at **−0.300** — inside 0.02 of the measurement. ⛔ **That is a post-hoc recomputation
and is NOT scored**; it is flagged so the Hub does not read `−0.316` as an independent hit.

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | **`62bd6c9`** (module + config, committed 20:08:04) · **`8217067`** (tests, 20:08:08). ⭐ **Proved, not asserted:** `exp_gen_codebook_exponent.py` has mtime **20:05:42** and the production process was launched at **20:05:45** — 3 s later — and the file's mtime never moved again, so the committed tree IS the tree that produced every number. `8217067` adds tests only and touches no harness file. Branch **not rebased** (base `main` never moved: `5a3c32c` at spawn and at finish). |
| branch / base / worktree | `agent/experiment-engineer/gen-dsprites-exponent` off **`main @ 5a3c32c`** · worktree `../CHLU-gen-dspexp` · **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0** |
| datasets · pretext · condition | **`shapes3d`** and **`dsprites`**, both `recon` / `recomb2element` (Montero). ⛔ Both run, in one process, through one code path |
| seeds | **0, 1, 2** on both datasets; `±` is **population sd, ddof = 0**; CI uses **`t(2, 0.975) = 4.3027`**, sample sd |
| ⭐ the swept knob | **`M` (codewords per well) ∈ {4, 8, 16, 32, 64, 128, 186}** — ⛔ ONE grid, both datasets, endpoints pinned to Shapes3D's historical `M ∈ [4, 186]` and **asserted** by `assert_shared_grid` (raises otherwise) |
| `K` | **5 wells on both datasets** (Shapes3D's shipped `K`). ⛔ `K` partitions the 2,048 encoder channels and does **not** track the factor count |
| partition | ⭐ **RE-DERIVED per seed on BOTH datasets** by `channel_correlation` + `grouping_labels(..., "correlation", ...)`, `n_corr = 16,384`, `grouping_method="auto"`, `min_well_frac = 0.10`. ⛔ Never loaded on one dataset and derived on the other. **Verified**: reproduces `gen-commitment-collapse`'s `labels_vec` **bit-identically, 3/3 Shapes3D seeds** (dSprites has no such cache — that is why it is derived) |
| encoder / feature cache | ⛔ **LOADED, frozen, hash-asserted** against each bridge manifest. Shapes3D = `gen-feature-bridge` (`98d326028316…`, `fc05e2c5f396…`, `154d49861064…`), dSprites = `gen-bridge-dsprites` (`e15cfe0266f8…`, `3eefabfd20e3…`, `29d7e2a983b3…`). Cache scope: Shapes3D seed 0 `full`/480,000, seeds 1–2 + all dSprites `used`/15,508 → missing rows re-encoded through the **same frozen encoder**, with the re-encode check **max \|Δ\| = 0.0 on 64 cached rows, 6/6 cells** |
| ⛔ decoder / probe | **NOT LOADED AT ALL.** No image is rendered and no probe read is taken, so `GenBridgeConfig.dec_recipe`'s default `"baseline"` (the recipe that FAILED the gate) **cannot reach any number here**. ⛔ No aggregate ceiling is quoted for dSprites (its probe is not saturated — `gen-bridge-dsprites` §5) |
| rows | `store_train` **32,768** (`select_store_rows`, `default_rng(seed·7919 + 20260910)`, `eval_a` excluded) · `SPREAD` **4,096** = `take[12288:16384]` of `default_rng(seed·7919 + 20260914).permutation(train_rows \ store_train)` — ⛔ `gen-codebook-allocation`'s draw **verbatim**, so the Shapes3D `ε` is measured on the very rows its published `ε` was measured on |
| store recipe (⛔ identical at every `M`, every seed, both datasets) | `train_steps=1200`, `train_batch=128`, `train_lr=3e-3`, `commitment_beta=0.25`, `tau 1.0 → 0.1`, key `PRNGKey(seed·10007 + 20260911)`, k-means++ init |
| the fitted quantity | `per_well_distortion(...)["relative_distortion"]` on **SPREAD**, then `np.polyfit(log M, log ε, 1)[0]` per `(dataset, seed, well)`; the seed's exponent = **mean over the 5 wells** |
| spectra | `well_spectra` on **16,384** rows drawn as a **seeded permutation** of `store_train` (⛔ never a prefix) |
| ⛔ NOT used | probes, decoders, `soft_read`, dreaming, coupling, whitening, allocation, the address book, the request side, CSF3, any CHLU dynamics flag (`langevin_noise`, `lyapunov`, `kinetic_mode`, γ, dt — this spoke touches no CHLU unit) |
| wall clock | Shapes3D **5,200 / 6,872 / 6,878 s**; dSprites **7,482 / 8,568 / 3,304 s**. CPU only, 16 GB. ⚠ One crash + resume (§7) |

**Pre-registration:** `.claude/outputs/gen-dsprites-exponent/PREREG.md`, mtime **2026-09-14 19:59:43**, ⛔ never
edited. `chlu/experiments/exp_gen_codebook_exponent.py` did not exist at that moment (first commit 20:xx, first
production artefact `exponent_shapes3d_s0.json` **21:32**). `POSTHOC_1.md` (the `PR` confound) is a **NEW FILE**,
mtime **20:45:19**, filed while the run was still on its **first** seed — ⛔ **no exponent existed on disk for any
dataset**, and `runs/` was empty. ⇒ every scored number postdates its declaration on disk.

## 1. ⛔ The dSprites traps, measured before anything was fitted (acceptance 2)

| | required | measured, through the harness's own loader, every seed |
|---|---|---|
| realised split total / **test** / train | 737,280 / **6,292** / 730,988 | **737,280 / 6,292 / 730,988** ✅ 3/3 |
| colour column (T1) | dropped at load; 5 factor columns | `shape, scale, orientation, posX, posY`, cardinalities **(3, 6, 40, 32, 32)** ✅ |
| angle constants (T2) | `a120 = 4π/3`, `a240 = 2π/3` — **swapped, and must stay so** | asserted in `tests/test_gen_codebook_exponent.py::test_the_angle_constants_are_STILL_swapped` ✅ ⛔ not touched |
| `montero_test_mask` | untouched | ⛔ **not touched** (`git diff` on `chlu/data/**` is empty) |
| ⚠ the paper's prose vs the code | the CODE wins | the shipped loader is used verbatim; nothing re-derived |

⭐ **`n_values_realised` on the SPREAD rows, 3/3 seeds, both datasets — no degenerate factor anywhere:**
dSprites `shape` 3/3 · `scale` 6/6 · `orientation` 40/40 · `posX` 32/32 · `posY` 32/32.
Shapes3D `floor_hue` 10/10 · `wall_hue` 10/10 · `object_hue` 10/10 · `scale` 8/8 · `shape` 4/4 · `orientation` 15/15.
⛔ The dSprites **probe** is not saturated — but **no probe is read here**, so R15's Shapes3D saturation claim is
neither used nor needed.

## 2. ⭐ The sweep (acceptance 3 + 4) — ⛔ one grid, both datasets

Variance-weighted `ε`, mean ± sd over 3 seeds:

| `M` | shapes3d | dsprites |
|---|---|---|
| 4 | 0.5500 ± 0.0072 | 0.6479 ± 0.0162 |
| 8 | 0.4465 ± 0.0065 | 0.5311 ± 0.0072 |
| 16 | 0.3778 ± 0.0076 | 0.4282 ± 0.0207 |
| **64** | ⭐ **0.2783 ± 0.0064** | 0.2990 ± 0.0092 |
| 128 | 0.2389 ± 0.0052 | 0.2538 ± 0.0091 |
| 186 | 0.2192 ± 0.0054 | 0.2335 ± 0.0095 |

⭐⭐ **The `M = 64` Shapes3D row IS `gen-codebook-allocation`'s uniform arm, re-fitted here from scratch, and it
reproduces `0.2783 ± 0.0064` exactly.** ⛔ That is the laundering control the task demanded, and it is a
measurement, not a claim.

**Fit quality:** all **30** per-well power-law fits have `R² ≥ 0.9446` (median ≈ 0.997), and `ε(M)` is **monotone
decreasing in 30/30 cells**. ⛔ The exponent is not being read off a noisy 2-point contrast, which is exactly the
limitation `gen-codebook-allocation` §11-L1 declared about `−0.2631`.

## 3. ⚠ Where the two datasets differ, and where they do not

- **The level differs and the shape does not.** dSprites' `ε` is uniformly **~7–18 % higher** at every `M`; both
  curves are clean power laws over the same range.
- **`PR` distributions overlap but are not matched** (dSprites 3.22–16.06, Shapes3D 3.12–13.64 over 15 cells each);
  §4 controls for this.
- ⚠ **Codebook occupancy is not 100 % at large `M`**: at `M = 186`, used/allocated is **885 / 921 / 926 of 930**
  (dSprites) and **929 / 930 / 930** (Shapes3D). ⛔ Declared, not hidden: dSprites leaves up to **4.8 %** of its
  codewords unused at the top of the grid, which if anything **flattens** its measured exponent — i.e. it works
  *against* the direction that was predicted, and the prediction still came out steeper on the mean.

## 4. ⚠ POST-HOC (`POSTHOC_1.md`, filed before the data existed) — the `PR` confound

| quantity | value |
|---|---|
| within-Shapes3D slope of exponent on `log PR` | **+0.0702 ± 0.0136** (`t = 5.17`, `r = 0.820`) |
| within-dSprites slope of exponent on `log PR` | **+0.0466 ± 0.0092** (`t = 5.08`, `r = 0.815`) |
| pooled `a ~ β0 + β1 log PR + β2·1[dsprites]`, 30 cells | `β1 = +0.0592 ± 0.0085`, ⭐ **`β2 = −0.0383 ± 0.0087`** (`t = −4.42`, `dof = 27`, **`p = 1.5e−4`**), `R² = 0.695` |
| `β2` ÷ the raw well-level difference (−0.0303) | **1.26** ⇒ ⭐ **the dataset effect is NOT a `PR` artefact** — controlling for `PR` makes it *larger* |
| `PR`-matched subset (`PR ∈ [3.22, 13.63]`, 14 cells each) | dSprites **−0.2828 ± 0.0273** vs Shapes3D **−0.2467 ± 0.0448** |
| Welch `t` on the **3 seed means** (⛔ the pre-registered unit) | **`t = −2.33`, `df = 3.26`, `p = 0.095`** |

⭐⭐ **The honest summary of the tension: the dataset difference is real and sizeable at the WELL level
(`p = 1.5e−4`) and does not reach the pre-registered bar at the SEED level (`p = 0.095`, and the non-overlap test
fails by 0.001).** ⛔ Three seeds is the entire reason; the effect is ~2 × the seed spread, which is precisely the
regime where 3 seeds cannot decide. ⛔ **The pre-registered answer stands at `NO`.**

⭐ **And the `PR` slope is itself the second finding (reconciliation item 2):** with a 7-point sweep the exponent is
**not** constant in `PR` — it runs from ≈ **−0.287** at `PR = 3` to ≈ **−0.200** at `PR = 13` — but that is a
**1.4×** range against the law's `−2/PR` **4.3×** range. ⛔ `ε ≈ M^(−2/PR)` still fails as a **rate**; it now fails
by a factor ≈ 3, not by "the exponent does not move at all".

## 5. Pre-registered predictions vs measurement (verbatim ids from `PREREG.md` §5)

| id | predicted | 80 % interval | measured | verdict |
|---|---|---|---|---|
| **E1** | ⭐ my Shapes3D exponent | −0.26 [−0.36, −0.18] | **−0.2500** | ⭐ HELD |
| **E2** | ⭐ my dSprites exponent | −0.33 [−0.48, −0.22] | **−0.2803** | ⭐ HELD (interval), point estimate too steep |
| **E3** | ⭐⭐ **the PRIMARY directional answer** (p = 0.55) | **YES** | ⛔ **NO** | ⛔⛔ **FALSIFIED — and it is the result** |
| **E4** | closer companion | `−0.316` | **`−0.316`** (0.0357 vs 0.1197) | ⭐ HELD (⚠ see the caveat above) |
| **E5** | Shapes3D seed-level sd | 0.02 [0.002, 0.08] | **0.0094** | ⭐ HELD |
| **E6** | dSprites seed-level sd | 0.03 [0.002, 0.10] | **0.0158** | ⭐ HELD |
| **E7** | fraction of monotone cells | 0.95 [0.70, 1.00] | **1.000 (30/30)** | ⭐ HELD |
| **E8** | Shapes3D `ε(M=64)`, seed 0 | 0.28 [0.20, 0.38] | **0.2784** | ⭐⭐ HELD — 2 dp |
| **E9** | dSprites `ε(M=64)`, seed 0 | 0.30 [0.15, 0.50] | **0.2966** | ⭐⭐ HELD — 2 dp |
| **E10** | dSprites realised split | exact | **737,280 / 6,292 / 730,988** | ⭐ HELD — exactly |
| **E11** | derived partition == shipped `labels_vec` | YES 3/3 | **True, True, True** | ⭐⭐ HELD |
| **E12** | dSprites `PR` range | 3 … 14 [1.5, 40] | **3.22 … 16.06** | ⭐ HELD (interval); point range exceeded at the top |

**11 held, 1 falsified — and the falsified one is `E3`, the only one that matters.** ⛔ The ten that held are
controls and arithmetic (the split, the partition, the two `ε(M=64)` levels to two decimals), which is what makes
the single failure credible rather than a harness artefact.

## 6. What would have proved this spoke wrong, and did not

| the task's named failure mode | status |
|---|---|
| ⛔ *"you compared your dSprites number to a TRANSCRIBED Shapes3D −0.2631"* | ⭐ **Did not happen.** The comparator is my own re-run; `−0.2631` appears only as a labelled historical reference, in a field whose `role` string says "NEVER the comparator", and is **never** read by `compare_datasets`. A test pins this (`test_the_historical_number_is_labelled_NEVER_the_comparator`). |
| ⛔ *"your dSprites sweep covers a different `M` range"* | ⭐ **Structurally impossible here:** one `m_grid` object is used for both datasets and `assert_shared_grid` **raises** unless its endpoints are exactly `[4, 186]`. The gate is shown raising in 3 parametrised tests, and it is **not** disabled by `--quick`. |
| ⛔ dSprites trap 1 (colour column) | 5-column loader used; 6,292 test rows realised — an empty test set would have been impossible to miss |
| ⛔ dSprites trap 2 (swapped angles) | not touched; asserted still swapped |
| ⛔ `dec_recipe` defaulting to `"baseline"` | **no decoder is loaded at all** (declared in `PREREG.md` §6 before the run) |

## 7. ⚠ Two operational facts, declared

1. ⛔ **The production process died silently between 05:49 and 05:51** (last log write 05:49; an independent
   watcher observed the process gone at **05:49:52**) during dSprites seed 2's feature gathering — no traceback,
   no Python error, the process simply vanished (5 of 6 cells already on disk). ⭐ **The harness is resumable by
   design** (`skip_if_exists` + per-seed artefacts): it was relaunched at 05:52, logged `artefact HIT` on all five
   completed cells, recomputed **only** dSprites seed 2, and finished at 06:49. ⛔ **No completed cell was
   recomputed and no number changed.** ⚠ Matches the standing "detached jobs die / lose their last write" hazard.
2. ⚠ **`--quick` cannot emit a directional answer.** `compare_datasets` returns `"⛔ NOT ANSWERABLE"` unless both
   datasets have ≥ 3 seeds, because the smoke run *did* print a confident `YES` on one seed before that guard was
   added. ⛔ That `YES` was a plumbing artefact and appears nowhere in this report.

## How I verified

```
# ⚠ plumbing smoke, numbers DISCARDED, separate --out
python -m chlu.experiments.exp_gen_codebook_exponent --quick --out .../scratch/gen-dsprites-exponent/quick   # EXIT 0
# production, 2 datasets x 3 seeds x 7 M points
python -u -m chlu.experiments.exp_gen_codebook_exponent                                   # died at 05:51 (§7.1)
python -u -m chlu.experiments.exp_gen_codebook_exponent                                   # resumed; EXIT 0, 06:49
```
with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen` and
`CHLU_CLAUDE_ROOT=/Users/user/Desktop/CHLU/.claude` (a worktree has no `.claude/`). The per-dataset bridge roots
come from the new config block, so `$CHLU_GEN_BRIDGE_ROOT` is **not** needed and **not** set.

**Targeted tests, from the worktree, with the data root set. ⛔ No full suite.** ⭐ Reported as
**passed / skipped / failed**:

| command | result |
|---|---|
| `pytest tests/test_gen_codebook_exponent.py` (new, 28 tests) | **28 passed · 0 skipped · 0 failed** (14 s) |
| `pytest test_gen_codebook_exponent + test_gen_alloc + test_config + test_gen_store` | ⭐ **127 passed · 0 skipped · 0 failed** (2:22) |
| `pytest test_gen_store + test_gen_bridge_dsprites + test_gen_data_splits + test_gen_whiten` | ⭐ **102 passed · 0 skipped · 0 failed** (5:25) |
| `ruff check` on all 3 touched files | **All checks passed** |

**Skips: none anywhere.** (Reported explicitly because the default is a *silent* skip: `h5py` is in the main venv
and `CHLU_GEN_DATA_ROOT` was set, so the `needs_dsprites` marks did not fire.)
**One test failed on first run and it was my test, not the code:** a float round-trip (`-0.26` vs
`-0.26000000000000006`) through the planted-exponent fit; fixed with `pytest.approx`.

## Artefacts
`.claude/outputs/gen-dsprites-exponent/` — `PREREG.md` (19:59:43, ⛔ unedited) · `POSTHOC_1.md` (20:45:19, filed
before any exponent existed) · `summary.json` · `tables.md` (6 sections: the headline + companions, `ε(M)`, all 30
per-well fits, provenance/controls, `n_values_realised`, mode usage) · `posthoc_pr_confound.json` ·
`codebook_exponent.png` · `production.log` · `runs/{shapes3d,dsprites}_recon/exponent_*_s{0,1,2}.json`.
Scratch: `.claude/scratch/gen-dsprites-exponent/` — `check_partition.py` (the E11 pre-check), `tables.py`,
`posthoc.py`, `quick*/` (⚠ smoke, numbers discarded).

## Git footprint
- **Branch** `agent/experiment-engineer/gen-dsprites-exponent`, off **`main @ 5a3c32c`**; **worktree
  `../CHLU-gen-dspexp`, left parked.** ⛔ Not pushed, no PR. **Rebase: no-op** — `main` was `5a3c32c` at spawn and
  at finish, `git rebase main` reported "up to date".
- `62bd6c9` `chlu/config.py` (+139/−0) · `chlu/experiments/exp_gen_codebook_exponent.py` (new, 716)
- `8217067` `tests/test_gen_codebook_exponent.py` (new, 363)
- `git diff --stat main..HEAD` = **3 files, 1,218 insertions, 0 deletions.**
- ⭐ **`chlu/config.py` is APPEND-ONLY and has ZERO deletions**: one contiguous
  `# --- BEGIN gen-dsprites-exponent` / `# --- END gen-dsprites-exponent` block at EOF, no `CHLUConfig` field, no
  `load_config`/`save_config` site — so the "take the peer's whole file, re-apply my block by its markers" merge
  rule holds mechanically. ⛔ No conflict occurred (nobody else moved the file).
- ⛔ **No CLI hook added**, following `gen-codebook-allocation`'s precedent: `chlu/cli/experiment_cmd.py` is live
  territory for concurrent spokes. The experiment runs via `python -m`. ⚠ Adding `chlu exp-gen-codebook-exponent`
  is a separate one-line task if the Hub wants it.
- ⛔ **Nothing else touched.** `chlu/data/**`, `chlu/core/**`, `chlu/eval/**`, `chlu/cli/**`, every other experiment
  and every other test are unmodified. The main checkout (which had another agent's branch checked out with a
  clean tree) was **never edited** — all work happened in the worktree.
- Branch ref verified **from the main repo**: `git -C /Users/user/Desktop/CHLU log main..agent/experiment-engineer/
  gen-dsprites-exponent` shows both commits.

## Open questions / follow-ups / risks
1. ⭐⭐ **The decisive experiment is now cheap and obvious: more seeds.** The effect is ~2 × the seed spread and the
   pre-registered test failed by **0.001**. **5–8 seeds per dataset** would settle it either way; at the observed
   spreads, 6 seeds gives the seed-level test roughly the power the well-level test already has. ⛔ It would need a
   **new** pre-registration — ⛔ re-running §11b's rule with more seeds after seeing a 0.001 miss is exactly the
   move the pre-registration rule exists to prevent, so the Hub must scope it as a fresh question with a fresh bar.
2. ⚠ **A third dataset is worth more than more seeds of these two.** Two points cannot distinguish "factor count"
   from "this encoder / this partition / this dataset". `n_factors ∈ {5, 6}` differ by 20 %; the measured exponents
   differ by 12 %. **Cars3D (3 factors) or MPI3D (7)** would put a real lever on the axis. ⛔ Not run.
3. ⚠ **The `PR` slope (item 2) deserves its own fit.** `a = β0 + β1 log PR` is a *description*, not a model; the
   data now support asking what `ε(M, PR)` actually is, over `M ∈ [4, 186]` and `PR ∈ [3, 16]`, with 30 clean
   power-law cells in hand. That is the measurement `§0.5` has needed since `gen-codebook-allocation` follow-up 1.
4. ⚠ **Codebook occupancy falls below 100 % on dSprites at large `M`** (down to 885/930). Whether that is a
   store-training-budget artefact (1,200 steps for 930 modes) or a real capacity statement is **untested**, and it
   biases dSprites' exponent toward *flat* — i.e. against the reported direction.
5. ⚠ **`K = 5` on a 5-factor dataset is a coincidence, not a design.** Both datasets got Shapes3D's `K` because a
   matched sweep demands it, but on dSprites that happens to equal `n_factors`. ⛔ Nothing here depends on it; it is
   flagged so nobody later reads meaning into it.

## Proposed handover updates (for the Hub)
- **§7 Known Issues — ADD:** *"`gen-codebook-allocation` §5d's `−0.2631 ± 0.0555` and its reading *'the exponent is
  essentially CONSTANT across `PR`'* are both superseded by `gen-dsprites-exponent` (7-point sweep, same rows, same
  code path): the Shapes3D exponent is **−0.2500 ± 0.0094** and the exponent **does** depend on `log PR`
  (`+0.0702 ± 0.0136`, `p = 2e−4`) — ~3× too weakly for `−2/PR`, but not constant. ⛔ The `± 0.0555` spread is a
  3-point-sweep artefact and must not be quoted as the precision of the exponent."*
- **§7 Known Issues — ADD:** *"§11b's factor-manifold prediction is **answered: NO** (pre-registered non-overlap
  rule, missed by 0.0011; means order correctly, well-level `p = 1.5e−4`). ⛔ Per §11b's own wording the
  factor-manifold account is REFUTED and the universality of `−0.26` is OPEN. ⛔ The result may not be re-labelled a
  pass on the strength of the mean."*
- **§3 config — ADD:** a new append-only block `GenCodebookExponentConfig` / `gen_codebook_exponent_config()` at the
  tail of `chlu/config.py` (defaults: `m_grid = [4, 8, 16, 32, 64, 128, 186]`, `k_wells = 5`,
  `datasets = [shapes3d, dsprites]`, per-dataset `bridge_root_by_dataset`). ⛔ Changes no existing default.
- **§5 provenance — ADD:** `exp_gen_codebook_exponent` is the **first** gen experiment that runs **both** datasets
  in one process through one code path, and the **first** place the dSprites bridge artefacts are consumed by a
  store. ⭐ Its `M = 64` Shapes3D row reproduces `gen-codebook-allocation`'s `0.2783 ± 0.0064` exactly — a usable
  standing cross-harness checksum for anything that re-fits that store.
