# PREREG — `camels-pub-tripwire` (matched-byte tripwire, CAMELS-US ungauged / PUB)

**Filed:** 2026-09-07, **before** any PUB tripwire/store number existed.
**Agent:** results-analyst. **Repo HEAD at filing:** `7fcef50`. `git status --porcelain`: **empty**.
**Zero worktrees, zero branches, zero tracked-code edits.** Local-only (M1 Pro, 8 cores, 16 GB RAM).
Protocol §5 pre-registration rule applies: the acceptance criterion is a measured comparison against
registered thresholds.

## ⭐ DIAL DECLARATION (echoed verbatim from the task file)
- **Dial:** *none — venue admissibility / instrument.* ⛔ No CLU cell, store, byte dividend or
  performance number is computed or implied. This measures the **classical** competition only.
- **Laundering control:** the shuffled-order null leg (§6) plus the companion rows (§5) — a store arm
  that does no better than day-of-year climatology is measuring the calendar, not the catchment.
- **Falsifies (fires the tripwire):** best in-budget store arm within 0.02 abs / 2 % rel of **any**
  reference in the frozen set.
- **Does NOT falsify:** the store losing badly to the LSTM references — that is the metric-native-ceiling
  theorem, is expected, and is not news. ⛔ Nor does a *fired* tripwire falsify the CLU; it falsifies the
  **venue**.

---

## §0 — DISCLOSURE of everything computed before this prereg was filed

**D0.1 — Precondition check.** `.claude/outputs/camels-caravan-refs.md` exists (39,175 B) and contains
the literal string `## REFERENCE SET (FROZEN)` (1 occurrence). Precondition satisfied.

**D0.2 — Documents read (no numbers of ours produced):** the amended refs output **including its §8
addendum V1–V5**; `.claude/outputs/c3-trackb-tripwire/PREREG.md` §1–§2; the gauged
`VERDICT.json`; the gauged code (`build_camels.py`, `camels_tripwire.py`, `camels_local.py`,
`analyse.py`). **Gauged result I am carrying knowledge of:** the gauged tripwire cleared at
E = 0.25586 vs S = 0.7580 (best in-budget arm: regional, W=30 ⊕ 27 statics, global z-scaling,
L = 2761 k-means, k = 3, distance-weighted, raw target). Gauged companion rows I know:
per-basin DOY climatology **0.0111**, train-period mean flow **−0.0073**, 1-day persistence
**0.4434** (out-of-protocol). ⚠ These bias my §7 point predictions and are declared here.

**D0.3 — Staged data checksums verified against `MANIFEST.json` (all three match):**

| file | sha256 | bytes |
|---|---|---|
| `attributes.npz` | `011f3d7c6525c91304d11e3d15427cb17fce3f1388a7a712dd6642752309b626` | 95,770 |
| `discharge.npz` | `f7c3f6976ea9b3753d8d3829d5bebdbdaa8889e0f638917906d9af333fc00a6e` | 9,702,885 |
| `forcing_daymet.npz` | `2577198f7a1f3852ff0113613c047f790166beea9e97638a155ab5e1a1d5d0c4` | 87,741,872 |

**D0.4 — Li et al. 2025 control artefacts downloaded** (Zenodo 10.5281/zenodo.16895228, CC BY 4.0,
`metadata.license.id = "cc-by-4.0"`, verified via the Zenodo REST API):

| file | bytes | md5 (published → observed) | sha256 (observed) |
|---|---|---|---|
| `py999_PUB_simulations.npy` | 530,583,824 | `1f83befcd26861e7588075522e1658d2` → **match** | `4ab3a47066f29c898d3b27dfb2aec2c76287828bed283385e14e54247c18a8e9` |
| `py999_OBS.npy` | 43,444,424 | `1d913eaf2cbe4c0807a6ccc1891fb8f2` → **match** | `299fb047eb63616f9073ddbe76eb2bc891233f00a25cfc307ba2664cf65128c4` |
| `py999_ids.npy` | 4,376 | `9ad03e474ef46299f8167e59088d98a2` | — |
| `readme.md` | 1,126 | `170b30f655c867b9b4e1eb8ae11a0b0e` | — |

**D0.5 — the Zenodo readme pins the array layout** (verbatim): *"axis = 0 — LSTM simulations under
Daymet, NLDAS, and Maurer forcings, followed by δHBV simulations. axis = 1 — Different random seeds
used for ensemble runs. axis = 2 — 531 basins, with corresponding basin IDs provided in the
accompanying ids file. axis = 3 — Daily time steps starting from **1980-10-01** (inclusive)."*
⇒ axis-0 order is registered as `[LSTM·Daymet, LSTM·NLDAS, LSTM·Maurer, δHBV·Daymet, δHBV·NLDAS,
δHBV·Maurer]` and this ordering is **itself tested** by the control in §3 (if the medians land on the
wrong published rows, the order is wrong and I say so).

**D0.6 — basin-id ordering mismatch found and will be handled by id-alignment** (task A3 instruction).
`py999_ids.npy` is `int64`, 531 entries; zero-padded to 8 chars it is the **same set** as
`ref/basin_list_531.txt` but a **different order** — 439 of 531 positions differ (first divergence at
index 22: Zenodo `01333000` vs list `04296000`). Zenodo's ids are numerically ascending; the published
basin list is not. ⇒ **all Li-2025 comparisons align by basin id, never by position.**

**D0.7 — the Feng/Li 35-attribute static set was built** (task A4).
`.claude/scratch/camels-pub-tripwire/ref/attrs35.npz`, sha256
`1904cfef7c5e1ed5fbc2e3cef50c4642a972a6cd1dfde127803adf9748695343`, 234,054 B, from the raw
`camels_{clim,topo,vege,soil,geol}.txt` files. All 35 names in refs §8 V5 resolve to a CAMELS column
(two spelling fixes registered: refs `soil_depth_statgso` → CAMELS `soil_depth_statsgo`; refs
`geol_porosity` → CAMELS's shipped typo `geol_porostiy`). Three attributes are **categorical**
(`dom_land_cover` 12 levels, `geol_class_1st` 12, `geol_class_2nd` 13 + a `NONE` level for basins with
no second class) and are carried in **two encodings**, ordinal (dim 35) and one-hot (dim 70); the max
over the two is consumed (§4). Missing-value counts before imputation, over 531 basins:
`root_depth_50` **13**, `geol_porosity` **3**, everything else **0**. Registered imputation rule:
median over the 531 basins (`geol_class_2nd_frac` would have taken the 0.0 rule but had 0 missing).
⛔ The gauged run's **27** statics are NOT used for any PUB arm.

**D0.8 — nothing else has been computed.** No fold assignment, no NSE of ours, no store, no kNN, no
loader-control number, no companion row.

---

## §1 — REFERENCE SET (FROZEN), copied verbatim from `.claude/outputs/camels-caravan-refs.md`
### as amended by its §8 addendum (V1–V5). The kill rule is evaluated against EVERY row.

| # | reference_name | median_NSE | status | basin_set | folds | period | notes |
|---|---|---|---|---|---|---|---|
| R1 | `li2025_LSTM+dHBV_123` ⭐FRONTIER | **0.785833** | confirmed (publisher xlsx, Hub-parsed) | 531 | 10 groups × 10 rounds, 3 seeds | train 1980–1999 / test 1995–1999 | tri-forcing ensemble |
| R2 | `li2025_LSTM_123` | 0.768181 | confirmed | 531 | as R1 | as R1 | |
| R3 | `li2025_dHBV_123` | 0.757749 | confirmed | 531 | as R1 | as R1 | |
| R4 | `li2025_LSTM+dHBV_daymet` ⭐**FORCING-MATCHED PRIMARY** | **0.742270** | confirmed | 531 | as R1 | as R1 | our cache is Daymet-only |
| R5 | `heudorfer2026_EMB-LSTM_ALPHAEARTHplus` | 0.721 | from its Table 3 (abstract's 0.726 is unsourced; quote 0.721) | 531 | 5-fold | test 1989-10→1999-09 | bootstrap-median convention |
| R6 | `li2025_dHBV_daymet` | 0.706809 | confirmed | 531 | as R1 | as R1 | |
| R7 | `li2025_LSTM_daymet` | 0.702636 | confirmed | 531 | as R1 | as R1 | |
| R8 | `heudorfer2026_baseLSTM_CAMELS27` | 0.696 | Tables 2/3 | 531 | 5-fold | as R5 | |
| R9 | `li2025_LSTM_maurer` | 0.694156 | confirmed | 531 | as R1 | as R1 | |
| R10 | `kratzert2019_LSTM_pub_ensemble` | 0.69 (Hub-recomputed **0.6931**, 531 rows) | confirmed | 531 | 12-fold, seeds 300–309 | test 1995-10→2010-09 | NLDAS-extended forcing |
| R11 | `feng2023_dHBV(bt,gt)_daymet` | 0.69 ⚠prov | provisional | 531 | 10 groups | 1989–1999 | |
| R12 | `li2025_dHBV_nldas` | 0.670636 | confirmed | 531 | as R1 | as R1 | |
| R13 | `feng2023_LSTM_daymet` | 0.66 ⚠prov | provisional | 531 | 10 groups | 1989–1999 | |
| R14 | `feng2023_LSTM_maurer` | 0.65 | confirmed (§8 V5) | 531 | 10 groups | 1989–1999 | |
| R15 | `kratzert2019_SACSMA_calibrated` | 0.64 (Hub-recomputed **0.6447**) | confirmed | 531 | n/a, per-basin calibrated | test 1995-10→2010-09 | |
| R16 | `feng2023_dHBV(bt,gt)_maurer` | 0.64 | confirmed (§8 V5) | 531 | 10 groups | 1989–1999 | |
| R17 | `heudorfer2026_baseLSTM_dynamiconly` | 0.634 | its Table 2 | 531 | 5-fold | as R5 | |
| R18 | `qu2026_ALPHAEARTH` | 0.612 ⚠secondary | secondary (via Heudorfer 2026 l.375) | 531 stated | not verified | not verified | |
| R19 | `kratzert2019_NWM_reanalysis` | 0.58 (Hub-recomputed **0.5766**) | confirmed | 531 | n/a | test 1995-10→2010-09 | ✅ lowest CONFIRMED row |
| R20 | `qu2026_CAMELSattributes` | **0.553** ⚠secondary | secondary | 531 stated | not verified | not verified | ⭐ **lowest live row — the binding threshold** |
| ⛔ | ~~`feng2023_MPR+mHM` 0.53~~ | **STRUCK** | in-sample / temporal test, NOT PUB (refs §8 V5) | — | — | — | ⛔ must not arm the kill rule |
| — | `caravan` | **NOT-FOUND** | `caravan_protocol: NONE` | — | — | — | out of scope (A7) |

**Primary threshold named:** the **forcing-matched primary is R4 = `li2025_LSTM+dHBV_daymet`,
median NSE 0.742270** — our staged cache is Daymet-only and Li publishes a Daymet-only column, so this
is the like-for-like headline. **The BINDING threshold (the row the kill rule actually bites on) is
R20 = `qu2026_CAMELSattributes` = 0.553** (⚠ secondary), with the lowest **confirmed** row being
R19 = 0.5766. ⇒ **the store must stay below ≈ 0.533** (2 % / 0.02 below R20) **to clear.**
⛔ The remembered "0.69 frontier" and the "0.51 floor" (refs §8 V4) are both superseded and are not used.

---

## §2 — PROTOCOL (pinned by task amendment A1)

**Verdict-binding protocol: Li et al. 2025 PUB.**
- 531 basins (`ref/basin_list_531.txt`), Daymet forcing only.
- 10 random basin groups, 10 rounds; every basin held out **exactly once**.
- Store/"training" window (exemplars, all scaling statistics): **1980-10-01 → 1999-09-30**.
- Test window: **1995-10-01 → 1999-09-30** (1,461 days).
- Test results **concatenated across rounds before metrics**; because every basin is held out exactly
  once, this is identical to computing each basin's NSE on its own held-out prediction.
- Metric: **NSE, unclipped, range (−∞, 1]**, per basin, then **median over 531**. Days with
  `obs < 0` (the −999 missing flag) are dropped from both arrays first (Kratzert `calc_nse`
  convention, inherited verbatim from the gauged run's D4). Mean and n(NSE ≤ 0) also reported.
- **Fold assignment: ⛔ RECONSTRUCTED, NOT PUBLISHED.** No PUB reference publishes its fold file
  (refs §3). I use `sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=s)` over the
  published `basin_list_531.txt` order, **s ∈ {0, 1, 2}**, and report the across-seed spread. Every
  reported margin is against the *published* protocol but *our* partition. This is stated in the
  report as a limitation, not hidden.

**Secondary, non-verdict-binding leg: Kratzert et al. 2019**, `KFold(n_splits=12, shuffle=True,
random_state=s)`, s ∈ {300…309}, test **1995-10-01 → 2010-09-30**, store window
1980-10-01 → 1995-09-30. Exactly reconstructible; run **if time allows**, and reported as NOT RUN
otherwise. ⚠ Forcing mismatch: R10 used NLDAS-extended, we have Daymet.

**⛔ A2 LEAK RULE (registered as an executable assertion).** PUB is a *spatial* holdout: dates may
overlap, basins may not. Therefore the store's exemplars are drawn **only from training-fold basins**.
`assert not (set(exemplar_basin_ids) & set(test_fold_basin_ids))` runs on **every** store built, in
every arm and every fold, and the count of assertion executions is reported. Any arm that needs the
held-out basin's own discharge (per-basin target standardisation by its own μ,σ; per-basin DOY
climatology; persistence) is **labelled LEAKY, reported as an out-of-protocol upper bound, and is
never verdict-admissible.**

---

## §3 — LOADER POSITIVE CONTROL (must pass before any tripwire number is quoted)

Compute NSE with **our own discharge array** (`staged/discharge.npz`), **our own date/hydrological-year
alignment**, **our own basin ordering (aligned by id)** and **our own NSE code**, against **Li 2025's
published per-basin daily simulations**, on the Li PUB window 1995-10-01 → 1999-09-30, seed-mean over
axis 1, median over 531.

| target row | published median NSE |
|---|---|
| `LSTM + δHBV, Daymet` (mean of the two models' seed-mean sims) | **0.742270** |
| `LSTM, Daymet` | **0.702636** |
| `δHBV, Daymet` | **0.706809** |

**PRE-REGISTERED PASS TOLERANCE (stated before running): `|Δ median NSE| ≤ 0.010` absolute on ALL
THREE rows simultaneously.** Rationale: the only degrees of freedom are our discharge array, our
date alignment and our NSE implementation; Li's own metric convention (unclipped, concatenated,
seed-averaged) is fully specified. A miss > 0.010 on any row means the loader, the window or the
metric is wrong. **If it misses: STOP and report. Do not proceed to tripwire numbers.**
A secondary, tighter *reporting* band `|Δ| ≤ 0.002` is recorded as "exact" but is **not** the pass bar.
I also register the ambiguity in "seed-mean": both `median(NSE(mean_over_seeds(sim)))` and
`mean_over_seeds(median(NSE(sim)))` are computed and the one matching the published value is
reported as the identified convention, with both printed.

**⛔ Known trap, must be re-armed (task §2).** The shipped **maurer** forcing product is defective
(`tmax == tmin` on > 99 % of days). The guard is `assert (tmax > tmin).mean() > 0.99`. I will
(a) confirm the guard **passes** on the staged Daymet array, and (b) **confirm the guard FIRES on
maurer**, read live out of `basin_timeseries_v1p2_metForcing_obsFlow.zip`, over a stated number of
basins. An untested guard is not a guard.

---

## §4 — ARMS, LADDER AND BYTE BUDGET

**Budget B = 1,966,080 B** (the program's Track-A budget, inherited from the gauged run).
**Second ladder point — parameter-matched budget.** The frozen set gives a parameter count only for
R10 (`kratzert2019`): *"hidden_size 256 … Params ≈ 4·(256·(D+256)+256)+257 ⇒ ≈ 0.30 M at D = 32
(5 forcings + 27 attributes) — ⚠ input dim assumed, not read from the paper"*. Taking that literally:
4·(256·288 + 256) + 257 = **296,193 params → B_param = 1,184,772 B** at float32.
⚠ **B_param = 0.603 × B — the two budgets differ by 1.66×. Both ladders are reported and the choice is
ESCALATED to the Hub, not silently picked.** The verdict uses the **more generous B = 1,966,080**
(anti-hobbling: a smaller budget can only weaken the store, and a store that cannot fire at B cannot
fire at B_param). ⛔ Li 2025's own parameter count is **NOT-FOUND** in the frozen set.

**Byte convention:** float32 unless a variant is explicitly declared float16; bytes =
(key_dim × 4) + (payload_floats × itemsize). Registered here so it cannot be tuned later.

### Family DAY — exemplar = one (train-fold basin, day) pair. The task-§3 literal arms.
- **key spaces:** `S` (35 statics) · `W1` · `W30` · `W365` (forcing windows, 5 channels/day) ·
  `S+W1` · `S+W30` · `S+W365` (**"both combined"**). Optional `+doy` (sin/cos of day-of-year, 2 dims).
- **feature scaling:** `raw` · pooled-`global` z-score · `perbasin` z-score, statistics from
  **training-fold basins over the store window only**.
- **targets:** `raw` (mm/day) · `pmean` (store q/p_mean, de-normalise with the **query** basin's
  p_mean — non-leaky, p_mean is a static known for an ungauged basin) · `oracle` (z-score by the
  basin's own store-window μ,σ, de-normalise with the **query** basin's own μ,σ — ⛔ **LEAKY**,
  upper bound only, never verdict-admissible).
- **k ∈ {1, 3, 5, 10, 25}**; **distance-weighted** (`1/max(d,1e-9)`, normalised) and **uniform**.
- **selection:** uniform **random** (seeds 0,1,2) and **k-means condensation**
  (MiniBatchKMeans, `n_init=1`, seeds 0,1,2, target = cluster-mean).
- **L ladder:** `{250, 500, 1000, 2000, 5000, L@B_param, L@B, 20000, 50000}` — points above B are
  reported as an **unbounded upper bound** and are **never admissible**.

| key space | dim | B/exemplar | **L@B** | L@B_param |
|---|---|---|---|---|
| `S` (35 statics, ordinal) | 35 | 144 | 13,653 | 8,227 |
| `S` one-hot | 70 | 284 | 6,922 | 4,171 |
| `W1` | 5 | 24 | 81,920 | 49,365 |
| `S+W1` | 40 | 164 | 11,988 | 7,224 |
| `W30` | 150 | 604 | 3,255 | 1,961 |
| `S+W30` | 185 | 744 | **2,642** | 1,592 |
| `S+W30` one-hot | 220 | 884 | 2,224 | 1,340 |
| `W365` | 1,825 | 7,304 | 269 | 162 |
| `S+W365` | 1,860 | 7,444 | **264** | 159 |

### Family DONOR ⭐ — exemplar = one **train-fold basin**. The donor-basin-regionalization analogue.
The task names *"kNN over static catchment attributes"* as the dangerous arm because the PUB classical
competition is donor-basin regionalization. A statics-only **DAY** store is degenerate (all days of a
donor share one key), so running only the literal arm would **hobble the dangerous arm** — the exact
failure mode the task's "what would prove this spoke wrong" section warns about. Family DONOR is
therefore run **in addition to**, not instead of, the literal arms, and the max over both is consumed.
- **key:** 35 statics (ordinal or one-hot, z-scored over train-fold basins) · optionally ⊕ forcing-window
  climate descriptors (per-basin store-window means/stds of the 5 Daymet channels, 10 dims) ·
  optionally ⊕ (lat, lon).
- **payload (a) `doyclim`:** the donor's 366-day day-of-year climatology of q/p_mean over the store
  window. **payload (b) `concurrent`:** the donor's actual daily q/p_mean over the 1,461 **test-window**
  days. ⚠ (b) is legitimate under A2 — donor basins are train-fold basins and are **gauged**; Li's own
  train period 1980–1999 overlaps the test dates for train basins. This is the strongest form of the
  published method and it is the arm most likely to fire.
- **rescaling to query units (all non-leaky):** × query `p_mean` · × donor-weighted predicted mean flow
  · × `area_gages2` ratio (drainage-area-ratio transfer, the classical hydrological form).
- **k ∈ {1,3,5,10,25}**, dist / unif. **Donor selection** when L < n_train: random (seeds 0,1,2) and
  k-means condensation over the statics space (seeds 0,1,2).
- **L ladder:** `{5, 10, 25, 50, 100, 200, L@B, all_train (≈478)}`.

| donor arm | key dim | payload | B/donor | **L@B** | all-478 bytes | in budget? |
|---|---|---|---|---|---|---|
| `doyclim`, fp32 | 35 | 366 | 1,604 | 1,225 | 766,712 | ✅ all donors fit |
| `concurrent`, fp32 | 35 | 1,461 | 5,984 | **328** | 2,860,352 | ❌ 1.455 × B ⇒ upper bound |
| `concurrent`, **fp16 payload** (declared variant) | 35 | 1,461 | 3,062 | 642 | **1,463,636** | ✅ **all 478 donors fit** |
| `concurrent`, fp32, one-hot key | 70 | 1,461 | 6,124 | 321 | 2,927,272 | ❌ |

⭐ The **fp16-payload concurrent-donor arm is registered as the maximum-strength in-budget dangerous
arm** and MUST be run: dropping it would let a float-precision convention decide the verdict.
fp16 on q/p_mean (range ≈ 0–50, ~3 significant decimal digits) is ample.

---

## §5 — COMPANION ROWS (mandatory, every table)
1. **per-basin day-of-year climatology** (store window) — ⛔ **LEAKY under PUB** (needs the held-out
   basin's own discharge); reported, labelled, never verdict-admissible.
2. **regional / train-fold day-of-year climatology** of q/p_mean × query p_mean — **the non-leaky
   calendar control**, and the real laundering control for the dial declaration.
3. **spatial-nearest-neighbour donor** — nearest train-fold basin by great-circle distance from
   `gauge_lat`/`gauge_lon` (camels_topo.txt), transferring q/p_mean × query p_mean; k ∈ {1,3,5}.
4. **regional mean flow** — train-fold mean of q/p_mean × query p_mean. Also the query basin's own
   test-period mean (NSE = 0 by definition) and its own store-period mean (⛔ LEAKY), both printed.
5. **published regionalization baseline sourced from the refs output:** R20
   `qu2026_CAMELSattributes = 0.553` and R18 `qu2026_ALPHAEARTH = 0.612` — both ⚠ **secondary**;
   task A6 requires retrieving arXiv:2601.01558 and confirming or refuting them against the primary.
6. **1-day persistence** — ⛔ **LEAKY / DIFFERENT TASK**, printed separately and labelled, exactly as
   the gauged run did.

## §6 — SHUFFLED-ORDER NULL LEG
Independent per-query random permutation of the W time slots (all 5 channels permuted jointly,
seed 0), applied to **queries and stored keys alike**, re-run on the best W30 and best W365 DAY arms.
If shuffling does not hurt, the window carries no exploitable *ordering* information at these budgets.

---

## §7 — POINT PREDICTIONS (committed before measurement)

| # | prediction | point | 80 % interval | prior |
|---|---|---|---|---|
| P1 | §3 loader control passes on all three rows at `\|Δ\| ≤ 0.010` | PASS | — | p = 0.80 |
| P1b | the registered axis-0 order `[LSTM·Dm, LSTM·NL, LSTM·Mau, δHBV·Dm, δHBV·NL, δHBV·Mau]` is right | TRUE | — | p = 0.85 |
| P1c | the maurer guard FIRES (`frac(tmax>tmin) ≤ 0.01` on maurer) and PASSES on daymet | TRUE | — | p = 0.95 |
| P2 | DAY `S+W30`, at-budget L = 2,642, best over all axes — median NSE | **0.24** | 0.05 … 0.40 | |
| P3 | DAY `S+W365`, at-budget L = 264, best over all axes — median NSE | **0.06** | −0.05 … 0.20 | |
| P4 | ⭐ DONOR `concurrent` fp16, all 478 donors, best k/weight/rescale — median NSE | **0.38** | 0.10 … 0.58 | |
| P5 | DONOR `doyclim`, all donors, best over axes — median NSE | **0.05** | −0.10 … 0.22 | |
| P6 | DAY statics-only `S` (the degenerate literal arm), at budget — median NSE | **0.03** | −0.10 … 0.15 | |
| P7 | companion: spatial-NN donor (k = 1, concurrent) — median NSE | **0.28** | 0.05 … 0.50 | |
| P8 | companion: train-fold regional mean flow — median NSE | **−0.08** | −0.35 … 0.05 | |
| P9 | companion: non-leaky regional DOY climatology — median NSE | **−0.03** | −0.25 … 0.12 | |
| P10 | companion: LEAKY per-basin DOY climatology — median NSE | **0.05** | −0.08 … 0.22 | |
| P11 | ⭐ **E** = best **in-budget** arm over ALL families — median NSE | **0.40** | 0.12 … 0.58 | |
| P12 | `margin_abs` = E − R4 (0.742270) | **−0.34** | −0.62 … −0.16 | |
| P13 | `fires` against R20 (0.553, the binding row) | **FALSE** | — | p = 0.72 |
| P14 | `fires` against R19 (0.5766, lowest confirmed) | **FALSE** | — | p = 0.78 |
| P15 | ⭐ overall `dies` (fires against ANY row) | **FALSE** | — | **p = 0.70** |
| P16 | shuffled-order null: Δ median NSE (shuffled − ordered), best W30 DAY arm | **−0.08** | −0.25 … +0.02 | |
| P17 | one-hot categorical encoding beats ordinal by ≤ 0.02 median NSE | TRUE | — | p = 0.7 |
| P18 | the `oracle` (LEAKY) target beats the best non-leaky target by ≥ 0.08 median NSE | TRUE | — | p = 0.75 |
| P19 | peak RSS of the largest arm stays under 8 GB (of 16 GB) | TRUE | — | p = 0.85 |
| P20 | across-fold-seed spread of E (3 fold seeds), max − min median NSE | **0.015** | 0.003 … 0.05 | |
| P21 | A6: Qu et al. 2026 primary confirms 0.553 / 0.612 to ±0.01 | TRUE | — | p = 0.6 |

**Why P15's prior (0.70) is far below the gauged run's 0.88.** The gauged tripwire's competition was a
temporal split, where a store is structurally disadvantaged. Here the classical competition **is** a
similarity lookup over static attributes, and the `concurrent`-donor arm has access to a gauged
neighbour experiencing the *same storms* — this is the published method, not a strawman. I am
genuinely less confident, and I am recording that before measuring.

**What would surprise me (registered in advance):**
- P4 landing **above 0.55** ⇒ the tripwire FIRES ⇒ per the kill rule I stop, write up, and escalate to
  the Head and the Main Advisor. ⛔ No softening, no re-framing, no hunting for a saving framing.
- P4 landing **below 0.10** ⇒ I treat the dangerous arm as **suspected hobbled/buggy** and chase it
  (task "what would prove this spoke wrong"). ⛔ A weak dangerous arm is NOT banked as good news.
- P10 (leaky per-basin DOY climatology) landing above 0.30 ⇒ contradicts the gauged run's 0.0111 and
  means the window/period handling differs; would be chased before anything is banked.

## §8 — KILL RULE, quoted verbatim from the task file (not re-derived)
> **The venue DIES if the best in-budget store arm comes within 0.02 absolute or 2 % relative of the
> best strong reference.** Evaluated and reported against **EVERY** reference in the frozen set, not
> just the strongest.

Operationalised, per reference `R` with median NSE `S`, and `E` = best **in-budget** store arm:
`margin_abs = E − S`; `margin_rel = (E − S)/S`;
`fires(R) = (margin_abs ≥ −0.02) OR (margin_rel ≥ −0.02)`; `dies = OR over all R in §1`.
⚠ A fired tripwire falsifies the **VENUE**, not the CLU.

## §9 — Registered free choices (so they cannot be tuned after the fact)
1. `float32` blocked distances via `‖q‖² + ‖k‖² − 2q·k`; ties → lowest index (the gauged-run convention).
2. Windows reaching outside the record are dropped; no zero-padding.
3. Wall-clock escape hatch: every **in-budget**, verdict-deciding number is computed on the **full**
   1,461-day test window with all 531 basins. Purely characterising **over-budget** ladder points may
   be computed on a 1-in-3 day subsample, flagged `[sub3]`. ⛔ No verdict number may come from a
   subsampled row. ⛔ No basin set is ever silently shrunk (task A8).
4. Seeds: fold partition {0,1,2}; exemplar random selection {0,1,2}; k-means `random_state` {0,1,2}.
   Deterministic arms carry no seed and say so.
5. Peak RSS is recorded per arm (`resource.getrusage(RUSAGE_SELF).ru_maxrss`). An arm that does not
   fit is reported **NOT RUN — exceeds local RAM**, naming the arm.
6. `p_mean` is taken from CAMELS `camels_clim.txt` (mm/day), i.e. a **static attribute**, and is
   therefore available for an ungauged basin. Using it for target rescaling is non-leaky by
   construction; this is stated so it cannot be challenged later.

---
*Filed before the loader control ran and before any store arm existed.
Scored honestly in `.claude/outputs/camels-pub-tripwire.md`.*
