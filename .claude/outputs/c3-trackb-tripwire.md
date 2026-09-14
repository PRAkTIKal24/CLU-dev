# c3-trackb-tripwire — results-analyst report

Task + acceptance criterion: file `PREREG.md` before any harness runs, then measure the criterion-4
tripwire against the scorecard's own `dies_if` rule for **CAMELS-US** and **N-CMAPSS DS02**, supply
N-CMAPSS's missing criterion-2 rows, and resolve the CAMELS licence. **Status: done.**

## ⛳ RECONCILIATION LIST (owner needed — protocol §5 corollary, in the first 10 lines)
1. **CAMELS licence is RESOLVED and OPEN: `cc-by-4.0`** (Zenodo 15529996 = DOI 10.5065/D6MW2F4D, v1.2).
   Every doc saying "explicit licence string NOT CONFIRMED — the one open blocker" must be updated.
2. **Scout correction:** `crit1_baselines.best_deep` says "LSTM-with-static-inputs ensemble **0.72**".
   0.72 is the **mean**; the **median** (the currency of `dies_if`) is **0.758**, and it is the
   *highest* number in Kratzert Table 3 — above EA-LSTM's 0.742. The registered `S` changes.
3. **Scout size figures are wrong in both directions:** CAMELS total is **14.57 GB measured** (the
   "15 GB" secondary figure survives; "130 GB uncompressed" is still NOT VERIFIED); **N-CMAPSS is
   15.76 GB, not "several GB"** — 3× the scout's relayed figure.
4. **⛔ The shipped CAMELS SAC-SMA model output is NOT an out-of-sample benchmark** and must never be
   quoted against the LSTM (§3.2). Kratzert's 0.603 is a split-sample series; the shipped one scores
   **0.708** on the same basins/window and ≈ the same in every window ⇒ calibration-period fit.
5. **⛔ The shipped CAMELS `maurer` forcing product is defective**: tmax == tmin on >99 % of days in
   20/20 basins sampled, and 3 files have a malformed header. Any harness must use `daymet` (or
   source `maurer_extended` separately). For `c3-csf3-harness`.
6. **A defect in my own harness, disclosed, corrected and quantified (§5.4):** the first
   distance-weighted rows omitted `‖q‖²` from the squared distance. Rankings — hence every
   uniform-weight and every k = 1 row — were unaffected; corrected re-runs moved the consumed max by
   **+0.0028** (0.2531 → 0.2559). No verdict, ordering or conclusion changes.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial / pillar:** **none — admissibility instrument.** No CLU cell, store, controller, dividend or
  performance number of any kind was run, computed or claimed here.
- **Laundering control:** **I am the laundering control.** The exemplar store at matched bytes is the
  launder and this task measures how strong it is.
- **Falsifies venue adoption:** the exemplar store meeting/beating the strong reference under the
  scorecard's `dies_if`.
- **Does NOT falsify anything about the CLU.**

> ⛔ **A fired tripwire falsifies the BENCHMARK's admissibility, not the CLU** (C2W10 §1, verbatim).
> Neither tripwire fired here; the same sentence governs the FNN-reference caveat in §4.4.

## ⭐ HEADLINE (one line per venue)

- **CAMELS-US — `criterion4_cleared = TRUE`. ADOPT-RECOMMENDED** (subject to the Head/Advisor; the
  other four criteria were already PASS with primary sources). `E = 0.2559` vs `S = 0.7580`,
  **margin_abs = −0.5021**, **margin_rel = −0.6625**. The tripwire does not fire against **any** of
  the **14** references available, including the weakest published model (VIC-CONUS 0.307,
  margin −0.0511). Threshold is −0.02; we are **25× clear of it**.
- **N-CMAPSS DS02 — `criterion4_cleared = TRUE`, and `criterion 2 = PASS` (now measurable).**
  `E = 7.988` RMSE vs best strong `4.14` (**+92.9 % relative**). ⚠ **But the byte-matched classical
  store ties the *weakest* published deep model** (FNN data-driven 7.89 ± 0.12; ratio 1.0124, which
  *does* trip the 2 % rule against that one reference). Criterion 5 remains PARTIAL. **FALLBACK
  ONLY, as ranked.**

---

## Flag provenance (mandatory, protocol §5)

| item | value |
|---|---|
| repo HEAD before **and** after | **`7fcef50`** — `git status --porcelain` shows **0 modified tracked files** at both times |
| tracked code touched | **none.** No `chlu/`, no `tests/`, no `pyproject.toml`, no `uv.lock`. No branch, no commit, no worktree, no push |
| environment | scratch venv `.claude/scratch/c3-trackb-tripwire/.venv`, CPython **3.12.9**, `numpy 2.5.2`, `pandas 3.0.5`, `scipy 1.18.0`, `scikit-learn 1.9.0`, `h5py 3.16.0`, `matplotlib`. Freeze: `…/c3-trackb-tripwire/scratch_venv_freeze.txt`. **Project venv and lock untouched** (Head ruling 6 / the pandas precedent) |
| CAMELS byte budget | **B = 1,966,080 B** (scorecard convention, unchanged; ladder reported so re-pricing is a lookup) |
| CAMELS protocol | 531-basin list (Newman/Kratzert), margins on the **447-basin** common set; train/store window **1999-10-01…2008-09-30**; test **1989-10-01…1999-09-30**; forcing **Daymet** (5 vars); discharge `28316846.592·Q_cfs·86400/(area·1e6)`; **`obs < 0` dropped** before NSE; 27 static attributes (exactly 27, 0 NaN) |
| CAMELS arm grid | window ∈ {1, 30, 365}; scaling ∈ {raw, per-basin z, pooled z} (**train-period stats only**); target ∈ {raw mm/day, per-basin standardised}; k ∈ {1,3,5,10,25}; weight ∈ {distance, uniform}; selection ∈ {random seeds 0,1,2 ; MiniBatchKMeans condensation `random_state=0, n_init=1, batch_size=4096`, fitted on a 50,000-row sample of the training pool}; store ∈ {regional, local}; L ladder {250,500,1000,2000,5000, at-budget} |
| CAMELS `[sub3]` | over-budget ladder points only, 1-in-3 test days (pre-registered escape hatch). **No verdict number is subsampled.** |
| N-CMAPSS protocol | DS02, units train {2,5,10,16,18,20} / test {11,14,15} (verified from the file); **1 Hz file down-sampled ×10 → 0.1 Hz** to match the reference's `m*`; dev 526,345 rows, test **125,375** rows; inputs `W`(4) ⊕ `X_s`(14) only — ⛔ `X_v`/`T` never used; RMSE per-sample; NASA s with α=1/13 under-estimate, 1/10 over |
| wall-clock | **≈ 2 h 50 m** total, laptop CPU only, **0 GPU, 0 CSF3 jobs, 0 training runs**. Downloads ≈ 55 min of it |
| commands | every script verbatim in `.claude/outputs/c3-trackb-tripwire/code/`; run lines in `run_regional.sh` / `run_phase2.sh` |

### Data provenance (measured, not assumed)

| file | bytes | hash | verified against |
|---|---|---|---|
| `basin_timeseries_v1p2_metForcing_obsFlow.zip` | 3,406,626,583 | md5 `8e9a466710e8270b58f01d332a87184f`; sha256 `bb90fca29a5bab8f0b300787a5ab5a669a366f40ba95b6877b631a81133f7e5d` | **md5 identical to the Zenodo manifest** |
| `basin_timeseries_v1p2_modelOutput_daymet.zip` | 4,207,763,546 | md5 `f2af624b6277b75b3e410d6a0365591a`; sha256 `5c92269e57808332d31e3ece9f3d45b22d8dd1fc8cad3b2f9c6c6a1cffd5af78` | **md5 identical to the Zenodo manifest** |
| `N-CMAPSS_DS02-006.h5` | 2,450,472,504 | sha256 `47971a68b239ecb756833218a95d68ded6eb7e63ee84e86671c8b188de1ca765` | extracted **from the NASA PCoE archive itself** by HTTP-range streaming (§4.1) |
| staged `forcing_daymet.npz` / `discharge.npz` / `attributes.npz` | 87,741,872 / 9,702,885 / 95,770 | sha256 `2577198f…d5d0c4` / `f7c3f697…c00a6e` / `011f3d7c…09b626` | `…/data/c3-camels/staged/MANIFEST.json` |

Cache paths (gitignored, stable, reachable by every spoke):
`/Users/user/Desktop/CHLU/.claude/data/c3-camels/` (7.2 GB) · `…/.claude/data/c3-ncmapss/` (2.3 GB).

---

## 1. The Head-facing blocker: the CAMELS licence is RESOLVED — `CC BY 4.0`

Checked first, in ~10 minutes, before any download (task §1).

- NCAR/RAL's CAMELS page names **`dx.doi.org/10.5065/D6MW2F4D`** as the dataset of record and says
  *"BY DOWNLOADING THE DATASET, YOU ARE AGREEING TO BE BOUND BY OUR TERMS OF USE"* + "License Type:
  License Required" — which is where the scout's uncertainty came from.
- **That DOI now resolves to Zenodo record 15529996.** Zenodo REST API:
  `metadata.license.id = "cc-by-4.0"`, `metadata.access_right = "open"`, `version 1.2`,
  `publication_date 2022-06-24`, **15 files, 14,565,262,987 B**, each with an upstream md5.
- Both files we downloaded reproduce their **Zenodo md5 exactly** ⇒ the copy is the copy of record.

⇒ **Mirroring to CSF3 is permitted with attribution.** Not blocked. Recommended citation strings are
in the record (Newman et al. 2014 dataset DOI; Addor et al. 2017 attributes DOI 10.5065/D6G73C3Q).
The scout's "≈15 GB compressed" is **corroborated by measurement (14.57 GB)**; its "130 GB
uncompressed" is **still NOT VERIFIED** and is not quoted anywhere in this report.

---

## 2. The strong reference, pinned to the primary table (and one scout correction)

`hess-23-5089-2019-t03.xlsx` — the publisher's own machine-readable Table 3 (447 basins,
validation period). This retires the scout's `~` rows:

| model | NSE mean | **NSE median** | n(NSE ≤ 0) |
|---|---|---|---|
| **LSTM (static inputs) ensemble** | 0.718 | **0.758** | 1 |
| **EA-LSTM ensemble** | 0.705 | **0.742** | 1 |
| LSTM single | 0.685 | 0.731 | 1 |
| EA-LSTM single | 0.674 | 0.714 | 2 |
| HBV upper (100 calibrated) | 0.631 | 0.676 | 9 |
| mHM (basin) | 0.627 | 0.666 | 7 |
| FUSE 902 / 900 / 904 | 0.611 / 0.587 / 0.582 | 0.650 / 0.639 / 0.622 | 10 / 12 / 9 |
| SAC-SMA | 0.564 | 0.603 | 13 |
| VIC (basin) | 0.518 | 0.551 | 10 |
| mHM (CONUS) | 0.442 | 0.527 | 29 |
| HBV lower (1000 uncal.) | 0.237 | 0.416 | 35 |
| VIC (CONUS) | 0.167 | 0.307 | 41 |

**Registered substitution, discharged:** the scorecard wanted "our re-trained LSTM/EA-LSTM"; a
training run is forbidden to this task, so I used the **published per-basin results themselves**.
`kratzert/ealstm_regional_modeling` (Apache-2.0) ships `notebooks/all_metrics.p` with per-basin NSE
/ α / β / FHV / FMS / FLV for every model and seed. The intersection of the ten benchmark models'
basin sets is **exactly 447 basins**, and recomputing Table 3 on it reproduces **12/12 rows to the
last printed digit** (SAC-SMA 0.6028→0.603, mean 0.5639→0.564, n≤0 13→13; LSTM ens 0.7580→0.758;
HBV-ub 0.6756→0.676; …). ⇒ **`S_primary = 0.7580`**, and the reference is available *per basin*,
which is what makes §3.4's regime-resolved comparison possible without training anything.

---

## 3. Venue A — CAMELS-US

### 3.1 The structural asymmetry under test (stated as the hypothesis, per task §1)
The target is a function of an **unobserved accumulated state** — soil moisture, groundwater,
snowpack. Two identical forcing windows in basins with different antecedent storage produce
different discharge, and the 27 static attributes only partially disambiguate. **If that
non-identifiability is material, no metric over the observable window can be the ceiling.** That is
the reason to *run* the tripwire, not a prediction that it clears. The measurement below says the
non-identifiability **is** material, and §3.4 localises where.

### 3.2 B1 — the loader positive control: **leg (i) PASS, leg (ii) FAIL-AS-REGISTERED with a diagnosed cause**

| leg | result |
|---|---|
| **(i) our discharge in mm/day vs CAMELS' own `OBS_RUN`** (447 basins × 3,653 days) | **PASS, decisively.** Per-basin max abs deviation: median **9.70e−5**, p95 **2.52e−4**, **max 4.03e−4 mm/day** — i.e. agreement to the printed precision of their own files. Area normalisation, date alignment and the −999 flag are verified end-to-end |
| **(ii) shipped SAC-SMA + Snow-17 median NSE vs Table 3's 0.603** | **FAIL vs the registered ±0.01 tolerance.** Ours: per-seed medians **0.7004–0.7069** (10 seeds), 10-seed ensemble **0.7079** (mean 0.678, n≤0 = 7). Δ = **+0.105** |

**Diagnosis (so the Hub can judge whether the "hard stop" bites).** It is *not* our loader:
- leg (i) is exact to 4e−4;
- the metric code is definitionally checked — the *test-period-mean* companion returns **NSE = 0.0000 exactly**;
- Table 3 is reproduced on 12/12 rows from Kratzert's own per-basin values using our 447 list and our aggregation (§2).

It is a **different SAC-SMA series**. On a 90-basin subsample the shipped output scores median NSE
**0.733 (test window) / 0.663 (train window) / 0.736 (1980–89)** — essentially window-independent,
the signature of a **full-record calibration**, i.e. in-sample everywhere. Kratzert's `SAC_SMA`
netCDF is the split-sample Newman-2017 benchmark, evaluated out-of-sample, hence 0.603.

⇒ **The intent of the control is discharged** (nothing in the verdict depends on our recomputation
of SAC-SMA: the store numbers depend on our forcing/discharge loader ✔, our NSE ✔, the 447 list ✔,
and the references are consumed as published). ⛔ **But reconciliation item 4 is now live for
`c3-csf3-harness`: the shipped CAMELS SAC-SMA output is an in-sample fit and is not a benchmark.**

### 3.3 The exemplar-store ladder (full table: `TABLES.md`; figure: `fig1_camels_ladder.png`)

Best over k × weight × target, per (window, scaling, L, selection). Selected rows:

| window | scaling | L | selection | store bytes | in budget | median NSE (447) |
|---|---|---|---|---|---|---|
| 30 d | **pooled z** | **2,761 (at budget)** | **k-means** | **1,965,832** | **yes** | **0.2559 ← E** |
| 30 d | pooled z | 2,000 | k-means | 1,424,000 | yes | 0.2536 |
| 30 d | pooled z | 2,761 | random (3 seeds) | 1,965,832 | yes | 0.1551 |
| 30 d | per-basin z | 2,000 | k-means | 1,424,000 | yes | 0.2393 |
| 30 d | **raw** | 2,761 | k-means | 1,965,832 | yes | **0.0485** |
| 30 d | pooled z | 5,000 | random | 3,560,000 | **NO** | 0.1397 `[sub3]` |
| 1 d | pooled z | 14,894 (at budget) | random | 1,966,008 | yes | 0.2520 |
| 1 d | pooled z | 5,000 | random | 660,000 | yes | 0.2109 |
| 365 d | pooled z | 265 (at budget) | k-means | 1,964,180 | yes | 0.0561 |
| 365 d | pooled z | 2,000 | k-means | 14,824,000 | **NO** | 0.0479 `[sub3]` |
| 365 d | pooled z | 5,000 | random | 37,060,000 | **NO** | 0.0173 `[sub3]` |

**Four things the ladder shows, all of which the C2W10 lessons predicted would matter:**
1. ⭐ **Anti-hobbling decided the size of the number again.** Raw-feature arms top out at **0.0485**;
   standardised arms reach **0.2559** — a **5.3×** difference. Running raw-only would have made the
   store look 5× weaker than it is.
2. ⭐ **A single store size cannot represent the family, and neither can a single *selection rule*.**
   k-means condensation beats random sampling by **+0.08 to +0.14 NSE at every L** (e.g. at the
   at-budget point 0.2559 vs 0.1551). The registered "random sample" arm alone would have
   under-stated the store by 65 %.
3. **Non-monotonicity in L reappears** (INSECTS' signature): 30 d peaks at the at-budget point but
   the *over-budget* L = 5,000 arm is **worse** (0.1397 pooled-z, 0.1263 per-basin-z); 365 d peaks
   at L = 265 (0.0561) and decays to 0.0173 pooled-z / **−0.0098** per-basin-z at L = 5,000. Storing
   more is not monotonically better here either.
4. **The 1,852-dim representation is a disaster for a 265-exemplar store** (0.0561) — the
   scorecard's mandatory 365-day variant is *dominated* by the 30-day variant at every budget.

### 3.4 The verdict, against **every** reference, plus the regime-resolved margin

`E = 0.2559` (best in-budget exemplar arm, maximised over {k, L, weight, raw|std, regional|local,
window, selection}).

| reference S (median NSE, 447) | margin_abs | margin_rel | fires? |
|---|---|---|---|
| **LSTM (static) ensemble 0.7580 — registered S** | **−0.5021** | **−0.6625** | **no** |
| EA-LSTM ensemble 0.7423 | −0.4864 | −0.6553 | no |
| LSTM single 0.731 | −0.4751 | −0.6500 | no |
| EA-LSTM single 0.714 | −0.4581 | −0.6417 | no |
| HBV upper 0.6756 | −0.4197 | −0.6213 | no |
| mHM (basin) 0.6659 | −0.4100 | −0.6158 | no |
| FUSE 902 / 900 / 904 | −0.3946 / −0.3830 / −0.3663 | −0.607 / −0.600 / −0.589 | no |
| SAC-SMA 0.6028 | −0.3469 | −0.5755 | no |
| VIC (basin) 0.5513 | −0.2954 | −0.5359 | no |
| mHM (CONUS) 0.5274 | −0.2715 | −0.5149 | no |
| HBV lower 0.4165 | −0.1606 | −0.3857 | no |
| **VIC (CONUS) 0.3070 — the weakest published model** | **−0.0511** | −0.1666 | **no** |

`dies_if` threshold: **−0.02**. ⇒ **`criterion4_cleared_camels = TRUE`.** Unlike INSECTS (which
fired by 0.10 points and would have flipped had ARF landed 0.11 higher), this verdict is **25× clear
of its threshold** and survives every reference including the worst one on the board.

**Flow-regime-resolved margin** (`deepdive_best2_ordered.json`; the venue's own Gupta/Yilmaz
decomposition, code copied from the reference implementation):

| metric | **store (E arm)** | LSTM ens | EA-LSTM ens | SAC-SMA | reading |
|---|---|---|---|---|---|
| median NSE | **0.2530** | 0.7580 | 0.7423 | 0.6028 | |
| **α-NSE** (σ_sim/σ_obs) | **0.440** | 0.843 | 0.810 | 0.779 | the store reproduces **44 %** of observed variability — regression to the mean |
| β-NSE | −0.056 | −0.032 | −0.030 | −0.066 | bias is *fine*; the store is unbiased and flat |
| **FHV** (top-2 % peak-flow bias) | **−56.4 %** | −15.7 % | −18.1 % | −20.4 % | the store misses **more than half of every flood peak** |
| FLV (bottom-30 % low-flow bias) | +27.1 % | +55.1 % | +31.9 % | +37.4 % | comparable |
| FMS (mid-slope bias) | −38.0 % | −8.8 % | −11.3 % | −14.3 % | |
| NSE on top-2 % flow days | **−1.771** | — | — | — | worse than that slice's own mean |
| NSE on Mar–Jun, snow basins (`frac_snow>0.3`, n=74) | **0.115** | — | — | — | the snowmelt-release slice is where it is weakest |

⭐ **This is the structural asymmetry, measured.** The store is a competent *climatological/wet-dry*
predictor (β ≈ 0, mid-range NSE ≈ 0.25) and a **failed peak and snowmelt predictor** (α 0.44,
FHV −56 %, high-flow NSE −1.77, snow-season 0.115) — exactly the regimes whose answer depends on
**accumulated storage that is not in the window**. The metric over the observable window is *not*
the ceiling on CAMELS, and the residual is concentrated where criterion 3 says the memory lives.

### 3.5 Every other required leg

- **LOCAL arms (same-basin, statics dropped) are WEAKER than regional at every budget.** The
  byte-matched local arm (5 exemplars/basin = 1,603,620 B) scores **−0.0432**; even the *entire*
  own-basin training record (3,287/basin = 1,054,219,788 B = **536× the budget**) reaches only
  **0.2173**, below the regional at-budget 0.2559. ⇒ the registered primary (regional) is also the
  stronger arm; no confound, and pooling beats per-basin memorisation on this venue.
- **Shuffled-order null (Metro precedent) — the venue is NOT moot.** Independent per-row permutation
  of the window's time slots, applied to **keys and queries alike**: the E-config falls
  **0.2530 → 0.0497 (Δ = −0.203)**; the per-basin-z config falls **0.2376 → 0.0685 (Δ = −0.169)`.**
  ⭐ Contrast Metro, where shuffling **helped** (320.98 → 311.75). **Ordering inside the window
  carries real, exploitable information on CAMELS**, so the tripwire result is not vacuous.
  (A query-only shuffle — which additionally breaks query/key correspondence — gives 0.0545.)
- **Mandatory companion rows:**

| arm | bytes | in budget | in protocol | median NSE (447) | mean | n(NSE≤0) |
|---|---|---|---|---|---|---|
| per-basin mean flow (**training** period) | 2,124 | yes | yes | **−0.0073** | −0.0212 | 447 |
| per-basin mean flow (**test** period) | 2,124 | yes | yes | **0.0000** | 0.0000 | 447 (all exactly 0) |
| per-basin **day-of-year climatology** | **777,384** | **yes** | yes | **0.0111** | −0.0359 | 202 |
| 1-day persistence | 4 | yes | ⛔ **NO — DIFFERENT TASK** | **0.4434** | 0.3729 | 72 |

  ⚠ **Persistence at 0.4434 beats every in-protocol classical arm we ran** — and it is *structurally
  unavailable* inside the simulation protocol (discharge is never an input). That is precisely the
  property that lets CAMELS survive the criterion-2 death that killed the LTSF suite and Metro; it is
  also a standing trap: **any harness that leaks q(t−1) into the input immediately manufactures a
  0.44-NSE "result".** ⛔ Never quote it in-protocol (the Metro leak error).
- **DECLARED POST-HOC, UNREGISTERED classical arms** (C2W10's "unregistered but stronger" precedent —
  reported so nobody can say the classical side was hobbled; **not** part of the registered
  arithmetic):

| arm | window | bytes | in budget | median NSE (447) |
|---|---|---|---|---|
| **per-basin ridge on the forcing window** | 30 d | **320,724** | **yes** | **0.4462** |
| pooled ridge (one model, + statics) | 30 d | 712 | yes | 0.3480 |
| pooled ridge (one model, + statics) | 365 d | 7,412 | yes | 0.3254 |
| per-basin ridge | 365 d | 3,878,424 | **NO** | 0.1801 |

  ⭐ **A 320 kB per-basin linear regression (0.446) is 1.74× stronger than the best 1.97 MB exemplar
  store (0.256)** — and still **−0.312 below the LSTM**. Even under this much harsher, non-exemplar
  reading of "classical method at matched bytes", the venue clears with a 0.31 margin.
- **Byte ledger:** 30 d ⊕ 27 statics = 177 dims → 712 B/exemplar → **2,761 at budget** (1,965,832 B,
  248 B slack) ✔ matches the scorecard; 365 d = 1,852 dims → 7,412 B/exemplar → **265** (1,964,180 B,
  1,900 slack) ✔; 1 d = 32 dims → 132 B → **14,894**; DOY climatology store 531×366×4 = **777,384 B**;
  local 30 d statics-dropped = 604 B/exemplar → 3,255 total ≈ **6 per basin**.

---

## 4. Venue B — N-CMAPSS DS02

### 4.1 Data, obtained from the original NASA PCoE files (⛔ CAFE embargo respected)
`https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`,
`Content-Length = 15,760,443,389` (**15.76 GB — the scout's "several GB" is wrong by ≈3×**). It is a
**nested** zip: one deflate member `data_set.zip` (15,814,385,805 B inflated) containing
`data_set/N-CMAPSS_DS0x-*.h5`. I extracted **only DS02** by HTTP-range streaming the outer member and
parsing the inner local headers on the fly (`code/ncmapss_stream_extract.py`) — ≈4.5 GB of transfer
instead of 15.76 GB, and the 15.7 GB blob was never stored.
`N-CMAPSS_DS02-006.h5`, 2,450,472,504 B, **sha256 `47971a68b239ecb756833218a95d68ded6eb7e63ee84e86671c8b188de1ca765`**.
⛔ **No CAFE artefact, preprocessing or number was read, used or quoted, in either direction.**
⛔ Classic C-MAPSS was not run, priced or compared (Advisor ruling 2026-08-13).

Verified from the file: `W`(4: alt, Mach, TRA, T2), `X_s`(14), `X_v`(14), `T`(10), `A`(unit, cycle,
Fc, hs); dev 5,263,447 rows / test 1,253,743 rows at 1 Hz; units **train {2,5,10,16,18,20}** with EOL
{75, 89, 82, 63, 71, 66} and **test {11,14,15}** with EOL {59, 76, 67} — identical to the reference's
Table 1, so the split is the canonical one.

### 4.2 The strong reference — **sourced to a primary, open-access table** (P15 survives)
**Arias Chao, Kulkarni, Goebel, Fink, "Fusing Physics-based and Deep Learning Models for
Prognostics", arXiv:2003.00732v2 (2020-10-27) = Reliab. Eng. Syst. Saf. 217:107961 (2022), Table 5**
— same DS02, same three test units, per-sample RMSE at 0.1 Hz:

| model | RMSE [cycles] | s × 10⁵ |
|---|---|---|
| **CNN hybrid** (physics-augmented inputs) | **4.14 ± 0.09** | 0.44 ± 0.02 |
| FNN hybrid | 4.22 ± 0.10 | 0.44 ± 0.01 |
| **CNN data-driven, inputs `[w, x_s]`** — the input-matched reference | **4.95 ± 0.15** | 0.56 ± 0.03 |
| FNN data-driven, inputs `[w, x_s]` | 7.89 ± 0.12 | 1.39 ± 0.04 |

⇒ the scout's relayed, unverified **"DS02 RMSE 5.04"** is retired: the number it was probably an
imprecise memory of is **4.95** (CNN data-driven). ⛔ 5.04 should not be quoted.

### 4.3 ⭐ The missing criterion-2 rows — supplied (they did not exist in the literature)

| baseline | inputs | state bytes | **RMSE [cycles]** | s × 10⁵ |
|---|---|---|---|---|
| **mean-RUL** (constant = training mean) | none | 4 | **19.904** | 10.566 |
| **affine in cycle index** (OLS `RUL = 72.144 − 0.9173·cycle`) | cycle | 8 | **12.393** | 4.353 |
| affine, clipped at 0 | cycle | 8 | 12.393 | 4.353 |
| **mean-train-EOL (74.33) − cycle** — the "RUL is *defined* piecewise-linear in cycle" construction | cycle | 8 | **11.973** | 4.155 |

**Criterion 2 = PASS, decisively.** Best trivial **11.973** vs best strong **4.14** ⇒ **65.4 %
relative reduction in RMSE** (the criterion is >10 %). The scout's warning that the
affine-in-cycle-index baseline would be "dangerously strong by construction" is **not borne out**: it
is 2.9× the strong reference's error. The reason is visible in the data — EOL ranges 59–89 cycles
across nine units, so a pooled affine fit inherits the across-unit EOL spread as irreducible error.
⇒ **N-CMAPSS's criterion-2 cell moves from `UNKNOWN — CANNOT BE SCORED` to `PASS`.**

### 4.4 The criterion-4 tripwire (figure: `fig2_ncmapss_ladder.png`; full ladder in `TABLES.md`)

All arms use only `[W, X_s]`; the `resid` features are the classical **condition-normalised health
index** (each `X_s` channel regressed on `[1, W, W²]` fitted on the healthy dev cycles ≤ 10 and
subtracted) — a deterministic function of `[W, X_s]`, hence **input-matched** to the CNN
data-driven reference. `resid_cycle` additionally uses the cycle index and is therefore an **input
superset**, reported but excluded from `E`.

| representation | best L | dim | bytes | in budget | **best RMSE** | s×10⁵ |
|---|---|---|---|---|---|---|
| **`resid` (raw), k = 25** | **32,768 (at budget)** | 14 | **1,966,080** | yes | **7.988 ← E** | 2.698 |
| `resid` (raw) | 5,000 | 14 | 300,000 | yes | 8.192 | 2.762 |
| **trajectory similarity, 14-d health-index curve, S = 1 cycle, k = 5** | 446 segments | 14 | **26,760** | yes | **8.561** | 2.817 |
| trajectory similarity, S = 20 cycles, 14-d, k = 25 | 332 | 280 | 373,168 | yes | 10.362 | 3.383 |
| trajectory similarity, 1-d health index, S = 20 (Euclid / **DTW**) | 332 | 20 | 27,888 | yes | 11.513 / 11.443 | 3.94 / 3.92 |
| `w_xs` (std) — raw sensors, no condition normalisation | 25,869 (at budget) | 18 | 1,966,044 | yes | 14.010 | 5.30 |
| `resid_cycle` (input superset, excluded from E) | 500 | 15 | 32,000 | yes | 8.416 | 2.816 |

`dies_if`: `RMSE_exemplar ≤ RMSE_best_strong × 1.02`.

| reference | S | E/S | rel. | fires? |
|---|---|---|---|---|
| **CNN hybrid 4.14 — best strong** | 4.14 | **1.929** | **+92.9 %** | **no** |
| FNN hybrid 4.22 | 4.22 | 1.893 | +89.3 % | no |
| CNN data-driven `[w,x_s]` 4.95 (input-matched) | 4.95 | 1.614 | +61.4 % | no |
| ⚠ **FNN data-driven `[w,x_s]` 7.89** | 7.89 | **1.0124** | **+1.24 %** | ⚠ **YES** |

⇒ **`criterion4_cleared_ncmapss = TRUE`** under the rule as written (which takes the **best** strong
reference). ⚠ **But the honest sentence is: a 1.97 MB byte-matched nearest-neighbour store over a
classical health index is statistically indistinguishable from the published FNN data-driven deep
model (7.988 vs 7.89 ± 0.12).** The venue's published criterion-4 hazard is real — it just does not
reach the *frontier* of the venue, because the frontier is held by a temporal (CNN) model and by
physics-augmented inputs. ⛔ Per protocol §7 and C2W10 §1, this caveat is a statement about the
**benchmark**, not about the CLU.

**Ladder shape (the C2W10 lesson-2 check):** on N-CMAPSS the store improves **monotonically** with L
(`resid` raw: 9.494 → 9.466 → 9.099 → 8.575 → 8.192 → **7.988** at L = 250/500/1000/2000/5000/32,768),
the **opposite** of INSECTS. ⇒ the "storing more is harmful" phenomenon is venue-specific, not a law;
the at-budget point is the right point to quote *here* and was the wrong one on INSECTS. Re-pricing
against a revised ≈2 MB budget is a lookup in this ladder.

**Anti-hobbling mattered here too, and more than scaling did:** condition-normalising the sensors
(`resid`) is worth **14.010 → 7.988**, a 43 % error reduction, versus a raw-vs-standardised effect of
only ~0.6 RMSE. A tripwire run without the classical health index would have understated the store
by nearly a factor of two — and would still have cleared, but for the wrong reason.

---

## 5. Bugs / defects found (for `experiment-engineer` and the harness)

1. ⛔ **CAMELS `maurer` forcing product is defective in the shipped v1.2 package.** `tmax == tmin` on
   **>99 % of days in 20/20 basins sampled** (it carries the daily *mean* duplicated). Additionally
   **3/531 files have a malformed header** (`02108000`, `05120500`, `09492400` lose the
   `Year Mnth Day Hr` column names, so `header=3` mis-parses). ⇒ **use `daymet`**, or source
   `maurer_extended` separately (Kratzert's LSTM was trained on `maurer_extended`, which is *not* in
   the CAMELS package). I declared `maurer` a **NOT-RUN** with this evidence.
2. ⛔ **The shipped CAMELS SAC-SMA/Snow-17 model output is an in-sample (full-record-calibration)
   fit**, median NSE ≈ 0.71 on the published validation window vs 0.603 for the true split-sample
   benchmark. **Never use it as a benchmark row.**
3. ⚠ **Loader contracts the harness should pin** (all verified here): 27 static attributes exactly,
   after dropping `huc_02`, `gauge_lat/lon` and Kratzert's `INVALID_ATTR` list; discharge conversion
   `28316846.592·Q·86400/(area·1e6)` with `area` from **line 3** of the forcing header; **`obs < 0`
   dropped before every metric**; 2.36 % of the 1980–2014 discharge record is missing/negative.
4. **My own defect, disclosed and quantified.** The first pass computed squared distance as
   `‖k‖² − 2q·k`, omitting `‖q‖²`. **Rankings — hence every uniform-weight row and every k = 1 row —
   are unaffected** (`‖q‖²` is constant within a query); only the *distance-weighted* rows used a
   wrong weight. I re-ran the decisive in-budget configs with the correction
   (`code/distfix.py`, `arms_distfix.jsonl`): the consumed max moved **0.2531 → 0.2559 (+0.0028)** and
   the 1-day at-budget arm **0.2438 → 0.2520**. No verdict, ordering or conclusion changes.
   ⚠ If the harness reuses this pattern, add the query term.
5. **N-CMAPSS chunking:** a naive `(n_query × L)` float64 distance buffer is 5.2 GB at L = 32,768 with
   20 k queries. Chunk to a byte budget, not a row count.

---

## 6. Limitations and confounds (stated, not minimised)

1. **`S` is a published number, not one we reproduced end-to-end.** We verified it per-basin against
   its own paper's table (12/12 rows exact), but we did not retrain an LSTM; if the C3 harness ever
   trains one on Daymet rather than maurer_extended, `S` should be re-measured. The direction of that
   confound favours the store (Daymet is generally the better forcing), so the margin is conservative.
2. **The store used Daymet, the reference used maurer_extended.** Declared deviation from the
   scorecard-registered plan for the store *and* from the reference's own setup. Forced by defect 5.1.
   Anti-hobbling-safe: the store got the better forcing product.
3. **Over-budget CAMELS ladder points are `[sub3]`** (1-in-3 test days), the pre-registered escape
   hatch. **No verdict number is subsampled.**
4. **k-means condensation was fitted on a 50,000-row subsample** of the ~1.75 M-row training pool
   (memory). A full-pool fit would likely raise the store slightly; the margin is 0.50, so no
   plausible improvement threatens the verdict.
5. **CAMELS exemplar arms have one seed family** (selection seeds 0,1,2 for random arms; k-means is
   deterministic at `random_state=0`). Spread across the three random seeds at the at-budget point is
   small (best-of-3 0.1551 vs the k-means arm's 0.2559 — the *selection rule* dominates the seed).
6. **N-CMAPSS arms are single-run and deterministic** (`rng` seed 0 for exemplar selection); the
   reference carries ±0.09–0.15 over five runs and we quote it with its spread.
7. **The N-CMAPSS trajectory arm is my construction**, not a published implementation; it reproduces
   the *shape* of the classical method (health-index curve matching, DTW and Euclidean) but a
   stronger published variant could exist. Its best (8.561) is close to the kNN arm's best (7.988),
   so the family's ceiling is probably near 8.
8. **Criterion 5 for N-CMAPSS is unchanged at PARTIAL** — this task did not test it, and nothing here
   improves the "no revisit, no re-identification, weak capacity pressure" reading.
9. The **1-day CAMELS variant and the pooled-z scaling are my additions**, not in the scorecard.
   Both were declared in `PREREG.md` §3/§8 before running. Pooled-z turned out to hold the max.

---

## 7. Pre-registration scored honestly (`PREREG.md` §7)

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | B1b loader control passes (p 0.85) | leg (i) **PASS** (≤4e−4); leg (ii) **FAIL vs ±0.01** (0.708 vs 0.603), cause diagnosed | ✘ **partial fail — a finding** (§3.2) |
| P2 | regional/365 d/at-budget = **0.20** (−0.20…0.45) | **0.0561** | ✔ in interval (low) |
| P3 | regional/30 d/at-budget = **0.38** (0.05…0.60) | **0.2559** | ✔ in interval |
| P4 | day-of-year climatology = **0.30** (0.10…0.45) | **0.0111** | ✘ **missed low by 0.29 — outside the interval** |
| P5 | per-basin training-mean flow = **−0.03** (−0.20…0.00) | **−0.0073** | ✔ |
| P6 | local/30 d/L=250 per basin = **0.50** (0.20…0.68) | **0.0957** | ✘ **missed high by 0.40 — outside** |
| P7 | E = 0.38, margin_abs = −0.378 (−0.70…−0.16) | E = **0.2559**, margin **−0.5021** | ✔ in interval |
| P8 | `criterion4_cleared_camels = TRUE` (p 0.88) | **TRUE**, by 25× the threshold | ✔ |
| P9 | shuffle null Δ = **−0.05** (−0.25…+0.02) | **−0.203** | ✔ in interval |
| P10 | per-basin target-standardisation worth ≥ 0.10 median NSE (p 0.8) | **no**: at the best per-basin-z arm std beats raw by only **+0.0219** (0.2393 vs 0.2174); at the winning **pooled-z** arm the **raw** target *wins* by +0.0167 (0.2559 vs 0.2392); same at 1-day (raw 0.2520 vs std 0.2284) | ✘ **failed — a finding** |
| P11 | mean-RUL RMSE **22** (14…32) | **19.904** | ✔ |
| P12 | affine-in-cycle RMSE **13** (8…20) | **12.393** | ✔ |
| P13 | kNN store at budget RMSE **10** (6…18) | **7.988** | ✔ |
| P14 | trajectory similarity RMSE **9** (5…16) | **8.561** | ✔ |
| P15 | a primary open-access DS02 table can be sourced (p 0.60) | **yes** — arXiv:2003.00732v2 Table 5 | ✔ |
| P16 | `criterion4_cleared_ncmapss = TRUE` (p 0.60) | **TRUE** (with the FNN caveat) | ✔ |
| P17 | N-CMAPSS criterion 2 **PASSES** (p 0.70) | **PASS**, 65.4 % relative | ✔ |

**13 survived, 4 failed (P1 partially). The failures are the interesting part:**
- **P4 (climatology 0.30 → 0.011).** I assumed daily discharge is substantially seasonal. It is not:
  a day-of-year climatology explains essentially **nothing** of daily discharge variance (median NSE
  0.0111, with **202 of 447 basins at NSE ≤ 0**). Daily streamflow is **event-driven**. This kills
  the "the store is just learning the seasonal cycle" reading and *strengthens* the venue: there is
  no cheap seasonal shortcut.
- **P6 (local store 0.50 → 0.096).** I assumed a per-basin store would be much stronger than a
  regional one. **It is much weaker**, at 40×–536× the byte budget. Nearest-neighbour matching inside
  one basin's own history does *not* recover its rainfall–runoff response; pooling across 531 basins
  does better. That is the same "one model for all basins beats per-basin calibration" phenomenon the
  EA-LSTM paper reports for learned models, showing up in the classical store as well.
- **P10 (worth ≥0.10 → worth +0.02, and negative for the winning arm).** I expected per-basin target
  standardisation to be the big anti-hobbling lever, because NSE is per-basin scale-free. It is worth
  **+0.022** when the *keys* are also per-basin standardised and **−0.017** when they are
  pooled-standardised: **scale-coupling with the key space, not a universal improvement**. The lever
  that actually mattered was the **exemplar-selection rule** (+0.10), for which I had registered no
  prediction at all — the gap in my own prereg.
- **P1's leg (ii)** is the most valuable failure: it produced reconciliation item 4, which would
  otherwise have become a wrong benchmark row in the harness.

---

## 8. Declared NOT-RUNs (never to be reported as nulls)

CAMELS `maurer` and `nldas` forcing products (maurer defective, §5.1; nldas simply not run) ·
`maurer_extended` (not in the CAMELS package; not sourced) · Caravan (not touched; its licence
remains **single-sourced** per the scout) · any CAMELS arm on the 671-basin superset · CAMELS local
arms at `global` scaling and at L ∈ {500, 2000} · CAMELS 365-day over-budget arms at full temporal
resolution (`[sub3]` only) · DTW on the 14-d N-CMAPSS health index (Euclid only; DTW run on the 1-d
index, where it was *worse*) · N-CMAPSS DS01/DS03–DS08 (only DS02 extracted) · N-CMAPSS `X_v`/`T`
inputs (a leak) · per-unit health-index extrapolation (the scout's third trivial baseline; the two
mandated ones were run) · **any CLU cell of any kind** · **any classic C-MAPSS number** · **anything
CAFE-derived, in either direction**.

---

## 9. Recommended next experiments (for the Hub, priority order)

1. ⛔ **This report is not a venue adoption.** Both tripwires cleared ⇒ the criterion-4 gate is
   discharged and the decision returns to the **Head + Advisor**. CAMELS now has all five criteria
   with primary-source evidence (1 ✔, 2 ✔, 3 ✔ external, 4 ✔ **measured here**, 5 ✔).
2. **If CAMELS is adopted, the loader is a small engineer task and its contracts are already written**
   (§5.3 + `code/build_camels.py` + `staged/MANIFEST.json` sha256s). Budget: hours, not days. The
   staged arrays are 97 MB total and already on disk.
3. **Re-price the ladder, don't re-run it,** when the ≈2 MB Track-A budget's last digit lands. The
   at-budget point moves along a curve that is already measured for both venues.
4. **Retrain the reference on Daymet** (or source `maurer_extended`) before any headline CLU-vs-LSTM
   table, so the block and the reference share a forcing product. Until then the −0.50 margin absorbs
   the confound.
5. **Adopt the regime-resolved decomposition (α, FHV, high-flow NSE, snow-season NSE) as the CLU's
   own diagnostic on this venue.** §3.4 shows the classical store's failure is *localised* in exactly
   the accumulated-storage regimes — that is the sharpest available operationalisation of "memory is
   the difficulty", and it is free once predictions exist.
6. **Standing practice, third confirmation:** an exemplar launder must sweep **(L ladder) × (scaling)
   × (selection rule)**. On INSECTS scaling decided the gate; here the *selection rule* was worth
   +0.10 NSE (65 %) and would have been missed by a random-sample-only arm.
7. **⛔ Do not run classic C-MAPSS, and do not treat N-CMAPSS's FNN tie as an invitation to shop for a
   better reference.** It is a caveat on the fallback, recorded.

## Git footprint
**None.** No branch, no commit, no worktree, no push, no tracked-file edit. HEAD `7fcef50` before and
after; `git status --porcelain` shows 0 modified tracked files. All artefacts under `.claude/`.

## Artefacts (exact paths)
- **`.claude/outputs/c3-trackb-tripwire/VERDICT.json`** — both verdicts, computed arithmetically,
  with the margin against every reference
- `…/PREREG.md` (filed before any harness existed) · `…/TABLES.md` (full ladders) ·
  `…/B1_loader_control.json` · `…/fig1_camels_ladder.png` · `…/fig2_ncmapss_ladder.png`
- `…/arms.jsonl` (2,640 CAMELS rows) · `…/arms_distfix.jsonl` (380) · `…/arms_local.jsonl` (280) ·
  `…/arms_ridge.jsonl` · `…/companions.jsonl` · `…/ncmapss_arms.jsonl` (281 rows)
- `…/deepdive_best2_{ordered,shuffled}.json|.npz`, `…/deepdive_best_{ordered,shuffled,shuffled_both}.json|.npz`
  (per-basin NSE/α/β/FHV/FMS/FLV + regime slices)
- `…/perbasin_*.npz` — per-basin NSE for every arm, every k, every weighting, every target
- `…/code/` — every script verbatim, incl. the NASA nested-zip range extractor and the distance fix
- `…/scratch_venv_freeze.txt` · frozen data: `.claude/data/c3-camels/` (7.2 GB, `staged/MANIFEST.json`),
  `.claude/data/c3-ncmapss/N-CMAPSS_DS02-006.h5` (2.3 GB)

---

# Proposed handover updates (for the Hub)

**§1.6 (experiments) — new rows**

- **`c3-trackb-tripwire`, 2026-08-13, HEAD `7fcef50`, scratch venv (numpy 2.5.2 / sklearn 1.9.0 /
  h5py 3.16.0), CPU-only, ≈2 h 50 m, zero worktrees/commits.** Both Track-B criterion-4 tripwires
  **CLEAR**; ⇒ **the criterion-4 theorem is NOT confirmed a seventh time**, and this is the **first
  venue family to survive it** in this program.
- **CAMELS-US: `criterion4_cleared = TRUE`.** `E = 0.2559` median NSE (447 basins) vs
  `S = 0.7580` (LSTM-with-static-inputs ensemble, Kratzert 2019 HESS 23:5089 **Table 3, read from the
  publisher's xlsx**). **margin_abs = −0.5021, margin_rel = −0.6625**; threshold −0.02 ⇒ **25× clear**.
  Does not fire against **any** of 14 references, incl. the weakest published model
  (VIC-CONUS 0.307, margin −0.0511). E-arm: 30-day window ⊕ 27 statics, **pooled-z** scaling,
  **k-means-condensed** store of **2,761 exemplars = 1,965,832 B** (at budget), k = 3, raw target.
- **CAMELS mechanism, quotable:** the store is a competent climatological predictor and a **failed
  peak/snowmelt predictor** — α-NSE **0.440** (LSTM 0.843), **FHV −56.4 %** (LSTM −15.7 %),
  high-flow (top-2 %) NSE **−1.771**, snow-basin Mar–Jun NSE **0.115**, β-NSE −0.056 (unbiased).
  ⇒ **the residual is concentrated exactly in the unobserved-storage regimes**, which is criterion 3
  measured rather than asserted.
- **CAMELS shuffled-order null (Metro precedent): ordering MATTERS here.** Independent per-row
  permutation of the window's time slots on keys and queries: **0.2530 → 0.0497 (Δ −0.203)** (second
  config −0.169). Contrast Metro, where shuffling *helped* (320.98 → 311.75). The venue is not moot.
- **CAMELS ladder (anti-hobbling, the third confirmation that it decides the number):** raw-feature
  arms top out at **0.0485** vs **0.2559** standardised (**5.3×**); **k-means condensation beats random
  sampling by +0.08…+0.14 NSE at every L** (at budget 0.2559 vs 0.1551). Non-monotone in L again
  (30 d: over-budget L=5,000 → 0.1397; 365 d: L=5,000 → −0.0026). The 365-day/1,852-dim variant is
  dominated everywhere (best 0.0561).
- **CAMELS LOCAL stores are WEAKER than regional:** byte-matched local (5/basin) **−0.0432**; the
  *entire* own-basin record (3,287/basin = **536× budget**) only **0.2173**. Pooling beats per-basin
  memorisation — the classical analogue of the EA-LSTM paper's regional result.
- **CAMELS companions:** per-basin train-mean **−0.0073**; test-mean **0.0000 exactly** (metric-code
  sanity check); **day-of-year climatology 0.0111** at 777,384 B — daily discharge is **event-driven,
  not seasonal**; **1-day persistence 0.4434** but ⛔ **out of protocol (a DIFFERENT TASK)** — that
  structural unavailability is exactly why CAMELS survives the criterion-2 death that killed LTSF and
  Metro, and it is a live leak hazard for the harness.
- **CAMELS post-hoc (unregistered) classical arms:** **per-basin ridge on the 30-day window,
  320,724 B, median NSE 0.4462** — 1.74× the best exemplar store, still **−0.312 below the LSTM**;
  pooled ridge (712 B) 0.3480. The venue clears even under this harsher reading.
- **N-CMAPSS DS02: `criterion4_cleared = TRUE`.** `E = 7.988` RMSE (condition-residual kNN,
  L = 32,768 at exactly 1,966,080 B, k = 25) vs best strong **4.14** ⇒ **+92.9 %**; vs the
  input-matched CNN data-driven **4.95** ⇒ **+61.4 %**. ⚠ **It TIES the weakest published deep model**
  (FNN data-driven **7.89 ± 0.12**, ratio 1.0124 — that single reference *does* trip the 2 % rule).
  Classical trajectory-similarity RUL matching (the venue's published hazard) reaches **8.561** at
  **26,760 B**. Ladder is **monotone increasing in L** here (9.494 → 7.988), the opposite of INSECTS.
- **⭐ N-CMAPSS criterion 2 moves from `UNKNOWN — CANNOT BE SCORED` to `PASS`** — the missing rows,
  supplied: **mean-RUL 19.904** (s×10⁵ 10.566), **affine-in-cycle-index 12.393** (`RUL = 72.144 −
  0.9173·cycle`, s×10⁵ 4.353), **mean-train-EOL − cycle 11.973**. Best trivial 11.973 vs best strong
  4.14 = **65.4 % relative** ⇒ PASS. ⚠ **The scout's "affine is dangerously strong by construction"
  expectation is NOT borne out** (it is 2.9× the strong reference) — because EOL ranges 59–89 cycles
  across the nine units, so a pooled affine fit inherits the across-unit EOL spread.

**§5 (provenance) — new rows**

- ⭐ **CAMELS-US licence RESOLVED: `cc-by-4.0`, `access_right = open`** — DOI 10.5065/D6MW2F4D now
  redirects to **Zenodo record 15529996** (v1.2, 2022-06-24, 15 files, **14,565,262,987 B measured**).
  **The one open blocker on the primary Track-B recommendation is closed; mirroring to CSF3 is
  permitted with attribution.** Both downloaded files reproduce their Zenodo md5 exactly
  (`8e9a466710e8270b58f01d332a87184f`, `f2af624b6277b75b3e410d6a0365591a`).
- **Frozen CAMELS cache** `.claude/data/c3-camels/` (7.2 GB) + consolidated arrays in `staged/`
  (`forcing_daymet.npz` sha256 `2577198f7a1f3852ff0113613c047f790166beea9e97638a155ab5e1a1d5d0c4`,
  `discharge.npz` `f7c3f6976ea9b3753d8d3829d5bebdbdaa8889e0f638917906d9af333fc00a6e`,
  `attributes.npz` `011f3d7c6525c91304d11e3d15427cb17fce3f1388a7a712dd6642752309b626`; MANIFEST.json).
  531-basin list and the **derived-and-verified 447-basin list** in
  `.claude/scratch/c3-trackb-tripwire/ref/`.
- **N-CMAPSS source measured: the NASA PCoE archive is 15,760,443,389 B (15.76 GB), not "several
  GB".** Nested zip; **DS02 extracted by HTTP-range streaming** (≈4.5 GB transferred, 15.7 GB never
  stored): `N-CMAPSS_DS02-006.h5`, 2,450,472,504 B, **sha256
  `47971a68b239ecb756833218a95d68ded6eb7e63ee84e86671c8b188de1ca765`**. Extractor:
  `.claude/outputs/c3-trackb-tripwire/code/ncmapss_stream_extract.py` (reusable for any PCoE item).
- **Kratzert 2019 Table 3 pinned from the publisher's machine-readable table** and reproduced
  **12/12 rows** from the authors' own per-basin metrics on our derived 447-basin set. ⚠ **Correction
  to `c3-benchmark-scout`: "LSTM-with-static ensemble 0.72" is the MEAN; the MEDIAN is 0.758** and it
  is the highest number in the table (above EA-LSTM's 0.742). `S` for the CAMELS `dies_if` is 0.758.
- **N-CMAPSS strong reference pinned:** Arias Chao, Kulkarni, Goebel, Fink, *Fusing Physics-based and
  Deep Learning Models for Prognostics*, **arXiv:2003.00732v2 = Reliab. Eng. Syst. Saf. 217:107961,
  Table 5**: CNN hybrid **4.14 ± 0.09** (s×10⁵ 0.44), FNN hybrid 4.22, **CNN data-driven [w,x_s]
  4.95 ± 0.15**, FNN data-driven 7.89 ± 0.12. ⛔ **The scout's unverified "5.04" is retired**; the
  number is 4.95. Table 1 of the same paper confirms the unit split and EOLs (train 75/89/82/63/71/66,
  test 59/76/67) and that the published metric is per-sample at **0.1 Hz** (the h5 is 1 Hz ⇒
  down-sample ×10 or the RMSE is not comparable).

**§8 (NOT-RUNs / registry) — new entries**

- ⛔ **CAMELS `maurer` (shipped v1.2) is DEFECTIVE and is a declared NOT-RUN**: `tmax == tmin` on
  >99 % of days in 20/20 basins sampled; 3 files (`02108000`, `05120500`, `09492400`) have a
  malformed header. `maurer_extended` (what the reference LSTM used) is **not in the CAMELS package**.
  Harness must use **Daymet** or source `maurer_extended` separately.
- ⛔ **The shipped CAMELS SAC-SMA + Snow-17 model output is an in-sample, full-record-calibration fit
  — NOT a benchmark.** Our recomputation: median NSE **0.7079** (10-seed ensemble; per-seed
  0.7004–0.7069) on the published validation window vs Kratzert's split-sample **0.603**; and it is
  window-independent (0.733 / 0.663 / 0.736 on test / train / 1980-89). **B1 leg (ii) therefore
  FAILED its registered ±0.01 tolerance and the failure is a data-provenance finding, not a loader
  defect** — B1 leg (i) matches CAMELS' own `OBS_RUN` to **max 4.03e−4 mm/day**.
- ⛔ NOT-RUN: `nldas` forcings · Caravan · 671-basin superset · full-resolution 365-day over-budget
  arms (`[sub3]` only) · DTW on the 14-d N-CMAPSS health index · N-CMAPSS DS01/DS03–DS08 ·
  N-CMAPSS `X_v`/`T` as inputs (a leak) · per-unit health-index extrapolation · **any CLU cell** ·
  **any classic C-MAPSS number** · **anything CAFE-derived, in either direction**.

**Registry corrections the Hub should push**

- `negative_results.md` / benchmark registry — **NEW, and it is the first of its kind:**
  **"CAMELS-US is NOT metric-native at matched bytes."** The best byte-matched exemplar store
  (1,965,832 B) reaches median NSE **0.2559** against a published frontier of **0.758** and against a
  *floor* of 0.307; the classical store's error is concentrated in peaks (FHV −56 %) and snowmelt
  (Mar–Jun NSE 0.115), i.e. in the unobserved-storage regimes. **Seven-for-seven becomes six-for-seven.**
- `negative_results.md` — **"N-CMAPSS DS02 passes criterion 2; its trivial baselines are NOT at the
  frontier."** mean-RUL 19.904 / affine-in-cycle 12.393 / mean-EOL−cycle 11.973 vs 4.14 strong. The
  standing worry that RUL's piecewise-linear definition makes the venue trivial is **measured and
  refuted** at DS02.
- **Standing practice, third confirmation:** an exemplar launder must sweep the **L ladder × the
  feature scaling × the exemplar-selection rule**. INSECTS: scaling decided the boolean. CAMELS:
  **selection rule** was worth +0.10 NSE (+65 %) and a random-sample-only arm would have understated
  the launder by two-thirds.
- **Hazard to record for the harness:** on CAMELS, `q(t−1)` scores **0.4434** median NSE — higher
  than every in-protocol classical arm we ran. It is structurally forbidden as an input. **Any
  accidental leak of lagged discharge manufactures a 0.44 "result".**
