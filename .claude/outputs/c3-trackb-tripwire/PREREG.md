# PREREG — `c3-trackb-tripwire` (criterion-4 tripwires: CAMELS-US · N-CMAPSS DS02)

**Filed:** 2026-08-13, **before** any tripwire/baseline harness was written or run.
**Agent:** results-analyst. **Repo HEAD at filing:** `7fcef50`, `git status --porcelain` empty.
**Zero worktrees, zero branches, zero tracked-code edits.**
Protocol §5 pre-registration rule (the acceptance criterion is a measured comparison against a registered threshold).

> ⛔ **A fired tripwire falsifies the BENCHMARK's admissibility, not the CLU.** No CLU cell, store,
> controller or performance number of any kind is run, computed or claimed in this task.

---

## 0. Disclosure of everything computed BEFORE this prereg was written

Per protocol §5 ("if a number is computed before the prereg is written — e.g. a deterministic
baseline with no free parameter — disclose it in the prereg", the C2W10 standard):

**D1 — CAMELS licence (the Head-facing blocker) is RESOLVED, and it is open.** `dx.doi.org/10.5065/D6MW2F4D`
(the citation-of-record on NCAR/RAL's own CAMELS page) now redirects to **Zenodo record 15529996**.
Zenodo REST API `metadata.license.id = "cc-by-4.0"`, `access_right = "open"`, version 1.2,
publication_date 2022-06-24. 15 files, **14,565,262,987 B total (14.57 GB)**, each with an
upstream md5. The RAL page also asserts UCAR's Terms of Use apply and labels the record
"License Type: License Required"; the Zenodo record of the same DOI carries the explicit
SPDX-style identifier `cc-by-4.0`. ⇒ **Not blocked**; mirroring is permitted with attribution.
(The scout's secondary-sourced "15 GB compressed" figure is corroborated by measurement at
14.57 GB; the "130 GB uncompressed" figure remains **NOT VERIFIED** and is not quoted.)

**D2 — Kratzert et al. 2019 Table 3, read from the publisher's own machine-readable table**
(`hess-23-5089-2019-t03.xlsx`, downloaded from hess.copernicus.org). Deterministic, published,
no free parameter. This supersedes the scout's single-retrieval `~` rows.

| model | NSE mean | **NSE median** | #basins NSE ≤ 0 |
|---|---|---|---|
| EA-LSTM single | 0.674 (±0.006) | 0.714 (±0.004) | 2 |
| **EA-LSTM ensemble** | 0.705 | **0.742** | 1 |
| LSTM single | 0.685 (±0.015) | 0.731 (±0.002) | 1 |
| **LSTM (static inputs) ensemble** | 0.718 | **0.758** | 1 |
| SAC-SMA | 0.564 | 0.603 | 13 |
| VIC (basin) | 0.518 | 0.551 | 10 |
| VIC (CONUS) | 0.167 | 0.307 | 41 |
| mHM (basin) | 0.627 | 0.666 | 7 |
| mHM (CONUS) | 0.442 | 0.527 | 29 |
| HBV (lower, 1000 uncal.) | 0.237 | 0.416 | 35 |
| HBV (upper, 100 cal.) | 0.631 | 0.676 | 9 |
| FUSE (900 / 902 / 904) | 0.587 / 0.611 / 0.582 | 0.639 / 0.650 / 0.622 | 12 / 10 / 9 |

⚠ **Scout correction (reconciliation item):** the scout reported "LSTM-with-static-inputs ensemble
**0.72**". That is the **mean** (0.718). The **median** — the metric the `dies_if` rule is written
in — is **0.758**, and it is the *highest* number in Table 3, above EA-LSTM's 0.742.

**D3 — the 447-basin list and per-basin reference NSEs, recovered and verified.**
`kratzert/ealstm_regional_modeling` (Apache-2.0) ships `notebooks/all_metrics.p`, which contains
**per-basin** NSE for every benchmark model and for each LSTM/EA-LSTM seed + ensemble. The
intersection of the ten benchmark models' basin sets is **exactly 447 basins** (SAC-SMA 670,
VIC-basin 670, VIC-CONUS 531, mHM-basin 492, mHM-CONUS 492, HBV-lb 671, HBV-ub 671, FUSE-900 576,
FUSE-902 553, FUSE-904 576). Recomputing Table 3's medians on that intersection reproduces the
published table **to the last printed digit for all 12 rows** (e.g. SAC-SMA 0.6028→0.603, mean
0.5639→0.564, n≤0 = 13→13; EA-LSTM ens 0.7423→0.742; LSTM ens 0.7580→0.758; HBV-ub 0.6756→0.676).
⇒ the 447 set and the metric convention are pinned, and **the strong reference is available
per-basin**, so no LSTM has to be trained by us (⛔ no training run, per task).
Basin list written to `.claude/scratch/c3-trackb-tripwire/ref/basins_447_derived.json`.

**D4 — metric + loader conventions adopted verbatim from the reference implementation**
(`papercode/metrics.py::calc_nse`, `papercode/datautils.py`, `papercode/evalutils.py`):
NSE = 1 − Σ(sim−obs)² / Σ(obs−mean(obs))²; **days with `obs < 0` (the −999 missing flag) are
dropped from both arrays before the metric**; forcings = `basin_mean_forcing/maurer_extended/**/*_forcing_leap.txt`,
columns `prcp(mm/day), srad(W/m2), tmax(C), tmin(C), vp(Pa)`; discharge =
`usgs_streamflow/**/*_streamflow_qc.txt` converted with `28316846.592 * QObs * 86400 / (area*1e6)`,
area read from line 3 of the forcing header.

**D5 — N-CMAPSS source located and its size measured.** NASA PCoE item **17. Turbofan Engine
Degradation Simulation-2**, `https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`,
**Content-Length = 15,760,443,389 B (15.76 GB)** — the scout's "several GB" is **wrong by ~3×**.
It is a nested zip (outer: one deflate member `data_set.zip`, 15,814,385,805 B inflated; inner:
`data_set/N-CMAPSS_DS0x-*.h5`, each deflate-compressed; DS01 = 2,873,351,432 B inflated).
⇒ DS02 is being extracted by HTTP-range streaming so we never store 15.76 GB.
⛔ No CAFE-derived artefact is used, read, or quoted in either direction (task §2 embargo).

**D6 — nothing else has been computed.** No exemplar arm, no kNN, no RUL baseline, no NSE of ours.

---

## 1. The `dies_if` rules, quoted from the scorecard (never re-derived)

**CAMELS-US (row 1), verbatim:**
> "Let S = median NSE over the test period of the best STRONG reference (our re-trained
> LSTM/EA-LSTM; sanity anchor 0.74 from Kratzert 2019 Table 3) and E = median NSE of the best
> exemplar arm at or under the byte budget, maximised over {k, L, raw|std, regional|local}. Compute
> margin_abs = E − S and margin_rel = (E − S)/S. THE VENUE DIES IF margin_abs ≥ −0.02 …, OR IF
> margin_rel ≥ −0.02. Report BOTH, and report the margin against EVERY strong reference available…
> Additionally report the FLOW-REGIME-RESOLVED margin (high-flow / low-flow / snowmelt-recession
> slices)."

**Registered substitution (declared, with reason):** the scorecard's S is "our re-trained
LSTM/EA-LSTM". ⛔ This task is forbidden a training run. We substitute the **published per-basin
LSTM/EA-LSTM ensemble results themselves** (D3), which is *stronger* than a re-trained proxy: it is
the venue's own frontier number, per basin, on the same 447 basins and the same period.
**S_primary = LSTM (static inputs) ensemble, median NSE = 0.7580 on the 447 set.** Every other
reference in D2 is reported as its own row.

**N-CMAPSS DS02 (row 2), verbatim:**
> "RMSE_exemplar_at_budget ≤ RMSE_best_strong × 1.02 (i.e. the classical store is within 2 %
> relative RMSE of the best learned model, the Metro relative convention). Also report the NASA
> s-score under the same rule, and report the margin against EVERY strong reference."

⚠ **Registered contingency:** the scout could not reach a primary table for the DS02 "5.04 RMSE"
claim and it is embargoed as NOT-VERIFIED. If no primary open-access DS02 table can be sourced,
**criterion 4 for N-CMAPSS is scored `NOT_SCOREABLE — reference gap`** and ⛔ we do **not**
substitute a weaker reference (task §4 stop condition). The criterion-2 rows are computed either way.

## 2. Byte budget and the store-size ladder (from the scorecard, §3 of the task)

Budget **B = 1,966,080 B**. ⚠ A ≈2 MB Track-A budget was ruled 2026-08-13 with its last digit
pending; the ladder is reported so that re-pricing is a lookup, not a re-run.

| representation | dims | B/exemplar | exemplars at budget |
|---|---|---|---|
| CAMELS 365-day ⊕ 27 statics | 1,852 | (1852+1)·4 = 7,412 | **265** (1,964,180 B used, 1,900 slack) |
| CAMELS 30-day ⊕ 27 statics | 177 | (177+1)·4 = 712 | **2,761** (1,965,832 B, 248 slack) |
| CAMELS 1-day ⊕ 27 statics (declared extra) | 32 | (32+1)·4 = 132 | **14,894** |
| CAMELS local 30-day, statics dropped (constant in-basin) | 150 | (150+1)·4 = 604 | **3,255** (≈6/basin) |
| CAMELS day-of-year climatology store (declared extra) | — | 531·366·4 | **777,384 B — inside budget** |
| N-CMAPSS 20-frame × 24-dim window | 480 | 1,920 + 4 = 1,924 | **1,021** |
| N-CMAPSS 1-frame (24-dim) | 24 | 96 + 4 = 100 | **19,660** |

**L ladder, run for every arm:** `{250, 500, 1000, 2000, 5000, at-budget}` (C2W10 lesson 2: on
INSECTS a single L was wrong by 19 points; the *at-budget* point was the **worst** arm).
**Ladder points above the budget are reported but are NOT admissible for the verdict**; the
registered `E` is the max over **in-budget** arms. The over-budget max is reported separately as an
"unbounded-store" upper bound (if even an unbounded store loses, the finding is stronger).

## 3. Anti-hobbling (C2W10 lesson 1 — this decided the last gate; MAX is consumed)

Every CAMELS exemplar arm is run in the cross-product and the **max** is consumed:
1. **Feature scaling:** `raw` · `per-basin z-score` · `pooled(global) z-score`. ⛔ All statistics
   from the **training period only** (1999-10-01…2008-09-30). Test-period statistics are never
   touched — the venue's own leak-free convention (note: in CAMELS the *test* window
   1989-10-01…1999-09-30 **precedes** the train window, so "causal" here means
   *training-period-only*, not *past-only*; this is declared, not silently assumed).
2. **Target space:** `raw mm/day` · `per-basin standardised` (store (q−μ_b)/σ_b with μ,σ from the
   basin's **training** discharge; de-normalise with the query basin's μ,σ). The second is
   admissible because the protocol is a *temporal* split — the learned reference has also seen
   every basin's training discharge — and it is expected to be much stronger, because NSE is
   per-basin scale-free.
3. **Neighbour weighting:** distance-weighted (`1/max(d,1e-9)`, the C2W10 shim convention) ·
   uniform mean. `k ∈ {1,3,5,10,25}`.
4. **Exemplar selection:** uniform random sample (seeds 0,1,2) · **k-means condensation**
   (MiniBatchKMeans, L centroids on the scaled training queries, target = cluster-mean), which is
   strictly stronger at small L and is *the* anti-hobbling arm for a 265-exemplar budget.
5. **Store arms:** **REGIONAL** (pooled over 531 basins; registered primary, byte-matched) and
   **LOCAL** (same-basin only). ⚠ **A local store is only byte-matched if the whole 531-basin system
   fits B** — at 365-day resolution that is <1 exemplar/basin, so the LOCAL ladder points at
   L/basin ∈ {250…5000} are **531× over budget** and are reported as an *upper bound*, exactly as
   C2W10 reported ARF (14.35× the budget) as not byte-matched. The byte-matched local arms are the
   30-day (5/basin) and statics-dropped (6/basin) rows in §2.
6. **Forcing product:** primary = **Maurer-extended** (what the reference LSTM was trained on;
   fidelity to S beats fidelity to the scorecard's "Daymet" here — declared deviation). **Daymet is
   run at the verdict-deciding at-budget points and the max is consumed.**

## 4. Mandatory companion rows (every CAMELS table)

`per-basin mean flow` (training-period mean → NSE slightly < 0; the test-period mean gives NSE = 0
**by definition** and both are printed) · `per-basin day-of-year climatology` (training period,
**777,384 B — inside budget**, i.e. a genuinely byte-matched classical store) · `1-day persistence`
reported **separately and explicitly labelled a DIFFERENT TASK** (persistence needs discharge as an
input, which the simulation protocol forbids — quoting it in-protocol would repeat the Metro leak error).

## 5. Positive controls (must pass BEFORE any tripwire number is quoted)

- **B1a (already discharged, D3):** reproduce Kratzert Table 3 from the published per-basin values on
  the derived 447 set. ✔ exact on 12/12 rows.
- **B1b (the real loader control):** from *our* loader, recompute (i) the observed discharge series in
  mm/day and check it against `OBS_RUN` in CAMELS' own shipped SAC-SMA model-output files, and
  (ii) median NSE of shipped **SAC-SMA + Snow-17** over the 447 basins in the test window.
  **Pass tolerance: |Δ median NSE| ≤ 0.01 vs 0.6028, and ≥ 90 % of basins within |Δ| ≤ 0.01.**
  Any mismatch ⇒ defective loader/split ⇒ hard stop.
- **N-CMAPSS:** unit split must be exactly train {2,5,10,16,18,20} / test {11,14,15}; RUL must be
  non-increasing within a unit; the file's sha256 is reported.

## 6. Extra null leg (Metro precedent, mandatory for CAMELS)

Independent per-sample random permutation of the 365 time slots (all 5 channels permuted jointly,
seed 0), applied to **queries and stored keys alike**, then re-run. If shuffling does not hurt the
store, the window carries no exploitable *ordering* information at these budgets (on Metro,
shuffling **helped**: 320.98 → 311.75).

## 7. POINT PREDICTIONS (commit to numbers; a survivor is evidence, a failure is a finding)

| # | prediction | point | 80 % interval | prior |
|---|---|---|---|---|
| P1 | B1b loader control passes | pass | — | p = 0.85 |
| P2 | CAMELS **regional / 365-day / at-budget L=265**, best over all §3 axes — median NSE | **0.20** | −0.20 … 0.45 | |
| P3 | CAMELS **regional / 30-day / at-budget L=2761**, best over all axes — median NSE | **0.38** | 0.05 … 0.60 | |
| P4 | CAMELS **day-of-year climatology** (byte-matched, 777 kB) — median NSE | **0.30** | 0.10 … 0.45 | |
| P5 | CAMELS **per-basin training-period mean flow** — median NSE | **−0.03** | −0.20 … 0.00 | |
| P6 | CAMELS **local / 30-day / L=250 per basin** (531× over budget) — median NSE | **0.50** | 0.20 … 0.68 | |
| P7 | **E** = best **in-budget** exemplar arm, and `margin_abs = E − 0.7580` | **E = 0.38, margin −0.378** | margin −0.70 … −0.16 | |
| P8 | **`criterion4_cleared_camels = TRUE`** (the store does NOT get within 0.02 of the LSTM) | TRUE | — | **p = 0.88** |
| P9 | shuffled-order null: Δ median NSE (shuffled − ordered), best 365-day arm | **−0.05** | −0.25 … +0.02 | |
| P10 | the per-basin *target-standardised* arm beats the raw-target arm by ≥ 0.10 median NSE | yes | — | p = 0.8 |
| P11 | N-CMAPSS **mean-RUL** RMSE (test units, per-sample) | **22 cycles** | 14 … 32 | |
| P12 | N-CMAPSS **affine-in-cycle-index** RMSE | **13 cycles** | 8 … 20 | |
| P13 | N-CMAPSS **k-NN store at budget** (L=1021, 20-frame W⊕X_s windows) RMSE | **10 cycles** | 6 … 18 | |
| P14 | N-CMAPSS **trajectory-similarity** (Euclid/DTW on a health index) RMSE | **9 cycles** | 5 … 16 | |
| P15 | a **primary, open-access** DS02 RMSE table can be sourced | yes | — | p = 0.60 |
| P16 | **`criterion4_cleared_ncmapss = TRUE`** (conditional on P15) | TRUE | — | p = 0.60 |
| P17 | N-CMAPSS **criterion 2 PASSES** (best strong ref beats best trivial by > 10 % relative) | PASS | — | p = 0.70 |

**Why P8 is high and P16 is not:** CAMELS' target is a function of an **unobserved accumulated
state** (soil moisture, groundwater, snowpack); two identical forcing windows with different
antecedent storage produce different discharge and the 27 statics only partially disambiguate — if
that non-identifiability is material, no metric over the observable window can be the ceiling.
N-CMAPSS has the opposite structure: **similarity-based RUL estimation is a published, competitive
classical method on the C-MAPSS family**, i.e. the venue arrives with a pre-existing reason to be
metric-native. ⛔ Neither statement is a result; both are the registered hypotheses.

## 8. Protocol choices I had freedom over (registered, so they cannot be tuned later)

1. Comparison set = **447 basins** (the set Table 3 is defined on) for every margin; the 531-basin
   set is reported alongside for the arms we compute ourselves.
2. Test window **1989-10-01 … 1999-09-30**; store/statistics window **1999-10-01 … 2008-09-30**.
   Windows that would reach outside the record are dropped (no zero-padding).
3. NSE per basin then **median across basins**; means and the NSE≤0 count reported too.
4. `float32` blocked distances via `‖q‖²+‖k‖²−2q·k`; ties → lowest index (C2W10 shim convention).
5. **Wall-clock escape hatch, declared in advance:** every **in-budget** arm and every verdict-
   deciding number is computed on the **full** test period. Purely characterising **over-budget**
   ladder points may be computed on a **1-in-3 day subsample**, and any such row is flagged `[sub3]`
   in the table. ⛔ No verdict number may come from a subsampled row.
6. Flow-regime slices: per basin, test days split by observed-discharge quantile —
   **high-flow = top 2 %** (the venue's own FHV convention), **low-flow = bottom 30 %** (FLV
   convention), **mid = the rest**; plus a **snow slice** = basins with `frac_snow > 0.3`
   (CAMELS `camels_clim.txt`), evaluated on Mar–Jun days.
7. N-CMAPSS inputs = `W` (4 flight-condition descriptors) ⊕ `X_s` (measured sensors) only.
   ⛔ `X_v` (virtual/unobservable sensors) and `T` (health parameters) are **not** inputs — using
   them would be a leak. Actual channel counts are measured from the file and the §2 byte
   arithmetic re-priced if they differ from the scorecard's assumed 24.
8. N-CMAPSS metric = per-sample RMSE over all test-unit rows (venue convention), plus the NASA
   s-score, plus a per-cycle-aggregated RMSE.
9. Seeds: exemplar-selection seeds {0,1,2} for the random-sample arms; k-means `random_state=0`,
   `n_init=1`. Deterministic arms carry no seed and say so.

---

*Filed before the harness existed. Scored honestly in `.claude/outputs/c3-trackb-tripwire.md` §7.*
