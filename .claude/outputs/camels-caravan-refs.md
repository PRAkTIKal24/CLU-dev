# camels-caravan-refs — web-scout report

Task + acceptance criterion: freeze the reference set the **CAMELS-US ungauged (PUB)** tripwire is
judged against, and answer the binary Caravan question. **Status: done**, with two declared partial
gaps (checksums; one Wiley 403) — see §6.

## ⛳ RECONCILIATION LIST (owner needed — protocol §5 corollary, in the first 10 lines)
1. ⭐ **The PUB frontier is NOT 0.69. It is 0.7858** (Li et al. 2025 HESS, Table D2, `LSTM+δHBV` all-forcing
   ensemble). Any doc/plan carrying "PUB SOTA = 0.69 (Kratzert 2019)" is stale by 7 years and by 0.096 NSE.
2. ⛔ **There is no single canonical PUB protocol.** Three mutually incompatible ones are in live use
   (12-fold/1995–2010, 10-fold/1995–1999, 5-fold/1989–1999). Downstream margins must name which one.
3. ⛔ **The gauged artefact does not transfer.** `ref/all_metrics.p` is 447 basins, 1989–1999, gauged.
   Every PUB reference reports on **531** basins. Use `ref/basin_list_531.txt`, not the 447 list.
4. ⚠ **Forcing mismatch, again.** The 2019 PUB reference was trained on **NLDAS-extended** (HydroShare,
   not in the CAMELS package); Li 2025's best arm fuses Daymet+NLDAS+Maurer. Our cache has Daymet only
   (and `maurer` is defective, per `c3-trackb-tripwire` §5.1).
5. ⚠ **`caravan_protocol: NONE`** — this kills any "SOTA on Caravan" framing. Owner needed for whichever
   plan doc assumed Caravan was a venue.

## ⭐ DIAL DECLARATION (protocol §7)
The task file **omitted** the mandatory DIAL DECLARATION block; I supply the honest one and proceed.
- **Dial:** *none — instrument / venue reconnaissance.* No CLU cell, store, byte budget, dividend or
  performance number was computed, quoted or implied. ⛔ Zero store numbers appear below, by construction.
- **Laundering control:** n/a (no measurement made). I am pricing the competition only.
- **Falsifies this spoke:** a published CAMELS-US PUB median NSE **above 0.7858**, or a Caravan protocol
  with a fixed split + defined metric + published baselines.
- **Does NOT falsify:** a *lower* published reference existing (that is expected and is listed); the
  frontier being held by an ensemble rather than a single model.
- **Pre-registration (§5):** **not required and not filed** — the acceptance criterion is a literature
  freeze, not a measured ratio/exponent/slope/law.

---

## 1. Answer first

The CAMELS-US ungauged venue **is auditable but is not a single protocol**: three incompatible k-fold
PUB conventions are in simultaneous use, and **no reference publishes its fold-assignment file**.
That matters far less than it sounds, because **every one of them rotates all folds so that all 531
basins are predicted out-of-sample exactly once** — the reported median is therefore basin-set-complete
and comparable across fold counts, differing only in training-set fraction (4/5, 9/10, 11/12) and,
much more importantly, in **test period and forcing product**. The **current frontier is median
NSE = 0.7858** (`LSTM+δHBV` cross-model, tri-forcing ensemble; Li et al. 2025 HESS), **not** the
0.69 of Kratzert et al. 2019; the weakest credible published reference on the board is **0.53**
(MPR+mHM, Feng et al. 2023). Two references ship per-basin artefacts, so **the frontier and the 2019
baseline are both reproducible without training**. **Caravan has no benchmark protocol at all** —
no split, no metric convention, no baseline numbers, not even in its own dataset paper.

---

## REFERENCE SET (FROZEN)

`venue | reference_name | median_NSE | basin_set | folds | period | metric_convention | per_basin_available (yes/no + path or URL) | reproducible_without_training (yes/no) | citation`

| venue | reference_name | median_NSE | basin_set | folds | period | metric_convention | per_basin_available | reproducible_without_training | citation |
|---|---|---|---|---|---|---|---|---|---|
| camels_us_pub | li2025_LSTM+dHBV_123 ⭐FRONTIER | **0.785833** | 531 (δHBV trained on 671) | 10 random groups; 10 rounds; all basins held out once; 3 seeds | train 1980–1999 / test 1995–1999 (hydro-yr Oct1–Sep30) | per-basin NSE then median over 531; unclipped (−∞,1]; value = mean over 3 seeds; folds concatenated before metric | yes — https://doi.org/10.5281/zenodo.16895228 (`py999_PUB_simulations.npy`, 531 MB, shape 6×3×531×6939) | yes | Li et al. (2025) HESS 29:6829–6861, doi:10.5194/hess-29-6829-2025, Table D2 |
| camels_us_pub | li2025_LSTM_123 | 0.768181 | 531 | as above | as above | as above | yes — same Zenodo | yes | ibid., Table D2 |
| camels_us_pub | li2025_dHBV_123 | 0.757749 | 531 | as above | as above | as above | yes — same Zenodo | yes | ibid., Table D2 |
| camels_us_pub | li2025_LSTM+dHBV_daymet | 0.742270 | 531 | as above | as above | as above | yes — same Zenodo | yes | ibid., Table D2 |
| camels_us_pub | heudorfer2026_EMB-LSTM_ALPHAEARTHplus | 0.721 (abstract says 0.726 — ⚠ unreconciled, §4) | 531 | 5-fold, 80 %/20 % of catchments per fold | train 1999-10-01→2008-09-30; val 1980-10-01→1989-09-30; test 1989-10-01→1999-09-30 | 8 seeds bagged → 100× bootstrap (80 % resample) → **median of 100 NSE realisations**; authors state this "leads to somewhat diminished NSE values" vs. Kratzert et al. 2021 | no — "made public … upon paper publication" | no | Heudorfer, Gupta, Dolich, Loritz (2026) EGUsphere preprint, doi:10.5194/egusphere-2026-1965, Table 3 |
| camels_us_pub | li2025_dHBV_daymet | 0.706809 | 531 | as li2025 | as li2025 | as li2025 | yes — Zenodo 16895228 | yes | Li et al. (2025), Table D2 |
| camels_us_pub | li2025_LSTM_daymet | 0.702636 | 531 | as li2025 | as li2025 | as li2025 | yes — Zenodo 16895228 | yes | Li et al. (2025), Table D2 |
| camels_us_pub | heudorfer2026_baseLSTM_CAMELS27 (their benchmark) | 0.696 | 531 | 5-fold | as heudorfer2026 | as heudorfer2026 | no | no | Heudorfer et al. (2026), Tables 2/3 |
| camels_us_pub | li2025_LSTM_maurer | 0.694156 | 531 | as li2025 | as li2025 | as li2025 | yes — Zenodo 16895228 | yes | Li et al. (2025), Table D2 |
| camels_us_pub | kratzert2019_LSTM_pub_ensemble | 0.69 | 531 | **12-fold**, 10 independent shuffled splits (seeds 300–309) | train 1980-10-01→1995-09-30 / test 1995-10-01→2010-09-30 (15 y) | per-basin NSE (unclipped, `calc_nse`) then median over 531 | yes — `analysis/stats/pub_lstm.csv` (107,960 B) in github.com/kratzert/lstm_for_pub | yes | Kratzert, Klotz, Herrnegger, Sampson, Hochreiter, Nearing (2019) WRR 55(12):11344–11354, doi:10.1029/2019WR026065 |
| camels_us_pub | feng2023_dHBV(βt,γt)_daymet | 0.69 ⚠prov | 531 | 10 random groups, 10 rounds | 1989-10-01→1999-09-30 | median over 531, unclipped | not confirmed (code Zenodo 7091334) | no | Feng, Beck, Lawson, Shen (2023) HESS 27:2357–2373, doi:10.5194/hess-27-2357-2023 |
| camels_us_pub | li2025_dHBV_nldas | 0.670636 | 531 | as li2025 | as li2025 | as li2025 | yes — Zenodo 16895228 | yes | Li et al. (2025), Table D2 |
| camels_us_pub | feng2023_LSTM_daymet | 0.66 ⚠prov | 531 | 10 groups | 1989–1999 | as feng2023 | no | no | Feng et al. (2023) |
| camels_us_pub | feng2023_LSTM_maurer | 0.65 ⚠prov | 531 | 10 groups | 1989–1999 | as feng2023 | no | no | Feng et al. (2023) |
| camels_us_pub | kratzert2019_SACSMA_calibrated | 0.64 | 531 | n/a (per-basin calibrated; the *gauged* comparator) | test 1995-10-01→2010-09-30 | as kratzert2019 | yes — `analysis/stats/benchmark_sacsma_ensemble.csv` (105,006 B) | yes | Kratzert et al. (2019) WRR |
| camels_us_pub | feng2023_dHBV(βt,γt)_maurer | 0.64 ⚠prov | 531 | 10 groups | 1989–1999 | as feng2023 | no | no | Feng et al. (2023) |
| camels_us_pub | heudorfer2026_baseLSTM_dynamiconly | 0.634 | 531 | 5-fold | as heudorfer2026 | as heudorfer2026 | no | no | Heudorfer et al. (2026), Table 2 |
| camels_us_pub | qu2026_ALPHAEARTH | 0.612 ⚠secondary | 531 (stated) | not verified | not verified | not verified | no | no | Qu et al. (2026) arXiv:2601.01558 — **number taken from Heudorfer et al. (2026) l.375, NOT from the primary** |
| camels_us_pub | kratzert2019_NWM_reanalysis | 0.58 | 531 | n/a (NOAA NWM retrospective) | test 1995-10-01→2010-09-30 | as kratzert2019 | yes — `analysis/stats/benchmark_nwm_retrospective.csv` (107,969 B) | yes | Kratzert et al. (2019) WRR |
| camels_us_pub | qu2026_CAMELSattributes | 0.553 ⚠secondary | 531 (stated) | not verified | not verified | not verified | no | no | Qu et al. (2026) arXiv:2601.01558 — via Heudorfer et al. (2026) l.375 |
| ⛔EXCLUDED (not PUB) | feng2023_MPR+mHM — **in-sample / temporal test, NOT ungauged** (Hub, §8 V5) | ~~0.53~~ | 531 | 10 groups | 1989–1999 | as feng2023 | no | no | Feng et al. (2023) HESS 27:2357–2373 |
| caravan | — | NOT-FOUND | — | — | — | — | — | — | no protocol exists (§5) |

caravan_protocol: NONE

⚠ legend: `⚠prov` = single web source (the article HTML), retrieved twice, table image not machine-parsed
→ **provisional**. `⚠secondary` = number reported *about* the paper by another paper; primary not read.
All unmarked numbers are read from the paper's own table/abstract text (§7 provenance).

---

## 3. Item A — the protocol, and whether it is auditable

| question | answer |
|---|---|
| Canonical fold count | **There is no single one.** 12 (Kratzert 2019), 10 (Feng 2023 → inherited by Li 2025), 5 (Heudorfer 2025/2026). |
| Basin→fold assignment | Kratzert 2019: `KFold(n_splits=12, shuffle=True, random_state=seed)` over `data/basin_list.txt` order, **seeds 300–309** (`train_pub.sh`). Feng/Li: "randomly divided all the CAMELS basins into 10 groups" (unspecified seed). Heudorfer: 5-fold, unspecified. |
| Basin set | **531** in all three (the Newman/Kratzert subset). `basin_list.txt` = 4,779 B = 531 × 9 B — arithmetic check passes. ⛔ **Not the 447 set**; 447 is the gauged-benchmark intersection and is irrelevant here. |
| Period | Kratzert 2019: train 1980-10-01→1995-09-30, test **1995-10-01→2010-09-30**. Li 2025: train 1980–1999 (on 9/10 of basins), test **1995–1999**. Heudorfer 2026: train 1999-10-01→2008-09-30, val 1980-10-01→1989-09-30, test **1989-10-01→1999-09-30**. |
| Metric | Plain NSE, **unclipped**, range (−∞, 1] (Li et al. Table C2, verbatim: NSE = 1 − Σ(Oᵢ−Sᵢ)²/Σ(Oᵢ−μ_o)²; "Range −∞ to 1.0"). Per-basin, then **median over the 531**. Test folds are **concatenated across rounds before metrics** ("evaluated after concatenating the test results for all basins"). Kratzert's `calc_nse` likewise does not clip. No NSE≤0 special handling in any reference. Heudorfer 2026 is the exception: median of a 100-sample bootstrap over bagged seeds. |
| **Paper of record** | **Kratzert et al. (2019) WRR** for the *idea and the name*; **but the numerically live protocol is Feng et al. (2023) HESS 27:2357, as used by Li et al. (2025) HESS 29:6829**, which is where the frontier sits. |
| **Is the fold assignment published as a file?** | ⛔ **NO — for none of them.** Kratzert 2019 *generates* `data/kfold_splits_seed{seed}.p` at runtime and `.gitignore`s it. Heudorfer 2026: "All data and code will be made public via a github (code) and zenodo (data) repository **upon paper publication**". Feng/Li: not stated. |

⭐ **Why "no fold file" does *not* make the venue unauditable.** All three protocols rotate every fold,
so each of the 531 basins is predicted exactly once out-of-sample and the median is taken over the
complete 531-basin set. The fold partition therefore affects only *which 4/5–11/12 of basins each model
saw during training* — a second-order effect — not *which basins the median is computed over*. Kratzert 2019
is additionally **exactly reconstructible** (published code + published seeds 300–309 + published basin list;
`sklearn.KFold` shuffling is deterministic given `random_state`). ⚠ Caveat to carry into the paper:
this reconstruction is stable only up to sklearn/numpy RNG behaviour, and it has **not been executed here**.

Verbatim, the protocol sentences (do not paraphrase these):
- Feng et al. (2023): *"For PUB experiments, we randomly divided all the CAMELS basins into 10 groups,
  trained the models on 9 groups, and tested it on the one group held out. By running this experiment for
  10 rounds and changing out the group held out for testing, we can obtain the out-of-sample PUB result
  for all basins."*
- Li et al. (2025), Table 2 footnote c: *"In the PUB and PUR tests, models are run for 10 and 7 rounds,
  respectively, with the group held out for testing changed in each round. The simulation performance was
  evaluated after concatenating the test results for all basins."* Footnote a: *"δHBV training followed
  Feng et al. (2023b) using all 671 CAMELS basins, while LSTM training followed Kratzert et al (2021) using
  the selected 531-basin subset. Test results and performance metrics for all models are reported for the
  531 basins."*
- Heudorfer et al. (2026): *"the models were run temporal as well as spatial out-of-sample, practically
  implemented as a 5-fold cross validation, where in each fold, training happened on the train period of
  80 % (N) of the 531 catchments, and prediction on the test period of the other 20 % (N) of catchments."*
- Kratzert et al. (2019) abstract: *"We trained and tested several LSTMs on 531 basins from the CAMELS data
  set using k-fold validation … This effectively 'ungauged' model was benchmarked over a 15-year validation
  period … The out-of-sample LSTM had higher median Nash-Sutcliffe Efficiencies across the 531 basins (0.69)
  than either the calibrated SAC-SMA (0.64) or the National Water Model (0.58)."*
  (k = 12 and the 1980–1995 / 1995–2010 dates come from the authors' own code: `train_pub.sh` `nsplits=12`,
  `main.py` `GLOBAL_SETTINGS` `train_start 01101980`, `train_end 30091995`, `val_start 01101995`,
  `val_end 30092010` — which independently corroborates the abstract's "15-year validation period".)

---

## 4. Item B — the 2019 → 2026 frontier search, documented as a search

**Frontier = 0.785833** (`LSTM+δHBV`, forcing-ensemble "All"), Li et al. 2025 HESS Table D2. Read directly
off the preprint's own table image (p. 59 of egusphere-2025-483), and consistent with the published
abstract's *"reaching median NSE values of ∼ 0.83 for the temporal test (densely trained scenario),
∼ 0.79 for the ungauged basin test (PUB … ), and ∼ 0.70 for the ungauged region test (PUR)"* — where the
PUR ∼0.70 cross-checks against the in-text *"(LSTM+δHBV)123 (median NSE ∼ 0.701)"*.

⚠ **Adversarial-verification note, recorded because it nearly cost us the number.** A first automated
extraction of Table D2 returned `(LSTM+δHBV)123 = 0.748`. That was **wrong** — it is inconsistent with the
paper's own abstract (∼0.79) and with the PUR cross-check. Reading the table image directly gave **0.785833**.
⛔ Do not consume any Table D2 value that did not come from the table image.

| # | query / route tried | outcome |
|---|---|---|
| 1 | Kratzert 2019 WRR PUB, k-fold, median NSE | found 0.69/0.64/0.58; 3 independent retrievals (S2 API, JKU portal, neuralhydrology post) → **confirmed** |
| 2 | repo forensics `kratzert/lstm_for_pub` (tree, `main.py`, `train_pub.sh`, `data/`, `analysis/stats/`) | k=12, seeds 300–309, dates, **per-basin CSVs found** |
| 3 | "CAMELS 531 PUB k-fold LSTM 2024–2026" | surfaced Heudorfer 2026 (0.721) |
| 4 | ALPHAEARTH / foundation-model hydrology + PUB | Heudorfer 2026; Qu 2026 (0.612/0.553, secondary) |
| 5 | differentiable / δHBV / hybrid-conceptual PUB | Feng 2023 (0.53–0.69); **Li 2025 (0.7858)** |
| 6 | "does anything beat 0.78/0.79/0.80 on CAMELS PUB, 2026" | **nothing found above 0.7858** |
| 7 | transformer / xLSTM / Mamba PUB | Liu et al. 2025 HESS 29:6811 — has a 3-fold *spatial* CV but **reports KGE, R², ubRMSE, not median NSE** ⇒ **NOT USABLE as a row** (NOT-FOUND for NSE, not zero). Akinrele et al. 2026 (arXiv:2606.02791) — **excluded**: NWM v3.0 hourly, 671 basins, *upstream* discharge target, 70/15/15 **temporal** split, metric NNSE ⇒ different task. |
| 8 | mass-conserving LSTM (MC-LSTM, Hoedt et al. ICML 2021) | **no PUB experiment found** — 447-basin gauged benchmark only |
| 9 | continental/global & foundation models (Nearing et al. 2024 Nature; AIFL; RiverMamba; DROP) | ungauged, but **not on the CAMELS-US 531 PUB protocol** ⇒ out of venue, listed here so the absence is visible |
| 10 | MHPI HydroDL benchmark page | used only for lead generation; its tables are *temporal*-test, so **no number was taken from it** |
| 11 | Kratzert et al. 2021 HESS 25:2685 (multi-forcing) | **gauged only** — it supplies the periods Heudorfer reuses, not a PUB row |

**Looked for and NOT found (report as NOT-FOUND, never as absent, never as zero):**
(a) any published CAMELS-US PUB median NSE **> 0.7858**; (b) any **fold-assignment file** for any PUB
reference; (c) a **per-basin artefact for Heudorfer 2026 or Feng 2023**; (d) NSE (as opposed to KGE) for
the Liu et al. 2025 spatial fold; (e) reconciliation of Heudorfer's **abstract 0.726 vs. body/Table 3
0.721** — the paper's Conclusion says *"reaching a new improved PUB performance of NSE_median = 0.721
compared to the previous benchmark of 0.696"*, and the promised "Appendix C" supplementary experiment
**is not present in the preprint**; ⇒ **0.721 is the defensible number; 0.726 is unsourced within its own paper.**

---

## 5. Item D — Caravan: the binary answer

**What it is** (Kratzert et al. 2023, *Scientific Data* 10:61, OA via PMC9887008; Zenodo v1.5):
6,830 catchments at publication (Zenodo changelog: expanded to **>15,000 gauges**, forcings extended to 2023);
sources CAMELS-US 482 · CAMELS-AUS 150 · CAMELS-BR 376 · CAMELS-CL 314 · CAMELS-GB 408 · HYSETS 4,621 ·
LamaH-CE 479; forcing **ERA5-Land** (precip, PET, air/dew-point temperature, shortwave & net-thermal
radiation, surface pressure, wind — daily min/max/mean aggregations); attributes = **HydroATLAS** six groups
+ 10 ERA5-Land climate indices (≈**221** attributes total); period **1981–2020** (1981 = start of ERA5-Land);
licence **CC BY 4.0**, `access_right: open` (Zenodo record 14673536, v1.5, 2025-01-16); download **≈39.3 GB**
(22.1 GB CSV + 17.0 GB netCDF + 4.2 MB code), split across a CSV and a netCDF Zenodo record.

⭐ **The decisive item, stated plainly: THERE IS NO RECOGNISED BENCHMARK PROTOCOL FOR CARAVAN.**
- The dataset paper's **Technical Validation section contains no model at all** — no LSTM, no baseline, no
  train/validation/test split, and **no NSE or KGE number**. It validates *data processing* only
  (HydroATLAS area-fraction aggregation; Pearson r between ERA5-Land and NLDAS/Maurer/Daymet over the 482
  US basins). Verbatim: *"We calculated the correlation (Pearson r) between each pair of forcing data
  products (NLDAS, Maurer, DayMet, ERA5-Land) separately in each basin (n = 482) …"*
- The words **benchmark / baseline / split / leaderboard do not appear** in a methodological sense anywhere
  in the paper (checked over the OA full text).
- The **`kratzert/Caravan` README** likewise contains **no** benchmark, baseline, split or leaderboard section.
- Downstream users consequently invent their own splits (holdout, stratified, climate-zone, gauge-holdout),
  which is exactly the non-comparability signature of a dataset without a protocol.
⇒ **Caravan is a dataset, not a venue. It cannot carry a SOTA claim.** No hedge.
(One near-miss to pre-empt: *Caravan MultiMet*, arXiv:2411.09459, and *GRDC-Caravan*, ESSD 17:4613 (2025),
are **data extensions**, not protocols — they add forcing/forecast products and gauges, no held-out convention.)

---

## 6. Item C — reproducibility, and the re-train price

| reference | reproducible without training? | what exactly is shipped |
|---|---|---|
| **Kratzert 2019 PUB (0.69 / 0.64 / 0.58)** | **YES** | Per-basin **metrics** (nse, bias, stdev, obs5/sim5, obs95/sim95, obsL/simL, obsH/simH, obsFDC/simFDC) for the PUB LSTM, the SAC-SMA benchmark and the NWM benchmark, as three CSVs in `analysis/stats/`. Header verbatim: `,basin,nse,bias,stdev,obs5,sim5,obs95,sim95,obs0,sim0,obsL,simL,obsH,simH,obsFDC,simFDC`. ⚠ **Row count and median NOT verified by me** (the fetch truncated at ~500 rows; 107,960 B / ~203 B per row ⇒ 531 is plausible). **The consuming spoke must download and recompute.** ⛔ Do not quote my fetch tool's "median ≈ 0.707 / 51 basins ≤ 0" — it is an artefact of truncation and is retired here. |
| **Li 2025 (the 0.7858 frontier)** | **YES** | Per-basin **daily simulations**: Zenodo 10.5281/zenodo.16895228, CC BY 4.0, `py999_PUB_simulations.npy` (531 MB, shape **6 × 3 × 531 × 6939** = 6 model/forcing configs × 3 seeds × 531 basins × days), `py999_OBS.npy` (43.4 MB), `py999_ids.npy`. ⇒ the frontier row can be re-derived, not merely cited. |
| Feng 2023 | **NO** (code only: Zenodo 7091334) | no per-basin artefact found |
| Heudorfer 2026 | **NO** | *"All data and code will be made public via a github (code) and zenodo (data) repository upon paper publication."* |
| Qu 2026 | **NO** | not retrieved |

**Re-train price, from the papers' own reported setups** (⇒ consumed directly by the cost spoke):

| reference | trainings needed for the PUB row | per-run cost as reported | total |
|---|---|---|---|
| **Li et al. 2025** | (3 LSTM forcings + 3 δHBV forcings + 1 LSTM^multi) × **10 rounds** × **3 seeds** = **210 model trainings** | verbatim: *"training LSTM and δHBV for temporal testing under a single meteorological forcing dataset takes approximately **5** and **21** hours, respectively, using a single **NVIDIA Tesla V100** GPU"* | ≈ 120 × 5 h + 90 × 21 h = **≈ 2,490 V100-GPU-hours ≈ 104 GPU-days** ⚠ the 5 h/21 h figures are quoted for the *temporal* test (train 1980–1995, all basins); PUB trains 9/10 basins over 1980–1999, so treat as same-order, not exact. The paper's whole study is (2×3+1)×(1+10+7)×3 = **378** trained models. |
| **Kratzert et al. 2019** | 12 folds × 10 seeds = **120 LSTM trainings** | hyperparameters from `main.py` `GLOBAL_SETTINGS`: hidden_size **256**, epochs **30**, batch_size **2000**, seq_length **270**, dropout 0.4, lr 1e-3, initial_forget_gate_bias 5, grad-clip 1.0; `concat_static=True`, forcing = **NLDAS-extended** (HydroShare, ⛔ *not* in the CAMELS package). Params ≈ 4·(256·(D+256)+256)+257 ⇒ **≈ 0.30 M** at D = 32 (5 forcings + 27 attributes) — ⚠ input dim assumed, not read from the paper. | ⛔ **wall-clock and hardware: NOT-FOUND.** `train_pub.sh` round-robins across 3 GPUs; no timing is published. Reported as NOT-FOUND, not estimated. |
| **Heudorfer et al. 2026** | 5 folds × 8 seeds = **40** trainings per configuration (Table 3; 5 seeds elsewhere) | LSTM 1 layer, **256** hidden, dropout 0.4, batch 256, **30 epochs**, Adam lr 1e-3 → 5e-4 @ep10 → 1e-4 @ep20, seq_length **365**, forget-gate bias 3, 15 dynamic features + 27–115 static | ⛔ **hardware / wall-clock: NOT-FOUND** (PyTorch stated; no GPU or timing reported). |

**Item C8 — the on-disk artefacts.** `ref/all_metrics.p` is the **gauged** HESS-2019 suite: 447-basin
intersection, 1989–1999, in-sample-basins. **It has no bearing on PUB** — different basin set, different
task, and (for the 2019 PUB reference) a different decade. The right on-disk list is
`ref/basin_list_531.txt`; `ref/basins_447_derived.json` should be **excluded** from all PUB work.
The PUB-equivalent of `all_metrics.p` **does exist**, in two forms: `lstm_for_pub/analysis/stats/*.csv`
(per-basin metrics, 1995–2010) and Zenodo 16895228 (per-basin daily sims, Li 2025).

---

## 7. Confidence, and the two declared gaps

| claim | status |
|---|---|
| PUB frontier = 0.785833; weakest published reference = 0.53 | **confirmed** — Table D2 read as an image from the authors' own preprint, cross-checked against the published abstract (∼0.79) and the PUR internal consistency (0.701 ↔ ∼0.70) |
| Kratzert 2019 = 0.69 / 0.64 / 0.58, 12-fold, 1980–1995 / 1995–2010, 531 basins | **confirmed** — ≥3 independent retrievals; dates independently corroborated by the authors' code *and* by the abstract's "15-year validation period" |
| Per-basin PUB artefacts exist for Kratzert 2019 and Li 2025 | **confirmed** (file listing + Zenodo record API) |
| `caravan_protocol: NONE` | **confirmed** — 3 independent routes (OA full text, repo README, downstream-usage survey), all negative |
| Feng 2023 rows (0.53–0.69) | **provisional** — one HTML source, fetched twice; the table itself is an image I did not parse |
| Qu 2026 rows (0.612 / 0.553) | **secondary** — reported by Heudorfer 2026, primary not read |
| Heudorfer 2026 = 0.721 | **confirmed from its own Table 3 + Conclusion**; the abstract's **0.726 is unreconciled and should not be quoted** |
| **GAP 1 — checksums** | ⛔ this spoke had **no Bash tool**; no md5/sha256 could be computed and no binary could be copied into `.claude/scratch/`. URLs, byte sizes and harness cache paths are in `.claude/scratch/camels-caravan-refs/SOURCES.md`. **Acceptance clause partially unmet, declared.** |
| **GAP 2 — Kratzert 2019 PDF** | ⛔ Wiley returned **HTTP 403** to `pdfdirect` (despite OpenAlex/S2 reporting CC-BY, publishedVersion). Its numbers therefore rest on the verbatim abstract (3 sources) + the authors' code, **not on its Table 1**. If the Hub needs the in-paper table, a human download is required. |

**What would still prove me wrong, ranked:** (1) a PUB row hiding in a 2026 WRR/JoH paper behind Wiley's
403 (Yu et al. 2024 WRR "Deciphering the Mechanism of Better Predictions of Regional LSTM Models in Ungauged
Basins" and Li et al. 2022 WRR "Regionalization … From Physical Descriptors to Random Vectors" were both
identified as likely PUB papers and **could not be retrieved** — NOT-FOUND, not absent); (2) Feng 2023's
image table differing from the HTML rendering I read; (3) a Caravan protocol defined in a venue I did not
search (I checked NeurIPS D&B, arXiv, HESS/ESSD, the repo).

---

## Git footprint
**None.** Read-only agent: no branch, no commit, no worktree, no tracked-file edit. Files written:
`.claude/outputs/camels-caravan-refs.md`, `.claude/scratch/camels-caravan-refs/SOURCES.md`. No runs,
no training, no CLU number, no store number.

---

## Bibtex-ready refs

```bibtex
@article{kratzert2019pub,
  author  = {Kratzert, F. and Klotz, D. and Herrnegger, M. and Sampson, A. K. and Hochreiter, S. and Nearing, G. S.},
  title   = {Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning},
  journal = {Water Resources Research}, volume = {55}, number = {12}, pages = {11344--11354}, year = {2019},
  doi     = {10.1029/2019WR026065}}

@article{li2025ensembling,
  author  = {Li, Peijun and Song, Yalan and Pan, Ming and Lawson, Kathryn and Shen, Chaopeng},
  title   = {Ensembling differentiable process-based and data-driven models with diverse meteorological
             forcing datasets to advance streamflow simulation},
  journal = {Hydrology and Earth System Sciences}, volume = {29}, number = {23}, pages = {6829--6861},
  year    = {2025}, doi = {10.5194/hess-29-6829-2025}}

@article{feng2023suitability,
  author  = {Feng, Dapeng and Beck, Hylke and Lawson, Kathryn and Shen, Chaopeng},
  title   = {The suitability of differentiable, physics-informed machine learning hydrologic models for
             ungauged regions and climate change impact assessment},
  journal = {Hydrology and Earth System Sciences}, volume = {27}, pages = {2357--2373}, year = {2023},
  doi     = {10.5194/hess-27-2357-2023}}

@misc{heudorfer2026better,
  author = {Heudorfer, Benedikt and Gupta, Hoshin and Dolich, Alexander and Loritz, Ralf},
  title  = {Better data or better architecture? Improving deep-learning-based prediction in ungauged basins},
  year   = {2026}, note = {EGUsphere preprint, discussion started 20 April 2026},
  doi    = {10.5194/egusphere-2026-1965}}

@article{kratzert2023caravan,
  author  = {Kratzert, F. and Nearing, G. and Addor, N. and Erickson, T. and Gauch, M. and Gilon, O. and
             Gudmundsson, L. and Hassidim, A. and Klotz, D. and Nevo, S. and Shalev, G. and Matias, Y.},
  title   = {Caravan -- A global community dataset for large-sample hydrology},
  journal = {Scientific Data}, volume = {10}, number = {1}, pages = {61}, year = {2023},
  doi     = {10.1038/s41597-023-01975-w}}

@misc{li2025zenodosims,
  author = {Li, Peijun and Song, Yalan and Pan, Ming and Lawson, Kathryn and Shen, Chaopeng},
  title  = {Streamflow Simulation Data from Differentiable HBV and LSTM Models Using CAMELS Datasets},
  year   = {2025}, publisher = {Zenodo}, doi = {10.5281/zenodo.16895228}}
```

---

# Proposed handover updates (for the Hub)

**§5 (provenance) — new rows**
- ⭐ **CAMELS-US PUB frontier PINNED: median NSE = 0.785833**, `LSTM+δHBV` tri-forcing ensemble,
  531 basins, 10-fold PUB (all basins held out once), test 1995–1999, mean of 3 seeds —
  **Li et al. (2025) HESS 29:6829–6861, Table D2**. Per-basin daily simulations are **downloadable**
  (Zenodo 10.5281/zenodo.16895228, CC BY 4.0, 531 MB, shape 6×3×531×6939). ⛔ The long-standing
  "PUB SOTA = 0.69" is **superseded by 0.096 NSE**.
- **The 2019 PUB reference is fully reproducible without training:** `kratzert/lstm_for_pub`
  ships per-basin metrics for the PUB LSTM (0.69), the calibrated SAC-SMA (0.64) and the NWM (0.58)
  at `analysis/stats/{pub_lstm,benchmark_sacsma_ensemble,benchmark_nwm_retrospective}.csv`.
  Protocol from the authors' code: **k = 12**, 10 shuffled splits (seeds **300–309**), train
  **1980-10-01→1995-09-30**, test **1995-10-01→2010-09-30**, forcing **NLDAS-extended** (⛔ not in the
  CAMELS package), NSE unclipped, median over 531.
- ⛔ **Three incompatible PUB protocols are live** (12-fold/1995–2010 · 10-fold/1995–1999 ·
  5-fold/1989–1999) and **no reference publishes its fold-assignment file.** Mitigation banked:
  all three rotate every fold ⇒ the median is over the complete 531-basin set in all cases, so the
  statistic is comparable; only training fraction, period and forcing differ. Kratzert 2019's folds are
  additionally reconstructible from published code + seeds.
- ⛔ **`ref/all_metrics.p` and `ref/basins_447_derived.json` DO NOT APPLY to PUB** (447 basins, gauged,
  1989–1999). All PUB references report on **531**. `ref/basin_list_531.txt` is the right list.
- **Weakest published PUB reference = 0.53** (MPR+mHM, Feng et al. 2023) — this is the number the
  kill rule bites against first, not the 0.786.

**§8 (NOT-RUNs / registry) — new entries**
- ⭐ **`caravan_protocol: NONE`.** The Caravan dataset paper (Sci Data 10:61) has **no benchmark protocol,
  no split, no baseline model and no NSE/KGE number** — its Technical Validation is data-processing only.
  The repo README has none either. ⇒ **Caravan cannot carry a SOTA claim** and should be removed from any
  venue shortlist. (Caravan is CC BY 4.0, ≈39.3 GB, 6,830 gauges at publication → >15,000 in v1.5,
  ERA5-Land forcing, 1981–2020, ≈221 HydroATLAS/ERA5 attributes — useful as *data*, never as a *venue*.)
- **NOT-FOUND (never a zero):** fold-assignment files for any PUB reference · per-basin artefacts for
  Heudorfer 2026 and Feng 2023 · NSE (vs. KGE) for Liu et al. 2025's spatial fold · wall-clock/hardware
  for Kratzert 2019 and Heudorfer 2026 · full text of Yu et al. 2024 WRR and Li et al. 2022 WRR
  (Wiley 403) · reconciliation of Heudorfer 2026's abstract 0.726 vs. body 0.721.
- **EXCLUDED after checking (documented, so nobody re-checks):** MC-LSTM (ICML 2021) — gauged 447 only,
  no PUB · Akinrele et al. 2026 arXiv:2606.02791 — NWM-hourly, upstream-discharge target, temporal split,
  NNSE ⇒ different task · Nearing et al. 2024 Nature / AIFL / RiverMamba / DROP — ungauged but not on the
  CAMELS-US 531 PUB protocol · Kratzert et al. 2021 HESS — gauged only.
- ⚠ **Two automated-extraction failures recorded as method lessons:** (i) an automated Table D2 read
  returned 0.748 where the table says **0.785833** — table *images* must be read as images; (ii) an
  automated CSV read of `pub_lstm.csv` returned a truncated 500 rows and an invented median ≈0.707 —
  **retired, do not quote.**

---

## 8. HUB VERIFICATION ADDENDUM — 2026-09-07 (added at review; closes GAP 1 and GAP 2)

The scout had no shell tool, so it could not checksum artefacts or parse the xlsx. The Hub did both.
Artefacts and checksums: `.claude/scratch/hub-refs-verify/`. Every byte size matches the scout's
fetch-reported size exactly.

| artefact | sha256 | bytes |
|---|---|---|
| `pub_lstm.csv` | `8e639bb9805eff180b00f9ec905dbbdfee9d92373edf291ed7a582dfee06b529` | 107,960 |
| `benchmark_sacsma_ensemble.csv` | `b4f4b9927156727b1289b4b3216f964b896769955ee83d375b75a796a2b811aa` | 105,006 |
| `benchmark_nwm_retrospective.csv` | `2c8084e2b607cc462012b3cc1683bee77a02bbc538385c9183377e656eeb0587` | 107,969 |
| `basin_list.txt` | `c3fb069f9fe644ed3102edca456989393ce28ee4864283c1fe37d5db6562bddf` | 4,779 |
| `li2025_tableD2.xlsx` (publisher's machine-readable Table D2) | `0a68ec453d5a128812028a90787374bb0f5c841b0e0c7a678c92c68ccba1d5ed` | 7,936 |

**V1 — Kratzert 2019 PUB reproduced from per-basin data, all 531 rows, no truncation:**

| row | published | Hub-recomputed median | mean | n(NSE ≤ 0) | rows |
|---|---|---|---|---|---|
| PUB LSTM ensemble | 0.69 | **0.6931** | 0.5444 | 27 | 531 |
| SAC-SMA calibrated | 0.64 | **0.6447** | 0.5100 | 12 | 531 |
| NWM retrospective | 0.58 | **0.5766** | 0.3060 | 53 | 531 |

⇒ all three reproduce to the published precision. `basin_list.txt` = **531 entries, 531 unique**.
⭐ **GAP 2 (Wiley 403) is now moot** — the per-basin CSVs reproduce the paper's own table, so the
publisher PDF is not needed. The scout's retired truncation artefact ("median ≈ 0.707, 51 basins ≤ 0")
is confirmed wrong; the true values are 0.6931 and 27.

**V2 — Li 2025 Table D2 machine-parsed from the publisher's xlsx. The frontier is CONFIRMED at 6 dp:**

| model | Daymet | NLDAS | Maurer | Daymet+NLDAS | Daymet+Maurer | NLDAS+Maurer | **All** |
|---|---|---|---|---|---|---|---|
| LSTM | 0.702636 | 0.695496 | 0.694156 | 0.757853 | 0.749151 | 0.753136 | 0.768181 |
| δHBV | 0.706809 | 0.670636 | 0.682998 | 0.748916 | 0.734052 | 0.733955 | 0.757749 |
| **LSTM + δHBV** | **0.742270** | 0.723778 | 0.722020 | 0.780625 | 0.764866 | 0.767761 | **0.785833** ⭐ |

⇒ **frontier = 0.785833, confirmed from the publisher's own machine-readable table.** The scout's
image read was right. The bad automated extraction that returned `0.748` was a **column slip** — it read
`Daymet+Maurer` (0.749151), not `All`. Diagnosis recorded so the failure mode is not repeated.

⭐ **V3 — the Daymet-matched reference, which is the one that actually binds our staged data.**
Our cache is **Daymet-only**. The Daymet-only column is therefore the like-for-like frontier:
**LSTM+δHBV Daymet = 0.742270**, LSTM Daymet = 0.702636, δHBV Daymet = 0.706809. The tri-forcing
0.785833 remains in the reference set (the kill rule is evaluated against every reference) but a
forcing-matched comparison is the defensible headline.

**V4 — the binding threshold is the WEAKEST reference, not the frontier.** The kill rule fires on
proximity to *any* reference. Lowest rows on the board:
`feng2023_MPR+mHM = 0.53` (⚠ still provisional — single HTML source, table image unparsed) and
`kratzert2019_NWM = 0.5766` (**now confirmed** from per-basin data, Hub-recomputed).
⇒ the store must stay **below ≈ 0.51** to clear. That, not 0.7858, is the number the tripwire is really
playing against.

**Still open after this addendum:** Feng 2023's rows remain provisional (worth firming, since 0.53 is the
binding threshold) · Heudorfer 2026 abstract 0.726 vs Table 3 0.721 unreconciled (**quote 0.721**) ·
Qu 2026 rows secondary · Yu et al. 2024 WRR and Li et al. 2022 WRR NOT RETRIEVED behind Wiley.

### V5 — CORRECTION to the frozen set: `feng2023_MPR+mHM = 0.53` is NOT a PUB number. Removed.

Hub pulled the Feng et al. (2023) article HTML (`.claude/scratch/hub-refs-verify/feng2023.html`,
289,211 B, HTTP 200) and read the sentences the scout's provisional row rested on. The paper says,
verbatim:

> *"We also used streamflow simulations from the multiscale parameter regionalization (MPR) scheme
> (Samaniego et al., 2010) applied to the mHM hydrologic model (Rakovec et al., 2019) to represent a
> traditional regionalized hydrologic model, **but only the temporal test** (training and testing in same
> basins but different [periods])"*

and

> *"δ(βt,γt) had a median PUB NSE of 0.64, which is only slightly lower than LSTM (0.65) and
> considerably higher than MPR + mHM (**0.53; this model is in sample – all basins were included in
> training**)"*

⇒ **MPR+mHM 0.53 is an in-sample / temporal-test number on a different task.** It is **struck from the
PUB reference set**. Carrying it would have armed the kill rule with a threshold from the wrong venue —
the single most consequential error that could have been made here, because it was the *binding* row.

**Also firmed by the same retrieval (these were `⚠prov`, now confirmed from the paper's own text):**
`feng2023_dHBV(βt,γt)_maurer = 0.64` · `feng2023_LSTM_maurer = 0.65`. The Daymet Feng rows remain
provisional; they are not binding.

**Revised binding threshold.** Lowest rows genuinely on the CAMELS-US PUB venue:

| rank from bottom | reference | median NSE | status |
|---|---|---|---|
| 1 | `qu2026_CAMELSattributes` | 0.553 | ⚠ **secondary** (via Heudorfer 2026; primary not read) |
| 2 | `kratzert2019_NWM_reanalysis` | **0.5766** | ✅ **confirmed, Hub-recomputed from per-basin data** |
| 3 | `heudorfer2026_baseLSTM_dynamiconly` | 0.634 | from its own Table 2 |

⇒ the store must stay **below ≈ 0.533** (against the secondary 0.553) or **≈ 0.557** (against the lowest
confirmed row). Both are well above the ≈ 0.51 stated in V4, which was computed against the struck row.
**V4's ≈ 0.51 is superseded by this section.**

⭐ **Bonus for the tripwire spoke — use the reference's own attribute set, do not invent one.**
`hess-27-2357-2023-t01.xlsx` (sha256 `1a5eb605d433aebebf4d925f16b68eec3e12d11895d9b1d8796fabc8c034f6b0`,
8,676 B, parsed) is Feng/Li's **35-attribute** static set, listed verbatim: p_mean, pet_mean,
p_seasonality, frac_snow, aridity, high_prec_freq, high_prec_dur, low_prec_freq, low_prec_dur, elev_mean,
slope_mean, area_gages2, frac_forest, lai_max, lai_diff, gvf_max, gvf_diff, dom_land_cover_frac,
dom_land_cover, root_depth_50, soil_depth_pelletier, soil_depth_statgso, soil_porosity,
soil_conductivity, max_water_content, sand_frac, silt_frac, clay_frac, geol_class_1st,
geol_class_1st_frac, geol_class_2nd, geol_class_2nd_frac, carbonate_rocks_frac, geol_porosity,
geol_permeability. ⚠ The gauged run used **27** statics. The donor-basin arm must use **these 35**, or it
is hobbled relative to the method it is impersonating.
