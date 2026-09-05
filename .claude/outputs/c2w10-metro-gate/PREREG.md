# PREREG — c2w10-metro-gate (criterion-4 tripwire on Metro + the drift map)

**Filed 2026-08-11 by `results-analyst`, BEFORE any scored quantity (MAE/RMSE/drift magnitude)
was computed on this stream.** Protocol §5 pre-registration rule: the acceptance criterion is a
**measured relative margin** (`m2_margin_rel`) and a **boolean derived from it**, so predictions are
committed here first.

## What was already computed when this file was written (full disclosure)
`prep.py` had run and produced **structure only** — record counts, timestamp duplication, grid gaps,
value ranges, vocabularies. **No MAE, no RMSE, no accuracy, no drift magnitude, no arm of any kind
had been run.** The structural facts known at filing time:

- 48,204 raw records; **7,629 duplicate rows** over **5,445 duplicated timestamps**; 40,575 unique hours.
- Grid 2012-10-02 09:00 → 2018-09-30 23:00 = **52,551 hours**, of which **11,976 (22.79 %) are missing**.
- **A 7,386-hour hole (2014-08-08 02:00 → 2015-06-11 19:00, ~10 months)** splits the stream into two eras;
  plus 2,192 isolated 1-hour gaps and 11 other gaps ≥ 24 h.
- `holiday` is non-`None` on **53 hours only** (the known UCI quirk: flagged at one hour per holiday, not per day).
- `traffic_volume` ∈ [0, 7280], mean 3290.65, SD 1984.75. `temp` = 0 K on 10 hours; `rain_1h` = 9831.3 on 1 hour.
- After the declared gap-fill (below): **34,848 scored pairs**.

---

## 1. Protocol, declared in advance (design, not outcome)

**Hidden clock.** `date_time` is withheld from every arm's feature vector. It is used by *me* only to
(i) build the hourly grid and (ii) **post-hoc label** the discovered drift regimes against the calendar.
No arm sees a calendar feature except the raw 53-hour `holiday` flag that ships in the 8 columns.

**Canonicalisation (declared rules).**
1. Duplicate `date_time` ⇒ **keep the FIRST row**.
2. Regular hourly grid over [t_first, t_last].
3. **Gap-fill maxgap = 3 h**: runs of ≤ 3 missing hours are filled — `traffic_volume` by **linear
   interpolation**, weather by **forward-fill**. Longer runs stay missing. (Ladder measured at filing
   time for transparency: maxgap 1/2/3/6/12 h ⇒ 29,126 / 33,303 / **34,848** / 37,174 / 38,773 scored
   pairs. maxgap = 3 is the declared primary; 6 is a robustness row if cheap.)
4. **A pair is scored only if its target hour is GENUINELY OBSERVED** (never imputed).

**Horizon.** 24 h ahead. Target index `j`; **origin** `i = j − 24`. Every arm sees only information
at or before `i`, plus the labels of all pairs whose target time is `< j` (plain prequential
test-then-train over pairs ordered by `j`; this is leak-free because pair `j`'s features stop at
`j − 24` and its label is revealed only after prediction).

**The one feature vector, identical for every arm (F3 anti-hobbling, both directions), 32 features:**
- 24 recent traffic lags `y_i, y_{i−1}, …, y_{i−23}` (the "past window" the k-NN attack uses);
- 3 weekly-echo lags `y_{i−143}, y_{i−144}, y_{i−145}` (= target hour −167/−168/−169 h), so that no arm
  is denied the weekly signal that the weekly seasonal-naive baseline exploits;
- 4 weather-at-origin: `temp, rain_1h, snow_1h, clouds_all`;
- 1 `holiday` flag at origin.
⇒ **bytes per exemplar = (32 + 1) × 4 = 132 B** (float32 features + float32 target), deliberately
within 1 B of INSECTS' 133 B so the two gates' byte budgets are directly comparable.
**Budget points:** 0.634 MiB = 665,000 B ⇒ **L = 5,037**; 133,000 B ⇒ **L = 1,007** (floor division).

**Arms.**
- *M1 naïve:* `persistence` (last observed at origin, `ŷ_j = y_i`); `seasonal_naive_24` (`ŷ_j = y_{j−24}`);
  `seasonal_naive_168` (`ŷ_j = y_{j−168} = y_{i−144}`); `mean_so_far`.
- *M2 exemplar/NN launder:* `knn_win` = distance-weighted k-NN (k = 5) over a sliding window of the last
  `L` pairs, run on the ladder `L ∈ {250, 500, 1000, 1007, 2000, 5000, 5037, 14882}` (14,882 = the
  CLU's own 1,966,080 B budget ÷ 132 B), **raw AND causally-standardised**, best (lowest MAE) consumed;
  `knn_sam` = a self-adjusting dual-memory regression analogue of SAM-kNN (declared as OUR adaptation —
  no published SAM-kNN *regressor* is being claimed) at both budgets.
- *M2 strong baselines:* `gbdt` (sklearn `HistGradientBoostingRegressor`, batch-incremental, refit every
  720 pairs on all past pairs); `gru` (torch GRU over the 24-lag sequence + static head, periodic
  minibatch training on a replay buffer); `rls` (exact recursive-least-squares ridge, fully online).
- Robustness row on the strong side: `gbdt+` with `weather_main` one-hot added (guards against the
  mirror-image hobble — an under-fed strong baseline makes criterion 4 fire spuriously).

**Criterion-4 arithmetic (registered, computed mechanically, not by judgement):**
```
m2_margin_rel = (mae_best_exemplar_arm − mae_best_strong_baseline) / mae_best_strong_baseline
criterion4_cleared_metro = (m2_margin_rel > 0.02)
```
i.e. the exemplar/NN store being **within 2 % relative MAE of the best strong baseline, or better,
FIRES criterion 4**. Robustness is reported against *every* strong-baseline reference, as the INSECTS
gate did against every ARF reference.

**M1 pass rule (registered):** the hidden-clock protocol has failed — and the venue is dead — if
persistence is at ceiling, operationalised as **`(mae_persistence − mae_best_strong) / mae_persistence
< 0.05`** (no strong baseline materially beats the naive). This is also M3's headroom quantity.

**Band-exclusion rule (registered BEFORE the map is built — the b = 4 lesson):** a candidate band is
**excluded** if any of
(a) best-strong-arm relative MAE improvement over persistence *within the band* < 5 %;
(b) band SD(y) < 0.25 × global SD(y) (a degenerate low-dispersion band);
(c) fewer than 200 scored pairs in the band.

**Drift-free null (registered construction):** a fixed-seed random permutation of the **pair sequence**
(each pair's feature vector and label kept intact, only the stream order destroyed). This preserves
the joint marginal P(X, y) **exactly** and destroys all regime/ordering structure. Note the property
that makes it a clean instrument: **persistence and both seasonal-naives are functions of a pair's own
features, so their scores are invariant under the shuffle by construction** — any change in a learner's
score is attributable to ordering alone.

---

## 2. Numeric predictions (committed; scored either way in the report)

Derivations are given because the protocol requires *how* the number was derived.

| # | quantity | prediction | 80 % interval | derivation |
|---|---|---|---|---|
| **P1** | `mae_persistence` (h = 24) | **700** | 500–950 | Same-hour-yesterday. ~5/7 of transitions are weekday→weekday (same-hour day-to-day spread ≈ 400 at a 3,290 mean), ~2/7 straddle a weekend boundary where the same-hour level moves by ≈ 1,500 (peak ≈ 6,000 weekday vs ≈ 3,500 weekend). 5/7·400 + 2/7·1500 ≈ 715 |
| **P2** | `rmse_persistence` | **1,150** | 800–1,500 | error distribution is heavy-tailed at the weekend boundaries ⇒ RMSE/MAE ≈ 1.6 |
| **P3** | `mae_seasonal_naive_168` | **520** | 380–700 | removes the day-type mismatch entirely; residual = week-to-week noise + seasonal drift, ≈ the weekday-only term of P1 plus holiday/weather misses |
| **P4** | `mae_seasonal_naive_24` **exactly equals** `mae_persistence` | **identical to 1e-9** | — | ⚠ **registered degeneracy**: at h = 24 the period-24 seasonal-naive *is* last-observed-at-origin. The task's two M1 baselines collapse into one; `seasonal_naive_168` is supplied as the non-degenerate second naive |
| **P5** | `mae_best_strong` (GBDT or GRU) | **420** | 320–550 | with the weekly echo + 24-lag window + weather, a boosted tree should recover most of the day-type/weather structure; ≈ 40 % better than persistence, ≈ 20 % better than the weekly naive |
| **P6** | `mae_knn_win` at 0.634 MiB (L = 5,037, best of raw/std) | **470** | 370–620 | analog forecasting on a 24-lag window is genuinely strong on diurnal traffic, but a 5-neighbour average is high-variance and mixes 4 weather scales into one Euclidean metric |
| **P7** | `m2_margin_rel` | **+0.10** | −0.05 … +0.30 | P6/P5 − 1 |
| **P8** | `criterion4_cleared_metro = true` | **p = 0.60** | — | genuinely uncertain. For clearing: a tight 2 %-relative bar; covariate-using GBDT usually beats a mixed-metric k-NN by 5–20 % on tabular regression. For firing: the scout's ⚠, and the INSECTS finding that short-window exemplar stores are astonishingly strong under drift |
| **P9** | `m3_headroom_rel` = (mae_pers − mae_best_strong)/mae_pers | **0.40** | 0.20–0.55 | P1/P5 |
| **P10** | k-NN window ladder is **non-monotone with an interior optimum**, as on INSECTS, with the best `L` in **[1000, 5037]** | p = 0.55 that the optimum is interior; p = 0.75 that the CLU-budget point L = 14,882 is **not** the best | — | INSECTS' finding was recency-as-hidden-regime-variable; Metro's regime clock is the calendar, whose period (1 week = 168 pairs) is *much shorter* than any budget point, so I expect the recency penalty to be far weaker here than on INSECTS and larger stores to be nearer-neutral |
| **P11** | drift map: number of retained regimes `K` | **4** | 3–6 | with week-long windows, hour-of-day and day-type average out; the surviving between-window axes are **season** (dominant), **weather**, and **the 2014–15 era break** |
| **P12** | the discovered top drift axis is **seasonal/annual**, not weekly | p = 0.70 | — | as P11 |
| **P13** | drift-free null: `mae_persistence` shuffled − unshuffled | **0.000** | exact | by construction (P4's mechanism) — this is the null's own positive control |
| **P14** | drift-free null: `knn_win(L = 1,007)` degrades by **≥ 5 % relative** under the shuffle | p = 0.65 | — | if a short-window store loses nothing when the order is destroyed, the stream has no exploitable regime structure at that scale — which would itself be the headline finding |
| **P15** | `gbdt+` (weather_main one-hot) improves on `gbdt` by | **< 2 % relative** | — | `weather_main` is largely redundant with `clouds_all`/`rain_1h` |

## 3. What would make me report failure
- M1 fails (headroom < 5 %) ⇒ **venue dead**, reported as such, no M2 verdict quoted as meaningful.
- Any NaN/divergence in an arm ⇒ reported with the evidence, arm marked NOT-RUN, never silently dropped.
- If the k-NN attack wins, **STOP** — no third venue is improvised (task ⛔⛔).
