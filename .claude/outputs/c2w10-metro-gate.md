# c2w10-metro-gate — results-analyst report

Task + acceptance criterion: run the criterion-4 tripwire on Metro Interstate under the hidden-clock 24-h protocol **before any Metro cell**, and publish the drift map PREREG-C2W10 §2 makes a precondition of any retention claim; write `METRO-GATE.json` with `criterion4_cleared_metro` computed arithmetically. **Status: done.**

## ⛳ RECONCILIATION LIST (owner needed — protocol §5 corollary, first 10 lines)
1. **`criterion4_cleared_metro = FALSE`. The tripwire has fired on Metro too**, at `m2_margin_rel = −0.0615`: a **0.634 MiB** exemplar store scores **314.575 MAE** against the best tuned strong baseline's **335.203**. It fires against **all 9** strong references. ⛔ **Two fired tripwires in one wave — the Hub's/Head's call, not mine. I did not improvise a third venue.**
2. **Metro ALSO fails criterion 2 independently.** The best strong baseline beats the **one-line weekly seasonal-naive** by **2.17 %** relative MAE (335.203 vs 342.651). That is the ELEC2 pathology one level up and it is a separate kill.
3. ⛔ **My own PREREG §1 protocol claim was WRONG and is corrected here:** plain prequential test-then-train at a 24-h horizon **leaks up to 23 h of future traffic** to any continuously-updated learner. A **24-h label embargo** is now implemented; **every primary number in this report is embargoed**. Any harness that consumes this stream must implement `A(t)`. The leak was worth **+10.9 %** to a 250-exemplar k-NN and **−0.3 %** to GBDT — asymmetric, in the direction of firing.
4. **PREREG-C2W10 §2's "seasonal-naive (t−24 h)" is DEGENERATE at a 24-h horizon** — it is bit-identical to persistence (max |diff| = 0.0). The non-degenerate naive is **t−168 h**, and it is the one that matters (item 2).
5. **The drift map is published and CLEAN**: 4 maps, up to **491 regime visits**, **zero excluded bands** (unlike INSECTS' `b=4`). It is usable if the Hub ever wants it — but see item 6.
6. ⛔ **The re-constructed drift-free null says the retention claim has no room on Metro:** destroying the stream order **does not hurt and slightly HELPS** the exemplar store at ≥ 5,037 exemplars (320.98 ordered → **311.75 shuffled**, −2.9 %). At the byte budgets in play, Metro's temporal ordering carries **no exploitable information**.
7. `1966080 / 132 = 14,894` exemplars at the CLU d=12 budget (PREREG.md §1 said 14,882 — my arithmetic slip, corrected).
8. Minor: `holiday` fires on **53 hours in 6 years** (UCI quirk, one flagged hour per holiday); `rls` with a forgetting factor **diverged** (RMSE 2.3e5) and is excluded.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** none — **benchmark-admissibility instrument + a drift map.** ⛔ No CLU cell, no store, no controller, no performance claim about the CLU of any kind was run or is made here.
- **Laundering control:** N/A — **I am the laundering control**, exactly as the INSECTS gate was.
- **Falsifies:** M1 failing ⇒ the harness/loader is defective and nothing on this stream is quotable. **M1 passed** (§2).
- **Does NOT falsify:** an exemplar/NN store beating the forecasting baselines falsifies **Metro's admissibility as a VALUE venue**, not the CLU.
- ⛔ NOT-RUNs declared, never nulls (§9).

## Flag provenance (mandatory, protocol §5)
| item | value |
|---|---|
| repo HEAD at run time | `7fcef50` — **unchanged before and after; 0 tracked files modified** (`git status --porcelain` empty of non-`??` entries at start and end) |
| tracked code touched | **none.** No `chlu/`, no `tests/`, no `pyproject.toml`, no `uv.lock`. No branch, no commit, no worktree, no push. |
| environment | **scratch venv** `.claude/scratch/c2w10-metro/.venv`, CPython **3.12.9**, `numpy==2.5.2`, `scikit-learn==1.9.0`, `scipy==1.18.0`, `torch==2.13.0`, `matplotlib==3.11.1`, `pandas==3.0.5` (installed, unused). Full freeze: `…/c2w10-metro-gate/scratch_venv_freeze.txt`. **`river` NOT used** — Metro is a regression venue and river ships no regression SAM-kNN |
| project venv / lock | **untouched** (Head ruling 6; the `pandas 3.0.3→2.3.3` precedent) |
| protocol | hidden clock (`date_time` withheld from every arm), horizon **24 h**, **24-h label embargo ON** (`METRO_EMBARGO=1`, the default), prequential MAE/RMSE over **all 34,848 scored pairs** |
| canonicalisation | duplicate `date_time` → **keep first**; hourly grid; gaps ≤ **3 h** filled (linear interp on traffic, forward-fill on weather); **target hour never imputed** |
| feature vector (identical for every arm) | **32 features** = 24 recent traffic lags `y_i…y_{i−23}` ⊕ 3 weekly echoes `y_{i−143,−144,−145}` ⊕ `temp, rain_1h, snow_1h, clouds_all` ⊕ `holiday`. **132 B/exemplar** = (32+1)×4 |
| exemplar arms | distance-weighted k-NN, **registered k = 5**, sliding window `L ∈ {250,500,1000,1007,2000,5000,5037,10000,14894,34847}`, **raw AND causally standardised**, best consumed. Unregistered anti-hobbling extension: k-ladder `k ∈ {1,3,10,25,50}` at L ∈ {1007, 5037, 14894}. `knnsam` = **our** dual-memory regression adaptation of Losing et al. 2016 |
| strong baselines | `gbdt` = `HistGradientBoostingRegressor(random_state=0, early_stopping=False)`, refit every **720** pairs on all embargoed past; `gbdt_tuned` = same + `max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=10`; `gbdt_recent` (window 8760); `gbdt_cat` (+11 `weather_main` one-hots); `gru_big` = torch GRU(1→128) over the 24-lag sequence ⊕ 8 static → MLP(64) head, refit every 720, 6 epochs, all past, Adam 1e-3, SmoothL1, seed 0; `gru` (h=64, 2 epochs, buffer 8760); `mlp` = sklearn MLP(256,128), refit every 1440; `ridge_batch` (α=1, refit 720); `rls` = exact recursive-least-squares ridge |
| standardisation | causal prequential per-feature z-score from pairs `< t` only, **clipped to ±10 SD** (declared harness fix, §8) |
| seeds | strong baselines seed 0 (single seed — see §10.3); exemplar arms are **deterministic** given the frozen stream; `knnsam` reservoir seed 0; null permutation seed 1 |
| commands | `.venv/bin/python prep.py → build.py → m1.py → m2.py <arm> → driftmap2.py → null_build.py → bands.py → plots.py → build_gate.py → add_notes.py`. All scripts copied verbatim to `…/c2w10-metro-gate/code/` |

**Pre-registration:** `.claude/outputs/c2w10-metro-gate/PREREG.md`, filed **after** `prep.py` (structure only — record counts, gaps, ranges) and **before** any MAE/RMSE/drift magnitude existed. 15 numeric predictions, scored in §7.

---

## 1. Data: fetched, frozen, hashed

**Declared cache path:** `/Users/user/Desktop/CHLU/.claude/data/c2w10-metro/`

| file | sha256 | bytes |
|---|---|---|
| `metro.zip` (UCI dataset 492, direct download) | `b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768` | 405,547 |
| `Metro_Interstate_Traffic_Volume.csv.gz` (as shipped inside the zip) | `0b3679ac15173f79c6dc6c5ef8a0798d806fa5c5d7f05c84a5fa711bd1b05f07` | 405,373 |
| `Metro_Interstate_Traffic_Volume.csv` (gunzipped, **the stream contract**) | `749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e` | 3,237,208 |
| built pair array (X⊕y⊕tgt bytes, **the second contract**) | `fbb0487b95c13ab795003dd09057fe2ad73f8ac5edd5f0e53db99d897f0cc661` | 34,848 × (32 f32 + 1 f32 + 1 i64) |

**Structural facts the scout did not have (all measured, none assumed):**
- 48,204 raw records — but only **40,575 unique hours**: **7,629 duplicate rows** over **5,445 duplicated timestamps** (multiple weather rows per hour). *A loader that does not dedupe will double-count 16 % of the stream.*
- Hourly grid 2012-10-02 09:00 → 2018-09-30 23:00 = **52,551 hours**, of which **11,976 (22.79 %) are missing**, including **a 7,386-hour hole (2014-08-08 → 2015-06-11, ~10 months)** that splits the record into two eras, plus 2,192 isolated 1-hour gaps and 11 further gaps ≥ 24 h.
- `holiday` is non-`None` on **53 hours in six years** — one flagged hour per holiday, not per day. As a feature it is ≈ noise.
- Dirty values left in, declared: `temp = 0 K` on 10 hours; `rain_1h = 9831.3` on 1 hour.
- Gap-fill ladder (measured before the prereg): maxgap 1/2/3/6/12 h ⇒ 29,126 / 33,303 / **34,848** / 37,174 / 38,773 scored pairs. **maxgap = 3 h is the declared primary.**

---

## 2. M1 — the loader/protocol control (reported before M2, as required)

| baseline | MAE | RMSE | note |
|---|---|---|---|
| **persistence** (last observed at origin) | **574.1347** | **1033.3894** | |
| **seasonal-naive t−24 h** | **574.1347** | **1033.3894** | ⛔ **bit-identical to persistence**, max abs diff **0.0** |
| **seasonal-naive t−168 h** (weekly) | **342.6513** | **643.7791** | the non-degenerate naive |
| seasonal-naive t−168 h, 3-h smoothed | 389.6153 | 652.7180 | |
| running mean so far | 1734.8623 | 1978.4562 | |
| 24-h window mean at origin | 1749.2778 | 2011.6903 | |
| global mean (non-causal reference) | 1733.8043 | 1978.1782 | y SD = 1978.178 |

### **`m1_pass = true`** — but read the second row of §3 before using it.
Registered rule: `(mae_persistence − mae_best_strong)/mae_persistence ≥ 0.05`. Measured **0.41616**. Persistence is **not** at ceiling and the hidden-clock protocol works as intended.

### ⛔ 2b. The registered M1 rule used the wrong naive — a finding
At a 24-h horizon, "last observed" **is** the period-24 seasonal-naive; the task's two M1 baselines collapse into one (predicted as **P4**, confirmed exactly). The strongest trivial rule on this stream is the **weekly** naive at **342.65**, and against *that*:

```
best strong baseline (gbdt_tuned)      335.203
one-line weekly seasonal-naive         342.651
headroom                               2.17 % relative MAE
```
**Six-and-a-half MAE units separate a tuned gradient-boosted ensemble from `ŷ_j = y_{j−168}`.** Souza §4.2 / Žliobaitė 2013's criterion-2 test, applied honestly, is **failed here independently of criterion 4**. (The exemplar store does clear it: 300.09 is 12.4 % better than the weekly naive — which is precisely the problem.)

---

## 3. M2 — the criterion-4 tripwire (the deciding measurement)

### 3.1 The gate's arithmetic

| arm | class | state bytes | MAE | RMSE |
|---|---|---|---|---|
| **k-NN windows, L = 5,037 (0.634 MiB), k = 10, raw** | exemplar | **664,884** | **314.575** | 560.341 |
| k-NN windows, L = 5,037, **registered k = 5**, raw | exemplar | 664,884 | 320.982 | 570.919 |
| **k-NN windows, L = 1,007 (133 kB), k = 5, raw** | exemplar | **132,924** | **327.882** | 587.538 |
| `knnsam` dual memory, L_max = 5,037 (ours) | exemplar | 664,884 | 325.709 | 580.253 |
| **`gbdt_tuned`** — best strong | strong | 3,618,071 | **335.203** | 602.736 |
| `gbdt` (defaults) | strong | 414,074 | 338.647 | 604.274 |
| `gbdt_cat` (+`weather_main`) | strong | 414,492 | 339.016 | 606.380 |
| `gbdt_recent` (1-yr window) | strong | 414,146 | 346.324 | 616.199 |
| `rls` (exact online ridge) | strong | 8,976 | 374.678 | 628.753 |
| `gru_big` (GRU-128, 6 ep, all past) | strong | 236,548 | 374.717 | 656.238 |
| `ridge_batch` | strong | 698 | 399.155 | 674.535 |
| `mlp` (256,128) | strong | 1,339,113 | 414.972 | 710.185 |
| `gru` (GRU-64, 2 ep, 1-yr buffer) | strong | 70,404 | 447.467 | 851.290 |
| **persistence** (mandatory in every table) | naive | 4 | 574.135 | 1033.389 |

```
mae_best_exemplar_at_budget = 314.575   (L = 5,037 = 0.634 MiB, k = 10, raw)
mae_best_strong_baseline    = 335.203   (gbdt_tuned)
m2_margin_rel = (314.575 − 335.203) / 335.203 = −0.061539
criterion4_cleared_metro = (−0.061539 > 0.02) = FALSE
```
### **⇒ CRITERION 4 HAS FIRED ON METRO.**

### 3.2 Robustness — it fires against **every** strong reference, by 4–25 %
Unlike the INSECTS gate (which fired by 0.10 points), this is not a threshold call.

| strong reference | MAE | `margin_rel` vs the 0.634 MiB store | fires? |
|---|---|---|---|
| `gbdt_tuned` (best strong; least favourable to firing) | 335.203 | **−0.0615** | ✔ |
| `gbdt` (registered default config) | 338.647 | −0.0710 | ✔ |
| `gbdt_cat` | 339.016 | −0.0721 | ✔ |
| `gbdt_recent` | 346.324 | −0.0916 | ✔ |
| `rls` | 374.678 | −0.1604 | ✔ |
| `gru_big` (the registered sequence model) | 374.717 | −0.1605 | ✔ |
| `ridge_batch` | 399.155 | −0.2119 | ✔ |
| `mlp` | 414.972 | −0.2419 | ✔ |
| `gru` | 447.467 | −0.2969 | ✔ |

Even the **133 kB** store (327.882) fires against `gbdt_tuned` at `margin_rel = −0.0218`, and the **registered k = 5** arm at 0.634 MiB (320.982) fires at `−0.0424`. The verdict does not depend on the unregistered k-ladder.

### 3.3 The mechanism — and it is the exact **inverse** of the INSECTS gate's
On INSECTS, exemplar accuracy was **monotonically decreasing** in store size above L ≈ 500 (recency was the hidden regime variable). On Metro it is **monotonically improving** across the whole ladder:

| L (k = 5, raw) | 250 | 500 | 1,000 | **1,007** | 2,000 | 5,000 | **5,037** | 10,000 | **14,894** | 34,847 |
|---|---|---|---|---|---|---|---|---|---|---|
| **MAE** | 407.91 | 347.89 | 327.79 | **327.88** | 325.22 | 321.22 | **320.98** | 307.08 | **306.76** | **304.02** |
| state bytes | 33,000 | 66,000 | 132,000 | 132,924 | 264,000 | 660,000 | 664,884 | 1,320,000 | 1,966,008 | 4,599,804 |

(figure `fig1_window_ladder.png`.) **More bytes always help on Metro.** The mechanism is stated plainly: Metro's regime clock — hour-of-day, day-type, week phase — is **fully encoded inside each pair's own 24-lag window**, so a stored exemplar never goes stale and the store is a pure capacity play. On INSECTS the regime variable (temperature) was *outside* the feature vector, which is why recency was the only proxy and old exemplars were poison. ⭐ **The two gates together give the wave a clean statement: an exemplar store's byte-frontier slope is determined by whether the regime variable is inside or outside the query.**

Two subsidiary readings:
- **RAW beats STANDARDISED everywhere** on Metro (320.98 vs 342.75 at L = 5,037) — the reverse of INSECTS, where standardising was worth +4 to +6 points. Cause: 27 of the 32 features are traffic volumes on one common scale, so the raw Euclidean metric is already well-conditioned; standardising only up-weights the 5 weather columns.
- **The dual-memory launder does not beat a plain window** here (325.71 vs 320.98 at 0.634 MiB). Its self-adjusting STM settles at a mean size of **2,367 of a permitted 5,037** and it predicts from the STM∪LTM union on 9,000 of 34,848 steps. On a stream where nothing goes stale, the STM/LTM machinery has nothing to buy.

### 3.4 Anti-hobbling, discharged in **both** directions (the F3 rule)
- **Launder side:** exemplar arms run raw **and** causally standardised (best consumed); a k-ladder `{1,3,10,25,50}` beyond the registered k = 5 (k = 10 wins at every L ≥ 5,037, and it is what the gate consumes); a dual-memory SAM-kNN analogue; and the CLU's own budget point.
- **Strong side — this is the one that decides the gate, so it was pushed hardest:** GBDT was run at defaults **and** tuned (500 iters, lr 0.05, 63 leaves — worth 1.0 %), with a recency window, and with the categorical `weather_main` added (worth **−0.1 %**, i.e. nothing: **P15 confirmed**). The sequence model was run small and large (GRU-64/2 epochs/1-yr buffer → 447.47; GRU-128/6 epochs/all past → **374.72**). An exact online ridge and a 256×128 MLP were added as further references. **Nine strong references; the tripwire fires against all nine.**
- ⭐ **Static-holdout sanity check** (single 70/30 chronological split, no streaming, no refits, to prove the streaming harness is not the story):

  | | tuned GBDT | GBDT default | k-NN k=25 | k-NN k=10 | k-NN k=5 | persistence | weekly naive |
  |---|---|---|---|---|---|---|---|
  | **holdout MAE** | 266.33 | 273.54 | **265.47** | 267.58 | 277.32 | 562.28 | 326.30 |

  **A plain distance-weighted k-NN edges the tuned GBDT by 0.3 % on a static holdout.** The k-NN/GBDT parity is a property of this feature space, not of my prequential wrapper. **Criterion 4 fires under the static reading too.**

### 3.5 ⛔ The protocol defect I found in my own pre-registration, and its correction
PREREG.md §1 asserted that plain prequential test-then-train is leak-free "because pair `j`'s features stop at `j−24` and its label is revealed only after prediction". **That is wrong.** Pair `t−1`'s *label* is the traffic volume at target time `j−1` — **23 hours after pair `t`'s forecast origin `j−24`**. Plain test-then-train hands a continuously-updated learner up to 23 h of future traffic, and it does so **asymmetrically**: a k-NN store that admits every new pair immediately gains from it; a GBDT refit every 720 pairs does not.

**Fix (all primary numbers above are post-fix):** `A(t) = ` index of the last pair whose *target* time is ≤ pair `t`'s *origin* time (`searchsorted(tgt, tgt−24, 'right') − 1`; median `t − A(t)` = 24, mean 22.86, max 24). Every store, fit and online update is restricted to indices ≤ `A(t)`.

**What the leak was worth** (full 45-arm table in `METRO-GATE.json → protocol_correction_24h_label_embargo.delta_table`):

| arm | leaky MAE | embargoed MAE | leak worth |
|---|---|---|---|
| k-NN L=250, std | 398.17 | 446.64 | **+10.85 %** |
| k-NN L=250, raw | 373.00 | 407.90 | +8.56 % |
| k-NN L=1,007, std | 344.28 | 363.59 | +5.31 % |
| k-NN L=5,037, raw | 312.06 | 320.98 | +2.78 % |
| k-NN L=14,894, raw | 300.42 | 306.76 | +2.07 % |
| **`gbdt`** | 339.80 | **338.65** | **−0.34 %** |
| **`mlp`** | 420.27 | **414.97** | −1.28 % |

**The verdict is unchanged by the fix** (leaky within-protocol `margin_rel` = −0.0946; embargoed = −0.0615) — the embargo *reduces* the exemplar store's advantage by ~3 points of relative MAE and criterion 4 still fires. ⚠ **But any C2W10 harness that runs this stream must implement `A(t)` or its numbers are not comparable to these.**

---

## 4. M3 — headroom

| quantity | value |
|---|---|
| best strong (`gbdt_tuned`) vs **persistence** | **41.62 %** relative MAE gain |
| best strong vs **weekly seasonal-naive** | **2.17 %** ⛔ |
| best exemplar (any budget) vs persistence | 47.73 % |
| best exemplar vs weekly seasonal-naive | 12.42 % |

`m1_pass = true` on the registered rule; **criterion 2 fails on the honest rule.** Both numbers are in the gate file and neither is hidden.

---

## 5. M4 — the byte ledger, computed from the data's own shape

`bytes_per_exemplar = (32 float32 features + 1 float32 target) × 4 = 132 B` (INSECTS' was 133 B — the two gates' budgets are directly comparable to within 0.8 %).

| entry | bytes | MiB | note |
|---|---|---|---|
| exemplar store, **L = 5,037** | **664,884** | 0.63407 | the 0.634 MiB budget point, 116 B of slack under 665,000 |
| exemplar store, **L = 1,007** | 132,924 | 0.12677 | the 133 kB budget point |
| **CLU store, d = 12** | **1,966,080** | 1.875 | `n_atoms = 512·√2¹² = 32,768`; `n_atoms × (dim+2) × 4`, `dim = 13` |
| exemplars affordable at the CLU budget | **14,894** = 1,966,008 B (72 B slack) | 1.87493 | ⚠ PREREG.md said 14,882 — my slip; `1966080 // 132 = 14894` |
| `gbdt` measured state | 414,074 | 0.395 | pickle protocol 5 of the fitted model at the last refit |
| **`gbdt_tuned` measured state** | **3,618,071** | 3.450 | **5.44× the 665,000 B exemplar budget it loses to** |
| `mlp` measured state | 1,339,113 | 1.277 | |
| `gru_big` parameters | 236,548 | 0.226 | ⚠ **excludes** optimizer state and the replay buffer, which is not byte-counted and would dominate (all-past × 132 B) |
| `rls` | 8,976 | 0.009 | (33² + 33) × 8 B, float64 |
| persistence | 4 | — | |

⚠ **Arms that are NOT byte-matched, declared:** every GBDT variant, `mlp`, and `gru_big` (whose replay buffer is uncounted). The byte-matched competitors are the exemplar arms — and those are the ones that win.

---

## 6. The drift map (Part B) — **OURS, NOT THE LITERATURE'S**

Method (recorded verbatim in `DRIFT-MAP.json`): **Webb, Hyde, Cao, Nguyen & Petitjean (2016), *Characterizing concept drift*, DMKD 30:964–994** — drift magnitude = **total variation distance between the distributions in two windows**. Estimator: 10 global-quantile bins of the target (⊕ `temp` and `clouds_all` for the "joint" maps), per-window normalised histogram, TV between **every** window pair. Windows = calendar days (≥ 12 scored pairs) or calendar weeks (≥ 60). Regimes = KMeans on the window signature, K by silhouette. **Calendar attributes were attached AFTER clustering, for labelling only, and were never inputs to the discovery.** ⛔ Metro has no published drift annotation (verified by `c2w10-benchmark-scout`); **this annotation is ours and every artifact says so.**

### 6.1 The four maps, and what they recovered

| map | windows | K (silhouette) | mean pairwise TV | **regime visits** | post-hoc identity |
|---|---|---|---|---|---|
| **D_primary** (day, y-only) | 1,534 days | **2** | 0.3734 | **239 / 252** | **day-type**: R0 = Sun 41.3 % + Sat 37.7 %, mean y **2,619**; R1 = Tue–Fri ~20 % each, mean y **3,634** |
| **D_fine4** (day, y-only, K=4 forced) | 1,534 | (2) | 0.3734 | 167/199/275/209 | R0 Sun-dominant (52.3 %), R1 Sat-dominant (51.7 %), R2 Fri+Mon shoulder, R3 Tue–Thu core |
| **D_joint** (day, y⊕temp⊕clouds) | 1,534 | **3** | 1.7519 | 52 / 193 / 166 | **season**: R0 winter (mean 257.7 K; Jan 35 %, Feb 28 %, Dec 26 %), R1 shoulder, R2 summer (288.1 K; Jul 17 %, Aug 16 %) |
| **W_season** (week, joint) | 225 weeks | **3** | 1.2410 | 12 / 22 / 20 | winter (260.6 K) / shoulder (272.5 K) / summer (290.6 K) weeks |

**The purely data-driven day map recovers the weekday/weekend split with no calendar input at all** — that is the map validating itself. The seasonal axis only appears once weather is admitted to the signature, which is the honest statement of what the target histogram alone can see.

- Figures: `fig2_drift_magnitude.png` (the TV matrices), `fig3_regime_timeline.png` (the revisit schedule over 6 years).
- **The explicit revisit-schedule index table** — the Metro analogue of INSECTS' published change points — is `revisit_schedule.tsv` (**1,101 rows**: map, regime, visit number, first/last day, n windows, n pairs, **first/last pair index**) and the same content is in `METRO-GATE.json → drift_map.maps.*.revisit_schedule`.

### 6.2 ⭐ Band diagnostics and the registered exclusions (the INSECTS `b = 4` lesson, applied BEFORE use)
Registered in PREREG.md §1 **before the map was built**: exclude a band if (a) the best strong arm's relative MAE gain over persistence within the band is < 5 %, or (b) band SD(y) < 0.25 × global SD(y), or (c) n < 200.

| map | band | n | SD(y) | H(y) bits | persistence | weekly naive | best strong | best exemplar | gain vs pers. | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| D_primary | R0 weekend | 12,078 | 1,550 | 3.00 | 792.09 | 455.35 | 414.35 | 390.14 | +0.477 | **kept** |
| D_primary | R1 weekday | 22,578 | 2,087 | 3.25 | 459.25 | 283.12 | 292.79 | 251.92 | +0.362 | **kept** |
| D_fine4 | R0 Sun | 5,858 | 1,426 | 2.79 | 755.57 | 558.46 | 510.61 | 481.43 | +0.324 | **kept** |
| D_fine4 | R1 Sat | 6,141 | 1,624 | 3.05 | 814.47 | 351.53 | 313.32 | 286.72 | +0.615 | **kept** |
| D_fine4 | R2 Fri/Mon | 9,974 | 2,003 | 3.26 | 546.94 | 310.85 | 310.68 | 267.82 | +0.432 | **kept** |
| D_fine4 | R3 Tue–Thu | 12,683 | 2,145 | 3.19 | 398.40 | 265.02 | 283.93 | 240.55 | +0.287 | **kept** |
| D_joint | R0 winter | 4,107 | 1,885 | 3.30 | 650.68 | 440.57 | 391.04 | 372.56 | +0.399 | **kept** |
| D_joint | R1 shoulder | 10,782 | 2,008 | 3.32 | 590.14 | 360.57 | 382.32 | 330.33 | +0.352 | **kept** |
| D_joint | R2 summer | 19,767 | 1,975 | 3.32 | 551.45 | 313.40 | 297.82 | 268.54 | +0.460 | **kept** |
| W_season | R0/R1/R2 | 5,080/10,158/19,090 | ~1,950 | 3.30–3.32 | 651.7/613.3/530.6 | 464.7/359.4/300.6 | 405.9/390.6/285.8 | 383.4/320.7/266.0 | +0.377/+0.363/+0.461 | **kept** |
| all maps | **UNASSIGNED** | 192 (D-maps) / 520 (W) | — | — | — | — | — | — | — | **EXCLUDED** (windows below the minimum pair count) |

**No band is persistence-trivial, no band is at ceiling, no band is degenerate.** The map is *clean* — the contrast with INSECTS' `b = 4` (entropy 0.84 bits, No-Change 91 %, every arm 97–99 %) is stark, and it is the one respect in which Metro is the better-behaved venue. ⚠ Note the **exemplar-vs-strong margin is negative in every band of every map** (−4.7 % to −17.9 %): criterion 4 does not fire because of one favourable slice.

### 6.3 The drift-free NULL, re-constructed (never quietly dropped)
`out-of-control` has no data source (the INSECTS gate established this). The control is **re-constructed**, not dropped: a fixed-seed (`default_rng(1)`) uniform permutation of the **pair sequence** — each pair's 32-D feature vector and target kept intact, only the stream order destroyed. `P(X, y)` and `P(y|X)` are preserved **exactly**; only the ordering is gone.
Path `…/c2w10-metro/pairs_shuffled_seed1.npz`, sha256 `a02cd4567559ad2b9cb3e5819be9b7de70c186a109fcf4c73c08431099c71f5c`, permutation sha256 `5be8314a77f2aa5f268d1bb68bf729f0451f1381fb7ab1d3aa6e078502092908`.

**Its own positive control passes exactly:** persistence and both seasonal-naives are functions of a pair's *own* features, so their scores must be invariant. Measured: **574.1346849611 ordered vs 574.1346849611 shuffled** (diff **0**), and 342.6513262671 vs 342.6513262671 (diff **0**). **P13 confirmed exactly.**

| arm | ordered MAE | shuffled MAE | Δ relative |
|---|---|---|---|
| persistence / weekly naive | 574.135 / 342.651 | 574.135 / 342.651 | **0.000** (by construction) |
| k-NN L = 250 | 407.905 | 418.873 | **+2.69 %** (order helps) |
| k-NN L = 1,007 (133 kB) | 327.882 | 343.323 | **+4.71 %** (order helps) |
| **k-NN L = 5,037 (0.634 MiB)** | **320.982** | **311.748** | **−2.88 %** (order **HURTS**) |
| **k-NN L = 14,894 (CLU budget)** | **306.762** | **301.068** | **−1.86 %** (order **HURTS**) |
| k-NN L = 34,847 (unbounded) | 304.016 | 297.390 | −2.18 % |
| `rls` | 374.678 | 377.209 | +0.68 % |
| `ridge_batch` | 399.155 | 400.667 | +0.38 % |

### ⭐⭐ 6.4 The reading the Hub needs most
**There is a crossover at L ≈ 2,000–5,000 exemplars.** Below it, temporal ordering is worth up to 4.7 % — recency is a real signal. **At and above the 0.634 MiB budget, destroying the stream order does not hurt the store, it HELPS it by ~2–3 %.** Once the store is large enough to hold ~7 months of pairs, the six-year mixture is a *better* neighbour pool than the recent past, and the arrival order carries no information the store can use.

**Consequence for PREREG-C2W10 §6.2's V1/V2:** a persistent-vs-episodic retention contrast measured on Metro at the wave's byte budgets is measuring **an ordering effect that this stream does not have**. `P14` predicted degradation ≥ 5 % under the shuffle at L = 1,007 and got **+4.71 %** — technically a near-miss, and the honest read is that the effect exists only at store sizes **below** the ones C2W10 would use. **This is a second, independent reason Metro cannot host the VALUE claim**, and it is the one that does not depend on any baseline being strong.

### 6.5 Decimation ladder (reported structurally, as asked)

| `m` | scored pairs | **horizon in records** | D_primary transitions | D_fine4 | D_joint | W_season | bands emptied |
|---|---|---|---|---|---|---|---|
| **1** | **34,848** | **24** | 509 | 869 | 430 | 67 | none |
| 2 | 17,424 | 12 | 509 | 869 | 430 | 67 | none |
| 5 | 6,970 | **4.8 ⛔** | 504 | 864 | 425 | 67 | none |
| 10 | 3,485 | **2.4 ⛔** | 502 | 863 | 423 | 64 | none |

⛔ **Hazard the Hub must file: at `m ∈ {5,10}` the 24-h horizon is not an integer number of records.** Uniform decimation of an hourly stream with a 24-h horizon silently redefines the horizon *and* every lag feature; only `m ∈ {1, 2}` are structurally usable. And Metro at `m = 1` is 0.44× INSECTS in scored pairs, with the whole 45-arm suite costing minutes on CPU — **decimation is not needed here; `m = 1` is the recommendation.** ⚠ The standing rule still applies: any adaptation quantity is **per-instance-since-change, never per-stream-position**.

---

## 7. Pre-registration scored (PREREG.md §2)

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | `mae_persistence` = 700 (80 % CI 500–950) | **574.13** | ✔ in interval |
| P2 | `rmse_persistence` = 1,150 (800–1,500) | **1,033.39** | ✔ |
| P3 | `mae_seasonal_naive_168` = 520 (380–700) | **342.65** | ✘ **missed low, outside the interval** |
| P4 | seasonal-naive t−24h ≡ persistence | **max abs diff 0.0** | ✔ exact |
| P5 | `mae_best_strong` = 420 (320–550) | **335.20** | ✔ |
| P6 | k-NN at 0.634 MiB = 470 (370–620) | **320.98** (k=5) / **314.58** (k=10) | ✘ **missed low, outside the interval** |
| P7 | `m2_margin_rel` = +0.10 (−0.05 … +0.30) | **−0.0615** | ✘ **outside the interval, wrong sign** |
| P8 | `criterion4_cleared_metro = true`, p = 0.60 | **false** | ✘ the 0.40 branch fired |
| P9 | `m3_headroom_rel` = 0.40 (0.20–0.55) | **0.4162** | ✔ |
| P10 | ladder non-monotone with an interior optimum in [1000, 5037]; p = 0.75 the CLU budget point is not the best | **monotone improving across the whole ladder**; the CLU point is 2nd-best of 10, beaten only by the unbounded store | ✘ / ✔ (the second clause holds trivially) |
| P11 | drift map `K` = 4 (3–6) | **K = 2** by silhouette on the primary map (3 on both joint maps) | ✘ |
| P12 | top drift axis is seasonal, not weekly; p = 0.70 | **day-type**, not seasonal — the seasonal axis appears only when weather enters the signature | ✘ |
| P13 | persistence invariant under the shuffle, exact | **diff = 0.0000000000** | ✔ exact |
| P14 | k-NN(1,007) degrades ≥ 5 % under the shuffle; p = 0.65 | **+4.71 %** | ✘ (near-miss; and it **inverts** to −2.9 % at 5,037) |
| P15 | `gbdt_cat` improves `gbdt` by < 2 % | **−0.11 %** (slightly worse) | ✔ |

**I got 7 of 15 and the three misses that matter all point the same way: I systematically underestimated how strong a nearest-neighbour lookup is on this stream.** P3, P6 and P7 are one error — I priced Metro as a covariate-regression problem when it is an *analog-forecasting* problem, in which the 24-lag window is a near-sufficient statistic and the best predictor is "find the most similar recent day". That error is exactly what the tripwire exists to catch, and it is why a fallback must be tested rather than argued. **P8 resolves NO: my 0.60 prior was wrong and the scout's ⚠ was right.**

---

## 8. Harness defects I hit (evidence, not summaries)
1. **Causal z-score blow-up.** The running per-feature variance of a near-constant early column (`snow_1h`, `holiday`) drove standardised values to **1.0e6** at t < 500. The first GRU run emitted **7,716,597** and **7,411,708** at pair indices 12,566 and 4,215 → **MAE 928.60, RMSE 57,288.6**. Fixed by a declared **±10 SD clip** applied to *every* standardised arm; the GRU then scored 442.36 (leaky) / 447.47 (embargoed). **The pre-fix number is reported here and used nowhere.**
2. **`rls_ff` (RLS with forgetting factor 0.999) diverged**: MAE 1,881.72 / **RMSE 52,680.9** on the first run, MAE 2,979.11 / **RMSE 227,708.7** on the second. Sherman–Morrison covariance update is numerically unstable at `forget < 1` on these features. **Excluded from every table; declared NOT-RUN under the embargoed protocol.**
3. **The 24-h label leak** (§3.5) — my own pre-registered protocol claim, wrong, found mid-run, corrected, both protocols reported.

## 9. Declared NOT-RUNs (never to be reported as nulls)
Any CLU cell of any kind · any retention `R(b)` / adaptation `A(b)` number on Metro (that is S4's, and §6.4 says the instrument has no room) · `rls_ff` under the embargo (diverged) · SAM-kNN-regressor at the CLU budget (the reference implementation is ours and O(L) per step; the plain window at 14,894 **was** run) · multi-seed strong baselines (single seed 0; see §10.3) · the `maxgap = 6 h` robustness variant of the pair stream (built structurally, **no arm run**) · any decimated (`m > 1`) arm (the ladder is structural only) · a third venue of any kind (⛔ task rule: a fired M2 stops here).

## 10. Limitations and confounds (stated, not minimised)
1. **The feature vector is mine.** 24 recent lags + 3 weekly echoes + 4 weather + holiday is a defensible, conventional choice, and it is identical for every arm — but a different window (e.g. 168 recent lags) would move every number. The k-NN attack is *defined* by the window, so the launder's strength is partly a design choice. Mitigation: the strong baselines get exactly the same features, and the static-holdout check (§3.4) reproduces the parity outside the streaming harness entirely.
2. **The streaming GBDT is refit every 720 pairs and frozen in between**; the k-NN store updates every step (subject to the embargo). That is a real advantage to the exemplar arm in the *prequential* numbers. It is not the story: `gbdt_tuned` at refit-720 (335.20) and the static-holdout tie (266.33 vs 265.47) bracket the effect, and criterion 4 fires at both ends.
3. **Single seed on the strong baselines.** GBDT/GRU/MLP ran at seed 0 only. Exemplar arms are deterministic. The margins are 4–25 %, far outside any plausible seed spread for these estimators, but **no headline number here carries a variance estimate** and I am not claiming one.
4. **The gap-fill is mine** (≤ 3 h, linear on traffic, forward-fill on weather). 2,771 of 43,346 valid hours (6.4 %) are imputed *features*; targets are never imputed. A stricter maxgap = 1 h costs 16 % of the pairs.
5. **The 7,386-hour hole** (10 months) means "Metro 2012–2018" is really two eras; the weekly-echo features are unavailable across it and those pairs are dropped. No arm is told about the break.
6. **`knnsam` is our regression adaptation**, not a published algorithm — I am not claiming to have run "SAM-kNN" on Metro. It loses to the plain window anyway, so nothing rests on it.
7. **The drift map's K is silhouette-selected**, and silhouette is known to favour K = 2. I report the whole silhouette curve (`{2: 0.373, 3: 0.208, 4: 0.183, 5: 0.180, 6: 0.189, 7: 0.197, 8: 0.183}`) and publish a forced-K = 4 map beside it rather than picking one.
8. **The drift-free null is a shuffle, not a drift-free data source.** It preserves the pooled six-year mixture, not a stationary generating process. Its four limits are enumerated in `null_meta.json` and travel with the artifact.

## 11. Recommended next experiments (for the Hub, in priority order)
1. ⛔ **Do not run a Metro cell.** Two independent kills (criterion 4 at −6.2 %; criterion 2 at 2.2 % headroom over a one-line rule) plus a third structural one (§6.4: ordering is worth ≤ 0 at the wave's byte budgets). **File Metro as the sixth confirmation of the criterion-4 theorem** and hand the venue question up — it is the Hub's call, above it the Head's, and explicitly not mine.
2. **The two gates now support a theorem worth writing down** (§3.3): *the sign of an exemplar store's byte-frontier slope is determined by whether the regime variable is inside or outside the query.* INSECTS (regime outside) → more bytes hurt, 75.56 → 59.75 %. Metro (regime inside the 24-lag window) → more bytes help monotonically, 407.91 → 304.02 MAE. **That is a publishable observation obtained from two admissibility gates, for ~5 h of CPU total.** It also predicts, in advance, what a *third* venue must look like: **the regime variable must be outside the query AND the store's byte-frontier must be non-degenerate.**
3. **Adopt "publish the drift-free-null shuffle" as standing practice for every future VALUE venue**, before the lifecycle build. It cost 20 seconds of CPU here and it independently kills the venue. A venue where a fixed-seed pair-shuffle does not degrade the byte-matched store **cannot host a retention claim**, whatever the tripwire says.
4. **Adopt the label-embargo check as standing practice** for any h-step-ahead stream: compute `A(t)`, and report the leak-delta table. It is three lines and it caught a 10.9 % artifact in my own pre-registered protocol.
5. If a persistent-store VALUE venue is still wanted, the criteria to shop against are now **four**, not five: strong baselines · **headroom over the *strongest trivial* rule, not over persistence** · regime variable **outside** the query · **shuffle-sensitivity at the target byte budget**. I can run this exact 4-test battery on a candidate for ≈ 1 h of CPU.

## Git footprint
**None.** No branch, no commit, no worktree, no push. `git status --porcelain` shows **0 modified tracked files** before and after; HEAD is `7fcef50` throughout. All artifacts live under `.claude/`.

## Bug reports for `experiment-engineer`
No `chlu/` bug found (I ran none of it). Four loader/protocol hazards to encode if this stream is ever touched again:
- **Dedupe first.** 7,629 of 48,204 rows are duplicate timestamps. A naive `read_csv` over-counts 16 % of the stream.
- **The 24-h label embargo is mandatory** (§3.5). Without `A(t)`, an online store gets up to 23 h of future traffic; the artifact is worth up to **+10.9 %** MAE and is **asymmetric across arm types**.
- **`holiday` is 53 hours in 6 years**, not a day-level flag. Do not treat it as a calendar feature.
- **Reproduction gate:** the harness's loader must reproduce **both** `749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e` (raw UCI CSV) and `fbb0487b95c13ab795003dd09057fe2ad73f8ac5edd5f0e53db99d897f0cc661` (the built 34,848-pair array: sha256 over `X.tobytes() + y.tobytes() + tgt.tobytes()`, float32/float32/int64, C order) **before** any number from `METRO-GATE.json` is consumed. Mismatch = hard stop.

## Artifacts (exact paths)
- ⭐ **`.claude/outputs/c2w10-metro-gate/METRO-GATE.json`** — the gate file; every required key present; `criterion4_cleared_metro` computed arithmetically
- **`.claude/outputs/c2w10-metro-gate/DRIFT-MAP.json`** — the standalone published map, with the ownership statement in its first key
- **`revisit_schedule.tsv`** (1,101 rows, the explicit index table) · `bands.json` · `driftmaps.json` · `decimation.json` · `null_meta.json` · `facts.json` · `pairs_meta.json` · `m1.json` · `metrics_table.txt` (all 53 arm runs)
- `PREREG.md` (filed before any scored quantity) · `scratch_venv_freeze.txt`
- Figures: `fig1_window_ladder.png` · `fig2_drift_magnitude.png` · `fig3_regime_timeline.png` · `fig4_prequential.png`
- `code/` — every script verbatim (11 files)
- Frozen data: `.claude/data/c2w10-metro/` (zip + csv.gz + csv). Scratch: `.claude/scratch/c2w10-metro/` (pair arrays, shuffled null, per-arm predictions in `res/`, the **leaky-protocol archive** in `res_leaky/`)

---

## Proposed handover updates (for the Hub)

**§1.6 (experiments) — new rows**
- **C2W10 Metro gate, 2026-08-11, HEAD `7fcef50`, scratch venv (numpy 2.5.2 / sklearn 1.9.0 / torch 2.13.0), 34,848 pairs, m = 1, 24-h label embargo ON.** M1 **PASS**: persistence **574.1347** MAE / 1033.3894 RMSE; **seasonal-naive t−24 h is bit-identical to persistence (diff 0.0)**; weekly seasonal-naive **342.6513**. M2: best exemplar arm at **0.634 MiB** = **314.575** vs best strong `gbdt_tuned` = **335.203** ⇒ **`m2_margin_rel = −0.06154`, `criterion4_cleared_metro = FALSE`** ⇒ **CRITERION 4 HAS FIRED ON METRO**, the **sixth confirmation of the criterion-4 theorem**, for ≈5 h of CPU.
- **Robustness:** fires against **all 9** strong references (−0.0615 gbdt_tuned · −0.0710 gbdt · −0.0721 gbdt_cat · −0.0916 gbdt_recent · −0.1604 rls · −0.1605 gru_big · −0.2119 ridge_batch · −0.2419 mlp · −0.2969 gru). The **133 kB** store fires at −0.0218; the **registered k = 5** arm fires at −0.0424. On a **static 70/30 holdout** (no streaming at all) k-NN k=25 **265.47** edges tuned GBDT **266.33** ⇒ fires there too.
- ⛔ **Metro fails criterion 2 independently:** best strong baseline beats the **one-line weekly seasonal-naive** by **2.17 %** (335.203 vs 342.651). The ELEC2 pathology one level up.
- ⛔ **Third, structural kill — the re-constructed drift-free null:** a fixed-seed pair-shuffle **improves** the byte-matched store (320.982 → **311.748**, −2.88 % at 0.634 MiB; 306.762 → 301.068 at the CLU budget) while **hurting** it below ~2,000 exemplars (327.882 → 343.323, +4.71 % at 133 kB). **At the wave's byte budgets Metro's temporal ordering carries no exploitable information**, so V1/V2 have nothing to measure there. Null positive control exact: persistence 574.1346849611 both ways, diff **0**.
- ⭐ **The two gates now support a theorem:** the **sign of an exemplar store's byte-frontier slope is set by whether the regime variable sits inside or outside the query.** INSECTS (temperature withheld ⇒ outside) — accuracy **decreases** with store size, 75.56 % @ L=500 → 59.75 % @ L=14,782. Metro (the clock is fully recoverable from each pair's own 24-lag window ⇒ inside) — MAE **decreases monotonically** with store size, 407.91 @ L=250 → 304.02 @ L=34,847, with **no interior optimum**. Corollary, and it is a *design rule for the next venue*: the regime variable must be **outside** the query.
- **Also:** RAW beats causally-standardised features everywhere on Metro (320.98 vs 342.75 at L=5,037) — the mirror image of INSECTS, where standardisation was worth +4 to +6 points and *decided* that gate.

**§5 (provenance) — new rows**
- Frozen stream, `.claude/data/c2w10-metro/`: `Metro_Interstate_Traffic_Volume.csv` sha256 **`749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e`** (3,237,208 B; 48,204 records), from `metro.zip` sha256 **`b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768`** (UCI 492, CC BY 4.0, direct download, HTTP 200). Built 34,848-pair array sha256 **`fbb0487b95c13ab795003dd09057fe2ad73f8ac5edd5f0e53db99d897f0cc661`**. Drift-free null `pairs_shuffled_seed1.npz` sha256 **`a02cd4567559ad2b9cb3e5819be9b7de70c186a109fcf4c73c08431099c71f5c`**.
- **Structural facts, measured:** 48,204 records but only **40,575 unique hours** (**7,629 duplicate rows** / 5,445 duplicated timestamps); **22.79 % of the hourly grid missing**, including a **7,386-hour (10-month) hole 2014-08-08 → 2015-06-11**; `holiday` non-`None` on **53 hours in 6 years**; `temp = 0 K` on 10 hours; `rain_1h = 9831.3` once.
- **Byte ledger:** **132 B/exemplar** ((32 f32 features + 1 f32 target) × 4) — within 0.8 % of INSECTS' 133 B, so the two gates' budgets are directly comparable. L = **5,037** at 665,000 B; **1,007** at 133,000 B; **14,894** at the CLU d=12 budget of 1,966,080 B. `gbdt_tuned` measured state **3,618,071 B = 5.44×** the exemplar budget it loses to.
- **Reproduction gate is LIVE (two hashes):** no C2W10 cell may consume a Metro number until the harness's loader reproduces **both** the raw-CSV sha256 and the pair-array sha256. Mismatch = hard stop.

**§8 (NOT-RUNs / registry) — new entries**
- ⛔ **Metro Interstate is NOT the VALUE venue.** Filed as a registered admissibility finding, the **sixth** confirmation of the criterion-4 theorem, with two further independent kills (criterion-2 headroom 2.17 %; shuffle-insensitivity at budget). ⛔ **No third venue was improvised** — that call is the Hub's, and above it the Head's.
- ⛔ Declared NOT-RUN: any Metro `R(b)`/`A(b)`; `rls_ff` (diverged, RMSE 2.3e5); multi-seed strong baselines; the maxgap = 6 h stream variant; every decimated arm; **any CLU cell of any kind**.

**Corrections the Hub should push to the registries / PREREG-C2W10**
- **AMENDMENT owed to PREREG-C2W10 §2.0/§2:** the Metro fallback fired, was tested, and **failed its own tripwire**. The pre-authorisation (Head ruling 1) is discharged; the venue question is open.
- **PREREG-C2W10 §2's Metro protocol wording is defective in two ways:** (i) "a seasonal-naive (t−24 h) baseline" is **degenerate at a 24-h horizon** — it is bit-identical to persistence, measured max |diff| **0.0**; the non-degenerate naive is **t−168 h**. (ii) The protocol needs an explicit **24-h label embargo**; plain prequential test-then-train leaks up to 23 h of future traffic, worth **+10.85 %** to a 250-exemplar store and **−0.34 %** to GBDT — asymmetric, in the direction of firing criterion 4.
- **The M1/criterion-2 rule should be re-registered against the strongest trivial rule, not against persistence.** On Metro, persistence-based headroom reads **41.6 %** (pass) while weekly-naive headroom reads **2.17 %** (fail). Both are true; only the second is informative. Recommend: *"headroom is measured against the best naive/trivial predictor available on the stream, and every band table prints it."*
- **`negative_results.md` — N-new-d: "Metro Interstate under the hidden-clock 24-h protocol is metric-native at matched bytes, and its headroom over a one-line weekly rule is 2.2 %."** A 0.634 MiB byte-matched exemplar store beats a tuned GBDT by **6.2 %** relative MAE and beats every one of nine strong references; a fixed-seed pair-shuffle **improves** that store. Sixth confirmation of the criterion-4 theorem.
- **New standing practice to register (two items, both cheap):** (1) **the shuffle null runs BEFORE the lifecycle build on every candidate VALUE venue** — 20 s of CPU, and on Metro it kills the venue on its own; (2) **the label-embargo delta table is mandatory for any h-step-ahead stream.**
- **The drift map is published and is OURS:** `DRIFT-MAP.json` + `revisit_schedule.tsv` (1,101 visits across 4 maps; up to 491 day-level regime visits; **zero excluded bands** — the b = 4 contamination that hit INSECTS does **not** occur here). Method = Webb et al. 2016 TV drift magnitude; calendar labels attached post hoc, never used for discovery. If Metro is ever re-used for anything, this map stands and the ownership sentence travels with it.
