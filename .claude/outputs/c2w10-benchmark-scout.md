# c2w10-benchmark-scout — web-scout report

Task + acceptance criterion: shortlist the REAL-DATA benchmark for charter §A21's C2W10 wave ("the persistent store") — non-stationary streaming prediction with **regime revisits**, judged on retention + adaptation at **matched state-bytes**; every candidate assessed against intervention §6's five admissibility criteria, especially criterion 4 (NOT metric-native).
Status: **done** (one decisive item flagged UNVERIFIED and converted into a 1-hour pre-scoping run — see §6).
Dial declaration (echoed, §7 of the protocol): **Dial = lifetimes + admission** (retention of revisited regimes; admission/eviction under capacity pressure). **Laundering control = an exemplar store at matched bytes** (SAM-kNN / windowed kNN, see §5). **Falsifies:** the exemplar store at equal state-bytes matches or beats the CLU store on revisit-recovery. **Does NOT falsify:** losing to a per-regime oracle, or losing on a stream where the persistent-vs-episodic contrast is absent (that is the metric-native-ceiling theorem, not news).

---

## 0. Answer first

**Primary: the INSECTS *reoccurring* streams** — `incremental-reoccurring` and `incremental-abrupt-reoccurring` (balanced + imbalanced), Souza et al. (2020), USP Data Stream Repository, CC BY 4.0. They are the only real, laptop-scale streams I could verify that carry (a) **published ground-truth change points** and an explicitly *recurrent* regime schedule (three cycles of the same hidden temperature sweep), (b) a **hidden regime variable** (temperature, deliberately withheld from the 33 features) which makes the answer a function of *accumulated context*, not of the current query point, and (c) an honest difficulty profile: the persistence/No-Change baseline sits at **28–42 %** while ARF sits at **75–78 %**, with a documented **~6-point (up to 20-point) regime-conditioning headroom** on top. **Fallback: Metro Interstate Traffic Volume** (UCI, CC BY 4.0, 48,204 hourly records, 2012–2018) run as a hidden-clock, multi-step-ahead regression stream — different domain, different failure mode, but its regime structure is **not documented in any stream-learning benchmark paper** and we would have to establish it ourselves (flagged, not hidden).

**Two rejections that matter and are fully verified.** *Electricity/ELEC2 is inadmissible* (criterion 2): the naive persistence classifier scores **85.3 %**, beating 10 of 12 MOA adaptive classifiers and 12 of 17 published results — a trivial method sits at ceiling. *The whole Losing/SAM-kNN real-world family (Weather-NOAA, Rialto, Outdoor, Cover Type) is inadmissible as primary* (criterion 4): a **plain distance-weighted kNN over a 5,000-example sliding window is the second-best method on real-world average**, and an explicit dual exemplar store (SAM-kNN) is first — that is criterion 4's theorem being demonstrated in the literature, on exactly these streams. Both facts are *useful* to us: they hand us a ready-made, published, byte-budgeted laundering control (§5).

---

## 1. Candidate table (all figures verified against the cited primary source unless marked)

| # | Stream | Regime-revisit structure (documented?) | Size | Licence | Std. baselines | Std. metrics | Laptop? |
|---|---|---|---|---|---|---|---|
| C1 | **INSECTS inc-reoccurring / inc-abrupt-reoccurring** (bal. + imbal.) | **YES, by construction + published change points.** "three recurrent cycles of incremental changes where the temperature increase from 20°C to 40°C" | bal. 79,986 × 33 feat × 6 cls (≈20.4 MiB CSV); imbal. 452,044 | **CC BY 4.0** (USP repo) | No-Change, Majority-Class, NB, VFDT/Hoeffding, Leveraging Bagging, **ARF**; MLPs via river/deep-river | prequential accuracy (window 1000); κ / **κ_per** available | **YES** — 80 k rows; CPU beats GPU for MLPs at this scale |
| C2 | **Metro Interstate Traffic Volume** | Partial — calendar/weather recurrence is intrinsic but **NOT documented as drift regimes in any stream-learning paper I could find** | 48,204 hourly, 8 feat, 2012–2018 | **CC BY 4.0** (UCI) | GBDT/GRU/Mamba/sliding-window attn (forecasting convention) | MAE/RMSE, prequential MAE; persistence baseline | **YES** |
| C3 | **NOAA Weather (Offutt AFB)** | Seasonal/cyclical drift claimed by Elwell & Polikar; **claim is single-sourced and the supplementary page does not restate it** | 18,159 daily, 8 feat, 2 cls (69 % no-rain) | not stated on the source page (**UNVERIFIED**) | Learn++.NSE, DACC, LVGB, kNN_S, kNN_WA, SAM-kNN | interleaved test-train error | YES |
| C4 | **Electricity / ELEC2** | Seasonality + autocorrelation documented; **regime *revisit* not annotated** | 45,312 × 6–8 feat × 2 cls | Public (OpenML d/151) | as C3, plus 17 published results | prequential accuracy, **κ_per** | YES |
| C5 | **Rialto Bridge Timelapse** | Daily lighting/weather cycle over 20 consecutive days ⇒ intrinsic daily revisit; drift *type* not annotated | 82,250 × 27 feat × 10 cls | not stated (**UNVERIFIED**) | SAM-kNN family | interleaved test-train error | YES |
| C6 | **Airlines (Data Expo 2009)** | **NO** — Souza et al. list it under "uncertainty about changes"; no annotated drift | 539,383 × 7 feat × 2 cls | Public (OpenML d/1169) | ARF, Hoeffding | prequential accuracy | YES |
| C7 | **UCI Household Power** | Diurnal/weekly/seasonal recurrence intrinsic; **no drift-benchmark paper documents it** | 2,075,259 min-level, 9 vars, 2006-12→2010-11, ~1.25 % missing | **CC BY 4.0** | forecasting baselines | MAE/RMSE | YES (subsample) |
| C8 | **NAB / Yahoo S5** | n/a — anomaly detection, not regime revisit | small | mixed | NAB detectors | contested NAB score | YES |

---

## 2. Evidence (with citations I actually fetched)

### 2.1 INSECTS — the recurrence is engineered and the ground truth is published
- Souza, Reis, Maletzke, Batista (2020), *"Challenges in benchmarking stream learning algorithms with real-world data"*, **Data Min. Knowl. Disc. 34(6):1805–1858** (peer-reviewed), arXiv:2005.00113. §5.5 defines the variants verbatim:
  - **Incremental-reoccurring:** *"there exist three cycles of incremental changes over time. In the first cycle, the temperature increases from 20°C to 40°C. In the second cycle, the temperature decreases from 40°C to 20°C. In the end, the temperature turns to increase to 40°C."*
  - **Incremental-abrupt-reoccurring:** *"This pattern provides three recurrent cycles of incremental changes where the temperature increase from 20°C to 40°C. Between the end and beginning of a cycle of incremental changes, we have an abrupt change."*
  - **Table 2 (change points):** Inc-reoccurring (bal.) **79,986** instances, change points **26,568; 53,364**; (imbal.) **452,044**, change points **150,683; 301,365**. Inc-abrupt-reoccurring identical counts. Out-of-control = **905,145**, *"this dataset must be drift-free"* — i.e. a **published null stream in the same feature space** (a gift: a ready-made same-keys/no-drift control).
- **The regime variable is hidden by design** (§5.5): *"we ordered the observations of the examples over time in the stream following different patterns of change in temperature while hiding this variable from the dataset"*, and *"for each temperature, we uniformly sampled examples that were collected within that temperature. As a result, we eliminate all other sources of drift beside the changes in temperature."* All variants: **33 features**, **6 classes** (species × sex), except out-of-control (**24 classes**, with class emergence/disappearance).
- **The regime-conditioning headroom is measured** (§5.6, Table 3/4; Random Forest 200 trees, 10-fold CV, all 33 features): *"The overlap when all temperatures are considered together is 36%, while the average overlap when each temperature is isolated is 23%."* And: *"The single classifier achieves 84% accuracy for the whole data, while individual classifiers average 90%."* At 24 °C the gap is **86 % (per-temperature) vs 66 % (pooled) = 20 points**. **This is the quantity C2W10 is trying to buy with a persistent store**, and it is pre-measured by the dataset's authors.
- **Baselines, verified numbers** (Souza Table 5, prequential accuracy, sliding window of 1,000, MOA implementations):

  | variant | No-Change | Maj.Class | NB | VFDT | Lev.Bag | **ARF** |
  |---|---|---|---|---|---|---|
  | Inc-reoc (bal.) | 40.46 | 16.66 | 48.77 | 47.83 | 72.30 | **77.13** |
  | Inc-reoc (imbal.) | 28.21 | 29.76 | 52.58 | 55.22 | 69.56 | **77.62** |
  | Inc-abrupt-reoc (bal.) | 42.39 | 16.65 | 58.55 | 58.39 | 70.91 | **74.95** |
  | Inc-abrupt-reoc (imbal.) | 28.16 | 29.76 | 52.34 | 51.03 | 69.13 | **77.60** |
  | Out-of-control (null) | 13.06 | 18.80 | 45.99 | 44.70 | 53.58 | **70.45** |

  Souza's own reading: *"For all datasets, we can note that the Adaptive Random Forest (ARF) presented the best overall results… The poor performance of baseline classifiers gives us empirical evidence that undesirable characteristics such as temporal dependence and the prevalence of majority classes are underrepresented in our data."* **That sentence is the criterion-2 clearance.**
- **Download & practicality.** USP Data Stream Repository, https://sites.google.com/view/uspdsrepository — fetched; states *"This repository is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0)"*, with a bulk ZIP and per-dataset CSVs on Google Drive; the paper's footnote adds *"The datasets are encrypted under the following password: DMKD2018"* (applies to the repo's own archive, not to river's mirror). **river** ships 7 variants directly (`river/datasets/insects.py`, fetched): `incremental_reoccurring_balanced` = **79,986 samples, 21,433,047 bytes**, `incremental_abrupt_balanced` = 79,986 / 21,421,452 B, `abrupt_imbalanced` = 355,275 / 94,893,622 B, all **33 features / 6 classes**. ⚠ river does **not** ship the *imbalanced reoccurring* variants — those need the USP repo directly.
- **Neural baselines run on it at laptop scale.** Kulbach, Cazzonelli, Ngo, Le-Nguyen, Bifet (2024), *"A Retrospective of the Tutorial on Opportunities and Challenges of Online Deep Learning"*, ECML-PKDD 2023 post-workshop proceedings, arXiv:2405.17222 — uses the **Insects-abrupt** stream with 1- and 2-hidden-layer MLPs up to 1024 units under **prequential accuracy**, and reports that on the first 10,000 samples *"the runtimes achieved with the CPU are significantly lower for all but the largest network with two hidden layers consisting of 1024"* units (CPU i5-9600K vs RTX 3090). Qualitative finding worth borrowing: *"the deeper network … yields significantly lower accuracy at the start of the stream as well as immediately prior to concept drifts."* ⚠ No numeric accuracies given in the text I fetched.

### 2.2 Electricity/ELEC2 — verified inadmissible (criterion 2)
- Žliobaitė (2013), *"How good is the Electricity benchmark for evaluating concept drift adaptation"*, arXiv:1301.3524 (2-page correspondence + appendix; **not peer-reviewed**, but its numbers are reproduced independently — see below). Verbatim: *"if we test this naive approach on the Electricity dataset it gives much higher 85% accuracy"* where an i.i.d. stream would give **51 %**; *"the more random change alarms the classifier fires, the better the accuracy"*; recommendation: *"we recommend at least comparing the testing accuracies with the accuracy of the moving average of one."* Appendix Table 1 (MOA): moving-average-of-one **85.3**, beaten only by LeveragingBag 88.6 and AdaHoeffdingOptionTree 86.7; EDDM 84.9, OzaBagADWIN 84.5, HoeffdingAdaptiveTree 83.6, DDM 82.7, NaiveBayes 74.2, MajorityClass 57.5. Table 2 (published literature): only DDM 89.6*, Learn++.CDS 88.5, KNN-SPRT 88.0, GRI 88.0, FISH3 86.2, EDDM-IB1 85.7 exceed 85.3.
- **Independently corroborated** by Souza et al. (2020) §4.2 (peer-reviewed): *"For Electricity data, the No-Change classifier shows an accuracy of 85.33%, while the Naive Bayes with DDM achieves only 81.23%."* Same section also kills two neighbours: **Forest Covertype** No-Change **95.07** vs NB-DDM 88.04, and **Poker-hand** No-Change **74.51** vs 61.96 — and reveals that MOA's Poker-hand ordering has temporal dependence absent from the UCI original (*"the No-Change baseline achieves 43% accuracy … the same baseline achieves staggering 75% accuracy on MOA's normalized version"*).
- Stats: 45,312 instances, 2 years, half-hourly; Žliobaitė says **6 input variables** and P(DOWN)=58 %; Souza's Table 1 says **8 attributes**; OpenML d/151 (fetched) lists 8 columns (Date, Day, Period, NSWprice, NSWdemand, VICprice, VICdemand, transfer), licence **Public**, download `https://openml.org/data/v1/download/2419/electricity.arff`. ⚠ The 6-vs-8 discrepancy is a column-counting convention, not a data discrepancy — state which you use.

### 2.3 The metric-native theorem, demonstrated on the real-stream family (criterion 4)
- Losing, Hammer, Wersing (2016), *"KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift"*, **IEEE ICDM 2016, pp. 291–300**, DOI 10.1109/ICDM.2016.0040 (preprint fetched: honda-ri.de/pubs/pdf/3277.pdf). SAM-kNN is *literally* a bounded dual exemplar store: STM (recent window) + LTM (kMeans++-compressed former concepts), distance-weighted Euclidean kNN, hyperparameters **k = 5, L_min = 50, L_max = 5000** (*"We used for all experiments k = 5, Lmin = 50, Lmax = 5000"*), and it is motivated by exactly our target phenomenon: *"One example for such a situation is reoccurring drift, as methods preserving knowledge in this case do not have to relearn former concepts and therefore produce fewer errors."*
- Their real-world Table II: Weather 18,159/8/2; Electricity 45,312/5/2; Cover Type 581,012/54/7; Poker Hand 829,201/10/10; Outdoor 4,000/21/40; Rialto 82,250/27/10. Table IV real-world **average interleaved test-train error** (window 5000): L++.NSE 30.90, DACC 23.21, LVGB 23.50, **kNN_S 18.03**, kNN_WA 20.87, **SAM 15.40**; average ranks 5.33 / 4.17 / 3.17 / **2.33** / 4.00 / **2.00**. Their own comment: *"It is quite surprising that the simple sliding window approach kNN_S performs comparably well or even better than more sophisticated methods such as DACC or L++.NSE."*
- **Reading for us:** on these tabular real streams, a raw-exemplar store queried by Euclidean distance in the *input* space is at or near ceiling. Any CLU store whose keys live in the input metric space on these streams is refuted before it runs (§8.4). ⚠ Per-dataset cells of Table IV were OCR-ambiguous in my extraction; I quote only the aggregate rows and the in-text sentence, both of which parsed cleanly.

### 2.4 Metrics: use κ_per, not accuracy alone
- Žliobaitė, Bifet, Read, Pfahringer, Holmes (2015), *"Evaluation methods and decision theory for classification of streaming data with temporal dependence"*, **Machine Learning 98(3):455–482**, DOI 10.1007/s10994-014-5441-4 (PDF fetched). Definitions, verbatim:
  - Persistent classifier: *ŷ_t = y_{t−1}*; its accuracy `p_per = P(y_t = y_{t−1}) = Σ_i P(y_t=i)P(y_t=i | y_{t−1}=i)` (Eq. 7), degenerating to `Σ_i P(y_t=i)²` under independence (Eq. 8).
  - **Kappa-Temporal:** `κ_per = (p − p_per) / (1 − p_per)` (Eq. 13). *"If the classifier is achieving the same accuracy as the Persistent classifier, then κ_per = 0."*
  - **Combined:** `κ⁺ = sqrt( max(0,κ) · max(0,κ_per) )` (Eq. 14) — *"if any measure is zero or below zero, the combined measure will give zero."*
  - Their recommendation: *"κ_per and κ measures can be seen as orthogonal, since they measure different aspects of performance. Hence, for a thorough evaluation we recommend measuring and combining both."*
  - Also (Prop. 8/§5.2): under temporal dependence, **false drift alarms can raise accuracy**, and drift detectors' statistical guarantees are invalidated. Directly relevant if C2W10's controller has an admission/eviction trigger — a "better" number may be a false-alarm artefact.

---

## 3. §6 admissibility, one criterion at a time

Legend: ✅ pass · ⚠ pass-with-condition · ❌ fail.

### C1 — INSECTS reoccurring (**PRIMARY**)
1. **Strong baselines that do well.** ✅ ARF 74.95–77.62 %, Leveraging Bagging 69–72 %, on 6 classes (chance 16.7 %); MLPs/online-DL are a published, laptop-scale baseline family on the same stream (arXiv:2405.17222). Mamba/GRU/sliding-window attention are all droppable in as sequence models over the 33-D stream. **The competition is present and healthy.**
2. **Real headroom.** ✅ ARF ≈ 77 % vs a per-regime oracle ≈ 90 % (Souza Table 4) — **~13 points of unclaimed regime-conditioning headroom**, and nothing is saturated. Persistence is at **28–42 %**, i.e. the ELEC2 pathology is absent by measurement.
3. **Memory management over time is the difficulty.** ✅ Three cycles over the same hidden temperature sweep with published change points; the *only* source of drift is the hidden regime variable (all others eliminated by uniform within-temperature sampling). Retention across a full cycle (≈26.5 k instances for balanced) is precisely "did the store keep regime-1 knowledge alive through regimes 2–3", not single-shot lookup. The imbalanced variants add class-prior drift on top (P(Y) *and* P(Y|X) both move).
4. **NOT metric-native.** ⚠ **This is the criterion that must be argued and then tested, not assumed.** The favourable structure: the label is *not* a function of the query point alone — the same 33-D region maps to different species at different temperatures (**36 % pooled class overlap vs 23 % per-regime**; 66 % vs 86 % accuracy at 24 °C), and the regime variable is **withheld from the features**, so the addressing information exists only in accumulated stream context. A per-instance Euclidean lookup therefore *cannot* be the ceiling in the way it provably is for Weather/Rialto. **But** I could not find any published SAM-kNN/windowed-kNN number on INSECTS, and §2.3 shows that family beating ensembles on every other real tabular stream. **Binding condition (see §6): run SAM-kNN and kNN_S on inc-reoccurring before the wave is scoped; if the exemplar store lands within noise of ARF, criterion 4 has fired and C1 is out.**
5. **Every lever can be active.** ✅ Learned φ (33-D → store space) is required (raw features are the metric-native trap; a learned φ is the escape and is charter policy §A4.3); learned ψ reads the trajectory; lifetimes/eviction matter because three cycles × capacity pressure is exactly the "k-streams-never-useful → trash" setting; the controller has real admit/evict decisions at the published change points; and the **out-of-control variant is a same-feature-space null** where the correct controller behaviour is "never fire".

**Extra fit for §A21's C2W10 wording ("the store survives document boundaries", "memory across streams").** The 11 INSECTS streams share one 33-D feature space, one 6-class label set, and one hidden regime variable (temperature 20–40 °C). So the persistent-vs-episodic contrast can be run **across streams**, not just within one: train through `incremental` → `abrupt` → `incremental-reoccurring`, and measure whether regimes learned in stream 1 are still live when stream 3 revisits the same temperature band. That is a genuine cross-context revisit protocol with published segment boundaries, and no other candidate offers it.

### C2 — Metro Interstate Traffic Volume (**FALLBACK**)
1. ✅ GBDT / GRU / Mamba / sliding-window attention are all competitive and conventional on hourly traffic. 2. ⚠ With the raw timestamp in the features, calendar regression is near-ceiling — headroom only exists under the **hidden-clock protocol** (drop `date_time`, predict a 24-h-ahead horizon so persistence dies). 3. ✅ under that protocol: the model must infer day-type/season/weather regime from accumulated context. 4. ⚠ Regression over an 8-D input; nearest-neighbour-over-past-windows is a real threat and must be laundered the same way (§5). 5. ✅. **Honest flag: no stream-learning paper documents its regimes** — recurrence is intrinsic (daily/weekly/annual + holidays + weather) and the UCI page states *"holidays included for impacts on traffic volume"*, but the *drift annotation we would rely on is ours, not the literature's*. That is exactly the "informal drift assumptions" failure Souza §4.1 warns about (*"Virtually all publications that present real data make informal assumptions regarding the existence of drift"*). Use only if C1's criterion-4 test fails.

### C3 — NOAA Weather (Offutt AFB)
1. ✅/2. ❌ — everything clusters within ~1.5 points of everything else on the SAM-kNN table (L++.NSE 22.88 … SAM 21.74 … kNN_S 21.53 in that dataset row; row-level OCR caution applies, but the spread is unambiguously tiny). 3. ⚠ 18,159 daily records is small for capacity pressure. 4. ❌ — **windowed kNN is best-or-tied here**; metric-native. 5. ⚠. **Verdict: inadmissible as primary.** Also: the cyclical-drift claim is single-sourced to Elwell & Polikar (2011) *IEEE TNN 22(10):1517–1531*; the authors' own supplementary page (users.rowan.edu/~polikar/nse.html) says only that it hosts *"the preprocessed Offutt Air Force Base in Bellevue, Nebraska dataset used in the paper"* and **does not restate any recurring-drift claim, nor a licence**. I did not obtain the paper's full text — the "cyclical drift" attribution is **search-summary-level, not primary-verified**.

### C4 — Electricity / ELEC2
1. ⚠ 2. ❌❌ **persistence at 85.3 % is at ceiling** 3. ❌ (the "difficulty" is autocorrelation, which is not memory management) 4. ❌ (kNN family competitive; see §2.3) 5. ⚠. **Verdict: inadmissible.** Keep it only as a *sanity/diagnostic* stream, and only ever reported with κ_per alongside — never as a claim venue.

### C5 — Rialto Bridge Timelapse
2. ⚠ / 4. ❌ — SAM-kNN and kNN_S dominate; 27-D RGB histograms are the definition of a metric-native key space. **Inadmissible as primary.** Licence unstated on the source (**UNVERIFIED**).

### C6 — Airlines
3. ❌ / criterion "documented regime revisit" ❌ — Souza et al. classify it among datasets where *"the type of change …, pattern (abrupt, gradual, incremental, or reoccurring) and the exact moment these drifts occurred are frequently unknown."* No revisit ground truth ⇒ no retention metric can be defined. **Out.**

### C7 — UCI Household Power Consumption
Same shape as C2 (intrinsic calendar recurrence, no documented drift regimes), plus 2.07 M minute-level rows and ~1.25 % missing values. Strictly dominated by C2 as a fallback (smaller, cleaner, hourly, already weather-featured). **Out, but a scale-stress option.**

### C8 — NAB / Yahoo S5
Anomaly detection, not regime revisit; **criterion 3 fails on task type**. The NAB scoring function is additionally contested (Singh & Olinsky, *"Demystifying Numenta anomaly benchmark"*, IJCNN 2017, DOI 10.1109/IJCNN.2017.7966038) — ⚠ I verified the citation's existence and venue only, **not** the paper's contents (abstract not fetched). **Out.**

---

## 4. Recommendation

### PRIMARY — INSECTS `incremental-reoccurring` (balanced **and** imbalanced), with `incremental-abrupt-reoccurring` as the second condition and `out-of-control` as the drift-free null
This is the only real stream I verified where the *recurrence is ground truth rather than an assumption*, the *regime variable is hidden from the query* (so the store's job is context accumulation, not input-space lookup), and the *headroom for regime-conditioning is pre-measured by the dataset's own authors* (pooled 84 % vs per-regime 90 %, and 66 % vs 86 % in the hardest band). For a retention-and-adaptation-at-matched-bytes protocol it gives us everything the harness needs for free: published change points at 26,568 / 53,364 (bal.) and 150,683 / 301,365 (imbal.) define the revisit boundaries, so **retention** = accuracy in the first N instances after re-entering a previously-seen temperature band (relative to accuracy at the *end* of the band's first visit) and **adaptation** = the recovery curve's time constant after each change point — both measurable per-arm at identical state-bytes. The imbalanced variants move P(Y) as well as P(Y|X), which is where an episodic store should visibly fail and a persistent one should not. The 6-class, 33-feature, 80 k-row scale means one full prequential pass is minutes on CPU, so multi-seed and a byte-sweep are affordable; the `out-of-control` variant (905 k instances, same feature space, *"must be drift-free"*) is a published null on which the correct behaviour of a persistent store is *no benefit*, giving the wave a pre-registered negative it is supposed to reproduce. Finally, the family solves C2W10's "across streams" wording that no single-stream candidate can: eleven streams share one feature space, one label set and one hidden regime axis, so the persistent-vs-episodic contrast can be run across stream boundaries with the regimes genuinely revisiting.

### FALLBACK — Metro Interstate Traffic Volume, hidden-clock, 24-h horizon
If C1 fails its criterion-4 pre-test (§6), the fallback deliberately changes *domain, task type and failure mode* rather than picking a sibling: hourly traffic regression on 48,204 records (2012–2018, CC BY 4.0), with `date_time` withheld and a multi-step-ahead horizon so that persistence — the baseline that destroyed ELEC2 — cannot win. The recurrence is dense and multi-scale (daily, weekly, holiday, seasonal, weather-regime), which is exactly the capacity-pressure profile C2W8's consolidation machinery and C2W10's trash plumbing are built for: many regimes, each returning, none affordable to keep verbatim. Strong baselines are unambiguous and well-tuned by convention (GBDT, GRU, Mamba, sliding-window attention), so criterion 1 is not in doubt. Its honest cost is criterion-3 documentation: **no peer-reviewed stream-learning paper annotates its regimes**, so we would be asserting the drift structure ourselves — which means the wave must *first* publish a drift map of the stream (Webb-style drift-magnitude between windows, as Souza §4.1 recommends) before any retention claim is made on it, and must state in the paper that the regime annotation is ours.

---

## 5. The matched-state-bytes protocol this shortlist hands you (free, from the literature)

The wave needs "CLU store vs baseline learners at equal state budget". The exemplar-store baseline and its byte budget already exist and are published:
- **SAM-kNN at its published defaults** is a bounded dual exemplar store with **L_max = 5,000 stored examples** (STM+LTM combined), k=5, L_min=50. On INSECTS (33 × float32 + 1 label byte) that is **5,000 × 133 B ≈ 665 kB ≈ 0.63 MiB of state** — a hard, defensible, *published-by-someone-else* budget to match the CLU store against.
- **kNN_S** (plain distance-weighted kNN over a 5,000 sample sliding window) is the *episodic* store at the same byte budget — i.e. the literature's own persistent-vs-episodic ablation, which is C2W10's exact contrast. Losing et al. also report a 1,000-sample variant, giving a second budget point ≈ 133 kB for free.
- Both are in **river** (`river.neighbors.SAMKNNClassifier` family) alongside ARF and the Insects loaders, so the launder arm is a one-line baseline, not an engineering project.
- **Metrics:** report prequential accuracy (window 1,000, matching Souza's convention so our numbers are comparable to Table 5) **and κ_per and κ⁺** (Žliobaitė et al. 2015, Eqs. 13–14), plus the two retention/adaptation curves. The persistence baseline must appear in every table — that is Žliobaitė's explicit recommendation and it is what kills ELEC2-style self-deception.

**Criterion-4 tripwire, stated in advance:** if the 0.63 MiB exemplar store (SAM-kNN) matches or beats the CLU store at the same bytes on revisit-recovery, the metric-native theorem has fired on this benchmark too and the result is reported as such — not re-framed.

---

## 6. Confidence & gaps (verified vs unverified, per candidate)

| item | status |
|---|---|
| INSECTS variant definitions, change points, instance counts, 33 feat / 6 cls, hidden temperature, overlap 36 % vs 23 %, 84 % vs 90 %, Table 5 baseline accuracies | **VERIFIED** — read directly from the arXiv:2005.00113 PDF (pages 34–40), peer-reviewed DMKD version |
| USP repo licence CC BY 4.0 + Google-Drive distribution; paper's `DMKD2018` archive password | **VERIFIED** (site fetched; footnote read in PDF) |
| river variant sample counts + byte sizes; absence of imbalanced-reoccurring in river | **VERIFIED** (`river/datasets/insects.py` fetched) |
| Žliobaitė 2013 ELEC2 critique: 85 % persistence, 51 % i.i.d., MOA + literature tables | **VERIFIED** (full text read) **and independently corroborated** by Souza §4.2 (85.33 % vs 81.23 %) |
| SAM-kNN architecture, L_max=5000/k=5/L_min=50, reoccurring-drift motivation, real-world aggregate errors + ranks, "kNN_S surprisingly competitive" | **VERIFIED** (ICDM preprint read). ⚠ **per-dataset cells of Table IV are OCR-ambiguous** — do not quote individual cells without re-reading the PDF |
| κ_per, κ⁺, persistent-classifier formulas, false-alarm/drift-detector propositions | **VERIFIED** (Machine Learning 2015 PDF, pages 8–14) |
| Electricity OpenML licence "Public", 45,312 × 8, download URL; Airlines OpenML "Public" | **VERIFIED** (OpenML API fetched). ⚠ 6-vs-8 feature-count convention differs between sources |
| Metro Interstate: 48,204 rows, 8 feat, 2012–2018, CC BY 4.0, donated 2019-05-06 | **VERIFIED** (UCI page fetched). ⚠ **Its regime/drift structure is UNVERIFIED and undocumented in the literature** |
| UCI Household Power: 2,075,259 rows, 9 vars, CC BY 4.0, ~1.25 % missing | **VERIFIED** (UCI page fetched) |
| NOAA Weather cyclical/seasonal drift claim (Elwell & Polikar 2011) | **UNVERIFIED at primary level** — search-summary only; the authors' supplementary page does not restate it, and states **no licence** |
| Rialto/Outdoor licences | **UNVERIFIED** |
| NAB critique (Singh & Olinsky, IJCNN 2017) contents | **UNVERIFIED** — citation/venue confirmed, paper not read |
| ⛔ **SAM-kNN / kNN_S performance on INSECTS** | **NOT FOUND anywhere.** This is the single decisive gap: it is the criterion-4 test for the PRIMARY recommendation |

**The one thing to do before the wave is scoped (≈1 h, laptop, no worktree).** Run `river`'s SAM-kNN and a plain windowed kNN (L_max = 5000 and 1000) against ARF on `insects incremental_reoccurring_balanced`, prequential window 1000, and report accuracy + κ_per. Pre-register the reading now: **if the exemplar store lands within ~2 points of ARF, INSECTS is metric-native at the input level and the PRIMARY must be re-argued** (either by proving the CLU's φ moves the keys out of the input metric — the store holds regime-conditional predictive state, never exemplars — or by switching to the fallback). If the exemplar store lands clearly *below* ARF (the pattern would then be the *opposite* of every other real tabular stream in §2.3, which is itself a publishable observation), criterion 4 is cleared on evidence rather than on argument.

**What I'd search next if the Hub wants more:** (i) any 2023–2026 paper reporting SAM-kNN on INSECTS (I found none; OEBench arXiv:2308.15059 collects 55 real relational streams and runs ARF/SEA/EWC/LwF/iCaRL but **does not separate reoccurring drift** — worth a deeper read if a second real stream is needed); (ii) Katakis et al.'s recurring-context email streams (Spam Assassin, Usenet1/2) as a *text-domain* second stream with documented user-interest recurrence — Souza's Table 1 confirms Spam Assassin at 9,324 × 97,851 × 2 and describes *"the user regaining interest in topics that he has been previously interested in"*, but I did not fetch the Katakis primary source; (iii) whether any Titans/TTT-class paper has been run on a drift-annotated real stream (would set the modern-memory-family comparison bar for the paper's rival row).

---

## 7. Bibtex-ready

```bibtex
@article{souza2020challenges,
  author  = {Souza, Vinicius M. A. and dos Reis, Denis M. and Maletzke, Andre G. and Batista, Gustavo E. A. P. A.},
  title   = {Challenges in benchmarking stream learning algorithms with real-world data},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {34}, number = {6}, pages = {1805--1858}, year = {2020},
  doi     = {10.1007/s10618-020-00698-5}, eprint = {2005.00113}, archivePrefix = {arXiv}
}
@article{zliobaite2013electricity,
  author = {\v{Z}liobait\.{e}, Indr\.{e}},
  title  = {How good is the Electricity benchmark for evaluating concept drift adaptation},
  year   = {2013}, eprint = {1301.3524}, archivePrefix = {arXiv}, primaryClass = {cs.LG}
}
@article{zliobaite2015evaluation,
  author  = {\v{Z}liobait\.{e}, Indr\.{e} and Bifet, Albert and Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff},
  title   = {Evaluation methods and decision theory for classification of streaming data with temporal dependence},
  journal = {Machine Learning}, volume = {98}, number = {3}, pages = {455--482}, year = {2015},
  doi     = {10.1007/s10994-014-5441-4}
}
@inproceedings{losing2016samknn,
  author    = {Losing, Viktor and Hammer, Barbara and Wersing, Heiko},
  title     = {{KNN} Classifier with Self Adjusting Memory for Heterogeneous Concept Drift},
  booktitle = {2016 IEEE 16th International Conference on Data Mining (ICDM)},
  pages     = {291--300}, year = {2016}, doi = {10.1109/ICDM.2016.0040}
}
@article{elwell2011nse,
  author  = {Elwell, Ryan and Polikar, Robi},
  title   = {Incremental Learning of Concept Drift in Nonstationary Environments},
  journal = {IEEE Transactions on Neural Networks},
  volume  = {22}, number = {10}, pages = {1517--1531}, year = {2011},
  doi     = {10.1109/TNN.2011.2160459}
}
@article{kulbach2024onlinedl,
  author = {Kulbach, Cedric and Cazzonelli, Lucas and Ngo, Hoang-Anh and Le-Nguyen, Minh-Huong and Bifet, Albert},
  title  = {A Retrospective of the Tutorial on Opportunities and Challenges of Online Deep Learning},
  year   = {2024}, eprint = {2405.17222}, archivePrefix = {arXiv},
  note   = {ECML-PKDD 2023 joint post-workshop proceedings}
}
@misc{uci_metro_traffic,
  title = {Metro Interstate Traffic Volume},
  howpublished = {UCI Machine Learning Repository, dataset 492},
  note = {CC BY 4.0; donated 2019-05-06}, doi = {10.24432/C5X60B}
}
```
⚠ The Metro Interstate DOI above is the UCI-convention identifier shown on the dataset page pattern; **treat it as unverified** — I confirmed the dataset page, licence and donation date but did not separately resolve the DOI.

---

## Proposed handover updates (for the Hub)

- **C2W10 benchmark shortlist landed.** PRIMARY = INSECTS `incremental-reoccurring` + `incremental-abrupt-reoccurring` (bal. + imbal.), USP DS Repository / river, CC BY 4.0; `out-of-control` (905,145) is the published drift-free null in the same feature space. FALLBACK = Metro Interstate Traffic Volume under a hidden-clock, 24-h-horizon protocol, with the honest caveat that its regime annotation would be ours.
- **Two verified exclusions to record in `negative_results.md`:** (N-new-a) *Electricity/ELEC2 is inadmissible under §6.2* — persistence scores 85.3 % and beats 10/12 MOA adaptive classifiers (Žliobaitė 2013; corroborated 85.33 % vs 81.23 % by Souza et al. 2020 §4.2). Same source kills Forest Covertype (No-Change 95.07) and MOA-Poker-hand (74.51). (N-new-b) *The Losing/SAM-kNN real-stream family (NOAA-Weather, Rialto, Outdoor, Cover Type) fails §6.4* — a plain 5,000-example windowed kNN is 2nd of 6 on real-world average rank (2.33) and an explicit dual exemplar store is 1st (2.00). This is the metric-native theorem confirmed **in the literature, on the exact family we were about to shop in** — a fifth confirmation, obtained without spending a wave.
- **A gift for the byte ledger:** the matched-state-bytes launder for C2W10 is off-the-shelf — SAM-kNN's published defaults (L_max = 5,000 exemplars ⇒ **≈0.63 MiB** on INSECTS' 33 float32 features) supply the budget, and kNN_S at the same budget *is* the episodic-store arm of the persistent-vs-episodic contrast. Second budget point at 1,000 exemplars (≈133 kB) also published.
- **Metric ruling requested:** adopt **prequential accuracy (window 1,000, Souza convention) + κ_per + κ⁺** (Žliobaitė et al. 2015, Eqs. 13–14) as C2W10's reporting triple, with the persistence baseline mandatory in every table. Note Prop. 8/§5.2: under temporal dependence, *false* drift alarms can raise accuracy and drift-detector guarantees are void — relevant to any controller admission/eviction trigger we report.
- **One gating pre-run before the wave is scoped (≈1 h, CPU, no worktree, prereg-able in three lines):** SAM-kNN + kNN_S (L_max ∈ {5000, 1000}) vs ARF on `insects incremental_reoccurring_balanced`. **Pre-registered reading:** exemplar store within ~2 pts of ARF ⇒ criterion 4 has fired on the PRIMARY and it must be re-argued or swapped to the fallback; exemplar store clearly below ARF ⇒ criterion 4 cleared on evidence (and, since that inverts the pattern of §2.3, it is itself a reportable observation).
- **Reconciliation list (owner needed):** nothing in the existing registries contradicts this brief, but if any prior wave doc names Electricity, NOAA-Weather or Rialto as a candidate stream, those mentions now carry the §6.2/§6.4 exclusions above and should be struck at the next curator pass.
