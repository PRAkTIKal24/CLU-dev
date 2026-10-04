# bprime-r6-cite-check — web-scout report
Task + acceptance criterion: verify the six citations the r6 fold introduced as UNVERIFIED (Souza · Losing · Gomes/ARF · river · UCI Metro · Webb 2016) against canonical primary sources; deliver record + Appendix-Q-pattern BibTeX + usage-check + flags for each.
Status: **done** — 6 of 6 verified against primary sources (publisher record / arXiv PDF read directly / author-preprint PDF read directly / UCI page + DOI resolution / official library docs + source tree). **0 could-not-verify.** 3 usage qualifiers owed; **1 is a headline finding.**

**DIAL DECLARATION (protocol §7, echoed): none — citation verification. No performance claim, no laundering control, nothing falsifiable in the dial sense.** Nothing here touches admission · lifetimes · isolation · compute-adaptive reads.

## ⛔ DOWNSTREAM RECONCILIATION LIST (owner needed — protocol §5 corollary, in the first 10 lines)
1. ⭐⭐ **HEADLINE — the Webb attribution needs one clause.** §R.2.2 says *"total-variation drift magnitude **in the sense of Webb et al., 2016**"*. Webb et al. **deliberately leave the distance function unspecified** (their Eq. 5) and **use Hellinger distance in their own case study** — total variation is *our* instantiation, not theirs. The framework attribution is sound; the *distance choice* must not read as theirs. **Owner: paper-writer, next revision.**
2. ⚠ **The published `77.13` ARF anchor is a MOA number and its tree count is not stated in the source.** Souza et al. ran everything in MOA and never print ARF's ensemble size; 100 is MOA's *default*, verified from MOA source. Our 78.8139 is `river.forest.ARFClassifier`. §R.2.1 / A.6 say *"ours is the stronger implementation"* without ever naming **MOA vs river**. **Owner: paper-writer.**
3. ⚠ **`a permitted 3,000` (SAM-kNN STM cap, §R.2.1) is not a published number.** It derives from `LTMSizeProportion = 0.4` (⇒ STM cap = 5000 − 2000), which appears in the **authors' reference implementation / scikit-multiflow**, **not** in the ICDM paper. The paper publishes only `k = 5, L_min = 50, L_max = 5000 (STM and LTM combined)`. The `0.634 MiB` budget itself **is** published and is clean. **Owner: paper-writer (A.6 label).**
4. ✅ **Discharge to record:** the r5-era scout's standing flag *"the Metro Interstate DOI … treat it as unverified"* (`c2w10-benchmark-scout.md` l.203) is **now discharged** — `10.24432/C5X60B` resolves 302 → `archive.ics.uci.edu/dataset/492`. **Owner: curator (one-line erratum on that report).**
5. Editorial item **10** of the r6 changelog ("⛔ None of these is in Appendix Q; a cite-check pass is owed before circulation") is **satisfied by this report** and must be rewritten at the fold. **Owner: paper-writer.**

---

# 1. Method + retrieval dates

All retrievals **2026-08-18**. Rule applied: publisher/arXiv/official-docs primary, aggregators only as corroboration, never alone.

| source | how verified | primary? |
|---|---|---|
| Souza et al. | **arXiv:2005.00113v2 PDF read directly** (pp. 33–40: Table 2, Table 5, §7.1 protocol text) + DBLP record for venue/vol/pages/DOI | ✅ full text |
| Losing et al. | **author preprint PDF read directly** (honda-ri.de/pubs/pdf/3277.pdf, 11 pp — Table III, Table IV, §IV-A, §VI) + DBLP for pages/DOI | ✅ full text |
| Gomes et al. | DBLP + MOA source javadoc (`AdaptiveRandomForest.java`) + HAL/IP-Paris correction record | ◐ record only, body not read |
| river | riverml.xyz **0.25.0** API overview + `github.com/online-ml/river/tree/main/river/neighbors` + PyPI 0.25.0 + JMLR v22 page | ✅ two independent |
| UCI Metro | `archive.ics.uci.edu/dataset/492` **+ DOI resolution test** | ✅ + DOI resolves |
| Webb et al. | **arXiv:1511.03816v6 PDF read directly** (pp. 6–7: Eqs. 5–6 and the distance-function paragraph) | ✅ full text |

⚠ Springer `link.springer.com` is SSO-gated on this machine (303 → `idp.springer.com`) for **both** DMKD papers; ACM DL returns 403. Venue/volume/pages/DOI for Souza and Webb are therefore sourced from **arXiv `journal-ref` + DBLP**, two independent records that agree exactly. Publisher-side page images **not** read — flagged per record below.

---

# 2. Per-citation sections

## 2.1 Souza et al. — the INSECTS benchmark ⭐ strongest record of the six

**(a) Canonical record.** Vinicius M. A. Souza, Denis M. dos Reis, Andre G. Maletzke, Gustavo E. A. P. A. Batista, *"Challenges in Benchmarking Stream Learning Algorithms with Real-world Data"*, **Data Mining and Knowledge Discovery 34(6):1805–1858, 2020**, DOI `10.1007/s10618-020-00698-5`, arXiv:2005.00113 (v1 2020-04-30, v2 2020-06-30). Peer-reviewed journal. arXiv comments field: *"Preprint of article accepted for publication in the journal Data Mining and Knowledge Discovery"*. Editorial item 10's *"year and venue not verified"* is now **closed: 2020, DMKD.**

**(c) Usage-check — every content-lean holds, verbatim.**

| draft site | draft's sentence | source's own words / number | verdict |
|---|---|---|---|
| A.6 | *"prequential accuracy, window 1000 (**the benchmark authors' convention**)"* | §7.1: *"We consider prequential evaluation (Gama et al., 2013) over a sliding window of **1,000 instances** to evaluate the classification performance of the algorithms."* | ✅ exact |
| A.6 | *"index 0 excluded ⇒ n_scored = **79,985**"* | Table 2: `Incremental-reoccurring (bal.)` = **79,986** instances; change points **26568; 53364** | ✅ 79,986 − 1 |
| §R.2.1 | *"No-Change **40.4526** vs the published **40.46**"* | Table 5, row `Inc-reoc (bal.)`, col `No-Change` = **40.46** | ✅ exact |
| §R.2.1 | *"vs the published **77.13**"* | Table 5, same row, col `ARF` = **77.13** | ✅ exact |
| §R.2.1 | *"a 500-example window spans ≈ 0.13 °C of **the venue's 20 °C temperature sweep**"* | §5.5, Incremental-reoccurring: *"the temperature increases from **20°C to 40°C**"* (three cycles) | ✅ a 20 °C span |
| §R.2.1 | *"the published **state-of-the-art ensemble**"* (of ARF 77.13) | §7.1: *"For all datasets, we can note that the **Adaptive Random Forest (ARF) presented the best overall results**, followed by the Leveraging Bagging"* | ◐ see flag F1 |
| §R.2.1 / App. J | *"The venue's own **drift-free-null** stream has no data source and is a declared NOT-RUN"* | §5.5, Out-of-control: *"As each example is sampled uniformly sampled at each time during the stream, this dataset **must be drift-free**."* | ✅ the "drift-free" label is the authors' own |
| A.6 | *"second condition `incremental-abrupt-reoccurring` (balanced)"* | Table 2 row exists, 79,986 instances, change points 26568; 53364; Table 5 ARF = **74.95** | ✅ |

**(d) Flags.**
- **F1 (wording, minor).** *"the published state-of-the-art ensemble"* — Souza's ARF is *"the best overall results"* **among the six methods they ran** (No-Change, Majority-Class, NB, VFDT, Leveraging Bagging, ARF), not a literature-wide SOTA claim. Safe fix: *"the venue's best published method"*.
- **F2 (author-name trap, never-copy).** **DBLP renders the first and third authors as "Vinícius M. A. *de* Souza" and "André *Gustavo* Maletzke"; the paper's own title page and running head print "Vinicius M. A. Souza … Andre G. Maletzke" and "*V. M. A. Souza et al.*"** ⇒ the short form is **"Souza et al."**, **never "de Souza et al."**. Do not copy DBLP's rendering into the in-text cite.
- **F3.** Publisher-side pagination (1805–1858) is arXiv `journal-ref` + DBLP, **two records that agree**; the Springer page itself is SSO-blocked and was not read. Not single-sourced, but not publisher-confirmed either.
- **F4 (⚠ feeds reconciliation item 2).** §7 states *"We run all experiments … using the **MOA** framework"*. The 77.13 is a **MOA** number. Souza et al. **never state ARF's ensemble size** anywhere in §7.1 or Table 5 — the "100 trees" reading is an inference from MOA's default, which I verified independently: `moa/classifiers/meta/AdaptiveRandomForest.java` declares `new IntOption("ensembleSize", 's', "The number of trees.", **100**, 1, Integer.MAX_VALUE)`. The inference is sound; it is still an inference and should be labelled.
- No trap found on the title's capitalisation; arXiv and DBLP agree on wording.

## 2.2 Losing, Hammer & Wersing — SAM-kNN ⭐ every quoted number verified from the authors' own Table IV

**(a) Canonical record.** Viktor Losing, Barbara Hammer, Heiko Wersing, *"KNN Classifier with Self Adjusting Memory for Heterogeneous Concept Drift"*, **2016 IEEE 16th International Conference on Data Mining (ICDM), pp. 291–300**, DOI `10.1109/ICDM.2016.0040`. Affiliations on the paper: Bielefeld University (Losing, Hammer) and HONDA Research Institute Europe (Losing, Wersing). Author preprint: `honda-ri.de/pubs/pdf/3277.pdf`, whose cover states *"This is an accepted article published in IEEE International Conference on Data Mining. The final authenticated version is available online at: https://doi.org/10.1109/ICDM.2016.0040."*

**(c) Usage-check.**

| draft site | draft's sentence | source's own words / number | verdict |
|---|---|---|---|
| §R.2.1 | *"SAM-kNN at **its own published 0.634 MiB budget** … at 665,000 B"* | §IV-A: *"We used for all experiments **k = 5, L_min = 50, L_max = 5000**."*; L_max defined as *"The maximum number of stored examples L_max (**STM and LTM combined**)"*; footnote 4: *"Regarding our approach, the available space is **shared** between the STM and LTM."* ⇒ 5,000 exemplars × 133 B = 665,000 B = 0.6342 MiB | ✅ the budget **is** published |
| A.6 | *"a second budget point … 133,000 B"* | §VI: *"Window based approaches were allowed to store 5000 samples (**we also report results for a size of 1000 samples**)"*; Table III lists `L_max = 5000, k = 5` for KNN_S, KNN_WA and SAM | ✅ |
| A.6 | *"validated against the authors' published Weather row, 21.70 / 21.68 % error vs published **21.74 / 21.53**"* | Table IV (*"Interleaved Test-Train **error rates**"*), window-size-5000 block, `Weather` row: `kNN_S` **21.53** (bold = lowest), `SAM` **21.74** | ✅ exact, and the draft correctly says **error**, not accuracy |
| A.6 / §R.2.1 | *"our own port (**the streaming library ships none**)"* | — (verified against river, §2.4) | ✅ |
| §R.2.2 / A.6 | *"our `knnsam` regression adaptation is **ours, not a published algorithm**"* | SAM-kNN is a **classifier** throughout; the paper's only forward-look is *"a combination with alternative models such as … incremental/decremental SVMs"* — no regression variant | ✅ the disclaimer is correct and necessary |
| §R.2.1 | *"SAM-kNN **discovers this itself** (its short-term memory averages 945 …)"* | §IV-B1: *"we do not explicitly detect a concept change, but instead we adjust the size such that the Interleaved Test-Train error of the remaining STM is minimized"* | ✅ mechanism claim supported |
| §R.2.1 | *"… of **a permitted 3,000** exemplars"* | **NOT IN THE PAPER** — see flag F5 | ⚠ label needed |

**(d) Flags.**
- **F5 (⚠ reconciliation item 3).** The `3,000` STM cap comes from `LTMSizeProportion = 0.4` in our harness (`c2w10-benchmark-gate.md` l.30), i.e. STM cap = 5000 − 0.4·5000 = 3,000. **The ICDM paper contains no LTM-size-proportion parameter at all** — its only LTM sizing statement is §IV-B3: *"We use the clustering algorithm kMeans++ with |M_LTĉ|/2 clusters"* (halving on overflow). `0.4` is the **authors' reference-implementation / scikit-multiflow** default: `SAMKNNClassifier(n_neighbors=5, weighting='distance', max_window_size=5000, **ltm_size=0.4**, min_stm_size=50, stm_size_option='maxACCApprox', use_ltm=True)`. Consequence: the gate report's *"Losing et al. ICDM 2016 published defaults, **unmodified**"* is **over-stated for `LTMSizeProportion` only**; `k`, `L_min`, `L_max`, `knnWeights='distance'` are all genuinely published. The draft's own text is safe (it says *"our own port"*); only the word *"permitted"* implies a published cap.
- **F6 (title trap, never-copy).** The **HRI cover page misspells the title as "heterog*e*nous"**; the paper's own title page, DBLP and IEEE all read **"Heterogeneous"**. Use the paper's spelling.
- **F7.** IEEE Xplore camera-ready **not read** (paywalled/SSO). Pages 291–300 and the DOI come from DBLP; all *content* quotes above are pinned to the **author preprint**, which self-declares as the accepted version. Any future quote should carry that pin, exactly as the Chesetti & Pandey entry does in Q.1.
- **F8 (context that must travel with the Weather row).** Table IV reports **Interleaved Test-Train error**, and `kNN_S` (21.53) **beats** SAM (21.74) on Weather — i.e. the authors' own table already shows a plain window beating the dual memory on that dataset. This is *helpful* to §R.2.2's finding (`knnsam` 325.71 loses to the plain window 320.98) and should arguably be cited as prior corroboration rather than left silent.

## 2.3 Gomes et al. — Adaptive Random Forests ◐ record verified, body not read

**(a) Canonical record.** Heitor Murilo Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal, Fabrício Enembreck, Bernhard **Pfahringer**, Geoff Holmes, Talel Abdessalem, *"Adaptive random forests for evolving data stream classification"*, **Machine Learning 106(9–10):1469–1495, 2017**, DOI `10.1007/s10994-017-5642-8`. ⚠ **A correction exists:** *"Correction to: Adaptive random forests for evolving data stream classification"*, Machine Learning **108(10):1877–1878, 2019**, DOI `10.1007/s10994-019-05793-3`.

**(c) Usage-check.** The draft **quotes no number from Gomes et al.** — every ARF number in §R.2.1 / A.6 is either ours (`78.8139 ± 0.0526`, `77.4129`, `9,542,925 B`) or **Souza's measurement of ARF** (77.13 / 74.95). This is the cleanest of the six: the citation carries **algorithm identity only**, and Souza et al. themselves cite it exactly that way (§7.1: *"Adaptive Random Forest (Gomes et al., 2017)"*). ✅ Nothing to qualify at the source level.

**(d) Flags.**
- **F9 (author-order trap, never-copy).** At least one aggregator search surface renders this as *"Albert Bifet, Heitor Gomes, …"*. **The canonical first author is GOMES.** Double-sourced: DBLP, and the MOA class javadoc for `AdaptiveRandomForest.java`, which prints the full list in order.
- **F10 (spelling trap, never-copy).** **MOA's own javadoc and the HAL surface misspell "Pfharinger"; the correct name is "Pfahringer"** (DBLP). Do not copy the MOA string into BibTeX.
- **F11.** The **2019 correction is not incorporated into our reading** — the paper body was not read, so we do not know what it corrects. Since we quote nothing from the paper, this is inert; if a later revision quotes an ARF number, the correction must be read first.
- **F12 (⚠ reconciliation item 2).** Nothing in Gomes et al. licenses treating our `river` ARF-100 and Souza's MOA ARF as the same object. `c2w10-benchmark-gate.md` §282 states it plainly (*"river ARF ≠ MOA ARF … they are different implementations"*); **the draft does not**. One clause in §R.2.1 fixes it.

## 2.4 `river` — the streaming library ✅ the load-bearing negative is double-sourced

**(a) Canonical record.** Software: **river 0.25.0**, released **2026-05-31** (PyPI), BSD-3-Clause, maintainer Max Halford, `riverml.xyz`. Paper of record: Jacob Montiel, Max Halford, Saulo Martiello Mastelini, Geoffrey Bolmier, Raphael Sourty, Robin Vaysse, Adil Zouitine, Heitor Murilo Gomes, Jesse Read, Talel Abdessalem, Albert Bifet, *"River: machine learning for streaming data in Python"*, **JMLR 22(110):1–8, 2021**, arXiv:2012.04740. (JMLR is open-access; the volume page was read directly.)

**(c) Usage-check — the one claim that matters is the negative.**

> A.6: *"SAM-kNN (⚠ **our own port; the streaming library ships none**, so every 'one-line baseline' cost estimate is void…)"*

**VERIFIED, two independent sources.** (i) `riverml.xyz/0.25.0` API overview — the entire `neighbors` module is **`KNNClassifier`, `KNNRegressor`, `LazySearch`, `SWINN`**; no SAM / self-adjusting-memory estimator anywhere in the module. (ii) `github.com/online-ml/river/tree/main/river/neighbors` — files are `ann/`, `__init__.py`, `base.py`, `knn_classifier.py`, `knn_regressor.py`, `lazy.py`; no SAM implementation. Corroborating provenance for *why*: **`SAMKNNClassifier` lives in `skmultiflow.lazy`**, one of the two ancestors river merged (creme + scikit-multiflow); it was **not carried over**. ✅ The draft's *"every cost estimate premised on a 'one-line baseline' is void"* is correct, and it correctly retires the earlier internal claim (`c2w10-benchmark-scout.md` l.126: *"Both are in **river** (`river.neighbors.SAMKNNClassifier` family)… the launder arm is a one-line baseline"*) — that line is **wrong** and was already corrected by the gate report.

Also verified: `river.forest.ARFClassifier` **does** exist in 0.25.0 (the `forest` module ships `AMFClassifier, AMFRegressor, ARFClassifier, ARFRegressor, OXTRegressor`), and `datasets` ships an `Insects` loader — both consistent with A.6's harness description.

**(d) Flags.**
- **F13.** PyPI's own metadata block prints an **incomplete** BibTeX (`author = {… and others}`, no journal/volume/pages). Do **not** copy it. The JMLR record (22(110):1–8) is the one to print.
- **F14 (version trap).** The draft correctly pins **`river 0.25.0`**. `riverml.xyz/0.25.0/` and PyPI 0.25.0 both exist and agree; a stale search surface asserted "0.25.0 released 2022-02-04", which is **wrong** (that surface is confusing river-the-ML-library with unrelated projects named `river`). Use the PyPI date: **2026-05-31**.
- **F15.** river is cited as *software actually run*, not as a claim source — so a `@misc` software entry **plus** the JMLR paper is the right pair; printing only the paper would misdescribe what was used.

## 2.5 UCI — Metro Interstate Traffic Volume ✅ DOI now resolves (prior flag discharged)

**(a) Canonical record.** *"Metro Interstate Traffic Volume"*, donated by **John Hogue**, **2019-05-06**, **UCI Machine Learning Repository dataset 492**, DOI **`10.24432/C5X60B`**, licence **CC BY 4.0**, **48,204 instances**, **8 features**, multivariate/sequential/time-series, regression task. Description: *"Hourly Minneapolis-St Paul, MN traffic volume for westbound I-94"*, 2012–2018, with weather and holiday features. UCI's own suggested citation: **"Hogue, J. (2019). Metro Interstate Traffic Volume [Dataset]. UCI Machine Learning Repository."**
⭐ **DOI resolution tested:** `https://doi.org/10.24432/C5X60B` → **302** → `https://archive.ics.uci.edu/dataset/492`. The r5-era *"treat it as unverified"* caveat is **discharged**.

**(c) Usage-check.** The draft leans on the dataset only for identity and shape — hourly, 2012–2018, weather + holiday features, `date_time` withheld ("hidden clock"). All ✅ against the UCI page.

**(d) Flags.**
- **F16 (apparent-contradiction trap — record it, don't fix the draft).** The UCI page states **"Has Missing Values? **No**"**, while A.6 declares *"gap-fill ≤ 3 h on features only (6.4 % of feature hours)"* and *"a 7,386-hour (10-month) sensor hole"*. **These do not contradict:** UCI means no missing *cells within the 48,204 delivered rows*; our gaps are missing *hours* on the reconstructed hourly grid (2012-10 → 2018-09 spans ≈ 52.5 k hours). A reviewer who opens the UCI page will hit this. Recommend one parenthetical in A.6: *"(UCI reports no missing cells; the gaps are missing hourly rows on the reconstructed grid, not NaNs)"*.
- **F17.** The **32-feature / 132 B exemplar vector is entirely ours** (24 lags ⊕ 3 weekly echoes ⊕ 4 weather ⊕ holiday) — UCI ships 8 columns. A.6 already declares this as *"the analyst's declared choice"*. ✅ no change needed, but the 8-vs-32 gap should never be presented as a dataset property.
- **F18.** No peer-reviewed paper is associated with the dataset; there is no author to cite beyond the donor. `@misc` is the only honest form. ⚠ Consistent with the r5-era scout's finding that *"no peer-reviewed stream-learning paper annotates its regimes"* — which is precisely why §R.2.2 must keep saying the drift map is **ours**. It does. ✅

## 2.6 Webb et al. (2016) — drift magnitude ⛔⛔ **the one content-lean that the source does not support as written**

**(a) Canonical record.** Geoffrey I. Webb, Roy Hyde, Hong Cao, Hai Long Nguyen, Francois Petitjean, *"Characterizing Concept Drift"*, **Data Mining and Knowledge Discovery 30(4):964–994, 2016**, DOI **`10.1007/s10618-015-0448-4`**, arXiv:1511.03816 (v6, 2016-04-08). The preprint's own header reads *"Accepted for publication in Data Mining and Knowledge Discovery on **December 10, 2015**"* — **the DOI's year digits are `015`, the citation year is `2016`** (online-first 2015, issue 2016). Never "Webb et al. (2015)".

**(c) Usage-check — the draft's sentence, and what the source actually says.**

> §R.2.2: *"A drift map built for the venue (**total-variation drift magnitude in the sense of Webb et al., 2016**; 1,101 revisit rows; the purely data-driven day map recovers the weekday/weekend split with no calendar input) confirms the firing is not a favourable slice…"*

What the source says, §3, read directly from the PDF:

> *"Quantifying the degree of difference between two points of time is a key characterization of any concept drift. We call this **drift magnitude**. However, the appropriate function for measuring drift magnitude may change from domain to domain. Rather than specifying which measure of distance between distributions should be used, **our definitions refer to an unspecified distribution distance function**:*  `D(t, t+m)`.  **(5)**

> *"Examples of distance functions that might be used include Kullback-Leibler Divergence … and Hellinger Distance …"* — followed by three desiderata (non-negativity; symmetry; triangle inequality) — *"**For this reason, in this paper we use Hellinger Distance in our case study.**"*

> Eq. (6): `Magnitude_{t,u} = D(t, u)`. *"The first of these is the **magnitude** of a drift, which is simply the distance between the concepts at the start and end of the period of drift."*

**Verdict: PARTIALLY SUPPORTED — split the claim in two.**
- ✅ *"drift magnitude"* as a named quantity, defined as a distribution distance between two time points (Eq. 6), **is** Webb et al.'s. Citing them for the **framework** is correct and is exactly what they ask for.
- ⛔ *"**total-variation** … in the sense of Webb et al."* — **the total-variation choice is ours, not theirs.** They explicitly decline to specify `D`; their own case study uses **Hellinger**. As written, the phrase can be read as attributing TV to them.
- ⭐ Mitigating and worth stating: **TV is a legitimate instantiation under their own stated criteria** — it is non-negative, symmetric and satisfies the triangle inequality (it is a metric), which is precisely the property list they give for preferring Hellinger. So the substance is defensible; only the attribution is loose.

**Recommended replacement clause (paper-writer's call):**
> *"…drift magnitude in the sense of Webb et al. (2016, Eq. 6), with the distribution distance instantiated as total variation — Webb et al. deliberately leave that function unspecified and use Hellinger distance in their own case study; total variation satisfies the metric properties they give as their reason for that choice."*

**(d) Flags.**
- **F19 (⛔ headline).** Above.
- **F20 (year/DOI trap, never-copy).** DOI is `10.1007/s10618-015-**0448-4**` (a `-015-` DOI on a **2016** paper). Do not "correct" the DOI to `-016-`, and do not down-date the citation to 2015.
- **F21.** Publisher page SSO-blocked; volume/issue/pages 30(4):964–994 confirmed by **arXiv `journal-ref` + the search-surface Monash/ACM records** — two independent, agreeing. Not publisher-read.
- **F22.** Webb et al. also define `PathLen`, `Duration`, `Rate` (Eqs. 7–10). The draft uses **only** magnitude. No over-claim; noted so a future revision does not silently borrow the rest of the vocabulary.

---

# 3. BibTeX — Appendix-Q house pattern, caveats in the `note` field

```bibtex
@article{souza2020challenges,
  title   = {Challenges in Benchmarking Stream Learning Algorithms with Real-world Data},
  author  = {Souza, Vinicius M. A. and dos Reis, Denis M. and Maletzke, Andre G. and
             Batista, Gustavo E. A. P. A.},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {34},
  number  = {6},
  pages   = {1805--1858},
  year    = {2020},
  doi     = {10.1007/s10618-020-00698-5},
  note    = {arXiv:2005.00113 (v1 30 Apr 2020, v2 30 Jun 2020). The INSECTS benchmark.
             Verified 2026-08-18: arXiv v2 PDF read directly (pp. 33--40) + DBLP for
             venue/volume/pages/DOI; the Springer page is SSO-blocked and was NOT read.
             ANCHORS WE REPRODUCE, both from Table 5 ("Prequential accuracy achieved by
             state-of-the-art methods in the Insect Stream Data"), row "Inc-reoc (bal.)":
             No-Change = 40.46, ARF = 77.13. Protocol (Sec 7.1): "We consider prequential
             evaluation ... over a sliding window of 1,000 instances". Stream length
             (Table 2): incremental-reoccurring (bal.) = 79,986 instances, change points
             26568; 53364. Temperature sweep (Sec 5.5): "from 20 C to 40 C".
             Out-of-control (Sec 5.5): "this dataset must be drift-free".
             CAUTION 1: all their numbers are MOA; ARF's ENSEMBLE SIZE IS NEVER STATED --
             100 is MOA's default (AdaptiveRandomForest.java: IntOption("ensembleSize",
             's', "The number of trees.", 100, 1, MAX)), so "ARF-100" is our inference.
             CAUTION 2: their ARF is "the best overall results" among the SIX methods they
             ran, not a literature-wide state of the art.
             NEVER-COPY: DBLP renders the authors "Vinicius M. A. DE Souza" and "Andre
             GUSTAVO Maletzke"; the paper's own title page and running head are
             "V. M. A. Souza et al." -- cite as "Souza et al.", never "de Souza et al."}}

@inproceedings{losing2016samknn,
  title     = {{KNN} Classifier with Self Adjusting Memory for Heterogeneous Concept Drift},
  author    = {Losing, Viktor and Hammer, Barbara and Wersing, Heiko},
  booktitle = {2016 {IEEE} 16th International Conference on Data Mining ({ICDM})},
  pages     = {291--300},
  year      = {2016},
  publisher = {IEEE},
  doi       = {10.1109/ICDM.2016.0040},
  note      = {SAM-kNN. Verified 2026-08-18 from the AUTHORS' ACCEPTED PREPRINT
             (honda-ri.de/pubs/pdf/3277.pdf, 11 pp., cover: "This is an accepted article
             published in IEEE International Conference on Data Mining"); IEEE Xplore
             camera-ready is paywalled and was NOT read, so pages/DOI are DBLP-sourced and
             all content quotes are pinned to the preprint.
             PUBLISHED DEFAULTS (Sec IV-A, verbatim): "We used for all experiments k = 5,
             L_min = 50, L_max = 5000." L_max is "The maximum number of stored examples
             L_max (STM and LTM combined)"; footnote 4: "the available space is shared
             between the STM and LTM". Sec VI: "Window based approaches were allowed to
             store 5000 samples (we also report results for a size of 1000 samples)."
             => the 5,000-exemplar (0.634 MiB at 133 B/exemplar) and 1,000-exemplar
             budgets ARE published.
             WEATHER ROW WE VALIDATE AGAINST (Table IV, "Interleaved Test-Train ERROR
             rates", window-size-5000 block): kNN_S = 21.53 (bold, lowest), SAM = 21.74.
             These are ERRORS, not accuracies; and note the authors' own table has the
             PLAIN WINDOW beating the dual memory on Weather.
             CAUTION: the paper contains NO LTM-size-proportion parameter. "LTMSizeProportion
             = 0.4" (hence a 3,000-example STM cap at L_max = 5000) is the AUTHORS'
             REFERENCE-IMPLEMENTATION / scikit-multiflow default
             (SAMKNNClassifier(..., ltm_size=0.4, ...)), NOT a published default -- do not
             call it one. The paper's only LTM sizing rule is Sec IV-B3: kMeans++ with
             |M_LTc|/2 clusters on overflow.
             NEVER-COPY: the HRI cover page misspells the title "heterogenous"; the paper's
             own title page, DBLP and IEEE all read "Heterogeneous".}}

@article{gomes2017arf,
  title   = {Adaptive random forests for evolving data stream classification},
  author  = {Gomes, Heitor Murilo and Bifet, Albert and Read, Jesse and Barddal, Jean Paul
             and Enembreck, Fabr{\'i}cio and Pfahringer, Bernhard and Holmes, Geoff and
             Abdessalem, Talel},
  journal = {Machine Learning},
  volume  = {106},
  number  = {9--10},
  pages   = {1469--1495},
  year    = {2017},
  doi     = {10.1007/s10994-017-5642-8},
  note    = {ARF. Record verified 2026-08-18 (DBLP + the MOA class javadoc in
             moa/classifiers/meta/AdaptiveRandomForest.java, which prints the full author
             list in order). THE PAPER BODY WAS NOT READ -- quote no number from it.
             This draft quotes none: every ARF number is ours or Souza et al.'s
             measurement of ARF.
             NEVER-COPY 1 (author order): the first author is GOMES, not Bifet -- at least
             one aggregator surface renders it "Bifet, Gomes, ...".
             NEVER-COPY 2 (spelling): MOA's own javadoc and some HAL surfaces print
             "Pfharinger"; the correct name is PFAHRINGER.
             A CORRECTION EXISTS and is not incorporated here: "Correction to: Adaptive
             random forests for evolving data stream classification", Machine Learning
             108(10):1877--1878, 2019, doi 10.1007/s10994-019-05793-3. Inert while we quote
             nothing; MUST be read before any ARF number is quoted.
             MOA's default ensembleSize is 100 (verified from source), which is the basis
             for reading Souza et al.'s ARF as ARF-100.}}

@article{montiel2021river,
  title   = {River: machine learning for streaming data in {P}ython},
  author  = {Montiel, Jacob and Halford, Max and Mastelini, Saulo Martiello and
             Bolmier, Geoffrey and Sourty, Raphael and Vaysse, Robin and Zouitine, Adil and
             Gomes, Heitor Murilo and Read, Jesse and Abdessalem, Talel and Bifet, Albert},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  number  = {110},
  pages   = {1--8},
  year    = {2021},
  note    = {arXiv:2012.04740. Verified 2026-08-18 (jmlr.org/papers/v22/20-1380.html read
             directly -- JMLR is open access, so this record is publisher-confirmed).
             NEVER-COPY: PyPI's own metadata prints a TRUNCATED BibTeX with
             "and others" and no journal/volume/pages -- do not copy it.}}

@misc{river2026software,
  title        = {river},
  author       = {{The river developers}},
  year         = {2026},
  note         = {Version 0.25.0, released 2026-05-31 (PyPI), BSD-3-Clause, maintainer
                  Max Halford. THE VERSION ACTUALLY RUN in Appendix A.6; cite alongside
                  montiel2021river, which is the paper, not the artifact.
                  LOAD-BEARING NEGATIVE, verified 2026-08-18 by TWO independent sources:
                  river ships NO SAM-kNN. riverml.xyz/0.25.0 API overview gives the whole
                  `neighbors` module as KNNClassifier, KNNRegressor, LazySearch, SWINN;
                  github.com/online-ml/river/tree/main/river/neighbors contains only
                  ann/, __init__.py, base.py, knn_classifier.py, knn_regressor.py, lazy.py.
                  SAMKNNClassifier lives in skmultiflow.lazy (one of river's two ancestor
                  packages) and was NOT carried into the merge. `forest.ARFClassifier` and
                  a `datasets.Insects` loader DO exist in 0.25.0.},
  howpublished = {\url{https://riverml.xyz/}}}

@misc{hogue2019metro,
  title        = {Metro Interstate Traffic Volume},
  author       = {Hogue, John},
  year         = {2019},
  publisher    = {UCI Machine Learning Repository},
  doi          = {10.24432/C5X60B},
  note         = {Dataset 492. Verified 2026-08-18: archive.ics.uci.edu/dataset/492 read
                  directly AND the DOI resolution tested (doi.org/10.24432/C5X60B -> 302 ->
                  archive.ics.uci.edu/dataset/492) -- this DISCHARGES the earlier internal
                  "treat the DOI as unverified" flag. Donated 2019-05-06; CC BY 4.0;
                  48,204 instances; 8 features; "Hourly Minneapolis-St Paul, MN traffic
                  volume for westbound I-94", 2012--2018, with weather and holiday features.
                  UCI's own suggested citation: "Hogue, J. (2019). Metro Interstate Traffic
                  Volume [Dataset]. UCI Machine Learning Repository."
                  CAUTION: the UCI page states "Has Missing Values? No" -- that means no
                  missing CELLS in the 48,204 delivered rows; our declared 6.4 % feature
                  gap-fill and 7,386-hour sensor hole are missing HOURLY ROWS on the
                  reconstructed grid. Not a contradiction, but a reviewer will hit it.
                  The 32-feature / 132 B exemplar vector is OURS; UCI ships 8 columns.
                  No associated peer-reviewed paper exists; the donor is the only creator
                  to cite, and no published drift annotation exists for this stream.}}

@article{webb2016characterizing,
  title   = {Characterizing concept drift},
  author  = {Webb, Geoffrey I. and Hyde, Roy and Cao, Hong and Nguyen, Hai Long and
             Petitjean, Francois},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {30},
  number  = {4},
  pages   = {964--994},
  year    = {2016},
  doi     = {10.1007/s10618-015-0448-4},
  note    = {arXiv:1511.03816 (v6, 8 Apr 2016). Verified 2026-08-18: arXiv PDF read directly
             (pp. 6--7); Springer page SSO-blocked and NOT read; volume/issue/pages from the
             arXiv journal-ref + independent DBLP/Monash records that agree.
             DATE TRAP: the preprint header reads "Accepted for publication in Data Mining
             and Knowledge Discovery on December 10, 2015" and THE DOI'S YEAR DIGITS ARE
             015, but the issue year is 2016. Cite as 2016; do NOT "fix" the DOI to -016-.
             MANDATORY CAVEAT ON ANY USE OF "DRIFT MAGNITUDE": Eq. (6) is
             Magnitude_{t,u} = D(t,u), where D is DELIBERATELY UNSPECIFIED -- Sec 3,
             verbatim: "Rather than specifying which measure of distance between
             distributions should be used, our definitions refer to an unspecified
             distribution distance function"; and "For this reason, in this paper we use
             Hellinger Distance in our case study." THE TOTAL-VARIATION CHOICE IS OURS,
             NOT THEIRS. It is a legitimate instantiation -- TV meets the three properties
             they give as their reason for choosing Hellinger (non-negativity, symmetry,
             triangle inequality) -- but the attribution must be split: framework theirs,
             distance ours.
             We use ONLY magnitude (Eq. 6); the paper also defines Duration (7), PathLen (8)
             and Rate (9--10), which this draft does not borrow.}}
```

---

# 4. Confidence & gaps

| claim | status |
|---|---|
| Souza Table 5 anchors 40.46 / 77.13; Table 2 = 79,986; window-1000 protocol; 20→40 °C | **VERIFIED, primary full text** (arXiv v2 PDF read here, this session — independent of the earlier internal scout, which agrees) |
| Losing k=5 / L_min=50 / L_max=5000 shared; Weather 21.74 / 21.53 error | **VERIFIED, primary full text** (author preprint read here) |
| river ships no SAM-kNN | **VERIFIED, double-sourced** (official 0.25.0 docs + GitHub tree) |
| UCI Metro record + DOI resolution | **VERIFIED, primary + resolution test** |
| Webb Eq. 5/6 and the Hellinger case-study sentence | **VERIFIED, primary full text** |
| ARF venue/vol/pages/DOI/author order | **VERIFIED, double-sourced** (DBLP + MOA javadoc). ⚠ **body not read** |
| Souza/Webb pagination publisher-side | ◐ **two agreeing secondary records** (arXiv journal-ref + DBLP); Springer SSO-blocked |
| Losing pages 291–300 | ◐ **DBLP-sourced**; IEEE Xplore paywalled |
| ARF 2019 correction contents | ⛔ **NOT READ** — inert only while we quote nothing from ARF |
| `LTMSizeProportion = 0.4` as the authors' own implementation default | ◐ confirmed via **scikit-multiflow's documented default `ltm_size=0.4`**, whose docs cite Losing et al. as the implemented paper; `github.com/vlosing/SAMkNN` raw file returned 404 on the path tried ⇒ **single-sourced on the 0.4 value's provenance**. The load-bearing part — *that it is absent from the paper* — is **verified from the paper**. |

**What I'd check next, in priority order.** (1) `github.com/vlosing/SAMkNN` under its actual file layout, to second-source `LTMSizeProportion = 0.4` as the *authors'* default rather than scikit-multiflow's. (2) The ARF 2019 correction (`10.1007/s10994-019-05793-3`, 2 pp.) — cheap, and it de-risks any future ARF quote. (3) Whether Souza et al. state ARF parameters anywhere outside §7.1 (I read pp. 33–40; a parameter table could exist in an appendix I did not open) — would turn reconciliation item 2 from an inference into a fact. (4) Publisher-side confirmation of Souza/Webb pagination if the Head has institutional access.

# 5. Relevance to CHLU / B′ positioning
Marginal but non-zero. Two things are worth the Hub's attention beyond bookkeeping. **First**, Losing et al.'s own Table IV already shows `kNN_S` (a plain sliding window) **beating** SAM-kNN (the dual short-term/long-term memory) on Weather, 21.53 vs 21.74 error — that is the *published* version of §R.2.2's finding that our `knnsam` loses to the plain window (325.71 vs 320.98). B′'s "a memory's dynamics must beat its own stored content" thesis has a decade-old, peer-reviewed instance of exactly that failure sitting in the source it already cites, and citing it as prior corroboration costs nothing and strengthens the admissibility argument from "our audit found this" to "this is a known pattern the field has published and not acted on." **Second**, Webb et al.'s framework is the right vocabulary for the drift maps and is *deliberately* distance-agnostic, so B′ can instantiate TV without apology — but the same generality means "drift magnitude" alone never pins a quantity, and any cross-paper comparison of drift magnitudes must state the distance. Nothing here bears on the four dials.

---

## Flags
- ⛔⛔ **F19 (headline):** §R.2.2's *"total-variation drift magnitude in the sense of Webb et al., 2016"* attributes to Webb et al. a distance choice they explicitly decline to make (Eq. 5) and do not use (they use Hellinger). Framework attribution ✅, distance attribution ⛔. Replacement clause drafted in §2.6.
- ⚠ **F4/F12:** the `77.13` anchor is **MOA**'s ARF at an **unstated** ensemble size; ours is **river**'s at 100. The draft says *"ours is the stronger implementation"* but never names the two frameworks. `c2w10-benchmark-gate.md` §282 already states it correctly — the draft is the only site missing it.
- ⚠ **F5:** *"a permitted 3,000"* implies a published STM cap. There isn't one; it comes from `LTMSizeProportion = 0.4`, a code default. The `0.634 MiB` / `L_max = 5000` budget **is** published and is clean.
- ⚠ **F16:** UCI says "Has Missing Values? No"; A.6 declares 6.4 % gap-fill and a 7,386-hour hole. Reconcilable (cells vs rows) but a reviewer trip-hazard; one parenthetical fixes it.
- ⚠ **F11:** an **ARF correction (2019)** exists and is unread. Harmless now (we quote no ARF number); blocking if that ever changes.
- ⚠ **F1:** *"the published state-of-the-art ensemble"* → Souza's ARF is best of the **six methods they ran**. Soften to "the venue's best published method."
- ✅ **NEVER-COPY register (six new entries):** (i) Souza is **"Souza et al."**, not "de Souza et al." (DBLP renders "de Souza"); (ii) ARF's first author is **Gomes**, not Bifet; (iii) ARF's sixth author is **Pfahringer**, not "Pfharinger" (MOA's javadoc is wrong); (iv) Losing's title is **"Heterogeneous"**, not the HRI cover's "heterogenous"; (v) Webb's DOI carries **`-015-`** on a **2016** paper — do not "fix" either; (vi) **do not copy PyPI's truncated river BibTeX** (`and others`, no venue).
- ✅ **Discharged:** the r5-era *"Metro Interstate DOI … treat it as unverified"* flag (`c2w10-benchmark-scout.md` l.203) — the DOI resolves.
- ✅ **Confirmed retraction:** `c2w10-benchmark-scout.md` l.126 (*"Both are in river (`river.neighbors.SAMKNNClassifier` family) … the launder arm is a one-line baseline"*) is **wrong**, independently re-verified here. The gate report already corrected it; recording it so the correction is double-sourced.
- ⛔ **No repo edits, no git footprint.** Read-only; one file written: `.claude/outputs/bprime-r6-cite-check.md`.

## Proposed handover updates (for the Hub)
1. **B′ r6 → r7 fold, citation layer.** All six entries are ready to paste into **Appendix Q.1**, which becomes **thirteen** verified entries. Editorial item **10** ("⛔ None of these is in Appendix Q; a cite-check pass is owed before circulation") is **discharged** and must be rewritten to point at Q.1.
2. **Three draft sentences need a qualifier before any freeze** (owner: paper-writer): §R.2.2's Webb clause (reconciliation item 1 — the only *substantive* one); §R.2.1/A.6's ARF **MOA-vs-river** naming; A.6's *"permitted 3,000"* label.
3. **Two optional strengthenings, Hub's call.** (a) Cite Losing et al.'s **Weather row** as *published prior corroboration* that a plain window can beat a dual memory — it is in the source B′ already cites and it converts §R.2.2's mechanism note from "our finding" to "a known pattern." (b) Add the UCI missing-cells-vs-missing-rows parenthetical to A.6 (F16).
4. **Q.2 candidates** (verified-but-not-printed, if the fold prefers minimality): the **ARF 2019 correction** (record verified, contents unread, quoted nowhere) and the **skmultiflow `ltm_size=0.4` provenance** (single-sourced on whose default it is; the load-bearing negative — that it is *not* in the paper — is paper-verified).
5. **Standing note for the program, not just B′:** every one of the six is now primary-verified except two body-texts (**Gomes**, and the publisher-side page images for **Souza/Webb**). If a future wave wants to quote an ARF or a DMKD number, that is the residual debt.
