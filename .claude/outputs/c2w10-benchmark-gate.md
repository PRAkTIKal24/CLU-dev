# c2w10-benchmark-gate — results-analyst report

Task + acceptance criterion: run PREREG-C2W10 §3's legs B1–B4 on the frozen INSECTS streams and write `BENCHMARK-GATE.json` with `criterion4_cleared` computed arithmetically. **Status: done.**

## ⛳ RECONCILIATION LIST (owner needed — protocol §5 corollary, in the first 10 lines)
1. **`criterion4_cleared = false`.** The Metro fallback is pre-authorized (PREREG §2.0 / Head ruling 1). Every doc naming INSECTS as C2W10's **VALUE** venue must move to "registered admissibility finding".
2. **`out-of-control` HAS NO DATA SOURCE.** Not in the 2024-04-16 USP archive (I extracted and listed it), not in river 0.25.0. **PREREG §6.2's V3 drift-free-null leg is a declared NOT-RUN** until re-sourced.
3. **`c2w10-benchmark-scout` §5 is wrong: river does NOT ship SAM-kNN.** "one-line baseline" → a ported reference implementation. Cost estimates that cite it must be revised.
4. **PREREG §2.2's band map is contaminated at `b = 4`** — the terminal band of every cycle is persistence-trivial and at ceiling for every arm. `R(4)`/`A(4)` are uninterpretable as registered.
5. **ARF is not byte-matched to anything**: measured state 9,542,925 B (100 trees) = **14.35× SAM-kNN's 665,000 B**. Any "ARF is the reference" sentence needs this caveat.
6. Minor: Souza's "incremental-abrupt-**reoccurring**" is river's `incremental_abrupt_balanced` (sha256-verified identical); class labels are **{2,3,4,5,11,12}**, not 0-based.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dial:** none — **benchmark admissibility instrument.** ⛔ No CLU cell, no store, no dividend, no performance claim about the CLU of any kind was run or is made here.
- **Laundering control:** N/A — **I am the laundering control.** The exemplar store at matched bytes is the wave's launder and this task measures how strong it is.
- **Falsifies:** B1 failing ⇒ our loader/ordering is defective and nothing on this stream is quotable. **B1 passed on both conditions.**
- **Does NOT falsify:** the exemplar store beating ARF does not falsify the CLU; it falsifies the **benchmark's** admissibility as a VALUE venue (criterion 4).
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ NOT-RUNs declared, never nulls (§8 below).

## Flag provenance (mandatory, protocol §5)
| item | value |
|---|---|
| repo HEAD at run time | `7fcef50` — **unchanged before and after; 0 tracked files modified** (`git status --porcelain` clean of non-`??` entries) |
| tracked code touched | **none.** No `chlu/`, no `tests/`, no `pyproject.toml`, no `uv.lock`. No branch, no commit, no push. |
| environment | **scratch venv** `.claude/scratch/c2w10/.venv`, CPython **3.12.9**, `river==0.25.0`, `numpy==2.5.2`, `scikit-learn==1.9.0`, `scipy==1.18.0`. Full freeze: `.claude/outputs/c2w10-benchmark-gate/scratch_venv_freeze.txt` |
| project venv / lock | **untouched** (Head ruling 6 respected; the `pandas 3.0.3→2.3.3` precedent) |
| ARF config | `river.forest.ARFClassifier(n_models=100, seed∈{1,2,3})` — **all other defaults**: `lambda_value=6`, `max_features='sqrt'`, `drift_detector=ADWIN(0.001)`, `warning_detector=ADWIN(0.01)`. Secondary row `n_models=10`, seeds {1,2,3} |
| SAM-kNN config | `k=5`, `L_min=50`, `L_max∈{5000,1000}`, `knnWeights='distance'`, `LTMSizeProportion=0.4`, `recalculateSTMError=False`, `useLTM=True` — Losing et al. ICDM 2016 published defaults, unmodified. Deterministic (KMeans `n_init=1, random_state=0`), no seed |
| kNN_S config | same file, `useLTM=False`, `recalculateSTMError=None`, `maxSize=L ∈ {250,500,1000,2000,5000,14782}`. Deterministic |
| feature scaling | exemplar arms run **raw AND `std`** (causal prequential per-feature z-score, statistics from instances `< t` only, updated after the prediction). ARF raw only (scale-invariant) |
| decimation | **`m = 1` (undecimated)** for every number in this report. The ladder is characterised, not run |
| metric window | prequential accuracy, sliding window **1000** (Souza convention); index 0 excluded for **every** arm (No-Change's first prediction is undefined) ⇒ n_scored = 79,985 |
| commands | `.venv/bin/python run_arm.py <stream>.npy <arm> <tag> [seed]` → `metrics.py` → `bandstats.py` → `build_gate.py` → `add_notes.py`. All scripts copied verbatim to `…/c2w10-benchmark-gate/code/` |

**Pre-registration:** `.claude/outputs/c2w10-benchmark-gate/PREREG.md`, filed **before** the ARF/SAM-kNN/kNN_S harness ran. It discloses that the No-Change baseline (a deterministic function of the label column, no seed, no free parameter) was computed **before** the prereg was written. It registers two protocol choices (ARF ensemble size = MOA's 100, not river's 10; exemplar arms run raw+std with the max consumed) and 9 point predictions. Scored in §7.

---

## 1. Data: fetched, frozen, hashed, and verified against the original

**Declared cache path (stable, gitignored, inside the repo so every spoke can reach it):**
`/Users/user/Desktop/CHLU/.claude/data/c2w10-streams/`

| file | sha256 | bytes | n | feat | cls | class histogram |
|---|---|---|---|---|---|---|
| `incremental_reoccurring_balanced.csv` | `f267c0fb1f4e1d68967e2a694305146425181c19a7d3c282b67b252b98407c7a` | 21,433,047 | 79,986 | 33 | 6 | 13,331 each of {2,3,4,5,11,12} |
| `incremental_abrupt_balanced.csv` | `c1cd19d429844349009311fa372ee700d5bf0e21545fb61d0d3fba724123188b` | 21,421,452 | 79,986 | 33 | 6 | 13,331 each |
| `incremental_reoccurring_imbalanced.csv` (stretch, **no arm run**) | `c881f34f4ba8029dfe9cdb948b20026a8fcd706c19ce3e8efc7bb5682dd4a3d6` | 120,798,571 | 452,044 | 33 | 6 | 125354/83794/29953/13331/64895/134717 |
| `USP_DS_Repository.zip` (source of record, all 10 INSECTS variants + 24 other streams) | — | 599,633,529 | — | — | — | — |

**Provenance verified, not assumed.** I downloaded the USP Data Stream Repository archive directly (600 MB, CC BY 4.0, no password needed despite the paper's `DMKD2018` footnote), extracted `USP DS Repository/INSECTS/INSECTS incremental-reoccurring_balanced.csv`, and its **sha256 is bit-identical to river's mirror** — likewise for incremental-abrupt. ⇒ *river's mirror is the USP original.* Both file sizes also match river's registry (21,433,047 / 21,421,452) and Souza Table 2's instance counts exactly.

**Change points 26,568 / 53,364, checked against the file's own length:** cycles are `[0, 26568)`, `[26568, 53364)`, `[53364, 79986)` of sizes **26,568 + 26,796 + 26,622 = 79,986 ✓**. A corroborating (not confirming) empirical check: the 2,000-smoothed mean of the 7 large-scale features traces **up (1448→1897) → down (1897→1571) → up**, matching the published 20→40→20→40 °C schedule; its turning points fall at 25,526 and 56,030, i.e. within 3.9 % and 5.0 % of the published indices. The exact indices are Souza's, not ours.

### ⛔ `out-of-control` — NOT OBTAINED, declared NOT-RUN (not a null)
The archive's `INSECTS/` folder contains **exactly 10 files** (abrupt/gradual/incremental/incremental-abrupt/incremental-reoccurring × balanced/imbalanced), **all 6-class**. There is **no `out-of-control`** file anywhere in the 600 MB archive, and river 0.25.0's `Insects.variant_configs` does not list it either (its `__init__` references the string `"out-of-control"` but no config exists — dead code). **PREREG-C2W10 §6.2's V3 drift-free null therefore has no data source as registered.**

---

## 2. B1 — the loader positive control (reported before B2, as required)

| arm | ours | Souza Table 5 | Δ | tol | verdict |
|---|---|---|---|---|---|
| **No-Change (persistence)** | **40.4526** | **40.46** | **−0.0074** | ±2.0 | ✓ |
| **ARF (100 trees, 3 seeds)** | **78.8139 ± 0.0526 SD** (78.8360 / 78.8523 / 78.7535) | **77.13** | **+1.6839** | ±2.0 | ✓ |
| ARF (10 trees, 3 seeds) — secondary | 77.4129 ± 0.1470 SD (77.5333/77.4608/77.2445) | 77.13 | +0.283 | — | (row, not a gate) |

### **`b1_pass = true`.**
The persistence number is essentially exact (0.0074 points) — that is the decisive ordering check, because `ŷ_t = y_{t−1}` is a pure function of the label sequence: **any shuffling, re-ordering or off-by-one in our loader would move it**, exactly the failure mode Souza §4.2 documents for MOA's Poker-hand. It does not move.

The second condition reproduces independently: **No-Change 42.3779 vs published 42.39** (Δ −0.012), **ARF-100 76.2251 ± 0.0381 vs published 74.95** (Δ +1.275).

*Nothing was tuned to make B1 pass.* The 100-tree ensemble was pre-registered as MOA's default before any ARF ran; the 10-tree river default happens to land even closer to Souza's number, and both pass.

---

## 3. B2 — the criterion-4 tripwire (the deciding measurement)

Same frozen stream, same prequential protocol (window 1000), `m = 1`.

### 3.1 The registered arm set (the gate's arithmetic)
| arm | state bytes | acc % | κ | κ_per | κ⁺ |
|---|---|---|---|---|---|
| **ARF (100 trees)** — reference | 9,542,925 (measured) | **78.8139** | 0.7458 | 0.6442 | 0.6931 |
| **SAM-kNN, L_max = 5000** (published defaults), std | 665,000 | **76.9157** | 0.7230 | 0.6123 | 0.6654 |
| **kNN_S, L = 1000**, std | 133,000 | 75.3616 | 0.7043 | 0.5862 | 0.6426 |
| **kNN_S, L = 5000**, std | 665,000 | 68.1490 | 0.6178 | 0.4651 | 0.5360 |
| **No-Change** (mandatory in every table) | 4 | 40.4526 | 0.2854 | 0.0000 | 0.0000 |

```
b2_margin_pts = best_exemplar − ARF = 76.9157 − 78.8139 = −1.8983
criterion4_cleared = (−1.8983 < −2.0) = FALSE
```
### **⇒ CRITERION 4 HAS FIRED ON THE PRIMARY.**

### 3.2 The margin is 0.10 points inside the threshold — so here is every reference I could compute
| ARF reference | value | margin | fires? |
|---|---|---|---|
| ours, 100 trees, 3-seed mean (**pre-registered**) | 78.8139 | **−1.898** | ✔ |
| ours, 100 trees, best seed | 78.8523 | −1.937 | ✔ |
| ours, 100 trees, worst seed | 78.7535 | −1.838 | ✔ |
| ours, 10 trees (river default), 3-seed mean | 77.4129 | −0.497 | ✔ |
| Souza's published ARF | 77.13 | **−0.214** | ✔ |

**It fires under all five.** The pre-registered choice (100 trees) was the one *least* favourable to firing — a stronger ARF makes the gap bigger — and it still fires with 0.10 points to spare. I am flagging the thinness explicitly rather than burying it: **had ARF-100 landed 0.11 points higher, the boolean would have flipped**, and the honest reading is "the exemplar store is at ARF's shoulder", not "the exemplar store loses".

And with the **unregistered but stronger** exemplar arm included, the picture is not marginal at all:
**SAM-kNN at L_max = 1000 scores 77.0632 %** — margin **−1.751** vs our best ARF, **−0.35 vs our 10-tree ARF**, and **−0.07 vs Souza's published ARF**. A 133 kB exemplar store is within **one-fifteenth of a point** of the published state-of-the-art ensemble on this stream.

### 3.3 ⭐ The mechanism — and it inverts the scout's §2.3 in the opposite direction to the one the Hub hoped for
The scout's §2.3 predicted the risk that a *large* (5,000-example) window would be at ceiling. What I measure is stronger and stranger: **accuracy is monotonically decreasing in store size above L ≈ 500.**

| kNN_S window `L` (std) | 250 | 500 | 1000 | 2000 | 5000 | 14,782 |
|---|---|---|---|---|---|---|
| **acc %** | 75.56 | **76.03** | 75.36 | 73.39 | 68.15 | **59.75** |
| state bytes | 33,250 | 66,500 | 133,000 | 266,000 | 665,000 | 1,966,006 |

(figure: `fig2_window_ladder.png`). **Recency *is* the hidden temperature variable.** A 500-example window spans ≈0.6 % of the stream ≈ 0.13 °C of the 20 °C sweep, so a short-window kNN is, for free, the per-temperature classifier that Souza measured at 90 % vs 84 % pooled. Storing *more* is actively harmful: the CLU-byte-matched store (14,782 exemplars, 1,966,006 B — i.e. **the CLU's own budget**) is the **worst** exemplar arm at 59.75 %, **19 points below** a store 30× smaller.

**SAM-kNN wins by discovering this itself.** At `L_max = 5000` its self-adjusting STM averages **945** examples out of a permitted 3,000 (final 3,023; LTM mean 2,097). Its published dual-memory design is doing exactly what the wave wants a controller to do, and the input metric is enough.

**ARF is also a recency mechanism here.** At the end of the pass **78 of its 100 trees hold exactly one node** (616 nodes total across the forest); a 10-tree probe sampled every 5,000 instances gives forest node totals 302 / 188 / 326 / 148 / 232 / 382 / 198 / 374, with individual trees repeatedly reset to 1. On INSECTS, ARF's edge is **churn, not accumulation.** Every arm at the top of this benchmark forgets fast.

### 3.4 Anti-hobbling (the F3 rule), discharged explicitly
- The exemplar arms use a Python-3 port of **the authors' own reference implementation** (github.com/vlosing/SAMkNN), at **published defaults**, with the `libNearestNeighbor` C extension replaced by numpy shims whose semantics I read off `nearestNeighbor.cpp` (squared-Euclidean; `nArgMin` ties → lowest index; `linearWeightedLabels` weight `1/max(d,1e-9)`, ties → lowest label). **Shim equivalence is unit-tested** against brute-force references (500 randomised cases incl. forced ties, all exact) and **kNN_S ≡ a brute-force sliding-window distance-weighted kNN, 400/400 agreement**.
- **The port reproduces the authors' published numbers.** On their own shipped Weather dataset (18,159 × 8, their Table II row), at `L=5000, k=5, distance` weights, raw features:

  | | ours | Losing et al. Table IV (scout's OCR-cautioned read) |
  |---|---|---|
  | SAM-kNN interleaved test-train **error** | **21.70 %** | 21.74 |
  | kNN_S interleaved test-train **error** | **21.68 %** | 21.53 |
- **Declared deviation, both sides reported:** the frozen file is **not normalised** (feature range 0.0020 → 8,014.5; 7 of 33 columns are O(10²–10³)). Raw Euclidean kNN is hobbled by that; ARF is scale-invariant. So every exemplar arm ran **twice** and B2 consumes the **max**. Standardisation is worth **+4.0 to +6.3 points** to the kNN arms (SAM-kNN 5000: 71.52 raw → 76.92 std; kNN_S 1000: 69.56 → 75.36). Had I run raw only, `criterion4_cleared` would have come out **true** — i.e. **the anti-hobbling rule is exactly what decided this gate.**

### 3.5 Second condition — same verdict, independently
`incremental-abrupt-reoccurring (balanced)`, 3 ARF seeds:

| arm | acc % | κ | κ_per | κ⁺ | state bytes |
|---|---|---|---|---|---|
| ARF (100 trees) | 76.2251 ± 0.0381 | 0.7147 | 0.5874 | 0.6479 | ~9.5 MB |
| **SAM-kNN L_max=1000 (std)** | **74.3802** | 0.6926 | 0.5554 | 0.6202 | 133,000 |
| SAM-kNN L_max=5000 (std) | 73.5238 | 0.6823 | 0.5405 | 0.6073 | 665,000 |
| kNN_S L=1000 (std) | 72.6786 | 0.6721 | 0.5259 | 0.5945 | 133,000 |
| kNN_S L=5000 (std) | 65.5623 | 0.5867 | 0.4024 | 0.4859 | 665,000 |
| No-Change | 42.3779 | 0.3085 | 0.0000 | 0.0000 | 4 |

margin = 74.3802 − 76.2251 = **−1.845** ⇒ `criterion4_cleared = false` here too.

---

## 4. B3 — temporal-dependence sanity, κ_per and κ⁺ for **every** arm

**`b3_kappa_per_arf = 0.64422 > 0` ⇒ B3 PASSES**; the stream is not persistence-trivial *globally* and is not excluded the way ELEC2 is. Full table (all arms, primary stream unless prefixed `ABR_`; No-Change present as required):

| arm | acc % | window-1000 mean % | κ | κ_per | κ⁺ | wall s |
|---|---|---|---|---|---|---|
| arf100_s2 | 78.85 | 78.71 | 0.7462 | 0.6449 | 0.6937 | 1048 |
| arf100_s1 | 78.84 | 78.69 | 0.7460 | 0.6446 | 0.6935 | 1039 |
| arf100_s3 | 78.75 | 78.62 | 0.7450 | 0.6432 | 0.6923 | 1043 |
| arf10_s1 | 77.53 | 77.40 | 0.7304 | 0.6227 | 0.6744 | 85 |
| arf10_s2 | 77.46 | 77.32 | 0.7295 | 0.6215 | 0.6733 | 179 |
| arf10_s3 | 77.24 | 77.10 | 0.7269 | 0.6179 | 0.6702 | 180 |
| **samknn_1000_std** | **77.06** | 76.91 | 0.7248 | 0.6148 | 0.6675 | 85 |
| **samknn_5000_std** | **76.92** | 76.76 | 0.7230 | 0.6123 | 0.6654 | 94 |
| ABR_arf100_s2 | 76.27 | 76.09 | 0.7152 | 0.5881 | 0.6486 | 1177 |
| ABR_arf100_s3 | 76.21 | 76.04 | 0.7145 | 0.5872 | 0.6477 | 1192 |
| ABR_arf100_s1 | 76.20 | 76.03 | 0.7143 | 0.5869 | 0.6475 | 1193 |
| knns_500_std | 76.03 | 75.89 | 0.7124 | 0.5975 | 0.6524 | 70 |
| knns_250_std | 75.56 | 75.40 | 0.7067 | 0.5895 | 0.6455 | 24 |
| knns_1000_std | 75.36 | 75.21 | 0.7043 | 0.5862 | 0.6426 | 25 |
| ABR_samknn_1000_std | 74.38 | 74.20 | 0.6926 | 0.5554 | 0.6202 | 60 |
| ABR_samknn_5000_std | 73.52 | 73.33 | 0.6823 | 0.5405 | 0.6073 | 98 |
| knns_2000_std | 73.39 | 73.21 | 0.6806 | 0.5531 | 0.6135 | 83 |
| ABR_knns_1000_std | 72.68 | 72.49 | 0.6721 | 0.5259 | 0.5945 | 21 |
| samknn_5000_raw | 71.52 | 71.32 | 0.6582 | 0.5217 | 0.5860 | 181 |
| samknn_1000_raw | 70.77 | 70.57 | 0.6493 | 0.5092 | 0.5750 | 88 |
| knns_1000_raw | 69.56 | 69.34 | 0.6347 | 0.4888 | 0.5570 | 22 |
| knns_5000_std | 68.15 | 67.92 | 0.6178 | 0.4651 | 0.5360 | 66 |
| ABR_knns_5000_std | 65.56 | 65.30 | 0.5867 | 0.4024 | 0.4859 | 49 |
| knns_5000_raw | 64.11 | 63.83 | 0.5693 | 0.3973 | 0.4756 | 64 |
| knns_14782_std | 59.75 | 59.44 | 0.5170 | 0.3241 | 0.4093 | 155 |
| knns_14782_raw | 55.75 | 55.39 | 0.4690 | 0.2569 | 0.3471 | 154 |
| ABR_nochange | 42.38 | 41.98 | 0.3085 | 0.0000 | 0.0000 | 0 |
| **nochange** | **40.45** | 40.03 | 0.2854 | **0.0000** | **0.0000** | 0 |

(No-Change's `κ_per = 0` exactly and `κ⁺ = 0` — the definitional sanity check on my metric code.)

### ⚠ 4b. B3 passes globally but the stream is LOCALLY persistence-trivial — a finding the Hub must act on
`κ_per > 0` is a whole-stream statistic and it hides a structure that directly breaks PREREG §2.2. Per-band class entropy and per-arm accuracy on **our** band map:

| band | H(y) bits | maj % | ARF-100 | SAM-kNN 1000 | kNN_S 1000 | **No-Change** |
|---|---|---|---|---|---|---|
| c1b0–c1b3 | 2.07–2.35 | 28–38 | 67.8–72.3 | 65.3–71.1 | 63.3–67.5 | 25.7–35.1 |
| **c1b4** | **0.842** | **73.0** | **99.38** | **98.93** | **97.82** | **91.25** |
| c2b0–c2b3 | 1.93–2.35 | 25–52 | 76.3–83.0 | 74.4–81.7 | 74.1–79.3 | 23.9–36.3 |
| **c2b4** | **1.052** | **54.2** | **98.47** | **97.70** | **96.62** | **71.32** |
| c3b0–c3b3 | 2.07–2.37 | 28–38 | 68.5–71.8 | 66.3–70.2 | 64.2–68.2 | 26.2–33.8 |
| **c3b4** | **0.839** | **73.2** | **99.34** | **99.12** | **98.14** | **90.78** |

The **terminal band of every cycle is at ceiling for every method** and the persistence baseline reaches 91 %. Supporting numbers: the longest single-label run in the stream is **3,597 instances** (starts at index 76,389, label 5); contiguous regions where the No-Change windowed accuracy exceeds 90 % are **[23634, 26753], [49007, 49209], [53269, 53561], [77043, 79985]** = **8.30 %** of the stream — and the first and third **straddle the published change points**.

**Consequence.** PREREG §2.2 defines `R(b)` against "the **last 1000 instances** of that band's first visit" and pairs `(c1,b) ↔ (c2,4−b)`. For `b = 4` the anchor sits inside the degenerate zone and is paired against a non-degenerate band. **`R(4)` and `A(4)` are uninterpretable as registered.** Recommendation: restrict every retention/adaptation claim to **bands 0–3** (12 of 15 bands, ~80 % of the stream) and print the per-band No-Change accuracy in **every** band-level table — Žliobaitė's rule applied per band rather than per stream.

### ⭐ 4c. The retention bar, measured on the existing arms (free, and V1 needs it)
Mean accuracy over bands 0–3, cycle 3 minus cycle 1, same pass (band 4 excluded as degenerate):

| arm | c1 (b0–3) | c3 (b0–3) | **revisit Δ pts** |
|---|---|---|---|
| ARF (100 trees) | 70.385 | 70.842 | **+0.457** |
| SAM-kNN L=1000 (persistent exemplar store) | 68.127 | 68.664 | **+0.537** |
| SAM-kNN L=5000 | 68.560 | 68.476 | −0.084 |
| kNN_S L=1000 (episodic) | 66.306 | 66.133 | −0.173 |
| kNN_S L=5000 | 61.158 | 57.541 | −3.618 |
| **No-Change (the noise floor)** | 30.664 | 30.191 | **−0.474** |

**No existing method extracts a material revisit benefit on this stream** — including an explicitly persistent exemplar store — and the persistence baseline's own drift (−0.474) is the same size as the largest positive. Two readings, both worth stating: (i) V1's `persistent − episodic` margin has to clear ~0.5 points to be distinguishable from nothing; (ii) SAM-kNN(1000) vs the best fixed window kNN_S(500) is **77.06 − 76.03 = +1.03 points** — that is *the literature's own persistent-vs-episodic contrast*, measured, and it is the honest bar for V1. Its LTM is selected as the predicting memory on **14,149 / 79,986 = 17.7 %** of instances at `L_max=5000`.

---

## 5. B4 — the byte ledger, computed from the data's own shape

`bytes_per_exemplar = 33 × 4 (float32) + 1 (label) = 133 B`

| entry | bytes | MiB | note |
|---|---|---|---|
| exemplar store, **L = 5000** (published budget) | **665,000** | **0.63419** | matches PREREG §3's 665,000 B / ≈0.634 MiB exactly |
| exemplar store, **L = 1000** (second budget point) | **133,000** | 0.12684 | |
| **CLU store, d = 12** | **1,966,080** | **1.875** | `n_atoms = 512·√(2¹²) = 32,768`; `n_atoms × (dim+2) × 4`, `dim = 13` |
| exemplars affordable at the CLU budget | **14,782** exemplars = 1,966,006 B (74 B slack) | 1.87493 | confirms PREREG §6.3's "≈14,782" **exactly** |
| CLU ÷ SAM-kNN(5000) | — | — | **2.9565×** |
| **ARF, 10 trees, measured** | **991,534** | 0.946 | |
| **ARF, 100 trees, measured** | **9,542,925** | **9.101** | **14.35× SAM-kNN(5000); 4.85× the CLU store** |
| No-Change | 4 | — | one stored label |

*ARF state-byte method, declared:* `pickle.dumps(model, protocol=5)` of the fitted `river` model after the full 79,986-instance pass, seed 1. That is a **measured upper bound on a minimal serialisation**, not a minimal encoding — most of it is ADWIN detector state, not tree structure (only 616 nodes across 100 trees). It is reported as measured.

⚠ **The consequence for the wave's two-sided ledger: ARF is not a byte-matched arm — it is the largest store in the comparison.** The byte-matched competitors are SAM-kNN and kNN_S, and *those* are the ones the CLU must beat. `criterion4_cleared` was computed against ARF because PREREG §3 registers ARF as the B2 reference; the byte-matched reading is if anything harsher.

---

## 6. Decimation ladder (Head ruling 5) — priced, and every `m` verified

| `m` | n instances | surviving change points | per-cycle counts | 3 cycles? | both cps? | Σ = n? |
|---|---|---|---|---|---|---|
| **1** | 79,986 | 26,568 / 53,364 | 26,568 / 26,796 / 26,622 | ✓ | ✓ | ✓ |
| **2** | 39,993 | 13,284 / 26,682 | 13,284 / 13,398 / 13,311 | ✓ | ✓ | ✓ |
| **5** | 15,998 | 5,314 / 10,673 | 5,314 / 5,359 / 5,325 | ✓ | ✓ | ✓ |
| **10** | 7,999 | 2,657 / 5,337 | 2,657 / 2,680 / 2,662 | ✓ | ✓ | ✓ |

Rule: keep index `i` iff `i mod m == 0`; the change point at `p` maps to `⌈p/m⌉`. **All three cycles and both change points survive the whole registered ladder.** Band sizes at `m=10` (15 bands): 532/531/532/531/531 · 536/536/536/536/536 · 532/533/532/533/532 — no band is emptied, so an engineer's pytest can assert these exact 15 integers. Every ladder point is in `BENCHMARK-GATE.json → decimation_ladder` including per-band counts.

⚠ **Hazard, restated because it is easy to lose:** decimation **compresses the drift timeline** — a change that took 1,000 instances takes `1000/m`. **Adaptation must be reported per-instance-since-change, never per-stream-position**, or decimation silently inflates apparent adaptation speed. Internal comparisons at fixed `m` are safe; any literature-facing sentence is not.

⚠ A second hazard I did not see registered: at `m = 10` a **window-1000** prequential metric covers 10,000 original instances ≈ 38 % of a cycle. The metric window must be decimated with the stream (`1000/m`) or the reported curve is a different estimator from Souza's.

---

## 7. Pre-registration scored (PREREG.md §3)

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | ARF-100 = 77.0 (80 % CI 73–80) | **78.81** | ✔ in interval |
| P2 | B1 passes (p = 0.55) | **passed** | ✔ |
| P3 | kNN_S(5000, std) = 76.0 (70–82) | **68.15** | ✘ **missed low by 7.9**, outside my interval |
| P4 | SAM-kNN(5000, std) = 78.0 (72–84) | **76.92** | ✔ in interval |
| P5 | kNN_S(1000, std) = 73.0 (66–80) | **75.36** | ✔ |
| P6 | kNN_S(5000, raw) = 62.0 (50–74) | **64.11** | ✔ |
| P7 | `b2_margin_pts` = +1.0 (−4 to +6) | **−1.898** | ✔ in interval, sign wrong |
| P8 | `criterion4_cleared = true` (p = 0.25) | **false** | ✔ the low prior was right |
| P9 | ARF κ_per > 0 (p = 0.97) | **0.644** | ✔ |

**The one real miss is P3**, and it is instructive: I predicted that a 5,000-example window would *be* the per-regime classifier. It is 6.3 % of the stream — already **too long**, and the optimum sits near L = 500. My mechanism was right and my scale was wrong by an order of magnitude, which is why the ladder in §3.3 is the reportable object rather than any single L. **The Hub's Q2 = 0.55 (criterion 4 clears) resolves NO**; my pre-registered 0.25 was the better-calibrated prior, for the reason stated in advance ("recency is the hidden regime variable").

---

## 8. Declared NOT-RUNs (never to be reported as nulls)
`out-of-control` (**no data source exists** — §1) · any baseline on the imbalanced-reoccurring stretch stream (frozen and hashed, no arm run) · any decimated (`m > 1`) baseline (the ladder is characterised structurally only; the published comparison is `m = 1`) · SAM-kNN at L = 14,782 (skipped: the reference implementation allocates an `(L+1)²` float64 distance cache = **1.75 GB** at that L; **kNN_S at 14,782 was run** and is the byte-matched exemplar point, 59.75 %) · Metro Interstate drift map · **any CLU cell of any kind** · any retention/adaptation `R(b)`/`A(b)` claim (S4's, not mine; §4c is a diagnostic on baseline arms).

## 9. Limitations and confounds (stated, not minimised)
1. **The 0.10-point margin.** `criterion4_cleared` is a threshold on a continuous quantity and we landed near it against our strongest ARF. It fires under all five ARF references and on the second condition, but the honest sentence is "the byte-matched exemplar store is at ARF's shoulder", not "beats it".
2. **`river` ARF ≠ MOA ARF.** Ours is +1.68 above Souza's on the primary and +1.28 on the second condition — inside tolerance, but they are different implementations, and the direction (ours stronger) is the one that makes criterion 4 *harder* to fire.
3. **Exemplar arms are single-run** (deterministic given the fixed stream and `random_state=0` KMeans) — there is no seed variance to quote, only implementation variance, which is bounded by the Weather validation (≤0.2 points of the authors' published numbers).
4. **My SAM-kNN is a port, not the authors' binary.** It reproduces their published Weather row to 0.04/0.15 points and its kNN shims are unit-tested exactly, but a residual port difference cannot be excluded.
5. **The causal standardiser is my choice**, not Souza's or Losing's (who used raw features on Weather). It is declared, pre-registered, causal, and both variants are reported.
6. **`κ` uses marginal-product chance agreement** computed over the whole scored range, not a windowed κ; Souza does not state which he used.
7. **§4c's revisit diagnostic is not `R(b)`** as PREREG §2.2 defines it (no post-re-entry 1,000-instance anchor). It is a cycle-mean contrast on the same pass and should be read as a bar, not as the metric.

## 10. Recommended next experiments (for the Hub, in priority order)
1. **Take the pre-authorized Metro fallback for the VALUE surface** and file INSECTS as the registered admissibility finding. No round-trip needed (Head ruling 1).
2. **Re-source or re-scope V3.** `out-of-control` does not exist in the distributed archive. Either find it (Souza's authors direct) or replace the drift-free null — INSECTS `incremental_balanced` (57,018, 6 classes, in river and in the archive) is the nearest same-feature-space candidate, but it is *not* published as drift-free.
3. **Before any Metro cell: run this exact tripwire on Metro.** It cost ~1 h of CPU here and it saved the wave a lifecycle build. A 24-h-horizon regression stream with a hidden clock has an obvious nearest-neighbour-over-past-windows attack (scout §3 C2 criterion 4 ⚠), and it is un-argued.
4. **Adopt a window ladder as standing practice** for any future exemplar launder: a single `L` cannot represent the family (76.03 at L=500 vs 59.75 at L=14,782 on the same data, same code).
5. **Amend the band map to bands 0–3** before any `R(b)` cell, and require per-band No-Change in every band table.
6. Optional and cheap: SAM-kNN at L=14,782 with a memory-light distance cache, to complete the CLU-byte-matched row on the exemplar side (kNN_S is done at 59.75 %).

## Git footprint
**None.** No branch, no commit, no worktree, no push. `git status --porcelain` shows **0 modified tracked files** before and after; HEAD is `7fcef50` throughout. All artifacts live under `.claude/`.

## Bug reports for `experiment-engineer`
No bug found in `chlu/` (I ran none of it). Two loader hazards to encode in the harness:
- **Class labels are `{2,3,4,5,11,12}`, not `0..5`.** A harness that assumes 0-based labels will silently mis-map.
- **The frozen CSV is headerless and unnormalised** (range 0.0020 → 8,014.5). Whatever scaling the CLU harness applies must be **causal** (no global statistics) or the prequential protocol is broken; `run_arm.py::causal_standardise` in `…/c2w10-benchmark-gate/code/` is a reference implementation.
- **Reproduction gate:** the harness's loader must reproduce `f267c0fb1f4e1d68967e2a694305146425181c19a7d3c282b67b252b98407c7a` for the primary stream **before** any C2W10 cell consumes a number from `BENCHMARK-GATE.json`. Mismatch = hard stop.

## Artifacts (exact paths)
- **`.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json`** — the gate file, every required key present, `criterion4_cleared` computed arithmetically
- `.claude/outputs/c2w10-benchmark-gate/PREREG.md` · `fig1_prequential_curves.png` · `fig2_window_ladder.png`
- `metrics.json` / `metrics_table.txt` (all 28 runs) · `bandstats.json` / `.txt` · `structure.json` (cycles, band map, decimation ladder) · `facts.json` · `arfbytes_{10,100}_1.json` · `scratch_venv_freeze.txt`
- `code/` — every script, verbatim, including the SAM-kNN port and its unit tests
- Frozen streams: `.claude/data/c2w10-streams/` (3 CSVs + the 600 MB USP archive of record)
- Raw per-instance predictions for all 28 runs: `.claude/scratch/c2w10/preds/*.npz` (kept; ~2 MB total)

---

## Proposed handover updates (for the Hub)

**§1.6 (experiments) — new rows**
- **C2W10 benchmark gate, 2026-08-10, HEAD `7fcef50`, scratch venv `river 0.25.0`, `m=1`.** B1 **PASS**: No-Change **40.4526** vs published 40.46 (Δ −0.007); ARF-100 **78.8139 ± 0.0526 SD** (3 seeds) vs published 77.13 (Δ +1.684). B3 **PASS**: ARF κ_per **0.64422**. B2: best registered exemplar arm **SAM-kNN L_max=5000 (std) = 76.9157**, `b2_margin_pts = −1.8983`, **`criterion4_cleared = FALSE`** ⇒ **criterion 4 has FIRED on the PRIMARY**; Metro fallback is pre-authorized, INSECTS is filed as a registered admissibility finding ("INSECTS is metric-native at matched bytes"), the **fifth confirmation of the criterion-4 theorem**, for ≈4 h wall-clock on CPU.
- **Robustness of the verdict:** fires against every ARF reference — ours-100 −1.898, ours-100 best seed −1.937, worst seed −1.838, ours-10 −0.497, Souza's published 77.13 → **−0.214**. Unregistered stronger arm **SAM-kNN L_max=1000 (std) = 77.0632** (margin −1.751; **−0.07 vs the published ARF**) at **133 kB of state**.
- **Second condition confirms:** inc-abrupt-reoccurring bal. — No-Change 42.3779 (pub. 42.39), ARF-100 76.2251 ± 0.0381 (pub. 74.95), best exemplar SAM-kNN(1000,std) 74.3802, margin **−1.845**, `criterion4_cleared = false`.
- **The mechanism, quotable:** exemplar accuracy is **monotonically decreasing in store size** above L≈500 — kNN_S(std) 75.56 / **76.03** / 75.36 / 73.39 / 68.15 / 59.75 at L = 250/500/1000/2000/5000/14782. The **CLU-byte-matched** store (14,782 exemplars = 1,966,006 B) is the **worst** exemplar arm. SAM-kNN discovers this: at L_max=5000 its STM averages **945** of a permitted 3,000. ARF too: **78/100 trees hold 1 node** at the end of the pass. On INSECTS the winning strategy is *forget fast*.
- **The bar for V1, measured:** SAM-kNN(1000) − best fixed window kNN_S(500) = **+1.03 pts** (the literature's own persistent-vs-episodic contrast). Cycle-3-minus-cycle-1 over bands 0–3: ARF **+0.457**, SAM-kNN(1000) **+0.537**, SAM-kNN(5000) −0.084, kNN_S(1000) −0.173, **No-Change −0.474 (the noise floor)** ⇒ **no existing method extracts a material revisit benefit on this stream.**

**§5 (provenance) — new rows**
- Frozen streams, `.claude/data/c2w10-streams/`: `incremental_reoccurring_balanced.csv` sha256 **`f267c0fb1f4e1d68967e2a694305146425181c19a7d3c282b67b252b98407c7a`** (21,433,047 B; 79,986 × 33; 6 classes × 13,331); `incremental_abrupt_balanced.csv` **`c1cd19d429844349009311fa372ee700d5bf0e21545fb61d0d3fba724123188b`** (21,421,452 B); `incremental_reoccurring_imbalanced.csv` **`c881f34f4ba8029dfe9cdb948b20026a8fcd706c19ce3e8efc7bb5682dd4a3d6`** (120,798,571 B; 452,044). Source of record: `USP_DS_Repository.zip` (599,633,529 B, CC BY 4.0, no password needed).
- **river's mirror ≡ the USP original, bit-for-bit** (sha256 of the extracted archive file equals sha256 of river's download, both balanced variants). This retires the "which copy did we use" question for the whole program.
- **Byte ledger:** 133 B/exemplar; L=5000 → **665,000 B (0.63419 MiB)**; L=1000 → 133,000 B; CLU d=12 → **1,966,080 B (1.875 MiB) = 2.9565× SAM-kNN(5000)**; CLU budget buys **exactly 14,782 exemplars** (74 B slack — confirms PREREG §6.3). **ARF measured state: 991,534 B (10 trees) / 9,542,925 B (100 trees) = 14.35× SAM-kNN(5000), 4.85× the CLU store** ⇒ **ARF is the largest store in the comparison and is not a byte-matched arm.**
- **Reproduction gate is LIVE:** no C2W10 cell may consume a number from `BENCHMARK-GATE.json` until the project harness's loader reproduces the primary sha256. Mismatch = hard stop.

**§8 (NOT-RUNs / registry) — new entries**
- ⛔ **`out-of-control` (905,145 × 24 classes) has no data source.** Absent from the 2024-04-16 USP archive (10 INSECTS files, all 6-class) and from river 0.25.0. **PREREG-C2W10 §6.2's V3 leg is a declared NOT-RUN**, never a null, until re-sourced.
- ⛔ Imbalanced-reoccurring: frozen and hashed, **no arm run**.
- ⛔ SAM-kNN at L=14,782: NOT-RUN (reference impl. allocates a 1.75 GB `(L+1)²` distance cache). kNN_S at 14,782 **was** run (59.75 %).

**Corrections the Hub should push to the registries**
- `negative_results.md` / benchmark registry: **N-new-c — "INSECTS reoccurring is metric-native at matched bytes."** A 133 kB byte-matched exemplar store scores within **0.07 points** of the published ARF and within **1.75 points** of our stronger ARF; store accuracy *decreases* with store size. Fifth confirmation of the criterion-4 theorem.
- **Correct `c2w10-benchmark-scout` §5**: river does **not** ship SAM-kNN (0.25.0 `neighbors` = KNNClassifier/KNNRegressor/LazySearch/SWINN). The launder is a ported reference implementation, validated against the authors' published Weather row (ours 21.70 % / 21.68 % error vs published 21.74 / 21.53).
- **Amend PREREG §2.2's band map** (append as AMENDMENT, per §9): **band 4 of every cycle is degenerate** — entropy 0.84/1.05/0.84 bits, majority class 54–73 %, No-Change 91.25/71.32/90.78 %, every arm 97–99 %; longest single-label run **3,597** instances at index 76,389; >90 %-persistence regions **[23634,26753], [49007,49209], [53269,53561], [77043,79985]** = 8.30 % of the stream, two of them straddling the published change points. **Restrict `R(b)`/`A(b)` to bands 0–3** and require per-band No-Change in every band table.
- **AMENDMENT 2 (the `m` line) is unblocked from my side:** all of `m ∈ {1,2,5,10}` preserve three cycles and both change points with the counts in `decimation_ladder`. ⚠ Add to the amendment: at `m>1` the **prequential window must be decimated too** (`1000/m`), else the reported curve is a different estimator from Souza's.
- **Naming**: Souza's "incremental-abrupt-reoccurring (balanced)" = river `incremental_abrupt_balanced` = USP `INSECTS incremental-abrupt_balanced.csv` (sha256-identical). **Labels are `{2,3,4,5,11,12}`, not 0-based.**
