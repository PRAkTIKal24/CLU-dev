# camels-pub-tripwire — results-analyst report

Task + acceptance criterion: run the matched-byte tripwire on the **CAMELS-US ungauged (PUB)** split
and return a fired / did-not-fire verdict with the margin against **every** reference in the frozen set.
**Status: done — ⛔ THE TRIPWIRE FIRED.**

## ⛳ ESCALATION + RECONCILIATION LIST (first 10 lines, protocol §5 corollary)
1. ⛔ **THE VENUE DIES. `dies = TRUE`.** Best in-budget, non-leaky, full-531-basin store arm
   **E = 0.6369** (mean over 3 fold-seed reconstructions; best single run 0.6435). It fires against
   **7 of 20** references (R14–R20), including the calibrated **SAC-SMA (0.6447, margin −0.0078)** and
   the **NWM (0.5766, margin +0.0603 — the store BEATS it)**. Per the pre-registered kill rule I have
   stopped, written up, and am **escalating to the Head and the Main Advisor.** ⛔ Not softened, not
   re-framed. **A fired tripwire falsifies the VENUE, not the CLU.**
2. ⭐ **Mechanism, measured:** the Li-2025 PUB split is a *random* basin holdout, so 71 % of held-out
   basins keep a gauged training neighbour within 50 km. Median NSE of a 3-neighbour flow copy is
   **0.822** for basins with a neighbour < 10 km and **−0.157** beyond 100 km (Spearman −0.542).
   The venue is scoring gauge density, not hydrology.
3. ⚠ **Correction to the frozen set (owner needed):** `qu2026_*` (R18/R20) are **671 basins, 5-fold,
   test 2010–2014, bootstrap-CDF median** — NOT the 531/PUB protocol the refs output attributes to them.
   Numerically confirmed from the primary (A6); protocol metadata was wrong. See §7.
4. ⚠ **My own harness bug, caught and fixed:** an `L{n}_all` tag collided across folds with 477 vs 478
   train basins, splitting the per-basin accumulator so 1,920 rows had medians over 54 or 477 basins
   instead of 531. Fixed; a hard `n_basins == 531` guard now gates verdict admissibility. §8.
5. ⚠ **Pre-registered prediction P15 (`dies = FALSE`, p = 0.70) is REJECTED**, as are P7, P11, P12,
   P13, P14, P18. My prior was wrong in a specific, informative way (§6).

## ⭐ DIAL DECLARATION (echoed from the task file, protocol §7)
- **Dial:** *none — venue admissibility / instrument.* ⛔ No CLU cell, store, byte dividend or
  performance number is computed or implied. This measures the **classical** competition only.
- **Laundering control:** shuffled-order null + companion rows. Ran; see §5.
- **Falsifies (fires):** best in-budget store arm within 0.02 abs / 2 % rel of **any** reference. **IT DID.**
- **Does NOT falsify:** the store losing to the LSTM references — and indeed it still loses to the
  forcing-matched primary R4 by −0.1054 and to the frontier R1 by −0.1489. That part is the
  metric-native-ceiling theorem and is *not* the finding. The finding is the bottom of the board.

---

## 1. Setup — flag provenance (protocol §5, mandatory)

| item | value |
|---|---|
| Repo HEAD at PREREG filing **and** at report | `7fcef50`, `git status --porcelain` **empty** both times |
| Tracked-code edits / branches / worktrees | **none** — read-only on `chlu/`; all artefacts under `.claude/` |
| Python / numpy / sklearn / pandas | `.claude/scratch/c3-trackb-tripwire/.venv`, numpy **2.5.2**, sklearn **1.9.0**, pandas **3.0.5** |
| Machine | M1 Pro, 8 cores, 16 GB RAM, CPU-only, local-only (task A8) |
| **Protocol (verdict-binding)** | **Li et al. 2025 PUB** — 531 basins, 10 folds, every basin held out exactly once |
| Store ("train") window | **1980-10-01 → 1999-09-30** |
| Test window | **1995-10-01 → 1999-09-30** = **1,461 days** (asserted) |
| Forcing | **Daymet only** (maurer is defective; guard re-armed, §3) |
| Metric | NSE **unclipped**, `obs < 0` dropped first (Kratzert `calc_nse`), per basin then **median over 531** |
| Fold assignment | ⛔ **RECONSTRUCTED, NOT PUBLISHED** — `KFold(n_splits=10, shuffle=True, random_state=s)` over `basin_list_531.txt` order, **s ∈ {0,1,2}** |
| Statics | **Feng/Li 35-attribute set** (task A4), `attrs35.npz` sha256 `1904cfef7c5e1ed5fbc2e3cef50c4642a972a6cd1dfde127803adf9748695343` |
| Byte budget | **B = 1,966,080 B**; parameter-matched **B_param = 1,184,772 B** (296,193 params × 4) |
| Selection seeds | random {0,1,2}, k-means {0,1,2} |
| Peak RSS | **1.33 GB** max over all arms (of 16 GB) — nothing came near the RAM ceiling; ⛔ nothing NOT RUN for RAM |
| **A2 leak assertions executed** | donor fs0 grid **120** · donor all-donor rerun **360** · DAY screen **17 × 10** · DAY ladder **10/config** · companions **10** · diagnostic **10** — **≥ 690 total, zero fired** |

**Commands (all reproducible):**
```
.venv/bin/python code/build_attrs35.py
.venv/bin/python code/guard_maurer.py
.venv/bin/python code/b1_pub_control.py
.venv/bin/python code/pub_donor.py --fold-seeds 0 --keysets S35,S35oh,S35+clim,S35+latlon,clim,latlon \
    --payloads concurrent,doyclim --Ls 5,10,25,50,100,200,328,478 --sel-seeds 0,1,2 --dtypes fp32,fp16
.venv/bin/python code/pub_donor.py --fold-seeds 0,1,2 --keysets <all> --payloads concurrent,doyclim \
    --Ls 9999 --sel-seeds 0 --dtypes fp32,fp16          # the fixed all-donor rerun
code/run_day_screen.sh ; code/run_day_ladder.sh ; .venv/bin/python code/pub_companions.py
.venv/bin/python code/analyse_pub.py
```
Code: `.claude/scratch/camels-pub-tripwire/code/` · artefacts: `.claude/outputs/camels-pub-tripwire/`.

---

## 2. PREREG — filed before any tripwire number existed
`.claude/outputs/camels-pub-tripwire/PREREG.md` (353 lines). Contains: §0 disclosure (including that I
knew the gauged run's E = 0.25586 vs S = 0.7580, and its companion rows), the frozen reference set copied
verbatim with R4 named primary and R20 = 0.553 named the binding threshold, the full arm list, the
L-ladder and byte tables, the kill rule verbatim, and **21 numeric point predictions with 80 % intervals**.

## 3. Positive controls — both passed BEFORE any tripwire number was quoted

### 3.1 ⭐ Loader control (task A3) — **PASSED, and far better than the pre-registered bar**
Computed with **our** discharge array, **our** date/hydrological-year alignment, **our** basin ordering
(aligned by id) and **our** NSE code, against **Li 2025's published per-basin daily simulations**.
Zenodo 10.5281/zenodo.16895228, CC BY 4.0. **Published md5s reproduced exactly:**
`py999_PUB_simulations.npy` 530,583,824 B `1f83befcd26861e7588075522e1658d2` ✓ ·
`py999_OBS.npy` 43,444,424 B `1d913eaf2cbe4c0807a6ccc1891fb8f2` ✓.
sha256 (ours): `4ab3a47066f29c898d3b27dfb2aec2c76287828bed283385e14e54247c18a8e9` /
`299fb047eb63616f9073ddbe76eb2bc891233f00a25cfc307ba2664cf65128c4`.

**Pre-registered tolerance: |Δ median NSE| ≤ 0.010 on all three targets. Observed worst: 0.00142.**

| target row | published | ours | Δ |
|---|---|---|---|
| LSTM + δHBV, Daymet | 0.742270 | **0.740850** | −0.00142 |
| LSTM, Daymet | 0.702636 | **0.702933** | +0.00030 |
| δHBV, Daymet | 0.706809 | **0.706600** | −0.00021 |

⭐ **Stronger than required: all 12 published Table D2 rows reproduce**, worst |Δ| = 0.0024, including the
**frontier 0.785833 → 0.784074** (Δ −0.00176). File: `B1_pub_loader_control.json`.

**Metric convention identified (registered in advance as a thing to identify, not to tune):** members are
averaged **within a seed**, then per-seed median over 531, then **mean over the 3 seeds** ("convention C").
The two obvious alternatives are both wrong and are printed for the record: NSE of the grand-mean sim
over-scores the single-model rows by up to +0.037; per-run-median-then-mean under-scores the ensemble rows
by up to −0.093. ⚠ Anyone reproducing Li's ensemble rows must use convention C.

**Basin-id ordering (A3 cross-check):** `py999_ids.npy` is the same **set** as `ref/basin_list_531.txt` but
a different **order** — **439 of 531 positions differ** (Zenodo is numerically ascending; the published
list is not). All comparisons align **by id**. Aligning by position would have silently corrupted every
margin.

### 3.2 Forcing guard (task §2) — **re-armed and TESTED, not merely asserted**
`GUARD_maurer.json`: guard **passes** on staged Daymet (`frac(tmax > tmin) = 0.99951`, all 531 basins,
full record) and **fires** on maurer read live from the source zip (29 basins sampled;
`frac(tmax > tmin) = 0.0`, `frac(tmax == tmin) = 1.0`; `assertion_raised = true`). 1 of 30 sampled maurer
files has the known malformed header (`05120500`) and is skipped, reported, not silently dropped.

---

## 4. Results — the arm ladder

Full tables: `TABLES.md`. Machine-readable: `VERDICT.json`, `donor_arms.jsonl` (42,240 rows),
`donor_all_fixed.jsonl` (2,880), `day_screen.jsonl` (150), `day_arms.jsonl`, `companions.jsonl`.
Per-basin NSE vectors for every arm are saved as `perbasin_*.npz` (111 artefact files).
**47,559 arm rows total; 34,492 verdict-admissible; 1,920 excluded by the `n_basins == 531` guard.**

### 4.1 ⭐ Family DONOR — the donor-basin regionalization analogue (the dangerous arm)
Exemplar = one **training-fold** basin; payload = that donor's **concurrent** daily specific discharge over
the 1,461 test days (legitimate under A2: donors are gauged train-fold basins, and Li's train period
1980–1999 covers the test dates for them), or its 366-day DOY climatology. Best per key space, in budget,
non-leaky, n = 531:

| key space | payload | best median NSE | bytes | config |
|---|---|---|---|---|
| **`clim`** (10 forcing-climate descriptors) | concurrent | **0.6435** | 1,412,874 | Lall, k=5, dist, pmean, **fp16** |
| **`latlon`** (spatial proximity) | concurrent | **0.6371** | 1,397,610 | Lall, k=5, dist, pmean, fp16 |
| `S35+clim` | concurrent | 0.5590 | 1,482,756 | Lall, k=5, dist, pmean, fp16 |
| `S35+latlon` | concurrent | 0.5174 | 1,467,460 | Lall, k=3, dist, pmean, fp16 |
| **`S35`** (the 35 Feng/Li statics alone) | concurrent | **0.5089** | 1,463,636 | Lall, k=3, dist, pmean, fp16 |
| `S35oh` (one-hot statics) | concurrent | 0.3897 | 1,530,556 | Lall, k=10, dist, pmean, fp16 |
| any | doyclim | 0.1041 | 369,972 | `clim`, Lall, k=3, dist, pmean, fp16 |

⭐ **The fp16-payload variant is what puts all ~478 donors inside the budget** (1.41 MB vs 2.81 MB at fp32)
and it is **lossless to 4 dp** (0.6435 fp16 vs 0.6435 fp32). This was pre-registered in PREREG §4 as
mandatory precisely so a float-precision convention could not decide the verdict. It didn't — but it did
decide *admissibility*: at fp32 the same arm is 1.43 × over budget.

**Fold-seed spread (P20):** best config `clim/concurrent/Lall/k=5/dist/pmean/fp16` →
fs0 **0.6405**, fs1 **0.6268**, fs2 **0.6435**; mean **0.6369**, spread **0.0167**.
The spatial `latlon` k=3 arm, independently recomputed with great-circle distance, gives
**0.6270 / 0.6240 / 0.6207** across the same three seeds — the result is not a fold-seed artefact.

**At the parameter-matched budget B_param = 1,184,772 B:** best admissible arm **0.6224**
(`clim`, 328 k-means-condensed donors, 867,866 B). ⚠ The two budgets differ by 1.66 ×; the verdict uses
the more generous B, and **the tripwire fires at both** (0.6224 still fires against R15–R20).
**Escalated to the Hub, not silently picked**, as the task requires.

### 4.2 Family DAY — the task-§3 literal arms (statics / forcing-window / both)
At-budget screen, fold seed 0, best non-leaky over k × weight × target (`day_screen.jsonl`):

| key space | scaling | L@B | median NSE |
|---|---|---|---|
| W1 forcing only | global | 81,920 | **0.2904** ← best DAY arm |
| W30 forcing only | global | 3,255 | 0.2717 |
| W30 ⊕ 35 statics ("both combined") | global | 2,642 | 0.2584 |
| W30 ⊕ statics | perbasin | 2,642 | 0.2402 |
| W1 ⊕ statics | global | 11,988 | 0.2019 |
| W30 ⊕ one-hot statics | global | 2,224 | 0.1949 |
| W365 ⊕ statics | global | 264 | 0.0469 |
| W30 ⊕ statics | raw | 2,642 | 0.0419 |
| **statics only (W = 0), 35 attrs** | global | 13,653 | **−0.0254** |
| statics only, one-hot | global | 6,922 | −0.0260 |

⭐ **The literal "kNN over static catchment attributes" DAY arm is degenerate, exactly as pre-registered
(P6 = 0.03, measured −0.0254).** With one key per basin, all days of a donor share a key, so the retrieval
returns arbitrary days. **This is why Family DONOR had to be built**: running only the literal arm would
have hobbled the dangerous arm and declared the venue safe. The task's "what would prove this spoke wrong"
clause anticipated exactly this, and it is the reason the verdict came out the other way.
The DAY family is **not** the deciding family: its best (0.2904) is 0.35 NSE below the donor family, and is
almost identical to the **gauged** run's 0.2559 — a useful cross-run consistency check.

### 4.3 DAY full L-ladder — **RUN**, three §3 families, selection seeds {0,1,2} rand + {0,1,2} k-means
Best non-leaky median NSE over k × weight × target × selection × seed at each ladder point:

| family | L=250 | 500 | 1000 | L@B_param | 2000 | **L@B** |
|---|---|---|---|---|---|---|
| **statics only (35 attrs)** | −0.0084 | −0.0090 | −0.0240 | −0.0232 (8,227) | −0.0257 | **−0.0224** (13,653) |
| **W30 forcing-window only** | 0.2072 | 0.2366 | 0.2572 | 0.2701 (1,961) | 0.2694 | **0.2793** (3,255) |
| **W30 ⊕ 35 statics ("both")** | 0.1721 | 0.2156 | 0.2412 | 0.2577 (1,592) | 0.2622 | **0.2634** (2,642) |

⭐ The statics-only family is **flat and negative at every ladder point** — its degeneracy is structural,
not a budget artefact. Over-budget DAY upper bounds (`[sub3]`, never admissible):
**1.89 × B → 0.2885**, **7.57 × B → 0.2525** (worse). ⇒ **even an unbounded DAY store stays near 0.29**;
the DAY family cannot approach E under any budget.

### 4.4 Ladder status — reported honestly
- Donor ladder `L ∈ {5,10,25,50,100,200,328,all}` × {rand, k-means} × seeds {0,1,2} × {fp32, fp16}:
  **RUN in full**, fold seed 0; all-donor point additionally run at fold seeds 1 and 2.
- DAY at-budget screen across all 10 key-space × scaling combinations: **RUN**.
- DAY full L-ladder + over-budget points + shuffled-order null: **RUN** (`day_arms.jsonl`, 3,510 rows).
- **k-means condensation is capped at L ≤ 5,000** (declared limit, `KMEANS_LMAX`): above that
  MiniBatchKMeans is prohibitive here and the store already covers most of the distinct key space.
  Larger L points use random selection only. ⚠ Declared, not silent.
- **NOT RUN:** the Kratzert-2019 12-fold secondary leg (PREREG §2) — pre-registered as
  "run if time allows"; it was not reached. ⛔ Reported as NOT RUN, not as zero.
- ⛔ Nothing was subsampled to fit RAM; no basin set was ever shrunk. Peak RSS 1.33 GB.

---

## 5. Companion rows and the laundering control (§5, §6)
`companions.jsonl`, fold seed 0, all n = 531:

| companion | leaky? | bytes | median NSE | mean | n(NSE ≤ 0) |
|---|---|---|---|---|---|
| **spatial-NN donor, k = 3** (great-circle) | no | 2,791,404 fp32 / **1,400,540 fp16 (in budget)** | **0.6271** | −0.3165 | 68 |
| spatial-NN donor, k = 5 | no | as above | 0.6006 | −0.2820 | 74 |
| spatial-NN donor, k = 1 | no | as above | 0.5860 | −0.9037 | 100 |
| 1-day persistence | ⛔ **LEAKY, DIFFERENT TASK** | 4 | 0.4362 | 0.3931 | 103 |
| per-basin DOY climatology | ⛔ **LEAKY** | 1,468 | 0.1421 | 0.1762 | 25 |
| **train-fold regional DOY climatology** | no | 1,468 | **0.0036** | −0.8812 | 253 |
| per-basin test-period mean | ⛔ leaky, definitional | 4 | 0.0000 | 0.0000 | 531 |
| per-basin store-period mean | ⛔ leaky | 4 | −0.0048 | −0.0194 | 531 |
| **train-fold regional mean flow** | no | 4 | **−0.0319** | −0.7815 | 531 |

⭐ **The laundering control is decisive and it clears the store.** The non-leaky calendar control
(regional DOY climatology) scores **0.0036** and regional mean flow **−0.0319**. E = 0.6369 is
**0.63 NSE above the calendar**. The winning arm is emphatically *not* measuring the day of the year —
it is measuring the observed flow of a nearby gauge. That makes the finding worse for the venue, not
better: the store is exploiting a real, admissible, protocol-legal signal.

**Best leaky / out-of-protocol arm (never admissible, reported for contrast):** 0.6759
(`latlon`, concurrent, oracle rescaling). Note it is only **+0.032** above the best legitimate arm —
i.e. the leak buys almost nothing here, which is itself evidence that the concurrent-donor signal, not
the target normalisation, is doing the work. **Pre-registered P18 (leak worth ≥ 0.08) is FALSE.**

**Shuffled-order null (§6) — RUN.** W30 ⊕ 35 statics, L = 2,642, seeds {0,1,2}, time slots permuted
jointly across all 5 channels in **both** keys and queries:
**ordered 0.2634 → shuffled 0.0789, Δ = −0.1845.** ⇒ the forcing window's *temporal ordering* carries
most of the DAY family's signal; the DAY arm is not laundering a bag-of-days. (Pre-registered P16 =
−0.08, 80 % interval [−0.25, +0.02] — **survived**.)
⚠ The null leg bears on the DAY family only: the deciding DONOR arm has no time-ordered key to permute
(its key is climate descriptors / lat-lon / statics), so it cannot change the verdict.

---

## 6. ⭐ The mechanism — why the venue dies (`DIAG_neighbour_distance.json`)
Arm: `latlon` k = 3, concurrent payload, fp16, fold seed 0, all 531 basins.

| nearest train-fold neighbour | n basins | median NSE |
|---|---|---|
| **< 10 km** | 47 | **0.8215** ← above the published frontier (0.7858) |
| 10–25 km | 123 | 0.7526 |
| 25–50 km | 207 | 0.6632 |
| 50–100 km | 116 | 0.4068 |
| **> 100 km** | 38 | **−0.1574** |

Nearest-neighbour distance: median **34.5 km**, p10 10.8 km, min **0.63 km**.
**2.4 %** of held-out basins have a gauged neighbour within 5 km, **8.9 %** within 10 km,
**32 %** within 25 km, **71 %** within 50 km, **93 %** within 100 km.
Spearman(NN distance, NSE) = **−0.542**.

⇒ **Li-2025 PUB holds out *basins*, not *regions*.** A random 10 % holdout of a dense CONUS gauge network
leaves almost every "ungauged" basin with a close gauged neighbour, and copying that neighbour's
concurrent specific discharge is worth median NSE 0.64 for ~1.4 MB and no training. The published
LSTMs still win by ~0.10, so the venue is not *meaningless* — but its **bottom half is not a floor**, and
any claim of the form "we beat SAC-SMA / the NWM / a CAMELS-attribute LSTM on PUB" is now unsafe: a
1.4 MB nearest-gauge lookup does that too.

---

## 7. THE VERDICT — kill rule vs EVERY reference (`VERDICT.json`)

**E = 0.6369** (seed-mean; best single run **0.6435**; at B_param **0.6224**).
Best over-budget upper bound 0.6435 (same arm at fp32, 1.43 × B) — the store is **budget-saturated**,
not budget-limited.

| id | reference | S | status | protocol vs ours | margin_abs | margin_rel | FIRES |
|---|---|---|---|---|---|---|---|
| R1 | li2025_LSTM+δHBV_123 ⭐frontier | 0.785833 | confirmed | exact | −0.1489 | −0.1895 | no |
| R2 | li2025_LSTM_123 | 0.768181 | confirmed | exact | −0.1313 | −0.1709 | no |
| R3 | li2025_δHBV_123 | 0.757749 | confirmed | exact | −0.1208 | −0.1595 | no |
| R4 | **li2025_LSTM+δHBV_daymet (PRIMARY)** | 0.742270 | confirmed | exact | **−0.1054** | **−0.1419** | no |
| R5 | heudorfer2026_EMB-LSTM_AE+ | 0.721 | own Table 3 | 5-fold, 1989–99, bootstrap | −0.0841 | −0.1166 | no |
| R6 | li2025_δHBV_daymet | 0.706809 | confirmed | exact | −0.0699 | −0.0989 | no |
| R7 | li2025_LSTM_daymet | 0.702636 | confirmed | exact | −0.0657 | −0.0935 | no |
| R8 | heudorfer2026_baseLSTM_CAMELS27 | 0.696 | own Tables 2/3 | 5-fold, 1989–99 | −0.0591 | −0.0849 | no |
| R9 | li2025_LSTM_maurer | 0.694156 | confirmed | maurer forcing | −0.0572 | −0.0825 | no |
| R10 | kratzert2019_LSTM_pub_ensemble | 0.6931 | Hub-recomputed | 12-fold, 1995–2010, NLDAS-ext | −0.0562 | −0.0811 | no |
| R11 | feng2023_δHBV_daymet | 0.69 | provisional | 10-fold, 1989–99 | −0.0531 | −0.0769 | no |
| R12 | li2025_δHBV_nldas | 0.670636 | confirmed | nldas forcing | −0.0337 | −0.0503 | no |
| R13 | feng2023_LSTM_daymet | 0.66 | provisional | 10-fold, 1989–99 | −0.0231 | −0.0350 | no |
| R14 | feng2023_LSTM_maurer | 0.65 | confirmed | 10-fold, 1989–99 | **−0.0131** | **−0.0201** | **YES** |
| R15 | **kratzert2019_SACSMA_calibrated** | 0.6447 | Hub-recomputed, 531 rows | per-basin calibrated, 1995–2010 | **−0.0078** | **−0.0121** | **YES** |
| R16 | feng2023_δHBV_maurer | 0.64 | confirmed | 10-fold, 1989–99 | **−0.0031** | **−0.0048** | **YES** |
| R17 | heudorfer2026_baseLSTM_dynamiconly | 0.634 | own Table 2 | 5-fold, 1989–99 | **+0.0029** | **+0.0046** | **YES** |
| R18 | qu2026_ALPHAEARTH | 0.612 | ⭐primary-confirmed here | ⚠ **671 basins, 5-fold, 2010–14** | **+0.0249** | **+0.0407** | **YES** |
| R19 | **kratzert2019_NWM_reanalysis** | 0.5766 | Hub-recomputed, 531 rows | NWM retrospective, 1995–2010 | **+0.0603** | **+0.1046** | **YES** |
| R20 | qu2026_CAMELSattributes | 0.553 | ⭐primary-confirmed here | ⚠ **671 basins, 5-fold, 2010–14** | **+0.0839** | **+0.1518** | **YES** |
| ⛔ | ~~feng2023_MPR+mHM 0.53~~ | — | **STRUCK** (refs §8 V5: in-sample) | — | — | — | not used |

**`dies = TRUE`. 7 of 20 references fired.** Even discounting the two protocol-mismatched Qu rows
(R18, R20) and the provisional Feng rows, the store still fires against **R15 (calibrated SAC-SMA)**,
**R17** and **R19 (the NWM)** — all three of which are **Hub-recomputed or own-table confirmed**.
⛔ The verdict does not depend on the weakest or the shakiest row.

### 7.1 A6 — Qu et al. 2026 retrieved from the primary (`A6_qu2026_primary.json`)
arXiv:2601.01558**v2**, PDF sha256 `115f9e4b987f0e74db890bcdcd2dde7173e4b017e5e30e3beffdd920bb6a44bb`,
8,569,509 B. **Numbers CONFIRMED verbatim:** *"the median NSE of the AEF embeddings rose to 0.612, while
the model relying on CAMELS attributes reached only 0.553"* (also IS: 0.708 / 0.706). ⇒ R18/R20 upgrade
from ⚠secondary to **primary-confirmed**.
⚠ **But the protocol metadata in the frozen set is wrong.** Primary says: *"We used all **671** catchments"*,
*"**five** random seeds and a **five-fold** cross validation"*, train 1980–2004 / val 2005–2009 /
**test 2010–2014**, metric = 5,000× bootstrap of the pooled basin-level NSE, median of the bootstrap CDF
(Heudorfer convention). **Not** 531, **not** 10-fold, **not** 1995–1999, **not** a plain median.
Also: the paper's Experiment B *is* the published donor-basin-selection study (top-k similar donors,
k ∈ {100…600}, 150 Köppen-stratified target basins; AEF similarity peaks at k = 400) — i.e. our dangerous
arm is a published method, as the task said.

---

## 8. Honesty section — what went wrong, and what I did not run
1. ⚠ **Harness bug I introduced and caught.** `condense()` tagged the all-donor store `L{n}_all`, and
   `n` is 477 or 478 depending on the fold, so the per-basin accumulator split by tag. 1,920 of 42,240
   rows had medians over **54** or **477** basins, not 531 — and the *apparent* top arm was 0.6559 over
   54 basins. Fixed (tag → `Lall`), the arm rerun on 3 fold seeds, and a hard `n_basins == 531` guard
   added to the analyser so no subset row can ever be verdict-admissible. **Excluded rows are counted in
   `VERDICT.json` (`n_rows_excluded_not_531_basins = 1920`), not deleted.**
   ⇒ Had I not checked `n_basins`, I would have reported E = 0.6559 — wrong, and wrong in the direction
   that made the finding look stronger.
2. ⚠ **The fold partition is a reconstruction.** No PUB reference publishes its fold file (refs §3).
   My folds are `KFold(shuffle=True, random_state ∈ {0,1,2})`. Margins are therefore against the
   *published protocol* but *our* partition. The 3-seed spread is 0.0167, so this is not the dominant
   uncertainty — but the number is not "against the published partition", and I say so.
3. ⛔ **NOT RUN:** the Kratzert-2019 12-fold secondary leg. Pre-registered as optional; not reached.
   Reported as NOT RUN, never as zero.
4. ⚠ **k-means condensation capped at L ≤ 5,000** — declared limitation, affects only large-L DAY points.
5. No divergences, no NaNs, no OOM. Peak RSS 1.33 GB of 16 GB.

## 9. Scoring the pre-registered predictions (a failed prediction is a finding)

| # | prediction | point [80 % int] | measured | verdict |
|---|---|---|---|---|
| P1 | loader control passes ≤ 0.010 | PASS | 0.00142 | ✅ survived |
| P1b | axis-0 order correct | TRUE | all 12 rows matched | ✅ |
| P1c | maurer guard fires, daymet passes | TRUE | 0.0 / 0.99951 | ✅ |
| P2 | DAY S+W30 at budget | 0.24 [0.05, 0.40] | **0.2584** | ✅ |
| P3 | DAY S+W365 at budget | 0.06 [−0.05, 0.20] | **0.0469** | ✅ |
| P4 | DONOR concurrent fp16, `S35` key | 0.38 [0.10, 0.58] | **0.5089** | ✅ (high in interval) |
| P5 | DONOR doyclim | 0.05 [−0.10, 0.22] | **0.1041** | ✅ |
| P6 | DAY statics-only (degenerate) | 0.03 [−0.10, 0.15] | **−0.0254** | ✅ |
| P7 | spatial-NN donor k = 1 | 0.28 [0.05, 0.50] | **0.5860** | ❌ **above interval** |
| P8 | regional mean flow | −0.08 [−0.35, 0.05] | **−0.0319** | ✅ |
| P9 | regional DOY climatology | −0.03 [−0.25, 0.12] | **0.0036** | ✅ |
| P10 | LEAKY per-basin DOY clim | 0.05 [−0.08, 0.22] | **0.1421** | ✅ |
| P11 | **E** | 0.40 [0.12, 0.58] | **0.6369** | ❌ **above interval** |
| P12 | margin_abs vs R4 | −0.34 [−0.62, −0.16] | **−0.1054** | ❌ above interval |
| P13 | fires vs R20 | FALSE (p = 0.72) | **TRUE** | ❌ |
| P14 | fires vs R19 | FALSE (p = 0.78) | **TRUE** | ❌ |
| P15 | ⭐ **`dies`** | **FALSE (p = 0.70)** | **TRUE** | ❌ **REJECTED** |
| P16 | shuffled-order null Δ | −0.08 [−0.25, +0.02] | **−0.1845** | ✅ |
| P17 | one-hot ≤ ordinal + 0.02 | TRUE | one-hot **worse** by 0.064 | ✅ |
| P18 | leak worth ≥ 0.08 | TRUE (p = 0.75) | **+0.032** | ❌ |
| P19 | peak RSS < 8 GB | TRUE | 1.33 GB | ✅ |
| P20 | fold-seed spread of E | 0.015 [0.003, 0.05] | **0.0167** | ✅ (near-exact) |
| P21 | Qu primary confirms 0.553/0.612 | TRUE (p = 0.6) | exact | ✅ |

**Where my prior was wrong, and why it matters.** I reasoned (PREREG §7) that CAMELS discharge depends on
unobserved antecedent storage, so no store over observable inputs could be competitive. That reasoning is
**correct for the DAY family** — which is exactly why P2/P3/P6 all landed inside their intervals and why
the gauged tripwire cleared at 0.256. It is **irrelevant to the DONOR family**, because a concurrent
neighbouring gauge *has already integrated* the unobserved state. I priced the wrong object. The
pre-registration is what makes that visible rather than deniable.

---

## 10. Limitations / confounds
- Fold partition reconstructed, not published (§8.2). Spread across 3 reconstructions: 0.0167.
- Daymet-only; R10 used NLDAS-extended and R9/R14/R16 used maurer. R4 is the forcing-matched primary.
- References span **four incompatible protocols** (10-fold/1995–99, 12-fold/1995–2010, 5-fold/1989–99,
  and Qu's 5-fold/671/2010–14). Every margin row is annotated with its protocol distance. The three rows
  that carry the verdict on their own (R15, R17, R19) are on 1995–2010 or 1989–99 test windows, not ours.
  ⚠ This is a genuine weakness of the *reference set*, inherited, not created here.
- The `concurrent` payload uses donor observations from the test dates. This is legal under A2 (spatial
  holdout) and is operationally realistic, but a venue could redefine PUB to forbid it. The
  `doyclim` payload, which uses only pre-test donor statistics, scores just **0.1041** — so the entire
  effect is carried by concurrent neighbouring observations. **This is the single most decision-relevant
  number for anyone trying to repair the venue.**
- E is a **median**; means are strongly negative for the donor arms (a minority of basins fail badly).
  The venue's own metric is the median, so this is in-convention — but the store is far more
  heavy-tailed than the LSTM references (n(NSE ≤ 0) = 69 vs Li's much smaller counts).

## 11. Recommended next experiments (for the Hub, not run here)
1. ⭐ **Re-run this tripwire on Li 2025's PUR (ungauged *region*) protocol**, 7 rounds, spatially blocked.
   Li publishes PUR at ~0.701 and the Zenodo record ships `py999_PUR_simulations.npy` (same 530 MB shape),
   so the loader control transfers for free. **Prediction to pre-register: the donor arm collapses toward
   the > 100 km bin (−0.157), i.e. PUR is the admissible venue and PUB is not.** This is the cheapest
   route to a defensible CAMELS venue and it reuses everything built here.
2. Add a **distance-blocked** variant of the *existing* PUB folds (hold out all basins within X km of any
   test basin) and sweep X — turns §6's table into a venue-design curve.
3. If a CAMELS claim must be made at all, make it **against R1–R11 only** and state the floor explicitly:
   "a 1.4 MB nearest-gauge lookup scores 0.64 here."
4. Report the Kratzert 12-fold secondary leg (NOT RUN here) if the Hub wants a second protocol.

---

## Git footprint
**None.** Read-only agent: no branch, no commit, no worktree, no tracked-file edit. `chlu/` untouched.
HEAD `7fcef50`, `git status --porcelain` empty at both PREREG filing and report time.
Files written: `.claude/outputs/camels-pub-tripwire.md`, `.claude/outputs/camels-pub-tripwire/**`,
`.claude/scratch/camels-pub-tripwire/**`. No CLU number, no store-of-ours number, no training run.

---

# Proposed handover updates (for the Hub)

**§1.6 (experiments) — new rows**
- ⛔ **CAMELS-US PUB matched-byte tripwire: FIRED. `dies = TRUE`.** Best in-budget, non-leaky,
  531-basin store arm **E = 0.6369** (3 fold-seed mean; best single run 0.6435; spread 0.0167), at
  **1,412,874 B ≤ B = 1,966,080 B**. Arm: donor-basin transfer, key = 10 forcing-climate descriptors,
  payload = the donor's **concurrent** daily specific discharge, **fp16**, all ~478 train-fold donors,
  k = 5, distance-weighted, p_mean rescaling. Fires against **7 of 20** references (R14–R20), incl.
  calibrated **SAC-SMA (0.6447, −0.0078)**, **heudorfer dynamic-only (0.634, +0.0029)** and the
  **NWM (0.5766, +0.0603 — beaten)**. Still loses to the forcing-matched primary R4 (0.742270) by
  −0.1054 and to the frontier R1 by −0.1489.
- ⭐ **Mechanism:** Li-2025 PUB is a *random basin* holdout. Median nearest-train-neighbour distance
  **34.5 km**; **71 %** of held-out basins have a gauged neighbour < 50 km. Median NSE of a 3-neighbour
  concurrent flow copy: **0.822** (< 10 km) → 0.753 (10–25) → 0.663 (25–50) → 0.407 (50–100) →
  **−0.157** (> 100 km). Spearman = −0.542. **The venue is scoring gauge density.**
- **The whole effect is carried by *concurrent* donor observations.** The same arm restricted to
  pre-test donor day-of-year climatology scores **0.1041**. Non-leaky calendar control (regional DOY
  climatology) **0.0036**; regional mean flow **−0.0319**. The laundering control clears the store.
- **The gauged verdict is unchanged and now has a companion:** gauged E = 0.2559 vs S = 0.7580 (cleared);
  PUB E = 0.6369 (fired). The DAY-family arms reproduce the gauged magnitude on PUB (best 0.2904),
  so the difference is entirely the donor-transfer object, not the split.
- ⛔ **The literal "kNN over static catchment attributes" DAY arm is degenerate: −0.0254.** Any future
  tripwire on a spatial-holdout venue must build the donor-basin (one exemplar per basin) form or it
  will hobble the dangerous arm and clear the venue wrongly.

**§5 (provenance) — new / corrected rows**
- ⭐ **Li 2025 Table D2 is reproducible end-to-end from our own loader.** All 12 published rows
  reproduce, worst |Δ| = 0.0024; the three PUB targets to |Δ| ≤ 0.00142; the frontier
  0.785833 → 0.784074. Zenodo 10.5281/zenodo.16895228, published md5s matched.
- ⚠ **Li's ensemble metric convention, pinned:** average members **within a seed** → per-seed median over
  531 → **mean over the 3 seeds**. NSE-of-the-grand-mean over-scores single-model rows by up to +0.037;
  per-run-median-then-mean under-scores ensemble rows by up to −0.093. Anyone re-deriving Table D2 needs this.
- ⚠ **`py999_ids.npy` is NOT in `basin_list_531.txt` order — 439 of 531 positions differ.** Align by id.
- ⚠ **CORRECTION to the frozen reference set (R18, R20):** Qu et al. 2026 is **671 basins, 5-fold,
  train 1980–2004 / test 2010–2014, bootstrap-CDF median** — not "531 (stated)". The *values*
  0.612 / 0.553 are now **primary-confirmed** (arXiv:2601.01558v2, sha256 `115f9e4b…6a44bb`); the
  *protocol metadata* was wrong and must be annotated wherever those rows are quoted.
- **Feng/Li 35-attribute static set built and checksummed:** `attrs35.npz` sha256 `1904cfef…95343`.
  Only `root_depth_50` (13) and `geol_porosity` (3) needed median imputation over 531.
  Two ref-vs-CAMELS spelling fixes: `soil_depth_statgso`→`soil_depth_statsgo`, `geol_porosity`→`geol_porostiy`.
- **maurer guard re-armed and tested:** passes on Daymet (0.99951), fires on maurer (0.0,
  `tmax == tmin` on 100 % of sampled days). 1/30 sampled maurer files has a malformed header (`05120500`).

**§8 (NOT-RUNs / registry) — new entries**
- **NOT RUN:** Kratzert-2019 12-fold PUB secondary leg (optional in PREREG, not reached).
- **Declared limit:** k-means condensation capped at L ≤ 5,000.
- **DAY-family ceiling, measured:** even at **7.57 × the byte budget** the DAY store reaches only 0.2525
  (best over-budget point 0.2885 at 1.89 × B). The DAY family cannot approach E under any budget —
  the donor-basin object, not the store size, is what fires the tripwire.
- **Shuffled-order null:** DAY W30 ordered 0.2634 → shuffled 0.0789 (Δ −0.1845), so the DAY arm is
  genuinely using temporal order, not laundering a bag-of-days.
- ⚠ **Method lesson (bug, caught):** a store tag that embeds a fold-dependent size (`L478_all`) silently
  splits a per-basin accumulator and yields medians over a subset. **Always assert `n_basins == 531`
  on every verdict row.** 1,920/42,240 rows were affected; the apparent top arm would have been
  0.6559 over 54 basins instead of 0.6435 over 531.
- ⚠ **fp16 payloads are the difference between admissible and not** for donor stores here
  (1.41 MB vs 2.81 MB) and are lossless to 4 dp. Byte budgets must state a dtype convention or the
  precision choice decides admissibility.

**§ Escalation (for the Head + Main Advisor)**
The CAMELS-US **PUB** venue cannot carry a bottom-of-board claim. Recommended replacement:
**Li 2025 PUR** (ungauged *region*, 7 rounds, spatially blocked, published at ~0.701, per-basin
simulations already on the same Zenodo record so the loader control transfers for free).
The Hub should scope that as the next spoke; I did not run it, and I did not look for a framing
that rescues PUB.
