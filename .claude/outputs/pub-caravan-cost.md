# pub-caravan-cost — results-analyst report

Task + acceptance criterion: **price the CAMELS-US ungauged (PUB) venue on hardware we can actually
reach today** — data volume, staging effort, loader delta, GPU-hours per cell — with every number
traceable to a command transcript. **Status: done.**
⚠ Slug is legacy: **Caravan is out of scope** (amendment B1). This is CAMELS-US PUB only.

## ⛳ RECONCILIATION LIST (owner needed — protocol §5 corollary, in the first 10 lines)
1. ⛔⛔ **PUB LEAKAGE HAZARD in the inherited harness — this is a code bug for `experiment-engineer`
   and an immediate hazard for the sibling `camels-pub-tripwire` spoke.** `camels_tripwire.py:59–64`
   computes per-basin train-period discharge statistics `qmu/qsd` for **all 531** basins, and
   `:221` un-standardises the prediction for query basin *i* with **that basin's own** `qsd[i], qmu[i]`
   (`camels_local.py:106` identically; `:48,:58` use `qmu[i]` as a fallback prediction). In the gauged
   temporal split that is legal. **In PUB the query basin is ungauged — its discharge mean and s.d. do
   not exist.** Every `target="std"` arm run under a PUB split therefore leaks the held-out basin's
   flow statistics, in the direction that makes the store look **stronger** (i.e. it can cause a
   **false tripwire fire**, which is the expensive failure). Fix = drop the `std` target family under
   PUB, or replace `qmu/qsd` with an attribute-regionalised estimate and declare it.
   ⭐ Forcing z-scoring (`scaled_forcing("perbasin")`) is **not** affected — forcings are available for
   ungauged basins. The defect is discharge-only.
2. ⚠ **Byte-budget convention under k-fold is undefined and must be ruled on.** A 10-fold PUB run
   materialises **10 stores**. Budget per fold (matching the reference, which trains a separate
   same-size LSTM per fold) or 10× the budget in aggregate? I recommend **per fold**; it needs a Hub ruling
   before any margin is quoted.
3. ⚠ Tri-forcing parity with the published frontier (0.785833) is **not reachable with staged data**
   (Maurer defective, NLDAS absent). The forcing-matched reference (0.742270, Daymet) is the only
   honest target. Owner: whichever doc states the target number.

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *none — feasibility / cost recon.* ⛔ Zero CLU numbers, zero store *performance* numbers,
  zero model training. Every performance-shaped number below is a **wall-clock or a byte count**.
- **Laundering control:** n/a (no performance measurement). Every estimate names its hardware.
- **Falsifies:** a measured cost differing from my extrapolation by > 2×. **See §1 — three of my nine
  pre-registered predictions were falsified, all in the "cheaper than predicted" direction; the
  load-bearing ones held.**
- **Does NOT falsify:** the venue being expensive; losing to a reference.
- **Prereg:** filed **before** any measurement at `.claude/outputs/pub-caravan-cost/PREREG.md`
  (HEAD `7fcef50`, clean tree). §0 discloses that the Aug-13 gauged-run logs were read first.

## Flag provenance (protocol §5)
| item | value |
|---|---|
| repo HEAD | `7fcef50`, tracked tree clean, **no branch, no worktree, no tracked-code edit** |
| hardware (all MEASURED rows) | Apple **M1 Pro**, 8 cores (6P+2E), **16 GB** unified RAM, 82→81 GB free, macOS 25.6.0 arm64. **No CUDA GPU.** |
| interpreter | `.claude/scratch/c3-trackb-tripwire/.venv/bin/python` — numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0 (`scratch_venv_freeze.txt`) ⛔ no JAX, no `chlu` import |
| data | `.claude/data/c3-camels/staged/{forcing_daymet,discharge,attributes}.npz`, sha256 **re-verified today, all 3 + source zip MATCH `MANIFEST.json`** (`meas1_checksums.txt`) |
| seeds | store arms `--seeds 0`, `--kmeans 0`; no seed-sensitive quantity is reported (timings only) |
| non-default flags | `--window {30,365} --scaling perbasin --Ls {250,2761,265} --sub {1,3,10} --product daymet` |
| ⚠ contention | machine shared with concurrent spokes; `vm.loadavg` 5.22 at measurement time. **Measured timings are upper bounds.** Cross-check: today's per-L-unit cost is **1.22× the Aug-13 idle-machine value** — consistent with mild contention, not with a regression. |
| Head rulings applied | **C1 (Item 0 CANCELLED — no cluster was probed, resolved, pinged or logged into)**, C3 (no local reference-LSTM benchmark), C4 (GPU-hours not calendar time), C5 (RSS reported), C6 (disk) |

---

## 1. Pre-registration outcome (stated before the results, per §5)

| # | predicted | measured | verdict |
|---|---|---|---|
| P1 | sha256 throughput 1.0 GB/s (0.5–2.0) | **0.271 GB/s** (3.407 GB in 12.57 s) | ⛔ **FALSIFIED**, 3.7× slow — `shasum` is single-threaded, no HW-SHA path. (`md5` on the same box: 0.50 GB/s.) |
| P2a | re-stage wall-clock 240 s (120–600) | **21.89 s** | ⛔ **FALSIFIED**, 11× cheaper than predicted |
| P2b | re-stage peak RSS 1.5 GB (0.8–3.0) | **0.387 GB** | ⛔ **FALSIFIED**, 3.9× lower |
| P2c | re-staged sha256 identical to `MANIFEST.json` | **identical, all 3** | ✅ confirmed — the loader is **bit-reproducible** |
| P3 | arm L=250 wall-clock 10 s (5–25) | **9.48 s** | ✅ |
| P3b | arm peak RSS 1.2 GB (0.6–2.5) | **0.803 GB** | ✅ |
| P4 | arm L=2761 wall-clock 70 s (40–150) | **79.52 s** | ✅ |
| P5 ⭐ | **gauged→PUB extrapolation factor 1.9× (1.5–2.5)** | **1.96×** (same test period) | ✅ — the load-bearing ratio |
| P6 | Zenodo throughput 10 MB/s (2–50) | **10.4 / 12.4 MB/s** | ✅ |

⇒ **The three falsified predictions are all staging/checksum costs, and all in the safe direction.**
The cost model that the deliverable rests on (P3–P6) survived. Reported as a finding, not smoothed away.

---

## 2. Item 0 — compute reachability: **CANCELLED BY HEAD RULING C1. NOT ESTABLISHED.**
⛔ No cluster hostname was resolved, pinged, ssh'd or submitted to by this spoke. The CSF3/CSF4
NXDOMAIN note in the task file is the Head's, is **not carried forward as verified**, and no CSF3 queue
limit, GPU type, walltime cap or credit figure appears anywhere below. **Every GPU row in §5 is a
PROJECTION against named published hardware (NVIDIA Tesla V100), not against any cluster we hold.**

## 3. Data volume & staging effort — MEASURED (item 2)

| item | value | evidence |
|---|---|---|
| CAMELS raw on disk | **7.2 GB** = 3.2 GB forcing/flow zip (3,406,626,583 B) + 3.9 GB `modelOutput_daymet` zip (4,207,763,546 B) | `disk.txt` |
| ⭐ reclaimable | **3.9 GB** — the `modelOutput` zip is **not read by `build_camels.py`** (only `basin_timeseries_v1p2_metForcing_obsFlow.zip` is) | `build_camels.py:29` |
| staged derived set | **93 MB** (87,741,872 + 9,702,885 + 95,770 B) | `disk.txt`, `MANIFEST.json` |
| checksum verification of the whole staged set + source | **13.0 s** (0.44 s npz + 12.57 s zip) — all 4 sha256 **MATCH** | `meas1_checksums.txt` |
| **full re-stage from the zip** (531 basins, Daymet + discharge + 27 attrs, incl. re-checksumming the 3.4 GB zip) | **21.89 s wall, 19.14 s user, peak RSS 386.7 MB**, outputs **bit-identical** | `restage.time`, `restage_MANIFEST.json` |
| re-download cost **if ever lost** | ⚠ **inherited measurement** (Aug-13 `fetch_camels.log`): 4.208 GB in 1,449 s = **2.90 MB/s** from the CAMELS host ⇒ 3.4 GB ≈ **20 min** | `.claude/scratch/c3-trackb-tripwire/fetch_camels.log` |
| **NEW download — Li 2025 Zenodo (B5)** | PUB subset = `py999_PUB_simulations.npy` 530,583,824 B + `py999_OBS.npy` 43,444,424 B + `py999_ids.npy` 4,376 B + `readme.md` 1,126 B = **574.03 MB**; **CC BY 4.0**, DOI 10.5281/zenodo.16895228 | `zenodo_16895228.json` |
| ⭐ already landed — **not re-downloaded** | The sibling `camels-pub-tripwire` spoke fetched it at 17:41 to `.claude/scratch/camels-pub-tripwire/zenodo/`. I **read-verified all 4 md5s against the Zenodo record: all MATCH** (1.15 s). ⛔ Nothing written into that spoke's directories. | terminal transcript, §7 |
| download cost had it not landed | measured Zenodo bandwidth **10.4–12.4 MB/s** (2 × 20 MB ranged GET, HTTP 206) ⇒ 574 MB ≈ **50 s**; the *full* record is 3.71 GB (`ALL`/`PUR` sims we do not need) ≈ 5.4 min | `zenodo_bw.txt` |
| licence | Zenodo bundle **CC BY 4.0** (record metadata). CAMELS-US licence **NOT VERIFIED BY THIS SPOKE** — do not assert it. | — |

**Staging bottom line: the PUB delta is ≈ 1 minute of machine time and +0.57 GB of disk.** Staging is
**not** a cost item for this venue. Disk after everything: 81 GB free.

## 4. Loader delta for the Li-2025 PUB split (item 3 / B4) — one file, ~45 lines

`build_camels.py` (157 lines): **ZERO lines change.** It already reads `basin_list_531.txt`, already
restricts to Daymet, already emits `basins`/`dates` index arrays, and re-stages bit-identically in 22 s.

`camels_tripwire.py` (281 lines) — the whole delta:

| # | change | lines | note |
|---|---|---|---|
| 1 | `TRAIN`/`TEST` → Li periods: train `1980-10-01→1999-09-30` (6,939 d), test `1995-10-01→1999-09-30` (1,461 d) | `:20–21`, 2 lines | ⚠ periods overlap by design — legal because the *basins* are disjoint |
| 2 | fold assignment: `KFold(n_splits=10, shuffle=True, random_state=…)` over the 531 list + a declared seed | ~6 new | ⛔ no reference publishes its fold file (refs §3); our seed must be declared, not matched |
| 3 | `build_regional_stores(...)`: exemplar pool restricted to the fold's **training** basins | `:137–186`, ~5 changed | `valid`/`counts` loop + both `rng.choice(C.nb, …)` calls take a basin subset |
| 4 | `evaluate(...)`: loop over **held-out** basins only, write into a global `(531,)` accumulator | `:196–226`, ~8 changed | |
| 5 | outer driver: 10 rounds → assemble the 531-basin array → median | ~12 new | ⭐ Li's "concatenate test results before metrics" is **equivalent** to per-fold per-basin NSE here, because each basin is held out **exactly once**; concatenation would only matter with repeated holdouts. No extra code. |
| 6 | delete the 447-basin mask and its three metric fields | `:250–252, 266–268`, ~6 deleted | PUB reports on 531 only (refs, reconciliation item 3) |
| 7 | ⛔ **the leakage fix** — `target="std"` family (`:59–64`, `:221`) | ~4 changed / family dropped | **§Reconciliation 1. Not optional.** |
| | **total** | **~45 lines, 1 file, 0 new files** | ≈ half a day of agent time; the risk lives in the positive control, not the code |

**Caravan (B1):** *not a venue, not costed.* `caravan_protocol: NONE` — no split, no metric convention,
no published baselines ⇒ nothing to price. Its 39.3 GB is moot and was **not** downloaded.

## 5. Cost per cell

### 5a. MEASURED — the store/tripwire arms, the thing that actually runs here
`cmd: .venv/bin/python my_tripwire.py --window W --scaling perbasin --Ls L --seeds 0 --kmeans 0 --sub S --product daymet`
(a copy of `camels_tripwire.py` with `OUTD` redirected to my scratch; the original was **not** edited)

| arm | W | D | L | store bytes | test days/basin | **wall** | **peak RSS** |
|---|---|---|---|---|---|---|---|
| A | 30 | 177 | 250 | 178,000 | 3,652 | **9.48 s** | **803.5 MB** |
| B | 30 | 177 | **2,761** (at budget) | 1,965,832 | 3,652 | **79.52 s** | **828.3 MB** |
| C | 365 | 1,852 | **265** (at budget) | 1,964,180 | 3,289 | **18.66 s** | **786.4 MB** |
| D₃ | 30 | 177 | 2,761 | — | 1,218 (`--sub 3`) | **26.04 s** | 788.3 MB |
| D₁₀ | 30 | 177 | 2,761 | — | 366 (`--sub 10`) | **7.34 s** | 787.2 MB |

**Measured cost law (B, D₃, D₁₀):** `t = −0.70 + 0.021963 · n_test_days` s, **R² = 1.00000** ⇒ query cost
is **exactly linear** in evaluated (basin × day) count = **7.95 µs per (L-unit × test-day × 531 basins)**.
This linearity is what licences the k-fold arithmetic below. (Cost is *not* linear in D: arm C runs at
104 GFLOP/s vs arm A's 21 GFLOP/s — larger keys are more BLAS-efficient. Do not extrapolate across W.)

**Whole-sweep numbers (⚠ INHERITED measurement, Aug-13, same machine, `run_regional.log`/`run_phase2.log`):**
the complete gauged store sweep (3 scalings × 3 windows × L-ladder × 3 seeds + k-means; 2,880 arm rows)
= **7,547 s = 2.10 h wall**, decomposed as **801 s store-build + 6,746 s evaluation**.

**PUB projection, from that decomposition + the measured linearity:**
each basin is still queried exactly once across the 10 rounds ⇒ **evaluation total is unchanged**;
the store is rebuilt per fold ⇒ **build ×10**.
- same test period: `(10×801 + 6,746) = 14,756 s = 4.10 h` → **extrapolation factor 1.96×** (P5 ✅)
- at Li's shorter PUB test period (1,461 d vs 3,652 d, ×0.400): `(8,010 + 2,690) = 10,709 s` = **2.97 h**, **1.42×**

⚠ **Robustness of that decomposition.** The 7,547 s wall includes one job the Aug-13 run **killed**
(`Terminated: 15`, the W=1 L≤14,894 arm). Counting only *completed* jobs' own `done in Xs` lines gives
5,511 s + 1,045 s = 6,556 s ⇒ PUB = **2.86 h (1.57×)** instead of 2.97 h (1.42×). **Both readings land in
2.9 ± 0.1 h**, so the conclusion is insensitive to the choice; the *factor* is not (1.42–2.10× across all
four combinations of decomposition × test period). Quote the hours, not the factor.

⇒ ⭐ **The full CAMELS-PUB store cell costs ≈ 3 wall-clock hours on this workstation and needs no GPU.**

**C5 — memory.** Peak RSS never exceeded **0.83 GB** on any arm; staging peaked at 0.39 GB. Against 16 GB,
the largest store cell that fits is `L·D ≲ 2.8×10⁹` (≈ **1.6×10⁷ exemplars** at D=177 — **5,700× the byte
budget**), because the distance buffer is query-chunked to ~160 MB by construction (`predict()`).
⇒ **Memory does not bind for this venue; time does. There is no NOT-RUN-for-RAM row.**

### 5b. `PROJECTION (V100-h, from published timings)` — any future **trained** arm
⛔ Not measured. There is no CUDA GPU on this machine (Head ruling C3); an M1 timing is not a V100 proxy.
Basis, verbatim (Li et al. 2025): *"training LSTM and δHBV … takes approximately **5** and **21** hours …
using a single **NVIDIA Tesla V100** GPU."*

| cell | trainings | **PROJECTION (V100-h)** | period-adjusted ×1.14 |
|---|---|---|---|
| 10 folds × 3 seeds × **1** arm | 30 | **150** | 171 |
| ⭐ **10 folds × 3 seeds × 5 arms** (the task's cell) | **150** | **750** | **855** |
| + ensemble top-up to match the reference's seed count | **+0** | — | Li's PUB row is **3 seeds**; our 3 seeds already match. **No top-up needed.** |
| if instead matched to Kratzert 2019 (12 folds × **10** seeds × 5 arms) | 600 | **3,000** | 3,420 |
| if forced to tri-forcing ensemble parity with the 0.785833 frontier | 450 | **2,250** | ⛔ **NOT REACHABLE** — Maurer defective, NLDAS not staged |
| *reference point:* Li's own published PUB row (120 LSTM + 90 δHBV) | 210 | **2,490** | — |

⚠ The ×1.14 adjustment = 0.9 basins × (19 y / 15 y): Li's 5 h is quoted for the **temporal** test
(all 531 basins, 1980–1995); PUB trains 9/10 basins over 1980–1999.
⚠ **This projection is NOT independent of B3** — it consumes the same published per-run figures, so
agreement with the Hub's 2,490 V100-h sanity check is *by construction*, not corroboration. The only
independent handle would be a measured CUDA run, which C3 forbids. **Confidence: moderate-low
(±2× is credible).** Kratzert 2019's wall-clock and hardware remain **NOT-FOUND**.
⚠ C4: no calendar time is quoted. Queue latency is the Head's variable, not mine.

## 6. ⭐ THE COST TABLE (item 6)

| option | download | staging (machine time) | new disk | loader delta | **cell cost — MEASURED (this M1 Pro)** | **cell cost — PROJECTION (V100-h)** | wall-clock at real queue limits | what it displaces |
|---|---|---|---|---|---|---|---|---|
| **CAMELS-US PUB — store/tripwire arms** ⭐ | **0 B** (staged Aug-13, sha256 re-verified today) + 574 MB Zenodo **already landed** by the sibling spoke | **21.9 s** re-stage + **13.0 s** checksums ≈ **35 s** | **+0.57 GB** (81 GB free) | **~45 lines in 1 file**; `build_camels.py` **unchanged**; ≈ ½ day agent time | **≈ 2.9 h wall, ≤ 0.83 GB RSS, 0 GPU** (1.4–2.1× the gauged sweep) | **0** | **NOT APPLICABLE — runs locally** | ~3 h of the single shared workstation |
| **CAMELS-US PUB — trained LSTM arms** | same | same | same | same + a training stack (**not costed here**) | ⛔ **NOT RUN — no CUDA GPU on this hardware.** Never a zero. | **750** (855 period-adj) for 10×3×5 | ⛔ **NOT ESTABLISHED** (ruling C1: no cluster was probed) | the Head's CSF3 allocation, whose limits are unverified |
| **Caravan** | ⛔ **not a venue** (B1) — no split, no metric, no baselines | — | — | — | **not costed** | **not costed** | — | — |

## 7. How I verified (commands, all transcripts under `.claude/outputs/pub-caravan-cost/`)
```
shasum -a 256 staged/*.npz ; shasum -a 256 basin_timeseries_v1p2_metForcing_obsFlow.zip   # meas1_checksums.txt
/usr/bin/time -l .venv/bin/python restage_camels.py                                        # restage.{log,time}
/usr/bin/time -l .venv/bin/python my_tripwire.py --window 30 --scaling perbasin \
     --Ls {250|2761} --seeds 0 --kmeans 0 --sub {1|3|10} --tag costX --out <my scratch>     # arm_timings/
/usr/bin/time -l ... --window 365 --Ls 265 ...                                             # arm_timings/costC.*
curl -s https://zenodo.org/api/records/16895228                                            # zenodo_16895228.json
curl -sL -r 0-20971519 <zenodo file> -o /dev/null -w '%{speed_download}'  (×2)              # zenodo_bw.txt
md5 py999_PUB_simulations.npy py999_OBS.npy py999_ids.npy readme.md   # vs record: 4/4 MATCH (read-only)
```
Scratch: `.claude/scratch/pub-caravan-cost/` (`restage_camels.py`, `my_tripwire.py` — **copies** with
output paths redirected; the originals in `.claude/outputs/c3-trackb-tripwire/code/` were not touched).

## Git footprint
**None.** No branch, no worktree, no commit, no tracked-file edit. `chlu/` untouched, HEAD `7fcef50`.

## Open questions / follow-ups / risks
1. ⛔ **The `std`-target leakage (Reconciliation 1) must reach the sibling spoke before it quotes a PUB
   margin.** It is the single failure mode that could manufacture a tripwire fire.
2. ⚠ The byte budget under k-fold (per-fold vs aggregate) needs a Hub ruling (Reconciliation 2).
3. ⚠ My V100 projection shares its inputs with B3 ⇒ **no independent check exists**. If ±2× matters to
   the decision, the only cure is one calibrated CUDA run — which requires cluster access that ruling C1
   puts out of scope. State the ±2× wherever the 750 V100-h number is quoted.
4. ⚠ Timings taken on a **shared** box (loadavg 5.22); they are upper bounds, ~1.22× the Aug-13 idle rate.
5. The 3.9 GB `modelOutput_daymet.zip` is dead weight — deletable if disk ever binds (it does not today).
6. Not costed by design: the CLU arm, any training stack, and anything on a cluster.

---

## Proposed handover updates (for the Hub)

**For §1.6 / the C3 venue ledger — CAMELS-US PUB feasibility (measured 2026-09-07, M1 Pro, HEAD `7fcef50`):**
- Staging delta ≈ **35 s** of machine time and **+0.57 GB** disk. CAMELS raw (7.2 GB) and the derived set
  (93 MB) are on disk and **sha256-re-verified today, 4/4 match**. The loader is **bit-reproducible**:
  a full re-stage takes **21.89 s / 386.7 MB RSS** and reproduces all three npz hashes exactly.
- Li 2025's Zenodo PUB bundle = **574.03 MB, CC BY 4.0**, landed by `camels-pub-tripwire`, **md5 4/4
  verified against the Zenodo record** by this spoke. Measured Zenodo bandwidth **10.4–12.4 MB/s**.
- **Loader delta = ~45 lines in one file** (`camels_tripwire.py`); `build_camels.py` needs **zero** changes.
- ⭐ **The full PUB store cell costs ≈ 2.9 h wall-clock (2.86–2.97 h, two decompositions) and ≤ 0.83 GB
  RSS on the workstation, with no GPU** — 1.4–1.6× the gauged sweep at Li's shorter test period, **1.96×** at
  matched test period (pre-registered 1.9×, measured 1.96×).
  Query cost is exactly linear in evaluated basin-days (**R² = 1.00000**, 7.95 µs per L-unit·test-day·531).
- ⭐ **`PROJECTION (V100-h, from published timings)`: a trained-arm cell of 10 folds × 3 seeds × 5 arms =
  150 trainings = 750 V100-h (855 period-adjusted).** No ensemble top-up is needed — Li's PUB reference is
  itself 3 seeds. Tri-forcing parity with the 0.785833 frontier would be 2,250 V100-h and is
  **NOT REACHABLE** with staged data. ⛔ Never quote this as a measurement; ±2× confidence.
- ⛔ **Cluster: NOT ESTABLISHED.** Item 0 was cancelled by Head ruling C1; no cluster was contacted and no
  CSF3/CSF4 limit is carried forward. All GPU rows are against a published V100, not against an allocation.
- **Caravan: not a venue, not costed** (`caravan_protocol: NONE`).

**For §5 (provenance):** add `.claude/outputs/pub-caravan-cost/` — `PREREG.md`, `env.txt`,
`meas1_checksums.txt`, `restage.{log,time}`, `restage_MANIFEST.json`, `arm_timings/`, `costmodel.txt`,
`zenodo_bw.txt`, `zenodo_16895228.json`, `disk.txt`.

**For §8 / risk register — flag to `experiment-engineer` (code bug):**
`camels_tripwire.py:59–64,221` and `camels_local.py:48,58,101,106` un-standardise predictions with the
**query basin's own** train-period discharge mean/s.d. Legal under a temporal split; **leakage under PUB**,
biased toward making the store look stronger. Fix before any PUB `target="std"` number is quoted.
