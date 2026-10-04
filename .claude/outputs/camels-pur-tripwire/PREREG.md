# PREREG — camels-pur-tripwire
**Filed 2026-09-09, BEFORE any tripwire arm was run.** Repo HEAD `7fcef50`, `git status --porcelain` empty.
Protocol §5 pre-registration rule. Predictions below are point values with 80 % intervals; a failed
prediction is a finding, not an embarrassment.

## §0 Disclosure — what I already knew when I wrote this
Full disclosure is mandatory; concealing prior knowledge would make the prereg worthless.

1. **The sibling PUB run fired.** `camels-pub-tripwire` measured E = 0.6369 (3 fold-seed mean; best
   single run 0.6435) for a donor-basin store at 1,412,874 B, fired against 7/20 PUB references,
   `dies = TRUE`.
2. **PUB's distance–skill curve, measured there** (arm `latlon` k=3, concurrent payload, fp16, fs0):
   < 10 km 0.8215 (n=47) · 10–25 km 0.7526 (123) · 25–50 km 0.6632 (207) · 50–100 km 0.4068 (116) ·
   **> 100 km −0.1574 (38)**. Spearman(NN distance, NSE) = −0.542.
3. **PUB companion rows:** spatial-NN donor k=3 0.6271, k=1 0.5860; regional DOY climatology 0.0036;
   regional mean flow −0.0319; LEAKY per-basin DOY climatology 0.1421; LEAKY persistence 0.4362;
   best DAY arm 0.2904 (W1 forcing key); DAY statics-only −0.0254; best leaky/oracle 0.6759.
4. **The PUR region assignment is already pinned** (see §2) and **its realised geography is already
   measured** (`GEO_partitions.json`, run before this prereg because it is an instrument check, not a
   claim): pinned 7-region HUC partition → nearest-*train*-neighbour distance **median 265.3 km**,
   p10 63.7 km, min 14.99 km, **0** basins < 10 km, 1.1 % < 25 km, 7.7 % < 50 km, 17.5 % < 100 km,
   **82.5 % > 100 km**. PUB, for contrast: median 34.5 km, 8.9 % < 10 km, 71 % < 50 km, 7.2 % > 100 km.
   ⚠ This is disclosed *because it makes the predictions below much easier*: I am predicting a
   collapse from an already-measured geography, not from a guess.
5. **PUR references are weaker than PUB's** — the whole point of the task. Frontier 0.700814 vs
   0.785833; forcing-matched primary 0.644398 vs 0.742270; Daymet LSTM 0.578365 vs 0.702636.
6. I have **not** loaded `py999_PUR_simulations.npy` into an NSE routine, have **not** run any store
   arm on PUR, and have **not** computed any per-region skill number.

## §1 Protocol (verdict-binding)
- **Li et al. 2025 PUR** (HESS 29, 6829–6863, 2025), 531 basins, **7 rounds**, spatially blocked.
- Li Table 2 (publisher xlsx `hess-29-6829-2025-t02.xlsx`, sha256
  `03053aedc2ebb2dce956ed3051c4b3e6d39fbe6359d6fd855d1e23ac80d756da`), verbatim row:
  `PUR | Random six of seven regions | 1980–1999 | Holdout | 1995–1999`.
  ⇒ **store window 1980-10-01 → 1999-09-30, test window 1995-10-01 → 1999-09-30 (1,461 d)** — identical
  to PUB, so the PUB harness's windows carry over unchanged.
- Forcing **Daymet only** (maurer is defective; guard re-armed).
- Metric: NSE **unclipped**, `obs < 0` dropped first (Kratzert `calc_nse`), per basin, **median over 531**.
- Ensemble convention to be re-identified, not tuned: PUB pinned "convention C" = average members
  **within a seed** → per-seed median over 531 → **mean over the 3 seeds**.

## §2 ⭐ The region assignment — PINNED FROM THE AUTHORS' OWN CODE, not reconstructed
Li 2025 §2.2: *"all basins were divided into 10 spatially stratified groups for the PUB test and
**7 fully disjoint regional groups for the PUR test** (Table 2) **in the same way as Feng et al. (2023b)**"*.
Feng et al. 2023 §2.4: *"we divided the whole CONUS into **seven continuous regions** (as shown in
Fig. B1 in Appendix B)"* — Fig. B1 is a **map image**, i.e. not machine-readable.

**The rule is published in the authors' released code.** `mhpi/hydroDL` (Fang, Shen & Feng 2021,
Zenodo 10.5281/zenodo.5015120 — the LSTM code Feng 2023 and Li 2025 both cite),
`example/PUR/trainPUR-Reg.py` lines 54–62, **verbatim**:
```python
# Divide CAMELS dataset into 7 PUR regions
regionDivide = [[1, 2], [3, 6], [4, 5, 7], [9, 10],
                [8, 11, 12, 13], [14, 15, 16, 18], [17]]  # seven regions
```
with membership by the CAMELS gauge **HUC-2** code (`camels.readGageInfo` → `gauge_information.txt`
field `huc`). Both local HUC sources (`camels_name.txt::huc_02` and the zip's
`gauge_information.txt::huc`) agree on **531/531** basins.

**Independent corroboration against Feng 2023's prose** (all seven descriptions match):
region 1 = HUC 1,2 = "northeast" ✓ · region 7 = HUC 17 = "northwest" ✓ ·
region 5 = HUC 8,11,12,13 = "the Southern Great Plains, with frequent flash floods and karst geology" ✓ ·
regions 4,5 = HUC 9,10 / 8,11,12,13 = "arid basins … in the middle CONUS" ✓ ·
regions 2,4,5 = "homogeneous forest or prairie on the Great Plains" (HUC 3,6 = SE forest) ✓ ·
regions 3,6,7 = "large diversity in the landscape" (Great Lakes/Ohio/UpperMiss; Colorado/GreatBasin/California; PNW) ✓.

⇒ **Primary partition = the pinned one.** Realised sizes: **92, 95, 93, 51, 68, 60, 72** (= 531).
Because it is pinned rather than reconstructed, the task's "≥ 3 partitions" clause is discharged as a
**sensitivity analysis** instead of an uncertainty budget: 3 independent lat/lon k-means 7-way blocks
(seeds 0,1,2) are run alongside and reported. The verdict is quoted on the pinned partition.

## §3 Positive control (must pass BEFORE any tripwire number is quoted)
Our discharge, our alignment, our NSE, against Li's published PUR simulations
(`py999_PUR_simulations.npy`, 530,583,824 B, md5 `1e908d47d25522f883300be159713b72`).
Targets (Li Table D3, publisher `hess-29-6829-2025-t07.xlsx`, sha256 `1b592ed0…edcfb939`, re-parsed here):

| target | published |
|---|---|
| LSTM + δHBV, Daymet | 0.644398 |
| δHBV, Daymet | 0.622278 |
| LSTM, Daymet | 0.578365 |
| LSTM + δHBV, All (frontier) | 0.700814 |

**Pre-registered tolerance: |Δ median NSE| ≤ 0.010 on all four targets.** Miss it ⇒ loader is wrong,
stop and report. Axis-0 ordering is **re-verified**, not assumed.

## §4 Arms (all reuse the PUB grid; ⛔ no CLU, no training)
- **Family DONOR** (the dangerous arm, mandated by the task): exemplar = one **train-region** basin;
  keys ∈ {S35, S35oh, S35+clim, S35+latlon, clim, latlon}; payload ∈ {concurrent 1,461-day specific
  discharge, 366-day DOY climatology}; k ∈ {1,3,5,10,25}; weight ∈ {dist, unif};
  normaliser ∈ {raw, pmean, qmean} admissible, **oracle quarantined as a leaky upper bound**;
  payload dtype ∈ {fp32, fp16}; L-ladder {5,10,25,50,100,200,328,all} × {rand, k-means} × sel-seeds {0,1,2}.
- **Family DAY**: statics-only / W1 / W30 / W365 forcing windows ⊕ statics, scaling ∈ {raw, global, perbasin}.
- **Companions**: day-of-year climatology (train-region, non-leaky) · spatial-nearest-neighbour donor ·
  regional mean flow · leaky rows labelled · **shuffled-order null**.
- **Budget: B = 1,966,080 B PER ROUND** (Head ruling 2026-09-08 — the reference trains a separate
  same-size model per round). Secondary ladder point B_param = 1,184,772 B.
- **≥ 690 leak assertions** expected (one per fold per arm-family pass); every one must pass.

## §5 KILL RULE (verbatim from the task)
> The venue DIES if the best in-budget store arm comes within **0.02 absolute or 2 % relative** of the
> best strong reference. Evaluated and reported against **EVERY** PUR reference.

**Frozen PUR reference set** (assembled here; Li Table D3 = all 21 model × forcing cells, Li Table D4
PUR rows, Feng 2023 PUR text values). Lowest rows, which are what the rule actually plays against:
`li2025_LSTM_nldas = 0.546217` · `feng2023_LSTM_daymet_PUR = 0.55` · `feng2023_LSTM_maurer_PUR = 0.55` ·
`feng2023_δHBV_maurer_PUR = 0.56`. ⇒ **the store must stay below ≈ 0.526 to clear.**
Forcing-matched primary = `li2025_LSTM+δHBV_daymet = 0.644398`; frontier = `li2025_LSTM+δHBV_All = 0.700814`.

## §6 ⭐ The pre-registered predictions
**The question is not "does the store get worse" — it is "does the store drop FASTER than the
references drop?"** Reference PUB→PUR retention: LSTM Daymet 0.578365/0.702636 = **0.823**;
LSTM+δHBV Daymet 0.644398/0.742270 = **0.868**; frontier 0.700814/0.785833 = **0.892**.

**Derivation of my point prediction for E.** Take PUB's measured distance–skill curve (§0.2) and
re-weight it by PUR's realised nearest-train-neighbour histogram (§0.4: 0 / 6 / 35 / 52 / 438 basins in
the < 10 / 10–25 / 25–50 / 50–100 / > 100 km bins). 82.5 % of PUR's held-out basins sit in the bin whose
PUB median NSE is **−0.157**, so a naive transfer puts the PUR median only a little above that bin's
median. Two corrections push it up: (i) PUR's `clim` key can find a *climatically* similar donor even at
600 km, where a pure `latlon` key cannot, and (ii) averaging k ≥ 5 distant donors smooths daily event
noise toward a seasonal-climatology predictor, whose PUB value is +0.10, not −0.16. Two corrections push
it down: PUR's distances are far larger than PUB's > 100 km bin (median 265 km, p90 684 km), and PUB's
> 100 km basins are geographically odd, not typical. Net: I put E just above zero.

| # | prediction | point | 80 % interval |
|---|---|---|---|
| P1 | loader control passes on all four PUR targets, worst \|Δ\| | PASS, 0.003 | [0.000, 0.010] |
| P1b | axis-0 order = PUB's (LSTM d/n/m, δHBV d/n/m) | TRUE | p = 0.85 |
| P1c | convention C is again the identified convention | TRUE | p = 0.80 |
| P2 | DAY family best in-budget arm | 0.10 | [−0.05, 0.25] |
| P3 | DAY statics-only at budget (degenerate) | −0.03 | [−0.15, 0.08] |
| P4 | DONOR concurrent, best key, best in budget | 0.12 | [−0.05, 0.35] |
| P5 | DONOR doyclim, best in budget | 0.09 | [−0.05, 0.22] |
| P6 | winning DONOR payload is `concurrent` (not `doyclim`) | TRUE | p = 0.55 |
| P7 | spatial-NN donor k = 3 companion | −0.05 | [−0.40, 0.20] |
| P8 | train-region DOY climatology (non-leaky) | −0.02 | [−0.30, 0.12] |
| P9 | train-region mean flow (non-leaky) | −0.10 | [−0.50, 0.03] |
| P10 | LEAKY per-basin DOY climatology (split-independent ⇒ must reproduce PUB) | 0.1421 | [0.1411, 0.1431] |
| P11 | ⭐ **E** (best in-budget, non-leaky, n = 531, pinned partition) | **0.13** | [−0.05, 0.36] |
| P12 | margin_abs vs primary (0.644398) | −0.51 | [−0.69, −0.28] |
| P13 | fires vs the lowest reference (0.546217) | FALSE | p = 0.88 |
| P14 | ⭐ **`dies`** | **FALSE** | p = 0.85 |
| P15 | ⭐ **retention ratio E_PUR / E_PUB (= E / 0.6369)** | **0.20** | [−0.08, 0.57] |
| P16 | ⭐ store retention < LSTM-Daymet reference retention (0.823) | TRUE | p = 0.92 |
| P17 | max E over the 3 k-means sensitivity partitions − E_pinned | +0.01 | [−0.10, +0.10] |
| P18 | no k-means sensitivity partition fires either | TRUE | p = 0.85 |
| P19 | best leaky/oracle arm − best legitimate arm | +0.05 | [0.00, 0.20] |
| P20 | shuffled-order null Δ on the DAY family | −0.06 | [−0.20, +0.02] |
| P21 | peak RSS | < 8 GB | p = 0.95 |
| P22 | region-check: regions 1 (NE) and 7 (NW) are the two best of the seven by median NSE of Li's own published PUR (LSTM+δHBV, Daymet), as Feng 2023 asserts | TRUE | p = 0.55 |
| P23 | leak assertions executed ≥ 400, zero fired | TRUE | p = 0.95 |

## §7 What would make me wrong, stated in advance
- If the pinned partition is not Li's, every margin is against a different task. Mitigation: §2's
  seven-way prose corroboration plus P22, a *data-side* test of the assignment.
- If E lands above ≈ 0.526 the venue fires and PUR dies with PUB. I would then report `dies = TRUE`
  and escalate without looking for a framing that saves it.
- If the 3 sensitivity partitions bracket the kill threshold, the verdict is **INCONCLUSIVE** and will
  be declared as such.
