# camels-caravan-feasibility — Hub consolidation

**For: the Head and the Main Advisor. ⛔ ESCALATION — the pre-registered tripwire FIRED.**
Consolidates `camels-caravan-refs.md` (+ Hub verification addendum §8), `camels-pub-tripwire.md`,
`pub-caravan-cost.md`. Hub review: all three accepted. Date 2026-09-08. Repo HEAD `7fcef50`, tree clean;
no branch, no worktree, no tracked-code edit was made by any of the three spokes.

---

## 1. RECOMMENDATION: **NEITHER.** Do not fund a loader or model work for either venue.

- **Caravan is not a venue.** It has no benchmark protocol at all — no split, no metric convention, no
  published baselines, not even in its own dataset paper, whose technical-validation section contains no
  model. It cannot carry a SOTA claim. Confirmed by three independent routes. Nothing was staged; its
  39.3 GB was not downloaded.
- **CAMELS-US PUB dies on its own tripwire.** A **1.41 MB, untrained, nearest-neighbour donor-basin store**
  reaches **median NSE 0.6369** on the ungauged split. It **fires against 7 of the 20 references** — it
  beats the National Water Model and the AlphaEarth foundation-model baseline outright, and effectively
  ties calibrated SAC-SMA. Pre-registered kill rule, evaluated against every reference: **`dies = true`.**

⚠ **This falsifies the VENUE, not the CLU.** No CLU number was computed anywhere in this work.

**There is a repair, and it is cheap** — see §6. It is the only reason this is not simply a dead end.

---

## 2. The tripwire, and why it fired

**E = 0.6369** — best in-budget, non-leaky, full-531-basin arm; mean over 3 fold-seed reconstructions
(per-seed 0.6405 / 0.6268 / 0.6435, spread 0.0167). Config: donor-basin store keyed on 10 forcing-climate
descriptors, payload = each donor's concurrent daily specific discharge, k=5, distance-weighted,
p_mean rescaling, fp16, 1,412,874 B against a 1,966,080 B budget. 34,492 admissible arms.

**Mechanism, measured — the venue is scoring gauge density, not hydrology.** Li-2025 PUB is a *random*
basin holdout, so a "prediction in ungauged basins" task mostly hands you a gauged neighbour:

| nearest gauged train neighbour | basins | median NSE of a 3-neighbour flow copy |
|---|---|---|
| < 10 km | 47 | **0.822** |
| 10–25 km | 123 | 0.753 |
| 25–50 km | 207 | 0.663 |
| 50–100 km | 116 | 0.407 |
| > 100 km | 38 | **−0.157** |

Median nearest-neighbour distance **34.5 km**; **71%** of held-out basins have a gauged neighbour inside
50 km. Spearman(distance, NSE) = **−0.542**. The entire effect is carried by *concurrent* donor
observations: the same arm restricted to pre-test donor day-of-year climatology scores **0.1041**.

## 3. Margins against every reference (in-budget, non-leaky, n=531)

Frontier: **median NSE 0.785833** — LSTM+δHBV, tri-forcing ensemble, Li et al. (2025) HESS 29:6829–6861,
doi:10.5194/hess-29-6829-2025, Table D2. ⚠ Not the 0.69 the program was carrying; that is seven years stale.
Forcing-matched primary (Daymet, our staged data): **0.742270**.

| # | reference | S | margin (abs / rel) | fires |
|---|---|---|---|---|
| R1 | li2025 LSTM+δHBV tri-forcing ⭐FRONTIER | 0.785833 | −0.1489 / −18.9% | no |
| R2 | li2025 LSTM tri-forcing | 0.768181 | −0.1313 / −17.1% | no |
| R3 | li2025 δHBV tri-forcing | 0.757749 | −0.1208 / −15.9% | no |
| R4 | li2025 LSTM+δHBV Daymet ⭐PRIMARY | 0.742270 | −0.1054 / −14.2% | no |
| R5 | heudorfer2026 EMB-LSTM AlphaEarth+ | 0.721 | −0.0841 / −11.7% | no |
| R6 | li2025 δHBV Daymet | 0.706809 | −0.0699 / −9.9% | no |
| R7 | li2025 LSTM Daymet | 0.702636 | −0.0657 / −9.4% | no |
| R8 | heudorfer2026 baseLSTM CAMELS-27 | 0.696 | −0.0591 / −8.5% | no |
| R9 | li2025 LSTM Maurer | 0.694156 | −0.0572 / −8.2% | no |
| R10 | kratzert2019 PUB LSTM ensemble | 0.6931 | −0.0562 / −8.1% | no |
| R11 | feng2023 δHBV Daymet ⚠prov | 0.69 | −0.0531 / −7.7% | no |
| R12 | li2025 δHBV NLDAS | 0.670636 | −0.0337 / −5.0% | no |
| R13 | feng2023 LSTM Daymet ⚠prov | 0.66 | −0.0231 / −3.5% | no |
| **R14** | feng2023 LSTM Maurer | 0.65 | −0.0131 / −2.0% | ⛔ **FIRES** |
| **R15** | kratzert2019 **SAC-SMA calibrated** | 0.6447 | −0.0078 / −1.2% | ⛔ **FIRES** |
| **R16** | feng2023 δHBV Maurer | 0.64 | −0.0031 / −0.5% | ⛔ **FIRES** |
| **R17** | heudorfer2026 baseLSTM dynamic-only | 0.634 | **+0.0029** | ⛔ **FIRES (beaten)** |
| **R18** | qu2026 **AlphaEarth embeddings** | 0.612 | **+0.0249** | ⛔ **FIRES (beaten)** |
| **R19** | kratzert2019 **NWM retrospective** | 0.5766 | **+0.0603** | ⛔ **FIRES (beaten)** |
| **R20** | qu2026 CAMELS attributes | 0.553 | **+0.0839** | ⛔ **FIRES (beaten)** |

At the stricter parameter-matched budget (1,184,772 B) E = 0.6224 — still fires against R17–R20.
Struck from the set before the run: `feng2023_MPR+mHM = 0.53`, which is an **in-sample/temporal-test**
number, not PUB. Had it stayed it would have armed the kill rule from the wrong task.

## 4. Controls — all passed, and the verdict is not an artefact

- **Loader positive control:** our discharge, our date alignment, our basin ordering, our NSE code against
  Li's published per-basin simulations. Pre-registered tolerance ≤ 0.010; **worst observed 0.00142.** All
  12 published Table D2 rows reproduce (worst |Δ| 0.0024). Zenodo md5s matched exactly.
- **Maurer guard:** passes on Daymet (99.95% tmax>tmin) **and fires on Maurer** (tmax==tmin on 100% of
  sampled days, assertion raised). The guard is itself tested.
- **Leak assertions:** ≥ 690 executed across all families, **zero fired**. Held-out basins never entered
  an exemplar set.
- **Laundering control clears the store:** non-leaky regional day-of-year climatology 0.0036, regional
  mean flow −0.0319. Shuffled-order null on the window family: 0.2634 → 0.0789.
- **The leaky upper bound buys almost nothing:** oracle rescaling (using the query basin's own flow
  statistics — excluded) reaches only 0.6759, +0.032 over the legitimate arm. The signal is the
  concurrent-donor observation, not the normaliser.
- ⭐ **Both spokes independently caught the same hazard.** The cost spoke found that the *inherited gauged*
  harness un-standardises with the query basin's own discharge statistics — legal gauged, a leak under PUB,
  and biased toward a **false fire**. The tripwire spoke never used that code: it wrote fresh paths and
  classified exactly that operation as `oracle`/leaky. **The deciding arm is clean.** The legacy bug is
  real and is logged for anyone reusing that file.

## 5. Cost — moot for the recommendation, recorded because it was asked for

Measured on the workstation (M1 Pro, 8 cores, 16 GB, no CUDA); local-only per the Head's ruling. No cluster
was probed.

| option | download | staging | new disk | loader delta | cell cost MEASURED | cell cost PROJECTION (V100-h) |
|---|---|---|---|---|---|---|
| CAMELS-PUB, store arms | 0 B (staged) + 574 MB Zenodo, landed | **35 s** | +0.57 GB | **~45 lines, 1 file**; `build_camels.py` unchanged | **≈ 2.9 h, ≤ 0.83 GB RSS, 0 GPU** | 0 |
| CAMELS-PUB, trained LSTM arms | same | same | same | + a training stack, not costed | ⛔ **NOT RUN — no CUDA GPU here** | **750** (855 period-adjusted), 10 folds × 3 seeds × 5 arms |
| Caravan | ⛔ not a venue | — | — | — | not costed | not costed |

Peak RSS across every tripwire arm was 1.33 GB of 16 GB; **nothing was skipped for memory.** Cluster
wall-clock is **NOT ESTABLISHED** — no cluster was contacted, by ruling.
**No reference re-train is needed at all:** both the frontier and the 2019 baseline ship per-basin
artefacts, so they are cited by recomputation. That removes ~2,490 V100-hours from the original estimate.

## 6. The repair, if a CAMELS venue is still wanted

**Li 2025's PUR — prediction in ungauged *regions*, spatially blocked, 7 rounds.** Li publishes it at
**~0.701**, and the same Zenodo record ships `py999_PUR_simulations.npy` at the same shape, so **the
loader positive control transfers for free.** Everything built here is reused.

**Pre-register this before running it:** the donor arm collapses toward the >100 km bin (−0.157), because
spatial blocking is precisely what removes the gauged neighbour that PUB leaves in. If that holds, **PUR is
admissible where PUB is not**, and the tripwire is what proved it. If the donor arm survives PUR too, the
whole CAMELS family is out and we stop.

⚠ **Do not repair PUB by forbidding concurrent donor observations.** Using a gauged neighbour 5 km away is
operationally realistic; a venue that bans it is measuring a handicap, not hydrology.

## 7. What would have to be true for "NEITHER" to be wrong

1. **The fold reconstruction is unrepresentative.** No reference publishes its fold file; ours is
   `KFold(shuffle=True, random_state ∈ {0,1,2})`. Spread across the three was 0.0167 — small, but the
   published partition is unavailable, so every margin is against a faithful reconstruction, not the
   real thing.
2. **The three references that carry the verdict alone (R15, R17, R19) sit on different test windows**
   — 1995–2010 or 1989–1999, not our 1995–1999. The reference set spans four incompatible protocols. This
   is an inherited weakness of the literature, not of this measurement, but it means the tightest margins
   are cross-protocol.
3. **R18/R20 (Qu 2026) are protocol-mismatched** — 671 basins, 5-fold, test 2010–2014. Numerically
   confirmed from the primary, but they are not on our split. Drop them and the verdict still stands on
   R14–R17 and R19.
4. **The venue reforms itself.** If the community adopts spatially blocked folds as standard PUB, the
   donor arm dies and the venue becomes admissible. Nothing here says CAMELS is permanently unusable — it
   says *this split* is.
5. **The Feng 2023 Daymet rows (R11, R13) are still provisional** — single HTML source, table image
   unparsed. Neither fires, so they do not carry the verdict.
6. **A stronger reference exists behind a paywall.** Yu et al. 2024 WRR and Li et al. 2022 WRR were
   identified as likely PUB papers and could not be retrieved (Wiley 403). **NOT-FOUND, not absent.** A
   higher frontier would not save the venue — the fire is at the bottom of the board, not the top.

## 8. Open items with owners

| item | owner |
|---|---|
| Legacy leak in `camels_tripwire.py:59–64,:221` (`std` target uses query-basin flow stats) | experiment-engineer, before any reuse |
| Byte-budget convention under k-fold: per fold or aggregate across 10 stores? Spoke recommends **per fold** | Head ruling needed before any future margin is quoted |
| `qu2026_*` protocol metadata was wrong in the frozen set (671/5-fold/2010–14, not 531/PUB) | corrected here; propagate |
| Tri-forcing parity (0.785833) unreachable with staged data — Maurer defective, NLDAS absent | any doc stating the target; use 0.742270 |
| PUB frontier is 0.7858, not 0.69 | any doc carrying the stale number |

---

# ⭐ AMENDMENT — 2026-09-09: the PUR tripwire ran and CLEARED. Recommendation changes to **CAMELS-US PUR**.

§1's "NEITHER" stood on PUB and Caravan alone; the repair in §6 was untested when it was written. It has
now been tested. **`dies = FALSE`, 0 of 32 references fired.**

| | PUB | **PUR** |
|---|---|---|
| best in-budget non-leaky store arm **E** | **0.6369** | **0.2743** |
| tightest margin to any reference | **−0.0031** (3 refs beaten outright) | **−0.2719 / −49.8%** |
| kill threshold | — | 0.5262; store reached 0.2743 |
| verdict | ⛔ **DIES** | ✅ **ADMISSIBLE** |

E = 0.2743 at 1,184,760 B — inside **both** the per-round budget (1,966,080 B, Head ruling) **and** the
tighter parameter-matched budget (1,184,772 B). Weakest reference `li2025_LSTM_nldas` 0.546217; primary
(Daymet) 0.644398; frontier 0.700814.

**The task's biggest risk was removed, not managed.** The region rule is **published in the authors' own
released code** (`mhpi/hydroDL`, `example/PUR/trainPUR-Reg.py` L54–62): `regionDivide =
[[1,2],[3,6],[4,5,7],[9,10],[8,11,12,13],[14,15,16,18],[17]]` over CAMELS HUC-2. It reproduces all seven
of Feng 2023's verbal region descriptions and is confirmed data-side against Li's own published PUR
simulations. **The partition is pinned, not reconstructed** — so the non-exchangeability worry does not
apply. Three alternative k-means partitions give 0.266–0.268 against the pinned 0.274; spread ≈0.008,
two orders below the tightest margin.

**Blocking verified to have done its job:** median nearest *train* neighbour **265.3 km** (PUB: 34.5 km),
**0** basins within 10 km (PUB: 47), **82.5%** beyond 100 km (PUB: 7.2%).

**The pre-registered question — does the store drop faster than the references? — answered YES, with a
twist worth carrying.** The donor-basin arm that killed PUB collapses 0.6435 → **0.1062** (retention
0.165) while the references retain 0.82–0.89. But **E is no longer that arm.** The PUR winner is the DAY
forcing-key arm, which is not distance-mediated and barely moves (0.2904 → 0.2743, retention 0.944).
Spearman(neighbour distance, NSE) = **−0.694** for the donor arm vs **−0.154** for the DAY arm.
⇒ **PUR's classical floor is a rainfall–runoff lookup, not a nearest-gauge lookup, and it sits 0.26 NSE
below the bar.** That is why the venue is admissible.

**Loader control passed on all 12 published PUR rows** under the convention pinned by the PUB run
(average members within a seed → per-seed median over 531 → mean over 3 seeds); worst |Δ| ≈ 0.0022,
frontier reproduced to 0.00064. ⚠ The single-run convention misses the fused rows by up to 0.119 — the
member-averaging convention is load-bearing and must be carried into any future PUR number.

## ⚠ What clearing does and does not mean
The tripwire says the venue is **admissible** — a byte-matched classical store cannot win it. It says
**nothing about whether we can win it.** The references sit at **0.578–0.701**; the classical floor is
0.274. A CLU arm has to be competitive in that upper band to matter, and the gate result gives no
evidence that it will be. **Admissibility is a precondition, not a result.**

## Open items added by this run
| item | owner |
|---|---|
| `pur_donor.py` / `pub_donor.py` write per-basin `.npz` under names that do not encode `--tagnote`; a re-run over the same keyset silently **overwrites** the earlier run's per-basin vectors. Hit and repaired in-flight; no verdict number affected | experiment-engineer, before any donor sweep is re-run |
| PUR reference set is 32 rows and is **PUR-only** — Kratzert 2019 / Heudorfer 2026 / Qu 2026 are PUB-only and excluded | carried; do not re-import them |
