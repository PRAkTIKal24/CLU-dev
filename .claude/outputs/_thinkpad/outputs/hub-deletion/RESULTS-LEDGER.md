# RESULTS LEDGER — `hub/deletion-2`
Every scored run, and every DEV run that produced a number I quote. Appended as it happens.
**Budget: 3 scored runs per target across all attempts and machines. Spent before today: 0/3.
⛔ This iteration: at most ONE, seed 0 only.**

| # | when (UTC) | kind | tag(s) | set | machine | scored? | result |
|---|---|---|---|---|---|---|---|
| — | 2026-09-17T11:03:55Z | GATE, not scored | `det_A`, `det_B` | — | FORGIS_14 Win | no | *(pending)* |

## Scored-run counter
**0 / 3 spent.** The single permitted seed-0 scored run is defined in `PREREG-01-design.md` §6 and has
not been fired.

## Provenance (Head R3)
Every number below was produced on **FORGIS_14, Windows 11 AMD64, 4 threads, CPU jax 0.9.0,
numpy 2.4.1, Python 3.11.16**, from the **Mac-written frozen feature cache**
(`features_sha256 d6728f01…`, `encoder_param_hash 98d32602…`, `scope="full"`, 480,000 rows) and the
**Mac-written frozen decoder/probe** (`decoder_param_hash` asserted at every eval).
⭐ Every arm in every headline comparison — CLU and `knn_proto` alike — is **re-run here**, so the
comparison is same-device by construction. ⛔ No Mac-published row is mixed inside a paired comparison.

| 1 | 2026-09-17T11:26:07Z | GATE (blocking), **not scored** | `det_A` vs `det_B` | n/a (full pool, 32,768 live rows) | FORGIS_14 Win | no | ⭐ **PASS** — 22/22 arrays bitwise identical; `store_digest d0eacf3b…`; `loss_final 32.049007415771484` both; walls 554.9 s vs 728.3 s (31% apart, same bytes) |
| 2 | 2026-09-17T11:38:51Z | GATE extension, **not scored** | `det_C` vs `det_A`, `det_C` vs `det_B` | n/a (full pool) | FORGIS_14 Win | no | ⭐ **PASS both** — three-way bitwise identity across 3 processes at walls 554.9 / 615.9 / 728.3 s. `det_C` built under the POST-edit code (`code.combined 481f65ea…`) ⇒ the 11:20:14Z edits are **measured** behaviourally inert |
| 3 | 2026-09-17T11:55:31Z | DEV, **not scored** | `del_f2v5` (delete `object_hue=5`) vs `det_B` | `dev` 4,096 + `tgt_f2v5` 1,024 | FORGIS_14 Win | no | **cond 1 PASS**: exact-match 0.5107→0.0049; hue 5 rendered on **0 of 4,096** dev requests. **A3.8 mechanism PASS bit-exact**. Collateral (3,637 non-`v` requests): `C_clu −0.0139`, `C_knn +0.0009`, **Δ = −0.0148 ⇒ tie under R1** (single deletion, uncorrected). Partition ARI 0.5656 |
| 4 | 2026-09-17T12:08:42Z | DEV, **not scored** | `plc_f2v5` (churn control, `n_live` = 29,561 identical to `del_f2v5`) | `dev` + `tgt_f2v5` | FORGIS_14 Win | no | ⭐ **Rebuild alone churns 18.95 % of answers** (deletion 19.77 %) ⇒ the churn is recanonicalisation, not deletion. Placebo **improved** the CLU (+0.0044). **Churn-corrected Δ = −0.019567**, CI [−0.0253, −0.0138] ⇒ **tie by 0.0004**. A3.7: placebo leaves hue 5 intact (0.5371) ⇒ condition 1 placebo-controlled. ARI(plc) 0.4924 < ARI(del) 0.5656 but κ(plc) 0.6086 > κ(del) 0.5726 ⇒ **layout/answer decoupling measured** |

---
# LEDGER CONTINUED by `clu-dev-1e` — the co-agent `clu-dev-89` stopped recording at 12:08Z
⚠ It disappeared as a session between 12:05Z and 14:04Z; its detached driver kept building. Rows 5+
are mine, every number re-derived from raw artefacts with my own code. See
`INCIDENT-two-agents-clu-dev-1e.md` §11.

| # | when (UTC) | kind | tag(s) | set | scored? | result |
|---|---|---|---|---|---|---|
| 5 | 2026-09-17T13:25–15:22Z | DEV, **not scored** | `del_f0v5`/`plc_f0v5`, `del_f1v5`/`plc_f1v5`, `del_f3v4`/`plc_f3v4`, `del_f4v2`/`plc_f4v2`, `del_f5v7`/`plc_f5v7` | `dev` 4,096 + targeted 1,024 each | no | five further deletion+placebo pairs, completing the six-deletion cell |
| 6 | 2026-09-17T14:42Z | DEV, **not scored** | `del_conj_oh5_sh2` / `plc_conj_oh5_sh2` | `dev` + `tgt_conj` 1,024 | no | ⭐⭐ **THE ONE GENUINE WIN.** Joint render of the deleted conjunction: CLU **0.3672 → 0.0107** (placebo 0.3037); `knn_proto` **0.9092 → 0.5488** (placebo 0.9053). The table **re-composes** what it cannot forget; the CLU cannot, because no mode was ever fitted on the combination. Collateral Δ = −0.0058 ⇒ tie ⇒ forgotten at no differential cost |
| 7 | 2026-09-17T15:07Z | DEV, **not scored** | `del_f5v7` (delete `orientation=7` = 0.0°, the null pose) | `tgt_f5v7` 1,024 + `dev` | no | ⛔ **CONDITION 1 FAILS, BOTH ARMS.** `cond_5[7]` bit-exactly uniform, yet CLU renders orientation 7 on **25.0 %** of requests naming it and **12.6 %** of all dev renders (§6a bar 0.117); rival 0.5977. ⇒ the mechanism cannot delete a value that is the model's **default output** |
| 8 | 2026-09-17T15:22Z | ⭐ **DEV HEADLINE**, **not scored** | all six `del_f*` vs their placebos vs `det_C` | `dev`, non-`v` requests | no | **Churn-corrected Δ per deletion**: +0.01162, +0.01514, −0.01957, −0.00195, +0.01074, +0.00216. **Mean +0.003023, sd 0.012774, bootstrap 95 % CI [−0.00714, +0.01121]** ⇒ **R1 TIE, banked.** 0 of 6 outside ±m. ⚠ **Supersedes row 4's single-unit reading** — `del_f2v5` is the only negative of six and is the outlier |
| 9 | 2026-09-17T16:01Z | ⭐⭐ **MECHANISM RESULT**, **not scored** | `plc_f4v2`, `plc_churn_r1/r2/r3` (4 replicates, `n_live` 24,534, 6 pairwise) | `dev` | no | **Rung 2 — pure recanonicalisation instability.** Two rebuilds that deleted nothing semantic, same `n_live`, differing only in which arbitrary 8,234 rows went, disagree on **18.83 % ± 0.84 %** of answers (kappa **0.5967 ± 0.0296**, ARI 0.681 ± 0.082). `knn_proto` **1.40 % ± 0.05 %**. **Ratio 13.4×.** Rung 1 (identical live set) = **0.0000 churn, kappa 1.0000** ⇒ not build noise |
| — | 2026-09-17T15:22Z | ⛔ **SCORED RUN: NOT FIRED** | held-out B (72 rows) | — | **no** | **Deliberate. 0 of 3 spent.** (i) foregone tie — pooled B half-width ≈ 0.01664 vs m = 0.020 with DEV's effect at +0.0030 ± 0.0058, so a verdict change needs ~6 DEV se; (ii) B is entirely `shape=cylinder` and **cannot test row 6**, the one genuine win; (iii) pre-deletion the CLU is at 1.0000 on all three hues and below the ablated store on `scale`/`orientation`. ⚠ My *power* objection was withdrawn — the six deletions are effectively independent on B (r = −0.0045). Recommendation: preserve all three for the 3-seed final |

## FINAL SCORED-RUN COUNTER
⭐ **0 / 3 spent on this target, on any machine.** Every number in this ledger and in
`REPORT-hub-deletion.md` is a **DEV** number on non-held-out rows (`dev` = 4,096 train rows the store
pool never held; targeted sets = 1,024 rows each, same provenance). ⛔ **Held-out B was never touched.**

## Provenance restated (Head R3)
All rows produced on **FORGIS_14, Windows 11 AMD64, 4 threads, CPU jax 0.9.0 / numpy 2.4.1 /
Python 3.11.16**, from the **Mac-written frozen feature cache** (`scope="full"`, 480,000 rows,
`features_sha256 d6728f01…`) and the **Mac-written frozen decoder/probe** (hash asserted at every
eval). ⭐ Both arms re-run here in-process ⇒ same-device by construction. ⛔ No Mac-published row
appears inside any paired comparison. ⛔ `coverage_shortfall = 0` and `encoder_constructed = false`
recorded in every `build.json` — **nothing was ever re-encoded.**
