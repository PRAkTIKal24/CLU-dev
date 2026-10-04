# PREREG — `gen-read-compose` (the B path through the READ)

**Filed by:** experiment-engineer · **Written BEFORE any harness that measures a reported number existed.**
**Base:** `main @ 1bcfcb9` (the task names `3faef27`; local `main` has since advanced by the
`gen-baselines` merge — `3faef27` is an ancestor, see report §Git).
**Path under test (R26 ruling 3):** `request → address_requests → per-well argmax → read_from_codes →
frozen decoder (v1_capacity_schedule) → frozen probe`.
⛔ No `DreamLandscape`, no `dream_rollout`, no `clamp_from_posterior`, no MALA/ULD, no coupling field.

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** *isolation* — does composing a request through the store's READ produce the requested combination?
- **Laundering control:** ⭐ `store_removed` (read path, store content ablated, `mean` deletion) **and**
  `proto` (no-store additive per-factor prototype composition). Both are full arms on every number.
- **Falsifies:** the read path does not beat `store_removed` on `shape` beyond seed spread.
- **Does NOT falsify:** losing to `proto`/`knn_proto` in absolute terms, or sitting below the decoder
  ceiling. Reported against **both** ceilings (R23.5), never absolutely.

## The arms (four, R27 amendment)
| arm | store→decoder channel | what it is |
|---|---|---|
| `read` | **30 bits** (`K·log2 M = 5·6`) | argmax mode per well, `read_from_codes` |
| `clu_wide` | composed per-well mode vector | mixture weights **softmaxed from the REQUEST's** per-well log-scores — ⛔ never from features |
| `store_removed` | 30 bits into an ablated store | the laundering control |
| `proto` | unbounded (2048 floats) | `FactorPrototypes.embed_many`, the no-store composition |

## Gate statistic (R20/R23, ⛔ not invented here)
In-corner `shape` **balanced accuracy**: positives = the arm's renders of the 72 held-out (B) requests,
negatives = its renders of the matched-seen requests (same colour corner, a seen shape). Chance = 0.5.
AUROC is a mandatory companion and never the gate. `shape` is reported separately from the aggregate always.

## Point predictions — committed before measurement

| # | prediction | interval | derivation |
|---|---|---|---|
| **P1** ⭐ | `read` in-corner `shape` balacc **> `store_removed`** by **≥ +0.10** | — | `store_removed` under `mean` deletion emits ONE constant feature vector for every request ⇒ one constant image ⇒ the probe's argmax is constant ⇒ TPR and TNR are complementary ⇒ balanced accuracy **exactly 0.5000**. `read` must clear 0.60 (see P2). **This is the whole claim; it is checked first.** |
| **P2** | `read` in-corner `shape` balacc = **0.65** | 0.55 – 0.72 | The store ceiling with *real-feature* addressing is **0.6866** (R22). Request-addressing recovers ~0.86 of the code set exactly (R26), so the read must sit at or a little below that ceiling. |
| **P3** | `store_removed` in-corner balacc = **0.5000** | 0.49 – 0.51 | see P1's derivation; `gen-dream-sampler` measured the same ablation collapsing to a single code configuration (A aggregate 0.1196 ± 0.0151). |
| **P4** | `read` A-split **aggregate** accuracy = **0.56** | 0.52 – 0.60 | `gen-dream-sampler` §4: the `full`-protocol **zero-step** control — algebraically the same object as this arm, decoded with all five wells set from the address book — measured **0.5611 ± 0.0041**. |
| **P5** | `store_removed` A-split aggregate = **0.12** | 0.09 – 0.15 | `gen-dream-sampler` measured 0.1196 ± 0.0151 / 0.1213 ± 0.0139 for the identical ablation. |
| **P6** ⭐ | `proto` in-corner balacc ≈ `knn_proto`'s **0.8884**, and it **BEATS** `read` | 0.85 – 0.93 | It is the same estimator (`FactorPrototypes`, R8) on the same rows through the same decoder. ⛔ Pre-registered as a **loss for us**: an over-resourced, unbounded-channel arithmetic baseline is expected to win on this cell, and that is R27.1's decided-in-advance reading. |
| **P7** ⭐ | `clu_wide` **beats `read`** by **+0.02 … +0.10** and **loses to `proto`** | 0.68 – 0.85 | Softening the 30-bit pointer to a mixture over the same modes strictly adds information about the request's posterior, but the modes remain a 64-entry-per-well vocabulary, unlike a per-(factor,value) mean. ⛔ **If `clu_wide` ties `proto` we say composed modes are prototype means with machinery attached** (R27 amendment, in those words). |
| **P8** | **Metric invariance check** (R26 standing rule, third strike): duplicating every negative (same features, different partition / class balance) leaves balanced accuracy and AUROC **bit-identical** (|Δ| < 1e-12) while raw accuracy moves by > 0.05 | — | Balanced accuracy is a mean of per-class recalls; duplicating whole items within a class changes neither recall. Raw accuracy is prior-weighted and must move. ⛔ Run and recorded **before** any arm is scored. |
| **P9** | **Byte ledger = 524,288 B** exactly, coupling field **excluded** | exact | `n_modes·D·4 = 64·2048·4`. R26's consequence: `fit_code_coupling` is referenced only by the retired landscape, so its 163,840 B leave the deployed store and the 688,128 B target of `gen-baselines` amendment A5 does not apply here. |
| **P10** | `read` B-split `shape` **raw** accuracy > `store_removed`'s, and the A-vs-B gap on `shape` is ≥ 0.10 | — | B is one held-out cell (`shape = cylinder` in the ≥0.8 hue corner); the compositional deficit is the only genuinely compositional term (R16). |

## What would make me report a FAILURE rather than a result
1. P1 fails ⇒ composition is not the store's. It returns to Head + Advisor, ⛔ not to tuning.
2. The decoder hash/recipe assertion fires, or the recipe is not `v1_capacity_schedule` ⇒ every number is void.
3. `clu_wide` touches any feature tensor ⇒ the arm voids the no-bypass guarantee and is withdrawn, not reported.

## ⛔ Out of scope, declared
No kill-rule verdict (that is `gen-cell-run`'s). No C set, no refusal number, no partial-request completion.
