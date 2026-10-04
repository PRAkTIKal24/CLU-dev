# `hub/composition-2` — REPORT (relaunch, iteration 2)
**Branch** `hub/composition-2` · **base** `945a3e9` (`base/iter2` = pin tree `345244bb…` + the
five reference-read files + the §2.2(b) gate patch) · **machine** Windows AMD64 (FORGIS_14)
**START** `2026-09-17T11:03:09Z` · **budget** 12 h true wall clock · **threads** 4
**Scored runs: 1 of 3 spent** — `eval_b`, **seed 0 only**. Quote every number as
**"scored 1/3 — seed 0 only, 3-seed pending full caches."**

---

## ⭐⭐ VERDICT
**The pre-registered bar is MET, with a WIN — and the single most useful thing this hub
produced is the measurement of how little that single-cell win can bear.**

> **R2's bar:** *distinct valid novel tuples on B at matched bytes, win-or-tie vs `knn_proto`
> under R1, with the capacity law and the decoder ceiling reported beside it.*

**On `eval_b`, seed 0, at 469,504 B = 0.9881 × `knn_proto`'s ledger:**

| | tuples_valid_novel / 72 |
|---|---|
| ⛔ decoder ceiling (NOT an arm) | **67** |
| ⭐⭐ **the object — `clu_linear`** | **42** |
| ⚠ `knn_proto` (the rival) | **12** |
| §C.1 matched-bytes lookup (feature medoids / factor medoids) | **0** / **0** |
| §C.2 landscape removed (zeroed / shuffled) | **0** / **0** |
| constant-output floor | **0** |

**Δ = +30.00 tuples, paired 95 % CI [+20.27, +39.73], McNemar exact p = 2.27 × 10⁻⁷ ⇒ WIN.**
⭐ **The verdict replicates independently.** A §A.5 spoke, barred from this hub's code, built
the object from the pinned library and measured **25 vs 12, Δ = +13.00 [+3.88, +22.12] — also
a WIN.** Two builds, two code paths, two wins.

### ⛔⛔ AND THE THREE NUMBERS THAT MUST TRAVEL WITH IT
1. ⭐⭐ **Over 105 held-out-shaped corners in two location-disjoint families the same object,
   rival and instrument give Δ = +1.09, CI95 [−1.52, +3.69] — a TIE.** That is the
   generalising estimate, and it is **not a tie for lack of power**: the CI is ±2.6 tuples.
   The two families agree to **0.24 tuples** (+0.96 over 49, +1.20 over 56) and three store
   seeds agree to **0.9 tuples**. ⛔ Nothing about it is fragile.
2. ⛔ **The B LEVEL does not replicate: 42 (this hub) vs 25 (the independent spoke)**, because
   my pre-registration failed to pin the store fit. ⛔ I do not quote "42" as reproducible.
3. ⭐⭐ **Both disagreements have one measured cause: the CORNER accounts for 90.9 % of the
   variance in Δ** (between-corner sd **12.92** tuples vs store-seed sd **4.09**).
   **B's +30 sits at z = +2.17** of a distribution now sampled 105 times, and the pooled CI
   **excludes +30 by an order of magnitude**.

⇒ **What I claim:** *the bar, as written, is met — and the object ties `knn_proto` at matched
bytes in expectation, measured 105 times. A 72-item single-cell bar cannot distinguish "won
this cell" from "ties in expectation", and this hub measured by exactly how much it cannot.*
⛔ I do not claim a 30-tuple advantage. ⛔ I do not claim the tie is a win.

### ⭐ A PRE-REGISTERED PREDICTION THAT SURVIVED — and one real mechanism
Before building a **fresh** family of 64 corners at `hue_lo ∈ {0.1, 0.3, 0.5, 0.7}` (disjoint
from every corner used for any selection here; B's 0.8 excluded), I filed six predictions
(`LOG.md` 16:20Z). **F1 — "Δ = +1.4, a TIE" — came back +1.196, TIE.** F5 (control = 0 on all
56) and F6 (the identity holds) were exact.
⭐ **F4 turned a post-hoc slice into a replicated mechanism:** the object's advantage is
**signature-dependent**, and all four signatures kept their sign **and their rank order** on
corners chosen before they were looked at —
`scale_hi ∧ orient_hi` **+8.50 → +6.21** · `scale_hi` only **+2.38 → +4.53** ·
`orient_hi` only **−5.00 → −2.67** · neither **−7.22 → −4.00**.
⚠ ⛔ It still does not explain B's +30 — B's signature is the best one, worth about **+6**,
not +30.

---

## 1. The four falsifiers
| # | falsifier | verdict | evidence |
|---|---|---|---|
| 1 | Held-out combinations rendered correctly, beating every rival including `knn_proto` | ⭐ **MET on B** (42 vs 12, p = 2.3 × 10⁻⁷; independently 25 vs 12, also a win); ⚠ **TIE over 105 corners** (+1.09 [−1.52, +3.69]) | `SCORED_B_seed0.json`, `pooled_dev.json`, `store_seed_spread.json`, `replication/REPLICATION.md` |
| 2 | ⛔ *retired by R2*; replaced by win-or-tie at matched bytes | ⭐ **MET** | as above |
| 3 | Dreaming: valid novel tuples at a declared temperature, **or DEAD** | ⛔ **DEAD**, in that word — with its mechanism | `c05_dream_*`, `c05a_readout_gap_*`, `c05b_coef_dream_*` |
| 4 | The store is load-bearing (`store_removed` collapses) | ⭐ **MET, absolutely** — 0 on B and **0.00 on all 105 corners**, zeroed *and* shuffled; independently confirmed | as above |

## 2. What was found
### 2.1 ⭐ The reference object is at the optimum of its own family — two searches say so
- **C04**: 45 head designs × rank × ridge at matched bytes (store fixed), 360 renders. Best
  beats the reference by **+0.375 tuples** — inside a corner sd of ~13.
- **C07**: 5 store geometries, each at its largest affordable M. Single-peaked in K:
  8.25 (K=8,M=44) · 11.12 (16,35) · ⭐ **22.50 (32,26)** · 21.62 (64,17) · 20.12 (128,10).
⇒ ⭐ **No departure**, which keeps charter §C.3's "same object" clause intact for the deletion
and refusal hubs. ⭐ C07 reproduced C02's reference row **exactly** through a different code
path — an internal consistency check that came free.

### 2.2 ⭐⭐ The store is used as a SIGNED BASIS, not a mixture
The head's coefficients are **~47 % negative**, carry **L1 ≈ 8.5 per well** (a probability
vector carries 1.0), and **0.00 % of wells — any request, any corner — are a probability
vector**. The read's output lies **far outside the convex hull of the stored modes**.
- ⭐ **That is why it is not a lookup.** §C.4 asks whether the state sits on its nearest atom;
  this output is 8.5 L1-units *outside the hull of its own atoms*. And the target tuple is
  absent from every stored row by construction, so ⛔ no lookup can be valid at all — measured:
  every matched-bytes lookup scores **0.00 on all 49 corners**.
- ⭐ **The pair-interaction block is load-bearing**: removing it costs **−6.55 tuples**
  (CI [−9.23, −3.87]) over 49 corners and turns the arm into a **loss** to `knn_proto`.
- ⭐⭐ **And it kills dreaming, structurally** (§2.3).

### 2.3 ⛔ DREAMING IS DEAD — with a mechanism, not as a null
Both mechanisms, the pre-registered grid `{0.0, 0.005, 0.05, 0.5}`, ⛔ never widened:
- **State space** (pinned `DreamLandscape` + `dream_rollout`): **0 at every β and every
  temperature**, every corner. ⭐ **CONTROL C proves the chain MOVED** — `|Δx|/|x|` rises
  0.134 → 0.254 → **0.718**, up to **46 % of wells change code**. ⇒ not a frozen chain: the
  dynamics travel far and the read still returns nothing. ⭐ Strictly more than iteration 1
  could establish, which could not separate "did nothing" from "did something invisible".
- **Coefficient space** (`a(T) = a(r) + T·σ_a⊙ξ`, declared before it ran): at **β = 0 — which
  is what dreaming IS — 0 at every temperature**, while producing **72/72 distinct tuples**
  (⭐ distinctness is free and worthless). At β = 1 temperature only **destroys**:
  38 → 35 → 34.7 → 0.7. ⭐ The maximum over the whole grid **is T = 0** = the zero-step launch.
- ⭐⭐ **Why:** every energy-driven state lies **inside** the mode hull; the object lives
  **outside** it. A τ sweep over three decades gives **0 at every τ**, best reconstruction
  still 25 % off. ⛔ Not fixed by temperature, τ, more steps, or a softer read — all measured.
- ⚠ **A defect in my own T = 0 leg**, found by reading the pinned code *after* the run and
  reported rather than silently re-run: `_mala_step`'s acceptance ratio is `0/0` at `temp = 0`,
  so every proposal is rejected. The corrected unadjusted leg moves 0.12 and still reads 0.
- ⛔ **Consequence I do not hide:** §C.3 lets a TIE become the win narrative only if the same
  object delivers refusal, exact deletion, **or** dreaming with valid novel tuples.
  ⭐ **Dreaming does not deliver it.** That clause must be carried by the deletion or refusal
  hubs on this same object, or not at all. ⛔ I do not claim it.

### 2.4 ⭐⭐ WHERE ALL THE VARIANCE LIVES — the result with the widest reach
Three store seeds × 49 corners, variance of the paired Δ decomposed:
| source | sd (tuples) | share |
|---|---|---|
| store-seed k-means init (within corner) | **4.09** | 9.1 % |
| ⭐⭐ **corner (between corners)** | ⭐ **12.92** | ⭐ **90.9 %** |

Every disagreement this hub encountered has the same single cause:
| disagreement | cause |
|---|---|
| selection half **+5.92** vs validation half **−3.80** | the corner |
| my B build **42** vs the spoke's B build **25** | the corner (the two builds **tie** over 49 corners, Δ = −0.53 [−1.94, +0.88]) |
| the spoke's raw-modes **36** vs Lloyd **25** on B | the corner (they **tie** over 49 corners, Δ = +1.04 [−1.87, +3.95]) |
| **B's +30** vs DEV's **+1.40** | the corner (z = +2.17) |

⭐ **Every alternative explanation was tested and every one failed to account for 17 tuples
on one cell:** the partition (Δ = −0.53, tie, n = 49) · the Lloyd budget (Δ = +1.04, tie,
n = 49) · the store seed (**9 %** of the variance, 3 seeds × 49 corners) · and the **shipped
Adam `train_store` fitter**, which the replication spoke named as the most likely explanation
and which moves the **wrong way** (38 → 33 and 35 → 28 on the two corners it completed; §4.9).
⛔ The corner-variance account is the only one left standing.

⛔ Not the construction. ⛔ Not Lloyd convergence. ⛔ Not the store seed. ⛔ Not the cell's
difficulty — the stratum of DEV corners with ceiling ≥ 65, i.e. as easy as B, is still a
**TIE** (Δ = +1.79). ⛔ Not grid position — Δ vs `hue_lo` has slope **−7.23/unit**, running the
**wrong way**; that was a hypothesis of mine and it is dead.
⭐ **It is which 72 rows you look at.**
⚠ **This bears on every hub whose bar is a single held-out cell, not only this one.** Flagged
for the coordinator as a programme-level observation.

### 2.5 R2's two required companions
- **Decoder ceiling: I measured 67 on B**, confirming the inherited **67.67 ± 2.49** almost
  exactly (pre-registered as P9, and independently re-measured as 67 by the replication
  spoke). Across 49 DEV corners it is **54.96 ± 16.07**.
- **The capacity law — BOTH of them, which is what R2 asks for.**
  | read | law | R² |
  |---|---|---|
  | ⛔ inherited **ARGMAX** quantiser (iteration 1) | `rel-err = e^1.346 · bytes^−0.1651` | 0.9960 |
  | ⭐ **LINEAR read (this object), measured here** (C11, 12 corners, 23× byte span) | `rel-err = e^0.244 · bytes^−0.1030` | 0.9082 |
  ⭐ The linear read's residual prefactor is **≈ 3× smaller at every byte count**
  (e^0.244 = 1.28 vs e^1.346 = 3.84). ⭐⭐ **That is the quantitative reason the same store
  renders 42/72 on B under this read and 0 under the argmax read**, and it is why iteration
  1's "77 GB / 147,000×" figure is ⛔ **not a bound on this object.** The inherited law is
  reported beside mine with its scope stated; ⛔ it does not stand in for a law it does not
  describe.
- ⭐⭐ **And the tuple curve SATURATES, which R2's phrasing could not have anticipated.**
  Tuples vs bytes (K, pairs, rank, ridge fixed; only M moves; 12 corners, ceiling 61.17):
  | × `knn_proto` | 0.17 | 0.32 | 0.51 | ⭐ **0.99** | 1.96 ⚠ | 3.89 ⚠ |
  |---|---|---|---|---|---|---|
  | tuples /72 | 0.25 | 1.67 | 3.83 | ⭐ **23.67** | **36.92** | **37.00** |
  ⭐ **Doubling the store from 2× to 4× budget buys +0.08 tuples.** ⇒ ⛔ above ~2× budget the
  store is **not** the binding constraint — the **addressing head** is (rank-16, additive +
  two pair blocks). ⭐ The shipped budget already buys **64 %** of everything capacity can
  buy and sits at the knee. ⚠ Per corner the curve is not monotone, only the mean is:
  reconstruction and composition come apart above the knee (c2: 35 → 18 → 8 tuples while its
  rel-err keeps falling). ⛔ Reported as measured; I have no mechanism for that split and do
  not invent one.

## 3. §C floor checks
| check | result |
|---|---|
| §C.1 matched-bytes lookup (57 real pool rows; feature medoids **and** factor-space medoids) | ⭐ **PASSES** — **0** on B, **0.00 on all 49 declared DEV corners** |
| §C.1 vs the rival `knn_proto` | ⭐ B: 42 vs 12 (and 25 vs 12 independently). ⚠ **105 DEV corners: tie** (+1.09 [−1.52, +3.69]) |
| §C.2 landscape removed (zeroed **and** shuffled, bytes identical) | ⭐ **PASSES** — **0.00 on all 105 corners** and on B, in every build; independently confirmed; the shuffled control makes 65 distinct tuples and **0 valid** ones |
| §C.3 "a tie is the win narrative" | ⛔ **NOT CLAIMED** — dreaming is DEAD here |
| §C.4 settled state on nearest atom ⇒ lookup | ⭐ **NOT a lookup** — 8.5 L1-units outside the hull of its own atoms |
| byte ledger | ⭐ **469,504 B = 0.9881 ×** `knn_proto` — matched **and under**; independently confirmed to the byte |
| `assert_request_only_factored` / `assert_no_bypass` | ⭐ **PASS on the scored run itself** (n = 72 / 64, bit-identical under feature replacement); independently confirmed |
| identity `distinct_valid_novel = 72 × exact_match` | ⭐ **asserted per render**; held on B; independently confirmed for all four arms |

## 4. ⛔ LIMITATIONS
1. ⛔⛔ **Every number is bridge seed 0.** Seeds 1 and 2 are cache-blocked (`scope="used"`;
   ~32,200 new encodes needed, forbidden). ⛔ The 3-**store**-seed spread in §2.4 is **not**
   bridge-seed spread and is never quoted as such.
2. ⛔⛔ **My PREREG under-specified the object** — it pinned ridge to one decimal and left the
   equal-well partition, the k-means++ RNG stream and the Lloyd budget free. The §A.5 spoke
   caught it and was right. ⭐ Amendment 1 (appended, dated) fixes it. ⚠ The B **level** is
   therefore not a reproducible number; the **verdict** is.
3. ⚠⚠ **The B win rests on ONE cell of 72 items**, and §2.4 quantifies what that is worth.
4. ⛔ **Pre-registered prediction P4 FAILED**: I predicted a TIE and measured a WIN. Recorded
   as a finding. P5 also failed on verdict (predicted loss, measured tie). P1 was badly
   under-predicted (19 vs 42, inside its interval). **P2, P6, P7, P8, P9, P10 were exact.**
5. ⚠ The corner-signature stratification (§2.4 of `LOG.md`) is **post-hoc**. ⛔ Not a result.
6. ⛔ `exp_gen_cell.ARMS` does not contain this object, so the **stock frozen cell cannot
   produce this column**. ⛔ I did not edit the pinned harness; I used the same frozen
   instruments through my own runner (R3-compliant, same-device). ⚠ **Open question for the
   coordinator:** whether `hub/generation-table-2`'s cell carries this exact object, in which
   case this scored run was avoidable — flagged in the PREREG **before** spending it.
7. ⚠ DEV corners are Montero-**shaped** cells cut from inside the train split; 8 + 7 were
   **refused** for touching `eval_a`, ⛔ never trimmed.
8. ⚠ My sub-agent left an orphaned 21 GB process running after it reported; I identified it by
   command line and stopped it (LOG 14:07Z). ⛔ No peer hub's process was touched.
   ⚠ **I later corrected myself** (LOG 18:13Z): I had called its scientific cost "near zero",
   and the spoke was right that the column it would have produced mattered. I ran that column
   myself as C12.
9. ⚠⚠ **C12 (the shipped `train_store` column) completed only 2 of 6 corners.** Adam training
   grew to **50.3 GB resident** and drove this shared four-hub box to **0.4 GB free of
   63.4 GB**; the third corner had been thrashing for 52 minutes. I stopped it and free RAM
   returned to 50.8 GB. ⛔ **My defect:** I launched it without a memory or per-corner
   wall-clock cap, having been told the same variant had already blown a 2,400 s cap once.
   ⭐ Resumability meant the kill cost **one corner, not the run**. ⛔ n = 2 supports only the
   narrow claim in §4.9 — it does **not** establish that Lloyd beats the shipped trainer.

### 4.9 ⭐ The shipped `train_store` column — the replication spoke's own follow-up, run
`fast.kmeans_pq`'s docstring says it is *"deliberately NOT the shipped `train_store`… The
shipped trainer is run beside it for the like-for-like column"* — and this hub had never run
it. The spoke named it the most likely explanation of 42 vs 25. ⛔ Only the mode fitter
differs; the byte ledger is asserted unchanged at 469,504 B.

| corner | Lloyd | **shipped `train_store`** (Adam, 1200 steps) | Δ | `knn_proto` |
|---|---|---|---|---|
| c0 | **38** | **33** | **−5** | 5 |
| c2 | **35** | **28** | **−7** | 24 |

⭐ **It moves the wrong way on both corners and beat Lloyd on neither** — mean −6.0. ⇒ the
42-vs-25 gap is **not** a hidden fitter difference, and the last live alternative to the
corner-variance account is closed. ⚠⚠ **n = 2**, stopped for the resource reason in §4.9's
limitation (see §4.9 note and Limitation 9): ⛔ this does **not** establish that Lloyd beats
the shipped trainer, only that the trainer did not reproduce 42 and did not move toward it.

## 5. COST LINE
| item | value |
|---|---|
| true wall clock | see `ELAPSED.txt` (measured against `START_UTC.txt` and `date -u`, ⛔ never model-perceived) |
| compute | **LOCAL CPU only**, 4 threads throughout (AMENDMENT 1) |
| GPU | ⛔ **none** |
| GCloud / paid | ⛔ **none. Zero spend.** |
| CSF3 | ⛔ **NOT RUN** (2FA, unreachable by any hub) |
| network | ⛔ none |
| scored runs | ⭐ **1 of 3** (seed 0, `eval_b`) |
| DEV renders scored | ~1,800 across **105 corners** in two families, 3 store seeds and 10 units — ⛔ none touching B |
| sub-agents | 1 (independent replication, Opus) — finished, and its orphaned process stopped |
| replication | ⭐ **independent spoke, own code, barred from my scripts**: 6 of 7 quantities CONFIRMED exactly; ⛔ **1 REFUTED** (the headline level) — and the refutation was then run down on 49 corners |
