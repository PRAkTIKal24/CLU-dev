# hub/composition — REPORT
**Branch** `hub/composition` · **base** `8e36bc9` (tree `345244bb…` = `pin_tree`, i.e. pin `4125fce7…`)
**START** `2026-09-16T19:10:40Z` · **Budget** 12 h true wall clock · **Core budget** 5 threads
**Scored runs spent: 0 of 3.** ⛔ None were needed for the target, and I recommend none be spent here.

---

## ⭐ VERDICT: **NOT SOLVED — and the target is not reachable through this store. Here is the number.**
A well-controlled failure that names the obstacle, as the charter says is the expected outcome when the
science does not land. ⛔ Nothing here was manufactured to fill the window; the hub finished its science
in **1 h 25 min** of a 12 h budget and spent the rest verifying and replicating it.

## 1. The four falsifiers, answered
| # | falsifier | verdict | evidence |
|---|---|---|---|
| 1 | Beat every rival incl. `knn_proto` on the compositional factor | ⛔ **NOT MET** | CLU arm **0.00**, `knn_proto` **14.33 ± 4.78** distinct valid novel tuples / 72 |
| 2 | ≥ 60 / 72 distinct **valid novel** tuples | ⛔ **NOT MET** | best on the whole table is `knn_proto`'s **14.33**; ours is **0.00**; the **decoder ceiling** is **67.67 ± 2.49** |
| 3 | Dreaming: valid novel tuples at a declared temperature, **or DEAD** | ⭐ **NOT DEAD — but INERT** | at **T = 0.5**, β = 0 gives **463 distinct / 447 novel / 4 valid-novel** of 512. ⛔ But the **zero-step launch gives the same**, and **β = 0 ≡ β = 1** at every grid temperature |
| 4 | The store is load-bearing (`store_removed` collapses) | ⭐ **MET** | `store_removed` → **1.00** distinct tuple, in-corner balacc exactly **0.5000**, 3/3 seeds |

⭐ Falsifiers 1, 2 and 4 were settled from **frozen, Mac-produced artefacts** — 0 scored runs.
Falsifier 3 was settled on `val` at the **pre-registered grid `temp ∈ {0.0, 0.005, 0.05, 0.5}`**, ⛔ never
widened.

## 2. ⭐⭐ WHY — the obstacle, isolated by controls and quantified
1. **It is not composition.** On `val` — combinations the store has **SEEN** — the decoder renders
   **491/512 (95.9 %)** correctly and the best read renders **28/512 (5.5 %)** at 8× the byte budget.
2. **It is not addressing.** An oracle allowed to see the true feature and pick the best atom per well
   beats the address book by **+2 items of 512**. No address book can buy more.
3. **It is not bytes-as-such, and it is not the metric.** Per-channel whitening — which I pre-registered
   as my predicted remedy — **failed** (47 → 39 at K = 64). ⛔ Reported as a failed prediction.
4. ⭐⭐ **It is SELF-ALIGNMENT of the quantiser's residual.** At one fixed rel-err of 0.3486
   (⭐ values below are 25-draw means ± sd after the independent replication **refuted** my original
   single-draw figures of 95 and 393 — see LOG 20:46:38Z):
   | perturbation | valid / 512 |
   |---|---|
   | isotropic Gaussian, matched norm | **382.4 ± 11.1** |
   | residual negated (`x − e`) | **162** (deterministic) |
   | residual permuted across items | **115.6 ± 7.7** |
   | ⛔ the store's actual output (`x + e`) | ⛔ **10** (deterministic) |
   ⭐ The load-bearing comparison is `x − e` (**162**) vs `x + e` (**10**) — **16×** at identical
   magnitude AND identical residual subspace — ⛔ not the Gaussian row, which the replication rightly
   called a weak control in D = 2048.
   A nearest-codeword read's error **is, by construction, the part of the item the store does not
   contain** — and that is exactly what the classifier reads factor identity from. ⛔ Re-weighting the
   metric cannot remove it (which is *why* 3 failed); only a finer tiling can.
5. ⭐ **And it is not a generalisation failure.** A **surrogate held-out cell** — the opposite corner of
   the grid, 72 rows / 72 tuples, held out of the store fit — has the **same** quantisation error as
   seen data (rel-err **0.3432** vs **0.3524**) and a valid count consistent with the same rate. ⇒ the
   bound measured on seen data **transfers to unseen combinations**; composition adds no extra penalty.
6. ⭐⭐ **So it is capacity, and capacity is measured.** Over a **128× span** of store sizes,
   `rel-err = e^1.346 · bytes^(−0.1651)`, **R² = 0.9960**. The bar needs rel-err **0.0612**.
   ⇒ **≈ 77 GB — about 147,000× the deployed 524,288 B store.**
   Merely to **match `knn_proto`**: **≈ 85 MB, 162×** (only ~5× beyond measured — the solid one).
   ⚠ Even granting an exponent **twice as favourable as any of the seven measured**, the bar still needs
   **383×** the budget.
   ⭐ For scale: storing **all 480,000 training rows verbatim** is **3.93 GB** — the quantiser needs
   **19.6× more than that**. At `d_k ∈ [303, 573]` the codebook is less byte-efficient than a lookup
   table, which is §C.4 arriving as economics.

## 3. What I found that is worth keeping
- ⭐ **An identity:** on B, `distinct_valid_novel = distinct_valid = 72 × exact_match`, because all 72
  held-out targets are distinct and **none of them occurs in any of the 479,928 train rows**. ⇒ the bar
  "≥ 60/72 valid novel" **is** "`exact_match` ≥ 0.8333 on B", and the whole frozen table's headline
  column is derivable with no new run.
- ⭐ **Novelty does not bind here.** 24/24 frozen rows had `distinct_novel == distinct`. Reported as a
  finding; ⛔ the definition was not changed.
- ⭐ **The partition is a real lever** at constant total bytes (`store = M·D·4` is independent of K):
  K = 5 → 64 lifted untrained round-trip from 6 to 47 of 512 (~8×). ⚠ It plateaus, because it does not
  touch self-alignment — and ⚠ my own *trained* K=5 store scored **below** the shipped one, so the init
  sweep needs re-checking before anyone builds on it.
- ⭐ **`exp_gen_cell` has never run the `x0` addressed read.** It is on the pin, unrun on this table, and
  `PREREG-hub-composition-1.md` is written so the Mac can run it at 3 seeds.

## 4. §C floor checks
| check | result |
|---|---|
| §C.1 matched-bytes lookup floor | ⛔ **FAILS for the CLU arm** — `clu` (8/512) is inside the bootstrap CI of the store's own round-trip (10/512, CI [4,16]); there is no addressing gain left |
| §C.2 landscape-removed control | ⭐ **PASSES** — `store_removed` → 1 distinct tuple, balacc exactly 0.5000 |
| ⭐ per-factor loss, with `n_values_realised` | `scale` **75.5 %** and `orientation` **82.6 %** of the ceiling lost; hue essentially intact — independently reproducing §E.4's "90.5 % of scale, 100 % of orientation" |
| §C.3 "tie is the win" clause | ⛔ not claimed |
| §C.4 settled state on nearest atom ⇒ lookup | ⭐ **reported as such** — and §2.5 gives the economic version |

## 5. ⛔ LIMITATIONS — stated plainly, not buried
1. ⛔⛔ **Every number I produced is SEED 0.** Seeds 1 and 2 are cache-blocked on this machine
   (32,395 / 32,437 of 33,039 needed rows missing; producing them means re-encoding on hardware that
   disagrees with the Mac cache by 51 ULP). I report bootstrap CIs and arm-seed spread instead, and ⛔ I
   do not pretend they are bridge-seed spread.
2. ⛔ The 147,000× figure extrapolates ~4,600× beyond the largest store measured. The **162×** to match
   `knn_proto` does not, and the conclusion survives an exponent twice as favourable as any measured.
3. ⛔ `exp_gen_cell` could not run **at all** here — the re-encode gate's `will_extend` scoping is not
   wired at its call site, so even seed 0, where the shortfall is provably 0, is refused. ⛔ I did not
   edit the frozen harness to get around it. **This is an open question for the Head.**
4. ⛔ The frozen-table verdicts inherit the Mac's numbers. I re-derived them from disk and my instrument
   reproduces the harness's `distinct` and `exact_match` on 24/24 rows, but I did not re-run them.
5. ⚠ **The round-trip bound is IN-SAMPLE for the store** (all 512 `val` rows are in `store_train`). The
   direction is conservative — an in-sample store does better, so 10/512 is a genuine upper bound — but
   "491 vs 10" pits a near-held-out decoder number against an in-sample store number.
6. ⛔ **The WEAK reading of "valid" is unmeasurable for the rivals.** `exp_gen_cell` persists per-render
   tuples only for `dynread`. For those 12 rows strict and weak **agree at 0** (⛔ 0/72 items land in the
   held-out cell at all). For `knn_proto`, `clu`, `lace`, `ebm_compose` it is **NOT MEASURABLE** without
   a run, ⛔ and is not reported as zero. PREREG Amendment 1 requires the scored run to persist them.

## 6. COST LINE
| item | value |
|---|---|
| true wall clock | see `ELAPSED.txt` (measured against `START_UTC.txt`, `date -u`) |
| compute | **LOCAL CPU only**, ≤ 5 threads, per AMENDMENT 1 |
| GPU | ⛔ **none** — no run here had a headline as its best case (§D) |
| GCloud / paid | ⛔ **none. Zero spend.** |
| CSF3 | ⛔ **NOT RUN** (2FA, unreachable by any hub) |
| network | one 268 MB public-bucket download (`3dshapes.h5`, Apache-2.0), sha256 verified against the code's own pin. Free. |
| scored runs | ⭐ **0 of 3 spent** |
| sub-agents | 1 (independent replication, ~7.8 min, 2 threads); confirmed finished before close |
| replication | ⭐ **independent spoke, own code, barred from my scripts**: 5 of 7 numbers CONFIRMED exactly, 1 within draw noise, ⛔ **1 REFUTED and corrected** |
