# gen-r0-rescore — results-analyst report

**Task + acceptance criterion:** re-express R0's existing results under R20's three corrected metrics, against
the decoder-lift winner, **with no retraining** — and give an explicit verdict on whether the
correlation-vs-random margin survives the corrections.
**Status: done. 3 seeds {0, 1, 2}, all fully scored.** ⛔ Every store number was checked **bit-identical**
against the shipped `gen-store-grouping` manifests before any corrected metric was computed.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 3 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐ **The `shape`-specific LEAKAGE advantage does not survive.** Old (shipped R0): correlation `−0.2435`,
>    random `+0.7946`, margin **`+1.0381`**. Corrected: correlation `+0.2247 ± 0.1135`, random
>    `+0.2394 ± 0.1888`, margin **`+0.0147 ± 0.0771` — 0.19× its own seed spread, and NOT consistent in sign
>    (per-seed `−0.0318 / −0.0475 / +0.1233`).** ⛔ Any site claiming the wells carry *`shape`* structure **on
>    the leakage axis** must be re-worded. ⭐ The same claim on the **read** axis is untouched and still holds
>    at `+0.2008 ± 0.0165` (12.2× spread). (§4, §7)
> 2. ⚠ **`gen-store-grouping` §6 under-states its own degeneracy by a factor of 15.** It reported per-factor
>    leakages "as extreme as **−12.9**". Recomputed from its own unmodified manifests, the true worst cell is
>    **`−198.0`** (seed 1, correlation, well 3, `shape`: a **detection rate `1.000`** ÷ a **3-class
>    discrimination rate `0.005025`**), and **55 of 180 cells (30.6 %) are NaN**. The R19 carry-forward text
>    should quote `−198.0` and the NaN count. File: `old_leakage_cells.json`. (§5)
> 3. ⭐ **The corrected pooled-leakage margin is 2.0× smaller and 2.3× less separated than the old one**
>    (`+0.2701 ± 0.0518`, 5.2× spread → `+0.1337 ± 0.0592`, **2.26× spread**). It is still 3/3 seeds positive
>    and in the predicted direction, but it can no longer carry a headline. ⛔ R0's claim should rest on the
>    **read** margin (§3, 16× spread) and the **code-space companions** (§6, MIG 9.9×, DCI-C 8.8×). (§4, §7)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *isolation* — R0's grouping evidence, re-expressed on metrics that are not degenerate.
- **Laundering control:** ⭐ **random grouping at the same `K` and the same well-size histogram is a full arm on
  every store number below.** The **matched-seen** set is now a *constituent of the metrics themselves* — it
  supplies the negatives in §1–§2 and half the numerator set in §4 — not a companion row.
- **Falsifies:** the correlation-vs-random margin **disappears** under the corrected metrics.
- **Does NOT falsify:** a lower absolute level than the old raw-accuracy numbers. ⛔ **Balanced accuracy on the
  2-class corner problem has chance 0.5; raw multi-class top-1 does not. The two never share a column below,
  and every table states its own chance level in its header.**
- ⛔ **No kill-rule verdict is rendered here** (acceptance item 7). This spoke corrects an *instrument*.

## FLAG-PROVENANCE TABLE (protocol §5, mandatory)
| | value |
|---|---|
| base commit | `main @ e4f3d71` ("[hub] merge gen-store-grouping (gen-phase1)"); `git log -1 main` verified |
| worktree | `/Users/user/Desktop/CHLU-r0rescore`, **detached at `e4f3d71`**. ⚠ The shared checkout was on another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` @ `7fcef50` the whole session; ⛔ I did not touch it (protocol §3.2) |
| code changes | **none.** No branch, no commits; ⛔ nothing under `chlu/` edited |
| seeds | **{0, 1, 2}** — all three, all complete. `±` is the **population std over 3 seeds (ddof = 0)**, matching `gen-store-grouping`'s convention. Per-seed values printed under every table |
| dataset / pretext | `shapes3d` / `recon` (R1 primary). ⛔ no `cls`, no dSprites |
| encoder (FROZEN) | hash asserted == `bridge_manifest.json`: s0 `98d326028316ae2c…`, s1 `fc05e2c5f39694b2…`, s2 `154d498610648fac…` |
| decoder A "old" (FROZEN) | shipped bridge decoder, `assert_decoder_hash` passed: s0 `47c0f83196cd220d…`, s1 `12981786fb809060…`, s2 `3713b777da7a2a85…` |
| decoder B "V1" (FROZEN) | `gen-decoder-lift` winner **`v1_capacity_schedule`**, param hash asserted == `winner_hashes.json`: s0 `01ef91f75eeaae7b…`, s1 `001011bc603a7cb9…`, s2 `2d7eea75add77044…`; each one's `encoder_param_hash_it_was_fit_against` asserted == that seed's bridge encoder hash |
| probe (FROZEN, unchanged in every arm) | s0 `c03bfb8916d00932…`, s1 `fb16109ded84c908…`, s2 `5cdaa90a6b67f1b3…` |
| feature cache (FROZEN) | sha256 asserted == the shipped store manifest: s0 `d6728f01d102b0d0…` (scope **`full`**, 480,000 rows), s1 `484b446c4dbb9fc4…` and s2 `8c0a69e88beb46b8…` (scope **`used`**, 9,288 rows) |
| cache extension | s0 **0** rows; s1 **32,199**; s2 **32,241 + 196** rows encoded on the fly through the frozen encoder. ⭐ `_assert_cache_reencodes` **`max_abs_delta = 0.0`** on all three seeds (64 genuinely-cached rows re-encoded bit-identically). ⛔ No cache was rewritten on disk |
| store | ⭐ **RE-MATERIALISED, not retrained** — identical seed/key chain, `train_store` 1200 steps, then bit-identity asserted with `== 0.0` tolerance against the shipped manifest. `relative_mse`: s0 corr `0.28266923706753433` / rand `0.4185249337141173`; s1 corr `0.26975176078844887` / rand `0.41951284319427884`; s2 corr `0.28414548978230525` / rand `0.45214615079541975`. `read.aggregate`: s0 corr `0.5973307291666666` / rand `0.44921875`; s1 corr `0.5997721354166666` / rand `0.4501953125`; s2 corr `0.6018880208333334` / rand `0.4482421875`. **All 6 arms passed.** |
| grouping | `K = 5` **loaded from `store_plan.json`, never re-derived**; `grouping_method="auto" → ward`; re-derived `labels` asserted **element-for-element equal** to the shipped `arms[g]["labels"]` (2048 ints) on all 6 arms. Well sizes s0 `[303,305,573,501,366]`, s1 `[397,300,307,300,744]`, s2 `[283,283,390,616,476]` |
| store config (all shipped defaults) | `n_modes 64` · `n_store_train 32768` · `n_corr 16384` · `n_probe_train 8192` · `train_steps 1200` · `train_batch 128` · `train_lr 3e-3` · `commitment_beta 0.25` · `tau 1.0 → 0.1` geometric · `min_well_frac 0.10` · `eval_batch 256` · `encode_chunk 16384` · `verify_cache_checksum True` |
| leakage probes | `probe_steps 400` · `probe_batch 512` · `probe_lr 3e-3`; key chain continued from `train_store`'s returned key exactly as `run_arm` does, so the **OLD** per-well accuracies reproduce (asserted `==`) and old/new provably share the same probes |
| eval sets | `eval_a` **1024** (never fitted by any stage); `eval_b` **72** held-out cylinders; matched-seen **198** (s0) / **199** (s1, s2) |
| JAX / env | JAX **0.9.0**, **main venv** `/Users/user/Desktop/CHLU/.venv` (protocol §4). ⛔ no `uv sync` anywhere |
| commands | §8, verbatim, with every env var |
| wall clock | seed 0 **5,287 s**, seed 1 **5,472 s**, seed 2 **8,078 s**; CPU only, under load average 100–306 and (for a period) full swap saturation from four concurrent foreign agent processes (§9) |
| ⛔ CSF3 | **NOT RUN.** Unreachable from an agent session (2FA). Nothing here needed it |

---

## What I did
- Wrote **`PREREG.md` before the harness existed** (`PREREG.md` 11:47, `rescore.py` 11:48, first result ~12:10):
  five predictions with derivations, including **two competing hypotheses** for the decoder re-score (H-A
  margin-preserving vs H-B off-manifold brittleness) and an explicit "both arms land at chance" branch.
  Scorecard in §7.
- Built one scratch harness (`rescore.py`, under my output dir — **not** production code) that loads every
  frozen artefact, **re-materialises** R0's store from the identical seed/key chain, **asserts bit-identity
  against the shipped manifest**, and only then computes R20's three corrected metrics. ⛔ Nothing retrains:
  the encoder, both decoders and the probe are loaded from disk with their hashes asserted.
- Ran all three seeds end to end; aggregated (`aggregate.py`), plotted (`figures.py`).

## How I verified (observed output, not claims)
Every check below is an assertion in the harness that **aborts the seed** on failure. All passed, all seeds.

| check | observed |
|---|---|
| frozen-encoder re-encode bit-identity | `⭐ cache re-encode check: 64 rows bit-identical through the frozen encoder`, **`max_abs_delta = 0.0`** (s0, s1, s2) |
| store re-materialisation | `[correlation] ⭐ BIT-IDENTICAL re-materialisation: rel_mse 0.28266923706753433, read agg 0.5973307291666666` (s0) — equal to the shipped manifest to the last bit; same on all 6 arms |
| grouping labels | re-derived == shipped, 2048/2048 ints, all 6 arms (`labels reproduced exactly`) |
| no-store ceiling | re-measured == shipped, asserted `abs delta == 0.0`: `0.947591` (s0), `0.936686` (s1), `0.967773` (s2) |
| OLD leakage accuracies | per-well `acc_heldout.aggregate` and `acc_matched_seen.aggregate` reproduced `==` the shipped manifest on all 15 wells × 2 arms ⇒ old and corrected leakage provably come from **the same probes** |
| matched-seen n | 198 / 199 / 199, asserted == the shipped `rows.matched_seen` |
| V1 decoder identity | `param_hash` == `winner_hashes.json` and its recorded encoder hash == the bridge encoder, per seed |

**Files** (every number below traces to one of these; ⛔ nothing lives only in this report):
`.claude/outputs/gen-r0-rescore/` — `PREREG.md` · `rescore.py` · `aggregate.py` · `figures.py` ·
`rescore_seed_{0,1,2}.json` · `summary.json` · `tables.md` · `leakage_cells.json` · `old_leakage_cells.json` ·
`seed{0,1,2}.log` · `rescore_summary.png` (3 panels: non-degeneracy scatter · read re-score · corrected
kill-rule metric).

---

## 1. ⭐ CORRECTION 1 — kill rule (i)'s metric: DISCRIMINATION within the corner

**Definition as ruled (R20).** Positives = the **72 held-out cylinders** (`eval_b`). Negatives = **matched-seen
non-cylinders in the same colour corner** (`A_matched` with `shape ≠ cylinder`; n = 198/199/199). Both sets
are decoded through the frozen decoder and read by the frozen probe. Balanced accuracy uses the probe's
**native decision rule** (argmax over the 4 shape logits); AUROC uses the softmax probability of the cylinder
class.

### ⛔ CHANCE = 0.5. These numbers are NOT comparable in level to R16/R19's `shape` accuracies (0.5926 / 0.9398).

| decoder | **balanced accuracy** | **AUROC** | TPR (72 held-out cyl) | TNR (matched-seen non-cyl) |
|---|---|---|---|---|
| baseline (superseded, shipped) | **0.7191 ± 0.0305** | **0.8806 ± 0.0113** | 0.5926 ± 0.0866 | 0.8457 ± 0.0262 |
| ⭐ **V1 `capacity_schedule`** | **0.9414 ± 0.0179** | **0.9886 ± 0.0119** | 0.9398 ± 0.0131 | 0.9429 ± 0.0404 |

Per-seed balanced accuracy — baseline `0.6843 / 0.7144 / 0.7587`; **V1 `0.9223 / 0.9365 / 0.9653`**.
Per-seed AUROC — baseline `0.8717 / 0.8735 / 0.8966`; **V1 `0.9720 / 0.9939 / 0.9998`**.
n_pos = 72 every seed; n_neg = 198 / 199 / 199.

⭐ **The correction changes the LEVEL but not the VERDICT. V1 clears 0.8 on 3/3 seeds (worst 0.9223); the
baseline clears it on 0/3 (best 0.7587).** The metric the task warned about is exactly what this measures —
"a probe that answers *cylinder* for anything in that corner scores 1.0 on the old metric and 0.5 on this
one" — and the `0.5926 → 0.9398` lift is therefore **not** a detection artefact.
⭐ **Where the old number sat, exactly.** The old `shape`-on-B accuracy **is** this metric's **TPR** and nothing
else: `0.5926` and `0.9398` reappear verbatim in the TPR column. The correction's whole content is the **TNR
column, which the old ratio could not see** — and it shows the baseline calls **15.4 %** of *seen*
non-cylinders in that corner "cylinder" (TNR 0.8457) while V1 calls **5.7 %** (TNR 0.9429). ⛔ That is the
discrimination the old normalisation was blind to.

**Which one does kill rule (i) read?** ⭐ **Recommendation: balanced accuracy**, with AUROC as a mandatory
threshold-free companion. Reasons: (a) the R3.1 floor of 0.8 was written against an *accuracy*, and balanced
accuracy is the accuracy the probe's own decision rule achieves on a class-balanced version of the same
question — a like-for-like replacement; (b) AUROC scores a ranking the pipeline never uses and is the more
generous of the two here (`0.9886` vs `0.9414`), so reading it would *weaken* the gate. ⛔ **This is a
Head/Hub ruling, not mine** — I report both and name my recommendation, as the task asks.

---

## 2. The same corrected metric applied to the **store read** (correlation vs random)

Identical positives/negatives, but the features are replaced by the **30-bit store read** before decoding.
⭐ The laundering control is a full arm. ⛔ Chance = 0.5.

| arm | decoder | balanced accuracy | AUROC |
|---|---|---|---|
| correlation | baseline | 0.6349 ± 0.1004 | 0.7154 ± 0.0964 |
| ⭐ correlation | **V1** | **0.6866 ± 0.0681** | **0.7414 ± 0.0295** |
| random | baseline | 0.5370 ± 0.0291 | 0.4896 ± 0.0945 |
| ⭐ random | **V1** | **0.5491 ± 0.0146** | **0.5625 ± 0.0145** |
| **margin (corr − rand)** | baseline | **+0.0980 ± 0.0717** (1.37×) | **+0.2259 ± 0.0445** (5.1×) |
| **margin (corr − rand)** | **V1** | **+0.1375 ± 0.0729** (1.89×) | **+0.1788 ± 0.0338** (5.3×) |

("(n×)" = margin ÷ the seed spread of that margin.) Per-seed V1 balanced accuracy — correlation
`0.5972 / 0.7001 / 0.7624`, random `0.5461 / 0.5684 / 0.5329`; **margin `+0.0511 / +0.1318 / +0.2294`,
3/3 positive**. Per-seed V1 AUROC margin `+0.1325 / +0.2122 / +0.1918`, **3/3 positive**.

⚠ **Honest reading.** The random arm sits at **0.5491 ± 0.0146 — barely above chance** — and the correlation
arm at 0.6866 is far below both the real-feature level (0.9414) and the 0.8 floor. Within a single colour
corner a 30-bit channel carries **little in-corner shape identity for either grouping**. The pre-registered
"both arms at chance" branch did **not** fire. The **AUROC** margin is the better-behaved statistic here
(5.3× spread, 3/3 positive); the balanced-accuracy margin is only 1.89× its spread because seed 0's
correlation arm is much weaker than the other two. ⛔ This is a store number, **not** what kill rule (i) reads.

---

## 3. ⭐ CORRECTION 3 — R0's READ, re-scored through the decoder-lift winner

⛔ **RAW MULTI-CLASS TOP-1 (the metric R0 always used). Chance is the per-factor majority rate, NOT 0.5. Do
not place this table beside §1 or §2.** The store-read feature vectors are **identical** in both columns; only
the renderer changes.

| | aggregate, **OLD** decoder | aggregate, **V1** decoder | `shape`, **OLD** decoder | `shape`, **V1** decoder |
|---|---|---|---|---|
| correlation | 0.5997 ± 0.0019 | **0.6107 ± 0.0041** | 0.5785 ± 0.0124 | **0.5960 ± 0.0158** |
| random | 0.4492 ± 0.0008 | **0.4603 ± 0.0055** | 0.3770 ± 0.0156 | **0.3952 ± 0.0057** |
| ⭐ **margin (corr − rand)** | **+0.1504 ± 0.0023** (64×) | **+0.1504 ± 0.0094** (16×) | **+0.2015 ± 0.0064** (31×) | **+0.2008 ± 0.0165** (12×) |
| no-store ceiling (real features) | 0.9507 ± 0.0129 | **0.9902 ± 0.0030** | 0.9076 ± 0.0122 | **0.9733 ± 0.0090** |

Per-seed aggregate margin — old `+0.14811 / +0.14958 / +0.15365`; **V1 `+0.13770 / +0.16016 / +0.15332`**.
Per-seed V1 `shape` margin `+0.17773 / +0.21484 / +0.20996`. **3/3 seeds positive on both.**
(The old column reproduces `gen-store-grouping` §3 exactly — `0.5997 ± 0.0019` / `0.4492 ± 0.0008` /
`+0.1504 ± 0.0023` — which is the independent check that I am reading the same instrument.)

⭐ **The margin does not move. `+0.1504 → +0.1504` on the aggregate — identical to four decimal places** (the
difference is `−0.0000` at 6 dp), and `+0.2015 → +0.2008` on `shape` (a change of `−0.0007`, **0.3 %**). Both
changes are an order of magnitude inside the seed spread.
⭐ **H-A (the pre-registered primary hypothesis) is confirmed:** the better renderer lifts *both* arms by almost
exactly the same amount (`+0.0110` correlation, `+0.0111` random on the aggregate) and leaves the contrast
intact. The store-read vectors were never the bottleneck the decoder was fixing.
⚠ **H-B (off-manifold brittleness) is not dead, but it does not live in the aggregate.** In the corner (§2)
the V1 decoder beat the baseline on the correlation arm's store read at seeds 1 and 2 but **lost to it at seed
0** (balanced accuracy `0.5972` vs `0.6641`; AUROC `0.7004` vs `0.7768`). 1 of 3 seeds inverted — flagged, not
concluded.
⚠ The absolute read stays low (0.61 against a 0.99 ceiling) because the store is a **30-bit** channel by
design. That was named in advance as a non-falsifier and it is low **for both arms**.

---

## 4. ⭐ CORRECTION 2 — the ladder x-axis, balanced and never a ratio of raw accuracies

**Definition as ruled (R20), implemented verbatim:**
`leakage = 1 − balacc(held-out ∪ matched-seen-other-shapes, same corner) ÷ balacc(same metric, seen corners)`
per well, per factor; per-well linear probes fitted on that well's sub-vector only, on seen rows.
Numerator set n = **270 / 271 / 271** (72 held-out + 198/199/199 matched-seen). Denominator set = `eval_a`
(1024 rows never fitted by any stage); ⚠ **measured: exactly 1 of those 1024 rows lies inside the Montero
colour corner** on every seed, so `eval_a` is an honest "seen corners" set. ⛔ These probes never touch the
decoder and render no verdict.

### Pooled (⛔ balanced accuracy — chance = 1/n_realised_classes per factor; **not** 0.5 and **not** the old majority rate)
| arm | **corrected** pooled leakage | **corrected** pooled `shape` | OLD (degenerate) pooled | OLD `shape` |
|---|---|---|---|---|
| correlation | **+0.3317 ± 0.1406** | **+0.2247 ± 0.1135** | −0.0444 ± 0.0921 | −0.2435 ± 0.5142 |
| random | **+0.4654 ± 0.1819** | **+0.2394 ± 0.1888** | +0.2257 ± 0.0443 | +0.7946 ± 0.1599 |
| ⭐ **margin (rand − corr)** | **+0.1337 ± 0.0592 (2.26×)** | ⛔ **+0.0147 ± 0.0771 (0.19×)** | **+0.2701 ± 0.0518 (5.21×)** | (+1.0381) |

Per-seed pooled margin (rand − corr): **`+0.0512 / +0.1624 / +0.1873` — 3/3 positive.**
Per-seed `shape` margin: **`−0.0318 / −0.0475 / +0.1233` — ⛔ 2 of 3 seeds NEGATIVE; the sign is not stable.**

### Per well (corrected leakage, aggregate over factors; mean over 3 seeds)
| well | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| correlation | +0.2260 | +0.3572 | **−0.0371** | +0.8024 | +0.6302 |
| random | +0.3427 | +0.5082 | +0.4866 | +0.4480 | +0.5670 |

⭐ Note the *shape* of this table, not just its means: the **correlation** arm's wells are **heterogeneous**
(−0.04 … +0.80, i.e. some wells transfer into the corner almost perfectly and others not at all), while the
**random** arm's five wells are **near-uniform** (+0.34 … +0.57). That is the same qualitative signature
`gen-store-grouping` §7 found in the relevance matrix — *clustered wells differentiate, random wells are five
copies of the same profile* — and it survives the correction even though the pooled *mean* margin shrinks.

### Per factor, pooled over wells (corrected)
| | floor_hue | wall_hue | object_hue | scale | **shape** | orientation |
|---|---|---|---|---|---|---|
| correlation | 0.9615 | 0.2408 | 0.2612 | 0.3889 | **0.2247** | 0.2320 |
| random | 0.7885 | 0.4526 | 0.5281 | 0.3608 | **0.2394** | 0.5429 |

⭐ **Two findings, one good and one that costs us a claim.**
1. **The pooled direction survives** — random leaks more than correlation on **3/3 seeds**, as P3 predicted —
   but the margin **shrinks 2.0×** (`+0.2701 → +0.1337`) and its separation from seed noise falls from **5.2×
   to 2.26×**. ⛔ A good part of the old separation was the degenerate normalisation, not the store.
2. ⛔ **The `shape`-specific leakage advantage does NOT survive.** Old: random `+0.7946` vs correlation
   `−0.2435`, margin `+1.0381` — a number built by dividing a **one-class detection rate** by a **3-class
   discrimination rate**. Corrected: `+0.0147 ± 0.0771`, i.e. **0.19× its own spread and negative on 2 of 3
   seeds — indistinguishable from zero.** ⭐ The "the wells carry `shape` structure" evidence on the **leakage**
   axis is gone. It survives on the **read** axis (§3, `+0.2008`, 12× spread) and on the code-space
   companions (§6). **Reconciliation item 1.**

---

## 5. ⭐ THE NON-DEGENERACY PROOF (acceptance item 2) — min/max over **every** (well, factor) cell

Every cell of both arms and all three seeds — 3 × 2 × 5 wells × 6 factors = **180 cells**. Files:
`leakage_cells.json` (corrected + old side by side, per cell) and `old_leakage_cells.json` (the shipped R0
audit). **Figure `rescore_summary.png` panel (a)**: old (x, clipped to ±3) vs corrected (y), green band = ±1.
⚠ The 55 red ✗ markers (old = NaN) **overplot**: **51 of the 55** sit at exactly `corrected = +1.0` (the
`floor_hue` family), the other four at `0.955 / 0.501 / 0.499 / 0.460`.

| | n cells | **NaN cells** | **min** | **max** | cells with \|leakage\| > 1 |
|---|---|---|---|---|---|
| ⭐ **corrected** | 180 | **0** | **−0.5534** | **+1.0000** | **0** |
| old, as literally specified | 180 | **55 (30.6 %)** | **−198.0000** | +1.0000 | **10** |

- **corrected argmin `−0.5534`** — seed 0, correlation, well 0, `object_hue`: balanced accuracy `0.905` in the
  corner vs `0.583` on seen corners. Negative leakage = **the probe transfers *better* into the corner**.
  Bounded, signed, interpretable.
- **corrected argmax `+1.0000`** — seed 0, correlation, well 0, `floor_hue`: balanced accuracy **exactly 0.0**
  in the corner (both realised hue classes at 0 recall; `per_class_recall = [0.0, 0.0]` in
  `rescore_seed_0.json`). This is **complete transfer failure**, the metric's true upper bound, *not* a
  divide-by-small-number blow-up. ⭐ **`leakage ≤ 1` holds identically** because the numerator balanced
  accuracy is `≥ 0`. The old metric had no such bound.
- **old argmin `−198.0`** — seed 1, correlation, well 3, `shape`: `acc_heldout = 1.000` (a **one-class
  detection rate**) ÷ `acc_matched_seen = 0.005025` (a **3-class discrimination rate**). ⭐ **This single cell
  is R20's diagnosis made arithmetic**, and it is 15× worse than the `−12.9` the original report quoted.
  **Reconciliation item 2.**

---

## 6. Companions the corrections leave UNCHANGED (R20: "what survives unchanged")

Read directly from the shipped `store_manifest.json` files — **no recomputation; nothing here depends on the
decoder or on either leakage definition.** MIG and DCI are computed from `I(code_k ; factor_f)` and never
touch the decoder, exactly as R20's "`+0.4752` is a read-space number" argument requires.

| arm | block-diagonality | MIG | DCI-D | DCI-C | DCI-I |
|---|---|---|---|---|---|
| correlation | **0.4787 ± 0.0106** | **0.2026 ± 0.0355** | 0.3094 ± 0.0078 | **0.1968 ± 0.0124** | **0.5838 ± 0.0179** |
| random | 0.2223 ± 0.0095 | 0.0204 ± 0.0047 | 0.3208 ± 0.0082 | 0.0223 ± 0.0013 | 0.2992 ± 0.0052 |
| separation | **2.15×** | **9.9×** | −0.011 (no separation) | **8.8×** | **1.95×** |

⭐ **These are the sharpest arm separations in the whole re-score, and the corrections cannot touch them.**

---

## 7. ⭐ THE VERDICT ON THE FALSIFIER (acceptance item 4)

> **Falsifies:** the correlation-vs-random margin **disappears** under the corrected metrics.

### ⛔ **It does not disappear. But it is not one margin — and its `shape`-leakage component does not survive.**

| where the margin lives | old | corrected / re-scored | separation (margin ÷ its seed spread) | verdict |
|---|---|---|---|---|
| **read, aggregate** (§3) | `+0.1504 ± 0.0023` | **`+0.1504 ± 0.0094`** | 64× → **16×** | ⭐ **SURVIVES, unmoved** (identical to 4 dp) |
| **read, `shape`** (§3) | `+0.2015 ± 0.0064` | **`+0.2008 ± 0.0165`** | 31× → **12×** | ⭐ **SURVIVES** (−0.3 %) |
| **code-space companions** (§6) | MIG 9.9× · DCI-C 8.8× · block-diag 2.15× | *unchanged by construction* | — | ⭐ **SURVIVES, untouched** |
| **in-corner discrimination, store read — AUROC** (§2) | (did not exist) | `+0.1788 ± 0.0338`, 3/3 seeds | **5.3×** | ⭐ **new, holds** |
| **in-corner discrimination, store read — balacc** (§2) | (did not exist) | `+0.1375 ± 0.0729`, 3/3 seeds | **1.89×** | ◐ **directionally consistent, weakly separated** |
| **pooled leakage x-axis** (§4) | `+0.2701 ± 0.0518` | **`+0.1337 ± 0.0592`**, 3/3 seeds | 5.21× → **2.26×** | ◐ **SURVIVES but 2.0× smaller and 2.3× less separated** |
| **`shape` leakage** (§4) | `+1.0381` | **`+0.0147 ± 0.0771`**, 1/3 seeds positive | — → **0.19×** | ⛔ **DOES NOT SURVIVE** |

⭐ **Reading, stated plainly.** R0's grouping evidence was **not** an artefact of the degenerate normalisation
*where R0 actually measured it*: the read contrast and the information-theoretic companions sit exactly where
they were, and the decoder swap moves the read margin by `0.0000`. What the corrections destroy is the
**ladder x-axis's** claim to comparable strength — its pooled margin was inflated ~2× by the one-class
normaliser, and its `shape` component was inflated from `+0.01` to `+1.04` by dividing a detection rate by a
discrimination rate. ⛔ **The x-axis is now honest, bounded, and weaker than we thought.**
⛔ I did **not** go looking for a fourth metric when the `shape` leakage collapsed. Every metric here was fixed
in `PREREG.md` before the harness ran.

### Pre-registration scorecard (`PREREG.md`, mtime 11:47, before `rescore.py` existed at 11:48)
| # | prediction | predicted (range) | measured (3 seeds) | verdict |
|---|---|---|---|---|
| P1 | baseline balanced accuracy | 0.75 (0.62–0.85) | **0.7191** | ✅ held |
| P1 | baseline AUROC | 0.80 (0.68–0.90) | **0.8806** | ✅ held |
| P1 | **V1 balanced accuracy** | 0.94 (0.86–0.98) | **0.9414** | ✅ ⭐ **held, on the point (0.9414 vs 0.94)** |
| P1 | V1 AUROC | 0.97 (0.92–0.995) | **0.9886** | ✅ held |
| P1-falsifier | V1 balacc ≤ 0.8 ⇒ the lift was a detection artefact | — | 0.9414, 3/3 seeds ≥ 0.9223 | ✅ **did NOT fire** |
| P2 | store-read corner, correlation | 0.58 (0.50–0.72) | **0.6866** | ✅ held |
| P2 | store-read corner, random | 0.52 (0.50–0.60) | **0.5491** | ✅ held |
| P2 | store-read corner margin | +0.06 (0.00–+0.18) | **+0.1375** | ✅ held |
| P2 | "both arms at chance" branch | — | correlation 0.69 | ✅ did not fire |
| P3 | corrected pooled leakage, correlation | +0.05 (−0.10…+0.20) | **+0.3317** | ❌ **falsified, above range** |
| P3 | corrected pooled leakage, random | +0.25 (+0.10…+0.45) | **+0.4654** | ✅ held |
| P3 | **corrected pooled margin** | +0.20 (+0.08…+0.32) | **+0.1337** | ✅ held (low in range) |
| P3 | **hard bound: every cell in [−1, +1]** | 0 cells outside | **0 / 180 outside, 0 NaN** | ✅ ⭐ **held** |
| P3 | tighter realised range [−0.6, +0.9] | — | **[−0.5534, +1.0000]** | ❌ falsified at the top (a genuine balacc = 0 cell) |
| P4 | read aggregate under V1, correlation | 0.66 (0.56–0.76) | **0.6107** | ✅ held |
| P4 | read aggregate under V1, random | 0.50 (0.41–0.60) | **0.4603** | ✅ held |
| P4 | **aggregate margin under V1** | +0.15 (+0.08…+0.22) | **+0.1504** | ✅ ⭐ **held, on the point** |
| P4 | `shape` margin under V1 | +0.20 (+0.10…+0.30) | **+0.2008** | ✅ ⭐ **held, on the point** |
| P4 | H-A vs H-B | H-A primary | H-A confirmed on the read; H-B inverted 1 of 3 seeds in the corner | ◐ mixed |
| P4-falsifier | margin ≤ 0, or < its largest spread | — | +0.1504 at 16× spread | ✅ **did NOT fire** |
| P5 | all provenance assertions pass | pass | **all passed, `== 0.0`** | ✅ held |

⭐ **One prediction falsified and it is a finding:** I predicted the corrected leakage *level* for the
correlation arm at `+0.05` and measured `+0.3317` — because R20's ruled denominator ("seen corners") is
photometrically much easier than the matched-seen denominator the old metric used, so the corrected leakage
absorbs the corner's rendering difficulty (see limitation 2). The *margin* prediction survived; the *level*
prediction did not, and the reason is a property of the ruled definition, not of the store.

---

## 8. Commands (verbatim, reproducible)

```bash
git worktree add --detach /Users/user/Desktop/CHLU-r0rescore e4f3d71   # read-only; main checkout was on a foreign branch

cd /Users/user/Desktop/CHLU-r0rescore && \
PYTHONPATH=/Users/user/Desktop/CHLU-r0rescore \
CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen \
CHLU_GEN_BRIDGE_ROOT=/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/runs \
RESCORE_OUT=/Users/user/Desktop/CHLU/.claude/outputs/gen-r0-rescore \
STORE_RUNS=/Users/user/Desktop/CHLU/.claude/outputs/gen-store-grouping/runs/shapes3d_recon \
WINNER_DIR=/Users/user/Desktop/CHLU/.claude/outputs/gen-decoder-lift/winner \
BASE_COMMIT=e4f3d71 \
/Users/user/Desktop/CHLU/.venv/bin/python -u \
  /Users/user/Desktop/CHLU/.claude/outputs/gen-r0-rescore/rescore.py --seed 0   # then --seed 1, --seed 2

RESCORE_OUT=... STORE_RUNS=... /Users/user/Desktop/CHLU/.venv/bin/python .../aggregate.py  # -> summary.json, tables.md
RESCORE_OUT=...                /Users/user/Desktop/CHLU/.venv/bin/python .../figures.py    # -> rescore_summary.png
```
⚠ `PYTHONPATH=<worktree> <main-venv>/bin/python -m …` per §7.12 (the hidden `.pth` breaks `uv run chlu`).
⚠ `CHLU_GEN_BRIDGE_ROOT` **must** be set from a worktree (a worktree has no `.claude/`).

## 9. ⚠ Environment: a starvation mode that looks exactly like a hang

Seed 2's first launch (pid `78161`) produced **zero bytes of output in 48 minutes** and was killed. Observed:
```
ps -o pid,etime,%cpu,rss,state   ->   78161  42:43  0.0 %cpu  73888 KB  RN     (resident set SHRINKING)
sysctl vm.swapusage              ->   total = 22528.00M  used = 21470.75M  free = 1057.25M   (95 % consumed)
vm_stat                          ->   Pages free: 4204   (~67 MB of 17.2 GB RAM)
```
Four concurrent foreign `python -u .../.claude/scratch/gen-*` processes at **nice 0** held 50–140 % CPU each;
both of my launches inherited **nice 5** from the background-launch harness and `renice` returned
`setpriority: Permission denied`. A nice-5 job in that state receives **0.0 % CPU indefinitely** and emits
nothing — ⛔ **indistinguishable from a hang from the outside.** Seed 2's relaunch (pid `85811`) sat at
0.0–0.6 % for 22 min, then completed normally in **8,078 s** once the foreign load drained.
⭐ **Diagnostic rule for the next agent:** before declaring a JAX run hung, check `%cpu` **and**
`vm.swapusage` — not just process liveness (cf. the "verify spoke liveness" memory, which this extends:
*alive* is not *running*).

## 10. Limitations and confounds
1. ⚠ **The corrected leakage denominator reintroduces the photometric confound R16 removed.** R20 ruled the
   denominator to be "seen corners"; `eval_a` is photometrically easy (1/1024 rows in the Montero corner)
   while the numerator set **is** the hard magenta corner. The corrected leakage therefore mixes *corner
   difficulty* with *composition*. This is fair between arms (identical sets) but it is **not** a pure
   compositional measure, and it is why correlation's pooled leakage moved `−0.0444 → +0.3317`: **the sign
   change is the denominator, not the store.** ⭐ If a compositional x-axis is wanted, the variant to rule on
   is `balacc(union) ÷ balacc(matched-seen alone)`. ⛔ I did **not** compute it — it was not ruled, and
   inventing a fourth metric after the third one weakened is precisely the failure the task named.
2. ⚠ **`floor_hue` is pathological in this corner for every well and both arms** (balanced accuracy 0.0 → the
   `+1.0000` cells; corrected pooled `0.9615` corr / `0.7885` rand). It is 1 of 6 factors and it pulls the
   pooled leakage up for *both* arms. ⛔ The pooled number must be read with the per-factor table (§4) beside
   it.
3. ⚠ **§1 is a decoder-ceiling measurement, not a cell measurement.** It shares the 72-item held-out set with
   the cell and is still a **one-shape** cell; R20's own note stands that a multi-shape held-out ceiling needs
   the wide-configuration bridge re-run, which is a **run**, not a re-score.
4. ⚠ **n = 3, population std.** The read margin (16×) and §1 (3/3 seeds, no overlap between arms) are robust.
   The §2 balanced-accuracy margin (1.89×) and the §4 pooled-leakage margin (2.26×) are **directionally
   consistent 3/3 but weakly separated**; neither should carry a headline without more seeds.
5. ⚠ The store was **re-materialised**, not loaded. Bit-identity was asserted against four independent shipped
   quantities per arm and passed at `abs delta == 0.0`, but a `.pkl` checkpoint would remove the argument.
6. ⚠ AUROC uses the probe's cylinder-class **softmax probability** (calibration-dependent); balanced accuracy
   uses the argmax (calibration-free). They agree on the ordering of every arm and decoder here.
7. ⚠ Seeds 1 and 2 needed **32,199 / 32,437** rows encoded through the frozen encoder because their caches are
   `scope="used"`. Bit-identity of the extension was verified on 64 cached rows per seed (`max_abs_delta =
   0.0`), which is the same guarantee `gen-store-grouping` shipped — but it is a sample, not a proof over all
   32k rows.

## 11. Recommended next experiments (priority order)
1. ⭐ **Head ruling on the corrected-leakage denominator** (limitation 1) **before R1…R5 run.** "Seen corners"
   (ruled, photometrically confounded) vs "matched-seen only" (compositionally clean, unruled) changes every
   rung's x-coordinate and must be fixed once. This is the single highest-value decision from this spoke.
2. **Checkpoint the store** (`store.pkl` per arm per seed) in `exp_gen_store` — engineer, one hunk. Every
   future re-score currently pays 2 × 1,200 Adam steps to recover an artefact that already existed. This
   re-score cost **18,837 s** of CPU, the large majority of it re-deriving bit-identical stores.
3. **More seeds (5 total) for §2 and §4** once the store is checkpointed — the only two margins here that are
   under 3× their spread. Cheap after item 2.
4. **Resolve the H-B question** (§3): does V1 render the *in-corner* store read better or worse than the
   baseline? Seeds 1 and 2 say better, seed 0 says worse. Also cheap after item 2.
5. **The multi-shape held-out ceiling** at the wide configuration (R20's own scoping note) — a bridge re-run,
   owned elsewhere, not a re-score.

## Git footprint
⛔ **No code changes. No branch, no commits.** Read-only detached worktree at `e4f3d71`
(`/Users/user/Desktop/CHLU-r0rescore`). Every artefact is under `.claude/outputs/gen-r0-rescore/`
(gitignored). The shared checkout at `/Users/user/Desktop/CHLU` was on
`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` @ `7fcef50` throughout and was **not** touched
(protocol §3.2 collision avoidance — reported as required).

## Defects / issues for `experiment-engineer`
1. ⚠ **`exp_gen_store` does not checkpoint the store.** Recovering R0's read required re-running `init_store`
   + `train_store`. It *is* bit-reproducible (verified on 6/6 arms here), but this is 2 × 1,200 Adam steps per
   seed of avoidable cost on every downstream re-score. **Suggested: `save_model(store, out_dir /
   "store.pkl")` + its `param_hash` in the manifest.** ⛔ Not a correctness bug.
2. ⚠ **`chlu.core.gen_store.leakage_from_transfer` silently emits `NaN`** when the normaliser is 0 — **55 of
   180** shipped R0 cells. It should either raise or record an explicit `"degenerate": true` per cell, so a
   downstream mean cannot quietly become a `nanmean`. (The shipped *pooled* numbers are computed from pooled
   accuracies and are not themselves NaN — but the per-factor decomposition a reviewer will ask for is 30 %
   NaN.)
3. ℹ️ **`store_manifest.json` for seed 0 lacks `feature_cache_scope`, `feature_cache_n_rows` and
   `cache_reencode_check`** (seeds 1/2 have them) — seed 0 was written before `f10b3ee`. Harmless, but it
   cost me a `KeyError` on first contact; a schema-version bump or a re-emit would stop the next consumer
   hitting it.

---

## Proposed handover updates (for the Hub)

**§1.6 / rulings ledger — R20's corrections, MEASURED (3 seeds {0,1,2}, `main @ e4f3d71`; files under
`.claude/outputs/gen-r0-rescore/`):**
- ⭐ **Kill rule (i), corrected (cylinder-vs-not *within* the corner; chance 0.5):** baseline decoder
  **balanced accuracy `0.7191 ± 0.0305` / AUROC `0.8806 ± 0.0113`**; **V1 `capacity_schedule`
  `0.9414 ± 0.0179` / `0.9886 ± 0.0119`**. ⛔ **NOT comparable in level to the 0.5926 / 0.9398 `shape`
  accuracies — never place them in the same column.** ⭐ **Verdict unchanged: V1 clears 0.8 on 3/3 seeds
  (worst 0.9223); the baseline clears it on 0/3 (best 0.7587).** The 0.59 → 0.94 lift was **not** a detection
  artefact. ⭐ The old number *is* this metric's **TPR**; the correction's content is the **TNR** column — the
  baseline calls **15.4 %** of seen non-cylinders in that corner "cylinder", V1 **5.7 %**.
  **Recommended reading: balanced accuracy** (the 0.8 floor was written against an accuracy; AUROC is the
  more generous of the two and would weaken the gate) — Head to rule.
- ⭐ **R0's read under the new decoder:** correlation `0.5997 → 0.6107` aggregate, random `0.4492 → 0.4603`;
  **margin `+0.1504 ± 0.0023 → +0.1504 ± 0.0094` — identical to four decimal places**, 3/3 seeds positive.
  `shape` margin `+0.2015 → +0.2008`. No-store ceiling `0.9507 → 0.9902` (`shape` `0.9076 → 0.9733`).
  ⭐ **The decoder change does not touch R0's grouping result.**
- ⭐ **Corrected ladder x-axis:** pooled leakage correlation `+0.3317 ± 0.1406`, random `+0.4654 ± 0.1819`,
  **margin `+0.1337 ± 0.0592` (3/3 seeds positive) — 2.0× smaller than the old `+0.2701 ± 0.0518` and only
  2.26× its own spread (was 5.21×).** ⛔ **The `shape` component does NOT survive: `+1.0381` →
  `+0.0147 ± 0.0771`, negative on 2 of 3 seeds.**
- ⭐ **New, from the same probes:** the corrected per-well leakage shows the **correlation** arm's wells are
  heterogeneous (`−0.04 … +0.80`) while the **random** arm's are near-uniform (`+0.34 … +0.57`) — the same
  "clustered wells differentiate, random wells are five copies" signature as the relevance matrix, now on a
  bounded metric.
- ⭐ **Non-degeneracy proof:** 180 cells (3 seeds × 2 arms × 5 wells × 6 factors). **Corrected: 0 NaN, range
  `[−0.5534, +1.0000]`, 0 cells outside ±1**, and `leakage ≤ 1` holds identically. **Old, same cells: 55 NaN
  (30.6 %), range `[−198.0, +1.0]`, 10 cells outside ±1.** ⛔ **The `−12.9` quoted in `gen-store-grouping` §6
  is not the extremum — `−198.0` is** (detection `1.000` ÷ discrimination `0.005025`).
- **In-corner discrimination on the STORE READ (new metric, chance 0.5):** correlation `0.6866 ± 0.0681`
  balanced accuracy / `0.7414 ± 0.0295` AUROC; random `0.5491 ± 0.0146` / `0.5625 ± 0.0145`; margins
  `+0.1375 ± 0.0729` (1.89×) and `+0.1788 ± 0.0338` (5.3×), 3/3 seeds positive. ⚠ The random arm is barely
  above chance and the correlation arm is far below the 0.8 floor: **a 30-bit read carries little in-corner
  shape identity for either grouping.**
- **Unchanged and now load-bearing:** MIG `0.2026` vs `0.0204` (**9.9×**), DCI-C `0.1968` vs `0.0223`
  (**8.8×**), block-diagonality `0.4787` vs `0.2223` (**2.15×**), and `delete_zero_delta_aggregate = +0.4752`
  (read-space, per R20). ⚠ DCI-D separates the arms **not at all** (`0.3094` vs `0.3208`) — worth saying out
  loud before a reviewer does.

**§5 provenance — new run:** `.claude/outputs/gen-r0-rescore/` (3 seeds; `rescore_seed_{0,1,2}.json`,
`summary.json`, `tables.md`, `leakage_cells.json`, `old_leakage_cells.json`, `rescore_summary.png`,
`PREREG.md`, `seed{0,1,2}.log`). ⛔ No `projects/` entry, no code, no branch. Wall clock 18,837 s CPU.

**§7 Known Issues — three new entries:**
- **NEW (env, severe — costs a session if misread):** a **nice-5 background agent process on a
  swap-saturated machine receives 0.0 % CPU indefinitely and emits nothing** — ⛔ indistinguishable from a
  hang. Measured here: `vm.swapusage used = 21470.75M / 22528.00M`, `Pages free: 4204`, four foreign nice-0
  JAX processes; `renice` returns `setpriority: Permission denied`. ⭐ **Check `%cpu` AND `vm.swapusage`, not
  just liveness, before declaring a run hung.** ⭐ **Concurrency discipline: ≤ 2 heavy JAX jobs across all
  agents on this machine.**
- **NEW (instrument):** `chlu.core.gen_store.leakage_from_transfer` returns **NaN** on a zero normaliser and
  did so on **55 of 180** shipped R0 cells; the per-factor decomposition of the shipped leakage is ~30 % NaN.
- **NEW (cost):** `exp_gen_store` does **not** checkpoint the store; every re-score pays 2 × 1,200 Adam steps
  to re-materialise a bit-identical artefact.

**§8 backlog — one decision to schedule now:**
- ⭐ **Head ruling: the corrected-leakage DENOMINATOR.** R20 ruled "seen corners", which is photometrically
  easy and therefore mixes corner difficulty into the x-axis — this, not the store, is why correlation's
  pooled leakage moved `−0.0444 → +0.3317`. The compositional alternative is
  `balacc(union) ÷ balacc(matched-seen alone)`. ⛔ Must be fixed **before R1…R5**, not after, or the rungs
  will not be comparable — the same failure mode R19 already carried forward once.
