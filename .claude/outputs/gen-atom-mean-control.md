# gen-atom-mean-control — experiment-engineer report

**Task + acceptance criterion:** build **atom-mean lookup** — `knn_proto` **restricted to the stored atoms**, at
**matched bytes, both budgets** — so the dynamical read has a like-for-like rival that stores *the same points* and
**simply does not settle**; six-factor retention ladder + read error, 3 seeds, `n_values_realised` on every row,
distinct configuration count on the 72 held-out requests, `knn_proto` reproduced bit-identically.
**Status: done.** 2 budgets × 3 seeds, **0 NOT RUN**.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⭐⭐ **`gen-dynamical-read/PREREG.md` §2's QUANTISER COLUMN IS OPTIMISTIC AND MUST BE RE-BASED BEFORE THE
>    SCORED RUN.** Its law `ε = 0.2783·(N/64)^(−2/6.503)` predicts **0.1468** at budget B; the **measured**
>    piecewise-constant floor over *centres* is **0.1695 ± 0.0021** (×1.155) and over the *atoms the landscape will
>    actually store* it is **0.2306 ± 0.0038** (k=1) / **0.1851 ± 0.0324** (best k). ⭐ At budget A the law is
>    *right* (measured centroid 0.2852 vs 0.2783, **×1.025**) — so the anchor is fine and **the EXPONENT is wrong**:
>    measured centroid slope **−0.2501 ± 0.0050** (`PR` = 8.00), ⛔ not −0.3076 (`PR` = 6.50).
>    **Owner: Hub, before `gen-dynamical-read`'s prereg is signed.** (§3, §4)
> 2. ⭐⭐ **THE BAR THE DYNAMICAL READ NOW HAS TO CLEAR, in the same units it pre-registered:** its predicted
>    aggregate errors **0.0642 (A) / 0.0163 (B)** are **5.56× / 11.36×** below the atom-mean lookup's best measured
>    read error, ⛔ not the 3.9× / 7.8× its own table implies against the centre-codebook. ⭐ **The exponent it must
>    beat is −0.2727 ± 0.0141 (`PR` 7.35), not −0.3076.** (§4)
> 3. ⚠ **`val` IS NOT EXCHANGEABLE WITH `fit`, AND EVERY GEN SPOKE SELECTS HYPERPARAMETERS ON IT.**
>    `select_baseline_rows` takes `fit = pool[:n_fit]` and `val = pool[n_fit:…]` of a **SORTED** row-index array, so
>    at production scale `val` realises **1 `floor_hue` value and 2 `wall_hue` values** against `fit`'s 10 and 10,
>    at **0.0924 feature variance vs 0.2229**. ⛔ The construction is **inherited verbatim here, not changed.**
>    Measured consequence: the two halves of `val` disagree about the winning trial on **5 of 6** (seed, budget)
>    cells. **Owner: Hub — a standing §7 entry, and a decision for every arm that tunes.** (§7)
> 4. ⚠ **The k-medoid atoms beat the random-subsample companion on read error 3/3 at BOTH budgets, but only 2/3 on
>    the 72-item in-corner statistic at each budget.** ⛔ Do not quote "k-medoids is the strong selection" off the
>    in-corner column; the direct quantity is the read error and it is unanimous. (§6)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — this is a CONTROL.** ⛔ No CLU number is produced here, no verdict, no kill rule.
- **Laundering control:** ⭐ **this spoke IS the control.** Its own falsifier is **a control that is weaker than it
  should be** — which is why the **random-subsample companion** and the **centroid control** run on every budget.
- **Falsifies:** `gen-dynamical-read`'s claim, if a plain lookup over the same atoms does the same job.
- **Does NOT falsify:** losing to `knn_proto`. ⭐ `knn_proto` decodes the whole `D`-float vector (an **unbounded**
  channel) and this arm is a **restriction** of it to a 30–150-bit read; a bounded read losing to its own unbounded
  parent is the metric-native-ceiling theorem, ⛔ not news. `knn_proto` is here **only** as the regression check.

---

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/gen-atom-mean-control`, base **`main @ 107fea9`** (unmoved during the run; rebase was a no-op) |
| commits | `0beb294` config block · `242c76a` `chlu/core/gen_atoms.py` · `162b390` harness + tests |
| worktree | `/Users/user/Desktop/CHLU-gen-atommean` (⛔ the shared checkout was on another agent's branch `agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring` and was **never touched**) |
| env | ⭐ **main venv reused, ⛔ no `uv sync` in the worktree** — `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`. `JAX_PLATFORMS=cpu`, float32. `CHLU_CLAUDE_ROOT` / `CHLU_GEN_DATA_ROOT` / `CHLU_GEN_BRIDGE_ROOT` all set (a worktree has no `.claude/`) |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero) |
| seeds | **0, 1, 2** — ⛔ none NOT RUN, at either budget |
| **budgets** | ⭐ **A = 524,288 B (64 atoms/well)** and **B = 4,194,304 B (512 atoms/well)**, both from `gen-dynamical-read` PREREG §3. **Byte equality asserted in-run on every (seed, budget, scope): `matched_exactly = True`, 12/12.** ⛔ `atoms_for_budget` **refuses** a non-lattice budget rather than flooring |
| **decoder** | ⭐ `decoder_source="winner"`, recipe **`v1_capacity_schedule` asserted by NAME and by HASH** (⛔ never the `"baseline"` default, which FAILED the gate). Hash `01ef91f75eea…` (seed 0) — **identical to `gen-cell-harness`'s** |
| **external classifier** | `gen-external-classifier`'s gated checkpoints, **loaded not fitted**, hash-asserted per seed: `c8741ac417d8…` / `aa69b436989b…` / `ad16c4e53273…` |
| store | ⛔ **the CLU store is NOT loaded and NOT used.** Only the **well partition** is taken from it (`store_plan.json` `K=5`, `n_modes=64`, `D=2048`) via `resolve_grouping(grouping_source="per_seed")`, which **reproduces the plan bit-identically on its own seed**. Well sizes `[303,305,573,501,366]` / `[397,300,307,300,744]` / `[283,283,390,616,476]` — **identical to `gen-cell-harness`'s** |
| **atoms** | ⭐ **actual TRAIN DATA POINTS from `store_train` (32,768 rows)**, ⛔ never k-means centres. Selection = **k-medoids** (Lloyd 25 iters from a k-means++ seed, then each centroid snapped to the nearest real row *inside its own cluster*), per well on that well's own channels. Empty clusters back-filled farthest-point; **exactly `N` distinct atoms asserted** |
| rows | B **72** · MS **198** · **A 512** (`eval_a`, ⛔ excluded from `store_train`) · fit **32,256** · val **512** — ⭐ `GenBaselinesConfig`'s own 32,256/512, ⛔ never my own |
| feature caches | seed 0 `scope="full"` (480,000 rows); seeds 1–2 `scope="used"`; checksum verified, re-encode check **bit-identical on 64 rows/seed** |
| tuning | **24 trials of the 36 available** — `k ∈ {1,3,5}` × metric ∈ {euclidean, cosine} × normaliser ∈ {none, zscore} × scope ∈ {per_well, global}. ⛔ Selected on **validation aggregate probe accuracy on IMAGES**; **B never enters selection** |
| randomness | ⛔ **no JAX key is derived by lengthening a split** (`jax_threefry_partitionable` makes `split(k,2)[1]` **IS** `split(k,3)[1]`). Atom streams are NumPy `default_rng(SeedSequence([seed, budget_tag, scope_tag, well]))`; a test asserts budget A and budget B produce **different** sets |
| non-default flags | ⛔ **none.** Every number is at `GenAtomMeanConfig`'s defaults; the block is new, so its defaults **are** the production configuration |
| wall clock | 214 / 452 / 468 s per seed (1,134 s total), **CPU only** |
| ⛔ CSF3 | **NOT RUN** — nothing here needs it (19 minutes of laptop CPU) |

---

## 1. ⭐⭐ What the arm is, and why it is the like-for-like rival

```
request → R8 composed per-factor prototype embedding   ← ⭐ knn_proto's OWN query, bit-identical
        → per well: the k NEAREST STORED ATOMS in that well's channels
        → their MEAN                                    ← ⛔ a lookup. It does not settle.
        → THE frozen v1_capacity_schedule decoder → image → THE gated external classifier
```

⭐ **Matched bytes falls out exactly.** A well `k` owns `d_k` channels with `Σ_k d_k = D = 2048`, so **one atom costs
`D·4 = 8,192 B` however it is sliced**, and `N` atoms/well cost `N·D·4`. Hence `N = 64` → **524,288 B**, which is
**the deployed CLU store exactly** (the store is 64 codewords *per well*, `n_modes·D·4`), and `N = 512` →
**4,194,304 B**. ⛔ Asserted in-run on every (seed, budget, scope); the run aborts otherwise.

⭐ **Channel:** `K·k·log2(N)` bits — **30.0 bits at budget A with k=1, i.e. the store's own declared width.**
⛔ Never `knn_proto`'s unbounded `D`-float channel.

**Both scopes are byte-identical and validation picks:** `per_well` (the like-for-like structure — the landscape
settles per well; the atom set is chosen independently per well, a product code) and `global` (whole-`D` lookup over
`N` whole rows, the weaker member). ⭐ **`per_well` won 6/6 (seed, budget) cells.**

---

## 2. ⭐⭐ THE TABLE — the control at both budgets, 3 seeds

Full precision and every companion column: `.claude/outputs/gen-atom-mean-control/tables.md` (regenerated from the
manifests by `table.py`; ⛔ no constant below is typed by hand).

| budget | bytes | atoms/well | byte match | channel bits | **ext in-corner `shape` balacc** | probe in-corner | ⭐ **distinct ATOM-CONFIGS on B** | distinct factor-tuples on B |
|---|---|---|---|---|---|---|---|---|
| **A** | 524,288 | 64 | ⭐ **exact** | 90 ± 49 | **0.6849 ± 0.0202** | 0.7088 ± 0.0443 | **24.3 ± 8.8** (32/12/29) | 14.3 ± 2.1 |
| **B** | 4,194,304 | 512 | ⭐ **exact** | 105 ± 42 | **0.6685 ± 0.0927** | 0.7241 ± 0.0686 | **37.3 ± 11.6** (21/47/44) | 24.3 ± 5.3 |
| *`knn_proto` (regression check — ⛔ UNBOUNDED channel)* | *475,136* | — | — | *inf* | *0.9119* | *0.8884 ± 0.0088* | — | *54.7* |
| *decoder ceiling (real features)* | — | — | — | — | *0.9357 ± 0.0193* | *0.9414 ± 0.0179* | — | — |

⭐ **For the Hub's comparison, from `gen-cell-harness` at the identical decoder/instrument/seeds:** the 30-bit CLU
read renders **9.0 ± 0.8** distinct configurations of 72 and scores **0.6568 ± 0.0226** (ext). ⭐ **The atom-mean
lookup at *the same 524,288 B and the same 30-bit class of channel* renders 24.3 ± 8.8 — 2.7× more distinct
configurations — at a statistically indistinguishable accuracy (0.6849 ± 0.0202 vs 0.6568 ± 0.0226, a gap of 0.028
against `gen-cell-harness`'s measured 0.0204 sampler-noise floor).** ⛔ I am **not** rendering a verdict on that; it
is the row the Hub asked for and it is reported beside its distinct-output count, as required.

⚠ **The in-corner statistic does NOT improve from budget A to budget B** (0.6849 → 0.6685 ext, seed-wise
0.658/0.690/0.706 → 0.551/0.676/0.778). ⛔ Do not read that as "more atoms do not help": on **every** other
instrument the budget ordering is clean — probe in-corner 0.7088 → 0.7241, the retention ladder **on all three row
sets** (§5), and the read error (§4, 43 % lower). The in-corner number is a **72-item one-class detection contrast**
whose seed spread (±0.093 at budget B) is **4.5× the 0.0204 noise floor**; it cannot resolve this.

---

## 3. ⭐⭐ THE NUMBER THE HUB NEEDS: the piecewise-constant floor, measured

Relative read error = `MSE(read, true features) / Var(true features)` — **identical in form** to
`exp_gen_variance_eq._hard_read_mse`, which is where the **0.2783** anchor came from, so the two are directly
comparable without a translation step. Measured on **`A` = `eval_a`, rows `select_store_rows` EXCLUDES from
`store_train`** — the unfitted set, the analogue of `gen-variance-equalisation`'s `SPREAD`.

| leg | budget A (N=64) | budget B (N=512) | ⭐ **fitted slope `d ln ε / d ln N`** | implied `PR` |
|---|---|---|---|---|
| ⚠ `centroid_control` k=1 *(fit-time, **0 deployed bytes**, ⛔ NOT an arm)* | **0.2852 ± 0.0048** | **0.1695 ± 0.0021** | **−0.2501 ± 0.0050** | 8.00 |
| ⭐ **`atom_mean` oracle k=1** | **0.4067 ± 0.0094** | **0.2306 ± 0.0038** | **−0.2727 ± 0.0141** | 7.35 |
| ⭐ **`atom_mean` oracle, selected k** | **0.3567 ± 0.0319** | **0.1851 ± 0.0324** | **−0.3204 ± 0.1054** | 7.09 |
| `atom_mean` **deployed** (request-addressed) k=1 | 0.4618 ± 0.0112 | 0.3299 ± 0.0053 | −0.1617 ± 0.0053 | 12.39 |
| `atom_mean` **deployed**, selected k | 0.4114 ± 0.0355 | 0.2819 ± 0.0394 | −0.1845 ± 0.0844 | 14.55 |
| *the PREREG §2 centroid law (RENDERED, ⛔ not measured here)* | *0.2783* | *0.1468* | *−0.3076* | *6.50* |

**The medoid penalty — the price of storing DATA POINTS rather than centres, at identical `N` and an identical Lloyd
fit.** ⭐ This is the quantity the whole control exists to isolate, and it is **measured, not assumed**:

| budget | **medoid ÷ centroid** | measured centroid ÷ the PREREG law |
|---|---|---|
| A | **×1.426 ± 0.016** | **×1.025** ⭐ |
| B | **×1.360 ± 0.011** | ×1.155 |

⭐⭐ **The ×1.025 is the apparatus check that makes everything else quotable:** my per-well k-means, run inside this
module on this seed's own features, reproduces the shipped store's published read error to **2.5 %**. So when the
same pipeline says the medoid costs **×1.43**, that number is the *atoms-vs-centres* trade and ⛔ not a difference
between my codebook and theirs.

---

## 4. ⭐⭐ What this does to `gen-dynamical-read`'s pre-registration

**(a) Its quantiser column is right at budget A and wrong at budget B.** Anchor ✓ (×1.025). Exponent ✗: the measured
centroid slope is **−0.2501 ± 0.0050** (`PR` **8.00**), not −0.3076 (`PR` 6.50). ⇒ the law under-states the
quantiser's error at large `N` by **15.5 %** and would flatter the landscape by the same factor.

**(b) The bar, restated in the units the landscape pre-registered** (its own §2 predictions 0.0642 / 0.0163):

| the landscape must beat | budget A | budget B |
|---|---|---|
| its own assumed quantiser (0.2535 / 0.1276) | 3.9× | 7.8× |
| ⭐ **the measured centre-codebook** (0.2852 / 0.1695) | **4.4×** | **10.4×** |
| ⭐⭐ **the atom-mean lookup over ITS OWN ATOMS, best k** (0.3567 / 0.1851) | **5.6×** | **11.4×** |
| the deployable atom-mean (request-addressed, best k) (0.4114 / 0.2819) | 6.4× | 17.3× |

**(c) ⭐ The exponent it has to beat is −0.2727 ± 0.0141, not −0.3076.** ⛔ And note the direction that matters:
**the atom lookup's exponent is STEEPER than the centre codebook's** (−0.2727 vs −0.2501, non-overlapping across 3
seeds). ⇒ ⭐⭐ **storing atoms instead of centres does NOT by itself buy a better exponent — it buys a WORSE level
(×1.43) and a slightly better slope, and the two nearly cancel.** That is exactly the confound the task said this
arm exists to close: a landscape that wins here has to win **by settling**, ⛔ not by having swapped centres for
atoms.

**(d) ⚠ A separate, load-bearing observation for the read's §7 bypass guarantee.** On the **deployed** path (query =
the R8 request embedding, ⛔ no feature tensor) the exponent collapses to **−0.162 ± 0.005** (`PR` 12.4) — the R8
embedding's own error flattens the curve by **41 %**. ⇒ ⛔ **An exponent measured with a feature-addressed query is
NOT the exponent of a request-driven read.** Both columns are in §3 and the Hub should decide *in the prereg* which
one the claim is about.

---

## 5. ⭐ Six-factor retention ladder — ⛔ every row with `n_values_realised`

Probe, 3 seeds. **`A` (512 `eval_a` rows) is the row set where the factors actually vary** and is the one the ladder
is for; `B` and `MS` are in `tables.md` §3.

**rows = `A`** (⭐ all six factors applicable)

| factor | `n_values_realised` | budget A | budget B | *`knn_proto` (unbounded)* |
|---|---|---|---|---|
| `floor_hue` | 5–6 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | *1.0000* |
| `wall_hue` | 10 | 0.9993 ± 0.0009 | 1.0000 ± 0.0000 | *1.0000* |
| `object_hue` | 10 | 0.7871 ± 0.0752 | 0.9603 ± 0.0396 | *1.0000* |
| `shape` | 4 | 0.4232 ± 0.0456 | 0.4766 ± 0.0299 | *0.6934* |
| `orientation` | 15 | 0.1862 ± 0.0357 | 0.3164 ± 0.0633 | *0.8646* |
| `scale` | 8 | 0.1270 ± 0.0248 | 0.1908 ± 0.0097 | *0.3092* |
| **aggregate** | — | **0.5871 ± 0.0221** | **0.6573 ± 0.0172** | *0.8112 ± 0.0159* |

⛔ **On `B` the `shape` row is `n_values_realised = 1` and is marked NOT APPLICABLE in the manifest, the aggregate
and this report** — it is a one-class **detection** rate (R31), ⛔ never a retention accuracy. Every aggregate row in
`tables.md` carries the per-factor realised counts beside it.

⭐ **The ladder moves the right way with bytes on all three row sets** — `A` 0.5871 → 0.6573, `B` 0.6520 → 0.6960,
`MS` 0.5422 → 0.6244 — which is the falsifier I pre-registered against myself, **passed**.
⚠ **`scale` and `orientation` are where the bounded read loses to its unbounded parent** (0.19 vs 0.31, 0.32 vs
0.86 at budget B). ⭐ That is the *same* pair the phase has been failing on since R31, reproduced here on an arm with
**no store, no landscape and no address book** — so ⛔ it cannot be attributed to the store's code.

---

## 6. ⛔ Never weaken this rival — the three things I did about it, and the one that is 2/3

1. **Row budget inherited, not chosen.** `n_fit = 32,256`, `n_val = 512` (`GenBaselinesConfig`'s). ⛔ I never set my
   own; the config docstring and a test pin it, citing the measured 0.8763 → 0.8308 cost of getting it wrong.
2. **The strongest honest atom selection ships, and the weak one runs beside it.** k-medoids vs a random subsample
   at identical bytes, every budget, every seed:

| budget | k-medoid stronger on **read error** | k-medoid stronger on in-corner | subsample in-corner |
|---|---|---|---|
| A | ⭐ **[True, True, True]** | ⚠ [True, True, **False**] | 0.6411 ± 0.0915 |
| B | ⭐ **[True, True, True]** | ⚠ [**False**, True, True] | 0.6316 ± 0.0218 |

   ⭐ **Unanimous on the direct quantity (read error), 6/6.** ⚠ On the 72-item in-corner statistic the subsample wins
   one seed at each budget — ⛔ reported, not absorbed; that column's seed spread (±0.09) is 4.5× the noise floor and
   cannot separate them. **The honest sentence is "k-medoids is the stronger selection on read error, 6/6, and the
   in-corner statistic cannot tell them apart."**
3. **The tuning grid includes the smoother.** `k ∈ {1,3,5}` is in the grid on purpose: averaging `k` atoms is already
   a zeroth-order interpolator and excluding it would have handed the landscape an easy win. It **is** selected —
   `k = 5/1/3` at budget A and `1/3/3` at budget B — and it recovers most of the medoid penalty (×1.426 → ×1.251 at
   A, ×1.360 → ×1.092 at B). ⭐ **The dynamical read must beat the smoothed lookup, not the raw nearest-atom one.**

---

## 7. ⚠ A trap I did not create, inherited verbatim, and measured

`select_baseline_rows` (shipped, and `GenCellConfig` copies it) takes `fit = pool[:n_fit]` and
`val = pool[n_fit : n_fit+n_val]` where `pool` is **SORTED**. ⛔ I did **not** change it. Production seed 0:

| row set | n | feature variance | `n_values_realised` per factor |
|---|---|---|---|
| `fit` | 32,256 | 0.2229 | floor 10, object 10, orientation 15, scale 8, shape 4, wall 10 |
| ⚠ **`val`** | 512 | **0.0924** | **floor 1**, object 10, orientation 15, scale 8, shape 4, **wall 2** |
| `A` (`eval_a`) | 512 | 0.2006 | floor 5, object 10, orientation 15, scale 8, shape 4, wall 10 |

⇒ ⭐ **`val` is a contiguous TAIL slice of the factor grid**, not an exchangeable holdout. Two consequences,
both handled rather than hidden:
- **Any variance-normalised statistic on `val` is distorted.** Measured: the same atom set's relative read error is
  **0.4067 on `A` and 0.94 on `val`** (early run), because `Var(val)` is 2.4× smaller. ⇒ the **primary** read-error
  rows in this module are **`A`**, and `val` ships as a labelled companion.
- **Selection rides that structure.** Scoring each trial on the two halves of `val` separately: the halves pick a
  different winner in **5 of 6** (seed, budget) cells (`tables.md` §7). ⚠ The *gaps* are tiny (best-to-second
  0.0007–0.0221 against a 0.09–0.15 trial spread), so the selection is sitting on a plateau rather than being
  wrong — but ⛔ **no `gen` spoke should quote a tuned configuration from this `val` as if it were identified.**

---

## 8. ⛔ The regression check — full, unrestricted `knn_proto` did NOT move (acceptance item 4)

| seed | measured probe in-corner (full precision) | published | Δ | bit-identical |
|---|---|---|---|---|
| 0 | `0.8762626262626263` | `0.8762626262626263` | **+0** | **True** |
| 1 | `0.8919597989949748` | `0.8919597989949748` | **+0** | **True** |
| 2 | `0.8969849246231156` | `0.8969849246231156` | **+0** | **True** |

⭐ **Δ = 0.0 EXACTLY, 3/3**, probe-vs-probe (⛔ never external-vs-probe, which would confound the instrument with the
arm). The guard is **hard**: `require_knn_proto_bit_identical=True` raises and aborts the seed, and
`test_knn_proto_regression_FIRES_when_the_rival_moves` shows it refusing a value perturbed by `1e-12`. The reference
constants come from `gen-arms-a7-wiring`'s manifests at full precision, and a test **refuses a constant that looks
like a 4-dp transcription**.
⭐ The external-instrument numbers also reproduce `gen-cell-harness`'s row (0.9040 / 0.9422 / 0.8894 → mean 0.9119),
and so does the decoder ceiling (0.9357 ± 0.0193 vs its 0.9357 ± 0.0193). ⇒ **nothing moved that I did not move.**

---

## 9. ⭐ The pre-registration scorecard

`PREREG.md` written **15:18:42**, **before a single line of `gen_atoms.py` or the harness existed**; the first
harness output (`smoke.log`) is **15:22:01** and the production run finished **15:44:21** — all mtime-verifiable.
⚠ **The margin is 3.3 minutes, not hours:** the prereg was written before the harness produced any number, but in
the same session. ⛔ Read it as *"committed before the first measurement"*, ⛔ not as an overnight gap. (The same
disclosure `gen-cell-harness` made at 3.5 minutes.)

| # | pre-registered | measured | verdict |
|---|---|---|---|
| **A1** | budget A oracle k=1 rel. read error **0.319** [0.25, 0.42] | **0.4067 ± 0.0094** | **HELD on the interval**, ⛔ point missed by +27 % (at the band edge) |
| **A2** | budget B **0.188** [0.14, 0.26] | **0.2306 ± 0.0038** | **HELD on the interval**, point missed by +23 % |
| **A3** | fitted exponent **−0.256** [−0.35, −0.18] | **−0.2727 ± 0.0141** | ⭐ **HELD** (point within 7 %) |
| **A3-mechanism** | the medoid penalty **GROWS** with `N` (1.147 → 1.278), so the atom exponent is **shallower** than the centroid's | penalty **SHRINKS** (**×1.426 → ×1.360**); atom exponent **−0.2727** is **steeper** than the measured centroid's **−0.2501** | ⛔⛔ **FALSIFIED — and it is the finding.** My `r_typ·n_c^(−1/PR)` nearest-point argument is a low-dimensional intuition; in `d_k` = 283–744 the nearest data point sits ≈ the cell radius away, so the penalty is set by the *within-cell* variance (→ ×√2-ish) and **falls** as cells shrink |
| **A4** | validation picks `k ≥ 3` at budget B; selected-`k` error lands **at or below** the centroid law | `k = 1/3/3` (2/3 seeds); selected-`k` **0.1851 > 0.1468** | ⛔ **SPLIT / FALSIFIED on the second clause.** Smoothing recovers **92 %** of the penalty at B and only 75 % at A |
| **A5** | the **deployed** (request-addressed) read error is budget-**INSENSITIVE**, A-vs-B < 20 % relative | **28.6 % (k=1) / 31.5 % (best k)** | ⛔ **FALSIFIED on the threshold.** ⭐ The *mechanism* held and is now measured: the embedding flattens the exponent from −0.273 to −0.162 (41 %) |
| **A6** | budget A ladder aggregate on B (probe) **0.72** [0.60, 0.83], below `knn_proto`'s 0.796 | **0.6520 ± 0.0212**, below `knn_proto`'s **0.8410** | **HELD on the interval** |
| **A7** | budget B **0.77** [0.65, 0.85], strictly between A and `knn_proto` | **0.6960 ± 0.0482**, strictly between 0.6520 and 0.8410 | ⭐ **HELD** (interval + ordering) |
| **A8** | distinct configurations on B, budget A: **34** [15, 65] | **24.3 ± 8.8** (32/12/29) | **HELD**; ⭐ **2.7× the CLU read's 9.0 ± 0.8** |
| **A9** | budget B ≥ A8, **55** [30, 72] | **37.3 ± 11.6** (21/47/44) | **HELD on the interval**; ⚠ B ≥ A on only **2/3** seeds (seed 0 inverts, 21 < 32) |
| **A10** | `knn_proto` bit-identical, 3/3 | **Δ = 0.0 EXACTLY, 3/3** | ⭐ **HELD, exactly** |
| **MY OWN FALSIFIER** | *"a control that is weaker than it should be"*: budget B must beat budget A on the ladder, and the oracle error must be within 1.5× the centroid floor | ladder **A < B on all three row sets**; oracle/measured-centroid **×1.426 (A), ×1.360 (B)** — both < 1.5 | ⭐ **PASSED.** ⚠ Against the *rendered PREREG law* rather than the measured centroid, budget B is **×1.571** — which is item 1 of the reconciliation list, ⛔ not a weak control |

---

## 10. How I verified — commands and observed output

```bash
# ⚠ smoke first (plumbing only, ⛔ never a number): 35 s, whole path, every mechanism fired
python -u -m chlu.experiments.exp_gen_atom_mean --stage all --quick --out .../smoke

# production, 3 seeds × 2 budgets, 1,134 s
python -u -m chlu.experiments.exp_gen_atom_mean --dataset shapes3d --pretext recon \
    --stage all --seed 0 --seed 1 --seed 2 \
    --out .claude/outputs/gen-atom-mean-control/runs/shapes3d_recon
  ⭐ PRECONDITION 1/2: classifier_gate.json gate=PASS   ⭐ PRECONDITION 2/2: status=complete
  ⭐ grouping re-derivation reproduces the plan BIT-IDENTICALLY on its own seed
  wells K=5 sizes=[303, 305, 573, 501, 366] (Σ=2048=D)
  ⭐ [knn_proto REGRESSION] probe in-corner 0.8762626262626263 vs published
     0.8762626262626263 -> Δ = +0, bit_identical=True
  ⭐ [A] budget 524,288 B -> N = 64 atoms/well (D=2048, an atom costs 8,192 B)
  ⭐ [A/per_well] byte ledger 524,288 B == budget 524,288 B (matched_exactly=True)
  ⭐ [A] relative read error on A (⛔ unfitted rows): oracle k=1 0.4199 · oracle k=5 0.3401
     · request k=5 0.3952 · ⚠ CENTROID control 0.2915 (medoid penalty ×1.441)
  ⭐ [B] budget 4,194,304 B -> N = 512 atoms/well
  ⭐ [B] relative read error on A: oracle k=1 0.2308 · request k=1 0.3375
     · ⚠ CENTROID control 0.1710 (medoid penalty ×1.349)
  ⚠ [A] weak companion (subsample): in-corner 0.6484 vs shipped 0.6578; oracle rel-mse
     0.5326 vs shipped 0.4199
```

**Tests** (⛔ no full suite started — R28/R30 policy, 16 GB machine; `CHLU_GEN_DATA_ROOT` set on every invocation):
```
pytest tests/test_gen_atom_mean.py                           -> 31 passed, 0 skipped, 0 failed
pytest test_gen_atom_mean test_config test_gen_cell test_gen_baselines test_gen_store
       test_gen_compose test_gen_data_splits test_gen_navigate
                                                             -> 246 passed, 0 skipped, 0 failed (3:15)
ruff check chlu/config.py chlu/core/gen_atoms.py
           chlu/experiments/exp_gen_atom_mean.py tests/test_gen_atom_mean.py
                                                             -> All checks passed!
```
⭐ **Skip count read and named (R31's fifth-strike rule): ZERO skips** — the real-data path **executed**.
⛔ `ruff format` would reformat shipped files outside my scope; the repo is not `ruff format`-clean, so I did not.

**⚠ Defects found on the way, each fixed and each reported**
1. ⛔ **My first `read_error_rows` was `val`, and the number was nonsense** (0.9385 instead of 0.42) because `val` is
   the low-variance tail slice of a sorted pool (§7). Fixed by making `A` primary and carrying `val` labelled; the
   *cause* is now a manifest diagnostic rather than a surprise.
2. ⛔ **My PREREG had no centroid control**, so the medoid penalty would have been confounded with "my k-means vs the
   shipped store". Added `AtomSet.centroids` (fit-time, **0 deployed bytes**, ⛔ never an arm) before the production
   run — and it is what produced §3's ×1.025 apparatus check and reconciliation item 1.
3. ⚠ **My stand-in test reader was not traceable under `eqx.filter_jit(jax.vmap(...))`** and the ladder test failed
   for a reason that was mine. Fixed as a **test**, ⛔ not as code — the identical defect `gen-cell-harness` recorded.
   ⭐ Recorded because a test that has never failed has never been tested.
4. ⚠ `gen_knn._kmeanspp` re-materialises an `(n, d)` difference array per centre, which at `m = 512`, `n = 32,768`
   is minutes of pure allocation. I wrote a BLAS-fused equivalent **in my own module** and ⛔ **did not touch
   `gen_knn.py`** — nothing `knn_proto` depends on moves.

---

## 11. Git footprint

| | |
|---|---|
| branch | `agent/experiment-engineer/gen-atom-mean-control` |
| base | `main @ 107fea9` (unmoved; `git rebase main` = "up to date") |
| commits | `0beb294` · `242c76a` · `162b390` |
| files | `chlu/config.py` **+196 / −1** · `chlu/core/gen_atoms.py` (new, 507) · `chlu/experiments/exp_gen_atom_mean.py` (new, 1,118) · `tests/test_gen_atom_mean.py` (new, 531) |
| ⛔ **the `−1` on `config.py` is the NO-NEWLINE-AT-EOF marker only** | the deleted line is `# --- END gen-variance-equalisation ---…`, re-added byte-identical with a trailing newline. **Content deletions: 0.** Verified: `git show 0beb294 -- chlu/config.py \| grep '^-' \| grep -v '^---'` prints exactly that one line |
| ⛔ not touched | `chlu/eval/rivals/gen_knn.py`, `gen_lace.py`, `chlu/core/gen_store.py`, `gen_compose.py`, `gen_request.py`, `gen_dream.py`, `gen_bridge.py`, `chlu/experiments/exp_gen_cell.py` (**imported read-only**), `exp_gen_baselines.py`, `chlu/cli/experiment_cmd.py`, `scripts/csf3/**` |
| config-block discipline | ⭐ **ONE contiguous `BEGIN`/`END` block at EOF with ZERO house wiring sites** (no `CHLUConfig` field, no `load_config`/`save_config` site) — reached through `gen_atom_mean_config()`, like `gen_navigate` / `gen_variance_eq`. A test asserts it. ⛔ On conflict: take the peer's whole file, re-apply by markers, assert 0 deletions |
| ⛔ no CLI hook | `chlu/cli/experiment_cmd.py` is a shared file and not in my scope; the module ships its own argparse `main()` (`python -m chlu.experiments.exp_gen_atom_mean`), as `exp_gen_cell` / `exp_gen_navigate` do. **The ~35-line hook is a Hub decision** (third module to make this call) |
| unresolved conflicts | none |
| ⛔ not pushed, no PR | branch left for Hub/Head review; the worktree `/Users/user/Desktop/CHLU-gen-atommean` is **left in place** |

**Verified from the MAIN repo** (protocol §3.2):
`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-atom-mean-control` lists all
three hashes; `git diff --numstat main..<branch>` = **2,352 insertions, 1 deletion, 4 files**.

**Artefacts** (all under `.claude/outputs/gen-atom-mean-control/`):
`PREREG.md` · `runs/shapes3d_recon/seed_{0,1,2}/atom_manifest.json` (every number, assertion, hash, byte ledger,
trial and stability record) · `runs/shapes3d_recon/seed_{0,1,2}/atoms_s{seed}_{A,B}.npz` ⭐ **the persisted atom
sets** · `runs/shapes3d_recon/atom_results.json` (3-seed aggregate) · `atom_table.png` · `tables.md` + `table.py`
(regenerates every table above from the manifests) · `run_main.log`.
Scratch: `.claude/scratch/gen-atom-mean-control/{smoke/, smoke.log, unit_atoms.py}`.

### ⭐ How `gen-dynamical-read` inherits the atoms (⛔ it must LOAD these, not re-derive them)
```python
from chlu.core.gen_atoms import load_atoms
A = load_atoms(".claude/outputs/gen-atom-mean-control/runs/shapes3d_recon/"
               f"seed_{seed}/atoms_s{seed}_{budget}.npz")
assert A.sha256() == EXPECTED[seed][budget]     # table below; assert it
assert A.deployed_bytes() == {"A": 524_288, "B": 4_194_304}[budget]
```

| seed | budget A sha256 | budget B sha256 |
|---|---|---|
| 0 | `e28c1f63a206fe1b…` | `72097d361ecbf4b1…` |
| 1 | `9fabb694388a43d4…` | `7c0cf9198720cb31…` |
| 2 | `1ab0bd6f2478f772…` | `31562d805d0bc362…` |

---

## 12. Open questions / follow-ups / risks

1. ⭐⭐ **The exponent gap between atoms and centres is small and the LEVEL gap is not.** −0.2727 vs −0.2501 (slopes)
   against ×1.43 vs ×1.36 (levels). ⇒ **at both budgets the atom lookup is *worse* than the centre codebook it
   replaces**, so "the landscape stores atoms" is, on its own, a **cost**. The landscape has to pay that ×1.4 back
   *and then* beat the centre codebook. ⛔ Worth saying in the prereg before the run rather than after.
2. ⚠ **Two budgets give a two-point slope, and my ±0.014 is a seed spread, not a fit uncertainty.** A third budget
   (e.g. `N = 128` → 1,048,576 B) would cost ~6 min/seed and turn the exponent from a chord into a regression. ⛔
   **NOT RUN**; it is the single cheapest strengthening of this control and I recommend it before the scored run.
3. ⚠ **The in-corner statistic cannot resolve this arm** (§2): ±0.093 seed spread at budget B against a 0.0204
   noise floor. The ladder on `A` (±0.017) and the read error (±0.004) can. ⛔ Do not let a 72-item one-class
   detection rate carry the comparison.
4. ⚠ **`scale` never works, on any arm here** (0.13 → 0.19 at 8× the bytes; `knn_proto` 0.31; decoder ceiling far
   above). Reproduced with **no store, no address book and no landscape** — so ⛔ it is not the store's defect, and
   a landscape that fixes `scale` would be a genuinely new fact.
5. ⚠ **One dataset, one condition, one pretext, one decoder recipe, `full` requests only.** Nothing here speaks to
   partial requests (`gen-dynamical-read` PREREG §8's secondary cell) or the wide condition.
6. ⚠ **`global` scope lost 6/6** and is therefore never the quoted arm — but it was swept, and it is in the trial
   record. ⛔ If a future spoke drops it from the grid, the "as strong as it can be" claim weakens.

---

## Proposed handover updates (for the Hub)

- **§7 NEW (row construction, standing, HIGH).** `select_baseline_rows` / `GenCellConfig` take `fit` and `val` as
  the **head and tail of a SORTED row-index array**, so `val` is **not exchangeable** with `fit`: at production
  scale it realises **1 `floor_hue` value and 2 `wall_hue` values** (vs 10 and 10) at **0.0924 feature variance
  vs 0.2229**. ⛔ Any variance-normalised statistic on `val` is distorted (measured: the same read scores 0.41 on
  `eval_a` and 0.94 on `val`), and the two halves of `val` disagree about the winning tuning trial in **5 of 6**
  cells. ⛔ The construction is **inherited, not changed** by this spoke. **Owner: Hub.**
- **§7 NEW (prereg correction, blocks `gen-dynamical-read`).** `gen-dynamical-read/PREREG.md` §2's quantiser law is
  **anchored correctly and sloped wrongly**: measured centre-codebook `PR` = **8.00 ± 0.02**, slope **−0.2501 ±
  0.0050**, ⛔ not `PR` 6.50 / −0.3076. Measured values: **0.2852 ± 0.0048 (N=64)**, **0.1695 ± 0.0021 (N=512)**.
  **Owner: Hub, before that prereg is signed.**
- **§7 NEW (the bar).** The dynamical read must beat **atom-mean lookup over its own atoms: 0.3567 ± 0.0319 (budget
  A) / 0.1851 ± 0.0324 (budget B)** on relative read error, at an exponent better than **−0.2727 ± 0.0141**, and on
  distinct configurations better than **24.3 ± 8.8 / 37.3 ± 11.6** of 72. ⛔ Its pre-registered 0.0642 / 0.0163 are
  **5.6× / 11.4×** below those, not the 3.9× / 7.8× its table implies.
- **§3 config default (⛔ unchanged, still a trap).** `GenBridgeConfig.dec_recipe` **still defaults to
  `"baseline"`** — the recipe that FAILED the gate. This is the **fifth** module to carry a private guard against
  it. Flip the default (one line + one test) or record it as permanent.
- **§3 NEW (config).** `GenAtomMeanConfig` + `gen_atom_mean_config()` at the tail of `chlu/config.py`
  (BEGIN/END-delimited, zero house wiring sites). Defaults **are** the production configuration.
- **Result to record.** Shapes3D `recon`, Montero condition, seeds {0,1,2}, `v1_capacity_schedule` decoder,
  `gen-external-classifier`'s gated instrument — **atom-mean lookup (`knn_proto`'s query restricted to the stored
  atoms), byte-matched EXACTLY at 524,288 B (64 atoms/well) and 4,194,304 B (512 atoms/well): relative read error
  on unfitted rows 0.4067 ± 0.0094 → 0.2306 ± 0.0038 (k=1) and 0.3567 ± 0.0319 → 0.1851 ± 0.0324 (best k); fitted
  exponent −0.2727 ± 0.0141 (`PR` 7.35); medoid-over-centroid penalty ×1.426 ± 0.016 → ×1.360 ± 0.011; six-factor
  retention ladder aggregate on `eval_a` 0.5871 ± 0.0221 → 0.6573 ± 0.0172; distinct atom configurations on the 72
  held-out requests 24.3 ± 8.8 → 37.3 ± 11.6; in-corner `shape` balacc (external) 0.6849 ± 0.0202 → 0.6685 ±
  0.0927.** ⭐ `knn_proto` (full, unrestricted) reproduced **bit-identically, Δ = 0.0 exactly, 3/3 seeds**.
  ⛔ **No CLU number and no dial verdict is rendered by this spoke — it is a CONTROL.**
