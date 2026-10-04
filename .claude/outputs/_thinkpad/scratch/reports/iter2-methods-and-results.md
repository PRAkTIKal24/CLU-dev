# ITERATION 2 — METHODS AND RESULTS, COMPILED
**Written by the coordinator (Opus, FORGIS_14), 2026-09-18.**
**Audience: the Mac advisor and the FORGIS_14 advisor, who will discuss this with the Head to set the
iteration-3 auto-hub plan.**

⭐ **This file is self-contained. It is written to be read without filesystem access.** Every number
below is quoted from a hub's own report or measured by me; paths are given for provenance only and
⛔ neither advisor needs to open one.

⚠ **This is the METHODS + RESULTS reference. It is deliberately longer than 5 KB.** The condensed
decision brief that goes into an advisor prompt is a separate, shorter document; this is the thing you
consult when the brief is not specific enough about what was actually run.

**Companion documents (coordinator-side, not needed to read this):** `reports/iter2-compiled.md` (the
scoreboard), `reports/refusal-closure-check.md`, `HANDOFF-coordinator.md`, `DIRECTIVE.md` (rev. 3).

---

# 0. THE SCOREBOARD, ON ONE SCREEN

| hub | object it ran | verdict | scored spend |
|---|---|---|---|
| `generation-table-2` | K=64 / 1,216 NMF atoms / shared-basis head | ⚠ **TIED** (Δ = −0.0093, m = 0.020) — banked | **1/3** |
| `composition-2` | K=32 / M=26 / rank-16 factored head | ⭐ **WIN on the pre-registered bar**, ⚠ **TIE on the generalising estimate** | **1/3** |
| `deletion-2` | the **shipped** K=5 / M=64 cell | ⚠ **banked TIE** + 2 unasked-for real results + 1 charter condition failed | **0/3** |
| `admission` | the **shipped** K=5 dynamical read, N=64 atoms/well | ⛔ **NOT SOLVED** — vacuous by algebra, exact reason | **0/3** |

**2 of 12 scored runs spent programme-wide. No bar moved. No pinned file edited. Nothing deleted.**
⛔ **Every number in this document is bridge seed 0.** Seeds 1 and 2 are unreachable on this box
(their caches are `scope="used"`; covering them is a re-encode, which is forbidden).

---

# 1. THE INSTRUMENT — shared by all four hubs, frozen, and **identical on both machines**

Everything in §1 is Mac-produced, frozen, checksum-pinned, and used unmodified by every hub here.
⛔ No hub on this box trained an encoder, a decoder or a judge.

## 1.1 Dataset and splits
- **Shapes3D**, 480,000 images, `(3, 64, 64)` uint8 RGB, scaled by 255.0. Six ground-truth factors:
  `floor_hue`, `wall_hue`, `object_hue`, `scale`, `shape`, `orientation`.
- Splits, per seed: **`train` 8,192 · `eval_a` 1,024 · `eval_b` 72**.
  - **`eval_b` (n = 72) is THE held-out combination cell** — the scored target. Disjointness from
    train is asserted, and the bridge manifest records `n_holdout_leaked: 0` of 72.
  - ⚠ **`eval_b` is entirely `shape = cylinder`.** This matters for what B can and cannot test.
- **`store_train`** — the pool the store is fitted on — is **32,768 rows** at the shipped cell scale.
- **"DEV corners"** used by `composition-2` and `generation-table-2` are Montero-**shaped** 72-row cells
  cut from *inside* the train split, constructed so they never intersect `eval_a`/`eval_b`.
  ⭐ Corners that touched `eval_a` were **refused, not trimmed** (composition-2 refused 8 + 7).

## 1.2 The encoder — and ⭐ **what "the feature" actually is**
A small conv encoder trained on the **train split only** with pretext `recon`, then **frozen and hashed**
(`encoder_param_hash 98d32602…`, seed 0).

> ⭐⭐ **Definitional point, because it has been asked and the answer is not what was assumed:**
> **in this codebase the encoder's output *is* the penultimate feature map. They are the same tensor.**
> `GenEncoder` is `[conv(k, stride) → GroupNorm → ReLU] × len(channels)`, and its docstring defines the
> **penultimate feature map as the output of the last block, post-ReLU** — "penultimate" being relative
> to the decoder/probe that consumes it, not a layer being skipped. There is no pooling step and no
> deeper head whose input is being bypassed: `gen_bridge.py` §8 states the 2048 latent dims are
> **128 feature maps × 4×4 spatial, ⛔ not pooled.**
> ⇒ **There is no Mac-vs-Windows representation difference here.** See §1.6.

- **Feature shape `(C, H, W) = (128, 4, 4)` ⇒ `D = C·H·W = 2048`.**
- ⭐ **`D = 2048` is a design constraint, not a free choice.** The LACE baseline is only valid in
  whitened coordinates and needs a `D × D` eigendecomposition, feasible only to `D ≲ 2×10⁴`. The
  manifest records `full_whitening_feasible: true`, so ⛔ LACE is **not** pushed onto channel-only
  whitening and the comparison is fair.
- **GroupNorm, not BatchNorm** — stateless, so the encoder is identical at fit time and read time,
  which is what "frozen" has to mean.

## 1.3 The feature cache — what the store is actually fitted on
| field | value |
|---|---|
| rows | **480,000** (`scope: "full"`) |
| `D` | **2048** |
| dtype | ⚠ **float16** |
| bytes | 1,966,080,128 (1.83 GiB) |
| `features.npy` sha256 | `d6728f01d102b0d0301198dc617f40bfb8ddb6c5b4a12b21a1ec08e1d974f2e2` |
| `feature_rows.npy` sha256 | `dbddda1cf6d5a73f0384365068bf7fd7d2db66d70053d4601b5c89a8eff24e6d` |
| written by | the **Mac** (`/Users/user/Desktop/CHLU/...`) |

⭐ **Written once, read by every arm, never recomputed per arm.** The re-encode prohibition exists to
protect exactly this: a hub that re-encodes a row breaks comparability with the frozen, Mac-produced
rival rows even if its own arithmetic is self-consistent.
⚠ **The cache is float16.** It is the one place a cross-ISA accumulation difference could be amplified;
it is also why the programme's parity test is "does a number we would REPORT move", not bit-identity.

## 1.4 The decoders — ⚠ **there are two, and the cell uses the second**
| decoder | params | hash | role |
|---|---|---|---|
| bridge decoder | 250,467 | `47c0f831…` | fit during bridge construction; **not** what the cell renders through |
| ⭐ **`gen-decoder-lift` winner** | **1,516,227** | ⭐ **`01ef91f7…`** | **THE decoder.** Recipe `v1_capacity_schedule`, channels [256,128,64], 1800 steps, edge-weighted MSE |

⭐ **One shared frozen decoder for every arm, forever**, fit on real `(feature, image)` pairs from the
**train split only**. ⛔ No arm may substitute its own; `assert_decoder_hash` is the check. The scored
cell manifests carry `decoder_param_hash: 01ef91f7…` on every arm — verified.

## 1.5 The judge — the external classifier
A separate frozen conv classifier (978,873 params, `param_hash c8741ac4…`) reads **decoded images** and
predicts the six factors. Its gate **PASSES** at `floor 0.95`, `mean 1.0`, `min 1.0` across 3 seeds,
with `n_holdout_images_in_training: 0` and `n_decoder_renders_in_training: 0`.
⭐ **The judge never sees a store, a code or a feature — only pixels.** "Composition-B" and "tuples" in
every table below are this classifier's readings of rendered images.

### The two scored quantities, defined
- **`composition-B` (aggregate)** — mean per-factor accuracy of the judge over the 72 `eval_b` renders.
  ⚠ One factor-accuracy step is `1/72/6 = 0.0023`, so a Δ of 0.0093 is **four images**.
- **`tuples` / `distinct_valid_novel`** — the count, of 72, of distinct *valid and novel* six-factor
  tuples realised. ⭐ This is the cross-arm comparable diversity count; raw "distinct feature vectors"
  is trivially N for a continuous sampler and is worthless.
- **Ceilings:** the frozen decoder on **real** `eval_b` features scores **0.9861** (generation-table's
  measurement) / **67 of 72 tuples** (composition's). ⛔ The ceiling is not an arm.
- **Floor:** best constant output on B = **0.5278**, 1 tuple, 0 bytes.

## 1.6 ⭐⭐ THE MAC-vs-WINDOWS QUESTION, ANSWERED WITH MEASUREMENTS
**The premise that this box works on a different representation from the Mac does not hold, and I
checked rather than assumed.**

1. **The representation is the same object by definition** — encoder output **is** the penultimate
   feature map in this codebase (§1.2).
2. ⭐ **The representation is the same *bytes*.** I re-ran `sha256sum` on the 1.83 GiB cache on this box
   today: `features.npy` → `d6728f01…` and `feature_rows.npy` → `dbddda1c…`, **both exactly the
   Mac-recorded values.** Nothing was re-encoded; the hubs read the Mac's cache.
3. ⭐ **The published comparison row reproduces.** `generation-table-2` re-ran the shipped `clu` arm here
   and got **composition-B 0.5903 / 9 tuples** against the Mac-published **0.5949 ± 0.0082 / 9.00 ± 0.82**
   — inside the published seed spread on both. The harness's own `rival_regression_check` reports
   `clu: bit_identical = true, delta = 0.0` against the Mac's `0.6609848484848484`, to every digit.
4. ⭐ 37/37 previously recorded artefact hashes reproduce on this box (spoke I, earlier round).

⇒ ⭐⭐ **Windows numbers enter the corpus. Parity is settled empirically, not argued.**

⚠ **The two genuine cross-machine residuals, stated so neither advisor over-reads point 3:**
- ⛔ **`train_store`'s `loss_final` differs from the Mac's by +1.27 %** while the channel partition is
  **ISA-invariant** (dead/live 71/1977, the rejected `average` attempt, the accepted `ward` attempt and
  the final `well_sizes` all identical, `loss_first` bit-identical). ⇒ ⭐ **byte-exact deletion
  provenance is a WITHIN-machine property**, and `deletion-2` says so explicitly. ⛔ Not a parity claim.
- A **rebuilt** k-means++ `store_hash` moves under pure re-batching *on one machine*, so it is
  over-specified as an identity; `well_sizes`, `codes_sum` and the byte ledger stay invariant.

⛔⛔ **The hard constraint is unchanged regardless:** a Windows hub must never mix Mac-produced and
Windows-produced quantities inside one paired comparison. Every comparison below is internally
same-machine.

⚠ **What I could NOT verify, and am flagging rather than guessing:** if the Mac programme has a
*different* line — outside the generative phase-1 cell — in which a CLU reads a pretrained classifier's
penultimate features, I have no visibility of it from this box and nothing above speaks to it.
**If that is what the question was about, tell me and I will scope it.** What I *can* say is that
inside this bundle there are two unrelated store families and they are easy to conflate — §2.1.

---

# 2. WHAT "THE CLU" MEANS HERE — ⭐ and it is **not one object** across the four hubs

## 2.1 ⚠ Two unrelated store families live in this codebase. Do not conflate them.
| | `chlu.core.factored_store.FactoredStore` | `chlu.core.gen_store.GenStore` |
|---|---|---|
| what it is | the **physics** store — an `AtomDictionaryPotential` landscape settled by damped Verlet | a **factored quantiser over cached encoder channels** |
| bound to | `CatTestConfig`, the synthetic set-code family, tier-ii's "cat test" | the generative phase-1 cell |
| needs | per-well **anchors** and an item→well incidence | neither |
| used by iteration 2? | ⛔ **no** | ⭐ **yes — all four hubs** |

⭐ `gen_store.py` states the reuse decision explicitly: forcing the physics store onto a channel
partition of a frozen feature cache "would have meant **inventing anchors, i.e. inventing the answer**".
What *is* reused is `multiwell_read.soft_occupancy` (⛔ no argmax in the differentiable path) and the
byte accounting from `FactoredStore.n_bytes`.

## 2.2 The generative store, defined
> The `D = 2048` coordinates of the frozen encoder's penultimate map are partitioned into **`K` wells**.
> Each well `k` owns `M` learned **modes** in `R^{d_k}`. A read picks one mode per well and
> concatenates. **The read channel is exactly `K·log2(M)` bits wide, and every one of those bits is a
> pointer into the store.**

- ⛔ **Standing convention: no well is ever named semantically.** Wells carry integer indices, in the
  code, the tests, the artefacts and the figures.
- ⭐ **Structural isolation is the point, not hygiene.** `read_from_codes(store, codes)` **does not take
  the features as an argument at all**. `assert_no_bypass` checks this by signature *and* numerically —
  perturb the features arbitrarily while holding the codes, and the output must be bit-identical.
  ⭐ The programme's standing negative result is that an earlier block *did* route around the memory
  (deleting the whole store changed the loss by < 0.001 bits), which is why the check exists.
- **Labels enter in exactly one place: the addressing head.** The modes are fitted on activations; the
  landscape, the write and the read never see a label (charter §B.6).

## 2.3 ⚠ Two partition functions, and this is where "K = 32" vs "K = 5" comes from
| function | behaviour | wells on the shipped cache |
|---|---|---|
| `gen_store.correlation_grouping` — **the SHIPPED one, unchanged** | cuts the ward dendrogram at K clusters; returns **unequal** wells | ⭐ **K = 5**, `well_channels [303, 305, 573, 501, 366]`, 71 dead channels |
| `gen_partition.balanced_correlation_partition` — **NEW on `base/iter2`, additive** | same ward linkage on the same \|corr\| matrix, but cuts the **dendrogram leaf order** into equal contiguous blocks | any K, equal `d_k` |

⭐ Both keep the correlation structure (same linkage, same input, no new criterion, ⛔ no label, request,
held-out row or arm outcome). The balanced version exists because with equal `d_k` the byte identity
`store_bytes = M·D·4` holds **per well**, and because a K = 64 store becomes one batched op instead of
64 unrolled scatters.

⚠⚠ **This resolves a live confusion.** `hub/admission` asked in its report: *"`K = 32` does not exist on
this object — where does K = 32 come from?"* **Answer: from `gen_partition`, which admission's task did
not put on its path.** Admission correctly ran the shipped K = 5 plan, and re-derived it bit-identically.
⛔ Nobody was wrong; the addendum was ambiguous. **Fix this wording in the iteration-3 directive.**

## 2.4 ⭐⭐ THE FOUR READ VERBS — this is the single most important table in this document
The store is the same *kind* of object throughout. **What changed between hubs is where the read's
coefficients come from**, and that changes the result by more than any hyperparameter does.

| verb | coefficients `a` come from | used by |
|---|---|---|
| ⛔ **`clu` (shipped, ARGMAX)** | **argmax** of a naive-Bayes posterior — one **vertex** of the simplex over each well's modes | the shipped cell; `deletion-2`; every published rival row |
| ⭐ **`clu_linear`** | a **real-valued, request-addressed linear** fit over the modes — the **span**, not a vertex | `generation-table-2` (as a control), `composition-2`, `generation-table-2`'s arm |
| **discriminative address book** | the **same parameter tensor and the same byte ledger** as the shipped head, but fitted by maximum conditional likelihood instead of naive-Bayes counting | available on `base/iter2`; naive Bayes is a closed-form setting of these weights, so `steps=0` returns it bit-identically |
| **dynamical read** | settle a state in the landscape; gate on free energy | `admission` |

⭐⭐ **The measurement that motivated the change** (`hub/generation-table` DEV-08), and it is the cleanest
result the programme has:

| what the modes are allowed to do | composition-B |
|---|---|
| nearest mode (a **vertex**) | 0.4884 |
| best **convex** combination (the simplex) | 0.5509 |
| ⭐ best **LINEAR** combination (the **span**) | ⭐ **0.8264** |
| *(`knn_proto`, for scale)* | *0.7801* |

> ⭐⭐ **The modes already span the answer. The shipped read was restricted to a corner of that span.**

## 2.5 The head, and why the store earns its bytes
`a(request) = Φ_add(request)·C_add + Φ_pair(request)·E·U`
- `Φ_add` — bias + one-hot per factor. ⭐ **This is `knn_proto`'s hypothesis class.**
- `Φ_pair` — one-hot per **selected factor pair**. ⭐ **This is the part `knn_proto` cannot afford.**
- `C_add` `(n_add, K, M)` — ⛔ deployed at **full rank**; DEV-15 measured that factorising it destroys it
  (it *is* `knn_proto`, and `knn_proto` is not low-rank).
- `E, U` `(n_pair, r)` and `(r, K, M)` — the interaction block at rank `r`.

`head_bytes = (n_add·K·M + n_pair·r + r·K·M)·4` · `store_bytes = M·D·4`

⭐⭐ **The byte argument, as arithmetic:** a pairwise request model costs `n_pair·D·4` in raw feature
space — **11.3 MB on Shapes3D, 24× the ceiling arm's entire table.** In the store's basis it costs
`n_pair·r + r·K·M`. **The store compresses D = 2048 to K·M, and that compression is what makes a
non-additive request model affordable at this budget.** `knn_proto` cannot do this: its prototype table
*is* its head, so widening its hypothesis class widens its ledger.

## 2.6 The rival, and the byte rule
- **`knn_proto`** is the ceiling rival: per-value prototypes, **475,136 B**, and ⚠ **its whole `D`-float
  vector reaches the decoder — an unbounded channel against the CLU's 30 bits.**
- ⭐ **Every arm is compared at matched-or-fewer bytes.** The ledger is asserted live against the
  pre-registration; `generation-table-2` added a hard abort after a dry run caught an arm being built
  474,336 B → 408,576 B with the interaction block silently absent.
- ⚠ **Ledger convention:** `store_bytes` and the **addressing head** are reported **beside** each other,
  never merged. A comparison that prices one arm's head and not another's is wrong — `generation-table-2`
  caught exactly that and fixed both rows to `store + address_book = 598,528 B`.

## 2.7 ⭐ THE FOUR OBJECTS ACTUALLY RUN — side by side
| | `generation-table-2` | `composition-2` | `deletion-2` | `admission` |
|---|---|---|---|---|
| partition | **balanced**, K = **64**, 32 ch/well | **balanced**, K = **32** | ⭐ **shipped**, K = **5** | ⭐ **shipped**, K = **5** |
| modes/atoms | **1,216** NMF atoms (M = 19/well), label-free | **M = 26**/well | **M = 64**/well | **N = 64**/well (rung A); 512 (rung B) |
| head | shared-basis: `C_add (52×1216)` + `E (220×52)`, `U := C_add` | rank-**16** factored, ridge 10.0 | shipped argmax address book | — (gate, not a head) |
| pairs | floor_hue×scale, object_hue×scale, shape×orientation | floor_hue×scale, object_hue×scale | — | — |
| read | `clu_wide_vector` (uniform alloc ⇒ the **shipped verb**) | `clu_wide_vector` (LINEAR) | shipped **argmax** | **dynamical**, free-energy gate |
| bytes | store 155,648 + head 298,688 = **454,336** (0.956×) | **469,504** (0.9881×) | store **524,288** + head 74,240 | **524,288** (A) / **4,194,304** (B) |
| channel bits | — | — | **30.0** (= 5·log₂64) | — |

⛔⛔ **Neither advisor should write "the CLU" in the iteration-3 directive without saying which of these
four it means.** Three different partitions, three different mode counts and two different read verbs
produced the four results below.

---

# 3. WHAT EACH HUB ACTUALLY RAN

All four: branch off **`base/iter2` = `945a3e9`** (= pin tree `345244bb` + four additive core modules +
one test + a 14-line `exp_gen_cell.py` gate patch; ⛔ `ULP_BOUND`/`FRAC_BOUND` untouched), machine
FORGIS_14 (Windows 11 AMD64, **local CPU only**, 4 threads, jax 0.9.0, numpy 2.4.1, Python 3.11.16),
seed 0, ⛔ no GPU, ⛔ no GCloud spend, CSF3 **NOT RUN** (2FA).

## 3.1 `hub/generation-table-2` — ⚠ TIED, banked. Scored 1/3.
**The headline target. Bar: beat `knn_proto` on composition-B at matched bytes under R1 (tie band
m = 0.020).**

**Selection discipline, declared before the first sweep:** 6 SELECT corners for search, **43 CONFIRM
corners never looked at during selection**, and the bar computed in advance — *a WIN at n = 43 needs
Δ > +0.018*. SELECT said +0.038 and the hub wrote down that SELECT is biased upward by an unknown
amount. ⛔ **It reported the tie it had pre-registered.**

**What was run, in order:** EXP-R1 lever isolation (6 corners × 48 units) → EXP-R2 atoms-vs-rank
frontier (20 configs × 6 corners = 120 units) → EXP-R4 shared-basis head → a **disqualification screen
run before spending ~2 h** (K = 128 disqualified, K = 64 clears §C.2) → NMF vs PCA → **config frozen
13:10Z** → CONFIRM (10 rows × 43 untouched corners = 430 units) → **the scored run** → EXP-R5, R7, R8.

**The scored run, seed 0, `eval_b`, one sitting, two legs:**
- **leg 1** = `exp_gen_cell --stage cell --seed 0` on the **unedited pinned harness** (814 s)
- **leg 2** = the hub's own `run_arm.py` (277 s), because `exp_gen_cell.ARMS` does not contain the new
  object and ⛔ the hub did not edit the pinned harness
- ⭐ **Leg 2 was debugged on a DRY RUN first** — `RUN_ARM_DRY=1` substitutes two Montero-shaped
  train-split corners asserted to intersect `eval_a`/`eval_b` in **zero** rows. **It found two real bugs
  that would have corrupted the scored run** (the rank-0 head, and the mispriced ledger).
- ⭐ Cache coverage checked first: **32,903 / 32,903 rows cached, `shortfall = 0`** ⇒ the run provably
  cannot extend the cache. The script raises `SystemExit` if that is ever untrue.
- ⭐ Store **LOADED** from `store_s0.pkl`, ⛔ never re-trained; grouping re-derivation reproduces the
  store plan **bit-identically**; `assert_no_bypass` passes.

**Results (`eval_b`, 72 images, seed 0):**

| row | composition-B | tuples/72 | bytes | × knn |
|---|---|---|---|---|
| frozen decoder ceiling (real features) | 0.9861 | — | — | — |
| ⚠ `atoms_kpp` — §C.1 adversarial control | **0.8634** | 46 | 454,336 | 0.956 |
| ⭐ **`knn_proto` — the rival** | **0.8588** | **57** | 475,136 | 1.000 |
| ⭐⭐ **THE ARM** | ⭐ **0.8495** | **51** | **454,336** | **0.956** |
| ⚠ `atoms_medoid` — §C.1 sharp control | **0.8495** | 46 | 454,336 | 0.956 |
| ⛔ `pca_affine` — R4 control | 0.8472 | 53 | 454,336 | 0.956 |
| same arm, k-means atoms | 0.8310 | 47 | 454,336 | 0.956 |
| ⭐⭐ **`clu_linear` — shipped store, only the READ VERB changed** | ⭐ **0.7917** | ⭐ **27** | 598,528 | — |
| D-matched Gaussian `random` null | 0.6944 | 44 | 454,336 | 0.956 |
| ⛔ **`clu` — the shipped 30-bit ARGMAX read** | **0.5903** | ⛔ **9** | 598,528 | — |
| best constant output | 0.5278 | 1 | 0 | — |
| §C.2 `store_removed` on the arm | 0.3657 | 11 | 454,336 | 0.956 |
| `ebm_compose` | 0.2315 | 72 | ⛔ 14,226,660 | ⛔ 29.9 |
| `store_removed`, shipped argmax path | 0.1111 | 1 | 524,288 | — |

> **Δ(arm − `knn_proto`) = −0.0093 at 0.956× the budget. |Δ| ≤ m = 0.020 ⇒ ⭐ TIE under R1, BANKED.**
> ⛔ Not a win, and the hub does not call it one.

- ⭐ **The powered estimate agrees:** the 42-corner DEV CONFIRM gives **Δ = −0.0131, CI [−0.0243, −0.0019]**
  — same sign, same magnitude, both inside the tie band.
- ⭐ **Where the deficit lives:** three hues are **1.0000 vs 1.0000**. The entire gap is **`shape`
  (0.9306 vs 0.9861) = four images.** ⭐ The arm **beats** the rival on `scale` (0.1806 vs 0.1667) — the
  one factor both are near a 1/3 floor on, and the factor the charter records the codebook as losing
  90.5 % of.
- ⭐⭐ **THE RESULT THAT IS BIGGER THAN THE TIE — and it got bigger on held-out data:**
  **changing only the read verb on the shipped store, at a byte-identical ledger, moves
  `0.5903 / 9 tuples` → `0.7917 / 27 tuples`: Δ = +0.2014, 9 → 27.** Confirmed at **43/43** corners and
  on `eval_b`. ⚠ On DEV this was +0.1192 (9.5 → 35.0) — **the effect is larger on the held-out cell.**
- ⭐ **Wiring validated:** leg 1's `knn_proto` `0.8588 / 57` and leg 2's independent recomputation
  `0.8588 / 57`; leg 1's `clu` `0.5903 / 9` and leg 2's independent reimplementation `0.5903 / 9`.
  ⇒ both legs measure the same object through the same instrument.
- ⭐ **§C.4 not-a-lookup:** distance-to-nearest-atom / atom spread = **3.382** on the scored arm.
- **Pre-registered predictions: 8 of 10 hit**, including *"I predict a TIE, not a WIN"* in writing.

**⛔ The two misses, both reported by the hub against itself:**
1. ⛔⛔ **§C.1 FAILS on the scored cell.** `atoms_medoid` ties the arm **exactly (Δ = 0.0000)** and
   `atoms_kpp` **beats** it by 0.0139. The hub predicted +0.01…+0.05 over medoid. **This is the
   important miss** and the hub says so.
2. ⚠ `pca_affine` landed *below* the arm (+0.0023), which the hub flagged **because it favours the arm**.

**⚠ Also reported, not absorbed:** the harness flagged **`lace` as regressed** — 0.6136 measured vs
0.7077 published, Δ −0.0941 against a 0.05 tolerance. Cause: `GenCellConfig.lace_selection_source` still
points at the broken-split LACE selection. ⛔ Not the hub's file to fix; it does not touch the
arm-vs-`knn_proto` comparison.

**⭐ Five of its own errors recorded as its own**, including an 18-minute thread-budget breach (which
incidentally produced a free determinism check) and hitting the documented import trap on the scored run.
**EXP-R5 tested the hub's own best idea and refuted it; EXP-R7's 6-corner call was reversed by the hub
on 43 corners.**

## 3.2 `hub/composition-2` — ⭐ WIN on the bar, ⚠ TIE in expectation. Scored 1/3.
**Bar (R2): distinct valid novel tuples on B at matched bytes, win-or-tie vs `knn_proto` under R1, with
the capacity law and the decoder ceiling reported beside it.**

**The object was pinned in `PREREG-object.json`, read by the scoring script so the PREREG and the run
cannot drift:** K = 32, M = 26, rank 16, ridge 10.0, pairs `floor_hue×scale` + `object_hue×scale`,
read `gen_compose.clu_wide_vector`, head `gen_factored_read.fit_factored_head`, **469,504 B =
0.9881 × `knn_proto`**.

**Results (`eval_b`, seed 0):**

| | tuples_valid_novel / 72 |
|---|---|
| ⛔ decoder ceiling (not an arm) | **67** |
| ⭐⭐ **the object — `clu_linear`** | **42** |
| ⚠ `knn_proto` | **12** |
| §C.1 matched-bytes lookup (feature medoids / factor medoids) | **0 / 0** |
| §C.2 landscape removed (zeroed / shuffled) | **0 / 0** |
| constant-output floor | **0** |

> **Δ = +30.00 tuples, paired 95 % CI [+20.27, +39.73], McNemar exact p = 2.27 × 10⁻⁷ ⇒ WIN.**
> ⭐ **Independently replicated** by a §A.5 spoke barred from the hub's code, own build, own code path:
> **25 vs 12, Δ = +13.00 [+3.88, +22.12] — also a win.**

### ⛔⛔ The three numbers that must travel with it — the hub's own insistence
1. ⭐⭐ **Over 105 held-out-shaped corners in two location-disjoint families: Δ = +1.09,
   CI [−1.52, +3.69] — a TIE.** That is the generalising estimate and ⛔ **it is not a tie for lack of
   power** (CI ±2.6). The two families agree to **0.24 tuples** (+0.96 over 49, +1.20 over 56); three
   store seeds agree to **0.9**.
2. ⛔ **The B LEVEL does not replicate — 42 (hub) vs 25 (spoke)** — because the PREREG failed to pin the
   store fit (it pinned ridge to one decimal and left the equal-well partition, the k-means++ RNG stream
   and the Lloyd budget free). ⭐ **The hub refuses to quote "42" as reproducible.** The *verdict* is.
3. **The win is one cell of 72 items**, and the hub's own framing is *"the measurement of how little that
   single-cell win can bear."*

### ⭐⭐ §2.4 — where all the variance lives (the widest-reaching result in iteration 2)
Three store seeds × 49 corners, variance of the paired Δ decomposed:

| source | sd (tuples) | share |
|---|---|---|
| store-seed k-means init (within corner) | 4.09 | 9.1 % |
| ⭐⭐ **corner (between corners)** | ⭐ **12.92** | ⭐ **90.9 %** |

**Every disagreement the hub met has the same single cause:** selection half +5.92 vs validation half
−3.80 · the hub's 42 vs the spoke's 25 (the two builds **tie** over 49 corners, Δ = −0.53 [−1.94, +0.88])
· raw-modes 36 vs Lloyd 25 (tie over 49, Δ = +1.04) · **B's +30 vs DEV's +1.40 (z = +2.17).**

⭐ **Every alternative explanation was tested and failed:** the partition (tie) · the Lloyd budget (tie)
· the store seed (9 % of variance) · ⛔ **and the shipped Adam `train_store` fitter, which the
replication spoke named as the most likely cause and which moves the WRONG way** (c0: 38 → 33;
c2: 35 → 28; mean −6.0). ⛔ Not the cell's difficulty either — the stratum of DEV corners with
ceiling ≥ 65, i.e. as easy as B, still ties (Δ = +1.79). ⛔ Not grid position — slope runs the wrong way.
⭐ **It is which 72 rows you look at.**

> ⚠⚠ **This bears on every hub whose bar is a single held-out cell, not only this one.** The hub flagged
> it for the coordinator as a programme-level observation. **It applies directly to §3.1's tie as well.**

### ⭐⭐ §2.2 — the store is used as a SIGNED BASIS, not a mixture
The head's coefficients are **~47 % negative**, carry **L1 ≈ 8.5 per well** (a probability vector carries
1.0), and **0.00 % of wells — any request, any corner — are a probability vector.** The output lies
**8.5 L1-units outside the convex hull of its own atoms**.
⭐ **That is why it is not a lookup** (§C.4), and the target tuple is absent from every stored row by
construction, so ⛔ no lookup can be valid at all — measured: every matched-bytes lookup scores **0.00 on
all 49 corners**. ⭐ **The pair-interaction block is load-bearing: removing it costs −6.55 tuples
[−9.23, −3.87] and turns the arm into a loss.**

### ⛔ §2.3 — DREAMING IS DEAD, with a mechanism
Both mechanisms, on the pre-registered grid `{0.0, 0.005, 0.05, 0.5}`, ⛔ never widened:
- **State space** (pinned `DreamLandscape` + `dream_rollout`): **0 at every β and every temperature**,
  every corner. ⭐ **Control C proves the chain MOVED** — `|Δx|/|x|` rises 0.134 → 0.254 → **0.718**, up
  to **46 % of wells change code**. ⇒ not a frozen chain: the dynamics travel and the read returns
  nothing. ⭐ Strictly more than iteration 1 could establish.
- **Coefficient space:** at **β = 0 — which is what dreaming IS — 0 at every temperature**, while
  producing 72/72 *distinct* tuples (⭐ distinctness is free and worthless). At β = 1 temperature only
  destroys: 38 → 35 → 34.7 → 0.7.
- ⭐⭐ **Why: every energy-driven state lies INSIDE the mode hull; the object lives OUTSIDE it.**
  ⛔ Not fixed by temperature, τ (three decades), more steps, or a softer read — all measured.
- ⚠ **A defect in the hub's own T = 0 leg**, found by reading the pinned code *after* the run and
  reported rather than silently re-run: `_mala_step`'s acceptance ratio is `0/0` at `temp = 0`, so every
  proposal is rejected. The corrected unadjusted leg moves 0.12 and still reads 0.
- ⛔⛔ **Consequence the hub does not hide:** charter §C.3 lets a TIE become the win narrative only if the
  same object delivers refusal, exact deletion, **or** dreaming with valid novel tuples.
  **Dreaming does not deliver it.** ⇒ **that clause must be carried by `deletion` or by a successor to
  `refusal`, on this same object, or not at all.**

### ⭐ §2.5 — the two companions R2 required, and one it could not have anticipated
| read | capacity law | R² |
|---|---|---|
| ⛔ inherited **ARGMAX** quantiser (iteration 1) | `rel-err = e^1.346 · bytes^−0.1651` | 0.9960 |
| ⭐ **LINEAR read (this object)**, 12 corners, 23× byte span | `rel-err = e^0.244 · bytes^−0.1030` | 0.9082 |

⭐ The linear read's residual prefactor is **≈ 3× smaller at every byte count** (1.28 vs 3.84).
⭐⭐ **That is the quantitative reason the same store renders 42/72 under this read and 0 under argmax**,
and it is why iteration 1's **"77 GB / 147,000×" figure is ⛔ NOT a bound on this object.**

⭐⭐ **And the tuple curve SATURATES** (K, pairs, rank, ridge fixed; only M moves; 12 corners):

| × `knn_proto` budget | 0.17 | 0.32 | 0.51 | ⭐ **0.99** | 1.96 | 3.89 |
|---|---|---|---|---|---|---|
| tuples / 72 | 0.25 | 1.67 | 3.83 | ⭐ **23.67** | **36.92** | **37.00** |

⭐ **Doubling the store from 2× to 4× budget buys +0.08 tuples.** ⇒ above ~2× budget the store is **not**
the binding constraint — **the addressing head is.** The shipped budget buys **64 %** of everything
capacity can buy and sits at the knee. ⚠ Per corner the curve is not monotone — reconstruction and
composition come apart above the knee (one corner: 35 → 18 → 8 tuples while its rel-err keeps falling).
⛔ Reported as measured; the hub has no mechanism for that split and did not invent one.

**Scale of work:** ~1,800 DEV renders across **105 corners** in two families, 3 store seeds, 10 units,
⛔ none touching B. C04 swept 45 head designs × rank × ridge (360 renders); C07 swept 5 store geometries,
single-peaked in K: 8.25 (K=8) · 11.12 (16) · ⭐ **22.50 (32)** · 21.62 (64) · 20.12 (128) ⇒ ⭐ the
reference object sits at the optimum of its own family, which keeps §C.3's "same object" clause intact.

**⚠ Two operational faults, both self-reported:** a sub-agent left an orphaned 21 GB process (identified
by command line and stopped); and **C12 completed only 2 of 6 corners** after Adam training grew to
**50.3 GB resident** and drove this shared four-hub box to **0.4 GB free of 63.4 GB** — launched without
a memory or per-corner wall-clock cap. ⭐ Resumability meant the kill cost one corner, not the run.

## 3.3 `hub/deletion-2` — ⚠ banked TIE + two results nobody asked for. Scored **0/3**.
**Ran the SHIPPED cell**: K = 5, M = 64, D = 2048, `n_live` = 32,768, 1200 steps, batch 128, store
**524,288 B**; rival `knn_proto` **475,136 B** (⚠ **the rival is the smaller model, 90.6 %**).
`delete(v) := rebuild(live ∖ rows(v))`.

**⭐ The blocking gate — PASSED, stronger than required.** `train_store` is **deterministic at shipped
cell scale**: **three independent OS processes, three different launch paths, 22/22 arrays bitwise
identical**, `store_digest d0eacf3b…`, across a **1.31× wall-clock spread** (554.9 / 617.9 / 728.3 s) —
which rules out the audit's top-ranked risk (BLAS thread count flipping a greedy ward merge). A fourth
confirmation through the whole eval path: `eval_a` predictions agree at exactly **1.000000**.

**Headline (pre-registered): a TIE, banked.** Churn-corrected collateral
Δ = `[C_clu(del) − C_clu(plc)] − [C_knn(del) − C_knn(plc)]` over six deletions:
floor_hue +0.01162 · wall_hue +0.01514 · object_hue −0.01957 · scale −0.00195 · shape +0.01074 ·
orientation +0.00216. **Mean +0.003023, sd 0.012774, bootstrap CI [−0.00714, +0.01121], m = 0.020,
0 of 6 outside the margin.** ⚠ The **uncorrected** Δ is −0.0033; both are reported.

### ⭐⭐ §3 — canonical rebuild is NOT answer-preserving (unasked-for, non-by-construction)
| rung | comparison | n_live | ARI | CLU churn | CLU κ | knn churn |
|---|---|---|---|---|---|---|
| 1 | identical live set | 32,768 | **1.0000** | **0.0000** | 1.0000 | 0.0000 |
| 2 | **same n_live, different arbitrary rows** (4 replicates, 6 pairs) | 24,534 | 0.681 ± 0.082 | **0.1883 ± 0.0084** | 0.597 ± 0.030 | **0.0140 ± 0.0005** |
| 3 | semantic deletion vs undeleted | 32,768/24,534 | 0.410 | 0.2122 | 0.5434 | 0.0644 |

> **Change nothing about a canonical store except which arbitrary 8,234 of 32,768 rows you drop, rebuild
> by the ported mechanism, and 18.8 % ± 0.8 % of its answers change — while a byte-matched prototype
> table under the identical operation changes 1.40 % ± 0.05 %. A 13.4× ratio.**

Rung 1 proves this is not build noise. **Rung 3 adds almost nothing over rung 2 ⇒ essentially all the
churn is recanonicalisation; the semantic content of the deletion is second-order.**

⭐⭐⭐ **And there is no small-deletion regime.** 16 semantically-empty rebuilds, 7 distinct sizes,
replicates at 4, spanning `n_deleted` **852 → 8,234** (2.6 % → 25.1 % of the pool, a **9.66×** range):

| | r(n_deleted) | max/min over the 9.7× span |
|---|---|---|
| **CLU** | **+0.016** | **1.110×** |
| `knn_proto` | +0.989 | 3.589× |

⭐ At four banded sizes CLU churn is 0.1929 / 0.1910 / 0.1920 / 0.1931 — **a 9.7× change in rows removed
producing a 1.011× change in answer churn.** ⇒ **the behavioural cost of a deletion does not shrink with
its size: deleting 852 rows costs essentially what deleting 8,234 costs.** For a deployed system, the
price of honouring one erasure request is the price of a full recanonicalisation, every time.

⚠ **Stated precisely, because a looser version would over-claim:** between-size sd 0.00694 against a
pooled within-size replicate sd of 0.00239 (2.90×), so this is **"does not scale with n"**, ⛔ **not "is
constant"**; the variation is non-monotone and driven by one single-replicate point. ⚠ Smallest measured
is 852 rows; ⛔ the hub does not extrapolate to a single row.

⭐ **Why this matters to the literature:** *Retention You Can Predict, Scope and Delete* prices the
mechanism honestly in **time** (O(n) per delete) and guarantees **byte-identity with a store that never
held the item** — this hub confirms both exactly. **It makes no claim about behavioural stability**, and
this is the measurement showing why that gap matters. ⛔ The table is immune only because it has no
layout to recanonicalise — the same property that makes it unable to forget a conjunction.

### ⭐⭐ §4 — the one genuine WIN: the CLU forgets a conjunction; the table cannot
Delete `(object_hue = 5 ∧ shape = 2)`, 852 rows. **Both parts survive individually and neither addressing
term is emptied** (`cond_2[5]` and `cond_4[2]` both still non-uniform), so the request stays fully
addressable. On the 1,024 requests naming both:

| | joint render rate | placebo control | retained |
|---|---|---|---|
| CLU | 0.3672 → **0.0107** | 0.3037 | ⭐ **2.9 %** |
| `knn_proto` | 0.9092 → **0.5488** | 0.9053 | ⛔ **60.4 %** |

⭐ **The table re-composes the deleted conjunction from parts it still holds** — its `oh=5` prototype is
still fitted on hue-5 rows of other shapes, its `shape=2` prototype on shape-2 rows of other hues. **The
CLU's store was retrained with no `(oh5, shape2)` row present, so no mode encodes the combination and the
composed code has nothing to point at.** Collateral cost: Δ = −0.0058, a tie ⇒ **forgotten at no
differential cost.** ⚠ One conjunction, one seed, and the CLU held it weakly to begin with.

### ⛔ §5 — charter condition 1 (unrecoverable) is NOT uniformly met, and the failure is the useful part
- `object_hue`: **met** — 0.000000 of 4,096 dev renders show hue 5 afterwards. ⛔ But `knn_proto` achieves
  it strictly better (1.0000 → 0.0000 vs 0.5107 → 0.0049), by construction.
- ⛔ **`orientation`: FAILED BY BOTH ARMS.** The median-index rule selects `orientation = 7` = exactly
  **0.0°, the null pose**. `cond_5[7]` is bit-exactly uniform — and the CLU still renders orientation 7
  on **25.0 %** of requests naming it and **12.6 %** of all dev renders (bar: 0.117). The rival fails
  worse (0.5977).
- ⛔ **`scale`: ALSO FAILED BY BOTH**, more starkly — `knn_proto`'s accuracy on the deleted value goes
  **UP** (0.8096 → 0.8252) and it still renders `scale = 4` on **54.5 %** of all dev requests; the CLU on
  23.6 %. `scale` index 4 = **1.0357, the default object size**.

⭐⭐ **Because a blurry reconstruction looks unrotated, the value is what the model degrades *towards*.**
⇒ **The mechanism guarantees byte-identity with a store that never held `v`; it does NOT guarantee that
no request can elicit `v`. Any generative method with a default output cannot delete that default by
removing rows, however canonical the rebuild.**

⭐⭐ **The failures are predictable in advance and separate perfectly 4–2:** `scale` and `orientation` are
exactly the two factors whose median index is a **physical null** (default size, null pose). The four
without one — three hues and shape — **all PASS in both arms**. ⇒ **which deletions will fail condition 1
is identifiable from the factor's geometry before any deletion is run.** ⚠ Perfect separation on n = 6,
one seed, drawn post-hoc — **a hypothesis, ⛔ not a law.**

### ⛔ §7 — why the scored run was NOT spent (0 of 3, on any machine)
1. ⭐ **The outcome is a foregone tie.** Pooled B half-width ≈ **0.01664** against m = 0.020, with DEV's
   effect at +0.0030 ± 0.0058. A loss needs a true effect ≤ −0.037, a win ≥ +0.037 — **both ~6 DEV
   standard errors away.** A run whose verdict is determined in advance buys no information.
2. ⭐ **B cannot test §4**, the one genuine win: **B is entirely `shape = cylinder`; the conjunction is
   `shape = 2`.**
3. ⚠ **Validity, not power:** pre-deletion on B the CLU is at exactly **1.0000 on all three hues**
   (collateral can only register as damage) and at 0.0417 `scale` / 0.0000 `orientation`, both *below*
   the fully-ablated store's 0.3333, against a best-constant floor of 0.5278.

⚠ The hub first argued B lacked power and **withdrew that**: measured pairwise correlation between the
six deletions on B's shared 72 rows is **r = −0.0045**, so pooling does reach adequate power. The
decision rests on 1–3. ⭐ **Recommendation: preserve all three for the 3-seed final.**

**Cost:** rebuild per delete **501–728 s** `unit_wall_s`, ⚠ plus **95–283 s of JAX teardown excluded from
that figure** — true process wall is what a delete actually costs (`det_B` 1011 s). **28 of 28 planned
units complete. Elapsed 6 h 32 m of 12 h.**

**⛔ §9 — four of the hub's own registered predictions were refuted and are listed**, including
⛔⛔ *"over-generalising from one deletion"*, which it names its worst error and retracts in full.
**⛔ §10 — one result reported as unexplained and not explained away:** deleting all 3,269
`floor_hue = 5` rows left the channel partition **bit-identical** (0 of 2048 channels moved) while its
own size-matched placebo moved 982 and every other deletion moved 864–1,246. Verified not an artefact.

## 3.4 `hub/admission` — ⛔ NOT SOLVED, dead on arrival, exact reason. Scored **0/3**, `eval_b` never read.
**Object: the SHIPPED dynamical read**, `shapes3d`/`recon`, seed 0, K = 5 wells (plan re-derived
bit-identically: `well_channels [303, 305, 573, 501, 366]`, ward, 71 dead channels), **rung A: N = 64
atoms/well, 524,288 B**; **rung B: N = 512, 4,194,304 B** as the capacity control.

**The gate, pre-registered and never moved:** *refuse well k iff `E_k(x_k) > 0`* on the kernel free
energy. ⛔ §7 forbids moving it off zero, scaling `σ_k`, or replacing the level set with a quantile.

**⭐ The gate is correctly derived, implemented and certified:**
- the identity `max_i E_k(a_ki) ≤ 0` holds on the live store **with 30 units to spare**;
- the per-well reduction reproduces the shipped `store_energy` to **1.2 × 10⁻⁷**;
- **certified fraction of refusals = 0.6936 (rung A), 0.7264 (rung B)** via the `σ_k√(2 ln N)` bound.

### ⛔⛔ It is nevertheless vacuous, and the reason is one table
| radius, in the gate's own units `d/σ_k` | well 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| rivals' **fitted** `q = 0.95` | 1.150 | 1.138 | 1.053 | 1.122 | 1.285 |
| ⭐ **the derived ZERO LEVEL SET** (measured, 256 rays/well) | **2.093** | **2.043** | **2.419** | **2.343** | **1.892** |
| the certified bound `σ_k√(2 ln N)` | 2.884 | 2.884 | 2.884 | 2.884 | 2.884 |

| population | n | mean `min_k d/σ_k` | mean `max_k E_k` | refuse |
|---|---|---|---|---|
| stored atoms | 64 | 0.000 | −42.25 | 0.0000 |
| ⭐ **real features, all `store_train`** | 4,096 | **0.787** | −38.10 | 0.0007 |
| ⭐ **real features, `val` rows** | 512 | **0.793** | −37.81 | 0.0000 |
| CLU launch `clamp_atom` (shipped) | 512 | **0.000** | −41.81 | 0.0000 |
| CLU launch `posterior_mean` | 512 | 0.814 | −54.34 | 0.0000 |
| R8 affine, `out_of_range` w = 2 → 13 | 3,072 ea. | 0.856 → **2.262** | −41.0 → **+793** | 0.0000 → 0.7337 |

> ⭐⭐ **The data lives at 0.79 σ. The gate fires at 2.0–2.4 σ. The fitted rival fires at 1.05–1.29 σ,
> just above the data. The derived gate's radius is ~2.6× further out than the thing it guards.**

### ⭐⭐ THE STRUCTURAL REASON — and it is a statement about the **bandwidth rule**, not about admission
> `σ_k` is `nn_median`, **the median nearest-neighbour distance between the atoms**. The atoms **are
> train rows**. So they tile their own manifold at spacing ≈ `σ_k` **by construction**, and any real
> point is ≈ 1 `σ_k` from one of them **whatever the store contains**.
> ⛔⛔ **The bandwidth rule and the signal it is meant to separate are the same number.** The zero level
> set can only ever sit at a fixed multiple of the data's own radius, **whatever `N` or the byte budget**.

**The capacity control confirms it:** 8× the atoms and 8× the bytes (rung B) moved `refuse_oor` by
**+0.028**; the level set moved *further out* (1.87–2.42 → 2.44–2.89 σ); real features stayed at ~0.75 σ.
⭐ **~+0.09 in `d/σ` per 8× store**, against a level set that recedes faster.

**Consequences, exact:**
- at the shipped **`clamp_atom`** launch the gate is **vacuous by algebra** — the launch point *is* a
  stored atom, so `E_k ≤ 0` always;
- at **`posterior_mean`** it is vacuous because a convex combination of atoms cannot leave their hull;
- only a **linear extrapolation** (the R8 affine head at severity `w ≥ 3`) ever reaches it.

### ⭐ The live-set claim — the one a threshold cannot fake — fails completely
Deleting every atom carrying a value changes the decision on real images of that value by **+0.0000, on
all six factors.** The same statistic at a **fitted** radius does detect it (factored kNN: 1.0000 vs
0.3359 on `floor_hue`). ⇒ ⛔ **the problem is not the signal, it is the radius — and §7 rightly forbids
moving it, because moving it is the entire claim.**

⚠ **A correction the hub made against its own artefact:** `LIVESET_RUNGB_seed0.json` carries
`"capacity_rescues_liveset": true`. **That flag is misleading** — it tests the §5.9 falsifier threshold
(separation ≥ 0.05), ⛔ not the §5 bar (`refuse ≥ 0.90`). The artefact was left unaltered per §0.1 and
corrected in prose. At rung B `wall_hue` reaches **+0.1719** — genuine, not noise, but **misses the bar
by 0.73, on one factor of two, for 8× the bytes**, while the rival reaches **1.0000 at both rungs for no
extra bytes**; and `floor_hue`, the *better*-localised factor, stays at exactly **0.0000 at both rungs**.

### The pre-registered pair, all six controls (DEV / `val`; ⛔ B untouched)
`score = ½(refuse_out_of_support + accept)`, with `accept_A7` beside `accept` per the programme-wide rule.

| arm | `accept_val` | `accept_A7` | `refuse_oor` | **score** | threshold |
|---|---|---|---|---|---|
| **`CLU_gate@r8`** — the arm | 1.0000 | 0.8056 | 0.3714 | **0.6857** | ⭐ **DERIVED, no free parameter** |
| `CLU_K1@r8` — control (c), same bytes | 1.0000 | 1.0000 | 0.2000 | 0.6000 | DERIVED |
| `knn_whole` — control (a) | 0.9512 | 0.6667 | 0.6213 | 0.7862 | FITTED `q = 0.95` |
| **`knn_factored`** — control (b), the fair rival | 0.7891 | 0.3889 | 0.8590 | **0.8241** | FITTED `q = 0.95` |
| `margin_M` — control (e), physics-free | 0.9590 | 0.9722 | 0.0398 | 0.4994 | FITTED `q = 0.95` |

⭐ **The honest characterisation:** the derived gate **never falsely refuses a valid request, in any mode
the hub could construct** (val 1.0000 · unseen interpolated values 1.0000 · A7 deduplicated 1.0000 ·
**free mode with 1/2/3 factors unnamed: 1.0000 / 1.0000 / 1.0000**, where `knn_factored` collapses to
0.5488 / 0.3633 / 0.2168) — **and it refuses only 0.2653 of the invalid ones.** Same cause: the radius
sits outside everything. ⛔ **On the balanced pre-registered score the arm loses at every operating
point, against both rivals**, and the hub does not offer the above as a rescue.

### ⭐⭐ FOUR MECHANISM FINDINGS FOR RE-IDEATION — the most valuable output of this hub
1. ⭐⭐ **The `logsumexp` is costing separation.** Operating-point-free, same store, same wells:

   | statistic | best score | AUROC |
   |---|---|---|
   | CLU `max_k E_k` — the free energy (the arm) | 0.7994 | 0.8619 |
   | ⭐ CLU `max_k d_k/σ_k` — **plain nearest-atom distance** | ⭐ **0.8488** | ⭐ **0.9107** |
   | `knn_factored` (fitted rival, at its best) | 0.8621 | 0.9198 |

   **The CLU's own distance beats the free energy built from it, and lands within 0.014 AUROC of the
   fitted rival.** ⭐ **Loss decomposition: of the arm's 0.177 deficit, 0.114 is the RADIUS and 0.063 is
   the SIGNAL.**
2. ⭐⭐ **The settled read pushes itself off the store — a defect in the READ, not the gate.**
   `total_energy = E_store + β·Σ_t(E_t − E_store)`, so at β = 1 with `T` requested terms **the store's own
   free energy enters with weight `1 − T`** — for a six-factor Shapes3D request that is **−5**.
   ⛔ **Settling pushes the state AWAY from the atoms by construction.** Measured on valid `val`:
   `mean max_k E_k` rises **−36.9 → −24.7** and `min_k d/σ_k` **0.000 → 0.849** over 1000 steps.
3. ⚠ **The settled read does not depend on its launch point** — flat at **0.3507** across a 6× sweep of
   launch radius at 1000 steps, with the request-off control at chance. ⇒ **gating the launch cannot gate
   the output**, so the claim's property *"one object, two outputs"* has no support on this rung.
4. ⭐ **A concrete, one-location, already-fixable defect:** the A7 false-refusal is R8's **double-count**,
   and `deduplicate_terms` is already in the codebase. It explains the published rivals' A7 rate that the
   §7b reporting rule was written for, and fixing it **improves every arm's A7 row**.

**⚠ The factor→well map is partial by construction:** six factors onto five wells cannot be injective,
and only three distinct argmax wells are used. **Only `floor_hue` is singly-assigned** (well 4, MI 0.9462);
⛔ **`scale` is barely carried anywhere** (max 0.1115). ⇒ exemplar mode is well-posed for `floor_hue`,
workable for `object_hue`/`shape`, ⛔ ill-posed for `scale` and `orientation`.

**⭐ Adversarial verification by a second Opus agent in float64 overturned nothing** and produced two
corrections **against** the arm plus one bug of the hub's own (a §5.6 check that was vacuous, reported
plainly). Confirmed independently: level set 2.090 / 2.038 / 2.410 / 2.342 / 1.901 · manifold 0.7852 /
0.7934 · `σ` to 4.5e-08 · float16 vs float32 identical to 16 digits · ⛔ no sign, `√` or `2σ²` error.

**⚠ Elapsed 1.04 h of 12 h, and the hub explains why it stopped:** every deliverable delivered, all five
falsifiers answered with numbers at both rungs, all six controls run, result adversarially re-derived,
and every escape hatch closed with a measurement. **The only remaining moves are the ones §7 forbids.**
⭐ The one thing 11 more hours could buy — a seed-1/2 replication — is unreachable on this box.
⭐ **The deadline is a cap, not a quota.**

---

# 4. CROSS-CUTTING FACTS — true across hubs

1. ⭐ **Parity settled empirically** (§1.6). Windows numbers enter the corpus.
2. ⭐ **The re-encode ruling held.** No hub re-encoded; no shortfall > 0 was hit; the ULP gate stayed at 1.
3. ⛔ **Everything is seed 0.** Composition's "3 store seeds" is **store**-seed spread and is ⛔ never
   quoted as bridge-seed spread.
4. ⭐ **`accept_A7` beside `accept_B`** was adopted programme-wide mid-iteration; `admission` is the only
   hub it bound.
5. ⚠⚠ **The single-cell warning generalises.** Composition measured that **the corner is 90.9 % of the
   variance in Δ**. Both scored results in iteration 2 are single 72-item cells. **Composition's +30 sits
   at z = +2.17 of a distribution sampled 105 times; generation-table's −0.0093 is four images.**
   ⇒ **Neither scored cell can distinguish "won/lost this cell" from "ties in expectation."**
6. ⭐ **Two hubs independently priced the same wall from opposite sides:** composition's saturation curve
   says **above ~2× budget the store is not the binding constraint — the addressing head is**; admission's
   capacity control says **8× the store buys +0.028 on `refuse_oor`**. ⭐ **More bytes is not the lever.**
7. ⭐ **The read verb is the biggest measured effect in the iteration** (+0.2014, 9 → 27 tuples, at a
   byte-identical ledger, confirmed 43/43) — and **it is a change to a pinned default, not to the object.**

## 4.1 ⚠ A CORRECTION I OWE, to my own predecessor's closure check
`reports/refusal-closure-check.md` reports the rung-B well-matrix ranks as
**236 / 305 / 512 / 479 / 366 of 512** and concludes *"four of five wells are rank-deficient"* and
*"a chunk of the extra atoms is linearly dependent."* **That reading is too strong, and I checked the
arithmetic rather than repeating it.**

`atoms[k]` is `(N, d_k)`, so `rank ≤ min(N, d_k)`. With `N = 512` and the shipped
`well_channels [303, 305, 573, 501, 366]`, the **maximum attainable** rank per well is
`[303, 305, 512, 501, 366]`. Measured: `[236, 305, 512, 479, 366]`.

⇒ **Wells 1, 2 and 4 are at the maximum rank possible. Only wells 0 and 3 are genuinely deficient**
(by 67 and 22 respectively).

⭐ **The underlying point survives and gets sharper, which is why this is worth correcting rather than
dropping:** in **three of five wells the ceiling is dimensional, not degenerate** — 512 atoms cannot buy
more than `d_k` independent directions no matter how many you add. ⇒ **"8× atoms" was never 8× capacity
in those wells, by linear algebra.** ⭐ This is the same shape of argument as admission's bandwidth
finding — **a quantity that looks like a free capacity knob is pinned by the geometry of the object** —
and the two should be considered together in re-ideation.
⛔ **This is arithmetic on an existing measurement, not a new run.** ⛔ Nothing was re-run to produce it.

## 4.2 The `refusal` closure, for completeness
`hub/refusal` is **RETIRED** (DIRECTIVE rev. 3 §1a), scored **0/3 across both iterations**, target
re-posed as `hub/admission`. The closure check that justified retirement:
⭐ **the §C.2 atom shuffle moves the address-book posterior by `max_abs(P_full − P_shuffled) = 0.0`
exactly, at both rungs, over 16 shuffles**, taken over `log_prior` and every factor's `log_cond`.
⇒ **a symbolic request's posterior is identical under the control, so no function of the posterior can be
certified by that shuffle.** ⚠ `reports/refusal-closure.md` — the full closure compilation — is
**still unwritten**; I am writing it next.

---

# 5. THE SCORED-RUN LEDGER

| target | spent | remaining | spent on what |
|---|---|---|---|
| generation-table | **1** | 2 | `eval_b`, seed 0, two legs, one sitting |
| composition | **1** | 2 | `eval_b`, seed 0 |
| deletion | **0** | 3 | ⛔ deliberately not spent — verdict determined in advance |
| admission (inherits refusal's 3) | **0** | 3 | ⛔ never reached the bar; `eval_b` never read |
| **total** | **2** | **10** | |

⛔⛔ **The budget reading in force is the STRICTER one: three per TARGET, across every attempt and every
machine.** The Advisor has not yet ruled; until they do, the strict reading binds. ⚠ **Two machines
pursuing one target cannot see each other's count** — a scored run spent here is one the Mac cannot have,
and it cannot know it was spent. **This is one of the open decisions in §7.**

---

# 6. LIMITS THAT BOUND EVERY NUMBER ABOVE

1. ⛔ **Seed 0 only**, everywhere, on every hub.
2. ⚠ **Two scored cells of 72 items each**, in a regime where the corner explains 90.9 % of the variance.
3. ⚠ **`eval_b` is entirely `shape = cylinder`** — it cannot test a `shape`-indexed claim (this is why
   deletion's §4 win is untestable on B).
4. ⚠ The feature cache is **float16**.
5. ⚠ **`loss_final` differs from the Mac by +1.27 %** ⇒ byte-exact deletion provenance is within-machine.
6. ⚠ **`lace` is flagged `regressed: true`** by the harness (0.6136 vs 0.7077 published) because
   `GenCellConfig.lace_selection_source` still points at the broken-split selection. ⛔ Not fixed here.
7. ⛔ `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` is **QUARANTINED** — the superseded
   broken-split file. ⛔ Never cite it. ⚠ It is manifest-pinned, so ⛔ do not move or delete it either.
8. ⚠ **Composition's B level (42) is not a reproducible number** — the PREREG under-specified the store
   fit. The *verdict* replicates; the *level* does not.
9. ⚠ Composition's C12 column is **n = 2 of 6 corners** and supports only a narrow claim.
10. ⚠ Deletion's condition-1 geometry rule is a **post-hoc hypothesis on n = 6, one seed** — ⛔ not a law.

---

# 7. WHAT I BELIEVE IS OPEN — for the two advisors and the Head

⭐ These are stated as questions, not recommendations. The coordinator does not decide any of them.

1. ⛔⛔ **PROCESS, and it should be settled first.** **Two autonomous agents executed `hub/deletion-2`
   concurrently**, sharing one worktree and one output directory, writing the same tag names under
   `runs/`, with `LOG.md` interleaving both. The hub filed the incident, noted that *"a log two writers
   append to is not a reliable record of either"*, and correctly carried on rather than escalating. The
   final report is authored by one session; the other's in-progress skeleton is preserved; ⭐ **nothing
   was deleted.** **This is a coordination failure at the launch level and needs a fix before iteration 3.**
2. ⭐⭐ **What may the composition result claim?** A WIN on the pre-registered bar (+30.00, p = 2.27e-7,
   independently replicated) **and** a TIE on the generalising estimate (+1.09, CI [−1.52, +3.69], 105
   corners, **not** underpowered) **and** a level that does not replicate (42 vs 25). All three are the
   hub's own disclosures. ⛔ **What the paper may say is a Head/Advisor call.**
3. ⭐⭐ **Is admission's finding really about admission?** §3.4's structural result is a statement about
   the **bandwidth rule** (`σ_k = nn_median` between atoms; atoms are train rows), and §4.1's rank
   correction is the same shape of argument. **Both say a quantity that looks like a free knob is pinned
   by the object's geometry.** They may bear on the read, on the landscape and on the earlier bandwidth
   defect. **My own view: these are one finding, and it may be larger than any of the four hub results.**
4. ⭐ **What happens to the admission target?** 3/3 scored runs unspent, a gate vacuous by algebra at the
   shipped launch point, and **four concrete mechanism findings** (§3.4) that point at the *read* rather
   than the gate. Re-pose a third time, retire it as `refusal` was, or fold it into (3)?
5. ⚠ **Seed 0 only** is the largest single exposure in the corpus and interacts with (2). Seeds 1–2 need
   full-scope caches from the Mac; ⛔ they cannot be produced here without a re-encode.
6. ⚠ **The scored-run budget reading is still unruled** (§5). Worth closing rather than carrying a third
   iteration.
7. ⭐ **Do deletion's two unasked-for results get promoted to pre-registered scored claims in
   iteration 3?** §3 (canonical rebuild is not answer-preserving, and does not scale with deletion size)
   and §4 (the CLU forgets a conjunction the byte-matched table re-composes). Both real, neither
   by construction, neither pre-registered. ⚠ **§4 cannot be tested on `eval_b` at all** (limit 3).
8. ⚠ **Charter §C.3's "a tie is the win narrative" clause now has one leg left.** Composition established
   that **dreaming is DEAD with a mechanism**, so the clause must be carried by deletion or by a
   successor to refusal, on the same object — **or not at all.**
9. ⚠ **A wording fix for the next directive, cheap:** say which `K` and which partition function are
   meant (§2.3). `hub/admission` lost time to an addendum that named K = 32 on an object whose shipped
   plan gives K = 5.

---

# 8. VERIFICATION STATUS — what I checked myself, and what I did not

⭐ **The loop requires the coordinator to verify headline claims before compiling, and to flag anything
unverified so the advisor treats the flag as the fact.** Stating that honestly:

**Verified by me, on this box, today:**
- ⭐ `features.npy` and `feature_rows.npy` sha256 — **both match the Mac-recorded values exactly** (§1.6).
- ⭐ The encoder/decoder/judge definitions, hashes and configs, read from the pinned manifests (§1.2–1.5).
- ⭐ The two-decoder fact, and that the scored cell manifests carry the lift-winner hash `01ef91f7…`.
- ⭐ The four hubs' object definitions, read from the code and the pinned PREREG objects (§2, §3).
- ⭐ `base/iter2`'s exact content (4 additive core modules + 1 test + a 14-line harness patch).
- ⭐ The K = 5 / K = 32 discrepancy — traced to two partition functions and resolved (§2.3).
- ⭐ The rung-B rank arithmetic — **a correction to an earlier coordinator check** (§4.1).
- ⭐ Machine state: `Get-Process python*` returns nothing. **No orphaned processes; nothing still running.**

**⚠ Taken from the hubs' own reports and NOT independently re-derived by me:**
- every **statistical** quantity — the CIs, the McNemar p, the bootstrap intervals, the variance
  decomposition, the churn ratios, the AUROCs, the capacity-law fits;
- the determinism gate's 22/22 bitwise-identical arrays;
- the radius atlas and level-set measurements.

⭐ **Mitigating this, and worth weighing:** three of the four hubs carried **independent replication by an
agent barred from the hub's own code** — composition's §A.5 spoke (which **refuted** the hub's headline
level and was right), admission's float64 adversarial verifier (which produced **two corrections against
the arm** and found the hub's own vacuous check), and generation-table's two-leg cross-validation
(independent reimplementations of `knn_proto` and `clu` agreeing to machine precision).
⚠ **`deletion-2` is the exception** — its numbers were re-derived from raw artefacts by the reporting
session using its own code, but by an agent **inside** the hub, and it is the hub that had the
concurrency incident. ⛔ **If one hub's numbers warrant an independent check before iteration 3, it is
that one.**

**⛔ I have re-run no scored comparison and will not** — re-running one would spend a scored run.
