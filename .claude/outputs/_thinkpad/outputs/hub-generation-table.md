# hub-generation-table — autonomous-hub report

**Task + acceptance criterion:** show the store is not decorative on the generation table, then beat the
known numbers and the rivals — composition-B aggregate, external classifier, shipped cell, 3 seeds, matched
declared bytes; WIN or TIE against `knn_proto` 0.8650 ± 0.0122 (`m = 0.020`).
**Status: PARTIAL — and the headline target is NOT SOLVED, truthfully.**
**Scored runs spent: 0 of 3. ⛔ Blocked twice over (below).**
**Branch:** `hub/generation-table` @ `cb6d4d4` off `8e36bc9` · **Elapsed:** 5 h 20 m of the 12 h window at the
time of writing · **Running log:** `.claude/outputs/hub-generation-table/LOG.md` (kept as I went, never
written at the end) · **Ledger:** `…/RESULTS-LEDGER.md` · **Prereg:** `…/PREREG-hub-generation-table-1.md`

---

## ⛔⛔ FIRST: THE TWO BLOCKERS, BECAUSE THEY DECIDE HOW TO READ EVERYTHING ELSE

**1. The scored-run budget was never given.** COMMON §1c: *"Do not perform your first scored run until a
human tells you how many of the three remain for your target."* Asked in the log's first entry at 19:10Z,
repeated since. No answer arrived. ⛔ I did not proceed on an assumption.

**2. ⛔⛔ THE SCORED CELL CANNOT RUN ON THIS MACHINE AT ALL — measured, on all three seeds.**
`exp_gen_store._assert_cache_reencodes`, the pin's own re-encode gate, **FAILS here**:

| seed | cache scope | encoder hash matches its cache manifest | gate | max \|Δ\| | ULP (bound **1**) | elements differing (bound 1 %) |
|---|---|---|---|---|---|---|
| 0 | `full` 480,000 | ✅ | ⛔ **FAIL** | 0.001953125 | **51.0** | 0.977 % |
| 1 | `used` 9,288 | ✅ | ⛔ **FAIL** | 0.001953125 | **81.0** | 0.488 % |
| 2 | `used` 9,288 | ✅ | ⛔ **FAIL** | 0.001953125 | **59.0** | 0.939 % |

`exp_gen_cell.run_seed:1780` calls it with the default `will_extend=True`, **before any arm and before the
decoder ceiling**, so the cell aborts having computed **no held-out number**. Seeds 1 and 2 would further
need **32,199 / 32,241** rows encoded on this box (their caches cover only 4.9 % / 4.8 % of what the cell
needs); the gate re-checks at full strength at the point of extension.
⭐ **The gate is right to fire** — it is COMMON §4 ("never mix Mac- and Windows-produced artefacts inside one
comparison") enforced in code, on a bundle whose `built_on_host` is `batMac.local`.
⭐ **Seed 0 needs zero new rows** (100 % coverage), so it could run if `run_seed`'s preflight were scoped the
way the pin already supports (`will_extend=False` + a reason, the `exp_gen_wells_route_b` pattern).
⛔ **That is a one-line change to a pinned harness file and charter §A.4 forbids it to a hub. I did not make it.**

⭐ **And the one piece of good news that underwrites the rest of this report: seed 0's cache covers 100 % of
every DEV corner I used. This hub re-encoded nothing, ever. Every number below was computed on the
Mac-written cached features.**

---

## ⭐⭐ WHAT I FOUND (all DEV: Montero-SHAPED 72-row corners cut from inside `split.train_rows`, held out of the DEV store; n = 49 corners unless stated)

### 1. ⭐⭐ The free floor on this metric is **0.5278**, not 0.3272 — and **six of the ten published rows lose to a constant**
`eval_b` is a complete Cartesian product realising 2 · 2 · 2 · 3 · **1** · 3 factor values, and composition-B
is the unweighted mean of six per-factor accuracies. A single constant output therefore scores at best
`mean(½,½,½,⅓,1,⅓)` = **0.5278 with 1 distinct tuple of 72** — and I *realised* it: emitting one real corner
feature 72 times scores **0.5278 exactly**. ⇒ ⛔ `x0` at matched bytes (0.5008), the atom-mean lookup
(0.3927), `lace` (0.3117), `ebm_compose` (0.1937), `dynread` (0.2114) and `store_removed` (0.1204) **all lose
to emitting the same picture 72 times.** The honest dynamic range is **0.5278 → 0.8650**, width 0.337, and
`clu`'s 0.5949 is 20 % of the way across it.

### 2. ⭐⭐⭐ `knn_proto` is not a ceiling. It is **ADDITIVE**, and the task is second-order.
A pairwise-interaction request→feature model, fitted on the store pool, scored on held-out corners:

| model | aggregate | distinct tuples / 72 | bytes if deployed |
|---|---|---|---|
| additive (= `knn_proto`'s class) | 0.8449 ± 0.0885 | 53.2 | 475,136 |
| ⭐⭐ **+ all 15 pairwise terms** | ⭐⭐ **0.9734 ± 0.0247** | ⭐⭐ **68.0** | ⛔ 11,304,960 |

**Δ = +0.1285, paired 6/6 corners**, and the variance across corners collapses (sd 0.0885 → 0.0247). Two of
six corners are solved **exactly** (72/72, 1.0000). The interactions that matter are `hue × scale` and
`shape × orientation` — physically sensible. ⛔ 11.3 MB = 24× the ceiling arm's ledger, so it is not an arm —
**but the programme's assumption that `knn_proto` is near a ceiling is now measured false.**

### 3. ⭐⭐⭐ The shipped read is restricted to a **vertex** of a subspace the store already owns
On the **shipped** `K=5, M=64, 524,288 B` store, reading the same modes three ways:

| read verb | aggregate | tuples |
|---|---|---|
| nearest mode (a vertex) — the shipped arm | 0.4884 | 11 |
| best **convex** combination (the simplex) | 0.5509 | 12 |
| ⭐⭐ best **linear** combination (the span) | ⭐⭐ **0.8264** | **42** |

⇒ **the 64 modes of each well span the answer; the 30-bit argmax throws it away.** This is the single
largest identified loss in the read path and it needed no new bytes to find.

### 4. ⭐⭐ The quantisation loss is **structure, not magnitude**
Decoding `x + ε` with isotropic `ε` at **exactly** the quantiser's own MSE costs almost nothing
(0.9630 → ~0.93); the quantiser's own residual at the same MSE costs up to **0.43**. ⇒ the codebook's error
is low-dimensional and aligned with the directions the decoder reads. ⭐ This also explains why a
**diagonal** decoder metric (DEV-04) was a null: the harmful directions are not coordinate-aligned.

### 5. The arm I built, and its honest verdict
**`K=32` equal correlation-partitioned wells · `M=26` k-means modes · a request head
`a = Φ_add·C_add + (Φ_pair·E)·U` (rank 16 over two pair terms) · the SHIPPED `gen_compose.clu_wide_vector`.**
store **212,992 B** + head **256,512 B** = **469,504 B = 0.988 × `knn_proto`'s 475,136 B**. Channel **832
reals** (`knn_proto`'s is 2048). Assertions pass: `assert_request_only_factored` (signature · independent
re-implementation bit-compared · provenance · bit-identity under feature replacement) and the shipped
`assert_no_bypass`.

| row (49 corners, seed 0, production objects) | aggregate | sd | **distinct tuples / 72** | bytes |
|---|---|---|---|---|
| §C.2 store-removed | 0.2860 | 0.0981 | 8.7 | 469,504 |
| best constant output | 0.5278 | — | **1** | 0 |
| ⛔ the shipped **ARGMAX** read on the SAME store | 0.6314 | 0.0719 | 18.9 | 469,504 |
| random-basis substitute | 0.6623 | 0.0792 | 33.4 | 469,504 |
| PCA-basis substitute | 0.7055 | 0.1149 | 48.7 | 469,504 |
| atom-basis §C.1 floor (random rows) | 0.7954 | 0.0806 | 51.9 | 469,504 |
| ⭐ **THE ARM** | **0.8135** | 0.0647 | **49.9** | **469,504** |
| ⭐⭐ `knn_proto` | **0.8307** | 0.0621 | 51.8 | 475,136 |

| paired comparison (n = 49) | Δ | SE | 95 % CI | wins | t |
|---|---|---|---|---|---|
| ⛔ arm − `knn_proto` | **−0.0172** | 0.0089 | [−0.0347, +0.0003] | 16/49 | −1.93 |
| ⚠ arm − atom basis (§C.1, random atoms) | +0.0180 | 0.0089 | [+0.0005, +0.0356] | 28/49 | +2.02 |
| ⛔ arm − PCA basis (⚠ **see the retraction below — this control was UNFAIR**) | +0.1079 | 0.0178 | [+0.073, +0.143] | 39/49 | +6.05 |
| ⭐ arm − random basis | +0.1512 | 0.0146 | [+0.123, +0.180] | 44/49 | +10.4 |
| ⭐⭐ arm − §C.2 store-removed | **+0.5275** | 0.0163 | [+0.496, +0.560] | **49/49** | +32.3 |
| ⭐⭐ **arm − the shipped ARGMAX read, same store, same bytes** | ⭐⭐ **+0.1821** | 0.0107 | **[+0.161, +0.203]** | **49/49** | **+17.0** |

### ⛔ RETRACTION — my PCA control was unfair, and the fair one reverses it
The `pca` substitute used the top `M` **centred** principal directions, which cannot represent the well's
mean, while the arm's k-means centroids carry it in every entry. The fair control at identical bytes is the
optimal affine subspace, `[well mean] + top (M−1) directions`. On 24 corners:

| dictionary, identical bytes | aggregate | tuples | Δ (arm − this) | 95 % CI | wins |
|---|---|---|---|---|---|
| ⭐⭐ **`pca_affine`** | ⭐⭐ **0.8365** | **56.5** | ⛔ **−0.0219** | [−0.060, +0.017] | **6/24** |
| the arm (learned k-means modes) | 0.8146 | 48.4 | — | | |
| ⛔ `pca_centred` (my old, unfair control) | 0.7832 | 55.8 | +0.0314 | [−0.010, +0.073] | 16/24 |

> ⛔ **"The store is not just any 832-dimensional subspace" is WITHDRAWN.** A per-well **affine PCA**
> dictionary matches or beats the learned k-means dictionary at identical bytes, and renders **more** distinct
> tuples. ⭐ The honest generalisation is narrower: **learning the dictionary beats storing rows; it does not
> beat taking the principal subspace.** (An affine PCA basis is ⛔ not charter-legal as a store — §B.1 wants
> atoms and basins — so it is not an arm; but it is what a reviewer would ask about, and now it is measured.)

**The §C.1 floor — the charter's actual one, a dictionary of STORED ROWS — made as hard as I could make it (n = 24):** vs **medoid** atoms — the real row nearest each
learned centre, i.e. differing from the arm in *exactly* whether a dictionary entry is a fitted centre or a
stored row — **+0.0850 [+0.056, +0.114], 23/24, t = 5.75**. vs random atoms **+0.0299 [+0.010, +0.050]**.
⚠ vs **maximally-spread (k-means++) atoms +0.0222, CI [−0.004, +0.049] — includes zero.**

### 6. ⭐⭐ THE ACTIONABLE ONE: the SHIPPED store, the SHIPPED bytes, only the read verb changed
The most conservative experiment in the run. The store is the **shipped** `K=5`, `M=64`, **ward-grouped**,
fitted by the **shipped `train_store`**; the head's additive block is `58·K·M·4` = **74,240 B**, the shipped
addressing head **to the byte**; total **598,528 B**, identical in every column to the shipped `clu` arm.
The only change is `read_from_codes(store, argmax)` → `clu_wide_vector(store, real request-addressed weights)`.

| row (6 corners, seed 0) | aggregate | **distinct tuples / 72** | deployed bytes |
|---|---|---|---|
| §C.2 store-removed (argmax / linear) | 0.0741 / 0.0710 | 1.0 / 1.5 | 598,528 |
| best constant | 0.5278 | **1** | 0 |
| ⛔ **`clu_argmax` — THE SHIPPED ARM** | **0.5490 ± 0.1027** | ⛔ **9.5** | **598,528** |
| ⭐⭐ **linear read, SAME store, SAME bytes** | ⭐⭐ **0.6682 ± 0.0587** | ⭐ **35.0** | ⭐ **598,528** |
| §C.1 medoid atoms under the linear read | 0.6755 ± 0.0702 | 37.3 | 598,528 |
| ⭐ + rank-16 pair terms | **0.7400 ± 0.0737** | **41.3** | ⚠ 629,248 (1.051×) |
| `knn_proto` | 0.7909 ± 0.0499 | 53.8 | 475,136 |

| paired difference (n = 6) | mean | 95 % CI | wins |
|---|---|---|---|
| ⭐⭐ **linear − argmax, byte-identical** | **+0.1192** | **[+0.031, +0.208]** | **6/6** |
| ⭐⭐ linear+pairs − argmax (+5 % bytes) | **+0.1910** | [+0.098, +0.284] | **6/6** |
| ⭐⭐ linear − §C.2 store-removed | +0.5972 | [+0.547, +0.647] | 6/6 |
| ⚠ **linear − §C.1 medoid atoms** | ⛔ **−0.0073** | [−0.052, +0.037] | **2/6** |
| ⛔ linear − `knn_proto` | −0.1227 | [−0.167, −0.079] | 0/6 |

⚠⚠ **The caveat, and it is not small: at `K = 5` the §C.1 medoid floor TIES the linear read.** A dictionary of
real stored rows does as well as the learned modes on the shipped 5-well store. ⇒ **the §C.1 pass exists only
at `K = 32, M = 26`** (+0.0850, 23/24), ⛔ **not at the shipped `K = 5`.** The read-verb change is worth taking
on its own merits; it does **not** by itself make the shipped 5-well store non-decorative. Two claims, kept apart.

⭐ **A free external validation:** `clu_argmax` reads **0.5490 with 9.5 distinct tuples** here against the
published B row's **0.5949 with 9.00 ± 0.82**. The distinct-tuple column — the most diagnostic one on this
table — reproduces almost exactly, on a different corner, on a different machine, through my own wiring.

⚠ **How that run ended:** DEV-29 died at corner 6 of 12 on host commit-space exhaustion
(`LLVM ERROR: Unable to allocate section memory` / *"The paging file is too small"*) — three hubs on one box,
and `train_store` re-JITs per corner because the ward wells differ in size (546 s → 1258 s per corner).
⭐ Six corners were already on disk because the ledger is written the moment each completes; ⛔ not restarted,
because the effect is 6/6 consistent and another hour would not change it.

---

## ⛔ THE THREE FALSIFIERS, ANSWERED IN THEIR OWN WORDS

**1. "Beats `store_removed`, `ebm_compose`, `lace`, `clu` AND the matched-bytes atom-mean lookup outright,
beyond seed spread — and WIN or TIE against `knn_proto`."**
⛔ **NOT ANSWERED ON `eval_b` — 0 scored runs, blocked as above.** On 49 DEV corners the arm beats the
shipped argmax read by **+0.182 (49/49)**, the §C.2 control by +0.528, the random substitute by +0.151
and the random-atom §C.1 floor by +0.018 (⛔ the PCA margin is withdrawn — see the retraction); **and it LOSES to `knn_proto` by −0.0172
(16/49).** The ceiling rule needs `Δ > 0.020` to win or `0 < Δ ≤ 0.020` to tie; ⛔ **the sign is negative, so
this is a LOSS, and falsifier 1 FAILS at the `knn_proto` clause.**

**2. "Clears the constant-output floor: distinct factor tuples well above 2.67/72."**
⭐ **PASSED on DEV, and not narrowly.** **0.8135 with 49.9 distinct factor tuples of 72**, against the
constant floor's **0.5278 with 1**, and against the shipped argmax read's **18.9**. (⭐ I also had to *raise*
that floor first: it is 0.5278, not 0.3272.)

**3. "The store is load-bearing: `store_removed` collapses your arm."**
⭐ **PASSED, hard.** Replacing every mode with its well mean takes the arm from **0.8135 (49.9 tuples)** to
**0.2860 (8.7 tuples)** — `Δ = +0.5275, 49/49 corners, t = 32.3`, and the collapsed arm sits far below the
constant floor. ⭐ And unlike the shipped construction this is not load-bearing *by construction*: the
**learned** dictionary also beats a **stored-row** dictionary of identical size by **+0.085 (23/24)** against
medoids — ⚠ **at `K = 32, M = 26`. ⛔ At the shipped `K = 5` that margin vanishes (−0.007, 2/6), so the §C.1
pass belongs to the finer store, not to the shipped one.**
⚠ **The honest caveat:** at `K ≥ 128` the §C.2 control **stops collapsing** (0.44–0.50), because a linear read
over `K` wells gets `K` free scalars from the **partition** alone. ⭐ I made that a hard filter and
disqualified those configurations **before** running them, rather than after.

---

## ⛔ WHAT I GOT WRONG, IN ORDER, AND WHEN I CAUGHT IT
1. **19:40Z** I called `quant_ceiling` a ceiling and concluded "fixing the address book is worth at most
   +0.016". ⛔ Withdrawn at **19:53Z**: it is the *distortion-optimal* addresser, not a bound on accuracy, and
   the discriminative head beat it by +0.042 at K=8.
2. **21:49Z** I reported a search maximum of **0.9097 (Δ +0.134 vs `knn_proto`)** and named, in the same
   entry, that a maximum over hundreds of cells on 72 items is biased. **22:12Z** six corners gave **+0.0147**.
3. **23:41Z** I called ten corners a **TIE (+0.0072)**. ⛔ **00:13Z, 49 corners: −0.0172. That is a LOSS.**
   The first ten were a favourable draw and one of them was the corner the pair set was selected on.
4. **23:10Z** I mis-added the head's byte ledger (259,136 vs the measured **256,512**). Corrected; every byte
   figure now comes from `byte_ledger()` on the live object, never from a formula.

---

## COST LINE
**5 h 20 m wall clock at the time of writing** (START `2026-09-16T19:10:50Z`; 12 h deadline
`2026-09-17T07:10:50Z`). **Local CPU only, 5 threads** (`OMP/MKL/OPENBLAS_NUM_THREADS=5`), per COMMON §3 and
AMENDMENT 1. ⛔ **No GPU. No GCloud. No money spent. CSF3 NOT RUN (2FA, unreachable by any hub).**
Additive environment work, both sanctioned: `h5py 3.16.0` installed (a declared `[project.optional-
dependencies] gen` extra; numpy stayed 2.4.1) and `3dshapes.h5` downloaded to the declared data root in 129 s,
**SHA-256 verified against the shipped constant**. ~27 DEV experiments, ~85 CPU-minutes of scored-harness
equivalent, 0 held-out evaluations.

## GIT FOOTPRINT
`hub/generation-table` @ **`cb6d4d4`** off `8e36bc9`. **5 new files, 940 insertions, 0 deletions, 0 modified.**
`chlu/core/gen_factored_read.py` · `chlu/core/gen_partition.py` · `chlu/core/gen_address.py` ·
`chlu/core/gen_sensitivity.py` · `tests/test_gen_factored_read.py` (**8 tests, all passing**).
⭐ Verified file-by-file that `exp_gen_cell.py`, `exp_gen_store.py`, `gen_compose.py`, `gen_store.py`,
`gen_dream.py`, `gen_request.py`, `gen_probe.py`, `gen_rowsplit.py` and `gen_landscape.py` are **byte-identical
to the pin**. Branch ref confirmed from the main repo, not just the worktree.

## ⚠ OPEN QUESTIONS FOR THE HEAD
1. ⛔ **How many of the 3 scored runs for `generation-table` remain?** Everything is pre-registered and ready.
2. ⛔ **The re-encode gate blocks the scored cell on this box for all three hubs.** Either scope
   `run_seed`'s preflight (one line, the pattern the pin already ships) — which unblocks **seed 0 only** — or
   re-issue seeds 1 and 2's bridge caches at `cache_scope="full"` **from the Mac**. ⛔ There is no correct way
   to produce them here.
3. ⭐ **The generation table's framing needs revisiting regardless of my arm:** the free floor is **0.5278**,
   and `knn_proto` is beaten by **+0.13** by a pairwise request model. The table is measuring an *additive*
   ceiling against arms that are being asked to be compositional.

## PROPOSED HANDOVER UPDATES (for the Hub)
- **§7-CURRENT, new:** *the re-encode gate fails on the Windows hub box on all three bridge seeds (51/81/59
  float16 ULP vs a bound of 1). **13 of the 16 call sites of `_assert_cache_reencodes` in the pinned tree are
  enforced** (11 by default, 2 explicitly `True`) — including `exp_gen_cell:1780`, `exp_gen_compose:408`,
  `exp_gen_baselines:1508`, `exp_gen_dream:519`, `exp_gen_atom_mean:802` and `exp_gen_dynamical_read:507` — so
  every one of them aborts there. **`exp_gen_wells_route_b:382` is the only site that scopes the gate to the
  hazard** (`will_extend=bool(cov["n_shortfall"] > 0)`) and is the worked exemplar for the fix.* **Owner:
  whoever owns the gen experiment modules.*
- **§E.4, new inherited fact:** *the constant-output floor on composition-B is **0.5278** (structural and
  realised), not 0.3272. Six of the ten published rows lose to it.*
- **§E.4, new inherited fact:** *the shipped 30-bit read is restricted to one **vertex** of the 64-dimensional
  span its own modes already form; the best linear combination of the SAME modes scores **0.8264** where the
  argmax scores 0.4884.*
- **§E.4, new inherited fact:** *`knn_proto` is the best **additive** request model, not a ceiling. Adding
  pairwise factor interactions reaches **0.9734 ± 0.0247 (68/72 tuples)** against its **0.8449 ± 0.0885
  (53.2)**, 6/6 corners — at 11.3 MB, so it is a bound, not an arm.*
- **§C, methodological:** *a linear/soft read over a fine partition gets `K` free scalars from the partition
  alone; the §C.2 control stops collapsing at `K ≳ 128`. **`K` must be reported with every §C.2 result.***
- **§C, methodological:** *a PCA substitute basis must be **AFFINE** (`[mean] + M−1 directions`). A centred
  `Vt[:M]` cannot represent the mean and is an unfairly weak control — it cost me a claim, caught at 00:47Z.*
- **§C, new standing control:** *the sharpest §C.1 form is the **MEDOID** dictionary — for each learned
  cluster, the real row nearest its centre. It differs from a fitted store in exactly one thing.*
