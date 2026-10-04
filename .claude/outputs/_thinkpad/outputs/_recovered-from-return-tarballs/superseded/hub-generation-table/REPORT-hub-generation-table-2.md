# hub/generation-table-2 — autonomous-hub report (RELAUNCH, 2026-09-17)

**Target:** WIN or TIE (R1) against `knn_proto` on composition-B, matched bytes, external classifier,
shipped cell, seed 0.
**Machine:** FORGIS_14, Windows 11 Pro AMD64, **CPU only** — every number below was produced here (R3).
**Branch:** `hub/generation-table-2` @ `2e65d96` · **Base:** `945a3e9` (`base/iter2`).
**Running log:** `.claude/outputs/hub-generation-table/LOG-iter2.md` (RELAUNCH section, appended
throughout, ⛔ never written at the end). **Ledger:** `RESULTS-LEDGER-2.md`. **PREREG:**
`PREREG-hub-generation-table-2.md`.
⛔ Iteration 1's `hub-generation-table.md` is left byte-untouched beside this file.

---

## VERDICT

> ## ⭐ **TIE — banked, truthfully, and NOT a win.**
> **Scored runs spent: 1 of 3** — *"scored 1/3 — seed 0 only, 3-seed pending full caches."*
>
> On `eval_b`, seed 0: **the arm 0.8495 with 51 distinct factor tuples of 72 at 454,336 B**, against
> **`knn_proto` 0.8588 with 57 tuples at 475,136 B**. **Δ = −0.0093; |Δ| ≤ m = 0.020 ⇒ TIE under R1.**
> The powered companion, on 42 held-out DEV corners: **Δ = −0.0131, 95 % CI [−0.0243, −0.0019], 19/42** —
> same sign, same magnitude, also inside the tie band.
> ⛔ **I did not get a win, I said in writing before measuring that I would not, and I am not dressing
> the tie up as one.**

**The three falsifiers, answered in the task file's own words:**

| # | falsifier | verdict |
|---|---|---|
| 1 | beats `store_removed`, `ebm_compose`, `lace`, `clu` **and** the matched-bytes lookup outright — and **WIN or TIE vs `knn_proto`** | ⭐ **PARTLY.** Rivals: beaten outright (+0.259 vs `clu`, +0.546 vs `lace`, +0.618 vs `ebm_compose`, +0.738 vs `store_removed`). `knn_proto`: ⭐ **TIE under R1.** ⛔ **The matched-bytes lookup clause FAILS on the scored cell** — see below. |
| 2 | clears the constant-output floor, tuples ≫ 2.67/72 | ⭐ **PASSED, not narrowly: 0.8495 with 51 tuples** against the floor's **0.5278 with 1**. ⚠ **Six published rows still sit BELOW that floor** (`lace` ×3, `ebm_compose`, `store_removed`, and the §C.2 control). |
| 3 | the store is load-bearing; `store_removed` collapses it | ⭐ **PASSED on §C.2 everywhere** — scored **0.3657 / 11 tuples** (K = 64), DEV **+0.4000, 42/42**. ⚠ **§C.1 passes on DEV (+0.0261 [+0.0161, +0.0361], 36/42) and FAILS on the scored cell.** ⭐ §C.4 ratio **3.382** ⇒ not a lookup table. |

### ⛔ The one I got wrong, stated first rather than buried
I pre-registered *"the arm beats its §C.1 `atoms_medoid` control by +0.01 … +0.05"*. On `eval_b` the arm
scores **0.8495** and a dictionary of **real stored rows** scores **0.8495** — the same number — while
maximally-spread real rows (`atoms_kpp`) score **0.8634**, *better* than the arm.
⛔ **On the scored cell, charter §C.1 does not pass, and by §C.1's own words the store is reported as
decorative there.** On 42 DEV corners the same comparison is **+0.0261, CI excluding zero, 36/42**, and
the arm renders **51 tuples against every atom control's 46**. ⭐ The honest reading is that the §C.1
margin is **real but small (~2 images on a 72-image cell)** and **one cell cannot resolve it**. ⛔ That is
a fact about the instrument, not a discharge of the failure.

---

## ⭐⭐ WHAT THE PROGRAMME GETS OUT OF THIS RUN, whatever happens to my arm

1. ⭐⭐⭐ **THE COST OF BEING A STORE IS NOW PRICED: +0.0270 [+0.0188, +0.0352], 39 of 43 corners.**
   A **storeless** reference with the same head and the same budget — 52 dense prototypes + the shared-basis
   pair remap, **471,744 B** — scores **0.8374 / 54.1 tuples** and **beats `knn_proto` by +0.0118
   [−0.0001, +0.0237], 29/43** (a tie, one corner short of a resolved win). The arm scores 0.8104.
   ⭐ `DENSE_ADD` (no interaction, **425,984 B = 10.3 % FEWER bytes than the ceiling**) reads 0.8289,
   **+0.0033 [+0.0004, +0.0061] over `knn_proto` — nominally a WIN by R1's letter, and I am flagging that
   rather than leaving it implicit.** It is a sanity check, not a result: it is `knn_proto`'s own model
   class re-parameterised on the reduced design, and it is storeless.
   ⇒ ⭐ **The gap between the arm and the ceiling (0.0152) is SMALLER than the factorisation loss (0.0270):
   if being a store were free, the arm would win.** The arm is not losing because the head is weak, the
   dictionary is badly fitted or the read verb is wrong.
2. ⭐⭐⭐ **And at this byte scale the store is not buying compression either.** `C_add` (52 × A) plus
   `B` (A × D/K) costs `A·(52 + D/K)·4`; their product costs `52·D·4`; they break even at
   **`A* = 52·D/(52 + D/K) = 1,268` atoms at K = 64.** The arm runs at **A = 1,216 = 96 % of break-even**:
   ⭐ **the product quantiser saves 4 % of bytes and costs 0.027 of accuracy.**
   ⇒ ⛔ **On this table a product-quantised store cannot be justified as compression.** It must be
   justified by what a dense prototype table cannot do — **exact atom-wise deletion, request-only
   addressing, the memory modes** — which is exactly what the programme's goal statement calls the core.
   ⚠ `A*` rises with `K` (592 / 918 / 1,268 / 1,566 at K = 16 / 32 / 64 / 128), so compression only starts
   to pay at large `K` — and **§C.2 disqualifies K = 128.** The two constraints close on each other, and
   that closure is the measured reason this table ties rather than wins.
3. ⭐⭐ **THE READ VERB, CONFIRMED ON `eval_b`, AND IT IS BIGGER THERE THAN ON DEV.** On the **shipped
   K = 5 store**, at **598,528 B in every column** (store 524,288 + addressing head 74,240), changing the
   read from the 30-bit argmax to a request-addressed **linear** combination of the same modes is worth
   **0.5903 → 0.7917 = +0.2014**, and takes distinct tuples from **9 to 27** of 72. Its §C.2 control
   collapses to **0.1111 / 4 tuples**. ⭐ Iteration 1 measured **+0.1192 (6/6)** on DEV and recommended it;
   **on the held-out cell it is +0.2014.** ⛔ No new store, no new bytes, no new decoder —
   `gen_compose.clu_wide_vector` already exists and already takes no feature argument.
   ⭐⭐⭐ **And it is now POWERED, not a one-cell claim.** On **43 held-out DEV corners**, same shipped
   store (loaded, never re-trained), with the linear head asserted **in code** to be 74,240 B = the
   shipped addressing head to the byte: **+0.1768 [+0.1512, +0.2024], 43 of 43 corners, tuples
   12.8 → 37.1.** Three independent instruments agree — iteration 1 at n = 6 (+0.1192, 6/6), this hub's
   scored cell at n = 1 (+0.2014), this hub's DEV at n = 43 (+0.1768) — and **the scored value sits
   inside the 43-corner CI.**
   ⚠ The DEV rows use the shipped store, which was fitted on a pool containing those corners, so **both**
   rows share that leak and ⛔ only the PAIRED difference is quoted; the absolute levels are not
   comparable with `eval_b`'s.
   ⚠ ⛔ **It does not make the shipped store non-decorative:** `linear` still loses to `knn_proto` by
   −0.0893 [−0.1025, −0.0760], 0/43, at **1.26 × the bytes**. Two separate claims, kept separate.
   ⭐⭐ **This is the programme's cheapest available gain and my clearest recommendation.**
4. ⭐⭐ **A new, byte-efficient head: the SHARED-BASIS interaction** (`chlu/core/gen_shared_basis_head.py`).
   Taking `U := C_add` makes a factor-pair interaction cost `n_pair·n_add·4` **independent of the atom
   count**, instead of `r·A` per rank unit. Measured contribution **+0.0255 [+0.0113, +0.0397], 32/42**,
   and **43.6 → 51.4 distinct tuples**. ⭐ The interaction was never worthless; **it was unaffordable.**
5. ⭐⭐ **The dictionary's OBJECTIVE matters, and the legal objective is the better one.** Fitting atoms
   for the **span** (NMF — the read verb's own objective, non-negative ⇒ atoms stay decodable and
   deletable) beats fitting them for the **vertex** (k-means) by **+0.0149 [+0.0080, +0.0219], 31/42**.
   ⭐ NMF recovers ~60–79 % of the R4 `pca_affine` control's advantage while staying charter-legal.
6. ⭐⭐ **THE BEST CONFIGURATION FOUND, measured on the same 43 corners and NEVER SCORED (the budget was
   spent before it existed): NMF atoms + the SHIPPED ward grouping + a WATER-FILLED per-well allocation.**
   **0.8219 / 52.0 tuples at 469,600 B, Δ vs `knn_proto` = −0.0037 [−0.0167, +0.0093], 23/43** —
   statistically **level with the ceiling**, and **+0.0115 [+0.0063, +0.0167], 31/43 over the frozen arm**.
   ⛔ Still not a win (n = 43 needs +0.0122). ⚠ Two things travel with it: a water-filled allocation is
   **ragged**, so it cannot use the pinned rectangular `clu_wide_vector` until that one function is
   generalised (`chlu/core/gen_wide_read.py` is the worked pattern); and I first called this lever NULL on
   6 corners (−0.0004) before 43 corners resolved it at +0.0105 — **the same instrument error as the
   headline, in the opposite direction.**
   ⭐ **The complete ordering, all on the same 43 held-out corners:** `DENSE_PAIR` (storeless bound,
   +0.0118) > `knn_proto` (0) > **BEST** (−0.0037) > **the scored arm** (−0.0152) > `ARM_KMEANS` (−0.0294).
   The factorisation loss falls from **+0.0270** against the frozen arm to **+0.0155** against BEST ⇒
   ⭐ **this run closes ~43 % of the cost of being a store, and the rest is still there.**
   ⚠ Predicting that gain by ADDING the lever's paired effect to the arm gave **+0.0172**; measuring it
   gave **+0.0115** — ⛔ **additivity over-predicted by a third**, which is exactly the size of error that
   turns a tie into a claimed win on paper.
7. ⭐ **Three measured NEGATIVES that should stop the next hub wasting a day:**
   - ⛔ **Every hyperparameter lever is noise.** Six of them (reduced design, metric-correct truncation,
     separate pair ridge, NMF alone, their combination, a third pair term) are each **≤ ±0.007** and would
     need **61–4,657 corners** to resolve. There are 49.
   - ⛔ **Trading atoms for interaction rank is strongly negative.** The configuration with a *completely
     unrestricted* pairwise block (A = 320, r = 160 = `n_pair`) is **−0.103** against A = 832, r = 22.
     Score is **monotone increasing in the atom count** with no peak inside the budget.
   - ⛔ **The low-`K` end is much worse, and it refuted my own best idea.** `K = 1` with 51 wide atoms —
     which by byte algebra should have *zero* factorisation loss — scores **0.65** against K = 64's
     **0.89**. ⇒ the binding constraint is **not** the dimension of the span but **whether a LABEL-FREE
     dictionary happens to contain the label-relevant directions**; generosity in atom count is how you
     buy that, and §B.6 is what makes it expensive.
8. ⭐⭐ **PARITY IS NOT A WORRY ON THIS TABLE, AND IT IS MEASURED.** Windows seed-0 `clu` composition-B
   **0.5903** against the Mac-published 3-seed **0.5949 ± 0.0082**, distinct tuples **9** against
   **9.00 ± 0.82** — ⭐ both inside bound. The harness's own `rival_regression_check` reports `clu`
   **`bit_identical: true, Δ = 0.0`** on the probe companion. `knn_proto` +0.0025.
   ⚠ ⛔ **One rival DID regress and the harness flagged it: `lace` −0.0941 against a 0.05 tolerance**
   (`regressed: true`) — reported, not absorbed, not mine to fix (`lace_selection_source` still points at
   the broken-split selection; charter declared property #2).
9. ⭐ **`K = 128` scored HIGHER than the configuration I kept (Δ vs `knn_proto` +0.0289 vs +0.0204 on the
   selection set) and is DISQUALIFIED**, because its §C.2 control reads **0.5147** and does not collapse
   below the 0.5278 floor. ⛔ Disqualified *before* scoring, never after. **That number was available and
   I did not take it.**

---

## THE TABLE, seed 0, `eval_b`, with the constant-output floor drawn on it
Figure: `.claude/outputs/hub-generation-table/relaunch/scored/generation_table_seed0.png`
⭐ Every row carries its **distinct-tuple count** and its **deployed byte ledger**, measured from the LIVE
objects (`gen_store.store_bytes` + the head's own `byte_ledger`), ⛔ never from a formula alone.

| row | composition-B | tuples / 72 | bytes | × budget |
|---|---|---|---|---|
| frozen DECODER CEILING | 0.9861 | — | — | — |
| ⚠ §C.1 `atoms_kpp` (adversarial control) | 0.8634 | 46 | 454,336 | 0.956 |
| ⭐ **`knn_proto` — the ceiling** | **0.8588** | **57** | 475,136 | 1.000 |
| ⭐⭐ **THE ARM** | **0.8495** | **51** | **454,336** | **0.956** |
| ⚠ §C.1 `atoms_medoid` (sharp control) | 0.8495 | 46 | 454,336 | 0.956 |
| ⛔ R4 `pca_affine` (**control only**) | 0.8472 | 53 | 454,336 | 0.956 |
| the arm with k-means atoms | 0.8310 | 47 | 454,336 | 0.956 |
| §C.1 `atoms_random` | 0.8194 | 46 | 454,336 | 0.956 |
| ⭐⭐ **`clu_linear` — shipped store, SHIPPED BYTES, read verb only** | **0.7917** | **27** | 598,528 | — |
| D-matched Gaussian null | 0.6944 | 44 | 454,336 | 0.956 |
| ⛔ **`clu` — the shipped 30-bit argmax read** | **0.5903** | ⛔ **9** | 598,528 | — |
| ⚠ **best constant output** | **0.5278** | ⛔ **1** | 0 | — |
| §C.2 `store_removed` on the arm (**K = 64**) | 0.3657 | 11 | 454,336 | 0.956 |
| `lace` (CELL instrument) | 0.3032 | 72 | — | — |
| `lace/ld_corrected` · `lace/ld_published` | 0.2662 · 0.2593 | 70 · 56 | — | — |
| `ebm_compose` | 0.2315 | 72 | ⛔ 14,226,660 | ⛔ 29.9 |
| `store_removed` (shipped argmax path) | 0.1111 | 1 | 524,288 | — |

⭐ **Where the −0.0093 actually lives:** floor/wall/object hue are 1.0000 for both. **`scale`: arm 0.1806
vs ceiling 0.1667 — the arm is BETTER.** `shape`: 0.9306 vs 0.9861. `orientation`: 0.9861 vs 1.0000.
⇒ ⭐ **the entire deficit is four images of `shape`.** One factor-accuracy step on a 72-image cell is
`1/(72·6) = 0.0023`, so ⛔ a single scored cell cannot carry a CI and must not be read as if it could.

---

## THE ARM
```
K = 64 wells (32 channels each, ward-leaf-order partition, dead channels spread) · label-free
A = 1,216 atoms (M = 19/well), NMF dictionary · label-free (sees train activations, nothing else)
head = SHARED BASIS: C_add (52 x 1216) + E (220 x 52); U := C_add, stored once
pairs = floor_hue x scale, object_hue x scale, shape x orientation
read = gen_compose.clu_wide_vector (uniform allocation => the SHIPPED verb, tested bit-identical)
store 155,648 B + head 298,688 B = 454,336 B = 0.956 x knn_proto's 475,136 B
channel = 1,216 real coefficients (WIDE - never quoted in the 30-bit column)
```
⛔ **No pinned harness file was edited.** `exp_gen_store`, `gen_compose`, `gen_store`, `gen_dream`,
`gen_request`, `gen_probe`, `gen_rowsplit`, `gen_landscape`, `gen_bridge`, `exp_gen_compose` are
byte-identical to the pin tree; `exp_gen_cell` is byte-identical to `base/iter2`.
⛔ **Nothing was encoded.** Both legs printed `shortfall = 0 of 33,038` before gathering a row.
⭐ **Replication (charter §A.5): leg 2 re-run from a CLEAN detached worktree of the committed branch is
BIT-IDENTICAL on all 13 rows** — every aggregate to 6 dp, every per-factor accuracy, every tuple count,
the byte ledger and the §C.4 ratio.

## METHOD NOTE THE PROGRAMME SHOULD ADOPT
⭐⭐ **Select on a declared subset, confirm on the rest, and declare the split before the first sweep.**
Here: **SELECT = 6 corners, CONFIRM = 42**, declared at 11:08Z. ~45 configurations were compared on
SELECT. **SELECT said Δ = +0.0378. CONFIRM said Δ = −0.0131.** ⭐ The selection bias was **+0.051** — a
factor of ~2.9 — and ⛔ **without the split this report would claim a win, and it would be false.**
⭐ Equally: **the comparison that chooses a mechanism must be variant-vs-variant, not variant-vs-rival.**
The paired sd against `knn_proto` is **0.070**; between two variants it is **0.015–0.03**. Using the wrong
one, six real levers all look like noise and a 4-corner effect looks like a 300-corner one.

## GIT FOOTPRINT
`hub/generation-table-2` @ **`2e65d96`** off `945a3e9` — **3 new files + 2 new test files, 0 modified,
0 deleted.** `chlu/core/gen_shared_basis_head.py` (new), `chlu/core/gen_wide_read.py` (new),
`chlu/core/gen_dict_fit.py` (inherited from the stopped predecessor, adopted 11:35Z),
`tests/test_gen_shared_basis_head.py` (8 tests), `tests/test_gen_wide_read.py` (4 tests) — **12 passing.**
Branch ref verified **from the main repo**, not only from the worktree (AGENT_PROTOCOL §3.2).

## COST
**Local CPU only, 4 threads throughout** (2 workers × 2, per COMMON §3). ⛔ **No GPU. No GCloud. Nothing
spent. CSF3 NOT RUN (2FA — unreachable by any hub).** ~1,050 DEV units, 1 scored cell run (814 s),
2 leg-2 runs (277 s + replication).
⚠ **Two operational faults of mine, both declared in the log:** I exceeded the 4-thread budget for ~18
minutes because `pkill` does not exist in this shell and my kills were silent no-ops (11:28Z); and I hit
`env_hub.sh`'s documented import trap on the first scored launch, importing the bundle's unpatched
harness (15:43Z) — the gate caught it, **before any held-out row was read**, so no scored run was spent.

## WHAT I WOULD DO NEXT, IN ORDER
1. ⭐⭐ **Ship the read-verb change.** `+0.2014` on `eval_b` at a byte-identical ledger, 9 → 27 tuples.
   It costs the programme one line and no bytes.
2. ⭐⭐ **Re-aim the generation table.** `knn_proto` is not the ceiling: a byte-matched storeless pairwise
   model reaches **+0.0118 / 54.1 tuples**. Put that row on the table as the *bound*, and put **0.5278**
   on it as the *floor*, so the next hub is not sent to beat an additive model believing it is a ceiling.
3. ⭐⭐ **Decide what the store is FOR at this byte scale**, because it is now priced: **+0.027 of accuracy
   and ~0 bytes saved** versus a dense prototype table. If the answer is deletion, addressing and the
   memory modes — which the goal statement says it is — then **the generation table is the wrong place to
   demonstrate the store**, and the deletion/refusal targets are the right ones.
4. ⛔ **Do not spend another hub on:** the settling dynamics, the codebook's capacity, interaction rank,
   `K < 32` or `K ≥ 128`, or hyperparameter tuning of this head. All five are measured here as not the lever.
5. ⭐ **3 seeds.** Seeds 1–2 need `scope="full"` caches re-issued **from the Mac**; there is no correct way
   to produce them on this box.
