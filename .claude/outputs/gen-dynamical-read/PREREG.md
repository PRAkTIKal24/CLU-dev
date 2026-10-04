# PREREG — the dynamical read (gen-dynamical-read)
**REVISION 3 — Hub, 2026-09-15. SIGNED. Filed BEFORE any scored measurement.**
⛔ **Head + Advisor sign this before the scored run. Nothing here is edited after the run; amendments are appended
with a date as a NEW file `PREREG-addendum-N.md`.**
⭐ Revision 1 is preserved verbatim at **`PREREG-REVISION-1-superseded.md`** (original mtime intact). Superseded
numbers below are struck through, ⛔ never deleted.

---

## 0. ⭐ WHAT CHANGED IN REVISION 2, AND WHO RULED IT
**Advisor→Hub, 2026-09-15 (Head present).** Four targets are now the whole focus; everything else is
rebuttal/appendix. Three consequences bind this document:

1. ⭐⭐ **THE PASS CONDITION MOVED FROM "BEAT" TO "WIN OR TIE" AT THE GENERATION TABLE.** ⛔ This is a **weakening of
   a pre-registered bar** and is recorded as such: it is made **before the scored run**, by the Advisor and Head, on
   the stated rationale that *a tie at matched bytes, from a store that also refuses / completes / deletes, is the
   win narrative*. ⛔ **"Tie" is worthless unless it is a number** — §2b declares the equivalence margin **now**.
2. ⛔⛔ **EXPONENT STUDIES ARE SUSPENDED** — **RULED (Head, 2026-09-15): picked back up in the REBUTTAL PERIOD,
   after version 1 of the paper is submitted. ⛔ Not a target of this run, not a companion, not scored.**
   ⭐ **The single exception, also ruled: ONE cheap LOCAL re-measurement** of the one stale number on the record
   (the deployable exponent, still a broken-split value) — so the record does not carry a known-stale figure into the
   rebuttal. ⛔ **It buys no claim and adjudicates nothing.** *(Hub's earlier reading — "deferred means still measured
   and reported" — was wrong and is corrected here.)*
3. ⭐ **ALL FOUR TARGETS ARE SCORED OFF THIS ONE RUN'S ARTEFACTS.** Refusal (§9), composition route A (§10) and
   dreaming (§11) are derived from the same chains, energies and decoded images. They are pre-registered here so that
   none of them can be defined after its numbers are seen.

**Deferred by the same ruling, and ⛔ therefore NOT scored here:** exponent follow-ups · codebook allocation ·
identification (`tune_n`/`n_val` as a research question — ⚠ but see §5b.0, where it re-enters as a *gate*) ·
the DCI/MIG ladder · **the dSprites exponent companion.**

⭐ **This disposes of revision 1's two blockers:**
- ⛔ **§11b (dSprites exponent) is WITHDRAWN from this pre-registration — DEFERRED, not passed.** It missed its own
  pre-registered directional rule by **0.0011** and ⛔ **is not re-labelled a pass.** It is struck from §2c below and
  survives only as a deferred question. Its failure stands on the record.
- **`knn_proto`'s post-repair fall** is disposed of in §2b: `knn_proto` is the **ceiling/context row**, ⛔ not a bar
  this run must clear, so the fall no longer sits on the critical path. Its status is still ruled on in §14.1.

---

## 1. What is being tested
`request → address-book read on the addressed wells + REQUEST ENERGY → launch particles → settle in a per-well
landscape over the WRITTEN ATOMS with a low-rank inter-well coupling → decode the SETTLED CONTINUOUS state through
store→decoder at prototype width`.

**Atoms = stored TRAIN DATA POINTS** (D2). ⛔ **Never the `k`-means centres** — per-value weights over centres are the
old flat count table wearing a different hat. *(Verified bit-for-bit in the build: `max_abs_delta_vs_train_rows = 0.0`,
64/64 and 512/512 distinct verbatim train rows.)*

### ⛔ Three corrections the build forced, now folded in (were revision 1 §11)
1. ⛔ **COUPLING STRENGTH IS NOT 1.0.** Measured **λ\* = 0.05428 (A) / 0.05473 (B)** from the closed-form coercivity
   certificate; the run uses `0.9 λ*` = **0.04885 / 0.04926** and logs the cap every run. ⭐ 1.0 was **dreaming's**
   value, inherited by accident. ⛔ **No text may say `coupling_strength = 1.0`.** *(A bilinear coupling is indefinite;
   the first smoke diverged to `E = −1.6e28`. Un-certified, a scored run would have mis-reported divergence as
   "still descending".)*
2. ⛔ **DREAMING'S THRESHOLDED CLAMP IS DEAD ON ATOMS** — it clamps **0.00 of 5** wells (mean posterior confidence
   **0.0508** at N=64, **0.0125** at N=512; 3.3× / 6.4× chance). **Binding: the launch uses the posterior ARGMAX.**
   The thresholded rule survives only as the `clamp_atom_thresholded` control.
   ⚠ **Direction:** with 0 wells clamped the request acts **only** through the energy term ⇒ ⭐ **the request-off
   control is the entire test of steering.** It is load-bearing, not a companion.
3. ⭐ **RULED (Head, 2026-09-14): BUDGET B RUNS AT 2000 STEPS, CHECKPOINTED AND RESUMABLE. Budget A stays at 1000.**
   Report the still-descending fraction at **both** 1000 and 2000. ⛔ Still descending at 2000 = **NOT CONVERGED**,
   ⛔ never a null. ⭐⭐ **NO anneal or step-size change may be chosen after seeing budget B.** The step count moved
   BEFORE the scored run and is the only thing that moved.
   *(Forced by: budget B was NOT CONVERGED at 1000 steps — **41.7 %** of `CORNER_B` chains and **17.2 %** of `SPREAD`
   chains still descending. R40.C's 1000 steps is validated for A and ⛔ not for B.)*
   ⭐ **This ruling is hereby CLOSED. Revision 1's "⬜ awaiting ruling" was stale.**

---

## 2. ⭐ THE BARS — ALL RE-BASED ONTO THE REPAIRED SPLIT
⛔ Every number in revision 1 §2 was measured on the **broken `fit`/`val` split** and is superseded.
`gen-val-split-fix` re-selected **every** configuration family on a stratified split and re-scored. ⭐ **The CLU arm
did not move — bit-identical, 3/3 seeds — and every rival did.** ⇒ ⛔ the repair cannot have manufactured a win for
us; it re-priced the comparators only.

### 2a. The read-error ladder (atom-mean family) — ⭐ REPAIRED-SPLIT VALUES ARE BINDING
| leg (3 seeds, error on `A`) | budget A (N=64) | budget B (N=512) |
|---|---|---|
| `centroid_control` k=1 *(fit-time, 0 deployed bytes, ⛔ not an arm)* | 0.2852 ± 0.0048 | 0.1695 ± 0.0021 |
| `atom_mean` oracle k=1 — the like-for-like rival | **0.3986 ± 0.0023** ~~0.4067 ± 0.0094~~ | **0.2306 ± 0.0038** *(bit-identical)* |
| ⭐⭐ **`atom_mean` best `k` — THE ROW THE READ MUST CLEAR** | **0.3451 ± 0.0382** ~~0.3567~~ | **0.2062 ± 0.0347** ~~0.1851~~ |
| deployable atom-mean (request-addressed, best `k`) | 0.4114 | 0.2819 |

⚠ **The bar got HARDER at budget A (−0.0116) and EASIER at budget B (+0.0211).** Both directions are recorded; ⛔
neither was chosen.

### 2b. ⭐⭐ THE GENERATION TABLE — TARGET 1, AND THE ROW THAT DECIDES THE PAPER
All external-instrument, repaired split, 3 seeds. ⭐ **`clu` here is the CODEBOOK read — the arm this run replaces.**

| arm | in-corner shape balacc | composition-B aggregate | distinct factor tuples / 72 |
|---|---|---|---|
| `store_removed` (**the null**) | 0.5000 | 0.1204 ± 0.0131 | 1 |
| `ebm_compose` | 0.5477 ± 0.0216 | 0.1937 ± 0.0182 | — |
| `lace` | 0.6278 ± 0.0289 | **0.3117 ± 0.0153** | — |
| ⭐ **`clu`, codebook read — what we must beat to have improved anything** | **0.6568 ± 0.0226** | **0.5949 ± 0.0082** | **9.00 ± 0.82** |
| `atom_mean` lookup (matched bytes) | 0.6364 ± 0.0645 (A) / 0.6997 ± 0.0400 (B) | — | 22.0 ± 5.7 (A) / 28.7 ± 6.2 (B) |
| ⭐⭐ **`knn_proto` — THE ROW TO TIE** | **0.9165 ± 0.0250** | **0.8650 ± 0.0122** | — |

⛔⛔ **LACE ROW CORRECTED 2026-09-16 — TWO INSTRUMENTS, NOT TWO ANSWERS.** ⭐ **Neither number was wrong.**
**0.3117 ± 0.0153** is `/arms/lace/external/composition_B/aggregate` from the **CELL HARNESS** — the instrument
this table is built from (per-seed 0.3009 / 0.3333 / 0.3009). **0.3241 ± 0.0165** is
`/lace/samplers/ode/composition_B/aggregate` from the **BASELINES** instrument at the historical **688,128 B** leg
(per-seed 0.3472 / 0.3102 / 0.3148). ⛔ **This table takes the CELL row; the baselines row is a different
instrument at a different byte budget and ⛔ is not comparable here.**
⚠ **Declared property of the frozen harness, ⛔ NOT fixed in the pin:** `GenCellConfig.lace_selection_source`
still points at `gen-arms-a7-wiring/runs/historical_688128`, the **broken-split** selection. ⛔ Re-pointing it
would move a rival's numbers and needs its own scoped re-score — ⛔ never a drive-by inside a pin.

⭐ **The gap to close on composition-B is 0.8650 − 0.5949 = 0.2701.** ⛔ Stated plainly so nobody discovers it late.

#### ⛔ THE TIE RULE — declared NOW, before the run. A tie without a number is not a result.
Let `Δ = mean_3seed(dynamical read) − mean_3seed(knn_proto)` on a named metric, at matched bytes.
Let **`m` = the equivalence margin = 0.020**, fixed here as the **sum of the two arms' published 3-seed sds**
(0.0122 + 0.0082 = 0.0204, rounded down to 0.020). ⛔ `m` is not re-derived after the run.

⭐⭐ **HARDENED by `PREREG-addendum-0` (Advisor, 2026-09-15). The table below is the binding one.**
| verdict | condition |
|---|---|
| ⭐ **WIN** | ⭐ **`Δ > m`** **and** the per-seed paired difference has the same sign on **3/3** seeds |
| ⭐ **TIE (favourable)** | ⭐ **`0 < Δ ≤ m`** with 3/3 same sign — ⛔ **a +0.001 with three agreeing signs is a TIE, never a WIN** |
| ⭐ **TIE** | `|Δ| ≤ m` **and** 3-seed intervals overlap |
| **LOSS** | `Δ < −m` |
| ⛔ **INDETERMINATE** | `Δ < 0`, `|Δ| ≤ m` **fails**, but intervals overlap ⇒ reported as such, ⛔ never as a tie |

⛔⛔ **A TIE IS AVAILABLE ONLY AGAINST THE CEILING (`knn_proto`).** ⭐ **Every like-for-like row must be BEATEN
OUTRIGHT, beyond seed spread** — `store_removed`, the codebook read, and the **atom-mean lookup at matched bytes**
(§2a: **0.3451 ± 0.0382** at A / **0.2062 ± 0.0347** at B). ⛔ **A tie against a like-for-like row is a LOSS.**
*(Advisor, on countersigning: "the like-for-like rows still have to be beaten outright. A tie only counts against
the ceiling.")*

⭐ **The headline metric for target 1 is `composition-B aggregate`.** `in-corner balacc` is reported beside it and is
⛔ not substitutable for it after the fact.
⚠ **`knn_proto` is the CEILING/CONTEXT row.** ⛔ Its post-repair fall (0.9005 → 0.8650, reproducible, cross-split
spread ± 0.0010) makes the tie **0.0355 easier** than the published number would have. ⭐ **This is declared, not
buried** — and see §14.1, which is the ruling still owed on it.

### 2c. ⛔ THE EXPONENT — SUSPENDED (Head, 2026-09-15)
⛔⛔ **Everything in this subsection is SUSPENDED until the rebuttal period.** It is kept on the page as a record of
what was measured, ⛔ **not as a live companion, and ⛔ nothing here adjudicates this run.**
| quantity (⛔ suspended) | repaired-split value |
|---|---|
| `atom_mean` oracle fitted exponent (k=1) / implied `PR` | −0.2632 ± 0.0095 / 7.60 ~~−0.2727 ± 0.0141 / 7.35~~ |
| `centroid_control` fitted exponent | −0.2501 ± 0.0050 |
| ⚠ **deployable (request-addressed) exponent** | **−0.162 ± 0.005** — ⛔ **STALE (broken split).** ⭐ **The one ruled cheap LOCAL re-measure**, so the rebuttal does not inherit a stale number. |
⭐ **Retained as standing wording, because it constrains how any future exponent is quoted:** ⛔ **the `−2/PR` RATE is
refuted (overpredicts ~3×); the law survives as a LEVEL (`r = 0.905`).** ⛔ **NEVER quote "6.5 million codewords".**
⛔ ~~**§11b, the dSprites exponent companion.**~~ **WITHDRAWN.** It failed its pre-registered directional rule by
**0.0011** and ⛔ **is not re-labelled a pass.** Its failure stands on the record; it returns, if at all, at rebuttal.
⚠ **The storage-vs-dynamics confound this subsection used to carry does NOT lapse with it:** storing atoms rather
than centres buys a worse level (×1.43) and a slightly better slope, and they nearly cancel ⇒ ⭐ **the atom-mean row
in §2a is still the row that separates storage from dynamics.** It is a LEVEL comparison and survives the suspension.

---

## 3. Budgets and byte ledger (D2, D3) — ⭐ UNCHANGED, and verified to the byte by the build
| | budget A — MATCHED | budget B — DECLARED LARGER |
|---|---|---|
| store (atoms) | **524,288 B** (64 atoms/well) | **4,194,304 B** (512 atoms/well) |
| low-rank coupling `Σ_{k<l} r(d_k+d_l)·4` | `r = 16` → **524,288 B** | `r = 128` → **4,194,304 B** |
| **total** | **1,048,576 B = 2.00×** | **8,388,608 B = 2.00×** |
⭐ Measured `n_bytes()` of the live objects equals this ledger exactly; the run aborts on mismatch.
⭐ **Coupling identity `Σ_{k<l} r(d_k+d_l)·4 = r(K−1)D·4`** holds for any partition of `D` ⇒ a per-seed drift in the
well sizes ⛔ cannot silently renegotiate the pre-registered rank.

---

## 4. Arms and controls — every row at EVERY budget, 3 seeds
**Falsifier-1 rows:** `store_removed` · codebook-argmax · **D6 best-codebook at matched bytes, configured with the
ERROR-OPTIMAL (`mse_optimal`) allocation** — the control must be at its best · **atom-mean lookup (matched bytes)**.
**Ceiling/context: `knn_proto`.**
**Mandatory controls:**
- ⭐⭐ **request-off, same chain, same launch, same key, β = 0** — **steering-attributable gain is the falsifier.**
  This is dreaming's lesson: its clamp produced a gain the coupling-off control took entirely.
- ⭐ **coupling-off** (D3), mandatory.
- `request_off_launch` (total request-off) and `zero_step` (exact null: displacement 0, energy unchanged).

⭐ **Build-scale evidence that the steering falsifier is answerable and was answered NO, emphatically:** at β = 0
**every chain collapses to ONE state** (mean pairwise separation `6.0e-8`, identical final energy `−626.460` from two
different launches) ⇒ **the request energy carries 100 % of the output's distinctness.** ⛔ These are smoke numbers at
2,048 train rows, seed 0 — ⛔ **not scored, and they do not pre-empt the scored answer.**

---

## 5. ⛔ Falsifiers — all answered in these words
1. **Beats `store_removed` AND codebook-argmax AND D6 best-codebook AND atom-mean lookup, beyond seed spread**, at
   each budget — and **WIN or TIE against `knn_proto` on composition-B by §2b's rule.**
2. ⭐ **≥ 60 / 72 DISTINCT FACTOR TUPLES as read by the EXTERNAL CLASSIFIER from the DECODED IMAGES.**
   ⛔ Not distinct settled states — continuous states are trivially distinct and that number is worthless.
   ⚠ **BOTH thresholds are reported, and the original is NOT lowered:**
   | threshold | value | status |
   |---|---|---|
   | ⭐ pre-registered absolute (revision 1) | **≥ 60 / 72** | ⛔ **KEPT. Set with no measured basis; ⛔ not moved now that a basis exists.** |
   | measured comparator floor | `clu` codebook **9.0**, atom-mean **28.7 ± 6.2** | reported beside it |
   ⇒ a result of, say, 35 is reported as **"fails the pre-registered 60/72, beats every measured comparator"** —
   ⛔ never as a pass.
3. **`scale` + `orientation` probe on the settled state above the codebook read's** (baseline `scale` **0.225**,
   `orientation` **0.243**).

---

## 5b. ⭐⭐ THE ITERATION LADDER — pre-registered in order, Advisor's ruling 2026-09-15
**Order, fixed, ⛔ not re-ordered after any rung reports:**
**(1) request strength → (2) kernel width → (3) atom budget → (4) coupling rank → (5) launch init → (6) steps.**
⭐ **The number of rungs climbed is reported with every quoted number.** A number from rung 4 is labelled "rung 4/6".

### 5b.0 ⛔⛔ THE IDENTIFICATION GATE — read this before climbing anything
⚠ **The split repair removed a BIAS. It did NOT buy IDENTIFICATION.** On the *exchangeable* interleaved halving the
two halves of `val` still pick **different winners in 5 of 6** cells; best-to-second gaps are **0.0007–0.0130**
against a trial spread of **0.068–0.111**. On our own arm the selection moved in **6 of 6** cells.
⇒ ⛔⛔ **A SIX-RUNG LADDER SELECTED ON 512 VALIDATION ROWS SELECTS NOISE.** The reported "iteration count" would then
measure effort, not learning.

**BINDING GATE — ⛔⛔ AMENDED 2026-09-15 AFTER `gen-atom-mean-resweep` MEASURED IT. The original single clause was
NECESSARY BUT NOT SUFFICIENT, and the spoke proved it.**

### ⛔ CLAUSE 1 — within-draw (the original gate). NECESSARY.
At the chosen `n_val`, the **interleaved-half** winner disagreement must be **≤ 1 of 6** cells.
⭐ **MEASURED: `n_val = 4,096` reaches exactly 1 of 6** (from 5 of 6 at the shipped 512). ⭐ `n_val` is **RECORDED AS
4,096** and fixed for every rung. ⚠ `n_fit` falls to **28,672 (88.9 %)**, and ⭐ **`knn_proto` did NOT degrade** —
0.877–0.883 across every point against a 0.8817 reference, so the rows were not bought by hobbling a rival.

### ⛔⛔ CLAUSE 2 — ACROSS-DRAW. THE BINDING ONE, AND IT IS NEW.
⭐ **The interleaved-half test compares two halves of the SAME draw** — same rows, same `fit`, same atoms, same
prototypes. ⇒ it measures **within-draw row noise ONLY** and is ⛔ **BLIND to across-draw variation.**
⛔⛔ **MEASURED: `n_val` 4,096 and 8,192 — TWO DRAWS THAT BOTH PASS CLAUSE 1 — select a DIFFERENT winner in 3 of 6
cells.** ⇒ ⛔ **a rung can pass clause 1 and still be unidentified.**
**BINDING: a rung's selection is quotable only if it is UNCHANGED under TWO INDEPENDENT DRAWS** (different split
salts) at the recorded `n_val`. ⛔ Clause 1 alone no longer licenses a number.

### ⭐⭐ CLAUSE 3 — THE LADDER IS BISECTED, AND THE BISECTION IS THE USABLE ANSWER
⛔ *"Is a six-rung ladder a measurement?"* has **no single answer — it depends on the rung.** Measured over all
**36** cell-points of the atom-mean grid:
| lever class | behaviour | verdict |
|---|---|---|
| ⭐ **like `scope` / `normaliser`** | selected **36/36 UNANIMOUS at EVERY `n_val`, including 512** | ✅ **quotable TODAY** |
| ⛔ **like `k` / `metric`** | ⛔ **not identified at ANY `n_val` in range** (4/6 stable even at `n_val ≥ 1024`) | ⛔ **NOT quotable at any affordable `n_val`** |
⇒ ⭐ **Each rung is CLASSIFIED by its own across-draw behaviour, not assumed.** A rung that moves under a second
draw is reported **"no lever found — unidentified"**, ⭐ **the default is kept, and the ladder moves on.**
⛔⛔ **An unidentified rung may not be quoted as a tuned value, and ⛔ more `n_val` will not rescue it** — that is the
measured finding, not a caution.
⭐ **This is the honest form of the gate: it does not bless the ladder, it tells us WHICH RUNGS CAN BE BELIEVED.**

### 5b.1 The rungs
| # | rung | grid (declared here; ⛔ a rung's grid may not grow after it runs) | byte-neutral? |
|---|---|---|---|
| 1 | **request strength `β`** | `{0, 0.25, 0.5, 1.0, 2.0, 4.0}` — ⭐ `β = 0` **is** the request-off control **and** the dreaming cell (§11) | ✔ free |
| 2 | **kernel width** `bandwidth_frac` × `nn_median` | `{0.25, 0.5, 1.0, 2.0, 4.0}` | ✔ free |
| 3 | **atom budget `N`** | `{32, 64, 128}` at A; `{256, 512, 1024}` at B | ⛔ **COSTS BYTES** |
| 4 | **coupling rank `r`** | `{4, 16, 64}` at A; `{32, 128, 512}` at B | ⛔ **COSTS BYTES** |
| 5 | **launch init** | `{clamp_atom (argmax), clamp_atom_thresholded, atom_mixture, random}` | ✔ free |
| 6 | **steps** | `{1000, 2000, 4000}` (B only beyond 2000) | ✔ free (compute only) |

⛔⛔ **RUNGS 3 AND 4 ARE CONSTRAINED TO THE ISO-BYTE CURVE.** `store + coupling ≤ 2× budget` (D3) is binding at every
rung. ⭐ **Raising `N` FORCES lowering `r`, and vice versa.** ⛔ **The ladder may never buy bytes.** Every rung-3/4
configuration ships its byte ledger and the run aborts if the ledger exceeds 2×.

### 5b.2 Rules that make the ladder a measurement rather than a search
- ⭐ **Selection happens on the repaired stratified `val`. Scoring happens on `A` / `B`.** ⛔ A configuration is never
  selected on the rows it is scored on.
- ⭐ **Each rung is written into a `PREREG-addendum-N.md` with its grid and its expected direction BEFORE it runs.**
- **Stop rule, declared now:** the ladder stops at the **first** of — (a) §2b's **WIN or TIE** is reached *(stop and
  report; ⛔ do not keep climbing to inflate it)*; (b) all six rungs climbed; (c) **two consecutive rungs report
  "no lever found"** ⇒ stop and report the arm as **not responsive to tuning**, which is a diagnosis, not a verdict.
- ⭐ **The bandwidth (rung 2) is the mechanism knob and is currently unvalidated.** Too small ⇒ the minima *are* the
  atoms and the read is discrete again — the very thing this arm exists to escape; too large ⇒ one blob per well.
  ⭐ **Report, at every rung-2 setting, the distance from each settled state to its nearest atom.** ⛔ If that
  distance → 0 the arm has silently become a lookup table and no win from it counts.

---

## 6. Dynamics reporting — binding, every run
**Report:** acceptance · **per-step displacement from step 1** · the **energy trace every step** · the **fraction of
chains still descending**, at **1000 and 2000**.
⭐ **A chain still descending at its step budget is reported NOT CONVERGED — ⛔ never as a null.**
⚠ Dreaming's acceptance was **0.952** on a chain that barely moved. The build's chains are different — acceptance
**0.64–0.91** with per-step displacement falling **85×** (0.0665 → 0.00078) — ⛔ but that is smoke, not a result.
⭐ **Report `pairwise_separation`, ⛔ never the participation ratio, for "are these states different".** Measured
trap: full-arm PR fell 5.31 → 1.08 (looks like collapse) while mean pairwise separation *rose* 0.248 → 0.385.
**PR counts directions, not distinct states.**

---

## 7. Channel and the bypass guarantee (D4)
- ⭐ **Declared channel: `D` = 2,048 float32 = 65,536 bits** — **parity with `knn_proto`.** ⛔ **No draft mentions
  30 bits for this arm.** *(Confirmed in the build.)*
- ⭐ **Bypass guarantee: the request-only assertion over launch + settle + read AS ONE PATH.** ⛔ `soft_read(store, X, τ)`
  takes FEATURES and voids it. *(Verified: real path clean at all three stages; the assertion was demonstrated FIRING
  at all three on a bypassed variant in all 4 cells.)*
- **Companion:** the empirical spread of the settled states, beside the declared width.

---

## 8. Cells
- ⭐ **HEADLINE: full requests.** The arm's job is **RESOLUTION INSIDE THE ADDRESSED WELLS** (D1).
- **SECONDARY, pre-registered: partial requests**, own falsifier, rival = `knn_proto` with the marginal mean for the
  unaddressed factors. Runs **after** the headline cell reports.

---

## 9. ⭐ REFUSAL — TARGET 2, pre-registered here, scored off this run's artefacts
**Definition, binding:** refusal is **the read's own settled energy together with its convergence**, i.e. a decision
function of `(final energy, still-descending flag)` and nothing else.
⛔ **"No support entry" is NOT a refusal** — an empty-table lookup is a missing row, not a judgement, and ⛔ may not be
scored as one.
⭐ **The other half of the same number is ACCEPT-ON-UNSEEN-VALID.** A refuser that refuses everything scores 0.
**Reported as one pair from one threshold**: (reject-invalid rate, accept-unseen-valid rate), plus AUROC.

### ⛔⛔ WHERE THE THRESHOLD COMES FROM — `PREREG-addendum-0` §B (Advisor, 2026-09-15), binding
⭐ **The single threshold on `(final energy, still-descending)` is chosen on the repaired stratified `val` split
using VALID REQUESTS ONLY** — e.g. a declared quantile of settled energy on seen-valid rows — **and written into
`PREREG-addendum-1.md` BEFORE any invalid (C-set) request is scored.**
⛔⛔ **Choosing or adjusting the threshold using ANY invalid example is invalid-example supervision and VOIDS the
zero-supervision claim for that run.** ⭐ This is the whole point of the refusal claim: *one dynamics-derived
criterion, with zero invalid supervision.* ⛔ A threshold tuned on the C set makes that sentence false.
**The (reject-invalid, accept-unseen-valid) pair and AUROC are then reported at that ONE threshold, 3 seeds.**
**Rivals, repaired split:** kNN refusal AUROC(C vs B) **0.8457 ± 0.0238** · A7-controlled `out_of_range` AUROC
**0.9583 ± 0.0048** · LACE `ld_published` **0.8097 ± 0.0210**.
⛔ **Falsifier:** the energy criterion fails to separate invalid from unseen-valid beyond seed spread, **or** it buys
its rejection rate by refusing unseen-valid at the same rate ⇒ **refusal is not a property of the landscape.**
⚠ **Wording, when restored:** *"one dynamics-derived criterion covers BOTH kinds with zero invalid supervision, where
each rival covers one"*; `out_of_range` vs kNN is a **TIE**. ⛔ Its quotable status is still formally suspended — §14.2.

---

## 10. ⭐ COMPOSITION VIA SEMANTICALLY PLACED WELLS — TARGET 3, route A only
**Route A (this run, cloud): the built read on the CURRENT wells.** ⛔ Route B (store-in-the-loop learned wells with a
between-well factorisation / within-well multimodality objective, label-free primary + a declared weakly-supervised
arm) is **CSF3, the first long job, and is NOT scored here.**

**Metric for "semantic", declared now: ADDRESS-BOOK SHARPNESS PER FACTOR.**
For each factor `f` and each well `k`, take the address-book posterior induced over `f`'s values and report
**normalised negentropy `S_fk = 1 − H(p)/log(n_values_realised(f))`**, `S = 0` uniform, `S = 1` a delta.
⛔⛔ **`n_values_realised` is reported on every row.** Montero's corner pins `shape` to **1 of 4** values and yields a
confident-looking zero; that has produced a wrong headline in this programme once already. ⛔ **A sharpness number
without its `n_values_realised` is void.**
**Measured starting point:** the address book is **diffuse but informative** — mean posterior confidence **0.0508** at
N=64 (**3.3×** chance) and **0.0125** at N=512 (**6.4×** chance).
⛔ **Falsifier:** sharpness is flat across factors ⇒ the wells are not semantic and route A's composition gain, if
any, is not coming from placement.

---

## 11. ⭐ DREAMING — TARGET 4, the zero-request end of the same dial
⭐ **Dreaming is not a separate run: it is rung 1 at `β = 0`.** Same landscape, same chain, same artefacts.
**Secondary result.** ⛔ Not a headline.
**Requirement, binding:** dreaming must produce **valid NOVEL factor tuples that are NOT stored atoms**, both judged
by the **external classifier on the decoded images**:
- `valid` = the classifier reads a legal tuple;
- `novel` = that tuple is **not** the tuple of any stored atom in the addressed wells. ⛔ **Both conditions, or it is
  a lookup demonstration and proves the opposite of what it is for.**

### ⛔⛔ A MEASURED OBSTACLE, DECLARED BEFORE THE RUN — dreaming is currently DEAD, and why
At `β = 0` the build's chains **collapse to a single state** (pairwise separation `6.0e-8`, identical final energy
`−626.460` from different launches). ⇒ **dreaming as built yields exactly ONE sample**, so it produces at most one
tuple and cannot satisfy the requirement.
⭐ **The cause is not a mystery and is not a tuning excuse: the settle ran at `temp_start = temp_end = 0.0`** — pure
zero-temperature descent, where MH degenerates to accept-iff-energy-fell. **A zero-temperature chain has no reason to
produce more than one mode.**
⭐ **PRE-REGISTERED CONSEQUENCE:** the dreaming cell runs at **finite temperature**, and **temperature is a declared
rung-1 sub-grid: `temp ∈ {0.0, 0.005, 0.05, 0.5}`** *(0.05/0.005 are the shipped `temp_start`/`temp_end` defaults
elsewhere in the codebase)*. ⛔ **Declared HERE, before the run** — ⛔ it may not be introduced after dreaming fails.
⚠ ⛔ **The temperature sub-grid applies to the DREAMING cell only.** The headline cells (§8) run at the declared
`temp = 0.0`; ⛔ raising temperature on the headline arm after seeing its result is a tuning decision on the outcome.
⛔ **Falsifier:** at every declared temperature, dreaming either collapses (≤ 2 distinct tuples) or its "novel" tuples
are stored-atom tuples ⇒ **the landscape is a lookup table with extra steps**, and that is reported.

---

## 12. Artefact paths, provenance and cost
`.claude/outputs/gen-dynamical-read/` — `PREREG.md` · `PREREG-REVISION-1-superseded.md` · `PREREG-addendum-N.md`
(one per rung) · `runs/<dataset>_<pretext>/budget_{A,B}/seed_*/` · `energy_traces_*.json|npz` ·
`settled_states_*.npz` · `curve_retention_vs_bytes.json` · `tables.md`.
⚠ The build's artefacts live under `smoke/` and are ⛔ **deliberately not** at these paths, so a smoke can never be
mistaken for the scored run. ⚠ The code currently writes `runs/<dataset>_<pretext>/seed_*/` with the budget **in the
filename** — ⭐ **the scored run must carry §12's literal `budget_{A,B}/` level** (a one-line change).

**Mandatory on every run, or the run is void:**
- ⭐ **`dec_recipe = "v1_capacity_schedule"`, asserted by name AND param hash `01ef91f7…`.** ⛔ The default is
  `"baseline"` — the decoder that FAILED the gate. A baseline run **looks normal in the logs and is void.**
- ⭐ **Arm keys derived with `fold_in`, ⛔ never by lengthening a split.** On JAX 0.9.0 here
  `split(k,2)[1] == split(k,3)[1]` **bit-for-bit**, so two arms would be identical rather than decorrelated.
- ⚠ **`--quick` clobbers explicit CLI flags** in the `exp_gen_*` mains (`apply_quick` *replaces* the seed/budget/arm/
  step lists; measured: `--quick --budget A B` ran budget A only, log entirely normal). Fixed in this experiment;
  ⛔ the scored run does not use `--quick`.
- ⭐ **A COST LINE in every report:** GPU type, wall hours, and dollar estimate.
  **Policy:** cloud for the ladder, the re-scores and the smokes; **CSF3 for long runs.** Checkpoint-first, sync
  artefacts back to `.claude/outputs/...` — ⛔ **a result that is not synced back does not exist.**
  ⚠ **MEASURED CLOUD REALITY, 2026-09-15 (`forgisprova`), ⛔ not the assumed one:**
  | card | europe-west4 | **us-central1 (where all prior jobs actually ran)** |
  |---|---|---|
  | **L4** | 8 / 8 spot | ⭐ **8, with 2 IN USE by another project ⇒ 6 free** |
  | A100-40GB on-demand | 1 | ⛔ **1, fully consumed by a RUNNING instance (`tsfm-pair-a`)** |
  | ⭐ **A100-40GB SPOT** | 16 | ⭐ **16 free — the only training path available today** |
  | ⚠ **A100-80GB** | **0** | ⛔ **0** — ⚠⚠ **BUT REGIONAL, NOT GLOBAL: `us-east4` has quota 2, ONE FREE on-demand (measured 2026-09-15, every region scanned). ⛔ "no 80 GB card exists for this project" was WRONG.** Quota-verified but never run — stock absent all window |
  | ⛔ **H100 / H200** | **not offered** | ⛔ **not offered** |
  ⇒ ⭐ **The ladder runs on L4 now. Training runs on A100-40GB SPOT, so checkpoint-first is MANDATORY, not advisory.**
  ⛔ **A100-80GB and H200 do not exist for this project until a quota request is granted.**
  ⛔ **Other projects hold live capacity in `us-central1` — ⚠ **FOUR** instances are running, not the one this document first named. ⛔ Evict none of them.**
  ⭐ **Pre-emption survival is PROVEN, not assumed:** a spot VM was taken away mid-run **twice unprompted** and the job still finished, with **five runs converging on one 64-char hash chain over 600 steps**. ⚠ **Bit-identity needs forced XLA determinism** — at default flags a resumed run agrees only to ~6 significant figures, an autotuning artefact, ⛔ not a broken resume.

---

## 13. What would make us wrong
- ⭐ **Request-off gives the same gain** ⇒ the chain is settling onto a mode, not being steered. **Dreaming exactly.**
- ⭐ **Settled states sit on top of their nearest atom** (rung-2 diagnostic → 0) ⇒ the arm is a lookup table and any
  win is storage, not dynamics.
- Chains still descending at the step budget ⇒ **NOT CONVERGED**, and ⛔ no number from that run is quotable.
- Falsifier 2 measured on settled states rather than decoded factor tuples ⇒ meaningless by construction.
- ⭐ **Every rung reports "no lever found"** ⇒ the arm is not responsive to tuning; the ladder's iteration count then
  measures effort, not learning, and ⛔ must be reported that way.
- A sharpness or per-factor number quoted without `n_values_realised`.
- ⭐ The read wins on level while its exponent matches the quantiser's ⇒ it is buying points, not representation.
  ⛔ Companion only in revision 2 — ⛔ it does not adjudicate.

---

## 14. ⭐ RULINGS — ALL FOUR GIVEN (Head, 2026-09-15)
1. ⭐ **`knn_proto`'s post-repair fall: ACCEPTED. The bar is 0.8650 ± 0.0122.** ⛔ Recorded plainly: this makes the
   tie **0.0355 easier** than the published 0.9005 would have. The rationale is that `knn_proto` has **no tuning
   axis**, so the fall is a `fit`-row rebalancing and 0.9005 is now known to be an artefact of an unrepresentative
   `fit` (alternative-salt control: cross-split spread **± 0.0010**).
2. ⭐ **Refusal: §9's definition REPLACES the suspended wording.** One dynamics-derived criterion — settled energy
   plus convergence — ⛔ not two definitions. The suspension is thereby discharged, ⛔ not "restored".
3. ⛔ **Exponent: SUSPENDED to the rebuttal period** (§0.2, §2c). One cheap local re-measure of the stale number only.
4. ⭐ **The equivalence margin `m = 0.020` is SIGNED** as the tie bar (§2b).

---
## SIGN-OFF
⭐ **REVISION 3. All four rulings at §14 given by the Head on 2026-09-15**, on the Advisor's four-target direction of
the same date. Revision 1's two blocking rulings were closed in revision 2 (budget-B steps; the `val` split).
- [x] **Head** — 2026-09-15, all four points.
- [x] **Advisor** — 2026-09-15. §2b margin `m = 0.020` and §9 refusal definition CONFIRMED, **conditional on
      `PREREG-addendum-0.md`** (A: WIN requires `Δ > m`, else TIE-favourable; B: refusal threshold set on valid data only,
      before any C-set row is scored; C: identification gate = one job, capped). Both A and B harden the bar.
⭐⭐ **BOTH BOXES TICKED — THE SCORED RUN IS CLEARED TO START (2026-09-15).**
⛔ **It runs under `PREREG-addendum-0`'s two hardenings**, folded into §2b and §9 above: a WIN must clear the
margin, and the refusal threshold is set from valid data only, before any invalid request is scored.
⛔ **The identification gate (§5b.0) is a PRECONDITION, not a study** — one calibration, run once, its `n_val`
fixed for every rung.
