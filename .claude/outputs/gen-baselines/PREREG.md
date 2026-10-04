# PREREG — gen-baselines (the two competitor arms)

**Filed before the harness existed.** Base `main @ 9a4457d`. Dataset Shapes3D, pretext `recon`,
seeds {0,1,2}, decoder = `gen-decoder-lift` winner `v1_capacity_schedule` (param hash asserted per seed
against `winner_hashes.json`). ⛔ No kill-rule verdict is rendered by this spoke.

## 0. Why a prereg at all
My acceptance criteria are *build* criteria, not a measured law, so protocol §5's pre-registration rule does
not strictly bind. I file one anyway because **this spoke's falsifier is inverted**: a weak kNN hands two kill
rules a false pass. Committing to the numbers in advance is the only defence against tuning my own competitor
down and then calling the result "what it measured".

## 1. Declared build decisions (⛔ fixed here, not after seeing a number)

### R8 request embedding — declared once, applied identically
- **kNN:** per-(factor, value) **prototype** = mean cached train feature over train rows with that factor value.
  A request composes additively in deviation space: `x̂ = μ_global + Σ_k (μ_{k,v_k} − μ_global)`.
  - A `duplicate_factor` request has **two** terms for one factor: both enter the sum (that is what
    "requested simultaneously" means), so the embedding is pulled by two conflicting deviations.
  - An `out_of_range` value has **no prototype**. It is obtained by **linear extrapolation in value space**
    from the two nearest grid prototypes of that factor. ⛔ Not clipped to the nearest grid value — clipping
    would silently turn an invalid request into a valid one and hand the kNN arm a free win on C.
- **LACE:** composed **classifier energies** (AND = sum of per-term conditional energies).
- ⛔ The CLU arm's addressing head is `gen-store-grouping`'s and is neither reimplemented nor borrowed.
- A, B, C are read from disk (`gen_splits` / `c_invalid.json`), never regenerated.

### Byte ledger (⭐ shrink-to-match, never grow, never hand-solved)
Target = the store's **524,288 B** (`gen-store-grouping/runs/shapes3d_recon/store_bytes.json`).
The kNN store is built by k-means on train features and then **quantised as deployed** (⛔ no dtype
normalisation — the ledger counts bytes actually shipped):

| deployment | entries n | bytes | formula |
|---|---|---|---|
| `fp32` | 64 | 524,288 | `n·2048·4` |
| `fp16` | 128 | 524,288 | `n·2048·2` |
| `int8` | 248 | 524,288 | `n·2048·1 + 2·2048·4` (per-dimension scale+offset counted) |

⭐ **Declared second kNN-family arm, `knn_proto`.** The prototype table itself is **57 × 2048 × 4 =
466,944 B ≤ 524,288 B**, i.e. it is *already inside the budget*. So "decode the composed prototype directly"
is a legitimate byte-matched competitor, not a cheat, and refusing to build it would be exactly the hobbling
this task forbids. It is reported as a **separate, clearly-labelled arm** with an **unbounded** read channel
(the full 2048-float vector reaches the decoder), against `knn`'s `k·log₂n` bits and the CLU's 30 bits.
⛔ If it beats the CLU it is reported unsoftened.

### Tuning-trial budget — ⭐ **36 per arm, equal across arms** (Head constraint)
- **kNN:** `k ∈ {1,3,5}` × `metric ∈ {euclidean, cosine}` × `normaliser ∈ {none, zscore}` ×
  `deployment ∈ {fp32, fp16, int8}` = **36**.
- **LACE:** `σ ∈ {0.03, 0.1, 0.3, 1.0}` × `energy_form ∈ {categorical, continuous, both}` ×
  `shrinkage ∈ {ledoit_wolf, 1e-3, 1e-1}` = **36**. (σ = 0.1 is `OFFICIAL IMPLEMENTATION:`.)
- **Selection is on a VALIDATION split of held-back TRAIN rows only** — disjoint from `eval_a` (A),
  `eval_b` (B), the matched-seen negatives and C. Criterion = **aggregate probe accuracy of the decoded
  generation**. ⛔ C never enters tuning; refusal thresholds are set on valid data only.
- ⭐ **Anti-hobbling companion, pre-registered:** alongside the composition-selected configuration I report
  `max over all 36 trials of AUROC(B vs C)` as an explicitly-labelled **ORACLE-SELECTED UPPER BOUND** for
  each arm. A CLU that beats the selected kNN but loses to the oracle kNN cannot then be flattered by my
  selection rule. Both numbers ship.
- OR/NOT constants (`cof`, the `0.1` in α): R9.3 grants tuning on a validation split. ⭐ **Declared now:
  every A/B/C request in this protocol is a pure conjunction, so OR and NOT are never exercised by any
  reported number.** The operators are implemented faithfully (non-commutative OR, ×`cof` on the left
  operand only, 10-operand ceiling) and the constants are calibrated on validation as a **separate,
  non-headline** diagnostic with its own declared budget. ⛔ Not folded into the 36.

### LACE build (R9)
Whitening is **mandatory and full**: the bridge manifest records `D = 2048 ≤ 20000`, `lace_option = 1`
(native whitening) — carried verbatim, ⛔ no PCA projection, ⛔ no channel-only fallback, ⛔ no learned flow
prior. Samplers: **VP probability-flow ODE** (`β ∈ [0.1,20]`, `T=1`, dopri5, atol=rtol=1e-3, times
`linspace(1,1e-3,2)`) and **LD at both temperatures** (`lr=1e-2`, `std=1e-2` as-published;
`std=√(2·lr)=0.14142` corrected; 100 steps). `n_epochs=50`, `batch=128`, `lr=1e-4` with 1000-iter linear
warmup then constant; ⛔ the staircase decay is not wired (it never fires in the original); ⛔ `dis_temp`
is not swept. NVIDIA **non-commercial** licence: nothing is vendored; this is a re-implementation from the
pinned spec. ⛔ `DO_NOT_USE_AS_A_BAR` — their 0.94–0.98 / FID 6.63 are not targets.

## 2. THE PREDICTIONS (point + interval + falsifier)
Metric for composition = **in-corner shape balanced accuracy** (R20/R23): positives = the arm's renders of
the 72 B requests; negatives = the arm's renders of the matched-seen requests (same corner, seen shape).
**Chance = 0.5.** Decoder ceiling (real features, V1) = **0.9414**; STORE ceiling (R22/R23) = **0.6866**.

| # | quantity | point | interval | falsified if |
|---|---|---|---|---|
| P1a | `knn` in-corner shape balacc on B | **0.62** | 0.55–0.70 | outside |
| P1b | ⭐ `knn_proto` in-corner shape balacc on B | **0.90** | 0.85–0.95 | outside |
| P2 | LACE (best sampler) in-corner shape balacc on B | **0.85** | 0.75–0.92 | outside |
| P3a | `knn` AUROC(B vs C) | **0.75** | 0.60–0.90 | outside |
| P3b | LACE AUROC(B vs C) | **0.90** | 0.80–0.98 | outside |
| P4 | realised kNN store bytes | **524,288 exactly**, all three deployments | — | any deployment ≠ 524,288 |
| P5a | composition: ODE ≥ LD-published ≥ LD-corrected | — | — | ordering inverts |
| P5b | refusal: LD-corrected AUROC(B vs C) ≥ LD-published by ≥ +0.02 | — | — | corrected loses |
| P6 | ⭐ **THIS SPOKE'S OWN FALSIFIER** — `max(knn, knn_proto)` in-corner shape balacc on B | — | **≥ 0.70** | **< 0.70 ⇒ I hobbled the kNN and must try harder before reporting** |

**Derivations, so a hit is evidence and not a guess.**
- **P1a** — the `knn` read channel is `k·log₂n` bits (6–8 bits at `k=1`). Resolving *which shape* sits inside a
  fixed hue corner is ~2 bits of the ~19.4 bits the factor grid needs, and the 64–248 k-means centroids are
  placed to minimise global reconstruction error, not corner-conditional shape error. A little above chance.
- **P1b** — the additive-deviation prototype is an unbiased first-order estimate of the true feature under an
  approximately additive code; `gen-store-grouping` measured the channel correlations that make this plausible.
  Decoded through a decoder that reaches 0.9414 on *real* features, I expect a small deficit only.
- **P2** — LACE gets dense factor labels (declared concession) and optimises a composed energy, but its
  classifier has only ever seen the *seen* combinations, so the held-out corner is an extrapolation of the
  decision boundary rather than of the feature itself. Below P1b, above P1a.
- **P3a** — C splits 36 `out_of_range` (extrapolated prototype lands far off-manifold ⇒ near-perfect) and 36
  `duplicate_factor` (the average of two same-factor prototypes lands *near* the manifold, close to an
  intermediate value ⇒ near-chance). `0.5·(1.0) + 0.5·(0.5) = 0.75`.
- **P3b** — LACE's *continuous* energy `½(ŷ−y)²/σ²` handles an off-grid value natively and blows up
  quadratically; the categorical form cannot express it at all (closed vocabulary, R9.7). `duplicate_factor`
  leaves an irreducible residual in the summed categorical energy. Hence high, but not 1.0.
- **P5a/P5b** — the published LD is `T_eff = std²/(2·lr) = 0.005`, ~200× colder than its own target
  (`gen-lace-spec` F2), i.e. near-deterministic gradient descent on the energy ⇒ *sharper* modes (better
  composition) and *less* exploration (a narrower energy gap between satisfiable and unsatisfiable requests).

## 3. What I will NOT do
⛔ No kill-rule verdict (acceptance 6). ⛔ No CSF3 run — 2FA, unreachable from an agent session; the submit
line is emitted and every cluster item is reported **NOT RUN**. ⛔ No regeneration of A, B or C. ⛔ No
substituted or retrained decoder; the V1 parameter hash is asserted per seed before any number is recorded.
⛔ No edit to `gen_store.py`, `gen_dream.py` or `gen_certificate.py`.

---

# ⟲ AMENDMENT A1 — 2026-09-11 23:37, after the plumbing smoke, BEFORE any reported number

**What changed:** the LACE shrinkage grid `{ledoit_wolf, 1e-3, 1e-1}` → **`{ledoit_wolf, 1e-3, 1e-2}`**.
The tuning budget is unchanged at **36** (4 σ × 3 forms × 3 shrinkages), so the Head's equal-budget
constraint still holds and kNN is still 36.

**Why, and on what evidence.** The `--quick` plumbing run measured the whitening check on **train
activations only**: at `shrinkage = 1e-1` the whitened train features have `mean |diag(cov) − 1| = 0.8582`.
⛔ `p₀(u) = N(0, I)` is therefore badly violated at that intensity — and that identity is the **only** thing
licensing LACE's ODE drift to omit the prior score (R9.1, `gen-lace-spec` F1). Leaving `1e-1` in the grid
would have meant offering LACE a trial in which **LACE is not LACE**, i.e. exactly the silent crippling R9.1
refuses. All three replacement values whiten near-exactly at `N = 32,256 ≫ D = 2,048`.

**Why this is admissible.** The trigger is a **property of the train activations alone**. ⛔ No held-out
quantity, no C item, no arm result and no seed other than the smoke's existed when it was taken — the
`gen-store-grouping` D6-b precedent exactly. The realised whitening check for **every** shipped shrinkage
ships in the manifest under `arms.lace.whitening.<shrinkage>.check`, with a boolean
`prior_identity_near_exact` (threshold `|diag−1| ≤ 0.05`), so the decision is checkable rather than trusted.

**Two additions made at the same time, neither of which touches a prediction:**
- `refusal.auroc_c_vs_b_by_kind` — AUROC(B vs C) split into `out_of_range` and `duplicate_factor`.
  P3a's derivation is *explicitly* a 50/50 mixture of an easy and a hard half, so reporting the halves is
  the direct test of the derivation rather than of the point value alone.
- `knn.refusal_reduction_sensitivity` — AUROC(B vs C) under `min` and `mean` distance reduction over the
  `k` neighbours. ⛔ **Not a 37th tuning trial:** `min` is the declared headline and both ship.

---

# ⟲ AMENDMENT A2 — 2026-09-12 00:20. Whitening: shrinkage was the thing breaking LACE.

**Changed:** `lace_shrinkages` `{ledoit_wolf, 1e-3, 1e-2}` → **`{none, ledoit_wolf, 1e-3}`**, where `none` means
λ = 0 plus a **relative eigenvalue floor** at `1e-8·λ_max`. Budget unchanged at **36**.

**Measured, on TRAIN activations only** (16,384 Shapes3D/recon train rows, D = 2048), `mean |diag(cov(u)) − 1|`:

| shrinkage | λ | `mean |diag−1|` | `mean |offdiag|` | eigendirections floored |
|---|---|---|---|---|
| 1e-3 | 1e-3 | 0.2422 | 1.8e-3 | 0 |
| Ledoit-Wolf | 3.49e-4 | 0.1520 | 1.0e-3 | 0 |
| ⭐ **none + rel. floor 1e-8** | **0** | **0.0343** | **7.8e-7** | **72** |

⭐ **The finding:** the shrinkage is *itself* what breaks `p₀(u) = N(0, I)` — and that identity is the **only**
thing licensing LACE's ODE drift to omit the prior score (R9.1). The 72 floored directions are exactly the
near-null ones left by the cache's dead (constant post-ReLU) coordinates — independently corroborating
`gen-store-grouping`'s 71 — and they carry no information, so flooring them costs LACE nothing while every other
direction is whitened exactly.

**Why this is admissible even though seed 0 had already run.** ⭐ **The change makes the competitor STRONGER.**
A correction in that direction cannot flatter us: it can only make the rivals harder to beat, which is the
entire purpose of this spoke. The diagnostic is a train-only quantity (`whitening_check`) that was already being
logged; ⛔ no arm metric, no C item and no held-out row entered the decision. The pre-seed-0 numbers are
superseded and the full run is re-executed from scratch on all three seeds.

# ⟲ AMENDMENT A3 — 2026-09-12 00:25. A second, equally valid-only selection rule.

**Added, not substituted:** each arm now also reports a configuration selected by **validation `shape`
accuracy**, alongside the pre-registered **validation aggregate** rule. ⛔ The pre-registered rule is not
withdrawn and both numbers ship.

**Why.** The reported composition metric is the **in-corner `shape` balanced accuracy**. But the pre-registered
criterion maximises the validation *aggregate*, and ⚠ four of the six Shapes3D factors sit at ≈1.0 and carry no
compositional information — so the pre-registered rule was selecting on a quantity that is **not** what gets
scored. Seed 0 showed the cost concretely: the aggregate rule picked a kNN configuration whose refusal AUROC was
**0.4228, below chance**, while a configuration one row away scored 0.6206 on the same valid-only evidence.
Reporting only the former would have handed kill rules (ii) and (iv) a false pass — the exact failure this task
names as its own falsifier.

The second rule is the **same supervision class**: validation rows are valid, seen train rows, ⛔ disjoint from
A, B, the matched-seen negatives and C.

**What I deliberately did NOT do.** A surrogate invalid set (applying `build_c_invalid`'s rule to validation
rows) would let refusal be tuned properly. ⛔ **Refused:** it would break the supervision-class match that makes
kill rule (iv) meaningful — R7 is explicit that kNN's distance threshold is *"the same supervision class as us:
no labels, no dynamics, **no invalid examples**."* So the honest position stands and is stated as a finding:
**no valid-only statistic selects for detection power**, which is why each arm ships a deployable number *and*
an explicitly-labelled oracle bound, and why I recommend kill rule (iv) be read against the **oracle**.

# ⟲ AMENDMENT A4 — arithmetic correction (not a prediction)
The prototype table is **475,136 B**, not the 466,944 B stated in §1: it carries `Σ_k C_k = 57` prototypes **plus
the global mean**, i.e. `58 · 2048 · 4`. Still inside the 524,288 B budget, so the `knn_proto` arm remains
byte-matched and P4 is unaffected (P4 is about the kNN *store*, which is exact).

---

# ⟲ AMENDMENT A5 — 2026-09-12 14:40. ⭐ THE BYTE-MATCH TARGET WAS WRONG: 524,288 → **688,128 B**.

**Head ruling, relayed 2026-09-12.** The landscape is **two** structures, not one. On top of the modes,
`gen_dream.fit_code_coupling` learns an **inter-well PMI field** `J[k, l][m, m'] = PMI(code_k = m ; code_l = m')`.
It is **extra** to `store_bytes`, and `gen_dream.coupling_bytes`' own docstring says so in terms:

> *"⚠ EXTRA to `gen_store.store_bytes`; a byte-matched baseline **must be told about it**. […] The coupling is a
> second structure and **must be declared separately or the byte match is wrong**."*

**Verified mechanically before acting** (⛔ the task forbids a hand-solved target):

| check | result |
|---|---|
| coupling in `main @ 9a4457d`? | ⛔ **No** — `GenStore` has exactly **5 array leaves** (one per well), `n_bytes()` = 524,288 |
| where it lives | `gen-dream-sampler`'s unmerged branch, `chlu/core/gen_dream.py::fit_code_coupling` |
| formula | `K(K-1)/2 · M² · 4` = `10 · 64² · 4` = **163,840 B** |
| total | `524,288 + 163,840` = ⭐ **688,128 B** |
| read channel | ⭐ **unchanged, `K·log₂M` = 30 bits** — R20's design constant is untouched |

⭐ **Why this had to change.** The kNN arm is the comparator for **two** kill rules — (ii) on B and (iv) on C.
Matching to `store_bytes` alone under-resources it by **24 %**, and an under-resourced comparator makes both kill
rules pass trivially. That is exactly the inversion this task names as its own falsifier: *"a hobbled baseline
does not cost us a fair fight; it costs us the truth, in the direction that flatters us."*

**Realised effect** — every deployment still lands on the budget **exactly, zero slack**:

| deployment | entries at 524,288 | entries at **688,128** | change |
|---|---|---|---|
| fp32 | 64 | **84** | +31.2 % |
| fp16 | 128 | **168** | +31.2 % |
| int8 | 248 | **328** | +32.3 % |

**How it is derived, so it stays mechanical:** `store_bytes` from `store_bytes.json`; the coupling from
`store_plan.json`'s **own** `K` and `n_modes` through the formula above. ⛔ Neither is hand-typed, and
`include_coupling_bytes=False` is the only way to exclude it — never a silent default.
The superseded store-only ledger ships alongside as `byte_ledger_store_only_SUPERSEDED`, so the correction is
visible rather than silent.

⚠ **One caveat the Hub must carry:** the coupling field is in an **unmerged peer branch**
(`agent/experiment-engineer/gen-dream-sampler @ 973e740`), not in my base `main @ 9a4457d`. If that branch lands
with a different `K`, `M`, or a different coupling parameterisation, **this target moves** and the kNN arm must
be refit. The derivation is mechanical precisely so that re-deriving it is a one-line change.

**Predictions:** P1a/P1b/P3a are re-read at the corrected budget. ⛔ I am **not** restating their point values —
the pre-registered numbers stand as filed and are scored against the corrected run, with the budget change noted
as the reason any shift is expected. P4 is **restated**: realised kNN store bytes = **688,128 exactly**, all
three deployments.
