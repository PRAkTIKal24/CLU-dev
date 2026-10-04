# c2w8p2-emission-head — experiment-engineer report

**Task + acceptance criterion (one line):** build ARM B (an MLP-class head on `φ` that **emits** the
well parameters — a forward pass instead of 300 gradient steps), prove **K7** first, and score the
arm on pass 1's **unmodified** census (G-CAP / G-DEC / G-DRIFT, ≥ 3 seeds) with K6, K8 and a
merciless byte ledger.
**Status: DONE.** ⭐ **K7 green (two-sided + planted geometry). THE GATE PASSES: all three legs, all
three seeds.** ⛔ **`NO_TIER_II_CLAIM` (private wells) — P6 realised, exactly as registered.**

## ⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary; detail in §8)
1. **(R1) ⭐⭐ THE PASS-2 GATE IS BLIND TO ADDRESSABILITY-FROM-`φ`, AND ARM B IS THE PROOF.** Arm B
   passes G-CAP/G-DEC/G-DRIFT 3/3 seeds while its read-from-query gets **worse than pass 1's**
   (read accuracy **0.135** vs launder **0.729**, margin **−0.594** against pass 1's −0.333;
   `n_never_read = 16/16` on **every** seed, i.e. **no read landed in any basin at all**). The three
   legs measure retrievability *at the sites*; nothing in them measures whether a query reaches its
   site. **The Hub must decide this before adjudicating the race**, because arm A may be scored on
   the same blind gate.
2. **(R2) Pass 1's own/foreign diagnosis is REVERSED, and its cause is now identified as the WRITE,
   not the atom budget.** foreign > own on **0 / 48** wells (pass 1: **45 / 48**); foreign median
   **0.0012 – 0.0152** against pass 1's **0.611 – 1.261**. Mechanism: the 300-step gradient write
   must *move* an item's atoms across the ball to dig, and those displaced atoms become everyone
   else's foreign background. A write that **places** atoms at the site deletes the channel.
   ⇒ `c2w8-well-lifecycle.md` §7 R5 and prereg §1(e)/(f) need this qualifier; it also re-prices
   arm A's premise (the tails are the write's, not only the kernel's).
3. **(R3) `PREREG-C2W8-PASS2.md` §5's P1 band is beaten at the top** (registered 0.60 – 0.95;
   measured **0.875 / 0.938 / 1.000**). P2 and P3 land inside their bands. Scorecard sites.
4. **(R4) My own E1 byte prediction is superseded by my own ERRATA §1 config change** — predicted
   22 060 B, measured **22 316 B**, the difference being exactly the 64-parameter skip added after
   the prereg was filed. Recorded so the delta is not read as a measurement error.

## ⭐ DIAL DECLARATION (echoed before the first result)
- **Dial:** none as a new claim — a **write-side simplification** measured on the capture gate.
  ⛔ No paper number, no tier-ii verdict, no full-CLU verdict, no I2 verdict.
- **Laundering control:** kNN-in-`φ` launder + byte ledger on **every** reading below. ⛔ No
  performance claim at pass 2. ⭐ The gate is **BYTE-BLIND** (`ERRATA-C2W8-PASS2.md` §1 Q3): no gate
  leg reads bytes; the ledger is reported anyway; **no performance number is quoted at the 1 253×
  ratio.**
- **Falsifies:** the gate fails ⇒ emitting well parameters does not make this store capture. *(It
  did not fail.)*
- **Does NOT falsify:** landing in the private-well configuration (P6 = 0.85) — a **claim-barring**
  outcome, not a gate failure. **Both halves are reported.**
- ⛔ Depth is not quotable as feature importance (§A23.5 ACTIVE).

---

## 1. ⛔ K8 FIRST — the label that travels with every number below

```
wells_per_item      = 1
vocabulary_shared   = False
vocabulary_size     = 16   (= n_items)
coefficients        = one-hot (degenerate)
tier_ii_status      = NO_TIER_II_CLAIM
factored_store_built= False
```
Emitted mechanically by `chlu.core.emission_head.emission_ledger`, present on **every** cell, on the
secondary cell and on the aggregate in `census_armB.json`, and **pytest-asserted present**
(`test_k8_ledger_declares_wells_per_item_and_vocabulary_shared`). **Every number in this report
carries it.**

### 1.1 The shared-vocabulary interface (acceptance criterion 6) — written down, and mechanically
###      shown to contain the private case

```
head(φ_i, a_i) ──► w_i ∈ R^V            # coefficients over a STORE-OWNED well vocabulary
store          ──► {θ_v}_{v=1..V}       # V shared well templates (center, width, depth, payload)
item i's contribution to the landscape  =  Σ_v w_iv · well(θ_v)
```
- **shared (tier-ii-capable, NOT BUILT):** `V < n_items`, `w_i` dense ⇒ `vocabulary_shared = True`.
- **private (SHIPPED HERE):** `V = n_items`, `w_i = one-hot(i)` ⇒ the degenerate special case.

Code: `WellVocabulary` · `compose_wells(vocab, coeffs)` · `private_vocabulary(params)`.
`test_the_private_well_case_is_the_degenerate_shared_vocabulary_case` asserts
`compose_wells(*private_vocabulary(p)) == p` **bitwise** on all three parameter blocks. ⇒ *"the arm
does not foreclose the factored store"* is a **green test**, not a promise. Nothing here trains a
shared vocabulary; the factored store remains a declared NOT-RUN.

### 1.2 ⛔ What stops the emitted center from being a pinned anchor (acceptance criterion 7)

Three structural facts and one measurement — and the honest history, because the first answer was
**not sufficient** and that is itself the arm's sharpest theoretical finding.

| # | claim | how it is checked |
|---|---|---|
| a | **No term in the objective is a function of `\|c − φ\|` alone.** The two terms that mention `φ` are the *reach hinge* and the *attribution margin*. | code + tests |
| b | The **reach hinge** `relu(\|q_launch − z\| − ρ·s)²` is **bitwise 0 with a bitwise-0 gradient in the center** once the launch is inside `ρ·s`, and is satisfiable **by growing `s` at a fixed center** — its zero set is a manifold. A pin (`\|c − φ\|²`) has a unique minimiser, is never 0 and pulls at every step. | `test_reach_penalty_is_not_a_pin_*` (2 tests, incl. the pin's own value 0.16 / gradient > 0.5 at the same point) |
| c | The **attribution margin** `relu(m + \|q_i − z_i\| − min_{j≠i}\|q_i − z_j\|)²` is **competitive**: its gradient depends on the nearest *other* well. It is **0 for a store whose wells all sit 3.0 from their launches** provided each launch's nearest well is its own, and it is **exactly 0 for a single item** — precisely where a pin is most active. | `test_attribution_margin_is_vacuous_for_a_single_item`, `..._is_satisfied_far_from_phi_when_attribution_is_right`, `..._has_an_exactly_zero_gradient_once_the_margin_holds` |
| d | The **trainable skip** is an *initialisation* (seeded at `gain·I`, the N98 localized-init precedent), and the only pressure acting on it afterwards is **decoupled weight decay, which shrinks it toward ZERO — toward IGNORING `φ`**, the opposite of a pin. | `test_center_skip_is_an_initialisation_not_a_constraint` |
| e | **MEASURED:** median `\|c − φ\|` = **1.127 / 0.942 / 1.074** = **8.01× / 6.85× / 7.31× the key spacing**, min 0.347. A pinned/snapped arm would read ≈ 0. | `census_armB.json → cells[i].placement` |

⭐ **The finding underneath (registered in `ERRATA-ARMB.md` §1 before the final run).** The reach
hinge **alone cannot supply the organization pressure**: at the emitted widths a Gaussian basin
(`ρ·s`) is comparable to the whole address ball, so *"the launch must be inside the basin"* is
near-vacuous and the head **collapses to a constant placement map**. The only way to make that hinge
bite is to shrink its slack to the key spacing — **at which point it IS a pin.** The escape is the
**competitive** form (attribution), which is what charter §A28.1's *designed write→φ organization
gradient* has to be if it is not to violate the Head's prohibition. **This is the theoretical result
of the arm and it applies to any future placement-emitting write.**

---

## 2. ⭐ K7 — the capture instrument is PROVEN able to report a positive (green BEFORE any arm number)

Predictions filed in `PREREG.md` **before the test file was executed**. Two-sided, plus a third cell
that shows the instrument recovers a *planted geometry* rather than a sign.
Rig: `plant_item` + `flatten_unused_groups` (both **read-only**, `chlu/core/well_lifecycle.py`
untouched), `addr_dim = 2`, `dim = 3`, `r_hi = 1.0`, `n_dirs = 16`, `steps = 8`, `seed = 0`,
`tol = query_sigma`.

| cell | construction | **PREDICTED** | **MEASURED** | verdict |
|---|---|---|---|---|
| **K7-a** POSITIVE | one planted well (D = 1.2, s = 0.25) at `(0.5, 0, 0.1)`, all else flat | `r_hi(1 − 2⁻⁸)` = **0.99609375** exactly | **0.99609375** (Δ = 0) | ✅ |
| **K7-b** NEGATIVE | same site, depth planted at **0.0**, all else flat | **0.0** exactly | **0.0** | ✅ |
| **K7-c** FINITE, R = 0.30 | two identical wells at `±R ê₀`; separatrix = the symmetry hyperplane ⇒ `R / max_k(−u_{k,0})`, `max_k(−u_{k,0}) = 0.9677334` | **0.31000** (±25 %) | **0.30078** (−2.97 %) | ✅ |
| **K7-c** FINITE, R = 0.60 | as above | **0.62001** (±25 %) | **0.61719** (−0.45 %) | ✅ |
| **K7-c** RATIO | doubling the planted separation | **2.000**, accept [1.6, 2.4] | **2.052** (+2.6 %) | ✅ |

Both point deviations are **downward**, which is the one-sided direction the bisection's lower bound
and the finite-horizon damped relaxation guarantee — i.e. the prediction failed in the only
direction it was allowed to.

⇒ **K7 is GREEN, two-sided, and the instrument recovers planted geometry.** A `capture_radius > 0`
reading is now evidence. ⚠ **Coordination note for the Hub:** K7 is asserted in **my own** test file
(`tests/test_emission_head.py`) against the read-only instrument, so wt1 can import/mirror it
without either arm editing the frozen census. **I landed it first** — the Hub should tell wt1.

---

## 3. K6 — OFF is bit-identical AND parameter-count-identical, verified against `main @ 80d7d4b` itself

Not merely "off-vs-off": a **detached worktree at `80d7d4b`** was created, the same digest script run
there and on this branch **with the same venv**, and removed.

| digest (sha256[:16]) | base `80d7d4b` | branch, flag OFF |
|---|---|---|
| `store.V.learned.amp` | `ded4a3bf3d2662f4` | `ded4a3bf3d2662f4` |
| `store.V.learned.centers` | `2f145e5679e3f188` | `2f145e5679e3f188` |
| `store.V.learned.log_width` | `a63027ab1b58580c` | `a63027ab1b58580c` |
| read `q_star` | `239fe00e9bcf4017` | `239fe00e9bcf4017` |
| `n_bytes()` | **664** | **664** |
| 3 write losses | `1.398954838514e-01 · 1.868946105242e-01 · 1.363351708278e-03` | identical to the last digit |
| self-probe acq/strict/decode | `0.8333 / 0.3333 / 0.6667` | identical |

Plus in-suite: `test_emission_head_off_is_bit_identical_and_parameter_count_identical` (read-model
parameter count OFF == absent == ON, `emission_bytes() == 0` OFF, bit-identical reads **and
writes**, `n_bytes()` unchanged) and `test_emission_head_off_is_the_shipped_default`
(`CluSystemConfig().as_flag_table() == {}`). ⇒ **K6 GREEN.**

---

## 4. ⭐⭐ THE GATE — pass 1's census re-run unmodified. **ALL THREE LEGS, ALL THREE SEEDS: PASS.**

Same function (`exp_well_lifecycle.run_census_cell`), same frozen instrument
(`chlu.core.well_lifecycle.census`), same stream, same `φ`, same store config — **the arm supplies
only its flag and a `post_build` hook**. `chlu/core/well_lifecycle.py` was not modified.

| leg | criterion | **pass 1** | **arm B, seed 0** | **seed 1** | **seed 2** |
|---|---|---|---|---|---|
| **G-CAP** | `capture_radius > 0` on a majority | **1 / 48** | **15/16 = 0.938** ✅ | **16/16 = 1.000** ✅ | **14/16 = 0.875** ✅ |
| **G-DEC** | `decode` > chance + 2 SE | **0.0625 = chance, 3/3** | **0.2500** vs 0.1053 ✅ | **0.2500** vs 0.1053 ✅ | **0.2500** vs 0.1053 ✅ |
| **G-DRIFT** | median `site_drift` < key spacing | **0.665 – 0.925** vs 0.14 | **0.0373** vs 0.1407 ✅ | **0.0305** vs 0.1375 ✅ | **0.0179** vs 0.1468 ✅ |

`chance = 0.0625`, `n_probed = 128`, `SE = √(p(1−p)/n) = 0.0214` ⇒ threshold `0.1053`; measured
`0.2500` is **8.8 SE above chance** on every seed.

**Supporting readings (same cells).** capture radius median **0.873 / 0.596 / 0.848** (max 0.9961 =
the search bound on all three); `λ_min` minimum over wells **5.28 / 6.48 / 7.59** (pass 1: 0.79–8.87);
self-probe **acq = strict = 0.984 / 0.961 / 0.938** (pass 1: acq 0.086 / 0.117 / 0.078);
`overdig = 2.00`, `n_live = 16`, **refusal rate 0.000** on every seed (pass 1's `vacuous_gate` monitor
still trips, unchanged and for the same reason). Depth median raw/netted **1.336/1.504 ·
1.333/1.530 · 1.374/1.515** (B1 netting +12.6 / +14.8 / +10.2 %).

### 4.1 Against the registered predictions (`PREREG-C2W8-PASS2.md` §5, implemented not re-derived)

| # | registered for arm B | measured | verdict |
|---|---|---|---|
| **P1** | G-CAP fraction **0.60 – 0.95**, P(majority) = 0.70 | **0.875 / 0.938 / 1.000** | ✅ leg passes; **the band is beaten at the top on seed 1** (R3) |
| **P2** | `decode` **0.15 – 0.50**, P(> chance 2 SE) = 0.70 | **0.250 / 0.250 / 0.250** | ✅ **inside the band**, 3/3 |
| **P3** | median `site_drift` **≈ 0**, P(< spacing) = 0.85 | **0.0373 / 0.0305 / 0.0179** (3.8 – 8.2× below the spacing) | ✅ inside |
| **P4** | all three legs, same arm, ≥ 3 seeds — P = **0.60** | **ALL THREE, ALL THREE SEEDS** | ✅ realised |
| **P6** | private-well configuration ⇒ `NO_TIER_II_CLAIM` — P = **0.85** | **private wells, `NO_TIER_II_CLAIM`** | ✅ realised — ⚠ **by construction, not by measurement** (the shipped head emits one private well per item; the shared vocabulary is a declared NOT-RUN). Reported as a *build fact*, never as a confirmed prediction. |

### 4.2 ⛔⛔ THE OTHER HALF — the gate passes and **addressability from `φ` gets WORSE** (R1)

| reading (⛔ diagnostic, not a gate leg, not a claim cell) | pass 1 | **arm B** |
|---|---|---|
| read accuracy from `φ` over the depth trace (mean, 12 read events / 3 seeds) | 0.25 / 0.41 / 0.28 | **0.135** (final per seed 0.125 / 0.250 / 0.375) |
| kNN-in-`φ` launder, matched **items** | 0.70 / 0.75 / 0.73 | **0.729** (final 0.312 / 0.625 / 0.875) |
| **margin** | **−0.333 ± 0.072** | **−0.594** |
| `n_never_read` (B2 telemetry) | 14/16 | **16/16 on EVERY seed** — **no read landed in any item's basin at all** |

Mechanism, stated exactly: the attribution margin only requires each launch's **nearest** well to be
its own; nothing requires the well to lie inside the read's **actual reach** from `φ`, and the reach
hinge was inactive (`ρ·s ≈ 0.6` while the emitted centers sit `≈ 1.0` from `φ`). At `s ≈ 0.30` a
launch 1.0 away feels `exp(−1.0²/2·0.30²) = 4e-3` of the well's force, so the 400-step address phase
never arrives. ⇒ **This store is retrievable from its own sites and is not addressable from its own
queries, and the three gate legs cannot tell the difference.** ⛔ This is not a reason to discount
the gate result; it is a reason the Hub must not read the gate as "the store works".

### 4.3 own/foreign — ⛔ DIAGNOSTIC ONLY, under BOTH aggregations (`ERRATA-C2W8-PASS2.md` §1 Q2)

| | **MEDIAN (canonical)** | | **MEAN** | | foreign > own |
|---|---|---|---|---|---|
| | own | foreign | own | foreign | |
| **pass 1** seed 0/1/2 | 0.518 / 0.282 / 0.123 | 1.261 / 0.947 / 0.611 | 0.472 / 0.327 / 0.169 | 1.189 / 0.929 / 0.562 | **45 / 48** |
| **arm B** seed 0/1/2 | **1.321 / 1.333 / 1.326** | **0.0012 / 0.0152 / 0.0050** | **1.312 / 1.327 / 1.320** | **0.0250 / 0.0378 / 0.0495** | **0 / 48** |

⛔ **Not a target, never tuned on, no gate leg reads it** (prereg §3). ⭐ But it identifies the
mechanism (R2): pass 1's foreign background was **the write's own displaced atoms** — a 300-step
gradient write must move an item's atoms across the ball to dig, and every item's travelling atoms
are every other item's foreign background. Arm B's write **places** atoms at the site, so the only
foreign contribution left is the untouched `amp² = 1e-4` init scatter. ⚠ In a factored store foreign
contribution would be the **signal**; this reversal is therefore reported, not celebrated.

---

## 5. ⭐ THE BYTE LEDGER — head parameters counted, against BOTH pass-1 anchors

⭐ Gate is **byte-blind** (§1 Q3). ⛔ **No performance number is quoted at any of these ratios.**

| | **arm B, GATE cell** (pass 1's atom budget) | **arm B, SECONDARY min-store cell** | pass 1 |
|---|---|---|---|
| CLU store (atoms) | 360 448 B (8 192 atoms) | **704 B** (16 atoms, 1/item) | 360 448 B |
| codebook | 512 B | 512 B | 512 B |
| **emission head** | **22 316 B** (5 579 params) | **22 316 B** | — |
| **CLU total** | **383 276 B** | **23 532 B** | **360 960 B** |
| head share of total | **5.8 %** | **94.8 %** | — |
| **vs pass 1 (360 960)** | **1.0618×** — *strictly worse* | **0.0652× = 15.34× SMALLER** | 1.000× |
| vs kNN launder (288 B) | 1 331× | **81.7×** | 1 253× |

- **⛔ At the gate's atom budget the head is a pure ADDITION and the arm is 6.2 % MORE expensive than
  pass 1.** That is the finding, not a footnote: emitting well parameters does not by itself move
  bytes out of the store, it moves them *in addition to* the store.
- **⭐ The structural half.** One designed well per item needs **one atom per item**, so the
  `min_atoms_base·√2^d` co-scaling (8 192 at d = 8, **131 072** at d = 16) buys the arm nothing. At
  `atoms_per_item = 1` the same three legs pass — **G-CAP 0.938 · G-DEC 0.2500 · G-DRIFT 0.0373,
  census numbers identical to the gate cell to 4 significant figures** (own/foreign, `P`, `M`,
  `θ_att`, depth medians) — at **23 532 B** and, separately, **45 s of wall against 805 s (17.9×)**.
  ⚠ Secondary cell, seed 0 only, declared as such; the **gate** cell keeps pass 1's atom budget so
  the race against arm A is a race.
- **Compute ledger (the amortised cost, not hidden).** Per-item write: **1 forward pass** vs pass 1's
  **300 Adam steps** (4 800 steps/seed). Head paid **once**: 600 steps × batch 16 = **9 600
  write-objective evaluations, 2.4 – 4.9 s** per seed. Census wall **805 / 692 / 508 s** vs pass 1's
  **1 132 / 960 / 920 s**.
- `gamma_phi_hole_bytes = 0` (flag OFF, no hole placed); `knn_launder_bytes = 288` (matched **items**,
  not matched bytes — pass 1's caveat carries unchanged).

---

## 6. Flag provenance (every number above)

| item | value |
|---|---|
| commits | `bd06561` (head) · `7a93ffa` (wiring) · `47116c8` (rig+CLI+config) · `1aef55e` (tests) · `4f92d7a` (config 2) · `f9cac86` (docstring); **every number below was produced at `4f92d7a`**; base **`main @ 80d7d4b`**, branch `c2w8p2-emission-head`, worktree `../CHLU-c2w8b` |
| env | **main venv reused** (`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python`), **no worktree `uv sync`** ⇒ no package drift (protocol §4) |
| seeds | **0, 1, 2** (gate); secondary min-store cell seed 0; tests seeded per case |
| rig | pass 1's, unchanged: `CluSystem`, learned `V_θ` (`DesignFreedomPotential`, rung `free_mlp`, family `atoms`), `exp_cl_entry` Split-MNIST Class-IL stream, PCA `φ` regime **`task1_only`** |
| addr_dim / payload_dim / dim / n_atoms | 8 / 1 / 9 / **8 192** (gate) · **16** (secondary) |
| capacity / budget / well_budget | 16 / 16 / 8 ⇒ `overdig = 2.00` every seed |
| lifetimes / admission | `stage_lifetimes=True`, `leak=0.02`, `permanent_per_task=1`; `d_safe_override = 0.88 × median-NN(task-1 φ keys)` = 0.1238 / 0.1210 / 0.1292 |
| read | `address_steps=400`, `read_steps=800`, `gamma_address=0.05`, `gamma_read=0.02`, `dt=0.05`, `kinetic_mode=newtonian_learned`, `query_sigma=0.15`, `payload_tol=0.1` |
| **arm-B flags (all non-default)** | `emission_head=True`, `emission_head_hidden=64`, `emission_head_layers=2`, `emission_width_min=0.30`, `emission_width_max=0.80`, `emission_depth_min=1.5`, `emission_depth_max=3.0`, `emission_payload_delta_max=0.05`, `emission_center_skip_gain=1.0` |
| **arm-B training (declared designed mechanism, §A28.1)** | `pretrain_steps=600`, `pretrain_batch=16`, `pretrain_lr=3e-3`, `pretrain_weight_decay=1e-4`, `pretrain_pool=256`, `reach_weight=1.0`, `reach_rho=2.0`, **`attr_weight=10.0`**, `attr_margin=0.15`, `crowd_weight=1.0` at `crowd_d_safe = d_safe`; loss = shipped `train_memory.write_loss` (`n_perturb=32`, `sigma_addr=0.25`, `sigma_pay=0.6`, `margin=0.15`, `barrier=0.2`, `barrier_pairs="nn"`, `crowd_targets` = the batch) |
| **training data provenance** | `φ` fit pool `task1_only`, 256 rows — **the same rows `φ` itself was fitted on**; **no stream item, no test item**; payloads **synthetic** `U(−0.5, 0.5)` ⇒ **the head is never shown a `(φ, label)` pair** |
| write mechanism | **1 forward pass/item** (`_emit_item`); the shipped 300-step `train_memory_landscape` path is not entered |
| census instrument | **UNCHANGED**: `capture_dirs=16`, `capture_bisect_steps=8`, `payload_thresh = payload_tol = 0.1`, `R_cert = 2 s_max + 2.576 σ_q` |
| `gamma_phi` | **OFF** in every cell (`gamma_phi_hole_bytes = 0`) |
| **promotable** | **NO** — inherits pass 1's reason (`phi_dim = 8 < 16`, ERRATA-C2W8 §3) |
| **K8** | `wells_per_item=1`, `vocabulary_shared=False` ⇒ **`NO_TIER_II_CLAIM`** |
| declared NOT-RUNs | merge · prune · depth restoration · every §2.7 claim cell · the factored store (**specified**, not built) · I2 correlation test · cross-stream criterion · wormholes/learned p₀ · any tier-ii/full-CLU/I2 verdict · CSF3 |

---

## 7. How I verified (commands + observed output)

* `python -m pytest tests/test_emission_head.py -q` → **22 passed** (K7 ×4, K6 ×2, K8 ×3, the
  degenerate-vocabulary equivalence, 5 anti-pin properties, 7 emission/pretraining).
* `python scripts/k7_report.py` → the K7 table in §2, verbatim.
* K6 cross-commit digest: `git worktree add --detach ../CHLU-c2w8b-base 80d7d4b`, `scripts/k6_digest.py` both
  sides, same venv → the §3 table; worktree removed.
* `python -m chlu.experiments.exp_capture_armB --quick` → 39 s, real `census_armB.json`.
* `python -m chlu.experiments.exp_capture_armB --seeds 0,1,2` → **2 050 s**; console verbatim in
  `armB_run.log`; artifact `census_armB.json`.
* `python -m pytest tests/test_config.py -q` → **7 passed** (the mutate-every-group round-trip, with
  the new group).
* `ruff check chlu/ tests/` → **All checks passed.**
* Full suite: §9.
* **Config-selection probe** (`.claude/outputs/c2w8p2-emission-head/scripts/diag2.py`, synthetic-`φ`): 14 configurations; results summarised
  in `ERRATA-ARMB.md` §1. ⛔ Scored only on "items are admitted", "centers clear `d_safe`", "centers
  stay in reach of `φ`" — **`capture_radius`, `decode` and `site_drift` were never evaluated during
  selection.**

### 7.1 ⚠ The first configuration produced a DEGENERATE CELL — reported, not hidden

`ERRATA-ARMB.md` §1 was filed **before** the final run. Config 1 (no skip, no attribution term):

| seed | admitted / target | `n_live` | `overdig` | G-CAP | G-DEC | G-DRIFT |
|---|---|---|---|---|---|---|
| 0 | **2 / 16** | 2 | 0.25 | 0.000 | 0.500 vs 0.750 | 0.339 vs 0.141 |
| 1 | **1 / 16** | 1 | 0.12 | 0.000 | 1.000 vs 1.000 | 0.213 vs 0.138 |

⛔ **Not arm B's reading** — 38 of 40 offers were refused because the head emitted near-identical
centers, so a gate scored on 1–2 wells is not the pass-2 gate. Artifacts kept as
`census_armB_CONFIG1_DEGENERATE.json` / `armB_run_CONFIG1_DEGENERATE.log`. The cause and the two
designed changes are in §1.2 and `ERRATA-ARMB.md` §1. The final configuration was run **once** per
seed; no seed was re-run after its result was seen.

---

## 8. Downstream reconciliation list (needs an owner)

| # | what changed | sites that must be reconciled |
|---|---|---|
| **R1** | ⭐⭐ **The pass-2 gate is blind to addressability-from-`φ`.** Arm B passes 3/3 legs 3/3 seeds while `n_never_read = 16/16` on every seed and the launder margin worsens to **−0.594** (pass 1: −0.333). | `PREREG-C2W8-PASS2.md` §3 (the gate's own definition); **the wt1-vs-wt2 adjudication** (arm A is scored on the same legs); any wave summary reading "the gate passed" as "the store works". **Decide before adjudicating the race.** |
| **R2** | **The foreign-domination diagnosis is the WRITE's, not the atom budget's.** foreign > own **0/48** vs **45/48**; foreign median **0.0012–0.0152** vs **0.611–1.261**. Cause: a gradient write must *move* atoms to dig, and displaced atoms are everyone else's background. | `c2w8-well-lifecycle.md` §7 R5 and §3.3; `PREREG-C2W8-PASS2.md` §1(e)/(f); **arm A's premise** (compact kernels attack the tails; this says the tails were largely put there by the write) |
| **R3** | **P1's registered band is beaten at the top** (0.60–0.95 registered; 1.000 measured on seed 1). P2/P3 land inside. | `PREREG-C2W8-PASS2.md` §5 scorecard; the C2W8 row of §A21 |
| **R4** | **My own E1 prediction is superseded by my own ERRATA §1**: 22 060 B predicted, **22 316 B** measured; the 256 B difference is exactly the 64-parameter skip added *after* the prereg. | `PREREG.md` §3 E1 (this spoke's file) — record as *superseded*, not as a miss |
| **R6** | **The base test count is 1445, not the 1443 recorded in `PREREG-C2W8-PASS2.md` §0** (measured by `--collect-only` at `80d7d4b` itself). | `PREREG-C2W8-PASS2.md` §0; any later spoke doing count arithmetic off that baseline |
| **R5** | New measured fact: **`theta_att = 0.0000` on seed 1** (every well captured), against 2.087/1.563/0.944 in pass 1. The census's `theta_att` rule (max fitted depth among non-capturing wells) degenerates when the arm captures everywhere; `is_attractor` then reduces to `λ_min > 0 ∧ captures`. | `ERRATA-C2W8.md` §1 (the `theta_att` rule) — not wrong, but its dynamic range is arm-dependent and any cross-arm comparison of `P` must say so (arm B's `P` = 0.125 / **0.875** / 0.000 is driven by this, not by usage) |

---

## 9. Git footprint

* **Worktree** `../CHLU-c2w8b`, **branch `c2w8p2-emission-head`** off `main @ 80d7d4b` — the branch
  name the task file specifies (not the protocol's `agent/<type>/<slug>` form; the task file wins,
  as in pass 1).
* Commits (verified from the main repo, `git log main..c2w8p2-emission-head`):
  * `bd06561` — `chlu/core/emission_head.py` (new): the head, the shared-vocabulary interface, the
    designed write objective, `emission_ledger` (K8)
  * `7a93ffa` — `chlu/core/clu_system.py`: flag + 4 delimited arm-B blocks + `emission_bytes()`
  * `47116c8` — `chlu/config.py` (additive group), `chlu/experiments/exp_well_lifecycle.py`
    (2 additive seams), `chlu/experiments/exp_capture_armB.py` (new), `chlu/cli/experiment_cmd.py`
    (new `exp-capture-armb` command)
  * `1aef55e` — `tests/test_emission_head.py` (new): K7 / K6 / K8 / anti-pin
  * `4f92d7a` — config 2: the skip init + `attribution_margin_penalty` + the measured bands, + tests
  * `f9cac86` — docstring correction (the anti-pin argument names both φ-mentioning terms); zero
    behaviour change, filed **after** the full suite, re-linted and `tests/test_emission_head.py`
    re-run green (22 passed)
* **Files touched:** `chlu/core/emission_head.py` (new) · `chlu/core/clu_system.py` (flag + wiring
  only) · `chlu/config.py` (additive only) · `chlu/experiments/exp_capture_armB.py` (new) ·
  `chlu/cli/experiment_cmd.py` (additive only) · `tests/test_emission_head.py` (new) ·
  **`chlu/experiments/exp_well_lifecycle.py` — see the declaration below.**
* ⚠ **OUT-OF-LIST EDIT, DECLARED (protocol §3.3).** `chlu/experiments/exp_well_lifecycle.py` is not
  in my task file's ownership list (nor on its read-only list). I made **three additive edits, all
  default-`None`/0, so the pass-1 path is unchanged**: (i) `store_config(..., overrides=None)`
  (2 lines), (ii) `run_census_cell(..., clu_overrides=None, post_build=None)` (3 lines), (iii) one
  `emission_head_bytes` line in `_byte_ledger` so head bytes are not silently booked as "codebook"
  (0 with the flag off). **Rationale:** the pass-2 race is only a race if **both arms are censused
  by the same function**; the alternative was to copy ~150 lines of the cell into my own file and
  let the two arms drift. The seams are generic and **wt1 can use them too**. ⚠ The Hub should
  expect a small adjacency conflict here if wt1 also touched this file.
* **DO-NOT-MODIFY list respected:** `chlu/core/well_lifecycle.py`, `chlu/experiments/usage_telemetry.py`,
  `chlu/core/memory_potentials.py`, `chlu/core/friction_field.py`, `chlu/experiments/cl_baselines.py`,
  `chlu/core/soft_certificate.py`, the C2W6 files and the C2W7 files — **all imported read-only,
  none edited** (`git diff --stat main..HEAD` confirms).
* **The shared main checkout was never edited.** It carries another agent's branch
  (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`); all work was done in
  `../CHLU-c2w8b`. A second, ephemeral **detached** worktree (`../CHLU-c2w8b-base`, at `80d7d4b`)
  was created for the K6 cross-commit digest and **removed**.
* Rebase onto the named base `main` is a **no-op** (`main` has not moved from `80d7d4b`);
  `origin/main` was never used (§7.21). **Nothing pushed, no PR.**
* **Test count arithmetic (measured, not quoted).** `pytest --collect-only -q` in a detached
  worktree at `80d7d4b`: **1445 collected**. Same command on this branch: **1467 collected** ⇒
  **exactly +22, all mine** (`tests/test_emission_head.py`). Of those, **2** are the network-hitting
  `tests/test_download_concurrency.py` (deselected) ⇒ **1465 to run**.
  ⚠ `PREREG-C2W8-PASS2.md` §0 records the base as **1443**; the measured collection at that exact
  commit is **1445**. Two tests, no verdict — recorded so the Hub's baseline is right (R6).

  ```
  $ pytest -q --deselect tests/test_download_concurrency.py --no-cov     # on 4f92d7a
  1465 passed, 2 deselected, 36 warnings in 2542.43s (0:42:22)
  ```
  ⇒ **FULL SUITE GREEN: 1465 passed / 0 failed** = 1443 pre-existing-run (1445 − 2 network) + **22
  mine**. No pre-existing test was reddened; no §7.23 ordering hazard fired (my tests build their
  own systems per case and use no module-scoped fixtures).

---

## 10. Open questions / risks

1. **⭐ R1 is the decision the Hub owns, not me:** should the pass-2 gate acquire a fourth leg
   (*a read launched from `φ(x)` lands in item x's basin*)? Arm B shows the current three can all
   pass while that fails completely. My recommendation: measure it on **both** arms from the data
   already in `census_armB.json` / arm A's equivalent (`usage.n_never_read` and the depth trace's
   `read_acc`) before adjudicating, rather than re-running anything.
2. **The arm's headline is two-sided and must be quoted that way.** *"Emitting the well parameters
   makes this store capture"* is supported. *"Emitting the well parameters makes this store work"*
   is **not**, and §4.2 is the counter-evidence.
3. **`NO_TIER_II_CLAIM` is not a formality here.** The store this arm builds is 16 explicit
   per-item wells set by a forward pass — the intervention doc's degenerate endpoint reached faster,
   exactly as ERRATA §1 Q4 predicted. Its gate result licenses **the mechanism**, not the store.
4. **Cost, declared:** 2 050 s for the 3 gate seeds + the secondary cell. I ran the gate cell **once**
   per seed; the pretraining probe (§7) never touched a gate quantity.
5. **The `attribution_margin_penalty` is new code with one wave of use.** It is the arm's designed
   §A28.1 gradient and it works, but it is a *placement* objective with no theory attached. If the
   Hub wants it carried into C2W9/C2W10 it should be re-registered with its own kill-condition.
6. **Not measured, declared:** whether a **shared** vocabulary trains at all (NOT-RUN); anything
   about `d ≥ 12` (the §7.30 inertness question is untouched — note the emission write does **not**
   depend on the atom-init scatter, so it is a *candidate* escape from 7.30, and that is a hypothesis
   this arm did not test).

---

## Proposed handover updates (for the Hub)

* **§3 config — new flags.** `CluSystemConfig.emission_head*` (**default OFF**, bit-identical and
  parameter-count-identical when off, verified against `80d7d4b` itself): `emission_head`,
  `emission_head_hidden/layers`, `emission_width_min/max`, `emission_depth_min/max`,
  `emission_payload_delta_max`, `emission_center_skip_gain`. New config group
  `experiment_capture_armb` (additive; no existing default touched). New CLI command
  **`chlu exp-capture-armb`**.
* **§2 architecture — one new module + one new experiment.** `chlu/core/emission_head.py` (the head,
  the shared-vocabulary interface, the designed write objective, the K8 ledger) and
  `chlu/experiments/exp_capture_armB.py`. `chlu/experiments/exp_well_lifecycle.py` now exposes two
  **arm seams** (`store_config(overrides=)`, `run_census_cell(clu_overrides=, post_build=)`), both
  default-inert — *this is the supported way for a pass-2 arm to re-run the frozen census.*
* **§7 new entry — 7.32 [OPEN, program-wide, first-order]** *The pass-2 capture gate
  (G-CAP/G-DEC/G-DRIFT) measures retrievability AT THE SITES and is blind to addressability from
  `φ`.* Measured 2026-08-07 (`c2w8p2-emission-head`): arm B passes all three legs on 3/3 seeds with
  `n_never_read = 16/16` on every seed and a launder margin of **−0.594** (pass 1: −0.333). Any
  "the gate passed" statement needs this qualifier.
* **§7 new entry — 7.33 [OPEN]** *Pass 1's foreign-atom domination is produced by the WRITE, not by
  the atom budget.* A 300-step gradient write must move an item's atoms across the ball to dig them;
  those displaced atoms are every other item's foreign background. A write that **places** atoms at
  the site gives foreign > own on **0/48** wells (pass 1: 45/48) with foreign median **0.0012–0.0152**
  vs **0.611–1.261**. Re-prices arm A's premise.
* **§7 new entry — 7.34 [OPEN, design rule]** *A reach hinge cannot supply a write→`φ` organization
  gradient.* At any usable well width the basin is comparable to the whole address ball, so
  "the launch must be inside the basin" is near-vacuous and an amortised head collapses to a
  **constant** placement map; tightening the hinge to the key spacing turns it into a pin (forbidden,
  §A28/prereg §0). The **competitive** form — each launch must be attributed to its own well by a
  margin — is the only one measured to work, and it is what charter §A28.1's designed gradient has
  to be.
* **§7 amend 7.30** — the `addr_dim ≥ 12` inertness of the learned `V_θ` store is a property of the
  **atom-init scatter that the writer must reach**; an *emitting* write never traverses that
  gradient. Not tested at `d ≥ 12` here (declared NOT-RUN) but it is a named candidate escape.
* **§10 running log** — C2W8 pass 2 arm B: K7 green two-sided (0.99609375 / 0.0 / ratio 2.052 vs a
  registered 2.000); **the capture gate PASSES 3 legs × 3 seeds**; `NO_TIER_II_CLAIM`; byte ledger
  **1.062× pass 1** at the gate budget and **15.34× smaller** at one atom per item (same legs pass,
  17.9× faster); and the two-sided caveat R1.
