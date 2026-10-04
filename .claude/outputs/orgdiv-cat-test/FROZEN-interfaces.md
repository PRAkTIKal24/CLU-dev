# ⛔ FROZEN INTERFACES — the cat test (C2W5 `orgdiv-cat-test`)

**Published 2026-08-01. This artifact gates `orgdiv-null-arms`' spawn (task §1.5).**
Everything below is FIXED and must be byte-identical on every arm. Any arm that deviates is not a
matched control and its number is void (`PhiMismatchError` precedent).

Code: branch `agent/experiment-engineer/orgdiv-cat-test`, `chlu/core/factored_store.py` @ `ad25d37`.
Import surface for the null arms:

```python
from chlu.core.factored_store import (
    CatTestConfig, build_family, build_phi, place_wells, FactoredStore,
    multi_particle_read, fit_readers, score_reader, score_curve,
    exact_set_accuracy, chance_accuracy, occupancy, occupancy_precision,
    reader_bytes, byte_ratio,
)
from chlu.experiments.exp_cat_test import build_physics_arm, stage_arm
```

⚠ **Read `.claude/outputs/orgdiv-cat-test.md` §"registered deviations" before building an arm** —
two registered parameters moved (`m = 1 -> 8`; a new declared designed mechanism
`atom_payload_init_radius`), both forced by measurements and both applying to EVERY arm identically.

---

## (i) The φ instance — FROZEN, bytes ledgered

| item | value |
|---|---|
| constructor | `build_phi(cfg, phi_seed=20260801)` — ⛔ `phi_seed` is **not** `cfg.seed`; re-seeding the family must not re-draw φ |
| query codes `e_j` | `(N_a, d) = (32, 4)`, unit-norm, one per well |
| launch offsets `o_p` | `(P, d) = (4, 4)`, unit directions × `launch_radius = 0.6`. **Designed parameters**, ledgered |
| set-code | `φ(x) = R · normalise(Σ_{j∈A(x)} e_j)`, `R = ball_radius = 2.0` |
| launch | `q0_p = (φ(x) + o_p + σ_q ξ_p , 0_m)`, `p0 = 0`, `σ_q = 0.15`. ⛔ **payload block pinned to 0** (anti-decoration guard) |
| **φ bytes** | `(N_a·d + P·d)·4 = (128 + 16)·4 = ` **576 B** — identical on every arm |
| gradient | φ is **frozen**; `∂q*/∂q₀ = 0` is discharged by construction, not fought (prereg §6 rule 1) |

## (ii) The reader class — FROZEN sizes and fitting protocol

Fitted on the **SEEN split only**; any hyperparameter selected on a held-out-from-seen validation
split, **never on `Q_unseen`**. All readers are permutation-invariant over the `P` particles and take
`z ∈ R^{P×(d+m)}` (the P settled states).

⭐ **Capacity bound, and it is a precondition of the metric, not a preference:** the ground truth
`1_A ↦ y` is linear with **`N_a·m = 256`** free parameters, so a reader with at least that many solves
the family from the SEEN split *with no store at all* (SP-1, measured: `1.000`, `‖v̂−v‖∞ = 4.2e-15`).
**Every member of the class carries fewer than `N_a·m` fitted parameters.**

| id | reader | form | fitted params (`d=4, m=8, P=4`) |
|---|---|---|---|
| **R1** | `sum_linear` | `ŷ = Σ_p wᵀz_p + b` | `(d+m+1)·m = 104` |
| **R2** | `well_table` | `p → argmin_j‖z_p[:d]−c_j‖`; `ŷ = a·Σ_p v̂_{j_p} + b`, `v̂_j` **read from the store**, not fitted | **72** *(corrected in place 2026-08-01; the doc said `2·m = 16`, the shipped `fit_readers` measures 72)* |
| **R3** | `knn` | k-NN over SEEN in canonicalised `z`, IDW, `k ∈ {1,2,3,5,10}` on the seen-validation split | 0 fitted (+128 stored; declared nonparametric) |
| **R4** | `mlp_small` | `ŷ = Σ_p g(z_p)`, `g: R^{d+m} → R^m`, one tanh hidden layer of **width 4**, Adam 600 steps @ 3e-2 | **92** *(corrected in place 2026-08-01; the doc said 88, the shipped `fit_readers` measures 92)* |

`OD(R) = score(R∘z_phys) − score(R∘z_null)`; **`OD_min = min_R OD(R)`** is primary.
⚠ **R2 is store-dependent by construction** and has no query-only counterpart; that is reported, not
papered over. Its null-arm form reads the *null arm's own* codebook payloads.

**Report the curve, not the endpoint:** every score is accompanied by `score_curve(...)`, the
accuracy at `tol × {0.25, 0.5, 1, 2, 4}`. At `m = 8` the metric is all-or-nothing (chance
`3.9e-4`), so a *poor* store and an *inert* store both read `0.000` at the registered `tol`.

## (iii) The split

| item | value |
|---|---|
| constructor | `build_family(cfg, seed)` |
| design point | `N_a = 32`, `F = 4`, `K = 128` seen, `n_unseen = 512` sampled from the rule-4-valid held-out set |
| seeds | **0, 1, 2, 3, 4** (5 seeds; multi-seed before any number leaves a report) |
| rule-4-valid held-out | **24 046 ± ** of 35 960 (measured, 5 seeds; prereg predicted 23 193) |
| payloads | `v_j` **unit-norm** on `S^{m-1}`, `payload_radius = 1.0`, drawn at write time |
| `tol` | `0.25 × RMS‖y − ȳ_seen‖`; **chance = 3.906e-4** (constant predictor) |
| insertion order | re-shuffled per seed (rule 1) |
| ⛔ | a split failing K2 is **rejected, not repaired** (`build_family` raises) |

## (iv) The launch protocol

`P = 4` designed offsets (parameters, ledgered above). The **occupancy vector is a per-read
transient** (F4), never state. Read budget: two-phase damped Verlet,
`(γ_address, N) = (0.05, 400) → (γ_read, N) = (0.02, 800)`, `dt = 0.05`,
`kinetic_mode = newtonian_learned`, `M = I`.
⚠ **Every γ statement is read-budget-scoped** — state the budget beside the band.

## Matched-capacity ledger (what a null arm must match)

| quantity | value | note |
|---|---|---|
| `N_a`, `F`, `K`, `d`, `m` | 32, 4, 128, 4, 8 | identical |
| store parameters | *(corrected in place 2026-08-01, see erratum below)* `a = 32` ⇒ `n_atoms·(dim+2) = 1024·14` floats = **57 344 B** | N1 matches exactly (same parameterisation) |
| φ bytes | 576 B | identical |
| launch offsets | 4×4 floats = 64 B (inside φ bytes) | identical |
| reader params | per table above | ledgered per arm |
| byte ratio | *(corrected in place 2026-08-01)* **9.67×** at the `a = 32` cell that ran (was stated as 3.20× for `a = 12`) | ⛔ **reported, never claimed** |
| tuning budget | ≥ 5 lr × 3 capacity × 3 seeds on SEEN, selected on a seen-validation split | **the physics arm gets the same, no more** |

## ⛔ What a null arm must NOT do

- re-draw φ, change `launch_radius`/`ball_radius`/`σ_q`, or change the read budget;
- fit any reader, hyperparameter or codebook on `Q_unseen`;
- use a reader with `≥ N_a·m = 256` fitted parameters (that is SP-1's out-of-class probe, and it
  scores 1.000 on **every** arm including a blank store — it measures the family, not an organizer);
- report occupancy precision against `F/N_a = 0.125` as a baseline. ⛔ **The admissible baseline is
  the BLANK STORE**, which already beats chance because the wells sit at the frozen query codes and
  the launch geometry alone is a matched filter (measured; encoded as a test).

---

## ⛔ DATED ERRATUM BANNER (Hub, 2026-08-01, `[C2W5]` second review — body above UNTOUCHED, C-3 precedent)
Two rows above are STALE against the cell that actually ran (`orgdiv-null-arms` reconciliations 1–2):
1. **Matched-capacity ledger row:** states `a = 12`, `384·14 = 21 504 B`, ratio 3.20×. The cell that ran
   used **`a = 32` (deviation D2): `n_atoms = 1024`, 57 344 B, ratio 9.67×**. Null-arms measured BOTH
   readings (identical result at every `a` — capacity is not binding, anchor L4).
2. **Reader parameter counts:** doc says `well_table = 16`, `mlp_small = 88`; the shipped `fit_readers`
   measures **72** and **92** (`sum_linear` 104, `knn` 0 agree). All four stay below the `N_a·m = 256`
   bound — no verdict moves; the ledger numbers were wrong.

---

## ⛔⛔ DATED CURATOR ADDENDUM BANNER (2026-08-06, `doc-curator-c2w7-fold`, [C2W7] — body above UNTOUCHED, C-3 precedent)

**Authority:** charter **ADDENDUM 8 §A26.3/§A26.4** + **ADDENDUM 9 §A27.2**, Head-ratified 2026-08-06.
**Sources:** `.claude/outputs/reader-fitting-audit.md` §1/§6/§7/§9 · `.claude/outputs/c2w7-read-cardinality.md`
§3/§4/reconciliation 1 · the 2026-08-06 `[C2W7]` §10 entry. ⛔ **Nothing in §(ii) is retracted and no
published number of this cell moves. This banner ADDS two standing preconditions to the reader-class
specification and re-registers one instrument.**

### 1. ⭐⭐ `K6` — the exactly-right fraction (MANDATORY at every future reader registration)

**Before any reader is fitted, report the fraction of queries whose asserted set is already exactly right.**
It is computable **without a store, a fit or a reader**; it is the **closed-form ceiling of every
zero-parameter reader**; and it is the **exact headroom of any reader-fitting artifact.**
Measured: **~18 %** at the C2W7 multiplicity cell (⇒ the artifact was worth **28×**) against
**2 / 2560** at `orgdiv-null-arms`, **3 / 1280** at `tierii-read-fix` and **0 / 2560** at this cell
(⇒ worth **2–3 queries**). *Derivation, pre-registered and exact:* payloads sit on a sphere of radius
`R` and `tol = 0.25·RMS‖y − ȳ‖`, so a single substituted well costs `‖v_a − v_b‖ ≈ √2 R` = **2.96×**
`tol` here (`R = 1.0`, `tol = 0.478`) and **3.02×** at tierii (`R = 0.5`, `tol = 0.2338`) ⇒ **a
zero-parameter reader's score IS the asserted set's exact-set accuracy, to within one query**
(measured: `gated_well_identity` = `exact_set_occupancy_gated` to 5 decimals).

### 2. ⭐⭐ Every reader class carries a ZERO-PARAMETER MEMBER — **added, never substituted, never reported alone**

⛔ **A fitted-reader 0 is quotable ONLY beside the zero-parameter member's score on the same latent.**
Why the class is *extended* and not *replaced*: the zero-parameter twin can be **strictly worse** —
measured **+0.0109** on a real cell (`tierii-read-fix` null `N1′`, SEEN: `sum_linear` 0.8078 vs
`sum_identity` 0.7969) and **> 0.99** on a designed α-scaling control (fitted 1.0000 vs identity
≤ 0.0104) — so substituting it would itself be a laundering step. Cost of adding it: **0 fitted
params, 0 bytes, `N_a·m = 256` mult-adds/query = 3.7e-6 of the physics read.**
⚠ **Known failure mode of the identity member, named where it can be checked:** it assumes the latent
is already in the target's units and scale — false when the arm's codebook is *fitted* rather than
*written*, when the code is not normalised to the target's cardinality, or when the payload table is
rescaled anywhere in the pipeline.

### 3. ⚠ The pathology this guards against is REAL and it fires in BOTH directions

Least squares fits a **continuous residual** while the metric is an **all-or-nothing threshold at
`tol`**. Measured `diag(W)` **0.128–0.446** across 7 of 8 fitted/identity pairs (the `tol` crossing
fires on 2 of them: identity residual **0.0000** where the fitted reader's is **2.47×** and **2.91×**
`tol`, keeping **3/0** and **2/0** correct queries) — ⛔ **and one INFLATION outlier at this very cell:
`sum_linear`'s `diag(W) = +46.53`, mean residual 13.48 against the identity's 1.84.** ⇒ ⛔⛔ **a fitted
reader's residual is never evidence that a store is "close."**

### 4. ⚠ Provenance footnote on this cell's published SEEN column (⛔ NO VERDICT MOVES)

`orgdiv-cat-test.md` §6.1's SEEN values **`sum_linear = 0.0109375`** and **`well_table = 0.003125`**
appear **nowhere in any `orgdiv-cat-test/results/*.json`**, and `stage_arm.json` carries **no
`seen`/`unseen` keys at all** (top-level: `cells` / `readers` / `aggregate`). The values **with**
provenance are `reader-fitting-audit`'s re-measurement on the same cell: **SEEN `sum_linear` 0.0094 ·
`well_table` 0.0047**. **The unseen column — which is the metric — reproduces BIT-FOR-BIT** (incl. the
non-zero seeds 1–2). ⇒ **cite the re-measured values; footnote the published column as
unreproducible-from-artifacts.** ⛔⛔ **This is a provenance footnote, NOT an erratum: SEEN was never
the metric (it is the in-sample liveness anchor), K5's kill is untouched, and the published unseen
`0.0008 ± 0.0008` stands.**

### 5. ⛔ `S_eff` — the `[8, 16]` band is RETIRED (§A26.4)

`S_eff = K·F/W = 512/W` with `W ≤ N_a = 32` ⇒ **`S_eff ≥ 16` always**: the band's lower half is
**unreachable by construction** and its sole attainable value is 16 (= every well visited). **The
instrument becomes direct wells-visited `W/N_a` with two-sided labels; ⛔ "COLLAPSED" is reserved for
CONCENTRATION.** ⚠ **This cell's own `S_eff` = 34.1 / 51.2 / 36.6 IS concentration, so its COLLAPSED
label survives**; what does not survive is the same word at the band's upper edge (C2W7's 16.77 =
**30.6 of 32 wells visited**, i.e. slight **UNDER-usage**).

---

## ⛔ DATED CURATOR ADDENDUM BANNER 2 (2026-08-06, `doc-curator-tierii-read-fix-catchup`, read-iteration-1 catch-up fold — body above UNTOUCHED, C-3 precedent)

**Authority:** the `[C2W7-CLOSE]` catch-up task; **source:** `.claude/outputs/tierii-read-fix.md`
reconciliation 5 + §2 (flag table). ⛔ **No published number of this cell moves; §(ii)'s table is not
edited.** This banner discharges **one owed one-line doc fix** (registry **N248(b)**).

### 6. ⚠ The §(ii) reader-param table is `(d, m)`-SPECIFIC and needs its column stated

§(ii)'s "fitted params" column is headed **`d=4, m=8, P=4`** and the 2026-08-01 erratum banner's
corrected counts (`sum_linear` **104** · `well_table` **72** · `mlp_small` **92** · `knn` **0**) are
**that cell's**. ⭐ **`tierii-read-fix` measured the same class at `d = 8, m = 8`: `sum_linear` 136 ·
`well_table` 72 · `mlp` 108 · `knn` 0 · `soft_well_table` 72 — all still `< N_a·m = 256`.**
⛔ **Only `well_table` is `d`-independent** (it reads `v̂_j` from the store and fits `a`, `b` per
output dim); `sum_linear` and the MLP both grow with `d`. ⇒ ⛔ **a reader-parameter count is not
checkable against the `N_a·m` cap unless its `(d, m)` is quoted with it**, and
`orgdiv-null-arms` recon-2's **72 / 92** are `d = 4` numbers.
⚠ **`soft_well_table` (the non-quantising twin, iteration 1's declared deviation D8) is a FIFTH
member of the class at 72 params** — recorded here as a fact of that wave's reader class, ⛔ **not as
an amendment to this cell's frozen class of four.**
