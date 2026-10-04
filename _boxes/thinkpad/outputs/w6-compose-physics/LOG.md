# w6 hub/compose-physics — LOG

**TARGET:** rows 3 and 4 with the PHYSICS read — settle the state in the landscape — against
`pca_tensor` and the full rival set. **Dial: the READ, specifically whether SETTLING adds anything
over a static read on the same content.** Content held FIXED at `pca_tensor`'s.
**Scored budget: composition target, 1 spent in w5, 2 remaining.**

---

## 2026-09-20

- **06:26Z** `START_UTC.txt` written to `/c/chlu/outputs/w6-compose-physics/`. No live one existed.
  Worktree `/c/chlu/worktrees/CHLU-w6-compose-physics`, branch `w6/compose-physics`, base
  `df02568` (`base/iter6`), tree clean at boot.
- **06:26Z** Instrument verified: `chlu.__file__` =
  `C:\chlu\worktrees\CHLU-w6-compose-physics\chlu\__init__.py` — inside my worktree. jax 0.9.0,
  numpy 2.4.1, equinox 0.13.4, system python, no venv created.
- **06:27Z** Booted by reading, in order and nothing else: `autonomous-hub-charter.md`,
  `w6-hub-common.md`, `w6-hub-compose-physics.md`. No earlier wave reports read.
- **06:30Z** Located the incumbent. `pca_tensor` lives in
  `chlu/experiments/exp_w5_compose_field.py`; it is a runtime-additive extension of
  `exp_gen_read` (`install()` appends arms, asserts every pre-existing arm unmoved). Its read path
  is `arm_features` verb `"tensor"` → `TensorAddress.read(store, values, grid.values, basis)`:
  a **static** per-well point emitted by the product address book and rendered into
  `span(G_k)`. Content = `PCASpanStore` = per-well `[mean; 63 leading principal directions]`,
  64 × 2048 f32 = 524,288 B, fitted on `partitions.npz['pool']`, unsupervised.
  Cost on w5's scored leg: address-book fit 623.5 s, all rows 389.2 s per seed.
- **06:32Z** Instrument sizes confirmed for the estimand: row 4 = **49 n-way corners**
  (per-corner ceiling 72), row 3 = **60 pair sub-corners** (ceiling 36). `run_arm` calls
  `arm_features` **once per corner** on all that corner's rows. ≥49 paired corners is therefore
  met by row 4 exactly and exceeded by row 3, with no change to the estimand.

### ⭐⭐ The ceiling that already bounds this hub, read off w5's ALREADY-SCORED table

- **06:38Z** ⛔ **Before writing any code: the maximum gain available to ANY read of the incumbent's
  content is already on the table, and it is smaller than the margin.** `oracle_content_pool` is
  literally `pca_tensor`'s own content (same `PCASpanStore`, `source="pool"`, same 64 atoms, same
  wells) read by an **oracle** — the requested row's true feature projected onto the span of the
  deployed atoms. By Eckart–Young that is the best answer any read whose output lies in
  `span(G_k)` can give. From w5's scored table, `pca_tensor` vs `oracle_content_pool`, paired
  delta (positive = incumbent ahead):
  - row 4: **−0.837 / +0.449 / −0.755** (seeds 0/1/2)
  - row 3: **−1.000 / −1.000 / −1.500**
  So the headroom a better read of this content can possibly capture is **at most 0.837 tuples per
  corner on row 4** — and on **seed 1 it is negative**: the perfect read of this content is
  already 0.449 **worse** than the incumbent's static address book.
  ⭐ w5's own pre-declared decision margin was **+1.0 tuple/corner**. **The entire read dial on
  this content is therefore bounded below that margin before a single settle step is taken.**
  ⚠ Scope of the bound, stated exactly: it binds reads whose output lies in the span of the
  deployed atoms. A settled read under a store energy built from those atoms lies there by
  construction, so it binds this hub's arm.
- **06:38Z** **What this does NOT settle:** whether settling *ties* or *loses*. The ceiling says
  settling cannot win by the declared margin; it does not say the dynamics are harmless. That is
  the empirical question this hub still has to answer, and it is the difference between
  "dynamics are decoration" and "dynamics cost composition". Both are findings; they are
  different findings. Proceeding to build and run the settled read.

### Reconnaissance on the incumbent's actual atom geometry

- **06:52Z** Copied w5's **val (dev)** leg `val_seed0` into
  `/c/chlu/outputs/w6-compose-physics/dev/val_seed0` — `pca_content.pkl` and the fitted
  `address_pca_prod_R128_lam0.01_sw40_n131072.pkl`. ⭐ This is how the content is held **fixed**:
  the dev leg reuses the incumbent's own content object and its own address book pickle, byte for
  byte, rather than refitting something nominally identical. Selection work is on `val` only.
- **06:53Z** Measured what `pca_tensor`'s 64 atoms per well actually **are**:

  | well | d_k | \|a_0\| | \|a_1..a_63\| | nearest-neighbour atom distance (median) |
  |---|---|---|---|---|
  | 0 | 771 | 13.156 | all exactly 1.000 | 1.4142 |
  | 1 | 940 | 18.183 | all exactly 1.000 | 1.4142 |
  | 2 | 217 | 7.872 | all exactly 1.000 | 1.4142 |
  | 3 | 60 | 1.073 | all exactly 1.000 (61 nonzero rows) | 1.4142 |
  | 4 | 60 | 4.080 | all exactly 1.000 (61 nonzero rows) | 1.4142 |

  ⭐ **The atoms are one mean vector plus 63 mutually ORTHONORMAL directions** — nn distance
  1.4142 = √2 is the signature of orthonormal unit vectors. **This content is a BASIS, not a
  point cloud.**
- **06:54Z** ⛔ **Consequence 1 — the programme's own kernel energy is the wrong instrument for
  this content, and its null is already banked.** `gen_landscape.AtomLandscape` defines
  `E_store = Σ_k −σ_k² logsumexp_i(−‖x_k−a_ki‖²/2σ_k²)`, i.e. the atoms as **points**. On this
  content those "points" are 63 unit basis vectors and a mean, so that energy's minima sit where
  no feature lives. Charter §E.4 already banks the answer: *the density-over-atoms descent
  regresses to seen combinations (launch 0.685 → settled 0.211)*. ⛔ I will not spend a scored run
  rediscovering it. Recorded as a cited known null, not re-measured.
- **06:56Z** ⛔ **Consequence 2 — an in-well settle on this content is an ALGEBRAIC IDENTITY, not
  an empirical tie.** The honest store energy for a basis is distance to its span,
  `E_store,k(x_k) = ½ dist²(x_k, span(G_k))`. The request enters as a well at the address book's
  own point, `E_req,k = ½‖x_k − a_k‖²`. But `TensorAddress.read` renders through the orthonormal
  basis `U_k` of `rowspace(G_k)`, so **`a_k ∈ span(G_k)` by construction** — both terms are
  minimised at the same point and the settled state *equals* the static read. Settling within a
  well cannot move anything here, for reasons of geometry rather than of measurement.
  ⭐ **Therefore the only place dynamics can act on this content is ACROSS wells.** That fixes
  the arm.

### ⚠ CORRECTION to the 06:38Z ceiling bullet

- **07:02Z** ⚠ My 06:38Z bullet called `oracle_content_pool` a **ceiling** on any in-span read.
  **That is too strong and I withdraw the word.** It is the **L2-optimal** in-span read (Eckart–
  Young minimises squared reconstruction error). The metric here is *distinct valid novel tuples
  per corner* under an external classifier, which rewards **diversity across a corner's rows** and
  is not monotone in L2. w5's own table proves the gap is real: on **seed 1, row 4,
  `pca_tensor` BEATS `oracle_content_pool` by +0.449** — the L2-optimal in-span read is *not* the
  metric-optimal one.
- **07:02Z** ⭐ The **surviving, correct** statement, which still frames this hub: across the three
  seeds the L2-optimal read of this exact content sits at **−0.449 to +0.837** (row 4) and
  **+1.000 to +1.500** (row 3) relative to the incumbent. **The read dial on this content has
  roughly one tuple per corner of headroom in either direction — the same size as the decision
  margin (+1.0).** It is strong prior evidence for "decoration", and it is *not* a proof. The
  empirical run is therefore genuinely necessary, not decorative.

### The design, fixed before any code

- **07:20Z** **What "settle the state in the landscape" has to mean on this content**, given the
  two consequences above.
  1. The state is parameterised **in the span** — `x_k = U_k z_k`. That *is* the store's energy
     `½ dist²(x_k, span(G_k))`, enforced exactly instead of penalised. It also makes the state
     ~312-dimensional instead of 2048, which is what makes 49 corners × settle steps affordable.
  2. The **request** enters as a basin at the incumbent address book's own point:
     `E_req = ½β Σ_k ‖z̃_k − z̃_k^req‖²`. Request-only — `TensorAddress.coefficients` takes
     values and grid values, no feature.
  3. The **only** term that can move the state is therefore **cross-well**:
     `E_couple = −λ Σ_{k<l} z̃_kᵀ W_kl z̃_l`, the programme's own bilinear low-rank form
     (`gen_landscape.fit_low_rank_coupling`), re-derived in coefficient space: standardised
     cross-covariance of the **pool's** span coefficients, top-`r` SVD, each pair normalised by
     its own leading singular value. ⛔ Label-free, pool-only.
  4. The bilinear form is **indefinite**, so the coercivity certificate is mandatory, not
     optional (`gen_landscape.coupling_stability` records `E = −1.6e28` on an uncertified chain).
     In coefficient space `D = (β/2)I` and `S_kl = ‖W_kl‖₂/2`, so
     `λ* = 1/ρ(D^{−1/2} S D^{−1/2})`; the used strength is `min(declared, 0.9 λ*)` and it is
     reported whenever the certificate binds.
- **07:22Z** ⭐⭐ **How the dial is isolated so the answer cannot be confounded by parameters.**
  The coupling is fitted, so it carries declared read-side bytes beside the address book (the
  same rule w5 used for `address_bytes`; **the store stays at 524,288 B, untouched**). A win for
  the settled arm over `pca_tensor` alone would be ambiguous — coupling parameters, or dynamics?
  So the hub's dial is measured by a **parameter-identical** contrast:
  - `pca_settle_zerostep` — 0 settle steps ⇒ must be **bit-identical** to `pca_tensor`.
  - `pca_settle_onestep`  — **one** gradient step, same `(β, λ, W)`.
  - `pca_settle`          — settled to convergence, same `(β, λ, W)`.
  ⭐ `pca_settle` − `pca_settle_onestep` is **the relaxation**, at identical parameters and
  identical bytes. That contrast *is* the dial and nothing else can explain it.
- **07:24Z** ⚠ **A property of this landscape I am declaring in advance, because it is a result
  and not a convenience.** A quadratic request basin plus a bilinear coupling is a *quadratic*
  energy, so its settled state is the closed-form linear map
  `z̃* = (βI − λW)^{−1} β z̃^req`. **On content that is a basis, the landscape is necessarily
  quadratic and "settling" is therefore an iterative solver for a linear operator.** That is not
  a strawman choice — it is what the store *is*. Any non-convexity I could add would have to come
  from somewhere other than the deployed atoms, which would change the content.
- **07:25Z** The one non-convex, genuinely store-grounded alternative is the atom-**kernel**
  energy in channel space (`AtomLandscape`), and charter §E.4 has already banked its null
  (density-over-atoms descent, launch 0.685 → settled 0.211). ⛔ No scored run on it.
  ⭐ I will still run it on **val only** (free) as `pca_settle_kernel`, to confirm the banked
  null holds on *this* content rather than assert it.

### The module, and the no-op proof

- **08:20Z** Wrote `chlu/experiments/exp_w6_compose_physics.py` — additive at runtime on
  top of `exp_w5_compose_field`, which is additive on `exp_gen_read`. `install()` asserts
  every pre-existing arm's store, verb and class are unchanged and that `ARMS` is not
  reordered; it passed (44 arms, `pre_existing_unchanged=True`). ⛔ Nothing edited on disk.
  Launcher `scripts/w6_compose_physics_leg.sh`, mirroring w5's, including the refusal to
  run `test` without a PREREG file, a `/c/chlu/SCORED-LEDGER.md` row and the generation-2
  cache root.
- **08:32Z** ⭐⭐ **The no-op proof passed.** `pca_settle_zerostep` is asserted in code
  (not hoped for) to be **bit-identical** to `TensorAddress.read` on 256 requests —
  `np.array_equal`, which raises if it fails. It did not raise, and on val it scores
  **r4 = 64.375, exactly `pca_tensor`'s 64.375**. ⭐ So the settle harness is a proven
  no-op at zero steps and every difference below is the dynamics and nothing else.
- **08:33Z** The coercivity certificate **bound**, as it was designed to. Declared
  `lambda = 1.0` gives certificate min eigenvalue **−1.5** — unbounded below, the
  landscape would not settle, it would diverge. `lambda* = 0.250000`, so the strength
  actually used is `0.9 x 0.25 = 0.225000` and the clip is recorded in
  `span_coupling_r16.json`. All ten pair spectral norms are 1.0000 by the
  per-pair normalisation, exactly as intended.
- **08:33Z** Byte ledger of the arm, all three components printed separately and never
  netted: **store 524,288 B (unchanged, matches the pin exactly)** + address book
  191,424 B (the incumbent's own) + coupling 79,872 B + standardisation 2,496 B.
  ⚠ **The settled read therefore costs 82,368 B of read-side bytes that the incumbent
  does not pay.** Declared here, and it is why the dial is measured by the
  parameter-identical zero/one/converged contrast rather than by the arm alone.

### ⭐⭐ THE VAL SELECTION SWEEP — settling loses, monotonically, at every setting

- **08:47Z** Swept the whole `(lambda, steps)` family on **val** (24 val corners;
  selection only, charter A.2, costs no scored run), plus a coupling-rank sweep. Sorted
  by how far the state actually moves:

  | leg | median displacement | val r4 | vs incumbent |
  |---|---|---|---|
  | `pca_tensor` / `pca_settle_zerostep` (λ=0) | 0.0000 | **64.375** | — |
  | λ = 0.01 × safe, converged | 0.0018 | 64.375 | 0.000 |
  | λ = 0.05 × safe, converged | 0.0092 | 64.125 | −0.250 |
  | λ = 0.10 × safe, converged | 0.0190 | 63.917 | −0.458 |
  | λ = 0.25 × safe, 3 steps | 0.0269 | 63.625 | −0.750 |
  | λ = 1.00 × safe, **ONE step** | 0.0451 | 62.375 | −2.000 |
  | λ = 0.25 × safe, converged | 0.0514 | 61.625 | −2.750 |
  | λ = 1.00 × safe, 3 steps | 0.1186 | 54.875 | −9.500 |
  | λ = 0.50 × safe, converged | 0.1205 | 53.583 | −10.792 |
  | λ = 1.00 × safe, 10 steps | 0.2696 | 34.708 | −29.667 |
  | **`pca_settle`** (λ = 1.00 × safe, converged) | 0.4195 | **22.458** | **−41.917** |

  ⭐⭐ **The curve is monotone decreasing in displacement. Every amount of settling costs
  composition, and the cost grows with how far the state moves. The maximum of the whole
  family is at the λ → 0 limit, which IS the incumbent's static read.**
- **08:47Z** **The coupling's capacity is not the explanation.** Converged, at
  `lambda = safe`: rank 4 → **22.208**, rank 16 → **22.458**, rank 64 → **22.750**. A
  16× change in coupling capacity moves the result by 0.5 tuples/corner while settling
  itself costs 42. ⛔ "You built the coupling too small/too big" is closed.
- **08:52Z** ⛔ **The banked null confirmed on THIS content, not asserted.**
  `pca_settle_kernel` — the atom-**kernel** (density-over-atoms) descent of charter §E.4,
  in channel space, launched from the incumbent's read — scores **val r4 = 0.000** with
  judge validity **0.0000**. On content that is an orthonormal basis, the kernel energy's
  minima are the unit atoms themselves and the read is destroyed outright. ⛔ No scored
  run spent on it, as §E.4 requires.

### The mechanism: settling walks the state OFF the request

- **08:58Z** Decomposed the loss (`scripts/w6_settle_mechanism.py`, reads artifacts only,
  scores nothing; full table in `dev/mechanism.txt`). There are two ways to lose distinct
  valid novel tuples and they mean opposite things — validity falling (the state leaves
  the request) or novelty falling (the state regresses to seen combinations, §E.4's
  banked mechanism). The numbers are unambiguous:

  | leg | displacement | judge validity | distinct valid | novel |
  |---|---|---|---|---|
  | incumbent (λ=0) | 0.0000 | **0.8941** | 64.375 | 64.375 |
  | λ=1.00, one step | 0.0451 | 0.8663 | 62.375 | 62.375 |
  | λ=0.50, converged | 0.1205 | 0.7442 | 53.583 | 53.583 |
  | λ=1.00, 10 steps | 0.2696 | 0.4821 | 34.708 | 34.708 |
  | `pca_settle` converged | 0.4195 | **0.3119** | 22.458 | 22.458 |

  ⭐ **`distinct valid` and `novel` are equal to the last digit on every leg, and the
  entire loss is carried by the judge's validity rate falling 0.8941 → 0.3119.** The
  cross-well relaxation is not regressing to seen combinations — it is **trading request
  fidelity for joint typicality, and the external classifier punishes it one for one.**
  ⚠ Stated exactly: under `condition=val` the val corners' rows are removed from the fit
  pool, so every tuple in a val corner is novel *by construction* and the SEEN column is
  identically 0 for every leg including the incumbent. ⛔ The seen/novel split is
  therefore uninformative on val; it is only informative on `test`, and I will report it
  there rather than claim it now.
- **08:58Z** ⭐ **Charter §C.4 is cleared decisively, in the direction that matters.**
  The settled state's median distance to its nearest atom is **1.42× the atoms' own
  nearest-neighbour spacing** (min over wells; the incumbent's is 0.52×). Both sit
  strictly between atoms — ⛔ neither is a lookup table. Confirmed independently by
  `pca_settle_lookup` (nearest atom to the settled state): **val r4 = 0.000, validity
  0.0000.** The atoms of this store are a basis, so snapping to one destroys the read.

### Pre-registration and the scored run

- **09:14Z** ⚠ **Process note worth recording because it cost me a correction.** A path
  written as `/c/chlu/...` *inside a Python string literal* is not MSYS-mangled and
  Windows Python resolves it to `C:\c\chlu\...`. My first ledger append therefore created
  a stray `C:\c\chlu\SCORED-LEDGER.md` instead of writing the real one. Moved the row into
  `/c/chlu/SCORED-LEDGER.md` (verified present, 30 lines) and deleted the stray file.
  ⛔ **I did NOT delete the stray `C:\c\chlu\outputs\` directory beside it: it contains
  `w6-potentials/dev/s0/`, which is ANOTHER HUB'S work.** Left untouched. (Paths passed as
  shell *arguments* are mangled correctly, which is why every leg wrote to the right place.)
- **09:14Z** Wrote `PREREG-w6-compose-physics-1.md` and appended the
  `/c/chlu/SCORED-LEDGER.md` row **before** the run. **composition 2/3** — 1 spent in
  wave 5, **run 3/3 remains BANKED**. Every number from this run is quoted **scored 1/2**.
- **09:14Z** **Bars declared in advance**, with my expected direction recorded against my
  own dial: **B1** `pca_settle − pca_settle_zerostep` ≥ +1.0 (expected NEGATIVE);
  **B2** `pca_settle − pca_tensor` ≥ +1.0 (expected NEGATIVE); **B3** the relaxation
  contrast vs `pca_settle_onestep` (expected NEGATIVE); **TIE TEST** |δ| < 1.0 with the
  whole 95 % CI inside ±1.0; **C.1/C.4** vs `pca_settle_lookup`; **C.2** removed/shuffled
  must collapse. Margin **±1.0 novel tuple/corner**, both rows, all three seeds — the same
  margin wave 5 won under, so the two hubs are decided on one scale.
- **09:14Z** ⭐ **The rival set, the incumbent and every ceiling are NOT re-run.** Their
  `rows.npz`, plus each seed's `pca_content.pkl` and address-book pickle, are copied
  bit-for-bit from `/c/chlu/outputs/w5-compose-field/scored/test_seed{0,1,2}`. That is
  strictly stronger than re-running them: the settled read is compared against the table
  wave 5 **published**, by the same objects on the same corners. `rival_lace` and
  `rival_poe_ebm` stay **DECLARED NOT-RUN** with wave 5's own refusal text — ⛔ a dash is
  never a win for us. The scored launcher **refuses to start** if that copied leg is
  absent, because a table without the bar is not a result.
- **09:15Z** ⭐⭐ **The no-op proof passed on the HELD-OUT condition as well**, not only on
  val: `pca_settle_zerostep: BIT-IDENTICAL to the incumbent read on 256 requests`. So on
  the scored leg too, everything below is the dynamics and nothing else.
- **09:15Z** Launched scored seed 0 (PID 45926), `scripts/w6_compose_physics_scored.sh`,
  cache generation 2, 3 threads. Arms, in the order the launcher fixes: `zerostep`,
  `f05_conv`, `onestep`, `pca_settle`, `lookup`, `removed`, `shuffled`.

### The scored run — execution

- **09:19Z** ⚠ **First scored leg: ALL SEVEN of my arms FAILED, and the cause is worth
  recording.** `fw.run_arm` calls `delete_value_from_head` for **row 7** on every arm;
  `read_dial` routes to `delete_value_from_address` only when the verb
  `startswith("tensor")`. This hub's verb is `"settle"`, so row 7 fell through to the
  incumbent-head deleter and died on `KeyError: 'spec'` — ⛔ **after** rows 1–6 and all
  109 corners had already been scored. Renaming the verb would not have helped:
  `delete_value_from_address` carries the same guard internally. ⭐ The rivals and the
  incumbent were untouched (they are wave 5's copied artifacts), and every corner was on
  disk in `partial.npz`, so the loss was ~5 minutes per arm and **no scored number was
  affected** — that leg produced no held-out number for my arms at all.
- **09:20Z** Fixed **additively**: `delete_value_from_settle` zeroes the value's row of
  `E_f` exactly as the incumbent's deleter does, and `install()` patches
  `fw.delete_value_from_head` **before** `read_dial` captures it, so every other arm's
  route is untouched and neither `exp_gen_read` nor `exp_gen_factor_wells` is edited.
  ⭐ Added a **fit-time guard** that exercises the delete path on 8 requests, so this
  class of failure now costs seconds instead of a leg.
- **10:01Z** All three scored seeds complete, **zero arm failures**. Seed 0 07:12–07:19Z
  (resumed), seed 1 07:19–07:38Z, seed 2 07:38–08:01Z.
- **10:03Z** ⭐ **PREREG §9's void conditions all clear.** Every `pca_settle` chain
  **converged** on its own tolerance (117 / 129 / 123 steps against a 200 cap), ⛔ **none
  diverged** on any seed. `pca_settle_zerostep` passed its bit-identity assertion on every
  seed. The store ledger reads **524,288 B on every leg, tolerance 0**. No frozen harness
  file was edited.
- **10:18Z** ⭐⭐ **Charter §A.5 replication: PASSED, on a clean checkout, to the last
  digit.** `git clone --branch w6/compose-physics` into a separate directory; the
  instrument verified as `C:\msys64\tmp\claude\repl-checkout\chlu\__init__.py`; the
  coupling **refitted from scratch** (reproducing 78,848 B and `lambda_used=0.225000`,
  `lambda*=0.250000` exactly). Seed 0, row 4 / row 3: `pca_settle` **23.653 / 13.917**,
  `onestep` **60.878 / 28.333**, `f05_conv` **63.082 / 29.683**, `zerostep`
  **63.612 / 29.717** — every one identical to the scored leg.

---

# FINAL REPORT — w6 hub/compose-physics

**TARGET:** rows 3 and 4 with the PHYSICS read — settle the state in the landscape —
against `pca_tensor` and the full rival set.
**Scored 1/2.** PREREG: `PREREG-w6-compose-physics-1.md`. Table: `SCORED-TABLE.md`.
**Run 3/3 of the `composition` target remains BANKED.**

## ⛔⛔ THE VERDICT

**The dynamics do not earn their place in the composition story. They are worse than
decoration: at every strength at which they actually move the state they COST
composition, and the only setting at which they tie the incumbent is the setting at
which they do nothing.**

Row 4 (49 n-way corners), mean paired delta against `pca_tensor`, all three seeds, beside
the median relative displacement the dynamics actually produce:

| leg | displacement | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|---|
| `pca_settle_zerostep` | 0.0000 | **0.000** | **0.000** | **0.000** | bit-identical to the incumbent |
| `pca_settle_f05_conv` | 0.009 | −0.531 | −0.510 | −0.388 | inside the margin, but **signed negative on every seed** |
| `pca_settle_onestep` | 0.046 | −2.735 | −3.796 | −2.959 | **loses** |
| **`pca_settle`** (THE ARM) | 0.39 | **−39.959** | **−46.918** | **−39.061** | **loses catastrophically** |

Row 3 (60 pair sub-corners): `f05_conv` −0.033 / −0.183 / −0.033 (**a TIE on all three
seeds**); `onestep` −1.383 / −1.617 / −0.983; **`pca_settle` −15.800 / −25.533 / −20.350**.

**Every pre-registered bar came out NEGATIVE, exactly as I declared in advance that I
expected it to:**

| bar | comparator | rows 4 / 3, all 3 seeds | outcome |
|---|---|---|---|
| **B1** | `pca_settle_zerostep` — ⭐ the dial, dynamics ON vs OFF | −39.96/−46.92/−39.06 · −15.80/−25.53/−20.35 | ⛔ **NOT met — NEGATIVE** |
| **B2** | `pca_tensor` — the incumbent and the bar | identical to B1 (zerostep *is* `pca_tensor`) | ⛔ **NOT met — NEGATIVE** |
| **B3** | `pca_settle_onestep` — the relaxation | −37.22/−43.12/−36.10 · −14.42/−23.92/−19.37 | ⛔ **NOT met — NEGATIVE** |
| **TIE** | on `pca_settle` | every 95 % CI lies far below −1.0 | ⛔ **NOT a tie — it loses** |

⚠ **And it loses while costing MORE.** The settled read pays **81,312 B** of read-side
bytes (coupling 78,848 + standardisation 2,464) that the incumbent does not pay, on top of
the same 524,288 B store and the same 189,344 B address book. ⛔ **More bytes, fewer
tuples.** There is no ambiguity left for a parameter argument to occupy.

## ⛔ THE CONSEQUENCE, STATED AS THE TASK FILE REQUIRES AND NOT SOFTENED

The task file fixed what follows from this result, and it follows:

> **the dynamics are decoration on composition, and the paper says so in those words.**

⚠ On this evidence the paper can put it **more strongly than "decoration"**: on the
content that actually wins rows 3 and 4, settling is not a neutral ornament, it is a **net
cost**, monotone in how far it moves the state. The composition result belongs to **what is
stored and how it is addressed** — the per-well principal subspace and the multiplicative
address book — and **not to the physics.**

⛔ **The physics must therefore carry rows 2, 5, 6 and 7, and wave 5 already measured that
most of that ground is gone:**
- **rows 5 and 6 are CLOSED** — an 8-byte presence+bracket rule ties the field store at
  (1.0000, 1.0000) on both; `table_absence` **is** `bitmap8`.
- **row 2 belongs to a generic learner** — a size-matched continuous MLP wins on every
  seed, in two independent implementations.
- That leaves **row 7**, where wave 5 found the coupling hole exact and free (refuses
  `red ∧ cylinder` at 1.0000 for 1 byte, 0.0000 composition cost, no refit) while eviction
  is not free. ⭐ **Row 7 is the only row on which the programme's physics still has an
  unclaimed position, and this hub's result narrows the case for the dynamics down to it.**
  ⛔ I am not claiming row 7 — `w6/retention` holds the deletion budget.

## Why it loses — the mechanism, measured

⭐ On both conditions and for **every** leg including the incumbent, the `SEEN` column is
identically **0**: under `val` the corners' rows are removed from the fit pool, and under
`test` the row-4 corners are Montero-shaped combinations the condition never emitted.
**So `distinct valid` equals `novel` by construction and the entire metric is carried by
the judge's validity rate.** Seed 0, held out:

| leg | displacement | judge validity | distinct = novel |
|---|---|---|---|
| `pca_tensor` / `zerostep` | 0.0000 | **0.8835** | 63.612 |
| `f05_conv` | 0.0085 | 0.8761 | 63.082 |
| `onestep` | 0.0418 | 0.8455 | 60.878 |
| `pca_settle` | 0.3284 | **0.3285** | 23.653 |

⛔ **The settled read is NOT regressing to seen combinations** — charter §E.4's banked
mechanism — **and it cannot be: the scored corners contain no seen combinations to regress
to.** It is **walking the state off the request**, trading request fidelity for cross-well
typicality, and the external classifier charges it one tuple for one.

⭐ **The deeper reason, which is the part worth putting in the paper.** The incumbent's
content is an orthonormal **BASIS** — one mean vector plus 63 unit directions per well,
nearest-neighbour atom spacing exactly √2. The honest store energy for a basis is distance
to its span, and `TensorAddress.read` renders *through that span's own orthonormal basis*,
so **the addressed point already sits at the joint minimum of the store energy and the
request energy. An in-well settle on this content is an algebraic identity, not an
empirical tie.** Dynamics can therefore only act across wells — and a quadratic request
basin plus a bilinear coupling is a quadratic energy whose settled state is the closed-form
linear map `z̃* = (βI − λW)⁻¹ β z̃^req`. **On content that is a basis the landscape has one
basin, and "settling" is an iterative solver for a linear operator.** The content that wins
composition is content whose landscape has no interesting dynamics to have.

## What I did to try to make the physics win, before concluding this

⛔ So that the negative is not read as a strawman:
1. **Swept the whole `(λ, steps)` family on val** (charter A.2, no scored run spent):
   λ ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 1.00} × the certified-safe strength, steps ∈
   {1, 3, 10, converged}. **Monotone decreasing in displacement at every point**; the
   family's maximum is the λ→0 limit, which *is* the incumbent's static read.
2. **Swept the coupling's capacity**: ranks 4 / 16 / 64 give val r4 **22.208 / 22.458 /
   22.750**. A 16× capacity change moves the result by 0.5 tuples while settling costs 42.
   ⛔ "The coupling was the wrong size" is closed.
3. **Promoted the gentlest non-trivial rung into the scored ladder**, so the family's
   strongest possible case for a tie was put on the held-out set rather than argued for.
4. **Confirmed the banked null rather than asserting it**: `pca_settle_kernel` — the
   atom-kernel (density-over-atoms) descent of §E.4 — scores val r4 **0.000** at validity
   **0.0000** on this content. ⛔ No scored run spent on it.

## Charter §C floor checks — one line each

- **C.1 (must beat the matched-bytes lookup):** ⭐ **MET.** `pca_settle` −
  `pca_settle_lookup` (nearest atom to the settled state, same 524,288 B) = **+23.653 /
  +17.408 / +28.592** on row 4 and **+13.917 / +6.083 / +11.100** on row 3, every bootstrap
  CI strictly positive. ⚠ Clearing the floor does not make the arm a result: it still loses
  to the incumbent.
- **C.2 (landscape-removed / landscape-shuffled must collapse):** ⭐ **MET.**
  `pca_settle_removed` and `pca_settle_shuffled` both score **0.000** on both rows on all
  three seeds.
- **C.3 (the "tie is the win" clause):** **not invoked.** ⛔ There is no tie to claim — the
  arm loses.
- **C.4 (a settled state on its nearest atom is a lookup):** ⭐ **MET, decisively and in the
  direction that matters.** The settled state's median distance to its nearest atom is
  **2.65 / 2.54 / 3.13 ×** the atoms' own nearest-neighbour spacing (the incumbent's is
  1.13 / 1.48 / 1.69 ×). ⛔ **Neither is a lookup table.** Confirmed independently: snapping
  the settled state to its nearest atom gives **0.000** at validity **0.0000**.

## The estimand — not shrunk

⭐ **Row 4 = 49 n-way corners, row 3 = 60 pair sub-corners**, corner as the unit, mean
paired difference, bootstrap 95 % CI, three seeds, cache **generation 2**, byte pin
**524,288 B at tolerance 0**. ≥ 49 paired corners is met by row 4 exactly and exceeded by
row 3. ⛔ **Settle time never threatened the estimand** — the state is ~312-dimensional
because it is parameterised in the span, so a converged settle costs about what the static
read costs (a full leg is ~20 min for seven arms). **No NOT-RUN was needed on this hub.**

## Labels

Used **once at fit time** to align the wells and to ground the address book, and at the
interface. ⭐ **The content is unsupervised, and so is the coupling** — it is a second
moment of the fit pool's own span coefficients, with no label and no held-out row in it.

## Cost line

| item | cost |
|---|---|
| wall clock | **06:26Z → 08:30Z ≈ 2 h 04 m**, inside the ≤10 h hub budget |
| compute | local CPU only, **3 threads**, no GPU, no cluster leg |
| scored runs | **1 of 2** spent (`composition` 2/3); ⭐ **run 3/3 BANKED** |
| val selection | 17 val legs over 24 corners — charter A.2, **no scored run spent** |
| scored legs | 3 seeds × 7 arms, ~20 min/seed |
| replication | 1 clean-checkout leg, 4 arms, ~13 min |
| rivals / incumbent / ceilings | **0** — reused bit-for-bit from wave 5's scored legs |
| GPU justification lines | **none needed; no GPU used** |

## What a reader should take from this hub

1. ⛔ **Settling does not add to composition on the content that wins composition. It
   subtracts, monotonically in how much settling you do.** Replicated on a clean checkout.
2. ⭐ **The no-op proof is what makes that attributable.** Zero settle steps is
   **bit-identical** to `pca_tensor` by assertion, so the ladder isolates the dynamics and
   nothing else — same content, same address book, same coupling, same bytes.
3. ⭐ **There is a structural reason, not merely a measurement.** Content that is a basis
   shapes a one-basin quadratic landscape and its settled state is a linear operator.
   **You cannot get interesting physics out of content chosen to be optimal in L2.**
4. ⚠ **The honest open question this leaves**, stated as a question and not as a result:
   whether content with a genuinely multi-basin landscape could both win rows 3 and 4 *and*
   give the dynamics something to do. ⛔ This hub cannot answer it — answering it means
   moving the content dial, and the wave-4 lesson is that moving two dials at once
   attributes nothing. ⛔ It is **not** a claim, and nothing here should be read as one.

---

## Closing verification (process rules 1–3)

- **08:35Z** ⛔ **No box-wide kill was ever issued.** Every PID I launched was captured at
  launch in `scored_pids.txt` and polled to completion **inside a turn**; all are
  confirmed finished. Three python processes remain on the box and belong to other
  agents — ⛔ **untouched.** I also left the stray `C:\c\chlu\outputs\w6-potentials\`
  directory alone once I saw it was another hub's work.
- **08:35Z** **Code verified by `git status` + `git bundle verify`.** Working tree
  **clean**; HEAD `25fc108` on `w6/compose-physics`; bundle written to
  `w6-compose-physics.bundle` (4,871,680 B), `git bundle verify` reports **"The bundle
  records a complete history"**, and `refs/heads/w6/compose-physics = 25fc108` is present
  in its head list.
- **08:35Z** **Outputs verified by directory listing** (⛔ `git status` cannot see them —
  they are gitignored and live outside the worktree): `/c/chlu/outputs/w6-compose-physics`
  = **283 MB**, holding `START_UTC.txt`, `LOG.md`, `PREREG-w6-compose-physics-1.md`,
  `SCORED-TABLE.md`, the three scored legs (**30 arms / 27 `rows.npz` / 4 `READ-TABLE*.json`
  each**), the replication leg (27 arms), four `mechanism_*.txt`, the dev/val leg, every
  run log, and the bundle.
- **08:35Z** Both `/c/chlu/SCORED-LEDGER.md` rows are present for this hub — the
  **pre-run** row (written before the run) and the **outcome** row. Composition is
  **2/3 spent; run 3/3 BANKED.**
