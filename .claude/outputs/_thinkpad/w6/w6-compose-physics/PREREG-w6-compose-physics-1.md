# PREREG — w6 hub/compose-physics — scored run 1 of 2

**Written 2026-09-20, BEFORE any held-out number for this hub exists.**
Charter §A.1/§A.3: the bars below are fixed here and are never edited. Amendments are
appended and dated.

**Hub:** `w6/compose-physics` · **Target:** the composition target, rows 3 and 4.
**Scored budget:** the composition target, 1 spent in wave 5, **2 remaining; this is 1 of 2.**
Every number produced by this run is quoted **"scored 1/2"**.

---

## 1. The question, and the dial

**Does SETTLING add anything to composition, over a static read on the same content?**

**Dial: the READ, and only the read.** The content is `pca_tensor`'s — the per-well
leading principal subspace of the fit pool, 64 atoms × 2048 float32 = **524,288 B**,
unsupervised. It is built by the incumbent's own function
(`exp_w5_compose_field.build_pca_content`) and its address book is the incumbent's own
**cached pickle, reused byte for byte**, not a refit. ⛔ Nothing about the content moves.

## 2. What is being scored, and what each leg is for

One leg per seed on `condition=test`, cache **generation 2**, seeds **0, 1, 2**.

### The dial ladder — all four share ONE fitted coupling and identical `(β, λ, W)`
| leg | settle steps | what it isolates |
|---|---|---|
| `pca_settle_zerostep` | 0 | ⭐ asserted **bit-identical** to `pca_tensor`; the no-op proof |
| `pca_settle_f05_conv` | converged at **λ = 0.05 × certified-safe** | the gentlest non-trivial dynamics |
| `pca_settle_onestep` | 1 | one step of relaxation |
| `pca_settle` | converged at **λ = certified-safe** | ⭐ **THE ARM**: the full physics read |

⭐ `pca_settle` − `pca_settle_onestep` − `pca_settle_zerostep` is a **parameter-identical**
ladder: same content, same address book, same coupling, same bytes. The only thing that
varies is **how far the state is allowed to relax.** That contrast *is* this hub's dial
and no parameter difference can explain it.

### Charter-§C family, on the same atom block
`pca_settle_removed` (C.2, atoms zeroed) · `pca_settle_shuffled` (C.2, atom values
permuted) · `pca_settle_lookup` (C.1/C.4, nearest atom to the **settled** state).

### The bar and the rival set
`pca_tensor` (**the incumbent and the bar**) and the **full rival set** — `rival_table`,
`rival_mlp`, `rival_mlp_180k`, `rival_cont_mlp`, `rival_transformer`, `rival_lstm`,
`rival_lace`, `rival_poe_ebm` — plus `span_head` and the ceilings
(`oracle_content_pool`, `oracle_span_field`, `oracle_content_items`,
`oracle_content_fine`, `oracle_true_feature`, `oracle_span`).

⭐ **These are not re-run.** Their `rows.npz`, and the per-seed address book and content
pickle, are **copied bit-for-bit from wave 5's own scored legs**
(`/c/chlu/outputs/w5-compose-field/scored/test_seed{0,1,2}`). That is deliberate and it
is strictly stronger than re-running them: it guarantees the rivals, the incumbent and
the ceilings are the *same objects on the same corners* that wave 5 reported, so the
settled read is compared against the published table and not against a re-derivation of
it. ⛔ `rival_lace` and `rival_poe_ebm` remain **DECLARED NOT-RUN** with wave 5's own
refusal text, printed as a dash and a footnote — ⛔ never a win for us.

## 3. The estimand

**Row 4 = 49 n-way corners. Row 3 = 60 pair sub-corners.** The unit is the **corner**;
the statistic is the **mean paired difference** across corners; the interval is the
bootstrap 95 % CI over corners. ⛔ A single cell never decides a target.
⭐ ≥ 49 paired corners is met by row 4 exactly and exceeded by row 3. **The estimand is
not shrunk.** Rivals are scored on the **same** corners.

**Margin: ±1.0 distinct valid novel tuples per corner**, on both rows — the same margin
wave 5 pre-declared and won under, so the two hubs are decided on one scale.
`beats` requires the mean paired delta to clear the margin **and** the bootstrap 2.5 %
quantile to clear it too, on **all three seeds**.

## 4. The bars, and the expected direction — declared BEFORE the run

| # | comparison | bar | expected direction | what it would mean |
|---|---|---|---|---|
| **B1** | `pca_settle` − `pca_settle_zerostep` | ≥ **+1.0**, rows 3 and 4, all 3 seeds | ⚠ **NEGATIVE** | B1 met ⇒ **the physics is load-bearing and the paper leads with it** |
| **B2** | `pca_settle` − `pca_tensor` | ≥ **+1.0**, rows 3 and 4, all 3 seeds | ⚠ **NEGATIVE** | B2 met ⇒ settling beats the incumbent outright |
| **B3** | `pca_settle` − `pca_settle_onestep` | reported; sign is the relaxation's | ⚠ **NEGATIVE** | isolates relaxation from the coupling's parameters |
| **T** | **TIE TEST** | \|delta\| < 1.0 **and** the whole 95 % CI inside ±1.0, rows 3 and 4, all 3 seeds | — | T met and B1/B2 not met ⇒ ⛔ **the dynamics are decoration on composition** |
| **C1** | `pca_settle` − `pca_settle_lookup` (C.1/C.4) | > 0 beyond seed spread | positive | otherwise the settled state is a lookup and is reported as one |
| **C2** | `pca_settle_removed`, `pca_settle_shuffled` | must **collapse** | ≈ 0 | charter C.2 |

⭐ **The honest prior, stated here so the result cannot be presented as a surprise.** The
val selection sweep (24 val corners, charter A.2, no scored run spent) found the whole
`(λ, steps)` family **monotone decreasing in how far the state moves**: 64.375 at λ→0
down to 22.458 fully converged, with coupling rank irrelevant (22.2 / 22.5 / 22.8 at
ranks 4 / 16 / 64). **I expect B1, B2 and B3 all to come out NEGATIVE, and I expect the
tie test T to FAIL in the losing direction on `pca_settle`.** I am scoring it anyway
because the task file states plainly that *a settled read that costs composition is a
finding*, and because the ladder is what turns "it lost" into "it lost monotonically in
the amount of dynamics, at fixed parameters and fixed bytes".

⚠ **If instead the tie test T passes** on the gentlest leg (`pca_settle_f05_conv`) while
B1 and B2 fail, the verdict is **decoration**, and it is reported in those words.

## 5. The byte ledger, declared and never netted

| component | bytes | note |
|---|---|---|
| store (atoms) | **524,288** | ⭐ the pin, tolerance 0. **Unchanged from the incumbent.** |
| address book | 191,424 | ⭐ the incumbent's own, reused byte for byte |
| coupling `U, V` | 79,872 | ⚠ **added by the settled read; the incumbent does not pay it** |
| coupling standardisation | 2,496 | ⚠ counted and printed, **not declared free** |

⚠ The settled read costs **82,368 B of read-side bytes the incumbent does not pay.** That
is exactly why B1/B3 — the parameter-identical ladder — and not B2 alone carry the
verdict. Parameter matching is secondary to the byte pin and cannot hold simultaneously;
both are printed.

## 6. Labels

Used **once at fit time** to align the wells and to ground the address book, and at the
interface. ⭐ The **content is unsupervised**, and so is the coupling: it is a second
moment of the fit pool's own span coefficients, with no label and no held-out row in it.

## 7. The coercivity certificate

⛔ The bilinear coupling is **indefinite**; an uncertified chain does not settle, it
diverges (`gen_landscape.coupling_stability` records `E = −1.6e28`). Declared `λ = 1.0`
has certificate min eigenvalue **−1.5**; `λ* = 0.250000`; the strength used is
`0.9 × λ* = 0.225000`, and the clip is recorded in `span_coupling_r16.json`. ⛔ A
diverged chain's energy trace is not a measurement and would void this run.

## 8. Replication

Charter §A.5: a separate spoke re-runs the winning configuration from this file on a
clean checkout of `w6/compose-physics`. **Unreplicated = not a result.**
Command: `scripts/w6_compose_physics_scored.sh <seed>`.

## 9. What would make this run void

- The certificate not holding, or any chain reporting `diverged`.
- `pca_settle_zerostep` failing its bit-identity assertion against `TensorAddress.read`.
- The store ledger moving off 524,288 B on any leg.
- Any edit to a frozen harness file (charter §A.4). ⛔ None is made; this hub is additive.
