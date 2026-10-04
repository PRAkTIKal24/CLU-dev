# EVIDENCE BASE — the paper's extended physics section

**w10 `spoke/physics-section-evidence` · agent `physics-theorist` · 2026-09-21.**
**Scored runs spent: ⛔ NONE (mandated).** Everything below is read out of the existing record, plus
one local numerical reproduction battery (`checks/derivation_checks.py`, pure stdlib, no `chlu`
import, no production code touched).

**Evidence base:** `/c/chlu/evidence/outputs/` (+ `claims_matrix.md`, `negative_results.md`).
⛔ `$CHLU_CLAUDE_ROOT` **not read.** Paths below are relative to `/c/chlu/` unless absolute.

---

## 0. HOW TO READ THIS FILE — the labels, and what they cost

Every physical claim carries **(a)** its derivation, **(b)** the measurement that confirms or refutes
it, and **(c)** one status. The statuses are:

| label | means |
|---|---|
| **CONFIRMED [CHLU]** | the derivation reproduces numerically **here** (§4) **and** it was measured on **this programme's field store**, held out, pre-registered, ≥ 3 seeds. |
| **CONFIRMED [shorts]** | the derivation reproduces numerically here **and** V1/V2/V5 report a confirming measurement. ⚠ **The programme never re-ran those measurements** — they were read out of the PDFs by `spoke/shorts-inventory`. Provenance is a paper, not an artefact in this tree. |
| **REFUTED** | the derivation fails numerically, **or** a measurement in this tree contradicts it. The failing step is named. |
| **UNTESTED** | reproducible as algebra; **no measurement anywhere in the evidence base tests it.** ⭐ An untested claim is welcome here and is labelled, not hidden. |
| **DEFINITION** | not a falsifiable claim (e.g. D15 defines the spectral mass). |

⛔ **Two provenance facts a writer must not lose.**
1. `evidence/claims_matrix.md` (v2.17) and `evidence/negative_results.md` (N1–N313) are
   **Campaign-2 / Campaign-3 accounting and end at the C2W11 fold.** ⛔ **Neither contains a single
   wave-5 or wave-6 entry** — `grep -in "gym-1|gym-2|illegal" evidence/negative_results.md` returns
   **0 hits**. The w5/w6 record lives in `evidence/outputs/w{5,6}-advisor-report.md` and
   `evidence/outputs/_thinkpad/w{5,6}/**`. Nothing in §5 of this file is drawn from the registries.
2. The `_thinkpad/w{5,6}/...` trees under `evidence/outputs/` are the wave records. Where a spoke's
   live tree also exists at `/c/chlu/outputs/<spoke>/` the two are byte-equal on the files I read;
   I cite the `evidence/` copy throughout.

---

## 1. ⛔ THE SCOPE LIMIT THAT MUST SURVIVE INTO THE PROSE

> ### **The CRR predicts a settle half-life. It supports no scoreboard claim.**

The Curvature–Retention Relationship is a statement about **how fast a perturbation of the field
store relaxes**. It is not a statement about how many valid novel tuples the store composes, about
refusal, about deletion, or about any row of the taxonomy. The section must say what it predicts
and stop.

This is not a stylistic preference. It is the shape of wave 6's own result, and the Advisor was
asked to rule on exactly this
(`evidence/outputs/w6-advisor-report.md` §6.2: *"May the CRR result carry a section? It is the
programme's cleanest theory-meets-measurement result. ⚠ It predicts settle half-life, not a taxonomy
row — so it supports no scoreboard claim."*).

⛔ **The same object refutes the scoreboard reading, measured, on the same seeds:**

| what the dynamics do to composition | mean paired Δ vs `pca_tensor`, row 4 (49 n-way corners), seeds 0/1/2 | median displacement |
|---|---|---|
| `pca_settle_zerostep` — bit-identical to the incumbent (`np.array_equal` asserted) | **0.000 / 0.000 / 0.000** | 0.0000 |
| `pca_settle_f05_conv` | **−0.531 / −0.510 / −0.388** | 0.009 |
| `pca_settle_onestep` | **−2.735 / −3.796 / −2.959** | 0.046 |
| **`pca_settle`** | **−39.959 / −46.918 / −39.061** | **0.39** |

Row 3: `pca_settle` **−15.800 / −25.533 / −20.350**. Every pre-registered bar negative, declared in
advance, and it loses **while paying 81,312 B more read-side**.
*Source:* `evidence/outputs/_thinkpad/w6/SCORED-LEDGER.md` (w6 `hub/compose-physics`, 2/3 SPENT row)
· `evidence/outputs/_thinkpad/w6/w6-compose-physics/LOG.md` §FINAL REPORT ·
`evidence/outputs/w6-advisor-report.md` §3.

⭐ **The one-sentence version for the paper:** *the shorts' retention law predicts this store's
settle half-life parameter-free and beats a sighted fitted rival 49/49 on three seeds; the same
dynamics, at every strength at which they move the state, **cost** composition. Both are measured on
the same object. The first is a dynamics result. The second is why it is not a scoreboard result.*

⚠ **And carry B4.** The half-life law **missed one of its six pre-registered bars on all three
seeds** (§3, B4). It is not a footnote; it is the bar that locates where the law stops.

---

## 2. THE SPINE — the CRR result, with its derivation and its measurement

### 2.1 What was predicted, and with what freedom

**The prediction, parameter-free, nothing fitted:**

```
n_half = ln2 / (−ln |λ_slow|),   λ_slow = the larger-modulus root of
λ² − (2−γ)(1 − h²/2) λ + (1−γ) = 0,   h = ε·μ,   ε = 0.05 inherited verbatim from the shorts
```

*Derivation:* **D3/D4** (the per-mode step matrix and its eigenvalues, V5 App. G.1/G.3; V2 App. C
Eqs. 3–4) closed with **D5** (V2 App. C Eq. 6). *Catalogued at*
`evidence/outputs/_thinkpad/w6/w6-shorts-inventory/shorts-experiments-inventory.md` §2.1–§2.2.

⭐ **Validated against the papers' own printed numbers BEFORE any data was touched:**

| check | the paper prints | the w6 formula gives | my re-derivation (§4) |
|---|---|---|---|
| V2 §4.1 underdamped floor at γ = 0.05 | **27.03** | **27.02681** | **27.02681** |
| V5 `λ_ret` at γ = 0.002 | **0.998999499** | **0.9989994995** | **0.9989994995** |
| V5 `n_half` at γ = 0.002 | **692.5** | **692.4538** | **692.4538** |

*Source:* `evidence/outputs/_thinkpad/w6/w6-potentials/LOG.md` §2 · my `checks/derivation_checks.out`
lines `D7a`, `D7c`.

The repo's own `velocity_verlet_step` **is** the shorts' map, asserted to **1.885e-08** on every leg
(tol 1e-5); the leg refuses to run otherwise. Confirmed in
`test/s{0,1,2}/results.json → integrator_check.max_abs_diff = 1.884681927188936e-08` on all three seeds.

### 2.2 The instrument floor, measured **before** the bar was fixed

On an exact quadratic — where the CRR is true *by construction* — the same measurement chain
recovers it to **median 0.0349 / p90 0.0749 / max 0.1601** (17 cells). Its worst cell sits at
`h/h* = 1.42`: **the exceptional point**. ⭐ **B1's 0.10 bar is ≈ 3× the instrument's own floor.**
*Source:* `test/s{0,1,2}/results.json → instrument_floor` (identical 0.03491890826972267 on all three
seeds) · `w6-potentials/LOG.md` §3.

### 2.3 ⭐⭐ THE RESULT

> **The Curvature–Retention Relationship predicts the field store's settle half-life,
> parameter-free, to a factor 1.10 at the median held-out corner — and beats a one-parameter rival
> FITTED WITH SIGHT OF THE HELD-OUT MEASUREMENTS on 49 of 49 corners, on every one of three seeds.**

**Estimand:** 49 held-out Montero n-way corners × 3 seeds, **192,975 rows** total
(64,335 / 64,320 / 64,320; of which **41,493 / 41,405 / 41,358** rows sit in the 49 *scored* corners,
the remainder in the 3 aux + 4 control corners). Cache **generation 2**. Byte pin **524,288 B,
tolerance 0, matched on every leg**. Field sha `697d60df… / 608fc3da… / fca1c4a8…`.

| bar | statistic | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|---|
| **B1** | median corner \|log₁₀ n_meas/n_pred\| ≤ 0.10 | **0.0433** | **0.0435** | **0.0431** | ⭐ **PASS ×3** (factor **1.105**; seed spread **0.0004**) |
| **B2** | mean paired Δ vs the FITTED one-parameter rival (< 0) | **−0.5968** | **−0.5796** | **−0.5982** | ⭐ **PASS ×3** |
| **B2** | corners better than the fitted rival (≥ 40/49) | **49** | **49** | **49** | ⭐ **PASS ×3** |
| **B3** | underdamped \|d log n / d log μ²\| ≤ 0.10 (median over γ) | 0.0357 | 0.0282 | 0.0343 | **PASS ×3** |
| **B3** | …max over γ (printed, the stricter reading) | 59.1158 | 0.1612 | 1883.2934 | ⚠ printed, bar not moved |
| **B3** | \|log₁₀ n_meas / (2ln2/−ln(1−γ))\| ≤ 0.10 | 0.0378 | 0.0378 | 0.0348 | **PASS ×3** |
| **B4** | V-curve argmin / γ_crit within ×2 | **0.3362** | **0.3301** | **0.3520** | ⛔ **MISS ×3** |
| **B4** | …against the EXACT inversion of h*(γ)=h | 0.3782 | 0.3784 | 0.3834 | ⛔ **MISS ×3** |
| **B5** | **D17** overdamped slope measured (derived **+1.1385**) | **+1.1368** | **+1.1367** | **+1.1368** | ⭐ **the sharpest number** |
| **B5** | …mean \|meas − derived\| (must beat naive) | **0.0038** | **0.0042** | **0.0040** | ⭐ **PASS ×3** |
| **B5** | …mean \|meas − naive +1\| | 0.1436 | 0.1430 | 0.1436 | **34–38× further** |
| **B5** | **D18** underdamped slope measured (derived ≈ −1.0050) | −1.1090 | −1.1090 | −1.1086 | ⚠ see below |
| **B5** | …mean \|meas − derived\| vs \|meas − naive −1\| | 0.1368 vs 0.1417 | 0.1277 vs 0.1324 | 0.1290 vs 0.1333 | ⛔ **thin on every seed** |
| **B6** | Hessian + D14 lift, **finite difference** ≤ 1e-9 | 3.519e-11 | 1.202e-10 | 2.699e-11 | **PASS ×3** |

*Source (every cell):* `evidence/outputs/_thinkpad/w6/w6-potentials/RESULTS-LEDGER.md`.
⭐ **B1, B2, the byte pin, the field sha, the integrator check, the instrument floor and B6 all
RECOMPUTE from the raw artefacts** `test/s{0,1,2}/results.json` — see `checks/derivation_checks.out`
lines `B1/B2` and `AUDIT`. The recomputed values are `0.043256 / 0.043516 / 0.043102` and
`−0.596831 / −0.579639 / −0.598172`, 49/49 on every seed.

**The fitted rival it beats.** `FIT1` = `n = C/(εμ)²` with `C` fitted **with sight of the held-out
measurements**: `C = 0.200699 / 0.206008 / 0.201184`, own median error **0.640132 / 0.623155 /
0.641281** in log₁₀. A fitted constant (`FIT2`) is beaten 49/49 as well.
*Source:* `test/s{0,1,2}/results.json → fit1_C`, `estimand.corners[*].median_err_fit1`.

### 2.4 ⭐ The sharpest number (D17), stated exactly

**Derivation (D17, V5 App. G.3):** `d ln n_half / d ln γ = 1 + γ/(2−γ)`, *"it approaches the +1
asymptote only as γ → 0 and **exceeds it at every finite γ**."*

**Reproduced here (§4, `D17a`):** `1 + γ/(2−γ)` **is** the exact log-derivative of the exact slow
root at fixed `h`. Derived vs numeric at `h = 1e-3`: γ=0.05 → 1.025641 vs 1.026347; γ=0.1 →
1.052632 vs 1.052950; γ=0.2 → 1.111111 vs 1.111168; γ=0.35 → 1.212121 vs 1.212030; γ=0.5 → 1.333333
vs 1.333266. **Reproduced here (`D17b`):** the slope exceeds +1 at every γ in (0.001, 1.999); the
minimum excess is 0.000500 and it is monotone increasing.

**Measured:** **+1.1368 / +1.1367 / +1.1368** against derived **+1.1385** — mean absolute error
**0.0038 / 0.0042 / 0.0040** against the textbook `+1`'s **0.1436 / 0.1430 / 0.1436**: **34–38×
closer on all three seeds.** The shorts derive that the slope *must* exceed +1 at finite γ. It does.

⚠ **STATUS: CONFIRMED [CHLU] — with one reproducibility caveat that is mine, not the record's.**
See §4.2. The **formula** reproduces exactly; the **constant +1.1385** does not reproduce from any
recorded machine-readable artefact, because the per-cell fit windows were not written to one.

### 2.5 ⭐ Where the law holds, and where it stops — resolved by `h/h*`

Held-out, seed 0 (`w6-potentials/LOG.md` §5):

| `h/h*` | n rows | median \|log₁₀ n_meas/n_pred\| |
|---|---|---|
| < 0.1 | 1,668 | **0.0008** (0.2 %) |
| 0.1 – 0.5 | 8,177 | 0.0264 |
| **0.5 – 2** | 14,452 | **0.1482** |
| 2 – 10 | 7,554 | 0.0454 |
| > 10 | 9,642 | 0.0262 |

⭐ **Deep on either branch the parameter-free prediction is good to 0.2 %, and ALL of the error sits
in the one bucket straddling the exceptional point** — where **D10** says the 2×2 map is defective
(one Jordan block, secular transients `~ n λⁿ`). ⭐ **The theory predicts where its own simplest form
stops.** My §4 check `D10` confirms the collision is exact (|λ₊−λ₋| = 2.1e-08, the square root of
double-precision cancellation) and `D11` confirms the square-root phase onset (exponent 0.500209 vs
derived 0.5).

### 2.6 ⛔ B4 — the miss, carried

**B4 missed on all three seeds: 0.3362 / 0.3301 / 0.3520 against a factor-2 bar** (and 0.3782 /
0.3784 / 0.3834 against the exact inversion of `h*(γ) = h`).

It is the same fact as §2.5 measured a second way, and the hub established that before reading it as
a failure of D16:

- ⭐ **On every seed the PREDICTED curve's own argmin over `γ*_exact` is 1.0000** — so the γ grid
  resolves the critical point and the shortfall is in the **rollout**, not the grid and not D16.
- **Why:** approaching the exceptional point the map is **defective** (D10), so the true envelope
  decays slower than `λⁿ`, the predicted dip fills in, and the measured minimum slides.
- The shorts measure D16 on **I-J, the exact one-step Jacobian**, where the measurement *is* the
  eigenvalue and the argmin sits on the crossover by construction. V5's own line:
  *"the instrument gap is a level, not a rate."*
- ⚠ **The miss was PRE-REGISTERED with its cause, in a dated amendment, before the run.**

*Source:* `w6-potentials/RESULTS-LEDGER.md` · `w6-potentials/LOG.md` §5 ·
`w6-potentials/PREREG-potentials-1.md` + Amendment 1 ·
`evidence/outputs/_thinkpad/w6/SCORED-LEDGER.md` (potentials outcome row).

### 2.7 ⭐⭐ The instrument was named in advance and it was worth an order of magnitude

V2 Fig. 1 reports **envelope** and **first-crossing** half-lives separately; the CRR predicts the
**envelope**. Deep underdamped the two differ by **9.24×**, and first-crossing's error is
**1.0088–1.0180** against the envelope's **0.0284–0.0296**.
⛔ **Scoring first-crossing would have reported the law as broken when the instrument was wrong.**
The scored instrument is the reverse cumulative maximum of `|x|` — monotone non-increasing by
construction, no fit and no window length to choose — and the amendment naming it was written before
any held-out number was seen.
*Source:* `w6-potentials/LOG.md` entries 2026-09-20T09:35Z and 06:59:14Z.

### 2.8 The controls, and the two traps cleared in advance

- **§C.2 satisfied in the SPECTRUM.** The CRR error is **identical across arms** (0.0428–0.0435 on
  `field` / `removed` / `shuffled` / `pca_tensor`) **because the law is about the dynamics — stated
  in advance.** Where the arms separate is the curvature spectrum: `field` **[0.2727, 1.0000]**,
  `shuffled` [0.7446, 1.0000], `pca_tensor` [0.9797, 0.9999], `removed` **[1.0000, 1.0000], no
  non-trivial curvature at all**.
- ⛔ **§C.1 DOES NOT APPLY and was not given a manufactured pass.** There is no lookup comparator for
  *"does a closed form predict a rollout"*, and **this hub claims no row against any rival.**
- **§C.4 measured, does not fire:** the settled minimum sits **1.014 / 1.224 / 0.938 / 0.326 / 0.736
  σ** from its nearest atom and the read draws on **30.9–59.5 of 64** atoms.
- ⚠ **The `knn_density` trap cleared with numbers on the held-out store:**
  `well_sigma / nn_median` = **0.805 / 0.895 / 0.900 / 1.759 / 1.207** — not 1.0 on any well, and on
  **both sides** of 1. The field store's width is a within-value pool spread, not an item spacing.
  (`knn_density` is the length scale that failed at **0.0203** as a refusal rival in w5 —
  `evidence/outputs/w5-advisor-report.md` §3. The failure is **not** inherited.)

### 2.9 Mechanism — ⛔ NOT scored, NOT a row

All five wells sit **below** the `Δ/σ = 2` bifurcation of `Hess E = I − Cov_w/σ²`
(**1.339 / 0.812 / 0.768 / 0.736 / 0.404**), and mean shift from 256 starts finds **one** minimum in
wells 1–4. ⚠ **Well 0 has two (225/31 starts) — the two-atom rule is reported MISSING there.** The
correct general criterion is the spectral one, `λ_max(Cov_w) < σ²`, which is what the scored
prediction uses. ⭐ The substantive statement survives: **no stored value owns a basin, so the store
interpolates rather than snapping.**

### 2.10 Replication and cost

⭐ **Bit-exact.** A clean checkout cloned from the bundle, asserting `chlu` resolves inside itself,
reproduced seed 0's **B1, B2, corners-better, store sha, bytes, fitted-rival C, integrator check and
B6 to the last printed digit** (`w6-potentials/replication/s0/`).
**Cost:** 1 h 55 m wall, local CPU, 3 threads. ⛔ No GPU, no cloud, no cluster.
⭐ **Runs 2/3 and 3/3 were RETURNED UNSPENT.**

### 2.11 ⭐ A SECOND, INDEPENDENT REPRODUCTION OF THE SAME LAW — `w6 hub/retention`

A different hub, a different target, a different code path (`gen_retention.self_check()` +
`verification_table()`), asserting the transcription against the papers' printed numbers before
anything may use it, and checking it against a **simulation of the paper's own map**:

| check | w6 `hub/retention` measured | the papers |
|---|---|---|
| floor at γ = 0.05 | **27.026815** | **27.03** |
| floor's curvature-independence | `1.6e-13` over the underdamped band | "a curvature-independent floor" |
| overdamped slope `log n_half` vs `log μ²` | **−1.00096** | predicted −1; V2 fitted −0.985, V5 fitted −1.006 |
| exceptional-point onset exponent | **0.50006** | predicted 1/2; V2 fitted 0.5165 |
| `h*` vs `γ/2` | rel. err `5.0e-4` at γ = 1e-3 | `h* = γ/2 + O(γ²)` |
| stability limit | `1 + tr + det = 0` to `3.3e-16` at `h = 2`, four γ | "`h = 2` exactly, for every γ" |
| `det A`, `tr A` | `1.1e-16`, `2.2e-16` | App. C Eq. (3) |
| latch | `ρ = 1.0` exact, `n_half = ∞`, displacement `7.8e-16` | "an infinite half-life" |
| **simulated envelope vs closed form** | rel. err **4e-13 – 8e-11 over ~7 orders of magnitude** of `n_half` | — |
| simulated floor past the crossover | **27.027 ± 0.03 over a 50× span in `h`** | 27.03 |
| first-crossing / envelope | 1.0000–1.0002 overdamped, **1.865× – 0.037× underdamped** | "agree to 0.01 %; diverge by up to 3.2×" |

⛔ **And one declared limit of the asymptotic form, from the same hub:** the overdamped asymptotic
`2γ ln2/[(2−γ)(εμ)²]` was **DECLINED as the operator** because it is **2.4 % off already at
`h = 0.30 h*`** and wrong by orders past the crossover. It is printed only as a comparator. ⭐ **The
paper must quote the exact-root form, not the asymptotic one.** (My §4 check `D5` measures the same
thing from the other side: the asymptotic is within 0.96 % only at `h ≤ h*/5`.)

*Source:* `evidence/outputs/_thinkpad/w6/w6-retention/LOG.md` §3.

---

## 3. THE CLAIM LEDGER — every derivation, its measurement, its status

Catalogue provenance: `evidence/outputs/_thinkpad/w6/w6-shorts-inventory/shorts-experiments-inventory.md`
(D1–D50 verbatim with paper+section in §2; A1–A8 in §3; the 62 experiments E1.1–E1.21 / E2.1–E2.19 /
E5.1–E5.23 in §5). **Reproduced-here** column cites `checks/derivation_checks.out`.

### 3.1 The shared map and the retention band structure (V2 §3 / App. C; V5 App. G.1–G.4)

| # | derivation | reproduced here | measurement | status |
|---|---|---|---|---|
| **D1** | `JᵀΩJ = (1−γ)Ω`, `det J = (1−γ)^d` — conformal symplecticity | — (structural) | V1 E1.1 landing rate **0** at every `d` with KE 0.72 < V_b 1; V2 E2.13 BIBO **1.00 vs 0.33**, E2.14 CLU holds a 0.20–0.23 plateau where the unconstrained twin diverges after ≈700 steps | **CONFIRMED [shorts]** |
| **D2** | position-gated form contributes exactly `(1 − γφ(q'))^d` | — | no measurement isolates the gated determinant | **UNTESTED** |
| **D3** | `tr J = (2−γ)(1−h²/2)`, `det J = 1−γ` | ✅ `D3/D4` (max residual **5.551e-17** over 20 cells) | w6-retention `det A` **1.1e-16**, `tr A` **2.2e-16`; V2 E2.1 Hessian↔Jacobian gap to **3.2e-10** over 70 tilt configs | **CONFIRMED [CHLU]** |
| **D4** | `λ_± = ½(tr J ± √(tr²J − 4 det J))` | ✅ `D3/D4` | as D3; the whole of B1/B2 is built on it | **CONFIRMED [CHLU]** |
| **D5** | overdamped `n_half = ln2/(−ln λ₊) ≈ 2γ ln2/[(2−γ)(εμ)²]` | ✅ `D5` (exact vs asymptotic agree to **0.96 %** at `h ≤ h*/5`) | ⭐ **B1 0.0433/0.0435/0.0431 on 49 held-out corners ×3 seeds** (the CRR uses the **exact root**). ⛔ **The ASYMPTOTIC form is 2.4 % off at `h = 0.30 h*`** (w6-retention) — quote the exact root | **CONFIRMED [CHLU]** (exact form) · ⚠ the asymptotic form is a **scope-limited** approximation, declared |
| **D6** | `n_half ∝ μ⁻²`, slope −1 in `log n_half` vs `log μ²` | ✅ `D6` (**−1.000014**) | ⭐ w6 `hub/retention` measures the overdamped slope at **−1.00096** against a simulation of the paper's own map; V2 E2.3 fitted **−0.985** (n = 35 overdamped rows), V5 fitted **−1.006**, and V2 E2.12 holds it at **−0.956 / −0.961** after 3000 anchored epochs. ⚠ **B3 is NOT this bar** — B3 tests the *underdamped*, curvature-**in**dependent leg (\|slope\| median 0.0357/0.0282/0.0343, i.e. ≈ 0, which is D7); B1 tests the whole curve including the `μ⁻²` branch | **CONFIRMED [CHLU]** |
| **D7** | curvature-independent floor `2 ln2/(−ln(1−γ))` | ✅ `D7a/D7b/D7c` (27.02681 at γ=0.05; identical for h = 0.2/0.5/1.0/1.5) | ⭐ **B3 floor agreement 0.0378 / 0.0378 / 0.0348**; w6-retention curvature-independence **1.6e-13**, simulated floor **27.027 ± 0.03 over a 50× span in h**; V2 E2.4 saturation at 27.03; V5 E5.5 both seeds return identical `λ_ret` | **CONFIRMED [CHLU]** |
| **D8** | crossover `h* = √(2/(2−γ))(1−√(1−γ)) = γ/2 + O(γ²)` | ✅ `D8/D9` (max \|tr²−4det\| at the band edges **3.109e-15**) | w6-retention `h*` vs γ/2 rel. err **5.0e-4** at γ=1e-3; V2 E2.5 metric bifurcation at the crossover | **CONFIRMED [CHLU]** |
| **D9** | upper band edge `h_f`; "the map loses stability at `h = 2` exactly, for every γ" | ✅ `D8/D9`, `D9b` (max\|λ\| = 1.000000 at h=2 for γ = 0.002/0.05/0.2/0.5) | w6-retention `1+tr+det = 0` to **3.3e-16** at h=2, four γ | **CONFIRMED [CHLU]** |
| **D10** | the exceptional point is **defective** — one Jordan block, secular transients `∝ n λⁿ` | ✅ `D10` (\|λ₊−λ₋\| = 2.107e-08 = √(machine ε); λ = 0.974679445 vs √(1−γ) = 0.974679434) | ⭐ **the `0.5–2` bucket carries ALL of B1's error (0.1482 vs 0.0008 deep)**; B4's miss ×3; the instrument floor's own worst cell at `h/h* = 1.42`; V2 E2.5 first-crossing/envelope diverge up to 3.2× | **CONFIRMED [CHLU]** |
| **D11** | square-root phase onset, exponent 1/2 | ✅ `D11` (**0.500209** over `h−h* ∈ [1e-6, 1e-4]`) | w6-retention **0.50006**; V2 E2.6 fitted **0.5165**, onset holds to `h−h* = 2.6e-6`; V2 E2.12 bit-identical at 0.5165 after 3000 anchored epochs | **CONFIRMED [CHLU]** |
| **D12** | the latch `q_∞ = q₀ + ε p₀/(Mγ)` at μ = 0 | ✅ `D12` (4e5-step rollout → **1.000000000000** vs derived 1.000000000000) | w6-retention control: `ρ = 1.0` **exact**, frozen displacement **7.8e-16**; V2 E2.7 obeyed to ≤ 1 %, written angle frozen to ≤ 1.2e-15 rad; V5 E5.8 \|\|λ_flat\|−1\| ≤ **1.7e-14**, drift ≤ **4.9e-12 rad / 200k steps**, 30/30 cells | **CONFIRMED [CHLU]** |
| **D13** | γ = 0 is a Jordan block: ballistic drift `θ_n = θ₀ + n ε p_θ/F²` | ✅ `D13` | V5 E5.9: **142.7 rad drift over 20k steps** — over a 10× *shorter* horizon than E5.8's 200k | **CONFIRMED [shorts]** |
| **D14** | confinement lifts every Hessian eigenvalue by `2α`; `τ_max = Γ/(2α)`, `Γ = γ/ε` | ✅ `D14a` (max \|lift − 2α\| = **1.976e-14**) · `D14b` (arithmetic printed) | ⭐ **B6, by FINITE DIFFERENCE of the exact gradient of the confined energy: 3.519e-11 / 1.202e-10 / 2.699e-11 against a 1e-9 bar** — 10 cells per seed, 5 wells × α ∈ {0, 1}; V5 E5.13 state fraction outside the boundary drops to **0** | **CONFIRMED [CHLU]** for the `2α` lift · ⚠ **the `τ_max = Γ/(2α)` CAP itself is UNTESTED** — no measurement anywhere bounds a lifetime by it |
| **D15** | `μ_k² = λ_k(M^{-1/2} ∇²V M^{-1/2})` | — | V2 E2.1 alignment to **3.2e-10** over 70 configs; E2.2 `μ²_meas/μ²_pred = 1.000000 ± 5e-12` over **4.5 orders of magnitude** | **DEFINITION** (its operational identity is CONFIRMED [shorts]) |

### 3.2 The friction dial and its critical point (V5 §2.1 / App. G.3)

| # | derivation | reproduced here | measurement | status |
|---|---|---|---|---|
| **D16** | `γ_crit = 2εμ` — the V-curve's computable minimum | ✅ `D16` (exact inversion of `h*(γ)=h`: 0.01 % error at h=1e-4, 0.99 % at h=1e-2, **5.97 % at h=0.063** where the w6 log quotes `γ*_exact = 0.1185`) | ⛔ **B4 MISS ×3 on the field store (0.3362/0.3301/0.3520 vs a ×2 bar)** — but **the predicted curve's own argmin over `γ*_exact` is 1.0000 on every seed**, so the grid resolves it and the gap is the rollout at the defective point. On the Jacobian instrument the shorts confirm it: V5 E5.1 argmin **0.076–0.096** vs predicted bounds **0.082–0.116**; E5.2 argmin/γ_crit **0.9032 ± 0.0027** (I-J) and **0.9001 ± 0.0052** (I-R3); E5.4 stable to **+1.4 %** across a 10× write-amplitude range; V2 E2.19 the curve is V-shaped with its minimum beneath `γ* ≈ 2εμ` | ⛔ **REFUTED as a ROLLOUT prediction on the field store** (the failing step is named: the pure-exponential envelope is invalid at the exceptional point, D10) · **CONFIRMED [shorts]** on the exact one-step Jacobian |
| **D17** | overdamped `d ln n_half/d ln γ = 1 + γ/(2−γ)`, **> +1 at every finite γ** | ✅ `D17a` (IS the exact log-derivative, agreement ≤ 1e-3 over γ ∈ [0.05, 0.5]) · ✅ `D17b` (> 1 everywhere in (0,2)) | ⭐⭐ **+1.1368 / +1.1367 / +1.1368 vs derived +1.1385; \|err\| 0.0038/0.0042/0.0040 vs the textbook +1's 0.1436/0.1430/0.1436 — 34–38× closer, all three seeds.** V5 E5.2 "slope above": I-J **+1.1182 ± 0.0107**, I-R3 **+1.1172 ± 0.0101**; V5 Table 1 headline **+1.116 ± 0.011**; V5 E5.1 designed arm **+1.23 to +1.27** | ⭐ **CONFIRMED [CHLU]** — with the reproducibility caveat in §4.2 |
| **D18** | underdamped `−γ/[(1−γ)(−ln(1−γ))] = −1 − γ/2 + O(γ²)` | ✅ `D18` (IS the exact log-derivative of the floor; matches the numeric to < 1e-6 at γ = 0.002/0.05/0.2/0.5) | ⛔ **NOT SUPPORTED on the field store**: measured **−1.1090/−1.1090/−1.1086** against derived **−1.0050**; \|err\| **0.1368/0.1277/0.1290** vs the naive −1's **0.1417/0.1324/0.1333** — the derived form "wins" by **0.005, 0.005 and 0.004**, i.e. a **3.5 % margin**, and the hub reports it as thin, not as a win. On the shorts' own instruments it is clean: V5 E5.1 **−1.006**, E5.2 I-J **−1.0020 ± 0.0003**, I-R3 **−0.9827 ± 0.0134** | ⚠ **INCONCLUSIVE on the field store** (declared thin by the hub; see §4.3 for a computed explanation I could not adjudicate) · **CONFIRMED [shorts]** |
| **D19** | a flat coset is the `μ → 0` limit of the same curve, `γ_crit → 0` | ✅ (algebraically immediate from `D16`) | no measurement isolates the unification claim | **UNTESTED** |

### 3.3 Finite temperature: FDT noise, diffusion, the sign flip (V5 App. G.5–G.6)

⛔ **The whole of §3.3 is UNTESTED ON THE CHLU FIELD STORE** — every w6 leg ran at `T = 0`. The
measurements are the shorts' own. ⚠ **Scope rider the papers attach to every `T > 0` statement:** it
requires **both** `langevin_noise="fdt"` **and** a Newtonian kinetic mode (V5 App. G.5 derives why —
a relativistic kinetic term leaves the stationary momentum law Gaussian while Gibbs demands a
non-Gaussian marginal).

| # | derivation | reproduced here | measurement | status |
|---|---|---|---|---|
| **D20** | `σ*_i = √(M_i T γ(2−γ))` forces `Var(p) = M T` | ✅ `D20` (residual **0.000e+00** at γ = 0.05/0.2/0.5) | V1 E1.20 shadow-bias finite-budget error **0.0995 → 0.0065**; V5 E5.11 `Var(p_i)/(M_i T) = 0.9986` at γ_φ = 0 | **CONFIRMED [shorts]** |
| **D21** | `D_θ = εT(2−γ)/(2F²γ)` | ✅ `D21` (`d log D/d log T = +1` exactly; the raw γ-slope is −1.037475 and `ln(1.9/1.95)/ln2 = −0.037475` is **exactly** the `(2−γ)` factor V5 isolates) | V5 E5.6, 25-cell grid: **`D̂/D_pred = 1.0068 ± 0.0219`**, bounded [0.9644, 1.0484]; `d log D/d log T` = +0.998/+0.997/+0.992/+1.010/+1.006; `d log D/d log γ` = −1.014/−0.980/−0.999/−1.001/−0.982 | **CONFIRMED [shorts]** |
| **D22** | `F² = M r*²` | — (a definition of the coset scale) | folded into D21's agreement | **CONFIRMED [shorts]** (indirectly, via D21) |
| **D23** | diffusive `n_half ∝ γ/[(2−γ)T]` — **friction preserves, temperature erases** | ✅ `D23` (γ-slope positive; predicted effect size for γ 0.05→0.2 is **4.333×**) | V5 E5.7: `d log n_half/d log γ` = **+0.9552 ± 0.0422**, positive in **10/10** seed×T conditions; effect size **3.77 ± 0.23×** (per-seed 3.55/3.64/3.61/3.89/4.15); `d log n_half/d log T` = −0.956 to −0.979; the massive underdamped mode responds with **−1.006** | **CONFIRMED [shorts]** ⚠ the derived effect size (4.333×) sits **15 % above** the measured 3.77 ± 0.23×; the **sign flip** is the claim and it is clean, the **magnitude** is approximate |
| **D24** | `l_θ = ε√(T/M)/(γ r*)` | — | used as a reporting scale (`Δ/l_θ` on every half-life); never isolated | **UNTESTED** as a law |
| **D25** | per-step coset increment; `p_θ` an AR(1), autocorrelation `1−γ`, variance `MT` | ✅ implied by `D20` | folded into D21 | **CONFIRMED [shorts]** (indirectly) |

### 3.4 The friction hole / memory vault (V5 §2.2 / App. G.7)

⛔ **UNTESTED ON THE CHLU FIELD STORE.** ⭐ **But this is the cleanest closed-form block in the whole
corpus and it reproduces to the digit** — see `checks/derivation_checks.out` `D26/D27`, `D29`, `D30`,
`D31`.

⚠ **One inference of mine, declared:** the inventory does not print V5's baseline `γ` for the vault
cells, so I **inferred `γ = 0.05`** — the shorts' standard value. The inference is not free-floating:
at `γ = 0.05` the formulas return V5's **own printed predictions exactly** (vault **110.25×**,
coupled-bath **13.88×**, refrigerator **7.942×**) and land within **0.6 %** of all four measured
`T_local/T` cells. ⛔ If the true `γ` were anything else, none of those six numbers would land.

| γ_φ | `γ_eff = 1−(1−γ)(1−γ_φ)` | vault `(γ_eff/γ)²` **predicted** | V5 E5.10 **measured** | `T_local/T` **predicted** | V5 E5.11 **measured** |
|---|---|---|---|---|---|
| 0.1 | 0.1450 | **8.41×** | **8.42×** | **0.36249** | **0.36171 ± 0.00129** |
| 0.2 | 0.2400 | **23.04×** | 22.39× | **0.23082** | **0.23044** |
| 0.3 | 0.3350 | **44.89×** | 44.31× | **0.17480** | **0.17513** |
| 0.5 | 0.5250 | **110.25×** | 107.77 ± 4.78× | **0.12591** | **0.12600 ± 0.00031** |

| # | derivation | status |
|---|---|---|
| **D26** | `γ_eff = 1 − (1−γ)(1−γ_φ)` | **CONFIRMED [shorts]** — it is the input that makes the table above close |
| **D27** | `T_local = T γ(2−γ)/[γ_eff(2−γ_eff)]` | **CONFIRMED [shorts]** — ⭐ **four cells, all within 0.6 % of measurement**; the refrigerator factor `1/T_local` = **7.9423×** vs V5's measured **7.94×** (designed) and **7.955×** (emergent, E5.14, over 24 cells at `observed/absorb = 0.9998 ± 0.0019`) |
| **D28** | `D_θ(hole) = εTγ(2−γ)/(2F²γ_eff²)` | **CONFIRMED [shorts]** (it is D29's numerator) |
| **D29** | **vault `= (γ_eff/γ)²`** | **CONFIRMED [shorts]** — ⭐ predicted **110.25×** is V5's own headline and my recomputation returns it exactly; emergent arm **106.1 ± 5.0×** (E5.15) |
| **D30** | the **rejected** coupled-bath alternative, vault `(2−γ)γ_eff/[γ(2−γ_eff)]` | ⭐ **REFUTED, and that is the point** — my recomputation gives **13.8814×** at γ_φ = 0.5, exactly the **13.88×** V5 refutes; the matched-`γ_eff` uniform-scalar control measures **13.28 ± 0.12×**, and the emergent arm measures the coupled bath's prediction of 1.0 at **0.2235** |
| **D31** | separation `= γ_eff(2−γ_eff)/[γ(2−γ)] ≡ T/T_local` **identically** | ✅ reproduced to **1e-12**: separation **7.9423** = `T/T_local` **7.9423**. **CONFIRMED [shorts]** |

⚠ **Carry V5's own disqualifications on this block** (`shorts-experiments-inventory.md` §5.3):
**E5.15** — the emergent **contrast** metric **TRIGGERED V5's OWN DECLARED FALSIFIER** (field-to-scalar
control variance **23.39 ± 10.06** against a pre-registered [6.5, 9.5]); the **absolute** vault
transfers, the contrast statement is native to the designed architecture only. **E5.16** — V5
declares the laundering/confinement comparison **VOID** (the control that must return exactly 1.000
returns **0.4586 ± 0.1181**); ⛔ no in/out spread ratio may be quoted from it. **E5.17** — an `n < 3`
cell, flagged by V5 itself.

### 3.5 Deletion (V5 §2.3 / App. G.8) — ⭐ the block the programme DID test, and it splits

| # | derivation | measurement | status |
|---|---|---|---|
| **D32a** | **decay commutes with deletion**: `(K, {a_i})` with canonical placement `L(K)` a function of the live key set and fixed geometry alone; decay acts item-wise on `{a_i}`; deletion touches `K` and `L(K)` only ⇒ **disjoint components ⇒ they commute** | ⭐⭐ **BAR 3 PASSES, ZERO TOLERANCE, ALL THREE SEEDS: `evict ∘ decay^n == decay^n ∘ evict`, 0 bytes differing in a 2,054,866 B image, in all 12 cells per seed** (`n ∈ {1,4,16,64,539,555}` × {disjoint item set, same item set}) | ⭐ **CONFIRMED [CHLU]** — and on a **learned** store, which V5 does not claim |
| **D32b** | …**and the post-delete store is byte-identical to the store built from the never-inserted history** | ⛔ **BAR 4 FAILS, as pre-registered, on all three seeds: 1,339,083 / 1,338,974 / 1,343,641 B differ = 65.17 % / 65.16 % / 65.39 % of the image.** ⭐ **The per-segment break-out is worth more than the bar:** of seed 0's 1,339,083 bytes — `coupling_v` **798,926**, `term_mu` **267,684**, `atoms` **221,590**, `coupling_u` 33,595, fitted widths/gaps/counts 16,769, live-set book-keeping 519. ⛔ **The placement is only 16.5 % of the obstruction; 83.5 % is that the address book and the coupling table are FITTED ON THE POOL** | ⛔ **REFUTED on this object, and the failing step is named:** V5 conditions byte-equality on canonical Blelloch–Golovin placement. ⚠ **This falsifies the PAPER'S PRECONDITION on this store, not the paper's law** — doing the Blelloch–Golovin thing would close **at most one sixth** of the gap here |
| **D33** | recency eviction is excluded because it makes `L` a function of insertion history | `gen_field`'s eviction is attribute-based (provenance ∪ geometry), never recency — which is exactly why D32a passes | **CONFIRMED [CHLU]** (as a precondition check) |

⭐ **Two consequences the hub DERIVED before measuring, both then exact:**
1. With `LIFETIME_EPS = 1e-6`, **`n_refuse = 19.9316 × n_half` for every curvature** — measured
   refusal step **539 = 19.9432 half-lives on all three seeds**, matching to the digit.
2. **`μ²` cancels**: in half-life units every lifetime is the one curve `2^(−t)`. Verified to
   **4.2e-14** in the law and **byte-exactly on the store** — two clocks a decade apart
   (`n_half = 27.027` vs `270.268`) give **16/16 horizons byte-identical, worst lifetime gap
   `0.000e+00`**. ⛔ **No curvature can trade refusal against composition.**

*Source:* `evidence/outputs/_thinkpad/w6/w6-retention/LOG.md` §3 §5 §6.

### 3.6 Reach, escape and access certificates (V1 App. H)

⛔ **UNTESTED ON THE CHLU FIELD STORE** — no wave 3–9 hub measured the V1 certificate family. The
derivations reproduce; the measurements are V1's own analytic testbed.

| # | derivation | reproduced here | measurement | status |
|---|---|---|---|---|
| **D34** | relativistic kinetic term `T(p) = √(c²pᵀM⁻¹p + m₀²c⁴)` | — (definition) | — | **DEFINITION** |
| **D35** | the causal box `L_i = T ε c/√M_i` | — | V1 E1.15 causal cap to relative error **2.0e-12**; E1.6 Newtonian control lands at every `d` (no box exists) | **CONFIRMED [shorts]** |
| **D36** | energy-blindness of the box: only `(ε, c, M_i)` enter | — | V1 E1.6 is the control that makes the point | **CONFIRMED [shorts]** |
| **D37** | `det S_z(M) = 1` exactly | ✅ `D37/D38` (\|cosh²z − sinh²z − 1\| ≤ **1.776e-15**) | V1 E1.2/E1.3 `det S = 1.000 ± 4e-6` | **CONFIRMED [shorts]** |
| **D38** | `H(S_z z) ≤ e^{2\|z\|} H(z)`, equality on `u = ±v` | ✅ `D37/D38` (**bound never violated** over 20,000 random `(u,v,z)`; max margin −1.292e-08) | V1 E1.3 `(z, ratio, bound)` = (0.25, 1.13, 1.65), (0.5, 1.55, 2.72), (1.0, 3.79, 7.39), (2.0, 27.5, 54.6) — **the ratio is below the bound at every rapidity** ⚠ V1 notes it is a *matched-quadratic* certificate; on a quartic well the raw ratio can legitimately exceed it | **CONFIRMED [shorts]** |
| **D39** | reach bracket, `z ≥ arcsinh((d − 0.4 − L) M₀/p₀)` | — | V1 E1.4: `d = 4.0` needs **z ≥ 2.0105**, `d = 5.0` needs **z ≥ 2.64**, both above the swept budget `z ≤ 2.0` — and E1.2 lands **1/0/0** at `d = 3.2/4.0/5.0`. ⭐ **The bracket predicts exactly where the squeeze stops** | **CONFIRMED [shorts]** |
| **D40** | `cosh 2z = 1 + 2(M₀δ/p₀)²` (quadratic in distance) while `e^{2\|z\|}` is exponential in **rapidity** | ✅ `D40` (exact to 1e-12 at δ = 0.1/0.5/1.0) | V1 E1.3's table is the same statement measured | **CONFIRMED [shorts]** |
| **D41** | gated channel: `det J = 1` exactly, independent of Δ and of the gate | — | V1 E1.5 `det J = 1.0` **exact**, energy change **0.0 exact**, landing 1 at every `d`; E1.15 Jacobian preservation to **2.2e-16**; contrast E1.7 state-replacing map `det J = 0.0` exact | **CONFIRMED [shorts]** |
| **D42** | channel energy `ΔH_wh = V(q+Δ) − V(q)` | — | V1 E1.5 | **CONFIRMED [shorts]** |
| **D43** | hard-gate defect: `det J = 1 + ∇g(q)ᵀΔ`; first order in the gate's variation, so no small Δ removes it | ✅ `D43` (matrix-determinant-lemma residual **5.773e-15** over 2,000 random cases in d=3) | V1 E1.14 `det J` measured at **2.05** vs 1 for a frozen gate | **CONFIRMED [shorts]** |
| **D44** | latch transport `Q' − Q = pᵀXΔ` | — | V1 E1.9 coset-tangent: predicted 0.0000, measured ΔQ mean **−0.0000**, max error **1.2e-7**, `std(Q_out) = std(Q_in) = 0.0803`; E1.10 across-coset: predicted **0.2500**, measured **0.2500**, max error **0.0**; ⭐ E1.11 negative control — a random shift with the same `det J = 1` scrambles the spread (**0.0972 ± 0.2465**, `std(Q_out) = 0.2793`) | **CONFIRMED [shorts]**, with a working negative control |
| **D45** | governor `γ_n = s·tanh(max(0, H−E*))`; re-absorption in `≈ 2z/γ_c` and `≈ (1/γ_c)ln(1 + ΔV/E*)` steps | — | no measurement isolates the re-absorption times | **UNTESTED** |

⚠ **The limit V1 measures on its own certificate (E1.13, the BIBO battery):** at exit `b = 5.0`,
**`ΔH = 0.0` exactly, the energy-only sub-level test admits, and the trajectory escapes** (max
excursion 0.09 certified vs **104.83** blind; escape rate 0 vs 1.0). ⛔ **`det J = 1` plus zero
energy change is NOT sufficient for BIBO stability.** The operative clause is coercive-component
membership — which is **A7**, asserted, not derived.

### 3.7 GMOR / spectral mass (V2 App. M — V2 itself flags this as *supporting*, not a claim)

| # | derivation | measurement | status |
|---|---|---|---|
| **D46** | `μ²F² = δΣ` | V2 E2.17, linear ambient spurion, 8 trained checkpoints, δ ∈ [1e-8, 0.3], f64, zero retraining: **max \|μ²F² − δΣ\| = 1.33e-15**; relative exactness ≤ 2.7e-14 for δ ≥ 1e-2 | **CONFIRMED [shorts]** ⚠ V2: *"a supporting result, not one of this paper's claims"* |
| **D47** | `K_θθ = δ/r*` with relative error `~ ε/δ` (a round-off floor) | V2 E2.18 on the **analytic Mexican hat**: cancellation-free Hessian **1.1–2.2e-16** at every δ; autodiff Hessian **2.28e-8 → 4.22e-15**, exactly `∝ 1/δ` | **CONFIRMED [shorts]** ⛔ **this is the corpus's ONLY occurrence of "Mexican hat" — see §3.9** |
| **D48** | expansion variable `x ≡ δ/(M_ch μ²_rad f)` | V2 E2.17's running condensate `Σ` scales by **+16.1 % to +210.1 %** across the δ grid | **CONFIRMED [shorts]** (weakly — it is a bookkeeping variable) |
| **D49** | tree-level Goldstone theorem `∇²V·Jq = 0` | V2 E2.1 flat mode measures `μ² ≈ 2.4e-15` (machine precision) on trained designed checkpoints | **CONFIRMED [shorts]** |
| **D50** | SO(2) invariance ⇔ purely radial `V` | structural | **DEFINITION** |

### 3.8 ⛔ ASSERTED WITHOUT DERIVATION — A1–A8, all **UNTESTED as laws**

These are **stated in the shorts and derived nowhere in them.** ⛔ **A section must not present any of
them as a result of the theory.**

| # | assertion | where | why it is not a derivation | what the record adds |
|---|---|---|---|---|
| **A1** | **the Lifetime Trilemma** (value fidelity · amplitude-independent hold · amplitude-independent latency — pick two) | V5 §2.3; V1 §5 | stated as fact in both; **no proof, no impossibility argument in either** | V5 E5.22 measures the *consequence* on the shipped store (lifetime correlation `r = −0.85` scaling with `a_i²`; `R50` contracts **1.146 → 0.752** as `A: 1 → 0.06`, a **1.52×** contraction, matching a pre-registered saddle prediction at all five points ± 0.20; a TTL vector store's radius is a constant hard step at 0.75–0.77, age-independent). ⛔ **The trilemma itself stays UNTESTED** |
| **A2** | throat latch impact `μ² ∝ A/l²` | V1 App. C Table 1 | a bare proportionality in a table cell; **`A` and `l` are never defined anywhere in the paper** | ⛔ **w6 `hub/potentials` REFUSED it as a substitute for the absent law** and said so first |
| **A3** | emergent register capacity "≈ 1–1.6 bits" | V2 §G.1/App. L; V5 Fig. 8 caption | a figure-level characterisation of washboard minima; no capacity derivation | V5 E5.18 is the evidence behind it and is a **declared negative**: on every emergent seed a written coset value relaxes to the washboard minimum and nothing is retained; the designed checkpoint returns it exactly |
| **A4** | "kinetic isotropy is the price for an equivariant write current" | V2 §5 Discussion | discussion-level; no derivation | — |
| **A5** | minimum-separation certificate from lattice packing | V5 App. F | the separation **1.540000** is a **MEASURED constant**; no closed form relates lattice geometry to it | ⛔ **also refused by w6 `hub/potentials`** as a substitute |
| **A6** | `G → H`: the `dim(G/H)` orbit tangents are "exactly flat" | V2 §5 | ⭐ **V2 itself says: *"This is a direction on which this paper offers no evidence, but is part of our future work."*** | — |
| **A7** | BIBO stability requires coercive-component membership | V1 §3, App. C.2 | **necessity is shown empirically** (the `b = 5.0` zero-energy escape), not derived | V1 E1.13 measures it: certified excursion **0.09** vs blind **104.83** at `b = 5.0` |
| **A8** | a flat direction keeps `μ² = 0` under inertial anisotropy by Sylvester's law | V2 App. F | cites Sylvester but gives **no derivation** | plausible and probably right; still a one-line assertion |

### 3.9 ⛔⛔ THE DECLARED ABSENCE — a law the programme planned against is not in its own papers

> **"Mexican-hat width / separation sets the interpolation floor and the refusal gap" DOES NOT EXIST
> in the corpus.**

- **"interpolation"** occurs **0 times** in the three papers.
- **"Mexican hat"** occurs **once**, in V2 App. M.4, as a **float64 round-off testbed** with **no
  width and no separation parameter** (that is D47/E2.18).
- **"centre-surround", "DoG", "RBF", "lateral inhibition"** occur **0 times.**
- ⭐ **Established independently twice** — by `spoke/shorts-inventory`'s exhaustive PDF search and by
  `hub/potentials`' own, run separately.
- ⭐⭐ **And the nearest genuinely derived object INVERTS it:** the floor of the V2/V5 retention curve
  is **curvature-INdependent** (D7). Substituting one for the other would have reversed the papers'
  finding.
- ⭐ Both near-substitutes (**A2**, an undefined table cell; **A5**, a measured constant) were
  **refused** as forbidden, and the absence is reported as `hub/potentials`' **first** result.

⚠ **"The exact-decay law" is likewise AMBIGUOUS** — there is **no single law by that name**; there
are **three non-interchangeable exact statements**, and the overdamped one is written `≈`/`≃`, not
`=`. A hub must pick one and name it. (w6 `hub/retention` did: it used the band structure as the
generator, D32 as the property under test, and **declined** the asymptotic rate as the operator, with
reasons logged.)

*Source:* `shorts-experiments-inventory.md` §1.1 §1.2 · `w6-potentials/LOG.md` §1 ·
`evidence/outputs/w6-advisor-report.md` §2.

### 3.10 The mechanism finding the retry produced — a resolution law (⛔ not scored, not a row)

`w6.1/compose-physics` was scoped on the Hub's own hypothesis that settling *leaves* the principal
subspace. ⛔⛔ **False, and false by construction.**

- Off-span fraction of the settled state: **≤ 2e−7** (the same float32 rounding the incumbent carries
  **standing still**); ≤ 6.8e−5 of the displacement; the channel-space kernel descent is confined to
  **9.9e-8**. The orthogonal fraction predicts the loss **at chance, AUC 0.457–0.559**.
  ⭐ **`span(G_k)` is an invariant subspace of the settle** because the settle is parameterised in
  span coordinates, emits through the same renderer, and its force is built from the atoms — and the
  atoms *are* the span.
- ⭐⭐ **What does break the read is motion INSIDE the subspace, and it is a resolution law.** The
  read's answers form a **dense lattice**: distance to the nearest answer differing in exactly one
  factor is **ρ₁ = 1.71 / 1.89 / 2.00** against a state norm of **31.3 / 32.3 / 32.8** — one factor
  step is **5.5 % of the state**. Spacing is very uneven: three hues **5.1–9.6** apart, scale ~2.0,
  orientation ~1.9–2.3.
- **The judge's decision cells ARE that lattice's cells:** pushing the answer a fraction `s` toward
  its nearest one-factor neighbour flips the read-back with probability crossing ½ at **s ≈ 0.5 in
  every one of six factors** (0.403–0.733 at s = 0.5; ≥ 0.97 at s = 1), over 15,552 judged states.
- **The dynamics step across those cells, and preferentially so:** the legs move **0.16 / 0.78 /
  5.86 lattice steps**; `onestep` — a *single* gradient step at the certified-safe strength — already
  moves **99.5 %** of requests more than half way to a *different valid* answer. The 51 lattice
  directions span a rank-51 tangent in a 308-dimensional state, so an isotropic step would place
  `√(51/308) = 0.407` of its length there; the settle places **0.78–0.86**, i.e. **3.7–4.4× the
  isotropic share** in exactly the directions that change the answer. Displacement in lattice units
  predicts the loss at **AUC 0.904**.
- ⭐ **0 scored runs; composition stayed 2/3 with 3/3 banked.**

⚠ **This is the mechanism that connects §1 and §2**: the CRR is right about *how fast* the state
moves, and the resolution law is why *any* such motion is a cost on this read. **They are consistent,
and neither licenses the other's claim.**

*Source:* `evidence/outputs/_thinkpad/w6/w61-compose-physics/LOG.md` §FINAL REPORT §1–§3.

---

## 4. MY OWN NUMERICAL REPRODUCTION — what I ran, and the two things it found

`checks/derivation_checks.py` → `checks/derivation_checks.out`. Pure stdlib; **no `chlu` import, no
production code touched, no scored run.** Reruns in ~4 s.

### 4.1 Result: **34 / 34 checks reproduce, 0 FAIL**

Covered: D3/D4, D5, D6, D7a/b/c, D8/D9, D9b, D10, D11, D12, D13, D14a/b, D16, D17a/b/c/d, D18,
D18-DIAG, D20, D21, D23, D26/D27, D29, D30, D31, D37/D38, D40, D43, plus `B1/B2`, `AUDIT` and
`D14-TAUTO` against the recorded artefacts.

⛔ **Three defects found — all three were in MY first pass, none in the evidence base**, and each
would have produced a false REFUTED:
1. **D12** — my rollout summed `(1−γ)ⁿ` from `n = 1`, not `n = 0`, so the latch landed at 0.965 vs
   the derived 1.000. The law was right; my loop order was wrong.
2. **D17** — I first took the γ-derivative at **fixed `h/h*`** instead of **fixed `h`**. Because
   `h* ∝ γ`, that returns **−1.026** instead of **+1.026**, i.e. it inverts the sign of the branch.
   ⚠ **This is a live trap for anyone re-deriving D17: the derivative is at fixed MASS, not at fixed
   position on the band.**
3. **D10** — I set the eigenvalue-collision tolerance below `√(machine ε)`. A discriminant that
   cancels to zero cannot be resolved below ~1e-8 in double precision; the collision is exact
   analytically.

### 4.2 ⚠ FINDING 1 — the headline constant **+1.1385** is not reproducible from the artefacts

`1 + γ/(2−γ) = 1.1385` at exactly **γ = 0.243303**.

The recorded scored γ grid is **[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]**
(`test/s0/results.json → pins.gammas`), plus a per-mode 5-point geometric bracket
`[γ*/2, 2γ*]` (`pins.gamma_refine = 5`, `gamma_refine_span = 2.0`). The derived slope at each grid
point is:

| γ | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.35 | 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| `1+γ/(2−γ)` | 1.0010 | 1.0025 | 1.0050 | 1.0101 | 1.0256 | 1.0526 | **1.1111** | **1.2121** | 1.3333 |

**γ = 0.2433 lies inside the overdamped span but is not a grid point**, so +1.1385 is an aggregate
over per-cell fit windows. ⛔ **Those per-cell slope rows are in no machine-readable artefact:**
`grep -rIl "1.1368\|1.1367"` under `w6-potentials/` hits **only** `LOG.md`, `REPORT.txt` and
`RESULTS-LEDGER.md` — three prose/ledger files. `results.json` carries `n_v_curves` as a count and
the 49 corner summaries, not the slopes.

**Consequence for the section, stated plainly:**
- ✅ the **law** D17 reproduces exactly, and its qualitative content (**> +1 at every finite γ**) is
  verified over the whole band;
- ✅ **B1, B2 and the whole audit trail reproduce from the artefacts** (§2.3);
- ⛔ **the specific pair "+1.1385 derived vs +1.1368 measured" is reproducible only from the hub's own
  report, not from a recorded number.** ⚠ It is the paper's sharpest claim and it currently has the
  **weakest artefact trail** of anything in §2.3. **Recommended, and cheap:** ask whoever owns the
  w6 branch to emit the per-cell `(γ_window, slope_measured, slope_derived)` rows into a JSON. This
  is a *reporting* gap, not evidence that the number is wrong — nothing I ran contradicts it.

### 4.3 ⚠ FINDING 2 — D18's "thin margin" has a computed candidate explanation I could not adjudicate

The two quoted D18 numbers sit at **different points of the same derived curve**:

- derived **−1.0050** ⇒ `−γ/[(1−γ)(−ln(1−γ))] = −1.0050` at **γ = 0.009917**;
- measured **−1.1090** ⇒ the same expression equals −1.1090 at **γ = 0.184104**.

Derived D18 on the recorded grid: γ=0.002 → −1.0010, 0.01 → −1.0050, 0.05 → −1.0261, 0.1 → −1.0546,
**0.2 → −1.1204**, 0.35 → −1.2500, 0.5 → −1.4427.

⭐ **If the underdamped fit window in fact sat near γ ≈ 0.2, the derived value would be −1.1204
against the measured −1.1090 — an error of 0.011 instead of the reported 0.1368, i.e. ~12× closer,
and the "thin" verdict would become a clean one.**

⛔ **I am NOT claiming that.** Three things cut against it and I could not resolve them:
1. the underdamped branch is by definition γ **below** γ_crit, and for the modes in this grid
   `γ*_exact` runs around 0.1185 — γ ≈ 0.184 would be on the wrong side;
2. **no secant of `ln(floor)` vs `ln γ` over the recorded grid reaches −1.109** (over [0.002, 0.1] it
   is −1.01309; over [0.002, 0.2], −1.02356);
3. the shorts' own underdamped slopes are **−1.006** (V5 E5.1) and **−1.0020 ± 0.0003** (V5 E5.2 I-J)
   — i.e. **the shorts measure the small-γ value, and w6's −1.1090 is the outlier**, not the
   derivation.

**Status:** the w6 field-store D18 leg is **INCONCLUSIVE**, exactly as the hub reported it. The
per-cell fit windows would settle it in one grep and are not recorded (same gap as §4.2).
⛔ **Do not write D18 up as a confirmation.** Write it up as: *the underdamped slope is confirmed on
the shorts' own designed and emergent testbeds (−1.0020 ± 0.0003) and is inconclusive on this store,
by a margin the hub itself declared thin.*

### 4.4 ⚠ FINDING 3 — one tautological field survives in the artefacts

`test/s{0,1,2}/results.json → d14_confinement.max_abs_residual` is **exactly `0.0` on 60 cells, all
three seeds.** It compares `hess_eig(α)` to `hess_eig(0) + 2α` — **the same algebra on both sides**,
so it can only ever pass. ⭐ **The hub caught this itself, said so, and replaced it**: the SCORED B6
is `hessian_fd_check`, a central finite difference of the exact gradient of the confined energy
(**3.519e-11 / 1.202e-10 / 2.699e-11**), which *can* fail.
⛔ **A writer reading `results.json` cold would quote the wrong field.** Quote `hessian_fd_check`.

---

## 5. THE BENCHMARK STATEMENT — a requirement list

⭐ Drafted from the **w5/w6 negatives**. Written as a **requirement list**, ⛔ **not** as an argument
that this programme deserves a benchmark. The full standalone version, for the paper-writer to lift,
is `BENCHMARK-STATEMENT.md` in this directory. Summary:

### 5.1 What the programme measured

Composition as **distinct valid novel tuples per corner**, on disjoint held-out Montero n-way corners
(49 n-way + 60 pair-tiled sub-corners), paired by corner, against parameter-matched trained learners,
at an **asserted byte pin** (524,288 B, tolerance 0), with **refusal** scored as the pair
`(reject_invalid, accept_unseen_valid)` and **deletion** scored as bytes-changed plus surviving
composition. Result: `pca_tensor` beats `mlp_180k` by **+10.143 / +8.673 / +7.796** on row 4 and
**+4.033 / +2.483 / +2.417** on row 3, three seeds, CI-lower above +1.0 every seed.

### 5.2 What no existing benchmark scores — the measured reasons

| # | what is unscored | the measurement that says so |
|---|---|---|
| **R1** | **An "inductive-bias" row has no comparator.** Refusal of never-seen conjunctions and post-deletion behaviour are DISPLAY rows: the only honest comparator is a transformer with a refuse token trained on impossible examples, and our line is *"no impossible example was ever seen."* | w5 `hub/refuse`: rows 5/6 tie an **8-byte rule** at **(1.0000, 1.0000)**; `table_absence` **IS** `bitmap8` at a byte multiple of **59,392×** |
| **R2** | **Composition and exact forgetting are measured on different objects and no benchmark scores both.** | w5: the content that **wins** composition (principal subspaces) **forfeits §B.4 exact per-item deletion by construction** — priced at **≈ 18.6 tuples/corner** to keep items. The object that **forgets exactly and free** (the coupling hole: **1 byte**, **0 bytes outside**, refusal **1.0000**, composition **exactly 0.0000**) **loses composition when it evicts atoms** (**−2.09 / −0.91 / −6.74**) |
| **R3** | **A lifetime is not a deletion primitive and nothing scores the gap.** | w6 `hub/retention`: **96.30–99.83 %** of the composition cost is paid by a **continuous** weight while the **discrete** liveness bit has not yet flipped; `t_pay(97 %)` = **15.984 / 7.992 / 11.988** half-lives against `t_refuse` = **19.943** |
| **R4** | **A dynamics result cannot enter the scoreboard at all.** | w6 `hub/potentials` §C.1: ⛔ *"there is no lookup comparator for 'does a closed form predict a rollout', and this hub claims no row against any rival."* |
| **R5** | **Two purpose-built gyms failed their own adoption gates, on two different grounds.** | GYM-1: matched-bytes stitching **0.8520**; a lookup table at **0.9797** beats both trained rules-visible 250 k-param baselines (**0.3700**, **0.4173**) by **2.3×**. GYM-2: **TW-B 0.9825** against a pre-registered **0.60**, and the table beats the better baseline by **5.7×** |

### 5.3 ⭐ What a benchmark would have to contain — the requirement list

Each requirement is followed by **the measurement that forces it**.

1. **At least one rule must be able to render a candidate ILLEGAL, not merely more expensive.**
   *Forced by:* GYM-2's **O3 composition-blind oracle scoring 1.0000** (0 faults) while
   obstacle-blind scores 0.0387 and speed-blind 0.4213. A speed limit is an upper bound, so composing
   it with an obstacle can only change a route's **cost**, never its **feasibility** — ⛔ *"no metric
   and no number of dimensions makes it composable."* Under an optimality-gap metric O3 only falls to
   **0.9367**, still above the 0.90 pass line.
2. **Score with an optimality gap, not a feasibility-or-goal-reaching metric.**
   *Forced by:* the same addendum — the optimality gap drops TW-B from **0.9825 to 0.3900**, below
   the 0.60 line, because stitched plans are valid but **sloppy** (1.25 extra steps, only 39 % exactly
   optimal). ⚠ It is strictly better **and** it does not rescue requirement 1.
3. **The state space must be large compared with `4·n_params / frag_bytes`.**
   *Forced by:* GYM-2's arithmetic — budget **1,001,188 B** buys **200,238** five-byte macro-actions
   against **32,400** states = **5.078 per state**, and *"a table with a useful macro-action at every
   state IS a planner"*. The coarse band crosses 0.60 at ≈ **46,000 B ≈ 11,000 params ≈ 0.28
   macro-actions per state** — **18× less than the matched budget**. At 250 k parameters that means
   an environment of order **7 × 10⁵ states** (a 100×100 grid at these velocities, not a 20×20 one).
4. **Run a matched-bytes STITCHING tripwire with a `min_frag` sweep, before adoption.**
   *Forced by:* GYM-1 — ⛔ **T1 exact-key lookup alone scored 0.0000 and would have certified a dead
   gym as sound.** In GYM-2, whole-trajectory retrieval died (**T1 0.0000, T3 0.0003** vs GYM-1's
   0.7027) while coarse recombination still fired at 0.9825. **Exact-key lookup is not a tripwire.**
5. **Make a rule-blind oracle a mandatory gate.** *Forced by:* it is nearly free and it is decisive —
   if a policy blind to a rule scores 1.0, that rule is untested and no claim may rest on it (GYM-2
   O3 = 1.0000).
6. **Measure held-out path/target diversity BEFORE adopting.** *Forced by:* GYM-2's gate **D**
   (N3-rel median **2.0** against a required ≥ 4, hard-fail ≤ 2). If the median held-out item tiles
   from two stored fragments, no split saves it.
7. **The validity check must be exact, with zero tolerance.** *Forced by:* both gyms passed gate V
   with **no tolerance used anywhere**, which is what makes a refusal claim non-approximate.
8. **It must contain a deletion request, scored on the SAME object and the SAME metric as
   composition.** *Forced by:* R2 above. A benchmark that scores only composition cannot see the
   programme's central open problem, which is **measured, not argued**.
9. **It must separate a CONTINUOUS degradation channel from a DISCRETE liveness channel.**
   *Forced by:* R3 — 96–99.8 % of the price arrives through a channel the refusal metric cannot see.
10. **It must not be able to score a dynamics prediction as a taxonomy row.** *Forced by:* R4, and by
    §1 of this file. ⛔ **A benchmark that let the CRR imply a scoreboard number would be the failure
    this evidence base exists to prevent.**

⛔ **What this list is not.** It is not a claim that the field lacks such a benchmark (no scout in this
evidence base surveyed the field for one at the ICLR-long wave level — `c3-benchmark-scout.md`,
`scout-benchmark-targeting.md` and `hepa-benchmark-scout.md` are **Campaign-3** documents and are
**declared NOT USED** here). It is a specification: **these ten properties are what would make the
programme's claims falsifiable**, and each one is forced by a number this programme already paid for.
The Head's own ruling stands: *"benchmark for the physics CLU declared future work"*
(`evidence/outputs/advisor-wave-plan-w8-w10.md` §0).

---

## 6. ⛔ DECLARED NOT-RUN (never a zero, never a blank)

| item | status | reason |
|---|---|---|
| Scored runs | ⛔ **NOT RUN** | mandated by the task file: *"Scored runs: ⛔ none."* |
| Re-measurement of any V1/V2/V5 experiment | ⛔ **NOT RUN** | the shorts' numbers are read out of the PDFs by `spoke/shorts-inventory`; re-running them is a wave, not a spoke. Every such row is tagged **[shorts]** above |
| Re-running any w5/w6 leg | ⛔ **NOT RUN** | out of scope and would spend budget; I recomputed from recorded artefacts instead (§2.3, §4.1) |
| Per-cell D17/D18 slope reproduction | ⛔ **CANNOT BE RUN** from the artefacts | the rows are not recorded (§4.2, §4.3). Declared as a reporting gap, not as a null |
| Finite-`T` laws (D20–D25) on the CHLU field store | ⛔ **NOT RUN** | every w6 leg ran at `T = 0`; the field store has no FDT-noise leg. Status **UNTESTED [CHLU]** throughout §3.3 |
| Vault laws (D26–D31) on the CHLU field store | ⛔ **NOT RUN** | same; §3.4 is **[shorts]** throughout |
| V1 certificate family (D34–D45) on the CHLU field store | ⛔ **NOT RUN** | no wave measured it; §3.6 is **[shorts]** throughout |
| `$CHLU_CLAUDE_ROOT` | ⛔ **NOT READ** | standing rule: frozen, stale snapshot |
| `claims_matrix.md` / `negative_results.md` as w5/w6 sources | ⛔ **NOT USED** | they end at the C2W11 fold and contain no w5/w6 entry (§0) |
| Campaign-3 benchmark scouts | ⛔ **NOT USED** | different campaign, different object; named in §5.3 so the omission is visible |

---

## 7. SOURCE INDEX — every path this file quotes

```
evidence/outputs/w6-advisor-report.md                                   §1 §2 §3 §4 §5 §6 §7
evidence/outputs/w5-advisor-report.md                                   §1 §2 §3 §5
evidence/outputs/advisor-wave-plan-w8-w10.md                            §0 §3
evidence/outputs/_thinkpad/w6/SCORED-LEDGER.md                          potentials / retention / compose-physics outcome rows
evidence/outputs/_thinkpad/w6/w6-potentials/LOG.md                      §1 §2 §3 §4 §5 §6 §7 §8 §9
evidence/outputs/_thinkpad/w6/w6-potentials/RESULTS-LEDGER.md           all bars, all seeds
evidence/outputs/_thinkpad/w6/w6-potentials/PREREG-potentials-1.md      P6 (the sharp test)
evidence/outputs/_thinkpad/w6/w6-potentials/test/s{0,1,2}/results.json  pins · estimand.corners · byte_ledger ·
                                                                        integrator_check · hessian_fd_check ·
                                                                        instrument_floor · d14_confinement · fit1_C
evidence/outputs/_thinkpad/w6/w6-potentials/replication/s0/             the bit-exact replication
evidence/outputs/_thinkpad/w6/w6-potentials/report_summary.json         B1/B2 per leg
evidence/outputs/_thinkpad/w6/w6-shorts-inventory/
        shorts-experiments-inventory.md                                 §1 (absences) §2 (D1-D50) §3 (A1-A8)
                                                                        §5.1 (E1.*) §5.2 (E2.*) §5.3 (E5.*) §6
evidence/outputs/_thinkpad/w6/w6-retention/LOG.md                       §2 §3 §4 §5 §6 §7
evidence/outputs/_thinkpad/w6/w6-compose-physics/LOG.md                 §FINAL REPORT
evidence/outputs/_thinkpad/w6/w61-compose-physics/LOG.md                §FINAL REPORT §1 §2 §3
evidence/outputs/_thinkpad/w5/w5-gym1/LOG.md                            §0 §1 §4 §5 §6
evidence/outputs/_thinkpad/w5/w51-gym2/LOG.md                           §0 §1 §2 §3 §4 §11
```

Produced by this spoke (in `/c/chlu/outputs/w10-physics-section-evidence/`):

```
START_UTC.txt                 LOG.md                 EVIDENCE.md (this file)
BENCHMARK-STATEMENT.md        checks/derivation_checks.py     checks/derivation_checks.out
```
