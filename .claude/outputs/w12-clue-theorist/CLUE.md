# CLUE — is the subspace store the CLU with settling switched off?

**w12 `spoke/clue-theorist` (physics-theorist), 2026-09-21. ZERO scored runs, zero GPU, 2 threads.**
Every number below names the file it came from. Anything not run is a **DECLARED NOT-RUN** with a
reason. Nothing here is new evidence about the scoreboard; the only things I *ran* are the five
re-runnable numerical checks in `checks/`, and they touch no scored artefact.

> **Answer in one line.** ⭐ **Yes, and it is an identity, not an approximation — but only inside a
> well.** ⛔ **The half of the story that is a prediction rather than a measurement is the CRR, and it
> predicts a HALF-LIFE, not a score.** ⚠ **And the CRR's sharpest number does not survive a provenance
> check well enough to lead the main text; its *law* does.**

---

## 1. ⭐ THE PARAGRAPH THE PAPER CAN CARRY (verbatim)

> The store we evaluate is the **zero-dynamics limit** of the CLU: the same content, the same wells,
> the same read, with the settling loop's step count set to zero. On this content that limit is not
> an approximation of convenience — **inside a well it is an algebraic identity**. The content is an
> orthonormal basis `U_k` of the per-well atom span, obtained by SVD of the deployed atoms and
> carrying no bytes of its own, and the read renders the request's address through *that same basis*,
> `x_k = U_k c_k`. The store's own energy for a basis is the distance to its span,
> `E_store,k = ½ dist²(x_k, span G_k)`, which is zero everywhere on the span; the request enters as a
> basin at the address book's own point, `E_req,k = ½β‖x_k − a_k‖²`, which is zero at the address.
> Because `a_k ∈ span(G_k)` **by construction of the renderer**, both terms attain their minima at the
> same point, so the addressed point *is* the joint minimum and an in-well settle has nowhere to go.
> We asserted this in code rather than assuming it: the zero-step arm was required to be
> **bit-identical to the static read by `np.array_equal` over 256 requests** — an assertion that
> raises if it fails — and it passed, on all three scored seeds, returning the incumbent's held-out
> scores to every printed digit (n-way row **63.612 / 64.327 / 67.653**; pair row **29.717 / 31.617 /
> 31.450**). The only place dynamics can act on this object is **across wells**: with a bilinear
> low-rank coupling `λW` the settled state is the closed form `z* = (βI − λW)⁻¹ β z_req`, which
> returns `z_req` exactly at `λ = 0` and departs from it as `λ` grows. We therefore describe the store
> as **a CLU with settling switched off**, and we report separately what switching it on costs: at
> every strength at which the dynamics actually move the state they **cost** composition — a mean
> paired **−0.531 / −0.510 / −0.388** n-way tuples per corner at the gentlest setting that moves it at
> all (median displacement 0.009), falling to **−39.959 / −46.918 / −39.061** at full strength (median
> displacement 0.39) — while paying **81,312–84,216 B** of read-side bytes the static read does not.

**Sources for every number in that paragraph**

| number | file |
|---|---|
| the identity argument (basis energy + request basin, `a_k ∈ span(G_k)`) | `evidence/outputs/_thinkpad/w6/w6-compose-physics/LOG.md` **06:56Z** |
| `U_k` is the orthonormal basis of `span(G_k)` by SVD of the deployed atoms, **zero bytes**, `span(U_k) == span(G_k)` | `chlu/core/gen_read.py` module docstring + `span_basis()` (lines 91–108) |
| the read is `render_span(store, coefficients(request))`, i.e. `x_k = U_k c_k` | `chlu/core/gen_read.py:126` (`render_span`) and `:248` (`TensorAddress.read`) |
| **`np.array_equal`, 256 requests**, asserted in code, did not raise | `.../w6-compose-physics/LOG.md` **08:32Z** (lines 163–167) |
| `zerostep_bit_identical_to_pca_tensor: true`, `settle_steps: 0`, **3/3 scored seeds** | `.../w6-compose-physics/scored/test_seed{0,1,2}/arms/pca_settle_zerostep/arm.json` |
| 63.612 / 64.327 / 67.653 and 29.717 / 31.617 / 31.450, **equal to `pca_tensor` digit for digit** | `.../w6-compose-physics/SCORED-TABLE.md` (rows `pca_settle_zerostep` and `pca_tensor`) |
| closed form `z* = (βI − λW)⁻¹ β z_req` | `.../w6-compose-physics/LOG.md` **07:24Z** |
| −0.531 / −0.510 / −0.388 (row 4, `f05_conv` vs `pca_tensor`) and −39.959 / −46.918 / −39.061 (`pca_settle`) | `.../w6-compose-physics/SCORED-TABLE.md`, "the ladder's own tie tests" and B1/B2 |
| displacements 0.009 and 0.39 | `evidence/outputs/w6-advisor-report.md` §3 table; val sweep in `.../LOG.md` **08:47Z** (0.0092 / 0.4195) |
| read-side bytes **81,312 B (seed 0) / 84,216 B (seeds 1, 2)** | `.../scored/test_seed{0,1,2}/arms/pca_settle*/arm.json → coupling_bytes.total_read_side_bytes_added`; **see §6 disclosure** |

---

## 2. ⭐ THE SECOND PARAGRAPH — what CRR predicts when settling is switched ON (also carriable)

> Switched on, the relaxation of the field store obeys the shorts' **Curvature–Retention
> Relationship**, and what the CRR predicts is a **time**, not a score. On the damped
> velocity-Verlet map the repository deploys, the one-step Jacobian has
> `tr J = (2−γ)(1 − h²μ²/2)` and `det J = 1 − γ`, and the CRR reads the retention time off its slow
> multiplier as `n_half = ln2 / (−ln|λ_slow|)`, with `ε = 0.05` inherited verbatim from the shorts and
> **nothing fitted**. On 49 held-out corners × 3 seeds the median `|log10(n_measured/n_predicted)|` is
> **0.0433 / 0.0435 / 0.0431** — a factor **1.105** at the median corner — and it beats a
> one-parameter rival **fitted with sight of the held-out measurements** on **49 of 49 corners on
> every seed** (mean paired Δ **−0.5968 / −0.5796 / −0.5982**). The law also gets the *sign of the
> finite-damping correction* right: in the overdamped branch `d ln n_half / d ln γ = 1 + γ/(2−γ)`,
> which exceeds the textbook `+1` at **every** finite `γ`, and it does on this store. **This buys the
> paper a dynamics section and it buys no scoreboard row**: the CRR predicts how long a state takes to
> relax, and the composition, deletion and refusal numbers are measured on a store in which the
> relaxation is switched off.

**Sources**

| number | file |
|---|---|
| `n_half = ln2/(−ln abs(λ_slow))`, `ε = 0.05` inherited, nothing fitted | `evidence/outputs/w6-advisor-report.md` §1 |
| B1 **0.0433 / 0.0435 / 0.0431**; B2 **49/49 ×3**; mean Δ **−0.5968 / −0.5796 / −0.5982** | `.../w6-potentials/RESULTS-LEDGER.md` — **and re-derived by me from `.../w6-potentials/test/s{0,1,2}/results.json → estimand.corners`: 0.043256 / 0.043516 / 0.043102 and −0.596831 / −0.579639 / −0.598172, 49/49 on all three seeds** (`checks/check_03_provenance.py`) |
| factor **1.105** = `10**0.0433` | `checks/check_02_crr.py` probe L (`check_02_crr.json → L_ratios`) |
| `tr J`, `det J`, and that the deployed integrator **is** that map | `chlu/core/integrators.py:28`; `.../w6-potentials/test/s0/results.json → integrator_check.max_abs_diff = 1.884681927188936e-08` (tol 1e-05); independently re-measured by me at **4.387e-08** (`checks/check_02_crr.py` probe G) |
| `d ln n_half / d ln γ = 1 + γ/(2−γ)` is the exact log-derivative | `checks/check_02_crr.py` probe H — max abs(measured − derived) **7.84e-04** over the small-step rows |
| "predicts a half-life, not a taxonomy row" | `evidence/outputs/w6-advisor-report.md` §6.2 and §7; `evidence/outputs/w10-physics-section-evidence/EVIDENCE.md` §1 |

---

## 3. ⛔ WHAT THIS MAY **NOT** SAY

1. ⛔ **It may not say "settling does nothing".** The identity is **in-well only**. With the
   cross-well coupling on, the settled state is `(βI − λW)⁻¹ β z_req` and it moves: in my own check
   the relative displacement runs **7.875e-05 → 7.882e-03 → 3.971e-02 → 1.720e-01** at
   `λ = 1e-4 / 1e-2 / 0.05 / 0.2` (`checks/check_01_zero_dynamics.json → E_rows`). On the real store
   the median displacement at full strength is **0.39** (`w6-advisor-report.md` §3).
2. ⛔ **It may not say the zero-dynamics limit holds for any store.** It holds under four conditions,
   all four met here and all four nameable (§4). Break any one and the settle moves — measured, not
   argued (§4, probes C and D).
3. ⛔ **It may not say "bit-identical" of a settle with steps > 0 on the strength of wave 6.** What
   wave 6 asserted is the **zero-step** arm against the static read (`np.array_equal`, 256 requests,
   3/3 seeds). A **λ = 0, steps > 0** arm was never run: the val sweep identifies `λ = 0` with
   `zerostep` (`.../w6-compose-physics/LOG.md` 08:47Z table). My `checks/check_01` fills that gap
   synthetically (steps 0 / 1 / 10 / 1000, `np.array_equal = True`, `max|diff| = 0.0`), ⚠ **on a
   synthetic store, not on the deployed one** — see the DECLARED NOT-RUN in §7.
4. ⛔ **It may not describe the 1.9e-7 off-span residual as motion.** It is `render_span`'s float32
   matmul rounding and it is **the same size for `pca_tensor`, which never moves at all**
   (`w61-compose-physics/LOG.md` 08:30Z: 1.96e-07 for the incumbent). My check reproduces the scale:
   off-span max **9.950e-08** (`check_01_zero_dynamics.json → A_offspan_max`).
5. ⛔ **It may not say the dynamics are "decoration".** ⚠ The wave-6 tie test says the opposite of a
   tie on the n-way row: `f05_conv` vs `pca_tensor` is **−0.531 / −0.510 / −0.388** with 95 % CIs
   `[−0.878, −0.204]`, `[−1.000, −0.082]`, `[−0.796, −0.041]` — **NOT a tie** on row 4, a **TIE** on
   row 3 only (`SCORED-TABLE.md`). The honest word is **worse than decoration**: the only setting that
   ties is the setting that does nothing.
6. ⛔ **It may not let the CRR imply any scoreboard number.** The CRR is validated on the **field**
   store's settle half-life; the composition / deletion / refusal tables are the **subspace** store
   with settling off. Two different quantities on two different objects.
7. ⛔ **It may not write D18 up as a confirmation.** Measured **−1.1090 / −1.1090 / −1.1086** against
   derived **−1.0050**; `|err|` **0.1368 / 0.1277 / 0.1290** against the naive `−1`'s **0.1417 /
   0.1324 / 0.1333** — the derived form wins by **0.0049 / 0.0047 / 0.0043**, a ~3.5 % margin, and the
   hub reported it as thin (`RESULTS-LEDGER.md`; `EVIDENCE.md` §4.3). ⭐ The correct write-up is
   EVIDENCE.md's: *the underdamped slope is confirmed on the shorts' own testbeds (−1.0020 ± 0.0003)
   and is inconclusive on this store.* **See §5.3 — I can now say exactly what would settle it.**
8. ⛔ **It may never quote `d14_confinement.max_abs_residual`.** It compares `hess_eig(α)` to
   `hess_eig(0) + 2α` — the same algebra on both sides — and is **exactly `0.0` on `n = 60` cells on
   all three seeds** (`.../w6-potentials/test/s{0,1,2}/results.json`, re-read by
   `checks/check_02_crr.py` probe M). I demonstrated the tautology on 200 random Hessians: max
   residual **1.954e-14**, and it cannot fail. ⭐ **Quote `hessian_fd_check` instead:
   3.519263e-11 / 1.202332e-10 / 2.699430e-11 at `eps_fd = 1e-05`, `n = 10` per seed** — a central
   finite difference of the exact gradient of the confined energy, which *can* fail.
9. ⛔ **It may not present "+1.1385 derived vs +1.1368 measured" as a reproducible result.** See §5.
10. ⛔ **It may not cite a registry row for any of this.** ⚠ `claims_matrix.md` and
    `negative_results.md` contain **zero** occurrences of `pca_settle`, `zerostep`,
    `Curvature-Retention` or `1.1385` (grep, 2026-09-21) — the w5/w6 gap the w10 advisor disclosed.
    ⚠ **And "w6" in `claims_matrix.md` (e.g. CM-1, CM-5, CM-7) is CAMPAIGN-1's wave 6, a different
    wave.** A writer who greps `w6` there will quote the wrong campaign.

---

## 4. ⭐ THE FOUR CONDITIONS — name them, because the limit is conditional

The zero-dynamics identity requires **all four**. Each is met by the deployed object, and each is
checkable in `checks/check_01_zero_dynamics.py`.

| # | condition | why it is met here | what breaks if it is not |
|---|---|---|---|
| **C1** | The store's energy is **distance to the atom span** — the atoms are a **basis**, not a cloud of points. | `span_basis()` returns an orthonormal `U_k` from an SVD of the deployed atoms; the honest store energy for a basis is `½ dist²(x, span G_k)` (`w6-compose-physics/LOG.md` 06:56Z). | ⛔ Probe **D**: with the atoms-as-POINTS (softmin/kernel) energy on the *same* content, the settle moves — relative displacement **median 1.7244, max 2.6682**. Wave 6 banked the same null on the real object (charter §E.4: density-over-atoms descent 0.685 → 0.211, `LOG.md` 07:25Z). |
| **C2** | The addressed point lies **in that span**: `a_k ∈ span(G_k)`. | `TensorAddress.read` = `render_span(coefficients(request))`, i.e. `x_k = U_k c_k`, by signature (`chlu/core/gen_read.py:248`, `:126`). | ⛔ Probe **C**: push the address off the span by `δ = 0.10` and the settle moves by exactly `δ/(1+β)` — measured **0.050000** against the closed form **0.050000**, relative error **2.235e-07**. ⭐ The no-op is bought by C2, **not** by settling being weak. |
| **C3** | The state is carried **in span coordinates** (`x_k = U_k z_k`), so `dist²(·, span)` is zero by construction rather than by cancellation. | The wave-6 settle is parameterised in span coordinates and emits through `render_span` — the same function and the same basis as the incumbent (`w61-compose-physics/LOG.md` 08:31Z). | ⚠ Probe **B2**: the *same* energy in channel coordinates is **not** `np.array_equal` — it drifts by **7.447e-08** relative, i.e. float32 rounding. In exact arithmetic both are identities; **only C3 makes it BIT-exact.** This is a real caveat and the paper should say "identical" not "bit-identical" unless the span parameterisation is stated. |
| **C4** | **No cross-well term**, i.e. `λ = 0`. | The zero-step arm has `settle_steps: 0`, so the coupling never acts (`arm.json`). | ⛔ Probe **E**: at `λ = 0`, `array_equal(z*, z_req) = True` and the displacement is exactly `0.0`; at `λ > 0` it is not. **Every unit of motion in this store is cross-well.** |

⭐ **Two structural riders worth a sentence in the paper.**
- **A full-rank well has no "outside" at all.** Wave 6's deployed store had span ranks **64/64/64/58/58**
  with wells 3 and 4 **full rank** (`w61-compose-physics/LOG.md` 08:26Z), so in those two wells the
  in-well settle is trivially a no-op whatever the energy. My synthetic store reproduces the shape
  (ranks 16/12/**20** in dims 40/30/**20**, `check_01_zero_dynamics.json → store`).
- **Quadratic in, linear out.** A quadratic request basin plus a bilinear coupling is a *quadratic*
  energy, so "settling" is an **iterative solver for a linear operator**, and its fixed point is
  `(βI − λW)⁻¹β z_req` in closed form (`w6-compose-physics/LOG.md` 07:24Z). ⭐ **That is not a straw
  man the hub chose — it is what a store whose content is a basis *is*.** My probe E confirms the
  iterate reaches the closed form to **≤1.55e-15** at every `λ` tested.
- ⭐ **The coercivity certificate is not decoration either, and it reproduces from first principles.**
  With `β = 1`, `K = 5` wells and per-pair unit spectral norms, `D = (β/2)I`, `S_kl = ‖W_kl‖₂/2` gives
  `λ* = 1/ρ(D^{-1/2} S D^{-1/2}) = **0.25000000**` and a certificate min eigenvalue at the declared
  `λ = 1` of **−1.50000000** — against the recorded `0.24999999969559716` and `−1.5000000024352236`
  (`scored/test_seed0/arms/pca_settle_zerostep/arm.json`; probe **F**). At the declared strength the
  landscape is **unbounded below**; the used strength is `0.9 λ* = 0.225000`.

---

## 5. ⚠ THE PROVENANCE CHECK ON THE CRR — I ran it, and here is how strong it actually is

The wave-10 referee's **C6** warning is that the programme's sharpest number has its thinnest trail.
I checked it myself rather than quoting `EVIDENCE.md`. `checks/check_03_provenance.py` scans **56
files (53 `.json`, 3 `.md`)** under `evidence/outputs/_thinkpad/w6/w6-potentials/`.

### 5.1 ⭐ What **does** reproduce — and it is most of the result

| claim | ledger | re-derived by me from `test/s{0,1,2}/results.json` | agreement |
|---|---|---|---|
| **B1** median `|log10 n_meas/n_pred|` | 0.0433 / 0.0435 / 0.0431 | **0.043256 / 0.043516 / 0.043102** | 4.4e-05 / 1.6e-05 / 1.8e-06 |
| **B2** mean paired Δ vs the **fitted** rival | −0.5968 / −0.5796 / −0.5982 | **−0.596831 / −0.579639 / −0.598172** | 3.1e-05 / 3.9e-05 / 2.8e-05 |
| **B2** corners better than the fitted rival | 49/49 ×3 | **49/49 on all three seeds** | exact |
| instrument floor | 0.0349 | **0.034919** (`instrument_floor.median_abs_log10`) | exact |
| integrator check | 1.885e-08 | **1.884682e-08** (`integrator_check.max_abs_diff`, tol 1e-05) | exact |
| byte pin | 524,288 B | **524,288 B, `matches_budget = true`, all 3 seeds** | exact |
| **B6** Hessian finite difference | 3.519e-11 / 1.202e-10 / 2.699e-11 | **3.519263e-11 / 1.202332e-10 / 2.699430e-11**, `n = 10`, `eps_fd = 1e-05` | exact |
| corners / rows / field sha | 49 / 64335·64320·64320 | **identical, three distinct field shas** | exact |

⭐ **Verdict on the headline: the CRR result is well-provenanced.** The parameter-free prediction, the
49/49 win over a *sighted* fitted rival, the instrument floor, the integrator identity and the byte
pin all come back out of the recorded JSON. ⭐ Independently, `checks/check_02_crr.py` probe G shows
the **production** integrator's Jacobian is the shorts' map to **4.387e-08**, so the law is being
applied to the object the repository actually runs.

### 5.2 ⛔ What does **not** — and it is exactly the sharpest number

- `1.1368` and `1.1367` appear in **two** files: `LOG.md` and `RESULTS-LEDGER.md`. `1.1385` appears in
  **one**: `LOG.md`. ⛔ **Zero hits in any of the 53 JSON artefacts.**
- `results.json` contains no `slope`, no `d17`, no `d18`, no `fit_window` and no `per_cell` key;
  `n_v_curves` is a **count** (`field: 3675`), not the curves.
- The derived constant sits at **γ = 0.243303** (`1 + γ/(2−γ) = 1.1385`, reproduced exactly in probe
  J), and that **is not a grid point**: `pins.gammas = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
  0.35, 0.5]`, whose derived slopes are `1.0010 … 1.1111 (γ=0.2) … 1.2121 (γ=0.35) … 1.3333`. So
  **+1.1385 is an aggregate over per-cell fit windows that were not saved.**

⭐ **What I can add to `EVIDENCE.md`'s finding: the *derived* half is fully reproducible.** Probe H
shows `1 + γ/(2−γ)` **is** the exact log-derivative of the exact slow root in the small-step limit —
at γ = 0.243303 the numeric derivative is **+1.138526** against the closed form **+1.138500**
(|diff| **2.61e-05**) — and that the error grows with `x = (hμ)²` (at `x = 1e-3` it is 3.09e-02).
⭐ I also **independently reproduced the derivation trap** `EVIDENCE.md` §4.1 warns about: taking the
γ-derivative at fixed `h/h*` instead of fixed `h` returns **−1.0260** where the correct value is
**+1.0256** — a pure sign inversion, and my number lands on their reported `−1.026` to 4 decimals.

> ⭐⭐ **My verdict on C6, stated plainly.** The **law** is strong and reproducible; the **pair of
> constants** is single-source prose. Nothing I ran contradicts `+1.1368` — it is consistent with the
> derivation and with the recorded grid straddling γ = 0.2433 — but it is **not recoverable from any
> artefact**, and the quantity it reports (a slope aggregated over unrecorded per-cell fit windows) is
> the one thing in the physics section a referee cannot check.
> **Recommendation:** ⭐ the paper leads the dynamics section with **B1/B2** (parameter-free, 49/49
> over a *sighted* rival, fully reproducible) and with D17's **qualitative** content (*the overdamped
> slope exceeds +1 at every finite γ*, verified across the whole band). ⚠ The specific pair
> **"+1.1385 derived vs +1.1368 measured"** either (a) moves to a footnote explicitly marked
> *reported by the wave-6 hub, not reproducible from the deposited artefacts*, or (b) is re-emitted
> first. ⭐ **(b) is cheap and is the better option**: one JSON of per-cell
> `(γ_window, x_window, slope_measured, slope_derived)` rows from the w6 branch, no scored run, no
> GPU. ⛔ **As it stands the number should not carry a main-text sentence on its own.**

### 5.3 ⭐ D18 — I can now say exactly what would settle it, in one recorded row

`EVIDENCE.md` §4.3 found a candidate explanation it "could not adjudicate". I can narrow it to a
single free parameter, and the reason is a small theorem rather than a guess.

**In the underdamped branch the slow multiplier is exactly `√(1−γ)`** — independent of the step and
of the curvature. Probe K measures `abs( abs(λ_slow) − √(1−γ) )` = **0.000e+00** over 7 underdamped cells.
Therefore `n_half = 2 ln2 / (−ln(1−γ))` is **curvature-independent** (which is also what
`hub/potentials`' B3 bar tests, and what the inventory spoke found inverts the non-existent
"Mexican-hat width → interpolation floor" law, `w6-advisor-report.md` §2), and its log-derivative is
`D18 = −γ/[(1−γ)(−ln(1−γ))]` **exactly** — probe K matches the numeric derivative to **1.43e-09**.

⭐ **Consequence: a measured floor slope has exactly one degree of freedom, and it is γ.**
- derived **−1.0050** ⇒ γ = **0.009917**;
- measured **−1.1090** ⇒ γ = **0.184104**.
(Both inverted in probe K; both match `EVIDENCE.md` §4.3 to six decimals.)
- ⭐ **And the branch itself constrains that γ.** Probe N computes `γ_crit(x)`: the underdamped branch
  only *reaches* γ = 0.1841 at `x = (ημ)² ≥ **0.010305**`, i.e. `ημ ≥ 0.1015`, i.e. **μ ≥ 2.030** at
  the recorded `pins.eta = 0.05`. `EVIDENCE.md` §4.3 reports `γ*_exact ≈ 0.1185` for the modes in this
  grid — **below** 0.1841, the wrong side of the boundary.

> ⛔ **So D18 is not a confirmation, and it is also not a mystery.** It is a **fit-window** question
> with one unknown. ⭐ **One recorded row — the underdamped window's `(γ, x)` — settles it either way,
> at zero compute.** Until that row exists, write D18 as `EVIDENCE.md` prescribes: confirmed on the
> shorts' own testbeds, **inconclusive on this store**.

### 5.4 ⚠ And one honesty note about "a factor 1.10"

`10**0.0433 = **1.1048**` (probe L). The measurement's **own** error floor, taken on an exact
quadratic where the CRR is true by construction, is `10**0.0349 = **1.0837**`. ⭐ **Read the right
way this makes the law look better, not worse** — the CRR's error is close to the instrument's
resolution (the excess is **0.0084** in log10). ⚠ **But it also means "a factor 1.10" is not a clean
statement about the law**: about four fifths of that 10.5 % is the instrument. The paper should quote
**both** numbers, exactly as `RESULTS-LEDGER.md` and `results.json → instrument_floor` do, and the
bar (0.10 ⇒ factor 1.2589) beside them.

---

## 6. ⚠ ONE DEFECT I FOUND IN THE ARTEFACTS (mine to report, not to fix)

`checks/check_04_byte_ledger.py` re-derives the settle ladder's byte ledger per seed from
`scored/test_seed{0,1,2}/arms/*/arm.json`.

- ⭐ **The dial holds exactly.** Within each seed **all seven settle legs are byte-identical** —
  address book, coupling, standardisation and store — so `pca_settle − pca_settle_zerostep` really is
  the dynamics and nothing else. ⭐ **The store pin is exact on every leg and every seed: 524,288 B,
  `matches_budget = true`.**
- ⚠ **But `SCORED-TABLE.md` prints ONE read-side column, and it is the seed-0 value.**
  Address book: **189,344 B (s0)** vs **195,064 B (s1, s2)**. Read-side added: **81,312 B (s0)** vs
  **84,216 B (s1, s2)**. The same single value 81,312 B is printed in the wave-6 LOG's ledger table
  and in `w61-compose-physics/LOG.md` 08:26Z.
- ⛔ This changes **no verdict**: the settle arm's byte cost is **understated by 2,904 B on two of
  three seeds**, so the correction makes the losing arm **worse**, never better. ⚠ The paper should
  quote the range **81,312–84,216 B**, or the per-seed values, not a single number.
- ⚠ Note also the dev/val leg carries a third ledger (191,424 / 79,872 / 2,496 ⇒ **82,368 B**,
  `dev/val_seed0/arms/pca_settle_zerostep/arm.json`), which is a *val* object and must not be mixed
  with the scored ones.

---

## 7. ⛔ DECLARED NOT-RUNS

| not run | reason |
|---|---|
| Any re-run of a wave-6 **scored** leg, or any re-emission of the deployed store's read | ⛔ The fitted stores are **not in the evidence base**: `find /c/chlu/evidence/outputs/_thinkpad/w6 -name '*.pkl' -o -name '*.npz'` returns **nothing**. Only the JSON summaries and the prose are deposited. My checks therefore exercise the **production code path on a synthetic store**, and I say so on every number. |
| The missing wave-6 arm: **λ = 0 with steps > 0** on the deployed object | ⛔ It was never run in wave 6 (the val sweep identifies λ=0 with `zerostep`, `LOG.md` 08:47Z) and I cannot run it without the store above. ⭐ `checks/check_01` probe B runs it synthetically (steps 0/1/10/1000, `np.array_equal = True`). |
| Re-emitting the per-cell `(γ_window, x_window, slope_measured, slope_derived)` rows that would fix C6 | ⛔ Not mine: it needs the w6 branch's own code and its fitted field store. ⭐ It is the single cheapest repair available and I have specified exactly what it must contain (§5.2, §5.3). |
| Any scored run, any GPU, any table row | ⛔ Out of scope by the task: *"Scored runs: none. Small numerical sanity checks only."* |
| A registry number for the zero-dynamics finding or the CRR | ⛔ Registry numbering is not mine to allocate, and N89/N90 are already double-allocated (`w10-advisor-report.md` §5). ⚠ I record the **absence** (§3 item 10) rather than inventing a row. |
| Reading `$CHLU_CLAUDE_ROOT` | ⛔ Forbidden by the standing rules: frozen, stale snapshot. Never opened. |

---

## 8. ⭐ THE NUMERICAL SANITY CHECK A READER CAN RE-RUN

Five scripts in `checks/`, pure CPU, **2 threads**, together a few seconds. Each writes its own JSON
beside itself. Every one was run to completion in-turn; no background job was left running and no
process was killed.

```
cd <your CHLU worktree>
OMP_NUM_THREADS=2 python checks/check_00_env.py            # asserts chlu.__file__ is YOUR worktree
OMP_NUM_THREADS=2 python checks/check_01_zero_dynamics.py  # the store side, probes A-F
OMP_NUM_THREADS=2 python checks/check_02_crr.py            # the CRR side, probes G-N
OMP_NUM_THREADS=2 python checks/check_03_provenance.py     # re-derives B1/B2, scans for the slopes
OMP_NUM_THREADS=2 python checks/check_04_byte_ledger.py    # the per-seed byte ledger
```

`check_02`, `check_03` and `check_04` read the evidence base read-only; override the path with
`W12_EVIDENCE` / `W12_COMPOSE` if it is mounted elsewhere. `check_01` and `check_02` probe G import
production `chlu` code (`chlu.core.gen_read`, `chlu.core.integrators`) — `check_00` is what proves
that code came from your own worktree and not from the shared editable `.pth`.

**Results as run here, 2026-09-21 (python 3.11.9, numpy 2.4.1, jax 0.9.0):**

| probe | what it shows | number |
|---|---|---|
| 00 | `chlu.__file__` inside the worktree | `C:\chlu\worktrees\CHLU-w12-clue-theorist\chlu\__init__.py` — **PASS** |
| A | the addressed point is in `span(G_k)` | off-span max **9.950e-08** (real store: 1.96e-07) |
| B | in-well settle is a no-op at **0 / 1 / 10 / 1000** steps | `np.array_equal = True`, `max|diff| = 0.0` |
| B2 | the same energy without span coordinates | `array_equal = False`, rel drift **7.447e-08** |
| C | address off the span by 0.10 | displacement **0.050000** vs closed form **0.050000** (rel err 2.235e-07) |
| D | atoms as points instead of a basis | rel displacement **median 1.7244 / max 2.6682** |
| E | `λ = 0` returns the static read exactly | rel displacement **0.0**, `array_equal(z*, z_req) = True`; iterate vs closed form ≤ **1.55e-15** |
| F | the coercivity certificate from first principles | `λ* = **0.25000000**`, min eig = **−1.50000000** |
| G | production integrator == the shorts' map | max&#124;diff&#124; **4.387e-08** (recorded: 1.884682e-08) |
| H | D17 is the exact log-derivative | max&#124;diff&#124; **7.84e-04** at small step; **2.61e-05** at γ = 0.243303 |
| I | the fixed-`h/h*` sign trap | **−1.0260** (EVIDENCE.md reports −1.026) |
| J | `1 + γ/(2−γ) = 1.1385` | γ = **0.243303**, **not** on `pins.gammas` |
| K | the underdamped floor | `abs(abs(λ)−√(1−γ)) = ` **0.000e+00**; D18 exact to **1.43e-09**; γ = **0.009917** / **0.184104** |
| L | "factor 1.10" | **1.1048**; instrument floor **1.0837**; bar **1.2589** |
| M | `d14_confinement` is a tautology | **0.0** on `n = 60` ×3 seeds; 200 random Hessians max **1.954e-14** |
| N | what would settle D18 | reaching γ = 0.1841 needs `x ≥ **0.010305**` ⇒ `μ ≥ **2.030**` at `η = 0.05` |
| 03 | B1/B2 re-derived from `results.json` | **0.043256 / 0.043516 / 0.043102**; **−0.596831 / −0.579639 / −0.598172**; **49/49 ×3** |
| 03 | where the D17/D18 constants live | `1.1368`/`1.1367`: 2 `.md` files · `1.1385`: 1 `.md` file · **0 JSON** (of 53) |
| 04 | the byte ledger | within-seed **byte-identical**; store pin **524,288 B** exact; read-side **81,312 / 84,216 / 84,216 B** |

⭐ **All five scripts exit 0 as run here.** `check_04` asserts only the three substantive properties
(within-seed byte identity, the store pin, the zero-step flag) and prints the cross-seed column
mismatch of §6 as a DISCLOSURE rather than a failure.

⭐ **The checks regenerate from the bundle alone.** `w12-clue-theorist.bundle` was cloned into a
clean directory and all four substantive scripts re-run there: **exit 0 each**, and all four JSON
outputs **byte-identical** to the ones in `checks/` (`diff -q`, 2026-09-21). ⛔ Nothing in this
directory depends on state that only exists in the worktree it was produced in, apart from `chlu`
itself, which `check_00` is there to pin.

---

## 9. ⛔ WHAT WOULD PROVE THIS SPOKE WRONG — and where I stand on each

| falsifier from the task | status |
|---|---|
| *"The zero-dynamics limit holds only under conditions we do not meet ⇒ name the conditions."* | ⭐ **The conditions are named (§4: C1–C4) and all four are met by the deployed object.** ⚠ One is weaker than the prose suggests: **C3 is what makes it BIT-exact** — without the span parameterisation the same energy drifts by float32 rounding (probe B2). The paragraph in §1 is written to survive that. |
| *"CRR's provenance does not survive your check ⇒ say so."* | ⚠ **It survives in part, and I say which part.** B1/B2/floor/integrator/byte pin/B6 **all re-derive from the artefacts** (§5.1). ⛔ **The +1.1385 / +1.1368 pair does not** — zero JSON hits in 53 files, and its γ is not on the recorded grid (§5.2). ⛔ **That number leaves the main text unless it is re-emitted.** |
| the paragraph the paper can carry | ⭐ §1, verbatim, with a per-number source table. |
| the "what this may NOT say" list | ⛔ §3, ten items. |
| a re-runnable numerical sanity check on the store | ⭐ §8 and `checks/` — 15 named probes across five scripts, four JSON outputs, a few seconds on 2 threads. |

⭐ **One thing I would add to the paper that is not currently anywhere:** the sentence in §4's riders —
*a store whose content is a basis has a quadratic energy, so "settling" on it is an iterative linear
solve.* ⛔ It means the negative result of wave 6 is **not** "we picked a weak landscape"; it is a
property of the object. ⭐ That is the strongest honest framing available for the dynamics section,
and it is what makes the zero-dynamics claim a **result** rather than an ablation.

---

**Written by** `w12 spoke/clue-theorist`. Log: `LOG.md`. Checks: `checks/`.
`START_UTC.txt` / `END_UTC.txt` in this directory. Branch `w12/clue-theorist`; bundle
`w12-clue-theorist.bundle` (that branch only).
