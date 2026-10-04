# hub/potentials — LOG

Target: do the shorts' potential laws hold on the field store? Dial: the kernel and the dynamics
(width, separation, friction, mass). Base `base/iter6 = df025685`. Worktree
`/c/chlu/worktrees/CHLU-w6-potentials`. Scored budget: `potentials` 3.

- 2026-09-20T06:25Z — START_UTC.txt written to `/c/chlu/outputs/w6-potentials/START_UTC.txt`
  (2026-09-20T06:25:56Z). No live START_UTC existed in that directory beforehand (directory was empty).
- 2026-09-20T06:26Z — Booted on charter -> w6-hub-common.md -> w6-hub-potentials.md, nothing else.
- 2026-09-20T06:27Z — Env verified: `chlu.__file__` =
  `C:\chlu\worktrees\CHLU-w6-potentials\chlu\__init__.py` (inside my worktree). jax 0.9.0, numpy 2.4.1,
  equinox 0.13.4. CHLU_ROOT first on PYTHONPATH. 3 threads. git status clean at df02568.
- 2026-09-20T06:27Z — Precondition check: `/c/chlu/outputs/w6-shorts-inventory/` is EMPTY — the
  inventory spoke has not published, not even a partial. I am the consumer of that spoke. Decision:
  do not block. I read the five PDFs in `/c/chlu/shorts/` myself for VERBATIM formulas (primary
  source, strictly better than a paraphrase), and cross-check against the spoke's inventory when it
  lands. Any law I cannot find verbatim in a PDF is declared ABSENT and NOT invented.

## The shorts: what is actually in them

- 2026-09-20T06:35Z — Five PDFs, three distinct papers, confirmed by md5:
  `13c6f496b1f600825ecc11c3708fa1c3` = `F_v1_LCFM_finalSub_2.pdf` = `F_v1_TTCL_finalSub_2.pdf` (V1, ONE paper,
  byte-identical, venue rename only). `4eba56c480472f9c9fb88ce690bc0436` = `F_v2MemShort_bento_final2.pdf` (V2).
  `c23e6fd9acd259a7d93ff9b48fb8a325` = `F_v5_PaLM_Subm3.pdf` = `F_v5_stody_Subm3.pdf` (V5, ONE paper,
  byte-identical). Extracted with `pdftotext -layout` (mingw64 poppler).
- 2026-09-20T06:36Z — ⛔ **LAW 1 OF MY TASK FILE IS ABSENT FROM THE SHORTS. I do not invent it.**
  The task asks: "Mexican-hat kernels: does width/separation set the interpolation floor and the refusal gap as
  the shorts derive?" Exhaustive search of all three papers:
  * "Mexican" occurs exactly ONCE in the whole corpus, V2 App. M.4 line 654: "on the analytic Mexican hat a
    cancellation-free Hessian returns relative deviation 1.1-2.2e-16". That is the **symmetry-breaking sombrero
    POTENTIAL** used as a float64 precision testbed for GMOR, ⛔ NOT a Mexican-hat (difference-of-Gaussians,
    centre-surround) **KERNEL**.
  * "interpolation" — **0 occurrences in all three papers.** There is no interpolation-floor derivation.
  * "on-center", "center-surround", "surround", "lateral inhibition", "DoG", "RBF", "receptive" — **0 occurrences.**
  * "kernel" occurs only in V1 App. G/I as **"certified Markov kernel"** (MCMC acceptance kernel), a different
    object entirely.
  * "refus" occurs twice and neither is a derivation: V1 line 533 ("refuses"), V5 line 838
    ("refuse-and-relocate" allocator baseline, 43/64 admits). There is **no derived refusal-gap formula.**
  * The only separation quantity in the corpus is V5 line 808/829: lattice-packing slots carry a
    "minimum-separation certificate", measured boundary **1.540000**. That is a **placement/deletion** certificate,
    not a kernel-width law, and it predicts no floor and no gap.
  ⇒ **DECLARED ABSENT:** there is no derivation in the shorts by which Mexican-hat kernel width and separation
  set an interpolation floor or a refusal gap. Per the task file's own falsification list, I say so and stop.
  Scope limits 1 and 2 already forbade claiming those rows; the law behind them does not exist either.
- 2026-09-20T06:37Z — ⭐ **LAW 2 OF MY TASK FILE IS PRESENT, COMPLETE, AND CLOSED-FORM.** The friction/mass
  settle-time law is the **Curvature-Retention Relationship (CRR)**, V2 Sec. 3 + App. C Eqs. (3)-(6), with V5
  supplying the instrument and the critical point. Quoted verbatim below in the PREREG. My three scored runs
  go there.

## The law, reconstructed and cross-checked against the papers' own numbers

- 2026-09-20T06:52Z — The CRR is stated in V2 Sec. 3 and derived in V2 App. C Eqs. (3)-(6). ⚠ `pdftotext`
  does not render the paper's greek glyphs, so the formulae arrive with gamma/mu/eta missing. I did NOT guess
  them: I reconstructed the map, then **validated the reconstruction against two numbers the papers report
  independently**, and both land.
  * One-step map on a mass-whitened normal mode of curvature `mu^2`, step `eta`, damping `gamma`, `h = mu*eta`:
    `A = [[1 - h^2/2, eta(1 - h^2/4)], [-(1-gamma) mu^2 eta (1 - h^2/4), (1-gamma)(1 - h^2/2)]]`,
    `det A = 1 - gamma`, `tr A = (2-gamma)(1 - h^2/2)`.
  * ⭐ **The map the paper writes IS the map this repo integrates.** I read `A` off
    `chlu.core.integrators.velocity_verlet_step` by finite difference on `H = p^2/2M + (1/2) mu^2 q^2` and
    compared to V2 Eq. (3): `max|A_paper - A_code| = 1.0e-07 .. 9.0e-05` (float32 gradient noise), with
    `det A` and `tr A` matching the paper's identities to 1e-11. The repo's integrator is the shorts' object.
  * **Band boundaries, V2 Eq. (5)**: `h*^2 = 2(1-sqrt(1-gamma))^2/(2-gamma)`, `h_f^2 = 2(1+sqrt(1-gamma))^2/(2-gamma)`.
    Computed: gamma=0.05 -> h*=0.0256431 (paper's stated `h* ~= gamma/2` = 0.025 ✓), h_f=1.9998356
    (paper's `h_f = 2 - O(gamma^2)` ✓). gamma=0.002 -> h*=0.0010010 vs gamma/2=0.001 ✓.
  * ⭐ **CROSS-CHECK 1 (V2 Sec. 4.1 + App. C):** underdamped curvature-independent floor
    `n_1/2 = 2 ln2 / (-ln(1-gamma))`. At gamma=0.05 this is **27.02681**. **V2 reports 27.03.** ✓
  * ⭐ **CROSS-CHECK 2 (V5 App. line 684):** V5 reports `lambda_ret = 0.998999499` with `n_1/2 = 692.5` at
    gamma=0.002. My form gives `|lambda| = sqrt(1-gamma) = 0.9989994995` (9 digits ✓) and
    `n_1/2 = ln2/(-ln|lambda|) = 692.4538`. **V5's own measured 692.5.** ✓ (V5 quotes the coarser
    `2ln2/gamma = 693.1` beside it; the exact form is the one that matches its measurement.)
- 2026-09-20T06:54Z — ⚠ **A CORRECTION TO THE SHORTS, stated as one.** V2 Sec. 3 and Eq. (6) print the
  overdamped half-life as `n_1/2 ~= 2 ln2 / [(2-gamma)(mu eta)^2]`. **That expression is missing a factor
  `gamma` in the numerator.** Expanding the characteristic polynomial at `lambda_+ = 1 - epsilon`:
  `1-(2-gamma)+(1-gamma) = 0` cancels, leaving `-gamma*epsilon + (2-gamma)h^2/2 + epsilon^2 = 0`, so
  `epsilon = (2-gamma)h^2/(2 gamma)` and **`n_1/2 ~= 2 gamma ln2 / [(2-gamma)(mu eta)^2]`**. V2's own printed
  error term `O(h^4/gamma^3)` is exactly what that expansion produces and is only dimensionally consistent
  WITH the gamma — so the paper's derivation has it and its displayed line has lost it. Numerically, at
  gamma=0.05, eta=0.05: the gamma-corrected asymptote tracks the exact eigenvalue to **0.04 %** at
  `h/h* = 0.039` and 4 % at `h/h* = 0.39`; the printed form is off by exactly `1/gamma` = 20x.
  ⛔ **I do not pre-register against the printed line.** I pre-register the **exact eigenvalue form**, which is
  unambiguous and is what both cross-checks above validate. The slope claim (`d log n_1/2 / d log mu^2 = -1`,
  V2's fitted -0.985) is unaffected: gamma is a constant prefactor.

## The inventory spoke landed — redirection, and a correction to my own bullet

- 2026-09-20T08:40Z — **REDIRECTION RECEIVED (Head, mid-turn).** `/c/chlu/outputs/w6-shorts-inventory/
  shorts-experiments-inventory.md` is on disk (66 KB, published 08:31Z). Section 1.3 lists what IS present
  and safe; section 2 is the verbatim derived-law catalogue with paper and section. I was pointed at
  **§2.3** the friction dial and its critical point (V5 §2.1, derived App. G.3), **§2.4** finite-temperature
  FDT noise/diffusion and the sign flip, **§2.5** the friction hole / memory vault (V5 §2.2, App. G.7),
  **§2.7** reach/escape and access certificates (V1, App. H) — any of which can carry the scored runs.
- 2026-09-20T08:41Z — ⭐ **The inventory independently confirms my §1 finding, in its own words.** Its §1.1:
  *"Mexican-hat width -> interpolation floor — ABSENT. Do not pre-register against it."*, *"the string
  'interpolation' does not occur once in any of the three papers"*, *"Mexican hat occurs exactly once ...
  in V2 App. M.4, and it is not a law"*. Two independent readings of the corpus, same verdict. ⛔ My task
  file's law 1 has no source and is not pre-registered against.
- 2026-09-20T08:42Z — ⚠ **CORRECTION TO MY OWN 06:54Z BULLET. I was wrong about the paper; I was right
  about the algebra.** I wrote that V2 Eq. (6) "is missing a factor gamma" and called it a correction to the
  shorts. It is not. The inventory quotes V2 App. C Eq. (6) verbatim as
  `n_half = ln2/(-ln lambda_+) ~= 2*gamma*ln2 / [(2-gamma)(eps*mu)^2]`, slow root
  `lambda_+ = 1 - (2-gamma)h^2/(2*gamma) + O(h^4/gamma^3)` — **the gamma is in the paper.** My
  `pdftotext` extraction silently dropped every greek glyph, and the gamma with them. ⭐ What survives is
  the useful half: I re-derived `eps = (2-gamma)h^2/(2*gamma)` from the characteristic polynomial without
  seeing the paper's line, and it **matches V2 Eq. (6) exactly**. ⛔ No correction to the theory is claimed
  here; the pre-registration is unaffected because it was always against the EXACT eigenvalue, never
  against the printed asymptote.
- 2026-09-20T08:43Z — The inventory's D-numbers confirm, verbatim and to the letter, every closed form I
  had already implemented in `chlu/core/w6_potentials.py`: **D3/D4** (the per-mode step matrix, `tr J`,
  `det J`, `lambda_pm`), **D5** (overdamped half-life), **D7** (curvature-independent floor), **D8**
  (`h* = sqrt(2/(2-gamma))(1-sqrt(1-gamma)) = gamma/2 + O(gamma^2)` — identical to my form), **D9**
  (`h_f = 2 - O(gamma^2)`), **D12** (the latch `q_inf = q_0 + eps*p_0/(M*gamma)`), and **D15** (the spectral
  mass `mu_k^2 = lambda_k(M^-1/2 grad^2 V M^-1/2)` — exactly what `curvature_spectrum` computes).
- 2026-09-20T08:44Z — **DECISION on where the 3 scored runs go, and why.** My task file's law 2 is
  *"friction and mass: do the derived settle-time laws match measurement?"* Inventory §1.3 says friction,
  mass and settle time ARE derived in closed form; §2.2 is the band structure and §2.3 is the friction dial.
  ⭐ **They are the same object measured along two axes** — sweeping `mu^2` at fixed `gamma` gives §2.2's
  bands, sweeping `gamma` at fixed `mu^2` gives §2.3's V-curve — so I take **§2.2 AND §2.3 together**, which
  is exactly my dial (width, separation, friction, mass) and exactly my task's law 2. ⛔ §2.4/§2.5 (finite T,
  the vault) need `T > 0` with `langevin_noise="fdt"`, a Newtonian kinetic mode and a **flat coset
  direction**; the field store's smallest measured curvature is `mu^2 = 0.16`, not zero, so it has no coset
  and those laws are OUT OF THEIR OWN STATED SCOPE on this object. ⛔ §2.7 (reach/escape) is V1's Lorentz
  squeeze on a designed testbed and is not this hub's dial. I name them as available and do not spend on them.
  ⭐ **§2.3 makes the test sharper, which is why I am taking the redirection rather than just noting it:**
  D16 predicts the V-curve minimum at `gamma_crit = 2*eps*mu` and D17/D18 predict the two branch log-slopes
  as `1 + gamma/(2-gamma)` and `-gamma/[(1-gamma)(-ln(1-gamma))]` — **explicitly NOT +1 and -1**. That turns
  the task file's "no better than a fitted one-parameter curve" falsifier into a real contest: the naive
  +-1 slopes ARE the one-parameter rival, and the paper's derivation says they are wrong.

- 2026-09-20T08:50Z — ⚠ **A tension between the inventory's §4 and my task file, resolved and stated.**
  The inventory's §4 mapping table routes **D5-D19 to `hub/retention`** and gives `hub/potentials`
  **D15, D46-D50, D34-D45, D14**. But (a) my task file's law 2 is verbatim *"Friction and mass: do the
  derived settle-time laws match measurement?"*, which IS D5-D19; (b) the Head's mid-turn redirection named
  **§2.3 (D16-D19)** to me explicitly; and (c) `hub/retention`'s budget is the **deletion** target
  (w6-common §2), so it is not spending its 2 runs on the CRR. ⭐ **Resolution:** I take §2.2+§2.3 as my
  scored law set, and I additionally take the two laws §4 assigns me that are inside my dial and measurable
  on this object:
  * **D15** — `mu_k^2 = lambda_k(M^-1/2 grad^2 V M^-1/2)` — is not an extra test, it IS my measurement
    instrument for `mu^2`, already implemented as `w6_potentials.curvature_spectrum`.
  * **D14** — *"with a confinement term `alpha||q||^2`, every Hessian eigenvalue is lifted by `2*alpha`, so
    a flat direction becomes `mu^2 = 2*alpha` and the overdamped rate caps the lifetime at
    `tau_max = Gamma/(2*alpha)`, damping rate `Gamma = gamma/eps"` — a second, independent, parameter-free
    prediction **about the potential itself**, which is this hub's name. Cheap to test and I take it.
  ⛔ **NOT taken, with reasons:** D46-D50 (GMOR, tree-level Goldstone, SO(2) invariance) need a vacuum
  orbit / exactly-flat coset; the field store's smallest measured curvature is `mu^2 = 0.16`, so it has
  none and those laws are outside their own stated scope here. D34-D45 (V1's causal box, Lorentz squeeze,
  gated channel) are the relativistic reach apparatus on a designed testbed and are not this hub's dial.
  ⛔ **A2 (`mu^2 ∝ A/l^2`) is NOT used** — the inventory's §3 flags it as an undefined table cell and says
  in terms *"This is the closest thing in the corpus to a 'width' law and it must not be treated as one."*
  ⛔ **A5 (minimum separation 1.540000) is NOT used** — §3 flags it as a measured constant, not a law.

## A third independent confirmation of the law, from a peer hub

- 2026-09-20T09:05Z — Reading `/c/chlu/SCORED-LEDGER.md` to write my own row, I find **`w6 hub/retention`
  (row 2026-09-20T07:40Z) has independently transcribed the same CRR** into `chlu/core/gen_retention.py`
  on its own branch, and validated it against the papers' printed numbers before use. Its reported
  self-check numbers: floor **27.026815** vs printed **27.03** at gamma=0.05; curvature-independence to
  1.6e-13; `h* = gamma/2`; stability at `h = 2` exactly for every gamma; overdamped slope **-1.00096**
  (predicted -1); onset exponent **0.50006** (predicted 1/2); the exact latch. ⭐ **Every one of those
  matches what `chlu/core/w6_potentials.py` computes here** — my floor is 27.02681, my `h*` is
  `gamma/2 + O(gamma^2)`, my `h_f` is 2. Three independent reconstructions of V2 App. C (mine from the
  characteristic polynomial, the inventory spoke's from the PDF text layer, retention's transcription) agree.
  ⛔ `gen_retention.py` is NOT in my worktree and I do not import it: my base is `df02568` and it lives on
  a peer's branch. My module is my own.
- 2026-09-20T09:06Z — ⚠ **No duplication with `hub/retention`, and the distinction is the point.**
  Retention uses the CRR as a **generator**: it derives a decay schedule from the law and spends its runs
  on the `deletion` target (2/3 spent). ⭐ **I use the CRR as the thing UNDER TEST**: I predict from it and
  measure against a nonlinear rollout on the field store's landscape, on the `potentials` target. Retention
  asks "what does the law let us build?"; I ask "is the law true of this object?" Neither answers the other.

## Building the rig, and what it cost to make it affordable

- 2026-09-20T08:30Z — `chlu/core/w6_potentials.py` written. LAW A = the shorts' CRR (exact eigenvalue,
  band edges, floor, asymptote); LAW B = the kernel curvature identity, derived here from the repo's own
  `gen_landscape.store_energy` and **labelled as mine, not the shorts'**:
  `grad E = x - <a>_w`, `Hess E = I - Cov_w(a)/sigma^2`. Checked against autodiff: `max|H_ad - H_closed|`
  = 1.1e-16 / 3.1e-15 / 2.2e-16 over three shapes; gradient to 4.4e-16.
  ⭐ Two consequences that make the hub affordable and make the width question answerable:
  (1) `Hess` differs from the identity only on a rank-63 subspace, so the whole curvature spectrum comes
  off a **64x64 Gram matrix** instead of a 2048x2048 Hessian; (2) for two equal-weight atoms at separation
  `Delta`, curvature along the separation axis is exactly `1 - Delta^2/(4 sigma^2)` — **one merged minimum
  below `Delta/sigma = 2`, two minima above**, verified to machine precision at
  `Delta/sigma = 0.5/1.0/1.5/1.9/2.0/2.1/3.0/4.0`.
- 2026-09-20T08:33Z — ⭐ **The repo's integrator IS the shorts' map, asserted not assumed.**
  `assert_map_matches_repo_integrator` rolls 5 steps of the float64 numpy map and of
  `chlu.core.integrators.velocity_verlet_step` on the same `H` and compares positions:
  `max|diff| = 5.5e-09 / 1.3e-08 / 1.9e-08` over three `(mass, gamma, eta)`. **The leg refuses to run if
  this fails.**
- 2026-09-20T08:34Z — ⚠ **THE knn_density TRAP, CHECKED AND CLEARED, WITH NUMBERS, ON THE REAL STORE.**
  `well_sigma / nn_median` across the five wells = **0.871, 0.853, 0.856, 2.097, 1.210**. Not 1.0 on any
  well, and on **both sides** of 1. The field store's width is the median within-value RMS spread of POOL
  items about their value's coordinate; `gen_landscape.atom_bandwidths(rule="nn_median")` — the spacing of
  stored items, which is what sank `knn_density` at 0.0203 — is a **different number on every well**.
  ⛔ The failure is not inherited. ⚠ Worth recording that the trap is real and live in this codebase:
  `atom_bandwidths` defaults to `rule="nn_median"` and its own docstring calls the bandwidth "the
  interpolation knob". The field store does not use it.
- 2026-09-20T09:10Z — **Cost. Three rounds of it, logged because the first two were wrong.**
  (i) First rig: one rollout per `(mass, gamma, amplitude)` cell, `slack = 6`, cap applied to the ROLLOUT
  rather than the PREDICTION, so cells rolled to 90,000 steps. One corner on 2 of 5 wells took 51 s and a
  probe on wells 0-1 did not finish in 8 minutes. Killed (own PID only).
  (ii) Batched the grid: per-probe mass/damping/confinement in `verlet_rollout_batch`, cells bucketed by
  required length, so the grid costs a few large matmuls per step instead of 90 small ones. Measured
  throughput on the real shapes: **3.5 GFlop/s, 69 us per probe-step at d=940**.
  (iii) Sized the grid against that number instead of guessing. The full grid is **300 min/seed** — not
  affordable. The shipped main grid (5 masses x 9 gammas x 3 modes x 1 amplitude, cap 5000, slack 3) is
  **22 min/seed** with 74 underdamped + 61 overdamped probes and 22/135 censored — ⭐ **both bands still
  covered, which is what the test needs.** The richer grid (2 amplitudes, 6 modes, confinement rolled)
  runs on a declared SUBSET of corners and says where the law stops, not whether it holds.
- 2026-09-20T09:20Z — ⚠ **Caught and removed a tautology in my own D14 check before it ran.** My first
  `d14_spectrum_check` computed the "measured" confined spectrum as `curvature_spectrum(...) + 2*alpha`
  and compared it to `base + 2*alpha` — the same algebra on both sides, which can only ever pass.
  Replaced with `hessian_fd_check`: the measured side is now a **central finite difference of the exact
  gradient of the confined energy** along the mode vectors, the predicted side is the Gram-matrix
  spectrum plus `2*alpha`. That version can fail, and it checks two things at once — that the Hessian I
  predict from is the Hessian of the energy actually integrated, and D14's lift.

- 2026-09-20T09:35Z — ⭐⭐ **A methodological error caught BEFORE the scored run, and it would have
  manufactured a failure of the law.** My first rig measured the half-life as the **first crossing** of
  `|x_0|/2`. That is the wrong instrument for the CRR in the underdamped band. V2 Fig. 1 plots *"Envelope
  half-life (circles) and raw first-crossing time (squares)"* and says they *"diverge after the crossover
  exceptional point"*; V5 §D says the same (*"the retention metric bifurcates"*). `n_half =
  ln2/(-ln|lambda|)` predicts the **ENVELOPE**. Underdamped, the mode oscillates inside a decaying
  envelope, so `|x|` first reaches half because of the PHASE — at a time set by the oscillation period —
  while `|lambda| = sqrt(1-gamma)` sets the envelope. Comparing first-crossing to the CRR there would have
  compared two different quantities and reported the law as broken when it was my instrument that was wrong.
  ⭐ **Fixed:** the scored instrument is now the envelope, taken as the **reverse cumulative maximum** of
  `|x|` — monotone non-increasing by construction, no fit and no window length to choose. The
  first-crossing time is recorded on every row beside it, and ⭐ **their divergence at the crossover is
  itself one of the shorts' predictions and is now tested rather than tripped over.** PREREG §3 amended to
  name the instrument. ⛔ No held-out number had been seen; the amendment is pre-registration, not selection.

- 2026-09-20T09:55Z — ⚠ **WHAT THIS TEST CAN AND CANNOT SHOW — stated before the numbers, so it is not a
  post-hoc hedge.** The prediction is the **linearised** map's eigenvalue; the measurement is the **full
  nonlinear** rollout in 2048 dimensions. At a small enough displacement the linearisation is excellent by
  construction, and the comparison then checks my linear algebra rather than any physics. ⛔ **I will not
  present a small-amplitude agreement as though it vindicated the theory.** The genuine empirical content
  of this hub is four things, and I will report them as the content:
  1. that the field store's landscape **has a well-defined curvature spectrum at the read's settling point
     at all** — it did not have to; the settle could have failed to converge, or landed on a saddle (it
     converges on all 5 wells, 0 saddles, `mu^2` in [0.16, 1.00]);
  2. that **one** measured spectrum predicts settle time across **4.5 orders of magnitude of `mu^2` and
     both bands**, including the curvature-INDEPENDENT floor, with no refit;
  3. ⭐ that the rollout at **deployment amplitudes stays inside the linear law's domain** — which is
     NOT guaranteed and is exactly why V5 sweeps "finite write amplitudes". The `amp = 0.5` aux arm is
     where this can fail, and it is reported whether or not it does;
  4. the V-curve's argmin (D16) and its two branch slopes (D17/D18), which are **not** `+-1` and are not
     recoverable from a one-parameter fit.
- 2026-09-20T09:56Z — Control fix: `landscape_removed` zeroes the atoms, so `Cov_w = 0` and the
  non-trivial curvature spectrum is EMPTY — the arm would have silently emitted zero rows and looked
  NOT-RUN. ⛔ A declared control that vanishes is not a control. Handled explicitly: with the atoms zeroed
  the energy is `||q - x*||^2/2` exactly, every direction carries the trivial `mu^2 = 1/mass`, and the arm
  now emits rows on random orthonormal directions flagged `degenerate_spectrum: true`. ⭐ That IS the
  charter §C.2 collapse, stated as a measurement: the store contributes **no curvature structure at all**,
  one number in every direction, and the CRR then holds trivially — which is the control working.

## ⚠ CORRECTION — the timestamps on several bullets above are wrong

- 2026-09-20T06:49:22Z — ⚠ **Bullets above carrying times from "08:40Z" onward are NOT true UTC.** I wrote them from
  the box's LOCAL clock (UTC+2) and in some cases from an estimate, without checking `date -u`. The true
  reference points, both verifiable on disk: **START_UTC.txt = 2026-09-20T06:25:56Z**, and this bullet is
  written at **2026-09-20T06:49:22Z** — so the whole hub to this point is **~23 minutes** of wall clock, not the ~3.5 h
  those bullets imply. ⛔ Per the logging rule I do not rewrite them; this bullet is the correction.
  **The ORDER of the bullets is correct and none of their content changes** — only their clock. Every
  timestamp from here on is taken from `date -u` at the moment of writing.
  Mapping for a reader: "08:30-08:44Z" was in fact ~06:33-06:38Z; "08:50-09:10Z" ~06:39-06:43Z;
  "09:20-09:56Z" ~06:44-06:48Z.

## WIDTH AND SEPARATION, measured as a property of the kernel (MECHANISM — ⛔ not a scored row)

- 2026-09-20T06:50:59Z — ⛔ Reminder of scope before the numbers: the shorts contain **no** Mexican-hat width law and
  **no** interpolation-floor derivation (inventory §1.1), so nothing below is attributed to them; the closed
  form is **mine**, derived from `gen_landscape.store_energy`. And row 2 belongs to a size-matched
  continuous MLP on every seed under two implementations (w6-common §3.7), so ⛔ **no row is contested here.**
- 2026-09-20T06:50:59Z — ⚠ **A bug in my own mechanism script, caught and fixed before it reached the report.** My first
  version pooled EVERY term's `basin_gap` against EVERY well's `well_sigma`, mixing factors that live in
  other wells. It produced gap/sigma = 0.55/0.36/0.61/**4.46**/**2.00** and "predicted SPLIT" on wells 3 and
  4 — a conclusion drawn from a number that meant nothing. Fixed to use only the terms of the factors each
  well actually hosts (`meta["host_factors"]` = [[0], [1,3], [2], [4], [5]] with `term_offsets`). The
  corrected numbers **reproduce the store's own build-time line** (`w0=1.331, w1=0.814, w2=0.767, w4=0.397`).
- 2026-09-20T06:50:59Z — ⭐ **RESULT (mechanism): every well of the field store is a SINGLE basin, and the derived
  bifurcation predicts it on all five.**

  | well | d | well_sigma | median gap/sigma (its own terms) | predicted | measured minima (192 starts) |
  |---|---|---|---|---|---|
  | 0 | 771 | 6.2215 | **1.331** | MERGED | **1** |
  | 1 | 940 | 9.4967 | **0.814** | MERGED | **1** |
  | 2 | 217 | 5.6396 | **0.767** | MERGED | **1** |
  | 3 |  60 | 0.7663 | **0.734** | MERGED | **1** |
  | 4 |  60 | 1.7054 | **0.397** | MERGED | **1** |

  All five sit **below the `Delta/sigma = 2`** bifurcation of `Hess E = I - Cov_w/sigma^2`, and mean shift
  from 192 Dirichlet starts on the atoms' own hull converges to **exactly one** fixed point in every well.
  ⭐ **So no stored value owns a basin.** The store cannot snap to a stored value because there is nothing
  to snap to: it interpolates by construction. **That is the interpolation floor as a property of the
  kernel** — measured, not asserted, and not claimed as a row.
  The width would have to shrink to `sigma_that_would_split` = 3.070 / 0.263 / 1.709 / 0.253 / 0.263 for a
  value to acquire its own basin — i.e. to **0.49x, 0.03x, 0.30x, 0.33x, 0.15x** of the deployed width.
- 2026-09-20T06:50:59Z — ⭐ **CHARTER §C.4 CHECK — the settled state is NOT on its nearest atom, so this is not a lookup
  table, and the number says so.** Distance from the well's single minimum to its nearest atom, in units of
  that well's own width: **1.053, 1.341, 1.074, 0.410, 0.778 sigma** — nowhere near 0. The settled read
  draws effectively on **31.4, 61.5, 61.4, 45.6, 46.8 of 64** atoms (participation ratio `1/sum w^2`), not
  on one. ⛔ §C.4's "distance -> 0 is a lookup table" does not fire.

- 2026-09-20T06:52:21Z — Held-out condition confirmed structurally before launch: `condition=test, seed=0` yields
  **49 n-way corners** (3,528 rows), 60 pair sub-corners, 24 val corners. The 49 meets w6-common §2's
  ">= 49 paired corners" exactly.
- 2026-09-20T06:52:21Z — ⚠ **A limit of my own estimand, stated before the numbers.** The corner is the declared unit
  (w6-common §2) and I use 49 of them, but on THIS hub the corners are **49 addresses into one landscape**,
  not 49 independent objects: the field store, its atoms and its widths are identical across corners, and
  only the settling point moves. ⛔ So the corner-level paired test is **not** 49 independent tests of the
  law, and a near-unanimous corner count should not be read as 49-fold evidence. The independent
  replication across **seeds 0/1/2** — three separately built stores — is what carries the weight, and the
  corner spread is reported so the reader can see how little of it there is.

## ⭐⭐ The instrument floor — measured before the scored run, and it calibrates the bar

- 2026-09-20T06:53:25Z — I asked the question that decides whether B1's bar means anything: **how well can this
  measurement chain recover the CRR on a landscape where the CRR is true BY CONSTRUCTION?** One atom at the
  origin gives `E = ||q||^2/2` exactly, so `hess_eig = 1`, `mu^2 = 1/mass`, and the one-step map IS the
  shorts' Eq. (3) matrix. Any disagreement is then **the instrument, not the physics.** 17 cells across
  both bands:
  * median `|log10(n_env/n_pred)|` = **0.0349** · p90 = **0.0749** · max = **0.1601**
  * median over cells near the crossover (`0.3 <= h/h* <= 3`) = **0.0540** (6 cells)
  ⭐ **So B1's bar of 0.10 is about 3x the instrument's own floor** — tight enough to mean something,
  loose enough to be measuring the law rather than my sampling. Fixed in the PREREG on that basis, before
  any held-out number. The leg now recomputes and records this on every run
  (`instrument_floor.json`), so the floor travels with the number.
- 2026-09-20T06:53:25Z — ⭐ **The worst cell is exactly where the theory says it must be.** Max error 0.1601 occurs at
  `h/h* = 1.42` — **at the exceptional point**, where V2 App. C (D10) says the 2x2 matrix is **defective**,
  a single Jordan block with secular transients `~ n lambda^n`, so the envelope is not a pure exponential
  and `ln2/(-ln|lambda|)` under-predicts. ⭐ **The shorts predict my instrument's worst case.** That is a
  small but real confirmation of the band structure, arrived at accidentally and worth stating.
- 2026-09-20T06:53:25Z — ⭐ **V2 Fig. 1's two-instrument divergence, reproduced on the exact quadratic.** Deep
  underdamped (`h/h* = 61.7`) the first crossing is **7.03** steps while the envelope is **23.86** — a
  factor 3.4 — and deeper still (`h/h* = 630`) the first crossing falls below the `n < 3` censoring rule
  entirely. At the exceptional point (`h/h* = 1.42`) the two instruments **coincide exactly** (8.98 and
  8.98), and on the overdamped branch they coincide at every cell. ⛔ This is exactly why reading
  first-crossing against the CRR would have manufactured a failure, and it is now a measurement rather
  than a trap I fell into.
- 2026-09-20T06:53:25Z — Systematic bias worth recording: the underdamped envelope ratios sit slightly **below** 1
  (0.868-0.969). Reverse-cummax on a discretely sampled oscillation cannot see the true peak between
  samples, so the envelope decays marginally early. Small, one-directional, and inside the floor above.

## A test suite against the papers' own numbers — and it caught two of my errors

- 2026-09-20T06:57:34Z — Wrote `tests/test_w6_potentials.py`: 20 tests, every one pinned to a number a paper
  **reports** or an identity a paper **states**. They exist so the closed forms cannot drift silently.
  Three failed on the first run; all three were mine, and two were real.
- 2026-09-20T06:57:34Z — ⚠ **REAL BUG, and it corrects my own 06:53Z-equivalent bullet.** `crr_matrix`'s `(0,1)` entry
  was `eta*(1 - h^2/4)`. It is **`eta`**, exactly. Reading the kick-drift-kick off directly:
  `p_half = p - (eta/2) mu^2 q`; `q' = (1 - h^2/2) q + eta p`; `p' = (1 - h^2/2) p - eta mu^2 (1 - h^2/4) q`.
  With my wrong entry `det A = (1-gamma)(1 - h^4/4 + h^6/16)`, i.e. **conformal symplecticity failed at
  order h^4** — 0.9499999879772388 instead of 0.95. With `eta` it is exact: **0.9500000000000001,
  0.7999999999999999, 0.4999999999999998** at gamma = 0.05/0.2/0.5, and `tr A` exact to 1e-14.
  ⚠ **This also corrects what I wrote earlier:** I reported `max|A_paper - A_code| = 1.0e-07 .. 9.0e-05`
  against the repo integrator and attributed it to "float32 gradient noise". **It was not noise, it was
  this bug** — the 9.031e-05 is exactly `eta*h^2/4` at `mu=1.7, eta=0.05`. The repo's integrator was right
  and my matrix was wrong.
  ⭐ **No prediction is affected.** `lambda_slow`, `n_half_crr`, `band_edges` and `n_half_floor` never
  touched `crr_matrix`: they build the characteristic polynomial from `tr` and `det` directly, and both
  are unchanged by the fix. `crr_matrix` was a display/diagnostic function. Every number quoted so far —
  the 27.02681 floor, 0.9989994995, 692.4538, the band edges, the instrument floor — stands unchanged.
- 2026-09-20T06:57:34Z — ⚠ **Second real fix: the latch returned 2.08e15, not `inf`.** At `mu^2 = 0` the roots are
  exactly `{1, 1-gamma}`, but float64 gives `0.9999999999999998`, and `-ln` of that is a meaningless
  rate. `n_half_crr` now returns `inf` at `mu^2 = 0` and for any modulus within 1e-14 of 1. D12's
  infinite half-life is now exact rather than astronomically large.
- 2026-09-20T06:57:34Z — ⭐⭐ **Third failure was not a bug but a genuine refinement of the law, and it changes what B4
  compares against.** My test expected the V-curve argmin at `gamma_crit = 2 eps mu` and got 0.559 where
  that form predicts 0.8 (at `h = 0.4`). The form is not wrong — **it is the small-h limit**, and the
  inventory quotes the shorts saying exactly that: D16 gives `h*(gamma) = (1 - sqrt(1-gamma)) sqrt(2/(2-gamma))`
  and then *"**Inverting at small h**, gamma* = 2 eps mu (1 + O(eps mu))"*.
  Inverting `h*(gamma) = h` exactly, with `r = sqrt(1-gamma)` and `c = 2 - h^2`: `c r^2 - 4r + c = 0`, so
  **`gamma* = 1 - r^2` with `r = (2 - sqrt(4 - c^2))/c`**. Checks: `h=0.001 -> 0.001998` (vs 2h = 0.002 ✓),
  `h=0.025 -> 0.048777` (vs 0.05), `h=0.1 -> 0.1816` (vs 0.2), **`h=0.4 -> 0.5631` (vs 0.8 — a 30 % gap)**.
  The small-h form recovers exactly as `h -> 0` (`gamma* = 2h - h^2`).
  ⭐ Added as `w6_potentials.gamma_crit_exact`; the V-curve analysis now reports the argmin against
  **both** the small-h form (B4's declared bar) and the exact inversion. ⛔ B4's bar is unchanged — a
  factor 2 still covers the gap — but quoting only `2 eps mu` at large `h` would have been quoting the
  approximation as though it were the law.
- 2026-09-20T06:57:34Z — **20/20 tests pass.** They run in 9.4 s and are part of the branch.

## SELECTION RUN on `val` — 12 corners, seed 0, 20,250 rows. ⛔ NOT a scored number.

- 2026-09-20T06:59:14Z — `/c/chlu/outputs/w6-potentials/val/s0`, field sha `784891df0dcf3b68`, ledger **524,288 B,
  matches_budget=true**, integrator check **1.885e-08 PASSED**, Hessian+D14 finite difference
  **2.018e-11** (B6 bar 1e-9 — passes by 50x).
- 2026-09-20T06:59:14Z — **What val says, bar by bar** (all on the selection set, none of it scored):
  * **B1 = 0.0293** (bar 0.10) — the parameter-free CRR predicts the median corner to a factor **1.070**.
    B1-resolved (cells with `n_pred >= 10`) = 0.0270.
  * **B2: mean paired delta = -0.7835, 12/12 corners better than FIT1** (`n = C/(eps mu)^2`, C fitted with
    sight of the data, C = 0.5505). CRR 0.0293 vs FIT1 0.8127 — the fitted power law is **27x worse**.
    Against FIT2 (a fitted constant, C = 138.7): delta -0.5746, 12/12.
  * **B5 — the sharpest result on val.** D17's overdamped branch slope: measured **+1.1366**, derived
    **+1.1385**, naive **+1**. `mean|meas - derived| = 0.0034` against `mean|meas - naive| = 0.1351` —
    ⭐ **the derived slope is 40x closer than the textbook +1**, and the shorts derive that it must exceed
    +1 at every finite gamma. D18's underdamped branch: measured **-1.0976**, derived **-1.0050**, naive
    **-1**; 0.1287 vs 0.1336 — derived wins, but only just, and I will report that margin as thin.
  * ⭐ **V2 Fig. 1's two-instrument divergence, reproduced on the real field store**: envelope /
    first-crossing = **1.000 overdamped** and **3.407 underdamped**. Had I scored first-crossing, the
    underdamped error would have been **0.5871** (a factor 3.9) instead of the envelope's **0.0443**.
    ⛔ That is the failure I would have reported as the law's, and it would have been mine.
- 2026-09-20T06:59:14Z — ⭐⭐ **Where the law holds, and it is exactly where the theory says.** CRR error against
  distance from the exceptional point:

  | `h/h*` | n | median \|log10\| | p90 |
  |---|---|---|---|
  | < 0.1 | 432 | **0.0008** | 0.0024 |
  | 0.1 - 0.5 | 1248 | **0.0134** | 0.0346 |
  | **0.5 - 2** | 1368 | **0.1059** | 0.2559 |
  | 2 - 10 | 1152 | 0.0607 | 0.2626 |
  | > 10 | 2352 | 0.0267 | 0.0561 |

  ⭐ Deep on either branch the parameter-free prediction is good to **0.1 %** (`h/h* < 0.1`); the error is
  concentrated in the `0.5 - 2` bucket — **at the exceptional point**, where V2 App. C (D10) says the
  2x2 map is **defective**, a single Jordan block whose secular transients `~ n lambda^n` make the
  envelope non-exponential. The same signature appeared in my instrument floor on an exact quadratic.
  ⛔ This is the theory predicting where the theory's own simplest form stops, not a get-out.
- 2026-09-20T06:59:14Z — ⚠ **B4 MISSED on val: argmin/gamma_crit median 0.326** (bar 0.5-2.0). I read the raw V-curves
  rather than accept the number, and the cause is **my grid, not D16**. Example (well 0, mass 0.1, mode 0,
  `h = 0.063`, `gamma* = 0.1185` exact): the 9-point gamma grid steps by ~2x, and because the underdamped
  floor `2ln2/(-ln(1-gamma))` **falls** with gamma, the argmin is biased onto the last underdamped grid
  point below `gamma*`. ⭐ **The PREDICTED curve's own argmin on that grid is 0.10 against a true
  `gamma*` of 0.1185** — so the grid alone costs ~0.8, and the measurement noise at the exceptional point
  (that same `0.5 - 2` bucket) costs the rest.
- 2026-09-20T06:59:14Z — **Fix, made on `val` where mechanism choices belong (charter §A.1).** Each mode now gets, in
  addition to the 9 global gammas, a **5-point geometric bracket around its OWN predicted critical
  damping** `[gamma*/2, 2 gamma*]` (`gamma_crit_exact`). ⛔ **This does not put the answer in by hand**:
  the measured argmin is free to land anywhere on the union of both grids, including on a global point far
  from the bracket, in which case B4 still misses. What it removes is a spacing artefact that had nothing
  to do with the law. Cost ~1.56x on the main grid. The D17/D18 fit windows now also use the exact
  `gamma*` rather than its small-h limit.
- 2026-09-20T06:59:14Z — Two report-side corrections, both made before the scored run: **(i)** B2's "40 of 49 corners"
  bar is defined only on the 49-corner scored set and is now marked `n/a` on any smaller set instead of
  printing a spurious MISS; **(ii)** B3's statistic was ambiguous in my PREREG ("`|d log10 n_meas /
  d log10 mu^2| <= 0.10`" without saying over what). On val the per-gamma slopes give **signed median
  +0.0050** and **max |.| 0.1526**. I read the bar on the **median over gammas** and print the max beside
  it every time, so the stricter reading is always visible; ⛔ neither number is hidden and a reader can
  apply either.
- 2026-09-20T06:59:14Z — **Amplitude: the linearisation does NOT break at deployment amplitudes.** aux at
  `amp = 0.05` (`amp/sigma ~ 0.009`) gives 0.0299; at `amp = 0.5` (`amp/sigma ~ 0.089`, a 10x larger
  write) gives **0.0293** — no degradation. With D14's confinement rolled (`alpha = 1`): 0.0408 and
  0.0401. ⭐ So the rolled confinement arm predicts too, and the third of the four empirical contents I
  declared at 06:48Z-equivalent is confirmed on the selection set.
- 2026-09-20T06:59:14Z — **Controls.** `field` 0.0293 vs `landscape_shuffled` 0.0286 — indistinguishable, **exactly as
  I said in advance they would be**: the CRR is a law about the DYNAMICS and holds of any landscape with a
  well-defined Hessian at a critical point. ⛔ A control is not expected to break it. What the controls
  collapse is the **curvature spectrum**, and that is where they differ: `field` spans
  `hess_eig in [0.1588, 1.0000]`, `landscape_shuffled` only `[0.6407, 1.0000]` — the shuffle compresses
  the spectrum toward the trivial value, i.e. destroys the structure the store contributes.
  ⚠ `landscape_removed` emitted **no rows** on this val run because the degenerate-spectrum fix post-dates
  its launch; the fix is in and must be verified on the next leg before it can be reported.

## ⭐⭐ B4 diagnosed on `val`: the grid was innocent, and the miss is PHYSICS

- 2026-09-20T07:01:16Z — Added the gamma refinement and re-ran 3 corners (`val2`, 59 s). B4 still missed
  (**0.349** against a 0.5-2.0 bar). ⛔ I did not accept the number; I printed a raw refined V-curve.
  Cell: well 0, mass 0.1, mode 0, `h = 0.06300`, `gamma*_exact = 0.1185`.

  | gamma | 0.0020 | 0.0050 | 0.0100 | 0.0200 | 0.0500 | 0.0592 | 0.0838 | 0.1000 | **0.1185** | 0.1676 | 0.2000 | 0.2370 | 0.3500 | 0.5000 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | **predicted** | 692.45 | 276.57 | 137.94 | 68.62 | 27.03 | 22.70 | 15.84 | 13.16 | **10.99** | 27.55 | 35.39 | 44.21 | 72.42 | 115.38 |
  | **measured** | 652.97 | 255.32 | 109.66 | 60.14 | **19.62** | 20.34 | 22.53 | 24.27 | 26.55 | 34.27 | 40.44 | 48.12 | 74.61 | 116.51 |

  ⭐ **The refinement worked: the PREDICTED curve's own minimum now lands exactly on
  `gamma*_exact = 0.1185`.** So the gamma grid resolves the critical point and ⛔ the residual gap is
  **not spacing**.
- 2026-09-20T07:01:16Z — ⭐⭐ **What the gap actually is, and it is the theory's own fine print.** From `gamma = 0.05`
  to `gamma = 0.1185` this cell's `h/h*` walks **2.46 -> 1.04**, i.e. it approaches the **exceptional
  point from above**. There V2 App. C (D10) says the 2x2 map is **defective** — one Jordan block — with
  secular transients `~ n lambda^n`, which decay **slower** than `lambda^n`. So the true half-life is
  **longer** than `ln2/(-ln|lambda|)` exactly there: measured 26.55 against predicted 10.99, a factor
  2.4, in the direction the theory requires. ⭐ **The defect fills in the predicted dip, so the measured
  V-curve minimum slides to lower gamma and B4 misses.**
  ⇒ **The statement that survives is sharper than the bar:** the derived critical point is where the
  *eigenvalues collide*, and the prediction's own V-curve minimum sits on it exactly — but the **rollout's**
  minimum does not, because the very degeneracy that defines the critical point is what invalidates the
  pure-exponential envelope there. ⛔ That is a correction to how the law is *used*, not a refutation of it.
- 2026-09-20T07:01:16Z — ⚠ **Consistent with the shorts' own instrument choice, which I should have noticed earlier.**
  V5 reports its argmin agreeing with `gamma_crit` to ~2 % and its two argmins to 0.35 % — but on the
  **I-J instrument**, *"the exact one-step Jacobian, where `n_half = ln2/(-ln|lambda_ret|)`"*. On I-J the
  "measurement" IS the eigenvalue, so the argmin sits on the crossover by construction. My measurement is
  a **rollout**, V5's I-R1-style instrument, and V5 itself says *"the instrument gap is a level, not a
  rate"*. ⭐ **I am measuring the one quantity where the two instruments must disagree.** B4 stands as
  declared and I expect it to miss on the held-out set; the report now prints the PREDICTED curve's argmin
  beside the measured one so the reader can see the grid is adequate and the gap is physical.
- 2026-09-20T07:01:16Z — Three further fixes, all on `val`, all before the scored run:
  * ⚠ **B3 was being fitted on the refined gammas and gave nonsense (median \|slope\| 0.6177 against
    0.0050 on the unrefined val run).** A refined gamma exists for exactly ONE `(well, mass, mode)`, so a
    "slope against `mu^2`" at that gamma is fitted on a single curvature. ⛔ B3 now fits **global gammas
    only**; the refinement is for locating the V-curve minimum and nothing else. Rows now carry
    `gamma_grid: global|refined` so this cannot recur silently.
  * `landscape_removed`, `landscape_shuffled` and `pca_tensor` were tied to the aux-corner count, so
    `--aux-corners 0` silently ran **no controls at all**. Controls now have their own
    `--control-corners`. ⛔ A control that quietly does not run is worse than no control.
  * `gamma_crit_exact` never reached the rows (my earlier edit's anchor had moved when I rewrote
    `corner_rows`), so the exact-inversion column printed nothing. Now carried per row.
  * The two-instrument divergence is now reported **deep underdamped (`h/h* > 5`)**, where V2 Fig. 1
    places it, rather than over the whole band where the refined cells near the crossover (at which the
    instruments coincide) wash the median back to 1.000.

## SCORED RUN 1 — authorised and launched

- 2026-09-20T07:03:22Z — **Ledger row written to `/c/chlu/SCORED-LEDGER.md` BEFORE the run** (w6-common §2), timestamped
  2026-09-20T07:03:19Z, target `potentials`, run **1/3**, pointing at
  `outputs/w6-potentials/PREREG-potentials-1.md`. The leg's own guard was verified to REFUSE without it:
  running `--condition test` with the PREREG present but no ledger row printed
  `REFUSING: no /c/chlu/SCORED-LEDGER.md row for w6/potentials` and stopped before touching anything.

- 2026-09-20T07:03:54Z — ⚠ **A caveat on the `pca_tensor` arm, written before its numbers exist so it cannot be
  framed favourably later.** The arm puts w5's **winning content** (the per-well leading principal
  subspace, verified here at **524,288 B, matches_budget=true**, variance explained
  0.9735/0.9572/0.9604/1.0/1.0) into **this hub's kernel landscape**, keeping the field store's own width.
  ⛔ **It is NOT w5's read.** `pca_tensor` won rows 3 and 4 through a **span projection**; and as *points*
  in a kernel landscape its block is the pool mean in row 0 plus 63 unit principal **directions**, which
  sit near the origin. The geometry is therefore genuinely different from the item store's. ⇒ The arm is
  reported as **a third landscape on which to ask whether the law still predicts**, ⛔ never as a test of
  the incumbent's read, and never as a comparison of the two contents' quality.

- 2026-09-20T07:06:22Z — **`val2` re-run with every fix, 3 corners, all four arms, 10,060 rows, 183 s.** Confirms the
  rig is sound before the scored run:
  * **B3 now PASSES** on global gammas only: `|slope|` median **0.0282** (was 0.6177 when refined gammas
    polluted the fit), max 0.1526 printed beside it. Floor agreement 0.0365.
  * ⭐⭐ **B4's diagnosis is confirmed mechanically: the PREDICTED curve's own argmin over
    `gamma*_exact` is exactly 1.0000.** The gamma grid resolves the critical point, so the measured
    argmin's shortfall (**0.3489** against the small-h form, **0.3835** against the exact inversion)
    is **the rollout disagreeing with the law at the exceptional point**, not spacing. ⛔ This is the
    outcome I pre-registered in Amendment A1.4 before the held-out run.
  * **B5 DERIVED WINS on both branches**: overdamped measured **+1.1367** vs derived **+1.1385** vs naive
    **+1** — `mean|meas-derived| = 0.0041` against `0.1373`, **33x closer than the textbook slope**.
  * ⭐ **Deep underdamped (`h/h* > 5`), the two instruments differ by a factor 9.232**, and the
    first-crossing error there is **1.0088** — a factor **10** — against the envelope's **0.0291**.
    ⛔ Naming the instrument in advance was worth an order of magnitude.
  * **All four arms now emit rows**, and the spectrum column is where they separate, exactly as stated in
    advance: `field` `hess_eig in [0.1588, 1.0000]`; `landscape_shuffled` **[0.6407, 1.0000]**;
    `pca_tensor` **[0.9492, 0.9999]**; `landscape_removed` **[1.0000, 1.0000]** with
    `degenerate_spectrum: true` — ⭐ **no non-trivial curvature at all, which IS the charter §C.2
    collapse, now a measurement rather than an empty table row.** The CRR error is indistinguishable
    across all four (0.0428-0.0435), which is the point: the law is about the dynamics, not the content.
  * Fixed on the way: the report died on a stray non-ASCII character under the console's cp1252 codec
    after most of it had already printed. stdout is now forced to UTF-8 so a number can never be lost to
    an encoding.

- 2026-09-20T07:06:52Z — ⚠ **The scored-run guard FIRED on launch, and it was right to.** My ledger row names the hub
  `w6 hub/potentials`; the leg's guard grepped for the literal `w6/potentials`, which does not appear
  (the row's path `outputs/w6-potentials/...` is a different string). Seed 0 printed
  `REFUSING: no /c/chlu/SCORED-LEDGER.md row for w6/potentials` and `set -euo pipefail` stopped the
  whole launcher before any held-out number was computed. ⛔ **No scored run was consumed** — the run
  never started. ⭐ This is the identical failure w5's compose-field launcher recorded and warned about,
  and I inherited its resolution: the guard now matches `w6[ /]hub/potentials|w6/potentials`, so it
  **fails on a MISSING row and never on a stylistic one**. Relaunching.
- 2026-09-20T07:07:48Z — ⚠⚠ **CORRECTION to my own bullet two above, and to a check I called a pass.** The guard was not
  refusing over a spelling. `Path("/c/chlu/SCORED-LEDGER.md")` is an **MSYS-style path, and Python on
  Windows reads it as `C:\c\chlu\SCORED-LEDGER.md`**, which does not exist — so `led.exists()` was
  `False`, the text read was empty, and the guard would have refused **every** run whatever the ledger
  said. ⛔ **That also means my earlier "the guard works, it refused without a ledger row" check was a
  FALSE POSITIVE**: it refused for the wrong reason, and I reported it as the guard working. The spelling
  fix was real but was not the cause.
  ⭐ Fixed properly: the ledger path is **derived** from `CHLU_HUB_OUTPUTS` (`C:/chlu/outputs` ->
  `C:\chlu\SCORED-LEDGER.md`), and the refusal message now prints **the path it looked at, whether it
  exists, and how many bytes it read**, so this class of failure is self-diagnosing instead of
  masquerading as a missing row. Verified positively: the leg now prints
  `ledger row found in C:\chlu\SCORED-LEDGER.md` and `scored run authorised`.
  ⛔ Still **no scored run consumed** — both refusals stopped before any held-out number.
- 2026-09-20T07:08:23Z — ⭐ **SCORED RUN 1 LAUNCHED AND AUTHORISED.** `scripts/w6_potentials_scored.sh`, one launcher,
  seeds 0/1/2 **sequentially** on 3 threads. Per seed: the **49 n-way held-out corners** on the main grid
  (`field`), the richer amplitude/confinement grid on 3 corners, and the three controls
  (`landscape_removed`, `landscape_shuffled`, `pca_tensor`) on 4 corners. Log
  `/c/chlu/outputs/w6-potentials/test.log`, outputs `/c/chlu/outputs/w6-potentials/test/s{0,1,2}`.
  Seed 0 printed `ledger row found` -> `scored run authorised` -> `integrator check 1.885e-08 PASSED`.
  ⛔ I do not end this turn while it runs; I poll it to completion inside the turn (w6-common §5.2).

## Charter §C floor checks — how each one maps onto a hub that scores a LAW (written before the results)

- 2026-09-20T07:08:31Z — §C is written for an arm whose claim is a **performance number on a row**. This hub's claim is
  *"does the derived law predict the measurement"*. I map each clause honestly rather than pretending to a
  check I cannot do.
  * **§C.1 (must beat the matched-bytes lookup) — DOES NOT APPLY, and I say so rather than manufacture a
    pass.** There is no "lookup" comparator for "does a closed form predict a rollout": an atom-mean or kNN
    read produces a reconstruction, not a half-life. ⛔ **This hub makes NO claim that the field store beats
    any rival on any row**, so the clause that guards against a decorative store has nothing to guard here.
    The clause's spirit — *is the store doing anything?* — is answered instead by §C.2 and §C.4 below.
  * **§C.2 (landscape-removed / landscape-shuffled must collapse) — SATISFIED, and the collapse is in the
    SPECTRUM, not in the law's error.** ⚠ Stated in advance: the CRR holds of **any** landscape with a
    well-defined Hessian at a critical point, so a control is **not** expected to break it, and it does not
    (all four arms sit at 0.0428-0.0435 on val). What collapses is the store's contribution to the
    curvature: `field` spans `hess_eig in [0.1588, 1.0000]`, `landscape_shuffled` **[0.6407, 1.0000]**,
    and `landscape_removed` **[1.0000, 1.0000]** with **no non-trivial spectrum at all**
    (`Cov_w = 0`, `degenerate_spectrum: true`).
  * **§C.3 (the tie clause) — DOES NOT APPLY.** No tie with a lookup is claimed, because no row is claimed.
  * **§C.4 (settled state on its nearest atom ⇒ a lookup table) — MEASURED AND DOES NOT FIRE.** The single
    minimum of each well sits **1.053 / 1.341 / 1.074 / 0.410 / 0.778** of that well's own width from its
    nearest atom, and the settled read draws effectively on **31.4 / 61.5 / 61.4 / 45.6 / 46.8 of 64**
    atoms. ⛔ Nothing approaches distance -> 0.
- 2026-09-20T07:09:11Z — Seed 0's **held-out-condition** field store built: widths
  `w0=6.184, w1=9.515, w2=5.634, w3=0.767, w4=1.661` and **basin gap / width
  `1.339, 0.812, 0.768, 0.699, 0.404`** — ⭐ all five again **below the `Delta/sigma = 2`** bifurcation,
  on a store fitted from a different pool than the `val` one. The single-basin mechanism finding is not an
  artefact of the selection condition. (Store-build numbers, not a scored measurement.)
- 2026-09-20T07:15:22Z — Seed 0 held-out leg running: store sha `697d60dfa3e4b70c`, **524,288 B**, integrator check
  **1.885e-08 PASSED**, **B6 (Hessian + D14 lift, finite difference) = 3.519e-11** against a 1e-9 bar.
  49 addresses formed; corner 5/49 at 287 s (the first corners carry the aux grid and the three controls).

## ⭐ SCORED RUN 1 — HELD-OUT RESULT, SEED 0 (49 corners, 64,335 rows, 885 s)

- 2026-09-20T07:26:02Z — `/c/chlu/outputs/w6-potentials/test/s0`, condition **test**, field sha `697d60dfa3e4b70c`,
  **524,288 B, matches_budget=true**, integrator **1.885e-08 PASSED**.
  Trap check on the **held-out** store: `well_sigma/nn_median` = **0.805, 0.895, 0.900, 1.759, 1.207** —
  again not 1.0 on any well and on both sides of 1.

  | bar | number | bar | verdict |
  |---|---|---|---|
  | **B1** median corner `\|log10 n_meas/n_pred\|` | **0.0433** (factor **1.105**) | <= 0.10 | ⭐ **PASS** |
  | **B2** mean paired delta vs FITTED rival | **-0.5968**, **49/49** corners better | < 0 and >= 40/49 | ⭐ **PASS** |
  | **B3** underdamped `\|slope\|` median over gammas | **0.0357** | <= 0.10 | **PASS** |
  | **B3** floor `\|log10 n_meas/(2ln2/-ln(1-gamma)))\|` | **0.0378** | <= 0.10 | **PASS** |
  | **B4** V-curve argmin / `gamma_crit` | **0.3362** (exact inversion **0.3782**) | within x2 | **MISS — as pre-registered** |
  | **B5** D17 overdamped `mean\|meas-derived\|` | **0.0038** vs naive **0.1436** | derived beats naive | ⭐ **PASS** |
  | **B5** D18 underdamped `mean\|meas-derived\|` | **0.1368** vs naive **0.1417** | derived beats naive | **PASS (thin)** |
  | **B6** Hessian + D14 lift, finite difference | **3.519e-11** | <= 1e-9 | **PASS** |

- 2026-09-20T07:26:02Z — **B1/B2 in words.** The **parameter-free** CRR predicts the median held-out corner's settle
  half-life to a factor **1.105**, against a one-parameter power law **fitted with sight of the held-out
  measurements** which manages **0.6401** — the derived law is **15x closer**, on **49 of 49 corners**, with
  a fitted constant beaten on 49/49 too. B1-resolved (quantisation < 0.04) is 0.0402.
- 2026-09-20T07:26:02Z — **B4 MISSED at 0.3362, exactly as Amendment A1.4 predicted it would, and for the reason given
  there.** ⭐ **The PREDICTED curve's own argmin over `gamma*_exact` is 1.0000** on the held-out set: the
  gamma grid resolves the critical point, so the shortfall is the **rollout** disagreeing with the law at
  the exceptional point, not spacing. ⛔ I do not get to call this a pass; it is a miss, and the
  pre-written statement stands.
- 2026-09-20T07:26:02Z — **B5 is the sharpest held-out number in the hub.** D17's derived overdamped branch slope
  `1 + gamma/(2-gamma)`: measured **+1.1368**, derived **+1.1385**, textbook **+1**. Mean absolute error
  **0.0038 against 0.1436** — ⭐ **the derived slope is 38x closer than +1**, over 932 V-curves. ⛔ D18's
  underdamped margin is **thin** (0.1368 vs 0.1417) and I report it as thin.
- 2026-09-20T07:26:02Z — ⭐ **Where the law holds, on held-out data:** `h/h* < 0.1` -> **0.0008** (0.2 %);
  `0.1-0.5` -> 0.0264; **`0.5-2` -> 0.1482**; `2-10` -> 0.0454; `>10` -> 0.0262. The error is
  concentrated in the one bucket straddling the **exceptional point**, where V2 App. C says the map is
  defective. Deep on either branch the parameter-free prediction is good to **a fifth of a percent**.
- 2026-09-20T07:26:02Z — ⭐ **The instrument choice was worth an order of magnitude on held-out data**: deep underdamped
  (`h/h* > 5`, 6,712 cells) the envelope and first-crossing half-lives differ by **9.239x**, and the
  first-crossing error is **1.0088** against the envelope's **0.0284**.
- 2026-09-20T07:26:02Z — ⚠ **A DEFECT IN MY OWN B3 STATISTIC, found by refusing to accept a number I could not
  explain.** B3's max `|slope|` printed **59.1158**, which is not a physical quantity. Cause: the
  pre-registered filter required ">= 3 distinct `log10 mu^2`" tested by **exact float equality**, which
  float noise satisfies even when a gamma's underdamped cells form only **two** clusters. At
  `gamma = 0.35` and `gamma = 0.5` there are **exactly 2 distinct curvatures** (98 and 48 cells), because
  the floor there is 3.2 and 2.0 steps and the `n < 3` censoring rule removes almost everything. A slope
  fitted across two near-identical curvatures is not a measurement of `d log n / d log mu^2` at all.
  ⛔ **I did NOT move the bar** (charter §A.3 — I have seen a result against it). The report now prints the
  pre-registered median and max **first**, then a per-gamma table showing n, distinct `mu^2` and span, then
  a clearly labelled correction: restricted to the **7** gammas spanning >= 0.5 decade of `mu^2`,
  `|slope|` median **0.0271**, max **0.1536**. ⭐ **The verdict is PASS either way** (0.0357 and 0.0271 both
  <= 0.10), so nothing is at stake in the correction — which is precisely why it can be reported plainly.

## ⭐⭐ SCORED RUN 1 — ALL THREE SEEDS. Held-out, 49 corners each, 193,000 rows total.

- 2026-09-20T08:06:53Z — Seeds 0/1/2 finished at 07:25 / 07:42 / 08:02Z; **885 / 1010 / 1097 s** per leg. Three
  **separately built** stores: sha `697d60dfa3e4b70c`, `608fc3da332b972d`, `fca1c4a89befc12e`, each
  **524,288 B with matches_budget=true**, each integrator-checked at **1.885e-08**. Full table in
  `/c/chlu/outputs/w6-potentials/RESULTS-LEDGER.md`; full per-seed detail in `REPORT.txt`.

  | bar | seed 0 | seed 1 | seed 2 | verdict |
  |---|---|---|---|---|
  | **B1** median corner `\|log10\|` (bar <= 0.10) | **0.0433** | **0.0435** | **0.0431** | ⭐ **PASS x3** |
  | **B2** mean paired delta vs FITTED rival (< 0) | **-0.5968** | **-0.5796** | **-0.5982** | ⭐ **PASS x3** |
  | **B2** corners better (bar >= 40/49) | **49/49** | **49/49** | **49/49** | ⭐ **PASS x3** |
  | **B3** `\|slope\|` median over gammas (<= 0.10) | 0.0357 | 0.0282 | 0.0343 | **PASS x3** |
  | **B3** floor agreement (<= 0.10) | 0.0378 | 0.0378 | 0.0348 | **PASS x3** |
  | **B4** argmin / `gamma_crit` (within x2) | **0.3362** | **0.3301** | **0.3520** | ⛔ **MISS x3 — as pre-registered** |
  | **B5** D17 `mean\|meas-derived\|` vs naive | **0.0038** vs 0.1436 | **0.0042** vs 0.1430 | **0.0040** vs 0.1436 | ⭐ **PASS x3** |
  | **B5** D18 `mean\|meas-derived\|` vs naive | 0.1368 vs 0.1417 | 0.1277 vs 0.1324 | 0.1290 vs 0.1333 | **PASS x3 (thin)** |
  | **B6** Hessian + D14, finite difference (<= 1e-9) | 3.519e-11 | 1.202e-10 | 2.699e-11 | **PASS x3** |

- 2026-09-20T08:06:53Z — ⭐ **THE RESULT, in one sentence: the shorts' Curvature-Retention Relationship predicts the
  field store's settle half-life, parameter-free, to a factor 1.10 at the median held-out corner, and beats
  a one-parameter rival FITTED WITH SIGHT OF THE HELD-OUT DATA on 49 of 49 corners on every one of three
  seeds.** The fitted power law's own error is 0.64 in `log10` against the derived law's 0.043 — **15x**.
  ⚠ **Seed spread on B1 is 0.0431-0.0435**, i.e. essentially none: the law's accuracy here is a property
  of the object and the instrument, not of the seed.
- 2026-09-20T08:06:53Z — ⭐ **D17 is the sharpest number in the hub.** The derived overdamped branch slope
  `1 + gamma/(2-gamma)` = **+1.1385**; measured **+1.1368 / +1.1367 / +1.1368**. Mean absolute error
  **0.0038 / 0.0042 / 0.0040** against the textbook `+1`'s **0.1436 / 0.1430 / 0.1436** — ⭐ **34-38x
  closer, on all three seeds**, over ~950 V-curves each. The shorts derive that this slope must *exceed*
  +1 at every finite gamma, and it does. ⛔ **D18's margin is thin on every seed** (0.1368 vs 0.1417;
  0.1277 vs 0.1324; 0.1290 vs 0.1333) and I report it as thin, not as a win.
- 2026-09-20T08:06:53Z — ⛔ **B4 MISSED on all three seeds (0.3362 / 0.3301 / 0.3520), exactly as Amendment A1.4
  predicted before the run.** On every seed **the PREDICTED curve's own argmin over `gamma*_exact` is
  1.0000**, so the grid resolves the critical point and the shortfall is the **rollout**, not spacing.
  The pre-written statement stands: *the derived critical point is where the eigenvalues collide and the
  prediction's own V-curve minimum sits on it exactly, but a rollout's minimum does not, because the
  degeneracy that defines the critical point is what invalidates the pure-exponential envelope there.*
- 2026-09-20T08:06:53Z — ⭐ **The instrument choice was worth an order of magnitude, on held-out data, on every seed.**
  Deep underdamped (`h/h* > 5`, ~6,800 cells/seed) the envelope and first-crossing half-lives differ by
  **9.24x**; the first-crossing error is **1.0088-1.0180** against the envelope's **0.0284-0.0296**.
- 2026-09-20T08:06:53Z — **Controls (4 corners/seed, seed 1 shown).** CRR error is indistinguishable across arms —
  `field` 0.0435, `landscape_removed` 0.0435, `landscape_shuffled` 0.0428, `pca_tensor` 0.0434 —
  ⭐ **exactly as stated in advance**: the CRR is a law about the dynamics, so a control cannot break it.
  ⭐ **Charter §C.2 is satisfied in the SPECTRUM**, where they separate cleanly:
  `field` `hess_eig in [0.2727, 1.0000]` · `landscape_shuffled` **[0.7446, 1.0000]** ·
  `pca_tensor` **[0.9797, 0.9999]** · `landscape_removed` **[1.0000, 1.0000]**, `degenerate_spectrum:
  true` — **no non-trivial curvature at all.**
- 2026-09-20T08:06:53Z — **Amplitude: the linearisation does not break at deployment amplitudes.** `amp = 0.05`
  (`amp/sigma ~ 0.009`) -> 0.0218; `amp = 0.5` (`amp/sigma ~ 0.086`, a **10x** larger write) -> 0.0228.
  With D14's confinement rolled (`alpha = 1`): 0.0488 and 0.0455 — still inside the bar.

## Width and separation on the HELD-OUT store — and the two-atom rule misses one well

- 2026-09-20T08:07:01Z — `kernel_mechanism_test_s0.json`, 256 mean-shift starts per well on seed 0's **held-out** store:

  | well | d | well_sigma | median gap/sigma | predicted | measured minima | basin sizes |
  |---|---|---|---|---|---|---|
  | 0 | 777 | 6.1844 | 1.3393 | MERGED | **2** | 225 / **31** |
  | 1 | 939 | 9.5152 | 0.8119 | MERGED | 1 | 256 |
  | 2 | 216 | 5.6343 | 0.7678 | MERGED | 1 | 256 |
  | 3 |  58 | 0.7668 | 0.7357 | MERGED | 1 | 256 |
  | 4 |  58 | 1.6614 | 0.4043 | MERGED | 1 | 256 |

  ⚠ **The two-atom rule MISSES on well 0 and I report it as a miss.** `gap/sigma = 1.339 < 2` predicts one
  merged minimum; mean shift finds **two**, the second holding **31 of 256** starts. ⛔ I do not round this
  to "one basin". The rule `1 - Delta^2/(4 sigma^2)` is exact for **two equal-weight atoms**; a 64-atom
  well with a weight profile can carry a second, shallower minimum even when the median basin gap is below
  `2 sigma`, and well 0 is the well where it happens (it is also the well whose settled read draws on the
  fewest atoms, **30.9 of 64**, i.e. the least averaged). ⭐ **The correct general criterion is the
  spectral one** — `lambda_max(Cov_w) < sigma^2`, which is what `curvature_spectrum` actually computes and
  what the scored prediction uses; the pairwise `Delta/sigma = 2` form is its two-atom special case and
  should be quoted as such.
  ⭐ **The substantive statement survives intact on 4 of 5 wells and in every well's read:** no stored value
  owns a basin of its own, so the store interpolates rather than snapping. **Charter §C.4 does not fire on
  any well** — the minimum sits **1.014 / 1.224 / 0.938 / 0.326 / 0.736** of the well's own width from its
  nearest atom, and the settled read draws effectively on **30.9 / 59.5 / 59.5 / 44.8 / 47.4 of 64** atoms.
- 2026-09-20T08:07:01Z — ⚠ **A second path bug, same family as the ledger one, found by the replication refusing to
  run.** The clean checkout imported `chlu` in its own assertion and then died with
  `ModuleNotFoundError: No module named 'chlu'` inside the leg. `PYTHONPATH` carried the MSYS `/c/...`
  form, which Python on Windows cannot use. ⛔ **The assertion passed anyway because `python -c` runs with
  the CWD on `sys.path`, whereas a script in `scripts/` puts the SCRIPT's directory there instead — so my
  check masked exactly the failure it existed to catch.** Fixed: `PYTHONPATH` and `CHLU_ROOT` go through
  `cygpath -w`, `CHLU_HUB_OUTPUTS` is written in Windows form, and the assertion now runs **from a file in
  `scripts/`** so it executes the same way the leg does.

## ⭐ REPLICATION (charter §A.5) — bit-exact from a clean checkout

- 2026-09-20T08:21:15Z — `scripts/w6_potentials_replicate.sh` cloned branch `w6/potentials` **from the bundle** into a
  fresh directory sharing nothing with the worktree but the read-only cache and the PREREG, asserted from a
  file in `scripts/` that **`chlu` resolves inside the clean checkout**, and re-ran seed 0's 49 held-out
  corners (49,015 rows, 771 s). Clean checkout HEAD `50f6d1f`, tree clean, 0 modified files.

  | quantity | primary leg | replication | |
  |---|---|---|---|
  | corners | 49 | 49 | **EXACT** |
  | **B1** | 0.04325618729174789 | 0.04325618729174789 | ⭐ **EXACT** |
  | **B2** mean delta | -0.5968308139918693 | -0.5968308139918693 | ⭐ **EXACT** |
  | corners better | 49 | 49 | **EXACT** |
  | field store sha | `697d60dfa3e4b70c` | `697d60dfa3e4b70c` | **EXACT** |
  | store bytes | 524,288 | 524,288 | **EXACT** |
  | fitted rival C | 0.20069888425001453 | 0.20069888425001453 | **EXACT** |
  | integrator max\|diff\| | 1.884681927188936e-08 | 1.884681927188936e-08 | **EXACT** |
  | **B6** | 3.519262659068545e-11 | 3.519262659068545e-11 | **EXACT** |

  ⭐ **Every quantity matches to the last printed digit.** The result is replicated, not merely repeated.

---

# FINAL REPORT — w6 hub/potentials

*Written 2026-09-20T08:22:23Z. Started 2026-09-20T06:25:56Z. The last section of this LOG, per charter §E.3.*

## 1. What was asked, and what was answerable

**Target:** take the derivations out of the three shorts and test them on the field store — (1) does
Mexican-hat **width/separation** set the interpolation floor and the refusal gap, and (2) do the derived
**friction and mass** settle-time laws match measurement?

⛔ **LAW 1 DOES NOT EXIST IN THE SHORTS.** "interpolation" occurs **zero** times in the three papers;
"Mexican hat" occurs **once**, in V2 App. M.4, as a float64 round-off testbed with no width and no
separation parameter; "centre-surround", "DoG", "RBF", "lateral inhibition" occur zero times; "kernel"
appears only as V1's *certified Markov kernel*, a different object. Established twice independently — by my
own exhaustive search of all three PDFs and by the inventory spoke (§1.1: *"ABSENT. Do not pre-register
against it."*). ⛔ The two nearest substitutes are forbidden by the inventory §3 and were not used: **A2**
(`mu^2 ∝ A/l^2`, an undefined table cell — *"the closest thing in the corpus to a 'width' law and it must
not be treated as one"*) and **A5** (the measured separation constant 1.540000).
⭐ **A declared absence is a result, and it is this hub's first one.**

⭐ **LAW 2 IS PRESENT, CLOSED-FORM, AND WAS TESTED.** All three scored runs went there, plus the §2.3
friction dial the Head redirected me to mid-run.

## 2. The predicted curve, and its validation *before* use

Parameter-free, nothing fitted: `n_half = ln2 / (-ln |lambda_slow|)`, `lambda_slow` the larger-modulus root
of `l^2 - (2-gamma)(1 - h^2/2) l + (1-gamma) = 0`, `h = eps*mu`, `eps = 0.05` inherited verbatim from the
shorts. Validated against the papers' own printed numbers before any data was touched:

| check | paper | this formula |
|---|---|---|
| V2 §4.1 underdamped floor at `gamma = 0.05` | **27.03** | **27.02681** |
| V5 `lambda_ret` at `gamma = 0.002` | **0.998999499** | **0.9989994995** |
| V5 `n_half` at `gamma = 0.002` | **692.5** | **692.4538** |

The repo's own `velocity_verlet_step` **is** the shorts' map, asserted to **1.885e-08** on every leg; the
leg refuses to run otherwise.

## 3. The instrument floor — measured before the bar was fixed

On an exact quadratic, where the CRR is true **by construction**, this chain recovers it to
**median 0.0349 / p90 0.0749 / max 0.1601**. The worst cell sits at `h/h* = 1.42` — **the exceptional
point**, where V2 App. C says the map is defective. **B1's 0.10 bar is about 3x that floor.**

## 4. THE RESULT — scored run 1/3, three seeds, 49 held-out corners each

⭐⭐ **The shorts' Curvature-Retention Relationship predicts the field store's settle half-life,
parameter-free, to a factor 1.10 at the median held-out corner — and beats a one-parameter rival FITTED
WITH SIGHT OF THE HELD-OUT MEASUREMENTS on 49 of 49 corners, on every one of three seeds.**

Full table: `RESULTS-LEDGER.md`. **B1 0.0433/0.0435/0.0431 · B2 49/49 x3 · B3 PASS x3 · B5 PASS x3 ·
B6 PASS x3 · B4 MISS x3.** Seed spread on B1 is **0.0004** — none.

⭐ **The sharpest number: D17.** The derived overdamped branch slope `1 + gamma/(2-gamma) = +1.1385`;
measured **+1.1368 / +1.1367 / +1.1368**. Mean absolute error **0.0038 / 0.0042 / 0.0040** against the
textbook `+1` at **0.1436 / 0.1430 / 0.1436** — **34-38x closer, on all three seeds**. The shorts derive
that this slope must *exceed* +1 at every finite gamma. It does.

⛔ **D18's margin is thin on every seed** (0.1368 vs 0.1417; 0.1277 vs 0.1324; 0.1290 vs 0.1333) and is
reported as thin, not as a win.

## 5. Where the law holds, and where it stops (held-out, seed 0)

| `h/h*` | n | median abs log10 |
|---|---|---|
| < 0.1 | 1,668 | **0.0008** |
| 0.1 - 0.5 | 8,177 | 0.0264 |
| **0.5 - 2** | 14,452 | **0.1482** |
| 2 - 10 | 7,554 | 0.0454 |
| > 10 | 9,642 | 0.0262 |

⭐ Deep on either branch the parameter-free prediction is good to **0.2 %**. **All the error is in the one
bucket straddling the exceptional point**, where V2 App. C (D10) says the map is defective with secular
transients `~ n lambda^n`. ⛔ **The theory predicts where its own simplest form stops.**

**B4's miss is the same fact, measured a second way.** The argmin ratio is **0.3362/0.3301/0.3520** against
a factor-2 bar, while **the PREDICTED curve's own argmin over `gamma*_exact` is 1.0000 on every seed** — so
the grid resolves the critical point and the shortfall is the **rollout**. Pre-written statement, unchanged:
*the derived critical point is where the eigenvalues collide and the prediction's own V-curve minimum sits
on it exactly, but a rollout's minimum does not, because the degeneracy that defines the critical point is
what invalidates the pure-exponential envelope there.* The shorts measure this on **I-J, the exact one-step
Jacobian**, where the measurement *is* the eigenvalue and the argmin lies on the crossover by construction;
V5 itself says *"the instrument gap is a level, not a rate"*.

⭐ **Naming the instrument in advance was worth an order of magnitude.** Deep underdamped (`h/h* > 5`,
about 6,800 cells per seed) the envelope and first-crossing half-lives differ by **9.24x**; first-crossing
error is **1.0088-1.0180** against the envelope's **0.0284-0.0296**.

## 6. Mechanism (⛔ NOT scored, NOT a row)

Width and separation, on the held-out store: all five wells sit **below** the `Delta/sigma = 2` bifurcation
of `Hess E = I - Cov_w/sigma^2` (1.339/0.812/0.768/0.736/0.404), and mean shift from 256 starts finds
**one** minimum in wells 1-4. ⚠ **Well 0 has two** (225/31 starts) — **the two-atom rule misses there and I
report it as a miss**; the correct general criterion is the spectral one, `lambda_max(Cov_w) < sigma^2`,
which is what the scored prediction actually uses. ⭐ The substantive statement survives: **no stored value
owns a basin, so the store interpolates rather than snapping.**

⚠ **The `knn_density` trap, checked in advance and cleared on the held-out store:**
`well_sigma/nn_median` = **0.805 / 0.895 / 0.900 / 1.759 / 1.207** — not 1.0 on any well, on both sides
of 1. The width is a within-value pool spread, not an item spacing.

## 7. Charter §C, one line each

- **§C.1** — ⛔ **DOES NOT APPLY and is not given a manufactured pass.** There is no lookup comparator for
  "does a closed form predict a rollout", and **this hub claims no row against any rival**.
- **§C.2** — **SATISFIED in the SPECTRUM**: `field` [0.2727, 1.0000] · `shuffled` [0.7446, 1.0000] ·
  `pca_tensor` [0.9797, 0.9999] · `removed` **[1.0000, 1.0000], no non-trivial curvature at all**. The
  CRR error is identical across arms (0.0428-0.0435) **because the law is about the dynamics** — stated in
  advance, not after.
- **§C.3** — does not apply; no tie is claimed.
- **§C.4** — **MEASURED, does not fire**: the minimum sits 1.014/1.224/0.938/0.326/0.736 sigma from its
  nearest atom and the read draws on 30.9-59.5 of 64 atoms.

## 8. Replication (§A.5) — **bit-exact**

A clean checkout cloned from the bundle, asserting `chlu` resolves inside itself, reproduced seed 0's
**B1, B2, corners-better, store sha, bytes, fitted-rival C, integrator check and B6 to the last printed
digit**.

## 9. What I got wrong, and corrected in this log rather than quietly

1. Claimed V2 Eq. (6) was "missing a factor gamma" — **it is not**; my `pdftotext` dropped every greek
   glyph. My independent derivation was right; my claim about the paper was wrong.
2. `crr_matrix` carried `eta(1-h^2/4)` where the KDK map has `eta`, breaking `det A = 1-gamma` at order
   `h^4`. My own test suite caught it. **No prediction was affected.**
3. Called a `1e-07..9e-05` disagreement "float32 noise" — **it was that bug**.
4. The latch returned `2.08e15` instead of `inf`.
5. Wrote a **tautological** D14 check that compared the same closed form to itself; replaced with a finite
   difference that can fail.
6. `landscape_removed` silently emitted **zero rows**; the controls were tied to the aux-corner count and
   could silently not run at all.
7. Pooled every term's `basin_gap` against every well's `sigma`, producing a spurious "SPLIT" verdict.
8. Fitted B3 on refined gammas (0.6177 vs 0.0282) and, on held-out data, found my three-distinct-`mu^2`
   filter used **exact float equality**, admitting 2-cluster fits and a max abs slope of 59. ⛔ **Bar not
   moved**; both readings printed.
9. **Two MSYS-vs-Windows path bugs**: the ledger guard would have refused *every* run, and my replication
   assertion **masked the very failure it existed to catch**.
10. LOG timestamps ran about 2 h ahead of the true clock for one stretch.

## 10. Cost line

**Wall clock 06:25:56Z to 08:21Z, about 1 h 55 m.** Local CPU only, **3 threads**, ⛔ **no GPU, no cloud,
no cluster**. Scored legs **886 + 885 + 1098 s**; replication **772 s**; selection legs about 1,030 s.
193,000 scored rows across three seeds. Outputs 103 MB (test) + 27 MB (replication) + 20 MB (selection).
20 unit tests, 9.4 s.

## 11. Runs 2 and 3

⭐ **RETURNED UNSPENT.** Run 1 answered the question on three seeds with a bit-exact replication, and every
bar that could move has moved. ⛔ A second run would re-measure the same law on the same object and buy
nothing. **`potentials` stands at 1/3 spent, 2 BANKED.**

---

## Close-out verification (w6-common §5, all three process rules)

- 2026-09-20T08:23:04Z — **Rule 1 — no box-wide kill.** ⛔ `taskkill /F /IM python.exe` was never run and no peer process
  was touched. I killed only PIDs I captured at launch (45245/45248 for the superseded val leg, 47168's
  probe), by explicit PID. Two python processes belonging to peer agents are running on the box right now
  and are untouched.
- 2026-09-20T08:23:04Z — **Rule 2 — nothing left running into silence.** Every PID this hub launched
  (45245, 45480, 46973, 47168, 47989, 48051, 48134, 51396, 51509) has exited; checked by PID, none alive.
  Every detached job was polled to completion inside the turn.
- 2026-09-20T08:23:04Z — **Rule 3 — committed and bundled WHOLE.**
  * **Code:** `git status` **clean**; HEAD **`50f6d1f`**; **15 commits** on `w6/potentials` beyond
    `base/iter6 = df02568`; `git diff --name-status` shows **9 files, all `A` (added), 0 modified,
    2,356 insertions**. ⛔ **No frozen-harness file is touched** — `gen_field`, `gen_landscape`,
    `gen_store`, `gen_bridge`, `gen_rowsplit`, `exp_gen_cell`, `exp_gen_store`, `exp_gen_factor_wells`,
    `exp_w5_compose_field` and `integrators` all show an **empty diff**.
  * **Bundle:** `/c/chlu/outputs/w6-potentials/w6-potentials.bundle`, rebuilt at HEAD,
    `git bundle verify` -> *"The bundle records a complete history"*, ref
    `50f6d1f39ee1f80910fa78997f6d773e57e42e72 refs/heads/w6/potentials`.
  * **Outputs verified by DIRECTORY LISTING** (git does not show gitignored outputs):
    `LOG.md`, `PREREG-potentials-1.md` (+ Amendment 1), `RESULTS-LEDGER.md`, `REPORT.txt`,
    `report_summary.json`, `START_UTC.txt`, `kernel_mechanism_{val,test}_s0.json`, the bundle, the run
    logs, and **four complete legs** — `test/s{0,1,2}` and `replication/s0`, each carrying
    `results.json`, `rows.pkl`, `field/`, `integrator_check.json`, `instrument_floor.json`,
    `hessian_fd_check.json`, `sigma_provenance.json`.
- 2026-09-20T08:23:04Z — **Ledger closed:** the outcome row is appended to `/c/chlu/SCORED-LEDGER.md` at
  2026-09-20T08:22:37Z — `potentials` **1/3 SPENT, runs 2 and 3 RETURNED UNSPENT**.
- 2026-09-20T08:23:04Z — ⛔ **A declared NOT-RUN, for the record:** no cluster leg was attempted (CSF3 is unreachable by
  any hub) and **no GPU run was made** — charter §D forbids a GPU for a supporting claim, and every number
  here is CPU-reproducible in under 20 minutes a leg. ⛔ These are NOT-RUNs, not nulls.
