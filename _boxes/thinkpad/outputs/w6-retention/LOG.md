# LOG — w6 hub/retention

Hub: `retention`. **TARGET: lifetimes as deletion-by-decay on the field store.**
Base: `base/iter6 = df025685`. Worktree: `/c/chlu/worktrees/CHLU-w6-retention`. Branch: `w6/retention`.
Scored budget: the **deletion** target, 1 spent in w5, **2 remaining**.

- **2026-09-20T06:26Z** — START_UTC.txt written; output dir was empty, no live START existed.
- **2026-09-20T06:26Z** — Environment verified: `chlu.__file__ = C:\chlu\worktrees\CHLU-w6-retention\chlu\__init__.py` (inside my worktree). jax 0.9.0, numpy 2.4.1, equinox 0.13.4. Worktree clean at `df02568`.
- **2026-09-20T06:27Z** — Read the inventory spoke's LOG (`/c/chlu/outputs/w6-shorts-inventory/LOG.md`). It is **still running** (started 06:25:55Z, last bullet "beginning text extraction"). It has **not yet reported the decay law**. Its finding so far: five PDFs in `/c/chlu/shorts/` are **three distinct papers** (V1, V2, V5); two filename pairs are byte-identical files. I will not block on the spoke — I go to the papers directly for the law, and will reconcile against the spoke's report when it lands.

## The decay law, taken from the paper's own formulae (not fitted)

- **2026-09-20T06:35Z** — No PDF library and no poppler on this box, so I wrote a zlib/content-stream text extractor (`/c/msys64/tmp/claude/pdfx/ext.py`) and pulled the text of all three distinct shorts. **The retention law is in V2, `F_v2MemShort_bento_final2.pdf` — "The Price of a Flat Direction: Transverse Curvature sets Retention in a Trained Recurrent Memory"** (Sec. 3, Sec. 4.1, App. C Eqs. 3-6, App. M). ⭐ **The law is present, closed-form, and has ZERO free parameters.** I transcribe it here before any measurement.

  **Setup (paper Sec. 3).** One dissipative velocity-Verlet step: step `eps`, damping `gamma`; kick-drift-kick of `H = T(p) + V(q)` followed by `p <- (1-gamma) p`. Conformally symplectic: `J^T Omega J = (1-gamma) Omega`, `det J = (1-gamma)^d`. Per normal mode of transverse curvature `omega^2 = k/m`, with leapfrog parameter `h = omega*eps`, the one-step map is the 2x2 block (App. C Eq. 3):
  - `det A = 1 - gamma`  (per-mode conformal symplecticity)
  - `tr A  = (2 - gamma) * (1 - h^2/2)`
  - `lambda_pm = ( tr A +- sqrt(Delta) ) / 2`,  `Delta = (2-gamma)^2 (1 - h^2/2)^2 - 4(1-gamma)`  (Eq. 4)

  **Three bands (paper Sec. 3, derived App. C):**
  1. **The Latch** — `omega^2 = 0`, symmetry-protected. Frozen displacement `q_inf = q_0 + eps*p_0/(gamma*M)`; **half-life infinite** (unit eigenvalue).
  2. **The Overdamped Register** — `0 < h < h*`. Slow root `lambda_+ = 1 - (2-gamma) h^2 / (2 gamma) + O(h^4/gamma^3)`, so (Eq. 6)
     **`n_half = ln2 / (-ln lambda_+)  ~=  2*gamma*ln2 / [ (2-gamma) * (omega*eps)^2 ]`**, i.e. **`n_half ∝ omega^-2`**, slope `-1` in `log n_half` vs `log omega^2`.
  3. **The Underdamped Working Memory** — `h* < h < h_f`. Eigenvalues are a complex pair, `|lambda|^2 = det A = 1-gamma` **regardless of omega**; envelope contracts by `sqrt(1-gamma)` per step, so
     **`n_half = 2*ln2 / (-ln(1-gamma))`** — a **curvature-independent floor**.

  **Boundaries (App. C Eq. 5):** `h* = sqrt( (2/(2-gamma)) * (2 - gamma - 2*sqrt(1-gamma)) ) = gamma/2 + O(gamma^2)` (the exceptional point, where `A` is defective); the map loses stability at `h = 2` exactly, for every `gamma`.

- **2026-09-20T06:36Z** — **Arithmetic check of the law against the paper's own printed number, before I touch any code.** The paper quotes the floor as **27.03 steps at gamma = 0.05** (Sec. 4.1, Fig. 1, Fig. 4). From my transcription: `2*ln2 / (-ln(0.95)) = 1.3862944 / 0.0512933 = 27.0269`. ⭐ **Matches to the printed precision. The transcription is correct and the floor carries no fitted constant.** Likewise the crossover: `h* = gamma/2 = 0.025` at `gamma = 0.05`.
- **2026-09-20T06:36Z** — ⭐ Every constant in the law is supplied by `(eps, gamma, omega)`. **There is no free parameter to fit.** Per the task file's falsifier list, this is a **law, not a fitted curve** — and I pre-register it as such before measuring.

## ⚠ The paper's OWN declared negative on my substrate — recorded before I measure

- **2026-09-20T06:37Z** — The paper declares two limits that land directly on the object this hub was given:
  - **App. F, "Tilt Limitations":** *"The tilt-as-a-dial construction is exclusively verified on designed geometries. As detailed in Appendix K, **explicit breaking on a learned, multi-atom store failed to yield predictable pseudo-Goldstone scaling**."* ⛔ **The field store IS a learned, multi-atom store.**
  - **App. G.1:** a generic (emergent) MLP potential's softest curvature is `omega^2 = 5.1/5.9/5.4 e-2` across 3 seeds vs `2.4e-15` for the designed flat mode — a **13-14 order gap**; the emergent arm "functions not on a continuum, but on discrete washboard minima (1-1.6 bits capacity), where any continuous written coordinate fully relaxes."
- **2026-09-20T06:37Z** — **Consequence I pre-register now:** I will **not** attempt to recover a pseudo-Goldstone continuous register on the field store — the paper says that route failed there, and inventing a fit for it is exactly what the task file forbids. ⭐ What the paper *does* license on a learned store is its own positioning line (App. L.4): *"a physics-structured associative memory where **retention is governed as a designed, per-item property controlled by omega^2**."* I take the law as an **exactly-instantiated per-item decay operator** (each atom carries its own `omega_i^2`, hence its own `lambda_+(omega_i)` and its own `n_half`), which is deletion-by-decay, and I measure **what it costs composition** — the question wave 5 handed me. The law is used as a *generator*, not fitted to data.

## CORRECTION AND EXTENSION: V5 is the paper for THIS hub, and V2 agrees with it

- **2026-09-20T06:50Z** — ⚠ **New bullet, correcting the scope of the bullets above (they are not wrong, they are incomplete).** I went to the other two shorts. **V1** (`F_v1_LCFM...`) is *"Certified Access: Test-Time Compute on a Conservative Physics-Structured AI Memory"* — no decay law (`delet*` 0 hits, `half-life` 0 hits). ⭐⭐ **V5** (`F_v5_PaLM_Subm3.pdf`) is **"Retention You Can Predict, Scope and Delete: a (mu; gamma; T) Law and a Structural Deletion Guarantee for a Physics-Structured Memory Store"** — that is *this hub's paper*, and the bullets above were written from V2 alone.
- **2026-09-20T06:50Z** — ⭐ **V2 and V5 carry the SAME law and my transcription covers both.** V5 App. D writes it in my exact form: **`n_half = ln2 / (-ln |lambda_ret|)`** (instrument I-J, the one-step Jacobian). V5 "Left Branch Exclusions" (App. D): *"for gamma < gamma_crit the coset eigenpair becomes complex with |lambda|^2 = 1-gamma. This forces n_half = ln2/(-(1/2) ln(1-gamma))"* = my floor `2 ln2/(-ln(1-gamma))`. V5 states the **same floor number, 27.03 steps at gamma = 0.05** (App. C, "Designed V-Curve Minimization"), and the **same critical point read the other way round**: V5's `gamma_crit = 2 eps mu` is V2's `h* = gamma/2` rearranged, and V2 App. C says so explicitly (*"read as a condition on gamma at fixed mu, the critical damping gamma ~= 2 eps mu of App. M"*). ⭐ **Two independent papers, one law, and it is the law I transcribed.**
- **2026-09-20T06:51Z** — ⚠ **Symbol correction** (the shorts-inventory spoke flagged the same extraction trap at 06:35Z; restating it here so this log stands alone): the papers' symbol for transverse curvature is the **spectral mass `mu`**, defined `mu^2_k = spec(M^-1/2 grad^2 V M^-1/2)`, so the law is `h = eps*mu` and `n_half ∝ mu^-2`. My module calls it `omega2` = their `mu^2`. **Same quantity, same law, different letter.** ⛔ No formula in my module is quoted from a glyph-dropped extraction: each is re-derived from the map and then asserted against a number the papers print.
- **2026-09-20T06:52Z** — ⭐⭐ **V5 supplies a SECOND derived law that lands exactly on this hub's target — App. G.8, "Decay commutes with deletion":** *"Let the store state be the live key set K with per-item amplitudes {a_i}, and let the layout be the canonical placement L(K) ... a function of K and the fixed geometry alone, and in particular **independent of the amplitudes**. Amplitude decay acts item-wise, a_i -> f(a_i), touching neither K nor L(K); deletion removes one key and recanonicalizes, touching neither the surviving amplitudes nor their decay clocks. The two operations act on disjoint components of the state (K, {a_i}), so **they commute**, and the post-delete store is **byte-identical to the store built from the never-inserted history with the same elapsed decay**. **Recency eviction makes L a function of the insertion history rather than of K, which is exactly why it is excluded.**"* ⭐ That is a byte-level, exact, pre-registerable prediction, on a store whose acceptance property is already byte-level. I test **both halves** and report them separately.
- **2026-09-20T06:52Z** — ⭐ V5 also states **the Lifetime Trilemma** (Sec. 2.3): *"Memory lifetimes are governed by a strict trilemma: exact value fidelity, amplitude-independent address hold, and amplitude-independent read latency. A system may only optimize two."* ⚠ That is the paper's own answer to the question wave 5 handed me — a lifetime is not free — and it is the frame I report the composition curve in.
- **2026-09-20T06:52Z** — ⚠ **V5's OWN scope limits, recorded before I measure, exactly as I recorded V2's:** the deletion guarantee is *"a store-level guarantee only; the frozen encoder and any residue of past writes in a learned landscape are separate channels, and we claim no certified (eps, delta) unlearning"*; the T=0 latch *"requires precise continuous coset geometries that must be designed into the architecture"*; and the exactness holds only *"under explicit conditions (a budget sufficient for the cell count, zero baseline leakage, and priority/attribute-based eviction)"*.

## THE LAW REPRODUCED — `chlu/core/gen_retention.py`, verified against a SIMULATION of the paper's own map

- **2026-09-20T06:55Z** — Built `chlu/core/gen_retention.py`. ⛔ **Nothing in it is fitted.** It builds the paper's one-step block from its kick/drift/damp factors (so `det A = 1-gamma` and `tr A = (2-gamma)(1-h^2/2)` are *consequences*, checked, not typed in), forms the discriminant **cancellation-free** as `Delta = [(1-s)^2 - (2-gamma) h^2/2] * [(2-gamma) u + 2 s]` with `s = sqrt(1-gamma)`, `u = 1-h^2/2` (⚠ the naive `tr^2 - 4 det` loses ~15 digits at the latch and turns an exact `rho = 1` into a spurious `1 + 1.1e-15`, i.e. an exact latch into growth — this bit me on the first run and the fix is in the module's docstring), and states the law as **one line covering all three bands**: `n_half = ln2 / (-ln rho(A))`.
- **2026-09-20T06:55Z** — **`self_check()` PASSES.** Against the papers' own printed numbers:

  | check | measured | paper |
  |---|---|---|
  | `det A - (1-gamma)`, 40 curvatures | `1.11e-16` | `= 1-gamma` (V2 App. C Eq. 3) |
  | `tr A - (2-gamma)(1-h^2/2)` | `2.22e-16` | as Eq. 3 |
  | floor at `gamma = 0.05` | **`27.026815`** | **`27.03`** (V2 Sec. 4.1/Fig. 1/Fig. 4; V5 App. C) |
  | floor's curvature-independence over the underdamped band | `1.6e-13` max dev | "saturates at a curvature-independent floor" |
  | `h*` vs `gamma/2`, rel. err at `gamma = 1e-3 / 0.05 / 0.5` | `5.0e-4 / 2.6e-2 / 3.5e-1` | `h* = gamma/2 + O(gamma^2)` (App. C Eq. 5) |
  | `Delta` at `h = h*` | `1.18e-16` | vanishes — the exceptional point |
  | `1 + tr + det` at `h = 2`, four `gamma` | `<= 3.3e-16` | "loses stability at `h = 2` **exactly, for every gamma**" |
  | overdamped slope, `log n_half` vs `log mu^2` | **`-1.00096`** | predicted `-1`; V2 fitted `-0.985`, V5 fitted `-1.006` |
  | exceptional-point onset exponent | **`0.50006`** | predicted `1/2`; V2 fitted `0.5165` |
  | phase below `h*` | **exactly zero** | "below `h*` the frequency `phi = 0` exactly" |
  | latch (`mu^2 = 0`) | `rho = 1.0` exactly, `n_half = inf` | "an infinite half-life" |
  | latch frozen displacement vs `q_0 + eps p_0/(gamma M)` | `7.8e-16` after 20,000 steps | V2 Sec. 3 |

- **2026-09-20T06:56Z** — ⭐ **The law against the SIMULATED map, not against itself.** `verification_table()` iterates the actual 2x2 map and measures the half-life off the trajectory two ways, as the paper does. On the overdamped branch the simulated **envelope** half-life reproduces the closed form to rel. err **`4e-13` .. `8e-11` across ~7 orders of magnitude of `n_half`** (54,056,578 steps down to 47.9). Past the crossover the simulated envelope **saturates at `27.027 ± 0.03` over a 50x span in `h`** — the floor, measured rather than assumed. ⭐ And the paper's own second-order signature reproduces: first-crossing / envelope is **1.0000-1.0002 (0.02 %) throughout the overdamped branch and diverges to 1.865x .. 0.037x above the exceptional point**; V2 App. G reports "agree to 0.01 %; underdamped, they diverge by up to 3.2x".
- **2026-09-20T06:56Z** — ⚠ The papers' **quoted asymptotic** `n_half ~= 2 gamma ln2 / [(2-gamma)(eps mu)^2]` (V2 Eq. 6) tracks the exact root to `2.5e-7` deep in the overdamped band, is **2.4 % off at `h = 0.30 h*`**, and is wrong by orders past the crossover. ⛔ **This hub's decay operator is the EXACT root, never the asymptotic**; the asymptotic is printed beside it so the truncation stays visible.
- **2026-09-20T06:57Z** — ⭐⭐ **THE CONSEQUENCE THAT DECIDES THIS HUB, derived and asserted BEFORE any store number.** `gen_field` calls an item dead at `LIFETIME_EPS = 1e-6` (its own constant; my module asserts the two agree) and `gap_read`'s `empty_basin` reads exactly that liveness bit. So the refusal horizon of a decaying item is `n_refuse = ln(1e-6)/ln(rho) = n_half * ln(1e-6)/ln(1/2)` = **`19.9316 * n_half`, for EVERY curvature.** Both horizons carry the same `1/(-ln rho)` factor, so **`mu^2` cancels**: `assert_reparameterisation_invariance` shows `lifetime(t * n_half) = 2^-t` to **`4.2e-14`** for curvatures spanning `n_half = 60 / 400 / 5000`. ⛔ **In half-life units there is exactly ONE curve, and no choice of curvature can move the refusal horizon relative to the composition horizon — `mu^2` sets the clock rate and nothing else.** That is precisely why this hub measures the composition cost against decay rather than the refusal alone: the refusal half is settled in closed form, and the only open question is *where along the decay the composition cost has already converged to instant eviction's*.

## WHICH exact statement about decay this hub chose, and why — answering the inventory spoke's ambiguity alarm

- **2026-09-20T07:20Z** — The shorts-inventory spoke raised (06:36Z) that **"the exact-decay law" is ambiguous**: no paper uses that phrase and the corpus holds three non-interchangeable exact statements about decay. ⭐ **I name my choice, and it is TWO of the three, used for two different jobs. Both are pre-registered before any held-out number.**
  1. ⭐⭐ **THE GENERATOR — the retention band structure** (spoke catalogue §2.2; V2 Sec. 3 derived App. C Eqs. (3)-(6); V5 App. D / App. G.3). This is what *produces* the decay: the per-mode spectral radius `rho(A)` of the dissipative velocity-Verlet block, and `n_half = ln2/(-ln rho)` across the latch / overdamped / underdamped bands. ⛔ **I take the EXACT root, never the asymptotic overdamped rate `2 gamma ln2/[(2-gamma)(eps mu)^2]`** — that is the corpus's *third* exact-decay statement and it is an **asymptotic**, 2.4 % off already at `h = 0.30 h*` and wrong by orders past the crossover (measured, `law.json`). ⚠ Calling the asymptotic "the exact-decay law" is precisely the confusion the spoke warned about; I use it only as a printed comparator.
  2. ⭐⭐ **THE PROPERTY UNDER TEST — "decay commutes with deletion"** (spoke catalogue §2.6; V5 Sec. 2.3 derived App. G.8). ⭐ **This is the statement that is simultaneously exact AND about decay in the DELETION sense**, and it is the one that bears on the question wave 5 handed me. Its hypothesis is that the layout `L(K)` is *"a function of K and the fixed geometry alone, and in particular **independent of the amplitudes**"*, so decay and deletion *"act on disjoint components of the state (K, {a_i})"*. ⭐ **"Acts on disjoint components" is exactly the property that would explain w5's gap**: the coupling hole touches a component the read's composition does not depend on (0.0000 cost), while eviction removes members of `K` itself (−2.09/−0.91/−6.74). Decay moves only `{a_i}` until an item dies — at which instant it moves `K`. ⛔ **So App. G.8 predicts that decay is free exactly while it stays in the amplitude component, and costs exactly what eviction costs the moment it crosses into the key component.** That prediction is testable both at byte level (`--stage commute`) and in composition (the scored curve), and it is the hub's central hypothesis.
  3. ⛔ **NOT chosen: the `|lambda_flat| = 1` latch.** It is exact and it is about decay, but it is the statement that decay *does not happen* (`mu^2 = 0`, infinite half-life). It is reproduced in `self_check` as a control — `rho = 1.0` exactly, frozen displacement to `7.8e-16` — and it is what every **non-target** item in this hub sits at, ⛔ but it is not a deletion operator and no claim rests on it.

## DEV — the mechanism runs, and the dev stage's own timing estimate was wrong by 8x

- **2026-09-20T07:22Z** — `--stage dev --seed 0` (⛔ UNSCORED; every read on the arm's own **fit pool**, never a held-out corner). Field store built in 33.1 s; ledger **524,288 B** atoms in a 2,054,866 B image, `matches_budget` true; request-only asserted.
- **2026-09-20T07:22Z** — The decay target: `object_hue = 0.0` (red) holds **6 items** — 5 by provenance, 5 by geometry, union 6 — ⭐ the same 6 items w5's eviction removed, so the two operations are on the same set by construction. Clock: `n_half = 27.026815` steps (**the law's floor**, `band = underdamped`), `rho = 0.974679434481`, **17 horizons**, derived refusal step **539**.
- **2026-09-20T07:23Z** — ⭐⭐ **BYTE PROPERTY, EXACT, no judge involved: the first horizon whose image is BYTE-IDENTICAL to the instantly evicted image is step 539** — which is exactly the derived `n_refuse = ceil(ln(1e-6)/ln(rho)) = 539`, i.e. **19.9432 half-lives**. ⛔ At and past that step, **deletion-by-decay and instant eviction are literally the same 2,054,866-byte object**, so every downstream number they produce is identical by construction, not by measurement. Before it, `assert_no_refit` passes at every horizon and the only segment that moves is `atom_life`.
- **2026-09-20T07:23Z** — ⚠ **CORRECTION to the dev stage's own printed estimate.** It reported "feature 7.95 s + judge 1.43 s per unit => ~494 min/seed" and I nearly re-planned the whole run around it. That was a **cold-start measurement**: re-timed over 6 corners, the feature render is **0.042 s/corner** (not 7.95) and the judge is **1.28 s/corner** (not 1.43). Real cost is **~117 units x ~25 judging generators x 1.32 s ≈ 65 min/seed**, so three seeds run in parallel fit comfortably. ⛔ The dev artifact keeps the wrong number in `timing`; this bullet is the correction and the report quotes this one.
- **2026-09-20T07:24Z** — ⚠ **Invariance, first pass, and a fix.** The store-level reparameterisation check initially matched the two clocks on the *horizon label* `t`, which rounds `t*n_half` independently for each clock; 6/16 horizons came out byte-identical and the worst lifetime gap was `9.0e-3`. That is **integer-step quantisation of the slow clock, not a failure of the law** — the law-level statement is quantisation-free and already holds to `4.2e-14`. Fixed to match on **steps** (`n2 = round(n1 * n_half2/n_half1)`) and to report the residual explicitly rather than hide it; re-run below.

## PRE-REGISTERED, then LAUNCHED

- **2026-09-20T07:35Z** — `PREREG-w6-retention-1.md` written. Seven bars, the margin, the estimand, the horizon grid and **both declared expectations against myself** (BAR 1 expected to PASS = my own dial falsified; BAR 2 expected to land on DOMINATED; BAR 4 expected to FAIL). ⛔ No bar is moved once a result is seen against it.
- **2026-09-20T07:40Z** — Scored-run row appended to `/c/chlu/SCORED-LEDGER.md` **before the run**: `w6 hub/retention | deletion | 2/3`. ⭐ **Run 3/3 of the `deletion` target remains BANKED.**
- **2026-09-20T07:44Z** — ⚠ **First launch of the scored stage died on all three seeds with `RecursionError`.** My runtime wrapper set `R7.build_arms = lambda ...: build_arms(...)` while `build_arms` called `R7.build_arms(...)` — so it called itself. Fixed by binding `_R7_BUILD_ARMS = R7.build_arms` **at import**, before any swap, and by refusing to stack wrappers (`if R7.build_arms is not _R7_BUILD_ARMS: raise`). ⛔ **No number was produced by the broken run and none was lost**; the failure was at arm-construction time, before the first corner. ⭐ The fix also makes the module's "w5's arms, verbatim" claim *mechanical* rather than positional.
- **2026-09-20T07:47Z** — Relaunched, three seeds in parallel, 3 BLAS threads each (PIDs 44761 / 44762 / 44763, captured at launch in `pids.txt`). ⛔ Only my own PIDs will ever be killed.

## ⭐ V5 App. G.8 TESTED AT BYTE LEVEL — half A passes exactly, half B fails exactly as predicted

- **2026-09-20T07:52Z** — **BAR 3 PASSES, EXACTLY, zero tolerance. `evict ∘ decay^n == decay^n ∘ evict`, byte for byte, in all 12 cells**: `n ∈ {1, 4, 16, 64, 539, 555}` × {a **disjoint** 6-item set, the **same** 6-item set}, **0 bytes differing** in a 2,054,866 B image every time. ⭐ **That is the paper's hypothesis confirmed on a learned store**: `gen_field`'s eviction is attribute-based (provenance ∪ geometry), never recency — which is the exclusion V5 App. G.8 names — and decay touches only `atom_life` until an item dies, so the two operations really do act on disjoint components of `(K, {a_i})`.
- **2026-09-20T07:52Z** — ⭐⭐ **And this is the mechanism that explains wave 5's gap.** App. G.8's *"acts on disjoint components"* is exactly the property that separates the coupling hole from eviction: the hole writes 1 byte of `coupling_hole`, a component the composition read never consults, so it costs **0.0000**; eviction removes members of `K` itself, so it costs **−2.09 / −0.91 / −6.74**. ⛔ **Decay lives in the amplitude component `{a_i}` until the instant it crosses into the key component `K`** — which the law puts at `19.9316` half-lives. The scored curve now measures whether "lives in the amplitude component" is the same thing as "is free".
- **2026-09-20T07:53Z** — **BAR 4 FAILS, as I declared in advance it would.** The post-delete store is **NOT** byte-identical to a store built from the never-inserted history: **1,339,083 of 2,054,866 B differ (65.16 %)**, after removing 11,044 of 131,072 pool rows and rebuilding in 43.2 s (rebuilt store's byte ledger `matches_budget: true`). ⛔ **This falsifies the PAPER'S PRECONDITION on this object, not the paper's law** — V5 conditions byte-equality on *canonical* Blelloch–Golovin placement, in which `L(K)` depends on the live key set and the fixed geometry alone.
- **2026-09-20T07:54Z** — ⭐ **The per-segment break-out says something sharper than "it failed", and it is worth more than the bar was.** Of the 1,339,083 differing bytes:
  | segment | bytes | what it is |
  |---|---|---|
  | `coupling_v` | 798,926 | fitted pair correction |
  | `term_mu` | 267,684 | the fitted address book |
  | **`atoms`** | **221,590** | **the greedy span selection = the PLACEMENT** |
  | `coupling_u` | 33,595 | fitted pair correction |
  | `global_mu`, `basin_*`, `well_sigma`, `coupling*`, `term_n` | 16,769 | fitted widths/gaps/counts |
  | `atom_life`/`alive`/`row`/`basin` | 519 | the live-set book-keeping |
  ⭐⭐ **So the placement is only 16.5 % of the obstruction; 83.5 % is that the address book and coupling table are FITTED ON THE POOL.** ⛔ **Canonicalising placement — doing the Blelloch–Golovin thing the paper asks for — would close at most 16.5 % of the gap on this object.** The other 83.5 % needs an interface that is not fitted on the rows being deleted, which is a different and much harder change than the paper's precondition suggests. ⚠ I record this as a **measured statement about the field store**, not as a criticism of V5, whose guarantee is explicitly *"a store-level guarantee only; the frozen encoder and any residue of past writes in a learned landscape are separate channels."* The field store's fitted interface **is** exactly that separate channel, and this measurement prices it.

## PROCESS NOTE — three seeds in parallel was going to OOM a shared box, so I stopped two of my own

- **2026-09-20T08:56Z (local 08:56, UTC 06:56)** — ⚠ The three parallel scored seeds were each growing at roughly **250 MB/min resident** (7.5 GB at 15 min, 10.9 GB at 17 min); aggregate python RSS on the box had reached **33,963 MB** and two other hubs' processes were running beside mine (`w6_potentials_leg.py`, `exp_w6_compose_physics`). Extrapolated to the ~45 min still needed, that lands near the box's 64 GB. ⛔ w5's rule 1 exists because a box-wide kill destroyed two hubs' work; **an OOM would do the same damage by a different route**, so I did not let it ride.
- **2026-09-20T08:57Z** — ⭐ **Stopped exactly two PIDs, both mine, both identified by their full command line first** (`--seed 1` = 27232, `--seed 2` = 32168). ⛔ **No box-wide kill. The other hubs' PIDs (17912, 12548) were never touched.** Seed 0 (PID 30140), the furthest along, was left to finish alone.
- **2026-09-20T08:57Z** — Seeds 1 and 2 will be re-run **sequentially and in bounded chunks** (`--max-seconds 1500`, re-invoked in a loop until `scores.npz` appears). The score stage is resumable by construction — it writes `scores_partial.pkl` at its wall-clock cap and skips completed `(arm, corner)` units on restart — so chunking costs only the ~40 s rebuild per chunk and **resets the process's resident memory each time**. ⛔ No number changes: the unit set, the judge passes and the arms are identical either way. ⚠ Cost: the ~17 min already spent on each of seeds 1 and 2 is discarded, because the partial is written only at the cap or at the end.

- **2026-09-20T09:12Z (local; UTC 07:12)** — ⚠ **Second memory decision, same reasoning, recorded so the cost is visible.** With seed 0 (uncapped) and seed 1 (chunked) running together, resident memory grew at a measured **3.0 MB per CPU-second** in the scoring loop — seed 0 reached **19,448 MB at 5,322 CPU-s** and seed 1 **12,370 MB at 885 s wall**. Extrapolating both to completion put the peak near **50 GB** on a 64 GB box shared with `hub/potentials` and `hub/compose-physics`. ⛔ I did not gamble on the headroom. **Stopped the seed-1 chunk runner (bash PID 46984) and its python child (PID 14284) — both mine, both identified by full command line — and left seed 0 alone to finish.** ⛔ Again: no box-wide kill; the other hubs' PIDs were read and deliberately not touched. ⚠ Cost of the decision: seed 1's 1,100 completed units (885 s) are discarded, and seeds 1 and 2 now run **sequentially** after seed 0 rather than beside it. Expected additional wall clock ≈ 2 x 56 min.
- **2026-09-20T09:12Z** — ⚠ **The growth is in the judge, not in my arms.** The accumulating `done` dict is ~7.3 kB per `(arm, corner)` unit (two `(n, 6)` int64 tuple arrays plus five boolean masks at `n ≈ 72`), i.e. ~31 MB for the whole 4,212-unit run — three orders of magnitude below the observed ~11 MB/unit. The 17 decayed field images are 2,054,866 B each, ~35 MB in total. ⛔ Neither accounts for it; the growth tracks judge passes and is a property of the **frozen instrument** (`Judge` = decoder + external classifier under JAX), which is read-only to hubs. I record it as an instrument cost, not as a bug I may fix, and w5's row-7 module already carries a memory note of the same kind (it reordered its loop corner-major "because the process grew past 4 GB on a box shared with three other agents").
- **2026-09-20T09:14Z** — ⛔ **Deliberately NOT editing `chlu/experiments/exp_w6_retention.py` while a seed is in flight.** The chunked runner re-imports the module on every chunk, so an edit mid-seed would change the scored object underneath a resumed run. The cost line therefore went into a **separate** file, `scripts/w6_retention_costline.py`, which only reads finished artifacts. ⚠ 4,212 units = 36 arms x 117 units (109 corners + 8 constructed probes); ~59 % of units need a judge pass, the rest are served by the generator cache.

## ⭐⭐ SCORED RESULT — SEED 0. Both curves, on one axis. The pre-registered expectation holds.

- **2026-09-20T09:52Z (local; UTC 07:52)** — Seed 0 complete: **4,212 units = 36 arms x 117 units** (109 held-out corners + 8 constructed probes), ~2,480 judge passes, 10,500 CPU-s. `scores.npz` on disk. ⚠ Two code fixes were made after launch and **both are report-side only** — `seen_tuple_mask(grid, train_rows)` had the wrong arity in `build_report`, and the markdown renderer was added; ⛔ neither is reachable from `stage_score`, `build_arms`, `horizon_schedule` or `decay_variants`, so **the scored object did not move**. The bar arithmetic lives in `scripts/w6_retention_bars.py`, which only reads finished artifacts.

### THE TWO CURVES (seed 0, 89 survivor corners, paired, vs the undeleted base)

| step `n` | `t` (half-lives) | lifetime | **refuse (deleted set)** | **composition Δ/corner** | CI95 | novel/corner |
|---:|---:|---:|---:|---:|:---:|---:|
| 0 | 0.000 | 1 | **0.0000** | **0.0000** | [0.000, 0.000] | 24.22 |
| 27 | 0.999 | 0.5003 | **0.0000** | **−0.0337** | [−0.112, +0.034] | 24.19 |
| 54 | 1.998 | 0.2503 | **0.0000** | **−0.0899** | [−0.225, +0.034] | 24.13 |
| 108 | 3.996 | 0.0627 | **0.0000** | **−0.4270** | [−0.686, −0.169] | 23.80 |
| 162 | 5.994 | 0.0157 | **0.0000** | **−0.7640** | [−1.225, −0.326] | 23.46 |
| 216 | 7.992 | 0.00393 | **0.0000** | **−1.4494** | [−2.101, −0.831] | 22.78 |
| 324 | 11.988 | 2.46e−4 | **0.0000** | **−1.9438** | [−2.764, −1.191] | 22.28 |
| 432 | 15.984 | 1.54e−5 | **0.0000** | **−2.0337** | [−2.876, −1.258] | 22.19 |
| 486 | 17.982 | 3.86e−6 | **0.0000** | **−2.0337** | [−2.876, −1.258] | 22.19 |
| 538 | 19.906 | 1.02e−6 | **0.0000** | **−2.0337** | [−2.876, −1.258] | 22.19 |
| **539** | **19.943** | **9.92e−7** | **1.0000** | **−2.0899** | [−2.933, −1.315] | 22.13 |
| 649 | 24.013 | 5.91e−8 | **1.0000** | **−2.0899** | [−2.933, −1.315] | 22.13 |

- **2026-09-20T09:53Z** — ⭐⭐ **THE ANSWER TO THE QUESTION WAVE 5 HANDED ME, on seed 0: a lifetime buys nothing eviction does not, and it buys it late.**
  - **BAR 1 PASSES.** At step 538 — the last horizon before the item dies — decay's composition cost differs from instant eviction's by **+0.0562**, inside the declared margin of 1.0. ⛔ **That is this hub's DIAL DECLARATION FALSIFIED, in the task file's own words: decay is behaviourally indistinguishable from eviction. I pre-registered that I expected this and I report it plainly.**
  - **BAR 2: DOMINATED.** `free window` (largest horizon still inside the margin of ZERO) = **5.994 half-lives**; `t_pay(50 %)` = **7.992**; `t_pay(90 %)` = **11.988**; `t_pay(97 %)` = **15.984**; **`t_refuse` = 19.943 measured = 19.943 derived, to the digit.** ⛔ **The cost is paid strictly before any refusal arrives.** In the **~14 half-lives** between the free window closing and the refusal firing, decay pays **the entire composition cost of eviction and refuses nothing at all.** Instant eviction gives the same cost *and* the refusal immediately, so **on this store decay is strictly dominated by eviction.**
- **2026-09-20T09:54Z** — ⭐ **THE COST DECOMPOSITION, which says exactly WHY the two horizons come apart, and it is a mechanism rather than a curve fit.** Of eviction's **−2.0899**:
  - **−2.0337 (97.31 %) is paid by WEIGHT**, while the items are still **live and still answering**;
  - **−0.0562 (2.69 %) arrives at DEATH**, when the content is zeroed.
  ⛔ **The composition channel reads the atom weight `lifetime * exp(-d^2/2 sigma^2)` CONTINUOUSLY; the refusal channel reads a LIVENESS BIT at `LIFETIME_EPS = 1e-6`.** That is the whole of it: 97 % of the price is charged by a continuous quantity that is already at `1.5e-5` when the discrete quantity has not yet flipped. ⭐ This was the mechanism I stated in the PREREG before measuring, and it is what the numbers show.
- **2026-09-20T09:55Z** — ⭐ **INDEPENDENT REPLICATION OF W5, on my own run and my own instrument:** instant eviction re-measured at **−2.0899** against w5's **−2.0899** (seed 0), and the coupling hole at **0.0000** against w5's **0.0000**. ⛔ Not quoted — re-run.
- **2026-09-20T09:55Z** — **BAR 5 and the declared identities: every one ASSERTED bit-identical on 117/117 units.** `field_decay_n0 == field_base`, `field_decay_n539 == field_evict`, `n554`, `n595`, `n649 == field_evict`, and w5's own `field_hole == field_base` and `field_decay == field_evict`. ⭐ **So "decay at the refusal step IS eviction" is not an interpretation of the numbers — the two are the same 2,054,866-byte object and the scoring loop proved every rendered feature bit-identical.**
- **2026-09-20T09:56Z** — **BAR 6 PASSES: the refusal curve is a pure STEP at the derived horizon.** `refuse_on_deleted_set = 0.0000` at every one of the 13 horizons below step 539 and **1.0000** at 539 and beyond; **survivor-set refusal is 0.0042 at EVERY horizon including 0** — i.e. drift **0.0000**, well inside the 0.01 bar, and the 0.0042 is the store's pre-existing baseline, not something decay caused.
- **2026-09-20T09:57Z** — **BAR 7 (charter §C) PASSES, and the store is not decorative.** Absolute distinct valid novel tuples per corner on the 89 survivors: field/base **24.22**, decay@539 = evict **22.13**, hole **24.22**, table/base **10.48**, table/row_removed **10.48**, **retrain-from-scratch (GOLD) 10.74**, §C.1 lookup_argmax **0.38**, lookup_mean **0.00**, §C.2 landscape_removed **0.00**, landscape_shuffled **0.00**. Paired: **decay@refusal beats retrain-from-scratch by +11.3933** (CI95 [+8.56, +14.27]), beats the row-removed table by **+11.6517**, and beats the matched-bytes lookups by **+21.7528 / +22.1348**. ⛔ §C.2 collapses to zero; §C.1 is cleared by a wide margin, so the landscape is load-bearing.
- **2026-09-20T09:57Z** — ⚠ **Reported against myself, because it is the other half of the same table:** on the *composition* axis the table rivals pay **nothing** for this value deletion (`table/row_removed` Δ **0.0000**, `table+gate8/clear_red` Δ **0.0000**, `table/retrain` Δ **+0.2584**) while all four refuse at **1.0000**. ⛔ **On the deleted VALUE, a table row-removal and an 8-byte bit-clear are strictly cheaper deletions than anything the field store can do** — they lose no composition because they hold no shared atoms. The field store's answer is that it is **11.39 tuples/corner higher in absolute composition to begin with**, and that is where the trade sits. ⭐ This is the same shape as w5's finding and it is not softened here.

## THE COST LINE (seed 0), and the charter §C.4 check

- **2026-09-20T10:05Z (local; UTC 08:05)** — `scripts/w6_retention_costline.py`.
  - ⚠ **CHURN — a COST LINE only, ONE definition, ⛔ claimed nowhere.** Definition fixed in the PREREG: *bytes of `atom_life` written per decay step x steps to the refusal horizon.* Measured: **24 B/step x 539 steps = 12,936 B written**, against instant eviction's single **49,374 B** write, i.e. **0.2620x**. ⭐ **So churn is NOT the argument against decay** — running the lifetime channel all the way to the refusal horizon writes about a quarter of the bytes eviction writes once, because a lifetime is 4 B per item while an item's own byte range is 8,229 B. ⛔ The argument against decay is the **composition curve**, not the write volume, and this line exists so that cannot be confused.
  - **Retrain-from-scratch, the GOLD standard, with its cost:** the prototype table refits in **3.45 s on 120,028 rows** (full-pool fit 3.58 s; row-removal 0.00 s). The **MLP** refits in **50.47 s on 120,028 rows** (full pool 55.68 s, 130,514 params against the 131,072 target, param error −0.0043 %) — and ⛔ **still refuses 0.0000 of the 720 deleted requests.** Its judge read-back of the asked `object_hue` falls from **1.0000 to 0.0750** and "satisfies the full request" from 0.5278 to 0.0194: ⭐ **retraining buys a WRONG ANSWER, not a refusal.** That reproduces w5's BAR-8 finding on my own run.
  - **The BAR-4 rebuild** (a store built from the never-inserted history): **43.23 s**, 11,044 rows removed, 120,028 refitted — and **1,339,083 B still differ**.
  - **Charter §C.4 — ⛔ not a lookup table.** Over 1,024 requests the settled state sits at **relative distance 0.3840** from its nearest atom (min 0.2225, mean absolute 11.696; the address itself is at 0.6100). ⭐ A strict interior point of the atoms' span, so the read is not a nearest-atom lookup in disguise.
- **2026-09-20T10:06Z** — ⭐ **Cost-line summary for the comparison the target file asks for**, all on seed 0: the **coupling hole** deletes for **1 byte, 0 s, 0.0000 composition**; **instant eviction** for **49,374 B, ~0 s, −2.0899**; **deletion-by-decay** for **12,936 B spread over 539 steps, −2.0899** (the same object at the end, byte for byte); the **8-byte gate** for **1 bit, 0 s, 0.0000** but only on a VALUE and **zero bits on a conjunction**; the **table row-removal** for **0.00 s, 0.0000**; **retrain-from-scratch** for **3.45 s (table) / 50.47 s (MLP) on 120,028 rows**, and the MLP still never refuses. **MLP / transformer / LSTM / LACE / PoE-EBM take a DASH on the refusal half, earned by measurement.** ⛔ A dash is never a win for us.

- **2026-09-20T11:00Z (local; UTC 09:00)** — Seeds 1 and 2 complete (chunked, resumed twice each; the resume skipped completed `(arm, corner)` units exactly as designed). All three seeds have `scores.npz`. Analysis, report and cost line re-run across 0/1/2.

---

# FINAL REPORT — w6 `hub/retention`

**TARGET: lifetimes as deletion-by-decay on the field store.**
Hub `w6/retention` · branch `w6/retention` · base `base/iter6 = df025685` · started **2026-09-20T06:25:57Z**.
**Scored budget: the `deletion` target. 1 spent by w5. This hub spent 1 (run 2/3). ⭐ Run 3/3 is RETURNED UNSPENT and remains BANKED.**
Pre-registration: `PREREG-w6-retention-1.md`, written 07:35Z; scored-ledger row appended 07:40Z; **both before any held-out number.**
⛔ `refusal` is 0/3 programme-wide and **no refusal row was scored here** — the refusal numbers below are the refuse-after-deletion half of row 7, which is the `deletion` target's own currency and exactly what w5 `hub/delete` scored.

## 1. The pre-registered runs — all of them

There was **one** scored run, `2/3` of `deletion`, and it is reported in full below. ⛔ No second scored run was needed and none was taken; nothing was selected on the held-out set at any point, because **the horizon grid, the clock and the refusal horizon are all derived from the papers' law in closed form** (§3) and the estimand, target, margin and instrument are **inherited verbatim from w5 `hub/delete`**.

## 2. ⭐⭐ THE ANSWER TO THE QUESTION WAVE 5 HANDED THIS HUB

> *"Decay is eviction spread over time. So the sharp question is: does a lifetime buy anything eviction does not — or does it simply pay the same composition cost more slowly?"*

**It pays the same composition cost, it pays most of it EARLY, and it delivers the refusal LATE. On this store a lifetime is strictly worse than instant eviction.**

**BAR 1 — the task file's own falsifier — PASSES on all three seeds.** At step 538, the last horizon before the items die, decay's composition cost differs from instant eviction's by **+0.0562 / +0.0337 / +0.0112** novel tuples per corner, against a declared margin of **1.0**. ⛔ **This hub's DIAL DECLARATION IS FALSIFIED, in its own words — "decay is behaviourally indistinguishable from eviction" — and I pre-registered that I expected exactly this.**

**BAR 2 — DOMINATED on all three seeds.** The composition cost is paid strictly *before* any refusal arrives:

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| `t_pay(50 %)` | 7.992 | 5.994 | 7.992 |
| `t_pay(90 %)` | 11.988 | 7.992 | 11.988 |
| `t_pay(97 %)` | 15.984 | 7.992 | 11.988 |
| **`t_refuse` measured** | **19.943** | **19.943** | **19.943** |
| `t_refuse` **derived** from the law | 19.943 | 19.943 | 19.943 |

⛔ **In the ~8–14 half-lives between paying 97 % of the cost and firing the refusal, deletion-by-decay holds a store that has already lost everything eviction would have cost it, and still answers every request about the value it is deleting.** Instant eviction gives the same cost *and* the refusal immediately. **A lifetime buys nothing, and it buys it late.**

## 3. THE DECAY LAW — reproduced exactly, with zero free parameters

Two of the corpus's three non-interchangeable exact statements about decay were used, for two different jobs, and the third was declined; all three choices are logged with reasons at 07:20Z.

- **The generator** — the retention band structure (V2 Sec. 3 der. App. C Eqs. (3)-(6); V5 App. D), stated as **one line covering all three bands**: `n_half = ln2 / (-ln rho(A))`, `rho` the spectral radius of the dissipative velocity-Verlet block.
- **The property under test** — **V5 App. G.8, "decay commutes with deletion"** (§5 below).
- ⛔ **Declined as the operator**: the asymptotic overdamped rate `2 gamma ln2/[(2-gamma)(eps mu)^2]`. Measured here to be **2.4 % off already at `h = 0.30 h*`** and wrong by orders past the crossover; it is printed only as a comparator. ⛔ Also declined: the `|lambda_flat| = 1` latch — it is the statement that decay does *not* happen, reproduced as a control (`rho = 1.0` exactly, frozen displacement `7.8e-16`), and no claim rests on it.

**`gen_retention.self_check()` asserts the transcription against the papers' own printed numbers before anything may use it**, and `verification_table()` checks it against a **simulation of the paper's own map**:

| check | measured here | the papers |
|---|---|---|
| floor at `gamma = 0.05` | **27.026815** | **27.03** (V2 Sec. 4.1/Figs 1,4; V5 App. C) |
| floor's curvature-independence | `1.6e-13` over the underdamped band | "a curvature-independent floor" |
| overdamped slope `log n_half` vs `log mu^2` | **−1.00096** | predicted −1; V2 fitted −0.985, V5 fitted −1.006 |
| exceptional-point onset exponent | **0.50006** | predicted 1/2; V2 fitted 0.5165 |
| `h*` vs `gamma/2` | rel. err `5.0e-4` at `gamma = 1e-3` | `h* = gamma/2 + O(gamma^2)` |
| stability limit | `1 + tr + det = 0` to `3.3e-16` at `h = 2`, four `gamma` | "`h = 2` exactly, for every `gamma`" |
| `det A`, `tr A` | `1.1e-16`, `2.2e-16` | App. C Eq. (3) |
| latch | `rho = 1.0` exact, `n_half = inf`, displacement `7.8e-16` | "an infinite half-life" |
| **simulated envelope vs closed form** | rel. err **`4e-13`–`8e-11` over ~7 orders of magnitude** of `n_half` | — |
| **simulated floor past the crossover** | **27.027 ± 0.03 over a 50x span in `h`** | 27.03 |
| first-crossing / envelope | **1.0000–1.0002 overdamped, 1.865x–0.037x underdamped** | "agree to 0.01 %; diverge by up to 3.2x" |

⭐ **Every constant is supplied by `(eps, gamma, mu^2)`; there is no free parameter to fit.** Per the task file's falsifier list, this is **a law, not a fitted curve**, and it was pre-registered as such.

⭐⭐ **Two consequences were DERIVED before any measurement, and both then held exactly:**
1. With `gen_field.LIFETIME_EPS = 1e-6`, **`n_refuse = 19.9316 * n_half` for every curvature** — measured refusal step **539 = 19.9432 half-lives on all three seeds**, matching the derivation to the digit.
2. **`mu^2` cancels**: in half-life units every lifetime is the one curve `2**(-t)`. Verified to `4.2e-14` in the law, and **byte-exactly on the store** — two clocks a decade apart (`n_half = 27.027` vs `270.268`, one at the floor and one deep in the overdamped band) give **16/16 horizons byte-identical, worst lifetime gap `0.000e+00`**. ⛔ **No curvature can trade refusal against composition; `mu^2` sets the clock rate and nothing else.** That is why nothing had to be selected.

## 4. ⭐⭐ THE TWO CURVES — the measurement this hub existed for

89 held-out survivor corners, paired by corner, `distinct_valid_novel_tuples`, against the undeleted base. Refusal on the 20 held-out deleted corners. Cache generation 2, byte pin 524,288 B (tolerance 0), image 2,054,866 B.

| step `n` | `t` (half-lives) | lifetime | **refuse (deleted)** | **Δ seed 0** | **Δ seed 1** | **Δ seed 2** |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 1 | **0.0000** | +0.0000 | +0.0000 | +0.0000 |
| 7 | 0.259 | 0.836 | **0.0000** | +0.0000 | +0.0112 | +0.0112 |
| 14 | 0.518 | 0.698 | **0.0000** | −0.0112 | −0.0225 | +0.0112 |
| 27 | 0.999 | 0.500 | **0.0000** | −0.0337 | −0.0112 | +0.0112 |
| 54 | 1.998 | 0.250 | **0.0000** | −0.0899 | −0.0899 | −0.1348 |
| 108 | 3.996 | 0.0627 | **0.0000** | −0.4270 | −0.4045 | −0.9663 |
| 162 | 5.994 | 0.0157 | **0.0000** | −0.7640 | −0.7640 | −2.0562 |
| 216 | 7.992 | 3.93e−3 | **0.0000** | −1.4494 | −0.9888 | −4.0449 |
| 324 | 11.988 | 2.46e−4 | **0.0000** | −1.9438 | −0.9663 | **−6.7416** |
| 432 | 15.984 | 1.54e−5 | **0.0000** | −2.0337 | −0.8764 | −6.7416 |
| 486 | 17.982 | 3.86e−6 | **0.0000** | −2.0337 | −0.8764 | −6.7303 |
| 538 | 19.906 | 1.02e−6 | **0.0000** | −2.0337 | −0.8764 | −6.7303 |
| **539** | **19.943** | **9.92e−7** | **1.0000** | **−2.0899** | **−0.9101** | **−6.7416** |
| 649 | 24.013 | 5.91e−8 | **1.0000** | −2.0899 | −0.9101 | −6.7416 |

⭐ **Seed 2 is the sharpest single row in this hub: at `t = 11.988` the composition cost is already `−6.7416`, which is 100.0 % of instant eviction's, and the refusal is still `0.0000`. It then holds full cost and zero refusal for a further 8 half-lives.**

### The cost decomposition — the mechanism, not a curve fit

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| paid by **WEIGHT**, items still live and still answering | −2.0337 | −0.8764 | −6.7303 |
| paid at **DEATH**, content zeroing | −0.0562 | −0.0337 | −0.0112 |
| **fraction paid by weight** | **97.31 %** | **96.30 %** | **99.83 %** |

⛔ **The composition channel reads the atom weight `lifetime * exp(-d^2 / 2 sigma^2)` CONTINUOUSLY; the refusal channel reads a LIVENESS BIT at `LIFETIME_EPS = 1e-6`.** 96–99.8 % of the price is charged by a continuous quantity that is already at `1.5e-5` while the discrete quantity has not yet flipped. ⭐ **That is the whole explanation of the gap, it was stated in the PREREG before measuring, and it is the one thing this hub found that a refusal curve alone could never have shown.**

⚠ **Reported against the neat story:** the curve is not perfectly monotone. On seed 1 it overshoots to **−0.9888 at `t = 7.992` (108.6 % of eviction's own cost)** before settling back to −0.8764; seed 2 shows a `−6.7416 → −6.7303` wobble. Both are inside the bootstrap CI95 and neither changes a bar, but a partially-decayed store can read **slightly worse than a fully evicted one**, and this log says so rather than smoothing it.

⚠ **The "free window" statistic degenerates on seed 1 and I will not quote it as a headline.** Defined as the largest horizon whose |Δ| is inside the margin of 1.0, it reads **5.994 / 24.013 / 3.996** half-lives — but seed 1's is the *last* horizon only because seed 1's **entire** eviction cost (−0.9101) is itself inside the margin. ⛔ The load-bearing statistic is `t_pay`, which is defined relative to each seed's own eviction cost and is well behaved on all three.

## 5. ⭐ V5 App. G.8 TESTED AT BYTE LEVEL — half A passes exactly, half B fails exactly as predicted

**BAR 3 PASSES, zero tolerance, all three seeds.** `evict ∘ decay^n == decay^n ∘ evict`, **0 bytes differing** in a 2,054,866 B image, in **all 12 cells per seed**: `n ∈ {1, 4, 16, 64, 539, 555}` × {a **disjoint** item set, the **same** item set}. ⭐ The paper's hypothesis confirmed on a **learned** store: `gen_field`'s eviction is attribute-based (provenance ∪ geometry), never recency — the exclusion V5 names — and decay touches only `atom_life` until an item dies, so the two operations really do act on disjoint components of `(K, {a_i})`.

⭐⭐ **And that is the property that explains wave 5's gap.** *"Acts on disjoint components"* is exactly what separates the coupling hole from eviction: the hole writes 1 byte of `coupling_hole`, a component the composition read never consults, so it costs **0.0000**; eviction removes members of `K` itself, so it costs **−2.09 / −0.91 / −6.74**. **Decay lives in the amplitude component `{a_i}` until it crosses into the key component `K` — and §4 shows that living in the amplitude component is NOT the same as being free.** The hole is free because the read does not depend on the hole's component *at all*; decay is not free because the read depends on the amplitude component *continuously*. ⛔ **Disjointness buys exactness, not freedom.** That is the sharpest thing this hub can hand back.

**BAR 4 FAILS, as pre-registered, on all three seeds.** The post-delete store is **not** byte-identical to a store built from the never-inserted history: **1,339,083 / 1,338,974 / 1,343,641 B differ — 65.17 % / 65.16 % / 65.39 %** of the image, after removing ~11,000 of 131,072 pool rows and rebuilding in 43.2 / 48.6 / 47.8 s. ⛔ **This falsifies the PAPER'S PRECONDITION on this object, not the paper's law**: V5 conditions byte-equality on *canonical* Blelloch–Golovin placement, in which `L(K)` is a function of the live key set and the fixed geometry alone.

⭐ **The per-segment break-out is worth more than the bar was.** Of seed 0's 1,339,083 differing bytes: `coupling_v` **798,926**, `term_mu` **267,684**, **`atoms` 221,590**, `coupling_u` 33,595, the fitted widths/gaps/counts 16,769, the live-set book-keeping 519.
⛔ **The placement is only 16.5 % of the obstruction; 83.5 % is that the address book and the coupling table are FITTED ON THE POOL.** So doing the Blelloch–Golovin thing the paper asks for would close **at most one sixth** of the gap on this object; the rest needs an interface that is not fitted on the rows being deleted — a different and much harder change than the precondition suggests. ⚠ This is a measured statement about the field store, **not** a criticism of V5, whose guarantee is explicitly *"a store-level guarantee only; the frozen encoder and any residue of past writes in a learned landscape are separate channels."* The field store's fitted interface **is** that separate channel, and this measurement prices it.

## 6. The remaining bars

- **BAR 5 PASSES, exact, all three seeds.** Before death only `atom_life` moves and `assert_no_refit` passes (14/14 fitted segments byte-identical) at every horizon. At and after step 539 the decayed image is **byte-identical to the instantly evicted image** — and the scoring loop's own `identical_to` machinery **asserted every declared identity bit-identical on 117/117 units**: `field_decay_n0 == field_base`, `field_decay_n{539,554,595,649} == field_evict`, plus w5's `field_hole == field_base` and `field_decay == field_evict`. ⭐ **So "decay at the refusal step IS eviction" is not an interpretation of the numbers — the two are the same 2,054,866-byte object and every rendered feature was proved bit-identical.**
- **BAR 6 PASSES.** The refusal curve is a **pure step**: `0.0000` at all 13 horizons below step 539, `1.0000` at 539 and beyond, on all three seeds. Survivor-set refusal is constant at **0.0042 / 0.0000 / 0.0000** across *every* horizon including `t = 0` — drift **0.0000**, against a bar of 0.01, and the 0.0042 is the store's pre-existing baseline, not something decay caused.
- **BAR 7 PASSES (charter §C).** §C.2 `landscape_removed` and `landscape_shuffled` collapse to **0.0000 / 0.0000 / 0.0000** absolute. §C.1 matched-bytes lookups (same atoms, same address book, same 524,288 B) score **0.382/0.191/0.191** (argmax) and **0.000/0.056/0.000** (mean) against the post-deletion store's **22.13/20.08/21.08**, i.e. the store beats its own floor by **+21.75/+19.89/+20.89**. §C.4: the settled state sits at relative distance **0.384 / 0.445 / 0.458** from its nearest atom — a strict interior point of the span, ⛔ **not a lookup table**.
- ⛔ **The central property held: no refit anywhere.** `assert_no_refit` passed on every horizon of every seed; refuse-after-decay needed no refit, so the property the task file said would fail the hub did not fail.

## 7. Against the full comparator set, on one axis

Absolute distinct valid novel tuples per corner, 89 held-out survivor corners (seed 0 / 1 / 2):

| arm | novel/corner | composition Δ | refuse (deleted) | price of the deletion |
|---|---|---|---:|---|
| `field/base` (pre-deletion) | 24.22 / 20.99 / 27.82 | — | 0.0000 | — |
| ⭐ **coupling hole** | **24.22 / 20.99 / 27.82** | **0.0000 / 0.0000 / 0.0000** | **1.0000** | **1 byte, 0 s, no refit** |
| **decay @ refusal step** | 22.13 / 20.08 / 21.08 | **−2.0899 / −0.9101 / −6.7416** | **1.0000** | 12,936–15,092 B over 539 steps |
| **instant eviction** | 22.13 / 20.08 / 21.08 | **−2.0899 / −0.9101 / −6.7416** | **1.0000** | 49,374–57,603 B, one write |
| `table/row_removed` | 10.48 / 8.28 / 14.63 | 0.0000 | 1.0000 | 0.00 s |
| `table/retrain` (**GOLD**) | 10.74 / 8.67 / 14.16 | +0.2584 / +0.3933 / −0.4719 | 1.0000 | **3.45 / 3.81 / 3.72 s, ~120,000 rows** |
| `table+gate8/clear_red` (**8-byte gate**) | — | 0.0000 | 1.0000 | **1 bit** — ⛔ but **zero bits** on a conjunction |
| `field/lookup_argmax` (§C.1) | 0.38 / 0.19 / 0.19 | — | — | floor |
| `field/lookup_mean` (§C.1) | 0.00 / 0.06 / 0.00 | — | — | floor |
| §C.2 removed / shuffled | 0.00 | — | — | control |
| **MLP / mlp_180k / cont_mlp / transformer / LSTM / LACE / PoE-EBM** | **DASH** | **DASH** | **DASH** | see below |

**Paired, decay@refusal minus rival, on the same 89 corners:** vs **retrain-from-scratch (GOLD)** **+11.3933 / +11.4045 / +6.9213**; vs the row-removed table **+11.6517 / +11.7978 / +6.4494**; vs the §C.1 floors **+21.75/+19.89/+20.89** and **+22.13/+20.02/+21.08**.

**The dash is earned by measurement, not asserted.** Retraining an MLP without the value costs **50.47 / 40.49 / 39.87 s on ~120,000 rows** and still **refuses 0.0000** of the 720 deleted requests; its judge read-back of the asked `object_hue` falls from **1.0000 to 0.0750 / 0.0042 / 0.0000**. ⭐ **Retraining buys a WRONG ANSWER, not a refusal.** ⛔ A dash is never a win for us.

⚠ **Reported against myself, because it is the other half of the same table:** on the *composition* axis the table rivals pay **nothing** for this value deletion while refusing at 1.0000. ⛔ **On a deleted VALUE, a table row-removal (0.00 s, Δ 0.0000) and an 8-byte bit-clear are strictly cheaper deletions than anything the field store can do**, because they hold no shared atoms to lose. The field store's answer is only that it starts **+11.4 / +11.4 / +6.9 tuples per corner above retrain-from-scratch**; that is where the trade sits, and it is the same shape w5 reported. ⛔ It is not softened here.

## 8. Independent replication of wave 5, on this run's own instrument

| quantity | w5 `hub/delete` | **re-measured here** |
|---|---|---|
| instant eviction, composition Δ | −2.0899 / −0.9101 / −6.7416 | **−2.0899 / −0.9101 / −6.7416** |
| coupling hole, composition Δ | 0.0000 | **0.0000** (exactly, all seeds) |
| field post-deletion vs retrain (GOLD) | +11.393 / +11.404 / +6.921 | **+11.3933 / +11.4045 / +6.9213** |
| refusal after eviction / hole | 1.0000 | **1.0000** |

⭐ Not quoted — re-run, by wrapping `exp_gen_delete_row7.build_arms` at runtime with **that module unedited on disk**, so every w5 arm kept its own code path byte for byte. **Charter §A.5's replication requirement is met in the strongest available form: the prior hub's numbers reproduce to four decimal places on all three seeds inside a run that also carries the new arms.**

## 9. Cost line

- **Wall clock:** START 06:25:57Z → final artifacts ~09:01Z, **~2 h 35 min**. Scoring: **4,212 units/seed = 36 arms × 117 units** (109 held-out corners + 8 constructed probes), ~2,500 judge passes/seed, ~10,500 CPU-s/seed, 3 threads. Seeds 1 and 2 ran in **resumable 1,500 s chunks** (2 resumes each).
- ⚠ **CHURN — a COST LINE only, ONE definition, ⛔ claimed nowhere.** *Bytes of `atom_life` written per decay step × steps to the refusal horizon*: **24 B × 539 = 12,936 B** (seeds 0, 1; 7 items on seed 2 → 28 B × 539 = **15,092 B**), against instant eviction's single **49,374 / 49,374 / 57,603 B** write — **0.2620×** on every seed. ⭐ **Churn is NOT the argument against decay**: running the lifetime channel to the refusal horizon writes about a quarter of the bytes eviction writes once. The argument is the composition curve, and this line exists so the two cannot be confused.
- **Retrain-from-scratch, with its cost:** table **3.45 / 3.81 / 3.72 s** on ~120,000 rows; MLP **50.47 / 40.49 / 39.87 s**, and it never refuses. **Never-inserted rebuild (BAR 4):** **43.2 / 48.6 / 47.8 s**.
- **Deletion target:** `object_hue = 0.0` (red) — **6 / 6 / 7 items**, 5 by provenance and 5 by geometry on seeds 0–1. ⛔ Not `scale`, not `orientation`, not `shape = cube`.
- **Process cost, honestly:** two launch failures and one resource decision. A `RecursionError` killed the first scored launch on all three seeds (the runtime wrapper called itself; fixed by binding the original at import, no numbers lost); and two of my own processes were stopped mid-run to keep aggregate memory off a 64 GB box shared with two other hubs, discarding ~17 min on seed 1 and ~17 min on seed 2 and forcing sequential scheduling. ⛔ **No box-wide kill at any point; every PID was identified by full command line before being stopped, and the other hubs' processes were read and deliberately left alone.**

## 10. Charter §C floor checks — one line each

- **§C.1 matched-bytes lookup** — ⭐ **CLEARED.** The post-deletion store beats its own atom-mean and nearest-atom lookups, on the same 524,288 B and the same address book, by **+21.75/+19.89/+20.89** and **+22.13/+20.02/+21.08** novel tuples per corner.
- **§C.2 landscape-removed / shuffled** — ⭐ **COLLAPSE.** Both to **0.0000** absolute novel tuples per corner on every seed.
- **§C.3 the tie clause** — ⭐ **SATISFIED and then some:** this object delivers *deletion that changes the output exactly* — at byte level, with `assert_no_refit` and a 1-byte conjunction hole.
- **§C.4 nearest-atom** — ⭐ **NOT a lookup table.** Settled state at relative distance **0.384 / 0.445 / 0.458** from its nearest atom.

## 11. What this closes, and what it hands on

1. ⛔ **Lifetimes as a deletion mechanism are CLOSED on the field store.** Decay reaches instant eviction's composition cost to within `+0.0562 / +0.0337 / +0.0112` and fires its refusal `8–14` half-lives *after* the cost is paid. ⭐ **The programme has carried "lifetimes" as an open mechanism for a long time; it can stop.** The task file said reporting this plainly would close a mechanism, and that is what this is.
2. ⭐ **The reason is exact and general, not a property of this seed or this target:** `n_refuse / n_half = ln(LIFETIME_EPS)/ln(1/2) = 19.9316` is a **pure constant**, so **no curvature can move the refusal horizon relative to the composition horizon**. ⛔ **There is no lifetime schedule that does better** — the whole family is one curve.
3. ⭐ **The one lever that WOULD move it is named and measured, and it is not a lifetime:** the gap exists only because the composition channel reads a **continuous weight** while the refusal channel reads a **discrete liveness bit at 1e-6**. ⚠ A refusal that read the basin's *live weight* instead of the liveness bit would move the two horizons together. ⛔ **I did NOT test that and make no claim about it** — `refusal` is 0/3 spent programme-wide and that change is a refusal-mechanism change. It is handed on as a named, unspent idea, not a result.
4. ⭐ **V5 App. G.8's commutation holds byte-exactly on a learned, multi-atom store** (0 bytes, 12 cells, 3 seeds) — and the hub's own reading of it is the finding: **disjointness buys exactness, not freedom.** The coupling hole is free because the read does not consult its component at all; decay is not free because the read consults its component continuously.
5. ⭐ **The never-inserted-history guarantee is priced:** 65 % of the image differs, and **only 16.5 % of that is placement**. Canonical placement alone cannot deliver it here; the pool-fitted address book and coupling table are 83.5 % of the obstruction.
6. ⛔ **Run 3/3 of the `deletion` target is RETURNED UNSPENT.** Nothing was selected on the held-out set, both pre-registered bars decided the target on the first run, and a second run would answer nothing that this one left open.

## 12. Artifacts

`/c/chlu/outputs/w6-retention/` — `START_UTC.txt` · `LOG.md` (this file) · `PREREG-w6-retention-1.md` · `pids.txt` · `run1/{law.json, bars.json, report.json, REPORT.md, costline.json}` · `run1/seed_{0,1,2}/{scores.npz, commutation.json, field/, tables/}` · `run1/seed_0/dev.json` · logs `score_s{0,1,2}.log`, `commute_s{0,1,2}.log`, `bars_*.log`, `costline_*.log` · `w6-retention.bundle`.
Code on branch `w6/retention`: `chlu/core/gen_retention.py` (the law), `chlu/experiments/exp_w6_retention.py` (the scored runner), `scripts/w6_retention_bars.py`, `scripts/w6_retention_costline.py`. ⛔ `chlu/experiments/exp_gen_delete_row7.py` and every other w5 file are **unedited on disk**; the frozen harness is untouched.

---

- **2026-09-20T09:06Z** — Outcome row appended to `/c/chlu/SCORED-LEDGER.md`: `w6 hub/retention | deletion | 2/3 SPENT — OUTCOME; 3/3 RETURNED UNSPENT`.
- **2026-09-20T09:07Z** — **Code committed and the branch bundled WHOLE.** `git status` clean on `w6/retention` at `3492231`; `git bundle create w6-retention.bundle w6/retention` → **"The bundle records a complete history"**, `bundle list-heads` shows `3492231... refs/heads/w6/retention`, 4,791,142 B. ⚠ Outputs verified by **directory listing** (gitignored, so `git status` cannot see them): 52 files, 90 MB, including `scores.npz` + `commutation.json` + `learners.json` + `c4_nearest_atom.json` for **all three seeds**.
- **2026-09-20T09:07Z** — ⛔ **No sub-agent and no process of mine is still running** (checked by `Win32_Process` command line, not by name). Every PID this hub launched is recorded in `pids.txt`; the two stopped mid-run were identified by full command line first, and ⛔ **no box-wide kill was issued at any point.**
