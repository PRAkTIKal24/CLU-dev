# gen-dynamical-read — experiment-engineer report

**Task + acceptance criterion:** BUILD the dynamical read (`request → clamp + request ENERGY → launch → settle over the WRITTEN ATOMS with a low-rank coupling → decode the settled state`) and smoke it end to end at tiny scale, with the bypass assertion demonstrated failing, both controls wired, the dynamics trace emitted and the byte ledger matching `PREREG.md` §3 — **⛔ no scored run, no verdict.**

**Status: done.** All six acceptance items met. ⛔ No falsifier answer is rendered.

## ⭐ DIAL DECLARATION (echoed, per protocol §7)
- **Dial:** *the read* — continuous settling over stored atoms, replacing codebook copy-back.
- **Laundering control:** ⭐ request-off on the **SAME chain** (same launch, same key, β = 0). Steering-attributable gain is the falsifier. Coupling-off mandatory beside it.
- **Falsifies:** the three falsifiers in `PREREG.md` §5 — ⛔ **adjudicated by the scored run, not here.**
- **Does NOT falsify:** losing to `knn_proto` on a metric-native protocol (metric-native-ceiling theorem).

## ⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner at the review that accepts this report
1. **`PREREG.md` §7's declared channel is confirmed at `D` = 2048 floats / 65,536 bits** (`knn_proto` parity). Any draft still carrying 30 bits for this arm is wrong. — *curator*
2. **The coupling strength cannot be 1.0.** Measured λ\* ≈ 0.054 at both budgets (§4 below). Any design text saying `coupling_strength = 1.0` (dreaming's value, carried into the design note §3) must be re-stated as *"λ capped by the coercivity certificate"*. — *Hub / curator*
3. **Dreaming's thresholded clamp rule is dead on atoms** (clamps 0.00 of 5 wells). The design note §4's "address-book clamp on the addressed wells" needs the argmax reading. — *Hub*
4. **Budget B's full arm is NOT CONVERGED at 1000 steps** (41.7 % of chains still descending on CORNER_B). R40.C's "1000 steps" is not yet validated for budget B; the scored run needs either more steps or a declared step-size/anneal change, **pre-registered before it runs**. — *Hub, before the scored run*

---

## What I did
- **New core module `chlu/core/gen_landscape.py`** (1,593 lines) — the whole path as one function.
  - `AtomStore`: `N` **stored train data points** per well, sliced through the *shipped* partition. `assert_atoms_are_train_rows` proves it **bit-for-bit** and also convicts a store collapsed to its own mean. ⛔ No code path produces a centroid.
  - Energy `E(x) = E_store(x) + β Σ_f [E_f(x) − E_store(x)] − λ E_couple(x)`:
    - `E_store = Σ_k −σ_k² logsumexp_i(−‖x_k−a_ki‖²/2σ_k²)` — kernel energy over the atoms, wells **emergent**, minima not enumerated (design note candidate **B**).
    - ⭐ **request = ENERGY TERM**, one expert per requested term over the well's atoms at declared strength β; at β = 1 the bracket is exactly the naive-Bayes log-likelihood-*ratio* `address_requests` composes in code space, lifted to the continuum. β = 0 is the request-off control on the same chain.
    - **coupling = LOW-RANK, label-free, train features only**: per pair `W_kl = U_klᵀV_kl` from the truncated SVD of the standardised cross-covariance. ⛔ Not dreaming's PMI table over discrete codes.
  - `launch_state` (request-only), `settle_chain` (MALA, energy/acceptance/displacement logged **every step from step 1**, annealed σ **and** a fixed-reference energy), `read_settled`, `convergence_report`, `settled_state_spread`, `pairwise_separation`, `save_trace`, `landscape_byte_ledger`, `decode_width`, `coupling_stability`/`safe_coupling_strength`, `arm_key`.
  - `assert_request_only` — the guarantee over **launch + settle + read as one path**, per stage, plus a structural signature leg.
- **New experiment `chlu/experiments/exp_gen_dynamical_read.py`** (817 lines) — config-load → frozen artifacts (grouping, shipped store, feature cache, winner decoder w/ hash + recipe guard) → per-budget landscape build + ledger assertion → 5 arms → decode → bypass report → per-unit incremental writes with `skip_if_exists`.
- **Config:** ONE new block `GenDynamicalReadConfig` + `gen_dynamical_read_config()`, **appended at EOF between `BEGIN`/`END` markers**, no `CHLUConfig` field, no `load_config`/`save_config` site.
- **CLI:** `chlu exp-gen-dynamical-read`, added between `BEGIN`/`END` markers in `experiment_cmd.py`.
- **Tests:** `tests/test_gen_dynamical_read.py`, **34 tests**, the load-bearing ones failing on purpose.

---

## How I verified

### Flag provenance (mandatory)
| field | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-dynamical-read` @ `4bc54b9` (base `main` @ `107fea9`) |
| dataset / pretext | `shapes3d` / `recon` |
| seed | **0** (⛔ single seed — this is a build smoke; a 3-seed cell is the scored run's) |
| bridge | `gen-feature-bridge/runs/shapes3d_recon/seed_0`, cache `scope="full"`, 480,000 rows, `D = 2048`, `feat_shape (128,4,4)`; encoder hash `98d3260283…`; **re-encode check 64 rows bit-identical** |
| partition | `gen-commitment-collapse` `labels_vec`, `K = 5`, well sizes **[303, 305, 573, 501, 366]** (Σ = 2048) |
| codebook store | `gen-read-compose/store_s0.pkl`, hash `3e3424a1ea40017e…` asserted, `M = 64`, 524,288 B |
| decoder | `gen-decoder-lift` **winner**, `dec_recipe="v1_capacity_schedule"`, param hash `01ef91f75eeaae7b…` — ⛔ **not** the `"baseline"` default |
| budgets | **A**: `N = 64`, `r = 16`; **B**: `N = 512`, `r = 128` (both declared in `PREREG.md` §3) |
| atom selection | `kmeanspp` **seeding over rows** (⛔ selects rows, never averages) |
| bandwidth | `nn_median`, `bandwidth_frac = 1.0` → A: σ = [4.491, 4.916, 8.928, 6.380, 4.917]; B: σ = [3.274, 3.649, 7.123, 4.992, 3.686] |
| settle | `steps = 1000`, `step_size = 0.2`, σ-anneal 4.0 → 1.0, `temp = 0.0` (pure settling; MH degenerates to accept-iff-energy-fell), `metropolis = True` |
| strengths | `request_strength = 1.0`; `coupling_strength` **declared 1.0, USED 0.04885 (A) / 0.04926 (B)** — capped by the certificate, see §4 |
| launch | `clamp_atom` (posterior **argmax**), `clamp_threshold = 0.5` (reporting only) |
| read | `read_mode = "state"` |
| convergence rule | `descent_window = 100`, `descent_tol = 1e-3` |
| rows | `store_train = 2048` (⚠ `--quick`), requests: `CORNER_B` = 72, `SPREAD` = 128 |
| ⚠ quick | `--quick` used: it shrinks **rows only**. ⛔ `N`, `r`, the ledger and `steps = 1000` were NOT shrunk. **No number here is a scored number.** |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0**, `CHLU_GEN_DATA_ROOT` / `CHLU_CLAUDE_ROOT` / `CHLU_GEN_BRIDGE_ROOT` all set |
| wall clock | **534.5 s** for the whole smoke (2 budgets × 2 request sets × 5 arms × 1000 steps) |

### Commands
```
# smoke, fresh workspace
PYTHONPATH=<worktree> .venv/bin/python -m chlu.experiments.exp_gen_dynamical_read \
  --quick --budgets A B --steps 1000 --out .claude/outputs/gen-dynamical-read/smoke
# tests
CHLU_GEN_DATA_ROOT=.../.claude/data/gen .venv/bin/python -m pytest \
  tests/test_gen_dynamical_read.py -q            -> 34 passed, 0 skipped, 0 failed
CHLU_GEN_DATA_ROOT=... .venv/bin/python -m pytest tests/test_config.py tests/test_cli_data.py \
  tests/test_gen_dream.py tests/test_gen_store.py tests/test_gen_dynamical_read.py -q -rs
                                                 -> 108 passed, 0 skipped, 0 failed (230 s)
.venv/bin/python -m ruff check <all 5 files>     -> All checks passed
```
**⛔ No full suite was started** (16 GB box, task prohibition). **0 skips**, so there is no skip to name.

---

## Findings

### 1. ⭐ Acceptance, item by item
| # | acceptance item | result |
|---|---|---|
| 1 | tiny-scale local smoke, full path, fresh workspace | ⭐ **PASS** — `.claude/outputs/gen-dynamical-read/smoke/`. Scale: **seed 0, 2,048 train rows, 72 + 128 full requests, N = 64/512 atoms, 1000 steps, both budgets, decoded to 64×64 RGB through the frozen winner decoder.** 534 s. |
| 2 | bypass assertion passes on the real path, **demonstrated failing** on a bypassed variant | ⭐ **PASS** — real path `{launch: 0.0, settle: 0.0, read: 0.0}` with the two feature tensors differing by `max|Δ| = 21.73 / 25.46`; the assertion **FIRED at all three stages** (`launch`, `settle`, `read`) in every one of the 4 budget×request-set cells, and the structural signature leg fires on a stage that takes `X`. |
| 3 | request-off and coupling-off wired and exercised | ⭐ **PASS** — plus `request_off_launch` (total request-off) and `zero_step` (dreaming's laundering control). See §3. |
| 4 | energy trace / acceptance / per-step displacement; convergence fraction | ⭐ **PASS** — full per-chain trace on disk every step (`energy_traces_*.npz`), `frac_still_descending` in §2. |
| 5 | byte ledger for both budgets, matching `PREREG.md` §3 exactly | ⭐ **PASS**, asserted in code (run aborts on mismatch). See §5. |
| 6 | targeted tests green with the data root set | ⭐ **PASS** — 34 / 108 passed, **0 skipped, 0 failed**. |

### 2. ⛔ Dynamics (`PREREG.md` §6, binding). Acceptance / displacement-from-step-1 / still-descending-at-1000
| budget | request set | arm | acc | disp[1] | disp[1000] | **still descending @1000** |
|---|---|---|---|---|---|---|
| A | CORNER_B | **full** | 0.786 | 0.0665 | 0.00078 | **0.000 — settled** |
| A | CORNER_B | request_off | 0.781 | 0.0825 | 0.00345 | ⛔ **1.000 — NOT CONVERGED** |
| A | CORNER_B | request_off_launch | 0.786 | 0.0811 | 0.00345 | ⛔ **1.000 — NOT CONVERGED** |
| A | CORNER_B | coupling_off | 0.682 | 0.0720 | 0.00024 | 0.000 — settled |
| A | SPREAD | **full** | 0.784 | 0.0665 | 0.00076 | **0.000 — settled** |
| A | SPREAD | request_off | 0.788 | 0.0812 | 0.00345 | ⛔ **1.000** |
| A | SPREAD | coupling_off | 0.637 | 0.0729 | 0.00020 | 0.000 |
| B | CORNER_B | **full** | 0.870 | 0.1032 | 0.00310 | ⛔ **0.417 — NOT CONVERGED** |
| B | CORNER_B | request_off | 0.873 | 0.1079 | 0.00097 | 0.000 |
| B | CORNER_B | coupling_off | 0.794 | 0.1007 | 0.00023 | 0.000 |
| B | SPREAD | **full** | 0.911 | 0.0973 | 0.00160 | ⛔ **0.172 — NOT CONVERGED** |
| B | SPREAD | request_off | 0.880 | 0.1027 | 0.00097 | 0.000 |
| B | SPREAD | coupling_off | 0.796 | 0.0976 | 0.00026 | 0.000 |

⭐ **This is not dreaming's chain.** Dreaming's acceptance was **0.952** on a chain that barely moved; here acceptance is **0.64 – 0.91** and the per-step displacement **falls 85 ×** (A: 0.0665 → 0.00078) over the run — i.e. it settles rather than sits still. `zero_step` is exact (displacement 0, energy unchanged).

⛔ **Budget B's full arm is NOT CONVERGED at 1000 steps** (0.417 / 0.172 of chains still descending). Reported as NOT CONVERGED, **never as a null**. **⚠ R40.C's 1000-step budget is validated for A and NOT for B.**

### 3. ⭐⭐ Request-off does NOT give the same trajectory — and it is dramatic. ⛔ This is the opposite of dreaming.
Mean **pairwise RMS separation** between the settled states of distinct requests (launch → settled):

| budget / set | arm | launch | settled | min pairwise | collapsed pairs |
|---|---|---|---|---|---|
| A / CORNER_B | **full** | 0.2478 | **0.3852** | 0.0000 → **0.0120** | 0.092 → **0.000** |
| A / CORNER_B | **request_off** | 0.2478 | ⛔ **0.0000** | 0.0 → 0.0 | 0.092 → 0.079 |
| A / CORNER_B | request_off_launch | 0.0000 | ⛔ 0.0000 | — | 1.000 → 0.801 |
| A / CORNER_B | coupling_off | 0.2478 | 0.1269 | 0.0000 → 0.0078 | 0.092 → 0.000 |
| A / SPREAD | **full** | 0.6772 | **0.7692** | 0.0000 → **0.0238** | 0.000 → 0.000 |
| A / SPREAD | request_off | 0.6772 | ⛔ **0.0000** | — | — |
| B / CORNER_B | **full** | 0.4153 | **0.7463** | 0.0000 → 0.0063 | 0.110 → 0.002 |
| B / CORNER_B | request_off | 0.4153 | ⛔ **0.0000** | — | — |
| B / SPREAD | **full** | 0.6497 | **0.9408** | 0.0000 → 0.0208 | 0.004 → 0.000 |
| B / SPREAD | coupling_off | 0.6497 | 0.3937 | 0.0000 → 0.0136 | 0.004 → 0.000 |

Three readings, in order of importance:
1. ⭐⭐ **With `request_strength = 0` every chain collapses to ONE state** (mean pairwise separation `6.0e-8` ≈ 0, identical final energy `−626.460` from two *different* launches). The landscape's own kernel energy has a single dominant basin and the launch is forgotten entirely. ⇒ **the request energy carries 100 % of the output's distinctness.** ⛔ The task's own "what would prove this spoke wrong" — *request-off already gives the same trajectory* — is **answered NO, emphatically**, at both budgets and both request sets.
2. ⭐ **The settling SEPARATES requests the discrete clamp COLLIDED.** 9.2 % (A) / 11.0 % (B) of launch pairs are at distance exactly 0 — two distinct requests addressing the same atom tuple, which is precisely the codebook failure this arm exists to escape. After settling, **0.0 % / 0.2 %** remain collided and the mean separation *rises* 1.55 × / 1.80 ×. This is the arm's mechanism visible at build scale.
3. **Coupling-off keeps ~1/2 to 1/3 of the separation.** So the coupling contributes, but it is **not** where the gain lives — the request energy is. ⛔ Whether any of this is a *gain* on a scored metric is not answered here.

⚠ **A metric trap I walked into and pinned:** the **participation ratio misled me**. Full-arm PR fell 5.31 (launch) → 1.08 (settled) and read like a collapse, while the separation had *risen*. PR counts **directions**, not distinct states. `pairwise_separation` is now reported beside it and both confusions are pinned by tests. ⛔ It is still **not** falsifier 2 (distinct factor tuples read by the external classifier off the decoded images).

### 4. ⛔⭐ MEASURED DEFECT #1 — the coupling is unbounded below at the declared strength; the fix is a certificate
The **first** real smoke diverged: `E → −1.6e28`, per-step displacement `8e11`. A bilinear coupling is an **indefinite** quadratic form. ⛔ Had this shipped, a 1000-step scored run would have reported it as *"still descending"* — a non-convergence — rather than as a **divergence**, and the trace would have looked plausible.

**The fix is derived from the atoms, not tuned.** With `x̃_k = (x_k−m_k)/s_k`, `u_k = ‖x̃_k‖`: the kernel energy obeys `E_k ≥ min_i‖x_k−a_ki‖²/2 − σ_k²logN` (quadratic coefficient `s_k²/2`) and `|x̃_kᵀW_kl x̃_l| ≤ ‖W_kl‖₂ u_k u_l`, so `E ≥ uᵀCu − const` with `C_kk = s_k²/2`, `C_kl = −λ‖W_kl‖₂/2`. Coercive iff `C ⪰ 0`; since `C(λ) = D − λS` the largest safe strength is **`λ* = 1/ρ(D^{−1/2} S D^{−1/2})`, closed form.** `‖W_kl‖₂` is read off `r×r` matrices (non-zero eigenvalues of `WᵀW` = those of `(UUᵀ)(VVᵀ)`) — at `r = 128` the explicit `W` would be 303×573 floats **per pair**.

**Measured:** λ\* = **0.05428** (A) / **0.05473** (B); `min eig` at the declared λ = 1 is **−1.884**. The used strength is `0.9 λ*` = **0.04885 / 0.04926**, and **the cap is logged on every run**. Every pair's spectral norm is 1.000 by construction (`coupling_normalise`), so λ\* is set purely by the well radii `s_k ≈ 0.45–0.51`.
⛔ **Consequence for the design note:** `coupling_strength = 1.0` (dreaming's value, carried into design-note §3) is **not available** on this landscape.

### 5. ⭐ Byte ledger — matches `PREREG.md` §3 **exactly**, asserted in code
| | store | coupling | total | × budget | D3 binding? |
|---|---|---|---|---|---|
| **budget A** (`N=64`, `r=16`) | **524,288 B** | **524,288 B** | **1,048,576 B** | **2.00×** | ✔ exactly |
| **budget B** (`N=512`, `r=128`) | **4,194,304 B** | **4,194,304 B** | **8,388,608 B** | **2.00×** | ✔ exactly |

Measured `n_bytes()` of the live objects equals the ledger to the byte at both budgets (asserted; run aborts otherwise). `store_bytes` is byte-matched to the shipped codebook's 524,288 B at `N = M = 64`. ⭐ **Coupling identity:** `Σ_{k<l} r(d_k+d_l)·4 = r(K−1)D·4` whenever the wells partition `D` — **independent of how the wells are cut**, so a per-seed drift in `d_k` cannot silently renegotiate the pre-registered rank (asserted over three very different partitions of 2048).
**Declared channel:** `D` = **2,048 float32 = 65,536 bits**, `knn_proto` parity. ⛔ Nothing in the module is expressed as `K·log₂M = 30 bits`. Companion spread reported per arm.

### 6. ⛔ MEASURED DEFECT #2 — dreaming's thresholded clamp clamps **0.00 of 5 wells** on atoms
Mean posterior confidence over atoms: **0.0508** at `N = 64` (floor 1/64 = 0.0156, so 3.3× chance) and **0.0125** at `N = 512` (floor 0.00195, 6.4× chance). At `clamp_threshold = 0.5` the rule clamps **0.00 of 5** wells, so the "clamp" degenerated to the prior's argmax and the launch carried **no request information at all** (`request_off_launch` was then bit-identical to `request_off`).
**Resolution:** nothing is frozen in this arm, so the threshold decides nothing. `clamp_atom` now takes the posterior **argmax** always; `mean_n_clamped` survives as the diagnostic it actually is; dreaming's rule is kept as the `clamp_atom_thresholded` control.
⭐ **Note the address book is still informative** (3.3–6.4× chance) — it is *diffuse*, not uninformative, which is the expected shape for a naive-Bayes posterior over 64–512 atoms.

### 7. ⛔ MEASURED DEFECT #3 — `--quick` silently clobbered explicit flags
`apply_quick` **replaces** the seed/budget/arm/step lists, so `--quick --budget A B --steps 1000` ran **budget A only** and the log looked entirely normal. Fixed in both the module CLI and `chlu exp-gen-dynamical-read` (**quick first, explicit flags last**) and pinned by a monkeypatched test. ⚠ Worth checking the other `exp_gen_*` mains for the same ordering.

### 8. Context numbers (⛔ not scored, not comparable across configs)
- Codebook read error on the **SPREAD** requests: **0.27598** (128 rows, seed 0) — consistent with `PREREG.md`'s anchor **0.2783 ± 0.0064**, so the reference row this arm will be measured against reproduces.
- Codebook on **CORNER_B**: relative read error **1.7150**, **8 distinct codes on 72 requests** (⚠ the corner is degenerate: `shape` realises **1 of 4** values; `n_values_realised` is on every row of the JSON).
- Decoded images are real 3×64×64 in [0,1] (`full`, A: mean 0.570, std 0.372) — the path reaches an image. ⛔ No probe, no classifier, no accuracy: those are the scored run's.

---

## Git footprint
- **Branch** `agent/experiment-engineer/gen-dynamical-read`, base **local `main` @ `107fea9`** (protocol §3.5; ⛔ never `origin/main`). Rebase onto `main` = no-op (base has not moved). Worktree `../CHLU-gen-dynread`; **branch ref verified from the MAIN repo**: `git log --oneline main..agent/experiment-engineer/gen-dynamical-read` shows all 4 commits. ⛔ Not pushed, no PR.
- Commits: `5a94f06` (the module + config + experiment + CLI) · `df57f1d` (coercivity certificate + clamp fix + 31 tests) · `1904ca4` (pairwise separation + flag-ordering fix) · `4bc54b9` (ledger test grounded in the measured wells).
- Files: `chlu/core/gen_landscape.py` **(new, 1593)** · `chlu/experiments/exp_gen_dynamical_read.py` **(new, 817)** · `tests/test_gen_dynamical_read.py` **(new, 544)** · `chlu/config.py` **(+227/−1)** · `chlu/cli/experiment_cmd.py` **(+108/−0)**.
- ⚠ **Merge note for the shared files.** `chlu/config.py` shows **1 deletion**: it is *only* the peer's `# --- END gen-variance-equalisation ---` line losing its no-trailing-newline status; **the identical line is re-added**. Net semantic deletions = **0**. My block is one contiguous region at EOF between `# --- BEGIN gen-dynamical-read` / `# --- END gen-dynamical-read`, with **no `CHLUConfig` field and no `load_config`/`save_config` site**. `experiment_cmd.py` likewise: two marker-delimited regions, **0 deletions**. ⛔ On conflict take the peer's whole file and re-apply my blocks by their markers, asserting 0 deletions.
- **Collision check:** the shared main checkout was on another agent's branch (`agent/experiment-engineer/pilot-ttt-nan-and-d5-wiring`) with 5 other worktrees live, so I worked entirely in `../CHLU-gen-dynread` and never touched the shared checkout.

## Artifacts
`.claude/outputs/gen-dynamical-read/smoke/` (19 MB) — `runs/shapes3d_recon/seed_0/dynread_s0.json`, `summary.json`, 20 × `energy_traces_s0_{A,B}_{CORNER_B,SPREAD}_<arm>.npz` (**per-chain energy, energy_ref, accept_prob, accepted, displacement — every step**), 4 × `settled_states_s0_*.npz`, 4 × `decoded_*.png`, `smoke.log`.
`.claude/scratch/gen-dynamical-read/core_smoke.{py,log}` — the dependency-free core smoke.
⚠ These are **smoke** artifacts under `smoke/`, deliberately *not* at `PREREG.md` §9's `runs/<dataset>_<pretext>/budget_{A,B}/seed_*/` paths, so the scored run cannot be confused with them. (⚠ The code's own layout is `runs/<dataset>_<pretext>/seed_*/` — **it does not yet carry §9's `budget_{A,B}/` level**; both budgets write into one seed directory with the budget in the filename. One-line change if the Hub wants §9 literally.)

## Open questions / follow-ups / risks
1. ⛔ **Budget B does not converge in 1000 steps** (§2). Needs a pre-registered decision *before* the scored run: more steps, larger `step_size`, or a different σ-anneal. ⛔ Do not let the scored run discover this.
2. ⚠ **The bandwidth `σ` is the interpolation knob and it is currently unvalidated.** Too small ⇒ the minima *are* the atoms and the read is discrete again (the Advisor's constraint re-entering by the back door); too large ⇒ one blob per well. `bandwidth_frac = 1.0` at `nn_median` is a *declared default*, not a measured optimum. A cheap σ-sweep against read error is the single highest-value pre-scored-run measurement.
3. ⚠ **The coupling is confined to ~5 % of its declared strength** by the certificate (§4). If the Hub wants the coupling to matter more, the honest lever is a *bounded* interaction (the bilinear form cannot be), not a larger λ.
4. `read_mode` defaults to `"state"` (PREREG §1's literal wording). `"atom_mixture"` — the softmax generalisation of `read_from_codes`' argmax gather, output inside the atoms' convex hull — is implemented and untested at scale. The scored run should declare which one it reads, **before** it runs.
5. **Falsifier 2 risk, flagged early:** the settled states are all distinct and *more* separated than the launches, but distinctness at the **decoded factor tuple** level is untested. The codebook itself gives only **8 distinct codes on the 72 corner requests**, so the headroom there is large but unmeasured.
6. ⚠ Seeds 1–2 use `scope="used"` caches; my smoke ran seed 0 (`scope="full"`) only. The re-encode path is exercised by `_assert_cache_reencodes` but the **seed 1/2 gather-with-extension** path is untested by me.
7. **Falsifier-1's rivals are not built** (`store_removed`, codebook-argmax, D6 best-codebook, atom-mean lookup, `knn_proto` ceiling). The task scoped build-only; they are a separate build before the scored run.

---

## Proposed handover updates (for the Hub)
- **§7 NEW — `--quick` clobbers explicit CLI flags in the `exp_gen_*` mains.** `apply_quick` *replaces* seed/budget/arm/step lists, so an explicit flag applied before it is discarded and the log looks normal (measured: `--quick --budget A B` ran budget A only). Fixed in `exp_gen_dynamical_read` + its CLI hook (**quick first, explicit flags last**, pinned by a test). ⚠ **The other `exp_gen_*` mains have the same ordering and are unaudited.**
- **§7 NEW — a bilinear inter-well coupling is UNBOUNDED BELOW.** Measured at `E = −1.6e28`, per-step displacement `8e11`. Anyone building a continuous coupling must carry `gen_landscape.coupling_stability`'s closed-form coercivity certificate (`λ* = 1/ρ(D^{−1/2}SD^{−1/2})`); measured λ\* ≈ **0.054**, so **`coupling_strength = 1.0` is not available** and design text quoting it must be corrected.
- **§7 NEW — dreaming's thresholded clamp rule is dead over atoms.** Clamps **0.00 of 5** wells at threshold 0.5 (mean confidence 0.0508 at N=64 / 0.0125 at N=512, i.e. 3.3×/6.4× chance). Nothing is frozen in the dynamical read, so the threshold decides nothing; `clamp_atom` = posterior argmax.
- **§7 NEW — the participation ratio is NOT distinctness.** PR fell 5.31 → 1.08 while the mean pairwise separation *rose* 0.248 → 0.385 and the *minimum* rose from 0.0. Use `gen_landscape.pairwise_separation` for "are these states different"; PR counts directions.
- **§3 / config defaults — new block `GenDynamicalReadConfig` at EOF of `chlu/config.py`** (accessor `gen_dynamical_read_config`, no `CHLUConfig` field, marker-delimited, append-only). Notable defaults: `dec_recipe="v1_capacity_schedule"` (⛔ never `"baseline"`), `steps=1000`, `budgets=["A","B"]`, `enforce_coupling_certificate=True`, `launch_mode="clamp_atom"`, `read_mode="state"`. New CLI: **`chlu exp-gen-dynamical-read`**.
- **JAX landmine confirmed, with a version:** on **JAX 0.9.0** in this venv, `jax_threefry_partitionable` is ON and `split(k,2)[1] == split(k,3)[1]` **bit-for-bit** (asserted as a positive control in `tests/test_gen_dynamical_read.py`). `gen_landscape.arm_key` (`fold_in` on a sha256 of the arm label) is the pattern; ⚠ any other harness deriving arm keys by lengthening a split is suspect.
- **Design note reconciliation:** candidate **B** (atoms) is now built and costs exactly `PREREG.md` §3's ledger; the note's §3 "full `W_kl` at 12.6× budget" is superseded by the low-rank fit at **1× budget**, and D4's open question is answered — the declared channel is `D` = **2,048 floats / 65,536 bits**, `knn_proto` parity.
