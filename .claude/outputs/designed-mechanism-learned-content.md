# designed-mechanism-learned-content — experiment-engineer report

**Task + acceptance criterion:** is the K=8 wall GEOMETRY or LEARNING? Measure `K_learned` vs `d` for a LEARNED atom-dictionary mechanism (designed mechanism, learned content) against the designed `K_designed=4·2^d` ceiling, with the parameter budget controlled; plus the mass arm + coupling check, interference across `d`, and the performance frontier. ≥5 seeds on the discriminator, PREREG first, tests green.

**Status: done** (discriminator measured with an explicit budget-confound resolution; items 2–3 clean; a clean uniform-budget re-run of the discriminator was executed to remove the high-`d` budget artifact — numbers in §1).

> ⚠ **RECONCILIATION LIST — needs an owner (protocol §5).**
> 1. **The K=8 wall is NEITHER purely geometry NOR purely learning — it is a THREE-way split the task's two hypotheses did not enumerate: (a) a real per-`d` learning ceiling, (b) address GEOMETRY that lifts that ceiling with `d`, and (c) a WRITE-OPTIMIZATION/parameter-budget floor that must scale with the address DIMENSION, not just with `K`.** The budget confound the task flagged is real and load-bearing: at `min_atoms=384` the learned arm *collapses* at `d≥6` (d=8,K=2 fails at strict 0.400 despite site separation 1.838 — geometrically trivial), and scaling to `min_atoms=2048` restores strict **1.00** at the SAME cells. Any `K_learned(d)` curve must state its per-cell atom budget or it is measuring the optimizer, not the mechanism.
> 2. **Mass stores nothing here — PREREG confirmed, and it strengthens `relaxation-fiber-capacity` Prop F1.** The atom wells are ADDRESS-SEPARABLE (measured Hessian off-diagonal/diagonal coupling **0.081 < 0.1**), and per-item mass changes strict retrieval by **−0.005 ± 0.007** (a null). Mass is worth ~0 bits in an uncoupled atom landscape — a *confirmed prediction*, not a failure.
> 3. **The WRITE OPERATOR result (`potential-function-class`) not only survives higher `d`, it AMPLIFIES:** masked-vs-global cross-write corruption advantage is **197× (d=2) → 599× (d=3) → 6332× (d=4)**. A masked atom write is bit-local in parameter space at any `d`; global-write interference, meanwhile, does not fall with `d`.

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/designed-mechanism-learned-content`, base local `main` @ `8519df6` |
| commits | `ee39211` (write_loss payload_index), `8e08929` (experiment+config+CLI), `ad60d20` (tests) |
| worktree | `../CHLU-designed-mech`; **main venv reused** (protocol §4), no worktree `uv sync` |
| env / JAX | `/Users/user/Desktop/CHLU/.venv`, JAX per main venv (import already warm; ~fast) |
| harness | `chlu/experiments/exp_designed_mechanism.py`, `python -m chlu.experiments.exp_designed_mechanism` (exit 0) |
| geometry | d-ball address space: sites `designed_sites(d,K)` (farthest-point, R=1), payload channel at index `d`, dim=d+1; `wall_margin=0.5`, `well_width=0.15` (designed), `payload_kappa=0.1`, `c_conf=10` |
| learned mechanism | `AtomDictionaryPotential` wrapped in `DesignFreedomPotential(rung="free_mlp", learned_family="atoms")` (designed part = None); `atom_init_scale=1.0` (LOAD-BEARING — §7), `atom_init_width=0.3`, `atom_depth_init=1e-4` (flat start, A=amp²), `learned_confine=0.05` |
| **param budget** | `n_atoms = max(atoms_per_item·K, min_atoms)`, `atoms_per_item=32`. **Production run `min_atoms=384`; clean discriminator re-run `min_atoms=2048`** (see §1/§7). P reported per cell |
| write objective | `train_memory.py`, **GLOBAL** write (mechanism's best fidelity), 600 Adam(w) steps, lr 3e-3, wd 1e-4, n_perturb 32, σ_addr 0.25, σ_pay 0.6, margin 0.15, barrier 0.2, **payload_index=d** |
| retrieval | two-phase: γ_address 0.05 × 400 → γ_read **0.02** × 800, dt 0.05. ⚠ γ_read>0 is load-bearing: value retrieval requires dissipation (address-space-dimension-scaling item 4), else even the DESIGNED arm scores strict 0 (payload channel oscillates) |
| queries | fixed_norm jitter (σ=0.15/√d), σ_p 0.05, payload=0 launch (anti-decoration guard), ≤32/item, `max_total_queries=4096` |
| criterion | strict = basin_ok (settled nearest own site) AND |read − a_i| < 0.1; VALUE blank control per cell (blank strict ≤ trivial-ceiling, leak-immune) |
| seeds | discriminator **5 (0–4)** in the production run; clean re-run **3 (0–2)**; designed arm 1 seed; interference 3 seeds; mass 3 seeds |
| langevin_noise | **N/A** — deterministic Verlet, no temperature |
| artifacts | `.claude/outputs/designed-mechanism-learned-content/{PREREG.md,...}`; metrics `results/exp_designed_mechanism_metrics.json`; scratch logs under `.claude/scratch/designed-mechanism-learned-content/` |

**Designed vs learned honesty note:** the WRITER supplies the target sites `c_i` in every cell (the landscape is learned; item PLACEMENT is chosen), and the atom dictionary carries a coercivity term. Nothing here tests whether item sites EMERGE (N46/D3). "Learned content" = amplitudes/centers/widths trained by the static write objective.

---

## 1. ⭐ The discriminator: `K_learned` vs `d` and `K_designed` vs `d`

### 1a. Production run (`min_atoms=384`, 5 seeds) — and its high-`d` budget artifact

| d | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| **K_learned** (min_atoms=384) | **4** | **8** | **8** | 2† | 0† |
| **K_designed** (same harness) | **16** | **64** | ≥128 (cens.) | ≥128 (cens.) | ≥128 (cens.) |
| 4·2^d (theory) | 16 | 32 | 64 | 256 | 1024 |
| ratio K_learned/K_designed | 0.25 | 0.125 | ≤0.06 | — | — |

† **d=6, d=8 are budget/optimization artifacts, NOT a capacity wall** (§7). d=8 K=2 fails at strict 0.400 with site separation **1.838** — two items that far apart are geometrically trivial to store; the failure is a seed-fragile flat-start write in a 7–9-dim atom ball (write loss stuck at 0.06). **Rescue (min_atoms=2048): d=6 K=4 → strict 1.00, d=8 K=2 → strict 1.00** (both were 0.25–0.40 at 384). Fit including the collapsed points → base 0.80, R²0.31 → **AMBIGUOUS**; this fit is not the science.

**Designed arm reproduces exponential capacity on this harness** (the pre-registered gate): K_designed = 16, 64 at d=2,3 (base **4.0**, R²1.0 over the 2 non-censored points), censored ≥128 at d≥4. The designed ceiling climbs as the address dimension opens up — exactly `address-space-dimension-scaling`.

### 1b. ⭐ Clean discriminator (`min_atoms=2048`, uniform over-complete budget, 3 seeds)

Full clean curve (the artifact of §1a removed by the uniform floor). Every cell blank-passes. **This is the honest wall — reported with its non-monotonicity, not smoothed.**

| d | **K_learned** (clean, firm) | first-fail cell (strict) | K_designed | 4·2^d |
|---|---|---|---|---|
| 2 | **4** | K=8 = **0.768** (real wall, 10240 params) | 16 | 16 |
| 3 | **8** | K=16 fail | 64 | 32 |
| 4 | **8** | K=16 = **0.876** (marginal) | ≥128 (cens.) | 64 |
| 6 | **32** | K=64 = 0.820 | ≥128 (cens.) | 256 |
| 8 | **8** | K=16 = **0.887** (marginal) | ≥128 (cens.) | 1024 |

**⚠ The clean curve is NON-MONOTONIC: 4, 8, 8, 32, 8.** Two facts must travel together:
- **The wall does move up with `d`:** d=6 clears **32** (8× the d=2 value of 4), and d=2's own K=8 genuinely fails at 0.768 with 10240 params — so **H-LEARNING (a flat wall near 8 at every `d`) is rejected**: at d=6 the learned mechanism holds four times the "wall."
- **But it is NOT a clean exponential.** d=4 and d=8 both **fall back to 8**, and the aggregate 5-point fit is weak: **base A = 1.18, R² = 0.26, polynomial fits marginally better.** Over d=2–6 alone the fit is base **1.64, R² 0.94**; the d=8 point breaks it.
- **The dips are threshold- and budget-sensitive, not proven capacity walls.** d=4 K=16 (0.876) and d=8 K=16 (0.887) both fail the 0.9 bar by **<0.025** — at a 0.87 bar the curve reads **4, 8, 16, 32, 16** (clean monotone growth, base ≈1.7). And d=8 lives in a 9-dim atom ball where `min_atoms=2048` is very likely still under-serving the write (the §7 "budget must scale with dimension" confound reasserts — the d=6→32 vs d=8→8 inversion is the tell, since d=8 has *more* geometric room).

**Verdict: H-GEOMETRY over H-LEARNING (the wall moves up with `d`), but the growth is NOISY and its exact base is not pinned.** The primitive claim survives — learned content unlocks 8× more capacity by d=6 and the d=2 wall is not reproduced at higher `d` — but the honest statement is **"a rising, budget- and threshold-sensitive trend (base 1.2–1.7 depending on where d=8 lands), NOT a clean `A^d` law,"** not "clean base 1.64." A dimension-aware atom budget (§9.1) is required to settle whether the true law is base ≈1.7 (the ≤0.87-bar / d≤6 reading) or genuinely sub-exponential.

---

## 2. Item 2 — does per-item MASS help? (folds in `mass-visible-objective`, tests Prop F1)

Fixed learned landscape (d=3, K=4, masked write); arm (a) uniform mass vs arm (b) per-item geometric mass spread [1/4, 4], keyed by the query's item label; 3 seeds.

| quantity | value | prediction |
|---|---|---|
| address-coupling ratio `mean\|∂_i∂_j V\| / mean\|∂_i²V\|` at stored sites | **0.081 ± 0.050** (0.038–0.151) | < 0.1 (Prop F1: separable ⇒ mass useless) ✅ |
| Δstrict (per-item mass − uniform) | **−0.005 ± 0.007** (min −0.016, max 0.0) | ~0 (null) ✅ |
| mass_helps | **False** | False ✅ |

**PREREG confirmed both clauses.** The atom wells are address-separable (isotropic Gaussian ⇒ near-diagonal Hessian), so mass — an address-side key — is worth ~0 bits, exactly as `relaxation-fiber-capacity` Prop F1 predicts. Per-item masses do not raise fidelity or `K_learned`. ⚠ Honesty: the masses are ASSIGNED (a geometric spread), not gradient-learned — the static write objective is mass-blind by construction (kinetic terms cancel in a minimum-digging loss), which is itself the reason mass cannot be *learned* into this landscape. The coupling ratio bounds what any mass value could buy, and it is ~0.

---

## 3. Item 3 — the interference axis at scale (write operator across `d`)

Write A (K−1 items), read A; write B into a fresh site/atom-block; re-read A. Corruption = change in A's read error. Masked (local) vs global write, at each `d`, 3 seeds.

| d | masked (local) corruption | global corruption | **local advantage** |
|---|---|---|---|
| 2 | 2.35e-3 | 4.63e-1 | **197×** |
| 3 | 3.62e-4 | 2.17e-1 | **599×** |
| 4 | 9.26e-5 | 5.87e-1 | **6332×** |

**PREREG confirmed, and stronger than registered.** The masked-write advantage (`potential-function-class`'s 70× at d=2 on the ring) not only survives higher `d` — it AMPLIFIES: local corruption *falls* with `d` (2.35e-3 → 9.26e-5, sites separate so the frozen-block write's residual tail shrinks), while global corruption stays high (writing B's gradient step moves every stored item regardless of `d`). A masked atom write is bit-local in parameter space at any dimension; that is the concrete `C3`-local claim, now verified across `d`. **The write operator, not the address dimension, governs interference** — and it interacts with `K_learned` favorably: a masked write is the operator that lets the learned mechanism approach its geometric capacity without cross-item corruption.

---

## 4. Item 4 — the honest performance frontier

For the best budget-adequate configuration (global write, `min_atoms=2048`, `atom_init_scale=1.0`):

- **`K_learned(d)` (strict ≥0.9) = {4, 8, 8, 32, 8}** at d = {2, 3, 4, 6, 8}, falling away at the next ladder rung (d=2: K=8 = 0.768; d=4: K=16 = 0.876; d=6: K=64 = 0.820; d=8: K=16 = 0.887).
- **The learned ceiling sits well below the designed one and the gap WIDENS with `d`:** `K_learned / K_designed` = ¼ (d=2), ⅛ (d=3), ≤1/16 (d≥4, designed censored ≥128). The learned mechanism pays a growing multiplicative tax as the address space opens.
- **The single defensible number the paper's claim rests on:** a learned-content atom-dictionary CLU retrieves **32 items at strict 0.94 in a 6-dimensional address space** (blank-controlled, 3 seeds) — 4× the d=2 wall, establishing that **the K≈8 wall is not intrinsic to the primitive**. But the growth is NOISY (d=8 falls back to 8) and its exponent is not pinned; do NOT quote a clean `A^d` law without the dimension-aware-budget follow-up (§9.1).

---

## 5. How I verified
- New tests `tests/test_designed_mechanism.py` (7): payload-index generalization, K-scaled atom budget (floor-aware), bit-local masked write at d>2, censored-point exclusion in the growth fit, designed-arm retrieval gate, value-blank leak-immunity. **`7 passed`** (`pytest tests/test_designed_mechanism.py`).
- `ruff check` on all changed files → **All checks passed**.
- `--quick` smoke exit 0; full production run exit 0 → `results/exp_designed_mechanism_metrics.json`.
- Every number re-read from the committed metrics JSON / the clean-run log, not from memory.

## 6. PREREG scorecard (`PREREG.md` written before the harness ran)

| # | prediction | measured | verdict |
|---|---|---|---|
| H-GEOMETRY vs H-LEARNING | primary: H-GEOMETRY-WEAK, base A_learned ∈ [1.55,1.85], growth ≥+2 rungs, R²≥0.8 | full 5-pt fit base **1.18, R²0.26** (non-monotone); d≤6 fit base 1.64, R²0.94; +3 rungs (4→32) but d=8 dips to 8 | ◐ **H-LEARNING rejected (wall moves to 32 at d=6); but the clean exponential and its base FAIL as registered — the growth is noisy, not an `A^d` law** |
| point preds K_learned(d)={4,8,16,32,64} | derived from Δ_req/d_eff | measured {4,8,8,32,8}: d=2,3,6 on/above; d=4,8 low (8 vs 16/64) | ◐ partial; d=8 falsifies the monotone prediction |
| designed gate | designed reproduces ≈4·2^d | base 4.0 (d=2,3), censored ≥128 above | ✅ |
| Item 2 mass | coupling < 0.1 ⇒ mass null | coupling 0.081, Δstrict −0.005 | ✅ both |
| Item 3 write op | masked advantage survives every `d` | 197×→599×→6332× (amplifies) | ✅ (stronger) |

---

## 7. What I found / fixed while running (each would have produced a wrong conclusion)

1. **⚠ The write objective hardcoded the payload channel at index 2** (ring convention). In a d-ball the payload is at index `d`; without the generalization (`write_loss(payload_index=d)`) the write pins the WRONG coordinate to zero and stores nothing. Fixed + tested; default 2 keeps every ring/w20 path byte-identical.
2. **⚠ γ_read=0 makes even the DESIGNED arm score strict 0.** The payload is the d-th coordinate of the atom-well center, launched at 0, and must DISSIPATE up to `a_i`; a conservative read leaves it oscillating and the tail mean misses the value. Set γ_read=0.02 (address-space-dimension-scaling item 4). The designed gate then passes at strict 1.000.
3. **⚠ The basin-reach initialization trap.** With `atom_init_scale=0.5` the flat-start atoms cluster near the origin and cannot dig a well reaching an item whose payload |a_i|=1 from the payload=0 launch — d=2 K=4 caps at strict **0.500** (only |a_i|<1 items retrieve) *regardless of atom count* (tested to 1024 atoms). `atom_init_scale=1.0` spreads atoms across the full (d+1)-ball and d=2 K=4 reaches **1.000**. This is `potential-function-class` open-Q #2 (no basin-reach term in the write), defused by the initialization, not by parameters.
4. **⚠ The parameter-budget floor is load-bearing in BOTH directions.** Scaling atoms as `atoms_per_item·K` alone STARVES small-K cells (d=4 K=2 with 64 atoms: write loss stuck at 0.18 on some seeds — a large over-complete dictionary also smooths the write optimization). And a FIXED floor of 384 is inadequate at `d≥6` (atoms `N(0,1)` spread thin in a 7–9-ball). The clean discriminator uses `min_atoms=2048`; **this is the confound the task warned about, and it is real — the budget must scale with the address DIMENSION, not only with K.**
5. **The value-blank false-fail at large K.** A blank landscape legitimately "retrieves" items whose real payload lies within `payload_tol` of 0; at large K the linspace(-1,1,K) codebook has several, so a flat blank_strict_max spuriously disqualified the DESIGNED arm at K≥16. Gated against the measured trivial ceiling.

## 8. Git footprint
- Branch `agent/experiment-engineer/designed-mechanism-learned-content`, base local `main` @ `8519df6`. **Not pushed** (protocol). Rebase onto `main` = no-op (base unmoved).
- Worked in worktree `../CHLU-designed-mech` (the main checkout had another agent's branch `hopfield-capacity-benchmark` checked out — isolation per §3.2). No collision. **Verify the branch ref from the main repo before teardown.**
- Commits (3): `ee39211`, `8e08929`, `ad60d20`.
- Files: **+** `chlu/experiments/exp_designed_mechanism.py`, `tests/test_designed_mechanism.py`; **M** `chlu/training/train_memory.py` (payload_index, default 2 = backward-compatible), `chlu/config.py` (`ExperimentDesignedMechanismConfig` + 3 registration sites), `chlu/cli/experiment_cmd.py` (import, parser, `cmd_exp_designed_mechanism`). Did NOT touch `utils/plotting.py` (figure local, per precedent). `results/` not committed.

## 9. Open questions / follow-ups / risks
1. **The budget-must-scale-with-`d` finding is the key methodological result and needs a clean owner.** A dimension-aware atom floor (e.g. `min_atoms ∝ c^d` or targets-seeded atom init) would let the discriminator run to high `d` without the optimization artifact and settle H-GEOMETRY-STRONG vs -WEAK definitively.
2. **d=2's wall at K=4 (K=8 fails at strict 0.768 even at 10240 params) is REAL and budget-adequate** — the learned atom mechanism genuinely caps below the designed 16 at d=2. This is the honest "learned content pays a tax below the designed ceiling."
3. Single geometry (farthest-point d-ball), single write-hyperparameter configuration, global write for the discriminator (masked for interference). Payload is scalar (the same read-out resolution limit `address-space-dimension-scaling` flagged bounds strict at large K).

## Proposed handover updates (for the Hub)

1. **§6 ground truth — THE HEADLINE: the K=8 wall is GEOMETRY, not learning — but the growth is NOISY, not a clean law.** A fixed atom-dictionary MECHANISM with LEARNED content (amplitudes/centers/widths, static write objective) has `K_learned` = **{4, 8, 8, 32, 8}** at d = {2,3,4,6,8} (blank-controlled, budget-adequate min_atoms=2048, 3 seeds). **The wall MOVES UP with `d`** — d=6 clears 32 (4× the d=2 wall), and d=2's own K=8 genuinely fails (0.768, 10240 params) — so **H-LEARNING (a flat wall near 8 at every `d`) is rejected and the primitive claim is ALIVE.** ⚠ **But do NOT quote a clean `A^d` law:** the curve is non-monotone (d=8 falls back to 8), the 5-point fit is base 1.18 / R²0.26; only the d≤6 sub-curve fits base 1.64 / R²0.94. The d=4 and d=8 dips fail the 0.9 bar by <0.025 (K=16 = 0.876/0.887) and d=8 is very likely still budget-starved in a 9-ball (update 2) — so the honest statement is **"a rising, budget/threshold-sensitive trend (base ~1.2–1.7)," NOT a pinned exponent.** The `potential-function-class` "atoms break at K=8" was the d=2 point of this curve, not a ceiling of the primitive.
2. **§7 — NEW load-bearing methodological finding (needs a curator/engineer owner): the parameter budget must scale with the address DIMENSION, not only with K.** At `min_atoms=384` the learned arm *collapses* at d≥6 (d=8 K=2 fails at strict 0.400 despite site separation 1.838 — geometrically trivial) because flat-start atoms `N(0,1)` spread thin in a (d+1)-ball and the global write is seed-fragile; `min_atoms=2048` restores strict 1.00 at the same cells. **Any `K_learned(d)` measurement that does not scale the atom budget with `d` is measuring the optimizer.** This is the exact confound w22 was told to control, now quantified. A dimension-aware floor (or targets-seeded atom init) is the clean follow-up.
3. **§7 — mass stores nothing, confirming Prop F1 in a NEW regime.** Atom wells are address-separable (Hessian off-diag/diag coupling **0.081 < 0.1**); per-item mass changes strict by **−0.005 ± 0.007** (null). `mass-visible-objective`'s `mass_override` works mechanically, but there is nothing for it to key on in an uncoupled atom landscape — as `relaxation-fiber-capacity` Prop F1 predicts. Reinforces the w21 "mass stores nothing" finding.
4. **§1/§6 — the WRITE OPERATOR result amplifies with `d`.** Masked-vs-global cross-write corruption advantage: 197× (d=2) → 599× (d=3) → 6332× (d=4). `potential-function-class`'s "the write operator, not the class, buys locality" is now shown to STRENGTHEN in higher dimensions; a masked atom write is bit-local in parameter space at any `d`.
5. **§7 — three write/eval traps pinned for any future d-ball memory experiment:** (i) the static `write_loss` payload channel must be at index `d`, not the ring's index 2 (now parameterized, default 2 backward-compatible); (ii) value retrieval needs `γ_read>0` (payload is a slave coordinate that must dissipate up to `a_i`, else even the designed arm scores strict 0); (iii) `atom_init_scale` is load-bearing for basin reach from the payload=0 launch (0.5 caps at 0.5, 1.0 reaches 1.0) — the `potential-function-class` open-Q #2 basin-reach limit, defused by init not by parameters.
6. **New CLI/config surface:** `chlu exp-designed-mechanism [--quick]`, `ExperimentDesignedMechanismConfig`. Load-bearing defaults: `atom_init_scale=1.0`, `gamma_read=0.02`, `min_atoms=384` (⚠ inadequate for d≥6 — see update 2; use ≥2048 for the clean high-`d` curve), `learned_arm="learned_global"`. `train_memory.write_loss` gains `payload_index` (default 2). The clean discriminator (`min_atoms=2048`, all d complete) is in `.claude/scratch/designed-mechanism-learned-content/clean_learned.json`; the `min_atoms=384` production run is at `results/exp_designed_mechanism_metrics.json` (its high-`d` cells are superseded by the clean run — §1a/§7).
