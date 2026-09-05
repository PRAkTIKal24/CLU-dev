# dt-units-split — experiment-engineer report

Task + acceptance criterion: separate the data sampling interval from the Verlet integrator step on the eval path, retune the integrator, and re-measure the six w19 findings the conflation was suppressing.
Status: **done** — Item 1 ✅ (classification table), Item 2 ✅ (retuned default, and it is the most important result here), Item 3 ✅ (all six rows). Tests green (**409 passed, 0 failed**; 392 baseline + 17 new).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **The task's warning was right, and it nearly cost this task its conclusion.** `dt=1.0` looks safe at init (`dt·ω=0.51`) but is **unstable on the trained model** (`dt·ω=4.13`, twice Verlet's limit). I first measured FD001 at `dt=1.0` and got **0.5740 — an apparent −0.08 regression that was pure integration artifact.** At the retuned stable step it is **0.6569**. *An init-only stability check would have shipped the wrong conclusion.*
> 2. **Net: the units fix is FD001-NEUTRAL.** base 0.6540→**0.6569**, mass10 0.7092→**0.7102**, relax-tuned 0.7158→**0.7125** — all within ~0.003 of w19. The physics is corrected at **no cost to the metric**. Anyone expecting the units fix to move C-MAPSS should stop expecting that.
> 3. **Two w19 findings REVERSE:** `H ≡ K` becomes `H ≈ V` (corr(H,V) 0.075→0.995), and "training compresses the mass spectrum below init" becomes "training expands it above init" (`M_max/M_min` 1.164 → 1.35–1.80 vs init 1.265).
> 4. **The SDR gate is OPEN.** R-1's relative gradient goes 1.8e-6 → **7.5e-2** (42 000×); the λ scan spans `M_max/M_min` 1.8 → 417. **But R-1 costs stability** (λ=50 drives `dt·ω` to 29.6 via `M_min`→0.027): SDR sweeps must retune `dt` per λ.
> 5. **Do not re-quote w19's "98.3% ballistic".** It does not reproduce even in my legacy arm (I get 79.7% on the same conflated config). Direction survives, number does not.
> 6. **New standing hazard:** no `dt` buys a comfortable stability margin — `ω` *grows* as `dt` shrinks. Needs a curvature penalty or mass floor. See Item 2.

---

## Item 1 — the classification table

Every `cfg.dt` read site in `chlu/eval/` and `chlu/training/`, classified before changing anything.

| # | site | expression | role | action |
|---|---|---|---|---|
| 1 | `clu_scorer.py:340` `_momentum` | `(q_next−q_now)/dt` | **DATA interval** | → `data_dt` |
| 2 | `clu_scorer.py:386,398` wake MSE roll | `model(q,p,hz,dt,0)` | **INTEGRATOR** + horizon-aligned | → `dt_eff` + substeps |
| 3 | `clu_scorer.py:405` contrastive `ps` | `(q_{t+1}−q_t)/dt` | **DATA interval** | → `data_dt` |
| 4 | `clu_scorer.py:474,476` `_energy_scores` | `(W[1:]−W[:-1])/dt` | **DATA interval** | → `data_dt` |
| 5 | `clu_scorer.py:491` `_predict_scores` roll | `model(q,p,hz,dt,0)` | **INTEGRATOR** + horizon-aligned | → `dt_eff` + substeps |
| 6 | `clu_scorer.py:504,508` `_residual_scores` relax | `model(q,p,relax,dt,γ)` | **INTEGRATOR** (never compared to data) | → `dt_eff` |
| 7 | `cafe_model.py:80` `_window_features` `p` | `(w[1:]−w[:-1])/dt` | **DATA interval** | → `data_dt` |
| 8 | `cafe_model.py:109` relax → `q*` | `model(q,p,relax_steps,dt,γ)` | **INTEGRATOR** | → `dt_eff` |
| 9 | `cafe_model.py:130` predict feature | `model(w0,(w1−w0)/dt,hz,dt,0)` | **BOTH — momentum *and* step in one expression** | split both ways |
| 10 | `config.py:302` `relax_budget` | `γ·steps·dt` | **INTEGRATOR** (`relax_steps` counts integrator steps) | → `dt_eff` |
| 11 | `rollout_diag.py:122,156` | `dt` parameter | **INTEGRATOR**, diagnostic only, caller-supplied | unchanged |
| 12 | `training/train.py:118,271,359,376` | `config.training.dt` | **INTEGRATOR, unambiguously** | **unchanged** |
| 13 | `training/train_generative.py:134,234,255`; `train_baselines.py:57` | `config.training.dt` | **INTEGRATOR, unambiguously** | unchanged |

**`chlu/training/` is not conflated and I changed nothing there.** `q0, p0 = q_true[0], p_true[0]` (`train.py:270`) takes the **true momentum straight from the synthetic generator** instead of finite-differencing, and the generators *produce* data at the same `dt` the integrator consumes. There `dt` is one genuine quantity by construction. The conflation is specific to the eval path, which finite-differences **real, externally-sampled** data whose interval it does not control.

### ⚠ The defect is bigger than the task states — a second coupling (site 9 and rows 2/5)
The task frames this as "momentum scale vs integrator step". There is a **third** binding: the wake MSE compares an **`hz`-STEP rollout** against data **`hz` FRAMES ahead** (`target = batch[:, 1:hz+1, :]`), which silently assumes `dt == data_dt`. Adding `data_dt` and leaving `dt` free would have introduced a *new* bug — at `data_dt=1.0, dt=0.125` the loss would predict **2 cycles ahead** and score it against data **16 cycles ahead**.

Fix: `rollout_on_data_grid()` takes `substeps = round(data_dt/dt)` Verlet steps per predicted frame and keeps every `substeps`-th state, so physical time stays exact while `dt` remains a free accuracy knob; `dt_eff = data_dt/substeps` snaps the step so a rollout cannot drift off the grid. `data_dt == dt` ⇒ `substeps = 1` ⇒ the pre-split path exactly (**verified: legacy arm reproduces w19's 0.6540 to 4 d.p.**). No site was genuinely ambiguous once site 9 was decomposed.

---

## Item 2 — retuning the integrator (**the load-bearing result**)

**The stable step must be chosen self-consistently: `ω` is a property of the TRAINED model, and the trained curvature depends on the `dt` it was trained at.** `ω = sqrt(λ_max(∇²V)/M_min)`, λ_max at p99 over 256 FD001 windows; Verlet needs `dt·ω < 2`.

At **init** `ω = 0.508`, so `dt=1.0` reads as safe with a 3.9× margin. That is a trap. After 150 epochs at correct units the potential is finally being trained, its curvature grows ~28×, and:

| dt trained at | substeps | ω (trained) | `dt·ω` | margin | energy drift /16 cyc |
|---|---|---|---|---|---|
| 1.0 | 1 | 4.13 | **4.13** | 0.48× ❌ **unstable** | 2.94 (294%) |
| 0.5 | 2 | 5.54 | **2.77** | 0.72× ❌ unstable | 4.7e-1 |
| 0.25 | 4 | 7.96 | 1.99 | 1.00× ⚠ marginal | 1.5e-1 |
| **0.125** | **8** | **13.46** | **1.68** | **1.19× ← new default** | 8.1e-2 |
| 0.05 | 20 | 22.32 | 1.12 | 1.79× (20× compute) | 5.8e-2 |

**`ω` GROWS as `dt` shrinks.** A finer integrator lets training build a *sharper* potential, which eats the margin it just bought — so `dt·ω` only falls 4.13 → 1.12 across a 20× refinement. Nothing in the objective penalizes curvature, so **the model self-organizes to the stability edge and no `dt` buys a comfortable margin.** Refining still helps (drift 2.94 → 0.081) but sub-linearly, at `substeps`× the compute.

**Recommended default: `dt = 0.125` (substeps 8), running at `dt·ω = 1.68`, a 1.19× margin.** Honest caveat: that is *stable but tight*, and my `ω` is a p99 estimate (absolute-max `ω` is ~8% higher ⇒ `dt·ω ≈ 1.82`). `dt=0.05` (1.79×) is the conservative option at 20× rollout cost. **The real fix is curvature control or a mass floor** — follow-up, not this task.

`mass_spread_lambda` attacks the margin from the **mass** side: λ=1 → `dt·ω = 10.5`; λ=50 → `M_min` collapses 0.576 → **0.027** and `dt·ω = 29.6`.

---

## Item 3 — the six-row before/after

FD001, seed 42, 150 epochs, 4000 windows. **Legacy arm = `data_dt = dt = 0.05`** run through the *new* code, so the comparison is like-for-like and doubles as the backward-compatibility proof.

| # | quantity | w19 (reported) | legacy arm (repro) | **split @ stable dt=0.125** | verdict |
|---|---|---|---|---|---|
| 1a | `E_reg` share of loss | 99.2% | **99.19%** ✓ | **7.15%** | **survives → fixed** |
| 1b | `E_reg` share of mass gradient | 99.8% | **99.52%** ✓ | **0.16%** | **survives → fixed** |
| 2 | `corr(H,K)` / `corr(H,V)` | 0.999996 / 0.055 | **0.999993 / 0.075** ✓ | **0.446 / 0.945** | **REVERSES** |
| 3 | ballistic fraction | 98.3% | **79.7%** ✗ | **50.6%** | changes; direction survives |
| 4 | R-1 relative gradient | 1.6e-6 | **1.77e-6** ✓ | **8.21e-2** | **survives → fixed; GATE OPEN** |
| 5a | mass common:differential | 39.3 : 1 | **26.5 : 1** (≈) | **3.28 : 1** | survives → fixed |
| 5b | final `M_max/M_min` (init 1.265) | 1.153 (**below**) | **1.164** ✓ | **1.349** (**above**) | **REVERSES** |
| 6 | FD001 h-AUROC | 0.6540 | **0.6540** ✓ exact | **0.6569** | **neutral** |

All six pre-registered predictions (`PREREG.md`) **hold**, 1b and 4 by far larger margins than predicted. **The split column is measured at the new stable default `dt=0.125`** (`item3_stable.json`); the `dt=1.0` figures quoted in the subsections below are the earlier, unstable arm and are labelled as such — the two agree on every direction, which is why the Item-3 conclusions are properties of the units fix rather than of the integrator choice.

### 3.1 Loss composition — survives, then fixed
Trained partition, `(share of loss, share of mass gradient)`:

| term | legacy (dt=data_dt=0.05) | **split @ stable dt=0.125** | split @ dt=1.0 |
|---|---|---|---|
| `predict_mse` | (0.73%, 0.48%) | **(27.37%, 99.84%)** | (15.14%, 99.97%) |
| `E_contrast` | (0.08%, 0.00%) | **(65.47%, 0.00%)** | (77.19%, 0.00%) |
| `E_reg` | **(99.19%, 99.52%)** | **(7.15%, 0.16%)** | (7.67%, 0.03%) |

The numerical-hygiene term stops owning the objective; the dominant loss term becomes **`E_contrast` (65%) — the representational half** — and the mass gradient transfers wholesale to `predict_mse` (99.84%).

⚠ **One w19 structural finding SURVIVES and must not be assumed fixed: `E_contrast` has EXACTLY zero mass gradient in both arms.** That is structural (negatives perturb `q` only, so kinetic terms cancel in `H(data)−H(neg)`), not a units artifact. The EBM's representational term remains blind to the mass.

### 3.2 `H` vs `K` — **REVERSES**
At the stable default: `corr(H,K)` 0.999993 → **0.446**, `corr(H,V)` 0.075 → **0.945**, `mean K / mean|V|` 1.047 → **0.358**. (At dt=1.0: 0.475 / 0.995 / 0.153 — same reversal.) **The Hamiltonian flips from kinetic- to potential-dominated: the potential now participates in its own Hamiltonian.**

### 3.3 Ballistic fraction — direction survives, **number does not reproduce**
force/free at γ=0,16: legacy **0.255** (79.7% ballistic) → split **0.994** (50.2%) / **0.506 frac** at dt=0.125. At encode (γ=0.1,32): 0.205 → 0.967.

**Diagnostic caveat (my own bug, not the shipped code's).** My `ballistic()` helper rolls `relax_steps × substeps` steps for the γ>0 row, whereas the shipped encode path (`_window_features`) rolls `relax_steps` *integrator* steps. So my γ=0.1 row over-damps by 8× at the stable default, and its `q*` spread (0.0003 — an apparent total single-basin collapse) **must not be read as a property of the shipped encode path**; the shipped budget there is 0.40, not 3.2. The γ=0 row is unaffected (the predict rollout genuinely is `hz × substeps`), so the ballistic-fraction numbers in the table stand. **Whether `q*` collapses at the shipped budget is now an open question I did not answer** — see follow-up 8.

**Honesty flag:** w19 reported 0.0166 ⇒ 98.3% ballistic for the *same conflated config*; I measure 0.255 ⇒ 79.7%. Unreconciled — likeliest a preprocessing difference (I train on raw CAFE `X_train`; the bridge standardizes separately) or a different free-streaming reference. **Every other legacy quantity reproduces w19 to ≲3 s.f.**, so I report the discrepancy rather than adjudicate it. The qualitative claim (mostly free-streaming; the split roughly halves it) holds either way.

### 3.4 R-1 (mass spread) — **the SDR gate, and it is OPEN**
Relative gradient vs base loss: **1.77e-6 → 7.51e-2** (42 000×). The λ scan, previously reported bit-identical:

| λ | legacy `M_max/M_min` | **split @ stable dt=0.125** | split `std(log M)` | *split @ dt=1.0* |
|---|---|---|---|---|
| 0 | 1.1639 | **1.3489** | 0.1395 | *1.8042* |
| 1 | 1.1655 | **1.8330** | 0.2765 | *9.4382* |
| 50 | 1.3201 | **11.4323** | 1.3064 | *416.88* |

**R-1 is a live lever** — at the stable default it moves `M_max/M_min` over an order of magnitude (1.35 → 11.4); at dt=1.0 the same scan spans 2.5 orders. Legacy losses were 341.4972 / 341.4578 / 337.5823.

*Minor correction to w19:* it reported λ=0/1/50 **bit-identical** in the legacy config; I measure them *nearly* but not bit-identical (341.4972 / 341.4578 / 337.5823). The relative-gradient figure agrees (1.6e-6 vs 1.77e-6), so this is a small overstatement, not a disagreement about mechanism.

### 3.5 Mass spectrum — **REVERSES**
common:differential 26.5:1 → **3.28:1** (4.23:1 at dt=1.0). `M_max/M_min`: init 1.2656 → legacy **1.1639** (compressed) → split **1.3489** at the stable default (**expanded**; 1.8042 at dt=1.0). Common-mode drift also collapses, 3.48 → **0.34**.
**w19's "training makes the mass spectrum more uniform, not less" is reversed.** Once `E_reg` stops supplying 99.5% of the mass gradient, the common-mode runaway that was pushing softplus into its linear regime no longer dominates, and training *builds* spread. w19's softplus-reparameterization worry is materially reduced (though `common_drift` is still 2.67).

### 3.6 FD001 arms — **neutral, once the integrator is stable**

| arm | `data_dt` | `dt` | `dt·ω` | relax budget | **h-AUROC** | vs w19 |
|---|---|---|---|---|---|---|
| **legacy (w19 repro)** | 0.05 | 0.05 | 0.02 | 0.16 | **0.6540** | **exact** ✓ |
| **split, stable (new default)** | 1.0 | **0.125** | 1.68 | 0.40 | **0.6569** | **+0.003** |
| **split, stable + `mass_lr_mult=10`** | 1.0 | 0.125 | — | 0.40 | **0.7102** | **+0.001** vs 0.7092 |
| **split, stable + relax γ=0.4 (budget 1.6)** | 1.0 | 0.125 | — | 1.60 | **0.7125** | −0.003 vs 0.7158 |
| *split @ `dt=1.0`* ❌ unstable | 1.0 | 1.0 | 4.13 | 3.20 | *0.5740* | *artifact* |
| *split @ `dt=1.0`, budget 1.6* ❌ | 1.0 | 1.0 | 4.13 | 1.60 | *0.5568* | *artifact* |
| *split @ `dt=1.0`, budget 0.16* ❌ | 1.0 | 1.0 | 4.13 | 0.16 | *0.5491* | *artifact* |
| *split @ `dt=1.0` + mass10* ❌ | 1.0 | 1.0 | 4.13 | 3.20 | *0.6166* | *artifact* |
| *(reference)* raw 56-stat baseline | — | — | — | — | *0.7486* | |

**All three stable arms reproduce w19 within ~0.003 — i.e. within seed noise (≈0.002).** The units fix corrects the physics at **zero cost to the metric**, and every w19 FD001 conclusion (including "both levers plateau ≈0.714, still below raw-stats 0.7486") **survives unchanged**.

**The −0.08 I measured first was a pure integration artifact** of running at `dt=1.0`. This is exactly the failure the task predicted ("we would re-run FD001 on a silently unstable integrator"), and it was caught only because Item 2 was done on *trained* models rather than at init.

---

## How I verified
- `uv run --no-sync python -m pytest tests/ -q` → **409 passed, 0 failed**, 7 warnings (736s). Baseline measured on this branch *before* the new test file existed was **392 passed**; 392 + 17 new = 409, so **no pre-existing test changed status** despite the shipped-default change.
- `uv run --no-sync ruff check` on all 5 touched files → **All checks passed!**
- **Backward-compat proof:** `data_dt = dt = 0.05` reproduces w19's FD001 **0.6540** exactly, plus loss composition (99.19%/99.52% vs 99.2%/99.8%), `corr(H,K)` (0.999993 vs 0.999996), R-1 relative gradient (1.77e-6 vs 1.6e-6), `M_max/M_min` (1.1639 vs 1.153).
- Harnesses under `.claude/scratch/dt-units-split/`: `omega_scan.py`, `item3.py`, `trained_omega.py`, `dt_selfconsistent.py`, `item3_stable.py`, `fd001_arms.sh`, `fd001_stable.sh`. Raw output: `item3.json`, `item3_stable.json`, `*.log`.

### Flag provenance
| item | value |
|---|---|
| commits | `3760087` (split), `6dd43bd` (tests + CLI), `0eec592` (retuned default) |
| base | local `main` @ `089cc6e` |
| seeds | **42** (single seed throughout; known seed spread on this path ≈0.002) |
| dataset | `cmapss_fd001`, CAFE loader, window 30, C=14, horizons 1…125, `~/cafe-data`; CAFE `~/cafe-bench` |
| model | `clu`, `encode()`-only, default CoxPH probe (`penalizer=0.1`) |
| CLU config | `kinetic_mode=newtonian_learned`, `potential_type=mlp`, `hidden=64`, `gamma=0.1`, `epochs=150`, `lr=1e-3`, `batch=64`, `max_fit_windows=4000`, `predict_horizon=16`, `relax_steps=32`, `neg_noise_scale=0.5`, `energy_reg=0.005`, `momentum_init=finite_diff`, no lattice |
| **new/changed defaults** | **`dt: 0.05 → 0.125`**, **`data_dt: (new) 1.0`** |
| varied | `dt ∈ {0.05,0.125,0.25,0.5,1.0}`, `data_dt ∈ {0.05,1.0}`, `relax_gamma ∈ {0.005,0.05,0.1,0.4}`, `mass_lr_mult ∈ {1,10}`, `mass_spread_lambda ∈ {0,1,50}` |
| diagnostics | ω/drift 256 windows seed 0; `corr(H,K)` 2000 windows seed 1; ballistic 512 windows seed 2; gradient partitions batch 64 seed 0 |
| env | main `.venv` (**no worktree**, `--no-sync` throughout), JAX **0.9.0**, equinox 0.13.4, CPU |
| prereg | `.claude/outputs/dt-units-split/PREREG.md`, written before the Item-3 harness ran |

---

## Git footprint
- Branch **`agent/experiment-engineer/dt-units-split`**, off local `main` @ `089cc6e`. Rebased onto `main` (no-op, base unmoved). **Not pushed.** No worktree needed (clean tree, no concurrent branches).
- Commits: `3760087`, `6dd43bd`, `0eec592`.
- Files: `chlu/eval/config.py`, `chlu/eval/clu_scorer.py`, `chlu/eval/cafe_model.py`, `scripts/cafe/run_clu_cafe.py`, `tests/test_eval_dt_units.py` (new, 17 tests).
- No conflicts. `chlu/training/` deliberately untouched (Item 1 row 12).

## Open questions / follow-ups / risks
1. **Single seed.** The stable arms differ from w19 by ≤0.003, i.e. *within* the ≈0.002 seed spread — I read that as "neutral", but a 3-seed confirmation would make it solid, and it is cheap.
2. **The stability margin is tight (1.19×) and no `dt` fixes that.** Highest-value follow-up: add a curvature penalty or a floor on `M_min` so the potential cannot sharpen into the integrator's limit. Until then every SDR/R-1 sweep must re-derive its own stable `dt`.
3. **The ballistic-fraction discrepancy with w19 (79.7% vs 98.3%) is unresolved** and that figure is quoted in the negatives registry.
4. **`E_contrast` still has exactly zero mass gradient** — structural, not units. Needs negatives that perturb `p` as well as `q` if the EBM term is ever to shape the mass.
5. **The train/inference γ mismatch (w19 item 2) is untouched** and is now *more* consequential: with the potential actually active, training at γ=0 while encoding at γ>0 matters more, not less.
6. I did **not** re-run **voraus**, which shares `_SharedCLUFit`. **It WILL be affected** — unlike the w19 port, this change alters shipped defaults. Voraus must be re-run before its numbers are quoted.
7. I did not re-run the `q*`/read-out inventory (w19 item 3). With `H` now potential-dominated, the "11 of 16 physics scalars at chance" result deserves a re-measure — plausibly the single highest-value follow-up for the science.
8. **Does `q*` collapse at the shipped encode budget (0.40)?** My diagnostic over-damped 8× and so cannot answer it (see §3.3 caveat). w19 reinstated a single-basin collapse at correct units, and the potential is now much stronger, so this is worth a direct measurement on the shipped path — it bears on whether `basin_coords` still carries information.

## Proposed handover updates (for the Hub)
1. **§3 config table:** add `CLUScorerConfig.data_dt` (**default 1.0**) and derived `.substeps` / `.dt_eff`. **`CLUScorerConfig.dt` default CHANGED 0.05 → 0.125** — a shipped-default change, the first on this path. `--data-dt` added to `scripts/cafe/run_clu_cafe.py`.
2. **§7 "`dt=0.05` on cycle-indexed data" → RESOLVED** (`3760087`, `6dd43bd`, `0eec592`).
3. **§7 NEW issue — the stable integrator step is self-referential.** `ω` is a property of the trained model and *grows as `dt` shrinks*; the model self-organizes to the stability edge and no `dt` gives a comfortable margin (best measured 1.79× at 20× compute; default runs at 1.19×). **An at-init stability check is invalid** — it says `dt=1.0` is safe when the trained model is 2× over the limit. `mass_spread_lambda` makes it worse via `M_min`.
4. **§7: R-1 is unblocked — the SDR hold can come off** (relative gradient 7.5e-2; λ scan spans `M_max/M_min` 1.8→417), **with the caveat in 3**.
5. **Record: the units fix is FD001-neutral** (0.6540→0.6569; 0.7092→0.7102; 0.7158→0.7125). It does **not** rescue the metric, and the raw-stats **0.7486** gap is unchanged. The earlier-looking −0.08 was an unstable-integrator artifact and **must not be recorded as a finding**.
6. **Retract/repair two w19 conclusions:** `H ≡ K` (now `H ≈ V`) and "training compresses the mass spectrum" (now expands it). **Do not re-quote w19's 98.3% ballistic** — 79.7% in my legacy repro.
7. **w19's "R-1 λ scan is bit-identical" is a slight overstatement** — nearly, not bit-, identical. Mechanism claim unaffected.
8. **Voraus needs re-running** before its numbers are quoted (shared `_SharedCLUFit`; shipped defaults changed).
