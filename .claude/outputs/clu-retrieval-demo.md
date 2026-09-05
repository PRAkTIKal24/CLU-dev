# clu-retrieval-demo — experiment-engineer report

**Task + acceptance criterion:** hand-build the write→address→retrieve loop and measure whether the *physics* supports selective, durable, linearly-readable storage — with a definitive answer on mass-as-address-key and one datapoint on address learnability.
**Status: done.** The loop **works** (100% selective at 2 items, 99.2% at 8, blank control at chance). **Two of the vision's load-bearing claims do not survive.**

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary).** Three live sites assert things this run contradicts:
> 1. Handover §vision item (4): *"CLU learns the MASSES and INITIAL LATENTS required to access each stored piece later"* — **address learning by gradient descent succeeds 4.2% of the time (1/24), and the mechanism says it cannot do better.** Needs re-wording or an explicit "unsolved" flag.
> 2. Handover vision table row *"masses as access keys | M as budget allocator (Thread-5)"* — needs the qualifier **1.58 bits / 3 items, non-monotone**.
> 3. The Hub's formalization *"the address is the pair (m, q₀, p₀)"* — **m contributes ~1.6 bits and only in a chaotic band; q₀ carries the addressing.** Suggest demoting m from the address tuple pending evidence.
> The theorist's parallel `clu-memory-architecture` task is the natural consumer.

---

## Flag-provenance table

| item | value |
|---|---|
| commit | `43e4aaa` (branch `agent/experiment-engineer/clu-retrieval-demo`, base `main` @ `1e7ace5`) |
| seed | `project.seed = 42` (single seed — see risks) |
| kinetic mode | **`newtonian_learned`** (NOT the Exp-A default `newtonian_identity`, which ignores `log_mass` entirely — the mass question is unaskable there) |
| potential | **hand-designed** `RingRegisterPotential` / `ThreeModePotential`; **no training anywhere** except GD *on the address* in item 5 |
| γ (friction) | **0.02** (non-zero is REQUIRED — see §6) · dt `0.05` · steps `1200` |
| landscape | λ=1.0, f=1.0, barrier **0.2**, payload_kappa=1.0, bump_width=0.05, payload_seed=0 |
| queries | 64/item, σ_θ=0.15, σ_r=0.05, σ_p=0.05; `q₀[2]=p₀[2]=0` always |
| read | tail 25%, 8 subsample points; probes: ridge-one-hot, **linear codebook**, nearest-centroid |
| address GD | Adam, lr 0.05, 300 steps, offsets {1,2,4}, K=8 |
| langevin_noise | **N/A** — deterministic Verlet only, no Langevin, no temperature |
| JAX | 0.9.0 (main venv reused per protocol §4, not a worktree sync) |

**Designed vs learned:** *everything* about the landscape and the addresses is designed. The only gradient descent is on the address (item 5). No claim here is evidence of emergence.

---

## Results

### 1 ⭐ The minimal demo — IT WORKS
| read | written | blank control | chance |
|---|---|---|---|
| payload-only, linear codebook | **1.000** | 0.469 | 0.500 |
| payload-only, nearest-centroid | **1.000** | — | 0.500 |
| payload-only, ridge-one-hot | **1.000** | 0.469 | 0.500 |
| full-state | 1.000 | **1.000** ⚠ | 0.500 |

Payload retrieved to **9.98e-4** absolute error. Confusion matrix perfectly diagonal `[[34,0],[0,30]]`.

⚠ **The full-state read scores 1.000 in the BLANK landscape** — it is reading the address back, not the memory. This is why the payload-only read exists; **any future retrieval claim that reads the address plane is decorative.** The blank control is the load-bearing number in this table.

**Durability:** payload-only accuracy is **1.000 at every read position** (steps 60/120/300/600/900/1188) — retrieval does not decay, it is a fixed point. At **γ=0 it degrades to 0.813**: the conservative particle never settles, so the read is unstable. *Retrieval requires dissipation.*

### 2 ⭐ MASS AS AN ADDRESS KEY — **it fails in the strong form. 1.58 bits.**
Fixed `q₀=(1,0,0)`, `p₀=(0.5,0.5,0)`, sweeping mass over 3 decades:

| measure | value |
|---|---|
| distinct items, scalar-mass sweep (raw) | 6 of 8 (2.58 bits) |
| distinct items, (m₀,m₁) vector sweep (raw) | 8 of 8 (3.00 bits) |
| **distinct items ROBUST to a 1% mass error** | **3 of 8 → 1.58 bits** |
| monotonicity corr(−log m, θ_landed) | **−0.07** (none) |

**The raw counts are an illusion.** Robustness detail (fraction retrieving the same item under ±1% mass jitter):

| m | 0.032 | 0.042 | 0.056 | 0.075 | 0.100 | 0.133 | 0.178 | 0.237 | ≥0.316 |
|---|---|---|---|---|---|---|---|---|---|
| item | 7 | 0 | 5 | 6 | 4 | 4 | 1 | 1 | 0 (all) |
| frac same | 0.38 | 0.25 | **0.00** | 0.38 | 1.00 | 1.00 | 1.00 | 0.88 | 1.00 |

The mass axis has exactly three regimes: a **chaotic band** (m ≲ 0.08) where 1% mass error lands anywhere — *the band containing most of the "distinct items"*; a **narrow usable window** (m ∈ [0.1, 0.24]) giving 2 non-trivial items; and a **dead plateau** (m ≳ 0.3, a 100× range) that always returns item 0 because the particle lacks the energy to leave its launch well.

⇒ **"Select the appropriate mass particle" cannot carry addressing.** Mass is an energy dial with ~1.6 usable bits, non-monotone, and its high-resolution region is chaotic. **This lands on top of v5-gate's independently measured 1–1.6 bit register capacity** — from completely different physics, which makes the coincidence worth the theorist's attention.

### 3 The three write modes — all three realized, cleanly
| quantity | measured | note |
|---|---|---|
| permanent item (channel angle) drift | **4.17e-7** | exact designed SO(2) ⇒ torque-free |
| **corruption of permanent by the decaying write** | **4.17e-7** | **no corruption** |
| decaying item half-life (envelope fit) | **66.87 steps** | vs predicted `2ln2/γ` = **69.31** (−3.5%) |
| half-life, sliding-window cross-check | 75 steps | consistent |
| μ²_rad | 8.0 | |
| uncorrelated item (broken-symmetry sign) | retained | |

**Write locality is DESIGNED** (additive separability + exact symmetry), not emergent. An MLP potential has no locality by default — this is the structured potential the task permits, and the zero-corruption result is a property of the design, not a discovery.

### 4 Interference — **practical ceiling 8 items (3.0 bits)**
| K | codebook | centroid | one-hot | payload R² | blank | **landed in correct well** | σ_θ/spacing |
|---|---|---|---|---|---|---|---|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.469 | 1.000 | 0.048 |
| 4 | 1.000 | 1.000 | 0.484 | 1.000 | 0.242 | 1.000 | 0.095 |
| 8 | **0.992** | 0.992 | 0.305 | 0.971 | 0.125 | 0.994 | 0.191 |
| 16 | 0.190 | 0.258 | 0.219 | 0.539 | 0.059 | **0.764** | 0.382 |
| 32 | 0.044 | 0.096 | 0.076 | 0.151 | 0.030 | **0.447** | 0.764 |

**The failure is ADDRESSING, not storage.** `frac landed in correct well` collapses in lockstep with accuracy, and the break sits where **σ_θ/spacing crosses ≈0.2** — i.e. sites must be ≳5 query-noise-σ apart. That gives a capacity law

> **K_max ≈ 0.2 · 2π / σ_θ** → 8.4 for σ_θ=0.15, **measured ceiling 8.**

Capacity is set by *query precision vs site spacing*, exactly like a physical addressable memory. It is **not** a limit of what `V` can hold — the payload regression R² is 0.971 at K=8 where addressing still works.
⚠ **One-hot probe accuracy in this table is estimator-limited, not physics** (0.484 at K=4 while payload error was 7e-4). See §"what I fixed".

### 5a ⭐ THE PRIMARY NUMBER — address restructuring **fails: 4.2%**
| | plain GD | γ-annealed GD |
|---|---|---|
| success rate (24 trials, K=8) | **0.042 (1/24)** | **0.042 (1/24)** |
| adjacent-site init (offset 1) | 0.125 (1/8) | 0.125 |
| offset 2 | 0.000 | — |
| **antipodal (offset 4)** | **0.000** | — |
| median steps when it succeeds | 233 | — |

A deliberately bad address is **not** restructured into a working one by descending a retrieval loss — even in the Head's weak, iterative, gauge-loose formulation, and even with the γ-anneal repair that the §5b diagnosis implies.

### 5b Smoothness — **the gradient VANISHES, it does not explode** (pre-registration falsified)
| rollout steps | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| ‖∇_address loss‖ | 7.7e-2 | 1.9e-1 | 8.9e-2 | 5.9e-2 | 5.0e-3 | **2.9e-4** |

Loss-vs-θ₀ surface: **8 local minima at K=8** (one per well), **cliff ratio 1.7e7** (max step / median step). It is a **staircase**: flat plateaus separated by discontinuities.

---

## 6 ⭐ WHY IT FAILS — the finding, and it is structural

**The γ-scan puts both sides of the tension on one axis** (same launch point, same loss):

| γ | 0.0 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|---|
| ‖∇_address‖ | **3.3e-1** | 9.1e-2 | 1.9e-2 | 1.7e-3 | 1.6e-5 | 5.0e-8 | 1.2e-7 |
| settling error (read quality) | **4.0e-1** | 1.2e-1 | 1.7e-2 | 2.7e-3 | 3.7e-3 | 3.7e-3 | 3.7e-3 |

**Retrieval and learnability are governed by the same knob, in opposite directions, and there is no window where both are good.** The address gradient falls **7 orders of magnitude** across the scan while the read only becomes usable at γ ≳ 0.01.

The mechanism is not tunable, it is definitional:
- A **readable** memory requires the retrieved state to be a **fixed point** — the same item must come back regardless of small query variation. That is exactly `∂(final state)/∂(q₀) → 0` **inside a basin**.
- **The address gradient IS that derivative.** Making retrieval robust is precisely the act of destroying the signal that gradient descent on the address would need.
- Hence the staircase: zero gradient inside each basin, a discontinuity at each separatrix. Gradient descent on a staircase does nothing — which is the 4.2%, and why γ-annealing (0.001→0.02) changed **nothing**.

**Which stage failed:** not the write (payload stored and read to 1e-3), not the read (linear probe, 100%/99.2%), not durability (flat over 1200 steps), not locality (corruption 4e-7). **The ADDRESS SELECTOR failed** — both of its proposed keys. Mass gives 1.58 robust bits; `q₀` gives a gradient-free staircase.

**Smallest changes that could make it work** (ranked; none tested beyond the γ-anneal, which failed):
1. **Do not learn the address by backprop through the rollout.** Learn a *feed-forward* map query→address (an encoder trained on (item, address) pairs where the address is *assigned*, not discovered). The Head's "allow arbitrary choices where they don't affect performance" already licenses fixing the assignment; T3 says the regularizer cannot choose it anyway. This sidesteps the staircase entirely and is the change I'd back.
2. **Make the loss depend on the whole trajectory at γ≈0**, then read at γ>0 — the γ=0 gradient is 3.3e-1, four orders larger. My γ-anneal did this crudely (annealing γ *inside* one loss) and failed; the untested version keeps a *separate* informative low-γ search objective from the high-γ read.
3. **Soften the basins** (small `barrier`) so the address landscape is convex — but at barrier=0.05 the well could not hold a jittered query at all (measured payload error 0.18 at K=8). This trade may have no feasible window; worth one cheap sweep.
4. **Abandon continuous address search** for content-addressable relaxation (drop the query *into* the landscape and let it settle — Hopfield-style), which is what the physics actually does well.

---

## PREREG scorecard (honest; `PREREG.md` written before any measurement)

| # | prediction | measured | verdict |
|---|---|---|---|
| P1 | ≥95% at 2 items, blank ≤60% | 1.000 / 0.469 | ✅ |
| P1 | flat survival at γ>0, degrades at γ=0 | 1.000 flat / 0.813 | ✅ |
| P2 | mass strong form FAILS, ≤3 robust items | **3 items, 1.58 bits** | ✅ |
| P2 | *mechanism*: monotone energy dial, sign-cone-restricted | corr = **−0.07**, NOT monotone; wrapping reaches the whole ring | ❌ **mechanism falsified** — right number, wrong reason |
| P3 | permanent drift ≤1e-6; corruption ≤1e-6 | 4.17e-7 / 4.17e-7 | ✅ |
| P3 | half-life `2ln2/γ` | 66.87 vs 69.31 (−3.5%) | ✅ (my numeric "139±40" was quoted at γ=0.01; the run is γ=0.02) |
| P4 | ≥95% at K=2,4 | 1.000, 1.000 | ✅ |
| P4 | **<90% at K=8** | **0.992** | ❌ **too pessimistic** |
| P4 | <70% at K=16; ceiling 4–8 | 0.190; ceiling **8** | ✅ |
| P5a | ≤30% at K=8; ~0% antipodal; O(100) steps | 4.2%; 0%; 233 | ✅ |
| P5a | ≥50% at K=4 | **not run** | ⚠ unmeasured |
| P5b | piecewise-smooth with cliffs | 8 minima, cliff ratio 1.7e7 | ✅ |
| P5b | **gradient norm GROWS with rollout length** | **falls 2.6 orders** | ❌ **falsified — and this is the wave's finding** |

Two falsifications, one of them (P5b) the mechanism behind the headline negative. I predicted exploding gradients from non-contraction; the truth is the opposite, and the opposite is worse: **friction contracts, and contraction is what kills addressability.**

---

## How I verified
- `uv`-equivalent: `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …` (main venv reused per protocol §4 w6 lesson; JAX **0.9.0**, no worktree sync).
- **Full suite: `383 passed` in 278.88s** (370 baseline + 13 new). `ruff check chlu/ tests/` → **All checks passed**.
- `python -m chlu.experiments.exp_retrieval` (full battery) and the CLI path `chlu exp-retrieval --quick` both exit 0.
- Every number above is from `results/exp_retrieval_metrics.json`, copied to `.claude/outputs/clu-retrieval-demo/` with the 3 figures.

### What I fixed mid-run (would have produced wrong conclusions)
1. **`barrier=0.05` could not hold a jittered query** (payload error 0.18 at K=8 vs 0.002 at 0.2). The first pass would have reported an interference ceiling of 2 items that was really a settling failure. Default is now 0.2, with the measurement in the config comment.
2. **The one-hot linear probe is estimator-limited on a 1-D non-monotone code** — 0.484 at K=4 while the payload itself was retrieved to 7e-4. I nearly reported a decoder artifact as interference. Added `linear_codebook_read` (linear regression onto the stored value + codebook decode) and nearest-centroid; ceiling now uses the codebook read. Regression test asserts one-hot *still fails* so this cannot silently flip.
3. **Half-life was measuring the first zero crossing of an oscillating |r−f|** (6 steps) instead of the envelope (66.9). Now an envelope fit + window cross-check, with a test pinning `2ln2/γ`.
4. **`grad V` = NaN at the origin** — the radial envelope multiplies the arctan2 singularity by zero, and `NaN*0 = NaN`. Masked with `jnp.where` before `arctan2`. Latent NaN-gradient bug for any trajectory reaching q=0.
5. **`save_config` silently dropped the new config group** (explicit per-group dict). Fixed. Caught by my own round-trip test; I then **verified that the existing `test_every_group_round_trips_mutated` also catches it** (reverted the fix → that test fails). Two independent guards, both working.

---

## Git footprint
- **Branch** `agent/experiment-engineer/clu-retrieval-demo`, off local `main` @ `1e7ace5` (rebase onto `main` = no-op; did **not** touch `origin/main`, §7.21).
- Worked in a **dedicated worktree** `../CHLU-clu-retrieval-demo`: the main checkout had another agent's uncommitted changes (`chlu/eval/clu_scorer.py`, `chlu/eval/config.py` — `clu-latent-io-audit`). **No collision.** Branch refs verified from the main repo before reporting (w4 lesson).
- Commits: `dcd17c9` (designed potentials), `e58181f` (experiment + config + CLI), `43e4aaa` (tests).
- Files: **+** `chlu/core/memory_potentials.py`, `chlu/experiments/exp_retrieval.py`, `tests/test_retrieval.py`; **M** `chlu/config.py` (new group + `save_config`/`load_config` entries), `chlu/cli/experiment_cmd.py` (import, parser, `cmd_exp_retrieval`). Did not touch `utils/plotting.py` (shared) — figures are local, per the `exp_paid_access` precedent. `results/` deliberately **not** committed.
- Worktree left in place for review; remove with `git worktree remove ../CHLU-clu-retrieval-demo`.

## Open questions / risks
- **Single seed (42), single geometry.** The 4.2% and 1.58-bit numbers are one draw. The *mechanism* (γ-scan) is seed-independent, but the exact rates need replication before they go in a paper.
- **2-D address plane only.** The K_max ≈ 0.2·2π/σ_θ law is a statement about a ring. A d-dimensional address space should scale far better (~(1/σ)^d sites) — **this is the single most valuable cheap follow-up**, because it decides whether "handful of items" is a property of the ring or of the primitive. My ceiling of 8 should **not** be quoted as CLU's capacity without it.
- Mass was swept **globally and per-axis on the address plane only**; a full learned diagonal `M` over a higher-dim latent is untested.
- Repair options 1/3/4 in §6 are **untested**; only the γ-anneal was run, and it failed.
- `results/` is untracked but **not gitignored** in this repo — other experiments write there too. Possible housekeeping item.

## Proposed handover updates (for the Hub)
1. **§3 / config-authoring note (NOT a defect):** a new config group must be added in **three** places — `CHLUConfig`, `load_config`, **and `save_config`** (all three enumerate groups explicitly). I missed `save_config` initially. ⚠ **Correction to my own first draft of this report:** I claimed the exhaustive `test_every_group_round_trips_mutated` fails to catch the omission. **That is false — I verified it by reverting the fix and the test FAILS as designed** (`tests/test_config.py:106`). The existing guard works; no §7 issue is needed, just the three-site note. (Recording the retraction because an unverified "the test doesn't catch it" claim would have invited someone to build a redundant guard.)
2. **§7 / numerics:** multiplying an `arctan2` singularity by a vanishing envelope does **not** remove a NaN gradient (`NaN*0=NaN`). Mask the argument with `jnp.where`. Relevant to any angular/coset potential.
3. **§6 ground truth — new entry:** the write→address→retrieve loop **has now been run**. Write/read/durability/locality all work under design; **addressing does not**. Add the γ-scan tension as a named result: *retrieval robustness and address learnability are the same derivative with opposite sign requirements.*
4. **Vision table amendments** — the three reconciliation items in the header block.
5. **New CLI/config surface:** `chlu exp-retrieval`, `ExperimentRetrievalConfig` (defaults chosen from measurement; `barrier=0.2` and `gamma=0.02` are load-bearing, not arbitrary).
6. **Suggest for the theorist (`clu-memory-architecture`):** the independent arrival at ~1.6 bits from ring-mass chaos vs v5-gate's washboard-minima counting may not be a coincidence; and the staircase result is a candidate *no-go* — "a dynamically-robust addressable memory has no useful address gradient" — worth trying to state as a proposition.
