# full-clu-harness — experiment-engineer report

Task + acceptance criterion: build the C2W1 synthesis harness (learned `V_θ` store · staged levers · 13 loud monitors · controller v0) and run **a nontrivial write/read stream without tripping a silent-collapse mode**; acceptance is *"does not collapse"*, not *"wins"*.
Status: **done.** Acceptance **MET, with a stated scope**: the full stream (10 offers → 1 refusal, 4 evictions, 1 deletion, 6 live) runs with **exactly one monitor tripped — #9 lifetimes, which my own PREREG declared known-uncleanable-by-any-verb** (its fix is C1W27's gated stiffness, which C2 must not build). The best configuration found (`R3`: the `anneal` verb + 2× read budget) trips **only #2**, the dividend monitor, which by design has **no verb** and escalates. ⛔ The **hard falsifier did NOT fire, but it came close on one axis and that is the headline finding**: the clean stages are the ones with well-separated basins; when separability is removed (S4) the CLU read falls to **0.479** while its own settle-deleted launder holds **0.854**.

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5 corollary, first-10-lines rule)
> **R1 — `controller-doctrine`'s Prop D1 bound `D ≤ U` is VIOLATED in the full system under the cheap `sep/2` inradius proxy.** Measured `D/U` = **1.50 / 1.50 / 2.50 / 2.00** at S0/S1/S2/S3 (the doctrine's own §2b guard says the proxy over-claims by "up to 4.8 %"; on a learned `V_θ` it is off by **50–150 %**). Prop D1 itself is not in question — the *proxy for the certified inradius* is. Any use of `r_i ≈ sep/2` as if it were the certificate must carry this. *(Owner: theorist `controller-doctrine` + engineer; monitor #2 reports `prop_D1_holds` per observation so the violation is visible in the artifact.)*
> **R2 — the doctrine's γ band does not transfer.** Row 1's band is `γ ∈ [0.05, 0.5]` at N=400 with `ρ_conv < 1e-6`; this harness's **shipped** read (`γ_read = 0.02`, 800 steps) sits at `ρ_conv = 4.3e-7`, i.e. **within 2.3× of the band edge**, and the annealed read pushes it to `3.6e-6` (**TRIP**). The band's *structure* transfers, its *constants* do not. *(Owner: curator/theorist — scope every γ-band statement by harness.)*
> **R3 — two of four declared read dials are semantically DEAD** (monitor #10 tier b): `gamma_read` ×4 moves the read by **7.3e-4** noise units and `traj_stride` by **3.0e-4**, both far below the 3σ bar, at S0 and S6. ⚠ Tier (a) (the O(1) access-counting config proxy) is **NOT implemented in v0** and says so in the artifact (`knob_tier_a_implemented: false`). *(Owner: engineer, next wave.)*
> **R4 — `eval/dividend.py` is co-owned and its signatures are now frozen**; `memory-gym-v0` lands the gym-side callers. The one semantic decision I made and they inherit: `same_keys_null` = same keys with **permuted payloads** (address structure kept, content destroyed), distinct from `settle_deleted_launder` = arg-min over the store's own keys returning the true payload. *(Owner: Hub, to confirm at review.)*

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — **instrument**. No performance claim, no leaderboard entry.
- **Laundering control:** built, not claimed. Settle-deleted launder + same-keys null + blank/empty store run on **every** stage; the doctrine's second (best-shared-metric) launder and the trajectory launder are implemented in `eval/dividend.py` but **NOT RUN** here (no shared metric is fitted at v0).
- **Falsifies:** a monitor trips and cannot be cleared by returning the responsible lever to its band. **⛔ Hard falsifier:** the only clean configuration is the degenerate one.
- **Does NOT falsify:** ≤0 dividend · slow wall-clock · #7/#13 not being runtime trips · trips on #9 (large excursion) and #2, both pre-declared uncleanable in PREREG §0.

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/full-clu-harness` @ **`c282089`** (rebased onto `main @ e0f06d3`) |
| artifact | `.claude/outputs/full-clu-harness/exp_clu_system_metrics.json` (11 records) · `…_stages.png` · `run_seed0.log` |
| command | `PYTHONPATH=. python -u -m chlu.experiments.exp_clu_system --seed 0` |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0** (no worktree `uv sync`), 1 worktree (`../CHLU-fullclu`) |
| **seeds** | **seed 0, SINGLE SEED.** This is an instrument report, not a paper number; multi-seed is owed before any of it is quoted as a result |
| geometry | `addr_dim (d) = 4`, `payload_dim (m) = 1`, latent `dim = 5`, `ball_radius = 1.0` (S4+: **0.45**) |
| store | learned `V_θ` = `DesignFreedomPotential(rung="free_mlp", family="atoms")`, **n_atoms = 2048** (= `max(32·K, 384, 512·√2^4)`), `atom_init_width 0.3`, `atom_init_scale 1.0`, `atom_depth_init 1e-4`, `confine (α) 0.05`, capacity 8, one atom group per slot |
| write | **masked/local** (C3-local), 300 steps, Adam(W) lr 3e-3, wd 1e-4, `n_perturb 32`, `σ_addr 0.25`, `σ_pay 0.6`, `margin 0.15`, `barrier 0.2`, `barrier_pairs "nn"`, crowd targets = live set |
| read | `dt 0.05`, **`γ_address 0.05`, `γ_read 0.02`**, 400 + 800 steps, `traj_stride 8`, `kinetic_mode newtonian_learned` with **M = I**, `σ_q 0.15`, `payload_tol 0.1`, 8 queries/item, ψ = settled point (S0–S5, R1–R3) / tail-mean trajectory (S6, R4) |
| control | `d_safe = 2·s_max + 2.576·σ_q` (**derived**, doctrine R5) = 0.9864 at s_max 0.3; S4+ `d_safe_override = 0.6·sep_expected` (**deliberately out of band**, see §3); budget 8 (S0/S1) → **6** (S2+); `leak 0.02` (S1+), `amp_floor 0.05`; retry τ 0.8, ≤2 rounds (S5+) |
| langevin / temperature | **N/A** — deterministic, `T = 0`, `p₀ = 0`, no Langevin step anywhere |
| lyapunov / wake–sleep | **N/A** — the write is `train_memory`'s static objective; neither `train.py` nor `train_generative.py` is used |
| N94 maturity | 300 write steps per masked item write ≥ 40 ⇒ **promotable**; monitor #13 carries it per reading |
| stream | 10 offers + 1 **collision offer** (a near-duplicate address, so the gate has something to refuse) + 1 **deletion** at S3+ |
| wall clock | 16–32 s per stage, **≈4.5 min** for all 11 records (7 stages + 4 remediation arms) |

---

## 1. What I did

1. **API FREEZE first (commit #1, `665fcb1`)** — public surface only, so `memory-gym-v0` and `trainability-spike` could branch immediately: `chlu/core/clu_system.py`, `chlu/core/monitors.py`, `chlu/core/clu_controller.py`, `chlu/eval/dividend.py`. ⭐ `read()` returns a **strided trajectory + `q*`**, and the settle exposes `fixed_point_residual()` for the implicit/DEQ attachment.
2. **Implemented the doctrine's 13-monitor table, not the Hub's provisional one.** `controller-doctrine` landed before I ran anything and explicitly assigned me R3/R4/R5; implementing the superseded predicates knowingly would have been wrong. Diff carried in `monitors.py`'s module docstring (2 replaced, 7 sharpened, 4 confirmed) + **M14**.
3. **Controller v0** over `{admit, place, evict, decay, route, retry, stop}` **+ the doctrine's `anneal` and `expand`**, wrapping C1's `Controller` as the address allocator/codebook (zero edits to C1W27 files). Guards are a **projection** (`project()`), never a penalty.
4. **The harness experiment** with 7 cumulative stages and 4 remediation arms, all three harness-native controls on every stage, byte accounting, a canary stream for M14, and a knob-liveness sweep.
5. **Two mechanism findings fixed in-code** (§4), 65 new tests, CLI hook `chlu exp-clu-system`.

## 2. How I verified

| check | command | observed |
|---|---|---|
| reach criterion | `pytest tests/test_monitors.py` | my independent re-implementation of criterion (U) reproduces **all seven** published `a_U` anchors to ≤0.1 % (base **1.0229** vs 1.023; narrow 0.6376 vs 0.638) and `κ_stat` = **3.33 / 4.08 / 4.67 / 6.06** at β = 1e1…1e6 **exactly** |
| margin law | same | `erf(2.576/√2) = 0.9900` — 99 % at **2.576 σ**, not 5 σ (doctrine R1 confirmed in code) |
| mass gauge (#7) | same | Newtonian, **trajectory-wise**: max relative deviation **< 1e-5** under `(M,V,p₀)→(2M,2V,2p₀)` |
| guards | `pytest tests/test_clu_controller.py` | every guard unbypassable: LRU eviction, an energy routing signal, a non-returning anneal schedule and a shrinking expand all **raise**; eviction under a class-I trip and an under-persistent eviction are **refused and logged** |
| the store is not a table | `pytest tests/test_clu_system.py` | corrupting the eval-only payload bookkeeping leaves the read **bit-identical**; masked writes leave other groups **bit-identical**; eviction **re-draws** the freed group |
| full suite | `uv run pytest -q` (main venv) | see §6 |
| lint | `ruff check chlu/` | clean |
| the harness itself | the run above | 11 records, all monitors observed at every stage |

---

## 3. Results

### 3.1 The staged run (seed 0; `decode` = nearest-stored-payload decoding, N110's honest metric)

| stage | live | adm/ref/evict/del | **decode** | settle-deleted launder | same-keys null | blank (bar) | **dividend** | self-probe acq | δ_read | trips |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 baseline | 8 | 10/1/2/0 | 0.9062 | 1.0000 | 0.1250 | 0.125 (0.249) | **−0.0938** | 0.828 | 0.0013 | #5, #9, #10 |
| S1 lifetimes | 8 | 10/1/2/0 | 0.9062 | 1.0000 | 0.1250 | 0.125 (0.249) | −0.0938 | 0.812 | 0.0019 | #5, #9, #12 |
| S2 admission+pressure | 6 | 10/1/4/0 | 0.8958 | 1.0000 | 0.0000 | 0.167 (0.328) | −0.1042 | 0.938 | 0.0014 | **#9 only** |
| **S3 deletion** | 6 | 10/1/4/**1** | **0.9167** | 1.0000 | 0.0000 | 0.167 (0.328) | −0.0833 | 0.938 | 0.0010 | **#9 only** |
| ⛔ S4 basin interaction | 6 | 10/1/4/1 | **0.4792** | **0.8542** | 0.0417 | 0.167 (0.328) | **−0.3750** | 0.542 | 0.0571 | #5, #8, #9, #12 |
| S5 + retry | 6 | 10/1/4/1 | 0.4792 | 0.8542 | 0.0417 | 0.167 (0.328) | −0.3750 | 0.542 | 0.0571 | #5, #8, #9, #12 |
| S6 + trajectory ψ | 6 | 10/1/4/1 | 0.4792 | 0.8542 | 0.0417 | 0.167 (0.328) | −0.3750 | 0.542 | 0.0571 | #5, #8, #9, #10, #12 |

**Acceptance criterion.** S2 and S3 — the stages that carry *real capacity pressure* (10 offers, budget 6, 4 evictions) **and** a *deletion demand* — run the full system with **one** monitor tripped, #9, which PREREG §0 declared known-uncleanable-by-any-verb (doctrine I-14). Every other monitor is clear or inapplicable.

### 3.2 The remediation arms — trips cleared by their designed restoring verb

| arm | verb applied | target | decode | trips after |
|---|---|---|---|---|
| **R1 anneal read** | `anneal([4,2,1])` | #5 addressing | **1.0000** (from 0.9062) | #1, #2 |
| R2 place pass | `place` (relaxation re-derivation, λ_min>0 + injectivity) | #8-N3 | 0.9062 | #5, #9 |
| ⭐ **R3 anneal + 2× steps** | `anneal` + the compute dial | #1 under the annealed read | **1.0000** | **#2 only** |
| R4 store-relative ψ | ψ representation (doctrine I-2) | #4 blank | 0.4792 | #5, #8, #9, #12 |

⭐ **R1 is the wave's cleanest mechanism result:** the annealed read — N109's measured fix, applied here as a controller **verb inside the running full system** — takes self-probe acquisition **0.828 → 1.000** and decode **0.9062 → 1.0000**, clearing **#5 and #9 simultaneously**, at zero extra bytes and (to within 1 step) the same compute. It *introduces* a #1 trip (`ρ_conv` 4.3e-7 → 3.6e-6), and **R3 clears that too** by spending the compute dial (read_steps 800 → 1600): `ρ_conv` back to **4.3e-7**, decode still **1.0000**.
⛔ **In that best configuration the only surviving trip is #2 with `D = 0.000000`, `U = 0.0625`, `ρ_ex = 0.000`** — the settle and the arg-min launder agree on *every* query. That is **exactly Prop D2a**, measured inside the full system on a learned `V_θ`: **the dividend at v0 is 0.0000, and it is structurally zero, not accidentally zero.**

### 3.3 The dividend, as an instrument reading (charter §2.1)

**`dividend = −0.0938` at S0** (full 0.9062 − settle-deleted launder 1.0000), **`−0.0833` at S3**, **`0.0000` at R1/R3**, **`−0.3750` at S4**.
Controls: **blank store 0.125–0.167 = chance, bar 0.249–0.328 → PASSES everywhere** (monitor #4 never trips); **same-keys null 0.000–0.125 ≤ chance**.
⚠ **Bytes are NOT matched and the harness says so:** `V_θ` **57 344 B** + codebook 128 B = **57 472 B** vs the launder's **160 B** — **ratio 359.2×**, `matched=False`. At K=8 the launder wins on bytes by 359× *and* on accuracy. Any future dividend claim must fix this axis first (either shrink `V_θ` or load the store far beyond the launder's table).

### 3.4 Certificates and geometry (per stage, from the artifact)

| stage | sep | sep/σ_q (N2 ≥ 5.15) | λ_min (N3 > 0) | payload gap | δ_read (basin-cond.) | erf accuracy |
|---|---|---|---|---|---|---|
| S0–S3 | 1.0245 | **6.830** ✓ | +2.29…+2.58 ✓ | 0.2222 | 0.0010–0.0019 | 0.9994 |
| S4–S6 | 0.4610 | **3.073** ✗ | +1.0634 ✓ | 0.2222 | **0.0571** | 0.8756 |

Reach (#11) never trips: worst margin **+0.135**, per-item `a_U` ∈ [1.078, 1.446] against payloads |a| ≤ 1. Fitted wells at S0: `D` 0.46–0.80, `s` 0.33–0.45, β 32–56 — and **wider wells for larger |a|** (s = 0.398/0.427 at |a|=1 vs 0.340/0.369 at |a|=0.33), an independent replication of w26's `corr(s_fit, |a_i|) = +0.821` on a different harness.

### 3.5 ⛔ The finding that matters most (and the §8.2 reading)

**The hard falsifier did not fire** — the clean configuration is *not* degenerate: items are held in a **learned `V_θ`** (57 KB of atoms, no per-item arrays in the read path), the payload is recovered from the landscape by 1200–2000 Verlet steps, the store is written by **masked local writes**, and evictions/deletions/decay are physical operations on the item's own atoms. **But one lever is doing more work than is comfortable: separability.**

* At `sep/σ_q = 6.83` (S0–S3) the system reads at 0.90–0.92 and, with the anneal verb, **1.000**.
* At `sep/σ_q = 3.07` (S4, basins genuinely overlapping) it reads at **0.479** while its own launder holds **0.854**, δ_read degrades **43×** (0.0013 → 0.0571) and acquisition halves (0.938 → 0.542).

So: *the physics is not currently buying anything that separation is not already providing, and it is losing more than the table does when separation is removed.* That is the honest state of the instrument at v0, it is exactly what §8.2 warns about, and I have **not** rolled S4 back to make the acceptance table look better (the stage is retained, tripping, in the shipped artifact).

⚠ One structural statement worth the Hub's attention: **permitted basin interaction and the merge certificate `2 s_max + κ′σ_q ≤ sep` are mutually exclusive by construction.** S4 can only exist with the admission radius **deliberately out of band** (`d_safe_override`), and the harness records that fact rather than hiding it. `controller-doctrine`'s P1 grid never tested this pair on a learned store; it is a candidate **empty band pair**, which its own falsifier calls "the single most valuable output of the wave".

### 3.6 Every monitor's trip state (the reported artifact; `n/a` = not a runtime trip by design)

| # | monitor | state | where it fired | false-trip mode (implemented + reported) |
|---|---|---|---|---|
| 1 | overdamping | **TRIPPED** | R1 (`ρ_conv` 3.6e-6) | flat minima ⇒ tiny ‖∇V(q₀)‖; denominator floored |
| 2 | settle→arg-min | **TRIPPED** | R1, R3 (`ρ_ex` = 0) | tight query law ⇒ `U`→0 ⇒ 0/0 ⇒ **inapplicable**, not passing |
| 3 | vacuous gate | ⚠ **UNTESTED** (never fired) | — | a genuinely well-separated stream gives f=0 legitimately |
| 4 | blank | ⚠ **UNTESTED** (never fired) | — | a skewed marginal makes 1/K the wrong bar (empirical chance used) |
| 5 | addressing | **TRIPPED** | S0, S1, S4–S6, R2, R4 | a deliberately decayed item self-probes badly |
| 6 | objective divergence | ⚠ **UNTESTED** (inapplicable: needs ≥4 consolidations) | — | a curriculum change flattens retrieval legitimately |
| 7 | mass gauge | **n/a by design** — passes in `pytest` (<1e-5, Newtonian) | — | endpoint-only comparison passes vacuously ⇒ trajectory used |
| 8 | certificates | **TRIPPED** | S4–S6, R4 (N2+N3) | a declared register/coset shares a site by design |
| 9 | lifetimes | **TRIPPED** | everywhere except R1/R3 | items with intentionally different `leak_i` |
| 10 | dead axis | **TRIPPED** | S0, S6 (`gamma_read`, `traj_stride` inert) | a live knob at a no-op value is not dead |
| 11 | reach | ⚠ **UNTESTED** (never fired; margin +0.135) | — | single-well ⇒ mis-flags crowded cells |
| 12 | starvation | **TRIPPED** | S1, S4–S6, R4 | an intentional update of a live item |
| 13 | maturity | **n/a by design** (provenance field) | — | — |
| M14 | guard liveness | ⚠ **UNTESTED** (never fired **after** the canary was fixed) | fired while `admit.reach` / `place.injective` were unexercised | a canary that does not exercise a guard |

**Canary guard counts (M14 input), S0:** `admit.priority 1 · admit.reach 1 · admit.merge 2 · admit.budget 1 · place.lambda_min 1 · place.injective 1 · evict.persistence 1 · evict.class_i 1 · evict.set_function 1 · decay.permanent 6 · route.signal 2 · retry.budget 1 · anneal.return 2 · expand.monotone 2` — **all 14 designed guards fire**.

---

## 4. Two mechanism findings fixed in-code (both measured, both now tested)

1. **⭐ Eviction must RE-DRAW the freed atom group, not zero it.** Zeroing the freed rows (the obvious "leave no trace") is wrong twice: it **starves the next item in that slot** — atoms at the origin cannot reach a well at |a| ≈ 1 from the payload-zero launch manifold (the `atom_init_scale` lesson) — measured as fitted depth `D = 0.00` and unretrievable items, dragging self-probe acquisition **0.83 → 0.33** and decode **0.91 → 0.70**; and it is a **membership leak** (a zeroed row is distinguishable from a never-used one, which holds a scattered draw at `amp = √1e-4`). Fixing it recovered the whole gap and removed 3 of 5 trips at S0.
2. **`place` needs an INJECTIVITY guard, not just `λ_min > 0`.** The first place-pass relaxed two items onto the **same** minimum and committed it: `sep → 0.0000`, N1/N2/N3 all failed, decode 0.906 → 0.859. A shared minimum is a perfectly good minimum. `place.injective` (a re-derived site must stay ≥ `d_safe` from every other live address) is now a designed guard, and R2 is harmless after it.

*(Both are C2-owned files only. Neither touches C1W27 territory.)*

---

## 5. PREREG scorecard (`.claude/outputs/full-clu-harness/PREREG.md`, written before any measured run)

| prediction | outcome |
|---|---|
| **dividend NEGATIVE, point −0.10, range [−0.30, 0]** | ✅ **−0.0938 at S0** (and −0.0833 at S3) — inside the range, **0.006 from the point estimate** |
| #5 addressing trips at S4 and is **cleared by `anneal`** | ✅✅ trips at S4 (0.542) **and** cleared by the anneal verb (0.828 → **1.000** at R1) |
| #9 lifetimes trips at S1 | ✅ (trips at S0 already, and everywhere except R1/R3) |
| #10 dead axis trips at startup on the first full run | ✅ trips at S0 and S6 (2 of 4 declared dials inert) |
| #7 not a runtime trip, passes in `pytest` | ✅ |
| #13 never trips | ✅ |
| #6 never trips | ✅ (but **inapplicable**, so it is UNTESTED, not confirmed) |
| M14 trips at S0/S1, cleared once the canary is fixed | ✅ (tripped on `admit.reach`, then on `place.injective`; clear after the canary exercised both) |
| #1 never trips | ◐ **partially refuted** — clear at S0 (4.3e-7) but **trips under the annealed read** (3.6e-6), and clears again with 2× steps |
| #12 starvation trips at S2/S3 | ◐ **wrong stage** — trips at S1 and S4–S6, clear at S2/S3 |
| #2 trips at S0 and stays tripped | ◐ **wrong stage, right mechanism** — **inapplicable/clear** at S0–S6 (`ρ_ex` 0.79–2.5), then **trips at R1/R3 with `D` exactly 0** |
| **#8 certificates trips FIRST at S0, cleared by `expand`** | ⛔ **REFUTED, and my derivation was wrong**: I computed the N2 bar with `σ_q = 0.24` (the doctrine's grid value) and then ran at the shipped `σ_q = 0.15`. Measured `sep/σ_q = 6.83 > 5.15` ⇒ clear at S0. It trips at S4 instead |
| #11 reach trips at S0 for the largest-\|a\| items | ⛔ **REFUTED** — worst margin **+0.135**, 0/8 unreachable |
| #3 vacuous gate trips at S2 on fire-rate | ⛔ **REFUTED** — the collision offer put the fire rate at 0.091 ∈ (0,1); #3 never fired at all (**UNTESTED**) |
| #4 blank trips at S6 (raw-trajectory ψ) | ⛔ **REFUTED and instructive** — blank stays at chance under the tail-mean ψ because that ψ reads **only the payload channel**. The `q₀ = φ(x)` leak the doctrine predicts needs a ψ with access to the **address** coordinates, i.e. a *learned* ψ. **Flagged to `trainability-spike`: the trajectory launder is mandatory the moment your ψ can see the address block.** |

**Score: 8 confirmed · 3 partial · 4 refuted.** The dividend prediction (the one that mattered) landed; the four refutations are all cases where I predicted a trip that a *correct* configuration prevented — which is the good direction to be wrong in.

## 6. Test suite

**`pytest -q` on the branch: `1 failed, 824 passed` in 14m29s** — and the single failure was **mine**:
`test_clu_system.py::test_fixed_point_residual_is_small_at_a_settled_point`, which **passes when the file
runs alone**. Cause, diagnosed and fixed: several repo test modules enable **`jax_enable_x64` at MODULE
import**, so x64 is globally ON in a full-suite run (the hazard is documented in
`test_hopfield_capacity`'s own fixture, handover §7.2); the harness settles in float32 and the residual
asserted on is a float32 quantity. Fixed with the repo's standard autouse `float32_dynamics` fixture
(commit `c282089`) and verified by **reproducing the exact failing ordering**:
`pytest tests/test_clu_system.py tests/test_goldstone.py` → **37 passed** (it fails without the fixture).

✅ **The full-suite re-run after the fix completed: `825 passed` in 13m26s, zero failures** (`pytest -q`
on the branch, worktree, main venv, JAX 0.9.0). The branch is green.

New tests: 65 — `tests/test_monitors.py` (25) · `tests/test_clu_controller.py` (19) ·
`tests/test_clu_system.py` (21).

## 7. Git footprint

- **Branch** `agent/experiment-engineer/full-clu-harness`, worktree `../CHLU-fullclu`, base local `main` (rebased onto **`e0f06d3`**, which advanced under me — C1W27 merged `r2-d-sweep-close` and `deletion-waitlist-stiffness` mid-task; **rebase was conflict-free**).
- **Commits** (5): `665fcb1` API FREEZE · `4cd1a9a` implement monitors/controller/system/dividend · `6d10298` exp_clu_system + CLI hook + tests · `43b25f6` monitor #10 wiring · `c282089` float32 test fixture.
- **Files touched — all C2-owned, zero read-only violations** (verified file-by-file against the task's list): `chlu/core/clu_system.py` (new) · `chlu/core/monitors.py` (new) · `chlu/core/clu_controller.py` (new) · `chlu/eval/dividend.py` (new) · `chlu/experiments/exp_clu_system.py` (new) · `tests/test_{clu_system,monitors,clu_controller}.py` (new) · `chlu/cli/experiment_cmd.py` (**+53 lines, one new parser block + one new command function**; the only shared file, and the task's read-only list does not name it — flagged here for the Hub).
- **`chlu/config.py` NOT touched.** `CluSystemConfig` lives in `clu_system.py` per the file-ownership rule.
- One correction made before finishing: the results JSON + PNG were accidentally committed in `6d10298` and were **removed from the commit** (artifacts live under `.claude/outputs/full-clu-harness/`).
- **Not pushed, not merged.** Branch left for Hub review.

## 8. Open questions / follow-ups / risks

1. **Single seed.** Everything here is seed 0. Multi-seed is owed before any number leaves this report.
2. **`chlu/cli/experiment_cmd.py`** is the one shared file I edited (+53 lines, purely additive at two sites). If another C2 engineer adds a CLI hook this wave, that is the only possible conflict point.
3. **Monitor #3, #4, #6, #11, M14 are UNTESTED** — they never fired on any configuration I ran. #6 in particular is *inapplicable* (it needs ≥4 consolidation windows; my stages run one each). A stream long enough to exercise #6 is a cheap follow-up.
4. **`shared_metric_launder` and `trajectory_launder` are implemented but NOT RUN** (doctrine I-12/I-2). They need a fitted shared metric and a ψ that sees the address block respectively — both are `memory-gym-v0`/`trainability-spike` territory.
5. **Monitor #10 tier (a) is not implemented** (the access-counting config proxy). Declared in the artifact, not silently passed.
6. **The `expand` verb is implemented and guarded but never *needed*** in this run (utilisation peaked at 0.80 at S4). Its N91 value is untested here.
7. **Risk to flag to the Hub:** the harness's clean band is narrow. `sep/σ_q` 6.83 → 3.07 is the difference between decode 0.92 and 0.48. If the gym's query law is any noisier than `σ_q = 0.15`, expect the S4 picture, not the S0 picture.

---

## Proposed handover updates (for the Hub)

**§2/§3 (architecture + config).**
- New module trio `chlu/core/{clu_system,monitors,clu_controller}.py` + `chlu/eval/dividend.py` + `chlu/experiments/exp_clu_system.py`, CLI `chlu exp-clu-system`. **`CluSystemConfig` deliberately does NOT live in `chlu/config.py`** (C2W1 ownership rule); the override path is a `clu_system:` block in the project YAML, read directly.
- ⭐ **`CluSystem.read()` returns the strided trajectory plus `q*`** — the first time in 27 waves that a read-out can see anything but the settled point.

**§7 (known issues / live).**
- ⚠ **Prop D1's `D ≤ U` is violated under the `sep/2` inradius proxy on a learned `V_θ`** (measured `D/U` 1.5–2.5). Do not treat `sep/2` as the certificate. *(R1 above.)*
- ⚠ **The doctrine's γ band is harness-specific.** Shipped read: `γ_read = 0.02`, 800 steps, `ρ_conv = 4.3e-7` — within 2.3× of the 1e-6 trip. *(R2.)*
- ⚠ **Two of four declared read dials are semantically dead** (`gamma_read`, `traj_stride`). *(R3.)*
- ⚠ **Eviction that zeroes a freed atom group starves the next item in that slot and leaves a membership trace.** Fixed in `LearnedVStore.reinit_group`; the same pattern may exist wherever a slot is recycled.

**§8/§10 (record).**
- **The C2W1 dividend at v0 is `−0.094` (S0) / `0.000` (best configuration), with `D = 0` exactly** — Prop D2a confirmed inside the full system on a learned store. The launder also holds a **359×** byte advantage at K=8.
- **The annealed read works as a controller verb**: acquisition 0.828 → 1.000, decode 0.906 → 1.000, clearing modes #5 and #9 simultaneously at zero extra bytes; the residual cost it introduces (#1) is bought back with 2× read steps.
- **Candidate empty-band pair for the doctrine's P1:** *permitted basin interaction* × *the merge certificate*. They are mutually exclusive by construction, and the C2W1 grid never tested that pair on a learned store.
