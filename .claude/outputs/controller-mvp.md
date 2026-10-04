# controller-mvp — experiment-engineer report
Task + acceptance criterion: build the hand-coded MVC-0 controller (admission + placement + eviction/decay, no learning) on a designed store and run the N75 rematch — retention-vs-K controller ON/OFF, four-primitive comparison, per-admitted AND per-offered, admitted-fraction-vs-packing-bound, geometry so the gate CAN fire, config flags, pre-registered, tests green.
Status: **done.**

> **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Two items.**
> 1. **The rematch has a two-part verdict and both must travel together.** CLU+controller is BEST-of-five on **retention-per-admitted** (1.000 flat to K=64). On **retention-per-offered** the verdict is *geometry-conditional*: on a FIXED small address space it LOSES to the GRU (0.081 vs 0.57 at K=64 — the anticipated headline), but once the disk is SIZED so the packing bound ≥ K it BEATS all four primitives (0.669 vs gru 0.57). The single-number claim "CLU+controller wins/loses the rematch" is false; the honest claim is "the address space, not the controller, is the binding constraint" (theorist A4). Whoever writes the CL entry (`continual-learning-recon`) owns this.
> 2. **PREREG P7 refined, not confirmed.** Sized geometry does NOT reach per-offered 1.0 (measured 0.669 at K=64): achieved packing of *random* disk proposals is below the ideal bound (admitted 42.8/64), the `d_eff < d` shell-concentration effect `address-space-dimension-scaling` already measured. The controller wins per-offered under sized geometry, but the residual gap to 1.0 is a packing-efficiency limit, not a controller limit.

---

## Flag-provenance table

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/controller-mvp`, base local `main` @ `7ff0651` |
| commits (5) | `4927f3b` store primitives · `a1db29f` Controller · `44f8146` config+CLI · `584b416` experiment · `fad2789` tests |
| seeds | **0,1,2,3,4** on every controller cell (paired: identical proposal sequence per seed, arms differ only in the controller) |
| designed store | `AtomStorePotential`: α=0.02, **s=0.35**, κ=1.0, dim=3 — the theorist's S3 values; **NO learning, no gradient anywhere in the controller loop** |
| admission gate | **d_safe = 4.4·s = 1.54**; disk proposals radius **R=2.0** (fixed geom) or R=`0.808·√K` (sized); 400 relocation candidates; budget = capacity (on_fixed/on_sized) or round(N_pack)=6 (on_evict) |
| decay demo | leak **0.35**/tick (half-life 1.98), amp_floor 0.05, item 0 permanent (leak forced 0) |
| retrieval | TWO-PHASE, γ_address **0.05**×400 → γ_read **0.0**×800, dt 0.05, tail 25%, 8 subsample; q2(0)=p2(0)=0 (anti-decoration guard) |
| metric | **value recovery** (basin AND \|read−stored\|<tol), tol capped at 0.35×codebook spacing; 16 queries/item |
| primitive baselines | **REUSED from w21 `sequential-write-interference` §3.2** (same harness, same protocol, same seeds) — `run_primitives=False` for this run; sanctioned by the task ("reuse their numbers") |
| JAX / equinox / optax | 0.9.0 / 0.13.4 / 0.2.6 — main venv (protocol §4, no worktree sync) |
| PREREG | `.claude/outputs/controller-mvp/PREREG.md`, written before the harness ran |
| metrics | `.claude/outputs/controller-mvp/exp_controller_mvp_metrics.json`; figure `controller_mvp_rematch.png` |

---

## 0. Headline

1. **The controller exists and every verb works.** admission (spacing gate + refuse-and-relocate), placement (derived address), eviction (staleness/depth budget), decay (leaky wells) — all hand-coded, all tested, no learning. The gate **demonstrably fires** on this geometry: live-site min-spacing **1.61 ≥ d_safe 1.54**, intervention rate rising to **0.97** at K=64 (contrast N74: on the w20 ring the same gate was arithmetically vacuous, spacing 1.4142 ≥ d_safe 1.10).
2. **Controller OFF (ungated designed store) reproduces w21 and then dies.** per-offered = per-admitted collapses `1.00 → 0.110 (K=16, EXACTLY w21 designed_ungated) → 0.009 (K=32) → 0.000 (K=64)`. In a fixed radius-2 disk, 64 atoms overlap (min-spacing 0.03 ≪ 2s=0.70).
3. **Controller ON, per-ADMITTED = 1.000 flat at every K, best of five.** The gate never stores an item it will corrupt; admitted count saturates at **5.2 ± 0.4 ≈ packing bound 6.1** on fixed geometry.
4. **Controller ON, per-OFFERED is geometry-conditional — the task's central question, answered honestly:**
   - **Fixed geometry:** per-offered = admitted/K = `5.2/K`: 0.325 (K=16), **0.081 (K=64) — LOSES to gru 0.57, mlp 0.43, attn 0.34, and even learned-CLU 0.16.** The abstention price, charged in full. **This is the headline the task told me to report plainly.**
   - **Sized geometry** (R grown so N_pack ≥ K): admitted 42.8/64, per-offered **0.669 at K=64 — BEATS all four primitives.** The controller wins per-offered *iff the address space is sized to the load*.
5. **Decay/eviction verbs confirmed.** A permanent item retains 1.000 through 8 decay ticks while leaky wells shallow as `exp(−0.35·t)` (half-life 1.98) and self-evict below floor (`decayed_out = 6/6` leaky items). Staleness eviction removes the LRU non-permanent item; a full all-permanent store raises a capacity alarm (refuse), never a silent overwrite.
6. **The admission test is cheap; refuse-and-relocate is the cost.** The gate itself is 0.006 ms (O(n_stored) distances). When a proposal collides and relocation fires (400 candidates × min-sep), it reaches 3.1 ms at n_stored=64 — ~4× one 0.7 ms read. Cheap vs a gradient-trained write; NOT free vs a read. **P9 partially falsified — reported honestly (§5).**

---

## 1. Item 1 — the MVP controller (hand-coded, designed store)

`chlu/core/controller.py` — a `Controller` wrapping an `AtomStorePotential`, mechanising the three decision rules of `clu-controller-spec` §3, **no learning anywhere**:

| verb | rule | implementation |
|---|---|---|
| **admission** (C5-A1/A2) | novelty test: admit iff `min_j d(q_new,q_j) ≥ d_safe`, else refuse-and-relocate | reuses `admission.admit_site`; d_safe from the packing geometry (N74's lesson: geometry chosen so admission is *decidable*) |
| **placement** (C1) | derived address = the admitted/relocated site; the writer records where it wrote | `AtomStorePotential.with_item(site,...)`; nothing searched (Prop 5's dead gradient routed around) |
| **eviction** (C5 budget / §3.C trash) | when full: evict LRU (staleness) or shallowest (depth) non-permanent; permanent never evicted; all-permanent full store → capacity alarm | new `AtomStorePotential.evict(slot)` |
| **decay** (per-item retention, w22) | `tick()`: leaky wells `amp *= exp(−leak)`, self-evict below floor; permanent (leak=0) untouched | new `AtomStorePotential.with_amps(amps)` |

Permanent + leaky wells coexist in one store: `permanent ⟺ leak 0` (a designed flat coset — `clu-controller-spec` Prop C-N: permanence is a constraint, not learned). Config-driven; all flags in `ExperimentControllerMvpConfig` (§4).

**Geometry so the gate CAN fire (reported, per acceptance):** d_safe = 1.54; fixed disk R=2 → **packing bound N_pack = πR²/(√3/2·d_safe²) = 6.12** (N74 measured 6.0±0.9 — reproduced here as admitted 5.2±0.4). Achieved live-site spacing **1.61 ≥ 1.54** confirms the gate certifies isolation; intervention rate 0.20→0.97 over K confirms it is not vacuous.

---

## 2. Item 2 — the N75 rematch (retention-vs-K, ON/OFF, per-admitted AND per-offered)

**CLU controller lines (designed store, 5 seeds, mean±std).** `adm` = admitted count, `interv` = (relocate+refuse)/offered.

| arm | metric | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | **K=64** |
|---|---|---|---|---|---|---|---|---|
| **OFF** (ungated) | per-offered=per-admitted | 1.000 | 0.800 | 0.787 | 0.448 | **0.110** | 0.009 | **0.000** |
| **ON fixed** | per-**admitted** | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** |
| **ON fixed** | per-**offered** | 1.000 | 1.000 | 1.000 | 0.650 | 0.325 | 0.163 | **0.081** |
| **ON fixed** | admitted / interv | 1.0/0 | 2.0/.20 | 4.0/.55 | 5.2/.78 | 5.2/.89 | 5.2/.94 | **5.2/.97** |
| **ON sized** | per-**admitted** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| **ON sized** | per-**offered** | 1.000 | 1.000 | 1.000 | 0.850 | 0.750 | 0.713 | **0.669** |
| **ON sized** | admitted / radius | 1/2.0 | 2/2.0 | 4/2.0 | 6.8/2.3 | 12.0/3.2 | 22.8/4.6 | **42.8/6.5** |

`on_evict` (budget=6, LRU) is identical to `on_fixed` here (budget rarely binds since admitted ≈ 5.2 < 6); the eviction verb is stressed in the decay demo (§3) and tests instead.

**Four-primitive w21 lines (reused, `sequential-write-interference` §3.2 — parametric sequential writes, matched params, 5 seeds).** Baselines accept every offer, so their mean-retention IS retention-per-offered:

| primitive | per-offered @K=16 | **per-offered @K=64** |
|---|---|---|
| gru | 0.85 | **0.57** |
| mlp | 0.70 | 0.43 |
| attention | 1.00 | 0.34 |
| clu (learned block) | 0.80 | 0.16 |

### The rematch verdict (both metrics, both go in the paper)

| ranking at K=64 | per-**admitted** | per-**offered** |
|---|---|---|
| 1 | **CLU+ctrl (any ON) 1.000** | **CLU+ctrl sized 0.669** |
| 2 | — | gru 0.57 |
| 3 | | mlp 0.43 |
| 4 | | attention 0.34 |
| 5 | | clu-learned 0.16 |
| 6 | | **CLU+ctrl fixed 0.081** |
| 7 | | CLU+ctrl OFF 0.000 |

- **Does designed-store+controller move CLU from worst toward best? YES — decisively on per-admitted (worst→best), and on per-offered too IF the address space is sized to the load (0.669 > gru 0.57).** On a fixed small space the controller abstains and is *last* on per-offered (0.081) — worse than the learned-CLU it replaces. **The task's warned-for headline is real and it is geometry-conditional.**
- **Admitted-fraction vs packing bound (acceptance check):** fixed geometry admits **5.2±0.4 vs N_pack 6.1** (0.85× the bound — random disk proposals + relocation fall a little short of ideal farthest-point packing, consistent with N74's 6.0±0.9). Sized geometry admits **42.8/64 vs target 64** — the `d_eff<d` shell-concentration shortfall.

---

## 3. Item 3 — the honest accounting

**(a) Per-admitted vs per-offered are different metrics (the abstention–accuracy trade, stated before a referee does).** The controller *refuses* items the baselines accept: at K=64 fixed geometry it stores 5.2 of 64 offers (intervention 0.97). Per-admitted (1.000) measures the purity of what it chose to keep; per-offered (0.081) charges every refusal as a miss. **Both are in the table above.** A report quoting only per-admitted would claim "CLU+controller is a perfect memory" — which is true only of the 8% it agreed to hold.

**(b) Decay/eviction demo (permanent + leaky wells, one store).** K=8, item 0 permanent, items 1–7 leaky (leak 0.35):

| tick | 0 | 1 | 2 | ... | 7 | 8 | (after) |
|---|---|---|---|---|---|---|---|
| permanent item-0 retention | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| leaky item-1 amp | 1.000 | 0.705 | 0.497 | ... | 0.086 | 0.061 | <0.05 → evict |
| leaky item-1 retention | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |

`decayed_out = 6/6` leaky wells self-evicted (amp < floor at t≈8.6 = ln(20)/0.35); the permanent well is untouched. The amplitude law is exactly `exp(−leak·t)` (measured 0.705 = exp(−0.35)). This is the w22 per-item retention machinery — **permanent AND forgettable content in one store, by a physical amplitude decay, not a bookkeeping delete.**

**(c) What the controller CANNOT fix (from `clu-controller-spec` §5, stated plainly):**
- It cannot make a τ=∞ item on an unconstrained *learned* V (Prop C-N) — permanence here is the *designed* `leak=0` coset, not a training outcome.
- It cannot beat the packing bound: on a fixed address space per-offered is capped at N_pack/K (measured 0.081 at K=64). Densifying past the bound crosses regime-2 (selectivity collapse) — the ungated arm's 0.000 is what that looks like.
- Its spacing certificate is **meaningless for a global-support learned write** (N75): the controller-ON arm is on the *designed* store only, by construction.

**(d) Cost of the admission test per write:**

| n_stored | admission (ms) | note |
|---|---|---|
| 1 | 0.006 | O(n_stored) distances, no relocation needed |
| 4 | 0.006 | " |
| 16 | 2.60 | proposal collides → 400-candidate relocation fires |
| 64 | 3.14 | " |
| — | **read = 0.70 ms** | one two-phase rollout (1200 steps), for scale |

The **gate itself is trivial** (0.006 ms). The cost is **refuse-and-relocate** (400 candidate draws × O(n_stored) min-sep), which at n_stored=64 is ~4× a read. It is ≪ a gradient-trained write (≈200 steps) but NOT free relative to a read — reduce `n_relocation_candidates` or narrow the proposal disk to trade admitted-quality for speed.

---

## 4. Config flags documented (`ExperimentControllerMvpConfig`)

| flag | default | meaning |
|---|---|---|
| `atom_width` / `atom_alpha` / `atom_amp` | 0.35 / 0.02 / 1.0 | designed-store shape; d_safe is the only length scale |
| `d_safe_mult` | 4.4 | d_safe = 4.4·atom_width = 1.54 (5-orders locality) |
| `proposal_radius` | 2.0 | fixed-geometry disk (N_pack 6.1) |
| `K_grid` | [1,2,4,8,16,32,64] | the rematch ladder |
| `budget` | None (=capacity) | max live items; finite → earlier eviction |
| `evict_policy` | "staleness" | LRU; or "depth" (shallowest first) |
| `leak` / `amp_floor` | 0.0 / 0.05 | default leaky-well decay rate / self-evict threshold |
| `run_sized_geometry` | True | also run the R=0.808·√K arm |
| `decay_demo_K` / `decay_demo_leak` | 8 / 0.35 | the permanent+leaky demo |
| `run_primitives` / `kv_primitives` / `kv_baseline_K` | True / [mlp,gru,attn,clu] / 64 | four-primitive re-run (set False to reuse w21) |

CLI: `chlu exp-controller-mvp [--project N] [--seed I] [--quick] [--items 1 2 3]`; direct `python -m chlu.experiments.exp_controller_mvp --quick`.

---

## 5. PREREG scorecard (`PREREG.md` written before the harness ran)

| # | registered | measured | verdict |
|---|---|---|---|
| P1 | admitted saturates 6±1; intervention >50% at K≥8; gate fires | **5.2±0.4** (0.85× bound); interv 0.78→0.97; minspace 1.61≥1.54 | ◐ gate fires ✅, admitted slightly under 6 (random≠farthest-point) |
| P2 | per-admitted ≈1.0 flat | **1.000 exactly, every K** | ✅ |
| P3 | per-offered = admitted/K, ≈0.38@16, 0.094@64 | 0.325@16, **0.081@64** (=5.2/K) | ✅ form exact; constant tracks measured admitted |
| P4 | OFF ≈0.11@K=16 (=w21 designed_ungated) | **0.110** @16, 0.000 @64 | ✅ exact reproduction |
| P5 | ON-fixed per-offered @64 LOSES to gru | **0.081 < 0.57** | ✅ **the anticipated headline, confirmed** |
| P6 | per-admitted best of five | **1.000, best** | ✅ |
| P7 | sized geometry per-offered ≈1.0, beats all | **0.669** — beats all (gru 0.57) but NOT 1.0 | ◐ **refined: sized WINS but achieved packing (42.8/64) caps it below 1.0 (d_eff<d)** |
| P8 | permanent survives, leaky decays+self-evicts | perm 1.00 ∀t; leaky exp(−0.35t), decayed_out 6/6 | ✅ |
| P9 | admission <5% of a gradient write | gate 0.006 ms; **relocation 3.1 ms ≈ 4× a read** | ◐ **cheap gate, but relocation ≠ free vs a read — falsified as stated for reads** |

---

## 6. How I verified

- **Unit tests:** `uv run pytest tests/test_controller_mvp.py -q` → **11 passed** (28.96 s). Cover: packing bound = N74 6.1; `evict`/`with_amps` functional; admission refuse/admit/relocate; LRU eviction; permanent-never-evicted + capacity alarm; leaky decay exp(−leak) with permanent untouched; leaky self-evict below floor.
  - **Bug found by running the tests:** a `permanent=True` item initially inherited the controller's default `leak` and decayed. Fixed: `permanent ⟺ leak 0` forced at write (`a1db29f`). Test `test_tick_decays_leaky_leaves_permanent` pins it.
- **Full suite:** `uv run pytest tests/ -q -p no:randomly --no-cov` → **595 passed, 7 warnings in 686.13 s** (0 failed; the 11 new controller tests included).
- **Smoke:** `python -m chlu.experiments.exp_controller_mvp --quick` → exit 0.
- **Real run:** 5 seeds, K=[1..64], all three items → `exp_controller_mvp_metrics.json` (exit 0). Every number in this report is re-derived from that JSON.
- **Ruff:** `ruff check` on all six touched files → All checks passed.

## Git footprint
- **Branch** `agent/experiment-engineer/controller-mvp`, base local `main` @ `7ff0651`. **Not pushed** (protocol). Rebase-onto-`main` = no-op (base unmoved; no concurrent edits observed — clean tree at start).
- **Commits (5):** `4927f3b` (store primitives) · `a1db29f` (Controller) · `44f8146` (config+CLI) · `584b416` (experiment) · `fad2789` (tests).
- **Files — 3 new, 3 surgical:** NEW `chlu/core/controller.py`, `chlu/experiments/exp_controller_mvp.py`, `tests/test_controller_mvp.py`. Modified `chlu/core/memory_potentials.py` (**appended** `evict`/`with_amps` to `AtomStorePotential`; existing methods untouched), `chlu/config.py` (+1 dataclass at all three registration sites), `chlu/cli/experiment_cmd.py` (+parser +`cmd_exp_controller_mvp`). **No shared physics/training/plotting code touched.** `results/` not committed (precedent).

## Open questions / follow-ups / risks
1. **The sized-geometry per-offered (0.669) is capped by achieved packing, not the controller.** Farthest-point proposal sampling (vs uniform-random) or a vector-valued payload would close the 42.8→64 gap; untested here. This is the cheapest lever to make CLU+controller dominate per-offered cleanly.
2. **`on_evict` never got stressed in the rematch** (budget 6 ≥ admitted 5.2). A run with budget < N_pack would show the LRU rolling-buffer trading which items survive; the eviction verb is proven only in the decay demo + tests, not on the rematch curve.
3. **Primitive lines are reused, not re-run.** Identical protocol/seeds/harness make this sanctioned, but if a reviewer wants CLU+controller and the four primitives in one process, set `run_primitives=True` (adds the w21 §3 training cost).
4. **Single designed store family, Newtonian, p₀=0.** The relativistic kinetic and per-address masses could shift the retrieval constants (theorist's F5 risk), not the controller's decision structure.

## Proposed handover updates (for the Hub)
1. **§6 ground truth — new entry.** *The MVC-0 controller is built and run (w23).* Every primitive verb (decide/add/trash/evict) now exists, hand-coded, no learning. **N75 rematch verdict is two-part:** CLU+controller is **best-of-five on retention-per-admitted (1.000 to K=64)** and, on **retention-per-offered**, is **geometry-conditional** — LOSES to the GRU on a fixed small address space (0.081 vs 0.57 @K=64, the abstention price) but BEATS all four primitives once the space is sized to the load (0.669 vs 0.57). The binding constraint is the address space, not the controller (theorist A4). **Both metrics must be quoted together; a single-number rematch claim is false.**
2. **§7 — controller/geometry note.** The admission *gate* is trivial (0.006 ms); *refuse-and-relocate* is the cost (∝ n_candidates × n_stored, ~4× a read at n_stored=64). And sized geometry does not reach per-offered 1.0 because random disk proposals pack below the ideal bound (42.8/64) — the `d_eff<d` effect. Neither is a controller defect; both are geometry/sampling choices with knobs (`n_relocation_candidates`, proposal sampler).
3. **New CLI/config surface:** `chlu exp-controller-mvp`; `ExperimentControllerMvpConfig` (all three sites); new `chlu/core/controller.py` (`Controller`, `packing_bound_disk`, `radius_for_capacity`) and `AtomStorePotential.evict`/`.with_amps`. These are the reusable MVC-0 primitives for `continual-learning-recon`.
4. **For `continual-learning-recon` (the CL entry design):** the per-offered loss on fixed geometry scopes the design — a CL benchmark on a bounded CLU store must either (a) size the address space to the task's item count (packing bound ≥ K), or (b) accept abstention and be scored per-admitted with the admitted-fraction reported. Betting a benchmark on per-offered without (a) reproduces the GRU loss.
