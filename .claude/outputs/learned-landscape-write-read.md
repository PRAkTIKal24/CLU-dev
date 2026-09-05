# learned-landscape-write-read — experiment-engineer report

**Task + acceptance criterion:** does the w19 write→address→read loop survive a **learned** landscape — items 1–4 with blank controls throughout, the design-freedom curve + minimum-viable-design point, the interference number, the 2-D γ map, tests green.
**Status: done.** The loop **partially survives, and the honest answer is more negative than the single-seed run first said.**

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary).** Three items:
> 1. **The w19 blank-control protocol is too weak and must be restated program-wide.** w19 gated on the codebook read only. Under a learned `V`, the *nearest-centroid* blank scores **0.992–1.000** on landscapes with **nothing stored**. Every future retrieval claim must take its blank control over the **strongest read in use**, and must prefer **value-recovery** metrics (leak-immune) over classification reads (fooled by an arbitrarily small leak). No w19 number is retracted — the designed landscape leaks exactly 0.000 — but the *method* is.
> 2. **w19's "write locality = 4.17e-7" is confirmed as an artifact of the design.** Under learning, corruption is **2.9e-2 … 5.0e-1**. The vision's "write near without disturbing" element is **not yet real** for any learned landscape.
> 3. **My own single-seed minimum-viable-design point is overturned by my own 5-seed check** (§4). Anything quoting `sites_learned_payload` / freedom-2 as the answer must use the 5-seed numbers instead.

---

## Flag-provenance table

| item | value |
|---|---|
| commit | `e8b1b95` (branch `agent/experiment-engineer/learned-landscape-write-read`, base `main` @ `089cc6e`) |
| headline run seed | `project.seed = 42`; **5-seed check at seeds 0–4** (§4) |
| kinetic mode | **`newtonian_learned`**, inertia = 1 (identity ignores `log_mass`) |
| potential | **`DesignFreedomPotential`**, 5 rungs (`designed` → `free_mlp`); learned part trained, designed part frozen |
| write objective | `training/train_memory.py` — static (no BPTT): ‖∇V(z_i)‖² + min-margin over perturbations incl. the q2=0 query manifold + inter-item barrier. **600 Adam(w) steps, lr 3e-3, wd 1e-4**, n_perturb 32, σ_addr 0.25, σ_pay 0.6, margin 0.15, barrier 0.2 |
| retrieval | **TWO-PHASE**: γ_address **0.05** × 400 steps → address; γ_read **0.0** × 800 steps → read. dt 0.05 |
| queries | 32/item, Gaussian jitter σ = f·0.15 (matched to w19 arc length), σ_p 0.05; **q2(0)=p2(0)=0 always** |
| read | tail 25%, 8 subsample points; linear codebook read + nearest-centroid + full-state probe |
| item counts | K ∈ {2, 4, 8, 16}; K=16 excluded by the capacity rule (§2) |
| landscape geometry | λ=1.0, f=1.0, barrier 0.2, payload_kappa 1.0, bump_width 0.05, payload_seed 0 — **identical to w19** |
| langevin_noise | **N/A** — deterministic Verlet, no Langevin, no temperature |
| JAX | 0.9.0 (main venv reused per protocol §4; **no worktree sync**) |

**Designed vs learned, stated up front:** in *every* rung the **writer supplies the target sites `c_i`**, and even `free_mlp` carries `PotentialMLP`'s `0.05‖q‖²` coercivity. "Free" means free *potential family*, **not free structure**. Nothing here tests whether item sites emerge, and no number below may be quoted as if it did.

---

## 0. Headline

1. **Write→relax consistency survives learning; the READ's validity does not.** Relaxation reliably lands where the writer wrote (basin ≥0.98 for rungs 0–2 at K≤8), and the stored *value* comes back. But on any learned landscape a **linear classifier reads the item identity off a landscape with nothing stored** (nearest-centroid blank 0.992–1.000), so every classification-based retrieval number on a learned `V` is uninterpretable. Value-recovery metrics remain valid and are what I report.
2. **Minimum viable design (5 seeds): at least rung 1 — the FULL designed landscape plus a small learned residual — and even that is marginal.** Nothing freer survives. The single-seed run said rung 2; 5 seeds overturn it.
3. **Additive separability does NOT survive learning.** w19's 4.17e-7 corruption → **2.9e-2 … 5.0e-1**. The rung that best passed items 1–2 is **destroyed by one subsequent write (strict 1.000 → 0.000)**.
4. **The 2-D γ map refutes the "off-diagonal" prediction in its strong form and confirms it in its weak form.** Fidelity is a function of **γ_address alone**: spread over γ_read at the best γ_address is **0.0000**; spread over γ_address is **0.5625**. γ_read≈0 is *permitted* (which is what matters — it keeps the read Prop-4 gradient-safe), not *required*.

---

## 1. Item 1 — write→relax consistency, fidelity, blank controls (seed 42)

`bCb`/`bNC` = blank codebook / blank **nearest-centroid**; `bStr` = blank strict.

| rung | K | basin | **strict** | cbRead | bCb | **bNC** | bStr | value-blank OK | class-blank OK |
|---|---|---|---|---|---|---|---|---|---|
| designed | 2 | 1.000 | **1.000** | 1.000 | 0.500 | **0.500** | 0.000 | ✅ | ✅ |
| designed | 4 | 1.000 | **1.000** | 1.000 | 0.234 | **0.234** | 0.000 | ✅ | ✅ |
| designed | 8 | 0.988 | **0.988** | 1.000 | 0.141 | **0.133** | 0.000 | ✅ | ✅ |
| designed | 16 | 0.775 | 0.176 | 0.133 | 0.055 | 0.082 | 0.096 | ✅ | ✅ |
| skeleton_residual | 2 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.000 | ✅ | ❌ |
| skeleton_residual | 4 | 1.000 | 0.828 | 0.797 | 0.438 | **1.000** | 0.000 | ✅ | ❌ |
| skeleton_residual | 8 | 0.984 | 0.984 | 0.992 | 0.141 | **0.992** | 0.000 | ✅ | ❌ |
| sites_learned_payload | 2 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.000 | ✅ | ❌ |
| sites_learned_payload | 4 | 1.000 | 1.000 | 1.000 | 0.438 | **1.000** | 0.000 | ✅ | ❌ |
| sites_learned_payload | 8 | 0.984 | 0.910 | 0.523 | 0.258 | **0.992** | 0.000 | ✅ | ❌ |
| local_rbf | 2 | 1.000 | 0.500 | 1.000 | 1.000 | **1.000** | 0.000 | ✅ | ❌ |
| local_rbf | 4 | 1.000 | 0.742 | 0.719 | 0.531 | **1.000** | 0.000 | ✅ | ❌ |
| local_rbf | 8 | 0.574 | 0.305 | 0.141 | 0.086 | 0.391 | 0.191 | ❌ | ❌ |
| free_mlp | 2 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.000 | ✅ | ❌ |
| free_mlp | 4 | 0.828 | 0.828 | 0.812 | 0.438 | **0.969** | 0.000 | ✅ | ❌ |
| free_mlp | 8 | 0.730 | 0.605 | 0.547 | 0.266 | 0.742 | 0.000 | ✅ | ❌ |

**w19 comparison (designed rung, through the NEW two-phase path):** payload abs err **6.66e-4 @K=4**, 1.59e-3 @K=2, 9.60e-3 @K=8 vs w19's **9.98e-4**; codebook read **1.000 @K=2 and @K=8** vs w19's 1.000 / 0.992. **The w19 loop reproduces.** (PREREG committed that if it didn't, the harness — not the physics — is at fault.)

### ⭐ The address leak — why every learned classification cell is disqualified
The anti-decoration guard fixes `q2(0)=p2(0)=0` **exactly**, so the payload channel is *deterministic given the address*: there is no noise in it to mask a leak. A learned `V` couples `q2` to `(q0,q1)`, so payload-channel curvature differs per site by a measured **~1e-4 in feature units** against payload values of order 1 — and a *perfectly systematic* 1e-4 difference is a **perfect item code**. Hence blank nearest-centroid ≈ 1.000 while the blank's settled-*value* spread is ~0. The **designed** landscape has an exactly separable payload spring, so with nothing stored `q2` is an independent oscillator with the *same* frequency at every site: leak **exactly 0.0000**, blank at chance.

**Consequence (method-level, propagates):** classification reads are fooled by an arbitrarily small leak; **value-recovery metrics are immune** (a blank returns ~0 for every item → blank strict 0.000). All primary scoring below uses the value criterion.

### Durability
Payload-only accuracy **flat at 1.000** across steps 40→792 for all five rungs at K=2, drop **0.000**. ⚠ **Caveat: this is a classification probe in a cell whose classification blank fails for the learned rungs**, so only the *designed* row is attributable (it replicates w19's flat 1.000). **P4's predicted decay for free rungs is neither confirmed nor refuted** — the instrument can't see it. Unmeasured, flagged.

---

## 2. Item 2 — the fidelity-vs-design-freedom curve

**Capacity rule (important, else the curve is wrong):** the design-freedom question is only asked at item counts where the **designed reference itself works**. The designed rung fails at **K=16 (strict 0.176)** — w19's ring capacity ceiling (`K_max ≈ 0.2·2π/σ_θ = 8.4`). Admissible **K ∈ {2,4,8}**; K=16 excluded. Scoring learned rungs against K=16 would charge them for a capacity limit that has nothing to do with learning (my first pass did exactly that and reported "no rung survives" for the wrong reason).

| rung | freedom | learned params | min strict (value-gated) | passes VALUE | passes COMBINED |
|---|---|---|---|---|---|
| designed | 0 | 0 | **0.988** | ✅ | ✅ |
| skeleton_residual | 1 | 4481 | 0.828 | ❌ | ❌ |
| sites_learned_payload | 2 | 4481 | **0.910** | ✅ | ❌ |
| local_rbf | 3 | 120 | 0.500 | ❌ | ❌ |
| free_mlp | 4 | 4481 | 0.605 | ❌ | ❌ |

- **PRIMARY (leak-immune value criterion), seed 42:** minimum viable design = **`sites_learned_payload`, freedom 2** — designed ring + K angular wells (address geometry), payload channel learned. Loop **survives**.
- **COMBINED (also requires a classification read validated by its own blank):** minimum viable design = **`designed`, freedom 0**. Loop does **not** survive. Every learned rung is disqualified by the leak.

**⚠ The single-seed PRIMARY answer does not replicate — see §4.**

---

## 3. Item 3 — cross-write interference: **additive separability does NOT survive learning**

Write A (3 items on a 4-site ring), re-read A; then write B at the fresh site by *continuing* training; re-read A.

| rung | err A before B | err A after B | **corruption** | strict A before | strict A after |
|---|---|---|---|---|---|
| designed | 5.55e-4 | 5.55e-4 | **0.000e+00** | 1.000 | 1.000 |
| skeleton_residual | 7.82e-4 | 2.95e-2 | **2.87e-2** | 1.000 | 1.000 |
| **sites_learned_payload** | 1.37e-3 | 4.96e-1 | **4.95e-1** | 1.000 | **0.000** |
| local_rbf | 4.17e-1 | 4.40e-1 | 2.26e-2 | 0.333 | 0.333 |
| free_mlp | 1.62e-2 | 3.69e-1 | **3.53e-1** | 0.990 | 0.333 |

Codebook spacing = 0.667, so corruption of 0.35–0.50 is **of order the whole code**: destructive.

**The finding, sharply:** the designed landscape corrupts by **exactly 0.000** (w19: 4.17e-7 ✓ — P5 confirmed). **The rung that best passed items 1–2 (`sites_learned_payload`) is completely destroyed by one subsequent write: strict 1.000 → 0.000.** The `local_rbf` rung's small corruption (2.26e-2) is *not* evidence that imposed locality helps — it was already failing (strict 0.333 before B), so there was little left to corrupt. **P5 fully confirmed: w19's write locality was an artifact of designed additive separability, and no learned rung reproduces it.**

---

## 4. ⭐ Multi-seed robustness — this overturns my own §2 answer

5 seeds (0–4), strict success (the leak-immune metric), full 600-step writes. `.claude/outputs/learned-landscape-write-read/seed_check.json`.

| rung | K=4 | K=8 |
|---|---|---|
| designed | **1.000 ± 0.000** | **0.986 ± 0.003** |
| skeleton_residual | 0.903 ± 0.101 | 0.959 ± 0.043 |
| sites_learned_payload | **0.989 ± 0.014** | **0.739 ± 0.079** |
| local_rbf | 0.623 ± 0.330 | 0.348 ± 0.117 |
| free_mlp | 0.853 ± 0.095 | 0.599 ± 0.059 |

**`sites_learned_payload` at K=8 is 0.739 ± 0.079, not the 0.910 of seed 42** — seed 42 was a favourable draw and the mean is far below the 0.9 bar. **The §2 minimum-viable-design point does not replicate.**

**Corrected answer:** across 5 seeds, **no learned rung clears 0.9 at both K=4 and K=8**. `skeleton_residual` (freedom 1 — the *full* designed landscape plus a small residual) is closest at 0.903 ± 0.101 / 0.959 ± 0.043, and its K=4 mean straddles the threshold with individual seeds down to 0.766. So:

> **Minimum designed structure that preserves the loop: essentially ALL of it.** Learning survives only as a small residual on top of a landscape that already works, and even that is marginal. As soon as the payload channel itself must be learned, fidelity degrades with item count (0.989@K=4 → 0.739@K=8) and one further write destroys the item.

`local_rbf`'s ±0.330 at K=4 also means "designed locality, learned placement" is not a stable configuration at all.

---

## 5. Item 4 — the 2-D γ map (K=4). Strict retrieval; rows = γ_address, cols = γ_read

**designed:**

| γ_addr \ γ_read | 0 | 0.005 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| **0** | 0.438 | 0.867 | 0.875 | 0.875 | 0.875 |
| **0.005** | 0.727 | 0.992 | 0.992 | 0.992 | 0.992 |
| **0.02** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| **0.05** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| **0.1** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |

`sites_learned_payload` is the same shape (γ_addr=0 row: 0.508→0.711; γ_addr≥0.02 rows: all 1.000).

| quantity | designed | sites_learned_payload |
|---|---|---|
| spread over **γ_read** at best γ_address | **0.0000** | **0.0000** |
| spread over **γ_address** at best γ_read | **0.5625** | 0.2969 |
| off-diagonal gain over best single-γ | **0.000** | 0.000 |
| strict at γ_address = 0 (best γ_read) | 0.875 | 0.711 |

**Verdict on the Hub's prediction — split, exactly as pre-registered:**
- **(a) γ_address > 0 is required: CONFIRMED.** γ_address ≥ 0.02 ⇒ 1.000; γ_address = 0 caps at 0.875 / 0.711.
- **(b) the good region is off-diagonal (γ_read must be ≈0): REFUTED.** Once γ_address ≥ 0.02, fidelity is **exactly invariant to γ_read** (spread 0.0000 across the whole grid, including γ_read = 0.1). Off-diagonal gain = 0.000.

**The resolution, stated precisely:** retrieval needs dissipation **somewhere**, and does not care where. If phase 1 relaxes the query to a minimum with p ≈ 0, that point is a fixed point of the damped *and* the conservative map, so phase 2 is γ_read-independent. The interesting cell is the **γ_address = 0 row**, where γ_read *does* matter (0.438 → 0.875): with no relaxation phase, the read phase must do the settling itself — **which is exactly the single-phase protocol w19 ran, and it is where w19's 0.813-at-γ=0 came from.** So w19's "retrieval requires dissipation" was a *single-phase* artifact. **This is good news for Prop 5:** γ_read = 0 is permitted at no fidelity cost, so the read can be run conservatively where gradients are safe (Prop 4), while addressing gets its dissipation in phase 1 where no gradient is needed.

---

## 6. PREREG scorecard (honest; `PREREG.md` written before any harness ran)

| # | prediction | measured | verdict |
|---|---|---|---|
| P1 | designed ≈1.00/1.00 | 1.000 / 0.988 @K≤8 | ✅ |
| P1 | skeleton_residual ≥0.95 basin / ≥0.85 strict | 1.000 / 0.828–0.984 (5-seed 0.903±0.101) | ◐ basin ✅, strict borderline |
| P1 | sites_learned_payload ≈1.00 basin / 0.60–0.90 strict | 0.984–1.000 / 5-seed 0.739–0.989 | ✅ |
| P1 | local_rbf 0.50–0.90 / 0.20–0.60 | 0.574–1.000 / 0.305–0.742 | ✅ |
| P1 | free_mlp ≤0.50 basin / ≤0.20 strict | **0.730–1.000 / 0.605–1.000** | ❌ **too pessimistic** — free MLP addresses better than predicted |
| P1 | basin/strict gap widens monotonically with freedom | gap: 0.000, 0.000, 0.074, 0.269, 0.125 | ◐ widens but **not monotone** (free_mlp < local_rbf) |
| P2 | **min viable design = rung 2** | **rung 2 at seed 42, but NOT at 5 seeds** | ❌ **overturned by my own replication** |
| P2 | locality alone (rung 3) insufficient | local_rbf worst rung, ±0.330 across seeds | ✅ |
| P3 | every blank at chance | **learned blanks 0.99–1.00 under nearest-centroid** | ❌ **failed — and this is the wave's method finding** |
| P3 | *committed rule:* treat a high blank as NOT a measurement, not a success | enforced in code + regression test | ✅ rule honoured |
| P4 | designed/skeleton flat ≤0.02 drop | 0.000 | ✅ |
| P4 | free rungs decay ≥0.10 | 0.000 drop, but instrument invalidated by the leak | ⚠ **unmeasurable** |
| P5 | designed ≈0; free_mlp ≥0.1 destructive | 0.000e+00; **3.53e-1** | ✅ |
| P5 | separability does not survive learning | 2.87e-2 … 4.95e-1 | ✅ |
| P6a | γ_address>0 required, ≤0.7 at γ=0 | required ✅; **0.875 at γ_address=0**, not ≤0.7 | ◐ direction ✅, magnitude ❌ |
| P6b | **γ_read variation < 0.05 ⇒ "off-diagonal" false in strong form** | **spread exactly 0.0000** | ✅ |
| P7 | free_mlp working loop would be extraordinary | free_mlp fails the combined criterion; passes nothing at 5 seeds | ✅ no extraordinary claim needed |

**Three falsifications, one of them (P3) the wave's most transferable result**, and one (P2) where **my own replication overturned my own single-seed measurement** — recorded rather than quietly re-tuned.

---

## 7. How I verified

- `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …` (main venv reused per protocol §4 w6 lesson; **JAX 0.9.0**, no worktree sync).
- **Full suite: `404 passed` in 679.85s** (383 baseline + 12 new + 9 from other merged work). `ruff check chlu/ tests/` → **All checks passed**.
- Full battery `python -m chlu.experiments.exp_learned_memory` exit 0; `--quick` exit 0 in 44s. **The reported numbers come from a re-run against the exact committed source** (the first run predated a cosmetic `ruff format`, so I re-ran rather than report numbers from unversioned code).
- All numbers from `results/exp_learned_memory_metrics.json`, copied with the figure and `seed_check.json` to `.claude/outputs/learned-landscape-write-read/`.

### What I fixed mid-run (each would have produced a wrong conclusion)
1. **Blank control gated only the codebook read** — passed cells whose nearest-centroid blank was 0.992. Would have reported "the loop survives learning" on cells that read the address back. Now returns `(classification_ok, value_ok)`, taken over the strongest read, with a regression test.
2. **Item 3 queried the wrong ring.** Writing K−1 items but generating queries on a (K−1)-site ring put every query at the wrong angle — it inflated the *designed* rung's read error to 0.44 and would have manufactured fake interference. Added `n_sites`.
3. **K=16 was being scored against learned rungs** although the *designed* rung fails there (capacity ceiling). Would have reported "no rung survives" for a reason unrelated to learning. Reference rung now defines admissible K.
4. **`trainable_filter` returned arrays, not booleans** — equinox rejects an array-valued filter spec.
5. **A stale key in the plotting code destroyed a completed 7-minute run** (JSON was written downstream of `_plot_all`). Metrics are now written *before* plotting.
6. **`--quick` at 100+100 steps** made even the designed baseline fail, so a smoke run printed what looks like a scientific negative. Raised to 400+200.

---

## Git footprint
- **Branch** `agent/experiment-engineer/learned-landscape-write-read`, off local `main` @ `089cc6e`. Rebase onto `main` = no-op (up to date). Did **not** touch `origin/main` (§7.21).
- Worked in a **dedicated worktree** `../CHLU-learned-landscape` — the main checkout was on another branch (`agent/experiment-engineer/dt-units-split`). ⚠ **Self-reported slip:** I made one edit to the main checkout's `memory_potentials.py` before creating the worktree; I reverted it immediately (`git checkout --`, verified clean) and did all work in the worktree thereafter. No foreign work was present or lost.
- Commits (6): `1a49b27` potentials + write objective · `8c2fa06` experiment + config + CLI · `e5f7e93` tests · `67cb4d5` blank-control split + capacity rule · `e839f3d` durable metrics + figure fix · `e8b1b95` quick-mode budget.
- Files: **+** `chlu/training/train_memory.py`, `chlu/experiments/exp_learned_memory.py`, `tests/test_learned_memory.py`; **M** `chlu/core/memory_potentials.py` (appended `RBFAtoms`, `DesignFreedomPotential`, `DESIGN_RUNGS`, `ring_sites`; existing classes untouched), `chlu/config.py` (new group + all **three** registration sites), `chlu/cli/experiment_cmd.py` (import, parser, `cmd_exp_learned_memory`). Did not touch `utils/plotting.py` (shared) — figure is local, per the `exp_paid_access`/w19 precedent. `results/` deliberately not committed.
- Worktree left in place for review; remove with `git worktree remove ../CHLU-learned-landscape`.

## Open questions / risks
- **The write objective is static (no BPTT through the rollout).** It is a legitimate and cheap way to make each item an attracting minimum, but a rollout-based write objective (θ-gradients through the damped relaxation, which Prop 5 does *not* forbid) might place minima where the *dynamics* actually settle rather than where the loss says. **This is the single most valuable follow-up** — it is the obvious candidate for closing the 0.739 gap at K=8.
- **The leak may be curable rather than fundamental.** Two untested repairs: (i) add noise to the payload channel at launch (breaks the determinism that makes a 1e-4 leak perfectly decodable — but weakens the anti-decoration guard, so it needs care); (ii) impose payload/address separability *architecturally* (`V = V_addr(q0,q1) + V_pay(q0,q1,q2)` with a designed separable coupling), which is a new rung between 1 and 2. I would test (ii) first.
- **Single geometry (2-D ring + 1 payload channel), dim=3.** The K=8/K=16 degradation is a ring-capacity property (w19's open question about d-dimensional address spaces applies unchanged and is still the most valuable cheap follow-up).
- **`skeleton_residual` underperforming `sites_learned_payload` at seed 42 is not robust** (0.903 vs 0.989 at K=4 over 5 seeds — overlapping). Do not quote a non-monotone design-freedom curve from this data.
- No hyperparameter search was run on the write objective (one configuration, chosen a priori). The learned rungs are therefore **untuned**, which is the honest baseline but means their failures are not proven to be irreducible.

## Proposed handover updates (for the Hub)
1. **§6 ground truth — new entry.** The w19 loop has now been run on a **learned** landscape. Write→relax consistency and value recovery survive; **classification-based reads do not** (address leak); **write locality does not** (corruption 2.9e-2…5.0e-1 vs designed 0.000). Minimum designed structure preserving the loop = **essentially the whole designed landscape** (5-seed).
2. **§7 — new methodological issue (high priority, affects all future retrieval work).** *Blank controls must be taken over the strongest read in use, and value-recovery metrics preferred over classification reads.* Under a deterministic payload channel an **arbitrarily small** address leak is a perfect item code (measured: 1e-4 feature spread ⇒ 0.992 blank accuracy). w19's protocol gated on one weak decoder.
3. **§1/§8 — the two-phase γ result.** Retrieval fidelity depends on **γ_address only**; γ_read is exactly irrelevant (spread 0.0000) once γ_address ≥ 0.02. **w19's "retrieval requires dissipation / 0.813 at γ=0" is a single-phase artifact.** Consequence: the read can run conservatively (γ_read=0), so Prop 4's gradient-safe read and Prop 5's gradient-opaque relaxation **can coexist in one loop** — the Hub's proposed mechanism is viable on this axis.
4. **Vision amendment:** "write near without disturbing" (write mode b) is **not yet real** for learned landscapes. w19's 4.17e-7 must always be quoted as *designed-only*.
5. **New CLI/config surface:** `chlu exp-learned-memory`, `ExperimentLearnedMemoryConfig` (`reference_rung`, `blank_strict_max`, `gamma_address`/`gamma_read` are load-bearing, chosen from measurement), `chlu/training/train_memory.py` as a **third** training path — flag it in §2's architecture map so it is never conflated with `train.py`/`train_generative.py`.
6. **For the theorist:** the leak result may be statable as a proposition — *a deterministic read channel makes any non-separable coupling a perfect address code, independent of coupling magnitude* — which would explain why the designed (exactly separable) landscape is the only leak-free one, and would predict that separability, not locality, is the load-bearing designed property.
