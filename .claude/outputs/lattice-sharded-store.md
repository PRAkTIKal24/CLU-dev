# lattice-sharded-store — experiment-engineer report (w25)

**Task + acceptance criterion:** build the first real N-unit sharded CLU store (the
theorist's 5-point list — `CLULattice` container, the N98 init fix, the global
allocator, R2/R3 routers, ⛔ no R1, per-shard query noise) and run the §5.1 **2×2
discriminator**: does splitting the same items across N units, each written by its own
dig, clear a wall the monolithic store cannot? Plus the §5.3 read-parity check and the
routing-vs-read wall-clock.

**Status: done.** Build complete (5/5 items); **full suite 682 passed** (657 + 25 new),
`ruff` clean. Every arm is at **3 seeds** except the d=8 monolithic laundering
line (2; declared §8). ⭐ **Verdict: THEOREM-L1** — sharding fixes the d=6 K=64 wall, does
**not** fix the geometry-bound d=4 K=32 control, and `K = 32N` is **not** unclamped.
Every reported PASS carries its own value-blank control.

> ### ⛔ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Four items; two touch tier-A registry entries.
> 1. **⭐ N97 must be EXTENDED and the router recommendation NARROWED to R2 alone.** N97 says "route on pre-settle energy (R2) **or settling displacement (R3)** — both exact (1.000)". Measured here on the *shipped* designed store, 3 seeds: **R3 COLLAPSES — 1.000 / 0.549 / 0.236 / 0.228 at N = 1/2/4/8 — while R2 stays 1.000 / 1.000 / 1.000 / 1.000.** Structural, not tuning: `BallRegisterPotential` is **flat inside the address ball by design**, so at an item's site the owning shard's gradient is ~0 (it is a minimum) *and* a foreign shard's is ~0 (flat vacuum + an exponentially small tail) — the displacements are comparable small numbers, while the *energies* differ by ≥3 orders. **Same failure class as R1; only R2 (and the classical registry) survive.** Owner: curator (N97 wording) + theorist (§2.1).
> 2. **⭐ N98 IS NOT A SIDE-ISSUE — the localized atom init is the single biggest lever measured this wave.** At d=6 K=64 it lifts the **monolithic** store 0.8288 → **0.8802** (3 seeds) and the **sharded** store 0.8434 → **0.9251** (3 seeds), *and it repairs the dynamical router* (route accuracy 0.913 → 0.970). It is an **initialisation**, so it is a finding about w23's `atom_init_scale = 1.0`, not about sharding — and it therefore **bears directly on N92/N96's contested mechanism**: part of what has been read as a "write-operator ceiling" is atoms starting inside other items' wells. It ships **OFF by default** (bit-identical when disabled); promoting it to a default is a Hub call. Owner: Hub, jointly with `r2-geometry-revival`'s width finding.
> 3. **The d=8 K=64 cell is NO LONGER A WALL in this harness and cannot discriminate.** My in-run laundering control puts the **monolithic** store at **0.9121 ± 0.0029 PASS** (2 seeds), consistent with w24's post-hoc 0.9067 ± 0.0068. The "0.883 FAIL" baseline the task quotes is superseded. The 2×2 therefore rests on **d=6 (fixed) + d=4 (not fixed)**, which is still a complete discriminator. Owner: Hub (task/table wording).
> 4. **w23's atom floor `512·√2^d` is per-STORE, not per-item**, so a parameter-matched N-way split is automatically N× below it. Every `sharded_matched` arm here is 2–8× starved. This makes its wins *conservative* and makes its N≥4 collapses **not capacity results** (N92 protocol). Owner: whoever scopes the next sharded run.

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)
- **Dials:** capacity (R2 revival route ii) + **isolation** — and the isolation half
  came out cleanest: in this build shards are **separate parameter trees**, so a write
  to one unit cannot touch another **by construction**, not by masking. W1∧W2 hold
  exactly, with no `atom_write_mask_fn` and **no residual at all** (N98's 1.4% amp
  deviation has no referent in a product store).
- **Laundering control:** the **monolithic store at the same `K_total`**, matched total
  atoms, identical geometry / rollout / write operator / query jitter / pass criterion
  — verified at N=1 to agree with the w23 read path to float32 rounding
  (`test_monolithic_arm_reproduces_the_w23_read_path`), and reproducing the published
  w23/w24 baselines within seed noise. The routing half is declared **classical NN
  indexing** up front (N89). ⚠ The stronger standing control is stated here rather
  than buried: **a classical kNN over the registry solves this task outright** (RG
  route accuracy = 1.000 in every cell; a full kNN over all `K` addresses would score
  ≈1.000 on strict). Nothing here beats it, and nothing here is framed as beating
  anything — per the Head's w24 boundary ruling the capacity law is a law about the
  primitive, exempt from the demotion, **but its figure never claims a win.**
- **Falsifies:** neither the d=6 nor the d=8 sharded cell fixes its wall.
- **Does NOT falsify:** the d=4 K=32 control still failing (it is *supposed* to);
  per-offered abstention costs under undersized geometry (N91).

---

## Flag-provenance table (governs every number below)

| item | value |
|---|---|
| branch / base | `agent/experiment-engineer/lattice-sharded-store`, base **local `main` @ `63c668d`** |
| commits | 5, head **`29e2924`**: `5cdce3b` core container + N98 init · `d372089` experiment + config + CLI · `cee0854` tests · `500b5d5` registry router + oracle decomposition · `29e2924` RG in the default router list |
| worktree | `../CHLU-shard`; **main venv reused** (protocol §4) — **JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, Python 3.11, **CPU only, 8 cores, no GPU** |
| harness | `chlu/experiments/exp_sharded_store.py` + `chlu/core/shard_store.py`; sweep via `.claude/scratch/lattice-sharded-store/{run_cells,pool}.py` (staged per-unit JSONL, 4–7 concurrent workers ≈1 core each) |
| geometry / read / write / pass criterion | **inherited verbatim from `experiment_designed_mechanism` (w23/w24)**: d-ball, farthest-point `designed_sites` R=1, wall_margin .5, well_width .15; γ_addr .05 × 400 → γ_read .02 × 800, dt .05; `n_atoms = max(32·K, 512·√2^d)`; atom init_scale 1.0, init_width .3, depth_init 1e-4, confine .05; write = ONE global Adam(3e-3, wd 1e-4) dig, **600 steps**, n_perturb 32, σ_addr .25, σ_pay .6, margin .15, barrier .2; pass = mean strict ≥ **0.90** ∧ value-blank ≤ trivial ceiling |
| read-cost reduction (declared, uniform across arms) | `n_query_per_item = 16` (as w24 ⇒ 1024 queries at K=64) |
| sharding knobs (new group `experiment_sharded_store`) | allocation `global`, partition `spread` (best-of-four deterministic), headline router **R2**, `abstain_deadband = 0.0` (**never abstain** ⇒ per-offered, comparable to monolithic), `atom_init_local = False` in the headline / `True` in the ablation, `atom_init_local_mult = 2.0` (radius `2s = 0.6`) |
| seeds | **3 (0,1,2)** on every arm except the d=8 `monolithic` laundering line (**2**; declared §8) |
| langevin_noise | **N/A** — everything here is deterministic Verlet / static write |
| γ | **uniform scalar** per rollout (§2.3 #11: heterogeneous per-unit γ breaks conformal symplecticity) |
| PREREG | `.claude/outputs/lattice-sharded-store/PREREG.md`, written **before any discriminator cell ran**; 10 registered predictions — **5 confirmed, 2 falsified, 2 split/partial, 1 not run** |
| raw data | `.claude/outputs/lattice-sharded-store/{cells.jsonl, aggregate.txt, items2346_metrics.json, items2346_final.log}` |

**Calibration — the laundering control is the same object w23/w24 measured.**

| cell | my `monolithic` | published |
|---|---|---|
| d=6 K=64, 4096 atoms | **0.8288 ± 0.0186** (3 seeds) | w23 0.855 · w24 0.858 ± 0.001 |
| d=6 K=64, **8192** atoms (2×) | **0.8008 ± 0.0037** (3 seeds) | w23 0.809 — *the "more atoms make it worse" anomaly reproduces independently* |
| d=4 K=32, 2048 atoms | **0.8346 ± 0.0151** (3 seeds) | w24 `baseline_global` 0.801 |
| d=8 K=64, 8192 atoms | **0.9121 ± 0.0029** (2 seeds) | w22/w23 0.883 · **w24 0.9067 ± 0.0068** |

Not bit-identical because the atom-init key comes from a 3-way split (shards) rather
than w23's 2-way split — same distribution, different draw. **Determinism verified:**
an independent re-run of d=6 K=64 sharded seed 0 returned **0.8105**, identical to
four digits.

---

## 1. ⭐⭐ The §5.1 2×2 discriminator

`strict` is w23's metric, unchanged, with the routing decision folded in **as a
failure mode, not as an excuse**: the predicted item is (routed shard, nearest center
*within that shard*), so a routing miss costs exactly what a basin miss costs in the
monolithic arm. Bar = **0.90**. Two router columns, always quoted together:
**R2** = `argmin_r V_r(q)`, the *dynamical* pre-settle-energy score (the headline);
**RG** = distance to the nearest address the writer recorded — an admissible O(1)
statistic by the theorist's §3, **declared classical indexing**, whose route accuracy
is **1.000 in every cell and seed** (verified analytically: RG depends only on the
registry and the queries, not on the store), so `strict(RG)` ≡ the shard-supplied
("oracle") score.

| cell | arm | init | N | atoms tot / shard | vs w23 floor | **strict (R2)** | **strict (RG)** | route | blank |
|---|---|---|---|---|---|---|---|---|---|
| **d=6 K=64** ⭐ | monolithic *(laundering)* | scatter | 1 | 4096 / 4096 | ✓ | **0.8288 ± 0.0186** | — | 1.000 | — |
| | monolithic_nx *(laundering, 2×)* | scatter | 1 | 8192 / 8192 | ✓ | 0.8008 ± 0.0037 | — | 1.000 | — |
| | **sharded 2×32 (matched)** | scatter | 2 | 4096 / 2048 | ½ ✗ | 0.8434 ± 0.0253 | **0.9043 ± 0.0064 PASS** | 0.913 | 2/2 ✓ |
| | sharded 2×32 (per-shard) | scatter | 2 | 8192 / 4096 | ✓ | 0.8477 ± 0.0148 | **0.9150 ± 0.0115 PASS** | 0.908 | 3/3 ✓ |
| | monolithic *(laundering)* | **local** | 1 | 4096 / 4096 | ✓ | **0.8802 ± 0.0075** | 0.8802 | 1.000 | — |
| | **⭐ sharded 2×32 (matched)** | **local** | 2 | 4096 / 2048 | ½ ✗ | **0.9251 ± 0.0030 PASS** | **0.9479 ± 0.0012 PASS** | 0.970 | **3/3 ✓** |
| **d=4 K=32 ⭐control** | monolithic *(laundering)* | scatter | 1 | 2048 / 2048 | ✓ | **0.8346 ± 0.0151** | — | 1.000 | — |
| | monolithic_nx | scatter | 1 | 4096 / 4096 | ✓ | 0.8158 ± 0.0166 | — | 1.000 | — |
| | sharded 2×16 (matched) | scatter | 2 | 2048 / 1024 | ½ ✗ | 0.8118 ± 0.0193 | **0.8932 ± 0.0056 FAIL** | 0.884 | 1/1 ✓ |
| | sharded 2×16 (per-shard) | scatter | 2 | 4096 / 2048 | ✓ | 0.7786 ± 0.0341 | **0.8984 ± 0.0028 FAIL** | 0.851 | 2/2 ✓ |
| **d=8 K=64** | monolithic *(laundering)* | scatter | 1 | 8192 / 8192 | ✓ | **0.9121 ± 0.0029 PASS** *(2 seeds)* | 0.9121 | 1.000 | 2/2 ✓ |
| | sharded 2×32 (matched) | scatter | 2 | 8192 / 4096 | ½ ✗ | 0.8864 ± 0.0255 | **0.9224 ± 0.0083 PASS** | 0.952 | 3/3 ✓ |
| | sharded 4×16 (matched) | scatter | 4 | 8192 / 2048 | ¼ ✗✗ | 0.6559 ± 0.0143 | 0.8053 ± 0.0280 | 0.670 | — |

### 1.1 The reading — **THEOREM-L1**

`K_total = min( K_addr(Q, s, σ_q) , Σ_r K_write(unit r) )`.

* **d=6 K=64 — the wall IS fixed by sharding.** Two independent routes clear the bar,
  both blank-controlled at 3 seeds:
  * **with the shipped dynamical router**, once the N98 init defect is repaired:
    **0.9251 ± 0.0030** vs the monolithic laundering line at *identical init and
    identical total atoms* **0.8802 ± 0.0075** — Δ = **+0.045**, ≈6 sd, and it crosses
    0.90 where the monolithic does not;
  * **without any init change**, with the classical registry router: **0.9043 ±
    0.0064** vs monolithic **0.8288 ± 0.0186** — Δ = **+0.076**.
* **d=4 K=32 — the control is NOT fixed, and it fails even with routing removed, in
  BOTH sharded arms** (RG = 0.8932 ± 0.0056 matched, 0.8984 ± 0.0028 per-shard; both
  < 0.90, both blank-controlled). It is *geometry*-bound, exactly as registered.
  **Had it passed, the law would be `K = 32N` unclamped. It did not.**
* **d=8 K=64 — uninformative** (reconciliation 3): the monolithic store already
  passes here (**0.9121 ± 0.0029**, 2 seeds), so there is no wall to fix. The
  2×-starved matched shard arm scores 0.8864 ± 0.0255 (R2) / **0.9224 ± 0.0083 (RG,
  PASS)** — i.e. **level with, and under the classical router slightly above, a
  monolithic store carrying 2× its per-unit atoms.**
* **4×16 vs 2×32 (P3b) — falsified, but confounded.** 0.6559 vs 0.8864 at matched
  atoms; but at N=4 each shard holds **2048 atoms against a floor of 8192** and the
  write loss shows it (**7.3e-3** vs **3.7e-5** at N=2). Per N92, *a stall under an
  inadequate budget is not a ceiling.* Honest conclusion: **the parameter-matched
  split does not survive past N=2 at d=8**, not "shard size matters".

### 1.2 The decomposition that makes the result interpretable

The `RG − R2` gap is a *measurement of which term of Theorem L1 binds*:

| cell (scatter init) | R2 (dynamical route) | RG (classical route) | **routing cost** |
|---|---|---|---|
| d=6 K=64 2×32 matched | 0.8434 | 0.9043 | **0.061** |
| d=6 K=64 2×32 per-shard | 0.8477 | 0.9150 | **0.067** |
| d=4 K=32 2×16 matched | 0.8118 | 0.8932 | 0.081 |
| d=4 K=32 2×16 per-shard | 0.7786 | 0.8984 | 0.120 |
| d=8 K=64 2×32 matched | 0.8864 | 0.9224 | 0.036 |
| d=8 K=64 4×16 matched | 0.6559 | 0.8053 | 0.149 |
| d=6 K=64 2×32 matched, **local init** | 0.9251 | 0.9479 | **0.023** |

**At d=6, ~60% of the monolithic store's entire deficit from the bar is routing, not
writing** — and the N98 init fix cuts the routing cost by a factor 2.6 (0.061 →
0.023) as a side effect of putting each shard's atoms where its own items are.

### 1.3 What is genuinely new, in one paragraph (approved wording)

*The write is additive at zero optimizer cost.* Splitting 64 items across two units,
each written by its own dig, produces a store that retrieves at **0.925** where one
unit digging all 64 valleys reaches **0.880** — at the **same total atom count**,
with **half the atoms per unit**, and with **no optimizer synchronisation of any
kind**: the shards are separate parameter trees, so Prop L2's "N independent
optimizers" is not approximated here, it is *literal*. *And the read stays O(1) in
depth, because a classical O(N) score suffices to route:* one joint Verlet rollout
settles every shard simultaneously, routing costs **0.6–2.8%** of one read, and read
time grows **1.37×** from N=2 to N=8 (§3). **What sharding does not do is multiply
capacity as a dynamical result** — the routing half is nearest-neighbour indexing,
and on a *learned* store the dynamical routing score is the measured bottleneck.

---

## 2. Item 2 — the §5.3 read-parity check (Prop L4), on DESIGNED shards

No write at all, so the read side is isolated from the write ceiling under test in
item 1. d=4, K=32, 3 seeds, deadband 0. (RG is omitted from the table because its
route accuracy is 1.000 everywhere ⇒ RG strict = the oracle column = **1.0000 in every
row**.)

| allocation | N | union sep | within-shard sep | **R2 strict / route** | **R3 strict / route** |
|---|---|---|---|---|---|
| **global** | 1 | 0.710 | 0.710 | **1.0000 / 1.000** | 1.0000 / 1.000 |
| **global** | 2 | 0.710 | 0.715 | **1.0000 / 1.000** | 0.5495 / 0.549 |
| **global** | 4 | 0.710 | 0.787 | **1.0000 / 1.000** | 0.2363 / 0.236 |
| **global** | 8 | 0.710 | 0.851 | **1.0000 / 1.000** | 0.2279 / 0.228 |
| local | 1 | 0.710 | 0.710 | 1.0000 / 1.000 | 1.0000 / 1.000 |
| local | 2 | **0.192** | 0.886 | 0.9935 / 0.993 | 0.8984 / 0.898 |
| local | 4 | **0.212** | 1.145 | 0.9863 / 0.986 | 0.4401 / 0.440 |
| local | 8 | **0.198** | 1.232 | 0.9779 / 0.978 | 0.2220 / 0.222 |

* **P5 (global-allocator parity): CONFIRMED exactly.** `max |Δ vs monolithic| =
  0.0000` at N = 2, 4, 8 with R2. **Prop L4 stands — sharding conserves read-side
  discrimination and there is NO unmodelled cross-shard channel.** (This was the item
  the task said to escalate on. Nothing to escalate.)
* **P6 (local allocation degrades): shape CONFIRMED, magnitude FALSIFIED.** Union
  separation collapses as predicted (0.710 → 0.192 / 0.212 / 0.198) and strict
  declines monotonically — but only **1.000 → 0.978**, not the theorist's
  0.92 / 0.73 / 0.62 / 0.55. Reason: his store was analytic-Gaussian settled by
  gradient flow; the shipped `BallRegisterPotential` has *narrow* wells (w = 0.15)
  relative even to the collapsed union separation (0.19), so a pre-settle energy
  score still resolves them. **The qualitative conclusion (the global allocator is
  required) survives; his numbers do not transfer.** The *allocator-level* evidence
  is much starker — see §4.
* **P7 (R1 not shipped): CONFIRMED** — `router_scores("R1", …)` raises with the N97
  reason; `test_r1_router_is_not_shipped`.
* ⛔ **NEW NEGATIVE — R3 is broken in the shipped geometry** (reconciliation 1).
  Mechanism, pinned by a cheap unit test rather than asserted: at item 0's site the
  **owning** shard has `V = −0.991`, gradient norm **2.9e-4** (it is a minimum), and
  the **foreign** shard has `V = −8.0e-5`, gradient norm **2.3e-3**. The energies
  differ by **4 orders**; the forces that drive the displacement are *both ≈0 and
  within an order of magnitude of each other*. R2 has a huge signal; R3 has none.

---

## 3. Item 3 — the cost claim (the O(1)-in-depth half)

d=6, K=64, designed shards, 64 queries, median of 3 repeats, jit warmed, **machine
essentially unloaded** (an earlier loaded measurement is retained in the artifacts and
gave the same ratios to within 3×).

| N | R2 routing (ms/query) | one two-phase read, 1200 Verlet steps (ms/query) | route / read |
|---|---|---|---|
| 2 | 0.0175 | 2.78 | **0.63%** |
| 4 | 0.0553 | 4.54 | 1.22% |
| 8 | 0.1072 | 3.80 | **2.82%** |

* **P8 (routing ≤5% of one read at N ≤ 8): CONFIRMED** — 0.63–2.82%. Routing time
  scales ≈linearly in N (0.0175 → 0.107 ms for N: 2 → 8) exactly as the O(N)-score
  account says.
* **P9 (read does not grow ∝N): CONFIRMED** — **1.37×** from N=2 to N=8 (the sweep
  max/min is 1.63×, at N=4). `V` is separable, so ONE joint `CLULattice` rollout
  settles all shards simultaneously: the total atom count costs, `N` does not.
* The design rule this rests on is now **enforced in code**: the routing statistic
  must be evaluable **without running the dynamics**. R2 and RG are; R1 is not — and
  R1 is also the one that does not work.

---

## 4. Item 4 — the global allocator, as a registry (build item 3)

24 items offered round-robin to 4 MVC-0 controllers on a radius-2 disk, `d_safe =
4.4·s = 1.54`, one proposal stream replayed identically in both regimes.

| regime | admitted | **union separation** | within-shard sep | union respects `d_safe`? |
|---|---|---|---|---|
| **global registry** (spacing test vs the UNION) | 4 / 24 | **1.698** | 1.698 | **yes** |
| per-shard registries (control) | 14 / 24 | **0.208** | 1.653 | **no** |

The per-shard regime admits 3.5× more items and produces a store whose addresses are
**7.4× too close to be told apart** — every shard's own gate is satisfied and the
union is unretrievable. This is the sharp version of Prop L4: the read-side condition
is on the **union**, and only a global registry can enforce it. And it is a *registry,
not an optimizer*: no gradient and no optimizer state crosses a shard boundary, only
the list of where things were written. *(Scope: `admit_site` ran without a relocation
proposer, so the global gate here refuses rather than relocates; the measured contrast
is the point, not the 4/24 rate. The disk's own bound is `N_pack = πR²/((√3/2)d_safe²)
= 6.1`, so 4 admits from random proposals with no relocation is in line with N74's
6.0 ± 0.9.)*

---

## 5. Item 5 — the N98 localized-init ablation (build item 2) ⭐

Applied **identically to both arms**, 3 seeds, d=6 K=64. Radius `2s = 0.6`, **address
axes only**.

| arm | scatter init (w23) | **localized init (N98 fix)** | Δ |
|---|---|---|---|
| monolithic (4096 atoms) | 0.8288 ± 0.0186 | **0.8802 ± 0.0075** | **+0.051** |
| sharded 2×32 (4096 atoms) | 0.8434 ± 0.0253 | **0.9251 ± 0.0030 PASS** | **+0.082** |
| — R2 route accuracy (sharded) | 0.913 | **0.970** | +0.057 |
| — payload abs err (sharded) | 0.1345 | **0.0678** | −50% |

* **P10 CONFIRMED and then some** (registered ≥+0.02 monolithic / ≥+0.01 sharded;
  measured +0.051 / +0.082).
* Two design choices are load-bearing and deliberate: **(a) default OFF** and
  bit-identical to the historical construction — the localized draw uses
  `jax.random.fold_in(key, 1)`, so even the RNG stream of the default path is
  untouched (`test_localized_atom_init_is_bit_identical_when_disabled`);
  **(b) address axes only** — the payload axis keeps the w23 `init_scale = 1.0`
  scatter, because localizing it would hand the writer the value it is supposed to
  learn (N46) *and* destroy the measured basin-reach property that makes
  `init_scale = 1.0` load-bearing (a well initialised at payload 0 cannot reach
  `|a_i| = 1`).
* ⚠ **It is an initialisation, so it is a finding about w23's init, not about
  sharding** — which is exactly why it was pre-registered as a separate axis and kept
  out of the headline 2×2. Its bearing on the **N92/N96 contested mechanism** is
  reconciliation item 2: some of what reads as a "write-operator ceiling" is atoms
  starting inside other items' wells. Note it does **not** on its own lift the
  monolithic cell over the bar (0.880 < 0.90); **sharding + the init fix does**
  (0.925).

---

## 6. Item 6 — the abstention deadband (secondary; never a headline)

Designed 8-shard store, d=4 K=32, R2. Per-offered and per-answered always travel
together (N91).

| deadband | strict per-**offered** | strict per-**answered** | abstain rate |
|---|---|---|---|
| 0.00 (headline) | 1.000 | 1.000 | 0.000 |
| 0.01 / 0.05 | 1.000 | 1.000 | 0.000 |
| 0.20 | 0.992 | **1.000** | 0.008 |
| 1.00 | 0.000 | 0.000 | 1.000 |

Behaves as designed (it trades offered for answered) but has nothing to buy on a store
that is already exact. Its value is on the *learned* store, where a routing miss is
total — the wrong shard answers with full confidence, and value accuracy tracks route
accuracy 1:1 in every row of §2. Not exercised on the learned cells (compute, §8).
**The headline used deadband 0 throughout, so no number in this report takes
abstention credit.**

---

## 7. The build — the theorist's five points, and where each lives

| # | build item | where | trap closed |
|---|---|---|---|
| 1 | `CLULattice(units, edges=(), couplings=())` as the shard container; **never** one wide relativistic CHLU | `core/shard_store.py: ShardedStore, build_sharded_store` | §2.3 #6/#7/#8 — a coupled lattice **raises**; `kinetic_mode="relativistic"` **raises** (not a docstring warning); per-unit `T` has **exactly 0.0** off-block second derivative (tested) |
| 2 | localized atom init per group (N98) | `core/memory_potentials.py: AtomDictionaryPotential(group_centers, local_radius)` + `_uniform_ball`, `_atom_group_owner`; plumbed via `DesignFreedomPotential(atom_group_centers, atom_local_radius)` | §2.3 #5 / N98 — default OFF & bit-identical; address axes only (N46) |
| 3 | global address allocator in `Controller` | `core/shard_store.py: ShardedRegistry` + `Controller(peer_addresses_fn=…)` (a **6-line additive hook**) | Theorem L1 condition W4; the ONLY global object; registry, not optimizer |
| 4 | routers **R2** and **R3** + top-2 abstention deadband; ⛔ **no R1** | `core/shard_store.py: r2_scores, r3_scores, rg_scores, route_from_scores, router_scores` | N97 — `"R1"` raises *with the reason*. ⚠ **R3 measured broken here** (§2) |
| 5 | per-shard query noise | `core/shard_store.py: assert_per_shard_query_noise`, called on **every scored cell** | §2.3 #10 — `σ/√(N·d)` raises with "fairness trap" |

Two additions beyond the list, both forced by the measurements and both labelled:
* **`RG`, the registry router** — explicitly sanctioned by the theorist's own O(1)
  condition ("pre-settle energy, **distance to the registry**, or an explicit tag").
  Declared classical (N89) wherever quoted. It exists because it separates *a store
  that cannot hold its items* from *a score that cannot tell which unit holds them*.
* **`strict_oracle_route`**, an explicitly-labelled **oracle diagnostic** (the same
  read scored with the true shard supplied). Never a headline — the w24
  masked-NN-oracle rule.

**Side finding, free and real: the sharded write is 4–5× cheaper at the same total
atom count.** d=6 K=64 end-to-end: monolithic **1900 s**, sharded 2×32 **412–492 s**.
`write_loss`'s barrier term is all-pairs, `O(K²)`; splitting into N shards makes it
`O(K²/N)`. **Sharding buys wall-clock additivity even where it does not buy capacity.**

---

## 8. Declared deviations — what did NOT get measured, and why

This harness costs ~700 s of *write* per cell at d=6 (600 Adam steps over 4096 atoms ×
64 items) and ~2× at d=8, on 8 CPU cores with no GPU, with each worker saturating ≈1
core (the read is a sequential 1200-step `lax.scan`). All of the following are compute
cuts, declared, not results:

1. **d=8 K=64 `monolithic` at 2 seeds** (its sharded arms have 3). That cell turned
   out to be uninformative anyway (reconciliation 3).
2. **d=8 `monolithic_nx` and `sharded_per_shard`** (16 384 / 32 768 atoms): not
   affordable. So the **budget-adequate** test at d=8 — the one that would tell us
   whether 4×16 fails from starvation or from sharding — is **not done**, and P3b's
   falsification is therefore reported as confounded rather than clean.
3. **d=8 K=256, 8×32 (P4): not run.** Registered as an *expected-fail* probe of where
   geometry takes over; its absence removes a confirmation, not a claim. (RG route
   accuracy there is 1.000 and union separation 0.714 — the prediction "FAIL,
   0.55–0.85, geometry-bound" is untested.)
4. **The init ablation was run only at d=6 K=64** and only for `monolithic` +
   `sharded_matched`.
6. **Mid-run code addition (provenance).** `strict_oracle_route`, `by_router` and
   `RG` were added *after* the first cells ran. The addition is **purely additive** —
   no primary metric changed — and this was verified by re-running d=6 K=64 sharded
   seed 0 under the new code: strict **0.8105**, identical to four digits. Rows
   without an RG column simply predate it. Recommended as a standing rule for
   additive diagnostics.
7. **The blank control is `if_pass`-gated** (the w24 rule) and its trigger now fires
   on the **best** router, so every PASS in the table carries its own leak-immunity
   control (`blank` column). Failing cells legitimately have none.

---

## 9. PREREG scorecard

| # | registered prediction | outcome |
|---|---|---|
| P1 | d=6 K=64 2×32 **PASS ≥ 0.90** | ⭐ **CONFIRMED** — 0.9043 ± 0.0064 (RG, scatter init) and **0.9251 ± 0.0030 (R2, the shipped dynamical router, with the N98 init)**, both blank-controlled. **FALSIFIED** in the one configuration I registered as headline (R2 + scatter init, 0.8434) — the pair must always be quoted together |
| P2 | d=4 K=32 2×16 **still FAILS ≤ 0.87** | **CONFIRMED** on the registered metric (0.8118 / 0.7786), and stronger than registered on mechanism: it fails at **0.8932 (matched) / 0.8984 (per-shard) with routing removed entirely** ⇒ geometry-bound, not router-bound. ⚠ Registered band was "≤0.87"; the RG numbers sit at 0.89–0.90, i.e. *just* under the bar — closer than I predicted, and worth a 5-seed re-check before the band is quoted |
| P3a | d=8 K=64 2×32 **PASS**, Δ ≥ +0.03 vs monolithic | **PASS confirmed (0.9224 ± 0.0083 RG, 3 seeds), Δ FALSIFIED (+0.010)** — because the **monolithic control also passes** (0.9121 ± 0.0029, 2 seeds): the cell is not a wall |
| P3b | 4×16 **PASS** and ≥ 2×32 | **FALSIFIED** (0.6559) — **confounded by a 4× atom-floor violation**; reported as "the parameter-matched split does not survive past N=2 at d=8" |
| P4 | d=8 K=256 8×32 FAIL 0.55–0.85 | **not run** (§8.3) |
| P5 | global-allocator parity, \|Δ\| ≤ 0.02, route ≥ 0.98 | ⭐ **CONFIRMED exactly** — Δ = **0.0000**, route 1.000 at N = 2/4/8 |
| P6 | local allocation degrades, reproducing 0.92/0.73/0.62/0.55 | **shape CONFIRMED, magnitude FALSIFIED** (1.000 → 0.978); his numbers do not transfer to the shipped store. The allocator-level contrast (§4) is where the effect is stark |
| P7 | R1 absent from `ROUTERS`; raises | **CONFIRMED** |
| P8 | routing ≤ 5% of one read at N ≤ 8 | **CONFIRMED** — 0.63–2.82% |
| P9 | read grows ≤ 1.5× from small N to N=8 | **CONFIRMED** — **1.37×** (sweep max/min 1.63× at N=4) |
| P10 | localized init helps both arms, ≥ +0.02 / +0.01 | ⭐ **CONFIRMED, 2–4× larger than registered** — +0.051 monolithic, +0.082 sharded, and it repairs the router (0.913 → 0.970) |

**Unregistered finding (declared as such): R3 is a broken router in the shipped
geometry** (§2). Not pre-registered because the theorist reported R3 exact at 1.000;
the falsification is *his own prediction failing on the real store*, and after the 2×2
it is the most consequential single measurement here.

---

## Git footprint

* **Branch:** `agent/experiment-engineer/lattice-sharded-store` (worktree
  `../CHLU-shard`), base local `main` @ `63c668d`. **Not pushed; left for review.**
* **Commits (5, head `29e2924`):** `5cdce3b` · `d372089` · `cee0854` · `500b5d5` ·
  `29e2924`. `git diff --stat main..<branch>` = **7 files, +2526 / −1**.
  Branch ref **verified from the MAIN repo** (protocol §3.2). `main` untouched at
  `63c668d`; my working tree is clean. **Worktree `../CHLU-shard` LEFT IN PLACE** for
  review (consistent with `cl-entry-build`, still live) — remove it at integration,
  after re-checking the ref from the main repo.
* **Files touched** — new: `chlu/core/shard_store.py`,
  `chlu/experiments/exp_sharded_store.py`, `tests/test_sharded_store.py`; modified:
  `chlu/core/memory_potentials.py` (two additive kwargs on `AtomDictionaryPotential`
  and `DesignFreedomPotential` + two module-level helpers),
  `chlu/core/controller.py` (**6 lines** — one optional `peer_addresses_fn` kwarg and
  its use in `offer`), `chlu/config.py` (**one new group**, registered at all four
  sites incl. `save_config`), `chlu/cli/experiment_cmd.py` (one new subcommand).
* **Parallel-safety:** I deliberately did **not** touch
  `ExperimentDesignedMechanismConfig` or `exp_designed_mechanism.py` —
  `r2-geometry-revival` is editing `atom_init_width` there this wave. Every knob of
  mine lives in the new `experiment_sharded_store` group. `controller.py` and
  `memory_potentials.py` may also be touched by `cl-entry-build`; my edits are
  additive kwargs at distinct sites and should merge cleanly.
* **Rebase onto local `main`:** no-op (base == `main`, unmoved). No conflicts.
* **Verification (all run on this branch, in the worktree, main venv):**
  * `ruff check chlu/ tests/` → **All checks passed**
  * `pytest tests/` (FULL SUITE) → **682 passed, 0 failed** in 810 s
    = the 657 the Hub last recorded **+ the 25 new sharded-store tests**, with no
    regression anywhere else.
  * `pytest tests/test_sharded_store.py` → **25 passed**
  * `python -m chlu.experiments.exp_sharded_store --quick` → the full pipeline green
    end-to-end, all six items (numbers meaningless at 30 write steps, by design).

---

## Open questions / follow-ups / risks

1. **Is the dynamical R2 router fixable on a learned store?** It is 1.000 on a
   *designed* store at N=8 but 0.91 (d=6, scatter) / 0.97 (d=6, local init) / 0.67
   (d=8, N=4, starved) on learned ones — so the score is not wrong in principle, the
   *learned* `V_r` is a poor nearest-well proxy, and the N98 init already fixes most
   of it. Cheap next steps: (a) score the deepest single atom rather than the summed
   `V_r`; (b) an explicit learned tag. Only these keep the routing claim dynamical;
   RG does not.
2. **The atom floor is per-store** (reconciliation 4). Every sharded design pays
   ~N× parameters just to stay adequate. Any R2 figure must say so: **additivity
   costs ~N× parameters** unless the floor is re-derived per shard. The *budget-adequate*
   arm (`sharded_per_shard`) also passes at d=6 (0.9150 RG), so this is a cost, not a
   blocker.
3. **The two w25 R2 routes now interact.** The N98 init fix narrows the *effective*
   atom placement; `r2-geometry-revival`'s `atom_init_width 0.30 → 0.15` narrows the
   *width*. Both attack "atoms in the wrong place" from different sides and both are
   large effects. **They should be measured jointly before either is promoted to a
   default** — the combination could easily be the biggest lever of the wave, or they
   could be the same effect twice.
4. **Risk to the claim:** the whole 2×2 lives on one synthetic geometry with
   *supplied* placement (N46 applies as always — the writer is given the sites).
   Nothing here says items find their own addresses; nothing here beats a kNN.
5. `spread_partition`'s best-of-four rule matters at low `d` (0.790 vs 0.441 at d=2
   K=16 N=4) and is immaterial at the deciding cells (<1% at d ≥ 6). The winning rule
   is recorded per cell in the metrics JSON (`greedy_balanced` at d=6 K=64 N=2).

---

## Proposed handover updates (for the Hub)

1. **§1 / theory — Theorem L1 CONFIRMED, with its binding term identified.**
   `K_total = min(K_addr, Σ_r K_write)`. The d=6 K=64 wall is fixed by sharding
   (0.9251 ± 0.0030 with the shipped router + the N98 init, vs a monolithic
   laundering line of 0.8802 ± 0.0075 at identical init and atoms; or 0.9043 vs
   0.8288 with a classical registry router and no init change) while the
   geometry-bound d=4 K=32 control is **not** fixed even with routing removed
   (0.8932) ⇒ **the ceiling is not entirely per-dig and `K = 32N` is NOT unclamped.**
   **Approved wording:** *"the write is additive at zero optimizer cost, and the read
   stays O(1) in depth, because a classical O(N) score suffices to route"* — plus the
   new half: **at d=6 roughly 60% of the monolithic store's deficit is routing, not
   writing.** ⛔ Never "capacity multiplies by sharding" as a dynamical claim.
2. **§7 / N97 — EXTEND (curator).** Post-settle energy (R1) is not the only broken
   router: **settling displacement (R3) is broken too on the shipped geometry** —
   1.000 / 0.549 / 0.236 / 0.228 at N = 1/2/4/8 vs R2's flat 1.000 — because
   `BallRegisterPotential` is flat inside the address ball, so the forces that drive
   the displacement are ≈0 in *every* shard while the energies differ by 4 orders.
   **Only R2 and the classical registry survive.** Tier A (it overturns a shipped
   recommendation and imposes a design rule).
3. **§7 / N98 — PROMOTE from "design guard" to a first-order lever.** Built, tested,
   ships OFF, bit-identical when disabled. Measured: **+0.051 monolithic / +0.082
   sharded at d=6 K=64, 3 seeds**, and it repairs the dynamical router (0.913 →
   0.970). It is an *initialisation*, so it **bears on N92/N96's contested
   mechanism** — some of the apparent "write-operator ceiling" is atoms starting
   inside other items' wells. Should be co-reviewed with `r2-geometry-revival`'s
   width result before either becomes a default (follow-up 3).
4. **§7 / NEW trap — the w23 atom floor `512·√2^d` is per-STORE, not per-item**, so a
   parameter-matched N-way split is automatically N× below it. Every `sharded_matched`
   arm here is 2–8× starved ⇒ its wins are conservative and its N ≥ 4 failures are
   **not** capacity results (N92 protocol).
5. **§10 / d=8 K=64 is no longer a wall** — my in-run monolithic control is
   **0.9121 ± 0.0029 PASS** (2 seeds), consistent with w24's post-hoc
   0.9067 ± 0.0068. Any task file or table still
   quoting "d=8 K=64 = 0.883 FAIL" as a live baseline should be corrected.
6. **Candidate negatives-registry entries** (curator to word, tiers to the Hub):
   **N-a** ⛔ *settling-displacement routing (R3) is broken on a flat-vacuum store* —
   tier **A**, supersedes half of N97's recommendation;
   **N-b** ◐ *sharded additivity is real on the write side and is partly spent on the
   read side* (routing costs 0.023–0.081 strict) — tier **A**, it is the R2-route-ii
   verdict;
   **N-c** ◐ *the d=4 K=32 wall is geometry, confirmed with routing removed*
   (0.8932 / 0.8984, both arms, blank-controlled) — tier **B**, a scope guard on any
   "K = 32N unclamped" reading; ⚠ it sits just under the bar, so quote it as
   "0.89–0.90, does not clear 0.90", not as a comfortable failure;
   **N-d** ◐ *the w23 atom floor is per-store, so parameter-matched sharding is
   automatically budget-starved* — tier **B**, a design guard with its arithmetic.
7. **Independent reproductions worth recording:** the w23 "more atoms make d=6 K=64
   *worse*" anomaly reproduces (0.8288 @4096 → 0.8008 @8192, 3 seeds each), and the
   w23/w24 monolithic baselines reproduce within seed noise at d=4, d=6 and d=8.
8. **Process:** mid-sweep *additive* instrumentation (the oracle/`by_router`/RG
   columns) was made safe by re-running one cell and matching to four digits.
   Recommend adopting that as the standing rule.
