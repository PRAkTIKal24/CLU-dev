# lattice-capacity-theory — physics-theorist report

**Task + acceptance criterion:** formalize when sharding items across `N` units multiplies capacity without optimizer synchronization (the write-locality claim), enumerate the failure modes, deliver the **read-cost verdict**, and give ≥1 concretely-sized falsifiable w25 prediction — with numerical checks that are allowed to refute me.
**Status: done.** No production code touched. Branch: none.

> ### ⛔ DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Three items, one of them touches a tier-A registry entry.
> 1. **⭐ N92's "d-INDEPENDENT WRITE ceiling ≈32" has a competing, more parsimonious geometric account that must be tested before the phrase is quoted again.** At *every* measured wall in the w23 sweep — designed **and** learned, d = 2…8 — the critical site separation sits at **≈2.4–3.0 × the well width** (`checkF`, real `designed_sites` allocator, §4.2 table). Because `minsep ∝ K^{-1/d}` flattens in high `d` (at d=8 a 32× change in K moves minsep only 1.7×), **one fixed width threshold in a fixed-radius ball produces a nearly K-independent wall — which reads exactly like a "d-independent ceiling."** The learned/designed capacity tax is then `(s_designed/s_learned)^d = 2^{-d}` from the shipped flags `well_width=0.15` vs `atom_init_width=0.30` — and **w23 measured the tax as exactly 1/4, 1/8, 1/16 at d = 2, 3, 4.** This is EXPLORATORY (not pre-registered; §5 gives the test). ⚠ Do **not** re-word N92 on my say-so — commission the width measurement in §5.0 first; it is ~free.
> 2. **`write-ceiling-break` Item 2 as specified will not test this.** Its scale-invariance ablation rescales **depths / margins / barrier**; the geometric account is about **widths and radius**. If that task is still in flight, the cheap amendment is one extra arm: `atom_init_width ∈ {0.30, 0.15}` and `designed_sites(..., R ∈ {1, 2})` at d = 4 and d = 6. Owner: whoever reviews `write-ceiling-break`.
> 3. **A fairness trap for any w25 sharding experiment:** the retrieval query uses `fixed_norm` jitter **σ/√d_total per axis**. If shards are implemented as coordinate blocks of ONE wide unit, `d_total = N·d` and the per-shard query noise silently *shrinks* with N — the sharded arm gets a free noise advantage. Query noise must be fixed **per shard**. Owner: the w25 engineer.

---

## Flag-provenance table (governs every number I produced)

| item | value |
|---|---|
| repo state | `main @ 5e466c0` (612 green), **no code modified, no branch created** |
| python / jax / equinox / optax | 3.11 / **0.9.0** / 0.13.4 / 0.2.6 — **main venv reused** (protocol §4); invoked as `PYTHONPATH=/Users/user/Desktop/CHLU .venv/bin/python …` (§7.12 editable-install bug) |
| scripts | `.claude/scratch/lattice-capacity-theory/{checkA_crosstalk_budget, checkB_factorization, checkC_barrier_dilution, checkD_read_routing, checkE_width_tax, checkF_separation_law}.py` |
| PREREG | `.claude/outputs/lattice-capacity-theory/PREREG.md` — **6 registered predictions, 2 confirmed, 1 refined, 3 REFUTED** (all three of mine). Written before the scripts ran; the §4.2 separation finding is declared EXPLORATORY. |
| seeds | `checkA` rng(0), 20 trials/cell · `checkB` PRNGKey(0), single seed (a bit-identity test — a second seed adds nothing) · `checkC` PRNGKey(0) · `checkD` seeds 0,1,2 × 8 queries/item · `checkE/F` rng(0)/seed 0 (deterministic allocator) |
| write path exercised | real `chlu.training.train_memory.write_loss`, real `AtomDictionaryPotential(n_groups=K)`, real `optax.adamw(3e-3, wd 1e-4)`, mask semantics copied verbatim from `atom_write_mask_fn` (the shipped one reaches through `.learned`; a bare dictionary needs the same map — see `checkB` note) |
| write hyperparameters mirrored from w23 | `n_perturb 16/32`, `σ_addr 0.25`, `σ_pay 0.6`, `margin 0.15`, `barrier 0.2`, `payload_index=d`, `init_scale 1.0`, `init_width 0.3`, `depth_init 1e-4`, `confine 0.05`, 200 Adam steps (vs w23's 600 — a *shorter* write; noted where it matters) |
| toy-store constants (`checkA/D/E`) | `s=0.35`, `α=0.02`, `d_safe=4.4s=1.54`, `κ` n/a (payload read modelled as the Gaussian profile), `R=0.5249·d_safe·√K` (= controller-mvp's `R=0.808√K`) |
| numbers inherited, not re-measured | w23 masked/global corruption **5.5e-5 / 1.3e-4** vs **0.47 / 0.46** (`dimension-aware-budget` §4 item 3) · `K_learned = min(2^d, ~32)` and the tax row 1/4,1/8,1/16,1/16,1/8,1/8 (§1/§2 ibid.) · read 0.70 ms / gate 0.006 ms / relocation 3.1 ms (`controller-mvp` §3d) · `N_pack = πR²/((√3/2)d_safe²)`, `R=0.808√K` (`controller-mvp` §1) |
| langevin_noise | **N/A** — everything here is deterministic (gradient flow / Verlet / static write) |

---

## 1. Item 1 — the formal claim, and exactly what "write-locality" has to mean

### 1.1 Definitions

A **store** is a potential `V_Θ : Q → ℝ` on an address space `Q`, plus a retrieval map
`R : q_query ↦ value` implemented as *settle-then-read* (`γ>0` relax to the address, then roll to read the payload channel). A **write operator** is a map `W : (Θ, S) ↦ Θ′` for an item set `S`.

Three architectures must be kept apart — they have *different* capacity laws and the program has been using one word ("lattice") for all three:

| arch | address space | parameters | what CLU code implements it |
|---|---|---|---|
| **(A) product store** | `Q = Q_1 × … × Q_N`, `q = (x^{(1)},…,x^{(N)})`, `V(q) = Σ_r V_r(x^{(r)})` | disjoint | `CLULattice(units, edges=(), couplings=())` |
| **(B) mixture store** | ONE shared `Q ⊆ ℝ^d`, `V(q) = Σ_r V_r(q; θ_r)` | disjoint | `AtomDictionaryPotential(n_groups=N)` + `atom_write_mask_fn` — **already shipped** |
| **(C) independent stores** | `N` separate CLUs queried separately | disjoint | nothing; would be an outer loop |

### 1.2 The locality conditions

- **W1 (parameter-support disjointness).** `∂V/∂θ_r` depends on `θ_r` only, i.e. `V` is a *sum* of per-unit terms. ✅ satisfied by (A) and (B) with the atom family. ❌ **violated by every MLP family** (`free_mlp`, `DeepPotentialMLP`): one hidden layer couples all of `Q`, so *no* sharding statement is available for MLP-family learned potentials. **Sharding is a property of local-support bases, not of CLU.**
- **W2 (update-support disjointness = the masked write).** Writing item `m ∈ S_r` applies `Δθ_j = 0 ∀ j ≠ r`. This is **strictly stronger** than W1: with a shared address space the *gradient* `∂L/∂θ_j` is non-zero for `j ≠ r` (item `m`'s loss sees the whole `V`), so only masking the **updates** achieves it. The shipped `atom_write_mask_fn` masks updates precisely because AdamW's decoupled weight decay would otherwise still shrink frozen rows — that docstring is load-bearing and correct.
- **W3 (loss-gradient separability).** `∂L(S_r)/∂θ_r` is independent of `θ_j`, `j≠r`. **This is what actually fails**, and it fails for two different reasons of very different size (§1.4).
- **W4 (global allocation).** The addresses `{c_m}` across *all* units are pairwise separated by ≥ the critical separation. Not a write property at all — a **placement** property.

### 1.3 Theorem L1 (additivity), with its conditions

> **Under W1 ∧ W2 ∧ W3 ∧ W4**, the sharded store's strict-retrieval capacity is
> `K_total = min( Σ_r K_write(unit r) , K_addr(Q, s, σ_q) )`
> where `K_write` is the per-dig write ceiling (w23: ≈32, and *shared* by every unit because it is a property of the operator, not the parameters) and `K_addr` is the address-space capacity at well width `s` and query noise `σ_q`.
> Consequently `K_learned(d) = min(2^d, K_ceiling)` becomes `min(K_addr(d), N·K_ceiling)` — **sharding does not raise capacity, it relocates the binding constraint from the write operator back onto geometry, and can never exceed geometry.**
>
> *Status:* **proven** given the conditions (it is bookkeeping: W1+W2 make the writes non-interfering, W3 makes them independent, W4 makes them jointly retrievable; the `min` is because a written-but-unretrievable item is not capacity). The content is entirely in *which conditions the code satisfies* — §1.4, §2.

### 1.4 Proposition L2 — there really is nothing to synchronize (the advisor's claim, proven and then measured)

> **AdamW is an elementwise map on parameters.** Its state (`m`, `v`) is per-parameter, so under W1∧W2 the optimizer state factorizes exactly into blocks. If additionally W3 holds, then *N independent optimizers* and *one masked joint optimizer* produce **bit-identical** parameters after any number of steps. **The Head's worry — "chaining optimizers / mapping disconnected optimizer spaces" — has no referent: a masked write on a block-diagonal problem IS N independent optimizers.** *Status: proven for the map; the only question is W3.*

**Measured (`checkB`, real write path, K=4 items, 4 atom groups × 32 atoms, sites 1.414 apart, 200 Adam steps):**

| quantity | joint-masked vs N-independent |
|---|---|
| loss trajectories | 0.1735→**0.00617** vs 0.1736→**0.00617**; 0.1724→0.00493 vs 0.1712→0.00501 |
| `centers` | max‖Δ‖ **6.50e-3**, rel **2.4e-3**, 223/384 entries differ |
| `log_width` | max‖Δ‖ **1.29e-2**, rel **8.7e-3**, 73/128 differ |
| `amp` | max‖Δ‖ **8.23e-3**, rel **1.4e-2**, 101/128 differ |
| `V` at the stored sites | joint `[−0.344, −0.559, −0.529, −0.254]` vs indep `[−0.344, −0.560, −0.526, −0.251]`; max Δ **2.9e-3** (≈0.5% of well depth) |
| CONTROL: same loop **without** the mask | **128/128 atoms move**, max Δamp **0.725** |

So: **W3 does not hold exactly, and — my pre-registered prediction P3 FAILED — the residual is ~3 orders larger than the site tail** `exp(−1.414²/2·0.3²) = 1.5e-5`. Cause, found post-hoc and it is an **initialization defect, not physics**: `init_scale=1.0` scatters *every* group's 32 atoms over the whole ball, so group `j` has atoms sitting inside item `i`'s well from step 0. The write for item `i` must compensate for foreign atoms; a shard-independent write cannot. **Engineer fix (precise): initialise group `j`'s atoms in a ball of radius ~`2s` around item `j`'s site (or spatially partition the atom initialisation).** Then W3 holds to the site tail and Prop L2 becomes exact in practice.

Residual budget with the defect left in: if the 1.4% per-write deviations accumulate incoherently (§1.5 shows crosstalk does), `√N · 1.4% < 10%` ⇒ **N ≲ 50 shards** before the init defect alone costs a tenth of the well depth. That is not the binding constraint for anything we plan.

### 1.5 Proposition L3 — a *finite* locality ratio degrades additivity only incoherently (my own bound was too pessimistic)

Because `V` is an exact **sum** of atoms, the perturbations that foreign items impose on item `A`'s minimum superpose exactly at first order; the displacement is `δx_A = −H_A^{-1} Σ_{m≠A} ∇V_m(c_A)`. Whether that sum grows like `m` or `√m` depends on the *directions*.

**Measured (`checkA` A1, foreign items on a shell at exactly `d_safe`, 20 trials):**

| d | m=1 | 2 | 4 | 8 | 16 | 32 | √m law |
|---|---|---|---|---|---|---|---|
| 2 | ‖δx‖ 9.59e-5 | 1.17e-4 | 1.81e-4 | 2.51e-4 | 3.65e-4 | **5.34e-4** | √32·9.59e-5 = **5.43e-4** (1.6% off) |
| 4 | 9.59e-5 | 1.27e-4 | 1.70e-4 | 2.42e-4 | 3.37e-4 | **5.75e-4** | same |

⇒ **accumulation is incoherent (`√m`), not linear** (P1 REFINED). Converting w23's measured per-write corruption `ε` into an additivity budget with `√K·ε < tol = 0.1`:

| write | ε (w23) | `K_max = (tol/ε)²` |
|---|---|---|
| masked, d=2 | 5.5e-5 | **≈3.3 × 10⁶** |
| masked, d=4 | 1.3e-4 | **≈5.9 × 10⁵** |
| global, d=2/4 | 0.47 / 0.46 | **< 1** |

> **Verdict on the task's question ("does a finite locality ratio degrade additivity?"): yes, but by a factor that is *five orders* from binding.** The measured 8474×/3434× locality advantage is not "very local but finite, so watch out" — it is *operationally infinite* for any K this program will reach. Conversely the **global** write's cross-corruption already exceeds the read tolerance at K = 2, which is a clean restatement of why one joint dig cannot hold many items. *Status: strongly evidenced (√m law verified to 2%, budget arithmetic elementary; the ε values are inherited single-config measurements).*

---

## 2. Item 2 — the failure modes (I attacked my own claim; three of six attacks landed)

### 2.1 Read-side crosstalk — Proposition L4 (**read-side discrimination is CONSERVED by sharding**)

> Sharding relaxes the **write-side** crowding (each unit holds `K/N` items, so within-unit separation grows) but does **not** relax the **read-side** discrimination problem: a content query must still be resolved against the *union* of all `K_total` addresses. Formally, for the product store with a broadcast query, the decision is `argmin` over the union of wells; for the mixture store the read is *literally identical* to the monolithic read (`V` is the same sum regardless of how atoms are grouped — grouping is a fiction of the optimizer, invisible to the physics).
> **Corollary: `K_addr` in Theorem L1 is untouched by `N`. Sharding is a write-side device only.** *Status: proven for (B) (identical `V`), proven-modulo-router for (A).*

**Measured (`checkD`, d=2, geometry sized to the load `R = 0.808√K`, 3 seeds × 8 queries/item, σ_q = 0.10):**

| config | K | sep within shard | sep across union | R1 route/value | **R2** route/value | **R3** route/value | union read |
|---|---|---|---|---|---|---|---|
| monolithic K=8 | 8 | 1.605 | 1.605 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000 |
| 2×4 GLOBAL alloc | 8 | **1.955** | 1.605 | 0.365/0.365 | **1.000/1.000** | 0.990/0.990 | 1.000 |
| 2×4 LOCAL alloc | 8 | 2.894 | **0.805** | 0.547/0.547 | 0.964/0.964 | 0.964/0.964 | 0.917 |
| monolithic K=16 | 16 | 1.342 | 1.342 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000 |
| 2×8 GLOBAL | 16 | **1.434** | 1.342 | 0.099/0.099 | **1.000/1.000** | 1.000/1.000 | 1.000 |
| 2×8 LOCAL | 16 | 2.211 | **0.122** | 0.398/0.398 | 0.917/0.917 | 0.911/0.911 | 0.729 |
| monolithic K=32 | 32 | 1.276 | 1.276 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000 |
| 4×8 GLOBAL | 32 | **1.457** | 1.276 | 0.031/0.031 | **1.000/1.000** | 1.000/1.000 | 1.000 |
| 4×8 LOCAL | 32 | 2.980 | **0.105** | 0.072/0.072 | 0.953/0.953 | 0.953/0.953 | 0.615 |
| 8×4 GLOBAL | 32 | **1.734** | 1.276 | 0.031/0.031 | **1.000/1.000** | 1.000/1.000 | 1.000 |
| 8×4 LOCAL | 32 | 5.270 | **0.059** | 0.184/0.184 | 0.917/0.917 | 0.921/0.921 | 0.552 |

Routers: **R1** = `argmin_r V_r(x_final)` (post-settle energy) · **R2** = `argmin_r V_r(q)` (pre-settle energy = nearest-well score) · **R3** = `argmin_r ‖x_final − q‖` (settling displacement).

Three findings, one of them a refutation of the obvious design:

1. ⛔ **R1 — "settle everywhere, keep the lowest energy" — is a BROKEN router.** It routes at or *below* chance (0.031 vs chance 0.25 at N=4; 0.031 vs 0.125 at N=8). **Reason (and it is structural, not a tuning issue): every shard's wells have the same depth, so the post-settle energy measures which well the query fell into, not how well it matched.** P6 REFUTED. If anyone in this program has been assuming energy-after-relaxation is a usable confidence/routing signal, that assumption is wrong at equal well depth.
2. ✅ **R2 and R3 are EXACT routers (1.000 route, 1.000 value) up to N=8 shards and K=32, with zero learning and fully interleaved shards.** R3 (`settling displacement`) is the physically native one — it is the work the dynamics had to do to accept the query.
3. ⚠ **Global allocation is REQUIRED.** Under LOCAL allocation (each shard placing addresses without knowing the others) cross-shard separation collapses `0.805 → 0.122 → 0.105 → 0.059` as N grows, the union read degrades `0.917 → 0.729 → 0.615 → 0.552`, and even R2/R3 fall to 0.917. **So the advisor's refinement is right in letter and needs one addition: the only global object is routing — and routing has TWO halves, a read-time score (free, non-parametric) and a write-time ALLOCATOR (a registry, cheap but genuinely global).**

**Required inter-unit separation condition (the task asked for an `N_pack`-style bound across units):**
`min_{m≠n, any units} ‖c_m − c_n‖ ≥ sep_crit ≈ 2.5–3.0 s` (§4.2), i.e. exactly the *same* condition as within a unit, applied to the union — hence `K_total ≤ N_pack(Q, sep_crit)` and **not** `N · N_pack`.

### 2.2 Routing error — cost, and is the router itself capacity-bounded?

- **Cost of a miss.** In the mixture store (B) a routing miss is not a distinct failure mode — there is no routing at read time at all. In the product store (A) a miss is **total**: the wrong block's payload is returned with full confidence (`checkD`: value accuracy tracks route accuracy exactly, 1:1, in every row). There is no graceful degradation. ⇒ **any product-store design needs an abstention rule** (e.g. refuse when the top-2 router scores are within `Δ`), which is the controller's existing admission-style deadband applied at read time.
- **Is routing capacity-bounded?** **No, for R2/R3.** They are non-parametric scores evaluated from the shards' own potentials — nothing is written by a gradient dig, so they cannot inherit the ~32 write ceiling. Their limit is the *same* separation criterion at the item level, which `checkD` shows is met at N=8 with fully **interleaved** shards (shard centroids all coincide near the ball centre and routing is still 1.000 — so a centroid/Voronoi router would fail where R2 succeeds; specify R2, not "which cell is q in").
- **The fallback that IS bounded.** If a future design routes with a *learned* module written by the same operator, it inherits the ceiling: `N ≤ 32` per level, `K_total ≤ 32^L` at read depth `L` — a tree, i.e. the classical result, `O(log_32 K)` sequential decisions. Worth recording as the pessimistic branch; **it is not needed while R2/R3 work.**

### 2.3 ⚠ Every global / shared term in the current code path (the audit the task asked for)

| # | term | where | separable? | verdict |
|---|---|---|---|---|
| 1 | `l_bar = mean_{i<j} relu(barrier + max(V_i,V_j) − V_mid)²` | `training/train_memory.py:110-115` | **NO — explicit all-pairs coupling** | It is the *only* crowding-aware term. Under a masked write its gradient reaches only the written block (fine) **but it requires every other item's target** ⇒ the write needs the registry. **This is the joint dig, by construction.** |
| 2 | `mean` over K in `l_grad`, `l_min`; `mean` over C(K,2) in `l_bar` | same file | global rescale | **My dilution conjecture (P4) is REFUTED — see §2.4.** Not a problem. |
| 3 | `confine · Σq²` | `AtomDictionaryPotential.__call__` | YES (sum of squares, static, no params) | safe |
| 4 | AdamW decoupled weight decay | `optax.adamw` in `train_memory_landscape` | elementwise, but *would* shrink frozen rows | **already correctly handled** by masking updates not gradients |
| 5 | ⭐ **atom init `init_scale=1.0`** | `AtomDictionaryPotential.__init__` | **NO — every group's atoms are scattered over the whole ball** | **the measured W3 violation (§1.4): 1.4% amp deviation, 3 orders above the site tail.** Fix: localize group `j`'s atom init near site `j`. |
| 6 | ⭐ relativistic `T = √(c²pᵀM⁻¹p + m₀²c⁴)` on ONE unit of dim `N·d` | `chlu_unit.py` | **NO — one global Lorentz factor couples every block** | measured `max|∂²T/∂p_i∂p_j|` off-block = **8.4e-2** vs diagonal 0.81; block-0 speed falls **0.482 → 0.043 (11×)** as neighbours heat to ‖p‖=10. **`CLULattice`'s per-unit `T` is exactly 0.000e+00.** ⇒ **implement shards as `CLULattice` units, never as one wide relativistic CHLU.** |
| 7 | `CLULattice.couplings` / `edges` | `core/lattice.py:426-433` | coupling terms are added to `V` | must be `edges=(), couplings=()` for true shards. A "lattice" in this codebase is by default **not** a set of shards. |
| 8 | `mass_parameterization="*_zeromean"` | `chlu_unit.mass_vector` | **NO** — centres `log_mass` over **all** coordinates at use time | inside one wide unit, a mass change in shard 1 shifts every shard's masses. Per-unit in a `CLULattice`. Another argument for #6's conclusion. |
| 9 | `tie_channel_mass` | ibid. | ties coords (0,1) | per-unit; harmless for shards, flagged for completeness |
| 10 | ⭐ `fixed_norm` query jitter `σ/√d` per axis | `exp_designed_mechanism.py:237` | **NO** | with shards as blocks of one wide unit, `d = N·d_shard` ⇒ **per-shard query noise shrinks with N: an unearned advantage.** Fix per shard. **Fairness trap — reconciliation item 3.** |
| 11 | heterogeneous per-unit `γ` | `CLULattice.gamma_vector` | volume law survives; **conformal symplecticity and the Prop-4 SV pairing do not** | if shards get different lifetimes (dial #2), the docstring's own warning applies; prefer uniform `γ` per rollout |

### 2.4 The one attack that did *not* land (I was wrong, on the record)

I predicted (P4) that the all-pairs `mean` normalisation drowns the crowding term as `1/K`, and that this was the real mechanism behind the ceiling — which would have made the Head's scale-invariance ablation the cheap fix. **Measured on the real `write_loss` (`checkC`, d=4, `designed_sites`, amps at 0.5):**

| K | site minsep | violating pairs | ‖∇l_bar‖ | ‖∇(l_grad+l_min)‖ | ratio | ratio·K |
|---|---|---|---|---|---|---|
| 4 | 1.400 | 0 | 4.24e-2 | 3.51e-1 | 0.121 | 0.48 |
| 8 | 1.139 | 5 | 1.66e-1 | 5.74e-1 | 0.290 | 2.32 |
| 16 | 0.903 | 57 | 2.18e-1 | 3.96e-1 | **0.549** | 8.79 |
| 32 | 0.710 | 216 | 2.75e-1 | 6.82e-1 | 0.404 | 12.9 |
| 64 | 0.549 | 1056 | 4.55e-1 | 1.61e+0 | 0.283 | 18.1 |
| 128 | 0.451 | 4503 | 7.92e-1 | 2.72e+0 | **0.291** | 37.2 |

The ratio is **roughly flat, not `1/K`**, because sites crowd as K grows so the *fraction* of violating pairs stays O(1) (4503/8128 at K=128) and cancels the `1/K²` normalisation. **P4 REFUTED. Prediction for `write-ceiling-break` Item 2: rescaling the barrier/margin weights alone will NOT move the ceiling.** (Caveat: a flat gradient-*norm* ratio is not a proof of equal influence under Adam's per-parameter normalisation; it does however remove the mechanism I proposed.)

---

## 3. Item 3 — the read-cost verdict (this decides whether the win is real)

**Verdict, in three parts:**

1. **Mixture store (B) — the shipped `n_groups` sharding — is `O(1)` in `N`, exactly and trivially.** `V(q) = Σ_r V_r(q)` is one function of one `d`-vector; atom grouping is invisible to the physics. Read cost depends on the total atom count, not on `N`. There is *no* routing step and *no* read-side penalty. **If the goal is "capacity without an N-fold read cost", (B) already delivers it — and it is the arm `write-ceiling-break` Item 1 is running.**
2. **Product store (A) is `O(1)` in rollout DEPTH and `O(N)` in pointwise FLOPs, with a constant that makes the FLOP term negligible for any plausible `N`.** `V` is separable ⇒ one Verlet rollout settles all shards simultaneously (no sequential scan). Routing with R2/R3 costs **`N` scalar potential evaluations at one point**, versus a read of ~**1200 Verlet steps each with a gradient** (`controller-mvp`: 0.70 ms per two-phase rollout). Ratio ≈ `N / 2400`. ⇒ **routing is under 5% of one read up to `N ≈ 100`, and only at `N ≳ 10³` does read cost become genuinely linear.** With the controller's site registry (which `controller-mvp` already maintains — "the writer records where it wrote"), the score is `K_total·d` flops, cheaper still (the gate is measured at **0.006 ms**, ~1% of a read).
3. ⛔ **The `O(N)` blow-up the task worried about happens in exactly one case: if the routing decision requires *settling* in each shard.** R1 (post-settle energy) is that case — and it is *also the case that does not work* (§2.1). So the design that costs `O(N)` reads is the design we must not build anyway. **Condition for `O(1)`: the routing statistic must be evaluable WITHOUT running the dynamics** — pre-settle energy, distance to the registry, or an explicit tag. This is a clean, checkable design rule.

**⚠ The honest caveat a referee will raise, stated first (N89 pattern).** R2 = `argmin_r V_r(q)` is, at the read moment, **a nearest-neighbour score over stored addresses** — the same object that laundered the φ result. So *the routing half of a sharded store is classical indexing*; the CLU dynamics contribute the settle-and-read (basin correction + payload recovery), not the routing. This is not fatal (every sharded memory — IVF/FAISS, mixture-of-experts — routes classically) but **"capacity multiplies by sharding" must never be presented as a dynamical result.** The defensible claim is narrower and true: *the write is additive at zero optimizer cost, and the read stays O(1) in depth, because a classical O(N) score suffices to route.*

---

## 4. Item 4 — both branches, plus the finding that changes the branch probabilities

### 4.1 If `write-ceiling-break` succeeds / if it fails

- **If the ceiling breaks:** the lattice is **not superseded, but it stops being a capacity story.** Theorem L1 says capacity then becomes `min(K_addr, ·)` — geometry-bound — and sharding cannot exceed `K_addr`. What sharding still uniquely buys maps onto the Head's other three dials: **isolation under sequential writes** (dial 3: W2 gives a *provable* non-interference guarantee, not an empirical one), **per-unit lifetimes** (dial 2: a shard is the natural granularity for `γ`/leak, cf. `CLULattice.gamma_vector`), and **admission scoping** (dial 1: refuse into a full shard, relocate to another). File it under control, not capacity.
- **If the ceiling holds:** the lattice is the primary route to R2, and Theorem L1 gives the law to claim: `K = min(K_addr(d), N·K_ceiling)`. **The engineer's w25 build is small because most of it exists** — see §5.

### 4.2 ⭐ EXPLORATORY — a geometric account of the ceiling that predicts d-independence

While computing the inter-unit separation condition I ran the harness's own allocator. **At every measured wall in the w23 sweep the critical separation is the same multiple of the well width** (`checkF`, `designed_sites(d,K,R=1)`, atom width 0.30):

| d | last PASS K | sep | sep/s | first FAIL K | sep | sep/s |
|---|---|---|---|---|---|---|
| 2 | 4 | 1.159 | 3.86 | 8 | 0.649 | 2.16 |
| 3 | 8 | 0.923 | 3.08 | 16 | 0.744 | 2.48 |
| 4 | 16 | 0.903 | 3.01 | 32 | 0.710 | 2.37 |
| 5 | 32 | 0.849 | 2.83 | 64 | 0.690 | 2.30 |
| 6 | **32** | 0.924 | **3.08** | 64 | 0.795 | 2.65 |
| 8 | **32** | 1.020 | **3.40** | 64 | 0.908 | 3.03 |

and the **designed** walls sit in the same window at *their* width 0.15 (K=16 at d=2 → 0.425 = 2.83·s; K=64 at d=3 → 0.402 = 2.68·s; K=128 at d=4 → 0.451 = **3.01·s**).

Three consequences, all of which the "d-independent write ceiling" reading does not supply:

1. **The apparent d-independence is a concentration-of-measure artifact.** `minsep ∝ K^{-1/d}`: at d=8 a **32×** change in K moves minsep only **1.72×** (1.374→0.798). A *fixed* width threshold therefore produces a wall at nearly the same `K` for all `d ≥ 5`. **A geometric criterion predicts a d-independent-looking ceiling; it does not need an operator ceiling to explain one.**
2. **The learned/designed tax is the width ratio to the d-th power.** `K ∝ sep_crit^{-d}`, `sep_crit ≈ 2.7 s`, and the shipped flags are `well_width = 0.15` (designed) vs `atom_init_width = 0.30` (learned) — exactly 2×. Predicted tax `2^{-d}`; **w23 measured 1/4, 1/8, 1/16 at d = 2, 3, 4.** A sum of positive-amplitude Gaussians of width `s` cannot be narrower than `s`, so `K_learned ≤ 2^{-d} K_designed` is an **upper bound**, and the data saturate it.
3. **It satisfies the task's mandatory consistency condition** ("any theory of the learned ceiling must be consistent with the designed write having no such ceiling") **without a second mechanism**: the designed store's wall is `2^d` higher, i.e. ≥256 at `d ≥ 5` — off the ladder cap. It also *predicts* w23's otherwise-odd observation that **more atoms make d=6 K=64 worse** (0.855→0.809): more width-0.30 atoms in a crowded ball merge basins.
4. **Where the width floor comes from.** It is set by the *query noise*, not the write: the basin must contain the jittered query, `s ≳ σ_q ≈ 0.15` (`fixed_norm`). **The designed width 0.15 sits exactly at that floor; the learned width starts at 2× it.** The write objective's `σ_addr = 0.25` neighbourhood does not force a wide well (a narrow deep well satisfies `l_min` easily), so **nothing in the objective forces 0.30 — it is an initialisation.**

⚠ **Status: CONJECTURED, not pre-registered** (declared in `PREREG.md`). Two honest weaknesses: (a) my greedy packing estimator is boundary-dominated at these K (`checkE`: the 2× width ratio costs only 2.25×/3.40×/6.43× rather than 4/8/16 at d=2/3/4), so the *exponent* is verified only through the `K·sep^d` volume law, whose spread is 2.4–4.1× across a 32× K range — the allocator's own separation-vs-K exponent is sub-volumetric at these sizes, **which is a property of `designed_sites` in a fixed unit ball, not a capacity exponent of CLU** (and per **CM-22(j)** I do not convert it into one); (b) `log_width` is *trainable*, so the trained widths might not be 0.30 at all. **(b) is decisive and free to check — §5.0.**

---

## 5. Falsifiable, concretely-sized predictions for w25

**§5.0 — do this first; it costs one script and can kill or confirm §4.2 outright.**
> **Measure the trained `log_width` distribution on the w23 `dimension-aware-budget` checkpoints** (or re-run one cell and dump it). **Prediction: the median effective well width at the stored sites is ≥ 0.28 (i.e. the write does not narrow the atoms), and the ratio `minsep(K_wall)/width ∈ [2.4, 3.1]` at every d.** Falsifier: widths trained down below ~0.18, or the ratio varying by >2× across d ⇒ §4.2 is dead and N92's operator reading stands unchallenged. *Owner: analyst or engineer, ≤1 hour.*

**§5.1 — the sharding prediction (the task's required deliverable), with a built-in discriminator.**
> Run the w23 designed-mechanism harness with the masked/sharded write (`n_groups`, `atom_write_mask_fn`, one item per group, sequential) plus **global farthest-point allocation over the union** and the localized atom init from §1.4, at **≥3 seeds** and the N92 2×-atom adequacy re-check:
>
> | cell | baseline (w23) | prediction | what it discriminates |
> |---|---|---|---|
> | **d=6, K=64, 2 shards × 32** | 0.855 FAIL | **strict ≥ 0.90 PASS** | `K = min(K_addr, N·32)` with `K_addr(6) ≫ 64` — the additivity claim |
> | **d=4, K=32, 2 shards × 16** | 0.83 FAIL (flat over a 16× atom sweep) | **still FAILS, ≤0.87** | ⭐ the control. d=4's wall (16) is *below* the 32 ceiling, so it must be `K_addr`-bound; if sharding fixes this cell too, then geometry never bound anything and `K = 32N` unclamped |
> | **d=8, K=64, 2×32 and 4×16** | 0.883 FAIL | 2×32 **PASS**; 4×16 **PASS**, and 4×16 ≥ 2×32 | additivity is in `N·K_ceiling`, not in shard size |
> | **d=8, K=256, 8 shards × 32** | untested | **PASS iff `K_addr(8) ≥ 256`** (designed reaches ≥256 there) | where the geometric term takes over |
>
> **The 2×2 is the result, not any single cell:** both fixed ⇒ the ceiling was entirely per-dig and `K_total = 32N`; only d=6 fixed ⇒ Theorem L1 with `K = min(K_addr, 32N)`; neither fixed ⇒ **additivity REFUTED, the ceiling is not per-dig**, and my Theorem L1's conditions must be re-audited (start with W3/§1.4).

**§5.2 — the non-sharding route to R2, which `write-ceiling-break` as written does not test.**
> One flag: `atom_init_width: 0.30 → 0.15` (matching the designed well width, which is at the query-noise floor), with the atom budget raised to keep coverage (`min_atoms_base` ×2^{d/2}). **Prediction: `K_learned(4)` rises 16 → 64–128 and `K_learned(6)` rises 32 → ≥128**, i.e. the "ceiling" moves by ~`2^d`. Falsifier: no movement at ≥3 seeds under an adequate budget ⇒ the ceiling is an operator limit and §4.2 is wrong. *This is a single-flag experiment with a `2^d` predicted effect; it should be run before any lattice engineering.*

**§5.3 — the read-side prediction (cheap, and it protects the claim).**
> With global allocation and R2/R3 routing, a product store of `N ≤ 8` shards retrieves at **parity with the monolithic store of the same `K_total`** (measured: 1.000 vs 1.000, `checkD`). With **local** (per-shard) allocation the union read degrades to **0.92 / 0.73 / 0.62 / 0.55** at N = 2/2/4/8. **Falsifier: if a global allocator is added and retrieval still degrades with N, Prop L4's "read-side is conserved" is wrong and there is a second, unmodelled cross-shard channel** (first suspect: item #6 or #10 in the §2.3 audit).

### What the engineer must build in w25 (if the ceiling holds) — mostly assembly

1. `CLULattice(units, edges=(), couplings=())` as the shard container — **never** one wide relativistic CHLU (§2.3 #6, #8). Assert `edges == ()`.
2. Localized atom initialisation for `AtomDictionaryPotential` groups (§1.4) — a ~5-line change plus a flag; without it W3 is violated at 1.4% and shards are not independent.
3. A **global address allocator** in `Controller` — it already records sites and already has `admit_site`/refuse-and-relocate; extend the spacing test to run **across shards** (`stored_addresses` becomes the union). This is the *only* global object.
4. Routers **R2** (`argmin_r V_r(q)`) and **R3** (`argmin_r ‖x_final − q‖`) plus a top-2 **abstention deadband**; explicitly **do not** ship R1.
5. Per-shard query noise (§2.3 #10) in the harness, or the sharded arm is unfairly advantaged.

---

## 6. How I verified — commands and observed numbers

```
PYTHONPATH=/Users/user/Desktop/CHLU .venv/bin/python \
  .claude/scratch/lattice-capacity-theory/checkA_crosstalk_budget.py   # numpy only
  …/checkB_factorization.py     # real write_loss + AtomDictionaryPotential + adamw
  …/checkC_barrier_dilution.py  # real write_loss term-by-term gradients
  …/checkD_read_routing.py      # numpy product store, 3 routers, 3 seeds
  …/checkE_width_tax.py         # greedy packing vs width
  …/checkF_separation_law.py    # real designed_sites allocator
```
All exit 0. Headline observed numbers, in one place:

| claim | script | observed |
|---|---|---|
| crosstalk superposes **incoherently** | A1 | ‖δx‖ 9.59e-5 (m=1) → 5.34e-4 (m=32); √32 law predicts 5.43e-4 (**1.6%**) |
| additivity budget | A2 | `(0.1/5.5e-5)² ≈ 3.3e6` (d=2), `≈5.9e5` (d=4); global write `<1` |
| random placement ⇒ `√N_pack` | A3 | K_random(50%) = 2 / 4 / 16 at d = 2/4/6 vs `√N_pack` = 3.1 / 9.9 / 31.0 |
| masked ≡ independent, up to init scatter | B1 | rel Δ = 2.4e-3 / 8.7e-3 / **1.4e-2**; ΔV at sites 2.9e-3; unmasked control moves **128/128** atoms |
| relativistic single unit is **not** shard-separable | B2 | off-block `∂²T` = **8.4e-2** (diag 0.81); `CLULattice` = **0.0**; block-0 speed 0.482 → **0.043** |
| barrier term is **not** diluted | C | ratio 0.121 → 0.549 → 0.291 over K = 4 → 16 → 128 |
| post-settle-energy routing is broken | D | route 0.365 / 0.099 / **0.031** vs chance 0.5 / 0.5 / 0.25 |
| pre-settle / displacement routing is exact | D | **1.000 / 1.000** at N = 2, 4, 8 (global alloc) |
| local allocation degrades the union read | D | 0.917 → 0.729 → 0.615 → 0.552 |
| the critical separation is width-locked | F | **2.16–3.86 s** across d = 2…8, transition in **2.4–3.0 s**; designed walls 2.68–3.01 s |

---

## 7. What I could NOT prove or verify (scoped open questions)

1. **Theorem L1's `K_addr` is not measured for the *learned* store.** Everything about the geometric term rests on `designed_sites` separations plus the designed store's censored ≥256. A learned-store `K_addr` requires §5.0/§5.2.
2. **§4.2 is a conjecture with one free-to-run falsifier (§5.0) and I did not run it** — it needs a trained checkpoint I do not have, and reproducing one is an engineer-scale run. I deliberately did not soften N92; I flagged it for an owner.
3. **`checkB` used 200 write steps, not w23's 600**, and K=4 at d=2 — the bit-identity question does not need scale, but the *size* of the init-scatter violation at the frontier (d=6, K=64, 4096 atoms) is extrapolated, not measured.
4. **`checkA`/`checkD` are analytic Gaussian stores with gradient-flow settling, not the two-phase `γ_address → γ_read` Verlet rollout.** The routing conclusions (R1 broken, R2/R3 exact) depend only on the *scores*, which are evaluated on the real functional form; the settling path could change R3's constants (not R2's, which is pre-settle).
5. **I have no result on whether `K_ceiling ≈ 32` is shared across shards or per-shard-per-dig at a *deeper* level** — Theorem L1 assumes each masked write is a fresh dig with the same ceiling. If the ceiling has a component that is per-*parameter-tree* rather than per-dig, additivity fails and §5.1's 2×2 will show it (neither cell fixed).
6. **Not addressed:** whether shards can share a payload channel; vector-valued payloads; and the `d_eff < d` shell-concentration effect (`address-space-dimension-scaling`), which will reduce `K_addr` below my ball estimates and is the most likely reason §5.1's d=8 K=256 cell disappoints.

---

## Proposed handover updates (for the Hub)

1. **§1 / new theory entry — Theorem L1 + Prop L2 (proven).** *Sharded capacity is additive at zero optimizer cost, and the additivity is a theorem about elementwise optimizers, not a hope:* AdamW's state factorizes over parameter blocks, so a masked write on disjoint atom groups **is** N independent optimizers. **The Head's "chain optimizers / map disconnected optimizer spaces" worry has no referent.** But the law is `K_total = min(K_addr(Q, s, σ_q), N·K_ceiling)` — **sharding relocates the binding constraint from the write operator onto geometry and can never exceed geometry** — and it requires a **global address allocator** (a registry, not an optimizer). *Only-global-object = routing is CORRECT, with the addition that routing has a write-time half.*
2. **§1 — Prop L4 (read-side conservation).** Sharding relaxes write-side crowding and **conserves** read-side discrimination. For the shipped mixture store (`n_groups`) the read is *literally identical* to monolithic ⇒ **read cost is O(1) in N, exactly.** For a product store, read is O(1) in rollout depth and O(N) in pointwise scores, `≈N/2400` of one read ⇒ **free to N≈100.** ⛔ **Never route on post-settle energy** (at or below chance, measured); route on pre-settle energy or settling displacement (1.000 at N=8, interleaved shards, zero learning). ⚠ File with the honest caveat: **the routing half is classical nearest-neighbour indexing** — the capacity-by-sharding claim must never be presented as a dynamical result (N89 discipline).
3. **§7 — five code-path defects/traps flagged for the engineer**, all specified in §2.3: (a) `AtomDictionaryPotential` `init_scale=1.0` scatters every group's atoms ⇒ the *measured* violation of shard independence (1.4% amp deviation, **3 orders above** the site tail — my own prediction failed here); (b) a **single relativistic CHLU of dim N·d is NOT shard-separable** (off-block `∂²T` = 8.4e-2; block speed falls 11× with hot neighbours) while `CLULattice`'s per-unit `T` is exactly separable ⇒ shards must be lattice units; (c) `mass_parameterization="*_zeromean"` centres masses over **all** coordinates; (d) `CLULattice` couplings must be empty; (e) ⚠ **`fixed_norm` query jitter `σ/√d_total` is a fairness trap** — sharding as blocks of one wide unit silently reduces per-shard query noise.
4. **§7 / N92 — RECONCILIATION, needs an owner (do not apply yet).** A fixed-width geometric criterion (`sep_crit ≈ 2.4–3.0 × well width`, holding at **every** designed and learned wall, d = 2…8) reproduces the tax as `2^{-d}` from the shipped `well_width 0.15` vs `atom_init_width 0.30` and **predicts a d-independent-looking ceiling** via `minsep ∝ K^{-1/d}` flattening in high d. It also predicts w23's "more atoms make d=6 K=64 *worse*". **CONJECTURED / EXPLORATORY.** The decisive check is free (§5.0: dump the trained `log_width`). If it survives, N92's "the ceiling belongs to the WRITE OPERATOR" needs qualification and **R2 has a one-flag route (`atom_init_width` 0.30→0.15, predicted ×`2^d`)** that requires no lattice at all.
5. **§8 — my three REFUTED pre-registrations are results and should be recorded as such** (`PREREG.md`): the crowding-term dilution mechanism is **absent** (⇒ predict `write-ceiling-break` Item 2's rescaling does **not** move the ceiling); post-settle-energy routing/confidence is **at or below chance at equal well depth** (a broader warning: relaxed energy is not a match score); and cross-item crosstalk accumulates **incoherently (√K)**, which makes the finite locality ratio *five orders* from binding rather than a caveat.
6. **Candidate negatives-registry entries** (curator to word, tiers to the Hub): **N-a** ⛔ *post-settle energy is not a routing/confidence signal for equal-depth wells* (measured at-or-below chance, 3 shard counts) — tier A if anything in the program relies on relaxed energy as a score; **N-b** ⛔ *shard independence is violated by atom initialisation, not by physics* (1.4% vs a 1.5e-5 tail) — tier B, with the fix; **N-c** ◐ *sharding conserves read-side discrimination* (`K_addr` is untouched by N) — tier B, a scope guard on any "capacity multiplies" wording.
