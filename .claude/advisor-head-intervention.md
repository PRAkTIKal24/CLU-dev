# HEAD + ADVISOR INTERVENTION — Stop testing levers. Build the CLU.

**Filed 2026-07-29 · Head + external Advisor · BINDING on all wave scoping from w27 onward.**
Companion to `.claude/handover_context.md` (state), `.claude/research_roadmap.md` (plan) and `.claude/outputs/head-advisor-w23-direction.md` (the w23–w26 direction chain, which this document supersedes wherever they conflict).

---

## 1. Why this document exists

Twenty-six waves have produced an unusually strong evidence base and an unusually honest one. They have **not** produced a test of the CLU. They have produced twenty-six tests of *pieces* of the CLU, each measured in a configuration chosen to make that piece legible — and legibility was bought, every time, by turning the other pieces off.

Two systematic errors, repeated across the whole program:

**Error 1 — we isolate one lever per experiment.** Capacity with the read fixed. The read with the write fixed. Admission with retrieval fixed. Lifetimes with everything static. Each result is clean, publishable and true *about that lever in isolation*. None of them is a result about the object we are proposing.

**Error 2 — we nerf learned structure down to designed structure to get a clean measurement.** Every time learning underperformed we replaced it with a designed mechanism, measured the designed mechanism, and reported the gap. The gap is real and worth knowing. But the endpoint of that process is a hand-built data structure, and we reached that endpoint: **the store we are currently benchmarking keeps 41 explicit floats per item — centres, payloads, amplitudes — in arrays, reads only the settled point, and enforces well separation by design so basins never interact.** In that configuration the settle computes arg-min over the stored centres, slightly worse and 150–1200 integration steps more expensively. w26's same-keys ablation measured exactly this and the lookup won 6/6 on both axes.

That is not a defeat of the CLU. **It is a correct measurement of a degenerate configuration of the CLU that we ourselves constructed.** The individual inferences are useful and they stay in the record. They are not what we want, and we will stop producing them.

---

## 2. What we have actually tested (honest inventory)

| lever | tested as | configuration it was tested in |
|---|---|---|
| capacity | `K_learned(d)` walls | one static global write, no read adaptation, no controller |
| the write operator | masked vs global, sequential vs joint | fixed geometry, fixed read, no lifetimes |
| reach / read-out geometry | excursion arms, annealed read | d=4 only, single-item retrieval, no stream |
| admission | spacing gate | designed store, no learning, static items |
| lifetimes | amplitude decay | bookkeeping-equivalent scalar per item, no spatial structure |
| deletion / placement | canonical placement | below capacity, no retrieval pressure |
| the compute dial (retry) | boost + re-settle | static retrieval benchmarks with a classical ceiling |
| read-in `φ` | frozen PCA / SimCLR | never trained through the store, never adapted online |
| isolation | interference ratios | pairwise, designed sites |

Every row is a single-lever row. There is no row for "all of them at once."

## 3. What has never been tested at all

1. **Items stored in a learned `V_θ` rather than in explicit arrays.** This is the difference between a dictionary and a compressed associative memory: a learned landscape reads in time independent of the number of items it holds; a table cannot. This was *impossible* until w26 broke the ~32-item learned-write ceiling at d=4. It is now merely untested.
2. **Trajectory as the latent.** The founding vision says *both trajectories and settled points carry information*. We have used settled points, only, in every experiment ever run.
3. **Non-separable landscapes.** We enforce `d_safe` spacing precisely so wells do not interact — engineering away the only condition under which the dynamics compute something arg-min cannot.
4. **The V2 physics.** Spontaneous symmetry breaking, Goldstone/flat directions, dissipation-proof memory channels — the theory half of the program — appear nowhere in the memory architecture. A flat direction stores a *manifold* of settled states, which no lookup table can express. (w25 saw a low-rank settled ray, participation ratio ≈1, and filed it as a minor observation.)
5. **The trash region `γ_φ(q)`.** Built, never used. Our decay lowers *an item's* well depth; a spatial friction field makes *a region of latent space* forgetful — a different and more useful granularity.
6. **Wormholes.** Built, never tested as memory. They are the third solution to the reach problem we spent two waves on: keep the address space large and tunnel it, rather than shrinking the excursion or extending the reach.
7. **A learned read-out `ψ`.** Still handcrafted. The read-in exists; the read-out never has.
8. **Multi-particle / parallel reads, and any form of consolidation across retrievals.**

---

## 4. What we actually want

**A full CLU — every lever active — on a general task, in a fair fight against the primitives people actually use (attention, SSM/Mamba, GRU, MLP), where those baselines also perform well.**

Concretely, "full CLU" means simultaneously live:

- **Write:** learned `V_θ` holding the items (not arrays), derived addressing, admission policy, per-item lifetimes, local/masked write, permitted basin interaction.
- **Read:** learned `φ` in, **learned `ψ` out**, two-phase relaxation, mass as selector, trajectory *and* settled point available to `ψ`, confidence-gated retry, wormhole hops where reach fails.
- **Structure:** symmetry/flat-direction storage where it earns its place, spatial trash regions, lattice sharding, the causal limit as a real constraint rather than an unused parameter.
- **Control:** a controller that decides — admit, place, evict, decay, route, retry, stop.

The claim architecture follows from this, and it is the inverse of what we have been doing:

> **Primary claim: CLU is a competitive general primitive.** Competitive-or-better against strong baselines on hard, standard tasks that those baselines also do well on.
> **Secondary claim: and it has capabilities they structurally lack** — settable per-item lifetimes, exact deletion, admission control, an anytime accuracy-vs-compute read.
>
> The secondary claims are *supporting evidence for a general-purpose primitive*, not the pitch. "CLU is great in general, and additionally can do these things nothing else can" — never "CLU wins the one axis where the competition is absent by construction."

---

## 5. The real research problem now: control, not levers

The reason we never ran the full system is that a full CLU with every lever free **collapses to a trivial response** unless something holds it in a productive regime. That is now the central engineering problem, and twenty-six waves gave us exactly the map needed to solve it: **we know the specific ways it collapses.**

**The anti-collapse checklist (each entry is a measured failure mode, not a worry):**

| # | collapse mode | what it looks like | source |
|---|---|---|---|
| 1 | **Overdamping → "the last observation"** | large γ makes `q*` ≈ the launch point; the physics is off and the number improves | C17-3 (corr(q*,q_last)→0.97) |
| 2 | **Settle → arg-min** | separable wells + settled-point-only read = a worse lookup table | w26 same-keys ablation |
| 3 | **Vacuous gate** | geometry makes the admission test arithmetically unable to fire | N74 (spacing 1.414 vs d_safe 1.10) |
| 4 | **Blank controls passing** | reads succeed on a store with nothing in it; a 1e-4 address leak makes classification perfect | N68 (blanks 0.992–1.000) |
| 5 | **Learned addressing dies** | gradient search for an address returns ~chance | w19 (0/18, 4.2%) |
| 6 | **Objective/goal divergence** | write loss reaches 0 while retrieval fails — the objective stops seeing crowding | w25/w26 |
| 7 | **Mass stores nothing** | `M`, `p₀` enter only via `M⁻¹∇V`, `M⁻¹p₀` — an exact gauge, not a channel | Prop F1 (×3) |
| 8 | **Learning erases design** | free `V_θ` destroys the structure a designed one provides | w20 |
| 9 | **Payload-dependent lifetimes** | retention ∝ −value²; "lifetime is a dial you set" is false without a fix | w25 (r=−0.85) |
| 10 | **Degenerate axes / silent knobs** | read-mode axis dead at `clu_steps=1`; `sleep_temperature` no-op at γ=0 | gamma-read-sweep, N19/N58 |
| 11 | **Reach failure** | wells invisible from the launch manifold; scored as a basin miss | w25/w26 |
| 12 | **Starve-and-overwrite** | naive sequential/masked writes give each item `atoms/K` and later writes bury earlier ones | w26 |
| 13 | **Under-trained artefacts** | diagnostics on immature fits are not properties of the shipped model | w25 reconciliations |

**The w27+ design task is to build the harness/controller that keeps every lever in its productive band simultaneously** — each checklist row becoming an explicit guard, an invariant, or a monitored quantity that fails loudly rather than silently improving a metric. This is where the program's accumulated intuition converts into an artefact. It is a *design* problem with a known failure surface, which is a far better position than it sounds.

---

## 6. Benchmark selection — binding criteria

A candidate task is admissible only if **all five** hold:

1. **Strong baselines that do well.** Attention / SSM / GRU / MLP are competitive on it. A task the competition fails by construction is inadmissible as a primary claim.
2. **Real headroom.** Nothing is saturated; no trivial method sits at ceiling.
3. **Memory management over time is the difficulty** — retention, interference, capacity pressure, selective recall under load — not single-shot lookup.
4. **NOT metric-native to the store.** If the query lives in the same metric space as the stored keys, a classical method is the provable ceiling. We have confirmed this four times; it is a theorem about our situation, not bad luck.
5. **Every lever can be active.** The task must permit learned `φ`/`ψ`, a live controller, lifetimes and retry to all matter at once.

Candidate families to scope (a recon picks one primary + one fallback): bounded-memory long-context sequence modelling · non-stationary / correlated streams where reservoir buffers fill with duplicates · long-horizon agent or episodic memory with genuine capacity pressure · algorithmic-reasoning tasks with a working-memory bottleneck. Weight class stays laptop/CSF3 — from-scratch, matched-parameter, matched-byte.

---

## 7. Papers

**The locked shorts assimilate into the ICLR paper.** They are not separate products to be defended in parallel; they are chapters-in-waiting — V1's certificates, V2's symmetry/memory correspondence, V3's lattice, V5's forgetting budget, the F5 formalism. The ICLR long is the full-CLU paper: the primitive, the control system that makes it work, the general-task result, then the capabilities others lack, with the laws (capacity/reach, lifetimes, compute-scaling) as the explanatory spine. **Flow is bidirectional:** new findings from the full-system work go back into the shorts before their own deadlines, and the shorts' results are re-used, not re-derived.

---

## 8. Binding prohibitions

1. **No more single-lever isolation studies as a wave deliverable.** Isolation is permitted only as a *diagnostic inside* a full-system experiment, never as the experiment.
2. **No more reducing CLU to a lookup approach.** Explicit per-item arrays, engineered separability, and settled-point-only reads are the degenerate configuration; moving *toward* them to obtain a clean number is forbidden. Items go into a learned `V_θ`; basins may interact; reads may use the trajectory.
3. **No primary claim on an axis where the competition is absent by construction.** Those results are recorded as secondary capabilities, in the "and also" position, always with their by-construction caveat.
4. **No benchmark that fails criterion 4** (metric-native). We have paid for that lesson four times.

## 9. What carries forward unchanged

The epistemics are the program's most valuable asset and none of this weakens them: the laundering control on every performance claim · pre-registration with declared falsifiers · the §7 dial-declaration rule at scoping time · multi-seed before any paper number · rescued baselines (N78) · negatives documented, never dropped · no direction closed as lost. The laws already banked — the reach/saddle condition, the capacity law and its d=4 unclamping, the write-operator locality result, the exact-decay law, canonical placement and exact deletion — are the spine of the paper and the raw material for the controller design. **We are not restarting. We are assembling, for the first time, the thing all of it was evidence about.**

---

*Filed by the Head + Advisor, 2026-07-29. The w27 Hub scopes against §4–§6 and is bound by §8.*
