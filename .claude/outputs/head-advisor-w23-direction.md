# HEAD + ADVISOR JOINT REPORT — W23 Review, Direction, and the Ambition (2026-07-24)

> **Provenance & authority.** Written jointly by the **Head** (direction, rulings, hypotheses) and the **external Advisor** (a read-only review pass over all six w23 outputs — no repo changes made). **Head positions in this document are binding program direction**; advisor analysis is input to the Hub's wave scoping. The compiling Hub should treat Part 2 as the direction queue for the next waves and reconcile this document with its own w23 review when integrating branches. Nothing here pre-empts the Hub's per-branch technical review (tests, merges, registry updates).

---

## Part 1 — W23 review (6/6 delivered, all pre-registered, quality uniformly high)

**The wave's central bet lost; the wave's components all won.** The bet — "the designed store will beat trivial baselines once a learned `φ` exists" — was falsified cleanly, on its own pre-registered control. But every piece of machinery the program's future depends on was independently validated, and three long-standing measurement conflicts were closed.

| task | verdict | what is now settled |
|---|---|---|
| `phi-read-in` | ⛔ headline negative | Laundering control fired on all 4 (dataset×arm) cells: kNN on the same features ties/beats CLU-in-φ everywhere ("the win is φ's, not ours"). Static recall is now **triple-confirmed dead** (w22 raw, w23 φ-space, w23 retry) — structurally: settling *approximates* the nearest-neighbour lookup, and an approximation cannot beat its target. ⭐ Survivors: `φ` lifts CLU from chance to 0.81–0.97 on CIFAR (the doctrine premise validated); CLU-in-φ decisively beats closed-form Hopfield-in-φ at load and noise; the retry confidence signal survives φ (AUROC 0.845–0.988) |
| `retry-compute-study` | ⭐ mechanism proven · ⛔ no absolute win | Random-kick and ensemble-of-k controls are **dead flat** in all 8 cells → the directed Lorentz-boost re-launch is a real mechanism, not a restart heuristic. The curve is monotone and **self-limiting** (auto-stops ~1.6×; ungated retry collapses). The trivial-NN floor dominates absolutely — because masked-pixel MNIST has no headroom. CM-23 must split: the *shape* claim survives (a rising curve feedforward cannot draw), the *dominance* claim is retracted |
| `controller-mvp` | ⭐ all verbs work | Retention-per-admitted **1.000 flat to K=64** (best of five). Per-offered is geometry-conditional: 0.081 on an undersized address space (the abstention price, worst of the table) but **0.669 — beats all four primitives (GRU 0.57) — once the space is sized to the load**. Per-item decay is exact (`exp(−leak·t)`); permanent + leaky wells coexist in one store; self-eviction works. **Extracted design rule: size the address space to the item load (packing bound ≥ K)** |
| `dimension-aware-budget` | ⭐ law pinned | **`K_learned(d) = min(2^d, K_ceiling≈32)`.** Geometry vindicated for d≤5 (capacity doubles per dimension — exactly the designed rate); w22's d=8→8 collapse was a budget artifact (resolved to 32, monotone restored). The **~32 ceiling belongs to the WRITE, not the terrain**: d-independent, *worsens* with more atoms (d=6: 0.855→0.809 at 2×), and write-loss hits zero while retrieval fails ⇒ joint-dig interference + an objective blind to crowding. Masked writes are bit-local (8474× advantage) → the prime fix candidate |
| `continual-learning-recon` | target picked; niche scoped honestly | Primary target: **rehearsal-free Class-IL, Split-MNIST → Split-CIFAR-10, from scratch** (van de Ven taxonomy, ACC + forgetting/BWT). Replay solves Class-IL at this scale (DGR 90.8 / iCaRL 94.6) and regularizers collapse to ~chance → the winnable claim is **"best replay-free method"**, a real refereed slot (iCaRL/SQHN precedent). SQHN (Nature Comms 2024) is the neighbour to out-differentiate on: continuous landscape · per-item decay schedules · retry read · learned φ. Mandatory baselines: tuned ER, iCaRL, **GDumb at matched memory**, EWC/SI as known-null |
| `r19-r20-reconciliations` | all three closed | **CLU does not beat persistence** (the audit's n=1 "win" was a stale L-2 launch frame; N86 upheld, quotable number = the L-1 loss). The single-basin collapse and the 98.3%-ballistic figure were both **under-trained-model artifacts** (≤40-epoch fits). New standing rule: *diagnostics on <40-epoch CAFE fits are not properties of the shipped model* |

**Program-level reading (joint).** Design keeps winning, learning keeps losing — and w23 finally *explains where the store's value lives*: not in static lookup (provably ceiling-capped by the operation it approximates) but in **control over memory in time** — admission, lifetimes, isolation under sequential writes, compute-adaptive reads. Every future claim should be a claim about one of those four dials.

---

## Part 2 — Direction for the next waves (Head rulings + jointly agreed points)

1. **[HEAD RULING] Masked recall is demoted to the appendix, permanently.** Its role: "CLU sits within 3–13pp of the NN ceiling it approximates, and decisively beats modern Hopfield at load/noise" — competence, not victory. It is the wrong lead task because CLU *approximates* the method that wins it; no lead claim on that axis again.
2. **[HEAD RULING] Build the headroom benchmark for retry** — a regime where all methods land ~0.6–0.7 and the boosted retry buys a visible margin at stated extra compute. Advisor design constraint from the w23 physics: the failure mode must be **ambiguity** (structured/correlated occlusion, well-crowding, partial keys) — not destruction (Gaussian noise), which the boost cannot recover past the basin cliff. Crowded-store retrieval *inside* the CL entry supplies this regime for free.
3. **The write-ceiling experiment (the gate to the capacity law).** Test masked/sequential writes against the joint global dig. Include the **Head's scale-invariance ablation** (rescale well depths/margins so the per-item write signal is size-independent) to properly close the Head's quantization/normalization hypothesis — the current evidence (d-independent cap, worsens with more atoms, designed reaches ≥256 under identical numerics) points to optimization interference, but the ablation closes it by measurement, not argument. The write-loss-zero-while-retrieval-fails finding also puts an **objective-side fix** (crowding-aware margin) on the table.
4. **The lattice question → V3-short / theorist pass [HEAD hypothesis, advisor-refined].** Sharding items across N units multiplies capacity (~32·N) **without optimizer synchronization iff writes are local** — masked writes share no parameters across units, so there is nothing to sync; the only global object is *routing*, which is the controller's placement verb, not an optimizer. The Head's "chain optimizers / map disconnected optimizer spaces" concern dissolves under write-locality; formalize this (and its failure modes) as a theorist task.
5. **The CL entry (Phase-2 core):** designed store + `φ` + controller + retry + per-item decay on rehearsal-free Class-IL, sizing rule baked in from day one, mandatory baseline table per the recon. **[ADVISOR FLAG — resolve before building] the φ-stream question:** in Class-IL, `φ` may not be trained on data from future tasks (the w23 `phi-read-in` disjoint pool saw all classes — fine there, leakage here). Options: φ from task 1 only · online φ · generic frozen features. Choose and **pre-register**, or a referee calls data leakage.
6. **[HEAD RULING] Extend "harder benchmarks" to replay-handicapped regimes:** strict single-pass online CL (replay structurally limited), tiny memory budgets (buffers dilute), long task sequences (20–50 tasks), and **privacy-constrained CL** (raw exemplars disallowed — replay illegal by rule; a landscape store is the only legal entrant). **Success bar (Head): win the replay-free class across a suite of relevant benchmarks, and approach — if not equal/beat — the replay methods at matched memory.**
7. **Multi-seed before any paper number.** Nearly all w23 headlines are seed-0; margins are large but referees will ask.
8. **Hygiene carried:** real Mamba before any SSM claim · Voraus re-run post-`dt` · the laundering control is mandatory on every future performance claim · the under-trained-diagnostics rule (Part 1, last row).

### Plain-language records requested by the Head (for onboarding and drafts)

**Atoms.** The landscape `V` is built from thousands of small Gaussian bumps ("atoms") — terrain sculpted from little mounds of clay. A stored item = a valley dug by arranging many atoms. The **designed write** places the clay by formula, one valley at a time. The **learned write** hands all atoms to one gradient optimization and asks it to dig all K valleys **jointly** — and that joint dig is what caps at ~32: different valleys fight over shared clay, and the write objective declares success (loss 0) while retrieval already fails. Not the terrain (designed digs ≥256), not parameters, not dimension, and — pending the ablation in point 3 — not numerics.

**The two gates.**
- **Admission gate (writing):** before storing, measure the distance from the proposed valley site to every existing valley; if closer than the safety distance (≈4.4× valley width — below which digging deforms a neighbour), relocate or refuse. A parking attendant who won't let you park close enough to dent the next car. The check is nearly free; hunting for a free spot in a crowded lot is what costs. This is why retention-per-admitted is a flat 1.000: nothing is stored that the store will later corrupt.
- **Confidence gate (reading):** after the ball settles, measure how close it stopped to a known valley centre. Dead-centre = confident, accept. In the no-man's-land between valleys = suspicious, boost and re-roll. Never re-kick a confidently settled ball — gateless retrying knocks correct answers into wrong valleys (measured: worse accuracy at more cost). The gate also makes compute self-limiting: when no low-confidence reads remain, spending stops on its own (~1.6×, not a fixed 9×).

---

## Part 3 — The ambition (Head's trajectory; the program's compass)

The transformer paper did not win by topping every benchmark — it won by being **parallelizable, general, simple to reimplement, and by doing one thing nothing else could do**, leaving its inefficiencies as gifts to a community that then optimized it for a decade. The goal is that trajectory: **propose the primitive so compellingly that thousands of researchers adopt, adapt, and optimize it in their own pipelines** — which then pulls hardware companies into building optimized kernels. The adoption question is not *"what leaderboard do we top?"* but **"what will researchers be unable to resist dropping into their own stack?"**

CLU's candidate answer: **it is the only memory with dials** — capacity, lifetime, compute, admission — each backed by a law. And it fits on one slide: ***write = dig a valley · read = drop a ball · forget = let it fill in.***

---

## Part 4 — The result set matching the ambition, and the pathway

**The five results to build toward (impact-ordered; chosen as targets, gated by evidence):**

- **R1 — Memory with a dial: certified per-item lifetimes.** Half-lives set at write time and measured to match the physics across orders of magnitude; permanent + scheduled-fade items in one store; deletion as a certified physical operation, not an approximate fine-tune. Lands directly on **machine unlearning / right-to-be-forgotten** — a regulation-driven field where every current method is approximate and expensive. *Closest to done:* `controller-mvp` demonstrated the machinery exactly; what's missing is packaging against the unlearning literature's benchmarks.
- **R2 — The capacity law, unclamped: K = 2^d addressable items.** Break the write ceiling (point 3 above — one experiment away from knowing), then demonstrate exponential-in-dimension capacity to d=16+ at O(1) read cost, matched parameters. The screenshot-able log-linear figure; also the NMI law.
- **R3 — The anytime read.** The monotone, self-limiting accuracy-vs-compute curve on the headroom benchmark (point 2), mechanism controls attached: *"the first associative memory with a test-time-compute knob."*
- **R4 — The replay-free CL sweep.** Best rehearsal-free method across 4–5 benchmarks including the replay-handicapped regimes, touching replay at matched memory, with the privacy framing (no raw data ever stored). The performance pillar that makes R1–R3 credible rather than curiosities.
- **R5 — The adoption hook: a drop-in `CLULayer`.** A bounded-memory long-context store for transformer stacks — a KV-cache that *doesn't grow* and has retention dials. A constraint pitch, not a supremacy pitch. Riskiest of the five (w22's sequence-slot result was levelling, not winning) and the one that converts readers into users, users into kernel-writers.

**The pathway:**
- **Phase 2 (next waves):** the CL entry + the write-ceiling experiment + the headroom retry benchmark + multi-seeding of w23; the φ-stream question resolved and pre-registered.
- **Phase 3:** R1 packaged against unlearning benchmarks (mostly framing — machinery exists); the capacity law pushed to d=16 with whichever write survives. **NMI forks here** — same results, laws foregrounded (per the filing rule: performance with the explaining law → NMI).
- **Phase 4:** the `CLULayer` long-context demo at CSF3 scale; the CL suite widened to the replay-handicapped regimes.
- **Assembly:** ICLR main text = the primitive in the hilly-landscape language + three laws (capacity, lifetime, compute-scaling) + the CL sweep + the unlearning demo. Appendix = the NN-ceiling proximity result, the full negatives registry material, the RUD-C benchmark spec. Release the module openly with a deliberate *"inefficiencies we leave to you"* section — the transformer's actual growth mechanism, done on purpose.

**The discipline that carries it:** no saturated leaderboards, ever; every claim about one of the four dials; every result shipping with its own laundering control. The program's unusual honesty is now a competitive asset — a paper whose referee attacks are pre-run is the paper that survives the spotlight this ambition invites.

---

*Filed by the Head + Advisor, 2026-07-24. For the Hub: reconcile with the formal w23 review at integration; Part 2 is the direction queue; Parts 3–4 are the standing compass for Phase 2+ scoping.*

---
---

# ADDENDUM (2026-07-25) — W24 Alignment Review, Corrections, and W25 Directions

> **Provenance & authority.** Head + Advisor joint addendum, filed at the Head's direction after reviewing the lead Hub's w24 rundown. **The corrections and w25 directions below are binding program direction** — the w25-scoping Hub applies them; the registry wordings in §A3 go to the next curator pass. This addendum extends, and where stated overrides, the w24 Hub's verdict framings; it does not dispute any w24 *measurement*.

## A0 — Alignment verdict on w24

**Execution: aligned and strong. Verdict-framing: misaligned in three places.** W24 delivered the gates it promised — ⭐ **Prop L2** (sharded masked writes = N independent optimizers; the Head's optimizer-sync concern has *no referent* — the lattice question closed by theorem), ⭐ **task1-only φ ratified at ≈0 cost** (w25 unblocked; `phi_dim ≥ 16`), multi-seed **0 flips** on every w23 headline, and referee-grade R1 recon findings ("certified" is a defended (ε,δ)-DP term · PALL ICLR'25 occupies the CL∩unlearning cell · the refuse-and-relocate **history-dependence gap** blocks even exact-deletion claims). All four dials measurably work. The Hub's honesty is exemplary — its own report contains every fact the corrections below rest on.

**The misalignment:** w24 ran the right experiments and **scored them with the retired scoreboard** — absolute wins against strongest-possible baselines — instead of the Part-1 claim rule (every claim is a dial claim, backed by a law, with its laundering control).

## A1 — The three corrections (binding)

**C1 — R3 was scored on the scoreboard the reframe retired.** The engineer added a masked-NN **oracle that is told which coordinates were erased** — information-theoretically perfect under known erasure — and the verdict was written as "leaderboard NO." But the R3 claim was never "wins static retrieval"; it is **the compute dial**: a monotone, auto-stopping accuracy-vs-compute curve whose mechanism has now survived every control **three consecutive times** (random-kick and ensemble dead flat, again). Losing to an oracle in a metric-native protocol is the *fourth confirmation* of the theorem this document already records ("where the store is metric-native to the query, a classical method is the ceiling") — not a verdict on the dial. The Hub's own §5 names the real untested item: **the non-metric-native retry regime**. R3 is *alive, untested in its native regime* — which is the w25 CL entry's crowded-store retrieval, where no mask-oracle exists.

**C2 — R2 was declared dead with the decisive experiment unrun.** "Dead via the write operator" was filed while the same report flags: the theorist's **competing geometric account** (fixed well-width concentration), a **free** decisive check (dump trained well-widths from an existing w23 checkpoint), and a **one-flag revival experiment** (`atom_init_width 0.30→0.15`) predicted to move the ceiling by `2^d`. This program's history is agents overturning confident verdicts; obituaries are not filed with a free falsification test on the table. Second live route the "every lever failed" line skips: every failed lever operated **inside one shared atom pool** — masked/sequential backfired by starving items to `atoms/K` and letting later writes overwrite earlier ones. The **Prop-L2 sharded store with per-unit budgets is architecturally immune to both failure modes and was never built.** Honest status: *capped in the single-store joint-write configuration; two revival routes (well-width geometry, lattice sharding) live and untested.*

**C3 — The sequencing inverted the strategy.** Deferring the CL entry to run gates was defensible once — but the CL entry is *simultaneously* **R4**, **R3's native headroom regime** (crowded-store retrieval of past items mid-stream), and **the honest home of R1's survivor** (scheduled per-item retention on a live stream). Deferring it deferred the only setting where three of the four dials meet an external benchmark, and R1/R3 were then scored as failing in settings this document had already closed. **Every w25 blocker is now cleared. Another gate wave is drift. The CL build is the wave.**

## A2 — W25 directions (binding; three tracks)

**Track 1 — THE CL ENTRY IS THE WAVE (R4 + R3 + R1 in one build).** Rehearsal-free Class-IL per the w23 recon: Split-MNIST → Split-CIFAR-10, from scratch, van de Ven taxonomy, **task1-only frozen φ (`phi_dim ≥ 16`)**, sizing rule (packing bound ≥ K) baked in from day one, designed store + controller + per-item decay. Mandatory table: tuned ER · iCaRL · **GDumb at matched memory** · EWC/SI as the known null. **Folded-in internal measurements:** (a) **R3-native** — retry measured on crowded-store retrieval of past items mid-stream (no oracle exists there; this is the headroom regime, obtained for free); (b) **R1-survivor** — scheduled per-item retention demonstrated on the live stream, worded **"designed scheduled retention" — the word "certified" is banned program-wide** (it is a defended DP term we do not satisfy). Split-CIFAR-10 is also where the strict φ should first genuinely bite (w24 flagged everything was MNIST) — report it either way.

**Track 2 — The R2 micro-track (cheap; runs beside Track 1, never competes with it).** In order, each gating the next: (1) the **free well-width dump** from a w23 checkpoint (tests the theorist's geometric account); (2) if it holds, the **one-flag `atom_init_width 0.30→0.15` experiment** (predicted to move the ceiling by `2^d`); (3) if the ceiling moves, the d-sweep — R2 revived with no lattice at all. In parallel: the **first real N-unit sharded store** (Prop L2's five predictions, §5.1; handle the five flagged code-path traps first, notably `init_scale` atom-scatter and relativistic non-separability) with **per-unit atom budgets** — the configuration immune to the two failure modes that killed masked/sequential in the shared pool.

**Track 3 — The R1 repair items.** (a) **Order-independent placement rule** — replace history-dependent refuse-and-relocate with deterministic content-addressed placement, converting the w24 blocking gap into a true exact-deletion property (an engineering task, not a thesis change); (b) the **MIA-distinguishability-vs-`leak·t` measurement** on a partially-decayed item — the genuinely open cell w24's recon identified; our novel contribution in unlearning-adjacent space, made without touching the DP vocabulary.

**Explicitly deferred, not dropped:** R5 (`CLULayer`) stays Phase-4; the d=6/d=8 ceiling confirmations continue as background; R19-4 remains an open proposal.

## A3 — Registry wording corrections (for the next curator pass; do not let the w24 verdicts fossilize)

- **R3 status →** *"The static-retrieval instantiation is CLOSED (4th confirmation of the metric-native ceiling: kNN/NN/oracle owns any protocol where the store is metric-native to the query). The compute-dial mechanism SURVIVES all controls (3rd independent confirmation, multi-seeded, 0 flips). Native-regime test = w25 CL-internal crowded-store retrieval. Do NOT record 'R3 dead.'"*
- **R2 status →** *"Capacity is CAPPED at K≈32 in the single-store joint-write configuration (multi-seed-confirmed; scale-invariance/quantization ruled out, |Δ|=0.010 null). Two revival routes LIVE and UNTESTED: the theorist's well-width geometric account (free check + one-flag experiment, w25 Track 2) and the Prop-L2 sharded store with per-unit budgets. Obituary premature; do NOT quote 'the write operator caps CLU at 32' unconditionally."*
- **R1 vocabulary →** *"'Certified' is banned (defended (ε,δ)-DP term; we are deterministic and satisfy none of it). Approved: 'designed scheduled retention' / 'exact per-item decay schedules.' Exact-deletion claims are BLOCKED until the order-independent placement rule lands (w25 Track 3). PALL (ICLR'25), SILO, SISA, Ticketed-L-U are must-cites; the open cell is MIA-vs-decay."*

## A4 — Governance fix (binding; prevents recurrence)

**The four-dials claim rule moves from review time to SCOPING time.** Every future task file must state, up front: **(1) which dial its claim addresses · (2) what the laundering control is · (3) which baselines would — and explicitly would NOT — falsify the claim.** Had `headroom-retry-benchmark`'s task file declared "a mask-informed oracle does not falsify the dial claim," the R3 verdict would have been written correctly on day one. The Hub applies this template to every w25 task file; the curator checks new task files against it at each wave pass.

## A5 — The one-line summary for the w25 Hub

**W24 ran the right experiments and scored them with the wrong scoreboard. Nothing measured contradicts the ambition — the dials all work, the mechanism is three-times-proven, every blocker is cleared. The correction is not new science: finish the two free checks w24 left on the table, and ship the one build where the dials finally meet a benchmark that cannot be won by a lookup.**

*Addendum filed by the Head + Advisor, 2026-07-25.*

---
---

# ADDENDUM 2 (2026-07-28) — W25 Review Ratified, the Five Decisions Answered, W26 Directions (two-advisor consensus)

> **Provenance & authority.** Head + two advisors (Advisor-1 = this document's author; Advisor-2's points relayed by the Head and merged **additively — no w26 idea removed**). The decision answers and w26 directions below are **binding**; the w26-scoping Hub applies them under the §A4 governance rule (every task file declares its dial, its laundering control, and what would and would not falsify the claim — the rule paid for itself in w25 and stands).

## B0 — W25 verdict (joint)

**The program's best wave, including its failures — nothing was scored on the wrong scoreboard.** ⭐ **The reach discovery is the most important scientific result the program has produced:** the capacity ceiling is neither the write operator (w23/w24 account) nor well-width geometry (theorist's account) but a **reach condition** — the read launches from the payload-zero manifold, and a well whose payload puts it far from that manifold is invisible from where the ball is dropped. Causal confirmation: halving read-out excursion (tolerance co-scaled) took the w23-firm wall cell 0.824 → **1.000** (payload error ÷5400) and a cell four rungs above the ceiling to 0.992, at both widths. Design rule: **well width is set by the read-out channel, not by address packing.** Also banked: the CL entry (best rehearsal-free on Split-MNIST 0.707 vs 0.196, launder honestly co-reported), R3's first non-loss (native-regime tie, mechanism survives a 5th control battery, auto-stop 1.4×), sharded write additivity (0.925 vs 0.880, same params, zero optimizer sync, 4–5× cheaper writes), **bit-exact deletion at negative packing cost** (61/64 vs 43/64), the membership-oracle hole (AUC 0.99985 post-eviction — the motivating attack our own placement rule defeats), and the pre-registered MIA null (decay ≡ boolean flag vs an exact adversary; the surviving differentiator is retrieval-geometry, not privacy). Two engineers pre-registered their own falsifiers and scored themselves refuted — the culture is working.

**⭐ Standing Head ruling (binding, program-wide):** *no direction is closed as lost; win-by-construction results (incumbents fail the class by construction) are SUPPLEMENTARY claims only — never the primary claim. The primary claim requires a genuine win on tasks where the baselines also do fairly well. We keep attacking with the cleverest ideas available, relentlessly.*

## B1 — The five w25 decisions, answered

1. **Duplicate-file cleanup: YES.** One `git rm` commit, push. Carried too long.
2. **Split-MNIST as "an external benchmark won": NO for the score sentence.** The incumbents fail Class-IL by construction → per the standing ruling this is a strong **supplementary** claim, recorded only in the two-clause form ("best rehearsal-free method on this benchmark, in a class whose incumbents fail it by construction"). The program's score sentence is unchanged until a contested win lands — this protects the credibility of that win when it arrives.
3. **Read-out excursion: YES — a legitimate interface parameter** (an *output code*, not a difficulty knob; every architecture chooses its output encoding). Commission as a **TWO-ARM task**: **(a) multi-channel payload** — the fixed value range split across payload channels, lowering per-axis reach at constant information; **(b) [Advisor-2] the annealed/continuation read** — widen atom widths on a schedule *during settling*, `σ_eff(t) = √(σ² + s(t)²)` (analytically free for Gaussians — a Gaussian convolved with a Gaussian is a Gaussian), decoupling storage width from read reach with **no change to the stored payload format at all**. Fairness conditions (binding, both arms): bits-per-item held constant · byte accounting pinned explicitly (no capacity smuggled via extra parameters) · payload noise ON (the current channel is noise-free; the free lunch must survive noise) · **baselines given the same format** · the laundering control travels.
4. **Localized-init default: NOT YET.** Run the **2×2 init×width factorial with the interaction term, 3 seeds** [Advisor-2 design]. Registered prediction (Advisor-2): *substantially one effect.* Neither lever becomes a default before this cell reads out.
5. **W26 shape: the merged list in §B3.** Spawn word for the documentation pass stands: `doc-curator-w25-sync`.

## B2 — The contested-win strategy (where the primary claim can come from)

Four-for-four laundering says the primary claim cannot come from any metric where the store approximates a lookup. W25 located the contested axes — each a dial claim, none win-by-construction:

- **Candidate 1 (nearest): the forgetting-vs-bytes Pareto frontier.** Replay is the *strongest* anti-forgetting method in existence — it does not fail by construction. The entry's own numbers: CLU retains 83% where parametric rehearsal-free methods retain 1%, and **forgets less than a replay buffer at the same item budget with 24.5× fewer floats**. Formalize forgetting/BWT as a function of memory **bytes** (not items), swept across budgets, against **tuned ER / DER++ / GDumb at every budget point**. Dominating a region of that frontier is a contested win on the axis the whole CL field optimizes, against its best methods. *Accuracy at matched memory is laundered; forgetting at matched bytes is not.*
- **Candidate 2: deletion cost at matched utility.** Once placement lands: bit-exact deletion, ~O(1) compute, negative packing cost, measured attack defeated (AUC 0.99985 → 0.5) — vs SISA-class/PALL incumbents who do well on this axis but pay retraining/maintenance per deletion. Gated on the citation scout.
- **Candidate 3: capacity-per-byte, if excursion unclamps the ceiling.** The w21 bits-per-param loss (~1.3 vs 2) was measured *under* the reach ceiling; if either B1.3 arm moves the wall, remeasure — the log-linear capacity figure returns with a mechanism story attached.

## B3 — W26 directions (merged; NOTHING removed — Head's instruction)

| # | task | notes |
|---|---|---|
| 1 | **Placement landing + AUC acceptance test** — co-headline | New controller verb `delete`; drive post-eviction hole-detection AUC 0.99985 → 0.5 under the real read path; **+ the array-scrub fix** (eviction currently leaves address/value verbatim in the arrays) [Advisor-2]. **Citation scout runs BEFORE any drafting** — the prior-art claims (SILO/SISA/Ticketed-L-U/PALL) are load-bearing and currently model-knowledge-only; the discrete skeleton of the rule is not novel and the paper must say so precisely |
| 2 | **The CL-capable encoder** — co-headline | The entry's single blocking dependency (Split-CIFAR null is a feature-space failure: kNN on the same features caps at 0.21). Cheap first step: AE arm at two embedding sizes; real answer: small conv/self-supervised encoder, task-1-only |
| 3 | **The matched-bytes forgetting frontier** — the contested-win experiment [both advisors; Advisor-2 insists] | Theorist/analyst formalization of the metric (forgetting/BWT vs bytes), **pre-registered against tuned ER and GDumb** (+ DER++, budget sweep). Starts on Split-MNIST now; extends when task 2 lands |
| 4 | **The two-arm R2 excursion task** | Per B1.3, both arms, all five fairness conditions in the task file |
| 5 | **The joint init×width cell** | 2×2 factorial + interaction, 3 seeds, Advisor-2's registered prediction on file |
| 6 | **The read-out-channel theorist task** (consolidated) | One rigorous pass over the same object: the reach condition · the payload-dependent-lifetime ruling (retention ∝ −value², r=−0.85 — currently undercuts the lifetime dial) · the theory of both excursion arms. **Include the cheap calibration option** [Advisor-2]: user specifies half-life, the store *solves* per-item leak numerically — potential untouched |
| 7 | **Carried, not dropped** | the un-reproduced baseline retune (before publication) · the single-seed retry-threshold observation (contradicts w24 — re-measure) · d∈{6,8} Stage-1 confirmations (background, compute-permitting) · the high-load sharding probe · R5 stays Phase-4 |

**Nothing is closed:** the MIA privacy angle is *scoped* (retrieval-geometry differentiator + resolution-bounded adversary honestly stated; the privacy claim pivots to deletion-exactness), the R2 width route is closed *by mechanism* (the R2 direction itself lives via both excursion arms + sharding), R4's accuracy launder pivots to the forgetting axis. R19-4 remains open.

## B4 — One line for the w26 Hub

**W25 found the mechanism and the contested axes. W26 is three named experiments from the primary claim: close R1 completely (placement + scrub + citations), build the encoder, and formalize the frontier — with the two-arm excursion task carrying R2's revival alongside. Nothing closed, nothing removed, every task file dial-tagged.**

*Addendum 2 filed by the Head + both Advisors, 2026-07-28.*
