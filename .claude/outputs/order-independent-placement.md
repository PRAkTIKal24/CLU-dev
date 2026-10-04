# order-independent-placement — physics-theorist report

Task + acceptance criterion: convert N99 into exact store-level deletion — a set-function placement rule + proof + green permutation harness + the packing-cost price of exactness + the scoped paper sentence (PREREG required).
Status: **done** (theory + reference implementation + all pre-registered checks green; one mild H6 prereg miss reported; production landing specified for the engineer, not landed by me).

> **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). Four items.**
> 1. **N99's disposition changes**: exit (b) is delivered at theory+reference level. N99 stays registered (the MVC-0 controller *as shipped* is still history-dependent); the entry needs an "Update" block: an order-independent rule (PGCP) now exists, proven + permutation-green, pending engineering landing. Owner: doc-curator next pass, after Hub review.
> 2. **R1 wording**: "exact store-level deletion (scoped)" becomes claimable **only after** (a) the engineer lands PGCP in `Controller` and (b) web-scout verifies the prior-art citations below (Blelloch–Golovin, Naor–Teague, Hartline et al. — load-bearing, asserted from my training knowledge, NOT verified from primaries this session). ⛔ "certified" stays forbidden (CM-22) — nothing here supplies (ε,δ).
> 3. **The rematch per-offered number moves**: PGCP's deterministic sized-geometry admission is **61/64 = 0.953** vs refuse-and-relocate's 42.8/64 = 0.669 (controller-mvp §2). Any site quoting "0.669 is the sized-geometry ceiling" must gain "…of the stochastic relocation rule; the canonical lattice rule admits 61/64." Owner: whoever writes the CL/controller paper section.
> 4. **A new controller verb is required**: `Controller.delete(item_id)` does not exist in shipped code (only budget-evict and decay). The deletion claim has no code path until it lands. Owner: experiment-engineer (spec §6).

---

## ⭐ DIAL DECLARATION (echo, protocol §7)
- **Dial:** lifetimes + admission (R1's structural underpinning).
- **Laundering control:** TTL-dict / kNN-row-delete claim-structure comparison — **run, see §5**: at the deletion-semantics level PGCP makes the CLU store exactly as deletable as a canonical dictionary — that is the point, and it is stated as such. The differentiators (continuous amplitude law, physical decay/permanence coexisting with a commutation proof, the spacing-certificate isolation calculus, packing economics) are named as what they are — properties *around* the deletion claim, not the deletion claim itself.
- **Falsifies:** a proof that no useful placement on a capacity-constrained continuous store can be order-independent. **Outcome: the opposite — a constructive rule exists and packs BETTER than the shipped order-dependent rule.**
- **Does NOT falsify:** eviction-under-overflow remaining order-dependent for recency policies (confirmed intrinsic, scoped out with a precise boundary, §4); performance cost vs refuse-and-relocate (measured: the cost is *negative* on packing, positive on delete-time churn — §5).

## Flag-provenance table (mandatory)
| item | value |
|---|---|
| repo commit | `63c668d` (local `main`; no tracked code touched, no branch) |
| harness | `.claude/scratch/order-independent-placement/{pgcp.py, run_exactness.py, run_costs.py, run_jax_h7.py}` — pure numpy float64 except H7 (JAX 0.9.0 / eqx 0.13.4, main venv, no `uv sync`) |
| PREREG | `.claude/outputs/order-independent-placement/PREREG.md`, written before any run |
| geometry | s=0.35, d_safe=4.4·s=1.54, α=0.02, κ=1.0, s_pay=s, dim=3, amp_floor=0.05, leak 0.35, codebook {±0.25,±0.75}, tol=0.35·0.5 — pinned to controller-mvp §Flag-provenance |
| seeds | explicit per test: 0 (perm rng), 101, 200+n, 300, 4000+trial, 500, 600, 9, 777, 2025, 31337, 10000+97·seed+K (seed=0..19) |
| training config | N/A — no CHLU training anywhere in this task (no lyapunov/langevin/anchor/epoch flags in effect) |

---

## §1 The rule — PGCP (priority-greedy canonical placement)

Every item is a record `r = (key κ, payload a, leak λ, permanent, born)`. Fixed store geometry `G`: address disk radius `R`, hex lattice `Λ` of cells with spacing `d_safe`, anchored at the origin, kept iff `|c| ≤ R`. Three deterministic per-key functions (splitmix64-based):
- `g(κ)` — hash point, uniform-in-disk;
- `prio(κ)` — 64-bit priority, strict total order (tie-break by key);
- `π(κ)` — probe order = all cells of `Λ` sorted by `(|c − g(κ)|, cell index)`.

**Canonical placement** `pos_S`: process the live keys of `S` in **descending priority**; each key takes the first cell of *its own* probe order not already taken by a higher-priority key; if none is free, `⊥` (refused/overflowed). `Store(S)` = atoms at `{pos_S(κ)}` with payloads/current amps, **slots packed in priority order** (canonical slot assignment is what makes bit-identity well-posed; the shipped `AtomStorePotential` needs no change — §H7).

The task's adversarial question — *can relocation be a function of the conflicting keys as a set?* — is answered by dissolving relocation entirely: there is no "conflict event" to resolve, because the configuration is defined globally (a set function) and maintained incrementally by bounded fix-ups. Refuse-and-relocate's 400-candidate stochastic search is replaced by a deterministic assignment that is simultaneously the admission gate (spacing is a lattice invariant) and the placement rule.

Rungs: **P0** = pure content-addressing, no collision handling (control); **P1** = the deliverable above; **P2** = P1 + a waitlist of refused records, giving full counterfactual exactness over the *offered* stream under overflow (specified §4, not benchmarked — declared out of scope in PREREG).

## §2 Proofs

Assumptions: distinct keys; fixed geometry `G` (part of the store, not of history); float ops deterministic (single-threaded IEEE-754 — holds in the harness and in the shipped store's slot-wise writes).

**Theorem 1 (unique representation — Store is a set function).** For any finite record set `S`, `pos_S` is well-defined and unique, and depends on nothing but `S` and `G`.
*Proof.* Strong induction over keys in decreasing priority. The defining recursion for key `k` references only `{pos_S(k′) : k′ ≻ k}`, which is uniquely determined by the induction hypothesis; the minimum over a fixed finite probe list is unique. No arrival order appears in the definition. ∎

**Lemma (suffix stability).** For `i ∈ S` and every `k ≻ i`: `pos_{S∖{i}}(k) = pos_S(k)`.
*Proof.* The greedy state seen by any `k ≻ i` (occupied cells of keys `≻ k`) is identical in `S` and `S∖{i}`, by downward induction from the top priority. ∎

**Theorem 2 (exact store-level deletion).** Define `delete(Store(S), i)` = remove atom `i`, then re-run the greedy over keys `≺ i` (the fix-up cascade). Then `delete(Store(S), i) = Store(S∖{i})` **bit-identically**.
*Proof.* Keys `≻ i` are unmoved (Lemma); the fix-up recomputes keys `≺ i` by exactly the recursion that defines `pos_{S∖{i}}`; amps factorize per record (Thm 4). Corollary: any interleaving of writes and deletes reaching live set `S` yields the same store as writing `S` fresh — the Thudi-compliant *algorithmic-level* form of the claim (the one auditable form; unlearning-recon §2.3). ∎

**Theorem 3 (spacing certificate for free).** Distinct cells of `Λ` are ≥ `d_safe` apart (nearest neighbors exactly `d_safe`), so every reachable configuration satisfies the admission invariant *by construction* — the set-predicate admission test the task said survives is absorbed into the geometry. Isolation calculus unchanged: neighbor gradient factor `e^{−d_safe²/2s²} = e^{−4.4²/2} = 6.3e-5` (the controller-spec's "5 orders").

**Theorem 4 (decay/permanence commute with deletion).** The tick operator acts diagonally on records: `amp_k(t) = base_k·e^{−λ_k(t−born_k)}`; floor self-eviction is a per-record predicate. Hence `tick ∘ delete_i = delete_i ∘ tick` on survivors: deleting a decaying item at *any* point of its schedule leaves every survivor bit-identical to the never-written history. `born_k` is item-*intrinsic* history and is part of the record — order-independence means independence of *other* items' presence and of arrival order given the records; stated, not hidden. ∎

**Status: proven** (constructive, elementary induction) **and numerically verified to the bit** (§3).

## §3 Numerical verification (H1–H7 vs PREREG)

`run_exactness.py` — **H1/H2 exactness, pass criterion = bit-identity (0.0, `tobytes()` equality), no tolerance:**
| test | result |
|---|---|
| T1: n=4, all 24 write orders | **24/24 bit-identical** |
| T2: n∈{8,16,40,64}, 200 random orders + fresh-build each | **all bit-identical**; min spacing = 1.540000 = d_safe exactly, every n |
| T3: 20-item pool, 100 random write/delete interleavings → same final set | **100/100 bit-identical** to fresh `Store(S_final)` |
| T4: leaky victim deleted at tick 3 of 5 vs never-written (victims 1,3,7) | **bit-identical survivor state**, all cases; permanent item amp = 1.0 after 5 ticks; leaky survivors at e^{−1.75} = 0.173774 exactly |
| T5: write-then-delete vs never-written | **bit-identical** |

Non-vacuousness: worst-case (ascending-priority) arrival order at K=64 produced **70 displacement moves across 64 inserts (28 inserts cascaded, max chain 8)** — the incremental fix-up path is genuinely exercised, not idle.

`run_costs.py` — **H3/H4 packing (the price of exactness), 20 seeds where stochastic:**
| K | geom | R | N_cells | **P1 (deterministic)** | P0 mean±sd | RR mean±sd |
|---|---|---|---|---|---|---|
| 8 | fixed | 2.00 | 7 | 7 | 3.2±0.7 | 5.3±0.7 |
| 16 | sized | 3.23 | 19 | 16 (=K) | 6.5±1.0 | 12.2±1.3 |
| 32 | sized | 4.57 | 31 | 31 | 12.2±1.5 | 22.8±1.0 |
| 64 | fixed | 2.00 | 7 | 7 | 4.8±0.8 | 6.0±0.7 |
| 64 | sized | 6.47 | 61 | **61 (0.953)** | 23.8±1.9 | 43.0±1.7 |

- **H3 CONFIRMED**: P1 sized K=64 admits **61/64 = 0.953** (prereg band [0.85,1.0], point 0.95) vs in-harness RR **43.0±1.7** (matches controller-mvp's 42.8 — the port is faithful) vs P0 23.8 = 0.372K (prereg point 0.40±0.08K; Matérn-II bound 0.268K < measured < jamming 0.603K ✓). **The price of exactness is NEGATIVE: +18 items (+42%) over refuse-and-relocate**, because the lattice is a designed near-optimal packing while stochastic relocation packs at random-candidate density.
- **H4 CONFIRMED exactly**: fixed R=2 lattice = **7 cells** (origin + one ring at 1.54; second ring at 1.54√3 = 2.67 > 2), predicted 7 in PREREG before measurement — above RR's 5.2–6.0 and above the asymptotic area bound 6.12 (the bound counts area, a designed lattice legitimately places 7 in-disk points at pairwise ≥ 1.54).
- Sizing rule for the engineer: `R(K)` from `radius_for_capacity` has a boundary deficit (N_cells: 61 at K=64, 7 at K=8). **Multiplier 1.05 gives N_cells ≥ K for K ∈ {16,32,64,128}** (19/37/73/139); K=8 is small-K discrete (needs next-ring radius, mult ≈1.35).

**H6 cascade cost (delete-time churn — the real price):** full store (61/61 cells) mean **2.84** moves/delete, median **3**, max 7, 8% zero-move; half load (32/61) mean **0.22**, median 0, 81% zero-move. **Prereg partial miss, reported honestly: median 3 vs pre-registered ≤2 at full load** (mean 2.84 ≤ 5 ✓, max 7 ≤ 15 ✓). Each move = `evict + with_item` (bookkeeping-cheap; no physics run), but *deletion moves other items' addresses* — reads must use the record's current center (the controller owns the records, so this is mechanical, but it must be stated: exact deletion of `i` legitimately relocates lower-priority survivors, exactly as the never-written store would have placed them).

**H5 retention at exact-lattice spacing:** 61-item store at min spacing exactly 1.5400; gradient-flow relaxation reads (numpy `V` identical term-by-term to `AtomStorePotential.__call__`), 16 jittered queries/item, basin AND |read−payload|<0.175: **976/976 = 1.0000** (prereg ≥0.98). ⚠ Scope: gradient-flow reads, not the shipped two-phase Verlet read (γ_address 0.05×400 → γ_read 0×800) — the engineer's rematch cell must confirm with the real read path before 0.953 per-offered is quoted as a measured number.

`run_jax_h7.py` — **H7 on the shipped PyTree (JAX 0.9.0):** two histories (16 written + 4 deleted vs 12 written scrambled) → canonical rebuild through the real `AtomStorePotential.with_item` → `centers/payloads/amps/active` **byte-equal**, `V(q)` **byte-equal on 64 random queries (max |ΔV| = 0.0)**; control: same items in non-canonical slot order → arrays differ (why the canonical re-pack step exists). **The shipped store class supports PGCP with zero production-code change; only the controller wrapper changes.**

## §4 The boundary (Item 2 — state it, don't hide it)

(a) **Below capacity (|S| ≤ N_cells): full exactness.** Admission is unconditional (probe order covers all cells), live set = written-minus-deleted, `Store(S)` bit-exact. This is the clean claim.
(b) **Overflow under PGCP = priority eviction**, a set-function policy: the overflowed lowest-priority key loses placement. Proven bonus: for *insert-only* streams the live set itself is order-independent (⊥ keys occupy nothing and ⊥ is monotone under further inserts, so discarding them is safe — live(P) = placed(pos_P)). The same argument covers **static-attribute eviction** (top-B by item-intrinsic attribute). With interleaved *deletes* under a binding budget, a previously-discarded key would counterfactually return: `Store(S_live)` stays exact, but counterfactual exactness over the offered stream needs the **P2 waitlist** (refused records kept in a plain — trivially deletable — side dict; on any delete, re-run admission over the waitlist by priority).
(c) **LRU/staleness eviction is intrinsically historical** (`last_used` is query history) and stays permanently outside the deletion claim. The shipped `evict_policy="staleness"` path must never be combined with the deletion claim; `"depth"` qualifies only because amp = base·e^{−λ·age} is item-intrinsic.

**The scoped paper sentence (deliverable):** *"Placement in the store is canonical — a deterministic function of the live item records and the store geometry alone — so store-level deletion is exact: removing an item reproduces, bit for bit, the store that holds the remaining records, with each survivor's scheduled decay and permanence unaffected (deletion and decay provably commute). The claim covers stores operating below capacity or under set-function (priority/attribute-based) eviction; recency-based eviction is intrinsically history-dependent and is excluded. This is a store-level guarantee only: the frozen encoder and any residue of past writes in a learned landscape are separate channels, measured separately; we do not claim certified (ε,δ) unlearning."*

## §5 Honest accounting: laundering control + residual channel (Item 4)

**Claim-structure vs a TTL-dict:** with PGCP, `delete` on the CLU store has *exactly the claim structure of `del d[k]` on a canonical dictionary* — that is what "exact deletion" means, and pretending otherwise would be laundering. What the dict does not have, stated as what they are: (i) the continuous amplitude law (decay is a measured physical trajectory, not a TTL bookkeeping bit) with the Thm-4 commutation proof; (ii) the spacing certificate tying placement to a quantitative isolation calculus (6.3e-5 neighbor leak) on a landscape a dynamical read relaxes into; (iii) permanence and decay coexisting in one potential; (iv) the packing economics of §3. The novelty claim must be *exactness in a continuous designed landscape with decay/permanence coexisting* — never exactness per se (the discrete skeleton is prior art, below).

**Residual channels (one paragraph, per task):** even with bit-exact store-level deletion, (1) a φ trained on data that contained item *i* still encodes *i* at the encoder level (here φ is task1_only/frozen per w24, which narrows but does not close this); (2) on any *learned* V, `evict` leaves curvature residue — PGCP applies to the designed store, where eviction of an atom is exact by construction, and says nothing about learned-landscape residue; (3) whether a *partially decayed* well (amp 0.06) is MIA-distinguishable at one half-life is unmeasured — `mia-decay-measurement` owns it. **Store-level scope only; never system-level deletion.**

**Prior art (novelty trap, CM-22 — ⚠ citations asserted from training knowledge, NOT verified from primaries; web-scout must verify before any draft cites them):** PGCP's discrete skeleton **is** strongly-history-independent (uniquely-represented) hashing: Naor & Teague, STOC 2001 ("anti-persistence: history-independent data structures"); Hartline et al. (SHI ⟺ canonical representation, for reversible structures); Blelloch & Golovin, FOCS 2007 (SHI open-address hashing via key priorities — our priority-displacement rule is theirs, transplanted to a continuous energy landscape with a metric spacing constraint); Micciancio 1997 (oblivious 2–3 trees). Consistent hashing (Karger et al., STOC 1997) is the order-independent-assignment neighbor without spacing/geometry semantics. SISA (Bourtoule et al., S&P 2021) / DaRE trees are exact-unlearning-by-retraining-structure — different mechanism, must be cited as the arena. This answers the recon's "search next (1)": yes, the caching/data-structures literature owns order-independence of *discrete* stores; our contribution is the continuous-landscape + decay/permanence + certificate composition, and the paper must say exactly that.

## §6 Engineering spec (for experiment-engineer — I touched no tracked code)

1. **`chlu/core/placement.py`** (new): port `.claude/scratch/order-independent-placement/pgcp.py` (`splitmix64`, `hash_point`, `prio`, `hex_cells`, probe orders, `CanonicalPlacer` with insert/delete fix-up cascades + move log). ~150 lines, pure numpy, no JAX needed.
2. **`Controller`**: flag `placement ∈ {"relocate" (default, unchanged), "canonical"}`. Under canonical: `offer` consults the placer (admission = "got a cell"; the spacing gate is a lattice invariant — assert `min_spacing ≥ d_safe` in tests, don't check per write); **new verb `delete(item_id)`** implementing Thm-2 (remove + fix-up; apply moves as `evict`+`with_item` per moved record; update `records[*].center`); slot layout = canonical priority order after every op (rebuild is O(n) `with_item` calls — H7 shows byte-identity; cheaper in-place slot moves are an optimization, not a requirement). LRU eviction hard-errors under `placement="canonical"` + any deletion-claim flag; `"depth"` allowed.
3. **Tests**: port T1–T5 as pytest with `tobytes()` asserts; packing regression `N_cells(R(K)·1.05) ≥ K` for K∈{16,32,64,128}; a cascade-count smoke.
4. **Rematch cell** (pre-registerable now, from this report): controller-mvp `on_sized` K=64 with canonical placement → predicted admitted **61/64 (deterministic)**, per-admitted **1.000**, per-offered **0.953** under the real two-phase read; sizing multiplier 1.05 → 73 cells → predicted per-offered **1.000**.
5. Trap carried from N98: if atoms get an *init* near their site (lattice builds), localize per-site — the `init_scale` scatter violation applies to any PGCP-adjacent build too.

## Verdict
- **Proven:** Theorems 1–4 + suffix-stability lemma (constructive induction; assumptions stated) — `Store(S)` is a set function; `delete = set-minus`, bit-exact; decay/permanence commute with deletion; spacing certificate holds by construction.
- **Strongly evidenced (measured):** bit-identity across 24 exhaustive + 800 random orders + 100 interleavings + mid-decay deletes (0 mismatches at 0 tolerance); packing 61/64 vs 43.0/64 (RR) vs 23.8/64 (P0) at K=64 sized; fixed-geometry 7 vs 5.2; cascade 2.84 moves/delete at full load (median 3 — mild H6 prereg miss), 0.22 at half load; retention 1.0000 at exact-d_safe spacing under gradient-flow reads; H7 byte-equality on the shipped PyTree.
- **Conjectured / open:** per-offered 0.953 under the *real* two-phase Verlet read (engineer's rematch cell); the P2 waitlist's overflow-counterfactual exactness (specified, proof sketched, not implemented); prior-art citations pending web-scout verification; semantic-address quantization cost (≤ covering radius d_safe/√3 = 2.54·s below capacity) is zero in today's random-proposal harness and becomes real only when φ-derived addresses carry similarity — flagged, unmeasured.

How I verified: commands + full observed numbers in §3; scripts under `.claude/scratch/order-independent-placement/`; every quantitative claim traces to a printed line reproduced above.
Git footprint: none (no tracked files touched; all artifacts under `.claude/`).

Open questions / follow-ups / risks:
1. Web-scout: verify Naor–Teague / Hartline / Blelloch–Golovin / Micciancio primaries (load-bearing for the novelty sentence). 2. Engineer: land §6 (incl. the missing `delete` verb — the claim has no code path until then) + rematch cell with the real read. 3. Analyst/MIA task: partially-decayed distinguishability (owns the measurement side of Item 4). 4. Risk: if the real Verlet read at exact-d_safe spacing underperforms gradient-flow relaxation (H5's scope gap), the lattice constant must inflate and re-pay some packing — quantified fallback: multiplier m on d_safe costs ≈ m⁻² in cells.

## Proposed handover updates (for the Hub)
- **§7/N99**: add Update block — *"w25 `order-independent-placement`: exit (b) delivered at theory+reference level. PGCP (priority-greedy canonical placement — SHI hashing transplanted to the continuous designed store): Store(S) proven a set function, delete = set-minus bit-exactly, decay/permanence commute (Thm 4), permutation harness green at 0 tolerance (24 exhaustive + 800 random orders + 100 interleavings + mid-decay deletes). Price of exactness is NEGATIVE on packing (sized K=64: 61/64 = 0.953 vs RR 42.8/64 = 0.669; fixed: 7 vs 5.2) and positive on delete churn (2.84 moves/delete at full load). Boundary: below-capacity or set-function eviction only; LRU permanently excluded. Blocking before the claim ships: engineer landing (incl. the missing `Controller.delete` verb) + prior-art citation verification (SHI data structures — the discrete skeleton is NOT novel). 'Certified' remains forbidden (CM-22)."*
- **§8**: R1 upgrade path: "scheduled retention" → "scheduled retention + exact store-level deletion (scoped)" is now an engineering + citation task, not a theory gap.
- **Quotation guard**: never quote "0.953" as measured retention until the real-read rematch cell runs (it is currently: deterministic admission 61/64 × gradient-flow-verified per-admitted 1.000).
