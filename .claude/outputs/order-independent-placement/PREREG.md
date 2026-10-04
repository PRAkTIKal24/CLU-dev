# PREREG — order-independent-placement (physics-theorist, w25)

Written BEFORE any harness run (protocol §5 pre-registration rule). Repo @ `63c668d`.
All predictions below are derived, with the derivation stated; the harness measures them afterwards.

## Rule family (committed now)

**PGCP — priority-greedy canonical placement.** Every item carries an intrinsic record
`r = (key κ, payload a, leak λ, permanent, born)`. Deterministic functions of the key alone:

- `g(κ)` — hash point in the address disk (two splitmix64-style hashes → uniform-in-disk),
- `prio(κ)` — a 64-bit hash, total order on keys (descending; distinct keys ⇒ no ties),
- `π(κ)` — probe order = the fixed hex-lattice cells (spacing `d_safe`, origin-anchored, kept iff `|c| ≤ R`) sorted by `(|c − g(κ)|, cell_index)`.

**Canonical placement** `pos_S`: process live keys in **descending priority**; each key takes the
first cell in its own probe order not already taken by a higher-priority key; no free probe cell ⇒ ⊥
(refused/overflowed). Store(S) = atoms at `{pos_S(κ)}` with payloads/amps from the records,
slots packed in priority order (canonical slot assignment ⇒ bit-identity is meaningful).

Rungs benchmarked:
- **P0** (control, no collision handling): site = `g(κ)` exactly; canonical thinning = Matérn-III-type
  priority greedy with exclusion `d_safe`.
- **P1** (the deliverable): hex-lattice PGCP as above.
- **RR** (incumbent benchmark): refuse-and-relocate as built (`admission.admit_site`, 400
  uniform-in-disk candidates), numpy port, sequential — the order-DEPENDENT baseline.

Geometry pinned to `controller-mvp`: `s=0.35`, `d_safe = 4.4·s = 1.54`, fixed disk `R=2.0`,
sized disk `R(K) = d_safe·√((√3/2)K/π)`; K ∈ {8,16,32,64}; 20 seeds for anything stochastic.

## Predictions (with derivations)

**H1 — exactness (pass criterion for the permutation harness).** For any set of records:
canonicalized store arrays (float64 `(centers, payloads, amps, active)` in priority slot order) are
**bit-identical (`tobytes()` equal, max |Δ| = 0.0)** across (a) all 24 write orders of a 4-item set,
(b) 200 random write orders at n ∈ {8,16,40,64}, (c) 100 random write/delete interleavings
reaching the same final record set, and (d) `delete(Store(S), i) = Store(S∖{i})` including
mid-decay deletion (delete a leaky item at tick 3 of 5 ≡ never written, for the survivors,
bit-identically) and delete-then-compare-to-never-written. Any nonzero diff at any tolerance = **FAIL**
(no "small numerical noise" excuse — the rule is exact by construction or it is not).
Derivation: uniqueness by induction on priority order — `pos_S(κ)` references only strictly-higher-priority
keys, so it is a set function; amps factorize per record (`amp = base·e^{−λ·(t−born)}`).

**H2 — spacing invariant.** Min pairwise distance of placed sites ≥ `d_safe` **exactly** (lattice
neighbors sit at exactly `d_safe`) — the admission certificate is carried by the geometry, not checked
per write.

**H3 — packing, sized geometry (the price of exactness — headline number).**
- **P1 admitted = min(K, N_cells(R(K)))** deterministically (probe order covers all cells).
  `N_cells ≈ πR²/((√3/2)d²) = K` up to boundary terms; predicted band **N_cells ∈ [0.85K, 1.15K]**,
  point estimate ≈ K ⇒ **admitted fraction ∈ [0.85, 1.0], point ≈ 0.95 at K=64** —
  **BEATING RR's measured 42.8/64 = 0.669** (controller-mvp §2). Prediction: *the price of exactness
  on sized geometry is NEGATIVE (a packing gain)*, because the lattice is a designed near-optimal
  packing while RR packs at random-candidate density (≈0.67 of bound, cf. N74 6.0±0.9 vs 6.12).
- **P0 admitted**: bounded below by Matérn II thinning: with sized geometry `λ·πd² = 2π/√3 = 3.628`
  independent of K ⇒ `E[admitted] ≥ (√3/(2π))K·(1−e^{−2π/√3}) = 0.268K`; bounded above by 2-D RSA
  jamming (coverage 0.547 ⇒ 0.603K). Point estimate **0.40K ± 0.08K** (K=64: 26 ± 5).
  P0 is the honest cost of *refusing to handle collisions at all*: ≈0.40 ≪ RR 0.669 ≪ P1 ≈0.95.

**H4 — packing, fixed geometry (R=2).** Hex lattice anchored at origin: origin + 6 neighbors at 1.54,
second ring at 1.54√3 = 2.67 > 2 ⇒ **N_cells = 7 exactly** ⇒ P1 admitted = min(K,7) = 7 for K ≥ 8 —
**above RR's 5.2 ± 0.4** and above the area-form bound 6.12 (the area bound is asymptotic; a designed
lattice legitimately reaches 7 in-disk points at pairwise ≥ 1.54). P0 fixed: same 0.268K–0.603K logic
with boundary inflation; at K=64, exclusion saturates ⇒ predict 5–8.

**H5 — retention per admitted at exact-lattice spacing = 1.000** (pass ≥ 0.98), by gradient-flow
relaxation reads (numpy V identical to `AtomStorePotential.__call__`), 16 jittered queries/item,
4-level payload codebook, tol = 0.35×codebook spacing. Derivation: at `d = d_safe = 4.4s` a neighbor
atom contributes `e^{−4.4²/2} = 6.3e-5` of its gradient scale; ≤6 neighbors ⇒ payload contamination
≤ 4e-4·|a|_max ≪ tol. (Tighter than RR's achieved 1.61 spacing, but still 4 decades under threshold.)

**H6 — deletion cascade cost.** Deleting one item from the full K=64 sized store moves (re-places)
lower-priority items whose canonical cell changes. Predicted **median ≤ 2 moves, mean ≤ 5, max ≤ 15**
(load factor ≈ 1 makes promotion chains possible; at lower load the mean should drop ≪ 1).
Each move = `evict + with_item` on the shipped store — cheap vs a read but not free; measured
distribution is the reported cost.

**H7 — real-PyTree bit-identity.** Rebuilding the canonical configuration through the *shipped*
`AtomStorePotential.with_item/evict` (JAX 0.9.0) from two different histories with the same final
record set yields byte-equal `centers/payloads/amps/active` and byte-equal `V(q)` on 64 random queries.
No production-code change is required for the store itself (the controller wrapper is where the
engineer's change lives).

## What would falsify the design (declared now)
- Any permutation/interleaving producing a non-identical canonical store (H1) ⇒ the uniqueness proof
  has a hole (e.g. hidden state in the probe order) ⇒ report as the failure it is.
- P1 admitted < RR admitted on sized geometry (H3) ⇒ the price of exactness is positive; quantify it —
  that trade is then the result (task Item 3 explicitly allows this outcome).
- Retention per admitted < 0.98 at exact-lattice spacing (H5) ⇒ `d_safe = 4.4s` is not actually safe
  at the lattice's tighter spacing ⇒ the rule must inflate the lattice constant and re-pay packing.

## Out of scope (declared now, per task Item 2)
LRU/staleness eviction (historical by definition); full counterfactual exactness over *refused* items
without a waitlist (P2 is specified in the report but not benchmarked); φ-level and curvature-residual
leak channels (mia-decay-measurement owns those).
