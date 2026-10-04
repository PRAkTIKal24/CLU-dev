# Task: order-independent-placement — convert N99 from a blocking gap into exact deletion (w25)

- **Agent:** `physics-theorist` (design + proof + small numerics; flags the ~small code change for the engineer or lands it if truly surgical) · **Output:** `.claude/outputs/order-independent-placement.md` · **Branch:** none expected (theory + verification harness in scratch; if you do land code, take a branch + worktree per §3.2)
- **Read first:** `.claude/AGENT_PROTOCOL.md` **§7 (dial declaration)** · `.claude/outputs/unlearning-recon.md` (recon item 3 — the gap; §2.3 Thudi — the *algorithmic-level* claim standard; the search-next item on caching/streaming-literature order-independence) · `.claude/outputs/controller-mvp.md` §1 (the placement/eviction verbs as built) · `negative_results.md` **N99**

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** lifetimes + admission (the R1-survivor's structural underpinning).
- **Laundering control:** the TTL-dict / kNN-row-delete comparison — if the order-independent CLU store's deletion story is indistinguishable *in claim structure* from a dict delete, say so; the differentiators (continuous amplitude, physical decay, packing interaction) must be stated as what they are.
- **Falsifies:** a proof that no useful placement rule on a capacity-constrained continuous store can be order-independent (that would scope R1 permanently to "scheduled retention," and is a *valuable* outcome — report it as the theorem it is).
- **Does NOT falsify:** eviction-under-overflow remaining order-dependent (expected — see scope); performance costs relative to refuse-and-relocate (report the trade, it doesn't kill the claim).

## Why — N99, the blocking gap
`refuse-and-relocate` places item *j* as a function of item *i* being present; LRU evicts *k* because *i* occupied budget. So removing *i* does **not** reproduce the never-written store — the definition of exact deletion fails, and R1 cannot make even the *algorithmic-level* claim (Thudi: the only auditable form) until placement is a function of the live item set only. The Head's w24 ruling: an engineering-shaped fix — **deterministic content-addressed placement** — converts the gap into real exact deletion.

## Item 1 — design the rule
A placement rule `site(item) = f(item key, store geometry)` — independent of arrival order and of other items' presence. Candidate family (evaluate, don't assume): content-addressed site = deterministic map of the item key into the address space, with **deterministic, order-free collision handling** (e.g. fixed probe sequences per key, lattice/grid quantization of φ-space, or farthest-point-from-key-seed) such that for any *set* S of live items the final configuration `Store(S)` is unique — `Store(S)` never depends on the order S was written or on items since deleted.
⚠ **Be adversarial about the admission gate:** the spacing test itself (admit iff min-dist ≥ d_safe) is a *set* predicate and survives; **refuse-and-RELOCATE is the order-dependent part.** The design question is whether relocation can be made a deterministic function of the *conflicting keys as a set* (e.g. all colliding keys re-placed by a symmetric rule) rather than of arrival history.

## Item 2 — the boundary (state it, don't hide it)
**Eviction under overflow is inherently historical for recency policies.** Scope precisely: (a) placement = order-independent (the claim this task delivers); (b) eviction = order-independent ONLY for set-function policies (evict-lowest-priority-by-item-attribute, deterministic on the live set) — LRU/staleness is not and stays out of the deletion claim; (c) the deletion claim therefore holds for stores operating **below capacity or with set-function eviction**, and the paper sentence must carry that scope. Check the recon's lead: does the caching/streaming-algorithms literature already have an order-independence result to cite or reuse (consistent hashing is the obvious neighbour — and also a novelty risk to check).

## Item 3 — prove and verify
- **Proof:** `Store(S)` well-defined as a set function under the rule; corollary `delete(Store(S), i) = Store(S∖{i})` — exact deletion, algorithmic-level, auditable (Thudi-compliant).
- **Numerics:** a permutation-test harness — random item sets, all/many write orders, interleaved deletes → **bit-identical stores** (or identical to the site tail). Include the decay/permanence machinery: deletion of a decaying item mid-schedule must equal never-written.
- **Cost:** placement quality vs refuse-and-relocate (packing efficiency achieved by the deterministic rule vs the 42.8/64 relocation benchmark; admitted-fraction deltas). If the deterministic rule packs worse, quantify the price of exactness — that trade IS the result.

## Item 4 — the residual channel (one paragraph, honest)
Even with exact store-level deletion: φ trained on data containing the item, and residual curvature after `evict`, are separate leak channels (recon §Item-3). Do not claim system-level deletion; state store-level scope. The MIA-vs-decay task (`mia-decay-measurement`) owns the measurement side.

## Acceptance
PREREG (which rule family, predicted packing cost band, the permutation-test pass criterion). The rule + proof + permutation harness green + the cost table + the scoped paper sentence (with the eviction boundary). If the outcome is an impossibility/triviality result instead, that is acceptance too — write it as the theorem. Echo the DIAL DECLARATION.

## ⚠ Standing traps
- Words: *scheduled retention / exact store-level deletion (scoped)*; ⛔ never "certified", never "unlearning" (CM-22 m/n/o).
- Novelty: consistent-hashing/SISA-adjacent prior art must be cited; the claim is exactness **in a continuous designed landscape with decay/permanence coexisting**, not exactness per se.
