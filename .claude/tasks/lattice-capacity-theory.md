# Task: lattice-capacity-theory — does sharding across units multiply capacity without optimizer synchronization? (w24)

- **Agent:** `physics-theorist` · **Output:** `.claude/outputs/lattice-capacity-theory.md` · **Branch:** none expected (a note + small numerical sanity checks; **does not change production model code** — flags what the engineer should build)
- **Read first:** `.claude/outputs/dimension-aware-budget.md` (§3 — the write ceiling and its d-independence) · `.claude/outputs/controller-mvp.md` (the placement verb + the packing bound `N_pack = πR²/((√3/2)·d_safe²)` and the sizing rule `R = 0.808·√K`) · `.claude/negative_results.md` **N92** (tier A) · `.claude/tasks/clu-controller-spec.md`
- **Runs in parallel with `write-ceiling-break` — you two are attacking the same ceiling from theory and experiment. Do not assume its outcome; Item 4 is where you handle both branches.**

## Why — the Head's hypothesis, advisor-refined
w23 pinned `K_learned(d) = min(2^d, K_ceiling≈32)` with the ceiling belonging to the **write operator**: one global gradient dig cannot carve more than ~32 disjoint valleys jointly. **[HEAD hypothesis]** Shard items across `N` units and capacity multiplies (~`32·N`) — the ceiling is per-dig, not per-system. The Head's stated worry was needing to *"chain optimizers / map disconnected optimizer spaces."*

**[ADVISOR refinement — the claim to formalize]** That worry **dissolves under write-locality**: masked writes share **no parameters across units**, so there is nothing to synchronize. The only global object is **routing** — which item goes to which unit — and routing is the **controller's placement verb**, not an optimizer. If that holds, sharded capacity is additive at no optimization cost, and R2 ("capacity unclamped") has a second route that does not require breaking the ceiling at all.

## Item 1 — formalize the claim
State precisely, with conditions, when `K_total = Σᵢ K_ceiling(unitᵢ)` holds. Make explicit what "write-locality" must mean formally (disjoint parameter support per unit? disjoint *gradient* support? bounded cross-unit coupling in `V`?) and which of these the masked write actually satisfies — the measured bit-locality is **8474× (d=2) / 3434× (d=4)** corruption advantage, which is *very local but finite*. Does a finite locality ratio degrade additivity, and how?

## Item 2 — the failure modes (be adversarial with your own claim)
Enumerate what breaks it:
- **Read-side crosstalk** — a query settles in the wrong unit. Additive *write* capacity is worthless if retrieval cannot tell units apart. What condition on inter-unit separation (an `N_pack`-style bound across units) is required?
- **Routing error** — the controller must place *and* re-find. What is the cost of a routing miss, and is routing itself capacity-bounded (does the router hit its own ~32-style ceiling)?
- **Shared normalization / global confinement** — any term coupling all units (a global confine, a shared `s` rule, batch-level normalization) silently re-introduces the joint dig. Flag every such term currently in the code path.

## Item 3 — the read-cost question (this decides whether the win is real)
Is retrieval `O(1)` or `O(N)` in the number of units? A capacity win paid for by an `N`-fold read cost is not a win — it is a linear scan with extra steps. State the condition under which routing is `O(1)` (the controller knows where it wrote — the placement verb records the site) versus when a query must be tried against all units.

## Item 4 — interaction with `write-ceiling-break` (both branches)
- **If the ceiling breaks** (that task succeeds): does the lattice argument still buy anything, or is it superseded?
- **If the ceiling holds**: the lattice becomes the *primary* route to R2. State what the engineer must build in w25 and the smallest experiment that would falsify additivity.

## Acceptance
A rigorous note containing: the formal claim with its conditions · the failure modes · the read-cost verdict · **at least one falsifiable, concretely-sized prediction an engineer can test in w25** (e.g. "2 units × d=4 should reach K=64 strict-retrieval at parity with 1 unit × K=32, with inter-unit separation ≥ X"). Small numerical sanity checks (jax/sympy/numpy) to confirm or refute your own claims are expected — **if the numbers refute the hypothesis, say so; a clean refutation now is worth more than a hopeful formalization.**

## ⚠ Notes
- Do not modify production model code. Flag what the engineer should change.
- The designed store already reaches ≥256 items (censored) under identical numerics — any theory of the *learned* ceiling must be consistent with the designed write having no such ceiling.
- Do not quote the base √2 / `d^1.62` exponent (**CM-22(j)**, never-quote). The pinned law is `min(2^d, K_ceiling≈32)`.
