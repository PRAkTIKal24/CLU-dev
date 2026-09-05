# placement-landing — land PGCP, scrub the arrays, defeat the measured attack

**Agent:** experiment-engineer. **Worktree MANDATORY** (§3.2 — 4 engineer tasks this wave).
**Co-headline of w26** (addendum-2 §B3.1). Base local `main` (post-w25, post-duplicate-cleanup).

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** lifetimes + admission (R1's structural underpinning).
- **Laundering control:** the TTL-dict / canonical-dictionary claim-structure comparison. PGCP
  makes the store **exactly as deletable as `del d[k]`** — that IS the claim, and the report says
  so. The differentiators are named as what they are (continuous amplitude law + commutation
  proof, spacing certificate, packing economics), never smuggled into the deletion claim itself.
- **Falsifies:** post-eviction hole-detection `AUC(z_hole)` does **not** fall to ~0.5 under the
  real two-phase read; or the real read at exact-`d_safe` lattice spacing underperforms the
  gradient-flow relaxation the theorist verified (H5 scope gap) badly enough that the lattice
  constant must inflate and re-pay the packing win.
- **Does NOT falsify:** delete-time churn (2.84 moves/delete at full load is the *expected*
  price, pre-measured); LRU/staleness eviction remaining permanently outside the claim;
  per-offered admission below 1.000 at the un-inflated sizing multiplier.

## Sources (read first, all three)
`.claude/outputs/order-independent-placement.md` — **§6 is your build spec, follow it**; §2 the
theorems; §3 the numbers you must reproduce. `.claude/outputs/mia-decay-measurement.md` — §2 is
your acceptance test, §8 defect D1. `.claude/scratch/order-independent-placement/pgcp.py` — the
reference implementation to port.

## What to build (theorist §6, in order)

1. **`chlu/core/placement.py`** (new, ~150 lines, pure numpy, no JAX): port `pgcp.py` —
   `splitmix64`, `hash_point`, `prio`, `hex_cells`, probe orders, `CanonicalPlacer` with
   insert/delete fix-up cascades and a move log.
2. **`Controller`**: new flag `placement ∈ {"relocate"` (default, **unchanged behaviour**)`,
   "canonical"}`. Under canonical, `offer` consults the placer (admission = "got a cell"; the
   spacing gate is a lattice invariant — assert `min_spacing ≥ d_safe` in tests, do not re-check
   per write). Slot layout = canonical priority order after every op.
3. ⭐ **New verb `Controller.delete(item_id)`** implementing Theorem 2 (remove + fix-up cascade;
   apply moves as `evict` + `with_item` per moved record; update `records[*].center`). **This verb
   does not exist today — the deletion claim has no code path until it lands.**
4. ⭐ **The array-scrub fix (mia D1)**: `AtomStorePotential.evict` currently clears `active` and
   `amps` only — `centers[slot]` and `payloads[slot]` retain the written address and value
   verbatim (measured max err 5.6e−8 / 0.0 over 3072 evictions). Zero them too. **No physics
   number moves** (`V` multiplies both terms by `active`) — verify that claim with a test, do not
   assert it.
5. **Guard:** LRU (`evict_policy="staleness"`) must **hard-error** under `placement="canonical"`
   combined with any deletion-claim flag; `"depth"` is allowed (amp = base·e^{−λ·age} is
   item-intrinsic). Theorist §4(c).

## Tests (port + new)
T1–T5 from the theorist's §3 as pytest with `tobytes()` asserts (exhaustive n=4 orders; random
orders at n∈{8,16,40,64}; write/delete interleavings; mid-decay delete; write-then-delete);
packing regression `N_cells(R(K)·1.05) ≥ K` for K∈{16,32,64,128}; a cascade-count smoke; the
scrub test; the LRU guard raise. All green before you run anything.

## ⭐ The acceptance test (this is the task's criterion, not a nice-to-have)
Re-run **`mia-decay-measurement` §2's history column** against the new placement rule. The
harness already exists (`.claude/outputs/mia-decay-measurement/mia_harness.py`) and nothing else
about it needs to change — swap the controller's placement mode.

**Target: post-eviction `AUC(z_hole)` 0.99985 → ~0.5** (with `TPR@FPR 1%` → ~0.01), and
`AUC(n_live)` 0.811 → ~0.5. Report the paired-placement column too (it must stay exactly
0.5000 — it is the harness sanity check). ⛔ **Until this reads ~0.5, no deletion-flavoured
sentence about this store is defensible** — say so in the report whatever the number is.

## The rematch cell (theorist §6.4, pre-registerable now)
`controller-mvp`'s `on_sized` K=64 with canonical placement, **under the real two-phase Verlet
read** (the theorist's H5 used gradient-flow relaxation — that is the one open scope gap):
predicted admitted **61/64**, per-admitted **1.000**, per-offered **0.953**; with sizing
multiplier 1.05 (→73 cells) predicted per-offered **1.000**. ⚠ **Never quote 0.953 as a measured
retention number until this cell runs** — it is currently deterministic admission × gradient-flow
per-admitted. N91 discipline: per-offered and per-admitted always travel together.

## Wording rules (⛔ binding)
⛔ **"certified" is banned program-wide** (a defended (ε,δ)-DP term we satisfy on zero counts).
⛔ No "unlearning", no "privacy guarantee". Approved: *"exact store-level deletion (scoped)"*,
*"scheduled per-item retention"*. **The claim's novelty wording is GATED on `deletion-prior-art`
(web-scout, this wave)** — the discrete skeleton of this rule is history-independent hashing and
is *not* novel; the contribution is exactness in a continuous designed landscape with
decay/permanence coexisting plus the spacing certificate. Write the scoped paper sentence from
theorist §4 verbatim and flag it as citation-gated. **Store-level scope only** — φ and the
learned-landscape residue are separate channels.

## Deliverable
`.claude/outputs/placement-landing.md` + PREREG first (before any measurement) at
`.claude/outputs/placement-landing/PREREG.md`. Standard report format: flag-provenance table,
what you built, PREREG scorecard, the acceptance-test number, the rematch cell, declared
deviations, git footprint, downstream reconciliation list in the first 10 lines.
Full `pytest tests/` green + `ruff check` clean. Commit atomically to
`agent/experiment-engineer/placement-landing`. Do not push.
