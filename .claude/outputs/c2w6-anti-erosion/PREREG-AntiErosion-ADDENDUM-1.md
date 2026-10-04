# PREREG-AntiErosion — ADDENDUM 1 (operationalisation)

**Filed 2026-08-05 by `experiment-engineer` (c2w6-anti-erosion), BEFORE any science
cell of the 1000-step harness has run.** Governs the cells run from `ae47a66`/`a1` onward.
The parent `PREREG-AntiErosion.md` is unedited. This addendum only *operationalises* the
already-registered predictions (the engineer does not re-derive them) and registers the
measurement definitions the harness needs, plus **one genuinely new registered prediction**
(A1-4) whose provenance is stated honestly below.

Build state at filing: branch `c2w6-anti-erosion`, commits `ffe7440` (P1 + I1 mechanism),
`ae47a66` (harness + tests). **K1/K2/K5 are green as tests** (18 passed) — no science cell
has run. Only three throwaway smoke runs (26 and 50 outer steps, seed 0) have executed;
they are named where they informed a definition.

---

## A1-1 — The erosion curve's y-axis and the E1/E2 ratio

* **Depth** = `cell_group_depth` (the published convention) of a slot's own atom rows,
  evaluated at that slot's **recorded site with the payload block zeroed** (the launch
  manifold), median over the **live** slots. Traced form = `blocks.fitted_well_depth`,
  asserted equal to the numpy form at 1e-5 relative (`test_the_traced_depth_matches…`).
* **Lane / tokens:** lane 0 of a FIXED validation batch `x0`, identical at every reading and
  identical across arms at a seed. ⚠ The trainer's own `store_watch` series uses the first
  TRAIN batch instead; both series are reported and they are two independent token
  samplings of the same quantity, never averaged together.
* **The E1/E2 ratio** `r = depth_median(final reading) / depth_median(reading nearest step
  200)`. `depth_ratio_final_over_untrained` is reported beside it, always.
* **Collapse floor (new, mechanical):** an arm whose step-200 depth is `< 1e-6` is at the
  collapse floor and its ratio is `0/0`. Registered: that counts as E1 **met** (it is a
  stronger form of "the OFF arm decays"), labelled `collapsed_by_200`, never as noise.

## A1-2 — K3 and K4, operational

* **K3 harm** iff `Δbpc(ON−OFF) > 2·SE` **and** `Δ > 0.01` (E3's own registered
  equivalence band). The second clause exists because 3 paired seeds can give `SE ≈ 0`,
  under which any positive Δ would "exceed 2 SE" — a degenerate rule, not a stricter one.
* **K4 fires** iff, on the ON arm, `mean|bpc_live − bpc_blank| < 1e-6` **and**
  `mean(bpc_memory_deleted − bpc_live) ≤ 1e-6`. Two memory-deleted columns are reported:
  (i) an **eval-time swap** (the trained block with `NullMemoryCell` in place of the store)
  and (ii) the **retrained `none` arm** (the probe's convention). (ii) is flag-independent,
  so at a fixed seed it must be the same number in every cell — a free consistency check.

## A1-3 — I1's event set, and why the registered I1-a rate is expected to be 0/0

**Registered event (the guard's own definition, as built):** an *admitted* write whose
target slot (a) already holds an item — the retained codebook row is non-zero — and (b) is
**not** being evicted (`reset = 0`, because an eviction re-draw is a designed channel and a
guard that fought it would fight deletion). Reference depth = the state the **inner write
starts from** (post-decay, post-eviction, post-placement), so designed decay and designed
placement are excluded from the erosion accounting by construction.

**Registered prediction, contradicting the parent prereg's I1-a band, and filed before the
cells:** at the registered rig the event count is **exactly 0**, for an arithmetic reason,
not an empirical one — `n_chunks = seq_len/chunk = 16 ≤ capacity = 8` (toy) and `= 16 ≤ 32`
(pilot), and the allocator only ever reuses a live slot through an eviction (the plan sets
`reset = 1` on the recipient slot whenever an eviction fired). A direct count on the
untrained model at seeds 0/1/2 measured `admits 13/19/16, slot reuse 0/0/0` across all 4
lanes. So **I1-a's registered 10–40 % band is unmeasurable at this rig, and that is a
structural finding about the controller, not a null.** I1's guard therefore ships built,
tested and OFF; its home is capacity pressure (C2W8) and the persistent store (C2W10).

## A1-4 — The adjacent channel that IS measurable (the interference audit)

Registered **in place of** I1-a's unmeasurable rate, same intent (#9/#12): for every pair
`(A, c)` where item A was live before chunk `c` and chunk `c` was admitted into a
**different** slot, split the fitted depth at A's own site into

* **own leg** — A's own atom rows. The write is masked to B's rows, so the designed
  prediction is `D_after = D_before · group_scale²` **exactly**.
  ⭐ **Registered: 0 violations of that prediction, residual ≤ 1e-5 relative** (this is a
  C3-locality regression check; a non-zero count is a bug, not a result).
* **foreign leg** — every other row's contribution at A's site. This is the real
  interference channel: B's atoms are dug near A and the landscape A sits in changes
  **without touching one of A's parameters**.
  ⭐ **Registered: `rate_up_foreign ≥ 0.5`** (foreign depth at a live item's site rises on
  the majority of events) **and `median_rel_change_foreign > 0`, both on ≥ 2/3 seeds.**
  ⚠ **Provenance, stated because it is not blind:** a 26-step smoke run at seed 0 (run
  before this addendum, while validating the instrument) read `rate_up_foreign = 0.83`,
  `median_rel_change_foreign = +1.58`, own-leg violations `0`, own-leg residual `2.4e-07`.
  The prediction above is registered for the **1000-step, 3-seed cells, which have not
  run**, and is deliberately weaker than the smoke value.

## A1-5 — I2's proxies (all three reported, never pooled)

* **erosion rate** of a well = `−` the least-squares slope of `ln(depth)` vs outer step over
  the run's readings (positive = eroding); wells with < 3 positive-depth readings → `nan`.
* **usefulness proxy 1 — read selection:** the CLU read is a relaxation, not a lookup, so
  there is no selection event. Registered proxy: the **nearest live site in address space
  to the launch point**, evaluated against the store as it stood *before* that chunk's write
  (read-before-write), counted per slot over the stream.
* **usefulness proxy 2 — loss contribution:** **leave-one-well-out** probe bpc — the same
  trained block, the same stream, with every chunk that would be admitted into that slot
  refused instead (a *plan* edit at layer 0, not a model edit), minus the full store's bpc.
  Larger = the well's deletion costs more = more useful. Taken at 4 checkpoints per run.
* **the driver:** `‖∂L_outer/∂(this slot's atom rows)‖`, per well, per reading — exactly
  0.0 on the P1-ON arm by construction, which is itself the partition's in-flight receipt.
* **ρ** = Spearman over the live wells of one run, per seed, then mean ± SE across seeds.
  Registered direction unchanged from the parent prereg §3 (`ρ ≥ +0.5` confirms; `ρ ≤ −0.3`
  is the registered refutation branch; `|ρ| < 0.3` = no usage structure, caveat stays).
  Slot identity across readings is the proxy for well identity; the per-reading site
  addresses are stored so the analyst can check for allocation drift.

## A1-6 — Declared cuts (never reported as nulls)

* The w40 pair is **priced before it runs**; if it does not fit the budget, **seeds are cut
  before the cell** (2 seeds, then 1), and the cut is declared with the wall-clock that
  forced it.
* The residual-off pair runs at **w4 only** and is labelled DIAGNOSTIC in every table.
* The memory-deleted column is the eval-time swap **plus** the retrained `none` arm; no
  separately-retrained memory-deleted arm *per flag* is run (the flags cannot reach it).
