# PREREG — controller-mvp (w23). Written BEFORE the harness ran.

Acceptance is a **measured retention-vs-K table** + an **admitted-fraction-vs-packing-bound**
check, so per protocol §5 these predictions and their derivations are committed first.

## Geometry (chosen so the gate CAN fire — N74's lesson)
Designed store = `AtomStorePotential`, atom width `s = 0.35`, so `d_safe = 4.4·s = 1.54`.
Proposals uniform in a disk of radius `R = 2.0`.
- **Packing bound (hex/farthest-point, N74's form):** `N_pack = π R² / (√3/2 · d_safe²)`
  `= π·4 / (0.866·1.54²) = 12.566 / 2.054 = 6.12`. N74 measured `6.0 ± 0.9` — this IS the bound.
- On the w20 **ring** geometry the gate was arithmetically vacuous (spacing 1.4142 ≥ d_safe 1.10).
  On this disk geometry a random pair in radius-2 disk has median separation ≈ 1.5 < d_safe,
  so the gate **fires** on a majority of offers at K ≥ 8. This is the whole point.

## Predictions
- **P1 (gate fires, admitted saturates at the bound).** On fixed R=2, controller-ON admitted
  count saturates at `6 ± 1` regardless of K (≥ 8), matching `N_pack = 6.1` within N74's ±0.9.
  Intervention rate (refuse+relocate)/offers > 50% at K ≥ 8.
- **P2 (per-admitted, controller ON, fixed geometry) ≈ 1.0.** The gate keeps live items ≥ d_safe
  apart ⇒ C5 selectivity 1.000 ⇒ per-admitted strict retention ≥ 0.95, flat in K.
- **P3 (per-offered, controller ON, fixed geometry) = admitted/K.** ≈ 1.0 at K ≤ 4, ≈ 0.38 at
  K=16 (6/16), ≈ 0.094 at K=64 (6/64). Declines as the packing bound / K — the abstention price.
- **P4 (controller OFF = designed_ungated).** per-offered = per-admitted, declining with
  collisions: ≈ 0.11 mean-retention at K=16 (reproduces w21 designed_ungated), lower at K=64.
- **P5 (per-OFFERED rematch verdict — the anticipated headline).** CLU+controller per-offered
  at K=64 (≈ 0.09, fixed geometry) **LOSES to the GRU (0.57)** and to every w21 primitive
  (mlp 0.43, attn 0.34, clu-learned 0.16). Abstention buys per-admitted purity at a capacity
  price the per-offered metric charges in full. **Registered as a loss, per the task's warning.**
- **P6 (per-ADMITTED rematch verdict).** CLU+controller per-admitted (≈ 1.0) is **BEST of five**
  at every K — it never stores an item it will corrupt.
- **P7 (sized geometry — the honest scoping message).** With R sized so `N_pack(R) ≥ K`
  (`R = d_safe·√(0.866 K/π) ≈ 0.808·√K`), controller-ON per-offered ≈ per-admitted ≈ 1.0 at all K
  ⇒ CLU+controller then **beats all four primitives on per-offered too**. Confirms the theorist's
  A4: item count never binds; local density does. The controller wins iff the address space is
  sized to the load.
- **P8 (decay/eviction — per-item retention machinery, w22).** With leaky wells (leak λ>0) and one
  item flagged permanent: after K subsequent offers the permanent item retains 1.0 while a leaky
  item's amplitude decays `A·exp(-λ·n_ticks)` and it self-evicts below `amp_floor`, retention → 0.
  Staleness eviction removes the least-recently-used non-permanent item first; a permanent item is
  never evicted (capacity alarm if the store is all-permanent and full).
- **P9 (admission cost per write).** The admission test is O(n_stored) distances + ≤ n_candidates
  relocation draws, NO relaxation and NO gradient step. Predicted wall-time per offer < 5% of one
  gradient-trained learned write, and ≪ a single AtomStore relaxation read.

## What the controller CANNOT fix (from clu-controller-spec §5, stated up front)
- It cannot create a τ=∞ permanent item on an unconstrained learned V (Prop C-N) — permanence is a
  *designed* flat coset, here supplied by `leak=0`, not learned.
- It cannot beat the packing bound: on a fixed address space it must refuse/evict, so per-offered is
  capped at `N_pack/K`. Densifying past the bound crosses regime-2 (selectivity collapse) — forbidden.
- It cannot rescue a **global-support** (learned-MLP) write: the spacing certificate is about the
  wrong thing there (N75). The controller-ON arm is therefore on the DESIGNED store only.
