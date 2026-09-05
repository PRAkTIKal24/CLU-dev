# PREREG — write-ceiling-break (w24)

**Written before the science harness was run.** Declared deviation: a **timing
probe** (`.claude_probe.py`, 1 seed, no blank control, cell `d=6 K=64` for arms
`baseline_global / sequential_masked / crowding_aware / combo`) was **launched**
before this file was written, to size the compute budget. Its output was **not
read** until after this file was saved (it also returns strict rates, so it is
treated as a measurement of the predictions below, not as an input to them).

Acceptance criterion being predicted: `K_learned(d)` at `d ∈ {4,5,6,8}` under
baseline / masked-sequential / scale-invariant / crowding-aware / combo, and the
verdict UNCLAMPED vs CEILING-RAISED vs CEILING-SURVIVES.

Baseline to beat (w23 `dimension-aware-budget`, budget-adequate):
`K_learned = {d4: 16, d5: 32, d6: 32, d8: 32}`; d=6 K=64 strict **0.855** at 4096
atoms and **0.809** at 8192; d=8 K=64 strict **0.883 / 0.894**.

---

## Derivation of the predictions

**(P0) Baseline reproduction.** With `n_query_per_item` 32→16 (declared
read-cost reduction; strict SE ≈ 1% at K≥32) the baseline arm should reproduce
w23 within noise: **d=6 K=64 strict 0.82–0.89**, d=6 K=32 PASS. If the baseline
does not reproduce, no arm number is interpretable.

**(P1) Locality lever — masked/sequential: PREDICTED NOT TO BREAK THE CEILING.**
Masking makes the write bit-local in *parameter* space (w23: local-vs-global
corruption advantage 8474×/3434× at K=4). But the ceiling failure is not a
*corruption* failure — it is a *fidelity* failure at a cell where the write loss
already reaches ~0. In function space the superposed wells of neighbouring items
still overlap (at d=6 K=64 the site separation is 0.795 against an atom width of
0.3), and masking does nothing about that; it only removes cross-item gradient
traffic. Against that, masking *costs* fidelity: w22 measured atoms global 1.000
vs local 0.859 at K=4, because a masked item may use only `n_atoms/K` atoms
(4096/64 = 64 at the frontier cell — adequate, but a hard partition rather than a
free allocation).
→ **Predicted: `K_learned` unchanged at 32 for d ∈ {5,6,8} and 16 at d=4.**
Predicted d=6 K=64 strict for `sequential_masked`: **0.78–0.90**, i.e. ≤ baseline
± 0.05, possibly *worse*. `sequential_free` (one item at a time, no mask) between
the two: **0.80–0.90**.
**Falsifier:** any sequential arm reaching mean strict ≥ 0.9 over ≥3 seeds at
K=64 at any d ∈ {5,6,8}.

**(P2) ⭐ Scale-invariance ablation: PREDICTED NULL (this is the point).** Two
sub-levers, and both are predicted ~null, for two *different* reasons that the
measurement separates:
- *Loss-scale (item_agg = "sum").* Adam is invariant to a global rescale of the
  loss; making the per-item gradient K-independent therefore changes nothing
  except the relative weight of decoupled weight decay. **Predicted |Δstrict| <
  0.02.** If signal dilution in the aggregation were the mechanism, Adam would
  already have neutralised it — this arm measures that instead of arguing it.
- *Length-scale (σ_addr, atom width ∝ site separation).* At fixed d the
  separation shrinks only as `K^{-1/d}`: at d=6 a K-doubling changes every length
  by `2^{-1/6} = 0.89`, i.e. **11%**. A ceiling that is a sharp cliff between
  K=32 (PASS) and K=64 (0.855) cannot be produced by an 11% drift in a length
  scale. **Predicted |Δstrict| < 0.03 at d=6 K=64**; the largest arm effect
  should appear at d=4 (K-doubling changes lengths by 16%), and it should be a
  small *positive* shift at most.
→ **Predicted: `K_learned` unchanged under scale invariance at every d.**
**Falsifier:** `scale_invariant` clears 0.9 at 3 seeds at any cell the baseline
fails. If that happens the diagnosis flips from *optimization interference* to
*signal dilution* and the fix is cheap.

**(P3) Crowding-aware objective: the most likely mover, still predicted to fall
short.** The w23 tell (`write_loss → 0` while retrieval fails) is a defect of the
objective, and the three sub-terms here are exactly its blind spots: the
minimum-violation is a **mean over perturbation directions** (the few directions
pointing at a crowded neighbour are outvoted), the barrier is a **mean over all
K(K−1)/2 pairs** while only O(K) pairs are ever violated (a `1/K` dilution —
measured factor 3.5× at K=8 in `tests/test_write_ceiling.py`), and nothing at all
looks at where the *atoms* sit. Making the objective see all three should raise
the failing cell — but the retrieval failure at K=64 must ultimately be a
*geometric* overlap of basins that no re-weighting of a static write can remove.
→ **Predicted: d=6 K=64 strict 0.87–0.93 (Δ = +0.02…+0.08 over baseline), which
does NOT reliably clear 0.9 at 3 seeds. `K_learned` stays 32 at d ∈ {5,6,8}.**
**Falsifier:** ≥0.9 at 3 seeds at K=64.

**(P4) Combo.** Best of the three, predicted **0.88–0.94** at d=6 K=64; may pass
at an isolated (d, seed) but not at 3 seeds across ≥2 dimensions.

**(P5) Headline verdict.** **CEILING-SURVIVES** (p ≈ 0.6) > CEILING-RAISED to 64
at one or two dimensions (p ≈ 0.35) > UNCLAMPED, capacity tracking `2^d` (p ≈
0.05). Registered numeric form of the surviving law: `K_learned(d) = min(2^d,
K'_ceiling)` with **K'_ceiling = 32** (no change) under all five arms.

**(P6) Mechanism read-out (registered because it decides the follow-up).** At
every failing cell I will report `basin_success_rate` (addressing) alongside
`strict` (value). Prediction: at K=64 the failure is **addressing**, i.e.
`basin ≈ strict` and both ≈0.85, not a payload-channel failure (`basin ≈ 1.0`
with a large payload error). If instead `basin ≈ 1.0`, the ceiling is a READ-side
value-recovery limit and every write-side lever in this task is aimed at the
wrong stage — a finding that would redirect the whole thread.

## Falsification bar for the task's headline sentence
"The ceiling survives all three levers" is **falsified** if ANY arm attains mean
strict ≥ 0.9 over ≥3 seeds at K=64 at ANY d ∈ {5,6,8} (or K=32 at d=4), with the
value-blank control passing. A FAIL only counts as a ceiling if its budget
adequacy is verified (2× atoms AND 2× write steps, per N92).

## Scope guards
- A ceiling broken only by making the write more **designed** (formula-placed
  centers, hand-set widths, per-item site supply beyond what w22/w23 already
  supplied) is a **scope collapse (N46), not a win** — and would be reported in
  those words. No arm here supplies placement; all five differ only in the write
  operator.
- Do not quote base √2 / `d^1.62` (**CM-22(j)**, never-quote).
