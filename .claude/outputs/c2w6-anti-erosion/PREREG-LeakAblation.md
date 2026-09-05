# PREREG — CSF3 runs 2 vs 3 as the WRITE→φ LEAK-CLOSURE ABLATION

**Filed 2026-08-06 by the [C2W6] Hub, BEFORE CSF3 run 3 submits.** Binding basis: charter
**ADDENDUM 9 §A28.2** (Head-ratified 2026-08-06). ⛔ **This file is a MECHANICAL PRECONDITION ON
RUN 3: run 3 must not be submitted until this file exists** — it does, as of this filing. Never
edited after run 3 submits; corrections go in dated addenda below the line.

## 1. Why this pair is an ablation at zero extra compute

Run 3 = **run 2 + `erosion_partition=True`**, a one-token `MEM` diff (Add.7 §A23.2; the corrected
flag block is in `c2w6-anti-erosion.md`'s erratum banner 1 — `refresh_monotonic` stays **False**).
The partition was ruled in as an anti-erosion shield, but it **also closes the write→φ placement
channel as a side effect** (Add.9 §A27.3): H1b's localized placement
(`centers[:, :addr] = z[:addr] + jig`) is a differentiable path from φ's output into the store
state, sitting outside the sign-gated inner loop, and carrying **27 % of layer-0's φ gradient**
(0.0908 → 0.0659, charter §A22; reproduced at cell scale as 9.36e-04 → exactly 0.0 bitwise).
⇒ **The run-2/run-3 pair is the scale test of whether that channel was teaching φ anything.**
⚠ It is an *accidental* channel, not a designed one — Add.9 §A28.1 registers the **designed**
write→φ organization gradient as a separate lever for the write-side iteration. ⛔ A positive
result here is **not** a reason to keep the accidental channel (w20 doctrine: build the designed
version, do not keep an accident because it points the right way).

## 2. Registered measurements (all three, both arms, per seed)

1. **Δbpc attributed to the partition** — paired per seed, run 3 − run 2, on the held-out test
   split, with the mandatory dynamic-evaluation substitute column beside it as always.
2. **Depth-trough telemetry on BOTH arms** — the C2W6 shape, at scale: **trough depth**,
   **step-of-minimum**, **final/min ratio**, plus final/untrained; per seed, from the in-flight
   `[watch/clu_store]` series and the artifact's `store_health`. ⚠ Report **linear and geometric**
   aggregates side by side (C2W6's estimator lesson; a positive quantity spanning decades).
3. **`gradient_probe`'s traj/point ratio on BOTH arms** — this is the direct read-out of the
   closure. ⚠ Carry the standing qualifier: at `atom_place_radius > 0` the settled-point arm has a
   **non-zero denominator floor**, so a run-2 ratio **UNDERSTATES** the read's share; run 3, with
   the channel closed, is the arm where the ratio is clean. **Conservative bias, never an
   inflation.**

## 3. Registered expectation and the decision rule

- **Expectation (from toy scale, w4, 3 paired seeds): NO bpc harm** — the toy paired effect was
  **−0.004853 ± 0.000780 (6.23 SE, 3/3 seeds better)** at w4 and **+0.005676 ± 0.005613 (n.s.)** at
  w40. ⛔ The claim carried forward is **"no harm"**, never "P1 improves bpc" (Add.7 §A22).
- ⭐ **The registered signal: a run-3 bpc REGRESSION beyond 2 SE** ⇒ the write→φ channel **was
  carrying organization**, and it **re-prices Add.9 §A28.1's designed write→φ lever UPWARD** (the
  designed replacement becomes urgent, not optional).
- **No regression beyond 2 SE** ⇒ the accidental channel was not teaching φ anything the block
  needed at scale; the shield is free, and §A28.1 stays a scheduled lever rather than an urgent one.
- ⚠ **Neither outcome re-opens the abort criterion.** It stays **SUSPENDED** (§A23.1, *strengthened*
  by C2W6: the measured 112× trough that recovers is exactly the false positive the suspension was
  ruled over). Depth telemetry is logged as evidence; **no run is paused or escalated on it.**
- ⛔ **No run-1 number is ever the scale verdict** (§A20.4). Run 1 is the baseline leg of the
  *decoder* ablation; this is a separate, later pair.

## 4. Confounds declared in advance

- **Run 3 changes exactly one token** — any other diff between the two submissions invalidates the
  attribution and must be reported, not absorbed.
- **The partition does two things at once** (anti-erosion shield **and** leak closure). A bpc
  regression is therefore attributable to the *pair* of effects, not to the leak alone; separating
  them needs the designed-lever arm (§A28.1), which is not funded here. **Stated so it is not
  claimed later.**
- **Monitor #13/N94:** the toy expectation is a w4 reading and is formally non-promotable; it is an
  *expectation*, not a bar. The scale run sets its own numbers.
- **The trough is seed-dependent at toy scale** (2/3 seeds dipped; one never did) — with 3 seeds at
  scale, absence of a trough on some seed is **not** evidence of closure.
