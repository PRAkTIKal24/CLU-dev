# c2w11-payload-reach-repair — close the ONE measured arithmetic blocker, then re-run the kill set

**Campaign 2, wave 11. Agent:** experiment-engineer. **CONTINUES ON SPOKE A's BRANCH AND WORKTREE** —
branch `c2w11-substrate-and-kills` @ `5db2496`, worktree `../CHLU-c2w11a`. ⛔ **No new worktree**
(wt1 is already yours; C2W10 holds the other).
Appends to `.claude/outputs/c2w11-substrate-and-kills.md` as a **dated §12 addendum** (⛔ the body
above it is UNTOUCHED — the C-3 precedent) and **re-emits**
`.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json`.
**Budget:** ≈ 0.5–1 day. ⭐ **This is a bounded repair of one measured number, not a new experiment.**

---

## ⭐ WHY YOU EXIST — read this before anything else

Your own spoke-A run produced a **clean, arithmetic, attributable** blocker, and it is the reason
`kills_all_passed = false`:

| quantity | measured |
|---|---|
| distance the read must cross from launch to a well's **full** target (`= ‖v_j‖`, because the launch pins the payload block to 0) | **1.0000** |
| measured **SC-6 capture radius**, median over wells | **0.8535** |
| ⛔ **launch-to-target ÷ capture radius** | ⛔ **1.172 > 1** |
| full-space coverage of needed wells | ⛔ **0.0000 covered / 1.0000 uncovered** — every query, every channel, every seed |

⇒ **The needed well is outside its own basin BY ARITHMETIC.** ⭐ **The address half of the launch is
solved** (K0 = **0.9967** vs C2W5's designed offsets at **0.0378** — a 26× move; `mean_distinct_wells`
= 3.997 of `F = 4`) **and the payload half is not.** K5's `0.0007` — read, table and chance all equal —
is *"not expressible at all"*, **not** *"table-expressible"* (`PREREG-C2W11.md` §4's registered K5
note). **You are closing the reach gap so that the wave's K5 verdict means something.**

## ⛔⛔ THE DIRECTION IS FORCED — this is a derivation, not a knob to fish with

There are only two ways to make `‖v_j‖ / capture_radius < 1`:
- **(b) RAISE the capture radius above 1.0 — ⛔ STRUCTURALLY BLOCKED, and your own numbers prove it.**
  Measured capture **0.8535** already sits at the store-population spacing **0.8586** (ratio 0.994).
  A capture radius materially above the spacing means neighbouring basins merge, and the registered
  operating point forbids it (`d/s` never ≤ 2.01 — the **merger** floor; your reconciliation 2 already
  measured `d/s = 0.57` merging every well at the inherited width). ⇒ **capture is capped by spacing.**
- ⭐ **(a) LOWER the payload target radius below the capture radius — the ONLY admissible direction.**

> ### ⭐⭐ THE DERIVED FAMILY-CONSTRUCTION LAW (register it as such, it outlives this wave)
> **`‖v_j‖ < capture_radius ≲ min-well-spacing.`**
> A compositional family whose payload targets sit outside the basins that must capture them is
> **unreadable by construction**, for reasons that have nothing to do with organization. C2W5 already
> half-discovered this (its deviation **D3** cut `‖v‖` from `√m ≈ 2.83` to 1.0 for exactly this
> reason, *"outside basin reach"*) — **and never measured whether 1.0 was inside.** It was not.

## ⛔ THE REGISTERED TARGET — declared BEFORE the sweep, and it is an ARITHMETIC condition

> **Sweep `payload_radius` (with `atom_payload_init_radius` co-scaled, per C2W5 deviation D4) to the
> largest value satisfying `‖v_j‖ / measured_capture_radius ≤ 0.75`** (a 25 % margin, so the condition
> holds per-well and not merely at the median — report the **per-well distribution**, not just the
> median, since the median is what hid this).

⛔⛔ **THIS IS NOT "SWEEP UNTIL K5 PASSES", AND THE DISTINCTION IS THE WHOLE EPISTEMIC POINT.**
The target is the **ratio**, fixed above, in advance. **You select the operating point on the ratio and
then score K5 ONCE at it.** ⛔ You may not select on K5, on `OD`, or on any score. If K5 still fails at
a ratio ≤ 0.75 with the controls green, **that is the wave's finding and it is a stronger one than we
have now** — *the read is not expressible with reach controlled for*. **Report it as the result it is.**

## WHAT YOU RUN

1. **Measure the per-well capture-radius distribution** at the selected width (not just the median).
2. **Sweep `payload_radius`** to satisfy the registered ratio; declare the selected value and the
   ratio achieved, per seed.
3. ⛔ **RE-RUN, and they are BLOCKING — the fix must not buy K5 by breaking a control:**
   - **K2** both halves (⚠ C2W5 D3 registered the metric as **scale-invariant in the payload radius**
     — `tol = 0.25 × RMS‖y − ȳ_seen‖` co-scales. **VERIFY that, do not assume it.** If K2's payload
     half degrades, the family needs `m > 8` and you report that instead of quietly raising it.)
   - **K3** · **K4** (store-only form) · **K7-CAP** · **K6** (report it; it is `0.0007` at `d = 4`).
   - **K0** — cheap, and it must not move (it is address-space; if it moves, say why).
4. **K5** — scored **once**, at the selected operating point.
5. ⭐ **M6** — the diagnostic that matters most. Banked at the broken reach: launch occupancy
   precision **0.2308** → settle **0.0736**, dividend **−0.1567 ± 0.0052**, 3/3 seeds; distinct wells
   **3.998 → 3.807**. ⭐ **With reach closed, does the dividend's SIGN move?** ⛔ It is a **DIAGNOSTIC**
   and cannot fail a gate (§A33.1) — and it is the wave's most informative single reading either way.
6. **The coverage trigger, re-measured.** ⚠ `TRAVERSAL-FAILURE-SIGNATURE.md` §1 is **already written
   and it STAYS** — a fired trigger is not un-fired by a later repair. **Append a dated §1b** recording
   the post-repair coverage numbers and stating plainly whether the coverage mode **persists or
   closes**. ⛔ Do not edit §1.
7. **Re-emit `FROZEN-INTERFACES-C2W11.json`** with the new operating point, a recomputed mechanical
   `kills_all_passed`, and **a new field `payload_reach_ratio`** (per seed + the per-well
   distribution). ⚠ **Keep `v3_budget_grid`, `k8_structural_split` and the reader class UNCHANGED**
   unless the sweep forces a change — if it does, say so loudly, because **spokes B and C are gated on
   this file and coordinate through nothing else.**

## ⛔ SCOPE — what you do NOT do
⛔ No ψ, no novelty head, no organization loss, no null arm, no `OD`, no VALUE leg, no swap.
⛔ **Do not re-open the width selection** (settled: your reconciliation 2) or the `a` choice (settled:
K1 passes at 4/12/32). ⛔ Do not touch the two-sided drift leg — **your reconciliation 1 is escalated
to the Head/Advisor and is explicitly NOT yours to resolve.**

## FILE OWNERSHIP — unchanged from spoke A
**You own:** `chlu/core/factored_store.py` · `chlu/core/feature_launch.py` ·
`chlu/experiments/exp_c2w11_substrate.py` · `tests/test_c2w11_substrate.py` ·
`tests/test_factored_store.py` · `chlu/cli/experiment_cmd.py`.
⛔ **`chlu/core/monitors.py` — IMPORT READ-ONLY** (C2W10's `c2w10-lifecycle-mechanics` is appending
`protected_saturation` there concurrently; any new monitor goes inside your own module).
⛔ **DO NOT TOUCH:** `chlu/config.py` · C2W8-close territory (`well_lifecycle.py`, `clu_system.py`,
`soft_certificate.py`, `tests/test_{well_lifecycle,gate_addr,cifar_strong_phi}.py`) · CSF3 / live-pilot
territory (`scripts/csf3/`, `train_cluformer.py`, `blocks.py`, `exp_cluformer_pilot.py`) · C2W10
territory (`store_lifecycle.py`, `exp_persistent_store.py`, `stream_sources.py`, `controller.py`,
`usage_telemetry.py`).

## Acceptance (mechanical)
1. A `PREREG.md` **addendum filed BEFORE the sweep**, restating the registered ratio target and your
   numeric predictions for K5 and M6 at the repaired operating point.
2. The **per-well** capture-radius distribution reported (not the median alone).
3. The selected `payload_radius`, the achieved ratio per seed, and **K5 scored ONCE at it.**
4. **K0, K2, K3, K4, K6, K7-CAP all re-run and green**, with K2's scale-invariance **verified, not
   assumed**. ⛔ A K5 pass bought by a degraded control is not a pass — say so if it happens.
5. **M6 re-reported with its sign and 2 SE**, beside the banked broken-reach values.
6. `FROZEN-INTERFACES-C2W11.json` re-emitted with `payload_reach_ratio` and a recomputed
   `kills_all_passed`; `TRAVERSAL-FAILURE-SIGNATURE.md` §1b appended, §1 untouched.
7. Full suite green on the branch, count arithmetic stated with the checkout named (base:
   **1 607 passed / 0 failed at `5db2496` in `../CHLU-c2w11a`**).
8. ⚠ **The x64 hazard is now a KNOWN defect class in this repo and it has bitten twice** (your
   `place_write` scan-carry dtype; `orgdiv-null-arms`' N5 fast weights). **Run the FULL suite, never
   the file alone** — a per-file run will not catch it.
9. Reconciliation list in the first 10 lines; registered deviations argued; NOT-RUNs declared as
   NOT-RUNs, never nulls.

⛔ Never push `origin`; the Hub integrates. `clu-dev` only, and only the Hub pushes it.
