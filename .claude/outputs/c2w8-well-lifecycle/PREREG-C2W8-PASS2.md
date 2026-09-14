# PREREG — C2W8 PASS 2: "Make the store capture"

**Filed 2026-08-06 by the C2W8 Hub, BEFORE any pass-2 harness cell runs.**
Base: `main @ 80d7d4b` (= `clu-dev/main`, 1443/0 HEAD-stable, zero worktrees).
Binding: the Head+Advisor **C2W8 PASS-2 SCOPING DIRECTIVE** (2026-08-06) · charter §A21 C2W8 row
as re-priced by it · **§A26.6** (the write side is the live route) · **§A28** · intervention §5/§8.
Companion: `PREREG-C2W8.md` (pass 1) and `ERRATA-C2W8.md`. **Neither is edited.** Pass-2 corrections
go in a dated `ERRATA-C2W8-PASS2.md`.

---

## 0. The object

**Make an item's own well the dominant feature at its own site, and make the launch land inside the
intended basin** — by controlling **how far each atom's influence reaches**, not by pinning where
wells sit.

⛔⛔ **BINDING HEAD RULING — DO NOT PIN ATTRACTORS TO DESIGNED ANCHORS.** Forcing the attractor to
equal `φ(item)` makes the settled point a deterministic function of item identity ⇒ **D2a ⇒
table-expressible**, violating intervention §8 prohibition 2, and it flattens the basin interaction,
superposition and manifold storage the programme exists for. **Placement stays LEARNED and
CONTINUOUS; basins stay free to interact.** Any pass-2 arm that pins, snaps, or regularizes the
attractor toward `φ(item)` is non-compliant and does not ship, whatever it scores.

---

## 1. What pass 1 established (re-derived by the Hub from `census.json`; these are the quotable forms)

| # | finding | measured |
|---|---|---|
| a | **The store has essentially no basins** | `capture_radius` **exactly 0.000 on 47 of 48 wells** (zeros 15/16 · 16/16 · 16/16; sole non-zero 0.5078) while **`λ_min > 0` everywhere** (0.791 – 8.873) ⇒ **positive curvature is necessary and NOT sufficient** (SC-6's lesson, reproduced) |
| b | **It re-reads its own items AT CHANCE** | self-probe `decode` **0.0625 = chance 0.0625**, 3/3 seeds (`n_probed = 128`) |
| c | **`P ≈ 0` is ABSENCE, not health** | "eroded, not an attractor" 15/16 · 16/16 · 16/16; live-attractor counts **1 / 0 / 0** |
| d | **K1 unlocked VACUOUSLY on `M` alone** | all **28 / 29 / 29** admitted pairs have `payload_dist` **exactly 0.0**; `R_cert` **1.5402 / 1.4177 / 1.6482** vs key spacing **0.1407 / 0.1375 / 0.1468** = **10.9× / 10.3× / 11.2×** ⇒ the geometric leg **refused nothing**; monitor **#3 `vacuous_gate` tripped 3/3 at refusal rate 0.000**. `M` measured *"a class-incremental stream contains two items of the same class"* — **true by construction** |
| e | ⭐ **THE DIAGNOSIS** | **foreign contribution EXCEEDS own on 45 of 48 wells** (own median **0.518 / 0.282 / 0.123** vs foreign **1.261 / 0.947 / 0.611**). An item's own well is a **MINORITY of the landscape at its own site** — which is why depth RISES (0.74 → 1.66) while retrieval FALLS, and why **site drift (0.003 – 1.473) reaches ~10× the key spacing (0.14)** |
| f | ⭐ **The sharpest form** | **C3 locality HOLDS in parameter space** (own-leg violation rate **0.000**, exact) **and FAILS in function space** (78–84 % of writes raise the foreign contribution) — **a write touches only its own atom block, but atoms have TAILS. Local in parameters is not local in the landscape.** |

⚠ **Two small deviations from the directive's prose, recorded so the quoted form is exact:** the Hub's
re-derivation gives own medians **0.518 / 0.282 / 0.123** (directive: 0.52 / 0.30 / 0.17) and foreign
**1.261 / 0.947 / 0.611** (directive: 1.29 / 0.96 / 0.62); and site drift's **low** end is **0.003**
(seed 0), not 0.22. Neither is decision-grade and no verdict moves — see §7 Q2.

**The design hint (the wave's own K2 result):** the trash region needed a **COMPACT** gate (exactly
zero beyond `r_k`) rather than a sigmoid, because **a sigmoid tail makes a "local" change global**.
Pass 2 applies that lesson **one level down, to the atoms themselves**.

---

## 2. The two arms (raced on the SAME census, which is frozen)

- **ARM A — compact / short-tailed atom influence.** Atom influence made compact or short-tailed,
  with widths **co-scaled to the measured key spacing** (`median_nn_task1` ≈ 0.14, per seed, measured
  not assumed). The functional form is the designed lever; theory picks the kernel.
- **ARM B — the emission head.** A standard MLP-class head on `φ` that **emits the well parameters
  directly** (center, width, depth, payload), with the well's functional form left designed exactly
  as now. A **forward pass instead of 300 gradient steps**. It removes the `min_atoms` co-scaling
  explosion, the write-budget/N94 caveats and the erodable-written-content channel, and it converts
  the accidental placement leak into the **DESIGNED write→φ organization gradient registered at
  §A28.1**.

⛔⛔ **ARM B'S DECLARED TRAP (stated here so it cannot be discovered late): an emission head that
produces ONE PRIVATE WELL PER ITEM restores explicit per-item store parameters and is LAUNDERED BY
CONSTRUCTION** — it is the intervention doc's own degenerate endpoint, reached faster. **The
configuration that can ever carry a tier-ii claim is per-item COEFFICIENTS OVER A SHARED WELL
VOCABULARY.** Pass 2 does **not** build the factored store; it **must not foreclose it** (§K8).

---

## 3. THE ACCEPTANCE GATE — pass 1's census, re-run unchanged

**Primary, mechanical, multi-seed (≥ 3 seeds). All three legs must hold:**

| leg | criterion | pass-1 value |
|---|---|---|
| **G-CAP** | `capture_radius > 0` on a **majority** of live wells | **1 / 48** |
| **G-DEC** | self-probe `decode` **above chance** (0.0625), beyond 2 SE | **exactly at chance, 3/3** |
| **G-DRIFT** | median `site_drift` **below the measured key spacing** (`median_nn_task1`, per seed) | **0.216 – 1.473 vs 0.14** |

⚠ **`own/foreign` is a DIAGNOSTIC, NOT a target, and this is binding.** Under private wells a high
foreign contribution is **interference**; in a factored store it is the **SIGNAL** (compositionality).
**Over-fitting to own-dominance now buys a reversal later.** Report own/foreign on every arm; **no
arm is tuned on it and no gate leg reads it.** ⭐ **The invariant that survives both designs is
RETRIEVABILITY**, which is what G-CAP/G-DEC/G-DRIFT measure.

⛔ **The census is FROZEN.** `chlu/core/well_lifecycle.py` is **read-only for both arms** — same
instrument, same arithmetic, or the race is not a race. A needed change routes to the Hub.

---

## 4. KILL-CONDITIONS (built before the thing they can kill)

**⭐ K7 — THE CAPTURE INSTRUMENT MUST BE PROVEN ABLE TO REPORT A POSITIVE. Built and green BEFORE
either arm runs; this is pass 2's most important kill-condition and it exists because of pass 1's
own failure.**
> Pass 1's gate legs were **forced false by construction** and nobody noticed until review. Before
> `capture_radius > 0` is trusted as a *success* criterion, the instrument must be shown to fire on a
> **planted** basin: construct a store with an analytically known capture radius and assert
> `capture_radius` recovers it within tolerance, **and** assert it returns 0 on a planted flat site.
> **Two-sided, pytest-asserted.** ⛔ Until K7 is green, a majority-positive G-CAP is not evidence —
> it is an untested instrument agreeing with us, which is exactly how pass 1 went wrong.
> ⚠ Pass 1 supplies exactly **one** non-zero reading (0.5078) across 48 wells — far too thin to
> license the instrument by observation.

**K6 — OFF is bit-identical.** Every pass-2 mechanism ships behind a flag whose OFF path is
**bit-identical AND parameter-count-identical** to `main @ 80d7d4b` (the K2 / P1 / psires precedent;
reddening the test un-ships the flag).

**K8 — the arm-B trap is machine-checkable, not a promise.** Every arm-B artifact declares
`wells_per_item` and `vocabulary_shared: bool` in its ledger, pytest-asserted present. ⛔ Any arm
with **private wells per item** is labelled **`NO_TIER_II_CLAIM`** in its own artifact and in
`census.json`, and the label travels with every number from it. The shared-vocabulary interface is
**specified** this wave even though the factored store is **not built**.

**K9 — the merge criterion must be able to REFUSE** (rider 2, and it gates any future merge verb
anywhere): re-registered at an operating point where `R_cert` is **commensurate with the key
spacing**, with a **designed negative proving refusal** — a pair that the criterion declines.
⛔ Shipping pass 1's criterion would ship *"collapse to one well per class"* wearing a certificate
costume.

**K5 (carried) — byte ledger on every arm including the launder**, with **(d, atom budget) declared
as ONE joint dial** and `γ_φ` holes counted. ⚠ Pass-1 baseline for scale: `clu_total_bytes` **360 960**
vs `knn_launder_bytes` **288** — a **1 253×** ratio. See §7 Q3.

---

## 5. NUMERIC PREDICTIONS (Hub's registered priors, before any cell)

| # | quantity | arm A (compact atoms) | arm B (emission head) |
|---|---|---|---|
| **P1** | G-CAP: fraction of wells with `capture_radius > 0` | **0.35 – 0.75**; P(clears majority) = **0.45** | **0.60 – 0.95**; P = **0.70** |
| **P2** | G-DEC: self-probe `decode` (chance 0.0625) | **0.08 – 0.20**; P(> chance, 2 SE) = **0.50** | **0.15 – 0.50**; P = **0.70** |
| **P3** | G-DRIFT: median `site_drift` vs key spacing ≈ 0.14 | **0.05 – 0.15**; P(< spacing) = **0.55** | **≈ 0**; P = **0.85** |
| **P4** | **all three legs, same arm, ≥ 3 seeds** | **0.35** | **0.60** |
| **P5** | own/foreign ratio (**diagnostic only**) | rises; foreign>own on < 24/48 | ambiguous by construction |
| **P6** | arm B lands in the **private-well** configuration (⇒ `NO_TIER_II_CLAIM`) | — | **0.85** |

⭐ **P6 is the honest expectation and it is why K8 exists:** arm B is *predicted to pass the gate and
to be unable to carry a tier-ii claim in the configuration that passes it.* Both halves get reported.

---

## 6. Declared NOT-RUNs (never reported as nulls)

- ⛔ **merge, prune, depth restoration and the §2.7 claim cells — NOT BUILT** until the capture gate
  passes. Pass 1's refusal to build verbs over empty populations was correct and stands.
- The **factored store / shared well vocabulary** — specified (K8), not built.
- **I2 correlation test** (C2W10) · **cross-stream criterion** (C2W10) · **wormholes / learned p₀
  traversal** (C2W9) · **any tier-ii, full-CLU or I2 verdict** (§A28.4).
- **CSF3** — untouched.

## 7. Questions carried to the Head/Advisor (numbered in the report; cells that depend on an answer
   do not run until it lands)

Q1 rider-1's likely discharge from banked evidence · Q2 the own/foreign aggregation · Q3 whether the
pass-2 gate is byte-blind at 1 253× · Q4 arm B's private-well configuration as a gate-passing but
claim-barred result.

---

*Filed by the C2W8 Hub, 2026-08-06, before any pass-2 cell. Corrections go in
`ERRATA-C2W8-PASS2.md`; this file is not edited.*
