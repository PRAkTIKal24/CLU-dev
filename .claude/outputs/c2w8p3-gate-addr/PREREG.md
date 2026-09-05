# PREREG — c2w8p3-gate-addr (G-ADDR construction + Ruling-3 counterfactual)

**Filed 2026-08-09 by the `experiment-engineer` spoke, BEFORE any measured cell of this task runs.**
Base `main @ 1eda6a0`, branch `c2w8p3-gate-addr`, worktree `../CHLU-c2w8p3a`.
Governing: `PREREG-C2W8-PASS3.md` §2/§4/§7/§8 + `ERRATA-C2W8-PASS3.md` §1 + charter §A30.
⛔ This file registers **leg definitions, thresholds and predicted values**. It is not edited after
the first cell; corrections go to `ERRATA-C2W8-PASS3.md` (Hub's shared file) or to a dated block at
the bottom of this file marked `POST-HOC` and labelled as such.

---

## 0. Dial declaration (echoed)
- **Dial:** none — instrument construction + one compliance counterfactual. No claim cell, no
  performance number, no arm-race adjudication.
- **Laundering control:** A3 IS a launder margin. Every quotation states **matched-items** and the
  **1 253×** byte ratio (matched-bytes is NOT met).
- **Falsifies:** a G-ADDR that cannot fail its designed negatives, or cannot pass its designed
  positive, does not ship.

---

## 1. G-ADDR leg definitions (registered before measurement)

**Query set (the "cue set"), ground truth by construction.** For each live item `i` with codebook
site `z_i = (c_i | a_i)` (address | payload), draw `n_q` queries
`x_k = c_i + κ_q · spacing · ε_k`, `ε_k ~ N(0, I_d)` per-coordinate, where
`spacing = median nearest-neighbour distance of the live codebook centers`.

⭐ **Why κ·spacing and not the shipped absolute `σ_q = 0.15`:** §4's scale-invariance guard. An
absolute jitter makes `σ/spacing` movable by rescaling φ with zero information gain, so a leg built
on it **measures the scale**. `κ_q` is dimensionless and co-scales. **Registered `κ_q = 1.0`**,
which is the census's own operating point to within 7 % (`σ_q/spacing = 0.15/0.138…0.147 = 1.02–1.09`).

The queries are read through the shipped `CluSystem.read` (payload channels zeroed by the read path,
as in every census read). The **settled point** is `q*` (`res.state.q_star`, full store space) — the
same object `_relaxed_sites` and the capture-radius instrument use.

| leg | definition | threshold (registered) |
|---|---|---|
| **A1 — correct-basin rate** | fraction of cue queries with **(i)** `argmin_j ‖q* − z_j‖ == i` (the settled point resolves to the QUERIED item) **and (ii)** `‖q* − z_i‖ ≤ ρ_i`, `ρ_i` = the item's **measured** SC-6 capture radius (`ρ_i` NaN or 0 ⇒ never correct) | `A1 ≥ max(4/n_items, chance + 2·SE)`, `chance = 1/n_items`, `SE = sqrt(chance(1−chance)/N_q)` |
| **A2 — never-addressed fraction** | fraction of live items receiving **zero** correct (A1) cue reads | `A2 ≤ 0.5` |
| **A3a — cue launder margin** | `A1_voronoi(store) − A1_1nn(launder)` on the **same** queries, **same decision rule** (which item did you resolve to). Launder = 1-NN in φ over the live keys on the RAW query. `A1_voronoi` = leg (i) alone. | `A3a > 0` |
| **A3b — stream launder margin** | `mean over stream read events of (read_acc − knn_acc)` — the census's own held-out class-accuracy margin against the ring-buffer kNN-in-φ launder, matched items | `A3b > 0`; **declared NOT-APPLICABLE** (never a null) when no stream exists (planted rigs) |

`gate_addr_pass = A1_pass ∧ A2_pass ∧ A3a_pass ∧ (A3b_pass if applicable)`.

**Reported beside every quotation, never gating:** `A1_voronoi` (any-item, no basin test), the
"any-basin" rate (the pass-2-style leg this exists to replace), the banked telemetry
`frac_never_read`, and the byte ledger with the **matched-items / NOT matched-bytes (1 253×)** label.

⚠ **A2 is deliberately NOT the banked telemetry `n_never_read/n_items`.** Reason, registered here
before measurement: `attach_reads` credits a read only when `covered = True`, and `covered` is
computed on the **launch point** `q0` (`min_j‖q0 − c_j‖ ≤ ½·min-separation`), not on the settle. It
is therefore a property of the *query distribution vs the codebook*, almost independent of the store
— the prediction this yields is **P1** below. The banked number is still reported.

## 2. Designed controls (all pytest-asserted)

| # | control | construction | registered outcome |
|---|---|---|---|
| **C+** | **positive** (a gate that cannot pass is as vacuous as one that cannot fail) | planted store, `n=6` wells at well-separated keys (spacing ≫ atom width), depth 1.0, unused groups flattened | **G-ADDR PASSES**; `A1 ≥ 0.80` |
| **N1** | ⭐ **arm B's banked configuration FAILS** | (a) the verdict arithmetic on arm B's **banked measured legs**; (b) a **live re-score** of arm B's census, 3 seeds | **FAIL**, prior **0.97** (Hub Q1) |
| **N1′** | **arm-B-class blind spot, live and cheap** | planted store, wells at the keys but **narrow** (`s` ≪ query displacement) ⇒ retrievable at their own sites, unreachable from a cue | **FAIL** on A1/A2 |
| **N2** | **planted permutation** | identical store and identical queries, targets permuted to the wrong sites | `A1 ≤ 0.02`; **FAIL** |
| **S** | **scale-only** (§4) | identical geometry, every address quantity × `a` (`a ∈ {0.8, 1.25}` planted; `a = 0.8` on the real rig) | `|ΔA1| ≤ 0.05` (Hub **Q8**, prior 0.90) |

## 3. Numeric predictions (mine unless marked Hub)

| # | quantity | prediction |
|---|---|---|
| **P1** | the banked `n_unassigned` is a launch-coverage statistic ⇒ it is (near-)identical across arms at fixed φ/codebook, and the settle-based never-addressed fraction differs from it | qualitative + `A2 ≠ frac_never_read` on ≥ 1 arm |
| **P2** | **Hub Q2** — A1 on pass-2 arm A, re-scored, PCA d=8 | **Hub: 0.03–0.12.** ⚠ **My band differs: 0.10–0.45**, derived from arm A's measured self-probe `acq = 0.414–0.492` at σ_q ≈ 1.02–1.09 spacings (A1 = acq × P(inside its own measured basin); median capture radius 0.29–0.43). Both bands are registered; at most one survives. |
| **P3** | A1 on arm B, re-scored | **0.30–0.95** (its `acq` is 0.94–0.98) — i.e. **I predict arm B does NOT fail on A1**; it fails on A3b (banked −0.594) and possibly A2 |
| **P4** | **Hub Q1** — arm B fails G-ADDR overall | fails (prior 0.97) |
| **P5** | A3b on arm A, re-scored | **−0.354 ± 0.05** (banked reproduction; deterministic at fixed seed) |
| **P6** | **Hub Q7 / §7** — Ruling-3 counterfactual: the attractor CAN move off the stored key | **moves**; registered statistic `follow = ‖q*_relaxed − c_i‖ / ‖δ‖` with `δ = 1.0 × spacing`; **PASS iff `follow ≥ 0.5`**; my point prediction **`follow ≥ 0.80`**. ⛔ `follow < 0.5` ⇒ compliance ruling REVERSES ⇒ **STOP AND ESCALATE** |
| **P7** | **Hub Q8** — scale-only moves G-ADDR ≈ 0 | `|ΔA1| ≤ 0.05` |
| **P8** | housekeeping 1 (cross-kernel `own_foreign_site_depth`): the Gaussian-hardcoded estimator **over-reads** the foreign leg under `wendland` at arm A's config | over-read factor **≥ 2×** on the foreign leg; the fixed estimator is **bit-identical** under `kernel="gaussian"` |

## 4. Declared NOT-RUNs (never reported as nulls)
merge / prune / restoration / any §2.7 claim cell · the arm-A-vs-arm-B **race adjudication** (VOID,
§A30.1) · any tier-ii / full-CLU / I2 verdict · any performance claim · the φ_dim→addr_dim projection
(wt2's) · the geometry precondition (wt2's) · the spine (wt3's) · monitor #3's refusal-rate defect
(reported, not fixed).

*Filed before the first cell of this task. — experiment-engineer, 2026-08-09.*
