# PREREG — C2W8 PASS 3: "Can the memory ADDRESS, once the encoder is not the bottleneck?"

**Filed 2026-08-09 by the C2W8 Hub, BEFORE any pass-3 harness cell runs.**
Base: `main @ 1eda6a0` (= `clu-dev/main`, 1495/0 HEAD-stable, zero worktrees).
Binding: charter **ADDENDUM 10 (§A29–§A30)** · **§A21 C2W8 row** · intervention **§5/§8** · the
Head+Advisor **PASS-3 DIRECTIVE** (2026-08-09). Companions: `PREREG-C2W8.md`, `PREREG-C2W8-PASS2.md`,
`ERRATA-C2W8-PASS2.md` — **none is edited**. Pass-3 corrections go in `ERRATA-C2W8-PASS3.md`.

---

## 0. Where pass 2 left it (Hub-re-derived from raw; these are the quotable forms)

**Capture is FIXED. Addressability did not move a digit.**

| | pass 1 | pass 2 arm A |
|---|---|---|
| basins (`capture_radius > 0`) | 1/48 | **46/48** |
| self-probe decode (chance 0.0625) | exactly AT chance 3/3 | **+3.65…+4.75 SE** |
| site drift vs spacing ≈0.14 | 0.216–1.473 | 0.001–0.007 |
| own-beats-foreign | 3/48 | **44/48** |
| ⛔ **reads landing in NO basin** | **58 / 62 / 62 of 64** | **58 / 62 / 62 of 64 — DIGIT-IDENTICAL** |
| ⛔ **items never read** | 14 / 14 / 14 of 16 | **15 / 15 / 14 — marginally WORSE** |
| ⛔ admission refusal rate | 0.000 (monitor #3) | **0.000 — unchanged** |

**The wells are real; queries still do not reach them.** Root cause is **encoder geometry, not
physics**: σ_q = 0.15 vs measured key spacing 0.138–0.147, so a noisy query is as far from its own
key as its key is from its nearest neighbour; the repaired basins (2–3× spacing) necessarily overlap;
the kNN launder itself degrades 1.00 → 0.31 across the stream. **At the PCA d=8 operating point the
items are not separable by ANY read.**

**Two attribution findings that outrank the gate result and shape this prereg (§A29.4):**
(i) **width co-scaling is the lever, not compactness** — a plain Gaussian at the same co-scaled width
clears the gate 3/3 ⇒ **pass 3's store arm is "co-scaled width", with kernel form a declared
SECONDARY axis**; (ii) **the interference was manufactured by the gradient write** — a write that
*places* atoms gives foreign>own on **0/48** (was 45/48).

---

## 1. ⛔⛔ THE DEFECT CLASS THIS PASS EXISTS TO CLOSE

Three instances in one wave: pass 1's **vacuous `M`** (geometric leg 10× the resolution, refused
nothing) · monitor #3's **never-refusing admission gate** (refusal rate 0.000, still open) · pass 2's
**gate blind to addressability** (arm B passed all three legs 3/3 with `n_never_read = 16/16` and
launder margin −0.594, **and reported it against its own interest**).

> **A GATE THAT CANNOT FAIL ON THE THING THAT MATTERS.**

⇒ **G-ADDR is built and validated BEFORE anything is scored**, and every later spoke carries a
**mechanical file precondition**, never a prose gate.

⚠ **And its mirror image, which this prereg also guards against: a gate that can be trivially
PASSED.** See §4 (the scale-invariance guard) — the address space is normalised, so any leg
expressed in absolute units can be moved by rescaling φ rather than by improving anything.

---

## 2. G-ADDR — the addressability leg (spoke 1's deliverable)

Three sub-legs, **all two-sided, each with a designed negative proving it CAN fail,
pytest-asserted**:

- **A1 — correct-basin rate.** Reads landing in the basin of **the QUERIED item**, not merely in
  *some* basin. ⛔ The pass-2 instrument only ever asked "any basin", which is why 58/64 unassigned
  coexisted with a passing gate.
- **A2 — never-read fraction** (`n_never_read / n_items`).
- **A3 — margin vs the kNN-in-φ launder**, at the **declared matching**. ⛔ **Every quotation states
  matched-items vs matched-bytes explicitly**; pass 1's **1 253×** byte-ratio caveat travels.

**Mandatory designed negatives (minimum):**
1. **Arm B's banked configuration MUST FAIL G-ADDR** — it is the measured blind spot, banked and free
   to use (`n_never_read = 16/16`, margin −0.594). A G-ADDR that passes arm B is not a gate.
2. **Planted-permutation control** — queries deliberately mapped to wrong sites must score ≈0 on A1.

**Deliverable (the mechanical precondition for every later spoke):**
> **`.claude/outputs/c2w8p3-gate-addr/GATE-ADDR-VALIDATED.json`**

---

## 3. ⛔⛔ THE MAPPING DEFECT — φ_dim → addr_dim DOES NOT EXIST TODAY (Hub finding, pre-scoping)

`exp_well_lifecycle.PhiAddress` **forces `phi_dim = addr_dim`** (`cl.phi_dim = int(w.addr_dim)`) and
**TRUNCATES**: `out[:, :addr_dim] = f[:, :addr_dim] * scale`. There is **no projection**.

⇒ "strong φ at `addr_dim = 8`" would today be either **a weak 8-dim simclr** (refit at d=8, not the
priced encoder) or **8 of 256 coordinates** (truncation discarding 248 dims). **Neither is the strong
φ that measured 0.161 → 0.319.**

⇒ **A genuine φ_dim → addr_dim map must be BUILT and DECLARED** (projection or read-in head), with:
- its **parameters on the byte ledger of EVERY arm including the launder**;
- ⛔ **the launder uses the SAME projected φ** — never the 256-dim φ. A launder reading 256 dims
  while the store reads 8 is not a launder, it is a handicap match (fairness invariant §A4.3:
  identical φ for CLU, baselines and launder).
- **`(d, atom budget)` declared as ONE joint dial** (§A4.3).

**Feasibility, computed (`min_atoms = round(512·√2^d)`):**

| d | 8 | 12 | 16 | 20 | 24 | 256 |
|---|---|---|---|---|---|---|
| atoms | 8 192 | 32 768 | **131 072** | 524 288 | 2 097 152 | **1.7e41 — impossible** |

⇒ **naive d = 256 addressing is forbidden by the atom law**; the feasible band is **d ∈ {8, 12, 16}**.

---

## 4. ⛔⛔ THE SCALE-INVARIANCE GUARD (Hub finding — this prereg's own anti-trivial-pass leg)

The rig sets `scale = 1 / r95(‖φ(pool)‖)`, i.e. **addresses are normalised to unit radius**, while
**σ_q = 0.15 is an absolute constant**. Therefore:
- with `n` items in a unit d-ball, median-NN key spacing is **essentially geometric** — a property of
  `(n, d)` and of how uniformly φ fills the ball, **not** of how "good" φ is in any task sense;
- ⇒ **σ_q / spacing can be moved by rescaling φ alone, with ZERO information gain.**

**Registered guards, mandatory on every pass-3 arm:**
- **every geometric quantity is reported as a DIMENSIONLESS RATIO** (σ_q / spacing, `d_safe` /
  spacing, capture radius / spacing) **with the scale stated**;
- **a SCALE-ONLY control arm** (identical φ, scale multiplied by a declared constant) **must move
  G-ADDR by ≈0**. ⛔ If rescaling moves G-ADDR, the leg is measuring the scale and not the memory,
  and it does not ship.

⭐ **Consequence for what pass 3 can possibly show:** strong φ cannot enlarge the normalised volume.
It can only help by **spreading items more uniformly within it**. That is a *measurable* property and
§5 measures it **before** the spine spends any cell.

---

## 5. THE GEOMETRY PRECONDITION (spoke 2) — measured BEFORE the spine runs

At each candidate `d`, on the projected φ, per seed:
`median_nn` key spacing · **σ_q / spacing** · `d_safe` / spacing · the achieved atom budget.
Plus the **revived (d, atom-budget) rider**: *was the banked `d ≥ 16` inertness measured with the
co-scaling honoured?* — **ONE CELL, NOT A SWEEP.**
Plus **re-price `d_safe` so monitor #3 can go quiet HONESTLY rather than vacuously** (pass 1/2:
refusal rate 0.000 at `d_safe ≈ 0.12` vs spacing ≈ 0.14).

**Deliverable:** `.claude/outputs/c2w8p3-phi-geometry/PHI-GEOMETRY.json`, carrying a **registered
mechanical reading**:
> **GO** iff strong φ improves **σ_q / spacing** over the PCA reference at the same `d`, beyond noise,
> ≥ 3 seeds. **NO-GO** otherwise.

⛔ **A NO-GO does not cancel the spine** — it **re-labels** it: the spine then measures the physics at
a substrate *known in advance* not to have fixed separability, and its null is attributable rather
than confounded. ⛔ **Without this file, a pass-3 null would reproduce pass 2's null for a different
reason and be misattributed.**

---

## 6. THE SPINE (spoke 3) — the capture gate at strong φ

Frozen census + the **COMPLETED** gate (**G-CAP · G-DEC · G-DRIFT · G-ADDR**), co-scaled-width store,
strong-φ rig reusing `c2w8-cifar-strong-phi`'s built-and-priced encoders (**simclr primary**
`enc_steps = 8000`; **randconv** the cheap control arm), `task1_only` regime, pass-1 provenance
discipline unchanged, φ params ledgered on every arm including the launder.

**BOTH branches are registered as REPORTABLE, before the first cell:**
- **(a) DAYLIGHT** — measurable separation opens between the settle and its own same-keys launder
  once both can address ⇒ the **first candidate physics signal on this substrate**.
- **(b) NO DAYLIGHT** — the CIFAR spoke's **±0.0007** result reproduced on the census rig ⇒ **the
  tier-i thesis measured at the CL substrate. A REPORTABLE FINDING, not a failure to be tuned away.**

⛔ **Neither branch is a tier-ii verdict** (no organizer swap exists here). ⛔ **No paper number.**
This is still a component build.

### ⚠ THE D2a WARNING TRAVELS (§A29.6, standing)
**G-DRIFT → 0 means the settled point approaches a deterministic function of the stored key = D2a =
table-expressible** — the configuration intervention §8.2 prohibits, and exactly what the CIFAR arm
measured at strong φ (settle = same-keys kNN to **±0.0007**). ⛔ **No leg, objective or tuning choice
may treat drift → 0 as a target. G-DRIFT is reported as a TWO-SIDED diagnostic.**

---

## 7. RULING 3's ATTACHED CHECK — outcome, not identity (spoke 1, early)

`atom_site_local_init` is ruled COMPLIANT **conditional on** a mechanical counterfactual: **show the
attractor CAN move off the stored key when the write objective demands it** — e.g. a write-objective
term preferring a displaced minimum on a designed cell ⇒ the settled point follows it off the key by
a **measurable margin**. ⛔ **If the attractor provably cannot move, capture was bought by algebraic
pinning, the compliance ruling REVERSES (§A30.3), and the Hub escalates to the Head rather than
shipping.** Placed in spoke 1 **deliberately**: it is a validity precondition on arm A's entire
pass-2 result, and burying it in the spine would put it after the work it can invalidate.

---

## 8. NUMERIC PREDICTIONS (Hub's registered priors, before any cell)

| # | quantity | prediction | prior |
|---|---|---|---|
| **Q1** | G-ADDR fails arm B's banked config (designed negative 1) | fails | **0.97** |
| **Q2** | G-ADDR A1 on pass-2 arm A, re-scored (PCA d=8) | **0.03 – 0.12** correct-basin | — |
| **Q3** | strong φ improves σ_q/spacing over PCA at the same d (**the §5 GO reading**) | — | **0.55** |
| **Q4** | spine A1 (correct-basin) at strong φ, best arm | **0.15 – 0.45** | — |
| **Q5** | spine A3 margin vs its own same-keys launder **crosses 0** (branch (a) daylight) | — | **0.15** |
| **Q6** | branch (b) — no daylight, ±0.0007-class reproduction | — | **0.70** |
| **Q7** | Ruling-3 counterfactual: the attractor CAN move off the key | moves | **0.85** |
| **Q8** | scale-only control moves G-ADDR by ≈0 (§4 guard holds) | ≈0 | **0.90** |

⭐ **Q5/Q6 are the honest shape: I expect NO daylight, and that outcome is pre-registered as
reportable.** Registering it now is what stops it being tuned away later.

---

## 9. Declared NOT-RUNs (never reported as nulls)

⛔ **merge / prune / restoration verbs and the §2.7 claim cells** — still deferred (no population;
monitor #3 defect open) · **any third read iteration** (§A26.6) · **the tier-ii organizer swap** ·
**any full-CLU verdict** · **the arm A vs arm B race** — ⛔ **VOID as a comparison and STAYS
UNADJUDICATED** until both are re-scored on the completed gate (§A30.1); arm B additionally remains
**claim-barred** (private wells, its own K8/P6 registration — honoured).

---

*Filed by the C2W8 Hub, 2026-08-09, before any pass-3 cell. Corrections go in
`ERRATA-C2W8-PASS3.md`; this file is not edited.*
