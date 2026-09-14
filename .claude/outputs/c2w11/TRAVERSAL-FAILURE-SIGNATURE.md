# TRAVERSAL-FAILURE-SIGNATURE — C2W11

**This file exists because a reach monitor FIRED.** It is C2W9's spawn trigger and it is a file, not
prose in a report (`PREREG-C2W11.md` §7).

> The reach monitor is **split**. **COVERAGE failure** — a needed feature well lies **outside the
> union of the `k` launch diamonds** (*a launch-head problem*) — is **spoke A's**. **TRAVERSAL
> failure** — in-flight evidence points outside the current particle's diamond (*the wormhole
> trigger*) — is **spoke B's**, which appends its own dated section below if it fires.

---

## §1 — COVERAGE failure. FIRED. (spoke A, `c2w11-substrate-and-kills`, 2026-08-10)

**Mode:** ⛔ **COVERAGE** (launch-head). Traversal is not scored here and is **NOT** claimed.

**Threshold, registered BEFORE the run** in `.claude/outputs/c2w11-substrate-and-kills/PREREG.md` §2:
the trigger fires iff the mean fraction of *needed wells not covered by the union of the `k` launch
diamonds* exceeds **0.20**. Registered point prediction **0.45**, band **[0.20, 0.70]**.

### Measured signature

| statistic | seed 0 | seed 1 | seed 2 | mean ± 2 SE |
|---|---|---|---|---|
| **fraction of needed wells UNCOVERED** (registered, address-space diamonds) | 0.7461 | 0.7520 | 0.7656 | **0.7546 ± 0.0116** |
| fraction of queries with **ZERO** needed wells covered | 0.2559 | 0.2676 | 0.3145 | 0.2793 ± 0.0361 |
| fraction of queries **fully** covered | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| median distance from the nearest launch point to a needed well (address) | 1.250 | 1.265 | 1.256 | 1.257 |
| ⭐ **FULL-SPACE variant** (the well's full target `(u_j, v_j)`, not just its address) | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Per-feature-channel breakdown** (⛔ slot positions in `A(x)`, carrying **no semantic content** —
wells are never named semantically): uncovered on **0.734 / 0.744 / 0.750 / 0.756** of queries for
slots 1–4. **Flat across slots: the failure is not concentrated in one channel.**

### Reach radii and the geometry that produced it

| quantity | value |
|---|---|
| launch-diamond radius `reach = 2.0 × s_measured` (registered `reach_radius_frac_s = 2.0`) | **0.640** |
| measured `s` (α‖q‖²-subtracted radial fit, `R² = 1.0000`) | **0.3199** |
| store-population spacing (median NN over the `N_a = 32` anchors) | **0.8586** |
| `reach / spacing` | **0.741** |
| measured SC-6 capture radius, median over wells | **0.8535** |
| ⛔ **distance the read must cross from its launch to a well's FULL target** (`= ‖v_j‖`, because the launch pins the payload block to 0) | **1.0000** |
| ⛔ **launch-to-target / capture radius** | **1.172 > 1** ⇒ `needed_well_inside_basin = FALSE` |

### ⭐⭐ The signature, in one paragraph

**The address half of the launch is solved and the payload half is not.** The feature-factored launch
head reaches `≥ F` distinct wells on **0.994–0.998** of unseen queries (C2W5's designed offsets:
0.039), so `k` particles *do* arrive in `F` different wells' address neighbourhoods. But the read
launches with the **payload block pinned to 0** (the anti-decoration guard — nothing may hand the read
the answer), while every well's full target sits at payload radius `‖v_j‖ = 1.0`. The **measured**
capture radius is **0.8535**. So the needed well is outside its own basin **by arithmetic**, on
**every query, every channel, every seed** — the full-space coverage statistic is **1.0000 uncovered**,
not approximately.

⇒ **The binding constraint is REACH, not capacity and not addressability.** This reproduces the
`d ≥ 16` inertness finding's shape at a fully honoured atom budget (`K1` passes here at `a = 4`, `12`
**and** `32`), and it names the missing distance exactly: **the payload axis**.

### What this does and does not license

- ✅ It licenses **C2W9's spawn**: learned `p₀` / wormholes are reach fixes, and reach is now the
  measured binding constraint with a number on it (1.172× the capture radius).
- ⛔ It does **not** license widening the atom width to cover the gap: the width was **re-selected**
  against the store population under a registered protocol, and every `w_frac` large enough to matter
  drives measured `d/s` below the **2.01 merger floor** (`w_frac = 0.75 → d/s = 1.25`;
  `w_frac = 1.5 → d/s = 0.57`, K1 FAILS).
- ⛔ It does **not** license launching from a non-zero payload. That is the anti-decoration guard and
  removing it hands the read the answer.
- ⛔ It is **not** a traversal claim. No trajectory statistic is measured here.

### Provenance

Commit `352ac46` (+ the freeze re-emit), branch `c2w11-substrate-and-kills`, base `main @ 2e1cdb2`,
worktree `../CHLU-c2w11a`, main venv. Seeds **0/1/2**, 512 unseen rule-4-valid queries per seed.
Config: `write_mode="placing"`, `launch_mode="feature_factored"`, `k = F = 4`,
`atom_width_frac_spacing = 0.37` (**selected**, seeds 100/101/102), `a = 12`, `m = 8`, `d = 4`,
`N_a = 32`, `K = 128`, `σ_q = 0.15`, `α = 0.05`, read 400 + 800 Verlet steps at `dt = 0.05`,
`γ_address = 0.05`, `γ_read = 0.02`. Artifacts:
`.claude/outputs/c2w11-substrate-and-kills/run1/stage_coverage.json` and `stage_k1.json`.

---

## §1b — POST-REPAIR RE-MEASUREMENT. ⛔ **The coverage mode PERSISTS.** (spoke A, `c2w11-payload-reach-repair`, 2026-08-11)

⛔ **§1 above is UNTOUCHED and STAYS.** A fired trigger is **not un-fired by a later repair**; this
section records what the repair moved and what it did not.

**What was repaired.** The payload radius was swept to the ratio target registered *before* the sweep
(`PREREG.md` ADDENDUM §A1: largest grid `payload_radius` with `‖v_j‖ / measured SC-6 capture ≤ 0.75`
on every selection seed 100/101/102). **Selected `payload_radius = 0.60`** (`atom_payload_init_radius`
co-scaled, D4), achieved ratio **0.677** (selection seeds) / **0.692 · 0.692 · 0.681** (claim seeds
0/1/2). ⛔ Selected on the **ratio**, never on a score.

### The two coverage statistics, before and after

| statistic (3 seeds × 512 unseen) | §1, `‖v‖ = 1.0` | **§1b, `‖v‖ = 0.6`** | verdict |
|---|---|---|---|
| **registered, address-space** mean fraction of needed wells UNCOVERED | 0.7546 ± 0.0116 | **0.7546 ± 0.0116** (0.7461 / 0.7520 / 0.7656) | ⛔ **IDENTICAL — the trigger still FIRES** (threshold 0.20) |
| fraction of queries **fully** covered | 0.0000 | **0.0000** | unchanged |
| fraction of queries with **zero** coverage | 0.2793 | **0.2793** | unchanged |
| ⭐ **FULL-SPACE variant** (the well's full target `(u_j, v_j)`) | **1.0000** | **0.9299 ± 0.0038** (0.9331 / 0.9268 / 0.9297) | moved, still ≫ 0.20 |
| median distance to the needed **full** target | 1.6062 | **1.3929** | −0.213 (= the payload block's contribution) |
| ⛔ `‖v_j‖ / measured capture radius` | **1.172** (outside its own basin) | **0.692** (inside) | ⭐ **CLOSED** |
| SC-6 capture radius, median over wells (claim seeds, `a = 12`) | 0.8535 | **0.8672 / 0.8672 / 0.8809** | ~unchanged, as predicted |
| **payload-direction** reach (the one direction the read actually crosses) | **0.855 < 1.0** | **0.932 > 0.6** | ⭐ closed on the favourable direction too |
| fraction of wells with `‖v_j‖ ≤ capture_j` (**per well**, claim seeds) | 0.000 | **1.000 / 0.938 / 0.938** | 45/48 wells inside |

### ⭐⭐ The finding, stated plainly

> **Reach is closed and the coverage mode PERSISTS.** The two failures were separable and are now
> separated: the **payload-axis reach** gap (`1.172 → 0.692`, and the full-space uncovered fraction
> `1.0000 → 0.9299`) was arithmetic and is repaired; the **address-space coverage** failure is
> **bit-identical** at **0.7546 uncovered**, because it never depended on a payload — the launch head
> simply does not select the needed well for ~3 of every 4 needed wells. ⛔ The remaining mode is a
> **launch-head precision** problem (`occupancy precision 0.2303`, `correct-and-distinct 0.92 of 4`),
> not a reach problem, and no payload radius can move it.

**What the repair DID move** (the settle is no longer destructive): the M6 occupancy dividend
**−0.1567 ± 0.0052 → −0.0015 ± 0.0026** (settle precision `0.0736 → 0.2288`, distinct wells
`3.807 → 3.990`) — with reach closed, the settle now **preserves** the launch geometry instead of
losing 68 % of it. ⛔ M6 is a DIAGNOSTIC and fails no gate (§A33.1).

**What it did NOT move:** ⛔ **K5 still FAILS and still VACUOUSLY** (best margin 0.0000 / 0.00195 /
0.0000; read, table and chance all ≈ 0.001), `kills_all_passed = false`. **This is the wave's result
with reach controlled for**, and it was registered as the likely outcome before the sweep
(`PREREG.md` ADDENDUM §A2: K5 margin 0.000, P(pass) = 0.10).

### Provenance

Commit `4324002` (branch `c2w11-substrate-and-kills`, base `main @ 2e1cdb2`, worktree
`../CHLU-c2w11a`, main venv, `jax 0.9.0`, float32). Seeds **0/1/2** (claim) and **100/101/102**
(payload-radius selection, disjoint). Config identical to §1 **except**
`payload_radius = atom_payload_init_radius = 0.60`. Artifacts:
`.claude/outputs/c2w11-substrate-and-kills/run2/{stage_payload_reach,stage_coverage,stage_k1,stage_m6,stage_k3_k4_k5}.json`.

---

## §2 — TRAVERSAL failure (spoke B) — **NOT MEASURED BY SPOKE A**

⛔ Spoke A does not own trajectories and files no traversal statistic. If traversal fires, spoke B
appends a dated §2 here. **The absence of a §2 is a declared NOT-RUN for traversal, not a measurement
that traversal is fine.**

---

## §2 — TRAVERSAL failure (spoke B, `c2w11-physics-organizer`) — ⛔ **FIRED**, 2026-08-11

**Registered before the run** (`.claude/outputs/c2w11-physics-organizer/PREREG.md` §P6, filed before
the first cell): at the end of the address phase (step 400) a particle's **causal diamond** is the
ball of radius `reach = 2.0 × s_measured = 0.6360` about its current position (spoke A's registered
coverage reach, deliberately the same instrument). For each query, `frac_unreachable` = the fraction
of needed wells `j ∈ A(x)` that are (i) occupied by **no** particle and (ii) outside **every**
particle's diamond. **The trigger fires iff `mean_x frac_unreachable > 0.20`** — deliberately spoke
A's coverage threshold.

### The measured signature

| statistic | value (5 seeds, mean ± 2 SE) |
|---|---|
| **mode** | **TRAVERSAL** (in-flight). ~~The COVERAGE half is spoke A's and did **not** fire~~ ⛔ **[STRUCK — HUB CORRECTION, 2026-08-12, charter Add.15 §A43.4]** see §2a below |
| `mean_frac_needed_wells_unreachable` | **0.7715 ± 0.0081** (per seed 0.7725 / 0.7827 / 0.7773 / 0.7627 / 0.7622) |
| threshold | 0.20 ⇒ **fired by 3.86×** |
| fraction of queries with ≥ 1 unreachable needed well | **0.9996** (1.0 / 0.998 / 1.0 / 1.0 / 1.0) |
| mean fraction of needed wells actually **visited** | **0.2254** (reproduces spoke A's occupancy precision 0.2303) |
| median distance to an unvisited needed well | **1.7483** = **2.75 × reach**, and **2.03 × the well spacing** (0.8586) |
| per-slot breakdown (position in `A(x)`, ⛔ never a semantic index) | 0.8066 / 0.7852 / 0.7559 / 0.7422 — **flat across slots**: this is not one bad slot, it is every slot |
| seeds | 0, 1, 2, 3, 4 · arm `phys` · 512 unseen queries/seed |

### ⛔⛔ §2a — HUB CORRECTION (2026-08-12, charter Add.15 §A43.4). **BOTH HALVES FIRED.**

⛔ **The struck cell above said *"the COVERAGE half … did not fire"*. That is WRONG, and it contradicts
§1 and §1b of this same file.** The correct record, from the measurements already in this document:

| half | owner | fired? | measured |
|---|---|---|---|
| **COVERAGE** (needed well outside the union of the `k` launch diamonds) | spoke A | ⛔ **FIRED** | **0.7546 ± 0.0116** address-space uncovered (§1), threshold 0.20 ⇒ **3.77×** — and **§1b records it PERSISTING BIT-IDENTICALLY** after the payload-reach repair (0.7461 / 0.7520 / 0.7656, unchanged) |
| **TRAVERSAL** (in-flight evidence outside the current particle's diamond) | spoke B | ⛔ **FIRED** | **0.7715 ± 0.0081**, threshold 0.20 ⇒ **3.86×** (§2, above) |

**How the error arose (recorded so it is not repeated):** spoke B ran concurrently with spoke A's
**payload-reach repair** and inherited the pre-repair framing in which the *payload* half of coverage
was the live failure. The repair **closed the payload half** (`‖v_j‖ / capture` 1.172 → 0.692) and
**left the address-space half untouched and firing** — §1b states exactly that. Spoke B's sentence
generalised "the payload half is closed" into "coverage did not fire".

⭐⭐ **What does NOT change, and it is the part that matters for C2W9:** both halves share **ONE
measured mechanism** — the launch head **asserts the wrong wells** (precision 0.2356 / occupancy
0.2303), and the needed wells sit ~2 well-spacings away **as a consequence**. ⛔ **A wormhole cannot
recover set information the greedy deflation already discarded.** C2W9 therefore remains **DEFERRED
behind the launch-head work** (Add.15 §A43.3, Head-ruled) with this signature banked and quantified.

⛔ **No measurement in §1, §1b or §2 is altered by this correction — only the erroneous sentence is
struck.** Registry: **N310**.

### ⭐ What this signature says, and what it does not

- **It is a REACH failure, not a capacity failure.** The needed well sits a median **2.75 diamond
  radii** away; no settle budget crosses that, and `d ≥ 16` inertness was already measured at a
  **fully honoured** atom budget.
- ⭐⭐ **The evidence is not merely outside the diamond — it is outside by a well-spacing.** A
  wormhole/learned-`p₀` fix (C2W9) has to move a particle across **~2 well spacings**, which is a
  quantitative scoping input C2W9 did not previously have.
- ⛔ **It is NOT a statement that the store is empty.** The same store, the same physics and the same
  budgets score **0.8621 ± 0.0036** exact-set on unseen queries under **oracle addressing** (spoke B
  §2). The particles cannot *get* to the wells; the wells are there and they compose.
- ⛔ It is **not** a coverage failure: the union of the launch diamonds is spoke A's statistic and it
  did not fire. The failure is **in flight**.

### Provenance
Commit `5144384`+ (branch `c2w11-physics-organizer`, base `main @ 168a892`, worktree
`../CHLU-c2w11b`, main venv, **jax 0.9.0**, float32). `payload_radius = atom_payload_init_radius =
0.60`, `atom_width_frac_spacing = 0.37` (selected), feature-factored launches `k = 4`,
`σ_q = 0.15`, placing write, `a = 12`, read 400 + 800 at `γ = 0.05 → 0.02`, `dt = 0.05`.
Artifact: `.claude/outputs/c2w11-physics-organizer/run1/stage_traversal.json`.
