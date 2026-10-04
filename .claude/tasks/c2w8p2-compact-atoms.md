# c2w8p2-compact-atoms — ARM A: make the store capture by bounding how far each atom reaches

**Campaign 2, wave C2W8 PASS 2. Agent:** experiment-engineer.
**Worktree 1.** ⭐ **The ≤3 engineer-worktree cap is LIFTED for this overnight pass only** (Head, `ERRATA-C2W8-PASS2.md` §1) — all three pass-2 spokes run in parallel alongside the concurrently-live `pilot-ttt-nan-and-d5-wiring` spoke. ⚠ The cap returns to ≤3 at the end of this pass; this is a one-pass exception, not precedent. Branch **`c2w8p2-compact-atoms`** from `main @ 80d7d4b`, worktree `../CHLU-c2w8a`.
Writes `.claude/outputs/c2w8p2-compact-atoms.md` + artifacts to `.claude/outputs/c2w8p2-compact-atoms/`.
**Budget:** ≈ 1 day build + ≈ 6–9 h measured. Price cells before running; **cut seeds before cutting a
cell, and declare the cut.**

**Binding documents, read first, in this order:**
- `.claude/outputs/c2w8-well-lifecycle/ERRATA-C2W8-PASS2.md` **§1 — the four Head rulings that close
  the prereg's carried questions. Read it before the prereg.**
- `.claude/outputs/c2w8-well-lifecycle/PREREG-C2W8-PASS2.md` — **the pass-2 prereg: your gate
  (§3), your kill-conditions K6/K7 (§4), the numeric predictions P1–P5 you are measured against
  (§5). You implement them; you do not re-derive or re-tune them.** Additions go in a dated
  `ERRATA-C2W8-PASS2.md` filed **BEFORE** the cells they govern.
- The Head+Advisor **C2W8 PASS-2 DIRECTIVE** (2026-08-06, in the `[C2W8]` §10 entry).
- `.claude/outputs/c2w8-well-lifecycle.md` + `census.json` (pass 1: what you are fixing) ·
  `ERRATA-C2W8.md` §3 (the reach arithmetic).
- charter **§A26.6** (the write side is the live tier-ii route) · **§A28** · intervention **§5**
  (collapse modes #2/#6/#8/#11/#12 are all live in your rig) and **§8** (prohibitions).

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** none as a new claim — this is an **instrument/mechanism repair** on the write side.
  ⛔ No paper number, no tier-ii verdict, no full-CLU verdict, no I2 verdict.
- **Laundering control:** the kNN-in-φ launder is carried on every reading with the **byte ledger
  beside it** (pass-1 baseline: `clu_total_bytes` 360 960 vs `knn_launder_bytes` **288** — a
  **1 253×** ratio). ⛔ **No performance claim is made at pass 2** — the gate is retrievability. ⭐ **HEAD RULING (§1 Q3): the pass-2 gate is BYTE-BLIND** — no gate leg reads bytes — **but the ledger is still reported on every arm, and no performance number is quoted at the 1 253× ratio.**
- **Falsifies:** the gate (§3) fails on ≥ 3 seeds ⇒ compact atom influence does not make this store
  capture, and the arm is a measured negative reported as such.
- **Does NOT falsify:** losing to the launder (no performance claim here); a high foreign
  contribution (**diagnostic, never a target** — see below); failing to beat arm B.
- ⛔ **Depth is not quotable as feature importance** (§A23.5, ACTIVE).

---

## ⛔⛔ THE BINDING PROHIBITION (Head ruling — read before designing anything)

**DO NOT PIN ATTRACTORS TO DESIGNED ANCHORS.** Forcing the attractor to equal `φ(item)` makes the
settled point a deterministic function of item identity ⇒ **D2a ⇒ table-expressible**, violating
intervention §8 prohibition 2, and it flattens the basin interaction, superposition and manifold
storage the programme exists for. **Placement stays LEARNED and CONTINUOUS; basins stay free to
interact.** Any mechanism that pins, snaps, or regularizes the attractor toward `φ(item)` is
**non-compliant and does not ship, whatever it scores.** If you believe your design skirts this,
stop and ask the Hub rather than deciding it yourself.

**What you ARE changing:** *how far each atom's influence reaches.* Nothing about where wells sit.

---

## The diagnosis you are fixing (pass 1, measured — do not re-measure, build against it)

- **C3 locality HOLDS in parameter space** (own-leg violation rate **0.000**, exact) **and FAILS in
  function space** (78–84 % of writes raise the foreign contribution). **A write touches only its own
  atom block, but atoms have TAILS. Local in parameters is not local in the landscape.**
- Consequence: **foreign contribution exceeds own on 45 of 48 wells** (own median 0.518/0.282/0.123
  vs foreign 1.261/0.947/0.611) — an item's own well is a **minority of the landscape at its own
  site**. Depth RISES (0.74 → 1.66) while retrieval FALLS; site drift reaches ~10× key spacing.
- **`capture_radius` is exactly 0.000 on 47 of 48 wells while `λ_min > 0` everywhere** (0.791–8.873)
  — positive curvature is **necessary and not sufficient**.

⭐ **The design hint is this wave's own K2 result:** the trash region needed a **COMPACT** gate
(exactly zero beyond `r_k`) rather than a sigmoid, because **a sigmoid tail makes a "local" change
global**. Apply that lesson **one level down, to the atoms**.

## The build

**1 — K7 FIRST, and nothing else runs before it is green** (prereg §4; this is pass 2's most
important kill-condition and it exists because pass 1's gate legs were forced false by construction
and nobody noticed until review).
Prove the capture instrument can report a **positive**, two-sided and pytest-asserted:
- a store with an **analytically known** capture radius ⇒ `capture_radius` recovers it within a
  declared tolerance;
- a planted **flat** site ⇒ `capture_radius` returns 0.
⛔ Until K7 is green, a majority-positive G-CAP is **not evidence** — it is an untested instrument
agreeing with us. Pass 1 supplies exactly **one** non-zero reading in 48, far too thin to license the
instrument by observation. **Report K7's numbers before any arm number.**

**2 — the mechanism.** Compact or short-tailed atom influence, with widths **co-scaled to the
MEASURED key spacing** (`geometry.median_nn_task1`, per seed — 0.1407 / 0.1375 / 0.1468 at pass 1;
**measure it on your own run, never hardcode**). The kernel choice is yours and is the designed
lever — compact support, truncation, or a short-tailed family — **whatever the theory supports**;
state the form, its support, and its continuity/differentiability properties in your report, and say
what it does to the write's gradient. ⚠ A truncation that makes the write gradient exactly zero
outside a radius will interact with the settle; measure it rather than assuming it.

**3 — K6: ships behind a flag whose OFF path is bit-identical AND parameter-count-identical** to
`main @ 80d7d4b` (the K2 / P1 / psires precedent). Reddening the test un-ships the flag.

**4 — the gate: re-run pass 1's census, UNCHANGED.** `chlu/core/well_lifecycle.py` is **READ-ONLY**
for you — same instrument, same arithmetic, or the race against arm B is not a race. **If you need a
change there, do not make it: report it to the Hub in your first 10 lines.** Gate legs (all three,
≥ 3 seeds): **G-CAP** `capture_radius > 0` on a majority of live wells · **G-DEC** self-probe
`decode` above chance (0.0625) beyond 2 SE · **G-DRIFT** median `site_drift` below the measured key
spacing.

**5 — own/foreign is a DIAGNOSTIC and this is binding.**

⭐ **HEAD RULING (`ERRATA-C2W8-PASS2.md` §1 Q2): report own/foreign under BOTH aggregations —
MEDIAN and MEAN — each labelled. The MEDIAN is canonical** (the Hub's re-derivation: own
0.518 / 0.282 / 0.123, foreign 1.261 / 0.947 / 0.611); the directive's near-identical figures use a
different aggregation and the two get reconciled once, here, rather than circulating side by side.
⛔ **This changes nothing about its status: diagnostic, never a target, never a gate leg.** Report
it; **do not tune on it and do not gate on it.** Under private wells a high foreign contribution is interference; **in a factored store
it is the SIGNAL (compositionality)**, so over-fitting to own-dominance now buys a reversal later.
⭐ **The invariant that survives both designs is RETRIEVABILITY** — that is what the gate measures.

**6 — byte ledger on every arm including the launder**, with **(d, atom budget) declared as ONE
joint dial** (`n_atoms = max(atoms_per_item·K, min_atoms, round(512·√2^d))`; bytes/well grow ∝ d;
reach tightens as σ√d) and `γ_φ` holes counted.

## FILE OWNERSHIP (declared)

**You own:** `chlu/core/memory_potentials.py` (the atom kernel) · `chlu/core/clu_system.py`
(**flag + wiring only**) · `chlu/config.py` (**additive only**) · `chlu/experiments/exp_capture_armA.py`
(**new**) · `tests/test_compact_atoms.py` (**new**).
⛔ **READ-ONLY / DO NOT MODIFY:** `chlu/core/well_lifecycle.py` and `chlu/experiments/usage_telemetry.py`
(**the frozen census — the race depends on it**) · `chlu/core/friction_field.py` (K2 shipped; do not
disturb) · everything owned by the concurrent spokes: `c2w8p2-emission-head` (wt2:
`chlu/core/emission_head.py`, `exp_capture_armB.py`), `c2w8p2-instruments-and-debt` (wt3:
`chlu/experiments/cl_baselines.py`, `chlu/core/soft_certificate.py`) · the C2W6 files
(`train_cluformer.py`, `blocks.py`, `scripts/csf3/`, `exp_anti_erosion.py`) · the C2W7 files
(`multiplicity_read.py`, `monitors.py`, `factored_store.py`, `multiwell_read.py`).
⚠ Work **in your worktree**, never the shared main checkout (three spokes have now slipped on this).

## Acceptance (mechanical)
1. **K7 green and reported before any arm number** (two-sided, pytest-asserted).
2. K6: OFF bit-identical **and** parameter-count-identical, pytest-asserted.
3. The census re-run **unmodified**, all three gate legs reported per seed, ≥ 3 seeds.
4. own/foreign reported and **explicitly labelled diagnostic-not-target**.
5. Byte ledger on every arm including the launder, `(d, atom budget)` as one declared joint dial.
6. Full suite green on your branch with the count arithmetic stated.
7. Report's **first 10 lines** name any downstream reconciliation list (protocol §5 corollary).
8. Declared NOT-RUNs listed as NOT-RUNs, never as nulls.

⛔ **You do NOT build merge, prune, restoration, or any §2.7 claim cell** — those stay unbuilt until
the capture gate passes (pass 1's refusal to build verbs over empty populations was correct and
stands). ⛔ Never push `origin`; the Hub integrates and pushes `clu-dev`.
