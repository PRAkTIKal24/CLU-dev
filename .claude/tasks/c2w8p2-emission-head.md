# c2w8p2-emission-head — ARM B: a standard head on φ that EMITS the well parameters

**Campaign 2, wave C2W8 PASS 2. Agent:** experiment-engineer.
**Worktree 2.** ⭐ **The ≤3 engineer-worktree cap is LIFTED for this overnight pass only** (Head, `ERRATA-C2W8-PASS2.md` §1) — all three pass-2 spokes run in parallel alongside the concurrently-live `pilot-ttt-nan-and-d5-wiring` spoke. ⚠ The cap returns to ≤3 at the end of this pass; this is a one-pass exception, not precedent. Branch **`c2w8p2-emission-head`** from `main @ 80d7d4b`, worktree `../CHLU-c2w8b`.
Writes `.claude/outputs/c2w8p2-emission-head.md` + artifacts to `.claude/outputs/c2w8p2-emission-head/`.
**Budget:** ≈ 1 day build + ≈ 4–6 h measured (a forward pass is cheap; the census is the cost).
**Raced against `c2w8p2-compact-atoms` (wt1) on the SAME frozen census.**

**Binding documents, read first:** `.claude/outputs/c2w8-well-lifecycle/ERRATA-C2W8-PASS2.md` **§1 —
the four Head rulings closing the prereg's carried questions; Q4 confirms this arm may pass the gate
in a permanently claim-barred configuration. Read it before the prereg.** · `PREREG-C2W8-PASS2.md` (**your
gate §3, kill-conditions K6/K7/K8 §4, predictions P1–P4/P6 §5 — implement, do not re-derive**) · the
Head+Advisor **C2W8 PASS-2 DIRECTIVE** · `.claude/outputs/c2w8-well-lifecycle.md` + `census.json` ·
charter **§A28.1** (the DESIGNED write→φ organization gradient — this arm is its vehicle) · **§A26.6** ·
**§A4.5** (the factored store you must not foreclose) · intervention **§8** (prohibitions — read
prohibition 2 twice).

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** none as a new claim — a **write-side simplification** measured on the capture gate.
  ⛔ No paper number, no tier-ii verdict, no full-CLU verdict, no I2 verdict.
- **Laundering control:** kNN-in-φ launder + byte ledger on every reading (pass-1 baseline
  `clu_total_bytes` 360 960 vs `knn_launder_bytes` **288** = **1 253×**). ⛔ No performance claim at
  pass 2 — the gate is retrievability. ⭐ **HEAD RULING (`ERRATA-C2W8-PASS2.md` §1 Q3):
  the pass-2 gate is BYTE-BLIND** — no gate leg reads bytes — **but the ledger is still reported on
  every arm, and no performance number is quoted at the 1 253× ratio.**
- **Falsifies:** the gate fails ⇒ emitting well parameters does not make this store capture.
- **Does NOT falsify:** landing in the private-well configuration (**predicted, P6 = 0.85**) — that
  is a *claim-barring* outcome, not a gate failure, and both halves get reported.
- ⛔ Depth is not quotable as feature importance (§A23.5 ACTIVE).

---

## What this arm is, and why it is worth racing

A standard **MLP-class head on `φ`** emits the well parameters — **center, width, depth, payload** —
with **the well's functional form left designed exactly as it is now**. It is a **strict
simplification of the current write: a forward pass instead of 300 gradient steps.** It would remove:
- the `min_atoms` co-scaling explosion (`round(512·√2^d)` — 8 192 atoms at d=8, **131 072** at d=16),
- the write-budget / **N94** caveats,
- the **erodable-written-content** channel (nothing is written into atoms to erode),

and it **converts the accidental placement leak into the DESIGNED write→φ organization gradient
registered at charter §A28.1** — which is exactly the object §A28.1 ruled should be *built* rather
than inherited from an accident.

## ⛔⛔ THE DECLARED TRAP — read before you design the head

**An emission head that produces ONE PRIVATE WELL PER ITEM restores explicit per-item store
parameters and is LAUNDERED BY CONSTRUCTION.** It is the intervention doc's own degenerate endpoint
(§8 prohibition 2: *explicit per-item arrays, engineered separability, settled-point-only reads*),
reached **faster** than before. **The configuration that can ever carry a tier-ii claim is per-item
COEFFICIENTS OVER A SHARED WELL VOCABULARY.**

⇒ **K8 makes this machine-checkable rather than a promise:**
- every artifact you emit declares **`wells_per_item`** and **`vocabulary_shared: bool`** in its
  ledger, **pytest-asserted present**;
- any arm with **private wells per item** is labelled **`NO_TIER_II_CLAIM`** in its own artifact
  **and** in the census output, and **that label travels with every number from it**.

⭐ **You are NOT asked to build the factored store this wave. You ARE asked to specify this arm so it
does not foreclose it.** Concretely: the head's output interface must be expressible as
*coefficients over a vocabulary* with the private-well case as the degenerate `vocabulary_size =
n_items, coefficients = one-hot` special case. Write that interface down in your report even where
the shipped code only exercises the degenerate case. If your design cannot express the
shared-vocabulary case without a rewrite, **say so in your first 10 lines** — that is a first-order
finding about the arm, not a detail.

## The build

**1 — K7 FIRST (shared with wt1, and neither arm's numbers count before it is green).** Prove the
capture instrument can report a **positive**: a planted basin with an analytically known capture
radius is recovered within tolerance; a planted flat site returns 0. **Two-sided, pytest-asserted.**
⚠ **Coordinate with wt1 through the Hub, not directly** — if wt1 lands K7 first, import it; if you
land it first, say so and the Hub tells wt1. **Do not both edit the frozen census to add it** — K7's
tests live in your own test file against the read-only instrument.

**2 — the head.** MLP-class on `φ`, emitting center / width / depth / payload. Designed well form
unchanged. Trained through the designed write→φ organization gradient (§A28.1) — **declared and
byte-ledgered as a designed mechanism**, never as an inherited leak.
⛔ **The Head's binding prohibition applies to you too:** the emitted center is a **learned,
continuous** function of `φ` and **must not be pinned/snapped/regularized to `φ(item)`** — that is
D2a and it is table-expressible. ⚠ **This arm is the one most at risk of drifting into the
prohibition**, because emitting a center *is* nearly the pinning operation. State explicitly in your
report what stops it from being one, and if you cannot, stop and ask the Hub.

**3 — K6:** flag with OFF **bit-identical AND parameter-count-identical** to `main @ 80d7d4b`.

**4 — the gate: re-run pass 1's census, UNCHANGED.** ⛔ `chlu/core/well_lifecycle.py` is **READ-ONLY**
— same instrument, same arithmetic, or the race is not a race. Needed change ⇒ report to the Hub.
Legs: **G-CAP** (majority `capture_radius > 0`) · **G-DEC** (`decode` above chance 0.0625, 2 SE) ·
**G-DRIFT** (median `site_drift` < measured key spacing), ≥ 3 seeds.

**5 — own/foreign is DIAGNOSTIC, never a target.** Report, do not tune on it, do not gate on it.

⭐ **HEAD RULING (`ERRATA-C2W8-PASS2.md` §1 Q2): report own/foreign under BOTH aggregations —
MEDIAN and MEAN — each labelled. The MEDIAN is canonical** (the Hub's re-derivation: own
0.518 / 0.282 / 0.123, foreign 1.261 / 0.947 / 0.611); the directive's near-identical figures use a
different aggregation and the two get reconciled once, here, rather than circulating side by side.
⛔ **This changes nothing about its status: diagnostic, never a target, never a gate leg.**

In a factored store foreign contribution is the **signal**, not interference. **Retrievability is the
invariant that survives both designs.**

**6 — the byte ledger is this arm's sharpest column and you must be merciless with it.** A head that
emits well parameters moves bytes from the atom store into **head parameters** — count them.
`(d, atom budget)` stays one declared joint dial; report the arm's total against the pass-1 baseline
(360 960) **and** against the launder (288). ⛔ A head whose parameters exceed the store it replaces
is a finding, not a footnote.

## FILE OWNERSHIP (declared)

**You own:** `chlu/core/emission_head.py` (**new**) · `chlu/experiments/exp_capture_armB.py` (**new**) ·
`tests/test_emission_head.py` (**new**) · `chlu/core/clu_system.py` (**flag + wiring only** — ⚠ wt1
also touches this file for its own flag; keep your edit to a separate, clearly-delimited block and
expect the Hub to resolve adjacency at merge) · `chlu/config.py` (**additive only**, same adjacency
note).
⛔ **READ-ONLY / DO NOT MODIFY:** `chlu/core/well_lifecycle.py` · `chlu/experiments/usage_telemetry.py`
(**the frozen census**) · `chlu/core/memory_potentials.py` (**wt1's atom kernel**) ·
`chlu/core/friction_field.py` · `chlu/experiments/cl_baselines.py` + `chlu/core/soft_certificate.py`
(**wt3's**) · the C2W6 files (`train_cluformer.py`, `blocks.py`, `scripts/csf3/`, `exp_anti_erosion.py`) ·
the C2W7 files (`multiplicity_read.py`, `monitors.py`, `factored_store.py`, `multiwell_read.py`).
⚠ Work **in your worktree**, never the shared main checkout.

## Acceptance (mechanical)
1. **K7 green and reported before any arm number.**
2. **K8: `wells_per_item` + `vocabulary_shared` declared and pytest-asserted; `NO_TIER_II_CLAIM`
   label applied and travelling** if the arm is private-well.
3. K6: OFF bit-identical **and** parameter-count-identical.
4. Census re-run **unmodified**; all three gate legs per seed, ≥ 3 seeds.
5. Byte ledger with **head parameters counted**, against both the pass-1 store and the launder.
6. The shared-vocabulary interface **written down**, with the private-well case shown as its
   degenerate special case — or an explicit first-10-lines statement that it cannot be.
7. An explicit statement of **what stops the emitted center from being a pinned anchor**.
8. Full suite green with count arithmetic; reconciliation list in the first 10 lines; NOT-RUNs
   declared as NOT-RUNs.

⛔ **You do NOT build merge, prune, restoration or any §2.7 claim cell.** ⛔ Never push `origin`.
