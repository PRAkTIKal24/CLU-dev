# c2w8-well-lifecycle — over-dig → merge-to-budget → prune-below-budget → `γ_φ(q)` (first use)

**Campaign 2, wave C2W8 ("Consolidation + trash" — the maintenance phase). Agent:** experiment-engineer.
**Worktree 1 of ≤3.** Branch **`c2w8-well-lifecycle`** from `main @ d70898b`, worktree `../CHLU-c2w8`.
Writes `.claude/outputs/c2w8-well-lifecycle.md` + artifacts to `.claude/outputs/c2w8-well-lifecycle/`.
**Budget:** ≈ 1 day stage-1 build + census; ≈ 1.5 days stage-2 build + cells, **only if stage 1 unlocks**.
Price every cell before running it; **cut seeds before cutting a cell, and declare the cut.**

**Binding documents, read first, in this order:**
- `.claude/outputs/c2w8-well-lifecycle/PREREG-C2W8.md` — **the wave prereg. Your kill-conditions
  K1–K5, your instrument definitions and the numeric predictions N1–N9 are already registered. You
  implement them; you do not re-derive them and you do not re-tune them.** Anything you must add
  goes in a dated `ERRATA-C2W8.md` block filed **BEFORE** the cells it governs.
- charter **§A21 C2W8 row** (your scope) · **§A20.3(d)** + **§A20.6-P2** (your specification) ·
  **Addendum 9 §A28.3** (the three mechanics you design against — §1 of the prereg).
- charter **§A9.9** (deletion-vs-sharing frontier — this is what K4 enforces) · **§A23.5** (I2
  deferred, depth-as-feature-importance caveat ACTIVE) · **§A27.1** (the decay-netting requirement).
- intervention doc **§8.1/§8.2** (no isolated arms; no moving toward per-item arrays) and **§5**
  (the 13 anti-collapse modes — #9 payload-dependent lifetimes and #12 starve-and-overwrite are
  live in your rig).
- `.claude/outputs/c2w6-anti-erosion/PREREG-AntiErosion.md` §I2 + `.claude/outputs/c2w6-erosion-adjudication.md`
  (why the I2 rule is self-capping, why slot ≠ well, why `ρ(LOO)` is undefined).

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** lifetimes + isolation (designed forgetting through merge/prune/trash), as a **full-CLU
  component build**. ⛔ No paper number. ⛔ **No tier-ii verdict, no full-CLU verdict** (§A28.4).
- **Laundering control:** **kNN-in-φ at matched memory** (N89 / CM-22(i)) on **every** performance
  cell, plus the ± consolidation ablation (full CLU with X vs full CLU without X), plus the byte
  ledger on all arms **including the launder and including the γ_φ holes' own parameters**.
- **Falsifies:** K1 kills (nothing to merge, nothing to prune) → stage 2 is not built. K3 fails
  (the criterion cannot separate deep-unread from shallow-read) → the prune verb does not ship.
  K2/K4 red → the trash flag / the merge verb does not ship.
- **Does NOT falsify:** losing to iCaRL or to replay (CM-23(q): never claimed under any outcome);
  a launder margin that stays negative (that is N5's registered prediction, prior 0.80 — it is a
  measured re-price, not a surprise); an eroded well having no depth to restore (mechanic 1).
- ⛔ **Depth is not quotable as feature importance** (§A23.5, ACTIVE). ⛔ **"+0.510" never without
  "−0.036 laundered"** in the same paragraph (CM-23(q)).

---

## FILE OWNERSHIP (declared; the w26 zero-conflict practice, standing)

**You own and may modify:**
`chlu/core/controller.py` · `chlu/core/clu_system.py` · `chlu/core/friction_field.py` ·
`chlu/core/well_lifecycle.py` (**new**) · `chlu/experiments/usage_telemetry.py` (**new**) ·
`chlu/experiments/exp_well_lifecycle.py` (**new**) · `tests/test_well_lifecycle.py` (**new**) ·
`tests/test_usage_telemetry.py` (**new**) · `chlu/config.py` (**additive only** — new fields, no
existing default changed) · `chlu/cli/experiment_cmd.py` (**additive only** — one new command).

⛔ **DO NOT TOUCH — other waves' declared files:**
- **C2W6 (closing out):** `chlu/training/train_cluformer.py` · `chlu/core/blocks.py` · `scripts/csf3/` ·
  `chlu/experiments/exp_anti_erosion.py` · `tests/test_anti_erosion.py`.
- **C2W7 (closing out):** `chlu/core/multiplicity_read.py` · `chlu/core/monitors.py` ·
  `chlu/core/factored_store.py` · `chlu/core/multiwell_read.py` · `tests/test_multiplicity_read.py` ·
  `tests/test_reader_identity.py`.
- **`c2w8-cifar-strong-phi` (your concurrent sibling, wt2):** `chlu/experiments/exp_cl_entry.py` ·
  `chlu/experiments/phi_encoders.py` · `chlu/experiments/exp_phi_read_in.py` ·
  `chlu/experiments/exp_phi_stream.py`.

⚠ **You need the CL stream but you do not own `exp_cl_entry.py`.** Import the stream builder and the
kNN-in-φ launder **read-only**. If you need a change there, **do not make it** — report it as a
one-line request in your first 10 lines and the Hub routes it to the sibling spoke.
⚠ Work **in your worktree**, never in the shared main checkout (two spokes have now made this
mistake; both caught it, do not be the third).

---

## STAGE 1 — the rig, the instrument, the census (the KILL comes before the BUILD)

**Nothing in stage 2 is written until stage 1's verdict is filed.** This is the C2W3
`route3-stage1-plus-2x2` shape, chosen deliberately: §A9.5 moved into stage 1 killed a stage-2 build
for hours of measurement, and that is now standing doctrine.

**1.1 — The rig: the CL stream on the FULL CLU system.**
Port the Split-MNIST stream onto `CluSystem` (learned `V_θ`, `build_system(cfg, key, phi, psi)`),
with `exp_cl_entry`/`exp_phi_stream` supplying **only** the stream, the φ (`task1_only` regime — the
binding primary, no leakage), the baseline table and the kNN-in-φ launder.
⛔ **The arm's store is the learned `V_θ`. The CL harness's designed per-item Gaussian array is NOT
an arm** — it is a **labelled reference row** carrying the banked w25 numbers. Moving toward
per-item arrays for a clean number is intervention §8.2 and is forbidden.
- **"Extension" is defined mechanically, not adjectivally:** the stream is lengthened until
  `overdig = n_items_admitted / well_budget ≥ 2.0` (prereg §3.4) so that **capacity pressure is the
  binding constraint**. Report `overdig` on every cell.
- Anti-collapse modes **#12 (starve-and-overwrite)** and **#9 (payload-dependent lifetimes)** are
  live here. Monitor them and report their trip state; a starved write reads as a capacity result
  when it is not (w26).
- ⚠ **The pilot's warning applies:** a `CluSystem` fed a stream may dig no wells at all
  (`cluformer-pilot`: depth saturated at ~0.045 vs the shipped 0.46–0.80; surviving hypothesis =
  atom placement at init). **Measure well depth on this rig before anything else and report it.**
  If the store is inert, say so immediately — an inert store makes the whole census vacuous for a
  reason that is *not* K1's reason, and the two must not be confused. `atom_local_radius`
  (localized atom init) is available to you as a **declared designed mechanism** in the ledger.

**1.2 — The `U` telemetry (prereg §3.2, build requirement B2).**
Item-id-keyed, never slot-keyed. **Registered primary proxy = `read_hits(i)`**, accumulated over the
stream via the existing `Controller.touch` path extended to record reads. The LOO
`loss_contribution` proxy is **reported only beside its ICC(1,1)** and, if ICC ≤ 0, labelled
**UNDEFINED** with **no number quoted from it** — C2W6 measured ICC negative 3/3 and `ρ(LOO) = +0.067`
was a non-measurement. ⛔ Depth does not enter `U`.
⛔ **You report no I2 verdict.** The I2 correlation test is a declared NOT-RUN (prereg §9), deferred
to C2W10. You are building the instrument that makes "never read" computable, nothing more.

**1.3 — Decay-netting (build requirement B1, §A27.1).**
Every depth/erosion curve is reported **RAW and NETTED**, side by side, per slot, keyed by
`last_write_chunk` (which moves 0→12 within a slot and drifts the exponent). The law is exact
(predicted per-tick drop `0.039211` vs measured median `0.039211` over 717 readings); extend the
existing residual instrument in `exp_anti_erosion.py` **by importing it, not by editing that file**
(C2W6 owns it). A flattening/restoration claim off a raw curve alone is non-compliant.

**1.4 — The census (K1) + its designed negatives.**
Compute `P` (prunable) and `M` (mergeable) per prereg §3.3, ≥ 3 seeds, at `overdig ≥ 2`.
Report `is_attractor` against the **measured** capture floor `θ_att` (SC-6 bisection), never a
guessed constant — mechanic 1 means an eroded well is already gone and must be counted separately
from a live-but-unread one. **Report both populations separately: `{eroded, not attractor}` vs
`{live attractor, never read}`.** Only the second is prunable.
**Designed negatives, pytest-asserted before the census runs:** a hand-built store with 4 known
never-read attractors must read `P ≥ 4/n_live`; one with 3 known near-duplicate pairs must read
`M ≥ 3/n_pairs`. **A census instrument that cannot see a planted population cannot license a kill.**

**1.5 — STAGE-1 DELIVERABLE, and it is the mechanical gate:**
> **`.claude/outputs/c2w8-well-lifecycle/census.json`**
containing, per seed: `P`, `M`, `overdig`, `theta_att`, the two well-state populations, the raw and
netted depth curves, the `U` telemetry summary, the designed-negative assertions' results, and the
field **`stage2_unlock: true|false`** computed **mechanically** by the prereg §5 K1 rule
(`UNLOCK iff P ≥ 0.05 or M ≥ 0.05` on the seed mean; `KILL iff both < 0.05 on every seed`).

⛔ **If `stage2_unlock` is `false`: STOP. Do not build stage 2.** Write your report with the census
as the wave's finding — a vacuity result with its mechanism, caught before a build instead of after
one. **That is a wave product, not a failure, and the Hub will report it as such.** Say so plainly
and hand back.

---

## STAGE 2 — the lifecycle (unlocks ONLY on `census.json: stage2_unlock == true`)

Build the kill-condition before each verb, in this order. **Each guard needs a demonstrated
designed negative, pytest-asserted** (standing rule).

**2.1 — K2 FIRST, then the trash region.** `γ_φ(q)` ships **OFF**; OFF is **bit-identical AND
parameter-count-identical** to the pre-build path (the P1 / psires precedent, and the C2W6 standing
rule that reddening the test un-ships the flag). Designed negatives, both pytest-asserted:
(a) a hole **at** a written well's site measurably destroys that well's retrievability;
(b) a hole **far from every well** leaves every read **bit-identical**.
Then wire `friction_field.py` into `CluSystem`'s settle — it is currently referenced nowhere in
`clu_system.py`, so **this is the trash region's genuine first use** (built C1, never used). Prop-11
gives exact volume contraction; the field is already learnable with adaptive spawn/prune — you are
plumbing and gating it, not rewriting it.

**2.2 — merge-to-budget, on MECHANICAL criteria.** Over-dig freely, then merge down to the designed
well budget. Criteria are **mechanical and measurable** (prereg §3.3 `M`): payload distance below
the registered threshold AND center separation below the SC-1/SC-2 certificate radius. ⛔ **Spurious
shallow wells are TRASHED, never merged into meaningful ones** (§A20.3(d), verbatim) — a merge that
absorbs a spurious well into a real one is a defect, and you assert against it.

**2.3 — K4 BEFORE the merge verb ships: deletion as a CURVE (§A9.9).** Exactness preserved on the
unmerged (private) fraction; **measured degradation** on the merged fraction. Designed negative,
pytest-asserted: deleting a merged item shows a measurable departure from `AUC = 0.5000`, **or** the
merge is recorded in the ledger as deletion-destroying for that item. ⛔ **Byte-exact deletion
(AUC 0.5000 ± 0.0000) is never spent silently** — it is a banked capability and the ledger is where
it is spent, visibly, or not at all.

**2.4 — K3 BEFORE the prune verb ships: prune is a DECISION, not depth.** Two planted wells,
pytest-asserted: a **deep but never-read** well **IS** pruned; a **shallow but frequently-read** well
is **NOT** pruned. ⛔ **If the criterion cannot separate these, it is a depth policy wearing a usage
costume and it does not ship** — regardless of benchmark performance. This is Add.9 §A28.3(ii) as a
test, and the depth-as-feature-importance caveat is why it exists.

**2.5 — prune-below-budget as a CONTROLLER DECISION, pruned wells → `γ_φ(q)`.** The controller
decides (verb + policy); the criterion is `U`-based (§3.2), never depth. Pruned wells are **routed
to the trash region**, not merely deleted — that routing is the capability under test, and N9 asks
whether it does anything beyond prune-alone.

**2.6 — depth restoration (A20.6-P2, "protection by upkeep").** Consolidation re-packs and restores
depths. ⚠ **Measured on the NETTED curve** (B1) — netting is exactly the correction that separates
real restoration from allocator drift (it moved C2W6's E1 seed 0 by −34 %). ⚠ Mechanic 1: a well
that eroded to zero is **not an attractor** and there is nothing there to restore; restoration is
measured only on wells that still exist. Extend `CluSystem.consolidate()` (currently re-pack +
self-probe + certificates); do not replace it.

**2.7 — the claim cells: full CLU ± consolidation, long stream, capacity pressure.**
- ≥ 3 seeds (≥ 5 where a margin is the claim), `overdig ≥ 2`, **N94-compliant or labelled
  non-promotable**.
- **K5 on every cell:** kNN-in-φ at matched memory + the ± ablation + the byte ledger on all arms
  **including the launder and including the γ_φ holes' parameters** (⚠ γ_φ holes are bytes; a trash
  region off the ledger is a hidden capacity increase — the §A9.6 ledger-drift collapse mode).
- ⭐ **Report N5 — the launder margin (CLU − kNN-in-φ) — as the wave's headline quantity**, banked at
  **−0.036**. Registered prediction: it does **not** cross zero (prior 0.80). Report it whichever way
  it lands, with its sign, its SE and its seed count.
- **Deletion/lifetimes ride in the "and also" position** (standing) — the K4 curve is that suite's
  contribution, never a headline.

---

## Acceptance (mechanical)

1. `census.json` exists with a mechanically-computed `stage2_unlock`, and the two designed-negative
   census assertions are green.
2. Every guard (K2, K3, K4) has its designed negative **pytest-asserted and named in your report**.
3. Every depth/erosion curve appears **raw AND netted**.
4. Every performance cell carries its launder, its ± ablation and its byte ledger.
5. `friction_field` OFF is bit-identical and parameter-count-identical; the full suite is green on
   your branch, with the count arithmetic stated (`baseline + your new tests = total`).
6. Your report's **first 10 lines** name any downstream reconciliation list (protocol §5 corollary —
   an unowned reconciliation sat live for two waves once; not again).
7. **Declared NOT-RUNs are listed as NOT-RUNs, never as nulls.**

## Honesty clauses carried
If the store is inert on the ported rig, say so in your first 10 lines — do not census an inert
store and call it K1. If a cell is cut for budget, declare the cut and what it cost. If a designed
negative fails, the flag does not ship and you report that; a guard that cannot fail is not a guard.
⛔ You never push `origin`; the Hub handles integration and `clu-dev`.
