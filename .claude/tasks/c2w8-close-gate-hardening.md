# c2w8-close-gate-hardening — repair the census gate before it scores anything else

**Campaign 2, C2W8 CLOSE-OUT. Agent:** experiment-engineer. **ONE worktree.**
Branch **`c2w8-close-gate-hardening`** from **`main @ 9e0bb25`**, worktree `../CHLU-c2w8close`.
⚠ **Name the base explicitly** — the shared checkout sits on a live spoke's branch:
`git worktree add ../CHLU-c2w8close -b c2w8-close-gate-hardening 9e0bb25`.
Writes `.claude/outputs/c2w8-close-gate-hardening.md` + artifacts to `.claude/outputs/c2w8-close/`.
**Budget:** ≈ 1 day. **No new science cells** — this is instrument repair. Any cell you run is a
designed negative or a regression check, not a result.

⛔⛔ **THIS SPOKE IS THE MECHANICAL GATE C2W11 WAITS ON.** Charter §A32.3: *"No future census number is
quotable until these land."* Your deliverable is the file the next wave's spokes will be gated on.

**Binding documents, read first, in this order:**
- charter **ADDENDUM 11 (§A31–§A32) IN FULL** — especially **§A31.5** (the D2a signature at design
  level) and **§A31.6** (the instrument debts, several of which are yours).
- charter **ADDENDUM 12 §A33.1 — THE MECHANICS/VALUE RULE, and it governs your design:**
  > every gate leg is labelled **MECHANICS** (does the mechanism work) or **VALUE** (is it worth
  > anything). Mechanics legs are pass/fail at component level. **VALUE legs exist only at tier level
  > with the tier's own control.** ⛔ **A launder margin on a component gate is a DIAGNOSTIC, NEVER a
  > pass condition.**
  ⇒ **A3 (the launder-margin leg) must be re-labelled DIAGNOSTIC and removed from the pass condition.**
  §A34.8: **G-ADDR is re-scoped as a per-feature MECHANICS instrument, permanently barred from VALUE
  duty.** Label **every** leg you touch.
- `.claude/outputs/c2w8p3-{gate-addr,phi-geometry,capture-strong-phi}.md` + their JSON artifacts ·
  the `[C2W8-PASS3-SPINE]` §10 entry · `PREREG-C2W8-PASS3.md` §4 (the scale guard you are fixing).

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — **instrument repair.** ⛔ No claim cell, no performance number, no verdict of any
  kind, no re-scoring of banked results into new claims.
- **Laundering control:** N/A — and note **that is now the point**: A3 becomes a diagnostic column,
  never a pass condition (§A33.1).
- **Falsifies:** a repaired leg that cannot fail its own designed negative does not ship.
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ N94 discipline on any reading.

---

## The six items (all land in one spoke; the deliverable is a per-item boolean table)

### (i) ⛔⛔ THE TWO-SIDED DRIFT LEG / FLOOR — the most important item
**Measured basis:** across all 9 pass-3 cells, **Spearman ρ(A1, G-DRIFT ratio) = −0.967** and
**ρ(A1, settle↔launder agreement) = +0.933**. ⇒ **the gate's score is a near-monotone function of
settle-collapse: it REWARDS table-behaviour**, which is exactly the D2a configuration intervention
**§8.2 prohibits**. ⚠ **The existing per-arm boolean `best_is_also_lowest_drift` HIDES this** — it
reads False on all three arms, decided by drift ties of **0.004**.
**Build: a TWO-SIDED drift leg (or an explicit drift FLOOR)** so the census **FAILS on drift → 0**
rather than rewarding it. Drift too large = cannot address; **drift too small = table-expressible and
must also fail.** The floor's value is yours to choose and **declare** — derive it from a measured
quantity (e.g. a fraction of the *codebook* spacing), never a bare constant.
⛔ **DESIGNED NEGATIVE, PYTEST-ASSERTED (mandatory):** a planted **near-zero-drift, table-like** store
must **FAIL** the new leg. A leg that cannot fail on the degenerate configuration is not a repair.

### (ii) A1 REPORTED WITH MARGIN-IN-SE BESIDE THE BOOLEAN
**Measured basis:** `randconv` scored **31 / 31 / 29 of 128** against a **32/128** threshold — it
**failed by ONE read on 2 of 3 seeds**. A discrete threshold with no margin reported turns a tie into
a verdict. **Report `margin_in_se` beside every A1 boolean**, everywhere A1 is emitted.

### (iii) THE RATIFIED SCALE RULING — full-state co-scaling INCLUDING the payload channel
**Head-ratified (§A31.6):** **address-only rescaling is NOT a symmetry of the system** — the payload
channel is absolute. **The legal rescale is FULL-STATE co-scaling, address AND payload together**,
under which **A1/A3 return to 4 dp and G-DEC/G-DRIFT survive**.
⇒ **Re-implement the `PREREG-C2W8-PASS3.md` §4 guard accordingly.** ⚠ **The old guard was defective in
a specific way you must not reproduce: it bounded the METRIC and not the VERDICT** — a legal rescale
moved A1 by **+0.0469** (inside the registered 0.05 bound) and **still flipped the leg's verdict**.
**The repaired guard must assert VERDICT STABILITY under the legal rescale, not merely bounded metric
movement.**

### (iv) ⛔⛔ THE `covered` / `n_never_read` FIX — they are LAUNCH-POINT statistics
**Measured basis (Advisor erratum 1, §A31.1):** `covered` is computed on **q₀** against the codebook
in `clu_system._read_diagnostics` ⇒ **store-invariant by construction: same φ + same codebook ⇒ the
same number whatever the store does.** That is why "58/62/62 unassigned, digit-identical" looked
decisive and was **vacuous**, and the banked `n_never_read` inherits it (⚠ `frac_never_read = 1.0000`
on **9/9** CIFAR cells while settle-side **A2 = 0.125–1.000**).
**Do BOTH:** replace with **settle-side equivalents** where a settle-side measure is what was meant,
**and relabel the launch-side quantity** unambiguously (e.g. `launch_coverage_*`) wherever it is kept.
**Correct the banked-telemetry captions** so no future reader re-derives the retired sentence.
⛔ **DESIGNED NEGATIVE, PYTEST-ASSERTED (mandatory):** a store **mutated so reads land differently**
must leave the **launch-side** statistic **unchanged** and move the **settle-side** one. That single
test is the proof the two are different quantities.

### (v) THE `d_safe` POPULATION FIX
`d_safe = d_safe_frac × med_nn` where `med_nn` is the NN spacing of a **~200-key sizing set**, applied
to a store holding **16 items** ⇒ **monitor #3's 0.000 refusal rate was arithmetic, not a finding.**
**Derive `d_safe` from the STORE population.** ⚠ **Report the refusal rate; do NOT tune it to a
target** — a gate that refuses *because it was tuned to* is the same defect wearing new clothes.
⚠ Pass 3 partially relieved this already (refusal 0.000–0.111, 5/9 non-zero); state the before/after.

### (vi) THE HOUSEKEEPING LIST (all six, none may be silently skipped)
1. **`own_foreign_site_depth` hard-codes the Gaussian kernel** ⇒ kernel-mismatched under any
   compact-atom arm. **Fix + a designed CROSS-KERNEL test.**
2. **`theta_att`'s arm-dependent range** (degenerates to 0.0000 when everything captures) ⇒ ⛔ **`P`
   is never comparable across arms without `n_non_capturing` beside it.** Enforce in the emitter.
3. **`ERRATA-C2W8-PASS2.md` §2 numbering collision** — ⚠ **the Hub already resolved this at pass-2
   integration** (wt1's later block renumbered to **§5** with a dated banner, because the earlier §2
   is cited by hash-stable references in tracked code). **VERIFY on disk and record it as closed; do
   NOT re-resolve it.** If your check disagrees with that account, say so.
4. **The stale `tests/test_cifar_strong_phi.py:66-72` comment** — the x64 bug it cites is **FIXED**.
   The `backbone = "mlp"` choice may stay; **the stated reason must change.**
5. ⭐ **NEW, from §A31.6 — the census must REFUSE to run at a non-selected width.** Arm A's banked
   runs used `atom_width_frac_spacing = 1.5` while the shipped config default is **0.5**. A census
   that silently runs at a width nobody selected produces numbers nobody can attribute. **Make it
   refuse, loudly.**
6. **The cue-difficulty arm-dependence** (κ_q normalised on *sizing* spacing while the read must beat
   the *codebook* spacing — 0.927 / 0.875 / **0.710**, a 30 % spread). **At minimum: emit
   `cue_sigma / codebook_spacing` on every cell** so no future cross-arm comparison is made blind to
   it. A full fix (normalise κ_q on the codebook spacing) is welcome if it is clean; **declare which
   you did.**

⭐ **Items (v) and (vi.6) are the SAME underlying defect** — the ~200-key *sizing* spacing standing in
for the 16-item *store* spacing — which also produced the retracted §A29.5 mechanism. **Fixing the
population choice fixes all three.** Say so in your report.

---

## DELIVERABLE (the mechanical gate C2W11 waits on)

> **`.claude/outputs/c2w8-close/GATE-HARDENING-DONE.json`**

Carrying a **per-item boolean table** — one entry per item (i)–(vi), each with `done: true|false`,
the test names that assert it, and a one-line statement of what was changed — plus a top-level
`gate_hardening_done` computed **mechanically** as the AND over all items. ⛔ **Anything you could not
land is `false` with its reason, never omitted and never quietly true.**

## FILE OWNERSHIP (declared)
**You own:** `chlu/core/well_lifecycle.py` · `chlu/experiments/exp_well_lifecycle.py` ·
`chlu/core/clu_system.py` (**the `_read_diagnostics` / `covered` fix + the width refusal**) ·
`chlu/core/soft_certificate.py` (item v) · `tests/test_gate_addr.py` · `tests/test_well_lifecycle.py` ·
`tests/test_cifar_strong_phi.py` (**comment only**) · `chlu/config.py` (**additive only**) ·
`.claude/outputs/c2w8-well-lifecycle/ERRATA-C2W8-PASS2.md` (**verify only, item vi.3**).
⛔ **DO NOT TOUCH — the live `pilot-ttt-nan-and-d5-wiring` spoke's territory:** `scripts/csf3/` ·
`chlu/training/train_cluformer.py` · `chlu/core/blocks.py` · `chlu/experiments/exp_cluformer_pilot.py`.
⛔ Also do not touch `chlu/core/emission_head.py` or `chlu/experiments/exp_capture_strong_phi.py`
(banked pass-2/3 arms — **read them, do not edit; re-running a banked arm is not your job**).
⚠ Work **in your worktree**, never the shared checkout.

## Acceptance (mechanical)
1. `GATE-HARDENING-DONE.json` exists with a per-item boolean table and a mechanical
   `gate_hardening_done`.
2. **Designed negatives pytest-asserted for (i) and (iv)** — the table-like store FAILS the drift
   leg; the mutated store moves the settle-side statistic and leaves the launch-side one unchanged.
3. **Every leg you touch is labelled MECHANICS or VALUE**, and **A3 is re-labelled DIAGNOSTIC and
   removed from the pass condition** (§A33.1).
4. The scale guard asserts **verdict stability**, not merely bounded metric movement.
5. Full suite green on your branch, **count arithmetic stated with the checkout named** (⚠ counts are
   comparable only within one checkout; the base at `9e0bb25` is **1555 selected** in a fresh
   worktree).
6. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs, never nulls.

⛔ You do NOT build merge/prune/restoration verbs (deferred). ⛔ You do NOT re-open the arm race.
⛔ You do NOT chase daylight — §A32.1 prohibits a pass 4. ⛔ Never push `origin`; the Hub integrates.
