# c3-ladder-amendment-2 — the byte-convention audit, the ceiling-as-ceiling rule, per-layer calibration, and ONE consolidated amendment

**Campaign 3, wave 1. Agent:** experiment-engineer. **ONE worktree (wt1).** ⛔⛔ **THIS GATES ALL LADDER TRAINING** — Advisor + Head, 2026-08-13: *"one review, then arms train."* **Zero cells have run**, so this is an amendment before the fact, not a revision after it (the C2W11 Amendment-1 precedent governs). ~1.5 days.
Branch **`agent/experiment-engineer/c3-ladder-amendment-2`** off **`main`** (all three C3 spokes are merged; the Hub confirms the SHA at spawn).
Writes `.claude/outputs/c3-ladder-amendment-2.md`; amends `.claude/outputs/c3-rival-ladder-prereg/PREREG-C3-LADDER.md`.

**Binding documents:** `PREREG-C3-LADDER.md` **IN FULL** (esp. **§2.5**, **§3**, **§4**, **§4.3**, **§7.1**, and **AMENDMENT 1**, whose format you follow) · `.claude/outputs/c3-rival-gdn2.md` **§F1–F2** · `.claude/outputs/c3-rival-mamba2.md` **§F1–F2** · `.claude/outputs/c3-gb-landing.md` reconciliation item 3.

---

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** **none — convention audit + prereg amendment.** ⛔ **Train zero ladder arms.** No bpc, no comparison.
- **Falsifies:** a rival spec whose pinned bytes disagree with what the official implementation actually allocates; an arm reported compliant that exceeds the ceiling under the ruled convention; a calibration change that alters the pilot/run-3 path.
- **Does NOT falsify:** a rival needing more shrinking than the prereg assumed. **That is the audit working.**

## 1. ⭐⭐ THE CONVENTION HOLDS — and now it is AUDITED, not derived

**RULED (Advisor + Head, no exceptions): total state bytes AS DEPLOYED INCLUDES AUXILIARY CACHES** — convolution state, normalizer state, decay accumulators, and anything else the arm must carry between chunks at inference. ⛔ Not just the "main recurrent state".

⭐⭐ **THE RIDER, and it is the substance of this task: "the formula is the documentation, the allocation is the truth."** Audit **every** rival spec by **counting the official implementation's allocated inference buffers** — the `state-spaces/mamba` `allocate_inference_cache` pattern is the reference method: find the function the official code uses to allocate its inference state, and count **what it actually allocates**, not what the paper's formula says. **Pin each result with a test** so a future edit that drifts from the official allocation fails loudly.

- Apply to **all six** `RIVAL_SPECS` rows, not only the two with live arms. A row without a live arm still enters the tier-iii table.
- Where the audited number **differs from the pinned one**, the audited number wins and the delta is **reported per row** with the buffer that was missing.
- Where the official implementation cannot be reached for a row, say **NOT OBTAINED** and mark that row's number **formula-derived, unaudited** — ⛔ do not guess and do not quietly keep the old value as if audited.

**Known already, fix as part of the audit:**
- ⛔ **GDN-2 omits short-convolution state (+220,320 B at the shrunk config)** ⇒ **1.0973× over** the ceiling under the ruled convention. **RE-SHRINK GDN-2 to fit total-including-conv.** Its paper-convention figure (0.9922×) **stays, as a clearly labelled diagnostic column** — ⛔ never as the compliance number.
- ⛔ **`ArmLedger.dtype_bytes` is hardcoded to fp32** and mis-states every bf16 row (the total is right, the declared width is wrong). Fix it so `dtype_bytes` reflects the row's actual width — the no-normalisation ruling makes the declared width load-bearing, not cosmetic.

## 2. ⚠⚠ AN ANTI-HOBBLING DEFECT THE AUDIT MUST ALSO CLOSE — the Hub is naming it, it is not in the relay

`shrink_to_budget()` solved Mamba-2's knob **on the 24-layer reference geometry**, but **the shell deploys 12 layers** ⇒ the arm sits at **0.4949×** of the ceiling. ⛔ **That is not thrift, it is hobbling:** at 12 layers the arm could afford roughly twice the `d_state` it was given and still fit under the ceiling, and **the anti-hobbling rule says a rival gets its strongest admissible form.** This program has already had **one C2W10 verdict inverted** by a hobbled classical arm; a hobbled *learned* rival in the tier-iii table would be worse, because it is the primary claim's control.

⇒ **Solve every rival's shrink knob AT THE DEPLOYED GEOMETRY, not at the paper's reference geometry**, and report each arm's occupancy after re-solving. ⛔ An arm materially under the ceiling with a knob it could have spent is a **finding to report**, not a saving. ⚠ This is a *selection rule* and therefore falls squarely under §3's standing rule — pre-register it.

## 3. ⭐ THE CEILING IS A CEILING (prereg amendment)

- The budget is a **CEILING: ≤ 2,097,152 B**, with **occupancy reported per arm**. It is not a target and not an equality.
- ⭐ **Deployed per-arm bytes under integer-geometry constraints are THE VALUES OF RECORD.** GDN-2's **2,080,800 B at 0.992× is COMPLIANT** — the pre-registered `2,097,152` was a *solver output* and `n_heads = 6` has no integer head geometry at `d_model 512` (`512/6 = 85.33…`), so it was never realizable.
- ⭐⭐ **STANDING RULE, added by this amendment: PRE-REGISTER CONSTRAINTS AND SELECTION RULES, NEVER SOLVER OUTPUTS.** A pre-registered number that a solver has to reproduce exactly will be wrong the moment a constraint (integer heads, layer count, an auxiliary buffer) bites. Pre-register *"the largest knob value whose deployed total is ≤ the ceiling, integer geometry required"* — then the solver's answer is a **result**, not a **prediction that failed**. State this rule generally; it outlives this ladder.

## 4. ⭐ CALIBRATION IS PER STORE-BEARING LAYER (prereg design line + code)

**RULED:** `calibrate_atom_group_centers` and `calibrate_phi_gain` calibrate **per store-bearing layer, against that layer's own activation statistics**, and the definition must be **placement-independent** — it must read correctly for G-B's 3-of-12, for 12-of-12, or for any other placement, ⛔ with no layer index hard-coded anywhere. (Under G-B, layer 0 has no store; the old behaviour calibrated a store against a storeless layer's latents.)

- Enters the amendment as a **design line**, since it moves an initialisation and is therefore claims-relevant.
- ⭐ **One smoke diagnostic: own-layer vs layer-0 calibration, clearly labelled, ⛔ never a claim** and never a pass condition. It exists so the size of the change is on the record, not to argue it was right.
- ⚠ **Do not disturb the pilot/run-3 path.** Run 3 is a bit-identical continuation of run 2 — if this change alters the pilot config's flag table or its initialisation, **run 3's exemption breaks**. **Assert with a test**, exactly as `store_layers` did.
- ⭐ The spoke that flagged rather than patched this is **noted approvingly by the Advisor**; hold that standard.

## 5. ⭐⭐ ONE CONSOLIDATED AMENDMENT — the deliverable

File **AMENDMENT 2** into `PREREG-C3-LADDER.md`, dated, in Amendment 1's format, marking exactly what changed. ⭐ **It must come to the Advisor AS ONE PIECE, carrying the geometry/placement lines with it** — i.e. **§2.5's G-B 3-of-12 argument travels inside this amendment's review**, so the Advisor reviews geometry, placement, byte convention, ceiling semantics and calibration **once**. ⛔ Do not file it as a fragment that assumes §2.5 was already accepted; restate the placement argument's claim in one short paragraph and point at §2.5 for the full text.

Contents: the audited per-rival byte table (audited vs previously-pinned, with the missing buffer named per row) · GDN-2 re-shrunk with its diagnostic column labelled · the re-solved-at-deployed-geometry occupancies (§2) · the ceiling-as-ceiling + values-of-record wording (§3) · the standing constraints-not-solver-outputs rule (§3) · the per-store-bearing-layer calibration design line + its smoke diagnostic (§4) · and a one-paragraph restatement of the placement argument.

## 6. Ownership, stops, acceptance

**Yours:** `chlu/eval/byte_ledger.py` · the rival specs/registry surface · the calibration functions · `PREREG-C3-LADDER.md` · tests. ⛔ **NOT yours:** the exemption's verification logic (**STOP and report** if it must change) · run 3's config · `RUN3-LAUNCH.md` (the Hub's) · the CLU arm's geometry (**ratified**).
⚠ **`BUDGET_IS_INTERIM` stays TRUE and the ladder guard stays live** — flipping it belongs with the Advisor *accepting* this amendment, not with writing it. ⛔ Do not flip it.

**Stops:** an audited rival cannot be shrunk under the ceiling with any admissible knob → **STOP and report**; that is a ceiling question, not an implementation one · the calibration change moves the pilot flag table → **STOP** (it breaks run 3).

**Acceptance (one line):** all six rival rows audited against the official implementations' **allocated** buffers (or explicitly NOT OBTAINED) and pinned by tests; GDN-2 re-shrunk to fit total-including-conv with its paper-convention figure retained as a labelled diagnostic; `dtype_bytes` truthful per row; every shrink knob re-solved **at the deployed geometry** with occupancies reported; calibration per store-bearing layer, placement-independent, with the pilot path provably unchanged and one labelled smoke diagnostic; **AMENDMENT 2 filed as one reviewable piece carrying the placement argument**; ⛔ zero ladder arms trained; `BUDGET_IS_INTERIM` untouched; full suite green with counts against a **named, re-verified HEAD**; branch ref verified from the main repo before the worktree is removed.

**Report:** protocol §5 + flag-provenance + dial declaration + git footprint.
