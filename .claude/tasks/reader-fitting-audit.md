# reader-fitting-audit — does a least-squares reader zero out informative latents? (C2W7 reconciliation 1)

**Campaign 2, wave C2W7 (follow-up). Agent:** experiment-engineer. **Worktree: wt2** (free — C2W7's
was removed; wt1 is C2W6's). Branch `reader-fitting-audit`. Writes
`.claude/outputs/reader-fitting-audit.md` + artifacts. **Small and bounded: ~1–2 h of compute on
already-banked cells. This is a RE-MEASUREMENT of published numbers, not new science.**

## Why this exists (the finding that triggered it)

`c2w7-read-cardinality` §4 measured, on the **same latent**, 5 seeds, unseen:

| reader | params | fitting | unseen exact-set |
|---|---|---|---|
| `count_identity` | **0** | none (identity) | **0.0539 ± 0.0207** |
| `count_table` | 72 | **least squares** | **0.0000 ± 0.0000** |

Mechanism the engineer measured: the reader class is fitted by **least squares** while the metric is
a **thresholded** exact-set accuracy. LSTSQ shrinks `diag(W)` to ≈0.40, so queries whose set is
*exactly right* land at residual 0.537 against `tol = 0.234` and score **zero**. A 2-parameter
gain+bias reader is shrunk just as hard. ⛔ **The fitted reader destroys a signal that a
zero-parameter reader recovers.**

⚠ **The exposure is program-wide and backwards-facing.** Nearly every tier-ii "no arm clears" number
was produced through a fitted reader — above all `orgdiv-null-arms`' **`null*` = 0.00117** (584
configs × 5 seeds, the computed grid-max) and its N1 = "1.0000 train / 0.0000 held-out", plus
`orgdiv-cat-test`'s arm zeros and `tierii-read-fix`'s. **If those zeros are a fitting artifact, the
C2W5 read-protocol refutation's attribution changes.** If they survive, the C2W7 finding is scoped to
its own wave and everything upstream stands.

**This task decides which — and nothing else.** ⛔ It is not a revival of any arm, not a new read
protocol, and not a tier-ii claim. It re-scores banked cells through an additional, admissible
zero-parameter reader.

## Pre-registration obligation (BEFORE any re-scoring runs)

File `PREREG.md` in your output dir with, at minimum:
1. **The reading, committed in advance**, in this form:
   - **SURVIVES** ⇒ the identity reader leaves every re-scored arm below its own registered bar
     (`chance + 0.05`) ⇒ the C2W7 finding is **scoped to the multiplicity read**; C2W5's conclusions
     stand unchanged and the wave's caveat is lifted.
   - **MOVES** ⇒ any re-scored arm's identity-reader score **clears its registered bar by 2 SE** ⇒
     the affected published zeros are **superseded**, an erratum is owed, and the C2W5 attribution
     is re-opened at the Advisor's review.
   - **PARTIAL** ⇒ scores rise materially (state your threshold numerically) without clearing a bar
     ⇒ the zeros stand as verdicts but the *instrument* is impeached; the doctrine question below
     goes to the Advisor.
2. **Your own numeric predictions** per re-scored arm (the w14 rule), filed before you look.
3. The exact arm/config list you will re-score and the seeds (must match the published cells
   **bit-for-bit** — same seeds, same φ, same launch keys; a re-scored cell that does not reproduce
   its *published* fitted number to the last digit is a **harness discrepancy you must report and
   stop on**, not silently absorb).

## Scope — what to re-score (in this order; stop-and-report if step 1 fails)

1. ⭐ **The reproduction gate FIRST.** Re-run one published cell end-to-end with the **existing**
   fitted readers and confirm it reproduces its published number **exactly** (`orgdiv-null-arms` N1
   at its selected config is the cleanest: published 0.0000 unseen / 1.0000 SEEN). ⛔ **If it does
   not reproduce, STOP and report** — everything downstream is void until that is explained.
2. **`orgdiv-null-arms`** — the decision-critical one. Add the zero-parameter identity reader to the
   scored class and re-score: **N1 at its selected config** (the "memorises perfectly, composes not
   at all" arm), plus **the `null*` argmax config (N5 `lr 3e-3, h=64, η=0.9, α=0.01, gate=none,
   chunk=1`)** and each arm's own grid-max config. ⛔ **The full 584-config grid is NOT required** —
   re-scoring the 5 argmax configs + N1 answers the question; if any of them moves, the full grid
   re-score becomes a separate funded task, not a silent expansion of this one.
3. **`tierii-read-fix`** (iteration 1) — its physics arm + its live launder, same treatment. This is
   the cell whose `OD_min = −0.0008 ± 0.0016` was called a "vacuous tie"; a vacuous tie produced by a
   shrinking reader is a different object from one produced by an inert store.
4. **`orgdiv-cat-test`**'s physics arm if and only if 2–3 are cheap and clean; otherwise declare it
   NOT-RUN with its reason.

## Constraints
- **The identity reader must be admissible on the published terms:** 0 parameters, inside the
  reader-class capacity bound (`< N_a·m = 256`), fitted on nothing, applied identically to **every**
  arm including launders and nulls. ⛔ **It is added to the class, never substituted for it** — every
  table reports fitted AND identity columns side by side, so nothing is quietly re-based.
- ⛔ **No selection on `Q_unseen` anywhere** (the wave-invalidating condition, `orgdiv-null-arms` §6).
- ⛔ **Do not re-tune, re-train or re-initialise any arm.** Same seeds, same weights, same latents;
  the ONLY change is which decoder reads them.
- Multi-seed (5) on every number. Declared NOT-RUNs never reported as nulls.
- ⛔ **Do not edit `PREREG-TierII.md`** (a revised pre-registration stops being one). Errata go in
  `ERRATA-TierII.md` as a dated block, per the standing convention.
- **File ownership:** `chlu/core/null_arms.py` + `chlu/experiments/exp_null_arms.py` (additive reader
  only) · `chlu/core/multiwell_read.py` / `multiplicity_read.py` (additive, read-only use preferred)
  · your tests · `chlu/cli/experiment_cmd.py` (additive). ⛔ **NOT yours:** `train_cluformer.py`,
  `blocks.py`, `scripts/csf3/` (C2W6/CSF3), `psi_readout.py`.
- **Git:** branch + scoped worktree off local `main` (expect `fdab86d` or later); verify commits from
  the main repo; never push `origin`; `clu-dev` only.

## The doctrine question you must answer with evidence (not opinion)
> **Should the shipped reader class carry a zero-parameter member everywhere, permanently?**

Answer it with the measurement, not a preference: report what the identity reader does on cells where
the fitted readers were *right* (does it ever score *worse*? at what cost?), and state the failure
mode it introduces (an identity reader assumes the latent is already in the target's units/scale —
say where that assumption is false). ⛔ **The Advisor rules; you supply the evidence and a
recommendation.**

## Acceptance
Reproduction gate passed (or reported and stopped) · PREREG filed before any re-score · fitted vs
identity columns side by side for every re-scored arm × seed · the mechanism confirmed or refuted on
at least one cell (the `diag(W)` shrinkage and the `tol` crossing are the named suspects — measure
them, don't assume them) · the SURVIVES/MOVES/PARTIAL verdict stated mechanically against the
registered reading · a dated `ERRATA-TierII.md` block **iff** the verdict is MOVES or PARTIAL ·
tests green · report → Hub, spawn nothing.
