# cl-encoder — the CL-capable read-in φ (the entry's single blocking dependency)

**Agent:** experiment-engineer. **Worktree MANDATORY** (§3.2). **Co-headline of w26**
(addendum-2 §B3.2). Base local `main` (post-w25).

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** none directly — this is an **enabler** for R4 (the CL entry). Declared explicitly so
  no one mistakes an encoder improvement for a store result.
- **Laundering control:** kNN-in-φ over the **same new features**, at matched memory — mandatory
  on every number. ⚠ **Read this twice: a better φ raises the launder by construction.** If the
  new encoder lifts CLU from 0.149 to 0.40 and lifts kNN-in-φ from 0.21 to 0.45, the entry is
  *still laundered* and the wave has bought scope, not a win. Say that plainly if it happens.
- **Falsifies:** kNN-in-φ over the new features fails to clear the gate below ⇒ the encoder is
  not the fix and the CIFAR null has a second cause we have not found.
- **Does NOT falsify:** the store still losing to iCaRL/replay (CM-23(n)); the launder still
  firing on ACC (w25 already relocated the honest claim to the forgetting axis — that is
  `matched-bytes-frontier`'s job, not yours).

## Why (w25 evidence, `.claude/outputs/cl-entry-build.md` §5 — read it)
Split-CIFAR-10 is a **null**: CLU 0.149 ± 0.013, below LwF 0.162. Three independent lines say the
failure is the **feature space**, not the stream discipline: (1) strict-φ cost is +0.001 —
strictness does not bite; (2) **kNN over the same PCA-32 features caps at 0.21**, so no store
built on those addresses can be competitive; (3) the store's own retrieval on those addresses is
near-perfect (1.000 at p=0.5). *The memory works; what it holds is uninformative about the label.*

## What to build

1. **Cheap first (do this before anything else):** the existing **`ae` arm at `phi_dim ∈ {32,64}`**
   on CIFAR-10, task-1-only. It may already be enough; if it is, you have the answer in an hour.
2. **The real answer:** a **small conv encoder** (and/or a light self-supervised objective —
   your call, justify it) fit **on task-1 classes only**, on a pool **disjoint from every stream
   item**, frozen at end of task 1, **never refit**, **never trained through the store**
   (w20 law). `PREREG_CL_PHI.md` is binding: `phi_dim ≥ 16`, `task1_only` = PRIMARY,
   `generic_frozen` = declared upper bound only.
3. Wire it as an **additive `phi_arm` option** — defaults unchanged, w24/w25 callers bit-identical.

## ⭐ The decisive gate (run this before spending on the full entry)
**kNN-in-φ, Class-IL, Split-CIFAR-10 reduced protocol, 200-item matched memory.** Current PCA-32
gives **0.21**. Reference points under the same reduced protocol: LwF 0.162 · GDumb 0.301 ·
ER 0.369 · iCaRL 0.419 · joint upper bound 0.480.
- **Gate: kNN-in-φ ≥ 0.35.** Clear it ⇒ proceed to run the full entry with the new φ (3 seeds,
  the standard mandatory baseline table).
- Miss it ⇒ **stop, report the gate as the finding**, and say what the second cause might be.
  A well-characterised miss is an acceptance, not a failure — do not burn compute on an entry
  run over addresses that cannot separate the classes.

## Report, whatever happens
- The **strict-φ cost** at the new arm (`generic_frozen − task1_only`). w25 measured ≈0 on a
  *broken* φ and correctly said that proves nothing; on a working φ strictness may finally bite.
  This is the number the w24 `phi-stream-discipline` report asked for and never got.
- **`phi_dim` quoted on every number** (binding since w24).
- The geometry: median-NN address spacing, `s`, `σ_q`, and the **corrected packing slack**
  (⛔ never the retracted 1.08). w25 CIFAR was 0.337–0.345.
- If the entry runs: the full mandatory table (tuned ER · iCaRL · GDumb at matched memory ·
  EWC/SI/LwF as the known null) and both laundering lines (same keys + ring buffer).

## File-ownership split (⚠ parallel-safety — 4 engineer worktrees this wave)
**You own:** `chlu/experiments/exp_phi_stream.py` and the φ config surface (`phi_arm`, `phi_dim`).
**You must NOT edit:** `chlu/experiments/cl_baselines.py` or the byte-accounting / budget-sweep
surface of `exp_cl_entry.py` — those belong to **`matched-bytes-frontier`** this wave. If you need
an entry-side change, keep it to a single additive config field and say so prominently in your
report so the Hub can resolve the merge additively. `chlu/core/*` belongs to `placement-landing`
and `r2-excursion-reach`; do not touch it.

## Deliverable
PREREG first (`.claude/outputs/cl-encoder/PREREG.md`) — register the gate value and your expected
kNN-in-φ band **before** running. Report at `.claude/outputs/cl-encoder.md`, standard format,
reconciliation list in the first 10 lines. Full `pytest tests/` green, `ruff` clean, atomic
commits on `agent/experiment-engineer/cl-encoder`. Do not push.
