# ADDENDUM — `hub/refusal`
**Read `LAUNCH-COMMON.md` first.** Then `.claude/tasks/hub-refusal.md` is your task.

- **Slug:** `hub-refusal` · **Branch:** `hub/refusal`
- **Worktree:** `git worktree add /c/Users/PratikJawahar/chlu/CHLU-hub-refusal -b hub/refusal HEAD`
  ⛔ NOT `ae907466` — see COMMON §1a.
- **Core budget: `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4` / `OPENBLAS_NUM_THREADS=4`.**
  You get the smallest share because your work is scoring, not sweeping.

## Why you are the safest of the three
⭐ **You score off traces that ALREADY EXIST.** You should not need to rebuild anything, which means
COMMON §4's hard constraint — never mix Mac-produced and Windows-produced artefacts inside one
comparison — costs you nothing.
⚠ **If you find yourself about to regenerate cell-arm traces or re-encode anything, STOP and report.**
That would put you in exactly the position this hub was chosen to avoid.

## Your inputs — present natively in v2, and verified on arrival
- `.claude/outputs/gen-read-cell-arm/runs/` — energy traces, settled states, decoded `.npz`,
  `cell_results.json`, and three `cell/seed_N/seed_N/cell_manifest.json`
- ⭐ `.claude/outputs/gen-val-split-fix/refusal_scores.json` (2,899,315 B) — **this is the one you score
  against**

Verified when they landed: 89/89 sha256 match, and **78/78 `.npz` opened via `numpy.load`, all arrays
materialised and finite.** You do not need to re-verify them; you may if you want cheap reassurance.

⛔⛔ **THE TRAP, stated again because it is yours specifically:**
`.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` is the **SUPERSEDED broken-split** file, and it
has the **same basename** as the good one. ⛔ Never quote it. **Check the directory, not the filename,
every single time.** It is manifest-pinned — never move or delete it, just never cite it.

## The bar you inherit
Your task file states it; do not re-derive it. **One** landscape-derived criterion — a function of settled
energy and convergence, ⛔ nothing else — refusing both `out_of_range` and `duplicate_factor` while
accepting unseen-valid, at a **single threshold chosen on valid data only**, with **zero invalid
supervision**, beating kNN (0.8457 / 0.9583) and LACE (0.8097) on both kinds over 3 seeds.

⛔ **Any threshold that saw invalid data is not a result.** That is the one way this hub fails silently:
a threshold tuned on the very data it is meant to refuse will look excellent and mean nothing.
