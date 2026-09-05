# c2w5-close-fixes — the wave-close fix pack (figures + harness + ruler)

**Campaign 2, wave C2W5. Agent:** experiment-engineer. **Small-medium.** Branch
`agent/experiment-engineer/c2w5-close-fixes` off `main` (post-integration main — it now carries all
three C2W5 engineer merges + the Hub's G-1 fix). **No worktree needed** (cap is free; no concurrent
spoke shares your files — the only parallel spoke is the paper-writer in `.claude/papers/`).
Output: `.claude/outputs/c2w5-close-fixes.md`. Five bounded items:

1. ⭐ **Fig 1/2/5 re-render (r4 reconciliation 1 — GATES the referee pass).** Renderer:
   `.claude/scratch/bprime-referee-closures/render_figures.py`; Mamba-2 values:
   `.claude/outputs/bprime-mamba2-arm/run_agg_n9/`. Target spec (r4 App K): **Fig 1 = seven bars**
   (6 rivals + CLU, uniform n = 9) with **three hatch classes** — NOT-RESCUED {`ttt_mlp`, CLU} ·
   INIT-UNSTABLE {`ttt_linear`} · SELECTION-DEPENDENT {`deltanet`, `gdn2`}; **Fig 2** gains the SSD
   bar (+ modal-ledger caps per r4 spec); **Fig 5** caption count → **0-of-20**. Figures land beside
   the existing renders (same paths, PNG + PDF); provenance table update in your report.
2. **The two declared missing aggregations (r4 reconciliation 2 — one re-aggregation of banked
   cells, NO new measurement):** (a) the SSD arm's **paired `full − null`** with SE; (b) its
   **per-reader `+0 B` means** and per-head-width frontier `+0 B` margins. Write the JSONs beside
   the run_agg_n9 artifacts; the r5+ writer folds them later (do not edit any draft).
3. **CLU columns into the harness (r4 editorial 4, ~30 lines + a test):** `audit_table` currently
   emits only `clu_reproduced.{full,launder,dividend}`; the blank / null / `+0 B` / lift columns are
   aggregated by a scratch script while the paper leans on them. Emit them first-class; regression
   test asserts the emitted values equal the published n = 9 numbers (−0.3906 blank / −0.6512 null /
   −0.2897 +0 B / −0.0465 lift, from the closures artifacts) bit-for-bit against the same inputs.
4. ⭐ **The `s`-ruler re-measurement (curator G-5 / N224 / charter §A20.5 — UNOWNED until now, and
   every `d/s` in the program rides on it):** re-measure `bprime-c6`'s rig with the corrected
   effective-`s` estimator (subtract the confinement term `α‖q‖²`; exemplar:
   `tierii-read-fix`'s confinement-subtracted fit, §7.28 datum `s = 0.2879`, `R² = 0.9986`).
   Deliverable: corrected `s` (± fit quality) for that rig, the implied `d/s`, and the verdict —
   `s = 0.40` CONFIRMED or CORRECTED (direction on record: subtraction makes `s` smaller and `d/s`
   LARGER, i.e. further inside the designed-gate regime). ⛔ Report the number; do NOT edit
   N224/registries (curator's next pass folds it).
5. **The psires CLI hook (psi-payload-residual §8, one-liner):** `chlu/cli/experiment_cmd.py` gains
   the `exp-psi-residual` subcommand on the `exp-tierii-read` pattern (that file now carries the
   tierii hook as precedent — no existing line altered).

House rules: main venv reused (no worktree `uv sync` hazard this time — you are on the main
checkout); suite green before handing back (baseline is the post-integration count the Hub gives at
spawn); `ruff` green; atomic tagged commits; report with flag provenance for anything you compute.
