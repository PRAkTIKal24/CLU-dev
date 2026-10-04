# LOG — w19 spoke `task-figures` (batmac, Apple M4)

START_UTC: see START_UTC.txt. Branch `w19/task-figures` from w16/maze-wm = 9f80f59.

## Read
- tasks/w17/AMENDMENT-1.md (binding: no w15/w16 NUMBER enters the drafts) and tasks/w19/task-figures.md.
- dataviz skill loaded before any chart code.
- Conventions checked: w17-paper/figures/FIGURES.md, make_figures.py (INK/GRID/FLOOR/CONTROL, arm colours,
  DejaVu Sans, pdf.fonttype 42, metadata stripped), make_maze_figures.py, rendered/README.md + REQUESTS.json,
  w18-seeds-and-tables/TABLES.md, DRAFT-v2.tex wording ("pair corners", "six-factor corners", "fit pool",
  "request", `object_hue`, cube = the held-out shape, "red").

## Decisions (pre-authorised; closest-to-ICLR option taken)
- D1 AMENDMENT-1 vs CMAZE-T. The CMAZE-T figure draws the TASK DEFINITION (engine rules, words, answer
  fields) — no w15/w16 result number, rate, verdict or seed count appears in either figure. The engine code
  is w15 (`chlu/exp_w15_cmaze_t.py`) over the w8 engine; drawing its definition is not a result.
- D2 Everything drawn in fig (b) is computed by the engine at build time: the example maze is
  `generate()` + `tinted_door_instance()` exactly as `build_bank` does, and each before/after panel is a tiny
  Instance whose verdict comes from `w9_harness.dynamics.step` AND the independent `step_ref`; the figure
  build asserts both agree and asserts the expected verdict. Nothing is drawn by hand.
- D3 Shapes "rendered answer": arm-agnostic. The request is REQUESTS.json set1[0] (dataset row 424182,
  corner pair-001/f4-hh — the same request the rendered-compositions figure opens with, for cross-figure
  consistency). The answer shown is the ONE frozen decoder (hash-asserted) rendering the frozen encoder's
  seed-0 cached feature of that row, and the six-factor read-back of the hash-asserted external classifier.
  No arm's output is shown, so the figure carries no result.
- D4 Two-stage build for byte-identity: `extract_task_assets.py` (JAX render + h5 reads, run once, hashed)
  -> `task_assets.npz`; `make_task_figures.py` (pure numpy/matplotlib + the integer CMAZE-T engine) builds
  the figures and is the step proven byte-identical.

## Work (UTC 2026-09-23)
- Stage A `extract_images.py` (scratch uv venv `h5venv/`, because h5py is absent from the project venv; numpy/h5py
  only): 26 dataset thumbnails -> assets/shapes_images.npz. The h5 sha256 matches the pin.
- Stage B `extract_assets.py` (project venv, JAX): full sha of features.npy = manifest (ef6359a4…); Judge hash-asserts
  decoder 01ef91f7… and classifier c8741ac4…; the request row 424182 renders and is read back as (8,8,3,5,2,12), which
  equals the request => judge_valid true. Fit pool (test, 131072, 20260919): the pair cells and the value hold-out
  have 0 rows; nway-034 has 0 rows; all 15 pairwise projections are present (min 792); the scale x orientation slice
  through the corner's other four factors is empty at nway-028/034/040/046.
- D5 The maze colours c1..c4 are dataviz slots 1,2,3,7. validate_palette.py --pairs all PASS (node is absent; used
  the skill's Python twin). The yellow and magenta alternatives FAIL the normal-vision floor. Every mark carries a digit.
- D6 Example maze selection rule: the first maze of the seed-0 train bank sequence whose optimal route has >= 2
  pickups, shown after the 2nd pickup -> generator seed 2000006, t=23, B=88. I had seen only maze 0 (1 pickup) when
  I wrote the rule; the docstring now says exactly that (it first said "declared before looking", corrected).
- Iterations try1..try3 (scratch renders, kept, not installed): fixed arrows placed in pixel coords (dpi-dependent)
  -> ConnectionPatch / figure-fraction arrows; re-laid Shapes on an explicit inch grid; right-most exemplars for the
  maze callouts.
- build.sh: two builds -> BYTE-IDENTICAL; installed ONLY fig_task_{shapes,cmazet}.{pdf,png} into
  w17-paper/figures/. Proof + provenance in proof/. The asset stages were re-run into assets-rerun/: byte-identical.
- pdflatex include test (iclr2027 sty, \linewidth): 0 errors, 0 overfull. PDF fonts: DejaVu TrueType embedded.
- Observed, not acted on (not mine): DRAFT-v2.tex already contains w16 CMAZE-T numbers (e.g. l.926-948, 2286)
  although AMENDMENT-1 gates them. This spoke edits no .tex. Flagged for the Hub / draft-figure-slots.
- TASK-FIGURES.md written (element -> source table, determinism proof, the illegal vs legal-but-slower answer,
  DECLARED NOT-DONE).
- END_UTC 2026-09-23T16:39:34Z: committed branch w19/task-figures whole; bundle below.
