# AMENDMENT 1 — the Head, via the Hub, 2026-09-24. ⭐ ADDITIVE: one more figure, same consistency goal.

## THE JOB: make the MAZE TASK figure match the panels
`fig_task_cmazet` (**Figure 2**, `fig:task-cmazet`) is the nomenclature/legend figure. ⛔ **Its agent glyph does not match the result panels**, so a reader learns one symbol and then meets another.

### Three changes
1. ⛔ **Remove the time and budget line** from the figure.
2. ⭐ **The agent becomes a CIRCLE.** ⛔ It is currently a **black downward triangle**.
3. ⭐ **The pocket is shown inside that circle, exactly as the panels do it:**
   - **empty pocket → black circle with a WHITE centre**
   - **pocket holding a key → black circle whose centre is THAT KEY'S COLOUR**

### ⭐⭐ The reference implementation already exists — copy its convention
`cmazet_panels.py` → `draw_state()`, whose docstring is the spec:
> *"the agent as a black disc whose centre shows the pocket's colour (white = empty)"*
⭐ **Match it exactly** — same colour source (`FLOOR`/`UNTAKEN`/`USED` from `w8_envscout.maze_colour`, pocket derived from `State(pos, ks, 0).pocket`), so the legend and the panels are the *same* symbol, not merely similar.

### Where the build code lives
⚠ `fig_task_cmazet` is **not** built by your tree — it belongs to `w19/task-figures`.
⭐ **Its bundle is already on this machine: `~/clu/outputs/w19-task-figures/w19-task-figures.bundle` (branch `w19/task-figures = b9c2f2d`).** Fetch it into a worktree and edit there. Existing outputs and provenance: `~/clu/outputs/w19-task-figures/{try3,proof,assets}/`.
⚠ The shapes task figure in that same tree needs `3dshapes.h5` and a separate venv — ⛔ **do not rebuild `fig_task_shapes`**; only the maze one.

## ACCEPTANCE
- `fig_task_cmazet` rebuilt with a **circular agent** whose centre shows the pocket colour, **no time/budget line**, and **two builds byte-identical**.
- ⭐ **State in your findings that the legend glyph and the panel glyph are now identical**, and how you verified it.
- Installed into **both** `w17-paper/figures/` and `paper-visuals/`; `PROVENANCE.json`, `FIGURES.md` and the `paper-visuals/README.md` entry updated.
- Its caption updated if it describes the old glyph.
- ⛔ If rebuilding proves to need an artefact you do not have, **write `NEEDS-ARTEFACTS.json` and still deliver everything else** — ⛔ do not hand-edit the PDF.
