# pw1-plot-shapes — plan items 10, 11, 12 (figure parts) + vocabulary in every other plotted figure
Session `pw1-plot-shapes` · worktree `~/clu/worktrees/CHLU-pw1-plot-shapes` · branch `pw1/plot-shapes` (from `pw1/plot-maze` 2b1791e, which already carries the finished maze-figure changes; do not alter them) · out `~/clu/outputs/pw1-plot-shapes/`
**Added by the Hub:** in `make_figures.py` `main()`, remove the two calls that still dispatch `fig_maze_readverb` and `fig_maze_requestlayer` (≈ lines 543–544, found by pw1/plot-maze). Item 14 retires both figures. Keep the functions; only stop building them. Then run the full `build.sh` into scratch destinations and show that neither stem is produced and every maze figure is cmp-equal to `~/clu/outputs/pw1-plot-maze/build/a/`.
Read `COMMON.md`, then PLAN.md §1 items 2, 3, 10, 11, 12, 13 in full. Load `dataviz` first. ⛔ **You do NOT edit either draft and you do NOT install.**

## The one job
Change the figure pipeline (`w21_figures/` in your worktree) so it draws:
- **Item 10**, `fig_ladder`: a single data-matched panel, with every one of the eight Head/Advisor points (no title; the x label; the legend at the right, vertical; no line-end labels; the legend names; the y label).
- **Item 11**, `fig_crossover`: the five points.
- **Item 12**, `fig_deletion`: keep only former (e), (f), relettered (a), (b), with the new titles and labels, "red sphere", and no "exact" anywhere (item 13).
- **Items 2–3 vocabulary** in every other figure the build draws and you own: `fig_headline_steps`, `fig_encoder_adapt`, and anything in `make_figures.py`. Every legend, axis label, title and in-image text: n-CLU, baseline, held-out combination, capsule, memory footprint, `$10\times$ training steps`, and plain control names.

## File ownership (`pw1/plot-maze` is finished; its files are frozen)
YOURS: `make_figures.py`, `data/ladder_rival_sets.json`, and INSIDE `make_figures_w21.py` / `make_figures_w22.py` only the functions that draw `fig_ladder`, `fig_crossover`, `fig_deletion`, `fig_headline_steps`, `fig_encoder_adapt`. ⛔ Not yours: anything `fig_maze_*` / `fig_cmazet_*`, `cmazet_*.py`, `maze_six.py`, `build.sh`, and shared helpers/palettes/label dictionaries (override locally inside your own functions). Minimal diffs, no reformatting.

## Deliverables in your out dir
- `build/a`, `build/b`: two builds of ONLY your figures, byte-identical, `SHA256-build{1,2}.txt`, `PROVENANCE-shapes.json`.
- `CAPTIONS.tex`: a new caption for each changed figure, keyed by `\label`. Ladder: the pool size and step counts move here, and n ≥ 3 is the same 120 held-out combinations. Crossover per item 11.5. Deletion: two panels. No banned words.
- `DELETION-FACTS.tex`: for item 12.7, the numbers that leave the deletion figure (n-CLU and address book equal the refit-without to 1e-8 on every scope and seed; the refusal display), as table rows plus ONE sentence phrased per item 13 (no "exact"). Every number carries its `% SRC` artefact pointer, verified verbatim against the file.
**What would prove this wrong:** a legend entry that disagrees with a plotted series, a number in a caption not found verbatim in its source, or two builds that differ.
