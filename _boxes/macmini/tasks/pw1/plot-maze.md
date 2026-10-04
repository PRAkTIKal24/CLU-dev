# pw1-plot-maze — plan items 1 and 16 (+ the figure half of item 8), CMAZE-T figures only
Session `pw1-plot-maze` · worktree `~/clu/worktrees/CHLU-pw1-plot-maze` · branch `pw1/plot-maze` (from `w22/figures3`) · out `~/clu/outputs/pw1-plot-maze/`
Read `COMMON.md`, then PLAN.md §1 items 1, 8, 16 in full. Load `dataviz` first. ⛔ **You do NOT edit either draft and you do NOT install into `w17-paper/figures/` or `paper-visuals/`.** A later spoke installs your build.

## The one job
Change the figure pipeline (`w21_figures/` in your worktree) so it draws:
- **Item 1:** `fig_maze_conditions` (label `fig:maze-wm`), exactly per the ten Head-ruled points. Panel (a) rows are words-full · observation · words+observation (n-CLU) · words+observation (n-CLU-REL), with Transformer-REL as the n-CLU-REL comparator. Every seed is drawn with its count printed. There is no label column, no overlaid text in (b)/(c), no ceiling line, and x = seed numbers. Panel (c) holds only n-CLU-REL, Transformer-REL and Highest control. Use the given titles, legends and `$10\times$ training steps` sub-labels, with no overlaps.
- **Item 16:** `fig_cmazet_families`, `fig_cmazet_per_arm`, `fig_cmazet_rel_per_arm`, `fig_cmazet_door_then_tint`. Remove the bottom explanatory paragraph from every image and keep the per-panel fact strips. Add n-CLU door successes BY RULE: search the persisted per-item scores (seeds 0–5, base n-CLU and n-CLU-REL: `data/cmazet_items.json`, `data/cmazet_items_rel.json`, and the persisted objects under `~/clu/stage/rel-persist/` and `~/clu/stage/w19-persist/`) for (i) key-consumed door openings and (ii) door-then-tint interactions where the n-CLU is correct, and separately for items where the n-CLU is correct and ≥ 4 of the 6 matched baselines are wrong. Draw with a fixed-seed draw among each qualifying set, for diversity of families and colours, and print the rule and the seed. If a class is empty, say so. Do not substitute. Apply the names from items 2–3, the column groups "matched baselines, 8,000 steps / 80,000 steps", "controls", "blind arms", and plain control names.
- **Item 8 (figure half):** no hard/easy shading, ticks, cuts or columns in any maze figure.
- **Retire from the build** (do not delete the old files; just stop building them): `fig_maze_readverb`, `fig_maze_requestlayer`, `fig_maze_rel_tuples`, `fig_cmazet_deletion` (items 14, 15, 17).
- The hash gate stays: all arm scores float-equal to the run JSONs, ambiguity check 0.

## File ownership (another spoke, `pw1/plot-shapes`, edits the same pipeline in parallel)
YOURS: `cmazet_panels.py`, `cmazet_extract_rel.py`, `cmazet_ambiguity.py`, `maze_six.py`, `data/cmazet_*.json`, `data/maze_conditions.json`, and INSIDE `make_figures_w21.py` / `make_figures_w22.py` only the functions that draw `fig_maze_*` / `fig_cmazet_*`, plus the calls that dispatch them. ⛔ Do not edit shared helpers, palettes, label dictionaries, `make_figures.py` or `build.sh`: override locally inside your own functions. ⛔ No reformatting of lines you do not need to change. Minimal diffs.

## Deliverables in your out dir
- `build/a/`, `build/b/`: two scratch builds of ONLY your figures, byte-identical, `SHA256-build{1,2}.txt`, `PROVENANCE-maze.json` (sources and hashes).
- `CAPTIONS.tex`: a new caption for each of your five figures, written from scratch per item 1.10 and item 16.3, with no banned words. Each caption is keyed by its `\label`.
- `TEXT-INSERTS.tex`: (a) the ONE paragraph that states what the panel fact strips mean, which item 16.1 moves into the text, and the sentence each panel caption uses to point to it; (b) the item 1.5 / item 8 task-section paragraph on the hold-out mechanism, with no seed labelled and no ceiling number.
- `DOOR-SEARCH.json` + a table in FINDINGS: qualifying counts per class per seed per model, the draw rule and seed, and the items drawn.
- A list of every table cell, caption phrase and paragraph in the drafts that carries the per-seed best-of-all-arms / ceiling row (item 1.5), with line numbers. You list these; a later spoke deletes them.
**What would prove this wrong:** a drawn panel that the hash gate cannot reproduce, a door-success row chosen by eye, or any overlaid claim text left in (b)/(c).
