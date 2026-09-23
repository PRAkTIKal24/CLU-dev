# TASK-FIGURES.md — w19 spoke `task-figures`

Two **task-definition** figures, to be placed **before any result** in the draft.
Installed in `~/clu/outputs/w17-paper/figures/`: `fig_task_shapes.{pdf,png}`, `fig_task_cmazet.{pdf,png}`. Nothing else was written to that directory.
Code: branch `w19/task-figures`, `w19_task_figures/` (`extract_images.py`, `extract_assets.py`, `make_task_figures.py`, `build.sh`).
Machine-readable provenance: `~/clu/outputs/w19-task-figures/proof/PROVENANCE-task-figures.json`; the asset record is `assets/ASSETS.json`.

⛔ **AMENDMENT 1 check.** Neither figure has a result in it. No score, rate, validity, margin, seed count or verdict of any arm from any wave appears. The CMAZE-T engine and the request layer are drawn as **definitions**. The only numbers are factor values, the grid values of the hold-out axes, and the example maze's step counter and budget (t = 23, B = 88). These are properties of one generated instance, and none of them is a measurement.

---

## Figure 1: `fig_task_shapes`: Shapes3D, the task

| panel | element | what is drawn | source (artefact / engine) |
|---|---|---|---|
| (a) Six factors | 6 rows × up to 5 thumbnails | the **real dataset images**. Each row changes one factor and holds the other five at the request's values. The value label above each image is read from the image's own `labels` row and asserted equal to the grid value | `3dshapes.h5` (sha256 `0a0f6ed9…` = the runbook pin `chlu.data.shapes3d.H5_SHA256`). Rows are listed in the provenance. Factor order and names come from `chlu/data/shapes3d.py:FACTOR_NAMES`. Shape names: 0 cube, 1 cylinder, 2 sphere, 3 capsule |
| (a) | factor labels | `floor hue, wall hue, object hue, scale, shape, orientation` | display forms of `FACTOR_NAMES`, the same words as the draft (`object_hue` → "object hue") |
| (b) Request → rendered answer | **request** | dataset row **424182**: floor hue 0.8, wall hue 0.8, object hue 0.3, scale 1.11, shape sphere, orientation +21°. This is the **same request that opens the rendered-compositions figure** (`rendered/REQUESTS.json` `requests[0]`, corner `pair-001/f4-hh`) | `figures/rendered/REQUESTS.json` (read only) |
| (b) | **answer** | the **one frozen decoder** (`decoder_v1_s0.pkl`, `param_hash 01ef91f7…`, asserted against `winner_hashes.json` by `exp_gen_factor_wells.Judge`) rendering the frozen encoder's **generation-2, seed-0 cached feature** of that row (`features.npy` sha256 `ef6359a4…`, verified in full against `feature_cache.json`). **No arm's output is shown.** This is the answer format, not a result | `stage/outputs/gen-decoder-lift/winner/`, `stage/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/seed_0/` |
| (b) | **judge** | the hash-asserted external classifier (`param_hash c8741ac4…`) reads back all six factors. The check is `exp_gen_factor_wells.judge_valid`: all six must match. The render→classify path is asserted equal to `Judge.read_back` | `stage/outputs/gen-external-classifier/seed_0/` |
| (c) 2-way hold-out | object hue × shape presence grid | grey = the cell has rows in the **scored fit pool**; white = none. The pool is condition `test`, drawn by `exp_gen_factor_wells.fit_pool` with the hub's pins `n_pool` 131,072 and `pool_rng_seed` 20,260,919 | `data/gen/shapes3d/value_holdout/{factor_index.npz, value_cells.json}` |
| (c) | **pair corner (2-way)** (solid outline) | the three `pair-*` cells: (0.0, cylinder), (0.3, sphere), (0.7, capsule). Both values are present elsewhere; only their co-occurrence is missing. The draft's "60 pair corners" are these three cells tiled into sub-corners (`pair_subcorners`), and the tiling removes nothing more from the pool | `value_cells.json` `cells[family=pair]`. The build asserts a pool count of 0 in each |
| (c) | **value hold-out** (hatched) | object hue 0.5 and **cube** are absent from the fit pool altogether (draft: "cube is the shape held out of the split entirely") | `value_cells.json` `config.held_out_{hue,shape}_value`, asserted absent |
| (d) Six-factor hold-out | three presence grids | corner **`nway-034`** is floor & wall hue {0.2, 0.3}, object hue {0.8, 0.9}, sphere, the 3 lowest scales and the 3 highest orientations. **Left and top:** two 2-factor projections over all pool rows. The corner (dashed) is **fully present**. **Bottom:** the same scale × orientation grid restricted to rows whose other four factors lie in the corner. There the corner is **empty**. The slice also shows the three six-factor corners that share it (`nway-028/040/046`) | same `value_cells.json` / fit pool. Asserted: **0** pool rows in the corner. **All 15** pairwise projections of the corner are present, with a minimum of 792 rows per projection cell (provenance). The corner is empty in the slice |
| legend | one legend for (c) and (d) | in fit pool · not in fit pool · pair corner (2-way) · six-factor corner (n-way) · value hold-out | — |

**Why (d) is drawn this way:** a six-factor hold-out is invisible in any two-factor view, and that is the point of it. The figure shows exactly that: present in the marginals, missing only in the joint slice. Showing the hole in a marginal would misstate the split.

## Figure 2: `fig_task_cmazet`: CMAZE-T, the task

Everything is **computed by the engine at build time**, and nothing is drawn by hand. The engine is `w8_envscout.maze_colour` (generate / solve, unmodified), with `w9_harness.dynamics.step` and the CMAZE-T layer `chlu/exp_w15_cmaze_t.py` (`tinted_door_instance`, `words_of`, `outcome_of`, `step_ref`) on base `w16/maze-wm` = 9f80f59.

| panel | element | glyph | source |
|---|---|---|---|
| (a) CMAZE-T grid | the maze | generator seed **2,000,006** of the **seed-0 TRAIN bank sequence**, replayed exactly as `build_bank` does it (base 2,000,000 per `scripts/w16_cmaze_t.py:banks`, rng `base + 5551`, `tinted_door_instance`, the transition and natural-row draws consuming the same stream). **Selection rule:** the first maze of that sequence whose optimal route picks up ≥ 2 keys, shown right after its second pickup. I wrote the rule after seeing only the bank's first maze, never the chosen one | provenance `fig_task_cmazet.a` (route, keystate, receipt) |
| (a) | wall / floor | dark / white cells | `inst.grid` |
| (a) | **tinted floor** | pale fill of colour *c* + a small digit *c* | `inst.tint` |
| (a) | **closed door** | solid square of colour *c*, ink border, white digit (drawn over the cell's tint where the door is tinted: CMAZE-T's one change) | `inst.door`, keystate ≠ USED |
| (a) | **open door** | dashed outline of colour *c*, coloured digit | `inst.door`, keystate = USED |
| (a) | **key** | disc of colour *c* with a digit | `inst.key_at`, keystate UNTAKEN |
| (a) | **agent** | black triangle | state `pos` |
| (a) | **pocket** | small key disc at the agent's shoulder (at most one key) | state `pocket` |
| (a) | **goal** | pentagon with a star | `inst.goal` |
| (a) | **budget** | `t = 23   budget B = 88` | step index; `inst.budget` = opt + ⌈10 %⌉ (`tinted_door_instance`, slack 0.10) |
| (a) | colour key | four discs 1–4, "colours" | `MAZE_COL` (see colour note) |
| (b) legal | before → after, `E` | pocket c1, target floor tinted **c2**, no door ⇒ moves ✓ | a 1×2 `Instance`. The verdict comes from `outcome_of` (engine `step`) **and** from `step_ref`, which is written independently from the rules. The build asserts the two agree and asserts ✓ |
| (c) illegal (rule B) | before → after | pocket c1, target floor tinted **c1**, no door ⇒ **the agent does not move** ✗ | same double check, asserts ✗ |
| (d) legal: key spent at door | before → after | pocket c1, target = **closed door c1 tinted c1**. The door consumes the key first, so the pocket is empty when rule B is read ⇒ moves, the door opens and the pocket becomes empty ✓ | same double check, asserts ✓ |
| (b–d) | **five request words** | `w_agent, w_key, w_door, w_tint, w_dir`, with values from `words_of` | names verbatim from `WORD_NAMES` in `exp_w15_cmaze_t.py` and the w15 harness PREREG/FINDINGS |
| (b–d) | **four-fact answer** | `legal, disp, event, pocket`, with values from `outcome_of` | the code blocks `(legal, disp, event, pocket_next)` of `exp_w15_cmaze_t.py` |

Panels (c) and (d) differ **only in the door**: the same pocket and the same tint colour give opposite verdicts. This is the irreducible pocket × door × tint interaction that the n-way hold-out holds out (family `interaction` vs `rule_b` in `_nway_candidates`).

### "What would prove this wrong": illegal vs legal-but-slower, answered

**Illegal is visibly different, and I did not fudge the panels to get there.** At the transition level, "illegal" is its own answer field (`legal = 0`, `disp = stay`, pocket unchanged). The agent does not move, and a rollout ends there. A legal step, even one on a longer detour, always has `legal = 1` and a real displacement. So the two cannot be confused at the level CMAZE-T is scored. At the route level, the exact judge rejects any route that contains an illegal step. A slower legal route is valid as long as its length is ≤ B. That is why the budget is drawn. The two-gym law is **legible** on this instrument.
⚠ **One caveat, drawn as it is:** after an illegal step the after-panel looks identical to the before-panel. The ✗ and `legal 0` carry the verdict, and colour carries none of it.

## Names and colours: consistency with the rest of the paper

- Checked against `figures/FIGURES.md` (the w17/plots table), `make_figures.py`, `make_maze_figures.py`, `rendered/README.md`, `w18-seeds-and-tables/TABLES.md` and `DRAFT-v2.tex`.
- **No arm appears in either figure,** so no arm colour is used for an arm. The fit-pool greys are neutral (`#d9d8d3`, never an arm or floor grey). Ink, secondary ink and hairline are the w17 values (`#0b0b0b`, `#52514e`, `#e4e3de`). Font is DejaVu Sans, 7.5/8 pt titles, `TEXT_W` 5.5 in, as in w17/plots.
- Words match the draft and the tables: **pair corner (2-way)** and **six-factor corner (n-way)** (draft: "60 pair corners", "49 six-factor corners"), **fit pool**, **request**, **value hold-out**, **cube** as the held-out shape. On CMAZE-T: `w_*` words and the `legal/disp/event/pocket` answer fields, verbatim from the harness.
- ⚠ **Judgement call, flagged: the maze colours.** CMAZE-T needs four key/door/tint colours, and no earlier figure defines them. They are dataviz reference slots 1, 2, 3 and 7 (blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, violet `#4a3aa7`). The dataviz validator (`validate_palette.py --pairs all`) **passes** this four-set on all pairs (worst CVD ΔE 9.2, normal-vision 16.3). The alternatives with yellow or magenta **fail** the normal-vision floor. Aqua is under 3:1 on white, so **every coloured mark carries its digit** (the relief rule), and identity is never carried by colour alone. These hues coincide with the CLU/Learned 1×/MLP/adapter arm hues elsewhere. In this figure they name **maze colours c1–c4 only**, a different kind of object, and no arm is ever drawn in a maze. If the Head prefers zero hue reuse, change the one constant `MAZE_COL` in `make_task_figures.py` and rebuild.

## Rebuild-determinism proof

`bash w19_task_figures/build.sh <assets> <proof>` builds **twice** into fresh scratch dirs, diffs the SHA-256 of every output and installs only on a match. Stack: Python 3.12.14 · matplotlib 3.10.8 · numpy 2.4.1 · backend Agg · DejaVu Sans (TrueType embedded, `pdf.fonttype 42`) · PDF/PNG metadata stripped · `SOURCE_DATE_EPOCH=0`.

```
build 1 = build 2  (proof/SHA256-build1.txt, proof/SHA256-build2.txt)
12ba42d6961b12bc1ebabf26497b0a9fc082493b925ac15c99f9fb62e9ed840a  fig_task_cmazet.pdf
d6965be2d9ba101bf3bf742e527548ea423ea7eae822ac9d1bcb2dc2dd0763b0  fig_task_cmazet.png
c83b24d551ec128efe0c4bd7c851a851b96620e0ccdd1fee34ca69674f4c7954  fig_task_shapes.pdf
78a8d7ed2995dd1bec91af097f004f9c344c1eab0f48a124d3678b8c23aa65fe  fig_task_shapes.png
REBUILD: BYTE-IDENTICAL
```
The installed files in `w17-paper/figures/` hash to the same four values. I also re-ran the **asset stages** (the h5 read, and the JAX decoder render + classifier read-back) into a fresh dir. `shapes_images.npz`, `task_assets.npz` and `ASSETS.json` came out byte-identical (`9f9d71bc…`, `bc6df90e…`, `8653edb5…`), so the whole chain reproduces, not just the plotting step.

Include test: both PDFs compile at `\linewidth` under `iclr2027_conference.sty` (pdflatex) with 0 errors and 0 overfull boxes. Text and marks are vector. The only raster content is the 30 thumbnails (dataset images plus the one decoder render), which are images by nature.

## For `draft-figure-slots` (suggested only; no `.tex` was edited)

```latex
% SOURCE: 3dshapes.h5 (pinned sha); value_holdout/{factor_index.npz,value_cells.json} (fit pool, pair-*, nway-034);
%         rendered/REQUESTS.json requests[0]; frozen decoder seed 0 (01ef91f7...), external classifier (c8741ac4...)
\includegraphics[width=\linewidth]{figures/fig_task_shapes.pdf}
% SOURCE: CMAZE-T engine (w8 maze_colour + w9 dynamics + exp_w15_cmaze_t), generator seed 2000006, seed-0 train bank sequence
\includegraphics[width=\linewidth]{figures/fig_task_cmazet.pdf}
```
Caption facts the figures deliberately leave out of the drawing (no in-figure prose):
- Shapes (d): the corner is absent from the fit pool, yet every two-factor projection of it is present.
- Shapes (b): the answer is the frozen decoder on the encoder's own feature, so it shows the answer format and no arm.
- CMAZE-T: rule A says a door of colour *c* needs key *c* and consumes it. Rule B says floor tinted *c* is forbidden while holding *c*. A is applied before B.

## DECLARED NOT-DONE

- **Seeds 1–2 of the feature cache: not needed.** One request rendered at seed 0 defines the answer format, so no `NEEDS-ARTEFACTS.json` was written.
- **The answer is not shown for an arm (CLU or rival).** That is a result, and it belongs to `success-failure-panels` / `figures/rendered/`. Keeping it out also keeps this figure free of any number AMENDMENT 1 gates.
- **The CMAZE-T n-way hold-out cells are not drawn as a grid.** The task file asks for the grid, the rules panels and the words/answer for CMAZE-T. Hold-out geometry is asked only for Shapes. Panels (c) and (d) are exactly the `rule_b` and `interaction` n-way families.
