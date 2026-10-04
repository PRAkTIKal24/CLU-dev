# w22 `spoke/figures-redraw` PASS 3 — FINDINGS (batmac, 2026-09-25)

**Zero scored runs. Nothing written to any ledger. No PREREG.** Every value drawn is read from an artefact that already existed. Every count below was measured by a script whose output is in this folder.

## ⭐⭐ Acceptance test: seed 2, item 482. **The CLU and MLP panels are now visibly different.**
The item is verified in the per-item re-read `data/cmazet_items.json`: `rule_a`, pocket 3, door 1, tint none, east. Truth is `[0,0,0,3]` (illegal). This item is row 3 of `fig_cmazet_families` (Figure 10 in the current DRAFT-v2; the Head saw it as Figure 14). The row selection is recorded in `PROVENANCE.json` as `selection[2] = {seed 2, i 482}`.

| arm | emitted | mark | panel state (unchanged) | **strip under the panel (new)** |
|---|---|---|---|---|
| CLU | `[1,0,0,3]` | ✗ `legality` | agent in place, pocket 3 | grey **`legal`**, then `·` (stay), then pocket 3 |
| rival: mlp | `[0,0,0,3]` | ✓ `ok` | the same | bold black **`illegal`**, then pocket 3 |

I checked this on the compiled page at print scale, not only in the PNG: DRAFT-v2 page 46, rendered at 220 dpi (`pages/p46-rule_a-crop2.png`). The drawn state and the disc are untouched. The pocket was not re-rendered, and it is the same pocket 3 in both panels.

## Same-picture / opposite-mark pairs: **0 after the fix, in every scope**
`w21_figures/cmazet_ambiguity.py` produces `AMBIGUITY.json` (installed as `figures/data/CMAZET-AMBIGUITY.json`). A pair is two arms on the same held-out transition. "Picture" means the rendered `draw_state` pixels. "After" means those pixels plus the tuple the strip displays.

| items | arms | opposite-mark pairs | same picture, before | if only legality were shown | Head's rule (differ only in the legal bit) | **after** |
|---|---|---|---|---|---|---|
| A1 (w19), seeds 0–2 | all 16 | 92,628 | **33,412** | 18,035 | **3,740** | **0** |
| A1 (w19), seeds 0–2 | 13 drawn | 63,624 | 22,340 | 12,624 | 1,261 | **0** |
| CLU-REL-A10, seeds 0–5 | all 22 | 373,583 | 154,671 | 107,148 | 19,915 | **0** |
| CLU-REL-A10, seeds 0–5 | 19 drawn | 295,294 | 124,450 | 88,384 | 13,115 | **0** |

⚠ **The brief's premise was only partly right, so here is the rest of it.** The Head's 3,740 reproduces exactly: 92,628 opposite-mark pairs, of which 3,740 have the same picture and emissions that differ only in the legal bit. But the full same-picture/opposite-mark count is **33,412**, not 3,740. Making legality visible alone would still leave **18,035**:
- **16,283** differ in the emitted pocket-next.
- **1,752** differ in an incoherent key event.

Both of those facts are invisible in the state. So "legality visible ⇒ 0" holds only for the Head's 3,740 class. The strip therefore prints every fact the judge scores:
- **line 1:** legal / illegal.
- **line 2:** move, key event, and pocket after the move.

For an illegal emission only the pocket is printed, because `judge_transition` scores nothing else then. With the strip, the count is 0 by construction (the verdict is a function of what is displayed), and it was measured at 0.
- This does not re-render the pocket. The disc still shows the state's pocket. The *emitted* pocket-next is a separate glyph in the strip, and on 16,283 pairs it is the only difference.
- The earlier Hub diagnosis blamed the undrawn pocket-next field **for the Head's test case**. For that case it is wrong, as the Head said: pocket 3 is emitted and drawn by both arms. As a general statement, the pocket-next field *is* invisible on 16,283 other pairs, so it is drawn now too.

Applied to **every** figure that shares `draw_state`/`grid`: `fig_cmazet_families`, `_per_arm`, `_door_then_tint`, `_deletion`, and the new `_rel_per_arm`. The captions now say that the tick is the judge's verdict on all four facts and that the picture shows only the state.

## B. REL per-arm panels. **Gate 2 PASSED.** Drawn.
- The two bundles were fetched as `w21/rel-newseeds @ 31eeadd` and `w21/maze-t-clu-improve @ f491a04`.
- The vendored `w21_clu.py` and `w19_symmetric.py` are SHA-equal to the versions at e7fdb9b/f491a04.
- The engine files are SHA-equal between the branches and this worktree.
- `w21_figures/cmazet_extract_rel.py` re-read the staged objects for **seeds 0–5**. Result: **132/132 arm scores float `==` the scored run JSONs** (`ROWS.nway`), 0 arms missing. **144 objects** were hash-checked before loading (file SHA = MANIFEST = `.sha256`, and byte count) and after (param hash). CLU per seed: 0.9080 / 0.8935 / 0.9840 / 0.9274 / 0.9004 / 0.9394. These equal the pass-2 values.
- **NEW `fig_cmazet_rel_per_arm`** (`fig:cmazet-rel-per-arm`, Figure 12): 19 arms, including the six `+rel` diagnostics, each with one success and one failure at seed 0. Every arm has both.
- Seeds 0–2 of REL use the same TEST rows as A1: the bank and the items are identical, and this is asserted.
- The REL deletion objects were **not** re-read, because none is drawn.

## AMENDMENT-1: `fig_task_cmazet`. **The legend glyph and the panel glyph are now identical.**
- Every maze cell in the task figure is drawn by `cmazet_panels.draw_state` itself, through `draw_grid`, the renderer the result panels use. The agent is a black disc whose centre is the pocket colour (white when empty), and the pocket comes from `State(pos, ks, 0).pocket`. **The time/budget line is removed.** Digits are overlaid as vector text. The leader-label order is chosen for fewest crossings, which is **0**.
- **How it was verified** (`w19_task_figures/verify_glyph.py`, which is a build gate; output `GLYPH-IDENTITY.json`, installed):
  - For every agent the task figure shows (pocket c3 in (a); c1 in (b)–(d); empty in (d) after), I compared the 52 disc pixels with the disc pixels of a real result-panel raster. For c3 that raster is the **seed 2, item 482 CLU panel**, and the whole cell is equal as well. **7/7 are equal.**
  - Ring `0c0c0c` and centres `cc302b`, `e0a800` and `ffffff` appear as the same bytes in the embedded image palettes of both `fig_task_cmazet.pdf` and `fig_cmazet_families.pdf`.
- ⚠ **Consistency changes this required, beyond the three the amendment named:**
  1. The task figure's c1–c4 were blue/orange/aqua/violet, while every result panel uses red/blue/amber/purple (c1 = blue there, red here). They now use the panels' `MAZE_RGB`.
  2. Keys are drawn as diamonds and the goal as a cross, because that is what `draw_state` draws. A disc-shaped key next to a disc-shaped agent would read as the agent.
- ⚠ **The panels' palette FAILED the dataviz validator.** This predates me (since pass 1). Blue↔purple had normal-vision ΔE 13.7, below the hard floor of 15, and CVD 7.1. **c4 was re-stepped** from (0.48,0.25,0.63) to (0.61,0.25,0.63). The set now passes all pairs: CVD 8.2, normal 17.8 (`PALETTE-MAZE_RGB{-before,}.txt`). Amber stays below 3:1 on white. That is a WARN, relieved by the digits in the task figure and the colour numbers in each panel's row label.
- `fig_task_shapes` was **not** rebuilt, as the amendment instructed.

## C. The 32,768-row ladder slot
Untouched. `data/ladder_rival_sets.json` is byte-equal to HEAD, and set 3 is `not_landed`. I did not wait for that run.

## Acceptance
- **Two builds byte-identical:** `SHA256-w22-build1.txt` = `build2` (32 entries; now including `fig_task_cmazet`, `fig_cmazet_rel_per_arm` and `GLYPH-IDENTITY.json`). The build is installed into **both** `w17-paper/figures/` and `paper-visuals/` and re-hashed at each destination: 0 failures, and the manifest equals the staged build.
- Relative to the live files before this pass, only these changed: the five `fig_cmazet_*` stems, `fig_task_cmazet`, `PROVENANCE.json`, `FIGURES.md`, the README and the new data/code files. Every other figure is byte-identical.
- **Drafts:** edits were made by `tools/apply_figs3.py`: 5 captions edited, the CMAZE-T SRCBLOCK line extended, 1 slot added. I checked the lock before editing and again before each install; no peer session was live. The drafts were `cmp`-equal to snapshot `snap-235858` before install, and the install is byte-identical.
- **Compile, built in `w17-paper/`:** both drafts exit 0 with 0 undefined citations, 0 undefined references, 0 errors, 0 overfull, 0 "Float too large", 0 "Figure pending" and 55 bibitems. **49 / 50 pp** (baseline 48 / 49).
- **All four style diffs EMPTY** (`STYLE-DIFF-CHECK-figs3.md`). **`\PLACEHOLDER` 4 = 4**, with ordered arguments identical (asserted). ⚠ Pass 2 reported 15. The drafts have been changed by others since then, and 4 is the count in the snapshot I started from.
- **Provenance checker:** findings are identical to the baseline. All 4 new caption number tokens are found in their cited artefacts: `22,340` in `figures/data/CMAZET-AMBIGUITY.json`, `132` in `cmazet_items_rel.REPRO.json`, and `8,000`/`80,000` in the REL run JSON.
- `PROVENANCE.json` (the `fig_task_cmazet` entry, with the glyph check merged in; REL/strip notes), `FIGURES.md` (pass-3 section, both destinations) and `paper-visuals/README.md` were updated. In the README, the rows for the changed figures were regenerated from the `.aux` and the drafts, and the new figure and files were added.

## ⚠ Things a reader must know
1. **The Head's defect file `evidence/outputs/FIGURE-DEFECT-cmazet-legality.md` does not exist.** I searched `~/clu` to depth 6. I worked from the worked example in the brief, and every number from it was re-derived from `cmazet_items.json`.
2. **The current draft no longer places** `fig_headline_steps` or the three `rendered/rc_w19_t7_*` figures, and `fig_maze_requestlayer` was already unplaced. Another session made this change. I did not touch it, and the README records it in a dated note. Other README rows were not regenerated, so their "where used" column may be stale.
3. In the figures, the families figure is now Figure 10 and the task figure is Figure 2.

## Did NOT run / not done
- No scored run, no ledger write, and no PREREG.
- REL deletion objects were not re-read, and no REL deletion panel was drawn.
- `fig_task_shapes` was not rebuilt (it needs `3dshapes.h5` and a separate venv).
- The 32,768-row ladder rival set is still running. Only its slot exists.
- The README's completeness counts and the rows for figures I did not change were not regenerated.
- The Node palette validator from the dataviz skill could not run (no `node` on batmac). The repo's Python port, `validate_palette.py`, which implements the same checks, was used instead.
