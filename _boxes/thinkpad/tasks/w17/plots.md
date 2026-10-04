# w17 spoke/plots — THE FIGURES
**Read `COMMON.md` first.** Slug `plots`. Agent type: results-analyst.
⭐⭐ **LOAD THE `dataviz` SKILL FIRST, before writing a single line of chart code.**

## The one job
Professional figures **from artefacts only**, into `~/clu/outputs/w17-paper/figures/`.

## The figures
1. **Crossover:** composition AND the honest four-part bytes vs K, **3 seeds**, seed band. Convention named on the figure.
2. **Ladder:** score vs n per arm with seed band; **second panel** = margin vs best rival.
3. **Deletion:** post-deletion composition + **adaptation cost** per arm.
4. **Encoder-adapt margins.**
5. **Maze figures:** placeholders drawn from w14/w9 data until wave 16 lands — ⭐ mark them clearly as placeholder so wave 18 regenerates them.

## Requirements
- **To-the-point titles.** ⛔ No in-plot captions or explanatory text.
- **Consistent arm names across EVERY figure** — one arm has one name and one colour everywhere.
- **Axis labels with units.**
- **Colour-blind-safe consistent palette.**
- **Vector PDF + PNG** for each.
- ⭐⭐ **Byte-identical on rebuild** — set every seed, sort every input, pin the matplotlib backend and font. **Prove it:** build twice, `shasum` both, show the match in `FINDINGS.md`.
- Every figure's data provenance recorded: figure → artefact path.

## Acceptance criterion
Every figure above as PDF+PNG, a `FIGURES.md` mapping each to its artefact, the rebuild-determinism proof, and a statement of which are placeholders awaiting wave 16.

## Files you own
`figures/` (⛔ except `figures/rendered/`, which belongs to rendered-compositions).

## ⭐⭐ AMENDMENT 1 (Head, 2026-09-22)
⛔ **No w15 or w16 data in any figure.** Every figure that would need a w15/w16 artefact is built from **w14-and-earlier** data and clearly marked a **placeholder** for wave 18 to regenerate. That includes the crossover (use the w12 seeds you have, not w15 `crossover-seeds`), the ladder, deletion, and every maze figure. State the placeholder list in `FIGURES.md`.
