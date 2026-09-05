# figure-layout-fix — results-analyst (the three figures that cannot be legible at their current footprint)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 43; the §8 flag of `outputs/figure-render-pass.md`, 2026-08-21).** Read `.claude/AGENT_PROTOCOL.md`, then this file. One report: `.claude/outputs/figure-layout-fix.md`, plus regenerated PNGs. ⛔ `papers/v2-short/**` and `papers/v5-short/**` untouched.

**DIAL DECLARATION: none — figure regeneration from banked artifacts; zero new measurements; plotted values provably unchanged (reuse the previous pass's data-tap comparator).**

## Why
The render pass reported that **V5 Fig 1, V5 Fig 2 and V5 Fig C.2 print 1.115 / 1.140 / 0.973 inches tall**. At that height the type target was unreachable: legends were deleted and two panels lost their in-figure numeric labels. A one-inch-tall multi-panel row is not a figure a reviewer can read. The render pass correctly stopped there and called it a layout question. This task answers it.

**The cause is aspect ratio, not width:** these are wide multi-panel rows, so constraining width to the text block leaves ~1 inch of height. The fix is layout, not shrinking.

## The rule to apply
1. **Appendix figures (V5 Fig 2, V5 Fig C.2): re-lay-out taller — this is free.** Appendices are excluded from the page limit at both venues, so height costs nothing. Re-arrange panels (row → grid or stack) to reach a height where the type targets of the previous pass are met (ticks ≥ 7 pt, labels/legend ≥ 8 pt, titles ≥ 9 pt effective) **with legends restored and every in-figure numeric label back**.
2. **Main-text figure (V5 Fig 1, the damping optimum, in a 4.00 pp build against a hard limit): do NOT make it taller. Reduce it instead** — main text carries the single money panel (the collapse), square-ish, fully legible, legend intact; the complete multi-panel version moves to the appendix at full size under a new figure number. Report the main-text figure's exact printed box so the writer can confirm the page split is unchanged.
3. If any panel's message cannot survive being separated from its siblings, say so and leave that figure for the Advisor rather than splitting it badly.

## Also
- Carry forward the previous pass's label hygiene: no prereg item names, internal result IDs or seed short-tags in any panel; seeds neutral-labelled with the mapping stated.
- Check whether **V2's three figures** have the same defect at their printed boxes (the render pass measured V2's as adequate; re-confirm at the new de-scoped build's widths, `papers/v2-neurreps-descoped/`) and fix any that do not meet the targets, under the same rule.
- **Install the results in all folders that use them** (`papers/palm-variant/v5/figs/`, `papers/neurreps-variants/v2/figs/`, `papers/v2-neurreps-descoped/figs/`), rebuild each affected variant, and report every page split against its current value. ⚠ **V5's PALM variant must stay 4.00 pp main / 9 pp total** — if the reduced main figure changes it, report and stop rather than adjusting prose.
- **List every caption edit owed** (yours plus the nine already listed in `outputs/figure-render-pass.md` §7, consolidated into one list keyed by file and figure number). ⛔ Make none — a caption-only writer pass owns them.

## Acceptance criteria
1. The three cramped figures meet the type targets with legends and numeric labels restored; values verified unchanged by the data-tap comparator.
2. The main-text/appendix split of V5 Fig 1 is executed as specified, with printed boxes reported.
3. All affected variants rebuilt; page splits reported; V5's 4.00/9 unchanged or the change flagged and unfixed.
4. One consolidated caption-edit list; no caption edited.
