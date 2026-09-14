# figure-render-pass — results-analyst (re-render every variant figure for its FINAL printed size; strip internal labels)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 40; Head directive 2026-08-21 — "do the figure render changes at once").** Read `.claude/AGENT_PROTOCOL.md`, then this file. You write regenerated PNGs into the two variant `figs/` directories, plus one report `.claude/outputs/figure-render-pass.md`. ⛔ No prose edits, no `.tex` content edits beyond a width value if one is strictly needed (state it), and ⛔ **`papers/v2-short/**` and `papers/v5-short/**` are NOT touched** — the live builds keep their current figures until the Head says otherwise.

**DIAL DECLARATION: none — figure regeneration from banked artifacts. Zero new measurements; every plotted value identical to the current figure's.**

## Why this exists
Both variants shrank their figures to buy page space, so the figures are now displayed far below the size their fonts were designed for:

| figure | printed at | source design width | effective shrink |
|---|---|---|---|
| V2 variant `fig1_mo_headtohead.png` | `0.68\linewidth` (≈3.7 in) | `0.9\linewidth` (≈4.9 in) | 24 % |
| V2 variant `fig2_anchor_cure_laws.png` | `0.86\linewidth` | `0.9\linewidth` | 4 % |
| V2 variant `fig3_gmor_condensate.png` | `0.60\textwidth` (≈3.3 in) | full-width render | ~40 % |
| V5 variant `fig1_damping_optimum.png` | `0.60\linewidth` (≈3.3 in) | `0.84\linewidth` (4.6 in) | 29 % |
| V5 variant `fig2_two_instruments.png` | `0.74\linewidth` | full-width render | ~26 % |
| V5 variant `figC2_vault_emergent.png` | `0.80\linewidth` | full-width render | ~20 % |

A reviewer reads these at 3.3–4 in wide; tick labels rendered for a 6-in figure are then marginal or illegible.

## The job
For each of the six, locate the generating script/artifact (the banked plotting code or the source report's figure step; `chlu.utils.plotting` where it applies), and **re-render at the final physical size** — set `figsize` to the printed width in inches rather than shrinking a large canvas — with type sizes chosen so that, at the printed size: tick labels ≥ 7 pt effective, axis labels and legend ≥ 8 pt, panel titles ≥ 9 pt. Increase line/marker weights to match.

⛔ **The hard constraint that protects pagination: preserve each figure's printed FOOTPRINT** — same aspect ratio, so at the same `\linewidth` fraction the block occupies the same height. Only type size, line weight, tick density and label text may change. (V5's variant sits at exactly 4.00 pp against a hard venue limit; a taller figure would break it.)

## Label hygiene, folded in (the standing flag from Add.36, and the same class in V2)
- ⛔ **Strip internal labels from panel titles, legends and insets:** pre-registration item names (`Q1`, `Q2`, `Q3`, `Q5`), internal result/task IDs (`T5`, `T6`, `R1`, `R3`, `Cor-13`-class tokens), source-report names. Replace with the reader-facing quantity the panel actually shows.
- **Seed short-tags** (`s42`, `s43`, `s44`): replace with neutral labels (`seed 1/2/3`) and state the mapping in the report so captions stay accurate. ⚠ If a caption in either `submission.tex` names a stripped or renamed label, list it in your report as a caption edit owed — do not edit the caption yourself.
- Sweep every regenerated figure for anything project-identifying (paths, worktree names, usernames) and report the result.

## Verify before you finish
1. **Numbers unchanged:** for each figure, confirm the plotted values against the same artifact the current figure used, and state how (checksum of the data array, or re-derivation from the banked JSON). ⛔ A figure whose values move is a defect, not an improvement.
2. **Rebuild both variants** (`pdflatex` ×3 in each variant directory) and report the page split for each against its current one: **V5 variant must stay 4.00 pp main / 9 pp total; V2 variant must not grow.** If a rebuild moves either, say so and stop rather than adjusting prose.
3. Report a before/after note per figure (what was illegible, what changed), and flag any figure where the legibility target is unreachable without changing the footprint — that one comes back to the Advisor as a layout question, not a render question.

## Acceptance criteria
1. Six figures regenerated at final size with the type targets met, footprints preserved, values verified unchanged.
2. Internal labels and seed tags stripped/neutralized; caption edits owed listed, none made.
3. Both variants rebuilt with page splits reported and unchanged (or the change flagged and unfixed).
4. `papers/v2-short/**` and `papers/v5-short/**` untouched — state the check.
