# v2-figure-text-pass — results-analyst — the Head's two figure TODOs: retire "decades" and relabel the head-to-head plot

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 67).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output report: `.claude/outputs/v2-figure-text-pass.md`.

**DIAL DECLARATION: none — a text-on-figure pass. ⛔ ZERO new measurements, ZERO data changes. Every plotted value must be provably identical to what ships today.**

**⛔ NOT GATED — run any time.** (It was previously sequenced behind the cite pass because the Head planned to bake a citation *number* into a figure; the Head has since ruled a descriptive label instead, which removes that dependency entirely.)

## ⛔⛔ THE HARD CONSTRAINT: the Head is editing `pj_sub.tex` RIGHT NOW
**You write NO file inside `.claude/NIPSsubmission/v2-neurreps/` except the PNGs under `figs/`.** ⛔ Not `pj_sub.tex` (a concurrent write would clobber the Head's in-flight edits — the file has changed twice today), ⛔ not `pj_sub.pdf/.aux/.log`. **Build and measure in a scratch copy** (`.claude/scratch/v2-figure-text-pass/build/`), per the standing lesson that a pass working near a live artifact copies it off-tree first. The two `\TODO` tags in the tex are **discharged by your work but removed by someone else** — name them in your report; do not edit them out.

**Back up before you overwrite:** copy all five current PNGs to `.claude/scratch/v2-figure-text-pass/figs-BEFORE/` with an md5 manifest, so any figure is one `cp` from restoration.

## The two changes (and nothing else)
**Change 1 — retire "decade" vocabulary, every figure that uses it.** The Head: *"change 'decades' in figure headline to orders of magnitude. do the same for all figures that use decade."* The word lives **inside the rendered images**, not in the captions (`decade` appears in `pj_sub.tex` only inside the TODO itself). ⇒ **Inspect all five PNGs** — `fig1_gmor.png` · `fig2_anchor_cure_laws.png` · `fig3_gmor_condensate.png` · `fig3_retention_overlay.png` · `fig_lifetime_headtohead.png` — for any rendered "decade"/"decades" in titles, axis labels, annotations or legends, and replace with "orders of magnitude" (or "orders" where the full phrase will not fit without shrinking type below the targets below — report any such case). ⚠ Report the per-figure inventory of hits **including the figures with zero hits**, so the sweep is auditable.

**Change 2 — `fig_lifetime_headtohead.png` only, two relabels (the Head's exact words):**
- **"published" → "single-exponential estimator"**, in **both the plot headline and the label**. ⭐ **Head ruling (2026-08-24): the figure stays consistent with the body text.** Advisor-verified in `pj_sub.tex` — the paper uses *"single-exponential"* at all four of its sites (*"single-exponential estimator"*, *"single-exponential lifetime estimator"* ×2, *"single-exponential estimation"*) and **never** *"single-exponent"*. ⛔ Use **"single-exponential estimator"** exactly. If it does not fit at the type targets, **wrap or rotate — do not abbreviate it**, and report what you did.
- **"5 trained models" → "CLU (5 seeds)"**.
⭐ **Why this wording matters (context, so you do not "improve" it):** the original "published" is factually wrong — the work is an unrefereed preprint — and the author's name is ⛔ **forbidden in any figure content, label or filename** (charter Add.45/51). A descriptive label is the compliant fix. ⛔ **No author name, and no citation number, may appear anywhere in any figure.** The caption cites the work; the image does not.

## Method — the discipline this program's figure work has already established
1. **Re-render from the original generators**, not by image editing. Provenance is banked in `.claude/outputs/figure-render-pass.md` (which established that five of six V2/V5 figures reproduce byte-for-byte from their generators, and named the one with no surviving generator, reconstructed via `sf3_reconstruct.py`). Candidate generators: `.claude/scratch/f1-gmor-condensate/analyze_and_figure.py` · `.claude/scratch/v2-full-runs/make_figures.py` · the render pass's own `orig_v2f1.py` / `new_v2f2.py` / `orig_v2f3.py` / `sf3_reconstruct.py`. ⚠ If a generator cannot be found or will not run, **say so and stop on that figure** — ⛔ never hand-edit pixels and never re-derive a number.
2. ⭐ **Prove the data did not move:** run each generator under a **data tap that hashes every array handed to matplotlib**, and show the hashes are identical before and after your text change. That is the instrument the render pass used; reuse it. **A text change that alters a plotted value is a failed pass.**
3. **Type targets (from the layout fix):** ticks ≥ 7 pt, axis labels/legend ≥ 8 pt, titles ≥ 9 pt **effective at printed size**. "Orders of magnitude" is longer than "decades" — if it forces a shrink below target, **rotate, wrap, or abbreviate rather than shrink**, and report what you did.
4. **Preserve printed footprints** so pagination does not move; measure printed boxes from the built PDF with `mutool`, never from `\linewidth` arithmetic.
5. **Label hygiene (standing):** no prereg item names, no internal result IDs, no seed short-tags, ⛔ no author token, no file paths.

## Deliverables
- The regenerated PNGs in `figs/` (same filenames, same footprints), with `figs-BEFORE/` + md5 manifest preserved in scratch.
- **A before→after table per figure**: every text string changed, plus the zero-hit figures listed as swept-clean.
- **The data-identity evidence**: per-figure array hashes before and after.
- **Printed-size measurements** from a scratch build, with the page split reported (⛔ page limits are deferred by Head ruling — report, do not optimize).
- **The two `\TODO` tags named** (l.81, l.95 at last reading) as discharged-but-not-removed, for the Head or the cite pass to delete.
- Any figure you could not regenerate, declared honestly with the reason.

## Acceptance criteria
Both changes executed on every affected figure; **plotted data provably unchanged (hashes printed)**; type targets met at printed size; footprints preserved; ⛔ zero writes to `pj_sub.tex`, `pj_sub.pdf` or anything outside `figs/` and scratch; ⛔ no author name or citation number anywhere in any figure; the swept-clean list covers all five figures.
