# v2-fig1-designed — results-analyst — one word in one figure: "learned vacuum" → "designed vacuum"

**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-24 (charter Addendum 81).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output: `.claude/outputs/v2-fig1-designed.md`.

**DIAL DECLARATION: none — a text-on-figure fix. ⛔ ZERO new measurements, ZERO data changes.**

## The defect
`figs/fig1_gmor.png` — the paper's **headline figure**, Figure 1 in main text — carries an in-image panel title reading **"GMOR spectral-mass law: exact on the LEARNED vacuum"**. The figure's caption, §4.1, and the paper's entire designed-vs-learned honesty architecture all say these results are on the **DESIGNED** vacuum. ⭐ **The image contradicts the paper's central epistemic distinction, in the first figure a reviewer sees.**
- **Source located (Advisor-verified):** `.claude/scratch/plain-text-builds/plain_v2_gmor.py`, **line 81**, inside `set_title("GMOR spectral-mass law: exact on the learned vacuum\n(max deviation $\\sim 10^{-12}$, all …")`.
- ⚠ This generator has been executed before by `v2-figure-text-pass`, which established it **reproduces the shipped PNG byte-for-byte** — so it is the right source and the rendering path is known-good.

## The change — exactly one word
`learned` → `designed`, in that title only. ⛔ **Nothing else in the generator, no other panel, no other figure.** Check whether the same word appears in any other panel title of the same figure and report it, ⛔ but do not change anything the Head has not named.

## Method (the discipline this program's figure work already established)
1. **Re-render from the generator**, never by image editing.
2. ⭐ **Prove the data did not move:** run under a **data tap that hashes every array handed to matplotlib**, before and after; the hashes must be identical. **A text change that alters a plotted value is a failed pass.**
3. **Preserve the printed footprint exactly** — the paper sits at ~4.35 pp main text against a 4-pp limit, so ⛔ **any change in figure height or width is a defect**, not a side effect. Measure the printed box from a built PDF (`mutool`), not from `\linewidth` arithmetic.
4. **Type targets:** ticks ≥ 7 pt, labels/legend ≥ 8 pt, titles ≥ 9 pt effective at printed size. "designed" is one character longer than "learned" — if it forces a wrap, report it rather than shrinking type.
5. **Back up first:** copy the current PNG to `.claude/scratch/v2-fig1-designed/figs-BEFORE/` with an md5 manifest.

## Constraints
- ⛔ **Write ONLY `figs/fig1_gmor.png` inside `.claude/NIPSsubmission/v2-neurreps/`.** Not the `.tex`, not the PDF, not any other figure. The Head is actively finishing this paper.
- **Build to verify in a scratch copy**, never in the live folder.
- ⛔ The author token appears in no figure content, label or filename.

## Acceptance criteria
One word changed in one panel title; array hashes identical before/after (printed); printed box identical; type targets met; the PNG the only file written in the live folder; before/after backup preserved.
