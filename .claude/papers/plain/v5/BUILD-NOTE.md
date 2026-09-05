# plain/v5 — BUILD NOTE (plain, unformatted build of the PALM-variant V5)

**Source (read-only, verified byte-identical after this pass):** `.claude/papers/palm-variant/v5/submission.tex` + `neurips_2025_ml4ps.sty`.
**Built with:** `/Library/TeX/texbin/pdflatex -interaction=nonstopmode submission.tex` ×3 (pdfTeX, TeX Live 2026).
**Result: 18 pages, main text 5.00 pp** (`pdftotext -bbox` page-split instrument, text block 72–720 pt).
⛔ **Not optimised.** Source, for reference only: 10 pp / **4.00 pp** main text.

⭐ **Finding for the Advisor, measured not asserted: V5's 4.00 pp main text was typographic.** A control build of this same plain file with the headline figure put back at `0.58\linewidth` (the source's page-saving width) still measures **5.00 pp main text and 18 pp total** — so the growth is the font-size and skip stripping, not the figure. At default formatting the same main-text words occupy 5 pages. (Control build kept at `.claude/scratch/plain-text-builds/ctrl-v5/`.)

## 1 — Page-fitting devices removed (item 1)
| device in the source | disposition |
|---|---|
| float-placement overrides `\topfraction 0.92`, `\bottomfraction 0.75`, `\textfraction 0.06`, `\floatpagefraction 0.85` | removed (LaTeX defaults) |
| `\textfloatsep`/`\intextsep` 6 pt, `\abovecaptionskip` 3 pt, `\belowcaptionskip` 0 pt | removed (defaults) |
| the `\@startsection` redefinitions of `\section`/`\subsection` (with the source comment stating they existed so the 4-pp limit was "met by typography rather than by cutting text") | removed (default heading skips) |
| `\scriptsize` on 8 body blocks (nomenclature, both related-work blocks, three fine-print riders, the falsifier clause, limitations) | removed |
| appendix-wide `\scriptsize` after `\appendix` | removed |
| references: `{\tiny …\begin{multicols}{2}` + `\itemsep0pt \parskip0pt \leftmargin0pt` + `\columnsep 14pt` | **single-column list at default size, default item spacing**; `\usepackage{multicol}` dropped as now unused |
| `\scriptsize` on 6 tables and `\tiny` on 2 table captions | removed |
| `{\tiny` on the negatives tables | removed |
| reduced figure widths `0.58`, `0.90` ×2, `0.86` | all → `\linewidth` |

**Permitted `\small` exceptions used: FIVE, all on tables that physically overflow the text block at default size, listed here in full.**
| table | overfull at default | with `\small` |
|---|---|---|
| "The $T>0$ face of the budget" (tex 130–138) | 252.33 pt | **196.50 pt — still overfull** |
| "The shape claims on all four instruments" (tex 173–184) | 684.20 pt | **604.92 pt — still overfull** |
| "The instrument gap is a level, not a rate" (tex 188–199) | 88.38 pt | **49.44 pt — still overfull** |
| emergent confinement / laundering-control table (tex 267–282) | 11.53 pt | resolved |
| "The laundering control that fires" (tex 308–317) | 2.51 pt | resolved |

⚠ **Two of the three residuals are pre-existing, not created here:** the source build already ships overfull boxes of **91.68 pt** (the $T>0$ budget table, at `\scriptsize`) and **406.18 pt** (the four-instrument table, at `\tiny`) — i.e. those two tables do not fit the text block at *any* permitted size and need restructuring (transpose, or split the four instruments across two tables). That is an editorial fix, out of scope for a typesetting pass, and it is flagged in the report. The third (instrument-gap table) fits only below `\small` and is newly visible at plain size.

**Boxes, reported and not fought:** 3 overfull `\hbox` (the three tables above), 0 overfull `\vbox`, 9 underfull `\hbox`, 4 underfull `\vbox`.

## 2 — The author token appears nowhere (item 2)
**Standalone-token sweep, `grep -c "\bMo\b"`:** `submission.tex` → **1 hit, the bibliography entry** (`\item Mo, H. H. (2026). … arXiv:2605.03338`), expressly kept. **Prose / section text / captions / labels / filenames: 0.** Pronoun sweep (`his/His/he/him`) → 0 (the source had none in this file).
- False positives: this file contains **no** "Morse"/"Moser" occurrences (0 and 0), so nothing could be lost; the same check on the V2 build shows Morse ×1 and Moser ×2 surviving.
- Positive control: the same regex on the source file fires **2 occurrences on 2 lines** (one prose, one bibliography).
- Rewrite, the only mandated wording change in this file: *"…and Mo (2026) proves that an exactly $G$-equivariant flow has at least $\dim(G/\mathcal H)$ zero Lyapunov exponents along the orbit"* → *"…and a recent preprint on symmetry-protected Lyapunov neutral modes (arXiv:2605.03338) proves that …"*. Claim, scope and the "kinematics/constitutive" contrast that follows are untouched.
- No figure file or label in this build carries the name.

## 3 — Banked figures (item 3): all seven restored, none excluded
Placement rule applied: everything to the appendix, grouped with the result it evidences. **Multi-seed status is stated in every caption.**
| figure | appendix home | evidences | multi-seed status printed |
|---|---|---|---|
| `figB_dlaw.png` | A (the $(\mu,\gamma,T)$ budget) | the 25-cell diffusion law $1.0068\pm0.0219$ | **single checkpoint (seed 44)** — stated, with a pointer that the five-seed statements are elsewhere in the same appendix |
| `figB_signflip.png` | A | the sign flip and $n_{1/2}\propto1/T$ | **single checkpoint (seed 44)** — stated |
| `figB_massive_vs_flat.png` | A | the two regimes and the exact latch | 5 designed seeds |
| `figC_lambda_coset.png` | B (the emergent arm) | the un-collapsed emergent V-curve + the $10^{-3}$ latch failure | 3 emergent seeds + designed control |
| `figC_register_capacity.png` | B | "no continuous coset register", the ≈1–1.6-bit statement | 3 emergent seeds + designed control |
| `figC_Tstar.png` | B | the crossover $T^\star\approx3\times10^{-3}$ and the bias correction | **two emergent seeds + one designed control — an $n<3$ cell**, stated in the caption |
| `fig2_vault.png` | C (the friction-hole vault), beside the designed-arm paragraph | the refrigerator, the $8\times$ mechanism contrast, the $107.77\pm4.78\times$ vault | 3 designed seeds |

All seven were checked panel by panel before captioning; **none carries an internal-apparatus token** on its canvas. They do carry seed tags (`s42/s43/s44`, `seed 44`), which is consistent with this paper's own appendix text and tables, where the same seed numbers are printed.
**Figure inventory of this build (11):** main text — Fig 1 damping optimum; appendix — Figs 2–4 (budget), 5 (full-size collapse), 6–8 (emergent arm), 9 (two instruments), 10 (vault, designed), 11 (vault, emergent).

**One caption clause was corrected because the mandated width change made it false:** the full-size collapse figure said it was *"printed large enough to carry the annotations the main-text box cannot hold"* — with both figures now at `\linewidth` that explanation is untrue, while the content difference is real, so the clause now reads *"carrying the annotations the main-text version omits"*. Nothing else in that caption changed.

## 4 — Numeric two-way check
Same instrument as the V2 note, comparing the source against this file with the three inserted figure blocks removed.
- distinct tokens: source 570, plain 570.
- in source, not in plain: `2026` ×1 (the replaced citation) and `4`, `2`, `0`×3 — all typography (`multicols{2}`, `\itemsep0pt \parskip0pt \leftmargin0pt`, `\columnsep 14pt`).
- in plain, not in source: `2605.03338` ×1 (already in the reference list).
- New numbers occur only inside the seven new captions and each traces to this paper's own text: 25 cells / 1.0068 ± 0.0219 / seed 44 / Δ = 0.5 rad; +0.9552 ± 0.0422 and 10/10; $\gamma_{\rm crit}=2\varepsilon\mu$; dim 4, hidden 64, ε = 0.05; $\sim10^{-3}$; ≈1–1.6 bits; $T^\star\approx3\times10^{-3}$, $\ell_\theta/\Delta<0.06$; $T_{\rm local}=1.26\times10^{-4}$ vs $10^{-3}$; $107.77\pm4.78\times$; $86.97\pm2.94\times$; $T=10^{-3}$. (The "20 000 steps" and "8×" in two captions are read off the figures' own axes/panel titles.)

## 5 — Sweeps (per file, positive-controlled)
- **Internal apparatus / paths / program vocabulary** (same regex as the V2 note): **0 hits.** Positive control on the task file: 6 hits.
- ⚠ **Pre-existing, inherited, not introduced here:** the instrument IDs `I-J` (×4), `I-R1` (×3), `I-R2` (×2), `I-R3` (×5) still appear in this file's appendix text and in the Fig 9 caption, although the re-rendered PNG no longer prints them. That is item 1 of the still-unexecuted caption-sync worklist, and it is inherited verbatim from the source.
- **Semantic hermeticity:** unchanged from source — theory note as *(Anonymous, 2026)*, and *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"* present verbatim.
- **Anonymization posture:** unchanged — empty `\author{}`, PDF metadata scrub retained, `\@notice` suppressed, anonymization note at the end of the appendix retained.

## 6 — Source folders untouched
Before/after sha manifests: `v2-short` (21 files), `v5-short` (25), `v2-neurreps-descoped` (10), `neurreps-variants` (11), `palm-variant` (16) — **all five byte-identical**. `pdflatex` ran only inside `plain/` and inside `.claude/scratch/plain-text-builds/ctrl-v5/`.
