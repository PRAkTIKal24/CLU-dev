# plain/v2 — BUILD NOTE (plain, unformatted build of the de-scoped V2)

**Source (read-only, verified byte-identical after this pass):** `.claude/papers/v2-neurreps-descoped/submission.tex` + `neurips_2025_ml4ps.sty`.
**Built with:** `/Library/TeX/texbin/pdflatex -interaction=nonstopmode submission.tex` ×3 (pdfTeX, TeX Live 2026).
**Result: 17 pages, main text 7.00 pp** (measured with the page-split instrument used by the earlier variant build notes: `pdftotext -bbox`, text block 72–720 pt).
⛔ **The page count is NOT optimised and must not be read as a page result.** Source, for reference only: 14 pp / 6.14 pp main text — that difference is entirely typographic plus two restored figures.

## 1 — Page-fitting devices removed (item 1)
| device in the source | count | disposition |
|---|---|---|
| `{\footnotesize …\par}` on body blocks (retirements paragraph, Setup fine print, mandatory-flag block, `\S`4.4 scope clause, Discussion scope box, reference list) | 6 | removed; text at document default size |
| `\begin{center}\footnotesize` on tables (loan ladder ×2, compute ×2, GMOR condensate) | 5 | removed |
| `\begin{center}\scriptsize` on the two negatives tables | 2 | removed |
| appendix-wide `\footnotesize` after `\appendix` | 1 | removed |
| `\raggedbottom` | 1 | removed |
| reference-list `\setlength{\parskip}{2pt}` | 1 | → `\smallskipamount` (document-standard length) |
| reduced figure widths `0.68\linewidth`, `0.86\linewidth`, `0.80\textwidth` | 3 | all → `\linewidth` (natural width) |

**Retained deliberately:** the reference list's hanging indent (`\parindent -0.22in` / `\leftskip 0.22in`) — standard bibliography geometry, not a compaction device; it is single-column at default size. The venue-neutral `\@notice` suppression is unchanged (anonymization posture untouched).

**Permitted `\small` exceptions used: ZERO.** No table in this build required one. (⚠ a `grep '\\small'` on this file returns one line — it is `\smallskipamount` in the reference list, not a size command.)

**Boxes, reported and not fought:** 1 overfull `\hbox` (**3.57 pt**, the loan-curve ladder table, tex lines 202–212), 0 overfull `\vbox`, 26 underfull `\hbox` and 14 underfull `\vbox` (ragged `p`-column negatives tables and float pages). Float placement was left to LaTeX: Figure 4 (retention overlay) floats one page past its own appendix section.

## 2 — The author token appears nowhere (item 2)
**Standalone-token sweep, `grep -c "\bMo\b"`:** `submission.tex` → **1 hit, and it is the bibliography entry** (`Mo, H. H. (2026). … arXiv:2605.03338`), which the directive expressly keeps. **Prose / section text / captions / labels / filenames: 0.**
- False positives shown as excluded: **Morse ×1** (Akhtiamov & Thomson, *Morse theory*) and **Moser ×2** (Gardner et al. grid-cell authors) — both survive untouched.
- Positive control: the same regex on the source file fires **13 occurrences on 9 lines**.
- Pronoun sweep (`\bhis\b|\bHis\b|\bhe\b|\bhim\b`) → **0**; the source carried 7 lowercase + 1 capitalised, all re-pointed to the work rather than its author.

**How each mention is now phrased.** First mention (Related work) introduces the work as a citation and names the handle used afterwards: *"A recent preprint on symmetry-protected Lyapunov neutral modes (arXiv:2605.03338), the equivariant-Lyapunov preprint below, proves that …"*; later mentions read *"the equivariant-Lyapunov preprint (arXiv:2605.03338)"*, *"its estimator"*, *"its autonomous protocol"*, *"the preprint's own instrument"*. The arXiv number resolves into the reference list, so every mention is a citation. **No claim's scope, hedge or number was altered by any of these rewrites** (see the numeric check, §4).

**Renames (labels + filenames carrying the name):**
| was | now |
|---|---|
| `figs/fig1_mo_headtohead.png` | `figs/fig_lifetime_headtohead.png` |
| `\label{fig:mo}` | `\label{fig:lifetime_headtohead}` |
| `\label{sec:mo}` (+ its 3 `\ref`s) | `\label{sec:headtohead}` |
| appendix title *"…head-to-head on Mo's own estimator"* | *"…head-to-head on the preprint's own estimator"* |

⚠ **The name was also printed INSIDE the headline figure** (title *"Mo's lifetime protocol on trained CLUs"*, legend *"Mo's median 1.013"*, in-axes note *"…same pattern as Mo's own ε=10⁻⁴ row"*). A PNG rename does not fix that, so the figure was **re-rendered from the render pass's own generator** (`.claude/outputs/figure-render-pass/new_v2f1.py`) with **four string literals changed and nothing else** (three text strings + the output filename; the diff is 4 lines, all string literals). Verification: the render pass's data tap gives digest `baedf7981e42e7035a193dc996d375d53c9f9075` over 9 data calls for **both** the banked figure and this render — **identical plotted values**, same pixel dimensions (2044×1118).

## 3 — Banked figures (item 3)
| banked figure | disposition | where | why |
|---|---|---|---|
| `fig3_retention_overlay.png` | **restored** | Appendix C (new, one figure + one lead line) | the baseline-collapse result of §4.3 shown directly. Placed in the appendix, not main text, because one of its three curves is a single checkpoint (see caption) and single-seed material is appendix material. ⚠ *Editorial question for the Hub in the report: the Head named this figure; promotion to main text is a one-word call.* |
| `fig1_gmor.png` | **restored, re-rendered** | main text, under §4.1 | it is the clearest presentation of the paper's single contribution (the μ⁻² law, the floor, the metric bifurcation), and §4.1 had no figure. ⚠ The banked PNG printed two internal-apparatus tokens in its legend (`F5 exact-map prediction`, `(C2 retention)`), so it was re-rendered from its own surviving generator (`.claude/scratch/v2-full-runs/make_figures.py::fig1_gmor`) with **two legend strings changed and nothing else**. Verification: the unmodified generator reproduces the banked PNG **byte-for-byte** (sha1 `00eaceca1adf82c24340f8628dc62a5ec7ce9789`), and the tap digest of the edited run is identical to the unmodified run (`9c284b32c84f341be86de43696c74e94212b6782`, 7 data calls) — identical plotted values. |
| `sf1_mo_estimator_overlay.png` | ⛔ **EXCLUDED** | — | **it cannot ship under directive 2 and cannot be fixed in this pass**: its canvas carries the author token three times (title `SF-1: Mo's own estimator …`, legend `prediction from Mo's OWN estimator`, legend `measured lifetime (Mo escape protocol)`) plus the internal ID `SF-1`, and **no generator survives** anywhere under `.claude` (sweep: 0 `.py` files reference `mo_estimator`/`estimator_overlay`). Its result is fully in the text (Appendix F: corr 0.9995, meas/pred 0.86–1.03, 0.30 at δ=4). **Owed re-render is actionable**: the extract is banked at `.claude/outputs/v2-referee-experiments/mo_estimator_extract.npz` (+ `mo_estimator_table.json`). |

**Figure inventory of this build (5):** main text — Fig 1 price list (`fig1_gmor.png`), Fig 2 lifetime head-to-head (`fig_lifetime_headtohead.png`); appendix — Fig 3 anchor cure, Fig 4 retention overlay, Fig 5 GMOR condensate.

## 4 — Numeric two-way check
Instrument: multiset of all numeric tokens in `submission.tex`, with graphics widths and `\setlength` arguments excluded as typography, comparing the source against this file **with the two inserted figure blocks removed**.
- distinct tokens: source 407, plain 407.
- in source, not in plain: `2026` ×2, `1` ×1 → the two `Mo (2026)` citations replaced by the arXiv number, and the `1` of the old filename `fig1_mo_headtohead`.
- in plain, not in source: `2605.03338` ×3 → the arXiv number, which is itself in the source's reference list.
- **New numbers appear only inside the two new captions**, and every one traces to a sentence of this same paper: −0.985 / −1, floor 27.03 at γ=0.05, 3.2× metric split, 1.000000 ± 5×10⁻¹² over 4.5 decades, 5 seeds, dim 4 (§4.1); ≈5.6/56/69 map-steps, 1.2 rad, ≈0.35 rad, 5/5 seeds, threshold 0.2 rad, dim 4 / hidden 64 (§4.3).

## 5 — Sweeps (per file, positive-controlled)
- **Internal apparatus / paths / program vocabulary** (`SF-[0-9]|F-[0-9]|F5|C2|C3|CM-[0-9]|Cor-[0-9]|.claude|/Users|scratch/|handover|Advisor|Hub|spoke|never-quote|PREREG|N[0-9]{3}|CSF3|CAMELS|CMAPSS|K5|organizer swap|13.9|bprime|CLU-former`): **0 hits.** Positive control on the task file: 6 hits.
- **Semantic hermeticity:** the only unpublished-work references are the source's own and unchanged — the theory note cited as *(Anonymous, 2026)*, and the naming-continuity sentence *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"* (present, verbatim).
- **Anonymization posture:** unchanged — empty `\author{}`, `\@notice` suppressed, no venue string.

## 6 — Source folders untouched
Full-file sha manifests taken before and after the pass: `v2-short` (21 files), `v5-short` (25), `v2-neurreps-descoped` (10), `neurreps-variants` (11), `palm-variant` (16) — **all five byte-identical**. `pdflatex` was run only inside `plain/`. The banked generator outputs were also left alone (`.claude/outputs/v2-full-runs/fig1_gmor.png` unchanged; all tapped renders were redirected to scratch).
