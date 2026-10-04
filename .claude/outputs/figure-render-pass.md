# figure-render-pass — results-analyst report

Task + acceptance criterion: re-render the six variant figures at their **final printed size** with legible type, footprints preserved, plotted values provably unchanged, internal labels stripped; rebuild both variants and report page splits.
Status: **done** — six figures regenerated and installed, all six values verified unchanged, both variants rebuilt with **page splits unchanged (V5 4.00 pp main / 9 pp total; V2 5.67 pp main / 13 pp total)**, `papers/v2-short/**` and `papers/v5-short/**` byte-identical (46-file manifest diff empty).

**⚠ DOWNSTREAM RECONCILIATION LIST — THIS REPORT CONTAINS ONE, IT NEEDS AN OWNER.** §7 lists **9 caption edits owed** in `papers/palm-variant/v5/submission.tex` and `papers/neurreps-variants/v2/submission.tex`. Three of them (V5 Fig 2's `I-J/I-R3/I-R1`, V5 Fig C.2's "panels are labelled by the pre-registered prediction each tests", V5 Fig C.2's colour key) **name or rely on labels that no longer exist in the shipped PNGs**, so the current builds have captions that point at absent artefacts. I did not edit any caption (task forbids it). This needs a curator/writer task at the review that accepts this report.

**⚠ SECOND FLAG — §8: three of the six figures cannot meet the type target at their current footprint.** V5 Fig 1, V5 Fig 2 and V5 Fig C.2 are printed **1.115 / 1.140 / 0.973 inches tall**; at that height every legend had to be deleted and two panels lost their in-figure numeric labels. That is a layout question for the Advisor, not a render question.

**DIAL DECLARATION (echoed): none — instrument/figure regeneration from banked artifacts. Zero new measurements. Laundering control: n/a. Falsifies: n/a. Does NOT falsify: n/a.**

---

## 1. What I did

1. Located the generating script or banked artifact behind each of the six figures (§2).
2. Built a **data tap** (`tap.py`) that hashes every numeric array handed to a matplotlib plotting call, and a comparator that treats the calls as an order-insensitive multiset. Ran every ORIGINAL generator under it (with `savefig` redirected so no banked artifact could be overwritten), then every NEW render under it, and compared (§4).
3. Measured each figure's **actual printed box** from the built PDFs with `mutool draw -F trace` rather than trusting the `\linewidth` arithmetic (§3) — the real shrink is much worse than the task's table assumed.
4. Re-rendered each figure on a canvas whose width **is** the printed width, at an integer pixel size in **exactly** the banked aspect ratio (`size.py`), so LaTeX places each in an identical box.
5. Stripped internal labels and seed short-tags (§6); swept the PNGs for identifying strings (§9).
6. Rebuilt both variants ×3 and measured the page split against a control build of the same `.tex` with the OLD figures (§5).

## 2. Provenance: which artifact each figure comes from

Every figure is a **replot of banked data**. Five of the six have a surviving generator; **running it today reproduces the banked PNG byte-for-byte** (SHA-1 match), which is the strongest provenance available.

| variant figure | banked SHA-1 | generator | re-run reproduces banked PNG? |
|---|---|---|---|
| V2 `fig1_mo_headtohead.png` | `1529629f…` | `scratch/v2-full-runs/make_figures.py::fig2_mo` → `outputs/v2-full-runs/fig2_mo.png` | **YES, byte-identical** |
| V2 `fig2_anchor_cure_laws.png` | `887980781e…` | **none survives** → `outputs/v2-referee-experiments/sf3_anchored3000_laws.png` | reconstructed + validated, §4.2 |
| V2 `fig3_gmor_condensate.png` | `b988a66c…` | `scratch/f1-gmor-condensate/analyze_and_figure.py` (figure block) | **YES, byte-identical** |
| V5 `fig1_damping_optimum.png` | `fd25cdbb…` | `scratch/v5-revision-1/fig1_collapse.py` → `papers/v5-short/figs/fig1_collapse.png` | **YES, byte-identical** |
| V5 `fig2_two_instruments.png` | `1ea68bba…` | `scratch/v5-vcurve-validation/a1_analyse.py` → `fig_me1_vcurve_rollout.png` | **YES, byte-identical** |
| V5 `figC2_vault_emergent.png` | `4e49bce1…` | `scratch/v5-vcurve-validation/a3_fig.py` → `fig_me3_vault_emergent.png` | **YES, byte-identical** |

### 2.1 Flag-provenance table (inherited — no new measurement was made)

| figure | source artifact(s) | seeds | non-default flags in effect at measurement time |
|---|---|---|---|
| V2 F1 | `outputs/v2-full-runs/gmor_sweep.npz` | designed 42–46 (5) | dim 4, hidden 64, ε=0.05, γ=0.05, dt=0.05, 150 ep; Mo protocol: phase 0.35 rad, threshold 0.2 rad, his censoring, cap 15000 steps; 14 breaking magnitudes; 10/70 runs censored (δ≤3×10⁻⁴) |
| V2 F2 | `outputs/v2-referee-experiments/sf3_{gmor,ep}_sweep.npz` | anchored 42–44 (3) | anchored λ=100, **3000 epochs**, γ=0.05, dt=0.05, δ∈[10⁻⁴,4] (14 pts), EP sweep 12 factors ×3 seeds (bit-identical across seeds) |
| V2 F3 | `outputs/f1-gmor-condensate/{gmor_condensate,angular_tilt_contrast}.npz` | designed 42–46 + anchored3000 42–44 (8 checkpoints) | probe-only, no retraining; linear ambient spurion δ∈[10⁻⁸,0.3] (10 pts); float64 |
| V5 F1 | `outputs/t-lever-forgetting/s4b_jacobian.json` (115 rows) + `outputs/v5-gate/e1c_vcurve.json` | designed 42–46 (5) + emergent 42–44 (3) | one-step Jacobian, dim 4, hidden 64, ε=0.05, `langevin_noise="fdt"`, Newtonian kinetic mode, laptop-CPU float64; colorbar floor marker = ring-profile probe floor 1.7×10⁻¹² |
| V5 F2 | `outputs/v5-vcurve-validation/m1_emergent.json` | emergent 42–44 (3) | `config`: 48 γ ∈ geomspace(0.002,0.5), δ∈{0.05,0.2,0.5} (plotted δ=0.05), dt=0.05, **T=0.0, noise="none (T=0 deterministic step)"**, retie=True, n_chunks 8000, min_steps 20000, n_halflives 30, ov_min 0.3 |
| V5 F C.2 | `outputs/v5-vcurve-validation/m3_{emergent_VDS,designed_crosscheck,emergent_X,fpt_emergent,fpt_samesite}.json` | emergent 42–44 (3), designed cross-check 42–44 | `config`: dt=0.05, γ=0.05, γ_φ∈{0,0.1,0.2,0.3,0.5}, Δ=0.5 rad, **noise="fdt"**, retie=True, T∈{4,8}×10⁻³, gamma_max=0.9, gate="compact", width=0.25, uniform_radius=50.0 |

Environment: repo HEAD `7fcef501fb1aeae33a8d149b046c9b6126dcecd7`, main venv, **matplotlib 3.10.8, numpy 2.4.1, Pillow 12.1.0**, TeX Live 2026 / pdfTeX 1.40.29. No JAX was needed (V2 F3's generator imports JAX only for its part (1), whose sole figure-side output `tr` was banked as `angular_tilt_contrast.npz`; §4.1 shows the isolated figure block still reproduces the PNG byte-for-byte).

## 3. The real shrink is far worse than the brief assumed

Printed boxes measured from the built PDFs (`mutool draw -F trace`, pt → in at 72 pt/in), against the canvas the banked PNG was actually rendered on (`px / dpi` from the PNG header):

| figure | rendered canvas | printed box | **linear scale** |
|---|---|---|---|
| V2 F1 `0.68\linewidth` | 6.813 × 3.727 in¹ | 269.28 × 147.29 pt = **3.740 × 2.046 in** | **0.549×** |
| V2 F2 `0.86\linewidth` | 11.000 × 4.200 in | 340.57 × 130.04 pt = **4.730 × 1.806 in** | **0.430×** |
| V2 F3 `0.60\textwidth` | 11.600 × 8.800 in | 237.60 × 180.25 pt = **3.300 × 2.503 in** | **0.284×** |
| V5 F1 `0.60\linewidth` | 9.941 × 3.359 in¹ | 237.60 × 80.28 pt = **3.300 × 1.115 in** | **0.332×** |
| V5 F2 `0.74\linewidth` | 15.000 × 4.200 in | 293.04 × 82.05 pt = **4.070 × 1.140 in** | **0.271×** |
| V5 F C.2 `0.80\linewidth` | 19.000 × 4.200 in | 316.79 × 70.03 pt = **4.400 × 0.973 in** | **0.232×** |

¹ saved with `bbox_inches="tight"`, so the canvas is the tight bbox, not the `figsize`. V2 F1's wide two-line title *expanded* the bbox from `figsize=(5.4,3.8)` to 6.81 in — which is why it is the worst-shrunk V2 panel relative to its design.

**Effective printed type size, before → after.** Because the new canvas width **is** the printed width, the specified pt sizes are the printed pt sizes (scale 1.000×). "Before" = the generator's specified pt × the scale above.

| figure | ticks (target ≥7) | axis labels (≥8) | panel titles (≥9) | legend / annotations (≥8) |
|---|---|---|---|---|
| V2 F1 | 4.94 → **7.0** | 4.94 → **8.0** | 5.93 → **9.0** | 3.57 → **8.0** legend, **7.0** fine-print note |
| V2 F2 | 4.30 → **7.0** | 4.30 → **8.0** | 5.16 → **9.0** | 3.44 → **8.0** (legends replaced by direct labels) |
| V2 F3 | 2.85 → **7.0** | 2.85 → **8.0** | 2.99 → **9.0** | 1.76–2.16 → **7.0–7.5** ⚠ below target |
| V5 F1 | 3.32 → **7.0** | 3.32 → **8.0** | 3.65 → **9.0** | 2.72 → **8.0** slope labels; **6.5** colour-key note ⚠ |
| V5 F2 | 2.71 → **7.0** | 2.71 → **8.0** | 3.26 → **9.0** | 1.63 → **7.0** ⚠ (legends deleted) |
| V5 F C.2 | 2.32 → **7.0** | 2.32 → **8.0** | 2.78 → **9.0** (letters only ⚠) | 1.39–1.85 → **7.0** ⚠ (legends deleted) |

Cross-check on the derivation: the banked V5 Fig C.2 tick digits measure 16 px of ink at 3040 px across a 316.79 pt box ⇒ 1.67 pt cap-height ⇒ ≈2.3 pt nominal, matching the 2.32 pt derived above.

Before/after sheets rasterised **at the printed width** (300 dpi) — this is what a reviewer sees:
`.claude/outputs/figure-render-pass/compare_{fig1_mo_headtohead,fig2_anchor_cure_laws,fig3_gmor_condensate,fig1_damping_optimum,fig2_two_instruments,figC2_vault_emergent}.png`

## 4. Verification that no plotted value moved

### 4.1 The five figures with surviving generators — hash-of-arrays comparison
`tap.py` wraps `Axes.{plot,loglog,semilogx,semilogy,scatter,bar,barh,errorbar,fill_between,axhline,axvline,axhspan,axvspan,step,stairs,hist,imshow,pcolormesh,contour,contourf}` and records a SHA-1 of every positional numeric argument, cast to `<f8`. Colours, labels, sizes, limits and tick choices are deliberately **not** recorded. Empty legend-proxy calls are dropped. Comparison is a multiset difference.

| figure | original digest | new digest | data calls | verdict |
|---|---|---|---|---|
| V2 F1 | `05cc1163…` | `baedf798…`² | 8 vs 8 | **IDENTICAL DATA** |
| V2 F2 | `02f7deeb…` | `02f7deeb…` | 9 vs 9 | **IDENTICAL DATA** |
| V2 F3 | `f674c5ad…` | `f674c5ad…` | 66 vs 66 | **IDENTICAL DATA** |
| V5 F1 | `36876fc3…` | `13bc1b22…`² | 23 vs 23 | **IDENTICAL DATA** |
| V5 F2 | `cb22bfd2…` | `cb22bfd2…` | 46 vs 46 | **IDENTICAL DATA** |
| V5 F C.2 | `6c11f9b0…` | `6c11f9b0…` | 38 vs 38 | **IDENTICAL DATA** |

² whole-run digest differs only because the number of *empty* legend-proxy calls changed; the multiset of non-empty data calls is exactly equal (comparator prints `in ORIGINAL not in NEW : 0` / `in NEW not in ORIGINAL : 0`). Digests: `.claude/outputs/figure-render-pass/digests/*.json`; comparator `compare.py`.

Additionally, running each original generator under the tap wrote a PNG whose SHA-1 **equals the banked figure's** (all five, §2) — so the tap itself is provably faithful to the shipped artefact.

### 4.2 V2 Fig 2 — no generator survives; reconstruction validated three ways
**An unbounded search of every `.py` file under `.claude/` (20,827 text files; no depth limit) for `anchored3000_laws` returned zero matches** — so no script anywhere in the workspace writes `sf3_anchored3000_laws.png`. A second unbounded sweep of all `.py/.md/.sh/.tex` for the six *variant* figure filenames returned 17 files, **all of them `.md` or `.tex`** (task file, four spoke reports, three BUILD-NOTEs, the `submission.tex`/`main_body.tex`/`appendix.tex` files) and **not one `.py`**, independently confirming that the generators only ever write *source* filenames (`fig2_mo.png`, `fig1_collapse.png`, `fig_me1_vcurve_rollout.png`, …) and the variant names were produced by copy-and-rename. `outputs/v2-style-pass.md` §"Instruction 3" corroborates this: it *promotes* an already-existing `fig2_anchor_cure_laws.png` from the appendix and never generates it. I re-derived it from the banked `sf3_gmor_sweep.npz` / `sf3_ep_sweep.npz` (`sf3_reconstruct.py`) and validated:

1. **The three fitted numbers printed on the banked figure are reproduced exactly from the npz.** Panel (a) legend "slope -0.961" ⇒ mean-over-3-seeds `n_half_env` vs δ, OLS over the first 7 δ (1e-4…6e-2, the overdamped rows) = **−0.96062**. "mass-indep floor 27.03" ⇒ 2 ln2 / (−ln(1−0.05)) = **27.02681**. Panel (b) "slope 0.516" ⇒ OLS of log φ on log(h−h*) over the 9 points with φ>0 = **0.51645**. A brute-force scan over 6 aggregations × all fit windows found **exactly one** combination giving −0.961, so the aggregation is identified, not guessed. (Note: the source report `outputs/v2-referee-experiments.md` quotes **−0.956**, which is the *per-point* fit over all overdamped rows — a different, also-correct statistic. The **figure's** number is −0.961 and that is what is reproduced.)
2. **Pixel geometry.** Rendering the reconstruction's data into axes boxes pinned to the banked figure's exact pixel frame (`sf3_verify_geom.py`) gives per-column curve agreement of **median 0.37 px / mean 0.39 px (green retention curve, 509 columns)** and **median 0.70 px / mean 0.69 px (purple EP points, 99 columns)**; maxima of 4.5 / 2.3 px occur only at marker edges (marker-size mismatch), not on the line bodies.
3. **Full-frame render.** A defaults-matched reconstruction at the banked canvas (11.0 × 4.2 in @ 130 dpi) is visually indistinguishable; the axes boxes differ by 5 px on the left and 18 px at the top, i.e. the lost original used slightly smaller title/label fonts than matplotlib defaults. That residual is typographic only and does not touch the data.

⚠ **Residual risk, stated:** for V2 Fig 2 alone, "values unchanged" rests on re-derivation, not on a byte-identical re-run. The evidence above is strong but it is not the same class of evidence as the other five.

## 5. Rebuilds and page splits

Instrument: `pagesplit.py` — fractional pages from PDF word bounding boxes against the text block (top 72 pt, bottom 720 pt, page 792 pt), the same instrument both BUILD-NOTEs used. Build: `pdflatex -interaction=nonstopmode` ×3 in each variant directory, TeX Live 2026.

| | control (old figs) | **after this pass** | target |
|---|---|---|---|
| **V5 variant** main text | 4.00 pp | **4.00 pp** | 4.00 pp hard limit — **held** |
| V5 variant total | 9 pp | **9 pp** | 9 pp — **held** |
| V5 overfull / undefined | 2 (`91.6832pt`, `406.18022pt`) / 0 | **2 (identical values) / 0** | inherited, unchanged |
| **V2 variant** main text | 5.67 pp | **5.67 pp** | must not grow — **held** |
| V2 variant total | 13 pp | **13 pp** | must not grow — **held** |
| V2 overfull / undefined / underfull | 0 / 0 / 15 | **0 / 0 / 15** | unchanged |

The V2 control is a real build: `submission.tex` + `.sty` copied to `/tmp/v2ctl` with the **old** PNGs restored, built ×3 — 13 pp, main 5.67 pp, 0 overfull, 15 underfull. So the 15 underfull `\vbox`/`\hbox` warnings are pre-existing, not introduced here.

**Footprint check — printed image boxes, before → after (pt):**

| figure | before | after | Δ |
|---|---|---|---|
| V2 F1 | 269.283 × 147.289 | 269.301 × 147.299 | +0.018 × +0.010 |
| V2 F2 | 340.568 × 130.035 | 340.571 × 130.036 | +0.003 × +0.001 |
| V2 F3 | 237.598 × 180.247 | 237.609 × 180.255 | +0.011 × +0.008 |
| V5 F1 | 237.598 × 80.277 | 237.616 × 80.283 | +0.018 × +0.006 |
| V5 F2 | 293.036 × 82.050 | 293.057 × 82.056 | +0.021 × +0.006 |
| V5 F C.2 | 316.788 × 70.027 | 316.808 × 70.031 | +0.020 × +0.004 |

All ≤0.03 pt (pdfTeX's rounding of `px/dpi` to bp). Aspect ratios are **exactly** preserved as integer pixel fractions — 1022/559, 55/21, 29/22, 1690/571, 25/7, 95/21 before and after (`size.py` chooses a dpi that lands on integer pixels in the reduced ratio).

## 6. Label hygiene — what was stripped

| figure | removed | replaced with |
|---|---|---|
| V2 F1 | 2-line title with duplicated headline numbers; per-seed legend `seed 42…46` | 1-line title "Mo's lifetime protocol on trained CLUs"; single legend entry "5 trained models". Censoring fine print **kept** (it is in no caption and no body sentence). |
| V2 F2 | panel titles `SF-3a:` / `SF-3b:`; parentheticals duplicated in the caption | "GMOR retention at 3000 ep" / "EP onset at 3000 ep"; the fitted numbers moved to colour-matched in-axes labels (grey = fit line, red = floor) |
| V2 F3 | suptitle `F-1: …`; panel-(a) legend of checkpoint short-tags `a3000 s42 … d150 s46`; verbose panel titles; the `\dfrac` LEC ylabel | no suptitle (caption carries it); short reader-facing panel titles; "LEC ratio"; in-axes identity label `μ²F²=δΣ` |
| V5 F1 | 4-line callout box (verbatim in the caption); legend of evidence tiers; 3-line colorbar annotation | "slope −1"/"slope +1" at 8 pt; one 6.5 pt crimson note "crimson: flat-coset probe floor" |
| V5 F2 | instrument IDs `I-J`, `I-R1`, `I-R3`, `Γ_jac/Γ_R3`, `n_jac/n_R1`; seed tags `s42/s43/s44`; 9-entry + 6-entry legends | titles "(a) T=0 V-curve" / "(b) instrument ratio" / "(c) collapsed"; in-axes "decay rate" and "threshold" |
| V5 F C.2 | pre-registration item names `Q1 Q2 Q3 Q5`; seed tags `s42/s43/s44`; 8-entry legend in (a); legends in (c),(d); the 9 hop-fraction and 3 vault-factor bar labels | panel letters only; direct labels "coupled bath" / "absorb-only"; x-axis "seed 1/2/3" |

**Seed mapping** (stated so captions stay accurate): everywhere a seed tag was neutralised the mapping is file order, **42→1, 43→2, 44→3** (and 45→4, 46→5 for the 5-seed designed families). V5 Fig 2 and Fig C.2 now show "1/2/3"; V2 Fig 1's five per-seed curves keep their colours but the legend no longer names them.

**Zero internal-label tokens remain in any string that reaches the canvas.** Sweep over all six new scripts for `SF-3|F-1|Q[1235]|T5|T6|Cor-13|CM-16|I-J|I-R|s4[2-6]|a3000|d150|_jac|_R[13]|scratch|.claude|/Users` restricted to `set_title|set_xlabel|set_ylabel|suptitle|set_label|.text|label=` lines: **0 hits in all six**. Positive control: the same regex on the original `a3_fig.py` fires **4**.

## 7. CAPTION EDITS OWED (I made none — this is the reconciliation list)

**`papers/palm-variant/v5/submission.tex`**
1. **Fig 2 caption (l.203)** names `(I-J, lines)`, `(I-R3, circles)`, `(I-R1, dotted)`. Those IDs are gone from the figure. → e.g. "on the one-step Jacobian (lines), the rollout envelope-rate instrument (circles) and the rollout first-crossing threshold (dotted)".
2. **Fig 2 caption**: panel (a)'s y-axis reads `n_{1/2}` in **steps**; the caption does not say so. Add units, or accept the axis label carries it (it does: "$n_{1/2}$ (steps)").
3. **Fig C.2 caption (l.262)** opens "panels are labelled by the pre-registered prediction each tests". **False after this pass** — the Q-labels are stripped. Delete that clause.
4. **Fig C.2 (a)**: the circle/square = temperature key was in the deleted legend. Add "circles $T=4\times10^{-3}$, squares $T=8\times10^{-3}$" to the caption.
5. **Fig C.2 (c)**: the colour key was in the deleted legend. Add "red = no hole ($\gamma$=0.05), grey = scalar control at matched $\gamma_{\rm eff}$, blue = $\gamma_\phi$ hole".
6. **Fig C.2 (c)**: caption says "stationary spread **and hop fraction**"; the per-bar hop-fraction labels no longer fit and were removed (the numbers 5.5/43.0/2.4 % and 0.73/10.2/0.26 % are already in §3.2 of the main text). Either drop "and hop fraction" or point at §3.2.
7. **Fig C.2 (d)**: the colour key was in the deleted legend. Add "red = no hole (same site), blue = $\gamma_\phi$ hole"; also state $T=4\times10^{-3}$ (panel (d) alone is at that temperature). The removed `>1379×/35×/>1290×` bar labels are already in the appendix text verbatim.

**`papers/neurreps-variants/v2/submission.tex`**
8. **Fig 3 caption (l.341)**: panel (a)'s marker key was in the deleted legend. Add "open circles = anchored 3000-epoch checkpoints, filled squares = designed 150-epoch checkpoints" (8 checkpoints total, already stated).
9. **Fig 3 caption**: panel (c)'s "max dev 2e−15" and panel (d)'s "slope −1.05" were legend/caption duplicates; (d)'s −1.05 is already in the caption, (c)'s 2.22×10⁻¹⁵ is already in §C.4. No action needed if the Advisor is content that they live in prose.

No caption in either file names a stripped **seed** tag, so the seed neutralisation is caption-safe.

## 8. FLAGGED FOR THE ADVISOR — legibility unreachable at the current footprint

The hard constraint is vertical. After a 9 pt title, 7 pt tick labels and an 8 pt axis label, the remaining **axes height** is:

| figure | printed height | axes height left | panels across | verdict |
|---|---|---|---|---|
| V2 F1 | 2.046 in | ≈1.35 in | 1 | **comfortable** — target met with a 3-entry 8 pt legend |
| V2 F2 | 1.806 in | ≈1.30 in | 2 | **met**, but only by replacing both legends with direct labels |
| V2 F3 | 2.503 in | ≈0.78 in / panel | 2×2 | **marginal** — all legends deleted, in-axes notes at 7–7.5 pt |
| **V5 F1** | **1.115 in** | **≈0.60 in** | 1 + colorbar | **UNREACHABLE.** A 3-entry 8 pt legend is 0.45 in tall = 75 % of the panel; I measured it covering the whole underdamped branch. Legend deleted, callout box deleted. |
| **V5 F2** | **1.140 in** | **≈0.67 in** | 3 | **UNREACHABLE.** 9- and 6-entry legends at 8 pt do not fit in a 1.36 in-wide panel. Both deleted; the caption already decodes them. |
| **V5 F C.2** | **0.973 in** | **≈0.54 in** | 4 | **UNREACHABLE, worst case.** Panel pitch 1.10 in, axes ≈0.70 × 0.54 in. No descriptive panel title fits at 9 pt (≈9 characters of room); panel letters only. All legends and all 12 numeric bar labels deleted. |

**Recommendations (Advisor's call, not taken here):**
- **V5 Fig 1**: `0.60\linewidth` → **`0.90\linewidth`** would give 1.67 in of height and restore the legend and the "μ→0 corner" callout at 8 pt. Cost: +0.35 in of column height.
- **V5 Fig 2**: `0.74` → **`1.00\linewidth`** gives 1.54 in; still no room for a 9-entry legend, but a 3-entry instrument legend at 8 pt fits. Alternatively drop panel (c) (its content is one sentence of §3.1) and print (a)+(b) at 0.74 — that *reduces* the block.
- **V5 Fig C.2**: this figure wants a **2×2 layout at ≈`0.80\linewidth` × 2.0 in tall** (≈+1.0 in of column height) or it wants to lose a panel. At 1×4 and one inch tall it is a strip of thumbnails whatever the type size. ⚠ V5's main text is at exactly 4.00 pp against a hard limit — but Fig C.2 is in **Appendix B**, so growing it costs appendix pages, not main pages. That is the cheap fix and I recommend it.
- **V2 Fig 3**: 2×2 at 3.30 in wide is tight; `0.60\textwidth` → `0.80\textwidth` would restore the panel-(a) marker legend. This figure is in an appendix too.

## 9. Anonymisation / identifying-string sweep

Per-file `strings -a | grep -Eic "user|Desktop|CHLU|claude|Forgis|x10719pj|worktree|/Users|agent/|scratch|papers|Manchester|CERN|matplotlib"` on each of the six new PNGs: **1 hit each before the fix, 0 after.** The single hit was matplotlib's own `Software` tEXt chunk (`Matplotlib version3.10.8, https://matplotlib.org/`), which the **banked** figures also carry; I stripped it via `savefig(..., metadata={"Software": None})`, so the new PNGs carry only a `dpi` chunk. Positive controls fired (`strings` finds `matplotlib` in the banked PNG = 1; finds `/Users/` in `analyze_and_figure.py` = 1). No path, username, worktree name or project string appears in any regenerated PNG.

## 10. Untouched-live-build check (acceptance criterion 4)

`shasum` manifest of **all 46 files** under `papers/v2-short/` and `papers/v5-short/`, taken before any work and re-taken at the end: `diff` **empty**. (`live_builds_BEFORE.txt` / `live_builds_AFTER.txt` in the scratch dir.) A mid-run intermediate check was also clean: `fig1_collapse.py` prints the path `papers/v5-short/figs/fig1_collapse.png`, but the tap's `savefig` redirect meant nothing was written there — verified by re-running the manifest immediately after that call.

## Git footprint

**None.** `git status --porcelain` is empty; HEAD unchanged at `7fcef501fb1aeae33a8d149b046c9b6126dcecd7`. No tracked file was created, modified or deleted. Everything written lives under `.claude/` (gitignored): the six PNGs in the two variant `figs/` directories, the rebuilt `submission.pdf`/`.aux`/`.log`/`.out` in the two variant directories, and `.claude/{scratch,outputs}/figure-render-pass/`.

## Files produced

- **Installed**: `.claude/papers/neurreps-variants/v2/figs/{fig1_mo_headtohead,fig2_anchor_cure_laws,fig3_gmor_condensate}.png`; `.claude/papers/palm-variant/v5/figs/{fig1_damping_optimum,fig2_two_instruments,figC2_vault_emergent}.png`; both `submission.pdf` rebuilt.
- **Report dir** `.claude/outputs/figure-render-pass/`: the six render scripts (`new_v2f{1,2,3}.py`, `new_v5f{1,2,3}.py`), the exact-canvas helper `size.py`, the verification harness (`tap.py`, `run_orig.py`, `run_new.py`, `compare.py`, `digests/`), the V2-F2 reconstruction + geometric validator (`sf3_reconstruct.py`, `sf3_verify_geom.py`), the isolated original generators (`orig_v2f1.py`, `orig_v2f3.py`), the page-split instrument (`pagesplit.py`), and the six print-size before/after sheets `compare_*.png`.
- **Scratch** `.claude/scratch/figure-render-pass/`: `figs_before/` (the six superseded PNGs), `baseline/` (both pre-pass PDFs), `orig_renders/`, `new_renders/`, the two live-build manifests.

## Open questions / follow-ups / risks

1. **The V2 Fig 2 generator is lost.** Verified by unbounded search: zero matches for `anchored3000_laws` across every `.py` under `.claude/`, and zero `.py` files referencing any of the six variant figure filenames (§4.2). It is now regenerable from `new_v2f2.py` + `sf3_reconstruct.py`. Recommend the Hub treat "every shipped figure has a checked-in generator" as a standing requirement — five of six passed, one did not, and only luck (the npz being banked) made it recoverable.
2. **The figure-vs-report slope discrepancy** (−0.961 on the figure, −0.956 in `outputs/v2-referee-experiments.md`) is a *statistic* difference, not an error, but the V2 caption quotes −0.961 and the source report quotes −0.956 for the same object. If anything downstream cites "the overdamped retention slope", it should say which fit. Not a defect I can fix from here.
3. **No code bug found** in `chlu/`; nothing to hand `experiment-engineer`. The one process hazard worth naming: `fig1_collapse.py` and `regen_figs.py` write **directly into `papers/v5-short/figs/`**, i.e. into the frozen live build. Anyone re-running them without a savefig redirect silently mutates a build that is supposed to be byte-frozen. Recommend those two scripts get an `OUT` environment override.
4. **I did not build the 2×2 / larger-footprint mock-ups** for the three flagged figures — the task scopes footprint changes to the Advisor. If the Advisor wants to see them before deciding, that is a 20-minute follow-up using the same scripts (`size.py` takes the printed width as an argument).

---

## Proposed handover updates (for the Hub)

**§1.6 / experiments — nothing to add: this pass made zero measurements.** All six figures are replots of banked artifacts; five reproduce their banked PNG byte-for-byte.

**§5 provenance — add these, they are new facts:**
- Generator map for the six variant figures, with byte-identical-reproduction status (report §2). `V2 fig2_anchor_cure_laws.png` has **no surviving generator**; it is now regenerable from `.claude/outputs/figure-render-pass/{sf3_reconstruct,new_v2f2}.py`.
- `scratch/v5-revision-1/fig1_collapse.py` and `scratch/v5-revision-1/regen_figs.py` **write into the frozen `papers/v5-short/figs/`**. Flag as a footgun.
- The banked figures' printed scale factors (report §3): **0.232×–0.549×**, i.e. effective tick-label sizes of **2.3–4.9 pt** in both live-adjacent variant builds. This is a *property of the shipped V2/V5 variant PDFs as they stand*, and by inheritance of `papers/v2-short` and `papers/v5-short` wherever those use the same PNGs at reduced widths — worth a look before any of them is sent anywhere.

**§8 / open items — add two owned actions:**
1. **CAPTION EDITS OWED (9 items, report §7)** in the two variant `submission.tex` files. Three are hard defects in the current build (captions naming `I-J/I-R3/I-R1`, the "panels are labelled by the pre-registered prediction" clause, and two deleted colour keys). **Needs a curator/writer task.** I was forbidden to edit captions.
2. **LAYOUT DECISION OWED (report §8)** — V5 Fig 1, V5 Fig 2 and V5 Fig C.2 cannot carry a legend at ≥8 pt inside their printed boxes (0.973–1.140 in tall). Recommended: V5 Fig C.2 → 2×2 at ~2.0 in (it is in Appendix B, so it costs appendix pages, **not** the 4.00 pp main-text budget); V5 Fig 1 → `0.90\linewidth`. **Advisor's call.**

**Also worth recording:** the page-split instrument (`pagesplit.py`) reproduces both BUILD-NOTEs' headline numbers (V5 main 4.00 pp; V2 total 13 pp) and is now a reusable, checked-in-under-`.claude` tool for any future pagination claim.
